"""Experiment driver: regenerates Tables II-IV and Figs. 3-5 of the manuscript.

Usage (from the repository root):
  python experiments/run.py main        --config configs/mimic3.yaml
  python experiments/run.py privacy     --config configs/mimic3.yaml
  python experiments/run.py sync        --config configs/mimic3.yaml
  python experiments/run.py ablation    --config configs/mimic3.yaml
  python experiments/run.py scalability --config configs/mimic3.yaml
  python experiments/run.py all         --config configs/uci_air.yaml --jobs 4

Every configuration is repeated over cfg.seeds (five by default) and reported
as mean +/- SD. Raw per-seed rows are written to CSV next to the summaries.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hcfl.config import load_config                      # noqa: E402
from hcfl.data import load_dataset                       # noqa: E402
from hcfl.fl import LABELS, run_method, run_threshold_rule  # noqa: E402
from hcfl import plots                                   # noqa: E402

SCALARS = ["acc", "macro_f1", "auroc", "auprc", "train_acc", "gap", "sync_rmse", "freshness",
           "recovery_s", "stale_downweighted", "pdi", "eps_spent_max", "rounds_run", "best_round",
           "agg_ms", "latency_ms", "conv_round", "sigma_mean"]

_DS_CACHE = {}


def _job(args):
    cfg, method, seed, cond, overrides, tag = args
    cfg = cfg.copy()
    for path, val in overrides.items():          # e.g. {"dp.epsilon": 0.6}
        obj = cfg
        *head, last = path.split(".")
        for h in head:
            obj = getattr(obj, h)
        setattr(obj, last, val)
    key = (cfg.data.dataset, cfg.data.path, seed)
    if key not in _DS_CACHE:
        _DS_CACHE[key] = load_dataset(cfg.data, seed)
    ds = _DS_CACHE[key]
    t0 = time.time()
    if method == "threshold_rule":
        r = run_threshold_rule(ds, cfg, seed, cond)
    else:
        r = run_method(ds, cfg, method, seed, cond)
    row = {"tag": tag, "method": method, "seed": seed, **{f"cond_{k}": v for k, v in cond.items()},
           **{f"ovr_{k}": v for k, v in overrides.items()},
           **{k: r.get(k, np.nan) for k in SCALARS}, "wall_s": time.time() - t0}
    print(f"done {tag:12s} {method:14s} seed={seed} cond={cond} ovr={overrides} "
          f"F1={row['macro_f1']:.2f} ({row['wall_s']:.1f}s)", flush=True)
    return row


def run_jobs(jobs, n_workers):
    if n_workers <= 1:
        return [_job(j) for j in jobs]
    with ProcessPoolExecutor(n_workers) as ex:
        return list(ex.map(_job, jobs))


def summarize(df, by, cols):
    g = df.groupby(by, sort=False)
    out = g[cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def fmt(m, s, nd=2):
    if pd.isna(m):
        return "N/A"
    return f"{m:.{nd}f} ± {0.0 if pd.isna(s) else s:.{nd}f}"


def write_md(path, title, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"### {title}\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")
    print(open(path, encoding="utf-8").read())


# --------------------------------------------------------------------- experiments
def exp_main(cfg, out, jobs_n):
    methods = ["fedavg", "fedprox", "scaffold", "priority_fl", "static_cp", "threshold_rule", "proposed"]
    jobs = [(cfg, m, s, {}, {}, "main") for m in methods for s in cfg.seeds]
    df = pd.DataFrame(run_jobs(jobs, jobs_n))
    df.to_csv(os.path.join(out, "main_raw.csv"), index=False)
    sm = summarize(df, ["method"], ["acc", "macro_f1", "auroc", "auprc", "sync_rmse", "latency_ms", "gap", "train_acc"])
    rows = []
    for _, r in sm.iterrows():
        cp = r["method"] in ("static_cp", "proposed")
        rows.append([LABELS[r["method"]], fmt(r.acc_mean, r.acc_std), fmt(r.macro_f1_mean, r.macro_f1_std),
                     fmt(r.auroc_mean, r.auroc_std, 3), fmt(r.auprc_mean, r.auprc_std, 3),
                     fmt(r.sync_rmse_mean, r.sync_rmse_std, 3) if cp else "N/A",
                     fmt(r.latency_ms_mean, r.latency_ms_std, 3)])
    write_md(os.path.join(out, "table2_comparison.md"), "Table II. Same-testbed comparison",
             ["Method", "Test accuracy (%)", "Macro-F1 (%)", "AUROC", "AUPRC", "Sync RMSE",
              "Latency (ms, this machine)"], rows)
    p = sm[sm.method == "proposed"].iloc[0]
    write_md(os.path.join(out, "table4_independent_test.md"), "Table IV. Independent test performance (proposed)",
             ["Dataset", "Test accuracy (%)", "Macro-F1 (%)", "AUROC", "AUPRC", "Train–test gap (%)"],
             [[cfg.data.dataset, fmt(p.acc_mean, p.acc_std), fmt(p.macro_f1_mean, p.macro_f1_std),
               fmt(p.auroc_mean, p.auroc_std, 3), fmt(p.auprc_mean, p.auprc_std, 3), f"{p.gap_mean:.2f}"]])


EPS_LIST = [0.6, 0.8, 1.0, 1.2]


def exp_privacy(cfg, out, jobs_n):
    eps_list = EPS_LIST
    jobs = [(cfg, "proposed", s, {}, {"dp.enabled": True, "dp.epsilon": e}, "privacy")
            for e in eps_list for s in cfg.seeds]
    jobs += [(cfg, "proposed", s, {}, {"dp.enabled": False}, "privacy_nodp") for s in cfg.seeds]
    df = pd.DataFrame(run_jobs(jobs, jobs_n))
    df.to_csv(os.path.join(out, "privacy_raw.csv"), index=False)
    nodp = df[df.tag == "privacy_nodp"]
    df = df[df.tag == "privacy"]
    sm = summarize(df, ["ovr_dp.epsilon"], ["macro_f1", "pdi", "acc", "auroc", "eps_spent_max", "sigma_mean"])
    sm.to_csv(os.path.join(out, "fig3_privacy_summary.csv"), index=False)
    unit = "total per client, δ = %g" % cfg.dp.delta if cfg.dp.mechanism == "dpsgd" else "per round"
    rows = [[str(r["ovr_dp.epsilon"]), fmt(r.macro_f1_mean, r.macro_f1_std), fmt(r.pdi_mean, r.pdi_std, 3),
             fmt(r.auroc_mean, r.auroc_std, 3), fmt(r.eps_spent_max_mean, r.eps_spent_max_std, 3),
             fmt(r.sigma_mean_mean, r.sigma_mean_std, 2)] for _, r in sm.iterrows()]
    rows.append(["no DP", fmt(nodp.macro_f1.mean(), nodp.macro_f1.std()), "0", fmt(nodp.auroc.mean(), nodp.auroc.std(), 3),
                 "∞", "0"])
    write_md(os.path.join(out, "fig3_privacy.md"),
             f"Fig. 3 data. Privacy–utility sensitivity ({cfg.dp.mechanism}, ε {unit})",
             ["Target ε", "Macro-F1 (%)", "PDI", "AUROC", "ε actually spent", "Noise multiplier σ"], rows)
    plots.fig3(sm, os.path.join(out, "fig3_privacy.png"))


def exp_sync(cfg, out, jobs_n):
    methods = ["priority_fl", "static_cp", "proposed"]
    sweeps = {"delay_ms": [0, 100, 250, 500], "packet_loss": [0.0, 0.05, 0.10, 0.20],
              "disconnect_s": [5, 10, 20, 30]}
    jobs = []
    for var, levels in sweeps.items():
        for lv in levels:
            for m in methods:
                for s in cfg.seeds:
                    jobs.append((cfg, m, s, {var: lv}, {}, f"sync_{var}"))
    df = pd.DataFrame(run_jobs(jobs, jobs_n))
    df.to_csv(os.path.join(out, "sync_raw.csv"), index=False)
    sums = {}
    for var in sweeps:
        sub = df[df.tag == f"sync_{var}"]
        sums[var] = summarize(sub, ["method", f"cond_{var}"], ["macro_f1", "sync_rmse", "recovery_s", "freshness"])
        sums[var].to_csv(os.path.join(out, f"fig4_{var}_summary.csv"), index=False)
    plots.fig4(sums, os.path.join(out, "fig4_sync_robustness.png"), LABELS)


def exp_ablation(cfg, out, jobs_n, cond=None):
    cond = cond or {"delay_ms": 250, "packet_loss": 0.10, "disconnect_s": 10}
    methods = ["priority_fl", "static_cp", "no_freshness", "no_correction", "proposed"]
    jobs = [(cfg, m, s, cond, {}, "ablation") for m in methods for s in cfg.seeds]
    df = pd.DataFrame(run_jobs(jobs, jobs_n))
    df.to_csv(os.path.join(out, "ablation_raw.csv"), index=False)
    sm = summarize(df, ["method"], ["macro_f1", "sync_rmse", "recovery_s", "latency_ms", "stale_downweighted"])
    rows = [[LABELS[r["method"]], fmt(r.macro_f1_mean, r.macro_f1_std),
             fmt(r.sync_rmse_mean, r.sync_rmse_std, 3) if r["method"] != "priority_fl" else "N/A",
             fmt(r.recovery_s_mean, r.recovery_s_std, 1) if r["method"] != "priority_fl" else "N/A",
             fmt(r.stale_downweighted_mean, r.stale_downweighted_std, 3)] for _, r in sm.iterrows()]
    write_md(os.path.join(out, "table3_ablation.md"),
             f"Table III. Ablation of holographic-counterpart components (condition {json.dumps(cond)})",
             ["Configuration", "Macro-F1 (%)", "Sync RMSE", "Recovery time (s)", "Stale down-weighted"], rows)
    print("Latency and energy columns of Table III come from edge/measure_overhead.py on the Raspberry Pi.")


def exp_scalability(cfg, out, jobs_n):
    methods = ["priority_fl", "proposed"]
    jobs = [(cfg, m, s, {}, {"fl.num_clients": k}, "scale_clients")
            for k in (5, 10, 20, 50) for m in methods for s in cfg.seeds]
    jobs += [(cfg, m, s, {}, {"fl.availability": a}, "scale_avail")
             for a in (1.0, 0.75, 0.5) for m in methods for s in cfg.seeds]
    df = pd.DataFrame(run_jobs(jobs, jobs_n))
    df.to_csv(os.path.join(out, "scalability_raw.csv"), index=False)
    a = summarize(df[df.tag == "scale_clients"], ["method", "ovr_fl.num_clients"], ["macro_f1", "agg_ms"])
    b = summarize(df[df.tag == "scale_avail"], ["method", "ovr_fl.availability"], ["macro_f1", "conv_round"])
    a.to_csv(os.path.join(out, "fig5a_clients_summary.csv"), index=False)
    b.to_csv(os.path.join(out, "fig5b_availability_summary.csv"), index=False)
    plots.fig5(a, b, os.path.join(out, "fig5_scalability.png"), LABELS)


EXPS = {"main": exp_main, "privacy": exp_privacy, "sync": exp_sync,
        "ablation": exp_ablation, "scalability": exp_scalability}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=[*EXPS, "all"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--eps", type=float, nargs="*", default=None, help="privacy budgets for Fig. 3")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.eps:
        EPS_LIST[:] = a.eps
    if a.seeds is not None:
        cfg.seeds = a.seeds
    out = a.out or os.path.join(cfg.out_dir, cfg.data.dataset)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config_used.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    for name in (EXPS if a.experiment == "all" else [a.experiment]):
        print(f"\n===== {name} =====")
        EXPS[name](cfg, out, a.jobs)


if __name__ == "__main__":
    main()
