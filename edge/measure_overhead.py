"""Table I: incremental cost of holographic-counterpart operation on the Raspberry Pi 4B.

Runs each configuration in a fresh process (so peak memory is not shared):
  none   -> priority-adaptive FL without counterpart
  static -> static counterpart
  full   -> full synchronized counterpart
and reports, over --repeats rounds x --seeds seeds:
  inference latency per sample (counterpart step + MLP forward), ms
  peak RSS (psutil), MB
  energy per communication round (INA219), mJ  [N/A without the sensor]
  counterpart-state communication per round, KB

Input stream: a CSV recorded from the NodeMCU nodes (one column per feature) via
--csv, or a synthetic stream if omitted (for testing the script only).

Run on the Pi:  python edge/measure_overhead.py --csv recorded_stream.csv
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def worker(mode, csv, seed, repeats, window):
    import psutil
    from edge.common import EnergyMeter
    from hcfl.config import load_config
    from hcfl.counterpart import HolographicCounterpart
    from hcfl.models import make_model
    from hcfl.privacy import laplace_mechanism

    cfg = load_config()
    rng = np.random.default_rng(seed)
    if csv:
        import pandas as pd
        X = pd.read_csv(csv).select_dtypes("number").values.astype(float)
    else:
        X = np.cumsum(rng.normal(0, 0.05, (window * 4, 15)), axis=0)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = (X[:, 0] > 0.5).astype(int)
    d = X.shape[1]
    cc = cfg.counterpart
    model = make_model(d, 2, cfg.train, seed)
    meter = EnergyMeter()
    rows = []
    for r in range(repeats):
        start = int(rng.integers(0, len(X) - window))
        Xw, yw = X[start:start + window], y[start:start + window]
        a = (rng.random(window) >= cfg.channel.packet_loss)
        meter.start()
        cp = HolographicCounterpart(d, cc.nu, cc.tau0, cc.kappa, cc.eps0,
                                    "none" if mode == "none" else mode)
        H = np.empty_like(Xw); lat = []
        theta = model.get_flat()
        for t in range(window):
            t0 = time.perf_counter()
            if mode == "none":
                h = Xw[t] if a[t] else np.zeros(d)
            else:
                h, *_ = cp.step(Xw[t], bool(a[t]), cfg.channel.delay_ms / 1000 + (0 if a[t] else 1))
            model.predict_proba(h[None, :])
            lat.append(1000 * (time.perf_counter() - t0))
            H[t] = h
        model.train_local(H, yw, cfg.train.local_epochs, cfg.train.batch_size, cfg.train.lr, seed=r)
        u = model.get_flat() - theta
        laplace_mechanism(u, cfg.dp.clip_C, cfg.dp.epsilon, rng)
        energy = meter.stop()
        if mode == "none":
            state_kb = 0.0
        elif mode == "static":
            state_kb = 4 * d / 1024                      # last held state h
        else:
            state_kb = cp.state_bytes() / 1024           # h^{t-1}, h^{t-2}, e, c, f, Delta
        rows.append({"latency_ms": float(np.median(lat)), "energy_mJ": energy, "state_KB": state_kb})
    peak = psutil.Process().memory_info().rss / 2 ** 20
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024   # KB -> MB on Linux
    except Exception:
        pass
    out = {k: float(np.mean([r[k] for r in rows])) if rows[0][k] is not None else None for k in rows[0]}
    out["peak_MB"] = peak
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--window", type=int, default=600, help="samples per round (1 Hz)")
    ap.add_argument("--_worker", default=None)
    ap.add_argument("--_seed", type=int, default=0)
    a = ap.parse_args()
    if a._worker:
        return worker(a._worker, a.csv, a._seed, a.repeats, a.window)
    names = {"none": "Priority-adaptive FL without counterpart", "static": "Static counterpart",
             "full": "Full synchronized counterpart"}
    res = {}
    for mode in names:
        runs = []
        for s in a.seeds:
            cmd = [sys.executable, __file__, "--_worker", mode, "--_seed", str(s),
                   "--repeats", str(a.repeats), "--window", str(a.window)] + (["--csv", a.csv] if a.csv else [])
            runs.append(json.loads(subprocess.check_output(cmd, cwd=ROOT).decode().strip().splitlines()[-1]))
        res[mode] = runs

    def ms(mode, k):
        v = [r[k] for r in res[mode] if r[k] is not None]
        return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0) if v else (np.nan, np.nan)

    keys = [("latency_ms", "Inference latency (ms)"), ("peak_MB", "Peak memory (MB)"),
            ("energy_mJ", "Energy/round (mJ)"), ("state_KB", "State communication (KB/round)")]
    print("| Configuration | " + " | ".join(k[1] for k in keys) + " |\n|" + "---|" * (len(keys) + 1))
    for mode, name in names.items():
        print(f"| {name} | " + " | ".join("N/A" if np.isnan(ms(mode, k)[0]) else
                                          f"{ms(mode, k)[0]:.3f} ± {ms(mode, k)[1]:.3f}" for k, _ in keys) + " |")
    print("| Increment (full - none) | " + " | ".join(
        "N/A" if np.isnan(ms("full", k)[0]) else f"+{ms('full', k)[0] - ms('none', k)[0]:.3f}" for k, _ in keys) + " |")


if __name__ == "__main__":
    main()
