"""Federated training loop implementing Algorithm 1 and all compared methods.

Methods (identical partitions, seeds, channel realizations and test samples):
  fedavg        FedAvg, raw input, data-size weights
  fedprox       FedProx (proximal term mu), raw input, data-size weights
  scaffold      SCAFFOLD (control variates, SGD local steps), raw input
  priority_fl   priority-adaptive FL without counterpart: weights ~ n * p~ (Eqs. 1,5-8 with c = 1)
  static_cp     static counterpart (zero-order hold) + priority weights
  no_freshness  ablation of the proposed model with f = 1
  no_correction ablation without confidence-weighted state correction
  proposed      full holographic-confidence FL (Eqs. 1-15, Algorithm 1)
"""
from __future__ import annotations

import time
import zlib

import numpy as np

from . import aggregation as agg
from .channel import ClientProfile, make_profiles, simulate
from .counterpart import run_counterpart
from .metrics import classification, recovery_time, sync_rmse
from .models import make_model
from .partition import dirichlet_blocks
from .priority import client_priority_terms
from .accountant import RDPAccountant, calibrate_sigma
from .privacy import PrivacyAccountant, laplace_mechanism, pdi

METHODS = {
    "fedavg":        {"input": "none",          "weights": "data",       "local": "adam"},
    "fedprox":       {"input": "none",          "weights": "data",       "local": "prox"},
    "scaffold":      {"input": "none",          "weights": "data",       "local": "scaffold"},
    "priority_fl":   {"input": "none",          "weights": "priority",   "local": "adam"},
    "static_cp":     {"input": "static",        "weights": "priority",   "local": "adam"},
    "no_freshness":  {"input": "no_freshness",  "weights": "confidence", "local": "adam"},
    "no_correction": {"input": "no_correction", "weights": "confidence", "local": "adam"},
    "proposed":      {"input": "full",          "weights": "confidence", "local": "adam"},
}

LABELS = {"fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD",
          "priority_fl": "Priority-adaptive FL without counterpart",
          "static_cp": "Static counterpart", "no_freshness": "Without freshness f",
          "no_correction": "Without state correction", "proposed": "Proposed holographic-confidence FL",
          "threshold_rule": "Threshold-rule classifier"}


def _rng(*keys):
    """Deterministic RNG keyed on (purpose, seed, ...) so every method sees identical randomness."""
    return np.random.default_rng(zlib.crc32(repr(keys).encode()))


def _stream(split, cfg, profile, mode, ds, rng, cond, n_disconnects=1):
    ch = simulate(split.X, split.seg, profile, cfg.channel, rng, cfg.counterpart.sample_period_s,
                  n_disconnects=n_disconnects, **cond)
    tr = run_counterpart(mode, ch.x_obs, ch.a, ch.age_s, split.seg, cfg.counterpart, ds.fill, ds.lo, ds.hi)
    return ch, tr


class _Sub:
    def __init__(self, X, y, seg):
        self.X, self.y, self.seg = X, y, seg


def run_method(ds, cfg, method, seed, cond=None, test_cond=None, n_test_disconnects=5, log=None):
    """Train one method with one seed and evaluate on the independent test partition."""
    spec = METHODS[method]
    cond = dict(cond or {})
    test_cond = dict(test_cond if test_cond is not None else cond)
    tc, pc, cc = cfg.train, cfg.priority, cfg.counterpart
    d = ds.train.X.shape[1]
    n_classes = int(max(ds.train.y.max(), ds.test.y.max())) + 1

    # ---- clients (same for every method given the seed)
    parts = dirichlet_blocks(ds.train.y, ds.train.seg, cfg.fl.num_clients, cfg.fl.dirichlet_alpha,
                             _rng("part", seed, cfg.fl.num_clients), cfg.fl.min_client_samples)
    clients = [_Sub(ds.train.X[ix], ds.train.y[ix], ds.train.seg[ix]) for ix in parts]
    clients = [c for c in clients if len(c.y) > 0]
    K = len(clients)
    profiles = make_profiles(K, d, cfg.channel, _rng("prof", seed, K))
    ref_profile = ClientProfile(1.0, 1.0, profiles[0].noisy_channels)  # used for val/test streams

    model = make_model(d, n_classes, tc, seed)
    theta = model.get_flat()
    P = model.num_params
    prev_w = {}
    dpc = cfg.dp
    use_dpsgd = dpc.enabled and dpc.mechanism == "dpsgd"
    acct = RDPAccountant(dpc.delta) if use_dpsgd else PrivacyAccountant()
    sigmas = {}
    if use_dpsgd:
        # calibrate each client's noise so that even participation in every round
        # (no early stopping) stays within the total budget (eps, delta)
        for i, cl in enumerate(clients):
            n = len(cl.y); q = min(1.0, dpc.batch_size / n)
            max_steps = tc.rounds * dpc.local_epochs * int(np.ceil(n / dpc.batch_size))
            sigmas[i] = calibrate_sigma(round(q, 6), max_steps, dpc.epsilon, dpc.delta)
            acct.register(i, round(q, 6), sigmas[i])
    c_glob = np.zeros(P); c_loc = [np.zeros(P) for _ in range(K)]   # SCAFFOLD

    # validation stream is fixed per seed (identical for all methods)
    _, val_tr = _stream(ds.val, cfg, ref_profile, spec["input"], ds, _rng("val", seed), cond)

    best = (np.inf, theta.copy(), 0)
    hist = []
    stale_cnt = tot_cnt = 0
    agg_times = []
    for t in range(1, tc.rounds + 1):
        avail_rng = _rng("avail", seed, t, cfg.fl.availability, K)
        A = [i for i in range(K) if avail_rng.random() < cfg.fl.availability] or [int(avail_rng.integers(K))]
        updates, ubars, n_i, ptil, conf, ids, dcs = [], [], [], [], [], [], []
        for i in A:
            cl = clients[i]
            ch, tr = _stream(cl, cfg, profiles[i], spec["input"], ds, _rng("train", seed, t, i), cond)
            terms = client_priority_terms(cl.y, ch, pc)
            model.set_flat(theta)
            hook, opt, lr = None, "adam", tc.lr
            if spec["local"] == "prox":
                hook = lambda w, th=theta: tc.fedprox_mu * (w - th)
            elif spec["local"] == "scaffold":
                hook = lambda w, ci=c_loc[i]: c_glob - ci
                opt, lr = "sgd", tc.scaffold_lr
            s_loc = zlib.crc32(repr((seed, t, i)).encode())
            if use_dpsgd:
                s_noise = zlib.crc32(repr(("dpnoise", seed, t, i, method)).encode())
                lr = dpc.lr if opt == "adam" else lr
                steps = model.train_dpsgd(tr.h, cl.y, dpc.local_epochs, dpc.batch_size, lr, dpc.clip_C,
                                          sigmas[i], s_loc, s_noise, opt, hook)
                u = model.get_flat() - theta
                acct.charge(i, steps)
                if dpc.compute_pdi:      # clipped non-private twin with identical batches and dropout
                    model.set_flat(theta)
                    model.train_dpsgd(tr.h, cl.y, dpc.local_epochs, dpc.batch_size, lr, dpc.clip_C,
                                      0.0, s_loc, s_noise, opt, hook)
                    u_clip = model.get_flat() - theta
                else:
                    u_clip = u
            else:
                steps = model.train_local(tr.h, cl.y, tc.local_epochs, tc.batch_size, lr, opt, hook, seed=s_loc)
                u = model.get_flat() - theta
            if spec["local"] == "scaffold":
                c_new = c_loc[i] - c_glob - u / (steps * lr)
                dcs.append(c_new - c_loc[i]); c_loc[i] = c_new
            if use_dpsgd:
                u_t, u_b = u, u_clip
            elif cfg.dp.enabled:
                u_t, u_b = laplace_mechanism(u, cfg.dp.clip_C, cfg.dp.epsilon, _rng("dp", seed, t, i, method))
                acct.charge(i, cfg.dp.epsilon)
            else:
                u_t = u_b = u
            updates.append(u_t); ubars.append(u_b)
            n_i.append(len(cl.y)); ptil.append(terms["p_tilde"]); conf.append(float(tr.c.mean())); ids.append(i)

        t0 = time.perf_counter()
        if spec["weights"] == "data":
            w = np.asarray(n_i, float) / np.sum(n_i)
        else:
            c_used = conf if spec["weights"] == "confidence" else np.ones(len(ids))
            w_bar = agg.preliminary_weights(n_i, ptil, c_used)
            w = agg.damped_weights(ids, w_bar, prev_w, cc.gamma)
        theta = agg.apply_update(theta, updates, w)
        if dcs:
            c_glob = c_glob + (len(dcs) / K) * np.mean(dcs, axis=0)
        agg_times.append(1000 * (time.perf_counter() - t0))

        # stale client-rounds that were down-weighted below their data share
        share = np.asarray(n_i, float) / np.sum(n_i)
        for cj, wj, sj in zip(conf, w, share):
            tot_cnt += 1
            stale_cnt += int(cj < cc.stale_threshold and wj < sj)

        model.set_flat(theta)
        vloss = model.loss(val_tr.h, ds.val.y)
        vf1 = classification(ds.val.y, model.predict_proba(val_tr.h))["macro_f1"]
        hist.append({"round": t, "val_loss": vloss, "val_f1": vf1, "pdi": pdi(updates, ubars),
                     "participants": len(A), "weights": dict(zip(ids, map(float, w)))})
        if log:
            log(f"  [{method} s{seed}] round {t:2d}  val_loss={vloss:.4f}  val_F1={vf1:.2f}  PDI={hist[-1]['pdi']:.3g}")
        if vloss < best[0]:
            best = (vloss, theta.copy(), t)
        elif t - best[2] >= tc.patience:
            break

    # ---- independent test evaluation with the selected model
    model.set_flat(best[1])
    te_ch, te_tr = _stream(ds.test, cfg, ref_profile, spec["input"], ds, _rng("test", seed), test_cond,
                           n_disconnects=n_test_disconnects)
    proba = model.predict_proba(te_tr.h)
    res = classification(ds.test.y, proba)
    # train-test gap: pooled training data through the same pipeline
    _, tr_tr = _stream(ds.train, cfg, ref_profile, spec["input"], ds, _rng("trainEval", seed), cond)
    res["train_acc"] = classification(ds.train.y, model.predict_proba(tr_tr.h))["acc"]
    res["gap"] = res["train_acc"] - res["acc"]
    res["sync_rmse"] = sync_rmse(te_tr.h, ds.test.X)
    res["freshness"] = float(te_tr.f.mean())
    disc_steps = int(round(test_cond.get("disconnect_s", cfg.channel.disconnect_s) / cc.sample_period_s))
    res["recovery_s"] = recovery_time(te_tr.h, ds.test.X, te_ch.reconnect_idx, disc_steps, cc.sample_period_s)
    res["stale_downweighted"] = stale_cnt / max(tot_cnt, 1)
    res["pdi"] = float(np.mean([h["pdi"] for h in hist])) if cfg.dp.enabled else 0.0
    res["eps_spent_max"] = acct.report()["max_eps"]
    res["sigma_mean"] = float(np.mean(list(sigmas.values()))) if sigmas else 0.0
    res["rounds_run"] = len(hist); res["best_round"] = best[2]
    res["agg_ms"] = float(np.mean(agg_times))
    # per-sample inference latency on this machine (Raspberry Pi values come from edge/)
    x1 = te_tr.h[:1]
    t0 = time.perf_counter()
    for _ in range(200):
        model.predict_proba(x1)
    res["latency_ms"] = 1000 * (time.perf_counter() - t0) / 200
    # convergence: first round whose val F1 is within 0.5 points of the best val F1
    f1s = [h["val_f1"] for h in hist]
    res["conv_round"] = next(h["round"] for h in hist if h["val_f1"] >= max(f1s) - 0.5)
    res["num_params"] = P
    res["history"] = hist
    return res


def run_threshold_rule(ds, cfg, seed, cond=None):
    """Applies the labelling rule to the received (noisy, delayed) test stream."""
    cond = dict(cond or {})
    d = ds.test.X.shape[1]
    prof = make_profiles(1, d, cfg.channel, _rng("prof", seed, 1))[0]
    prof = ClientProfile(1.0, 1.0, prof.noisy_channels)
    _, tr = _stream(ds.test, cfg, prof, "static", ds, _rng("test", seed), cond)
    raw = ds.inverse(tr.h)
    t0 = time.perf_counter()
    for _ in range(200):
        ds.label_rule(raw[:1])
    lat = 1000 * (time.perf_counter() - t0) / 200
    pred = ds.label_rule(raw)
    res = classification(ds.test.y, pred=pred)
    res["latency_ms"] = lat
    return res
