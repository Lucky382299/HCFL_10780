"""Predictive and synchronization metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def classification(y, proba=None, pred=None):
    if pred is None:
        pred = proba.argmax(1)
    out = {"acc": 100 * accuracy_score(y, pred),
           "macro_f1": 100 * f1_score(y, pred, average="macro", zero_division=0)}
    if proba is not None and len(np.unique(y)) > 1:
        out["auroc"] = roc_auc_score(y, proba[:, 1])
        out["auprc"] = average_precision_score(y, proba[:, 1])
    else:
        out["auroc"] = np.nan; out["auprc"] = np.nan
    return out


def sync_rmse(h, x_true):
    """Physical-virtual synchronization error: RMSE between model input state and the true physical state."""
    return float(np.sqrt(np.mean((h - x_true) ** 2)))


def recovery_time(h, x_true, reconnect_idx, disc_steps, period_s=1.0, k_consec=3, factor=2.0, floor=0.02):
    """Seconds after reconnection until the relative state error stays below a
    threshold for k_consec consecutive samples.

    Threshold = max(factor * median pre-disconnection error, floor).
    Returns the mean over all disconnection windows (nan if none).
    """
    if len(reconnect_idx) == 0:
        return np.nan
    err = np.linalg.norm(h - x_true, axis=1) / (np.linalg.norm(x_true, axis=1) + 1e-9)
    T = len(err)
    times = []
    for r in reconnect_idx:
        s = r - disc_steps
        pre = err[max(0, s - 30):s]
        thr = max(factor * (np.median(pre) if len(pre) else 0.0), floor)
        k = 0
        while r + k + k_consec <= T and not np.all(err[r + k:r + k + k_consec] <= thr):
            k += 1
        times.append(k * period_s)
    return float(np.mean(times))


def mean_sd(values):
    v = np.asarray([x for x in values if x is not None], float)
    v = v[~np.isnan(v)] if v.size else v
    if v.size == 0:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0
