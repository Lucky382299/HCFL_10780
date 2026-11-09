"""Counterpart-confidence aggregation (Eqs. 8 and 14)."""
from __future__ import annotations

import numpy as np


def preliminary_weights(n, p_tilde, c):
    """omega_bar_i = n_i p~_i c_i / sum_j n_j p~_j c_j  (Eq. 8, left)."""
    s = np.asarray(n, float) * np.asarray(p_tilde, float) * np.asarray(c, float)
    tot = s.sum()
    if tot <= 0:
        return np.full(len(s), 1.0 / len(s))
    return s / tot


def damped_weights(ids, w_bar, prev: dict, gamma: float):
    """omega_i^t = (1-gamma) omega_i^{t-1} + gamma omega_bar_i^t  (Eq. 8, right).

    Under partial participation the previous weights of the clients in A_t do
    not sum to one, so they are renormalized over A_t first. A client with no
    previous weight starts from its preliminary weight. This keeps
    omega >= 0 and sum_{A_t} omega = 1 in every round.
    """
    w_prev = np.array([prev.get(i, wb) for i, wb in zip(ids, w_bar)], float)
    w_prev = w_prev / w_prev.sum()
    w = (1.0 - gamma) * w_prev + gamma * np.asarray(w_bar)
    w = w / w.sum()
    for i, wi in zip(ids, w):
        prev[i] = float(wi)
    return w


def apply_update(theta, updates, weights):
    """Eq. (14): theta^{t+1} = theta^t + sum_i omega_i u~_i."""
    out = theta.copy()
    for u, w in zip(updates, weights):
        out += w * u
    return out
