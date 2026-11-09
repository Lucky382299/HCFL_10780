"""Lightweight holographic sensor counterpart (Eqs. 2-4).

Modes
-----
full           Eqs. (2)-(4): trend prediction, divergence, freshness, confidence,
               confidence-weighted state correction.
no_freshness   ablation: f = 1 in Eq. (3).
no_correction  ablation: h = x when received, h = x_hat when missing (no blending);
               confidence is still computed and used for aggregation.
static         static virtual representation: zero-order hold of the last received
               observation, no prediction, no confidence (c = 1).
none           no counterpart: raw received observation; a lost packet is replaced
               by the training-mean value (fill_value).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODES = ("full", "no_freshness", "no_correction", "static", "none")


@dataclass
class CounterpartTrace:
    h: np.ndarray   # (T, d) counterpart state used as model input
    c: np.ndarray   # (T,)  counterpart confidence
    f: np.ndarray   # (T,)  freshness
    e: np.ndarray   # (T,)  physical-virtual divergence

    def summary(self):
        return {"c_mean": float(self.c.mean()), "f_mean": float(self.f.mean()),
                "e_mean": float(self.e.mean())}


class HolographicCounterpart:
    """Streaming counterpart for one client (used on the Raspberry Pi edge client).

    The batch function `run_counterpart` below is the vectorized equivalent used
    in the simulation experiments; both implement the same equations.
    """

    def __init__(self, d: int, nu=0.3, tau0=2.0, kappa=2.0, eps0=1e-6, mode="full",
                 fill_value=None, lo=None, hi=None):
        assert mode in MODES
        self.d, self.nu, self.tau0, self.kappa, self.eps0, self.mode = d, nu, tau0, kappa, eps0, mode
        self.fill = np.zeros(d) if fill_value is None else np.asarray(fill_value, float)
        self.lo, self.hi = lo, hi
        self.reset()

    def reset(self):
        self.h1 = None   # h^{t-1}
        self.h2 = None   # h^{t-2}
        self.e = 0.0

    def step(self, x, received: bool, age_s: float):
        x = None if x is None else np.asarray(x, float)
        if self.h1 is None:  # initialization h^{-1} = h^0 = x^0
            init = x if (received and x is not None) else self.fill
            self.h1 = init.copy(); self.h2 = init.copy()
        h, c, f, e = _step(self, x, received, age_s)
        self.h2, self.h1 = self.h1, h
        return h, c, f, e

    def state_bytes(self) -> int:
        """Size of the counterpart state exchanged with the edge/server (float32)."""
        return 4 * (2 * self.d + 4)  # h^{t-1}, h^{t-2}, e, c, f, Delta


def _step(cp, x, received, age_s):
    mode = cp.mode
    if mode == "none":
        h = x if received else cp.fill.copy()
        return h, 1.0, 1.0, 0.0
    if mode == "static":
        h = x if received else cp.h1.copy()
        return h, 1.0, 1.0, 0.0
    x_hat = cp.h1 + cp.nu * (cp.h1 - cp.h2)                     # Eq. (2)
    if cp.lo is not None:
        x_hat = np.clip(x_hat, cp.lo, cp.hi)
    if received:                                               # Eq. (3)
        cp.e = float(np.linalg.norm(x - x_hat) / (np.linalg.norm(x) + cp.eps0))
    f = 1.0 if mode == "no_freshness" else float(np.exp(-age_s / cp.tau0))
    c = f * float(np.exp(-cp.kappa * cp.e))
    if not received:
        h = x_hat                                              # Eq. (4), a = 0
    elif mode == "no_correction":
        h = x
    else:
        h = c * x + (1.0 - c) * x_hat                          # Eq. (4), a = 1
    return h, c, f, cp.e


def run_counterpart(mode, x_obs, a, age_s, seg, cp_cfg, fill_value, lo=None, hi=None) -> CounterpartTrace:
    """Run the counterpart over a whole stream, resetting at segment (patient) boundaries."""
    T, d = x_obs.shape
    h = np.empty((T, d)); c = np.empty(T); f = np.empty(T); e = np.empty(T)
    cp = HolographicCounterpart(d, cp_cfg.nu, cp_cfg.tau0, cp_cfg.kappa, cp_cfg.eps0, mode,
                                fill_value, lo if cp_cfg.clamp_state else None,
                                hi if cp_cfg.clamp_state else None)
    prev = None
    for t in range(T):
        if seg[t] != prev:
            cp.reset(); prev = seg[t]
        h[t], c[t], f[t], e[t] = cp.step(x_obs[t], bool(a[t]), float(age_s[t]))
    return CounterpartTrace(h, c, f, e)
