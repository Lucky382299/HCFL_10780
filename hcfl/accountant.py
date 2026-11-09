"""Renyi-DP accountant for the Poisson-subsampled Gaussian mechanism (DP-SGD).

RDP of one step at integer order a, sampling rate q, noise multiplier sigma:
    A_a = sum_{k=0}^{a} C(a,k) (1-q)^(a-k) q^k exp((k^2 - k) / (2 sigma^2))
    rdp(a) = log(A_a) / (a - 1)
RDP composes additively over steps. Conversion to (eps, delta)-DP uses
    eps = min_a [ rdp_total(a) + log((a-1)/a) - (log(delta) + log(a)) / (a - 1) ].
"""
from __future__ import annotations

import math

import numpy as np

ORDERS = list(range(2, 65)) + [72, 80, 96, 112, 128, 160, 192, 256]


def _log_add(a, b):
    m = max(a, b)
    if m == -math.inf:
        return m
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def rdp_sampled_gaussian(q: float, sigma: float, order: int) -> float:
    if q <= 0:
        return 0.0
    if sigma == 0:
        return math.inf
    if q >= 1.0:
        return order / (2 * sigma ** 2)
    log_a = -math.inf
    for k in range(order + 1):
        term = (math.lgamma(order + 1) - math.lgamma(k + 1) - math.lgamma(order - k + 1)
                + (order - k) * math.log1p(-q) + k * math.log(q) + (k * k - k) / (2 * sigma ** 2))
        log_a = _log_add(log_a, term)
    return log_a / (order - 1)


def epsilon(q: float, sigma: float, steps: int, delta: float) -> float:
    if steps == 0:
        return 0.0
    best = math.inf
    for a in ORDERS:
        r = steps * rdp_sampled_gaussian(q, sigma, a)
        e = r + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
        best = min(best, e)
    return max(best, 0.0)


from functools import lru_cache


@lru_cache(maxsize=4096)
def calibrate_sigma(q: float, steps: int, target_eps: float, delta: float,
                    lo: float = 0.3, hi: float = 200.0, tol: float = 1e-3) -> float:
    """Smallest noise multiplier whose total epsilon after `steps` steps is <= target_eps."""
    if epsilon(q, hi, steps, delta) > target_eps:
        raise ValueError("target epsilon not reachable; increase hi or reduce steps")
    while hi - lo > tol * hi:
        mid = 0.5 * (lo + hi)
        if epsilon(q, mid, steps, delta) <= target_eps:
            hi = mid
        else:
            lo = mid
    return hi


class RDPAccountant:
    """Tracks actual DP-SGD steps per client; reports the (eps, delta) spent."""

    def __init__(self, delta: float):
        self.delta = delta
        self.steps = {}
        self.params = {}

    def register(self, cid, q, sigma):
        self.params[cid] = (q, sigma)
        self.steps.setdefault(cid, 0)

    def charge(self, cid, steps):
        self.steps[cid] = self.steps.get(cid, 0) + steps

    def report(self):
        eps = [epsilon(*self.params[c], s, self.delta) for c, s in self.steps.items() if s > 0]
        return {"max_eps": float(max(eps)) if eps else 0.0, "mean_eps": float(np.mean(eps)) if eps else 0.0}
