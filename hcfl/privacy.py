"""Differential privacy on client updates (Eqs. 12, 13) and the Privacy Deviation Index (Eq. 15)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def clip_l1(u: np.ndarray, C: float) -> np.ndarray:
    """u_bar = u / max(1, ||u||_1 / C)."""
    return u / max(1.0, float(np.abs(u).sum()) / C)


def laplace_mechanism(u: np.ndarray, C: float, eps: float, rng: np.random.Generator):
    """Eq. (13): clip to L1 bound C, add element-wise Lap(0, 2C/eps).

    Two clipped updates differ by at most 2C in L1 norm, so scale 2C/eps gives
    pure eps-DP (delta = 0) for one round.
    Returns (u_tilde, u_bar).
    """
    u_bar = clip_l1(u, C)
    xi = rng.laplace(0.0, 2.0 * C / eps, size=u.shape)
    return u_bar + xi, u_bar


def pdi(u_tildes, u_bars, eps0=1e-12) -> float:
    """Eq. (15): sum ||u~ - u_bar||_2 / (sum ||u_bar||_2 + eps0)."""
    num = sum(float(np.linalg.norm(ut - ub)) for ut, ub in zip(u_tildes, u_bars))
    den = sum(float(np.linalg.norm(ub)) for ub in u_bars) + eps0
    return num / den


def expected_pdi_lower_bound(num_params: int, C: float, eps: float) -> float:
    """Analytic reference value for Eq. (15) under Eq. (13).

    E||xi||_2 ~= sqrt(num_params) * sqrt(2) * (2C/eps) while ||u_bar||_2 <= ||u_bar||_1 <= C.
    So PDI >= approximately 2*sqrt(2*num_params)/eps.
    """
    return 2.0 * np.sqrt(2.0 * num_params) / eps


class PrivacyAccountant:
    """Basic sequential composition: total eps of a client = sum of its per-round budgets."""

    def __init__(self):
        self.spent = defaultdict(float)

    def charge(self, client_id: int, eps: float):
        self.spent[client_id] += eps

    def report(self):
        if not self.spent:
            return {"max_eps": 0.0, "mean_eps": 0.0}
        v = np.array(list(self.spent.values()))
        return {"max_eps": float(v.max()), "mean_eps": float(v.mean())}
