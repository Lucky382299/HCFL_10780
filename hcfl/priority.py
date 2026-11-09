"""Contextual priority, reliability and network-condition terms (Eqs. 1, 5, 6, 7)."""
from __future__ import annotations

import numpy as np


def initial_priority(C, eta, H, pc):
    """Eq. (1)."""
    s = pc.alpha_c + pc.alpha_q + pc.alpha_h
    assert abs(s - 1.0) < 1e-6 and min(pc.alpha_c, pc.alpha_q, pc.alpha_h) >= 0
    return pc.alpha_c * C + pc.alpha_q * (1.0 - eta) + pc.alpha_h * H


def reliability(eta, pc):
    """Eq. (5)."""
    return float(np.exp(-pc.beta * eta))


def network_condition(B, L, PER, pc):
    """Eq. (6)."""
    s = pc.theta_B + pc.theta_L + pc.theta_P
    assert abs(s - 1.0) < 1e-6
    return pc.theta_B * B + pc.theta_L * (1.0 - L) + pc.theta_P * (1.0 - PER)


def adjusted_priority(p, R, N, pc):
    """Eq. (7)."""
    return p * R * (1.0 + pc.lam * N)


def client_priority_terms(y_local, ch_out, pc):
    """Compute C, eta, H, p, R, N and p~ for one client in one round.

    C   : contextual criticality = share of locally observed high-risk / poor-air
          states in the client's current window (labels are local to the client).
    eta : normalized noise level estimated from the received stream.
    H   : sensor health = 1 - fraction of the window spent disconnected.
    """
    C = float(np.mean(y_local == 1)) if len(y_local) else 0.0
    eta = float(np.clip(ch_out.noise_level / pc.eta_ref, 0.0, 1.0))
    H = 1.0 - float(ch_out.disconnected.mean())
    p = initial_priority(C, eta, H, pc)
    R = reliability(eta, pc)
    N = network_condition(ch_out.B, ch_out.L, ch_out.PER, pc)
    return {"C": C, "eta": eta, "H": H, "p": p, "R": R, "N": N,
            "p_tilde": adjusted_priority(p, R, N, pc)}
