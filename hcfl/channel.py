"""Sensing and communication channel simulator.

Turns a clean physical stream x_true (T x d) into what the edge gateway
actually receives: delayed, noisy observations with packet loss and
temporary disconnections. It also reports the synchronization age Delta
(Eq. 3) and the network descriptors B, L, PER used in Eq. (6).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChannelOutput:
    x_obs: np.ndarray        # (T, d) received observation (only valid where a == 1)
    a: np.ndarray            # (T,)  1 if the physical packet was received
    age_s: np.ndarray        # (T,)  synchronization age Delta (s)
    disconnected: np.ndarray # (T,)  1 inside a disconnection window
    reconnect_idx: np.ndarray  # indices where a disconnection window ends
    B: float                 # normalized bandwidth in [0,1]
    L: float                 # normalized latency in [0,1]
    PER: float               # empirical packet-error rate in [0,1]
    noise_level: float       # estimated noise eta in [0,1] (before eta_ref scaling)


@dataclass
class ClientProfile:
    """Static hardware/network heterogeneity of a client."""
    noise_mult: float
    bandwidth: float
    noisy_channels: np.ndarray  # indices of channels with additive noise


def make_profiles(num_clients: int, d: int, ch_cfg, rng: np.random.Generator):
    profiles = []
    n_noisy = max(1, int(round(ch_cfg.noisy_channel_frac * d)))
    for _ in range(num_clients):
        profiles.append(ClientProfile(
            noise_mult=float(rng.uniform(*ch_cfg.noise_mult_range)),
            bandwidth=float(rng.uniform(*ch_cfg.bandwidth_range)),
            noisy_channels=np.sort(rng.choice(d, n_noisy, replace=False)),
        ))
    return profiles


def _segment_starts(seg: np.ndarray) -> np.ndarray:
    starts = np.zeros(len(seg), dtype=bool)
    starts[0] = True
    starts[1:] = seg[1:] != seg[:-1]
    return starts


def _delayed(x: np.ndarray, seg: np.ndarray, delay_steps: float) -> np.ndarray:
    """x(t - delay) with linear interpolation, never crossing a segment start."""
    if delay_steps <= 0:
        return x.copy()
    T = len(x)
    k = int(np.floor(delay_steps))
    r = delay_steps - k
    starts = np.where(_segment_starts(seg))[0]
    seg_start = np.zeros(T, dtype=int)
    seg_start[starts] = starts
    seg_start = np.maximum.accumulate(seg_start)
    idx = np.arange(T)
    i0 = np.maximum(idx - k, seg_start)
    i1 = np.maximum(idx - k - 1, seg_start)
    return (1.0 - r) * x[i0] + r * x[i1]


def noise_estimate(x_obs: np.ndarray, a: np.ndarray) -> float:
    """Robust noise estimate from second differences of the received stream.

    For white noise of std s, the second difference has std s*sqrt(6).
    The median absolute deviation makes the estimate robust to true
    physical changes.
    """
    xr = x_obs[a.astype(bool)]
    if len(xr) < 4:
        return 0.0
    d2 = np.diff(xr, n=2, axis=0)
    mad = np.median(np.abs(d2 - np.median(d2, axis=0)), axis=0) * 1.4826
    return float(np.mean(mad) / np.sqrt(6.0))


def simulate(x_true: np.ndarray, seg: np.ndarray, profile: ClientProfile, ch_cfg,
             rng: np.random.Generator, sample_period_s: float = 1.0,
             delay_ms: float | None = None, packet_loss: float | None = None,
             disconnect_s: float | None = None, noise_std: float | None = None,
             n_disconnects: int = 1) -> ChannelOutput:
    delay_ms = ch_cfg.delay_ms if delay_ms is None else delay_ms
    packet_loss = ch_cfg.packet_loss if packet_loss is None else packet_loss
    disconnect_s = ch_cfg.disconnect_s if disconnect_s is None else disconnect_s
    noise_std = ch_cfg.noise_std if noise_std is None else noise_std

    T, d = x_true.shape
    delay_s = delay_ms / 1000.0
    x_obs = _delayed(x_true, seg, delay_s / sample_period_s)

    sigma = noise_std * profile.noise_mult
    if sigma > 0:
        x_obs[:, profile.noisy_channels] += rng.normal(0.0, sigma, size=(T, len(profile.noisy_channels)))

    a = (rng.random(T) >= packet_loss).astype(np.int8)

    disconnected = np.zeros(T, dtype=np.int8)
    reconnect_idx = np.array([], dtype=int)
    n_disc = int(round(disconnect_s / sample_period_s))
    if n_disc > 0:
        # place n_disconnects non-overlapping windows, separated by >= 3*n_disc steps
        slot = n_disc * 4
        n_win = max(1, min(n_disconnects, (T - 20) // slot)) if T > slot + 20 else 0
        starts_w = []
        for w in range(n_win):
            lo = 10 + w * ((T - 20) // n_win)
            hi = lo + (T - 20) // n_win - slot
            if hi <= lo:
                continue
            s = int(rng.integers(lo, hi))
            disconnected[s:s + n_disc] = 1
            a[s:s + n_disc] = 0
            starts_w.append(s + n_disc)
        reconnect_idx = np.array(starts_w, dtype=int)

    # synchronization age: delay + time elapsed since the last received packet
    starts = _segment_starts(seg)
    age = np.empty(T)
    since = 0.0
    for t in range(T):
        if starts[t]:
            since = 0.0
        if a[t]:
            since = 0.0
        else:
            since += sample_period_s
        age[t] = delay_s + since

    per = 1.0 - float(a.mean())
    L = min(1.0, delay_ms / max(ch_cfg.max_delay_ms, 1e-9))
    return ChannelOutput(x_obs=x_obs, a=a, age_s=age, disconnected=disconnected,
                         reconnect_idx=reconnect_idx, B=profile.bandwidth, L=L, PER=per,
                         noise_level=noise_estimate(x_obs, a))
