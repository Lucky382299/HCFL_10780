"""Configuration for the holographic-counterpart federated-learning (HC-FL) framework.

All defaults follow the experimental protocol of the manuscript
("Holographic Counterpart-Enabled Privacy-Preserving Federated Learning
for IoT Consumer Electronics"). Values marked "tuned on validation" must be
selected using the validation partition only.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import yaml


@dataclass
class PriorityCfg:
    # Eq. (1): p = a_c*C + a_q*(1-eta) + a_h*H, a_c > a_q >= a_h (tuned on validation)
    alpha_c: float = 0.5
    alpha_q: float = 0.3
    alpha_h: float = 0.2
    # Eq. (5): R = exp(-beta*eta)
    beta: float = 1.0
    # Eq. (6): N = th_B*B + th_L*(1-L) + th_P*(1-PER)
    theta_B: float = 0.3
    theta_L: float = 0.3
    theta_P: float = 0.4
    # Eq. (7): p~ = p*R*(1+lambda*N)
    lam: float = 0.5
    # reference value used to normalize the noise estimate eta into [0,1]
    eta_ref: float = 0.10


@dataclass
class CounterpartCfg:
    # Eq. (2): trend-retention parameter nu in [0,1]
    nu: float = 0.3
    # Eq. (3): tau_0 (s), kappa, eps_0
    tau0: float = 2.0
    kappa: float = 2.0
    eps0: float = 1e-6
    # Eq. (8): damping gamma in (0,1]
    gamma: float = 0.5
    # optional: clamp predicted state to the training feature range during
    # long disconnections (not part of Eqs. 2-4; off by default)
    clamp_state: bool = False
    # sampling interval of the sensing stream (s); hardware uses 1 Hz
    sample_period_s: float = 1.0
    # client-rounds with c below this value are counted as "stale"
    stale_threshold: float = 0.5


@dataclass
class ChannelCfg:
    """Nominal sensing/communication condition used for training and Table II."""
    noise_std: float = 0.02            # Gaussian noise on selected channels (normalized units)
    noisy_channel_frac: float = 0.5    # fraction of feature channels that receive noise
    delay_ms: float = 100.0            # synchronization delay
    packet_loss: float = 0.05          # packet-loss rate
    disconnect_s: float = 0.0          # one temporary disconnection per stream (s)
    max_delay_ms: float = 1000.0       # used to normalize latency for Eq. (6)
    # client heterogeneity: per-client multipliers drawn uniformly from these ranges
    noise_mult_range: List[float] = field(default_factory=lambda: [0.5, 2.0])
    bandwidth_range: List[float] = field(default_factory=lambda: [0.3, 1.0])


@dataclass
class DPCfg:
    enabled: bool = False      # turned on by the privacy experiment (Fig. 3)
    # "dpsgd"  : record-level (eps, delta)-DP via DP-SGD + RDP accounting (corrected method).
    #            epsilon = TOTAL budget per client over the whole training.
    # "laplace": original Eq. (13), update-level clipping + Laplace noise.
    #            epsilon = per-round budget eps_r. Kept for comparison only.
    mechanism: str = "dpsgd"
    epsilon: float = 1.0
    delta: float = 1e-5        # DP-SGD only; should be < 1/n_i
    clip_C: float = 1.0        # DP-SGD: per-example L2 bound; Laplace: L1 bound on the update
    batch_size: int = 256      # DP-SGD expected batch size (Poisson sampling); larger batches raise signal-to-noise
    lr: float = 0.01           # DP-SGD learning rate (Adam)
    local_epochs: int = 1      # DP-SGD local epochs per round (fewer passes -> less noise per step)
    compute_pdi: bool = True   # DP-SGD: train a clipped non-private twin to measure PDI (evaluation only)


@dataclass
class TrainCfg:
    hidden: List[int] = field(default_factory=lambda: [64, 32, 16])
    dropout: float = 0.25
    lr: float = 1e-3
    batch_size: int = 16
    local_epochs: int = 5
    rounds: int = 12
    patience: int = 3
    fedprox_mu: float = 0.01
    scaffold_lr: float = 0.05     # SCAFFOLD is defined for SGD local steps
    backend: str = "auto"         # "auto" | "torch" | "numpy"
    device: str = "cpu"


@dataclass
class FLCfg:
    num_clients: int = 5
    dirichlet_alpha: float = 0.5
    min_client_samples: int = 32
    availability: float = 1.0    # fraction of clients participating per round


@dataclass
class DataCfg:
    dataset: str = "synthetic"   # "mimic3" | "uci_air" | "synthetic"
    path: Optional[str] = None
    split: List[float] = field(default_factory=lambda: [0.7, 0.1, 0.2])
    max_patients: Optional[int] = None
    synthetic_samples: int = 6000


@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    fl: FLCfg = field(default_factory=FLCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    dp: DPCfg = field(default_factory=DPCfg)
    channel: ChannelCfg = field(default_factory=ChannelCfg)
    counterpart: CounterpartCfg = field(default_factory=CounterpartCfg)
    priority: PriorityCfg = field(default_factory=PriorityCfg)
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    out_dir: str = "results"

    def copy(self) -> "Config":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return asdict(self)


def _merge(dc, d: dict):
    for k, v in d.items():
        if not hasattr(dc, k):
            raise KeyError(f"Unknown config key: {k}")
        cur = getattr(dc, k)
        if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(dc, k, v)


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> Config:
    cfg = Config()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        _merge(cfg, overrides)
    return cfg
