"""Datasets, splitting, leakage-free preprocessing and threshold labelling.

MIMIC-III  : six vital signs grouped by patient, ffill+bfill within patient,
             Min-Max scaling, patient-wise 70/10/20 split.
UCI AirQ   : -200 treated as missing, short gaps interpolated, causal rolling mean,
             z-score, chronological 70/10/20 split.
synthetic  : MIMIC-like vital-sign streams so the full pipeline runs without
             credentialed data (for testing only; not for reporting).

Splitting is done first; every fitted quantity (medians, scaler) is estimated on
the training partition only and then applied unchanged to validation/test.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

VITALS = ["hr", "sbp", "dbp", "rr", "spo2", "temp"]

# Abnormal physiological ranges used for the high-risk label (edit to match the
# ranges stated in the manuscript). A sample is high-risk if ANY vital is outside.
MIMIC_RANGES = {"hr": (50, 120), "sbp": (90, 160), "dbp": (50, 100),
                "rr": (10, 24), "spo2": (92, 101), "temp": (35.5, 38.3)}

# Plausibility limits used to drop charting errors before aggregation
MIMIC_VALID = {"hr": (20, 250), "sbp": (40, 260), "dbp": (20, 180),
               "rr": (3, 70), "spo2": (50, 100), "temp": (30, 43)}

# MIMIC-III CHARTEVENTS ITEMIDs (CareVue + MetaVision), as used in MIT-LCP mimic-code
MIMIC_ITEMIDS = {
    "hr": [211, 220045],
    "sbp": [51, 442, 455, 6701, 220179, 220050],
    "dbp": [8368, 8440, 8441, 8555, 220180, 220051],
    "rr": [615, 618, 220210, 224690],
    "spo2": [646, 220277],
    "temp_c": [223762, 676],
    "temp_f": [223761, 678],
}

UCI_FEATURES = ["CO(GT)", "PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)",
                "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"]
UCI_LABEL_COL = "CO(GT)"
UCI_THRESHOLD = 2.5      # mg/m^3 ; poor air quality if CO(GT) > threshold (edit to match manuscript)
UCI_BLOCK = 24           # hourly samples per contiguous block (1 day)


@dataclass
class Split:
    X: np.ndarray          # normalized features (n, d)
    y: np.ndarray          # labels (n,)
    seg: np.ndarray        # stream segment id (patient or day block)


@dataclass
class Dataset:
    name: str
    features: List[str]
    train: Split
    val: Split
    test: Split
    inverse: Callable[[np.ndarray], np.ndarray]     # normalized -> raw units
    label_rule: Callable[[np.ndarray], np.ndarray]  # raw units -> labels
    fill: np.ndarray                                # training mean in normalized units
    lo: np.ndarray
    hi: np.ndarray
    info: Dict = field(default_factory=dict)


# ---------------------------------------------------------------- scalers
class MinMax:
    def fit(self, X):
        self.lo = X.min(0); self.hi = X.max(0); self.rng = np.where(self.hi > self.lo, self.hi - self.lo, 1.0); return self
    def transform(self, X): return (X - self.lo) / self.rng
    def inverse(self, Z): return Z * self.rng + self.lo


class ZScore:
    def fit(self, X):
        self.mu = X.mean(0); self.sd = np.where(X.std(0) > 0, X.std(0), 1.0); return self
    def transform(self, X): return (X - self.mu) / self.sd
    def inverse(self, Z): return Z * self.sd + self.mu


def mimic_label_rule(raw: np.ndarray) -> np.ndarray:
    y = np.zeros(len(raw), dtype=int)
    for j, v in enumerate(VITALS):
        lo, hi = MIMIC_RANGES[v]
        y |= ((raw[:, j] < lo) | (raw[:, j] > hi)).astype(int)
    return y


def _patient_split(subjects: np.ndarray, frac, rng):
    u = np.array(sorted(set(subjects)))
    rng.shuffle(u)
    n1 = int(round(frac[0] * len(u))); n2 = n1 + int(round(frac[1] * len(u)))
    return set(u[:n1]), set(u[n1:n2]), set(u[n2:])


def _finish_vitals(df: pd.DataFrame, frac, seed, name) -> Dataset:
    """df: columns subject_id, t (sortable), and VITALS in raw units."""
    df = df.sort_values(["subject_id", "t"]).reset_index(drop=True)
    df[VITALS] = df.groupby("subject_id")[VITALS].transform(lambda s: s.ffill().bfill())
    rng = np.random.default_rng(seed)
    tr_ids, va_ids, te_ids = _patient_split(df["subject_id"].values, frac, rng)
    parts = {k: df[df["subject_id"].isin(ids)].copy() for k, ids in
             (("train", tr_ids), ("val", va_ids), ("test", te_ids))}
    med = parts["train"][VITALS].median()                  # train-only statistic
    for k in parts:
        parts[k][VITALS] = parts[k][VITALS].fillna(med)
    scaler = MinMax().fit(parts["train"][VITALS].values)   # train-only scaler
    splits = {}
    for k, p in parts.items():
        raw = p[VITALS].values.astype(float)
        splits[k] = Split(scaler.transform(raw), mimic_label_rule(raw), p["subject_id"].values)
    Xtr = splits["train"].X
    return Dataset(name, VITALS, splits["train"], splits["val"], splits["test"],
                   scaler.inverse, mimic_label_rule, Xtr.mean(0), Xtr.min(0), Xtr.max(0),
                   {"pos_rate_train": float(splits["train"].y.mean()),
                    "n": {k: len(v.y) for k, v in splits.items()}})


# ---------------------------------------------------------------- MIMIC-III
def extract_mimic_vitals(mimic_dir: str, out_csv: str, max_patients=None, chunksize=5_000_000):
    """One-off extraction of hourly vitals from CHARTEVENTS (needs PhysioNet access)."""
    files = glob.glob(os.path.join(mimic_dir, "CHARTEVENTS.csv*"))
    if not files:
        raise FileNotFoundError(f"CHARTEVENTS.csv(.gz) not found in {mimic_dir}")
    id2var = {i: k for k, ids in MIMIC_ITEMIDS.items() for i in ids}
    keep = set(id2var)
    rows = []
    for ch in pd.read_csv(files[0], usecols=["SUBJECT_ID", "ITEMID", "CHARTTIME", "VALUENUM"],
                          chunksize=chunksize, low_memory=False):
        ch = ch[ch["ITEMID"].isin(keep) & ch["VALUENUM"].notna()]
        if len(ch):
            rows.append(ch)
    ev = pd.concat(rows, ignore_index=True)
    ev["var"] = ev["ITEMID"].map(id2var)
    f = ev["var"] == "temp_f"
    ev.loc[f, "VALUENUM"] = (ev.loc[f, "VALUENUM"] - 32.0) * 5.0 / 9.0
    ev.loc[ev["var"].str.startswith("temp"), "var"] = "temp"
    for v, (lo, hi) in MIMIC_VALID.items():
        m = ev["var"] == v
        ev = ev[~m | ev["VALUENUM"].between(lo, hi)]
    if max_patients:
        keep_ids = np.sort(ev["SUBJECT_ID"].unique())[:max_patients]
        ev = ev[ev["SUBJECT_ID"].isin(keep_ids)]
    ev["t"] = pd.to_datetime(ev["CHARTTIME"]).dt.floor("h")
    wide = ev.pivot_table(index=["SUBJECT_ID", "t"], columns="var", values="VALUENUM", aggfunc="mean").reset_index()
    wide = wide.rename(columns={"SUBJECT_ID": "subject_id"})
    for v in VITALS:
        if v not in wide:
            wide[v] = np.nan
    wide[["subject_id", "t", *VITALS]].to_csv(out_csv, index=False)
    return out_csv


def load_mimic3(path: str, frac, seed=0, max_patients=None) -> Dataset:
    """path: either a pre-extracted CSV (subject_id,t,hr,sbp,dbp,rr,spo2,temp) or the MIMIC-III directory."""
    if os.path.isdir(path):
        cache = os.path.join(path, "hcfl_vitals_hourly.csv")
        if not os.path.exists(cache):
            extract_mimic_vitals(path, cache, max_patients)
        path = cache
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "charttime" in df and "t" not in df:
        df = df.rename(columns={"charttime": "t"})
    if max_patients:
        df = df[df["subject_id"].isin(np.sort(df["subject_id"].unique())[:max_patients])]
    return _finish_vitals(df[["subject_id", "t", *VITALS]], frac, seed, "mimic3")


# ---------------------------------------------------------------- UCI Air Quality
def load_uci_air(path: str, frac, seed=0) -> Dataset:
    df = pd.read_csv(path, sep=";", decimal=",")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df = df[[c for c in df.columns if c in UCI_FEATURES]].astype(float)
    df = df.replace(-200.0, np.nan)
    n = len(df)
    n1 = int(round(frac[0] * n)); n2 = n1 + int(round(frac[1] * n))
    parts = {"train": df.iloc[:n1].copy(), "val": df.iloc[n1:n2].copy(), "test": df.iloc[n2:].copy()}
    for k in parts:  # per-partition, so no information crosses split boundaries
        parts[k] = parts[k].interpolate(limit=3, limit_direction="forward")
        parts[k] = parts[k].rolling(3, min_periods=1).mean()
    med = parts["train"].median()
    for k in parts:
        parts[k] = parts[k].fillna(med)
    feats = list(parts["train"].columns)
    j = feats.index(UCI_LABEL_COL)

    def rule(raw):
        return (raw[:, j] > UCI_THRESHOLD).astype(int)

    scaler = ZScore().fit(parts["train"].values)
    splits = {}
    for k, p in parts.items():
        raw = p.values
        seg = np.arange(len(raw)) // UCI_BLOCK if k == "train" else np.zeros(len(raw), int)
        splits[k] = Split(scaler.transform(raw), rule(raw), seg)
    Xtr = splits["train"].X
    return Dataset("uci_air", feats, splits["train"], splits["val"], splits["test"],
                   scaler.inverse, rule, Xtr.mean(0), Xtr.min(0), Xtr.max(0),
                   {"pos_rate_train": float(splits["train"].y.mean()),
                    "n": {k: len(v.y) for k, v in splits.items()}})


# ---------------------------------------------------------------- synthetic (testing only)
def make_synthetic(n_samples=6000, frac=(0.7, 0.1, 0.2), seed=0) -> Dataset:
    rng = np.random.default_rng(1000 + seed)
    base = np.array([80, 120, 75, 16, 97, 36.9])
    sd = np.array([12, 15, 10, 3, 1.5, 0.5])
    rows = []
    pid = 0
    while sum(len(r) for r in rows) < n_samples:
        L = int(rng.integers(40, 120))
        mu = base + rng.normal(0, 0.6, 6) * sd
        x = np.empty((L, 6)); x[0] = mu
        shift = np.zeros(6)
        for t in range(1, L):
            if rng.random() < 0.03:     # start/stop an abnormal episode
                shift = np.zeros(6) if shift.any() else rng.normal(0, 2.2, 6) * sd * (rng.random(6) < 0.35)
            x[t] = x[t - 1] + 0.15 * (mu + shift - x[t - 1]) + rng.normal(0, 0.12, 6) * sd
        rows.append(pd.DataFrame(x, columns=VITALS).assign(subject_id=pid, t=np.arange(L)))
        pid += 1
    df = pd.concat(rows, ignore_index=True)
    return _finish_vitals(df, frac, seed, "synthetic")


def load_dataset(dc, seed=0) -> Dataset:
    if dc.dataset == "mimic3":
        return load_mimic3(dc.path, dc.split, seed, dc.max_patients)
    if dc.dataset == "uci_air":
        return load_uci_air(dc.path, dc.split, seed)
    if dc.dataset == "synthetic":
        return make_synthetic(dc.synthetic_samples, dc.split, seed)
    raise ValueError(dc.dataset)
