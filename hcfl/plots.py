"""Matplotlib figures for Figs. 3-5 (static, publication-style PNG at 300 dpi).

Series colors follow a fixed categorical order (blue, orange, aqua), and each
method also has its own marker, so identity never depends on color alone.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {"priority_fl": ("#2a78d6", "o"), "static_cp": ("#eb6834", "s"), "proposed": ("#1baf7a", "^")}
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)


def _lines(ax, df, xcol, ycol, labels):
    for m, sub in df.groupby("method", sort=False):
        col, mk = STYLE.get(m, ("#6250d6", "D"))
        sub = sub.sort_values(xcol)
        ax.errorbar(sub[xcol], sub[f"{ycol}_mean"], yerr=sub[f"{ycol}_std"].fillna(0), color=col, marker=mk,
                    markersize=6, linewidth=2, capsize=3, label=labels.get(m, m))


def fig3(sm, path):
    fig, ax1 = plt.subplots(1, 2, figsize=(8, 3.2))
    x = sm["ovr_dp.epsilon"]
    ax1[0].errorbar(x, sm["macro_f1_mean"], yerr=sm["macro_f1_std"].fillna(0), color="#2a78d6", marker="o",
                    linewidth=2, capsize=3)
    _ax(ax1[0], "Privacy budget ε", "Macro-F1 (%)", "(a) Utility")
    ax1[1].errorbar(x, sm["pdi_mean"], yerr=sm["pdi_std"].fillna(0), color="#2a78d6", marker="o",
                    linewidth=2, capsize=3)
    _ax(ax1[1], "Privacy budget ε", "PDI", "(b) Privacy Deviation Index")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig4(sums, path, labels):
    fig, ax = plt.subplots(2, 2, figsize=(8.5, 6.2))
    _lines(ax[0, 0], sums["delay_ms"], "cond_delay_ms", "macro_f1", labels)
    _ax(ax[0, 0], "Synchronization delay (ms)", "Macro-F1 (%)", "(a) Update delay")
    pl = sums["packet_loss"].copy(); pl["cond_packet_loss"] *= 100
    _lines(ax[0, 1], pl, "cond_packet_loss", "macro_f1", labels)
    _ax(ax[0, 1], "Packet loss (%)", "Macro-F1 (%)", "(b) Packet loss")
    d = sums["delay_ms"]
    _lines(ax[1, 0], d[d.method != "priority_fl"], "cond_delay_ms", "sync_rmse", labels)
    _ax(ax[1, 0], "Synchronization delay (ms)", "Sync RMSE", "(c) Physical–virtual divergence")
    _lines(ax[1, 1], sums["disconnect_s"], "cond_disconnect_s", "recovery_s", labels)
    _ax(ax[1, 1], "Disconnection duration (s)", "Recovery time (s)", "(d) Temporary disconnection")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path, dpi=300); plt.close(fig)


def fig5(a, b, path, labels):
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.4))
    _lines(ax[0], a, "ovr_fl.num_clients", "macro_f1", labels)
    _ax(ax[0], "Number of clients", "Macro-F1 (%)", "(a) Client population")
    b = b.copy(); b["ovr_fl.availability"] *= 100
    _lines(ax[1], b, "ovr_fl.availability", "macro_f1", labels)
    _ax(ax[1], "Client availability per round (%)", "Macro-F1 (%)", "(b) Partial participation")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(path, dpi=300); plt.close(fig)
