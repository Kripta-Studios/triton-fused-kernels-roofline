"""Turn results/*.csv + results/peak.json into the README figures (CPU only).

    python benchmarks/plot_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# Fixed implementation -> color/marker mapping (identity never depends on rank).
STYLE = {
    "triton_fused": ("#2a78d6", "o"), "triton_fa2": ("#2a78d6", "o"),
    "eager": ("#eb6834", "s"),
    "torch_compile": ("#1baf7a", "^"),
    "torch_native": ("#eda100", "D"), "sdpa_flash": ("#eda100", "D"),
    "sdpa_efficient": ("#e87ba4", "v"),
    "sdpa_cudnn": ("#008300", "P"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
})


def roofline(peak: dict, rms: pd.DataFrame, att: pd.DataFrame) -> None:
    s = peak["summary"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ai = np.logspace(-1, 4, 400)
    for bw, tf, ls, label in (
        (s["datasheet_dram_gbps"], s["datasheet_tensor_fp16_fp32acc_tflops"], "--", "datasheet"),
        (s["measured_dram_gbps"], s["measured_tensor_tflops"], "-", "measured (copy / cuBLAS)"),
    ):
        ax.plot(ai, np.minimum(tf, bw * ai / 1e3), ls, color=INK, lw=1.4, label=f"roof: {label}")
        ridge = tf * 1e3 / bw
        ax.annotate(f"ridge {ridge:.0f} FLOP/B", (ridge, tf), textcoords="offset points",
                    xytext=(6, -14 if ls == "-" else 6), color=MUTED, fontsize=8)
    big = rms[(rms.tokens >= 4096) & (rms.impl == "triton_fused")]
    for residual, mk in ((False, "o"), (True, "s")):
        d = big[big.residual == residual]
        ax.scatter(d.ai, d.tflops, s=46, marker=mk, color="#2a78d6", edgecolor="#fcfcfb", lw=1.5,
                   zorder=3, label=f"Triton RMSNorm{' + residual' if residual else ''} (M>=4096)")
    for dd, mk in ((64, "^"), (128, "D")):
        d = att[(att.impl == "triton_fa2") & (att.D == dd)]
        ax.scatter(d.ai, d.tflops, s=46, marker=mk, color="#eb6834", edgecolor="#fcfcfb", lw=1.5,
                   zorder=3, label=f"Triton causal FA2, D={dd} (N=512..16k)")
        for _, r in d.iloc[[0, -1]].iterrows():
            ax.annotate(f"N={r.N}", (r.ai, r.tflops), textcoords="offset points", xytext=(6, -10),
                        color=MUTED, fontsize=8)
    ax.set(xscale="log", yscale="log", xlabel="arithmetic intensity (FLOP / byte of DRAM traffic)",
           ylabel="achieved TFLOP/s", ylim=(1e-2, 120), xlim=(0.1, 1e4))
    ax.set_title("Roofline, RTX 5070 Ti Laptop (bf16)", loc="left", color=INK)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "roofline.png", dpi=130)
    plt.close(fig)


def rmsnorm_plot(rms: pd.DataFrame, peak: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, residual in zip(axes, (False, True)):
        d = rms[(rms.residual == residual) & (rms.hidden == 4096)]
        for impl, g in d.groupby("impl", sort=False):
            c, m = STYLE[impl]
            ax.plot(g.tokens, g.gbps, marker=m, color=c, lw=2, ms=7, label=impl)
        ax.axhline(peak["summary"]["measured_dram_gbps"], color=MUTED, ls=":", lw=1.2)
        ax.annotate("measured copy peak", (1, peak["summary"]["measured_dram_gbps"]),
                    textcoords="offset points", xytext=(2, 4), color=MUTED, fontsize=8)
        ax.set(xscale="log", xlabel="tokens (rows), hidden = 4096")
        ax.set_title("RMSNorm + residual add" if residual else "RMSNorm", loc="left", color=INK)
    axes[0].set_ylabel("effective bandwidth (GB/s)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "rmsnorm_bandwidth.png", dpi=130)
    plt.close(fig)


def attention_plot(att: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, dd in zip(axes, (64, 128)):
        d = att[att.D == dd]
        for impl, g in d.groupby("impl", sort=False):
            c, m = STYLE[impl]
            ax.plot(g.N, g.tflops, marker=m, color=c, lw=2, ms=7, label=impl)
        ax.set(xscale="log", xlabel="sequence length N (B=1, H=16, causal)")
        ax.set_xticks(sorted(d.N.unique()), [str(n) for n in sorted(d.N.unique())])
        ax.minorticks_off()
        ax.set_title(f"Causal attention forward, head_dim={dd}", loc="left", color=INK)
    axes[0].set_ylabel("achieved TFLOP/s (bf16)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "attention_tflops.png", dpi=130)
    plt.close(fig)


def main() -> None:
    peak = json.loads((RES / "peak.json").read_text())
    rms = pd.read_csv(RES / "rmsnorm.csv")
    att = pd.read_csv(RES / "attention.csv")
    roofline(peak, rms, att)
    rmsnorm_plot(rms, peak)
    attention_plot(att)
    print("wrote", [p.name for p in RES.glob("*.png")])


if __name__ == "__main__":
    main()
