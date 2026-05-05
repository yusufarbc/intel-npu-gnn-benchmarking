from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plot_config import apply_ieee_style, savefig_ieee, IEEE_COLORS, shorten_label

apply_ieee_style()


@dataclass
class FusionGainConfig:
    matrix_csv: Path
    out_dir: Path


def plot_fusion_gain(cfg: FusionGainConfig) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cfg.matrix_csv)

    required = {"model", "b_mean_ms", "o_mean_ms", "speedup"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {cfg.matrix_csv}: {sorted(missing)}")

    # Latency improvement (%): positive means optimized faster.
    df = df.copy()
    df["lat_impr_pct"] = (df["b_mean_ms"] - df["o_mean_ms"]) / df["b_mean_ms"].replace(0, np.nan) * 100.0
    df["fgr"] = df["speedup"].astype(float)

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.scatter(df["fgr"], df["lat_impr_pct"], color=IEEE_COLORS[0], s=26, alpha=0.9)

    for _, r in df.iterrows():
        ax.text(
            float(r["fgr"]),
            float(r["lat_impr_pct"]),
            shorten_label(str(r["model"]), max_len=12),
            fontsize=6,
            ha="left",
            va="bottom",
        )

    ax.axvline(1.0, color="black", lw=0.8, ls="--")
    ax.axhline(0.0, color="black", lw=0.6, ls=":")

    ax.set_xlabel("Fusion Gain Ratio (FGR = baseline / optimized)")
    ax.set_ylabel("Latency improvement (%)")
    ax.set_title("Fusion Gain vs. Performance")

    out = cfg.out_dir / "fusion_gain_vs_latency_improvement"
    savefig_ieee(fig, out)
    plt.close(fig)
    return out.with_suffix(".png")


def parse_args() -> FusionGainConfig:
    p = argparse.ArgumentParser(description="Plot Fusion Gain Ratio vs latency improvement.")
    p.add_argument("--matrix-csv", default="results/scalability_matrix.csv")
    p.add_argument("--out-dir", default="results/figures")
    a = p.parse_args()
    return FusionGainConfig(matrix_csv=Path(a.matrix_csv).resolve(), out_dir=Path(a.out_dir).resolve())


def main() -> None:
    cfg = parse_args()
    fig = plot_fusion_gain(cfg)
    print(f"Figure: {fig}")


if __name__ == "__main__":
    main()
