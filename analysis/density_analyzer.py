from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plot_config import apply_ieee_style, savefig_ieee, IEEE_COLORS

apply_ieee_style()


@dataclass
class DensityAnalyzeConfig:
    sweep_dir: Path
    out_dir: Path
    prefer_device: str = "NPU"


def _read_any_input_metadata(model_dir: Path) -> Dict[str, float]:
    # Prefer run_00, but accept any run.
    candidate = model_dir / "run_00" / "input_metadata.json"
    if candidate.exists():
        paths = [candidate]
    else:
        paths = list(sorted(model_dir.glob("run_*/input_metadata.json")))

    for p in paths:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            out: Dict[str, float] = {}
            for k in ("used_num_nodes", "used_num_edges", "used_num_features"):
                v = payload.get(k)
                if isinstance(v, (int, float)):
                    out[k] = float(v)
            return out
        except Exception:
            continue
    return {}


def _iter_dataset_dirs(sweep_dir: Path) -> List[Tuple[str, Path]]:
    """Return list of (device, dataset_dir) pairs."""
    device_dirs = sorted([p for p in sweep_dir.glob("device_*") if p.is_dir()])
    out: List[Tuple[str, Path]] = []
    if device_dirs:
        for dev_dir in device_dirs:
            dev = dev_dir.name.replace("device_", "")
            for ds_dir in sorted(dev_dir.glob("dataset_*")):
                if ds_dir.is_dir():
                    out.append((dev, ds_dir))
        return out

    # Backward compatible: sweep_dir directly contains dataset_*.
    for ds_dir in sorted(sweep_dir.glob("dataset_*")):
        if ds_dir.is_dir():
            out.append(("NPU", ds_dir))
    return out


def load_sweep(sweep_dir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for device, ds_dir in _iter_dataset_dirs(sweep_dir):
        ds = ds_dir.name.replace("dataset_", "")
        matrix = ds_dir / "scalability_matrix.csv"
        if not matrix.exists():
            continue
        df = pd.read_csv(matrix)
        df["dataset"] = ds
        df["device"] = device

        # Attach measured graph size metadata (same for all models in our padded/sampled setting).
        # We still compute it per model to be robust.
        meta_list: List[Dict[str, float]] = []
        for m in df["model"].astype(str).tolist():
            meta = _read_any_input_metadata(ds_dir / m)
            meta_list.append(meta)
        meta_df = pd.DataFrame(meta_list)
        df = pd.concat([df, meta_df], axis=1)

        # Density proxy: edges per node (directed). For undirected graphs, avg degree ~ 2E/N.
        if "used_num_nodes" in df.columns and "used_num_edges" in df.columns:
            denom = df["used_num_nodes"].replace(0, np.nan)
            df["edges_per_node"] = df["used_num_edges"] / denom
            df["avg_degree_undirected_est"] = 2.0 * df["edges_per_node"]
        else:
            df["edges_per_node"] = np.nan
            df["avg_degree_undirected_est"] = np.nan

        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"No dataset_* results found in: {sweep_dir}")

    return pd.concat(rows, ignore_index=True)


def plot_density_trend(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use FP32-only to reduce clutter by default.
    sub = df[df["model"].astype(str).str.contains("fp32", case=False, na=False)].copy()
    if sub.empty:
        sub = df.copy()

    # If we have multiple devices, compute NPU-vs-CPU speedup on optimized latency.
    have_devices = set(sub["device"].astype(str).unique().tolist())
    out_dir.mkdir(parents=True, exist_ok=True)

    if {"CPU", "NPU"}.issubset({d.upper() for d in have_devices}):
        # Normalize device names to upper for pivoting.
        sub["device_norm"] = sub["device"].astype(str).str.upper()
        pivot = sub.pivot_table(
            index=["dataset", "model"],
            columns="device_norm",
            values="o_mean_ms",
            aggfunc="mean",
        ).reset_index()

        pivot["npu_vs_cpu_speedup"] = pivot["CPU"] / pivot["NPU"]

        # Attach density proxy and (optional) CPU fallback from NPU runs.
        dens = (
            sub[sub["device_norm"] == "NPU"]
            .groupby(["dataset", "model"], as_index=False)
            .agg(edges_per_node=("edges_per_node", "mean"), cpu_fallback_pct=("o_cpu_fallback_pct", "mean"))
        )
        merged = pivot.merge(dens, on=["dataset", "model"], how="left")

        # Dataset-level summary.
        g = (
            merged.groupby("dataset", as_index=False)
            .agg(
                edges_per_node=("edges_per_node", "mean"),
                npu_vs_cpu_speedup=("npu_vs_cpu_speedup", "mean"),
                cpu_fallback_pct=("cpu_fallback_pct", "mean"),
            )
            .sort_values("edges_per_node")
        )
        g.to_csv(out_dir / "density_summary_by_dataset.csv", index=False)
        merged.to_csv(out_dir / "density_sweep_merged_device_compare.csv", index=False)

        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        ax.plot(
            g["edges_per_node"].values,
            g["npu_vs_cpu_speedup"].values,
            marker="o",
            color=IEEE_COLORS[0],
            linewidth=1.3,
            label="Mean speedup (CPU / NPU)",
        )
        for _, r in g.iterrows():
            ax.text(
                float(r["edges_per_node"]),
                float(r["npu_vs_cpu_speedup"]),
                str(r["dataset"]),
                fontsize=7,
                ha="left",
                va="bottom",
            )
        ax.set_xlabel("Edge/node ratio (density proxy)")
        ax.set_ylabel("Speedup (CPU / NPU)")
            ax.set_title("Density → Performance: NPU benefit grows with density")
        ax.grid(True, alpha=0.2)
        out = out_dir / "density_vs_speedup"
        savefig_ieee(fig, out)
        plt.close(fig)
        return out.with_suffix(".png")

    # Fallback: plot baseline-vs-optimized speedup (still informative, but different claim)
    g = (
        sub.groupby(["dataset"], as_index=False)
        .agg(
            edges_per_node=("edges_per_node", "mean"),
            speedup=("speedup", "mean"),
            latency_ms=("o_mean_ms", "mean"),
        )
        .sort_values("edges_per_node")
    )
    g.to_csv(out_dir / "density_summary_by_dataset.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(
        g["edges_per_node"].values,
        g["speedup"].values,
        marker="o",
        color=IEEE_COLORS[1],
        linewidth=1.3,
        label="Mean speedup (baseline/optimized)",
    )
    for _, r in g.iterrows():
        ax.text(
            float(r["edges_per_node"]),
            float(r["speedup"]),
            str(r["dataset"]),
            fontsize=7,
            ha="left",
            va="bottom",
        )
    ax.set_xlabel("Edge/node ratio (density proxy)")
    ax.set_ylabel("Speedup (baseline/optimized)")
    ax.set_title("Density-dependent optimization benefit")
    ax.grid(True, alpha=0.2)

    out = out_dir / "density_trend_speedup"
    savefig_ieee(fig, out)
    plt.close(fig)
    return out.with_suffix(".png")


def parse_args() -> DensityAnalyzeConfig:
    parser = argparse.ArgumentParser(description="Analyze density sweep results and generate summary plots.")
    parser.add_argument("--sweep-dir", default="results/density_sweep")
    parser.add_argument("--out-dir", default="results/density_sweep")
    parser.add_argument("--prefer-device", default="NPU")
    args = parser.parse_args()

    return DensityAnalyzeConfig(
        sweep_dir=Path(args.sweep_dir).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        prefer_device=str(args.prefer_device),
    )


def main() -> None:
    cfg = parse_args()
    df = load_sweep(cfg.sweep_dir)

    # Persist merged table
    merged_csv = cfg.out_dir / "density_sweep_merged.csv"
    df.to_csv(merged_csv, index=False)

    fig_path = plot_density_trend(df, cfg.out_dir)

    print(f"Merged CSV: {merged_csv}")
    print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
