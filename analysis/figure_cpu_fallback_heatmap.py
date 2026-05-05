from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.ort_profile_utils import iter_operator_events, load_events
from analysis.plot_config import apply_ieee_style, savefig_ieee, shorten_label

apply_ieee_style()


@dataclass
class HeatmapConfig:
    results_dir: Path
    out_dir: Path
    mode: str = "optimized"  # baseline|optimized
    top_ops: int = 25


def _find_latest_trace(model_dir: Path, mode: str) -> Path | None:
    # Prefer highest run index for stability.
    traces = sorted(model_dir.glob(f"run_*/{mode}_profiling.json"))
    if not traces:
        return None
    return traces[-1]


def _op_cpu_flag(events: List[dict]) -> Dict[str, float]:
    # Returns operator -> cpu_fraction_time (0..1), based on duration sums.
    sums: Dict[str, Dict[str, float]] = {}
    for op_name, dur_us, provider, _cat in iter_operator_events(events):
        op = str(op_name)
        prov = str(provider).lower()
        d = float(dur_us)
        if op not in sums:
            sums[op] = {"cpu": 0.0, "total": 0.0}
        sums[op]["total"] += d
        if "cpu" in prov:
            sums[op]["cpu"] += d

    out: Dict[str, float] = {}
    for op, v in sums.items():
        tot = v["total"]
        out[op] = (v["cpu"] / tot) if tot > 0 else 0.0
    return out


def build_heatmap(cfg: HeatmapConfig) -> Tuple[pd.DataFrame, Path]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # model -> op -> cpu_fraction
    model_maps: Dict[str, Dict[str, float]] = {}
    op_total_time: Dict[str, float] = {}

    for model_dir in sorted([p for p in cfg.results_dir.iterdir() if p.is_dir()]):
        trace = _find_latest_trace(model_dir, cfg.mode)
        if trace is None:
            continue

        try:
            events = load_events(trace)
        except Exception:
            continue

        # Also accumulate total time per op to select top ops.
        per_op = {}
        per_op_total = {}
        for op_name, dur_us, provider, _cat in iter_operator_events(events):
            op = str(op_name)
            per_op_total[op] = per_op_total.get(op, 0.0) + float(dur_us)
        cpu_frac = _op_cpu_flag(events)

        model = model_dir.name
        model_maps[model] = cpu_frac
        for op, t in per_op_total.items():
            op_total_time[op] = op_total_time.get(op, 0.0) + float(t)

    if not model_maps:
        raise FileNotFoundError(f"No profiling traces found under: {cfg.results_dir}")

    top_ops = [op for op, _ in sorted(op_total_time.items(), key=lambda kv: kv[1], reverse=True)[: cfg.top_ops]]

    models = sorted(model_maps.keys())
    mat = np.full((len(top_ops), len(models)), np.nan, dtype=np.float32)

    for j, m in enumerate(models):
        mp = model_maps[m]
        for i, op in enumerate(top_ops):
            if op in mp:
                mat[i, j] = float(mp[op])

    df = pd.DataFrame(mat, index=top_ops, columns=models)

    # Plot: discrete-ish colormap where 0=NPU-only, 1=CPU-only.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(df.values, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0, cmap="viridis")

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([shorten_label(m, max_len=12) for m in models], rotation=40, ha="right")
    ax.set_yticks(np.arange(len(top_ops)))
    ax.set_yticklabels([shorten_label(op, max_len=24) for op in top_ops])

    ax.set_xlabel("Model")
    ax.set_ylabel("Operator")
    ax.set_title("CPU fallback heatmap (fraction of op time on CPU)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("CPU time fraction")

    out = cfg.out_dir / f"cpu_fallback_heatmap_{cfg.mode}"
    savefig_ieee(fig, out)
    plt.close(fig)

    df.to_csv(cfg.out_dir / f"cpu_fallback_heatmap_{cfg.mode}.csv")
    return df, out.with_suffix(".png")


def parse_args() -> HeatmapConfig:
    p = argparse.ArgumentParser(description="Generate CPU fallback heatmap from ORT profiling traces.")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="results/figures")
    p.add_argument("--mode", default="optimized", choices=["baseline", "optimized"])
    p.add_argument("--top-ops", type=int, default=25)
    a = p.parse_args()

    return HeatmapConfig(
        results_dir=Path(a.results_dir).resolve(),
        out_dir=Path(a.out_dir).resolve(),
        mode=str(a.mode),
        top_ops=int(a.top_ops),
    )


def main() -> None:
    cfg = parse_args()
    _df, fig = build_heatmap(cfg)
    print(f"Figure: {fig}")


if __name__ == "__main__":
    main()
