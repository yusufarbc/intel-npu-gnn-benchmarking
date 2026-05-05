from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import onnx
import pandas as pd

from analysis.plot_config import apply_ieee_style, savefig_ieee, IEEE_COLORS, shorten_label

apply_ieee_style()


@dataclass
class OpBreakdownConfig:
    models_dir: Path
    out_dir: Path
    fp32_only: bool = True


# Categories requested by user.
CATEGORIES = ["SpMM/MatMul", "MLP", "Activation", "Memory/Shape", "Other"]


def _categorize_node(op_type: str) -> str:
    op = str(op_type)
    op_low = op.lower()

    # SpMM proxy: GEMM/MatMul often dominates message passing and MLPs.
    # We'll separate MLP vs SpMM by presence of Sparse-ish patterns is not reliable in ONNX,
    # so we treat MatMul/Gemm as SpMM/MatMul and classify Conv as MLP/Other.
    if op in {"MatMul", "Gemm"}:
        return "SpMM/MatMul"

    # MLP proxy: linear layers sometimes appear as Gemm/MatMul + Add; Add handled in Other.
    if op in {"Conv"}:
        return "MLP"

    if op in {"Relu", "Sigmoid", "Tanh", "Gelu", "LeakyRelu", "Elu", "Softmax", "LogSoftmax"}:
        return "Activation"

    if op in {
        "Gather",
        "GatherElements",
        "ScatterElements",
        "ScatterND",
        "Slice",
        "Concat",
        "Split",
        "Transpose",
        "Reshape",
        "Squeeze",
        "Unsqueeze",
        "Expand",
        "Tile",
        "Shape",
        "Cast",
        "Where",
        "ConstantOfShape",
        "ReduceSum",
        "ReduceMean",
        "ReduceMax",
        "ReduceMin",
        "NonZero",
        "TopK",
        "ArgMax",
        "ArgMin",
    }:
        return "Memory/Shape"

    return "Other"


def compute_operator_mix(model_path: Path) -> Dict[str, float]:
    model = onnx.load(str(model_path))
    counts: Dict[str, int] = {c: 0 for c in CATEGORIES}

    for node in model.graph.node:
        cat = _categorize_node(node.op_type)
        counts[cat] = counts.get(cat, 0) + 1

    total = float(sum(counts.values()) or 1)
    return {k: float(v) / total * 100.0 for k, v in counts.items()}


def plot_operator_breakdown(cfg: OpBreakdownConfig) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(cfg.models_dir.glob("*.onnx"))
    if cfg.fp32_only:
        models = [m for m in models if "fp32" in m.stem.lower()]
    if not models:
        raise FileNotFoundError(f"No ONNX models found in: {cfg.models_dir}")

    rows: List[Dict[str, float]] = []
    labels: List[str] = []

    for m in models:
        mix = compute_operator_mix(m)
        rows.append(mix)
        labels.append(m.stem)

    df = pd.DataFrame(rows, index=labels)
    df.to_csv(cfg.out_dir / "operator_mix_by_model_pct.csv")

    # Stacked bar
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(df.index))
    bottom = np.zeros(len(df.index), dtype=np.float32)

    colors = {
        "SpMM/MatMul": IEEE_COLORS[0],
        "MLP": IEEE_COLORS[1],
        "Activation": IEEE_COLORS[2],
        "Memory/Shape": IEEE_COLORS[6],
        "Other": IEEE_COLORS[7],
    }

    for cat in CATEGORIES:
        vals = df[cat].fillna(0.0).values
        ax.bar(x, vals, bottom=bottom, label=cat, color=colors.get(cat, None), edgecolor="k", linewidth=0.25)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([shorten_label(s, max_len=10) for s in df.index], rotation=40, ha="right")
    ax.set_ylabel("Share of ONNX nodes (%)")
    ax.set_title("Operator breakdown (structural mix; ONNX node share)")
    ax.set_ylim(0, 100)
    ax.legend(ncol=2, framealpha=0.9, fontsize=7)

    out = cfg.out_dir / "operator_breakdown_stacked"
    savefig_ieee(fig, out)
    plt.close(fig)
    return out.with_suffix(".png")


def parse_args() -> OpBreakdownConfig:
    p = argparse.ArgumentParser(description="Stacked bar: operator category breakdown per model (ONNX structural mix).")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--out-dir", default="results/figures")
    p.add_argument("--all-precisions", action="store_true", help="Include int8 models too")
    a = p.parse_args()

    return OpBreakdownConfig(
        models_dir=Path(a.models_dir).resolve(),
        out_dir=Path(a.out_dir).resolve(),
        fp32_only=not bool(a.all_precisions),
    )


def main() -> None:
    cfg = parse_args()
    fig = plot_operator_breakdown(cfg)
    print(f"Figure: {fig}")


if __name__ == "__main__":
    main()
