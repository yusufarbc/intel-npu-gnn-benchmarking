"""Regenerate the crowded single-column paper figures at their final print size.

The paper places these figures at one IEEE column (3.5 in).  Generating a wide
10 in canvas and scaling it down in LaTeX also scales 8 pt labels to roughly
3 pt.  This script instead draws at the final physical width and exports PNG,
SVG, and PDF variants with 8 pt tick and legend text.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "figures"
PAPER = ROOT / "paper" / "figures"

DEVICES = ["NPU", "GPU", "CPU"]
DEVICE_LABELS = {"NPU": "NPU", "GPU": "iGPU", "CPU": "CPU"}
COLORS = {"NPU": "#0072B2", "GPU": "#E69F00", "CPU": "#009E73"}

MODEL_ORDER = [
    "APPNP", "GAT", "GATv2", "GCN", "GIN", "GraphSAGE",
    "GraphTransformer", "MPNN", "SGC", "bert-tiny", "efficientnet-b0",
    "mobilenetv2", "resnet50", "vit-tiny",
]

SHORT_LABELS = {
    "GraphSAGE": "G-SAGE",
    "GraphTransformer": "G-Trans",
    "bert-tiny": "BERT-T",
    "efficientnet-b0": "EffNet",
    "mobilenetv2": "MobileNet",
    "resnet50": "ResNet50",
    "vit-tiny": "ViT-T",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8,
    "axes.linewidth": 0.7,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "intel-npu-gnn-hpec-2026",
})


def short_label(name: str) -> str:
    return SHORT_LABELS.get(name, name)


def ordered_models(values: pd.Series) -> list[str]:
    present = set(values.dropna().astype(str))
    known = [model for model in MODEL_ORDER if model in present]
    return known + sorted(present.difference(known), key=str.casefold)


def finish_axis(ax: mpl.axes.Axes) -> None:
    ax.grid(axis="y", linestyle="--", color="#b8b8b8")
    ax.set_axisbelow(True)
    ax.tick_params(width=0.7, length=2.5, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def export(fig: mpl.figure.Figure, stem: str) -> None:
    """Write the same final-size artwork to results and paper directories."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "generate_ieee_paper_figures.py"}
    outputs = []
    for suffix, kwargs in (
        (".png", {"dpi": 600}),
        (".svg", {"metadata": metadata}),
        (".pdf", {"metadata": metadata}),
    ):
        output = RESULTS / f"{stem}{suffix}"
        fig.savefig(output, facecolor="white", **kwargs)
        if suffix == ".svg":
            # Matplotlib emits trailing spaces inside path-data lines. Keep the
            # generated artifact clean for git diff checks without changing it.
            svg = "\n".join(line.rstrip() for line in output.read_text(encoding="utf-8").splitlines()) + "\n"
            output.write_text(svg, encoding="utf-8", newline="\n")
        outputs.append(output)
    for output in outputs:
        shutil.copy2(output, PAPER / output.name)
    plt.close(fig)


def figure_1_latency() -> None:
    data = pd.read_csv(RESULTS / "master_results.csv")
    data = data[data["precision"].str.lower() == "fp32"].copy()
    models = ordered_models(data["model"])
    mean = data.groupby(["model", "device"])["mean_ms"].mean().unstack()
    std = data.groupby(["model", "device"])["std_ms"].mean().unstack()

    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.36, top=0.88)
    x = np.arange(len(models))
    width = 0.24
    for index, device in enumerate(DEVICES):
        values = mean.reindex(models).get(device, pd.Series(0, index=models)).fillna(0)
        errors = std.reindex(models).get(device, pd.Series(0, index=models)).fillna(0)
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            yerr=errors,
            capsize=1.2,
            linewidth=0,
            color=COLORS[device],
            label=DEVICE_LABELS[device],
        )
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x, [short_label(model) for model in models], rotation=62,
                  ha="right", rotation_mode="anchor")
    ax.legend(loc="upper left", ncol=3, frameon=False, handlelength=1.0,
              columnspacing=0.8, borderaxespad=0.2)
    finish_axis(ax)
    export(fig, "fig1_latency_comparison")


def figure_3_operators() -> None:
    data = pd.read_csv(RESULTS / "operator_mix.csv")
    models = ordered_models(data["model"])
    data = data.set_index("model").reindex(models)
    categories = ["SpMM/MatMul", "MLP", "Activation", "Attention", "Memory/Shape", "Other"]
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]

    fig, ax = plt.subplots(figsize=(3.5, 3.05))
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.34, top=0.79)
    x = np.arange(len(models))
    bottom = np.zeros(len(models))
    for category, color in zip(categories, colors):
        values = data[category].fillna(0).to_numpy()
        ax.bar(x, values, bottom=bottom, width=0.78, color=color,
               edgecolor="white", linewidth=0.25, label=category)
        bottom += values
    ax.set_ylabel("Share of ONNX nodes (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x, [short_label(model) for model in models], rotation=62,
                  ha="right", rotation_mode="anchor")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              frameon=False, handlelength=0.9, handletextpad=0.35,
              columnspacing=0.65, borderaxespad=0)
    finish_axis(ax)
    export(fig, "fig3_operator_breakdown")


def figure_4_optimization() -> None:
    frames = [pd.read_csv(path) for path in sorted((ROOT / "results").glob("*/scalability_matrix.csv"))]
    data = pd.concat(frames, ignore_index=True)
    data = data[data["speedup"].notna()].copy()
    data["base_model"] = data["model"].map(
        lambda value: re.sub(r"_(?:fp32|int8)$", "", str(value), flags=re.IGNORECASE)
    )
    models = ordered_models(data["base_model"])
    mean = data.groupby(["base_model", "device"])["speedup"].mean().unstack()

    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.36, top=0.88)
    x = np.arange(len(models))
    width = 0.24
    for index, device in enumerate(DEVICES):
        values = mean.reindex(models).get(device, pd.Series(np.nan, index=models))
        ax.bar(x + (index - 1) * width, values, width, linewidth=0,
               color=COLORS[device], label=DEVICE_LABELS[device])
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Optimization speedup (x)")
    ax.set_xticks(x, [short_label(model) for model in models], rotation=62,
                  ha="right", rotation_mode="anchor")
    ax.legend(loc="upper left", ncol=3, frameon=False, handlelength=1.0,
              columnspacing=0.8, borderaxespad=0.2)
    finish_axis(ax)
    export(fig, "fig5a_opt_speedup")


def main() -> None:
    figure_1_latency()
    figure_3_operators()
    figure_4_optimization()
    print("Generated IEEE-size PNG/SVG/PDF versions for paper Figures 1, 3, and 4.")


if __name__ == "__main__":
    main()
