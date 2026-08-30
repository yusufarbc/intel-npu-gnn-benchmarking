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


def export(fig: mpl.figure.Figure, stem: str, *, tight: bool = False) -> None:
    """Write the same final-size artwork to results and paper directories."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "generate_ieee_paper_figures.py"}
    outputs = []
    crop = {"bbox_inches": "tight", "pad_inches": 0.02} if tight else {}
    for suffix, kwargs in (
        (".png", {"dpi": 600}),
        (".svg", {"metadata": metadata}),
        (".pdf", {"metadata": metadata}),
    ):
        output = RESULTS / f"{stem}{suffix}"
        fig.savefig(output, facecolor="white", **crop, **kwargs)
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


def figure_2_int8_heatmap() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "results").glob("*/precision_gain.csv")):
        frame = pd.read_csv(path)
        if frame.empty or "speedup" not in frame:
            continue
        for device, group in frame.groupby("device"):
            speedup = pd.to_numeric(group["speedup"], errors="coerce").mean()
            if pd.notna(speedup):
                rows.append({"model": path.parent.name, "device": device, "speedup": speedup})

    data = pd.DataFrame(rows)
    models = ordered_models(data["model"])
    devices = ["CPU", "GPU", "NPU"]
    pivot = data.pivot(index="model", columns="device", values="speedup").reindex(
        index=models, columns=devices
    )
    values = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    cmap = mpl.colormaps["RdYlGn"].copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    fig.subplots_adjust(left=0.25, right=0.88, bottom=0.13, top=0.98)
    image = ax.imshow(masked, cmap=cmap, vmin=0.8, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(devices)), [DEVICE_LABELS[device] for device in devices])
    ax.set_yticks(range(len(models)), [short_label(model) for model in models])
    ax.set_xlabel("Device")
    ax.set_ylabel("Model")
    ax.tick_params(width=0.7, length=0, pad=2)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                color = "white" if value < 1.05 or value > 2.0 else "black"
                ax.text(column, row, f"{value:.1f}", ha="center", va="center",
                        fontsize=8, color=color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.055, pad=0.04)
    colorbar.ax.tick_params(labelsize=8, width=0.7, length=2.5)
    colorbar.outline.set_linewidth(0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    export(fig, "fig2_int8_speedup_heatmap")


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
    # The legend and rotated labels make the useful artwork substantially
    # shorter than the nominal canvas. Crop the vector page to the artwork so
    # LaTeX does not reserve a large blank block below the x-axis labels.
    export(fig, "fig3_operator_breakdown", tight=True)


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


def figure_5_roofline() -> None:
    frames = [pd.read_csv(path) for path in sorted((ROOT / "results").glob("*/scalability_matrix.csv"))]
    data = pd.concat(frames, ignore_index=True)
    data["ai"] = pd.to_numeric(data["ai"], errors="coerce")
    data["throughput_gflops"] = pd.to_numeric(data["throughput_gflops"], errors="coerce")

    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.20, top=0.97)
    for device in DEVICES:
        subset = data[
            (data["device"] == device)
            & (data["ai"] > 0)
            & data["throughput_gflops"].notna()
        ]
        ax.scatter(subset["ai"], subset["throughput_gflops"], s=18, alpha=0.8,
                   linewidth=0, color=COLORS[device], label=DEVICE_LABELS[device])
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda value, _position: f"{value:g}" if value >= 0.1 else ""
    ))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Throughput (GFLOP/s)")
    ax.legend(loc="upper left", ncol=3, frameon=False, handlelength=0.8,
              handletextpad=0.3, columnspacing=0.7, borderaxespad=0.2)
    finish_axis(ax)
    export(fig, "fig5b_roofline")


def figure_6_density() -> None:
    measurements = pd.read_csv(RESULTS / "master_results.csv")
    datasets = pd.read_csv(RESULTS / "dataset_stats.csv")
    measurements = measurements[
        (measurements["device"] == "NPU")
        & (measurements["precision"].str.lower() == "fp32")
    ]
    measurements = measurements.groupby(["model", "dataset"], as_index=False)["mean_ms"].mean()
    data = measurements.merge(
        datasets[["dataset", "edges_per_node"]], on="dataset", how="inner"
    ).dropna(subset=["mean_ms", "edges_per_node"])

    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.20, top=0.97)
    dataset_order = ["ogbn-arxiv", "ogbn-products", "ogbn-proteins"]
    dataset_colors = dict(zip(dataset_order, [COLORS["NPU"], COLORS["GPU"], COLORS["CPU"]]))
    for dataset in dataset_order:
        subset = data[data["dataset"] == dataset]
        ax.scatter(subset["edges_per_node"], subset["mean_ms"], s=22, alpha=0.82,
                   linewidth=0, color=dataset_colors[dataset], label=dataset)

    x = data["edges_per_node"].to_numpy(dtype=float)
    y = data["mean_ms"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="#222222", linewidth=1.1)
    if len(x) > 2:
        residual = y - (slope * x + intercept)
        residual_error = np.sqrt(np.sum(residual ** 2) / (len(x) - 2))
        spread = np.sum((x - x.mean()) ** 2)
        mean_error = residual_error * np.sqrt(1 / len(x) + (x_line - x.mean()) ** 2 / spread)
        ax.fill_between(x_line, y_line - 1.96 * mean_error, y_line + 1.96 * mean_error,
                        color="#777777", alpha=0.2, linewidth=0)
    ax.text(0.30, 0.95, "r≈0", transform=ax.transAxes,
            ha="center", va="top", fontsize=8)
    ax.set_xlabel("Edges per node (graph density)")
    ax.set_ylabel("NPU latency (ms)")
    ax.legend(loc="lower center", ncol=3, frameon=False, handlelength=0.8,
              handletextpad=0.3, columnspacing=0.7, borderaxespad=0.2)
    finish_axis(ax)
    export(fig, "fig7_density_vs_latency")


def main() -> None:
    figure_1_latency()
    figure_2_int8_heatmap()
    figure_3_operators()
    figure_4_optimization()
    figure_5_roofline()
    figure_6_density()
    print("Generated IEEE-size PNG/SVG/PDF versions for paper Figures 1-6.")


if __name__ == "__main__":
    main()
