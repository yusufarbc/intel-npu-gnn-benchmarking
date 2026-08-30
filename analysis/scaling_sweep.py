from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Fix sys.path for subprocess execution
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.model_prep import export_gnn_models
from analysis.plot_config import apply_ieee_style, savefig_ieee, IEEE_COLORS
from analysis.scalability_analyzer import MultiModelPipeline, ScalabilityConfig

DEFAULT_EDGES_PER_NODE = {
    "ogbn-arxiv": 7,
    "ogbn-products": 25,
    "ogbn-proteins": 452,
}

apply_ieee_style()


@dataclass
class ScalingConfig:
    out_dir: Path
    dataset: str
    device: str
    dataset_root: Path
    sizes: List[int]
    edges_per_node: int
    features: int
    classes: int
    repeats: int
    iterations: int
    warmup: int
    profile: bool
    model_filter: str


def run_scaling(cfg: ScalingConfig) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n in cfg.sizes:
        e = int(max(1, n * cfg.edges_per_node))
        models_dir = cfg.out_dir / "models" / f"n{n}_e{e}"
        models_dir.mkdir(parents=True, exist_ok=True)

        if not list(models_dir.glob("*.onnx")):
            export_gnn_models(models_dir, num_nodes=n, num_edges=e, num_features=cfg.features, num_classes=cfg.classes)

        models = sorted([m for m in models_dir.glob("*.onnx") if cfg.model_filter.lower() in m.stem.lower()])
        if not models:
            raise FileNotFoundError(f"No models matching '{cfg.model_filter}' in {models_dir}")

        results_dir = cfg.out_dir / "runs" / f"n{n}_e{e}"
        scfg = ScalabilityConfig(
            models=models,
            results_dir=results_dir,
            device=cfg.device,
            iterations=cfg.iterations,
            warmup_iterations=cfg.warmup,
            repeats=cfg.repeats,
            enable_profiling=cfg.profile,
            input_source=cfg.dataset,
            dataset_root=cfg.dataset_root,
        )
        MultiModelPipeline(scfg).run()

        matrix_path = results_dir / "scalability_matrix.csv"
        if not matrix_path.exists():
            print(f"  ⚠️ No results for n={n}, e={e}: {matrix_path} not found. Skipping.")
            continue

        mat = pd.read_csv(matrix_path)
        if mat.empty:
            print(f"  ⚠️ Empty results for n={n}, e={e}. Skipping.")
            continue
        # Take first row (single model)
        r = mat.iloc[0].to_dict()
        r.update({"num_nodes": n, "num_edges": e, "edges_per_node": float(e) / float(n)})
        rows.append(r)

    df = pd.DataFrame(rows).sort_values("num_nodes")
    df.to_csv(cfg.out_dir / "scaling_sweep.csv", index=False)

    # Create comprehensive scaling figure with both node and edge scaling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0))

    # Panel 1: Nodes vs Latency
    ax1.plot(df["num_nodes"], df["o_mean_ms"], marker="o", color=IEEE_COLORS[0],
             linewidth=1.5, markersize=6, label='Measured')

    # Add trend line
    z = np.polyfit(df["num_nodes"], df["o_mean_ms"], 1)
    p = np.poly1d(z)
    ax1.plot(df["num_nodes"], p(df["num_nodes"]), "--", color=IEEE_COLORS[4],
             linewidth=1.0, alpha=0.7, label=f'Linear fit (slope={z[0]:.4f})')

    ax1.set_xlabel("Number of Nodes (N)", fontsize=9)
    ax1.set_ylabel("Latency (ms)", fontsize=9)
    ax1.set_title(f"Scaling by Nodes\n{cfg.model_filter} on {cfg.dataset}", fontsize=9)
    ax1.legend(fontsize=7, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.4)

    # Panel 2: Edges vs Latency
    ax2.plot(df["num_edges"], df["o_mean_ms"], marker="s", color=IEEE_COLORS[1],
             linewidth=1.5, markersize=6, label='Measured')

    # Add trend line
    z2 = np.polyfit(df["num_edges"], df["o_mean_ms"], 1)
    p2 = np.poly1d(z2)
    ax2.plot(df["num_edges"], p2(df["num_edges"]), "--", color=IEEE_COLORS[4],
             linewidth=1.0, alpha=0.7, label=f'Linear fit (slope={z2[0]:.6f})')

    ax2.set_xlabel("Number of Edges (E)", fontsize=9)
    ax2.set_ylabel("Latency (ms)", fontsize=9)
    ax2.set_title(f"Scaling by Edges\n{cfg.model_filter} on {cfg.dataset}", fontsize=9)
    ax2.legend(fontsize=7, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.4)

    fig.suptitle('Scaling Characteristics: Latency vs Graph Size',
                 fontsize=10, y=1.02)

    plt.tight_layout()

    out = cfg.out_dir / "fig6_scaling_nodes_edges_vs_latency"
    savefig_ieee(fig, out)
    plt.close(fig)
    return out.with_suffix(".png")


def parse_args() -> ScalingConfig:
    p = argparse.ArgumentParser(description="Run a node-scaling sweep and plot latency vs node count.")
    p.add_argument("--out-dir", default="results/scaling_sweep")
    p.add_argument("--dataset", default="ogbn-arxiv")
    p.add_argument("--device", default="NPU")
    p.add_argument("--dataset-root", default="data")
    p.add_argument("--sizes", default="512,1024,2048,4096")
    p.add_argument("--edges-per-node", type=int, default=None)
    p.add_argument("--features", type=int, default=1433)
    p.add_argument("--classes", type=int, default=7)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--model", default="GCN", help="Model name substring (e.g., GCN, GAT, GraphSAGE)")
    a = p.parse_args()

    sizes = [int(x.strip()) for x in str(a.sizes).split(",") if x.strip()]
    ds = str(a.dataset).strip().lower()
    epn = int(a.edges_per_node) if a.edges_per_node is not None else int(DEFAULT_EDGES_PER_NODE.get(ds, 25))

    return ScalingConfig(
        out_dir=Path(a.out_dir).resolve(),
        dataset=ds,
        device=str(a.device).strip(),
        dataset_root=Path(a.dataset_root).resolve(),
        sizes=sizes,
        edges_per_node=epn,
        features=int(a.features),
        classes=int(a.classes),
        repeats=int(a.repeats),
        iterations=int(a.iterations),
        warmup=int(a.warmup),
        profile=bool(a.profile),
        model_filter=str(a.model),
    )


def main() -> None:
    cfg = parse_args()
    fig = run_scaling(cfg)
    print(f"Figure: {fig}")


if __name__ == "__main__":
    main()
