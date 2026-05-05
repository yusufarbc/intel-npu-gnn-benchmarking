from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.density_sweep import DEFAULT_EDGES_PER_NODE
from analysis.model_prep import export_gnn_models
from analysis.plot_config import apply_ieee_style, savefig_ieee, IEEE_COLORS
from analysis.scalability_analyzer import MultiModelPipeline, ScalabilityConfig

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

        mat = pd.read_csv(results_dir / "scalability_matrix.csv")
        # Take first row (single model)
        r = mat.iloc[0].to_dict()
        r.update({"num_nodes": n, "num_edges": e, "edges_per_node": float(e) / float(n)})
        rows.append(r)

    df = pd.DataFrame(rows).sort_values("num_nodes")
    df.to_csv(cfg.out_dir / "scaling_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(df["num_nodes"], df["o_mean_ms"], marker="o", color=IEEE_COLORS[0], linewidth=1.3)
    ax.set_xlabel("#Nodes (N)")
    ax.set_ylabel("Latency (ms) [optimized]")
    ax.set_title(f"Scaling: {cfg.model_filter} on {cfg.dataset} ({cfg.device})")
    ax.grid(True, alpha=0.2)

    out = cfg.out_dir / "scaling_nodes_vs_latency"
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
