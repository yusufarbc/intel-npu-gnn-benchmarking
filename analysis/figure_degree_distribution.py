from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from analysis.plot_config import apply_ieee_style, savefig_ieee, IEEE_COLORS

apply_ieee_style()


@dataclass
class DegreePlotConfig:
    datasets: List[str]
    dataset_root: Path
    out_dir: Path
    max_degree: int | None = None
    use_undirected: bool = True


def _load_pyg_data(name: str, dataset_root: Path):
    name = name.strip().lower()

    if name in {"cora", "citeseer", "pubmed"}:
        from torch_geometric.datasets import Planetoid  # type: ignore

        ds = Planetoid(root=str(dataset_root / "planetoid"), name=name.capitalize())
        return ds[0]

    if name == "reddit":
        from torch_geometric.datasets import Reddit  # type: ignore

        ds = Reddit(root=str(dataset_root / "reddit"))
        return ds[0]

    if name in {"ogbn-arxiv", "ogbn-products"}:
        from ogb.nodeproppred import PygNodePropPredDataset  # type: ignore

        ds = PygNodePropPredDataset(name=name, root=str(dataset_root / "ogb"))
        return ds[0]

    raise ValueError(f"Unknown dataset: {name}")


def _degree_hist(edge_index: "np.ndarray", num_nodes: int, *, undirected: bool) -> Dict[int, int]:
    # edge_index: shape [2, E]
    src = edge_index[0]
    dst = edge_index[1]

    deg = np.zeros((num_nodes,), dtype=np.int64)
    # out-degree
    np.add.at(deg, src, 1)
    if undirected:
        np.add.at(deg, dst, 1)

    values, counts = np.unique(deg, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def plot_degree_distributions(cfg: DegreePlotConfig) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.8, 3.2))

    max_k_seen = 0
    all_hists: Dict[str, Dict[int, int]] = {}

    for i, ds in enumerate(cfg.datasets):
        data = _load_pyg_data(ds, cfg.dataset_root)
        edge_index = data.edge_index.detach().cpu().numpy()
        num_nodes = int(getattr(data, "num_nodes", 0) or (int(data.x.shape[0]) if getattr(data, "x", None) is not None else 0) or 1)

        hist = _degree_hist(edge_index, num_nodes, undirected=cfg.use_undirected)
        if cfg.max_degree is not None:
            hist = {k: v for k, v in hist.items() if k <= int(cfg.max_degree)}

        if hist:
            max_k_seen = max(max_k_seen, max(hist.keys()))
        all_hists[ds] = hist

        # Convert to sorted arrays for log-log.
        ks = np.array(sorted(hist.keys()), dtype=np.int64)
        fs = np.array([hist[int(k)] for k in ks], dtype=np.int64)

        # Avoid zeros for log scale.
        mask = (ks > 0) & (fs > 0)
        ks = ks[mask]
        fs = fs[mask]

        ax.plot(
            ks,
            fs,
            label=ds,
            linewidth=1.2,
            color=IEEE_COLORS[i % len(IEEE_COLORS)],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree (k)")
    ax.set_ylabel("Frequency (#nodes)")
    ax.set_title("Degree distribution (log-log)")
    ax.legend(framealpha=0.9)

    out = cfg.out_dir / "degree_distribution_loglog"
    savefig_ieee(fig, out)
    plt.close(fig)
    return out.with_suffix(".png")


def parse_args() -> DegreePlotConfig:
    p = argparse.ArgumentParser(description="Plot degree distributions for real graph datasets (log-log).")
    p.add_argument("--datasets", default="ogbn-arxiv,reddit,ogbn-products")
    p.add_argument("--dataset-root", default="data")
    p.add_argument("--out-dir", default="results/figures")
    p.add_argument("--max-degree", type=int, default=None)
    p.add_argument("--directed", action="store_true", help="Use out-degree only (no symmetrization).")
    a = p.parse_args()

    datasets = [d.strip() for d in str(a.datasets).split(",") if d.strip()]

    return DegreePlotConfig(
        datasets=datasets,
        dataset_root=Path(a.dataset_root).resolve(),
        out_dir=Path(a.out_dir).resolve(),
        max_degree=a.max_degree,
        use_undirected=not bool(a.directed),
    )


def main() -> None:
    cfg = parse_args()
    path = plot_degree_distributions(cfg)
    print(f"Figure: {path}")


if __name__ == "__main__":
    main()
