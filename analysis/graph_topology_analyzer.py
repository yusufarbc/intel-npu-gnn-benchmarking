"""
Graph Topology Analyzer
=====================
Analyzes graph topological properties for real datasets.
Supports OGBN-Arxiv, Reddit, OGBN-Products.

Generates:
- Degree distribution plot (log-log) - REQUIRED FIGURE 2
- Comprehensive statistics CSV/JSON
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


@contextlib.contextmanager
def _suppress_stdout_stderr():
    """Temporarily suppress stdout/stderr to hide tqdm progress bars."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

from analysis.plot_config import (
    apply_ieee_style, savefig_ieee, IEEE_COLORS,
    SINGLE_COL, DOUBLE_COL
)

apply_ieee_style()


@dataclass
class GraphStatistics:
    """Comprehensive statistics for a graph dataset."""
    dataset_name: str
    num_nodes: int
    num_edges: int
    avg_degree: float
    degree_variance: float
    degree_std: float
    edge_node_ratio: float
    density: float
    clustering_coeff: float
    num_connected_components: int
    diameter_approx: Optional[int]
    max_degree: int
    min_degree: int
    median_degree: float
    power_law_alpha: Optional[float]
    is_directed: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


class GraphTopologyAnalyzer:
    """
    Analyzes graph topological properties for real datasets.
    Generates publication-quality degree distribution plots.
    """

    # Dataset configurations with expected characteristics
    DATASETS = {
        "ogbn-arxiv": {
            "loader": "ogb",
            "expected_nodes": 169343,
            "expected_edges": 1166243,
            "description": "Sparse citation graph - demonstrates memory-bound execution",
        },
        "reddit": {
            "loader": "pyg",
            "expected_nodes": 232965,
            "expected_edges": 114615892,
            "description": "Medium density social network",
        },
        "ogbn-products": {
            "loader": "ogb",
            "expected_nodes": 2449029,
            "expected_edges": 61859140,
            "description": "Dense product co-purchase graph",
        },
    }

    # Static cache to avoid re-loading datasets across analyses
    _dataset_cache: Dict[str, Any] = {}

    def __init__(self, results_dir: Path, dataset_root: Path = None):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_root = Path(dataset_root) if dataset_root else Path("data")
        self.stats_cache: Dict[str, GraphStatistics] = {}

    def load_dataset(self, name: str):
        """Load dataset using appropriate loader."""
        name = name.strip().lower()
        cache_key = f"{name}:{self.dataset_root}"
        if cache_key in GraphTopologyAnalyzer._dataset_cache:
            print(f"  [cache] Using cached dataset: {name}")
            return GraphTopologyAnalyzer._dataset_cache[cache_key]

        config = self.DATASETS.get(name)
        if config is None:
            raise ValueError(f"Unknown dataset: {name}")

        print(f"Loading dataset: {name}...")
        start = time.time()

        if config["loader"] == "ogb":
            from ogb.nodeproppred import PygNodePropPredDataset
            with _suppress_stdout_stderr():
                dataset = PygNodePropPredDataset(name=name, root=str(self.dataset_root / "ogb"))
                data = dataset[0]
        elif config["loader"] == "pyg":
            if name == "reddit":
                from torch_geometric.datasets import Reddit
                with _suppress_stdout_stderr():
                    dataset = Reddit(root=str(self.dataset_root / "reddit"))
                    data = dataset[0]
            else:
                raise ValueError(f"Unknown PyG dataset: {name}")
        else:
            raise ValueError(f"Unknown loader: {config['loader']}")

        load_time = time.time() - start
        print(f"  Loaded in {load_time:.2f}s - {data.num_nodes:,} nodes, {data.num_edges:,} edges")
        GraphTopologyAnalyzer._dataset_cache[cache_key] = data
        return data

    def compute_statistics(self, name: str) -> GraphStatistics:
        """Compute comprehensive statistics for a dataset."""
        if name in self.stats_cache:
            return self.stats_cache[name]

        data = self.load_dataset(name)
        edge_index = data.edge_index.cpu().numpy()

        num_nodes = int(data.num_nodes)
        num_edges = int(data.num_edges)

        # Compute degree distribution
        degrees = self._compute_degrees(edge_index, num_nodes)

        # Basic statistics
        avg_degree = float(np.mean(degrees))
        degree_variance = float(np.var(degrees))
        degree_std = float(np.std(degrees))
        max_degree = int(np.max(degrees))
        min_degree = int(np.min(degrees))
        median_degree = float(np.median(degrees))

        # Graph density (for undirected graphs)
        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0

        # Edge/node ratio
        edge_node_ratio = num_edges / num_nodes if num_nodes > 0 else 0

        # Power law estimation using linear regression on log-log data
        power_law_alpha = self._estimate_power_law_alpha(degrees)

        # Clustering coefficient (approximate for large graphs using sampling)
        clustering_coeff = self._approximate_clustering(data, edge_index, num_nodes, sample_size=10000)

        # Connected components and diameter (using sampling for large graphs)
        num_components, diameter = self._approximate_topology(data, edge_index, num_nodes, sample_size=5000)

        stats = GraphStatistics(
            dataset_name=name,
            num_nodes=num_nodes,
            num_edges=num_edges,
            avg_degree=avg_degree,
            degree_variance=degree_variance,
            degree_std=degree_std,
            edge_node_ratio=edge_node_ratio,
            density=density,
            clustering_coeff=clustering_coeff,
            num_connected_components=num_components,
            diameter_approx=diameter,
            max_degree=max_degree,
            min_degree=min_degree,
            median_degree=median_degree,
            power_law_alpha=power_law_alpha,
            is_directed=False,
        )

        self.stats_cache[name] = stats
        return stats

    def _compute_degrees(self, edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
        """Compute degree for each node."""
        degrees = np.zeros(num_nodes, dtype=np.int64)
        src = edge_index[0]
        dst = edge_index[1]
        np.add.at(degrees, src, 1)
        np.add.at(degrees, dst, 1)
        return degrees

    def _estimate_power_law_alpha(self, degrees: np.ndarray) -> Optional[float]:
        """Estimate power law exponent using linear regression on log-log CCDF."""
        # Filter out zero degrees
        positive_degrees = degrees[degrees > 0]
        if len(positive_degrees) < 100:
            return None

        # Compute CCDF
        unique_degrees, counts = np.unique(positive_degrees, return_counts=True)
        ccdf = np.cumsum(counts[::-1])[::-1] / len(positive_degrees)

        # Filter for linear regression (k >= k_min)
        k_min = np.percentile(unique_degrees, 10)
        mask = unique_degrees >= k_min

        if np.sum(mask) < 10:
            return None

        log_k = np.log(unique_degrees[mask])
        log_ccdf = np.log(ccdf[mask])

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_ccdf)
        alpha = -slope + 1  # Convert CCDF slope to power law alpha

        return float(alpha) if not np.isnan(alpha) and alpha > 1 else None

    def _approximate_clustering(self, data, edge_index: np.ndarray, num_nodes: int, sample_size: int = 10000) -> float:
        """Approximate clustering coefficient using node sampling."""
        try:
            import torch
            from torch_geometric.utils import subgraph

            # Sample nodes
            if num_nodes <= sample_size:
                sample_nodes = torch.arange(num_nodes)
            else:
                sample_nodes = torch.randperm(num_nodes)[:sample_size]

            # Extract subgraph
            edge_index_tensor = torch.from_numpy(edge_index)
            sub_edge_index, _ = subgraph(sample_nodes, edge_index_tensor, relabel_nodes=True, num_nodes=num_nodes)

            # Count triangles approximately
            num_sample = len(sample_nodes)
            adj = torch.zeros((num_sample, num_sample), dtype=torch.bool)
            adj[sub_edge_index[0], sub_edge_index[1]] = True

            # C_i = number of triangles / (k_i * (k_i - 1))
            triangles = torch.mm(adj.float(), adj.float()) * adj.float()
            triangle_counts = triangles.sum(dim=1) / 2

            degrees_sample = adj.sum(dim=1).float()
            possible_triangles = degrees_sample * (degrees_sample - 1) / 2
            possible_triangles = torch.clamp(possible_triangles, min=1)

            clustering = (triangle_counts / possible_triangles).mean().item()
            return float(clustering) if not np.isnan(clustering) else 0.0
        except Exception as e:
            print(f"  Warning: Could not compute clustering coefficient: {e}")
            return 0.0

    def _approximate_topology(self, data, edge_index: np.ndarray, num_nodes: int, sample_size: int = 5000) -> Tuple[int, Optional[int]]:
        """Approximate connected components and diameter using BFS sampling."""
        try:
            from collections import deque

            # Build adjacency list for sampled nodes
            if num_nodes > sample_size:
                sample_nodes = np.random.choice(num_nodes, size=sample_size, replace=False)
                sample_set = set(sample_nodes)

                # Filter edges to sampled nodes
                mask = np.isin(edge_index[0], sample_nodes) & np.isin(edge_index[1], sample_nodes)
                sub_edges = edge_index[:, mask]

                # Remap node indices
                node_map = {n: i for i, n in enumerate(sample_nodes)}
                adj_list = [set() for _ in range(sample_size)]
                for i in range(sub_edges.shape[1]):
                    src = node_map.get(int(sub_edges[0, i]))
                    dst = node_map.get(int(sub_edges[1, i]))
                    if src is not None and dst is not None:
                        adj_list[src].add(dst)
                        adj_list[dst].add(src)

                num_components, max_diameter = self._bfs_components_diameter(adj_list)
                return num_components, max_diameter
            else:
                # Full graph analysis
                adj_list = [set() for _ in range(num_nodes)]
                for i in range(edge_index.shape[1]):
                    src = int(edge_index[0, i])
                    dst = int(edge_index[1, i])
                    adj_list[src].add(dst)
                    adj_list[dst].add(src)

                num_components, max_diameter = self._bfs_components_diameter(adj_list)
                return num_components, max_diameter

        except Exception as e:
            print(f"  Warning: Could not compute topology metrics: {e}")
            return 1, None

    def _bfs_components_diameter(self, adj_list: List[set]) -> Tuple[int, int]:
        """BFS to count components and estimate diameter."""
        n = len(adj_list)
        visited = [False] * n
        num_components = 0
        max_diameter = 0

        for start in range(n):
            if visited[start]:
                continue

            num_components += 1
            # BFS
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            component_nodes = [start]

            while queue:
                u = queue.popleft()
                for v in adj_list[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                        component_nodes.append(v)

            # Update visited
            for node in component_nodes:
                visited[node] = True
                if dist[node] > max_diameter:
                    max_diameter = dist[node]

        return num_components, max_diameter

    def analyze_degree_distribution(self, dataset_names: List[str]) -> pd.DataFrame:
        """
        REQUIRED FIGURE 2: Degree Distribution (log-log)

        Calculates and plots degree distribution for multiple datasets.
        Generates publication-quality log-log plot comparing all datasets.
        """
        fig, ax = plt.subplots(figsize=SINGLE_COL)

        all_stats = []
        degree_data = {}

        for i, name in enumerate(dataset_names):
            print(f"\nProcessing dataset: {name}")
            stats = self.compute_statistics(name)
            all_stats.append(stats.to_dict())

            # Get degrees for plotting
            data = self.load_dataset(name)
            edge_index = data.edge_index.cpu().numpy()
            degrees = self._compute_degrees(edge_index, data.num_nodes)
            degree_data[name] = degrees

            # Histogram for log-log plot
            positive_degrees = degrees[degrees > 0]
            if len(positive_degrees) > 0:
                counts, bins = np.histogram(positive_degrees,
                                            bins=np.logspace(0, np.log10(max(positive_degrees)), 50))
                bin_centers = (bins[:-1] + bins[1:]) / 2

                # Filter out zeros for log plot
                mask = counts > 0
                ax.loglog(bin_centers[mask], counts[mask], marker='o', markersize=2,
                          linestyle='-', linewidth=1.2, label=name,
                          color=IEEE_COLORS[i % len(IEEE_COLORS)])

                # Add power law fit line if available
                if stats.power_law_alpha:
                    x_fit = np.logspace(np.log10(bin_centers[mask].min()),
                                       np.log10(bin_centers[mask].max()), 100)
                    y_fit = (x_fit ** (-stats.power_law_alpha + 1)) * counts[mask].max()
                    ax.loglog(x_fit, y_fit, '--', linewidth=0.8,
                             color=IEEE_COLORS[i % len(IEEE_COLORS)], alpha=0.5)

        ax.set_xlabel("Degree (k)", fontsize=9)
        ax.set_ylabel("Frequency P(k)", fontsize=9)
        ax.set_title("Degree Distribution (Log-Log Scale)", fontsize=10)
        ax.legend(fontsize=7, framealpha=0.9)
        ax.grid(True, which="both", alpha=0.3, linestyle='--', linewidth=0.4)

        # Add annotation about power-law behavior
        ax.annotate("Heavy-tailed\ndistribution", xy=(0.7, 0.7), xycoords='axes fraction',
                   fontsize=7, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        savefig_ieee(fig, self.results_dir / "fig2_degree_distribution_loglog")
        plt.close(fig)
        print(f"\nGenerated: {self.results_dir / 'fig2_degree_distribution_loglog.png'}")

        # Save statistics to CSV and JSON
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(self.results_dir / "dataset_statistics.csv", index=False)

        with open(self.results_dir / "dataset_statistics.json", "w") as f:
            json.dump(all_stats, f, indent=2)

        print(f"Saved statistics to:")
        print(f"  - {self.results_dir / 'dataset_statistics.csv'}")
        print(f"  - {self.results_dir / 'dataset_statistics.json'}")

        return stats_df

    def get_statistics_summary(self) -> pd.DataFrame:
        """Return cached statistics as DataFrame."""
        if not self.stats_cache:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in self.stats_cache.values()])


if __name__ == "__main__":
    analyzer = GraphTopologyAnalyzer(
        results_dir=Path("results/figures"),
        dataset_root=Path("data")
    )
    stats = analyzer.analyze_degree_distribution(["ogbn-arxiv", "reddit", "ogbn-products"])
    print("\nDataset Statistics Summary:")
    print(stats.to_string(index=False))
