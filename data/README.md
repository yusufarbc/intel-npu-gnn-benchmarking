# Data — Ogb Graph Datasets

Downloaded Open Graph Benchmark datasets for gnn benchmarking.

## Contents

| Dataset | Nodes | Edges | Features | Domain |
|---------|-------|-------|----------|-------|
| `ogbn-arxiv/` | 169K | 1.16M | 128 | Citation network |
| `ogbn-products/` | 2.45M | 61.9M | 100 | Amazon product co-purchasing |
| `ogbn-proteins/` | 132K | 39.6M | 8 | Protein-protein associations |

## Download

Datasets are downloaded by running Stage 1-A of the Jupyter notebook, which loads them via `ogb.nodeproppred.PygNodePropPredDataset`.

## Notes

- These datasets are **large** (~15 GB total) and excluded from git tracking.
- Only `ogbn-arxiv`, `ogbn-proteins`, and `ogbn-products` are used in the paper.
- The notebook loader falls back to a subprocess-based loading strategy to avoid kernel crashes from native library conflicts.
- The paper benchmarks fixed 2,708-node, 10,000-edge inference tensors derived from these source graphs; it does not execute the complete graphs.
