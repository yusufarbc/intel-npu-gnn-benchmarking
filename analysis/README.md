# Analysis — Benchmarking And Analysis Scripts

Python scripts for onnx model export, benchmarking, profiling, and figure generation.

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `benchmark_runner.py` | Core benchmarking engine: ONNX Runtime session management, NPU/GPU/CPU provider selection, latency measurement |
| `scalability_analyzer.py` | Multi-model, multi-dataset, multi-device scalability analysis; produces `scalability_matrix.csv` |
| `density_sweep.py` | Sweeps graph density (edges/node) to characterize NPU memory-bound behavior |
| `scaling_sweep.py` | Sweeps node/edge counts for scaling characteristic plots |
| `model_prep.py` | Exports PyTorch GNN models to ONNX and applies INT8 quantization |
| `ort_profile_utils.py` | Utilities for parsing ONNX Runtime profiling JSON traces |
| `plot_config.py` | Centralised IEEE-compatible matplotlib style (`savefig_ieee`, `IEEE_COLORS`) |
| `energy_analyzer.py` | Post-processes SoCWatch energy traces |
| `energy_monitor.py` | Real-time energy monitoring during inference |
| `energy_correlation.py` | Correlates energy consumption with model features |
| `graph_topology_analyzer.py` | Analyzes ONNX graph topology (operator counts, fusion opportunities) |
| `hw_comparison.py` | Cross-architectural (CPU/GPU/NPU) latency comparison |
| `hw_comparison_runner.py` | Batch runner for hardware comparison experiments |
| `npu_internal_analyzer.py` | Internal NPU plugin analysis (subgraph decomposition) |
| `profiling_analyzer.py` | Aggregates profiling traces across runs |
| `verify_pipeline.py` | Validates that all expected outputs exist |

## Usage

Most scripts are called from the Jupyter notebook (`npu_gnn_benchmarking.ipynb`). Standalone usage:

```bash
python analysis/scalability_analyzer.py --model GCN --device NPU --iterations 100
python analysis/model_prep.py
python analysis/density_sweep.py --devices CPU,GPU,NPU
```
