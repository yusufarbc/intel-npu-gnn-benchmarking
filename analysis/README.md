# Analysis — Benchmarking and Analysis Scripts

Python scripts for ONNX model export, benchmarking, profiling, and figure generation.

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `benchmark_runner.py` | Core benchmarking engine: ONNX Runtime session management, NPU/GPU/CPU provider selection, latency measurement |
| `scalability_analyzer.py` | Multi-model, multi-dataset, multi-device scalability analysis; produces `scalability_matrix.csv` |
| `density_sweep.py` | Sweeps graph density (edges/node) to characterize NPU memory-bound behavior |
| `scaling_sweep.py` | Sweeps node/edge counts for scaling characteristic plots |
| `model_prep.py` | Exports PyTorch GNN models to ONNX and applies INT8 quantization via NNCF |
| `ort_profile_utils.py` | Utilities for parsing ONNX Runtime profiling JSON traces |
| `plot_config.py` | Centralised IEEE-compatible matplotlib style (`savefig_ieee`, `IEEE_COLORS`) |

## Usage

Most scripts are invoked from the Jupyter notebook (`npu_gnn_benchmarking.ipynb`). They can also be run standalone:

```bash
# Run scalability analysis for a specific model/device
python analysis/scalability_analyzer.py --model GCN --device NPU --iterations 100

# Regenerate all ONNX models (FP32 + INT8)
python analysis/model_prep.py

# Sweep graph density across devices
python analysis/density_sweep.py --devices CPU,GPU,NPU
```

## Notes

- All scripts write output to `results/` (CSV files, profiling traces, figures).
- `benchmark_runner.py` is the shared engine used by `scalability_analyzer.py` and the sweep scripts.
- ONNX Runtime profiling traces are saved as JSON under `results/<model>/profiling_traces/`.
