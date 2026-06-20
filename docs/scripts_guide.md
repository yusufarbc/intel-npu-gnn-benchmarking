# Technical Guide: Python Analysis Scripts

Technical roles and usage details for the analysis and automation scripts developed in this project.

## 1. Core Execution Scripts

### `analysis/benchmark_runner.py`
The main inference engine that runs models on target hardware (NPU/GPU/CPU).

- **GNN Inputs:** Prepares `x` and `edge_index` tensors from real OGB datasets (with sampling/padding to match fixed ONNX input shapes).
- **Profiling:** Generates ONNX Runtime profiling traces in JSON format.
- **Provider Fallback:** Automatically detects OpenVINO EP failures and falls back to CPUExecutionProvider with a warning.

### `analysis/model_prep.py`
Exports PyTorch Geometric (PyG) GNN models to ONNX format and applies INT8 quantization.

- **Static Shapes:** Applies static shape padding instead of dynamic shapes for NPU compatibility.
- **Quantization:** Uses NNCF (Neural Network Compression Framework) for post-training INT8 quantization.

## 2. Analysis and Visualization Scripts

### `analysis/scalability_analyzer.py`
Measures performance variation across model size and complexity.

- **Roofline Model:** Visualizes model efficiency against hardware limits.
- **Pareto Frontier:** Analyzes the trade-off between parameter count and latency.
- **Output:** `results/<model>/scalability_matrix.csv`

### `analysis/density_sweep.py`
Sweeps graph density (edges-per-node ratio) to characterize NPU memory-bound behavior.

### `analysis/scaling_sweep.py`
Sweeps node/edge counts for scaling characteristic plots used in Fig. 6.

### `analysis/ort_profile_utils.py`
Parses ONNX Runtime profiling JSON traces to extract per-operator timing data.

### `analysis/plot_config.py`
Centralizes IEEE-compatible matplotlib styling (`savefig_ieee`, `IEEE_COLORS`, figure dimensions).
