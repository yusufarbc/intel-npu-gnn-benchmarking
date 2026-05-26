# BENCHMARKING GNN INFERENCE BOTTLENECKS ON INTEL CORE ULTRA NPUS

A comprehensive benchmarking framework for evaluating graph neural network (gnn) inference on intel core ultra (meteor lake) npus, comparing against cpu and integrated gpu (igpu) backends under openvino.

## KEY FEATURES

- **14 Models:** 9 gnns (gcn, gat, gatv2, gin, graphsage, sgc, appnp, graphtransformer, mpnn) + 5 dense baselines (resnet50, mobilenetv2, efficientnet-b0, vit-tiny, bert-tiny)
- **3 Hardware Backends:** Cpu, integrated gpu (xe-lpg), and npu (intel ai boost)
- **3 Real-World Datasets:** Ogbn-arxiv, ogbn-proteins, ogbn-products from the Open Graph Benchmark
- **Int8 Quantization Analysis:** Fp32 vs int8 speedup across all models and devices
- **Operator Profiling:** Per-operator cpu fallback detection and onnx operator composition analysis
- **Publication-Ready Figures:** 7 ieee-format figures (png + svg, 300 dpi)
- **Accompanying Paper:** Latex source for ieee conference submission
- **Interactive Presentation:** Web-based Slidev presentation with custom academic theme and interactive term notes (`showcase/`)

## PROJECT STRUCTURE

| Directory | Description |
|-----------|-------------|
| `analysis/` | Python scripts for benchmarking, profiling, and analysis |
| `data/` | Ogb graph datasets (excluded from git) |
| `docs/` | Technical documentation and methodology guides |
| `models/` | Pre-exported onnx models (fp32 and int8) |
| `paper/` | Latex paper source and compiled pdf |
| `results/` | Benchmark outputs, csv matrices, and figures |
| `showcase/` | Interactive Slidev presentation source and assets |

## REQUIREMENTS

This project requires an **Intel Core Ultra (Meteor Lake)** system with:

| Dependency | Purpose | Installation |
|-----------|---------|-------------|
| **Python ≥ 3.10** | Runtime | [python.org](https://python.org) |
| **OpenVINO 2024.1+** | NPU/GPU/CPU inference via ONNX Runtime | `pip install openvino` |
| **ONNX Runtime** | Model execution with NPU plugin | `pip install onnxruntime onnxruntime-openvino` |
| **Intel SoCWatch** | Energy/power profiling (optional) | Ships with **Intel VTune Profiler** — install from [Intel.com](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html) |
| **Python packages** | See `requirements.txt` | `pip install -r requirements.txt` |

> **SoCWatch** is optional. The notebook auto-detects it. Set `SOCWATCH_ENABLED = False` in the first code cell to disable energy profiling and halve benchmark runtime.

## INSTALLATION

```bash
# 1. Clone the repository
git clone https://github.com/yusufarbc/intel-npu-gnn-benchmarking.git
cd intel-npu-gnn-benchmarking

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## USAGE — Reproducing the Paper Results

> **⚠️ Hardware requirement:** This project requires an **Intel Core Ultra (Meteor Lake)** processor with NPU. Benchmarks run on CPU, iGPU, and NPU backends.

### Prerequisites

Before running the notebook, ensure the following are installed:

1. **Intel OpenVINO 2024.1+** — Provides the NPU/GPU/CPU inference backend.
   ```bash
   pip install openvino onnxruntime onnxruntime-openvino
   ```

2. **Intel SoCWatch** (optional, for energy profiling) — Ships with **Intel VTune Profiler**.
   - Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html
   - The notebook auto-detects SoCWatch; set `SOCWATCH_ENABLED = False` in Cell 1 to skip energy measurements.

3. **Python dependencies** — Install via:
   ```bash
   pip install -r requirements.txt
   ```

### Step-by-step

```bash
# Activate environment (if not already active)
.venv\Scripts\activate

# Launch Jupyter
jupyter notebook npu_gnn_benchmarking.ipynb
```

### Notebook Cells Overview

Run cells **sequentially** from top to bottom:

| Stage | Cells | Description |
|-------|-------|-------------|
| **Config** | Cell 1 | Edit model list, devices, iterations, SoCWatch toggle |
| **Stage 0** | Cell 2–4 | Imports, helper functions |
| **Stage 0-B** | Cell 5 | ⚡ **Auto-downloads OGB datasets & generates ONNX models** (if missing) |
| **Stage 1-A/B/C** | Cells 6–9 | Dataset metadata, model inventory, hardware health check |
| **Stage 2** | Cells 10–13 | **Unified batch benchmark** — all models × datasets × devices × precisions |
| **Stage 3-A/B/C** | Cells 14–17 | Merge results, comparison tables, summary statistics |
| **Stage 4** | Cells 18–39 | 7 publication-ready figures (PNG + SVG) |

> ⚡ **Datasets and models are downloaded/generated automatically** by Stage 0-B. No manual download needed.  
> ⏱️ Full benchmark run (14 models × 3 datasets × 3 devices × 2 precisions) takes **2–4 hours** with SoCWatch enabled.

### Outputs

All generated artifacts are saved to:

```
results/
├── scalability_matrices/        # Raw latency CSVs
├── comparison_table.csv         # Aggregated results
├── summary_statistics.csv       # Statistical summaries
├── profiling_traces/            # ONNX Runtime profiling JSON
└── figures/                     # Publication figures (PNG + SVG)
```

### Standalone Scripts

Individual analysis scripts can also be run directly:

```bash
# Generate ONNX models (FP32 + INT8 quantized)
python analysis/model_prep.py

# Run a single benchmark configuration
python analysis/scalability_analyzer.py --model GCN --device NPU --iterations 100

# Sweep graph density
python analysis/density_sweep.py --devices CPU,GPU,NPU

# Verify all expected outputs exist
python analysis/verify_pipeline.py
```

## KEY FINDINGS

- The NPU delivers strong FP32 performance for dense vision models (MobileNetV2: 1.97ms, ResNet50: 3.92ms)
- GNN workloads show limited NPU advantage (within 15% of CPU latency)
- INT8 quantization degrades NPU latency for most models (SGC INT8 is 2× slower than FP32)
- Attention-based GNNs (GAT, GATv2) fail INT8 compilation entirely
- Graph density shows no correlation with NPU latency (static-shape compilation)

## LICENSE

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Developed by Yusuf Talha ARABACI (Karabük University)*
