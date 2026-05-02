# Evaluating Operator Fusion and the Memory Wall for GNNs on Edge NPUs

This project provides a comprehensive framework for analyzing the performance of **Graph Neural Networks (GNNs)** on the **Intel Core Ultra (Meteor Lake)** architecture. It enables cross-architectural benchmarking (CPU vs iGPU vs NPU) and introduces formal metrics to quantify the efficiency of graph-based hardware acceleration.

## Key Features

- **10-Model Taxonomy:** Evaluation across spectral, inductive, and attention-based GNNs (GCN, GAT, GraphSAGE, etc.) alongside CV (ResNet) and NLP (BERT) baselines.
- **Advanced Metrics:** Introduction of **Fusion Gain Ratio (FGR)** and **Compilation Efficiency Index (CEI)** to diagnose compiler efficacy beyond raw latency.
- **Dataset Integration:** Standardized support for the **Cora** citation network topology (2708 nodes, 5429 edges).
- **Academic Quality Visualizations:** Automatic generation of 300 DPI PNG and vector SVG plots (Roofline, Pareto Frontier, Latency Breakdown).
- **IEEE-Compliant Manuscript:** Integrated LaTeX pipeline (`paper/paper.tex`) ready for academic submission.

## Project Structure

- `analysis/`: High-level analyzers for scalability, hardware comparison, and profiling.
- `scripts/`: Utilities for model generation (`generate_gnn_models.py`) and dataset preparation.
- `docs/`: Technical guides for [models](docs/models_guide.md), [visualizations](docs/visualizations_guide.md), and methodology.
- `models/`: Standardized ONNX models (FP32 & INT8).
- `results/`: Empirical data, CSV matrices, and high-resolution research figures.
- `paper/`: The 5-page IEEE-compliant research manuscript and bibliography.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Pipeline

### 1. Hardware-Software Comparison (3-Way)
To compare **CPU**, **iGPU (Intel Arc)**, and **NPU (AI Boost)** performance for a specific model:
```bash
python analysis/hw_comparison.py --model models/GCN_fp32.onnx --iterations 100
```

### 2. Scalability and Roofline Analysis
To evaluate model behavior across hidden dimensions and generate the Roofline Model:
```bash
python analysis/scalability_analyzer.py --models-dir models/ --device NPU
```

### 3. Full Research Pipeline
To execute the complete suite (Benchmarking -> Analysis -> Figure Export):
```bash
python run_pipeline.py --iterations 100 --repeats 3
```

## Research Findings: The "Memory Wall"
Our analysis indicates that while the Intel Core Ultra NPU excels at dense CNN workloads (ResNet50), it is significantly bottlenecked by memory subsystems when executing sparse GNNs. We identify a **"Fusion Overhead Paradox"** where aggressive compiler optimizations can lead to performance regression in shallow graph structures.

---
*Developed by Yusuf Talha ARABACI (Karabük University) & Antigravity AI Assistant*
