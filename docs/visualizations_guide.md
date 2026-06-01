# Visualizations & Academic Evidence Guide

This document explains how to interpret the figures produced by the benchmark pipeline in the context of an IEEE/ACM-format academic paper.

## Fig. 1 — Latency Comparison (`fig1_latency_comparison`)
**Purpose:** Cross-device (CPU / iGPU / NPU) latency comparison across all 14 models.

- **X-Axis:** Model name.
- **Y-Axis:** Mean inference latency (ms), log scale.
- **Interpretation:** Dense vision models (ResNet50, MobileNetV2) show strong NPU advantage; GNNs show limited or no NPU benefit.
- **Finding:** "NPU delivers strong FP32 throughput for dense compute workloads but shows limited advantage for sparse GNN inference."

## Fig. 2 — INT8 Speedup Heatmap (`fig2_int8_speedup_heatmap`)
**Purpose:** Shows INT8 vs FP32 speedup ratio per model per device.

- **Color Scale:** Values > 1.0 (green) indicate INT8 is faster; values < 1.0 (red) indicate slowdown.
- **Interpretation:** NPU INT8 often degrades performance for GNNs (e.g., SGC INT8 is 2× slower than FP32 on NPU).
- **Finding:** "Standard INT8 quantization is not unconditionally beneficial on NPU for sparse workloads."

## Fig. 3 — Operator Breakdown (`fig3_operator_breakdown`)
**Purpose:** ONNX operator composition analysis showing which op types dominate each model.

- **Interpretation:** GNNs rely heavily on `Gather`/`Scatter` ops not natively accelerated by the NPU, unlike MatMul-dominated vision models.

## Fig. 4 — CPU Fallback Heatmap (`fig4_cpu_fallback_heatmap`)
**Purpose:** Shows the proportion of operators executed on CPU (fallback) vs. NPU per model.

- **Interpretation:** High CPU fallback percentage indicates poor NPU operator coverage for that model.
- **Finding:** "Attention-based GNNs (GAT, GATv2) fail NPU compilation entirely and fall back to CPU."

## Fig. 5a — Optimization Speedup (`fig5a_opt_speedup`)
**Purpose:** Measures graph optimization benefit (Fusion Gain Ratio — FGR).

- **Threshold:** 1.0 (baseline).
- **Interpretation:** Values > 1.0 (green) indicate beneficial optimization; values < 1.0 (red) show the "Fusion Overhead Paradox" where optimization adds overhead.
- **Finding:** "Standard deep learning optimization strategies do not always yield positive results for sparse GNN structures."

## Fig. 5b — Roofline Model (`fig5b_roofline`)
**Purpose:** Positions each model relative to the NPU's hardware compute and memory bandwidth limits.

- **X-Axis:** Arithmetic Intensity (FLOPs/Byte).
- **Y-Axis:** Achieved Performance (GFLOPS).
- **Interpretation:** GNNs (GCN, GAT) fall in the left "Memory-Bound" region; ResNet50 approaches the "Compute-Bound" ceiling.
- **Finding:** "GNN inference is bottlenecked by memory bandwidth, not compute — explaining why NPU TOPS ratings do not translate to GNN speedup."

## Fig. 6 — Scaling Sweep (`fig6_scaling`)
**Purpose:** Latency vs. graph size (node/edge count) across devices.

- **Interpretation:** NPU latency is flat across graph sizes due to static-shape compilation — the graph is recompiled once at the maximum input size.
- **Finding:** "Graph density has no correlation with NPU latency under static-shape compilation."

## Fig. 7 — Density vs. Latency (`fig7_density_vs_latency`)
**Purpose:** Correlation analysis between graph density (edges/node) and inference latency per device.

## Fig. 8 — Latency Heatmap (`fig8_latency_heatmap`)
**Purpose:** Full latency matrix — all models × all devices × all precision levels as a color-coded table.

---

*Last updated: 1 June 2026*
