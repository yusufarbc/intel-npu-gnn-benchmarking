# Figure Guide

How to read the figures produced by the benchmark pipeline.

## Fig. 1 — Latency Comparison (`fig1_latency_comparison`)
Cross-device (CPU / iGPU / NPU) latency across all 14 models.

- **X-Axis:** Model name.
- **Y-Axis:** Mean inference latency (ms), log scale.
- **Interpretation:** Dense vision models (ResNet50, MobileNetV2) show strong NPU advantage; GNNs show limited or no NPU benefit.

## Fig. 2 — INT8 Speedup Heatmap (`fig2_int8_speedup_heatmap`)
INT8 vs FP32 speedup ratio per model per device.

- **Color Scale:** Values > 1.0 (green) indicate INT8 is faster; values < 1.0 (red) indicate slowdown.
- **Interpretation:** NPU INT8 often degrades performance for GNNs (e.g., SGC INT8 is 2× slower than FP32).

## Fig. 3 — Operator Breakdown (`fig3_operator_breakdown`)
ONNX operator composition per model.

- **Interpretation:** GNNs rely heavily on Gather/Scatter ops not natively accelerated by NPU, unlike MatMul-dominated vision models.

## Fig. 4 — CPU Fallback Heatmap (`fig4_cpu_fallback_heatmap`)
Proportion of operators executed on CPU (fallback) vs. NPU per model.

- **Interpretation:** High CPU fallback indicates poor NPU operator coverage. Attention-based GNNs (GAT, GATv2) fail NPU compilation entirely.

## Fig. 5a — Optimization Speedup (`fig5a_opt_speedup`)
Graph optimization benefit (Fusion Gain Ratio). Values > 1.0 indicate beneficial optimization; values < 1.0 show optimization adds overhead.

## Fig. 5b — Roofline Model (`fig5b_roofline`)
Model position relative to NPU compute and memory bandwidth limits.

- **X-Axis:** Arithmetic Intensity (FLOPs/Byte).
- **Y-Axis:** Achieved Performance (GFLOPS).
- **Interpretation:** GNNs fall in the memory-bound region; ResNet50 approaches the compute-bound ceiling.

## Fig. 6 — Scaling Sweep (`fig6_scaling`)
Latency vs. graph size across devices. NPU latency is flat across graph sizes due to static-shape compilation — the graph is recompiled once at maximum input size.

## Fig. 7 — Density vs. Latency (`fig7_density_vs_latency`)
Correlation between graph density (edges/node) and inference latency per device.

## Fig. 8 — Latency Heatmap (`fig8_latency_heatmap`)
Full latency matrix: all models × all devices × all precision levels.
