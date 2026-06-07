# Methodology — NPU GNN Performance Characterization

This document details the scientific analysis methodology applied to characterize Graph Neural Network (GNN) inference on the Intel Core Ultra NPU architecture.

## 1. Core Performance Metrics

Beyond standard latency measurements, the following metrics are used to understand hardware–software interaction:

### Fusion Gain Ratio (FGR)
Measures the real speedup effect of compiler-level operator fusion optimizations.

$$FGR = \frac{Latency_{Baseline}}{Latency_{Optimized}}$$

- **FGR > 1:** Optimization is beneficial — fusion reduces execution time.
- **FGR < 1:** "Fusion Overhead Paradox" — the optimization overhead exceeds its gains.

### Compilation Efficiency Index (CEI)
Represents the ratio of model compilation time to the performance improvement it provides.

$$CEI = \frac{Compilation\_Time}{Latency\_Reduction}$$

### Arithmetic Intensity (AI)
Determines whether models are compute-bound or memory-bound.

$$AI = \frac{Total\_FLOPs}{Total\_Memory\_Transfer\_Bytes}$$

---

## 2. Experimental Setup

- **Hardware:** Intel Core Ultra 5 125H (Meteor Lake) — NPU 3720 (VPUX37XX)
- **Software:** OpenVINO 2025.4, ONNX Runtime 1.24.4, NNCF 3.1.0
- **Datasets:** OGB benchmark graphs (`ogbn-arxiv`, `ogbn-proteins`, `ogbn-products`). Inputs are padded to fixed shapes for NPU static-shape compilation.
- **Precision:** FP32 (baseline) and INT8 (post-training quantization via NNCF for native NPU mode)
- **Measurement:** 100 inference iterations after 5 warmup runs; mean, std, and P95 latency reported

---

## 3. Analytical Models

### Roofline Performance Model
Compares the NPU's peak compute throughput and memory bandwidth limits against the model's actual performance. Used to prove why GNNs hit the "Memory Wall" bottleneck.

### Latency Breakdown
Decomposes total inference time into three main components:
1. **Compute:** Pure kernel execution time.
2. **DMA:** Data transfer time between CPU and NPU.
3. **Dispatch:** Kernel launch and scheduling overhead.

---

*Last updated: 1 June 2026*
