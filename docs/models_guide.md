# Model Guide — NPU GNN Benchmarking

This document describes the Graph Neural Networks (GNNs) and baseline models used in the Intel Core Ultra NPU benchmark study.

## 1. Graph Neural Networks (GNNs)

These models were selected to measure NPU performance on sparse data structures and irregular memory access patterns.

| Model | Architecture | Selection Rationale |
| :--- | :--- | :--- |
| **GCN** | Spectral Convolution | The foundational baseline of GNN literature. |
| **GAT** | Attention Mechanism | Measures the overhead of dynamic attention weighting on NPU. |
| **GATv2** | Dynamic Attention | Enhanced attention with improved expressiveness over GAT. |
| **GraphSAGE** | Inductive Learning | Tests sampling efficiency for large graphs. |
| **GIN** | Isomorphism Network | Maximum expressivity and complex aggregation test. |
| **SGC** | Simplified Convolution | Measures the effect of removing unnecessary non-linear layers. |
| **APPNP** | Personalized PageRank | Effect of multi-hop neighborhood propagation on NPU memory management. |
| **GraphTransformer** | Hybrid Self-Attention | NPU compatibility of GNN–Transformer hybrid architectures. |
| **MPNN** | Message Passing | General message-passing framework; baseline for learned edge features. |

## 2. Dense Baseline Models

Used to contrast NPU performance in its intended use case (dense CNNs/Transformers) against GNNs.

| Model | Type | FP32 | INT8 | Params |
|-------|------|------|------|--------|
| **ResNet50** | CNN | ✅ | ✅ | 25.5M |
| **MobileNetV2** | CNN | ✅ | ✅ | 3.5M |
| **EfficientNet-B0** | CNN | ✅ | ❌ | 5.3M |
| **ViT-Tiny** | Vision Transformer | ✅ | ❌ | 5.7M |
| **BERT-Tiny** | NLP Transformer | ✅ | ✅ | 4.4M |

## 3. Precision and Versions

Each model is tested at two precision levels:

1. **FP32 (Floating Point 32):** Original precision. Highest accuracy but least efficient on NPU hardware.
2. **INT8 (Integer 8):** Quantized via NNCF (Neural Network Compression Framework). Activates the NPU's hardware accelerators (Movidius VPU/NPU IP) at full capacity — the "native" NPU mode.

## 4. Known Limitations and Compilation Failures

Several architecture-specific incompatibilities were discovered during benchmarking. These are reported as research findings in the paper under *"NPU Hardware/Software Maturity Analysis"*:

### INT8 Quantization Failures
- **GAT / GATv2:** The OpenVINO Execution Provider fails with `Output names mismatch between OpenVINO and ONNX` during NPU compilation. The NPU plugin cannot yet fully reconcile the output naming of attention sub-graphs. These models run in CPU fallback mode.
- **EfficientNet-B0:** INT8 compilation fails due to unsupported operator patterns in the NPU plugin.
- **ViT-Tiny:** INT8 compilation fails due to shape inference issues during quantization.

### NPU Hardware Compiler Constraints (VPUX37XX)

1. **Negative Post-Shift Quantization Error:** Some GNN INT8 models (e.g., `APPNP_int8`) produce extreme scaling factors during NNCF quantization. The NPU hardware compiler rejects these with: `postShift is not supported`. This is a hardware limitation — not a model bug.

2. **Intel Graphics Compiler (IGC) Crashes:** The shared IGC compiler can produce fatal memory segmentation errors (`intersects with V37`) when processing certain INT8 GNN graphs. The pipeline includes GPU bypass mechanisms to prevent full-process crashes.

> **Academic Note:** These failures are valuable research findings demonstrating current limitations of the Intel Core Ultra NPU software stack for sparse GNN workloads, rather than simple bugs.

---

*Last updated: 1 June 2026*