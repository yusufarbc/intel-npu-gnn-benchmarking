# Reference Library

Comprehensive list of all papers and technical reports referenced in
*Benchmarking GNN Inference Bottlenecks on Intel Core Ultra NPUs*.
Each entry includes a brief annotation explaining its relevance to the benchmark.

> BibTeX source: [`paper/references.bib`](../paper/references.bib)  
> Full text (where available): `paper/references/` directory

---

## 1. GNN Architectures (Models Benchmarked)

These are the nine GNN architectures evaluated on the Intel Core Ultra NPU.

| # | Citation | Venue | Role in Benchmark |
|---|----------|-------|-------------------|
| [1] | Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, 2017 | ICLR 2017 | **GCN** — spectral convolution baseline; foundational GNN model |
| [2] | Veličković et al., *Graph Attention Networks*, 2018 | ICLR 2018 | **GAT** — attention-based GNN; fails INT8 NPU compilation |
| [3] | Brody et al., *How Attentive are Graph Attention Networks?*, 2022 | ICLR 2022 | **GATv2** — dynamic attention variant; also fails INT8 on NPU |
| [4] | Hamilton et al., *Inductive Representation Learning on Large Graphs*, 2017 | NeurIPS 2017 | **GraphSAGE** — sampling-based inductive learning |
| [5] | Xu et al., *How Powerful are Graph Neural Networks?*, 2019 | ICLR 2019 | **GIN** — maximum expressivity; isomorphism network |
| [6] | Wu et al., *Simplifying Graph Convolutional Networks*, 2019 | ICML 2019 | **SGC** — removes non-linearities; shows INT8 paradox on NPU |
| [7] | Gasteiger et al., *Predict then Propagate: GNNs meet Personalized PageRank*, 2019 | ICLR 2019 | **APPNP** — propagation decoupled from prediction |
| [8] | Gilmer et al., *Neural Message Passing for Quantum Chemistry*, 2017 | ICML 2017 | **MPNN** — general message-passing framework |
| [9] | Dwivedi & Bresson, *A Generalization of Transformer Networks to Graphs*, 2021 | arXiv | **GraphTransformer** — hybrid GNN-Transformer architecture |

---

## 2. Dense Baseline Models

Five dense vision/NLP models used to contrast NPU behavior on its intended workload.

| # | Citation | Venue | Role |
|---|----------|-------|------|
| [10] | He et al., *Deep Residual Learning for Image Recognition*, 2016 | CVPR 2016 | **ResNet-50** — CNN NPU performance ceiling (3.92ms) |
| [11] | Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, 2018 | CVPR 2018 | **MobileNetV2** — best NPU latency (1.97ms) |
| [12] | Tan & Le, *EfficientNet: Rethinking Model Scaling*, 2019 | ICML 2019 | **EfficientNet-B0** — INT8 compilation fails on NPU |
| [13] | Dosovitskiy et al., *An Image is Worth 16×16 Words: ViT*, 2021 | ICLR 2021 | **ViT-Tiny** — Vision Transformer; INT8 fails |
| [14] | Turc et al., *Well-Read Students Learn Better (BERT-Tiny)*, 2019 | arXiv | **BERT-Tiny** — NLP Transformer; demonstrates Fusion Overhead Paradox |

---

## 3. Datasets

| # | Citation | Dataset | Used For |
|---|----------|---------|---------|
| [15] | Hu et al., *Open Graph Benchmark*, 2020 | NeurIPS 2020 | **ogbn-arxiv**, **ogbn-proteins**, **ogbn-products** — all three datasets in benchmark |

---

## 4. GNN Hardware Acceleration

Key references on accelerator design and performance characterization for GNNs.

| # | Citation | Venue | Key Contribution |
|---|----------|-------|-----------------|
| [16] | Auten et al., *Hardware Acceleration of GNNs*, 2020 | DAC 2020 | First GNN-specific accelerator; 7.5× over GPU |
| [17] | Liang et al., *EnGN: High-Throughput GNN Accelerator*, 2021 | IEEE TC | Ring-edge-reduce dataflow; 1800× over CPU |
| [18] | Yan et al., *HyGCN: Hybrid GCN Accelerator*, 2020 | HPCA 2020 | Aggregation+combination hybrid; 1509× over CPU |
| [19] | Kiningham et al., *GRIP: GNN Accelerator Architecture*, 2023 | IEEE TC | Low-latency GNN inference; 17× over CPU |
| [20] | Abadal et al., *Computing GNNs: Algorithms to Accelerators*, 2021 | ACM CSUR | Comprehensive survey; framing memory-bound argument |
| [21] | Zhang et al., *GNN Acceleration Survey*, 2026 | ACM CSUR | Taxonomy of GNN acceleration techniques |
| [22] | Baruah et al., *GNNMark: Benchmark Suite for GNN Training*, 2021 | ISPASS 2021 | GPU-based GNN benchmark; comparison baseline |

---

## 5. Intel NPU and AI Hardware Architecture

| # | Citation | Source | Relevance |
|---|----------|--------|-----------|
| [23] | Jouppi et al., *In-Datacenter Performance Analysis of a TPU*, 2017 | ISCA 2017 | TPU as domain-specific accelerator reference |
| [24] | Intel Corp., *Heterogeneous AI Powerhouse: Intel Core Ultra NPU*, 2024 | Whitepaper | Official Meteor Lake NPU architecture (NPU 3720) |
| [25] | Lam, *Intel Meteor Lake's NPU*, 2024 | Chips & Cheese | Technical NPU microarchitecture analysis |
| [47] | Chen et al., *Eyeriss: Energy-Efficient CNN Accelerator*, 2016 | ISSCC 2016 | Reference NPU design baseline |
| [48] | Chen et al., *Eyeriss v2: Flexible DNN Accelerator*, 2019 | IEEE JETCAS | Mobile NPU design reference |

---

## 6. Edge AI and NPU Inference

| # | Citation | Venue | Key Finding |
|---|----------|-------|------------|
| [26] | Xu et al., *Fast On-device LLM Inference with NPUs*, 2025 | ASPLOS 2025 | NPU-specific operator optimization strategies |
| [27] | Gao et al., *LLM-NPU: Efficient Foundation Model Inference on NPUs*, 2025 | IEEE CS 2025 | Memory bandwidth as NPU bottleneck |
| [28] | Tummalapalli & Arayakandy, *LLM Inference at the Edge*, 2026 | arXiv 2026 | Mobile/NPU/GPU sustained load trade-offs |
| [29] | Kachris et al., *Cloud to Edge: Benchmarking LLM Inference*, 2025 | arXiv 2025 | Hardware-accelerated single-board benchmark |
| [45] | Heo et al., *IANUS: NPU-PIM Unified Memory System*, 2024 | ASPLOS 2024 | NPU-PIM co-design for memory-bound workloads |
| [57] | Singh & Gill, *Edge AI: A survey*, 2023 | IoT and Cyber-Physical Systems 2023 | Comprehensive survey of edge computing paradigms and transition to Edge AI |

---

## 7. Operator Fusion and Compiler Optimization

| # | Citation | Venue | Relevance |
|---|----------|-------|-----------|
| [30] | Niu et al., *DNNFusion: Advanced Operator Fusion*, 2021 | PLDI 2021 | Operator fusion theory; motivates FGR metric |
| [31] | Zhang et al., *Unified Operator Fusion for Heterogeneous Hardware*, 2025 | arXiv 2025 | Cross-device fusion optimization |
| [32] | Zhang et al., *Forge-UGC: Universal Graph Compiler*, 2026 | arXiv 2026 | Graph-level compiler optimization |
| [33] | Tang et al., *Parallax: Adaptive DAG Partitioning for CPU Fallbacks*, 2025 | arXiv 2025 | CPU fallback scheduling — directly related to our fallback detection |
| [46] | Shu et al., *FlashMem: DNN Workloads on Mobile GPU*, 2026 | arXiv 2026 | Memory hierarchy optimization for mobile inference |
| [35] | Pagoda et al., *RooflineBench: On-device LLM Benchmarking*, 2026 | arXiv 2026 | Roofline methodology for edge accelerators |

---

## 8. Quantization

| # | Citation | Venue | Relevance |
|---|----------|-------|-----------|
| [34] | Jacob et al., *Quantization and Training of NNs for Integer-Arithmetic Inference*, 2018 | CVPR 2018 | Foundation for INT8 PTQ (used via NNCF) |

---

## 9. GNN Scalability and Benchmarking

| # | Citation | Venue | Relevance |
|---|----------|-------|-----------|
| [36] | Shirzad et al., *Exphormer: Sparse Transformers for Graphs*, 2023 | ICML 2023 | Sparse graph transformer; comparison for graph density analysis |
| [37] | Tönshoff et al., *Where Did the Gap Go?*, 2023 | LoG 2023 | Rigorous GNN benchmarking methodology |
| [38] | Thomas et al., *GNNs Designed for Different Graph Types*, 2023 | TMLR 2023 | Survey of graph type diversity |
| [39] | Zeng et al., *GraphSAINT: Graph Sampling for Inductive Learning*, 2020 | ICLR 2020 | Sampling-based training scalability |
| [40] | Cong et al., *Minimal Variance Sampling for GNNs*, 2020 | KDD 2020 | Variance reduction in GNN training |
| [41] | Huang et al., *TC-GNN: Sparse GNN on Dense Tensor Cores*, 2023 | USENIX ATC 2023 | GPU sparse-dense bridge for GNN ops |
| [42] | Fan et al., *HGL: Heterogeneous GNN Training*, 2025 | IEEE 2025 | GNN training optimization |
| [43] | Qu et al., *TT-GNN: On-Chip GNN Training*, 2023 | MICRO 2023 | Tensor-train GNN; on-chip memory-efficient training |
| [44] | Guan et al., *DynaGraph: Dynamic GNNs at Scale*, 2022 | SIGMOD 2022 | Dynamic GNN optimization |
| [54] | Zhang et al., *Graph-less Neural Networks via Distillation*, 2022 | ICLR 2022 | MLP distillation vs GNN inference |
| [55] | Tailor et al., *Do We Need Anisotropic GNNs?*, 2022 | ICLR 2022 | Isotropic vs anisotropic GNN efficiency |

---

## 10. Lightweight and Recommendation GNNs

| # | Citation | Venue | Relevance |
|---|----------|-------|-----------|
| [51] | He et al., *LightGCN*, 2020 | SIGIR 2020 | Simplified GCN for recommendation; NPU-friendly structure |
| [52] | Cai et al., *LightGNN*, 2025 | arXiv 2025 | Ultra-lightweight GNN variant |

---

## 11. Related Work and Context

| # | Citation | Venue | Relevance |
|---|----------|-------|-----------|
| [49] | Mattson et al., *MLPerf Inference Benchmark*, 2020 | IEEE Micro 2020 | Inference benchmarking standard |
| [50] | Dhar et al., *Roadmap for Edge AI: A Dagstuhl Perspective*, 2022 | Commun. ACM | Edge AI research roadmap |
| [53] | Han et al., *EIE: Efficient Inference Engine on Compressed DNNs*, 2016 | ISCA 2016 | Sparse inference engine; NPU comparison context |
| [56] | Bayraktar, *Beyond GNNs: Feature Efficiency for Link Prediction*, 2026 | KAIS 2026 | Challenges automatic GNN preference on sparse graphs |

---

*Last updated: 1 June 2026 — Generated from `paper/references.bib` (1,179 entries).*
