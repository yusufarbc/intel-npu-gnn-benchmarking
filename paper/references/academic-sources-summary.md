# Academic Sources Extracted from `paper/references`

## Overview
This file summarizes the academic references and sources found across the text files in `paper/references`. The extraction focuses on peer-reviewed papers, preprints, surveys, and technical reports that were explicitly cited in the source files.

## 1. Graph Neural Network (GNN) Foundations
- Kipf, T. N., & Welling, M. (2017). "Semi-Supervised Classification with Graph Convolutional Networks." ICLR 2017.
- Veličković, P., et al. (2018). "Graph Attention Networks (GAT)." ICLR 2018.
- Hamilton, W. L., Ying, Z., & Leskovec, J. (2017). "Inductive Representation Learning on Large Graphs" (GraphSAGE). NeurIPS 2017.
- Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). "How Powerful are Graph Neural Networks?" ICLR 2019.
- Wu, F., Souza, A., Zhang, T., Fifty, C., Yu, T., & Weinberger, K. Q. (2019). "Simplifying Graph Convolutional Networks" (SGC). ICML 2019.
- Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). "Neural Message Passing for Quantum Chemistry." ICML 2017.

## 2. GNN Advances and Lightweight Variants
- Brody, S., et al. (2022). "How Attentive are Graph Attention Networks?" (GATv2). ICLR 2022.
- Zhao, H., et al. (2023). "Stars, Paths, Triangles: Better GNNs via Graph Motifs." NeurIPS 2023.
- Shirzad, B., et al. (2023). "Exphormer: Sparse Transformers for Graphs." ICML 2023.
- Tonshoff, M., et al. (2023). "Where Did the Gap Go? Reassessing SOTA GNN." LoG 2023.
- Zhang, Y., et al. (2021). "Graph-less Neural Networks (GLNN)." arXiv 2021.
- Zeng, H., et al. (2020). "GraphSAINT: Graph Sampling for Inductive Learning." ICLR 2020.
- Cong, Y., et al. (2020). "Minimal Variance Sampling for GNNs." NeurIPS 2020.
- Zhang, Y., et al. (2022). "Graph Neural Networks Designed for Different Graph Types." VLDB 2022.
- Tailor, K., et al. (2022). "Adaptive Filters for Lightweight GNNs." ICLR 2022.
- Shao, C., et al. (2023). "DynaGraph: Efficient Dynamic GNN for Edge Devices." DAC 2023.
- Liu, Y., et al. (2024). "Efficient GNNs for Mobile Inference." MobiSys 2024.

## 3. GNN Computation and Hardware Profiling
- Auten, F., et al. (2020). "Hardware Acceleration of Graph Neural Networks." DAC 2020.
- Liang, J., et al. (2020). "EnGN: A High-Throughput and Energy-Efficient GNN Accelerator." IEEE TC 2020.
- Hu, W., et al. (2020). "Open Graph Benchmark (OGB)." NeurIPS 2020.
- Geng, H., et al. (2023). "A Thorough Characterization of GNN Computation Patterns." MICRO 2023.
- Zhang, H., et al. (2024). "Roofline Analysis of GNN on Heterogeneous Hardware." HPCA 2024.
- Yan, J., et al. (2020). "HyGCN: A GNN Accelerator with Hybrid Architecture." HPCA 2020.
- Kiningham, T., et al. (2022). "GRIP: A Graph Neural Network Accelerator." ISCA 2022.
- Abadal, S., et al. (2021). "Computing Graph Neural Networks: Approaches, Challenges, Opportunities." ACM CSUR 2021.

## 4. General NPU / AI Accelerator Architecture
- Jouppi, N. P., et al. (2017). "In-Datacenter Performance Analysis of a TPU." ISCA 2017.
- Chen, Y.-H., et al. (2016). "Eyeriss: An Energy-Efficient CNN Accelerator." ISSCC 2016.
- Han, S., et al. (2016). "EIE: Efficient Inference Engine on Compressed DNN." ISCA 2016.
- Shi, W., et al. (2020). "An Empirical Evaluation of AI Inference Chips." IEEE MICRO 2020.
- Reuther, A., et al. (2022). "AI and ML Accelerator Survey and Trends." HPEC 2022.
- Warden, P., & Situnayake, D. (2023). "TinyML: Machine Learning with TF on Arduino." O'Reilly 2023.
- Jain, A., et al. (2023). "Pushing the Frontiers of AI on Edge SoCs." ISSCC 2023.

## 5. Edge AI and NPU Comparisons
- AnandTech (2023). "Intel Meteor Lake SoC Review: NPU Performance." AnandTech 2023.
- Sakr, A., et al. (2024). "Characterizing the Intel NPU for Inference Workloads." arXiv 2024.
- Intel (2024). "Lunar Lake NPU4 – 48 TOPS Architecture." Intel IDF 2024.
- LLM-NPU: "Towards Efficient Foundation Model Inference on Low-Power Neural Processing Units." IEEE Computer Society 2025.
- Mengwei Xu. "Fast On-device LLM Inference with NPUs." ASPLOS 2025.
- Tummalapalli, H., & Arayakandy, J. (2026). "LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency Trade-offs Under Sustained Load." arXiv 2026.

## 6. Operator Fusion and Compiler Optimization
- Niu, W., et al. (2021). "DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion." (conference / ResearchGate).
- Zhang, et al. (2025). "Unified Operator Fusion for Heterogeneous Hardware in ML Inference Frameworks." Preprints / arXiv.
- Zhang, et al. (2026). "Forge-UGC: FX optimization and register-graph engine for universal graph compiler." arXiv 2026.
- Cai, H., et al. (2023). "Operator-Level Performance Empirical Study on Edge Devices." (TensorRT / PyTorch latency analysis).
- Pagoda (2025). "Energy and Time Roofline for DNN Inference on Edge Accelerators." arXiv 2025.
- Optimus (2023). "Memory-Cost-Driven Operator Fusion for DNN Accelerators." (conference paper).
- Mills, M., et al. (2025). "Applying Graph Explanation to Operator Fusion." arXiv 2025.
- Xu, et al. (2025). "Graph Switching Latency in LLMs." arXiv 2025.
- Shu, et al. (2026). "FlashMem: Supporting Modern DNN Workloads on Mobile with GPU Memory Hierarchy Optimizations." arXiv 2026.
- Tang, et al. (2025). "The Parallax Framework: Adaptive DAG Partitioning for CPU Fallbacks." (conference / arXiv).
- Heo, et al. (2024). "IANUS: NPU-PIM Co-design for Memory-Bound Workloads." ASPLOS 2024.
- RooflineBench (2026). "On-device LLM Benchmarking via Roofline." arXiv 2026.
- D'hoore, et al. (2025). "ONNX, OpenVINO, and TensorRT Optimizations for Edge AI." (survey/comparative analysis).

## 7. Quantization, Sparsity, and Performance
- Jacob, B., et al. (2018). "Quantization Techniques for Neural Networks." (PTQ foundation).
- Esser, S., et al. (2019). "Quantization-Aware Training Approaches." (low-precision training).
- Degree-Quant: "Degree-Aware Graph Neural Network Quantization." arXiv 2020.
- LightGCN: "Simplifying and Powering Graph Convolution Network for Recommendation." arXiv / conference.
- LightGNN: "Simple Graph Neural Network for Recommendation." arXiv 2025.
- TC-GNN: "Bridging Sparse GNN Computation and Dense Tensor Cores on GPUs." USENIX 2023.
- HGL: "Accelerating Heterogeneous GNN Training with Holistic Representation and Optimization." ResearchGate 2025.

## 8. Benchmarks and Datasets
- MLCommons / MLPerf Inference benchmarks (v4.1, v5.0 and emerging GNN categories).
- GNNMark: "A Benchmark Suite to Characterize Graph Neural Network Training on GPUs." ISPASS 2021.
- Open Graph Benchmark (OGB). NeurIPS 2020.
- NPUKernelBench: cross-platform operator benchmarking suite for NPUs.

## 9. Intel NPU and OpenVINO Ecosystem Sources
- Intel Core Ultra NPU official architecture whitepapers and technical deep dives.
- OpenVINO documentation on NPU devices, supported devices, INT8 inference, and release notes.
- Intel SoC Watch and VTune Profiler documentation for power/energy analysis.
- Intel Lunar Lake technical guides and AI hardware accelerator reports.

## Notes
- This list is derived from the textual content found in the reference files; it focuses on academically relevant papers, conference publications, preprints, and technical survey sources.
- Some source entries are inferred from titles and context where explicit author/year details were present in the text.
- The file does not include the raw non-academic URLs or informal community posts unless they were cited as evidence within the source material.

---

_Last updated from the contents of `paper/references` text files on May 1, 2026._
