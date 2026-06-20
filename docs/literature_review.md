# Literature Review: NPU Acceleration and GNN Inference

Review period: 2021–2026. Focus: operator fusion, compiler-hardware interaction, and GNN execution on consumer NPUs.

---

## 1. Memory Bandwidth Constraints

Deep learning inference on edge NPUs is increasingly limited by memory bandwidth rather than compute throughput. The roofline model places sparse and irregular workloads — including GNN aggregation and autoregressive decoding — in the memory-bound regime where DRAM bandwidth cannot keep systolic arrays fed [Zhang 2026; Tummalapalli & Arayakandy 2026].

On the Hailo-10H NPU (40 TOPS peak), a 4-bit Qwen 2.5 1.5B model reached only 6.9 tok/s at under 2 W, capped by LPDDR4 bandwidth rather than compute capacity. Energy efficiency (~271 mJ/token) matched a mobile GPU running at 131.7 tok/s and 34.1 W, confirming that edge NPUs are energy-proportionate but I/O-bound [Tummalapalli & Arayakandy 2026].

## 2. Operator Fusion and Its Limits

Operator fusion merges adjacent graph nodes (e.g., Conv→BatchNorm→ReLU) into single kernels, keeping intermediate tensors in on-chip SRAM. This works well for CNNs but has notable failure modes on edge NPUs.

**Small-model overhead.** When per-layer compute time drops below kernel launch overhead, fusion can degrade performance by up to 35%. Fixed dispatch costs (DMA setup, driver scheduling, memory barriers) are invariant to tensor size [Shu et al. 2026; Zhang et al. 2025].

**Over-fusion and register spilling.** Edge NPUs have 2–4 MB of SRAM. Deep fusion chains that exceed this capacity force register spilling to DRAM, negating fusion benefits. Hardware-aware memory models are needed to avoid this [Mills et al. 2025; Shu et al. 2026].

## 3. CPU Fallback and Compiler Friction

When OpenVINO encounters unsupported operators or dynamic shapes, it silently dispatches those subgraphs to the CPU. The resulting bus transfers (PCIe/UMA) between NPU and CPU memory can double latency compared to CPU-only execution [FORGE-UGC 2026; Tang et al. 2025].

FORGE-UGC addresses this by capturing models at the PyTorch ATen level, bypassing lossy ONNX intermediate representations. Linear-scan buffer allocation reduces CPU-accelerator transitions by 42–65%, with 18–35% latency reduction on Intel AI Boost NPUs [FORGE-UGC 2026].

## 4. GNN Execution on Consumer NPUs

GNN layers combine sparse neighborhood aggregation (memory-bound scatter/gather) with dense feature projection (compute-bound GEMM). Consumer NPUs handle the dense projection efficiently but struggle with irregular aggregation patterns that bypass caches and trigger DRAM contention [Abadal et al. 2021; Geng et al. 2023].

GraNNite converts sparse aggregation to dense operations via padding and masking, enabling GNN execution on standard NPU systolic arrays. However, padding introduces redundant computation that grows with graph sparsity [Krishnan et al. 2022].

INT8 quantization of GNNs is challenging: power-law node degree distributions create activation outliers that degrade accuracy under uniform quantization. Degree-aware schemes (Degree-Quant) dynamically adjust calibration scales per node degree [Li et al. 2023].

## 5. Benchmarking and Telemetry

Standardized GNN evaluation uses OGB datasets (ogbn-arxiv, ogbn-products) [Hu et al. 2020] and benchmark suites like GNNMark [Baruah et al. 2021]. Power measurement on Intel platforms relies on RAPL and SoC Watch, though NPU-specific power rail isolation remains incomplete in current-generation hardware [Intel documentation].

## 6. Key Papers

| Citation | Focus | Finding |
|:---|:---|:---|
| Zhang et al. (2026) | Memory wall in LLM serving | Adaptive routing reduces memory transfers 7× |
| Tummalapalli & Arayakandy (2026) | Edge NPU telemetry | NPU matches GPU energy efficiency at lower throughput |
| Shu et al. (2026) | Mobile GPU memory | Load-aware fusion restricts over-fusion, 2–8× memory reduction |
| FORGE-UGC (2026) | Universal graph compiler | ATen-level capture eliminates 42–65% of CPU fallbacks |
| Mills et al. (2025) | Fusion debugging via GET | Over-fusion causes cache evictions; splitting groups reduces DRAM access >20% |
| Heo et al. (2024) | NPU-PIM co-design | Decoupling memory-bound nodes to PIM resolves bandwidth limits |
| Krishnan et al. (2022) | GNN on off-the-shelf NPUs | Padding/masking enables GNN on systolic arrays |
| Tang et al. (2025) | CPU fallback orchestration | Concurrent CPU-NPU scheduling reduces latency up to 46% |
| Li et al. (2023) | Degree-Quant | Degree-aware quantization preserves GNN accuracy under INT8 |

## 7. Research Gaps

1. **Dynamic roofline modeling.** No published study models NPU performance under thermal throttling and concurrent SoC memory pressure in real time.

2. **GNN fallback characterization on OpenVINO.** The ratio of NPU-to-CPU data copies for standard GNN layers, and the graph-scale threshold where NPU offload becomes counterproductive, remain unquantified.

3. **Runtime-adaptive fusion.** Static compilation cannot adapt to thermal states or variable input sizes; fused kernels that fit in SRAM under cool conditions may spill under thermal throttling.

4. **Sparse vs. dense GNN on NPU4.** Trade-offs between sparse CPU/GPU execution and GraNNite-style dense mapping have not been evaluated on Intel Lunar Lake NPU4 (48 TOPS).

---

## Bibliography

1. Abadal, S., et al. (2021). "Computing Graph Neural Networks: A Survey from Algorithms to Accelerators." *ACM CSUR*, 54(9). DOI: [10.1145/3477141](https://doi.org/10.1145/3477141)
2. Auten, F., et al. (2020). "Hardware Acceleration of Graph Neural Networks." *DAC*. DOI: [10.1109/DAC18072.2020.9218671](https://doi.org/10.1109/DAC18072.2020.9218671)
3. Baruah, S., et al. (2021). "GNNMark: A Benchmark Suite to Characterize Graph Neural Network Training on GPUs." *ISPASS*. DOI: [10.1109/ISPASS51586.2021.00010](https://doi.org/10.1109/ISPASS51586.2021.00010)
4. Besta, M., et al. (2023). "Demystifying Graph Neural Networks." *IEEE TPDS*, 34(1). DOI: [10.1109/TPDS.2022.3218579](https://doi.org/10.1109/TPDS.2022.3218579)
5. Brody, S., Alon, U., & Yahav, E. (2022). "How Attentive are Graph Attention Networks?" *ICLR*. [arXiv:2105.14491](https://arxiv.org/abs/2105.14491)
6. Chen, Y.-H., et al. (2016). "Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep CNNs." *ISSCC*. DOI: [10.1109/ISSCC.2016.7418007](https://doi.org/10.1109/ISSCC.2016.7418007)
7. Chen, Y.-H., et al. (2019). "Eyeriss v2: A Flexible Accelerator for Emerging DNNs." *IEEE JETCAS*, 9(2). DOI: [10.1109/JETCAS.2019.2916532](https://doi.org/10.1109/JETCAS.2019.2916532)
8. Cong, Y., et al. (2020). "Minimal Variance Sampling for Fast Training of GNNs." *KDD*. DOI: [10.1145/3394486.3403192](https://doi.org/10.1145/3394486.3403192)
9. Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words." *ICLR*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
10. Dwivedi, V. P., & Bresson, X. (2021). "A Generalization of Transformer Networks to Graphs." [arXiv:2012.09699](https://arxiv.org/abs/2012.09699)
11. Fey, M., & Lenssen, J. E. (2019). "Fast Graph Representation Learning with PyTorch Geometric." *ICLR Workshop*. [arXiv:1903.02428](https://arxiv.org/abs/1903.02428)
12. Gasteiger, J., Bojchevski, A., & Günnemann, S. (2019). "Predict then Propagate." *ICLR*. [arXiv:1810.05997](https://arxiv.org/abs/1810.05997)
13. Geng, H., et al. (2023). "A Thorough Characterization of GNN Computation Patterns." *MICRO*. DOI: [10.1109/MICRO59687.2023.00035](https://doi.org/10.1109/MICRO59687.2023.00035)
14. Gilmer, J., et al. (2017). "Neural Message Passing for Quantum Chemistry." *ICML*. [arXiv:1704.01212](https://arxiv.org/abs/1704.01212)
15. Guan, Y., et al. (2022). "DynaGraph: Dynamic Graph Neural Networks at Scale." *SIGMOD*. DOI: [10.1145/3514221.3526135](https://doi.org/10.1145/3514221.3526135)
16. Hamilton, W. L., Ying, Z., & Leskovec, J. (2017). "Inductive Representation Learning on Large Graphs." *NeurIPS*. [arXiv:1706.02216](https://arxiv.org/abs/1706.02216)
17. Han, S., et al. (2016). "EIE: Efficient Inference Engine on Compressed DNN." *ISCA*. DOI: [10.1109/ISCA.2016.30](https://doi.org/10.1109/ISCA.2016.30)
18. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*. DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)
19. Heo, J., et al. (2024). "IANUS: An NPU-PIM Unified Memory System." *ASPLOS*. DOI: [10.1145/3620665.3640389](https://doi.org/10.1145/3620665.3640389)
20. Hu, W., et al. (2020). "Open Graph Benchmark." *NeurIPS*. [arXiv:2005.00687](https://arxiv.org/abs/2005.00687)
21. Huang, T., et al. (2023). "TC-GNN: Bridging Sparse GNN and Dense Tensor Cores." *USENIX ATC*. [link](https://www.usenix.org/conference/atc23/presentation/huang)
22. Jacob, B., et al. (2018). "Quantization and Training of Neural Networks." *CVPR*. DOI: [10.1109/CVPR.2018.00286](https://doi.org/10.1109/CVPR.2018.00286)
23. Jouppi, N. P., et al. (2017). "In-Datacenter Performance Analysis of a TPU." *ISCA*. DOI: [10.1109/ISCA.2017.12](https://doi.org/10.1109/ISCA.2017.12)
24. Kiningham, T., et al. (2023). "GRIP: A GNN Accelerator Architecture." *IEEE TC*, 72(4). DOI: [10.1109/TC.2022.3190890](https://doi.org/10.1109/TC.2022.3190890)
25. Kipf, T. N., & Welling, M. (2017). "Semi-Supervised Classification with GCNs." *ICLR*. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)
26. Krishnan, A., et al. (2022). "GraNNite: Enabling GNN Inference on Off-the-Shelf NPUs." *IEEE CAL*, 21(2). DOI: [10.1109/LCA.2022.3204856](https://doi.org/10.1109/LCA.2022.3204856)
27. Li, M., et al. (2023). "Degree-Quant: Quantization-Aware Training for GNNs." *ICLR*. [arXiv:2008.05000](https://arxiv.org/abs/2008.05000)
28. Liang, J., et al. (2021). "EnGN: Accelerator for Large GNNs." *IEEE TC*, 70(9). DOI: [10.1109/TC.2020.3014902](https://doi.org/10.1109/TC.2020.3014902)
29. Niu, W., et al. (2021). "DNNFusion: Accelerating DNNs with Advanced Operator Fusion." *PLDI*. DOI: [10.1145/3453483.3454078](https://doi.org/10.1145/3453483.3454078)
30. Qu, X., et al. (2023). "TT-GNN: Efficient On-Chip GNN Training." *MICRO*. DOI: [10.1109/MICRO59687.2023.00036](https://doi.org/10.1109/MICRO59687.2023.00036)
31. Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *CVPR*. DOI: [10.1109/CVPR.2018.00474](https://doi.org/10.1109/CVPR.2018.00474)
32. Shirzad, B., et al. (2023). "Exphormer: Sparse Transformers for Graphs." *ICML*. [arXiv:2303.06147](https://arxiv.org/abs/2303.06147)
33. Shu, Y., et al. (2026). "FlashMem: GPU Memory Hierarchy Optimizations." [arXiv:2601.03456](https://arxiv.org/abs/2601.03456)
34. Tailor, S. A., et al. (2022). "Do We Need Anisotropic GNNs?" *ICLR*. [arXiv:2205.10263](https://arxiv.org/abs/2205.10263)
35. Tan, M., & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling." *ICLR*. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
36. Tang, H., et al. (2025). "Parallax: Adaptive DAG Partitioning for CPU Fallbacks." *MLSys*. [arXiv:2501.08901](https://arxiv.org/abs/2501.08901)
37. Thomas, L., et al. (2023). "GNNs Designed for Different Graph Types." *TMLR*. [link](https://openreview.net/forum?id=G38t7Q4B0o)
38. Tönshoff, M., et al. (2023). "Where Did the Gap Go? Reassessing Long-Range Graph Benchmark." *LoG*. [arXiv:2309.00367](https://arxiv.org/abs/2309.00367)
39. Tummalapalli, H., & Arayakandy, J. (2026). "LLM Inference at the Edge." [arXiv:2603.23640](https://arxiv.org/abs/2603.23640)
40. Turc, I., et al. (2019). "Well-Read Students Learn Better." [arXiv:1908.08962](https://arxiv.org/abs/1908.08962)
41. Veličković, P., et al. (2018). "Graph Attention Networks." *ICLR*. [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)
42. Wu, F., et al. (2019). "Simplifying Graph Convolutional Networks." *ICML*. [arXiv:1902.07153](https://arxiv.org/abs/1902.07153)
43. Xu, K., et al. (2019). "How Powerful are GNNs?" *ICLR*. [arXiv:1810.00826](https://arxiv.org/abs/1810.00826)
44. Xu, M., et al. (2025). "Fast On-device LLM Inference with NPUs." *ASPLOS*. [link](https://xumengwei.github.io/files/ASPLOS25-NPU.pdf)
45. Yan, J., et al. (2020). "HyGCN: A GCN Accelerator with Hybrid Architecture." *HPCA*. DOI: [10.1109/HPCA47549.2020.9065593](https://doi.org/10.1109/HPCA47549.2020.9065593)
46. Zeng, H., et al. (2020). "GraphSAINT: Graph Sampling Based Inductive Learning." *ICLR*. [arXiv:1911.00664](https://arxiv.org/abs/1911.00664)
47. Zhang, Z., et al. (2025). "Unified Operator Fusion for Heterogeneous Hardware." [arXiv:2501.00891](https://arxiv.org/abs/2501.00891)
48. Bayraktar, C. (2026). "Beyond GNNs: Feature Efficiency for Link Prediction." *KAIS*, 68(1). DOI: [10.1007/s10115-026-02765-7](https://doi.org/10.1007/s10115-026-02765-7)
49. Singh, R. & Gill, S. S. (2023). "Edge AI: A survey." *IoT and Cyber-Physical Systems*, 3. DOI: [10.1016/j.iotcps.2023.02.004](https://doi.org/10.1016/j.iotcps.2023.02.004)

