# Slide 1
Good afternoon. My name is Yusuf Talha Arabacı, and today I will present our study, Benchmarking GNN Inference on the Intel Core Ultra NPU: A Latency, Quantization, and Energy Analysis. This work was conducted alongside my colleagues Emrullah Demiral and Ömer Faruk Acar at Karabük University. We investigate how consumer-grade Neural Processing Units handle Graph Neural Networks, which present highly irregular and memory-bound workloads, and contrast their performance against standard CPU and integrated GPU backends.

# Slide 2
The presentation is structured into five main parts. I will begin with the architectural motivation behind our study, followed by our experimental methodology, which evaluates fourteen models across three datasets. Next, I will present the empirical results concerning latency, quantization trade-offs, and energy footprint. We will then analyze the underlying hardware and compiler bottlenecks before concluding with practical deployment recommendations for edge devices.

# Slide 3
The primary motivation for this work is the structural mismatch between commodity AI accelerators and graph workloads. While consumer NPUs efficiently accelerate dense tensor operations such as grid-based convolutions, Graph Neural Networks rely on sparse, non-Euclidean structures. The fundamental computation, Sparse-Dense Matrix Multiplication, requires non-sequential memory accesses and indirect indexing. Consequently, NPUs frequently experience pipeline stalls while waiting for data from main memory, leading to poor hardware utilization.

# Slide 4
We address three main research questions. First, how do consumer NPUs compare to CPUs and iGPUs in executing sparse GNN inference? Second, does reduced-precision INT8 quantization yield the expected performance scaling on NPUs, or does it introduce latency regressions? Third, how effectively does the OpenVINO compiler lower irregular operators, such as Gather and Scatter, onto the NPU microarchitecture?

# Slide 5
To visualize the problem, we compare the dataflow of convolutional networks with that of GNNs. Convolutional layers benefit from high spatial locality and regular memory access, allowing the compiler to maximize on-chip SRAM reuse. This represents a compute-bound workload where NPUs excel. Conversely, GNNs aggregate features across arbitrary graph topologies, triggering irregular memory strides. This shifts the bottleneck from compute units to memory bandwidth, subjecting the workload to the DRAM latency wall.

# Slide 6
Our experimental platform consists of an Intel Core Ultra 5 125H processor with 16 gigabytes of LPDDR5x memory. We evaluate three execution backends on this SoC: the 14-core CPU, the integrated Arc GPU, and the AI Boost NPU. The software stack is built on OpenVINO 2024.1 and ONNX Runtime 1.18. To ensure measurement reliability, each model runs for five warm-up iterations followed by one hundred timed iterations, repeated across three separate sessions. System power is monitored using Intel SoCWatch.

# Slide 7
Our benchmark suite spans fourteen models, including nine GNNs representing spectral, spatial, attentional, and propagation-based paradigms. We include standard CNN and Transformer architectures as baselines. These models are evaluated on three datasets from the Open Graph Benchmark: ogbn-arxiv, which has low density; ogbn-products, representing medium density; and ogbn-proteins, which represents a dense graph topology with an average of four hundred and fifty-one edges per node.

# Slide 8
The FP32 latency results reveal a clear performance split. For dense baselines, the NPU provides substantial acceleration, achieving up to an eleven-fold speedup over the CPU for Vision Transformers. For GNNs, however, NPU execution is roughly equivalent to CPU execution, remaining within a six percent performance margin. The integrated GPU consistently yields the lowest latency for GNN workloads. We also note that GraphSAGE performs poorly on the NPU due to long chains of serialized Gather and Scatter operations.

# Slide 9
The evaluation of INT8 quantization shows counter-intuitive outcomes on the NPU. Most GNNs demonstrate minimal scaling, typically between three and five percent. In the case of the Simple Graph Convolution model, quantization causes a two-point-two-fold latency regression due to execution dispatch overhead. Attentional architectures fail compilation entirely. Additionally, we observed silent CPU fallback where quantized baselines, such as MobileNetV2, revert to CPU execution without compiler warnings.

# Slide 10
The impact of memory regularity is clearly shown when comparing the Vision Transformer with the Graph Transformer. The Vision Transformer operates on a static grid, enabling the compiler to apply kernel fusion passes and achieve high NPU throughput. Conversely, the Graph Transformer calculates attention over dynamic, non-contiguous graph neighborhoods. This prevents compiler-driven fusion and results in heavy DRAM traffic. Consequently, the Graph Transformer achieves no acceleration on the NPU despite having significantly fewer parameters.

# Slide 11
The roofline analysis confirms these structural limitations. GNNs cluster in the memory-bound region, characterized by low arithmetic intensity—typically between zero-point-one and ten FLOPs per byte. Their performance is restricted by DRAM bandwidth rather than arithmetic throughput. In contrast, dense convolutional models operate at higher operational intensity, allowing them to utilize the parallel compute arrays of the NPU and approach the platform's peak execution ceiling.

# Slide 12
A key characteristic of the NPU execution model is its response to graph scaling. While CPU and GPU latencies scale with graph size and density, NPU latency remains constant. The OpenVINO toolchain compiles GNN graphs with static shapes, fixing the tensor dimensions during compilation. Consequently, the execution time is decoupled from graph sparsity, showing no statistical correlation with input size. This provides predictable latency but prevents the backend from exploiting sparsity to accelerate inference.

# Slide 13
Telemetry data from SoCWatch indicates that the integrated GPU draws approximately seven percent more power than the CPU during GCN execution, but achieves comparable energy-per-inference due to reduced latency. For quantized models, results are highly model-dependent. On the CPU, INT8 quantization reduces energy consumption by eighteen percent for GCN, but increases energy consumption by fifty-nine percent for MPNN due to a severe latency penalty. This underscores that precision reduction does not guarantee energy efficiency on sparse workloads.

# Slide 14
In summary, the limitations of consumer NPUs on GNNs are driven by the memory wall of irregular accesses, compiler constraints that prevent kernel fusion, and limited operator coverage. For instance, the Message Passing Neural Network triggers complete fallback to the CPU because operators like index-add are unsupported on the NPU. This incomplete coverage leads to silent fallback failures where the toolchain routes execution back to the CPU, masking hardware-level execution bottlenecks.

# Slide 15
This study has several limitations. Our evaluation is restricted to a single hardware platform. Furthermore, due to the system telemetry definitions of the Meteor Lake SoC, direct power measurement of the NPU rail was not possible, requiring the use of package-level estimates. Additionally, our energy calculations include background operating system activity, serving as an upper bound. Finally, because compiler toolchains evolve, future updates may expand operator coverage and improve these baselines.

# Slide 16
We propose the following deployment recommendations. For dense vision workloads, the NPU should be targeted at FP32 to leverage its parallel hardware capability, while avoiding INT8 to prevent silent CPU fallback. For Graph Neural Networks, the integrated GPU is the most effective target. Its larger cache size and superior effective memory bandwidth allow it to mitigate GNN memory bottlenecks far more effectively than the NPU under the current software stack.

# Slide 17
In conclusion, consumer NPUs are highly effective for dense vision and structured attention patterns at FP32 precision. However, they yield no performance benefit for GNNs due to irregular data paths and quantization limitations. For these sparse, memory-bound workloads, the integrated GPU remains the optimal deployment backend. Furthermore, system developers must carefully monitor toolchain behavior to identify silent fallbacks and static-shape constraints.

# Slide 18
This slide outlines the primary references for our research, spanning the Meteor Lake architecture, custom graph accelerators such as HyGCN and EnGN, and the Open Graph Benchmark datasets.

# Slide 19
Thank you for your attention. The benchmark suite, dataset configurations, and telemetry logs are open-source and available on GitHub via the QR code on the slide. I am now available to address any questions you may have.
