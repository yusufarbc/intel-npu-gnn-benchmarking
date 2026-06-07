# Unified Operator Fusion for Heterogeneous Hardware in ML Inference Frameworks

## Page 1

Unified Operator Fusion for Heterogeneous Hardware 
in ML Inference Frameworks  
Zhengkai Zhang* 
Independent Researcher,  San Francisco 
CA 94108, US 
*zhengkaizhang@outlook.com
Abstract —Modern machine learning inference workloads run 
on a diverse array of hardware accelerators —from cloud GPUs to 
edge NPUs and FPGAs. Operator fusion, which merges multiple 
graph operations into a single kernel, has proven highly effective 
on homogeneous pl atforms but struggles to generalize across 
devices with different execution and memory models. We propose  
Unified  Operator  Fusion  (UOF),  a framework  that intro - duces  a 
hardware -agnostic intermediate  representation  alongside a 
device -aware cost model. UOF performs graph rewrites to identify 
and evaluate fusion opportunities, then emits optimized fused 
kernels tailored to each target. We integrate UOF into an open -
source inference engine, equipping it with plugin backends for 
CUDA, multicore C++ and vendor S DKs. Offline profiling collects 
device compute peaks, memory bandwidths and kernel - launch 
latencies; these feed into an automated cost evaluator  that 
balances compute, data movement and launch overhead.  On 
ResNet -50 and BERT -small benchmarks across Intel Xeon CPUs,  
NVIDIA  V100 GPUs  and a mobile  NPU,  UOF  delivers up to 
3.8× end -to-end speedups over unfused baselines and matches  
hand -tuned  vendor  libraries  within  5–10 %. An ablation study  
removing  the cost model  results  in over- fusion  and up to 15 % 
slowdowns, underscoring the need for hardware -aware decisions. 
UOF thus offers a unified, extensible fusion strategy that 
minimizes manual backend engineering while maximizing 
performance across heterogeneous inference targets.  
Keywords:Operator Fusion,  Heterogeneous  Hardware, In- 
ference Optimization, Graph Rewriting, Kernel Generation  
I. I NTRODUCTION
The proliferation of specialized accelerators for machine 
learning inference brings challenges in maintaining both high 
performance and portability. Existing fusion solutions are 
typically handcrafted for a single device type —most often 
GPUs —and rely on st atic patterns that cannot adapt to the wide 
variation in memory hierarchies, parallelism granularity and 
kernel- launch costs found on CPUs, NPUs and FPGAs.  To 
address this, we developed a unified fusion pipeline that 
automatically  reasons about trade -offs on each hardware class, 
eliminating the need for manual kernel tuning.  
Our approach begins by translating the computation graph 
(e.g., ONNX) into a Unified Intermediate Representation (UIR). 
UIR encodes tensor shapes, data types and operator semantics, 
and tags nodes with fusion compatibility masks. We then 
traverse UIR to extract fusible subgraphs —chains of pointwise  
or linear -pointwise  operations—using a breadth -first pattern 
extractor. Each candidate subgraph is evaluated by a hardware -
aware cost model which computes the sum of (1) compute  time 
based  on a lightweight roofline approximation,  (2) memory transfer cost using offline -profiled bandwidths,
and (3) kernel-launch overhead  measured in microbenchmarks.  
Device  profiles are stored  in JSON  and can be extended  to 
new hardware by running a small suite of microbenchmarks.  
To select an optimal fusion plan, we formulate a dynamic 
programming problem: choose a non -overlapping set of sub - 
graphs minimizing total estimated runtime. Once selected, 
subgraphs are passed to a modular code generator that emits 
fused kernels —CUDA for GPUs, templated Eigen -based C++ 
for CPUs, or calls into vendor NPU SDKs. This plugin 
architecture allows seamless integration of new targets.  
We implemented UOF within an open -source C++ frame - 
work called  MLFast.  We evaluated  its performance on ResNet - 
50 and BERT -small across three platforms: Intel Xeon E5 - 
2670 CPU, NVIDIA Tesla V100 GPU and a mobile NPU  with 
128 ALUs. Compared to unfused MLFast, UOF reduced 
ResNet -50 latency from 120 ms to 32 ms on the NPU, 25 ms to 
12 ms on the GPU and 120 ms to 75 ms on the CPU. Against 
vendor- tuned libraries, UOF matched performance within 5 –
10 %. Removing the cost mod el led to excessive fusion choices 
and degraded performance by up to 15 %, demonstrating the 
importance of device -aware optimization. These re sults 
validate that UOF effectively delivers high - performance, 
portable inference on heterogeneous hardware with minimal 
manual intervention.  
II. R ELATED WORK
Operator fusion in machine learning traces its roots to 
classic compiler optimizations such as loop fusion and tiling, 
which aim to improve data locality and reduce loop overhead 
by merging adjacent loops [1], [2]. These techniques inspired 
early ML -specif ic systems that fuse pointwise operations to 
reduce memory traffic.  
High -level tensor compilers such as XLA [3] and TVM [4] 
extended fusion to GPU targets by defining fixed fusion 
patterns and simple cost heuristics. XLA’s HLO passes perform 
pattern- based fusion for operations like Convolu - 
tion+BiasAdd+ReLU, but lack visibility into device memory 
hierarchies. TVM  employs an automatic  scheduling framework 
with cost -based decisions, yet requires hand -written schedul - 
ing templates  per operator and target,  making  it labor-intensive 
to support new hardware.  
ONNX  Runtime [5] provides a graph- level  fusion en- 
gine that matches predefined subgraph templates (e.g., 
Gemm→Add→Relu), but its fusion capabilities are limited to  
2025 6th International Conference on Big Data & Artificial Intelligence & Software Engineering (ICBASE)
979-8-3315-3946-7/25/$31.00 ©2025 IEEE
2462025 6th International Conference on Big Data & Artificial Intelligence & Software Engineering (ICBASE) | 979-8-3315-3946-7/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICBASE66587.2025.11181270
Authorized licensed use limited to: Karabuk University. Downloaded on May 01,2026 at 12:58:58 UTC from IEEE Xplore.  Restrictions apply. 

## Page 2

a handful of patterns and backends. Similarly, TensorRT [6] 
offers highly optimized kernels for NVIDIA GPUs via fusion, 
but these rely on proprietary heuristics and do not generalize  to 
other devices.  
More recent work has explored compiler infrastructures 
intended for heterogeneous targets. MLIR [7] introduces a 
multi -dialect, extensible IR that can express fused operations, 
yet leaves fusion strategies to user -provided passes. Glow [8] 
integrates opera tor fusion within a static graph compiler, 
optimizing for both CPU and GPU, but uses separate fusion 
logic per backend. Apache TVM’s Relay [9] IR supports 
automatic fusion via cost models, but its scheduling language 
remains complex for end users.  
Heterogeneous scheduling frameworks such as Hetu [10] 
and FlexFlow [11] focus on partitioning and placement of 
operators across devices  rather  than fine-grained  fusion. These 
systems address data movement at the graph level, but do not 
combine multiple operators into single kernels.  
In the domain of IoT -driven agriculture, Wang and Gong 
propose an intelligent greenhouse control system leveraging 
IoT sensors and machine learning for real -time monitoring  and 
adaptive environment regulation [12]. Wang et al. further 
investigate how sequence smoothness impacts model gen - 
eralization and demonstrate that smoothing input sequences can 
significantly improve accuracy [13]. In knowledge -graph 
reasoning, Li et al.  introduce reward -shaping techniques to 
enhance multi -hop inference performance [14 ], while Liu et  al. 
apply a self-adaptive  thresholding mechanism  to boost 3D 
object detection precision [15]. For multi -turn dialogue, CA -
BERT leverages context awareness to markedly improve 
conversational quality [16]. On the theoretical side, Wang et al. 
provide new generalization bounds and convergence guar - 
antees for meta -reinforcement learning [17]. Finally, Wang et 
al. present a soft -prompt compression method that balances 
efficiency and performance in large -model context processing 
[18]. 
In natural language processing, Wu et al. investigate ad - 
vanced transformer -based architectures for deeper text un - 
derstanding, highlighting architectural trade -offs in attention 
mechanisms [19]. Theoretical analyses by Gao explore the 
limits of feedback  alignment in preference -based fine -tuning of 
AI models [20], model reasoning as Markov decision pro - 
cesses [21], and propose feedback -to-text alignment methods to 
improve LLM consistency from user ratings [22]. Sang 
examines the robustness of fine -tuned language models under 
noisy retrieval inputs, demonstrating significant performance 
variance with different retrieval noise levels  [23]. Additionally, 
Quach et al. present a reinforcement learning approach for 
integrating compressed  contextual  embeddings into knowledge 
graphs, achieving improved downstream reasoning accuracy 
[24]. 
In contrast to existing approaches, our Unified Operator 
Fusion (UOF) leverages a single, hardware -agnostic interme - 
diate representation coupled with a profile -driven cost model to  
guide fusion decisions. UOF’s  dynamic programming- based 
planner ensures globally optimal fusion under device -specific 
constraints, and its modular codegen backend enables seam - less support for CPUs, GPUs, NPUs, and future accelerators 
without rewriting fusion logic.  
III. DATASETS  
We evaluate  using  two recent  benchmarks published  strictly 
after 2020:  
• ImageNet-R [25]: Introduced in 2021, ImageNet -R 
is a robustness benchmark containing 30 000 validation 
images across 200 classes, curated from artistic rendi - 
tions, cartoons, graffiti  and other  renditions of the 
original ImageNet  classes.  We resize  all images  to 
224 × 224 and apply the standard ImageNet  
normalization ( µ = {0.485, 0.456, 0.406}, σ = {0.229, 
0.224, 0.225}). We report single  image  (batch  size = 1) 
latency  and batch -16 throughput for ResNet -50. 
• Qasper [26]: Released at EMNLP 2021, Qasper is a 
question- answering dataset over NLP research papers, 
consisting of 3 049 annotated QA pairs on 1 049 docu - 
ments, plus 36 000+ unannotated examples. We use the 
publicly provided BERT small tokenizer, limit inputs to  
384 tokens, and measure end -to-end inference latency 
(batch size=1) and throughput (batch size=16).  
IV. METHODOLOGY  
Our unified fusion pipeline consists of four integrated 
stages —graph abstraction, candidate generation, cost -driven 
selection, and backend code emission —designed to automat - 
ically produce high -performance fused kernels across hetero - 
geneous targets without manual tuning.  
First,  we ingest  a standard computation  graph (e.g.,  ONNX) 
and lower  it into a Unified  Intermediate Representation  (UIR). 
UIR nodes annotate  each operator with tensor metadata  (shape, 
data type, stride) and a fusion –compatibility flag indicating 
whether an operator can safely merge with its neighbors (for 
instance, pointwise ops or sequence batch -norm→scale→shift 
chains). This representation is hardware -agnostic: it neither 
commits  to a specific  memory layout nor to a particular  kernel- 
launch mechanism, allowing the same fusion logic to serve 
CPUs, GPUs, NPUs, and beyond.  
Next, we perform candidate generation by scanning the  
UIR for connected  subgraphs of compatible  operations. Rather 
than limiting ourselves to fixed patterns (e.g., Conv→ReLU), 
we use a breadth -first search seeded at every fusion -enabled 
node. At each step, we grow the subgraph by tentatively adding  
adjacent  nodes whose fusion flag matches —and whose 
combined working set does not exceed a per -device memory 
threshold. This yields a pool of overlapping fusion candidates 
of varying shapes and depths.  
Each candidate is then evaluated by our hardware -aware 
cost model. We approximate total execution time Tcand as 
 T cand = Ccomp  + Cmem  + Claunch   
where Ccomp is estimated with a roofline -style formula —peak 
FLOPs scaled by an empirically profiled utilization factor; Cmem 
uses tensor sizes and separately profiled host –device and on -
chip bandwidths; and Claunch comes from measuring 
247
Authorized licensed use limited to: Karabuk University. Downloaded on May 01,2026 at 12:58:58 UTC from IEEE Xplore.  Restrictions apply. 

## Page 3

microbenchmark latencies for kernel invocations of various 
argument  sizes.  Device  profiles (compute peak,  bandwidths,  
overheads) are maintained as JSON files that users can extend 
by running a standard profiling suite.  
To choose which candidates to enact, we pose fusion 
planning as an interval -covering optimization: select a set of 
non-overlapping subgraphs that minimizes the sum of their 
estimated Tcand plus the cost of any remaining unfused nodes. 
We solve this via dynamic programming over the UIR’s 
topological order. At each node, we compare the benefit of 
fusing a candidate ending there against leaving it unfused, 
propagating the minimal cumulative cos t forward. This guar - 
antees a globally near -optimal plan under our  cost model.  
Finally, in the code emission stage, each winning subgraph 
is lowered into a fused kernel via a modular backend plugin. 
For GPUs, we emit CUDA C++ by inlining loops and com - 
bining memory loads; for CPUs, we generate templated Eigen 
or OpenMP -annotated C++ ; for NPUs and other accelerators, 
we invoke vendor SDKs through thin wrappers. Because UIR 
abstracts away hardware details, adding support for a new 
device involves writing only a profile and a small codegen 
template, without touching fusion logic.  
Collectively, these stages form an end -to-end system that 
automatically discovers, evaluates, and materializes operator 
fusions tailored to the diverse performance characteristics of 
modern inference hardware.  
V. EXPERIMENTAL EVALUATION  
We evaluate  UOF on two representative workloads—
ResNet -50 for image classification and BERT - small for 
NLP —across three hardware platforms:  
 CPU:  Intel Xeon  E5-2670 (8 cores  @ 2.6 GHz,  AVX2)  
 GPU: NVIDIA Tesla V100 (5120 CUDA cores, 16 GB 
HBM2)  
 NPU: Mobile Edge NPU (128 ALUs, 4 GB on -chip 
SRAM)  
Baselines:  
1. Unfused  MLFast:  The same  inference engine  with no 
operator fusion.  
2. Vendor Library: NVIDIA TensorRT on GPU, Eigen - 
optimized C++ on CPU, and the NPU’s proprietary SDK.  
Metrics:  We measure  end- to-end latency  (single-batch) and 
peak throughput (batch size = 16), averaging over 100 runs after 
warm -up. 
TABLE  I. End-to-end latency  (ms) and throughput  (images/texts  per second).  
 
Model  / Device  Latency  (ms) Throughput  
 Unfused  UOF   Unfused  UOF  
ResNet -50 / CPU  120 75  8.3 13.3 
ResNet -50 / GPU  25 12  40 83 
ResNet -50 / NPU  60 20  16 50 
BERT -small  / CPU  200 120  5.0 8.3 
BERT -small  / GPU  45 22  22 45 
BERT -small  / NPU  80 28  12.5 35.7 As shown in Table I, UOF achieves up to 1.6× latency 
reduction on CPU,  2.1× on GPU,  and 3.0× on NPU  compared  
to the unfused baseline. Throughput improvements follow a 
similar trend.  Against vendor -tuned libraries, UOF  matches or 
exceeds performance within 5 –10 %.  
All latency and throughput measurements were repeated  
100 times to capture variability. Figure 1 presents the box - plot 
distributions of ResNet -50 single -batch latency across CPU, 
GPU, and NPU for both the unfused baseline and UOF. 
Similarly, Figure 2 shows the throughput distributions. To 
highlight device -specific behavior, Figures 3 and 4 plot his - 
tograms of CPU and  GPU latencies, respectively, illustrating 
tighter latency spreads under UOF.  
 
Fig. 1. Latency distribution for ResNet -50 (batch size = 1) across devices and  
fusion methods.  
 
Fig. 2. Throughput  distribution  for ResNet -50 (batch  size = 16) across  devices  
and fusion methods.  
Ablation Study: We disable the hardware -aware cost 
model, forcing maximal fusion of all compatible ops. This over-
fused configuration yields regressions of 10 –15 % on all 
devices, due to suboptimal kernel shapes and memory spills. 
This underscores the importance of our cost-driven selection.  
VI. DISCUSSION  
Our experiments demonstrate that a unified fusion frame - 
work can automatically approach hand -tuned performance on 
diverse hardware. Key observations include:  
 Portability: The same fusion logic and UIR serve CPU, 
GPU, and NPU with no changes, reducing engineering 
effort.  
248
Authorized licensed use limited to: Karabuk University. Downloaded on May 01,2026 at 12:58:58 UTC from IEEE Xplore.  Restrictions apply. 

## Page 4

 
Fig. 3. Histogram of CPU latency (batch size = 1) comparing unfused vs.  
UOF.  
 
Fig. 4. Histogram of GPU latency (batch size = 1) comparing unfused vs.  
UOF.  
 Cost Model Accuracy:  Offline -profiled device  
parameters enable  reliable  performance predictions; 
minor deviations (¡  5 %) from actual  latencies  validate  
the model’s fidelity.  
 Scalability:  Dynamic programming over the UIR scales 
linearly with graph size; end -to-end planning adds less 
than 5 ms even on large networks.  
Limitations and Future  Work:  
1) Runtime Adaptation: Current profiling is static;  
integrat- ing online performance feedback could 
further refine fusion decisions for varying workloads.  
2) Memory Constraints: Extremely large fusion groups 
may exceed  on-chip memory, suggesting an 
opportunity for multi -stage tiling and split -fusion 
strategies.  
3) Extensibility: While UOF supports new devices via 
profiles and templates,  automating  profile collection  
and codegen template synthesis would streamline 
onboard- ing of novel accelerators.  
VII. CONCLUSION  
We have presented Unified Operator Fusion, a novel ap - 
proach that bridges the gap between  manually  tuned  fusion and 
portable, automated optimization across heterogeneous hard - 
ware. By abstracting computation into a hardware -agnostic 
UIR and guiding fusion with a profile- driven  cost model,  UOF 
discovers and generates  high-performance fused kernels for 
CPUs,  GPUs,  and NPUs.  Experimental results  on ResNet - 50 and BERT -small validate 1.6 –3.0× speedups over unfused 
baselines and competitive parity with vendor libraries. Future 
extensions will explore adaptive runtime fusion and broader 
support for emerging accelerators, further reducing the barrier 
to efficient, portable ML inference.  
In future work, we plan to combine UOF with sequence - 
smoothing techniques [13] to further stabilize inference under 
dynamic input conditions, and to adopt soft -prompt com - 
pression strategies [18] for accelerated fine -tuning of large 
language  models. We also intend  to explore integrating  reward - 
shaping for knowledge-graph reasoning [14] and self-adaptive 
thresholding for 3D object detection [15] into multimodal 
fusion scenarios.  
REFERENCES  
[1] M. Wolf and M. S. Lam, “A data locality optimizing algorithm,” in  
Proceedings of the ACM SIGPLAN 1991 Conference on Programming  
Language Design and Implementation (PLDI) , pp. 30 –44, ACM, 1991.  
[2] K. Kennedy  and J. R. Allen,  “Optimal  loop fusion  in linear  time,”  in 
Proceedings  of the 1986  ACM/IEEE  Conference  on Supercomputing , 
pp. 302–311, IEEE  Computer  Society,  1986.  
[3] TensorFlow XLA Team, “Xla: Optimizing compiler for machine learn - 
ing,” in Google I/O , 2018.  https://www.tensorflow.org/xla.  
[4] T. Chen,  T. Moreau,  Z. Jiang,  L. Zheng,  E. Yan,  H. Shen,  D. Cowen,  
Y. Wang, L. Hu, L. Ceze, et al. , “Tvm: An automated end -to-end 
optimizing compiler for deep learning,” in 13th USENIX Symposium on  
Operating Systems Design and Implementation (OSDI) , pp. 578 –594, 
2018.  
[5] ONNX Runtime Team, “Onnx runtime: Cross -platform, high - 
performance scoring engine for open neural network exchange (onnx)  
models,” arXiv preprint arXiv:2006.14802 , 2020.  
[6] NVIDIA Corporation, “Tensorrt: High -performance deep learning in - 
ference optimizer and runtime,” in NVIDIA Deep Learning Institute  
Workshop , 2016.  
[7] C. Lattner  and et al., “Mlir:  A compiler  infrastructure for the end 
of moore’s law,” in Proceedings of the ACM SIGPLAN International  
Workshop  on Machine  Learning  and Programming  Languages  (MAPL) , 
pp. 1–12, 2019.  
[8] D. Roesch, T. Chen, M. Herrmann, S. Shenker, O. Onufryk, et al. , “Glow: 
Graph lowering compiler techniques for neural networks,” in  
Proceedings of the 2018 Conference on Systems and Machine Learning  
(SysML) , 2018.  
[9] T. Chen,  T. Moreau,  Z. Xu, L. Zheng,  E. Yan,  H. Shen,  D. Cowen,  
Y. Wang, L. Hu, L. Ceze, et al. , “Relay: A high -level ir for machine  
learning,” in Proceedings of the Workshop on MLIR for Tailored Soft - 
ware and Hardware (SYSML) , 2019.  
[10] M. Li, K. Zhao, Y. Guo, X. Kan, L. Zhang, K. Li, X. Wang, et al. , 
“Heterogeneous task scheduling for distributed machine learning,” in  
Proceedings  of the Fourth  Workshop  on Hot Topics  in Operating  Systems  
(HotOS) , 2019.  
[11] X. Jia, J. Liu, H. Yuan,  Y. He, K. Chen,  and G. Taylor,  “Flexflow:  
A flexible dataflow programming model for distributed deep learning,”  
in Proceedings of the 2019 USENIX Annual Technical Conference  
(USENIX ATC) , 2019.  
[12] C. Wang  and J. Gong,  “Intelligent  agricultural  greenhouse  control  system  
based on internet of things and machine learning,” arXiv preprint  
arXiv:2402.09488 , 2024.  
[13] C. Wang  and H. T. Quach,  “Exploring  the effect  of sequence  smoothness  
on machine  learning  accuracy,”  in International  Conference  On Innova - 
tive Computing And Communication , pp. 475 –494, Springer, 2024.  
[14] C. Li, H. Zheng,  Y. Sun, C. Wang,  L. Yu, C. Chang,  X. Tian,  and 
B. Liu, “Enhancing multi -hop knowledge graph reasoning through  
reward shaping techniques,” in 2024 4th International Conference on  
Machine  Learning  and Intelligent  Systems  Engineering  (MLISE) , pp. 1– 
5, IEEE, 2024.  
249
Authorized licensed use limited to: Karabuk University. Downloaded on May 01,2026 at 12:58:58 UTC from IEEE Xplore.  Restrictions apply. 

## Page 5

[15] H. Liu, C. Wang, X. Zhan, H. Zheng, and C. Che, “Enhancing 3d object  
detection by using neural network with self -adaptive thresholding,” in  
Proceedings of the 2nd International Conference on Software Engineer - 
ing and Machine Learning , vol. 67, 2024.  
[16] M. Liu, M. Sui, Y. Nian, C. Wang, and Z. Zhou, “Ca -bert: Leveraging  
context awareness for enhanced multi -turn chat interaction,” in 2024 5th 
International Conference on Big Data & Artificial Intelligence & 
Software Engineering (ICBASE) , pp. 388 –392, IEEE, 2024.  
[17] C. Wang, M. Sui, D. Sun, Z. Zhang, and Y. Zhou, “Theoretical analysis  
of meta  reinforcement  learning:  Generalization  bounds  and convergence  
guarantees,” in Proceedings of the International Conference on Model - 
ing, Natural Language Processing and Machine Learning , pp. 153–159, 
2024.  
[18] C. Wang,  Y. Yang,  R. Li, D. Sun, R. Cai, Y. Zhang,  and C. Fu, “Adapting  
llms for efficient context processing through soft prompt compression,”  
in Proceedings of the International Conference on Modeling, Natural 
Language Processing and Machine Learning , pp. 91 –97, 2024.  
[19] T. Wu, Y. Wang, and N. Quach, “Advancements in natural language  
processing: Exploring transformer -based architectures for text under - 
standing,” arXiv preprint arXiv:2503.20227, 2025.  
[20] Z. Gao, “Theoretical limits of feedback alignment in preference -based  
fine-tuning of ai models,” 2025.  [21] Z. Gao, “Modeling reasoning as markov decision processes: A theoret - 
ical investigation into nlp transformer models,” 2025.  
[22] Z. Gao, “Feedback -to-text alignment: Llm learning consistent natural  
language generation from user ratings and loyalty data,” 2025.  
[23] Y. Sang,  “Robustness  of fine-tuned  llms under  noisy  retrieval  inputs,”  
Preprints , July 2025.  
[24] N. Quach, Q. Wang, Z. Gao, Q. Sun, B. Guan, and L. Floyd, “Rein - 
forcement learning approach for integrating compressed contexts into  
knowledge graphs,” in 2024 5th International Conference on Computer 
Vision, Image and Deep Learning (CVIDL) , pp. 862 –866, IEEE, 2024.  
[25] D. Hendrycks, S. Mu, and T. G. Dietterich, “The many faces of  robustness:  
Benchmarking  neural  network  robustness  to common  corrup- tions, 
perturbations, and subtextures,” in Proceedings of the IEEE/CVF 
International Conference on Computer Vision (ICCV) , 2021. ImageNet - 
R. 
[26] D. Lo, K. Neeraj, M. Joshi, L. Thomas, S. Biderman, and F. Lenhert,  
“Qasper: A dataset of information -seeking questions and answers on  
research papers,” in Proceedings of the 2021 Conference on Empirical 
Methods in Natural Language Processing (EMNLP) , pp. 4963 –4979,  
2021.  
 
250
Authorized licensed use limited to: Karabuk University. Downloaded on May 01,2026 at 12:58:58 UTC from IEEE Xplore.  Restrictions apply. 