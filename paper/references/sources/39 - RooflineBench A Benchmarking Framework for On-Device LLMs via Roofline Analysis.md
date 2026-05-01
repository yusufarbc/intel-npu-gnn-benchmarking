# RooflineBench A Benchmarking Framework for On-Device LLMs via Roofline Analysis

## Page 1

RooflineBench: A Benchmarking Framework for
On-Device LLMs via Roofline Analysis
Zhen Bi1 *Xueshu Chen1 *Luoyang Sun2Yuhang Yao3Qing Shen1Jungang Lou1†Cheng Deng4†
Abstract
The transition toward localized intelligence
through Small Language Models (SLMs) has in-
tensified the need for rigorous performance char-
acterization on resource-constrained edge hard-
ware. However, objectively measuring the theo-
retical performance ceilings of diverse architec-
tures across heterogeneous platforms remains a
formidable challenge. In this work, we propose
a systematic framework based on the Roofline
model that unifies architectural primitives and
hardware constraints through the lens of opera-
tional intensity ( OI). By defining an inference-
potential region, we introduce theRelative In-
ference Potentialas a novel metric to compare
efficiency differences between Large Language
Models (LLMs) on the same hardware substrate.
Extensive empirical analysis across diverse com-
pute tiers reveals that variations in performance
andOIare significantly influenced by sequence
length. We further identify a critical regression
inOIas model depth increases. Additionally,
our findings highlight an efficiency trap induced
by hardware heterogeneity and demonstrate how
structural refinements, such as Multi-head Latent
Attention ( MLA ), can effectively unlock latent
inference potential across various hardware sub-
strates. These insights provide actionable direc-
tions for hardware-software co-design to align
neural structures with physical constraints in on-
device intelligence. The released code is available
in the Appendix C.
1. Introduction
The rapid advancement of Large Language Models (LLMs)
has redefined the landscape of artificial intelligence, yet their
1Huzhou University2Institution of Automation, Chinese
Academy of Sciences3Carnegie Mellon University4University
of Edinburgh. Correspondence to: Jungang Lou and Cheng Deng
<bizhen@zjhu.edu.cn|cdeng@ed.ac.uk>.
Preprint. March 16, 2026.substantial scales continue to impose significant burdens on
resource efficiency and deployment costs (Hoffmann et al.,
2022). This challenge has catalyzed a paradigm shift toward
Small Language Models (SLMs) that prioritize capacity den-
sity and on-device accessibility (Liu et al., 2024). Recent
breakthroughs, including the Gemma, Phi-3, and MiniCPM
families, demonstrate that compact architectures can rival
significantly larger models while remaining deployable on
consumer hardware (Team, 2024a; Abdin et al., 2024; Hu
et al., 2024). This trajectory further extends to the multi-
modal domain, where efficient models like MiniCPM-V and
LLaV A-v1.6 surpass proprietary counterparts with minimal
computational overhead (Yao et al., 2024; Yu et al., 2025;
Li et al., 2023). Such developments underscore a critical
transition toward localized intelligence that ensures data
privacy and democratizes powerful AI capabilities across
edge devices.
Recent literature has increasingly prioritized the quantitative
assessment of inference efficiency through specialized met-
rics such as Model Bandwidth Utilization ( MBU ) (Agarwal
et al., 2023), Model FLOPs Utilization ( MFU ) (Chowdhery
et al., 2023), and evaluation frameworks like MoeCap (Jiang
et al., 2025). Despite these advancements, establishing a
comprehensive understanding of on-device intelligence re-
mains a formidable task. Primarily, objectively character-
izing the theoretical performance upper bound for specific
large model architectures when deployed on heterogeneous
hardware platforms is challenging because of the complex
interplay between software kernels and hardware substrates
(Dao et al., 2022). Furthermore, conventional evaluation
methods often lack the analytical depth required to pinpoint
the fundamental physical constraints limiting inference ef-
ficacy in resource-constrained environments. These limita-
tions necessitate a more robust analytical framework that
can decompose hardware-software interactions to identify
precise bottlenecks across diverse compute tiers.
To address these challenges, we leverage the Roofline model
as a rigorous analytical bridge to characterize the interplay
between large model inference and underlying hardware
capabilities (Williams et al., 2009). Our approach entails
the empirical measurement of peak FLOPS and memory
bandwidth to establish a realistic performance envelope
1arXiv:2602.11506v3  [cs.LG]  13 Mar 2026

## Page 2

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
On-Device LLM
(Model Weights & 
Activation)
Hardware 
Specification
(Compute & Memory)RooflineBench
(Profiling & Analysis)A. RooflineBench Framework B. Inference via Roofline Analysis
Operational Intensity  ( FLOPs / Byte )Performance  ( FLOPs / Sec )Peak Compute 
Performance
Peak Memory 
Bandwidth
Ridge Point
Inference Potential
Current LLM 
Performance
Figure 1.Overview of the RooflineBench framework and analytical methodology.Left:The integrated profiling system incorporating
on-device LLM configurations and hardware specifications for systematic analysis.Right:Graphical representation of the Roofline model
illustrating theRelative Inference Potentialbetween observed throughput and theoretical hardware ceilings across both memory-bound
and compute-bound regimes.
across diverse platforms. We also argue that LLMs exhibit
an inference potential region on edge hardware devices (the
blue region in in Figure 1), and we propose a novel metric
(Relative Inference Potential) to compare the differences in
inference potential between two LLMs on the same hard-
ware device. The primary contributions of this work are
three fold:
•Integrated Benchmarking Framework: We propose
a systematic Roofline based framework that unifies
architectural primitives and hardware constraints via
operational intensity ( OI), defining an inference poten-
tial region to introduce the Relative Inference Potential
(Φ) for comparative efficiency analysis.
•Comprehensive Empirical Analysis: Through exten-
sive experiments across heterogeneous compute tiers,
we reveal that inference efficiency is primarily gov-
erned by context length and attention architectures,
while identifying a critical OIregression at increased
model depths.
•Hardware-Software Co-design Inspirations: Our
findings identify an efficiency trap induced by hard-
ware heterogeneity and demonstrate how architectural
refinements can maximize utilization across diverse
hardware substrates to effectively bridge the gap be-
tween theoretical potential and real world execution.
2. Background
The Physics of On-Device DecodingFollowing the an-
alytical framework ofPope et al.(Pope et al., 2023), we
deconstruct Transformer inference into two phases: prefilland decoding. While the prefill phase processes input tokens
in parallel and is typically compute-bound, the decoding
phase generates tokens autoregressively, where each step
sequentially depends on previously generated tokens. This
sequential nature inherently shifts the execution bottleneck
toward memory bandwidth.
Mathematical Derivation of the Bandwidth Bottleneck
For a decoder-only model with nparams parameters, it has
been shown that the forward pass requires approximately
2·nparams floating-point operations (FLOPs) per token . Let
Ppeak be the peak performance of the hardware and BW
be the memory bandwidth. During decoding, the model
weights and the attention key-value (KV) cache must be
transferred from memory to compute cores for every single
token. We define thecompute time( Tcomp ) andmemory
time(T mem) as:
Tcomp=2·n params
Ppeak(1)
Tmem≈nparams·bprec
BW(2)
where bprec is the bytes per parameter. The system is
memory-bandwidth bound when Tmem > T comp . This
constraint is captured by theOperational Intensity(OI):
OIdec≈2·n params
nparams·bprec=2
bprec(3)
For 16-bit precision ( bprec= 2),OIdecode≈1FLOP/Byte.
Since modern edge accelerators often have Ppeak/BW ra-
tios far exceeding 1, the computational cores remain essen-
tially idle while waiting for tensors to be loaded, leading to
low model FLOPs utilization (MFU).
2

## Page 3

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
Edge Heterogeneity and KV Cache ScalingOn
resource-constrained edge devices, the additional memory
traffic from the KV cache further reduces the system’s Oper-
ational Intensity (OI), shifting the execution point to theleft
on the Roofline model (Williams et al., 2009). Simultane-
ously, the limited memory bandwidth ( BW ) of mobile SoCs
causes the hardware’s “knee point” ( Ppeak/BW ) to shift to
theright. This dual effect forces the decoding phase deep
into the severe memory-bandwidth bound region, where
the computational cores remain underutilized while waiting
for tensor loading. Consequently, architectural optimiza-
tions such as multiquery attention (MQA) are essential to
reduce the KV cache footprint and alleviate this bandwidth
pressure.
3. Methodology: The Integrated Standard
Roofline Framework
This section presents our performance analysis framework.
Unlike simulation-based approaches, our tool is runtime-
integrated, designed to quantify LLM’s inference potential
by comparing real-time inference telemetry against empiri-
cally measured hardware limits.
3.1. Standard Roofline Formulation
We adopt the standard Roofline model to establish the the-
oretical performance upper bound. For a given hardware
platform, attainable performance P(GFLOPS) is bounded
by either compute capacity or memory bandwidth:
P= min(P peak, OI×BW peak)
Where Ppeak: Theoretical peak compute performance
(GFLOPS). BW peak: Achievable peak memory bandwidth
(GB/s). OI(Operational Intensity): The ratio of floating-
point operations to bytes of memory traffic (FLOPs/Byte).
Given the autoregressive nature of LLM decoding (token-by-
token generation), the process is inherently memory-bound.
Consequently, our analysis focuses primarily on thesloped
regionof the Roofline graph, where performance is strictly
limited byOI×BW peak.
3.2. Applying the Standard Roofline Model to LLMs
To rigorous quantify the efficiency of LLM decoding, we
instantiate the Roofline model by defining its three core co-
ordinates: theoretical FLOPs, memory traffic, and empirical
hardware limits.
•FLOPs ( W): Analytical Estimation.Directly mea-
suring FLOPs via hardware counters can be imprecise
due to instrumentation overhead. Instead, we adopt
the analytical formulation from Sun et al. (2025) to
estimate the theoretical FLOPs for heterogeneous ar-
chitectures (e.g., MHA, GQA). Given a model withhidden dimension H, sequence length N, and head
configurations {nq, nk, nv}, we precisely calculate the
computational load for the Linear, Attention, and FFN
layers per decoding step. This ensures a uniform stan-
dard for calculating Performance ( P) and Operational
Intensity (OI) across different backends.
•Memory Traffic ( Q): Data Movement Approxima-
tion.In the context of memory-bound LLM decoding,
memory traffic is dominated by loading model weights
and the KV cache. We approximate the total data
movement Qper token generation as the summation
of the model parameters size and the active KV cache
entries read/written during that step. This approach
captures the requisite data transfer volume regardless
of the specific memory management implementation
(e.g., Unified Memory on Apple Silicon vs. GDDR6X
on NVIDIA GPUs).
•Performance & Operational Intensity Formulation.
Based on the derived WandQ, combined with the
measured end-to-end latency ( T), we define the run-
time metrics as:
Performance (GFLOPS)=W
T(4)
Operational Intensity (FLOPs/Byte)=W
Q(5)
•Hardware Limit Characterization.We establish
the theoretical ceiling by empirically profiling the
peak memory bandwidth ( BW peak) and compute per-
formance ( Ppeak). For Apple Silicon devices, we
benchmark the Unified Memory bandwidth, as the
GPU shares system memory. Conversely, for CUDA-
enabled devices (e.g., RTX 4090), we isolate and mea-
sure the dedicated Video Memory (VRAM) bandwidth,
as it constitutes the primary bottleneck for on-device
inference.
3.3. Characterizing Relative Inference Potential across
Performance Regimes
We defineRelative Inference Potential( Φ) to quantify op-
timization headroom based on the spatial relationship be-
tween a performance point P(OI p, Perf p)and the hard-
ware ridge R(OI r, π). The calculation logic varies across
different regimes to reflect shifting physical constraints.
Memory Bound Regime ( OI < OI r):Φis the Euclidean
distance to R, capturing the dual necessity of increasing
operational intensity and throughput:
Φ(P) =q
(OI r−OI p)2+ (π−Perf p)2 (6)
Compute Bound Regime ( OI≥OI r): Since horizontal
OIgains yield negligible ceiling increases, Φis defined as
3

## Page 4

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Qwen2.5-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Llama-3.2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: PLM-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: SmolLM2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
Figure 2.Comparison of roofline models across different input-output sequence length scenarios on Apple M1 Pro. Performance is
evaluated in FP16 precision for Qwen2.5-Instruct (GQA), Llama-3.2-Instruct (GQA), PLM-Instruct (MLA), and SmolLM2-Instruct
(MHA). The markers represent four inference scenarios: SISO, SILO, LISO, and LILO. Detailed analysis of the sensitivity to attention
mechanisms is provided in subsequent sections. For comprehensive and detailed benchmarks across a wider range of hardware devices,
please refer to Appendix D.
the vertical distance to the peak compute limitπ:
Φ(P) =π−Perf p (7)
Cross Regime Incomparability: Data points on opposite
sides of the ridge are fundamentally incomparable due to
distinct physical bottlenecks (bandwidth vs computation),
necessitating regime specific evaluation.
4. Comprehensive Characterization Study
This section provides a systematic characterization of LLM
inference using the proposedRelative Inference Potential
(Φ) within a Roofline analytical framework. Unlike tradi-
tional macro metrics such as throughput (TPS), Φfacilitates
a regime aware evaluation of execution efficiency by quanti-
fying the spatial discrepancy between realized performance
and theoretical hardware limits. We evaluate task level sen-
sitivity across four sequence patterns (Sec. 4.1), investigate
parameter level scaling and bottleneck shifts through model
depth profiling (Sec. 4.2), and assess algorithmic impacts
from precision and architectural refinements such as MLA
(Sec. 4.3).By identifying critical transitions between
memory bound and compute bound regimes, this multi-
dimensional analysis leverages Φto reveal the internal
complexity and optimization headroom of localized in-
telligence.Experimental details are in Appendix C.
4.1. Sensitivity to Input-Output Sequence Lengths
To evaluate the impact of task-level workloads on on-device
decoding efficiency, we categorize four representative se-
quence patterns based on typical edge-side use cases: (i)
SISO (Short In, Short Out), such as local voice commands;
(ii) SILO (Short In, Long Out), typical of creative writing or
code completion; (iii) LISO (Long In, Short Out), reflecting
RAG-based context extraction or document summarization;
and (iv) LILO (Long In, Long Out), characteristic of docu-
ment translation. These configurations fundamentally alterthe ratio between attention-related computation and model
weight loading during each autoregressive step.
In Figure 2, decoding performance exhibits significant sensi-
tivity to sequence lengths across all tested models. Notably,
the LISO scenario (yellow dots) consistently achieves the
highest execution efficiency, positioning itself as the closest
observed case to the Roofline ridge point. This proximity
stems from the larger input context, which increases the
relative computational demand of the attention mechanism
within each decoding step. By effectively amortizing the
fixed memory overhead of loading model weights, LISO
exhibits a high computational proportion, thereby elevating
both the operational intensity ( OI) and attainable perfor-
mance ( GFLOPS ) toward the hardware’s compute-bound
limit. Conversely, the SILO scenario (red dots) resides deep
within the memory-bound regime. In this scenario, the neg-
ligible computational requirement of processing a minimal
context cannot offset the massive data movement of weights,
leading to severe hardware underutilization.
Insight.1.Context length is the primary factor deter-
mining both the operational intensity and performance
of on-device decoding: the LISO scenario approaches
the compute-bound limit due to its inherently high com-
putational proportion, while the SILO scenario remains
severely memory-bound.
4.2. Bottleneck Evolution under Parameter Expansion
To investigate the impact of model scale on on-device in-
ference dynamics, we evaluate a series of configurations by
scaling the number of Transformer layers from 2 to 64. This
range encompasses both ultra-lightweight edge models and
standard mobile-class LLMs, providing a continuous spec-
trum to observe performance scaling laws across varying
parameter capacities.
As illustrated in Figure 3, scaling model depth reveals a dis-
4

## Page 5

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
0.5 1.0 1.5 2.0 2.5 3.0 3.512345Performance(FLOPs/sec)1e9 FP16 | LILO
10 20 30 40 50 60123456781e10 FP16 | LISO
0.005 0.010 0.015 0.020 0.025 0.030 0.035123451e7 FP16 | SILO
0.2 0.4 0.6 0.8 1.00.250.500.751.001.251.501.751e9 FP16 | SISO
1 2 3 4 5 6 7123456Performance(FLOPs/sec)1e9 Q8_0 | LILO
20 40 60 80 100 120123456781e10 Q8_0 | LISO
0.01 0.02 0.03 0.04 0.05 0.06 0.0712345671e7 Q8_0 | SILO
0.50 0.75 1.00 1.25 1.50 1.75 2.00 2.25 2.500.51.01.52.01e9 Q8_0 | SISO
2 4 6 8 10 12 14
Operational Intensity (FLOPs/Byte)12345678Performance(FLOPs/sec)1e9 Q4_K_M | LILO
25 50 75 100 125 150 175 200
Operational Intensity (FLOPs/Byte)123456781e10 Q4_K_M | LISO
0.02 0.04 0.06 0.08 0.10 0.12 0.14
Operational Intensity (FLOPs/Byte)1234567891e7 Q4_K_M | SILO
1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0
Operational Intensity (FLOPs/Byte)0.51.01.52.02.53.01e9 Q4_K_M | SISOLayers: 2
Layers: 4
Layers: 6
Layers: 8
Layers: 10
Layers: 12
Layers: 14
Layers: 16
Layers: 18
Layers: 20
Layers: 22
Layers: 24
Layers: 26
Layers: 28
Layers: 30
Layers: 32
Layers: 34
Layers: 36
Layers: 38
Layers: 40
Layers: 42
Layers: 44
Layers: 46
Layers: 48
Layers: 50
Layers: 52
Layers: 54
Layers: 56
Layers: 58
Layers: 60
Layers: 62
Layers: 64
Figure 3.The figure shows the inference performance of Qwen2.5-Instruct FP16, Q8 0, and Q4 KM with 2 to 64 layers (Experimental
platform: Apple M1 Pro). Comprehensive and detailed benchmarks across a wider range of hardware devices are in Appendix D.
tinct, non-linear evolution of hardware utilization on edge
platforms. Initially, the attainable performance ( GFLOPS )
rises consistently and becomes increasingly dense as lay-
ers increase (transitioning from light yellow to dark purple
dots). This behavior suggests that deeper models more
effectively amortize fixed system-level overheads—such
as kernel launch latencies, memory synchronization, and
command orchestration—thereby enabling the hardware to
operate closer to its steady-state compute limits.
More significantly, the operational intensity ( OI) exhibits
a non-monotonic ”arch” trajectory with a clear inflection
point at remarkably shallow depths. As depth increases from
2 to approximately 3–5 layers, the data points initially shift
to the right, indicating an initial gain in arithmetic reuse
as the system moves away from the overhead-dominated
regime. However, beyond this threshold, the OIbegins to
retreat to the left. In resource-constrained on-device environ-
ments, this regression arises because the cumulative memory
bandwidth pressure of streaming weights for additional lay-
ers begins to outpace the marginal gains in computational
reuse during the decoding phase. Consequently, the hard-
ware ”memory-wall” is encountered significantly earlier
than theoretical predictions suggest, forcing the decoding
process into an increasingly memory-bound regime as the
model scales, which further constrains the release of relative
inference-potential.Insight.2.On-device decoding reaches peak opera-
tional intensity at a remarkably shallow depth (3–5 lay-
ers), beyond which the memory bandwidth overhead of
weight-streaming triggers a regression in operational
intensity.
4.3. Algorithmic Influences on Operational Intensity
Algorithmic optimizations fundamentally reshape the in-
ference profile on the Roofline plane by modulating the
balance between data movement and computation. This sec-
tion evaluates the impact of numerical precision (Sec. 4.3.1)
and attention architectures (Sec. 4.3.2) on the operational
intensity (OI) and bottleneck transitions of on-device LLMs.
4.3.1. NUMERICALPRECISION: FROMFP16TO4-BIT
QUANTIZATION
To evaluate the impact of numerical precision on on-device
efficiency, we compare FP16, Q8 0, and Q KM formats.
Quantization directly reduces the memory footprint of
model weights, fundamentally reshaping the data movement
characteristics during edge deployment. Detailed FLOPs
calculation formulas for the KV cache and attention mecha-
nisms across these architectures are in Appendix C.
5

## Page 6

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Qwen2.5-1.5B-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
Qwen2.5-Instruct FP16
Qwen2.5-Instruct Q8_0
Qwen2.5-Instruct Q4_K_M
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Llama-3.2-1B-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
Llama-3.2-Instruct FP16
Llama-3.2-Instruct Q8_0
Llama-3.2-Instruct Q4_K_M
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: PLM-1.8B-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
PLM-Instruct FP16
PLM-Instruct Q8_0
PLM-Instruct Q4_K_M
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Qwen3-0.6B
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
Qwen3 FP16
Qwen3 Q8_0
Qwen3 Q4_K_M
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Fox-1-1.6B
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
Fox-1 FP16
Fox-1 Q8_0
Fox-1 Q4_K_M
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: SmolLM2-1.7B-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SmolLM2-Instruct FP16
SmolLM2-Instruct Q8_0
SmolLM2-Instruct Q4_K_M
Figure 4.Performance comparison of various models across three different precision settings and all benchmarked hardware platforms.
The results demonstrate a consistent performance gap and uniform trends, regardless of the specific device or precision level employed.
In Figure 4, lowering numerical precision triggers a diag-
onal shift of the data points toward the upper-right region
of the Roofline plane. However, the magnitude of this im-
provement is highly sensitive to the task-level bottleneck. In
memory-bound scenarios such as SILO (bottom-left), tran-
sitioning from FP16 to Q4 KM yields dramatic gains in
both operational intensity (OI) and attainable performance
(GFLOPS ), as the reduced bandwidth pressure directly
accelerates the execution. Conversely, in the LISO scenario
(top-right), the clusters for the three precisions appear sig-
nificantly tighter. Since LISO is already positioned near the
Roofline ridge with high arithmetic intensity, the benefits of
reduced data movement are less pronounced, indicating that
the execution has largely transitioned from being memory-
constrained to being limited by the hardware’s peak compute
capability.
Insight.3.Quantization provides maximal efficiency
gains for memory-bound tasks (e.g., SILO), whereas in
compute-heavy scenarios (e.g., LISO), the performance
becomes increasingly saturated as it approaches the
hardware’s theoretical peak.
4.3.2. ATTENTIONMECHANISMS: MULTI-HEAD VS.
SPARSEVARIATIONS
To investigate the impact of attention architectures on hard-
ware efficiency, we compare three prevalent designs: Multi-
Head Attention (MHA), Grouped-Query Attention (GQA)
and Multi-head Latent Attention (MLA). To ensure fairness,all LLMs are scaled to a unified size of approximately 1.5B
parameters by adjusting their respective layer counts.
In Figure 6, the choice of attention mechanism significantly
shifts the inference profile on the Roofline plane. MLA (blue
dots) consistently achieves the highest operational intensity
(OI) and attainable performance ( GFLOPS ) across all
four sequence scenarios. By utilizing latent compression
for the KV cache, MLA substantially reduces the volume of
data movement required for each decoding step compared
to the standard MHA, thereby shifting the execution sig-
nificantly closer to the Roofline ridge point. Interestingly,
at this specific 1.5B scale on edge hardware, GQA (red
dots) exhibits the lowest efficiency among the three. It in-
dicates that while GQA reduces the number of Key/Value
heads, the latent-based compression in MLA provides a
more effective balance between memory traffic reduction
and computational throughput for on-device decoding.
Insight.4.Attention architecture is a decisive factor
for decoding efficiency: MLA outperforms both MHA
and GQA by effectively compressing KV cache traffic to
maximize operational intensity on resource-constrained
devices.
5. A Fairness-View: Hardware Utilization
Efficiency
Hardware utilization efficiency is constrained by the intrin-
sic disparities of heterogeneous platforms and the structural
6

## Page 7

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
102
101
100101102103
Operational Intensity (FLOPs/Byte)10910101011101210131014Performance (FLOPs/sec)Theoretical Roofline Comparison
Apple M1 Pro (Ridge: 25.39)
RTX 3090 (Ridge: 38.00)
RTX 3070Ti Laptop (Ridge: 37.05)
Jetson Orin Nano Super 8G (Ridge: 20.48)
Raspberry Pi 5 (Ridge: 8.98)
2×1013×1014×1016×101
Operational Intensity (FLOPs/Byte)1011Performance (FLOPs/sec)Measured Performance - LISO Scenario
[M1 Pro] GQA: Qwen2.5-Instruct FP16
[M1 Pro] MLA: PLM-Instruct FP16
[M1 Pro] MHA: SmolLM2-Instruct FP16
[3070Ti Laptop] GQA: Qwen2.5-Instruct FP16
[3070Ti Laptop] MLA: PLM-Instruct FP16
[3070Ti Laptop] MHA: SmolLM2-Instruct FP16
Figure 5.Analysis of hardware utilization efficiency and architectural scalability across heterogeneous platforms. The left panel presents
the Theoretical Roofline Comparison for five representative devices including RTX 3090, RTX 3070Ti Laptop, Apple M1 Pro, Jetson
Orin Nano, and Raspberry Pi 5, illustrating a significant disparity in theoretical ridge points ranging from 8.98 to 38.00 FLOPs/Byte .
The right panel depicts the Measured Performance in the LISO scenario for MLA, MHA, and GQA configurations across the Apple M1
Pro and RTX 3070Ti Laptop, highlighting the consistent efficiency baseline and parallel shift in operational intensity and performance
when scaling across hardware tiers.
variations of model architectures. This section evaluates the
fairness of on-device inference by analyzing performance
gaps across diverse hardware environments (Sec. 5.1) and
examining how architectural choices influence the equity of
resource saturation at a unified model scale (Sec. 5.2).
5.1. Disparity across Heterogeneous Hardware
Platforms
Hardware utilization efficiency is strictly governed by the
physical constraints of the underlying platform. As illus-
trated in Fig. 5, the theoretical Roofline profiles of five
representative devices, ranging from the high-performance
RTX 3090 to the ultra-low-power Raspberry Pi 5, reveal
a staggering disparity in hardware capabilities. The theo-
retical peak performance spans three orders of magnitude
from1011to1013FLOPs/sec, while the memory bandwidth
defines distinct operational boundaries for each platform.
The critical indicator of hardware unfairness is the theoreti-
cal ridge point, which dictates the minimum OIrequired to
saturate computational resources. High-performance GPUs
such as the RTX 3090 (Ridge: 38.00) and RTX 3070Ti
Laptop (Ridge: 37.05) possess significantly higher ridge
points, meaning they require a much higher OIto transi-
tion from memory-bound to compute-bound regimes. In
contrast, edge-side chips like the Apple M1 Pro (Ridge:
25.39), Jetson Orin Nano (Ridge: 20.48) and Raspberry Pi
5 (Ridge: 8.98) have much lower ridge points. While these
devices are easier to saturate, their absolute performance
ceilings are considerably lower. This disparity implies that
a specific model architecture, such as a 1.5B model in the
LISO scenario, may achieve near-optimal saturation on a
Jetson Orin or Raspberry Pi but remain severely underuti-
lized on an RTX 3090 because of the structural differences
in theoretical memory-to-compute ratios.Insight.5.Hardware heterogeneity creates an ”effi-
ciency trap”: the disparity in ridge points across devices
ensures that a single model architecture cannot achieve
uniform utilization equity, as the same operational in-
tensity leads to vastly different bottleneck regimes on
different platforms.
5.2. Architectural Sensitivity of Utilization Equity
The cross-platform evaluation demonstrates that the effi-
ciency dividends of architectural optimization are consis-
tently preserved across varying hardware tiers. As illustrated
in Fig. 6, we analyze the performance of different archi-
tectural configurations including MLA, MHA, and GQA
on the Apple M1 Pro and the RTX 3070Ti Laptop under
the compute-intensive LISO scenario. The results indicate
that as the execution environment scales from an edge SoC
to a high-performance GPU, all architectural designs ben-
efit from increased memory bandwidth and computational
peaks, resulting in a parallel shift toward the upper-right
region of the Roofline plane.
Despite the collective performance gains, the relative hi-
erarchy of utilization equity remains sensitive to the spe-
cific architectural design. MLA (blue markers) consistently
maintains a superior position in both operational intensity
(OI) and attainable performance ( GFLOPS ) on both plat-
forms. Rather than being a platform-specific enhancement,
the latent-based compression in MLA functions as a robust
efficiency baseline. This stability proves that optimized
architectural structures can reliably capture available hard-
ware resources across the device spectrum, ensuring that the
performance advantages of high-end accelerators are not
overshadowed by the structural inefficiencies of traditional
attention mechanisms.
7

## Page 8

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
Insight.6.Architectural optimization demonstrates con-
sistent cross-platform robustness because while hard-
ware scaling elevates the performance of all configu-
rations, optimized structures such as MLA maintain a
higher baseline of operational intensity and attainable
performance, ensuring superior utilization of heteroge-
neous computational resources.
6. Discussion
Synergy between Architectural Innovation and Hard-
ware ConstraintsThe architectural evolution toward
Multi head Latent Attention (MLA), as seen in DeepSeek
V2(DeepSeek-AI, 2024) and PLM(Deng et al., 2025),
along with the trainable sparse attention mechanism in
MiniCPM(Hu et al., 2024), exemplifies the necessity of
hardware aware innovation to transcend the memory wall in
on device environments. MLA utilizes latent compression
for the KV cache to reduce the volume of data movement
required for each decoding step. Simultaneously, the sparse
attention in MiniCPM reduces computational overhead by
processing less than 5%of tokens in long context scenar-
ios. Both structural choices elevate the operational intensity
(OI) and shift the inference profile toward the compute
bound regime. As shown in the cross platform compari-
son, MLA based configurations achieve a higher proficiency
in capturing available computational peaks on high perfor-
mance hardware compared to traditional mechanisms. Such
synergy suggests that future on device architectures should
prioritize optimizations such as latent compression or train-
able sparsity to better align with the divergent theoretical
ridge points of heterogeneous hardware.
Evolution of Capacity Density in Resource-Constrained
EnvironmentsCapacity density, defined as the ratio of
effective to actual parameter size, provides a unified frame-
work for evaluating model quality according to the Densing
Law(Xiao et al., 2024). This metric is crucial for resource-
constrained environments where traditional scaling through
model size is increasingly unsustainable. Layer-wise anal-
ysis of models such as Qwen2.5(Yang et al., 2024) reveals
that operational intensity ( OI) does not scale linearly with
depth but reaches a peak at low layer counts. In the LISO
scenario, increasing depth beyond this optimal threshold
leads to an OIfallback. Because edge platforms such as
the Raspberry Pi 5 or Jetson Orin Nano have low theoretical
ridge points, they are highly sensitive to OIfluctuations.
Consequently, improving the capacity density of shallow
architectures is a more sustainable strategy for on-device
intelligence than simply stacking layers, as it maximizes
effective parameters within a restricted memory footprint.
103
102
101
100101102103104
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Roofline Analysis of Attention Variants (~1.5B)
Theoretical Memory Access Bound
Theoretical Computation Bound
Measured Memory Access Bound
Measured Computation Bound
Measured Ridge: 35.83 FLOPs/Byte
GQA: Qwen2.5-1.5B-Instruct-28-layers FP16
MLA: PLM-1.8B-Instruct-26-layers FP16
MHA: SmolLM2-1.7B-Instruct-21-layers FP16Figure 6.Impact of MHA, GQA, and MLA architectures on
inference efficiency. All models are scaled to 1.5B parameters and
tested at FP16 precision on an Apple M1 Pro. The distribution
across the Roofline plane illustrates the efficiency gains of latent-
based attention variations.
Paradigm of Hardware-Targeted Specialization for Ar-
chitectural PrimitivesThe gap between theoretical peaks
and realized throughput suggests that framework design
must prioritize data movement over raw compute capacity.
Although TFLOPS have grown rapidly, inference remains
fundamentally limited by memory access patterns across di-
verse devices. Refining the interaction between hierarchical
memory structures and execution kernels is essential to miti-
gate the bandwidth constraints that govern large scale model
deployment. Additionally, hardware heterogeneity necessi-
tates specializing AI compute units for specific high load
primitives. General purpose logic frequently fails to exploit
the structural advantages of modern architectures, leading
to efficiency losses in on device scenarios. Hardware level
optimization for critical primitives such as MLA and sparse
attention is therefore crucial for maximizing throughput.
Dedicated silicon support ensures that the theoretical opera-
tional intensity ( OI) of optimized models can reach its full
potential in real world execution.
7. Conclusion
We characterize LLM inference on edge devices using the
Roofline model and introduceRelative Inference Potential
to compare efficiency across heterogeneous substrates. Find-
ings show that sequence length dictates distinct performance
and operational intensity ( OI) dynamics. We identify a non-
monotonic OItrajectory: performance initially rises via
overhead amortization but regresses beyond 3–5 layers as
parameter loading outpaces computational reuse. Our analy-
sis reveals an efficiency trap from hardware ridge-point dis-
parities, while structural refinements sustain higher OIvia
optimized cache management. These results advocate for
hardware-software co-design to align model architectures
with physical constraints and unlock localized intelligence.
8

## Page 9

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
References
Abdin, M. I., Jacobs, S. A., Awan, A. A., Aneja, J., Awadal-
lah, A., Awadalla, H., Bach, N., Bahree, A., Bakhtiari, A.,
Behl, H. S., Benhaim, A., Bilenko, M., Bjorck, J., Bubeck,
S., Cai, M., Mendes, C. C. T., Chen, W., Chaudhary, V .,
Chopra, P., Giorno, A. D., de Rosa, G., Dixon, M., El-
dan, R., Iter, D., Garg, A., Goswami, A., Gunasekar, S.,
Haider, E., Hao, J., Hewett, R. J., Huynh, J., Javaheripi,
M., Jin, X., Kauffmann, P., Karampatziakis, N., Kim, D.,
Khademi, M., Kurilenko, L., Lee, J. R., Lee, Y . T., Li,
Y ., Liang, C., Liu, W., Lin, E., Lin, Z., Madan, P., Mitra,
A., Modi, H., Nguyen, A., Norick, B., Patra, B., Perez-
Becker, D., Portet, T., Pryzant, R., Qin, H., Radmilac, M.,
Rosset, C., Roy, S., Ruwase, O., Saarikivi, O., Saied, A.,
Salim, A., Santacroce, M., Shah, S., Shang, N., Sharma,
H., Song, X., Tanaka, M., Wang, X., Ward, R., Wang, G.,
Witte, P. A., Wyatt, M., Xu, C., Xu, J., Yadav, S., Yang,
F., Yang, Z., Yu, D., Zhang, C., Zhang, C., Zhang, J.,
Zhang, L. L., Zhang, Y ., Zhang, Y ., Zhang, Y ., and Zhou,
X. Phi-3 technical report: A highly capable language
model locally on your phone.CoRR, abs/2404.14219,
2024. doi: 10.48550/ARXIV.2404.14219. URL https:
//doi.org/10.48550/arXiv.2404.14219.
Agarwal, M., Qureshi, A., Sardana, N., Li, L., Quevedo, J.,
and Khudia, D. https://www.databricks.com
/blog/llm-inference-performance-engin
eering-best-practices, 2023.
Alizadeh, K., Mirzadeh, I., Belenko, D., Khatamifard, S.,
Cho, M., del Mundo, C. C., Rastegari, M., and Fara-
jtabar, M. LLM in a flash: Efficient large language
model inference with limited memory. In Ku, L., Mar-
tins, A., and Srikumar, V . (eds.),Proceedings of the
62nd Annual Meeting of the Association for Computa-
tional Linguistics (Volume 1: Long Papers), ACL 2024,
Bangkok, Thailand, August 11-16, 2024, pp. 12562–
12584. Association for Computational Linguistics, 2024.
doi: 10.18653/V1/2024.ACL-LONG.678. URL
https://doi.org/10.18653/v1/2024.acl
-long.678.
Chen, L., Zaharia, M., and Zou, J. Frugalgpt: How to use
large language models while reducing cost and improving
performance.Trans. Mach. Learn. Res., 2024, 2024. URL
https://openreview.net/forum?id=cSim
Kw5p6R.
Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra,
G., Roberts, A., Barham, P., Chung, H. W., Sutton,
C., Gehrmann, S., Schuh, P., Shi, K., Tsvyashchenko,
S., Maynez, J., Rao, A., Barnes, P., Tay, Y ., Shazeer,
N., Prabhakaran, V ., Reif, E., Du, N., Hutchinson, B.,
Pope, R., Bradbury, J., Austin, J., Isard, M., Gur-Ari,
G., Yin, P., Duke, T., Levskaya, A., Ghemawat, S., Dev,S., Michalewski, H., Garcia, X., Misra, V ., Robinson,
K., Fedus, L., Zhou, D., Ippolito, D., Luan, D., Lim,
H., Zoph, B., Spiridonov, A., Sepassi, R., Dohan, D.,
Agrawal, S., Omernick, M., Dai, A. M., Pillai, T. S., Pel-
lat, M., Lewkowycz, A., Moreira, E., Child, R., Polozov,
O., Lee, K., Zhou, Z., Wang, X., Saeta, B., Diaz, M.,
Firat, O., Catasta, M., Wei, J., Meier-Hellstern, K., Eck,
D., Dean, J., Petrov, S., and Fiedel, N. PaLM: Scaling
language modeling with pathways.J. Mach. Learn. Res.,
24:240:1–240:113, 2023.
Dao, T., Fu, D. Y ., Ermon, S., Rudra, A., and R ´e, C. Flashat-
tention: Fast and memory-efficient exact attention with
io-awareness, 2022. URL https://arxiv.org/ab
s/2205.14135.
DeepSeek-AI. Deepseek-v2: A strong, economical, and
efficient mixture-of-experts language model.CoRR,
abs/2405.04434, 2024. doi: 10.48550/ARXIV.2405.
04434. URL https://doi.org/10.48550/arX
iv.2405.04434.
Deng, C., Sun, L., Jiang, J., Zeng, Y ., Wu, X., Zhao, W.,
Xiao, Q., Wang, J., Li, H., Chen, L., Ni, L. M., Zhang, H.,
and Wang, J. PLM: efficient peripheral language models
hardware-co-designed for ubiquitous computing.CoRR,
abs/2503.12167, 2025. doi: 10.48550/ARXIV.2503.12
167. URL https://doi.org/10.48550/arXiv
.2503.12167.
Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D.
GPTQ: accurate post-training quantization for genera-
tive pre-trained transformers.CoRR, abs/2210.17323,
2022. doi: 10.48550/ARXIV.2210.17323. URL https:
//doi.org/10.48550/arXiv.2210.17323.
Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M.,
Song, D., and Steinhardt, J. Measuring massive multitask
language understanding. In9th International Conference
on Learning Representations, ICLR 2021, Virtual Event,
Austria, May 3-7, 2021. OpenReview.net, 2021. URL
https://openreview.net/forum?id=d7KB
jmI3GmQ.
Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E.,
Cai, T., Rutherford, E., de Las Casas, D., Hendricks,
L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E.,
Millican, K., van den Driessche, G., Damoc, B., Guy,
A., Osindero, S., Simonyan, K., Elsen, E., Rae, J. W.,
Vinyals, O., and Sifre, L. Training compute-optimal
large language models.CoRR, abs/2203.15556, 2022.
doi: 10.48550/ARXIV.2203.15556. URL https:
//doi.org/10.48550/arXiv.2203.15556.
Hu, S., Tu, Y ., Han, X., He, C., Cui, G., Long, X., Zheng,
Z., Fang, Y ., Huang, Y ., Zhao, W., Zhang, X., Thai, Z. L.,
Zhang, K., Wang, C., Yao, Y ., Zhao, C., Zhou, J., Cai,
9

## Page 10

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
J., Zhai, Z., Ding, N., Jia, C., Zeng, G., Li, D., Liu, Z.,
and Sun, M. Minicpm: Unveiling the potential of small
language models with scalable training strategies.CoRR,
abs/2404.06395, 2024. doi: 10.48550/ARXIV.2404.06
395. URL https://doi.org/10.48550/arXiv
.2404.06395.
Jiang, Y ., Fu, Y ., Huang, Y ., Nie, P., Lu, Z., Xue, L., He, C.,
Sit, M.-K., Xue, J., Dong, L., Miao, Z., Du, D., Xu, T.,
Zou, K., Ponti, E., and Mai, L. Moe-cap: Benchmarking
cost, accuracy and performance of sparse mixture-of-
experts systems, 2025. URL https://arxiv.org/
abs/2412.07067.
Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B.,
Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and
Amodei, D. Scaling laws for neural language models.
CoRR, abs/2001.08361, 2020. URL https://arxiv.
org/abs/2001.08361.
Li, C., Wong, C., Zhang, S., Usuyama, N., Liu, H., Yang, J.,
Naumann, T., Poon, H., and Gao, J. Llava-med: Training
a large language-and-vision assistant for biomedicine
in one day. In Oh, A., Naumann, T., Globerson, A.,
Saenko, K., Hardt, M., and Levine, S. (eds.),Advances
in Neural Information Processing Systems 36: Annual
Conference on Neural Information Processing Systems
2023, NeurIPS 2023, New Orleans, LA, USA, December
10 - 16, 2023, 2023. URL http://papers.nip
s.cc/paper_files/paper/2023/hash/5ab
cdf8ecdcacba028c6662789194572-Abstrac
t-Datasets_and_Benchmarks.html.
Liang, P., Bommasani, R., Lee, T., Tsipras, D., Soylu, D.,
Yasunaga, M., Zhang, Y ., Narayanan, D., Wu, Y ., Ku-
mar, A., Newman, B., Yuan, B., Yan, B., Zhang, C.,
Cosgrove, C., Manning, C. D., R ´e, C., Acosta-Navas,
D., Hudson, D. A., Zelikman, E., Durmus, E., Ladhak,
F., Rong, F., Ren, H., Yao, H., Wang, J., Santhanam,
K., Orr, L. J., Zheng, L., Y ¨uksekg ¨on¨ul, M., Suzgun, M.,
Kim, N., Guha, N., Chatterji, N. S., Khattab, O., Hen-
derson, P., Huang, Q., Chi, R., Xie, S. M., Santurkar,
S., Ganguli, S., Hashimoto, T., Icard, T., Zhang, T.,
Chaudhary, V ., Wang, W., Li, X., Mai, Y ., Zhang, Y .,
and Koreeda, Y . Holistic evaluation of language models.
Trans. Mach. Learn. Res., 2023, 2023. URL https:
//openreview.net/forum?id=iO4LZibEqW.
Lin, J., Tang, J., Tang, H., Yang, S., Chen, W., Wang,
W., Xiao, G., Dang, X., Gan, C., and Han, S. AWQ:
activation-aware weight quantization for on-device LLM
compression and acceleration. In Gibbons, P. B., Pekhi-
menko, G., and Sa, C. D. (eds.),Proceedings of the Sev-
enth Annual Conference on Machine Learning and Sys-
tems, MLSys 2024, Santa Clara, CA, USA, May 13-16,
2024. mlsys.org, 2024. URL https://proceedings.mlsys.org/paper_files/paper/2024/h
ash/42a452cbafa9dd64e9ba4aa95cc1ef21
-Abstract-Conference.html.
Liu, Z., Zhao, C., Iandola, F. N., Lai, C., Tian, Y ., Fedorov,
I., Xiong, Y ., Chang, E., Shi, Y ., Krishnamoorthi, R., Lai,
L., and Chandra, V . Mobilellm: Optimizing sub-billion
parameter language models for on-device use cases. In
Forty-first International Conference on Machine Learn-
ing, ICML 2024, Vienna, Austria, July 21-27, 2024, 2024.
URL https://openreview.net/forum?id=
EIGbXbxcUQ.
Pope, R., Douglas, S., Chowdhery, A., Devlin, J., Bradbury,
J., Heek, J., Xiao, K., Agrawal, S., and Dean, J. Efficiently
scaling transformer inference. In Song, D., Carbin, M.,
and Chen, T. (eds.),Proceedings of the Sixth Conference
on Machine Learning and Systems, MLSys 2023, Miami,
FL, USA, June 4-8, 2023. mlsys.org, 2023. URL https:
//proceedings.mlsys.org/paper_files/
paper/2023/hash/c4be71ab8d24cdfb45e3
d06dbfca2780-Abstract-mlsys2023.html.
Prieto, P. and Abad, P. Edge deployment of small language
models, a comprehensive comparison of cpu, GPU and
NPU backends.CoRR, abs/2511.22334, 2025. doi: 10
.48550/ARXIV.2511.22334. URL https://doi.or
g/10.48550/arXiv.2511.22334.
Rajesh, V ., Jodhpurkar, O., Anbuselvan, P., Singh, M.,
Jallepali, A., Godbole, S., Sharma, P. K., and Shrivas-
tava, H. Production-grade local LLM inference on apple
silicon: A comparative study of mlx, mlc-llm, ollama,
llama.cpp, and pytorch MPS.CoRR, abs/2511.05502,
2025. doi: 10.48550/ARXIV.2511.05502. URL https:
//doi.org/10.48550/arXiv.2511.05502.
Reddi, V . J., Cheng, C., Kanter, D., Mattson, P.,
Schmuelling, G., Wu, C., Anderson, B., Breughe, M.,
Charlebois, M., Chou, W., Chukka, R., Coleman, C.,
Davis, S., Deng, P., Diamos, G., Duke, J., Fick, D.,
Gardner, J. S., Hubara, I., Idgunji, S., Jablin, T. B.,
Jiao, J., John, T. S., Kanwar, P., Lee, D., Liao, J.,
Lokhmotov, A., Massa, F., Meng, P., Micikevicius, P.,
Osborne, C., Pekhimenko, G., Rajan, A. T. R., Se-
queira, D., Sirasao, A., Sun, F., Tang, H., Thomson,
M., Wei, F., Wu, E., Xu, L., Yamada, K., Yu, B., Yuan,
G., Zhong, A., Zhang, P., and Zhou, Y . Mlperf infer-
ence benchmark.CoRR, abs/1911.02549, 2019. URL
http://arxiv.org/abs/1911.02549.
Sun, L., Jiang, J., Deng, C., Wu, X., Zhang, H., Chen, L., Ni,
L. M., and Wang, J. GTA: grouped-head latent attention.
CoRR, abs/2506.17286, 2025. doi: 10.48550/ARXIV.2
506.17286. URL https://doi.org/10.48550
/arXiv.2506.17286.
10

## Page 11

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
Team, G. Gemma: Open models based on gemini research
and technology.CoRR, abs/2403.08295, 2024a. doi:
10.48550/ARXIV.2403.08295. URL https://doi.
org/10.48550/arXiv.2403.08295.
Team, L. The llama 3 herd of models.CoRR,
abs/2407.21783, 2024b. doi: 10.48550/ARXIV.240
7.21783. URL https://doi.org/10.48550/a
rXiv.2407.21783.
Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi,
A., Babaei, Y ., Bashlykov, N., Batra, S., Bhargava, P.,
Bhosale, S., Bikel, D., Blecher, L., Canton-Ferrer, C.,
Chen, M., Cucurull, G., Esiobu, D., Fernandes, J., Fu,
J., Fu, W., Fuller, B., Gao, C., Goswami, V ., Goyal, N.,
Hartshorn, A., Hosseini, S., Hou, R., Inan, H., Kardas,
M., Kerkez, V ., Khabsa, M., Kloumann, I., Korenev, A.,
Koura, P. S., Lachaux, M., Lavril, T., Lee, J., Liskovich,
D., Lu, Y ., Mao, Y ., Martinet, X., Mihaylov, T., Mishra,
P., Molybog, I., Nie, Y ., Poulton, A., Reizenstein, J.,
Rungta, R., Saladi, K., Schelten, A., Silva, R., Smith,
E. M., Subramanian, R., Tan, X. E., Tang, B., Taylor,
R., Williams, A., Kuan, J. X., Xu, P., Yan, Z., Zarov, I.,
Zhang, Y ., Fan, A., Kambadur, M., Narang, S., Rodriguez,
A., Stojnic, R., Edunov, S., and Scialom, T. Llama 2:
Open foundation and fine-tuned chat models.CoRR,
abs/2307.09288, 2023. doi: 10.48550/ARXIV.2307.09
288. URL https://doi.org/10.48550/arXiv
.2307.09288.
Williams, S., Waterman, A., and Patterson, D. A. Roofline:
an insightful visual performance model for multicore
architectures.Commun. ACM, 52(4):65–76, 2009. doi:
10.1145/1498765.1498785. URL https://doi.or
g/10.1145/1498765.1498785.
Xiao, C., Cai, J., Zhao, W., Zeng, G., Lin, B., Zhou, J.,
Zheng, Z., Han, X., Liu, Z., and Sun, M. Densing law of
llms.CoRR, abs/2412.04315, 2024. doi: 10.48550/ARX
IV.2412.04315. URL https://doi.org/10.485
50/arXiv.2412.04315.
Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B.,
Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu,
J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang,
K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M.,
Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Xia, T., Ren,
X., Ren, X., Fan, Y ., Su, Y ., Zhang, Y ., Wan, Y ., Liu, Y .,
Cui, Z., Zhang, Z., and Qiu, Z. Qwen2.5 technical report.
CoRR, abs/2412.15115, 2024. doi: 10.48550/ARXIV.2
412.15115. URL https://doi.org/10.48550/a
rXiv.2412.15115.
Yao, Y ., Yu, T., Zhang, A., Wang, C., Cui, J., Zhu, H., Cai,
T., Li, H., Zhao, W., He, Z., Chen, Q., Zhou, H., Zou, Z.,
Zhang, H., Hu, S., Zheng, Z., Zhou, J., Cai, J., Han, X.,Zeng, G., Li, D., Liu, Z., and Sun, M. Minicpm-v: A GPT-
4V level MLLM on your phone.CoRR, abs/2408.01800,
2024. doi: 10.48550/ARXIV.2408.01800. URL https:
//doi.org/10.48550/arXiv.2408.01800.
Yu, T., Wang, Z., Wang, C., Huang, F., Ma, W., He, Z., Cai,
T., Chen, W., Huang, Y ., Zhao, Y ., Xu, B., Cui, J., Xu, Y .,
Ruan, L., Zhang, L., Liu, H., Tang, J., Liu, H., Guo, Q.,
Hu, W., He, B., Zhou, J., Cai, J., Qi, J., Guo, Z., Chen,
C., Zeng, G., Li, Y ., Cui, G., Ding, N., Han, X., Yao, Y .,
Liu, Z., and Sun, M. Minicpm-v 4.5: Cooking efficient
mllms via architecture, data, and training recipe, 2025.
URLhttps://arxiv.org/abs/2509.18154.
Zhang, P., Zeng, G., Wang, T., and Lu, W. Tinyl-
lama: An open-source small language model.CoRR,
abs/2401.02385, 2024. doi: 10.48550/ARXIV.2401.02
385. URL https://doi.org/10.48550/arXiv
.2401.02385.
Zheng, Y ., Chen, Y ., Qian, B., Shi, X., Shu, Y ., and Chen,
J. A review on edge large language models: Design,
execution, and applications.ACM Comput. Surv., 57
(8):209:1–209:35, 2025. doi: 10.1145/3719664. URL
https://doi.org/10.1145/3719664.
11

## Page 12

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
A. Related Work
Trends in Efficient LLMs.With the rapid proliferation of open-source foundation models such as the Llama series
(Touvron et al., 2023) (Team, 2024b) , scaling laws (Kaplan et al., 2020) have revealed a positive correlation between
model capability and parameter scale, driving the trend toward increasingly massive models. However, driven by concerns
regarding data privacy, latency sensitivity, and operational costs, deploying LLMs on on-device hardware has become an
imperative, catalyzing a fundamental paradigm shift from cloud-based inference to edge computing (Zheng et al., 2025).
To accommodate constrained on-device resources, compact architectures optimized for mobile scenarios have emerged,
such as TinyLlama (Zhang et al., 2024) and MobileLLM (Liu et al., 2024) . Concurrently, mature post-training quantization
techniques like GPTQ (Frantar et al., 2022) and AWQ (Lin et al., 2024) have significantly reduced memory footprints and
computational loads. Despite these advancements in model compression, as noted by Alizadeh et al. (Alizadeh et al., 2024) ,
the performance bottleneck for generative LLMs on edge devices is typically strictly governed by memory bandwidth rather
than compute capacity (the ”Memory Wall”). The limitations of these hardware characteristics necessitate a more profound,
hardware-aware analysis of inference performance, moving beyond a mere evaluation of inference metrics.
General Capabilities Benchmarking.Existing evaluation frameworks for Large Language Models (LLMs) predominantly
focus on their cognitive and linguistic capabilities. Among these, MMLU (Hendrycks et al., 2021) serves as a foundational
benchmark, covering 57 tasks across diverse domains such as mathematics, history, and law to assess multi-task problem-
solving abilities. Furthermore, HELM (Liang et al., 2023)implements a holistic, multi-metric approach to evaluate dozens of
mainstream LLMs, offering a comprehensive view of their semantic performance across various core scenarios.
While these benchmarks are essential for measuring ”model intelligence,” they typically treat the inference process as
a system-agnostic black box. Consequently, they prioritize task-specific accuracy while disregarding the underlying
computational overhead, such as system efficiency, real-time resource consumption, and hardware utilization. For resource-
constrained edge devices, relying solely on these capability-centric metrics is insufficient to guarantee deployment viability,
underscoring the urgent need for a more granular, system-level evaluation perspective.
System Efficiency and On-Device Benchmarking.To bridge the gap between algorithmic capability and physical
deployment, recent research has shifted toward system-level efficiency. Industry standards like MLPerf (Reddi et al., 2019)
establish rigorous protocols for evaluating inference systems across heterogeneous backends, while FrugalGPT (Chen et al.,
2024) introduces a cost-centric perspective, analyzing the trade-offs between API expenditures and model performance.
Specifically for on-device environments, several recent studies have provided in-depth empirical analyses. Edge Deployment
of SLMs (Prieto & Abad, 2025) conducts a comprehensive evaluation across mobile CPUs, GPUs, and NPUs, highlighting
the superior energy efficiency of NPUs and the necessity of bandwidth normalization for cross-architecture fairness. Within
specific hardware ecosystems, Local LLM Inference on Apple Silicon (Rajesh et al., 2025) systematically compares runtimes
such as MLX and MLC-LLM, characterizing their throughput and latency within unified memory architectures. Furthermore,
for sparse models, MoE-CAP (Jiang et al., 2025) identifies the inherent trade-offs between Cost, Accuracy, and Performance
(CAP), proposing sparsity-aware utilization metrics.
Despite these advancements, a fundamental limitation persists: existing benchmarks primarily report observed execution
metrics (e.g., TTFT, throughput) without contextualizing them against the theoretical limits of the hardware. Such empirical
observations often fail to decouple software-level optimizations from inherent hardware capabilities, potentially leading
to biased conclusions regarding architectural efficiency. Our work addresses this gap by leveraging the Roofline Model
(Williams et al., 2009). By characterizing performance as a function of arithmetic intensity and hardware ceilings (peak
compute and memory bandwidth), our framework enables a more transparent and fair benchmarking process that quantifies
how effectively an LLM implementation approaches its hardware’s theoretical potential.
B. Future Work
While this study provides an extensive empirical characterization of mainstream small language models across diverse
hardware tiers, we identify several strategic avenues for future exploration. Primarily, we intend to expand the taxonomy
of our benchmarking framework to incorporate a broader range of attention mechanisms. Although our current analysis
provides a comprehensive evaluation of critical patterns such as MLA andGQA , we aim to adapt our methodology to
emerging structural variants and hybrid architectures to ensure the continued relevance of our operational intensity ( OI)
insights across the evolving architectural landscape.
12

## Page 13

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
A significant frontier lies in the analytical characterization of Mixture of Experts (MoE) architectures. Quantifying the
theoretical FLOPS for MoE models during dynamic inference remains a formidable challenge because the stochastic
nature of token routing complicates the estimation of the execution ceiling. While the current work focuses on dense
architectures, extending the Roofline model to account for sparse activation regimes will be essential for identifying the
unique physical constraints and efficiency traps inherent in localized MoE deployment.
Finally, we plan to broaden the experimental scope across both hardware and software dimensions. On the hardware side, we
aim to scale our testing to a more diverse array of heterogeneous edge devices to further validate the cross platform robustness
of our findings. On the software side, we recognize that different inference engines, such as TensorRT LLM, vLLM, and
ONNX Runtime, introduce substantial variance in performance. Investigating how these software stacks influence OIand
realized throughput will provide a more holistic understanding of the entire deployment pipeline, facilitating a more effective
transition toward true hardware software co design.
C. Experimental Details
The code for this work is available athttps://github.com/banbu-ai/roofline_bench.
C.1. Evaluated Model Architectures
To investigate the scaling laws and efficiency of intelligence on-device, we select a diverse set of models that represent
current trends in localized AI. These models are evaluated across two categories based on the computational tiers of the
hardware substrates:
•Edge-Scale Models: For the most resource-limited tier represented by the Raspberry Pi 5, we evaluate ultra-compact
models including the Pythia (160M, 410M), Qwen2.5-0.5B, and SmolLM2 (135M, 360M) families. These models
primarily utilize Q8 0 quantization to fit within the constraints of general-purpose CPUs.
•Mobile-Class Models: For performance-oriented platforms such as the Apple M1 Pro, RTX 3070 Ti Laptop, and
Jetson Orin Nano Super, we employ models ranging from 0.6B to 1.8B parameters. This group features diverse types
of attention to observe their impact on the potential for inference, including Grouped-Query Attention (GQA) for
Qwen2.5-1.5B, Llama-3.2-1B, Qwen3-0.6B, and Fox-1-1.6B; Multi-Head Attention (MHA) for SmolLM2-1.7B; and
Multi-Head Latent Attention (MLA) for PLM-1.8B.
All architectures are tested under four distinct scenarios—SISO, LISO, SILO, and LILO—to capture performance dynamics
across varying sequence lengths. This selection provides a rigorous basis for the co-design of models and hardware by
identifying how structural primitives like attention mechanisms interact with physical memory boundaries.
C.2. Evaluated Hardware Substrates
To ensure a rigorous evaluation across various tiers of computation, we conduct our analysis on four distinct hardware
platforms. These substrates represent a broad spectrum of architectural designs and power envelopes, reflecting the diverse
physical constraints encountered in on-device intelligence. The selection includes the Apple M1 Pro (a unified system on a
chip), the RTX 3070 Ti Laptop (a discrete GPU for mobile workstations), the Jetson Orin Nano Super (a dedicated module
for AI at the edge), and the Raspberry Pi 5 (a ubiquitous platform based on a general CPU).
Table 1 summarizes the specifications for heterogeneous platforms, ranging from Discrete GPUs to General CPUs. By
incorporating theoretical and measured values for peak performance ( π) and memory bandwidth ( β), these metrics establish
the boundaries for calculating Relative Inference Potential ( Φ). Such characterization is essential to identify requirements
for effective hardware-software co-design in intelligence on-device.
C.3. Experimental Setting
Hardware Characterization and Roofline Benchmarking.To establish the empirical bounds for the Roofline model, we
evaluate the peak throughput and memory bandwidth across a heterogeneous suite of devices, including macOS, Windows,
NVIDIA Jetson Orin Nano, and Raspberry Pi platforms. We conduct synthetic benchmarks involving large-scale matrix
addition and multiplication using the PyTorch framework. To ensure optimal performance across architectures, we utilize
13

## Page 14

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
Table 1.Hardware Specifications and Performance Metrics across Heterogeneous Platforms. Theo denotes the theoretical peak value
provided by the manufacturer, while Meas denotes the measured peak performance under benchmarking workloads. Notably, the measured
FP16 peak performance for NVIDIA RTX 30 series GPUs and Jetson Orin Nano significantly exceeds their theoretical FP32 peak values
because of the activation of specialized hardware such as Tensor Cores during evaluation. Values marked with †are in GFLOPS, while all
other compute metrics are in TFLOPS.
Chip Architecture Bandwidth (GB/s) FP16 Peak (TFLOPS) FP32 Peak (TFLOPS)
Theo / Meas Theo / Meas Theo / Meas
NVIDIA RTX 3090 Discrete GPU 936.20 / 560.02 35.58 / 66.20 35.58 / 24.28
NVIDIA RTX 3070 Ti Laptop Discrete GPU 448.00 / 217.00 16.60 / 31.76 16.60 / 9.51
Apple M1 Pro Unified SoC 204.80 / 120.03 — / 4.61 5.20 / 4.31
Jetson Orin Nano Super 8G Edge AI Module 102.00 / 59.40 4.178 / 9.56 2.089 / 1.34
Raspberry Pi 5 General CPU 17.10 / 3.98 — / 1.48†153.60†/ 78.56†
the CUDA backend for NVIDIA hardware and the Metal Performance Shaders (MPS) backend for Apple Silicon devices.
These measurements provide the hardware-specific constants required for our performance analysis.
FLOPs Calculation Methodology.To evaluate operational intensity ( OI) and performance within the Roofline model, we
estimate the floating point operations ( FLOPs ) associated with each inference configuration. This estimation framework
accounts for linear projections, attention mechanisms, and KV cache management by leveraging key architectural parameters
including the hidden dimension ( h), the number of query heads ( nh), and the number of key-value heads ( nkv). The specific
estimation logic Sun et al. (2025), with detailed formulas provided in Table 2. The resulting FLOPs values serve as an
approximated baseline for derivingOIand throughput metrics.
Table 2.Comparison of computational complexity and memory requirements for different attention mechanisms. His the hidden
dimension, Nis the sequence length, nq, nk, nv, ncare the number of query, key, value, and latent value heads, respectively; dhis the
per-head dimension, dlis the latent dimension. For MLA-based structures, dcdenotes the KV compression dimension, while drope and
dnope represent the rotary and non-rotary components of the embedding dimension, respectively.
Attention KV Cache per LayerComputation per LayerExpressivityAttention Linear
MHA2n hdhN2n hdhN24NH2Strong
GQA2n kdhN2n hdhN22NH2+ 2n kdhNHModerate
MLA(d c+d rope)N n h(drope+ 2d nope)N2
(dc+d rope)H+n h(drope+d nope)H+ 2n hdldnope+H2
NStrong
GV A(H+n kdh)N(n qdh+n hdh)N22NH2+ 2n kdhNHModerate
GHA(n kdh+n vdh)N(n qdh+n hdh)N2NH2+n qdhNH+n kdhNH+n vdhNHWeak
GTA(n kdh+n cdl)N n q(dk+d l)N22NH2+ (n qdh+n kdh+n cdl+d l)NHStrong
Inference Memory Monitoring.We employ platform-specific strategies to monitor memory utilization during llama-bench
execution. For devices with Unified Memory Architectures (UMA)—including macOS and NVIDIA Jetson—as well
as CPU-based platforms like the Raspberry Pi 5, we track physical memory usage by mapping the target process name
to its Process ID (PID) and sampling its resident set size (RSS) during inference. On Windows systems equipped with
consumer-grade discrete GPUs, monitoring process-specific VRAM presents a technical challenge. Due to the Windows
Display Driver Model (WDDM) constraints on consumer hardware—which lack support for the Tesla Compute Cluster
(TCC) mode required for granular per-process accounting—standard utilities such as nvidia-smi and pynvml often return
null values for individual process memory. To circumvent this, we implement an exclusive-access monitoring strategy: by
parsing nvidia-smi XML outputs, we ensure the discrete GPU is dedicated solely to the target PID during benchmarking.
Under these isolated conditions, the total VRAM utilization of the device serves as a precise proxy for the inference memory
footprint.
Inference Workflow and Data Acquisition.For each experimental trial, we execute llama-bench with parameterized
input/output sequence lengths and thread counts. To ensure temporal synchronization across data streams, each session is
assigned a unique timestamp. During execution, our framework concurrently monitors the memory footprint and records the
runtime metrics. The resulting inference logs, formatted in JSON as per llama-bench specifications, capture comprehensive
metadata including device architecture, backend configurations, model types, and specific command-line arguments. Upon
completion of each inference cycle, our framework performs automated post-hoc analysis and computational evaluation
14

## Page 15

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
based on these synchronized logs and memory traces.
Memory Management and Offloading Strategy.While the inference framework allows for explicit control over offloading,
we incorporate this strategy to handle scenarios where model depth exceeds the 8 GB VRAM capacity of the RTX 3070Ti
Laptop. The observed linear performance degradation in specific experimental trials is primarily attributed to the automated
offloading mechanism in llama.cpp. When the GPU-resident buffers are insufficient to accommodate weight tensors, the
framework dynamically reallocates storage to system memory. This allocation follows a tiered priority hierarchy: ACCEL
(accelerated CPU regions) →GPU Host (pinned memory for optimized DMA transfer) →CPU Extra →Standard CPU
(general-purpose system memory).
D.Comprehensive Benchmarking Analysis of Model Architectures on Heterogeneous Hardware
Platforms
This appendix provides a comprehensive suite of benchmarking results to substantiate the empirical findings and theoretical
analyses presented in the main text. We detail the performance profiles of various large language model architectures
including Multi head Latent Attention and trainable sparse attention across a spectrum of heterogeneous hardware platforms
ranging from high performance accelerators to edge side SoCs. The presented data encompass a wide array of quantization
precisions and operational scenarios while offering granular insights into the relationship between architectural design, layer
wise operational intensity ( OI), and hardware resource saturation. By providing these extended datasets, we aim to offer a
rigorous validation of the scaling dynamics and utilization equity discussed throughout the study.
D.1. Analysis and Findings on the Optimality Gap
The vertical distance between the actual operating point and the theoretical Roofline ceiling in the performance chart (as
illustrated across different devices in Figure 7). This gap quantifies the ”left-on-the-table” performance and serves as a direct
indicator of optimization headroom.
101
100101102
Operational Intensity (FLOPs/Byte)10111012Performance (FLOPs/sec)Apple M1 Pro
Theoretical
Measured
Theo Ridge: 25.39
Real Ridge: 35.83
Bandwidth Gap
Computation Gap
101
100101102
Operational Intensity (FLOPs/Byte)101110121013Performance (FLOPs/sec)RTX 3070Ti Laptop
Theoretical
Measured
Theo Ridge: 37.05
Real Ridge: 43.18
Bandwidth Gap
Computation Gap
101
100101102
Operational Intensity (FLOPs/Byte)101010111012Performance (FLOPs/sec)Jetson Orin Nano Super 8G
Theoretical
Measured
Theo Ridge: 20.48
Real Ridge: 22.56
Bandwidth Gap
Computation Gap
101
100101102
Operational Intensity (FLOPs/Byte)10910101011Performance (FLOPs/sec)Raspberry Pi 5
Theoretical
Measured
Theo Ridge: 9.00
Real Ridge: 20.13
Bandwidth Gap
Computation Gap
Figure 7.Performance gap analysis using Roofline models on different hardware. Green and purple arrows represent Bandwidth and
Compute Gaps between theoretical (solid blue) and measured (dashed red) performance. Vertical lines mark the theoretical and real-world
ridge points.
D.2. Extended Benchmark Results: Prefilling and Decoding TPS
Table 3.Detailed inference throughput (TPS) on Raspberry Pi 5 across various scenarios. Data are presented as Prefilling TPS / Decoding
TPS.
Model Precision SISO LISO SILO LILO
pythia-160m Q8 0 149 / 70 63 / 71 150 / 25 64 / 27
pythia-410m Q8 0 43 / 26 20 / 26 43 / 9 21 / 9
Qwen2.5-0.5B-Instruct Q8 0 38 / 20 22 / 20 38 / 14 22 / 14
SmolLM2-135M-Instruct Q8 0 115 / 65 42 / 65 115 / 29 42 / 29
SmolLM2-360M-Instruct Q8 0 41 / 26 20 / 26 42 / 14 20 / 14
15

## Page 16

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
Table 4.Cross device inference throughput (TPS) comparison for Qwen2.5-1.5B-Instruct in Q8 0 precision. Data are presented as
Prefilling TPS / Decoding TPS.
Model Precision Device SISO LISO SILO LILO
Qwen2.5-1.5B-Instruct Q8 0M1Pro 1349 / 63 1204 / 62 1349 / 52 1204 / 52
RTX 3070Ti L 4465 / 127 5286 / 130 4295 / 118 5256 / 119
Jetson Orin 1203 / 25 1075 / 25 1281 / 23 1073 / 23
Table 5.Detailed inference throughput (TPS) on Apple M1 Pro across various scenarios. Data are presented as Prefilling TPS / Decoding
TPS.
Model Precision SISO LISO SILO LILO
Qwen2.5-1.5B-InstructFP16 1424 / 50 1273 / 50 1423 / 43 1273 / 43
Q80 1349 / 63 1204 / 62 1349 / 52 1204 / 52
Q4KM 1199 / 90 1078 / 90 1200 / 69 1078 / 69
Qwen3-0.6BFP16 3168 / 102 2140 / 102 3164 / 69 2143 / 71
Q80 2984 / 105 2072 / 105 3018 / 72 2070 / 72
Q4KM 2765 / 100 1939 / 160 2770 / 94 1942 / 94
Llama-3.2-1B-InstructFP16 2002 / 65 1594 / 65 2000 / 51 1594 / 51
Q80 1894 / 92 1509 / 92 1894 / 67 1509 / 67
Q4KM 1691 / 133 1369 / 133 1691 / 86 1370 / 85
PLM-1.8B-InstructFP16 1114 / 40 939 / 40 1113 / 30 939 / 30
Q80 1053 / 55 891 / 55 1053 / 38 890 / 38
Q4KM 942 / 75 812 / 75 942 / 46 812 / 46
Fox-1-1.6BFP16 1522 / 47 1239 / 47 1521 / 38 1239 / 38
Q80 1438 / 63 1174 / 63 1438 / 49 1174 / 49
Q4KM 1295 / 87 1070 / 88 1295 / 62 1071 / 62
SmolLM2-1.7B-InstructFP16 1212 / 46 970 / 46 1212 / 35 971 / 35
Q80 1139 / 64 918 / 64 1138 / 46 916 / 46
Q4KM 1002 / 90 827 / 90 1002 / 58 827 / 58
16

## Page 17

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
D.3. Extended Empirical Characterization: Comprehensive Roofline Profiles
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Apple M1 Pro
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
Qwen2.5-Instruct FP16
Qwen2.5-Instruct Q8_0
Qwen2.5-Instruct Q4_K_M
Llama-3.2-Instruct FP16
Llama-3.2-Instruct Q8_0
Llama-3.2-Instruct Q4_K_M
PLM-Instruct FP16
PLM-Instruct Q8_0
PLM-Instruct Q4_K_M
Qwen3 FP16
Qwen3 Q8_0
Qwen3 Q4_K_M
Fox-1 FP16
Fox-1 Q8_0
Fox-1 Q4_K_M
SmolLM2-Instruct FP16
SmolLM2-Instruct Q8_0
SmolLM2-Instruct Q4_K_M
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
RTX 3070Ti Laptop
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 43.18
Qwen2.5-Instruct FP16
Qwen2.5-Instruct Q8_0
Qwen2.5-Instruct Q4_K_M
Llama-3.2-Instruct FP16
Llama-3.2-Instruct Q8_0
Llama-3.2-Instruct Q4_K_M
PLM-Instruct FP16
PLM-Instruct Q8_0
PLM-Instruct Q4_K_M
Qwen3 FP16
Qwen3 Q8_0
Qwen3 Q4_K_M
Fox-1 FP16
Fox-1 Q8_0
Fox-1 Q4_K_M
SmolLM2-Instruct FP16
SmolLM2-Instruct Q8_0
SmolLM2-Instruct Q4_K_M
102
100102104106
Operational Intensity (FLOPs/Byte)106107108109101010111012Performance (FLOPs/sec)
Jetson Orin Nano Super 8G
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 22.56
Qwen2.5-Instruct FP16
Qwen2.5-Instruct Q8_0
Qwen2.5-Instruct Q4_K_M
Llama-3.2-Instruct FP16
Llama-3.2-Instruct Q8_0
Llama-3.2-Instruct Q4_K_M
PLM-Instruct FP16
PLM-Instruct Q8_0
PLM-Instruct Q4_K_M
Qwen3 FP16
Qwen3 Q8_0
Qwen3 Q4_K_M
Fox-1 FP16
Fox-1 Q8_0
Fox-1 Q4_K_M
SmolLM2-Instruct FP16
SmolLM2-Instruct Q8_0
SmolLM2-Instruct Q4_K_M
102
100102104106
Operational Intensity (FLOPs/Byte)10610710810910101011Performance (FLOPs/sec)
Raspberry Pi 5
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 20.13
Qwen2.5-0.5B-Instruct Q8_0
SmolLM2-135M-Instruct Q8_0
pythia-160m Q8_0
SmolLM2-360M-Instruct Q8_0
pythia-410m Q8_0
Figure 8.Hardware Specifications and Performance Metrics across Heterogeneous Platforms. Theo denotes the theoretical peak value
provided by the manufacturer, while Meas denotes the measured peak performance under benchmarking workloads. Notably, the measured
FP16 peak performance for NVIDIA RTX 30 series GPUs and Jetson Orin Nano significantly exceeds their theoretical FP32 peak values
because of the activation of specialized hardware such as Tensor Cores during evaluation. Values marked with †are in GFLOPS, while all
other compute metrics are in TFLOPS.
17

## Page 18

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
1 2 3 4 50.20.40.60.81.0Performance(FLOPs/sec)1e10 FP16 | LILO
20 40 60 80 1000.20.40.60.81.01.21.41.61e11 FP16 | LISO
0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.080.20.40.60.81.01.21.41.61e8 FP16 | SILO
0.5 1.0 1.5 2.0 2.5 3.01234561e9 FP16 | SISO
2 4 6 8 100.20.40.60.81.01.2Performance(FLOPs/sec)1e10 Q8_0 | LILO
25 50 75 100 125 150 1750.20.40.60.81.01.21.41.61e11 Q8_0 | LISO
0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.160.250.500.751.001.251.501.752.001e8 Q8_0 | SILO
1 2 3 4 5 6 7123456781e9 Q8_0 | SISO
2 4 6 8 10 12 14 16
Operational Intensity (FLOPs/Byte)0.20.40.60.81.01.21.4Performance(FLOPs/sec)1e10 Q4_K_M | LILO
50 100 150 200 250
Operational Intensity (FLOPs/Byte)0.20.40.60.81.01.21.41.61e11 Q4_K_M | LISO
0.05 0.10 0.15 0.20 0.25
Operational Intensity (FLOPs/Byte)0.51.01.52.02.51e8 Q4_K_M | SILO
2 4 6 8 10 12 14
Operational Intensity (FLOPs/Byte)0.20.40.60.81.01e10 Q4_K_M | SISOApple M1 Pro - Model: PLM-Instruct
Layers: 2
Layers: 4
Layers: 6
Layers: 8
Layers: 10
Layers: 12
Layers: 14
Layers: 16
Layers: 18
Layers: 20
Layers: 22
Layers: 24
Layers: 26
Layers: 28
Layers: 30
Layers: 32
Layers: 34
Layers: 36
Layers: 38
Layers: 40
Layers: 42
Layers: 44
Layers: 46
Layers: 48
Layers: 50
Layers: 52
Layers: 54
Layers: 56
Layers: 58
Layers: 60
Layers: 62
Layers: 64
Figure 9.Layer-wise operational intensity trends for PLM-Instruct on Apple M1 Pro. This evaluation monitors the progression of
computational intensity throughout the layer stack under diverse inference workloads. The profiles reveal that while most layers maintain
a stable intensity, certain points exhibit a steep drop in metrics. Such abrupt decreases indicate that the model has reached a performance
bottleneck, where the throughput is increasingly restricted by the underlying hardware architecture, leading to a noticeable reduction in
layer-wise efficiency.
18

## Page 19

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
1 2 3 4 5 6 70.250.500.751.001.251.501.752.002.25Performance(FLOPs/sec)1e10 FP16 | LILO
20 40 60 80 100 1200.51.01.52.02.53.03.54.01e11 FP16 | LISO
0.02 0.04 0.06 0.08 0.10 0.120.51.01.52.02.53.03.51e8 FP16 | SILO
1.0 1.5 2.0 2.5 3.0 3.50.20.40.60.81.01e10 FP16 | SISO
2 4 6 8 10 12 140.51.01.52.02.53.0Performance(FLOPs/sec)1e10 Q8_0 | LILO
50 75 100 125 150 175 200 225123451e11 Q8_0 | LISO
0.05 0.10 0.15 0.20 0.25123451e8 Q8_0 | SILO
2 3 4 5 6 7 80.20.40.60.81.01.21.41e10 Q8_0 | SISO
4 6 8 10 12 14 16 18
Operational Intensity (FLOPs/Byte)0.51.01.52.02.53.03.5Performance(FLOPs/sec)1e10 Q4_K_M | LILO
50 100 150 200 250
Operational Intensity (FLOPs/Byte)123451e11 Q4_K_M | LISO
0.05 0.10 0.15 0.20 0.25 0.30
Operational Intensity (FLOPs/Byte)1234561e8 Q4_K_M | SILO
2 4 6 8 10 12
Operational Intensity (FLOPs/Byte)0.250.500.751.001.251.501.751e10 Q4_K_M | SISORTX 3070 Ti Laptop - Model: PLM-Instruct
Layers: 2
Layers: 4
Layers: 6
Layers: 8
Layers: 10
Layers: 12
Layers: 14
Layers: 16
Layers: 18
Layers: 20
Layers: 22
Layers: 24
Layers: 26
Layers: 28
Layers: 30
Layers: 32
Layers: 34
Layers: 36
Layers: 38
Layers: 40
Layers: 42
Layers: 44
Layers: 46
Layers: 48
Layers: 50
Layers: 52
Layers: 54
Layers: 56
Layers: 58
Layers: 60
Layers: 62
Layers: 64
Figure 10.Layer-wise operational intensity trends for PLM-Instruct on NVIDIA RTX 3070 Ti Laptop. The subplots map the operational
intensity (FLOPs/Byte) against the Transformer layer index across diverse inference workloads. Although the computational intensity is
generally uniform throughout the layer stack, certain data points exhibit a significant downward trend. Such abrupt decreases mark the
onset of hardware-specific bottlenecks, where the data movement or processing constraints lead to a noticeable reduction in layer-wise
efficiency compared to the stable segments of the model.
19

## Page 20

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
0.50 0.75 1.00 1.25 1.50 1.75 2.00 2.250.51.01.52.02.53.03.54.04.5Performance(FLOPs/sec)1e9 FP16 | LILO
10 15 20 25 30 35 40 45 50123456781e10 FP16 | LISO
0.010 0.015 0.020 0.025 0.030 0.03512345671e7 FP16 | SILO
0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.00.250.500.751.001.251.501.752.001e9 FP16 | SISO
2 3 4 5123456Performance(FLOPs/sec)1e9 Q8_0 | LILO
20 30 40 50 60 70 80 90 1000.20.40.60.81.01e11 Q8_0 | LISO
0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.090.20.40.60.81.01e8 Q8_0 | SILO
0.5 1.0 1.5 2.0 2.50.51.01.52.02.53.03.51e9 Q8_0 | SISO
2 3 4 5 6 7 8 9
Operational Intensity (FLOPs/Byte)12345678Performance(FLOPs/sec)1e9 Q4_K_M | LILO
40 60 80 100 120 140 160
Operational Intensity (FLOPs/Byte)0.20.40.60.81.01.21e11 Q4_K_M | LISO
0.04 0.06 0.08 0.10 0.12 0.14 0.16
Operational Intensity (FLOPs/Byte)0.20.40.60.81.01.21.41e8 Q4_K_M | SILO
1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0
Operational Intensity (FLOPs/Byte)12341e9 Q4_K_M | SISOJetson Orin Nano Super 8G - Model: PLM-Instruct
Layers: 2
Layers: 4
Layers: 6
Layers: 8
Layers: 10
Layers: 12
Layers: 14
Layers: 16
Layers: 18
Layers: 20
Layers: 22
Layers: 24
Layers: 26
Layers: 28
Layers: 30
Layers: 32
Figure 11.Layer-wise operational intensity trends for PLM-Instruct on Jetson Orin Nano Super 8G. This figure evaluates the distribution
of computational intensity across Transformer layers for FP16, Q8 0, and Q4 KM precisions. While the operational intensity typically
maintains a steady profile throughout the model depth, the sharp declines observed in specific scenarios signify that the execution has
encountered critical hardware bottlenecks. These straight-line drops indicate regions where the performance is heavily constrained by
system resource limits, preventing the layers from sustaining their peak arithmetic intensity.
20

## Page 21

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
1.0 1.5 2.0 2.5 3.0123456789Performance(FLOPs/sec)1e9 FP16 | LILO
10 20 30 40 50 600.20.40.60.81.01.21e11 FP16 | LISO
0.010 0.015 0.020 0.025 0.030 0.0350.20.40.60.81.01e8 FP16 | SILO
0.2 0.4 0.6 0.8 1.0 1.2 1.40.51.01.52.02.53.03.51e9 FP16 | SISO
2 3 4 5 6 70.20.40.60.81.0Performance(FLOPs/sec)1e10 Q8_0 | LILO
20 40 60 80 100 1200.20.40.60.81.01.21.41e11 Q8_0 | LISO
0.02 0.03 0.04 0.05 0.06 0.07 0.080.20.40.60.81.01.21.41e8 Q8_0 | SILO
0.5 1.0 1.5 2.0 2.5 3.0123451e9 Q8_0 | SISO
4 6 8 10 12
Operational Intensity (FLOPs/Byte)0.20.40.60.81.01.2Performance(FLOPs/sec)1e10 Q4_K_M | LILO
40 60 80 100 120 140 160 180
Operational Intensity (FLOPs/Byte)0.20.40.60.81.01.21.41e11 Q4_K_M | LISO
0.04 0.06 0.08 0.10 0.12 0.14 0.16
Operational Intensity (FLOPs/Byte)0.20.40.60.81.01.21.41.61e8 Q4_K_M | SILO
1 2 3 4 5 6
Operational Intensity (FLOPs/Byte)1234561e9 Q4_K_M | SISOApple M1 Pro - Model: Fox-1
Layers: 2
Layers: 4
Layers: 6
Layers: 8
Layers: 10
Layers: 12
Layers: 14
Layers: 16
Layers: 18
Layers: 20
Layers: 22
Layers: 24
Layers: 26
Layers: 28
Layers: 30
Layers: 32
Layers: 34
Layers: 36
Layers: 38
Layers: 40
Layers: 42
Layers: 44
Layers: 46
Layers: 48
Layers: 50
Layers: 52
Layers: 54
Layers: 56
Layers: 58
Layers: 60
Layers: 62
Layers: 64
Figure 12.Layer-wise operational intensity trends for Fox-1 on Apple M1 Pro. This figure analyzes the stability of computational intensity
across successive Transformer layers under FP16, Q8 0, and Q4 KM precisions. Across the diverse inference scenarios (SISO, SILO,
LISO, and LILO), the operational intensity remains remarkably constant as the layer index increases. This horizontal trend suggests that
the ratio of arithmetic operations to memory traffic is intrinsically determined by the model architecture and scenario type rather than the
specific depth of the layer.
21

## Page 22

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
0.5 1.0 1.5 2.0 2.5 3.0 3.5123456Performance(FLOPs/sec)1e9 FP16 | LILO
10 20 30 40 50 60 701234567891e10 FP16 | LISO
0.005 0.010 0.015 0.020 0.025 0.030 0.035 0.04012345671e7 FP16 | SILO
0.2 0.4 0.6 0.8 1.0 1.2 1.40.51.01.52.02.51e9 FP16 | SISO
1 2 3 4 5 6 71234567Performance(FLOPs/sec)1e9 Q8_0 | LILO
20 40 60 80 100 12024681e10 Q8_0 | LISO
0.02 0.04 0.06 0.0824681e7 Q8_0 | SILO
0.5 1.0 1.5 2.0 2.5 3.0 3.50.51.01.52.02.53.03.51e9 Q8_0 | SISO
2 4 6 8 10 12
Operational Intensity (FLOPs/Byte)2468Performance(FLOPs/sec)1e9 Q4_K_M | LILO
25 50 75 100 125 150 175 200
Operational Intensity (FLOPs/Byte)24681e10 Q4_K_M | LISO
0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16
Operational Intensity (FLOPs/Byte)0.20.40.60.81.01.21e8 Q4_K_M | SILO
1 2 3 4 5 6
Operational Intensity (FLOPs/Byte)123451e9 Q4_K_M | SISOApple M1 Pro - Model: Llama-3.2-Instruct
Layers: 2
Layers: 4
Layers: 6
Layers: 8
Layers: 10
Layers: 12
Layers: 14
Layers: 16
Layers: 18
Layers: 20
Layers: 22
Layers: 24
Layers: 26
Layers: 28
Layers: 30
Layers: 32
Layers: 34
Layers: 36
Layers: 38
Layers: 40
Layers: 42
Layers: 44
Layers: 46
Layers: 48
Layers: 50
Layers: 52
Layers: 54
Layers: 56
Layers: 58
Layers: 60
Layers: 62
Layers: 64
Figure 13.Layer-wise operational intensity trends for Llama-3.2-Instruct on Apple M1 Pro. The subplots illustrate the distribution of
FLOPs/Byte across Transformer layers for FP16, Q8 0, and Q4 KM precisions. While the operational intensity is generally consistent
across the model depth, a sharp decline is observed in specific scenarios, signifying that the execution has encountered a critical hardware
bottleneck. This transition highlights the segments where the computational efficiency is significantly hampered by data movement
constraints or other resource limitations.
22

## Page 23

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
0.5 1.0 1.5 2.0 2.5 3.0 3.512345Performance(FLOPs/sec)1e9 FP16 | LILO
10 20 30 40 50 60123456781e10 FP16 | LISO
0.005 0.010 0.015 0.020 0.025 0.030 0.035123451e7 FP16 | SILO
0.2 0.4 0.6 0.8 1.00.250.500.751.001.251.501.751e9 FP16 | SISO
1 2 3 4 5 6 7123456Performance(FLOPs/sec)1e9 Q8_0 | LILO
20 40 60 80 100 120123456781e10 Q8_0 | LISO
0.01 0.02 0.03 0.04 0.05 0.06 0.0712345671e7 Q8_0 | SILO
0.50 0.75 1.00 1.25 1.50 1.75 2.00 2.25 2.500.51.01.52.01e9 Q8_0 | SISO
2 4 6 8 10 12 14
Operational Intensity (FLOPs/Byte)12345678Performance(FLOPs/sec)1e9 Q4_K_M | LILO
25 50 75 100 125 150 175 200
Operational Intensity (FLOPs/Byte)123456781e10 Q4_K_M | LISO
0.02 0.04 0.06 0.08 0.10 0.12 0.14
Operational Intensity (FLOPs/Byte)1234567891e7 Q4_K_M | SILO
1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0
Operational Intensity (FLOPs/Byte)0.51.01.52.02.53.01e9 Q4_K_M | SISOApple M1 Pro - Model: Qwen2.5-Instruct
Layers: 2
Layers: 4
Layers: 6
Layers: 8
Layers: 10
Layers: 12
Layers: 14
Layers: 16
Layers: 18
Layers: 20
Layers: 22
Layers: 24
Layers: 26
Layers: 28
Layers: 30
Layers: 32
Layers: 34
Layers: 36
Layers: 38
Layers: 40
Layers: 42
Layers: 44
Layers: 46
Layers: 48
Layers: 50
Layers: 52
Layers: 54
Layers: 56
Layers: 58
Layers: 60
Layers: 62
Layers: 64
Figure 14.Layer-wise operational intensity trends for Qwen2.5-Instruct on Apple M1 Pro. These profiles characterize the layer-wise
behavior of arithmetic intensity across multiple inference scenarios. The horizontal alignment of data points across the layer index
confirms that the computational intensity is invariant to layer depth. This stability is maintained across all tested precisions, demonstrating
that quantization shifts the absolute intensity but does not alter the uniform trend across the model’s structural hierarchy.
23

## Page 24

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
1 2 3 4 512345Performance(FLOPs/sec)1e9 FP16 | LILO
20 40 60 80 10024681e10 FP16 | LISO
0.010 0.015 0.020 0.025 0.030 0.035 0.040 0.045123451e7 FP16 | SILO
0.50 0.75 1.00 1.25 1.50 1.75 2.000.250.500.751.001.251.501.752.001e9 FP16 | SISO
1 2 3 4 5 6 7 8123456Performance(FLOPs/sec)1e9 Q8_0 | LILO
20 40 60 80 100 120 14024681e10 Q8_0 | LISO
0.01 0.02 0.03 0.04 0.05 0.06 0.07123451e7 Q8_0 | SILO
1.0 1.5 2.0 2.5 3.0 3.50.250.500.751.001.251.501.752.001e9 Q8_0 | SISO
2 4 6 8 10 12 14
Operational Intensity (FLOPs/Byte)1234567Performance(FLOPs/sec)1e9 Q4_K_M | LILO
25 50 75 100 125 150 175 200
Operational Intensity (FLOPs/Byte)24681e10 Q4_K_M | LISO
0.02 0.04 0.06 0.08 0.10 0.12
Operational Intensity (FLOPs/Byte)1234561e7 Q4_K_M | SILO
1 2 3 4 5 6 7
Operational Intensity (FLOPs/Byte)0.51.01.52.02.53.01e9 Q4_K_M | SISOApple M1 Pro - Model: Qwen3
Layers: 2
Layers: 4
Layers: 6
Layers: 8
Layers: 10
Layers: 12
Layers: 14
Layers: 16
Layers: 18
Layers: 20
Layers: 22
Layers: 24
Layers: 26
Layers: 28
Layers: 30
Layers: 32
Layers: 34
Layers: 36
Layers: 38
Layers: 40
Layers: 42
Layers: 44
Layers: 46
Layers: 48
Layers: 50
Layers: 52
Layers: 54
Layers: 56
Layers: 58
Layers: 60
Layers: 62
Layers: 64
Figure 15.Layer-wise operational intensity trends for Qwen3 on Apple M1 Pro. The subplots map the operational intensity against the
Transformer layer index for various precisions. Regardless of the inference workload (SISO, SILO, LISO, or LILO), the metrics exhibit a
flat progression from the initial to the final layers. This trend indicates that the bottleneck characteristics, whether memory-bound or
compute-bound, are shared equally by all layers within the model architecture.
24

## Page 25

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Qwen2.5-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Llama-3.2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: PLM-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Qwen3
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Fox-1
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: SmolLM2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)Apple M1 Pro - Precision: FP16
Figure 16.Performance profiles of diverse inference scenarios on Apple M1 Pro (FP16). This figure evaluates various models across four
distinct operational scenarios: SISO (Short In, Short Out), SILO (Short In, Long Out), LISO (Long In, Short Out), and LILO (Long In,
Long Out). The results illustrate how operational intensity (FLOPs/Byte) shifts performance (FLOPs/sec). LISO scenarios typically reach
the real computation bound , while decoding-heavy SILO/LILO scenarios remain memory-bound. The hardware exhibits a real ridge
point of 35.83.
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Qwen2.5-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Llama-3.2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: PLM-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Qwen3
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Fox-1
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: SmolLM2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)Apple M1 Pro - Precision: Q8_0
Figure 17.Performance profiles of diverse inference scenarios on Apple M1 Pro (Q8 0). This set of roofline models highlights the
performance of 8-bit quantized LLMs under four characteristic workloads: SISO, SILO, LISO, and LILO. The plots demonstrate that
as output length increases (SILO/LILO), the operational intensity drops significantly, trapping the performance within the memory-
bandwidth-limited regime. In contrast, LISO tasks with long input sequences move toward the compute-bound ceiling. The real ridge
point remains constant at 35.83.
25

## Page 26

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Qwen2.5-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Llama-3.2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: PLM-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Qwen3
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Fox-1
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: SmolLM2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 35.83
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)Apple M1 Pro - Precision: Q4_K_M
Figure 18.Performance profiles of diverse inference scenarios on Apple M1 Pro (Q4 KM). This figure depicts the performance of 4-bit
quantized models across varied input and output context lengths. By categorizing tasks into SISO, SILO, LISO, and LILO clusters , the
profiles reveal the impact of quantization on arithmetic intensity and throughput. Quantization allows models to achieve higher throughput
relative to the real computation bound in prefill-dominant scenarios (e.g., LISO). The dashed vertical line marks the system’s empirical
ridge point at 35.83.
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Qwen2.5-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 43.18
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Llama-3.2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 43.18
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: PLM-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 43.18
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Qwen3
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 43.18
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Fox-1
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 43.18
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: SmolLM2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 43.18
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)RTX 3070 Ti Laptop - Precision: FP16
Figure 19.Performance profiles of diverse inference scenarios on RTX 3070 Ti Laptop (FP16). This figure evaluates the throughput
(FLOPs/sec ) across various models (e.g., Qwen2.5-Instruct, Llama-3.2-Instruct, and PLM-Instruct) under four distinct operational
scenarios: SISO (Short In, Short Out), SILO (Short In, Long Out), LISO (Long In, Short Out), and LILO (Long In, Long Out). As
operational intensity increases, tasks transition from the memory-bound region (e.g., SILO and LILO) toward the empirical computation
bound, with prefill-heavy LISO scenarios exhibiting the highest arithmetic intensity. The hardware configuration exhibits a real ridge
point of 43.18.
26

## Page 27

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Qwen2.5-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 43.18
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Llama-3.2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 43.18
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: PLM-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 43.18
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Qwen3
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 43.18
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Fox-1
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 43.18
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: SmolLM2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 43.18
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)RTX 3070 Ti Laptop - Precision: Q8_0
Figure 20.Performance profiles of diverse inference scenarios on RTX 3070 Ti Laptop (Q8 0). These roofline models characterize
the performance of 8-bit quantized LLMs (Q8 0) under four characteristic workloads: SISO, SILO, LISO, and LILO. The results
demonstrate that as output length increases (SILO/LILO), the operational intensity drops significantly, trapping performance within the
memory-bandwidth-limited regime. In contrast, LISO tasks with long input sequences move toward the hardware’s empirical computation
ceiling. The real ridge point is consistently identified at 43.18.
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Qwen2.5-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 43.18
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Llama-3.2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 43.18
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: PLM-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 43.18
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Qwen3
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 43.18
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: Fox-1
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 43.18
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)1061071081091010101110121013Performance (FLOPs/sec)
Model: SmolLM2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 43.18
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)RTX 3070 Ti Laptop - Precision: Q4_K_M
Figure 21.Performance profiles of diverse inference scenarios on RTX 3070 Ti Laptop (Q4 KM). This set of profiles evaluates 4-bit
quantized LLMs categorized into SISO, SILO, LISO, and LILO clusters to illustrate the impact of quantization on throughput across
varied input/output lengths. Quantization shifts the compute-to-memory balance, allowing models to achieve higher throughput relative to
the real computation bound in prefill-dominant scenarios (LISO). The dashed vertical line marks the system’s empirical ridge point at
43.18.
27

## Page 28

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
102
100102104106
Operational Intensity (FLOPs/Byte)106107108109101010111012Performance (FLOPs/sec)
Model: Qwen2.5-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 22.56
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)106107108109101010111012Performance (FLOPs/sec)
Model: Llama-3.2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 22.56
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)106107108109101010111012Performance (FLOPs/sec)
Model: PLM-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 22.56
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)106107108109101010111012Performance (FLOPs/sec)
Model: Qwen3
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 22.56
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)106107108109101010111012Performance (FLOPs/sec)
Model: Fox-1
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 22.56
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)106107108109101010111012Performance (FLOPs/sec)
Model: SmolLM2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 22.56
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)Jetson Orin Nano Super 8G - Precision: FP16
Figure 22.Performance profiles of diverse inference scenarios on Jetson Orin Nano Super 8G (FP16). This figure evaluates various
models—including Qwen2.5-Instruct, PLM-Instruct, Qwen3, Fox-1, and SmolLM2-Instruct—across four distinct operational scenarios:
SISO (Short In, Short Out) , SILO (Short In, Long Out) , LISO (Long In, Short Out) , and LILO (Long In, Long Out). The subplots
illustrate how operational intensity determines the achieved throughput ( FLOPs/sec ). While decoding-heavy scenarios (SILO/LILO)
remain memory-bound , prefill-dominant LISO tasks approach the real computation bound. The platform is characterized by a real ridge
point of 22.56.
102
100102104106
Operational Intensity (FLOPs/Byte)106107108109101010111012Performance (FLOPs/sec)
Model: Qwen2.5-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 22.56
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)106107108109101010111012Performance (FLOPs/sec)
Model: Llama-3.2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 22.56
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)106107108109101010111012Performance (FLOPs/sec)
Model: PLM-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 22.56
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)106107108109101010111012Performance (FLOPs/sec)
Model: Qwen3
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 22.56
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)106107108109101010111012Performance (FLOPs/sec)
Model: Fox-1
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 22.56
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)106107108109101010111012Performance (FLOPs/sec)
Model: SmolLM2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 22.56
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)Jetson Orin Nano Super 8G - Precision: Q8_0
Figure 23.Performance profiles of diverse inference scenarios on Jetson Orin Nano Super 8G (Q8 0). These roofline models highlight the
performance of 8-bit quantized LLMs—including Qwen2.5-Instruct, Llama-3.2-Instruct, Qwen3, Fox-1, PLM-Instruct, and SmolLM2-
Instruct—under four characteristic workloads: SISO, SILO, LISO, and LILO. The plots demonstrate that as output length increases
(SILO/LILO), the operational intensity drops significantly, trapping the performance within the memory-bandwidth-limited regime. In
contrast, LISO tasks with long input sequences move toward the hardware’s empirical computation ceiling. Consistent with the platform
profile, the real ridge point is consistently identified at 22.56.
28

## Page 29

RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis
102
100102104106
Operational Intensity (FLOPs/Byte)106107108109101010111012Performance (FLOPs/sec)
Model: Qwen2.5-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 22.56
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)106107108109101010111012Performance (FLOPs/sec)
Model: Llama-3.2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 22.56
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)106107108109101010111012Performance (FLOPs/sec)
Model: PLM-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 22.56
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)106107108109101010111012Performance (FLOPs/sec)
Model: Qwen3
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 22.56
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)106107108109101010111012Performance (FLOPs/sec)
Model: Fox-1
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 22.56
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)106107108109101010111012Performance (FLOPs/sec)
Model: SmolLM2-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 22.56
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)Jetson Orin Nano Super 8G - Precision: Q4_K_M
Figure 24.Performance profiles of diverse inference scenarios on Jetson Orin Nano Super 8G (Q4 KM). This figure depicts the
performance distribution of 4-bit quantized models—including Qwen2.5-Instruct, Llama-3.2-Instruct, PLM-Instruct, Qwen3, Fox-1, and
SmolLM2-Instruct—across varied input and output context lengths. By categorizing tasks into SISO, SILO, LISO, and LILO clusters , the
profiles reveal the impact of quantization on arithmetic intensity and throughput. Quantization allows models to achieve higher efficiency
relative to the real computation bound, especially in LISO scenarios. The vertical dashed line indicates the system’s empirical ridge point
at 22.56.
102
100102104106
Operational Intensity (FLOPs/Byte)10610710810910101011Performance (FLOPs/sec)Model: Qwen2.5-0.5B-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 20.13
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)10610710810910101011Performance (FLOPs/sec)Model: SmolLM2-135M-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 20.13
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)10610710810910101011Performance (FLOPs/sec)Model: pythia-160m
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 20.13
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)10610710810910101011Performance (FLOPs/sec)Model: SmolLM2-360M-Instruct
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 20.13
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)
102
100102104106
Operational Intensity (FLOPs/Byte)10610710810910101011Performance (FLOPs/sec)Model: pythia-410m
Theo Memory Access Bound
Theo Computation Bound
Real Memory Access Bound
Real Computation Bound
Real Ridge: 20.13
SISO (Short In, Short Out)
SILO (Short In, Long Out)
LISO (Long In, Short Out)
LILO (Long In, Long Out)Raspberry Pi 5 - Precision: Q8_0
Figure 25.Performance profiles of diverse inference scenarios on Raspberry Pi 5 (Q8 0). This figure characterizes the performance of
8-bit quantized lightweight LLMs—including SmolLM2-360M-Instruct and Pythia-410m—across four distinct operational workloads:
SISO (Short In, Short Out), SILO (Short In, Long Out), LISO (Long In, Short Out), and LILO (Long In, Long Out). The subplots illustrate
that in this resource-constrained environment, most scenarios are trapped within the memory-bandwidth-limited regime, particularly when
output length increases (SILO/LILO). In contrast, LISO tasks with long input sequences exhibit significantly higher operational intensity,
moving closer to the hardware’s empirical computation ceiling. Theoretical and real bounds are plotted with a consistent real ridge point
identified at 20.13.
29