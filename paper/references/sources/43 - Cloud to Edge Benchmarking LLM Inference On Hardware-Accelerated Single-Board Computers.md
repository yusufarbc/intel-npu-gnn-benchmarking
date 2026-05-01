# Cloud to Edge Benchmarking LLM Inference On Hardware-Accelerated Single-Board Computers

## Page 1

CLOUD TOEDGE: BENCHMARKINGLLM INFERENCE ON
HARDWARE-ACCELERATEDSINGLE-BOARDCOMPUTERS
Harri Renney
Kaze Technologies
Kaze Consulting
Bath, BA1 2HN, UK
harri@kaze-consulting.comFouad Trad
Electrical and Computer Engineering
Lebanese American University
Byblos, Lebanon
fouad.trad@lau.edu.lbMichael Mattarock
Carnegie Mellon University
Pittsburgh
Pennsylvania
USA
mattarock@cmu.edu
Zena Wood
University of Exeter
Exeter, UK
Z.M.Wood2@exeter.ac.uk
April 29, 2026
ABSTRACT
Large language models (LLMs) are becoming increasingly capable at small parameter scales. At
the same time, conventional cloud-centric deployment introduces challenges around data privacy,
latency, and cost that are acute in operational technology and defence environments. Advances in
model distillation, quantisation, and affordable edge accelerators now make local LLM inference
on single-board computers feasible, but the high dimensionality of the configuration space makes
identifying optimal deployments difficult without structured evaluation. Existing LLM-specific edge
benchmarking efforts rely on CPU-only inference, poor coverage of genuine single-board computers,
and generic evaluation tasks that lack multi-dimensional assessment of hardware effectiveness. This
paper proposes a multi-dimensional benchmarking methodology that jointly evaluates inference
performance and hardware efficiency across four IoT-suitable edge platform configurations testing
single-board computers with the latest available hardware accelerators. Our results reveal the benefits
of using hardware accelerators such as NPUs and GPUs, along with multi-dimensional evaluations
quantifying the trade-offs between power efficiency, physical device size and token throughput;
offering practical guidance for deploying generative AI in privacy-sensitive and connectivity-limited
environments such as unmanned vehicles and portable, ruggedised operations.
KeywordsLarge Language Models ·Edge Computing ·Internet of Things ·Critical Infrastructure ·Single-Board
Computers·Cybersecurity·Benchmarking
1 Introduction
Since the introduction of the transformer architecture [1], large language models (LLMs) have demonstrated remarkable
emergent capabilities across reasoning, creative writing, and professional examinations [2 –5]. However, LLM deploy-
ment remains predominantly cloud-centric, with large models accessed via API endpoints as LLM-as-a-Service [6],
introducing challenges around data privacy, latency, and dependence on stable connectivity [7, 8].
These limitations are acute in critical industries, defined here as sectors whose networks, systems, and assets are
fundamental to public health, economic stability, and national security (e.g., those within energy, health, financial, man-
ufacturing, transportation, or communications sectors). In such operational technology (OT) and defence environments,
sensitive data often cannot leave a controlled network perimeter, making localised inference a security imperative. AsarXiv:2604.24785v1  [cs.AR]  24 Apr 2026

## Page 2

APREPRINT- APRIL29, 2026
Internet of Everything deployments across these sectors increasingly demand intelligent local processing, the ability
to run LLMs on cost-effective edge hardware becomes essential, from distributed cyber-physical systems to remote
satellite ground stations. Three convergent developments have recently changed the picture: (i) distilled small language
models in the 1.5B–7B parameter range that retain strong generative capability [9, 10]; (ii) post-training quantisation to
INT4/INT8 precision, substantially reducing memory requirements with only modest accuracy loss [11, 12]; and (iii) a
new generation of affordable edge accelerators, such as the Hailo-10H (40 TOPS), NVIDIA Jetson Orin Nano Super
(67 TOPS), and integrated NPUs like the M5Stack AX630C, bringing meaningful AI compute to platforms under $350.
The resulting configuration space spans device type, accelerator, model family, parameter count, quantisation level,
and inference runtime, making it difficult to identify optimal configurations without structured evaluation. Existing
edge benchmarking tools, such as DeepEdgeBench [13], primarily target general problem-solving performance. While
more recent frameworks, including LEAF [14] and BeDGED [15], expand the evaluation to additional edge-relevant
dimensions, they remain limited by reliance on CPU-only or legacy GPU-based LLM inference on IoT-class hardware.
This paper makes three contributions:
•A gap analysis of existing edge LLM benchmarking approaches, identifying the absence of multi-dimensional
evaluation frameworks that jointly consider inference performance, hardware efficiency, and physical deploy-
ment constraints for IoT-suitable single-board computers.
•Initial steps toward a multi-dimensional evaluation framework for edge LLM deployment, introducing two
composite metrics, throughput density (Tps/m3) and energy per million tokens (MJ/Mtok), that extend
conventional single-axis benchmarking to account for physical and energy deployment constraints. These
metrics are validated through benchmarking four edge platforms and multiple model families at the 1.5B–3B
parameter scale.
•An analysis of the trade-offs between token throughput, energy efficiency, and physical device size, with direct
applicability to Internet of Everything deployments in critical industries, including distributed cyber networks,
unmanned aerial vehicles, and satellite ground stations, along with cybersecurity considerations.
2 Related Work
Table 1: Positioning of this work relative to closest prior studies
LEAF [14] BeDGED [15] LLMs-at-Edge [16] Abstreiter et al. [17] Tummalapalli et al.
[18]This work
Edge hardware RPi 4/5, Jetson Nano RPi 5 cluster Jetson AGX OrinRPi 5, Jetson Orin
NanoMobile, RPi 5 +
NPU, Laptop GPUM5Stack, RPi 5, RPi 5 + AI
HAT+, Jetson Orin
Edge accelerators Maxwell GPU None (CPU only) Ampere GPU GPU only GPU + NPU GPU + NPU
Inference runtime Ollama Ollama Ollama llama.cppMLC, MLX,
vLLMOllama + StackFlow
MetricsLatency, accuracy,
Energy-per-TokenThroughput, la-
tency, memoryThroughput, en-
ergy, accuracyPerformance, en-
ergy, micro-archPerformance,
power, thermalThroughput, TTFT, energy
(MJ/Mtok), throughput den-
sity (Tps/m3)
Edge AI is a well-established field, with comprehensive taxonomies covering infrastructure, resource management, and
scheduling for conventional workloads [19], and general edge AI benchmarking frameworks such as DeepEdgeBench
[13] standardising evaluation on constrained hardware. However, these tools and taxonomies do not address LLM
inference, and DeepEdgeBench specifically does not account for the generation dynamics of LLMs [14].
2.1 Quantisation and Small Models for Edge Deployment
The feasibility of edge LLM inference rests on two parallel advances. Distilled architectures at 1.5B to 7B parameters,
including Llama 3.2, Qwen 2.5, Phi-3.5/4-mini, and Gemma 2/3, now retain strong generative capability through
knowledge distillation and synthetic data training [9, 10]. On the compression side, post-training quantisation to
INT4/INT8 reduces memory and bandwidth requirements substantially with modest accuracy loss [11, 12]. The GGUF
format and Q4_K_M quantisation scheme have become a practical standard for portable deployment across ARM
CPUs and NVIDIA GPUs, with Ollama applying Q4_K_M by default. Reports demonstrate 4×–16× model size
reduction when compressing FP16 to INT4 (i.e., 75–93.75% size reduction) [20], while hardware evaluations further
show 57–61% reductions in area and power consumption, demonstrating practical efficiency gains [21] that greatly
benefit edge computing.
2

## Page 3

APREPRINT- APRIL29, 2026
2.2 Benchmarking LLMs at the Edge
Two recent frameworks begin to address this gap. LEAF [14] assesses LLMs and introduces sustainability metrics
(Circular Economy Score, Energy-per-Token) alongside latency and accuracy, testing 4-bit quantised models via
Ollama on only two SBCs (a Raspberry Pi 4/5 and a Jetson Nano) alongside desktops with an NVIDIA T400 and
other legacy GPUs. BeDGED [15] deploys a Raspberry Pi cluster in a small base station (SBS) architecture with
lightweight Kubernetes, capturing throughput, latency, and memory utilisation, but performs inference purely on the
CPU. While both make valuable contributions, neither tests dedicated edge accelerators such as the Hailo-10H NPU or
current-generation Jetson hardware; both rely exclusively on Ollama; and both evaluate against generic benchmarks
rather than edge-specific metrics.
Abstreiter et al. [17] further investigate on-device LLM inference across CPU- and GPU-based SBC platforms,
evaluating inference speed, energy, and micro-architectural behaviour. However, the study is limited to two devices
(Raspberry Pi 5 and Jetson Orin Nano) and does not consider dedicated edge NPUs or integrate findings into an edge
multi-dimensional framework. Tummalapalli et al. [18] benchmark a quantised 1.5B model across mobile devices,
GPUs, and a Hailo-10H NPU, focusing on throughput, power, and thermal behaviour under sustained inference.
Their findings highlight thermal throttling and memory bandwidth as key constraints, but the study is restricted to a
single model and does not explore the broader configuration space or SBC-focused trade-offs. In OT and industrial
contexts, existing LLM applications such as PLC code generation [22] and maritime data enrichment [23] have relied
on cloud-hosted endpoints, underscoring the need for localised inference on cost-effective SBC hardware.
2.3 Research Gap
As positioned in Table 1, our work addresses these gaps in three key ways: (i) benchmarking across modern edge
accelerators, including Hailo-10H, NVIDIA Ampere, and AX630C NPUs, not evaluated in prior LLM studies;
(ii) supporting multiple inference runtimes (e.g., Ollama and StackFlow), reflecting platform-specific deployment
constraints; and (iii) introducing multi-dimensional evaluation metrics that jointly consider token throughput alongside
energy efficiency and physical device size, enabling informed device selection for Internet of Everything applications.
3 Experimental Setup
3.1 IoT-Suitable Hardware Platforms
Experiments were conducted across four edge platforms in five configurations, representing a spectrum of compute
capabilities as detailed in Table 2. These configurations span CPU-only, NPU-accelerated, and GPU-accelerated
inference, enabling comparative evaluation of heterogeneous edge compute architectures.
3.2 Model Selection
We evaluate a set of compact LLMs representative of current edge-deployable models, spanning parameter sizes from
0.5B to 3B:
• DeepSeek-R1-Distill-Qwen (1.5B)
• Qwen 2.5 family (0.5B and 1.5B variants)
• Qwen 2.5 Instruct and Coder variants (1.5B)
• Llama 3.2 (1B and 3B)
These models were selected to reflect a range of architectures and training objectives (general-purpose, instruction-tuned,
and code-specialised) at parameter scales suitable for edge deployment. Critically, all selected models are well supported
by their respective vendors for edge inference.
3.3 Inference Configuration
Three runtime configurations were used across the testbed:
•Native Ollama: Ollama’s default HTTP API ( :11434/api/chat ), used on the Raspberry Pi 5 (CPU) and
Jetson Orin Nano (CPU and GPU).
•Hailo Ollama: Hailo’s custom Ollama server implementation ( :8000/api/chat ) for NPU-accelerated
inference on the Raspberry Pi 5 + AI HAT+.
3

## Page 4

APREPRINT- APRIL29, 2026
Table 2: Hardware characteristics relevant to efficiency benchmarking.
Device Dimensions Price ($) CPU AI Accelerator
Name Cores Speed Memory Name Speed / TOPS Memory
M5Stack LLM
(Module LLM,
M140)54.0 × 54.0 ×
13.0 mm$99.90 AX630C SoC 2 Up to
1.2GHz4 GB
LPDDR4
(1 GB usable)AX630C NPU 3.2 TOPS
(INT8)∼3 GB acceler-
ator RAM
Raspberry Pi 585 × 56 × 17 mm $222.75 Broadcom
BCM2712
(Cortex-A76)4 2.4 GHz 8 GB
LPDDR4X -4267— — —
Raspberry Pi 5 +
AI HAT+ 285 × 56 × 20 mm $451.83 Broadcom
BCM2712
(Cortex-A76)4 2.4 GHz 8 GB
LPDDR4X -4267Hailo-10H 40 TOPS
(INT4)8 GB accelera-
tor RAM
NVIDIA Jetson
Orin Nano Su-
per Dev Kit100 × 79 × 21 mm $304.78 Arm
Cortex-A78AE6 Up to
1.7 GHz8 GB
LPDDR5,
102 GB/s BWNVIDIA Ampere -
1024 CUDA cores,
32 tensor cores67 TOPS
(INT8)Shared 8 GB
LPDDR5
•M5Stack StackFlow: Runtime for the AX630C NPU.
All Ollama-based configurations used streaming inference, where tokens are returned incrementally as they are generated.
Models were executed using 4-bit quantisation (Q4_K_M) as the primary precision level. Each inference call used a
fixed prompt (“Explain why the sky is blue in two or more paragraphs.”) with a generation length capped at 100 tokens
via the num_predict parameter. Decoding parameters were held constant across all platforms to ensure comparability.
3.4 Evaluation & Multi-Dimensional Composite Metrics
Performance was evaluated using three primary metrics:
•Throughput (tokens/s): the rate of token generation during inference.
•Time-to-first-token (TTFT): latency between prompt submission and generation of the first token.
•Energy consumption (MJ/Mtok): energy required to generate one million tokens, capturing hardware
efficiency.
These metrics jointly capture responsiveness, sustained generation performance, and energy efficiency.
Edge and IoT deployments impose simultaneous constraints across power, physical footprint, and performance that
conventional single-axis LLM benchmarks do not capture. A drone may require minimal power draw to preserve flight
time, a satellite ground station may have a desired form-factor limit to accommodate other peripherals. To capture these
trade-offs, we define two composite metrics that normalise token throughput against the physical and energy constraints
relevant to IoT-suitable hardware:
•Throughput density (Tps/m3): Token throughput per second normalised by device volume in cubic metres.
This metric quantifies inference capability per unit of physical space, enabling comparison across platforms
with substantially different form factors. It is particularly relevant for space-constrained deployments such as
unmanned aerial vehicles.
•Energy per million tokens (MJ/Mtok): Total energy consumed to generate one million tokens, derived from
measured device power draw and generation time. This metric captures sustained operational cost and is
critical for battery-powered or energy-budgeted deployments where inference must coexist with other system
functions such as sensing, communication, and control, as is typical in autonomous systems and remote
systems operating without fixed power infrastructure.
Combined with raw throughput and TTFT, these composite metrics form the foundations of a multi-dimensional
evaluation framework that enables practitioners to select hardware configurations based on their specific deployment
constraints rather than a single performance axis.
3.5 Measurement Methodology
Benchmarking was automated using a custom Python harness that interfaces with each runtime’s HTTP API1. For each
model and hardware configuration, a warmup request is issued to ensure the model is loaded into memory before timed
1Source code is publicly availablehttps://github.com/SquidyBallinx11011/LLM-Edge-Benchmarking-Suite
4

## Page 5

APREPRINT- APRIL29, 2026
runs begin. The benchmark then issues a streaming inference request, recording wall-clock time at submission and at
receipt of the first token (yielding TTFT), with throughput calculated as tokens generated divided by total elapsed time.
Each configuration is executed for n=5 runs, the mean reported and per-run deviations logged to assess variance. Power
consumption was measured using a Mecheer JK-PM07 power meter and scaled to derive energy per million tokens
(MJ/Mtok).
4 Results & Analysis
The salient results2are presented in Table 3. The experimental design enables comparison across hardware classes
(CPU vs NPU vs GPU), model scales (0.5B–3B), and efficiency-constraint trade-offs.
Table 3: LLM inference benchmarking results across edge hardware configurations
Model Size Metric M5Stack LLM RPi 5 RPi5+HAT+ Jetson Orin CPU Jetson Orin GPU
DeepSeek-R1-Distill-Qwen-1.5B 1.5BThroughput (tok/s) 2.42 0.32 1.53 6.01 9.59
Time-to-first-token (ms) 2.15 20.79 10.47 2.08 1.38
Energy Consumption (MJ/Mtok) 0.57 33.24 3.47 2.09 1.15
Qwen 2.50.5BThroughput (tok/s) – 0.86 – 13.04 13.31
Time-to-first-token (ms) – 8.96 – 1.51 1.43
Energy Consumption (MJ/Mtok) – 10.83 – 0.92 1.20
1.5BThroughput (tok/s) – 0.27 4.37 4.21 9.37
Time-to-first-token (ms) – 29.98 2.36 4.74 2.13
Energy Consumption (MJ/Mtok) – 35.30 1.30 1.38 1.11
qwen2.5:1.5b-instruct 1.5BThroughput (tok/s) 2.55 0.3 6.34 6.9 9.09
Time-to-first-token (ms) 2.11 26.44 0.55 2.86 2.20
Energy Consumption (MJ/Mtok) 0.57 35.31 0.88 1.83 1.17
qwen2.5-coder:1.5b 1.5BThroughput (tok/s) – 0.3 2.07 6.76 8.66
Time-to-first-token (ms) – 26.90 7.43 2.95 2.30
Energy Consumption (MJ/Mtok) – 36.01 2.57 1.87 1.16
Llama 3.21BThroughput (tok/s) 3.44 0.39 – 9.51 10.96
Time-to-first-token (ms) 1.78 19.93 – 2.09 1.82
Energy Consumption (MJ/Mtok) 0.41 27.87 – 1.64 1.23
3BThroughput (tok/s) – 0.14 1.01 4.3 6.31
Time-to-first-token (ms) – 58.71 13.11 3.92 3.16
Energy Consumption (MJ/Mtok) – 76.64 5.51 3.04 1.72
Results are mean ofn= 5runs. Generation length fixed at 100 tokens.
– = model not supported for device. MJ/Mtok = Energy consumption per million tokens (Mtok) in MegaJoules (MJ)
4.1 Impact of Hardware Acceleration on Inference Efficiency
Figure 1 compares energy efficiency when offloading LLM inference from the CPU to a dedicated accelerator on both
the Raspberry Pi 5 (Hailo-10H NPU) and Jetson Orin Nano Super (Ampere GPU). On the RPi 5, the Hailo NPU delivers
9.57×to 39.97 ×energy efficiency gains, reducing consumption from 27–77 MJ/Mtok (CPU) to 0.88–5.51 MJ/Mtok,
while raising throughput from 0.14–0.86 tok/s to 1.01–6.34 tok/s.
On the Jetson, GPU offloading provides a more modest 1.24 ×to 1.82 ×efficiency gain, reflecting its stronger CPU
baseline. In both cases, the accelerator also frees the host CPU for concurrent tasks such as sensor acquisition or
network communication, a critical consideration in SBS deployments where the SBC serves multiple roles.
4.2 Physical and Power Constraints for IoT Deployment
Figure 2 plots power consumption against throughput for each configuration, with bubble size representing physical
device volume (cm3). The Jetson Orin GPU & CPU configurations deliver the highest throughput but at 12–13W and
166 cm3. The AX630c occupies the opposite corner: the smallest device (38cm3) with lowest power draw (~1.4W),
2The full set of benchmarking results across all models and configurations is publicly available online https://osf.io/5r9t4/
overview
5

## Page 6

APREPRINT- APRIL29, 2026
Figure 1: Comparison of power efficiency between CPU and available hardware accelerators on the Raspberry Pi 5
(left) and the Jetson Nano (right).
achieving 2–3 tok/s. The RPi5+HAT+ is between these at 6W and 95 cm3, while the RPi 5 CPU-only configuration
draws the most power relative to its output, consuming 11W for under 0.4 tok/s.
Notably, the RPi5+HAT+ delivers comparable throughput to the Jetson CPU at roughly half the power and in a smaller
form factor, making it a strong candidate for power and space constrained deployments where the Jetson’s GPU
performance is not required.
Figure 2: Device power consumption (W) against token throughput per second (T/s) where bubbles are sized according
to the volume of the device (cm3).
4.3 Multi-Dimensional Benchmarking Results
To evaluate edge LLM performance beyond raw speed, we introduce multi-dimensional throughput ratios that account
for physical form factor and energy cost alongside tokens per second. Figure 3 presents three surface plots across
devices and the three models supported by all configurations (qwen2.5-instruct-1.5B, llama3.2-1B, deepseek-r1-1.5B).
Raw throughput (Figure 3a) follows an expected hierarchy: the Jetson Orin GPU leads at 9–10 tok/s, followed by
Jetson CPU (4–7 tok/s), the RPi5+HAT+ and M5Stack AX630c (2–6 tok/s), and RPi 5 CPU trailing at under 0.4 tok/s.
However, when throughput is normalised by device volume (Figure 3b), the M5Stack AX630c dominates at up to
90K+ Tps/m3, owing to its 54 ×54×13mm form factor. This metric is relevant for space-constrained deployments such
as ruggedised enclosures, wearable systems, or embedded installations where physical footprint is a primary design
constraint.
6

## Page 7

APREPRINT- APRIL29, 2026
(a) Token throughput (T/s).
 (b) V olume-to-throughput ratio (Tps/m3).
 (c) Energy consumption (MJ/Mtok).
Figure 3: Multi-dimensional throughput ratios across hardware configurations for all shared supported LLMs: (a) raw
token throughput, (b) throughput normalised by device volume, and (c) energy consumption per million tokens.
Energy consumption per million tokens (Figure 3c) reveals a different ranking. The RPi 5 CPU is an order of magnitude
worse than all accelerated platforms (33–76 MJ/Mtok vs 0.6–5.5 MJ/Mtok). Among accelerated configurations, the
M5Stack and Jetson GPU are the most energy-efficient at 0.5–1.7 MJ/Mtok, while the RPi5+HAT+ varies more widely
by model (0.8–5.5 MJ/Mtok), suggesting that NPU efficiency is possibly more sensitive to model architecture than
GPU-based inference.
5 Discussion:
Implications for IoT in Critical Industries
5.1 Resilient Distributed Cyber-Physical Systems
The emergence of power-efficient edge AI hardware is fundamentally reshaping the design space for distributed AI in
autonomous systems and contested environments. Compared to centralised infrastructure, which can require 300–380W
or more per GPU, edge platforms operating in the 5–25W range enable scalable deployment across many distributed
nodes, while also being mindful of privacy considerations and data contamination. This reduction in power consumption,
combined with hardware acceleration, allows for efficient on-device inference while freeing the host CPU for concurrent
tasks such as sensor processing, communications, and control. As a result, individual nodes can operate as independent
intelligent agents, supporting decentralised architectures that are more resilient to network disruption, jamming, or
single points of failure with the proliferation of cyber attacks and data corruption. This becomes increasingly important
in environments with multi-level security data streams or nodes.
5.2 Energy-Constrained Autonomous Systems
These characteristics are particularly impactful in autonomous and defence contexts, such as drone operations and
distributed sensing environments. With constrained onboard energy budgets (e.g., 90–260Wh/kg [24]), drones benefit
significantly from efficient inference that minimises power draw while enabling local decision-making and real-time
data summarisation. This reduces the need for continuous high-bandwidth communication with centralised command
systems, lowering both latency and detectability in contested environments. More broadly, distributed edge AI supports
operational resilience, scalability, and stealth, but also introduces new challenges around coordination, trust, and security
across nodes. As edge capabilities continue to improve, a key question emerges: whether centralised AI architectures
become a strategic liability in scenarios where distributed, low-power intelligence can operate effectively at the tactical
edge offering greater privacy, especially when deployed through federated learning.
5.3 Distributed Satellite Ground Stations
SBCs are becoming widely used for deploying portable, cost-effective satellite ground stations across distributed
geographic locations [25], with the Raspberry Pi particularly established as an accessible platform for open-source
ground station implementations [26]. Our results demonstrate that 1.5B parameter LLMs can run on these platforms
at 2–10 tok/s with energy consumption as low as 0.57 MJ/Mtok, with a strong design incentive to use a hardware
accelerator such as the Hailo-10H NPU for improved power efficiency, throughput, and to free the host CPU for
concurrent ground station tasks such as antenna tracking, signal demodulation, and telemetry logging. Embedding LLM
reasoning directly on the ground station enables a degree of intelligent autonomy: parsing and interpreting incoming
7

## Page 8

APREPRINT- APRIL29, 2026
commands, reasoning over sensor telemetry and system state to make best-available decisions, and adapting station
behaviour during narrow satellite communication windows without depending on instructions from a centralised facility.
6 Conclusion
This paper presented initial steps toward a multi-dimensional evaluation framework for LLM inference on IoT-suitable
edge hardware, introducing two composite metrics: throughput density (Tps/m3) and energy per million tokens
(MJ/Mtok). These metrics extend conventional single-axis benchmarking to account for the physical and energy
constraints of edge deployment. Benchmarking across four platforms revealed that hardware accelerators can deliver up
to 40×energy efficiency gains over CPU-only inference. The M5Stack AX630c achieves the highest throughput density
for space-constrained applications, while the Jetson Orin Nano GPU offers the best absolute throughput and the RPi 5 +
Hailo-10H provides a strong balance of efficiency and form factor. These findings have direct implications for Internet
of Everything deployments in critical industries, from resilient distributed cyber-physical systems to energy-constrained
autonomous platforms and satellite ground stations. Future work will expand the framework with additional composite
metrics, evaluate domain-specific inference tasks for OT applications, assess resiliency from attacks, and benchmark
larger models as vendor support for edge accelerators matures.
Availability of Data and Materials:The data supporting the findings of this study are openly available from the Open
Science Framework athttps://osf.io/5r9t4.
Ethics Approval:Not applicable.
Conflicts of Interest:The authors declare no conflicts of interest to report regarding the present study.
References
[1]Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and
Illia Polosukhin. Attention is all you need.Advances in neural information processing systems, 30, 2017.
[2]Wei Zhang, Chaoqun Wan, Yonggang Zhang, Yiu-ming Cheung, Xinmei Tian, Xu Shen, and Jieping Ye. Inter-
preting and improving large language models in arithmetic calculation. InProceedings of the 41st International
Conference on Machine Learning, pages 59932–59950, 2024.
[3]Carlos Gómez-Rodríguez and Paul Williams. A confederacy of models: A comprehensive evaluation of llms on
creative writing.arXiv preprint arXiv:2310.08433, 2023.
[4]Dana Brin, Vera Sorin, Eli Konen, Girish Nadkarni, Benjamin S Glicksberg, and Eyal Klang. How large language
models perform on the united states medical licensing examination: a systematic review.MedRxiv, pages 2023–09,
2023.
[5]Daniel Martin Katz, Michael James Bommarito, Shang Gao, and Pablo Arredondo. Gpt-4 passes the bar exam.
Philosophical Transactions of the Royal Society A, 382(2270):20230254, 2024.
[6]Vasiliki Liagkou, George Fragiadakis, Evangelia Filiopoulou, Mara Nikolaidou, and Christos Michalakelis.
Taming the llmaas market: A decision-making framework utilizing diverse enterprise-critical selection factors.
Available at SSRN 5406285, 2025.
[7]Yashothara Shanmugarasa, Ming Ding, Chamikara Mahawaga Arachchige, and Thierry Rakotoarivelo. Sok: The
privacy paradox of large language models: Advancements, privacy risks, and mitigation. InProceedings of the
20th ACM Asia Conference on Computer and Communications Security, pages 425–441, 2025.
[8]Yue Zheng, Yuhao Chen, Bin Qian, Xiufang Shi, Yuanchao Shu, and Jiming Chen. A review on edge large
language models: Design, execution, and applications.ACM Computing Surveys, 57(8):1–35, 2025.
[9]Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle,
Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models.arXiv preprint
arXiv:2407.21783, 2024.
[10] A Yang Qwen, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengpeng Li, Dayiheng Liu,
Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report.arXiv preprint, 2024.
[11] Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke Zettlemoyer. 8-bit optimizers via block-wise quantization.
arXiv preprint arXiv:2110.02861, 2021.
[12] Tianyao Shi and Yi Ding. Systematic characterization of llm quantization: A performance, energy, and quality
perspective.arXiv preprint arXiv:2508.16712, 2025.
8

## Page 9

APREPRINT- APRIL29, 2026
[13] Stephan Patrick Baller, Anshul Jindal, Mohak Chadha, and Michael Gerndt. Deepedgebench: Benchmarking deep
neural networks on edge devices, 2021.
[14] Mustafa Abdulkadhim and Sandor R Repas. Introducing leaf: Llm edge assessment framework for generative ai
on the edge.Machine Learning and Knowledge Extraction, 8(2):48, 2026.
[15] Zeinab Nezami, Maryam Hafeez, Karim Djemame, Syed Ali Raza Zaidi, and Jie Xu. Descriptor: Benchmark
dataset for generative ai on edge devices (bedged).IEEE Data Descriptions, 2025.
[16] Donghao Huang and Zhaoxia Wang. Llms at the edge: Performance and efficiency evaluation with ollama on
diverse hardware. In2025 International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE, 2025.
[17] Maximilian Abstreiter, Sasu Tarkoma, and Roberto Morabito. Sometimes painful but certainly promising:
Feasibility and trade-offs of language model inference at the edge.arXiv preprint arXiv:2503.09114, 2025.
[18] Pranay Tummalapalli, Sahil Arayakandy, Ritam Pal, and Kautuk Kundan. Llm inference at the edge: Mobile, npu,
and gpu performance efficiency trade-offs under sustained load.arXiv preprint arXiv:2603.23640, 2026.
[19] Sukhpal Singh Gill, Muhammed Golec, Jianmin Hu, Minxian Xu, Junhui Du, Huaming Wu, Guneet Kaur Walia,
Subramaniam Subramanian Murugesan, Babar Ali, Mohit Kumar, et al. Edge ai: A taxonomy, systematic review
and future directions.Cluster Computing, 28(1):18, 2025.
[20] picovoice. Sub-4-bit llm quantization: Enterprise guide to model compression & accuracy tradeoffs, 2026.
[21] Dongyoung Lee, Seungkyu Choi, and Ik Joon Chang. QRazor: Reliable and effortless 4-bit LLM quantization by
significant data razoring, 2025.
[22] Kilian Tran, Jingxi Zhang, Jérôme Pfeiffer, Andreas Wortmann, and Bianca Wiesmayr. Generating plc code with
universal large language models. In2024 IEEE 29th International Conference on Emerging Technologies and
Factory Automation (ETFA), pages 1–8. IEEE, 2024.
[23] Donghao Huang, Xiuju Fu, Xiaofeng Yin, Haibo Pen, and Zhaoxia Wang. Automating maritime risk data
collection and identification leveraging large language models. In2024 IEEE International Conference on Data
Mining Workshops (ICDMW), pages 433–439. IEEE, 2024.
[24] Tavish Pattanayak and Dimitri Mavris. Battery technology for sustainable aviation: a review of current trends and
future prospects.Applied Energy, 397:126356, 2025.
[25] Wenchang Chai, Jinhong Liu, Ziyue Zhang, Xianjin Xia, Yuanqing Zheng, Ningning Hou, Qiang Yang, Weiwei
Chen, and Tao Gu. Satellite iot in practice: A first measurement study on network availability, performance, and
costs. InProceedings of the 2025 ACM Internet Measurement Conference, IMC ’25, page 891–899, New York,
NY , USA, 2025. Association for Computing Machinery.
[26] Nicolae Cri s,an. Design and implementation of a full-duplex ground station for the qo-100 satellite system based
on sdr and raspberry pi.Acta Technica Napocensis, 64(2):9–14, 2024.
9