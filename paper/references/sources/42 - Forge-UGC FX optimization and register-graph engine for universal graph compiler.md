# Forge-UGC FX optimization and register-graph engine for universal graph compiler

## Page 1

FORGE-UGC: FX OPTIMIZATION& REGISTER-GRAPHENGINE—
UNIVERSALGRAPHCOMPILER
Satyam Kumar1Saurabh Jha1
ABSTRACT
The rise of autonomous AI agents, systems that perceive, reason, and act across heterogeneous compute substrates,
is fundamentally reshaping the demands placed on silicon and the software that programs it. These agents are
inherently heterogeneous: a single inference pipeline may traverse NPUs for dense tensor work, GPUs for flexible
parallel compute, and CPUs for control flow, all within the same SoC. As ASIC-class custom silicon proliferates
to meet this demand, the bottleneck shifts from transistor performance tohardware–software co-design: the
compiler that bridges high-level models and low-level accelerator instruction streams. We define this convergence,
autonomous agents running on heterogeneous custom silicon, unified by intelligent compilation as thefuture of
computing. This paper presents our vision and its first production-ready prototype.
Existing deployment frameworks—OpenVINO and ONNX Runtime—rely on opaque, monolithic compilation
pipelines with static intermediate representations, offering no pass-level visibility, no principled buffer man-
agement, and compilation times that scale super-linearly with model depth. We presentFORGE-UGC(FX
Optimization & Register-Graph Engine — Universal Graph Compiler), a four-phase compiler designed to be
hardware-agnosticby architecture: its frontend capture, middle-end optimization passes, and typed intermediate
representation are decoupled from any specific backend, enabling the same pipeline to target any accelerator
through pluggable backend modules. In this work, we validate FORGE-UGC on Intel’s AI Boost NPU as the first
target backend, with Qualcomm Hexagon, AMD XDNA, and Apple ANE backends planned as future extensions.
Phase 1 captures the computation graph via torch.export ; Phase 2 applies six composable, inspectable
optimization passes—dead code elimination, common subexpression elimination, constant folding, attention
fusion, operator fusion, and layout optimization—reducing graph nodes by 17.4% on GPT-2; Phase 3 lowers the
optimized graph to a typed intermediate representation (NPUIR) with explicit virtual register assignments; Phase 4
performs liveness analysis, linear-scan buffer allocation, and instruction scheduling to minimize NPU ↔CPU
device transitions.
Evaluated on WikiText-103 and GLUE across six model families (125M–8B parameters), FORGE-UGC achieves
6.9–9.2 ×faster compilationthan OpenVINO and ONNX Runtime while delivering18.2–35.7% lower end-
to-end inference latencyand30.2–40.9% lower energy consumption per inference. Numerical fidelity is
confirmed through both perplexity agreement and fine-grained logit-level analysis: max-abs logit differences
remain below 2.1×10−5and KL divergence below 8.4×10−9across all models. We introduce three evaluation
metrics—Fusion Gain Ratio(FGR),Compilation Efficiency Index(CEI), and per-pass execution profiling—
enabling principled ablation. To our knowledge, FORGE-UGC is among the first universal graph compilers
to expose a fully transparent, composable optimization pipeline with formal buffer allocation for transformer
workloads across heterogeneous accelerator targets. The first prototype is ready to ship.
1 INTRODUCTION
We are entering an era where autonomous AI agents—
systems that perceive their environment, reason over multi-
modal inputs, and take consequential actions—are no longer
research curiosities but production requirements. From on-
device personal assistants orchestrating perception and lan-
guage understanding in real time, to industrial edge con-
trollers fusing sensor streams with large language model
reasoning, the demand for agents that operateautonomouslyand locallyis accelerating across every sector. These agents
are inherently heterogeneous: a single inference pipeline
may traverse dense tensor operations on an NPU, flexible
parallel compute on a GPU, and irregular control flow on
a CPU—all within the same system-on-chip, all within a
single power envelope.
This heterogeneity is driving a tectonic shift in silicon
strategy. ASIC-class custom accelerators—Neural Pro-
cessing Units, domain-specific tensor engines, and special-arXiv:2604.16498v1  [cs.AR]  14 Apr 2026

## Page 2

ized inference cores—are proliferating precisely because
no general-purpose processor can deliver the throughput-
per-watt that autonomous agents demand. Yet hardware
alone is insufficient. The true bottleneck ishardware–
software co-design: the compiler infrastructure that trans-
lates a high-level PyTorch model into an optimized instruc-
tion stream for each accelerator, manages data movement
across device boundaries, and does so transparently, com-
posably, and fast enough for iterative development. We
define this convergence—autonomous agents running on
heterogeneous custom silicon, unified by intelligent, univer-
sal compilation—as thefuture of computing.
This paper presents FORGE-UGC, a universal graph com-
piler born from this vision. Starting in December 2025,
the two authors of this paper set out to build the complete
heterogeneous compilation stack from scratch—from Py-
Torch FX graph capture through NPU-specific optimization
passes, typed intermediate representation design, formal
buffer allocation, and liveness-guided instruction scheduling.
What began as a focused research collaboration between
the two of us has produced a first prototype that isready to
ship: validated on Intel’s AI Boost NPU across six model
families spanning 125M to 8B parameters, delivering 6.9–
9.2×faster compilation and 18.2–35.7% lower inference la-
tency than industry-standard frameworks. The speed of this
development—a production-grade, four-phase compiler in
under six months by the two co-authors—itself demonstrates
the architectural thesis: when the compiler is designed as
composable, transparent, and hardware-agnostic from day
one, extending it to new accelerator targets becomes an
engineering exercise rather than a research problem.
1.1 The Heterogeneous Computing Challenge
The future of computing is fundamentallyheterogeneous,
characterized by the integration of low-power Neural Pro-
cessing Units (NPUs), high-throughput GPUs, and general-
purpose CPUs within a unified system. As modern work-
loads evolve beyond static inference toward dynamic,
multi-stage pipelines—spanning perception, reasoning, and
decision-making—no single compute substrate can effi-
ciently execute the entire computation graph. NPUs pro-
vide superior energy efficiency for dense tensor operations,
GPUs offer high parallel throughput and flexibility, while
CPUs handle control flow and irregular computation. Con-
sequently, system performance is no longer determined by
individual accelerators in isolation, but by the efficiency
ofcross-device partitioning, scheduling, and data move-
ment. This shift elevates the role of the compiler from
a single-device code generator to asystem-level orches-
trator, responsible for mapping high-level programs onto
heterogeneous hardware while minimizing latency, energy
consumption, and device transitions. In this setting, a unified
compiler abstraction is essential to transform heterogeneouscollections of accelerators into a cohesive and adaptive com-
pute fabric capable of supporting next-generation edge and
on-device intelligence.
1.2 The NPU Compilation Gap
Neural Processing Units (NPUs) are emerging as dedicated
accelerators for transformer inference on edge devices (In-
tel Corporation, 2024a). Intel’s AI Boost NPU, integrated
into Meteor Lake and Arrow Lake processors, provides up
to 11 TOPS of INT8 throughput at less than 10W thermal
design power—an order of magnitude more power-efficient
than discrete GPUs for memory-bound autoregressive de-
coding (Williams et al., 2009). However, realizing this
efficiency requires compilers that can (i) capture the full Py-
Torch computation graph without lossy intermediate exports,
(ii) apply domain-specific optimizations such as attention
fusion, and (iii) manage the NPU’s constrained buffer hier-
archy through principled register allocation.
Existing deployment frameworks fail to meet these require-
ments. OpenVINO (Intel Corporation, 2022) requires an
intermediate export to its proprietary IR format—a pro-
cess that breaks on models with dynamic control flow, tied
weights, or modern operators (RoPE, GQA, SwiGLU) intro-
duced in PyTorch 2.x. ONNX Runtime (Microsoft, 2021)
suffers from operator coverage gaps where the ONNX opset
lags PyTorch’s ATen library by months, and its Execu-
tion Provider abstraction provides no cost-model guidance
for NPU dispatch. Both frameworks treat the compilation
pipeline as a black box, preventing developers from inspect-
ing which optimizations fired, debugging performance re-
gressions, or conducting principled ablation studies.
1.3 Limitations of Existing Frameworks
We identify five fundamental limitations shared by Open-
VINO and ONNX Runtime that constrain NPU deployment:
Limitation 1—Lossy Export Requirements.Both frame-
works require converting PyTorch models through inter-
mediate formats (TorchScript, ONNX) that cannot repre-
sent modern LLM constructs. TorchScript fails on data-
dependent control flow; ONNX export fails on operators
lacking opset equivalents. Models such as Llama-3, Mis-
tral, and Qwen2 require manual operator decomposition
before export. FORGE-UGC bypasses this entirely by using
torch.export.export() , which operates at the ATen
operator level and handles tied weights, dynamic shapes
within static bounds, and modern architectures natively.
Limitation 2—No Pass-Level Optimization Visibil-
ity.Neither framework exposes individual optimiza-
tion passes. Developers cannot inspect which fu-
sion rules fired, quantify the contribution of each
pass, or conduct ablation studies. There is no

## Page 3

equivalent of FORGE-UGC’s CompilationResult
struct reporting fxnodes before ,fxnodes after ,
fxfused ops, and fxattention fused . This
opacity makes performance debugging impossible and pre-
vents principled optimization.
Limitation 3—Super-Linear Compilation Time.Both
frameworks exhibit compilation times that scale super-
linearly with model depth, reaching 58–62 seconds for 8B-
parameter models—prohibitive for iterative development
and just-in-time deployment scenarios. OpenVINO’s mono-
lithic IR conversion and ONNX Runtime’s EP initialization
dominate compilation time, with no incremental compila-
tion support.
Limitation 4—No Principled Buffer Management.Nei-
ther framework exposes liveness analysis, virtual register
abstraction, or instruction scheduling for NPU deployment.
OpenVINO does not expose a programmable low-level IR
with explicit buffer allocation control. ONNX Runtime
performs memory planning at the EP level without user vis-
ibility, and no exposed liveness analysis or virtual register
abstraction minimizes device transitions. For NPU deploy-
ment, this causes unnecessary CPU-NPU data copies when
operations that could be batched into a single NPU dispatch
are separated by intervening CPU operations.
Limitation 5—No Autotuning for NPU.Neither frame-
work provides systematic exploration of compilation config-
urations (fusion aggressiveness, layout strategy, precision)
for NPU-specific performance. OpenVINO’s hint system
(PERFORMANCE HINT: LATENCY orTHROUGHPUT ) is
coarse-grained. ONNX Runtime’s EP selection is rule-based
and static with no cost-model feedback about whether NPU
execution actually improves performance over CPU fall-
back.
1.4 FORGE-UGC: From Black Box to Transparent
Pipeline
We present FORGE-UGC, a four-phase compiler that ad-
dresses each limitation through principled compiler design.
Critically, FORGE-UGC is architected as auniversal graph
compiler: its frontend graph capture, middle-end optimiza-
tion passes, and typed intermediate representation are en-
tirely backend-agnostic, with hardware-specific logic iso-
lated in pluggable backend modules. In this work, we vali-
date the complete pipeline on Intel’s AI Boost NPU as the
first target backend; however, the same optimization passes
and IR infrastructure are designed to extend to Qualcomm
Hexagon, AMD XDNA, Apple ANE, and other accelerator
targets through backend-specific code generation and dis-
patch modules. The high-level architecture is illustrated in
Figure 1.1.5 Contributions
This work makes the following contributions:
1.Direct FX Graph Compilation Pipeline.We in-
troduce a four-phase compiler pipeline that oper-
ates directly on PyTorch FX graphs captured via
torch.export.export() , entirely eliminating the
lossy intermediate export steps (TorchScript, ONNX)
required by OpenVINO and ONNX Runtime. While
torch.compile with Inductor similarly operates on
FX graphs, it targets CPU and GPU backends only and
does not support Intel NPU dispatch, NNFactory integra-
tion, or NPU-aware buffer allocation. FORGE-UGC’s
novelty lies not in FX graph consumption per se, but in
thecomplete NPU compilation stack built on top of it:
ATen-level capture, NPU-specific optimization passes,
NPUIR lowering, and liveness-guided hardware-aware
scheduling—none of which exist in any current FX-
based framework for Intel NPU targets. By working at
the ATen operator level, FORGE-UGC natively supports
modern LLM constructs—including Rotary Position Em-
beddings (RoPE), Grouped-Query Attention (GQA), and
SwiGLU activations—without manual operator decom-
position. The pipeline also incorporates automatic tied
weight resolution, enabling models such as GPT-2 to be
compiled without user intervention. This design ensures
that any model traceable by torch.export can be
compiled for NPU execution with full semantic fidelity.
2.Composable, Inspectable Optimization Passes.We
design six composable, independently measurable opti-
mization passes—dead code elimination, common subex-
pression elimination, constant folding, attention fusion,
operator fusion, and layout optimization—that collec-
tively reduce graph complexity by 14.2–21.8% across
model families. Attention fusion alone reduces graph
nodes by 14.6% on average by pattern-matching de-
composed multi-head attention subgraphs and replacing
them with single fused dispatches. Each pass reports its
execution time, node delta, and transformation details
through a structured CompilationResult interface,
enabling developers to conduct principled ablation stud-
ies and identify performance bottlenecks—a level of
transparency unavailable in any existing NPU deploy-
ment framework.
3.Typed IR with Formal Buffer Allocation.We design a
typed intermediate representation (NPUIR) with explicit
virtual register assignments, paired with a linear-scan
buffer allocation algorithm that reduces peak buffer count
by 30–48% through liveness-guided reuse. The NPUIR
assigns each instruction an opcode, typed virtual regis-
ters, device placement (NPU or CPU), and a pre-resolved
callable, enabling the instruction scheduler to minimize
NPU↔CPU device transitions by 42–65%. This formal
buffer management layer is, to our knowledge, the first of

## Page 4

Figure 1. FORGE-UGC four-phase architecture.Phase 1: FX graph capture via torch.export with tied weight resolution.Phase 2:
Six composable optimization passes (DCE, CSE, constant folding, attention fusion, operator fusion, layout optimization) with optional
autotuning.Phase 3: Lowering to NPUIR with typed instructions, virtual registers, and device placement.Phase 4: Liveness analysis,
linear-scan buffer allocation, instruction scheduling, and code generation producing aCompiledNPUExecutor.
its kind for NPU compilation of transformer workloads.
4.Novel Evaluation Metrics.We introduce three evalu-
ation metrics—Fusion Gain Ratio (FGR), Compilation
Efficiency Index (CEI), and per-pass execution profiling—
that enable principled compiler comparison and fine-
grained ablation. FGR is acost-model-internal diagnos-
ticthat isolates the impact of fusion passes on estimated
execution cost; CEI quantifies inference speedup deliv-
ered per second of compile time, most relevant for it-
erative development and just-in-time deployment; and
per-pass profiling reveals the cost–benefit tradeoff of
each optimization stage.
5.Comprehensive Empirical Evaluation.We conduct
a thorough evaluation across six model families (GPT-
2 125M, Granite-350M, Qwen2-0.5B, Llama-3.2-1B,
LFM2-2.6B, Llama-3.1-8B) spanning 125M to 8B pa-
rameters on WikiText-103 and GLUE benchmarks.
FORGE-UGC achieves 6.9–9.2 ×compilation speedup
and 18.2–35.7% inference latency reduction versus Open-
VINO and ONNX Runtime, with improvements scaling
consistently with model depth. Numerical fidelity is
validated through both perplexity agreement and fine-
grained logit-level analysis (max-abs diff <2.1×10−5,
KL divergence <8.4×10−9), confirming near-bit-exact
output preservation. The evaluation further demonstrates
high reproducibility (CV <2.5% across all metrics) and
tight P99 tail latency distributions critical for SLA-bound
edge deployments.2 MOTIVATION: THECOMPILER AS A
SYSTEM-LEVELORCHESTRATOR
The autonomous AI agents reshaping industry—from on-
device assistants to robotic controllers to real-time analytics
engines—share a common architectural reality: they are
multi-stage, multi-device pipelines. A single agent inference
pass may begin with a vision encoder on the NPU, route
through a language model whose attention layers run on
the NPU while its embedding lookups and control logic
execute on the CPU, and conclude with a decision head that
dispatches actions back to the host. No single accelerator
can efficiently execute this entire computation graph, and no
single compilation strategy can optimally serve every stage.
Modern edge and on-device AI systems integrate low-power
Neural Processing Units (NPUs), high-throughput GPUs,
and general-purpose CPUs within a single silicon package,
each accelerator offering distinct advantages: NPUs deliver
superior energy efficiency for dense tensor operations, GPUs
provide massive parallel throughput for flexible workloads,
and CPUs handle irregular control flow and host-side orches-
tration. As ASIC-class custom silicon proliferates to serve
the compute demands of autonomous agents, the critical
challenge is no longer building faster individual accelera-
tors, but building thesoftware infrastructurethat transforms
a heterogeneous collection of accelerators into a cohesive,
adaptive compute fabric. This is the hardware–software

## Page 5

co-design imperative: the compiler must understand the
cost characteristics of each accelerator, partition computa-
tion graphs across devices to minimize latency and energy,
and manage data movement across device boundaries—all
transparently and composably.
In this emerging landscape, the compiler occupies a
uniquely strategic position as themiddle layerbetween
high-level application frameworks and low-level hardware
dispatch. Rather than serving as a device-specific code
generator—the traditional compiler role—the compiler must
evolve into asystem-level orchestratorthat understands the
cost characteristics of each accelerator, partitions computa-
tion graphs across devices to minimize latency and energy,
and manages data movement across device boundaries. For
autonomous agents that must operate within strict power
and latency budgets on edge hardware, this orchestration is
not optional—it is the enabling technology. This is precisely
the role FORGE-UGC is designed to fulfill.
FORGE-UGC is architected to serve as the compilation
backend for heterogeneous computing orchestrators such as
QEIL (Kumar & Jha, 2026), which route transformer layers
across CPU, GPU, and NPU devices based on workload-
specific energy models. By integrating FORGE-UGC as the
NPU compilation target within such orchestration frame-
works, the combined system enables not onlydeciding
which layers belong on which accelerator, but alsooptimally
compilingthose layers with attention fusion, operator fusion,
and buffer allocation—closing the loop between workload-
aware routing and hardware-aware compilation. The or-
chestrator gains access to FORGE-UGC’s compilation-time
metrics (FGR, CEI, node reduction, energy estimates) that
can inform routing decisions, while the compiler gains ac-
cess to runtime telemetry (thermal state, memory pressure,
battery level) that can guide autotuning.
This vision motivates FORGE-UGC’s architectural design
choices. The separation between hardware-agnostic opti-
mization passes (Phase 2) and hardware-specific backend
lowering (Phases 3–4) is deliberate: it ensures that as new
accelerator targets—Qualcomm Hexagon, AMD XDNA,
Apple ANE, Samsung NPU—become available, only the
backend modules need to be extended while the entire op-
timization pipeline is reused. The frontend (Phase 1) is
similarly universal, operating on PyTorch FX graphs that
are agnostic to the downstream target. In this work, we
validate the complete pipeline on Intel’s AI Boost NPU;
the results demonstrate that the architecture’s core design
principles—composable passes, typed IR, formal buffer allo-
cation, and liveness-guided scheduling—generalize beyond
any single hardware target and position FORGE-UGC as
a critical component in a broader compute fabric for next-
generation edge intelligence.3 BACKGROUND& RELATEDWORK
3.1 Deep Learning Compiler Landscape
The deployment of neural networks on specialized hardware
has driven the development of domain-specific compilers.
TVM (Chen et al., 2018) established the canonical three-
stage design (frontend capture, middle-end optimization,
backend code generation) using the Relay IR for graph-
level transformations and the TIR for low-level tensor oper-
ations. MLIR (Lattner et al., 2021) introduced a multi-level
IR infrastructure enabling progressive lowering across ab-
straction boundaries. XLA (Google Brain, 2019) pioneered
whole-program optimization with operator fusion and lay-
out optimization for TPU targets. Glow (Rotem et al., 2018)
demonstrated two-phase lowering from high-level graph to
instruction-level IR with quantization and memory planning.
IREE (Intermediate Representation Execution Environ-
ment).IREE (IREE Authors, 2024a) is the most directly
comparable MLIR-based compiler to FORGE-UGC. Built
on MLIR’s progressive lowering infrastructure, IREE pro-
vides end-to-end compilation from high-level frameworks
(TensorFlow, JAX, PyTorch via torch-mlir ) to multiple
hardware backends including CPU, GPU (Vulkan, CUDA),
and experimental accelerator targets. IREE’s architec-
ture shares FORGE-UGC’s philosophy of composable, in-
spectable passes and explicit buffer management. However,
IREE (i) requires model conversion through torch-mlir
or StableHLO, reintroducing the export-gap problem for
modern PyTorch operators; (ii) does not currently provide an
Intel NPU backend or NNFactory integration; (iii) performs
buffer management through MLIR’s built-in buffer deal-
location passes rather than NPU-specific liveness-guided
allocation; and (iv) offers no NPU-specific cost model or au-
totuning for Intel AI Boost targets. FORGE-UGC differs by
operating natively on PyTorch FX graphs at the ATen level,
eliminating the need for MLIR ingestion, and by providing
NPU-specific scheduling and buffer allocation that IREE’s
generic backend infrastructure does not accommodate.
torch.compile and Inductor.PyTorch 2.0 introduced
torch.compile (Ansel et al., 2024), which uses Torch-
Dynamo for bytecode-level graph capture and Inductor as
its primary CPU/GPU backend, generating Triton or C++
kernels. Inductor represents the closest prior work to our
approach: like FORGE-UGC, it operates on FX graphs
captured via torch.export and applies composable op-
timization passes. However, Inductor targets GPU (CUDA
Triton) and CPU (C++ codegen) backends exclusively. The
torch.compile ecosystem does support custom back-
ends through the torch. dynamo.backends registry,
which allows third-party integrations. We considered build-
ing FORGE-UGC as a torch.compile custom backend;
Section 3.2 explains in detail why a dedicated, standalone
compiler pipeline was ultimately preferable for Intel NPU

## Page 6

targets. Critically, neither torch.compile nor Induc-
tor provides NPU-aware instruction scheduling, liveness-
guided buffer allocation for NPU SRAM, or cost-model-
driven autotuning for Intel NPU dispatch—capabilities that
are central to FORGE-UGC’s design.
Qualcomm QNN SDK.The Qualcomm Neural Network
(QNN) SDK (Qualcomm Technologies, Inc., 2023) pro-
vides a deployment compiler for Qualcomm Hexagon NPUs,
offering graph-level operator fusion and quantization for
mobile inference. QNN operates on ONNX or Tensor-
Flow Lite models and generates Hexagon-specific bina-
ries. While QNN demonstrates the value of hardware-
specialized compilation, it (i) requires ONNX/TFLite export
and thus shares the lossy-export limitation with OpenVINO,
(ii) targets Hexagon DSPs rather than Intel’s NNFactory
dispatch model, (iii) provides no pass-level visibility or pro-
grammable buffer management, and (iv) offers no PyTorch
FX integration. FORGE-UGC differs architecturally by op-
erating natively on ATen-level FX graphs and exposing a
fully inspectable, composable pipeline.
Hexagon-MLIR.Concurrently with our work, Absar et
al. (Absar et al., 2026) introduced Hexagon-MLIR, an
open-source MLIR-based compilation stack targeting Qual-
comm’s Hexagon NPU. Hexagon-MLIR ingests both Py-
Torch models (via Torch-MLIR) and Triton kernels (via a
Triton-to-Linalg converter), lowering them through a struc-
tured sequence of MLIR passes—including operator fusion,
tiling for the NPU’s Tightly Coupled Memory (TCM) hier-
archy, HVX vectorization, multi-threading across hardware
vector contexts, and double buffering to overlap DMA trans-
fers with computation. Their generative approach treats
fusion as a first-class compiler pass, enabling specialized
mega-kernels for arbitrary operator chains that maximize
data locality in TCM—a philosophy aligned with FORGE-
UGC’s emphasis on fusion as the most impactful single
optimization. Hexagon-MLIR achieves substantial vector-
ization speedups (up to 63.9 ×for GELU on float16) and
demonstrates effective multi-pass interactions across its op-
timization pipeline.
Hexagon-MLIR and FORGE-UGC are complementary in
both target hardware and architectural approach. Where
Hexagon-MLIR operates within the MLIR ecosystem and
targets Qualcomm’s Hexagon NPU with its HVX vector
extensions and TCM memory hierarchy, FORGE-UGC op-
erates natively on PyTorch FX graphs and targets Intel’s AI
Boost NPU via NNFactory dispatch. The two compilers
share key design principles—composable and inspectable
passes, explicit buffer management, and hardware-aware
scheduling—but arrive at them through different IR strate-
gies: MLIR’s multi-level dialect infrastructure versus FX’s
Python-native graph representation. Notably, Hexagon-
MLIR’s demonstration of effective Triton kernel compi-lation for NPU targets validates the broader thesis that NPU-
specific compilers can match or exceed library-based ap-
proaches, and motivates our planned integration of Triton
kernel support within FORGE-UGC’s pipeline (Section 10).
FORGE-UGC’s architecture is explicitly designed for multi-
backend portability; a future Qualcomm Hexagon back-
end module could leverage insights from Hexagon-MLIR’s
TCM-aware tiling and double-buffering strategies while
reusing FORGE-UGC’s entire frontend and middle-end op-
timization pipeline.
FORGE-UGC follows the three-stage paradigm but dis-
tinguishes itself by operating directly on PyTorch FX
graphs (Reed et al., 2022) rather than requiring model re-
export to a framework-specific IR. This preserves the full
semantic richness of PyTorch’s ATen operator set and avoids
the coverage gaps that plague ONNX and OpenVINO inges-
tion paths.
3.2 Why Nottorch.compilewith a Custom
Backend?
torch.compile with a custom backend is a natural alter-
native design point, and we investigated this path carefully
before committing to the standalone FORGE-UGC architec-
ture. Three fundamental constraints led us to reject it:
(1) Backend API opacity.The torch.compile custom
backend API exposes the optimized FX graph to the back-
end but does not provide hooks for injecting custom IR
passes between Dynamo’s graph capture and the backend’s
code generation. FORGE-UGC requires six composable,
independently measurable passes that must be interleaved
with the graph—a structure not natively supported by the
backend interface. Emulating this within a monolithic back-
end callable would sacrifice the pass-level visibility that is a
primary design goal of FORGE-UGC.
(2) NNFactory incompatibility.Intel’s NNFactory API
(the gateway to AI Boost NPU dispatch) requires an explicit
compile-then-runexecution model: a graph is compiled
into an NNFactory program once and executed as a sin-
gle dispatch unit. The torch.compile execution model
assumes kernels are callable Python/Triton functions—an
abstraction mismatch that would require wrapping each NN-
Factory program in a Python callable with significant over-
head at the dispatch boundary, negating the NPU’s latency
advantage.
(3) No liveness-aware buffer management.
torch.compile ’s memory planning is performed
by Inductor’s buffer scheduler, which is GPU-centric and
not exposed as a pluggable component. FORGE-UGC’s
linear-scan buffer allocator must reason about NPU-specific
live intervals and physical buffer slots—a concern that
Inductor’s abstractions do not accommodate.

## Page 7

Given these constraints, a standalone compiler operat-
ing directly on torch.export.export() graphs pro-
vides a cleaner, more principled path to NPU deployment.
FORGE-UGC’s architecture is deliberatelycomplemen-
taryto torch.compile : future integration could use
Dynamo for graph capture while routing NPU-eligible
subgraphs to FORGE-UGC’s optimization and lowering
pipeline.
3.3 Early Experiment: MLIR-Based Compilation via
IREE-Turbine
Prior to adopting the PyTorch FX-based approach,
we conducted an exploratory experiment using IREE-
Turbine (IREE Authors, 2024b)—the PyTorch frontend for
IREE that converts nn.Module instances to MLIR via the
torch-mlir FX Importer. Using IREE-Turbine’s ahead-
of-time export toolkit, we generated MLIR representations
(in the Torch dialect, subsequently lowered through Linalg)
for our transformer models, with the intent of applying
custom NPU-specific optimization passes and routing the
result to Intel’s AI Boost NPU through a custom backend.
However, we encountered two compounding limitations that
made this path impractical.
First, MLIR’s pass infrastructure is fundamentally C++-
native: custom optimization passes must be implemented
as C++ OperationPass subclasses registered with the
pass manager (Lattner et al., 2021). While MLIR does ex-
pose Python bindings for IR inspection and construction,
these bindings do not support defining custom transforma-
tion passes in Python—all pass logic must be implemented
through the C++ framework or invoked via mlir-opt .
This meant that implementing the six composable, indepen-
dently measurable optimization passes central to FORGE-
UGC’s design (DCE, CSE, constant folding, attention fu-
sion, operator fusion, layout optimization) would have re-
quired a substantial C++ development effort with signifi-
cantly slower iteration cycles than Python-based graph ma-
nipulation.
Second, and more critically, IREE does not provide an In-
tel NPU backend. IREE’s supported targets include CPU
(via LLVM), GPU (via Vulkan/SPIR-V , CUDA, HIP), and
experimental accelerator targets—but no Intel AI Boost
NPU dispatch or NNFactory integration exists. Building a
complete Intel NPU backend within IREE’s C++ compila-
tion infrastructure—including NPU-specific liveness anal-
ysis, buffer allocation mapped to the NPU’s constrained
SRAM hierarchy, and instruction scheduling to minimize
CPU↔NPU device transitions—would have required im-
plementing an entirely new IREE backend in C++, a pro-
hibitive engineering effort. Furthermore, MLIR’s built-in
buffer deallocation and memory planning passes are de-
signed for generic backends and do not accommodate theNPU-specific live interval reasoning needed to minimize
device transitions on Intel hardware.
This experience directly motivated our shift to the PyTorch
FX graph representation, which provides a fully Python-
native, programmatically inspectable graph structure where
custom optimization passes can be implemented, tested,
and iterated upon entirely in Python. The FX approach
enabled us to build the complete FORGE-UGC pipeline—
from graph-level optimizations through NPU-specific buffer
allocation—in a fraction of the development time that an
MLIR-based approach would have required.
3.4 Operator Fusion for Inference
Operator fusion—merging multiple operations into a sin-
gle kernel—is the most impactful single optimization for
inference latency. FlashAttention (Dao et al., 2022) demon-
strated that fusing the Q ·KT, scaling, masking, softmax, and
V multiplication into a single IO-aware kernel reduces atten-
tion from O(N2)memory to O(N) while achieving 2–4 ×
wall-clock speedup. FlashAttention-2 (Dao, 2024) improved
parallelism and work partitioning for further gains.
FORGE-UGC’s FXAttentionFusionPass is
directly inspired by FlashAttention’s IO-awareness
principle: it pattern-matches the decomposed attention
subgraph in the FX IR and replaces it with a single
scaled dotproduct attention dispatch. Unlike
FlashAttention, which targets GPU SRAM, FORGE-UGC
targets NPU dispatch via NNFactory (Intel Corporation,
2024b), enabling fused attention on Intel NPU hardware
where FlashAttention kernels are unavailable.
TASO (Jia et al., 2019) automatically generates graph sub-
stitutions through equivalence verification; DNNFusion (Li
et al., 2021) demonstrated advanced fusion patterns for infer-
ence; Nakandala et al. (Nakandala et al., 2020) addressed fu-
sion for prediction serving. FORGE-UGC combines pattern-
matched fusion (attention, linear+activation) with config-
urable aggressiveness, enabling hardware-specific tuning.
3.5 OpenVINO: Operation and Limitations
OpenVINO (Intel Corporation, 2022) is Intel’s inference
toolkit operating through a four-stage pipeline: (1) a Model
Optimizer converting models to OpenVINO IR (.xml/.bin
format); (2) a runtime Inference Engine dispatching to hard-
ware plugins; (3) hardware-specific plugins translating IR to
device kernels; and (4) NNCF for post-training compression.
OpenVINO IR uses a static graph representation requiring
fully determined tensor shapes at export time.
OpenVINO’s PyTorch ingestion path requires either Torch-
Script or ONNX as intermediaries—both known to fail on
modern LLMs with dynamic control flow, tied weights,
and operators lacking opset equivalents. Its optimization

## Page 8

pipeline is a black box with no pass-level visibility, prevent-
ing ablation studies or debugging. No autotuning mecha-
nism exists for NPU-specific configuration search, and no
programmable low-level IR with explicit buffer allocation
control is exposed.
3.6 ONNX Runtime: Operation and Limitations
ONNX Runtime (Microsoft, 2021) consumes ONNX for-
mat models through three optimization levels (semantics-
preserving, platform-specific, hardware-specific) before dis-
patching to Execution Providers (EPs). PyTorch models
require conversion via torch.onnx.export() , which
traces the model and maps operations to ONNX opset equiv-
alents.
The ONNX opset lags PyTorch’s ATen library, causing mod-
ern LLM components (RoPE, GQA, SwiGLU) to either
lack equivalents or require decomposition into dozens of
primitive ops. Dynamic sequence length support is partial
and EP-dependent. Attention fusion exists for CUDA and
CPU EPs but not for NPU execution via OpenVINO EP.
ONNX export of models with tied weights (GPT-2, OPT)
either duplicates tensors or requires manual preprocessing.
EP selection is rule-based without cost-model feedback, and
no register allocation or instruction scheduling is exposed.
3.7 Intel NPU Acceleration Library
The Intel NPU Acceleration Library (NNAL) (Intel Cor-
poration, 2024b) provides low-level NPU access through
runmatmul() for individual matrix multiplications and
NNFactory for multi-operation graph compilation. While
NNAL enables direct NPU dispatch, it provides no graph-
level optimization, no PyTorch integration, no autotuning,
no buffer management strategy, and limited operator cover-
age beyond matmul and elementwise operations. FORGE-
UGC builds upon NNAL as its NPU backend while provid-
ing the full compiler infrastructure above it.
4 THEFORGE-UGC METHODOLOGY
4.1 Notation and Symbols
Table 1 summarizes the key notation used throughout the
FORGE-UGC methodology.
4.2 Phase 1: FX Graph Capture (Frontend)
FORGE-UGC captures the PyTorch computation graph us-
ingtorch.export.export() , which performs sym-
bolic tracing at the ATen operator level:
G=trace tofx(M, x example )→fx.GraphModule
(1)Table 1.Notation used in the FORGE-UGC methodology.
Symbol Description
G= (V, E)FX computation graph
VSet of graph nodes (operations)
ESet of data-dependency edges
nbefore, nafter Node count before/after optimization
P={p 1, . . . , p K}Set of optimization passes
τ(pk)Execution time of passp k(ms)
R={r 1, . . . , r N}Virtual register set
B={b 1, . . . , b M}Physical buffer set (M≪N)
[si, ei]Live interval of registerr i
INPUIR instruction stream
δ(I)Device transitions inI
where Mis the pretrained model and xexample is an example
input tensor. Unlike TorchScript tracing (which fails on
data-dependent control flow) or ONNX export (which re-
quires opset mapping), torch.export captures the full
ATen-level graph including modern operators (RoPE, GQA,
SwiGLU) without decomposition.
Listing 1 shows the graph capture entry point.
The call to torch.export.export() with
suppress errors=True ensures that trace-time warn-
ings from in-development operators are silenced without
aborting the export. The returned ExportedProgram
carries both the fx.GraphModule (the pure compu-
tation graph) and a complete state dictionary of lifted
parameter and buffer tensors, which the subsequent
wrap exported fornpu() step binds back into the
graph.
1deftrace_to_fx(model, example_input):
2"""Capture model as an fx.GraphModule
at ATen level.
3Returns (graph_module,
ExportedProgram) for NPU lowering.
4"""
5model.eval()
6torch._dynamo.config.suppress_errors
= True
7ep =torch.export.export(model, args
=(example_input,))
8# ep.graph_module: pure FX graph, no
Python side-effects
9# ep carries lifted params/buffers
for weight binding
10returnep.graph_module, ep
Listing 1.FX graph capture viatorch.export(Phase 1
frontend).
4.2.1 Tied Weight Resolution
Models with shared parameters (e.g., GPT-2’s embedding
layer and LM-head share the same weight tensor) require
special handling: torch.export creates a distinct graph
placeholder for each logical parameter name, so a tied
weight appears twice in the placeholder list but must re-

## Page 9

solve to the same physical tensor at dispatch time.
FORGE-UGC’s wrap exported fornpu() imple-
ments this detection by iterating every nn.Module in the
model hierarchy and matching tensoridentities(Python
id()) rather than parameter names:
tied map[n j] =n iifid(W j) =id(W i), j > i
(2)
When a placeholder name resolves to a tensor already
registered under a canonical key, the canonical value is
reused. This preserves memory efficiency without user
intervention—a capability absent from both OpenVINO and
ONNX export.
1formod_name, modinoriginal_model.
named_modules():
2forparam_name, paraminmod.
_parameters.items():
3ifparamisNone:continue
4full_name = f"{mod_name}.{
param_name}"ifmod_name \
5elseparam_name
6ukey = full_name.replace(".", "_"
)
7ifukeynot instate_underscore:
8# tensor identity match: find
the canonical name
9forcanon_ukey, (_, canon_val
)in\
10state_underscore.
items():
11ifcanon_valisparam:
# id() equality
12tied_map[ukey] =
canon_ukey
13break
Listing 2.Tied weight detection in
wrap exported fornpu.
The resulting tied map is consulted during placeholder
binding: any placeholder whose underscore-transformed
name does not appear directly in the state dictionary is re-
solved through the map to its canonical tensor, ensuring that
tied parameters share a single physical buffer throughout
the NPU execution.
4.3 Phase 2: Graph Optimization (Middle-End)
The optimization pipeline applies K= 6 composable passes
sequentially. Each pass pktransforms the graph Gby mutat-
inggm.graph in-place and calling gm.recompile()
at the end:
Gk=pk(Gk−1), k= 1, . . . , K(3)
withG0being the raw captured graph and GKthe fully
optimized graph. All passes inherit from FXPassBase
and expose a single run(gm) -> bool interface thatreturns True if the graph was modified. A fixpoint loop
inrunfxpasses iterates each pass until convergence
(default: 2 rounds), ensuring that earlier passes do not mask
opportunities for later ones.
4.3.1 Pass 1: Dead Code Elimination (FXDCEPass)
Dead code elimination performs a backward reachability
walk from the graph’s output node. Only nodes that can
be reached by traversing allinput nodes in reverse
are marked live; all others are erased:
Vlive=backward reachable(V output, E)(4)
G′=G[V live],|V′| ≤ |V|(5)
Unreachable nodes ( V\V live) are erased in a single forward
pass. This removes debugging artifacts, gradient-related
branches, and dead sub-expressions introduced during graph
capture (Muchnick, 1997).
1defrun(self, gm) ->bool:
2graph = gm.graph
3# find the single output node
4output_node =next(nforningraph.
nodes
5ifn.op == "output
")
6# backward BFS from outputs
7live =set()
8stack =list(output_node.
all_input_nodes)
9whilestack:
10n = stack.pop()
11ifninlive:continue
12live.add(n)
13stack.extend(n.all_input_nodes)
14# erase unreachable call nodes
15to_erase = [nforningraph.nodes
16ifn.opnot in("output",
"placeholder")
17andnnot inlive]
18forninto_erase:
19graph.erase_node(n)
20return len(to_erase) > 0
Listing 3.DCE: backward reachability walk and node erasure
(FXDCEPass).
4.3.2 Pass 2: Common Subexpression Elimination
(FXCSEPass)
CSE identifies nodes that compute identical operations on
identical inputs and replaces all but the first occurrence with
a reference to the canonical result:
∀vi, vj∈V:op(v i) =op(v j)
∧args(v i) =args(v j)⇒v j7→vi(6)
This is implemented viahash-consingof (target,
arg-tuple, kwargs-tuple) triples (Click, 1995).

## Page 10

The fxnode key helper converts FX node references
within arguments to their .name strings (stable unique iden-
tifiers within the graph), so two nodes are considered equal if
and only if they call the same operator on the same producer
nodes:
1def_fx_node_key(node):
2"""Canonical (op, args, kwargs)
triple for CSE."""
3def_arg_key(a):
4# node references -> stable name
string
5returna.nameif hasattr(a, "name
")elsea
6args =tuple(_arg_key(a)forain
node.args)
7kwargs =tuple(sorted(node.kwargs.
items()))
8return(node.target, args, kwargs)
9
10# --- in FXCSEPass.run ---
11canonical = {}
12fornodein list(graph.nodes):
13ifnode.opnot in("call_function","
call_method",
14"call_module"):
15continue
16key = _fx_node_key(node)
17ifkeyincanonical:
18# redirect all uses to the first
occurrence
19node.replace_all_uses_with(
canonical[key])
20graph.erase_node(node)
21else:
22canonical[key] = node
Listing 4.Hash-consing key for CSE ( fxnode key+
FXCSEPass.run).
4.3.3 Pass 3: Constant Folding (FXConstantFoldingPass)
Constant folding evaluates operations whose inputs are all
compile-time constants and replaces the result with a literal:
∀v∈V:all constant(args(v))⇒v7→eval(v)
(7)
In FORGE-UGC’s FX context, “compile-time constant”
means the operand is a Python scalar literal appearing di-
rectly in node.args . The pass currently folds identity
arithmetic ( x + 0 ,x*1) that arises in shape calcula-
tions, RoPE frequency pre-computation, and dtype-cast
chains introduced during tracing. These patterns occur fre-
quently in transformer graphs because torch.export
preserves every scalar operation visible in the traced byte-
code.
4.3.4 Pass 4: Attention Fusion (FXAttentionFusionPass)
The attention fusion pass is the most impactful single opti-
mization in FORGE-UGC. Standard multi-head attention, astraced by torch.export , appears as a chain of discrete
ATen operations:
Q·KT→scale→[mask]→softmax→ ·V
(8)
Each arrow represents a separate FX node and a separate
NPU dispatch round-trip. The N×N attention score matrix
Sand probability matrix Pare materialized as intermediate
tensors in memory between dispatches.
The pass replaces this entire chain with a single
NPUFusedScaledDotProductAttention module
that calls F.scaled dotproduct attention , di-
rectly inspired by FlashAttention’s IO-awareness princi-
ple (Dao et al., 2022):
SDPA(Q,K,V) =softmaxQ·KT
√dk
·V(9)
The pattern matching begins from every aten.matmul
node and walks forward through an optional scale, optional
mask, a required softmax, optional dropout, and a final mat-
mul with the value tensor. A critical subtlety iskey transpose
unwrapping: the QK matmul computes Q·KT, so the sec-
ond argument is already transposed. SDPA expects the un-
transposed Kand transposes internally; the pass recovers
the original Kby pattern-matching aten.transpose ,
aten.permute, or.t()in the argument:
1def_match_attention_pattern(self,
qk_matmul):
2"""Walk Q@KˆT -> [scale] -> [mask] ->
softmax
3-> [dropout] -> @V. Returns chain
dict or None."""
4chain = {’qk_matmul’: qk_matmul, ’
scale’: None,
5’mask’: None, ’softmax’:
None,
6’dropout’: None, ’pv_matmul’
: None}
7cur = qk_matmul
8# each node must have exactly one
consumer
9[nxt] =list(cur.users)# fails if
branching
10ifself._is_scale(nxt)andself.
_is_scalar_scale(nxt):
11chain[’scale’] = nxt; [nxt] =
list(nxt.users)
12ifself._is_mask(nxt):
13chain[’mask’] = nxt; [nxt] =
list(nxt.users)
14if notself._is_softmax(nxt):return
None
15chain[’softmax’] = nxt; [nxt] =list(
nxt.users)
16ifself._is_dropout(nxt):
17chain[’dropout’] = nxt; [nxt] =
list(nxt.users)

## Page 11

18if notself._is_matmul(nxt):return
None
19chain[’pv_matmul’] = nxt
20returnchain
21
22@staticmethod
23def_unwrap_transpose(node):
24"""Recover original K from KˆT
argument of QK matmul."""
25tgt =str(getattr(node, ’target’, ’’)
)
26if’transpose’intgtand{node.args
[1],node.args[2]} \
27in({-2,-1},{2,3}):
28returnnode.args[0]#
aten.transpose
29iftgt.endswith(’.t’)or’t.default’
intgt:
30returnnode.args[0]# .t
() shorthand
31returnNone
Listing 5.Attention subgraph pattern matching and
K-transpose unwrapping
(FXAttentionFusionPass).
The number of nodes eliminated per attention block is:
∆n attn=nQ·KT+n scale+n mask+n softmax +n V-mul−1(10)
For a model with Ltransformer blocks, the total reduction is
L·∆n attn. The fusion is parameterized by an aggressiveness
threshold α∈[0,1] , where α= 0 disables fusion and α= 1
fuses all matching patterns; the autotuner explores this knob
as part of its configuration search.
4.3.5 Pass 5: Operator Fusion (FXOperatorFusionPass)
The operator fusion pass targets a complementary set of pat-
terns: sequences of a linear projection immediately followed
by a point-wise activation function. These are common at
the output of every FFN sub-layer in transformer models. In
the unoptimized FX graph each linear and each activation
is a separate call function node, dispatched indepen-
dently to the NPU with an intermediate tensor allocation
between them.
FORGE-UGC replaces matched Linear
→ReLU/GELU/SiLU chains with a
NPUFusedLinearReLU/GELU/SiLU module. The
fused module delegates to runnpufused op, which
builds a single NNFactory computation graph containing
both the matmul and the activation, then compiles and
caches it for reuse:
1def_run_npu_fused_op(x, weight, bias,
activation, op_id):
2"""Build an NNFactory graph: matmul +
activation,
3compiled once and cached by op_id."""4cache_key = f"fused_{op_id}_{
activation}"
5ifcache_keynot in_npu_fused_cache:
6try:
7f = NNFactory()
8inp = f.parameter(list(x.
shape))# input
9wt = f.parameter(list(weight
.shape))
10out = f.matmul(inp, wt)
# WˆT x
11ifbiasis notNone:
12bp = f.parameter([1] *(x.
dim()-1)
13+ [
weight.shape[0]])
14out = f.eltwise_add(out,
bp)
15act_fn =getattr(f, ACT_MAP[
activation])
16out = act_fn(out)
# fused act
17f.compile()
18_npu_fused_cache[cache_key] =
(’npu’, f, bias)
19exceptException:
20_npu_fused_cache[cache_key] =
(’cpu’, activation)
21cached = _npu_fused_cache[cache_key]
22# single NPU dispatch --- no CPU
round-trip for activation
23ifcached[0] == ’npu’:
24_, factory, _ = cached
25return torch.from_numpy(
26factory.run(x.numpy(), weight
.numpy()))
Listing 6.Single-pass NNFactory dispatch for fused
Linear+Activation ( runnpufused op).
The fused operation dispatches as a single NNFactory
graph to the NPU, eliminating both the intermediate ma-
terialization of the linear output tensor and the extra NPU
dispatch round-trip for the activation. The pass matches
four fusion patterns: linear+relu ,linear+gelu ,
linear+silu , and mm+add (residual addition after a
raw matrix multiply), covering the full range of activation
functions present in the model families evaluated.
Linear(x)→ReLU/GELU/SiLU(y)
7→NPUFusedLinear{Act}(x)(11)
4.3.6 Pass 6: Layout Optimization
(FXLayoutOptimizationPass)
The layout optimization pass ensures that tensors flow-
ing into NPU-bound operations are in the memory layout
that minimises layout-conversion overhead at the hardware
boundary. The Intel AI Boost NPU operates most efficiently

## Page 12

on contiguous or channels-last (NHWC) tensors. Tensors
that arrive from transpose or permute operations are non-
contiguous and require an implicit copy before the NPU can
consume them.
The pass queries a static NPUPREFERRED LAYOUTS
table—keyed by ATen operator name—
and inserts explicit .contiguous() or
.contiguous(memory format=torch.channels last)
calls at input boundaries where non-contiguous tensors are
detected:
xNCHWtochannels last− − − − − − − − − − − →x NHWC (12)
A secondary sub-pass cancelsredundantconversions: two
consecutive contiguous() calls on the same tensor are
collapsed to one. This prevents the pass from inflating the
graph when applied in a fixpoint loop. Conversions are
inserted at the minimum necessary points—after embed-
ding layers and before the first NPU-dispatched operation—
rather than at every tensor boundary, avoiding unnecessary
memory traffic.
4.4 Phase 3: Lowering to NPUIR
The optimized FX graph is lowered to a typed intermediate
representation (NPUIR) where each FX node becomes an
instruction with explicit metadata:
NPUIROp= (opcode,vreg in,
vreg out,device,callable)(13)
Theopcode encodes the operation class ( npu.module
for NPU-dispatched ops, cpu.aten. *for host-side ATen
ops,cpu.method. *for tensor methods); vreg inand
vreg out are integer virtual register IDs naming input
and output tensors abstractly; device is either ’npu’ or
’cpu’ ; andcallable is the pre-resolved Python function
or module instance to invoke.
The device routing rule is simple and determin-
istic: any call module node whose name con-
tains npulinear ,npufused ,npumm, or
npuaddmm is routed to the NPU; all other nodes (ATen
functions, tensor methods, shape operations) run on the
CPU. This binary classification means the instruction sched-
uler has full visibility into the device assignment of every
instruction before any hardware is touched.
Arguments arefrozenat lowering time: every FX node ref-
erence in node.args is replaced with a RegRef marker
carrying the virtual register ID of the producing instruction.
At runtime the executor resolves these markers from the live
register file without any attribute lookup or graph traversal.1classNPUIROp:
2"""Single typed instruction in the
NPUIR.
3
4Frozen at compile time: args contain
_RegRef markers
5instead of live tensor objects,
resolved at runtime.
6"""
7__slots__ = (’op_id’,’opcode’,’
output_reg’,’input_regs’,
8’device’,’op_type’,’
target’,
9’frozen_args’,’
frozen_kwargs’,’_name’)
10
11defexecute(self, regs:dict):
12"""Resolve _RegRef markers, then
dispatch."""
13args =tuple(_resolve_args(a,
regs)
14forainself.
frozen_args)
15kwargs = {k: _resolve_args(v,
regs)
16fork, vinself.
frozen_kwargs.items()}
17ifself.op_type == ’call_module’:
18returnself.target( *args, **
kwargs)# NPU/CPU
19elifself.op_type == ’
call_function’:
20returnself.target( *args, **
kwargs)# CPU ATen
21elifself.op_type == ’call_method
’:
22return getattr(args[0], self.
target)(# tensor method
23 *args[1:], **
kwargs)
24
25# --- device routing in _lower_node ---
26is_npu =any(tin str(node.target)fort
in
27(’_npu_linear_’,’_npu_fused_’,’
_npu_mm_’,’_npu_addmm_’))
28opcode = "npu.module"ifis_npuelse"cpu
.module"
29device = ’npu’ifis_npuelse’cpu
’
Listing 7.NPUIR instruction structure and device routing
(NPUIRLowering. lower node).
The lowering proceeds by a single topological traversal of
the FX graph (Algorithm 1). Placeholder nodes for model
weights and buffers are resolved to their tensor values and
stored in a constant table; the single input ids place-
holder is assigned the program’s input register. The output
node’s argument determines the output register.

## Page 13

Algorithm 1FX→NPUIR Lowering
Require:Optimized FX graphG K
1:I ←[];vreg counter←0
2:foreach nodevin topological order ofG Kdo
3:op←classify(v){MATMUL, ATTN, etc.}
4:dev←route(v){ NPU if matmul/attn, else CPU }
5:vreg out←vreg counter++
6:vreg in←lookup vregs(args(v))
7: Append (op,vreg in,vreg out,dev,callable(v))
toI
8:end for
9:returnI
4.5 Phase 4: IR Analysis & Optimization
Phase 4 operates entirely on the flat NPUIR instruction
list and produces a CompiledNPUExecutor ready for
direct hardware dispatch. It comprises three sub-stages: live-
ness analysis, linear-scan buffer allocation, and instruction
scheduling.
4.5.1 Liveness Analysis
For each virtual register ri, we compute itslive interval
[si, ei]where siis the instruction index of the first write
(the unique instruction whose output reg equals ri) and
eiis the instruction index of the last read:
si= min
j:output reg(I j)=rij, e i= max
j:ri∈input regs(I j)j
(14)
Two virtual registers ri, rjareinterference-freeif their
live intervals do not overlap: [si, ei]∩[s j, ej] =∅ . Non-
interfering registers can share the same physical buffer slot.
The liveness analyzer also produces a dead after map
that lists, for each instruction index, the registers whose
last use coincides with that instruction. The executor uses
this map to eagerly free register-file entries and keep peak
memory bounded.
4.5.2 Linear-Scan Buffer Allocation
We employ the classic linear-scan register allocation algo-
rithm (Poletto & Sarkar, 1999) to map Nvirtual registers
toMphysical buffer slots ( M≪N ). The algorithm is
O(NlogN) in the number of live intervals—a substantial
improvement over the O(N2)graph-coloring approaches
used internally by OpenVINO. The implementation main-
tains a free pool of released physical buffer slots and an
activelist of intervals still in flight:
1@staticmethod
2defallocate(lifetimes, pinned=None):
3"""Map N virtual regs to M physical
buffers (M << N)."""
4pinned = pinnedor set()
5# sort by interval start for greedy
left-to-right scan6sorted_regs =sorted(lifetimes,
7key=lambdar:
lifetimes[r][0])
8reg_to_buf, free_bufs, active = {},
[], []
9next_buf = 0
10
11forreginsorted_regs:
12start, end = lifetimes[reg]
13# expire intervals that ended
before this one starts
14still_alive, freed = [], []
15for(end_t, buf_id)inactive:
16(freedifend_t < startelse
still_alive)\
17.append((end_t, buf_id))
18active = still_alive
19free_bufs.extend(bfor_, bin
freed)
20
21ifreginpinnedor notfree_bufs
:
22buf = next_buf; next_buf += 1
# allocate new
23else:
24buf = free_bufs.pop(0)
# reuse expired
25reg_to_buf[reg] = buf
26active.append((end, buf))
27
28returnreg_to_buf, next_buf
Listing 8.Linear-scan buffer allocation
(BufferAllocator.allocate).
Algorithm 2Linear-Scan Buffer Allocation
Require:Live intervals{[s i, ei]}N
i=1
1:Sort intervals bys i(ascending)
2:active← ∅;free pool← {b 1, . . . , b M}
3:foreach interval[s i, ei]in sorted orderdo
4: Expire intervals in active where ej< si; return
their buffers tofree pool
5:iffree pool̸=∅then
6:Assignr i7→pop(free pool)
7:else
8:Allocate new bufferb M+1 ;M←M+ 1
9:end if
10:Add(r i, ei)toactive
11:end for
The buffer reduction ratio is:
ρbuf= 1−M
N(15)
where ρbuf= 0.30 –0.48 for transformer models in our ex-
periments, meaning 30–48% fewer physical buffers than
virtual registers.

## Page 14

4.5.3 Instruction Scheduling
The scheduler reorders NPUIR instructions to minimize
device transitions while respecting data dependencies:
I∗= arg min
I′∈topo valid(I)δ(I′)(16)
whereδ(I′)counts the number of NPU↔CPU transitions:
δ(I′) =|I′|−1X
j=11[dev(I′
j)̸=dev(I′
j+1)](17)
The scheduler implements a priority-based topological sort
over the dependency graph of NPUIR instructions: at each
step, among all instructions whose data dependencies are
already satisfied, it first picks an instruction on thesame
deviceas the most recently scheduled instruction. When no
same-device instruction is ready, it falls back to any ready
instruction. This greedy device-affinity heuristic clusters
consecutive NPU operations and consecutive CPU opera-
tions into maximal contiguous runs, reducing δby 40–65%
compared to the natural FX node ordering.
Each device transition incurs approximately 0.3–0.8 ms of
overhead from PCIe/MMIO data movement between the
host and the NPU SRAM. On the 32-layer Llama-3.1-8B
model, the scheduler reduces transitions from 264 to 93,
eliminating 50–130 ms of per-inference overhead and ac-
counting for 11.2% of the total latency improvement.
4.5.4 Code Generation: CompiledNPUExecutor
The output of Phase 4 is a CompiledNPUExecutor that
runs the flat, pre-scheduled instruction stream directly:
1defexecute(self, input_ids) -> np.
ndarray:
2"""Run the compiled NPUIR program.
3No FX graph walk, no Python attribute
lookup at runtime.
4"""
5# initialise register file from pre-
loaded constants
6regs =dict(self.constants)
7regs[self.input_reg] = to_tensor(
input_ids, self.seq_len)
8
9with torch.no_grad():
10foridx, opin enumerate(self.ops
):
11# dispatch: NPU or CPU, pre-
resolved callable
12result = op.execute(regs)
13ifop.output_reg >= 0:
14regs[op.output_reg] =
result
15# eager GC: free registers
that are no longer live
16fordead_reginself.dead_map
.get(idx, []):17regs.pop(dead_reg, None)
18
19out = regs[self.output_reg]
20returnout.numpy()if isinstance(out,
torch.Tensor) \
21elseout
Listing 9.Compiled executor: flat instruction dispatch with
register-file management
(CompiledNPUExecutor.execute).
The executor’s flat instruction loop provides three key prop-
erties compared to a Python FX graph interpreter: (1)no
attribute lookup overheadbecause all callables are pre-
resolved at lowering time; (2)no dynamic memory allo-
cationbecause physical buffer slots are pre-assigned by
the allocator; and (3)deterministic schedulingbecause the
instruction order is fixed at compile time with no runtime fu-
sion decisions. These properties produce the tight P99/P50
latency ratio of 1.20 observed in our experiments (compared
to 1.27–1.28 for OpenVINO and ONNX Runtime).
4.6 NPU Cost Model
FORGE-UGC includes a heuristic cost model that estimates
execution cost without hardware profiling:
Score(G) =w 1·nops+w 2·nweights +w 3·nlinear+w 4·dgraph+w 5·sparams
(18)
where nopsis the op count, nweights the weight tensor count,
nlinearthe fraction of linear operations, dgraphthe graph depth,
andsparams the parameter size. Fusion bonuses multiply the
score when fusion aggressiveness or attention fusion is en-
abled, allowing the autotuner to distinguish configurations
without hardware execution. Lower scores indicate config-
urations better suited for NPU execution. We emphasize
that this cost model is aheuristic proxyfor hardware perfor-
mance; its scores should not be interpreted as proportional
to wall-clock latency (see Section 7.8.3 for discussion).
4.7 Autotuning Compiler
The AutotuningCompiler extends
FXNPUGraphCompiler by systematically search-
ing over the configuration space:
C={α, λ, π, ι}where (19)
•α∈ {0.2,0.4,0.6,0.8,1.0}: fusion aggressiveness,
•λ∈ {auto,channels-last,contiguous}: layout strategy,
•π∈ {fp16,int8,mixed}: NPU precision,
•ι∈ {1,2,3}: max fixpoint iterations.
The search generates |C|= 45 candidate configurations,
compiles each using the cost model (no hardware execution
required), and selects:
c∗= arg min
c∈CScore(G K(c))(20)

## Page 15

This completes in under 200ms per model—negligible com-
pared to a single compilation. Table 1 summarises how
the six listings presented in this section map to the phases:
Listings 1–2 cover Phase 1 (frontend capture); Listings 3–6
cover the first five optimization passes of Phase 2; Listing 7
covers Phase 3 lowering; and Listings 8–9 cover Phase 4
allocation, scheduling, and code generation.
5 NOVELEVALUATIONMETRICS
We introduce three metrics that enable principled compiler
comparison beyond raw latency.
5.1 Metric 1: Pass Execution Time per Pass
Each optimization passp kis individually timed:
τ(pk) =t end(pk)−t start(pk)[ms] (21)
This isolates which passes contribute most to compilation
overhead versus speedup, enabling informed decisions about
which passes to enable for latency-sensitive deployments.
5.2 Metric 2: Fusion Gain Ratio (FGR)
FGR measures the impact of operator and attention fusion
on thecost model’s estimated execution cost, decoupled
from layout optimization or constant folding:
FGR=CostModel(α= 0)
CostModel(α= 1.0)(22)
A value >1.0 means fusion reduces the cost model’s esti-
mated cost; a larger ratio indicates stronger estimated fu-
sion benefit.Important caveat:FGR is a cost-model-
internal diagnostic—it quantifies how much fusion reduces
the heuristic score (Eq. 18),notwall-clock latency. Because
the cost model uses a weighted sum of structural features
rather than calibrated hardware timings, FGR values are
not linearly proportional to measured speedup. The corre-
sponding measured latency gains from fusion are reported
separately in Table 15 (16.6–29.6% wall-clock reduction).
FGR’s value lies in providing a hardware-independent, re-
producible diagnostic for comparing fusion effectiveness
across models and compiler configurations.
5.3 Metric 3: Compilation Efficiency Index (CEI)
CEI quantifies the return-on-investment of compilation time
as the ratio of inference speedup (relative to a given baseline
B) to compilation time expressed in seconds:
CEIB=SB
T(s)
compile=LB/LFORGE
T(s)
compile(23)
where LBis the mean inference latency of baseline B(in
ms),LFORGE is the mean inference latency of FORGE-UGC(in ms), SB=LB/LFORGE≥1is the dimensionless latency
speedup ratio, and T(s)
compile is the total compilation time in
seconds. A CEI of 1.0, for instance, means the compiler
delivers a 1×latency speedup per second of compilation—
recovering its compilation cost after one second of cumula-
tive inference.
Since OpenVINO and ONNX Runtime exhibit different
baseline latencies, we instantiate two separate CEI values:
CEI OV=LOV/LFORGE
T(s)
compile,CEI ONNX =LONNX/LFORGE
T(s)
compile
(24)
Both variants share the same compilation-time denominator;
the difference arises solely from the distinct speedup numer-
ators, making the two CEI values directly comparable as a
function of baseline choice. A higher CEI is preferable, as
it indicates more inference benefit is recovered per unit of
compilation investment—a property particularly relevant for
iterative development and just-in-time deployment scenar-
ioswhere models are recompiled frequently. For the more
commoncompile-once-run-millionsproduction deployment
pattern, CEI is less informative since even large compilation
costs are trivially amortized; in that regime, absolute latency
improvement (Table 7) is the primary metric.
6 EXPERIMENTALSETUP
6.1 Hardware Platform
All experiments are conducted on a single workstation with
the specifications listed in Table 2.
Table 2.Hardware platform specifications.
Component Specification
CPU Intel Core Ultra 9 285HX
NPU Intel AI Boost (11 TOPS INT8)
NPU Driver 32.0.100.4514
NPU Memory Shared LPDDR5, 72.7 GB
NPU Location PCI bus 0, device 11, function 0
GPU NVIDIA RTX PRO 5000 Blackwell
RAM 128 GB DDR5-5600
OS Windows 11 24H2
6.2 Models and Architectural Diversity
Experiments span six model families covering 125M–8B
parameters, as detailed in Table 3.
6.2.1 Precision Strategy for Llama-3.1-8B
An 8B-parameter model at fp16 precision requires approx-
imately 16GB of weight storage—exceeding what can be
efficiently dispatched through the 11 TOPS Intel AI Boost
NPU in a single pass. For Llama-3.1-8B, FORGE-UGC
leverages NNFactory’s built-in symmetric int8 weight quan-
tization with fp16 activations (W-int8/A-fp16), reducing

## Page 16

Table 3. Model specifications used in experiments. ThePrecision
column indicates the numerical precision used during NPU in-
ference. Models ≤2.6B use fp16 weights; Llama-3.1-8B uses
NNFactory’s built-in symmetric int8 weight quantization with
fp16 activations to fit within the NPU’s memory and compute con-
straints (see Section 6.2.1).
Model Params Hidden Layers Attn Precision
GPT-2 125M 768 12 MHA fp16
Granite-350M 350M 1024 24 MHA fp16
Qwen2-0.5B 500M 1024 24 GQA fp16
Llama-3.2-1B 1.0B 2048 16 GQA fp16
LFM2-2.6B 2.6B 2560 32 MHA fp16
Llama-3.1-8B 8.0B 4096 32 GQAW-int8/
A-fp16
weight memory to approximately 8GB. This quantization
is appliedat the NNFactory dispatch levelduring Phase 4
code generation, after all graph-level optimization passes
have completed on the fp16 graph. The optimization passes
themselves operate on the unquantized graph and are fully
semantics-preserving; the int8 dispatch introduces a small
quantization error that is captured in our fidelity measure-
ments (Table 6). Additionally, layers that cannot be effi-
ciently dispatched to the NPU (e.g., embedding lookups,
final layer norm) fall back to CPU execution, with FORGE-
UGC’s instruction scheduler minimizing the resulting de-
vice transitions.
For models ≤2.6B parameters, all weights are dispatched at
fp16 precision with no quantization applied.
Both baselines (OpenVINO and ONNX Runtime) use their
respective default precision settings for NPU dispatch,
which similarly apply int8 quantization for the 8B model
through their internal optimization pipelines.
6.3 Datasets
WikiText-103(Merity et al., 2016): Standard language mod-
eling benchmark (103M tokens). We evaluate perplexity and
generation latency using 128-token input sequences with
64-token generation.
GLUE(Wang et al., 2019): Multi-task NLU benchmark.
We measure inference latency on SST-2 (sentiment clas-
sification, 872 dev examples) and MNLI (textual entail-
ment, 9832 dev examples) using batch size 1 with 128-token
padded inputs.
6.4 Baselines
OpenVINO 2024.4(Intel Corporation, 2022): In-
tel’s NPU plugin with default optimization settings
(PERFORMANCE HINT: LATENCY ). Models are ex-
ported via ONNX then converted to OpenVINO IR.
ONNX Runtime 1.19(Microsoft, 2021): Mi-crosoft’s inference engine with OpenVINO Execution
Provider targeting Intel NPU. Models are exported via
torch.onnx.export()with opset 17.
6.5 Evaluation Protocol
All latency measurements use 50 inference iterations after
10 warmup iterations. We report mean, P50, P90, and P99
latency. Compilation time is measured end-to-end including
graph capture, optimization, lowering, and code generation.
All results are averaged over 3 independent runs with fixed
seeds. Raw per-run latencies for all models are provided in
Appendix Table 25.
Numerical fidelity protocol.To verify that FORGE-UGC’s
optimization passes preserve output quality, we measure:
(1) language-model perplexity on the full WikiText-103 val-
idation set (217,646 tokens) and on the concatenated SST-2
+ MNLI development text for GLUE, using a sliding win-
dow of 512 tokens with a stride of 256; (2)max-abs logit
differencebetween pre- and post-compilation outputs on
1,000 randomly sampled sequences; and (3)KL divergence
between pre- and post-compilation output distributions. Per-
plexity agreement confirms coarse-grained semantic preser-
vation; the logit-level metrics provide fine-grained evidence
of numerical fidelity (Table 6).
7 RESULTS
7.1 Compilation Time
Figure 2. Compilation time comparison on GPT-2 (125M).
FORGE-UGC compiles in 1,000ms versus 6,930ms (OpenVINO)
and 7,271ms (ONNX Runtime)—a 6.9× and7.3× speedup re-
spectively.
Table 4 presents compilation times across all model families.
FORGE-UGC’s compilation time scales approximately lin-
early with layer count ( Tcompile≈210·L ms), while Open-
VINO and ONNX Runtime exhibit super-linear scaling
(T∝L1.4) due to their monolithic optimization passes
and IR conversion overhead. The advantage is most pro-
nounced on the largest model (Llama-3.1-8B): 6.7s versus
58.4–62.2s.

## Page 17

Table 4. Compilation time (ms) across model families. FORGE-
UGC achieves 6.9–9.2 ×speedup over both baselines consistently.
We note that 78% of FORGE-UGC’s compilation time is spent
intorch.export graph capture (Section 7.2), which is shared
upstream infrastructure; the FORGE-UGC-specific optimization
and backend phases account for only 22% of total time.
Model FORGE OpenVINO ONNX RT
GPT-2 (125M)1,0006,930 7,271
Granite-350M1,4209,840 10,320
Qwen2-0.5B1,68011,240 11,890
Llama-3.2-1B2,34018,520 19,840
LFM2-2.6B3,85032,160 34,280
Llama-3.1-8B6,72058,430 62,150
Speedup range— 6.9–8.7×7.3–9.2×
7.2 Compilation Phase Breakdown
Figure 3. Compilation phase breakdown for GPT-2 (125M). FX
Capture dominates at 773.5ms (78.4%), while all six optimization
passes complete in 208ms (21.1%). IR lowering, buffer allocation,
and scheduling together require only 8ms (0.8%).
The phase breakdown reveals that FX Capture
(torch.export ) accounts for 78.4% of compila-
tion time. The six optimization passes collectively require
only 208ms, and the backend phases (lowering, allocation,
scheduling) add merely 8ms. This indicates that further
speedup would primarily require faster graph capture,
as the optimization and backend phases are already
near-instantaneous. We note that this FX Capture time
is a property of torch.export itself, shared by any
framework that consumes FX graphs; FORGE-UGC’sown
compilation contribution (passes + backend) completes in
∼216ms. The end-to-end speedup over baselines therefore
reflects both (a) avoiding the additional ONNX/TorchScript
export step that baselines requireon top ofgraph capture,
and (b) FORGE-UGC’s lightweight pass and backend
design.
Figure 4. Graph node count after each optimization pass for GPT-
2 (125M). Attention fusion provides the largest single reduction
(403→344,−14.6%), followed by operator fusion (344 →332,
−3.5%). Total reduction: 403→333 (−17.4%).
7.3 Graph Node Reduction
Table 5 reports node reduction across all models.
Table 5. Graph node reduction across model families. Attention
fusion consistently provides the largest reduction.
Model Initial Final∆%
GPT-2 (125M) 403 333−70−17.4
Granite-350M 782 636−146−18.7
Qwen2-0.5B 804 652−152−18.9
Llama-3.2-1B 562 468−94−16.7
LFM2-2.6B 1,068 834−234−21.9
Llama-3.1-8B 1,124 896−228−20.3
Mean— — —−18.8%
7.4 Numerical Fidelity
Table 6. Numerical fidelity analysis.PPL Pre/Post: perplexity be-
fore and after FORGE-UGC compilation on WikiText-103.Max-
Abs∆Logit: maximum absolute difference between pre- and
post-compilation logits across 1,000 test sequences.KL Div:
KL divergence between pre- and post-compilation output distribu-
tions. For models using fp16 dispatch (125M–2.6B), the near-zero
differences confirm that the graph-level optimization passes are
semantics-preserving within floating-point rounding. For Llama-
3.1-8B (W-int8/A-fp16 dispatch), the slightly higher but still negli-
gible max-abs diff reflects int8 weight quantization at the NNFac-
tory dispatch level, not the optimization passes themselves.
Model PPL Pre PPL PostMax-Abs
∆LogitKL Div
GPT-2 (125M) 29.41 29.416.2e−6 1.8e−10
Granite-350M 16.22 16.228.4e−6 3.2e−10
Qwen2-0.5B 14.87 14.877.1e−6 2.7e−10
Llama-3.2-1B 9.76 9.769.8e−6 4.1e−10
LFM2-2.6B 7.52 7.521.2e−5 6.3e−10
Llama-3.1-8B 6.24 6.242.1e−5 8.4e−9
Table 6 provides fine-grained numerical fidelity evidence
beyond perplexity agreement. For all fp16-dispatched mod-
els (125M–2.6B), the max-abs logit difference is below
1.2×10−5—within fp16 rounding tolerance—confirming

## Page 18

that the six optimization passes introduce no numerically
significant error. The KL divergences of 10−10indicate that
the output distributions are statistically indistinguishable.
For Llama-3.1-8B, the slightly larger max-abs diff of
2.1×10−5and KL divergence of 8.4×10−9reflect the
int8 weight quantization applied at the NNFactory dispatch
level (Section 6.2.1), not the graph optimization passes.
This quantization error is consistent with symmetric int8
quantization precision and does not affect perplexity at two-
decimal precision. We acknowledge that perplexity rounded
to two decimal places is acoarsefidelity bound; the logit-
level metrics in Table 6 provide the stronger evidence.
7.5 End-to-End Inference Latency (WikiText-103)
FORGE-UGC consistently achieves the lowest latency
across all model families (Table 7). The advantage scales
with model size: 19.3% improvement on GPT-2 (125M)
growing to 35.7% on Llama-3.1-8B versus ONNX Runtime.
This scaling behavior is attributable to attention fusion—
which eliminates more intermediate nodes as transformer
depth increases—and instruction scheduling, which reduces
device transitions proportionally to layer count.
P99 tail latencies are particularly favorable: FORGE-UGC’s
P99 is 6–15% above its P50, compared to 21–27% for both
baselines. The tighter distribution reflects the deterministic
scheduling and pre-allocated buffers of the compiled execu-
tor, versus the dynamic memory management and runtime
fusion decisions of OpenVINO and ONNX Runtime.
7.6 End-to-End Inference Latency (GLUE)
GLUE results (Table 8) confirm task-agnostic improve-
ments: mean latency reductions of 18.2–35.3% mirror
WikiText-103 patterns, with slightly lower absolute values
due to the classification-only (no autoregressive generation)
workload. The consistency across benchmarks (standard de-
viation <1.2% between WikiText-103 and GLUE relative
improvements) validates that FORGE-UGC’s gains derive
from graph-level optimizations rather than task-specific arti-
facts.
7.7 Energy Efficiency Analysis
Beyond latency and compilation speed, energy consumption
per inference is a critical metric for edge deployment, where
devices operate under strict thermal and battery constraints.
FORGE-UGC’s optimizations—reduced device transitions,
tighter buffer allocation, and fewer dispatched operations—
directly translate to lower energy consumption because each
CPU↔NPU transition incurs data movement overhead that
draws additional power, and longer active inference times
sustain higher system-level power draw.We measure energy consumption per inference by record-
ing system-level power draw (CPU + NPU subsystem) us-
ing Intel’s Running Average Power Limit (RAPL) inter-
face (Intel Corporation, 2024a) during inference, and com-
puting energy as E= ¯Pactive×T inference , where ¯Pactive is the
mean active power during inference execution. FORGE-
UGC exhibits lower average active power ( ∼10.2W) than
OpenVINO ( ∼11.8W) and ONNX Runtime ( ∼12.1W), at-
tributable to two factors: (1) fewer device transitions re-
duce the CPU-side dispatch overhead and associated power
spikes, and (2) pre-allocated buffers eliminate runtime mem-
ory allocation activity that drives additional DRAM power
consumption.
Table 9 reports per-inference energy consumption across
all model families. FORGE-UGC achieves 30.2–40.9%
energy reduction over OpenVINO and 37.0–46.2% over
ONNX Runtime. The energy savings consistently exceed
the corresponding latency savings (18.2–35.7%) by 5–12
percentage points, confirming that FORGE-UGC’s optimiza-
tions yield acompoundingbenefit: lower latency reduces
the time under active power, while fewer device transitions
and tighter buffer management reduce the average power
draw itself. The scaling trend is particularly notable: energy
savings grow from 30.2% on the smallest model (GPT-2,
125M) to 40.9% on the largest (Llama-3.1-8B) versus Open-
VINO, because deeper models have more device transitions
that FORGE-UGC’s instruction scheduler eliminates. For
battery-constrained edge devices operating at thermal limits,
this 36.1% mean energy reduction directly translates to ei-
ther longer battery life or the ability to serve more inference
requests within the same thermal envelope—a critical ad-
vantage for deploying capable language models on-device.
7.8 Novel Metrics Results
7.8.1 Pass Execution Time
Table 10 presents the per-pass execution time measured
on GPT-2 (125M). Operator fusion is the most expensive
pass at 72ms, accounting for 34.6% of total optimization
time, but it provides the second-largest node reduction ( −12
nodes). Attention fusion requires only 38ms yet delivers
the largest single reduction ( −59 nodes), making it the
most cost-effective pass in terms of nodes eliminated per
millisecond—achieving a ratio of 1.55 nodes/ms compared
to operator fusion’s 0.17 nodes/ms, a 9.1× efficiency ad-
vantage. The three lightweight passes—DCE (7ms), CSE
(9ms), and constant folding (11ms)—show zero node re-
duction on GPT-2 because the traced graph contains few
dead nodes or redundant expressions; however, these passes
remain essential for larger models where graph capture intro-
duces more artifacts and redundant subexpressions. Layout
optimization adds one node (the channels-last conversion
marker) while device constant insertion similarly adds a

## Page 19

Table 7. End-to-end inference latency (ms) on WikiText-103. FORGE-UGC achieves 18.2–35.7% lower mean latency than both baselines
across all model families. Latency measured as wall-clock time from input tensor to output logits over 50 warmup-excluded iterations,
averaged over 3 runs. Raw per-run data is provided in Appendix Table 25.
FORGE-UGC (ms) OpenVINO (ms) ONNX Runtime (ms)
Model Mean P50 P90 P99 Mean P50 P90 P99 Mean P50 P90 P99
GPT-2 (125M) 6.82 6.74 7.31 8.12 8.45 8.38 9.12 10.84 9.13 9.02 9.87 11.52
Granite-350M 9.41 9.32 10.08 11.24 12.67 12.54 13.81 15.93 13.28 13.12 14.55 16.87
Qwen2-0.5B 11.83 11.72 12.64 14.03 15.42 15.28 16.91 19.47 16.21 16.04 17.68 20.35
Llama-3.2-1B 18.24 18.08 19.52 21.67 24.81 24.62 27.16 31.28 26.37 26.14 28.89 33.23
LFM2-2.6B 31.56 31.28 33.74 37.48 45.23 44.87 49.52 57.01 48.14 47.72 52.69 60.64
Llama-3.1-8B 62.48 61.92 66.84 74.23 91.37 90.62 100.01 115.16 97.82 96.98 107.08 123.22
Table 8. End-to-end inference latency (ms) on GLUE (SST-2 + MNLI, batch=1, seq=128). Results confirm that FORGE-UGC’s gains are
task-agnostic.
FORGE-UGC (ms) OpenVINO (ms) ONNX Runtime (ms)
Model Mean P50 P90 P99 Mean P50 P90 P99 Mean P50 P90 P99
GPT-2 (125M) 5.94 5.88 6.37 7.08 7.36 7.30 7.95 9.44 7.95 7.86 8.59 10.03
Granite-350M 8.21 8.13 8.79 9.80 11.04 10.93 12.04 13.88 11.57 11.44 12.68 14.71
Qwen2-0.5B 10.32 10.22 11.02 12.23 13.44 13.32 14.74 16.97 14.13 13.98 15.41 17.74
Llama-3.2-1B 15.91 15.76 17.02 18.89 21.62 21.46 23.68 27.26 22.98 22.78 25.18 28.96
LFM2-2.6B 27.52 27.28 29.42 32.68 39.43 39.10 43.15 49.67 41.97 41.60 45.93 52.86
Llama-3.1-8B 54.50 54.01 58.28 64.72 79.64 78.98 87.16 100.36 85.26 84.53 93.34 107.41
Table 9. Energy consumption per inference (mJ) on WikiText-103.
FORGE-UGC achieves 30.2–40.9% lower energy than OpenVINO
and 37.0–46.2% lower energy than ONNX Runtime. Energy sav-
ings exceed latency savings because FORGE-UGC also reduces
average active power through fewer device transitions and pre-
allocated buffers.
ModelFORGE
(mJ)OV
(mJ)ONNX
(mJ)∆OV∆ONNX
GPT-2 (125M)69.699.7 110.5−30.2%−37.0%
Granite-350M96.0149.5 160.7−35.8%−40.3%
Qwen2-0.5B120.7181.9 196.1−33.6%−38.4%
Llama-3.2-1B186.0292.8 319.1−36.5%−41.7%
LFM2-2.6B322.0533.7 582.5−39.7%−44.7%
Llama-3.1-8B637.31,078.2 1,183.6−40.9%−46.2%
Mean— — —−36.1%−41.4%single node for explicit device placement, reflecting the cost
of explicit device annotations in the NPUIR. All six passes
collectively complete in 208ms, representing only 21.1%
of total compilation time—confirming that the optimization
pipeline itself is not the bottleneck.
Table 10. Per-pass execution time (ms) on GPT-2 (125M). Oper-
ator fusion is the most expensive pass (72ms) but provides the
second-largest node reduction. All six passes complete in 208ms
total.
Pass Time (ms)∆Nodes
DCE 7 0
CSE 9 0
Constant Folding 11 0
Device Constant 21 +1
Attention Fusion 38−59
Operator Fusion 72−12
Layout Optimization 25 +1
Total 208−69
7.8.2 Pass Execution Time Scaling
Table 11 reports how total optimization time and attention
fusion time scale across model families. Optimization time
scales approximately linearly with transformer layer count:
12-layer GPT-2 requires 208ms, while 32-layer Llama-3.1-
8B requires 572ms ( ≈2.75× increase for 2.67× more lay-
ers). Attention fusion time follows the same linear trend,

## Page 20

consistently accounting for 18–19% of total optimization
time across all models. Notably, models with identical
layer counts but different parameter counts show similar
optimization times: Granite-350M and Qwen2-0.5B (both
24 layers) require 382ms and 396ms respectively, despite
Qwen2-0.5B having 43% more parameters. This confirms
that pass complexity is dominated by graph topology (pro-
portional to layer count) rather than tensor dimensionality,
because the optimization passes operate on graph structure
and do not perform tensor-level computation. This linear
scaling ensures that FORGE-UGC’s optimization overhead
remains practical even for production-scale models, with
the 32-layer models requiring under 600ms for all six passes
combined.
Table 11. Total optimization time (ms) and attention fusion time
(ms) across model families. Optimization time scales linearly with
layer count.
Model Layers Opt. Time Attn. Fusion
GPT-2 12 208 38
Granite-350M 24 382 71
Qwen2-0.5B 24 396 74
Llama-3.2-1B 16 284 52
LFM2-2.6B 32 548 102
Llama-3.1-8B 32 572 108
7.8.3 Fusion Gain Ratio (FGR)
Table 12. Fusion Gain Ratio (FGR) across model families. FGR
is acost-model-internal diagnostic: it measures the ratio of cost
model scores with fusion disabled ( α= 0 ) versus fully enabled
(α= 1 ). Because the cost model is a heuristic weighted sum
(Eq. 18) that isnotcalibrated to wall-clock time, FGR values
should not be interpreted as latency speedups. The corresponding
measured latency gains from fusion are 16.6–29.6% (Table 15).
Model Score (α=0) Score (α=1) FGR
GPT-2 (125M) 364.87 8.64 42.3
Granite-350M 718.42 14.82 48.5
Qwen2-0.5B 742.18 15.24 48.7
Llama-3.2-1B 1,246.34 24.68 50.5
LFM2-2.6B 2,482.61 38.92 63.8
Llama-3.1-8B 4,218.47 62.14 67.9
Mean— —53.6
FGR values range from 42.3 to 67.9 (Table 12), indicating
that fusion passes reduce the cost model score by 42–68 ×.
The large FGR magnitudes reflect the cost model’s sensitiv-
ity to per-op dispatch overhead terms, which fusion dramati-
cally reduces by collapsing many operations into single dis-
patches. However,FGR should not be confused with mea-
sured latency speedup: a 67.9 ×FGR for Llama-3.1-8B
corresponds to a 29.6% measured wall-clock latency reduc-
tion (Table 15), because the cost model’s heuristic weighted
sum is non-linear in latency-relevant quantities. The discrep-
ancy is expected and does not indicate a deficiency—FGR’spurpose is to provide areproducible, hardware-independent
diagnosticfor comparing fusion effectiveness across models
and compiler versions, complementing (not replacing) the
measured latency improvements reported throughout this pa-
per. The scaling trend is informative: deeper models benefit
more from fusion because each additional transformer layer
introduces more fusible patterns. The non-fused cost model
scores ( α= 0 ) grow linearly with parameter count, while
after fusion ( α= 1.0 ), scores compress to a narrow range,
demonstrating that fusion effectively amortizes per-layer
cost in the cost model’s estimation.
7.8.4 Compilation Efficiency Index (CEI)
Table 13. Compilation Efficiency Index (CEI) across model fami-
lies. CEIB= (L B/LFORGE)/T(s)
compile where T(s)
compile is compilation
time in seconds. Higher CEI means more latency-speedup ratio
per second of compilation. CEI is most informative for iterative
development and JIT scenarios; for compile-once-run-millions
production deployment, absolute latency (Table 7) is the primary
metric.
Model CEI OV CEI ONNX Compile (s)
GPT-2 (125M) 1.239 1.339 1.00
Granite-350M 0.948 0.994 1.42
Qwen2-0.5B 0.776 0.815 1.68
Llama-3.2-1B 0.581 0.618 2.34
LFM2-2.6B 0.372 0.396 3.85
Llama-3.1-8B 0.218 0.233 6.72
Mean 0.689 0.733—
CEI decreases with model size because compilation time
grows (linearly), but the high absolute values ( >0.2 even
for 8B models) in Table 13 confirm that FORGE-UGC de-
livers substantial speedup relative to its compilation cost.
For GPT-2, a CEI ONNX of 1.339 means the compiler deliv-
ers a 1.34× latency speedup ratio per second of compila-
tion. The monotonic decrease from 1.339 (GPT-2) to 0.233
(Llama-3.1-8B) reflects the linear growth in compilation
time outpacing the sub-linear growth in inference speedup.
We emphasize that CEI is most relevant foriterative devel-
opmentscenarios (model debugging, hyperparameter search,
edge deployment prototyping) where models are recompiled
frequently. In the more commoncompile-once-run-millions
production deployment pattern, even the 6.72s compilation
of the 8B model is trivially amortized over millions of in-
ference calls, making the absolute latency improvement
(18.2–35.7%) the decisive metric.
8 ABLATIONSTUDIES
8.1 Pass-Level Ablation
Removing attention fusion causes a catastrophic 27.6 ×cost
model score increase (Table 14), confirming it as the single
most critical optimization. Operator fusion contributes a

## Page 21

Table 14. Pass ablation on GPT-2 (125M). Each row removes one
pass from the full pipeline. Attention fusion removal causes a
27.6×cost model degradation, confirming it as the most critical
pass.
Configuration Cost Score∆vs. Full
All Passes8.64—
w/o DCE 8.69 +0.6%
w/o CSE 8.64 +0.0%
w/o Constant Folding 8.64 +0.0%
w/o Device Constant 8.64 +0.0%
w/o Attention Fusion 238.34 +2,658%
w/o Operator Fusion 8.84 +2.3%
w/o Layout Opt. 8.62−0.2%
modest 2.3% improvement. DCE, CSE, and constant fold-
ing show minimal impact on GPT-2 specifically because the
traced graph has few dead nodes or redundant expressions;
however, these passes are essential for larger models with
more complex graph structures.
8.2 Cross-Model Ablation: Attention Fusion Impact
Table 15. Attention fusion impact across model families.Mea-
sured wall-clock latencyreduction scales with model depth as
more attention blocks are fused. These measured values comple-
ment the cost-model-based FGR diagnostic (Table 12).
Model Layers w/ Fusion w/o Fusion∆%
GPT-2 12 6.82 8.18−16.6%
Granite-350M 24 9.41 12.32−23.6%
Qwen2-0.5B 24 11.83 15.48−23.6%
Llama-3.2-1B 16 18.24 22.71−19.7%
LFM2-2.6B 32 31.56 44.18−28.6%
Llama-3.1-8B 32 62.48 88.72−29.6%
The latency reduction from attention fusion scales with
model depth (Table 15): 12-layer models gain 16.6%, while
32-layer models gain 28.6–29.6%. This confirms that at-
tention fusion is the key optimization for deep transformer
models on NPU hardware.
8.3 Buffer Allocation Efficiency
Table 16. Buffer allocation statistics across model families. Linear-
scan allocation reduces physical buffer count by 30–48% through
liveness-guided reuse.
Model V-Regs Phys.ρ buf Trans.↓
GPT-2 333 218 34.5% 42.1%
Granite-350M 636 412 35.2% 48.3%
Qwen2-0.5B 652 418 35.9% 49.1%
Llama-3.2-1B 468 324 30.8% 44.7%
LFM2-2.6B 834 478 42.7% 58.2%
Llama-3.1-8B 896 468 47.8% 64.8%
Buffer reduction improves with model depth (Table 16):
8B models achieve 47.8% reduction (896 →468 buffers)
because deeper models have more overlapping live intervalsthat permit reuse. Device transition reduction ( δdecrease)
correlates with buffer reduction, as the instruction scheduler
benefits from the tighter buffer layout.
8.4 Fusion Aggressiveness Sensitivity
Table 17. Fusion aggressiveness αsensitivity on GPT-2 (125M).
α= 1.0 yields the best cost model score; aggressive fusion con-
sistently helps.
αCost Score Nodes Fused Ops
0.0 364.87 403 0
0.2 142.31 392 4
0.4 58.42 378 8
0.6 22.18 358 12
0.8 10.84 342 18
1.08.64 333 24
The cost model score improves monotonically with fusion
aggressiveness (Table 17), confirming that for NPU targets,
aggressive fusion is always beneficial—unlike GPU targets
where excessive fusion can cause register pressure. This is
because NPU execution via NNFactory dispatches entire
fused subgraphs in single calls, eliminating per-op dispatch
overhead.
8.5 Autotuning vs. Default Configuration
Table 18. Autotuning vs. default configuration across model fam-
ilies. Autotuning improves cost model score by 4.2–8.7% while
adding<200ms to compilation.
Model Default Autotuned∆%
GPT-2 8.64 8.28−4.2%
Granite-350M 14.82 13.92−6.1%
Qwen2-0.5B 15.24 14.18−7.0%
Llama-3.2-1B 24.68 22.84−7.5%
LFM2-2.6B 38.92 35.74−8.2%
Llama-3.1-8B 62.14 56.72−8.7%
Autotuning becomes more impactful as model size increases
(Table 18), because larger models have more diverse sub-
graph patterns that benefit from configuration-specific opti-
mization. The autotuning overhead ( <200ms) is amortized
after a single inference iteration for all models.
8.6 Variance and Reproducibility
All metrics exhibit CV <2.5% (Table 19). Node reduction
has zero variance because the optimization passes are deter-
ministic. The low latency variance reflects the pre-allocated
buffers and deterministic scheduling of the compiled execu-
tor, which eliminates the runtime allocation jitter present in
dynamic frameworks. We note that the compiled executor’s
flat instruction loop with pre-resolved callables avoids the
interpreter overhead and dynamic memory management that
contribute to variance in OpenVINO and ONNX Runtime.

## Page 22

Table 19. Variance across 10 independent runs on GPT-2 (125M)
with FORGE-UGC. CV <2.5% across all metrics confirms high
reproducibility. The low latency variance reflects the pre-allocated
buffers and deterministic scheduling of the compiled executor,
which eliminates runtime allocation jitter. We acknowledge that
CV<2.5% is tighter than typical for shared-memory NPU sys-
tems; raw per-run data supporting these statistics is provided in
Appendix Table 25.
Metric Mean Std Dev CV (%)
Compilation Time (ms) 1,000 18.4 1.84
Inference Latency (ms) 6.82 0.14 2.05
P99 Latency (ms) 8.12 0.19 2.34
Node Reduction (%) 17.4 0.0 0.00
FGR 42.3 0.82 1.94
8.7 Comprehensive Cross-Model Summary
Table 20 consolidates all key metrics into a single view.
Across all six model families, FORGE-UGC achieves a
mean latency reduction of 26.1% over the best baseline,
compilation speedup of 86.7%, mean FGR of 53.6 (a cost-
model diagnostic indicating fusion substantially reduces es-
timated execution cost), and mean buffer reduction of 37.8%.
Combined with the 36.1% mean energy reduction (Table 9),
these results demonstrate that FORGE-UGC delivers com-
prehensive improvements across all deployment-critical met-
rics. The improvements scale consistently with model size:
the largest model (Llama-3.1-8B) shows the greatest latency
reduction ( −31.6%), buffer reduction (47.8%), and energy
reduction ( −40.9%), confirming that FORGE-UGC’s op-
timizations become more impactful as models grow. This
positive scaling relationship is particularly significant for
edge deployment scenarios where the most capable mod-
els that fit within an NPU’s power envelope are precisely
the models that benefit most from FORGE-UGC’s compila-
tion. The compilation speedup is remarkably stable across
model sizes ( −85.1% to −88.5%), reflecting the architec-
tural advantage of direct FX graph operation over lossy ex-
port pipelines—an advantage that holds regardless of model
complexity.
8.8 Instruction Scheduling Ablation
Each device transition incurs ∼0.3–0.8ms of overhead from
PCIe/MMIO data movement. As shown in Table 21, re-
ducing transitions from 264 to 93 on Llama-3.1-8B elim-
inates ∼50–130ms of per-inference overhead, accounting
for 11.2% of the total latency improvement. The schedul-
ing benefit compounds with model depth because deeper
models have more opportunities for NPU operation clus-
tering. The transition reduction percentage scales from
41.9% on GPT-2 (12 layers) to 64.8% on Llama-3.1-8B (32
layers), exhibiting a super-linear relationship with depth:
doubling the layer count from 16 to 32 increases the transi-
tion reduction from 44.4% to 64.8% (a 1.46× improvement)because deeper models present longer sequences of consec-
utive NPU-eligible operations that the scheduler can cluster
together. The corresponding latency improvement follows
the same trend, growing from 4.2% on GPT-2 to 11.2% on
Llama-3.1-8B, confirming that instruction scheduling is a
critical optimization for large-scale transformer deployment
on NPU hardware.
8.9 P99 Tail Latency Analysis
FORGE-UGC’s P99/P50 ratio is consistently 1.20 (Ta-
ble 22), meaning P99 is 20% above P50, versus 1.27–1.28
for both baselines. The tighter distribution reflects: (1) pre-
allocated buffers eliminating runtime allocation jitter; (2)
deterministic instruction scheduling with no runtime fusion
decisions; and (3) the compiled executor’s flat instruction
loop with no interpreter overhead. This 6–8 percentage
point improvement in tail stability is critical for latency-
sensitive edge deployments where SLA compliance requires
predictable worst-case behavior.

## Page 23

Table 20. Comprehensive performance summary across all model families on WikiText-103. FORGE-UGC achieves the best latency,
fastest compilation, highest FGR, and highest CEI across all models. The “ ∆vs. Best Baseline” columns report improvement over
whichever baseline (OpenVINO or ONNX RT) performs best on each model.
Mean Latency (ms) ∆vs. Best Baseline FORGE-UGC Metrics
Model FORGE OV ONNX Latency Compile FGR CEI OV ρbuf
GPT-2 (125M) 6.828.45 9.13 −19.3%−85.6% 42.3 1.24 34.5%
Granite-350M 9.4112.67 13.28 −25.7%−85.6% 48.5 0.95 35.2%
Qwen2-0.5B 11.8315.42 16.21 −23.3%−85.1% 48.7 0.78 35.9%
Llama-3.2-1B 18.2424.81 26.37 −26.5%−87.4% 50.5 0.58 30.8%
LFM2-2.6B 31.5645.23 48.14 −30.2%−88.0% 63.8 0.37 42.7%
Llama-3.1-8B 62.4891.37 97.82 −31.6%−88.5% 67.9 0.22 47.8%
Mean — — — −26.1%−86.7% 53.6 0.69 37.8%
Table 21. Instruction scheduling impact across model families. De-
vice transitions ( δ) are reduced by 42–65%, directly translating to
latency improvement.
Modelδ before δafter ∆% Lat.∆%
GPT-2 86 50−41.9%−4.2%
Granite-350M 168 87−48.2%−5.8%
Qwen2-0.5B 174 89−48.9%−6.1%
Llama-3.2-1B 126 70−44.4%−5.1%
LFM2-2.6B 248 104−58.1%−8.4%
Llama-3.1-8B 264 93−64.8%−11.2%
Table 22. P99/P50 latency ratio across frameworks. FORGE-UGC
exhibits the tightest tail distribution (ratio closest to 1.0), critical
for SLA-bound deployments.
Model FORGE OV ONNX RT
GPT-2 1.20 1.29 1.28
Granite-350M 1.21 1.27 1.29
Qwen2-0.5B 1.20 1.27 1.27
Llama-3.2-1B 1.20 1.27 1.27
LFM2-2.6B 1.20 1.27 1.27
Llama-3.1-8B 1.20 1.27 1.27
Mean 1.201.27 1.289 ANALYSIS& DISCUSSION
9.1 Why FORGE-UGC is Faster
FORGE-UGC’s compilation speedup stems from three ar-
chitectural decisions: (1)direct FX graph operationelimi-
nates the model export step (TorchScript/ONNX conversion
accounts for 40–60% of OpenVINO/ONNX RT compila-
tion time); (2)composable passeseach perform a single
well-defined transformation in O(|V|+|E|) time, avoid-
ing the monolithic O(|V|2)pattern-matching of framework-
internal optimizers; and (3)linear-scan allocationruns in
O(NlogN) versus the O(N2)graph-coloring approaches
used internally by OpenVINO. These algorithmic advan-
tages compound: the total compilation complexity is O(K·
(|V|+|E|) +NlogN) , where K= 6 passes, compared to
O(|V|2+N2)for the baselines. This asymptotic advantage
explains why FORGE-UGC’s speedup grows with model
size—from 6.9 ×on GPT-2 to 8.7 ×on Llama-3.1-8B versus
OpenVINO.
We note that 78% of FORGE-UGC’s compilation time is
spent in torch.export graph capture—shared upstream
PyTorch infrastructure that any FX-consuming framework
must invoke. The compilation speedup over baselines is
therefore attributable to two factors: (a) FORGE-UGC
avoids theadditionalexport step to ONNX/TorchScript
that baselines require on top of model loading, and (b)
FORGE-UGC’s own optimization and backend phases are
lightweight ( ∼216ms for GPT-2). The baselines’ compila-
tion overhead comes predominantly from their proprietary
IR conversion and monolithic optimization passes, which
FORGE-UGC’s composable design avoids entirely.
The inference latency advantage comes from four com-
plementary mechanisms: (1)attention fusioneliminates
14.6–21.9% of graph nodes, each of which would otherwise
require a separate NPU dispatch or CPU fallback; (2)oper-
ator fusionreduces CPU ↔NPU transitions by combining
linear+activation into single NNFactory dispatches; (3)in-
struction schedulingreduces device transitions by 42–65%;
and (4)buffer allocationreduces peak memory pressure by
30–48%, minimizing cache thrashing on the NPU’s limited

## Page 24

SRAM. These optimizations are multiplicative rather than
additive: attention fusion reduces the number of operations
that instruction scheduling must arrange, and tighter buffer
allocation enables longer NPU operation clusters without
memory spills.
9.2 Comparison with State-of-the-Art Compilers
FORGE-UGC uniquely combines all capabilities required
for transparent, efficient Intel NPU deployment (Table 23).
TVM offers autotuning but requires model re-export
and does not target Intel NPU. XLA provides whole-
program optimization but is restricted to TPU/GPU tar-
gets. IREE (IREE Authors, 2024a) is the most architec-
turally comparable framework: it provides composable
MLIR-based passes, multi-backend code generation, and
explicit buffer management through MLIR’s buffer deallo-
cation infrastructure. However, IREE requires model con-
version through torch-mlir or StableHLO (reintroduc-
ing export gaps for cutting-edge PyTorch operators), does
not support Intel NPU dispatch or NNFactory integration,
and provides no NPU-specific cost model or autotuning.
torch.compile (Inductor) shares FORGE-UGC’s FX
graph input path and eliminates lossy export, but targets
CPU and GPU kernels exclusively: it provides no Intel NPU
dispatch, no NNFactory integration, no liveness-aware NPU
buffer allocation, and no NPU-specific cost model. Open-
VINO and ONNX Runtime target Intel NPU but lack pass-
level visibility, autotuning, and explicit buffer management.
NNAL provides low-level NPU access but no graph-level
optimization whatsoever. Qualcomm’s QNN SDK demon-
strates hardware-specialized fusion for Hexagon NPUs but
requires ONNX/TFLite export and provides no PyTorch FX
integration or inspectable passes. Hexagon-MLIR (Absar
et al., 2026) provides an MLIR-based compilation stack for
Qualcomm Hexagon NPU with composable passes and Tri-
ton kernel support, but targets a different hardware ecosys-
tem and relies on the MLIR/Linalg IR rather than operating
natively on PyTorch FX graphs.
Critically, FORGE-UGC is, to our knowledge, the only
framework that simultaneously provides direct PyTorch FX
input, tied weight handling, pass-level visibility, NPU-aware
explicit buffer allocation, and liveness-guided instruction
scheduling—capabilities that are individually available in
some frameworks but have not previously been unified in
a single NPU-targeting compiler. The inclusion of both
torch.compile and IREE in Table 23 makes clear that
FX graph access alone (torch.compile) and composable
MLIR passes alone (IREE) are each insufficient: the critical
differentiators are the NPU-specific optimization passes,
the formal buffer management layer, and the NNFactory
dispatch integration that together enable FORGE-UGC’s
latency and compilation-time advantages.9.3 Cross-Dataset Robustness
The remarkable consistency across WikiText-103 and
GLUE (Table 24; standard deviation <0.3% for latency,
exactly 0% for compilation time) confirms that FORGE-
UGC’s gains are graph-level optimizations independent of
the downstream task. This near-zero variance is expected be-
cause FORGE-UGC’s optimizations—attention fusion, op-
erator fusion, buffer allocation, and instruction scheduling—
operate on graph structure rather than on data content. Com-
pilation time shows exactly 0% variance across datasets
because the compiler operates on the model architecture
alone, with no data-dependent compilation paths.
9.4 Latency Scaling Analysis
End-to-end latency scales approximately linearly with pa-
rameter count across all three frameworks, but FORGE-
UGC’s slope is shallower. Fitting a simple linear model to
the six data points (we note this is anapproximatetrend
with limited data points spanning a 64× parameter range,
not a precise predictive model):
LFORGE≈0.007·P+ 6(ms,Pin millions) (25)
LOV≈0.011·P+ 7(ms) (26)
LONNX≈0.011·P+ 8(ms) (27)
The per-parameter cost is approximately 30–35% lower
for FORGE-UGC. This is because fusion and scheduling
gains compound with depth, while the baselines’ per-layer
overhead remains constant. We caution that these fits are
approximate given the limited number of data points and
the wide parameter range.
9.5 Limitations and Future Work
FORGE-UGC has been validated on Intel AI Boost NPU
as the first target backend. While the optimization passes
(Phase 2) and typed IR (Phase 3) are hardware-agnostic by
design, extending FORGE-UGC to additional NPU archi-
tectures (Qualcomm Hexagon, AMD XDNA, Apple ANE,
Samsung NPU) requires implementing new backend dis-
patch modules in Phase 4—an effort that reuses the entire
frontend and middle-end pipeline. The current prototype
uses single-batch inference; production deployment would
benefit from batched execution support. The autotuning
search is grid-based; Bayesian optimization or learned cost
models could further improve configuration selection. The
NPUIR currently supports a fixed set of opcodes; extending
it with custom operator registration would broaden model
coverage.
As discussed in Section 3.3, our early experiments with
MLIR-based compilation via IREE-Turbine revealed two

## Page 25

Table 23. FORGE-UGC vs. existing frameworks across key compiler capabilities. FORGE-UGC is the only framework providing
pass-level visibility, NPU-aware autotuning, explicit buffer allocation, and direct PyTorch FX integration for Intel NPU targets.
torch.compile (Inductor backend) targets CPU/GPU and lacks NPU dispatch, liveness-aware allocation, and NNFactory inte-
gration. IREE provides composable MLIR passes and multi-backend support but requires torch-mlir conversion and has no Intel
NPU backend.
Feature TVM XLA IREE torch.compile OpenVINO ONNX RT NNAL FORGE-UGC
Direct PyTorch FX input No No NoYesNo No NoYes
No lossy export required No Partial NoYesNo No N/AYes
Pass-level visibility Partial NoYesPartial No No N/AYes
Attention fusion (NPU) No Yes Partial No No No NoYes
Operator fusion (NPU) Partial Yes Partial GPU/CPU only Partial Partial NoYes
NPU-aware autotuning Yes No No No No No NoYes
Explicit buffer allocation Internal Internal Internal Internal Internal Internal NoYes (NPU)
Intel NPU target No No No No Yes Yes YesYes
Tied weight handling N/A N/A N/A Partial No No NoYes
Cost model (NPU-specific) Yes No No No No No NoYes
Liveness-aware scheduling Internal Internal Internal Internal No No NoYes
Table 24. Cross-dataset robustness: mean latency improvement
(%) of FORGE-UGC over baselines. Standard deviation <1.2%
confirms task-agnostic improvements.
Metric WikiText-103 GLUE Std Dev
∆vs. OpenVINO−26.4%−26.1% 0.21%
∆vs. ONNX RT−30.2%−29.8% 0.28%
∆Compile Time−85.6%−85.6% 0.00%
compounding limitations: MLIR’s C++-native pass infras-
tructure required substantial development effort for custom
NPU-specific optimizations, and IREE lacks an Intel NPU
backend entirely. This experience confirmed that the Py-
Torch FX graph representation—with its fully Python-native,
programmatically inspectable structure—provides the flex-
ibility and iteration speed required for rapid development
of hardware-specific optimization passes, and motivated the
FORGE-UGC architecture presented in this work.
For models up to 2.6B parameters, all optimizations are fully
semantics-preserving at fp16 precision. For the 8B model,
NNFactory’s int8 weight quantization introduces a small
quantization error at the dispatch level (max-abs logit diff
2.1×10−5); a future quantization-aware compilation path
with explicit precision-accuracy tradeoff controls would
give users finer-grained control over this balance.
The FGR metric currently operates on the heuristic cost
model and is not calibrated to wall-clock latency; future
work could develop a hardware-calibrated cost model that
would make FGR directly interpretable as a latency ratio.
Energy measurements currently rely on system-level RAPL
readings; future work could incorporate per-component
power sensors for more precise energy attribution across
CPU and NPU subsystems.
Beyond these refinements, we are actively developing two
major extensions: (i) Triton kernel compilation within the
FORGE-UGC pipeline, enabling custom NPU kernel de-velopment in Triton’s high-level DSL with automatic low-
ering through our optimization and NPUIR backend; and
(ii) a self-evolving compiler module that leverages runtime
telemetry to progressively refine pass ordering and fusion
strategies across compilations. Both capabilities are cur-
rently under testing and will be released in the next version
of this work.
10 CONCLUSION& FUTUREWORK
This paper presents FORGE-UGC, a four-phase universal
graph compiler validated on Intel NPU that replaces the
opaque, monolithic pipelines of OpenVINO and ONNX
Runtime with a transparent, composable, and formally
grounded compilation framework. By operating directly
on PyTorch FX graphs—and deliberately eschewing the
torch.compile /Inductor path, which lacks NPU dis-
patch integration and liveness-aware buffer allocation—
FORGE-UGC eliminates the lossy export steps that pre-
vent modern LLMs from being deployed on NPU hard-
ware. Six independently measurable optimization passes
reduce graph complexity by 14.2–21.9% across model fami-
lies. Numerical fidelity is confirmed through both perplexity
agreement and fine-grained logit-level analysis (max-abs
diff<2.1×10−5, KL divergence <8.4×10−9), with
the caveat that the 8B model’s slightly higher error reflects
int8 weight quantization at the NNFactory dispatch level
rather than the optimization passes themselves. Linear-scan
buffer allocation maps virtual registers to physical buffer
slots with 30–48% reduction. Instruction scheduling reduces
NPU↔CPU device transitions by 42–65%.
Evaluated on WikiText-103 and GLUE across six model
families (125M–8B parameters), FORGE-UGC achieves
6.9–9.2 ×faster compilation, 18.2–35.7% lower end-to-end
inference latency, and 30.2–40.9% lower energy consump-
tion per inference versus both baselines. Three evaluation
metrics—Fusion Gain Ratio (a cost-model diagnostic for

## Page 26

comparing fusion effectiveness), Compilation Efficiency In-
dex (most informative for iterative development scenarios),
and per-pass profiling—enable principled ablation of NPU
compilation for transformer workloads.
Toward a universal compilation framework.FORGE-
UGC’s architecture is deliberately designed for portabil-
ity beyond Intel NPU. The hardware-agnostic optimiza-
tion passes (Phase 2) and typed intermediate representation
(Phase 3) are decoupled from any specific backend; only
the code generation and dispatch modules in Phase 4 are
target-specific. Extending FORGE-UGC to additional ac-
celerator backends—Qualcomm Hexagon, AMD XDNA,
Apple ANE, Samsung NPU—requires implementing new
backend dispatch modules while reusing the entire opti-
mization pipeline. As discussed in Section 2, this positions
FORGE-UGC not merely as a standalone NPU compiler, but
as a critical middle layer in a broader heterogeneous com-
pute fabric, enabling system-level orchestrators to optimally
compile workloads for whichever accelerator is selected at
dispatch time.
Triton kernel integration and self-evolving compilation.
We are currently testing two capabilities that will be released
in the next version of this work. First, we are integrating
Triton kernel compilationinto the FORGE-UGC pipeline,
enabling developers to author custom NPU kernels in Tri-
ton’s high-level DSL and have them automatically lowered
through FORGE-UGC’s optimization passes and NPUIR
backend—inspired by the Triton-to-NPU compilation path
demonstrated by Hexagon-MLIR (Absar et al., 2026) for
Qualcomm targets, but operating natively within our FX-
based IR and targeting Intel NPU dispatch. This will allow
FORGE-UGC to serve not only as a whole-model compiler
but also as a kernel-level compiler for custom operator devel-
opment. Second, we are developing aself-evolving compiler
module to automatically refine optimization pass ordering,
fusion aggressiveness, and autotuning configurations across
successive compilations. The self-evolving compiler treats
the compilation pipeline itself as a learning system, progres-
sively adapting to the workload characteristics and hardware
constraints of each deployment target. Together, these exten-
sions will transform FORGE-UGC from a static compilation
framework into an adaptive, continuously improving com-
piler infrastructure for heterogeneous edge intelligence.
FORGE-UGC demonstrates that the path to practical accel-
erator deployment lies not in proprietary black-box frame-
works, but in transparent, composable compiler infrastruc-
ture built on open standards. By showing that a PyTorch-
native universal compiler can outperform established indus-
try frameworks on compilation speed, inference latency, and
energy efficiency, this work opens the door to a new gener-
ation of hardware-aware compilers that treat optimization
as a first-class, inspectable, and configurable concern acrossheterogeneous accelerator targets.
A RAWPER-RUNLATENCYDATA
Table 25. Raw per-run mean inference latency (ms) on WikiText-
103 across 3 independent runs for FORGE-UGC. These raw values
support the variance statistics reported in Table 19
Model Run 1 Run 2 Run 3 Mean
GPT-2 (125M) 6.78 6.84 6.83 6.82
Granite-350M 9.38 9.44 9.40 9.41
Qwen2-0.5B 11.79 11.86 11.84 11.83
Llama-3.2-1B 18.19 18.28 18.24 18.24
LFM2-2.6B 31.48 31.62 31.57 31.56
Llama-3.1-8B 62.31 62.58 62.54 62.48
REFERENCES
Absar, M. J., Baskaran, M., Sharma, A., Bhandari, A., Ag-
garwal, A., Rangasamy, A., Das, D., Hosseini, F., Slama,
F., Brumar, I., Verma, J., Bindumadhavan, K., Kothari,
M., Gupta, M., Kolachana, R., Lethin, R., Narang, S.,
Ladwa, S. M., Jain, S., Dalvi, S. S., Rahman, T., Ko-
matireddy, V . R. R., Pandya, V . V ., Shi, X., and Zipper,
Z. Hexagon-MLIR: An AI compilation stack for Qual-
comm’s neural processing units (NPUs).arXiv preprint
arXiv:2602.19762, 2026.
Ansel, J., Yang, E., He, H., Gimelshein, N., Jain, A., V oz-
nesensky, M., Bao, B., Bell, P., Berard, D., Burber, E.,
Chauhan, G., Chourdia, A., Constable, W., Desmaison,
A., DeVito, Z., Ellison, E., Feng, W., Gong, J., Gschwind,
M., Gutteridge, B., Hirsh, S., Huang, Y ., Jain, K., Lazos,
S., Leber, M., Liang, J., Liang, Y ., Lu, Y ., Luk, C., Ma-
her, B., Pan, Y ., Puhrsch, C., Reso, M., Saroufim, M.,
Siraichi, M. Y ., Suk, H., Suo, M., Tillet, P., Wang, E.,
Wang, X., Wen, W., Zhang, S., Zhao, X., Zhou, K., Zou,
R., Mathews, A., Chanan, G., Wu, P., and Chintala, S.
PyTorch 2: Faster machine learning through dynamic
Python bytecode transformation and graph compilation.
InProceedings of the 29th ACM International Conference
on Architectural Support for Programming Languages
and Operating Systems (ASPLOS), pp. 929–947, 2024.
Chen, T., Moreau, T., Jiang, Z., Zheng, L., Yan, E., Cowan,
M., Sber, H., Wang, L., Hu, Y ., Ceze, L., Guestrin, C.,
and Krishnamurthy, A. TVM: An automated end-to-end
optimizing compiler for deep learning. InProceedings
of the 13th USENIX Symposium on Operating Systems
Design and Implementation (OSDI), pp. 578–594, 2018.
Click, C. Global code motion / global value numbering. In
Proceedings of the ACM SIGPLAN Conference on Pro-
gramming Language Design and Implementation (PLDI),
pp. 246–257, 1995.

## Page 27

Dao, T. FlashAttention-2: Faster attention with better par-
allelism and work partitioning. InProceedings of the
International Conference on Learning Representations
(ICLR), 2024.
Dao, T., Fu, D. Y ., Ermon, S., Rudra, A., and R ´e, C. FlashAt-
tention: Fast and memory-efficient exact attention with
IO-awareness. InAdvances in Neural Information Pro-
cessing Systems (NeurIPS), pp. 16344–16359, 2022.
Google Brain. XLA: Compiling machine learning for peak
performance.arXiv preprint, 2019. https://www.
tensorflow.org/xla.
Intel Corporation. OpenVINO toolkit: Open vi-
sual inference and neural network optimization.
https://docs.openvino.ai, 2022.
Intel Corporation. Intel AI boost NPU: Neural processing
unit for Meteor Lake and Lunar Lake architectures.
Intel Technology Documentation, 2024a. https:
//www.intel.com/content/www/us/en/
products/docs/processors/core-ultra.
Intel Corporation. Intel NPU acceleration library.GitHub
Repository, 2024b. https://github.com/intel/
intel-npu-acceleration-library.
IREE Authors. IREE: Intermediate representation execu-
tion environment. https://iree.dev , 2024a. Open-
source MLIR-based compiler and runtime for ML mod-
els. GitHub: https://github.com/iree-org/
iree.
IREE Authors. IREE-Turbine: Pytorch-to-MLIR in-
gestion and compilation. https://github.com/
iree-org/iree-turbine , 2024b. PyTorch fron-
tend for IREE providing torch-mlir-based model inges-
tion and MLIR generation.
Jia, Z., Padon, O., Thomas, J., Warszawski, T., Zaharia, M.,
and Aiken, A. TASO: Optimizing deep learning computa-
tion with automatic generation of graph substitutions. In
Proceedings of the 27th ACM Symposium on Operating
Systems Principles (SOSP), pp. 47–62, 2019.
Kumar, S. and Jha, S. Quantifying edge intelli-
gence: Inference-time scaling formalisms for hetero-
geneous computing. InProceedings of arXiv, 2026.
arXiv:2602.06057v2.
Lattner, C., Amini, M., Bondhugula, U., Cohen, A., Davis,
A., Pienaar, J., Riddle, R., Shpeisman, T., Vasilache, N.,
and Zinenko, O. MLIR: Scaling compiler infrastructure
for domain specific computation. InProceedings of the
IEEE/ACM International Symposium on Code Genera-
tion and Optimization (CGO), pp. 2–14, 2021.Li, W. et al. DNNFusion: Accelerating deep neural networks
execution with advanced operator fusion. InProceedings
of the 42nd ACM SIGPLAN Conference on Programming
Language Design and Implementation (PLDI), pp. 883–
898, 2021.
Merity, S., Xiong, C., Bradbury, J., and Socher, R.
Pointer sentinel mixture models.arXiv preprint
arXiv:1609.07843, 2016.
Microsoft. ONNX runtime: Cross-platform,
high performance ML inferencing and training.
https://onnxruntime.ai, 2021.
Muchnick, S. S.Advanced Compiler Design and Implemen-
tation. Morgan Kaufmann, 1997.
Nakandala, S., Zhang, Y ., and Kumar, A. A tensor com-
piler for unified machine learning prediction serving. In
Proceedings of the 14th USENIX Symposium on Oper-
ating Systems Design and Implementation (OSDI), pp.
899–917, 2020.
Poletto, M. and Sarkar, V . Linear scan register allocation.
ACM Transactions on Programming Languages and Sys-
tems (TOPLAS), 21(5):895–913, 1999.
Qualcomm Technologies, Inc. Qualcomm AI
engine direct SDK (QNN SDK). https:
//developer.qualcomm.com/software/
qualcomm-neural-processing-sdk , 2023.
Accessed: 2025.
Reed, J., DeVito, Z., He, H., Ussery, A., and Ansel, J.
torch.fx: Practical program capture and transformation
for deep learning in Python. InProceedings of Machine
Learning and Systems (MLSys), volume 4, pp. 638–651,
2022.
Rotem, N., Fix, J., Abdulrasool, S., Catron, G., Deng, S.,
Dzhabarov, R., Gibson, N., Hegeman, J., Lele, M., Lev-
enstein, R., Marescotti, J., Padon, O., Park, J., Rber,
A., Reagen, B., Sapra, M., Shi, B., Tulloch, A., Wu,
X., and Smelyanskiy, M. Glow: Graph lowering com-
piler techniques for neural networks. InarXiv preprint
arXiv:1805.00907, 2018.
Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., and
Bowman, S. R. GLUE: A multi-task benchmark and
analysis platform for natural language understanding. In
Proceedings of the International Conference on Learning
Representations (ICLR), 2019.
Williams, S., Waterman, A., and Patterson, D. Roofline:
An insightful visual performance model for multicore
architectures.Communications of the ACM, 52(4):65–76,
2009.