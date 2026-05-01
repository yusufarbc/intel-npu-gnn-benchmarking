# IANUS Integrated Accelerator based on NPU-PIM Unified Memory System

## Page 1

IANUS: Integrated Accelerator based on NPU-PIM
Unified Memory System
Minseok Seo
Seoul National
University
South KoreaXuan Truong
Nguyen
Seoul National
University
South KoreaSeok Joong
Hwang
SAPEON Inc.
South KoreaYongkee
Kwon
SK hynix
South KoreaGuhyun Kim
SK hynix
South KoreaChanwook
Park
SK hynix
South Korea
Ilkon Kim
SK hynix
South KoreaJaehan Park
SK hynix
South KoreaJeongbin Kim
SK hynix
South KoreaWoojae Shin
SK hynix
South KoreaJongsoon Won
SK hynix
South KoreaHaerang Choi
SK hynix
South Korea
Kyuyoung
Kim
SK hynix
South KoreaDaehan Kwon
SK hynix
South KoreaChunseok
Jeong
SK hynix
South KoreaSangheon Lee
SAPEON Inc.
South KoreaYongseok
Choi
SAPEON Inc.
South KoreaWooseok
Byun
SAPEON Inc.
South Korea
Seungcheol
Baek
SAPEON Inc.
South KoreaHyuk-Jae Lee
Seoul National
University
South KoreaJohn Kim
KAIST
South Korea
Abstract
Accelerating end-to-end inference of transformer-based large
language models (LLMs) is a critical component of AI ser-
vices in datacenters. However, diverse compute character-
istics of end-to-end LLM inference present challenges as
previously proposed accelerators only address certain oper-
ations or stages (e.g., self-attention, generation stage, etc.).
To address the unique challenges of accelerating end-to-end
inference, we propose IANUS – Integrated Accelerator based
onNPU-PIM Unified Memory System. IANUS is a domain-
specific system architecture that combines a Neural Pro-
cessing Unit (NPU) with a Processing-in-Memory (PIM) to
leverage both the NPU’s high computation throughput and
the PIM’s high effective memory bandwidth. In particular,
IANUS employs a unified main memory system where the
PIM memory is used both for PIM operations and for NPU’s
main memory. The unified main memory system ensures that
This paper is an updated version of the paper that appeared in the Proceed-
ings of the 29th ACM International Conference on Architectural Support
for Programming Languages and Operating Systems (ASPLOS), April 2024.
Permission to make digital or hard copies of part or all of this work for
personal or classroom use is granted without fee provided that copies are
not made or distributed for profit or commercial advantage and that copies
bear this notice and the full citation on the first page. Copyrights for third-
party components of this work must be honored. For all other uses, contact
the owner/author(s).
ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA
©2024 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-0386-7/24/04.
https://doi.org/10.1145/3620666.3651324memory capacity is efficiently utilized and the movement of
shared data between NPU and PIM is minimized. However,
it introduces new challenges since normal memory accesses
and PIM computations cannot be performed simultaneously.
Thus, we propose novel PIM Access Scheduling that manages
normal memory accesses and PIM computations through
workload mapping and scheduling across the PIM and the
NPU. Our detailed simulation evaluations show that IANUS
improves the performance of GPT-2 by 6.2 ×and 3.2×, on
average, compared to the NVIDIA A100 GPU and the state-
of-the-art accelerator. As a proof-of-concept, we develop a
prototype of IANUS with a commercial PIM, NPU, and an
FPGA-based PIM controller to demonstrate the feasibility of
IANUS.
CCS Concepts: •Computer systems organization →
Heterogeneous (hybrid) systems ;•Computing method-
ologies→Planning and scheduling .
Keywords: Accelerators, Heterogeneous Architectures, Neu-
ral Processing Unit, Processing-in-memory, Large Language
Model, Workload Mapping, Scheduling
ACM Reference Format:
Minseok Seo, Xuan Truong Nguyen, Seok Joong Hwang, Yong-
kee Kwon, Guhyun Kim, Chanwook Park, Ilkon Kim, Jaehan Park,
Jeongbin Kim, Woojae Shin, Jongsoon Won, Haerang Choi, Kyuy-
oung Kim, Daehan Kwon, Chunseok Jeong, Sangheon Lee, Yongseok
Choi, Wooseok Byun, Seungcheol Baek, Hyuk-Jae Lee, and JohnarXiv:2410.15008v1  [cs.AR]  19 Oct 2024

## Page 2

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA Seo et al.
Kim. 2024. IANUS: Integrated Accelerator based on NPU-PIM Uni-
fied Memory System. In 29th ACM International Conference on Ar-
chitectural Support for Programming Languages and Operating Sys-
tems, Volume 3 (ASPLOS ’24), April 27-May 1, 2024, La Jolla, CA,
USA. ACM, New York, NY, USA, 16 pages. https://doi.org/10.1145/
3620666.3651324
1 Introduction
Transformer [ 44], BERT [ 9], and GPT [ 39] have been widely
used for natural language processing (NLP) services at dat-
acenters. Although GPUs are commonly used to accelerate
the inference of deep learning models, GPUs are less effec-
tive in handling transformer models because of multi-head
attention and inference stages that are memory-bound [ 19].
To address the limitations of GPU for transformer models,
many recent works [ 13,14,34,45] have proposed to acceler-
ate multi-head attention through dedicated accelerators and
algorithmic changes; however, these prior work do not fully
address the challenges of end-to-end inference acceleration.
Recently, DFX [ 19] proposed an FPGA-based appliance that
is designed for memory-bound transformer inference stages;
however, it is sub-optimal for the compute-bound stages in
end-to-end inference.
One of the main challenges in accelerating end-to-end in-
ference of transformer-based large language models (LLMs)
is their diverse characteristics, which exhibit a broad range
of computational intensities. For example, GPT includes
complex vector operations, multi-head attention, and fully-
connected (FC) layers that present both compute-bound
matrix-matrix multiplication as well as memory-bound matrix-
vector multiplication. Consequently, to accelerate end-to-end
inference of LLMs, hardware must be capable of efficiently
handling all these diverse operations.
Neural processing units (NPUs) [ 6,21,23] have been widely
proposed to accelerate deep neural networks (DNNs). How-
ever, NPUs are often limited by memory-bound operations
even when high-bandwidth memory is utilized. In compari-
son, processing-in-memory (PIM) [ 8,29,30] minimizes data
movement by enabling computation near memory and pro-
vides higher effective memory bandwidth. Recent PIM chips
[29,30] are effective “domain-specific” memory as they ac-
celerate memory-bound operations by guaranteeing full in-
ternal memory bandwidth utilization for processing units
in memory on domain-specific kernels. However, compute-
bound operations such as matrix-matrix computations or
complex vector operations are not efficient on PIM because
of the limitations of DRAM technology that is highly area-
constrained.
To address the challenges of end-to-end LLM inference,
we propose an NPU-PIM architecture that provides the bene-
fit of both a domain-specific accelerator (i.e., NPU) as well as
a domain-specific memory (i.e., PIM), effectively supporting
a broad range of arithmetic intensities in LLMs. In particu-
lar, we propose IANUS – Integrated Accelerator based onNPU-PIM Unified Memory System.1To the best of our
knowledge, this is one of the first works that integrate
a commercial NPU with a commercial PIM memory
to enable a domain-specific system architecture. Pre-
viously proposed PIM-based systems view PIM as an “ac-
celerator” [ 8,24,26,31] and employ a partitioned memory
system that uses the dedicated memory for the xPU (e.g.,
GPU, CPU) and the PIM accelerator memory. This leads to
inefficient memory capacity usage as shared data between
xPU and PIM tend to be duplicated in both memories for
optimal performance. This is especially problematic for LLM
where parameters of FC layers represent a large portion of
data that need to be shared between the NPU and the PIM.
In light of these challenges, we propose a unified memory
system where PIM memory also serves as the main mem-
ory for the NPU. This approach removes the need for any
data duplication and movement of shared data. However, the
unified memory system in an NPU-PIM system introduces
new challenges as PIM computations and normal memory
accesses cannot be performed concurrently. In this work, we
propose a novel PIM Access Scheduling (PAS) that schedules
PIM computations and normal memory accesses through
mapping and scheduling of the workload on the NPU-PIM
architecture with a unified memory system. The challenges
of PIM computation in a unified memory system include
memory resource conflict with normal memory accesses as
well as the failure to leverage the potential for parallel ex-
ecution with computations performed on the NPU. Thus,
PAS takes into account both resource conflicts and paral-
lelizability of operations between the NPU and PIM to fully
exploit the parallelism across the different resources. We also
demonstrate the proof-of-concept of IANUS by prototyping
the system with an FPGA. In summary, the key contributions
of this work include the following.
1.Architecture : We propose IANUS, a novel heteroge-
neous architecture that combines a dedicated hardware
accelerator (NPU) with a specialized memory (PIM),
to accelerate operations with diverse characteristics
in the end-to-end LLM inference.
2.Unified Memory System & PIM Access Scheduling : Iden-
tifying about 90% of model parameters shared between
the NPU and PIM in the LLM, we propose a unified
memory system where the memory for the NPU and
the PIM memory is shared to efficiently utilize the
memory capacity. We also propose PIM Access Sched-
uling (PAS) that manages the challenges of the unified
memory system with effective workload mapping and
scheduling. Through a detailed simulation of IANUS,
IANUS with PAS achieves 6.2 ×and 3.2×speedup in
1IANUS is a Roman god with two faces that represented the middle ground
between both concrete and abstract dualities. The IANUS architecture in
this work shares similarities as it represents a “middle ground” architecture
between NPU and PIM architectures.

## Page 3

IANUS: Integrated Accelerator based on NPU-PIM Unified Memory System ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA
(a)
(b)
Figure 1. (a) Structure of GPT with vector operations marked
with(V)and (b) multi-head attention mechanism shown in
detail.
GPT-2 compared to the A100 GPU and the state-of-
the-art prior work (DFX [19]), respectively.
3.System integration and FPGA prototyping : To demon-
strate the feasibility of IANUS, we build an integrated
system including a commercial NPU, commercial PIM
chips, and an FPGA-based PIM controller.
2 Background
2.1 Transformer-based LLMs
NLP usually consists of two stages: input token summa-
rization stage ( summarization ) and output token generation
stage ( generation ). While the summarization stage processes
all input tokens collectively, the generation stage deals with
one generated token per stage. In text generation tasks, the
summarization stage initially handles all inputs, followed by
thegeneration stage processing each produced token.
Transformer-based LLMs, such as BERT and GPT, use mul-
tiple encoder or decoder blocks, followed by a task-specific
head (Figure 1a). Each block consists of multi-head attention
module, feed-forward network (FFN) module, layer normal-
ization [ 3], and residual addition [ 15]. During the summa-
rization stage, FC layers typically operate as matrix-matrix
multiplication with multiple input tokens, while in the gener-
ation stage, they perform matrix-vector multiplication with
a single token. The multi-head attention mechanism is de-
picted in Figure 1b. Input tokens ( 𝑥) are multiplied with
weight matrices to generate query ( 𝑄), key (𝐾), and value ( 𝑉)
(1). In the generation stage, new𝐾and𝑉are concatenated
with previous ones. For self-attention, 𝑄,𝐾, and𝑉are split-
ted into multiple heads. The matrix product of query and
transposed key ( 𝑄𝐾𝑇) (2) are executed to compute attention
score (𝑆) (3) and then output ( 𝑆𝑉) (4) within each head is
generated. Finally, the outputs of all heads are merged and
processed by the following FC layer ( 5).
(a)
(b)
Figure 2. Generation stage of GPT-2 XL (a) Latency and
FLOPs breakdown of decoders. (b) Latency breakdown of
self-attention. Results are obtained using an A100 GPU.
2.2 Platforms for DNN Inference
Domain-specific Accelerators : DNN accelerators [ 2,5,17,
23,27,33,36] mainly focus on accelerating convolution com-
putation. Therefore, these accelerators often face bandwidth
bottlenecks during the generation stage of LLMs, primar-
ily involving matrix-vector multiplication. To tackle this
problem, DFX [ 19], an FPGA-based appliance, maximizes
bandwidth utilization by designing peak FLOPS to match the
memory bandwidth. However, while providing significant
benefits on the generation stage, the benefits of DFX on the
summarization stage are small because of limited FLOPS.
Processing-in-Memory : PIM refers to the technology
of implementing processing units inside memory to accel-
erate specific workloads or save energy consumption. Re-
cently, PIM based on commercial DRAMs have been an-
nounced (Accelerator-in-Memory (AiM) [ 26,30], HBM-PIM
[29,31], and UPMEM-PIM [ 8]). They are suitable for memory-
intensive workloads by utilizing all- or half-bank parallelism.
As a result, they are considered promising solutions for gen-
eration stages of LLMs because LLMs include matrix-vector
multiplication in generation stage.
3 Motivations
In this section, we show the diverse computation require-
ments of LLMs and present challenges in designing the accel-
erator system for their end-to-end inference. This motivates
the need for a heterogeneous architecture that combines a
domain-specific accelerator with high compute capability
and a domain-specific memory with high memory band-
width. We also demonstrate the motivation of a unified main
memory organization in an NPU-PIM system for LLMs.
3.1 Diverse Computational Requirements of LLMs
The generation and the summarization stages exhibit differ-
ent computational characteristics as the generation stage

## Page 4

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA Seo et al.
of LLMs is often memory-bound with matrix-vector oper-
ations while summarization stage is compute-bound with
matrix-matrix operations. While compute-bound compo-
nents are well-matched to compute accelerators (e.g., NPU,
GPU), memory-bound operations are not. For example, when
generating two tokens with 512 input tokens, the generation
stage requires 512×fewer FLOPs compared to the summa-
rization stage. However, the execution time of the generation
stage is 88.5% of the summarization stage on A100 GPU. As
shown in Figure 2a, FCs and FFNs in the generation stage that
consist of matrix-vector multiplications account for 45.4%
of the total latency and are well-matched to be accelerated
by PIM. In comparison, the summarization stage shows an
even greater reliance on FCs and FFNs, which mainly employ
matrix-matrix multiplications, thereby necessitating a com-
pute accelerator (e.g., NPU) for effective acceleration. Thus,
in this work, we propose a heterogeneous accelerator that
integrates both an NPU with a PIM to address the diverse
computation requirements in LLMs.
In addition, LLMs also include vector operations such as
layer normalization and non-computing operations such
as transpose of matrix. As shown in Figure 2a, layer nor-
malization and residual addition represent 13.2% of the total
latency while representing less than 0.06% of the total FLOPs,
raising a need for a dedicated vector processing unit. Addi-
tionally, a significant portion of self-attention latency in the
decoder is attributed to non-computing operations within
the self-attention, as in Figures 2a and 2b. Among opera-
tions in self-attention that accounts for 41.4% of the total
decoder latency, non-computing operations occupy 66.1%
of the total self-attention latency. This substantial impact
of non-computing operations highlights the necessity for a
domain-specific accelerator with flexible data manipulation.
3.2 Partitioned vs. Unified Memory Systems in LLMs
Systems using commercial PIM [ 8,24,26,31] with CPU or
GPU typically employ a partitioned main memory system
where some main memory is dedicated for PIM accelerator’s
memory while the remaining memory is used by the host
(i.e., CPU or GPU). This approach can maximize parallelism
as both PIM and the host can access their own memory.
However, partitioned memory can be problematic if there
is significant sharing of data between the host and the PIM
accelerator as the same data need to be duplicated across both
memories to maximize the parallelism. Without duplicating
data, substantial data transfers between two memories are
necessary, potentially deteriorating performance.
In LLMs, the parameters of FC layers need to be shared
between the NPU and the PIM since they are utilized both in
the matrix-matrix and matrix-vector computation. Since the
FC parameters constitute a large fraction of data required for
inference (e.g., 91% in GPT-2), using a partitioned memory
in the NPU-PIM system for LLMs results in inefficient usage
of the memory. As a result, we employ a unified memoryorganization where the PIM is used as the main memory
for both the PIM accelerator and the NPU – resulting in
approximately 2×reduction in memory footprint compared
to partitioned memory system.
However, a unified memory presents new challenges, com-
pared to the partitioned memory system, as the PIM memory
is responsible for both “normal” memory accesses from the
NPU as well as the PIM computation and these two steps
cannot be executed in parallel. As naïve scheduling does not
consider memory resource conflicts between PIM compu-
tations and normal memory accesses and fails to observe
the parallelizability between PIM computations and other
computations, it cannot exploit available parallelism across
the NPU and the PIM. In this work, we propose PIM Ac-
cess Scheduling that addresses such challenges of the unified
memory system.
4 IANUS Architecture
To accelerate the end-to-end inference of transformer-based
LLMs, we introduce IANUS ( Integrated Accelerator based
onNPU-PIM Unified Memory System) that integrates NPU
and PIM (Figure 3). This section describes the IANUS archi-
tecture, including the NPU and the PIM architecture that we
leverage, details the transformer-aware microarchitecture
within NPU and PIM, as well as introduces new microarchi-
tectural components that we propose to enable IANUS with
a unified memory system architecture.
4.1 NPU & PIM Architecture
Computation Units in NPU: As in Figure 3, a single core
of NPU comprises two computing units: the matrix unit (MU)
and the vector unit (VU). The MU is built on a systolic array
[25] of 128×64 processing elements to accelerate matrix-
matrix multiplication, such as FC layers. To enable efficient
pre- or post-processing, the MU also supports output scaling
and bias addition. The VU consists of sixteen 4-wide VLIW
processors [ 11]. As it is designed to manage vector opera-
tions and general purpose operations that the MU cannot
efficiently perform, the VU supports element-wise addition,
layer normalization [ 3], masking, and non-linear activation
functions such as softmax [4] and GELU [18].
Scratch-pad Memories in NPU: The activation scratch-
pad memory (AM) and the weight scratch-pad memory (WM)
in the core of NPU supply data to the computing units. The
WM provides weights, scales, and biases to the matrix unit.
The AM serves as a data storage for both computing units,
typically providing input or activation data. The AM adopts
a transposed data addressing layout relative to the WM to
fully exploit the benefits of the matrix unit’s systolic array.
Moreover, the size of data accessed by a single address in
each scratch-pad ( entry ) is aligned with the corresponding
dimension of the matrix unit’s systolic array. Specifically,
the entry size of the AM is twice that of the WM.

## Page 5

IANUS: Integrated Accelerator based on NPU-PIM Unified Memory System ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA
NPU
PIM
MCPIM
MCNetwork -on-Chip
PIMCoreDraft v5 –24.03 –
BK 1Processing 
Unit
Cell
BK 1
Processing 
UnitCellcontrol streaming data
DMA Scratch -pad
Computing unitBank 0
Processing
UnitCellBank 2
Processing
UnitCell
Bank 1Processing
Unit
Cell
Bank 3Processing
Unit
CellPERIGlobal BufferWeight Scratch- pad
Network Interface UnitActivation Scratch- pad Vector Unit Matrix Unit
LoadLoad,
Store
Weight Scratch- pad
Matrix
Unit
LoadPIM Control 
UnitFigure 3
Command 
Scheduler
Figure 3. (Left) Architecture of a core in NPU. (Middle) PIM architecture. (Right) Overall architecture of IANUS.
PIM Architecture: PIM architecture for IANUS is based
on the commercial PIM (AiM [ 26,30]) that i) exploits true
all-bank parallelism, ii) is designed to accelerate end-to-
end matrix-vector multiplication and activation functions
in DRAM, and iii) is based on commodity DRAM (GDDR6).
Processing units (PUs) are implemented at each bank of the
PIM and a global buffer is implemented at the peripheral
circuit (Figure 3). The global buffer is shared between all PUs
and stores an input vector, often reused multiple times when
processing matrix-vector products. In comparison, large data
with low reusability such as weight matrix, often read just
once during matrix-vector product, are stored at each bank.
Each PU, associated with each bank, includes a set of multipli-
ers, an adder tree, an accumulator for Multiply-Accumulate
(MAC) operation, and an activation function unit.
4.2 Transformer-Aware NPU & PIM
Microarchitecture
In this subsection, we highlight the NPU microarchitecture
designed to accelerate self-attention and vector operations
in transformer-based LLMs, along with the data allocation
scheme in PIM aimed at optimizing FC operations.
4.2.1 Data Manipulation in Self-Attention.
Key Transpose: The transpose operation requires data trans-
fer between on-chip and off-chip memory without dedicated
hardware, potentially delaying PIM operations that also uti-
lize off-chip memory. We avoid off-chip access by executing
transpose within on-chip through incorporating a stream-
ing path between DMAs (light blue boxes in Figure 3) of
two scratch-pads. However, moving data from the activation
scratch-pad (AM) to the weight scratch-pad (WM) through
on-chip DMA only performs a partial transpose operation
because of the mismatch of the entry sizes for the two scratch-
pads. Thus, we introduce a streaming buffer between the
two scratch-pads for on-chip data movement during on-chip
DMA. We then implement weight interleaving within the
matrix unit, enabling access to the WM entry with a specific
stride.
Splitting / Merging Attention Heads: Splitting and
merging attention heads represent a large fraction of the
self-attention latency in a GPU due to the data reordering.
Channel 0’s tile
Tile 0 Tile 1
Tile 2 Tile 3
…Bank 0
PUBank 15
PU…# bank (16) x # channel (8)
1
32
1024 BF16 = 2KBBank
0
12
…
15PIM (Channel 0) Weight matrix
Tile 0 Tile 1
Tile 2 Tile 3
…# bank (16) x # channel (8)1
32
1024 BF16 = 2KB
Weight matrix Channel 0’s tileBank 0
PUBank 15
PU…
PIM (Channel 0)Bank
012
…
15Figure 4
v1
v2
Tile 0 Tile 1
Tile 2 Tile 3
…# bank (16) x # channel (8)1
32
1024 BF16 = 2KBBank 0
PUBank 15
PU…Bank
0
1
2
…
15v2Figure 4. Data allocation and tiling scheme for a matrix-
vector multiplication in PIM.
Our compiler avoids such data reordering by carefully defin-
ing and generating activation scratch-pad addresses of input
and output data in the command. For instance, when gener-
ating commands for the FC operation that produces 𝑄, the
compiler generates as many commands as the number of
heads. The compiler then assigns a distinct output address
for each command, guiding the matrix unit to store 𝑄in
the scratch-pad in a split manner. Hence, no data reordering
overhead is required. Similarly, the compiler ensures con-
secutive output addresses of each head’s 𝑆𝑉command for
merging attention heads.
4.2.2 Vector Operations in Vector Unit.
Layer Normalization: Given the limited amount of on-chip
memory within the vector unit (VU), a two-phase approach
is used where VU calculates the mean and variance of the
tokens in the first phase while the normalization is done in
the second phase.
Masked Softmax: We combine masking and softmax [ 4]
within a single kernel. Each mask is stored as a 1-bit bitmap,
reducing data movement and memory usage. In softmax, we
subtract the max value for stability instead of the large value.
GELU: For the GELU activation [ 18], VU uses a lookup
table (LUT) approximation, widely employed due to its ac-
curacy and performance [ 19,50]. GELU activation is also
supported in PIM by reserving some DRAM rows inside PIM
as LUT for the activation function and linearly interpolating
data from the LUT within the processing unit of PIM.
4.2.3 Data Allocation in PIM.
PIM is exploited for matrix-vector multiplication during the
FC layers in the generation stages. We exploit data allocation

## Page 6

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA Seo et al.
and tiling scheme that maximize the performance of FC
layers in PIM and demonstrate these strategies with the
weight matrix of an FC layer in Figure 4. The weight matrix
is divided into tiles with each tile consisting of 16 (number of
banks per channel) ×8 (number of channels for IANUS) rows
and up to 1024 columns (number of elements in one DRAM
row). Each row in the tile is allocated to the same DRAM
row address across each bank and each channel to maximize
performance as PIM can perform computation across all
banks and all channels in parallel. While the optimal tiling
can vary across workloads, we assume row-major tiling.
4.3 IANUS Microarchitecture
Command Scheduler: The command scheduler is respon-
sible for checking dependencies between each command
and the status of each compute, DMA, and PIM unit and
sending commands to each unit. When a command has no
dependency and the corresponding unit is in an idle state,
the scheduler pushes the command into the “issue” queue
of the unit, and the unit executes it. If the command cannot
be issued, the command is pushed into the “pending” queue.
Upon completion of execution, the scheduler resolves the de-
pendencies between the command and the other commands.
PIM Control Unit and PIM Memory Controller: Or-
chestrating multiple PIM chips is not trivial as it requires
scheduling a large number of PIM commands and increases
the complexity of the command scheduler. More importantly,
the efficiency of PIM computation diminishes if a standard
memory command is inserted in the middle of multiple PIM
commands for a single “operation”, such as a matrix-vector
multiplication. Thus, we propose macro PIM command for
scheduling. One macro PIM command, which represents a
single operation, comprises multiple micro PIM commands
(e.g., a single matrix-vector operation is executed through
a macro PIM command that consists of multiple micro PIM
commands, including providing the input vector, performing
the MAC operation, etc.). To support macro PIM commands,
a PIM control unit (PCU) and PIM memory controller (PIM
MC) are added, as shown in Figure 3.
When one macro PIM command reaches “ready” state, the
command scheduler forwards the macro PIM command to
the PCU. At the same time, the scheduler puts other DMA
commands related to the off-chip memory into “wait” state
to ensure the PIM execution is not interrupted. Once the
PCU receives the macro PIM command, PCU decodes it into
multiple micro PIM commands and forwards these to the
PIM MC through the network-on-chip (NoC).
The PIM MC supports both PIM commands and normal
memory commands. Similar to conventional memory con-
trollers, PIM MC tracks the state of each memory bank and
generates appropriate commands following pre-defined tim-
ing constraints as well as newly introduced states and timing
Row Channel Bank Column Offset
Row index in a tile Column index in a tile Tile indexIANUS’s
address mappingFigure 5. IANUS’s DRAM address mapping with the map-
ping of tile shown in Figure 4.
constraints of PIM operations. When all micro PIM com-
mands within one macro PIM command finish, the comple-
tion signal is forwarded to the command scheduler to enable
DMA commands associated with the off-chip memory.
Network-on-chip: The NoC topology in IANUS provides
all-to-all connectivity between all of the cores and the PIM
MCs. The NoC traffic for IANUS consists of both the memory
traffic as well as the PIM traffic to support the unified mem-
ory system. All-to-all connectivity ensures that each core
can access any memory channel when PIM is used as the
main memory of the NPU. In addition, the NoC is also used
for PIM commands from the PCU to the PIM MCs. The NoC
supports broadcasting of PIM commands to all PIM MCs to
reduce NoC bandwidth demand while providing parallel PIM
operations across all PIM channels.
DRAM Address Mapping: The DRAM address mapping
of IANUS is shown in Figure 5. IANUS employs an address
mapping of (MSB) Row-Channel-Bank-Column (LSB) and
the main goal of the IANUS’s mapping is to maximize PIM
computation performance through PIM-aware tile (Figure 4)
placement. By using the row address bit as the MSB and
using those bits as the index of a tile, data within a single tile
share the same row address that ensures row conflicts do not
occur during the compute operations related to a single tile.
In addition, each tile is assigned to a different row address.
The column address bit is used as the LSB to ensure that
operations on all elements of a single row within a tile are
handled by one processing unit to execute MAC within one
bank. Placing channel and bank address bits between the row
and column address bits allows each row within a tile to be
distributed across different channels and banks. This enables
the PIM to concurrently compute all rows within a tile by
leveraging channel and bank parallelism and maximize PIM
computation throughput.
5 PIM Access Scheduling
The integrated NPU-PIM architecture with a unified main
memory presents challenges as the main memory is used by
both the NPU and the PIM compute logic. In this section, we
propose PIM Access Scheduling (PAS) that enables efficient
sharing of the physical memory between NPU and PIM. Un-
like traditional memory access scheduling [ 40] that involves
scheduling of memory commands, PAS not only needs to
consider scheduling normal DRAM commands and PIM com-
mands but also needs to address the challenges of workload
mapping across the NPU and the PIM. More importantly,
the scheduling or mapping of the workloads impacts how

## Page 7

IANUS: Integrated Accelerator based on NPU-PIM Unified Memory System ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA
Layer
NormSelf-
Attention Sync Add Sync
LayerNorm Self-
Attention Sync Add SyncCore 0
Core 1…
…Divide head- wise𝑾𝑾𝟎𝟎𝑸𝑸𝑾𝑾𝟎𝟎𝑲𝑲𝑾𝑾𝟎𝟎𝑽𝑽
𝑾𝑾𝟏𝟏𝑸𝑸𝑾𝑾𝟏𝟏𝑲𝑲𝑾𝑾𝟏𝟏𝑽𝑽PIM (Chip 0)
PIM (Chip 1)
𝑾𝑾𝑶𝑶
Divide column- wise
𝑾𝑾𝟏𝟏𝑶𝑶𝑾𝑾𝟎𝟎𝑶𝑶
FC for Q,K,VFC for Q,K,V
FC
for AttentionFC
for AttentionVector Unit Matrix Unit PIM-Processing Unit
Figure 6. Workload mapping and execution flow, featuring
intra-layer parallelism and attention head parallelism. For
simplicity, only one attention head per core is shown. The
mapping of operations in self-attention is detailed in Sec-
tion 5.3.
Algorithm 1 Adaptive mapping algorithm for FC layers.
Input/Output: CMDs (a sequence of commands)
Params: n(number of input tokens), T(the size of MU)
Define:𝑉𝑈,𝑀𝑈,𝑃𝐼𝑀,𝐷𝑀𝐴 (analytical model of units)
1:for𝑖,𝑐𝑚𝑑 in𝐶𝑀𝐷𝑠 do
2: if𝑐𝑚𝑑.𝑡𝑦𝑝𝑒 ==𝑀𝑈𝐹𝐶then
3:𝑝𝑟𝑒𝑣_𝑐𝑚𝑑←𝐶𝑀𝐷𝑠[𝑖−1]
4: // Check prefetching
5: if𝑝𝑟𝑒𝑣_𝑐𝑚𝑑.𝑡𝑦𝑝𝑒 ==𝑉𝑈then
6: 𝑡𝑝𝑟𝑒𝑓𝑒𝑡𝑐ℎ←𝑉𝑈(𝑛,𝑝𝑟𝑒𝑣 _𝑐𝑚𝑑.𝑑𝑖𝑚)
7: // Consider tiling and pipelining for MU
8:𝑤𝑐𝑓𝑔←𝑐𝑚𝑑.𝑤𝑒𝑖𝑔ℎ𝑡 _𝑐𝑓𝑔
9:𝑤𝑙𝑜𝑎𝑑←𝐷𝑀𝐴𝑤𝑒𝑖𝑔ℎ𝑡(𝑤𝑐𝑓𝑔)
10:𝑚𝑢𝐹𝐶←𝑀𝑈𝐹𝐶(𝑛,𝑤𝑐𝑓𝑔)
11:𝑚𝑢𝑡𝑖𝑚𝑒←𝑝𝑖𝑝𝑒((𝑤𝑙𝑜𝑎𝑑,𝑚𝑢𝐹𝐶),𝑇)−𝑡𝑝𝑟𝑒𝑓𝑒𝑡𝑐ℎ
12: // Calculate PIM time
13:𝑝𝑖𝑚𝑡𝑖𝑚𝑒←𝑛×𝑃𝐼𝑀(𝑤𝑐𝑓𝑔)
14: if𝑝𝑖𝑚𝑡𝑖𝑚𝑒<𝑚𝑢𝑡𝑖𝑚𝑒then
15: Replace𝐶𝑀𝐷𝑠[𝑖].𝑡𝑦𝑝𝑒 with𝑃𝐼𝑀
the DRAM/PIM commands are scheduled. In this section, we
describe PAS within the context of IANUS, particularly on
FC operations and multi-head attention in LLMs, and how
they are mapped/scheduled on IANUS.
5.1 Overview
We present the execution flow and workload mapping of
LLMs for IANUS in Figure 6. To leverage parallelism across
all cores in the NPU as well as across all PIM chips, we exploit
attention head parallelism by partitioning the weights of the
FC for𝑄,𝐾, and𝑉across PIM chips in a head-wise scheme.
Through the head-wise partitioning, each core can access
the memory in parallel to load the weights or the output of
PIM compute for the multi-head attention.
For other FC operations, we leverage intra-layer paral-
lelism to minimize data movement of weights that are consid-
erably larger than input or activation data in LLMs. To reduce
synchronization overhead between each core in the NPU, wepartition the weights of FC column-wise. Synchronization
occurs four times: once after multi-head attention, twice af-
ter each residual addition, and once after GELU. Meanwhile,
layer normalization and residual addition are mapped to the
vector unit (VU) within the NPU (Figure 6).
5.2 FC Operation
FC can be mapped to either the matrix unit (MU) or the PIM.
The summarization stage often has a large input token size
and results in high computation requirements – thus, it is
more appropriate to map the FC to the matrix unit. When
the input token size is small, loading the weights from the
memory can become the bottleneck because the arithmetic
intensity of the FC operation decreases. Thus, an adaptive
mapping algorithm is necessary to determine whether to
map the FC to the PIM or the MU within the NPU.
An overview of the adaptive mapping algorithm is sum-
marized in Algorithm 1. To determine the appropriate unit
for FC, we develop a simple analytical model that estimates
the execution time across different execution units (e.g., MU,
VU, DMA, PIM) based on the number of input tokens at
compile time. The input for the adaptive mapping algorithm
is a sequence of commands based on mapping of the FC to
the matrix unit. When estimating the time of FC on MU,
we assume a pipelined scheme for both weight loading and
computation, as well as tiling configured to match the MU’s
size (lines 8-11). We also account for weight prefetching time
if an operation of VU precedes the FC operation (lines 5-6).
We then compare the estimated time of FC on MU with that
of PIM and assign the FC to the execution unit that can com-
plete sooner (lines 13-15). If the first FC of FFN is mapped
to the PIM, the GELU will also be allocated to the PIM since
the PIM is designed to support GELU right after FC.
5.3 Multi-Head Attention
As described earlier in Figure 1b, multi-head attention con-
sists of a series of operations that have various computational
requirements. While IANUS provides computing capability
of both NPU and PIM, naïve scheduling that overlooks par-
allelizability and resource conflicts between operations may
lead to the under-utilization of both units with considerable
latency overhead. To address this challenge, we propose uni-
fied memory-aware scheduling for multi-head attention at
both the summarization andgeneration stages.
Summarization stage: As shown in Figure 7a, FC layers
for𝑄,𝐾, and𝑉typically operate as matrix-matrix multiplica-
tions with multiple input tokens ( 𝑥) – thus are computed in
the matrix unit, while weight matrices ( 𝑊𝑄,𝐾,𝑉) are loaded
from the memory via DMA. To efficiently process multi-head
attention, we utilize both intra-attention head parallelism
and inter-attention head pipelining. We prioritize key gen-
eration to execute key transpose in parallel with value gen-
eration. As DMAs are utilized for on-chip transpose, they

## Page 8

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA Seo et al.
Figure 7
Memory 𝑾𝑾𝒊𝒊𝑽𝑽𝑾𝑾𝒊𝒊𝑸𝑸𝑾𝑾𝒊𝒊𝑲𝑲𝑾𝑾𝒊𝒊+𝟏𝟏𝑲𝑲𝑾𝑾𝒊𝒊+𝟏𝟏𝑸𝑸𝑾𝑾𝒊𝒊𝑸𝑸𝒙𝒙 𝑾𝑾𝒊𝒊𝑲𝑲𝒙𝒙 𝑾𝑾𝒊𝒊𝑽𝑽𝒙𝒙𝑸𝑸𝒊𝒊𝑲𝑲𝒊𝒊𝑻𝑻 𝑺𝑺𝒊𝒊𝑽𝑽𝒊𝒊𝑾𝑾𝒊𝒊+𝟏𝟏𝑲𝑲𝒙𝒙
𝑲𝑲𝒊𝒊Matrix Unit
Vector Unit
𝑽𝑽𝒊𝒊Load Store PIM Intra -head parallel Inter -head pipeline
Softmax
𝑾𝑾𝑽𝑽,𝒊𝒊𝒙𝒙 𝑸𝑸𝒊𝒊𝑲𝑲𝒊𝒊𝑻𝑻 𝑺𝑺𝒊𝒊𝑽𝑽𝒊𝒊 𝑽𝑽𝒊𝒊Matrix Unit
Vector Unit
Head i+1Memory 𝑾𝑾𝒊𝒊𝑲𝑲𝒙𝒙 𝑾𝑾𝑲𝑲 ,𝒊𝒊+𝟏𝟏𝒙𝒙𝑲𝑲
scaleSoftmax
𝑾𝑾𝒊𝒊𝑸𝑸𝒙𝒙𝑾𝑾𝒊𝒊𝑽𝑽𝒙𝒙Matrix Unit
Vector Unit
Head iMemory 𝑾𝑾𝒊𝒊𝑲𝑲𝒙𝒙 𝑾𝑾𝒊𝒊+𝟏𝟏𝑲𝑲𝒙𝒙𝑲𝑲cat𝑸𝑸𝒊𝒊𝑲𝑲𝒊𝒊𝑻𝑻
𝑽𝑽𝒊𝒊,𝒄𝒄𝒄𝒄𝒄𝒄𝑺𝑺𝒊𝒊𝑽𝑽𝒊𝒊
𝑾𝑾𝒊𝒊+𝟏𝟏𝑸𝑸𝒙𝒙𝑲𝑲cat
𝑲𝑲𝒑𝒑𝒑𝒑𝒑𝒑1 4
𝑲𝑲 ,𝑽𝑽𝒊𝒊𝑲𝑲𝒊𝒊
124
Head i𝑾𝑾𝒊𝒊𝑸𝑸𝒙𝒙
SoftmaxInter-headHead i+1Head iLoad (DMA) Store (DMA) PIM op Intra -head parallel
TimeInter -head pipeline
1 3
2 3
Head i+1𝑾𝑾𝒊𝒊𝑸𝑸 𝑾𝑾𝒊𝒊𝑸𝑸 𝑾𝑾𝒊𝒊𝑸𝑸Time
Time
(a)
Figure 7
Memory 𝑾𝑾𝒊𝒊𝑽𝑽𝑾𝑾𝒊𝒊𝑸𝑸𝑾𝑾𝒊𝒊𝑲𝑲𝑾𝑾𝒊𝒊+𝟏𝟏𝑲𝑲𝑾𝑾𝒊𝒊+𝟏𝟏𝑸𝑸𝑾𝑾𝒊𝒊𝑸𝑸𝒙𝒙 𝑾𝑾𝒊𝒊𝑲𝑲𝒙𝒙 𝑾𝑾𝒊𝒊𝑽𝑽𝒙𝒙𝑸𝑸𝒊𝒊𝑲𝑲𝒊𝒊𝑻𝑻 𝑺𝑺𝒊𝒊𝑽𝑽𝒊𝒊𝑾𝑾𝒊𝒊+𝟏𝟏𝑲𝑲𝒙𝒙
𝑲𝑲𝒊𝒊Matrix Unit
Vector Unit
𝑽𝑽𝒊𝒊Load Store PIM Intra -head parallel Inter -head pipeline
Softmax
𝑾𝑾𝑽𝑽,𝒊𝒊𝒙𝒙 𝑸𝑸𝒊𝒊𝑲𝑲𝒊𝒊𝑻𝑻 𝑺𝑺𝒊𝒊𝑽𝑽𝒊𝒊 𝑽𝑽𝒊𝒊Matrix Unit
Vector Unit
Head i+1Memory 𝑾𝑾𝒊𝒊𝑲𝑲𝒙𝒙 𝑾𝑾𝑲𝑲 ,𝒊𝒊+𝟏𝟏𝒙𝒙𝑲𝑲
scaleSoftmax
𝑾𝑾𝒊𝒊𝑸𝑸𝒙𝒙𝑾𝑾𝒊𝒊𝑽𝑽𝒙𝒙Matrix Unit
Vector Unit
Head iMemory 𝑾𝑾𝒊𝒊𝑲𝑲𝒙𝒙 𝑾𝑾𝒊𝒊+𝟏𝟏𝑲𝑲𝒙𝒙𝑲𝑲cat𝑸𝑸𝒊𝒊𝑲𝑲𝒊𝒊𝑻𝑻
𝑽𝑽𝒊𝒊,𝒄𝒄𝒄𝒄𝒄𝒄𝑺𝑺𝒊𝒊𝑽𝑽𝒊𝒊
𝑾𝑾𝒊𝒊+𝟏𝟏𝑸𝑸𝒙𝒙𝑲𝑲cat
𝑲𝑲𝒑𝒑𝒑𝒑𝒑𝒑1 4
𝑲𝑲 ,𝑽𝑽𝒊𝒊𝑲𝑲𝒊𝒊
124
Head i𝑾𝑾𝒊𝒊𝑸𝑸𝒙𝒙
SoftmaxInter-headHead i+1Head iLoad (DMA) Store (DMA) PIM op Intra -head parallel
TimeInter -head pipeline
1 3
2 3
Head i+1𝑾𝑾𝒊𝒊𝑸𝑸 𝑾𝑾𝒊𝒊𝑸𝑸 𝑾𝑾𝒊𝒊𝑸𝑸Time
Time
(b)
Figure 7
Memory 𝑾𝑾𝒊𝒊𝑽𝑽𝑾𝑾𝒊𝒊𝑸𝑸𝑾𝑾𝒊𝒊𝑲𝑲𝑾𝑾𝒊𝒊+𝟏𝟏𝑲𝑲𝑾𝑾𝒊𝒊+𝟏𝟏𝑸𝑸𝑾𝑾𝒊𝒊𝑸𝑸𝒙𝒙 𝑾𝑾𝒊𝒊𝑲𝑲𝒙𝒙 𝑾𝑾𝒊𝒊𝑽𝑽𝒙𝒙𝑸𝑸𝒊𝒊𝑲𝑲𝒊𝒊𝑻𝑻 𝑺𝑺𝒊𝒊𝑽𝑽𝒊𝒊𝑾𝑾𝒊𝒊+𝟏𝟏𝑲𝑲𝒙𝒙
𝑲𝑲𝒊𝒊Matrix Unit
Vector Unit
𝑽𝑽𝒊𝒊Load Store PIM Intra -head parallel Inter -head pipeline
Softmax
𝑾𝑾𝑽𝑽,𝒊𝒊𝒙𝒙 𝑸𝑸𝒊𝒊𝑲𝑲𝒊𝒊𝑻𝑻 𝑺𝑺𝒊𝒊𝑽𝑽𝒊𝒊 𝑽𝑽𝒊𝒊Matrix Unit
Vector Unit
Head i+1Memory 𝑾𝑾𝒊𝒊𝑲𝑲𝒙𝒙 𝑾𝑾𝑲𝑲 ,𝒊𝒊+𝟏𝟏𝒙𝒙𝑲𝑲
scaleSoftmax
𝑾𝑾𝒊𝒊𝑸𝑸𝒙𝒙𝑾𝑾𝒊𝒊𝑽𝑽𝒙𝒙Matrix Unit
Vector Unit
Head iMemory 𝑾𝑾𝒊𝒊𝑲𝑲𝒙𝒙 𝑾𝑾𝒊𝒊+𝟏𝟏𝑲𝑲𝒙𝒙𝑲𝑲cat𝑸𝑸𝒊𝒊𝑲𝑲𝒊𝒊𝑻𝑻
𝑽𝑽𝒊𝒊,𝒄𝒄𝒄𝒄𝒄𝒄𝑺𝑺𝒊𝒊𝑽𝑽𝒊𝒊
𝑾𝑾𝒊𝒊+𝟏𝟏𝑸𝑸𝒙𝒙𝑲𝑲cat
𝑲𝑲𝒑𝒑𝒑𝒑𝒑𝒑1 4
𝑲𝑲 ,𝑽𝑽𝒊𝒊𝑲𝑲𝒊𝒊
124
Head i𝑾𝑾𝒊𝒊𝑸𝑸𝒙𝒙
SoftmaxInter-headHead i+1Head iLoad (DMA) Store (DMA) PIM op Intra -head parallel
TimeInter -head pipeline
1 3
2 3
Head i+1𝑾𝑾𝒊𝒊𝑸𝑸 𝑾𝑾𝒊𝒊𝑸𝑸 𝑾𝑾𝒊𝒊𝑸𝑸Time
Time
(c)
Figure 7. Unified memory-aware scheduling for multi-head
attention at (a) summarization stage where FCs are mapped
to the matrix unit and generation stage where FCs are mapped
to the PIM: 𝑄𝐾𝑇and𝑆𝑉mapping to (b) PIM or (c) matrix
unit. Figures (b) and (c) are drwan on the same time scale to
show the latency difference.
are not used for PIM access during transpose ( 1). Given that
the matrix unit supports output scaling (Section 4.1), the
key scaling operation is omitted. We also ensure that key
and value are stored during computations ( 2). To hasten the
start of the 𝑆𝑉operation, values are moved to the weight
scratch-pad via on-chip data transfer during the softmax
(3). In addition, we utilize inter-attention head pipelining
by prefetching the weight of the next head ( 4).
Generation stage: FC layers mainly perform matrix-vector
multiplications with one input token ( 𝑥), making them well-
suited for PIM computation. Similarly, since 𝑄𝐾𝑇and𝑆𝑉
operations involve matrix-vector multiplications and require
loading previously generated keys and values, their execu-
tions can appear to be more suitable for PIM. As shown in
Figure 7b, mapping 𝑄𝐾𝑇and𝑆𝑉to PIM avoids such load op-
erations. However, the overall performance benefit is limited
since parallelism across both the PIM and the NPU cannot be
exploited well as the PIM performs most of the operation. In
addition, computing 𝑄𝐾𝑇and𝑆𝑉in PIM results in poor effi-
ciency because of the mismatch between the PIM DRAM row
size and the data size. For example, with a head dimension of
64, PIM computational efficiency of 𝑄𝐾𝑇is only 6.25% due
to only 64 BF16 elements being utilized for computation out
of the 1024 elements available in one DRAM row.
As a result, we propose mapping 𝑄𝐾𝑇and𝑆𝑉operations
to the matrix unit, and accordingly, scheduling based on
this mapping. To exploit inter-attention head parallelism, asTable 1. Simulation parameters for IANUS.
NPUComposition 4 cores, 8 PIM memory controllers
Host interface PCIe 5.0×16
Frequency 700 MHz
CoreMatrix unit 128x64 processing elements (PEs), 4 MACs per PE, 46 TFLOPS
Vector unit Sixteen 4-wide VLIW processors
Scheduler4 command slots per issue queue of units,
256 command slots in pending queue
Scratch-pad Activation 12 MB, Weight 4 MB
PIMMemory
configurationGDDR6 16 Gb/s;×16 organization; 8 channels; 256 GB/s;
2 channels per chip, 16 banks per channel, row (page) size 2 KB
Timing parameters𝑡𝐶𝐾=0.5𝑛𝑠,𝑡𝐶𝐶𝐷 𝑆=𝑡𝐶𝐶𝐷 𝐿=1𝑛𝑠,𝑡𝑅𝐴𝑆=21𝑛𝑠,
𝑡𝑊𝑅=36𝑛𝑠,𝑡𝑅𝑃=30𝑛𝑠,𝑡𝑅𝐶𝐷𝑅𝐷 =36𝑛𝑠,𝑡𝑅𝐶𝐷𝑊𝑅 =24𝑛𝑠
Processing unit (PU) 1 GHz; 1 PU per bank; 32 GFLOPS per PU
Global buffer One 2 KB global buffer per channel
Table 2. Specifications of A100 GPU, DFX [ 19], and IANUS.
A100 [37] DFX [19] IANUS
ComputeFrequency 1155 MHz 200 MHz 700 MHz
Throughput 255 TFLOPS 1.64 TFLOPS 184 TFLOPS
On-chip
MemoryCapacity RF, L1, L2: 84 MB ∼40 MBActivation Scratch-pad: 48 MB
Weight Scratch-pad: 16 MB
Off-chip
MemoryType HBM2e HBM2 GDDR6
Capacity 80 GB 32 GB 8 GB
Bandwidth 2039 GB/s 1840 GB/s 256 GB/s
Internal BW N/A N/A 4096 GB/s
shown in Figure 7c, we execute key concatenation in the
vector unit instead of storing the key ( 1), enabling its simul-
taneous execution with query generation in PIM. Loading
the previously generated keys ( 𝐾𝑝𝑟𝑒) of𝑖th head is omitted in
Figure 7c, as its small size compared to the FC weight allows
for prefetching. We then transpose concatenated keys within
on-chip while performing query generation in PIM. Further-
more, we execute 𝑄𝐾𝑇and softmax respectively in parallel
with value generation by mapping 𝑄𝐾𝑇to matrix unit ( 2).
After value generation, storing generated keys and values
and loading concatenated values ( 𝑉𝑐𝑎𝑡) are performed during
softmax ( 3). We also employ inter-attention head pipelining
by prefetching 𝐾𝑝𝑟𝑒of the next head during 𝑆𝑉(4). If the
prefetching ends before the completion of 𝑆𝑉, the key gen-
eration of the next head is performed in conjunction with
𝑆𝑉. Consequently, our scheduling enhances performance by
maximizing both intra-parallelism and inter-pipelining of
attention head.
6 Evaluations
6.1 Methodology
To evaluate the performance of IANUS, we developed a
cycle-accurate in-house simulator to model IANUS. The sim-
ulator integrates an NPU simulator based on a commercial
NPU [ 1,20,41] as well as a PIM simulator modeled after
the real PIM chip, AiM [ 26,30]. Both the NPU and the PIM
simulator are validated against their respective real hard-
ware counterparts within a 5% error margin. An overview of
the key simulation parameters is summarized in Table 1. In
addition, we modeled the new components added to enable
IANUS, including the PIM control unit (PCU), and modified
the memory controller to support both PIM commands and

## Page 9

IANUS: Integrated Accelerator based on NPU-PIM Unified Memory System ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA
15 111 870 6,938 
15 111 872 7,130 
15 112 879 7,221 
2,024 
22 164 1,271 10,274 
23 164 1,299 10,291 
23 168 1,299 10,401 
2,950 
29 212 1,698 13,622 
29 220 1,740 13,701 
31 221 1,801 14,239 
3,962 
32 242 1,916 15,411 
33 245 1,928 15,436 
39 248 2,009 15,480 
4,418 
5 12 68 576 
6 13 74 609 
9 17 84 673 
179 
10 25 151 1,261 
13 29 161 1,323 
18 36 182 1,447 
388 
18 43 251 2,073 
22 49 267 2,171 
31 60 299 2,367 
637 
32 71 388 3,261 
38 79 418 3,462 
50 97 478 3,864 
1,020 
1.E+01.E+31.E+6
(128,1)
(128,8)
(128,64)
(128,512)
(256,1)
(256,8)
(256,64)
(256,512)
(512,1)
(512,8)
(512,64)
(512,512)
Avg
(128,1)
(128,8)
(128,64)
(128,512)
(256,1)(256,8)
(256,64)
(256,512)
(512,1)
(512,8)
(512,64)
(512,512)
Avg
(128,1)
(128,8)
(128,64)
(128,512)
(256,1)
(256,8)
(256,64)
(256,512)
(512,1)
(512,8)
(512,64)
(512,512)
Avg
(128,1)
(128,8)
(128,64)
(128,512)
(256,1)
(256,8)
(256,64)
(256,512)
(512,1)
(512,8)
(512,64)
(512,512)
Avg
(Input size, Output size)
GPT-2 M(Input size, Output size)
GPT-2 L(Input size, Output size)
GPT-2 XL(Input size, Output size)
GPT-2 2.5BLATENCY (MS)GPU IANUS
11.3x 7.6x4.3x 6.2x(ms)
Figure 8. Inference latency of various GPT-2 models on A100 GPU and IANUS.
Table 3. Network configuration details.
NameEmbedding
dimensionHead
dimension# Heads # Blocks # Params Workload
BERTB 768 64 12 12 110MQuestion-
answering
(QA)L 1024 64 16 24 340M
1.3B 2048 64 32 24 1.3B
3.9B 2560 64 40 48 3.9B
GPTM 1024 64 16 24 345MLanguage
modeling
(LM)L 1280 64 20 36 762M
XL 1536 64 24 48 1.5B
2.5B 1920 96 20 54 2.5B
normal memory commands. To avoid latency overhead from
the PCU, we designed its operations to be pipelined with
PIM computations. Our simulator also provides statistics
on energy consumption. It measures the dynamic energy
consumed by cores in NPU, PIM operations, and standard
DRAM operations. Based on prior analysis [ 26], we assume
that the power consumption of PIM computing operations
is 3×of that for DRAM read operations.
We compare the performance of IANUS against a GPU,
state-of-the-art prior work (DFX [ 19]), as well as the NPU
without PIM memory. For the GPU, we use an NVIDIA A100-
SXM GPU [ 37] with Pytorch 2.0 and CUDA Toolkit 11.8 and
GPU-optimized source codes from Huggingface [ 47] and
Megatron-LM [ 43]. The latency of models is measured using
thetorch.cuda.Event API. DFX [ 19] is a multi-FPGA appli-
ance specifically designed to accelerate the generation stage
of GPT models. We assume a DFX with 4 FPGAs that can
support GPT-2 XL model. We also compare IANUS with a
commercial NPU [1, 20, 41] (the same NPU used in IANUS)
without PIM, but with standard GDDR6 memory ( NPU-MEM ).
It shares identical specifications with IANUS in Table 2
except for the internal memory bandwidth and features a
peak throughput of 184 TFLOPS. IANUS is identical to NPU-
MEM, except that standard GDDR6 memory is replaced with
PIM based on AiM [ 26,30]. Each PIM chip achieves a peak
throughput of 1 TFLOPS with 32 processing units utilizing
1024 GB/s internal memory bandwidth. The specifications
of each architecture are summarized in Table 2.We evaluate two notable transformer-based LLMs, BERT
[9] and GPT [ 39] with the BF16 [ 46] data type, which main-
tains the accuracy of the full-precision model. The configu-
rations and tasks of each model are presented in Table 3. We
exploit a GPT-2 XL model with its attention heads reduced
from 25 to 24, whose accuracy was validated in [ 19], to op-
timize parallelism. We assess the end-to-end performance
of models with input sizes of 128, 256, and 512 tokens. For
the GPT-2, we use output sizes of 1, 8, 64, and 512 tokens.
These sizes represent the typical user request ranges for NLP
services in datacenters [ 38]. Due to the time overhead as-
sociated with gathering inputs from multiple users, current
datacenters prefer running the model with non-batched in-
put [ 12,19]; therefore, we evaluate our work using a batch
size of 1.
6.2 Performance Results
End-to-end Inference Latency: Figure 8 presents the end-
to-end latency of GPT-2 models on the GPU and IANUS. The
result shows that IANUS achieves a 4.3 ×speedup compared
to the GPU for the 2.5B model, on average. For the workload
with significantly more output tokens than input tokens, i.e.,
(128,512), IANUS demonstrates 12.0 ×, 8.1×, and 6.6×lower
latency than the GPU for the GPT-2 M, L, and XL models,
respectively. These substantial speedups are obtained from
the high utilization of PIM chips’ internal bandwidth of 4096
GB/s for matrix-vector multiplication in the generation stage.
On average, IANUS takes about 5.7 ms per token for gen-
eration stages of the GPT-2 2.5B model with configuration
(128,64), while the GPU takes about 29.9 ms.
In Figure 9, we conduct a comparison of the GPT-2 XL’s la-
tency among IANUS, NPU-MEM, and DFX with four FPGAs
[19], which provides state-of-the-art performance for GPT-2.
Input and output token sizes for the comparison are ob-
tained from [ 19]. IANUS achieves a 49.3 ×speedup compared
to DFX for the (128,1) configuration. IANUS and NPU-MEM
present similar performance for this configuration, as the
PIM in IANUS operates as a standard GDDR6 except for
the LM head. For the generation stage, DFX achieves 6.9 ms

## Page 10

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA Seo et al.
227 330 1,981 447 550 2,201 887 991 2,642 
18 247 3,970 
18 246 3,972 
18 249 3,983 
18 73 989 
18 72 990 
18 73 997 
1.E+01.E+21.E+41.E+6
(32,1) (32,16) (32,256) (64,1) (64,16) (64,256) (128,1) (128,16) (128,256)
(Input size, Output size)
GPT-2 XLLATENCY ( MS) DFX NPU-MEM IANUS(ms)
Figure 9. Inference latency of GPT-2 XL on DFX [ 19], NPU-
MEM, and IANUS.
109 333 141 539 
54 177 88 322 
269 1275 467 2358 
65 293 127 568 
0 2000 4000IANUSNPU-MEMIANUSNPU-MEMGPT-2 L
(128,256)GPT-2 XL
(128,256)
LATENCY ADDLayerNorm
Self-attention
FC for Attention + Add
FFN+Add
FC for Q,K,V
(ms)
Figure 10. Latency breakdown of GPT-2 XL and L’s genera-
tionstages for NPU-MEM and IANUS.
3.7 1.0 7.7 
2.1 13.9 
3.6 25.1 
5.8 
01530
NPU-
MEMIANUS NPU-
MEMIANUS NPU-
MEMIANUS NPU-
MEMIANUS
GPT-2 M GPT-2 L GPT-2 XL GPT-2 2.5BNormalized
Dynamic EnergyGDDR6: Normal op GDDR6: PIM op NPU's cores
Figure 11. Dynamic energy of NPU-MEM and IANUS, nor-
malized to IANUS with GPT-2 M.
to generate one token for the (64,256) configuration, while
IANUS generates a token in 3.8 ms for the same configura-
tion, achieving a speedup of 1.8 ×. Without the benefits of
PIM, NPU-MEM takes 15.5 ms. To this end, IANUS achieves
an average speedup of 3.2 ×compared to DFX, while NPU-
MEM results in 24% slowdown.
Latency Breakdown: To investigate the impact of using
PIM, we measure the latency of operations in the decoder for
NPU-MEM and IANUS in the generation stages of GPT-2 L
and XL. As residual additions are executed with adjacent FC
and FFN using a pipelining scheme, we collectively measure
their latency. As shown in Figure 10, IANUS reduces the exe-
cution time of two FCs in multi-head attention from 890 ms to
215 ms for the GPT-2 XL model, achieving a speedup of 4.1 ×.
Since the FFN has a four times larger weight size compared to
these two FCs, it achieves a higher speedup of 5.1 ×. IANUS
also achieves a speedup of 4.3 ×for self-attention without
offloading any operation in self-attention. This speedup is
obtained from prefetching previously generated keys and
values instead of the weight for generating 𝑄,𝐾, and𝑉by
offloading FC for 𝑄,𝐾, and𝑉generation to PIM. Overall,
IANUS achieves speedups of 4.0 ×and 3.6×for GPT-2 XL and
L models, respectively, compared to NPU-MEM.
04080
4 8 16 4 8 16 4 8 16 4 8 16
GPT-2 M GPT-2 L GPT-2 XL GPT-2 2.5BLATENCY
(ms) Matrix unit PIM Mapped unit with Algorithm 1Figure 12. Performance evaluation of the adaptive mapping
algorithm for FC across different GPT-2 models as the num-
ber of input tokens are varied from 4, 8, to 16.
Energy Efficiency: Figure 11 presents dynamic energy
consumption of IANUS and NPU-MEM for GPT-2 models
where input and output token sizes are set to 256 and 512, re-
spectively. The energy values are normalized to the dynamic
energy consumed by IANUS with GPT-2 M. By offloading FC
layers of the generation stage to PIM, IANUS achieves 10.5-
13.4×reduction in energy consumption for normal memory
operations across all models. The energy consumption for
computation of cores in NPU is also decreased by a factor of
6.3-10.2×. The reduction in energy consumption for cores’
computation and normal memory operations tends to in-
crease as the model size expands. Meanwhile, the energy is
consumed by PIM operations in IANUS. As a result, IANUS
obtains 3.7×, 3.6×, 3.9×, and 4.4×improvement in energy-
efficiency compared to NPU-MEM for GPT-2 M, L, XL, and
2.5B, respectively. Despite its larger model size, GPT-2 L re-
sults in a smaller energy efficiency improvement compared to
GPT-2 M due to its embedding dimension size of 1280, which
results in twice the number of row activations, compared
to GPT-2 M’s size of 1024 for PIM computation. Note that
energy efficiency in a real system can be further improved if
static energy consumption is also considered2.
Adaptive Mapping Algorithm for FC: To evaluate the
benefits of Algorithm 1, we evaluate the performance of GPT
models when FC is mapped to PIM and matrix unit on various
input token sizes and compared it to the result of Algorithm
1. As illustrated in Figure 12, Algorithm 1 is effective with
small input sizes and chooses the appropriate computation
unit on various model sizes and input token sizes. When
executing FC layers in PIM, execution time is proportional
to the input token size as PIM sequentially repeats matrix-
vector multiplication as much as the input token size. On
the other hand, the matrix unit shows similar performance
across 4, 8, and 16 input tokens because of the capability
of processing 128 tokens in parallel. Therefore, the matrix
unit achieves better performance for large input token sizes.
Another factor for workload mapping is the embedding size
of the model. As the global buffer and row size of PIM is
2KB (= 1024 BF16), models with embedding sizes that are
multiples of 1024 can fully utilize PIM. As a result, PIM shows
higher performance than the matrix unit at an input size of 8
for GPT-2 M (embedding size of 1024) and GPT-2 2.5B (1920,
2Static energy consumption was not incorporated in the analysis because
of the challenge in providing fair comparisons.

## Page 11

IANUS: Integrated Accelerator based on NPU-PIM Unified Memory System ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA
1.0 1.0 1.0 1.0 1.4 1.3 1.3 1.2 1.3 1.5 1.5 3.5 
1.5 1.6 1.6 3.7 
1.6 1.7 1.7 3.5 
1.9 2.0 2.0 4.3 
0.02.55.0
GPT-2 M GPT-2 L GPT-2 XL GPT-2 2.5BSpeedupMatrix unit PIMScheduledUnified Partitioned
Figure 13. Performance comparisons between unified and
partitioned memory systems and the impact of unified
memory-aware scheduling. Dashes in the bar border indicate
the system type. Colors represent mapped units of 𝑄𝐾𝑇and
𝑆𝑉. The pattern indicates the application of scheduling.
nearly 2×1024). With Algorithm 1, we achieve an average
speedup of 1.4×and 1.2×when compared to mapping FC to
PIM and the matrix unit, respectively.
Unified vs. Partitioned Memory System: In Figure 13,
we evaluate the performance benefits of unified memory sys-
tem (IANUS) compared to partitioned memory organization.
Both configurations have the same total memory capacity of
8 GB – with the unified system having 8 GB for both the PIM
and the NPU, while for the partitioned configuration, 4 GB is
dedicated for the NPU’s main memory and 4 GB is dedicated
for PIM. While the memory capacity is the same, the uni-
fied memory has the benefit of additional compute provided
through the extra amount of PIM memory available.
We evaluate both systems using GPT-2 models with a
(256,512) configuration. In the partitioned memory, all FC
parameters shared between PIMs and NPU are duplicated
across the both memory to avoid performance overhead
caused by data movement between standard DRAMs and
PIMs. However, for the 2.5B model, the entire FC parameters
cannot be duplicated across both the PIM and the DRAM
because of the limited memory capacity. Thus, to minimize
transfer overhead between the two types of memories, the
NPU’s matrix unit is mainly responsible for the FC operations
on the non-duplicated parameters. For a fair comparison, we
implement scheduling for the partitioned memory system
that maximizes the benefits from parallel executions of NPU
and PIM by mapping the 𝑄𝐾𝑇and𝑆𝑉to the matrix unit.
As shown in Figure 13, the concurrent execution of NPU’s
DRAM accesses and PIM computations results in an average
1.3×speedup in the partitioned system.
For GPT-2 M, L, and XL models, IANUS–the unified mem-
ory system–(the rightmost bar for each model) outperforms
the scheduled partitioned memory system by 1.4-1.6 ×speedup
(Figure 13). These speedups result from the doubled PIM
throughput that is available in the unified memory config-
uration. For the GPT-2 2.5B model, IANUS shows a larger
performance improvement due to the performance overhead
in the partitioned system, stemming from the data movement
of non-duplicated parameters from the PIM to the NPU. Sim-
ilar to performance trends of other models, while not shown,
IANUS achieves approximately 1.5 ×speedup in GPT-2 2.5B
060120Throughput 
(TFLOPS)GPU IANUS
3.1x2.0x0.8x0.6x
02040
128 256 512 Avg 128 256 512 Avg 128 256 512 Avg 128 256 512 Avg
BERT-B BERT-L BERT-1.3B BERT-3.9BUtilization
(%)5.2x 3.3x1.3x 1.0xFigure 14. Throughput and compute utilization of the BERT
models on A100 GPU and IANUS.
compared to the partitioned system if sufficient memory ca-
pacity is provided such that all FC parameters can be stored
in each memory type.
Unified Memory-Aware Scheduling for Multi-Head
Attention: Figure 13 demonstrates the performance en-
hancement through mapping of 𝑄𝐾𝑇and𝑆𝑉operations and
corresponding scheduling for multi-head attention in IANUS
(the unified memory system). As in the figure, scheduling
for the mapping of 𝑄𝐾𝑇and𝑆𝑉to PIM results in an average
performance boost of 7% across all models compared to naïve
scheduling. When 𝑄𝐾𝑇and𝑆𝑉operations are mapped to
the matrix unit, a reduction in computation time for these
operations leads to superior performance than the case of
scheduling with PIM mapping for all models except GPT-2
2.5B. For the GPT-2 2.5B model, which has a larger head
dimension size of 96 than other models, the loading time
for the previously generated keys and values increases. This
loading time is not required when 𝑄𝐾𝑇and𝑆𝑉are mapped
to PIM, thus reducing the benefits gained through matrix
unit mapping. However, through effective scheduling, we
attain a performance improvement of 24% for GPT-2 2.5B.
Consequently, unified memory-aware scheduling yields an
average performance improvement of 34%.
Throughput and Compute Utilization: Figure 14 presents
the throughput and utilization of the IANUS and the GPU
for BERT models. In IANUS, only the matrix unit and vector
unit of the NPU are utilized for computation, excluding PIM,
as BERT models do not include matrix-vector multiplication.
By managing complex data manipulation in self-attention
through on-chip data movement, IANUS attains 3.1 ×and
2.0×higher average throughput for BERT-B and L, respec-
tively, despite having 1.4 ×lower peak FLOPS than the GPU.
As the FLOPs increase with model size, IANUS’s through-
put becomes less than the GPU due to its limited peak FLOPS.
However, IANUS achieves 5.2 ×, 3.3×, 1.3×, and 1.0×higher
average utilization for BERT-B, L, 1.3B, and 3.9B compared
to the GPU. This enhanced utilization is attributed to the
efficient execution of vector operations with the vector unit
in addition to the benefits gained from self-attention.
Sensitivity Study of Design Parameters: We conduct
sensitivity studies on the number of cores in NPU and PIM

## Page 12

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA Seo et al.
0.01.53.0
1 2 4 1 2 4
# of cores # of PIMsSlowdown(256,1) (256,512)
Figure 15. Sensitivity studies for summarization -only (256,1)
andgeneration -dominant cases (256,512) as the numbers of
cores and PIM chips are varied. Results are normalized to 4
cores and 4 PIMs.
NPU Simulator
DMACommand 
SchedulerCompiler
PIM Runtime Library
PIM Device DriverPIM Hardware Platform
PIM
PCUPIM MCs
FPGA
NPU SW model
DMACommand 
SchedulerCompiler
PIM Runtime Library
PIM Device DriverPIM Hardware Platform
PIM
PCUPIM MCs
FPGA
Figure 16. System prototype of IANUS (PCU: PIM Control
Unit).
chips. To show the sensitivities of NPU and PIM computa-
tion capabilities, we keep memory bandwidth the same as
the baseline while varying the number of cores and PIM
chips. We present summarization -only (256,1) and genera-
tion-dominant (256,512) cases for comprehensive analysis
with GPT-2 L to isolate the impacts from reduced on-chip
memory or PIM capacity. As shown in Figure 15, the fewer
cores result in slowdowns for both cases due to the decreased
intra-layer and attention-head parallelism, and summariza-
tion-only case suffers more as NPU executes all but one
computation (LM head). On the other hand, PIM’s computa-
tion capability significantly affects the generation -dominant
configuration, where a significant fraction of FC operations
are executed on PIMs.
6.3 IANUS System Prototyping
We develop a system prototype of IANUS to validate feasibil-
ity as shown in Figure 16. Our prototype is based on commod-
ity Xilinx FPGA board (VCU118) [ 48] to assess the feasibility
of IANUS with real PIM chips, GDDR6-AiM [ 26,30]. Specif-
ically, we use the AiM-embedded FPGA Mezzanine Card
(FMC) and connect it to the FPGA via the FMC connector [ 28].
As shown, the PIM control unit (PCU) and PIM memory con-
trollers (PIM MCs) are implemented on FPGA whereas we
leverage our NPU simulator as the NPU of IANUS since its
RTL design was too big to fit in a single FPGA. When macro
PIM commands are ready to be executed in the NPU, they
are dispatched to PCU through the PCIe interface by the
PIM runtime library and device driver. These macro com-
mands are then converted into corresponding micro PIM
commands and are transferred to PIM through PCU and
PIM MCs. DMA commands from the NPU simulator are alsoTable 4. Network configurations of larger LLMs.
# ParamsEmbedding
dimensionHead
dimension# Heads # Blocks Workload
GPT6.7B 4096 128 32 32 Language
modeling
(LM)13B 5120 128 40 40
30B 7168 128 56 48
33 160 1168 9457 
2705 
54 251 1801 14812 
4229 
107 484 3486 28230 
8077 
52 101 504 3901 
1139 
64 118 554 4217 
1238 
95 161 694 5126 
1519 
1.E+01.E+21.E+41.E+6
(256,1)
(256,8)
(256,64)
(256,512)
Avg
(256,1)(256,8)
(256,64)
(256,512)
Avg
(256,1)(256,8)
(256,64)
(256,512)
Avg
(Input size, Output size)
6.7B, 2 IANUSs(Input size, Output size)
13B, 4 IANUSs(Input size, Output size)
30B, 8 IANUSsLATENCY   GPU IANUS(ms)2.4x3.4x5.3x
Figure 17. Inference performance scalability for larger LLMs
with multiple IANUS devices. The results are compared to a
single A100 GPU.
similarly transferred to PIM. To validate the functionality
of a system prototype for IANUS, we evaluate the accuracy
of our system using pretrained models of GPT-2 [ 39] on
the WikiText-2 dataset. Our system prototype achieves per-
plexity scores of 30.92 and 22.60, 19.39, and 17.48 for GPT-2
Base (117M), M, L, and XL, respectively, achieving similar
perplexity scores as the full-precision models.
7 Discussion
7.1 Scalability Analysis
Given the limited memory capacity of IANUS compared to
modern GPUs, the memory capacity of IANUS needs to be
scaled to run larger LLMs. The memory (PIM) capacity of
IANUS can be expanded in two ways: 1) increase the amount
of PIM per NPU, or 2) scale the number of IANUS devices.
The first approach can be achieved by adding more PIM
controllers or employing a clamshell configuration of GDDR6
devices [ 35]. However, this approach requires modifications
to the IANUS architecture. In this work, we leverage the
second approach to analyze the scalability of IANUS.
The larger LLMs used in the scalability analysis are sum-
marized in Table 4. For each model, the number of IANUS
devices is selected to provide sufficient memory capacity to
support the model – i.e., two, four, and eight IANUS devices
are used to support the GPT 6.7B, 13B, and 30B models, re-
spectively. The multiple IANUS devices are assumed to be
interconnected through PCIe 5.0 ×16 host interface. To max-
imize parallelism across IANUS devices, both intra-layer par-
allelism and attention head parallelism are exploited among
devices.
As shown in Figure 17, multiple IANUS devices provide
average speedups of 2.4 ×, 3.4×, and 5.3×across respective

## Page 13

IANUS: Integrated Accelerator based on NPU-PIM Unified Memory System ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA
127.1 211.6 317.6 
0200400
2 IANUSs 4 IANUSs 8 IANUSsTokens per 
second1.67x1.50x
Figure 18. Strong scaling of IANUS on the GPT 6.7B model.
models, compared to a single A100 GPU, which has suffi-
cient memory capacity for the larger LLMs. Multiple IANUS
devices not only provide additional memory capacity but
also increase effective memory bandwidth with extra PIM
capability. For larger LLMs, the key system component that
impacts overall performance is the memory bandwidth. This
is because the proportion of FC layers in LLMs, which are bot-
tlenecked by memory bandwidth, increases as the size of the
LLMs grows. Leveraging the PIM’s internal memory band-
width, the effective memory bandwidth of IANUS reaches
approximately 2.4 TB/s, 9-10 ×higher than external mem-
ory bandwidth of GDDR6 memories. Thus, with two IANUS
devices, the total effective bandwidth is 4.8 TB/s, which is ap-
proximately 2.4×higher than the A100 memory bandwidth
(2039 GB/s). This difference in memory bandwidth nearly
matches the observed performance benefits of two IANUS
devices over the A100 GPU. However, scaling the number
of IANUS devices comes at the cost of communication over-
head between IANUS devices compared to a single GPU. As
a result, the performance benefits with four and eight IANUS
devices do not match the theoretical memory bandwidth dif-
ference; however, there is still significant speedup compared
to a single A100 GPU.
Strong scaling of IANUS is shown in Figure 18, using the
6.7B model with a 256:64 token configuration. As described
earlier, the additional IANUS devices provide higher effective
memory bandwidth and result in performance gain – 2.5 ×
performance improvement when the number of IANUS is
increased by 4×. While the performance of IANUS improves
with extra devices, linear speedup is not obtained because of
the communication overhead between multiple devices. A
multi-IANUS device system presents new opportunities for
optimizing communication across the devices but we leave
such exploration as part of future work.
7.2 Cost Analysis
Providing a fair cost comparison between two different sys-
tem architectures (e.g., HBM memory with interposer vs
GDDR6-based PIM memory) is a challenge since many fac-
tors impact cost. However, prior work has shown how ther-
mal design power (TDP) can approximate total cost of owner-
ship (TCO) in datacenters [ 22]. Therefore, we use TDP for the
cost comparison. The TDP of A100 GPU is estimated at 400
W [37] while the TDP of IANUS is conservatively assumed
to be 120 W, based on estimates from the NPU [ 1,20,41] andPIM [ 26,30] components. Using the performance/TDP met-
ric for the cost-efficiency evaluation, configurations of two,
four, and eight IANUS devices yield improvements in cost-
efficiency of 3.9×, 2.7×, and 2.1×over the single A100 GPU
for the 6.7B, 13B, and 30B models, respectively. For each
comparison, the performances of IANUS devices and the
GPU for the performance/TDP metric are measured with a
256:64 input-to-output token ratio. While the cost-efficiency
benefits of IANUS devices are evident, they diminish as the
number of IANUS devices increases. The cost efficiency of
IANUS can potentially be enhanced by leveraging PIM chips
with higher memory capacity and/or more PIM chips con-
nected to a single NPU.
8 Related Works
Domain-specific Accelerators: Various hardware acceler-
ators have been proposed to accelerate transformer models.
TurboTransformer [ 10] executes BERT variants effectively
by operation fusion and pipelining. Unfortunately, this ap-
proach suffers from severe under-utilization in text genera-
tion workloads. Several accelerators [ 13,14,34,45,49] focus
on multi-head-attention mechanisms only, requiring addi-
tional hardware to handle other operations such as layer
normalization. Meanwhile, IANUS utilizes integrated both
NPU and PIM to accelerate end-to-end inference of LLMs.
PIM Accelerators : Utilizing large in-memory bandwidth,
PIM architectures reduce massive data movement between
DRAMs and a host [ 16,24,29–32]. McDRAM [ 42] presents
near-memory structures and a horizontal arrangement of
data within memory banks. TransPIM [ 51] introduces the
first memory-based accelerator for end-to-end inference of
transformers. However, it only achieves an average through-
put of 734 GOPS due to area and power constraints. Chopim [ 7]
employs a unified memory system between the near-data
accelerator and the host. Unlike Chopim, which focuses on
homogeneous kernels, IANUS targets heterogeneous kernels
that require scheduling to optimize performance.
9 Conclusion
We propose IANUS, an integrated accelerator based on NPU-
PIM unified memory system, that fully exploits the bene-
fits of NPU and PIM to accelerate end-to-end inference in
transformer-based LLMs. To overcome the challenges posed
by a unified memory system with PIM, we propose PIM Ac-
cess Scheduling that schedules PIM operations and normal
memory accesses through the workload mapping and sched-
uling. IANUS results in 6.2 ×and 3.2×speedup in compari-
son to GPU and DFX-based solutions for the GPT-2 model,
highlighting the potential of such hybrid architectures. To
demonstrate the feasibility of IANUS, we constructed an
FPGA prototype system based on a commercial NPU and
real PIM chips.

## Page 14

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA Seo et al.
Acknowledgments
We thank the shepherd and all reviewers for their valuable
comments. This work was supported in part by the IITP
grant funded by the MSIT (No.RS 2023-00228255, PIM-NPU
Based Processing System Software Developments for Hyper-
scale Artificial Neural Network Processing) and in part by
the IITP grant funded by the MSIT (No. 2021-0-00106, AI
accelerator-optimized neural network automatic generation
technology and open service platform development).
References
[1]Minwook Ahn, Seok Joong Hwang, Wonsub Kim, Seungrok Jung,
Yeonbok Lee, Mookyoung Chung, Woohyung Lim, and Youngjoon Kim.
Aix: A high performance and energy efficient inference accelerator on
fpga for a dnn-based commercial speech recognition. In 2019 Design,
Automation & Test in Europe Conference & Exhibition (DATE) , pages
1495–1500. IEEE, 2019.
[2]Jorge Albericio, Patrick Judd, Tayler Hetherington, Tor Aamodt, Na-
talie Enright Jerger, and Andreas Moshovos. Cnvlutin: Ineffectual-
neuron-free deep neural network computing. ACM SIGARCH Com-
puter Architecture News , 44(3):1–13, 2016.
[3]Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer nor-
malization. arXiv preprint arXiv:1607.06450 , 2016.
[4]John Bridle. Training stochastic model recognition algorithms as
networks can lead to maximum mutual information estimation of
parameters. Advances in neural information processing systems , 2, 1989.
[5]Tianshi Chen, Zidong Du, Ninghui Sun, Jia Wang, Chengyong Wu,
Yunji Chen, and Olivier Temam. Diannao: A small-footprint high-
throughput accelerator for ubiquitous machine-learning. ACM
SIGARCH Computer Architecture News , 42(1):269–284, 2014.
[6]Yu-Hsin Chen, Tushar Krishna, Joel S Emer, and Vivienne Sze. Eyeriss:
An energy-efficient reconfigurable accelerator for deep convolutional
neural networks. IEEE journal of solid-state circuits , 52(1):127–138,
2016.
[7]Benjamin Y. Cho, Yongkee Kwon, Sangkug Lym, and Mattan Erez.
Near data acceleration with concurrent host access. In 2020 ACM/IEEE
47th Annual International Symposium on Computer Architecture (ISCA) ,
pages 818–831, 2020.
[8]Fabrice Devaux. The true processing in memory accelerator. In 2019
IEEE Hot Chips 31 Symposium (HCS) , pages 1–24. IEEE Computer
Society, 2019.
[9]Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
Bert: Pre-training of deep bidirectional transformers for language
understanding. arXiv preprint arXiv:1810.04805 , 2018.
[10] Jiarui Fang, Yang Yu, Chengduo Zhao, and Jie Zhou. Turbotrans-
formers: an efficient gpu serving system for transformer models. In
Proceedings of the 26th ACM SIGPLAN Symposium on Principles and
Practice of Parallel Programming , pages 389–402, 2021.
[11] Joseph A Fisher. Very long instruction word architectures and the
eli-512. In Proceedings of the 10th annual international symposium on
Computer architecture , pages 140–150, 1983.
[12] Jeremy Fowers, Kalin Ovtcharov, Michael Papamichael, Todd Mas-
sengill, Ming Liu, Daniel Lo, Shlomi Alkalay, Michael Haselman,
Logan Adams, Mahdi Ghandi, Stephen Heil, Prerak Patel, Adam
Sapek, Gabriel Weisz, Lisa Woods, Sitaram Lanka, Steven K. Reinhardt,
Adrian M. Caulfield, Eric S. Chung, and Doug Burger. A configurable
cloud-scale dnn processor for real-time ai. In 2018 ACM/IEEE 45th An-
nual International Symposium on Computer Architecture (ISCA) , pages
1–14, 2018.
[13] Tae Jun Ham, Sung Jun Jung, Seonghak Kim, Young H. Oh, Yeonhong
Park, Yoonho Song, Jung-Hun Park, Sanghee Lee, Kyoung Park, Jae W.
Lee, and Deog-Kyoon Jeong. 𝑎3: Accelerating attention mechanismsin neural networks with approximation. In 2020 IEEE International
Symposium on High Performance Computer Architecture (HPCA) , pages
328–341, 2020.
[14] Tae Jun Ham, Yejin Lee, Seong Hoon Seo, Soosung Kim, Hyunji Choi,
Sung Jun Jung, and Jae W Lee. Elsa: Hardware-software co-design
for efficient, lightweight self-attention mechanism in neural networks.
In2021 ACM/IEEE 48th Annual International Symposium on Computer
Architecture (ISCA) , pages 692–705. IEEE, 2021.
[15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep
residual learning for image recognition. In Proceedings of the IEEE
conference on computer vision and pattern recognition , pages 770–778,
2016.
[16] Mingxuan He, Choungki Song, Ilkon Kim, Chunseok Jeong, Seho
Kim, Il Park, Mithuna Thottethodi, and TN Vijaykumar. Newton: A
dram-maker’s accelerator-in-memory (aim) architecture for machine
learning. In 2020 53rd Annual IEEE/ACM International Symposium on
Microarchitecture (MICRO) , pages 372–385. IEEE, 2020.
[17] Kartik Hegde, Jiyong Yu, Rohit Agrawal, Mengjia Yan, Michael Pel-
lauer, and Christopher Fletcher. Ucnn: Exploiting computational reuse
in deep neural networks via weight repetition. In 2018 ACM/IEEE
45th Annual International Symposium on Computer Architecture (ISCA) ,
pages 674–687. IEEE, 2018.
[18] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus).
arXiv preprint arXiv:1606.08415 , 2016.
[19] Seongmin Hong, Seungjae Moon, Junsoo Kim, Sungjae Lee, Minsub
Kim, Dongsoo Lee, and Joo-Young Kim. Dfx: A low-latency multi-fpga
appliance for accelerating transformer-based text generation. In 2022
55th IEEE/ACM International Symposium on Microarchitecture (MICRO) ,
pages 616–630. IEEE, 2022.
[20] Seok Joong Hwang, Jeongho Han, Minwook Ahn, Seungrok Jung,
Wonsub Kim, Yongshik Moon, Sangjun Yang, Moo-Kyoung Chung,
Jaehyeok Jang, Youngjae Jin, Yongsang Park, Namseob Lee, Daewoo
Kim, Euiseok Kim, Choong Hwan Choi, and Heeyul Lee. Aix v2:
Flexible high performance ai inference accelerator for datacenters. In
2019 IEEE Hot Chips 31 Symposium (HCS) , 2019.
[21] Norm Jouppi, George Kurian, Sheng Li, Peter Ma, Rahul Nagarajan,
Lifeng Nai, Nishant Patil, Suvinay Subramanian, Andy Swing, Brian
Towles, Clifford Young, Xiang Zhou, Zongwei Zhou, and David A
Patterson. Tpu v4: An optically reconfigurable supercomputer for ma-
chine learning with hardware support for embeddings. In Proceedings
of the 50th Annual International Symposium on Computer Architec-
ture, ISCA ’23, New York, NY, USA, 2023. Association for Computing
Machinery.
[22] Norman P. Jouppi, Doe Hyun Yoon, Matthew Ashcraft, Mark Gottscho,
Thomas B. Jablin, George Kurian, James Laudon, Sheng Li, Peter Ma,
Xiaoyu Ma, Thomas Norrie, Nishant Patil, Sushma Prasad, Cliff Young,
Zongwei Zhou, and David Patterson. Ten lessons from three genera-
tions shaped google’s tpuv4i : Industrial product. In 2021 ACM/IEEE
48th Annual International Symposium on Computer Architecture (ISCA) ,
pages 1–14, 2021.
[23] Norman P. Jouppi, Cliff Young, Nishant Patil, David Patterson, Gau-
rav Agrawal, Raminder Bajwa, Sarah Bates, Suresh Bhatia, Nan Bo-
den, Al Borchers, Rick Boyle, Pierre-luc Cantin, Clifford Chao, Chris
Clark, Jeremy Coriell, Mike Daley, Matt Dau, Jeffrey Dean, Ben Gelb,
Tara Vazir Ghaemmaghami, Rajendra Gottipati, William Gulland,
Robert Hagmann, C. Richard Ho, Doug Hogberg, John Hu, Robert
Hundt, Dan Hurt, Julian Ibarz, Aaron Jaffey, Alek Jaworski, Alexan-
der Kaplan, Harshit Khaitan, Daniel Killebrew, Andy Koch, Naveen
Kumar, Steve Lacy, James Laudon, James Law, Diemthu Le, Chris
Leary, Zhuyuan Liu, Kyle Lucke, Alan Lundin, Gordon MacKean, Adri-
ana Maggiore, Maire Mahony, Kieran Miller, Rahul Nagarajan, Ravi
Narayanaswami, Ray Ni, Kathy Nix, Thomas Norrie, Mark Omernick,
Narayana Penukonda, Andy Phelps, Jonathan Ross, Matt Ross, Amir

## Page 15

IANUS: Integrated Accelerator based on NPU-PIM Unified Memory System ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA
Salek, Emad Samadiani, Chris Severn, Gregory Sizikov, Matthew Snel-
ham, Jed Souter, Dan Steinberg, Andy Swing, Mercedes Tan, Gregory
Thorson, Bo Tian, Horia Toma, Erick Tuttle, Vijay Vasudevan, Richard
Walter, Walter Wang, Eric Wilcox, and Doe Hyun Yoon. In-datacenter
performance analysis of a tensor processing unit. In Proceedings of the
44th Annual International Symposium on Computer Architecture , ISCA
’17, page 1–12, New York, NY, USA, 2017. Association for Computing
Machinery.
[24] Jin Hyun Kim, Shin-Haeng Kang, Sukhan Lee, Hyeonsu Kim, Yuhwan
Ro, Seungwon Lee, David Wang, Jihyun Choi, Jinin So, YeonGon Cho,
JoonHo Song, Jeonghyeon Cho, Kyomin Sohn, and Nam Sung Kim.
Aquabolt-xl hbm2-pim, lpddr5-pim with in-memory processing, and
axdimm with acceleration buffer. IEEE Micro , 42(3):20–30, 2022.
[25] Hsiang Tsung Kung and Charles E Leiserson. Systolic arrays (for vlsi).
InSparse Matrix Proceedings 1978 , volume 1, pages 256–282. Society
for industrial and applied mathematics Philadelphia, PA, USA, 1979.
[26] Daehan Kwon, Seongju Lee, Kyuyoung Kim, Sanghoon Oh, Joonhong
Park, Gi-Moon Hong, Dongyoon Ka, Kyudong Hwang, Jeongje Park,
Kyeongpil Kang, Jungyeon Kim, Junyeol Jeon, Nahsung Kim, Yongkee
Kwon, Vladimir Kornijcuk, Woojae Shin, Jongsoon Won, Minkyu Lee,
Hyunha Joo, Haerang Choi, Guhyun Kim, Byeongju An, Jaewook Lee,
Donguc Ko, Younggun Jun, Ilwoong Kim, Choungki Song, Ilkon Kim,
Chanwook Park, Seho Kim, Chunseok Jeong, Euicheol Lim, Dongkyun
Kim, Jieun Jang, Il Park, Junhyun Chun, and Joohwan Cho. A 1ynm
1.25 v 8gb 16gb/s/pin gddr6-based accelerator-in-memory supporting
1tflops mac operation and various activation functions for deep learn-
ing application. IEEE Journal of Solid-State Circuits , 58(1):291–302,
2022.
[27] Hyoukjun Kwon, Ananda Samajdar, and Tushar Krishna. Maeri: En-
abling flexible dataflow mapping over dnn accelerators via reconfig-
urable interconnects. ACM SIGPLAN Notices , 53(2):461–475, 2018.
[28] Yongkee Kwon, Kornijcuk Vladimir, Nahsung Kim, Woojae Shin, Jong-
soon Won, Minkyu Lee, Hyunha Joo, Haerang Choi, Guhyun Kim,
Byeongju An, Jeongbin Kim, Jaewook Lee, Ilkon Kim, Jaehan Park,
Chanwook Park, Yosub Song, Byeongsu Yang, Hyungdeok Lee, Seho
Kim, Daehan Kwon, Seongju Lee, Kyuyoung Kim, Sanghoon Oh, Joon-
hong Park, Gimoon Hong, Dongyoon Ka, Kyudong Hwang, Jeongje
Park, Kyeongpil Kang, Jungyeon Kim, Junyeol Jeon, Myeongjun Lee,
Minyoung Shin, Minhwan Shin, Jaekyung Cha, Changson Jung, Kijoon
Chang, Chunseok Jeong, Euicheol Lim, Il Park, Junhyun Chun, and
Sk Hynix. System architecture and software stack for gddr6-aim. In
2022 IEEE Hot Chips 34 Symposium (HCS) , pages 1–25. IEEE, 2022.
[29] Young-Cheon Kwon, Suk Han Lee, Jaehoon Lee, Sang-Hyuk Kwon,
Je Min Ryu, Jong-Pil Son, O Seongil, Hak-Soo Yu, Haesuk Lee,
Soo Young Kim, Youngmin Cho, Jin Guk Kim, Jongyoon Choi, Hyun-
Sung Shin, Jin Kim, BengSeng Phuah, HyoungMin Kim, Myeong Jun
Song, Ahn Choi, Daeho Kim, SooYoung Kim, Eun-Bong Kim, David
Wang, Shinhaeng Kang, Yuhwan Ro, Seungwoo Seo, JoonHo Song,
Jaeyoun Youn, Kyomin Sohn, and Nam Sung Kim. 25.4 a 20nm
6gb function-in-memory dram, based on hbm2 with a 1.2 tflops pro-
grammable computing unit using bank-level parallelism, for machine
learning applications. In 2021 IEEE International Solid-State Circuits
Conference (ISSCC) , volume 64, pages 350–352. IEEE, 2021.
[30] Seongju Lee, Kyuyoung Kim, Sanghoon Oh, Joonhong Park, Gimoon
Hong, Dongyoon Ka, Kyudong Hwang, Jeongje Park, Kyeongpil Kang,
Jungyeon Kim, Junyeol Jeon, Nahsung Kim, Yongkee Kwon, Korni-
jcuk Vladimir, Woojae Shin, Jongsoon Won, Minkyu Lee, Hyunha Joo,
Haerang Choi, Jaewook Lee, Donguc Ko, Younggun Jun, Keewon Cho,
Ilwoong Kim, Choungki Song, Chunseok Jeong, Daehan Kwon, Jieun
Jang, Il Park, Junhyun Chun, and Joohwan Cho. A 1ynm 1.25 v 8gb,
16gb/s/pin gddr6-based accelerator-in-memory supporting 1tflops mac
operation and various activation functions for deep-learning applica-
tions. In 2022 IEEE International Solid-State Circuits Conference (ISSCC) ,
volume 65, pages 1–3. IEEE, 2022.[31] Sukhan Lee, Shin-haeng Kang, Jaehoon Lee, Hyeonsu Kim, Eojin Lee,
Seungwoo Seo, Hosang Yoon, Seungwon Lee, Kyounghwan Lim, Hyun-
sung Shin, Jinhyun Kim, O Seongil, Anand Iyer, David Wang, Kyomin
Sohn, and Nam Sung Kim. Hardware architecture and software stack
for pim based on commercial dram technology : Industrial product.
In2021 ACM/IEEE 48th Annual International Symposium on Computer
Architecture (ISCA) , pages 43–56, 2021.
[32] Shuangchen Li, Dimin Niu, Krishna T Malladi, Hongzhong Zheng, Bob
Brennan, and Yuan Xie. Drisa: A dram-based reconfigurable in-situ
accelerator. In Proceedings of the 50th Annual IEEE/ACM International
Symposium on Microarchitecture , pages 288–301, 2017.
[33] Daofu Liu, Tianshi Chen, Shaoli Liu, Jinhong Zhou, Shengyuan Zhou,
Olivier Teman, Xiaobing Feng, Xuehai Zhou, and Yunji Chen. Pudi-
annao: A polyvalent machine learning accelerator. ACM SIGARCH
Computer Architecture News , 43(1):369–381, 2015.
[34] Liqiang Lu, Yicheng Jin, Hangrui Bi, Zizhang Luo, Peng Li, Tao Wang,
and Yun Liang. Sanger: A co-design framework for enabling sparse
attention using reconfigurable architecture. In MICRO-54: 54th Annual
IEEE/ACM International Symposium on Microarchitecture , pages 977–
991, 2021.
[35] Micron. Gddr6 datasheet. [Online]. Available: https://media-
www.micron.com/-/media/client/global/documents/products/data-
sheet/dram/gddr/gddr6/gddr6_sgram_8gb_brief.pdf .
[36] Thomas Norrie, Nishant Patil, Doe Hyun Yoon, George Kurian, Sheng
Li, James Laudon, Cliff Young, Norman Jouppi, and David Patterson.
The design process for google’s training chips: Tpuv2 and tpuv3. IEEE
Micro , 41(2):56–63, 2021.
[37] NVIDIA. Nvidia a100 tensor core gpu. [Online]. Available: https:
//www.nvidia.com/en-us/data-center/a100/ .
[38] OpenAI. Input:output token ratio. [Online]. Available: https://beta.
openai.com/docs/usage-guidelines/use-case-guidelines .
[39] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei,
and Ilya Sutskever. Language models are unsupervised multitask
learners. OpenAI blog , 1(8):9, 2019.
[40] Scott Rixner, William J Dally, Ujval J Kapasi, Peter Mattson, and John D
Owens. Memory access scheduling. ACM SIGARCH Computer Archi-
tecture News , 28(2):128–138, 2000.
[41] SAPEON. Product of SAPEON - X330. [Online]. Available: https:
//www.sapeon.com/products/sapeon-x330 .
[42] Hyunsung Shin, Dongyoung Kim, Eunhyeok Park, Sungho Park,
Yongsik Park, and Sungjoo Yoo. Mcdram: Low latency and energy-
efficient matrix computations in dram. IEEE Transactions on Computer-
Aided Design of Integrated Circuits and Systems , 37(11):2613–2622, 2018.
[43] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley,
Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-
billion parameter language models using model parallelism. arXiv
preprint arXiv:1909.08053 , 2019.
[44] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention
is all you need. Advances in neural information processing systems , 30,
2017.
[45] Hanrui Wang, Zhekai Zhang, and Song Han. Spatten: Efficient sparse
attention architecture with cascade token and head pruning. In 2021
IEEE International Symposium on High-Performance Computer Archi-
tecture (HPCA) , pages 97–110. IEEE, 2021.
[46] Shibo Wang and Pankaj Kanwar. Bfloat16: The secret to high perfor-
mance on cloud tpus. Google Cloud Blog , 4, 2019.
[47] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond,
Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf,
Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen,
Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Syl-
vain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush.
Huggingface’s transformers: State-of-the-art natural language process-
ing. arXiv preprint arXiv:1910.03771 , 2019.

## Page 16

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA Seo et al.
[48] Xilinx. Xilinx VCU118 Evaluation Kit. [Online]. Available: https:
//www.xilinx.com/products/boards-and-kits/vcu118.html .
[49] Amir Yazdanbakhsh, Ashkan Moradifirouzabadi, Zheng Li, and Mingu
Kang. Sparse attention acceleration with synergistic in-memory prun-
ing and on-chip recomputation. In 55th IEEE/ACM International Sym-
posium on Microarchitecture, MICRO 2022, Chicago, IL, USA, October
1-5, 2022 , pages 744–762. IEEE, 2022.
[50] Joonsang Yu, Junki Park, Seongmin Park, Minsoo Kim, Sihwa Lee,
Dong Hyun Lee, and Jungwook Choi. Nn-lut: neural approximationof non-linear operations for efficient transformer inference. In Pro-
ceedings of the 59th ACM/IEEE Design Automation Conference , pages
577–582, 2022.
[51] Minxuan Zhou, Weihong Xu, Jaeyoung Kang, and Tajana Rosing.
Transpim: A memory-based acceleration via software-hardware co-
design for transformer. In 2022 IEEE International Symposium on
High-Performance Computer Architecture (HPCA) , pages 1071–1085.
IEEE, 2022.