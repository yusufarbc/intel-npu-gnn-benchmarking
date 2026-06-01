# Do We Need Anisotropic Graph Neural Networks (EGC)


## Page 1

Published as a conference paper at ICLR 2022
DOWENEED ANISOTROPIC GRAPH NEURAL
NETWORKS ?
Shyam A. Tailor1Felix L. Opolka1;2Pietro Li `o1Nicholas D. Lane1;3
1Department of Computer Science and Technology, University of Cambridge
2Invenia Labs, Cambridge, UK
3Samsung AI Center, Cambridge, UK
ABSTRACT
Common wisdom in the graph neural network (GNN) community dictates that
anisotropic models—in which messages sent between nodes are a function of
both the source and target node—are required to achieve state-of-the-art perfor-
mance. Benchmarks to date have demonstrated that these models perform better
than comparable isotropic models—where messages are a function of the source
node only. In this work we provide empirical evidence challenging this narra-
tive: we propose an isotropic GNN, which we call Efﬁcient Graph Convolution
(EGC), that consistently outperforms comparable anisotropic models, including
the popular GAT or PNA architectures by using spatially-varying adaptive ﬁlters.
In addition to raising important questions for the GNN community, our work has
signiﬁcant real-world implications for efﬁciency. EGC achieves higher model ac-
curacy, with lower memory consumption and latency, along with characteristics
suited to accelerator implementation, while being a drop-in replacement for exist-
ing architectures. As an isotropic model, it requires memory proportional to the
number of vertices in the graph ( O(V)); in contrast, anisotropic models require
memory proportional to the number of edges ( O(E)). We demonstrate that EGC
outperforms existing approaches across 6 large and diverse benchmark datasets,
and conclude by discussing questions that our work raise for the community go-
ing forward. Code and pretrained models for our experiments are provided at
https://github.com/shyam196/egc .
1 I NTRODUCTION
Graph Neural Networks (GNNs) have emerged as an effective way to build models over arbitrarily
structured data. For example, they have successfully been applied to computer vision tasks: GNNs
can deliver high performance on point cloud data (Qi et al., 2017) and for feature matching across
images (Sarlin et al., 2020). Recent work has also shown that they can be applied to physical
simulations (Pfaff et al., 2020; Sanchez-Gonzalez et al., 2020). Code analysis is another application
domain where GNNs have found success (Guo et al., 2020; Allamanis et al., 2017).
In recent years, the research community has devoted signiﬁcant attention to building more expres-
sive, and better performing, models to process graphs. Efforts to benchmark GNN models, such as
Open Graph Benchmark (Hu et al., 2020), or the work by Dwivedi et al. (2020), have attempted to
more rigorously quantify the relative performance of different proposed architectures. One common
conclusion—explicitly stated by Dwivedi et al. (2020)—is that anisotropic1models, in which mes-
sages sent between nodes are a function of both the source and target node, are the best performing
models. By comparison, isotropic models, where messages are a function of the source node only,
achieve lower accuracy, even if they have efﬁciency beneﬁts over comparable anisotropic models.
Intuitively, this conclusion is satisfying: anisotropic models are inherently more expressive, hence
we would expect them to perform better in most situations. Our work provides a surprising chal-
lenge to this wisdom by providing an isotropic model, called Efﬁcient Graph Convolution ( EGC ),
Corresponding author. Contact at sat62@cam.ac.uk
1Deﬁnition from Dwivedi et al. (2020). Equal to attentional & message passing from Bronstein et al. (2021).
1arXiv:2104.01481v5  [cs.LG]  9 May 2022

## Page 2

Published as a conference paper at ICLR 2022
x
Messages are functions of source and target
nodes and hence must be materializedAnisotropic
b
f(x;b)a
f(x;a)
cf(x;c)
df(x;d)x
Messages are functions of source only ; propagation can
be implemented using matrix multiplication-style approachesIsotropic
b
f(b)a
f(a)
cf(c)
df(d)
Figure 1: Many GNN architectures (e.g. GAT (Veli ˇckovi ´c et al., 2018), PNA (Corso et al., 2020))
incorporate sophisticated message functions to improve accuracy (left). This is problematic as we
must materialize messages, leading to O(E)memory consumption and OPs to calculate messages;
these dataﬂow patterns are also difﬁcult to optimize for at the hardware level. This work demon-
strates that we can use simple message functions, requiring only O(V)memory consumption
(right) and improve performance over existing GNNs.
that outperforms comparable anisotropic approaches, including the popular GAT (Veli ˇckovi ´c et al.,
2018) and PNA (Corso et al., 2020) architectures.
In addition to providing a surprising empirical result for the community, our work has signiﬁcant
practical implications for efﬁciency, as shown in Figure 1. As EGC is an isotropic model achieving
high accuracy, we can take advantage of the efﬁciency beneﬁts offered by isotropic models without
having to compromise on model accuracy. We have seen memory consumption and latency for
state-of-the-art GNN architectures increase to O(E)in recent years, due to state-of-the-art models
incorporating anisotropic mechanisms to boost accuracy. EGC reduces the complexity to O(V),
delivering substantial real-world beneﬁts, albeit with the precise beneﬁt being dependent on the
topology of the graphs the model is applied to. The reader should note that our approach can also
be combined with other approaches for improving the efﬁciency of GNNs. For example, common
hardware-software co-design techniques include quantization and pruning (Sze et al., 2020) could
be combined with this work, which proposes an orthogonal approach of improving model efﬁciency
by improving the underlying architecture design. We also note that our approach can be combined
with graph sampling techniques (Zeng et al., 2019; Hamilton et al., 2017; Chen et al., 2018a) to
improve scalability further when training on graphs with millions, or billions, of nodes.
Contributions (1) We propose a new GNN architecture, Efﬁcient Graph Convolution ( EGC ),
and provide both spatial and spectral interpretations for it. (2)We provide a rigorous evaluation of
our architecture across 6 large graph datasets covering both transductive and inductive use-cases,
and demonstrate that EGC consistently achieves better results than strong baselines. (3)We pro-
vide several ablation studies to motivate the selection of the hyperparameters in our model. (4)We
demonstrate that our model simultaneously achieves better parameter efﬁciency, latency and mem-
ory consumption than competing approaches. Code and pre-trained models for our experiments
(including baselines) can be found at https://github.com/shyam196/egc . At time of
publication, EGC has also been upstreamed to PyTorch Geometric (Fey & Lenssen, 2019).
2 B ACKGROUND
2.1 H ARDWARE -SOFTWARE CO-DESIGN FOR DEEPLEARNING
Several of the popular approaches for co-design have already been described in the introduction:
quantization, pruning, and careful architecture design are all common for CNNs and Transform-
ers (Vaswani et al., 2017). In addition to enabling better performance to be obtained from general
purpose processors such as CPUs and GPUs, these techniques are also essential for maximizing the
return from specialized accelerators; while it may be possible to improve performance over time
due to improvements in CMOS technology, further improvements plateau without innovation at the
algorithmic level (Fuchs & Wentzlaff, 2019). As neural network architecture designers, we cannot
simply rely on improvements in hardware to make our proposals viable for real-world deployment.
2

## Page 3

Published as a conference paper at ICLR 2022
2.2 G RAPH NEURAL NETWORKS
Many GNN architectures can be viewed as a generalization of CNN architectures to the irregular
domain: as in CNNs, representations at each node are built based on the local neighborhood using
parameters that are shared across the graph. GNNs differ as we cannot make assumptions about
the the size of the neighborhood, or the ordering. One common framework used to deﬁne GNNs
is the message passing neural network (MPNN) paradigm (Gilmer et al., 2017). A graph G=
(V;E)has node features X2RNF, adjacency matrix A2RNNand optionally D-dimensional
edge features E2RED. We deﬁne a function that calculates messages from node uto node
v, a differentiable and permutation-invariant aggregator , and an update function 
to calculate
representations at layer l+ 1:h(i)
l+1=
(h(i)
l;j2N(i)[(h(i)
l;h(j)
l;eij)]). Propagation rules for
baseline architecture are provided in Table 1, with further details supplied in Table 5 in the Appendix.
Relative Expressivity of GNNs Common wisdom in the research community states that isotropic
GNNs are less expressive than anisotropic GNNs; empirically this is well supported by benchmarks.
Brody et al. (2022) prove that GAT models can be strictly more expressive than isotropic models.
Bronstein et al. (2021) also discuss the relative expressivity of different classes of GNN layer, and
argue that convolutional (also known as isotropic) models are well suited to problems leveraging
homophily2in the input graph. They further argue that attentional, or full message passing, models
are suited to handling heterophilous problems, but they acknowledge the resource consumption and
trainability of these architectures may be prohibitive—especially in the case of full message passing.
Scaling and Deploying GNNs While GNNs have seen success across a range of domains, there
remain challenges associated with scaling and deploying them. Graph sampling is one approach
to scaling training for large graphs or models which will not ﬁt in memory. Rather than training
over the full graph, each iteration is run over a sampled sub-graph; approaches vary in whether they
sample node-wise (Hamilton et al., 2017), layer-wise (Chen et al., 2018a; Huang et al., 2018), or sub-
graphs (Zeng et al., 2019; Chiang et al., 2019). Alternatively, systems for distributed GNN training
have been proposed (Jia et al., 2020) to scale training beyond the limits of a single accelerator. Some
works have proposed architectures that are designed to accommodate scaling: graph-augmented
MLPs, such as SIGN (Rossi et al., 2020), are explicitly designed as a shallow architecture, as all
the graph operations are done as a pre-processing step. Other work includes applying neural archi-
tecture search (NAS) to arrange existing GNN layers (Zhao et al., 2020), or building quantization
techniques for GNNs (Tailor et al., 2021). Finally, a recent work has shown that using memory-
efﬁcient reversible residuals (Gomez et al., 2017) for GNNs (Li et al., 2021) enables us to train far
deeper and larger GNN models than before, thereby progressing the state-of-the-art accuracy.
Why Are Existing Approaches Not Sufﬁcient? It is worth noting that many of these approaches
have signiﬁcant limitations that we aim to address with our work. Sampling methods are often
ineffective when applied to many problems which involve model generalization to unseen graphs—a
common use-case for GNNs. We evaluated a variety of sampling approaches and observed that even
modest sampling levels, which provide little beneﬁt to memory or latency, cause model performance
to decline noticeably. In addition, these methods do not accelerate the underlying GNN, hence they
may not provide any overall beneﬁt to inference latency. There is also no evidence that we are aware
of that graph-augmented MLPs perform adequately when generalizing to unseen graphs; indeed,
they are known to be theoretically less expressive than standard GNNs (Chen et al., 2021). We
also investigated this setup, and found that these approaches do not offer competitive accuracy with
state-of-the-art approaches. Experiment details and results, along with further discussion of the
limitations of existing work, is provided in Appendix B.
In summary, our work on efﬁcient GNN architecture design is of interest to the community for two
reasons: ﬁrstly, it raises questions about common assumptions, and how we design and evaluate
GNN models; secondly, our work may enable us to scale our models further, potentially yielding
improvements in accuracy. In addition, for tasks where we need to generalize to unseen graphs, such
as code analysis or point cloud processing, we reduce memory consumption and latency, thereby
enabling us to deploy our models to more resource-constrained devices than before. We note that
efﬁcient architecture design can be usefully combined with other approaches including sampling,
quantization, and pruning, where appropriate.
2Homophily means that if two nodes are connected, then they have high similarity
3

## Page 4

Published as a conference paper at ICLR 2022
x(i)Pw(i)=x(i)+b
1
2

3
Run separate graph ﬁlters
parameterised by bover graphApply per-node weightings
w(i)to each basis ﬁlterh(i)
1
h(i)
2
h(i)
3y(i)
Figure 2: Visual representation of our EGC-S layer. In this visualization we have 3 basis ﬁlters (i.e.
B= 3), which are combined using per-node weightings w. This simpliﬁed ﬁgure does not show
the usage of heads, or multiple aggregators, as used by EGC-M.
3 O URARCHITECTURE : EFFICIENT GRAPH CONVOLUTION (EGC)
In this section we describe our approach, and delay theoretical analysis to the next section. We
present two versions: EGC-S (ingle), using a single aggregator, and EGC-M (ulti) which generalizes
our approach by incorporating multiple aggregators. Our approach is visualized in Figure 2.
3.1 A RCHITECTURE DESCRIPTION
For a layer with in-dimension of Fand out-dimension of F0we useBbasis weights b2RF0F.
We compute the output for node iby calculating combination weighting coefﬁcients w(i)2RB
per node , and weighting the results of each aggregation using the different basis weights b. The
output for node iis computed in three steps. First, we perform the aggregation with each set of basis
weights b. Second, we compute the weighting coefﬁcients w(i)=x(i)+b2RBfor each node
i, where 2RBFandb2RBare weight and bias parameters for calculating the combination
weighting coefﬁcients. Third, the layer output for node iis the weighted combination of aggregation
outputs:
y(i)=BX
b=1w(i)
bX
j2N(i)(i;j)bx(j)(1)
where(i;j)is some function of nodes iandj, andN(i)denotes the in-neighbours of i. A popular
method pioneered by GAT (Veli ˇckovi ´c et al., 2018) to boost representational power is to represent 
using a learned function of the two nodes’ representations. While this enables anisotropic treatment
of neighbors, and can boost performance, it necessarily results in memory consumption of O(E)
due to messages needing to be explicitly materialized, and complicates hardware implementation for
accelerators. If we choose a representation for that is not a function of the node representations—
such as(i;j) = 1 to recover the add aggregator used by GIN (Xu et al., 2019), or (i;j) =
1=p
deg(i)deg(j)to recover symmetric normalization used by GCN (Kipf & Welling, 2017)—
then we can implement our message propagation phase using sparse matrix multiplication (SpMM),
and avoid explicitly materializing each message, even for the backwards pass. In this work, we
assume(i;j)to be symmetric normalization as used by GCN unless otherwise stated; we use this
normalization as it is known to offer strong results across a variety of tasks; more formal justiﬁcation
is provided in section 4.2.
Adding Heads as a Regularizer We can extend our layer through the addition of heads, as
used in architectures such as GAT or Transformers (Vaswani et al., 2017). These heads share the
basis weights, but apply different weighting coefﬁcients per head. We ﬁnd that adding this degree of
freedom aids regularization when the number of heads ( H) is larger than B, as bases are discouraged
from specializing (see section 5.3), without requiring the integration of additional loss terms into the
optimization—hence requiring no changes to code for downstream users. To normalize the output
4

## Page 5

Published as a conference paper at ICLR 2022
dimension, we change the basis weight matrices dimensions toF0
HF. Usingkas the concatenation
operator, and making the use of symmetric normalization explicit, we obtain the EGC-S layer:
y(i)=H
k
h=1BX
b=1w(i)
h;bX
j2N(i)[fig1p
deg(i)deg(j)bx(j)(2)
EGC works by combining basis matrices. This idea was proposed in R-GCN (Schlichtkrull et al.,
2018) to handle multiple edge types; Xu et al. (2021) can be viewed as a generalization of this
approach to point cloud analysis. In this work we are solving a different problem to these works: we
are interested in designing efﬁcient architectures, rather than new ways to handle edge information.
3.2 B OOSTING REPRESENTATIONAL CAPACITY
Recent work by Corso et al. (2020) has shown that using only a single aggregator is sub-optimal:
instead, it is better to combine several different aggregators. In Equation (2) we deﬁned our layer to
use only symmetric normalization. To improve performance, we propose applying different aggre-
gators to the representations calculated by bx(j). The choice of aggregators could include different
variants of summation aggregators e.g. mean or unweighted addition, as opposed to symmetric nor-
malization that was proposed in the previous section. Alternatively, we can use aggregators such as
stddev, min or max which are not based on summation. It is also possible to use directional aggre-
gators proposed by Beaini et al. (2021), however this enhancement is orthogonal to this work. If we
have a set of aggregators A, we can extend Equation (2) to obtain our EGC-M layer:
y(i)=H
k
h=1X
2ABX
b=1w(i)
h;;bM
j2N(i)[figbx(j)(3)
whereis an aggregator. With this formulation, we are reusing the same messages we have calcu-
lated as before—but we are applying several aggregation functions to them at the same time.
Aggregator Fusion It would appear that adding more aggregators would cause latency and mem-
ory consumption to grow linearly. However, this is not true in practice. Firstly, since sparse op-
erations are typically memory bound in practice, we can apply extra aggregators to data that has
already arrived from memory with little latency penalty. EGC can also efﬁciently inline the node-
wise weighting operation at inference time, thereby resulting in relatively little memory consump-
tion overhead. The equivalent optimization is more difﬁcult to apply successfully to PNA due to the
larger number of operations per-node that must be performed during aggregation, caused by scaling
functions being applied to each aggregation, before all results are concatenated and a transformation
applied. More details, including proﬁling and latency measurements, can be found in Appendix D.
4 I NTERPRETATION AND BENEFITS
This section will explain our design choices, and why they are better suited to the hardware. We
emphasize that our approach does notdirectly correspond to attention.
4.1 S PATIAL INTERPRETATION : NODE-WISEWEIGHT MATRICES
In our approach, each node effectively has its own weight matrix. We can derive this by re-arranging
eq. (2) by factorizing the bterms out of inner sum:
y(i)=H
k
h=1(i)
h
|{z}
Varying per Node0
@X
j2N(i)[fig1p
deg(i)deg(j)x(j)1
A
| {z }
Computable via SpMM(4)
In contrast, GAT shares weights, and pushes complexity into the message calculation phase by
calculating per-message weightings. MPNN (Gilmer et al., 2017) and PNA (Corso et al., 2020)
further increase complexity by explicitly calculating each message—resulting in substantial latency
overhead due to the number of dense operations increasing by roughlyjEj
jVj. Speciﬁcally, we have:
5

## Page 6

Published as a conference paper at ICLR 2022
y(i)
GAT=H
k
h=1
|{z}
Shared Weights0
@X
j2N(i)[figh;i;jx(j)1
A
| {z }
Calculated Message Weightingy(i)
PNA=U(x(i);M
j2N(i)M(x(i);x(j))|{z}
Explicit Message
Calculation)
From an efﬁciency perspective we observe that our approach of using SpMM has better character-
istics due to it requiring only O(V)memory consumption—no messages must be explicitly mate-
rialized to use SpMM. We note that although it is possible to propagate messages for GAT with
SpMM, there is no way to avoid storing the weightings during training as they are needed for back-
propagation, resulting in O(E)memory consumption. We also note that fusing the message and
aggregation steps for certain architectures may be possible at inference time, but this is a difﬁcult
pattern for hardware accelerators to optimize for.
Relation To Attention Our method is not directly related to attention , which relies upon pairwise
similarity mechanisms, and hence results in a O(E)cost when using the common formulations.
Alternatives to attention-based Transformers proposed by Wu et al. (2019a) are a closer analogue to
our technique, but rely upon explicit prediction of the per-timestep weight matrix. This approach is
not viable for graphs, as the neighborhood size is not constant.
4.2 S PECTRAL INTERPRETATION : LOCALISED SPECTRAL FILTERING
We can also interpret our EGC-S layer through the lens of graph signal processing (Sandryhaila &
Moura, 2013). Many modern graph neural networks build on the observation that the convolution
operation for the Euclidean domain when generalised to the graph domain has strong inductive
biases: it respects the structure of the domain and preserves the locality of features by being an
operation localised in space. Our method can be viewed as a method of building adaptive ﬁlters for
the graph domain. Adaptive ﬁlters are a common approach when signal or noise characteristics vary
with time or space; for example, they are commonly applied in adaptive noise cancellation. Our
approach can be viewed as constructing adaptive ﬁlters by linearly combining learnable ﬁlter banks
with spatially varying coefﬁcients.
The graph convolution operation is typically deﬁned on the spectral domain as ﬁltering the input
signal x2RNon a graph with Nnodes with a ﬁlter gparameterized by . This requires translating
between the spectral and spatial domain using the Fourier transform. As on the Euclidean domain,
the Fourier transform on the graph-domain is deﬁned as the basis decomposition with the orthogonal
eigenbasis of the Laplace operator, which for a graph with adjacency matrix A2RNNis deﬁned
asL=D A, where Dis the diagonal degree matrix with Dii=PN
j=1Aij. The Fourier transform
of a signal x2RNthen isF(x) =U>x, where L=UU>, with orthogonal eigenvector-matrix
U2RNNand diagonal eigenvalue-matrix 2RNN. The result of a signal xﬁltered bygis
y=g(L)x=Ug()U>xwhere the second equality holds if the Taylor expansion of gexists.
Our approach corresponds to learning multiple ﬁlters and computing a linear combination of the
resulting ﬁlters with weights depending on the attributes of each node locally. The model therefore
allows applying multiple ﬁlters for each node, enabling us to obtain a spatially-varying frequency
response, while staying far below O(E)in computational complexity. Using a linear combination
of ﬁlters, the ﬁltered signal becomes y=PB
b=1wbgb(L)x, where wb2RNare the weights of
ﬁlterbfor each of the Nnodes in the graph. If we parameterize our ﬁlter using ﬁrst-order Chebyshev
polynomials as used by Kipf & Welling (2017) our ﬁnal expression for the ﬁltered signal becomes
Y=PB
b=1wb(~D 1
2~A~D 1
2)Xb, where ~A=A+INis the adjacency matrix with added
self-loops and ~Dis the diagonal degree matrix of ~Aas deﬁned earlier. This justiﬁes the symmetric
normalization aggregator we chose in Equation (2).
Cheng et al. (2021) proposed an approach for localized ﬁltering. However, their approach does not
generalize to unseen topologies or scale to large graphs as it requires learning the coefﬁcients of
several ﬁlter matrices Skof sizeNN. Our approach does not suffer from these constraints.
6

## Page 7

Published as a conference paper at ICLR 2022
Architecture Propagation Rule MemoryZINC (MAE#) CIFAR (Acc. ") MolHIV (ROC-AUC ") Code-V2 (F1 ") Arxiv (Acc.")
Unseen Graph Unseen Graph Unseen Graph Unseen Graph Transductive Node
Regression Classiﬁcation Classiﬁcation Classiﬁcation Classiﬁcation
GCN y(i)=P
j1p
deg(i)deg(j)x(j)O(V) 0.4590.006 55.710.38 76.141.29 0.1480 0.0018 71.920.21
GIN y(i)=f[(1 +)x(i)+P
jx(j)]O(V) 0.3870.015 55.261.53 76.021.35 0.1481 0.0027 67.331.47
GraphSAGE y(i)=1x(i)+L
j2x(j)O(V) 0.4680.003 65.770.31 75.971.69 0.1453 0.0028 71.730.26
GAT y(i)=i;ix(i)+P
ji;jx(j)O(E) 0.4750.007 64.220.46 77.171.37 0.1513 0.001171.810.23
GATv2 y(i)=i;ix(i)+P
ji;jx(j)O(E) 0.4470.015 67.480.53 77.151.55 0.1537 0.002271.870.43
MPNN-Sum y(i)=U(x(i);P
jM(x(i);x(j)))O(E) 0.3810.005 65.390.47 75.193.57 0.1470 0.001766.110.56
MPNN-Max y(i)=U(x(i);maxjM(x(i);x(j)))O(E) 0.4680.002 69.700.55 77.071.37 0.1552 0.002271.020.21
PNA y(i)=U(x(i);L
jM(x(i);x(j)))O(E) 0.3200.032 70.210.15 79.051.320.15700.003271.210.30
EGC-S (Ours) Equation 2 O(V) 0.3640.020 66.920.37 77.441.08 0.1528 0.0025 72.210.17
EGC-M (Ours) Equation 3 O(V) 0.2810.007 71.030.42 78.181.53 0.15950.0019 71.960.23
Table 1: Results (mean standard deviation) for parameter-normalized models run on 5 datasets.
Details of the speciﬁc aggregators chosen per dataset and further experimental details can be found
in the supplementary material. Results marked with ran out of memory on 11GB 1080Ti and
2080Ti GPUs. EGC obtains best performance on 4 of the tasks, with consistently wide margins.
5 E VALUATION
5.1 P ROTOCOL
We primarily evaluate our approach on 5 datasets taken from recent works on GNN benchmarking.
We use ZINC and CIFAR-10 Superpixels from Dwivedi et al. (2020) and Arxiv, MolHIV and Code
from Open Graph Benchmark (Hu et al., 2020). These datasets cover a wide range of domains,
cover both transductive and inductive tasks, and are larger than datasets which are typically used in
GNN works. We use evaluation metrics and splits speciﬁed by these papers. Baseline architectures
chosen reﬂect popular general-purpose choices (Kipf & Welling, 2017; Xu et al., 2019; Hamilton
et al., 2017; Veli ˇckovi ´c et al., 2018; Gilmer et al., 2017), along with the state-of-the-art PNA (Corso
et al., 2020) and GATv2 (Brody et al., 2022) architectures.
In order to provide a fair comparison we standardize all parameter counts, architectures and opti-
mizers in our experiments. All experiments were run using Adam (Kingma & Ba, 2014). Further
details on how we ensured a fair evaluation can be found in the appendix.
We do not use edge features in our experiments as for most baseline architectures there exist no
standard method to incorporate them. We do not use sampling, which, as explained in Section 2.2,
is ineffective for 4 datasets; for the remaining dataset, Arxiv, we believe it is not in the scientiﬁc
interest to introduce an additional variable. This also applies to GraphSAGE, where we do not use
the commonly applied neighborhood sampling. All experiments were run 10 times.
5.2 M AINRESULTS
Our results across the 5 tasks are shown in Table 1. We draw attention to the following observations:
•EGC-S is competitive with anisotropic approaches . We outperform GAT(v1) and
MPNN-Sum on all benchmarks, despite our resource efﬁciency. The clearest exception
is MPNN-Max on CIFAR & Code, where the max aggregator provides a stronger inductive
bias. We observe that GATv2 improves upon GAT, but does not clearly outperform EGC.
•EGC-M outperforms PNA . The addition of multiple aggregator functions improves per-
formance of EGC to beyond that obtained by PNA. We hypothesize that our improved
performance over PNA is related to PNA’s reliance on multiple degree-scaling transforms.
While this approach can boost the representational power of the architecture, we believe
that it can result in a tendency to overﬁt to the training set.
•EGC performs strongly without running out of memory. We observe that EGC is one of
only three architectures that did not exhaust the VRAM of the popular Nvidia 1080/2080Ti
GPUs, with 11GB VRAM, when applied to Arxiv: we had to use an RTX 8000 GPU with
48GB VRAM to run these experiments. PNA, our closest competing technique accuracy-
wise, exhausted memory on the Code benchmark as well. Detailed memory consumption
ﬁgures are provided in Table 4.
7

## Page 8

Published as a conference paper at ICLR 2022
1248 1 6
Heads1 2 4 8 16Bases0.438
± 0.011
Hidden: 1420.425
± 0.026
Hidden: 1860.435
± 0.008
Hidden: 2320.408
± 0.014
Hidden: 2720.391
± 0.015
Hidden: 288
0.387
± 0.022
Hidden: 1050.389
± 0.022
Hidden: 1400.378
± 0.014
Hidden: 1800.388
± 0.022
Hidden: 2160.378
± 0.014
Hidden: 224
0.402
± 0.029
Hidden: 760.399
± 0.019
Hidden: 1040.405
± 0.019
Hidden: 1360.364
± 0.020
Hidden: 1680.393
± 0.015
Hidden: 176
0.396
± 0.020
Hidden: 540.391
± 0.024
Hidden: 740.397
± 0.023
Hidden: 1000.396
± 0.029
Hidden: 1200.369
± 0.023
Hidden: 1 12
0.397
± 0.017
Hidden: 390.419
± 0.011
Hidden: 520.400
± 0.028
Hidden: 680.391
± 0.022
Hidden: 800.383
± 0.027
Hidden: 64
0.3750.3800.3850.3900.3950.400
(a) Constant parameter count (100k)
1 2 4 8 16
Heads0.340.360.380.400.420.440.46Mean Absolute Er r or (MAE)Bases
1
2
4
8
16 (b) Constant hidden dimension (128)
Figure 3: Study over the number of heads ( H) and bases ( B). Study run on ZINC dataset with
EGC-S. Metric is MAE (mean standard deviation): lower is better. We study keeping the total
parameter count constant, and ﬁxing the hidden dimension. Each experiment was tuned individually.
SettingB >H does not necessarily improve performance due to the risk of overﬁtting, and forces
the usage of a smaller hidden dimension to retain a constant parameter count.
Overall, EGC obtains the best performance on 4 out of the 5 main datasets; on the remaining dataset
(MolHIV), EGC is the second best architecture. This represents a signiﬁcant achievement: our
architecture demonstrates that we do not need to choose between efﬁciency and accuracy.
5.3 A DDITIONAL STUDIES
How Should Heads and Bases be Chosen? To understand the trade-off between the number of
heads (H) and bases ( B), we ran an ablation study on ZINC using EGC-S; this in shown in Figure 3.
The relationship between these parameters is non-trivial. There are several aspects to consider: (1)
increasingHandBmeans that we spend more of our parameter budget to create the combinations,
which reduces hidden dimension—as shown in Figure 3. This is exacerbated if we use multiple
aggregators: our combination dimension must be HBjAj. (2) Increasing Bmeans we must reduce
the hidden size substantially, since it corresponds to adding more weights of sizeF0
HF. (3)
IncreasingHallows us to increase hidden size, since each basis weight becomes smaller. We see
in Figure 3 that increasing BbeyondHdoes not yield signiﬁcant performance improvements: we
conjecture that bases begin specializing for individual heads; by sharing, there is a regularizing
effect, like observed in Schlichtkrull et al. (2018). This regularization stabilizes the optimization
and we observe lower trial variance for smaller B.
We adviseB=HorB=H
2. We ﬁndH= 8to be effective with EGC-S; for EGC-M, where more
parameters are spent on combination weights, we advise setting H= 4. This convention is applied
consistently for Table 1; full details are supplied in the appendix and code.
Activation EGC-S EGC-M
Identity 0.3640.020 0.2810.007
Hardtanh 0.435 0.010 0.2930.013
Sigmoid 0.366 0.008 0.3030.016
Softmax 0.404 0.010 0.3070.013
Table 2: Activating the combination weightings w
harms performance. Run on ZINC; lower is better.Should The Combination Weightings ( w) Be
Activated? Any activation function will
shrink the space the per-node weights (i)
hcan
lie in, hence we would expect it to harm perfor-
mance; this is veriﬁed in Table 2. Activating
wmay improve training stability, but we did
not observe to be an issue in our experiments.
Another issue is that different aggregators re-
sult in outputs with different means and vari-
ances (Tailor et al., 2021), hence they need to
be scaled by different factors to be combined.
Applying EGC to Large-Scale Heterogeneous Graphs We evaluated EGC on the heteroge-
neous OGB-MAG dataset, containing 2M nodes and 21M edges. On a homogeneous version of the
graph, we exceed the baselines’ performance by 1.5-2%; we experiment with both symmetric nor-
malization (EGC-S) and the mean aggregators to demonstrate that the mechanism utilized by EGC
8

## Page 9

Published as a conference paper at ICLR 2022
is effective, regardless of which aggregator provides the stronger inductive bias for the dataset. Our
architecture can be expanded to handle different edge types, yielding the R-EGC architecture which
improves performance over R-GCN by 0.9%. We expect that accuracy can be further improved by
using sampling techniques to regularize the optimization, or using pretrained embeddings; however,
adding these techniques makes comparing the results more difﬁcult as it is known that sampling
techniques can affect each architecture’s performance differently (Liu et al., 2020).
5.4 M EMORY AND LATENCY BENCHMARKS
Method Test Accuracy % "
MLP 26.92 0.26
GCN 30.43 0.25
GraphSAGE-Mean 31.53 0.15
EGC-S 32.13 0.73
EGC (=Mean) 33.22 0.50
R-GCN (Full Batch) 46.29 0.45
R-EGC (Full Batch) 47.21 0.40
Table 3: EGC can be applied applied to large
scale heterogeneous graphs, outperforming R-
GCN by 0.9%.
Model Peak Training
Memory (MB)GPU Training
Latency (ms)GPU Inference
Latency (ms)
GCN 1905 159.8 4.6 35.20.1
GIN 1756 155.2 3.9 35.20.1
GraphSAGE 1352 113.9 5.4 25.10.4
GAT 10841 324.3 1.2 84.70.3
GATv2 14124 341.8 0.5 129.20.2
MPNN-Sum 14323 768.2 0.8 230.70.3
MPNN-Max 14623 797.8 0.9 258.20.4
PNA 14533 892.7 1.1 305.70.5
EGC-S 2430 177.7 2.2 37.30.1
EGC-M 4068 220.9 0.6 42.20.3
Table 4: Memory and latency statistics for
parameter-normalized models (used in ta-
ble 1) on Arxiv. Note that EGC-M memory
consumption and latency can be reduced with
aggregator fusion at inference time.We now assess our model’s resource efﬁciency. For
CPU measurements we used an Intel Xeon Gold
5218 and for GPU we used an Nvidia RTX 8000.
Aggregator Fusion We evaluated aggregator fu-
sion across several topologies on both CPU and
GPU. For space reasons, we leave full details to
Appendix D. In summary, we observe that using 3
aggregators over standard SpMM incurs an addi-
tional overhead of only 14% , enabling us to im-
prove model performance without excessive com-
putational overheads at inference time.
End-to-End Latency We provide latency and
memory statistics for parameter-normalized mod-
els in Table 4. We draw attention to how slow
and memory-intensive the O(E)models are for
training and inference. The reader should note
that the inference latency and memory consump-
tion for MPNN and PNA rises by 6-7 rela-
tive to EGC, corresponding to the largejEj
jVjra-
tio for Arxiv. EGC-M offers substantially lower
latency and memory consumption than its near-
est competitor, PNA, however the precise bene-
ﬁt will be dataset-dependent. Extended results
can be found in Appendix E, including results for
accuracy-normalized models on Arxiv, which fur-
ther demonstrate EGC’s resource efﬁciency.
6 D ISCUSSION AND CONCLUSION
How Surprising Are Our Results? We observed that it was possible to design an isotropic GNN
that is competitive with state-of-the-art anisotropic GNN models on 6 benchmarks. This result con-
tradicts common wisdom in the GNN community. However, our results may be viewed as part of
a pattern visible across the ML community: both the NLP and computer vision communities have
seen works indicating that anisotropy offered by Transformers may not be necessary (Tay et al.,
2021; Liu et al., 2022), consistent with our observations. It is worth asking why we observe our re-
sults, given that they contradict properties that have been theoretically proven. We believe that there
are a variety of reasons, but the most important is that most real world datasets do not require the
theoretical power these more expressive models provide to achieve good results. In particular, many
real world datasets are homophilous: therefore simplistic approaches, such as EGC, can achieve
high performance. This implies that the community should consider adding more difﬁcult datasets
to standard benchmarks, such as those presented in Lim et al. (2021) and Veli ˇckovi ´c et al. (2021).
Our proposed layer, EGC, can be used as a drop-in replacement for existing GNN layers, and
achieves better results across 6 benchmark datasets compared to strong baselines, with substantially
lower resource consumption. Our work raises important questions for the research community,
while offering signiﬁcant practical beneﬁts with regard to resource consumption. We believe the
next step for our work is incorporation of edge features e.g. through line graphs (Chen et al., 2020b)
or topological message passing (Bodnar et al., 2021).
9

## Page 10

Published as a conference paper at ICLR 2022
ACKNOWLEDGEMENTS
This work was supported by the UK’s Engineering and Physical Sciences Research Council (EP-
SRC) with grant EP/S001530/1 (the MOA project) and the European Research Council (ERC) via
the REDIAL project (Grant Agreement ID: 805194). FLO acknowledges funding from the Huawei
Studentship at the Department of Computer Science and Technology of the University of Cambridge.
The authors would like to thank Ben Day, Javier Fernandez-Marques, Chaitanya Joshi, Titouan
Parcollet, Petar Veli ˇckovi ´c, and the anonymous reviewers who have provided comments on earlier
versions of this work.
ETHICS STATEMENT
The method described in this paper is generic enough that it can be applied to any problem GNNs
are applied to. The ethical concerns associated with this work are related to enabling more efﬁcient
training and deployment of GNNs. This may be positive (drug discovery) or negative (surveillance),
but these concerns are inherent to any work investigating efﬁciency for machine learning systems.
REPRODUCIBILITY STATEMENT
We have supplied the code required to regenerate our results, along with the hyperparameters re-
quired. In addition, we supply pre-trained models. Resources associated with this paper can be
found at https://github.com/shyam196/egc .
REFERENCES
Miltiadis Allamanis, Marc Brockschmidt, and Mahmoud Khademi. Learning to represent programs
with graphs. arXiv preprint arXiv:1711.00740 , 2017.
Gene M Amdahl. Validity of the single processor approach to achieving large scale computing
capabilities. In Proceedings of the April 18-20, 1967, spring joint computer conference , pp. 483–
485, 1967.
Dominique Beaini, Saro Passaro, Vincent L ´etourneau, William L. Hamilton, Gabriele Corso, and
Pietro Li `o. Directional graph networks, 2021.
Cristian Bodnar, Fabrizio Frasca, Yu Guang Wang, Nina Otter, Guido Mont ´ufar, Pietro Li `o, and
Michael Bronstein. Weisfeiler and lehman go topological: Message passing simplicial networks,
2021.
Shaked Brody, Uri Alon, and Eran Yahav. How attentive are graph attention networks? In Interna-
tional Conference on Learning Representations , 2022. URL https://openreview.net/
forum?id=F72ximsx7C1 .
Michael M. Bronstein, Joan Bruna, Taco Cohen, and Petar Veli ˇckovi ´c. Geometric deep learning:
Grids, groups, graphs, geodesics, and gauges, 2021.
Jie Chen, Tengfei Ma, and Cao Xiao. Fastgcn: fast learning with graph convolutional networks via
importance sampling. arXiv preprint arXiv:1801.10247 , 2018a.
Lei Chen, Zhengdao Chen, and Joan Bruna. On graph neural networks versus graph-augmented
fmlpgs. In International Conference on Learning Representations , 2021. URL https://
openreview.net/forum?id=tiqI7w64JG2 .
Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Meghan Cowan, Haichen
Shen, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy. Tvm:
An automated end-to-end optimizing compiler for deep learning. In Proceedings of the 13th
USENIX Conference on Operating Systems Design and Implementation , OSDI’18, pp. 579–594,
USA, 2018b. USENIX Association. ISBN 9781931971478.
10

## Page 11

Published as a conference paper at ICLR 2022
Xiaobing Chen, Yuke Wang, Xinfeng Xie, Xing Hu, Abanti Basak, Ling Liang, Mingyu Yan, Lei
Deng, Yufei Ding, Zidong Du, Yunji Chen, and Yuan Xie. Rubik: A Hierarchical Architecture for
Efﬁcient Graph Learning. arXiv:2009.12495 [cs] , September 2020a. URL http://arxiv.
org/abs/2009.12495 . arXiv: 2009.12495.
Zhengdao Chen, Xiang Li, and Joan Bruna. Supervised community detection with line graph neural
networks, 2020b.
Xiuyuan Cheng, Zichen Miao, and Qiang Qiu. Graph convolution with low-rank learnable local
ﬁlters. In International Conference on Learning Representations , 2021.
Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, and Cho-Jui Hsieh. Cluster-gcn: An
efﬁcient algorithm for training deep and large graph convolutional networks. In Proceedings of
the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining , pp.
257–266, 2019.
Intel Corporation. Intel® 64 and ia-32 architectures optimization reference manual.
Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Li `o, and Petar Veli ˇckovi ´c. Principal
Neighbourhood Aggregation for Graph Nets. arXiv:2004.05718 [cs, stat] , June 2020. URL
http://arxiv.org/abs/2004.05718 . arXiv: 2004.05718.
Vijay Prakash Dwivedi, Chaitanya K. Joshi, Thomas Laurent, Yoshua Bengio, and Xavier Bresson.
Benchmarking Graph Neural Networks. arXiv:2003.00982 [cs, stat] , July 2020. URL http:
//arxiv.org/abs/2003.00982 . arXiv: 2003.00982.
Matthias Fey and Jan Eric Lenssen. Fast Graph Representation Learning with PyTorch Geometric,
5 2019. URL https://github.com/pyg-team/pytorch_geometric .
Adi Fuchs and David Wentzlaff. The accelerator wall: Limits of chip specialization. In 2019 IEEE
International Symposium on High Performance Computer Architecture (HPCA) , pp. 1–14. IEEE,
2019.
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural
message passing for quantum chemistry. In International Conference on Machine Learning , pp.
1263–1272. PMLR, 2017.
Aidan N. Gomez, Mengye Ren, Raquel Urtasun, and Roger B. Grosse. The reversible residual
network: Backpropagation without storing activations, 2017.
Daya Guo, Shuo Ren, Shuai Lu, Zhangyin Feng, Duyu Tang, Shujie Liu, Long Zhou, Nan Duan,
Jian Yin, Daxin Jiang, et al. Graphcodebert: Pre-training code representations with data ﬂow.
arXiv preprint arXiv:2009.08366 , 2020.
William L Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large
graphs. arXiv preprint arXiv:1706.02216 , 2017.
Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta,
and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. arXiv
preprint arXiv:2005.00687 , 2020.
Wenbing Huang, Tong Zhang, Yu Rong, and Junzhou Huang. Adaptive sampling towards fast graph
representation learning. arXiv preprint arXiv:1809.05343 , 2018.
Zhihao Jia, Sina Lin, Mingyu Gao, Matei Zaharia, and Alex Aiken. Improving the accuracy, scala-
bility, and performance of graph neural networks with roc. Proceedings of Machine Learning and
Systems , 2:187–198, 2020.
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint
arXiv:1412.6980 , 2014.
Thomas N. Kipf and Max Welling. Semi-Supervised Classiﬁcation with Graph Convolutional Net-
works. In International Conference on Learning Representations , 2017.
11

## Page 12

Published as a conference paper at ICLR 2022
Guohao Li, Matthias M ¨uller, Bernard Ghanem, and Vladlen Koltun. Training graph neural networks
with 1000 layers, 2021.
Derek Lim, Xiuyu Li, Felix Hohne, and Ser-Nam Lim. New benchmarks for learning on non-
homophilous graphs, 2021.
Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie.
A convnet for the 2020s, 2022.
Ziqi Liu, Zhengwei Wu, Zhiqiang Zhang, Jun Zhou, Shuang Yang, Le Song, and Yuan Qi. Bandit
samplers for training graph neural networks, 2020.
Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, and Peter W Battaglia. Learning mesh-
based simulation with graph networks. arXiv preprint arXiv:2010.03409 , 2020.
Charles R Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning
on point sets in a metric space. arXiv preprint arXiv:1706.02413 , 2017.
Yu Rong, Wenbing Huang, Tingyang Xu, and Junzhou Huang. Dropedge: Towards deep graph
convolutional networks on node classiﬁcation, 2020.
Emanuele Rossi, Fabrizio Frasca, Ben Chamberlain, Davide Eynard, Michael Bronstein, and Fed-
erico Monti. Sign: Scalable inception graph neural networks. arXiv preprint arXiv:2004.11198 ,
2020.
Alvaro Sanchez-Gonzalez, Jonathan Godwin, Tobias Pfaff, Rex Ying, Jure Leskovec, and Pe-
ter W Battaglia. Learning to simulate complex physics with graph networks. arXiv preprint
arXiv:2002.09405 , 2020.
Aliaksei Sandryhaila and Jos ´e MF Moura. Discrete signal processing on graphs. IEEE transactions
on signal processing , 61(7):1644–1656, 2013.
Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue:
Learning feature matching with graph neural networks. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition , pp. 4938–4947, 2020.
Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne Van Den Berg, Ivan Titov, and Max
Welling. Modeling relational data with graph convolutional networks. In European Semantic Web
Conference , pp. 593–607. Springer, 2018.
Vivienne Sze, Yu-Hsin Chen, Tien-Ju Yang, and Joel S Emer. Efﬁcient processing of deep neural
networks. Synthesis Lectures on Computer Architecture , 15(2):1–341, 2020.
Shyam A. Tailor, Javier Fernandez-Marques, and Nicholas D. Lane. Degree-Quant: Quantization-
Aware Training for Graph Neural Networks. In International Conference on Learning Represen-
tations , 2021.
Yi Tay, Mostafa Dehghani, Jai Gupta, Dara Bahri, Vamsi Aribandi, Zhen Qin, and Donald Metzler.
Are pre-trained convolutions better than pre-trained transformers?, 2021.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Lukasz Kaiser, and Illia Polosukhin. Attention is All you Need. In I. Guyon, U. V . Luxburg,
S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neu-
ral Information Processing Systems 30 , pp. 5998–6008. Curran Associates, Inc., 2017. URL
http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf .
Petar Veli ˇckovi ´c, Adri `a Puigdom `enech Badia, David Budden, Razvan Pascanu, Andrea Banino,
Misha Dashevskiy, Raia Hadsell, and Charles Blundell. The clrs algorithmic reasoning bench-
mark. 2021.
Petar Veli ˇckovi ´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li `o, and Yoshua
Bengio. Graph attention networks. In International Conference on Learning Representations ,
2018.
12

## Page 13

Published as a conference paper at ICLR 2022
Felix Wu, Angela Fan, Alexei Baevski, Yann N. Dauphin, and Michael Auli. Pay less attention with
lightweight and dynamic convolutions, 2019a.
Felix Wu, Tianyi Zhang, Amauri Holanda de Souza Jr. au2, Christopher Fifty, Tao Yu, and Kilian Q.
Weinberger. Simplifying graph convolutional networks, 2019b.
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How Powerful are Graph Neural
Networks? arXiv:1810.00826 [cs, stat] , February 2019. URL http://arxiv.org/abs/
1810.00826 . arXiv: 1810.00826.
Mutian Xu, Runyu Ding, Hengshuang Zhao, and Xiaojuan Qi. Paconv: Position adaptive convolu-
tion with dynamic kernel assembling on point clouds, 2021.
Mingyu Yan, Lei Deng, Xing Hu, Ling Liang, Yujing Feng, Xiaochun Ye, Zhimin Zhang, Dongrui
Fan, and Yuan Xie. HyGCN: A GCN Accelerator with Hybrid Architecture. In 2020 IEEE Inter-
national Symposium on High Performance Computer Architecture (HPCA) , pp. 15–29, February
2020. doi: 10.1109/HPCA47549.2020.00012. ISSN: 2378-203X.
Carl Yang, Aydin Buluc, and John D. Owens. Design principles for sparse matrix multiplication on
the gpu, 2018.
Haoran You, Tong Geng, Yongan Zhang, Ang Li, and Yingyan Lin. Gcod: Graph convolutional net-
work acceleration via dedicated algorithm and accelerator co-design. In 28th IEEE International
Symposium on High-Performance Computer Architecture (HPCA 2022) , 2021.
Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, Rajgopal Kannan, and Viktor Prasanna. Graph-
saint: Graph sampling based inductive learning method. arXiv preprint arXiv:1907.04931 , 2019.
Yongan Zhang, Haoran You, Yonggan Fu, Tong Geng, Ang Li, and Yingyan Lin. G-cos: Gnn-
accelerator co-search towards both better accuracy and efﬁciency, 2021.
Yiren Zhao, Duo Wang, Xitong Gao, Robert Mullins, Pietro Lio, and Mateja Jamnik. Probabilistic
dual network architecture search on graphs. arXiv preprint arXiv:2003.09676 , 2020.
13

## Page 14

Published as a conference paper at ICLR 2022
Method Propagation Rule Memory Notes
GCN (Kipf &
Welling, 2017)y(i)=P
j2N(i)[fig1p
deg(i)deg(j)x(j)O(V) Formally deﬁned for undirected graphs with self-loops; moti-
vated by graph signal processing.
GIN (Xu et al.,
2019)y(i)=f[(1 +)x(i)+P
j2N(i)x(j)]O(V)fis a learnable function, typically parameterized as an MLP or
linear layer; may be ﬁxed or learned.
GraphSAGE (Hamil-
ton et al., 2017)y(i)=1x(i)+L
j2N(i)2x(j)O(V)Ltypically parameterized as mean or max.
GAT (Veli ˇckovi ´c
et al., 2018)y(i)=i;ix(i)+P
j2N(i)i;jx(j)O(E) Attention coefﬁcients calculated using: i;j =
exp(LeakyReLU (a>[x(i)kx(j)]))
P
k2N(i)[figexp(LeakyReLU (a>[x(i)kx(k)])). Common
to deﬁne multiple attention heads and concatenate.
GATv2 (Brody
et al., 2022)y(i)=i;ix(i)+P
j2N(i)i;jx(j)O(E) Similar to GAT, but with attention calcula-
tion redeﬁned to improve expressivity i;j =
exp(a>LeakyReLU ([xikxj]))
P
k2N(i)[figexp(a>LeakyReLU( [xikxk])):
MPNN (Gilmer
et al., 2017)y(i)=U(x(i);L
j2N(i)M(x(i);x(j);eij))O(E)U;M typically deﬁned as linear layers acting on concatenated
features;Lmay be any valid aggregator, typically sum or max.
PNA (Corso
et al., 2020)y(i)=U(x(i);L
j2N(i)M(x(i);x(j);eij))O(E) Similar to MPNN, but withLdeﬁned to use 4 aggregators
(mean, standard deviation, max, and min) scaled by 3 different
functions of node degree, resulting in 12 different aggregations
by default.
Table 5: Propagation rules for general-purpose GNN architectures we compare against in this work;
rules are provided using node-wise formulations. We evaluate against popular architectures, and a
recent proposal that has achieved state-of-the-art performance, PNA.
A F URTHER EXPERIMENT DETAILS
A.1 E NSURING FAIRNESS
We expand on our experimental protocol, with a particular focus on describing measures that we
took to ensure that the results we report are not unfairly biased towards EGC.
For EGC-S, we use H= 8 andB= 4, as implied by our ablation for all experiments, with the
single exception of OGB-Code, where we use H=B= 8. The beneﬁt of using a smaller set of
bases is that we can increase the hidden dimension, but this is not viable in this case since most of
the 11M parameters in the model correspond to the token read-out layers, which quickly increases
as the model hidden dimension grows. As shown by section 5.3, if we cannot increase the hidden
dimension, it is better to increase the bases. This is the only exception we make.
For EGC-M, we use H=B= 4 for all experiments. The main challenge is aggregator selection,
and this remains a major challenge for our work. We were unable to ﬁnd a satisfactory technique
for automated discovery of aggregator choices, hence we rely on heuristics to ﬁnd them. We restrict
ourselves to use 3 aggregators for each model (yielding 35 possible choices). In order to determine
the aggregators, we use two heuristics: (1) aggregators should be “diverse” and (2) aggregators
should be chosen based on inductive bias for task. Using these two rules, we try up to 3 possible
choices of aggregators; all choices considered are shown in Table 6. We note that while some choices
do improve performance, our conclusions are not invalidated; it is also worth noting that it is likely
that better aggregator choices can be found.
A.2 C LUSTER DETAILS
Most of our experiments were run on several machines in our SLURM cluster using Intel CPUs and
NVIDIA GPUs. Each machine was running Ubuntu 18.04. The GPU models in our cluster were
RTX 2080Ti and GTX 1080Ti. High-memory experiments were run on V100s in our cluster and an
RTX 8000 virtual machine we had access to.
14

## Page 15

Published as a conference paper at ICLR 2022
Dataset Add Mean Symmetric Normalization Max Min Std Var Result
Zinc X X X 0.2810.007
Zinc X X X 0.2840.045
CIFAR X X X 71.030.42
CIFAR X X X 70.051.14
MolHIV X X X 78.191.54
MolHIV X X X 77.401.02
MolHIV X X X 77.981.65
Arxiv X X X 71.960.23
Arxiv X X X 70.590.67
Arxiv X X X 70.380.76
Code-V2 X X X 0.15950.0019
Code-V2 X X X 0.15720.0029
Code-V2 X X X 0.15780.0021
Table 6: Possible aggregators tried for EGC-M. Up to 3 combinations (from a possible 35) were
tried, as we limited ourselves to always using 3 aggregators. Our conclusions do not change, and it
is likely that better results can be found with more optimal aggregator choices.
Model ZINC (MAE #) CIFAR (Acc.") MolHIV (ROC-AUC ") Code-V2 (F1")
GA-MLP 0.5100.037 58.130.65 75.50 1.32 0.1485 0.0027
GCN 0.4590.006 55.710.38 76.14 1.29 0.1480 0.0018
EGC-S 0.3640.020 66.920.37 77.44 1.08 0.1528 0.0025
Table 7: Results of applying graph-augmented MLPs (GA-MLPs) to tasks requiring generalization
to unseen graphs. We see that the performance is broadly similar to the corresponding GCN model—
and far weaker than EGC-S, which uses the same aggregator. We also considered using the add
aggregation on ZINC, and achieved 0.444 0.019. By comparison, the equivalent GIN model
(which uses the add aggregation) achieved 0.387 0.015.
B L IMITATIONS OF EXISTING APPROACHES
As explained in the related work, existing approaches to improving GNN efﬁciency have severe
limitations. In this section we elaborate upon them, and provide experimental evidence where nec-
essary.
We ﬁrst examine graph-augmented MLPs (GA-MLPs); to our knowledge there are few experimental
results assessing the performance of these models when they are applied to problems requiring
generalization to unseen graphs. In the literature they are generally applied to large scale node
classiﬁcation benchmarks, such as those found in the OGB benchmark suite (Wu et al., 2019b;
Rossi et al., 2020). It is known that these models are theoretically less expressive than standard
GNNs, however their performance when applied to node classiﬁcation datasets has been acceptable.
We consider models of the form:
Y=Readout (MLP(4
k
k=0SkXW )) (5)
We use up to the 4th power of the diffusion operator Sto emulate the depth of the corresponding
GNNs, which all use 4 layers. We set Sto use symmetric normalization, as used by GCN and EGC-
S; we also consider setting Sto the adjacency matrix on ZINC, to emulate the operations used by
GIN. The results are provided in Table 7. We see that the GA-MLP models offer similar (but often
worse) performance than the corresponding GNN baselines. The models are not competitive with
approaches such as GAT or MPNN, and is outperformed by a wide margin by EGC-S. Achieving
15

## Page 16

Published as a conference paper at ICLR 2022
Experiment ZINC (MAE #) CIFAR (Acc.") MolHIV (ROC-AUC ") Code-V2 (F1")
EGC-S 0.3640.020 66.920.37 77.44 1.08 0.1528 0.0025
EGC-S + DropEdge ( p= 0:1) 0.4680.007 66.370.28 77.41 1.32 0.1553 0.0021
EGC-S + DropEdge ( p= 0:5) 0.6290.023 64.600.58 75.33 0.82 0.1527 0.0019
EGC-S + GraphSAINT Node
Sampler (p= 0:1)0.6310.012 61.370.75 73.65 1.41 0.1461 0.0027
Table 8: Results of applying sampling approaches to EGC-S. We do not enable sampling at test
time—hence these approaches do not offer any test time reductions to computation. Sampling, in
principle, is applicable to any underlying GNN: our conclusions will transfer to other underlying
GNNs. We see that DropEdge (Rong et al., 2020) with a low drop probability ( p= 0:1) can aid
model performance; however, setting pthis low does not signiﬁcantly reduce memory consumption
or computation. Increasing pto 0.5 does reduce resource consumption noticeably, but results in
noticeable degradation to model performance.
state-of-the-art performance with GA-MLPs does not appear to be possible, at least with our current
understanding of these models.
We now proceed to investigate sampling, in which each training step does not run over the entire
graph, but over some sampled subgraph. These methods have seen great popularity when applied to
tasks such as node classiﬁcation, and they are able to deliver sampling ratios in excess of 20, hence
yielding noticeable improvements to memory consumption (the primary limitation with large scale
training). However, we note that the beneﬁt of these methods has not been examined carefully for
many graph topologies; while they have been shown to be effective for many “small-world” graph
topologies—which arise in many graphs—it is not the case that all graphs fall into this category. For
example, molecule graphs would not ﬁt this category.
We ﬁrst assess the sampling strategy from GraphSAINT (Zeng et al., 2019) by applying it to EGC-S
models. For our experiments, we use the node-centric sampler. The results are presented in Table 8;
we disable sampling at test time. Even with a relatively low dropping probability of 10% (i.e. 90%
of nodes are retained), the model performance degradation is severe. We note that the computational
savings achieved are modest when using this drop probablility. We also observed similar results
when using the edge-centric sampler from GraphSAINT.
We also attempted a different approach for sampling proposed by DropEdge (Rong et al., 2020);
as before, we apply it to EGC-S models. In this simple scheme, elements of the adjacency matrix
are dropped; this scheme was shown to be effective for training deep graph networks, but it is also
useful for reducing the computational footprint of models, since it effectively reduces the number of
messages that have to be computed. We also provide results in Table 8. The results are signiﬁcantly
better than we observe when using GraphSAINT’s sampling strategies: at 10% drop probabilities
we even see an improvement in some cases, due to the regularization effect. However, once we
increase the drop probability to levels where we would observe a noticeable reduction in compu-
tational demand, we observe that model performance declines. In summary, while DropEdge is a
more effective strategy for sampling on many inductive tasks, it is not beneﬁcial as a method to
reduce the computational burden.
Finally, we discuss the limitations of other approaches proposed in the literature. Quantization is
an approach that is typically applied at inference time; mixed-precision approaches can be applied
at training time, however care must be taken for GNNs to avoid biasing the gradients (Tailor et al.,
2021). Additionally, while neural architecture search (NAS) may be useful to ﬁnd memory-efﬁcient
models if the search objective is set appropriately, they suffer from some limitations—primarily
search time and memory consumption. Finally, approaches such as reversible residuals (Li et al.,
2021) are useful to architecture design, they do not tackle issues such as high peak memory usage
induced by the message passing step.
C A PPROACHES FOR HARDWARE ACCELERATION OF GNN S
The reader should note that most existing work for GNN hardware acceleration focuses on support-
ing only a subset of GNNs: speciﬁcally, they tend to only support models that can be implemented
16

## Page 17

Published as a conference paper at ICLR 2022
using SpMM. Approaches in the literature falling into this area include Chen et al. (2020a), Yan
et al. (2020), You et al. (2021) and Zhang et al. (2021). It is possible to add greater ﬂexibility to
the accelerator to support more expressive message passing schemes, however this necessarily im-
plies greater complexity. As Amdahl’s law (Amdahl, 1967) implies, increasing ﬂexibility is likely
to reduce peak performance, while increasing silicon area requirements. Therefore, aiming for the
simplest primitive (as we do with EGC) is the most sensible approach to obtain hardware accelera-
tion.
D A GGREGATOR FUSION
Algorithm 1 Aggregator Fusion with aggregators A. This method is a modiﬁcation of the Com-
pressed Sparse Row (CSR) SpMM algorithm, where we maximize re-use of matrix B. Maximizing
re-use enables us to obtain signiﬁcantly better accuracy with minimal impact on memory and la-
tency. For simplicity, pseudocode assumes H=B= 1. This version demonstrates how we can
remove memory overheads at inference time.
Input: CSRA2RNN, Dense B2RNF, Combination weightings w2RNjAj
Output: Dense C2RNF
fori= 0toA:rows 1do
forjj=A:row pointer [i]toA:row pointer [i+ 1] do
j=A:column index [jj]
Init temp arrays of length Fper aggregator
aij=A:values [jj]
fMay be faster to interleave these calls: g
for2A do
process row(aij;B[i;:];temp)
end for
end for
fCan be generalized to H;B > 1:g
C[i;:] =P
2Aw[i;]temp[:]
end for
The naive approach of performing each aggregation sequentially would cause a linear increase in
latency with respect to jAj. However, a key observation to note is that we are memory-bound : the
bottleneck with sparse operations is waiting for the data to arrive from memory. This observation
applies to both GPUs and CPUs, and justiﬁed through proﬁling. Using a proﬁler on a GTX 1080Ti
we observed that SpMM using the Reddit graph Hamilton et al. (2017) with feature sizes of 256
achieved just 1.2% of the GPU’s peak FLOPS, with 88.5% of stalls being caused by unmet memory
dependencies. The fastest processing order performs as much work as possible with data that has
already been fetched from memory, rather than fetching it multiple times. This concept is illustrated
in Algorithm 1 in the appendix. We can perform all aggregations as a lightweight modiﬁcation to
the standard compressed sparse row (CSR) SpMM algorithm.
The second observation we make is that storing the results of all aggregations is unnecessary at in-
ference time. Note that the CSR SpMM algorithm processes each row in the output matrix sequen-
tially: rather than storing the aggregations for every row, instead we should store only the weighted
results. This approach not only reduces memory consumption, but also latency as we improve the
effectiveness of our cache and reduce memory system contention. In practice, this optimization is
especially important when performing inference on topologies which have more frequent periods
where the processing has become compute-bound, since we reduce contention between load and
store units Corporation. This is also demonstrated in Algorithm 1.
We evaluated aggregator fusion across four different topologies, on both CPU and GPU; our results
can be found in Table 9. We assumed all operations are 32-bit ﬂoating point, and that we were
using three aggregators: summation-based, max, and min; these aggregators match those used for
EGC-M Code. Our benchmarks were conducted on a batch of 10k graphs from the ZINC and Code
datasets, Arxiv, and the popular Reddit dataset (Hamilton et al., 2017), which is one of the largest
graph datasets commonly evaluated on in the GNN literature. Our SpMM implementation on GPU
is based on Yang et al. (2018). Code for the kernels are provided in our repo.
17

## Page 18

Published as a conference paper at ICLR 2022
CPU (Xeon Gold 5218) GPU (RTX 8000)
Method Reddit / s Code / s Arxiv / s ZINC / s Reddit / ms Code / ms Arxiv / ms ZINC / ms
Weight Matmul 0.070.00 0.4230.023 0.0550.013 0.0740.010 2.360.00 13.660.12 1.740.01 2.360.02
CSR SpMM 29.080.12 1.9430.040 0.7600.021 0.3150.006 186.440.05 19.880.10 5.560.02 3.390.01
Naive Fusion 88.250.20 8.6800.094 2.6310.016 1.4820.010 595.910.13 112.520.22 23.810.06 19.090.05
+Faster Ordering 40.050.10 4.5920.054 1.3030.037 0.7720.008 214.600.10 29.380.12 7.630.03 4.780.02
+Store Weighted Result Only 38.630.22 1.7520.018 0.9520.026 0.2780.003 208.220.13 26.640.09 6.750.03 4.380.02
Table 9: Inference latency (mean and standard deviation) for CSR SpMM, used by GCN/GIN,
and aggregator fusion. Assuming a feature dimension of 256 and H=B= 1 per Algorithm 1.
We observe that aggregator fusion results in an increase of 34% in the worse case; in contrast,
the naive implementation has a worst case increase of 466%. Also included are timings for dense
multiplication with a square weight matrix; we observe that sparse operations dominate latency
measurements.
Model Train Epoch Time (s) Test Epoch Time (s) Peak Train Memory (MB)
GCN 144.70.5 6.3 0.3 1337
GIN 134.50.1 5.7 0.3 1331
GraphSAGE 140.30.3 5.6 0.3 1226
GAT 176.00.7 6.3 0.3 2885
MPNN-Sum 305.32.7 6.8 0.3 3448
MPNN-Max 319.01.3 7.6 0.4 3901
PNA 575.42.3 12.0 0.2 9399
EGC-S 166.42.7 6.2 0.3 1842
EGC-M 225.70.6 7.9 0.3 3470
Table 10: Latency and memory results for parameter-normalized models on OGB Code-V2. Despite
having a lowerjEj
jVjratio of 2.75 relative to Arxiv (13.67), we see that the trends we observed in for
Arxiv broadly remain. It is worth noting again that EGC-M is far more efﬁcient both latency and
memory-wise than PNA.
As expected, our technique optimizing for input re-use achieves signiﬁcantly lower inference latency
than the naive approach to applying multiple aggregators. While the naive approach results in a mean
increase in latency of 331%, our approach incurs a mean increase of only 14% relative to ordinary
SpMM, used by GCN and GIN. The increase is topology dependent, with larger increases in latency
being observed for topologies which are less memory-bound. We also provide timings for dense
matrix multiplication (i.e. X) to justify our focus on optimizing sparse operations in this work:
CSR SpMM operation is 4.7slower (geomean) than the corresponding weight multiplication. We
believe further optimizations of the operations used by architecture are achievable through the use of
auto-tuning frameworks e.g. TVM (Chen et al., 2018b), but this lies beyond the scope of this work.
E L ATENCY AND MEMORY CONSUMPTION ON OTHER DATASETS
In the evaluation, we assessed the memory consumption and latency for the parameter-normalized
models on Arxiv. In this section, we consider a similar exercise for OGB Code models. The results
are provided in Table 10. Our conclusions remain broadly similar, with EGC-M offering clear
improvements to memory consumption, latency, and parameter efﬁciency relative to PNA. EGC-S
is superior to GAT, with similar inference latency, better model performance, and noticeably lower
memory consumption. We train models using batch size 128; the reader should note that the memory
consumption ﬁgures can vary between runs (even for the same model), since graphs vary in the
number of nodes and edges.
So far we have not demonstrated that our approach is more efﬁcient than the baselines on Arxiv,
which is the only dataset where EGC-M and PNA are not the best performing. To demonstrate that
we are more efﬁcient we must show that we achieve lower memory consumption and latency for
a given accuracy-level—i.e. we must consider models that are accuracy-normalized . We evaluate
increasing the parameter count for baseline models until they achieve the same accuracy as EGC-S;
18

## Page 19

Published as a conference paper at ICLR 2022
Accuracy-Normalized
ModelParameters GPU Training
Latency (ms)GPU Inference
Latency (ms)Peak Memory
(MB)
GCN 184k 208.6 2.2 44.9 0.4 2549
GIN FAIL FAIL FAIL FAIL
GraphSAGE 593k 335.8 2.4 60.2 0.2 3208
EGC-S 100k 177.7 2.2 37.3 0.1 2430
Table 11: Latency and memory statistics for accuracy-normalized models on Arxiv. To achieve the
same accuracy as EGC-S, we must boost the size of the baseline models. We observe that EGC-S
offers noticeable reductions to both memory consumption and latency for a given accuracy level.
we also gave these models an extra advantage over our method by increasing the hyperparameter
search budget. The results are shown in Table 11, where we observe that EGC-S is more efﬁcient
once we are comparing models achieving the same accuracy.
The reader should note that GAT (but not GATv2) can be implemented to reduce memory consump-
tion by noting that the left and right halves of the attention vector can be computed separately, and
added together as appropriate. We use an optimized implementation of GAT for our experiments
(from PyTorch Geometric); we refer the reader to the implementation for further details.
F G ENERALIZING TO HETEROGENEOUS GRAPHS
Our R-EGC model is similar to the baseline R-GCN model included in the OGB repository. The
OGB model deviates from the standard deﬁnition of R-GCN (Schlichtkrull et al., 2018) since it han-
dles different node types, not just edge types. The baseline model has weights to generate messages
for each relation-type, and a weight matrix to update each individual node type. This corresponds
to:
y(i)=x(i)+X
r2R1
jNr(i)jX
j2Nr(i)rx(j)(6)
wherecorresponds to the node type of nodei, andRrepresents the set of relation types. Note that
the mean aggregator is used.
We deviate from the baseline by using a single set of basis weights. Instead, we use a different
weighting calculation layers ( w(i)=x+b) per node and relation type.
y(i)=H
k
h=1BX
b=1w(i)
;h;bbx(j)+X
r2R1
jNr(i)jH
k
h=1BX
b=1w(i)
r;h;bX
j2N(i)bx(j)(7)
19