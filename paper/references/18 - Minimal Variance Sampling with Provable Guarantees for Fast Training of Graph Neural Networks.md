# Minimal Variance Sampling with Provable Guarantees for Fast Training of Graph Neural Networks

## Page 1

Minimal Variance Sampling with Provable Guarantees for Fast
Training of Graph Neural Networks
Weilin Cong
The Pennsylvania State University
wxc272@psu.eduRana Forsati
Microsoft Bing
raforsat@microsoft.com
Mahmut Kandemir
The Pennsylvania State University
mtk2@psu.eduMehrdad Mahdavi
The Pennsylvania State University
mzm616@psu.edu
ABSTRACT
Sampling methods (e.g., node-wise, layer-wise, or subgraph) has
become an indispensable strategy to speed up training large-scale
Graph Neural Networks (GNNs). However, existing sampling meth-
ods are mostly based on the graph structural information and ignore
the dynamicity of optimization, which leads to high variance in
estimating the stochastic gradients. The high variance issue can be
very pronounced in extremely large graphs, where it results in slow
convergence and poor generalization. In this paper, we theoretically
analyze the variance of sampling methods and show that, due to
the composite structure of empirical risk, the variance of any sam-
pling method can be decomposed into embedding approximation
variance in the forward stage and stochastic gradient variance in the
backward stage that necessities mitigating both types of variance to
obtain faster convergence rate. We propose a decoupled variance re-
duction strategy that employs (approximate) gradient information
to adaptively sample nodes with minimal variance, and explicitly
reduces the variance introduced by embedding approximation. We
show theoretically and empirically that the proposed method, even
with smaller mini-batch sizes, enjoys a faster convergence rate and
entails a better generalization compared to the existing methods.
Code is public available at here.1
CCS CONCEPTS
•Computing methodologies →Machine learning ;Learning
latent representations .
KEYWORDS
Graph neural networks, minimal variance sampling
ACM Reference Format:
Weilin Cong, Rana Forsati, Mahmut Kandemir, and Mehrdad Mahdavi. 2020.
Minimal Variance Sampling with Provable Guarantees for Fast Training of
Graph Neural Networks. In Proceedings of the 26th ACM SIGKDD Conference
1Please notice that we fixed a typo of our objective function defined in Eq. 5 on
09/05/2021 .
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than ACM
must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,
to post on servers or to redistribute to lists, requires prior specific permission and/or a
fee. Request permissions from permissions@acm.org.
KDD ’20, August 23–27, 2020, Virtual Event, CA, USA
©2020 Association for Computing Machinery.
ACM ISBN 978-1-4503-7998-4/20/08. . . $15.00
https://doi.org/10.1145/3394486.3403192on Knowledge Discovery and Data Mining (KDD ’20), August 23–27, 2020,
Virtual Event, CA, USA. ACM, New York, NY, USA, 12 pages. https://doi.org/
10.1145/3394486.3403192
1 INTRODUCTION
Graph Neural Networks (GNNs) are powerful models for learn-
ing representation of nodes and have achieved great success in
dealing with graph-related applications using data that contains
rich relational information among objects, including social network
prediction [ 8,11,13,20,24], traffic prediction [ 7,14,15,21], knowl-
edge graphs [ 18,25,26], drug reaction [ 9,10] and recommendation
system [2, 27].
Despite the potential of GNNs, training GNNs on large-scale
graphs remains a big challenge, mainly due to the inter-dependency
of nodes in a graph. In particular, in GNNs, the representation (em-
bedding) of a node is obtained by gathering the embeddings of its
neighbors from the previous layers. Unlike other neural networks
that the final output and gradient can be perfectly decomposed over
individual data samples, in GNNs, the embedding of a given node
depends recursively on all its neighbor’s embedding, and such de-
pendency grows exponentially with respect to the number of layers,
a phenomenon known as neighbor explosion , which prevents their
application to large-scale graphs. To alleviate the computational
burden of training GNNs, mini-batch sampling methods, including
node-wise sampling [ 11,27], layer-wise sampling [ 3,16,33], and
subgraph sampling [ 5,28], have been proposed that only aggregate
the embeddings of a sampled subset of neighbors of each node in
the mini-batch at every layer.
Although empirical results show that the aforementioned sam-
pling methods can scale GNN training to a large graph, these meth-
ods incur a high variance that deteriorates the convergence rate
and leads to a poor generalization. To reduce the variance of sam-
pling methods, we could either increase the mini-batch size per
layer or employ adaptive sampling methods (gradient information
or representations) to reduce the variance. The computation and
memory requirements are two key barriers to increase the number
of sampled nodes per layer in a sampled mini-batch.
In importance sampling or adaptive sampling methods, the key
idea is to utilize the gradient information which changes during
optimization to sample training examples (e.g., nodes in GNNs) to
effectively reduce the variance in unbiased stochastic gradients. Re-
cently, different adaptive sampling methods are proposed in the lit-
erature to speed up vanilla Stochastic Gradient Descent (SGD), e.g.,
importance sampling [ 31], adaptive importance sampling [ 6,17],arXiv:2006.13866v2  [cs.LG]  5 Sep 2021

## Page 2

Figure 1: Comparing full-batch GNNs versus sampling based
GNNs. The sampling based GNNs incurs two types of vari-
ance: embedding approximation variance and stochastic gra-
dient variance.
gradient-based sampling [ 17,30,32], safe adaptive sampling [ 23],
bandit sampling [ 22], and determinantal point processes based sam-
pling [ 29]– to name a few. Although adaptive sampling methods
have achieved promising results for training neural networks via
SGD, the generalization of these methods to GNNs is not straightfor-
ward. As we will elaborate later, the key difficulty is the multi-level
composite structure of the training loss in GNNs, where unlike stan-
dard empirical risk minimization, any sampling idea to overcome
neighbor explosion introduces a significant bias due to estimating
embedding of nodes in different layers, which makes it difficult to
accurately estimate the optimal sampling distribution.
The overarching goal of this paper is to develop a novel decou-
pled variance reduction schema that significantly reduces the
variance of sampling based methods in training GNNs, and enjoys
the beneficial properties of adaptive importance sampling meth-
ods in standard SGD. The motivation behind the proposed schema
stems from our theoretical analysis of the variance of the sampled
nodes. Specifically, we show that due to the composite structure of
the training objective, the stochastic gradient is a biased estimation
of the full-batch gradient that can be decomposed into two types
of variance: embedding approximation variance and stochastic gra-
dient variance. As shown in Figure 1, embedding approximation
variance exists because a subset of neighbors are sampled in each
layer to estimate the exact node embedding matrix, while stochastic
gradient variance exists because a mini-batch is used to estimate
the full-batch gradient (similar to vanilla SGD). Besides, the bias of
the stochastic gradient is proportional to the embedding approxi-
mation variance, and the stochastic gradient becomes unbiased as
embedding approximation variance reduces to zero.
The proposed minimal variance sampling schema, dubbed as
MVS-GNN , employs the dynamic information during optimization to
sample nodes and composes of two key ingredients: (i) explicit em-
bedding variance reduction by utilizing the history of embeddings
of nodes, (ii) gradient-based minimal variance sampling by utiliz-
ing the (approximate) norm of the gradient of nodes and solving
an optimization problem. The proposed schema can be efficiently
computed and is always better than uniform sampling or static im-
portance sampling, as we demonstrate theoretically. We empirically
compare MVS-GNN through various experimental results on differ-
ent large-scale real graph datasets and different sampling methods,
where MVS-GNN enjoys a faster convergence speed by significantlyreducing the variance of stochastic gradients even when signifi-
cantly smaller mini-batches are employed. Our empirical studies
also corroborates the efficiency of proposed algorithm to achieve
better accuracy compared to competitive methods.
Organization. The remainder of this paper is organized as follows.
In Section 2, we review related literature on different sampling
methods to train GNNs. In Section 3, we provide the analysis of
variance of the structural based sampling methods. In Section 4,
we propose a decoupled variance reduction algorithm and analyze
its variance. Finally, we empirically verify the proposed schema in
Section 5 and conclude the paper in Section 6.
2 ADDITIONAL RELATED WORK
A key idea to alleviate the neighbor explosion issue in GNNs is to
sample a mini-batch of nodes and a subset of their neighbors at
each layer to compute the stochastic gradient at each iteration of
SGD. Recently, different sampling strategies with the aim of re-
ducing variance are proposed. For instance, node-wise sampling is
utilized in GraphSage [11] to restrict the computation complexity
by uniformly sampling a subset of nodes from the previous layer’s
neighbors. However, the variance of nodes’ embedding might be
significantly large if the number of sampled neighbors is small.
VRGCN [4] further restricted the neighborhood size by requiring
only two support nodes in the previous layer, and used the histor-
ical activation of the previous layer to reduce variance. Though
successfully achieved comparable convergence as GraphSage , the
computation complexity is high as additional graph convolution op-
erations are performed on historical activation to reduce variance.
More importantly, node-wise sampling methods require sample
nodes recursively for each node and each layer, which results in a
significant large sample complexity.
Instead of performing node-wise sampling, layer-wise sampling
methods, such as FastGCN [3], independently sample nodes using
importance sampling, which results in a constant number of nodes
with low variance in all layers. However, since the sampling oper-
ation is conduced independently at each layer, it requires a large
sample size to guarantee the connectivity between the sampled
nodes at different layers. LADIES [ 33] further improve the sample
density and reduce the sample size by restricting the candidate
nodes in the union of the neighborhoods of the sampled nodes in
the upper layer. However, they need to track the neighbors of nodes
in the previous layer and calculate a new importance sampling dis-
tribution for each layer.
Another direction of research uses subgraph sampling. For
instance, ClusterGCN [5] proposed to first partition graph into
densely connected clusters during pre-processing, then construct
mini-batches by randomly selecting subset of clusters during train-
ing. However, its performance is significantly sensitive to the clus-
ter size, and performing graph partition of a large graph is time-
consuming. GraphSaint [28] proposed to construct mini-batches
by importance sampling, and apply normalization techniques to
eliminate bias and reduce variance. However, since the sampling
operation is conducted independently for each node, it cannot guar-
antee the connectivity between nodes in the sampled subgraph,
which incurs a large variance due to the approximate embedding.

## Page 3

3 PROBLEM STATEMENT
In this section, we formally define the problem and present a math-
ematical derivation of the variance of sampling strategies.
3.1 Problem definition
Suppose we are given a graph G(V,E)of𝑁=|V|nodes and
|E|edges as input, where each node is associated with a feature
vector and label(𝒙𝑖,𝑦𝑖). LetX=[𝒙1,..., 𝒙𝑁]and𝒚=[𝑦1,...,𝑦𝑁]
denote the feature matrix and labels for all 𝑁nodes, respectively.
Given a𝐿-layer GNN, the ℓth graph convolution layer is defined
asH(ℓ)=𝜎(LH(ℓ−1)W(ℓ)) ∈R𝑁×𝐹, where Lis the normalized
Laplacian matrix, 𝐹is embedding dimension which we assume
is the same for all layers for ease of exposition, and 𝜎(·)is the
activation function (e.g., ReLU). Letting A∈ {0,1}𝑁×𝑁andD
be the adjacency matrix and diagonal degree matrix associated
withG, the normalized Laplacian matrix Lis calculated as L=
D−1/2AD−1/2orL=D−1A. To illustrate the key ideas we focus on
the semi-supervised node classification problem, where the goal is
to learn a set of per-layer weight matrices 𝜽={W(1),...,W(𝐿)}
by minimizing the empirical loss over all nodes
L(𝜽)=1
𝑁∑︁
𝑖∈V𝜙(H(𝐿)
𝑖,𝑦𝑖), (1)
where𝜙(·)stands for the loss function (e.g., cross entropy loss)
andH(𝐿)
𝑖is the node embedding of the 𝑖th node at the final layer
computed by
H(𝐿)=𝜎
L𝜎 ...𝜎(LXW(1))|        {z        }
H(1)...W(𝐿)
.
withH(0)=Xis set to be the input for the first layer. To efficiently
solve the optimization problem in Eq. 1 using mini-batch SGD, in
the standard sampling based methods, instead of computing the
full-gradient, we only calculate an unbiased gradient based on a
mini-batchVBof nodes with size 𝐵to update the model,
g=1
𝐵∑︁
𝑖∈VB∇𝜙(H(𝐿)
𝑖,𝑦𝑖). (2)
However, computing the gradient in Eq. 2 requires the embedding
of all adjacent neighbors in the previous layers which exponentially
grows by the number of layers. A remedy is to sample a subset of
nodes at each layer to construct a sparser Laplacian matrix from Lto
estimate the node embedding matrices ˜H(ℓ)forℓ=1,2,...,𝐿 , that
results in a much lower computational and memory complexities
for training.
In node-wise sampling (e.g., GraphSage ,VRGCN ), the main idea
is to first sample all the nodes needed for the computation using
neighbor sampling (NS), and then update the parameters. Specifi-
cally, for each node in the ℓth layer, NS randomly samples 𝑠of its
neighbors at(ℓ−1)th layer and formulate ˜L(ℓ)by
˜𝐿(ℓ)
𝑖,𝑗=(|N(𝑖)|
𝑠×𝐿𝑖,𝑗,if𝑗∈bN(ℓ)(𝑖)
0, otherwise, (3)
whereN(𝑖)is full set of the 𝑖th node neighbor, bN(ℓ)(𝑖)is the sam-
pled neighbors of node 𝑖forℓth GNN layer.In layer-wise sampling (e.g., FastGCN ,LADIES ), the main idea
is to control the size of sampled neighborhoods in each layer. For
theℓth layer, layer-wise sampling methods sample a set of nodes
Vℓ⊆ V of size𝑠under a distribution 𝒑∈R|V|
+,Í
𝑖𝑝𝑖=1to
approximate the Laplacian by
˜𝐿(ℓ)
𝑖,𝑗=(1
𝑠×𝑝𝑗×𝐿𝑖,𝑗,if𝑗∈Vℓ
0, otherwise(4)
Subgraph sampling (e.g., GraphSaint ,ClusterGCN ) is similar to
layer-wise sampling by restricting ˜L(1)=˜L(2)=...=˜L(𝐿).
3.2 Variance analysis
While being computationally appealing, the key issue that sampling
methods suffer from is the additional bias introduced to the sto-
chastic gradients due to the approximation of node embeddings at
different layers. To concretely understand this bias, let us formulate
a𝐿-layer sampling based GNN as a multi-level composite stochastic
optimization problem of the following form
min𝑓(𝜽):=E𝜔𝐿h
𝑓(𝐿)
𝜔𝐿
E𝜔𝐿−1
𝑓(𝐿−1)
𝜔𝐿−1 ...E𝜔1[𝑓(1)
𝜔1(𝜽)]...i
,
(5)
where the random variables 𝜔ℓcapture the stochasticity due
to sampling of nodes at the ℓth layer, i.e., the deterministic
function at ℓth layer𝑓(ℓ)(𝜽):=𝜎(LH(ℓ−1)W(ℓ))and its sto-
chastic variant 𝑓(ℓ)
𝜔ℓ(𝜽):=𝜎(˜L(ℓ)˜H(ℓ−1)W(ℓ))induced by 𝜔ℓ.
We denote the deterministic composite function at ℓth layer
by𝐹(ℓ)(·):=𝑓(ℓ)◦𝑓(ℓ−1)◦...◦𝑓(1)(·). By the chain rule,
the full gradient can be computed as ∇𝑓(𝜽)=∇𝑓(1)(𝜽) ·
∇𝑓(2)(𝐹(1)(𝜽))...∇𝑓(𝐿)(𝐹(𝐿−1)(𝜽)). For a given sample path
(𝜔1,...,𝜔𝐿), one may formulate an unbiased estimate of ∇𝑓(𝜽)as
g=∇𝑓(1)
𝜔1(𝜽)·∇𝑓(2)
𝜔2(𝐹(1)(𝜽))...∇𝑓(𝐿)
𝜔𝐿(𝐹(𝐿−1)(𝜽)), which can-
not be calculated because 𝐹(ℓ)(𝜽)=𝑓(ℓ)◦𝑓(ℓ−1)◦...◦𝑓(1)(𝜽)
forℓ≥2are unfortunately not known. In other words, the sto-
chastic gradient ˜gis a biased estimation of ∇𝑓(𝜽), where ˜g:=
∇𝑓(1)
𝜔1(𝜽)∇𝑓(2)
𝜔2(𝑓(1)
𝜔1(𝜽))...∇𝑓(𝐿)
𝜔𝐿(𝑓(𝐿−1)
𝜔𝐿−1◦...◦𝑓(1)
𝜔1(𝜽)). We note
that this is in contrast to the standard SGD where the gradient can
be decomposed over training examples; thereby, the average gra-
dient computed at a mini-batch is an unbiased estimator of full
gradient. To outline the role of bias and variance in the stochastic
gradients of training GNNs, we note that in vanilla SGD for em-
pirical risk minimization, we assume the variance of the unbiased
stochastic gradients gare bounded, i.e., E[∥g−∇𝑓(𝜽)∥2], but in
GNNs due to sampling at inner layers, this no longer holds. In fact,
the noise of stochastic gradient estimator ˜g, can be decomposed as
E[∥˜g−∇𝑓(𝜽)∥2]=E[∥˜g−g∥2]
bias(V)+E[∥g−∇𝑓(𝜽)∥2]
variance(G),
where bias is due to the inner layers embedding approximation in
forward pass, and the variance corresponds to the standard vari-
ance due to mini-batch sampling. We make the following standard
assumption on the Lipschitz continuity of functions 𝑓(ℓ)(·).
Assumption 1. For eachℓ=1,...,𝐿 and each realization of 𝜔ℓ,
the mapping 𝑓(ℓ)
𝜔ℓ(·)is𝜌ℓ-Lipschitz and its gradient ∇𝑓(ℓ)
𝜔ℓ(·)is𝐺ℓ-
Lipschitz.

## Page 4

Table 1: Summary of function approximation variance. Here 𝐷denotes the average node degree, 𝑠denotes the neighbor sam-
pling size,𝑁ℓdenotes the size of nodes sampled in ℓth layer,𝛾ℓdenotes the upper-bound of ∥H(ℓ−1)
𝑖W(ℓ)∥2, and Δ𝛾ℓdenotes the
upper-bound of∥(H(ℓ−1)
𝑖−¯H(ℓ−1)
𝑖)W(ℓ)∥2for any𝑖∈V. We useO(·) to hide constants that remain the same between different
algorithms.
Method GraphSage VRGCN LADIES GraphSaint MVS-GNN
VarianceO(𝐷𝛾2
ℓ/𝑠)O(𝐷Δ𝛾2
ℓ/𝑠)O(𝑁𝛾2
ℓ/𝑁ℓ)O(𝑁2𝛾2
ℓ/𝑁2
ℓ)O(𝐷Δ𝛾2
ℓ)
The following lemma shows that the bias of stochastic gradient
can be decomposed as a combination of embedding approximation
variance of different layers.
Lemma 3.1. LetVℓ:=E[∥𝑓(ℓ)
𝜔ℓ(𝐹(ℓ−1)(𝜽))−𝐹(ℓ)(𝜽)∥2]be the
per-layer embedding approximation variance. Suppose Assumption 1
holds. Then, the bias of stochastic gradient E[∥g−˜g∥2]can be bounded
as:
E[∥g−˜g∥2]≤𝐿·𝐿∑︁
ℓ=2ℓ−1Î
𝑖=1𝜌2
𝑖 𝐿Î
𝑖=ℓ+1𝜌2
𝑖
𝐺2
ℓ·ℓℓ∑︁
𝑖=1 
ℓÎ
𝑗=𝑖+1𝜌2
𝑗V𝑗!
.
Proof. Proof is deferred to Appendix A. □
From decomposition of variance and Lemma 3.1, we conclude
that any sampling method introduces two types of variance, i.e., em-
bedding approximation variance Vand stochastic gradient variance
G, that controls the degree of biasedness of stochastic gradients.
Therefore, any sampling strategy needs to take into account both
kinds of variance to speed up the convergence. Indeed, this is one of
the key hurdles in applying adaptive importance sampling methods
such as bandit sampling or gradient based importance sampling
to sampling based GNN training – originally developed for vanilla
SGD, as accurate estimation of gradients is crucial to reduce the
variance, which is directly affected by variance in approximating
the embedding matrices at different layers.
Remark 1. We emphasize that the aforementioned sampling meth-
ods are solely based on the Laplacian matrix and fail to explicitly
leverage the dynamic information during training to further reduce
the variance. However, from Lemma 3.1, we know that the bias of
stochastic gradient can be controlled by applying explicit variance
reduction to function approximation variance V, which motivates us
developing a decoupled variance reduction algorithm to reduce the
both types of variance.
4 ADAPTIVE MINIMAL VARIANCE
SAMPLING
Motivated by the variance analysis in the previous section, we
now present a decoupled variance reduction algorithm, MVS-GNN ,
that effectively reduces the variance in training GNNs using an
adaptive importance sampling strategy by leveraging gradient and
embedding information during optimization. To sample the nodes,
we propose a minimal variance sampling strategy based on the
estimated norm of gradients. To reduce the effect of embedding
approximation variance in estimating the gradients, we explicitly
reduce it at each layer using the history of embeddings of nodes in
the previous layer.4.1 Decoupled variance reduction
The detailed steps of the proposed algorithm are summarized in
Algorithm 1. To effectively reduce both types of variance, we pro-
pose an algorithm with two nested loops. In the outer-loop, at each
iteration𝑡=1,2,...,𝑇 we sample a large mini-batch VSof size
𝑆=𝑁×𝛾uniformly at random, where 𝛾∈(0,1]is the sampling
ratio, to estimate the gradients and embeddings of nodes. The outer-
loop can be considered as a checkpoint to refresh the estimates
as optimization proceeds, where 𝛾controls the accuracy of esti-
mations at the checkpoint. Specifically, at every checkpoint, we
calculate the per sample gradient norm as ¯g=[¯𝑔1,..., ¯𝑔𝑆]and save
it to memory for further calculation of the importance sampling
distribution.
Meanwhile, we also compute the node embedding for each node
inVS. To do so, we construct {˜L(ℓ)}𝐿
ℓ=1that only contains nodes
needed for calculating embeddings of nodes in VS, without node-
wise or layer-wise node sampling. Then, we calculate the node
embedding ˜H(ℓ)and update its history embedding ¯H(ℓ)as
˜H(ℓ)
𝑖=𝜎©­
«∑︁
𝑗∈V˜𝐿(ℓ)
𝑖,𝑗˜H(ℓ−1)
𝑖W(ℓ)ª®
¬,¯H(ℓ)
𝑖=˜H(ℓ)
𝑖. (6)
Every iteration of outer-loop is followed by 𝐾iterations of the
inner-loop, where at each iteration 𝑘=2,...,𝐾 , we sample a small
mini-batchVB⊂VSof size𝐵, and prepare the Laplacian matrix
of each layer{˜L(ℓ)}𝐿
ℓ=1to estimate the embeddings for nodes in
VBand update the parameters of GNN. Our key idea of reducing
the variance of embeddings is to use the history embeddings of
nodes in the previous layer ¯H(ℓ−1)as a feasible approximation to
estimate the node embeddings in the current layer ˜H(ℓ). Each time
when ˜H(ℓ)
𝑖is computed, we update ¯H(ℓ)
𝑖with ˜H(ℓ)
𝑖:
˜H(ℓ)
𝑖=𝜎©­
«∑︁
𝑗∈Vℓ−1˜𝐿(ℓ)
𝑖,𝑗˜H(ℓ−1)
𝑖W(ℓ)+∑︁
𝑗∈V\Vℓ−1𝐿𝑖,𝑗¯H(ℓ−1)
𝑖W(ℓ)ª®
¬,
¯H(ℓ)
𝑖=˜H(ℓ)
𝑖
(7)
The sampling of nodes in VBis based on a novel gradient-based
minimal variance strategy to compute the to optimal sampling dis-
tribution 𝒑that will be detailed later. After updating the parameters,
we use the freshly computed gradient and embedding of nodes in
VBto update the stale information. We note that as the gradient
of objective vanishes when we approach the optimal solution, we
can use larger 𝐾in later steps to reduce the number of checkpoints.
Besides, we only need to maintain the norm of the gradient for
nodes which requires only an additional 𝑂(𝑁×𝛾)memory which
is negligible (e.g, we set 𝛾=0.02for the Yelp dataset).

## Page 5

Variance analysis and time complexity. We summarized the
embedding approximation variance of different sampling based
GNN training methods in Table 1. We provide a detailed analysis of
the embedding approximation variance of MVS-GNN in Appendix B.
Comparing with GraphSage ,LADIES , and GraphSaint ,MVS-GNN
enjoys a much smaller variance because ∥(H(ℓ−1)
𝑖−¯H(ℓ−1)
𝑖)W(ℓ)∥2
is usually much smaller than ∥H(ℓ−1)
𝑖W(ℓ)∥2. On the other hand,
although the embedding approximation variance of VRGCN is𝑠times
smaller than MVS-GNN , since full-batch GNN are performed once a
while, the staleness of {H(ℓ)}𝐿
ℓ=1can be well controlled, which is
not true in VRGCN .
Remark 2. Since both MVS-GNN andVRGCN utilize explicit variance
reduction on estimating the embedding matrix, here we emphasize
the key differences:
•MVS-GNN is one-shot sampling, i.e., it only needs to sample one
time to construct a mini-batch, while VRGCN requires samplers
to explore recursively for each layer and each node in the mini-
batch. Notice that the sample complexity can be much higher
than computation complexity when the graph is large.
•MVS-GNN requires a constant number of nodes at each layer, de-
spite the fact the dependency grows exponentially with respect
to the number of layers.
•MVS-GNN requires to multiply adjacency matrix with embed-
ding matrix one time for each forward propagation, while
VRGCN requires twice. Therefore, the computation cost of our
algorithm is relatively lower, especially when the number of
layers is large.
4.2 Gradient-based minimal variance sampling
Here we propose a minimal variance sampling strategy to reduce
the stochastic gradient variance where nodes with larger gradient
are chosen with higher probability than ones with smaller gradi-
ent. To do so, recall the optimization problem for GNN is 𝑓(𝜽):=
𝜎
L𝜎 ...𝜎(LXW(1))...W(𝐿)
. Let𝑓𝑖(𝜽)as the𝑖th output of
𝑓(𝜽). Formally, we consider the loss function and full-gradient
asL(𝜽)=Í𝑁
𝑖=1𝜙(𝑓𝑖(𝜽),𝑦𝑖)where∇L(𝜽)=Í𝑁
𝑖=1∇𝜙(𝑓𝑖(𝜽),𝑦𝑖).
Rather than using all samples at each steps, we sample a sequence
of random variables {𝜉𝑖}𝑁
𝑖=1, where𝜉𝑖∼Bernoulli(𝑝𝑖), and𝜉𝑖=1
indicates that the 𝑖th node is sampled and should be used to cal-
culate the stochastic gradient g=Í𝑁
𝑖=1𝜉𝑖
𝑝𝑖∇𝜙(𝑓𝑖(𝜽),𝑦𝑖). Define
G=E[∥g−E[g]∥2]. For a given mini-batch size 𝐵, our goal is to
find the best sampling probabilities {𝑝𝑖}𝑁
𝑖=1to minimize G, which
can be casted as the following optimization problem:
min𝑝𝑖𝑁∑︁
𝑖=11
𝑝𝑖∥∇𝜙(𝑓𝑖(𝜽),𝑦𝑖)∥2
subject to𝑁∑︁
𝑖=1𝑝𝑖=𝐵, 𝑝𝑖∈(0,1]for all𝑖.
Although this distribution can minimize the variance of the stochas-
tic gradient, it requires the calculation of 𝑁derivatives at each step,
which is clearly inefficient. As mentioned in [ 12,30], a practicalAlgorithm 1: MVS-GNN
input: initial point 𝜽={W(1),W(2),...,W(𝐿)}, learning
rate𝜂, mini-batch size 𝐵, importance sampling ratio 𝛾
SetH(0)=X
for𝑡=1,...,𝑇 do
/* Run large-batch GNN*/
SampleVS⊆V of size𝑆=𝑁×𝛾uniformly at random
Construct{˜L(ℓ)}𝐿
ℓ=1based on sampled nodes in VS
forℓ=1,...,𝐿 do
Estimate embedding matrices using Eq. 6 and update
history embeddings
end
Update parameters 𝜽←𝜽−𝜂1
𝑆Í
𝑖∈VS∇𝜙(˜H(𝐿)
𝑖,𝑦𝑖)
𝑝𝑖
Calculate gradient norm ¯g=[¯𝑔1,..., ¯𝑔𝑆]where
¯𝑔𝑖=∥∇𝜙(˜H(𝐿)
𝑖,𝑦𝑖)∥
/* Run mini-batch GNN*/
for𝑘=2,...,𝐾 do
Calculate the sampling distribution 𝒑=[𝑝1,...,𝑝𝑆]
using Eq. 9 based on ¯g
Sample nodesVB⊂VSof size𝐵with probability 𝒑
Construct{˜L(ℓ)}𝐿
ℓ=1for nodes inVB
forℓ=1,...,𝐿 do
Calculate embeddings using Eq. 7 and update
history embeddings
end
Update parameters 𝜽←𝜽−𝜂1
𝐵Í
𝑖∈VB∇𝜙(˜H(𝐿)
𝑖,𝑦𝑖)
𝑝𝑖
Update ¯gfor𝑖∈VBusing the norm of fresh
gradients
end
end
output: 𝜽
solution is to relax the optimization problem as follows
min𝑝𝑖𝑁∑︁
𝑖=1¯𝑔2
𝑖
𝑝𝑖
subject to𝑁∑︁
𝑖=1𝑝𝑖=𝐵, 𝑝𝑖∈(0,1]for all𝑖,(8)
where ¯𝑔𝑖≥∥∇𝜙(𝑓𝑖(𝜽),𝑦𝑖)∥is the upper-bound of the per-sample
gradient norm as estimated in Algorithm 1. In practice, we choose
to estimate ¯𝑔𝑖using the stochastic gradient of the last GNN layer.
Theorem 4.1. There exist a value 𝜇such that𝑝𝑖=min
1,¯𝑔𝑖
𝜇
is
the solution of Eq. 8.
Proof. The Lagrange function of Eq. 8 has form:
𝐿(𝛼,𝜷,𝜸)=𝑁∑︁
𝑖=1¯𝑔2
𝑖
𝑝𝑖+𝛼 𝑁∑︁
𝑖=1𝑝𝑖−𝐵!
−𝑁∑︁
𝑖=1𝛽𝑖𝑝𝑖−𝑁∑︁
𝑖=1𝛾𝑖(1−𝑝𝑖).

## Page 6

From the KKT conditions, we have
 
𝜕𝐿
𝜕𝑝𝑖=−¯𝑔2
𝑖
𝑝2
𝑖+𝛼−𝛽𝑖−𝛾𝑖=0for all𝑖
𝛽𝑖𝑝𝑖=0for all𝑖
𝛾𝑖(1−𝑝𝑖)=0for all𝑖
By examining these conditions, it is easy to conclude that optimal
solution has the following properties: (a) Since every 𝑝𝑖>0, we
have𝛽𝑖=0for all𝑖; (b) If𝛾𝑖>0, then𝑝𝑖=1and ¯𝑔2
𝑖>𝛼+𝛾𝑖>𝛼;
(c) If𝛾𝑖=0, then𝑝𝑖=√︃
¯𝑔2
𝑖/𝛼.
Putting all together, we know that there exist a threshold√𝛼
that divides sample into two parts: {𝑖:¯𝑔𝑖<√𝛼}of size𝜅with
𝑝𝑖=√︃
¯𝑔2
𝑖/𝛼and{𝑖:¯𝑔𝑖>√𝛼}of size𝑁−𝜅with𝑝𝑖=1
Therefore, it is sufficient to find 𝛼=𝛼★such thatÍ𝑁
𝑖=1𝑝𝑖=𝐵.
The desired value of 𝛼★can be found as a solution ofÍ𝑁
𝑖=1𝑝𝑖=
Í𝜅
𝑖=1√︃
¯𝑔2
𝑖
𝛼+𝑁−𝜅=𝐵. We conclude the proof by setting 𝜇=√
𝛼★. □
From Theorem 4.1, we know that given per-sample gradient, we
can calculate a Bernoulli importance sampling distribution 𝒑:=
{𝑝𝑖}𝑁
𝑖=1that minimize the variance. The following lemma gives
a brute-force algorithm to compute the 𝜇which can be used to
compute the optimal sampling probabilities.
Lemma 4.2. Suppose ¯𝑔𝑖is sorted such that 0<¯𝑔𝑖≤...≤¯𝑔𝑁. Let
𝜅be the largest integer for which 𝐵+𝜅−𝑁≤¯𝑔𝑖/(Í𝜅
𝑖=1¯𝑔𝑖), then
𝜇=(𝐵+𝜅−𝑁)/(Í𝜅
𝑖=1¯𝑔𝑖), and the probabilities can be computed by
𝑝𝑖=(
(𝐵+𝜅−𝑁)¯𝑔𝑖Í𝜅
𝑗=1¯𝑔𝑗if𝑖≤𝜅
1 if𝑖>𝜅(9)
Proof. The correctness of Lemma 4.2 can be shown by plugging
the result back to Theorem 4.1. □
If we assume 𝐵¯𝑔𝑁≤Í𝑁
𝑖=1¯𝑔𝑖, then𝜅=𝑁and𝑝𝑖=𝐵¯𝑔𝑖/(Í𝑁
𝑖=1¯𝑔𝑖).
Note that this assumption can be always satisfied by uplifting the
smallest ¯𝑔𝑖. We now compare the variance of the proposed impor-
tance sampling method with the variance of naive uniform sampling
in Lemma 4.3.
Lemma 4.3. Let𝒑𝑢𝑠=[𝑝1,...,𝑝𝑁]be the uniform sampling dis-
tribution with 𝑝𝑖=𝐵/𝑁, and 𝒑𝑖𝑠=[𝑝1,...,𝑝𝑁]as the minimal vari-
ance sampling distribution with 𝑝𝑖=𝐵¯𝑔𝑖/(Í𝑁
𝑖=1¯𝑔𝑖). Define G(𝒑𝑢𝑠)
andG(𝒑𝑖𝑠)as the variance of the stochastic gradient of uniform
and minimal variance sampling, respectively. Then, the difference
between the variance of uniform sampling and importance sampling
is proportion to the Euclidean distance between 𝒑𝑢𝑠and𝒑𝑖𝑠, i.e.,
G(𝒑𝑢𝑠)−G(𝒑𝑖𝑠)=Í𝑁
𝑖=1¯𝑔𝑖2
𝐵3𝑁∥𝒑𝑖𝑠−𝒑𝑢𝑠∥2
2.
Proof. Proof is deferred to Appendix 4.3. □
From Lemma 4.3, we observe that the variance of importance
sampling G(𝒑𝑖𝑠)is smaller than the variance of uniform sampling
G(𝒑𝑢𝑠)if the optimal importance sampling distribution is different
from uniform sampling distribution (the per sample gradient norm
is not all the same), i.e., 𝒑𝑖𝑠≠𝒑𝑢𝑠where 𝒑𝑖𝑠is defined in Eq. 9.Besides, the effect of variance reduction becomes more significant
when the difference between optimal importance sampling distribu-
tion and uniform sampling distribution is large (i.e., the difference
between per-sample gradient norm is large).
4.3 Implementation challenges
Calculating the optimal importance sampling distribution requires
having access to the stochastic gradient for every example in the
mini-batch. Unfortunately, existing machine learning packages,
such as Tensorflow [ 1] and PyTorch [ 19], does not support comput-
ing gradients with respect to individual examples in a mini-batch.
A naive approach to calculate the per sample gradient of 𝑁nodes
is to run backward propagation 𝑁times with a mini-batch size of 1.
In practice, the naive approach performs very poorly because back-
ward propagation is most efficient when efficient matrix operation
implementations can exploit the parallelism of mini-batch training.
As an alternative, we perform backward propagation only once
and reuse the intermediate results of backward propagation for per
sample gradient calculation. Recall that the embedding of node 𝑖at
theℓth GNN layer can be formulated as ˜H(ℓ)
𝑖=𝜎(˜L(ℓ)
𝑖˜H(ℓ−1)W(ℓ)).
During the forward propagation we save the ˜L(ℓ)
𝑖˜H(ℓ−1)and during
backward propagation we save the ∇˜H(ℓ)
𝑖L(𝜽). Then, the gradient
of updating W(ℓ)is calculated as
∇˜H(ℓ)
𝑖L(𝜽)
˜L(ℓ)
𝑖˜H(ℓ−1)
. De-
spite the need for additional space to store the gradient, the time it
takes to obtain per sample gradient is much lower.
5 EXPERIMENTS
In this section, we conduct experiments to evaluate MVS-GNN for
training GNNs on large-scale node classification datasets2.
Experiment setup. Experiments are under semi-supervised learn-
ing setting. We evaluate on the following real-world datasets: (1)
Reddit : classifying communities of online posts based on user com-
ments; (2) PPIandPPI-large : classifying protein functions based
on the interactions of human tissue proteins; (3) Yelp : classifying
product categories based on customer reviewers and friendship.
Detailed information are summarised in Table 2.
We compare with five baselines: node-wise sampling methods
GraphSage andVRGCN , a layer-wise sampling method LADIES , and
subgraph sampling methods ClusterGCN andGraphSaint . For a
given dataset, we keep the GNN structure the same across all meth-
ods. We train GNN with the default Laplacian multiplication aggre-
gation defined in [13] for Reddit dataset
H(ℓ)
𝑖=𝜎©­
«∑︁
𝑗∈N(𝑖)𝐿𝑖,𝑗H(ℓ−1)
𝑗W(ℓ)ª®
¬,
and add an extra concatenate operation defined in [ 11] for PPI,
PPI-large , and Yelp datasets. We train GNN with the default
Laplacian multiplication aggregation defined in [ 13] for Reddit
dataset
H(ℓ)
𝑖=𝜎©­
«concat©­
«H(ℓ−1)
𝑖,∑︁
𝑗∈N(𝑖)𝐿𝑖,𝑗H(ℓ−1)
𝑗ª®
¬W(ℓ)ª®
¬.
2The implementation of algorithms are publicly available at here.

## Page 7

YelpFigure 2: Convergence curves and gradient variance of 2-layer MVS-GNN and baseline models on Reddit ,PPI,PPI-large , and Yelp
dataset with batch size 512.
Table 2: Dataset statistics. s and m stand for single and multi-class classification problems, respectively.
Dataset Nodes Edges Degree Feature Classes Train/Val/Test
Reddit 232,965 11,606,919 50 602 41(s) 66%/10%/24%
PPI 14,755 225,270 15 50 121(m) 66%/12%/22%
PPI-large 56,944 2,818,716 14 50 121(m) 79%/11%/10%
Yelp 716,847 6,977,410 10 300 100(m) 75%/10%/15%
We make this decision because the default Laplacian multiplication
aggregation is prone to diverge on multi-class classification dataset.
By default, we train 2-layer GNNs with hidden state dimension
as𝐹=256. For node-wise sampling methods, we chose 5neighbors
to be sampled for GraphSage and2neighbors to be sampled for
VRGCN . For the layer-wise sampling method, we choose the layer
node sample size the same as the current batch size for LADIES
(e.g., if the mini-batch size is 512, then the layer node sample size
also equals to 512nodes). For the subgraph sampling method, we
partition a graph into clusters of size 128and construct the mini-
batch by choosing the desired number of clusters for ClusterGCN ,
and choose node sampling method for GraphSaint . We chose the
checkpoint sampling ratio ( 𝛾)10%forReddit ,100% forPPI,20%for
PPI-large , and 2%forYelp dataset. All methods terminate when
the validation accuracy does not increase a threshold 0.01for400
mini-batches on Reddit ,Yelp datasets and 1000 mini-batches on
PPIandPPI-large datasets. We conduct training for 3times and
take the mean of the evaluation results. We choose inner-loop size
𝐾=20as default and update the model with Adam optimizer with
a learning rate of 0.01.
The effect of mini-batch size. Table 3 shows the accuracy com-
parison of various methods using different batch sizes. Clearly,with decoupled variance reduction, MVS-GNN achieves significantly
higher accuracy, even when the batch size is small. Compared with
VRGCN , since MVS-GNN has “free” and “up-to-date” full-batch history
activations every 𝐾iterations, this guarantees the effectiveness
of function value variance reduction of our model during train-
ing. Compared with GraphSaint and ClusterGCN ,GraphSaint
performs node-wise graph sampling, which leads to a sparse
small graph with high variance when batch size is small, while
ClusterGCN first partition graph into several clusters and construct
a dense small graph which is highly biased when the batch size is
small.
Effectiveness of variance reduction. Figure 2 shows the mean-
square error of stochastic gradient and convergence of various
methods. Clearly, minimal variance sampling can lead to a variance
reduction of mini-batch estimated gradient and has a positive effect
on model performance.
Evaluation of total time. Table 4 shows the comparison of time
𝑇Sample ,𝑇Train,𝑇Dists onPPIdataset.𝑇Sample is defined as the time
of constructing 20mini-batches for training (in MVS-GNN is the time
of1large-batch and 19mini-batches). 𝑇Train is defined as the time
to run 20mini-batches for training (in MVS-GNN is the time of 1
large-batch and 19mini-batches). 𝑇Dists is defined as the time to

## Page 8

Table 3: Comparison of test set F1-micro for various batch
sizes.¶stands for out of memory error.
Batch
SizeMethod Reddit PPI PPI-large Yelp
256MVS-GNN 0.938 0.836 0.841 0.613
GraphSage 0.920 0.739 0.660 0.589
VRGCN 0.917 0.812 0.821 0.555
LADIES 0.932 0.583 0.603 0.596
ClusterGCN 0.739 0.586 0.608 0.538
GraphSaint 0.907 0.506 0.427 0.514
512MVS-GNN 0.942 0.859 0.864 0.617
GraphSage 0.932 0.781 0.766 0.606
VRGCN 0.929 0.831 0.829 0.607
LADIES 0.938 0.607 0.600 0.596
ClusterGCN 0.897 0.590 0.605 0.577
GraphSaint 0.921 0.577 0.531 0.540
1024MVS-GNN 0.946 0.864 0.875 0.619
GraphSage 0.939 0.809 0.789 0.611
VRGCN 0.934 0.848 0.849 0.615
LADIES 0.937 0.659 0.599 0.599
ClusterGCN 0.923 0.587 0.639 0.595
GraphSaint 0.929 0.611 0.558 0.550
2048MVS-GNN 0.949 0.880 0.892 0.620
GraphSage 0.944 0.839 0.833 0.617
VRGCN 0.945 0.844 0.856¶
LADIES 0.943 0.722 0.623 0.602
ClusterGCN 0.939 0.592 0.647 0.616
GraphSaint 0.931 0.633 0.593 0.559
Table 4: Comparison of average time (seconds) on PPI
dataset for 5-layer GNN with batch size 512.
Method 𝑇Sample𝑇Train𝑇Dists𝑇total
MVS-GNN 1.057 0.646 0.088 1.791
GraphSage 9.737 0.688 0 10.425
VRGCN 10.095 1.038 0 11.133
LADIES 1.031 0.295 0 1.326
ClusterGCN 1.140 0.672 0 1.812
GraphSaint 0.793 0.214 0 1.007
calculate the importance sampling distribution of each node for
minimal variance sampling. Therefore, the total time for 20itera-
tions is𝑇total=𝑇Sample+𝑇Train+𝑇Dists. To achieve fair comparison
in terms of sampling complexity, we implement all sampling meth-
ods using Python scipy.sparse andnumpy.random package, and
construct 20mini-batches in parallel by Python multiprocessing
package with 10threads. We choose the default setup and calcu-
late the sample distribution every 20iterations for MVS-GNN with
importance sampling ratio 100% . Because our method does not
need to recursively sample neighbors for each layer and each node
in the mini-batch, less time is required. Besides, since a constant
Figure 3: Comparison of gradient variance, training loss,
and testing loss on Reddit dataset with different number of
inner-loop iterations ( 𝐾=10,20,30,40).
number of nodes are calculated in each layer, our method is expo-
nentially faster than node-wise sampling algorithms with respect
to the number of layers.
Evaluation on inner-loop interval. MVS-GNN requires perform-
ing large-batch training periodically to calculate the importance
sampling distribution. A larger number of inner-loop interval ( 𝐾)
can make training speed faster, but also might make the importance
sample distribution too stale to represent the true distribution. In
Figure 3 , we show the comparison of gradient variance, training
loss, and testing loss with different number of inner-loop intervals
onReddit dataset. We choose mini-batch size 512, dropout rate 0.1,
importance sampling ratio 10%, and change the inner-loop intervals
from 10mini-batches to 30mini-batches.
Evaluation on small mini-batch size. In Figure 4, we show
the effectiveness of minimal variance sampling using small mini-
batch size on Cora ,Citeseer , and Pubmed dataset introduce in
[13]. To eliminate the embedding approximation variance, we use
all neighbors to inference the embedding matrix, such that the
only randomness happens at choosing nodes in mini-batch, which
is the original intention minimal variance sampling designed for.
We choose importance sampling ratio as 50%for Pubmed, 100% for
Cora and Citeseer, and update the importance sampling distribution
every 10iterations (shown as 1epoch in Figure 4). We choose hidden
state as 64, dropout ratio as 0.1, change the mini-batch size ( bz),
and monitor the difference of gradient variance, training loss, and
testing loss between minimal variance sampling (MVS) and uniform
sampling (UNS). Our result shows that minimal variance sampling
can significantly reduce the gradient variance and accelerate the
convergence speed during training.
6 CONCLUSION
In this work, we theoretically analyzed the variance of sampling
based methods for training GCNs and demonstrated that, due to
composite structure of empirical risk, the variance of any sampling
method can be decomposed as embedding approximation variance
and stochastic gradient variance. To mitigate these two types of

## Page 9

Figure 4: Comparison of gradient variance, training loss, and testing loss with small mini-batch size on Cora,Citeseer , and
Pubmed datasets.
variance and obtain faster convergence, a decoupled variance re-
duction strategy is proposed that employs gradient information
to sample nodes with minimal variance and explicitly reduce the
variance introduced by embedding approximation. We empirically
demonstrate the superior performance of the proposed decoupled
variance reduction method in comparison with the exiting sampling
methods, where it enjoys a faster convergence rate and a better
generalization performance even with smaller mini-batch sizes. We
leave exploring the empirical efficiency of proposed methods to
other variants of GNNs such as graph classification and attention
based GNNs as a future study.
REFERENCES
[1]Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey
Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al .
2016. Tensorflow: A system for large-scale machine learning. In 12th{USENIX}
Symposium on Operating Systems Design and Implementation ( {OSDI}16). 265–
283.
[2]Rianne van den Berg, Thomas N Kipf, and Max Welling. 2017. Graph convolu-
tional matrix completion. arXiv preprint arXiv:1706.02263 (2017).
[3]Jie Chen, Tengfei Ma, and Cao Xiao. 2018. Fastgcn: fast learning with graph
convolutional networks via importance sampling. arXiv preprint arXiv:1801.10247
(2018).
[4]Jianfei Chen, Jun Zhu, and Le Song. 2017. Stochastic training of graph con-
volutional networks with variance reduction. arXiv preprint arXiv:1710.10568
(2017).
[5]Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, and Cho-Jui Hsieh.
2019. Cluster-gcn: An efficient algorithm for training deep and large graph
convolutional networks. In Proceedings of the 25th ACM SIGKDD International
Conference on Knowledge Discovery & Data Mining . 257–266.
[6]Dominik Csiba, Zheng Qu, and Peter Richtárik. 2015. Stochastic dual coordinate
ascent with adaptive probabilities. In ICML . 674–683.
[7]Zhiyong Cui, Kristian Henrickson, Ruimin Ke, and Yinhai Wang. 2019. Traffic
graph convolutional recurrent neural network: A deep learning framework for
network-scale traffic learning and forecasting. IEEE Transactions on Intelligent
Transportation Systems (2019).
[8]Songgaojun Deng, Huzefa Rangwala, and Yue Ning. 2019. Learning Dynamic
Context Graphs for Predicting Social Events. In KDD . 1007–1016.
[9]Kien Do, Truyen Tran, and Svetha Venkatesh. 2019. Graph transformation policy
network for chemical reaction prediction. In KDD . 750–760.[10] David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell,
Timothy Hirzel, Alán Aspuru-Guzik, and Ryan P Adams. 2015. Convolutional
networks on graphs for learning molecular fingerprints. In NeurIPS . 2224–2232.
[11] Will Hamilton, Zhitao Ying, and Jure Leskovec. 2017. Inductive representation
learning on large graphs. In NeurIPS . 1024–1034.
[12] Angelos Katharopoulos and François Fleuret. 2018. Not all samples are created
equal: Deep learning with importance sampling. arXiv preprint arXiv:1803.00942
(2018).
[13] Thomas N Kipf and Max Welling. 2016. Semi-supervised classification with graph
convolutional networks. arXiv preprint arXiv:1609.02907 (2016).
[14] Srijan Kumar, Xikun Zhang, and Jure Leskovec. 2019. Predicting dynamic em-
bedding trajectory in temporal interaction networks. In KDD . 1269–1278.
[15] Jia Li, Zhichao Han, Hong Cheng, Jiao Su, Pengyun Wang, Jianfeng Zhang, and
Lujia Pan. 2019. Predicting Path Failure In Time-Evolving Graphs. In KDD .
1279–1289.
[16] Ruoyu Li, Sheng Wang, Feiyun Zhu, and Junzhou Huang. 2018. Adaptive graph
convolutional neural networks. In AAAI .
[17] Guillaume Papa, Pascal Bianchi, and Stéphan Clémençon. 2015. Adaptive sam-
pling for incremental optimization using stochastic gradient descent. In ALT.
Springer, 317–331.
[18] Namyong Park, Andrey Kan, Xin Luna Dong, Tong Zhao, and Christos Faloutsos.
2019. Estimating node importance in knowledge graphs using graph neural
networks. In KDD . 596–606.
[19] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory
Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al .2019.
PyTorch: An imperative style, high-performance deep learning library. In NeurIPS .
8024–8035.
[20] Jiezhong Qiu, Jian Tang, Hao Ma, Yuxiao Dong, Kuansan Wang, and Jie Tang.
2018. DeepInf: Modeling influence locality in large social networks. In KDD .
[21] Afshin Rahimi, Trevor Cohn, and Timothy Baldwin. 2018. Semi-supervised user
geolocation via graph convolutional networks. arXiv preprint arXiv:1804.08049
(2018).
[22] Farnood Salehi, Patrick Thiran, and Elisa Celis. 2018. Coordinate descent with
bandit sampling. In NeurIPS . 9247–9257.
[23] Sebastian U Stich, Anant Raj, and Martin Jaggi. 2017. Safe adaptive importance
sampling. In NeurIPS . 4381–4391.
[24] Hao Wang, Tong Xu, Qi Liu, Defu Lian, Enhong Chen, Dongfang Du, Han Wu,
and Wen Su. 2019. MCNE: An End-to-End Framework for Learning Multiple
Conditional Network Representations of Social Network. In KDD . 1064–1072.
[25] Hongwei Wang, Fuzheng Zhang, Mengdi Zhang, Jure Leskovec, Miao Zhao,
Wenjie Li, and Zhongyuan Wang. 2019. Knowledge-aware graph neural networks
with label smoothness regularization for recommender systems. In KDD . 968–
977.
[26] Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu, and Tat-Seng Chua. 2019. Kgat:
Knowledge graph attention network for recommendation. In KDD . 950–958.

## Page 10

[27] Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L Hamilton,
and Jure Leskovec. 2018. Graph convolutional neural networks for web-scale
recommender systems. In KDD . 974–983.
[28] Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, Rajgopal Kannan, and Viktor
Prasanna. 2019. Graphsaint: Graph sampling based inductive learning method.
arXiv preprint arXiv:1907.04931 (2019).
[29] Cheng Zhang, Hedvig Kjellstrom, and Stephan Mandt. 2017. Determinantal point
processes for mini-batch diversification. arXiv preprint arXiv:1705.00607 (2017).
[30] Peilin Zhao and Tong Zhang. 2015. Stochastic optimization with importance
sampling for regularized loss minimization. In ICML . 1–9.
[31] Q Zheng, P Richtárik, and T Zhang. 2014. Randomized dual coordinate ascent
with arbitrary sampling.
[32] Rong Zhu. 2016. Gradient-based sampling: An adaptive importance sampling for
least-squares. In NeurIPS . 406–414.
[33] Difan Zou, Ziniu Hu, Yewen Wang, Song Jiang, Yizhou Sun, and Quanquan Gu.
2019. Layer-Dependent Importance Sampling for Training Deep and Large Graph
Convolutional Networks. In NeurIPS . 11247–11256.

## Page 11

A PROOF OF LEMMA 3.1
We can bound∥g−˜g∥by adding and subtracting intermediate terms inside such that each adjacent pair of products differ at most in one
factor as follows:
E[∥g−˜g∥2]=E[∥∇𝑓(1)
𝜔1(𝜽𝑡)·∇𝑓(2)
𝜔2(𝐹(1)(𝜽))·∇𝑓(2)
𝜔3(𝐹(2)(𝜽))···∇𝑓(𝐿)
𝜔𝐿(𝐹(𝐿−1)(𝜽))
−∇𝑓(1)
𝜔1(𝜽)·∇𝑓(2)
𝜔2(𝑓(1)
𝜔1(𝜽))·∇𝑓(3)
𝜔3(𝑓(2)
𝜔2◦𝑓(1)
𝜔1(𝜽))···∇𝑓(𝐿)
𝜔𝐿(𝑓(𝐿−1)
𝜔𝐿−1◦···◦𝑓(1)
𝜔1(𝜽))∥2]
≤𝐿·(E[∥∇𝑓(1)
𝜔1(𝜽)·∇𝑓(2)
𝜔2(𝐹(1)(𝜽))·∇𝑓(3)
𝜔3(𝐹(2)(𝜽))···∇𝑓(𝐿)
𝜔𝐿(𝐹(𝐿−1)(𝜽))
−∇𝑓(1)
𝜔1(𝜽)·∇𝑓(2)
𝜔2(𝑓(1)
𝜔1(𝜽))·∇𝑓(3)
𝜔3(𝐹(2)(𝜽))···∇𝑓(𝐿)
𝜔𝐿(𝐹(𝐿−1)(𝜽))∥2]
+E[∥∇𝑓(1)
𝜔1(𝜽)·∇𝑓(2)
𝜔2(𝑓(1)
𝜔1(𝜽))·∇𝑓(3)
𝜔3(𝐹(2)(𝜽))···∇𝑓(𝐿)
𝜔𝐿(𝐹(𝐿−1)(𝜽))
−∇𝑓(1)
𝜔1(𝜽)·∇𝑓(2)
𝜔2(𝑓(1)
𝜔1(𝜽))·∇𝑓(3)
𝜔3(𝑓(2)
𝜔2◦𝑓(1)
𝜔1(𝜽))···∇𝑓(𝐿)
𝜔𝐿(𝐹(𝐿−1)(𝜽))∥2]+···
+E[∥∇𝑓(1)
𝜔1(𝜽)·∇𝑓(2)
𝜔2(𝑓(1)
𝜔1(𝜽))·∇𝑓(3)
𝜔3(𝑓(2)
𝜔2◦𝑓(1)
𝜔1(𝜽))···∇𝑓(𝐿)
𝜔𝐿(𝐹(𝐿−1)(𝜽))
−∇𝑓(1)
𝜔1(𝜽)·∇𝑓(2)
𝜔2(𝑓(1)
𝜔1(𝜽))·∇𝑓(3)
𝜔3(𝑓(2)
𝜔2◦𝑓(1)
𝜔1(𝜽))···∇𝑓(𝐿)
𝜔𝐿(𝑓(𝐿−1)
𝜔𝐿−1◦···◦𝑓(1)
𝜔1(𝜽))∥2])
≤𝐿·𝐿∑︁
ℓ=2ℓ−1Î
𝑖=1𝜌2
𝑖 𝐿Î
𝑖=ℓ+1𝜌2
𝑖
𝐺2
ℓ·E[∥𝐹(ℓ−1)(𝜽)−𝑓(ℓ−1)
𝜔ℓ−1◦···◦𝑓(1)
𝜔1(𝜽)∥2].(10)
We can bound E[∥𝐹(ℓ)(𝜽)−𝑓(ℓ)
𝜔ℓ◦···◦𝑓(1)
𝜔1(𝜽)∥2]by adding and subtracting intermediate terms inside the such that each adjacent pair
of products differ at most in one factor.
E[∥𝐹(ℓ)(𝜽)−𝑓(ℓ)
𝜔ℓ◦···◦𝑓(1)
𝜔1(𝜽)∥2]=E[∥𝑓(ℓ)
𝜔ℓ◦𝑓(ℓ−1)
𝜔ℓ−1◦···◦𝑓(1)
𝜔1(𝜽𝑡)−𝑓(ℓ)◦𝑓(ℓ−1)◦···◦𝑓(1)(𝜽𝑡)∥2]
≤ℓ
E[∥𝑓(ℓ)
𝜔ℓ◦𝑓(ℓ−1)
𝜔ℓ−1◦···◦𝑓(1)
𝜔1(𝜽𝑡)−𝑓(ℓ)
𝜔ℓ◦𝑓(ℓ−1)
𝜔ℓ−1◦···◦𝑓(2)
𝜔2(𝐹(1)(𝜽𝑡))∥2]
+E[∥𝑓(ℓ)
𝜔ℓ◦𝑓(ℓ−1)
𝜔ℓ−1◦···◦𝑓(2)
𝜔2(𝐹(1)(𝜽𝑡))−𝑓(ℓ)
𝜔ℓ◦𝑓(ℓ−1)
𝜔ℓ−1◦···◦𝑓(3)
𝜔3(𝐹(2)(𝜽𝑡))∥2]+···
+E[∥𝑓(ℓ)
𝜔ℓ(𝐹(ℓ−1)
𝑡)−𝐹(ℓ)
𝑡∥2]
≤ℓℓÎ
𝑖=2𝜌2
𝑖E[∥𝑓(1)
𝜔1(𝜽𝑡)−𝐹(1)(𝜽𝑡)∥2]+ℓÎ
𝑖=3𝜌2
𝑖E[∥𝑓(2)
𝜔2(𝐹(1)(𝜽𝑡))−𝐹(2)(𝜽𝑡)∥2]
+···+ E[∥𝑓(ℓ)
𝜔ℓ(𝐹(ℓ−1)(𝜽𝑡))−𝐹(ℓ)(𝜽𝑡)∥2]
=ℓℓ∑︁
𝑖=1 
ℓÎ
𝑗=𝑖+1𝜌2
𝑗E[∥𝑓(𝑗)
𝜔𝑗(𝐹(𝑗−1)(𝜽𝑡))−𝐹(𝑗)(𝜽𝑡)∥2]!
.(11)
LetVℓ:=E[∥𝑓(ℓ)
𝜔ℓ(𝐹(ℓ−1)(𝜽𝑡))−𝐹(ℓ)(𝜽𝑡)∥2]as the per layer embedding approximation variance. Combining Eq. 10 and Eq. 11, we obtain
the upper bound on the bias of stochastic gradient E[∥g−˜g∥2]as a linear combination of per layer embedding approximation variance:
E[∥g−˜g∥2]≤𝐿·𝐿∑︁
ℓ=2ℓ−1Î
𝑖=1𝜌2
𝑖 𝐿Î
𝑖=ℓ+1𝜌2
𝑖
𝐺2
ℓ·(ℓ−1)ℓ−1∑︁
𝑖=1 
ℓ−1Î
𝑗=𝑖+1𝜌2
𝑗V𝑗!
≤𝐿·𝐿∑︁
ℓ=2ℓ−1Î
𝑖=1𝜌2
𝑖 𝐿Î
𝑖=ℓ+1𝜌2
𝑖
𝐺2
ℓ·ℓℓ∑︁
𝑖=1 
ℓÎ
𝑗=𝑖+1𝜌2
𝑗V𝑗!
.
B EMBEDDING APPROXIMATION VARIANCE ANALYSIS
In this section, we analyze the variance of the approximation embedding for the sampled nodes at ℓth layer.
Lemma B.1 (Variance of MVS-GNN ).We assume that for each node, MVS-GNN randomly sample 𝑁ℓnodes atℓth layer to estimate the node
embedding, then we have Vℓ≤𝐷𝛽2
ℓΔ𝛾2
ℓ, where𝐷is the average node degree, Δ𝛾ℓis the upper bound of ∥(H(ℓ−1)
𝑖−¯H(ℓ−1)
𝑖)W(ℓ)∥, and𝛽ℓis the
upper bound of∥L𝑖,∗∥for any𝑖∈V.
Proof of Lemma B.1. By the update rule, we have
Vℓ=E[∥𝑓(ℓ)
𝜔ℓ(𝐹(ℓ−1)(𝜽𝑡))−𝐹(ℓ)(𝜽𝑡)∥2]
=1
𝑁ℓ∑︁
𝑖∈VℓE[∥∑︁
𝑗∈Vℓ−1˜𝐿(ℓ)
𝑖,𝑗H(ℓ−1)
𝑗W(ℓ)+∑︁
𝑗∈V\Vℓ−1𝐿𝑖,𝑗¯H(ℓ−1)
𝑗W(ℓ)−∑︁
𝑗∈V𝐿𝑖,𝑗H(ℓ−1)
𝑗W(ℓ)∥2]
=1
𝑁ℓ∑︁
𝑖∈VℓE[∥∑︁
𝑗∈Vℓ−1˜𝐿(ℓ)
𝑖,𝑗H(ℓ−1)
𝑗W(ℓ)+∑︁
𝑗∈V𝐿𝑖,𝑗¯H(ℓ−1)
𝑗W(ℓ)−∑︁
𝑗∈Vℓ−1˜𝐿(ℓ)
𝑖,𝑗¯H(ℓ−1)
𝑗W(ℓ)−∑︁
𝑗∈V𝐿𝑖,𝑗H(ℓ−1)
𝑗W(ℓ)∥2]

## Page 12

Since MVS-GNN performs subgraph sampling, only the node in the mini-batch are guaranteed to be sampled in the inner layers. Therefore,
the embedding approximation variance of MVS-GNN is similar to VRGCN with neighbor sampling size 𝑠=1. Denoting ΔH=H−¯H, we have
Vℓ≤1
𝑁ℓ∑︁
𝑖∈VℓE[∥∑︁
𝑗∈Vℓ−1˜L(ℓ)
𝑖,𝑗ΔH(ℓ−1)
𝑗W(ℓ)−∑︁
𝑗∈V𝐿𝑖,𝑗ΔH(ℓ−1)
𝑗W(ℓ)∥2]
=1
𝑁ℓ∑︁
𝑖∈Vℓ©­
«𝐷∑︁
𝑗∈V∥˜𝐿(ℓ)
𝑖,𝑗ΔH(ℓ−1)
𝑗W(ℓ)∥2−∥L𝑖,∗ΔH(ℓ−1)W(ℓ)∥2ª®
¬
≤1
𝑁ℓ∑︁
𝑖∈Vℓ𝐷∑︁
𝑗∈V∥˜𝐿(ℓ)
𝑖,𝑗∥2∥ΔH(ℓ−1)
𝑗W(ℓ)∥2≤𝐷𝛽2
ℓΔ𝛾2
ℓ
□
C PROOF OF LEMMA 4.3
Proof. According to the definition of G(𝒑𝑢𝑠)andG(𝒑𝑖𝑠), we have
G(𝒑𝑢𝑠)−G(𝒑𝑖𝑠)=1
𝑁2𝑁∑︁
𝑖=1¯𝑔2
𝑖𝑁
𝐵−1
𝑁2𝑁∑︁
𝑖=1Í𝑁
𝑗=1¯𝑔𝑗
𝐵¯𝑔𝑖¯𝑔2
𝑖
=1
𝐵𝑁∑︁
𝑖=1¯𝑔2
𝑖
𝑁−1
𝐵 𝑁∑︁
𝑖=1¯𝑔𝑖
𝑁!2
=(Í𝑁
𝑖=1¯𝑔𝑖)2
𝐵𝑁3𝑁∑︁
𝑖=1 
𝑁2¯𝑔2
𝑖
(Í𝑁
𝑗=1¯𝑔𝑗)2−1!
=(Í𝑁
𝑖=1¯𝑔𝑖)2
𝐵3𝑁𝑁∑︁
𝑖=1 
𝐵2¯𝑔2
𝑖
(Í𝑁
𝑗=1¯𝑔𝑗)2−𝐵2
𝑁2!
Using the fact thatÍ𝑁
𝑖=11/𝑁=1, we complete the derivation.
G(𝒑𝑢𝑠)−G(𝒑𝑖𝑠)=(Í𝑁
𝑖=1¯𝑔𝑖)2
𝐵3𝑁𝑁∑︁
𝑖=1 
𝐵¯𝑔𝑖Í𝑁
𝑗=1¯𝑔𝑗−𝐵
𝑁!2
=(Í𝑁
𝑖=1¯𝑔𝑖)2
𝐵3𝑁∥𝒑𝑖𝑠−𝒑𝑢𝑠∥2
2.
□
Evaluation on gradient distribution. To further illustrate the importance of minimal variance sampling, we show the distribution of per
sampler gradient during training on Cora dataset in Figure 5, where the dash line stands for the full-batch gradient. We observe that certain
stochastic gradients have more impact on the full-batch gradient than others, which motivates us to further reduce the variance of mini-bath
by sampling nodes with (approximately) large gradients more frequently.
Figure 5: The per sample gradient distribution during training.