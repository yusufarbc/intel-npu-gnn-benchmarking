# GraphSAINT Graph Sampling Based Inductive Learning Method


## Page 1

Published as a conference paper at ICLR 2020
GraphSAINT : G RAPH SAMPLING BASED INDUCTIVE
LEARNING METHOD
Hanqing Zeng
University of Southern California
zengh@usc.eduHongkuan Zhou
University of Southern California
hongkuaz@usc.edu
Ajitesh Srivastava
University of Southern California
ajiteshs@usc.eduRajgopal Kannan
US Army Research Lab
rajgopal.kannan.civ@mail.mil
Viktor Prasanna
University of Southern California
prasanna@usc.edu
ABSTRACT
Graph Convolutional Networks (GCNs) are powerful models for learning repre-
sentations of attributed graphs. To scale GCNs to large graphs, state-of-the-art
methods use various layer sampling techniques to alleviate the “neighbor explosion”
problem during minibatch training. We propose GraphSAINT , agraph sampling
based inductive learning method that improves training efﬁciency and accuracy in
a fundamentally different way. By changing perspective, GraphSAINT constructs
minibatches by sampling the training graph, rather than the nodes or edges across
GCN layers. Each iteration, a complete GCN is built from the properly sampled
subgraph. Thus, we ensure ﬁxed number of well-connected nodes in all layers. We
further propose normalization technique to eliminate bias, and sampling algorithms
for variance reduction. Importantly, we can decouple the sampling from the for-
ward and backward propagation, and extend GraphSAINT with many architecture
variants (e.g., graph attention, jumping connection). GraphSAINT demonstrates
superior performance in both accuracy and training time on ﬁve large graphs, and
achieves new state-of-the-art F1 scores for PPI (0.995) and Reddit (0.970).
1 I NTRODUCTION
Recently, representation learning on graphs has attracted much attention, since it greatly facilitates
tasks such as classiﬁcation and clustering (Wu et al., 2019; Cai et al., 2017). Current works on Graph
Convolutional Networks (GCNs) (Hamilton et al., 2017; Chen et al., 2018b; Gao et al., 2018; Huang
et al., 2018; Chen et al., 2018a) mostly focus on shallow models (2 layers) on relatively small graphs.
Scaling GCNs to larger datasets and deeper layers still requires fast alternate training methods.
In a GCN, data to be gathered for one output node comes from its neighbors in the previous layer.
Each of these neighbors in turn, gathers its output from the previous layer, and so on. Clearly, the
deeper we back track, the more multi-hop neighbors to support the computation of the root. The
number of support nodes (and thus the training time) potentially grows exponentially with the GCN
depth. To mitigate such “neighbor explosion”, state-of-the-art methods use various layer sampling
techniques. The works by Hamilton et al. (2017); Ying et al. (2018a); Chen et al. (2018a) ensure
that only a small number of neighbors (typically from 2 to 50) are selected by one node in the next
layer. Chen et al. (2018b) and Huang et al. (2018) further propose samplers to restrict the neighbor
expansion factor to 1, by ensuring a ﬁxed sample size in all layers. While these methods signiﬁcantly
speed up training, they face challenges in scalability, accuracy or computation complexity.
Equal contribution
1arXiv:1907.04931v4  [cs.LG]  16 Feb 2020

## Page 2

Published as a conference paper at ICLR 2020
Present work We present GraphSAINT (Graph SA mpling based INductive learning me Thod) to
efﬁciently train deep GCNs. GraphSAINT is developed from a fundamentally different way of
minibatch construction. Instead of building a GCN on the full training graph and then sampling
across the layers, we sample the training graph ﬁrst and then build a full GCN on the subgraph. Our
method is thus graph sampling based. Naturally, GraphSAINT resolves “neighbor explosion”, since
every GCN of the minibatches is a small yet complete one. On the other hand, graph sampling based
method also brings new challenges in training. Intuitively, nodes of higher inﬂuence on each other
should have higher probability to form a subgraph. This enables the sampled nodes to “support”
each other without going outside the minibatch. Unfortunately, such strategy results in non-identical
node sampling probability, and introduces bias in the minibatch estimator. To address this issue, we
develop normalization techniques so that the feature learning does not give preference to nodes more
frequently sampled. To further improve training quality, we perform variance reduction analysis, and
design light-weight sampling algorithms by quantifying “inﬂuence” of neighbors. Experiments on
GraphSAINT using ﬁve large datasets show signiﬁcant performance gain in both training accuracy
and time. We also demonstrate the ﬂexibility of GraphSAINT by integrating our minibatch method
with popular GCN architectures such as JK-net (Xu et al., 2018) and GAT (Veli ˇckovi ´c et al., 2017).
The resulting deep models achieve new state-of-the-art F1 scores on PPI (0.995) and Reddit (0.970).
2 R ELATED WORK
A neural network model that extends convolution operation to the graph domain is ﬁrst proposed
by Bruna et al. (2013). Further, Kipf & Welling (2016); Defferrard et al. (2016) speed up graph
convolution computation with localized ﬁlters based on Chebyshev expansion. They target relatively
small datasets and thus the training proceeds in full batch. In order to scale GCNs to large graphs,
layer sampling techniques (Hamilton et al., 2017; Chen et al., 2018b; Ying et al., 2018a; Chen et al.,
2018a; Gao et al., 2018; Huang et al., 2018) have been proposed for efﬁcient minibatch training.
All of them follow the three meta steps: 1. Construct a complete GCN on the full training graph.
2. Sample nodes or edges of each layer to form minibatches. 3. Propagate forward and backward
among the sampled GCN. Steps (2) and (3) proceed iteratively to update the weights via stochastic
gradient descent. The layer sampling algorithm of GraphSAGE (Hamilton et al., 2017) performs
uniform node sampling on the previous layer neighbors. It enforces a pre-deﬁned budget on the
sample size, so as to bound the minibatch computation complexity. Ying et al. (2018a) enhances the
layer sampler of Hamilton et al. (2017) by introducing an importance score to each neighbor. The
algorithm presumably leads to less information loss due to weighted aggregation. S-GCN (Chen et al.,
2018a) further restricts neighborhood size by requiring only two support nodes in the previous layer.
The idea is to use the historical activations in the previous layer to avoid redundant re-evaluation.
FastGCN (Chen et al., 2018b) performs sampling from another perspective. Instead of tracking down
the inter-layer connections, node sampling is performed independently for each layer. It applies
importance sampling to reduce variance, and results in constant sample size in all layers. However,
the minibatches potentially become too sparse to achieve high accuracy. Huang et al. (2018) improves
FastGCN by an additional sampling neural network. It ensures high accuracy, since sampling is
conditioned on the selected nodes in the next layer. Signiﬁcant overhead may be incurred due to the
expensive sampling algorithm and the extra sampler parameters to be learned.
Instead of sampling layers, the works of Zeng et al. (2018) and Chiang et al. (2019) build mini-
batches from subgraphs. Zeng et al. (2018) proposes a speciﬁc graph sampling algorithm to ensure
connectivity among minibatch nodes. They further present techniques to scale such training on
shared-memory multi-core platforms. More recently, ClusterGCN (Chiang et al., 2019) proposes
graph clustering based minibatch training. During pre-processing, the training graph is partitioned
into densely connected clusters. During training, clusters are randomly selected to form minibatches,
and intra-cluster edge connections remain unchanged. Similar to GraphSAINT , the works of Zeng
et al. (2018) and Chiang et al. (2019) do not sample the layers and thus “neighbor explosion” is
avoided. Unlike GraphSAINT , both works are heuristic based, and do not account for bias due to the
unequal probability of each node / edge appearing in a minibatch.
Another line of research focuses on improving model capacity. Applying attention on graphs, the
architectures of Veli ˇckovi ´c et al. (2017); Zhang et al. (2018); Lu et al. (2019) better capture neighbor
features by dynamically adjusting edge weights. Klicpera et al. (2018) combines PageRank with
GCNs to enable efﬁcient information propagation from many hops away. To develop deeper models,
2

## Page 3

Published as a conference paper at ICLR 2020
“skip-connection” is borrowed from CNNs (He et al., 2015; Huang et al., 2017) into the GCN context.
In particular, JK-net Xu et al. (2018) demonstrates signiﬁcant accuracy improvement on GCNs with
more than two layers. Note, however, that JK-net (Xu et al., 2018) follows the same sampling strategy
as GraphSAGE (Hamilton et al., 2017). Thus, its training cost is high due to neighbor explosion. In
addition, high order graph convolutional layers (Zhou, 2017; Lee et al., 2018; Abu-El-Haija et al.,
2019) also help propagate long-distance features. With the numerous architectural variants developed,
the question of how to train them efﬁciently via minibatches still remains to be answered.
3 P ROPOSED METHOD :GraphSAINT
Graph sampling based method is motivated by the challenges in scalability (in terms of model depth
and graph size). We analyze the bias (Section 3.2) and variance (Section 3.3) introduced by graph
sampling, and thus, propose feasible sampling algorithms (Section 3.4). We show the applicability of
GraphSAINT to other architectures, both conceptually (Section 4) and experimentally (Section 5.2).
In the following, we deﬁne the problem of interest and the corresponding notations. A GCN learns
representation of an un-directed, attributed graph G(V;E), where each node v2V has a length- f
attribute xv. LetAbe the adjacency matrix and eAbe the normalized one (i.e., eA=D 1A, andD
is the diagonal degree matrix). Let the dimension of layer- `input activation be f(`). The activation
of nodevisx(`)
v2Rf(`), and the weight matrix is W(`)2Rf(`)f(`+1). Note that xv=x(1)
v.
Propagation rule of a layer is deﬁned as follows:
x(`+1)
v = X
u2VeAv;u
W(`)T
x(`)
u!
(1)
whereeAv;uis a scalar, taking an element of eA. And()is the activation function (e.g., ReLU).
We use subscript “s” to denote parameterd of the sampled graph (e.g., Gs,Vs,Es).
GCNs can be applied under inductive and transductive settings. While GraphSAINT is applicable
to both, in this paper, we focus on inductive learning. It has been shown that inductive learning is
especially challenging (Hamilton et al., 2017) — during training, neither attributes nor connections
of the test nodes are present. Thus, an inductive model has to generalize to completely unseen graphs.
3.1 M INIBATCH BY GRAPH SAMPLING
0
1
234
568
9
7 0 1 2 3 5 70 1 2 3 5 70 1 2 3 5 7
Gs=SAMPLE (G) Full GCN onGs
Figure 1: GraphSAINT training algorithm
GraphSAINT follows the design philosophy of directly sampling the training graph G, rather than
the corresponding GCN. Our goals are to 1. extract appropriately connected subgraphs so that little
information is lost when propagating within the subgraphs, and 2. combine information of many
subgraphs together so that the training process overall learns good representation of the full graph.
Figure 1 and Algorithm 1 illustrate the training algorithm. Before training starts, we perform
light-weight pre-processing on Gwith the given sampler SAMPLE . The pre-processing estimates the
probability of a node v2V and an edge e2Ebeing sampled by SAMPLE . Such probability is later
used to normalize the subgraph neighbor aggregation and the minibatch loss (Section 3.2). Afterwards,
3

## Page 4

Published as a conference paper at ICLR 2020
Algorithm 1 GraphSAINT training algorithm
Input: Training graphG(V;E;X); Labels Y; Sampler SAMPLE ;
Output: GCN model with trained weights
1:Pre-processing: Setup SAMPLE parameters; Compute normalization coefﬁcients ,.
2:foreach minibatch do
3:Gs(Vs;Es) Sampled sub-graph of Gaccording to SAMPLE
4: GCN construction on Gs.
5:fyvjv2Vsg Forward propagation of fxvjv2Vsg, normalized by 
6: Backward propagation from -normalized loss L(yv;yv). Update weights.
7:end for
training proceeds by iterative weight updates via SGD. Each iteration starts with an independently
sampledGs(wherejVsjjVj ). We then build a full GCN on Gsto generate embedding and
calculate loss for every v2Vs. In Algorithm 1, node representation is learned by performing node
classiﬁcation in the supervised setting, and each training node vcomes with a ground truth label yv.
Intuitively, there are two requirements for SAMPLE : 1. Nodes having high inﬂuence on each other
should be sampled in the same subgraph. 2. Each edge should have non-negligible probability to
be sampled. For requirement (1), an ideal SAMPLE would consider the joint information from node
connections as well as attributes. However, the resulting algorithm may have high complexity as it
would need to infer the relationships between features. For simplicity, we deﬁne “inﬂuence” from the
graph connectivity perspective and design topology based samplers. Requirement (2) leads to better
generalization since it enables the neural net to explore the full feature and label space.
3.2 N ORMALIZATION
A sampler that preserves connectivity characteristic of Gwill almost inevitably introduce bias into
minibatch estimation. In the following, we present normalization techniques to eliminate biases.
Analysis of the complete multi-layer GCN is difﬁcult due to non-linear activations. Thus, we analyze
the embedding of each layer independently. This is similar to the treatment of layers independently
by prior work (Chen et al., 2018b; Huang et al., 2018). Consider a layer- (`+ 1) nodevand a layer-`
nodeu. Ifvis sampled (i.e., v2Vs), we can compute the aggregated feature of vas:
(`+1)
v =X
u2VeAv;u
u;v
W(`)T
x(`)
u 1ujv=X
u2VeAv;u
u;v~x(`)
u 1ujv; (2)
where ~x(`)
u= 
W(`)Tx(`)
u, and 1ujv2f0;1gis the indicator function given vis in the subgraph
(i.e., 1ujv= 0ifv2Vs^(u;v)62Es; 1ujv= 1if(u;v)2Es; 1ujvnot deﬁned if v62Vs). We refer
to the constant u;vasaggregator normalization . Deﬁnepu;v=pv;uas the probability of an edge
(u;v)2Ebeing sampled in a subgraph, and pvas the probability of a node v2V being sampled.
Proposition 3.1. (`+1)
v is an unbiased estimator of the aggregation of vin the full (`+ 1)thGCN
layer, ifu;v=pu;v
pv. i.e.,E
(`+1)
v
=P
u2VeAv;u~x(`)
u.
Assuming that each layer independently learns an embedding, we use Proposition 3.1 to normalize
feature propagation of each layer of the GCN built by GraphSAINT . Further, let Lvbe the loss on vin
the output layer. The minibatch loss is calculated as Lbatch=P
v2GsLv=v, wherevis a constant
that we term loss normalization . We setv=jVjpvso that:
E(Lbatch) =1
jGjX
Gs2GX
v2VsLv
v=1
jVjX
v2VLv: (3)
Feature propagation within subgraphs thus requires normalization factors and, which are com-
puted based on the edge and node probability pu;v,pv. In the case of random node or random
edge samplers, pu;vandpvcan be derived analytically. For other samplers in general, closed form
expression is hard to obtain. Thus, we perform pre-processing for estimation. Before training starts,
4

## Page 5

Published as a conference paper at ICLR 2020
we run the sampler repeatedly to obtain a set of Nsubgraphs G. We setup a counter CvandCu;vfor
eachv2V and(u;v)2E, to count the number of times the node or edge appears in the subgraphs
ofG. Then we set u;v=Cu;v
Cv=Cv;u
Cvandv=Cv
N. The subgraphsGs2Gcan all be reused as
minibatches during training. Thus, the overhead of pre-processing is small (see Appendix D.2).
3.3 V ARIANCE
We derive samplers for variance reduction. Let ebe the edge connecting u,v, andb(`)
e=eAv;u~x(` 1)
u+
eAu;v~x(` 1)
v . It is desirable that variance of all estimators (`)
vis small. With this objective, we deﬁne:
=X
`X
v2Gs(`)
v
pv=X
`X
v;ueAv;u
pvu;v~x(`)
u 1v 1ujv=X
`X
eb(`)
e
pe1(`)
e: (4)
where 1e= 1ife2Es; 1e= 0ife62Es. And 1v= 1ifv2Vs; 1v= 0ifv62Vs. The factorpuin
the ﬁrst equality is present so that is an unbiased estimator of the sum of all node aggregations at all
layers: E() =P
`P
v2VE
(`)
v
. Note that 1(`)
e= 1e;8`, since once an edge is present in the
sampled graph, it is present in all layers of our GCN.
We deﬁne the optimal edge sampler to minimize variance for every dimension of . We restrict
ourselves to independent edge sampling. For each e2E, we make independent decision on whether
it should be inGsor not. The probability of including eispe. We further constrainPpe=m, so
that the expected number of sampled edges equals to m. The budget mis a given sampling parameter.
Theorem 3.2. Under independent edge sampling with budget m, the optimal edge probabilities to
minimize the sum of variance of each ’s dimension is given by: pe=mP
e0


P
`b(`)
e0





P
`b(`)
e


.
To prove Theorem 3.2, we make use of the independence among graph edges, and the dependence
among layer edges to obtain the covariance of 1(`)
e. Then using the fact that sum of peis a constant,
we use the Cauchy-Schwarz inequality to derive the optimal pe. Details are in Appendix A.
Note that calculating b(`)
erequires computing ~x(` 1)
v , which increases the complexity of sampling.
As a reasonable simpliﬁcation, we ignore ~x(`)
vto make the edge probability dependent on the graph
topology only. Therefore, we choose pe/eAv;u+eAu;v=1
deg(u)+1
deg(v).
The derived optimal edge sampler agrees with the intuition in Section 3.1. If two nodes u,vare
connected and they have few neighbors, then uandvare likely to be inﬂuential to each other. In this
case, the edge probability pu;v=pv;ushould be high. The above analysis on edge samplers also
inspires us to design other samplers, which are presented in Section 3.4.
Remark We can also apply the above edge sampler to perform layer sampling. Under the indepen-
dent layer sampling assumption of Chen et al. (2018b), one would sample a connection 
u(`);v(`+1)
with probability p(`)
u;v/1
deg(u)+1
deg(v). For simplicity, assume a uniform degree graph (of degree d).
Thenp(`)
e=p. For an already sampled u(`)to connect to layer `+ 1, at least one of its edges has to
be selected by the layer `+ 1sampler. Clearly, the probability of an input layer node to “survive” the
Lnumber of independent sampling process is
1 (1 p)dL 1
. Such layer sampler potentially
returns an overly sparse minibatch for L>1. On the other hand, connectivity within a minibatch of
GraphSAINT never drops with GCN depth. If an edge is present in layer `, it is present in all layers.
3.4 S AMPLERS
Based on the above variance analysis, we present several light-weight and efﬁcient samplers that
GraphSAINT has integrated. Detailed sampling algorithms are listed in Appendix B.
Random node sampler We samplejVsjnodes fromVrandomly, according to a node probability
distributionP(u)/


eA:;u


2
. This sampler is inspired by the layer sampler of Chen et al. (2018b).
5

## Page 6

Published as a conference paper at ICLR 2020
Random edge sampler We perform edge sampling as described in Section 3.3.
Random walk based samplers Another way to analyze graph sampling based multi-layer GCN is
to ignore activations. In such case, Llayers can be represented as a single layer with edge weights
given by B=eAL. Following a similar approach as Section 3.3, if it were possible to pick pairs of
nodes (whether or not they are directly connected in the original eA) independently, then we would
setpu;v/Bu;v+Bv;u, where Bu;vcan be interpreted as the probability of a random walk to
start atuand end atvinLhops (and Bv;uvice-versa). Even though it is not possible to sample
a subgraph where such pairs of nodes are independently selected, we still consider a random walk
sampler with walk length Las a good candidate for L-layer GCNs. There are numerous random
walk based samplers proposed in the literature (Ribeiro & Towsley, 2010; Leskovec & Faloutsos,
2006; Hu & Lau, 2013; Li et al., 2015). In the experiments, we implement a regular random walk
sampler (with rroot nodes selected uniformly at random and each walker goes hhops), and also a
multi-dimensional random walk sampler deﬁned in Ribeiro & Towsley (2010).
For all the above samplers, we return the subgraph induced from the sampled nodes. The induction
step adds more connections into the subgraph, and empirically helps improve convergence.
4 D ISCUSSION
Extensions GraphSAINT admits two orthogonal extensions. First, GraphSAINT can seamlessly
integrate other graph samplers. Second, the idea of training by graph sampling is applicable to many
GCN architecture variants: 1. Jumping knowledge (Xu et al., 2018): since our GCNs constructed
during training are complete, applying skip connections to GraphSAINT is straightforward. On
the other hand, for some layer sampling methods (Chen et al., 2018b; Huang et al., 2018), extra
modiﬁcation to their samplers is required, since the jumping knowledge architecture requires layer- `
samples to be a subset of layer- (` 1)samples. 2.Attention (Veli ˇckovi ´c et al., 2017; Fey, 2019;
Zhang et al., 2018): while explicit variance reduction is hard due to the dynamically updated attention
values, it is reasonable to apply attention within the subgraphs which are considered as representatives
of the full graph. Our loss and aggregator normalizations are also applicabley. 3.Others : To support
high order layers (Zhou, 2017; Lee et al., 2018; Abu-El-Haija et al., 2019) or even more complicated
networks for the task of graph classiﬁcation (Ying et al., 2018b), we replace the full adjacency matrix
Awith the (normalized) one for the subgraph Asto perform layer propagation.
Comparison GraphSAINT enjoys: 1. high scalability and efﬁciency, 2. high accuracy, and 3. low
training complexity. Point (1) is due to the signiﬁcantly reduced neighborhood size compared with
Hamilton et al. (2017); Ying et al. (2018a); Chen et al. (2018a). Point (2) is due to the better inter-
layer connectivity compared with Chen et al. (2018b), and unbiased minibatch estimator compared
with Chiang et al. (2019). Point (3) is due to the simple and trivially parallelizable pre-processing
compared with the sampling of Huang et al. (2018) and clustering of Chiang et al. (2019).
5 E XPERIMENTS
Setup Experiments are under the inductive, supervised learning setting. We evaluate GraphSAINT
on the following tasks: 1. classifying protein functions based on the interactions of human tissue
proteins (PPI), 2. categorizing types of images based on the descriptions and common properties
of online images (Flickr), 3. predicting communities of online posts based on user comments
(Reddit), 4. categorizing types of businesses based on customer reviewers and friendship (Yelp),
and 5. classifying product categories based on buyer reviewers and interactions (Amazon). For PPI,
we use the small version for the two layer convergence comparison (Table 2 and Figure 2), since
Hamilton et al. (2017) and Chen et al. (2018a) report accuracy for this version in their original papers.
We use the large version for additional comparison with Chiang et al. (2019) to be consistent with its
reported accuracy. All datasets follow “ﬁxed-partition” splits. Appendix C.2 includes further details.
The skip-connection design proposed by Huang et al. (2018) does not have such “subset” requirement, and
thus is compatible with both graph sampling and layer sampling based methods.
yWhen applying GraphSAINT to GAT (Veli ˇckovi ´c et al., 2017), we remove the softmax step which normalizes
attention values within the same neighborhood, as suggested by Huang et al. (2018). See Appendix C.3.
6

## Page 7

Published as a conference paper at ICLR 2020
Table 1: Dataset statistics (“m” stands for multi-class classiﬁcation, and “s” for single-class.)
Dataset Nodes Edges Degree Feature Classes Train / Val / Test
PPI 14,755 225,270 15 50 121 (m) 0.66 / 0.12 / 0.22
Flickr 89,250 899,756 10 500 7 (s) 0.50 / 0.25 / 0.25
Reddit 232,965 11,606,919 50 602 41 (s) 0.66 / 0.10 / 0.24
Yelp 716,847 6,977,410 10 300 100 (m) 0.75 / 0.10 / 0.15
Amazon 1,598,960 132,169,734 83 200 107 (m) 0.85 / 0.05 / 0.10
PPI (large version) 56,944 818,716 14 50 121 (m) 0.79 / 0.11 / 0.10
We open source GraphSAINTz. We compare with six baselines: 1. vanilla GCN (Kipf & Welling,
2016), 2. GraphSAGE (Hamilton et al., 2017), 3. FastGCN (Chen et al., 2018b), 4. S-GCN (Chen
et al., 2018a), 5. AS-GCN (Huang et al., 2018), and 6. ClusterGCN (Chiang et al., 2019). All
baselines are executed with their ofﬁcially released code (see Appendix C.3 for downloadable URLs
and commit numbers). Baselines and GraphSAINT are all implemented in Tensorﬂow with Python3.
We run experiments on a NVIDIA Tesla P100 GPU (see Appendix C.1 for hardware speciﬁcation).
5.1 C OMPARISON WITH STATE -OF-THE-ART
Table 2 and Figure 2 show the accuracy and convergence comparison of various methods. All results
correspond to two-layer GCN models (for GraphSAGE, we use its mean aggregator). For a given
dataset, we keep hidden dimension the same across all methods. We describe the detailed architecture
and hyperparameter search procedure in Appendix C.3. The mean and conﬁdence interval of the
accuracy values in Table 2 are measured by three runs under the same hyperparameters. The training
time of Figure 2 excludes the time for data loading, pre-processing, validation set evaluation and
model saving. Our pre-processing incurs little overhead in training time. See Appendix D.2 for cost
of graph sampling. For GraphSAINT , we implement the graph samplers described in Section 3.4. In
Table 2, “Node” stands for random node sampler; “Edge” stands for random edge sampler; “RW”
stands for random walk sampler; “MRW” stands for multi-dimensional random walk sampler.
Table 2: Comparison of test set F1-micro score with state-of-the-art methods
Method PPI Flickr Reddit Yelp Amazon
GCN 0.5150.006 0.4920.003 0.9330.000 0.3780.001 0.2810.005
GraphSAGE 0.637 0.006 0.5010.013 0.9530.001 0.6340.006 0.7580.002
FastGCN 0.5130.032 0.5040.001 0.9240.001 0.2650.053 0.1740.021
S-GCN 0.9630.010 0.4820.003 0.9640.001 0.6400.002 —z
AS-GCN 0.6870.012 0.5040.002 0.9580.001 —z—z
ClusterGCN 0.875 0.004 0.4810.005 0.9540.001 0.6090.005 0.7590.008
GraphSAINT -Node 0.9600.001 0.5070.001 0.9620.001 0.6410.000 0.7820.004
GraphSAINT -Edge 0.9810.007 0.5100.002 0.9660.001 0.6530.003 0.8070.001
GraphSAINT -RW 0.9810.004 0.5110.001 0.9660.001 0.6530.003 0.8150.001
GraphSAINT -MRW 0.9800.006 0.5100.001 0.9640.000 0.6520.001 0.8090.001
Table 3: Additional comparison with ClusterGCN (test set F1-micro score)
PPI (large version) Reddit
2512 52048 2128 4128
ClusterGCN 0.903 0.002 0.9940.000 0.9540.001 0.9660.001
GraphSAINT 0.9410.003 0.9950.000 0.9660.001 0.9700.001
zOpen sourced code: https://github.com/GraphSAINT/GraphSAINT
zThe codes throw runtime error on the large datasets (Yelp or Amazon).
7

## Page 8

Published as a conference paper at ICLR 2020
0 20 40 600:40:60:81Validation F1-microPPI
0 10 20 30 400:440:460:480:50:52Flickr
0 50 100 1500:90:920:940:960:98Reddit
0 200 400 600 8000:250:450:65Yelp
0 200 4000:20:40:60:8
Training time (second)Amazon
GCN GraphSAGE
FastGCN* S-GCN
AS-GCN ClusterGCN
GraphSAINT
*: CPU execution time-RW
Figure 2: Convergence curves of 2-layer models on GraphSAINT and baselines
Clearly, with appropriate graph samplers, GraphSAINT achieves signiﬁcantly higher accuracy on all
datasets. For GraphSAINT -Node, we use the same node probability as FastGCN. Thus, the accuracy
improvement is mainly due to the switching from layer sampling to graph sampling (see “Remark” in
Section 3.3). Compared with AS-GCN, GraphSAINT is signiﬁcantly faster. The sampler of AS-GCN
is expensive to execute, making its overall training time even longer than vanilla GCN. We provide
detailed computation complexity analysis on the sampler in Appendix D.2. For S-GCN on Reddit, it
achieves similar accuracy as GraphSAINT , at the cost of over 9longer training time. The released
code of FastGCN only supports CPU execution, so its convergence curve is dashed.
Table 3 presents additional comparison with ClusterGCN. We use Lfto specify the architecture,
whereLandfdenote GCN depth and hidden dimension, respectively. The four architectures are
the ones used in the original paper (Chiang et al., 2019). Again, GraphSAINT achieves signiﬁcant
accuracy improvement. To train models with L>2often requires additional architectural tweaks.
ClusterGCN uses its diagonal enhancement technique for the 5-layer PPI and 4-layer Reddit models.
GraphSAINT uses jumping knowledge connection (Xu et al., 2018) for 4-layer Reddit.
Evaluation on graph samplers From Table 2, random edge and random walk based samplers
achieve higher accuracy than the random node sampler. Figure 3 presents sensitivity analysis on
parameters of “RW”. We use the same hyperparameters (except the sampling parameters) and network
architecture as those of the “RW” entries in Table 2. We ﬁx the length of each walker to 2(i.e., GCN
depth), and vary the number of roots rfrom 250to2250 . For PPI, increasing rfrom 250to750
signiﬁcantly improves accuracy. Overall, for all datasets, accuracy stabilizes beyond r= 750 .
5.2 GraphSAINT ONARCHITECTURE VARIANTS AND DEEPMODELS
In Figure 4, we train a 2-layer and a 4-layer model of GAT (Veli ˇckovi ´c et al., 2017) and JK-net (Xu
et al., 2018), by using minibatches of GraphSAGE and GraphSAINT respectively. On the two 4-layer
architectures, GraphSAINT achieves two orders of magnitude speedup than GraphSAGE, indicating
much better scalability on deep models. From accuracy perspective, 4-layer GAT-SAGE and JK-
SAGE do not outperform the corresponding 2-layer versions, potentially due to the smoothening
effect caused by the massive neighborhood size. On the other hand, with minibatches returned by our
edge sampler, increasing model depth of JK- SAINT leads to noticeable accuracy improvement (from
0.966 of 2-layer to 0.970 of 4-layer). Appendix D.1 contains additional scalability results.
6 C ONCLUSION
We have presented GraphSAINT , a graph sampling based training method for deep GCNs on large
graphs. We have analyzed bias and variance of the minibatches deﬁned on subgraphs, and proposed
8

## Page 9

Published as a conference paper at ICLR 2020
0 1;000 2;0000:40:60:81
Number of walkersTest F1-microPPI Flickr Reddit
Yelp Amazon
Figure 3: Sensitivity analysis1001021040:930:940:950:960:97
Training time (second)Validation F1-microGAT
100102104JK-netGraphSAINT 2-layer GraphSAINT 4-layer
GraphSAGE 2-layer GraphSAGE 4-layer
Figure 4: GraphSAINT with JK-net and GAT (Reddit)
normalization techniques and sampling algorithms to improve training quality. We have conducted
extensive experiments to demonstrate the advantage of GraphSAINT in accuracy and training time.
An interesting future direction is to develop distributed training algorithms using graph sampling
based minibatches. After partitioning the training graph in distributed memory, sampling can be
performed independently on each processor. Afterwards, training on the self-supportive subgraphs can
signiﬁcantly reduce the system-level communication cost. To ensure the overall convergence quality,
data shufﬂing strategy for the graph nodes and edges can be developed together with each speciﬁc
graph sampler. Another direction is to perform algorithm-system co-optimization to accelerate
the training of GraphSAINT on heterogeneous computing platforms (Zeng et al., 2018; Zeng &
Prasanna, 2019). The resolution of “neighbor explosion” by GraphSAINT not only reduces the
training computation complexity, but also improves hardware utilization by signiﬁcantly less data
trafﬁc to the slow memory. In addition, task-level parallelization is easy since the light-weight graph
sampling is completely decoupled from the GCN layer propagation.
ACKNOWLEDGEMENT
This material is based on work supported by the Defense Advanced Research Projects Agency
(DARPA) under Contract Number FA8750-17-C-0086 and National Science Foundation (NSF) under
Contract Numbers CCF-1919289 and OAC-1911229. Any opinions, ﬁndings and conclusions or
recommendations expressed in this material are those of the authors and do not necessarily reﬂect the
views of DARPA or NSF.
REFERENCES
Sami Abu-El-Haija, Bryan Perozzi, Amol Kapoor, Hrayr Harutyunyan, Nazanin Alipourfard, Kristina
Lerman, Greg Ver Steeg, and Aram Galstyan. Mixhop: Higher-order graph convolution architec-
tures via sparsiﬁed neighborhood mixing. arXiv preprint arXiv:1905.00067 , 2019.
Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann LeCun. Spectral networks and locally
connected networks on graphs. CoRR , abs/1312.6203, 2013. URL http://arxiv.org/abs/
1312.6203 .
HongYun Cai, Vincent W. Zheng, and Kevin Chen-Chuan Chang. A comprehensive survey of
graph embedding: Problems, techniques and applications. CoRR , abs/1709.07604, 2017. URL
http://arxiv.org/abs/1709.07604 .
Jianfei Chen, Jun Zhu, and Le Song. Stochastic training of graph convolutional networks with
variance reduction. In ICML , pp. 941–949, 2018a.
Jie Chen, Tengfei Ma, and Cao Xiao. Fastgcn: Fast learning with graph convolutional networks via
importance sampling. In International Conference on Learning Representations (ICLR) , 2018b.
Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, and Cho-Jui Hsieh. Cluster-gcn: An ef-
ﬁcient algorithm for training deep and large graph convolutional networks. CoRR , abs/1905.07953,
2019. URL http://arxiv.org/abs/1905.07953 .
9

## Page 10

Published as a conference paper at ICLR 2020
Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on
graphs with fast localized spectral ﬁltering. In Advances in Neural Information Processing Systems ,
pp. 3844–3852, 2016.
Matthias Fey. Just jump: Dynamic neighborhood aggregation in graph neural networks. CoRR ,
abs/1904.04849, 2019. URL http://arxiv.org/abs/1904.04849 .
Hongyang Gao, Zhengyang Wang, and Shuiwang Ji. Large-scale learnable graph convolutional
networks. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge
Discovery & Data Mining , KDD ’18, pp. 1416–1424, New York, NY , USA, 2018. ACM. ISBN
978-1-4503-5552-0.
Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In
Advances in Neural Information Processing Systems 30 , pp. 1024–1034. 2017.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
recognition. CoRR , abs/1512.03385, 2015. URL http://arxiv.org/abs/1512.03385 .
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation , 9(8):
1735–1780, 1997.
Pili Hu and Wing Cheong Lau. A survey and taxonomy of graph sampling. arXiv preprint
arXiv:1308.5865 , 2013.
Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected
convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern
recognition , pp. 4700–4708, 2017.
Wenbing Huang, Tong Zhang, Yu Rong, and Junzhou Huang. Adaptive sampling towards fast graph
representation learning. In Advances in Neural Information Processing Systems , pp. 4558–4567,
2018.
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint
arXiv:1412.6980 , 2014.
Thomas N. Kipf and Max Welling. Semi-supervised classiﬁcation with graph convolutional networks.
CoRR , abs/1609.02907, 2016. URL http://arxiv.org/abs/1609.02907 .
Johannes Klicpera, Aleksandar Bojchevski, and Stephan Günnemann. Personalized embedding propa-
gation: Combining neural networks on graphs with personalized pagerank. CoRR , abs/1810.05997,
2018. URL http://arxiv.org/abs/1810.05997 .
John Boaz Lee, Ryan A. Rossi, Xiangnan Kong, Sungchul Kim, Eunyee Koh, and Anup Rao. Higher-
order graph convolutional networks. CoRR , abs/1809.07697, 2018. URL http://arxiv.org/
abs/1809.07697 .
Jure Leskovec and Christos Faloutsos. Sampling from large graphs. In Proceedings of the 12th ACM
SIGKDD international conference on Knowledge discovery and data mining , pp. 631–636. ACM,
2006.
R. Li, J. X. Yu, L. Qin, R. Mao, and T. Jin. On random walk based graph sampling. In 2015 IEEE
31st International Conference on Data Engineering , pp. 927–938, April 2015. doi: 10.1109/ICDE.
2015.7113345.
Haonan Lu, Seth H. Huang, Tian Ye, and Xiuyan Guo. Graph star net for generalized multi-task
learning. CoRR , abs/1906.12330, 2019. URL http://arxiv.org/abs/1906.12330 .
Bruno Ribeiro and Don Towsley. Estimating and sampling graphs with multidimensional random
walks. In Proceedings of the 10th ACM SIGCOMM conference on Internet measurement , pp.
390–403. ACM, 2010.
Petar Veli ˇckovi ´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua
Bengio. Graph attention networks. arXiv preprint arXiv:1710.10903 , 2017.
10

## Page 11

Published as a conference paper at ICLR 2020
Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and Philip S. Yu. A
comprehensive survey on graph neural networks. CoRR , abs/1901.00596, 2019. URL http:
//arxiv.org/abs/1901.00596 .
Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, and Stefanie
Jegelka. Representation learning on graphs with jumping knowledge networks. arXiv preprint
arXiv:1806.03536 , 2018.
Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L. Hamilton, and Jure Leskovec.
Graph convolutional neural networks for web-scale recommender systems. In Proceedings of the
24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining , KDD ’18,
2018a. ISBN 978-1-4503-5552-0.
Rex Ying, Jiaxuan You, Christopher Morris, Xiang Ren, William L. Hamilton, and Jure Leskovec.
Hierarchical graph representation learning with differentiable pooling. In Proceedings of the 32Nd
International Conference on Neural Information Processing Systems , NIPS’18, pp. 4805–4815,
USA, 2018b. Curran Associates Inc. URL http://dl.acm.org/citation.cfm?id=
3327345.3327389 .
Hanqing Zeng and Viktor Prasanna. GraphACT: Accelerating GCN training on CPU-FPGA hetero-
geneous platforms. arXiv preprint arXiv:2001.02498 , 2019.
Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, Rajgopal Kannan, and Viktor K. Prasanna.
Accurate, efﬁcient and scalable graph embedding. CoRR , abs/1810.11899, 2018. URL http:
//arxiv.org/abs/1810.11899 .
Jiani Zhang, Xingjian Shi, Junyuan Xie, Hao Ma, Irwin King, and Dit-Yan Yeung. Gaan: Gated atten-
tion networks for learning on large and spatiotemporal graphs. arXiv preprint arXiv:1803.07294 ,
2018.
Zhenpeng Zhou. Graph convolutional networks for molecules. CoRR , abs/1706.09916, 2017. URL
http://arxiv.org/abs/1706.09916 .
A P ROOFS
Proof of Proposition 3.1. Under the condition that vis sampled in a subgraph:
E
(`+1)
v
=E X
u2VeAv;u
u;v~x(`)
u 1ujv!
=X
u2VeAv;u
u;v~x(`)
uE 
1ujv
=X
u2VeAv;u
u;v~x(`)
uP((u;v)sampledjvsampled )
=X
u2VeAv;u
u;v~x(`)
uP((u;v)sampled )
P(vsampled )
=X
u2VeAv;u
u;v~x(`)
upu;v
pv(5)
where the second equality is due to linearity of expectation, and the third equality (conditional edge
probability) is due to the initial condition that vis sampled in a subgraph.
It directly follows that, when u;v=pu;v
pv,
E
(`+1)
v
=X
u2VeAv;u~x(`)
u
11

## Page 12

Published as a conference paper at ICLR 2020
Proof of Theorem 3.2. Below, we use Cov()to denote covariance and Var()to denote variance.
For independent edge sampling as deﬁned in Section 3.3, Cov
1(`1)
e1; 1(`2)
e2
= 0;8e16=e2. And for
a full GCN on the subgraph, Cov
1(`1)
e; 1(`2)
e
=pe p2
e. To start the proof, we ﬁrst assume that
theb(`)
eis one dimensional (i.e., a scalar) and denote it by b(`)
e. Now,
Var() =X
e;` 
b(`)
e
pe!2
Var
1(`)
e
+ 2X
e;`1<`2b(`1)
eb(`2)
e
p2eCov
1(`1)
e; 1(`2)
e
=X
e;`
b(`)
e2
pe X
e;`
b(`)
e2
+ 2X
e;`1<`2b(`1)
eb(`2)
e
p2e 
pe p2
e
=X
eP
`b(`)
e2
pe X
e X
`b(`)
e!2
(6)
Let a given constant m=P
epebe the expected number of sampled edges. By Cauchy-Schwarz in-
equality:P
e(P
`b(`)
e)2
pem=P
eP
`b(`)
eppe2P
e ppe2P
e;`b(`)
e2
. The equality is achieved
whenP
`b(`)
eppe/ppe. i.e., variance is minimized when pe/P
`b(`)
e.
It directly follows that:
pe=m
P
e0P
`b(`)
e0X
`b(`)
e
For the multi-dimensional case of b(`)
e, following similar steps as above, it is easy to show that the
optimal edge probability to minimizeP
iVar(i)(whereiis the index for ’s dimensions) is:
pe=m
P
e0


P
`b(`)
e0







X
`b(`)
e





B S AMPLING ALGORITHM
Algorithm 2 lists the four graph samplers we have integrated into GraphSAINT . The naming of the
samplers follows that of Table 2. Note that the sampling parameters nandmspecify a budget rather
than the actual number of nodes and edges in the subgraph Gs. Since certain nodes or edges in the
training graphGmay be repeatedly sampled under a single invocation of the sampler, we often have
jVsj<n for node and MRW samplers, jVsj<2mfor edge sampler, and jVsj<rhfor RW sampler.
Also note that the edge sampler presented in Algorithm 2 is an approximate version of the independent
edge sampler deﬁned in Section 3.4. Complexity (excluding the subgraph induction step) of the
original version in Section 3.4 is O(jEj), while complexity of the approximate one is O(m). When
mjEj , the approximate version leads to identical accuracy as the original one, for a given m.
C D ETAILED EXPERIMENTAL SETUP
C.1 H ARDWARE SPECIFICATION AND ENVIRONMENT
We run our experiments on a single machine with Dual Intel Xeon CPUs (E5-2698 v4 @ 2.2Ghz),
one NVIDIA Tesla P100 GPU (16GB of HBM2 memory) and 512GB DDR4 memory. The code is
written in Python 3.6.8 (where the sampling part is written with Cython 0.29.2). We use Tensorﬂow
1.12.0 on CUDA 9.2 with CUDNN 7.2.1 to train the model on GPU. Since the subgraphs are sampled
independently, we run the sampler in parallel on 40 CPU cores.
12

## Page 13

Published as a conference paper at ICLR 2020
Algorithm 2 Graph sampling algorithms by GraphSAINT
Input: Training graphG(V;E); Sampling parameters: node budget n; edge budget m; number of
rootsr; random walk length h
Output: Sampled graphGs(Vs;Es)
1:function NODE(G,n) .Node sampler
2:P(v):=


eA:;v


2
=P
v02V


eA:;v0


2
3:Vs nnodes randomly sampled (with replacement) from Vaccording to P
4:Gs Node induced subgraph of GfromVs
5:end function
6:function EDGE(G,m) .Edge sampler (approximate version)
7:P((u;v)):=
1
deg(u)+1
deg(v)
=P
(u0;v0)2E
1
deg(u0)+1
deg(v0)
8:Es medges randomly sampled (with replacement) from Eaccording to P
9:Vs Set of nodes that are end-points of edges in Es
10:Gs Node induced subgraph of GfromVs
11:end function
12:function RW(G,r,h) .Random walk sampler
13:Vroot rroot nodes sampled uniformly at random (with replacement) from V
14:Vs V root
15: forv2V rootdo
16:u v
17: ford= 1tohdo
18: u Node sampled uniformly at random from u’s neighbor
19:Vs V s[fug
20: end for
21: end for
22:Gs Node induced subgraph of GfromVs
23:end function
24:function MRW(G,n,r) .Multi-dimensional random walk sampler
25:VFS rroot nodes sampled uniformly at random (with replacement) from V
26:Vs V FS
27: fori=r+ 1tondo
28: Selectu2V FSwith probability deg (u)=P
v2V FSdeg(v)
29:u0 Node randomly sampled from u’s neighbor
30:VFS (VFSnfug)[fu0g
31:Vs V s[fug
32: end for
33:Gs Node induced subgraph of GfromVs
34:end function
13

## Page 14

Published as a conference paper at ICLR 2020
10010110210310410510 610 410 2100
DegreeP(degreek)PPI
Flickr
Reddit
Yelp
Amazon
Figure 5: Degree Distribution
C.2 A DDITIONAL DATASET DETAILS
Here we present the detailed procedures to prepare the Flickr, Yelp and Amazon datasets.
The Flickr dataset originates from NUS-wide§. The SNAP website{collected Flickr data from
four different sources including NUS-wide, and generated an un-directed graph. One node in the
graph represents one image uploaded to Flickr. If two images share some common properties (e.g.,
same geographic location, same gallery, comments by the same user, etc.), there is an edge between
the nodes of these two images. We use as the node features the 500-dimensional bag-of-word
representation of the images provided by NUS-wide. For labels, we scan over the 81 tags of each
image and manually merged them to 7 classes. Each image belongs to one of the 7 classes.
The Yelp dataset is prepared from the raw json data of businesses, users and reviews provided in
the open challenge websitek. For nodes and edges, we scan the friend list of each user in the raw
json ﬁle of users. If two users are friends, we create an edge between them. We then ﬁlter out all
the reviews by each user and separate the reviews into words. Each review word is converted to a
300-dimensional vector using the Word2Vec model pre-trained on GoogleNews. The word vectors
of each node are added and normalized to serve as the node feature (i.e., xv). As for the node labels,
we scan the raw json ﬁle of businesses, and use the categories of the businesses reviewed by a user
vas the multi-class label of v.
For the Amazon dataset, a node is a product on the Amazon website and an edge (u;v)is created if
productsuandvare bought by the same customer. Each product contains text reviews (converted to
4-gram) from the buyer. We use SVD to reduce the dimensionality of the 4-gram representation to
200, and use the obtained vectors as the node feature. The labels represent the product categories
(e.g., books, movies, shoes).
Figure 5 shows the degree distribution of the ﬁve graphs. A point (k;p)in the plot means the
probability of a node having degree at least kisp.
C.3 A DDITIONAL DETAILS IN EXPERIMENTAL CONFIGURATION
Table 4 summarizes the URLs to download the baseline codes.
The optimizer for GraphSAINT and all baselines is Adam (Kingma & Ba, 2014). For all baselines
and datasets, we perform grid search on the hyperparameter space deﬁned by:
Hidden dimension: f128;256;512g
§http://lms.comp.nus.edu.sg/research/NUS-WIDE.htm
{https://snap.stanford.edu/data/web-flickr.html
khttps://www.yelp.com/dataset
https://code.google.com/archive/p/word2vec/
14

## Page 15

Published as a conference paper at ICLR 2020
Table 4: URLs and commit number to run baseline codes
Baseline URL Commit
Vanilla GCN github.com/williamleif/GraphSAGE a0fdef
GraphSAGE github.com/williamleif/GraphSAGE a0fdef
FastGCN github.com/matenure/FastGCN b8e6e6
S-GCN github.com/thu-ml/stochastic_gcn da7b78
AS-GCN github.com/huangwb/AS-GCN 5436ec
ClusterGCN github.com/google-research/google-research/tree/master/cluster_gcn 99021e
Table 5: Training conﬁguration of GraphSAINT for Table 2
Sampler DatasetTraining Sampling
Learning rate Dropout Node budget Edge budget Roots Walk length
NodePPI 0.01 0.0 6000 — — —
Flickr 0.01 0.2 8000 — — —
Reddit 0.01 0.1 8000 — — —
Yelp 0.01 0.1 5000 — — —
Amazon 0.01 0.1 4500 — — —
EdgePPI 0.01 0.1 — 4000 — —
Flickr 0.01 0.2 — 6000 — —
Reddit 0.01 0.1 — 6000 — —
Yelp 0.01 0.1 — 2500 — —
Amazon 0.01 0.1 — 2000 — —
RWPPI 0.01 0.1 — — 3000 2
Flickr 0.01 0.2 — — 6000 2
Reddit 0.01 0.1 — — 2000 4
Yelp 0.01 0.1 — — 1250 2
Amazon 0.01 0.1 — — 1500 2
MRWPPI 0.01 0.1 8000 — 2500 —
Flickr 0.01 0.2 12000 — 3000 —
Reddit 0.01 0.1 8000 — 1000 —
Yelp 0.01 0.1 2500 — 1000 —
Amazon 0.01 0.1 4500 — 1500 —
Dropout:f0:0;0:1;0:2;0:3g
Learning rate:f0:1;0:01;0:001;0:0001g
The hidden dimensions used for Table 2, Figure 2, Figure 3 and Figure 4 are: 512 for PPI, 256 for
Flickr, 128 for Reddit, 512 for Yelp and 512 for Amazon.
All methods terminate after a ﬁxed number of epochs based on convergence. We save the model
producing the highest validation set F1-micro score, and reload it to evaluate the test set accuracy.
For vanilla GCN and AS-GCN, we set the batch size to their default value 512. For GraphSAGE, we
use the mean aggregator with the default batch size 512. For S-GCN, we set the ﬂag -cv -cvd
(which stand for “control variate” and “control variate dropout”) with pre-computation of the ﬁrst
layer aggregation. According to the paper (Chen et al., 2018a), such pre-computation signiﬁcantly
reduces training time without affecting accuracy. For S-GCN, we use the default batch size 1000,
and for FastGCN, we use the default value 400. For ClusterGCN, its batch size is determined by two
parameters: the cluster size and the number of clusters per batch. We sweep the cluster size from
500to10000 with step 500, and the number of clusters per batch from f1;10;20;40gto determine
the optimal conﬁguration for each dataset / architecture. Considering that for ClusterGCN, the
cluster structure may be sensitive to the cluster size, and for FastGCN, the minibatch connectivity
may increase with the sample size, we present additional experimental results to reveal the relation
between accuracy and batch size in Appendix D.3.
15

## Page 16

Published as a conference paper at ICLR 2020
Table 6: Training conﬁguration of GraphSAINT for Table 3
Arch. Sampler DatasetTraining Sampling
Learning rate Dropout Node budget Edge budget Roots Walk length
2512 MRW PPI (large) 0.01 0.1 1500 — 300 —
52048 RW PPI (large) 0.01 0.1 — — 3000 2
2128 Edge Reddit 0.01 0.1 — 6000 — —
4128 Edge Reddit 0.01 0.2 — 11000 — —
Table 7: Training conﬁguration of GraphSAINT for Figure 4 (Reddit)
2-layer GAT- SAINT 4-layer GAT- SAINT 2-layer JK- SAINT 4-layer JK- SAINT
Hidden dimension 128 128 128 128
AttentionK 8 8 — —
AggregationL— — Concat. Concat.
SamplerRW RW Edge Edge
(root: 3000; length: 2) (root: 2000; length: 4) (budget: 6000) (budget: 11000)
Learning rate 0.01 0.01 0.01 0.01
Dropout 0.2 0.2 0.1 0.2
Conﬁguration of GraphSAINT to reproduce Table 2 results is shown in Table 5. Conﬁguration of
GraphSAINT to reproduce Table 3 results is shown in Table 6.
Below we describe the conﬁguration for Figure 4.
The major difference between a normal GCN and a JK-net (Xu et al., 2018) is that JK-net has an
additional ﬁnal layer that aggregates all the output hidden features of graph convolutional layers 1to
L. Mathematically, the additional aggregation layer outputs the ﬁnal embedding xJKas follows:
xJK= 
WT
JKLM
`=1x(`)
v!
(7)
where based on Xu et al. (2018),Lis the vector aggregation operator: max-pooling, concatenation
or LSTM (Hochreiter & Schmidhuber, 1997) based aggregation.
The graph attention of GAT (Veli ˇckovi ´c et al., 2017) calculates the edge weights for neighbor
aggregation by an additional neural network. With multi-head ( K) attention, the layer- (` 1)
features propagate to layer- (`)as follows:
x(`)
v=




K
k=10
@X
u2neighbor (v)k
u;vWkx(` 1)
v1
A (8)
wherekis the vector concatenation operation, and the coefﬁcient is calculated with the attention
weights akby:
k
u;v=LeakyReLU 
akT
WkxukWkxv
(9)
Note that the calculation is slightly different from the original equation in Veli ˇckovi ´c et al. (2017).
Namely, GAT- SAINT does not normalize by softmax across all neighbors of v. We make such
modiﬁcation since under the minibatch setting, node vdoes not see all its neighbors in the training
graph. The removal of softmax is also seen in the attention design of Huang et al. (2018). Note that
during the minibatch training, GAT- SAINT further applies another edge coefﬁcient on top of attention
for aggregator normalization.
Table 7 shows the conﬁguration of the GAT- SAINT and JK- SAINT curves in Figure 4.
16

## Page 17

Published as a conference paper at ICLR 2020
2 3 4 5 602468
GCN depthNormalized training timeGraphSAINT : Reddit S-GCN: Reddit
GraphSAINT : Yelp S-GCN: Yelp
Figure 6: Comparison of training efﬁciency
PPI
Flickr
Reddit
Yelp
Amazon00:511:5Fraction of training timeNode Edge RW MRW
Figure 7: Fraction of training time on sampling
D A DDITIONAL EXPERIMENTS
D.1 T RAINING EFFICIENCY ON DEEPMODELS
We evaluate the training efﬁciency for deeper GCNs. We only compare with S-GCN, since implemen-
tations for other layer sampling based methods have not yet supported arbitrary model depth. The
batch size and hidden dimension are the same as Table 2. On the two large graphs (Reddit and Yelp),
we increase the number of layers and measure the average time per minibatch execution. In Figure
6, training cost of GraphSAINT is approximately linear with GCN depth. Training cost of S-GCN
grows dramatically when increasing the depth. This reﬂects the “neighbor explosion” phenomenon
(even though the expansion factor of S-GCN is just 2). On Yelp, S-GCN gives “out-of-memory” error
for models beyond 5 layers.
D.2 C OST OF SAMPLING AND PRE-PROCESSING
Cost of graph samplers of GraphSAINT Graph sampling introduces little training overhead. Let
tsbe the average time to sample one subgraph on a multi-core machine. Let ttbe the average
time to perform the forward and backward propagation on one minibatch on GPU. Figure 7 shows
the ratiots=ttfor various datasets. The parameters of the samplers are the same as Table 2. For
Node, Edge and RW samplers, we observe that time to sample one subgraph is in most cases less
than 25% of the training time. The MRW sampler is more expensive to execute. Regarding the
complete pre-processing procedure, we repeatedly run the sampler for N= 50jVj=jVsjtimes
before training, to estimate the node and edge probability as discussed in Section 3.2 (where jVsjis
the average subgraph size). These sampled subgraphs are reused as training minibatches. Thus, if
training runs for more than Niterations, the pre-processing is nearly zero-cost. Under the setting
of Table 2, pre-processing on PPI and Yelp and Amazon does not incur any overhead in training
time. Pre-processing on Flickr and Reddit (with RW sampler) takes less than 40% and 15% of their
corresponding total training time.
Cost of layers sampler of AS-GCN AS-GCN uses an additional neural network to estimate the
conditional sampling probability for the previous layer. For a node valready sampled in layer `,
features of layer- (` 1)corresponding to all v’s neighbors need to be fed to the sampling neural
network to obtain the node probability. For sake of analysis, assume the sampling network is a single
layer MLP, whose weight WMLPhas the same shape as the GCN weights W(`). Then we can show,
for aL-layer GCN on a degree- dgraph, per epoch training complexity of AS-GCN is approximately

= (dL)=PL 1
`=0d`times that of vanilla GCN. For L= 2, we have
2. This explains the
observation that AS-GCN is slower than vanilla GCN in Figure 2. Additional, Table 8 shows the
training time breakdown for AS-GCN. Clearly, its sampler is much more expensive than the graph
sampler of GraphSAINT .
17

## Page 18

Published as a conference paper at ICLR 2020
Table 8: Per epoch training time breakdown for AS-GCN
Dataset Sampling time (sec)Forward / Backward
propagation time (sec)
PPI 1.1 0.2
Flickr 5.3 1.1
Reddit 20.7 3.5
Cost of clustering of ClusterGCN ClusterGCN uses the highly optimized METIS softwareyyto
perform clustering. Table 9 summarizes the time to obtain the clusters for the ﬁve graphs. On the
large and dense Amazon graph, the cost of clustering increase dramatically. The pre-processing
time of ClusterGCN on Amazon is more than 4of the total training time. On the other hand, the
sampling cost of GraphSAINT does not increase signiﬁcantly for large graphs (see Figure 7).
Table 9: Clustering time of ClusterGCN
PPI Flickr Reddit Yelp Amazon
Time (sec) 2.2 11.6 40.0 106.7 2254.2
Taking into account the pre-processing time, sampling time and training time altogether, we sum-
marize the total convergence time of GraphSAINT and ClusterGCN in Table 10 (corresponding to
Table 2 conﬁguration). On graphs that are large and dense (e.g., Amazon), GraphSAINT achieves
signiﬁcantly faster convergence. Note that both the sampling of GraphSAINT and clustering of
ClusterGCN can be performed ofﬂine.
Table 10: Comparison of total convergence time (pre-processing + sampling + training, unit: second)
PPI Flickr Reddit Yelp Amazon
GraphSAINT -Edge 91.0 7.0 16.6 273.9 401.0
GraphSAINT -RW 103.6 7.5 17.2 310.1 425.6
ClusterGCN 163.2 12.9 55.3 256.0 2804.8
D.3 E FFECT OF BATCH SIZE
Table 11 shows the change of test set accuracy with batch sizes. For each row of Table 11, we ﬁx the
batch size, tune the other hyperparameters according to Appendix C.3, and report the highest test set
accuracy achieved. For GraphSAGE, S-GCN and AS-GCN, their default batch sizes (512,1000 and
512, respectively) lead to the highest accuracy on all datasets. For FastGCN, increasing the default
batch size (from 400 to 4000) leads to noticeable accuracy improvement. For ClusterGCN, different
datasets correspond to different optimal batch sizes. Note that the accuracy in Section 5.1 is already
tuned by identifying the optimal batch size on a per graph basis.
For FastGCN, intuitively, increasing batch size may help with accuracy improvement since the
minibatches may become better connected. Such intuition is veriﬁed by the rows of 400 and 2000.
However, increasing the batch size from 2000 to 4000 does not further improve accuracy signiﬁcantly.
For ClusterGCN, the optimal batch size depends on the cluster structure of the training graph. For
PPI, small batches are better, while for Amazon, batch size does not have signiﬁcant impact on
accuracy. For GraphSAGE, overly large batches may have negative impact on accuracy due to
neighbor explosion. Approximately, GraphSAGE expand 10more neighbors per layer. For a
2-layer GCN, a size 2103minibatch would then require the support of 2105nodes from the
yyhttp://glaros.dtc.umn.edu/gkhome/metis/metis/download
Default batch size
{The training does not converge.
zThe codes throw runtime error on the large datasets (Yelp or Amazon).
18

## Page 19

Published as a conference paper at ICLR 2020
Table 11: Test set F1-micro for the baselines under various batch sizes
Method Batch size PPI Flickr Reddit Yelp Amazon
GraphSAGE256 0.600 0.474 0.934 0.563 0.428
5120.637 0.501 0.953 0.634 0.758
1024 0.610 0.482 0.935 0.632 0.705
2048 0.625 0.374 0.936 0.563 0.447
FastGCN4000.513 0.504 0.924 0.265 0.174
2000 0.561 0.506 0.934 0.255 0.196
4000 0.564 0.507 0.934 0.260 0.195
S-GCN500 0.519 0.462 —{—{—z
10000.963 0.482 0.964 0.640 —z
2000 0.646 0.482 0.949 0.614 —z
4000 0.804 0.482 0.949 0.594 —z
8000 0.694 0.481 0.950 0.613 —z
AS-GCN256 0.682 0.504 0.950 —z—z
5120.687 0.504 0.958 —z—z
1024 0.687 0.502 0.951 —z—z
2048 0.670 0.502 0.952 —z—z
ClusterGCN500 0.875 0.481 0.942 0.604 0.752
1000 0.831 0.478 0.947 0.602 0.756
1500 0.865 0.480 0.954 0.602 0.752
2000 0.828 0.469 0.954 0.609 0.759
2500 0.849 0.476 0.954 0.598 0.745
3000 0.840 0.473 0.954 0.607 0.754
3500 0.846 0.473 0.952 0.602 0.754
4000 0.853 0.472 0.949 0.605 0.756
input layer. Note that the full training graph size of Reddit is just around 1:5105. Thus, no matter
which nodes are sampled in the output layer, GraphSAGE would almost always propagate features
within the full training graph for initial layers. We suspect this would lead to difﬁculties in learning.
For S-GCN, with batch size of 500, it fails to learn properly on Reddit and Yelp. The accuracy
ﬂuctuates in a region of very low value, even after appropriate hyperparameter tuning. For AS-GCN,
its accuracy is not sensitive to the batch size, since AS-GCN addresses neighbor explosion and also
ensures good inter-layer connectivity within the minibatch.
19