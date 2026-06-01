# A Generalization of Transformer Networks to Graphs

## Page 1

A Generalization of Transformer Networks to Graphs
Vijay Prakash Dwivedi,¶Xavier Bresson¶
¶School of Computer Science and Engineering, Nanyang Technological University, Singapore
vijaypra001@e.ntu.edu.sg ,xbresson@ntu.edu.sg
Abstract
We propose a generalization of transformer neural network
architecture for arbitrary graphs. The original transformer
was designed for Natural Language Processing (NLP), which
operates on fully connected graphs representing all connec-
tions between the words in a sequence. Such architecture does
not leverage the graph connectivity inductive bias, and can
perform poorly when the graph topology is important and
has not been encoded into the node features. We introduce a
graph transformer with four new properties compared to the
standard model. First, the attention mechanism is a function
of the neighborhood connectivity for each node in the graph.
Second, the positional encoding is represented by the Lapla-
cian eigenvectors, which naturally generalize the sinusoidal
positional encodings often used in NLP. Third, the layer
normalization is replaced by a batch normalization layer,
which provides faster training and better generalization per-
formance. Finally, the architecture is extended to edge feature
representation, which can be critical to tasks s.a. chemistry
(bond type) or link prediction (entity relationship in knowl-
edge graphs). Numerical experiments on a graph benchmark
demonstrate the performance of the proposed graph trans-
former architecture. This work closes the gap between the
original transformer, which was designed for the limited case
of line graphs, and graph neural networks, that can work with
arbitrary graphs. As our architecture is simple and generic,
we believe it can be used as a black box for future applica-
tions that wish to consider transformer and graphs.1
1 Introduction
There has been a tremendous success in the ﬁeld of nat-
ural language processing (NLP) since the development of
Transformers (Vaswani et al. 2017) which are currently the
best performing neural network architectures for handling
long-term sequential datasets such as sentences in NLP.
This is achieved by the use of attention mechanism (Bah-
danau, Cho, and Bengio 2014) where a word in a sentence
attends to each other word and combines the received in-
formation to generate its abstract feature representations.
From a perspective of message-passing paradigm (Gilmer
AAAI’21 Workshop on Deep Learning on Graphs: Methods and
Applications (DLG-AAAI’21). Copyright © 2021, Association for
the Advancement of Artiﬁcial Intelligence (www.aaai.org). All
rights reserved.
1https://github.com/graphdeeplearning/graphtransformer.et al. 2017) in graph neural networks (GNNs), this process
of learning word feature representations by combining fea-
ture information from other words in a sentence can alter-
natively be viewed as a case of a GNN applied on a fully
connected graph of words (Joshi 2020). Transformers based
models have led to state-of-the-art performance on several
NLP applications (Devlin et al. 2018; Radford et al. 2018;
Brown et al. 2020). On the other hand, graph neural net-
works (GNNs) are shown to be the most effective neural
network architectures on graph datasets and have achieved
signiﬁcant success on a wide range of applications, such
as in knowledge graphs (Schlichtkrull et al. 2018; Chami
et al. 2020), in social sciences (Monti et al. 2019), in physics
(Cranmer et al. 2019; Sanchez-Gonzalez et al. 2020), etc.
In particular, GNNs exploit the given arbitrary graph struc-
ture while learning the feature representations for nodes and
edges and eventually the learned representations are used for
downstream tasks. In this work, we explore inductive biases
at the convergence of these two active research areas in deep
learning towards presenting an improved version of Graph
Transformer (see Figure 1) which extends the key design
components of the NLP transformers to arbitrary graphs.
1.1 Related Work
As a preliminary, we highlight the most recent research
works which attempt to develop graph transformers (Li et al.
2019; Nguyen, Nguyen, and Phung 2019; Zhang et al. 2020)
with few focused on specialized cases such as on heteroge-
neous graphs, temporal networks, generative modeling, etc.
(Yun et al. 2019; Xu, Joshi, and Bresson 2019; Hu et al.
2020; Zhou et al. 2020).
The model proposed in Li et al. (2019) employs attention
to all graph nodes instead of a node’s local neighbors for
the purpose of capturing global information. This limits the
efﬁcient exploitation of sparsity which we show is a good in-
ductive bias for learning on graph datasets. For the purpose
of global information, we argue that there are other ways to
incorporate the same instead of letting go sparsity and local
contexts. For example, the use of graph-speciﬁc positional
features (Zhang et al. 2020), or node Laplacian position
eigenvectors (Belkin and Niyogi 2003; Dwivedi et al. 2020),
or relative learnable positional information (You, Ying, and
Leskovec 2019), virtual nodes (Li et al. 2015), etc. Zhang
et al. (2020) propose Graph-BERT with an emphasis on pre-arXiv:2012.09699v2  [cs.LG]  24 Jan 2021

## Page 2

Add&NormAdd&Norm
Add&NormAdd&Norm
Add&NormAdd&Norm
Graph T ransformer LayerGraph T ransformer Layer
with edge features+ +
Laplacian	EigV ecs	as
Positional	Encoding++Figure 1: Block Diagram of Graph Transformer with Laplacian Eigvectors ( ) used as positional encoding (LapPE). LapPE
is added to input node embeddings before passing the features to the ﬁrst layer. Left: Graph Transformer operating on node
embeddings only to compute attention scores; Right : Graph Transformer with edge features with designated feature pipeline to
maintain layer wise edge representations. In this extension, the available edge attributes in a graph is used to explicitly modify
the corresponding pairwise attention scores.
training and parallelized learning using a subgraph batch-
ing scheme that creates ﬁxed-size linkless subgraphs to be
passed to the model instead of the original graph. Graph-
BERT employs a combination of several positional encod-
ing schemes to capture absolute node structural and rela-
tive node positional information. Since the original graph is
not used directly in Graph-BERT and the subgraphs do not
have edges between the nodes ( i.e., linkless), the proposed
combination of positional encodings attempts at retaining
the original graph structure information in the nodes. We
perform detailed analysis of Graph-BERT positional encod-
ing schemes, along with experimental comparison with the
model we present in this paper in Section 4.1.
Yun et al. (2019) developed Graph Transformer Networks
(GTN) to learn on heterogeneous graphs with a target to
transform a given heterogeneous graph into a meta-path
based graph and then perform convolution. Notably, their
focus behind the use of attention framework is for inter-preting the generated meta-paths. There is another trans-
former based approach developed for heterogeneous in-
formation networks, namely Heterogeneous Graph Trans-
former (HGT) by Hu et al. (2020). Apart from its ability
of handling arbitrary number of node and edge types, HGT
also captures the dynamics of information ﬂow in the hetero-
geneous graphs in the form of relative temporal positional
encoding which is based on the timestamp differences of the
central node and the message-passing nodes. Furthermore,
Zhou et al. (2020) proposed a transformer based generative
model which generates temporal graphs by directly learn-
ing from dynamic information in networks. The architecture
presented in Nguyen, Nguyen, and Phung (2019) somewhat
proceeds along our goal to develop graph transformer for ar-
bitrary homogeneous graphs with a coordinate embedding
based positional encoding scheme. However, their experi-
ments show that the coordinate embeddings are not universal
in performance and only helps in a couple of unsupervised

## Page 3

learning experiments among all evaluations.
1.2 Contributions
Overall, we ﬁnd that the most fruitful ideas from the trans-
formers literature in NLP can be applied in a more efﬁcient
way and posit that sparsity and positional encodings are two
keyaspects in the development of a Graph Transformer. As
opposed to designing a best performing model for speciﬁc
graph tasks, our work attempts for a generic, competitive
transformer model which draws ideas together from the do-
mains of NLP and GNNs. For an overview, this paper brings
the following contributions:
• We put forward a generalization of transformer networks
to homogeneous graphs of arbitrary structure, namely
Graph Transformer, and an extended version of Graph
Transformer with edge features that allows the usage of
explicit domain information as edge features.
• Our method includes an elegant way to fuse node po-
sitional features using Laplacian eigenvectors for graph
datasets, inspired from the heavy usage of positional en-
codings in NLP transformer models and recent research
on node positional features in GNNs. The comparison
with literature shows Laplacian eigenvectors to be well-
placed than any existing approaches to encode node posi-
tional information for arbitrary homogeneous graphs.
• Our experiments demonstrate that the proposed model
surpasses baseline isotropic and anisotropic GNNs. The
architecture simultaneously emerges as a better attention
based GNN baseline as well as a simple and effective
Transformer network baseline for graph datasets for fu-
ture research at the intersection of attention and graphs.
2 Proposed Architecture
As stated earlier, we take into account two key aspects to
develop Graph Transformers – sparsity and positional en-
codings which should ideally be used in the best possible
way for learning on graph datasets. We ﬁrst discuss the mo-
tivations behind these using a transition from NLP to graphs,
and then introduce the architecture proposed.
2.1 On Graph Sparsity
In NLP transformers, a sentence is treated as a fully con-
nected graph and this choice can be justiﬁed for two reasons
– a) First, it is difﬁcult to ﬁnd meaningful sparse interactions
or connections among the words in a sentence. For instance,
the dependency of a word in a sentence on another word can
vary with context, perspective of a user and speciﬁc applica-
tion. There can be numerous plausible ground truth connec-
tions among words in a sentence and therefore, text datasets
of sentences do not have explicit word interactions available.
It thereby makes sense to have each word attending to each
other word in a sentence, as followed by the Transformer
architecture (Vaswani et al. 2017). – b) Next, the so-called
graph considered in an NLP transformer often has less than
tens or hundreds of nodes ( i.e.sentences are often less than
tens or hundreds of words). This makes for computationallyfeasibility and large transformer models can be trained on
such fully connected graphs of words.
In case of actual graph datasets, graphs have arbitrary con-
nectivity structure available depending on the domain and
target of application, and have node sizes in ranges of up
to millions, or billions. The available structure presents us
with a rich source of information to exploit as an inductive
bias in a neural network, whereas the node sizes practically
makes it impossible to have a fully connected graph for such
datasets. On these accounts, it is ideal and practical to have
a Graph Transformer where a node attends to local node
neighbors, same as in GNNs (Defferrard, Bresson, and Van-
dergheynst 2016; Kipf and Welling 2017; Monti et al. 2017;
Gilmer et al. 2017; Veli ˇckovi ´c et al. 2018; Bresson and Lau-
rent 2017; Xu et al. 2019).
2.2 On Positional Encodings
In NLP, transformer based models are, in most cases, sup-
plied with a positional encoding for each word. This is criti-
cal to ensure unique representation for each word, and even-
tually preserve distance information. For graphs, the design
of unique node positions is challenging as there are sym-
metries which prevent canonical node positional informa-
tion (Murphy et al. 2019). In fact, most of the GNNs which
are trained on graph datasets learn structural node informa-
tion that are invariant to the node position (Srinivasan and
Ribeiro 2020). This is a critical reason why simple attention
based models, such as GAT (Veli ˇckovi ´c et al. 2018), where
the attention is a function of local neighborhood connectiv-
ity, instead full-graph connectivity, do not seem to achieve
competitive performance on graph datasets. The issue of po-
sitional embeddings has been explored in recent GNN works
(Murphy et al. 2019; You, Ying, and Leskovec 2019; Srini-
vasan and Ribeiro 2020; Dwivedi et al. 2020; Li et al. 2020)
with a goal to learn both structural and positional features.
In particular, Dwivedi et al. (2020) make the use of avail-
able graph structure to pre-compute Laplacian eigenvectors
(Belkin and Niyogi 2003) and use them as node positional
information. Since Laplacian PEs are generalization of the
PE used in the original transformers (Vaswani et al. 2017)
to graphs and these better help encode distance-aware in-
formation ( i.e., nearby nodes have similar positional fea-
tures and farther nodes have dissimilar positional features),
we use Laplacian eigenvectors as PE in Graph Transformer.
Although these eigenvectors have multiplicity occuring due
to the arbitrary sign of eigenvectors, we randomly ﬂip the
sign of the eigenvectors during training, following Dwivedi
et al. (2020).We pre-compute the Laplacian eigenvectors of
all graphs in the dataset. Eigenvectors are deﬁned via the
factorization of the graph Laplacian matrix;
 = I D 1=2AD 1=2=UTU; (1)
whereAis thennadjacency matrix, Dis the degree
matrix, and ,Ucorrespond to the eigenvalues and eigen-
vectors respectively. We use the ksmallest non-trivial eigen-
vectors of a node as its positional encoding and denote by i
for nodei. Finally, we refer to Section 4.1 for a comparison
of Laplacian PE with existing Graph-BERT PEs.

## Page 4

2.3 Graph Transformer Architecture
We now introduce the Graph Transformer Layer and Graph
Transformer Layer with edge features. The layer archi-
tecture is illustrated in Figure 1. The ﬁrst model is de-
signed for graphs which do not have explicit edge attributes,
whereas the second model maintains a designated edge fea-
ture pipeline to incorporate the available edge information
and maintain their abstract representations at every layer.
Input First of all, we prepare the input node and edge em-
beddings to be passed to the Graph Transformer Layer. For
a graphGwith node features i2Rdn1for each node i
and edge features ij2Rde1for each edge between node
iand nodej, the input node features iand edge features
ijare passed via a linear projection to embed these to d-
dimensional hidden features h0
iande0
ij.
^h0
i=A0i+a0;e0
ij=B0ij+b0; (2)
whereA02Rddn,B02Rddeanda0;b02Rdare the
parameters of the linear projection layers. We now embed
the pre-computed node positional encodings of dim kvia a
linear projection and add to the node features ^h0
i.
0
i=C0i+c0;h0
i=^h0
i+0
i; (3)
whereC02Rdkandc02Rd. Note that the Laplacian po-
sitional encodings are only added to the node features at the
input layer and not during intermediate Graph Transformer
layers.
Graph Transformer Layer The Graph Transformer is
closely the same transformer architecture initially proposed
in (Vaswani et al. 2017), see Figure 1 (Left). We now pro-
ceed to deﬁne the node update equations for a layer `.
^h`+1
i=O`
hHn
k=1X
j2Niwk;`
ijVk;`h`
j
; (4)
where,wk;`
ij=softmax jQk;`h`
iKk;`h`
jpdk
; (5)
andQk;`;Kk;`;Vk;`2Rdkd,O`
h2Rdd,k= 1toHde-
notes the number of attention heads, and kdenotes concate-
nation. For numerical stability, the outputs after taking expo-
nents of the terms inside softmax is clamped to a value be-
tween 5to+5. The attention outputs ^h`+1
iare then passed
to a Feed Forward Network (FFN) preceded and succeeded
by residual connections and normalization layers, as:
^^h`+1
i =Norm
h`
i+^h`+1
i
; (6)
^^^h`+1
i =W`
2ReLU (W`
1^^h`+1
i); (7)
h`+1
i =Norm^^h`+1
i+^^^h`+1
i
; (8)
whereW`
1;2R2dd,W`
2;2Rd2d,^^h`+1
i;^^^h`+1
idenote in-
termediate representations, and Norm can either be Layer-
Norm(Ba, Kiros, and Hinton 2016) or BatchNorm (Ioffe and
Szegedy 2015). The bias terms are omitted for clarity of pre-
sentation.Graph Transformer Layer with edge features The
Graph Transformer with edge features is designed for bet-
ter utilization of rich feature information available in several
graph datasets in the form of edge attributes. See Figure 1
(Right) for a reference to the building block of a layer. Since
our objective remains to better use the edge features which
are pairwise scores corresponding to a node pair, we tie these
available edge features to implicit edge scores computed by
pairwise attention. In other words, say an intermediate at-
tention score before softmax, ^wij, is computed when a node
iattends to node jafter the multiplication of query andkey
feature projections, see the expression inside the brackets in
Equation 5. Let us treat this score ^wijas implicit information
about the edge < i;j > . We now try to inject the available
edge information for the edge < i;j > and improve the al-
ready computed implicit attention score ^wij. It is done by
simply multiplying the two values ^wijandeij, see Equa-
tion 12. This kind of information injection is not seen to be
explored much, or applied in NLP Transformers as there is
usually no available feature information between two words.
However, in graph datasets such as molecular graphs, or
social media graphs, there is often some feature informa-
tion available on the edge interactions and it becomes nat-
ural to design an architecture to use this information while
learning. For the edges, we also maintain a designated node-
symmetric edge feature representation pipeline for propagat-
ing edge attributes from one layer to another, see Figure 1.
We now proceed to deﬁne the layer update equations for a
layer`.
^h`+1
i =O`
hHn
k=1X
j2Niwk;`
ijVk;`h`
j
; (9)
^e`+1
ij =O`
eHn
k=1
^wk;`
ij
;where, (10)
wk;`
ij =softmax j( ^wk;`
ij); (11)
^wk;`
ij =Qk;`h`
iKk;`h`
jpdk
Ek;`e`
ij;(12)
andQk;`;Kk;`;Vk;`;Ek;`2Rdkd,O`
h;O`
e2Rdd,
k= 1toHdenotes the number of attention head, and kde-
notes concatenation. For numerical stability, the outputs af-
ter taking exponents of the terms inside softmax is clamped
to a value between  5to+5. The outputs ^h`+1
iand^e`+1
ijare
then passed to separate Feed Forward Networks preceded
and succeeded by residual connections and normalization
layers, as:
^^h`+1
i =Norm
h`
i+^h`+1
i
; (13)
^^^h`+1
i =W`
h;2ReLU (W`
h;1^^h`+1
i); (14)
h`+1
i =Norm^^h`+1
i+^^^h`+1
i
; (15)

## Page 5

whereW`
h;1;2R2dd,W`
h;2;2Rd2d,^^h`+1
i;^^^h`+1
idenote
intermediate representations,
^^e`+1
ij =Norm
e`
ij+ ^e`+1
ij
; (16)
^^^e`+1
ij =W`
e;2ReLU (W`
e;1^^e`+1
ij); (17)
e`+1
ij =Norm
^^e`+1
ij+^^^e`+1
ij
; (18)
whereW`
e;1;2R2dd,W`
e;2;2Rd2d,^^e`+1
ij;^^^e`+1
ijdenote
intermediate representations.
Task based MLP Layers The node representations ob-
tained at the ﬁnal layer of Graph Transformer are passed
to a task based MLP network for computing task-dependent
outputs, which are then fed to a loss function to train the
parameters of the model. The formal deﬁnitions of the task
based layers that we use can be found in Appendix A.1.
3 Numerical Experiments
We evaluate the performance of proposed Graph Trans-
former on three benchmark graph datasets– ZINC (Irwin
et al. 2012), PATTERN and CLUSTER (Abbe 2017) from
a recent GNN benchmark (Dwivedi et al. 2020).
ZINC, Graph Regression ZINC (Irwin et al. 2012) is a
molecular dataset with the task of graph property regres-
sion for constrained solubility. Each ZINC molecule is rep-
resented as a graph of atoms as nodes and bonds as edges.
Since this dataset have rich feature information in terms of
bonds as edge attributes, we use the ‘Graph Transformer
with edge features’ for this task. We use the 12K subset of
the data as in Dwivedi et al. (2020).
PATTERN, Node Classiﬁcation PATTERN is a node
classiﬁcation dataset generated using the Stochastic Block
Models (SBM) (Abbe 2017). The task is classify the nodes
into 2 communities. PATTERN graphs do not have explicit
edge features and hence we use the simple ‘Graph Trans-
former’ for this task. The size of this dataset is 14K graphs.
CLUSTER, Node Classiﬁcation CLUSTER is also a
synthetically generated dataset using SBM model. The task
is to assign a cluster label to each node. There are total 6
cluster labels. Similar to PATTERN, CLUSTER graphs do
not have explicit edge features and hence we use the simple
‘Graph Transformer’ for this task. The size of this dataset is
12K graphs. We refer the readers to (Dwivedi et al. 2020)
for additional information, inlcuding preparation, of these
datasets.
Model Conﬁgurations For experiments, we follow the
benchmarking protocol introduced in Dwivedi et al. (2020)
based on PyTorch (Paszke et al. 2019) and DGL (Wang et al.
2019). We use 10 layers of Graph Transformer layers with
each layer having 8 attention heads and arbitrary hidden di-
mensions such that the total number of trainable parameters
is in the range of 500k. We use learning rate decay strategy
to train the models where the training stops at a point whenthe learning rate reaches to a value of 110 6. We run each
experiment with 4 different seeds and report the mean and
average performance measure of the 4 runs. The results are
reported in Table 1 and comparison in Table 2.
4 Analysis and Discussion
We now present the analysis of our experiments on the pro-
posed Graph Transformer Architecture, see Tables 1 and 2.
• The generalization of transformer network on graphs is
best when Laplacian PE are used for node positions and
Batch Normalization is selected instead of Layer Normal-
ization. For all three benchmark datasets, the experiments
score the highest performance in this setting, see Table 1.
• The proposed architecture performs signiﬁcantly better
than baseline isotropic and anisotropic GNNs (GCN and
GAT respectively), and helps close the gap between the
original transformer and transformer for graphs. Notably,
our architecture emerges as a fresh and improved attention
based GNN baseline surpassing GAT (see Table 2), which
employs multi-headed attention inspired by the original
transformer (Vaswani et al. 2017) and have been often
used in the literature as a baseline for attention-based
GNN models.
• As expected, sparse graph connectivity is a critical in-
ductive bias for datasets with arbitrary graph structure, as
demonstrated by comparing sparse vs. full graph experi-
ments.
• Our proposed extension of Graph Transformer with edge
features reaches close to the best performing GNN,
i.e.,GatedGCN, on ZINC. This architecture speciﬁcally
brings exciting promise to datasets where domain infor-
mation along pairwise interactions can be leveraged for
maximum learning performance.
4.1 Comparison to PEs used in Graph-BERT
In addition to the reasons underscored in Sections 1.1 and
2.2, we demonstrate the usefulness of Laplacian eigenvec-
tors as a suitable candidate PE for Graph Transformer in
this section, by its comparison with different PE schemes ap-
plied in Graph-BERT (Zhang et al. 2020).2In Graph-BERT,
which operates on ﬁxed size sampled subgraphs, a node at-
tends to every other node in a subgraph. For a given graph
G= (V;E)withVnodes andEedges, a subgraph giof size
k+ 1is created for every node iin the graph, which means
the original single graph Gis converted toVsubgraphs. For
a subgraphgicorresponding to node ui, thekother nodes
are the ones which have the top kintimacy scores with node
uibased on a pre-computed intimacy matrix that maps every
edge in the graphGto an intimacy score. While the sampling
is great for parallelization and efﬁciency, the original graph
structure is not directly used in the layers. Graph-BERT uses
2Note that we do not perform empirical comparison with other
PEs in Graph Transformer literature except Graph-BERT, because
of two reasons: i) Some existing Graph Transformer methods do
not use PEs, ii) If PEs are used, they are usually specialised; for
instance, Relative Temporal Encoding (RTE) for encoding dynamic
information in heterogeneous graphs in (Hu et al. 2020).

## Page 6

Sparse Graph Full Graph
Dataset LapPE L #Param Test Perf.s.d. Train Perf.s.d. #Epoch Epoch/Total Test Perf.s.d. Train Perf.s.d. #Epoch Epoch/Total
Batch Norm: False ; Layer Norm: True
ZINCx 10 588353 0.2780.018 0.0270.004 274.75 26.87s/2.06hr 0.7410.008 0.4310.013 196.75 37.64s/2.09hr
X 10 588929 0.2840.012 0.0310.006 263.00 26.64s/1.98hr 0.7350.006 0.4420.031 196.75 31.50s/1.77hr
CLUSTERx 10 523146 70.8790.295 86.1740.365 128.50 202.68s/7.32hr 19.5962.071 19.5702.053 103.00 512.34s/15.15hr
X 10 524026 70.6490.250 86.3950.528 130.75 200.55s/7.43hr 27.0913.920 26.9163.764 139.50 565.13s/22.37hr
PATTERNx 10 522742 73.14013.633 73.07013.589 184.25 276.66s/13.75hr 50.8540.111 50.9060.005 108.00 540.85s/16.77hr
X 10 522982 71.00511.831 71.12511.977 192.50 294.91s/14.79hr 56.4823.549 56.5653.546 124.50 637.55s/22.69hr
Batch Norm: True ; Layer Norm: False
ZINCx 10 588353 0.2640.008 0.0480.006 321.50 28.01s/2.52hr 0.7240.013 0.5180.013 192.25 50.27s/2.72hr
X 10 588929 0.2260.014 0.0590.011 287.50 27.78s/2.25hr 0.5980.049 0.3390.123 273.50 45.26s/3.50hr
CLUSTERx 10 523146 72.1390.405 85.8570.555 121.75 200.85s/6.88hr 21.0920.134 21.0710.037 100.25 595.24s/17.10hr
X 10 524026 73.1690.622 86.5850.905 126.50 201.06s/7.20hr 27.1218.471 27.1928.485 133.75 552.06s/20.72hr
PATTERNx 10 522742 83.9490.303 83.8640.489 236.50 299.54s/19.71hr 50.8890.069 50.8730.039 104.50 621.33s/17.53hr
X 10 522982 84.8080.068 86.5590.116 145.25 309.95s/12.67hr 54.9413.739 54.9153.769 117.75 683.53s/22.77hr
Table 1: Results of GraphTransformer (GT) on all datasets. Performance Measure for ZINC is MAE, for PATTERN and CLUS-
TER is Acc. Results (higher is better for all except ZINC) are averaged over 4 runs with 4 different seeds. Bold : the best
performing model for each dataset. We perform each experiment with given graphs (Sparse Graph) and(Full Graph) in
which we create full connections among all nodes; For ZINC full graphs, edge features are discarded given our motive of the
full graph experiments without any sparse structure information.
Model ZINC CLUSTER PATTERN
GNN BASELINE SCORES from (Dwivedi et al. 2020)
GCN 0.3670.011 68.4980.976 71.8920.334
GAT 0.3840.007 70.5870.447 78.2710.186
GatedGCN 0.2140.013 76.0820.196 86.5080.085
OUR RESULTS
GT (Ours) 0.2260.014 73.1690.622 84.8080.068
Table 2: Comparison of our best performing scores (from
Table 1) on each dataset against the GNN baselines (GCN
(Kipf and Welling 2017), GAT (Veli ˇckovi ´c et al. 2018), Gat-
edGCN(Bresson and Laurent 2017)) of 500k model param-
eters. Note: Only GatedGCN and GT models use the avail-
able edge attributes in ZINC.
a combination of node PE schemes to inform the model on
node structural, positional, and distance information from
original graph– i) Intimacy based relative PE, ii) Hop based
relative distance encoding, and iii) Weisfeiler Lehman based
absolute PE (WL-PE). The intimacy based PE and the hop
based PE are variant to the sampled subgraphs, i.e., these
PEs for a node in a subgraph gidepends on the node uiw.r.t
which it is sampled, and cannot be directly used in other
cases unless we use similar sampling strategy. The WL-PE
which are absolute structural roles of nodes in the original
graph computed using WL algorithm (Zhang et al. 2020;
Niepert, Ahmed, and Kutzkov 2016), are not variant to the
subgraphs and can be easily used as a generic PE mecha-
nism. On that account, we swap Laplacian PE in our experi-
ments for an ablation analysis and use WL-PE from Graph-
BERT, see Table 3. As Laplacian PE capture better struc-
tural and positional information about the nodes, which es-
sentially is the objective behind using the three Graph-BERT
PEs, they outperform the WL-PE. Besides, WL-PEs tend to
overﬁt SBM datasets and lead to poor generalization.Sparse Graph
Dataset PE #Param Test Perf.s.d. Train Perf.s.d. #Epoch Epoch/Total
Batch Norm: True ; Layer Norm: False ;L= 10
ZINCx 588353 0.2640.008 0.0480.006 321.50 28.01s/2.52hr
L 588929 0.2260.014 0.0590.011 287.50 27.78s/2.25hr
W 590721 0.2670.012 0.0590.010 263.25 27.04s/2.00hr
CLUSTERx 523146 72.1390.405 85.8570.555 121.75 200.85s/6.88hr
L 524026 73.1690.622 86.5850.905 126.50 201.06s/7.20hr
W 531146 70.7900.537 86.8290.745 119.00 196.41s/6.69hr
PATTERNx 522742 83.9490.303 83.8640.489 236.50 299.54s/19.71hr
L 522982 84.8080.068 86.5590.116 145.25 309.95s/12.67hr
W 530742 75.4890.216 97.0280.104 109.25 310.11s/9.73hr
Table 3: Analysis of GraphTransformer (GT) using different
PE schemes. Notations x: No PE; L: LapPE (ours); W: WL-
PE (Zhang et al. 2020). Bold : the best performing model for
each dataset.
5 Conclusion
This work presented a simple yet effective approach to gen-
eralize transformer networks on arbitrary graphs and intro-
duced the corresponding architecture. Our experiments con-
sistently showed that the presence of – i) Laplacian eigen-
vectors as node positional encodings and – ii) batch normal-
ization, in place of layer normalization, around the trans-
former feed forward layers enhanced the transformer univer-
sally on all experiments. Given the simple and generic na-
ture of our architecture and competitive performance against
standard GNNs, we believe the proposed model can be used
as baseline for further improvement across graph applica-
tions employing node attention. In future works, we are in-
terested in building upon the graph transformer along as-
pects such as efﬁcient training on single large graphs, ap-
plicability on heterogeneous domains, etc., and perform ef-
ﬁcient graph representation learning keeping in account the
recent innovations in graph inductive biases.
Acknowledgments
XB is supported by NRF Fellowship NRFF2017-10.

## Page 7

References
Abbe, E. 2017. Community detection and stochastic block
models: recent developments. The Journal of Machine
Learning Research 18(1): 6446–6531.
Ba, J. L.; Kiros, J. R.; and Hinton, G. E. 2016. Layer nor-
malization. NeurIPS workshop on Deep Learning .
Bahdanau, D.; Cho, K.; and Bengio, Y . 2014. Neural ma-
chine translation by jointly learning to align and translate.
arXiv preprint arXiv:1409.0473 .
Belkin, M.; and Niyogi, P. 2003. Laplacian eigenmaps for
dimensionality reduction and data representation. Neural
computation 15(6): 1373–1396.
Bresson, X.; and Laurent, T. 2017. Residual gated graph
convnets. arXiv preprint arXiv:1711.07553 .
Brown, T. B.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J.;
Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell,
A.; et al. 2020. Language models are few-shot learners.
arXiv preprint arXiv:2005.14165 .
Chami, I.; Wolf, A.; Juan, D.-C.; Sala, F.; Ravi, S.; and R ´e,
C. 2020. Low-Dimensional Hyperbolic Knowledge Graph
Embeddings. arXiv preprint arXiv:2005.00545 .
Cranmer, M. D.; Xu, R.; Battaglia, P.; and Ho, S. 2019.
Learning Symbolic Physics with Graph Networks. arXiv
preprint arXiv:1909.05862 .
Defferrard, M.; Bresson, X.; and Vandergheynst, P. 2016.
Convolutional Neural Networks on Graphs with Fast Local-
ized Spectral Filtering. In Advances in Neural Information
Processing Systems 29 , 3844–3852.
Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2018.
Bert: Pre-training of deep bidirectional transformers for lan-
guage understanding. arXiv preprint arXiv:1810.04805 .
Dwivedi, V . P.; Joshi, C. K.; Laurent, T.; Bengio, Y .; and
Bresson, X. 2020. Benchmarking graph neural networks.
arXiv preprint arXiv:2003.00982 .
Gilmer, J.; Schoenholz, S. S.; Riley, P. F.; Vinyals, O.; and
Dahl, G. E. 2017. Neural message passing for quantum
chemistry. In Proceedings of the 34th International Confer-
ence on Machine Learning-Volume 70 , 1263–1272. JMLR.
org.
Hu, Z.; Dong, Y .; Wang, K.; and Sun, Y . 2020. Heteroge-
neous graph transformer. In Proceedings of The Web Con-
ference 2020 , 2704–2710.
Ioffe, S.; and Szegedy, C. 2015. Batch normalization: Accel-
erating deep network training by reducing internal covariate
shift. arXiv preprint arXiv:1502.03167 .
Irwin, J. J.; Sterling, T.; Mysinger, M. M.; Bolstad, E. S.;
and Coleman, R. G. 2012. ZINC: a free tool to discover
chemistry for biology. Journal of chemical information and
modeling 52(7): 1757–1768.
Joshi, C. 2020. Transformers are Graph Neural Networks.
The Gradient .
Kipf, T. N.; and Welling, M. 2017. Semi-Supervised Clas-
siﬁcation with Graph Convolutional Networks. In Interna-
tional Conference on Learning Representations (ICLR) .Li, P.; Wang, Y .; Wang, H.; and Leskovec, J. 2020. Dis-
tance Encoding–Design Provably More Powerful GNNs
for Structural Representation Learning. arXiv preprint
arXiv:2009.00142 .
Li, Y .; Liang, X.; Hu, Z.; Chen, Y .; and Xing, E. P. 2019.
Graph Transformer. URL https://openreview.net/forum?id=
HJei-2RcK7.
Li, Y .; Tarlow, D.; Brockschmidt, M.; and Zemel, R. 2015.
Gated graph sequence neural networks. arXiv preprint
arXiv:1511.05493 .
Monti, F.; Boscaini, D.; Masci, J.; Rodola, E.; Svoboda, J.;
and Bronstein, M. M. 2017. Geometric Deep Learning on
Graphs and Manifolds Using Mixture Model CNNs. 2017
IEEE Conference on Computer Vision and Pattern Recogni-
tion (CVPR) doi:10.1109/cvpr.2017.576.
Monti, F.; Frasca, F.; Eynard, D.; Mannion, D.; and Bron-
stein, M. M. 2019. Fake news detection on social
media using geometric deep learning. arXiv preprint
arXiv:1902.06673 .
Murphy, R.; Srinivasan, B.; Rao, V .; and Ribeiro, B. 2019.
Relational Pooling for Graph Representations. In Interna-
tional Conference on Machine Learning , 4663–4673.
Nguyen, D. Q.; Nguyen, T. D.; and Phung, D. 2019. Univer-
sal Self-Attention Network for Graph Classiﬁcation. arXiv
preprint arXiv:1909.11855 .
Niepert, M.; Ahmed, M.; and Kutzkov, K. 2016. Learning
convolutional neural networks for graphs. In International
conference on machine learning , 2014–2023.
Paszke, A.; Gross, S.; Massa, F.; Lerer, A.; Bradbury, J.;
Chanan, G.; Killeen, T.; Lin, Z.; Gimelshein, N.; Antiga, L.;
Desmaison, A.; K ¨opf, A.; Yang, E.; DeVito, Z.; Raison, M.;
Tejani, A.; Chilamkurthy, S.; Steiner, B.; Fang, L.; Bai, J.;
and Chintala, S. 2019. PyTorch: An Imperative Style, High-
Performance Deep Learning Library.
Radford, A.; Narasimhan, K.; Salimans, T.; and Sutskever,
I. 2018. Improving language understanding by generative
pre-training.
Sanchez-Gonzalez, A.; Godwin, J.; Pfaff, T.; Ying, R.;
Leskovec, J.; and Battaglia, P. W. 2020. Learning to sim-
ulate complex physics with graph networks. arXiv preprint
arXiv:2002.09405 .
Schlichtkrull, M.; Kipf, T. N.; Bloem, P.; Van Den Berg, R.;
Titov, I.; and Welling, M. 2018. Modeling relational data
with graph convolutional networks. In European Semantic
Web Conference , 593–607. Springer.
Srinivasan, B.; and Ribeiro, B. 2020. On the Equivalence
between Node Embeddings and Structural Graph Represen-
tations. International Conference on Learning Representa-
tions .
Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones,
L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. At-
tention is all you need. In Advances in neural information
processing systems , 5998–6008.

## Page 8

Veliˇckovi ´c, P.; Cucurull, G.; Casanova, A.; Romero, A.; Li `o,
P.; and Bengio, Y . 2018. Graph Attention Networks. Inter-
national Conference on Learning Representations .
Wang, M.; Yu, L.; Zheng, D.; Gan, Q.; Gai, Y .; Ye, Z.; Li,
M.; Zhou, J.; Huang, Q.; Ma, C.; Huang, Z.; Guo, Q.; Zhang,
H.; Lin, H.; Zhao, J.; Li, J.; Smola, A. J.; and Zhang, Z.
2019. Deep Graph Library: Towards Efﬁcient and Scalable
Deep Learning on Graphs. ICLR Workshop on Representa-
tion Learning on Graphs and Manifolds .
Xu, K.; Hu, W.; Leskovec, J.; and Jegelka, S. 2019. How
Powerful are Graph Neural Networks? In International Con-
ference on Learning Representations .
Xu, K.; Li, C.; Tian, Y .; Sonobe, T.; Kawarabayashi, K.-
i.; and Jegelka, S. 2018. Representation learning on
graphs with jumping knowledge networks. arXiv preprint
arXiv:1806.03536 .
Xu, P.; Joshi, C. K.; and Bresson, X. 2019. Multi-graph
transformer for free-hand sketch recognition. arXiv preprint
arXiv:1912.11258 .
You, J.; Ying, R.; and Leskovec, J. 2019. Position-aware
graph neural networks. International Conference on Ma-
chine Learning .
Yun, S.; Jeong, M.; Kim, R.; Kang, J.; and Kim, H. J. 2019.
Graph transformer networks. In Advances in Neural Infor-
mation Processing Systems , 11983–11993.
Zhang, J.; Zhang, H.; Sun, L.; and Xia, C. 2020. Graph-Bert:
Only Attention is Needed for Learning Graph Representa-
tions. arXiv preprint arXiv:2001.05140 .
Zhou, D.; Zheng, L.; Han, J.; and He, J. 2020. A Data-
Driven Graph Generative Model for Temporal Interaction
Networks. In Proceedings of the 26th ACM SIGKDD Inter-
national Conference on Knowledge Discovery & Data Min-
ing, 401–411.A Appendix
A.1 Task based MLP layer equations
Graph prediction layer For graph prediction task, the ﬁ-
nal layer node features of a graph is averaged to get a d-
dimensional graph-level feature vector yG.
yG=1
VVX
i=0hL
i; (19)
The graph feature vector is then passed to a MLP to obtain
the un-normalized prediction score for each class, ypred2
RCfor each class:
ypred=PReLU (QyG); (20)
whereP2RdC;Q2Rdd;Cis the number of task la-
bels (classes) to be predicted. Since we perform single-target
graph regression in ZINC, C= 1, and the L1-loss between
the predicted and groundtruth values is minimized during
training.
Node prediction layer For node prediction task, each
node’s feature vector is passed to a MLP for computing
the un-normalized prediction scores yi;pred2RCfor each
class:
yi;pred=PReLU 
QhL
i
; (21)
whereP2RdC;Q2Rdd. During training, the cross-
entropy loss weighted inversely by the class size is used.
As a note, these task based layers can be modiﬁed as
per the requirements of the dataset, and or the prediction to
be done. For example, the Graph Transformer edge outputs
(Figure 1 (Right)) can be used for edge prediction tasks and
the task based MLP layers can be deﬁned in similar fashion
as we do for node prediction. Besides, different styles of us-
ing ﬁnal and/or intermediate Graph Transformer layers can
be used as inputs to the task based MLP layers, such as JK
Readout (Jumping Knowledge) (Xu et al. 2018), etc. used
often in GNNs.
A.2 Hardware Information
All experiments are run on Intel Xeon CPU E5-2690 v4
server with 4 Nvidia 1080Ti GPUs. At a given time, 4 exper-
iments were run on the server with each single GPU running
1 experiment. The maximum training time for an experiment
is limited to 24 hours.