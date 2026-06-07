# A Survey on Graph Neural Network Acceleration Algorithms, Systems, and Customized Hardware

## Page 1

A Survey on Graph Neural Network Acceleration: Algorithms,
Systems, and Customized Hardware
SHICHANG ZHANG, University of California, Los Angeles, USA
ATEFEH SOHRABIZADEH, University of California, Los Angeles, USA
CHENG WAN, Georgia Institute of Technology, USA
ZIJIE HUANG, University of California, Los Angeles, USA
ZINIU HU, University of California, Los Angeles, USA
YEWEN WANG, University of California, Los Angeles, USA
YINGYAN (CELINE) LIN, Georgia Institute of Technology, USA
JASON CONG, University of California, Los Angeles, USA
YIZHOU SUN, University of California, Los Angeles, USA
Graph neural networks (GNNs) are emerging for machine learning research on graph-structured data. GNNs
achieve state-of-the-art performance on many tasks, but they face scalability challenges when it comes to
real-world applications that have numerous data and strict latency requirements. Many studies have been
conducted on how to accelerate GNNs in an effort to address these challenges. These acceleration techniques
touch on various aspects of the GNN pipeline, from smart training and inference algorithms to efficient
systems and customized hardware. As the amount of research on GNN acceleration has grown rapidly, there
lacks a systematic treatment to provide a unified view and address the complexity of relevant works. In this
survey, we provide a taxonomy of GNN acceleration, review the existing approaches, and suggest future
research directions. Our taxonomic treatment of GNN acceleration connects the existing works and sets the
stage for further development in this area.
CCS Concepts: •Computing methodologies →Neural networks ;•Computer systems organization →
Distributed architectures ; Data flow architectures; •Hardware→Integrated circuits .
Additional Key Words and Phrases: graph neural networks, model acceleration, hardware acceleration/accel-
erator
ACM Reference Format:
Shichang Zhang, Atefeh Sohrabizadeh, Cheng Wan, Zijie Huang, Ziniu Hu, Yewen Wang, Yingyan (Celine)
Lin, Jason Cong, and Yizhou Sun. XXXX. A Survey on Graph Neural Network Acceleration: Algorithms,
Systems, and Customized Hardware. ACM Comput. Surv. 1, 1 (June XXXX), 54 pages. https://doi.org/XXXXXXX.
XXXXXXX
Authors’ addresses: Shichang Zhang, shichang@cs.ucla.edu, University of California, Los Angeles, USA; Atefeh Sohrabizadeh,
atefehsz@cs.ucla.edu, University of California, Los Angeles, USA; Cheng Wan, chwan@gatech.edu, Georgia Institute of
Technology, USA; Zijie Huang, zijiehuang@cs.ucla.edu, University of California, Los Angeles, USA; Ziniu Hu, bull@cs.ucla.
edu, University of California, Los Angeles, USA; Yewen Wang, wyw10804@cs.ucla.edu, University of California, Los Angeles,
USA; Yingyan (Celine) Lin, celine.lin@gatech.edu, Georgia Institute of Technology, USA; Jason Cong, cong@cs.ucla.edu,
University of California, Los Angeles, USA; Yizhou Sun, yzsun@cs.ucla.edu, University of California, Los Angeles, USA.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and
the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
©XXXX Association for Computing Machinery.
0360-0300/XXXX/6-ART $15.00
https://doi.org/XXXXXXX.XXXXXXX
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.arXiv:2306.14052v1  [cs.LG]  24 Jun 2023

## Page 2

2 Zhang, Shichang, et al.
1 Introduction
Graphs are a natural and powerful data structure for representing entities and their relationships.
Many real-world data can be represented as graphs with nodes denoting a collection of entities
and edges denoting their pairwise relationships, such as individuals in social networks, financial
transactions between firms and banks, atoms and bonds in molecules, and vehicles in transportation
systems. Graph neural networks (GNNs) [ 45,71,125] have recently become the most widely used
graph machine learning (ML) model for learning knowledge and making predictions on graph data.
GNNs have achieved state-of-the-art performance in many graph ML applications. They are used, for
example, in recommendations on social graphs [ 89,136,165], fraud account detection on financial
graphs [ 31], drug discoveries from molecule graphs [ 64], traffic forecasting on transportation
graphs [65], and so on.
The superior performance of GNNs on graphs is mainly due to their ability to combine the entity
information, represented as the node features, and the relationships, represented as the graph
structure. GNNs take node features and the graph structure as input and output node representations,
which can be used for various graph ML tasks like node classification, edge prediction, and graph
classification. For each node in a graph, a GNN learns its representation by message passing ,
which aggregates messages (features) from the node’s neighbors and updates the representation
by transforming the aggregated messages with a neural network. Each aggregate-and-update
operation is called a GNN layer, which allows the node representation to combine information from
the node’s direct neighbors. Stacking multiple GNN layers will apply such operations recursively so
the node representation can gain information from multi-hop neighbors. Thus, the representation
can capture the contextual information of a local neighborhood and be used to answer complex
ML questions on graphs. For a target node with an 𝐿-layer GNN applied on it, all of its neighbors
within𝐿hops form the receptive field of the target node. A recursively constructed 𝐿-layer tree
graph with the target node as the root and the neighbors of each node as its children is called a
computation graph of the target node.
As the field of graph ML quickly develops, graph data is becoming gigantic with numerous nodes
and edges in a single graph. For example, the Twitter user graph has 288M monthly active users
(nodes) and an estimated 208 follow relations (edges) per user as of 3/2015, and the Facebook user
graph has 1.39B active users and more than 400B total edges as of 12/2014 [ 22]. These numbers
have grown much larger and become hard to estimate. The commonly used academic benchmarks
have also increased from thousands of nodes, e.g., Cora [ 108], to as many as 240M nodes in the
recent Open Graph Benchmark (OGB) [ 50]. Large graphs raise scalability challenges for both
GNN training and inference. Although deep neural networks (DNNs) for other data types also
face scalability challenges when the number of DNN parameters becomes large, the challenge for
GNNs is unique and there is greater emphasis on the large graph data [ 63,163] rather than the
large number of model parameters. When training DNNs on images or text, even if the amount of
data is large, data can be processed by randomly sampling mini-batches because all data instances
are assumed to be independent and identically distributed (iid). The DNN inference can also be
done independently for each instance. For GNNs on graphs, however, the nodes are dependent on
each other, which brings both benefits and challenges. GNNs leverage node dependency to learn
informative representations and make predictions, but node dependency also means multi-hop
neighbors are involved in the GNN computation for every single node. Therefore, it is nontrivial
to partition graph nodes into mini-batches for GNN training considering the node dependency.
Moreover, node dependency means that the sizes of computation graphs grow exponentially in
the number of GNN layers [ 2]. On large graphs where deeper GNNs are often needed for better
performance [ 14,73], not only does the mini-batch training need to process a huge computation
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 3

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 3
graph but so does the inference on a single node. Take the Twitter user graph as an example, with
208 neighbors per node, a computation graph of a three-layer GNN involves all neighbors of a
node within three steps, which can contain millions of nodes (i.e., roughly 2083minus overlapped
nodes). This node dependency problem makes applying GNNs on large graphs very challenging. A
different situation for large datasets is a graph database with many graphs, for example, chemical
molecules. Since graphs in these databases often only have dozens to hundreds nodes, the node
dependency problem is not a big concern, and they are not the focus of this survey.
Scalability challenges also show up for GNN systems, including both Commercial-off-the-Shelf
(COTS) systems and customized hardware. Large computation graphs need to be processed because
of the node dependency problem, but memory consumption becomes infeasible when the graph
structure is represented as a huge adjacency matrix. In reality, the graph structure often needs to
be processed in the sparse matrix format. However, sparse matrices have different computation
patterns from standard DNNs. Therefore, the driving force of the success of deep learning on large
data, i.e., the parallelism of computation devices like GPUs, cannot be fully exploited for efficient
computation. This demands efficient systems with accelerated computation kernels and customized
hardware accelerators like FPGA.
The scalability challenges of GNNs require acceleration techniques that will lead to resource-
efficient GNN training and fast GNN inference for latency-constraint applications. Acceleration
techniques can also benefit useful applications like neural architecture search. Many GNN ac-
celeration techniques with different emphases have been proposed in the literature, including
smart training and inference algorithms, efficient systems, and customized hardware designs. In
this survey, we provide a unified view of different GNN acceleration techniques to address their
complexity. We show a taxonomy of GNN acceleration techniques in Figure 1 that categorizes GNN
acceleration techniques into three categories: (1) algorithms, (2) COTS systems, and (3) customized
hardware. The algorithms category includes training acceleration techniques that modify the graph
or sample from the graph, alleviating the node dependency problem. The category also includes
inference acceleration techniques like pruning, quantization, and distillation, which can convert
a trained GNN into a simpler and faster model. The system category includes GPU kernel accel-
eration for sparse matrix operations, user-defined function optimization to generate code better
adjusted for the target hardware, and the design of scalable systems for distributed training on
multiple devices. The customized hardware category includes hardware accelerators with different
properties that cover both cases of allowing or forbidding layer customization, and have different
levels of parallelization and sparsity support. Besides the three main categories, we also discuss
techniques that accelerate GNNs on special heterogeneous and dynamic graphs. We discuss all the
techniques in detail with regard to their advantages, applicability, and limitations. We also suggest
research directions to set the stage for further development of this research topic.
2 Notations and Preliminaries
In this section, we clarify the common notations used throughout the paper in Table 1. Also, we
review graph machine learning with GNNs by going through the graph data representation, the
GNN model architecture, and the machine learning tasks GNNs can solve.
2.1 Graph Data Representation
An attributed graph Gincludes the graph structure (V,E)and node features 𝑿, withVdenotes a set
of nodes, andEdenotes a set of edges. A straightforward way to represent the graph structure (edges
E) is to use an adjacency matrix 𝑨∈R|V|×|V|, with|V|denotes the number of nodes. Many graph
algorithms, including the matrix version of GNNs, are presented in terms of 𝑨. However, naively
representing edges with the full matrix 𝑨requires𝑂(|V|2)space complexity, which is impractical to
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 4

4 Zhang, Shichang, et al.
*11$FFHOHUDWLRQ$OJRULWKPV7UDLQLQJ*UDSK0RGLILFDWLRQ&RDUVHQLQJ6SDUVLILFDWLRQ&RQGHQVDWLRQ6DPSOLQJ1RGHZLVH/D\HUZLVH6XEJUDSKZLVH,QIHUHQFH3UXQLQJ4XDQWL]DWLRQ'LVWLOODWLRQ&2766\VWHPV*38.HUQHO$FFHOHUDWLRQ8VHU'HILQHG)XQFWLRQ2SWLPL]DWLRQ6FDODEOH7UDLQLQJ6\VWHPV&XVWRPL]HG+DUGZDUH*HQHUDO:RUNORDG$OO6WDJH8QLILHG$UFKLWHFWXUH'HGLFDWHG0RGXOHSHU6WDJH6SHFLDO:RUNORDG/D\HU6SHFLILF&XVWRPL]DEOH'HHS3LSHOLQH$OO/D\HU8QLILHG$UFKLWHFWXUH6SHFLDO*UDSKVDQG6SHFLDO*11V+HWHURJHQHRXV*UDSKV'\QDPLF*UDSKV
Fig. 1. A taxonomy of GNN acceleration. Specifically, we discuss training algorithms in Section 3, inference
algorithms in Section 4, COTS systems in Section 5, customized hardware in Section 6, and special graphs
and GNNs in Section 7.
implement for large graphs. In practice, most of the graphs are sparse, i.e., |E|<<|V|2. Leveraging
the sparsity to represent Gin more efficient formats can help save memory and accelerate many
computations. One popular sparse representation is the Coordinate (COO) [131] format, which is
also known as the “triplet” format. For this format, each edge (𝑖,𝑗)∈E is represented as a triplet
(𝑖,𝑗,𝑤𝑖𝑗)with𝑤𝑖𝑗being the edge weight for weighted graphs, and 𝑤𝑖𝑗equals one for unweighted
graphs. The space complexity of a COO graph scales only with the number of edges, e.g., 𝑂(|E|) .
Another popular choice is the Compressed Sparse Row (CSR) format [ 103,131]. It is similar to the
COO format but compresses those |E|triplets into three|E|-dimensional arrays. One array for
all the row indices, e.g., the first entry in COO. One array for all the column indices, e.g., the
second entry in COO. One array for all the edge weights, e.g., the third entry in COO. The space
complexity of a CSR graph is also 𝑂(|E|) . Additionally, graphs often come with node features,
which are represented as a feature matrix 𝑿∈R|V|×𝐷.𝑿is often referred to as the raw features to
distinguish them from the hidden features learned by the model. Similarly, there could be features
for graph edges, but we do not assume edge features are given in this survey because they are not
generally available.
2.2 Graph Neural Networks
GNN is a family of neural network models that achieve state-of-the-art performance on many
machine learning tasks on graphs. GNNs take a graph and node features as input and output node
representations. The node representations can be used for various downstream ML tasks like node
classification and edge prediction. Node representations can also be aggregated over the whole
graph to get a graph representation for graph classification.
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 5

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 5
Table 1. Major Notations
Category Notations Descriptions
GraphsG A graph with feature-enriched nodes
𝑁 The total number of nodes in G, i.e.,|V|
V The node set ofG
E The edge set ofG
𝑿 Raw node features in R𝑁×𝐷
𝒙𝑖 The𝑖th row of 𝑿, which is the raw feature of the node 𝑖∈V
𝑨 The adjacency matrix. 𝑨∈{0,1}𝑁×𝑁.𝑨𝑖,𝑗=1if edge(𝑖,𝑗)∈E , otherwise 0
˜𝑨 The adjacency matrix with self-loops. ˜𝑨=𝑨+𝑰
𝑫,˜𝑫 Degree matrices of 𝑨and ˜𝑨.𝑫𝑖,𝑖=Í
𝑗𝑨𝑗,𝑖and𝑫𝑖,𝑗=0if𝑖≠𝑗. Similarly for ˜𝑫
N(𝑣) Neighbors of a node 𝑣∈V .N(𝑣)={𝑢∈V|(𝑢,𝑣)∈E}
𝑑(𝑣) The degree of node 𝑣.𝑑(𝑣)=|N(𝑣)|
GNNsGNN𝜃 The GNN model with parameters 𝜃
𝐿 Total number of layers of a GNN
𝑯(𝑙)Hidden node representations at layer 𝑙.𝑯(0)=𝑿.𝑯(𝑙)∈R𝑁×𝐹(assume𝐹is shared for 𝑙≥1)
𝒉(𝑙)
𝑖The𝑖th row of 𝑯(𝑙), which is the hidden feature of node 𝑖at layer𝑙
𝜙(·,·) The message (a.k.a. feature extraction) function which constructs a message between two nodes
𝒎(𝑙)
𝑢,𝑣 The constructed message from node 𝑢to node𝑣at layer𝑙
AGGR The aggregation function, usually order invariant, e.g., mean
𝒂(𝑙)
𝑣 The aggregated feature of node 𝑣at layer𝑙
UPDATE The update function. A neural network. Usually a multi-layer perceptron (MLP)
𝑾(𝑙)Neural network weights need to be learned in the UPDATE function at layer 𝑙
𝜎 The nonlinear activation function of the UPDATE neural network
𝑷 The filter matrix for aggregating messages, usually a function of 𝑨and𝑫
Labels &
Training𝒀 One-hot node labels to predict. 𝒀∈{0,1}𝑁×𝐶
𝒚𝑖 The𝑖th row of 𝒀, which is the one-hot node label for the node 𝑖
ˆ𝒚𝑖 Prediction of the node 𝑖∈V , a probability vector. ˆ𝒚𝑖∈[0,1]𝐶
ONEHOT One-hot vector function. Takes 𝑐∈{1,...,𝐶}and outputs 𝒚∈{0,1}𝐶with only 𝒚𝑐=1
L(𝒚𝑖,ˆ𝒚𝑖)The loss function for training GNN𝜃
There are many different GNN model architectures. For example, the graph convolutional
network (GCN) [ 71], graph information network (GIN) [ 141], graph attention network (GAT) [ 125],
etc. Most of the model variants belong to the message-passing GNN framework [ 42]. Message
passing is a mechanism that iteratively updates the hidden representation of a node by aggregating
messages from its neighbors and transforming the aggregated messages with a neural network.
Each message-passing iteration is called a GNN layer, which allows the hidden representation
to combine information from the node’s direct neighbors. Stacking multiple GNN layers allows
multi-hop neighbors to pass messages to the central node. Thus, the node representation can
capture the contextual information of a local neighborhood and be used to answer complex ML
questions on graphs.
We now describe the detailed message-passing operation using the notations defined in Table
1. For layer 𝑙, the hidden representation 𝒉(𝑙)of a node𝑣is updated by first aggregating messages
(a function of hidden representations from the previous iteration) from its neighbors and then
combining the aggregated messages with its own hidden representation. The message-passing
operation:
𝒉(𝑙)
𝑣=UPDATE(𝒉(𝑙−1)
𝑣,AGGR({𝜙(𝒉(𝑙−1)
𝑣,𝒉(𝑙−1)
𝑢)|𝑢∈N(𝑣)})). (1)
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 6

6 Zhang, Shichang, et al.
where𝜙(·,·)constructs the message between two nodes from their hidden representations; AGGR
is the aggregation function, usually being order-invariant, e.g., mean; and UPDATE is the update
function, usually being a multi-layer perceptron (MLP) with learnable weights 𝑾(𝑙). Equation (1)
shows one message passing step for layer 𝑙, i.e., the𝑙-th GNN layer. For a GNN with 𝐿total layers,
message passing is applied 𝐿times to aggregate messages from neighbors within 𝐿-hops of𝑣to
compute the representation 𝒉(𝐿)
𝑣. All the neighbors within 𝐿-hop of𝑣form the receptive field of𝑣.
A recursively constructed 𝐿-layer tree graph with 𝑣as the root and the neighbors of each node as
its children are called a computation graph of𝑣.
For many GNNs, the message-passing operation in Equation (1) can be implemented by a clean
matrix multiplication to update hidden representations of all 𝑣∈V at once, which is also called
thefull-batch version. The full-batch updates:
𝑯(𝑙)=𝜎(𝑷𝑯(𝑙−1)𝑾(𝑙)), (2)
where the matrix 𝑯(𝑙−1)contains hidden representations of all nodes at the (𝑙−1)-th step. It is
first multiplied by a filter matrix 𝑷(on the left) and a neural network weight matrix 𝑾(𝑙)(on the
right) and then is passed through a nonlinear activation function 𝜎(·).𝑷corresponds to the AGGR
function and 𝑾(𝑙)and𝜎(·)correspond to the UPDATE function (when it is an MLP) in Equation (1).
𝑷can take different forms for different GNNs, and it is usually a function of 𝑨. For example, 𝑷is a
degree-normalized version of the adjacency matrix with self-loops for GCN, i.e., 𝑷=˜𝑫−1
2˜𝑨˜𝑫−1
2.
Algorithm 1 SAGA (Scatter-ApplyEdge-Gather-ApplyVertex) based GNN Message Passing
Input: A graphG=(V,E), node features 𝑿
𝑯(0)=𝑿
for𝑙←1 to𝐿do
for(u, v) inEdo
𝒎(𝑙)
𝑢,𝑣←𝜙(𝒉(𝑙−1)
𝑣,𝒉(𝑙−1)
𝑢)
end for
forv inVdo
𝒂(𝑙)
𝑣←AGGR({𝒎(𝑙)
𝑢,𝑣|𝑢∈N(𝑣)})
𝒉(𝑙)
𝑣←UPDATE(𝒉(𝑙−1)
𝑣,𝒂(𝑙)
𝑣)
end for
end for
The matrix version looks cleaner but it is implemented less often in practice since the full
matrix representation of 𝑷usually has space complexity 𝑂(|V|2), so it is infeasible to fit such 𝑷
in memory for large graphs. More frequently, 𝑷is represented as a sparse matrix (e.g., in COO or
CSR format) mentioned above, and message passing is implemented using the Scatter-ApplyEdge-
Gather-ApplyVertex (SAGA) paradigm [ 86]. For SAGA, messages are first computed and stored on
edges, and then each node aggregates messages from its edges and updates the representation. We
show the SAGA GNN message passing in Algorithm 1, which has time complexity 𝑂(|E|) .
2.3 Graph Machine Learning Tasks
GNNs are capable of solving many graph machine-learning tasks. For example, at the node level, it
can classify fraudulent nodes. At the edge level, it can predict edges for recommendations. At the
graph level, it can classify properties for molecule graphs. For all three kinds of tasks, the standard
approaches share a common step of getting node representations 𝑯. Then 𝒉𝑣can be used for solving
node-level tasks, or pairs (𝒉𝑢,𝒉𝑣)can be used for solving edge-level tasks, and 𝑯may go through
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 7

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 7
a pooling layer over all of its rows to become a single representation for solving graph-level tasks.
The last prediction step is usually running through a few fully-connected network layers and is
not time-consuming [ 71]. Therefore, for GNN acceleration, we focus on how 𝑯can be efficiently
learned on large graphs, and we use multi-class node classification as a concrete example in this
survey.
Node classification is one of the most widely considered tasks on graphs, which classifies each
node𝑣∈V to one of𝐶classes. It is usually semi-supervised meaning that the entire graph with
all node features and a few node labels are observed. The goal is to leverage the observed labels
and the graph information to classify other nodes without labels. When inferring the class of node
𝑣, its final layer hidden representation 𝒉(𝐿)
𝑣is first generated via message passing, and then 𝒉(𝐿)
𝑣is
plugged into a prediction head 𝑓:R𝐹→R𝐶to get the final prediction ˆ𝒚𝑣=𝑓(𝒉(𝐿)
𝑣).𝑓(·)is usually
an MLP learned jointly with the GNN using the cross-entropy loss. The loss will be computed for
all nodes with observed labels during training, and for a single node 𝑣, it is:
L(𝒚𝑣,ˆ𝒚𝑣)=𝐶∑︁
𝑐=1𝒚𝑣,𝑐logˆ𝒚𝑣,𝑐. (3)
3 GNN Acceleration: Training Algorithms
Given a GNN𝜃with randomly initialized parameters 𝜃and a graphGwith node features 𝑿and
labels 𝒀, the goal of GNN training is to find an optimal set of parameters 𝜃∗that minimizes
the loss, i.e., 𝜃∗=arg min𝜃L(GNN𝜃(G,𝑿),𝒀). The major latency for GNN training algorithms
comes from aggregating messages from (multi-hop) neighbor nodes in the receptive field, and the
computation graphs can become huge for deep GNNs on large graphs. The general idea of GNN
training acceleration is thus to reduce the computation graphs. Also, training with acceleration is
desired to produce a GNN with similar performance to the GNN trained without acceleration.
In this section, we discuss two categories of training acceleration techniques: graph modification
and sampling. Both methods aim to reduce the computation graph to accelerate training. The
difference is whether there is a modified graph G′as an intermediate output. For graph modification,
the procedure can be separated into two steps. The first step outputs a graph G′that is smaller
thanGand can be fastly trained on. The second stage is regular GNN training with G′as the
input. For sampling, a subset of nodes/edges is selected to construct smaller computation graphs in
each training iteration. Generally speaking, sampling also modifies G, but the modification is done
dynamically and implicitly, so there is no intermediate G′output. Since the computation graph
can differ from iteration to iteration, all the nodes and edges have the chance to be covered by
sampling. In contrast, for graph modification, not all the nodes and edges will appear in G′, but
new nodes and edges may be created.
3.1 Graph Modification
Graph modification accelerates GNN training in two steps. The first step takes the graph G=(V,E),
node features 𝑿, and labels 𝒀as input and outputs modified G′=(V′,E′),𝑿′, and 𝒀′for training.
Training onG′will be faster, ifG′is smaller thanGin the sense that|V′|<|V|and/or|E′|<|E|.
The second step trains a GNN on G′such that the GNN has a similar performance on Gas if it
is trained onG. Mathematically, regular training outputs 𝜃∗=arg min𝜃L(GNN𝜃(G,𝑿),𝒀), and
training with graph modification outputs 𝜃′∗=arg min𝜃′L(GNN𝜃′(G′,𝑿′),𝒀′)such that GNN𝜃′∗
has similar performance as GNN𝜃∗onG. In the following, we go over several types of graph
modification methods including graph coarsening, graph sparsification, and graph condensation.
Each of these methods modifies the graph Gdifferently to create its own G′, but allG′are smaller
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 8

8 Zhang, Shichang, et al.
SparsificationCondensationGeneratorMatchingCondensedgraphCoarseningClusteringMerging
updateTrainTrainSparsifyPropertymatchingSparsifiedgraphCoarsegraph
𝐺𝑁𝑁!!Original GraphModified Graph
𝐺𝑁𝑁!
Fig. 2. Illustration of graph modification methods: Graph Coarsening methods perform graph clustering
and merge clusters of nodes into a super-node. Graph Sparsification methods remove less important edges.
Graph Condensation methods generate a new condensed graph using a randomly initialized generative model.
For the modified graph (the rightmost column), black nodes/edges are from the original graph, and colored
nodes/edges are newly created.
graphs that accelerate GNN training. An illustration of these graph modification methods is shown
in Figure 2.
3.1.1 Graph Coarsening Graph coarsening is a technique aiming at reducing the graph size
while preserving its overall structure. Given a graph G, graph coarsening merges nodes in the
same local structure of Ginto a single “super-node” and merges edges connecting super-nodes
into a “super-edge”. The coarse graph G′thus has much fewer nodes and edges compared to G.
The key step of coarsening is to cluster nodes in Ginto𝐾clusters for merging. Nodes in the same
cluster will be merged into a super-node, and then a super-edge (𝑘,𝑜)connecting super-node 𝑘
and super-node 𝑜will be constructed if there are edges connecting nodes in cluster 𝑘and nodes in
cluster𝑜. An illustration of the clustering and merging of graph coarsening is shown in Figure 2
top row.
Graph coarsening, with its central step being graph clustering, has been studied long before
GNNs were proposed [ 69,70,126], and new coarsening algorithms have been proposed until very
recently [ 8,84,85]. The key idea of these algorithms is to coarsen the graph while preserving
some graph properties, which are often related to the graph spectrum, e.g., restricted spectral
approximation [ 85] and inverse Laplacian [ 8]. Below, we discuss the graph coarsening algorithms
related to GNNs.
Huang et al. [58] discuss a general framework using graph coarsening to accelerate GNN
training. In particular, a graph Gis partitioned into K clusters with sizes 𝑁1,...,𝑁𝐾. A matrix
𝑪∈R𝑁×𝐾is used to represent the partition, where 𝑪𝑖,𝑗=1/√︁𝑁𝑗if and only if node 𝑖is assigned to
cluster𝑗, and 𝑪𝑖,𝑗=0otherwise, so that 𝑪⊤𝑪=𝑰. The super-nodes and super-edges are constructed
according to 𝑪. Moreover, the super-node features 𝑿′are set to be the weighted average of node
features within each cluster, i.e., 𝑿′=𝑪⊤𝑿. The super-nodes labels 𝒀′are constructed by taking
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 9

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 9
the dominant label of nodes in each super-node, i.e., 𝒚′
𝑘=ONEHOT(arg max((𝑪⊤𝒀)𝑘)). [58]
considers different partition algorithms for coarsening the graph: spectral clustering [ 126] and
several partition algorithms from [ 84] with restricted spectral approximation. Empirically, the
Variation Neighborhoods method from [84] works better than other partition algorithms.
There are some other related works on the topic of graph coarsening and GNNs, but their goals
are different from our focus on graph coarsening for GNN training acceleration. GOREN [10] uses
GNNs as a tool to help do better graph coarsening, where the goal is to output a coarse graph
with better Rayleigh loss and Eigenerror. GraphZoom [26] uses graph coarsening to improve the
accuracy and scalability of graph embedding algorithms like DeepWalK [98] and Node2Vec [43].
3.1.2 Graph Sparsification Graph sparsification by removing redundant edges is a natural
idea of graph modification to reduce the computation graph and accelerate GNN training. An
illustration of graph sparsification is shown in Figure 2 middle row. Like graph coarsening, graph
sparsification has been studied and widely used before GNNs became popular [ 6,111,118,119].
Graph sparsification methods usually try to find a sparsified G′that can well approximate the
properties of the original G. That is keeping all the nodes in Gand removing less-important edges
without disconnecting G. Example properties preserved by existing graph sparsification methods
include the total weight of cuts [ 6], spectral properties [ 118,119], and hierarchical structures [ 111].
With the goal being GNN training acceleration, the above graph sparsification algorithms can be
applied as a pre-processing step before GNN training starts. Since the sparsification algorithms are
usually highly scalable and only need to be run once, the time it takes is negligible compared to
GNN training. A sparsified graph can reduce the full-batch time complexity of training an 𝐿-layer
GNN from𝑂(𝐿|E|)to𝑂(𝐿|E′|), where|E′|could be much smaller than |E|.
There are some related works performing GNN-based graph sparsification to improve GNN
accuracy [ 74] or robustness [ 170] but not for GNN acceleration explicitly. Their difference from
the traditional methods is the sparsification criteria consider both Gand the GNN instead of the
properties ofGonly. These criteria are usually formed as optimization problems and solved while
training the GNN. For example, [ 74] finds a sparsified graph G′via the alternating direction method
of multipliers. [ 170] learns a sparsification strategy by generating sparse k-neighbor subgraphs. It
is not guaranteed that these sparsification methods will always accelerate single GNN training.
Although the sparse graph reduces Eto accelerate message passing and is likely to make the model
training converge faster, solving the extra optimization problem takes time and can slow down the
training. These methods may be used to accelerate the training of the GNNs after the first one if
multiple GNNs need to be trained on the same graph, as for the neural architecture search task.
A closely related concept that may be confused with sparsification is pruning. Some papers use
these two terms interchangeably, but conventionally pruning refers to a modification of the neural
network model in the deep learning context. Thus, model pruning and graph pruning are totally
different concepts with pruning more often referring to the former. We will discuss model pruning
in more detail in Section 4.1 for GNN inference acceleration. For graph pruning, generally speaking,
it is broader than graph sparsification and includes pruning graph nodes as well. However, since
GNN is trained to learn node representations, it is rare to prune nodes in Gcompletely without
preserving their information. What is more often done is dynamically exclude different nodes in
each training iteration, but each node has a probability to be included in the entire training process.
This is the sampling procedure that we will discuss in Section 3.2.
3.1.3 Graph Condensation Graph condensation generates a completely new graph that pre-
serves the training dynamics of the original graph [ 66,67]. In other words, the result of training
two GNNs with identical architecture on the original graph Gand the condensed graph G′(the
generated new graph) respectively should match. Meanwhile, G′can be much smaller than Gand
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 10

10 Zhang, Shichang, et al.
thus accelerating GNN training. An illustration of graph condensation is shown in Figure 2 bottom
row.
GCond [67] matches the training gradients of two GNNs to generate a condensed graph. In
particular, a randomly initialized generative model first generates the condensed graph G′=
(V′,E′,𝑿′)and labels 𝒀′, given a condensation ratio 𝑟, and two GNNs 𝐺𝑁𝑁𝜃and𝐺𝑁𝑁𝜃′are
trained in parallel using stochastic gradient descent (SGD) on GandG′respectively. At each training
step, SGD generates gradients ∇𝜃L(𝐺𝑁𝑁𝜃(G),𝒀)and∇𝜃′L(𝐺𝑁𝑁𝜃′(G′),𝒀′)for updating 𝜃and𝜃′
respectively. Then gradient matching is performed by tuning the generative model to minimize the
distance between∇𝜃L(𝐺𝑁𝑁𝜃(G),𝒀)and∇𝜃′L(𝐺𝑁𝑁𝜃′(G′),𝒀′), so thatG′and𝒀′better preserve
the training dynamics can be generated in the next iteration. For the generative model, since E′and
𝑿′depend on each other in general graphs, GCond chooses to generate 𝑿′first and then construct
E′from 𝑿′. Specifically, GCond initializes 𝑟|V|node features 𝑿′as free learnable parameters,
which will be directly updated by the gradient-matching loss. To generate condensed edges E′,
GCond uses an MLP as a feature-based structure prediction model, which takes input 𝒙′
𝑖and𝒙′
𝑗
and predicts edge 𝑖-𝑗. Finally, for condensed labels 𝒀′, GCond initializes a 𝒚′for each 𝒙′
𝑖and fixes
𝒚′by randomly selecting a class following the label distribution in the original graph Gwithout
learning. Since all 𝑿′are randomly initialized, it doesn’t matter what is the value of 𝒚′assigned
to each 𝒙′
𝑖, and only the distribution of 𝒀′matters. GCond sets the distribution 𝒀′to follow the
original 𝒀so that the class ratios in Gare kept inG′.
Note that like the GNN-based sparsification methods, GCond can only accelerate training for the
second and subsequent GNNs when multiple GNNs need to be trained on the same graph because
the gradient matching step requires training a GNN on the original graph G. However, GCond is
shown to achieve an astonishing less than 1% condensation ratio on several graph benchmarks
while maintaining over 95% accuracy as the original GNN, and the same condensed G′can be
transferred to train GNNs with different architectures to achieve good performance. It is thus a great
fit for tasks like neural architecture search. A neural architecture search experiment is performed
in GCond, and hundreds of GNNs were quickly trained to achieve competitive performance after a
goodG′is generated. However, to accelerate the training of a single GNN, GCond should either be
combined with other acceleration methods or better condensation strategies that can match 𝐺𝑁𝑁𝜃
and𝐺𝑁𝑁𝜃′more efficiently without training to convergence is needed.
3.2 Sampling
Sampling is a prevalent technique for accelerating GNN training, especially under computation
resource limitation [ 81,110]. The basic idea is to dynamically select a subset of nodes and edges
during each training iteration and build the computation graph only with the selected nodes and
edges. Sampling can greatly reduce the computation graph size and accelerate model training. It is
found that the model accuracy can also be maintained by sampling since the full neighborhood is
not always required to predict a target node. The shared goal of different sampling algorithms is to
reduce the time and memory consumption as much as possible while maintaining model accuracy.
Note that sampling is different from the graph modification discussed above. Sampling is done
dynamically and implicitly and there is no intermediate modified graph output.
In the following, we introduce different sampling algorithms in a unified framework. Throughout
the section, we use SAMPLE(·)to denote performing the sampling operation and use ·on our
notations to denote the sampled version. Given each target node 𝑣, instead of aggregating from all its
full neighborhoodN(𝑣)for conducting message passing, sampling selects a subset of neighboring
nodes, denotedN(𝑣), which provides the most important information for predicting 𝑣. A sampled
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 11

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 11
version of GNN message passing in Equation (1) and Equation (2) can approximate the node
representations as:
𝒉(𝑙)
𝑣≈UPDATE(𝒉(𝑙−1)
𝑣,AGGR({𝜙(𝒉(𝑙−1)
𝑣,𝒉(𝑙−1)
𝑢)|𝑢∈N(𝑣)(𝑙)})) (4)
𝑯(𝑙)≈𝜎(𝑷(𝑙)𝑯(𝑙−1)𝑾(𝑙)) (5)
whereN(𝑣)(𝑙)and𝑷(𝑙)indicate the layer- 𝑙-sampled neighborhood and its corresponding filter
matrix respectively.
The central idea of sampling is to find informative neighborhoods that can construct small but
effective computation graphs. Existing sampling algorithms can be categorized into three types
according to their input to the SAMPLE(·)operation: 1) node-wise , which takes a single node
neighborhood as input and sample within it; 2) layer-wise , which takes all nodes in the same
computation graph layer and sample all their neighborhoods jointly; 3) subgraph-wise , which
takes a whole graph as input and sample a subgraph as neighborhoods for nodes in the graph.
Figure 3 illustrates the workflow of these three types of sampling algorithms. In the following, we
discuss the details of sampling algorithms in each category and summarize their pros and cons.
Moreover, we compare the performance of the most popular sampling methods in terms of
their run time and accuracy. The results are shown in Table 2. The results are on three commonly
used benchmark datasets: Pubmed [ 109], PPI [ 178], and Reddit [ 46].Pubmed [109] is a citation
graph consisting of 19,717 scientific publications pertaining to diabetes from the PubMed database
as the graph nodes and 44,338 citations as graph links. Each publication is described with a 500-
dimensional TF/IDF weighted word vector feature [ 88]. The task is to classify nodes into one of
the three classes each representing one type of diabetes. PPI[178] is a graph describing protein-
protein interaction consisting of 14,755 nodes and 225,270 edges, each node is described with a
50-dimensional feature vector based on the positional gene set, motif gene sets, and immunological
signatures. The task is to classify nodes into one of the 121 classes based on their gene ontology
set.Reddit [46] is a graph constructed from Reddit posts. Each of the 232,965 nodes corresponds to
one Reddit post, and two nodes are connected if the same user comments on both posts, resulting
in 11,606,919 links in the graph. The node features are 602-dimensional vectors concatenating the
average embedding of the post title and that of all the post’s comments, the post’s score, and the
number of comments made on the post, where the average embedding is obtained by averaging
the 300-dimensional GloVe CommonCrawl word vectors [ 97]. The task is to classify posts (nodes)
into one of the 41 Reddit communities. For all the datasets and all methods, we report their relative
and absolute accuracy and total running time. The layer-wise sampling methods work better on
Reddit, and the subgraph-wise methods work better on PPI.
3.2.1 Node-wise Sampling Node-wise sampling methods are applied to each target node 𝑣at
each layer𝑙. The node-wise sampled neighborhood is obtained by: N(𝑣)(𝑙)←SAMPLE(𝑙) N(𝑣).
Given𝑠(𝑙)
𝑛𝑜𝑑𝑒as the number of neighboring nodes we want to sample for each node at layer 𝑙, the
sampling operation SAMPLE(𝑙)(N(𝑣))is performed 𝑠(𝑙)
𝑛𝑜𝑑𝑒times. We use the distribution 𝑢∼𝑝(𝑙)
𝑣(𝑢)
to denote the probability node 𝑢is sampled as part of the computation graph for node 𝑣. An example
of the sampling procedure is illustrated in Figure 3(a).
GraphSAGE [45] is the pioneering work to reduce node receptive field with node-wise sampling.
For each layer 𝑙, GraphSAGE uniformly samples a fixed number ( 𝑠(𝑙)
𝑛𝑜𝑑𝑒) of neighboring nodes for
each node. In other words, each neighbor node 𝑢inN(𝑣)has an equal probability 1/𝑑(𝑣)to be
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 12

12 Zhang, Shichang, et al.
ABCHDENode-wiseLayer-wiseSubgraph-wiseComputation GraphSampled GraphNodes sampled in the 1st GNN layerNodes sampled in the 2nd GNN layerSelected input nodes in the current batchABCEFDEGABCDEFHGABCDEFHGABCDEFHGBACABCABC
Fig. 3. Illustration of different Graph Sampling Methods. Node-wise Sampling methods sample per node
for each computation layer, which might lead to redundant nodes (e.g., node Eis sampled twice), and missing
edges (e.g., edge between node Cand node Dis missing); Layer-wise Sampling methods sample per layer based
on nodes in the previous layer. Subgraph-wise Sampling methods sample a list of nodes and their induced
subgraph, and then conduct message passing via all the edges within the sampled subgraph.
sampled, i.e. 𝑝(𝑙)
𝑣(𝑢)=1/𝑑(𝑣). The same sampling procedure is repeated multiple times to construct
a computation graph layer that is smaller than the full neighbor.
The GraphSAGE sampling algorithm can also be formalized in matrix form, with 𝑷(𝑙)𝑯(𝑙−1)
denoting the full-neighbor aggregation and 𝑷(𝑙)𝑯(𝑙−1)denoting the aggregation with sampling.
The sampled filter matrix 𝑷(𝑙)can be computed with a mask matrix M∈R𝑁×𝑁as𝑷(𝑙)=M◦𝑷(𝑙).
Here◦means item-wise multiplication, and the mask matrix Mis defined as:
M𝑢,𝑣=(|N(𝑣)|
𝑠(𝑙)
𝑛𝑜𝑑𝑒𝑢∈N(𝑣)(𝑙), whereN(𝑣)(𝑙)←SAMPLE(𝑙) N(𝑣)
0 otherwise(6)
The sampled estimation 𝑷(𝑙)𝑯(𝑙−1)is an unbiased estimation of ground-truth 𝑷(𝑙)𝑯(𝑙−1), i.e., the
expectation E
𝑷(𝑙)𝑯(𝑙−1)
=𝑷(𝑙)𝑯(𝑙−1), as sampling is uniform over each node.
VR-GCN [13] proposes to use historical embeddings in GraphSAGE workflow to reduce embed-
ding variance and accelerate training convergence. VR-GCN applies the same sampling operation
with the same uniform probability distribution as in GraphSAGE. Under the assumption that
the model weight 𝑾(𝑙)would not change significantly between two training iterations, VR-GCN
stores the historical embeddings 𝑯(𝑙,ℎ𝑖𝑠𝑡𝑜𝑟𝑦)from the previous epoch. With that, 𝑷(𝑙)𝑯(𝑙−1)is
be estimated by 𝑷(𝑙)𝑯(𝑙−1)+𝑷(𝑙)𝑯(𝑙−1,ℎ𝑖𝑠𝑡𝑜𝑟𝑦). Such historical embedding usage causes some
extra memory, but it is usually acceptable. The benefit is that VR-GCN can enjoy the advantage
of a smaller training variance that accelerates training convergence. In practice, [ 13] found that
VR-GCN can get a comparable performance of GraphSAGE and other existing methods with a
much smaller sample size.
Except for GraphSAGE and VR-GCN, there are many other works exploring different aspects of
node-wise sampling. Ying et al . [151] propose PinSAGE Sampler , which is a node-wise sampling
with random walk frequencies-based sampling distribution. It allows GNNs to be applicable to
industry-level large-scale recommender systems. MVS-GNN [23] extends VR-GCN by explicitly
reducing both the embedding approximation variance and stochastic gradient variance. GCN-BS
[82] formulates the neighbor sampling as a multi-armed bandit problem, and proposes to estimate
a learnable sampler that can be updated towards minimal sampling variance. BNS [149] first
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 13

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 13
Table 2. Performance comparison of different sampling methods. Absolute performance is adopted from [ 81],
and we add relative performance with GraphSAGE set as the benchmark (1 ×), as it is the simplest and the
most widely used sampling method. All results assume a 2-layer backbone GCN model with the sampling
size and the optimizer the same as the original papers.
Pubmed PPI Reddit
Accuracy (%) Total Time (s) Accuracy (%) Total Time (s) Accuracy (%) Total Time (s)
No Sample GCN 78.6 (0.97×) 24.8 (0.65×) N/A N/A N/A N/A
Node-wiseGraphSAGE 81.4 (1×) 37.9 (1×) 61.6 (1×) 47.5 (1×) 95.0 (1×) 326.7 (1×)
VR-GCN 81.4 (1×) 5.9 (0.16×) 97.5 (1.58×) 83.5 (1.76×) 96.3 (1.01×) 390.6 (1.20×)
Layer-wiseFastGCN 86.3 (1.06×) 13.7 (0.36×) N/A N/A 92.8 (0.98×) 294.5 (0.90×)
AS-GCN 89.5 (1.10×) 39.1 (1.03×) N/A N/A 96.5 (1.02×) 1633.7 (5.00×)
LADIES 75.7 (0.93×) 6.4 (0.17×) N/A N/A N/A N/A
Subgraph-wiseCluster-GCN N/A N/A 95.8 (1.56×) 202.3 (4.26×) 95.8 (1.01×) 981.8 (3.00×)
GraphSAINT N/A N/A 98.7 (1.60×) 550.1 (11.58×) 96.7 (1.02×) 124.0 (0.38×)
conducts random node-wise sampling, then uniformly selects a small portion of sampled neighbors
to stochastically blocks their ongoing expansion.
Although node-wise sampling methods can alleviate the scalability challenge of large-scale GNN
training by reducing the receptive field and can guarantee that each node at each layer can have
a reasonable number of connected neighbors to update its hidden representation, they still have
some limitations:
•The exponential neighbor growth is not avoided as the GNN goes deeper, even though the
neighborhood size is reduced. As shown in Figure 3(a), the computation graph keeps growing
by a factor of size 𝑠𝐿
𝑛𝑜𝑑𝑒in each layer.
•A node can be repeatedly sampled since the sampling procedures for each node in the same
layer is independent. In Figure 3(a), node Eis sampled by both node Band node C, and the
message passing computation is repeated twice in the computation graph (marked by the
blue dotted box).
•Some edges between sampled nodes across layers are missing, which leads to computation
resource waste. In Figure 3(a), node Cand node Dare connected in the original graph.
However, since node Dis sampled by node Bbut not node C, the edge (B, C) is missing,
which could have been used to pass messages from BtoCgiven these two are included in
the computation graph anyway.
3.2.2 Layer-wise Sampling Layer-wise sampling methods are applied to all nodes in the same
computation layer instead of a single node. When a set of nodes are selected as the prediction target
during training, a computation graph is constructed with all these nodes belonging to the first
computation layer. Their direct neighbors belong to the next computation layer, and their 2nd-hop
neighbors belong to the computation layer after, and so on. Let V(𝑙)denote all the nodes at the 𝑙−th
layer, layer-wise sampling methods choose a subset of nodes from Vto be in the next computation
layer. In this way, for each node 𝑣∈V(𝑙), its layer-sampled neighborhood is the intersection of the
whole layer-sampled node set and its original neighborhood: N(𝑣)(𝑙)←SAMPLE(V)∩N(𝑣). Let
𝑠(𝑙)
𝑙𝑎𝑦𝑒𝑟be the number of neighbor nodes to be sampled for 𝑙, each Layer-wise sampling operation
SAMPLE(𝑙)(V) is performed 𝑠(𝑙)
𝑙𝑎𝑦𝑒𝑟times. Finally, the sampled nodes at the current computation
layer𝑙would be used to construct the neighborhood of the later computation layer 𝑙+1for message
passing. An example of the sampling procedure is illustrated in Figure 3(b).
FastGCN [12] proposes the first layer-wise importance sampling scheme to address the scalability
issue in node-wise sampling. For each computation layer, FastGCN samples nodes from the full node
setVwith the node-degree-based probability distribution: 𝑝(𝑙)
V(𝑢)=∥𝑷:,𝑢∥2
2/Í
𝑢′∈V∥𝑷:,𝑢′∥2
2. Note
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 14

14 Zhang, Shichang, et al.
that, in FastGCN, the same sampling distribution is used for different layers, i.e. 𝑝(𝑙)
V(𝑢)=𝑝V(𝑢)
for all𝑙. Next, the approximation of 𝑷(𝑙)𝑯(𝑙−1)with 𝑷(𝑙)𝑯(𝑙−1)can be computed by constructing
𝑷(𝑙). The(𝑢,𝑣)entry of 𝑷(𝑙)is given by
𝑷(𝑙)
𝑣,𝑢= 
𝑷(𝑙)
𝑣,𝑢
𝑠(𝑙)
𝑙𝑎𝑦𝑒𝑟𝑝V(𝑢)if𝑣is sampled in layer ( 𝑙) and𝑢∈N(𝑣)
0, otherwise(7)
Despite its efficacy and efficiency, FastGCN suffers the connectivity challenge since the sampling
for each layer is conducted independently. In other words, the nodes sampled in two consecutive
layers may not be connected, some nodes sampled in the previous layer may not be able to pass
their information to the next layer, and some nodes sampled from the later layer may not be able
to get any information from its neighborhood. This would lead to a very sparse 𝑷(𝑙)matrix and
may make the approximated embeddings have a large variance.
AS-GCN [54] proposes an adaptive layer-wise sampling method to accelerate the training of
GCNs on large-scale graphs. The proposed method samples the lower layer conditioned on the top
one, where the sampled neighborhoods are shared by different parent nodes and over-expansion is
avoided. AS-GCN explicitly calculate the embedding variance, and add a regularization to reduce the
variance. In this way, the model could learns to give higher weights to those more connected nodes,
which has been demonstrated to achieve higher accuracy than FastGCN. Additionally, AS-GCN
preserves second-order proximity by formulating a skip connection across two layers.
LADIES [179] adopts similar intuition as AS-GCN to only sample within the the immediate
neighborhood of the later layer. This constraint can guarantee that each sampled node in the
previous layer will have at least one successor node in the later layer. In other words, it can make
sure all the sampled nodes can deliver their information to at least one of their neighbor nodes in
the next layer. To apply this constraint, a row selection matrix 𝑸(𝑙)∈R𝑠(𝑙)
𝑙𝑎𝑦𝑒𝑟×|V(𝑙)|is constructed
as
𝑸(𝑙)
𝑣,𝑢=1if𝑣is sampled in layer 𝑙and𝑢∈N(𝑣)
0,otherwise,(8)
Then, node 𝑢in layer(𝑙−1)is sampled with probability: 𝑝(𝑙−1)
V(𝑢)=∥𝑸(𝑙)𝑷:,𝑢∥2
2
∥𝑸(𝑙)𝑷∥2
𝐹. A𝑷(𝑙)can be
constructed in the same way as in Equation (7).
To maintain the scale of embeddings in the forward process, LADIES also normalizes 𝑷(𝑙)to
compute the 𝑷(𝑙)𝑯(𝑙−1). It computes 𝑷(𝑙)←𝑫−1
𝑷(𝑙)𝑷(𝑙), where 𝑫𝑷(𝑙)is a diagonal matrix with its
(𝑖,𝑖)entry equalsÍ
𝑗𝑷(𝑙)
𝑖𝑗.
Layer-wise sampling methods solve the exponential neighborhood expansion limitation for
node-wise sampling methods and enjoy a linear time and memory complexity with respect to the
number of GNN layers. However, the connectivity problem still exists despite the efforts to make
the sampled nodes in two consecutive layers connected and guarantee all the sampled nodes can
pass out their information. The connectivity problem says that sampled nodes in a later layer can
have very limited or no sampled neighbors from the previous layer. Thus, the messages passed
between two consecutive layers for some sampled nodes can be insufficient, which degrades the
model performance.
3.2.3 Subgraph-wise Sampling Subgraph-wise sampling methods are applied with the whole
graph as input and output a sample subgraph, i.e., G← SAMPLE(G). Afterward, GNN training
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 15

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 15
takes the sampled subgraph as input to conduct full-batch message passing. For these methods,
the sampling procedure may be conducted before the training starts as a pre-processing step,
which makes them more similar to graph modification methods. However, there are still two key
differences that make these methods sampling rather than modification. First, to overcome the
scalability challenge, the subgraphs are usually much smaller than the original, which only represent
local information instead of global information as the graph modification methods discussed in
Section 3.1. Second, directly training with such pre-processed small subgraphs often result in
low model accuracy. Thus, some stochasticity needs to be added back during training to improve
accuracy, which makes the modification in these methods dynamic, e.g., the ClusterGCN method
below.
ClusterGCN [21] utilizes graph clustering algorithms (i.e., METIS [ 69]) to partition the input
whole graphGinto clusters, where the nodes within each cluster form a densely connected
subgraph and inter-cluster edges are minimized. Afterward, ClusterGCN conducts full-batch GNNs
training on each subgraph cluster, which significantly improves the scalability of GNN training.
One limitation of this approach is that the clustering result is fixed, so inter-cluster edges will be
missing throughout training. ClusterGCN further proposes to randomly choose several subgraphs
in each training mini-batch and add back inter-cluster edges between the subgraphs. This operation
covers all the inter-cluster edges in the long run and is proven to provide an unbiased estimate of
the full-neighborhood embeddings.
GraphSAINT [158] is another widely used subgraph-wise sampling method, which first sam-
ples nodes and then constructs subgraphs induced by the sampled nodes. GraphSAINT proposes
four different algorithms to sample nodes: 1) node sampler, which randomly samples nodes with
probability proportional to node degrees; 2) edge sampler, which samples edges with probability
proportional to the inverse of the end node degrees, and then keeps the end nodes. The edge
sampler is proven to reduce the estimation variance; 3) random walk sampler, which conducts
random walks on a random set of nodes and selects nodes on the walks; 4) multi-dimensional
random walk, which is similar to random walk but adds an additional requirement to start random
walking from high-degree nodes in the sampled set. Among the four proposed algorithms, the
random-walk sampler performs better empirically and thus is more widely used.
Shadow-GNN [157] samples different subgraphs for each target node (similar to ego-networks).
The sampling could be done via random walk or personalized page rank. Specifically, the authors
introduce a design principle that separates the depth of a GNN from its receptive field size, which is
the range of nodes and edges that influence a node’s representation. By doing so, the depth of a GNN
can be increased without increasing its receptive field size exponentially. It also presents a theoretical
analysis of expressivity from three different perspectives, and also rich design components (e.g.,
subgraph extraction functions, architecture extensions) to implement such design principles.
There also exist many other follow-up works that conduct subgraph-wise sampling. RWT [4]
proposes a ripple walk sampler. It first randomly samples a small set of nodes, then it randomly
samples nodes from the neighbors of nodes in the set and adds them to the set. The process is
repeated until a prespecified budget is filled. Zeng et al . [159] provides the parallelized version
of the existing frontier sampling which can sample subgraphs that can approximate the original
graph w.r.t. various connectivity measures. In addition, some layer-wise sampling methods could
be used to construct subgraphs [53].
The key advantage of subgraph-wise sampling methods is that they do not dependent on the
GNN model and the generated embedding. Therefore, they could be done purely in parallel or
beforehand the training procedure. However, this advantage could also be regarded as a limitation,
as the sampling only considers the graph structure without taking into account model training
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 16

16 Zhang, Shichang, et al.
dynamics. Therefore, it still remains an open question how to incorporate some variance-reduction
studies, including VR-GCN [13] and AS-GCN [54], into the subgraph-wise sampling procedure.
4 GNN Acceleration: Inference Algorithms
GivenG𝑡𝑟𝑎𝑖𝑛 as the training graph, G𝑡𝑒𝑠𝑡as the test graph, and GNN𝜃∗as the optimal GNN trained
onG𝑡𝑟𝑎𝑖𝑛, GNN inference generates representations for nodes in G𝑡𝑒𝑠𝑡and uses them to make
predictions. The goal of GNN inference acceleration is to construct another model ˜GNN ˜𝜃such that
˜GNN ˜𝜃inference onG𝑡𝑒𝑠𝑡has similar accuracy as GNN𝜃∗but is much faster. During the construction
of ˜GNN ˜𝜃, access toG𝑡𝑒𝑠𝑡should not be assumed, since fast inference is desired on unseen test
graphs for real-world use cases. In contrast, acceleration methods usually assume access to G𝑡𝑟𝑎𝑖𝑛,
since some extra work like retraining is usually needed for a faster ˜GNN ˜𝜃to achieve similar accuracy
asGNN𝜃∗. In practice,G𝑡𝑒𝑠𝑡andG𝑡𝑟𝑎𝑖𝑛 can have substantial overlap or even be the same graph
(called the transductive setting). For the sake of generality, we assume they are different graphs in
our discussion below.
GNN inference acceleration can be quite different from GNN training acceleration. Training
acceleration is more graph-centric and inference acceleration is more model-centric. For training
acceleration, the GNN architecture is fixed and only the process for finding 𝜃∗is accelerated.
Methods focus on modifying the graph or sampling from the graph to reduce the computation
graph and thus accelerate training. For inference acceleration, ˜GNN ˜𝜃can be substantially different
from GNN𝜃∗in terms of model architecture, parameter numerical precision, and parameter values.
Changes are made only on the model. The test graph G𝑡𝑒𝑠𝑡is assumed unseen and thus untouched.
Only for cases where G𝑡𝑒𝑠𝑡andG𝑡𝑟𝑎𝑖𝑛 are overlapped or being the same, some graph-centric
training acceleration methods like graph sparsification can also accelerate inference because the
sparsification is carried from G𝑡𝑟𝑎𝑖𝑛 toG𝑡𝑒𝑠𝑡. Even in these cases, graph modification methods like
coarsening and condensation do not fit because they can affect the node for inference and thus lose
the prediction target. Sampling methods are also rarely used because usually only one forward
computation is performed in inference rather than many times in training. Therefore, sampling
only a random partial neighborhood, in general, can guarantee neither prediction accuracy nor
consistency.
Fast inference with ˜GNN ˜𝜃does not mean ˜GNN ˜𝜃can be constructed fast. Rather, the speed of
˜GNN ˜𝜃construction is often sacrificed to achieve inference acceleration. Nevertheless, as long as a
˜GNN ˜𝜃can make fast inference is constructed, the goal is achieved. Even slowly constructed ˜GNN ˜𝜃
will be useful for many practical scenarios. Especially when the construction doesn’t need to be
done very frequently, but inference latency is a big concern. In this section, we discuss three kinds
of methods for inference acceleration: pruning, quantization, and distillation. All three methods
are widely used for accelerating general DNNs. The works we discuss below adapt them on GNNs
for graphs. The major challenge they solve is how to preserve the graph structure information and
solve the major source of latency caused by message aggregation.
4.1 Pruning
Model pruning or simply pruning is a popular approach for accelerating neural network (NN)
inference [ 48]. The research of NN pruning started in the 1990s [ 101], with the idea being removing
unimportant parts of an NN while maintaining its accuracy. The most common NN pruning method
is the weight magnitude pruning, which prunes the connections between the NN “neurons" with
small weights or equivalently reduces the L1 norm of NN weights. Like other inference acceleration
methods, pruning provides a trade-off between model speed and accuracy. When more weights are
removed from the NN, the model is likely to become faster but less accurate, and vice versa.
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 17

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 17
Pruning methods have been proposed for accelerating GNN inference [ 16,174]. Notice that GNN
pruning targets the model and is different from graph pruning which targets the graph data. Graph
pruning is a concept related to graph sparsification, which we discussed in Section 3.1.2. For GNN
model pruning, selecting which GNN weights to prune is the central question and is nontrivial. It
is challenging because GNN inference on a graph node depends on its local graph structure and
neighboring nodes, which is more complicated than the regular NN inference on data types with
independent instances like image classification.
Zhou et al. [174] propose to accelerate GNN inference with channel pruning, where channels
refer to the dimensions of the node representations 𝒉, or equivalently the rows of the GNN weight
matrix 𝑾. Given a GNN with weights 𝑾, Zhou et al. formulate the pruning problem as a LASSO
regression problem for each GNN layer, and the objective is:
arg min
ˆ𝜷,ˆ𝑾∥𝑯(𝑙)−𝑷𝑯(𝑙−1)(ˆ𝜷⊙ˆ𝑾(𝑙))∥2+𝜆∥ˆ𝜷∥1. (9)
There are two sets of parameters to optimize for this objective: a learnable mask ˆ𝜷∈R𝐹that
chooses channels to prune from 𝐹total channels, and an updated GNN weight matrix ˆ𝑾for better
accuracy of the pruned model. ˆ𝜷is initialized as all ones and ˆ𝑾is initialized as 𝑾.⊙denotes
an element-wise product for a column vector on each column of a matrix. 𝑷denotes the filter
matrix for message propagation. Zhou et al. optimize these two sets of parameters alternatively. ˆ𝜷
is optimized first with fixed ˆ𝑾and stochastic gradient descent (SGD). Then ˆ𝑾can be computed
in closed-form as a least square solution for fixed ˆ𝜷. The L1 regularization shrinks entries of ˆ𝜷
to zeros, which leads to the pruning of corresponding channels. Zhou et al. show that the GNN
accuracy only decreases marginally when 3/4of the GNN channels are pruned, which achieves
3.27×speedup on large-scale graph inference.
Another recent popular line of pruning research is the Lottery Ticket Hypothesis (LTH) [ 36],
which shows that a pruned sparse NN can be retrained to achieve similar accuracy as the original NN.
The significance of LTH compared to the regular pruning methods is in the retraining. Most pruning
methods result in oneset of NN weights with minor accuracy loss, but they cannot guarantee
that the pruned NN can be retrained with the original ML objective and still maintain similar
accuracy loss on the same dataset. LTH shows that the same level of accuracy loss can be achieved
with iterative magnitude pruning (IMP), i.e., prune the small NN weights iteratively and perform
retraining after each pruning.
Chen et al. [16] test LTH for GNNs by doing IMP. They indeed observe speed gain and marginal
accuracy loss even for retrained GNNs. You et al. [154] further test the early-bird LTH on GNNs,
which means the retraining in each IMP iteration can stop very early before it converges. Notice
that pruning accelerates model inference but not always model training. The iterative pruning
requires retraining the model multiple times, possibly resulting in a slower training procedure than
before. The advantage of early-bird LTH is to save iterative training time for finding the optimal
pruned model. Moreover, in [ 16] and [ 154], the input graph is pruned together with the GNN model,
so parts of these methods are doing graph pruning (sparsification) as well.
4.2 Quantization
Quantization is a widely used technique for accelerating general ML models. The idea is to use lower
numerical precision for the model parameters, say replace the 32-bit floating point numbers (FP32)
with 8-bit integers (INT8). For DNNs, a significant part of latency comes from matrix multiplications
which boil down to multiply–accumulate (MAC) operations. Quantized low-precision DNNs have
a smaller number of MAC operations and thus accelerate inference. Besides the MAC operation
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 18

18 Zhang, Shichang, et al.
reduction, quantization also comes with other benefits that further accelerate the model, like
reducing memory access time [ 44,60]. Another reason for quantization to become popular is that
it is easy to use. The numerical precision can be easily changed for any given model. The only
concern is that blindly reducing the numerical precision of a given model at the inference stage
may result in a large accuracy loss. Quantization research is thus aiming for inference acceleration
while maintaining inference accuracy as much as possible, and many quantization techniques for
achieving this goal have been proposed [ 47,72]. Below, we first introduce the idea of quantization
on general NNs and then get to quantization for GNNs.
Quantization on NNs boils down to operation on tensors, which could be weights or activations
in NNs. Suppose we want to quantize a given tensor 𝒙into a𝑞-bit representation. Let 𝑞𝑚𝑖𝑛and𝑞𝑚𝑎𝑥
be the minimum and maximum values we want the 𝑞-bit representations to take. Let 𝑠be a factor
that scales 𝒙to the range[𝑞𝑚𝑖𝑛,𝑞𝑚𝑎𝑥]. Let𝑧be a special zero-point . The zero-point is meant to
ensure the point zero is quantized with no error, which is important as NNs often have operations
like zero padding. Both 𝑠and𝑧are in𝑞-bit. We use the floor function ( ⌊·⌋) to indicate cutting the
representation to 𝑞-bit (i.e., assuming quantize to integers). Then the quantized tensor 𝒙𝑞is
𝒙𝑞=⌊𝒙
𝑠+𝑧⌋. (10)
The quantized 𝒙𝑞is in the desired precision. A dequantization is then performed to map 𝒙𝑞toˆ𝒙:
𝒙≈ˆ𝒙=(𝒙𝑞−𝑧)𝑠. (11)
With the quantize and then dequantize operations, a 𝑞-bit representation ˆ𝒙can be used to replace
model tensor 𝒙.ˆ𝒙closely approximates 𝒙, but computation using ˆ𝒙will be much faster.
With the same general quantization idea, there are in general two types of quantization algorithms
for NNs: Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) [ 32,95]. PTQ
takes a trained high-precision NN and directly converts it to lower precision. PTQ is easy to use
and can be applied to any architecture without accessing the training pipeline or any data. PTQ
mainly focuses on two questions to trade off accuracy for speed: 1) Which tensor object should be
quantized, e.g., NN weights or NN activations? 2) What is a proper quantization bit 𝑞and a proper
range[𝑞𝑚𝑖𝑛,𝑞𝑚𝑎𝑥]? The concern of PTQ is that when the precision 𝑞is set low, like 4 bits and
below, the accuracy can always be low no matter which tensor or range is chosen. QAT algorithms,
on the other hand, model the quantization error during training. For example, train a model with
low-precision along with the original high-precision model and minimize the error between them.
QAT can thus find quantized models better than PTQ in general. The cost of QAT is that the model
training time is longer and data is required.
Existing quantization methods are mostly for convolutional neural networks (CNNs). Quantiza-
tion for GNNs, however, has its own challenges and requires special treatment. We now discuss
GNN quantization methods for both PTQ and QAT.
SGQuant [33] is one of the earliest work studying GNN quantization. It is a PTQ method
and focuses on the two core PTQ questions mentioned above. For the first question about the
quantization object, SGQuant focuses on the quantization of node representations learned by each
GNN layer, i.e.,{𝑯(𝑙)}1≤𝑙≤𝐿. SGQuant makes the empirical observation that node representations
consume much more memory than GNN weights 𝑾. This observation and object choice is unique
to GNNs but not for other NNs because of node dependency. Unlike other NNs which process
one data instance one at a time, for one target node, GNNs store node representations for all the
multi-hop neighboring nodes within the receptive field of the target, which could involve a huge
number of nodes. For the second question, SGQaunt focuses on selecting the quantization bit 𝑞
and simply uses the full range of 𝑞-representable numbers. SGQuant proposes multi-granularity
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 19

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 19
quantization, where different objects are associated with different bits to optimize quantization
efficiency. Applying this principle to graph data, SGQuant uses topology-aware and layer-aware
quantization to group node representations, where nodes with similar degrees are grouped together,
and representations from the same layer are grouped together. This is essentially grouping rows
of{𝑯(𝑙)}1≤𝑙≤𝐿, where each row corresponds to one node and 𝑙corresponds to layer depth. Then
node representations are assigned with different bits depending on which group they are from.
The multi-granularity quantization decides the bits for each group by optimizing quantized model
accuracy over a set of precision candidates.
Degree-Quant [121] is a QAT method quantizes both GNN weights 𝑾and intermediate node
representations 𝑯, which results in INT8 GNNs with comparable accuracy as their FP32 counterparts
but up to 4.7X faster. Degree-Quant is like many other QAT methods for general NNs, for which
a quantized model is trained via gradients by simulating its numerical errors from the original
model. During the backward pass, the straight-through estimator (STE) [ 7] is used to compute
gradients since the quantization operation, e.g., round to integers, can be non-differentiable. The
unique challenge for GNN quantization via STE is that the STE performance is degraded for high-
degree nodes, which leads to poor weight updates. Degree-Quant verifies the degradation both
theoretically and empirically on common GNNs architectures including GCN, GIN and GAT. It
then proposes a protective mask to improve weight update accuracy. A protective mask 𝜷is a
length-𝑁binary vector for all 𝑁nodes in graphG. During the quantization-aware GNN training,
node representation 𝒉𝑖with 𝜷𝑖=1is kept with full precision, while 𝒉𝑖with 𝜷𝑖=0and all
weights 𝑾are quantized. The protective mask 𝜷is sampled as 𝜷∼𝐵𝑒𝑟𝑛𝑜𝑢𝑙𝑙𝑖(𝒑)with 𝒑containing
probabilities for each node. Degree-Quant treats the highest probability and lowest probability in
𝒑as two hyperparameters, and probabilities are assigned to nodes by ranking nodes according to
their degrees and interpolating these two values according to the ranking. High-degree nodes are
assigned higher probabilities, which are encouraged to be kept by the protective mask for more
accurate gradient updates. Degree-Quant also proposes to clip the 0.1% top and bottom quantized
values to improve model accuracy, because large fluctuations in the variance of message-passing
aggregation are observed.
Overall, quantization is one of the easiest GNN acceleration methods to apply but shows signifi-
cant speed improvement. It can also be combined with other acceleration methods to further boost
speed gain. For example, [ 169] combines quantization with network architecture search (NAS). In
[169], a quantization search space are specified and the best quantized GNN is selected via NAS.
4.3 Distillation
Knowledge distillation (KD) or simply distillation is a technique to compress classification models
to smaller and thus faster models [ 9]. It was shown to be able to produce much faster models with
very little, if not none, accuracy loss for image classification and speech recognition [ 49]. The basic
idea of KD is to train two models in a teacher-student fashion. The teacher outputs a probability
vector for each input data containing probabilities for the data to be classified into 𝐶classes. The
probability vectors (or a re-normalized version of them) are in [0,1]𝐶and are called soft labels, in
contrast to the ground-truth hard labels in {0,1}𝐶. Such soft labels will be used to train a simpler
student model. Specifically, the standard KD loss on a data instance 𝑖is computed as a distance
between the teacher model output (soft label) of data 𝑖, i.e., ˆ𝒚𝑡
𝑖, and the student model output of
data𝑖, i.e., ˆ𝒚𝑠
𝑖. Two outputs are matched by minimizing this loss, and thus knowledge is transferred
from the teacher to the student. Since both outputs are probability vectors, a common choice of the
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 20

20 Zhang, Shichang, et al.
distance metric is the Kullback-Leibler (KL) divergence denoted as 𝐷𝐾𝐿,
𝐷𝐾𝐿(ˆ𝒚𝑡
𝑖||ˆ𝒚𝑠
𝑖)=∑︁
𝑗ˆ𝒚𝑡
𝑖𝑗logˆ𝒚𝑡
𝑖𝑗
ˆ𝒚𝑠
𝑖𝑗. (12)
In most cases, the teacher model can be trained alone first, and the student is trained next with
the trained teacher fixed. Thus, we can drop the constant ˆ𝒚𝑡
𝑖𝑗logˆ𝒚𝑡
𝑖𝑗term to get the form after ∝for
student training. The knowledge distillation loss L𝐾𝐷
𝑖will only be used to update the parameters
of the student but not the teacher,
L𝐾𝐷
𝑖∝𝐷𝐾𝐿(ˆ𝒚𝑡
𝑖||ˆ𝒚𝑠
𝑖)=−∑︁
𝑗ˆ𝒚𝑡
𝑖𝑗logˆ𝒚𝑠
𝑖𝑗. (13)
KD can be naturally applied in a semi-supervised setting, a common setting for graph node
classification, by training the teacher with labeled data and generating soft labels for the unlabeled
data to train the student [ 139]. KD has been wildly used for compressing and accelerating DNNs
like CNNs [ 49], and recently it has been extended to GNNs as well [ 143,148,166]. We discuss these
GNN KD methods in detail below. They mostly follow the KD framework discussed above and
mainly differ in two perspectives: 1) What are the teacher and the student models? 2) What is the
KD objective?
LSP [148] is the pioneer work exploring KD for accelerating GNNs. Targeting node classification,
the teacher model LSP considers is a GAT and the student model is a GAT with fewer parameters.
LSP proposes a new KD objective termed local structure preserving loss. The intuition is that node
representations are generated by aggregating messages within the local structure, so the desired
distillation function should transfer not only the final aggregation result but also the local structure
as part of the knowledge. LSP defines the local structure as the similarity between each node to
its neighbors and encourages the local structure of the student to match the teacher. Specifically,
the local structure of node 𝑖is represented as a vector 𝒔(𝑖)∈R𝑑(𝑣), with 𝒔(𝑖)
𝑗representing the local
influence of node 𝑗to node𝑖defined as
𝒔(𝑖)
𝑗=exp(SIM( 𝒉𝑖,𝒉𝑗))Í
𝑘:(𝑘,𝑖)∈Eexp(SIM( 𝒉𝑖,𝒉𝑘)),whereSIM( 𝒉𝑖,𝒉𝑘)=||𝒉𝑖−𝒉𝑘||2
2. (14)
Plug-in node representations 𝒉of the teacher or the student gives the teacher local structure
𝒔(𝑖),𝑡and the student local structure 𝒔(𝑖),𝑠respectively. Then, the local structure preserving loss
L𝐿𝑆𝑃
𝑖is defined using the KL divergence, similarly as the L𝐾𝐷
𝑖in Equation (15). The difference
beingL𝐿𝑆𝑃
𝑖computes the KL-divergence of 𝒔(𝑖),𝑠against 𝒔(𝑖),𝑡, instead of the teacher probability
output ˆ𝒚𝑡
𝑖against student output ˆ𝒚𝑠
𝑖. That is,
L𝐿𝑆𝑃
𝑖=𝐷𝐾𝐿(𝒔(𝑖),𝑠||𝒔(𝑖),𝑡)=∑︁
𝑗∈N(𝑖)𝒔(𝑖),𝑠
𝑗log𝒔(𝑖),𝑠
𝑗
𝒔(𝑖),𝑡
𝑗. (15)
TinyGNN [143] shares a similar intuition as LSP to preserve local structures, but it chooses to add
a special layer termed a Peer-Aware Module (PAM) to the student model instead of defining a new
KD objective as in LSP. TinyGNN adopts the standard KL divergence loss L𝐾𝐷
𝑖. The teacher model
TinyGNN considers is GAT. The student model, however, is a GAT with fewer message-passing
layers than the teacher but a special PAM layer inserted before each message-passing layer. The
idea of PAM is to encourage nodes in the same message-passing hierarchy to directly interact. For
example, when the GNN is only one layer, a message passing step centered at node 𝑣will aggregate
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 21

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 21
messages from its neighbors N(𝑣), but nodes inN(𝑣)will not directly interact with each other
during this message passing step. In other words, the computation graph is a one-layer tree and
has no edges between its leaves. TinyGNN argues that sharing messages between nodes in N(𝑣)
is important for capturing the local structure around 𝑣. This message sharing happens naturally
when the GNN is deep like the teacher, but it is lacking if the GNN is only one or two layers like
the student in TinyGNN. Thus, the PAM is proposed to encourage this message sharing by adding
a self-attention layer between all nodes in N(𝑣). It was shown that a GAT student with PAM has a
higher distilled model accuracy than a regular GAT student, though the speed gain is not as much
given the extra self-attention computation.
GLNN [166] explores KD for accelerating GNNs by making the student model as simple as a
“graph-less neural network”, i.e., an MLP. GLNN follows the standard KL divergence loss L𝐾𝐷
𝑖,
but the student model in GLNN is a pure MLP (a zero-layer GNN) with no node dependency.
GLNN shows that even with a student as simple as an MLP, KD can help the student MLP to
achieve competitive performance to the teacher GNN on many benchmark datasets, even when
used for the inductive prediction on unseen new nodes. GLNN can accelerate GNNs to orders of
magnitude faster because the node dependency is dropped for the student. It is also shown to work
with different teacher GNNs including GCN, GAT, GraphSAGE, etc. However, GLNN is still not
a universal solution for all cases. The analysis in [ 166] shows that GLNN fails to work when the
conditional mutual information between the node features 𝑿and the node labels 𝒀given the graph
structure 𝑨is low. Intuitively, if node labels depend mainly on the graph structure but not the node
feature, then MLPs won’t be able to classify the nodes even with KD.
Overall, the methods above apply the general KD idea to GNNs by adapting graph properties
different perspectives, including both new objects and new student designs. They all show exciting
improvements over the original teacher GNN, but this line of research is still at an early stage. These
KD methods were developed under different settings. Thus, we are unaware of the applicability of
these distillation techniques to each others setting and whether we can combine them. Since KD
in general is always a trade-off between model accuracy and model speed, we don’t know which
method or which combination will give the best result balancing these two perspectives. Also, we
don’t know under what exact conditions, simple methods like GLNN [ 166] will be applicable. These
many interesting open questions make KD a promising direction to explore.
Finally, there are some other related works applying KD techniques on GNNs, for which the goal
is less accuracy loss but not necessarily inference acceleration [ 18,145,173]. Different from the
methods we introduced above, these works often choose students that are not necessarily simpler
than the teacher, which can sometimes outperform the teacher in terms of accuracy, but they could
be even slower than the teacher. We only briefly mention them here given they are less related to
acceleration. GNN Self-Distillation (GNN-SD) [18] considers teacher-free self-distillation where
the knowledge is extracted and transferred between layers of the same GNN. GNN-SD is shown
to consistently achieve better accuracy than the standard KD objective L𝐾𝐷
𝑖.Combination of
Parameterized label propagation and Feature transformation (CPF) [145] combines KD and
label propagation (LP) together. The student in CPF includes an MLP module for learning feature
information and an LP module for learning structure information. The classification accuracy
of a CPF student can generally outperform the accuracy of teacher GNNs after KD. However,
since the LP module still has node dependency like the message passing (labels instead of the
hidden features are passed as messages), the speed gain is likely limited and not discussed in [ 145].
Cold Brew [173] utilizes KD to improve GNN performance on nodes with incomplete or missing
neighbors, i.e., the cold-start nodes. Cold Brew greatly improves the accuracy on these cold-start
nodes, but its inference stage requires a scan through the whole graph to select existing nodes
as neighbor candidates for the cold-start nodes. Cold Brew thus has even more node dependency
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 22

22 Zhang, Shichang, et al.
and runs slower than a standard message-passing GNN. More related works include Graph-Free
Knowledge Distillation (GFKD) [27] transfers knowledge from a teacher GNN to a student GNN
via generating fake graphs. GFKD focused on graph-level prediction tasks, for which data instances
are independent graphs and the scalability problem caused by node dependence is less urgent.
4.4 Combine Together
Inference acceleration techniques in this section have been combined together to achieve better
results. For example, quantization is combined with distillation in [ 3]. In [ 3], GNNs weights and
activations are excessively quantized to binary. This quantization is very ambitious and hard to
achieve in one-step. [ 3] uses a cascaded distillation scheme to gradually quantize more and more
parts of the GNN and distill knowledge step-by-step to the next more quantized GNN down the
sequence. The final quantized-and-distilled binary GNN can achieve 2X speedup on Raspberry Pi
4B with only moderate accuracy loss.
5 GNN Acceleration: COTS Systems
In addition to the efficient algorithms for GNN training/inference, optimizing the underlying
system is essential for improving end-to-end throughputs of GNNs. In particular, existing works
accelerate GNN systems from three perspectives: GPU kernel acceleration, user-defined function
(UDF) optimization, and scalable training systems. Different from most deep learning (DL) tasks,
GNN computation follows a message-passing paradigm where it is highly desirable to have efficient
sparse operations; however, this cannot be sufficiently handled by existing DL platforms and
infrastructure. Various efficient GNN kernels and UDF optimization techniques have been proposed
for bridging the gap between the lack of fundamental support and high demand. Furthermore, due
to the irregular dependency among nodes, training a graph with a huge number of nodes and a
giant feature matrix is particularly challenging. How to design an efficient scalable GNN training
system is still an open research question for the ML system community. Currently, two widely used
frameworks for GNNs are PyG [34] and DGL [131].
5.1 GPU Kernel Acceleration
GPUs are powerful hardware accelerators that can greatly speed up computations required in DNN
training and inference. Under this context, GPU kernels refer to specialized programs optimized to
be executed in parallel on the many processing units of a GPU, enabling DNN computations to
be much faster and more efficient. Although GPUs are popular in being used for DL acceleration,
accelerating GNNs with GPUs is still challenging due to the unique sparsity and irregularity in
graphs. In recent years, many works propose novel GPU kernels for accelerating GNN workloads.
Table 3 summarizes recent works targeting different GNN kernels as detailed below.
PCGCN [123] improves the data locality in GNN computation by leveraging the unique sparsity
pattern in graphs that nodes are usually clustered. As such, processing GNN workloads in terms of
subgraphs substantially enhances the data locality. Furthermore, PCGCN employs a dual-mode
computation so that SpMM (sparse matrix-matrix multiplication) is used for sparse graphs while
GeMM (general matrix-matrix multiplication) is leveraged for dense graphs. For ensuring coalesced
memory accesses of features, SpMM processes all edges linking each pair of partitions at each stage.
The proposed technique has been implemented on top of TensorFlow. However, PCGCN relies on
graph clustering, which may incur non-trivial preprocessing overhead needed for graph partition.
fuseGNN [20] develops two workload abstractions of GNN aggregation, assuming that the input
graph adopts the COO format: Gather-ApplyEdge-Scatter (GAS) and Gather-ApplyEdge-Reduce
(GAR). Specifically, GAS works with the COO format graph, and processes edge-wise aggregation;
GAR handles node-wise aggregation, which is particularly useful for the CSR format as we defined
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 23

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 23
Table 3. Summary of GPU Kernel Acceleration Methods.
Work Methods Benefits
PCGCN [123]Workload Reorder,
Dual-Mode ComputationBetter Data Locality,
Coalesced Memory Access
fuseGNN [20]Kernel Fusion,
Workload Reorder,
Dual-Mode ComputationDecreased Memory Traffic,
Saved Memory
FusedMM [99] Kernel FusionBalanced Workload,
Saved Memory
TLPGNN [38]Workload Reorder,
Kernel Fusion,
Dynamic Workload AssignmentDecreased Memory Traffic,
Coalesced Memory Access,
Balanced Workload,
No Branch Divergence
Zhang et al. [164]Workload Reorder,
Kernel Fusion,
Recomputing EmbeddingReduced Computation,
Reduced I/O,
Saved Memory
in Section 2.1. Although GAR is usually efficient because it relies on on-chip reduction with a shared
output memory, creating CSR from COO is usually time costly, thus degrading the achievable
efficiency of GNN computation. As a result, fuseGNN uses GAS for graphs with a low average
degree (e.g., Cora[ 108]) and GAR for graphs with a high average degree (e.g., Reddit [ 45]). All
operations in the three kernels, including graph processing for graph format conversion, GAS, and
GAR, are fused for alleviating the memory traffic. fuseGNN is developed for PyTorch and can be
extended to multi-GPU scenarios.
FusedMM [99] is a CPU kernel but the authors claim that it can be extended to GPU acceleration.
The work divides popular graph operators into five types of basic operations that can be launched
sequentially (e.g., element-wise additive, reduction operation, element-wise self-operation) and
defined by users for designing customized GNN layers. During GNN computations, the operations
associated with each target node are fused together for achieving memory reduction by eliminating
the necessity of saving intermediate results. Since each CPU thread is assigned multiple target
nodes, FusedMM further adopts an edge-balanced partition to ensure a balanced workload among
threads. The proposed kernel of FusedMM can be integrated into DGL [131].
TLPGNN [38] develops two levels of parallelism: node- and feature-level parallelism. In the
first level (i.e., warp-level), TLPGNN assigns each target node to a warp of threads for avoiding
branch divergence caused by uneven node degrees. For computing each node’s new embedding, the
second level (i.e., thread-level) adopts feature-level parallelism that the threads in the same warp
process feature in parallel while processing each edge sequentially, which allows coalesced memory
accesses to input features for each source node. To achieve a balanced workload, a dynamic workload
assignment is developed, which allocates the next computation task once one hardware resource is
released. TLPGNN further performs kernel fusion to avoid unnecessary memory accesses.
Zhang et al. [164] first decompose a GNN operation into four parts (i.e., scatter across edges,
apply edge, gather from edges, and apply node), and then identify three challenges in GNN com-
putations: redundant neural operator computation, inconsistent thread mapping, and excessive
intermediate data. Specifically, redundant neural operators can be caused when the same feature
transformation is employed to the same node features after they are scattered across edges. To
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 24

24 Zhang, Shichang, et al.
address this issue, the scatter operation can be postponed after the feature transformation is finished.
For the second issue, if the thread mappings are inconsistent (e.g., scatter threads are assigned with
edge-centric tasks, while gather threads are assigned with node-centric tasks), these threads have
to be processed sequentially and cannot be merged. This work leverages unified thread mapping
for fusion. The third issue identifies that intermediate results may occupy up to 92% of the total
memory requirement, which can be resolved by recomputation during backward propagation.
Overall, workload reordering is necessary for identifying GPU-friendly processing patterns. In
particular, most works sequentially compute output nodes and process features in parallel. The
former allows kernel fusion and avoids branch divergence while the latter guarantees coalesced
memory accesses. Because the graph structure can be highly irregular and can be presented in
different formats, some work leverage dual-mode computation and dynamic workload assignment
for handling graphs of different properties.
5.2 User-defined Function Optimization
The flexibility in selecting the message, aggregation, and update functions has been a critical
component in the success of GNNs under the message-passing paradigm. GNN systems possess
the ability for users to natively express these functions using tensor operators in Python and
execute them through APIs provided by the system. This programming paradigm is termed user-
defined functions (UDFs). Once defined, these functions can be reused throughout the program,
increasing code modularity and reducing code duplication. Overall, UDFs provide a powerful tool
for creating more efficient and maintainable codes. A few recent works have been developed to
ease the deployment of user-defined GNNs.
FeatGraph [52] provides a flexible programming interface on top of the compiler stack for
DL systems: TVM [ 15] for supporting friendly user-defined functions. The users can manually
select the tiling factor (the number of features being processed in each thread) for maximizing the
cache utilization. However, because TVM lacks the support for sparse operations, FeatGraph only
supports SpMM/SDDMM-based operations.
Seastar [137] develops a vertex-centric programming interface for user-defined functions, and
generates an execution plan that optimizes memory consumption and data locality, resulting
in higher performance. Specifically, Seastar first translates the vertex-centric logic into tensor
operations via a tracer, and then identifies operator fusion opportunities by the notion of Seastar
computation pattern. This approach reduces memory consumption and enhances data locality via
fusing operations of each target node by leveraging the merits of vertex-centric programming,
similar to FusedMM [ 99]. Additionally, Seastar uses a distributed graph communication library
(DGCL) to enable scalable GNN training on distributed GPUs, and its DL backend is MindSpore, an
all-scenario AI computing framework [59].
Graphiler [140] develops an automated tool for scatter/gather-related tasks. Specifically, com-
pared with [ 164] mentioned in the previous section that manually implements operation reordering
for some specific GNN operations, Graphiler enjoys better flexibility by automatically detecting re-
dundant communication and then adjusting their order accordingly, and further fuses all operations
after gathering from edges to reduce intermediate memory accesses.
Existing automation tools for GNN computation are not prosperous. This is because concurrent
DL backends do not well support sparse operations. We believe this direction is getting mature as
several recent backends for sparse DL operations have been proposed. For example, SparseTIR [ 150]
is a general compiler infrastructure for sparse DL operations. It develops a set of intermediate
representations that can represent operators supported by the aforementioned Seastar, FeatGraph,
and Graphiler work.
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 25

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 25
5.3 Scalable Training Systems
To support the training of GNNs with large-scale graphs, many recent works have developed GNN
systems targeting scalable training. From the system perspective, there are two types of scalability:
•Vertical scalability handles scaling data with restricted computation resources. As the
training graph of a GNN can be giant, this direction is highly practical. For the GNN systems
pursuing vertical scalability, they usually assume the use of low-cost or unlimited host
memory.
•Horizontal scalability expects better efficiency with scaling resources . However, horizontal
scalability is particularly challenging in GNN training because more workers incur more
dependency across devices, leading to more significant communication and synchronization
overhead and thus degrading the achievable training throughput.
Some concurrent works summarize recent distributed GNN training systems [ 77,112]. In par-
ticular, Shao et al . [112] classifies recent works in terms of hardware facilities (e.g., multi-GPU,
GPU cluster, and CPU cluster). On the other hand, Lin et al . [77] categorizes all works as full-batch
training or mini-batch training. In this work, we focus on scalable GNN training, which is a wider
topic than distributed GNN training, as it not only includes efficient distributed training (i.e., hori-
zontal scalability) but also considers large-scale training with limited computing resources (i.e.,
vertical scalability). Furthermore, we observe that the key contribution of each system overlapped
among (between) the categories adopted in existing survey papers, hindering the understanding of
their technical differences. To facilitate a clearer comparison of the technical contributions made by
different scalable GNN training systems, we categorize them based on the method by which data is
loaded onto the computation devices: on-device systems, swap-based systems, and sampling-based
systems. An overview of these categories is demonstrated in Figure 4, and we compare their scope
in Table 4.
Table 4. Comparison of The Three Types of Scalable GNN Training Systems.
System Type Main Focus Challenges and Contribution
On-device system Full-graph aggregation Communication and memory pattern
Swap-based system Full-graph aggregation Workload scheduler
Sampling-based system Sampling-based aggregation Data movement and caching strategy
Here we briefly summarize these three types of systems:
•On-device Systems (Figure 4(a)) distributedly store the entire graph and associated features
on the computation devices (e.g., GPUs or CPUs). Since each worker only maintains a portion
of data, systems based on this storage architecture mostly focus on full-graph aggregation
for better leveraging the computation resources. During training, each worker needs to
transfer the intermediate embeddings for supporting the full-graph aggregation in the next
layer. A naive system that follows this direction suffers from frequent and overwhelming
communications as well as exploded memory requirements for caching the embeddings from
other workers. Therefore, existing systems of this category mainly focus on maximizing
communication and memory savings to enhance horizontal scalability.
•Swap-based Systems (Figure 4(b)) store all data in (distributed) shared memory and swap
it and the intermediate results to workers for computations. This approach enables each
node to fully utilize its neighbors’ embeddings. However, developing a workload scheduler
that ensures vertical scalability poses two challenges for swap-based systems: 1) balancing
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 26

26 Zhang, Shichang, et al.
(Distributed) Shared Memory
Worker Worker Worker(Distributed) Shared Memory
Subgraph Scheduler Sampler Sampler
Worker WorkerWorker Worker
D
C B
A EA DA
BC
DE
FA
BC
DE
F
A
C D
F EA FA C
D FA
BC
D FEA
BC
DE
F
(a) On-device Systems (b) Swap-based Systems (c) Sampling-based Systems
Fig. 4. Overview of three types of scalable GNN training systems. The dashed lines in (a) indicate the
dependency between nodes from different workers, while the grey nodes in (b) serve only as input nodes for
subgraph computations. Model synchronization among workers is not shown for better visual clarity.
workload among different GPUs is non-trivial, and 2) naive full-graph aggregation with
limited devices incurs long iteration latency because it only updates the model once per
training epoch. Despite these challenges, effective solutions to this problem are crucial for
improving the achievable efficiency of these systems and enhancing their overall performance.
•Sampling-based Systems (Figure 4(c)) also save the whole graph and features in (dis-
tributed) shared memory; however, instead of performing full-neighbor aggregation, they
adopt neighbor sampling for neighbor aggregation. To implement this approach, each sampler
first determines sampled input nodes based on the output nodes for mini-batch training. The
features of sampled input nodes are then fetched from the (distributed) shared memory. The
primary challenge for sampling-based systems is data movement for mini-batch preparation,
which can significantly limit performance. To overcome this bottleneck, these systems focus
on developing better data movement and caching strategies to optimize communication.
Overall, sampling-based systems enhance both horizontal and vertical scalability, making
them a promising solution for GNN training on large-scale graphs.
5.3.1 On-device Systems One convincing approach for GNN training with large-scale graphs is
to partition the data into multiple workers. All workers update the local model using the locally
labeled nodes and then synchronize the local models. This design is very similar to distributed data
parallelism in DNN training. However, there are two fundamental differences: 1) GNN training
requires feature transfer for each layer, which can explode the required communication volume;
and 2) the node feature/embedding tensors are substantially larger than the model. These unique
communication and memory challenges motivate the development of various GNN training systems.
Since systems under this paradigm update models with dependent partitions instead of independent
data, this approach is also named as partition parallelism .
CAGNET [124] develops four communication patterns by partitioning the adjacency matrix of
target GNNs into 1D, 1.5D, 2D and 3D formats and then distributing the workload across workers
accordingly. This work further analyzes the communication latency for each design. For ensuring a
balanced workload, the random partition is adopted. However, because communication does not
consider the sparse pattern, broadcasting all features causes redundant data transfer (i.e., workers
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 27

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 27
Table 5. Summary of Scalable Training Systems.
System
TypeSystemComputation
HardwareAggregationAdjust
AlgorithmOpen
Source
On-device
SystemCAGNET [124] GPU Full-graph ✗ ✓
Dorylus [122] Hybrid Full-graph ✗ ✓
𝑃3[39] GPU Sampling ✓ ✗
PipeGCN [129] GPU Full-graph ✓ ✓
LLCG [100] GPU Full-graph ✓ ✓
BNS-GCN [128] GPU Hybrid ✓ ✓
SAR [94] CPU Full-graph ✗ ✓
Sancus [96] GPU Full-graph ✓ ✓
Swap-based
SystemNeuGraph [87] GPU Full-graph ✗ ✗
Roc [62] GPU Full-graph ✗ ✓
GNNAdvisor [133] GPU Full-graph ✗ ✓
GNNAutoScale [35] GPU Full-graph ✓ ✓
GraphFM [155] GPU Full-graph ✓ ✓
Sampling-based
SystemAliGraph [177] CPU Sampling ✓ ✓
PaGraph [79] GPU Sampling ✗ ✓
DistDGL [171] CPU Sampling ✗ ✓
DistDGLv2 [172] GPU Sampling ✗ ✗
SALIENT [68] GPU Sampling ✗ ✓
GNNLab [146] GPU Sampling ✗ ✓
BGL [80] GPU Sampling ✗ ✗
GNS [29] GPU Sampling ✓ ✗
MariusGNN [127] GPU Sampling ✓ ✓
receive node features that are not required), leading to both significant communication overhead
and memory consumption.
Dorylus [122] is a serverless platform for saving the cost of training GNN models, enabling
affordable, scalable, and accurate GNN training. Because conventional GPU servers are expensive
and have restricted memory, Dorylus presents a distributed system with much lower-cost CPU
clusters and Lambda threads, a representative of serverless threads. The authors notice that the
training time of GNNs is dominated by neighbor propagation, rather than neural network oper-
ations. As parallel processing does not effectively enhance the speed of neighbor propagation,
Dorylus executes 1) the neighbor propagation in CPU clusters for achieving the best performance
per dollar, and 2) the neural network operations in Lambda threads for avoiding the necessity
of unneeded resources (e.g., storage) that conventional distributed platforms (e.g., CPU clusters)
require. To further minimize the network latency between the aforementioned CPU clusters and
Lambda threads, Dorylus adopts fine-grained computation and asynchronous execution through
pipelined computation and communication including embedding transfer and model synchroniza-
tion. Following VR-GCN (see Section 3.2), the convergence analysis based on historical embeddings
is provided. Overall, Dorylus achieves the best performance per dollar against its baseline methods.
P3[39] is a distributed system with sampling-based aggregation focusing on the scenario where
a mini-batch can not fit into the memory in a single GPU. Since the main contribution of 𝑃3is a
novel communication pattern among workers, we place it in the category of on-device systems.
Specifically, 𝑃3assumes that a GNN model’s hidden embedding dimension is significantly smaller
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 28

28 Zhang, Shichang, et al.
than that of the input features. Based on this assumption, 𝑃3leverages inter-layer model parallelism
for the first layer (i.e., distributedly computing 𝑿𝑾(1)by splitting 𝑿along the column dimension)
to distribute the burdensome storage of input features and then uses data parallelism (i.e., dis-
tributively storing 𝑨for computing node embeddings) for reducing the required communications
for transferring intermediate embeddings. Additionally, 𝑃3further pipelines communication and
computation across mini-batches for hiding the communication overhead. The limitation of this
design is that 𝑃3degrades the performance when the hidden dimension is increased.
LLCG [100] is the acronym of “learn locally correct globally” which reduces the communication
frequency of model synchronization through periodic data transfer. In particular, each training
round of LLCG consists of two phases. In the first phase, each worker constructs local mini-batches
and updates local parameters following conventional sampling-based training without data transfer.
For the second phase, the model parameters in all workers are averaged in a server that keeps
on updating the model through mini-batch training with full-graph aggregation. As a variant of
partition parallelism, LLCG does not transfer node embedding but synchronizes model parameters
instead. The theoretical analysis shows that the stochastic gradient computed by LLCG is bounded
by a constant that tends to zero when increasing the size of the sampling size. Since the achievable
accuracy of LLCG mainly benefits from the full-graph aggregation on the server side, we regard it
as a type of full-graph aggregation method.
PipeGCN [129] optimizes the straightforward implementation of partition parallelism through
pipelining embedding computation and the dependent communication across training iterations. In
particular, for distributed full-batch training, each worker needs to wait for dependent neighbors’
embedding (or embedding gradient) to be transferred before performing local forward (or backward)
computation. In PipeGCN, each worker no longer needs to wait for the dependent neighbors.
Instead, they use the transferred embeddings/embedding gradients in the previous iteration. This
design breaks this computation-communication dependency and allows the training system to
perform computation and communication in parallel. Although the model update does not follow
traditional gradient descent because it introduces stale embeddings and stale embedding gradients,
the convergence analysis shows that PipeGCN enjoys the convergence rate of O(𝑇−2
3), with𝑇
being the number of training iterations, which is better than popular sampling-based methods
(O(𝑇−1
2)). PipeGCN further employs embedding/embedding gradient momentum for mitigating
the error caused by the stale data transfer.
BNS-GCN [128] accelerates partition parallelism from a different angle. The authors observe
three major issues associated with partition parallelism: significant communication overhead,
scaled-out memory requirement, and imbalanced memory distribution. They further observe that
both communication volume and memory consumption are correlated to the number of boundary
nodes (i.e., the dependent neighbors that need to be received for each partition). To mitigate the
issues caused by boundary nodes, BNS-GCN simply adopts Boundary Node Sampling, achieving
drastically saved communication, reduced memory, and balanced computation, while maintaining
full-graph accuracy. The sampling rate can be as low as 0.1 as suggested by the authors.
SAR [94] also discovers the exploded memory issue in partition parallelism and proposes
Sequential Aggregation and Rematerialization scheme. Specifically, each worker receives dependent
neighbors through point-to-point communication sequentially. Once the dependent data from one
remote worker is received, the receiver worker consumes it, deletes the data to release the memory,
and receives the node embeddings from the next worker. For supporting backward propagation,
SAR uses reversible neural networks to efficiently reconstruct (rematerialize) the deleted data.
Sancus [96] improves CAGNET by performing expensive broadcast communication selectively.
To avoid overwhelming broadcast communication, which may occupy over 80% of the total training
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 29

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 29
time, each worker maintains a copy of other workers’ embeddings and only refreshes them when
the embeddings are too stale. Based on this design, Sancus is able to skip considerable rounds of
communication, thus achieving better throughput. Three staleness metrics are proposed for the
staleness check.
5.3.2 Swap-based Systems. Because full-graph aggregation with on-device storage has expensive
storage requirements for scaling graphs, another direction of full-graph aggregation is to store the
entire graph in the external shared memory, which motivates swap-based systems.
NeuGraph [87] and Roc [62] are pioneering works for GNN training with scaling graphs. Given
restricted GPU resources, they process output nodes sequentially and store all intermediate results
in the host CPU DRAM for the computation of backward propagation. Both works support GNN
training with multiple GPUs. NeuGraph only supports distributed training in a single machine, and
adopts equal-vertex partition for heuristically balancing workload. Roc strengthens the scalability
by supporting multi-machine training. It further leverages a learnable cost model for partitioning a
graph toward better workload balance.
GNNAdvisor [133] is another work that targets scalable GNN training with limited devices.
Compared with NeuGraph and Roc, GNNAdvisor further explores feature-level partition. It imple-
ments a low-overhead cost model to determine the best graph-level and feature-level partition. The
current implementation of GNNAdvisor only supports single-device training.
GNNAutoScale [35] and GraphFM [155] develop scalable algorithms for efficient GNN systems.
Specifically, to enable the training within limited GPU memory, GNNAutoScale leverages Clus-
terGCN (described in Section 3.2). During the training process, all nodes maintain their features
and the latest intermediate embedding in the host CPU memory. For better approximating neighbor
aggregation during the training, GNNAutoScale accesses all maintained historical embeddings for
unsampled neighbors. With this design, all nodes can access all neighbor embeddings although the
embeddings of unsampled nodes are stale. GraphFM is a follow-up work of GNNAutoScale that
develops GraphFM-IB and improves the approximated neighbor aggregation by integrating feature
momentum for alleviating incurred staleness. Their convergence analysis shows that GraphFM-IB
does not rely on large sampling size. To further mitigate staleness, a variant named GraphFM-OM
is developed by pushing the intermediate embeddings of sampled nodes to update the unsampled
nodes, achieving better accuracies.
5.3.3 Sampling-based Systems The last paradigm uses neighbor sampling for saving both training
computation and memory requirement. For reducing data movement, various sampling and caching
techniques are proposed.
AliGraph (also known as Graph-Learn) [ 177] is a CPU-based distributed training framework
for a wide range of GNN variants and training algorithms. For supporting the efficient training
of a large-scale graph, AliGraph utilizes distributed storage with cached important nodes and an
optimized data sampler for fetching dependent node features. Specifically, the system implements
four graph partition algorithms for different tasks (e.g., METIS is used for separating sparse graphs,
streaming-style partition works on dynamic graphs with frequent edge change). Since distributed
storage increases data access overhead, AliGraph caches frequently visited nodes for avoiding
repetitive data transfer. The system further implements three sampling strategies for speeding up
GNN training and accelerates the samplers with dynamic sampling weights. AliGraph has been
deployed at Alibaba for business.
PaGraph [79] is a multi-GPU system for GNN training. For reducing the data transfer between
CPU and GPU for neighbor sampling, PaGraph heuristically minimizes data movement by caching
nodes with large out-degrees in GPUs. To enable multi-GPU training, PaGraph devices a graph
partition algorithm to balance training nodes across partitions, achieving approximately balanced
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 30

30 Zhang, Shichang, et al.
workload. Each partition only maintains nodes within the partition and 𝐿-hop neighbors for an
𝐿-layer model for avoiding redundant storage.
DistDGL [171] and DistDGLv2 [172] are distributed training frameworks based on DGL [ 131].
DistDGL is developed for CPU clusters where each worker maintains a partitioned subgraph
in the host memory with its node features and one-hop neighbors. During the training process,
each worker first launches processes for mini-batch sampling which create training batches while
fetching features of sampled remote neighbors. METIS is used for balancing training nodes across
partitions and reducing edge cuts so that the communication across workers is minimized. To
further approximately balance workload, multi-constraint partition is adopted for balancing nodes
and edges. For the graphs with learnable node embedding, DistDGL adopts asynchronous update
of sparse embedding for overlapping communication and computation while enjoying seldom
data access conflicts. DistDGLv2 extends DistDGL to heterogeneous graphs and GPU clusters
and deploys asynchronous mini-batch generation for fully utilizing all computation resources.
Overall, DistDGLv2 is 2 ×-3×faster than DistDGL. Both projects have be integrated with DGL and
DistDGLv2’s APIs are compatible with DGL.
SALIENT [68] boosts the training of GNNs with an efficient dataloader. This work identifies
that for the conventional implementation of GNN training where neighbor sampling is conducted
in CPU and the GNN model is computed in GPUs, batch preparation and data transfer takes around
72% total epoch time, which severely degrades the efficiency of GNN training. In particular, batch
preparation contains two stages: 1) sampling and creating message-flow graphs; and 2) fetching
feature and labels through tensor slicing. The first stage has a large optimization space because
neighbor sampling can be accomplished by multiple design choices and implementation (e.g.,
mapping node indices between the original graph and the sampled message-flow graph, fusing
the operations of neighbor sampling and minibatch construction). SALIENT empirically explores
a fast sampler that is microbenchmarked in ogbn-products [ 51] for each individual hop, which
yields 2.5x faster speed than PyG. For accelerating the second stage, SALIENT implements C++
threads for parallelly slicing tensors in shared memory. To further mitigate the overhead of data
transfer, SALIENT pipelines data transfer and GPU computation. SALIENT also evaluates neighbor
sampling for efficient GNN inference.
GNNLab [146] improves PaGraph with sampling minibatches in GPUs. This improvement is not
trivial because on-device sampling requires extra memory which restricts cache size thus increases
missing rates. In addition, simply caching nodes with large out-degree is not optimal because it
ignores the pattern of training nodes. Therefore, some highly-connected nodes are not frequently
sampled during the training process. To address this issue, GNNLab first performs serveral rounds
of pre-sampling. Most frequently visited nodes are selected as cached nodes. In addition to the
pre-sampling based caching policy, GNNLab further enhances cache size through space-sharing
design. Some GPUs are selected as samplers and the others are trainers. This strategy enhances
data locality for all workers. To balance the execution time of samplers and trainers, GNNLab first
determines the number of samplers and trainers based on their execution time in one epoch so
that samplers can finishes sampling tasks in shorter time, and then, idle samplers are switched to
trainers for fully leverage computation resources.
BGL [80] also observes data movement bottlenecks in sampling-based GNN training systems
and develops a dynamic GPU cache by leveraging temporal locality among mini-batches, which
can be strengthened by selecting training nodes via BFS sequences. However, a straightforward
implementation incurs biased label distribution. To mitigate this issue, BGL generates multiple
BFS sequences by randomly selecting BFS roots. Furthermore, for each training epoch, the BFS
sequences will be shifted to add randomness for batch generation. Under the multi-GPU setting,
BGL ensures no duplicated cached nodes because cached nodes can be shared via NVLink and this
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 31

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 31
design saves GPU memory. BGL further develops a scalable graph partition algorithm with BFS
while heuristically ensuring a balanced workload. Finally, BGL optimizes resource allocation of
data preprocessing by formulating it as an optimization problem.
GNS [29] stands for Global Neighbor Sampling, which develops an efficient and scalable sampling
algorithm for reducing the data movement between CPU and GPU. To achieve this, GNS periodically
select a subset of all nodes and caching their node features in GPUs. During the neighbor sampling
process, if the GPU caches contain sufficient neighbors, GNS only samples neighbors from the
caches. Otherwise, extra neighbors outside GPUs are sampled. GNS devices two caching strategies:
1) the nodes are cached with the probability proportional to their degrees so that only a small
cache needs to be maintained for a power-law graph; 2) the sampling probability for each node is
determined by random walks where more frequently visited nodes are more likely to be sampled.
MariusGNN [127] is a GNN training system targeting deeper vertical scalability, which is
developed based on Marius [ 93]. The above-mentioned works assume that all data can be saved
in the host CPU memory. This work considers a more resource-constraint scenario where CPU
memory is limited. For reducing expensive I/O operations, MariusGNN develops a sophisticated
dataloader, which enjoys reduced I/O operations and continuous external memory access.
DSP [11] accelerates GNNs by addressing two main challenges associated with multi-GPU
training: low GPU utilization and high communication costs. To address the first challenge, DSP
uses a tailored data layout that stores the graph topology and popular node features in GPU memory
and leverages NVLink to share GPU caches among all trainers, which allows for efficient graph
sampling with multiple GPUs. Additionally, DSP introduces a collective sampling primitive (CSP)
that pushes the sampling tasks to data to reduce communication costs. To address the second
challenge, DSP employs a producer-consumer-based pipeline that allows tasks from different
mini-batches to run congruently, improving GPU utilization.
Overall, the core challenge towards scalable GNN training systems is data transfer. To address
this issue, several methods have been widely adopted, such as caching frequently visited nodes
to avoid redundant communication, using asynchronous communication to hide communication
overhead, and leveraging stale embeddings to reduce communication volume. However, in addition
to reducing data transfer, saving memory and balancing workload are also critical factors to consider
when designing scalable GNN training systems. Achieving these goals requires a combination
of effective algorithms and system optimizations to ensure efficient and effective training on
large-scale graphs.
6 GNN Acceleration: Customized Hardware
The increasing interest in GNNs has led to the development of customized accelerators (FPGA or
ASIC) for faster processing. While GNNs share similarities with CNNs in network architecture,
the differences in computation complexity and communication patterns make numerous CNN
accelerators [ 19,115,167] unsuitable for GNNs. Specifically, GNNs require Matrix Multiplication
(MM) units and have irregular memory access due to the unstructured nature of graphs. On the
other hand, although both the Aggregation and Update stages (the main stages of a typical GNN)
can be modeled as MMs, their computation and communication pattern differ. The Aggregation
stage handles an ultra-sparse adjacency matrix, while the matrices in the Update unit are dense
or have a much lower rate of sparsity. As a result, the use of only dense MM units [ 130] or the
sparse MM (SpMM) units [ 117,120] is inefficient for this application. Furthermore, GNNs work with
vectors assigned to each node, unlike traditional graph algorithms that use scalar values, resulting
in distinct computation and communication requirements. These distinctions have motivated
researchers to design specialized hardware modules to effectively process GNNs.
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 32

32 Zhang, Shichang, et al.
Abadal et al. [ 1] concisely review the endeavors undertaken in this domain, offering insightful and
informative perspectives. We try to accentuate the challenges and the differences in the employed
optimization techniques more clearly. Section 6.1 describes the challenges that the designer would
face and the decisions one must make in developing the accelerator. Section 6.2 reviews the most
prominent accelerators proposed to date and discusses how each work approached the problem.
6.1 Challenges in Customized Hardware Design
Graphs and GNNs possess specific characteristics that necessitate special attention in the design of
customized hardware. Specifically, designing customized accelerators for these structures requires
consideration of the following unique features:
Wide range of GNN layers. Various GNN models utilize distinct Aggregation and Update
methods, in addition to potentially incorporating graph/subgraph pooling layers and non-linear
activation functions. These differences may impact the execution bottlenecks and areas requiring
acceleration, resulting in a trade-off between flexibility and performance. This is because the
capability to support multiple modes of computation necessitates generality, which may prevent us
from getting the optimal performance for each specific case.
Computation/communication pattern and sparsity rate disparity. The computation and
communication requirements of different steps in a GNN may vary significantly due to the distinct
sparsity rates of the matrices involved and their irregular access pattern. For instance, the adjacency
matrix is an ultra-sparse matrix, whereas the weight matrix is dense. The node embeddings in
general are dense but if they are generated by activation functions such as ReLU, they may create a
sparse matrix, but the sparsity rate is much lower than that of the adjacency matrix. Additionally,
an end-to-end GNN application may utilize other computation patterns such as a Multi-Layer
Perceptron (MLP), which creates a dense multiplication unit to process the node/graph embeddings.
These differences in computation and communication patterns pose a significant challenge to GNN
hardware designers, as they must address the following sources of disparity and irregular access
patterns:
•Graph (adjacency matrix) sparsity: The graph is a large and ultra-sparse matrix and poses
challenges to the memory design. For instance, a graph such as Orkut [92] comprises over 3
million nodes and more than 223 million edges, yet the density is only 2.5×10−5. The large
graph size requires an on-chip and off-chip memory hierarchy for storage, while the sparsity
translates to irregular accesses of graph elements that span multiple memory hierarchies.
•Sparsity introduced by activation operators: The introduction of sparsity by activation
operators is a relevant aspect to consider for GNN hardware design. Specifically, activation
operators like ReLU can produce sparsity in the node embeddings or features. To enable
efficient computation on the node embeddings, it is desirable to address both graph and feature
sparsity. This can be achieved by implementing zero elimination or bypassing techniques in
the computation units. By doing so, we can accelerate the computation and avoid wasting
computation cycles on zero-valued operands.
•Coordination between modules: The differences in data layout caused by sparsity necessitate
the coordination of computation modules. Specifically, the Aggregation step in GNNs often
requires sparse matrix-vector multiplication (SpMV) or SPMM modules, which typically
shuffle the data layout to enable efficient processing. On the other hand, the Update (also
called transformation) step usually works on continuous matrix/vector layouts. Consequently,
an accelerator must coordinate the data layout between modules. As mentioned before, in
some GNNs, the weights or embeddings in the Update stage may also be sparse. Therefore, to
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 33

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 33
further reduce processing latency, the accelerator must further coordinate the computation
pattern and sparsity of the Update module with the SpMV and SpMM modules.
Dynamic input and network structure. Given the dynamic nature of graphs, the architectural
decisions for GNNs are highly dependent on the specific characteristics of the input graph and
the model’s hyperparameters. The memory and computation requirements of a GNN accelerator
are significantly impacted by factors such as the size of the graph, its level of sparsity, and the
dimensions of the vectors involved. Therefore, the features and properties of each input graph can
be highly distinct from one another, including the following:
•Different scales and structure of graphs: Real-world graphs exhibit a wide range of node
counts, ranging from a few nodes (e.g., 10) [ 102] to millions [ 92]. This variation in scale
results in differences in the computation resources needed to perform GNN computations.
Moreover, recent studies have demonstrated that graph size can significantly impact the
selection of scheduling and optimization techniques [ 40,114]. Additionally, the varying
number of neighbors of nodes in the graph introduces a workload imbalance in processing
them.
•Dealing with vectors with various sizes per node: Traditional graph algorithms like BFS,
SSSP, and PageRank typically assign a single scalar value to each node. In contrast, GNNs
utilize long feature vectors for nodes. This difference in representation affects the memory
access pattern and parallelism. The use of long vectors allows for intra-node parallelism and
data reuse opportunities when applying the same weight matrix to all node embeddings. A
GNN is composed of multiple layers that can operate on vectors of different sizes. This leads
to changes in the access pattern and the required degree of parallelism across different layers.
These differences create additional opportunities for customization in the hardware design
of GNN accelerators.
Given the special features of graphs and GNNs, designers must carefully consider how to address
the aforementioned critical factors when designing customized hardware accelerators. These include
selecting appropriate data structures and hardware architectures that can handle the large and
irregular data structures involved in graph processing, optimizing parallelism and exploiting the
available sparsity to achieve high-performance computing, balancing the workload across the
processing units, minimizing communication overheads between processing units, and adjusting
the flow of data to avoid potential bottlenecks. Additionally, designers must also consider how
to optimize the accelerator for the specific GNN model and its corresponding computation and
communication patterns. Furthermore, given the wide range of GNN models, each with its unique
characteristics and performance requirements, the designer must consider the trade-off between
generality (flexibility), scalability, and specialization to achieve optimal performance for a given
application.
6.2 Summary of The Existing Customized Accelerators
The accelerators presented in this section employ distinct approaches to address the aforementioned
challenges, influenced in part by the neural networks and datasets they target. Such a trade-off can
affect the attainable peak performance and the generalizability of the proposed accelerators. Natu-
rally, as the application scope of the work narrows, more opportunities arise for customization of the
accelerator, thereby improving its performance, albeit at the expense of flexibility to adapt to new
experimental settings. These accelerators are primarily intended for deployment, focusing on infer-
ence rather than training. The constantly changing network architecture during training presents
challenges for adapting the hardware design or requiring regeneration and/or reconfiguration.
Consequently, customized accelerators designed for the training stage are rare [17, 156].
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 34

34 Zhang, Shichang, et al.
The summarized works are presented in Table 6, which compares them based on five main
features: graph size, target GNNs, support for layer customization based on the dynamic structure
of the input, the approach to handle the inherent sparsity of the target application, and levels of
parallelization employed. The prevalent parallelization schemes used in GNN accelerators include
parallelizing the feature dimension for all stages, parallelizing nodes in the Update stage, and
pipelining the main stages of each layer for intra-layer parallelization. Some approaches also
involve exploiting parallel processing of multiple edges in the Aggregation stage, pipelining the
execution of multiple layers (inter-layer), or processing multiple graphs in parallel (batch). In the
remainder of this section, the related works are categorized based on the flexibility of their workload.
Section 6.3 provides a summary of the notable works that propose accelerators to target multiple
GNN algorithms. Conversely, another category of works concentrates mainly on the operations of
GCNs, which are one of the most widely used GNN algorithms. These works further customize the
microarchitecture for GCNs, and their details are presented in Section 6.4.
6.3 Accelerators for General Workloads
As previously stated, a fundamental decision when designing a customized accelerator is to de-
termine the range of applications it aims to support. In this section, we provide an overview of
the works that propose a flexible accelerator capable of targeting multiple GNN algorithms. These
works either create a unified architecture that handles all computation stages, as discussed in
Section 6.3.1, or develop a specialized engine for the primary computation stages due to the distinct
computation and communication patterns, as summarized in Section 6.3.2.
6.3.1 All-Stage Unified Architecture. The works in this category propose a unified architecture
to process multiple stages of computation, despite their division into distinct phases, such as the
Aggregation and Update steps. The proposed accelerators in this category develop a singular
engine capable of handling these stages, even though they have different requirements in terms of
computation and communication. We elaborate on their approaches below:
EnGN [76] targets graph convolution network (GCN) [ 71], relational GCN (R-GCN) [ 107], gated
GCN [ 25], GraphSage [ 45], and graph recurrent network [ 104]. To process them, the operations
are divided into three stages: feature extraction, aggregation, and update. The authors develop a
Neural Graph Processing Unit (NGPU) as a unified architecture for executing them. These stages are
processed as pipelined matrix multiplications of the feature matrix, neural weights, and adjacency
matrix. The NGPU uses a 32-bit fixed-point 128×16systolic array as the main computation
unit, which maps nodes to different rows and their features to different columns. To perform the
aggregation, the authors propose a ring-edge-reduce dataflow that connects processing elements
(PEs) in the same column with a ring so that the results can be passed through and added together
based on the adjacency matrix. As the adjacency matrix is sparse, the authors reorganize the edges
to reduce the number of idle PEs during the execution of aggregation. Additionally, high-degree
nodes are cached to decrease the number of off-chip transactions since they are reused across
multiple operations. EnGN analyzes the effect of different schedules on the number of operations
and develops a dimension-aware stage reordering strategy, which changes the order of execution
between the aggregation and feature extraction stage based on the input and output dimensions,
to reduce the number of operations. Furthermore, it splits the graphs into 2D tiles to fit them
into on-chip resources and develops an analytical formula to decide whether for a given layer, a
column-order traversal of the tiles results in fewer I/O operations or their row-order traversal.
Rubik [17] develops an accelerator tailored to train GIN and GraphSage. To enhance graph
locality and data reuse, the proposed design leverages a pre-processor that reorders the graph
to group nodes with common neighbors. The graph subsequently passes through a hierarchical
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 35

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 35
Table 6. Customized Accelerators Properties
WorkGraph
SizeTarget
GNNsLayer
CustomizationSparsity
SupportParallelization
EnGN
[76]Large, GCN, GRN Reorganize edges Intra-layer, Node-level
Ultra Gated GCN ✗ for aggregation Feature-level, Edge-level
Large R-GCN
GraphSage
Rubik
[17]Small, GIN ✗ Pre-process to Intra-layer, Node-level
Large GraphSage reorder the graph Feature-level, Edge-level
HyGCN
[144]GCN Window sliding and Intra-layer, Node-level
Large GraphSage ✗ shrinkage + sampler as Feature-level, Edge-level
GIN a sparsity eliminator (if feature dimension is
DiffPool tool for aggregation small)
FlowGNN
[106]Small GCN, GIN On-the-fly multicasting Intra-layer, Node-level
Large GAT, PNA ✗ for data distribution Feature-level, Edge-level
DGN, VN
BlockGNN
[176]Large GS-Pool Normal SIMD process Intra-layer, Node-level
GCN, GAT ✗ Feature-level, Edge-level
GGCN
DeepBurning
-GL [75]Large GCN Degree-aware caching Intra-layer, Node-level
GS-Pool ✓ Feature-level, Edge-level
EdgeConv
AWB-GCN
[40]Load-balancing
for both stages: Inter-layer, Intra-layer
Large GCN ✓ distribution smoothing , Node-level, Edge-level
remote switching ,
evil row remapping
StreamGCN
[114]Pre-process for Inter-layer, Intra-layer
Small GCN ✓ aggregation - In-situ Feature-level, Node-level
support for update to Edge-level, Batch
prune zeros on-the-fly
GraphACT
[156]Redundancy reduction Intra-layer, Node-level
Small GCN ✗ to reduce the number of Feature-level
edges for aggregation
Zhang et al.
[161]Pre-process to partition Intra-layer, Node-level
Large GCN ✗ and reorder the graph, Feature-level, Edge-level
and redundancy reduction
BoostGCN
[160]3-D tiling of edges, Intra-layer, Node-level
Large GCN ✗ nodes, and features; Feature-level, Edge-level
sort-and-combine unit
G-CoS
[168]Small GCN, GraphSage On-the-fly multicasting Intra-layer, Node-level
Large GAT, GIN ✓ for data distribution Feature-level
GCoD
[153]GCN, GIN Workload balanced denser, Intra-layer, Node-level
Small GAT ✗ light workload sparser Feature-level, Edge-level
I-GCN
[41]Small GCN On the fly reorder Intra-layer, Node-level
Large GraphSage ✓ and islandize the graph Feature-level
accelerator architecture, consisting of a PE array interconnected via a 2D-mesh network NoC. Each
PE embedded in this architecture features a multiply-and-accumulate (MAC) array. To map the graph
and features to the accelerator, the authors presented graph-level and node-level mappings. The
former entails mapping a window of nodes to one PE, allowing for task-level parallelism, with PEs
working on distinct node windows. The latter involves tiling the dense vector-matrix multiplication
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 36

36 Zhang, Shichang, et al.
for the feature update (aggregation and update) using the MAC array. Rubik incorporates a global
buffer for the PE array, two private caches (for data and instruction) in each PE, and register files
(RFs) in every MAC to facilitate data reuse. G-CoS [168] is an automated framework that aims to
optimize the performance and efficiency of GNNs by co-searching for the best GNN structures
and accelerators. This framework consists of two components: (1) a one-shot co-search algorithm
for GNN structures and their matched accelerators, and (2) a generic accelerator search space
that can be applied to various GNN structures. The accelerator search space in G-CoS is based
on a template comprising multiple sub-accelerators that can handle both sparse and dense matrix
multiplications. By configuring the settings of each sub-accelerator, such as tiling sizes, tiling order,
and interconnection style, they can be specialized to process different clusters of GNN operators
with distinct data sizes and sparsity, which ultimately enhances the hardware efficiency. In addition,
G-CoS employs local buffers assigned to the intermediate features, index for sparse features (in
COO format), and weights and intermediate outputs for each sub-accelerator. These buffers can
be interconnected to allow for data sharing and reuse across different sub-accelerators, reducing
the need for costly off-chip memory access. G-CoS is a pioneering effort towards automating the
search for both GNN model structure and accelerator design knobs.
GCoD [153] is a co-design framework that addresses the challenge of extreme sparsity in GNN
inference by optimizing both the algorithm and hardware accelerator. At the algorithm level, GCoD
polarizes the graphs into either denser or sparser local neighborhoods without compromising model
accuracy. This polarization results in adjacency matrices with two levels of workload, which greatly
improves regularity and ease of acceleration. On the hardware level, GCoD integrates a dedicated
two-pronged accelerator comprising a dense branch and a sparse branch processor. The dense
branch processor employs a chunk-based micro-architecture with several heterogeneous computing
modules to balance the workload across different subgraphs from the polarized adjacency matrix.
This balance is achieved by allocating computing resources and bandwidth proportionally based
on the data volume and operation size involved in each subgraph’s processing. The sparse branch
processor accelerates the remaining sparse workloads, which constitute a small portion of nonzero
data, mostly on-chip. To store data in a sparse format, the sparse branch processor uses the CSC
format, which significantly reduces the storage overhead of adjacency matrices. GCoD’s sparse
branch processor also employs a query-based weight forwarding mechanism that flexibly shares
on-chip input data from the dense branch processor to enhance data reuse. The weight forwarding is
performed on-demand of the sparse branch to achieve more efficient control, and it accesses around
63% of the data, which drastically reduces the off-chip access cost. Overall, GCoD’s algorithm-
hardware co-design framework offers better accuracy-efficiency trade-offs than optimizing each
aspect separately. By polarizing the graphs into denser and sparser local neighborhoods and
integrating a two-pronged accelerator with a dense and sparse branch processor, GCoD provides a
comprehensive solution to tackle the challenge of extreme sparsity in GNN inference.
I-GCN [41] proposes a new approach called "islandization" to enhance data locality in GNNs.
By identifying clusters of nodes with strong internal connections, called "islands," I-GCN aims
to minimize off-chip memory accesses and improve on-chip data reuse. To support islandization,
I-GCN’s hardware architecture includes an Island Locator and an Island Consumer. The Island
Locator searches the graph from multiple nodes in parallel to locate highly-connected nodes termed
as hubs. From these hubs, multiple tailored breadth-first-search engines are used to identify islands.
The Island Consumer then combines and aggregates the islands and hubs, with a focus on reusing
aggregation results among nodes with shared neighbors. To optimize the aggregation process,
a ring-based on-chip network is used to distribute partial results and detect reuse opportunities
among processing elements (PEs), and an in-network reduction scheme is employed to minimize
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 37

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 37
network communication latency. Overall, I-GCN’s islandization approach and hardware design
achieve better data locality and efficiency in GNNs.
6.3.2 Dedicated Module per Stage. In contrast to the previous category, the works in this category
propose a specialized engine for the primary computation stages of the GNN algorithms. By
adopting this approach, each engine can be customized according to the requirements of each stage.
It can also facilitate the concurrent execution of different stages.
HyGCN [144] targets GCN, GraphSage, GINConv [ 141], and DiffPool [ 152] and treats their
operations as two main stages, aggregation, and combination (Update in Eq. 1). Due to the distinctive
execution patterns of the aggregation and combination stages, the proposed accelerator employs a
separate processing engine for each stage and processes them in a dataflow manner. Specifically,
the Aggregation engine utilizes parallel single-instruction multiple-data (SIMD) cores to exploit the
intra-vertex parallelism resulting from the long feature vectors assigned to each vertex by the GNN
algorithms. If the feature vector length is smaller than the total number of cores, the unassigned
cores are allocated to other nodes, thereby enabling edge-level parallelism for this stage. As the
graph connections are sparsely distributed, a sparsity eliminator is employed to identify a region
in the adjacency matrix of the corresponding graph partition that stores the effective edges, using
a window sliding and shrinkage technique. To further reduce the computation complexity, the
aggregation operation is performed on a sampled set of neighbors. The sparsity eliminator and
sampler also help to avoid fetching feature vectors of the nodes that are not connected to any edge
in the current graph partition. Subsequently, the Combination engine treats the computation as a
dense matrix-vector multiplication (MVM) and is realized as a group of systolic arrays that can
work independently or cooperatively depending on the configuration. Each systolic array processes
a small group of nodes, with the feature vectors (weights) of the nodes flowing through the row
(column) dimension.
FlowGNN [106] proposes a generic dataflow architecture for accelerating GNNs that can flexibly
support the majority of message-passing GNN algorithms, including GCN, GINConv, PNA [ 24],
GAT [ 125], DGN [ 5], and GNN with virtual node (VN) [ 42]. The authors argued that common
pre-processing techniques used to exploit data locality are not feasible for real-time applications
with millions of input graphs with varied structures. Thus, they directly take the graph in COO
format without any pre-processing to reorganize the data. FlowGNN’s architecture is based on the
idea that any graph-related functions can be expressed with pair-wise message passing. It divides
the operations into three steps: 1) the gather phase, which aggregates messages from neighbors; 2)
the node transformation (NT) phase, which applies the update function (or feature transformation
function) to messages; and 3) the scatter phase, which constructs messages for the next layer
by applying the respective message transformation function (e.g., applying the edge weights) on
the node embeddings produced in step two. The computations are divided into these steps to
integrate edge embeddings and different aggregation functions. The gather and scatter phases
handle per-edge computations and are implemented as a single unit (MP unit) that follows the
NT unit in a dataflow fashion. The NT unit handles per-node computations. As such, the MP (NT)
unit consists of parallel PEs that parallelize the edges (nodes). Each of the PEs further parallelizes
the features. To avoid bank conflicts, each of the PEs in the MP unit processes one particular bank
of edges. FlowGNN uses an NT-to-MP adapter to perform on-the-fly multicasting of the node
embeddings generated by the NT unit to their right PE in the MP unit.
BlockGNN [176] aims to compress the update stage (feature transformation) of GNN models for
GraphSage-Pool (GS-Pool), GCN, GGCN [ 90], and GAT that require a large weight memory, high
computing resources, and high computing latency when the feature size is large. To achieve this,
they leverage block-circulant weight matrices, which are configured in the form of block-circulant
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 38

38 Zhang, Shichang, et al.
matrices that can be accelerated by Fast Fourier Transform (FFT) and Inverse Fast Fourier Transform
(IFFT). This transformation reduces the computation intensity and required storage for the update
stage. The computation engine of BlockGNN is composed of two main units, the CirCore unit for
the update stage and the VPU (vector processing unit) for the aggregation stage and non-linear
operations. The CirCore unit is a three-stage pipeline, consisting of an FFT unit that converts data
from the spatial domain to the spectral domain, a systolic array that performs element-wise product
and accumulation, and an IFFT unit that transforms the results back to the spatial domain. The
VPU is organized as a SIMD unit with m lanes, each with a parallel factor of 16.
DeepBurning-GL [75] is an automated framework designed to facilitate the development of
hardware accelerators for GNNs. The framework takes a GNN model as input, which is designed
using software frameworks such as DGL and PyG, and generates a hardware accelerator for
the target FPGA platform. For the hardware design, the authors provide three templates. The
first template is the GNN computation template that handles dense and sparse multiplications.
Dense multiplications are implemented via a systolic array or a dot production array, while sparse
multiplications are handled by an array of SIMD units or a ring-reduce topology (similar to EnGN).
The second template is the memory template, which consists of a distributed on-chip buffer,
cache, or degree-aware hierarchy on-chip buffer to support both regular and irregular memory
accesses. The third template is the graph manipulation template, which supports graph sampling
and reconstruction in various GNN models. To adapt to the specific characteristics of the target
model and FPGA platform, the authors develop analytical models that analyze the processing
requirements and adopt a design space exploration to tune hardware parameters for the required
templates based on the graph properties, network architecture, and resource constraints. The
authors validate their framework on various GNN algorithms, including GCN, GS-Pool, R-GCN,
and EdgeConv [134].
6.4 Accelerators for Special Workload
The literature in this field endeavors to direct their efforts toward a more specialized algorithm. More
specifically, it concentrates on a singular GNN algorithm (here, mostly GCN), and subsequently
tailors the proposed solutions accordingly. Given that the target model may contain multiple layers,
the first subgroup develops a deep pipeline with custom layer modifications, which are elaborated
upon in Section 6.4.1. Conversely, the second subgroup adopts a fixed hardware approach for all
layers, which is described in detail in Section 6.4.2.
6.4.1 Layer-Customizable Deep Pipeline. The studies presented in this section argue that a funda-
mental optimization in a GCN accelerator should involve reducing the expenses associated with
memory transactions. To this end, they construct a deep pipeline that circumvents the need for
off-chip (global) memory transactions concerning intermediate results among distinct GCN layers.
Through the allocation of specific engines for each layer, they can additionally tailor hardware
parameters based on the individual workload of each layer.
AWB-GCN [40] argues that many real-world graphs follow the power-law distribution, which
implies that the number of nodes with a given degree x follows a proportional relationship with
𝑥−𝛽for a constant 𝛽. This leads to a situation where a small number of rows/columns in the
adjacency matrix contain the majority of non-zero elements, while the rest contain only a few
non-zero elements. The authors also note that ReLU activation leads to an abundant number of zero
elements, making it more efficient to treat the feature transformation step (Update step) as a sparse
computation. In response to these observations, the authors propose an accelerator architecture
called Autotuning-Workload-Balancing (AWB) GCN, which includes distribution smoothing ,evil
row remapping , and remote switching . Since GCNs have linear aggregation and update functions,
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 39

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 39
the authors propose executing the update stage before the aggregation stage since this approach
helps reduce the number of non-zero operations. This is because by this change, in both steps,
we are dealing with two SpMM as opposed to an SpMM and a dense matrix multiplication. AWB-
GCN’s architecture supports inter-layer parallelism in addition to the commonly used intra-layer
parallelism and is based on the inner-product matrix multiplication. The authors partition the
graph and implement a task distributor to navigate the data to idle PEs to balance the workload.
The architectural optimizations are designed to overcome the load imbalance problem. Distribution
smoothing is local and tries to balance the load among neighboring PEs within up to three hops.
The remote switching addresses regional clustering by exchanging the workloads of under- and
overloaded PEs. Evil row remapping distributes rows with the most non-zeros to the most under-
loaded PEs.
StreamGCN [114] presents an efficient and flexible GCN accelerator for streaming small graphs
- from the DRAM, the host CPU, and through the network - and exploiting all the available sparsity.
The authors argue that in memory-bound applications like this, it is essential to minimize memory
transactions, and they propose a scheduling mechanism to achieve this. In this scheduling, input data
is fetched from global memory (DRAM) only once and is reused for all corresponding computations
before being evicted. The proposed GCN accelerator includes dedicated modules for each layer,
which not only allows customization of parallelization based on the workload of each layer but
also enables inter-layer pipelining. As a result, the intermediate results are directly passed to the
modules for the next layer through on-chip FIFOs which further helps with decreasing the number
of DRAM transactions. To avoid read-after-write (RAW) dependencies in the aggregation unit, the
edges are pre-processed and reordered. Both the aggregation unit and the feature transformation
unit contain multiple SIMD PE with the SIMD dimension parallelizing the node features. The PEs
in the aggregation unit process different pairs of neighbors in parallel with their respective edge
weight (edge parallelism), while in the feature transformation unit, they parallelize the nodes. These
units are connected in a dataflow fashion. Similar to AWB-GCN, StreamGCN treats the feature
transformation step as a sparse computation. However, due to the smaller size of the target graphs,
the method is based on outer-product-based multiplication. This scheduling is shown to reduce the
number of pipeline stalls due to RAW dependencies. Since the intermediate results are to directly
be processed as they are being produced, StreamGCN has in-situ sparsity support by developing a
mechanism to prune the zero elements of the node embeddings on-the-fly, while they are being
generated. This is realized by a pruner and an arbiter unit. The pruner pre-fetches the results at a
higher rate than the next layer’s processing rate to pass non-zero elements through FIFOs. The
arbiter then checks them for RAW dependencies and dispatches enough elements to fill the PEs.
Overall, the proposed StreamGCN architecture is ideal for real-time or near-real-time graph search
and similarity computation for many biological, chemical, or pharmaceutical applications.
6.4.2 All-Layer Unified Architecture. As seen in the previous section, a GCN algorithm consists
of multiple layers that have distinct features (e.g., node embedding dimension). In contrast to
the works detailed in Section 6.4.1, the studies presented in this category put forth the idea of
constructing a more adaptable architecture across distinct layers and utilizing the same engines for
all layers. Since the works in this category focus on a specific workload, namely GCN, they offer
greater possibilities for customization than those presented in Section 6.3, as indicated below:
GraphACT [156] develops an accelerator for training GCNs targeting small graphs on CPU-
FPGA heterogeneous systems. The small graphs are created by sampling subgraphs of the input
graph to be able to fit them into the on-chip resources. GraphACT does not work directly with
the normalized adjacency matrix and rather defines three types of multiplications for GCNs: 1)
dense (weights) - dense (embeddings), 2) binary sparse (adjacency matrix) - dense, 3) diagonal
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 40

40 Zhang, Shichang, et al.
(degree matrix) - dense. The first multiplication is performed using a 2D systolic array called
the Weight Transformation module, where nodes are mapped to the row dimension and output
features to the column dimension. The Feature Aggregation module implements steps two and
three. The third multiplication (Step three) is treated as scaling the rows of the second matrix in the
multiplication with minimal hardware overhead. GraphACT implements a redundancy reduction
technique to reduce the computation cost of the second step, which seeks to pre-compute the
repeated aggregations on CPU. This is possible since GraphACT directly works with the adjacency
matrix in its binary (not normalized) format. The redundancy reduction algorithm identifies the
common pairs of neighbors and merges each of the most repeated ones in a node. This is done
as a pre-processing step on CPU since it is communication-intensive. The Feature Aggregation
module has a 1D accumulator array that parallelizes the feature dimension. It adds the features
of the identified node pairs and performs the aggregation based on the adjacency matrix of the
merged graph (generated after merging the node pairs). Finally, it scales the node features based on
the node degrees (Step 3). Both the Feature Aggregation and Weight Transformation modules are
reused for all the 𝐿layers of the GCN.
Zhang et al. [161] focus on accelerating GCN inference for large graphs. The approach begins by
partitioning the input data into smaller sizes that can fit into the on-chip resources for processing.
Following this, a redundancy reduction scheme, such as GraphACT, and a node-reordering phase
are applied to decrease global memory access. Upon completion of the pre-processing stage, the
graph is passed to a hardware architecture that incorporates two primary modules: Aggregation and
Weight Transformation. The Aggregation module employs parallel vector accumulators (VAs) for
sparse matrix multiplication in the GCN aggregation process. Here, each VA parallelizes the feature
dimension and different VAs process distinct edges. The Weight Transformation module leverages
a systolic array to implement the update stage. These modules are bi-directionally connected to
enable different modes of scheduling, where the order of executing the aggregation and update
stage can vary based on the GCN and dataset properties.
BoostGCN [160] presents a framework aimed at optimizing GCN inference on FPGAs. The
hardware architecture of BoostGCN includes two primary modules, namely the feature aggregation
module (FAM) and the feature update module (FUM). These modules are interconnected through
internal buffers that cache intermediate results, and the connection between them is bi-directional,
as in Zhang et al.’s work. To reduce memory traffic and address workload imbalance, the authors
propose Partition-Centric Feature Aggregation (PCFA), which leverages tiling in three dimensions:
edges, nodes, and features. FAM operates on multiple edges in parallel and further parallelizes the
feature dimension for them. Additionally, FAM uses a sort-and-combine (SaC) unit to sort the inputs
and combine them if two of them have the same destination, which reduces congestion. BoostGCN
employs two types of FUM based on the sparsity of the feature matrix (node embeddings) – Sparse-
FUM and Dense-FUM. Sparse-FUM uses the same structure as FAM, with a format transformation
module in the beginning that transforms the input to COO format. Dense-FUM, on the other hand,
implements a 2D systolic array.
Table 7 shows a review of accelerators, focusing on their validation methodology and reported
speedup. These accelerators have been implemented on FPGAs or simulated as an ASIC design. A
prevalent baseline for reporting the accelerator’s performance is CPU and GPU, with most works
utilizing the state-of-the-art implementation of PyG for this purpose. The reported results may
vary regarding whether they exclusively measure the runtime of the execution of the main kernel
for their accelerator or include other associated overheads and present an end-to-end runtime.
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 41

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 41
Table 7. Performance and Validation Methodology of the Customized Accelerators
Work Validation MethodologyEnd-to-end
MeasurementReported Speedup
EnGN
[76]ASIC (TSMC 14nm with HBM 2.0 at 1GHz) ✗ 1802.9×,19.75×, and 2.97×over CPU
cycle-accurate simulator (PyG &DGL), GPU (PyG &DGL), and HyGCN
Rubik
[17]ASIC (45nm at 500MHz) ✗ 3.42−46.7×, and 1.3−14.16×over
cycle-accurate simulator PyG-GPU, and Eyeriss [19]-like accelerator
HyGCN
[144]ASIC (TSMC 12nm with HBM at 1GHz) ✗ 1509×, and 6.5×over
cycle-accurate simulator PyG-CPU, and PyG-GPU
FlowGNN
[106]FPGA (Xilinx Alveo U50 at 300MHz) ✓ 24−254×,1.3−477×, and 1.26×over
on-board evaluation PyG-CPU, PyG-GPU, and I-GCN
BlockGNN
[176]FPGA (Xilinx ZC706 at 100MHz) ✗ 2.3×, and 4.2−8.3×over
on-board evaluation and analytical modeling CPU (TensorFlow-based), and HyGCN
DeepBurning
-GL [75]FPGA (Xilinx ZC706, KCU1500, and Alveo U50) ✗ 7.43−346.98×,3.7−16.5×, and 6.28×over
on-board evaluation at 100, 200, and 200MHz DGL-CPU, DGL-GPU, and HyGCN on FPGA
AWB-GCN
[40]FPGA (Intel Stratix 10 SX) ✗ 3255×,80.3×,5.1×over
on-board evaluation PyG-CPU, PyG-GPU, and HyGCN
StreamGCN
[114]FPGA (Xilinx KU15P, Alveo U50 and U280) ✓ 8.2−18.2×, and 12.1−26.9×over
on-board evaluation at 201, 279, 290MHz PyG-CPU, and PyG-GPU
GraphACT
[156]FPGA (Xilinx Alveo U200 at 200MHz)✓
(training speed)12−15×, and 1.1−1.5×over
on-board evaluation CPU, and GPU
Zhang et al.
[161]FPGA (Xilinx Alveo U200 at 250MHz) ✗ 30×, and 2×over
on-board evaluation CPU, and GPU - TensorFlow and C++ (Cuda)
BoostGCN
[160]FPGA (Intel Stratix 10 GX 10M at 250MHz) ✗ 100×,30×,3-45×over CPU (PyG &DGL),
on-board evaluation GPU (PyG &DGL), and FPGA ([161] &HyGCN)
G-CoS
[168]FPGA (Xilinx VCU128 at 330 MHz)
on-board evaluation✗ 5.52×, and 1.92×speedups over
HyGCN, AWB-GCN
GCoD
[153]FPGA (Xilinx VCU128 at 330 MHz)
on-board evaluation✗ 7.8×, and 2.5×over
HyGCN, AWB-GCN
I-GCN
[41]FPGA (Intel Stratix 10 SX at 330 MHz)
on-board evaluation✗ 368×,453×,16×, and 5.7×over
PyG-GPU, DGL-GPU, SIGMA, and AWB-GCN
7 GNN Acceleration: Special Graphs and Special GNNs
In the sections above we talked about acceleration techniques for the regular message-passing
GNNs on regular graphs. However, many practical problems are better modeled with special
graph types, e.g., heterogeneous graphs and dynamic graphs (both defined below), and solved
with more sophisticated GNN architectures, e.g., heterogeneous GNNs and GNNs with attention.
Many methods we discussed above may be applied to these special cases if their specialties are
ignored, e.g, treating a heterogeneous graph as if it is homogeneous. However, the result would be
suboptimal such as having high accuracy loss. In this section, we discuss techniques for handling
these special cases especially.
7.1 GNN Acceleration on Heterogeneous Graphs
The graphs we talked about so far have nodes/edges of the same type, and they are called homo-
geneous graphs. The complex graphs in the real world are often associated with various types
of nodes and edges [ 30,55,61], establishing a heterogeneous structure. For example in a citation
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 42

42 Zhang, Shichang, et al.
graph, nodes can be authors, venues, and papers, and edges can denote an author published a paper,
a paper published in a venue, etc. Different types of nodes and edges have substantially different
meanings from each other and it is better to distinguish them explicitly. Heterogeneous graphs as a
new data structure are defined below.
Definition 7.1. A heterogeneous graph is defined as a directed graph G=(V,E)associated with
a node type mapping function 𝜙:V→A and an edge type mapping function 𝜏:E→R . Each
node𝑣∈V belongs to one node type 𝜙(𝑣)∈A and each edge 𝑒∈Ebelongs to one edge type
𝜏(𝑒)∈R .
Many GNNs are specially designed for heterogeneous graphs [ 53,83,162]. They face similar
scalability challenges as regular GNNs on homogeneous graphs and require acceleration. In the
previous sections, we discussed acceleration methods that could be used for GNNs on heteroge-
neous graphs by disregarding their heterogeneity (i.e., different node and edge types). However,
applying acceleration methods in a homogeneous manner by ignoring heterogeneity often leads to
a significant decrease in performance. For example, for sampling algorithms, if a message passing
step sample node neighbors uniformly without considering the node types, some infrequent types
are likely missed out. Completely missing one type of node can downgrade the GNN performance
a lot [ 162]. A better sampling algorithm on heterogeneous graphs should perform a weighted
sampling to include neighbor nodes of all types. Similarly for model pruning, heterogeneous GNNs
often have separate model weights in the update function for each edge types [ 107], and ignoring
heterogeneity may prune the weights completely for some edge types, which makes the GNN
unable to pass messages that type of edges and downgrade the performance. Therefore, accelerating
GNNs while taking into account heterogeneity presents additional challenges and requires special
treatment to capture heterogeneity. In the following, we will provide a summary of recent research
on modeling and accelerating GNNs on heterogeneous graphs.
HetGNN [162] is a GNN on heterogeneous graphs that introduces a novel heterogeneous
neighbor sampler. The sampler belongs to the node-wise sampling category discussed in Section 3.2.1
but overcomes a special challenge on heterogeneous graphs. On heterogeneous graphs, node types
in a target node’s receptive field can be very imbalanced, e.g., some types are very infrequent
compared to other types. HetGNN argues that a regular sampler may miss some less common node
types completely and result in insufficient node representations. Therefore, instead of randomly
sampling, the HetGNN sampler first runs random walks with restart (RWR) in the neighborhood
and then chooses nodes to form the computation graph according to their hit frequency by RWR.
The sampler will also balance node types, such that the top frequent nodes of all types are chosen.
With this RWR-based and type-balanced sampler, HetGNN can achieve fast training with good
accuracy performance.
HGT [53] is a transformer-based GNN on heterogeneous graphs, which also introduces an
efficient sampler to make the model handle Web-scale graph data with billions of edges. The HGT
samples subgraphs in mini-batches. The sampler is able to 1) maintain a similar number of nodes
and edges for each type in each subgraph and 2) keep the sampled subgraph dense. Specifically, for
each node type it maintains a node budget and iteratively samples the same number of nodes with
an importance sampling strategy to reduce variance. The sampling probability for each node type
is calculated based on the square of the cumulative normalized degree of other nodes that have
already been sampled in its budget. The model is shown to achieve state-of-the-art results on the
Open Academic Graph (OAG) which contains millions of nodes and billions of edges.
Pigeon [135] is an intermediate representation (IR) and code generator for end-to-end training
and inference of relational graph neural networks (RGNNs) [ 107] on heterogeneous graphs. Pigeon
addresses the performance challenges posed by RGNNs by decoupling the model semantics, data
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 43

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 43
layout, and operator-specific schedule, and expressing these opportunities to allow them to be
integrated into the design space as integral elements. The proposed solution includes a compact
tensor materialization scheme and a linear operator fusion pass to achieve both training and
inference. The compact tensor materialization scheme reduces the resources spent on computing
and storing common subexpressions in RGNNs. Specifically, certain edge data are determined by
source node features and edge types, and instead of computing and storing such data for each edge,
the system computes and stores the data once for each (edge type, unique node index) pair. This
reduces the amount of computation and storage required for common subexpressions. The linear
operator fusion pass switches the order of linear operators, such as linear layers and inner products.
When a linear operator is followed by another linear operator, their order may be switched to
reduce the strength.
7.2 GNN Acceleration on Dynamic Graphs
Spatio-temporal data appears frequently in scientific research and real-world applications [ 56,105].
Dynamic graphs have shown great performance for modeling such spatio-temporal data to solve
problems like molecular dynamics in biology [ 147], information diffusion over social networks [ 132],
and epidemic spread in public health [ 57]. Dynamic graphs are graphs that can change over
time. Changes include the addition and removal of nodes and edges to capture the evolving
nature of the data. The evolving nature has imposed additional challenges for corresponding
learning and reasoning on dynamic graphs as opposed to static graphs. Therefore, researchers have
developed special spatio-temporal GNNs to jointly model the graph structural information on the
spatial aspect and the evolving dynamics on the temporal aspect. The regular message passing in
Equation (1) has been modified as is shown in Equation (16) for spatio-temporal GNNs, where the
node representations are updated by both their spatial neighbors and their historical states.
𝒉𝑡
𝑣=UPDATE(𝒉𝑡−𝑘:𝑡−1
𝑣,AGGR({𝜙(𝒉𝑡−1
𝑣,𝒉𝑡−1
𝑢)|(𝑢𝑡−1,𝑣𝑡−1∈E𝑡−1})) (16)
HereE𝑡−1denotes the edges for graph at timestamp 𝑡−1, and sometimes the edges can be latent
and need to be inferred first before the message passing.
Spatio-temporal GNNs are often applied to computation-heavy problems like the simulation
of large-scale dynamical systems [ 105], which have similar scalability concerns as regular GNNs
on static graphs and require acceleration. However, spatio-temporal GNN acceleration incurs
special challenges because model training needs to be done chronologically and graph operations
(modification and sampling) need to consider both spatial and temporal neighbors. For example, the
sampling methods introduced in Section 3.2 may be individually applied to dynamic graphs at each
timestamp𝑡, but they can fall short in performance as they only consider spatial properties of nodes
but omit the temporal influence. In this section, we summarize recent works on spatio-temporal
GNN acceleration, which are often applied for learning and reasoning over large-scale dynamic
graphs.
TGL [175] proposes a unified framework for accelerating continuous-time temporal GNN training.
It designs a temporal-CSR data structure for rapid access of temporal neighborhood candidates,
and a parallel sampler is further equipped to accelerate temporal neighborhood sampling. Also,
it proposes a novel random chunk scheduling technique for training acceleration where simply
increasing the batch size would result in the obsolete node memory issue. TGL [ 175] shows its great
capability in handling large-scale dynamic graphs where it achieves an average of 173 ×speedup
on a multi-core CPU compared with the baselines.
MTGNN [138] applies GNNs to model the spatio-temporal graph for solving the multivariate
time series prediction problem, where variables are dependent on each other but their dependence
(graph structure) is latent. Naively assuming a fully-connected graph structure will result in 𝑁2
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 44

44 Zhang, Shichang, et al.
edges and impose huge GNN computation time. MTGNN proposes a graph learning module to
automatically extract the uni-directed relations among nodes, which greatly simplifies the structure
and accelerates GNN computation. MTGNN also applies clustering to reduce memory usage by the
intermediate states of nodes and save memory access time. Specifically, the edge search space is
reduced by first randomly separating nodes into clusters and then only forming edges within each
cluster. The clustering process for nodes is repeated in each iteration, which allows every pair of
nodes in the same cluster to be possibly connected.
GraphODEs [56,57] combine the expressive power of GNNs with the principled ordinary
differential equation (ODE) modeling for learning dynamical systems. LG-ODE [56] is a latent
ODE generative model, which models multi-agent system dynamics with fixed graph structure in a
continuous manner. The ODE function in LG-ODE is a GNN to capture the continuous interaction
among agents, and the latent initial states for ODEs are inferred through another GNN-based
encoder. Towards acceleration, existing discrete neural models learn a fixed-step transition function
that takes the system state at time 𝑡as input to predict the state at time 𝑡+1. An adaptive ODE
solver will automatically adjust the times, where the graph interaction module is called to provide
the balance between model performance and time consumption. Additionally, LG-ODE is especially
superior at making long-range predictions. It is able to adjust its prediction length without re-
feeding its previous predictions at the same fixed step during training, which greatly accelerates
model inference. Moreover, since LG-ODE is only capable of handling dynamic graphs with evolving
nodes but fixed edges, the follow-up work CG-ODE [57] is proposed to jointly model the evolution
of nodes and edges through two coupled ODE functions.
GN-ETA [28] proposes a large-scale GNN model for estimating the time of arrival (ETA) in real
time. Supersegment-based acceleration is applied so that the complex interaction of traffic roads and
the traffic conditions evolution can be joint modeled over time. In particular, GN-ETA segments the
traffic network into several supersegments and is trained to predict both the across-supersegments
time and within-supersegments time. For model inference, the sequence of supersegments will be
put in a route sequentially where the later supersegements would utilize the prediction time of
an earlier one to determine its relevant prediction horizons for scaling up. Additionally, GN-ETA
adopts the MetaGradients methodology [ 142] to stabilize the training process raised by uneven
query batches that are commonly seen in real-life applications. GN-ETA is able to handle web-scale
applications like Google Maps. Recently, the follow-up work CompactETA [ 37] proposes to encode
temporal information using positional encoding instead of recurrent networks to further accelerate
the model.
Overall, spatio-temporal GNNs for dynamic graphs are extensions of static GNNs with special
treatment for the temporal aspect. Traditional methods usually employ recurrent neural networks
to capture the evolving nature of networks over time, but are usually time-consuming and fail to
capture the long-term dependency. Recent advances to use positional encoding and neural ODEs,
etc have shown promising results in speeding up dynamic GNNs. The works discussed above
accelerate spatio-temporal GNNs in various ways to scale up them to computation-heavy dynamic
system modeling and web-scale applications.
The proposed hardware accelerators in Section 6.2 have demonstrated the capability to support
various graph topologies, provided that the graph is within the scale specified by the accelerator. This
is attributed to the implementation of preprocessing techniques, such as graph re-ordering and tiling,
within these works. However, challenges arise in supporting dynamic and heterogeneous graphs
when the algorithm utilized to process them is altered. To address this issue without necessitating
a complete redesign of the accelerator for each algorithmic change, software-hardware co-design
can be extended. On the software front, an analyzer can be developed to evaluate the requirements
of the graph and target algorithm and determine the optimal partitioning, mapping, and scheduling
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 45

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 45
of operations to enable efficient execution on the hardware. On the hardware side, options include
moving towards more general design architectures or supporting multiple specialized architectures
that the software component can select based on the application requirements.
8 Future Directions
While significant progress has been achieved, the investigation into GNN acceleration remains in
its early stages and has not kept pace with the rapid expansion of graph data. Numerous promising
research avenues exist within this field, particularly in terms of acceleration from a COTS system and
customized hardware perspective, which have only recently begun to garner attention. Additionally,
there is a need to explore acceleration techniques tailored for heterogeneous and dynamic graphs.
This section highlights several captivating research directions that warrant further exploration.
Better Graph Modification Algorithms Improving the efficiency of graph modification meth-
ods, either coarsening, sparsification, or condensation, represents a promising avenue for future
research. In many graph datasets, nodes often have an excessive number of neighbors, which
may provide redundant information or even noise. To accelerate GNN training, it may be possible
to remove these neighbors more aggressively at the pre-processing stage. For example, through
sparsification without preserving the global graph properties as practical GNNs are trained for
preserving locality. Graph condensation is another promising technique that can enhance training
efficiency, but it currently requires running the training process at least once to identify the best
condensed graph. If the optimal graph can be determined earlier, e.g., identify a good condensed
graph by training only a few epochs, this technique would become much more useful. Good graph
modification methods can also benefit the automating designing and tuning GNN architectures.
In practice, preprocessed simpler graphs will significantly accelerate the whole pipeline of GNN
development.
Better Sampling Algorithms There are still several potential improvements for future research
in sampling methods for GNNs. One potential direction is to investigate hierarchical sampling
methods that can exploit the multi-scale structure of graphs to improve sampling efficiency and
capture multi-scale features of graphs. Hierarchical sampling can be particularly useful for large-
scale graphs with complex structures, such as social networks, where nodes can be organized into
clusters or communities at different levels. Another possibility is to further investigate the sampling
method for dynamic graphs, where nodes and edges change over time. Individually applying
sampling to dynamic graphs at each timestamp is still not very efficient and fails to capture the
temporal dynamics of nodes and edges on dynamic graphs.
Combined Inference Acceleration Algorithms Inference acceleration methods, such as
pruning, quantization, and distillation, are often orthogonal to each other and can be combined to
further improve the speed. Some initial steps have already been taken, such as leveraging distillation
when ambitiously quantizing weights to binary. As the inference stage is more flexible training,
e.g., no concerns like training stability and model convergence, these methods can be more easily
combined in various ways. For instance, pruning and quantization can be combined to achieve a
sparse model with low precision. Additionally, pruning can be used in conjunction with distillation
to further refine the distilled student model.
COTS Systems Existing work mainly focuses on improving the run-time efficiency of GNNs.
In contrast, the memory requirement can often be the bottleneck of accelerating GNNs on large-
scale data, which has not been well studied yet. Various techniques including rematerialization,
quantization, and data pruning can be considered for reducing the memory overhead. In addition,
because most GNN models have irregular computation and data access patterns, the advanced
training/inference algorithms can be system unfriendly. This makes system-algorithm co-design
for system-friendly GNN algorithms a promising research direction. Finally, full-stack development
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 46

46 Zhang, Shichang, et al.
for GNN systems is still missing in the existing literature. Combining existing techniques at
different levels (e.g., algorithm, system, hardware) towards an ultra-efficient GNN system is another
interesting research topic.
Customized Hardware As explained in Section 6, memory access presents a significant bot-
tleneck in GNN acceleration. Thus, a key aspect of accelerating the processing of GNNs is to
minimize the amount of data that must be fetched from off-chip memory. In addition to the
techniques employed by the works reviewed, such as data reuse, on-chip memory caching, and
dataflow optimization, exploring approaches such as pruning, quantization, and compression may
reduce the volume of data fetched from off-chip memory. Furthermore, a critical requirement is
to develop a more advanced automated design generator that facilitates GNN accelerator design.
Techniques such as Neural Architecture Search (NAS)[ 78], learning-based models for accelerated
hardware design evaluation[ 113], and intelligent design space exploration to navigate through
design candidates[91, 116] can be suitably adapted for GNN accelerator design.
9 Conclusion
In conclusion, we have presented a systematic view of acceleration methods for GNNs on large-scale
datasets. Our survey covers GNN training and inference algorithms, efficient GNN systems, and
customized hardware for GNNs. Additionally, we have discussed the special cases when GNNs
are applied to heterogeneous graphs and dynamic graphs. Acceleration methods pave the way for
more efficient GNN pipelines, enabling the use of larger GNN models for solving more complex
tasks on graphs.
Acknowledgments
We thank Linghao Song for the helpful discussions and comments throughout this work. We thank
Xiang Song and Kezhao Huang for providing helpful feedback on an early draft of the paper. This
work is partially supported by NSF (2211557, 1937599, 2119643), NASA, SRC, Okawa Foundation
Grant, Amazon Research Awards, Cisco Research Grant, Picsart Gifts, and Snapchat Gifts.
References
[1]Sergi Abadal, Akshay Jain, Robert Guirado, Jorge López-Alonso, and Eduard Alarcón. 2021. Computing graph neural
networks: A survey from algorithms to accelerators. ACM Computing Surveys (CSUR) 54, 9 (2021), 1–38.
[2]Uri Alon and Eran Yahav. 2021. On the Bottleneck of Graph Neural Networks and its Practical Implications. In
International Conference onLearning Representations. https://openreview.net/forum?id=i80OPhOCVH2
[3]Mehdi Bahri, Gaétan Bahl, and Stefanos Zafeiriou. 2021. Binary graph neural networks. In Proceedings ofthe
IEEE/CVF Conference onComputer Vision andPattern Recognition. 9492–9501.
[4]Jiyang Bai, Yuxiang Ren, and Jiawei Zhang. 2021. Ripple walk training: A subgraph-based training framework for
large and deep graph neural network. In 2021 International Joint Conference onNeural Networks (IJCNN) . IEEE,
1–8.
[5]Dominique Beaini, Saro Passaro, Vincent Létourneau, Will Hamilton, Gabriele Corso, and Pietro Liò. 2021. Directional
graph networks. In International Conference onMachine Learning. PMLR, 748–758.
[6]András A Benczúr and David R Karger. 1996. Approximating st minimum cuts in Õ (n 2) time. In Proceedings ofthe
twenty-eighth annual ACM symposium onTheory ofcomputing. 47–55.
[7]Yoshua Bengio, Nicholas Léonard, and Aaron Courville. 2013. Estimating or propagating gradients through stochastic
neurons for conditional computation. arXiv preprint arXiv:1308.3432 (2013).
[8]Gecia Bravo Hermsdorff and Lee Gunderson. 2019. A unifying framework for spectrum-preserving graph sparsification
and coarsening. Advances inNeural Information Processing Systems 32 (2019).
[9]Cristian Bucilu ˇa, Rich Caruana, and Alexandru Niculescu-Mizil. 2006. Model compression. In Proceedings ofthe12th
ACM SIGKDD international conference onKnowledge discovery anddata mining. 535–541.
[10] Chen Cai, Dingkang Wang, and Yusu Wang. 2021. Graph Coarsening with Neural Networks. In International
Conference onLearning Representations. https://openreview.net/forum?id=uxpzitPEooJ
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 47

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 47
[11] Zhenkun Cai, Qihui Zhou, Xiao Yan, Da Zheng, Xiang Song, Chenguang Zheng, James Cheng, and George Karypis.
2023. DSP: Efficient GNN Training with Multiple GPUs. In Proceedings ofthe28th ACM SIGPLAN Annual Symposium
onPrinciples andPractice ofParallel Programming. 392–404.
[12] Jie Chen, Tengfei Ma, and Cao Xiao. 2018. Fastgcn: fast learning with graph convolutional networks via importance
sampling. arXiv preprint arXiv:1801.10247 (2018).
[13] Jianfei Chen, Jun Zhu, and Le Song. 2017. Stochastic training of graph convolutional networks with variance reduction.
arXiv preprint arXiv:1710.10568 (2017).
[14] Ming Chen, Zhewei Wei, Zengfeng Huang, Bolin Ding, and Yaliang Li. 2020. Simple and Deep
Graph Convolutional Networks. In Proceedings ofthe37th International Conference onMachine Learning
(Proceedings of Machine Learning Research, Vol. 119), Hal Daumé III and Aarti Singh (Eds.). PMLR, 1725–1735.
https://proceedings.mlr.press/v119/chen20v.html
[15] Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan
Wang, Yuwei Hu, Luis Ceze, et al .2018.{TVM}: An automated{End-to-End}optimizing compiler for deep learning.
In13th USENIX Symposium onOperating Systems Design andImplementation (OSDI 18). 578–594.
[16] Tianlong Chen, Yongduo Sui, Xuxi Chen, Aston Zhang, and Zhangyang Wang. 2021. A unified lottery ticket hypothesis
for graph neural networks. In International Conference onMachine Learning. PMLR, 1695–1706.
[17] Xiaobing Chen, Yuke Wang, Xinfeng Xie, Xing Hu, Abanti Basak, Ling Liang, Mingyu Yan, Lei Deng, Yufei Ding, Zidong
Du, et al .2021. Rubik: A hierarchical architecture for efficient graph neural network training. IEEE Transactions on
Computer-Aided Design ofIntegrated Circuits andSystems (2021).
[18] Yuzhao Chen, Yatao Bian, Xi Xiao, Yu Rong, Tingyang Xu, and Junzhou Huang. 2021. On Self-Distilling Graph
Neural Network. In Proceedings oftheThirtieth International Joint Conference onArtificial Intelligence, IJCAI-21 ,
Zhi-Hua Zhou (Ed.). International Joint Conferences on Artificial Intelligence Organization, 2278–2284. https:
//doi.org/10.24963/ijcai.2021/314 Main Track.
[19] Yu-Hsin Chen, Joel Emer, and Vivienne Sze. 2016. Eyeriss: A spatial architecture for energy-efficient dataflow for
convolutional neural networks. ACM SIGARCH Computer Architecture News 44, 3 (2016), 367–379.
[20] Zhaodong Chen, Mingyu Yan, Maohua Zhu, Lei Deng, Guoqi Li, Shuangchen Li, and Yuan Xie. 2020. fuseGNN:
accelerating graph convolutional neural network training on GPGPU. In 2020 IEEE/ACM International Conference
OnComputer Aided Design (ICCAD). IEEE, 1–9.
[21] Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, and Cho-Jui Hsieh. 2019. Cluster-gcn: An efficient
algorithm for training deep and large graph convolutional networks. In Proceedings ofthe25th ACM SIGKDD
International Conference onKnowledge Discovery &Data Mining. 257–266.
[22] Avery Ching, Sergey Edunov, Maja Kabiljo, Dionysios Logothetis, and Sambavi Muthukrishnan. 2015. One trillion
edges: Graph processing at facebook-scale. Proceedings oftheVLDB Endowment 8, 12 (2015), 1804–1815.
[23] Weilin Cong, Rana Forsati, Mahmut Kandemir, and Mehrdad Mahdavi. 2020. Minimal variance sampling with
provable guarantees for fast training of graph neural networks. In Proceedings ofthe26th ACM SIGKDD International
Conference onKnowledge Discovery &Data Mining. 1393–1403.
[24] Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Liò, and Petar Veličković. 2020. Principal neighbourhood
aggregation for graph nets. Advances inNeural Information Processing Systems 33 (2020), 13260–13271.
[25] Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier. 2017. Language modeling with gated convolutional
networks. In International conference onmachine learning. PMLR, 933–941.
[26] Chenhui Deng, Zhiqiang Zhao, Yongyu Wang, Zhiru Zhang, and Zhuo Feng. 2020. GraphZoom: A Multi-level Spectral
Approach for Accurate and Scalable Graph Embedding. In International Conference onLearning Representations .
https://openreview.net/forum?id=r1lGO0EKDH
[27] Xiang Deng and Zhongfei Zhang. 2021. Graph-Free Knowledge Distillation for Graph Neural Networks.
arXiv:2105.07519 [cs.LG]
[28] Austin Derrow-Pinion, Jennifer She, David Wong, Oliver Lange, Todd Hester, Luis Perez, Marc Nunkesser, Seongjae
Lee, Xueying Guo, Brett Wiltshire, et al .2021. Eta prediction with graph neural networks in google maps. In
Proceedings ofthe30th ACM International Conference onInformation &Knowledge Management. 3767–3776.
[29] Jialin Dong, Da Zheng, Lin F Yang, and Geroge Karypis. 2021. Global neighbor sampling for mixed CPU-GPU training
on giant graphs. arXiv preprint arXiv:2106.06150 (2021).
[30] Yuxiao Dong, Ziniu Hu, Kuansan Wang, Yizhou Sun, and Jie Tang. 2020. Heterogeneous Network Representation
Learning.. In IJCAI, Vol. 20. 4861–4867.
[31] Yingtong Dou, Zhiwei Liu, Li Sun, Yutong Deng, Hao Peng, and Philip S. Yu. 2020. Enhancing Graph Neural Network-
Based Fraud Detectors against Camouflaged Fraudsters. In Proceedings ofthe29th ACM International Conference on
Information &Knowledge Management (Virtual Event, Ireland) (CIKM ’20) . Association for Computing Machinery,
New York, NY, USA, 315–324. https://doi.org/10.1145/3340531.3411903
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 48

48 Zhang, Shichang, et al.
[32] Steven K Esser, Jeffrey L McKinstry, Deepika Bablani, Rathinakumar Appuswamy, and Dharmendra S Modha. 2019.
Learned step size quantization. arXiv preprint arXiv:1902.08153 (2019).
[33] Boyuan Feng, Yuke Wang, Xu Li, Shu Yang, Xueqiao Peng, and Yufei Ding. 2020. Sgquant: Squeezing the last bit
on graph neural networks with specialized quantization. In 2020 IEEE 32nd International Conference onTools with
Artificial Intelligence (ICTAI). IEEE, 1044–1052.
[34] Matthias Fey and Jan E. Lenssen. 2019. Fast Graph Representation Learning with PyTorch Geometric. In ICLR
Workshop onRepresentation Learning onGraphs andManifolds.
[35] Matthias Fey, Jan E Lenssen, Frank Weichert, and Jure Leskovec. 2021. Gnnautoscale: Scalable and expressive graph
neural networks via historical embeddings. In International Conference onMachine Learning. PMLR, 3294–3304.
[36] Jonathan Frankle and Michael Carbin. 2018. The lottery ticket hypothesis: Finding sparse, trainable neural networks.
arXiv preprint arXiv:1803.03635 (2018).
[37] Kun Fu, Fanlin Meng, Jieping Ye, and Zheng Wang. 2020. CompactETA: A Fast Inference System for Travel Time
Prediction. In Proceedings ofthe26th ACM SIGKDD International Conference onKnowledge Discovery &amp; Data
Mining. 3337–3345.
[38] Qiang Fu, Yuede Ji, and H Howie Huang. 2022. TLPGNN: A Lightweight Two-Level Parallelism Paradigm for Graph
Neural Network Computation on GPU. In Proceedings ofthe31st International Symposium onHigh-Performance
Parallel andDistributed Computing. 122–134.
[39] Swapnil Gandhi and Anand Padmanabha Iyer. 2021. P3: Distributed deep graph learning at scale. In 15th{USENIX}
Symposium onOperating Systems Design andImplementation ({OSDI}21). 551–568.
[40] Tong Geng, Ang Li, Runbin Shi, Chunshu Wu, Tianqi Wang, Yanfei Li, Pouya Haghi, Antonino Tumeo, Shuai Che, Steve
Reinhardt, et al .2020. AWB-GCN: A graph convolutional network accelerator with runtime workload rebalancing. In
2020 53rd Annual IEEE/ACM International Symposium onMicroarchitecture (MICRO). IEEE, 922–936.
[41] Tong Geng, Chunshu Wu, Yongan Zhang, Cheng Tan, Chenhao Xie, Haoran You, Martin Herbordt, Yingyan Lin,
and Ang Li. 2021. I-GCN: A graph convolutional network accelerator with runtime locality enhancement through
islandization. In MICRO-54: 54th annual IEEE/ACM international symposium onmicroarchitecture. 1051–1063.
[42] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. 2017. Neural message passing
for quantum chemistry. In International conference onmachine learning. PMLR, 1263–1272.
[43] Aditya Grover and Jure Leskovec. 2016. node2vec: Scalable feature learning for networks. In Proceedings ofthe22nd
ACM SIGKDD international conference onKnowledge discovery anddata mining. 855–864.
[44] Suyog Gupta, Ankur Agrawal, Kailash Gopalakrishnan, and Pritish Narayanan. 2015. Deep learning with limited
numerical precision. In International conference onmachine learning. PMLR, 1737–1746.
[45] Will Hamilton, Zhitao Ying, and Jure Leskovec. 2017. Inductive representation learning on large graphs. In Advances
inneural information processing systems. 1024–1034.
[46] Will Hamilton, Zhitao Ying, and Jure Leskovec. 2017. Inductive representation learning on large graphs. Advances in
neural information processing systems 30 (2017).
[47] Song Han, Huizi Mao, and William J Dally. 2015. Deep compression: Compressing deep neural networks with pruning,
trained quantization and huffman coding. arXiv preprint arXiv:1510.00149 (2015).
[48] Song Han, Jeff Pool, John Tran, and William J. Dally. 2015. Learning Both Weights and Connections for Efficient
Neural Networks. In Proceedings ofthe28th International Conference onNeural Information Processing Systems -
Volume 1 (Montreal, Canada) (NIPS’15). MIT Press, Cambridge, MA, USA, 1135–1143.
[49] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the Knowledge in a Neural Network.
arXiv:1503.02531 [stat.ML]
[50] Weihua Hu, Matthias Fey, Hongyu Ren, Maho Nakata, Yuxiao Dong, and Jure Leskovec. 2021. OGB-LSC: A Large-Scale
Challenge for Machine Learning on Graphs. arXiv preprint arXiv:2103.09430 (2021).
[51] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure
Leskovec. 2020. Open Graph Benchmark: Datasets for Machine Learning on Graphs. CoRR abs/2005.00687 (2020).
arXiv:2005.00687 https://arxiv.org/abs/2005.00687
[52] Yuwei Hu, Zihao Ye, Minjie Wang, Jiali Yu, Da Zheng, Mu Li, Zheng Zhang, Zhiru Zhang, and Yida Wang. 2020.
Featgraph: A flexible and efficient backend for graph neural network systems. In SC20: International Conference for
High Performance Computing, Networking, Storage andAnalysis. IEEE, 1–13.
[53] Ziniu Hu, Yuxiao Dong, Kuansan Wang, and Yizhou Sun. 2020. Heterogeneous graph transformer. In Proceedings of
TheWeb Conference 2020. 2704–2710.
[54] Wenbing Huang, Tong Zhang, Yu Rong, and Junzhou Huang. 2018. Adaptive sampling towards fast graph representa-
tion learning. Advances inneural information processing systems 31 (2018).
[55] Zijie Huang, Zheng Li, Haoming Jiang, Tianyu Cao, Hanqing Lu, Bing Yin, Karthik Subbian, Yizhou Sun, and Wei
Wang. 2022. Multilingual Knowledge Graph Completion with Self-Supervised Adaptive Graph Alignment. In Annual
Meeting oftheAssociation forComputational Linguistics (ACL).
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 49

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 49
[56] Zijie Huang, Yizhou Sun, and Wei Wang. 2020. Learning Continuous System Dynamics from Irregularly-Sampled
Partial Observations. In Advances inNeural Information Processing Systems.
[57] Zijie Huang, Yizhou Sun, and Wei Wang. 2021. Coupled Graph ODE for Learning Interacting System Dynamics. In
Proceedings ofthe27th ACM SIGKDD Conference onKnowledge Discovery &Data Mining.
[58] Zengfeng Huang, Shengzhong Zhang, Chong Xi, Tang Liu, and Min Zhou. 2021. Scaling Up Graph Neural Networks
Via Graph Coarsening. In Proceedings ofthe27th ACM SIGKDD Conference onKnowledge Discovery amp; Data
Mining (Virtual Event, Singapore) (KDD ’21) . Association for Computing Machinery, New York, NY, USA, 675–684.
https://doi.org/10.1145/3447548.3467256
[59] Huawei. 2020. MindSpore. https://e.huawei.com/en/products/cloud-computing-dc/atlas/mindspore
[60] Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. 2018. Quantized Neural
Networks: Training Neural Networks with Low Precision Weights and Activations. Journal ofMachine Learning
Research 18, 187 (2018), 1–30. http://jmlr.org/papers/v18/16-456.html
[61] Amin Javari, Zhankui He, Zijie Huang, J. Jacob Durai Raj, and Kevin Chen-Chuan Chang. 2020. Weakly Supervised
Attention for Hashtag Recommendation using Graph Data. Proceedings ofTheWeb Conference 2020 (2020).
[62] Zhihao Jia, Sina Lin, Mingyu Gao, Matei Zaharia, and Alex Aiken. 2020. Improving the accuracy, scalability, and
performance of graph neural networks with roc. Proceedings ofMachine Learning andSystems 2 (2020), 187–198.
[63] Zhihao Jia, Sina Lin, Rex Ying, Jiaxuan You, Jure Leskovec, and Alex Aiken. 2020. Redundancy-free computation for
graph neural networks. In Proceedings ofthe26th ACM SIGKDD International Conference onKnowledge Discovery
&Data Mining. 997–1005.
[64] Dejun Jiang, Zhenxing Wu, Chang-Yu Hsieh, Guangyong Chen, Ben Liao, Zhe Wang, Chao Shen, Dongsheng Cao, Jian
Wu, and Tingjun Hou. 2021. Could graph neural networks learn better molecular representation for drug discovery?
A comparison study of descriptor-based and graph-based models. Journal ofcheminformatics 13, 1 (2021), 1–23.
[65] Weiwei Jiang and Jiayun Luo. 2022. Graph neural network for traffic forecasting: A survey. Expert Systems with
Applications (2022), 117921.
[66] Wei Jin, Xianfeng Tang, Haoming Jiang, Zheng Li, Danqing Zhang, Jiliang Tang, and Bing Yin. 2022. Condensing
Graphs via One-Step Gradient Matching. In Proceedings ofthe28th ACM SIGKDD Conference onKnowledge
Discovery andData Mining. 720–730.
[67] Wei Jin, Lingxiao Zhao, Shichang Zhang, Yozen Liu, Jiliang Tang, and Neil Shah. 2021. Graph Condensation for Graph
Neural Networks. arXiv preprint arXiv:2110.07580 (2021).
[68] Tim Kaler, Nickolas Stathas, Anne Ouyang, Alexandros-Stavros Iliopoulos, Tao Schardl, Charles E Leiserson, and
Jie Chen. 2022. Accelerating training and inference of graph neural networks with fast sampling and pipelining.
Proceedings ofMachine Learning andSystems 4 (2022), 172–189.
[69] George Karypis and Vipin Kumar. 1998. A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs.
SIAM J.Sci.Comput. 20, 1 (1998), 359–392. https://doi.org/10.1137/S1064827595287997
[70] Brian W Kernighan and Shen Lin. 1970. An efficient heuristic procedure for partitioning graphs. TheBellsystem
technical journal 49, 2 (1970), 291–307.
[71] Thomas N Kipf and Max Welling. 2016. Semi-supervised classification with graph convolutional networks. arXiv
preprint arXiv:1609.02907 (2016).
[72] Raghuraman Krishnamoorthi. 2018. Quantizing deep convolutional networks for efficient inference: A whitepaper.
arXiv preprint arXiv:1806.08342 (2018).
[73] Guohao Li, Matthias Müller, Bernard Ghanem, and Vladlen Koltun. 2021. Training graph neural networks with 1000
layers. In International conference onmachine learning. PMLR, 6437–6449.
[74] Jiayu Li, Tianyun Zhang, Hao Tian, Shengmin Jin, Makan Fardad, and Reza Zafarani. 2020. Sgcn: A graph sparsifier
based on graph convolutional networks. In Pacific-Asia Conference onKnowledge Discovery andData Mining .
Springer, 275–287.
[75] Shengwen Liang, Cheng Liu, Ying Wang, Huawei Li, and Xiaowei Li. 2020. Deepburning-gl: an automated framework
for generating graph neural network accelerators. In 2020 IEEE/ACM International Conference OnComputer Aided
Design (ICCAD). IEEE, 1–9.
[76] Shengwen Liang, Ying Wang, Cheng Liu, Lei He, LI Huawei, Dawen Xu, and Xiaowei Li. 2020. Engn: A high-
throughput and energy-efficient accelerator for large graph neural networks. IEEE Trans. Comput. 70, 9 (2020),
1511–1525.
[77] Haiyang Lin, Mingyu Yan, Xiaochun Ye, Dongrui Fan, Shirui Pan, Wenguang Chen, and Yuan Xie. 2022. A Compre-
hensive Survey on Distributed Training of Graph Neural Networks. arXiv preprint arXiv:2211.05368 (2022).
[78] Yujun Lin, Mengtian Yang, and Song Han. 2021. Naas: Neural accelerator architecture search. In 2021 58th ACM/IEEE
Design Automation Conference (DAC). IEEE, 1051–1056.
[79] Zhiqi Lin, Cheng Li, Youshan Miao, Yunxin Liu, and Yinlong Xu. 2020. Pagraph: Scaling gnn training on large graphs
via computation-aware caching. In Proceedings ofthe11th ACM Symposium onCloud Computing. 401–415.
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 50

50 Zhang, Shichang, et al.
[80] Tianfeng Liu, Yangrui Chen, Dan Li, Chuan Wu, Yibo Zhu, Jun He, Yanghua Peng, Hongzheng Chen, Hongzhi Chen,
and Chuanxiong Guo. 2021. Bgl: Gpu-efficient gnn training by optimizing graph data i/o and preprocessing. arXiv
preprint arXiv:2112.08541 (2021).
[81] Xin Liu, Mingyu Yan, Lei Deng, Guoqi Li, Xiaochun Ye, and Dongrui Fan. 2021. Sampling methods for efficient
training of graph convolutional networks: A survey. IEEE/CAA Journal ofAutomatica Sinica 9, 2 (2021), 205–234.
[82] Ziqi Liu, Zhengwei Wu, Zhiqiang Zhang, Jun Zhou, Shuang Yang, Le Song, and Yuan Qi. 2020. Bandit samplers for
training graph neural networks. Advances inNeural Information Processing Systems 33 (2020), 6878–6888.
[83] Qingqing Long, Lingjun Xu, Zheng Fang, and Guojie Song. 2021. HGK-GNN: Heterogeneous Graph Kernel Based
Graph Neural Networks. In Proceedings ofthe27th ACM SIGKDD Conference onKnowledge Discovery &amp; Data
Mining (Virtual Event, Singapore) (KDD ’21) . Association for Computing Machinery, New York, NY, USA, 1129–1138.
https://doi.org/10.1145/3447548.3467429
[84] Andreas Loukas. 2019. Graph Reduction with Spectral and Cut Guarantees. J.Mach. Learn. Res. 20, 116 (2019), 1–42.
[85] Andreas Loukas and Pierre Vandergheynst. 2018. Spectrally Approximating Large Graphs with Smaller Graphs. In
Proceedings ofthe35th International Conference onMachine Learning (Proceedings of Machine Learning Research,
Vol.80), Jennifer Dy and Andreas Krause (Eds.). PMLR, 3237–3246. https://proceedings.mlr.press/v80/loukas18a.html
[86] Lingxiao Ma, Zhi Yang, Youshan Miao, Jilong Xue, Ming Wu, Lidong Zhou, and Yafei Dai. 2019. {NeuGraph}: Parallel
Deep Neural Network Computation on Large Graphs. In 2019 USENIX Annual Technical Conference (USENIX ATC
19). 443–458.
[87] Lingxiao Ma, Zhi Yang, Youshan Miao, Jilong Xue, Ming Wu, Lidong Zhou, and Yafei Dai. 2019. {NeuGraph}: Parallel
Deep Neural Network Computation on Large Graphs. In 2019 USENIX Annual Technical Conference (USENIX ATC
19). 443–458.
[88] Christopher D Manning. 2009. Anintroduction toinformation retrieval. Cambridge university press.
[89] Kelong Mao, Jieming Zhu, Xi Xiao, Biao Lu, Zhaowei Wang, and Xiuqiang He. 2021. UltraGCN: Ultra Simplification
of Graph Convolutional Networks for Recommendation. In Proceedings ofthe30th ACM International Conference
onInformation &Knowledge Management. 1253–1262.
[90] Diego Marcheggiani and Ivan Titov. 2017. Encoding sentences with graph convolutional networks for semantic role
labeling. arXiv preprint arXiv:1703.04826 (2017).
[91] Azalia Mirhoseini, Anna Goldie, Mustafa Yazgan, Joe Wenjie Jiang, Ebrahim Songhori, Shen Wang, Young-Joon Lee,
Eric Johnson, Omkar Pathak, Azade Nazi, et al .2021. A graph placement methodology for fast chip design. Nature
594, 7862 (2021), 207–212.
[92] Alan Mislove, Massimiliano Marcon, Krishna P Gummadi, Peter Druschel, and Bobby Bhattacharjee. 2007. Mea-
surement and analysis of online social networks. In Proceedings ofthe7thACM SIGCOMM conference onInternet
measurement. 29–42.
[93] Jason Mohoney, Roger Waleffe, Henry Xu, Theodoros Rekatsinas, and Shivaram Venkataraman. 2021. Marius: Learning
Massive Graph Embeddings on a Single Machine. In 15th USENIX Symposium onOperating Systems Design and
Implementation (OSDI 21). USENIX Association, 533–549. https://www.usenix.org/conference/osdi21/presentation/
mohoney
[94] Hesham Mostafa. 2022. Sequential aggregation and rematerialization: Distributed full-batch training of graph neural
networks on large graphs. Proceedings ofMachine Learning andSystems 4 (2022), 265–275.
[95] Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko, Mart van Baalen, and Tijmen Blankevoort.
2021. A white paper on neural network quantization. arXiv preprint arXiv:2106.08295 (2021).
[96] Jingshu Peng, Zhao Chen, Yingxia Shao, Yanyan Shen, Lei Chen, and Jiannong Cao. 2022. Sancus: staleness-aware
communication-avoiding full-graph decentralized training in large-scale graph neural networks. Proceedings ofthe
VLDB Endowment 15, 9 (2022), 1937–1950.
[97] Jeffrey Pennington, Richard Socher, and Christopher D Manning. 2014. Glove: Global vectors for word representation.
InProceedings ofthe2014 conference onempirical methods innatural language processing (EMNLP). 1532–1543.
[98] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. Deepwalk: Online learning of social representations. In
Proceedings ofthe20th ACM SIGKDD international conference onKnowledge discovery anddata mining . 701–710.
[99] Md Khaledur Rahman, Majedul Haque Sujon, and Ariful Azad. 2021. Fusedmm: A unified sddmm-spmm kernel
for graph embedding and graph neural networks. In 2021 IEEE International Parallel andDistributed Processing
Symposium (IPDPS). IEEE, 256–266.
[100] Morteza Ramezani, Weilin Cong, Mehrdad Mahdavi, Mahmut Kandemir, and Anand Sivasubramaniam. 2022. Learn
Locally, Correct Globally: A Distributed Algorithm for Training Graph Neural Networks. In International Conference
onLearning Representations. https://openreview.net/forum?id=FndDxSz3LxQ
[101] Russell Reed. 1993. Pruning algorithms-a survey. IEEE transactions onNeural Networks 4, 5 (1993), 740–747.
[102] Kaspar Riesen and Horst Bunke. 2008. IAM graph database repository for graph based pattern recognition and
machine learning. In Joint IAPR International Workshops onStatistical Techniques inPattern Recognition (SPR) and
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 51

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 51
Structural andSyntactic Pattern Recognition (SSPR). Springer, 287–297.
[103] Yousef Saad. 2003. Iterative methods forsparse linear systems. SIAM.
[104] Hojjat Salehinejad, Sharan Sankar, Joseph Barfett, Errol Colak, and Shahrokh Valaee. 2017. Recent advances in
recurrent neural networks. arXiv preprint arXiv:1801.01078 (2017).
[105] Alvaro Sanchez-Gonzalez, Jonathan Godwin, Tobias Pfaff, Rex Ying, Jure Leskovec, and Peter Battaglia. 2020. Learning
to simulate complex physics with graph networks. In International Conference onMachine Learning. PMLR, 8459–
8468.
[106] Rishov Sarkar, Stefan Abi-Karam, Yuqi He, Lakshmi Sathidevi, and Cong Hao. 2022. FlowGNN: A Dataflow Architecture
for Universal Graph Neural Network Inference via Multi-Queue Streaming. arXiv preprint arXiv:2204.13103 (2022).
[107] Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, and Max Welling. 2018. Modeling
relational data with graph convolutional networks. In European semantic web conference. Springer, 593–607.
[108] Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and Tina Eliassi-Rad. 2008. Collective
classification in network data. AImagazine 29, 3 (2008), 93–93.
[109] Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and Tina Eliassi-Rad. 2008. Collective
classification in network data. AImagazine 29, 3 (2008), 93–93.
[110] Marco Serafini and Hui Guan. 2021. Scalable Graph Neural Network Training: The Case for Sampling. ACM SIGOPS
Operating Systems Review 55, 1 (2021), 68–76.
[111] M Ángeles Serrano, Marián Boguná, and Alessandro Vespignani. 2009. Extracting the multiscale backbone of complex
weighted networks. Proceedings ofthenational academy ofsciences 106, 16 (2009), 6483–6488.
[112] Yingxia Shao, Hongzheng Li, Xizhi Gu, Hongbo Yin, Yawen Li, Xupeng Miao, Wentao Zhang, Bin Cui, and Lei Chen.
2022. Distributed Graph Neural Network Training: A Survey. arXiv preprint arXiv:2211.00216 (2022).
[113] Atefeh Sohrabizadeh, Yunsheng Bai, Yizhou Sun, and Jason Cong. 2022. Automated Accelerator Optimization Aided
by Graph Neural Networks. In 2022 59th ACM/IEEE Design Automation Conference (DAC).
[114] Atefeh Sohrabizadeh, Yuze Chi, and Jason Cong. 2022. StreamGCN: Accelerating Graph Convolutional Networks
with Streaming Processing. In 2022 IEEE Custom Integrated Circuits Conference (CICC). IEEE, 1–8.
[115] Atefeh Sohrabizadeh, Jie Wang, and Jason Cong. 2020. End-to-End Optimization of Deep Learning Applications. In
FPGA. 133–139.
[116] Atefeh Sohrabizadeh, Cody Hao Yu, Min Gao, and Jason Cong. 2022. AutoDSE: Enabling Software Programmers to
Design Efficient FPGA Accelerators. ACM Transactions onDesign Automation ofElectronic Systems (TODAES) 27,
4 (2022), 1–27.
[117] Linghao Song, Yuze Chi, Atefeh Sohrabizadeh, Young-kyu Choi, Jason Lau, and Jason Cong. 2022. Sextans: A streaming
accelerator for general-purpose sparse-matrix dense-matrix multiplication. In Proceedings ofthe2022 ACM/SIGDA
International Symposium onField-Programmable Gate Arrays. 65–77.
[118] Daniel A Spielman and Nikhil Srivastava. 2011. Graph sparsification by effective resistances. SIAM J.Comput. 40, 6
(2011), 1913–1926.
[119] Daniel A Spielman and Shang-Hua Teng. 2014. Nearly linear time algorithms for preconditioning and solving
symmetric, diagonally dominant linear systems. SIAM J.Matrix Anal. Appl. 35, 3 (2014), 835–885.
[120] Nitish Srivastava, Hanchen Jin, Jie Liu, David Albonesi, and Zhiru Zhang. 2020. Matraptor: A sparse-sparse matrix
multiplication accelerator based on row-wise product. In 2020 53rd Annual IEEE/ACM International Symposium on
Microarchitecture (MICRO). IEEE, 766–780.
[121] Shyam A Tailor, Javier Fernandez-Marques, and Nicholas D Lane. 2020. Degree-quant: Quantization-aware training
for graph neural networks. arXiv preprint arXiv:2008.05000 (2020).
[122] John Thorpe, Yifan Qiao, Jonathan Eyolfson, Shen Teng, Guanzhou Hu, Zhihao Jia, Jinliang Wei, Keval Vora, Ravi Ne-
travali, Miryung Kim, et al .2021. Dorylus: Affordable, Scalable, and Accurate {GNN}Training with Distributed {CPU}
Servers and Serverless Threads. In 15th USENIX Symposium onOperating Systems Design andImplementation
(OSDI 21). 495–514.
[123] Chao Tian, Lingxiao Ma, Zhi Yang, and Yafei Dai. 2020. Pcgcn: Partition-centric processing for accelerating graph
convolutional network. In 2020 IEEE International Parallel andDistributed Processing Symposium (IPDPS) . IEEE,
936–945.
[124] Alok Tripathy, Katherine Yelick, and Aydın Buluç. 2020. Reducing communication in graph neural network training. In
SC20: International Conference forHigh Performance Computing, Networking, Storage andAnalysis. IEEE, 1–14.
[125] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. 2017. Graph
attention networks. arXiv preprint arXiv:1710.10903 (2017).
[126] Ulrike Von Luxburg. 2007. A tutorial on spectral clustering. Statistics andcomputing 17, 4 (2007), 395–416.
[127] Roger Waleffe, Jason Mohoney, Theodoros Rekatsinas, and Shivaram Venkataraman. 2022. MariusGNN: Resource-
Efficient Out-of-Core Training of Graph Neural Networks. arXiv preprint arXiv:2202.02365 (2022).
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 52

52 Zhang, Shichang, et al.
[128] Cheng Wan, Youjie Li, Ang Li, Nam Sung Kim, and Yingyan Lin. 2022. BNS-GCN: Efficient full-graph training of graph
convolutional networks with partition-parallelism and random boundary node sampling. Proceedings ofMachine
Learning andSystems 4 (2022).
[129] Cheng Wan, Youjie Li, Cameron R Wolfe, Anastasios Kyrillidis, Nam Sung Kim, and Yingyan Lin. 2022. PipeGCN:
Efficient full-graph training of graph convolutional networks with pipelined feature communication. arXiv preprint
arXiv:2203.10428 (2022).
[130] Jie Wang, Licheng Guo, and Jason Cong. 2021. Autosa: A polyhedral compiler for high-performance systolic arrays
on fpga. In The2021 ACM/SIGDA International Symposium onField-Programmable Gate Arrays. 93–104.
[131] Minjie Wang, Da Zheng, Zihao Ye, Quan Gan, Mufei Li, Xiang Song, Jinjing Zhou, Chao Ma, Lingfan Yu, Yu Gai,
Tianjun Xiao, Tong He, George Karypis, Jinyang Li, and Zheng Zhang. 2019. Deep Graph Library: A Graph-Centric,
Highly-Performant Package for Graph Neural Networks. arXiv preprint arXiv:1909.01315 (2019).
[132] Ruijie Wang, Zijie Huang, Shengzhong Liu, Huajie Shao, Dongxin Liu, Jinyang Li, Tianshi Wang, Dachun Sun,
Shuochao Yao, and Tarek Abdelzaher. 2021. Dydiff-vae: A dynamic variational framework for information diffu-
sion prediction. In Proceedings ofthe44th International ACM SIGIR Conference onResearch andDevelopment in
Information Retrieval. 163–172.
[133] Yuke Wang, Boyuan Feng, Gushu Li, Shuangchen Li, Lei Deng, Yuan Xie, and Yufei Ding. 2021. {GNNAdvisor}:
An Adaptive and Efficient Runtime System for {GNN}Acceleration on{GPUs}. In15th USENIX Symposium on
Operating Systems Design andImplementation (OSDI 21). 515–531.
[134] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E Sarma, Michael M Bronstein, and Justin M Solomon. 2019. Dynamic
graph cnn for learning on point clouds. Acm Transactions OnGraphics (tog) 38, 5 (2019), 1–12.
[135] Kun Wu, Mert Hidayetoğlu, Xiang Song, Sitao Huang, Da Zheng, Israt Nisa, and Wen-mei Hwu. 2023. PIGEON:
Optimizing CUDA Code Generator for End-to-End Training and Inference of Relational Graph Neural Networks.
arXiv preprint arXiv:2301.06284 (2023).
[136] Shiwen Wu, Fei Sun, Wentao Zhang, Xu Xie, and Bin Cui. 2020. Graph neural networks in recommender systems: a
survey. ACM Computing Surveys (CSUR) (2020).
[137] Yidi Wu, Kaihao Ma, Zhenkun Cai, Tatiana Jin, Boyang Li, Chenguang Zheng, James Cheng, and Fan Yu. 2021. Seastar:
vertex-centric programming for graph neural networks. In Proceedings oftheSixteenth European Conference on
Computer Systems. 359–375.
[138] Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Xiaojun Chang, and Chengqi Zhang. 2020. Connecting the
Dots: Multivariate Time Series Forecasting with Graph Neural Networks. In Proceedings ofthe26th ACM SIGKDD
International Conference onKnowledge Discovery &Data Mining.
[139] Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V Le. 2020. Self-training with noisy student improves imagenet
classification. In Proceedings oftheIEEE/CVF conference oncomputer vision andpattern recognition . 10687–10698.
[140] Zhiqiang Xie, Minjie Wang, Zihao Ye, Zheng Zhang, and Rui Fan. 2022. Graphiler: Optimizing Graph Neural Networks
with Message Passing Data Flow Graph. Proceedings ofMachine Learning andSystems 4 (2022), 515–528.
[141] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. 2018. How powerful are graph neural networks? arXiv
preprint arXiv:1810.00826 (2018).
[142] Zhongwen Xu, Hado P van Hasselt, and David Silver. 2018. Meta-gradient reinforcement learning. Advances in
neural information processing systems 31 (2018).
[143] Bencheng Yan, Chaokun Wang, Gaoyang Guo, and Yunkai Lou. 2020. TinyGNN: Learning Efficient Graph Neural
Networks. In Proceedings ofthe26th ACM SIGKDD International Conference onKnowledge Discovery &#38; Data
Mining (KDD). 1848–1856.
[144] Mingyu Yan, Lei Deng, Xing Hu, Ling Liang, Yujing Feng, Xiaochun Ye, Zhimin Zhang, Dongrui Fan, and Yuan Xie.
2020. Hygcn: A gcn accelerator with hybrid architecture. In 2020 IEEE International Symposium onHigh Performance
Computer Architecture (HPCA). IEEE, 15–29.
[145] Cheng Yang, Jiawei Liu, and Chuan Shi. 2021. Extract the knowledge of graph neural networks and go beyond it: An
effective knowledge distillation framework. In Proceedings oftheWeb Conference 2021. 1227–1237.
[146] Jianbang Yang, Dahai Tang, Xiaoniu Song, Lei Wang, Qiang Yin, Rong Chen, Wenyuan Yu, and Jingren Zhou. 2022.
GNNLab: a factored system for sample-based GNN training over GPUs. In Proceedings oftheSeventeenth European
Conference onComputer Systems. 417–434.
[147] Shuwen Yang, Ziyao Li, Guojie Song, and Lingsheng Cai. 2021. Deep Molecular Representation Learning via Fusing
Physical and Chemical Information. Advances inNeural Information Processing Systems 34 (2021), 16346–16357.
[148] Yiding Yang, Jiayan Qiu, Mingli Song, Dacheng Tao, and Xinchao Wang. 2020. Distilling knowledge from graph
convolutional networks. In Proceedings oftheIEEE/CVF Conference onComputer Vision andPattern Recognition .
7074–7083.
[149] Kai-Lang Yao and Wu-Jun Li. 2021. Blocking-based Neighbor Sampling for Large-scale Graph Neural Networks. In
International Joint Conference onArtificial Intelligence.
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 53

A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware 53
[150] Zihao Ye, Ruihang Lai, Junru Shao, Tianqi Chen, and Luis Ceze. 2022. SparseTIR: Composable Abstractions for Sparse
Compilation in Deep Learning. arXiv preprint arXiv:2207.04606 (2022).
[151] Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L Hamilton, and Jure Leskovec. 2018. Graph
convolutional neural networks for web-scale recommender systems. In Proceedings ofthe24th ACM SIGKDD
international conference onknowledge discovery &data mining. 974–983.
[152] Zhitao Ying, Jiaxuan You, Christopher Morris, Xiang Ren, Will Hamilton, and Jure Leskovec. 2018. Hierarchical graph
representation learning with differentiable pooling. Advances inneural information processing systems 31 (2018).
[153] Haoran You, Tong Geng, Yongan Zhang, Ang Li, and Yingyan Lin. 2022. Gcod: Graph convolutional network accelera-
tion via dedicated algorithm and accelerator co-design. In 2022 IEEE International Symposium onHigh-Performance
Computer Architecture (HPCA). IEEE, 460–474.
[154] Haoran You, Zhihan Lu, Zijian Zhou, Yonggan Fu, and Yingyan Lin. 2021. Early-Bird GCNs: Graph-Network Co-
Optimization Towards More Efficient GCN Training and Inference via Drawing Early-Bird Lottery Tickets. arXiv
preprint arXiv:2103.00794 (2021).
[155] Haiyang Yu, Limei Wang, Bokun Wang, Meng Liu, Tianbao Yang, and Shuiwang Ji. 2022. Graphfm: Improving large-
scale gnn training via feature momentum. In International Conference onMachine Learning. PMLR, 25684–25701.
[156] Hanqing Zeng and Viktor Prasanna. 2020. GraphACT: Accelerating GCN training on CPU-FPGA heterogeneous
platforms. In Proceedings ofthe2020 ACM/SIGDA International Symposium onField-Programmable Gate Arrays .
255–265.
[157] Hanqing Zeng, Muhan Zhang, Yinglong Xia, Ajitesh Srivastava, Andrey Malevich, Rajgopal Kannan, Viktor Prasanna,
Long Jin, and Ren Chen. 2021. Decoupling the Depth and Scope of Graph Neural Networks. Advances inNeural
Information Processing Systems 34 (2021).
[158] Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, Rajgopal Kannan, and Viktor Prasanna. 2019. Graphsaint: Graph
sampling based inductive learning method. arXiv preprint arXiv:1907.04931 (2019).
[159] Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, Rajgopal Kannan, and Viktor Prasanna. 2021. Accurate, efficient
and scalable training of Graph Neural Networks. J.Parallel andDistrib. Comput. 147 (2021), 166–183.
[160] Bingyi Zhang, Rajgopal Kannan, and Viktor Prasanna. 2021. BoostGCN: A framework for optimizing GCN inference
on FPGA. In 2021 IEEE 29th Annual International Symposium onField-Programmable Custom Computing Machines
(FCCM). IEEE, 29–39.
[161] Bingyi Zhang, Hanqing Zeng, and Viktor Prasanna. 2020. Hardware acceleration of large scale gcn inference. In 2020
IEEE 31st International Conference onApplication-specific Systems, Architectures andProcessors (ASAP) . IEEE,
61–68.
[162] Chuxu Zhang, Dongjin Song, Chao Huang, Ananthram Swami, and Nitesh V. Chawla. 2019. Heterogeneous Graph
Neural Network. In Proceedings ofthe25th ACM SIGKDD International Conference onKnowledge Discovery &
Data Mining (KDD ’19). 793–803.
[163] Dalong Zhang, Xin Huang, Ziqi Liu, Zhiyang Hu, Xianzheng Song, Zhibang Ge, Zhiqiang Zhang, Lin Wang, Jun
Zhou, Yang Shuang, et al .2020. Agl: a scalable system for industrial-purpose graph machine learning. arXiv preprint
arXiv:2003.02454 (2020).
[164] Hengrui Zhang, Zhongming Yu, Guohao Dai, Guyue Huang, Yufei Ding, Yuan Xie, and Yu Wang. 2022. Understanding
gnn computational graph: A coordinated computation, io, and memory perspective. Proceedings ofMachine Learning
andSystems 4 (2022), 467–484.
[165] Jiani Zhang, Xingjian Shi, Shenglin Zhao, and Irwin King. 2019. Star-gcn: Stacked and reconstructed graph convolu-
tional networks for recommender systems. arXiv preprint arXiv:1905.13129 (2019).
[166] Shichang Zhang, Yozen Liu, Yizhou Sun, and Neil Shah. 2022. Graph-less Neural Networks: Teaching Old MLPs New
Tricks Via Distillation. In International Conference onLearning Representations . https://openreview.net/forum?id=
4p6_5HBWPCw
[167] Xiaofan Zhang, Junsong Wang, Chao Zhu, Yonghua Lin, Jinjun Xiong, Wen-mei Hwu, and Deming Chen. 2018.
DNNBuilder: an automated tool for building high-performance DNN hardware accelerators for FPGAs. In ICCAD .
IEEE, 1–8.
[168] Yongan Zhang, Haoran You, Yonggan Fu, Tong Geng, Ang Li, and Yingyan Lin. 2021. G-CoS: Gnn-accelerator
co-search towards both better accuracy and efficiency. In 2021 IEEE/ACM International Conference OnComputer
Aided Design (ICCAD). IEEE, 1–9.
[169] Yiren Zhao, Duo Wang, Daniel Bates, Robert Mullins, Mateja Jamnik, and Pietro Lio. 2020. Learned low precision
graph neural networks. arXiv preprint arXiv:2009.09232 (2020).
[170] Cheng Zheng, Bo Zong, Wei Cheng, Dongjin Song, Jingchao Ni, Wenchao Yu, Haifeng Chen, and Wei Wang. 2020.
Robust graph representation learning via neural sparsification. In International Conference onMachine Learning .
PMLR, 11458–11468.
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.

## Page 54

54 Zhang, Shichang, et al.
[171] Da Zheng, Chao Ma, Minjie Wang, Jinjing Zhou, Qidong Su, Xiang Song, Quan Gan, Zheng Zhang, and George
Karypis. 2020. Distdgl: distributed graph neural network training for billion-scale graphs. In 2020 IEEE/ACM 10th
Workshop onIrregular Applications: Architectures andAlgorithms (IA3). IEEE, 36–44.
[172] Da Zheng, Xiang Song, Chengru Yang, Dominique LaSalle, Qidong Su, Minjie Wang, Chao Ma, and George Karypis.
2021. Distributed hybrid CPU and GPU training for graph neural networks on billion-scale graphs. arXiv preprint
arXiv:2112.15345 (2021).
[173] Wenqing Zheng, Edward W Huang, Nikhil Rao, Sumeet Katariya, Zhangyang Wang, and Karthik Subbian. 2021.
Cold Brew: Distilling Graph Node Representations with Incomplete or Missing Neighborhoods. arXiv preprint
arXiv:2111.04840 (2021).
[174] Hongkuan Zhou, Ajitesh Srivastava, Hanqing Zeng, Rajgopal Kannan, and Viktor Prasanna. 2021. Accelerating large
scale real-time GNN inference using channel pruning. arXiv preprint arXiv:2105.04528 (2021).
[175] Hongkuan Zhou, Da Zheng, Israt Nisa, Vasileios Ioannidis, Xiang Song, and George Karypis. 2022. TGL: A General
Framework for Temporal GNN Training on Billion-Scale Graphs. arXiv preprint arXiv:2203.14883 (2022).
[176] Zhe Zhou, Bizhao Shi, Zhe Zhang, Yijin Guan, Guangyu Sun, and Guojie Luo. 2021. BlockGNN: Towards Efficient
GNN Acceleration Using Block-Circulant Weight Matrices. In 2021 58th ACM/IEEE Design Automation Conference
(DAC). IEEE, 1009–1014.
[177] Rong Zhu, Kun Zhao, Hongxia Yang, Wei Lin, Chang Zhou, Baole Ai, Yong Li, and Jingren Zhou. 2019. Aligraph: a
comprehensive graph neural network platform. arXiv preprint arXiv:1902.08730 (2019).
[178] Marinka Zitnik and Jure Leskovec. 2017. Predicting multicellular function through multi-layer tissue networks.
Bioinformatics 33, 14 (2017), i190–i198.
[179] Difan Zou, Ziniu Hu, Yewen Wang, Song Jiang, Yizhou Sun, and Quanquan Gu. 2019. Layer-dependent importance
sampling for training deep and large graph convolutional networks. Advances inneural information processing
systems 32 (2019).
ACM Comput. Surv., Vol. 1, No. 1, Article . Publication date: June XXXX.