# Published as a conference paper at ICLR 2022

## Page 1

Published as a conference paper at ICLR 2022
HOWATTENTIVE ARE GRAPH ATTENTION
NETWORKS ?
Shaked Brody
Technion
shakedbr@cs.technion.ac.ilUri Alon
Language Technologies Institute
Carnegie Mellon University
ualon@cs.cmu.edu
Eran Yahav
Technion
yahave@cs.technion.ac.il
ABSTRACT
Graph Attention Networks (GATs) are one of the most popular GNN architectures
and are considered as the state-of-the-art architecture for representation learning
with graphs. In GAT, every node attends to its neighbors given its own represen-
tation as the query. However, in this paper we show that GAT computes a very
limited kind of attention: the ranking of the attention scores is unconditioned
on the query node . We formally deﬁne this restricted kind of attention as static
attention and distinguish it from a strictly more expressive dynamic attention. Be-
cause GATs use a static attention mechanism, there are simple graph problems
that GAT cannot express: in a controlled problem, we show that static attention
hinders GAT from even ﬁtting the training data. To remove this limitation, we
introduce a simple ﬁx by modifying the order of operations and propose GATv2:
adynamic graph attention variant that is strictly more expressive than GAT. We
perform an extensive evaluation and show that GATv2 outperforms GAT across 12
OGB and other benchmarks while we match their parametric costs. Our code is
available at https://github.com/tech-srl/how_attentive_are_
gats .1GATv2 is available as part of the PyTorch Geometric library,2the Deep
Graph Library,3and the TensorFlow GNN library.4
1 I NTRODUCTION
Graph neural networks (GNNs; Gori et al., 2005; Scarselli et al., 2008) have seen increasing popularity
over the past few years (Duvenaud et al., 2015; Atwood and Towsley, 2016; Bronstein et al., 2017;
Monti et al., 2017). GNNs provide a general and efﬁcient framework to learn from graph-structured
data. Thus, GNNs are easily applicable in domains where the data can be represented as a set of
nodes and the prediction depends on the relationships (edges) between the nodes. Such domains
include molecules, social networks, product recommendation, computer programs and more.
In a GNN, each node iteratively updates its state by interacting with its neighbors. GNN variants
(Wu et al., 2019; Xu et al., 2019; Li et al., 2016) mostly differ in how each node aggregates and
combines the representations of its neighbors with its own. Veli ˇckovi ´c et al. (2018) pioneered the use
of attention-based neighborhood aggregation, in one of the most common GNN variants – Graph
Attention Network (GAT). In GAT, every node updates its representation by attending to its neighbors
using its own representation as the query. This generalizes the standard averaging or max-pooling
of neighbors (Kipf and Welling, 2017; Hamilton et al., 2017), by allowing every node to compute
aweighted average of its neighbors, and (softly) select its most relevant neighbors. The work of
1An annotated implementation of GATv2 is available at https://nn.labml.ai/graphs/gatv2/
2from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
3from dgl.nn.pytorch import GATv2Conv
4from tensorflow_gnn.graph.keras.layers.gat_v2 import GATv2Convolution
1arXiv:2105.14491v3  [cs.LG]  31 Jan 2022

## Page 2

Published as a conference paper at ICLR 2022
k0k1k2k3k4k5k6k7k8k9
q0
q1
q2
q3
q4
q5
q6
q7
q8
q90.08 0.10 0.10 0.07 0.08 0.08 0.11 0.09 0.20 0.08
0.05 0.10 0.10 0.04 0.04 0.04 0.13 0.06 0.38 0.04
0.05 0.10 0.10 0.04 0.05 0.05 0.13 0.06 0.38 0.05
0.08 0.10 0.10 0.07 0.08 0.08 0.10 0.09 0.24 0.08
0.08 0.09 0.09 0.07 0.07 0.07 0.10 0.08 0.27 0.07
0.09 0.11 0.11 0.08 0.09 0.08 0.11 0.10 0.16 0.09
0.04 0.10 0.11 0.03 0.04 0.04 0.14 0.06 0.40 0.04
0.07 0.09 0.09 0.06 0.07 0.07 0.10 0.08 0.29 0.07
0.04 0.11 0.11 0.02 0.04 0.03 0.14 0.07 0.41 0.04
0.07 0.09 0.09 0.06 0.07 0.07 0.11 0.08 0.30 0.07
k0k1k2k3k4k5k6k7k8k90.10.20.30.4q0
q1
q2
q3
q4
q5
q6
q7
q8
q9
(a) Attention in standard GAT (Veli ˇckovi ´c et al. (2018))
k0k1k2k3k4k5k6k7k8k9
q0
q1
q2
q3
q4
q5
q6
q7
q8
q90.95 0.00 0.00 0.01 0.01 0.00 0.00 0.02 0.01 0.00
0.01 0.92 0.01 0.01 0.01 0.00 0.01 0.01 0.00 0.02
0.00 0.00 0.95 0.00 0.00 0.01 0.02 0.01 0.00 0.00
0.01 0.01 0.00 0.94 0.00 0.01 0.00 0.00 0.02 0.01
0.00 0.00 0.00 0.00 0.96 0.00 0.00 0.01 0.01 0.00
0.00 0.01 0.01 0.01 0.01 0.89 0.01 0.01 0.04 0.02
0.00 0.01 0.04 0.00 0.01 0.01 0.86 0.02 0.01 0.03
0.04 0.02 0.01 0.01 0.03 0.01 0.00 0.87 0.00 0.01
0.01 0.00 0.01 0.01 0.01 0.01 0.01 0.00 0.94 0.00
0.01 0.02 0.01 0.01 0.01 0.01 0.01 0.00 0.00 0.93
k0k1k2k3k4k5k6k7k8k90.00.20.40.60.81.0
q0
q1
q2
q3
q4
q5
q6
q7
q8
q9 (b) Attention in GATv2, our ﬁxed version of GAT
Figure 1: In a complete bipartite graph of “query nodes” fq0;:::;q 9gand “key nodes”fk0;:::;k 9g:
standard GAT (Figure 1a) computes static attention – the ranking of attention coefﬁcients is global
for all nodes in the graph, and is unconditioned on the query node. For example, all queries ( q0to
q9) attend mostly to the 8th key ( k8). In contrast, GATv2 (Figure 1b) can actually compute dynamic
attention, where every query has a different ranking of attention coefﬁcients of the keys.
Veliˇckovi ´c et al. also generalizes the Transformer’s (Vaswani et al., 2017) self-attention mechanism,
from sequences to graphs (Joshi, 2020).
Nowadays, GAT is one of the most popular GNN architectures (Bronstein et al., 2021) and is
considered as the state-of-the-art neural architecture for learning with graphs (Wang et al., 2019a).
Nevertheless, in this paper we show that GAT does not actually compute the expressive, well known,
type of attention (Bahdanau et al., 2014), which we call dynamic attention. Instead, we show that
GAT computes only a restricted “static” form of attention: for any query node, the attention function
ismonotonic with respect to the neighbor (key) scores. That is, the ranking (the argsort ) of attention
coefﬁcients is shared across all nodes in the graph, and is unconditioned on the query node. This fact
severely hurts the expressiveness of GAT, and is demonstrated in Figure 1a.
Supposedly, the conceptual idea of attention as the form of interaction between GNN nodes is
orthogonal to the speciﬁc choice of attention function. However, Veli ˇckovi ´c et al.’s original design of
GAT has spread to a variety of domains (Wang et al., 2019a; Yang et al., 2020; Wang et al., 2019c;
Huang and Carley, 2019; Ma et al., 2020; Kosaraju et al., 2019; Nathani et al., 2019; Wu et al., 2020;
Zhang et al., 2020) and has become the default implementation of “graph attention network” in all
popular GNN libraries such as PyTorch Geometric (Fey and Lenssen, 2019), DGL (Wang et al.,
2019b), and others (Dwivedi et al., 2020; Gordi ´c, 2020; Brockschmidt, 2020).
To overcome the limitation we identiﬁed in GAT, we introduce a simple ﬁx to its attention function
by only modifying the order of internal operations. The result is GATv2 – a graph attention variant
2

## Page 3

Published as a conference paper at ICLR 2022
that has a universal approximator attention function, and is thus strictly more expressive than GAT .
The effect of ﬁxing the attention function in GATv2 is demonstrated in Figure 1b.
In summary, our main contribution is identifying that one of the most popular GNN types, the graph
attention network, does not compute dynamic attention, the kind of attention that it seems to compute.
We introduce formal deﬁnitions for analyzing the expressive power of graph attention mechanisms
(Deﬁnitions 3.1 and 3.2), and derive our claims theoretically (Theorem 1) from the equations of
Veliˇckovi ´c et al. (2018). Empirically, we use a synthetic problem to show that standard GAT cannot
express problems that require dynamic attention (Section 4.1). We introduce a simple ﬁx by switching
the order of internal operations in GAT, and propose GATv2, which does compute dynamic attention
(Theorem 2). We further conduct a thorough empirical comparison of GAT and GATv2 and ﬁnd
that GATv2 outperforms GAT across 12 benchmarks of node-, link-, and graph-prediction. For
example, GATv2 outperforms extensively tuned GNNs by over 1.4% in the difﬁcult “UnseenProj
Test” set of the VarMisuse task (Allamanis et al., 2018), without any hyperparameter tuning; and
GATv2 improves over an extensively-tuned GAT by 11.5% in 13 prediction objectives in QM9. In
node-prediction benchmarks from OGB (Hu et al., 2020), not only that GATv2 outperforms GAT
with respect to accuracy – we ﬁnd that dynamic attention provided a much better robustness to noise.
2 P RELIMINARIES
A directed graphG= (V;E)contains nodesV=f1;:::;ngand edgesEVV , where (j;i)2E
denotes an edge from a node jto a nodei. We assume that every node i2 V has an initial
representation h(0)
i2Rd0. An undirected graph can be represented with bidirectional edges.
2.1 G RAPH NEURAL NETWORKS
A graph neural network (GNN) layer updates every node representation by aggregating its neighbors’
representations. A layer’s input is a set of node representations fhi2Rdji2Vg and the set of
edgesE. A layer outputs a new set of node representations fh0
i2Rd0ji2Vg , where the same
parametric function is applied to every node given its neighbors Ni=fj2Vj (j;i)2Eg :
h0
i=f(hi;AGGREGATE (fhjjj2Nig)) (1)
The design of fandAGGREGATE is what mostly distinguishes one type of GNN from the other. For
example, a common variant of GraphSAGE (Hamilton et al., 2017) performs an element-wise mean
asAGGREGATE , followed by concatenation with hi, a linear layer and a ReLU as f.
2.2 G RAPH ATTENTION NETWORKS
GraphSAGE and many other popular GNN architectures (Xu et al., 2019; Duvenaud et al., 2015)
weigh all neighbors j2Niwith equal importance (e.g., mean or max-pooling as AGGREGATE ). To
address this limitation, GAT (Veli ˇckovi ´c et al., 2018) instantiates Equation (1) by computing a learned
weighted average of the representations of Ni. A scoring function e:RdRd!Rcomputes a score
for every edge (j;i), which indicates the importance of the features of the neighbor jto the nodei:
e(hi;hj) = LeakyReLU 
a>[WhikWhj]
(2)
where a2R2d0,W2Rd0dare learned, andkdenotes vector concatenation. These attention scores
are normalized across all neighbors j2Niusing softmax, and the attention function is deﬁned as:
ij= softmax j(e(hi;hj)) =exp (e(hi;hj))P
j02Niexp (e(hi;hj0))(3)
Then, GAT computes a weighted average of the transformed features of the neighbor nodes (followed
by a nonlinearity ) as the new representation of i, using the normalized attention coefﬁcients:
h0
i=X
j2NiijWhj
(4)
From now on, we will refer to Equations (2) to (4) as the deﬁnition of GAT.
3

## Page 4

Published as a conference paper at ICLR 2022
3 T HEEXPRESSIVE POWER OF GRAPH ATTENTION MECHANISMS
In this section, we explain why attention is limited when it is not dynamic (Section 3.1). We then
show that GAT is severely constrained, because it can only compute static attention (Section 3.2).
Next, we show how GAT can be ﬁxed (Section 3.3), by simply modifying the order of operations.
We refer to a neural architecture (e.g., the scoring or the attention function of GAT) as a family of
functions , parameterized by the learned parameters. An element in the family is a concrete function
with speciﬁc trained weights. In the following, we use [n]to denote the set [n] =f1;2;:::;ngN.
3.1 T HEIMPORTANCE OF DYNAMIC WEIGHTING
Attention is a mechanism for computing a distribution over a set of input keyvectors, given an
additional query vector. If the attention function always weighs one key at least as much as any other
key,unconditioned on the query , we say that this attention function is static :
Deﬁnition 3.1 (Static attention) .A (possibly inﬁnite) family of scoring functions F  
RdRd!R
computes static scoring for a given set of key vectors K=fk1;:::;kngRdand
query vectors Q=fq1;:::;qmgRd, if for every f2F there exists a “highest scoring” key jf2[n]
such that for every query i2[m]and keyj2[n]it holds that f 
qi;kjf
f(qi;kj). We say
that a family of attention functions computes static attention givenKandQ, if its scoring function
computes static scoring, possibly followed by monotonic normalization such as softmax.
Static attention is very limited because every function f2F has a key that is always selected ,
regardless of the query. Such functions cannot model situations where different keys have different
relevance to different queries. Static attention is demonstrated in Figure 1a.
The general and powerful form of attention is dynamic attention :
Deﬁnition 3.2 (Dynamic attention) .A (possibly inﬁnite) family of scoring functions F  
RdRd!R
computes dynamic scoring for a given set of key vectors K=fk1;:::;kngRd
and query vectors Q=fq1;:::;qmgRd, if for anymapping': [m]![n]there existsf2F such
that for any query i2[m]and any key j6='(i)2[n]:f 
qi;k'(i)
>f(qi;kj). We say that a family
of attention functions computes dynamic attention forKandQ, if its scoring function computes
dynamic scoring, possibly followed by monotonic normalization such as softmax.
That is, dynamic attention can select every key'(i)using the query i, by making f 
qi;k'(i)
the
maximal inff(qi;kj)jj2[n]g. Note that dynamic andstatic attention are exclusive properties,
but they are not complementary. Further, every dynamic attention family has strict subsets of static
attention families with respect to the same KandQ. Dynamic attention is demonstrated in Figure 1b.
Attending by decaying Another way to think about attention is the ability to “focus” on the most
relevant inputs, given a query. Focusing is only possible by decaying other inputs, i.e., giving these
decayed inputs lower scores than others. If one key is always given an equal or greater attention score
than other keys (as in static attention), no query can ignore this key or decay this key’s score.
3.2 T HELIMITED EXPRESSIVITY OF GAT
Although the scoring function ecan be deﬁned in various ways, the original deﬁnition of Veli ˇckovi ´c
et al. (2018) (Equation (2)) has become the de facto practice: it has spread to a variety of domains and
is now the standard implementation of “graph attention network” in all popular GNN libraries (Fey
and Lenssen, 2019; Wang et al., 2019b; Dwivedi et al., 2020; Gordi ´c, 2020; Brockschmidt, 2020).
The motivation of GAT is to compute a representation for every node as a weighted average of its
neighbors. Statedly, GAT is inspired by the attention mechanism of Bahdanau et al. (2014) and the
self-attention mechanism of the Transformer (Vaswani et al., 2017). Nonetheless:
Theorem 1. A GAT layer computes only static attention, for any set of node representations K=
Q=fh1;:::;hng. In particular, for n>1, a GAT layer does not compute dynamic attention.
Proof. LetG= (V;E)be a graph modeled by a GAT layer with some aandWvalues (Equations (2)
and (3)), and having node representations fh1;:::;hng. The learned parameter acan be written as a
4

## Page 5

Published as a conference paper at ICLR 2022
concatenation a= [a1ka2]2R2d0such that a1;a22Rd0, and Equation (2) can be re-written as:
e(hi;hj) = LeakyReLU 
a>
1Whi+a>
2Whj
(5)
SinceVis ﬁnite, there exists a node jmax2V such that a>
2Whjmaxis maximal among all nodes
j2V (jmax is thejfrequired by Deﬁnition 3.1). Due to the monotonicity of LeakyReLU and
softmax , for every query node i2V, the nodejmax also leads to the maximal value of its attention
distributionfijjj2Vg . Thus, from Deﬁnition 3.1 directly, computes only static attention . This
also implies that does not compute dynamic attention, because in GAT, Deﬁnition 3.2 holds only
forconstant mappings'that map all inputs to the same output.
The consequence of Theorem 1 is that for any set of nodes Vand a trained GAT layer, the attention
functiondeﬁnes a constant ranking ( argsort ) of the nodes, unconditioned on the query nodes
i. That is, we can denote sj=a>
2Whjand get that for any choice of hi,is monotonic with
respect to the per-node scores fsjjj2Vg . This global ranking induces the local ranking of every
neighborhoodNi. The only effect of hiis in the “sharpness” of the produced attention distribution.
This is demonstrated in Figure 1a (bottom), where different curves denote different queries ( hi).
Generalization to multi-head attention Veliˇckovi ´c et al. (2018) found it beneﬁcial to employ H
separate attention heads and concatenate their outputs, similarly to Transformers. In this case,
Theorem 1 holds for each head separately: every head h2[H]has a (possibly different) node that
maximizesfs(h)
jjj2Vg , and the output is the concatenation of Hstatic attention heads.
3.3 B UILDING DYNAMIC GRAPH ATTENTION NETWORKS
To create a dynamic graph attention network, we modify the order of internal operations in GAT and
introduce GATv2 – a simple ﬁx of GAT that has a strictly more expressive attention mechanism.
GATv2 The main problem in the standard GAT scoring function (Equation (2)) is that the learned
layers Wandaare applied consecutively, and thus can be collapsed into a single linear layer. To ﬁx
this limitation, we simply apply the alayer after the nonlinearity ( LeakyReLU ), and the Wlayer
after the concatenation,5effectively applying an MLP to compute the score for each query-key pair:
GAT (Veli ˇckovi ´c et al., 2018): e(hi;hj) =LeakyReLU 
a>[WhikWhj]
(6)
GATv2 (our ﬁxed version): e(hi;hj) =a>LeakyReLU ( W[hikhj]) (7)
The simple modiﬁcation makes a signiﬁcant difference in the expressiveness of the attention function:
Theorem 2. A GATv2 layer computes dynamic attention for any set of node representations K=
Q=fh1;:::;hng.
We prove Theorem 2 in Appendix A. The main idea is that we can deﬁne an appropriate function
that GATv2 will be a universal approximator (Cybenko, 1989; Hornik, 1991) of. In contrast, GAT
(Equation (52)) cannot approximate any such desired function (Theorem 1).
Complexity GATv2 has the same time-complexity as GAT’s declared complexity: O(jVjdd0+jEjd0).
However, by merging its linear layers, GAT can be computed faster than stated by Veli ˇckovi ´c et al.
(2018). For a detailed time- and parametric-complexity analysis, see Appendix G.
4 E VALUATION
First, we demonstrate the weakness of GAT using a simple synthetic problem that GAT cannot even ﬁt
(cannot even achieve high training accuracy), but is easily solvable by GATv2 (Section 4.1). Second,
we show that GATv2 is much more robust to edge noise , because its dynamic attention mechanisms
allow it to decay noisy (false) edges, while GAT’s performance severely decreases as noise increases
(Section 4.2). Finally, we compare GAT and GATv2 across 12 benchmarks overall. (Sections 4.3
to 4.6 and appendix D.3). We ﬁnd that GAT is inferior to GATv2 across all examined benchmarks.
5We also add a bias vector bbefore applying the nonlinearity, we omit this in Equation (7) for brevity.
5

## Page 6

Published as a conference paper at ICLR 2022
A,4 B,3 C,2 D,1A, ? B, ? C, ? D, ?
Figure 2: The DICTIONARY -
LOOKUP problem of size k=4: ev-
ery node in the bottom row has an
alphabetic attribute (fA;B;C;:::g)
and a numeric value (f1;2;3;:::g);
every node in the upper row has
only an attribute; the goal is to pre-
dict the value for each node in the
upper row, using its attribute.45678910111213141516171819200102030405060708090100
k(number of different keys in each graph)AccuracyGATv2 test
GAT 8htest
GAT 1htrain
GAT 1htest
Figure 3: The DICTIONARY LOOKUP problem: GATv2 easily
achieves 100% train and test accuracies even for k=100 and
using only a single head.
Setup When previous results exist, we take hyperparameters that were tuned for GAT and use them
in GATv2, without any additional tuning. Self-supervision (Kim and Oh, 2021; Rong et al., 2020a),
graph regularization (Zhao and Akoglu, 2020; Rong et al., 2020b), and other tricks (Wang, 2021;
Huang et al., 2021) are orthogonal to the contribution of the GNN layer itself, and may further improve
all GNNs. In all experiments of GATv2, we constrain the learned matrix by setting W= [W0kW0],
to rule out the increased number of parameters over GAT as the source of empirical difference (see
Appendix G.2). Training details, statistics, and code are provided in Appendix B.
Our main goal is to compare dynamic and static graph attention mechanisms. However, for reference,
we also include non-attentive baselines such as GCN (Kipf and Welling, 2017), GIN (Xu et al., 2019)
and GraphSAGE (Hamilton et al., 2017). These non-attentive GNNs can be thought of as a special
case of attention, where every node gives all its neighbors the same attention score. Additional
comparison to a Transformer-style scaled dot-product attention (“DPGAT”), which is strictly weaker
than our proposed GATv2 (see a proof in Appendix E.1), is shown in Appendix E.
4.1 S YNTHETIC BENCHMARK : DICTIONARY LOOKUP
TheDICTIONARY LOOKUP problem is a contrived problem that we designed to test the ability of
a GNN architecture to perform dynamic attention. Here, we demonstrate that GAT cannot learn
this simple problem. Figure 2 shows a complete bipartite graph of 2knodes. Each “key node” in
the bottom row has an attribute (fA;B;C;:::g) and a value (f1;2;3;:::g). Each “query node” in
the upper row has only an attribute (fA;B;C;:::g). The goal is to predict the value of every query
node (upper row), according to its attribute. Each graph in the dataset has a different mapping from
attributes to values. We created a separate dataset for each k=f1;2;3;:::g, for which we trained a
different model, and measured per-node accuracy.
Although this is a contrived problem, it is relevant to any subgraph with keys that share more than
one query, and each query needs to attend to the keys differently. Such subgraphs are very common
in a variety of real-world domains. This problem tests the layer itself because it can be solved using a
single GNN layer, without suffering from multi-layer side-effects such as over-smoothing (Li et al.,
2018), over-squashing (Alon and Yahav, 2021), or vanishing gradients (Li et al., 2019). Our code
will be made publicly available, to serve as a testbed for future graph attention mechanisms.
Results Figure 3 shows the following surprising results: GAT with a single head (GAT 1h) failed to
ﬁt the training set for any value of k, no matter for how many iterations it was trained, and after
trying various training methods. Thus, it expectedly fails to generalize (resulting in low test accuracy).
Using 8 heads, GAT 8hsuccessfully ﬁts the training set, but generalizes poorly to the testset. In
contrast, GATv2 easily achieves 100% training and 100% test accuracies for any value of k, and even
fork=100 (not shown) and using a single head , thanks to its ability to perform dynamic attention.
These results clearly show the limitations of GAT, which are easily solved by GATv2. An additional
comparison to GIN, which could notﬁt this dataset, is provided in Figure 6 in Appendix D.1.
6

## Page 7

Published as a conference paper at ICLR 2022
Visualization Figure 1a (top) shows a heatmap of GAT’s attention scores in this DICTIONARY -
LOOKUP problem. As shown, all query nodes q0toq9attend mostly to the eighth key ( k8), and have
the same ranking of attention coefﬁcients (Figure 1a (bottom)). In contrast, Figure 1b shows how
GATv2 can select a different key node for every query node, because it computes dynamic attention.
The role of multi-head attention Veliˇckovi ´c et al. (2018) found the role of multi-head attention to
be stabilizing the learning process. Nevertheless, Figure 3 shows that increasing the number of heads
strictly increases training accuracy, and thus, the expressivity. Thus, GAT depends on having multiple
attention heads. In contrast, even a single GATv2 head generalizes better than a multi-head GAT.
4.2 R OBUSTNESS TO NOISE
We examine the robustness of dynamic andstatic attention to noise. In particular, we focus on
structural noise: given an input graph G= (V;E)and a noise ratio 0p1, we randomly sample
jEjpnon-existing edges E0fromVVnE . We then train the GNN on the noisy graph G0=(V;E[E0).
0 0:1 0:2 0:3 0:4 0:566687072
p– noise ratioAccuracyGATv2 (this work)
GAT
(a)ogbn-arxiv0 0:1 0:2 0:3 0:4 0:5283032
p– noise ratioAccuracyGATv2 (this work)
GAT
(b)ogbn-mag
Figure 4: Test accuracy compared to the noise ratio: GATv2 is more robust to structural noise
compared to GAT. Each point is an average of 10 runs, error bars show standard deviation.
Results Figure 9 shows the accuracy on two node-prediction datasets from the Open Graph Bench-
mark (OGB; Hu et al., 2020) as a function of the noise ratio p. Aspincreases, all models show
a natural decline in test accuracy in both datasets. Yet, thanks to their ability to compute dynamic
attention, GATv2 shows a milder degradation in accuracy compared to GAT, which shows a steeper
descent. We hypothesize that the ability to perform dynamic attention helps the models distinguishing
between given data edges ( E) and noise edges ( E0); in contrast, GAT cannot distinguish between
edges, because it scores the source and target nodes separately. These results clearly demonstrate the
robustness ofdynamic attention over static attention in noisy settings, which are common in reality.
4.3 P ROGRAMS : VARMISUSE
Setup VARMISUSE (Allamanis et al., 2018) is an inductive node-pointing problem that depends on
11 types of syntactic and semantic interactions between elements in computer programs.
We used the framework of Brockschmidt (2020), who performed an extensive hyperparameter tuning
by searching over 30 conﬁgurations for every GNN type. We took their best GAT hyperparameters
and used them to train GATv2, without further tuning.
Results As shown in Figure 5, GATv2 is more accurate than GAT and other GNNs in the SeenProj
test sets. Furthermore, GATv2 achieves an even higher improvement in the Unseen Proj test set.
Overall, these results demonstrate the power of GATv2 in modeling complex relational problems,
especially since it outperforms extensively tuned models, without any further tuning by us.
7

## Page 8

Published as a conference paper at ICLR 2022
Figure 5: Accuracy (5 runs stdev) on VARMISUSE . GATv2 is more accurate than all GNNs in both
test sets, using GAT’s hyperparameters. ypreviously reported by Brockschmidt (2020).
Model SeenProj UnseenProj
No-
AttentionGCNy87.21.5 81.42.3
GINy87.10.1 81.10.9
AttentionGATy86.90.7 81.20.9
GATv2 88.01.1 82.81.7
4.4 N ODE-PREDICTION
We further compare GATv2, GAT, and other GNNs on four node-prediction datasets from OGB.
Table 1: Average accuracy (Table 1a) and ROC-AUC (Table 1b) in node-prediction datasets (10
runsstd). In all datasets, GATv2 outperforms GAT. y– previously reported by Hu et al. (2020).
(a)
Model Attn. Heads ogbn-arxiv ogbn-products ogbn-mag
GCNy0 71.74 0.29 78.970.33 30.430.25
GraphSAGEy0 71.49 0.27 78.700.36 31.530.15
GAT1 71.59 0.38 79.041.54 32.201.46
8 71.54 0.30 77.232.37 31.751.60
GATv2 (this work)1 71.78 0.18 80.630.70 32.610.44
8 71.870.25 78.462.45 32.520.39(b)
ogbn-proteins
72.510.35
77.680.20
70.775.79
78.631.62
77.233.32
79.520.55
Results Results are shown in Table 1. In all settings and all datasets, GATv2 is more accurate than
GAT and the non-attentive GNNs. Interestingly, in the datasets of Table 1a, even a single head of
GATv2 outperforms GAT with 8 heads . In Table 1b ( ogbn-proteins ), increasing the number of heads
results in a major improvement for GAT (from 70.77 to 78.63), while GATv2 already gets most of
the beneﬁt using a single attention head. These results demonstrate the superiority of GATv2 over
GAT in node prediction (and even with a single head), thanks to GATv2’s dynamic attention.
4.5 G RAPH -PREDICTION : QM9
Setup In the QM9 dataset (Ramakrishnan et al., 2014; Gilmer et al., 2017), each graph is a molecule
and the goal is to regress each graph to 13 real-valued quantum chemical properties. We used the
implementation of Brockschmidt (2020) who performed an extensive hyperparameter search over
500 conﬁgurations; we took their best-found conﬁguration of GAT to implement GATv2.
Table 2: Average error rates (lower is better), 5 runs for each property, on the QM9 dataset. The best
result among GAT and GATv2 is marked in bold ; the globally best result among all GNNs is marked
inbold andunderline .ywas previously tuned and reported by Brockschmidt (2020).
Predicted Property Rel. to
Model 1 2 3 4 5 6 7 8 9 10 11 12 13 GAT
GCNy3.21 4.22 1.45 1.62 2.42 16.38 17.40 7.82 8.24 9.05 7.00 3.93 1.02 -1.5%
GINy2.64 4.67 1.42 1.50 2.27 15.63 12.93 5.88 18.71 5.62 5.38 3.53 1.05 -2.3%
GATy2.68 4.65 1.48 1.53 2.31 52.39 14.87 7.61 6.86 7.64 6.54 4.11 1.48 +0%
GATv2 2.65 4.28 1.41 1.47 2.29 16.37 14.03 6.07 6.28 6.60 5.97 3.57 1.59 -11.5%
Results Table 2 shows the main results: GATv2 achieves a lower (better) average error than GAT, by
11.5% relatively. GAT achieves the overall highest average error. In some properties, the non-attentive
8

## Page 9

Published as a conference paper at ICLR 2022
GNNs, GCN and GIN, perform best. We hypothesize that attention is not needed in modeling these
properties. Generally, GATv2 achieves the lowest overall average relative error (rightmost column).
4.6 L INK-PREDICTION
We compare GATv2, GAT, and other GNNs in link-prediction datasets from OGB.
Table 3: Average Hits@50 (Table 3a) and mean reciprocal rank (MRR) (Table 3b) in link-prediction
benchmarks from OGB (10 runs std). The best result among GAT and GATv2 is marked in bold ;
the best result among all GNNs is marked in bold andunderline .ywas reported by Hu et al. (2020).
(a)
ogbl-collab
Model Attn. Heads w/o val edges w/ val edges
No-
AttentionGCNy44.751.07 47.141.45
GraphSAGEy48.100.81 54.631.12
GATGAT 1h 39.323.26 48.104.80
GAT 8h 42.372.99 46.632.80
GATv2GATv2 1h 42.002.40 48.022.77
GATv2 8h 42.852.64 49.703.08(b)
ogbl-citation2
80.040.25
80.440.10
79.840.19
75.951.31
80.330.13
80.140.71
Results Table 3 shows that in all datasets, GATv2 achieves a higher MRR than GAT, which achieves
the lowest MRR. However, the non-attentive GraphSAGE performs better than all attentive GNNs.
We hypothesize that attention might not be needed in these datasets. Another possibility is that
dynamic attention is especially useful in graphs that have high node degrees : inogbn-products and
ogbn-proteins (Table 1) the average node degrees are 50.5 and 597, respectively (see Table 5 in
Appendix C). ogbl-collab andogbl-citation2 (Table 3), however, have much lower average node
degrees – of 8.2 and 20.7. We hypothesize that a dynamic attention mechanism is especially useful to
select the most relevant neighbors when the total number of neighbors is high. We leave the study of
the effect of the datasets’s average node degrees on the optimal GNN architecture for future work.
4.7 D ISCUSSION
Inallexamined benchmarks, we found that GATv2 is more accurate than GAT . Further, we found
that GATv2 is signiﬁcantly more robust to noise than GAT. In the synthetic DICTIONARY LOOKUP
benchmark (Section 4.1), GAT fails to express the data, and thus achieves even poor training accuracy.
In few of the benchmarks (Table 3 and some of the properties in Table 2) – a non-attentive model
such as GCN or GIN achieved a higher accuracy than all GNNs that do use attention.
Which graph attention mechanism should I use? It is usually impossible to determine in advance
which architecture would perform best. A theoretically weaker model may perform better in practice,
because a stronger model might overﬁt the training data if the task is “too simple” and does not
require such expressiveness. Intuitively, we believe that the more complex the interactions between
nodes are – the more beneﬁt a GNN can take from theoretically stronger graph attention mechanisms
such as GATv2. The main question is whether the problem has a global ranking of “inﬂuential” nodes
(GAT is sufﬁcient), or do different nodes have different rankings of neighbors (use GATv2).
Veliˇckovi ´c, the author of GAT, has conﬁrmed on Twitter6that GAT was designed to work in the
“easy-to-overﬁt” datasets of the time (2017), such as Cora, Citeseer and Pubmed (Sen et al., 2008),
where the data might had an underlying static ranking of “globally important” nodes. Veli ˇckovi ´c
agreed that newer and more challenging benchmarks may demand stronger attention mechanisms
such as GATv2. In this paper, we revisit the traditional assumptions and show that many modern graph
benchmarks and datasets contain more complex interactions, and thus require dynamic attention .
6https://twitter.com/PetarV_93/status/1399685979506675714
9

## Page 10

Published as a conference paper at ICLR 2022
5 R ELATED WORK
Attention in GNNs Modeling pairwise interactions between elements in graph-structured data goes
back to interaction networks (Battaglia et al., 2016; Hoshen, 2017) and relational networks (Santoro
et al., 2017). The GAT formulation of Veli ˇckovi ´c et al. (2018) rose as the most popular framework
for attentional GNNs, thanks to its simplicity, generality, and applicability beyond reinforcement
learning (Denil et al., 2017; Duan et al., 2017). Nevertheless, in this work, we show that the popular
and widespread deﬁnition of GAT is severely constrained to static attention only.
Other graph attention mechanisms Many works employed GNNs with attention mechanisms
other than the standard GAT’s (Zhang et al., 2018; Thekumparampil et al., 2018; Gao and Ji, 2019;
Lukovnikov and Fischer, 2021; Shi et al., 2020; Dwivedi and Bresson, 2020; Busbridge et al., 2019;
Rong et al., 2020a; Veli ˇckovi ´c et al., 2020), and Lee et al. (2018) conducted an extensive survey
of attention types in GNNs. However, none of these works identiﬁed the monotonicity of GAT’s
attention mechanism, the theoretical differences between attention types, nor empirically compared
their performance. Kim and Oh (2021) compared two graph attention mechanisms empirically, but in a
speciﬁc self-supervised scenario, without observing the theoretical difference in their expressiveness.
The static attention of GAT Qiu et al. (2018) recognized the order-preserving property of GAT, but
did not identify the severe theoretical constraint that this property implies: the inability to perform
dynamic attention (Theorem 1). Furthermore, they presented GAT’s monotonicity as a desired trait (!)
To the best of our knowledge, our work is the ﬁrst work to recognize the inability of GAT to perform
dynamic attention and its practical harmful consequences.
6 C ONCLUSION
In this paper, we identify that the popular and widespread Graph Attention Network does not compute
dynamic attention. Instead, the attention mechanism in the standard deﬁnition and implementations
of GAT is only static : for any query, its neighbor-scoring is monotonic with respect to per-node
scores. As a result, GAT cannot even express simple alignment problems. To address this limitation,
we introduce a simple ﬁx and propose GATv2: by modifying the order of operations in GAT, GATv2
achieves a universal approximator attention function and is thus strictly more powerful than GAT.
We demonstrate the empirical advantage of GATv2 over GAT in a synthetic problem that requires dy-
namic selection of nodes, and in 11 benchmarks from OGB and other public datasets. Our experiments
show that GATv2 outperforms GAT in all benchmarks while having the same parametric cost.
We encourage the community to use GATv2 instead of GAT whenever comparing new GNN ar-
chitectures to the common strong baselines. In complex tasks and domains and in challenging
datasets, a model that uses GAT as an internal component can replace it with GATv2 to bene-
ﬁt from a strictly more powerful model. To this end, we make our code publicly available at
https://github.com/tech-srl/how_attentive_are_gats , and GATv2 is available
as part of the PyTorch Geometric library, the Deep Graph Library, and TensorFlow GNN. An anno-
tated implementation is available at https://nn.labml.ai/graphs/gatv2/ .
ACKNOWLEDGMENTS
We thank Gail Weiss for the helpful discussions, thorough feedback, and inspirational paper (Weiss
et al., 2018). We also thank Petar Veli ˇckovi ´c for the useful discussion about the complexity and
implementation of GAT.
REFERENCES
Miltiadis Allamanis, Marc Brockschmidt, and Mahmoud Khademi. Learning to represent programs with graphs.
InInternational Conference on Learning Representations , 2018. URL https://openreview.net/
forum?id=BJOFETxR- .
10

## Page 11

Published as a conference paper at ICLR 2022
Uri Alon and Eran Yahav. On the bottleneck of graph neural networks and its practical implications. In Interna-
tional Conference on Learning Representations , 2021. URL https://openreview.net/forum?id=
i80OPhOCVH2 .
James Atwood and Don Towsley. Diffusion-convolutional neural networks. In Advances in neural information
processing systems , pages 1993–2001, 2016.
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to
align and translate. CoRR , abs/1409.0473, 2014. URL http://arxiv.org/abs/1409.0473 .
Peter Battaglia, Razvan Pascanu, Matthew Lai, Danilo Jimenez Rezende, and Koray kavukcuoglu. Interaction
networks for learning about objects, relations and physics. In Proceedings of the 30th International Conference
on Neural Information Processing Systems , pages 4509–4517, 2016.
Marc Brockschmidt. Gnn-ﬁlm: Graph neural networks with feature-wise linear modulation. Proceedings of
the 36th International Conference on Machine Learning, ICML , 2020. URL https://github.com/
microsoft/tf-gnn-samples .
Michael M Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, and Pierre Vandergheynst. Geometric deep
learning: going beyond euclidean data. IEEE Signal Processing Magazine , 34(4):18–42, 2017.
Michael M. Bronstein, Joan Bruna, Taco Cohen, and Petar Veli ˇckovi ´c. Geometric deep learning: Grids, groups,
graphs, geodesics, and gauges, 2021.
Dan Busbridge, Dane Sherburn, Pietro Cavallo, and Nils Y Hammerla. Relational graph attention networks.
arXiv preprint arXiv:1904.05811 , 2019.
George Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of control, signals
and systems , 2(4):303–314, 1989.
Misha Denil, Sergio Gómez Colmenarejo, Serkan Cabi, David Saxton, and Nando de Freitas. Programmable
agents. arXiv preprint arXiv:1706.06383 , 2017.
Yan Duan, Marcin Andrychowicz, Bradly Stadie, Jonathan Ho, Jonas Schneider, Ilya Sutskever, Pieter Abbeel,
and Wojciech Zaremba. One-shot imitation learning. In Proceedings of the 31st International Conference on
Neural Information Processing Systems , pages 1087–1098, 2017.
David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell, Timothy Hirzel, Alán Aspuru-
Guzik, and Ryan P Adams. Convolutional networks on graphs for learning molecular ﬁngerprints. In
Advances in neural information processing systems , pages 2224–2232, 2015.
Vijay Prakash Dwivedi and Xavier Bresson. A generalization of transformer networks to graphs. arXiv preprint
arXiv:2012.09699 , 2020.
Vijay Prakash Dwivedi, Chaitanya K Joshi, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Benchmarking
graph neural networks. arXiv preprint arXiv:2003.00982 , 2020.
Matthias Fey and Jan E. Lenssen. Fast graph representation learning with PyTorch Geometric. In ICLR Workshop
on Representation Learning on Graphs and Manifolds , 2019.
Ken-Ichi Funahashi. On the approximate realization of continuous mappings by neural networks. Neural
networks , 2(3):183–192, 1989.
Hongyang Gao and Shuiwang Ji. Graph representation learning via hard and channel-wise attention networks.
InProceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining ,
pages 741–749, 2019.
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing
for quantum chemistry. In Proceedings of the 34th International Conference on Machine Learning-Volume
70, pages 1263–1272. JMLR. org, 2017.
Aleksa Gordi ´c. pytorch-gat. https://github.com/gordicaleksa/pytorch-GAT , 2020.
Marco Gori, Gabriele Monfardini, and Franco Scarselli. A new model for learning in graph domains. In
Proceedings. 2005 IEEE International Joint Conference on Neural Networks, 2005. , volume 2, pages 729–
734. IEEE, 2005.
Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Advances
in neural information processing systems , pages 1024–1034, 2017.
11

## Page 12

Published as a conference paper at ICLR 2022
Kurt Hornik. Approximation capabilities of multilayer feedforward networks. Neural networks , 4(2):251–257,
1991.
Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal
approximators. Neural networks , 2(5):359–366, 1989.
Yedid Hoshen. Vain: attentional multi-agent predictive modeling. In Proceedings of the 31st International
Conference on Neural Information Processing Systems , pages 2698–2708, 2017.
Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure
Leskovec. Open graph benchmark: Datasets for machine learning on graphs. arXiv preprint arXiv:2005.00687 ,
2020.
Binxuan Huang and Kathleen M Carley. Syntax-aware aspect level sentiment classiﬁcation with graph attention
networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and
the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) , pages 5472–5480,
2019.
Qian Huang, Horace He, Abhay Singh, Ser-Nam Lim, and Austin Benson. Combining label propagation and
simple models out-performs graph neural networks. In International Conference on Learning Representations ,
2021. URL https://openreview.net/forum?id=8E1-f3VhX1o .
Chaitanya Joshi. Transformers are graph neural networks. The Gradient , 2020.
Dongkwan Kim and Alice Oh. How to ﬁnd your friendly neighborhood: Graph attention design with
self-supervision. In International Conference on Learning Representations , 2021. URL https://
openreview.net/forum?id=Wi5KUNlqWty .
Thomas N Kipf and Max Welling. Semi-supervised classiﬁcation with graph convolutional networks. In ICLR ,
2017.
Vineet Kosaraju, Amir Sadeghian, Roberto Martín-Martín, Ian Reid, Hamid Rezatoﬁghi, and Silvio Savarese.
Social-bigat: Multimodal trajectory forecasting using bicycle-gan and graph attention networks. In H. Wallach,
H. Larochelle, A. Beygelzimer, F. d Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Infor-
mation Processing Systems , volume 32. Curran Associates, Inc., 2019. URL https://proceedings.
neurips.cc/paper/2019/file/d09bf41544a3365a46c9077ebb5e35c3-Paper.pdf .
John Boaz Lee, Ryan A Rossi, Sungchul Kim, Nesreen K Ahmed, and Eunyee Koh. Attention models in graphs:
A survey. arXiv preprint arXiv:1807.07984 , 2018.
Moshe Leshno, Vladimir Ya Lin, Allan Pinkus, and Shimon Schocken. Multilayer feedforward networks with a
nonpolynomial activation function can approximate any function. Neural networks , 6(6):861–867, 1993.
Guohao Li, Matthias Muller, Ali Thabet, and Bernard Ghanem. Deepgcns: Can gcns go as deep as cnns? In
Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 9267–9276, 2019.
Qimai Li, Zhichao Han, and Xiao-Ming Wu. Deeper insights into graph convolutional networks for semi-
supervised learning. In Thirty-Second AAAI Conference on Artiﬁcial Intelligence , 2018.
Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard Zemel. Gated graph sequence neural networks. In
International Conference on Learning Representations , 2016.
Denis Lukovnikov and Asja Fischer. Gated relational graph attention networks, 2021. URL https://
openreview.net/forum?id=v-9E8egy_i .
Thang Luong, Hieu Pham, and Christopher D. Manning. Effective approaches to attention-based neural
machine translation. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language
Processing, EMNLP 2015, Lisbon, Portugal, September 17-21, 2015 , pages 1412–1421, 2015. URL http:
//aclweb.org/anthology/D/D15/D15-1166.pdf .
Nianzu Ma, Sahisnu Mazumder, Hao Wang, and Bing Liu. Entity-aware dependency-based deep graph attention
network for comparative preference classiﬁcation. In Proceedings of the 58th Annual Meeting of the
Association for Computational Linguistics , pages 5782–5788, 2020.
Federico Monti, Davide Boscaini, Jonathan Masci, Emanuele Rodola, Jan Svoboda, and Michael M Bronstein.
Geometric deep learning on graphs and manifolds using mixture model cnns. In Proceedings of the IEEE
conference on computer vision and pattern recognition , pages 5115–5124, 2017.
12

## Page 13

Published as a conference paper at ICLR 2022
Deepak Nathani, Jatin Chauhan, Charu Sharma, and Manohar Kaul. Learning attention-based embeddings for
relation prediction in knowledge graphs. In Proceedings of the 57th Annual Meeting of the Association for
Computational Linguistics , pages 4710–4723, 2019.
Sejun Park, Chulhee Yun, Jaeho Lee, and Jinwoo Shin. Minimum width for universal approximation. In Interna-
tional Conference on Learning Representations , 2021. URL https://openreview.net/forum?id=
O-XJwyoIF-k .
Allan Pinkus. Approximation theory of the mlp model. Acta Numerica 1999: Volume 8 , 8:143–195, 1999.
Jiezhong Qiu, Jian Tang, Hao Ma, Yuxiao Dong, Kuansan Wang, and Jie Tang. Deepinf: Social inﬂuence
prediction with deep learning. In Proceedings of the 24th ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining (KDD’18) , 2018.
Raghunathan Ramakrishnan, Pavlo O Dral, Matthias Rupp, and O Anatole V on Lilienfeld. Quantum chemistry
structures and properties of 134 kilo molecules. Scientiﬁc data , 1:140022, 2014.
Yu Rong, Yatao Bian, Tingyang Xu, Weiyang Xie, Ying Wei, Wenbing Huang, and Junzhou Huang. Self-
supervised graph transformer on large-scale molecular data. Advances in Neural Information Processing
Systems , 33, 2020a.
Yu Rong, Wenbing Huang, Tingyang Xu, and Junzhou Huang. Dropedge: Towards deep graph convolutional
networks on node classiﬁcation. In International Conference on Learning Representations , 2020b. URL
https://openreview.net/forum?id=Hkx1qkrKPr .
Adam Santoro, David Raposo, David GT Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, and
Timothy Lillicrap. A simple neural network module for relational reasoning. In Proceedings of the 31st
International Conference on Neural Information Processing Systems , pages 4974–4983, 2017.
Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph
neural network model. IEEE Transactions on Neural Networks , 20(1):61–80, 2008.
Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and Tina Eliassi-Rad. Collective
classiﬁcation in network data. AI magazine , 29(3):93–93, 2008.
Yunsheng Shi, Zhengjie Huang, Shikun Feng, and Yu Sun. Masked label prediction: Uniﬁed massage passing
model for semi-supervised classiﬁcation. arXiv preprint arXiv:2009.03509 , 2020.
Kiran K Thekumparampil, Chong Wang, Sewoong Oh, and Li-Jia Li. Attention-based graph neural network for
semi-supervised learning. arXiv preprint arXiv:1803.03735 , 2018.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser,
and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems , pages
6000–6010, 2017.
Petar Veli ˇckovi ´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. Graph
attention networks. In International Conference on Learning Representations , 2018.
Petar Veli ˇckovi ´c, Lars Buesing, Matthew Overlan, Razvan Pascanu, Oriol Vinyals, and Charles Blundell. Pointer
graph networks. Advances in Neural Information Processing Systems , 33, 2020.
Petar et al. Veli ˇckovi ´c. Graph attention networks. 2018.
Guangtao Wang, Rex Ying, Jing Huang, and Jure Leskovec. Improving graph attention networks with large
margin-based constraints. arXiv preprint arXiv:1910.11945 , 2019a.
Minjie Wang, Da Zheng, Zihao Ye, Quan Gan, Mufei Li, Xiang Song, Jinjing Zhou, Chao Ma, Lingfan Yu,
Yu Gai, Tianjun Xiao, Tong He, George Karypis, Jinyang Li, and Zheng Zhang. Deep graph library: A
graph-centric, highly-performant package for graph neural networks. arXiv preprint arXiv:1909.01315 ,
2019b.
Xiao Wang, Houye Ji, Chuan Shi, Bai Wang, Yanfang Ye, Peng Cui, and Philip S Yu. Heterogeneous graph
attention network. In The World Wide Web Conference , pages 2022–2032, 2019c.
Yangkun Wang. Bag of tricks of semi-supervised classiﬁcation with graph neural networks. arXiv preprint
arXiv:2103.13355 , 2021.
Gail Weiss, Yoav Goldberg, and Eran Yahav. On the practical computational power of ﬁnite precision rnns
for language recognition. In Proceedings of the 56th Annual Meeting of the Association for Computational
Linguistics (Volume 2: Short Papers) , pages 740–745, 2018.
13

## Page 14

Published as a conference paper at ICLR 2022
Felix Wu, Amauri Souza, Tianyi Zhang, Christopher Fifty, Tao Yu, and Kilian Weinberger. Simplifying graph
convolutional networks. In International conference on machine learning , pages 6861–6871. PMLR, 2019.
Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and S Yu Philip. A comprehensive
survey on graph neural networks. IEEE Transactions on Neural Networks and Learning Systems , 2020.
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks?
InInternational Conference on Learning Representations , 2019. URL https://openreview.net/
forum?id=ryGs6iA5Km .
Yiding Yang, Jiayan Qiu, Mingli Song, Dacheng Tao, and Xinchao Wang. Distilling knowledge from graph
convolutional networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR) , June 2020.
Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, Rajgopal Kannan, and Viktor Prasanna. Graphsaint: Graph
sampling based inductive learning method. arXiv preprint arXiv:1907.04931 , 2019.
Jiani Zhang, Xingjian Shi, Junyuan Xie, Hao Ma, Irwin King, and Dit-Yan Yeung. Gaan: Gated attention
networks for learning on large and spatiotemporal graphs. In Proceedings of the Thirty-Fourth Conference on
Uncertainty in Artiﬁcial Intelligence , pages 339–349, 2018.
Kai Zhang, Yaokang Zhu, Jun Wang, and Jie Zhang. Adaptive structural ﬁngerprints for graph attention networks.
InInternational Conference on Learning Representations , 2020. URL https://openreview.net/
forum?id=BJxWx0NYPr .
Lingxiao Zhao and Leman Akoglu. Pairnorm: Tackling oversmoothing in gnns. In International Conference on
Learning Representations , 2020. URL https://openreview.net/forum?id=rkecl1rtwB .
14

## Page 15

Published as a conference paper at ICLR 2022
A P ROOF FOR THEOREM 2
For brevity, we repeat our deﬁnition of dynamic attention (Deﬁnition 3.2):
Deﬁnition 3.2 (Dynamic attention). A (possibly inﬁnite) family of scoring functions F  
RdRd!R
computes dynamic scoring for a given set of key vectors K=fk1;:::;kngRd
and query vectors Q=fq1;:::;qmgRd, if for anymapping': [m]![n]there existsf2F such
that for any query i2[m]and any key j6='(i)2[n]:f 
qi;k'(i)
>f(qi;kj). We say that a family
of attention functions computes dynamic attention forKandQ, if its scoring function computes
dynamic scoring, possibly followed by monotonic normalization such as softmax.
Theorem 2. A GATv2 layer computes dynamic attention for any set of node representations K=
Q=fh1;:::;hng.
Proof. LetG= (V;E)be a graph modeled by a GATv2 layer, having node representations
fh1;:::;hng, and let': [n]![n]be any node mapping [n]![n]. We deﬁne g:R2d!R
as follows:
g(x) =19i:x=
hikh'(i)
0otherwise(8)
Next, we deﬁne a continues functioneg:R2d!Rthat equals to gin only speciﬁc n2inputs:
eg([hikhj]) =g([hikhj]);8i;j2[n] (9)
For all other inputs x2R2d,eg(x)realizes to any values that maintain the continuity of eg(this is
possible because we ﬁxed the values of egfor only a ﬁnite set of points).7
Thus, for every node i2V andj6='(i)2V:
1 =eg 
hikh'(i)
>eg([hikhj]) = 0 (10)
If we concatenate the two input vectors, and deﬁne the scoring function eof GATv2 (Equation (7)) as
a function of the concatenated vector [hikhj], from the universal approximation theorem (Hornik
et al., 1989; Cybenko, 1989; Funahashi, 1989; Hornik, 1991), ecan approximate egfor any compact
subset of R2d.
Thus, for any sufﬁciently small (any 0<< 1=2) there exist parameters Wandasuch that for
every nodei2V and everyj6='(i):
eW;a 
hi;h'(i)
>1 >0 +>eW;a(hi;hj) (11)
and due to the increasing monotonicity of softmax :
i;'(i)>i;j (12)
The choice of nonlinearity In general, these results hold if GATv2 had used any common non-
polynomial activation function (such as ReLU, sigmoid, or the hyperbolic tangent function). The
LeakyReLU activation function of GATv2 does not change its universal approximation ability (Leshno
et al., 1993; Pinkus, 1999; Park et al., 2021), and it was chosen only for consistency with the original
deﬁnition of GAT.
7The function egis a function that we deﬁne for the ease of proof, because the universal approximation
theorem is deﬁned for continuous functions, and we only need the scoring function of GATv2 eto approximate
the mapping 'in a ﬁnite set of points. So, we need the attention function eto approximate g(from Equation 8)
in some speciﬁc points. But, since gis not continuous, we deﬁne egand use the universal approximation theorem
foreg. Since eapproximates eg,ealso approximates gin our speciﬁc points, as a special case. We only require
thategwill be identical to gin speciﬁc n2points f[hikhj]ji; j2[n]g. For the rest of the input space, we don’t
have any requirement on the value of eg, except for maintaining the continuity of eg. There exist inﬁnitely many
such possible egfor every given set of keys, queries and a mapping ', but the concrete functions are not needed
for the proof.
15

## Page 16

Published as a conference paper at ICLR 2022
B T RAINING DETAILS
In this section we elaborate on the training details of all of our experiments. All models use residual
connections as in Veli ˇckovi ´c et al. (2018). All used code and data are publicly available under the
MIT license.
B.1 N ODE-AND LINK-PREDICTION
We used the provided splits of OGB (Hu et al., 2020) and the Adam optimizer. We tuned the
following hyperparameters: number of layers 2f2;3;6g, hidden size2f64;128;256g, learning rate
2f0:0005;0:001;0:005;0:01gand sampling method – full batch, GraphSAINT (Zeng et al., 2019)
and NeighborSampling (Hamilton et al., 2017). We tuned hyperparameters according to validation
score and early stopping. The ﬁnal hyperparameters are detailed in Table 4.
Dataset # layers Hidden size Learning rate Sampling method
ogbn-arxiv 3 256 0.01 GraphSAINT
ogbn-products 3 128 0.001 NeighborSampling
ogbn-mag 2 256 0.01 NeighborSampling
ogbn-proteins 6 64 0.01 NeighborSampling
ogbl-collab 3 64 0.001 Full Batch
ogbl-citation2 3 256 0.0005 NeighborSampling
Table 4: Training details of node- and link-prediction datasets.
B.2 R OBUSTNESS TO NOISE
In these experiments, we used the same best-found hyperparameters in node-prediction, with 8
attention heads in ogbn-arxiv and 1 head in ogbn-mag . Each point is an average of 10 runs.
B.3 S YNTHETIC BENCHMARK : DICTIONARY LOOKUP
In all experiments, we used a learning rate decay of 0:5, a hidden size of d= 128 , a batch size of
1024 , and the Adam optimizer.
We created a separate dataset for every graph size ( k), and we split each such dataset to train and
test with a ratio of 80:20. Since this is a contrived problem, we did not use a validation set, and the
reported test results can be thought of as validation results. Every model was trained on a ﬁxed value
ofk. Every key node (bottom row in Figure 2) was encoded as a sum of learned attribute embedding
and a value embedding, followed by ReLU.
We experimented with layer normalization, batch normalization, dropout, various activation functions
and various learning rates. None of these changed the general trend, so the experiments in Figure 3
were conducted without any normalization, without dropout and a learning rate of 0:001.
B.4 P ROGRAMS : VARMISUSE
We used the code, splits, and the same best-found conﬁgurations as Brockschmidt (2020), who
performed an extensive hyperparameter tuning by searching over 30 conﬁgurations for each GNN
type. We trained each model ﬁve times.
We took the best-found hyperparameters of Brockschmidt (2020) for GAT and used them to train
GATv2, without any further tuning.
B.5 G RAPH -PREDICTION : QM9
We used the code and splits of Brockschmidt (2020) who performed an extensive hyperparameter
search over 500 conﬁgurations. We took the best-found hyperparameters of Brockschmidt (2020)
16

## Page 17

Published as a conference paper at ICLR 2022
for GAT and used them to train GATv2. The only minor change from GAT is placing a residual
connection after every layer, rather than after every other layer, which is within the experimented
hyperparameter search that was reported by Brockschmidt (2020).
B.6 C OMPUTE AND RESOURCES
Our experiments consumed approximately 100 days of GPU in total. We used cloud GPUs of type
V100, and we used RTX 3080 and 3090 in local GPU machines.
C D ATA STATISTICS
C.1 N ODE-AND LINK-PREDICTION DATASETS
Statistics of the OGB datasets we used for node- and link-prediction are shown in Table 5.
Dataset # nodes # edges Avg. node degree Diameter
ogbn-arxiv 169,343 1,166,243 13.7 23
ogbn-mag 1,939,743 21,111,007 21.7 6
ogbn-products 2,449,029 61,859,140 50.5 27
ogbn-proteins 132,534 39,561,252 597.0 9
ogbl-collab 235,868 1,285,465 8.2 22
ogbl-citation2 2,927,963 30,561,187 20.7 21
Table 5: Statistics of the OGB datasets (Hu et al., 2020).
C.2 QM9
Statistics of the QM9 dataset, as used in Brockschmidt (2020) are shown in Table 6.
Training Validation Test
# examples 110,462 10,000 10,000
# nodes - average 18.03 18.06 18.09
# edges - average 18.65 18.67 18.72
Diameter - average 6.35 6.35 6.35
Table 6: Statistics of the QM9 chemical dataset (Ramakrishnan et al., 2014) as used by Brockschmidt
(2020).
C.3 V ARMISUSE
Statistics of the VARMISUSE dataset, as used in Allamanis et al. (2018) and Brockschmidt (2020),
are shown in Table 7.
Training Validation UnseenProject Test SeenProject Test
# graphs 254360 42654 117036 59974
# nodes - average 2377 1742 1959 3986
# edges - average 7298 7851 5882 12925
Diameter - average 7.88 7.88 7.78 7.82
Table 7: Statistics of the VARMISUSE dataset (Allamanis et al., 2018) as used by Brockschmidt
(2020).
17

## Page 18

Published as a conference paper at ICLR 2022
10 20 30 40 50 60 700102030405060708090100
k(number of different keys in each graph)AccuracyGATv2 1htrain
GATv2 1htest
GIN train
GIN test
Figure 6: Train and test accuracy across graph sizes in the DICTIONARY LOOKUP problem. GATv2
easily achieves 100% train and test accuracy even for k=100 and using only a single head. GIN
(Xu et al., 2019), although considered as more expressive than other GNNs, cannot perfectly ﬁt the
training data (with a model size of d= 128 ) starting from k=20.
D A DDITIONAL RESULTS
D.1 D ICTIONARY LOOKUP
Figure 6 shows additional comparison between GATv2 and GIN (Xu et al., 2019) in the DICTIO -
NARY LOOKUP problem. GATv2 easily achieves 100% train and test accuracy even for k=100 and
using only a single head. GIN, although considered as more expressive than other GNNs, cannot
perfectly ﬁt the training data (with a model size of d= 128 ) starting from k=20.
D.2 QM9
Standard deviation for the QM9 results of Section 4.5 are presented in Table 8.
Predicted Property
Model 1 2 3 4 5 6 7
GCNy3.210.06 4.220.45 1.450.01 1.620.04 2.420.14 16.380.49 17.403.56
GINy2.640.11 4.670.52 1.420.01 1.500.09 2.270.09 15.631.40 12.931.81
GAT 1h 3.080.08 7.821.42 1.790.10 3.961.51 3.581.03 35.4329.9 116.510.65
GAT 8hy2.680.06 4.650.44 1.480.03 1.530.07 2.310.06 52.3942.58 14.872.88
GATv2 1h 3.040.06 6.380.62 1.680.04 2.180.61 2.820.25 20.560.70 77.1337.93
GATv2 8h 2.650.05 4.280.27 1.410.04 1.470.03 2.290.15 16.370.97 14.031.39
Predicted Property Rel. to
Model 8 9 10 11 12 13 GAT 8h
GCNy7.820.80 8.241.25 9.051.21 7.001.51 3.930.48 1.020.05 -1.5%
GINy5.881.01 18.7123.36 5.620.81 5.380.75 3.530.37 1.050.11 -2.3%
GAT 1h 28.1016.45 20.8013.40 15.805.87 10.802.18 5.370.26 3.110.14 +134.1%
GAT 8hy7.610.46 6.860.53 7.640.92 6.540.36 4.110.27 1.480.87 +0%
GATv2 1h10.190.63 22.5617.46 15.044.58 22.9417.34 5.230.36 2.460.65 +91.6%
GATv2 8h 6.070.77 6.280.83 6.600.79 5.970.94 3.570.36 1.590.96 -11.5 %
Table 8: Average error rates (lower is better), 5 runs standard deviation for each property, on
the QM9 dataset. The best result among GAT and GATv2 is marked in bold; the globally best
result among all GNNs is marked in bold andunderline .ywas previously tuned and reported by
Brockschmidt (2020).
18

## Page 19

Published as a conference paper at ICLR 2022
D.3 P UBMED CITATION NETWORK
We tuned the following parameters for both GAT and GATv2: number of layers 2f0;1;2g, hidden
size2f8;16;32g, number of heads 2f1;4;8g, dropout2f0:4;0:6;0:8g, bias2fTrue;Falseg,
share weights2fTrue;Falseg, use residual2fTrue;Falseg. Table 9 shows the test accuracy
(100 runsstdev) using the best hyperparameters found for each model.
Table 9: Accuracy (100 runs stdev) on Pubmed. GATv2 is more accurate than GAT.
Model Accuracy
GAT 78.1 0.59
GATv2 78.50.38
It is important to note that PubMed has only 60 training nodes , which hinders expressive models
such as GATv2 from exploiting their approximation and generalization advantages. Still, GATv2
is more accurate than GAT even in this small dataset. In Table 14, we show that this difference is
statistically signiﬁcant (p-value <0:0001 ).
E A DDITIONAL COMPARISON WITH TRANSFORMER -STYLE ATTENTION
(DPGAT)
The main goal of our paper is to highlight a severe theoretical limitation of the highly popular GAT
architecture, and propose a minimal ﬁx.
We perform additional empirical comparison to DPGAT, which follows Luong et al. (2015) and the
dot-product attention of the Transformer (Vaswani et al., 2017). We deﬁne DPGAT as:
DPGAT (Vaswani et al., 2017): e(hi;hj) = 
h>
iQ
 
h>
jK>
=p
dk (13)
Variants of DPGAT were used in prior work (Gao and Ji, 2019; Dwivedi and Bresson, 2020; Rong
et al., 2020a; Veli ˇckovi ´c et al., 2020; Kim and Oh, 2021), and we consider it here for the conceptual
and empirical comparison with GAT.
Despite its popularity, DPGAT is strictly weaker than GATv2. DPGAT provably performs dynamic
attention for any set of node representations only if they are linearly independent (see Theorem 3
and its proof in Appendix E.1). Otherwise, there are examples of node representations that are
linearly dependent and mappings ', for which dynamic attention does not hold (Appendix E.2).
This constraint is not harmful when violated in practice, because every node has only a small set of
neighbors, rather than all possible nodes in the graph; further, some nodes possibly never need to be
“selected” in practice.
E.1 P ROOF THAT DPGAT P ERFORMS DYNAMIC ATTENTION FOR LINEARLY INDEPENDENT
NODE REPRESENTATIONS
Theorem 3. A DPGAT layer computes dynamic attention for any set of node representations K=
Q=fh1;:::;hngthat are linearly independent.
Proof. LetG= (V;E)be a graph modeled by a DPGAT layer, having linearly independent node
representationsfh1;:::;hng. Let': [n]![n]be any node mapping [n]![n].
We denote the ithrow of a matrix MasMi.
We deﬁne a matrix Pas:
Pi;j=1j='(i)
0otherwise(14)
LetX2RnRdbe the matrix holding the graph’s node representations as its rows:
19

## Page 20

Published as a conference paper at ICLR 2022
X=2
664—h1—
—h2—
...
—hn—3
775(15)
Since the rows of Xare linearly independent, it necessarily holds that dn.
Next, we ﬁnd weight matrices Q2RdRdandK2RdRdsuch that:
(XQ)(XK )>=P (16)
To satisfy Equation (16), we choose QandKsuch that XQ =UandXK =P>Uwhere Uis
an orthonormal matrix ( UU>=U>U=I).
We can obtain Uusing the singular value decomposition (SVD) of X:
X=UV>(17)
Since2RnRnandXhas a full rank, is invertible, and thus:
XV 1=U (18)
Now, we deﬁne Qas follows:
Q=V 1(19)
Note that XQ =U, as desired.
To ﬁnd Kthat satisﬁes XK =P>U, we use Equation (17) and require:
UV>K=P>U (20)
and thus:
K=V 1UTP>U (21)
We deﬁne:
z(hi;hj) =e(hi;hj)p
dk (22)
Whereeis the attention score function of DPGAT (Equation (13)).
Now, for a query iand a keyj, and the corresponding representations hi;hj:
z(hi;hj) = 
h>
iQ
 
h>
jK>(23)
= (XiQ)(XjK)>(24)
SinceXiQ= (XQ)iandXjK= (XK )j, we get
z(hi;hj) = (XQ)i
(XK )j>
=Pi;j (25)
Therefore:
z(hi;hj) =1j='(i)
0otherwise(26)
And thus:
e(hi;hj) =1=pdkj='(i)
0otherwise(27)
To conclude, for every selected query iand any key j6='(i):
e 
hi;h'(i)
>e(hi;hj) (28)
and due to the increasing monotonicity of softmax :
i;'(i)>i;j (29)
20

## Page 21

Published as a conference paper at ICLR 2022
Hence, a DPGAT layer computes dynamic attention.
In the case that d>n , we apply SVD to the full-rank matrix XX>2Rnn, and follow the same
steps to construct QandK.
In the case that Q2RdRdkandK2RdRdkanddk> d, we can use the same QandK
(Equations (19) and (21)) padded with zeros. We deﬁne the Q02RdRdkeyandK02RdRdkey
as follows:
Q0
i;j=Qi;jjd
0 otherwise(30)
K0
i;j=Ki;jjd
0 otherwise(31)
E.2 DPGAT IS STRICTLY WEAKER THAN GAT V2
There are examples of node representations that are linearly dependent and mappings ', for which
dynamic attention does not hold. First, we show a simple 2-dimensional example, and then we show
the general case of such examples.
xy
h0=^xh1=^x+^yh2=^x+ 2^y
Figure 7: An example for node representations that are linearly dependent, for which DPGAT cannot
compute dynamic attention, because no query vector q2R2can “select” h1.
Consider the following linearly dependent set of vectors K=Q(Figure 7):
h0=^x (32)
h1=^x+^y (33)
h2=^x+ 2^y (34)
where ^xand^yare the cartesian unit vectors. We deﬁne 2f0;1;2gto expressfh0;h1;h2gusing
the same expression:
h=^x+^y (35)
Letq2Qbe any query vector. For brevity, we deﬁne the unscaled dot-product attention score as s:
s(q;h) =e(q;h)p
dk (36)
Whereeis the attention score function of DPGAT (Equation (13)). The (unscaled) attention score
between qandfh0;h1;h2gis:
s(q;h) = 
q>Q 
h>
K>(37)
= 
q>Q
(^x+^y)>K>
(38)
= 
q>Q ^x>K+^y>K>(39)
= 
q>Q ^x>K>+ 
q>Q ^y>K>(40)
The ﬁrst term 
q>Q ^x>K>is unconditioned on , and thus shared for every h. Let us focus
on the second term  
q>Q ^y>K>. If 
q>Q ^y>K>>0, then:
e(q;h2)>e(q;h1) (41)
21

## Page 22

Published as a conference paper at ICLR 2022
Otherwise, if 
q>Q ^y>K>0:
e(q;h0)e(q;h1) (42)
Thus, for any query q, the key h1can never get the highest score, and thus cannot be “selected”. That
is, the key h1cannot satisfy that e(q;h1)is strictly greater than any other key.
In the general case, let h0;h12Rdbe some non-zero vectors , and is some scalar such that
0<< 1.
Consider the following linearly dependent set of vectors:
K=Q=fh1+ (1 )h0j2f0;;1gg (43)
For any query q2Qand2f0;;1gwe deﬁne:
s(q;) =e(q;(h1+ (1 )h0))p
dk (44)
Whereeis the attention score function of DPGAT (Equation (13)).
Therefore:
s(q;) = 
q>Q
(h1+ (1 )h0)>K>
(45)
= 
q>Q 
h>
1K+ (1 )h>
0K>(46)
= 
q>Q 
h>
1K+h>
0K h>
0K>(47)
= 
q>Q 
 
h>
1K h>
0K
+h>
0K>(48)
= 
q>Q 
h>
1K h>
0K>+ 
q>Q 
h>
0K>(49)
If 
q>Q 
h>
1K h>
0K>>0:
e(q;h1)>e(q;h) (50)
Otherwise, if 
q>Q 
h>
1K h>
0K>0:
e(q;h0)e(q;h) (51)
Thus, for any query q, the key hcannot be selected. That is, the key hcannot satisfy that e(q;h)
is strictly greater than any other key. Therefore, there are mappings ', for which dynamic attention
does not hold.
While we prove that GATv2 computes dynamic attention (Appendix A) for anyset of node represen-
tations K=Q, there are sets of node representations and mappings 'for which dynamic attention
does not hold for DPGAT. Thus, DPGAT is strictly weaker than GATv2.
E.3 E MPIRICAL EVALUATION
Here we repeat the experiments of Section 4 with DPGAT. We remind that DPGAT is strictly weaker
than our proposed GATv2 (see a proof in Appendix E.1).
F S TATISTICAL SIGNIFICANCE
Here we report the statistical signiﬁcance of the strongest GATv2 and GAT models of the experiments
reported in Section 4.
22

## Page 23

Published as a conference paper at ICLR 2022
0 0:1 0:2 0:3 0:4 0:566687072
p– noise ratioAccuracyDPGAT
GATv2
GAT
(a)ogbn-arxiv0 0:1 0:2 0:3 0:4 0:5283032
p– noise ratioAccuracyDPGAT
GATv2
GAT
(b)ogbn-mag
Figure 8: Test accuracy compared to the noise ratio: GATv2 and DPGAT are more robust to structural
noise compared to GAT. Each point is an average of 10 runs, error bars show standard deviation.
Table 10: Accuracy (5 runs stdev) on VARMISUSE . GATv2 is more accurate than all GNNs in both
test sets, using GAT’s hyperparameters. y– previously reported by Brockschmidt (2020).
Model SeenProj UnseenProj
No-
AttentionGCNy87.21.5 81.42.3
GINy87.10.1 81.10.9
AttentionGATy86.90.7 81.20.9
DPGAT 88.00.8 81.51.2
GATv2 88.01.1 82.81.7
Table 11: Average accuracy (Table 11a) and ROC-AUC (Table 11b) in node-prediction datasets (10
runsstd). In all datasets, GATv2 outperforms GAT. y– previously reported by Hu et al. (2020).
(a)
Model Attn. Heads ogbn-arxiv ogbn-products ogbn-mag
GCNy0 71.74 0.29 78.970.33 30.430.25
GraphSAGEy0 71.49 0.27 78.700.36 31.530.15
GAT1 71.59 0.38 79.041.54 32.201.46
8 71.54 0.30 77.232.37 31.751.60
DPGAT1 71.52 0.17 76.490.78 32.770.80
8 71.48 0.26 73.530.47 27.749.97
GATv2 (this work)1 71.78 0.18 80.630.70 32.610.44
8 71.870.25 78.462.45 32.520.39(b)
ogbn-proteins
72.510.35
77.680.20
70.775.79
78.631.62
63.472.79
72.880.59
77.233.32
79.520.55
23

## Page 24

Published as a conference paper at ICLR 2022
Table 12: Average error rates (lower is better), 5 runs standard deviation for each property, on the
QM9 dataset. The best result among GAT, GATv2 and DPGAT is marked in bold ; the globally best
result among all GNNs is marked in bold andunderline .ywas previously tuned and reported by
Brockschmidt (2020).
Predicted Property
Model 1 2 3 4 5 6 7
GCNy3.210.06 4.220.45 1.450.01 1.620.04 2.420.14 16.380.49 17.403.56
GINy2.640.11 4.670.52 1.420.01 1.500.09 2.270.09 15.631.40 12.931.81
GAT 1h 3.080.08 7.821.42 1.790.10 3.961.51 3.581.03 35.4329.9 116.510.65
GAT 8hy2.680.06 4.650.44 1.480.03 1.530.07 2.310.06 52.3942.58 14.872.88
DPGAT 8h 2.630.09 4.370.13 1.440.07 1.400.03 2.100.07 32.5934.77 11.661.00
DPGAT 1h 3.200.17 8.350.78 1.710.03 2.170.14 2.880.12 25.212.86 65.7939.84
GATv2 1h 3.040.06 6.380.62 1.680.04 2.180.61 2.820.25 20.560.70 77.1337.93
GATv2 8h 2.650.05 4.280.27 1.410.04 1.470.03 2.290.15 16.370.97 14.031.39
Predicted Property Rel. to
Model 8 9 10 11 12 13 GAT 8h
GCNy7.820.80 8.241.25 9.051.21 7.001.51 3.930.48 1.020.05 -1.5%
GINy5.881.01 18.7123.36 5.620.81 5.380.75 3.530.37 1.050.11 -2.3%
GAT 1h 28.1016.45 20.8013.40 15.805.87 10.802.18 5.370.26 3.110.14 +134.1%
GAT 8hy7.610.46 6.860.53 7.640.92 6.540.36 4.110.27 1.480.87 +0%
DPGAT 1h12.931.70 13.322.39 14.421.95 13.832.55 6.370.28 3.281.16 +77.9%
DPGAT 8h 6.950.32 7.090.59 7.300.66 6.520.61 3.760.21 1.180.33 -9.7%
GATv2 1h 10.190.63 22.5617.46 15.044.58 22.9417.34 5.230.36 2.460.65 +91.6%
GATv2 8h 6.070.77 6.280.83 6.600.79 5.970.94 3.570.36 1.590.96 -11.5 %
0 0:1 0:2 0:3 0:4 0:566687072
0.0002
<0.0001<0.0001
0.0001<0.0001
<0.0001p-value<0.0001
<0.0001
noise ratioAccuracyGATv2 (p-value)
GAT
(a)ogbn-arxiv0 0:1 0:2 0:3 0:4 0:5283032 0.0006
0.0001p-value<0.0001
<0.0001<0.0001 <0.0001<0.0001
<0.0001
noise ratioAccuracyGATv2 (p-value)
GAT
(b)ogbn-mag
Figure 9: Test accuracy and statistical signiﬁcance compared to the noise ratio: GATv2 is more robust
to structural noise compared to GAT. Each point is an average of 10 runs, error bars show standard
deviation.
Table 13: Accuracy (5 runs stdev) on VARMISUSE . GATv2 is more accurate than all GNNs in both
test sets, using GAT’s hyperparameters. y– previously reported by Brockschmidt (2020).
Model SeenProj UnseenProj
GATy86.90.7 81.20.9
GATv2 88.01.1 82.81.7
p-value 0.048 0.049
24

## Page 25

Published as a conference paper at ICLR 2022
Table 14: Accuracy (100 runs stdev) on Pubmed. GATv2 is more accurate than GAT.
Model Accuracy
GAT 78.1 0.59
GATv2 78.50.38
p-value < 0.0001
Table 15: Average accuracy (Table 15a) and ROC-AUC (Table 15b) in node-prediction datasets (30
runsstd). We report on the best GAT / GATv2 from Table 1.
(a)
Model ogbn-arxiv ogbn-products ogbn-mag
GAT 71.65 0.38 79.041.54 32.361.10
GATv2 71.930.35 80.630.70 33.010.41
p-value 0.0022 <0.0001 0.0018(b)
ogbn-proteins
78.291.59
78.961.19
0.0349
Table 16: Average Hits@50 (Table 16a) and mean reciprocal rank (MRR) (Table 16b) in link-
prediction benchmarks from OGB (30 runs std). We report on the best GAT / GATv2 from Table 3.
(a)
ogbl-collab
Model w/o val edges w/ val edges
GAT 42.24 2.26 46.024.09
GATv2 43.822.24 49.062.50
p-value 0.0043 0.0005(b)
ogbl-citation2
79.910.13
80.200.62
0.0075
Table 17: Average error rates (lower is better), 20 runs standard deviation for each property, on the
QM9 dataset. We report on GAT and GATv2 with 8 attention heads.
Predicted Property
Model 1 2 3 4 5 6 7
GAT 2.74 0.08 4.730.40 1.470.06 1.530.06 2.440.60 55.2142.33 25.3631.42
GATv2 2.670.08 4.280.23 1.430.05 1.510.07 2.210.08 16.641.17 13.611.68
p-value 0.0043 <0.0001 0.0138 0.1691 0.0487 0.0001 0.0516
Predicted Property
Model 8 9 10 11 12 13
GAT 7.36 0.87 6.790.86 7.360.93 6.690.86 4.100.29 1.510.84
GATv2 6.130.59 6.330.82 6.370.86 5.950.62 3.660.29 1.090.85
p-value <0.0001 0.0458 0.0006 0.0017 <0.0001 0.0621
25

## Page 26

Published as a conference paper at ICLR 2022
G C OMPLEXITY ANALYSIS
We repeat the deﬁnitions of GAT, GATv2 and DPGAT:
GAT (Veli ˇckovi ´c et al., 2018): e(hi;hj) =LeakyReLU 
a>[WhikWhj]
(52)
GATv2 (our ﬁxed version): e(hi;hj) =a>LeakyReLU ( W[hikhj]) (53)
DPGAT (Vaswani et al., 2017): e(hi;hj) = 
h>
iQ
 
h>
jK>
=p
d0 (54)
G.1 T IMECOMPLEXITY
GAT As noted by Veli ˇckovi ´c et al. (2018), the time complexity of a single GAT head may be
expressed asO(jVjdd0+jEjd0). Because of GAT’s static attention, this computation can be further
optimized, by merging the linear layer a1withW, merging a2withW, and only then compute
a>
f1;2gWhifor everyi2V.
GATv2 require the same computational cost as GAT’s declared complexity: O(jVjdd0+jEjd0): we
denote W= [W1kW2], where W12Rd0dandWd0d
2 contain the left half and right half of the
columns of W, respectively. We can ﬁrst compute W1hiandW2hjfor everyi;j2V. This takes
O(jVjdd0).
Then, for every edge (j;i), we compute LeakyReLU ( W[hikhj])using the precomputed W1hi
andW2hj, since W[hikhj] =W1hi+W2hj. This takesO(jEjd0).
Finally, computing the results of the linear layer atakes additionalO(jEjd0)time, and overall
O(jVjdd0+jEjd0).
DPGAT also takes the same time. We can ﬁrst compute h>
iQandh>
jKfor everyi;j2V. This
takesO(jVjdd0). Computing the dot-product 
h>
iQ 
h>
jK>for every edge (j;i)takes additional
O(jEjd0)time, and overallO(jVjdd0+jEjd0).
G.2 P ARAMETRIC COMPLEXITY
GAT GATv2 DPGAT
Ofﬁcial 2d0+dd0d0+ 2dd02ddk+dd0
In our experiments 2d0+dd0d0+dd02dd0
Table 18: Number of parameters for each GNN type, in a single layer and a single attention head.
All parametric costs are summarized in Table 18. All following calculations refer to a single layer
having a single attention head, omitting bias vectors.
GAT has learned vector and a matrix: a2R2d0andW2Rd0d, thus overall 2d0+dd0learned
parameters.
GATv2 has a matrix that is twice larger: W2Rd02d, because it is applied on the concatenation
[hikhj]. Thus, the overall number of learned parameters is d0+ 2dd0. However in our experiments,
to rule out the increased number of parameters over GAT as the source of empirical difference, we
constrained W= [W0kW0], and thus the number of parameters were d0+dd0.
DPGAT hasQandKmatrices of sizes ddkeach, and additional dd0parameters in the value matrix
V, thus 2ddk+dd0parameters overall. However in our experiments, we constrained Q=Kand
setdk=d0, and thus the number of parameters is only 2dd0.
26