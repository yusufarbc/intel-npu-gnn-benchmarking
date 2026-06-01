# Exphormer Sparse Transformers for Graphs


## Page 1

Published as a conference paper at ICLR 2022
GRAPH -LESS NEURAL NETWORKS : T EACHING OLD
MLP SNEWTRICKS VIA DISTILLATION
Shichang Zhang
University of California, Los Angeles
shichang@cs.ucla.eduYozen Liu
Snap Inc.
yliu2@snap.com
Yizhou Sun
University of California, Los Angeles
yzsun@cs.ucla.eduNeil Shah
Snap Inc.
nshah@snap.com
ABSTRACT
Graph Neural Networks (GNNs) are popular for graph machine learning and have
shown great results on wide node classiﬁcation tasks. Yet, they are less popular
for practical deployments in the industry owing to their scalability challenges
incurred by data dependency. Namely, GNN inference depends on neighbor nodes
multiple hops away from the target, and fetching them burdens latency-constrained
applications. Existing inference acceleration methods like pruning and quantiza-
tion can speed up GNNs by reducing Multiplication-and-ACcumulation (MAC)
operations, but the improvements are limited given the data dependency is not
resolved. Conversely, multi-layer perceptrons (MLPs) have no graph dependency
and infer much faster than GNNs, even though they are less accurate than GNNs
for node classiﬁcation in general. Motivated by these complementary strengths and
weaknesses, we bring GNNs and MLPs together via knowledge distillation (KD).
Our work shows that the performance of MLPs can be improved by large margins
with GNN KD. We call the distilled MLPs Graph-less Neural Networks ( GLNN s) as
they have no inference graph dependency. We show that GLNN s with competitive
accuracy infer faster than GNNs by 146 ×-273×and faster than other acceleration
methods by 14 ×-27×. Under a production setting involving both transductive and
inductive predictions across 7 datasets, GLNN accuracies improve over stand-alone
MLPs by 12.36% on average and match GNNs on 6/7 datasets. Comprehensive
analysis shows when and why GLNN s can achieve competitive accuracies to GNNs
and suggests GLNN as a handy choice for latency-constrained applications.
1 I NTRODUCTION
Graph Neural Networks (GNNs) have recently become very popular for graph machine learning
(GML) research and have shown great results on node classiﬁcation tasks (Kipf & Welling, 2016;
Hamilton et al., 2017; Veli ˇckovi ´c et al., 2017) like product prediction on co-purchasing graphs
and paper category prediction on citation graphs. However, for large-scale industrial applications,
MLPs remain the major workhorse, despite common (implicit) underlying graphs and suitability for
GML formalisms. One reason for this academic-industrial gap is the challenges in scalability and
deployment brought by data dependency in GNNs (Zhang et al., 2020; Jia et al., 2020), which makes
GNNs hard to deploy for latency-constrained applications that require fast inference.
Neighborhood fetching caused by graph dependency is one of the major sources of GNN latency.
Inference on a target node necessitates fetching topology and features of many neighbor nodes,
especially on small-world graphs (detailed discussion in Section 4). Common inference acceleration
techniques like pruning (Zhou et al., 2021) and quantization (Tailor et al., 2021; Zhao et al., 2020)
can speed up GNNs to some extent by reducing Multiplication-and-ACcumulation (MAC) operations.
Work done when author was an intern at Snap Inc. Code available at https://github.com/
snap-research/graphless-neural-networks
1arXiv:2110.08727v2  [cs.LG]  23 Mar 2022

## Page 2

Published as a conference paper at ICLR 2022
However, their improvements are limited given the graph dependency is not resolved. Unlike GNNs,
MLPs have no dependency on graph data and are easier to deploy than GNNs. They also enjoy
the auxiliary beneﬁt of sidestepping the cold-start problem that often happens during the online
prediction of relational data (Wei et al., 2020), meaning MLPs can infer reasonably even when
neighbor information of a new encountered node is not immediately available. On the other hand, this
lack of graph dependency typically hurts for relational learning tasks, limiting MLP performance on
GML tasks compared to GNNs. We thus ask: can we bridge the two worlds, enjoying the low-latency,
dependency-free nature of MLPs and the graph context-awareness of GNNs at the same time?
Present work. Our key ﬁnding is that it is possible to distill knowledge from GNNs to MLPs without
losing signiﬁcant performance, but reducing the inference time drastically for node classiﬁcation.
The knowledge distillation (KD) can be done ofﬂine, coupled with model training. In other words,
we can shift considerable work from the latency-constrained inference step, where time reduction
in milliseconds makes a huge difference, to the less time-sensitive training step, where time cost
in hours or days is often tolerable. We call our approach Graph-less Neural Network ( GLNN ).
Speciﬁcally, GLNN is a modeling paradigm involving KD from a GNN teacher to a student MLP;
the resulting GLNN is an MLP optimized through KD, so it enjoys the beneﬁts of graph context-
awareness in training but has no graph dependency in inference. Regarding speed, GLNN s have
superior efﬁciency and are 146×-273× faster than GNNs and 14×-27× faster than other inference
acceleration methods. Regarding performance, under a production setting involving both transductive
and inductive predictions on 7 datasets, GLNN accuracies improve over MLPs by 12.36% on average
and match GNNs on 6/7 datasets. We comprehensively study when and why GLNN s can achieve
competitive results as GNNs. Our analysis suggests the critical factors for such great performance
are large MLP sizes and high mutual information between node features and labels. Our observations
align with recent results in vision and language, which posit that large enough (or slightly modiﬁed)
MLPs can achieve similar results as CNNs and Transformers (Liu et al., 2021; Tolstikhin et al., 2021;
Melas-Kyriazi, 2021; Touvron et al., 2021; Ding et al., 2021). Our core contributions are as follows:
•We propose GLNN , which eliminates neighbor-fetching latency in GNN inference via KD to MLP.
•We show GLNN s has competitive performance as GNNs, while enjoying 146×-273× faster inference
than vanilla GNNs and 14×-27× faster inference than other inference acceleration methods.
•We study GLNN properties comprehensively by investigating their performance under different
settings, how they work as regularizers, their inductive bias, expressiveness, and limitations.
2 R ELATED WORK
Graph Neural Networks. The early GNNs generalize convolution nets to graphs (Bruna et al., 2014;
Defferrard et al., 2017) and later simpliﬁed to message-passing neural net (MPNN) by GCN (Kipf
& Welling, 2016). Most GNNs after can be put as MPNNs. For example, GAT employs attention
(Veli ˇckovi ´c et al., 2017), PPNP employs personalized PageRank (Klicpera et al., 2019), GCNII and
DeeperGCN employ residual connections and dense connections (Chen et al., 2020; Li et al., 2019).
Inference Acceleration. Inference acceleration have been proposed by hardware improvements
(Chen et al., 2016; Judd et al., 2016) and algorithmic improvements through pruning (Han et al.,
2015), quantization (Gupta et al., 2015). For GNNs, pruning (Zhou et al., 2021) and quantizing GNN
parameters (Zhao et al., 2020) have been studied. These approaches speed up GNN inference to a
certain extent but do not eliminate the neighbor-fetching latency. In contrast, our cross-model KD
solves this issue. Concurrently, Graph-MLP also tries to bypass GNN neighbor fetching (Hu et al.,
2021) by training an MLP with a neighbor contrastive loss, but it only considers transductive but not
the more practical inductive setting. Some sampling works focus on speed up GNN training (Zou
et al., 2019; Chen et al., 2018), which are complementary to our goal on inference acceleration.
GNN distillation. Existing GNN KD works try to distill large GNNs to smaller GNNs. LSP (Yang
et al., 2021b) and TinyGNN (Yan et al., 2020) do KD while preserving local information. Their
students are GNNs with fewer parameters but not necessarily fewer layers. Thus, both designs still
require latency-inducing fetching. GFKD (Deng & Zhang, 2021) does graph-level KD via graph
generation. In GFKD, data instances are independent graphs, whereas we focus on dependent nodes
within a graph. GraphSAIL (Xu et al., 2020) uses KD to learn students work well on new data while
preserving performance on old data. CPF (Yang et al., 2021a) combines KD and label propagation
(LP). The student in CPF is not a GNN, but it is still heavily graph-dependent as it uses LP.
2

## Page 3

Published as a conference paper at ICLR 2022
Figure 1: The number of fetches and the inference time of GNNs are both magnitudes more than
MLPs and grow exponentially as functions of the number of layers. Left: neighbors need to be
fetched for two GNN layers. Middle : the total number of fetches for inference. Right : the total
inference time. (Inductive inference for 10 random nodes on OGB Products (Hu et al., 2020))
3 P RELIMINARIES
Notations. For GML tasks, the input is usually a graph and its node features, which we write as
G= (V;E), withVstands for all nodes, and Estands for all edges. Let Ndenote the total number of
nodes. We use X2RNDto represent node features, with row xvbeing theD-dimensional feature
of nodev2V. We represent edges with an adjacency matrix A, withAu;v= 1if edge (u;v)2E,
and 0 otherwise. For node classiﬁcation, one of the most important GML applications, the prediction
targets are Y2RNK, where row yvis aK-dim one-hot vector for node v. For a givenG, usually
a small portion of nodes will be labeled, which we mark using superscriptL, i.e.VL,XL, andYL.
The majority of nodes will be unlabeled, and we mark using the superscriptU, i.e.VU,XU, andYU.
Graph Neural Networks. Most GNNs ﬁt under the message-passing framework, where the rep-
resentation hvof each node vis updated iteratively in each layer by collecting messages from its
neighbors denoted as N(v). For thel-th layer, h(l)
vis obtained from the previous layer representation
h(l 1)
u (h(0)
u=xu) via an aggregation operation AGGR followed by an UPDATE operation as
h(l)
N(v)=AGGR (fh(l 1)
u :u2N(v)g) and h(l)
v=UPDATE (h(l)
N(v);h(l 1)
v)
4 M OTIVATION
GNNs have considerable inference latency due to graph dependency. One more GNN layer means
fetching one more hop of neighbors. To infer a node with a L-layer GNN on a graph with average
degreeRrequiresO(RL)fetches.Rcan be large for real-world graphs, e.g. 208 for the Twitter
(Ching et al., 2015). Also, as layer fetching must be done sequentially, the total latency explodes
quickly asLincreases. Figure 1 shows the dependency added by each GNN layer and the exponential
explosion of inference time. In contrast, the MLP inference time is much smaller and grows linearly.
This marked gap contributes greatly to the practicality of MLPs in industrial applications over GNNs.
The node-fetching latency is exacerbated by two factors: ﬁrstly, newer GNN architectures are getting
deeper from 64 layers (Chen et al., 2020) to even 1001 layers (Li et al., 2021). Secondly, industrial-
scale graphs are frequently too large to ﬁt into the memory of a single machine (Jin et al., 2022),
necessitating sharding of the graph out of the main memory. For example, Twitter has 288M monthly
active users (nodes) and an estimated 60B followers (edges) as of 3/2015. Facebook has 1.39B
active users with more than 400B edges as of 12/2014 (Ching et al., 2015). Even when stored in a
sparse-matrix-friendly format (often COO or CSR), these graphs are on the order of TBs and are
constantly growing. Moving away from in-memory storage results in even slower neighbor-fetching.
MLPs, on the other hand, lack the means to exploit graph topology, which hurts their performance for
node classiﬁcation. For example, test accuracy on Products is 78.61 for GraphSAGE compared to
62.47 for an equal-sized MLP. Nonetheless, recent results in vision and language posit that large (or
slightly modiﬁed) MLPs can achieve similar results as CNNs and Transformers (Liu et al., 2021).
We thus also ask: Can we bridge the best of GNNs and MLPs to get high-accuracy and low-latency
models? This motivates us to do cross-model KD from GNNs to MLPs.
3

## Page 4

Published as a conference paper at ICLR 2022
Offline Training with DistillationOnline Prediction on New Nodes
Only NodeFeaturesTrainedGNNTeacher
Distilled KnowledgeSoft Targets
DeployNo dependency on graph in greyNew node/edges in dashed linesMLPStudent
DeployedGLNN
Figure 2: The GLNN framework: In ofﬂine training, a trained GNN teacher is applied on the graph for
soft targets. Then, a student MLP is trained on node features guided by the soft targets. The distilled
MLP, now GLNN , is deployed for online predictions. Since graph dependency is eliminated for
inference, GLNN s infer much faster than GNNs, and hence the name “Graph-less Neural Network.”
5 G RAPH -LESS NEURAL NETWORKS
We introduce GLNN and answer exploration questions of its properties: 1)How do GLNN s compare
to MLPs and GNNs? 2)CanGLNN s work well under both transductive and inductive settings? 3)
How do GLNN s compare to other inference acceleration methods? 4)How do GLNN s beneﬁt from
KD? 5)DoGLNN s have sufﬁcient model expressiveness? 6)When will GLNN s fail to work?
5.1 T HEGLNN FRAMEWORK
The idea of GLNN is straightforward, yet as we will see, extremely effective. In short, we train a
“boosted” MLP via KD from a teacher GNN. KD was introduced in Hinton et al. (2015), where
knowledge was transferred from a cumbersome teacher to a simpler student. In our case, we generate
soft targets zvfor each node vwith a teacher GNN. Then we train a student MLP with both true
labelsyvandzv. The objective is as Equation 1, with being a weight parameter, Llabel being the
cross-entropy between yvand student predictions ^yv,Lteacher being the KL-divergence.
L=v2VLLlabel(^yv;yv) + (1 )v2VLteacher (^yv;zv) (1)
The model after KD, i.e. GLNN , is essentially a MLP. Therefore, GLNN s have no graph dependency
during inference and are as fast as MLPs. On the other hand, through ofﬂine KD, GLNN parameters
are optimized to predict and generalize as well as GNNs, with the added beneﬁt of faster inference
and easier deployment. In Figure 2, we show the ofﬂine KD and online inference steps of GLNN s.
5.2 E XPERIMENT SETTINGS
Datasets. We consider all ﬁve datasets used in the CPF paper (Yang et al., 2021a), i.e. Cora ,
Citeseer ,Pubmed ,A-computer , and A-photo . To fully evaluate our method, we also
include two more larger OGB datasets (Hu et al., 2020), i.e. Arxiv andProducts .
Model Architectures. For consistent results, we use GraphSAGE (Hamilton et al., 2017) with GCN
aggregation as the teacher. We conduct ablation studies of other GNN teachers like GCN (Kipf &
Welling, 2016), GAT (Veli ˇckovi ´c et al., 2017) and, APPNP (Klicpera et al., 2019) in Section 6.
Evaluation Protocol. For all experiments in this section, we report the average and standard deviation
over ten runs with different random seeds. Model performance is measured as accuracy, and results
are reported on test data with the best model selected using validation data.
Transductive vs. Inductive. GivenG,X, andYL, we consider node classiﬁcation under two
settings: transductive ( tran) and inductive ( ind). For ind, we hold out some test data for inductive
evaluation only. We ﬁrst select inductive nodes VU
indVU, which partitionsVUinto the disjoint
inductive subset and observed subset, i.e. VU=VU
obstVU
ind. Then we hold out v2VU
indand all edges
connected to v2VU
ind, which leads to two disjoint graphs G=GobstGindwith no shared nodes or
4

## Page 5

Published as a conference paper at ICLR 2022
Table 1: GLNN s outperform MLPs by large margins and match GNNs on 5 of 7 datasets under the
transductive setting. MLP (GNN )represents difference between the GLNN and a trained MLP
(GNN). Results show accuracy (higher is better); GNN0indicates GLNN outperforms GNN.
Datasets SAGE MLP GLNN MLP GNN
Cora 80.521.77 59.22 1.31 80.541.35 21.32 (36.00%) 0.02 (0.02%)
Citeseer 70.331.97 59.61 2.88 71.772.01 12.16 (20.40%) 1.44 (2.05%)
Pubmed 75.392.09 67.55 2.31 75.422.31 7.87 (11.65%) 0.03 (0.04%)
A-computer 82.972.16 67.80 1.06 83.031.87 15.23 (22.46%) 0.06 (0.07%)
A-photo 90.900.84 78.77 1.74 92.111.08 13.34 (16.94%) 1.21 (1.33%)
Arxiv 70.920.17 56.050.46 63.46 0.45 7.41 (13.24%) -7.46 (-10.52%)
Products 78.610.49 62.470.10 68.86 0.46 6.39 (10.23%) -9.75 (-12.4%)
Table 2: Enlarged GLNN s match the performance of GNNs on the OGB datasets. For Arxiv , we use
MLPw4 ( GLNN w4). For Products , we use MLPw8 ( GLNN w8).
Datasets SAGE MLP+ GLNN + MLP GNN
Arxiv 70.920.17 55.31 0.47 72.150.27 16.85 (30.46%) 0.51 (0.71%)
Products 78.610.49 64.500.45 77.65 0.48 13.14 (20.38%) -0.97 (-1.23%)
edges. Node features and labels are partitioned into three disjoint sets, i.e. X=XLtXU
obstXU
ind,
andY=YLtYU
obstYU
ind. Concretely, the input/output of both settings become:
•tran: train onG,X, andYL; evaluate on (XU;YU); KD uses zvforv2V.
•ind: train onGobs,XL,XU
obs, andYL; evaluate on (XU
ind;YU
ind); KD uses zvforv2VLtVU
obs.
Note that for tran, all the nodes in the graph including the validation and test nodes are used to
generate z. A discussion of this choice along with other experiment details are in Appendix A.
5.3 H OW DO GLNN S COMPARE TO MLP S AND GNN S?
We start by comparing GLNN s to MLPs and GNNs with the same number of layers and hidden
dimensions. We ﬁrst consider the standard transductive setting, so our results in Table 1 are directly
comparable to results reported in previous literature like Yang et al. (2021a) and Hu et al. (2020).
As shown in Table 1, the performance of all GLNN s improve over MLPs by large margins. On smaller
datasets (ﬁrst 5 rows), GLNN s can even outperform the teacher GNNs. In other words, for each task,
with the same parameter budget, there exists a set of MLP parameters that has GNN-competitive
performance (detailed discussion in Sections 5.6 and 5.7). For the larger OGB datasets (last 2 rows),
theGLNN performance is improved over MLPs but still worse than the teacher GNNs. However, as
we show in Table 2, this gap can be mitigated by increasing MLP size to MLPw i1. In Figure 3 (right),
we visualize the trade-off between prediction accuracy and model inference time with different model
sizes. We show that gradually increasing GLNN size pushes its performance to be close to SAGE. On
the other hand, when we reduce the number of layers of SAGE2, the accuracy quickly drops to be
worse than GLNN s. A detailed discussion of the rationale for increasing MLP sizes is in Appendix B.
5.4 C ANGLNN S WORK WELL UNDER BOTH TRANSDUCTIVE AND INDUCTIVE SETTINGS ?
Although transductive is the commonly studied setting for node classiﬁcation, it does not encompass
prediction on unseen nodes. Therefore, it may not be the best way to evaluate a deployed model,
which must often generate predictions for new data points as well as reliably maintain performance on
old ones. Thus, to better understand the effectiveness of GLNN , we also consider their performance
under a realistic production setting, which contains both transductive and inductive predictions.
To evaluate a model inductively, we hold out some test nodes from training to form an inductive set,
i.e.VU=VU
obstVU
ind. In production, a model might be re-trained periodically, e.g. weekly. The
hold-out nodes in VU
indrepresent new nodes entered the graph between two trainings. VU
indis usually
1-wimeans i-times wider hidden layers, e.g. hidden layers of MLPw4 are 4-times wider than the given MLP.
2-Liis used to explicitly note a model with ilayers, e.g. SAGE-L2 represents a 2-layer SAGE.
5

## Page 6

Published as a conference paper at ICLR 2022
1 2 3 4
# Layers100101102103104105Log Scale Inference Time (ms)
1.31.541.75
1.843.347.56
2.345.0917.2
0.6929.31101.812071.333006.4MLP
MLPw4
MLPw8
SAGE
101102103104
Log Scale Inference Time (ms)6065707580Accuracy
All MLPsGLNNGLNNw4GLNNw8
SAGE-L1SAGE-L2SAGE-L3
Figure 3: Enlarged MLPs ( GLNN s) can match GNN accuracy, but infer dramatically faster. Plots are
under the same setting as Figure 1. Left: inference time of MLPs vs. GNN (SAGE) for different
model sizes. Right : model accuracy vs. inference time. Note: time axes are log-scaled.
Table 3: GLNN s match GNN performance on a production setting with both inductive andtrans-
ductive predictions. We use MLP for the 5 CPF datasets, MLPw4 for Arxiv , and MLPw8 for
Products .indresults onVU
ind,tran results onVU
obs, and the interpolated prod results are reported.
Datasets Eval SAGE MLP/MLP+ GLNN /GLNN +MLP GNN
Cora prod 79.29 58.98 78.28 19.30 (32.72%) -1.01 (-1.28%)
ind 81.332.19 59.09 2.96 73.82 1.93 14.73 (24.93%) -7.51 (-9.23%)
tran 78.781.92 58.95 1.66 79.39 1.64 20.44 (34.66%) 0.61 (0.77%)
Citeseer prod 68.38 59.81 69.27 9.46 (15.82%) 0.89 (1.30%)
ind 69.753.59 60.06 5.00 69.25 2.25 9.19 (15.30%) -0.5 (-0.7%)
tran 68.043.34 59.75 2.48 69.28 3.12 9.63 (15.93%) 1.24 (1.82%)
Pubmed prod 74.88 66.80 74.71 7.91 (11.83%) -0.17 (-0.22%)
ind 75.262.57 66.85 2.96 74.30 2.61 7.45 (11.83%) -0.96 (-1.27%)
tran 74.782.22 66.79 2.90 74.81 2.39 8.02 (12.01%) 0.03 (0.04%)
A-computer prod 82.14 67.38 82.29 14.90 (22.12%) 0.15 (0.19%)
ind 82.081.79 67.84 1.78 80.92 1.36 13.08 (19.28%) -1.16 (-1.41%)
tran 82.151.55 67.27 1.36 82.63 1.40 15.36 (22.79%) 0.48 (0.58%)
A-photo prod 91.08 79.25 92.38 13.13 (16.57%) 1.30 (1.42%)
ind 91.500.79 79.44 1.72 91.18 0.81 11.74 (14.78%) -0.32 (-0.35%)
tran 90.800.77 79.20 1.64 92.68 0.56 13.48 (17.01%) 1.70 (1.87%)
Arxiv prod 70.73 55.30 65.09 9.79 (17.70%) -5.64 (-7.97%)
ind 70.640.67 55.40 0.56 60.48 0.46 4.3 (7.76%) -10.94 (-15.49%)
tran 70.750.27 55.28 0.49 71.46 0.33 11.16 (20.18%) -4.31 (-6.09%)
Products prod 76.60 63.72 75.77 12.05 (18.91%) -0.83 (-1.09%)
ind 76.890.53 63.70 0.66 75.16 0.34 11.44 (17.96%) -1.73 (-2.25%)
tran 76.530.55 63.73 0.69 75.92 0.61 12.20 (19.15%) -0.61 (-0.79%)
small compared to VU
obs– e.g. Graham (2012) estimates 5-7% for the fastest-growing tech startups.
In our case, to mitigate randomness and better evaluate generalizability, we use VU
indcontaining 20%
of the test data. We also evaluate on VU
obscontaining the other 80% of the test data, representing the
standard transductive prediction on observed unlabeled nodes, since inference is commonly redone
on existing nodes in real-world cases. We report both results and a interpolated production ( prod )
results in Table 3. The prod results paint a clearer picture of model generalization as well as accuracy
in production. See Section 6 for an ablation study of different inductive split rates other than 20-80.
In Table 3, we see that GLNN s can still improve over MLP by large margins for inductive predictions.
On 6/7 datasets, the GLNN prod performance are competitive to GNNs, which supports deploying
GLNN as a much faster model with no or only slight performance loss. On the Arxiv dataset, the
GLNN performance is notably less than GNNs – we hypothesize this is due to Arxiv having a
particularly challenging data split which causes distribution shift between test nodes and training
6

## Page 7

Published as a conference paper at ICLR 2022
Table 4: While other inference acceleration methods speed up SAGE, they are considerably slower
thanGLNN s. Numbers (in ms) are inductive inference time on 10 randomly chosen nodes.
Datasets SAGE QSAGE PSAGE Neighbor Sample GLNN +
Arxiv 489.49 433.90 (1.13×) 465.43 (1.05×) 91.03 (5.37×) 3.34 (146.55×)
Products 2071.30 1946.49 (1.06×) 2001.46 (1.04×) 107.71 (19.23×) 7.56 (273.98×)
nodes, which is hard for GLNN s to capture without utilizing neighbor information like GNNs.
However, we note that GLNN performance is substantially improved over MLP.
5.5 H OW DO GLNN S COMPARE TO OTHER INFERENCE ACCELERATION METHODS ?
Common techniques of inference acceleration include pruning and quantization. These approaches
can reduce model parameters and Multiplication-and-ACcumulation (MACs) operations. Still, they
don’t eliminate neighbor-fetching latency. Therefore, their speed gain on GNNs is less signiﬁcant
than on NNs. For GNNs, neighbor sampling is also used to reduce the fetching latency. We show an
explicit speed comparison between vanilla SAGE, quantized SAGE from FP32 to INT8 (QSAGE),
SAGE with 50% weights pruned (PSAGE), inference neighbor sampling with fan-out 15, and GLNN
in Table 4. With the same setting as Figure 1, we see that GLNN is considerably faster.
Two other kinds of methods considered as inference acceleration are GNN-to-GNN KD like TinyGNN
(Yan et al., 2020) and Graph Augmented-MLPs (GA-MLPs) like SGC (Wu et al., 2019) or SIGN
(Frasca et al., 2020). Inference of GNN-to-GNN KD is likely to be slower than a GNN-L iwith
the sameias the student, since there will usually be some extra overheads like the Peer-Aware
Module (PAM) in TinyGNN. GA-MLPs precompute augmented node features and apply MLPs to
them. With precomputation, their inference time will be the same as MLPs for dimension-preserving
augmentation (SGC) and the same as enlarged MLPw ifor augmentation involves concatenation
(SIGN). Thus, for both kinds of approaches, it is sufﬁcient to compare GLNN with GNN-L iand
MLPwi, which we have already shown in Figure 3 (left). We see that GNN-L is are much slower than
MLPs. For GA-MLPs, since full pre-computation cannot be done for inductive nodes, GA-MLPs still
need to fetch neighbor nodes. This makes them much slower than MLPw iin the inductive setting,
and even slower than pruned GNNs and TinyGNN as shown in Zhou et al. (2021).
5.6 H OW DOES GLNN BENEFIT FROM DISTILLATION ?
We showed that GNNs are markedly better than MLPs on node classiﬁcation tasks. But, with
KD,GLNN s can often become competitive to GNNs. This indicates that there exist suitable MLP
parameters which can well approximate the ideal prediction function from node features to labels.
However, these parameters can be difﬁcult to learn through standard stochastic gradient descent. We
hypothesize that KD helps to ﬁnd them through regularization and transfer of inductive bias.
First, we show that KD can help to regularize the student model. From loss curves of a directly trained
MLP and the GLNN in Figure 4, we see the gap between training and validation loss is visibly larger
for MLPs than GLNN s, and MLPs show obvious overﬁtting trends. Second, we analyze the inductive
bias that makes GNNs powerful on node classiﬁcation, which suggests that node inferences should be
inﬂuenced by the graph topology. Whereas MLPs have less inductive bias. Similar difference exists
between Transformers (Vaswani et al., 2017) and MLPs. Liu et al. (2021) shows that the inductive
bias in Transformers can be mitigated by a simple gate on large MLPs. For node classiﬁcation, we
hypothesize that KD helps to mitigate the inductive bias, so GLNN s can perform competitively. Soft
labels from GNN teachers are heavily inﬂuenced by the graph topology due to inductive bias. They
maintain nonzero probabilities on classes other than the ground truth provided by labels, which can
be useful for the student to learn to complement the missing inductive bias in MLPs. To evaluate
this hypothesis quantitatively, we deﬁne the cut loss Lcut2[0;1]in Equation 2 to measure the
consistency between model predictions and graph topology (details in Appendix C):
Lcut=Tr(^YTA^Y)
Tr(^YTD^Y)(2)
Here ^Y2[0;1]NKis the soft classiﬁcation probability output by the model, AandDare the
adjacency and degree matrices. When Lcutis close to 1, it means the predictions and the graph
7

## Page 8

Published as a conference paper at ICLR 2022
Figure 4: Loss curves on CPF datasets show GLNN distillation can help to regularize the training.
Here the training loss of GLNN is on hard labels, only corresponding to the ﬁrst term in Equation 1.
topology are very consistent. In our experiment, we observe that the average Lcutfor SAGE over ﬁve
CPF datasets is 0.9221, which means high consistency. The same Lcutfor MLPs is only 0.7644, but
forGLNN s it is 0.8986. This shows that the GLNN predictions indeed beneﬁt from the graph topology
knowledge contained in the teacher outputs (the full table of Lcutvalues in Appendix C).
5.7 D OGLNN S HAVE ENOUGH MODEL EXPRESSIVENESS ?
Intuitively, the addition of neighbor information makes GNNs more powerful than MLPs when
classifying nodes. Thus, a natural question regarding KD from GNNs to MLPs is whether MLPs are
expressive enough to represent graph data as well as GNNs. Many recent works studied GNN model
expressiveness (Xu et al., 2018; Chen et al., 2021). The latter analyzed GNNs and GA-MLPs for
node classiﬁcation and characterized expressiveness as the number of equivalence classes of rooted
graphs induced by the model (formal deﬁnitions in Appendix D). The conclusion is that GNNs are
more powerful than GA-MLPs, but in most real-world cases their expressiveness is indistinguishable.
We adopt the analysis framework from Chen et al. (2021) and show in Appendix D that the number
of equivalence classes induced by GNNs and MLPs are jXj+m 2
m 12L 1andjXjrespectively. Here
mdenotes the max node degree, Ldenotes the number of GNN layers, and Xdenotes the set
of all possible node features. The former is apparently larger which concludes that GNNs are
more expressive. Empirically, however, the gap makes little difference when jXjis large. In real
applications, node features can be high dimensional like bag-of-words, or even word embeddings,
thus makingjXjenormous. Like for bag-of-words, jXjis in the order ofO(pD), whereDis the
vocabulary size, and pis the max word frequency. The expressiveness of a L-layer GNN is lower
bounded by jXj+m 2
m 12L 1=O(pD(m 1)(2L 1)), but empirically, both MLPs and GNNs should
have enough expressiveness given Dis usually hundreds or bigger (see Table 5).
5.8 W HEN WILL GLNN S FAIL TO WORK ?
As discussed in Section 5.7 and Appendix D, the goal of GML node classiﬁcation is to ﬁt a function
fon the rooted graph G[i]and label yi. From the information theoretic perspective, ﬁtting fby
minimizing the commonly used cross-entropy loss is equivalent to maximizing the mutual information
(MI),I(G[i];yi)as shown in Qin et al. (2020). If we consider G[i]as a joint distribution of two
random variables X[i]andE[i]representing the node features and edges in G[i]respectively, we have
I(G[i];yi) =I(X[i];E[i];yi) =I(E[i];yi) +I(X[i];yijE[i]) (3)
I(E[i];yi)only depends on edges and labels, thus MLPs can only maximize I(X[i];yijE[i]). In the
extreme case, I(X[i];yijE[i])can be zero when y[i]is conditionally independent from X[i]given
E[i]. For example, when every node is labeled by its degree or whether it forms a triangle. Then MLPs
won’t be able to ﬁt meaningful functions, and neither will GLNN s. However, such cases are typically
rare, and unexpected in practical settings our work is mainly concerned with. For real GML tasks,
node features and structural roles are often highly correlated (Lerique et al., 2020), hence MLPs can
achieve reasonable results even only based on node features, and thus GLNN s can potentially achieve
much better results. We study the failure case of GLNN s by creating a low MI scenario in Section 6.
8

## Page 9

Published as a conference paper at ICLR 2022
0.0 0.2 0.4 0.6 0.8 1.0
Noise Level 
102030405060708090AccuracyMLP-ind
GNN-ind
GLNN-ind
10-90 20-80 30-70 40-60 50-50
Inductive:Transductive Ratio60657075808590MLP-ind
GNN-ind
GLNN-indMLP-tran
GNN-tran
GLNN-tran
MLP SAGE GCN GAT APPNP
Model Architectures60657075808590
66.479.280.1 80.2 80.179.4 79.9 79.778.8MLP GNN GLNN
Figure 5: Left: Node feature noise. GLNN has comparable performance to GNNs only when nodes
are less noisy. Adding more noise decreases GLNN performance faster than GNNs. Middle : Inductive
split rate. Altering the inductive:transductive ratio in the production setting doesn’t affect the accuracy
much. Right : Teacher GNN architecture. GLNN s can learn from different GNN teachers to improve
over MLPs and achieve comparable results. Accuracies are averaged over ﬁve CPF datasets.
6 A BLATION STUDIES
In this section, we do ablation studies of GLNN s on node feature noise, inductive split rates, and
teacher GNN architecture. Reported results are test accuracies averaged over ﬁve datasets in CPF.
More experiments can be found in Appendix including advanced GNN teachers (Appendix F),
GA-MLP student (Appendix G), and non-homogeneous data (Appendix I).
Noisy node features. Following Section 5.8, we investigate failure cases of GLNN by adding different
levels of Gaussian noise to node features to decrease their mutual information with labels. Speciﬁcally,
we replace Xwith ~X= (1 )X+.is an isotropic Gaussian independent from X, and
2[0;1]denotes the noise level. We show the inductive performance of MLP, GNN, and GLNN
under different noise levels in Figure 5 (left). We see that as increases, the accuracy of MLPs and
GLNN s decrease faster than GNNs, while the performance of GLNN s and GNNs are still comparable
for smalls. Whenreaches 1, ~XandYwill become independent corresponding to the extreme
case discussed in Section 5.8. A more detailed discussion is in Appendix J.
Inductive split rate. In Section 5.4, we use a 20-80 split of the test data for inductive evaluation. In
Figure 5 (middle), we show the results under different split rates (More detailed plots in Appendix H).
We see that as the inductive portion increase, GNN and MLP performance stays roughly the same,
and the GLNN inductive performance drops slightly. We only consider rates up to 50-50 since having
50% or even more inductive nodes is highly atypical in practice. When a large amount of new data
are encountered, practitioners can opt to retrain the model on all the data before deployment.
Teacher GNN architecture. We used SAGE to represent GNNs so far. In Figure 5 (right), we show
results with other various GNN teachers, e.g. GCN, GAT, and APPNP. We see that GLNN s can learn
from different teachers and improve over MLPs. The performance is similar for all four teachers,
with the GLNN distilled from APPNP very slightly worse than others. In fact, a similar phenomenon
has been observed in Yang et al. (2021a) as well, i.e. APPNP beneﬁts the student the least. One
possible reason is that the ﬁrst step of APPNP is to utilize the node’s own feature for prediction (prior
to propagating over the graph), which is very similar to what the student MLP is doing, and thus
provides less additional information to MLPs than other teachers.
7 C ONCLUSION AND FUTURE WORK
In this paper, we explored whether we can bridge the best of GNNs and MLPs to achieve accurate and
fast GML models for deployment. We found that KD from GNNs to MLPs helps to eliminate inference
graph dependency, which results in GLNN s that are 146 ×-273×faster than GNNs while enjoying
competitive performance. We do a comprehensive study of GLNN properties. The promising results
on 7 datasets across different domains show that GLNN s can be a handy choice for deploying latency-
constraint models. In our experiments, the current version of GLNN s on the Arxiv dataset doesn’t
show competitive inductive performance. More advanced distillation techniques can potentially
improve the GLNN performance, and we leave this investigation as future work.
9

## Page 10

Published as a conference paper at ICLR 2022
REFERENCES
Filippo Maria Bianchi, Daniele Grattarola, and Cesare Alippi. Mincut pooling in graph neural
networks. CoRR, abs/1907.00481, 2019. URL http://arxiv.org/abs/1907.00481 .
Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann LeCun. Spectral networks and locally
connected networks on graphs, 2014.
Jie Chen, Tengfei Ma, and Cao Xiao. FastGCN: Fast learning with graph convolutional networks
via importance sampling. In International Conference onLearning Representations , 2018. URL
https://openreview.net/forum?id=rytstxWAW .
Lei Chen, Zhengdao Chen, and Joan Bruna. On graph neural networks versus graph-augmented
{mlp}s. In International Conference onLearning Representations , 2021. URL https://
openreview.net/forum?id=tiqI7w64JG2 .
Ming Chen, Zhewei Wei, Zengfeng Huang, Bolin Ding, and Yaliang Li. Simple and deep graph convo-
lutional networks. In Hal Daumé III and Aarti Singh (eds.), Proceedings ofthe37th International
Conference onMachine Learning , volume 119 of Proceedings ofMachine Learning Research , pp.
1725–1735. PMLR, 13–18 Jul 2020. URL https://proceedings.mlr.press/v119/
chen20v.html .
Yu-Hsin Chen, Joel Emer, and Vivienne Sze. Eyeriss: A spatial architecture for energy-efﬁcient
dataﬂow for convolutional neural networks. In 2016 ACM/IEEE 43rd Annual International
Symposium onComputer Architecture (ISCA), pp. 367–379, 2016. doi: 10.1109/ISCA.2016.40.
Avery Ching, Sergey Edunov, Maja Kabiljo, Dionysios Logothetis, and Sambavi Muthukrishnan.
One trillion edges: Graph processing at facebook-scale. Proceedings oftheVLDB Endowment , 8
(12):1804–1815, 2015.
Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on
graphs with fast localized spectral ﬁltering, 2017.
Xiang Deng and Zhongfei Zhang. Graph-free knowledge distillation for graph neural networks, 2021.
Inderjit S. Dhillon, Yuqiang Guan, and Brian Kulis. Kernel k-means: Spectral clustering and normal-
ized cuts. In Proceedings oftheTenth ACM SIGKDD International Conference onKnowledge
Discovery andData Mining , KDD ’04, pp. 551–556, New York, NY , USA, 2004. Associa-
tion for Computing Machinery. ISBN 1581138881. doi: 10.1145/1014052.1014118. URL
https://doi.org/10.1145/1014052.1014118 .
Xiaohan Ding, Xiangyu Zhang, Jungong Han, and Guiguang Ding. Repmlp: Re-parameterizing
convolutions into fully-connected layers for image recognition, 2021.
Fabrizio Frasca, Emanuele Rossi, Davide Eynard, Ben Chamberlain, Michael Bronstein, and Federico
Monti. Sign: Scalable inception graph neural networks, 2020.
Paul Graham. Want to start a startup? http://www.paulgraham.com/growth.html, 2012.
Suyog Gupta, Ankur Agrawal, Kailash Gopalakrishnan, and Pritish Narayanan. Deep learn-
ing with limited numerical precision. In Francis Bach and David Blei (eds.), Proceedings
ofthe32nd International Conference onMachine Learning , volume 37 of Proceedings of
Machine Learning Research , pp. 1737–1746, Lille, France, 07–09 Jul 2015. PMLR. URL
https://proceedings.mlr.press/v37/gupta15.html .
Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In
Advances inneural information processing systems, pp. 1024–1034, 2017.
Song Han, Jeff Pool, John Tran, and William J. Dally. Learning both weights and connections
for efﬁcient neural networks. In Proceedings ofthe28th International Conference onNeural
Information Processing Systems -V olume 1, NIPS’15, pp. 1135–1143, Cambridge, MA, USA,
2015. MIT Press.
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network, 2015.
10

## Page 11

Published as a conference paper at ICLR 2022
Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta,
and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. CoRR ,
abs/2005.00687, 2020. URL https://arxiv.org/abs/2005.00687 .
Yang Hu, Haoxuan You, Zhecan Wang, Zhicheng Wang, Erjin Zhou, and Yue Gao. Graph-mlp:
Node classiﬁcation without message passing in graph. CoRR , abs/2106.04051, 2021. URL
https://arxiv.org/abs/2106.04051 .
Qian Huang, Horace He, Abhay Singh, Ser-Nam Lim, and Austin Benson. Combining la-
bel propagation and simple models out-performs graph neural networks. In International
Conference onLearning Representations , 2021. URL https://openreview.net/forum?
id=8E1-f3VhX1o .
Sergei Ivanov and Liudmila Prokhorenkova. Boost then convolve: Gradient boosting meets graph
neural networks. CoRR , abs/2101.08543, 2021. URL https://arxiv.org/abs/2101.
08543 .
Zhihao Jia, Sina Lin, Rex Ying, Jiaxuan You, Jure Leskovec, and Alex Aiken. Redundancy-free
computation for graph neural networks. In Proceedings ofthe26th ACM SIGKDD International
Conference onKnowledge Discovery &Data Mining, pp. 997–1005, 2020.
Wei Jin, Lingxiao Zhao, Shichang Zhang, Yozen Liu, Jiliang Tang, and Neil Shah. Graph condensation
for graph neural networks. In ICLR, 2022.
Patrick Judd, Jorge Albericio, Tayler Hetherington, Tor M. Aamodt, Natalie Enright Jerger, and
Andreas Moshovos. Proteus: Exploiting numerical precision variability in deep neural networks.
InProceedings ofthe2016 International Conference onSupercomputing, pp. 23, 2016.
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR (Poster) ,
2015. URL http://arxiv.org/abs/1412.6980 .
Thomas N Kipf and Max Welling. Semi-supervised classiﬁcation with graph convolutional networks.
arXiv preprint arXiv:1609.02907, 2016.
Johannes Klicpera, Aleksandar Bojchevski, and Stephan Günnemann. Predict then propagate: Graph
neural networks meet personalized pagerank, 2019.
Sébastien Lerique, Jacob Levy Abitbol, and Márton Karsai. Joint embedding of structure and features
via graph convolutional networks. Applied Network Science, 5(1):1–24, 2020.
Guohao Li, Matthias Muller, Ali Thabet, and Bernard Ghanem. Deepgcns: Can gcns go as deep as
cnns? In Proceedings oftheIEEE International Conference onComputer Vision , pp. 9267–9276,
2019.
Guohao Li, Matthias Müller, Bernard Ghanem, and Vladlen Koltun. Training graph neural networks
with 1000 layers. In International Conference onMachine Learning (ICML), 2021.
Derek Lim, Xiuyu Li, Felix Hohne, and Ser-Nam Lim. New benchmarks for learning on non-
homophilous graphs. CoRR , abs/2104.01404, 2021. URL https://arxiv.org/abs/2104.
01404 .
Hanxiao Liu, Zihang Dai, David R. So, and Quoc V . Le. Pay attention to mlps, 2021.
Luke Melas-Kyriazi. Do you even need attention? a stack of feed-forward layers does surprisingly
well on imagenet, 2021.
Zhenyue Qin, Dongwoo Kim, and Tom Gedeon. Rethinking softmax with cross-entropy: Neural
network classiﬁer as mutual information estimator, 2020.
Shyam Anil Tailor, Javier Fernandez-Marques, and Nicholas Donald Lane. Degree-quant:
Quantization-aware training for graph neural networks. In International Conference onLearning
Representations, 2021. URL https://openreview.net/forum?id=NSBrFgJAHg .
11

## Page 12

Published as a conference paper at ICLR 2022
Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Un-
terthiner, Jessica Yung, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, et al. Mlp-mixer: An
all-mlp architecture for vision. arXiv preprint arXiv:2105.01601, 2021.
Hugo Touvron, Piotr Bojanowski, Mathilde Caron, Matthieu Cord, Alaaeldin El-Nouby, Edouard
Grave, Armand Joulin, Gabriel Synnaeve, Jakob Verbeek, and Hervé Jégou. Resmlp: Feedforward
networks for image classiﬁcation with data-efﬁcient training. arXiv preprint arXiv:2105.03404 ,
2021.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
Kaiser, and Illia Polosukhin. Attention is all you need, 2017.
Petar Veli ˇckovi ´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua
Bengio. Graph attention networks. arXiv preprint arXiv:1710.10903, 2017.
Minjie Wang, Da Zheng, Zihao Ye, Quan Gan, Mufei Li, Xiang Song, Jinjing Zhou, Chao Ma,
Lingfan Yu, Yu Gai, Tianjun Xiao, Tong He, George Karypis, Jinyang Li, and Zheng Zhang.
Deep graph library: A graph-centric, highly-performant package for graph neural networks. arXiv
preprint arXiv:1909.01315, 2019.
Tianxin Wei, Ziwei Wu, Ruirui Li, Ziniu Hu, Fuli Feng, Xiangnan He, Yizhou Sun, and Wei Wang.
Fast adaptation for cold-start collaborative ﬁltering with meta-learning. In 2020 IEEE International
Conference onData Mining (ICDM), pp. 661–670. IEEE, 2020.
Felix Wu, Tianyi Zhang, Amauri Holanda de Souza Jr. au2, Christopher Fifty, Tao Yu, and Kilian Q.
Weinberger. Simplifying graph convolutional networks, 2019.
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural
networks? arXiv preprint arXiv:1810.00826, 2018.
Yishi Xu, Yingxue Zhang, Wei Guo, Huifeng Guo, Ruiming Tang, and Mark Coates. Graphsail: Graph
structure aware incremental learning for recommender systems. In Proceedings ofthe29th ACM
International Conference onInformation; Knowledge Management , CIKM ’20, pp. 2861–2868,
New York, NY , USA, 2020. Association for Computing Machinery. ISBN 9781450368599. doi:
10.1145/3340531.3412754. URL https://doi.org/10.1145/3340531.3412754 .
Bencheng Yan, Chaokun Wang, Gaoyang Guo, and Yunkai Lou. Tinygnn: Learning efﬁcient
graph neural networks. In Proceedings ofthe26th ACM SIGKDD International Conference on
Knowledge Discovery &#38; Data Mining (KDD), pp. 1848–1856, 2020.
Cheng Yang, Jiawei Liu, and Chuan Shi. Extract the knowledge of graph neural networks and go
beyond it: An effective knowledge distillation framework, 2021a.
Yiding Yang, Jiayan Qiu, Mingli Song, Dacheng Tao, and Xinchao Wang. Distilling knowledge from
graph convolutional networks, 2021b.
Dalong Zhang, Xin Huang, Ziqi Liu, Zhiyang Hu, Xianzheng Song, Zhibang Ge, Zhiqiang Zhang,
Lin Wang, Jun Zhou, Yang Shuang, et al. Agl: a scalable system for industrial-purpose graph
machine learning. arXiv preprint arXiv:2003.02454, 2020.
Yiren Zhao, Duo Wang, Daniel Bates, Robert Mullins, Mateja Jamnik, and Pietro Lio. Learned low
precision graph neural networks, 2020.
Hongkuan Zhou, Ajitesh Srivastava, Hanqing Zeng, Rajgopal Kannan, and Viktor K. Prasanna.
Accelerating large scale real-time GNN inference using channel pruning. CoRR , abs/2105.04528,
2021. URL https://arxiv.org/abs/2105.04528 .
Difan Zou, Ziniu Hu, Yewen Wang, Song Jiang, Yizhou Sun, and Quanquan Gu. Layer-dependent
importance sampling for training deep and large graph convolutional networks. arXiv preprint
arXiv:1911.07323, 2019.
12

## Page 13

Published as a conference paper at ICLR 2022
A D ETAILED EXPERIMENT SETTINGS
A.1 D ATASETS
Here we provide a detailed description of the datasets we used to support our argument. Out of these
datasets, 4 of them are citation graphs. Cora, Citeseer, Pubmed, ogbn-arxiv with the node features
being descriptions of the papers, either bag-of-word vector, TF-IDF vector, or word embedding
vectors.
In Table 5, we provided the basic statistics of these datasets.
Table 5: Dataset Statistics.
Dataset # Nodes # Edges # Features # Classes
Cora 2,485 5,069 1,433 7
Citeseer 2,110 3,668 3,703 6
Pubmed 19,717 44,324 500 3
A-computer 13,381 245,778 767 10
A-photo 7,487 119,043 745 8
Arxiv 169,343 1,166,243 128 40
Products 2,449,029 61,859,140 100 47
For all datasets, we follow the setting in the original paper to split the data. Speciﬁcally, for the
ﬁve smaller datasets from the CPF paper, we use the CPF splitting strategy and each random seed
corresponds to a different split. For the OGB datasets, we follow the OGB ofﬁcial splits based on
time and popularity for Arxiv andProducts respectively.
A.2 M ODEL HYPERPARAMETERS
The hyperparameters of GNN models on each dataset are taken from the best hyperparameters
provided by the CPF paper and the OGB ofﬁcial examples. For the student MLPs and GLNN s, unless
otherwise speciﬁed with -w ior -Li, we set the number of layers and the hidden dimension of each
layer to be the same as the teacher GNN, so their total number of parameters stays the same as the
teacher GNN.
Table 6: Hyperparameters for GNNs on ﬁve datasets from the CPF paper.
SAGE GCN GAT APPNP
# layers 2 2 2 2
hidden dim 128 64 64 64
learning rate 0.01 0.01 0.01 0.01
weight decay 0.0005 0.001 0.01 0.01
dropout 0 0.8 0.6 0.5
fan out 5,5 - - -
attention heads - - 8 -
power iterations - - - 10
Table 7: Hyperparameters for GraphSAGE on OGB datasets.
Dataset Arxiv Products
# layers 3 3
hidden dim 256 256
learning rate 0.01 0.003
weight decay 0 0
dropout 0.2 0.5
normalization batch batch
fan out [5, 10, 15] [5, 10, 15]
13

## Page 14

Published as a conference paper at ICLR 2022
ForGLNN s we do a hyperparameter search of learning rate from [0.01, 0.005, 0.001], weight decay
from [0, 0.001, 0.002, 0.005, 0.01], and dropout from [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
A.3 K NOWLEDGE DISTILLATION
We use the distillation method proposed in Hinton et al. (2015) as in Equation 1, the hard labels are
found to be helpful, so nonzero s was suggested. In our case, we did a little tuning for but didn’t
ﬁnd nonzero s to be very helpful. Therefore, we report all of our results with = 0, i.e. only the
second term involving soft labels is effective. More careful tuning of should further improve the
results since the searching space is strictly larger. We implemented a weighted version in our code,
and we leave the choice of as future work.
A.4 T HETRANSDUCTIVE SETTING AND THEINDUCTIVE SETTING
GivenG,X, andYL, the goal of node classiﬁcation can be divided into two different settings,
i.e. transductive and inductive. In real applications, the former can correspond to predict missing
attributes of a user based on the user proﬁle and other existing users, and the latter can correspond to
predict labels of some new nodes that are only seen during inference time. To create the inductive
setting on a given dataset, we hold out some nodes along with edges connected to these nodes during
training and use them for inductive evaluation only. These nodes and edges are picked from the test
data. Using notation deﬁned above, we pick the inductive nodes VU
indVU, which partitionsVU
into the disjoint inductive subset and observed subset, i.e. VU=VU
obstVU
ind. Then we can take
all the edges connected to nodes in VU
indto further partition the whole graph, so we end up with
G=GobstGind,X=XLtXU
obstXU
ind, andY=YLtYU
obstYU
ind. We show the input and
output of both settings using the notations below.
We visualize the difference between the inductive setting and the transductive setting in Figure 6.
Test Node1-hop2-hop New node/edgeTest Node 1-hop2-hop InductiveTransductiveTrain graph
Figure 6: The transductive setting and inductive setting illustrated by a 2-layer GNN. The middle
shows the original graph used for training. The leftshows the transductive setting, where the test
node is in red and within the graph. The right shows the inductive setting, where the test node is an
unseen new node.
A.5 C HOOSING SOFT TARGETS UNDER THE TRANSDUCTIVE SETTING
For the transductive setting in Section 5.3, all the nodes in the graph, including the validation and test
nodes, are used for the soft target generation. It seems less practical compared to the inductive case,
but it is a necessary step to develop our argument. We now discuss the rationale behind this choice.
Firstly, the transductive setting is the most common setting for graph data and it was used in most
GNN architecture works and GNN acceleration works we mentioned in related work. Therefore, to
avoid any confusion and for a fair comparison with numbers from previous literature, we start our
experiments with exactly the same input and output as the standard transductive setting. Under this
setting, the inputs to GNNs include all the node features and the graph structure, so GLNN is set to
be able to access the same input. As GLNN includes a teacher training step and a distillation step,
the soft labels of all the nodes are intermediate outputs produced by the teacher training step, and
thus used for the second distillation step for the best GLNN performance. This transductive setting
can boil down to a sanity check when the student is sufﬁciently large. Therefore, we separate the
setting to be GLNN andGLNN + and report the results in Table 1 and Table 2 separately. In Table
1, we are checking how well GLNN s can perform compared to GNNs under the equal-parameter
14

## Page 15

Published as a conference paper at ICLR 2022
constraint. The results can be interpreted as given a ﬁxed parameter budget, whether there exists one
set of parameters (one instantiation of the MLP) that can achieve competitive results as the GNN.
Only when this holds, should we further investigate the more interesting and challenging inductive
case as in Section 5.4.
Secondly, the task we focus on is node classiﬁcation, which in many cases is considered as semi-
supervised learning with very scarce labels. For example, Pubmed only uses 60 labeled nodes (20
per class) out of 20K nodes for training. Rather than design an advanced model that can do few-shot
learning, our goal here is to leverage as much data as possible to simplify the model for more efﬁcient
inference. We thus utilize the soft pseudo-labels on all the unlabelled nodes for the best GLNN
performance. In reality, when there is a large amount of separate unlabeled data, these unlabeled
data can be used for GLNN distillation training and a different set of labeled data can be used for
evaluation. In our case, we mimic this scenario in the inductive setting in Section 5.4.
A.6 I MPLEMENTATION AND HARDWARD DETAILS
The experiments on both baselines and our approach are implemented using PyTorch, the DGL (Wang
et al., 2019) library for GNN algorithms, and Adam (Kingma & Ba, 2015) for optimization. We run
all experiments on a machine with 80 Intel(R) Xeon(R) E5-2698 v4 @ 2.20GHz CPUs, and a single
NVIDIA V100 GPU with 16GB RAM.
B S PACE AND TIMECOMPLEXITY OF GNN S VS. MLP S
Compared to MLP and GNN, GLNN provides a handy tool for users to trade-off between model
accuracy and time complexity, which does not directly focus on space complexity. Given the space
and time complexity are related, we now provide a more detailed discussion regarding these two
complexities in our experiments.
In Table 1, the model comparison was between equal-sized MLPs ( GLNN s) and GNNs. While ﬁxing
parameter budget to control space complexity is a standard approach when comparing models, it is
not completely fair for cross-model comparison especially for MLPs vs. GNNs. To do inference with
GNNs, the graph needs to be loaded in the memory either entirely or batch by batch, and may use
much larger space than the model parameters. Thus, the actual space complexity of GNNs is much
higher than equal-sized MLPs. From the time complexity perspective, the major inference latency of
GNNs comes from the data dependency as shown in Section 4. Under the same setting as Figure 1, we
show in Figure 3 Left that even a 5-layer MLP with 8 times wider hidden layers still runs much faster
than a single-layer SAGE. Another example of cross-model comparison is Transformers vs. RNNs.
Large Transformers can have more parameters than RNNs because of the attention mechanism, but
they are also faster than RNNs in general, which is an important consideration in the context of
inference time minimization.
In Table 1, we saw that for equal-sized comparison, GLNN s are not as accurate as GNNs on the OGB
datasets. Following the discussion above and given the GLNN s used in Table 1 are relatively small (3
layers and 256 hidden dimensions) for millions of nodes in the OGB datasets, we ask whether this
gap can be mitigated by increasing the MLP and thus GLNN sizes. The answer is yes as shown in
Table 2.
C C ONSISTENCY MEASURE OF MODEL PREDICTIONS AND GRAPH
TOPOLOGY BASED ON MIN-CUT
We introduce a metric to measure the consistency between model predictions and graph topology
based on the min-cut problem in Section 5.6. The K-way normalized min-cut problem, or simply
min-cut, partitions Nnodes inVintoKdisjoint subsets by removing the minimum volume of edges.
According to Dhillon et al. (2004), the min-cut problem can be expressed as
max1
KKX
k=1CT
kACk
CT
kDCk(4)
s:t:C2f0;1gNK;C1K=1N
15

## Page 16

Published as a conference paper at ICLR 2022
withCbeing the node assignment matrix that partitions V, i.e.Ci;j= 1if nodeiis assigned to class
j.Abeing the adjacency matrix and Dbeing the degree matrix. This quantity we try to maximize
here tells us whether the assignment is consistent with the graph topology. The bigger it is, the
less edges need to be removed, and the assignment is more consistent with existing connections
in the graph. In Bianchi et al. (2019), the authors show that when replacing the hard assignments
C2f0;1gNKwith a soft classiﬁcation probability ^Y2[0;1]NK, a cut lossLcutin Equation 2
can become a good approximation of Equation 4 and be used as the measuring metric.
Table 8: GLNN predictions are much more consistent with the graph topology than MLPs. We show
theLcutvalues of GNNs, MLPs, and GLNN s on ﬁve CPF datasets. GLNNLcutvalues become pretty
close to the highLcutvalues of GNNs, which were closely related to the GNN inductive bias.
Datasets SAGE MLP GLNN
Cora 0.9347 0.7026 0.8852
Citeseer 0.9485 0.7693 0.9339
Pubmed 0.9605 0.9455 0.9701
A-computer 0.9003 0.6976 0.8638
A-photo 0.8664 0.7069 0.8398
Average 0.9221 0.7644 0.8986
D E XPRESSIVENESS OF GNN S VS. MLP S IN TERMS OF EQUIVALENCE
CLASSES OF ROOTED GRAPHS
In Chen et al. (2021), the expressiveness of GNNs and GA-MLPs were theoretically quantiﬁed in
terms of induced equivalence classes of rooted graphs. We adopt their framework and perform a
similar analysis for GNNs vs. MLPs. We ﬁrst deﬁne rooted graphs.
Deﬁnition 1 (Rooted Graph) .A rooted graph, denoted as G[i]is a graph with one node iinG[i]
designated as the root. GNNs, GA-MLPs, and MLPs can all be considered as functions on rooted
graphs. The goal of a node-level task on node iwith label yiis to ﬁt a function to the input-output
pairs (G[i],yi).
We denote the space of rooted graphs as E. Following Chen et al. (2021), the expressive power
of a model on graph data is evaluated by its ability to approximate functions on E. This is further
characterized as the number of induced equivalence classes of rooted graphs on E, with the equivalence
relation deﬁned as the following. Given a family of functions FonE, we deﬁne an equivalence
relation'E;Famong all rooted graphs such that 8G[i];G0[j]2E;G[i]'E;FG0[j]if and only if
8f2F;f(G[i]) =f(G0[j]). We now give a proposition to characterize the GNN expressive power
(proof in Appendix E).
Proposition 1. WithXdenotes the set of all possible node features and assuming jXj 2, withm
denotes the maximum node degree and assuming m3, the total number of equivalence classes of
rooted graphs induced by an L-layer GNN is lower bounded by jXj+m 2
m 12L 1.
As shown in Proposition 1, the expressive power of GNNs grows doubly-exponentially in the number
of layersL, which means it grows linearly in Lafter taking log(log()). The expressive power
GA-MLPs only grows exponentially in Las shown in Chen et al. (2021). Under this framework, the
expressive power of MLPs, which corresponds to a 0-layer GA-MLP, is jXj. Since the former is
much larger than the latter, the conclusion will be GNNs are much more expressive than MLPs. The
gap between these two numbers indeed exists, but empirically this gap will only make a difference
whenjXjis small. As in Chen et al. (2021), both the lower bound proof and the constructed examples
showing GNNs are more powerful than GA-MLPs assumed jXj= 2. In real applications and datasets
considered in this work, the node features can be high dimensional vectors like bag-of-words, which
makesjXjenormous. Thus, this gap doesn’t matter much empirically.
16

## Page 17

Published as a conference paper at ICLR 2022
E P ROOF OF THEPROPOSITION 1
To prove Proposition 1, we ﬁrst deﬁne rooted aggregation trees, which is similar to but different from
rooted graphs.
Deﬁnition 2 (Rooted Aggregation Tree) .The depth-K rooted aggregation tree of a rooted graph G[i]
is a depth-K rooted tree with a (possibly many-to-one) mapping from every node in the tree to some
node inG[i], where (i) the root of the tree is mapped to node i, and (ii) the children of every node jin
the tree are mapped to the neighbors of the node in G[i]to whichjis mapped.
A rooted aggregation tree can be obtained by unrolling the neighborhood aggregation steps in the
GNNs. An illustration of rooted graphs and rooted aggregation trees can be found in Chen et al.
(2021) Figure 4. We denote the set of all rooted aggregation trees of depth L using TL. Then we use
TL;X;mto denote a subset of TL, where the node features belong to X, and all the nodes have exactly
degreem(mchildren), and at least two nodes out of these m nodes have different features. In other
words, a node can’t have all identical children. With rooted aggregation trees deﬁned, we are ready to
prove Proposition 1. The proof is adapted from the proof of Lemma 3 in Chen et al. (2021).
Proof. Since the number of equivalence classes on Einduced by the family of all depth-L GNNs
consists of all rooted graphs that share the same rooted aggregation tree of depth-L (Chen et al.,
2021), the lower bound problem in Proposition 1 can be reduced to lower bound jTLj, which can
be further reduced to lower bound the subset jTL;X;mj. We now showjTL;X;mj jXj+m 2
m 12L 1
inductively.
WhenL= 1, the root of the tree can have jXjdifferent choices. For the children nodes, we
pickmfeatures fromjXjand repetitions are allowed. This leads to jXj+m 1
m
cases. Therefore,
TL+1;X;m=jXj jXj+m 1
m
 jXj+m 2
m 1
.
Assuming the statement holds for L, we show it holds for L+ 1by constructing trees in TL+1;X;m
fromT;T02TL;X;m. We do this by assigning node features in Xto themchildren of each leaf
node inTandT0. First note that when TandT0are two non-isomorphic trees, two depth-L+1 trees
constructed from TandT0will be different no matter how the node features are assigned. Now we
consider all the trees can be constructed from Tby assign node features of children to leaf nodes.
We ﬁrst consider all paths from the root to leaves in T. Each path consists of a sequence of nodes
where the node features form a one-to-one mapping to an L-tuple 2f(x1;:::;x L) :xi2Xg .
Leaf nodes are called node underif the path from the root to it corresponds to . The children of
nodes under different s are always distinguishable, and thus any assignments lead to distinct rooted
aggregation trees of depth L+ 1. The assignment of children of nodes under the same , on the other
hand, could be overcounted. Therefore, to lower bound TL+1;X;m, we only consider a special way of
assignments to avoid over counting, which is that children of all nodes under the same are assigned
the same set of features.
Since we assumed that at least two nodes of Thave different features, there are at least 2Ldifferent
s corresponding to the path from the root to leaves. For a leaf node junder a ﬁxed , one of its
children needs to have the same feature as j’s parent node. This restriction is due to the deﬁnition
of rooted aggregation trees. Therefore, we only pick features for the other m 1nodes, which will
be jXj+m 2
m 1
cases for each j. Then through this construction, the total number of depth-L+1 trees
fromTcan be lower bounded by jXj+m 2
m 12L
. Finally, we have this lower bound holds for all
T2TL;X;m, so we deriveTL+1;X;m jXj+m 2
m 12L
TL;X;m, andTL;X;m jXj+m 2
m 1PL
l=12l
=
 jXj+m 2
m 12L 1
17

## Page 18

Published as a conference paper at ICLR 2022
F A DVANCED GNN ARCHITECTURES AS THE TEACHER
In our experiment, SAGE teacher is used throughout to avoid inﬂuence by model architecture. Some
other GNNs like GCN are also considered in the ablation studies, but they are not the best known
architecture for a speciﬁc dataset. To show GLNN has stronger performance given a stronger teacher,
we consider the best teacher we can access on Products . We take MLP+C&S Huang et al. (2021)
from the OGB leaderboard as a new teacher, which has reported accuracy 84.18% and ranks #8 on
the leadarboard as of Nov 2021. We choose MLP+C&S instead of the other top 7 because the others
either rely on raw text (additional info to the given node feature), or require a large GPU with >16GB
memory, which we don’t have access to. Also, their improvement is not super signiﬁcant compared
to MLP+C&S, i.e. 84% to 86%. The result with MLP+C&S teacher is shown in Table 9. We see
that with the new teacher, performance of GLNN+ improves to be even better than SAGE (78.61%),
which shows GLNN can get stronger given a stronger teacher.
Table 9: GLNN+ with MLP+C&S teacher on Products
MLP+C&S MLP+ GLNN +
Acc 84.18 64.50 82.94
GGLNN WITH FEATURE AUGMENTATION FROM ONE -HOP NEIGHBORS
In our main experiment, the inductive performance of GLNN on the Arxiv dataset is less desirable
than others. We thus consider augment the node features with their one-hop neighbors to include
more graph information. This can be seen as a middle ground between pure GLNN s and GNNs. For
this new experiment, we follow the setting in Table 3 but with two new approaches. We explain the
setting of these two approaches below.
1.1-hop GA-MLP: ﬁrstly, for each node v, we collect features of its 1-hop neighbors uto
augment the raw feature of v, i.e.xv!~xv, like in SGC. Then we train an MLP on the
graph with ~xv. Note ifvis in the observed graph but uis in the inductive (unobserved
during training) part, then vdoesn’t collect features from u.
2.1-hop GA-GLNN: Go through the same feature augmentation step as 1-hop GA-MLP. Then
train an MLP with distillation from teacher GNN.
3. In summary, we compare 5 different models in the table below
(a) SAGE: single model on xv
(b) MLP: single model on xv
(c) GLNN: SAGE teacher and MLP student on xv
(d) 1-hop GA-MLP: single model on ~xv
(e) 1-hop GA-GLNN: SAGE teacher on xv, MLP student on ~xv
We show in the table below, with 1-hop neighbor features, performance of GLNN improves a lot.
This is expected as we also observe signiﬁcant improvement from MLP to 1-hop GA-MLP. However,
we indeed see 1-hop GA-GLNN (68.83) can further improve from 1-hop GA-MLP (66.62) and nearly
match the teacher (70.64).
Table 10: GLNN with feature augmentation from one-hop neighbor on Arxiv
Eval SAGE MLP GLNN 1-hop GA-MLP 1-hop GA-GLNN
Arxiv ind 70.64 55.40 60.48 66.62 68.83
tran 70.75 55.28 71.46 66.67 69.82
As we have shown in Figure 3, the 1-Layer GNN in our case is roughly 4 times slower than GLNN
(29.31ms vs. 7.56ms), which should be a good approximation for the speed comparison between
1-hop GA-MLP/GA-GLNN and GLNN. This result is practically beneﬁcial, as it gives practitioners
more ﬂexibility about how much accuracy they want to trade for less inference time.
18

## Page 19

Published as a conference paper at ICLR 2022
H M ODEL PERFORMANCE UNDER DIFFERENT INDUCTIVE SPLIT RATE
This section is a continuation of the ablation study of inductive split rate in Section 6. It generalizes
Figure 5 Middle to more split rates (from 10:90 to 90:10), and explicitly show the inductive and
transductive performance on each dataset. For better visualization, the training data label rate is also
reduced from 20 per class to 5 per class in the following plots.
Figure 7: Model inductive performance comparison between MLP, GNN(SAGE), and GLNN under
different inductive split rate in the production setting.
Figure 8: Model transductive performance comparison between MLP, GNN(SAGE), and GLNN
under different inductive split rate in the production setting.
19

## Page 20

Published as a conference paper at ICLR 2022
IGLNN UNDER NODE FEATURE HETEROGENEITY AND NON-HOMOPHILY
Besides the 7 datasets used in the main experiments, we consider 4 more datasets from Ivanov &
Prokhorenkova (2021) and Lim et al. (2021) to further evaluate GLNN .
TheHouse_class andVK_class datasets are from Ivanov & Prokhorenkova (2021). The node
features of these two graphs are based on tabular data, which have different types, scales, and
meanings as the opposite of the bag-of-word node features in Cora and etc. Some basic statistics of
the datasets are shown in the following table.
Table 11: Statistics of dataset with heterogeneous node features
Dataset # Nodes # Edges # Features # Classes
House_class 20,640 182,146 6 5
VK_class 54,028 213,644 14 7
We apply the GLNN onHouse_class andVK_class using the best BGNN model from Ivanov
& Prokhorenkova (2021) as the teacher. The comparison is shown in the following table. Ivanov &
Prokhorenkova (2021) also includes GAT, GCN, AGNN, and APPNP as baselines, whose performance
on these two datasets are quite similar (difference < 0.025). We compare with these baselines by
including the best result among the 4 GNN models and refer it as GNN in the table below, i.e. GNN
= max(GAT, GCN, AGNN, APPNP). From the table, we see that GLNN can improve from MLP,
outperform GNN and LightGBM, and become competitive to the teacher BGNN.
Table 12: GLNN on datasets with heterogeneous node features. Numbers other than GLNN are taken
from Ivanov & Prokhorenkova (2021)
Dataset LightGBM GNNs BGNN MLP GLNN
House_class 0.55 0.625 0.682 0.534 0.672
VK_class 0.57 0.577 0.683 0.567 0.641
We further pick the non-homophilous Penn94 andPokec datasets from Lim et al. (2021). Some
basic statistics of the datasets are shown in the following table.
Table 13: Statistics of non-homophilous datasets
Dataset # Nodes # Edges # Features # Classes
Penn94 41,536 1,590,655 5 2
Pokec 1,632,803 30,622,564 65 2
Using the GCN teacher, we see that the performance of GLNN is improved over MLP and becomes
competitive to the teacher GCN on Penn94 . However, on Pokec , the simple LINK model can
achieve very good performance, and it is better than most GNNs reported in Lim et al. (2021). LINK
is a purely structural model which does not use node features at all. This shows that the Pokec
dataset corresponds to the setting we discussed in Sec 5.8 (limitations of GLNN ) – if the node labels
can be largely determined by only the graph structure, then GLNN will struggle. We observe that
GLNN is not as good as LINK owing to this limitation. However, we still see that for most of the
non-homophilous datasets, MLPs already work quite well on them, and we can use GLNN for the
other ones like Penn94 .
Table 14: GLNN on non-homophilous datasets. Numbers other than GLNN are taken from Lim et al.
(2021)
Dataset LINK GCN MLP GLNN
Penn94 80.79 82.47 73.61 81.69
Pokec 80.54 75.45 62.37 61.32
20

## Page 21

Published as a conference paper at ICLR 2022
J M ODEL COMPARISON WITH NOISY NODE FEATURES
In Section 6, we conducted an ablation study to compare model performance with noisy node features,
and the result is shown in the left plot in Figure 5. There are two subtle points in this plot. (1)The
performance of GNN is still relatively high for high noisy features, even when = 1and the features
are completely random. (2)For completely random features, the performance of GLNN is still higher
than MLP. We now discuss and explain them in more detail.
GNN Performance on Random Features. GNN still performs well because nodes with the same
labels are likely to be connected and GNN can overﬁt the training data. We explain the detail through
a toy example. Suppose there is a 4-clique containing nodes A, B, C, D in the graph with only a single
edge D-E connects this clique to other graph nodes. Suppose A, B, C, D all have iid random Gaussian
raw features and the same class label c. Let’s pick A to be the inductive test node and assume E and
the triangle formed by B, C, D to be in the training graph. Let’s consider a simple example for 1-layer
GCN and break down message passing into feature aggregation and nonlinear transformation. During
training, GNN can overﬁt the data by learning a nonlinear transformation which maps the aggregated
features of B, C, D to class c. The aggregated features of B and C will just be the average of the
raw features of B, C, D. Although E is also involved in D’s feature aggregation step, the aggregated
features of D will also be very close to this average. Then when test on A, the aggregated feature of
A will likely be classiﬁed to the same class c by the overﬁtted nonlinear transformation because it is
the average of raw node features of A, B, C, D. In this case, GNN can actually correctly classify A
because of the overﬁtting. For GNNs with more layers and graphs with more neighbor nodes, the
conclusion may be generalized.This is roughly sort of a “majority vote” process. For a test node A,
if many nodes, which A collects features from, have the same class label and appear in the training
graph, then A will be classiﬁed as this class by an overﬁtted classiﬁer.
GLNN and MLP Performance on Random Features. The gap between MLP and GLNN is due to
imbalanced datasets. The GLNN can learn the imbalance from soft labels, whereas MLPs can only
access uniformly picked training nodes. We explain more detail using the A-computer dataset as
an example, for which the gap between MLP and GLNN is obvious. The task is 10-class classiﬁcation.
With random node features ( =1), the inductive accuracy for MLP is 0.0652 and 0.2538 for GLNN .
If the data labels are uniform, then both models should give an accuracy around 0.1. However, the
labels on the inductive dataset are actually imbalanced. We show the results in Figure 9. The hist
on the left is the label distribution of the inductive test set. In particular, class 4 takes about 40%.
However, given this imbalance, the standard train-test split selects training nodes uniformly among
labels. In this case, 20 nodes per class. Therefore, the predictions of MLP on random features are
expected to be relatively uniform because the 200 nodes we train it on are uniform. This gives the
hist shown in the middle, where the largest class takes about 17.5%. Finally, for GLNN, we train it
on all the 200 training nodes with hard labels, plus soft labels of other nodes in the observed graph
Gobs(see Section 5.2). Since these extra nodes are selected randomly, whose label distribution is
actually similar to the label distribution on the whole data and the distribution on the inductive test
set. Therefore, we get the GLNN predictions hist on the right. Although for each node, we can’t
assign a prediction correlated to its feature, on average the distribution is very close to the true label
distribution on the inductive test set and has a much higher expectation. In fact, if the prediction
distribution is exactly the true distribution on the inductive test set, the expectation will be 0.2169.
GLNN actually does even a bit better by putting its bet more on the largest class.
Figure 9: Inductive (predicted) label distribution on the A-computer dataset. Left: true labels.
Middle: predicted labels by MLP. Right: predicted labels by GLNN .
21