# Open Graph Benchmark

## Page 1

Open Graph Benchmark:
Datasets for Machine Learning on Graphs
Weihua Hu1, Matthias Fey2, Marinka Zitnik3, Yuxiao Dong4,
Hongyu Ren1, Bowen Liu5, Michele Catasta1, Jure Leskovec1
1Department of Computer Science,5Chemistry, Stanford University
2Department of Computer Science, TU Dortmund University
3Department of Biomedical Informatics, Harvard University
4Microsoft Research, Redmond
ogb@cs.stanford.edu
Steering Committee
Regina Barzilay, Peter Battaglia, Yoshua Bengio, Michael Bronstein,
Stephan Günnemann, Will Hamilton, Tommi Jaakkola, Stefanie Jegelka,
Maximilian Nickel, Chris Re, Le Song, Jian Tang, Max Welling, Rich Zemel
Abstract
We present the OPEN GRAPH BENCHMARK (OGB), a diverse set of challenging
and realistic benchmark datasets to facilitate scalable, robust, and reproducible
graph machine learning (ML) research. OGB datasets are large-scale, encompass
multiple important graph ML tasks, and cover a diverse range of domains, ranging
from social and information networks to biological networks, molecular graphs,
source code ASTs, and knowledge graphs. For each dataset, we provide a uniﬁed
evaluation protocol using meaningful application-speciﬁc data splits and evaluation
metrics. In addition to building the datasets, we also perform extensive benchmark
experiments for each dataset. Our experiments suggest that OGB datasets present
signiﬁcant challenges of scalability to large-scale graphs and out-of-distribution
generalization under realistic data splits, indicating fruitful opportunities for future
research. Finally, OGB provides an automated end-to-end graph ML pipeline that
simpliﬁes and standardizes the process of graph data loading, experimental setup,
and model evaluation. OGB will be regularly updated and welcomes inputs from the
community. OGB datasets as well as data loaders, evaluation scripts, baseline code,
and leaderboards are publicly available at https://ogb.stanford.edu .
Contents
1 Introduction 2
2 Shortcomings of Current Benchmarks 5
3 OGB: Overview 6
4 OGB Node Property Prediction 7
4.1 ogbn-products : Amazon Products Co-purchasing Network . . . . . . . . . . . 8
4.2 ogbn-proteins : Protein-Protein Association Network . . . . . . . . . . . . . 9
4.3 ogbn-arxiv : Paper Citation Network . . . . . . . . . . . . . . . . . . . . . . . 10
4.4 ogbn-papers100M : Paper Citation Network . . . . . . . . . . . . . . . . . . . . 11
4.5 ogbn-mag : Heterogeneous Microsoft Academic Graph (MAG) . . . . . . . . . . 12
34th Conference on Neural Information Processing Systems (NeurIPS 2020), Vancouver, Canada.
1arXiv:2005.00687v7  [cs.LG]  25 Feb 2021

## Page 2

5 OGB Link Property Prediction 13
5.1 ogbl-ppa : Protein-Protein Association Network . . . . . . . . . . . . . . . . . 14
5.2 ogbl-collab : Author Collaboration Network . . . . . . . . . . . . . . . . . . 15
5.3 ogbl-ddi : Drug-Drug Interaction Network . . . . . . . . . . . . . . . . . . . . 16
5.4 ogbl-citation2 : Paper Citation Network . . . . . . . . . . . . . . . . . . . . 17
5.5 ogbl-wikikg2 : Wikidata Knowledge Graph . . . . . . . . . . . . . . . . . . . 18
5.6 ogbl-biokg : Biomedical Knowledge Graph . . . . . . . . . . . . . . . . . . . 19
6 OGB Graph Property Prediction 20
6.1 ogbg-mol *: Molecular Graphs . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
6.2 ogbg-ppa : Protein-Protein Association Network . . . . . . . . . . . . . . . . . 22
6.3 ogbg-code2 : Abstract Syntax Tree of Source Code . . . . . . . . . . . . . . . . 23
7 OGB Package 24
7.1 OGB Data Loaders . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
7.2 OGB Evaluators . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
8 Conclusions 25
A More Benchmark Results on ogbg-mol *Datasets 33
1 Introduction
Graphs are widely used for abstracting complex systems of interacting objects, such as social net-
works (Easley et al., 2010), knowledge graphs (Nickel et al., 2015), molecular graphs (Wu et al.,
2018), and biological networks (Barabasi & Oltvai, 2004), as well as for modeling 3D objects (Si-
monovsky & Komodakis, 2017), manifolds (Bronstein et al., 2017), and source code (Allamanis et al.,
2017). Machine learning (ML), especially deep learning, on graphs is an emerging ﬁeld (Hamilton
et al., 2017b; Bronstein et al., 2017). Recently, signiﬁcant methodological advances have been made
in graph ML (Grover & Leskovec, 2016; Kipf & Welling, 2017; Ying et al., 2018b; Veli ˇckovi ´c
et al., 2019; Xu et al., 2019), which have produced promising results in applications from diverse
domains (Ying et al., 2018a; Zitnik et al., 2018; Stokes et al., 2020).
How can we further advance research in graph ML? Historically, high-quality and large-
scale datasets have played signiﬁcant roles in advancing research, as exempliﬁed by IMA-
GENET(Deng et al., 2009) and MS COCO (Lin et al., 2014) in computer vision, GLUE
BENCHMARK (Wang et al., 2018) and SQUAD (Rajpurkar et al., 2016) in natural lan-
guage processing, and LIBRISPEECH (Panayotov et al., 2015) and CH IME (Barker et al.,
2015) in speech processing. However, in graph ML research, commonly-used datasets
and evaluation procedures present issues that may negatively impact future progress.
  SCALESmall
Medium
Large
Nodes
LinksGraphsNature
InformationSociety  DOMAINS
Figure 1: OGB provides
datasets that are diverse in
scale, domains, and task cat-
egories.Issues with current benchmarks . Most of the frequently-used
graph datasets are extremely small compared to graphs found in
real applications (with more than 1 million nodes or 100 thousand
graphs) (Wang et al., 2020; Ying et al., 2018a; Wu et al., 2018; Hu-
sain et al., 2019; Bhatia et al., 2016; Vrande ˇci´c & Krötzsch, 2014).
For example, the widely-used node classiﬁcation datasets, CORA,
CITESEER, and PUBMED (Yang et al., 2016), only have 2,700 to
20,000 nodes, the popular graph classiﬁcation datasets from the
TU collection (Yanardag & Vishwanathan, 2015; Kersting et al.,
2020) only contain 200 to 5,000 graphs, and the commonly-used
knowledge graph completion datasets, FB15 KandWN18 (Bor-
des et al., 2013), only have 15,000 to 40,000 entities. As models
are extensively developed on these small datasets, the majority of
them turn out to be not scalable to larger graphs (Kipf & Welling,
2017; Velickovic et al., 2018; Bordes et al., 2013; Trouillon et al.,
2016). The small datasets also make it hard to rigorously evaluate
data-hungry models, such as Graph Neural Networks (GNNs) (Li
et al., 2016; Duvenaud et al., 2015; Gilmer et al., 2017; Xu et al.,
2019). In fact, the performance of GNNs on these datasets is often unstable and nearly statistically
2

## Page 3

OGB Graph
Datasets
OGB Data
Loader
Your ML
Model
OGB
Evaluator
OGB
Leaderboards
(a) (b) (c) (d) (e)
Figure 2: Overview of the OGB pipeline: (a) OGB provides realistic graph benchmark datasets
that cover different prediction tasks (node, link, graph), are from diverse application domains, and
are at different scales. (b)OGB fully automates dataset processing and splitting. That is, the OGB
data loaders automatically download and process graphs, provide graph objects (compatible with
PYTORCH1(Paszke et al., 2019) and its associated graph libraries, PYTORCH GEOMETRIC2(Fey &
Lenssen, 2019) and DEEPGRAPH LIBRARY3(Wang et al., 2019a)), and further split the datasets
in a standardized manner. (c)After an ML model is developed, (d)OGB evaluates the model in
a dataset-dependent manner, and outputs the model performance appropriate for the task at hand.
Finally, (e)OGB provides public leaderboards to keep track of recent advances.
identical to each other, due to the small number of samples the models are trained and evaluated
on (Dwivedi et al., 2020; Hu et al., 2020a).
Furthermore, there is no uniﬁed and commonly-followed experimental protocol. Different studies
adopt their own dataset splits, evaluation metrics, and cross-validation protocols, making it challeng-
ing to compare performance reported across various studies (Shchur et al., 2018; Errica et al., 2019;
Dwivedi et al., 2020). In addition, many studies follow the convention of using random splits to
generate training/test sets (Kipf & Welling, 2017; Xu et al., 2019; Bordes et al., 2013), which is not
realistic or useful for real-world applications and generally leads to overly optimistic performance
results (Lohr, 2009).
Thus, there is an urgent need for a comprehensive suite of real-world benchmarks that combine
a diverse set of datasets of various sizes coming from different domains. Data splits as well as
evaluation metrics are important so that progress can be measured in a consistent and reproducible
way. Last but not least, benchmarks also need to provide different types of tasks, such as node
classiﬁcation, link prediction, and graph classiﬁcation.
Present work: OGB . Here, we present the OPEN GRAPH BENCHMARK (OGB) with the goal
of facilitating scalable, robust, and reproducible graph ML research. The premise of OGB is to
develop a diverse set of challenging and realistic benchmark datasets that can empower the rigorous
advancements in graph ML. As illustrated in Figure 1, the OGB datasets are designed to have the
following three characteristics:
1.Large scale. The OGB datasets are orders-of-magnitude larger than existing benchmarks and
can be categorized into three different scales (small, medium, and large). Even the “small” OGB
graphs have more than 100 thousand nodes or more than 1 million edges, but are small enough to
ﬁt into the memory of a single GPU, making them suitable for testing computationally intensive
algorithms. Additionally, OGB introduces “medium” (more than 1 million nodes or more than 10
million edges) and “large” (on the order of 100 million nodes or 1 billion edges) datasets, which
can facilitate the development of scalable models based on mini-batching and distributed training.
2.Diverse domains. The OGB datasets aim to include graphs that are representative of a wide
range of domains, as shown in Table 1. The broad coverage of domains in OGB empowers the
development and demonstration of general-purpose models, and can be used to distinguish them
from domain-speciﬁc techniques. Furthermore, for each dataset, OGB adopts domain-speciﬁc
data splits ( e.g., based on time, species, molecular structure, GitHub project, etc.) that are more
realistic and meaningful than conventional random splits.
3.Multiple task categories. Besides data diversity, OGB supports three categories of fundamental
graph ML tasks, i.e., node, link, and graph property predictions, each of which requires the models
to make predictions at different levels of graphs, i.e., at the level of a node, link, and entire graph,
respectively.
1https://pytorch.org
2https://pytorch-geometric.readthedocs.io
3https://www.dgl.ai
3

## Page 4

Table 1: Overview of currently-available OGB datasets (denoted in green). Nature domain
includes biological networks and molecular graphs, Society domain includes academic graphs and
e-commerce networks, and Information domain includes knowledge graphs. More datasets will be
added in the future to increase the coverage.
TaskNode property prediction
ogbn-
Domain Nature Society Information
Small arxiv
Medium proteins products mag
Large papers100M
TaskLink property prediction
ogbl-
Domain Nature Society Information
Small ddi collab biokg
Medium ppa citation2 wikikg2
Large
TaskGraph property prediction
ogbg-
Domain Nature Society Information
Small molhiv
Medium molpcba /ppa code2
Large
The currently-available OGB datasets are categorized in Table 1 according to their task categories,
application domains, and scales. Currently, OGB includes 15 diverse graph datasets, with at
least 4 datasets for each task category. All the datasets are constructed by ourselves, except for
ogbn-products ,ogbg-molpcba , and ogbg-molhiv , whose graphs and target labels are
adopted from Chiang et al. (2019) and Wu et al. (2018). For these datasets, we resolve critical issues
of the existing data splits by presenting more meaningful and standardized splits.
In addition to building the graph datasets, we also perform extensive benchmark experiments for
each dataset. Through the experiments and ablation studies, we highlight research challenges and
opportunities provided by each dataset, especially on (1) scaling models to large graphs, and (2)
improving out-of-distribution generalization performance under the realistic data split scenarios.
Finally, as illustrated in Figure 2, OGB presents an automated end-to-end graph ML pipeline
that simpliﬁes and standardizes the process of graph data loading, experimental setup, and model
evaluation, in the same spirit as OpenML (Vanschoren et al., 2013; Feurer et al., 2019). Speciﬁcally,
given the OGB datasets (a), the end-users can focus on developing their graph ML models (c) by
using the OGB data loaders (b) and evaluators (d), both of which are provided by our OGB Python
package4. OGB also hosts a public leaderboard5(e) for publicizing state-of-the-art, reproducible
graph ML research. As a starting point, for each dataset, we include results from a suite of well-known
baselines, particularly GNN-based methods, together with code to reproduce our results.
OGB is an on-going open-source, community-driven initiative. Over time we plan to release new
versions of the datasets and methods, and provide updates to the leaderboard. The OGB web-
4https://github.com/snap-stanford/ogb
5https://ogb.stanford.edu/docs/leader_overview
4

## Page 5

site (https://ogb.stanford.edu ) provides the documentation, example scripts, and public
leaderboard. We also welcome inputs from the community at ogb@cs.stanford.edu .
2 Shortcomings of Current Benchmarks
We ﬁrst review commonly-used graph benchmarks and discuss the current state of the ﬁeld. We
organize the discussion around three categories of graph ML tasks: predictions at the level of nodes,
links, and graphs.
Node property prediction . Currently, the three graphs ( CORA,CITESEERandPUBMED) proposed
in Yang et al. (2016) have been widely used as semi-supervised node classiﬁcation datasets, particu-
larly for evaluating GNNs. The sizes of these graphs are rather small, ranging from 2,700 to 20,000
nodes. Recent studies suggest that datasets at this small scale can be solved quite well with simple
GNN architectures (Shchur et al., 2018; Wu et al., 2019), and the performance of different GNNs on
these datasets is nearly statistically identical (Dwivedi et al., 2020; Hu et al., 2020a). Furthermore,
there is no consensus on the splitting procedures for these datasets, which makes it hard to fairly
compare different model designs (Shchur et al., 2018). Finally, a recent study (Zou et al., 2020)
shows that these datasets have some fundamental data quality issues. For example, in CORA, 42% of
the nodes leak information between their features and labels, and 1% of the nodes are duplicated.
The situation for C ITESEER is even worse, with leakage rates of 62% and duplication rates of 5%.
Some recent works in graph ML have proposed relatively large datasets, such as PPI(56,944 nodes),
REDDIT (334,863 nodes) (Hamilton et al., 2017b) and AMAZON 2M(2,449,029 nodes) (Chiang et al.,
2019). However, there exist some inherent issues with the proposed data splits. Speciﬁcally, 83%,
65% and 90% of the nodes are used for training in the PPI,REDDIT andAMAZON 2Mdatasets,
respectively, which results in an artiﬁcially small distribution shift across the training/validation/test
sets. Consequently, as may be expected, the performance improvements on these benchmarks have
quickly saturated. For example, recent GNN models (Chiang et al., 2019; Zeng et al., 2020) can
already yield F1 scores of 99.5 for PPIand 97.0 for REDDIT , and 90.4% accuracy for AMAZON 2M,
with extremely small generalization gaps between training and test accuracy. Finally, it is also
practically required for graph ML models to handle web-scale graphs (beyond 100 million nodes and
1 billion edges) in industrial applications (Ying et al., 2018a). However, to date, there have been no
publicly available graph datasets of such scale with label information.
In summary, several factors ( e.g., size, leakage, splits, etc.) associated with the current use of existing
datasets make them unsuitable as benchmark datasets for graph ML.
Link property prediction . Broadly, there are two lines of efforts for the link-level task: link
prediction in homogeneous networks (Liben-Nowell & Kleinberg, 2007; Zhang & Chen, 2018) and
relation completion in (heterogeneous) knowledge graphs (Bordes et al., 2013; Nickel et al., 2015;
Hu et al., 2020b). There are several problems with the current benchmark datasets in these areas.
First, representative datasets are either extremely small or do not come with input node features.
For link prediction, while the well-known recommender system datasets used in Berg et al. (2017)
include node features, their sizes are very small, with the largest having only 6,000 nodes. On the
other hand, although the Open Academic Graph (OAG) used in Qiu et al. (2019) comprises tens of
millions of nodes, there are no associated node features. Regarding the knowledge graph completion,
the widely-used dataset, FB15 K, is very small, containing only 14,951 entities, which is a tiny subset
of the original Freebase knowledge graph with more than 50 million entities (Bollacker et al., 2008).
Second, similar to the node-level task, random splits are predominantly used in link-level predic-
tion (Bordes et al., 2013; Grover & Leskovec, 2016). The random splits are not realistic in many
practical applications such as friend recommendation in social networks, in which test edges (friend
relations after a certain timestamp) naturally follow a different distribution from training edges (friend
relations before a certain timestamp).
Finally, the existing datasets are mostly oriented to applications in recommender systems, social
media and knowledge graphs, in which the graphs are typically very sparse. This may result in
techniques specialised for sparse link inference that are not generalizable to domains with dense
graphs, such as the protein-protein association graphs and drug-drug interaction networks typically
found in biology and medicine (Szklarczyk et al., 2019; Wishart et al., 2018; Davis et al., 2019;
Szklarczyk et al., 2016; Piñero et al., 2020). Very recently, Sinha et al. (2020) proposed a synthetic
5

## Page 6

link prediction benchmark to diagnose model’s logical generalization capability. Their focus is on
synthetic tasks, which is complementary to OGB that focuses on realistic tasks.
Graph property prediction . Graph-level prediction tasks are found in important applications in
natural sciences, such as predicting molecular properties in chemistry (Duvenaud et al., 2015; Gilmer
et al., 2017; Hu et al., 2020a), where molecules are naturally represented as molecular graphs.
In graph classiﬁcation, the most widely-used graph-level datasets from the TU collection (Kersting
et al., 2020) are known to have many issues, such as small sizes ( i.e., most of the datasets only
contain less than 1,000 graphs),6unrealistic settings ( e.g., no bond features for molecules), random
data splits, inconsistent evaluation protocols, and isomorphism bias (Ivanov et al., 2019). A very
recent attempt (Dwivedi et al., 2020) to address these issues mainly focuses on benchmarking ML
models, speciﬁcally the building blocks of GNNs, rather than developing application-oriented realistic
datasets. In fact, ﬁve out of the six proposed datasets are synthetic.
Recent work in graph ML (Hu et al., 2020a; Ishiguro et al., 2019) has started to adopt MOLECU -
LENET(Wu et al., 2018) which contains a set of realistic and large-scale molecular property prediction
datasets. However, there is limited consensus in the dataset splitting and molecular graph features,
making it hard to compare different models in a fair manner. OGB adopts the MOLECULE NET
datasets, while providing uniﬁed dataset splits as well as the molecular graph features that are found
to provide favorable performance over naïve features.
Beyond molecular graphs, OGB also provides graphs from other domains, such as biological networks
and Abstract Syntax Tree (AST) representations of source code. These types of graphs exhibit distinct
characteristics from molecular graphs, enabling the evaluation of the versatility of graph ML models.
3 OGB: Overview
The goal of OGB is to support and catalyze research in graph ML, which is a fast-growing and
increasingly important area. OGB datasets cover a variety of real-world applications and span several
important domains. Furthermore, OGB provides a common codebase using popular deep learning
frameworks for loading, constructing, and representing graphs as well as code implementations of
established performance metrics for fast model evaluation and comparison.
In the subsequent sections (Sections 4, 5, and 6), we describe in detail each of the datasets in
OGB, focusing on the properties of the graph(s), the prediction task, and the dataset splitting
scheme. The currently-available datasets are summarized in Table 2. We also compare datasets from
diverse application domains by inspecting their basic graph statistics, e.g., node degree, clustering
coefﬁcient, and diameter, as summarized in Table 3. We show that OGB datasets exhibit diverse
graph characteristics. The difference in graph characteristics results in the inherent difference in
how information propagates in the graphs, which can signiﬁcantly affect the behavior of many graph
ML models such as GNNs and random-walk-based node embeddings (Xu et al., 2018). Overall, the
diversity in graph characteristics originates from the diverse application domains and is crucial to
evaluate the versatility of graph ML models.
In the same sections, we additionally present an extensive benchmark analysis for each dataset,
using representative node embedding models, GNNs, as well as recently-introduced mini-batch-
based GNNs. We discuss our initial ﬁndings, and highlight research challenges and opportunities
in: (1) scaling models to large graphs, and (2) improving out-of-distribution generalization under
the realistic data splits. We repeat each experiment 10 times using different random seeds, and
report the mean and unbiased standard deviation of all training and test results corresponding to
the best validation results. All code to reproduce our baseline experiments is publicly available at
https://github.com/snap-stanford/ogb/tree/master/examples and is meant
as a starting point to accelerate further research on our proposed datasets. We refer the interested
reader to our code base for the details of model architectures and hyper-parameter settings.
Finally, in Section 7, we brieﬂy explain the usage of our OGB Python package that can be readily
installed via pip. We demonstrate how the OGB package makes the pipeline shown in Figure 2
easily accessible by providing the automatic data loaders and evaluators. The package is publicly
6Recently, some progress has been made to increase the dataset sizes: http://graphlearning.io .
Nevertheless, most of them are still small compared to the OGB datasets, and evaluation protocols are not
standardized.
6

## Page 7

Table 2: Summary of currently-available OGB datasets. An OGB dataset, e.g.,ogbg-molhiv ,
is identiﬁed by its preﬁx ( ogbg- ) and its name ( molhiv ). The preﬁx speciﬁes the category of the
graph ML task, i.e., node ( ogbn- ), link ( ogbl- ), or graph ( ogbg- ) property prediction. A realistic
split scheme is provided for each dataset, whose detail can be found in Sections 4, 5, and 6.
Category NameNode EdgeDirected Hetero #TasksSplit Split TaskMetricFeat. Feat. Scheme Ratio Type
Node
ogbn-products " – – – 1 Sales rank 8/2/90 Multi-class class. Accuracy
proteins –" – – 112 Species 65/16/19 Binary class. ROC-AUC
arxiv " –" – 1 Time 54/18/28 Multi-class class. Accuracy
papers100M " –" – 1 Time 78/8/14 Multi-class class. Accuracy
mag " " " " 1 Time 85/9/6 Multi-class class. Accuracy
Link
ogbl-ppa " – – – 1 Throughput 70/20/10 Link prediction Hits@100
collab " – – – 1 Time 92/4/4 Link prediction Hits@50
ddi – – – – 1 Protein target 80/10/10 Link prediction Hits@20
citation2 " –" – 1 Time 99/1/1 Link prediction MRR
wikikg2 –" " – 1 Time 94/3/3 KG completion MRR
biokg –" " " 1 Random 94/3/3 KG completion MRR
Graph
ogbg-molhiv " " – – 1 Scaffold 80/10/10 Binary class. ROC-AUC
molpcba " " – – 128 Scaffold 80/10/10 Binary class. AP
ppa –" – – 1 Species 49/29/22 Multi-class class. Accuracy
code2 " " " – 1 Project 90/5/5 Sub-token prediction F1 score
Table 3: Statistics of currently-available OGB datasets. The ﬁrst 3 statistics are calculated over
raw training/validation/test graphs. The last 4 graph statistics are calculated over the ‘standardized’
training graphs, where the graphs are ﬁrst converted into undirected and unlabeled homogeneous
graphs with duplicated edges removed. The SNAP library (Leskovec & Sosi ˇc, 2016) is then used to
compute the graph statistics, where the graph diameter is approximated by performing BFS from
1,000 randomly-sampled nodes. The MaxSCC ratio represents the fraction of nodes in the largest
strongly connected component of the graph.
Category Name #GraphsAverage Average Average Average MaxSCC Graph
#Nodes #Edges Node Deg. Clust. Coeff. Ratio Diameter
Node
ogbn-products 1 2,449,029 61,859,140 50.5 0.411 0.974 27
proteins 1 132,534 39,561,252 597.0 0.280 1.000 9
arxiv 1 169,343 1,166,243 13.7 0.226 1.000 23
papers100M 1 111,059,956 1,615,685,872 29.1 0.085 1.000 25
mag 1 1,939,743 21,111,007 21.7 0.098 1.000 6
Link
ogbl-ppa 1 576,289 30,326,273 73.7 0.223 0.999 14
collab 1 235,868 1,285,465 8.2 0.729 0.987 22
ddi 1 4,267 1,334,889 500.5 0.514 1.000 5
citation2 1 2,927,963 30,561,187 20.7 0.178 0.996 21
wikikg2 1 2,500,604 17,137,181 12.2 0.168 1.000 26
biokg 1 93,773 5,088,434 47.5 0.409 0.999 8
Graph
ogbg-molhiv 41,127 25.5 27.5 2.2 0.002 0.993 12.0
molpcba 437,929 26.0 28.1 2.2 0.002 0.999 13.6
ppa 158,100 243.4 2,266.1 18.3 0.513 1.000 4.8
code2 452,741 125.2 124.2 2.0 0.0 1.000 13.5
available at https://github.com/snap-stanford/ogb , and its documentation can be
found at https://ogb.stanford.edu .
4 OGB Node Property Prediction
We currently provide 5 datasets, adopted from 3 different application domains, for predicting the prop-
erties of individual nodes. Speciﬁcally, ogbn-products is an Amazon products co-purchasing
network (Bhatia et al., 2016) originally developed by Chiang et al. (2019) ( cf.Section 4.1). The
ogbn-arxiv ,ogbn-mag , and ogbn-papers100M datasets are extracted from the Microsoft
Academic Graph (MAG) (Wang et al., 2020), with different scales, tasks, and include both ho-
mogeneous and heterogeneous graphs. Speciﬁcally, ogbn-arxiv is a paper citation network of
arXiv papers ( cf.Section 4.3), ogbn-mag is a heterogeneous academic graph containing differ-
7

## Page 8

ent node types (papers, authors, institutions, and topics) and their relations ( cf.Section 4.5), and
ogbn-papers100M is an extremely large paper citation network from the entire MAG with more
than 100 million nodes and 1 billion edges ( cf.Section 4.4). The ogbn-proteins dataset is a
protein-protein association network (Szklarczyk et al., 2019) ( cf.Section 4.2).
The 5 datasets exhibit highly diverse graph statistics, as shown in Table 3. Notably, the biological
network, ogbn-proteins , is much denser than the social/information networks as can be observed
from its large average node degree and small graph diameter. On the other hand, the co-purchasing
network, ogbn-products , has more clustered graph structure than the other datasets, as can
be seen from its large average clustering coefﬁcient. Finally, the heterogeneous academic graph,
ogbn-mag , exhibits rather interesting graph characteristics, simultaneously having small average
node degree, clustering coefﬁcient, and graph diameter.
Baselines . We consider the following representative models as our baselines unless otherwise
speciﬁed.
•MLP : A multilayer perceptron (MLP) predictor that uses the given raw node features
directly as input. Graph structure information is not utilized.
•NODE2VEC: An MLP predictor that uses as input the concatenation of the raw node
features and N ODE2VECembeddings (Grover & Leskovec, 2016; Perozzi et al., 2014).
•GCN : Full-batch Graph Convolutional Network (Kipf & Welling, 2017).
•GRAPH SAGE : Full-batch GraphSAGE (Hamilton et al., 2017a), where we adopt the mean
pooling variant and a simple skip connection to preserve central node features.
•NEIGHBOR SAMPLING (optional): A mini-batch training technique of GNNs (Hamilton
et al., 2017a) that samples neighborhood nodes when performing aggregation.
•CLUSTER GCN (optional): A mini-batch training technique of GNNs (Chiang et al., 2019)
that partitions the graphs into a ﬁxed number of subgraphs and draws mini-batches from
them.
•GRAPH SAINT (optional): A mini-batch training technique of GNNs (Zeng et al., 2020)
that samples subgraphs via a random walk sampler.
The three mini-batch GNN training, NEIGHBOR SAMPLING ,CLUSTER GCN , and GRAPH SAINT ,
are explored only for graph datasets where full-batch GCN /GRAPH SAGE did not ﬁt into the common
GPU memory size of 11GB. The mini-batch GNNs are more GPU memory-efﬁcient than the full-
batch GNNs because they ﬁrst partition and sample the graph into subgraphs. Hence, in order to
train the network, they require only a small amount of nodes to be loaded into the GPU memory at
each mini-batch. Inference is then performed on the whole graph without GPU usage. To choose
the architecture for the mini-batch GNNs, we ﬁrst run full-batch GCN andGRAPH SAGE on an
NVIDIA Quadro RTX 8000 with 48GB of memory, and then adopt the best performing full-batch
GNN architecture for the mini-batch GNNs. All models are trained with a ﬁxed hidden dimensionality
of 256, a ﬁxed number of two or three layers, and a tuned dropout ratio 2f0:0;0:5g.
4.1 ogbn-products : Amazon Products Co-purchasing Network
Theogbn-products dataset is an undirected and unweighted graph, representing an Amazon
product co-purchasing network (Bhatia et al., 2016). Nodes represent products sold in Amazon, and
edges between two products indicate that the products are purchased together. The graphs, target
labels, and node features are generated following Chiang et al. (2019), where node features are
dimensionality-reduced bag-of-words of the product descriptions. Our contribution, when adopting
the dataset in OGB, is to resolve its critical data split issue by presenting a more realistic and
challenging split (see below).
Prediction task . The task is to predict the category of a product in a multi-class classiﬁcation setup,
where the 47 top-level categories are used for target labels.
Dataset splitting . We consider a more challenging and realistic dataset splitting that differs from
the one used in Chiang et al. (2019). Instead of randomly assigning 90% of the nodes for training
and 10% of the nodes for testing (without a validation set), we use the sales ranking (popularity) to
split nodes into training/validation/test sets. Speciﬁcally, we sort the products according to their sales
ranking and use the top 8% for training, next top 2% for validation, and the rest for testing. This is a
more challenging splitting procedure that closely matches the real-world application where manual
labeling is prioritized to important nodes in the network and ML models are subsequently used to
make predictions on less important ones.
8

## Page 9

Table 4: Results for ogbn-products .
yRequires a GPU with 33GB of memory.
MethodAccuracy (%)
Training Validation Test
MLP 84.03 0.93 75.540.14 61.060.08
NODE2VEC 93.390.10 90.320.06 72.490.10
GCNy93.560.09 92.000.03 75.640.21
GRAPH SAGEy94.090.05 92.240.07 78.500.14
NEIGHBOR SAMPLING 92.960.07 91.700.09 78.700.36
CLUSTER GCN 93.75 0.13 92.120.09 78.970.33
GRAPH SAINT 92.71 0.14 91.620.08 79.080.24
Train Validation Test
Figure 3: T-SNE visualization of training/validation/test nodes in ogbn-products .
Discussion . Our benchmarking results in Table 4 show that the highest test performances are
attained by GNNs, while the MLP baseline that solely relies on a product’s description is not
sufﬁcient for accurately predicting the category of a product. Even with the GNNs, we observe
the huge generalization gap7, which can be explained by differing node distributions across the
splits, as visualized in Figure 3. This is in stark contrast with the conventional random split used by
Chiang et al. (2019). Even with the same split ratio (8/2/90), we ﬁnd GRAPH SAGE already achieves
88.200.08% test accuracy with only 1percentage points of generalization gap. These results
indicate that the realistic split is much more challenging than the random split and offer an important
opportunity to improve out-of-distribution generalization.
Table 4 also shows that the recent mini-batch-based GNNs8give promising results, even slightly
outperforming the full-batch version of GRAPH SAGE that does not ﬁt into ordinary GPU memory.
The improved performance can be attributed to the regularization effects of mini-batch noise and
edge dropout (Rong et al., 2020b). Nevertheless, the mini-batch GNNs have been much less explored
compared to the full-batch GNNs due to the prevalent use of the extremely small benchmark datasets
such as CORA andCITESEER. As a result, many important questions remain open, e.g., what
mini-batch training methods can induce the best regularization effect, and how to allow mini-batch
training for advanced GNNs that rely on large receptive-ﬁeld sizes (Xu et al., 2018; Klicpera et al.,
2019; Li et al., 2019), since the current mini-batch methods are rather limited by the number of nodes
from which they aggregate information. Overall, ogbn-products is an ideal benchmark dataset
for the ﬁeld to move beyond the extremely small graph datasets and to catalyze the development of
scalable mini-batch-based graph models with improved out-of-distribution prediction accuracy.
4.2 ogbn-proteins : Protein-Protein Association Network
Theogbn-proteins dataset is an undirected, weighted, and typed (according to species) graph.
Nodes represent proteins, and edges indicate different types of biologically meaningful associations
between proteins, e.g., physical interactions, co-expression or homology (Szklarczyk et al., 2019;
7Deﬁned by the difference between training and test accuracy.
8The G RAPH SAGE architecture is used for neighbor aggregation.
9

## Page 10

Table 5: Results for ogbn-proteins .
MethodROC-AUC (%)
Training Validation Test
MLP 81.78 0.48 77.060.14 72.040.48
NODE2VEC 79.761.88 70.070.53 68.810.65
GCN 82.77 0.16 79.210.18 72.510.35
GRAPH SAGE 87.86 0.13 83.340.13 77.680.20
Consortium, 2018). All edges come with 8-dimensional features, where each dimension represents
the approximate conﬁdence of a single association type and takes on values between 0 and 1 (the
larger the value is, the more conﬁdent we are about the association). The proteins come from 8
species.
Prediction task . The task is to predict the presence of protein functions in a multi-label binary
classiﬁcation setup, where there are 112 kinds of labels to predict in total. The performance is
measured by the average of ROC-AUC scores across the 112 tasks.
Dataset splitting . We split the protein nodes into training/validation/test sets according to the species
which the proteins come from. This enables the evaluation of the generalization performance of the
model across different species.
Discussion . The ogbn-proteins dataset does not have input node features9, but has edge features
on more than 30 million edges. In our baseline experiments, we opt for simplicity and use the average
edge features of incoming edges as node features.
Benchmarking results are shown in Table 5. Surprisingly, simple MLPs10perform better than more
sophisticated approaches like NODE2VECandGCN . Only GRAPH SAGE is able to outperform the
naïve MLP approach, which indicates that central node information (that is not explicitly modeled
in GCN in its message-passing) already contains a lot of crucial information for making correct
predictions.
We further evaluate the best performing GRAPH SAGE on conventional random split with the same
split ratio as the species split. On the random split, we ﬁnd the generalization gap is extremely low,
with 87.830.03% test ROC-AUC that is only 0.27 percentage points lower than the training ROC-
AUC (88.100.01%). This is in contrast to 10.18 percentage points of generalization gap (training
AUC minus test AUC) in the species split, as calculated from the GRAPH SAGE experiment in Table
5. The result suggests the unique challenge of across-species generalization that needs to be tackled
in future research.
Since the number of nodes in ogbn-proteins is fairly small and easily ﬁt onto common GPUs, we
did not run the CLUSTER GCN andGRAPH SAINT experiments. Nonetheless, this dataset presents
an interesting research question of how to utilize edge features in a more sophisticated way than just
naïve averaging, e.g., by the usage of attention or by treating the graph as a multi-relational graph (as
there are 8 different association types between proteins). The challenge is to scalably handle the huge
number of edge features efﬁciently on GPU, which might require clever graph partitioning based on
the edge weights.
4.3 ogbn-arxiv : Paper Citation Network
Theogbn-arxiv dataset is a directed graph, representing the citation network between all Com-
puter Science (CS) ARXIVpapers indexed by MAG (Wang et al., 2020). Each node is an ARXIV
paper and each directed edge indicates that one paper cites another one. Each paper comes with a
128-dimensional feature vector obtained by averaging the embeddings of words in its title and abstract.
The embeddings of individual words are computed by running the WORD 2VEC model (Mikolov
9In our preliminary experiments, we used one-hot encodings of species ID as node features, but that did not
work well empirically, which can be explained by the fact that the species ID is used for dataset splitting.
10Note that the input features here are graph-aware in some sense, because they are obtained by averaging the
incoming edge features.
10

## Page 11

Table 6: Results for ogbn-arxiv .
MethodAccuracy (%)
Training Validation Test
MLP 63.58 0.81 57.650.12 55.500.23
NODE2VEC 76.430.81 71.290.13 70.070.13
GCN 78.87 0.66 73.000.17 71.740.29
GRAPH SAGE 82.35 1.51 72.770.16 71.490.27
et al., 2013) over the MAG corpus. In addition, all papers are also associated with the year that the
corresponding paper was published.
Prediction task . The task is to predict the 40 subject areas of ARXIVCS papers,11e.g.,cs.AI ,
cs.LG , and cs.OS , which are manually determined ( i.e., labeled) by the paper’s authors and ARXIV
moderators. With the volume of scientiﬁc publications doubling every 12 years over the past
century (Dong et al., 2017b), it is practically important to automatically classify each publication’s
areas and topics. Formally, the task is to predict the primary categories of the ARXIVpapers, which
is formulated as a 40-class classiﬁcation problem.
Dataset splitting . The previously-used Cora, CiteSeer, and PubMed citation networks are split
randomly (Yang et al., 2016). In contrast, we consider a realistic data split based on the publication
dates of the papers. The general setting is that the ML models are trained on existing papers and then
used to predict the subject areas of newly-published papers, which supports the direct application of
them into real-world scenarios, such as helping the ARXIVmoderators. Speciﬁcally, we propose to
train on papers published until 2017, validate on those published in 2018, and test on those published
since 2019.
Discussion . Our initial benchmarking results are shown in Table 6, where the directed graph is
converted to an undirected one for simplicity. First, we observe that the naïve MLP baseline that does
not utilize any graph information is signiﬁcantly outperformed by the other three models that utilize
graph information. This suggests that graph information can dramatically improve the performance
of predicting a paper’s category. Comparing models that do utilize graph information, we ﬁnd GNN
models, i.e.,GCN andGRAPH SAGE , slightly outperform the NODE2VECmodel. We also conduct
additional experiments on conventional random split with the same split ratio. On the random split,
we ﬁnd that GCN achieves 73.54 0.13% test accuracy, suggesting that the realistic time split is indeed
more challenging than the random split, providing an opportunity to improve the out-of-distribution
generalization performance. Furthermore, we think it will be fruitful to explore how the edge direction
information as well as the node temporal information (e.g., year in which papers are published) can
be taken into account to improve prediction performance.
4.4 ogbn-papers100M : Paper Citation Network
Theogbn-papers100M dataset is a directed citation graph of 111 million papers indexed by
MAG (Wang et al., 2020). Its graph structure and node features are constructed in the same way as
ogbn-arxiv in Section 4.3. Among its node set, approximately 1.5 million of them are ARXIV
papers, each of which is manually labeled with one of ARXIV’s subject areas ( cf.Section 4.3). Overall,
this dataset is orders-of-magnitude larger than any existing node classiﬁcation datasets.
Prediction task . Given the full ogbn-papers100M graph, the task is to predict the subject
areas of the subset of papers that are published in ARXIV. The majority of nodes (corresponding
to non- ARXIVpapers) are not associated with label information, and only their node features and
reference information are given. The task is to leverage the entire citation network to infer the labels
of the ARXIVpapers.12In total, there are 172 ARXIVsubject areas, making the prediction task a
172-class classiﬁcation problem.
Dataset splitting . The splitting strategy is the same as that used in ogbn-arxiv ,i.e., the time-
based split. Speciﬁcally, the training nodes (with labels) are all ARXIVpapers published until 2017,
11https://arxiv.org/corr/subjectclasses
12In practice, the trained models can also be used to predict labels of even non- ARXIVpapers.
11

## Page 12

Table 7: Results for ogbn-papers100M .
MethodAccuracy (%)
Training Validation Test
MLP 54.84 0.43 49.600.29 47.240.31
SGC 67.54 0.43 66.480.20 63.290.19
while the validation nodes are the ARXIVpapers published in 2018, and the models are tested on
ARXIVpapers published since 2019.
Discussion . Our initial benchmarking results are shown in Table 7, where the directed graph is
converted to an undirected one for simplicity. As most existing models have difﬁculty handling such a
gigantic graph, we benchmark the two simplest models,13MLP andSGC (Wu et al., 2019), which is
a simpliﬁed variant of textscGCN (Kipf & Welling, 2017) that essentially pre-processes node features
using graph adjacency information. We obtain SGC node embeddings on the CPU (requiring more
than 100GB of memory), after which we train the ﬁnal MLP with mini-batches on an ordinary GPU.
We see from Table 7 that the graph-based model, SGC , despite its simplicity, performs much better
than the naïve MLP baseline. Nevertheless, we observe severe underﬁtting of SGC , indicating that
using more expressive GNNs is likely to improve both training and test accuracy. It is therefore
fruitful to explore how to scale expressive and advanced GNNs to the gigantic Web-scale graph,
going beyond the simple pre-processing of node features. Overall, ogbn-papers100M is by far
the largest benchmark dataset for node classiﬁcation over a homogeneous graph, and is meant to
signiﬁcantly push the scalability of graph models.
4.5 ogbn-mag : Heterogeneous Microsoft Academic Graph (MAG)
Theogbn-mag dataset is a heterogeneous network composed of a subset of the Microsoft Academic
Graph (MAG) (Wang et al., 2020). It contains four types of entities—papers (736,389 nodes),
authors (1,134,649 nodes), institutions (8,740 nodes), and ﬁelds of study (59,965 nodes)—as well
as four types of directed relations connecting two types of entities—an author “is afﬁliated with”
an institution,14an author “writes” a paper, a paper “cites” a paper, and a paper “has a topic of” a
ﬁeld of study. Similar to ogbn-arxiv described in Section 4.3, each paper is associated with a
128-dimensional WORD 2VEC feature vector, and all the other types of entities are not associated with
input node features.
Prediction task . Given the heterogeneous ogbn-mag data, the task is to predict the venue (confer-
ence or journal) of each paper, given its content, references, authors, and authors’ afﬁliations. This is
of practical interest as some manuscripts’ venue information is unknown or missing in MAG, due to
the noisy nature of Web data. In total, there are 349 different venues in ogbn-mag , making the task
a 349-class classiﬁcation problem.
Dataset splitting . We follow the same time-based strategy as ogbn-arxiv and
ogbn-papers100M to split the paper nodes in the heterogeneous graph, i.e., training models
to predict venue labels of all papers published before 2018, validating and testing the models on
papers published in 2018 and since 2019, respectively.
Discussion . As ogbn-mag is a heterogeneous graph, we consider slightly different sets of GNN
and node embedding baselines. Speciﬁcally, for GCN andGRAPH SAGE , as they are originally
designed for homogeneous graphs, we apply the models over the homogeneous subgraph, retaining
only paper nodes and their citation relations. We also consider the RELATIONAL -GCN (R-GCN )
(Schlichtkrull et al., 2018) that is speciﬁcally designed for heterogeneous graphs and uses specialized
message passing parameters for different edge types. Since only “paper” nodes come with node
features, we use trainable embeddings for the remaining nodes. For the node embedding model,
instead of NODE2VEC, we adopt METAPATH 2VEC(Dong et al., 2017a), as it is speciﬁcally designed
for heterogeneous graphs. For each relation, e.g., an author “writes” a paper, the reverse relation, e.g.,
a paper “is written by” an author, is added to allow bidirectional message passing in GNNs.
13NODE2VECis omitted as it is computationally costly on such a gigantic graph.
14For each author, we include all the institutions that the author has ever belonged to.
12

## Page 13

Table 8: Results for ogbn-mag .
yRequires a GPU with 14GB of memory.
MethodAccuracy (%)
Training Validation Test
MLP 28.33 0.20 26.260.16 26.920.26
GCN 29.71 0.19 29.530.22 30.430.25
GRAPH SAGE 30.79 0.19 30.700.19 31.530.15
METAPATH2VEC 38.351.39 35.060.17 35.440.36
R-GCNy75.874.19 40.840.41 39.770.46
NEIGHBOR SAMPLING 68.537.27 47.610.68 46.780.67
CLUSTER GCN 79.65 4.12 38.400.31 37.320.37
GRAPH SAINT 79.64 1.70 48.370.26 47.510.22
Our benchmarking results are shown in Table 8. First, we see that MLP ,GCN , and GRAPH SAGE
perform worse than the models that actually utilize heterogeneous graph information, i.e.,METAP-
ATH2VEC,R-GCN , and the mini-batch GNNs.15This highlights that exploiting the heterogeneous
nature of the graph is essential to achieving good performance on this dataset.
Second, we see that the mini-batch GNNs, especially NEIGHBOR SAMPLING andGRAPH SAINT ,
give surprisingly promising results, outperforming the full-batch R-GCN by a large margin. This is
likely due to the regularization effect of the noise induced by mini-batch sampling and edge dropout
(Rong et al., 2020b). In contrast, CLUSTER GCN gives worse performance than its full-batch variant,
indicating that the bias introduced by the pre-computed partitioning has a negative effect on the
model’s performance (as can be also seen by its highly overﬁtting training performance).
Nevertheless, heterogeneous graph models as well as their mini-batch training methods have been
much less explored compared to their homogeneous counterparts, due to the smaller number of
established benchmarks. Overall, ogbn-mag is meant to catalyze the development of scalable and
accurate heterogeneous graph models, going beyond homogeneous graphs. A fruitful research direc-
tion is to adopt advanced techniques developed for homogeneous graphs to improve the performance
on heterogeneous graphs.
5 OGB Link Property Prediction
We currently provide 6 datasets, adopted from diverse application domains for predicting the prop-
erties of links (pairs of nodes). Speciﬁcally, ogbl-ppa is a protein-protein association network
(Szklarczyk et al., 2019) ( cf.Section 5.1), ogbl-collab is an author collaboration network (Wang
et al., 2020) ( cf.Section 5.2), ogbl-ddi is a drug-drug interaction network (Wishart et al., 2018)
(cf.Section 5.3), ogbl-citation2 is a paper citation network (Wang et al., 2020) ( cf.Section 5.4),
ogbl-biokg is a heterogeneous knowledge graph compiled from a large number of biomedical
repositories ( cf.Section 5.6), and ogbl-wikikg2 is a Wikidata knowledge graph (Vrande ˇci´c &
Krötzsch, 2014) ( cf.Section 5.5).
The different datasets are highly diverse in their graph structure, as shown in Table 3. For ex-
ample, the biological networks ( ogbl-ppa andogbl-ddi ) are much denser than the academic
networks ( ogbl-collab andogbl-citation2 ) and the knowledge graphs ( ogbl-wikikg2
andogbl-biokg ), as can be seen from the large average node degree, small number of nodes,
and the small graph diameter. On the other hand, the collaboration network, ogbl-collab , has
more clustered graph structure than the other datasets, as can be seen from its high average clustering
coefﬁcient. Comparing the two knowledge graph datasets, ogbl-wikikg2 andogbl-biokg , we
see that the former is much more sparse and less clustered than the latter.
Baselines . We implement different sets of baselines for link prediction datasets that only have a
single edge type, and KG completion datasets that have multiple edge/relation types.
15The R-GCN architecture is used for neighbor aggregation.
13

## Page 14

Baselines for link prediction datasets . We consider the following representative models as our
baselines for the link prediction datasets unless otherwise speciﬁed. For all models, edge features
are obtained by using the Hadamard operator between pair-wise node embeddings, and are then
inputted to an MLP for the ﬁnal prediction. During training, we randomly sample edges and use
them as negative examples. We use the same number of negative edges as there are positive edges.
Below, we describe how each model obtains node embeddings:
•MLP : Input node features are directly used as node embeddings.
•NODE2VEC: The node embeddings are obtained by concatenating input features and
NODE2VECembeddings (Grover & Leskovec, 2016; Perozzi et al., 2014).
•GCN : The node embeddings are obtained by full-batch Graph Convolutional Networks
(GCN) (Kipf & Welling, 2017).
•GRAPH SAGE : The node embeddings are obtained by full-batch GraphSAGE (Hamilton
et al., 2017a), where we adopt its mean pooling variant and a simple skip connection to
preserve central node features.
•MATRIX FACTORIZATION : The distinct embeddings are assigned to different nodes and
are learned in an end-to-end manner together with the MLP predictor.
•NEIGHBOR SAMPLING (optional): A mini-batch training technique of GNNs (Hamilton
et al., 2017a) that samples neighborhood nodes when performing aggregation.
•CLUSTER GCN (optional): A mini-batch training technique of GNNs (Chiang et al., 2019)
that partitions the graphs into a ﬁxed number of subgraphs and draws mini-batches from
them.
•GRAPH SAINT (optional): A mini-batch training technique of GNNs (Zeng et al., 2020)
that samples subgraphs via a random walk sampler.
Similar to the node property prediction baselines, the mini-batch GNN training,
NEIGHBOR SAMPLING ,CLUSTER GCN , and GRAPH SAINT , are experimented only for graph
datasets where full-batch GCN andGRAPH SAGE did not ﬁt into the common GPU memory size of
11GB. To choose the GNN architecture for the mini-batch GNNs, we ﬁrst run full-batch GCN and
GRAPH SAGE on a NVIDIA Quadro RTX 8000 with 48GB of memory, and then adopt the best
performing full-batch GNN architecture for the mini-batch GNNs. All models are trained with a ﬁxed
hidden dimensionality of 256, a ﬁxed number of three layers, and a tuned dropout ratio 2f0:0;0:5g.
Baselines for KG completion datasets . We consider the following representative KG embedding
models as our baselines for the KG datasets unless otherwise speciﬁed.
•TRANS E: Translation-based KG embedding model by Bordes et al. (2013).
•DISTMULT: Multiplication-based KG embedding model by Yang et al. (2015).
•COMPL EX: Complex-valued multiplication-based KG embedding model by Trouillon et al.
(2016).
•ROTAT E: Rotation-based KG embedding model by Sun et al. (2019).
For KGs with many entities and relations, the embedding dimensionality can be limited by the
available GPU memory, as the embeddings need to be loaded into GPU all at once. We therefore
choose the dimensionality such that training can be performed on a ﬁxed-budget of GPU memory.
Our training procedure follows Sun et al. (2019), where we perform negative sampling and use
margin-based logistic loss for the loss function.
5.1 ogbl-ppa : Protein-Protein Association Network
Theogbl-ppa dataset is an undirected, unweighted graph. Nodes represent proteins from 58
different species, and edges indicate biologically meaningful associations between proteins, e.g.,
physical interactions, co-expression, homology or genomic neighborhood (Szklarczyk et al., 2019).
We provide a graph object constructed from training edges ( i.e., no validation and test edges are
contained). Each node contains a 58-dimensional one-hot feature vector that indicates the species
that the corresponding protein comes from.
Prediction task . The task is to predict new association edges given the training edges. The evaluation
is based on how well a model ranks positive test edges over negative test edges. Speciﬁcally, we rank
each positive edge in the validation/test set against 3,000,000 randomly-sampled negative edges, and
count the ratio of positive edges that are ranked at the K-th place or above (Hits@ K). We found
K= 100 to be a good threshold to rate a model’s performance in our initial experiments. Overall,
14

## Page 15

Table 9: Results for ogbl-ppa .
MethodHits@100 (%)
Training Validation Test
MLP 0.46 0.00 0.460.00 0.460.00
NODE2VEC 24.430.92 22.530.88 22.260.83
GCN 19.89 1.51 18.451.40 18.671.32
GRAPH SAGE 18.53 2.85 17.242.64 16.552.40
MATRIX FACTORIZATION 81.659.15 32.284.28 32.290.94
this metric is much more challenging than ROC-AUC because the model needs to consistently rank
the positive edges higher than nearly all the negative edges.
Dataset splitting . We provide a biological throughput split of the edges into training/valida-
tion/test edges. Training edges are protein associations that are measured experimentally by a
high-throughput technology ( e.g., cost-effective, automated experiments that make large scale rep-
etition feasible (Macarron et al., 2011; Bajorath, 2002; Younger et al., 2017)) or are obtained
computationally ( e.g., via text-mining). In contrast, validation and test edges contain protein asso-
ciations that can only be measured by low-throughput, resource-intensive experiments performed
in laboratories. In particular, the goal is to predict a particular type of protein association, e.g.,
physical protein-protein interaction, from other types of protein associations ( e.g., co-expression,
homology, or genomic neighborhood) that can be more easily measured and are known to correlate
with associations that we are interested in.
Discussion . Our initial benchmarking results are shown in Table 9. First, the MLP baseline16
performs extremely poorly, which is to be expected since the node features are not rich in this dataset.
Surprisingly, both GNN baselines ( GCN ,GRAPH SAGE ) and NODE2VECfail to overﬁt on the
training data and show similar performances across training/validation/test splits. The poor training
performance of GNNs suggests that positional information, which cannot be captured by GNNs alone
(You et al., 2019), might be crucial to ﬁt training edges and obtain meaningful node embeddings.
On the other hand, we see that MATRIX FACTORIZATION , which learns a distinct embedding for
each node (thus, it can express any positional information of nodes), is indeed able to overﬁt on
the training data, while also reaching better validation and test performance. However, the poor
generalization performance still encourages the development of new research ideas to close this gap,
e.g., by injecting positional information into GNNs or by developing more sophisticated negative
sampling techniques.
5.2 ogbl-collab : Author Collaboration Network
Theogbl-collab dataset is an undirected graph, representing a subset of the collaboration network
between authors indexed by MAG (Wang et al., 2020). Each node represents an author and edges
indicate the collaboration between authors. All nodes come with 128-dimensional features, obtained
by averaging the word embeddings of papers that are published by the authors. All edges are
associated with two types of meta-information: the year and the edge weight, representing the number
of co-authored papers published in that year. The graph can be viewed as a dynamic multi-graph
since there can be multiple edges between two nodes if they collaborate in more than one year.
Prediction task . The task is to predict the author collaboration relationships in a particular year given
the past collaborations. As the task is a time-series problem, it is natural for models to incorporate
the most recent edge information to make prediction, e.g., use validation edges when predicting test
edges. The evaluation metric is similar to ogbl-ppa in Appendix 5.1, where we would like the
model to rank true collaborations higher than false collaborations. Speciﬁcally, we rank each true
collaboration among a set of 100,000 randomly-sampled negative collaborations, and count the ratio
of positive edges that are ranked at K-place or above (Hits@ K). We found K= 50 to be a good
threshold in our preliminary experiments.
Dataset splitting . We split the data according to time, in order to simulate a realistic application in
collaboration recommendation. Speciﬁcally, we use the collaborations until 2017 as training edges,
those in 2018 as validation edges, and those in 2019 as test edges.
16Here we obtain node embeddings by applying a linear layer to the raw one-hot node features.
15

## Page 16

Table 10: Results for ogbl-collab .
MethodUse most Hits@50 (%)
recent edges Training Validation Test
MLP % 45.701.66 24.021.45 19.271.29
NODE2VEC % 99.730.36 57.030.52 48.880.54
GCN % 84.281.78 52.631.15 44.751.07
GRAPH SAGE % 93.580.59 56.880.77 48.100.81
MATRIX FACTORIZATION % 100.000.00 48.960.29 38.860.29
GCN " 84.281.78 52.631.15 47.141.45
GRAPHS AGE " 93.580.59 56.880.77 54.631.12
Discussion . Our initial benchmarking results are shown in Table 10. First, we consider the con-
ventional setting where validation edges are used only for model selection. From the upper half
of Table 10, we see that NODE2VECachieves the best results, followed by the two GNN models
andMATRIX FACTORIZATION . This can be explained by the fact that positional information, i.e.,
past collaborations, is a much more indicative feature for predicting future collaboration than solely
relying on the average paper representations of authors, i.e., the same research interest. Notably,
MATRIX FACTORIZATION achieves nearly perfect training results, but cannot transfer the good results
to the validation and test splits, even when heavy regularization is applied. Overall, it is fruitful to
explore injecting positional information into GNNs, and develop better regularization methods. This
dataset further provides a unique research opportunity for dynamic multi-graphs. To demonstrate
the potential beneﬁt of time-series modeling, we use the same GCN andGRAPH SAGE models as
before but at test time, we additionally incorporate the most recent edges ( i.e., validation edges) as
input to the models. From the lower half of Table 10, we see that the test performances of both
GNN models increase signiﬁcantly by using validation edges at the inference time. One promising
direction to further increase the performance is to treat edges at different timestamps differently, as
recent collaborations may be more indicative about the future collaborations than the past ones.
5.3 ogbl-ddi : Drug-Drug Interaction Network
Theogbl-ddi dataset is a homogeneous, unweighted, undirected graph, representing the drug-drug
interaction network (Wishart et al., 2018). Each node represents an FDA-approved or experimental
drug. Edges represent interactions between drugs and can be interpreted as a phenomenon where
the joint effect of taking the two drugs together is considerably different from the expected effect in
which drugs act independently of each other.
Prediction task . The task is to predict drug-drug interactions given information on already known
drug-drug interactions. The evaluation metric is similar to ogbl-collab discussed in Section 5.2,
where we would like the model to rank true drug interactions higher than non-interacting drug pairs.
Speciﬁcally, we rank each true drug interaction among a set of approximately 100,000 randomly-
sampled negative drug interactions, and count the ratio of positive edges that are ranked at K-place
or above (Hits@ K). We found K= 20 to be a good threshold in our preliminary experiments.
Dataset splitting . We develop a protein-target split , meaning that we split drug edges according
to what proteins those drugs target in the body. As a result, the test set consists of drugs that
predominantly bind to different proteins from drugs in the train and validation sets. This means that
drugs in the test set work differently in the body, and have a rather different biological mechanism of
action than drugs in the train and validation sets. The protein-target split thus enables us to evaluate
to what extent the models can generate practically useful predictions (Guney, 2017), i.e., non-trivial
predictions that are not hindered by the assumption that there exist already known and very similar
medications available for training.
Discussion . Our initial benchmarking results are shown in Table 11. Since ogbl-ddi does not
contain any node features, we omit the graph-agnostic MLP baseline for this experiment. Furthermore,
forGCN andGRAPH SAGE , node features are also represented as distinct embeddings and learned
in an end-to-end manner together with the GNN parameters.
16

## Page 17

Table 11: Results for ogbl-ddi .
MethodHits@20 (%)
Training Validation Test
NODE2VEC 37.821.35 32.921.21 23.262.09
GCN 63.95 2.17 55.502.08 37.075.07
GRAPH SAGE 72.24 0.45 62.620.37 53.904.74
MATRIX FACTORIZATION 56.5613.88 33.702.64 13.684.75
Interestingly, both the GNN models and the MATRIX FACTORIZATION approach achieve signiﬁcantly
higher training results than NODE2VEC. However, only the GNN models are able to transfer this
performance to the test set to some extent, suggesting that relational information is crucial to allow
the model to generalize to unseen interactions. Notably, most of the models show high performance
variance, which can be partly attributed to the dense nature of the graph and the challenging data
split. We further perform the conventional random split of edges, where we ﬁnd GRAPH SAGE is
able to achieve 80.88 2.42% test Hits@20. This indicates that the protein-target split is indeed more
challenging than the conventional random split. Overall, ogbl-ddi presents a unique challenge of
predicting out-of-distribution links in dense graphs.
5.4 ogbl-citation2 : Paper Citation Network
Theogbl-citation2 dataset17is a directed graph, representing the citation network between
a subset of papers extracted from MAG (Wang et al., 2020). Similar to ogbn-arxiv in Section
4.3, each node is a paper with 128-dimensional WORD 2VEC features that summarizes its title and
abstract, and each directed edge indicates that one paper cites another. All nodes also come with
meta-information indicating the year the corresponding paper was published.
Prediction task . The task is to predict missing citations given existing citations. Speciﬁcally, for
each source paper, two of its references are randomly dropped, and we would like the model to rank
the missing two references higher than 1,000 negative reference candidates. The negative references
are randomly-sampled from all the previous papers that are not referenced by the source paper. The
evaluation metric is Mean Reciprocal Rank (MRR), where the reciprocal rank of the true reference
among the negative candidates is calculated for each source paper, and then the average is taken over
all source papers.
Dataset splitting . We split the edges according to time, in order to simulate a realistic application in
citation recommendation ( e.g., a user is writing a new paper and has already cited several existing
papers, but wants to be recommended additional references). To this end, we use the most recent
papers (those published in 2019) as the source papers for which we want to recommend the references.
For each source paper, we drop twopapers from its references—the resulting two dropped edges
(pointing from the source paper to the dropped papers) are used respectively for validation and testing.
All the rest of the edges are used for training.
Discussion . Our initial benchmarking results are shown in Table 12, where the directed graph
is converted to an undirected one for simplicity. Here, the GNN models achieve the best results,
followed by MATRIX FACTORIZATION andNODE2VEC. Among the GNNs, GCN performs better
than GRAPH SAGE . However, these GNNs use full-batch training; thus, they are not scalable
and require more than 40GB of GPU memory to train, which is intractable on most of the GPUs
available today. Hence, we also experiment with the scalable mini-batch training techniques of
GNNs, NEIGHBOR SAMPLING ,CLUSTER GCN , and GRAPH SAINT . Interestingly, we see from
Table 12 that these techniques give worse performance than their full-batch counterpart, which is
in contrast to the node classiﬁcation datasets ( e.g.,ogbn-products andogbn-mag ), where the
mini-batch-based models give stronger generalization performances. This limitation presents a unique
challenge for applying the mini-batch techniques to link prediction, differently from those pertaining
to node prediction. Overall, ogbl-citation2 provides a research opportunity to further improve
GNN models and their scalable mini-batch training techniques in the context of link prediction.
17The older version ogbl-citation has been deprecated due to a bug in negative samples of validation
and test sets.
17

## Page 18

Table 12: Results for ogbl-citation2 .
yRequires a GPU with 40GB of memory
MethodMRR
Training Validation Test
MLP 0.2884 0.0014 0.28910.0012 0.28950.0014
NODE2VEC 0.70040.0012 0.61240.0011 0.61410.0011
GCNy0.90920.0019 0.84790.0023 0.84740.0021
GRAPH SAGEy0.89700.0056 0.82630.0033 0.82600.0036
MATRIX FACTORIZATION 0.91850.0170 0.51810.0436 0.51860.0443
NEIGHBOR SAMPLING 0.86450.0015 0.80540.0009 0.80440.0010
CLUSTER GCN 0.8749 0.0035 0.79940.0025 0.80040.0025
GRAPH SAINT 0.8682 0.0026 0.79750.0039 0.79850.0040
5.5 ogbl-wikikg2 : Wikidata Knowledge Graph
Theogbl-wikikg2 dataset18is a Knowledge Graph (KG) extracted from the Wikidata knowledge
base (Vrande ˇci´c & Krötzsch, 2014). It contains a set of triplet edges (head, relation, tail), capturing
the different types of relations between entities in the world, e.g.,Canadacitizen    ! Hinton . We retrieve
all the relational statements in Wikidata and ﬁlter out rare entities. Our KG contains 2,500,604 entities
and 535 relation types.
Prediction task . The task is to predict new triplet edges given the training edges. The evaluation
metric follows the standard ﬁltered metric widely used in KGs (Bordes et al., 2013; Yang et al., 2015;
Trouillon et al., 2016; Sun et al., 2019). Speciﬁcally, we corrupt each test triplet edges by replacing
its head or tail with randomly-sampled 1,000 negative entities (500 for head and 500 for tail), while
ensuring the resulting triplets do not appear in KG. The goal is to rank the true head (or tail) entities
higher than the negative entities, which is measured by Mean Reciprocal Rank (MRR).
Dataset splitting . We split the triplets according to time, simulating a realistic KG completion
scenario that aims to ﬁll in missing triplets that are not present at a certain timestamp. Speciﬁcally,
we downloaded Wikidata at three different time stamps19(May, August, and November of 2015), and
constructed three KGs where we only retain entities and relation types that appear in the earliest May
KG. We use the triplets in the May KG for training, and use the additional triplets in the August and
November KGs for validation and test, respectively. Note that our dataset split is different from the
existing Wikidata KG dataset that adopts a conventional random split (Wang et al., 2019b), which
does not reﬂect the practical usage.
Discussion . Our benchmark results are provided in Table 13, where the upper-half baselines are
implemented on a single commodity GPU with 11GB memory, while the bottom-half baselines are
implemented on a high-end GPU with 45GB memory.20Training MRR in Table 13 is an unﬁltered
metric,21as ﬁltering is computationally expensive for the large number of training triplets.
First, we see from the upper-half of Table 13 that when the limited embedding dimensionality is
used, COMPL EXperforms the best among the four baselines. With the increased dimensionality,
all four models are able to achieve higher MRR on training, validation and test sets, as seen from
the bottom-half of Table 13. This suggests the importance of using a sufﬁcient large embedding
18The older version ogbl-wikikg has been deprecated due to a bug in negative samples of validation and
test sets.
19Available at https://archive.org/search.php?query=creator%3A%22Wikidata+
editors%22
20Given a ﬁxed 11GB GPU memory budget, we adopt 100-dimension embeddings for DISTMULT and
TRANS E. Since ROTAT EandCOMPL EXrequire the entity embeddings with the real and imaginary parts, we
train these two models with the dimensionality of 50 for each part. On the other hand, on the high-end GPU with
45GB memory, we are able to train all the models with 5larger embedding dimensionality.
21This means that the training MRR is computed by ranking against randomly-selected negative entities
without ﬁltering out triplets that appear in KG. The unﬁltered metric has the systematic bias of being smaller
than the ﬁltered counterpart (computed by ranking against “true” negative entities, i.e., the resulting triplets do
not appear in the KG) (Bordes et al., 2013).
18

## Page 19

Table 13: Results for ogbl-wikikg2 .
yRequires a GPU with 45GB of memory.
MethodMRR
Training (Unﬁltered) Validation (Filtered) Test (Filtered)
TRANS E 0.3408 0.0044 0.24650.0020 0.26220.0045
DISTMULT 0.41150.0077 0.31500.0088 0.34470.0082
COMPL EX 0.45730.0035 0.35340.0052 0.38040.0022
ROTAT E 0.3464 0.0015 0.22500.0035 0.25300.0034
TRANS E (5dim)y0.61740.0026 0.42720.0030 0.42560.0030
DISTMULT(5dim)y0.43500.0038 0.35060.0042 0.37290.0045
COMPL EX(5dim)y0.47600.0030 0.37590.0016 0.40270.0027
ROTAT E (5dim)y0.61110.0032 0.43530.0028 0.43320.0025
dimensionality to achieve good performance in this dataset. Interestingly, although TRANS Eand
ROTAT Eunderperform with the limited dimensionality, they obtain the best performances with the
increased dimensionality. Nevertheless, the extremely low test MRR22suggests that our realistic KG
completion dataset is highly non-trivial. It presents a realistic generalization challenge of discovering
new triplets based on existing ones, which necessitates the development of KG models with more
robust and generalizable reasoning capability. Furthermore, this dataset presents an important
challenge of effectively scaling embedding models to large KGs—naïvely training KG embedding
models with reasonable dimensionality would require a high-end GPU, which is extremely costly
and not scalable to even larger KGs. A promising approach to improve scalability is to distribute
training across multiple commodity GPUs (Zheng et al., 2020; Zhu et al., 2019; Lerer et al., 2019). A
different approach is to share parameters across entities and relations, so that a smaller number of
embedding parameters need to be put onto the GPU memory at once.
5.6 ogbl-biokg : Biomedical Knowledge Graph
Theogbl-biokg dataset is a Knowledge Graph (KG), which we created using data from a large
number of biomedical data repositories. It contains 5 types of entities: diseases (10,687 nodes),
proteins (17,499), drugs (10,533 nodes), side effects (9,969 nodes), and protein functions (45,085
nodes). There are 51 types of directed relations connecting two types of entities, including 39 kinds of
drug-drug interactions, 8 kinds of protein-protein interaction, as well as drug-protein, drug-side effect,
drug-protein, function-function relations. All relations are modeled as directed edges, among which
the relations connecting the same entity types ( e.g., protein-protein, drug-drug, function-function) are
always symmetric, i.e., the edges are bi-directional.
This dataset is relevant to both biomedical and fundamental ML research. On the biomedical side,
the dataset allows us to get better insights into human biology and generate predictions that can
guide downstream biomedical research. On the fundamental ML side, the dataset presents challenges
in handling a noisy, incomplete KG with possible contradictory observations. This is because the
ogbl-biokg dataset involves heterogeneous interactions that span from the molecular scale ( e.g.,
protein-protein interactions within a cell) to whole populations ( e.g., reports of unwanted side effects
experienced by patients in a particular country). Further, triplets in the KG come from sources with
a variety of conﬁdence levels, including experimental readouts, human-curated annotations, and
automatically extracted metadata.
Prediction task . The task is to predict new triplets given the training triplets. The evaluation protocol
is exactly the same as ogbl-wikikg2 in Section 5.5, except that here we only consider ranking
against entities of the same type . For instance, when corrupting head entities of the protein type, we
only consider negative protein entities.
22Note that our test MRR on ogbl-wikikg2 is computed using only 500 negative entities per triplet, which
is much less than the number of negative entities used to compute MRR in the existing KG datasets, such
asFB15 KandFB15 K-237 (around 15,000 negative entitiesf). Nevertheless, ROTAT Egives either lower or
comparable test MRR on ogbl-wikikg2 compared to FB15 Kand FB15 K-237 (Sun et al., 2019).
19

## Page 20

Table 14: Results for ogbl-biokg .
MethodMRR
Training (Unﬁltered) Validation (Filtered) Test (Filtered)
TRANS E 0.5145 0.0005 0.74560.0003 0.74520.0004
DISTMULT 0.52500.0006 0.80550.0003 0.80430.0003
COMPL EX 0.53150.0006 0.81050.0001 0.80950.0007
ROTAT E 0.5363 0.0007 0.79970.0002 0.79890.0004
Dataset splitting . For this dataset, we adopt a random split. While splitting the triplets according to
time is an attractive alternative, we note that it is incredibly challenging to obtain accurate information
as to when individual experiments and observations underlying the triplets were made. We strive to
provide additional dataset splits in future versions of the OGB.
Discussion . Our benchmark results are provided in Table 14, where we adopt 2000-dimensional
embeddings for DISTMULT andTRANS E, and 1000-dimensional embeddings for the real and
imaginary parts of ROTAT EandCOMPL EX. Negative sampling is performed only over entities of the
same types. Similar to Table 13 in Section 5.5, training MRR in Table 14 is an unﬁltered metric.23
Among the four models, COMPL EXachieves the best test MRR, while TRANS Egives signiﬁcantly
worse performance compared to the other models. The worse performance of TRANS Ecan be
explained by the fact that TRANS Ecannot model symmetric relations (Trouillon et al., 2016) that are
prevalent in this dataset, e.g., protein-protein and drug-drug relations are all symmetric. Overall, it
is of great practical interest to further improve the model performance. A promising direction is to
develop a more specialized method for the heterogeneous knowledge graph, where multiple node
types exist and the entire graph follows the pre-deﬁned schema.
6 OGB Graph Property Prediction
We currently provide 4 datasets, adopted from 3 distinct application domains, for predicting the
properties of entire graphs or subgraphs. Speciﬁcally, ogbg-molhiv andogbg-molpcba are
molecular graphs originally curated by Wu et al. (2018) ( cf.Section 6.1), ogbg-ppa is a set of
protein-protein association subgraphs (Zitnik et al., 2019) ( cf.Section 6.2), and ogbg-code2 is a
collection of ASTs of source code (Husain et al., 2019) ( cf.Section 6.3).
The different datasets are highly diverse in their graph structure, as shown in Table 3. For example,
compared with the other graph datasets, the biological subgraphs, ogbg-ppa , have much larger
number of nodes per graph, as well as much denser and clustered graph structure, as seen by the large
average node degree, large average clustering coefﬁcient, and large graph diameter.
This is contrast to the molecular graphs, ogbg-molhiv andogbg-molpcba , as well as the ASTs,
ogbg-code2 , both of which are tree-like graphs—in fact, ASTs are exactly trees—with small
average node degrees, small average clustering coefﬁcient, and large average graph diameter. Despite
the similarity, the molecular graphs and the ASTs are distinct in that the ASTs have much larger
number of nodes with well-deﬁned root nodes.
Baselines . We consider the following representative GNNs as our baselines unless otherwise speciﬁed.
GNNs are used to obtain node embeddings, which are then pooled to give the embedding of the entire
graph. Finally, a linear model is applied to the graph embedding to make predictions.
•GCN : Graph Convolutioanl Networks (Kipf & Welling, 2016).
•GCN+ VIRTUAL NODE :GCN that performs message passing over augmented graphs
with virtual nodes, i.e., additional nodes that are connected to all nodes in the original
graphs (Gilmer et al., 2017; Li et al., 2017; Ishiguro et al., 2019).
•GIN : Graph Isomorphism Network (Xu et al., 2019).
23In Table 14, training MRR is lower than validation and test MRR because it is an unﬁltered metric (computed
by ranking against randomly-selected negative entities), and is expected to give systematically lower MRR than
the ﬁltered metric (computed by ranking against “true” negative entities, i.e., the resulting triplets do not appear
in the KG).
20

## Page 21

Table 15: Results for ogbg-molhiv .
MethodAdditional Virtual ROC-AUC (%)
Features Node Training Validation Test
GCN% " 88.651.01 83.730.78 74.181.22
" % 88.652.19 82.041.41 76.060.97
" " 90.074.69 83.840.91 75.991.19
GIN% " 93.892.96 84.11.05 75.21.30
" % 88.642.54 82.320.90 75.581.40
" " 92.733.80 84.790.68 77.071.49
•GIN+ VIRTUAL NODE :GIN that performs message passing over augmented graphs with
virtual nodes.
To include edge features, we follow Hu et al. (2020a) and add transformed edge features into the
incoming node features. For all the experiments, we use 5-layer GNNs, average graph pooling, a
hidden dimensionality of 300, and a tuned dropout ratio 2f0:0;0:5g.
6.1 ogbg-mol *: Molecular Graphs
Theogbg-molhiv andogbg-molpcba datasets are two molecular property prediction datasets
of different sizes: ogbg-molhiv (small) and ogbg-molpcba (medium). They are adopted from
theMOLECULE NET(Wu et al., 2018), and are among the largest of the MOLECULE NETdatasets.
Besides the two main molecule datasets, we also provide the 10 other MOLECULE NETdatasets,
which are summarized and benchmarked in Appendix A. These datasets can be used to stress-test
molecule-speciﬁc methods (Yang et al., 2019; Jin et al., 2020) and transfer learning (Hu et al.,
2020a). All the molecules are pre-processed using RDK IT(Landrum et al., 2006). Each graph
represents a molecule, where nodes are atoms, and edges are chemical bonds. Input node features
are 9-dimensional, containing atomic number and chirality, as well as other additional atom features
such as formal charge and whether the atom is in the ring. Input edge features are 3-dimensional,
containing bond type, bond stereochemistry as well as an additional bond feature indicating whether
the bond is conjugated. Note that the above additional features are not needed to uniquely identify
molecules, and are not adopted in the previous work (Hu et al., 2020a; Ishiguro et al., 2019). In
the experiments, we perform an ablation study on the molecule features and ﬁnd that including the
additional features improves generalization performance.
Prediction task . The task is to predict the target molecular properties as accurately as possible,
where the molecular properties are cast as binary labels, e.g., whether a molecule inhibits HIV
virus replication or not. For evaluation metric, we closely follow Wu et al. (2018). Speciﬁcally, for
ogbg-molhiv , we use ROC-AUC for evaluation. For ogbg-molpcba , as the class balance is
extremely skewed (only 1.4% of data is positive) and the dataset contains multiple classiﬁcation tasks,
we use the Average Precision (AP) averaged over the tasks as the evaluation metric.24
Dataset splitting . We adopt the scaffold splitting procedure that splits the molecules based on their
two-dimensional structural frameworks. The scaffold splitting attempts to separate structurally differ-
ent molecules into different subsets, which provides a more realistic estimate of model performance
in prospective experimental settings. The scaffold splitting was originally proposed by Wu et al.
(2018) and has been adopted by the follow-up works (Yang et al., 2019; Hu et al., 2020a; Ishiguro
et al., 2019; Rong et al., 2020a); however, the precise implementation differs signiﬁcantly across
works, making their results not directly comparable to each other. In OGB, we aim to standardize the
scaffold split by adopting its most challenging version where test molecules are maximally diverse.
Discussion . Benchmarking results are given in Tables 15 and 16. We see that GIN with the additional
features and VIRTUAL NODES provides the best performance in the two datasets. In Appendix A, we
show that even for the other MOLECULE NETdatasets, the additional features consistently improve
24Wu et al. (2018) originally used a closely-related metric, PRC (Precision Recall Curve)-AUC, but Davis &
Goadrich (2006) showed that AP is more appropriate to summarize the non-convex nature of PRC.
21

## Page 22

Table 16: Results for ogbg-molpcba .
MethodAdditional Virtual AP (%)
Features Node Training Validation Test
GCN% " 36.250.71 23.880.22 22.910.37
" % 28.040.58 20.590.33 20.200.24
" " 38.250.50 24.950.42 24.240.34
GIN% " 45.700.61 27.540.25 26.610.17
" % 37.050.31 23.050.27 22.660.28
" " 46.960.57 27.980.25 27.030.23
generalization performance. In OGB, we therefore include the additional node/edge features in our
molecular graphs.
We further report the performance on the random splitting, keeping the split ratio the same as the
scaffold splitting. We ﬁnd the random split to be much easier than scaffold split. On random splits
ofogbg-molhiv andogbg-molpcba , the best GIN achieves the ROC-AUC of 82.73 2.02%
(5.66 percentage points higher than scaffold) and AP of 34.40 0.90% (7.37 percentage points higher
than scaffold), respectively. The same trend holds true for the other MOLECULE NETdatasets, e.g.,
the best GIN performance on the random split of ogbg-moltox21 is 86.031.37%, which is 8.46
percentage points higher than that of the best GIN for the scaffold split (77.57 0.62% ROC-AUC).
These results highlight the challenges of the scaffold split compared to the random split, and opens up
a fruitful research opportunity to increase the out-of-distribution generalization capability of GNNs.
6.2 ogbg-ppa : Protein-Protein Association Network
Theogbg-ppa dataset is a set of undirected protein association neighborhoods extracted from
the protein-protein association networks of 1,581 different species (Szklarczyk et al., 2019) that
cover 37 broad taxonomic groups ( e.g., mammals, bacterial families, archaeans) and span the tree
of life (Hug et al., 2016). To construct the neighborhoods, we randomly selected 100 proteins
from each species and constructed 2-hop protein association neighborhoods centered on each of the
selected proteins (Zitnik et al., 2019). We then removed the center node from each neighborhood
and subsampled the neighborhood to ensure the ﬁnal protein association graph is small enough (less
than 300 nodes). Nodes in each protein association graph represent proteins, and edges indicate
biologically meaningful associations between proteins. The edges are associated with 7-dimensional
features, where each element takes a value between 0 and 1 and represents the approximate conﬁdence
of a particular type of protein protein association such as gene co-occurrence, gene fusion events, and
co-expression.
Prediction task . Given a protein association neighborhood graph, the task is a 37-way multi-class
classiﬁcation to predict what taxonomic group the graph originates from. The ability to successfully
tackle this problem has implications for understanding the evolution of protein complexes across
species (De Juan et al., 2013), the rewiring of protein interactions over time (Sharan et al., 2005; Zitnik
et al., 2019), the discovery of functional associations between genes even for otherwise rarely-studied
organisms (Cowen et al., 2017), and would give us insights into key bioinformatics tasks, such as
biological network alignment (Malod-Dognin et al., 2017).
Dataset splitting . Similar to ogbn-proteins in Section 4.2, we adopt the species split , where
the neighborhood graphs in validation and test sets are extracted from protein association networks of
species that are notseen during training but belong to one of the 37 taxonomic groups. This split
stress-tests the model’s capability to extract graph features that are essential to the prediction of the
taxonomic groups, which is important for biological understanding of protein associations.
Discussion . Benchmarking results are given in Table 17. Interestingly, similar to the ogbg-mol *
datasets, GIN with VIRTUAL NODE provides the best performance. Nevertheless, the generalization
gap is huge (almost 30 percentage points). For reference, we also experiment with the random
splitting scenario, where we use the same model ( GIN+ VIRTUAL NODE ) on the same split ratio.
On the random split, the test accuracy is 92.91 0.27%, which is more than 20 percentage points
22

## Page 23

Table 17: Results for ogbg-ppa .
MethodVirtual Accuracy (%)
Node Training Validation Test
GCN% 97.680.22 64.970.34 68.390.84
" 97.001.00 65.110.48 68.570.61
GIN% 97.550.52 65.621.07 68.921.00
" 98.280.46 66.781.05 70.371.07
QH[WWRNHQHGJH)XQFWLRQ'HIUXQBPRGHO1DPHPRGHODUJXPHQWV&DOO$WWULEXWHUXQ1DPHPRGHO$EVWUDFW6\QWD[7UHH$676RXUFHFRGH/HJHQG
$67HGJH0RGXOH$671RGH7\SHSUHRUGHU')6LQGH[QRGHDWWULEXWH_mask_
Figure 4: Example input graph in ogbg-code2 , obtained by augmenting the original Python AST.
In our AST, node “#1” is always the main function deﬁnition (FunctionDef or AsyncFunctionDef),
and our goal is predict its tokenized attribute, e.g., {run,model } in the above example. To avoid
data leakage, we replace the attribute of node “#1” with a special “ _mask_ ” token. We also mask
out attributes of recursive function deﬁnitions if there are any.
higher than the species split. This again encourages future research to improve the out-of-distribution
generalization with more challenging and meaningful split procedure.
6.3 ogbg-code2 : Abstract Syntax Tree of Source Code
Theogbg-code2 dataset25is a collection of Abstract Syntax Trees (ASTs) obtained from approx-
imately 450 thousands Python method deﬁnitions. Methods are extracted from a total of 13,587
different repositories across the most popular projects on GITHUB(where “popularity” is deﬁned as
number of stars and forks). Our collection of Python methods originates from GITHUBCodeSearch-
Net (Husain et al., 2019)26, a collection of datasets and benchmarks for machine-learning-based
code retrieval. The authors paid particular attention to avoid common shortcomings of previous
source code datasets (Allamanis, 2019), such as duplication of code and labels, low number of
projects, random splitting, etc. In ogbg-code2 , we contribute an additional feature extraction step,
which includes: AST edges, AST nodes (associated with features such as their types and attributes),
tokenized method name (see Figure 4). Altogether, ogbg-code2 allows us to capture source code
with its underlying graph structure, beyond its token sequence representation.
Prediction task . The task is to predict the sub-tokens forming the method name, given the Python
method body represented by AST and its node features— i.e., node type (from a pool of 97 types),
node attributes (such as variable names, with a vocabulary size of 10030, depth in the AST, pre-order
traversal index (as illustrated in Figure 4). This task is often referred to in the literature as “code
summarization” (Allamanis et al., 2016; Alon et al., 2019, 2018), because the model is trained to
ﬁnd succinct and precise description ( i.e., the method name chosen by the developer) for a complete
logical unit ( i.e., the method body). Code summarization is a representative task in the ﬁeld of
machine learning for code not only for its straightforward adoption in developer tools, but also
because it is a proxy measure for assessing how well a model captures the code semantic (Allamanis
et al., 2018). Following Alon et al. (2019, 2018), we use an F1 score to evaluate predicted sub-tokens
25The older version ogbg-code has been deprecated due to the prediction target leakage in input AST.
26https://github.com/github/CodeSearchNet
23

## Page 24

Table 18: Results for ogbg-code2 .
MethodVirtual F1 score (%)
Node Training Validation Test
GCN% 30.062.95 13.990.17 15.070.18
" 30.942.34 14.610.13 15.950.18
GIN% 26.491.60 13.760.16 14.950.23
" 30.401.98 14.390.20 15.810.26
against ground-truth sub-tokens.27The average length of a method name in the ground-truth is 2:6
sub-tokens, following a power-law distribution.
Dataset splitting . We adopt a project split (Allamanis, 2019), where the ASTs for the train set are
obtained from GITHUBprojects that do not appear in the validation and test sets. This split respects
the practical scenario of training a model on a large collection of source code (obtained, for instance,
from the popular GITHUBprojects), and then using it to predict method names on a separate code
base. The project split stress-tests the model’s ability to capture code’s semantics, and avoids a model
that trivially memorizes the idiosyncrasies of training projects (such as the naming conventions and
the coding style of a speciﬁc developer) to achieve a high test score.
Discussion . Benchmarking results are given in Table 18, where we add “next-token edges” on top
of the AST (as illustrated in Figure 4) to better capture the semantics of code graphs (Dinella et al.,
2020).28For the decoder, we use independent linear classiﬁers to predict sub-tokens at each position
of the sub-token sequence.29The evaluation is performed against the ground-truth sub-tokens. We
see from Table 18 that GCN with VIRTUAL NODES provides the best performance. Nevertheless, we
observe a huge generalization gap (around 15 percentage points). For reference, we also experiment
with the random splitting scenario, where we apply the same model ( GCN+ VIRTUAL NODE ) on the
same split ratio. On the random split, the test F1 score is 21.64 0.26%, which is approximately 6
percentage points higher than that of the project split in Table 18, indicating that the project split is
indeed harder than the random split. Overall, this dataset presents an interesting research opportunity
to improve out-of-distribution generalization under the meaningful project split, with a number of
fruitful future directions: how to leverage the fact that the original graphs are actually trees with
well-deﬁned root nodes, how to pre-train GNNs to improve generalization (Hu et al., 2020a), and
how to design a better encoder-decoder architecture with the graph data. To facilitate these directions,
we provide enough meta-information, such as the original code snippet as well as an easy-to-use
script to transform raw Python code snippets into the ASTs.
7 OGB Package
The OGB package is designed to make the pipeline of Figure 2 easily accessible to researchers,
by automating the data loading and the evaluation parts. OGB is fully compatible with PYTORCH
and its associated graph libraries: PYTORCH GEOMETRIC andDEEP GRAPH LIBRARY . OGB
additionally provides library-agnostic dataset objects that can be used by any other Python deep
learning frameworks such as TENSOR FLOW (Abadi et al., 2016) and MXN ET(Chen et al., 2015).
Below, we explain the data loading ( cf.Section 7.1) and evaluation ( cf.Section 7.2). For simplicity,
27The previous works ﬁnd that the F1 score over sub-tokens is suitable to assess the quality of a method name
prediction, as the semantic of a method name depends solely on its sub-tokens. Note that the F1 score does
not take the sub-token ordering into account; thus, “ run_model ” and “ model_run ” are considered as exact
match.
28The inverse edges are also added to allow bidirectional message passing. The edge direction is recorded in
the edge features.
29Although the F1 score is order-insensitive, in our preliminary experiments, we ﬁnd that our order-sensitive
decoder performs slightly better than order-insensitive decoder (predicting whether each vocabulary is included
in the target sequence or not). During training, all the target sequences are truncated to the length of 5 (covering
99% of the target sequences), and vocabulary size of 5,000 is used for prediction (covering 90% of the sub-tokens
in the target sequences). We additionally added one vocabulary “UNK” to handle any rare/unknown sub-tokens.
Predicting “UNK” sub-token is counted as false positive when the F1 score is calculated.
24

## Page 25

we focus on the task of the graph property prediction ( cf.Section 6) using PYTORCH GEOMETRIC .
For the other tasks, libraries, and more details, refer to https://ogb.stanford.edu .
7.1 OGB Data Loaders
The OGB package makes it easy to obtain a dataset object that is fully compatible with PYTORCH
GEOMETRIC . As shown in Code Snippet 1, it can be done with only a single line of code, with the
end-users only needing to specify the name of the dataset. The OGB package will then download,
process, store, and return the requested dataset object. Furthermore, the standardized dataset splitting
can be readily obtained from the dataset object.
>>> from ogb.graphproppred import PygGraphPropPredDataset
>>> dataset = PygGraphPropPredDataset(name="ogbg-molpcba")
# Pytorch Geometric dataset object
>>> split_idx = dataset.get_idx_split()
# Dictionary containing train/valid/test indices.
>>> train_idx = split_idx["train"]
# torch.tensor storing a list of training indices.
Code Snippet 1: OGB Data Loader
7.2 OGB Evaluators
OGB also enables standardized and reliable evaluation with the ogb. *.Evaluator class. As
shown in Code Snippet 2, the end-users ﬁrst specify the dataset they want to evaluate their models
on, after which the users can learn the format of the input they need to pass to the Evaluator
object. The input format is dataset-dependent. For example, for the ogbg-molpcba dataset, the
Evaluator object requires as input a dictionary with y_true (a matrix storing the ground-truth
binary labels30), and y_pred (a matrix storing the scores output by the model). Once the end-users
pass the speciﬁed dictionary as input, the Evaluator object returns the model performance that is
appropriate for the dataset at hand, e.g., the Average Precision for ogbg-molpcba .
>>> from ogb.graphproppred import Evaluator
# Get Evaluator for ogbg-molpcba
>>> evaluator = Evaluator(name = "ogbg-molpcba")
# Learn about the specification of input to the Evaluator.
>>> print(evaluator.expected_input_format)
# Prepare input that follows input spec.
>>> input_dict = {"y_true": y_true, "y_pred": y_pred}
# Get the model performance.
result_dict = evaluator.eval(input_dict)
Code Snippet 2: OGB Evaluator
8 Conclusions
To enable scalable, robust, and reproducible graph ML research, we introduce the Open Graph
Benchmark (OGB)—a diverse set of realistic graph datasets in terms of scales, domains, and task
categories. We employ realistic data splits for the given datasets, driven by application-speciﬁc use
cases. Through extensive benchmark experiments, we highlight that OGB datasets present signiﬁcant
challenges for ML models to handle large-scale graphs and make accurate prediction under the
realistic data splitting scenarios. Altogether, OGB presents fruitful opportunities for future research
to push the frontier of graph ML.
30The shape of the matrix is the number of data points times the number of tasks. The matrix can be either a
PYTORCH tensor or N UMPYarray.
25

## Page 26

OGB is an open-source initiative that provides ready-to-use datasets as well as their data loaders,
evaluation scripts, and public leaderboards. We hereby invite the community to develop and contribute
state-of-the-art graph ML models at https://ogb.stanford.edu .
Acknowledgements
We thank Adrijan Bradaschia and Rok Sosic for their help in setting up the server and website.
We also thank Emma Pierson and Shigeru Maya for their suggestions on the paper writing, and
Charles Sutton for pointing out the data leakage in one of our datasets. Finally, we thank the entire
community of graph ML for providing valuable feedback to improve OGB. Weihua Hu is supported
by Funai Overseas Scholarship and Masason Foundation Fellowship. Matthias Fey is supported
by the German Research Association (DFG) within the Collaborative Research Center SFB 876
“Providing Information by Resource-Constrained Analysis”, project A6. Marinka Zitnik is in part
supported by NSF IIS-2030459. We gratefully acknowledge the support of DARPA under Nos.
FA865018C7880 (ASED), N660011924033 (MCS); ARO under Nos. W911NF-16-1-0342 (MURI),
W911NF-16-1-0171 (DURIP); NSF under Nos. OAC-1835598 (CINES), OAC-1934578 (HDR),
CCF-1918940 (Expeditions), IIS-2030477 (RAPID); Stanford Data Science Initiative, Wu Tsai
Neurosciences Institute, Chan Zuckerberg Biohub, Amazon, Boeing, JPMoran Chase, Docomo,
Hitachi, JD.com, KDDI, NVIDIA, Dell. Jure Leskovec is a Chan Zuckerberg Biohub investigator.
References
Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin,
Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. Tensorﬂow: A system for large-scale
machine learning. In Symposium on Operating Systems Design and Implementation OSDI) , pp.
265–283, 2016.
Miltiadis Allamanis. The adverse effects of code duplication in machine learning models of code. In
Proceedings of the 2019 ACM SIGPLAN International Symposium on New Ideas, New Paradigms,
and Reﬂections on Programming and Software , pp. 143–153, 2019.
Miltiadis Allamanis, Hao Peng, and Charles Sutton. A convolutional attention network for extreme
summarization of source code. In International conference on machine learning , pp. 2091–2100,
2016.
Miltiadis Allamanis, Marc Brockschmidt, and Mahmoud Khademi. Learning to represent programs
with graphs. arXiv preprint arXiv:1711.00740 , 2017.
Miltiadis Allamanis, Earl T Barr, Premkumar Devanbu, and Charles Sutton. A survey of machine
learning for big code and naturalness. ACM Computing Surveys (CSUR) , 51(4):1–37, 2018.
Uri Alon, Shaked Brody, Omer Levy, and Eran Yahav. code2seq: Generating sequences from
structured representations of code. arXiv preprint arXiv:1808.01400 , 2018.
Uri Alon, Meital Zilberstein, Omer Levy, and Eran Yahav. code2vec: Learning distributed rep-
resentations of code. Proceedings of the ACM on Programming Languages , 3(POPL):1–29,
2019.
Jürgen Bajorath. Integration of virtual and high-throughput screening. Nature Reviews Drug
Discovery , 1(11):882–894, 2002.
Albert-Laszlo Barabasi and Zoltan N Oltvai. Network biology: understanding the cell’s functional
organization. Nature reviews genetics , 5(2):101–113, 2004.
Jon Barker, Ricard Marxer, Emmanuel Vincent, and Shinji Watanabe. The third ‘chime’speech
separation and recognition challenge: Dataset, task and baselines. In 2015 IEEE Workshop on
Automatic Speech Recognition and Understanding (ASRU) , pp. 504–511. IEEE, 2015.
Rianne van den Berg, Thomas N. Kipf, and Max Welling. Graph convolutional matrix completion.
arXiv preprint arXiv:1706.02263 , 2017.
26

## Page 27

K. Bhatia, K. Dahiya, H. Jain, A. Mittal, Y . Prabhu, and M. Varma. The extreme classiﬁcation reposi-
tory: Multi-label datasets and code, 2016. URL http://manikvarma.org/downloads/
XC/XMLRepository.html .
Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. Freebase: a collabo-
ratively created graph database for structuring human knowledge. In Special Interest Group on
Management of Data (SIGMOD) , pp. 1247–1250. AcM, 2008.
Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko.
Translating embeddings for modeling multi-relational data. In Advances in Neural Information
Processing Systems (NeurIPS) , pp. 2787–2795, 2013.
Michael M Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, and Pierre Vandergheynst. Geometric
deep learning: going beyond euclidean data. IEEE Signal Processing Magazine , 34(4):18–42,
2017.
Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao, Bing Xu,
Chiyuan Zhang, and Zheng Zhang. Mxnet: A ﬂexible and efﬁcient machine learning library for
heterogeneous distributed systems. In NeurIPS workshop on Machine Learning Systems , 2015.
Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, and Cho-Jui Hsieh. Cluster-GCN: An
efﬁcient algorithm for training deep and large graph convolutional networks. In ACM SIGKDD
Conference on Knowledge Discovery and Data Mining (KDD) , pp. 257–266, 2019.
Gene Ontology Consortium. The gene ontology resource: 20 years and still going strong. Nucleic
acids research , 47(D1):D330–D338, 2018.
Lenore Cowen, Trey Ideker, Benjamin J Raphael, and Roded Sharan. Network propagation: a
universal ampliﬁer of genetic associations. Nature Reviews Genetics , 18(9):551, 2017.
Allan Peter Davis, Cynthia J Grondin, Robin J Johnson, Daniela Sciaky, Roy McMorran, Jolene
Wiegers, Thomas C Wiegers, and Carolyn J Mattingly. The comparative toxicogenomics database:
update 2019. Nucleic Acids Research , 47(D1):D948–D954, 2019.
Jesse Davis and Mark Goadrich. The relationship between precision-recall and roc curves. In
International Conference on Machine Learning (ICML) , pp. 233–240, 2006.
David De Juan, Florencio Pazos, and Alfonso Valencia. Emerging methods in protein co-evolution.
Nature Reviews Genetics , 14(4):249–261, 2013.
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale
hierarchical image database. In cvpr, pp. 248–255. Ieee, 2009.
Elizabeth Dinella, Hanjun Dai, Ziyang Li, Mayur Naik, Le Song, and Ke Wang. Hoppity: Learning
graph transformations to detect and ﬁx bugs in programs. In International Conference on Learning
Representations (ICLR) , 2020.
Yuxiao Dong, Nitesh V Chawla, and Ananthram Swami. metapath2vec: Scalable representation
learning for heterogeneous networks. In ACM SIGKDD Conference on Knowledge Discovery and
Data Mining (KDD) , pp. 135–144, 2017a.
Yuxiao Dong, Hao Ma, Zhihong Shen, and Kuansan Wang. A century of science: Globalization of
scientiﬁc collaborations, citations, and innovations. In ACM SIGKDD Conference on Knowledge
Discovery and Data Mining (KDD) , pp. 1437–1446. ACM, 2017b.
David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell, Timothy Hirzel, Alán
Aspuru-Guzik, and Ryan P Adams. Convolutional networks on graphs for learning molecular
ﬁngerprints. In Advances in Neural Information Processing Systems (NeurIPS) , pp. 2224–2232,
2015.
Vijay Prakash Dwivedi, Chaitanya K Joshi, Thomas Laurent, Yoshua Bengio, and Xavier Bresson.
Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982 , 2020.
David Easley, Jon Kleinberg, et al. Networks, crowds, and markets , volume 8. Cambridge university
press Cambridge, 2010.
27

## Page 28

Federico Errica, Marco Podda, Davide Bacciu, and Alessio Micheli. A fair comparison of graph
neural networks for graph classiﬁcation. arXiv preprint arXiv:1912.09893 , 2019.
Matthias Feurer, Jan N van Rijn, Arlind Kadra, Pieter Gijsbers, Neeratyoy Mallik, Sahithya Ravi,
Andreas Müller, Joaquin Vanschoren, and Frank Hutter. Openml-python: an extensible python api
for openml. arXiv preprint arXiv:1911.02490 , 2019.
M. Fey and J. E. Lenssen. Fast graph representation learning with PyTorch Geometric. In ICLR
Workshop on Representation Learning on Graphs and Manifolds , 2019.
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural
message passing for quantum chemistry. In International Conference on Machine Learning (ICML) ,
pp. 1273–1272, 2017.
Aditya Grover and Jure Leskovec. node2vec: Scalable feature learning for networks. In ACM
SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) , pp. 855–864. ACM,
2016.
Emre Guney. Reproducible drug repurposing: When similarity does not sufﬁce. In Paciﬁc Symposium
on Biocomputing , pp. 132–143, 2017.
William L Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large graphs.
InAdvances in Neural Information Processing Systems (NeurIPS) , pp. 1025–1035, 2017a.
William L Hamilton, Rex Ying, and Jure Leskovec. Representation learning on graphs: Methods and
applications. IEEE Data Engineering Bulletin , 40(3):52–74, 2017b.
Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec.
Strategies for pre-training graph neural networks. In International Conference on Learning
Representations (ICLR) , 2020a.
Ziniu Hu, Yuxiao Dong, Kuansan Wang, and Yizhou Sun. Heterogeneous graph transformer. In
Proceedings of the International World Wide Web Conference (WWW) , pp. n/a, 2020b.
Laura A Hug, Brett J Baker, Karthik Anantharaman, Christopher T Brown, Alexander J Probst,
Cindy J Castelle, Cristina N Butterﬁeld, Alex W Hernsdorf, Yuki Amano, Kotaro Ise, et al. A new
view of the tree of life. Nature Microbiology , 1(5):16048, 2016.
Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. Code-
searchnet challenge: Evaluating the state of semantic code search. arXiv preprint arXiv:1909.09436 ,
2019.
Katsuhiko Ishiguro, Shin-ichi Maeda, and Masanori Koyama. Graph warp module: An auxiliary
module for boosting the power of graph neural networks. arXiv preprint arXiv:1902.01020 , 2019.
S. Ivanov, S. Sviridov, and E. Burnaev. Understanding isomorphism bias in graph data sets. arXiv
preprint arXiv:1910.12091 , 2019.
Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Hierarchical generation of molecular graphs
using structural motifs. arXiv preprint arXiv:2002.03230 , 2020.
Kristian Kersting, Nils M Kriege, Christopher Morris, Petra Mutzel, and Marion Neumann. Bench-
mark data sets for graph kernels, 2020. URL http://www.graphlearning.io/ .
Thomas N. Kipf and Max Welling. Variational graph auto-encoders. arXiv preprint arXiv:1611.07308 ,
2016.
Thomas N. Kipf and Max Welling. Semi-supervised classiﬁcation with graph convolutional networks.
InInternational Conference on Learning Representations (ICLR) , 2017.
Johannes Klicpera, Aleksandar Bojchevski, and Stephan Günnemann. Predict then propagate:
Graph neural networks meet personalized pagerank. In International Conference on Learning
Representations (ICLR) , 2019.
Greg Landrum et al. Rdkit: Open-source cheminformatics, 2006.
28

## Page 29

Adam Lerer, Ledell Wu, Jiajun Shen, Timothee Lacroix, Luca Wehrstedt, Abhijit Bose, and
Alex Peysakhovich. Pytorch-biggraph: A large-scale graph embedding system. arXiv preprint
arXiv:1903.12287 , 2019.
Jure Leskovec and Rok Sosi ˇc. Snap: A general-purpose network analysis and graph-mining library.
ACM Transactions on Intelligent Systems and Technology (TIST) , 8(1):1–20, 2016.
Guohao Li, Matthias Muller, Ali Thabet, and Bernard Ghanem. Deepgcns: Can gcns go as deep as
cnns? In IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , pp. 9267–9276,
2019.
Junying Li, Deng Cai, and Xiaofei He. Learning graph-level representation for drug discovery. arXiv
preprint arXiv:1709.03741 , 2017.
Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard Zemel. Gated graph sequence neural
networks. In International Conference on Learning Representations (ICLR) , 2016.
David Liben-Nowell and Jon M. Kleinberg. The link-prediction problem for social networks. Journal
of the Association for Information Science and Technology , 58(7):1019–1031, 2007.
Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr
Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In eccv, pp. 740–755.
Springer, 2014.
Sharon L Lohr. Sampling: design and analysis . Nelson Education, 2009.
Ricardo Macarron, Martyn N Banks, Dejan Bojanic, David J Burns, Dragan A Cirovic, Tina
Garyantes, Darren VS Green, Robert P Hertzberg, William P Janzen, Jeff W Paslay, et al. Impact of
high-throughput screening in biomedical research. Nature Reviews Drug discovery , 10(3):188–195,
2011.
Noël Malod-Dognin, Kristina Ban, and Nataša Pržulj. Uniﬁed alignment of protein-protein interaction
networks. Scientiﬁc Reports , 7(1):1–11, 2017.
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations
of words and phrases and their compositionality. In Advances in Neural Information Processing
Systems (NeurIPS) , pp. 3111–3119, 2013.
Maximilian Nickel, Kevin Murphy, V olker Tresp, and Evgeniy Gabrilovich. A review of relational
machine learning for knowledge graphs. Proceedings of the IEEE , 104(1):11–33, 2015.
Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: an asr corpus
based on public domain audio books. In 2015 IEEE International Conference on Acoustics, Speech
and Signal Processing (ICASSP) , pp. 5206–5210. IEEE, 2015.
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. PyTorch: An imperative style,
high-performance deep learning library. In Advances in Neural Information Processing Systems
(NeurIPS) , pp. 8024–8035, 2019.
Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. Deepwalk: Online learning of social represen-
tations. In ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) , pp.
701–710. ACM, 2014.
Janet Piñero, Juan Manuel Ramírez-Anguita, Josep Saüch-Pitarch, Francesco Ronzano, Emilio
Centeno, Ferran Sanz, and Laura I Furlong. The DisGeNET knowledge platform for disease
genomics: 2019 update. Nucleic Acids Research , 48(D1):D845–D855, 2020.
Jiezhong Qiu, Yuxiao Dong, Hao Ma, Jian Li, Chi Wang, Kuansan Wang, and Jie Tang. Netsmf:
Large-scale network embedding as sparse matrix factorization. In Proceedings of the International
World Wide Web Conference (WWW) , pp. 1509–1520, 2019.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for
machine comprehension of text. arXiv preprint arXiv:1606.05250 , 2016.
29

## Page 30

Yu Rong, Yatao Bian, Tingyang Xu, Weiyang Xie, Ying Wei, Wenbing Huang, and Junzhou Huang.
Grover: Self-supervised message passing transformer on large-scale molecular data. arXiv preprint
arXiv:2007.02835 , 2020a.
Yu Rong, Wenbing Huang, Tingyang Xu, and Junzhou Huang. Dropedge: Towards deep graph convo-
lutional networks on node classiﬁcation. In International Conference on Learning Representations
(ICLR) , 2020b.
Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne Van Den Berg, Ivan Titov, and Max
Welling. Modeling relational data with graph convolutional networks. In European Semantic Web
Conference , pp. 593–607. Springer, 2018.
Roded Sharan, Silpa Suthram, Ryan M Kelley, Tanja Kuhn, Scott McCuine, Peter Uetz, Taylor Sittler,
Richard M Karp, and Trey Ideker. Conserved patterns of protein interaction in multiple species.
Proceedings of the National Academy of Sciences , 102(6):1974–1979, 2005.
Oleksandr Shchur, Maximilian Mumme, Aleksandar Bojchevski, and Stephan Günnemann. Pitfalls
of graph neural network evaluation. arXiv preprint arXiv:1811.05868 , 2018.
Martin Simonovsky and Nikos Komodakis. Dynamic edge-conditioned ﬁlters in convolutional neural
networks on graphs. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR) ,
pp. 3693–3702, 2017.
Koustuv Sinha, Shagun Sodhani, Joelle Pineau, and William L Hamilton. Evaluating logical general-
ization in graph neural networks. arXiv preprint arXiv:2003.06560 , 2020.
Jonathan M Stokes, Kevin Yang, Kyle Swanson, Wengong Jin, Andres Cubillos-Ruiz, Nina M
Donghia, Craig R MacNair, Shawn French, Lindsey A Carfrae, Zohar Bloom-Ackerman, et al. A
deep learning approach to antibiotic discovery. Cell, 180(4):688–702, 2020.
Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian Tang. Rotate: Knowledge graph embedding by
relational rotation in complex space. In International Conference on Learning Representations
(ICLR) , 2019.
Damian Szklarczyk, Alberto Santos, Christian von Mering, Lars Juhl Jensen, Peer Bork, and Michael
Kuhn. STITCH 5: augmenting protein–chemical interaction networks with tissue and afﬁnity data.
Nucleic Acids Research , 44(D1):D380–D384, 2016.
Damian Szklarczyk, Annika L Gable, David Lyon, Alexander Junge, Stefan Wyder, Jaime Huerta-
Cepas, Milan Simonovic, Nadezhda T Doncheva, John H Morris, Peer Bork, et al. STRING v11:
protein–protein association networks with increased coverage, supporting functional discovery in
genome-wide experimental datasets. Nucleic Acids Research , 47(D1):D607–D613, 2019.
Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard. Complex
embeddings for simple link prediction. In International Conference on Machine Learning (ICML) ,
pp. 2071–2080, 2016.
Joaquin Vanschoren, Jan N. van Rijn, Bernd Bischl, and Luis Torgo. Openml: Networked science in
machine learning. SIGKDD Explorations , 15(2):49–60, 2013. doi: 10.1145/2641190.2641198.
URLhttp://doi.acm.org/10.1145/2641190.2641198 .
Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua
Bengio. Graph attention networks. In International Conference on Learning Representations
(ICLR) , 2018.
Petar Veli ˇckovi ´c, William Fedus, William L Hamilton, Pietro Liò, Yoshua Bengio, and R Devon
Hjelm. Deep graph infomax. In International Conference on Learning Representations (ICLR) ,
2019.
Denny Vrande ˇci´c and Markus Krötzsch. Wikidata: a free collaborative knowledgebase. Communica-
tions of the ACM , 57(10):78–85, 2014.
Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue:
A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint
arXiv:1804.07461 , 2018.
30

## Page 31

Kuansan Wang, Zhihong Shen, Chiyuan Huang, Chieh-Han Wu, Yuxiao Dong, and Anshul Kanakia.
Microsoft academic graph: When experts are not enough. Quantitative Science Studies , 1(1):
396–413, 2020.
Minjie Wang, Lingfan Yu, Da Zheng, Quan Gan, Yu Gai, Zihao Ye, Mufei Li, Jinjing Zhou, Qi Huang,
Chao Ma, Ziyue Huang, Qipeng Guo, Hao Zhang, Haibin Lin, Junbo Zhao, Jinyang Li, Alexander J
Smola, and Zheng Zhang. Deep graph library: Towards efﬁcient and scalable deep learning on
graphs. ICLR Workshop on Representation Learning on Graphs and Manifolds , 2019a. URL
https://arxiv.org/abs/1909.01315 .
Xiaozhi Wang, Tianyu Gao, Zhaocheng Zhu, Zhiyuan Liu, Juanzi Li, and Jian Tang. Kepler: A
uniﬁed model for knowledge embedding and pre-trained language representation. arXiv preprint
arXiv:1911.06136 , 2019b.
David S Wishart, Yannick D Feunang, An C Guo, Elvis J Lo, Ana Marcu, Jason R Grant, Tanvir
Sajed, Daniel Johnson, Carin Li, Zinat Sayeeda, et al. DrugBank 5.0: a major update to the
DrugBank database for 2018. Nucleic Acids Research , 46(D1):D1074–D1082, 2018.
Felix Wu, Tianyi Zhang, Amauri Holanda de Souza Jr, Christopher Fifty, Tao Yu, and Kilian Q
Weinberger. Simplifying graph convolutional networks. In International Conference on Machine
Learning (ICML) , 2019.
Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S
Pappu, Karl Leswing, and Vijay Pande. Moleculenet: a benchmark for molecular machine learning.
Chemical science , 9(2):513–530, 2018.
Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, and Stefanie
Jegelka. Representation learning on graphs with jumping knowledge networks. In International
Conference on Machine Learning (ICML) , pp. 5453–5462, 2018.
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural
networks? In International Conference on Learning Representations (ICLR) , 2019.
Pinar Yanardag and SVN Vishwanathan. Deep graph kernels. In ACM SIGKDD Conference on
Knowledge Discovery and Data Mining (KDD) , pp. 1365–1374. ACM, 2015.
Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. Embedding entities and
relations for learning and inference in knowledge bases. In International Conference on Learning
Representations (ICLR) , 2015.
Kevin Yang, Kyle Swanson, Wengong Jin, Connor Coley, Philipp Eiden, Hua Gao, Angel Guzman-
Perez, Timothy Hopper, Brian Kelley, Miriam Mathea, et al. Analyzing learned molecular
representations for property prediction. Journal of chemical information and modeling , 59(8):
3370–3388, 2019.
Zhilin Yang, William W Cohen, and Ruslan Salakhutdinov. Revisiting semi-supervised learning with
graph embeddings. In International Conference on Machine Learning (ICML) , pp. 40–48, 2016.
Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L Hamilton, and Jure Leskovec.
Graph convolutional neural networks for web-scale recommender systems. In ACM SIGKDD
Conference on Knowledge Discovery and Data Mining (KDD) , pp. 974–983, 2018a.
Rex Ying, Jiaxuan You, Christopher Morris, Xiang Ren, William L Hamilton, and Jure Leskovec.
Hierarchical graph representation learning with differentiable pooling. In Advances in Neural
Information Processing Systems (NeurIPS) , 2018b.
Jiaxuan You, Rex Ying, and Jure Leskovec. Position-aware graph neural networks. In International
Conference on Machine Learning (ICML) , 2019.
David Younger, Stephanie Berger, David Baker, and Eric Klavins. High-throughput characterization
of protein–protein interactions by reprogramming yeast mating. Proceedings of the National
Academy of Sciences , 114(46):12166–12171, 2017.
Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, Rajgopal Kannan, and Viktor Prasanna. Graph-
Saint: Graph sampling based inductive learning method. In International Conference on Learning
Representations (ICLR) , 2020.
31

## Page 32

Muhan Zhang and Yixin Chen. Link prediction based on graph neural networks. In Advances in
Neural Information Processing Systems (NeurIPS) , pp. 5165–5175, 2018.
Da Zheng, Xiang Song, Chao Ma, Zeyuan Tan, Zihao Ye, Jin Dong, Hao Xiong, Zheng Zhang,
and George Karypis. Dgl-ke: Training knowledge graph embeddings at scale. arXiv preprint
arXiv:2004.08532 , 2020.
Zhaocheng Zhu, Shizhen Xu, Jian Tang, and Meng Qu. Graphvite: A high-performance cpu-
gpu hybrid system for node embedding. In Proceedings of the International World Wide Web
Conference (WWW) , pp. 2494–2504, 2019.
Marinka Zitnik, Monica Agrawal, and Jure Leskovec. Modeling polypharmacy side effects with
graph convolutional networks. Bioinformatics , 34(13):i457–i466, 2018.
Marinka Zitnik, Marcus W Feldman, Jure Leskovec, et al. Evolution of resilience in protein
interactomes across the tree of life. Proceedings of the National Academy of Sciences , 116(10):
4426–4433, 2019.
Xu Zou, Qiuye Jia, Jianwei Zhang, Chang Zhou, Zijun Yao, Hongxia Yang, and Jie Tang. Dimensional
reweighting graph convolution networks, 2020. URL https://openreview.net/forum?
id=SJeLO34KwS .
32

## Page 33

Table 19: Summary of ogbg-mol *datasets. For all the datasets, we use the scaffold split with
the split ratio of 80/10/10.
Category Name #GraphsAverage Average#TasksTaskMetric#Nodes #Edges Type
Molecular
Graph
ogbg-moltox21 7,831 18.6 19.3 12 Binary class. ROC-AUC
toxcast 8,576 18.8 19.3 617 Binary class. ROC-AUC
muv 93,087 24.2 26.3 17 Binary class. AP
bace 1,513 34.1 36.9 1 Binary class. ROC-AUC
bbbp 2,039 24.1 26.0 1 Binary class. ROC-AUC
clintox 1,477 26.2 27.9 2 Binary class. ROC-AUC
sider 1,427 33.6 35.4 27 Binary class. ROC-AUC
esol 1,128 13.3 13.7 1 Regression RMSE
freesolv 642 8.7 8.4 1 Regression RMSE
lipo 4,200 27.0 29.5 1 Regression RMSE
A More Benchmark Results on ogbg-mol *Datasets
Here we perform benchmark experiments on the other 10 datasets from MOLECULE NET(Wu et al.,
2018). The datasets are summarized in Table 19. The detailed description of each dataset is provided
in Wu et al. (2018). We use the same experimental protocol and hyper-parameters as in Section
6.1. The dropout rate is ﬁxed to 0.5. As evaluation metrics, we adopt ROC-AUC for all the binary
classiﬁcation datasets except for ogbg-molmuv that exhibits signiﬁcant class imbalance (only 0.2%
of labels are positive). For the ogbg-molmuv dataset, we use Average Precision (AP), which is a
more appropriate metric for heavily-imbalanced data (Wu et al., 2018; Davis & Goadrich, 2006). For
the regression datasets, we adopt Root Mean Squared Error (RMSE); the lower, the better.
The benchmark results for each dataset are provided in Tables 20–29. We observe the followings.
•The additional features almost always help improve generalization performance. In fact, on
top of GIN+V IRTUAL NODE , including the additional features gives either comparable or
improved performance on 9 out of the 10 datasets (except for ogbg-molbace in Table
23). This motivates us to include these additional features in our OGB molecular graphs.
•Adding VIRTUAL NODES often improves generalization performance; for example, on top
ofGIN , adding VIRTUAL NODES gives either comparable or improved performance on 9
out of the 10 datasets (except for ogbg-clintox in Table 25).
•The optimal GNN architectures ( GCN orGIN ) vary across the datasets. This raises a
natural question: can we design a GNN architecture that performs well across the molecule
datasets?
Altogether, we hope our extensive benchmark results on a variety of molecule datasets provide useful
baselines for further research on molecule-speciﬁc graph ML models.
33

## Page 34

Table 20: Results for ogbg-moltox21 .
MethodAdd. Virt. ROC-AUC (%)
Feat. Node Training Validation Test
GCN% " 90.011.8181.120.3775.511.00
" % 92.060.9379.040.1975.290.69
" " 93.282.1882.050.4377.460.86
GIN% " 93.130.94 81.470.376.210.82
" % 93.060.8878.320.4874.910.51
" " 93.671.0382.170.3577.570.62Table 21: Results for ogbg-moltoxcast .
MethodAdd. Virt. ROC-AUC (%)
Feat. Node Training Validation Test
GCN% " 88.890.8870.520.3466.330.35
" % 85.211.6967.480.3363.540.42
" " 89.890.871.650.3866.710.45
GIN% " 85.510.5969.620.6666.180.68
" % 84.651.5668.620.6363.410.74
" " 86.420.4972.320.3566.130.50
Table 22: Results for ogbg-molmuv .
MethodAdd. Virt. AP (%)
Feat. Node Training Validation Test
GCN% " 6.673.87 8.481.58 2.482.83
" % 22.726.9 21.41.46 11.392.87
" " 23.647.12 22.11.98 10.982.91
GIN% " 26.497.2415.742.19 7.912.13
" % 17.944.0619.002.15 8.782.07
" " 25.957.8517.421.32 9.842.71Table 23: Results for ogbg-molbace .
MethodAdd. Virt. ROC-AUC (%)
Feat. Node Training Validation Test
GCN% " 87.855.0878.992.0371.444.01
" % 91.741.9073.741.4979.151.44
" " 91.162.8680.251.4368.936.95
GIN% " 87.841.7577.211.0176.412.68
" % 92.072.6273.301.9572.974.00
" " 92.045.7780.811.7173.465.24
Table 24: Results for ogbg-molbbbp .
MethodAdd. Virt. ROC-AUC (%)
Feat. Node Training Validation Test
GCN% " 90.423.8293.460.2768.622.19
" % 96.971.3194.740.3168.871.51
" " 98.291.7995.950.4067.802.35
GIN% " 94.061.8594.660.3569.881.70
" % 95.992.4494.830.5268.171.48
" " 97.701.7195.680.4069.711.92Table 25: Results for ogbg-molclintox .
MethodAdd. Virt. ROC-AUC (%)
Feat. Node Training Validation Test
GCN% " 83.115.7288.781.4868.664.95
" % 98.140.9199.240.4791.301.73
" " 97.351.3099.570.1588.552.09
GIN% " 86.125.5090.791.1061.794.77
" % 96.311.7798.540.4888.142.51
" " 93.511.7899.180.5384.063.84
Table 26: Results for ogbg-molsider .
MethodAdd. Virt. ROC-AUC (%)
Feat. Node Training Validation Test
GCN% " 73.821.0259.860.8161.651.06
" % 82.742.9964.640.8259.601.77
" " 77.502.5861.880.8959.841.54
GIN% " 72.370.7859.840.8657.751.14
" % 80.132.9164.141.2457.601.40
" " 76.601.3862.410.9957.571.56Table 27: Results for ogbg-molesol .
MethodAdd. Virt. RMSE
Feat. Node Training Validation Test
GCN% " 0.8830.096 1.1280.032 1.1430.075
" % 0.6290.041 1.0220.034 1.1140.036
" " 0.7580.147 0.9910.04 1.0150.096
GIN% " 0.7460.158 0.9210.045 1.0260.063
" % 0.6280.041 1.0070.028 1.1730.057
" " 0.6750.131 0.8780.036 0.9980.066
Table 28: Results for ogbg-molfreesolv .
MethodAdd. Virt. RMSE
Feat. Node Training Validation Test
GCN% " 1.1630.157 2.7440.201 2.4130.195
" % 0.9820.109 2.5820.297 2.6400.239
" " 1.2190.153 2.9220.185 2.1860.120
GIN% " 1.0060.225 2.5670.19 2.3070.340
" % 1.2050.360 2.3420.378 2.7550.349
" " 0.9340.138 2.1810.205 2.1510.295Table 29: Results for ogbg-mollipo .
MethodAdd. Virt. RMSE
Feat. Node Training Validation Test
GCN% " 0.6690.058 0.8550.032 0.8230.029
" % 0.6620.046 0.8160.024 0.7970.023
" " 0.5450.041 0.7660.011 0.7710.016
GIN% " 0.4880.029 0.7490.018 0.7410.024
" % 0.4790.027 0.7420.011 0.7570.018
" " 0.3990.023 0.6790.014 0.7040.015
34