# MobileNetV2 Inverted Residuals and Linear Bottlenecks

## Page 1

MobileNetV2: Inverted Residuals and Linear Bottlenecks
Mark Sandler Andrew Howard Menglong Zhu Andrey Zhmoginov Liang-Chieh Chen
Google Inc.
fsandler, howarda, menglong, azhmogin, lcchen g@google.com
Abstract
In this paper we describe a new mobile architecture,
MobileNetV2, that improves the state of the art perfor-
mance of mobile models on multiple tasks and bench-
marks as well as across a spectrum of different model
sizes. We also describe efﬁcient ways of applying these
mobile models to object detection in a novel framework
we call SSDLite. Additionally, we demonstrate how
to build mobile semantic segmentation models through
a reduced form of DeepLabv3 which we call Mobile
DeepLabv3.
is based on an inverted residual structure where
the shortcut connections are between the thin bottle-
neck layers. The intermediate expansion layer uses
lightweight depthwise convolutions to ﬁlter features as
a source of non-linearity. Additionally, we ﬁnd that it is
important to remove non-linearities in the narrow layers
in order to maintain representational power. We demon-
strate that this improves performance and provide an in-
tuition that led to this design.
Finally, our approach allows decoupling of the in-
put/output domains from the expressiveness of the trans-
formation, which provides a convenient framework for
further analysis. We measure our performance on
ImageNet [1] classiﬁcation, COCO object detection [2],
VOC image segmentation [3]. We evaluate the trade-offs
between accuracy, and number of operations measured
by multiply-adds (MAdd), as well as actual latency, and
the number of parameters.
1. Introduction
Neural networks have revolutionized many areas of
machine intelligence, enabling superhuman accuracy for
challenging image recognition tasks. However, the drive
to improve accuracy often comes at a cost: modern state
of the art networks require high computational resources
beyond the capabilities of many mobile and embeddedapplications.
This paper introduces a new neural network architec-
ture that is speciﬁcally tailored for mobile and resource
constrained environments. Our network pushes the state
of the art for mobile tailored computer vision models,
by signiﬁcantly decreasing the number of operations and
memory needed while retaining the same accuracy.
Our main contribution is a novel layer module: the
inverted residual with linear bottleneck. This mod-
ule takes as an input a low-dimensional compressed
representation which is ﬁrst expanded to high dimen-
sion and ﬁltered with a lightweight depthwise convo-
lution. Features are subsequently projected back to a
low-dimensional representation with a linear convolu-
tion. The ofﬁcial implementation is available as part of
TensorFlow-Slim model library in [4].
This module can be efﬁciently implemented using
standard operations in any modern framework and al-
lows our models to beat state of the art along multiple
performance points using standard benchmarks. Fur-
thermore, this convolutional module is particularly suit-
able for mobile designs, because it allows to signiﬁ-
cantly reduce the memory footprint needed during in-
ference by never fully materializing large intermediate
tensors. This reduces the need for main memory access
in many embedded hardware designs, that provide small
amounts of very fast software controlled cache memory.
2. Related Work
Tuning deep neural architectures to strike an optimal
balance between accuracy and performance has been
an area of active research for the last several years.
Both manual architecture search and improvements in
training algorithms, carried out by numerous teams has
lead to dramatic improvements over early designs such
as AlexNet [5], VGGNet [6], GoogLeNet [7]. , and
ResNet [8]. Recently there has been lots of progress
in algorithmic architecture exploration included hyper-
parameter optimization [9, 10, 11] as well as variousarXiv:1801.04381v4  [cs.CV]  21 Mar 2019

## Page 2

methods of network pruning [12, 13, 14, 15, 16, 17] and
connectivity learning [18, 19]. A substantial amount of
work has also been dedicated to changing the connectiv-
ity structure of the internal convolutional blocks such as
in ShufﬂeNet [20] or introducing sparsity [21] and oth-
ers [22].
Recently, [23, 24, 25, 26], opened up a new direc-
tion of bringing optimization methods including genetic
algorithms and reinforcement learning to architectural
search. However one drawback is that the resulting net-
works end up very complex. In this paper, we pursue the
goal of developing better intuition about how neural net-
works operate and use that to guide the simplest possible
network design. Our approach should be seen as compli-
mentary to the one described in [23] and related work.
In this vein our approach is similar to those taken by
[20, 22] and allows to further improve the performance,
while providing a glimpse on its internal operation. Our
network design is based on MobileNetV1 [27]. It re-
tains its simplicity and does not require any special op-
erators while signiﬁcantly improves its accuracy, achiev-
ing state of the art on multiple image classiﬁcation and
detection tasks for mobile applications.
3. Preliminaries, discussion and intuition
3.1. Depthwise Separable Convolutions
Depthwise Separable Convolutions are a key build-
ing block for many efﬁcient neural network architectures
[27, 28, 20] and we use them in the present work as well.
The basic idea is to replace a full convolutional opera-
tor with a factorized version that splits convolution into
two separate layers. The ﬁrst layer is called a depthwise
convolution, it performs lightweight ﬁltering by apply-
ing a single convolutional ﬁlter per input channel. The
second layer is a 11convolution, called a pointwise
convolution, which is responsible for building new fea-
tures through computing linear combinations of the in-
put channels.
Standard convolution takes an hiwidiin-
put tensorLi, and applies convolutional kernel K2
Rkkdidjto produce an hiwidjoutput ten-
sorLj. Standard convolutional layers have the compu-
tational cost of hiwididjkk.
Depthwise separable convolutions are a drop-in re-
placement for standard convolutional layers. Empiri-
cally they work almost as well as regular convolutions
but only cost:
hiwidi(k2+dj) (1)
which is the sum of the depthwise and 11pointwiseconvolutions. Effectively depthwise separable convolu-
tion reduces computation compared to traditional layers
by almost a factor of k21. MobileNetV2 uses k= 3
(33depthwise separable convolutions) so the compu-
tational cost is 8to9times smaller than that of standard
convolutions at only a small reduction in accuracy [27].
3.2. Linear Bottlenecks
Consider a deep neural network consisting of nlayers
Lieach of which has an activation tensor of dimensions
hiwidi. Throughout this section we will be dis-
cussing the basic properties of these activation tensors,
which we will treat as containers of hiwi“pixels”
withdidimensions. Informally, for an input set of real
images, we say that the set of layer activations (for any
layerLi) forms a “manifold of interest”. It has been long
assumed that manifolds of interest in neural networks
could be embedded in low-dimensional subspaces. In
other words, when we look at all individual d-channel
pixels of a deep convolutional layer, the information
encoded in those values actually lie in some manifold,
which in turn is embeddable into a low-dimensional sub-
space2.
At a ﬁrst glance, such a fact could then be captured
and exploited by simply reducing the dimensionality of
a layer thus reducing the dimensionality of the oper-
ating space. This has been successfully exploited by
MobileNetV1 [27] to effectively trade off between com-
putation and accuracy via a width multiplier parameter,
and has been incorporated into efﬁcient model designs
of other networks as well [20]. Following that intuition,
the width multiplier approach allows one to reduce the
dimensionality of the activation space until the mani-
fold of interest spans this entire space. However, this
intuition breaks down when we recall that deep convo-
lutional neural networks actually have non-linear per co-
ordinate transformations, such as ReLU . For example,
ReLU applied to a line in 1D space produces a ’ray’,
where as inRnspace, it generally results in a piece-wise
linear curve with n-joints.
It is easy to see that in general if a result of a layer
transformation ReLU(Bx)has a non-zero volume S,
the points mapped to interiorSare obtained via a lin-
ear transformation Bof the input, thus indicating that
the part of the input space corresponding to the full di-
mensional output, is limited to a linear transformation.
In other words, deep networks only have the power of
a linear classiﬁer on the non-zero volume part of the
1more precisely, by a factor k2dj=(k2+dj)
2Note that dimensionality of the manifold differs from the dimen-
sionality of a subspace that could be embedded via a linear transfor-
mation.

## Page 3

Input Output/dim=2 Output/dim=3 Output/dim=5 Output/dim=15 Output/dim=30Figure 1: Examples of ReLU transformations of
low-dimensional manifolds embedded in higher-dimensional
spaces. In these examples the initial spiral is embedded into
ann-dimensional space using random matrix Tfollowed by
ReLU , and then projected back to the 2D space using T 1.
In examples above n= 2;3result in information loss where
certain points of the manifold collapse into each other, while
forn= 15 to30the transformation is highly non-convex.
(a) Regular
 (b) Separable
(c) Separable with linear
bottleneck
(d) Bottleneck with ex-
pansion layer
Figure 2: Evolution of separable convolution blocks. The
diagonally hatched texture indicates layers that do not contain
non-linearities. The last (lightly colored) layer indicates the
beginning of the next block. Note: 2d and 2c are equivalent
blocks when stacked. Best viewed in color.
output domain. We refer to supplemental material for
a more formal statement.
On the other hand, when ReLU collapses the chan-
nel, it inevitably loses information in that channel . How-
ever if we have lots of channels, and there is a a structure
in the activation manifold that information might still be
preserved in the other channels. In supplemental ma-
terials, we show that if the input manifold can be em-
bedded into a signiﬁcantly lower-dimensional subspace
of the activation space then the ReLU transformation
preserves the information while introducing the needed
complexity into the set of expressible functions.
To summarize, we have highlighted two properties
that are indicative of the requirement that the manifold
of interest should lie in a low-dimensional subspace of
the higher-dimensional activation space:
1. If the manifold of interest remains non-zero vol-
ume after ReLU transformation, it corresponds to
a linear transformation.(a) Residual block
 (b) Inverted residual block
Figure 3: The difference between residual block [8, 30]
and inverted residual. Diagonally hatched layers do not
use non-linearities. We use thickness of each block to
indicate its relative number of channels. Note how clas-
sical residuals connects the layers with high number of
channels, whereas the inverted residuals connect the bot-
tlenecks. Best viewed in color.
2.ReLU is capable of preserving complete informa-
tion about the input manifold, but only if the input
manifold lies in a low-dimensional subspace of the
input space.
These two insights provide us with an empirical hint
for optimizing existing neural architectures: assuming
the manifold of interest is low-dimensional we can cap-
ture this by inserting linear bottleneck layers into the
convolutional blocks. Experimental evidence suggests
that using linear layers is crucial as it prevents non-
linearities from destroying too much information. In
Section 6, we show empirically that using non-linear
layers in bottlenecks indeed hurts the performance by
several percent, further validating our hypothesis3. We
note that similar reports where non-linearity was helped
were reported in [29] where non-linearity was removed
from the input of the traditional residual block and that
lead to improved performance on CIFAR dataset.
For the remainder of this paper we will be utilizing
bottleneck convolutions. We will refer to the ratio be-
tween the size of the input bottleneck and the inner size
as the expansion ratio .
3.3. Inverted residuals
The bottleneck blocks appear similar to residual
block where each block contains an input followed
by several bottlenecks then followed by expansion [8].
However, inspired by the intuition that the bottlenecks
actually contain all the necessary information, while an
expansion layer acts merely as an implementation detail
that accompanies a non-linear transformation of the ten-
sor, we use shortcuts directly between the bottlenecks.
3We note that in the presence of shortcuts the information loss is
actually less strong.

## Page 4

Figure 3 provides a schematic visualization of the differ-
ence in the designs. The motivation for inserting short-
cuts is similar to that of classical residual connections:
we want to improve the ability of a gradient to propagate
across multiplier layers. However, the inverted design is
considerably more memory efﬁcient (see Section 5 for
details), as well as works slightly better in our experi-
ments.
Running time and parameter count for bottleneck
convolution The basic implementation structure is il-
lustrated in Table 1. For a block of size hw, ex-
pansion factor tand kernel size kwithd0input chan-
nels andd00output channels, the total number of multi-
ply add required is hwd0t(d0+k2+d00). Com-
pared with (1) this expression has an extra term, as in-
deed we have an extra 11convolution, however the
nature of our networks allows us to utilize much smaller
input and output dimensions. In Table 3 we compare the
needed sizes for each resolution between MobileNetV1,
MobileNetV2 and ShufﬂeNet.
3.4. Information ﬂow interpretation
One interesting property of our architecture is that it
provides a natural separation between the input/output
domains of the building blocks (bottleneck layers), and
thelayer transformation – that is a non-linear function
that converts input to the output. The former can be seen
as the capacity of the network at each layer, whereas the
latter as the expressiveness . This is in contrast with tra-
ditional convolutional blocks, both regular and separa-
ble, where both expressiveness and capacity are tangled
together and are functions of the output layer depth.
In particular, in our case, when inner layer depth
is0the underlying convolution is the identity function
thanks to the shortcut connection. When the expansion
ratio is smaller than 1, this is a classical residual con-
volutional block [8, 30]. However, for our purposes we
show that expansion ratio greater than 1is the most use-
ful.
This interpretation allows us to study the expressive-
ness of the network separately from its capacity and we
believe that further exploration of this separation is war-
ranted to provide a better understanding of the network
properties.
4. Model Architecture
Now we describe our architecture in detail. As dis-
cussed in the previous section the basic building block
is a bottleneck depth-separable convolution with resid-
uals. The detailed structure of this block is shown inInput Operator Output
hwk 1x1conv2d ,ReLU6 hw(tk)
hwtk 3x3dwise s=s,ReLU6h
sw
s(tk)
h
sw
stk linear 1x1 conv2dh
sw
sk0
Table 1: Bottleneck residual block transforming from k
tok0channels, with stride s, and expansion factor t.
Table 1. The architecture of MobileNetV2 contains the
initial fully convolution layer with 32ﬁlters, followed
by19residual bottleneck layers described in the Ta-
ble 2. We use ReLU6 as the non-linearity because of
its robustness when used with low-precision computa-
tion [27]. We always use kernel size 33as is standard
for modern networks, and utilize dropout and batch nor-
malization during training.
With the exception of the ﬁrst layer, we use constant
expansion rate throughout the network. In our experi-
ments we ﬁnd that expansion rates between 5and10re-
sult in nearly identical performance curves, with smaller
networks being better off with slightly smaller expan-
sion rates and larger networks having slightly better per-
formance with larger expansion rates.
For all our main experiments we use expansion factor
of6applied to the size of the input tensor. For example,
for a bottleneck layer that takes 64-channel input tensor
and produces a tensor with 128channels, the intermedi-
ate expansion layer is then 646 = 384 channels.
Trade-off hyper parameters As in [27] we tailor our
architecture to different performance points, by using
the input image resolution and width multiplier as tun-
able hyper parameters, that can be adjusted depending
on desired accuracy/performance trade-offs. Our pri-
mary network (width multiplier 1,224224), has a
computational cost of 300 million multiply-adds and
uses 3.4 million parameters. We explore the perfor-
mance trade offs, for input resolutions from 96to224,
and width multipliers of 0:35to1:4. The network com-
putational cost ranges from 7multiply adds to 585M
MAdds, while the model size vary between 1.7M and
6.9M parameters.
One minor implementation difference, with [27] is
that for multipliers less than one, we apply width multi-
plier to all layers except the very last convolutional layer.
This improves performance for smaller models.

## Page 5

Input Operator tcns
22423 conv2d - 32 12
112232 bottleneck 1 16 11
112216 bottleneck 6 24 22
56224 bottleneck 6 32 32
28232 bottleneck 6 64 42
14264 bottleneck 6 96 31
14296 bottleneck 6 160 32
72160 bottleneck 6 320 11
72320 conv2d 1x1 -1280 11
721280 avgpool 7x7 - - 1-
111280 conv2d 1x1 - k -
Table 2: MobileNetV2 : Each line describes a sequence
of 1 or more identical (modulo stride) layers, repeated
ntimes. All layers in the same sequence have the same
numbercof output channels. The ﬁrst layer of each
sequence has a stride sand all others use stride 1. All
spatial convolutions use 33kernels. The expansion
factortis always applied to the input size as described
in Table 1.
Size MobileNetV1 MobileNetV2 ShufﬂeNet
(2x,g=3)
112x112 64/1600 16/400 32/800
56x56 128/800 32/200 48/300
28x28 256/400 64/100 400/600K
14x14 512/200 160/62 800/310
7x7 1024/199 320/32 1600/156
1x1 1024/2 1280/2 1600/3
max 1600K 400K 600K
Table 3: The max number of channels/memory (in
Kb) that needs to be materialized at each spatial res-
olution for different architectures. We assume 16-bit
ﬂoats for activations. For ShufﬂeNet, we use 2x;g=
3that matches the performance of MobileNetV1 and
MobileNetV2. For the ﬁrst layer of MobileNetV2 and
ShufﬂeNet we can employ the trick described in Sec-
tion 5 to reduce memory requirement. Even though
ShufﬂeNet employs bottlenecks elsewhere, the non-
bottleneck tensors still need to be materialized due to the
presence of shortcuts between the non-bottleneck ten-
sors.
5. Implementation Notes
5.1. Memory efﬁcient inference
The inverted residual bottleneck layers allow a partic-
ularly memory efﬁcient implementation which is very
important for mobile applications. A standard efﬁ-
(a) NasNet[23]
inputDwise 3x3,
stride=s, Relu6conv 1x1, Relu6 (b) MobileNet[27]
(c) ShufﬂeNet [20]
Conv 1x1, Relu6Dwise 3x3, Relu6
inputconv 1x1, LinearAdd
Conv 1x1, Relu6Dwise 3x3,
stride=2, Relu6
inputconv 1x1, Linear
Stride=1 block Stride=2 block (d) Mobilenet V2
Figure 4: Comparison of convolutional blocks for dif-
ferent architectures. ShufﬂeNet uses Group Convolu-
tions [20] and shufﬂing, it also uses conventional resid-
ual approach where inner blocks are narrower than out-
put. ShufﬂeNet and NasNet illustrations are from re-
spective papers.
cient implementation of inference that uses for instance
TensorFlow[31] or Caffe [32], builds a directed acyclic
compute hypergraph G, consisting of edges represent-
ing the operations and nodes representing tensors of in-
termediate computation. The computation is scheduled
in order to minimize the total number of tensors that
needs to be stored in memory. In the most general case,
it searches over all plausible computation orders (G)
and picks the one that minimizes
M(G) = min
2(G)max
i21::n2
4X
A2R(i;;G )jAj3
5+size(i):
whereR(i;;G )is the list of intermediate tensors that
are connected to any of i:::nnodes,jAjrepresents
the size of the tensor Aandsize(i)is the total amount
of memory needed for internal storage during operation
i.
For graphs that have only trivial parallel structure
(such as residual connection), there is only one non-
trivial feasible computation order, and thus the total
amount and a bound on the memory needed for infer-

## Page 6

ence on compute graph Gcan be simpliﬁed:
M(G) = max
op2G2
4X
A2opinpjAj+X
B2opoutjBj+jopj3
5
(2)
Or to restate, the amount of memory is simply the max-
imum total size of combined inputs and outputs across
all operations. In what follows we show that if we treat
a bottleneck residual block as a single operation (and
treat inner convolution as a disposable tensor), the total
amount of memory would be dominated by the size of
bottleneck tensors, rather than the size of tensors that are
internal to bottleneck (and much larger).
Bottleneck Residual Block A bottleneck block oper-
atorF(x)shown in Figure 3b can be expressed as a
composition of three operators F(x) = [ANB]x,
whereAis a linear transformation A:Rssk!
Rssn,Nis a non-linear per-channel transformation:
N:Rssn! Rs0s0n, andBis again a linear
transformation to the output domain: B:Rs0s0n!
Rs0s0k0.
For our networksN= ReLU6dwiseReLU6 ,
but the results apply to any per-channel transformation.
Suppose the size of the input domain is jxjand the size
of the output domain is jyj, then the memory required
to compute F(X)can be as low asjs2kj+js02k0j+
O(max(s2;s02)).
The algorithm is based on the fact that the inner ten-
sorIcan be represented as concatenation of ttensors, of
sizen=teach and our function can then be represented
as
F(x) =tX
i=1(AiNBi)(x)
by accumulating the sum, we only require one interme-
diate block of size n=tto be kept in memory at all times.
Usingn=twe end up having to keep only a single
channel of the intermediate representation at all times.
The two constraints that enabled us to use this trick is
(a) the fact that the inner transformation (which includes
non-linearity and depthwise) is per-channel, and (b) the
consecutive non-per-channel operators have signiﬁcant
ratio of the input size to the output. For most of the tra-
ditional neural networks, such trick would not produce a
signiﬁcant improvement.
We note that, the number of multiply-adds opera-
tors needed to compute F(X)usingt-way split is in-
dependent of t, however in existing implementations we
ﬁnd that replacing one matrix multiplication with sev-
eral smaller ones hurts runtime performance due to in-
7.5 10 15 20 30 40 50 75 100 150 200 300 400 500600
Multiply-Adds, Millions35.037.540.042.545.047.550.052.555.057.560.062.565.067.570.072.575.077.5Accuracy, Top 1, %V2 1.0V2 1.496x96
128x128
160x160
192x192
224x224
NasNet
MobileNetV1
ShuffleNetFigure 5: Performance curve of MobileNetV2 vs
MobileNetV1, ShufﬂeNet, NAS. For our networks we
use multipliers 0:35,0:5,0:75,1:0for all resolutions,
and additional 1:4for for 224. Best viewed in color.
0 1 2 3 4 5 6 7
Step, millions66676869707172Top 1 Accuracy
Linear botleneck
Relu6 in bottleneck
(a) Impact of non-linearity in
the bottleneck layer.
0 1 2 3 4 5 6 7
Step, millions66676869707172Top 1 Accuracy
Shortcut between bottlenecks
Shortcut between expansions
No residual(b) Impact of variations in
residual blocks.
Figure 6: The impact of non-linearities and various
types of shortcut (residual) connections.
creased cache misses. We ﬁnd that this approach is the
most helpful to be used with tbeing a small constant
between 2and5. It signiﬁcantly reduces the memory
requirement, but still allows one to utilize most of the ef-
ﬁciencies gained by using highly optimized matrix mul-
tiplication and convolution operators provided by deep
learning frameworks. It remains to be seen if special
framework level optimization may lead to further run-
time improvements.
6. Experiments
6.1. ImageNet Classiﬁcation
Training setup We train our models using
TensorFlow[31]. We use the standard RMSPropOp-
timizer with both decay and momentum set to 0:9.
We use batch normalization after every layer, and the
standard weight decay is set to 0:00004 . Following
MobileNetV1[27] setup we use initial learning rate of
0:045, and learning rate decay rate of 0:98per epoch.
We use 16 GPU asynchronous workers, and a batch size
of96.

## Page 7

Results We compare our networks against
MobileNetV1, ShufﬂeNet and NASNet-A models.
The statistics of a few selected models is shown in
Table 4 with the full performance graph shown in
Figure 5.
6.2. Object Detection
We evaluate and compare the performance of
MobileNetV2 and MobileNetV1 as feature extractors
[33] for object detection with a modiﬁed version of the
Single Shot Detector (SSD) [34] on COCO dataset [2].
We also compare to YOLOv2 [35] and original SSD
(with VGG-16 [6] as base network) as baselines. We do
not compare performance with other architectures such
as Faster-RCNN [36] and RFCN [37] since our focus is
on mobile/real-time models.
SSDLite : In this paper, we introduce a mobile
friendly variant of regular SSD. We replace all the regu-
lar convolutions with separable convolutions (depthwise
followed by 11projection) in SSD prediction lay-
ers. This design is in line with the overall design of
MobileNets and is seen to be much more computation-
ally efﬁcient. We call this modiﬁed version SSDLite.
Compared to regular SSD, SSDLite dramatically re-
duces both parameter count and computational cost as
shown in Table 5.
For MobileNetV1, we follow the setup in [33]. For
MobileNetV2, the ﬁrst layer of SSDLite is attached to
the expansion of layer 15 (with output stride of 16). The
second and the rest of SSDLite layers are attached on top
of the last layer (with output stride of 32). This setup is
consistent with MobileNetV1 as all layers are attached
to the feature map of the same output strides.
Network Top 1 Params MAdds CPU
MobileNetV1 70.6 4.2M 575M 113ms
ShufﬂeNet (1.5) 71.5 3.4M 292M -
ShufﬂeNet (x2) 73.7 5.4M 524M -
NasNet-A 74.0 5.3M 564M 183ms
MobileNetV2 72.0 3.4M 300M 75ms
MobileNetV2 (1.4) 74.7 6.9M 585M 143ms
Table 4: Performance on ImageNet, comparison for dif-
ferent networks. As is common practice for ops, we
count the total number of Multiply-Adds. In the last
column we report running time in milliseconds (ms) for
a single large core of the Google Pixel 1 phone (using
TF-Lite). We do not report ShufﬂeNet numbers as efﬁ-
cient group convolutions and shufﬂing are not yet sup-
ported.Params MAdds
SSD[34] 14.8M 1.25B
SSDLite 2.1M 0.35B
Table 5: Comparison of the size and the computa-
tional cost between SSD and SSDLite conﬁgured with
MobileNetV2 and making predictions for 80classes.
Network mAP Params MAdd CPU
SSD300[34] 23.2 36.1M 35.2B -
SSD512[34] 26.8 36.1M 99.5B -
YOLOv2[35] 21.6 50.7M 17.5B -
MNet V1 + SSDLite 22.2 5.1M 1.3B 270ms
MNet V2 + SSDLite 22.1 4.3M 0.8B 200ms
Table 6: Performance comparison of MobileNetV2 +
SSDLite and other realtime detectors on the COCO
dataset object detection task. MobileNetV2 + SSDLite
achieves competitive accuracy with signiﬁcantly fewer
parameters and smaller computational complexity. All
models are trained on trainval35k and evaluated on
test-dev . SSD/YOLOv2 numbers are from [35]. The
running time is reported for the large core of the Google
Pixel 1 phone, using an internal version of the TF-Lite
engine.
Both MobileNet models are trained and evalu-
ated with Open Source TensorFlow Object Detection
API [38]. The input resolution of both models is 320
320. We benchmark and compare both mAP (COCO
challenge metrics), number of parameters and number
of Multiply-Adds. The results are shown in Table 6.
MobileNetV2 SSDLite is not only the most efﬁcient
model, but also the most accurate of the three. No-
tably, MobileNetV2 SSDLite is 20more efﬁcient and
10smaller while still outperforms YOLOv2 on COCO
dataset.
6.3. Semantic Segmentation
In this section, we compare MobileNetV1 and
MobileNetV2 models used as feature extractors with
DeepLabv3 [39] for the task of mobile semantic seg-
mentation. DeepLabv3 adopts atrous convolution [40,
41, 42], a powerful tool to explicitly control the reso-
lution of computed feature maps, and builds ﬁve paral-
lel heads including (a) Atrous Spatial Pyramid Pooling
module (ASPP) [43] containing three 33convolu-
tions with different atrous rates, (b) 11convolution
head, and (c) Image-level features [44]. We denote by

## Page 8

output stride the ratio of input image spatial resolution
to ﬁnal output resolution, which is controlled by apply-
ing the atrous convolution properly. For semantic seg-
mentation, we usually employ output stride = 16 or8
for denser feature maps. We conduct the experiments
on the PASCAL VOC 2012 dataset [3], with extra anno-
tated images from [45] and evaluation metric mIOU.
To build a mobile model, we experimented with three
design variations: (1) different feature extractors, (2)
simplifying the DeepLabv3 heads for faster computa-
tion, and (3) different inference strategies for boost-
ing the performance. Our results are summarized in
Table 7. We have observed that: (a) the inference
strategies, including multi-scale inputs and adding left-
right ﬂipped images, signiﬁcantly increase the MAdds
and thus are not suitable for on-device applications,
(b) using output stride = 16 is more efﬁcient than
output stride = 8, (c) MobileNetV1 is already a pow-
erful feature extractor and only requires about 4:9 5:7
times fewer MAdds than ResNet-101 [8] ( e.g., mIOU:
78.56 vs82.70, and MAdds: 941.9B vs4870.6B), (d)
it is more efﬁcient to build DeepLabv3 heads on top of
the second last feature map of MobileNetV2 than on the
original last-layer feature map, since the second to last
feature map contains 320channels instead of 1280 , and
by doing so, we attain similar performance, but require
about 2:5times fewer operations than the MobileNetV1
counterparts, and (e) DeepLabv3 heads are computa-
tionally expensive and removing the ASPP module sig-
niﬁcantly reduces the MAdds with only a slight perfor-
mance degradation. In the end of the Table 7, we identify
a potential candidate for on-device applications (in bold
face), which attains 75:32% mIOU and only requires
2:75B MAdds.
6.4. Ablation study
Inverted residual connections. The importance of
residual connection has been studied extensively [8,
30, 46]. The new result reported in this paper is that
the shortcut connecting bottleneck perform better than
shortcuts connecting the expanded layers (see Figure 6b
for comparison).
Importance of linear bottlenecks. The linear bottle-
neck models are strictly less powerful than models with
non-linearities, because the activations can always op-
erate in linear regime with appropriate changes to bi-
ases and scaling. However our experiments shown in
Figure 6a indicate that linear bottlenecks improve per-
formance, providing support that non-linearity destroys
information in low-dimensional space.Network OS ASPP MF mIOU Params MAdds
MNet V1 16X 75.29 11.15M 14.25B
8X X 78.56 11.15M 941.9B
MNet V2* 16X 75.70 4.52M 5.8B
8X X 78.42 4.52M 387B
MNet V2* 16 75.32 2.11M 2.75B
8X 77.33 2.11M 152.6B
ResNet-101 16X 80.49 58.16M 81.0B
8X X 82.70 58.16M 4870.6B
Table 7: MobileNet + DeepLabv3 inference strategy
on the PASCAL VOC 2012 validation set. MNet
V2*: Second last feature map is used for DeepLabv3
heads, which includes (1) Atrous Spatial Pyramid Pool-
ing ( ASPP ) module, and (2) 11convolution as well
as image-pooling feature. OS:output stride that con-
trols the output resolution of the segmentation map. MF:
Multi-scale and left-right ﬂipped inputs during test. All
of the models have been pretrained on COCO. The po-
tential candidate for on-device applications is shown in
bold face. PASCAL images have dimension 512512
and atrous convolution allows us to control output fea-
ture resolution without increasing the number of param-
eters.
7. Conclusions and future work
We described a very simple network architecture that
allowed us to build a family of highly efﬁcient mobile
models. Our basic building unit, has several proper-
ties that make it particularly suitable for mobile appli-
cations. It allows very memory-efﬁcient inference and
relies utilize standard operations present in all neural
frameworks.
For the ImageNet dataset, our architecture improves
the state of the art for wide range of performance points.
For object detection task, our network outperforms
state-of-art realtime detectors on COCO dataset both in
terms of accuracy and model complexity. Notably, our
architecture combined with the SSDLite detection mod-
ule is 20less computation and 10less parameters
than YOLOv2.
On the theoretical side: the proposed convolutional
block has a unique property that allows to separate the
network expressiveness (encoded by expansion layers)
from its capacity (encoded by bottleneck inputs). Ex-
ploring this is an important direction for future research.
Acknowledgments We would like to thank Matt
Streeter and Sergey Ioffe for their helpful feedback and
discussion.

## Page 9

References
[1] Olga Russakovsky, Jia Deng, Hao Su, Jonathan
Krause, Sanjeev Satheesh, Sean Ma, Zhiheng
Huang, Andrej Karpathy, Aditya Khosla, Michael
Bernstein, Alexander C. Berg, and Li Fei-Fei. Im-
agenet large scale visual recognition challenge.
Int. J. Comput. Vision , 115(3):211–252, December
2015. 1
[2] Tsung-Yi Lin, Michael Maire, Serge Belongie,
James Hays, Pietro Perona, Deva Ramanan, Piotr
Doll´ar, and C Lawrence Zitnick. Microsoft COCO:
Common objects in context. In ECCV , 2014. 1, 7
[3] Mark Everingham, S. M. Ali Eslami, Luc Van
Gool, Christopher K. I. Williams, John Winn, and
Andrew Zisserma. The pascal visual object classes
challenge a retrospective. IJCV , 2014. 1, 8
[4] Mobilenetv2 source code. Available from
https://github.com/tensorflow/
models/tree/master/research/slim/
nets/mobilenet . 1
[5] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E.
Hinton. Imagenet classiﬁcation with deep convolu-
tional neural networks. In Bartlett et al. [48], pages
1106–1114. 1
[6] Karen Simonyan and Andrew Zisserman. Very
deep convolutional networks for large-scale image
recognition. CoRR , abs/1409.1556, 2014. 1, 7
[7] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre
Sermanet, Scott E. Reed, Dragomir Anguelov, Du-
mitru Erhan, Vincent Vanhoucke, and Andrew Ra-
binovich. Going deeper with convolutions. In
IEEE Conference on Computer Vision and Pattern
Recognition, CVPR 2015, Boston, MA, USA, June
7-12, 2015 , pages 1–9. IEEE Computer Society,
2015. 1
[8] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and
Jian Sun. Deep residual learning for image recog-
nition. CoRR , abs/1512.03385, 2015. 1, 3, 4, 8
[9] James Bergstra and Yoshua Bengio. Random
search for hyper-parameter optimization. Journal
of Machine Learning Research , 13:281–305, 2012.
1
[10] Jasper Snoek, Hugo Larochelle, and Ryan P.
Adams. Practical bayesian optimization of ma-
chine learning algorithms. In Bartlett et al. [48],
pages 2960–2968. 1[11] Jasper Snoek, Oren Rippel, Kevin Swersky, Ryan
Kiros, Nadathur Satish, Narayanan Sundaram, Md.
Mostofa Ali Patwary, Prabhat, and Ryan P. Adams.
Scalable bayesian optimization using deep neu-
ral networks. In Francis R. Bach and David M.
Blei, editors, Proceedings of the 32nd Interna-
tional Conference on Machine Learning, ICML
2015, Lille, France, 6-11 July 2015 , volume 37
ofJMLR Workshop and Conference Proceedings ,
pages 2171–2180. JMLR.org, 2015. 1
[12] Babak Hassibi and David G. Stork. Second or-
der derivatives for network pruning: Optimal brain
surgeon. In Stephen Jose Hanson, Jack D. Cowan,
and C. Lee Giles, editors, Advances in Neural In-
formation Processing Systems 5, [NIPS Confer-
ence, Denver, Colorado, USA, November 30 - De-
cember 3, 1992] , pages 164–171. Morgan Kauf-
mann, 1992. 2
[13] Yann LeCun, John S. Denker, and Sara A. Solla.
Optimal brain damage. In David S. Touretzky,
editor, Advances in Neural Information Process-
ing Systems 2, [NIPS Conference, Denver, Col-
orado, USA, November 27-30, 1989] , pages 598–
605. Morgan Kaufmann, 1989. 2
[14] Song Han, Jeff Pool, John Tran, and William J.
Dally. Learning both weights and connec-
tions for efﬁcient neural network. In Corinna
Cortes, Neil D. Lawrence, Daniel D. Lee, Masashi
Sugiyama, and Roman Garnett, editors, Advances
in Neural Information Processing Systems 28: An-
nual Conference on Neural Information Process-
ing Systems 2015, December 7-12, 2015, Mon-
treal, Quebec, Canada , pages 1135–1143, 2015.
2
[15] Song Han, Jeff Pool, Sharan Narang, Huizi Mao,
Shijian Tang, Erich Elsen, Bryan Catanzaro, John
Tran, and William J. Dally. DSD: regulariz-
ing deep neural networks with dense-sparse-dense
training ﬂow. CoRR , abs/1607.04381, 2016. 2
[16] Yiwen Guo, Anbang Yao, and Yurong Chen. Dy-
namic network surgery for efﬁcient dnns. In
Daniel D. Lee, Masashi Sugiyama, Ulrike von
Luxburg, Isabelle Guyon, and Roman Garnett, ed-
itors, Advances in Neural Information Processing
Systems 29: Annual Conference on Neural Infor-
mation Processing Systems 2016, December 5-10,
2016, Barcelona, Spain , pages 1379–1387, 2016.
2

## Page 10

[17] Hao Li, Asim Kadav, Igor Durdanovic, Hanan
Samet, and Hans Peter Graf. Pruning ﬁlters for
efﬁcient convnets. CoRR , abs/1608.08710, 2016.
2
[18] Karim Ahmed and Lorenzo Torresani. Connec-
tivity learning in multi-branch networks. CoRR ,
abs/1709.09582, 2017. 2
[19] Tom Veniat and Ludovic Denoyer. Learning time-
efﬁcient deep architectures with budgeted super
networks. CoRR , abs/1706.00046, 2017. 2
[20] Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, and
Jian Sun. Shufﬂenet: An extremely efﬁcient
convolutional neural network for mobile devices.
CoRR , abs/1707.01083, 2017. 2, 5
[21] Soravit Changpinyo, Mark Sandler, and Andrey
Zhmoginov. The power of sparsity in convolu-
tional neural networks. CoRR , abs/1702.06257,
2017. 2
[22] Min Wang, Baoyuan Liu, and Hassan Foroosh.
Design of efﬁcient convolutional layers using sin-
gle intra-channel convolution, topological subdivi-
sioning and spatial ”bottleneck” structure. CoRR ,
abs/1608.04337, 2016. 2
[23] Barret Zoph, Vijay Vasudevan, Jonathon Shlens,
and Quoc V . Le. Learning transferable archi-
tectures for scalable image recognition. CoRR ,
abs/1707.07012, 2017. 2, 5
[24] Lingxi Xie and Alan L. Yuille. Genetic CNN.
CoRR , abs/1703.01513, 2017. 2
[25] Esteban Real, Sherry Moore, Andrew Selle,
Saurabh Saxena, Yutaka Leon Suematsu, Jie Tan,
Quoc V . Le, and Alexey Kurakin. Large-scale
evolution of image classiﬁers. In Doina Pre-
cup and Yee Whye Teh, editors, Proceedings of
the 34th International Conference on Machine
Learning, ICML 2017, Sydney, NSW, Australia,
6-11 August 2017 , volume 70 of Proceedings of
Machine Learning Research , pages 2902–2911.
PMLR, 2017. 2
[26] Barret Zoph and Quoc V . Le. Neural architec-
ture search with reinforcement learning. CoRR ,
abs/1611.01578, 2016. 2
[27] Andrew G. Howard, Menglong Zhu, Bo Chen,
Dmitry Kalenichenko, Weijun Wang, Tobias
Weyand, Marco Andreetto, and Hartwig Adam.Mobilenets: Efﬁcient convolutional neural net-
works for mobile vision applications. CoRR ,
abs/1704.04861, 2017. 2, 4, 5, 6
[28] Francois Chollet. Xception: Deep learning
with depthwise separable convolutions. In The
IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , July 2017. 2
[29] Dongyoon Han, Jiwhan Kim, and Junmo Kim.
Deep pyramidal residual networks. CoRR ,
abs/1610.02915, 2016. 3
[30] Saining Xie, Ross B. Girshick, Piotr Doll ´ar,
Zhuowen Tu, and Kaiming He. Aggregated
residual transformations for deep neural networks.
CoRR , abs/1611.05431, 2016. 3, 4, 8
[31] Mart ´ın Abadi, Ashish Agarwal, Paul Barham,
Eugene Brevdo, Zhifeng Chen, Craig Citro,
Greg S. Corrado, Andy Davis, Jeffrey Dean,
Matthieu Devin, Sanjay Ghemawat, Ian Goodfel-
low, Andrew Harp, Geoffrey Irving, Michael Isard,
Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser,
Manjunath Kudlur, Josh Levenberg, Dan Man ´e,
Rajat Monga, Sherry Moore, Derek Murray, Chris
Olah, Mike Schuster, Jonathon Shlens, Benoit
Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,
Vincent Vanhoucke, Vijay Vasudevan, Fernanda
Vi´egas, Oriol Vinyals, Pete Warden, Martin Wat-
tenberg, Martin Wicke, Yuan Yu, and Xiaoqiang
Zheng. TensorFlow: Large-scale machine learning
on heterogeneous systems, 2015. Software avail-
able from tensorﬂow.org. 5, 6
[32] Yangqing Jia, Evan Shelhamer, Jeff Donahue,
Sergey Karayev, Jonathan Long, Ross Girshick,
Sergio Guadarrama, and Trevor Darrell. Caffe:
Convolutional architecture for fast feature embed-
ding. arXiv preprint arXiv:1408.5093 , 2014. 5
[33] Jonathan Huang, Vivek Rathod, Chen Sun, Men-
glong Zhu, Anoop Korattikara, Alireza Fathi,
Ian Fischer, Zbigniew Wojna, Yang Song, Sergio
Guadarrama, et al. Speed/accuracy trade-offs for
modern convolutional object detectors. In CVPR ,
2017. 7
[34] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
Christian Szegedy, Scott Reed, Cheng-Yang Fu,
and Alexander C Berg. Ssd: Single shot multibox
detector. In ECCV , 2016. 7

## Page 11

[35] Joseph Redmon and Ali Farhadi. Yolo9000:
Better, faster, stronger. arXiv preprint
arXiv:1612.08242 , 2016. 7
[36] Shaoqing Ren, Kaiming He, Ross Girshick, and
Jian Sun. Faster r-cnn: Towards real-time object
detection with region proposal networks. In Ad-
vances in neural information processing systems ,
pages 91–99, 2015. 7
[37] Jifeng Dai, Yi Li, Kaiming He, and Jian Sun. R-
fcn: Object detection via region-based fully con-
volutional networks. In Advances in neural infor-
mation processing systems , pages 379–387, 2016.
7
[38] Jonathan Huang, Vivek Rathod, Derek Chow,
Chen Sun, and Menglong Zhu. Tensorﬂow object
detection api, 2017. 7
[39] Liang-Chieh Chen, George Papandreou, Florian
Schroff, and Hartwig Adam. Rethinking atrous
convolution for semantic image segmentation.
CoRR , abs/1706.05587, 2017. 7
[40] Matthias Holschneider, Richard Kronland-
Martinet, Jean Morlet, and Ph Tchamitchian.
A real-time algorithm for signal analysis with
the help of the wavelet transform. In Wavelets:
Time-Frequency Methods and Phase Space , pages
289–297. 1989. 7
[41] Pierre Sermanet, David Eigen, Xiang Zhang,
Micha ¨el Mathieu, Rob Fergus, and Yann Le-
Cun. Overfeat: Integrated recognition, localiza-
tion and detection using convolutional networks.
arXiv:1312.6229 , 2013. 7
[42] George Papandreou, Iasonas Kokkinos, and Pierre-
Andre Savalle. Modeling local and global defor-
mations in deep learning: Epitomic convolution,
multiple instance learning, and sliding window de-
tection. In CVPR , 2015. 7
[43] Liang-Chieh Chen, George Papandreou, Iasonas
Kokkinos, Kevin Murphy, and Alan L Yuille.
Deeplab: Semantic image segmentation with deep
convolutional nets, atrous convolution, and fully
connected crfs. TPAMI , 2017. 7
[44] Wei Liu, Andrew Rabinovich, and Alexander C.
Berg. Parsenet: Looking wider to see better. CoRR ,
abs/1506.04579, 2015. 7[45] Bharath Hariharan, Pablo Arbel ´aez, Lubomir
Bourdev, Subhransu Maji, and Jitendra Malik. Se-
mantic contours from inverse detectors. In ICCV ,
2011. 8
[46] Christian Szegedy, Sergey Ioffe, and Vincent Van-
houcke. Inception-v4, inception-resnet and the im-
pact of residual connections on learning. CoRR ,
abs/1602.07261, 2016. 8
[47] Guido Mont ´ufar, Razvan Pascanu, Kyunghyun
Cho, and Yoshua Bengio. On the number of linear
regions of deep neural networks. In Proceedings of
the 27th International Conference on Neural Infor-
mation Processing Systems , NIPS’14, pages 2924–
2932, Cambridge, MA, USA, 2014. MIT Press. 13
[48] Peter L. Bartlett, Fernando C. N. Pereira, Christo-
pher J. C. Burges, L ´eon Bottou, and Kilian Q.
Weinberger, editors. Advances in Neural Infor-
mation Processing Systems 25: 26th Annual Con-
ference on Neural Information Processing Systems
2012. Proceedings of a meeting held December 3-
6, 2012, Lake Tahoe, Nevada, United States , 2012.
9
A. Bottleneck transformation
In this section we study the properties of an operator
AReLU(Bx), wherex2Rnrepresents an n-channel
pixel,Bis anmnmatrix andAis annmmatrix.
We argue that if mn, transformations of this form
can only exploit non-linearity at the cost of losing infor-
mation. In contrast, if nm, such transforms can be
highly non-linear but still invertible with high probabil-
ity (for the initial random weights).
First we show that ReLU is an identity transforma-
tion for any point that lies in the interior of its image.
Lemma 1 LetS(X) =fReLU(x)jx2Xg. If a vol-
ume ofS(X)is non-zero, then interiorS(X)X.
Proof: LetS0= interior ReLU( S). First we note that
ifx2S0, thenxi>0for alli. Indeed, image of ReLU
does not contain points with negative coordinates, and
points with zero-valued coordinates can not be interior
points. Therefore for each x2S0,x= ReLU(x)as
desired.
It follows that for an arbitrary composition of inter-
leaved linear transformation and ReLU operators, if it
preserves non-zero volume, that part of the input space
Xthat is preserved over such a composition is a lin-
ear transformation, and thus is likely to be a minor con-
tributor to the power of deep networks. However, this

## Page 12

is a fairly weak statement. Indeed, if the input mani-
fold can be embedded into (n 1)-dimensional mani-
fold (out of ndimensions total), the lemma is trivially
true, since the starting volume is 0. In what follows we
show that when the dimensionality of input manifold is
signiﬁcantly lower we can ensure that there will be no
information loss.
Since the ReLU(x)nonlinearity is a surjective func-
tion mapping the entire ray x0to0, using this nonlin-
earity in a neural network can result in information loss.
Once ReLU collapses a subset of the input manifold to a
smaller-dimensional output, the following network lay-
ers can no longer distinguish between collapsed input
samples. In the following, we show that bottlenecks with
sufﬁciently large expansion layers are resistant to infor-
mation loss caused by the presence of ReLU activation
functions.
Lemma 2 (Invertibility of ReLU) Consider an opera-
torReLU(Bx), whereBis anmnmatrix andx2
Rn. Lety0= ReLU(Bx0)for somex02Rn, then
equationy0= ReLU(Bx)has a unique solution with
respect toxif and only if y0has at leastnnon-zero val-
ues and there are nlinearly independent rows of Bthat
correspond to non-zero coordinates of y0.
Proof: Denote the set of non-zero coordinates of y0
asTand letyTandBTbe restrictions of yandB
to the subspace deﬁned by T. IfjTj< n , we have
yT=BTx0whereBTis under-determined with at
least one solution x0, thus there are inﬁnitely many so-
lutions. Now consider the case of jTjnand let the
rank ofBTben. Suppose there is an additional solu-
tionx16=x0such thaty0= ReLU(Bx1), then we have
yT=BTx0=BTx1, which cannot be satisﬁed unless
x0=x1.
One of the corollaries of this lemma says that if m
n, we only need a small fraction of values of Bxto be
positive for ReLU(Bx)to be invertible.
The constraints of the lemma 2 can be empirically
validated for real networks and real inputs and hence we
can be assured that information is indeed preserved. We
further show that with respect to initialization, we can be
sure that these constraints are satisﬁed with high proba-
bility. Note that for random initialization the conditions
of lemma 2 are satisﬁed due to initialization symmetries.
However even for trained graphs these constraints can be
empirically validated by running the network over valid
inputs and verifying that all or most inputs are above
the threshold. On Figure 7 we show how this distribu-
tion looks for different MobileNetV2 layers. At step 0
the activation patterns concentrate around having half ofthe positive channel (as predicted by initialization sym-
metries). For fully trained network, while the standard
deviation grew signiﬁcantly, all but the two layers are
still above the invertibility thresholds. We believe fur-
ther study of this is warranted and might lead to helpful
insights on network design.
Theorem 1 LetSbe a compact n-dimensional subman-
ifold ofRn. Consider a family of functions fB(x) =
ReLU(Bx)fromRntoRmparameterized by mn
matricesB2B. Letp(B)be a probability density on
the space of all matrices Bthat satisﬁes:
P(Z) = 0 for any measure-zero subset ZB;
(a symmetry condition) p(DB) =p(B)for any
B2B and anymmdiagonal matrix Dwith
all diagonal elements being either +1or 1.
Then, the average n-volume of the subset of Sthat is
collapsed by fBto a lower-dimensional manifold is
V Nm;nV
2m;
whereV= volSand
Nm;nm nX
k=0m
k
:
Proof: For any= (s1;:::;sm)withsk2
f 1;+1g, letQ=fx2Rmjxisi>0gbe a corre-
sponding quadrant in Rm. For anyn-dimensional sub-
manifold  Rm,ReLU acts as a bijection on  \Q
ifhas at leastnpositive values4and contracts  \Q
otherwise. Also notice that the intersection of BSwith
Rmn([Q)is almost surely (n 1)-dimensional. The
averagen-volume ofSthat is not collapsed by applying
ReLU toBSis therefore given by:
X
2nEB[V(B)]; (3)
where n=f(s1;:::;sm)jP
k(sk)ng,is a step
function and V(B)is a volume of the largest subset
ofSthat is mapped by BtoQ. Now let us calcu-
lateEB[V(B)]. Recalling that p(DB) =p(B)for
anyD= diag(s1;:::;sm)withsk2f  1;+1g, this
average can be rewritten as EBED[V(DB)]. Notic-
ing that the subset of Smapped by DB toQis
also mapped by BtoD 1Q, we immediately obtain
4unless at least one of the positive coordinates for all x2 \Q
is ﬁxed, which would not be the case for almost all Band  =BS

## Page 13

0 2 4 6 8 10 12 14 16
Layer N02004006008001000Num positive filtersavg
min
max
total
threshold
0 2 4 6 8 10 12 14 16
Layer N0.10.20.30.40.50.60.70.80.91.0Fraction of filters(a) At step 0
0 2 4 6 8 10 12 14 16
Layer N02004006008001000Num positive filtersavg
min
max
total
threshold
0 2 4 6 8 10 12 14 16
Layer N0.00.20.40.60.81.0Fraction of filters(b) Fully trained
Figure 7: Distribution of activation patterns. The x-axis is the layer index, and we show minimum/maximum/average
number of positive channels after each convolution with ReLU .y-axis is either absolute or relative number of chan-
nels. The “threshold” line indicates the ReLU invertibility threshold - that is the number of positive dimensions is
higher than the input space. In our case this is 1=6fraction of the channels. Note how at the beginning of the train-
ing on Figure 7a the distribution is much more tightly concentrated around the mean. After the training has ﬁnished
(Figure 7b), the average hasn’t changed but the standard deviation grew dramatically. Best viewed in color.
P
0V[diag(0)B] =P
0V0[B] = volSand there-
foreEB[V(B)] = 2 mvolS. Substituting this and
jnj=Pm n
k=0 m
k
into Eq. 3 concludes the proof. 
Notice that for sufﬁciently large expansion layers
withmn, the fraction of collapsed space Nm;n=2m
can be bounded by:
Nm;n
2m1 mn+1
2mn!1 2(n+1) logm m1 2 m=2
and therefore ReLU(Bx)performs a nonlinear transfor-
mation while preserving information with high proba-
bility.
We discussed how bottlenecks can prevent manifold
collapse, but increasing the size of the bottleneck expan-
sion may also make it possible for the network to repre-
sent more complex functions. Following the main re-
sults of [47], one can show, for example, that for any in-
tegerL1andp>1there exist a network of LReLU
layers, each containing nneurons and a bottleneck ex-
pansion of size pnsuch that it maps pnLinput volumes
(linearly isomorphic to [0;1]n) to the same output re-
gion [0;1]n. Any complex possibly nonlinear function
attached to the network output would thus effectively
compute function values for pnLinput linear regions.
B. Semantic segmentation visualization re-
sults

## Page 14

Figure 8: MobileNetv2 semantic segmentation visualization results on PASCAL VOC 2012 valset.OS:output stride .
S: single scale input. MS+F : Multi-scale inputs with scales = f0:5;0:75;1;1:25;1:5;1:75gand left-right ﬂipped
inputs. Employing output stride = 16 and single input scale = 1 attains a good trade-off between FLOPS and accuracy.