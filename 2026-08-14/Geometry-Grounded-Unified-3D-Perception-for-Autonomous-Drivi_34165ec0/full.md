# Geometry-Grounded Unified 3D Perception for Autonomous Driving

Longfei Xu<sup>1,\*</sup> xulongfei@buaa.edu.cn

Xiaohui Wang<sup>2,\*</sup> nekomio@bupt.edu.cn

Zehao Huang zehaohuang18@gmail.com

Han Li<sup>3</sup> lihan0620@buaa.edu.cn

Ya Yang<sup>2</sup> yangya@bupt.edu.cn

Naiyan Wang winsty@gmail.com

Si Liu<sup>3,†</sup> liusi@buaa.edu.cn

<sup>1</sup> School of Computer Science and   
Engineering   
Beihang University   
Beijing, China

<sup>2</sup> School of Computer Science Beijing University of Posts and Telecommunications Beijing, China

<sup>3</sup> School of Artificial Intelligence Beihang University Beijing, China

## Abstract

Camera-based autonomous driving perception requires a shared representation that preserves metric 3D structure across synchronized multi-camera streams. However, existing image-based frameworks often rely on backbones pretrained for semantic recognition, and introduce 3D geometry through downstream task-specific modules. As a result, their shared representations may fail to preserve explicit metric geometry and consistent 3D scene structure. In this paper, we present a Geometry-grounded Unified 3D Perception (GeoUP) framework that adapts the reconstruction-oriented latent of VGGT to calibrated, streaming multi-camera driving scenes. GeoUP factorizes cross-image interaction into self, temporal, and view attention to capture structurally distinct temporal and cross-view correspondences. It further injects calibration-aware raymap encodings to provide metric scale and camera geometry. The resulting geometry-grounded latent is decoded for metric depth estimation, 3D object detection, and semantic occupancy prediction, corresponding to surface-, instance-, and volume-level readouts of the same 3D scene. Through joint multi-task and multi-dataset training, GeoUP effectively leverages heterogeneous annotations and generalizes across diverse sensor configurations and perception ranges. Extensive experiments on nuScenes, Argoverse 2, Waymo, KITTI, and DDAD demonstrate that GeoUP achieves SOTA performance across detection, occupancy, and depth estimation. These results validate the effectiveness of geometrygrounded representations for unified 3D driving perception. Our project page is available at https://buaa-colalab.github.io/geoup\_page.

![](images/44cddd663aba16adb747a0fd1fe410791d72a83dcfc53cef0402664963cf4cb3.jpg)  
Figure 1: Comparison of pretraining paradigms for camera-based autonomous driving perception. (a) Recognition pretraining, e.g., ImageNet [10] classification with ResNet-50 [18], learns semantic features but lacks geometry and multi-view consistency. (b) Geometry pretraining, e.g., monocular depth with VoVNet-99 [28], adds geometric awareness but lacks cross-view and temporal modeling. (c) 3D reconstruction pretraining, e.g., VGGT [63], learns from multi-image geometric objectives, yielding a semantically rich, geometrically grounded, and temporally consistent representation.

## 1 Introduction

A central challenge in camera-based autonomous driving perception is to recover a metric and temporally consistent 3D scene representation from synchronized multi-view camera streams. Core 3D perception tasks can be regarded as different readouts of the underlying scene: depth estimation predicts image-aligned surface geometry [16, 17], 3D object detection localizes metric object instances [21, 66, 69], and semantic occupancy prediction estimates voxel-wise 3D scene states [22, 71, 79]. Although these tasks differ in output format and granularity, they impose common requirements on the underlying representation, including semantic discriminability, metric geometry, multi-camera consistency, and temporal correspondence. This motivates a unified 3D perception representation [34, 39, 41, 73] that is organized around the 3D scene rather than being tailored to a specific perception task.

The quality of such a unified representation depends critically on the geometric capacity of the underlying backbone. Most camera-based driving perception systems, however, still build their representations on general-purpose image backbones pretrained for objectives designed primarily for semantic recognition. These backbones, pretrained by supervised classification [11, 18], self-supervised visual representation learning [20, 46], or vision-language alignment [49], provide semantically discriminative features, but do not explicitly encode metric 3D structure, as illustrated in Figure 1(a). Consequently, geometry is often introduced by task-specific downstream modules, such as depth heads, BEV lifting [21, 41], positional encodings [39, 69], or query-based geometric reasoning [34, 39]. Some methods further propose geometry-oriented pretraining, such as through monocular depth estimation [28, 32] or monocular 3D detection [47, 66, 73], which improves geometric awareness but does not yield a multi-view, temporally consistent scene representation, as shown in Figure 1(b). In contrast, visual geometry foundation models trained with multi-image 3D reconstruction objectives provide a more natural basis for learning such a representation, as shown in Figure 1(c). This motivates us to explore whether such a shared representation can capture a coherent metric 3D scene and support heterogeneous perception tasks.

Among recent visual geometry foundation models [27, 29, 65, 68, 81], VGGT [63] is a representative feed-forward model whose multi-image latent is structured by 3D geometry, while its DINOv2 [46] image encoder provides semantically discriminative features. This combination makes VGGT an appealing starting point for autonomous driving perception. However, vanilla VGGT [63] is not directly suited to calibrated streaming multi-camera driving scenes. First, it does not explicitly inject camera calibration priors, leaving its representation unaware of the absolute metric scale that driving perception requires. Second, its global cross-image attention is computationally expensive and does not distinguish temporal from cross-view interactions in driving scenes. These gaps motivate a driving-oriented adaptation with calibration-aware geometric encoding and structured spatiotemporal attention.

In this paper, we present a Geometry-grounded Unified 3D Perception (GeoUP) framework that adapts VGGT [63] to calibrated streaming multi-camera driving scenes. GeoUP retains the self-attention in VGGT [63] and decomposes its global cross-image attention into temporal and view attention. Self attention models image-level context, temporal attention captures correspondences within each camera stream, and view attention exchanges information across simultaneous cameras. To make the representation aware of camera geometry and absolute scale, GeoUP further injects patch-aligned raymap encodings [56, 80] derived from camera intrinsics and global camera poses. With these adaptations, the shared latent representation learned by GeoUP supports metric depth estimation, 3D object detection, and semantic occupancy prediction, corresponding to surface-, instance-, and volume-level understanding of the same metric scene.

We evaluate GeoUP on widely used driving benchmarks, including nuScenes [3], Argoverse 2 [72], Waymo [58], KITTI [15], and DDAD [17], where it achieves state-of-theart performance across detection, occupancy, and depth estimation. Furthermore, we introduce a joint multi-task and multi-dataset training strategy across these benchmarks, which strengthens the geometry-grounded representation and brings consistent gains. Beyond perception, we further evaluate GeoUP as the visual backbone of an end-to-end planner. On the NAVSIMv2 [4, 9] benchmark, GeoUP improves the EPDMS of DriveSuprim [75] over the Depth Anything [74] pretrained backbone. Together, these results demonstrate the effectiveness of adapting visual geometry foundation models for unified 3D driving perception and provide evidence that the learned representation transfers to downstream planning.

In summary, our contributions are:

• We propose GeoUP, a geometry-grounded unified 3D perception framework that adapts the reconstruction-oriented latent of VGGT [63] to autonomous driving and supports metric depth estimation, 3D object detection, and semantic occupancy prediction within a single architecture.

• We introduce driving-oriented adaptations for visual geometry foundation models, including temporal-view attention factorization and calibration-aware raymap injection, to handle streaming multi-camera inputs with structurally distinct temporal and crossview interactions.

• We conduct joint multi-task and multi-dataset training across widely used autonomous driving datasets, showing that a geometry-grounded latent can benefit from heterogeneous supervision and generalize across surface-, instance-, and volume-level 3D perception tasks.

## 2 Related Work

## 2.1 Camera-based Backbone Pretraining

Camera-based 3D perception commonly relies on general-purpose image backbones, such as CNNs and vision transformers pretrained by supervised recognition [11, 18, 40, 42], selfsupervised learning [6, 7, 19, 46], masked image modeling [1, 13, 14, 20], or vision-language supervision [49, 78]. These backbones provide strong semantic features, but their pretraining objectives mainly emphasize 2D appearance and semantic discrimination rather than metric 3D structure. To improve geometric awareness, some methods introduce depth, monocular 3D detection, or geometry-aware pretraining [47, 66, 73]. However, these strategies are often single-frame or task-specific, and do not explicitly form a multi-view, temporal, metric scene representation. In contrast, our method adapts a reconstruction-oriented visual geometry backbone to calibrated streaming driving scenes, making geometry part of the shared representation.

## 2.2 Task-specific Geometry Modeling

Metric geometry is widely used in camera-based depth estimation, 3D detection, and occupancy prediction. Depth estimation directly predicts image-aligned surface geometry [16, 17]; 3D detection introduces depth-aware lifting [21, 48], 3D query projection [30, 69, 70], 3D positional encoding [38, 39], ray modeling [36], and temporal fusion [34, 64] for metric object localization; occupancy prediction models voxel-wise or sparse point-wise scene states through volumetric, tri-plane, rendering-based, or query-based designs [22, 33, 37, 61, 71, 79]. Although effective, these methods usually model geometry inside task-specific pipelines. As a result, depth, detection, and occupancy may each learn separate geometric cues from a shared but weakly geometric image backbone. Our method instead treats them as complementary readouts from one geometry-grounded latent, covering surface-, instance-, and volume-level 3D perception.

## 2.3 Visual Geometry Models

Recent 3D reconstruction has shifted from specialized matching, structure from motion (SFM), and multi-view stereo (MVS) pipelines [51, 52, 57, 60, 76] toward visual geometry foundation models [2, 29, 62, 63, 65]. These models learn transferable geometric priors from reconstruction and correspondence objectives, and can predict cameras, depth, point maps, tracks, or dense correspondences from image inputs. VGGT [63] is a representative feed-forward model that jointly predicts multiple geometric outputs with a reconstructionoriented latent. However, such models are not directly tailored for autonomous driving. Driving data has calibrated multi-camera rigs, limited cross-view overlap, strong temporal continuity, and large metric-scale scenes. Recent VGGT-style [63] driving adaptations focus mainly on reconstruction [24, 82]. Different from them, our method uses the visual geometry latent as a shared perception backbone and decodes it for depth estimation, 3D object detection, and semantic occupancy prediction.

## 3 Method

We propose GeoUP, a unified perception framework that adapts VGGT [63] to calibrated streaming multi-camera driving scenes. As shown in Figure 2, GeoUP augments image patch tokens with calibration-derived raymap embeddings and camera tokens to form geometryaware input representations, which are then processed by a spatiotemporal backbone with self, temporal, and view attention. The resulting token features are decoded by task-specific heads for metric depth estimation, 3D object detection, and semantic occupancy prediction.

![](images/3d58ab68e9461ba0562d4c4320f6945cacaa87483c42e057b21f1a12346ad9d7.jpg)  
Figure 2: Overall pipeline of GeoUP. Given streaming multi-view inputs, GeoUP constructs geometry-aware tokens Z by combining image patch tokens X, raymap embeddings R, and camera tokens C. The tokens are processed by a factorized backbone with self, temporal, and view attention. The output patch tokens are used for depth estimation, 3D object detection, and occupancy prediction, while camera tokens are decoded for auxiliary camera pose prediction.

We describe the geometry-grounded backbone in Sec. 3.1, the perception heads in Sec. 3.2, and the training strategy in Sec. 3.3.

## 3.1 Geometry-Grounded Backbone

GeoUP takes as input synchronized surround-view images from V cameras over a temporal window of T frames. We denote the image from camera v at timestamp t as $\mathbf { I } _ { t } ^ { \nu } \in \mathbb { R } ^ { H \times \mathbf { \bar { W } } \times 3 }$ where $\nu = 1 , \ldots , V$ and $t = 1 , \ldots , T$ . The last timestamp $t = T$ is used as the current frame for downstream perception, and serves as the reference frame for each camera’s temporal window. Each image is divided into patches with patch size P and encoded by a DINOv2 encoder [46] into patch tokens ${ \bf X } _ { t } ^ { \nu } \in \mathbb { R } ^ { \bar { N } \times C }$ , where $N = H W / P ^ { 2 }$ and C is the token dimension. To avoid redundant computation, GeoUP caches historical patch tokens and only encodes the current-frame images at each step. The full spatiotemporal token set is denoted as $\mathcal { X } = \{ \mathbf { X } _ { t } ^ { \nu } \in$ $\mathbb R ^ { N \times C } \mid t = 1 , \ldots , T , \nu = 1 , \ldots , V \big \}$

To inject explicit camera geometry, we encode each patch center as a Plücker ray [56, 80] derived from the camera intrinsics and camera poses. For camera v at time t, the raymap feature at grid location $( x , y )$ is computed as:

$$
\mathbf { R } _ { t } ^ { \nu } ( x , y ) = \mathbf { M L P } \big ( \mathrm { P r o j } ( x , y , \mathbf { K } _ { t } ^ { \nu } , \mathbf { E } _ { t } ^ { \nu } ) \big ) ,\tag{1}
$$

where $\mathbf { K } _ { t } ^ { \nu } \in \mathbb { R } ^ { 4 \times 4 }$ and $\mathbf { E } _ { t } ^ { \nu } \in \mathbb { R } ^ { 4 \times 4 }$ denote the unnormalized camera intrinsic matrix and camera-to-reference poses (relative to camera v at $t = T )$ , respectively. Proj(·) constructs the 6D Plücker representation from the patch center and camera parameters, followed by an MLP projects the result to the token dimension, yielding $\mathbf { R } _ { t } ^ { \nu } \in \mathbb { R } ^ { N \times C }$ . The geometryenhanced token is then defined as $\tilde { \mathbf { X } } _ { t } ^ { \nu } = \mathbf { X } _ { t } ^ { \nu } + \mathbf { R } _ { t } ^ { \nu }$

Following VGGT [63], we prepend a camera token $\mathbf { C } _ { t } ^ { \nu } \in \mathbb { R } ^ { 1 \times C }$ and register tokens ${ \bf Q } _ { t } ^ { \nu } \in$ $\mathbb { R } ^ { 4 \times C }$ to each frame-view pair:

$$
\mathbf { Z } _ { t } ^ { \nu } = \operatorname { C o n c a t } ( \left[ \mathbf { C } _ { t } ^ { \nu } ; \mathbf { Q } _ { t } ^ { \nu } ; \tilde { \mathbf { X } } _ { t } ^ { \nu } \right] ) .\tag{2}
$$

In VGGT [63], the first input image serves as the reference frame and uses a dedicated learnable camera token, while all other frames share a separate one. We adapt this to the multi-camera setting by treating the current frame $( t = T )$ of each view as its reference frame. All reference frames share one learnable embedding, and all non-reference frames $( t < T )$ share another. The full set of tokens $\mathbf { Z } = \{ \mathbf { Z } _ { t } ^ { \nu } \mid t = 1 , \ldots , T , \nu = 1 , \ldots , V \}$ forms the initial backbone representation $\mathbf { F } ^ { ( 0 ) }$ . Then, $\mathbf { F } ^ { ( 0 ) }$ is refined through L Transformer layers that alternate between intra-image and cross-image attention. The original VGGT [63] design performs global cross-image attention over all input tokens, without distinguishing temporal and cross-view interactions. We instead factorize each layer into self-attention, temporal attention, and view attention. As illustrated in Figure 3, the input features are organized along token, view, and frame dimensions, and each attention block operates on a specific grouping of this token volume. Let $l \in \{ 1 , \ldots , L \}$ denote the index of a backbone layer. The layer-wise update is formulated as:

![](images/7bc36029af2bc9a6e46aa94808b417f51f15c6457129373b43d8fe4d511629f8.jpg)  
Figure 3: Illustration of the factorized attention in GeoUP Transformer layers. GeoUP applies self-attention within each image, temporal attention across frames from the same camera, and view attention across cameras at the same timestamp.

$$
\mathbf { F } ^ { ( l ) } = \mathrm { A t t n } _ { \mathrm { v i e w } } ^ { ( l ) } \Big ( \mathrm { A t t n } _ { \mathrm { t e m p } } ^ { ( l ) } \big ( \mathrm { A t t n } _ { \mathrm { s e l f } } ^ { ( l ) } ( \mathbf { F } ^ { ( l - 1 ) } ) \big ) \Big ) ,\tag{3}
$$

where $\mathrm { A t t n } _ { \mathrm { s e l f } }$ operates within each image, $\mathrm { A t t n } _ { \mathrm { t e m p } }$ aggregates tokens from the same camera across frames, and $\mathsf { A t t n } _ { \mathrm { v i e w } }$ exchanges information among cameras within each timestamp. This structured attention captures temporal continuity and cross-view geometric correspondence while avoiding the cost of full global attention.

After L layers, the backbone produces per-layer token features $\mathbf { F } _ { t } ^ { ( l ) , \nu }$ for each timestamp, view and layer. We extract the image patch tokens from each layer as $\bar { \mathbf X } _ { t } ^ { ( l ) , \nu }$ for downstream perception tasks, discarding the camera and register tokens. Each task $a \in \{ \mathrm { d e p } , \mathrm { d e t } , \mathrm { o c c } \}$ selects features from a task-specific subset of layer indices $\mathcal { T } _ { a }$

## 3.2 Multi-task Perception Heads

Given the shared GeoUP features, we attach task-specific readout heads for depth estimation, 3D object detection, occupancy prediction, and camera pose prediction. These heads mainly follow existing task designs, while GeoUP focuses on providing a unified geometrygrounded representation for different levels of perception.

Depth Head. For depth estimation, we use a DPT-based [50] dense prediction head to decode per-view metric depth from the selected current-frame GeoUP features $\{ \bar { \mathbf X } _ { T } ^ { ( l ) , \nu } \} _ { l \in \mathcal { T } _ { \mathrm { d e p } } }$ For each layer $l \in \mathcal { T } _ { \mathrm { d e p } }$ , we inject camera intrinsics into the patch tokens before depth prediction:

$$
\mathbf { Y } _ { T } ^ { ( l ) , d , \nu } = \bar { \mathbf { X } } _ { T } ^ { ( l ) , \nu } + \mathbf { M L P } \big ( \mathbf { K } _ { T } ^ { \nu } \big ) ,\tag{4}
$$

where ${ \bf K } _ { T } ^ { \nu } \in \mathbb { R } ^ { 4 \times 4 }$ is the camera intrinsic matrix. This explicit conditioning provides camera geometry information to the depth branch and helps reduce the domain gap caused by

different imaging geometries across datasets.

The depth head predicts a dense metric depth map $\mathbf { D } _ { T } ^ { \nu } \in \mathbb { R } ^ { H \times W }$ for each current-frame view:

$$
\mathbf { D } _ { T } ^ { \nu } = \mathcal { H } _ { \mathrm { d e p } } \big ( \{ \mathbf { Y } _ { T } ^ { ( l ) , d , \nu } \} _ { l \in \mathcal { T } _ { \mathrm { d e p } } } \big ) .\tag{5}
$$

The prediction is scale-normalized metric depth, which is converted to real-scale depth using a predefined depth scale.

Detection Head. For 3D object detection, we use a RayDN-based [36] query detection head as the detection readout. The selected current-frame features $\bar { \mathbf X } _ { T } ^ { ( l ) , \nu }$ are passed through a detection neck to produce $\mathbf { Y } _ { T } ^ { b , \nu }$ . Following Stream3DPPE [54], we further inject depth-derived 3D position embeddings into the detection features. Based on the features, the detection head predicts 3D boxes, classification scores, and object velocities following RayDN [36].

Occupancy Head. For occupancy prediction, we use an OPUS-V2-style [61] query decoder as the occupancy readout. Since occupancy requires scene-level spatial reasoning, this branch consumes multi-frame GeoUP features and performs temporal alignment inside the occupancy head. For each timestamp t, the selected multi-view and multi-level features are first processed by an occupancy neck to produce $\mathbf { Y } _ { t } ^ { o , \nu }$ . The OPUS-V2-style decoder then takes the multi-frame and multi-view image features jointly and aggregates them through query-based feature sampling to predict semantic occupancy at the current timestamp:

$$
\begin{array} { r } { \mathbf { O } _ { T } = \mathcal { H } _ { \mathrm { o c c } } \left( \{ \mathbf { Y } _ { t } ^ { o , \nu } \} _ { t = 1 , \dots , T ; \nu = 1 , \dots , V } \right) , } \end{array}\tag{6}
$$

where $\mathbf { O } _ { T } = \{ ( \mathbf { p } _ { T , i } , s _ { T , i } ) \} _ { i = 1 } ^ { N _ { \mathrm { o c c } } }$ denotes a set of sparse occupied points with semantic labels. Camera Head. Following VGGT [63], we keep the camera prediction branch during training. It reads the camera-token features $\{ \bar { \mathbf { C } } _ { t } ^ { \nu } \} _ { t = 1 , \ldots , T , \nu = 1 , \ldots , V }$ and predicts a 9D camera representation for each timestamp and view:

$$
\mathbf { P } _ { t } ^ { \nu } = \mathcal { H } _ { \mathrm { c a m } } \left( \{ \bar { \mathbf { C } } _ { t } ^ { \nu } \} _ { t = 1 , \dots , T , \nu = 1 , \dots , V } \right) .\tag{7}
$$

This branch predicts camera-to-reference poses following the parametrization of VGGT [63], provides a camera-level training constraint together with the perception objectives.

## 3.3 Training Strategy

We train GeoUP in two stages. In the first stage, we initialize the backbone from the pretrained VGGT [63] weights and finetune it on driving datasets with the depth and camera branches:

$$
{ \mathcal { L } } _ { \mathrm { f t } } = { \mathcal { L } } _ { \mathrm { d e p } } + { \mathcal { L } } _ { \mathrm { c a m } } .\tag{8}
$$

This stage adapts the reconstruction-oriented VGGT [63] representation to driving scenes while preserving its geometric modeling capability.

In the second stage, we introduce the detection and occupancy heads and train GeoUP with a unified multi-task objective:

$$
\mathcal { L } = m _ { \mathrm { d e p } } \lambda _ { \mathrm { d e p } } \mathcal { L } _ { \mathrm { d e p } } + m _ { \mathrm { c a m } } \lambda _ { \mathrm { c a m } } \mathcal { L } _ { \mathrm { c a m } } + m _ { \mathrm { d e t } } \mathcal { L } _ { \mathrm { d e t } } + m _ { \mathrm { o c c } } \mathcal { L } _ { \mathrm { o c c } } ,\tag{9}
$$

where $m _ { \mathrm { d e p } } , m _ { \mathrm { c a m } } , m _ { \mathrm { d e t } }$ , and $m _ { \mathrm { o c c } }$ indicate whether the corresponding annotation is available for the sampled data. This masking strategy enables joint training on multiple autonomous driving datasets with heterogeneous task annotations. Since the first-stage finetuning has already equipped GeoUP with strong depth and camera estimation ability, we set $\lambda _ { \mathrm { d e p } } =$ $\lambda _ { \mathrm { c a m } } = 0 . 1$ in the second stage to let the model focus more on the newly introduced detection and occupancy tasks while retaining geometry-oriented regularization.

<table><tr><td>Method</td><td>Backbone</td><td>Input Size</td><td>mAP↑</td><td>NDS↑</td><td>mATE↓</td><td>mASE↓</td><td>mAOE↓</td><td>mAVE↓</td><td>mAAE↓</td></tr><tr><td>Stream3DPPE []</td><td>VoV-99</td><td> $\overline { { 8 0 0 \times 3 2 0 } }$ </td><td>50.0</td><td>58.5</td><td>0.565</td><td>0.261</td><td>0.376</td><td>0.251</td><td>0.200</td></tr><tr><td>RoPETR []</td><td>VoV-99</td><td> $8 0 0 \times 3 2 0$ </td><td>52.9</td><td>61.4</td><td>0.537</td><td>0.255</td><td>0.289</td><td>0.229</td><td>0.195</td></tr><tr><td>StreamPETR []</td><td>EVA02</td><td> $8 0 0 \times 3 2 0$ </td><td>52.1</td><td>60.8</td><td>0.547</td><td>0.252</td><td>0.283</td><td>0.242</td><td>0.201</td></tr><tr><td>RayDN []</td><td>EVA02</td><td> $8 0 0 \times 3 2 0$ </td><td>54.1</td><td>62.4</td><td>0.518</td><td>0.252</td><td>0.274</td><td>0.230</td><td>0.195</td></tr><tr><td>GeoUP (Ours)</td><td>ViT-L</td><td> $8 0 0 \times 3 2 0$ </td><td>57.9</td><td>64.4</td><td>0.516</td><td>0.259</td><td>0.254</td><td>0.223</td><td>0.204</td></tr><tr><td>GeoUP† (Ours)</td><td>ViT-L</td><td> $8 0 0 \times 3 2 0$ </td><td>59.2</td><td>65.3</td><td>0.496</td><td>0.254</td><td>0.271</td><td>0.217</td><td>0.196</td></tr></table>

Table 1: Main results of 3D object detection on the nuScenes validation set [3]. GeoUP<sup>†</sup> denotes the multi-dataset jointly trained model.

<table><tr><td>Method</td><td>Backbone</td><td>Input Size</td><td>mAP↑</td><td>CDS↑</td><td>mATE↓</td><td>mASE↓</td><td>mAOE↓</td></tr><tr><td>StreamPETR [4]</td><td>VoV-99</td><td> $9 6 0 \times 6 4 0$ </td><td>20.3</td><td>14.6</td><td>0.843</td><td>0.321</td><td>0.650</td></tr><tr><td>RayDN []</td><td>VoV-99</td><td> $9 6 0 \times 6 4 0$ </td><td>22.3</td><td>16.1</td><td>0.825</td><td>0.325</td><td>0.629</td></tr><tr><td>Far3D []</td><td>VoV-99</td><td> $9 6 0 \times 6 4 0$ </td><td>24.4</td><td>18.1</td><td>0.796</td><td>0.304</td><td>0.538</td></tr><tr><td>Far3D []</td><td>ViT-L</td><td> $1 5 3 6 \times 1 5 3 6$ </td><td>31.6</td><td>23.9</td><td>0.732</td><td>0.303</td><td>0.459</td></tr><tr><td>GeoUP (Ours)</td><td>ViT-L</td><td> $9 6 0 \times 6 4 0$ </td><td>37.2</td><td>28.2</td><td>0.735</td><td>0.315</td><td>0.461</td></tr><tr><td>GeoUP† (Ours)</td><td>ViT-L</td><td> $9 6 0 \times 6 4 0$ </td><td>43.6</td><td>33.8</td><td>0.668</td><td>0.304</td><td>0.428</td></tr></table>

Table 2: Main results of 3D object detection on the Argoverse 2 validation set [72]. $\mathrm { G e o U P ^ { \dagger } }$ denotes the multi-dataset jointly trained model.

The depth loss $\mathcal { L } _ { \mathrm { d e p } }$ combines metric depth regression with gradient-based regularization. The camera loss ${ \mathcal { L } } _ { \mathrm { c a m } }$ supervises the range-normalized camera pose targets with L1 losses on translation, rotation, and intrinsic parameters, following VGGT [63]. The detection and occupancy branches adopt the standard objectives of RayDN [36] and OPUS-V2 [61] respectively.

## 4 Experiments

## 4.1 Datasets and Metrics

We evaluate 3D object detection on nuScenes [3], Argoverse 2 [72], and Waymo [58], following their official evaluation protocols. These benchmarks cover different camera configurations, perception ranges, and object category settings, allowing us to evaluate the generalization of GeoUP across diverse driving scenarios. We report mAP and NDS on nuScenes, mAP and CDS on Argoverse 2, and LET-3D-APL, LET-3D-AP, and LET-3D-APH on Waymo. Semantic occupancy prediction is evaluated on Occ3D-nuScenes [59], which provides dense 3D semantic annotations for surround-view scenes. We report mIoU and RayIoU [37], including RayIoU under the 1 m, 2 m, and 4 m thresholds. Depth estimation is evaluated on KITTI [15] and DDAD [17] using Absolute Relative Error (Abs Rel) and the threshold accuracy $\delta < 1 . 2 5$

## 4.2 Implementation Details

The original VGGT [63] backbone contains 24 attention blocks, which incurs substantial computational cost. For efficiency, we construct a compact 12-block variant, denoted as VGGT-12, by selecting every other layer from the pretrained VGGT [63] checkpoint. Unless otherwise specified, the length of the temporal window is 4 frames. For perception tasks, we follow the default configurations of the corresponding heads. The RayDN-style [36] detection head uses 900 object queries and retains queries from 4 frames. The OPUS-V2-style [61] occupancy head uses 8-frame image features for sparse semantic occupancy prediction. The depth head decodes metric depth from the shared geometry-grounded representation.

<table><tr><td>Method</td><td>Backbone</td><td>Input Size</td><td>mAPL↑</td><td>mAP↑</td><td>mAPH↑</td></tr><tr><td>PETRv2 [59]</td><td>R101 R101</td><td> $\overline { { 1 9 2 0 \times 1 2 8 0 } }$ </td><td>36.6</td><td>51.9</td><td>47.9</td></tr><tr><td>StreamPETR []</td><td></td><td> $1 9 2 0 \times 1 2 8 0$ </td><td>39.9</td><td>55.3</td><td>51.7</td></tr><tr><td>MV-FCOS3D++ []</td><td>R101-DCN</td><td> $1 9 2 0 \times 1 2 8 0$ </td><td>38.5</td><td>53.2</td><td>50.0</td></tr><tr><td>BEVFormer []</td><td>R101-DCN</td><td> $1 9 2 0 \times 1 2 8 0$ </td><td>38.2</td><td>55.1</td><td>51.1</td></tr><tr><td>DenseBEV++ []</td><td>R101-DCN</td><td> $1 9 2 0 \times 1 2 8 0$ </td><td>42.4</td><td>60.2</td><td>56.4</td></tr><tr><td>GeoUP (Ours)</td><td>ViT-L</td><td> $9 6 0 \times 6 4 0$ </td><td>51.5</td><td>67.0</td><td>63.5</td></tr><tr><td>GeoUP† (Ours)</td><td>ViT-L</td><td> $9 6 0 \times 6 4 0$ </td><td>54.3</td><td>70.7</td><td>67.7</td></tr></table>

Table 3: Main results of 3D object detection on the Waymo validation set [58]. ${ \bf G e o U P } ^ { \dagger }$ denotes the multi-dataset jointly trained model.

<table><tr><td>Method</td><td>Backbone</td><td>Input Size</td><td>mIoU↑</td><td>RayIoU↑</td><td> $\overline { { \mathrm { R a y I o U } _ { 1 m } } }$  ←</td><td> $\overline { { \mathrm { R a y I o U } _ { 2 m } } }$  ←</td><td> $\overline { { \mathrm { R a y I o U } _ { 4 m } \uparrow } }$ </td></tr><tr><td>FB-Occ [] (16f)</td><td>R50</td><td> $\overline { { 7 0 4 \times 2 5 6 } }$ </td><td>39.1</td><td>33.5</td><td>26.7</td><td>34.1</td><td>39.7</td></tr><tr><td>SparseOcc [] (16f)</td><td>R50</td><td> $7 0 4 \times 2 5 6$ </td><td>30.6</td><td>35.1</td><td>29.1</td><td>35.8</td><td>40.3</td></tr><tr><td>OPUS-L [] (8f)</td><td>R50</td><td> $7 0 4 \times 2 5 6$ </td><td>36.2</td><td>41.2</td><td>34.7</td><td>42.1</td><td>46.7</td></tr><tr><td>OPUS-V2-L [](8f)</td><td>R50</td><td> $7 0 4 \times 2 5 6$ </td><td>38.6</td><td>44.0</td><td>38.0</td><td>45.0</td><td>49.2</td></tr><tr><td>OPUS-V2-L [] (8f)</td><td>ViT-L</td><td> $7 0 4 \times 2 5 6$ </td><td>39.9</td><td>45.2</td><td>39.0</td><td>46.3</td><td>50.4</td></tr><tr><td>GeoUP (Ours) (8f)</td><td>ViT-L</td><td> $7 0 4 \times 2 5 6$ </td><td>41.5</td><td>45.9</td><td>39.3</td><td>47.1</td><td>51.3</td></tr><tr><td>GeoUP† (Ours) (8f)</td><td>ViT-L</td><td> $7 0 4 \times 2 5 6$ </td><td>42.3</td><td>47.0</td><td>40.7</td><td>48.1</td><td>52.2</td></tr></table>

Table 4: Semantic occupancy prediction results on Occ3D-nuScenes [59]. 8f and 16f denote models fusing temporal information from 8 or 16 frames, respectively. GeoUP<sup>†</sup> denotes the multi-dataset jointly trained model. OPUS-V2-L‡ denotes our reproduced variant trained with a ViT-L backbone using the official OPUS-V2 [61] codebase.

We evaluate GeoUP under both single-dataset and multi-dataset training settings. For single-dataset training, we follow the schedules of prior methods for fair comparison. GeoUP is trained for 24 epochs on nuScenes [3], 6 epochs on Argoverse 2 [72], and 24 epochs on Waymo [58]. We use the AdamW optimizer [44] with a batch size of 16. The initial learning rate is set to $4 \times 1 0 ^ { - 4 }$ with a cosine annealing schedule [43]. For multi-dataset joint training, the dataset sampling ratio is designed according to the scale of each dataset. Argoverse 2 [72] is downsampled by a factor of 4, consistent with its shorter single-dataset training schedule, while Waymo [58] is sampled every five frames. Accordingly, the sampling ratio for nuScenes [3], Argoverse 2 [72], Waymo [58], DDAD [17], and KITTI [15] is set to $8 : 8 : 9 : 3 : 1$ . Since joint training over heterogeneous datasets is more difficult to converge, we train the multi-dataset model with twice the training epochs of the single-dataset setting. The joint model is trained with a batch size of 64. We use the Muon optimizer [26] with an initial learning rate of $6 \times 1 0 ^ { - 4 }$ and the same cosine annealing schedule [43].

## 4.3 Main Results

3D Detection. Tables 1, 2, and 3 report the 3D object detection results on nuScenes, Argoverse 2, and Waymo, respectively. GeoUP achieves strong performance across all three benchmarks, demonstrating the effectiveness and generality of the proposed geometry-grounded representation for camera-based 3D detection. On nuScenes, GeoUP achieves 57.9% mAP and 64.4% NDS with a ViT-L backbone under the same input resolution as previous representative methods, outperforming RayDN by 3.8% mAP and 2.0% NDS. On Argoverse 2, GeoUP obtains 34.3% mAP and 25.6% CDS using an input size of $9 6 0 \times 6 4 0$ , surpassing Far3D with a ViT-L backbone even though Far3D uses a much higher resolution of

<table><tr><td rowspan="2">Method</td><td colspan="2">KITTI</td><td colspan="2">DDAD</td></tr><tr><td>Abs Rel↓</td><td> $\delta < 1 . 2 5 \uparrow$ </td><td>Abs Rel↓</td><td> $\delta < 1 . 2 5 \uparrow$ </td></tr><tr><td>StreamVGGT [[]</td><td>0.102</td><td>90.3</td><td>0.343</td><td>42.2</td></tr><tr><td>VGGT [3]</td><td>0.102</td><td>89.8</td><td>0.391</td><td>65.4</td></tr><tr><td>MapAnything []</td><td>0.210</td><td>52.2</td><td>0.185</td><td>77.1</td></tr><tr><td>DVGT []</td><td>0.117</td><td>87.3</td><td>0.197</td><td>70.7</td></tr><tr><td>GeoUP (Ours)</td><td>0.072</td><td>93.8</td><td>0.114</td><td>89.5</td></tr><tr><td>GeoUP† (Ours)</td><td>0.075</td><td>92.9</td><td>0.123</td><td>87.6</td></tr></table>

Table 5: Depth estimation results on KITTI [15] and DDAD [17]. GeoUP<sup>†</sup> denotes the multi-dataset jointly trained model.

<table><tr><td>Backbone</td><td>NC↑</td><td>DAC↑</td><td>DDC↑</td><td>TL↑</td><td>EP↑</td><td>TTC↑</td><td>LK↑</td><td>HC↑</td><td>EC↑</td><td>EPDMS↑</td></tr><tr><td>DA-ViT-L []</td><td>98.4</td><td>98.6</td><td>99.6</td><td>99.8</td><td>90.5</td><td>97.8</td><td>97.0</td><td>98.3</td><td>78.6</td><td>87.1/90.5</td></tr><tr><td>DINOv2-L [日</td><td>97.9</td><td>97.0</td><td>99.3</td><td>99.7</td><td>87.7</td><td>97.0</td><td>95.5</td><td>98.2</td><td>77.1</td><td>83.7/87.2</td></tr><tr><td>DINOv3-L []</td><td>98.1</td><td>97.8</td><td>99.5</td><td>99.7</td><td>89.5</td><td>97.4</td><td>96.2</td><td>98.3</td><td>77.3</td><td>85.4/89.0</td></tr><tr><td>GeoUP (Ours)</td><td>98.4</td><td>98.8</td><td>99.5</td><td>99.9</td><td>91.9</td><td>98.0</td><td>97.4</td><td>98.3</td><td>78.8</td><td>87.9/91.4</td></tr></table>

Table 6: Planning results on NAVSIMv2 [4, 9]. All variants use the same DriveSuprim [75] planning decoder and training configuration, differing only in the visual backbone. DA-ViT-L denotes the Depth Anything [74] backbone used in DriveSuprim.

1536 × 1536. On Waymo, GeoUP also achieves clearly better results than previous camerabased detectors, reaching 51.5% mAPL, 67.0% mAP, and 63.5% mAPH with a lower input resolution than prior methods. We further report a multi-dataset jointly trained model, denoted as $\mathrm { G e o U P ^ { \dagger } }$ This model is trained on nuScenes, Argoverse 2, Waymo, DDAD, and KITTI, where each dataset only supervises the heads with available annotations. Compared with the single-dataset model, GeoUP<sup>†</sup> further improves the results on all three benchmarks, achieving 59.2% mAP and 65.3% NDS on nuScenes, 43.6% mAP and 33.8% CDS on Argoverse 2, and 70.7% mAP and 67.7% mAPH on Waymo. These results indicate that heterogeneous supervision from datasets with diverse sensor configurations, scene distributions, and annotation types helps GeoUP learn a more robust geometry-grounded representation.

Occupancy Prediction. Table 4 reports the semantic occupancy prediction results on the Occ3D-nuScenes benchmark. GeoUP achieves 41.5 mIoU and 45.9 RayIoU under singledataset training, outperforming OPUS-V2-L with the same ViT-L scale and OPUS-V2-style decoder. This indicates that the improvement mainly comes from our geometry-grounded backbone. With multi-dataset joint training, our method further improves to 42.3 mIoU and 47.0 RayIoU, achieving the best results across all metrics.

Depth Prediction. Table 5 reports the depth estimation results on KITTI and DDAD. GeoUP<sup>†</sup> achieves strong depth prediction performance on both datasets. KITTI mainly consists of forward-facing stereo images with large view overlap, which is relatively favorable for visual geometry foundation models. Under this setting, VGGT and StreamVGGT already produce competitive depth estimates, while GeoUP<sup>†</sup> further improves over VGGT by reducing Abs Rel from 0.102 to 0.075 and increasing $\delta < 1 . 2 5$ from 89.8% to 92.9%. DDAD, in contrast, provides large-scale multi-camera driving scenes and better reflects the calibrated surround-view setting of autonomous driving. On this more challenging benchmark, generic visual geometry models such as VGGT and StreamVGGT show limited accuracy, while methods with explicit camera conditioning or driving-scene adaptation, such as MapAnything and DVGT, perform more favorably. ${ \bf G e o U P } ^ { \dagger }$ achieves the strongest results on DDAD, with 0.123 Abs Rel and 87.6% $\delta < 1 . 2 5$ , showing the benefit of calibration-aware geometry modeling and unified training for metric depth prediction in multi-camera driving scenes.

<table><tr><td rowspan="2">Global</td><td rowspan="2">View</td><td rowspan="2">Temporal</td><td rowspan="2">Plücker Ray</td><td colspan="2">Det.</td><td colspan="2">Occ.</td></tr><tr><td>mAP↑</td><td>NDS↑</td><td>mIoU↑</td><td>RayIoU↑</td></tr><tr><td>√</td><td></td><td></td><td></td><td>53.8</td><td>61.7</td><td>37.3</td><td>41.8</td></tr><tr><td></td><td>√</td><td></td><td></td><td>53.3</td><td>61.1</td><td>37.5</td><td>41.4</td></tr><tr><td></td><td></td><td>√</td><td></td><td>54.6</td><td>62.0</td><td>38.8</td><td>42.6</td></tr><tr><td></td><td>√</td><td>√</td><td></td><td>55.5</td><td>62.3</td><td>39.9</td><td>43.6</td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td>56.4</td><td>63.0</td><td>40.3</td><td>44.5</td></tr></table>

Table 7: Ablation study on the proposed backbone design on the nuScenes validation set [3]. Global denotes the original global cross-image attention in VGGT-12. View and Temporal denote the factorized cross-view and temporal attention designs, respectively. Plücker Ray denotes the camera-aware geometric prior injected into the backbone.

<table><tr><td rowspan="2">Head Setting</td><td rowspan="2">Frames</td><td colspan="2">Det.</td><td colspan="2">Occ.</td><td colspan="2">Depth.</td></tr><tr><td>mAP↑</td><td>NDS↑</td><td>mIoU↑</td><td>RayIoU↑</td><td>Abs Rel↓</td><td>δ &lt; 1.25↑</td></tr><tr><td rowspan="4">Current-only heads</td><td>1</td><td>48.7</td><td>52.9</td><td>36.7</td><td>40.8</td><td>0.102</td><td>90.1</td></tr><tr><td>2</td><td>51.0</td><td>57.9</td><td>37.9</td><td>41.7</td><td>0.098</td><td>90.6</td></tr><tr><td>4</td><td>52.8</td><td>59.2</td><td>38.9</td><td>42.2</td><td>0.098</td><td>90.7</td></tr><tr><td>8</td><td>53.6</td><td>60.1</td><td>38.7</td><td>43.3</td><td>0.098</td><td>90.7</td></tr><tr><td rowspan="4">Temporal heads</td><td>1</td><td>55.1</td><td>62.5</td><td>40.8</td><td>45.0</td><td>0.098</td><td>90.6</td></tr><tr><td>2</td><td>55.7</td><td>62.8</td><td>41.4</td><td>45.4</td><td>0.094</td><td>91.2</td></tr><tr><td>4</td><td>56.5</td><td>63.6</td><td>41.5</td><td>45.9</td><td>0.092</td><td>91.4</td></tr><tr><td>8</td><td>56.9</td><td>63.6</td><td>41.6</td><td>46.1</td><td>0.092</td><td>91.4</td></tr></table>

Table 8: Ablation on the number of input frames on the nuScenes validation set [3]. All models are trained for 50 epochs. Current-only heads denote that the heads use only currentframe information, while Temporal heads denote that the downstream heads also exploit temporal information.

Transfer to End-to-End Planning. To assess whether GeoUP transfers beyond perception, we integrate it into DriveSuprim [75] and evaluate it on the NAVSIMv2 [4, 9] benchmark. We keep the planning decoder and training configuration unchanged and replace only the visual backbone. To maintain comparability with the released DriveSuprim results, we report EPDMS using both its original evaluation implementation and the corrected NAVSIM evaluator, denoted as original/corrected.<sup>1</sup> As shown in Table 6, DINOv2-L and DINOv3- L achieve 83.7%/87.2% and 85.4%/89.0% EPDMS, respectively, whereas DA-ViT-L (the Depth Anything [74] pretrained ViT-L) used in DriveSuprim obtains 87.1%/90.5%. GeoUP reaches 87.9%/91.4%, outperforming DA-ViT-L by 0.8 and 0.9 points under the original and corrected evaluators, respectively. These results suggest that GeoUP provides a stronger visual representation for the fixed DriveSuprim planner, supporting its transferability beyond perception tasks.

## 4.4 Ablation Study

We conduct ablation studies mainly on the nuScenes validation set. Unless otherwise specified, all models are trained for 24 epochs with an input resolution of 672 × 224.

Effect of Driving-Oriented Backbone Adaptation. Table 7 validates the proposed adaptations of the VGGT backbone to multi-camera driving scenes. We first compare different cross-image attention strategies. In the global-attention setting, all input images are jointly attended, following the original VGGT-style interaction, where the front camera of the current frame is used as the reference frame. For temporal attention, attention is only performed among images from the same view across different timestamps. To ensure that each temporal window has its own reference, we use the current frame of each view as the corresponding reference frame. Similarly, for view attention, we perform cross-view interaction within each timestamp and use the front camera of each frame as the reference frame. Temporal attention achieves the best single-factorized performance, even surpassing global attention, thanks to the strong temporal continuity within each camera stream. View attention alone is less effective, as the limited overlap across cameras makes cross-view matching difficult without additional context. However, when combined with temporal attention, view attention provides complementary cross-view information and further improves the shared representation. Finally, adding Plücker ray embeddings injects explicit camera-aware geometric priors into the backbone, leading to the best overall performance across detection and occupancy metrics. These results demonstrate the effectiveness of adapting the VGGT backbone to the structure of streaming multi-view driving data.

<table><tr><td rowspan="2">Method</td><td colspan="2">Backbone Configuration</td><td colspan="2">Det.</td><td colspan="2">Occ.</td></tr><tr><td>VGGT Blocks VGGT Init. Drive Adapt.</td><td></td><td>mAP↑</td><td>NDS↑</td><td>mIoU↑</td><td>RayIoU↑</td></tr><tr><td>DINOv2-L [0]</td><td>0</td><td></td><td>51.9</td><td>59.7</td><td>34.2</td><td>38.8</td></tr><tr><td>DINOv2-T (ours)</td><td>12</td><td></td><td>52.1</td><td>60.3</td><td>37.1</td><td>41.8</td></tr><tr><td>VGGT-12 []</td><td>12</td><td>√</td><td>54.6</td><td>62.0</td><td>38.8</td><td>42.6</td></tr><tr><td>VGGT-24 []</td><td>24</td><td>√</td><td>55.5</td><td>62.6</td><td>39.6</td><td>43.4</td></tr><tr><td>GeoUP (Ours)</td><td>12</td><td>√</td><td>56.4</td><td>63.0</td><td>40.3</td><td>44.5</td></tr></table>

Table 9: Effect of backbone configurations on the nuScenes validation set [3]. VGGT Blocks reports the number of VGGT-style Transformers blocks. VGGT Init. denotes initialization from geometry-pretrained VGGT weights, while Drive Adapt. identifies the proposed driving-oriented adaptations.

<table><tr><td rowspan="2">Occ. Sup. Det. Sup.</td><td rowspan="2"></td><td colspan="2">Det.</td><td colspan="2">Occ.</td></tr><tr><td>mAP↑</td><td>NDS↑</td><td>mIoU↑</td><td>RayIoU↑</td></tr><tr><td></td><td>√</td><td>55.9</td><td>62.8</td><td></td><td></td></tr><tr><td>√</td><td></td><td></td><td></td><td>39.8</td><td>43.8</td></tr><tr><td>√</td><td>√</td><td>56.4</td><td>63.0</td><td>40.3</td><td>44.5</td></tr></table>

Table 10: Ablation study on multi-task learning on the nuScenes validation set [3].

Number of Frames. Table 8 examines the effect of temporal context under two head configurations: current-only heads that decode current-frame features, and temporal heads with task-specific temporal modeling. All models are trained for 50 epochs. Increasing the number of input frames generally improves detection, occupancy, and depth, with most gains obtained by four frames and performance largely saturating thereafter. Temporal heads consistently outperform current-only heads, indicating that backbone-level multi-frame aggregation complements temporal modeling in downstream heads. Considering the trade-off between efficiency and performance, we use 4 input frames as the default setting in our experiments.

Effect of Backbone Configuration. Table 9 compares different backbone configurations on the nuScenes validation set. DINOv2-L serves as the image-encoder-only reference, while DINOv2-T augments it with 12 randomly initialized VGGT-style Transformers blocks. With the architecture and input configuration held fixed, initializing these blocks from pretrained VGGT weights (VGGT-12) improves mAP/NDS by 2.5/1.7 points and mIoU/RayIoU by 1.7/0.8 points. Increasing the number of VGGT-style blocks from 12 to 24 yields further improvements. In comparison, GeoUP introduces the proposed driving-oriented adaptations on top of VGGT-12 and achieves the best results across all detection and occupancy metrics, even outperforming the deeper VGGT-24 backbone.

![](images/e0ad0f574d2ca6e5a5cfb5a0b4e7f8464a1208437b7707fa544381cec5257a4b.jpg)

Figure 4: Qualitative comparison of 3D detection results over consecutive frames. The red dashed regions indicate the corresponding vehicle areas in BEV, and the GT BEV region is linked to the associated vehicles in the front-camera image. Compared with RayDN [36], GeoUP maintains more stable predictions for the static vehicle and continuously detects the moving vehicle across all four frames.  
![](images/00e83eef9689876b718779e80352027124fe18b3e92d9583cf27bff1a4881dcb.jpg)  
Figure 5: Qualitative comparison of semantic occupancy prediction. GeoUP better preserves the road layout and surrounding scene structures compared with OPUS-V2 [61].

Effect of Multi-task Learning. Table 10 studies the effect ofjoint optimization between occupancy prediction and 3D detection on the nuScenes validation set. Compared with training each task separately, the multi-task setting improves both detection and occupancy performance. These results suggest that multi-task learning encourages the backbone to learn a more complete and transferable 3D representation for autonomous driving perception.

## 4.5 Qualitative Analysis

We provide qualitative results across different perception tasks. For 3D detection, Figure 4 compares GeoUP with RayDN [36]. The red dashed regions mark the corresponding vehicle areas in BEV and link the GT BEV region to the front-camera image. For the highlighted static white vehicle, GeoUP maintains more stable box locations across frames. For the highlighted moving black vehicle, GeoUP detects it in all four frames, while RayDN misses it in two intermediate frames. These results show that the shared geometry-grounded representation can better aggregate historical observations and improve spatiotemporal consistency in 3D detection. For semantic occupancy prediction, Figure 5 compares GeoUP with OPUS-V2 [61]. GeoUP better preserves the continuous road layout and surrounding scene structures, producing more complete and coherent occupancy predictions. This suggests that the proposed geometry-grounded backbone provides stronger scene-level geometric and semantic understanding for occupancy prediction. For depth estimation, Figure 6 shows point maps reconstructed from the predicted depth. The clear road surfaces, vehicles, and surrounding structures indicate that GeoUP preserves the 3D reconstruction capability inherited from VGGT [63]. Overall, these qualitative results demonstrate the effectiveness of GeoUP in producing accurate, coherent, and geometrically consistent perception outputs across detection, occupancy, and depth reconstruction. More comprehensive visualizations across all evaluated autonomous driving datasets and perception tasks are provided in the appendix.

![](images/c26f45314edb1bc219a7d1aab34208e1edf5e8633d562c27b0d4826df951a46a.jpg)  
Figure 6: Point maps reconstructed from the predicted depth. GeoUP preserves the 3D reconstruction capability inherited from VGGT [63].

## 5 Conclusion and limitations

In this paper, we presented GeoUP, a geometry-grounded unified perception framework for autonomous driving. By adapting visual geometry foundation models to calibrated multicamera driving scenes, GeoUP learns a shared 3D scene latent that supports metric depth estimation, 3D object detection, and semantic occupancy prediction. With temporal-view attention factorization, calibration-aware raymap injection, and joint multi-task and multidataset training, GeoUP effectively transfers reconstruction-oriented representations to driving perception. Extensive experiments demonstrate state-of-the-art performance across multiple perception tasks, validating geometry-grounded representation learning for unified autonomous driving perception.

Despite its strong performance, GeoUP has two main limitations. First, the visual geometry backbone is computationally heavy, leading to a large model size and limited inference speed. Efficient 3D reconstruction models, lightweight transformers, and model compression [45, 53, 55, 77, 81] are promising directions for improvement. A detailed latency analysis is provided in the appendix. Second, the perception heads remain task-specific despite sharing the same geometry-grounded latent. Future work may explore a unified decoder for dense geometry, object instances, and semantic occupancy.

## Acknowledgment

This research is supported in part by the National Natural Science Foundation of China (No. 62461160308, U23B2010, 62576024), the Beijing Natural Science Foun dation (No. L231011), the Fundamental Research Funds for the Central Universities (No. 501RCQD2025141 BeiHang GanWei Project (No. 502GWXM2024141001), the National Science Foundation Support Projects (No. 62425303)

## References

[1] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254, 2021.

[2] Yohann Cabon, Lucas Stoffl, Leonid Antsfeld, Gabriela Csurka, Boris Chidlovskii, Jerome Revaud, and Vincent Leroy. Must3r: Multi-view network for stereo 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1050–1060, 2025.

[3] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020.

[4] Wei Cao, Marcel Hallgarten, Tianyu Li, Daniel Dauner, Xunjiang Gu, Caojun Wang, Yakov Miron, Marco Aiello, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas Geiger, and Kashyap Chitta. Pseudo-simulation for autonomous driving. In Conference on Robot Learning (CoRL), 2025.

[5] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European conference on computer vision, pages 213–229, 2020.

[6] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.

[7] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607, 2020.

[8] Marius Dähling, Sebastian Krebs, and J Marius Zöllner. Densebev: Transforming bev grid cells into 3d objects. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 2370–2379, 2026.

[9] Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, et al. Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems, 37:28706–28719, 2024.

[10] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

[11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[12] Kaiwen Duan, Song Bai, Lingxi Xie, Honggang Qi, Qingming Huang, and Qi Tian. Centernet: Keypoint triplets for object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6569–6578, 2019.

[13] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19358–19369, 2023.

[14] Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva-02: A visual representation for neon genesis. Image and Vision Computing, 149: 105171, 2024.

[15] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The international journal of robotics research, 32(11): 1231–1237, 2013.

[16] Clément Godard, Oisin Mac Aodha, Michael Firman, and Gabriel J Brostow. Digging into self-supervised monocular depth estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3828–3838, 2019.

[17] Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos, and Adrien Gaidon. 3d packing for self-supervised monocular depth estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2485–2494, 2020.

[18] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 770–778, 2016.

[19] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020.

[20] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022.

[21] Junjie Huang, Guan Huang, Zheng Zhu, Yun Ye, and Dalong Du. Bevdet: Highperformance multi-camera 3d object detection in bird-eye-view. arXiv preprint arXiv:2112.11790, 2021.

[22] Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou, and Jiwen Lu. Triperspective view for vision-based 3d semantic occupancy prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9223– 9232, 2023.

[23] Hang Ji, Tao Ni, Xufeng Huang, Zhan Shi, Tao Luo, Xin Zhan, and Junbo Chen. Ropetr: Improving temporal camera-only 3d detection by integrating enhanced rotary position embedding. arXiv preprint arXiv:2504.12643, 2025.

[24] Xiaosong Jia, Yanhao Liu, Junqi You, Renqiu Xia, Yu Hong, and Junchi Yan. Drivevggt: Visual geometry transformer for autonomous driving. arXiv preprint arXiv:2511.22264, 2025.

[25] Xiaohui Jiang, Shuailin Li, Yingfei Liu, Shihao Wang, Fan Jia, Tiancai Wang, Lijin Han, and Xiangyu Zhang. Far3d: Expanding the horizon for surround-view 3d object detection. In Proceedings ofthe AAAI conference on artificial intelligence, volume 38, pages 2561–2569, 2024.

[26] Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https://kellerjordan. github. io/posts/muon, 6(3):4, 2024.

[27] Nikhil Keetha, Norman Müller, Johannes Schönberger, Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, et al. Mapanything: Universal feed-forward metric 3d reconstruction. arXiv preprint arXiv:2509.13414, 2025.

[28] Youngwan Lee, Joong-won Hwang, Sangrok Lee, Yuseok Bae, and Jongyoul Park. An energy and gpu-computation efficient backbone network for real-time object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 0–0, 2019.

[29] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. Grounding image matching in 3d with mast3r. In European Conference on Computer Vision, pages 71–91, 2024.

[30] Han Li, Zehao Huang, Zitian Wang, Wenge Rong, Naiyan Wang, and Si Liu. Enhancing 3d lane detection and topology reasoning with 2d lane priors. arXiv preprint arXiv:2406.03105, 2024.

[31] Xiang Li, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang, and Jian Yang. Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection. Advances in neural information processing systems, 33: 21002–21012, 2020.

[32] Yinhao Li, Zheng Ge, Guanyi Yu, Jinrong Yang, Zengran Wang, Yukang Shi, Jianjian Sun, and Zeming Li. Bevdepth: Acquisition of reliable depth for multi-view 3d object detection. In Proceedings ofthe AAAI conference on artificial intelligence, volume 37, pages 1477–1485, 2023.

[33] Zhiqi Li, Zhiding Yu, David Austin, Mingsheng Fang, Shiyi Lan, Jan Kautz, and Jose M Alvarez. Fb-occ: 3d occupancy prediction based on forward-backward view transformation. arXiv preprint arXiv:2307.01492, 2023.

[34] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(3):2020–2036, 2024.

[35] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017.

[36] Feng Liu, Tengteng Huang, Qianjing Zhang, Haotian Yao, Chi Zhang, Fang Wan, Qixiang Ye, and Yanzhao Zhou. Ray denoising: Depth-aware hard negative sampling for multi-view 3d object detection. In European Conference on Computer Vision, pages 200–217. Springer, 2024.

[37] Haisong Liu, Yang Chen, Haiguang Wang, Zetong Yang, Tianyu Li, Jia Zeng, Li Chen, Hongyang Li, and Limin Wang. Fully sparse 3d occupancy prediction. In European Conference on Computer Vision, pages 54–71. Springer, 2024.

[38] Yingfei Liu, Tiancai Wang, Xiangyu Zhang, and Jian Sun. Petr: Position embedding transformation for multi-view 3d object detection. In European Conference on Computer Vision, pages 531–548, 2022.

[39] Yingfei Liu, Junjie Yan, Fan Jia, Shuailin Li, Aqi Gao, Tiancai Wang, and Xiangyu Zhang. Petrv2: A unified framework for 3d perception from multi-camera images. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 3262–3272, 2023.

[40] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021.

[41] Zhijian Liu, Haotian Tang, Alexander Amini, Xinyu Yang, Huizi Mao, Daniela L Rus, and Song Han. Bevfusion: Multi-task multi-sensor fusion with unified bird’s-eye view representation. In 2023 IEEE international conference on robotics and automation (ICRA), pages 2774–2781. IEEE, 2023.

[42] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 11976–11986, 2022.

[43] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016.

[44] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

[45] Si-Yu Lu, Po-Ting Chen, Hui-Che Hsu, Sin-Ye Jhong, Wen-Huang Cheng, and Yung-Yao Chen. Ovggt: O(1) constant-cost streaming visual geometry transformer. arXiv preprint arXiv:2603.05959, 2026.

[46] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

[47] Dennis Park, Rares Ambrus, Vitor Guizilini, Jie Li, and Adrien Gaidon. Is pseudolidar needed for monocular 3d object detection? In Proceedings of the IEEE/CVF international conference on computer vision, pages 3142–3152, 2021.

[48] Jonah Philion and Sanja Fidler. Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d. In European conference on computer vision, 2020.

[49] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

[50] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179–12188, 2021.

[51] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4938– 4947, 2020.

[52] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104–4113, 2016.

[53] You Shen, Zhipeng Zhang, Yansong Qu, Xiawu Zheng, Jiayi Ji, Shengchuan Zhang, and Liujuan Cao. Fastvggt: Training-free acceleration of visual geometry transformer. arXiv preprint arXiv:2509.02560, 2025.

[54] Changyong Shu, Jiajun Deng, Fisher Yu, and Yifan Liu. 3dppe: 3d point positiona encoding for transformer-based multi-camera 3d object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3580–3589, 2023.

[55] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

[56] Vincent Sitzmann, Semon Rezchikov, Bill Freeman, Josh Tenenbaum, and Fredo Durand. Light field networks: Neural scene representations with single-evaluation rendering. Advances in Neural Information Processing Systems, 34:19313–19325, 2021.

[57] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. Loftr: Detector-free local feature matching with transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8922–8931, 2021.

[58] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2446–2454, 2020.

[59] Xiaoyu Tian, Tao Jiang, Longfei Yun, Yucheng Mao, Huitong Yang, Yue Wang, Yilun Wang, and Hang Zhao. Occ3d: A large-scale 3d occupancy prediction benchmark for autonomous driving. Advances in Neural Information Processing Systems, 36:64318– 64330, 2023.

[60] Fangjinhua Wang, Silvano Galliani, Christoph Vogel, Pablo Speciale, and Marc Pollefeys. Patchmatchnet: Learned multi-view patchmatch stereo. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14194– 14203, 2021.

[61] Jiabao Wang, Zhaojiang Liu, Qiang Meng, Liujiang Yan, Ke Wang, Jie Yang, Wei Liu, Qibin Hou, and Ming-Ming Cheng. Opus: occupancy prediction using a sparse set. Advances in Neural Information Processing Systems, 37:119861–119885, 2024.

[62] Jianyuan Wang, Nikita Karaev, Christian Rupprecht, and David Novotny. Vggsfm: Visual geometry grounded deep structure from motion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21686–21697, 2024.

[63] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5294– 5306, 2025.

[64] Shihao Wang, Yingfei Liu, Tiancai Wang, Ying Li, and Xiangyu Zhang. Exploring object-centric temporal modeling for efficient multi-view 3d object detection. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 3621– 3631, 2023.

[65] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697–20709, 2024.

[66] Tai Wang, Xinge Zhu, Jiangmiao Pang, and Dahua Lin. Fcos3d: Fully convolutional one-stage monocular 3d object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 913–922, 2021.

[67] Tai Wang, Qing Lian, Chenming Zhu, Xinge Zhu, and Wenwei Zhang. Mv-fcos3d++: Multi-view camera-only 4d object detection with pretrained monocular backbones. arXiv preprint arXiv:2207.12716, 2022.

[68] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π<sup>3</sup>: Permutation-equivariant visual geometry learning. arXiv preprint arXiv:2507.13347, 2025.

[69] Yue Wang, Vitor Campagnolo Guizilini, Tianyuan Zhang, Yilun Wang, Hang Zhao, and Justin Solomon. Detr3d: 3d object detection from multi-view images via 3d-to-2d queries. In Conference on robot learning, pages 180–191, 2022.

[70] Zitian Wang, Zehao Huang, Jiahui Fu, Naiyan Wang, and Si Liu. Object as query: Lifting any 2d object detector to 3d detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3791–3800, 2023.

[71] Yi Wei, Linqing Zhao, Wenzhao Zheng, Zheng Zhu, Jie Zhou, and Jiwen Lu. Surroundocc: Multi-camera 3d occupancy prediction for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21729– 21740, 2023.

[72] Benjamin Wilson, William Qi, Tanmay Agarwal, John Lambert, Jagjeet Singh, Siddhesh Khandelwal, Bowen Pan, Ratnesh Kumar, Andrew Hartnett, Jhony Kaesemodel Pontes, et al. Argoverse 2: Next generation datasets for self-driving perception and forecasting. arXiv preprint arXiv:2301.00493, 2023.

[73] Chenyu Yang, Yuntao Chen, Hao Tian, Chenxin Tao, Xizhou Zhu, Zhaoxiang Zhang, Gao Huang, Hongyang Li, Yu Qiao, Lewei Lu, et al. Bevformer v2: Adapting modern image backbones to bird’s-eye-view recognition via perspective supervision. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 17830–17839, 2023.

[74] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10371–10381. IEEE, 2024.

[75] Wenhao Yao, Zhenxin Li, Shiyi Lan, Zi Wang, Xinglong Sun, Jose M Alvarez, and Zuxuan Wu. Drivesuprim: Towards precise trajectory selection for end-to-end planning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 11910–11918, 2026.

[76] Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan. Mvsnet: Depth inference for unstructured multi-view stereo. In Proceedings of the European conference on computer vision (ECCV), pages 767–783, 2018.

[77] Shuai Yuan, Yantai Yang, Xiaotian Yang, Xupeng Zhang, Zhonghao Zhao, Lingming Zhang, and Zhipeng Zhang. Infinitevggt: Visual geometry grounded transformer for endless streams. arXiv preprint arXiv:2601.02281, 2026.

[78] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

[79] Yunpeng Zhang, Zheng Zhu, and Dalong Du. Occformer: Dual-path transformer for vision-based 3d semantic occupancy prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9433–9443, 2023.

[80] Chuanxia Zheng and Andrea Vedaldi. Free3d: Consistent novel view synthesis without 3d representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9720–9731, 2024.

[81] Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, and Jiwen Lu. Streaming 4d visual geometry transformer. arXiv preprint arXiv:2507.11539, 2025.

[82] Sicheng Zuo, Zixun Xie, Wenzhao Zheng, Shaoqing Xu, Fang Li, Shengyin Jiang, Long Chen, Zhi-Xin Yang, and Jiwen Lu. Dvgt: Driving visual geometry transformer. arXiv preprint arXiv:2512.16919, 2025.

## Appendix

## A Additional Dataset and Implementation Details

Datasets and evaluation protocols. We provide additional details of the datasets and evaluation protocols used in the main paper.

nuScenes and Occ3D-nuScenes. nuScenes [3] contains 1,000 driving scenes captured by six cameras with 360<sup>◦</sup> surround-view coverage and provides 3D annotations for 10 object categories. For 3D detection, we follow the official protocol and report mean Average Precision (mAP), nuScenes Detection Score (NDS), and five true-positive metrics: Average Translation Error (mATE), Average Scale Error (mASE), Average Orientation Error (mAOE), Average Velocity Error (mAVE), and Average Attribute Error (mAAE). For occupancy prediction, we evaluate on Occ3D-nuScenes [59], which provides annotations for 18 semantic occupancy classes. We report mean Intersection-over-Union (mIoU) and Ray-level mIoU (RayIoU) [37], including RayIoU under the 1 m, 2 m, and 4 m thresholds.

Waymo Open Dataset. Waymo [58] provides large-scale driving sequences captured by five cameras covering the front and side views. Following common camera-only 3D detection practice, we use 20% of the training split and report the official Longitudinal Error Tolerant 3D Average Precision (LET-3D-AP), its heading-accuracy-weighted variant (LET-3D-APH), and its longitudinal-affinity-weighted variant (LET-3D-APL).

Argoverse 2. Argoverse 2 [72] provides 360<sup>◦</sup> multi-camera driving data captured by seven ring cameras, together with additional stereo cameras, and contains 3D annotations for 26 object categories. We follow the official 3D detection protocol and report mAP, Composite Detection Score (CDS), Average Translation Error (mATE), Average Scale Error (mASE), and Average Orientation Error (mAOE).

KITTI. KITTI [15] provides forward-view driving data captured by a front-facing stereo camera setup. We use it to evaluate monocular depth transferability to forward-view scenes and report Absolute Relative Error (Abs Rel) and threshold accuracy δ < 1.25.

DDAD. DDAD [17] provides high-resolution driving data captured by six synchronized cameras with 360<sup>◦</sup> surround-view coverage. We use it to evaluate dense metric depth prediction and cross-dataset geometric generalization, using the same Abs Rel and δ < 1.25 metrics as KITTI.

Multi-dataset configuration. For multi-dataset training, GeoUP is trained on nuScenes [3], Argoverse 2 [72], Waymo [58], DDAD [17], and KITTI [15]. These datasets provide heterogeneous annotations for different tasks. For depth estimation and camera pose prediction, all datasets are used for supervision, and the depth targets are normalized by a unified depth scale to maintain consistent metric supervision across datasets, where we set the depth scale to 90 m. For 3D detection, we mainly use nuScenes [3], Waymo [58], and Argoverse 2 [72], where all 3D boxes are represented under a nuScenes-style [3] LiDAR coordinate convention. Since the perception scale differs across datasets, we use dataset-specific point cloud ranges, with maximum horizontal ranges of 51.2 m, 74.88 m, and 152.4 m for nuScenes [3], Waymo [58], and Argoverse 2 [72], respectively. The detection box regression branch is shared across datasets, while the classification heads are separated according to dataset-specific category spaces. For occupancy prediction, supervision is only available on nuScenes [3], so the occupancy branch follows the nuScenes occupancy configuration and is optimized only on nuScenes samples.

Loss functions. GeoUP is optimized with task-specific objectives for depth estimation, 3D object detection, occupancy prediction, and camera pose prediction. For depth estimation, the DPT-style [50] head predicts a dense depth map and is trained with a depth objective that combines direct depth regression and gradient-based regularization. The direct regression term supervises metric depth values with an L2 objective, and the gradient term penalizes image-space depth-gradient errors to improve local geometric consistency. The depth loss is written as:

$$
\mathcal { L } _ { \mathrm { d e p } } = 1 . 0 \cdot \mathcal { L } _ { \mathrm { r e g } } + 1 . 0 \cdot \mathcal { L } _ { \mathrm { g r a d } } ,\tag{10}
$$

where $\mathcal { L } _ { \mathrm { r e g } }$ and $\mathcal { L } _ { \mathrm { g r a d } }$ denote the direct depth regression loss and the gradient regularization loss, respectively.

For 3D object detection, the RayDN-style [36] detection head follows DETR-style [5] Hungarian assignment and denoising training. The matching cost uses focal classification cost [35] and L1 box regression cost. The detection objective contains focal classification loss [35], L1 box regression loss, and denoising loss:

$$
\mathcal { L } _ { \mathrm { d e t } } = 2 . 0 \cdot \mathcal { L } _ { \mathrm { c l s } } ^ { \mathrm { 3 D } } + 0 . 2 5 \cdot \mathcal { L } _ { \mathrm { b o x } } ^ { \mathrm { 3 D } } + 1 . 0 \cdot \mathcal { L } _ { \mathrm { d n } } .\tag{11}
$$

The focal loss [35] uses $\gamma = 2 . 0$ and $\alpha = 0 . 2 5$

We also keep an auxiliary 2D head during training to provide image-space localization supervision. This branch uses Quality Focal Loss [31] for 2D classification, Gaussian focal loss [12] for centerness prediction, L1 loss for 2D box regression, GIoU loss for 2D box overlap, and L1 loss for projected center regression:

$$
\mathcal { L } _ { \mathrm { 2 D } } = 2 . 0 \cdot \mathcal { L } _ { \mathrm { q f l } } + 1 . 0 \cdot \mathcal { L } _ { \mathrm { c e n t e r } } + 5 . 0 \cdot \mathcal { L } _ { \mathrm { b o x } } ^ { \mathrm { 2 D } } + 2 . 0 \cdot \mathcal { L } _ { \mathrm { g i o u } } ^ { \mathrm { 2 D } } + 1 0 . 0 \cdot \mathcal { L } _ { \mathrm { c e n t e r 2 d } } .\tag{12}
$$

For occupancy prediction, the OPUS-V2-style [61] head predicts sparse occupied points and their semantic labels. The semantic branch is supervised by focal classification loss, and the point branch is supervised by Smooth-L1 regression loss with $\beta = 0 . 2$ . The occupancy objective is:

$$
\mathcal { L } _ { \mathrm { o c c } } = 2 . 0 \mathcal { L } _ { \mathrm { c l s } } ^ { \mathrm { o c c } } + 0 . 5 \mathcal { L } _ { \mathrm { p t s } } ^ { \mathrm { o c c } } .\tag{13}
$$

The camera head follows the VGGT-style [63] camera representation and is supervised with an L1 camera loss:

$$
\mathcal { L } _ { \mathrm { c a m } } = 5 . 0 \cdot \mathcal { L } _ { 1 } ^ { \mathrm { c a m } } .\tag{14}
$$

For multi-task and multi-dataset training, we apply supervision masks according to annotation availability. The final training objective is:

$$
\mathcal { L } = m _ { \mathrm { d e p } } \lambda _ { \mathrm { d e p } } \mathcal { L } _ { \mathrm { d e p } } + m _ { \mathrm { d e t } } \mathcal { L } _ { \mathrm { d e t } } + m _ { \mathrm { 2 D } } \mathcal { L } _ { \mathrm { 2 D } } + m _ { \mathrm { o c c } } \mathcal { L } _ { \mathrm { o c c } } + m _ { \mathrm { c a m } } \lambda _ { \mathrm { c a m } } \mathcal { L } _ { \mathrm { c a m } } ,\tag{15}
$$

where $m _ { \mathrm { d e p } } , m _ { \mathrm { d e t } } , m _ { \mathrm { 2 D } } , m _ { \mathrm { o c c } }$ , and $m _ { \mathrm { c a m } }$ indicate whether the corresponding supervision is available for the sampled data. Following the main training setting, the geometry-oriented depth and camera objectives are down-weighted in the unified stage with $\lambda _ { \mathrm { d e p } } = \lambda _ { \mathrm { c a m } } = 0 . 1$

## B Efficiency Analysis

We further analyze the inference efficiency of GeoUP. All runtime results are measured on a single NVIDIA H20 GPU with batch size 1 after 20 warm-up iterations and 100 inference iterations. For this profiling experiment, we use an input resolution of $6 7 2 \times 2 2 4$ . The reported FPS is computed from the average end-to-end runtime. Since GeoUP jointly predicts depth, 3D detection, occupancy, and camera pose, its runtime includes the shared geometry backbone and all task readout heads. For reference, we also compare with task-specific RayDN [36] and OPUS-V2 [61] models. As shown in Table 11, GeoUP is slower than task-specific perception models because it uses a VGGT-style [63] geometry backbone and produces multiple perception outputs in a single forward pass. When the temporal window increases from 1 frame to 4 frames, the FPS decreases from 2.18 to 0.81. This indicates that the main efficiency bottleneck comes from multi-frame geometry modeling in the shared backbone, rather than from the downstream task heads.

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Backbone</td><td rowspan=1 colspan=1>Image Size</td><td rowspan=1 colspan=1>FPS ↑</td></tr><tr><td rowspan=1 colspan=1>RayDN [B0]OPUS-V2 []</td><td rowspan=1 colspan=1>EVA02ViT-L</td><td rowspan=1 colspan=1> $6 7 2 \times 2 2 4$ 672× 224</td><td rowspan=1 colspan=1>9.945.90</td></tr><tr><td rowspan=2 colspan=1>GeoUP (Ours, 1f)GeoUP (Ours, 4f)</td><td rowspan=1 colspan=1>ViT-L</td><td rowspan=1 colspan=1>672×224</td><td rowspan=1 colspan=1>2.18</td></tr><tr><td rowspan=1 colspan=1>ViT-L</td><td rowspan=1 colspan=1>672× 224</td><td rowspan=1 colspan=1>0.81</td></tr></table>

Table 11: FPS comparison with task-specific baselines at ${ \overline { { 6 7 2 \times 2 } } } 2 4$ resolution.

<table><tr><td rowspan="2">Module</td><td colspan="2">1 frame</td><td colspan="2">4 frames</td></tr><tr><td>Latency (ms)</td><td>Ratio (%)</td><td>Latency (ms)</td><td>Ratio (%)</td></tr><tr><td>Geometry-grounded backbone</td><td>292.7</td><td>66.5</td><td>1062.1</td><td>87.8</td></tr><tr><td>Detection head</td><td>28.5</td><td>6.5</td><td>28.7</td><td>2.4</td></tr><tr><td>Depth head</td><td>32.5</td><td>7.4</td><td>32.5</td><td>2.7</td></tr><tr><td>Occupancy head</td><td>81.5</td><td>18.5</td><td>81.6</td><td>6.7</td></tr><tr><td>Camera head</td><td>5.1</td><td>1.2</td><td>5.1</td><td>0.4</td></tr><tr><td>Full model</td><td>440.3</td><td>100.0</td><td>1210.0</td><td>100.0</td></tr></table>

Table 12: Module-wise runtime breakdown of GeoUP. Ratio denotes the percentage of each module in the total module latency.

To better understand the computational cost, we provide a module-wise runtime breakdown in Table 12. For clarity, each task head includes its corresponding lightweight neck and prediction head. The auxiliary 2D branch has negligible measured latency in this profiling and is included in the detection head. The runtime breakdown shows that the shared geometry backbone dominates the total latency. In the single-frame setting, the backbone takes 292.7 ms, accounting for 66.5% of the total module latency. In the 4-frame setting, the backbone latency increases to 1062.1 ms and accounts for 87.8% of the computation. In contrast, the task heads introduce relatively small and stable overheads. For example, the detection head remains around 28.6 ms, the depth head remains around 32.5 ms, and the occupancy head remains around 81.6 ms across the two settings.

These results suggest that the most direct direction for improving GeoUP efficiency is to optimize the geometry-grounded backbone, adopt a lighter base model, and reduce the computational burden introduced by multi-frame processing. Another promising direction is to design a more unified decoder that shares computation across task readouts and processes multiple heads in parallel, instead of relying on separate task-specific decoders.

## C Additional Quantitative Results

Additional low-resolution multi-dataset results. We further investigate the impact of multi-dataset training under low-resolution settings. Specifically, we jointly train GeoUP on nuScenes [3], Argoverse 2 [72], Waymo [58], DDAD [17], and KITTI [15], where each dataset only supervises the heads with available annotations. For the input resolution, we use 672×224 for nuScenes [3], 448×336 for Argoverse 2 [72], and 504×336 for Waymo [58]. As shown in Table 13, multi-dataset training consistently improves the results. Compared with single-dataset training, the jointly trained model achieves better detection and occupancy performance across different benchmarks. This demonstrates that heterogeneous supervision from datasets with different sensor layouts, scene distributions, perception ranges, and annotation types helps GeoUP learn a more robust geometry-grounded representation. These results further support the effectiveness of multi-dataset training for improving the generalization ability of unified 3D perception models.

<table><tr><td colspan="3">(a) Argoverse 2</td><td colspan="4">(b) Waymo</td></tr><tr><td>Setting</td><td>mAP↑</td><td>CDS↑</td><td>Setting</td><td>mAPL↑</td><td>mAP↑</td><td>mAPH↑</td></tr><tr><td>Single</td><td>23.3</td><td>17.6</td><td>Single</td><td>43.4</td><td>58.7</td><td>55.4</td></tr><tr><td>Multi.</td><td>28.4</td><td>22.0</td><td>Multi.</td><td>44.6</td><td>60.1</td><td>57.0</td></tr></table>

(c) nuScenes
<table><tr><td>Setting</td><td>mAP↑</td><td>NDS↑</td><td>mIoU↑</td><td>RayIoU↑</td></tr><tr><td>Single</td><td>56.5</td><td>63.6</td><td>41.6</td><td>46.1</td></tr><tr><td>Multi.</td><td>57.6</td><td>64.1</td><td>42.3</td><td>47.0</td></tr></table>

Table 13: Additional low-resolution results of multi-dataset training. The joint model is trained on nuScenes [3], Argoverse 2 [72], Waymo [58], DDAD [17], and KITTI [15], where each dataset only supervises the heads with available annotations.

<table><tr><td>Setting</td><td>mAP↑</td><td>NDS↑</td><td>mIoU↑</td><td>RayIoU↑</td></tr><tr><td>w/o camera pose</td><td>55.3</td><td>62.4</td><td>40.1</td><td>43.9</td></tr><tr><td>w/ camera pose</td><td>56.4</td><td>62.9</td><td>40.6</td><td>44.1</td></tr></table>

Table 14: Effect of camera pose supervision on nuScenes [3]. The camera pose branch is only used as auxiliary supervision during training and is not directly used by downstream perception heads during inference.

Effect of camera pose supervision. We further study the effect of the auxiliary camera pose supervision. Although the predicted camera pose is not directly used by the downstream perception heads during inference, it provides an explicit camera-level geometric constraint during training. As shown in Table 14, removing camera pose supervision consistently degrades detection and occupancy performance. This indicates that camera pose prediction is not merely an auxiliary output, but helps the shared backbone organize multi-view observations into a more metric and geometry-grounded 3D scene latent. Such a latent benefits both instance-level detection and volume-level occupancy prediction.

## D More Qualitative Results.

We provide more qualitative results in Figures 7, 8, and 9. These visualizations cover different datasets and perception tasks, including semantic occupancy prediction on Occ3DnuScenes [59], point map visualization generated from predicted depth on KITTI [15] and DDAD [17], and 3D object detection on nuScenes [3], Argoverse 2 [72], and Waymo [58]. The results further show that GeoUP produces reliable perception outputs across diverse sensor configurations, driving scenes, and task settings.

![](images/829fc184bfa281f1e71ee2043212ed4fa12870164054fd01230806be3e4c8b0a.jpg)  
Figure 7: Occupancy prediction on Occ3D-nuScenes [59]. GeoUP produces more complete occupancy predictions than OPUS-V2 [61].

![](images/813749ccac0284cd9f10384a12f27d05317200e691f0031b09f23210de35d207.jpg)  
Figure 8: Point maps generated from predicted depth on KITTI [15] and DDAD [17].

![](images/3540693ed4e41cae496f62e33c40057dab379158d36a48ad29b9520478455294.jpg)  
Figure 9: 3D object detection on nuScenes [3], Argoverse 2 [72], and Waymo [58]. Blue boxes denote ground-truth annotations, and red boxes denote predictions.