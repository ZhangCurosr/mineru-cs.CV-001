# COSTA: A Cluster-Centric Paradigm for Annotation-Free Open-Set Semantic Segmentation of Aerial Point Clouds with Domain Shifts

Yanghong Lin<sup>1,2,3,4†</sup>, Li Fang<sup>2,3,4†</sup>, Tianyu Li<sup>2,3,5</sup>, Shudong Zhou<sup>2,3,5</sup>, Wei Yao<sup>2,3\*</sup>

<sup>1\*</sup>Fujian Institute of Research on the Structure of Matter, Chinese Academy of Sciences, Fuzhou, 350108, Fujian, China.

<sup>2</sup>Spatial Intelligence and Urban Computing, Institute of Urban Environment, Chinese Academy of Sciences, Xiamen, 361021, Fujian, China.

<sup>3</sup>State Key Lab for Ecological Security of Regions and Cities, Institute of Urban Environment, Chinese Academy of Sciences, Xiamen, 361021, Fujian, China.

<sup>4</sup>University of Chinese Academy of Sciences, Beijing, 101408, Beijing, China.

<sup>5</sup>School of Resource and Environmental Sciences, Wuhan University, Wuhan, 430072, Hubei, China.

\*Corresponding author(s). E-mail(s): wyao@iue.ac.cn; Contributing authors: linyanghong21@mails.ucas.ac.cn; lfang@iue.ac.cn; 2025182050063@whu.edu.cn; sdzhou@whu.edu.cn; <sup>†</sup>These authors contributed equally to this work.

## Abstract

Semantic segmentation of aerial point cloud is trapped in a generalization crisis under distinct domain shifts. While test-time adaptation ofers a privacypreserving and computationally eficient way to adapt pre-trained models to unlabeled target-domain data during inference, existing methods, bound to closed-set label assumptions and non-scalable point-wise segmentation pipelines, still struggle with semantic shifts. We ask: can we adapt any given pre-trained aerial point cloud segmentation model to a shifted target domain at the inference

phase alone, without additional training, while segmenting target-specific categories beyond the source label space on demand? This paper introduces COSTA, which breaks this limitation by shifting from closed-set point-wise adaptation to cluster-centric open-set semantic propagation. Our core discovery is that, once efectively adapted at test time, the rich feature distribution of aerial point clouds can be distilled into a compact set of well-separated semantic centroids that are transferable across label spaces. COSTA leverages this to reformulate openset semantic segmentation as a cluster-level propagating process: it first bridges the domain gap through proven test-time adaptation, then groups each batch of target-domain points into a small set of semantic clusters based on the similarity distribution in the adapted feature space, and finally propagates high-confidence pseudo labels obtained from an open-vocabulary vision-language model to all points through cluster-level voting. This cluster-centric paradigm enables testtime adaptation of aerial point clouds under significant domain gaps with mixed semantic shifts. With DALES as the source domain, COSTA enables on-demand segmentation across three aerial point cloud benchmarks with distinct domains and heterogeneous category spaces, achieving up to 70.09% mIoU under this new setting.

Keywords: Point cloud semantic segmentation, Open-set test-time adaptation, Rendered multi-view images

## 1 Introduction

Semantic segmentation of aerial point clouds is a fundamental task for urban perception and 3D scene understanding. By assigning semantic labels to individual 3D points, it provides structured scene knowledge for urban 3D modeling (Yao et al. 2011; Xiang et al. 2024; Shao et al. 2024), aerial-ground collaborative perception (Polewski et al. 2015; Zhu et al. 2023; Hou et al. 2026) and autonomous UAV navigation (Wang et al. 2025; Gao et al. 2026).

In recent years, deep neural networks have emerged as leading methods for this task by learning high-level representations from irregular point clouds (Hu et al. 2022; Li et al. 2025). Despite remarkable advances, existing methods are still largely built upon a closed-set and domain-specific learning paradigm (Choy et al. 2019; Kolodiazhnyi et al. 2024; Wu et al. 2024). Most models are trained and evaluated on specific 3D benchmarks with predefined semantic categories and densely annotated point-wise labels, often ignoring generalization and transferability across diferent 3D domains. However, various 3D domains may difer drastically in data distributions, spatial scales, point densities, and noise patterns, leading to significant degradation of domain-specific pre-trained models (Wang et al. 2025; Zou et al. 2026). To address this issue, domain adaptation (DA) has been widely explored to alleviate such domain shifts by transferring knowledge from a labeled source domain to an unlabeled or weakly labeled target domain. Among diferent adaptation settings, test-time adaptation (TTA) adapts a pre-trained model directly to unlabeled target-domain data during inference, without accessing the source-domain data or launching an additional training process. Due to its eficiency, low computational cost, and ability to avoid the potential security risks associated with the source, TTA has attracted increasing attention and has been explored for aerial point clouds in recent years (Wang et al. 2025; Gao et al. 2026). While ofering a promising solution to this problem, existing TTA methods remain confined to closed-set adaptation, due to closed-set label assumptions and non-scalable point-wise segmentation frameworks. In real-world deployments, shifts in geographic location, data distribution, and user requirements naturally introduce target data that contain not only known classes overlapping with the source, but also novel classes with finer granularity or previously unseen semantics. This open-set challenge fundamentally limits the applicability of current TTA techniques, whose closed-set predictions inherently fail to accommodate emerging classes. Although recent advances (Li et al. 2023; Gao et al. 2024; Zou et al. 2026) have begun to acknowledge the open-set challenge, they typically treat unseen classes merely as out-of-distribution (OOD) samples, with the aim of preventing degradation of adaptation in known classes. As a result, segmentation remains restricted to categories that overlap with the source.

Open-vocabulary vision-language models (VLMs) provide a promising opportunity to break this semantic bottleneck. By conditioning segmentation on natural language prompts, VLMs can segment or localize categories beyond the source, enabling ondemand semantic understanding. Nevertheless, directly transferring open-vocabulary image segmentation to aerial point clouds is non-trivial. Most powerful VLMs are trained and operate primarily on 2D natural images (Carion et al. 2025; Yuan et al. 2025), while aerial point clouds are sparse, irregular, and three-dimensional, inevitably resulting in a substantial modality gap. A straightforward and widely adopted strategy is to exploit images aligned well with point clouds (Peng et al. 2023; Yang et al. 2024; Jiang et al. 2024). These images are fed into 2D VLMs to obtain categoryspecific 2D prediction masks or visual features, which are then back-projected onto the corresponding 3D points as pseudo labels or feature-alignment supervision for 3D semantic segmentation. However, these attempts to circumvent the modality gap are fundamentally limited in aerial scenarios. Drone or satellite images inevitably difer from natural images in viewpoint, spatial resolution, and imaging geometry, and are subject to occlusion and incomplete visibility (Wang et al. 2026; Lin et al. 2026). As a result, the 2D supervision derived from VLMs is often incomplete, noisy, and spatially inconsistent. Direct domain adaptation on the VLMs to mitigate these discrepancies is also a suboptimal solution: it requires updating a much larger parameter space, introduces substantial computational overhead, and may not guarantee direct improvements in 3D point cloud segmentation. Moreover, in most aerial point cloud benchmarks, the frequent absence of well-aligned synchronously acquired images further limits the direct applicability of 2D VLMs (Xu et al. 2025; Wang et al. 2026). Even for the few benchmarks equipped with auxiliary imagery, such as Hessigheim 3D (K¨olle et al. 2021), the large temporal gap between image and point cloud acquisition makes the supervision for dynamic or time-sensitive categories, such as vehicles and growing vegetation, unreliable or invalid.

To this end, we consider a more realistic setting, i.e. annotation-free open-set test-time adaptation (OSTTA): during inference, a pre-trained aerial point cloud segmentation model is required to adapt to an unlabeled target domain without source data or additional training, while simultaneously recognizing target-specific categories beyond the source label space. To the best of our knowledge, it has not been explored before. We posit that the key lies in a paradigm shift from non-scalable closed-set point-wise adaptation and prediction to cluster-centric open-set semantic voting. Our core observation is that, after efective test-time adaptation, points with similar semantics in the target domain tend to form compact local clusters in the adapted feature space, and each cluster exhibits high semantic purity (Fig. 1). This observation extends beyond the source-domain label space and helps counteract the noisy pseudo-labels produced by VLMs. Inspired by this, we introduce COSTA, a novel cluster-centric paradigm for annotation-free open-set semantic segmentation of aerial point clouds under significant domain gaps with mixed semantic shifts.

![](images/8056d8adfe69d659910563bc70071350da269b253080e3a171654754d7429471.jpg)

![](images/0eeddccec0e4d7c18e1102e0fdb88f41c88a7bb42dbc6388bad75a87e96ccedd.jpg)  
Fig. 1 With DALES as the source domain and SUM as the target domain, we partition the targetdomain data with K-Means in TTA feature space. The results show that most clusters exhibit high semantic purity: over 80% of the clusters have a proportion of same-class points ranging from 0.9 to 1.0. Notably, this observation extends beyond the source label space.

COSTA exploits the complementary strengths of the TTA 3D segmentation network and open-vocabulary VLMs using clusters as a semantic bridge. The 3D network provides a target-adapted feature space where semantically coherent clusters can be discovered beyond the source label space, while the VLMs supply flexible category-level evidence on demand. Rather than blindly trusting noisy point-wise VLMs predictions or being restricted to a closed-set classifier head, COSTA aggregates reliable openvocabulary cues at the cluster level and propagates them back to points. This design makes open-set semantic adaptation both label-extensible and robust to noisy pseudo labels.

Our contributions are summarized as follows.

• We introduce COSTA, the first cluster-centric open-set test-time adaptation framework for aerial point cloud semantic segmentation. By integrating a lightweight test-time adaptation strategy that optimizes learnable BN parameters via entropy minimization, information maximization, and OOD discovery, COSTA enables a source-pretrained model to adapt to unlabeled target domains during inference and segment target-specific categories beyond the source label space on demand.

• We design an adaptive hybrid-view projection module that combines oblique virtualcamera and BEV projections, allowing seamless application of 2D vision-language models to 3D aerial point clouds.

• We propose a reliability-aware cluster-centric propagation mechanism that bridges the adapted 3D feature space with open-vocabulary VLMs evidence. By aggregating multi-view cues through source-specific response thresholds and point-level reliability estimation, it propagates high-confidence pseudo labels via cluster-level voting, enabling robust generalization under domain gaps with mixed semantic shifts.

• We establish three heterogeneous aerial point cloud benchmarks covering two representative semantic shifts: unseen classes and cross-granularity classes. Extensive experiments demonstrate that COSTA enables annotation-free on-demand openset test-time adaptation segmentation and achieves SOTA performance under this setting.

## 2 Related Work

## 2.1 Point Cloud Semantic Segmentation

The interpretation of 3D point clouds has evolved from basic geometric grouping to complex semantic scene parsing. Recently, point-based networks have become a dominant paradigm for directly learning from unordered point sets. PointNet (Qi et al. 2017) first introduced symmetric aggregation functions to achieve permutation invariance, while PointNet++ (Qi et al. 2017) further incorporated hierarchical local feature learning to better capture fine-grained geometric structures. Later works improved point-wise representation learning through more efective local aggregation and scalable sampling strategies. For example, RandLA-Net (Hu et al. 2020) employs random sampling with local feature aggregation to eficiently process large-scale point clouds, and PointNeXt (Qian et al. 2022) revisits PointNet++ with stronger training and scaling strategies, showing that carefully designed optimization recipes and scalable architectures can substantially improve segmentation performance.

Another line of research converts irregular point clouds into structured or semistructured representations to enable convolutional processing. Sparse convolutional networks, such as MinkowskiNet (Choy et al. 2019), operate on sparse tensors and significantly reduce computation by applying convolutions only to occupied locations. KPConv (Thomas et al. 2019) defines convolution kernels directly in Euclidean space through learnable kernel points, enabling flexible local geometric modeling without voxelizing the input. Related sparse point-voxel designs, such as SPV-NAS/SPVCNN (Tang et al. 2020), further balance point-level geometric precision and voxel-level computational eficiency. These convolution-based methods have achieved strong performance on both indoor and outdoor benchmarks and remain widely used as backbones for large-scale 3D point cloud segmentation.

More recently, Transformer-based architectures have advanced fully supervised point cloud segmentation by modeling long-range dependencies and global context.

Point Transformer and its successors (Zhao et al. 2021; Wu et al. 2022, 2024) adapt self-attention to irregular 3D data and achieve strong performance through local attention, grouped vector attention, and scalable point serialization. Swin3D (Yang et al. 2025) introduces sparse-window attention and pretraining strategies for 3D scene understanding, while OctFormer (Wang 2023) leverages octree-based attention to improve scalability on large point clouds. Recent unified frameworks such as One-Former3D (Kolodiazhnyi et al. 2024) further extend Transformer-based designs to semantic, instance, and panoptic segmentation within a single architecture. These methods demonstrate the strong representation capacity of the transformer-based backbone.

Despite these advances, point cloud segmentation remains constrained by a closedset and annotation-intensive learning paradigm. The model is trained on a specific benchmark to predict a fixed set of predefined classes, and its performance largely depends on the availability of dense point-wise annotations in the target domain (Wang and Yao 2022). When deployed in aerial or urban scenes with domain shifts (e.g., diferent sensors, regions, classes), supervised models typically sufer significant performance degradation or fail entirely, necessitating costly re-annotation and retraining. This limitation motivates our open-set test-time adaptation setting, where a sourcepretrained model is adapted to unlabeled target point clouds during inference while supporting user-specified classes beyond the source.

## 2.2 Test-time Adaptation

Among various approaches for addressing covariate shifts induced by domain gaps, test-time adaptation has attracted increasing attention due to its challenging setting, where a source-trained model is adapted to the target distribution during inference accessing only unlabeled target data. Early work focused primarily on 2D vision, especially for image classification and semantic segmentation. Representative methods such as TENT (Wang et al. 2021) minimize prediction entropy by updating normalization statistics and afine parameters of the BN layer during inference, while EATA (Niu et al. 2022) improves eficiency and stability through reliable sample selection and anti-forgetting regularization. Continual TTA methods further consider non-stationary target streams, where CoTTA (Wang et al. 2022) reduces error accumulation by using weight-averaged predictions and stochastic restoration. More recently, TTA has been extended to 3D point cloud understanding. GIPSO (Saltori et al. 2022) is the first method for TTA 3D point cloud segmentation by introducing geometrically informed pseudo-label propagation for online adaptation. HGL (Zou et al. 2024) exploits local-global hierarchical pseudo-label strategy to improve TTA 3D segmentation. For geospatial point clouds, recent work adapts pretrained models by updating the statistics and learnable afine parameters of BN layer with self-supervised learning (Wang et al. 2025). TTT-KD (Weijler et al. 2025) further explores test-time training for 3D semantic segmentation by distilling knowledge from 2D foundation models. While promising, most existing TTA methods are designed under a closedset assumption, where the source and target domains share the same semantic. These methods mainly address covariate shifts caused by sensor changes, weather, or density variations, but do not explicitly handle target classes that are absent from or inconsistent with the source.

Recently, some works have paid more attention to improving the robustness of TTA methods. NOTE (Gong et al. 2022) uses instance-aware batch normalization and prediction-balanced reservoir sampling to alleviate biased online updates, while RoTTA (Yuan et al. 2023) introduces robust teacher-student adaptation with timeaware reweighting for dynamic scenarios. SAR (Niu et al. 2023) analyzes several practical failure cases of TTA, including mixed distribution shifts, small batch sizes, and imbalanced online label distributions, and stabilizes the minimization of entropy by filtering unreliable samples and encouraging flat minima. Another line of work studies open-world TTA, where the test data contains unknown classes. OWTTT (Li et al. 2023) improves robustness by pruning strong OOD samples and expanding prototypes. UniEnt (Gao et al. 2024) performs entropy minimization on in-distribution (ID) samples and entropy maximization on pseudo-covariate-shifted OOD samples. Multimodal and stabilized open world TTA methods further refine unknown-sample filtering and entropy-aware optimization (Dong et al. 2025; Lee et al. 2025). Recently, GOOD (Zou et al. 2026) introduced the first open world TTA method for 3D point cloud semantic segmentation, which enhances the discrimination between ID and OOD regions by integrating a novel confidence metric and prototype representation. However, these robust TTA methods focus on improving the model’s ability to distinguish between ID and OOD samples and preventing OOD or unknown samples from degrading the adaptation of known classes. This fundamentally difers from our OSTTA setup, which aims not only to preserve the model’s adaptability to known classes but also to recognize user-specified target classes beyond the source.

## 2.3 Open-Vocabulary Semantic Segmentation

To overcome closed-set constraints, open-vocabulary segmentation has been developed using large-scale vision-language models (VLMs). Early works focused on 2D vision, primarily using 2D VLMs pre-trained on image-text pairs to perform open-vocabulary classification or segmentation of images. OpenSeg (Ghiasi et al. 2022) and MaskCLIP (Dong et al. 2023) exploit CLIP-like representations for openvocabulary semantic, instance, or panoptic segmentation. Subsequent methods further improve the spatial precision and semantic alignment of open-vocabulary segmentation. ODISE (Xu et al. 2023) leverages text-to-image difusion representations for open-vocabulary panoptic segmentation, while CAT-Seg (Cho et al. 2024) formulates segmentation as a cost aggregation between dense image features and textual embeddings. These methods demonstrate the strong potential of language-conditioned visual models to generalize arbitrary classes.

Driven by the success of 2D VLMs, recent eforts have explored how to transfer open-vocabulary semantics from 2D foundation models to 3D representations. A common strategy is distilling knowledge from 2D VLMs into 3D backbones or aligning class-agnostic 3D proposals with multi-view image features. OpenScene (Peng et al. 2023) aligns 3D point features with CLIP image and text embeddings through multi-view distillation. PLA (Ding et al. 2023) introduces caption-assisted language supervision to associate 3D representations with semantically richer textual descriptions. SEAL (Liu et al. 2023) and OV3D (Jiang et al. 2024) further exploit 2D–3D correspondences and foundation models to improve open-vocabulary 3D semantic segmentation. Region-level methods, such as RegionPLC (Yang et al. 2024) and Open-Mask3D (Takmaz et al. 2023), aggregate multi-view language-aligned features over class-agnostic 3D regions or instance masks, thereby improving robustness to noisy point-wise alignment. However, these methods rely on spatially and temporally wellaligned RGB images, which are often not satisfied in aerial point cloud benchmarks. OpenUrban3D (Wang et al. 2026) leverages multi-view rendered point clouds and vision–language feature distillation to achieve open vocabulary semantic understand ing in large-scale urban scenes. More recently, some works, such as CitySeg (Xu et al. 2025), have attempted to replicate the success of 2D VLMs by directly incorporating textual semantics during large-scale city-scale point cloud training. However, due to the scarcity of 3D-text datasets, 3D point cloud open-vocabulary semantic segmentation models typically have fewer parameters and weaker generalization performance compared to 2D VLMs. The more pronounced disparities in perspective, geographical location, and density in aerial point clouds further amplify this issue (Wang et al. 2025). Moreover, the costly retraining inevitably limits the practicality of such methods.

## 3 Methods

## 3.1 Overview

Given an aerial point cloud segmentation model $M _ { s }$ pre-trained on a source domain $D _ { s }$ with source classes $\mathcal { C } _ { s }$ , our objective is to design an open-set test-time adaptation method $f$ that adapts $M _ { s }$ to an unlabeled target domain $D _ { t }$ and predicts point-wise semantic labels during inference. Unlike conventional closed-set TTA, the user can specify a set of target classes $\mathcal { C } _ { t }$ on demand via text queries, where $\mathcal { C } _ { t }$ is not restricted to the source classes $( \mathrm { i . e . , } \mathcal { C } _ { t }$ may contain categories beyond $\mathcal { C } _ { s } )$ . Formally, the method $f$ produces an adapted model $M _ { t }$ and the corresponding point-wise predictions $\hat { Y } _ { t }$ as

$$
M _ { t } , \hat { Y } _ { t } = f ( M _ { s } , D _ { t } , \mathcal { C } _ { t } ) .\tag{1}
$$

with the objective of minimizing the expected risk over the target distribution while producing predictions under semantic shifts.

To address the challenge of cross-domain generalization under semantic shif ${ \mathrm { ; s , } }$ we propose a novel cluster-centric framework for annotation-free open-set test-time adaptation, namely COSTA (see Fig. 2). First, raw point clouds are rendered into a complementary 2D image via adaptive hybrid multi-view projection, including oblique virtual camera projections and bird’s-eye-view (BEV) projections, thereby avoiding dependency on well-aligned synchronously acquired images. By querying with userspecified text prompts, the VLMs generate probabilistic 2D masks from the rendered image for arbitrary classes demanded in the target domain. To recover high-quality 3D pseudo-supervision, we further introduce a reliability-aware pseudo-labeling strategy based on class-wise adaptive detection banks and multi-view evidence accumulation strategy, which back-projects and fuses VLMs-derived probability masks into pointwise pseudo labels. Subsequently, we directly optimize the batch normaolization (BN) layers of the source pre-trained model through a diferentiated self-supervised entropy optimization strategy on pseudo-ID and pseudo-OOD samples from the test data, which aims to learn target domain-specific knowledge. Finally, the adapted pointwise features are grouped by K-Means into semantically coherent clusters, followed by open-set reliability-weighted semantic voting over all trustworthy bank points within each cluster.

![](images/aa358bdd0b7a211937cbee6a5b6515b8eec1ff47d9b259d6ddde95158ac31ddd.jpg)  
Fig. 2 Overview of the proposed COSTA for OSTTA. We first perform hybrid multi-view rendering on target point cloud by exploiting complementary BEV and oblique virtual camera projection. This yields hybrid view set with pixel-to-point correspondences for seamless VLMs semantic querying without well-aligned images. Given user-specified target classes, we prompt VLMs to generate open-vocabulary 2D probability masks, which are filtered adaptively and fused through reliabilityconstrained multi-view evidence accumulation. This provides reliability-aware pseudo labels for the next step. Given the pseudo labels, we adapt source-pretrained 3D model to the target data through diferentiated self-supervised learning during inference. Finally, we perform reliability-weighted cluster-level semantic propagation by clustering the adapted point features. The framework produces on-demand open-set semantic segmentation on the target domain without source-data replay, target annotations, or ofline retraining.

## 3.2 Pseudo-labeling with Reliability Constraint

A direct approach to utilizing VLMs in open-set TTA is to fetch predictions for target classes from synchronously acquired, well-calibrated images and then establish the correspondence between the point cloud and the image with prior calibration information, thereby inferring the point predictions from the VLMs. However, most aerial point cloud datasets either lack such calibrated imagery or exhibit significant temporal misalignment, severely limiting the applicability of VLMs. Moreover, these raw predictions are noisy and can introduce error accumulation from the 2D VLMs into the 3D models. Therefore, we introduce an efective, reliability-aware pseudo-labeling strategy for filtering VLMs predictions inferred from hybrid rendered images.

## 3.2.1 Hybrid Multi-View Rendering

To address the challenge of the frequent absence of corresponding images in aerial point clouds, we design an adaptive hybrid multi-view projection module that synthesizes information-rich rendered images. The hybrid multi-view projection module constructs a hybrid view set that fully exploits the respective advantages of tilted oblique virtualcamera and BEV projections (see Algorithm 1). The former captures comprehensive side-view geometric appearance of vertical structures from diverse perspectives, while the latter provides stable top-down observations that are particularly efective for small top-visible objects. Unlike fixed projection parameters, the projection configuration is derived from the spatial extent of each scene, making the rendered images adaptive to scene scale and point density.

Formally, given a target point cloud $\mathcal { P } _ { t } = \{ p _ { i } \} _ { i = 1 } ^ { N }$ , where $p _ { i } = [ x _ { i } , y _ { i } , z _ { i } , r _ { i } , g _ { i } , b _ { i } ]$ ], we compute its axis-aligned bounds and scene scale as

$$
W = x _ { \mathrm { m a x } } - x _ { \mathrm { m i n } } , \quad L = y _ { \mathrm { m a x } } - y _ { \mathrm { m i n } } , \quad H = z _ { \mathrm { m a x } } - z _ { \mathrm { m i n } } , \quad s = \sqrt { W L } .\tag{2}
$$

Tiled oblique virtual-camera projection: For the tilted virtual-camera projection, we partition the horizontal plane into $K _ { l } \times K _ { l }$ local regions. For each region $( i , j )$ with $i , j \in \{ 1 , \dots , K _ { l } \}$ , a camera anchor $\mathbf { a } _ { i j }$ and a target point $\mathbf { t } _ { i j }$ are defined adaptively based on the scene scale to guide the viewpoint of the virtual camera as

$$
{ \bf a } _ { i j } = \left( x _ { \operatorname* { m i n } } + \frac { i W } { K _ { l } + 1 } , y _ { \operatorname* { m i n } } + \frac { j L } { K _ { l } + 1 } , z _ { \operatorname* { m i n } } + H + \frac { s } { 2 K _ { l } } \right) ,\tag{3}
$$

$$
\mathbf { t } _ { i j } = \left( a _ { i j } ^ { x } , a _ { i j } ^ { y } , z _ { \operatorname* { m i n } } + H + \frac { s } { \rho K _ { l } } \right) ,\tag{4}
$$

Multiple virtual cameras orbit around the camera anchor $\mathbf { a } _ { i j }$ along a circular horizontal path of radius $R _ { l }$ and are oriented toward the target point $\mathbf { t } _ { i j }$ . Thus, the world coordinates of the virtual camera orbiting camera anchor $\mathbf { a } _ { i j }$ are computed as

$$
{ \bf c } _ { i j k } = { \bf a } _ { i j } + \left( R _ { l } \cos \theta _ { k } , R _ { l } \sin \theta _ { k } , 0 \right) , \quad R _ { l } = \frac { s } { \rho K _ { l } } .\tag{5}
$$

where the hyper-parameter $\rho$ controls the radius of the camera’s circular orbit. To ensure that each virtual camera covers the entire local region, the field of view (FOV) of each virtual camera is set adaptively. Let $d _ { l } = \sqrt { ( W / K _ { l } ) ^ { 2 } + ( L / K _ { l } ) ^ { 2 } }$ be the diagonal length of a local region cell and $h _ { l } = a _ { i j } ^ { z } - t _ { i j } ^ { z }$ be the vertical camera-target distance. The FOV is given by

$$
\mathrm { F O V } _ { i j } = \operatorname* { m i n } \left( 2 \arctan \frac { d _ { l } } { 2 h _ { l } } , 1 2 0 ^ { \circ } \right) .\tag{6}
$$

which ensures that each local view contains suficient spatial context while preventing extreme perspective distortion with safety margin.

Given the camera position, target point, and FOV, each virtual camera is represented by a standard pinhole camera model. The model consists of an extrinsic matrix, which transforms points from the world coordinate system to the camera coordinate system, and an intrinsic matrix, which maps camera-coordinate of points onto the image plane.

The extrinsic matrix is first determined by the camera pose. Specifically, for a virtual camera located at c and looking at target point t, we construct an orthonormal camera coordinate system using the world-up direction $\mathbf { u } = [ 0 , 0 , 1 ] ^ { \top }$ . The forward, right, and image-downward axes are computed as

$$
\mathbf { f } = { \frac { \mathbf { t } - \mathbf { c } } { \| \mathbf { t } - \mathbf { c } \| _ { 2 } } } , \quad \mathbf { r } = { \frac { \mathbf { f } \times \mathbf { u } } { \| \mathbf { f } \times \mathbf { u } \| _ { 2 } } } , \quad \mathbf { d } = { \frac { \mathbf { f } \times \mathbf { r } } { \| \mathbf { f } \times \mathbf { r } \| _ { 2 } } } .\tag{7}
$$

Here, f represents the viewing direction, while r and d define the horizontal and vertical axes of the image plane, respectively. The world-to-camera rotation and translation are then obtained as

$$
{ \bf R } = \left[ \begin{array} { l l } { { \bf r } ^ { \top } } \\ { { \bf d } ^ { \top } } \\ { { \bf f } ^ { \top } } \end{array} \right] , \quad { \bf t } _ { \mathrm { c a m } } = - { \bf R } { \bf c } ,\tag{8}
$$

forming the extrinsic matrix

$$
{ \bf E } = \left[ { \bf R \Lambda t } _ { \mathrm { c a m } } \right] .\tag{9}
$$

The intrinsic matrix is determined by the image resolution and the FOV. For an image of size $W _ { I } \times H _ { I }$ , the focal length is $f _ { I } = ( W _ { I } / 2 ) / \tan ( \mathrm { F O V / 2 } )$ , yielding the intrinsic matrix

$$
\mathbf { K } = \left[ \begin{array} { l l l } { f _ { I } } & { 0 } & { W _ { I } / 2 } \\ { 0 } & { f _ { I } } & { H _ { I } / 2 } \\ { 0 } & { 0 } & { 1 } \end{array} \right] .\tag{10}
$$

A point is projected onto the image plane through this camera model. During rendering, we maintain a depth bufer: when multiple points fall into the same pixel, only the point with the nearest camera depth is preserved. This produces a rendered RGB image, a depth map, and a visibility-aware point-to-pixel correspondence.

BEV projection: For BEV projection, we set a sliding window to process overlapping point cloud tiles. Given $s _ { g } , G$ , and $\Delta _ { g }$ that controls the scale, window size, and moving steps of the sliding windows, we extract points in the current sliding windows to reduce the following data processing volume. For the m-th BEV tile with origin $\mathbf { o } _ { m } = \left( o _ { m } ^ { x } , o _ { m } ^ { y } \right)$ , a point $p _ { i }$ is rasterized into pixel coordinates $( u _ { i , m } ^ { \mathrm { b e v } } , v _ { i , m } ^ { \mathrm { b e v } } )$ for BEV RGB and depth queries, where u and v are column and row indices, respectively:

$$
u _ { i , m } ^ { \mathrm { b e v } } = \left\lfloor \frac { x _ { i } - o _ { m } ^ { x } } { s _ { g } } \right\rfloor , \quad v _ { i , m } ^ { \mathrm { b e v } } = \left\lfloor \frac { y _ { i } - o _ { m } ^ { y } } { s _ { g } } \right\rfloor .\tag{11}
$$

Since each tile covers an area of $G \times G$ meters and one pixel corresponds to $s _ { g }$ meters, the resulting BEV image has a spatial resolution of $\lfloor G / s _ { g } \rfloor \times \lfloor G / s _ { g } \rfloor$ pixels. Only points whose computed pixel indices fall within this range are considered for the tile. If multiple points fall into the same BEV pixel, the topmost point is retained. The retained point defines both the BEV RGB value and the depth map indexed by $( u _ { i , m } ^ { \mathrm { b e v } } , v _ { i , m } ^ { \mathrm { b e v } } )$ . This top-down projection is particularly suitable for aerial scenarios, which reduces the occlusion efects caused by oblique projections on low or small objects.

To mitigate sparse holes caused by low point density, we further apply pixel-level completion for rendered images, especially for the internal areas and margins. For BEV images, missing pixels are filled by iterative $ { 3 \times 3 ~ 2 \mathrm { D } }$ max-pooling:

$$
X ^ { ( l + 1 ) } ( u , v ) = \left\{ \begin{array} { l l } { X ^ { ( l ) } ( u , v ) , } & { X ^ { ( l ) } ( u , v ) \neq \emptyset , } \\ { \underset { ( a , b ) \in \mathcal { N } _ { 3 } ( u , v ) } { \operatorname* { m a x } } X ^ { ( l ) } ( a , b ) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{12}
$$

Here, $X ^ { ( l ) }$ denotes the BEV map after the l-th completion iteration, where $l = 0$ corresponds to the initially rasterized sparse map. For virtual-camera views, completion is similar and depth-guided. At each iteration, for a missing pixel $( u , v )$ , COSTA searches its $3 \times 3$ neighborhood and selects the valid neighboring pixel with the smallest positive depth. The missing RGB value is then duplicated from this nearest visible neighbor, which reduces background bleed-in and preserves object boundaries. The completion only improves visual continuity for VLMs inference; the subsequent back-projection still relies on the original visibility-aware correspondences.

## 3.2.2 Reliability-constrained Open-vocabulary Pseudo-labeling

Pre-trained VLMs ofer an efective approach to open-vocabulary semantic segmentation. Given the hybrid rendered view set $\mathcal { V } = \mathcal { V } _ { \mathrm { o b l } } \cup \mathcal { V } _ { \mathrm { b e v } }$ , COSTA queries the VLMs in a class-prompt manner. For each target class $c \in { \mathcal { C } } _ { t } .$ the VLMs return a probabilistic mask $\bar { S _ { \nu , c } } \in \ [ 0 , 1 ] ^ { H \times W }$ for the rendered view $\nu \in \mathcal V$ . This prompt-conditioned inference allows COSTA to obtain semantic evidence for arbitrary user-specified classes, including categories beyond the source.

However, raw VLMs masks are not directly reliable as 3D pseudo labels. Due to significant domain gaps, occlusions, and poor rendering quality, these predictions are incomplete and inconsistent across views. Using them for direct propagation might be suboptimal. Therefore, we propose a reliability-aware pseudo-labeling strategy based on class-wise adaptive detection banks and multi-view evidence accumulation to suppress unreliable predictions naturally.

Algorithm 1 Hybrid Multi-View Projection   
Require: Target point cloud $\mathcal { P } _ { t _ { \star } , \tau } = \{ p _ { i } = [ x _ { i } , y _ { i } , z _ { i } , r _ { i } , g _ { i } , b _ { i } ] \} _ { i = 1 } ^ { N } ;$ local grid number   
$K _ { l } ;$ camera angles $\Theta = \{ \theta _ { k } \} _ { k = 1 } ^ { K _ { \theta } } ;$ orbit factor $\rho ;$ image size $( W _ { I } , H _ { I } )$ ; BEV scale   
$s _ { g } ;$ BEV tile size $G ;$ BEV stride $\Delta _ { g } ;$ completion iterations $L _ { c }$   
Ensure: Hybrid rendered view set $\mathcal { V } = \mathcal { V } _ { \mathrm { o b l } } \cup \mathcal { V } _ { \mathrm { b e v } }$   
1: Step 1: Compute adaptive scene scale   
2: Compute scene bounds $( x _ { \mathrm { m i n } } , x _ { \mathrm { m a x } } , y _ { \mathrm { m i n } } , y _ { \mathrm { m a x } } , z _ { \mathrm { m i n } } , z _ { \mathrm { m a x } } )$ from $\mathcal { P } _ { t }$   
3: $W \gets x _ { \mathrm { m a x } } - x _ { \mathrm { m i n } } , L \gets y _ { \mathrm { m a x } } - y _ { \mathrm { m i n } } , H \gets z _ { \mathrm { m a x } } - z _ { \mathrm { m i n } } , s \gets \sqrt { W L }$   
4: Initialize $\gamma _ { \mathrm { o b l } }  \emptyset$ and $\nu _ { \mathrm { b e v } }  \emptyset$   
5: Step 2: Generate tiled oblique virtual-camera views   
6: $d _ { l } \gets \sqrt { ( W / K _ { l } ) ^ { 2 } + ( L / K _ { l } ) ^ { 2 } }$   
7: for $i = 1$ to $K _ { l }$ do   
8: for $j = 1 \mathrm { t o } K _ { l }$ do   
9: $\begin{array} { r } { \mathbf { a } _ { i j } \gets \left( x _ { \operatorname* { m i n } } + \frac { i W } { K _ { l } + 1 } , y _ { \operatorname* { m i n } } + \frac { j L } { K _ { l } + 1 } , z _ { \operatorname* { m i n } } + H + \frac { s } { 2 K _ { l } } \right) } \end{array}$ ▷ camera anchor   
10: $\begin{array} { r } { \mathbf { t } _ { i j } \gets \left( a _ { i j } ^ { x } , a _ { i j } ^ { y } , z _ { \operatorname* { m i n } } + H + \frac { s } { \rho K _ { l } } \right) } \end{array}$ ▷ target point   
11: $\begin{array} { r } { R _ { l } \gets \dot { \frac { s } { \rho K _ { l } } } , h _ { l } \gets a _ { i j } ^ { z } - t _ { i j } ^ { z } } \end{array}$   
12: $\begin{array} { r } { \mathrm { F O V } _ { i j } \gets \operatorname* { m i n } \left( 2 \arctan \frac { d _ { l } } { 2 h _ { l } } , 1 2 0 ^ { \circ } \right) } \end{array}$ ▷ field of view   
13: for all $\theta _ { k } \in \Theta$ do   
14: ${ \bf c } _ { i j k } \gets { \bf a } _ { i j } + ( R _ { l } \cos \theta _ { k } , R _ { l }$ sin $\theta _ { k } , 0 )$ ▷ coord of virtual camera   
15: $\mathbf { E } _ { i j k }  \mathrm { L o o k A t } ( \mathbf { c } _ { i j k } , \mathbf { t } _ { i j } )$ ▷ extrinsic of virtual camera   
16: $\mathbf { K } _ { i j k } \gets \mathrm { I n t r i n s i c } ( \dot { W } _ { I } , \dot { H } _ { I } , \mathrm { F O V } _ { i j } )$ ▷ intrinsic of virtual camera   
17: Render RGB image $I ,$ depth map D, and correspondence map Γ with   
depth bufering   
18: $( I , D ) $ DepthGuidedCompletion $( I , D , L _ { c } )$   
19: $\mathcal { V } _ { \mathrm { o b l } }  \mathcal { V } _ { \mathrm { o b l } } \cup \{ ( I , D , \Gamma , \mathbf { K } _ { i j k } , \mathbf { E } _ { i j k } ) \}$   
20: end for   
21: end for   
22: end for   
23: Step 3: Generate sliding-window BEV views   
24: Generate BEV window origins $\mathcal { O } = \{ ( o _ { m } ^ { x } , o _ { m } ^ { y } ) \} _ { m = 1 } ^ { M }$ with tile size $G$ and stride $\Delta _ { g }$   
25: for all $\mathbf { o } _ { m } = ( o _ { m } ^ { x } , o _ { m } ^ { y } ) \in \mathcal { O }$ do   
26: $R _ { \mathrm { b e v } }  \lfloor G / s _ { g } \rfloor$ ▷ spatial resolution of BEV   
27: Rasterize points into BEV RGB image $I ^ { \mathrm { b e v } }$ , height map $Z ^ { \mathrm { b e v } }$ , and correspon  
dence map Γ<sup>bev</sup>   
28: Retain the topmost point if multiple points fall into the same BEV pixel   
29: $( I ^ { \mathrm { b e v } } , Z ^ { \mathrm { b e v } } ) \doteq$ DepthGuidedCompletion $( I ^ { \mathrm { b e v } } , Z ^ { \mathrm { b e v } } , L _ { c } )$   
30: $\mathcal { V } _ { \mathrm { b e v } }  \mathcal { V } _ { \mathrm { b e v } } \cup \{ ( I ^ { \mathrm { b e v } } , Z ^ { \mathrm { b e v } } , \Gamma ^ { \mathrm { b e v } } , \mathbf { o } _ { m } , s _ { g } ) \}$   
31: end for   
32: Step 4: Return hybrid rendered views   
33: $\mathcal { V }  \mathcal { V } _ { \mathrm { o b l } } \cup \mathcal { V } _ { \mathrm { b e v } }$   
34: return V

For each queried class $^ { c , }$ a detection bank is initially constructed from mask responses larger than a low response threshold $\tau _ { 0 } .$ , inspired by Alami and Remondino (2026). The class-wise adaptive threshold is computed based on the statistical distribution as

$$
\boldsymbol { \tau } _ { c } = \mu _ { c } + \boldsymbol { a } _ { c } \boldsymbol { \sigma } _ { c } ,\tag{13}
$$

where $\mu _ { c } ^ { q }$ and $\sigma _ { c } ^ { q }$ denote the mean and standard deviation of valid mask responses collected from the rendered view set V. $\mathrm { B y }$ default, we set $a _ { c } ^ { q } = 0$ , reducing the threshold to the class-wise mean. This class-adaptive strategy avoids using a fixed global threshold, which is often suboptimal, since the response magnitudes of VLMs vary significantly across diferent classes.

After thresholding, COSTA back-projects the 2D masks to 3D points utilizing the established point-image correspondences. For a visible point $p _ { i }$ , if the mask response for class $c$ exceeds the adaptive threshold $\tau _ { c } .$ , it converts into normalized semantic evidence for a specific view:

$$
e _ { i , \nu , c } = \left\{ \begin{array} { l l } { \displaystyle \frac { S _ { \nu , c } ( \pi _ { \nu } ( i ) ) - \tau _ { c } } { 1 - \tau _ { c } } , } & { \displaystyle S _ { \nu , c } ( \pi _ { \nu } ( i ) ) \geq \tau _ { c } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{14}
$$

Here, $\pi _ { \nu } ( i )$ denotes the projected pixel coordinate of the point $p _ { i }$ in view $\nu .$ The point-wise evidence for class c is then aggregated by averaging all valid observations:

$$
s _ { i , c } = \frac { 1 } { | \mathscr { V } _ { i } | } \sum _ { \nu \in \mathscr { V } _ { i } } e _ { i , \nu , c } .\tag{15}
$$

where $\mathcal { V } _ { i } \subseteq \mathcal { V }$ denotes the set of views where $p _ { i }$ is visible. The preliminary pseudo label is determined by the class exhibiting the maximum aggregated evidence. To prevent erroneous forced assignments, points lacking valid evidence or possessing insuficient confidence are explicitly assigned an unknown label.

To select trustworthy semantic seeds, COSTA further computes a reliability score $r _ { i }$ for each pseudo-labeled point. The score is formulated by jointly considering five cues: fused confidence, $\mathrm { { t o p } \mathrm { { - } 1 / \mathrm { { t o p } \mathrm { { - } 2 } } } }$ margin, evidence purity, cross-source agreement, and observation support. Let $\alpha _ { i }$ and $\beta _ { i }$ denote the top-1 and top-2 highest class confidence, respectively. The t $\mathrm { { \ o p { - } 1 / t o p { - } 2 } }$ margin, defined as $\Delta _ { i } = \alpha _ { i } - \beta _ { i }$ , quantifies the separability between the dominant class and its closest competitor.

The reliability score is computed as a weighted combination of these normalized cues:

$$
r _ { i } = [ O _ { i } \left( 0 . 3 5 \alpha _ { i } + 0 . 2 5 M _ { i } + 0 . 2 0 Q _ { i } + 0 . 2 0 A _ { i } \right) ] ^ { \gamma } .\tag{16}
$$

In this formulation, $O _ { i } = \log ( 1 + | \mathcal { V } _ { i } | ) / \log ( 1 + \kappa )$ measures the observation support, with the scale factor κ set to 6 by default; $M _ { i } = \Delta _ { i } / \lambda _ { \Delta }$ is the normalized margin score; $Q _ { i } ~ = ~ \alpha _ { i } / ( \sum _ { c } s _ { i , c }$ measures the purity of the dominant evidence; $A _ { i } = | \{ \nu \in \mathcal { V } _ { i } : \hat { y } _ { i , \nu } = \tilde { y } _ { i } \} | / ( | \mathcal { V } _ { i } | )$ measures multi-view agreement, $\hat { y } _ { i , \nu } = \arg \operatorname* { m a x } _ { c } e _ { i , \nu , c }$ is view-level prediction; and $\gamma$ is a sharpening exponent set to 1.35 in this implementation. Unknown predictions always receive $r _ { i } = 0$ . The output of this stage is thus a reliability-aware pseudo-labeled point cloud such that:

$$
\tilde { \mathcal { P } } _ { t } = \{ ( p _ { i } , \tilde { y } _ { i } , r _ { i } ) \} _ { i = 1 } ^ { N } ,\tag{17}
$$

where $\tilde { y } _ { i } \in \mathcal { C } _ { t } \cup \{ c _ { \mathrm { u n k } } \}$ and $r _ { i } \in [ 0 , 1 ]$ . This stage is not intended to produce the final segmentation result. Instead, it converts open-vocabulary VLMs masks into reliable-aware 3D semantic seeds to facilitate subsequent adaptation and cluster-level propagation.

## 3.3 Learnable Parameter Adaptation with Diferentiated Self-Supervised Learning

Although VLMs provide open-vocabulary semantic evidence, they remain vulnerable to domain shifts. This issue is particularly pronounced in aerial point clouds due to severe modality and viewpoint discrepancies. Direct adaptation of VLMs entails optimizing a huge parameter space, making it an impractical solution. Therefore, we alternatively adopt a 3D segmentation model pre-trained on the source domain to exploit geometric features. Subsequently, we adapt the pre-trained 3D model to the target domain via a lightweight test-time adaptation strategy driven by diferentiated self-supervised learning, where only BN layers are updated to capture domain-specific statistics (see Fig. 3).

The motivation is that batch normalization layers are highly sensitive to domain statistics (Wang et al. 2021, 2025). When the target distribution difers from the source distribution, the normalization statistics of the source domain can miscalibrate intermediate features. Updating only the afine parameters of BN provides a compact adaptation space, reducing the risk of overfitting noisy pseudo labels while efectively absorbing target-domain statistics. Formally, the learnable test-time parameters are as follows.

$$
\Theta _ { \mathrm { T T A } } = \{ \gamma _ { l } , \beta _ { l } \} _ { l \in \mathcal { L } _ { \mathrm { B N } } } ,\tag{18}
$$

where $\gamma _ { l }$ and $\beta _ { l }$ denote the afine scale and shift of the l-th BN layer. All other parameters in the source model remain frozen during inference.

During inference, the adapted model generates source-space logits and decoder features for a given target batch. Since the source classifier is still defined over $\mathcal { C } _ { s } .$ COSTA does not attempt to directly classify new target classes in $\mathcal { C } _ { t }$ . Instead, we leverage the VLM-derived pseudo-labels to partition target points into in-distribution (ID) and out-of-distribution (OOD) sets. The assignment relies on the overlap between the target classes $\mathcal { C } _ { t }$ and the source classes $\mathcal { C } _ { s } \mathrm { : }$ a target category $c \in { \mathcal { C } } _ { t }$ is regarded as ID if it also appears in $\mathcal { C } _ { s }$ , and as OOD otherwise. Formally,

$$
\begin{array} { r l r } { \mathcal { C } _ { \mathrm { i d } } = \mathcal { C } _ { t } \cap \mathcal { C } _ { s } , } & { { } \ } & { \mathcal { C } _ { \mathrm { o o d } } = ( \mathcal { C } _ { t } \setminus \mathcal { C } _ { s } ) \cup \{ c _ { \mathrm { u n k } } \} . } \end{array}\tag{19}
$$

Based on the pseudo label $\tilde { y } _ { i } ,$ , the target points are partitioned as pseudo-ID and pseudo-OOD samples.

$$
{ \mathcal { T } } _ { \mathrm { i d } } = \{ i \mid { \tilde { y } } _ { i } \in { \mathcal { C } } _ { \mathrm { i d } } \} , \quad { \mathcal { T } } _ { \mathrm { o o d } } = \{ i \mid { \tilde { y } } _ { i } \in { \mathcal { C } } _ { \mathrm { o o d } } \} .\tag{20}
$$

![](images/e7fd9ea6a49a53928c951689162bbf402b418f5ea867ea328ff5daac4c1a4b93.jpg)  
Fig. 3 Illustration of BN TTA with diferentiated self-supervised learning. For each target batch, only the afine parameters $\{ \gamma _ { l } , \beta _ { l } \} _ { l \in \mathcal { L } _ { \mathrm { B N } } }$ of the BN layers are optimized, while all remaining network parameters are frozen. Reliable pseudo-ID points are encouraged to produce confident source-space predictions through entropy minimization, whereas pseudo-OOD points are prevented from being over-confidently assigned to source classes through entropy maximization. A diversity regularization term further prevents class collapse and preserves batch-level discriminability.

This design makes the open-set assignment fully controllable: by specifying $\mathcal { C } _ { t }$ , the user defines the ID/OOD split through the intersection with $\mathcal { C } _ { s }$

Inspired by Gao et al. (2024), COSTA follows the principle of unified entropy optimization for open-set test-time adaptation. Let $\mathbf { z } _ { i }$ be the source-space logit of point $p _ { i }$ , and let

$$
p _ { i } ( c ) = \frac { \exp ( z _ { i } ^ { c } ) } { \sum _ { c \in \mathcal { C } _ { s } } \exp ( z _ { i } ^ { c } ) } , \quad c \in \mathcal { C } _ { s } ,\tag{21}
$$

denote the source classifier probability. For pseudo-ID points, COSTA minimizes prediction entropy to encourage confident decisions for source-explainable semantics:

$$
\mathcal { L } _ { \mathrm { i d } } = \frac { 1 } { \vert \mathcal { Z } _ { \mathrm { i d } } \vert } \sum _ { i \in \mathcal { T } _ { \mathrm { i d } } } H ( p _ { i } ) , \quad H ( p _ { i } ) = - \sum _ { c \in \mathcal { C } _ { s } } p _ { i } ( c ) \log p _ { i } ( c ) .\tag{22}
$$

This term sharpens the source classifier response on target points that align with the source label space, thereby calibrating the target-domain features.

Conversely, for pseudo-OOD points, the objective is inverted. Since these points correspond to novel or target-specific semantics, forcing them into the source classes would destroy the model confidence. COSTA therefore maximizes their entropy over

the closed-set source classifier:

$$
\mathcal { L } _ { \mathrm { o o d } } = \frac { 1 } { | \mathcal { T } _ { \mathrm { o o d } } | } \sum _ { i \in \mathcal { T } _ { \mathrm { o o d } } } H ( p _ { i } ) .\tag{23}
$$

By subtracting this OOD entropy term in the final minimization objective, the model is penalized for producing over-confident source-class predictions on target-specific classes. Crucially, unlike conventional methods that infer OOD samples solely from intrinsic classifier uncertainty, our OOD partition is explicitly guided by VLMs-derived open-vocabulary pseudo-labels.

Furthermore, COSTA incorporates a diversity regularization term to circumvent degenerate solutions. Entropy minimization may collapse most target points into a few dominant source classes. To preserve batch-level discriminability, the marginal prediction distribution is encouraged to remain diverse. Given the batch-averaged source prediction,

$$
\bar { p } ( c ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } p _ { i } ( c ) ,\tag{24}
$$

COSTA minimizes the negative entropy of this marginal distribution:

$$
\mathcal { L } _ { \mathrm { d i v } } = - \sum _ { c \in \mathcal { C } _ { s } } \bar { p } ( c ) \log \left( \bar { p } ( c ) \right) ,\tag{25}
$$

Minimizing ${ \mathcal { L } } _ { \mathrm { d i v } }$ is equivalent to maximizing the marginal entropy, which prevents class collapse and ensures separable adapted features.

The final test-time adaptation objective is formulated as

$$
\mathcal { L } _ { \mathrm { T T A } } = \mathcal { L } _ { \mathrm { i d } } - \lambda _ { \mathrm { o o d } } \mathcal { L } _ { \mathrm { o o d } } + \lambda _ { \mathrm { d i v } } \mathcal { L } _ { \mathrm { d i v } } .\tag{26}
$$

Only $\Theta _ { \mathrm { T T A } }$ in Eq. (18) is optimized by back-propagating Eq. (26). Note that this stage is not intended to generate the final open-set labels. The classifier head still operates in the source label space $\mathcal { C } _ { s } .$ . It plays a key role in the reshaping of the target-domain feature distribution so that semantically analogous points cluster compactly in the decoder feature space. The final open-set predictions are subsequently produced by the cluster-centric semantic propagation detailed in the next section.

## 3.4 Cluster-Centric Open-Set Semantic Propagation

This stage leverages the cluster-centric semantic propagation mechanism to integrate the domain generalization strengths of the TTA model with the open-vocabulary segmentation capability of VLMs. As shown in Fig. 1, we empirically found that targetdomain points that share similar semantics tend to form compact clusters within the adapted decoder feature space following proven test-time adaptation. Intuitively, this property provides a natural mechanism to mitigate the noise inherent in VLMsderived pseudo-labels. Rather than independently assigning labels to individual points, COSTA conceptualizes each cluster as a local semantic unit and performs semantic propagation at the cluster level.

Given the adapted decoder feature $\mathbf { h } _ { i }$ of point $p _ { i }$ , COSTA first normalizes the features followed by K-Means clustering:

$$
\bar { \mathbf { h } } _ { i } = \frac { \mathbf { h } _ { i } } { \| \mathbf { h } _ { i } \| _ { 2 } } , \quad \{ \pmb { \mu } _ { j } \} _ { j = 1 } ^ { J } , a _ { i } = \mathrm { K M e a n s } ( \{ \bar { \mathbf { h } } _ { i } \} _ { i = 1 } ^ { N } ) ,\tag{27}
$$

where $\mu _ { j }$ represents the center of the $j \mathrm { - t h }$ cluster and $a _ { i }$ indicates the cluster assignment of the point $p _ { i }$ . The number of clusters is capped by the batch size, e.g., $J = \operatorname* { m i n } ( 1 0 0 , N )$ . Each cluster is expected to contain points with similar semantics.

Given the inherent noise in pseudo-labels generated by VLMs, we avoid using all points within a cluster for equal propagation. Instead, it constructs a reliability-filtered semantic bank inside each cluster. $\mathrm { A }$ point is admitted to the cluster bank $B _ { j }$ only if its pseudo-label is valid and its reliability score exceeds a threshold $\rho _ { c } .$ . These reliability score is derived from the pseudo-labeling stage (Sec. 3.2.2). Formally, for $j \cdot$ -th cluster $C _ { j } .$ , this filtering could be formulated as

$$
B _ { j } = \{ p _ { i } \in C _ { j } \mid \tilde { y } _ { i } \in \mathcal { C } _ { t } , \ r _ { i } \geq \rho _ { c } \} ,\tag{28}
$$

This filtering is important because various classes may exhibit varying response quality of VLMs, projection visibility, and spatial compactness.

During inference, we propagate the prediction of the cluster-level weighted voting to all points in each cluster. Unlike nearest-center propagation, we aggregate all reliable seeds in $B _ { j }$ within the group using the reliability-weighted strategy, which ensures that high-confidence seeds contribute more significantly than low-confidence seeds. The semantic prediction of each cluster is formulated as

$$
\ell _ { j } = \arg \operatorname* { m a x } _ { c \in \mathcal { C } _ { t } } \sum _ { i \in \mathcal { B } _ { j } } r _ { i } \mathbb { I } [ \tilde { y } _ { i } = c ] .\tag{29}
$$

where I denotes the indicator function. If $B _ { j }$ is empty, the cluster is temporarily assigned to the unknown label $c _ { \mathrm { u n k } }$ . Otherwise, the class that accumulates the evidence with the highest reliability is designated as the cluster label. The final point-wise prediction is obtained by propagating the cluster label to all constituent points in the cluster.

## 4 Experiments

In this section, we first describe our experimental setup for OSTTA (Sec. 4.1). Then, we demonstrate the generality of our approach through cross-dataset evaluation (Sec. 4.2). Finally, we discuss and ablate our methods (Sec. 4.3).

## 4.1 Evaluation Protocol of OSTTA

## 4.1.1 Benchmark

We set up an evaluation protocol to simulate the conditions that occur when a pretrained aerial point cloud segmentation model is deployed to unseen scenes. In such applications, the source data is collected in a limited set of regions and annotated with source domain-specific classes. The deployed model is expected to operate on previously unseen scenes (cross-geographic and cross-sensor) without accessing source data and target annotations, which is reasonable in practice when source data cannot be redistributed or stored due to privacy, licensing, or storage constraints, and rapid deployment to a new target domain is required. Importantly, unlike closed-set TTA protocols that retain only shared classes or merge similar fine-grained classes for evaluation, we retain the native target classes to simulate the segmentation demanded by users. To study OSTTA in such a setting, we base our experiment setup on 4 public aerial point cloud datasets, including DALES (Varney et al. 2020), SUM (Gao et al. 2021), Hessigheim 3D (K¨olle et al. 2021), and CUS3D (Gao et al. 2024). The characteristics of these datasets are described below.

DALES. Dayton Annotated Laser Earth Scan (DALES) is a large-scale benchmark dataset for 3D scene understanding. It consists of 40 airborne LiDAR tiles, each covering approximately 500m × 500m, collected in urban and suburban areas of the city of Surrey in British Columbia, Canada. The dataset provides dense per-point semantic labels across 9 classes.

SUM. Semantic Urban Mesh (SUM) is a photogrammetric benchmark dataset designed for semantic segmentation of urban environments, covering approximately 4 km² in Helsinki, Finland. The dataset is divided into 64 equally sized areas with a size of 250m x 250m and encompasses six annotated classes.

Hessigheim 3D. Hessigheim 3D (H3D) is a high-density airborne LiDAR point cloud dataset of approximately 800 points/m2. The dataset is captured by a Riegl VUX-1LR Scanner integrated with RGB images with a ground sampling distance (GSD) of 2 to 3 centimeters in the village of Hessigheim, Germany. A total of 11 fine-grained classes have been manually annotated. Here, we use test data collected in March 2018 for a standardized comparative evaluation.

CUS3D. CUS3D is a comprehensive urban-scale semantic segmentation benchmark dataset built upon large-scale 3D reconstruction using drone aerial imagery. It provides both color point clouds and textured mesh 3D data. The dataset covers approximately 2.85 km² of urban, suburban, and rural scenes, containing over 152 million 3D points. For mixed urban and rural scenes, each 3D point is annotated with one of 10 semantic classes.

We therefore use DALES as the labeled source domain and construct 3 OSTTA benchmarks, namely DALES-SUM, DALES-H3D, and DALES-CUS3D. The detailed semantic relationships of 3 constructed OSTTA benchmarks are provided in Table. 1. These benchmarks cover two representative semantic shifts: unseen classes and crossgranularity classes. Specifically, DALES-SUM mainly evaluates open-set adaptation to newly introduced target classes, where water and boat have no reliable counterparts in the source. DALES-H3D focuses on semantic granularity discrepancies, where most target classes are finer-grained or structurally redefined variants of source-domain semantics, such as vegetation-related and building-related subclasses. DALES-CUS3D provides a more comprehensive setting that simultaneously involves both semantic shift patterns.

Table 1 Semantic relationship of the constructed OSTTA benchmarks between the target domain and the source domain.
<table><tr><td>Benchmark</td><td>Known classes</td><td>Cross-granularity classes</td><td>Unseen classes</td><td></td></tr><tr><td>DALES-SUM</td><td>terrain; building; car</td><td>high vegetation</td><td>water; boat 1</td><td></td></tr><tr><td>DALES-H3D</td><td>vehicle</td><td>low vegetation; shrub; tree; impervious surface; soil; roof; facade; chimney; vertical surface</td><td></td><td></td></tr><tr><td>DALES-CUS3D building; car;</td><td></td><td>ground; road; grass; high vegetation</td><td>playground; water; farmland; building site</td><td></td></tr></table>

Known classes denote target classes that directly overlap with the source classes. Crossgranularity classes denote target classes that are finer or coarser semantic variants of the source classes. Unseen classes denote target classes without reliable source-domain counterparts.

## 4.1.2 Metrics

To comprehensively evaluate the performance of our approach, we evaluate semantic segmentation performance using per-class intersection over inion (IoU) and three overall metrics: overall accuracy (OA), mean intersection over union (mIoU), and mean class accuracy (mAcc). Let C denote the set of evaluated classes. For each class $c \in { \mathcal { C } }$ we compute the true positives $T P _ { c } .$ , false positives $F P _ { c } .$ , and false negatives $F N _ { c }$ from the prediction results. The IoU of class c is defined as

$$
\mathrm { I o U } _ { c } = \frac { T P _ { c } } { T P _ { c } + F P _ { c } + F N _ { c } } .\tag{30}
$$

The mean IoU is then obtained by averaging over all evaluated classes:

$$
\mathrm { m I o U } = \frac { 1 } { | \mathcal { C } | } \sum _ { c \in \mathcal { C } } \mathrm { I o U } _ { c } .\tag{31}
$$

In addition, we report overall accuracy, which measures the proportion of correctly classified points among all valid points:

$$
\mathrm { O A } = \frac { \sum _ { c \in \mathcal { C } } T P _ { c } } { \sum c \in \mathcal { C } ( T P _ { c } + F N _ { c } ) } .\tag{32}
$$

To assess the average accuracy across classes, especially under class imbalance, we further report mean class accuracy:

$$
\mathrm { m A c c } = \frac { 1 } { | \mathcal { C } | } \sum _ { c \in \mathcal { C } } \frac { T P _ { c } } { T P _ { c } + F N _ { c } } .\tag{33}
$$

## 4.1.3 Implementation details

We note that all experiments are run on NVIDIA RTX PRO 6000 GPUs and Linux OS with Ubuntu 22.04, AMD EPYC 9V74 CPU. We use KPConv (Thomas et al. 2019) as the lightweight 3D segmentation model and a frozen Sa2VA (Yuan et al. 2025) as the 2D VLM. The 3D model is pre-trained solely on the source domain, DALES. During training, we standardize the grid size to 0.25 m to subsample the original point cloud and improve computational eficiency, and set the overlapping spherical subcloud sampling to 15 m for batch creation. Once pretraining on the source domain was complete, we deployed the pretrained model for semantic segmentation on the target test dataset during the inference stage, without accessing the source data and target annotations. Due to scale diferences between datasets, we set overlap spherical sampling radius of 20 m, 15 m, and 15 m for SUM, H3D, and CUS3D, respectively, during the inference stage. All models are optimized by Adam optimizer with a learning rate of $1 0 ^ { - 4 }$ and are implemented using the PyTorch framework.

For oblique virtual-camera projection, we set $K _ { l } = 6$ and place 3 cameras around each anchor, yielding $6 ^ { 2 } \times 3 = 1 0 8$ candidate oblique virtual-camera views per point cloud. For BEV projection, we set $s _ { g } \ = \ 0 . 0 5 , G \ = \ 5 0 \mathrm { m } , \Delta _ { g } \ = \ 2 5$ for SUM and $G = 2 5 \mathrm { m } , \Delta _ { g } = 1 2 . 5$ for H3D/CUS3D. For pseudo-labeling, Sa2VA was queried using the prompt template “Please segment all {category}.” Mask responses below $\tau _ { 0 } =$ 0.1 were excluded. During cluster-centric propagation, every spherical sub-cloud was partitioned into 200 clusters using k-means++ with a random seed of 0. In addition, $\rho _ { c }$ in Eq. (28) is set as 0.4.

## 4.2 Results

## 4.2.1 Quantitative results

In this section, we evaluate our approach under the OSTTA setup using the evaluation protocol described above. To comprehensively evaluate performance, we compare it with three types of baselines. As summarized in Table 2, the compared methods follow diferent supervision and adaptation protocols. Fully-supervised methods are trained with target-domain annotations, open-vocabulary distillation methods require target-domain ofline training, and multi-dataset pretraining methods rely on massive annotated datasets for foundation model construction. In contrast, COSTA follows a stricter open-set test-time adaptation setting: the source-pretrained model is directly deployed on unlabeled target data, without source-data replay, target annotations, or ofline retraining. Therefore, the results should be interpreted not only by absolute performance, but also by the amount of supervision and training cost required by each setting. We provide the quantitative results on the three constructed OSTTA benchmarks in Table 3–5.

DALES-SUM. The DALES-SUM benchmark mainly evaluates adaptation performance to unseen target classes. As shown in Table 3, COSTA achieves 70.09% mIoU and 88.52% OA under the strict open-set TTA setting, which outperforms most baselines. Among fully-supervised methods, Point Transformer V3 achieves the best performance with 74.00% mIoU and 94.70% OA, benefiting from its transformer-based architecture and target-domain supervised training. In contrast, COSTA obtains a comparable mIoU without any target annotation and ofline re-training. In particular, our method surpasses representative supervised baselines such as SparseConv and KPConv by 27.49% and 1.29% mIoU, respectively. Compared with open-vocabulary distillation methods, COSTA yields a performance on par with OpenUrban3D, with merely a 5.31% mIoU gap, while requiring no target-domain distillation and ofline re-training. COSTA outperforms OpenScene by 28.99% mIoU and 13.92% OA. We attribute this to the limitation that direct open-vocabulary feature transfer is insufficient under domain and semantic shifts. For multi-dataset pretraining methods, it should be noted that this setting follows a closed-set evaluation protocol, where models are pretrained on multiple datasets including the train-set of target dataset before testing. Even under this favorable setting, COSTA is only 1.61% mIoU lower than previous SoTA CitySeg-FM and exceeds Sonata by 8.89% mIoU and 18.22% OA. For unseen target classes, COSTA improves boat from 61.32% to 77.99% and water from 85.83% to 88.16% over Sa2VA + hybrid, and further outperforms OpenUrban3D on boat by 9.89% IoU. The results demonstrate that our approach can efectively alleviate domain shift in known classes and successfully discover unseen target-specific classes beyond the source.

DALES-H3D. The DALES-H3D benchmark introduces the challenge of crossgranularity semantic shifts, where most target classes are not strictly unseen but correspond to finer-grained or structurally redefined variants of source-domain semantics, such as vegetation-related subclasses and building-related subclasses. As shown in Table 4, COSTA achieves 34.13% mIoU and 70.68% OA. Compared with fullysupervised methods, our approach operates under the OSTTA setting, and thus still exhibits a clear gap from strong fully-supervised models on this fine-grained benchmark. Nevertheless, COSTA exceeds PointNet++ by 4.23% mIoU and achieves competitive OA compared to early supervised baselines such as PointNet and Point-Net++. Compared with multi-dataset pretraining methods, COSTA outperforms Sonata by 0.33% mIoU and 7.58% OA, and surpasses PTv3 + PPT by 3.48% OA, although it remains 17.77% mIoU lower than SoTA CitySeg-FM. This indicates that large-scale multi-dataset pretraining is still advantageous for dense fine-grained target classes, while COSTA provides a training-free alternative that can be deployed directly at test time. At the class level, COSTA performs reasonably well on low vegetation, impervious surface, vehicle, roof, tree, and especially soil/gravel, where it achieves 51.58% IoU and even surpasses the best fully-supervised result by 8.13%. However, its performance remains limited on highly fine-grained, sparse, and small scale classes such as urban furniture and chimney. These results suggest that crossgranularity OSTTA is substantially more challenging than unseen-class adaptation because it requires not only recognizing target-specific semantics, but also separating subtle semantic subdivisions that are absent in the source. Furthermore, poor image rendering quality is also a significant constraint that hinders tiny object recognition.

Table 2 Comparison of diferent segmentation and adaptation settings.
<table><tr><td>Setting</td><td>Pretrained model Target data Offline training</td><td></td><td></td></tr><tr><td>Fully-supervised method</td><td>×</td><td> $x ^ { t } , y ^ { t }$ </td><td>√</td></tr><tr><td>Open-vocabulary distillation</td><td>√</td><td> $x ^ { t }$ </td><td>√</td></tr><tr><td>Multi-dataset pretraining</td><td>√</td><td> $x ^ { t } , y ^ { t }$ </td><td>√</td></tr><tr><td>Open-set test-time adaptation</td><td>√</td><td>2e</td><td>×</td></tr></table>

DALES-CUS3D. The DALES-CUS3D benchmark provides a more comprehensive OSTTA setting that jointly involves unseen classes and cross-granularity semantic shifts. As shown in Table 5, COSTA achieves 49.30% mIoU and 77.98% OA under the open-set TTA setting. Among all fully-supervised methods, KPConv obtains the best overall performance with 59.72% mIoU and 89.42% OA with dense targetdomain supervision. While there is 10.42% mIoU gap against the best fully-supervised method (KPConv) due to the absence of target annotations and ofline training, our approach outperforms representative supervised baselines like SPGraph by 18.86% mIoU, and achieves competitive performance compared with RandLA-Net and Point-Net++. Compared with the open-vocabulary baseline Sa2VA, COSTA improves mIoU from 39.28% to 49.30% and OA from 61.53% to 77.98%. This indicates that directly transferring VLM-derived pseudo labels is insuficient for reliable 3D segmentation due to significant domain shift, while our approach can substantially improve 3D semantic consistency without additional model retraining. At the class level, COSTA brings clear improvements over Sa2VA on both unseen and cross-granularity classes. For unseen classes, it improves playground from 6.24% to 32.53%, farmland from 53.92% to 63.06%, building site from 3.68% to 11.47%, and water from 78.06% to 79.83%. For cross-granularity classes, it improves high vegetation from 53.70% to 74.99% and grass from 33.71% to 38.87%, while maintaining comparable performance on road. Notably, COSTA also achieves strong results on car and water, reaching 53.58% and 79.83% IoU, respectively. These results demonstrate that COSTA can handle mixed semantic shifts more efectively than direct open-vocabulary methods, especially for semantically well-defined and large-scale objects.

## 4.2.2 Qualitative results

In Fig. 4–6, we show several visual examples for COSTA in the three benchmarks. As can be seen, COSTA performs well in challenging cases with known classes and unseen classes in various scenes. For example, in the DALES-SUM benchmark, our method segments the known classes under covariate shifts, such as terrain, high Veg. and building. Moreover, COSTA segments the unseen classes under semantic shifts, such as water and boat. This is especially true for water that is relatively large in scale and has clearly defined categories. Similar observations could also be found in the DALES-H3D (see Fig. 5) and DALES-CUS3D (see Fig. 6) benchmark. Our method successfully segments impervious surfaces, roofs, farmland, etc.

Table 3 Quantitative comparison on the DALES-SUM benchmark. Target classes are divided into known, cross-granularity, and unseen classes according to their semantic relationship with the source.
<table><tr><td>Method</td><td>mIoU</td><td>OA</td><td>Terrain</td><td>High Veg.</td><td>Building</td><td>Water</td><td>Vehicle</td><td>Boat</td></tr><tr><td colspan="9">Fully-supervised</td></tr><tr><td>PointNet (Qi et al. 2017)</td><td>23.70</td><td>80.70</td><td>67.90</td><td>89.50</td><td>80.00</td><td>0.00</td><td>0.00</td><td>3.90</td></tr><tr><td>PointNet++ (Qi et al. 2017)</td><td>32.90</td><td>84.30</td><td>72.40</td><td>94.20</td><td>84.70</td><td>2.70</td><td>2.00</td><td>25.70</td></tr><tr><td>SPGraph (Landrieu and Simonovsky 2018)</td><td>37.20</td><td>76.90</td><td>69.90</td><td>94.50</td><td>88.80</td><td>32.80</td><td>12.50</td><td>15.70</td></tr><tr><td>SparseConv (Graham et al. 2018)</td><td>42.60</td><td>85.20</td><td>74.10</td><td>97.90</td><td>94.20</td><td>63.30</td><td>7.50</td><td>24.20</td></tr><tr><td>MinkUnet (Choy et al. 2019)</td><td>70.00</td><td>93.30</td><td>84.40</td><td>92.00</td><td>92.90</td><td>76.80</td><td>61.60</td><td>12.60</td></tr><tr><td>KPConv (Thomas et al. 2019)</td><td>68.80</td><td>94.20</td><td>86.50</td><td>88.40</td><td>92.70</td><td>77.70</td><td>54.30</td><td>13.30</td></tr><tr><td>RandLA-Net (Hu et al. 2020)</td><td>38.60</td><td>74.90</td><td>38.90</td><td>59.60</td><td>81.50</td><td>27.70</td><td>22.00</td><td>2.10</td></tr><tr><td>Point Transformer V2 (Wu et al. 2022)</td><td>65.40</td><td>93.10</td><td>82.80</td><td>89.60</td><td>93.60</td><td>87.40</td><td>19.80</td><td>18.80</td></tr><tr><td>Point Transformer V3 (Wu et al. 2024)</td><td>74.00</td><td>94.70</td><td>87.30</td><td>92.30</td><td>94.20</td><td>88.50</td><td>49.40</td><td>32.40</td></tr><tr><td colspan="9">Open-vocabulary distillation</td></tr><tr><td>OpenScene (Peng et al. 2023)</td><td>41.10</td><td>74.60</td><td>35.60</td><td>54.10</td><td>73.70</td><td>53.90</td><td>0.10</td><td>28.80</td></tr><tr><td>PLA (Ding et al. 2023)</td><td>4.30</td><td>26.70</td><td>24.60</td><td>0.00</td><td>0.20</td><td>0.00</td><td>0.90</td><td>0.00</td></tr><tr><td>RegionPLC (Yang et al. 2024)</td><td>8.10</td><td>26.40</td><td>16.20</td><td>7.50</td><td>19.30</td><td>3.10</td><td>0.10</td><td>2.60</td></tr><tr><td>Sa2VA + hybrid (Yuan et al. 2025)</td><td>62.89</td><td>81.07</td><td>63.17</td><td>63.10</td><td>78.19</td><td>85.83</td><td>25.75</td><td>61.32</td></tr><tr><td>OpenUrban3D (Wang et al. 2026)</td><td>75.40</td><td>90.50</td><td>69.30</td><td>83.60</td><td>90.20</td><td>92.70</td><td>48.30</td><td>68.10</td></tr><tr><td colspan="9">Multi-dataset pretraining</td></tr><tr><td>PTv3 + PPT (Wu et al. 2024)</td><td>53.8</td><td>63.7</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Sonata (Wu et al. 2025)</td><td>61.2</td><td>70.3</td><td></td><td></td><td></td><td>1</td><td></td><td></td></tr><tr><td>CitySeg-FM (Xu et al. 2025)</td><td>71.7</td><td>91.7</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">Open-set TTA</td></tr><tr><td>COSTA</td><td>70.09</td><td>88.52</td><td>68.52</td><td>72.45</td><td>86.53</td><td>88.16</td><td>26.90</td><td>77.99</td></tr><tr><td colspan="9">Known classes Cross-granularity classes</td></tr></table>

However, we also observe failure cases. In Fig 5, our method struggles to segment too tiny objects and targets with unclear category definitions. For example, in the DALES-H3D benchmark, the performance of small-scale chimney and urban furniture mixing various types of urban infrastructure is not satisfactory.

## 4.3 Analysis

## 4.3.1 Analysis of hybrid multi-view projection

A prerequisite for good pseudo-labels in VLMs is high-quality multi-view projections. Here, we study the pseudo-label quality of our method with diferent multi-view projection strategies based on the same VLM (Yuan et al. 2025). Compared to the BEV-only projection, the oblique-only projection substantially improves OA, mAcc, and mIoU by 29.90%, 16.63%, and 15.86% in the SUM dataset, while reducing the unknown ratio from 38.83% to 8.26% (Table 6). Similar trends can also be observed in the H3D and CUS3D dataset. The oblique-only projection generally produces higher-quality pseudo labels. This is mainly because oblique virtual-camera views provide provide more comprehensive observations from diverse perspectives, which helps the 2D VLMs generate more complete prediction masks for reference. In contrast, BEV projection only provides a top-down observation. Although it ofers a stable global layout, it tends to lose vertical and side-view information and is more sensitive to top-view occlusion, especially for structures such as building facades, vertical surfaces, and objects partially hidden by higher ones. As a result, BEV-only projection produces a larger proportion of unknown predictions.

<sub>ss-gran</sub>u<sup>larity</sup> <sup>cla</sup>

<sub>no</sub>w<sup>n</sup> <sup>class</sup>

<sub>nsee</sub>n <sup>classe</sup>

<table><tr><td colspan="12">Table 4 Quantitative comparison on the DALES-to-H3D benchmark. Target classes are grouped into known, cross-granularity, and unseen classes according to their semantic relationship with the DALES source label space. Vertical</td></tr><tr><td rowspan="2">Method</td><td rowspan="2">mIoU OA</td><td rowspan="2">Low Veg. Surface</td><td rowspan="2">Imp. Vehicle</td><td rowspan="2">Urban Furn.</td><td rowspan="2"></td><td colspan="7">Soil/ Roof FaçadeShrub Tree</td></tr><tr><td></td><td></td><td></td><td>Gravel</td><td>Surf.</td><td>Chimney</td></tr><tr><td>Fully-supervised</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PointNet (Qi et al. 2017)</td><td>36.64 71.05</td><td>56.00</td><td>49.03</td><td>2.00</td><td>21.06</td><td>69.12</td><td>43.21</td><td>23.02</td><td>83.03</td><td>3.09</td><td>42.97</td><td>1.02</td></tr><tr><td>PointNet++ (Qi et al. 2017)</td><td>29.90 68.50</td><td>64.13</td><td>56.26</td><td>18.85</td><td>7.30</td><td>58.71</td><td>31.37</td><td>16.49</td><td>56.00</td><td>5.09</td><td>12.07</td><td>2.24</td></tr><tr><td>RandLA-Net (Hu et al. 2020)</td><td>57.31 83.99</td><td>69.05</td><td>72.03</td><td>37.26</td><td>43.60</td><td>85.33</td><td>68.61</td><td>45.32</td><td>90.24</td><td>12.96</td><td>63.03</td><td>42.97</td></tr><tr><td>PIIE-DSA-Net (Gao et al. 2022)</td><td>52.4884.20</td><td>78.09</td><td>74.86</td><td>35.51</td><td>22.53</td><td>90.55</td><td>53.00</td><td>31.12</td><td>89.42</td><td>14.37</td><td>49.22</td><td>38.59</td></tr><tr><td>KPConv (Thomas et al. 2019)</td><td>63.2087.69</td><td>79.54</td><td>80.11</td><td>69.72</td><td>46.94</td><td>94.51</td><td>74.21</td><td></td><td>60.32 94.90</td><td>27.09</td><td>67.87</td><td>0.00</td></tr><tr><td>SCN (Kölle et al. 2021)</td><td>64.87 88.42</td><td>85.65</td><td>78.83</td><td>46.53</td><td>39.97</td><td>93.94</td><td>71.07</td><td>52.18</td><td>94.15</td><td>28.90</td><td>64.24</td><td>58.19</td></tr><tr><td>Point Transformer V3 (Wu et al. 2024) 70.08 92.8188.71</td><td></td><td></td><td>85.67</td><td>65.18</td><td>53.9094.20</td><td></td><td>72.83</td><td>51.35</td><td>90.88</td><td>43.45</td><td>66.64</td><td>58.12</td></tr><tr><td colspan="9">Open-vocabulary distillation</td><td></td><td></td><td></td></tr><tr><td>Sa2VA + hybrid (Yuan et al. 2025)</td><td>28.3860.1654.04 56.55</td><td></td><td></td><td>35.24</td><td></td><td></td><td>8.00 71.49 12.04</td><td></td><td></td><td>5.01 31.82 29.93</td><td>7.77</td><td>0.34</td></tr><tr><td colspan="9"></td><td></td><td></td><td></td></tr><tr><td>Multi-dataset pretraining PTv3 + PPT (Wu et al. 2024)</td><td>35.70 67.20</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Sonata (Wu et al. 2025)</td><td>33.8063.10</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CitySeg-FM (Xu et al. 2025)</td><td>51.90 81.20</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">Open-set TTA</td><td rowspan="2"></td><td rowspan="2">0.00</td><td rowspan="2">8.44</td></tr><tr><td>COSTA</td><td>34.13 70.6861.63 61.53</td><td></td><td></td><td>47.54</td><td></td><td>4.94 66.94 12.92</td><td></td><td>2.51 57.41 51.58</td></tr></table>

<sub>s-gran</sub>u<sup>larity</sup> <sup>cla</sup>  
Table5QuantitativecomparisonontheDALES-to-CUS3Dbenchmark.Targetclassesaregroupedintoknown,cross-granularity,andunseen classesaccordingtotheirsemanticrelationshipwiththeDALESsourcelabelspace.
<table><tr><td>Method</td><td></td><td></td><td></td><td></td><td>mIoU OA mAccGroundBuild.Road</td><td>Veg.</td><td>High Water</td><td>Car</td><td></td><td></td><td>Play. Farm.Grass</td><td>Building Site</td></tr><tr><td colspan="10">Fully-supervised</td></tr><tr><td></td><td></td><td>42.54 74.57 94.91</td><td></td><td></td><td></td><td>78.34 73.32</td><td>25.22</td><td>20.35</td><td>8.76</td><td>8.02</td><td>39.11</td><td>16.66</td></tr><tr><td>PointNet (Qi et al. 2017) PointNet++ (Qi et al. 2017)</td><td>55.16</td><td>84.52</td><td>96.90</td><td>78.36 25.78</td><td>65.81 80.36</td><td>57.77</td><td>86.18</td><td>57.52 36.13</td><td>11.38</td><td>40.88</td><td>40.17</td><td>13.28</td></tr><tr><td>RandLA-Net (Hu et al. 2020)</td><td>52.98</td><td>77.12</td><td>95.42</td><td>15.29</td><td>76.18</td><td>49.52</td><td>78.42 62.30</td><td>49.89</td><td>35.91</td><td>42.11</td><td>39.74</td><td>29.86</td></tr><tr><td>KPConv (Thomas et al. 2019)</td><td></td><td>59.72 89.42 97.88</td><td></td><td>25.73</td><td>82.89</td><td>48.69 89.64</td><td>48.43</td><td>44.69</td><td>25.72</td><td>41.86</td><td>44.84</td><td>15.50</td></tr><tr><td>SPGraph (Landrieu and Simonovsky 2018)</td><td></td><td>30.44 67.23</td><td>40.73</td><td>12.05</td><td>63.40</td><td>30.56</td><td>63.40 31.71</td><td>4.40</td><td>7.65</td><td>36.66</td><td>31.36</td><td>28.84</td></tr><tr><td>SQN (Hu et al. 2022)</td><td>56.82</td><td>78.19</td><td>95.64</td><td>30.65</td><td>79.08</td><td>60.22</td><td>79.08 59.21</td><td></td><td>55.4634.77</td><td>39.88</td><td>45.54</td><td>30.73</td></tr><tr><td>Open-vocabulary distillation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Sa2VA+hybrid (Yuan et al. 2025)</td><td></td><td></td><td>3.68</td></tr><tr><td></td><td>39.28 61.53 57.23</td><td></td><td></td><td>8.02</td><td></td><td>58.74 57.5853.70 78.06 39.13 6.24</td><td></td><td></td><td></td><td></td><td>53.92 33.71</td><td></td></tr><tr><td>Open-set test-time adaptation</td><td>49.30 77.98 61.40</td><td></td><td></td><td>7.81</td><td>72.78 58.0774.9979.83 53.5832.53 63.06 38.87</td><td></td><td></td><td></td><td></td><td></td><td></td><td>11.47</td></tr><tr><td colspan="10">COSTA</td></tr></table>

<sub>no</sub>w<sup>n</sup> <sup>class</sup>

![](images/d2e6ffe1c8c36ec300a71c30b01e06c9c7ca0b95d0974f00b1632fb2af88dccc.jpg)  
Fig. 4 Classification result of SUM dataset with a model adapted from DALES dataset.

![](images/9b83c31705a4bd0bd065e74c584b39f7ebf45c3bc5cd84702d7fff00e6fe5277.jpg)  
Fig. 5 Classification result of H3D dataset with a model adapted from DALES dataset.

![](images/f1d9cf06de4121444294da09d2eca31bc2b1991cb50e243fed3aad0376f9f6d5.jpg)  
Fig. 6 Classification result of CUS3D dataset with a model adapted from DALES dataset.

The best performance is achieved when fusing both projection strategies, with a performance gain of approx. +4%, +1%, +6% mIoU across SUM, H3D and CUS3D. Similar trends are observed for the mAcc and unknown ratio (Table 6). The results demonstrate the efectiveness of the proposed hybrid multi-view evidence accumulation fusion strategy. By integrating complementary evidence from BEV and oblique virtualcamera projections, the fusion strategy can suppress unreliable predictions naturally, improve pseudo-label coverage, and generate more accurate pseudo labels.

The detailed per-class results (see Table 7–Table 9) provide further insight into the complementarity between BEV and oblique projections. BEV projection is beneficial for some compact or top-view distinguishable classes. For example, it achieves the best IoU for vehicle on SUM and H3D, outperforming the oblique-only by approx. 10%. This is mainly because BEV provides a geometrically stable overhead footprint, which can efectively preserve the spatial layout of small objects that are visible from the top. However, BEV projection is less efective for classes that require oblique or sideview information. For example, its IoU on SUM building is approximately 30% lower than the oblique-only, and its IoU on H3D facade and vertical surface is also limited, indicating that top-down projection alone cannot suficiently capture vertical structures. In contrast, oblique projection is more efective for large-scale and spatially extended categories, with a performance gain of approx. +20% on water of SUM.

Table 6 Overall pseudo-label evaluation results under diferent projection modes. The best OA, mAcc, and mIoU within each dataset are highlighted in bold, and the lowest unknown ratio is also highlighted.
<table><tr><td>Dataset</td><td>Mode</td><td>OA (%)</td><td>mAcc (%)</td><td>mIoU (%)</td><td>Unknown ratio (%)</td></tr><tr><td rowspan="3">SUM</td><td>BEV-only</td><td>48.82</td><td>51.20</td><td>43.06</td><td>38.83</td></tr><tr><td>oblique-only</td><td>78.72</td><td>67.83</td><td>58.92</td><td>8.26</td></tr><tr><td>Fused</td><td>81.07</td><td>73.08</td><td>62.89</td><td>5.90</td></tr><tr><td rowspan="3">H3D</td><td>BEV-only</td><td>48.94</td><td>33.18</td><td>24.17</td><td>20.61</td></tr><tr><td>oblique-only</td><td>60.84</td><td>38.41</td><td>27.22</td><td>14.80</td></tr><tr><td>Fused</td><td>60.16</td><td>42.20</td><td>28.38</td><td>7.27</td></tr><tr><td rowspan="3">CUS3D</td><td>BEV-only</td><td>46.87</td><td>50.97</td><td>33.12</td><td>26.72</td></tr><tr><td>oblique-only</td><td>50.91</td><td>49.46</td><td>33.51</td><td>9.09</td></tr><tr><td>Fused</td><td>61.53</td><td>57.23</td><td>39.28</td><td>5.15</td></tr></table>

Particularly, the fused strategy combines the advantages of both projection modes and achieves more balanced performance across categories. Compared to single projection strategies, the fused setting obtains the best IoU on ground, high vegetation, building, and boat, while maintaining competitive performance on water and vehicle on the SUM dataset. Similar results were observed on both the H3D and CUS3D datasets. The results confirm that BEV and oblique projections provide complementary semantic evidence: BEV captures stable global layout and topview object footprints, whereas oblique projections ofer diverse viewpoint-dependent details. Also, the multi-view evidence accumulation fusion strategy efectively combines the advantages of both projection. Overall, consistent improvements across various metrics demonstrate that the proposed hybrid projection and multi-view evidence accumulation fusion strategy can efectively enhance pseudo-label quality.

## 4.3.2 Analysis of reliability score

The reliability weight provides an efective criterion for VLMs pseudo-label selection. Since VLM operates under domain shifts, the reliability weight plays a crucial role in distinguishing trustworthy semantic seeds from unreliable VLM predictions. To assess whether the proposed reliability weight can accurately reflect pseudo-label quality, we analyze the precision and recall of selected pseudo labels under diferent reliability thresholds on SUM, H3D, and CUS3D. As shown in Fig. 7, a clear precision–recall trade-of can be observed across all three benchmarks. On the SUM dataset, increasing the threshold consistently improves precision from 86% to around 99%, while recall decreases from 94% to a small fraction of points. A similar trend is observed on H3D and CUS3D. These consistent trends across datasets indicate that larger reliability weights are correlated with higher pseudo-label correctness, confirming that the proposed reliability score provides an efective quality indicator for VLM-derived semantics.

Table 7 Per-class IoU comparison of diferent projection modes on SUM. The best result for each class is highlighted in bold.
<table><tr><td>Mode</td><td>Terrain</td><td>High Vegetation</td><td>Building</td><td>Water</td><td>Vehicle</td><td>Boat</td></tr><tr><td>BEV-only</td><td>55.63</td><td>43.29</td><td>35.95</td><td>69.01</td><td>27.67</td><td>26.83</td></tr><tr><td>Oblique-only</td><td>63.06</td><td>54.09</td><td>76.58</td><td>88.73</td><td>11.95</td><td>59.10</td></tr><tr><td>Fused</td><td>63.17</td><td>63.10</td><td>78.19</td><td>85.83</td><td>25.75</td><td>61.32</td></tr></table>

Table 8 Per-class IoU comparison of diferent projection modes on H3D. The best result for each class is highlighted in bold.
<table><tr><td>Mode</td><td>Low Veg.</td><td>Imp. Surf.</td><td>Vehicle</td><td>Urb. Furn.</td><td>Roof</td><td>Facade</td><td>Shrub</td><td>Tree</td><td>Soil/Gravel</td><td>Vert. Surf.</td><td>Chimney</td></tr><tr><td>BEV-only</td><td>41.82</td><td>47.69</td><td>42.51</td><td>6.63</td><td>73.63</td><td>0.48</td><td>4.44</td><td>16.34</td><td>29.88</td><td>2.30</td><td>0.16</td></tr><tr><td>Oblique-only</td><td>58.39</td><td>65.15</td><td>17.02</td><td>5.48</td><td>60.75</td><td>10.76</td><td>2.09</td><td>31.34</td><td>39.31</td><td>8.74</td><td>0.37</td></tr><tr><td>Fused</td><td>54.04</td><td>56.55</td><td>35.24</td><td>8.00</td><td>71.49</td><td>12.04</td><td>5.01</td><td>31.82</td><td>29.93</td><td>7.77</td><td>0.34</td></tr></table>

Table 9 Per-class IoU comparison of diferent projection modes on CUS3D. The best result for each class is highlighted in bold.
<table><tr><td>Mode</td><td>Building</td><td>Road</td><td>Car</td><td>Grass</td><td>High Veg.</td><td>Playground</td><td>Water</td><td>Farmland</td><td>Bldg. sites</td><td>Ground</td></tr><tr><td>BEV-only</td><td>36.89</td><td>54.92</td><td>38.39</td><td>29.46</td><td>40.48</td><td>14.19</td><td>67.25</td><td>31.90</td><td>14.39</td><td>3.29</td></tr><tr><td>Oblique-only</td><td>46.84</td><td>51.97</td><td>22.96</td><td>25.14</td><td>39.77</td><td>3.37</td><td>79.84</td><td>57.94</td><td>3.35</td><td>3.87</td></tr><tr><td>Fused</td><td>58.74</td><td>57.58</td><td>39.13</td><td>33.71</td><td>53.70</td><td>6.24</td><td>78.06</td><td>53.92</td><td>3.68</td><td>8.02</td></tr></table>

Meanwhile, Fig. 7 also reveals the limitation of hard reliability thresholding. Although a high threshold can suppress noisy pseudo labels and improve precision, it also discards many valid target points, leading to insuficient semantic coverage. This is undesirable for cluster-centric propagation. Therefore, instead of relying only on hard selection, COSTA sets a low threshold for the cluster bank (in Eq. (28)) followed by a soft reliability-weighted voting strategy during cluster propagation.

## 4.3.3 Ablation study

We conduct ablation study on the DALES-SUM benchmark to validate the contribution of each component in COSTA. The Base setting starts with hybrid projection and VLMs pseudo-label generation without TTA or cluster-centric propagation, which is based on Sa2VA (Yuan et al. 2025) and hybrid multi-view projection strategy (Sec. 3.2.1). As shown in Table 10, it obtains 62.89% mIoU and 81.07% OA, which has been demonstrated in Sec. 4.3.1. By introducing BN-only TTA with entropy minimization and diversity regularization, followed by cluster-centric propagation, the CP setting improves mIoU to 65.96% and OA to 86.38%, yielding gains of 3.07% and 5.31%, respectively. This confirms that adapting the feature distribution at inference time helps form more coherent target-domain clusters, while cluster-level propagation efectively reduces point-wise pseudo-label noise. Adding the OOD discovery loss further improves performance to 67.80% mIoU and 87.42% OA. It demonstrates that explicitly encouraging high uncertainty for pseudo-OOD samples prevents targetspecific classes from being over-confidently absorbed into source classes, which is crucial for OSTTA. Finally, the full model equipped with reliability-weighted clustercentric propagation achieves the best performance, reaching 70.09% mIoU and 88.52% OA. Compared with $O O D / / C P ,$ it further improves mIoU by 2.29%, demonstrating that treating all pseudo-label seeds equally is suboptimal. By weighting cluster votes according to pseudo-label reliability, COSTA suppresses unreliable VLM predictions and allows high-confidence semantic evidence to dominate the cluster decision. Overall, the full pipeline improves over the base setting by 7.20% mIoU and 7.45% OA, validating the contribution of each component in COSTA.

![](images/0076c0abf7a0387d3d111d9ad174aa1cf747b5cab6a7d583a5cdcf6544895877.jpg)

![](images/049f8a2d826087f61cd4be77f72558d29386204268ac7cca54148257603315e1.jpg)

![](images/e44e6d0532798c7b6fc34c8b6ce6f5c3fc891a744402a4e7006a40c691f96ef2.jpg)  
Fig. 7 Overall pseudo-label quality changes w.r.t. to the reliability weight thresholds across SUM, H3D, and CUS3D. Bars show pseudo-label recall, and plots show pseudo-label precision with diferent reliability thresholds.

Table 10 Ablation study of the proposed pipeline on the DALES-SUM benchmark.
<table><tr><td>Setting</td><td>H+S</td><td>Ent.</td><td>Div.</td><td>OOD</td><td>RW-CP</td><td>mIoU (%)</td><td>OA (%)</td></tr><tr><td>Base</td><td>√</td><td>1</td><td></td><td></td><td></td><td>62.89</td><td>81.07</td></tr><tr><td>CP</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td>65.96</td><td>86.38</td></tr><tr><td>OOD+CP</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>67.80</td><td>87.42</td></tr><tr><td>Full</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>70.09</td><td>88.52</td></tr></table>

H+S denotes hybrid projection with Sa2VA pseudo-label generation. Ent. and Div. denote entropy minimization and diversity regularization used in BN-only test-time adaptation. OOD denotes the out-of-distribution entropy. CP denotes cluster-centric propagation, and RW-CP denotes reliability-weighted cluster-centric propagation.

## 4.3.4 Discovery of the underlying semantics

To assess whether our method learns to aggregate samples from target classes into diferent clusters, we visualize the feature representations of the SUM test samples via t-SNE (Van der Maaten and Hinton 2008) in Fig. 8. It can be observed that the adapted feature space exhibits a clear and well-structured organization. Each class occupies distinct regions in the embedding space. In particular, terrain, high vegetation, and building form relatively compact and separable manifolds, suggesting that the adapted model can efectively capture their semantic regularities under domain shift. Furthermore, target-specific water and boat also exhibit small and identifiable semantic structures, allowing them to be clustered into identifiable groups. This indicates that our method has learned the underlying semantics of known and unseen classes by producing a well-structured feature space for target-specific classes, which also supports our observation in Fig. 1.

![](images/a05fca187a14a44e2a10bba168d53765d079abd70fdb73a65d0c32a1f7aa8f26.jpg)  
Fig. 8 Feature space visualization for target classes on SUM. Our method produces a well-structured feature space for target-specific classes.

## 5 Discussion and Conclusion

We investigate aerial point cloud semantic segmentation in an open-set test-time adaptation setting (OSTTA), for which we establish a new evaluation protocol covering both domain shifts and semantic shifts. We demonstrate that COSTA achieves better performance than direct open-vocabulary transfer and competitive results against fully-supervised methods, while requiring no target annotations, source-data replay, or ofline retraining. In addition to better generalization across domains, COSTA addresses the fundamental closed-set limitation of existing test-time adaptation methods by supporting user-specified target classes. Furthermore, we observe that adapted features naturally form semantically pure clusters transferable across label spaces, enabling better cross-semantics generalization.

The advantages and limitations of COSTA are inherently related. The proposed hybrid multi-view rendering allows 2D VLMs to be seamlessly applied to aerial point clouds through prompt-based segmentation without well-aligned synchronously acquired images. This design improves the flexibility of OSTTA, but also makes the quality of VLM-derived pseudo labels dependent on point density, visibility, and rendering quality. As a result, COSTA remains less efective for small-scale, sparse, or fine-grained categories, such as urban furniture, facade, and chimney. These classes are more sensitive to occlusion, poor rendering quality, and ambiguous semantic boundaries. Beyond rendering quality, a deeper issue lies in semantic ambiguity: the definitions of certain aerial point cloud categories are not well-defined. For example, urban furniture is a heterogeneous umbrella category that may include benches, poles, lights, and other visually diverse small objects, rather than a visually atomic category with consistent appearance. In this case, a simple text prompt such as “urban furniture” may not provide suficiently discriminative visual grounding for VLMs. Therefore, COSTA is efective when reliable semantic seeds and coherent feature structures can be jointly established, but remains challenged by weakly observable objects and semantically ambiguous category definitions.

These findings provide guidance for future research on open-world 3D scene understanding. We envision COSTA as a step toward a more flexible and practical paradigm for aerial point cloud understanding in open environments, where perception models can adapt to new geographic regions, sensors, and task-specific category definitions without expensive re-annotation or re-training. Future work could be extended to a unified end-to-end approach for online discovery of novel classes to enable continual learning in real-world deployments. In particular, 3D Gaussian Splatting ofers a promising 2D-3D intermediate representation to model geometry, appearance, opacity, and visibility in a continuous and diferentiable form. Compared with direct pointto-image projection, Gaussian-based rendering may provide denser and more realistic views, better preserve occlusion relationships, and accumulate multi-view VLM evidence at the Gaussian level before propagating. We hope that our insights inspire future investigation and contribute to building models capable of generalizing to unseen environments.

Acknowledgements. This work is supported by Jing-Jin-Ji Regional Integrated Environmental Improvement National Science and Technology Major Project (No.2026ZD1214401).

## Declarations

The authors have no relevant financial or non-financial interests to disclose.

## References

Alami, A., Remondino, F.: Open-vocabulary segmentation of aerial point clouds. Remote Sensing 18(4), 572 (2026)

Carion, N., Gustafson, L., Hu, Y.-T., Debnath, S., Hu, R., Suris, D., Ryali, C., Alwala, K.V., Khedr, H., Huang, A., Lei, J., Ma, T., Guo, B., Kalla, A., Marks, M., Greer, J., Wang, M., Sun, P., R¨adle, R., Afouras, T., Mavroudi, E., Xu, K., Wu, T.-H., Zhou, Y., Momeni, L., Hazra, R., Ding, S., Vaze, S., Porcher, F., Li, F., Li, S., Kamath, A.,

Cheng, H.K., Doll´ar, P., Ravi, N., Saenko, K., Zhang, P., Feichtenhofer, C.: SAM 3: Segment Anything with Concepts (2025). https://arxiv.org/abs/2511.16719

Choy, C., Gwak, J., Savarese, S.: 4d spatio-temporal convnets: Minkowski convolutional neural networks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3075–3084 (2019)

Cho, S., Shin, H., Hong, S., Arnab, A., Seo, P.H., Kim, S.: Cat-seg: Cost aggregation for open-vocabulary semantic segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4113–4123 (2024)

Dong, X., Bao, J., Zheng, Y., Zhang, T., Chen, D., Yang, H., Zeng, M., Zhang, W., Yuan, L., Chen, D., Wen, F., Yu, N.: Maskclip: Masked self-distillation advances contrastive language-image pretraining. In: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10995–11005 (2023). https: //doi.org/10.1109/CVPR52729.2023.01058

Dong, H., Chatzi, E., Fink, O.: Towards robust multimodal open-set test-time adaptation via adaptive entropy-aware optimization. arXiv preprint arXiv:2501.13924 (2025)

Ding, R., Yang, J., Xue, C., Zhang, W., Bai, S., Qi, X.: Pla: Language-driven openvocabulary 3d scene understanding. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7010–7019 (2023)

Gao, Y., Cao, D., Xi, X., Nie, S., Xia, S., Wang, C.: Source-free domain adaptation for geospatial point cloud semantic segmentation. arXiv preprint arXiv:2601.08375 (2026)

Graham, B., Engelcke, M., Van Der Maaten, L.: 3d semantic segmentation with submanifold sparse convolutional networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 9224–9232 (2018)

Ghiasi, G., Gu, X., Cui, Y., Lin, T.-Y.: Scaling open-vocabulary image segmentation with&nbsp;image-level labels. In: Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXVI, pp. 540–557. Springer, Berlin, Heidelberg (2022). https://doi.org/10.1007/ 978-3-031-20059-5 31

Gong, T., Jeong, J., Kim, T., Kim, Y., Shin, J., Lee, S.-J.: Note: Robust continual test-time adaptation against temporal correlation. Advances in Neural Information Processing Systems 35, 27253–27266 (2022)

Gao, L., Liu, Y., Chen, X., Liu, Y., Yan, S., Zhang, M.: Cus3d: A new comprehensive urban-scale semantic-segmentation 3d benchmark dataset. Remote Sensing 16(6), 1079 (2024)

Gao, Y., Li, C., You, Z., Liu, J., Zhen, L., Chen, P., Chen, Q., Tang, Z., Wang, L., Tang, Y., et al.: Openfly: A comprehensive platform for aerial vision-language navigation. In: International Conference on Learning Representations, vol. 2026, pp. 101858–101871 (2026)

Gao, W., Nan, L., Boom, B., Ledoux, H.: Sum: A benchmark dataset of semantic urban meshes. ISPRS Journal of Photogrammetry and Remote Sensing 179, 108– 120 (2021) https://doi.org/10.1016/j.isprsjprs.2021.07.008

Gao, F., Yan, Y., Lin, H., Shi, R.: Piie-dsa-net for 3d semantic segmentation of urban indoor and outdoor datasets. Remote Sensing 14(15), 3583 (2022)

Gao, Z., Zhang, X.-Y., Liu, C.-L.: Unified entropy optimization for open-set test-time adaptation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23975–23984 (2024)

Hu, Q., Yang, B., Fang, G., Guo, Y., Leonardis, A., Trigoni, N., Markham, A.: Sqn: Weakly-supervised semantic segmentation of large-scale 3d point clouds. In: European Conference on Computer Vision, pp. 600–619 (2022). Springer

Hu, Q., Yang, B., Khalid, S., Xiao, W., Trigoni, N., Markham, A.: Sensaturban: Learning semantics from urban-scale photogrammetric point clouds. International Journal of Computer Vision 130(2), 316–343 (2022)

Hu, Q., Yang, B., Xie, L., Rosa, S., Guo, Y., Wang, Z., Trigoni, N., Markham, A.: Randla-net: Eficient semantic segmentation of large-scale point clouds. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11108–11117 (2020)

Hou, Y., Zou, B., Zhang, M., Yang, S., Zhang, Y., Zhuo, J., Chen, S., Chen, J., Ma, H.: Agc-drive: A large-scale dataset for real-world aerial-ground collaboration in driving scenarios. Advances in Neural Information Processing Systems 38 (2026)

Jiang, L., Shi, S., Schiele, B.: Open-vocabulary 3d semantic segmentation with foundation models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21284–21294 (2024)

K¨olle, M., Laupheimer, D., Schmohl, S., Haala, N., Rottensteiner, F., Wegner, J.D., Ledoux, H.: The hessigheim 3d (h3d) benchmark on semantic segmentation of highresolution 3d point clouds and textured meshes from uav lidar and multi-view-stereo. ISPRS Open Journal of Photogrammetry and Remote Sensing 1, 100001 (2021)

Kolodiazhnyi, M., Vorontsova, A., Konushin, A., Rukhovich, D.: Oneformer3d: One transformer for unified point cloud segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20943–20953 (2024)

Liu, Y., Kong, L., Cen, J., Chen, R., Zhang, W., Pan, L., Chen, K., Liu, Z.: Segment

any point cloud sequences by distilling vision foundation models. In: Advances in Neural Information Processing Systems (2023)

Lee, B.-J., Lee, J.-S., Lee, J.-H.: Stabilizing open-set test-time adaptation via primary-auxiliary filtering and knowledge-integrated prediction. arXiv preprint arXiv:2508.18751 (2025)

Li, C., Liu, Y., Li, X., Zhang, Y., Li, T., Yuan, J.: Deep hierarchical learning for 3d semantic segmentation. International Journal of Computer Vision 133(7), 4420– 4441 (2025)

Lin, Y., Li, T., Zhou, S., Zhang, J., Fang, L., Yao, W.: Towards open-vocabulary als point clouds semantic segmentation: An empirical study. The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences 49, 223– 229 (2026)

Landrieu, L., Simonovsky, M.: Large-scale point cloud semantic segmentation with superpoint graphs. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4558–4567 (2018)

Li, Y., Xu, X., Su, Y., Jia, K.: On the robustness of open-world test-time training: Self-training with dynamic prototype expansion. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 11836–11846 (2023)

Niu, S., Wu, J., Zhang, Y., Chen, Y., Zheng, S., Zhao, P., Tan, M.: Eficient testtime model adaptation without forgetting. In: International Conference on Machine Learning, pp. 16888–16905 (2022). PMLR

Niu, S., Wu, J., Zhang, Y., Wen, Z., Chen, Y., Zhao, P., Tan, M.: Towards stable test-time adaptation in dynamic wild world. arXiv preprint arXiv:2302.12400 (2023)

Peng, S., Genova, K., Jiang, C., Tagliasacchi, A., Pollefeys, M., Funkhouser, T., et al.: Openscene: 3d scene understanding with open vocabularies. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 815–824 (2023)

Polewski, P., Yao, W., Heurich, M., Krzystek, P., Stilla, U.: Detection of fallen trees in als point clouds using a normalized cut approach trained by simulation. ISPRS Journal of Photogrammetry and Remote Sensing 105, 252–271 (2015) https://doi. org/10.1016/j.isprsjprs.2015.01.010

Qian, G., Li, Y., Peng, H., Mai, J., Hammoud, H., Elhoseiny, M., Ghanem, B.: Pointnext: Revisiting pointnet++ with improved training and scaling strategies. Advances in neural information processing systems 35, 23192–23204 (2022)

Qi, C.R., Su, H., Mo, K., Guibas, L.J.: Pointnet: Deep learning on point sets for 3d classification and segmentation. In: Proceedings of the IEEE Conference on Computer

Qi, C.R., Yi, L., Su, H., Guibas, L.J.: Pointnet++: Deep hierarchical feature learning on point sets in a metric space. Advances in neural information processing systems 30 (2017)

Saltori, C., Krivosheev, E., Lathuili´ere, S., Sebe, N., Galasso, F., Fiameni, G., Ricci, E., Poiesi, F.: Gipso: Geometrically informed propagation for online adaptation in 3d lidar segmentation. In: European Conference on Computer Vision, pp. 567–585 (2022). Springer

Shao, J., Yao, W., Wang, P., He, Z., Luo, L.: Urban geobim construction by integrating semantic lidar point clouds with as-designed bim models. IEEE Transactions on Geoscience and Remote Sensing 62, 1–12 (2024)

Takmaz, A., Fedele, E., Sumner, R.W., Pollefeys, M., Tombari, F., Engelmann, F.: Openmask3d: Open-vocabulary 3d instance segmentation. arXiv preprint arXiv:2306.13631 (2023)

Tang, H., Liu, Z., Zhao, S., Lin, Y., Lin, J., Wang, H., Han, S.: Searching eficient 3d architectures with sparse point-voxel convolution. In: European Conference on Computer Vision, pp. 685–702 (2020). Springer

Thomas, H., Qi, C.R., Deschaud, J.-E., Marcotegui, B., Goulette, F., Guibas, L.J.: Kpconv: Flexible and deformable convolution for point clouds. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6411–6420 (2019)

Varney, N., Asari, V.K., Graehling, Q.: Dales: A large-scale aerial lidar data set for semantic segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp. 186–187 (2020)

Maaten, L., Hinton, G.: Visualizing data using t-sne. Journal of machine learning research 9(11) (2008)

Wang, P.-S.: Octformer: Octree-based transformers for 3d point clouds. ACM Transactions on Graphics (TOG) 42(4), 1–11 (2023)

Wu, X., DeTone, D., Frost, D., Shen, T., Xie, C., Yang, N., Engel, J., Newcombe, R., Zhao, H., Straub, J.: Sonata: Self-supervised learning of reliable point representations. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22193–22204 (2025)

Wang, Q., Fink, O., Van Gool, L., Dai, D.: Continual test-time domain adaptation. In: Proceedings of Conference on Computer Vision and Pattern Recognition (2022)

Wu, X., Jiang, L., Wang, P.-S., Liu, Z., Liu, X., Qiao, Y., Ouyang, W., He, T., Zhao, H.: Point transformer v3: Simpler faster stronger. In: Proceedings of the IEEE/CVF

Wang, C., Jing, K., Zhu, J., Wang, D.: Openurban3d: Label-free open-vocabulary semantic segmentation of large-scale urban point clouds. IEEE Transactions on Geoscience and Remote Sensing (2026)

Wu, X., Lao, Y., Jiang, L., Liu, X., Zhao, H.: Point transformer v2: Grouped vector attention and partition-based pooling. Advances in Neural Information Processing Systems 35, 33330–33342 (2022)

Weijler, L., Mirza, M.J., Sick, L., Ekkazan, C., Hermosilla, P.: Ttt-kd: Test-time training for 3d semantic segmentation through knowledge distillation from foundation models. In: 2025 International Conference on 3D Vision (3DV), pp. 1264–1274 (2025). IEEE

Wang, D., Shelhamer, E., Liu, S., Olshausen, B., Darrell, T.: Tent: Fully test-time adaptation by entropy minimization. In: International Conference on Learning Representations (2021). https://openreview.net/forum?id=uXl3bZLkr3c

Wu, X., Tian, Z., Wen, X., Peng, B., Liu, X., Yu, K., Zhao, H.: Towards large-scale 3d representation learning with multi-dataset point prompt training. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19551–19562 (2024)

Wang, P., Yao, W.: A new weakly supervised approach for als point cloud semantic segmentation. ISPRS Journal of Photogrammetry and Remote Sensing 188, 237– 254 (2022)

Wang, X., Yang, D., Kwan, H., Chen, J., Li, H., Liao, Y., Liu, S., et al.: Towards realistic uav vision-language navigation: Platform, benchmark, and methodology. In: International Conference on Learning Representations, vol. 2025, pp. 7292–7310 (2025)

Wang, P., Yao, W., Shao, J., He, Z.: Test-time adaptation for geospatial point cloud semantic segmentation with distinct domain shifts. ISPRS Journal of Photogrammetry and Remote Sensing 229, 422–435 (2025)

Xiang, X., Jiang, H., Yu, Y., Shen, D., Zhen, J., Bao, H., Zhou, X., Zhang, G.: Eficient high-quality vectorized modeling of large-scale scenes. International Journal of Computer Vision 132(10), 4564–4588 (2024)

Xu, J., Liu, S., Vahdat, A., Byeon, W., Wang, X., De Mello, S.: Open-vocabulary panoptic segmentation with text-to-image difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2955– 2966 (2023)

Xu, J., Wei, Z., You, W., Li, L., Sun, W.: Cityseg: A 3d open vocabulary semantic segmentation foundation model in city-scale scenarios. arXiv preprint arXiv:2508.09470 (2025)

Yang, J., Ding, R., Deng, W., Wang, Z., Qi, X.: Regionplc: Regional point-language contrastive learning for open-world 3d scene understanding. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19823– 19832 (2024)

Yang, Y.-Q., Guo, Y.-X., Xiong, J.-Y., Liu, Y., Pan, H., Wang, P.-S., Tong, X., Guo, B.: Swin3d: A pretrained transformer backbone for 3d indoor scene understanding. Computational Visual Media 11(1), 83–101 (2025)

Yao, W., Hinz, S., Stilla, U.: Extraction and motion estimation of vehicles in single-pass airborne lidar data towards urban trafic analysis. ISPRS Journal of Photogrammetry and Remote Sensing 66(3), 260–271 (2011) https://doi.org/10.1016/j.isprsjprs. 2010.10.005

Yuan, H., Li, X., Zhang, T., Sun, Y., Huang, Z., Xu, S., Ji, S., Tong, Y., Qi, L., Feng, J., et al.: Sa2va: Marrying sam2 with llava for dense grounded understanding of images and videos. arXiv preprint arXiv:2501.04001 (2025)

Yuan, L., Xie, B., Li, S.: Robust test-time adaptation in dynamic scenarios. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15922–15932 (2023)

Zhao, H., Jiang, L., Jia, J., Torr, P.H., Koltun, V.: Point transformer. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16259–16268 (2021)

Zhu, Y., Kong, Y., Jie, Y., Xu, S., Cheng, H.: Graco: A multimodal dataset for ground and aerial cooperative localization and mapping. IEEE Robotics and Automation Letters 8(2), 966–973 (2023) https://doi.org/10.1109/LRA.2023.3234802

Zou, T., Qu, S., Li, Z., Knoll, A., He, L., Chen, G., Jiang, C.: Hgl: Hierarchical geometry learning for test-time adaptation in 3d point cloud segmentation. In: European Conference on Computer Vision, pp. 19–36 (2024). Springer

Zou, T., Yu, G., Wu, Y., Lu, F., Xu, Z., Bo, Z., Wang, Z., Qu, S., Chen, G.: Good: Geometry-guided out-of-distribution modeling for open-set test-time adaptation in point cloud semantic segmentation. In: Proc. Int. Conf. Learn. Representations (2026)