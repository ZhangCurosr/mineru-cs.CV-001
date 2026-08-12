# Compact Feed-Forward 3D Gaussians via Saliency-Guided Primitive Merging

Tim-Felix Faasch<sup>1</sup> tim-felix.faasch@de.bosch.com

Jochen Kall<sup>1</sup> jochen.kall@de.bosch.com

Cyrill Stachniss<sup>2,</sup> <sup>3</sup> cyrill.stachniss@igg.uni-bonn.de

<sup>1</sup> Bosch Research Hildesheim, Germany

<sup>2</sup> University of Bonn Bonn, Germany

<sup>3</sup> Lamarr Institute for Machine Learning and Artificial Intelligence Bonn, Germany

## Abstract

3D scene reconstruction, modeling, and rendering are highly relevant for numerous tasks, and 3D Gaussian splatting has become a standard choice in this context. Its feedforward variants provide fast reconstruction from sparse input views but often produce per-pixel primitives, leading to highly redundant and thus inefficient representations. We present a structure-aware merging pipeline that takes per-pixel primitives from any feedforward method and consolidates them into a compact, content-adaptive Gaussian set while largely retaining visual quality at just 1/20<sup>th</sup> of the Gaussians of a per-pixel method. We group spatially coherent Gaussians of similar appearance into variable-size clusters via adaptive superpixel segmentation guided by a saliency map, which allocates fine segments to textured regions and coarse segments to homogeneous areas. We compress each cluster into a compact latent representation through a learned encoder, then match and consolidate representations across views based on geometric overlap and feature similarity via a learned merger. A level-of-detail decoder then produces the final Gaussians at a controllable resolution, enabling a flexible quality-efficiency trade-off at inference. As a post-processing module, the pipeline is backbone-agnostic, leveraging the strengths of existing feed-forward methods. This leads to better and more robust quality than achieved by previous approaches that target a reduction in primitive count, while providing a highly compact representation, that can be rendered efficiently.

## Introduction

<sup>v</sup>3D Gaussian splatting (3DGS) [13] has emerged as a powerful representation for reconstructing 3D scenes with photorealistic quality, leading to its widespread adoption in sim-<sub>r</sub>ulation [12, 42], 3D generation [53, 55], and robotics [34, 46]. Traditional per-scene opti-<sup>a</sup>mization requires dense scene observations and consumes substantial compute, limiting its applicability to large-scale scenarios. Feed-forward (FF) reconstruction methods address this by predicting Gaussian primitives directly from input views [6, 8, 11, 33, 44, 45, 47, 48],

![](images/e15cda4e7ba6b133a6c309d936fb2223783417b2fba45b27eb815587eb40079a.jpg)  
Mom. Match. Ours K=1 Ours K=2 Ours K=4 ReSplat Init ReSplat Rec. VolSplat Unmerged

Figure 1: NVS quality vs. relative primitive count (DA3 backbone, averaged over DL3DV-Bench, MipNeRF360, and Tanks & Temples). Arrows indicate favorable direction.

but they typically produce per-pixel or per-voxel Gaussians, resulting in highly redundant representations [25, 39].

Since rendering cost in 3DGS scales with primitive count — each Gaussian must be sorted, projected, and alpha-composited — redundant primitives inflate inference latency and memory footprint without contributing additional visual detail. This inefficiency is particularly problematic for applications that scale to many scenes and complex downstream tasks, such as simulation-driven training of models for robotics or autonomous driving, where fast rendering is crucial for large-scale data generation. Redundancy can also hinder generalization to novel views, as overfitting to input views may lead to high-frequency artifacts. To address these challenges, we focus on reducing the number of primitives in FF-reconstructed scenes while maintaining visual quality. A good merging strategy should be content-adaptive, allocating representational capacity where it matters most, and it should be able to consolidate redundant primitives across overlapping views.

Existing approaches have made significant progress in addressing the problem of redundant Gaussians through spatial discretization [11, 38, 39], iterative refinement [43], or through detection-based primitive placement [25]. While these methods achieve notable reductions in primitive count with good rendering quality, opportunities for further improvement remain: uniform discretization strategies do not fully exploit the varying visual importance across scene regions, iterative refinement methods can lead to overfitting on input views, and detection-based methods operate during initial reconstruction without support for merging primitives across multiple views or adapting to new reconstruction backbones. We aim to address these open challenges with our work.

The main contribution of this paper is a novel technique to substantially reduce the complexity of 3D Gaussian scenes that feed-forward methods produce. We propose a structureaware, superpixel-based primitive merging strategy that takes per-pixel Gaussians from any feed-forward method and consolidates them into a highly compact representation. Contentadaptive superpixel segmentation, guided by local image structure, groups spatially coherent, perceptually similar per-pixel Gaussians. A learned set-attention encoder compresses each group into a compact latent representation, which we then match across views and merge using a learned fusion module. A local self-attention module refines the result, and a decoder produces output Gaussians at a controllable level of detail. This encode-merge-decode pipeline retains the bulk of the novel view synthesis quality at just $1 / { 2 0 } ^ { \mathrm { t h } }$ of the Gaussians of a per-pixel method, vastly increasing the rendering speed for downstream applications while providing more robust and reliable reconstructions than competing reduced-primitive methods, particularly in sparse-view settings. Crucially, the merging module attaches to any per-pixel Gaussian predictor without retraining the backbone, making it a flexible solution for improving the efficiency of existing FF 3DGS methods.

## 2 Related Work

3D Scene Representations: Traditional computer graphics has long relied on explicit representations such as meshes and point clouds for rasterization-based rendering pipelines [2]. Neural Radiance Fields (NeRF) [24] introduced a neural implicit approach that learns to represent scenes from multi-view imagery, enabling applications in reconstruction [5, 24], simulation [34], and content generation [27, 29]. Despite their quality, NeRFs suffer from high computational demands due to volumetric rendering. 3D Gaussian splatting (3DGS) [13] is an explicit representation that addresses this limitation by representing scenes as collections of 3D Gaussians rendered through differentiable rasterization, achieving both faster optimization and interactive rendering frame rates. However, 3DGS still requires per-scene optimization, which incurs significant computational cost when scaling to a large number of scenes and fails to converge to a meaningful geometry for sparse-view settings.

Feed-Forward Reconstruction: To overcome the limitations of per-scene optimization, feed-forward (FF) methods predict 3D Gaussians directly from input views [6, 8, 11, 33, 44, 45, 47, 48]. The backbone is typically a large vision transformer (ViT) [9] that encodes the input views and regresses the Gaussian parameters. FF reconstruction runs significantly faster than per-scene optimization and can handle sparse-view settings, but often places Gaussians less efficiently. Most existing FF methods predict per-pixel Gaussians [6, 8, 19, 33, 44, 45, 47, 48], which results in a highly redundant representation with an excessive number of sub-optimally placed primitives.

Efficient 3D Gaussian Splatting: The number of primitives in a 3DGS reconstruction directly affects rendering speed, memory consumption, and storage cost. Several strategies aim to reduce this overhead. Compression techniques such as quantization lower the perprimitive memory footprint without reducing the number of Gaussians [7, 16, 26, 36]. In contrast, compaction methods aim to reduce the primitive count itself: hierarchical representations merge primitives that contribute primarily to the background [14], and modified densification strategies prune redundant Gaussians during optimization [3, 10, 16, 17, 22]. However, these approaches are tightly coupled to the per-scene optimization process and are therefore not directly compatible with feed-forward reconstruction methods.

Some methods specifically tackle the efficiency of FF-produced representations. The most common approach is voxelization, where some methods directly predict voxel-aligned Gaussians [23, 28, 38] and others apply post-processing voxelization [11, 39]; yet uniform spatial discretization does not account for varying visual complexity across scene regions. Detection-based approaches [25] move from per-pixel regression to importance-driven Gaussian placement, allocating primitives according to visual saliency. Iterative refinement methods [43] predict Gaussians in a subsampled space and refine them using gradient-free feedback from the rendering error, yielding 16× fewer primitives than per-pixel methods. Both directions integrate Gaussian reduction into the reconstruction model itself, preventing them from leveraging the strong performance and generalization of large pre-trained backbones.

Graph-based fusion mechanisms [40, 50] address the redundancy of Gaussians in overlapping views by progressively merging per-pixel Gaussians across views via depth-proximity matching, GRU-based feature updates, or graph convolutions with overlap-weighted edges followed by pooling layers that prune geometrically close primitives. These methods rely on globally uniform proximity thresholds and do not account for visual complexity, allocating primitives uniformly regardless of scene content. In contrast, our method groups per-pixel Gaussians using content-adaptive superpixel segmentation, enabling structure-aware merging that respects object boundaries and concentrates primitive capacity in visually complex regions, while operating as a backbone-agnostic plug-in. A complementary direction compresses redundant multi-view inputs into a compact latent state before Gaussian prediction using an Information Bottleneck formulation, enabling FF models to scale to over a hundred input views [37]. This acts as an input-side compression module and does not address the redundancy of the predicted per-pixel Gaussians; it could therefore be combined with our output-side primitive merging to obtain an end-to-end efficient pipeline.

Superpixel Segmentation: Superpixel segmentation is a widely used technique in computer vision that groups pixels into small, perceptually meaningful regions based on color and spatial proximity [1, 31, 51]. Practitioners often apply it as a pre-processing step for tasks such as object recognition, image segmentation, and scene understanding to reduce computational complexity and improve efficiency of subsequent processing steps [4]. Recent superpixel methods with adaptive sizing capabilities [35, 52] are particularly effective, as they can allocate more segments to visually complex regions. However, as far as we are aware, no prior work has applied these methods to 3D scene reconstruction.

## 3 Preliminaries

3D Gaussian Splatting. In 3DGS [13], a scene is represented as a set of N anisotropic Gaussians $\mathcal { G } = \{ \mathcal { G } _ { 1 } , \ldots , \mathcal { G } _ { N } \}$ . Each Gaussian $\mathcal { G } _ { i }$ is parameterized by a mean $\boldsymbol { \mu _ { i } } \in \mathbb { R } ^ { 3 }$ and a covariance $\Sigma _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ , with its evaluation at position $\mathbf { x } \in \mathbb { R } ^ { 3 }$ given by

$$
\begin{array} { r } { \mathcal G _ { i } ( { \bf x } ) = \exp \left( - \frac { 1 } { 2 } ( { \bf x } - { \boldsymbol \mu } _ { i } ) ^ { \top } \Sigma _ { i } ^ { - 1 } ( { \bf x } - { \boldsymbol \mu } _ { i } ) \right) . } \end{array}\tag{1}
$$

For rendering, Gaussians are projected into image space via EWA splatting [56], yielding a 2D covariance $\Sigma _ { i } ^ { \prime } = \mathbf { J } \mathbf { W } \Sigma _ { i } \mathbf { W } ^ { \top } \mathbf { J } ^ { \top }$ with the viewing transformation W and the projection Jacobian J. The 3D covariance is decomposed as $\Sigma _ { i } = \mathbf { R } _ { i } \mathbf { S } _ { i } \mathbf { S } _ { i } ^ { \top } \mathbf { R } _ { i } ^ { \top }$ into rotation $\mathbf { R } _ { i }$ and diagonal scale $\mathbf { S } _ { i } .$ , and is parameterized by a quaternion ${ \bf q } _ { i } \in \mathbb { R } ^ { 4 }$ and scale vector $\mathbf { s } _ { i } \in \mathbb { R } ^ { 3 }$ . The viewdependent color $\mathbf { c } _ { i }$ of $\mathcal { G } _ { i }$ is encoded via spherical harmonic (SH) coefficients $\hat { \mathbf { c } } _ { 0 , i } , \hdots , \hat { \mathbf { c } } _ { M , i } .$ where $\begin{array} { r } { \mathbf { c } _ { i } = \sum _ { k = 0 } ^ { M } \hat { \mathbf { c } } _ { k , i } H _ { k } ( \mathbf { d } ) } \end{array}$ for view direction d $\in \mathbb { R } ^ { 3 }$ , with the SH basis functions $H _ { k }$ . The final pixel color is obtained by front-to-back alpha blending:

$$
{ \bf c } _ { \mathrm { p i x e l } } = \sum _ { i = 1 } ^ { N } T _ { i } \alpha _ { i } { \bf c } _ { i } , \quad T _ { i } = \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{2}
$$

where $T _ { i }$ is the transmittance, and $\alpha _ { i }$ the opacity of $\mathcal { G } _ { i }$ . Each Gaussian is fully described by its set of parameters $\Theta _ { i } = ( \mu _ { i } , \mathbf { q } _ { i } , \mathbf { s } _ { i } , \alpha _ { i } , \hat { \mathbf { c } } _ { 0 \ldots M , i } )$ . In the standard per-scene optimization setting, these parameters are optimized via gradient descent, minimizing a combination of different photometric losses between images rendered through differentiable rasterization and ground-truth images.

![](images/54f8a502b21038e844739871bef891d2c1f35bc3e5ae5d5b2e2a7f5129b9149f.jpg)  
Figure 2: Pipeline overview. Per-pixel Gaussians $\mathcal { G } ^ { \mathrm { F F } }$ are grouped into saliency-guided superpixels. The encoder ${ \mathcal { M } } ^ { \mathrm { e n c } }$ compresses each group into a single Feature Gaussian (FG), the merger $\mathcal { M } ^ { \mathrm { m r g } }$ fuses FGs that overlap across views, the refiner $\mathcal { M } ^ { \mathrm { r e f } }$ updates each FG based on its neighbors. The decoder $\mathcal { M } _ { K } ^ { \mathrm { d e c } }$ expands each FG into K output Gaussians, resulting in the output scene $\mathcal { G } _ { K }$

Feed-Forward Gaussian Splatting. Feed-forward (FF) methods [8, 11, 19, 44] bypass per-scene optimization by predicting Gaussian parameters directly from N input views $\{ \mathbf { I } _ { \nu } \} _ { \nu = 1 } ^ { N }$ . A typical architecture feeds each view through a vision transformer (ViT) [9] backbone to extract per-patch features, and applies multi-view cross-attention to obtain multiview-aware feature maps. Prediction heads then regress point maps, which can be projected into 3D space to retrieve the Gaussian means $\mu _ { i }$ using known camera parameters. The remaining parameters are similarly predicted from the patch features. FF methods have varying input requirements: posed methods require known camera intrinsics and extrinsics [6, 8], whereas unposed methods jointly predict camera parameters and scene geometry [11, 47].

## 4 Saliency-Guided Superpixel Merging

Given per-pixel Gaussian primitives predicted by an arbitrary frozen feed-forward backbone $\mathcal { M } ^ { \mathrm { F F } }$ , our method consolidates them into a compact representation that largely retains visual fidelity. A key design goal is content-adaptivity: homogeneous regions (e.g., sky, walls) can be aggressively compressed, whereas fine detail (e.g., at edges or textures) requires a higher representational capacity. We achieve this through an encode-merge-decode pipeline depicted in Fig. 2. First, saliency-guided superpixel segmentation groups spatially coherent, perceptually similar Gaussians at a content-adaptive granularity (Sec. 4.1). A learned encoder ${ \mathcal { M } } ^ { \mathrm { e n c } }$ compresses each group into a latent-augmented representation we call a Feature Gaussian (Sec. 4.2). A level-of-detail decoder $\mathcal { M } _ { K } ^ { \mathrm { { \tiny ~ d e c } } }$ reconstructs 3D Gaussians, compatible with existing rendering pipelines, at a controllable resolution K (Sec. 4.3). Cross-view matching identifies overlapping Feature Gaussians, and a learned merger ${ \mathcal { M } } ^ { \mathrm { m r g } }$ consolidates them, enabling further compression (Sec. 4.4). A local self-attention refiner $\mathcal { M } ^ { \mathrm { r e f } }$ provides neighborhood awareness so that thin structures spanning multiple groups can be faithfully reconstructed (Sec. 4.5). The pipeline is trained end-to-end with photometric losses on the rendered output (Sec. 4.6).

![](images/d32e28e4f3b5596229d277933c165bfdb6ed40e7e2327d935331774a1b2abedc.jpg)  
(a) Input $\mathbf { I } _ { \nu }$

![](images/c56a8e2bf379fcdd42c734702ef42ecf3953cb2713bebd666f51b2bcd1d12ca8.jpg)  
(b) Saliency $\lambda _ { \nu }$

![](images/205957cbbd1539b92d041fde2063575d0371b9717d3ee48c8d8a9c61ee747926.jpg)  
(c) SLIC

![](images/3d8ded79969b8c236a019b68d05c12ec64550e9d9eb7f2b625f4fa48e60ee92f.jpg)  
(d) BASS

![](images/4bff30b68549ae1f6b806879d77e92dc2eff959cfbe8be74cbb48a7ececbdb4a.jpg)  
(e) BASS +λ<sub>v</sub>  
Figure 3: Superpixel comparison at similar segment count: SLIC (uniform), BASS with uniform seeds, and BASS with saliency-guided seeds $\lambda _ { \nu }$

## 4.1 Content-Adaptive Superpixel Grouping

The first step partitions each view’s per-pixel Gaussians into groups that the encoder can compress. A good group for merging should be color-homogeneous (so appearance can be preserved), spatially contiguous (necessary for 3D coherence), and have regular shape (easier to represent with few Gaussians). At the same time, group size should adapt to local scene complexity: edges and corners carry geometric detail that merging would destroy, so they require smaller groups and thus more output primitives; flat regions can be aggressively merged.

Bayesian Adaptive Superpixel Segmentation (BASS) [35] produces segments with the first three properties by iteratively maximizing a posterior that balances color homogeneity with spatial compactness, while maintaining regular segment boundaries. Standard BASS initializes seeds on a uniform hexagonal grid, and achieves content adaptivity by iteratively shifting superpixel centers and borders. For very small segment sizes, this can still lead to a large number of segments in flat regions, which is undesirable for our application. We replace the uniform initialization with a saliency-guided seed placement: we compute the Shi-Tomasi corner response $\lambda _ { \nu }$ [32] at each pixel (the minimum eigenvalue of the image structure tensor [30]) and sample seeds densely where this response is high and sparsely elsewhere. BASS then refines these seeds into final segments of similar color and a regular shape. The output is a per-view partition map $\mathbf { M } _ { \nu }$ that assigns each pixel (and its associated Gaussian) to a superpixel group $S _ { j }$ . Through the combination of saliency-guided seeding and BASS segmentation, we obtain small segments in regions of high detail and larger segments in flatter regions, effectively adapting to local scene complexity (Fig. 3).

## 4.2 Feature Gaussian Encoder

Given a superpixel group $\boldsymbol { S } _ { j }$ , the encoder ${ \mathcal { M } } ^ { \mathrm { e n c } }$ compresses its Gaussians into an intermediate representation that downstream stages (matching, merging, refinement) can process efficiently. To achieve this, we introduce the Feature Gaussian $\tilde { \mathcal { G } } _ { j }$ , a latent-augmented representation that encodes the entire group with a single set of parameters. A Feature Gaussian has the same spatial format as a regular Gaussian (position $\mu _ { j { \ ' } }$ , scale $\mathbf { s } _ { j } .$ , and rotation $\mathbf { q } _ { j } )$ , but replaces view-dependent color with a base color $\mathbf { c } _ { j }$ and a latent vector $\mathbf { z } _ { j } \in \mathbb { R } ^ { d }$ . The latent stores the visual appearance and geometric structure of the entire group so that the decoder can later expand it into one or more output Gaussians that capture the group’s complexity.

![](images/3d30ca99c484a79bb409cf2fc1a993072fad32e4f6fdc28bc8a129210b5c02d2.jpg)

![](images/ae292f7522961929f0e897114ad57c1f20ef513b67424862820cfba88fc4fc38.jpg)  
Figure 4: Internal architecture of our modules built on the Set Transformer [18] building blocks SAB and PMA. (a) The encoder and merger share the same many-to-one pattern: n input tokens are refined by SABs and aggregated into a single output via PMA and a learnable query $\mathcal { Q } .$ (b) The decoder inverts this pattern: a single Feature Gaussian is expanded into K output Gaussians via learned slot tokens.

Compressing a variable-size set of Gaussians into a single Feature Gaussian requires an architecture that is permutation-invariant (the output should not depend on input order) and handles arbitrary group sizes. We use the Set Transformer [18]: an MLP projects per-Gaussian attributes into tokens, a stack of Set Attention Blocks (SABs) refines them via pairwise interactions, and a Pooling-by-Multihead-Attention (PMA) layer aggregates the set into a single output using a learnable query Q (Fig. 4). A final MLP produces the latent feature $\mathbf { z } _ { j }$ , and a position offset from the opacity-weighted centroid, allowing the encoder to compensate for outliers that would otherwise skew the placement of the Feature Gaussian. The base color $\mathbf { c } _ { j }$ is the opacity-weighted average of the group members’ diffuse color. Notably, the encoder does not predict the scale ${ \bf s } _ { j }$ or rotation ${ \bf q } _ { j }$ that are needed for cross-view matching; these are produced by the decoder $\mathcal { M } _ { 1 } ^ { \mathrm { d e c } }$ . This design prevents a collapse mode where the encoder shrinks geometry to avoid cross-view matching entirely, which would simplify reconstruction but hurts compression.

## 4.3 Level-of-Detail Decoder

The decoder $\mathcal { M } _ { K } ^ { \mathrm { d e c } }$ produces the K output Gaussians from each Feature Gaussian, which are compatible with any differentiable rasterizer and can be directly rendered. We train multiple decoder heads for different K: the K=1 head provides the geometry of the Feature Gaussian, which is used for cross-view matching and serves as maximum-compression output, while heads with $K > 1$ trade a higher number of Gaussians for higher reconstruction fidelity. This exposes K as an inference-time knob, allowing the user to balance quality against rendering compute cost without retraining.

Expanding a single latent into K outputs requires learned specialization. We use a slotbased design (Fig. 4): a slot-seed MLP maps the latent feature $\mathbf { z } _ { j }$ and base color $\mathbf { c } _ { j }$ into K slot tokens, per-slot self-attention lets slots coordinate, and output heads predict each Gaussian’s parameters. Mean $\mu _ { i }$ and base color $\mathbf { c } _ { i }$ are predicted as offsets from the Feature Gaussian’s parameters. At inference, we prune Gaussians with $\alpha _ { i } < \tau _ { a }$ for further compression.

## 4.4 Cross-View Matching and Merging

Feature Gaussians $\tilde { \mathcal { G } }$ from different views that observe the same 3D region should be merged to reduce redundancy. We identify candidate groups using spatial overlap and feature similarity, then fuse them with a learned merger $\mathcal { M } ^ { \mathrm { m r g } }$

For each Feature Gaussian $\tilde { \mathcal { G } } _ { j }$ , we retrieve the k nearest neighbors via 3D kNN on positions, excluding candidates from the same input view. A candidate pair passes the match gate if both (i) the cosine similarity between latent features exceeds $\tau _ { f } .$ , and (ii) the AABB intersection-over-union of the FGs’ geometry exceeds $\tau _ { g }$ . This produces a set of matched edges over all Feature Gaussians.

Since one FG may match several others, and matching is not transitive, we take the connected components of this edge graph as the final merge groups. Feature Gaussians with no matches are appended to the final set $\tilde { \mathcal { G } }$ unchanged. $\mathcal { M } ^ { \mathrm { m r g } }$ merges each connected group using the same SAB+PMA [18] backbone as the encoder (Fig. 4). Dedicated zeroinitialized residual heads predict parameter updates to the $0 ^ { \mathrm { t h } }$ Feature Gaussian in the group: $\Delta \mu , \Delta \mathbf { z } , \Delta m , \Delta \mathbf { c } _ { 0 } , \Delta \mathbf { s } , \Delta \mathbf { q }$ . This way the merger starts as an identity and training remains stable.

## 4.5 Refinement and Online Reconstruction

Refinement. The final rendering quality depends in part on how well neighboring Feature Gaussians $\tilde { \mathcal { G } }$ interact — particularly along thin structures $( e . g .$ , fences, poles) or at depth boundaries. In these areas, individually compressed groups might not perfectly line up in the final reconstruction. The refiner $\mathcal { M } ^ { \mathrm { r e f } }$ addresses this by updating each Feature Gaussian $\tilde { \mathcal { G } } _ { j }$ based on its kNN neighborhood through self-attention: neighbors’ relative positions and attributes serve as context tokens, and residual heads predict parameter updates $\Delta \mu , \Delta \mathbf { z } , \Delta m , \Delta \mathbf { c } _ { 0 } , \Delta \mathbf { s } , \Delta \mathbf { q }$ . As with the merger, all residual projections are zero-initialized for stable training.

Online reconstruction. Our pipeline also supports online reconstruction, where views are processed as a sequence as they arrive rather than in a single batch. This allows for updates to the reconstructed scene as new observations become available and enables reconstruction of large scenes that do not fit into memory at once. In practice, we initialize the Feature Gaussian set $\tilde { \mathcal { G } }$ from a first batch of observations. When a new view arrives, the pipeline groups and encodes its per-pixel Gaussians into new Feature Gaussians, then matches and merges them into the existing set $\tilde { \mathcal { G } }$ . The decoder produces output Gaussians on demand from the current state of $\tilde { \mathcal { G } }$ via $\mathcal { M } _ { K } ^ { \mathrm { d e c } }$

## 4.6 Training Objectives

During training, the feed-forward backbone $\mathcal { M } ^ { \mathrm { F F } }$ remains frozen; we optimize only the encoder ${ \mathcal { M } } ^ { \mathrm { e n c } }$ , merger $\mathcal { M } ^ { \mathrm { m r g } }$ , refiner $\mathcal { M } ^ { \mathrm { r e f } }$ , and decoders $\mathcal { M } _ { K } ^ { \mathrm { d e c } }$ . We train the pipeline end-toend on multi-view image data with the following losses.

Photometric reconstruction. The primary supervision consists of Mean Squared Error (MSE), SSIM [41], and LPIPS [49] losses between rendered and ground-truth images, optionally computed on both input views and held-out novel views.

Teacher loss. Due to randomly initialized weights of our modules, the geometry of the compressed Gaussians is in turn essentially random during early training. This can cause single Gaussians to cover the renders from the training views entirely, inhibiting gradient flow. To address this, we supervise the encoder with a closed-form moment-matching target [14]: for each superpixel group $\boldsymbol { S } _ { j }$ , we fit a single Gaussian to the group’s opacity-weighted mean

$$
\mu _ { j } ^ { \mathrm { t e a c h } } = \frac { \sum _ { i \in { \mathcal { S } } _ { j } } \alpha _ { i } \mu _ { i } } { \sum _ { i \in { \mathcal { S } } _ { j } } \alpha _ { i } }\tag{3}
$$

and opacity-weighted covariance

$$
\Sigma _ { j } ^ { \mathrm { t e a c h } } = \frac { \sum _ { i \in { \cal S } _ { j } } \alpha _ { i } ( \mu _ { i } - \mu _ { j } ^ { \mathrm { t e a c h } } ) ( \mu _ { i } - \mu _ { j } ^ { \mathrm { t e a c h } } ) ^ { \top } } { \sum _ { i \in { \cal S } _ { j } } \alpha _ { i } } .\tag{4}
$$

We compare the resulting Gaussian parameters to the output of our merging pipeline using a weighted sum of a geodesic quaternion loss for ${ \bf q } _ { i }$ and L1 losses for the remaining parameters. This teacher signal decays on a schedule, allowing the learned encoder to eventually surpass the moment-matched initialization.

Decoder diversification. For $K > 1$ decoders, a regularizer encourages the K output Gaussians to be spatially diverse without collapsing into identical copies. We define a target AABB-IoU and apply an attraction/repulsion term around it, and additionally penalize large opacity differences within a group to prevent single-primitive dominance. Decaying this loss during training lets the photometric objective override it once diversity is established, allowing the model to collapse to fewer Gaussians where a single primitive suffices.

The complete training objective combines all terms:

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { \mathrm { M S E } } \mathcal { L } _ { \mathrm { M S E } } + \lambda _ { \mathrm { S S I M } } \mathcal { L } _ { \mathrm { S S I M } } + \lambda _ { \mathrm { L P I P S } } \mathcal { L } _ { \mathrm { L P I P S } } + \lambda _ { \mathrm { t e a c h } } \mathcal { L } _ { \mathrm { t e a c h } } + \lambda _ { \mathrm { d i v } } \mathcal { L } _ { \mathrm { d i v } } , } \end{array}\tag{5}
$$

where $\lambda _ { \mathrm { t e a c h } }$ and $\lambda _ { \mathrm { d i v } }$ follow a decay schedule and concrete weight values are provided in the supplementary material.

## 5 Experimental Evaluation

We evaluate our saliency-guided superpixel-based primitive merging strategy on novel view synthesis quality and number of primitives.

## 5.1 Experimental Setup

Implementation Details. We evaluate our approach with three complementary FF backbones: DepthSplat [44], a posed method that leverages multi-view cost volumes augmented with monocular depth features; AnySplat [11], an unposed method that jointly estimates geometry from unconstrained views; and Depth Anything 3 (DA3) [19], a foundation model that unifies depth estimation, pose estimation, and per-pixel 3D Gaussian prediction in a single ViT backbone, supporting both posed and unposed operation. We train all added modules with the AdamW optimizer [21] for 300 000 steps on the DL3DV-10K dataset [20], using a cosine schedule with a maximum learning rate of $1 0 ^ { - 6 }$ . For the level-of-detail decoder, we maintain three heads with K = 1, K = 2, and K = 4 output Gaussians per Feature Gaussian, which are jointly trained. We keep the backbones frozen during training.

Evaluation Details. We test the reconstruction quality on DL3DV-Bench [20], which contains 140 scenes not used during training. We additionally evaluate on MipNeRF360 [5] and Tanks and Temples [15], two widely used benchmarks that were not included in the training data. We sample input views at regular intervals along the camera trajectory, with novel views placed in-between. We mask out areas of the scene not visible in any input view and exclude them from metric computation to focus evaluation on the quality of the reconstruction rather than the extrapolation. Each experiment uses 3, 6, 9, and 12 input views; results are averaged across all scenes and view splits. Detailed per-view-count results and input view reconstruction metrics appear in the supplementary material. We report PSNR, SSIM [41], and LPIPS [49] for novel view synthesis. For efficiency we report the relative primitive count $r _ { c }$ (fraction of the unmerged count that is retained), which directly correlates with rendering time and memory usage.

In addition to reporting the quality of the reconstruction directly produced by the three backbones, we compare against ReSplat [43], which predicts Gaussians in a 16× subsampled space and refines them recurrently, and VolSplat [38], a voxel-based method producing Gaussian in a regular grid. We additionally report the performance of AnySplat’s built-in post-hoc voxelization and of heuristic moment matching of Gaussians grouped into superpixels. Due to unavailable public code, we could not include Off The Grid [25] and Fuse-and-Refine [39] in our evaluation.

## 5.2 Main Results

Novel View Synthesis Quality. Tab. 1 reports novel view synthesis results for all methods. Using our pipeline with the strongest backbone (DA3), the K = 1 decoder achieves 17.41 dB PSNR averaged over all benchmarks at $r _ { c } = 4 . 4 \%$ , even slightly surpassing the DA3 unmerged quality at 16.82 dB. The gain in performance over DA3 is only achieved on MipNeRF360 and Tanks & Temples, which have much larger viewpoint changes than DL3DV-Bench. This reflects the regularising effect of superpixel-based merging: fusing per-pixel Gaussians into compact, spatially coherent groups can remove floaters and overly large Gaussians, and reduces high-frequency noise. These effects are consistent across all backbones, with the largest gain for DepthSplat, which sometimes produces overly large Gaussians that strongly impact the reconstruction quality. LPIPS responds very sensitively to texture degradation from merging, and therefore our pipeline does not fully preserve it. ReSplat achieves a comparable $r _ { c }$ of 6.2% but with lower quality than our method, and without the flexibility of adapting to more powerful foundation models, as they become available. VolSplat performs significantly worse than the other methods, and fails to reproduce the reconstruction quality reported in its original paper. This is likely due to the fact that VolSplat was exclusively trained on RealEstate-10K [54], which mostly contains scenes with smaller viewpoint changes, different to our benchmarks.

Table 1: Novel view synthesis quality for all backbone variants and baselines. Best in bold, second best underlined.
<table><tr><td></td><td></td><td></td><td colspan="3">DL3DV-Bench</td><td colspan="3">MipNeRF360</td><td colspan="3">T&amp;T</td></tr><tr><td>Backbone</td><td>Method</td><td> $r _ { c }$ </td><td>(%)↓ PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td><td></td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td rowspan="6">AnySplat</td><td>Unmerged</td><td>100.0%</td><td>13.73</td><td>0.361</td><td>0.495</td><td>12.17</td><td>0.345</td><td>0.497</td><td>11.89</td><td>0.416</td><td>0.466</td></tr><tr><td>Voxelized</td><td>83.2%</td><td>13.76</td><td>0.360</td><td>0.497</td><td>12.17</td><td>0.344</td><td>0.498</td><td>11.89</td><td>0.413</td><td>0.468</td></tr><tr><td>Mom. Match.</td><td>5.3%</td><td>14.45</td><td>0.410</td><td>0.566</td><td>13.46</td><td>0.398</td><td>0.559</td><td>12.49</td><td>0.463</td><td>0.517</td></tr><tr><td>Ours (K = 1)</td><td>4.4%</td><td>14.70</td><td>0.408</td><td>0.550</td><td>14.48</td><td>0.405</td><td>0.544</td><td>13.24</td><td>0.458</td><td>0.504</td></tr><tr><td>Ours (K = 2)</td><td>8.9%</td><td>14.81</td><td>0.411</td><td>0.549</td><td>14.74</td><td>0.406</td><td>0.544</td><td>13.81</td><td>0.473</td><td>0.505</td></tr><tr><td>Ours (K =4)</td><td>17.8%</td><td>14.78</td><td>0.409</td><td>0.547</td><td>14.81</td><td>0.408</td><td>0.542</td><td>13.93</td><td>0.474</td><td>0.503</td></tr><tr><td rowspan="5">DepthSplat</td><td>Unmerged</td><td>100.0%</td><td>15.78</td><td>0.577</td><td>0.347</td><td>13.96</td><td>0.480</td><td>0.389</td><td>13.72</td><td>0.549</td><td>0.377</td></tr><tr><td>Mom. Match.</td><td>6.1%</td><td>13.16</td><td>0.466</td><td>0.573</td><td>12.35</td><td>0.424</td><td>0.569</td><td>11.43</td><td>0.498</td><td>0.550</td></tr><tr><td>Ours (K = 1)</td><td>5.3%</td><td>17.17</td><td>0.529</td><td>0.440</td><td>15.37</td><td>0.457</td><td>0.452</td><td>14.33</td><td>0.508</td><td>0.461</td></tr><tr><td>Ours (K = 2)</td><td>10.5%</td><td>17.41</td><td>0.536</td><td>0.436</td><td>16.05</td><td>0.469</td><td>0.448</td><td>14.60</td><td>0.517</td><td>0.458</td></tr><tr><td>Ours (K =4)</td><td>21.1%</td><td>17.50</td><td>0.539</td><td>0.433</td><td>16.31</td><td>0.474</td><td>0.446</td><td>14.78</td><td>0.522</td><td>0.457</td></tr><tr><td rowspan="5">DA3</td><td>Unmerged</td><td>100.0%</td><td>18.68</td><td>0.620</td><td>0.297</td><td>16.71</td><td>0.495</td><td>0.356</td><td>15.07</td><td>0.546</td><td>0.353</td></tr><tr><td>Mom. Match.</td><td>4.9%</td><td>13.79</td><td>0.459</td><td>0.637</td><td>14.04</td><td>0.440</td><td>0.601</td><td>12.55</td><td>0.505</td><td>0.611</td></tr><tr><td>Ours (K = 1)</td><td>4.4%</td><td>18.37</td><td>0.578</td><td>0.422</td><td>17.87</td><td>0.514</td><td>0.442</td><td>16.00</td><td>0.562</td><td>0.421</td></tr><tr><td>Ours (K =2)</td><td>8.8%</td><td>18.50</td><td>0.582</td><td>0.417</td><td>18.02</td><td>0.517</td><td>0.440</td><td>16.09</td><td>0.566</td><td>0.418</td></tr><tr><td>Ours (K =4)</td><td>17.3%</td><td>18.54</td><td>0.583</td><td>0.416</td><td>18.06</td><td>0.518</td><td>0.439</td><td>16.09</td><td>0.566</td><td>0.418</td></tr><tr><td rowspan="2">ReSplat</td><td>Init</td><td>6.2%</td><td>13.17</td><td>0.380</td><td>0.556</td><td>15.57</td><td>0.369</td><td>0.566</td><td>12.37</td><td>0.383</td><td>0.591</td></tr><tr><td>Recurrent 4</td><td>6.2%</td><td>15.39</td><td>0.468</td><td>0.519</td><td>16.79</td><td>0.418</td><td>0.558</td><td>13.38</td><td>0.437</td><td>0.578</td></tr><tr><td>VolSplat</td><td></td><td>93.4%</td><td>14.12</td><td>0.366</td><td>0.607</td><td>11.81</td><td>0.251</td><td>0.700</td><td>11.71</td><td>0.320</td><td>0.650</td></tr></table>

Table 2: Online reconstruction quality at end-of-sequence, averaged over all 9 MipNeRF360 scenes (target views). Best per column in bold.
<table><tr><td>Method</td><td>#Prim.↓</td><td> $r _ { c } ( \% ) \downarrow$ </td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FPS↑</td></tr><tr><td>Naïve backbone</td><td>16.1M</td><td>100.0</td><td>15.27</td><td>0.372</td><td>0.560</td><td>60</td></tr><tr><td>Ours K =1</td><td>1.0M</td><td>6.2</td><td>15.33</td><td>0.367</td><td>0.612</td><td>365</td></tr><tr><td>Ours K=2</td><td>2.0M</td><td>12.4</td><td>15.41</td><td>0.368</td><td>0.611</td><td>378</td></tr><tr><td>Ours K =4</td><td>4.0M</td><td>24.8</td><td>15.45</td><td>0.370</td><td>0.610</td><td>245</td></tr></table>

Primitive Count vs. Quality Trade-off. Fig. 1 shows the quality–efficiency trade-off for the DA3 backbone, averaged over all three benchmarks. Our method (K = 1,2,4) traces a favorable curve that stays close to the unmerged quality ceiling with $r _ { c }$ between 4.4% and 17.3%. ReSplat and VolSplat produce suboptimal results, failing to achieve the same quality as our method at comparable or higher primitive counts.

Qualitative Comparisons. Fig. 5 shows qualitative novel view synthesis results at 12 input views across all three benchmarks. Our method (K = 1, DA3 backbone) largely reproduces the unmerged quality at a fraction of the primitive count. ReSplat occasionally yields sharper detail but introduces more severe structured artifacts elsewhere. VolSplat fully fails to reconstruct the scene structure for many scenes.

Online Reconstruction. We evaluate our pipeline in a online setting where views are integrated incrementally. Starting from an empty scene, batches of 12 views are processed in sequence, in the order of the camera trajectory, until all context views are integrated. Half of the total views are used as input and the rest as target views for evaluation. We report primitive growth averaged over the 7 scenes from MipNeRF360 that provide at least 84 context views (bicycle, bonsai, counter, flowers, garden, kitchen, room), and novel view synthesis quality at the end of the sequence averaged over all 9 MipNeRF360 scenes. Fig. 6 compares the cumulative Gaussian count of our three decoders against the naïve baseline, which accumulates all raw per-pixel Gaussians without merging. The naïve baseline grows at ∼1.84 M Gaussians per 12 views; our K = 1 decoder grows at ∼0.12 M per step — a 16× reduction in slope owing to within-step compression from superpixel grouping, with additional savings from cross-step merging of overlapping regions. At 84 integrated views, K=1 retains 0.82 M Gaussians against 12.9 M for the naïve baseline. Tab. 2 summarises perscene-average quality at the end of each sequence. Despite holding 16× fewer primitives, our K =1 decoder closely approaches the naïve backbone on PSNR and SSIM while trading off some perceptual fidelity (LPIPS), and renders 6.1× faster.

DL3DV-Bench
<table><tr><td>GT</td><td>DA3 (backbone)</td><td>Ours (K = 1)</td><td>ReSplat</td><td>VolSplat</td></tr><tr><td>rc (%)</td><td>100</td><td>4.4</td><td>6.2</td><td>93.4</td></tr></table>

![](images/45098d7b6a14ae683512ddc45f835121854ba967fe358cb561dae1a9d9342957.jpg)  
Figure 5: Qualitative novel view synthesis comparison at 12 input views (DA3 backbone). Unseen areas are highlighted in red and excluded from metric computation.

![](images/8f9fe5111b534ef97912c05db82436ce27590e795ec93e16f6cd0f8f7f30d156.jpg)  
Figure 6: Gaussian count vs. integrated views in the online setting, averaged over 7 MipNeRF360 scenes.

Table 3: Ablation study (AnySplat backbone, K =1, averaged over all benchmarks). Best in bold, second best underlined.
<table><tr><td>Variant</td><td> $r _ { c }$  (%)↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Backbone (unm.)</td><td>100.0%</td><td>12.59</td><td>0.374</td><td>0.486</td></tr><tr><td>Ours (full)</td><td>4.4%</td><td>14.14</td><td>0.423</td><td>0.533</td></tr><tr><td>w/o refiner</td><td>4.7%</td><td>13.49</td><td>0.389</td><td>0.534</td></tr><tr><td>w/o cross-view</td><td>5.3%</td><td>14.13</td><td>0.424</td><td>0.531</td></tr><tr><td>zero shot larger SPs</td><td>2.0%</td><td>13.97</td><td>0.424</td><td>0.548</td></tr><tr><td>zero shot smaller SPs</td><td>8.7%</td><td>14.11</td><td>0.420</td><td>0.523</td></tr></table>

## 5.3 Ablation Studies

Replacing our learned merger with heuristic moment matching to combine the Gaussians within each superpixel performs worse across all metrics, confirming that a learned representation is essential for quality-preserving fusion. The magnitude of the benefit of the learned merging pipeline depends on the uniformity of the Gaussians predicted by the backbone: with the more uniform AnySplat Gaussians, moment matching performs better and the gap to our method is smaller (Tab. 1).

All further ablations use the AnySplat backbone and report the K = 1 decoder averaged over DL3DV-Bench, MipNeRF360, and Tanks & Temples. Tab. 3 summarizes all variants against the unmerged reference.

Removing the refiner $\mathcal { M } ^ { \mathrm { r e f } }$ degrades all metrics (Tab. 3), indicating that the refiner contributes significantly to overall performance. Disabling cross-view merging slightly improves PSNR at the cost of a higher primitive count, suggesting that cross-view fusion reduces redundancy at the cost of a slight quality drop. Within-view merging however contributes much more to the overall compression.

Superpixel Algorithm Comparison. To evaluate the segmentation quality of different superpixel algorithms across a wide range of segment counts, without the cost of the full pipeline evaluation, we use a proxy metric: each pixel in an image is replaced by the mean color of its superpixel, and the resulting image is compared to the ground truth. This measures how faithfully a given superpixel layout can represent the visual image content, which determines the upper bound on merging quality. We apply this proxy to 1 000 images (100 scenes × 10 views) from the training set and report PSNR against the original pixels. Fig. 7 shows the quality vs. superpixel count trade-off for BASS [35] with uniform seeding, BASS with our Shi-Tomasi corner detector seeding, and SLIC [1]. BASS with Shi-Tomasi corner seeding consistently dominates the other algorithms at all budgets, achieving higher proxy PSNR at equal superpixel count. Based on these results, we evaluate the full pipeline (without retraining) with BASS and Shi-Tomasi seeding at three superpixel size settings, which yield ∼23k, ∼12k, and ∼5k superpixels per image on average.

![](images/57190e2b49c7954bec16672f9d939d82aa0bab033963b2b11425d61e6ca29ee4.jpg)

![](images/5c94334a5b9c5c55f2865280547418f869237b2a679c52b5f5d29ffc7df079ce.jpg)  
Figure 7: Left: Proxy PSNR vs. superpixel count for three segmentation algorithms (100 scenes × 10 views). Each pixel is replaced by its superpixel mean colour and compared to the original image. Right: NVS PSNR vs. relative primitive count for three SP sizes (AnySplat backbone, zero-shot, averaged over all benchmarks). Arrows indicate favorable direction.

Larger superpixels reduce the relative primitive count $r _ { c }$ at the cost of quality, e.g., from 14.14 dB PSNR to 13.97 dB PSNR at roughly half the $r _ { c }$ (Tab. 3). This confirms that finer superpixels better preserve reconstruction quality, at the cost of a higher $r _ { c } .$ with diminishing returns at very small superpixel sizes. Using larger superpixels therefore provides a similar trade-off to using the higher K decoder heads, with the decoder heads being slightly more effective at the same $r _ { c }$ . For example, the K = 2 decoder achieves 14.45 dB PSNR at $r _ { c } =$ 8.9%, while the K = 4 decoder with larger superpixels achieves 14.45 dB PSNR at a similar $r _ { c } = 8 . 0 \%$ . This provides the user with two complementary mechanisms to flexibly adjust the quality–efficiency trade-off according to their needs, both without retraining.

## 6 Conclusion

We present a structure-aware approach for strongly compressing Gaussian splatting representations generated by feed-forward networks. Our superpixel-based primitive merging strategy intelligently consolidates per-pixel Gaussians into a compact, content-adaptive representation. We group Gaussians via saliency-guided superpixel segmentation, encode each group into a Feature Gaussian with a learned encoder, and match and merge Feature Gaussians across views. A level-of-detail decoder provides explicit control over the output primitive count at inference, enabling flexible quality-efficiency trade-offs. The pipeline operates as a backbone-agnostic post-processing module that retains the bulk of the reconstruction quality across three benchmarks and three feed-forward backbones at just $1 / { 2 0 } ^ { \mathrm { t h } }$ of the Gaussians, accepting a slight reduction in fine-detail fidelity in exchange for dramatically improved efficiency. Compared to other reduced-primitive methods, our approach provides more robust and higher quality reconstructions, particularly in sparse-view settings. Additionally, for large viewpoint changes, the merged primitives can even improve some metrics by suppressing common FF artifacts such as semi-transparent fog, or high-frequency noise.

Limitations. Our method depends on the quality of the underlying feed-forward reconstruction; if the initial primitives are of low quality, merging may not recover a good representation. The matching and merging pipeline introduces small computational overhead of 540 ms compared to direct feed-forward inference (details in the supplementary material).

Future work. Promising directions include extensions to dynamic scenes where temporal consistency must be maintained, and replacing the algorithmic superpixel grouping with a learned grouping mechanism that can be optimized jointly with the merging pipeline.

## References

[1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Süsstrunk. SLIC Superpixels Compared to State-of-the-Art Superpixel Methods. IEEE Trans. on Pattern Analysis and Machine Intelligence (TPAMI), 34(11):2274– 2282, 2012. doi: 10.1109/TPAMI.2012.120.

[2] Tomas Akenine-Möller, Eric Haines, Naty Hoffman, Angelo Pesce, Michaël Iwanicki, and Sébastien Hillaire. Real-Time Rendering. CRC Press, 4 edition, 2018. doi: 10. 1201/b22086.

[3] Yinlong Bai, Hongxin Zhang, Sheng Zhong, Junkai Niu, Hai Li, Yijia He, and Yi Zhou. MemGS: Memory-Efficient Gaussian Splatting for Real-Time SLAM. In Proc. of the IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), 2025.

[4] Isabela B. Barcelos, Felipe D.C. Belem, Leonardo D.M. Joao, Zenilton K.G.D. Patrocínio Jr., Alexandre X. Falcao, and Silvio J.F. Guimarães. A Comprehensive Review and New Taxonomy on Superpixel Segmentation. ACM Computing Surveys, 56(8): 1–39, 2024. doi: 10.1145/3643826.

[5] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2022.

[6] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction. In Proc. ofthe IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[7] Yihang Chen, Qianyi Wu, Mengyao Li, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai. Fast Feedforward 3D Gaussian Splatting Compression. In Proc. of the Intl. Conf. on Learning Representations (ICLR), 2025.

[8] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images. In Proc. ofthe Europ. Conf. on Computer Vision (ECCV), 2024.

[9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In Proc. of the Intl. Conf. on Learning Representations (ICLR), 2021.

[10] Guangchi Fang and Bing Wang. Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians. In Proc. of the Europ. Conf. on Computer Vision (ECCV), 2024.

[11] Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren, Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, and Feng Zhao. AnySplat: Feed-Forward 3D Gaussian Splatting from Unconstrained Views. ACM Trans. on Graphics (TOG), 44(6):1–16, 2025. doi: 10.1145/3763326.

[12] Ying Jiang, Chang Yu, Tianyi Xie, Xuan Li, Yutao Feng, Huamin Wang, Minchen Li, Henry Y.K. Lau, Feng Gao, Yin Yang, and Chenfanfu Jiang. VR-GS: A Physical Dynamics-Aware Interactive Gaussian Splatting System in Virtual Reality. In Proc. of the Intl. Conf. on Computer Graphics and Interactive Techniques (SIGGRAPH), 2024.

[13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans. on Graphics (TOG), 42(4):1–14, 2023. doi: 10.1145/3592433.

[14] Bernhard Kerbl, Andréas Meuleman, Georgios Kopanas, Michael Wimmer, Alexandre Lanvin, and George Drettakis. A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets. ACM Trans. on Graphics (TOG), 43(4):1–15, 2024. doi: 10.1145/3658160.

[15] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction. ACM Trans. on Graphics (TOG), 36(4):1–13, 2017. doi: 10.1145/3072959.3073599.

[16] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3D Gaussian Representation for Radiance Field. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[17] Joo Chan Lee, Jong Hwan Ko, and Eunbyung Park. Optimized Minimal 3D Gaussian Splatting. In Proc. of the Conf. on Neural Information Processing Systems (NeurIPS), 2025.

[18] Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, and Yee Whye Teh. Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. In Proc. of the Intl. Conf. on Machine Learning (ICML), 2019.

[19] Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth Anything 3: Recovering the Visual Space from Any Views. arXiv preprint, arXiv:2511.10647, 2025.

[20] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, Xuanmao Li, Xingpeng Sun, Rohan Ashok, Aniruddha Mukherjee, Hao Kang, Xiangrui Kong, Gang Hua, Tianyi Zhang, Bedrich Benes, and Aniket Bera. DL3DV-10K: A Large-Scale Scene Dataset for Deep Learning-Based 3D Vision. In Proc. ofthe IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[21] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In Proc. of the Intl. Conf. on Learning Representations (ICLR), 2019.

[22] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl, Markus Steinberger, Francisco Vicente Carrasco, and Fernando De La Torre. Taming 3DGS: High-Quality Radiance Fields with Limited Resources. In Proc. ofthe Intl. Conf. on Computer Graphics and Interactive Techniques (SIGGRAPH) Asia, 2024.

[23] Sheng Miao, Jiaxin Huang, Dongfeng Bai, Xu Yan, Hongyu Zhou, Yue Wang, Bingbing Liu, Andreas Geiger, and Yiyi Liao. EVolSplat: Efficient Volume-Based Gaussian Splatting for Urban View Synthesis. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2025.

[24] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In Proc. ofthe Europ. Conf. on Computer Vision (ECCV), 2020.

[25] Arthur Moreau, Richard Shaw, Michal Nazarczuk, Jisu Shin, Thomas Tanay, Zhensong Zhang, Songcen Xu, and Eduardo Pérez-Pellitero. Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2026.

[26] Simon Niedermayr, Josef Stumpfegger, and Rüdiger Westermann. Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[27] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, and Bernard Ghanem. Magic123: One Image to High-Quality 3D Object Generation Using Both 2D and 3D Diffusion Priors. In Proc. of the Intl. Conf. on Learning Representations (ICLR), 2024.

[28] Xuanchi Ren, Yifan Lu, Hanxue Liang, Zhangjie Wu, Huan Ling, Mike Chen, Sanja Fidler, Francis Williams, and Jiahui Huang. SCube: Instant Large-Scale Scene Reconstruction Using VoxSplats. In Proc. ofthe Conf. on Neural Information Processing Systems (NeurIPS), 2024.

[29] Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan, Dmitry Lagun, Li Fei-Fei, Deqing Sun, and Jiajun Wu. ZeroNVS: Zero-Shot 360-Degree View Synthesis from a Single Image. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[30] Hanno Scharr. Optimale Operatoren in der digitalen Bildverarbeitung. PhD thesis, Ruprecht-Karls-Universität Heidelberg, 2000.

[31] Jianbo Shi and Jitendra Malik. Normalized Cuts and Image Segmentation. IEEE Trans. on Pattern Analysis and Machine Intelligence (TPAMI), 22(8):888–905, 2000. doi: 10.1109/34.868688.

[32] Jianbo Shi and Carlo Tomasi. Good Features to Track. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 1994.

[33] Stanislaw Szymanowicz, Christian Rupprecht, and Andrea Vedaldi. Splatter Image: Ultra-Fast Single-View 3D Reconstruction. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[34] Adam Tonderski, Carl Lindström, Georg Hess, William Ljungbergh, Lennart Svensson, and Christoffer Petersson. NeuRAD: Neural Rendering for Autonomous Driving. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[35] Roy Uziel, Meitar Ronen, and Oren Freifeld. Bayesian Adaptive Superpixel Segmentation. In Proc. ofthe IEEE/CVF Intl. Conf. on Computer Vision (ICCV), 2019.

[36] Haishan Wang, Mohammad Hassan Vali, and Arno Solin. Smol-GS: Compact Representations for Abstract 3D Gaussian Splatting. arXiv preprint, arXiv:2512.00850, 2025.

[37] Weijie Wang, Donny Y. Chen, Zeyu Zhang, Duochao Shi, Akide Liu, and Bohan Zhuang. ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS. In Proc. ofthe Conf. on Neural Information Processing Systems (NeurIPS), 2025.

[38] Weijie Wang, Yeqing Chen, Zeyu Zhang, Hengyu Liu, Haoxiao Wang, Zhiyuan Feng, Wenkang Qin, Zheng Zhu, Donny Y. Chen, and Bohan Zhuang. VolSplat: Rethinking Feed-Forward 3D Gaussian Splatting with Voxel-Aligned Prediction. arXiv preprint, arXiv:2509.19297, 2025.

[39] Yiming Wang, Lucy Chai, Xuan Luo, Michael Niemeyer, Manuel Lagunas, Stephen Lombardi, Siyu Tang, and Tiancheng Sun. Learning Efficient Fuse-and-Refine for Feed-Forward 3D Gaussian Splatting. In Proc. of the Conf. on Neural Information Processing Systems (NeurIPS), 2025.

[40] Yunsong Wang, Tianxin Huang, Hanlin Chen, and Gim Hee Lee. FreeSplat: Generalizable 3D Gaussian Splatting Towards Free-View Synthesis of Indoor Scenes. In Proc. of the Conf. on Neural Information Processing Systems (NeurIPS), 2024.

[41] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE Trans. on Image Processing, 13(4):600–612, 2004. doi: 10.1109/TIP.2003.819861.

[42] Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang. PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[43] Haofei Xu, Daniel Barath, Andreas Geiger, and Marc Pollefeys. ReSplat: Learning Recurrent Gaussian Splatting. arXiv preprint, arXiv:2510.08575, 2025.

[44] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. DepthSplat: Connecting Gaussian Splatting and Depth. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2025.

[45] Yinghao Xu, Zifan Shi, Wang Yifan, Hansheng Chen, Ceyuan Yang, Sida Peng, Yujun Shen, and Gordon Wetzstein. GRM: Large Gaussian Reconstruction Model for Efficient 3D Reconstruction and Generation. In Proc. of the Europ. Conf. on Computer Vision (ECCV), 2024.

[46] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong Wang, and Xuelong Li. GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[47] Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys, Ming-Hsuan Yang, and Songyou Peng. No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images. arXiv preprint, arXiv:2410.24207, 2024.

[48] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting. In Proc. ofthe Europ. Conf. on Computer Vision (ECCV), 2024.

[49] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2018.

[50] Shengjun Zhang, Xin Fei, Fangfu Liu, Haixu Song, and Yueqi Duan. Gaussian Graph Network: Learning Efficient and Generalizable Gaussian Representations from Multi-view Images. In Proc. of the Conf. on Neural Information Processing Systems (NeurIPS), 2024.

[51] Yuhang Zhang, Richard Hartley, John Mashford, and Stewart Burn. Superpixels via Pseudo-Boolean Optimization. In Proc. of the IEEE/CVF Intl. Conf. on Computer Vision (ICCV), 2011.

[52] Tingyu Zhao, Bo Peng, Zhenguang Zhang, Daipeng Yang, and Xi Wu. Content-Aware Dynamic Superpixel Segmentation. In Proc. ofthe IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025.

[53] Yuhang Zheng, Xiangyu Chen, Yupeng Zheng, Songen Gu, Runyi Yang, Bu Jin, Pengfei Li, Chengliang Zhong, Zengmao Wang, Lina Liu, Chao Yang, Dawei Wang, Zhen Chen, Xiaoxiao Long, and Meiqing Wang. GaussianGrasper: 3D Language Gaussian Splatting for Open-Vocabulary Robotic Grasping. IEEE Robotics and Automation Letters (RA-L), 9(9):7827–7834, 2024. doi: 10.1109/LRA.2024.3432348.

[54] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo Magnification: Learning View Synthesis Using Multiplane Images. ACM Trans. on Graphics (TOG), 37(4):1–12, 2018. doi: 10.1145/3197517.3201323.

[55] Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang. DrivingGaussian: Composite Gaussian Splatting for Surrounding Dynamic Autonomous Driving Scenes. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[56] Matthias Zwicker, Hanspeter Pfister, Jeroen van Baar, and Markus Gross. EWA Volume Splatting. In Proc. ofthe IEEE Visualization Conf. (VIS), 2001.