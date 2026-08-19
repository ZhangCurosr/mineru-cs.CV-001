# GenRec: Knowing Where to Reconstruct and Where to Generate

Ata Çelen<sup>1</sup> Jaewoo Jung<sup>1,4</sup> Federico Tombari<sup>2</sup> Marc Pollefeys<sup>1,3</sup> Sunghwan Hong<sup>1</sup> Michael Niemeyer<sup>2</sup> Daniel Barath<sup>1,2</sup> <sup>1</sup>ETH Zürich <sup>2</sup>Google <sup>3</sup>Microsoft <sup>4</sup>KAIST

https://atcelen.github.io/GenRec/

## Abstract

Generative novel view synthesis from sparse input images is rarely all reconstruction or all generation: pixels visible in some source view have a unique correct value modulated only by view-dependent shading, while pixels in disocclusions or beyond the captured volume admit a distribution of plausible completions. Existing generative novel-view-synthesis methods conflate these regimes under a single uniform loss, blurring the line between geometric fidelity and creative hallucinations even when scene geometry is injected through warped point clouds or projected depth. We introduce GenRec, a multi-view flow matching model that builds the reconstruction–generation split directly into its architecture, supervision, and gradient flow. Guided by an observation mask derived from the source cameras and a monocular depth estimator, a flow matching backbone jointly denoises RGB and scene-coordinate maps across all target views, while a pixel-space refinement stage restores high-frequency detail on observed pixels; the same mask gates supervision so regression signals do not contaminate the generative prior. Across RealEstate10K, DL3DV-10K, and Mip-NeRF 360, in both single-view extrapolation and two-view interpolation, GenRec attains the best reconstruction fidelity in observed regions while also surpassing purely generative baselines on perceptual quality in unobserved ones, showing the effectiveness of our approach.

## 1 Introduction

Consider two scenarios: a robot navigates an indoor space from a handful of casually captured photographs, while a headset user wanders freely through the same scene in mixed reality. The robot must plan collision-free trajectories around furniture seen from a single angle, judge whether a gap between two chairs is wide enough to traverse, and anticipate what lies beyond a half-open door before committing to a turn. The headset user expects every captured chair and painting to remain exactly in its place from any angle, and any corridor they step into beyond what was captured to fill in seamlessly around them. All of these capabilities reduce to a single problem: generative novel-view synthesis with sparse input views. Yet within a single synthesized view, different pixels carry very different burdens. Where the new view overlaps with what the camera already observed, the system must reconstruct the scene faithfully: the same wall, the same texture, modulated only by view-dependent effects (e.g., specular highlights) arising from the angular shift. Where the new view ventures into unobserved regions (around a corner, behind an occluder, or beyond the captured volume), the inputs alone do not pin down a single answer, and the system must infer the most likely completion consistent with the visible evidence and a learned prior over the world.

Today’s 3D scene representations, such as 3D Gaussian Splatting (3DGS) [20] and NeRFs [29], only handle the first half. They are interpolators: dense observations yield photorealistic in-between views, but they have nothing to say about the parts of the world the camera never looked at, and degrade sharply as input becomes sparse [46, 30]. To bridge this gap, a recent line of generative reconstruction methods leverages image and video diffusion priors [49, 12, 56, 34, 59, 21, 42, 1, 25] to hallucinate plausible novel views from one or a few input views, which can then be fed back as pseudo-observations to a 3DGS or NeRF [48]. These methods differ mainly in how they tell the generator where to look: pose-only conditioning via Plücker rays [13, 12, 59, 18] suffers from metricscale ambiguity, while warp-based conditioning that injects forward-projected depth or point-cloud renders [56, 34, 11] grounds the model in the scene but inherits warp artifacts.

For all these methods, the generator must produce every pixel under a single distribution-matching loss. Yet, this overlooks a basic structural property of the task. A novel view is rarely all generation or all reconstruction; it is a combination of both. Pixels on a surface visible in the captured scene have a unique correct answer, dictated by geometry and appearance. Pixels in disocclusions or beyond the captured volume have many plausible answers, and the model must commit to a coherent one. Asking a single network to satisfy both regimes with the same supervisory signal forces a compromise: a loss that fits a distribution where many answers are valid penalizes fidelity where only one is.

We propose GenRec, which incorporates this separation directly into its architecture: the network reconstructs observed regions and infers unobserved ones. Given the input cameras and an offthe-shelf depth estimator, we forward-project source pixels into each target view to obtain a binary observed/unobserved mask. The generative backbone produces a full target view; pixel-space refinement then operates on the observed pixels, leveraging warped input evidence and learned 3D correspondences to recover the high-frequency detail suppressed by latent flow matching. These modules do not share parameters with the backbone and their gradients do not propagate into it. Thus, the distribution-matching objective is no longer required to anchor pixels whose answer is uniquely determined by geometry; the refinement modules supply this fidelity signal on observed pixels alone.

Our contributions are:

• A geometry-grounded decomposition of generative NVS that processes observed/unobserved pixels with separate modules and losses, governed by a per-pixel observation mask.

• A multi-viewflow matching backbone that jointly denoises RGB and scene-coordinate maps, yielding cross-view consistency and a 3D correspondence signal in a single pass.

• A pixel-space reconstruction branch with sparse 3D cross-attention that recovers detail lost to latent compression without disturbing the generative prior in unobserved regions.

We validate GenRec on RealEstate10K, DL3DV-10K, and Mip-NeRF 360 under both single-view extrapolation and two-view interpolation. Across these settings, GenRec achieves the best reconstruction fidelity in observed regions while surpassing purely generative baselines on perceptual quality in unobserved ones, demonstrating that the separation of these two tasks yields both better reconstruction and generation quality. At the same time, our model is two orders of magnitude faster at inference than the strongest baseline.

## 2 Related Work

Novel view synthesis from sparse inputs. Neural Radiance Fields [29] and 3D Gaussian Splatting [20], together with their follow-ups for unbounded scenes [2] and faster rendering [3], set the modern bar for in-distribution view synthesis. They are, however, fundamentally interpolators: they require dense captures, are optimized per scene, and degrade once the camera leaves the convex hull of the inputs [46]. Geometric priors such as depth smoothness [30] or monocular cues [57] partially mitigate the sparse-view regime by regularizing the optimization. Feed-forward alternatives sidestep per-scene fitting altogether: PixelNeRF [55] conditions a radiance field on image features in a single forward pass, and recent work predicts 3D Gaussians directly from sparse views [4, 5, 53, 43, 52, 54], while large transformer synthesizers such as LVSM [18] and projection-conditioned variants [51] dispense with explicit 3D representations entirely. Across this entire family of methods, the output is deterministic, which is sufficient when the target view is well-constrained but breaks down when large regions must be imagined; the gap that the next family fills.

Pose-only multi-view diffusion. These methods condition a multi-view diffusion model purely on the target camera, typically through Plücker ray embeddings [40, 13]. Beginning with score distillation [33] and image-conditioned single-object generators [26, 38, 39], it has scaled to scenelevel diffusion that produces tens of consistent novel views in a single denoising pass [49, 12, 59], and to joint RGB-plus-pointmap sampling for fast 3D scene generation [42]. Pose-only conditioning is flexible but struggles with the metric-scale ambiguity of ray embeddings. Crucially, the content of the scene actually visible in the input is not given any privileges: it is regenerated from scratch like the rest of the frame.

![](images/fa351f601b1ca99d8229a6d6d0d330ef3f32bd7b2ead8671e98be001b4baed8e.jpg)  
Figure 1: Overview of GenRec (right) vs. prior generative reconstruction methods (left). Prior methods condition multi-view image or video models on warped point clouds or Plücker rays, supervising output pixels with one generative loss regardless of source coverage. GenRec factors the task by an observation mask: a generative backbone predicts every pixel, and a pixel-space reconstruction module refines the observed ones under a dedicated loss. The two pathways share neither parameters nor gradients, preserving the generative prior in unobserved regions.

Warp- and geometry-conditioned generation. Another line of work explicitly injects scene geometry into the diffusion model. The most common recipe takes a depth-warped point cloud or RGB-D rendering of the source views and feeds it as a frame-aligned signal to a video diffusion backbone [56, 21, 1, 25]. GEN3C [34] formalizes this with a persistent 3D cache that is rendered into the model for precise camera control, and Geometry Forcing [47] pushes the video backbone toward 3D-consistent latents via auxiliary feature alignment. FlowR [11] replaces video diffusion with multi-view flow matching that maps renderings of a sparse 3DGS to renderings of a dense one. A complementary thread bridges geometric foundation models with diffusion latents [45, 22, 16, 17], and a third uses generative priors to clean or harmonize existing reconstructions [48]. These methods inject geometry where pose-only models cannot, but the supervision they apply remains uniform across the output: regions where the warped condition is reliable and regions where it is empty are pulled toward the same target by the same loss.

Region-aware supervision. Binary masks do appear in the geometry-coupled literature, but always at the periphery of the generative loop: dropping parts of the 3D representation during training [50], marking unwarped pixels in the input conditioning [56], or reweighting rendering losses in the downstream 3DGS optimization. None of these methods uses a per-pixel observation signal to split reconstruction from generation inside the generator itself. A related observation, recently sharpened by pixel-space NVS diffusion [8], is that VAE latents impose a photometric fidelity ceiling on the detail that observed regions can carry through. GenRec addresses both: a shared observation mask binds architecture and loss, and a dedicated pixel-space branch restores detail where the inputs constrain it while leaving the generative prior untouched elsewhere.

## 3 Method

Problem statement. We are given $N _ { \mathrm { s r c } }$ posed source views $\{ ( I _ { j } ^ { \mathrm { s r c } } , \pi _ { j } ^ { \mathrm { s r c } } ) \} _ { j = 1 } ^ { N _ { \mathrm { s r c } } }$ with images $I _ { j } ^ { \mathrm { s r c } } \in$ $[ 0 , 1 ] ^ { 3 \times H \times W }$ , and N target poses $\{ \pi _ { i } \} _ { i = 1 } ^ { N }$ . The task is to synthesize the corresponding target images $\{ \hat { I } _ { i } ^ { \mathrm { r e c } } \} _ { i = 1 } ^ { N }$ , with $\hat { I } _ { i } ^ { \mathrm { r e c } } \in [ 0 , 1 ] ^ { 3 \times H \times W }$ . As preprocessing, we run an off-the-shelf monocular depth prediction network on each source view and forward-warp the source RGB and the back-projected source 3D points into every target frame. This yields, for each target view i, three precomputed signals: a soft observation mask $\mathcal { \bar { O } } _ { i } \in \mathsf { \bar { \Pi } } [ 0 , 1 ] ^ { 1 \times H \times W }$ recording which target pixels receive evidence, and permodality warped renderings $\mathring { W _ { i } ^ { I } } \in [ 0 , 1 ] ^ { 3 \times H \times W }$ of the source RGB and $W _ { i } ^ { S C } \in [ - 1 , 1 ] ^ { 3 \times H \times W }$ of the source scene coordinates.

![](images/40f08673b609bbb095b3ac70da0194a88be22ea3c702604581df64178a5c1b56.jpg)  
Figure 2: Overview of GenRec. Given posed input views and target poses, monocular depth and forward warping produce per-target observation masks and warped renders. These condition a (I) multi-view flow matching backbone, which jointly denoises RGB and scene-coordinate latents through cross-view and cross-modal attention. A (II) reconstruction branch then refines only the observed pixels via sparse 3D attention guided by the predicted scene coordinates, while the observation mask gates gradients so the generative prior is preserved in unobserved regions.

Generation–reconstruction split. Not all pixels in a target view are equally constrained by the source views. Pixels with source coverage have a near-deterministic target: the source observation collapses the predictive distribution to a tight mode varying only with viewing direction. Pixels without coverage (in disocclusions or beyond the captured frustum) have a stochastic target: only a posterior over completions consistent with the visible scene is identifiable. These regimes demand structurally different supervision (regression in the first, distribution matching in the second), and the latent generative path discards the high-frequency content the deterministic regime must preserve. GenRec therefore factors the model along $O _ { i }$ at every level (Fig. 2). A multi-view flow-matching backbone (Sec. 3.1) generates every pixel of every target view, handling the distributional regime. A lightweight pixel-space reconstruction stage (Sec. 3.2) then corrects the observed pixels, supplying the photometric signal the latent path cannot provide. The mask further isolates gradient flow: reconstruction gradients reach only the reconstruction parameters, never the backbone (Sec. 3.3).

## 3.1 Multi-View Flow Matching Backbone

The backbone is a denoising network $v _ { \theta }$ trained under the Conditional Flow Matching framework. It transports Gaussian noise $x _ { 0 } \sim \mathcal { N } ( 0 , I )$ to a clean latent $x _ { 1 }$ along the path $x _ { t } = \left( 1 - t \right) x _ { 0 } + t x _ { 1 } . \mathrm { ~ A ~ }$ single denoising trajectory operates jointly on all N target views and two co-registered modalities, producing per-view, per-modality clean-end estimates $\overline { { \boldsymbol { x } } } _ { 1 , i } ^ { m } \in \mathbb { R } ^ { C \times h \times w }$ for $i \in \{ 1 , \ldots , N \}$ and $m \in \{ I , S C \}$ . The refinement stage of Sec. 3.2 consumes these latents.

Joint RGB and scene-coordinate denoising. Each target view comprises two co-registered modalities: the RGB image and a scene-coordinate map $\overset { \triangledown } { \boldsymbol { S } } \overset { \triangledown } { \boldsymbol { C } } \in [ - 1 , 1 ] ^ { 3 \times \mathbf { \breve { H } } \times \boldsymbol { W } }$ encoding the normalized world-space $( x , y , z )$ position of every pixel. Joint denoising enables the geometry-aware refinement of Sec. 3.2: the decoded scene coordinates $\widehat { S C } _ { i } = { \cal D } _ { S C } ( \hat { x } _ { 1 , i } ^ { S C } )$ ) supply the 3D positions used by its cross-attention. Both modalities use pretrained VAEs $( \mathcal { E } _ { I } , \mathcal { D } _ { I } )$ and $( \mathcal { E } _ { S C } , \mathcal { D } _ { S C } )$ with shared channel count C and spatial resolution $h \times w$ . Following Fang et al. [10], we extend each transformer block of $v _ { \theta }$ along two axes: (i) self-attention is rewired into cross-view attention over all N target tokens, yielding multi-view consistency in one forward pass; (ii) a zero-initialized cross-modal attention layer couples the RGB and scene-coordinate tokens of the same view. The zero initialization preserves the pretrained prior at training start; the modalities are paired as their joint distribution is learned.

Conditioning on geometric evidence. At each flow time $t \in [ 0 , 1 ]$ , the network input is a tensor $\mathbf { Z } _ { \mathrm { i n } } ( t ) \in \mathbb { R } ^ { \breve { N } \times 2 \times \breve { C } _ { \mathrm { i n } } \times h \times w }$ where the second axis indexes the two modalities and $C _ { \mathrm { i n } }$ is the permodality concatenated channel count. For every view i and modality $m \in \{ I , S C \}$ ,

$$
\mathbf { Z } _ { \mathrm { i n } } ^ { i , m } ( t ) \ = \ \big [ x _ { t , i } ^ { m } \big \| { \mathbf { r } } _ { i } \big \| \tilde { O } _ { i } \big \| \mathcal { E } _ { m } \big ( W _ { i } ^ { m } \big ) \big ] ,\tag{1}
$$

where ∥ denotes channel-wise concatenation; $x _ { t , i } ^ { m }$ is the noisy latent at time $t ; { \bf r } _ { i }$ is a Plücker ray map of the target, embedded by a small patch ViT [7] whose tokens are folded back to the latent grid; ${ \tilde { O } } _ { i }$ is the observation mask area-pooled to $h \times w ;$ and $\mathcal { E } _ { m } ( W _ { i } ^ { m } )$ encodes the modality-specific warp. This tells the backbone where evidence is available (via ${ \tilde { O } } _ { i } )$ , what that evidence is (via the encoded warps), and which 3D ray each pixel corresponds to (via $\mathbf { r } _ { i } )$ , even in regions it must hallucinate.

Flow-matching parameterization. We initialize $v _ { \theta }$ from a pretrained v-prediction latent diffusion model [35] and fine-tune it under Conditional Flow Matching with the Diff2Flow reparameterization of Schusterbauer et al. [37], which converts the v-prediction into a flow velocity in closed form. This preserves the pretrained weights (cross-modal layer is zero-initialized) while letting us train straight flows that admit fast Euler sampling at inference [27]. The clean estimate is recovered in closed form as $\hat { x } _ { 1 , i } ^ { m } = x _ { t , i } ^ { m } + \left( 1 - t \right) v _ { \theta } ( \mathbf { Z } _ { \mathrm { i n } } ( t ) , \bar { t } ) _ { i , m } .$ where $( \cdot ) _ { i , m }$ selects the per-view, per-modality slice.

## 3.2 Pixel-Space Refinement of Observed Regions

The refinement stage takes the backbone’s clean latents $\{ \hat { x } _ { 1 , i } ^ { I } , \hat { x } _ { 1 , i } ^ { S C } \} _ { i = 1 } ^ { N }$ together with the precomputed mask $O _ { i }$ and warps $W _ { i } ^ { I }$ , and produces the final RGB outputs $\{ \hat { I } _ { i } ^ { \mathrm { r e c } } \} _ { i = 1 } ^ { N } .$ . Latent generative models impose a photometric ceiling on what can be recovered through the VAE decoder (acceptable for hallucinated content, but limiting where evidence already constrains the answer), so we restrict refinement to observed pixels. The stage has two cooperating components: a geometry-aware decoder adapter that injects warped RGB at the latent→pixel boundary and outputs an intermediate image $\hat { I } _ { i } .$ and a sparse 3D cross-attention branch that adds an explicitly geometry-aware residual $\Delta _ { i }$ on top.

Geometry-Aware Decoder Adapter. We adapt the RGB decoder $\mathcal { D } _ { I }$ , which maps the latent $\hat { x } _ { 1 , i } ^ { I }$ to pixels; both VAEs are otherwise frozen. The adaptation has two parts. Following Parmar et al. [31], Wu et al. [48], zero-initialized 1×1 skip convolutions additively inject features from $\mathcal { E } _ { I } ( W _ { i } ^ { I } )$ before each up-block of $\mathcal { D } _ { I }$ . A LoRA adapter [15] on the decoder’s convolutions and self-attention projections is jointly trained, allowing the decoder to consume these injected features. We denote the resulting decoder $\widetilde { \cal D } _ { I }$ , with output:

$$
\hat { I } _ { i } \ = \ \widetilde { \mathcal { D } } _ { I } \big ( \hat { x } _ { 1 , i } ^ { I } , W _ { i } ^ { I } \big ) \ \in \ [ 0 , 1 ] ^ { 3 \times H \times W } .
$$

The two adaptations are complementary. Locally, the skip connections copy fine-grained photometric detail from the warped conditioning, recovering high-frequency content the latent representation cannot preserve. Globally, the LoRA weights affect every pixel and shift the decoder’s output distribution toward sharper samples, so perceptual benefits extend beyond the observation mask. Supervision is gated by ${ \bar { O } } _ { i }$ (Sec. 3.3) to prevent drift in disoccluded regions.

Sparse 3D Reconstruction Branch. The reconstruction branch consumes the decoder-adapter output $\hat { I } _ { i }$ together with the warped RGB $W _ { i } ^ { I }$ , the mask $O _ { i }$ , the source images $\{ I _ { j } ^ { \mathrm { s r c } } \} _ { j = 1 } ^ { N _ { \mathrm { s r c } } }$ , and the decoded scene coordinates $\widehat { S C } _ { i } = { \cal D } _ { S C } ( \widehat { x } _ { 1 , i } ^ { S C } )$ ). It predicts a pixel-space residual $\Delta _ { i } \in \mathbb { R } ^ { 3 \times H \times W }$ that is added back only on observed pixels, producing the final image

$$
\hat { I } _ { i } ^ { \mathrm { r e c } } = \hat { I } _ { i } + O _ { i } \odot \Delta _ { i } .
$$

Whereas the decoder adapter copies coarse appearance through skip connections, this branch reasons explicitly about 3D correspondences: each target pixel attends to a small set of geometrically nearest neighbours (first, in the source views, then across target views), with $\widehat { S C } _ { i }$ supplying the 3D positions used to define the neighbour sets. The branch is composed of two sparse cross-attention blocks followed by FiLM-conditioned residual blocks [32] whose scale and shift are produced from the detached temporal embedding of the backbone, so the branch modulates its behaviour with the flow time t. The output projection of every cross-attention block and final convolution are zero-initialized, so $\Delta _ { i } = { \bf 0 }$ at the start of training and the branch contributes only as it acquires useful signal.

Sparse source-to-target attention. The first attention block operates at full pixel resolution. For each target view i, queries and values are built from the per-view tensor $[ \hat { I } _ { i } \parallel W _ { i } ^ { I } \parallel O _ { i } ]$ , and keys are produced by encoding the source images $\{ I _ { j } ^ { \mathrm { s r c } } \}$ with a VAE-initialized encoder. To avoid quadratic cost across all source pixels, we precompute, for each target pixel n, the indices of its K geometrically nearest source pixels using $\widehat { S C } _ { i }$ and the source point cloud, and restrict attention to that fixed neighbour set. Indexing those neighbours by $k \in \{ 1 , \ldots , K \}$ , we form the attention logits:

$$
\mathrm { a t t n } _ { n , k } \ = \ \frac { Q _ { n } ^ { \top } K _ { n , k } } { \sqrt { d } } \ + \ \log w _ { n , k } , \qquad w _ { n , k } \ = \ \frac { ( d _ { n , k } + \varepsilon ) ^ { - 1 } } { \sum _ { k ^ { \prime } } ( d _ { n , k ^ { \prime } } + \varepsilon ) ^ { - 1 } } ,\tag{2}
$$

where d is the head dimension, $\mathbf { p } _ { n } ^ { \mathrm { t g t } } \in \mathbb { R } ^ { 3 }$ is the target-pixel 3D position read from $\widehat { S C } _ { i } , \mathbf { p } _ { k } ^ { \mathrm { s r c } } \in \mathbb { R } ^ { 3 }$ is the 3D position of its k-th source neighbour, and $\bar { d } _ { n , k } = \| \mathbf { p } _ { n } ^ { \mathrm { t g t } ^ { * } } - \mathbf { p } _ { k } ^ { \mathrm { s r c } } \| _ { 2 }$ . Since $\begin{array} { r } { \sum _ { k } w _ { n , k } = 1 } \end{array}$ , the additive log $w _ { n , k }$ acts as a soft logit prior favouring geometrically close correspondences while still allowing evidence to override the prior; invalid slots are masked $\mathrm { t o } - \infty$ before the softmax.

Sparse cross-target attention. Per-view source-to-target attention may produce refinements that disagree where different target frames observe the same surface. A second attention block re-imposes consistency by reusing the same operator, but with the neighbour set now constructed across target views: for each target pixel n in view i, we use $\widehat { S C } _ { i }$ and $\widehat { S C } _ { i ^ { \prime } \neq i }$ to retrieve the K geometrically nearest pixels in the other target views. Queries come from the residual stream of the source-to-target stage, and keys and values are produced by re-encoding the per-view tensor $[ \hat { I } _ { i ^ { \prime } } \parallel W _ { i ^ { \prime } } ^ { I } \parallel O _ { i ^ { \prime } } ]$ from each other view. Eq. (2) is reused unchanged, with $d _ { n , k }$ now measuring inter-target distance.

## 3.3 Training and Inference

Decoupled optimization. We train the model in two phases. In the first phase, only the multi-view flow-matching backbone is optimized, using the standard Conditional Flow Matching loss of Eq. (9), denoted ${ \mathcal { L } } _ { \mathrm { C F M } }$ . In the second phase, the backbone is frozen and we train the decoder adapter and the reconstruction branch on its detached outputs, so no gradient from this phase can reach the backbone parameters. To prevent the adapters from drifting in disoccluded regions, gradients through the global LPIPS loss [58] are routed only through observed pixels using the masked stop-gradient construction:

$$
\tilde { Y } _ { i } = O _ { i } \odot Y _ { i } + ( 1 - O _ { i } ) \odot \mathrm { s g } ( Y _ { i } ) ,\tag{3}
$$

applied to each output of interest $Y _ { i } \in \{ \hat { I } _ { i } , \hat { I } _ { i } ^ { \mathrm { r e c } } \}$

Refinement losses. The decoder-adapter output is supervised with a masked $\ell _ { 1 }$ photometric term and a full-image LPIPS term gated by Eq. (3),

$$
\mathcal { L } _ { \mathrm { d e c } } = \sum _ { i = 1 } ^ { N } \Big [ \lambda _ { 1 } \big \lVert O _ { i } \odot ( \hat { I } _ { i } - I _ { i } ) \big \rVert _ { 1 } + \lambda _ { \mathrm { p } } \mathcal { L } _ { \mathrm { L P I P S } } \big ( \widetilde { I } _ { i } , I _ { i } \big ) \Big ] ,\tag{4}
$$

and the refined output by an analogous loss with an additional $\ell _ { 1 }$ regularizer on $\Delta _ { i }$ that keeps the refinement small by default,

$$
\mathcal { L } _ { \mathrm { r e c } } \ = \ \sum _ { i = 1 } ^ { N } \Big [ \lambda _ { 1 } \big \lVert O _ { i } \odot \big ( \hat { I } _ { i } ^ { \mathrm { r e c } } - I _ { i } \big ) \big \rVert _ { 1 } \ + \ \lambda _ { \mathrm { p } } \mathcal { L } _ { \mathrm { L P I P S } } \big ( \tilde { I } _ { i } ^ { \mathrm { r e c } } , I _ { i } \big ) \ + \ \lambda _ { \Delta } \big \lVert \Delta _ { i } \big \rVert _ { 1 } \ \Big ] .\tag{5}
$$

The second phase optimizes $\mathcal { L } _ { \mathrm { d e c } } + \mathcal { L } _ { \mathrm { r e c } }$ over the decoder adapter and reconstruction branch with the backbone kept fixed. Combined with the O<sub>i</sub>-gating in Eq. (3), this two-phase schedule implements the generation–reconstruction split at the level of every gradient that touches the network.

Iterative-denoising supervision for the refinement stage. A subtle but important design choice concerns which clean-end estimate $\hat { x } _ { 1 , i } ^ { I }$ is used to train the refinement stage. The naive recipe samples $t \in [ 0 , 1 ]$ , forms the linear interpolant $x _ { t , i } ^ { I } = ( 1 - t ) x _ { 0 , i } ^ { I } + t x _ { 1 , i } ^ { I }$ , and feeds the corresponding one-step estimate $\hat { x } _ { 1 , i } ^ { I } = x _ { t , i } ^ { I } + ( 1 - t ) v _ { \theta } ( \mathbf { Z } _ { \mathrm { i n } } ( t ) , t ) _ { i , I }$ to the refinement stage. This is misaligned with test-time behaviour: at inference, the refinement stage operates on a $\hat { x } _ { 1 , i } ^ { I }$ produced from a multi-step Euler trajectory that has accumulated errors, whereas $x _ { t , i } ^ { I }$ lies exactly on the noise–data line. We instead sample $t \in [ 0 , 1 ]$ , integrate the flow from pure noise up to time t to obtain a trajectory state $\hat { x } _ { t , i } ^ { I }$ , and supervise the refinement stage on $\hat { x } _ { 1 , i } ^ { I } = \hat { x } _ { t , i } ^ { \prime } + ( 1 - t ) \mathbf { \bar { \nu } } v _ { \theta } ( \mathbf { Z } _ { \mathrm { i n } } ( t ) , t ) _ { i , I }$

Inference. Given source views and target poses, we precompute $O _ { i } , W _ { i } ^ { I } , W _ { i } ^ { S C }$ , and the geometric neighbour indices. Starting from $x _ { 0 , i } ^ { m } \sim \bar { \mathcal { N } } ( 0 , I )$ , we run T Euler steps of the joint RGB+SC flow in a single multi-view denoising trajectory, yielding $\hat { x } _ { 1 , i } ^ { I }$ and $\hat { x } _ { 1 , i } ^ { S C }$ . Decoding $\hat { x } _ { 1 , \ i } ^ { I }$ <sub>i</sub> through $\widetilde { \cal D } _ { I }$ produces ${ \hat { I } } _ { i } ;$ a single forward pass through the reconstruction branch then yields $\Delta _ { i }$ and the final $\hat { I } _ { i } ^ { \mathrm { r e c } }$ . Both stages share the observation mask $O _ { i } ,$ so the generation–reconstruction split holds at test time as well.

## 4 Experiments

We organize our experiments into four groups. First, we evaluate GenRec’s extrapolation capabilities in the single-view conditioned setting. Second, we assess its interpolation abilities in the two-view conditioned setting, where target views lie between the two source views. Third, we measure the multi-view consistency of the generated frames using 3D-based metrics. Finally, we conduct ablation studies that validate our design choices and quantify the impact of our observation-based formulation on overall performance.

Table 1: Quantitative results on RealEstate10K in the single-view extrapolation setting. Global metrics are computed over the full target frame; Observed restricts them to pixels that receive source evidence under the shared observation mask, and Unobserved to pixels that do not. Inference time is reported per scene. Best results in bold.
<table><tr><td rowspan="2">Method</td><td colspan="4">Global</td><td colspan="2">Observed</td><td colspan="2">Unobserved</td><td rowspan="2">|Inf. Time</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>FID↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>CameraCtrl [13]</td><td>13.40</td><td>0.4424</td><td>0.4761</td><td>57.42</td><td>15.53</td><td>0.4568</td><td>12.40</td><td>0.3935</td><td>~12s</td></tr><tr><td>ViewCrafter [56]</td><td>13.29</td><td>0.4792</td><td>0.4293</td><td>46.27</td><td>15.93</td><td>0.5196</td><td>11.83</td><td>0.3937</td><td>~200s</td></tr><tr><td>SEVA [59]</td><td>13.50</td><td>0.4839</td><td>0.4674</td><td>41.64</td><td>15.97</td><td>0.4999</td><td>12.40</td><td>0.4189</td><td>~20s</td></tr><tr><td>GF [47]</td><td>13.11</td><td>0.3806</td><td>0.4945</td><td>54.58</td><td>15.56</td><td>0.3986</td><td>11.96</td><td>0.3125</td><td>~40s</td></tr><tr><td>LVSM [18]</td><td>13.81</td><td>0.4094</td><td>0.5186</td><td>57.89</td><td>16.32</td><td>0.4224</td><td>12.61</td><td>0.3435</td><td>~1s</td></tr><tr><td>GLD [17]</td><td>13.73</td><td>0.4232</td><td>0.4699</td><td>44.29</td><td>16.07</td><td>0.4533</td><td>12.82</td><td>0.3765</td><td>~42s</td></tr><tr><td>Gen3R [16]</td><td>13.29</td><td>0.4747</td><td>0.5107</td><td>49.46</td><td>15.82</td><td>0.4862</td><td>12.14</td><td>0.4044</td><td>~90s</td></tr><tr><td>GEN3C [34]</td><td>15.24</td><td>0.5289</td><td>0.3624</td><td>35.68</td><td>18.13</td><td>0.5689</td><td>13.63</td><td>0.4529</td><td>~1200s</td></tr><tr><td>Ours</td><td>17.05</td><td>0.6071</td><td>0.3220</td><td>33.09</td><td>20.85</td><td>0.6738</td><td>14.18</td><td>0.4743</td><td>~11s</td></tr></table>

Table 2: Quantitative Results on DL3DV-10K. We compare our method with previous works on single-view conditioned generation. We follow the same evaluation protocol as in Table 1.
<table><tr><td></td><td colspan="4">Global</td><td colspan="2">Observed</td><td colspan="2">Unobserved</td><td></td></tr><tr><td>Method</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td><td>FID↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>PSNR ↑</td><td>SSIM↑</td><td>Inf. Time</td></tr><tr><td>CameraCtrl [13]</td><td>13.26</td><td>0.3346</td><td>0.4893</td><td>69.49</td><td>13.45</td><td>0.3418</td><td>12.76</td><td>0.2957</td><td>~12s</td></tr><tr><td>ViewCrafter [56]</td><td>17.57</td><td>0.5302</td><td>0.3092</td><td>46.36</td><td>18.75</td><td>0.5659</td><td>15.36</td><td>0.4282</td><td>~200s</td></tr><tr><td>SEVA [59]</td><td>14.58</td><td>0.4300</td><td>0.4359</td><td>43.51</td><td>14.86</td><td>0.4401</td><td>13.92</td><td>0.3819</td><td>~20s</td></tr><tr><td>GLD [17]</td><td>15.54</td><td>0.4001</td><td>0.3979</td><td>51.85</td><td>16.10</td><td>0.4184</td><td>14.33</td><td>0.3368</td><td>~42s</td></tr><tr><td>Gen3R [16]</td><td>13.74</td><td>0.3754</td><td>0.5173</td><td>130.10</td><td>14.02</td><td>0.3886</td><td>12.95</td><td>0.3048</td><td>~90s</td></tr><tr><td>GEN3C [34]</td><td>18.47</td><td>0.5717</td><td>0.2860</td><td>44.54</td><td>19.66</td><td>0.6026</td><td>16.24</td><td>0.4802</td><td>~1200s</td></tr><tr><td>Ours</td><td>19.80</td><td>0.6195</td><td>0.2188</td><td>32.36</td><td>21.35</td><td>0.6605</td><td>16.66</td><td>0.4983</td><td>~11s</td></tr></table>

Experimental Setup. We evaluate in two settings. Single-view extrapolation uses the first frame of each test sequence as the source view and samples target views from subsequent frames at a fixed stride. Two-view interpolation additionally uses the last frame as a second source. The stride is 4 on DL3DV-10K and Mip-NeRF 360, and 25 on RE10K to account for its slower camera motion. Video-architecture baselines receive the full pose sequence and produce outputs at the strided frames. Baselines conditioned on forward-warped RGB renders use DA3 for depth estimation and pose-depth alignment, matching GenRec. All baselines are open-source; we follow the evaluation protocols from their official repositories.

Single-View Conditioned Extrapolation. We evaluate on two in-distribution benchmarks, RealEstate10K [60] (RE10K) and DL3DV-10K [23], and on Mip-NeRF 360 [2] for out-of-distribution generalization. We sample 100 scenes from the official test splits of RE10K and DL3DV-10K and use all 9 Mip-NeRF 360 scenes. We report PSNR, SSIM, and LPIPS for reconstruction quality and FID for generation quality. Unlike prior work, we report reconstruction metrics globally and separately for observed and unobserved regions. For fair comparison, observed and unobserved regions are scored against a shared binary observation mask from Gen3C’s forward-warping, applied to all methods. We compare against baselines spanning different conditioning strategies and architectures: CameraCtrl [13], ViewCrafter [56], Stable Virtual Camera (SEVA) [59], GLD [17], Gen3R [16], and Gen3C [34]. On RE10K, we additionally compare against Geometry Forcing (GF) [47] and LVSM [18], both RE10K-only methods.

Quantitative Results. Tables 1, 2, and 3 report quantitative results across the benchmarks. GenRec outperforms the baselines on every metric, demonstrating the effectiveness of our approach. In single-view extrapolation, baselines conditioned solely on Plücker rays are limited by pose-only scale ambiguity, while those using forward-warped RGB renders exploit metric-scale geometry for more consistent generations. Figure 3 compares qualitatively against our strongest baseline, Gen3C [34]. Gen3C uses the forward-warped RGB render directly as the denoising target, yielding highly 3Dconsistent outputs when warps are reliable; as camera motion grows, warp quality degrades and so do the generated images. This matches Gen3C’s higher FID on DL3DV-10K and Mip-NeRF 360, which exhibit larger viewpoint changes than RE10K. We strongly encourage the readers to consult the supplementary for more examples. App. B.1 ablates the generation–reconstruction split directly, and App. B.7 reports depth-robustness studies.

Table 3: Quantitative results on Mip-NeRF 360. Out-of-distribution comparison with prior methods in the single-view extrapolation setting. We follow the same evaluation protocol as in Table 1.
<table><tr><td></td><td colspan="4">Global</td><td colspan="2">Observed</td><td colspan="2">Unobserved</td><td></td></tr><tr><td>Method</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>FID↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>PSNR ↑</td><td>SSIM↑</td><td>Inf. Time</td></tr><tr><td>CameraCtrl [13]</td><td>10.89</td><td>0.1870</td><td>0.6534</td><td>214.2</td><td>11.31</td><td>0.1869</td><td>10.67</td><td>0.1821</td><td>~12s</td></tr><tr><td>ViewCrafter [56]</td><td>11.75</td><td>0.2197</td><td>0.5694</td><td>172.5</td><td>14.46</td><td>0.2510</td><td>10.74</td><td>0.1913</td><td>~200s</td></tr><tr><td>SEVA [59]</td><td>11.52</td><td>0.2231</td><td>0.5857</td><td>139.8</td><td>12.09</td><td>0.2315</td><td>11.30</td><td>0.2171</td><td>~20s</td></tr><tr><td>GLD [17]</td><td>12.03</td><td>0.1986</td><td>0.5631</td><td>162.1</td><td>13.32</td><td>0.2044</td><td>11.45</td><td>0.1900</td><td>~42s</td></tr><tr><td>Gen3R [16]</td><td>11.07</td><td>0.2056</td><td>0.6564</td><td>223.6</td><td>11.21</td><td>0.2072</td><td>10.96</td><td>0.2015</td><td>~90s</td></tr><tr><td>GEN3C [34]</td><td>12.86</td><td>0.2343</td><td>0.5796</td><td>207.4</td><td>14.43</td><td>0.2608</td><td>12.12</td><td>0.2117</td><td>~1200s</td></tr><tr><td>Ours</td><td>13.10</td><td>0.2478</td><td>0.4895</td><td>139.4</td><td>15.37</td><td>0.2858</td><td>12.16</td><td>0.22</td><td>~11s</td></tr></table>

![](images/73f3a43c460dc9849c2ed3781daa5d19f4df294c824eb9cd5ba44932bd1c7811.jpg)  
Figure 3: Comparative qualitative results. Single-view extrapolation against our strongest baseline, Gen3C [34]. Each row shows the input view, ground-truth target, GenRec, and Gen3C.

Two-View Conditioned Interpolation. Next we evaluate GenRec in a predominantly reconstructive regime, where most of the information required to synthesize the target frames is already present in the two source views. We conduct this evaluation on RE10K and DL3DV-10K datasets. Table 4 reports reconstruction quality in the interpolation setting. GenRec outperforms all baselines on every reconstruction metric, confirming that our approach remains effective even when the task reduces to a nearly reconstruction-only (i.e., no generation) problem. App. B.3 extends this comparison to dedicated reconstruction methods, and App. B.2 evaluates interpolation from four and six input views.

Table 4: Quantitative results on RE10K and DL3DV-10K. Comparison with prior methods in the two-view interpolation setting.
<table><tr><td rowspan="2">Method</td><td colspan="3">RE10K</td><td colspan="3">DL3DV-10K</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>CameraCtrl [13]</td><td>12.87</td><td>0.4271</td><td>0.5022</td><td>15.60</td><td>0.3456</td><td>0.4705</td></tr><tr><td>ViewCrafter [56]</td><td>12.61</td><td>0.4550</td><td>0.4573</td><td>17.38</td><td>0.5214</td><td>0.3087</td></tr><tr><td>SEVA [59]</td><td>16.15</td><td>0.5833</td><td>0.3477</td><td>21.38</td><td>0.6906</td><td>0.1845</td></tr><tr><td>LVSM [18]</td><td>16.88</td><td>0.5493</td><td>0.3983</td><td>21.31</td><td>0.6801</td><td>0.2223</td></tr><tr><td>GLD [17]</td><td>17.64</td><td>0.5707</td><td>0.2900</td><td>19.49</td><td>0.5575</td><td>0.2376</td></tr><tr><td>Gen3R [16]</td><td>12.17</td><td>0.4385</td><td>0.5697</td><td>13.99</td><td>0.3813</td><td>0.5030</td></tr><tr><td>GEN3C [34]</td><td>15.70</td><td>0.5632</td><td>0.3282</td><td>21.16</td><td>0.6869</td><td>0.2137</td></tr><tr><td>Ours</td><td>19.80</td><td>0.7043</td><td>0.2205</td><td>21.99</td><td>0.7009</td><td>0.1558</td></tr></table>

Table 5: Ablation study on the effectiveness of individual components on RE10K and DL3DV. Starting from a vanilla multi-view flow matching baseline, we progressively add the VAE skip connections + LoRA weights, and the reconstruction branch.
<table><tr><td rowspan="2" colspan="2">Method</td><td colspan="4">Global</td><td colspan="2">Observed</td></tr><tr><td>PSNR↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>FID↓</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td rowspan="3">RE10K</td><td>Generative Backbone</td><td>15.86</td><td>0.5483</td><td>0.3402</td><td>37.82</td><td>19.20</td><td>0.6139</td></tr><tr><td>+ VAE Skip &amp; LoRA</td><td>15.86</td><td>0.5591</td><td>0.3394</td><td>35.27</td><td>19.23</td><td>0.6152</td></tr><tr><td>+ Reconstruction Branch</td><td>17.05</td><td>0.6071</td><td>0.3220</td><td>33.09</td><td>20.85</td><td>0.6738</td></tr><tr><td></td><td>Generative Backbone</td><td>18.62</td><td>0.5796</td><td>0.2556</td><td>39.94</td><td>19.96</td><td>0.6163</td></tr><tr><td>DDV</td><td>+ VAE Skip &amp; LoRA</td><td>18.64</td><td>0.5828</td><td>0.2512</td><td>36.90</td><td>20.02</td><td>0.6215</td></tr><tr><td></td><td>+ Reconstruction Branch</td><td>19.80</td><td>0.6195</td><td>0.2188</td><td>32.36</td><td>21.35</td><td>0.6605</td></tr></table>

Ablating Individual Components. Table 5 ablates the contribution of each architectural component on RE10K and DL3DV-10K. The VAE skip connections together with the decoder LoRA and the reconstruction branch play complementary roles. Adding the skip connections and LoRA weights primarily improves generative quality, as reflected in the lower FID scores on both datasets. The reconstruction branch, in contrast, sharpens reconstruction fidelity (PSNR, SSIM, LPIPS) without sacrificing generative quality.

## 5 Conclusion

We presented GenRec, a multi-view flow matching model that treats generative novel-view synthesis as a structured composition of reconstruction and generation rather than a single distribution-matching problem. A per-pixel observation mask, derived from monocular depth and forward warping, governs the architecture, the loss, and the gradient flow: a flow matching backbone jointly denoises RGB and scene-coordinate latents to handle the distributional regime, while a pixel-space refinement stage with a skip- and LoRA-adapted decoder and a sparse 3D cross-attention branch restores highfrequency detail on observed pixels alone, with masked stop-gradients keeping its regression signal away from the generative prior. Across RealEstate10K, DL3DV-10K, and Mip-NeRF 360, in both single-view extrapolation and two-view interpolation, GenRec attains the best reconstruction fidelity in observed regions and matches or surpasses the strongest generative baselines on perceptual quality in unobserved ones, while running an order of magnitude faster than warp-as-target video baselines.

Limitations. GenRec inherits the limitations of its monocular depth front-end, whose accuracy bounds the quality of the observation mask and warped conditioning, though monocular metric depth estimators continue to improve rapidly. Furthermore, computational constraints, compounded by our architecture’s bi-modal nature, currently limit training to a limited number of views per scene. Apps. B.8 and B.7 state our failure modes explicitly and quantify the depth dependence.

## References

[1] Sherwin Bahmani, Tianchang Shen, Jiawei Ren, Jiahui Huang, Yifeng Jiang, Haithem Turki, Andrea Tagliasacchi, David B Lindell, Zan Gojcic, Sanja Fidler, et al. Lyra: Generative 3d scene reconstruction via video diffusion model self-distillation. arXiv preprint arXiv:2509.19296, 2025.

[2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 5470–5479, 2022.

[3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf: Antialiased grid-based neural radiance fields. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 19697–19705, 2023.

[4] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19457–19467, 2024.

[5] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In European conference on computer vision, pages 370–386. Springer, 2024.

[6] Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models on differentiable rewards. In International Conference on Learning Representations, 2024.

[7] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[8] Noam Elata, Bahjat Kawar, Yaron Ostrovsky-Berman, Miriam Farber, and Ron Sokolovsky. Novel view synthesis with pixel-space diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26756–26766, 2025.

[9] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

[10] Chuan Fang, Heng Li, Yixun Liang, Jia Zheng, Yongsen Mao, Yuan Liu, Rui Tang, Zihan Zhou, and Ping Tan. Spatialgen: Layout-guided 3d indoor scene generation. arXiv preprint arXiv:2509.14981, 3, 2025.

[11] Tobias Fischer, Samuel Rota Bulò, Yung-Hsu Yang, Nikhil Keetha, Lorenzo Porzi, Norman Müller, Katja Schwarz, Jonathon Luiten, Marc Pollefeys, and Peter Kontschieder. Flowr: Flowing from sparse to dense 3d reconstructions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 27702–27712, 2025.

[12] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view diffusion models. arXiv preprint arXiv:2405.10314, 2024.

[13] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024.

[14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[15] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022.

[16] Jiaxin Huang, Yuanbo Yang, Bangbang Yang, Lin Ma, Yuewen Ma, and Yiyi Liao. Gen3r: 3d scene generation meets feed-forward reconstruction. arXiv preprint arXiv:2601.04090, 2026.

[17] Wooseok Jang, Seonghu Jeon, Jisang Han, Jinhyeok Choi, Minkyung Kwon, Seungryong Kim, Saining Xie, and Sainan Liu. Repurposing geometric foundation models for multi-view diffusion. arXiv preprint arXiv:2603.22275, 2026.

[18] Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang Xu. Lvsm: A large view synthesis model with minimal 3d inductive bias. arXiv preprint arXiv:2410.17242, 2024.

[19] Nikhil Keetha, Norman Müller, Johannes Schönberger, Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, Jonathon Luiten, Manuel Lopez-Antequera, Samuel Rota Bulò, Christian Richardt, Deva Ramanan, Sebastian Scherer, and Peter Kontschieder. MapAnything: Universal feed-forward metric 3D reconstruction. In International Conference on 3D Vision (3DV), 2026.

[20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis, et al. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.

[21] Hanwen Liang, Junli Cao, Vidit Goel, Guocheng Qian, Sergei Korolev, Demetri Terzopoulos, Konstantinos N Plataniotis, Sergey Tulyakov, and Jian Ren. Wonderland: Navigating 3d scenes from a single image. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 798–810, 2025.

[22] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.

[23] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160–22169, 2024.

[24] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

[25] Fangfu Liu, Wenqiang Sun, Hanyang Wang, Yikai Wang, Haowen Sun, Junliang Ye, Jun Zhang, and Yueqi Duan. Reconx: Reconstruct any scene from sparse views with video diffusion model. IEEE Transactions on Image Processing, 2026.

[26] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9298–9309, 2023.

[27] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

[28] Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, Benjamin Bossan, and Marian Tietz. PEFT: State-of-the-art parameter-efficient fine-tuning methods. https://github.com/ huggingface/peft, 2022.

[29] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications ofthe ACM, 65 (1):99–106, 2021.

[30] Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5480–5490, 2022.

[31] Gaurav Parmar, Taesung Park, Srinivasa Narasimhan, and Jun-Yan Zhu. One-step image translation with text-to-image models. arXiv preprint arXiv:2403.12036, 2024.

[32] Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In Proceedings ofthe AAAI conference on artificial intelligence, volume 32, 2018.

[33] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.

[34] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed world-consistent video generation with precise camera control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6121–6132, 2025.

[35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

[36] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

[37] Johannes Schusterbauer, Ming Gui, Frank Fundel, and Björn Ommer. Diff2flow: Training flow matching models via diffusion model alignment. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 28347–28357, 2025.

[38] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110, 2023.

[39] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023.

[40] Vincent Sitzmann, Semon Rezchikov, Bill Freeman, Josh Tenenbaum, and Fredo Durand. Light field networks: Neural scene representations with single-evaluation rendering. Advances in Neural Information Processing Systems, 34:19313–19325, 2021.

[41] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

[42] Stanislaw Szymanowicz, Jason Y Zhang, Pratul Srinivasan, Ruiqi Gao, Arthur Brussee, Aleksander Holynski, Ricardo Martin-Brualla, Jonathan T Barron, and Philipp Henzler. Bolt3d: Generating 3d scenes in seconds. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 24846–24857, 2025.

[43] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In European Conference on Computer Vision, pages 1–18. Springer, 2024.

[44] Jiahao Wang, Yufeng Yuan, Rujie Zheng, Youtian Lin, Jian Gao, Lin-Zhuo Chen, Yajie Bao, Yi Zhang, Chang Zeng, Yanxi Zhou, et al. Spatialvid: A large-scale video dataset with spatial annotations. arXiv preprint arXiv:2509.09676, 2025.

[45] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025.

[46] Frederik Warburg, Ethan Weber, Matthew Tancik, Aleksander Holynski, and Angjoo Kanazawa. Nerfbusters: Removing ghostly artifacts from casually captured nerfs. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18120–18130, 2023.

[47] Haoyu Wu, Diankun Wu, Tianyu He, Junliang Guo, Yang Ye, Yueqi Duan, and Jiang Bian. Geometry forcing: Marrying video diffusion and 3d representation for consistent world modeling. arXiv preprint arXiv:2507.07982, 2025.

[48] Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Gojcic, and Huan Ling. Difix3d+: Improving 3d reconstructions with single-step diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26024–26035, 2025.

[49] Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong Park, Ruiqi Gao, Daniel Watson, Pratul P Srinivasan, Dor Verbin, Jonathan T Barron, Ben Poole, et al. Reconfusion: 3d reconstruction with diffusion priors. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 21551–21561, 2024.

[50] Sibo Wu, Daniel Barath, Konrad Schindler, and Andreas Geiger. Genfusion: Closing the loop between reconstruction and generation via videos. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[51] Zirui Wu, Zeren Jiang, Martin R Oswald, and Jie Song. From rays to projections: Better inputs for feed-forward view synthesis. arXiv preprint arXiv:2601.05116, 2026.

[52] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 16453–16463, 2025.

[53] Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys, Ming-Hsuan Yang, and Songyou Peng. No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images. arXiv preprint arXiv:2410.24207, 2024.

[54] Botao Ye, Boqi Chen, Haofei Xu, Daniel Barath, and Marc Pollefeys. Yonosplat: You only need one model for feedforward 3d gaussian splatting. Iclr, 2026.

[55] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4578–4587, 2021.

[56] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024.

[57] Zehao Yu, Songyou Peng, Michael Niemeyer, Torsten Sattler, and Andreas Geiger. Monosdf: Exploring monocular geometric cues for neural implicit surface reconstruction. Advances in neural information processing systems, 35:25018–25032, 2022.

[58] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

[59] Jensen Zhou, Hang Gao, Vikram Voleti, Aaryaman Vasishta, Chun-Han Yao, Mark Boss, Philip Torr, Christian Rupprecht, and Varun Jampani. Stable virtual camera: Generative view synthesis with diffusion models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 12405– 12414, 2025.

[60] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817, 2018.

## A Qualitative Results

![](images/9d0316a802c8f2392dc17901867f5de92d6ef90ba2ebcaa9a3ae09f7bead29e8.jpg)  
Figure 4: Qualitative comparison on Mip-NeRF 360 [2]. Each row shows a held-out target view rendered from a single reference, compared against GEN3C. Mip-NeRF 360’s unbounded outdoor scenes and wide-baseline trajectories stress geometric consistency in the unobserved background; our model preserves thin structures and produces sharper far-field detail than the baselines.

![](images/d59287eb5637cbf13c4d15bca1483df88a02fc0aa9f0d3ea6ff41d49c1b8b7c5.jpg)

![](images/bef11bcf3cd5bacce272628f15304d112f5437ab865650af01eaa23567674fad.jpg)

![](images/b5bd6f2fbc6bc5ad7de4c3588140109d4c466df5270faf5fcf76700b5f83c92c.jpg)  
Figure 5: Qualitative comparison on RealEstate10K [60]. Indoor real-estate walkthroughs feature predominantly forward camera motion and narrow baselines. We highlight cases with disocclusions of structured surfaces (walls, doorways, furniture edges); our method recovers plausible texture in unobserved regions while baselines exhibit blur or ghosting along occlusion boundaries.

![](images/b7a0f569dcba6fdd28b3253e83a3650ceb8ce66e9b96a0782c7ec27796d38569.jpg)  
Figure 6: Qualitative comparison on DL3DV-10K [23]. DL3DV-10K provides diverse, widebaseline captures of indoor and outdoor scenes. The larger camera motions amplify the gap between methods that only inpaint warped pixels and methods that synthesize globally consistent geometry; our outputs remain sharp and view-consistent under aggressive target poses.

## B Additional Experiments and Ablation Studies

## B.1 Effectiveness of the Generation–Reconstruction Split

Table 6: Ablation of the generation–reconstruction split on RealEstate10K, single-view extrapolation. Rec.: dedicated pixel-space reconstruction branch; in (a) the branch is removed and the mask-gated regression loss is applied to the backbone directly. Gate: mask-gated supervision, i.e., flow-matching supervision on all pixels with an additional regression supervision on observed ones. BP: reconstruction gradients back-propagated into the backbone through the last K=1 sampling steps, following Clark et al. [6]. Rows (b)–(d) share the same architecture, parameter count, and training procedure. Best results in bold.
<table><tr><td></td><td></td><td></td><td></td><td colspan="4">Global</td><td colspan="2">Observed</td><td colspan="2">Unobserved</td></tr><tr><td></td><td>Rec.</td><td>Gate</td><td>BP</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td><td>FID↓</td><td>PSNR ↑</td><td>SSIM↑</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>(a)</td><td>X</td><td>√</td><td>一</td><td>16.51</td><td>0.5779</td><td>0.3459</td><td>36.84</td><td>19.88</td><td>0.6535</td><td>13.96</td><td>0.4667</td></tr><tr><td>(b)</td><td>√</td><td>X</td><td>X</td><td>16.97</td><td>0.6020</td><td>0.3433</td><td>40.43</td><td>20.55</td><td>0.6724</td><td>14.28</td><td>0.4940</td></tr><tr><td>(c)</td><td>√</td><td>√</td><td>√</td><td>16.99</td><td>0.6067</td><td>0.3429</td><td>34.92</td><td>20.77</td><td>0.6732</td><td>14.13</td><td>0.4738</td></tr><tr><td>(d)</td><td>√</td><td>√</td><td>×</td><td>17.05</td><td>0.6071</td><td>0.3220</td><td>33.09</td><td>20.85</td><td>0.6738</td><td>14.18</td><td>0.4743</td></tr></table>

Table 6 isolates the effect of the generation–reconstruction split on RE10K in the single-view extrapolation setting. In row (a), the reconstruction branch is removed entirely, and the mask-gated supervision is instead applied to the generative backbone directly through a pixel-wise loss. In row (b), the branch is added but the gating is removed, so the branch is supervised on all pixels. In row (c), both the branch and the gating are present, but reconstruction gradients are back-propagated into the backbone through the last K=1 sampling steps, following Clark et al. [6]. Row (d) is our full model: the branch is added, the gating is applied, and no reconstruction gradient reaches the backbone. Rows (b)–(d) share the same architecture, parameter count, and training procedure, differing only in whether supervision is gated by the observation mask and whether reconstruction gradients reach the backbone; row (a) removes the branch entirely as a reference point.

Supervision — rows (b) vs. (d). These rows differ only in the mask gating. Removing the gating significantly degrades performance, with FID increasing by 7.34. Although row (b) has more capacity than row (a), the lack of gating significantly damages the generative quality of the model.

Gradient flow — rows (c) vs. (d). These rows differ only in whether reconstruction gradients reach the backbone. Allowing them through leaves row (c) behind the decoupled row (d) on every metric we measure, even though row (c) effectively has more trainable parameters through the additional finetuning of the backbone. The extra gradient flow from the reconstruction objective improves neither reconstruction fidelity nor generative quality.

Architecture — rows (a) vs. (d). With gating applied in both, routing the regression signal through a separate, gradient-isolated pixel-space branch rather than applying it to the backbone directly yields significant improve ments in both reconstruction fidelity (PSNR, SSIM) and generative quality (FID, LPIPS), indicating that the reconstruction-branch architecture is itself an important factor in the performance of the model.

## B.2 Multi-View Interpolation Beyond the Training Regime

Table 7: Multi-view interpolation on RE10K. GenRec is trained with at most two input views, so the 4- and 6-view results are zero-shot (marked OOD), whereas Gen3C’s 3D cache ingests additional views natively. Best result per view count in bold.
<table><tr><td>#Views</td><td>Method</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td rowspan="2">2</td><td>Gen3C [34]</td><td>15.70</td><td>0.5632</td><td>0.3282</td></tr><tr><td>GenRec</td><td>19.80</td><td>0.7043</td><td>0.2205</td></tr><tr><td rowspan="2">4</td><td>Gen3C [34]</td><td>20.80</td><td>0.7596</td><td>0.1860</td></tr><tr><td>GenRec (OOD)</td><td>23.68</td><td>0.7988</td><td>0.1387</td></tr><tr><td rowspan="2">6</td><td>Gen3C [34]</td><td>20.86</td><td>0.7656</td><td>0.1793</td></tr><tr><td>GenRec (OOD)</td><td>24.07</td><td>0.8061</td><td>0.1342</td></tr></table>

The architecture places no cap on the number of source views: sources enter only through the warp and mask precomputation and through the key set of the sparse source-to-target attention, both linear in the number of source views. The two restrictions in our benchmarks are budgetary rather than architectural: the number of input views $( T _ { \mathrm { i n } } \in \{ 1 , 2 \} )$ and the total number of views per training sample (App. C.4) bound the training sequence length, and neither is enforced at inference. Table 7 evaluates multi-view interpolation on RE10K against Gen3C, the strongest baseline in our comparisons and the closest to GenRec in conditioning. Since we train on at most two input views, the 4- and 6-view results are zero-shot, whereas Gen3C’s 3D cache ingests additional views natively.

GenRec leads Gen3C on every metric at every view count, by 2.9 to 4.1 dB PSNR, despite never seeing more than two input views during training. The trend is more informative than the margins: between four and six views, GenRec gains 0.39 dB while Gen3C gains 0.06 dB and has effectively saturated. We read this through the paper’s own lens. More source views raise the observed fraction of each target frame, exactly the regime the reconstruction branch exists to exploit, whereas a warp-as-target formulation stops benefiting once its cache already covers the target. The decomposition therefore scales with the available evidence rather than saturating with it, and does so zero-shot at four and six views.

## B.3 Comparison with Reconstruction Methods

Table 8: Comparison with reconstruction methods in the two-view interpolation setting on RE10K and DL3DV-10K. The baselines are specialised, deterministic reconstruction models; GenRec is generative. Best results in bold.
<table><tr><td rowspan="2">Method</td><td colspan="3">RE10K</td><td colspan="3">DL3DV-10K</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td></tr><tr><td>3DGS [20]</td><td>14.95</td><td>0.543</td><td>0.421</td><td>19.76</td><td>0.681</td><td>0.248</td></tr><tr><td>Difix3D+ [48]</td><td>15.06</td><td>0.543</td><td>0.387</td><td>19.27</td><td>0.655</td><td>0.250</td></tr><tr><td>pixelSplat [4]</td><td>16.96</td><td>0.614</td><td>0.392</td><td>19.29</td><td>0.625</td><td>0.328</td></tr><tr><td>MVSplat [5]</td><td>16.64</td><td>0.592</td><td>0.383</td><td>17.53</td><td>0.535</td><td>0.366</td></tr><tr><td>DepthSplat [52]</td><td>19.92</td><td>0.723</td><td>0.277</td><td>19.79</td><td>0.673</td><td>0.253</td></tr><tr><td>GenRec</td><td>19.80</td><td>0.704</td><td>0.221</td><td>21.99</td><td>0.701</td><td>0.156</td></tr></table>

The two-view interpolation setting is predominantly reconstructive, so we additionally compare against dedicated reconstruction methods: 3DGS [20], Difix3D+ [48], pixelSplat [4], MVSplat [5], and DepthSplat [52], in Table 8. GenRec leads on all three metrics on DL3DV-10K (+2.20 dB PSNR over DepthSplat, with the best SSIM and LPIPS) and on LPIPS on both datasets. DepthSplat remains ahead on RE10K PSNR and SSIM, though GenRec trails by only 0.12 dB. We attribute the remaining gap to a systematic effect: deterministic regressors trained under squared error tend toward the conditional mean, which PSNR and SSIM reward and LPIPS penalizes. These baselines are specialised reconstruction models, whereas GenRec is generative; it is therefore on par with them in the regime they are built for, while additionally extrapolating beyond the input views, which they cannot do.

## B.4 3D Consistency

Table 9: Camera pose accuracy and multi-view consistency on RE10K and DL3DV-10K. ATE and RPE are reported in the scale of the dataset camera poses; RPE in degrees. Inference time is reported per scene. Lower is better for all metrics.
<table><tr><td></td><td colspan="3">RE10K</td><td colspan="3">DL3DV</td><td></td></tr><tr><td>Method</td><td>ATE↓</td><td>RPEt↓</td><td>RPEr ↓</td><td>ATE ↓</td><td>RPEt↓</td><td>RPEr ↓</td><td>Inf. Time</td></tr><tr><td>GLD [17]</td><td>0.131</td><td>0.158</td><td>0.837</td><td>0.030</td><td>0.041</td><td>0.395</td><td>~42s</td></tr><tr><td>ViewČrafter [56]</td><td>0.134</td><td>0.148</td><td>1.446</td><td>0.045</td><td>0.060</td><td>0.535</td><td>~200s</td></tr><tr><td>Gen3R [16]</td><td>0.134</td><td>0.162</td><td>1.124</td><td>0.370</td><td>0.526</td><td>4.900</td><td>~90s</td></tr><tr><td>Gen3C [34]</td><td>0.085</td><td>0.083</td><td>0.809</td><td>0.037</td><td>0.052</td><td>0.430</td><td>~1200s</td></tr><tr><td>Ours</td><td>0.087</td><td>0.106</td><td>1.048</td><td>0.021</td><td>0.029</td><td>0.252</td><td>~11s</td></tr></table>

Beyond per-frame image quality, a generative novel view synthesis model should produce frames that are mutually consistent in 3D. We evaluate this property for GenRec in the single-view conditioned extrapolation setting on the two in-distribution datasets, RE10K and DL3DV-10K. We pass the generated frames through an off-the-shelf feed-forward 3D reconstruction model [45] to estimate their camera poses. Because VGGT recovers poses only up to a global similarity, we Sim(3)-align the estimates to the ground-truth target poses before computing Absolute Trajectory Error (ATE) and Relative Pose Error in rotation (RPE ) and translation (RPE<sub>t</sub>). The resulting values are therefore reported up to scale, but remain directly comparable across methods evaluated under the same protocol. Together, these metrics quantify how faithfully the generated frames adhere to the target pose conditioning.

Table 9 reveals a clear dataset-dependent pattern. On DL3DV-10K, GenRec achieves the best performance across all three metrics, with reductions of 29–36% relative to the strongest baseline, GLD. On RE10K, Gen3C is the strongest method on every metric; GenRec performs comparably on ATE but underperforms on the relative-pose metrics, where it is also outperformed by GLD on rotation. We attribute this dataset-dependent ordering to a warp-quality dynamic: RE10K’s narrow camera baselines yield reliable forward warps, which directly benefit Gen3C’s warp-as-target formulation, whereas DL3DV-10K’s larger viewpoint changes degrade warp quality and surface the advantages of GenRec’s observation-based design.

## B.5 Reconstruction Quality Across Observation Coverage

![](images/a99382862b3dd1cd2853d5d443360dfb4c091aea7203f69c302fdabc79732ae1.jpg)  
(a) RealEstate10K.

![](images/5fc433f689a1ec8ff3e0ad7acc69d415a1fa87136b29f020a77a0d3b00486f76.jpg)  
(b) DL3DV-10K.  
Figure 7: PSNR as a function of per-scene observation percentage on (a) RealEstate10K and (b) DL3DV-10K in the single-view conditioned setting. Each point is one test scene; lines are per-method linear fits. GenRec (red) sits above every baseline across the full range, from generation-heavy scenes on the left to near-reconstruction scenes on the right.

Figure 7 stratifies single-view PSNR by per-scene observation percentage on RealEstate10K and DL3DV-10K, sweeping from generation-heavy scenes (∼30% observed) to near-reconstruction scenes (>90% observed). GenRec dominates across the full range on both datasets: the regression line sits above every baseline at low observation rates, where most of the target view must be hallucinated, and remains the steepest at high observation rates, where the forward-warped point cloud is an accurate prior. This second regime is the more telling one. Methods such as GEN3C use the warped render directly as the denoising target and therefore inherit a strong reconstruction signal essentially for free in well-observed scenes. GenRec matches or exceeds them despite never tying its denoising target to the warped pixels, which we attribute to the reconstruction branch supplying the photometric fidelity that the latent generative path cannot. The result confirms that the generation–reconstruction split adapts gracefully to the actual observation budget of each scene.

## B.6 Effectiveness of Multi-Modal Generation

Table 10: Ablation on multi-modal generation. We compare our joint RGB and scene-coordinate (SC) backbone against an RGB-only variant trained from the same initialization for 20K steps, evaluated in the single-view conditioned extrapolation setting.
<table><tr><td></td><td colspan="4">RE10K</td><td colspan="4">DL3DV-10K</td></tr><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS ↓</td><td>FID↓</td><td>PSNR↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>FID↓</td></tr><tr><td>RGB</td><td>15.09</td><td>0.5384</td><td>0.3602</td><td>38.22</td><td>18.05</td><td>0.5681</td><td>0.2631</td><td>38.53</td></tr><tr><td> $\mathrm { R G B } + \mathrm { S C } \left( \mathrm { O u r s } \right)$ </td><td>15.39</td><td>0.5457</td><td>0.3528</td><td>37.13</td><td>18.31</td><td>0.5749</td><td>0.2575</td><td>37.41</td></tr></table>

In Table 10 we isolate the effect of the joint RGB and scene-coordinate denoising introduced in Section 3.1. We train an RGB-only variant of the backbone that removes the scene-coordinate input channels, the scene-coordinate output head, and the zero-initialized cross-modal attention layers, leaving the cross-view attention and all other components unchanged. Both checkpoints are trained from the same Stable Diffusion 2.1 [35] initialization for

20K steps (∼1/3 of full convergence) and evaluated under the single-view conditioned extrapolation protocol on RE10K and DL3DV-10K.

Across both datasets, the multi-modal variant improves every metric, including FID and LPIPS, indicating that the joint denoising not only sharpens reconstruction in observed regions but also tightens the perceptual quality of the generated image. We attribute this to the cross-modal attention forcing the RGB branch to commit to a 3D-consistent interpretation of the scene at every denoising step, a regularizer that pure RGB diffusion lacks.

## B.7 Robustness to the Depth Front-End

Table 11: Depth source at inference on RealEstate10K, single-view extrapolation. GenRec is trained on DA3 renders (†); the estimator is swapped at test time with no retraining. Evaluated on the same RealEstate10K subset as Table 12, so values are not comparable to Table 1. Best results in bold.
<table><tr><td></td><td colspan="4">Global</td><td colspan="2">Observed</td><td colspan="2">Unobserved</td></tr><tr><td>Depth</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FID↓</td><td>PSNR ↑</td><td>SSIM↑</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>DA3† [22]</td><td>17.35</td><td>0.6169</td><td>0.3206</td><td>63.69</td><td>22.34</td><td>0.6736</td><td>15.26</td><td>0.5242</td></tr><tr><td>MapAnything [19]</td><td>17.23</td><td>0.6151</td><td>0.3236</td><td>63.66</td><td>22.08</td><td>0.6717</td><td>15.15</td><td>0.5192</td></tr><tr><td>VGGT [45]</td><td>16.43</td><td>0.5734</td><td>0.3478</td><td>67.48</td><td>20.54</td><td>0.6092</td><td>14.79</td><td>0.4952</td></tr></table>

GenRec relies on a monocular depth front-end to build its observation mask and warped conditioning. We quantify this dependence with two studies.

Swapping the depth estimator at inference. Table 11 replaces the depth front-end at inference only, measuring what is lost when depth no longer comes from the estimator the model was trained with (DA3 [22]). Swapping in MapAnything [19] costs 0.12 dB PSNR and marginally improves FID, so the model is not tied to the estimator it was trained with; VGGT [45], whose depth differs more, costs 0.92 dB. Neither swap requires retraining, so GenRec also inherits future improvements in monocular depth for free: the front-end is a replaceable component, not a fixed dependency.

Table 12: Depth-robustness sweep on RealEstate10K (25-scene subset, so clean values are not comparable to Tables 1–3). At inference, the source depth is corrupted after the Umeyama alignment that fixes scale; every method builds its own conditioning from the corrupted depth, and all runs are scored against a single frozen observation mask. Each entry averages over the severity levels of the corresponding family: three levels for the affine corruptions, one for boundary, and three for noise. Best result per block in bold.
<table><tr><td rowspan="2"></td><td colspan="4">Global</td><td colspan="2">Observed</td><td colspan="2">Unobserved</td></tr><tr><td>Corruption Method</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td><td>FID↓</td><td>PSNR ↑ SSIM ↑</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td rowspan="3">Clean</td><td>GenRec</td><td>17.35</td><td>0.6196</td><td>0.3201</td><td>63.12</td><td>22.36</td><td>0.6759</td><td>15.25 0.5263</td></tr><tr><td>Gen3C [34]</td><td>15.44</td><td>0.5613</td><td>0.3627 67.04</td><td>20.04</td><td>0.6037</td><td>13.81</td><td>0.4893</td></tr><tr><td>ViewCrafter [56]</td><td>13.01</td><td>0.4749</td><td>0.4505 85.04</td><td>17.27</td><td>0.5222</td><td>11.43</td><td>0.3875</td></tr><tr><td rowspan="3">Affine</td><td>GenRec</td><td>15.08</td><td>0.5122</td><td>0.4101</td><td>67.08</td><td>18.55</td><td>0.5232</td><td>14.09</td><td>0.4691</td></tr><tr><td>Gen3C [34]</td><td>14.64</td><td>0.5240</td><td>0.4071</td><td>70.91</td><td>18.30</td><td>0.5404</td><td>13.56</td><td>0.4736</td></tr><tr><td>ViewCrafter [56]</td><td>12.04</td><td>0.4217</td><td>0.4992</td><td>91.53</td><td>15.76</td><td>0.4491</td><td>10.92</td><td>0.3584</td></tr><tr><td rowspan="3">Boundary</td><td>GenRec</td><td>17.23</td><td>0.6130</td><td>0.3251</td><td>65.87</td><td>22.09</td><td>0.6653</td><td>15.13</td><td>0.5169</td></tr><tr><td>Gen3C [34]</td><td>15.53</td><td>0.5690</td><td>0.3620</td><td>66.95</td><td>19.73</td><td>0.5961</td><td>14.06</td><td>0.5042</td></tr><tr><td>ViewCrafter [56]</td><td>12.83</td><td>0.4674</td><td>0.4589</td><td>89.49</td><td>17.02</td><td>0.5118</td><td>11.21</td><td>0.3776</td></tr><tr><td rowspan="3">Noise</td><td>GenRec</td><td>16.53</td><td>0.5765</td><td>0.3666</td><td>76.48</td><td>20.55</td><td>0.5986</td><td>14.86</td><td>0.5077</td></tr><tr><td>Gen3C [34]</td><td>15.25</td><td>0.5524</td><td>0.3793</td><td>67.96</td><td>19.32</td><td>0.5761</td><td>13.88</td><td>0.4902</td></tr><tr><td>ViewCrafter [56]</td><td>11.66</td><td>0.3944</td><td>0.5202</td><td>103.82</td><td>15.12</td><td>0.4129</td><td>10.69</td><td>0.3403</td></tr></table>

Stress-testing beyond real estimator error. The second study corrupts the source depth at inference time, after the Umeyama alignment that fixes scale, using three families of corruption: an affine scale-and-shift in disparity, which models the residual ambiguity that survives alignment; boundary artifacts at depth discontinuities, which attack the observed/unobserved partition most directly; and spatially varying noise. Each family is evaluated at several severity levels, and Table 12 reports the average over levels. Since every method builds its own conditioning from the corrupted depth, the corruption propagates to both the reconstruction branch and the backbone; all runs are scored against a single frozen observation mask, so the pixel sets remain fixed across comparisons. One caveat concerns Gen3C: its clean FID comes from the filter-disabled clean run, which is also the correct baseline for its noise rows, since those require the filter disabled (at the default setting, the renderer returns an empty mask at the strongest noise level). Measured against that run, Gen3C’s noise loss is 0.53 dB rather than the 0.20 dB implied by the clean row.

Across the sweep, GenRec retains the PSNR lead in every corruption family and in every region, and the FID lead in all families except noise. The pattern of where ground is lost is more informative than the margins themselves. Boundary corruption, which attacks the observed/unobserved partition most directly, is the setting our design is built for, and it is also the one GenRec survives best. The affine family costs the most, exactly as the design predicts: affine rescaling deforms the warped evidence that the reconstruction branch consumes, so the path that benefits most from accurate depth is also the one most exposed to inaccurate depth. The same logic explains where the damage lands: in every family, the observed-region loss is two to five times the unobserved-region loss, i.e., the reconstruction path absorbs the depth error while the generative path remains comparatively insulated from it. Finally, the sweep perturbs the clean DA3 prediction rather than sensor depth, so it quantifies sensitivity to depth error and should not be read as bounding absolute accuracy.

## B.8 Failure Modes

The studies above isolate concrete failure modes of the unified formulation and of GenRec itself, which we state explicitly here.

Uniform supervision. Row (b) of Table 6 is the unified-supervision configuration, and its failure is specific: fidelity on observed pixels improves while realism degrades (FID 40.43, the weakest in the table), appearing as over-smoothed or mean-collapsed content in disocclusions.

Depth-induced mask errors. When depth is substantially wrong, the mask can label a genuinely unobserved pixel as observed, and the reconstruction branch then confidently reconstructs a mis-warped region rather than deferring to the generative prior. The depth-corruption sweep of Table 12 confirms this mechanism: in every corruption family, the damage is two to five times larger in observed than in unobserved regions, exactly as the split predicts, while GenRec remains the strongest method on PSNR throughout.

Very low coverage. When a target view is almost entirely unobserved, the mask approaches zero and GenRec reduces to its backbone, so the split contributes little.

Strong view-dependent effects. On observed surfaces with strong view-dependent appearance, the source evidence available to the reconstruction branch is not the target appearance.

## C Preliminaries

## C.1 Flow Matching

Flow Matching [24] has emerged as a simpler and more efficient alternative to diffusion-based approaches for generative modeling. It is built around a time-dependent vector field $v : [ 0 , 1 ] \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { d }$ , whose trajectories define a time-dependent diffeomorphism $\phi : [ 0 , 1 ] \times \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ called a flow. For each initial point $x ,$ the curve $t \mapsto \phi _ { t } ( x )$ is the integral curve of $v _ { t }$ starting at x, characterized by the ordinary differential equation:

$$
\frac { d } { d t } \phi _ { t } ( x ) = v _ { t } ( \phi _ { t } ( x ) ) , \quad \phi _ { 0 } ( x ) = x .\tag{6}
$$

The flow $\phi _ { t }$ induces a probability path $p _ { t }$ by pushing forward the initial distribution $p _ { 0 }$ through the flow: $p _ { t } = ( \phi _ { t } ) _ { \# } p _ { 0 }$ , where $p _ { 0 } = \mathcal { N } ( 0 , I )$ is a standard Gaussian prior and $p _ { 1 } \approx q ( x )$ is the target data distribution. The goal of Flow Matching is to learn a parametric vector field v<sub>θ</sub> whose induced probability path transforms $p _ { 0 }$ into $q .$

Following the optimal-transport conditional path of Lipman et al. [24], a sample at any intermediate time $t \in [ 0 , 1 ]$ is obtained by linearly interpolating between a noise sample $x _ { 0 } \sim p _ { 0 }$ and a data sample $x _ { 1 } \sim q \colon$

$$
x _ { t } = \left( 1 - t \right) x _ { 0 } + t x _ { 1 } , \quad t \in [ 0 , 1 ] .\tag{7}
$$

Differentiating the interpolant with respect to t yields the conditional velocity field

$$
u _ { t } ( x _ { t } \mid x _ { 0 } , x _ { 1 } ) = { \frac { d } { d t } } \big [ ( 1 - t ) x _ { 0 } + t x _ { 1 } \big ] = x _ { 1 } - x _ { 0 } ,
$$

which is constant along each interpolation trajectory. Because $x _ { t }$ depends linearly on the endpoints, $x _ { t }$ together with the velocity at time t uniquely determines $x _ { 1 }$ . Rearranging the interpolant yields a closed-form estimate of the clean sample at any $t \in [ 0 , 1 )$

$$
\hat { x } _ { 1 } = x _ { t } + \left( 1 - t \right) v _ { \theta } ( x _ { t } , t ) ,\tag{8}
$$

so the learned velocity network $v _ { \theta }$ implicitly acts as an x<sub>1</sub>-predictor at every timestep. A single forward pass at arbitrary t therefore yields both a denoising direction and an estimate of the target, without the noise-leveldependent reweighting required by ϵ-prediction in diffusion models [14].

The network $v _ { \theta }$ is trained with the Conditional Flow Matching (CFM) objective [24], which regresses onto the per-sample target velocity:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C F M } } ( \theta ) = \mathbb { E } _ { t , x _ { 0 } , x _ { 1 } } \left. v _ { \theta } ( x _ { t } , t ) - ( x _ { 1 } - x _ { 0 } ) \right. ^ { 2 } , } \end{array}\tag{9}
$$

where $x _ { 0 } \sim \mathcal { N } ( 0 , I ) , x _ { 1 } \sim q ( x )$ , and $x _ { t } = \left( 1 - t \right) x _ { 0 } + t x _ { 1 }$ . We follow Esser et al. [9] and draw t from a logit-normal distribution, $t = \sigma ( u )$ for training. Because both the interpolated sample and its target velocity are available in closed form, this loss admits unbiased stochastic estimates and is simulation-free, meaning training never requires integrating the ODE. At inference, a data sample is generated by drawing $x _ { 0 } \sim \mathcal { N } ( 0 , I )$ and integrating the learned vector field from $t = 0 \mathrm { t o } t = 1$ . Discretizing the interval into N steps with $\Delta t = 1 / N$ the simplest Euler update advances the state from $t _ { k }$ to $t _ { k + 1 } = t _ { k } + \Delta t$ as

$$
{ x _ { t } } _ { k + 1 } = { x _ { t } } _ { k } + \Delta t \cdot { v _ { \theta } } ( { x _ { t } } _ { k } , t _ { k } ) ,\tag{10}
$$

which can be replaced with any higher-order ODE solver (e.g. Heun or Runge–Kutta) for improved accuracy at the same number of function evaluations. We utilize the Euler update in our work as shown in Equation 10.

## C.2 Diff2Flow [37]

While Flow Matching offers attractive properties such as straighter sampling trajectories and simulation-free training, the most widely available early foundation generative models $\left( \mathbf { e . g } \right.$ ., Stable Diffusion [35]) are trained as denoising diffusion models [14, 41] rather than flow models. Diff2Flow bridges this gap by aligning the two paradigms through coupled reparameterizations.

A diffusion model corrupts a data sample $x _ { 0 } ^ { \mathrm { D M } } \sim q ( x )$ with Gaussian noise $x _ { T } ^ { \mathrm { D M } } \sim \mathcal { N } ( 0 , I )$ along discrete timesteps t<sub>DM</sub> $\in \mathbb { Z } _ { \geq 0 } \cap { \mathrm { [ 0 , ~ } } T { \mathrm { ] } }$ . Under a variance-preserving schedule, the noisy interpolant is

$$
x _ { t _ { \mathrm { D M } } } ^ { \mathrm { D M } } = \alpha _ { t _ { \mathrm { D M } } } x _ { 0 } ^ { \mathrm { D M } } + \sigma _ { t _ { \mathrm { D M } } } x _ { T } ^ { \mathrm { D M } } , \qquad \alpha _ { t _ { \mathrm { D M } } } ^ { 2 } + \sigma _ { t _ { \mathrm { D M } } } ^ { 2 } = 1 ,\tag{11}
$$

with $\alpha _ { t _ { \mathrm { D M } } }$ monotonically decreasing and $\sigma _ { t _ { \mathrm { D M } } }$ monotonically increasing in t . The network is trained either to regress the noise ϵ or, in the v-parameterization [36], the target

$$
v _ { t _ { \mathrm { D M } } } ^ { \mathrm { D M } } = \alpha _ { t _ { \mathrm { D M } } } x _ { T } ^ { \mathrm { D M } } - \sigma _ { t _ { \mathrm { D M } } } x _ { 0 } ^ { \mathrm { D M } } ,\tag{12}
$$

the parameterization used by Stable Diffusion $2 . 1$ , on which we focus throughout. Note the convention mismatch with Flow Matching: the diffusion timestep treats $x _ { 0 } ^ { \mathrm { D M } }$ as data and $x _ { T } ^ { \mathrm { D M } }$ as noise, opposite to the FM convention introduced above. The two are reconciled by the boundary identification $x _ { 0 } ^ { \mathrm { D M } } \equiv \dot { x } _ { 1 }$ and $x _ { T } ^ { \mathrm { D M } } \equiv x _ { 0 }$

To align the trajectories, Diff2Flow constructs invertible reparameterizations $f _ { t } : [ 0 , T ]  [ 0 , 1 ]$ on time and $f _ { x } : \bar { \mathbb { R } ^ { d } }  \mathbb { R } ^ { d }$ on the interpolant, chosen such that the diffusion trajectory in Eq. (11) coincides with the FM trajectory $x _ { t } = \left( 1 - t \right) x _ { 0 } + t x _ { 1 }$ under the above boundary identification. Equating the two expressions and solving for t in terms of the diffusion coefficients yields

$$
f _ { t } ( t _ { \mathrm { D M } } ) = \frac { \alpha _ { t _ { \mathrm { D M } } } } { \alpha _ { t _ { \mathrm { D M } } } + \sigma _ { t _ { \mathrm { D M } } } } , \qquad f _ { x } \big ( x _ { t _ { \mathrm { D M } } } ^ { \mathrm { D M } } \big ) = \frac { 1 } { \alpha _ { t _ { \mathrm { D M } } } + \sigma _ { t _ { \mathrm { D M } } } } x _ { t _ { \mathrm { D M } } } ^ { \mathrm { D M } } .\tag{13}
$$

Under both variance-preserving and variance-exploding schedules $f _ { t }$ is strictly monotonic and hence invertible. The discrete coefficients $\{ \alpha _ { t _ { \mathrm { D M } } } , \sigma _ { t _ { \mathrm { D M } } } \}$ are linearly interpolated between their nearest integer neighbors so that $f _ { t }$ and $f _ { t } ^ { - 1 }$ are well-defined on the entire continuous interval; this extension is justified empirically by the observation that diffusion networks trained on integer timesteps produce coherent outputs at fractional ones [37]. Given an FM timestep $t \in [ 0 , 1 ]$ and interpolant $x _ { t } ,$ the diffusion-side quantities are recovered as

$$
t _ { \mathrm { D M } } = f _ { t } ^ { - 1 } ( t ) , \qquad x _ { t _ { \mathrm { D M } } } ^ { \mathrm { D M } } = f _ { x } ^ { - 1 } ( x _ { t } ) = \left( \alpha _ { t _ { \mathrm { D M } } } + \sigma _ { t _ { \mathrm { D M } } } \right) x _ { t } .\tag{14}
$$

Rather than fine-tuning the diffusion network to directly emit FM velocities, which forces it to abandon its existing parameterization and degrades performance under parameter-efficient adaptation [37], Diff2Flow keeps the network in its native parameterization and analytically converts its output. Solving the linear system formed by Eq. (11) and Eq. (12) for the endpoints gives

$$
\begin{array} { r } { \hat { x } _ { 0 } ^ { \mathrm { D M } } = \alpha _ { t _ { \mathrm { D M } } } x _ { t _ { \mathrm { D M } } } ^ { \mathrm { D M } } - \sigma _ { t _ { \mathrm { D M } } } v _ { \theta } ^ { \mathrm { D M } } ( x _ { t _ { \mathrm { D M } } } ^ { \mathrm { D M } } , t _ { \mathrm { D M } } ) , \qquad \hat { x } _ { T } ^ { \mathrm { D M } } = \sigma _ { t _ { \mathrm { D M } } } x _ { t _ { \mathrm { D M } } } ^ { \mathrm { D M } } + \alpha _ { t _ { \mathrm { D M } } } v _ { \theta } ^ { \mathrm { D M } } ( x _ { t _ { \mathrm { D M } } } ^ { \mathrm { D M } } , t _ { \mathrm { D M } } ) . } \end{array}\tag{15}
$$

Substituting the boundary identification into the FM target $x _ { 1 } - x _ { 0 }$ then yields a closed-form FM velocity expressed entirely through the diffusion network’s v-prediction:

$$
v _ { \theta } ( x _ { t } , t ) \ = \ \hat { x } _ { 0 } ^ { \mathrm { D M } } - \hat { x } _ { T } ^ { \mathrm { D M } } \ = \ \left( \alpha _ { t _ { \mathrm { D M } } } - \sigma _ { t _ { \mathrm { D M } } } \right) x _ { t _ { \mathrm { D M } } } ^ { \mathrm { D M } } \ - \ \left( \alpha _ { t _ { \mathrm { D M } } } + \sigma _ { t _ { \mathrm { D M } } } \right) v _ { \theta } ^ { \mathrm { D M } } \big ( x _ { t _ { \mathrm { D M } } } ^ { \mathrm { D M } } , t _ { \mathrm { D M } } \big ) ,\tag{16}
$$

with $t _ { \mathrm { D M } } ~ = ~ f _ { t } ^ { - 1 } ( t )$ and $x _ { t _ { \mathrm { D M } } } ^ { \mathrm { D M } } ~ = ~ f _ { x } ^ { - 1 } ( x _ { t } )$ given by Eqs. (13)–(14). An analogous derivation holds for ϵ-parameterized priors.

Training proceeds with the standard CFM loss in Eq. (9), where the velocity prediction is evaluated through Eq. (16). Sampling uses the Euler update of Eq. (10) on the FM trajectory: at each step, the current state $\boldsymbol { x } _ { t _ { k } }$ and timestep $t _ { k }$ are mapped to the diffusion side via $f _ { t } ^ { - 1 }$ and $f _ { x } ^ { - 1 }$ , the network is queried in its native parameterization, and its v-prediction is converted back to an FM velocity through Eq. (16).

Inference Protocol. At test time we sample with 50 solver steps using the EMA weights accumulated during training. We employ classifier-free guidance with a scale of 1.5.

## C.3 Implementation Details

Generative Backbone. We initialize the multi-view generative backbone from the Stable Diffusion 2.1 (SD-2.1) checkpoint [35] and zero-initialize the cross-modal attention layers. The number of attention heads and head dimensions are reused from SD-2.1 and is symmetric for both cross-view and cross-modal attention. We reuse the pretrained SD-2.1 VAE for the image modality and adopt the scene-coordinate VAE from SpatialGen [10] for the scene-coordinate modality; both VAEs remain frozen throughout training. Plücker ray conditioning is embedded via the SpatialGen ray encoder.

VAE Skip Adapter and LoRA. The mask-conditioned skip adapter (Section 3.1) uses a bias-free 1×1 convolution per skip, initialized with standard deviation $1 0 ^ { - 5 }$ so that an untrained adapter reproduces the pretrained decoder within numerical noise. We adapt the decoder with low-rank residuals via PEFT [28], targeting every convolution and attention projection together with the four skip convolutions: rank $r = 8 ,$ scaling $\alpha = r ,$ Gaussian initialization, biases untouched.

Reconstruction Branch. The branch operates at an internal width of D=64 with four FiLM-conditioned residual blocks interleaved with sparse geometric attention layers (4 heads, head dimension 16). The attention layers between the The query and source streams are projected to D via 3×3 and 1×1 convolutions respectively. Sparsity in both the geometric cross-attention and self-attention is enforced through precomputed k-nearestneighbour index maps with $k { = } 8$

## C.4 Training Details

Our pipeline follows two stages: we first train the generative backbone together with the VAE skip connections and the decoder LoRA weights, then freeze it and train the reconstruction branch with iterative denoising supervision.

Data. We train our method on a mixture of RealEstate10K [60], DL3DV-10K [23] and SpatialVID [44] with sampling weights 0.5, 0.2 and 0.3 respectively, all loaded at a longest-side resolution of 616 pixels with dataset specific frame strides. From SpatialVID we filter out dynamic scenes by thresholding the depth-projection error between frames computed with DA3 depth, retaining only scenes whose static-geometry assumption holds. These datasets together form a collection of around 160K indoor and outdoor multi-view consistent scenes. For every scene we sample $T = 8$ consecutive frames at the dataset-specific stride; the number of clean input views $T _ { \mathrm { i n } }$ is drawn from $\{ \bar { 1 } , 2 \}$ with probabilities 0.8 and 0.2, placed at the first $T _ { \mathrm { i n } }$ positions of the sampled window, and the remaining ${ \dot { T } } _ { \mathrm { o u t } } = T - \mathbf { \hat { \quad } } T _ { \mathrm { i n } }$ views are noised. Metric depth is computed by Depth Anything 3 (DA3) [22] at 504 resolution, matching DA3’s base training resolution. We use Umeyama alignment between the predicted camera poses from DA3 and the ground-truth poses to scale the depth estimation for training.

Multi-view Generation Backbone. We train the generative backbone for 20 epochs (∼72K optimizer steps) with an effective batch size of 16 using 8-bit AdamW, learning rate $1 0 ^ { - 5 }$ , no weight decay, and a global gradient-norm clip of 1.0. The learning rate is linearly warmed up over 500 steps and decayed with a cosine schedule to a minimum of $1 0 ^ { - 6 }$ . To enable classifier-free guidance at inference, we drop the geometric conditioning (warped renders and observation mask) independently with probability 0.1 during training. Training is performed in bfloat16 mixed precision with gradient checkpointing enabled throughout, distributed across 16 GH200 GPUs via PyTorch DDP for a total wall-clock time of approximately three days. We maintain an exponential-moving-average copy of the weights with decay 0.9999.

Reconstruction Branch. The branch is trained in the same optimization loop as the frozen generative backbone but with fully decoupled gradients: gradients influence only the residual predictor. To match the distribution of samples encountered at inference, the detached $\hat { x } _ { 1 } ^ { I }$ is produced by an unrolled denoising rollout of 50 no-gradient solver steps, terminated at a randomly chosen step whose corresponding diffusion timestep, recovered from the flow time t via the Diff2Flow mapping $f _ { t } ^ { - 1 }$ on SD2.1’s 1000-step schedule, falls below 200. This exposes the branch to realistic, near-final samples rather than the single-step estimates available during diffusion training. The branch parameters form a separate optimizer group with learning rate $5 \times 1 0 ^ { - 5 }$ weight decay $1 0 ^ { - 4 }$ , and a 1000-step linear warm-up; all other settings (8-bit AdamW, gradient-norm clip 1.0, bfloat16 mixed precision) match those of the backbone. The branch converges within 10K steps.

Loss Weights. For the geometry-aware decoder adapter $( \mathrm { E q . 4 } ) ,$ , we set $\lambda _ { 1 } = 0 . 1$ and $\lambda _ { \mathrm { p } } = 0 . 3$ . For the pixel-space reconstruction branch $( \dot { \mathrm { E q } } . 5 )$ , we set $\lambda _ { 1 } = \bar { 0 . 1 } , \lambda _ { \mathrm { { p } } } = 0 . 5$ , and $\lambda _ { \Delta } = 0 . 0 5$

## D Societal Impacts

GenRec synthesizes novel views of real-world scenes from one or two posed images, with applications in robotic navigation under sparse observation, mixed-reality and accessibility. The order-of-magnitude inference speedup over warp-as-target video baselines makes such uses practical on commodity hardware.

As with any photorealistic generative model, outputs could in principle be misused to fabricate imagery, and the learned prior reflects the distribution of its training data. We train on RealEstate10K, DL3DV-10K, and SpatialVID, which are publicly released, mostly widely adopted benchmarks in the 3D vision community; the resulting biases are those already characterized in the literature surrounding these datasets, and the privacy considerations are governed by the curation policies under which they were released. We do not introduce new data collection.

A property of GenRec’s design that is directly relevant here is that the observation mask $O _ { i }$ separates pixels recovered by reconstruction from pixels produced by the generative prior, and this separation is available at inference at no additional cost. The mask functions as a built-in provenance signal: it identifies, per pixel, whether the model is reporting evidence or imagining a plausible completion. We encourage downstream systems to surface this signal (through overlays, metadata, or content-credential standards such as C2PA) rather than discarding it once the final image is composited.