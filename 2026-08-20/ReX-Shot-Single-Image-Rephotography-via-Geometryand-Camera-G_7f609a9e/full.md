# ReX-Shot: Single-Image Rephotography via Geometryand Camera-Grounded Generation

Ruiqi Zhang<sup>†</sup>, Hao Zhu<sup>†</sup>, Wenhao Zhang, Qi Zhang, Junqi Shi, Ming Lu, Xun Cao, Zhan Ma

Abstract—Single-image rephotography aims to synthesize new shots of the same scene from a single reference image with specified viewpoints, focal lengths, and parameterized photographic effects, all of which are intrinsically coupled in the imaging process. However, existing methods typically treat them as separate subproblems and struggle under joint control, where novel view synthesis may distort geometry under focal-length changes, while super resolution and instruction-guided editing remain confined to 2D and cannot extend detail restoration or appearance control to novel viewpoints. We find that these limitations arise from two fundamental issues, namely the ill-posed nature of single-image 3D reconstruction and the sampling limitation of continuous focal-length enlargement. The former makes geometric inaccuracies unavoidable during pixel-space explicit projection, while the latter requires recovering details beyond the reference image’s sampling density. To mitigate projection bias from imperfect geometry, we leverage implicitly transformed foundation-model features to provide robust target-view guidance. We then formulate focal-length enlargement as a geometry-guided super-resolution problem, using generative detail priors to recover details missing from sparse 3D resampling. Building on this 3D-aware generative backbone, we lift photographic-effect control from 2D filtering to 3D-aware appearance editing, preserving content consistency across viewpoints and focal lengths. Together, these components form ReX-Shot, a geometry- and camera-grounded generative framework for single-image rephotography. To the best of our knowledge, ReX-Shot is the first unified framework to jointly control viewpoint, focal length, and parameterized photographic effects from a single reference image. Experiments demonstrate that ReX-Shot outperforms representative baselines across viewpoint control, focal-length control, and photographic-effect control, while enabling interactive rephotography at near-real-time speeds. Project page: https://ruiqi-nju.github.io/ReX-Shot/

Index Terms—Single-image rephotography, camera-controlled rendering, photographic effects

✦

## 1 INTRODUCTION

P formation process that maps the high-dimensional physical world onto a static two-dimensional sensor plane. Within this process, camera geometry (viewpoint and focal length) determines projection and composition, and photographic effects determine image appearance. Once the shutter button is pressed, these settings are fixed in the captured image. Therefore, conventional post-processing cannot ensure scene consistency as if it were recaptured under new camera settings. To address this limitation, we define single-image rephotography as synthesizing new shots of the same scene from a single reference image with specified viewpoints, focal lengths, and parameterized photographic effects, such as color temperature, exposure, saturation, and contrast. Unlike generic image editing, rephotography requires preserving scene content and spatial relationships while faithfully responding to the target camera geometry and photographic-effect parameters.

Existing methods typically treat viewpoint control, focallength control, and photographic-effect control as separate subproblems. However, these subproblems are tightly coupled in rephotography, as spatial composition and image appearance must be controlled jointly to produce a coherent final capture. For viewpoint control, novel view synthesis and camera-aware generation methods [2], [3], [4], [5], [6], [7], [8], [9], [10] synthesize target views from sparse or single inputs. These methods have demonstrated strong novelview generation capabilities and effective camera motion control. When directly applied to focal-length control, however, they may produce incorrect fields of view, object deformation, blurred details, or unfaithful over-generation artifacts. For focal-length control, generative super-resolution and restoration methods [11], [12], [13], [14], [15], [16] are effective at enhancing enlarged images and recovering fine details, but they do not ensure geometric consistency during continuous focal-length changes. For photographic effects, instruction-guided image editing methods [17], [18], [19], [20], [21], [22] can modify image appearance, and recent generative photography methods further explore numerical or parameterized control [23], [24], [25], [26], but they remain largely confined to the 2D image plane and cannot be integrated with camera-geometry control. Consequently, despite substantial progress along each individual dimension, jointly controlling viewpoint, focal length, and photographic effects within a unified framework for single-image rephotography remains unexplored.

These limitations arise from two fundamental issues. First, 3D reconstruction from a single image is inherently ill-posed [27], as the same 2D observation can correspond to multiple plausible 3D geometries. Consequently, geometry estimation errors are unavoidable and can lead to inaccurate projections of object shapes and spatial relationships onto the target image plane. Second, focal-length enlargement is sampling limited, as continuous focal-length changes introduce different degrees of sparsity in the target image, requiring details increasingly beyond the original sampling density of the reference image. Therefore, focal-length control must recover these missing details while preserving geometric consistency across this range. These issues further hinder the establishment of a stable 3D-aware backbone for integrating photographic effects with camera geometry, even though such effects are key elements of real photography and often depend heavily on photographic experience.

![](images/1bdc3a04e6a6659f16571cc29beb21da65267a6de6e8f380d8b4c34f7cd9bd9a.jpg)  
Fig. 1. Overview of ReX-Shot for single-image rephotography. Left: Multidimensional rephotography capabilities. ReX-Shot virtually recaptures the reference scene as target shots with controllable camera geometry and photographic effects, including viewpoint, focal length, color temperature, exposure, saturation, and contrast. Right: Quantitative performance comparison. ReX-Shot supports viewpoint, focal-length, and photographic-effect control while outperforming representative baselines from different method categories. All results in the radar chart are evaluated on the DL3DV-140 [1] test set. For each metric, we globally normalize the scores across the three tasks, and the numerical labels on the concentric grid rings shown in the Viewpoint section also apply to the same metric in the Focal Length and Photographic Effects sections. The thick solid, thin solid, dashed, and dotted lines denote ReX-Shot, Novel View Synthesis, Super Resolution, and Instruction-Guided Editing, respectively.

To address the aforementioned issues, we introduce two key designs. First, we propose dual-space geometry and camera grounding. We observe that, although recent stateof-the-art visual geometry models [28], [29], [30] provide strong scene reconstruction performance, geometric inaccuracies remain unavoidable, and even small structural errors can be amplified into noticeable local offsets during pixel-space explicit projection. To mitigate this pixelspace projection bias, we introduce feature-space implicit projection that learns an implicit transformation of VGGT and DINOv2 [31] features from the reference view to the target view. Leveraging robust foundation-model features, this projection provides context-aggregated semantic and geometric cues beyond explicit point-wise correspondences, leading to more stable target-view guidance under imperfect geometric projection.

Second, we observe that increasing the focal length reduces the spatial sampling density, making detail recovery naturally related to super resolution. Therefore, to address the sampling limitation, we leverage the detail prior of a super-resolution model [11], [12], [13], [16], which has been used in 2D image enhancement but is rarely explored for alleviating sparsity caused by 3D scene resampling. We fine-tune a one-step generative super-resolution model with dual-space conditions, allowing missing details to be recovered with geometric guidance while preserving inference efficiency.

These two designs establish a 3D-aware generative backbone that supports content-consistent synthesis across viewpoints and focal lengths. Building on this backbone, we further incorporate photographic-effect control. Specifically, we continuously parameterize each photographic effect and encode it with a dedicated encoder, whose control signal is then injected into the backbone. In this way, photographiceffect control is lifted from conventional 2D filtering to 3Daware appearance editing that preserves content consistency across viewpoints and focal lengths.

Together, these components form ReX-Shot, the first unified framework for single-image rephotography, as illustrated in Fig. 1. Our contributions are summarized as follows:

We define single-image rephotography as a virtual reshooting process, which recaptures the reference scene with specified viewpoints, focal lengths, and parameterized photographic effects.

We identify two fundamental issues behind singleimage rephotography, namely the ill-posed nature of single-image 3D reconstruction and the sampling limitation under continuous focal-length enlargement.

We propose two key designs to address these issues, i.e., feature-space implicit projection for mitigating projection bias caused by imperfect geometry, and a generative super-resolution prior for geometryguided restoration under focal-length changes.

We substantiate that ReX-Shot outperforms representative baselines across viewpoint control, focallength control, and photographic-effect control, while enabling near-real-time interactive rephotography.

## 2 RELATED WORK

## 2.1 Camera-Controlled View Synthesis

Novel view synthesis has progressed from per-scene optimized 3D representations to generalizable reconstruction and view generation models. Classical neural rendering methods, such as NeRF [32] and 3D Gaussian Splatting [33], optimize a scene-specific radiance field or Gaussian representation from posed images and render novel views from the learned scene. While effective, such per-scene optimization typically requires multiple observations and is not directly suited to single-image rephotography. Beyond per-scene optimization, recent methods have explored predicting scene representations or target views from sparse or single inputs. Representative feed-forward Gaussian reconstruction methods, including pixelSplat [34], MVSplat [35], and AnySplat [6], predict explicit Gaussian scene representations from image pairs, sparse multi-view inputs, or unconstrained views, retaining a renderable 3D proxy while improving generalization. Transformer-based novel-viewsynthesis methods such as LVSM [7] synthesize target views from sparse posed images, while RayZer [36] follows the same line by jointly estimating cameras and scene representations from unposed and uncalibrated inputs.

Recent generative models also expose camera geometry as a condition for view generation. MotionCtrl [5] controls video generation with camera pose sequences, while CameraCtrl [10] and CAT3D [3] condition diffusion generation with dense camera ray maps. Stable Virtual Camera [37] generalizes this setting to any number of input views and target cameras. Another line of geometry-guided generative methods uses 3D cues to steer diffusion-based view generation. Methods with explicit geometric conditions, including GenWarp [38], MultiDiff [39], and ViewCrafter [8], warp or project the reference image with estimated depth or point-based geometry before using diffusion priors to complete missing content. Geometry-aware generators such as GeNVS [40] and PE-Field [9] instead inject 3D structure into the generative model through latent 3D feature volumes or 3D positional encodings.

These methods substantially advance controllable view generation, but they mainly target viewpoint or cameratrajectory control and do not explicitly handle focal-length changes. ReX-Shot instead represents viewpoint and focal length jointly as target camera geometry, grounding generation through pixel-space explicit projection and featurespace implicit projection.

## 2.2 Image Editing and Generative Photography

Generative image editing provides a flexible route for controllable appearance modification by translating naturallanguage instructions into image changes. Early instructionguided methods mainly obtain edit supervision from paired editing data, with InstructPix2Pix [17] learning from synthetic instruction-image pairs and MagicBrush [18] improving this paradigm with manually annotated real-image editing triplets. Subsequent work strengthens the data-driven formulation along two complementary directions. Emu Edit [19] broadens the supervision signal with recognitionand-generation tasks and a richer taxonomy of editing operations, and AnyEdit [20] further scales high-quality multimodal instruction-editing pairs to cover diverse edit types and improve cross-task generalization. In parallel, methods that enhance instruction understanding use richer language or multimodal context to handle more complex user intents. MGIE [21] leverages multimodal large language models to produce expressive editing guidance, while ICEdit [22] formulates image editing as in-context generation in a largescale diffusion transformer.

Generative photography further studies how to expose photographic controls to generative models. Camera Settings as Tokens [23] encodes numerical photographic-effect parameters into latent diffusion models, enabling both textto-image generation and image editing under specified photographic settings. Generative Photography [24] focuses on text-to-image synthesis and improves scene-consistent photographic control through camera-condition embeddings. In the image-to-image setting, CamEdit [25] introduces continuous prompting with photographic-effect parameters and parameter-aware modulation for fine-grained appearance editing. CameraMaster [26] further explores unified photographic retouching by incorporating both semantic camera directives and numerical parameter conditions for compositional control over multiple photographic effects.

Instruction-guided editing supports broad semantic changes but lacks precise parameterized control over photographic effects. Generative photography moves closer by conditioning on numerical photographic-effect parameters, yet most methods still rely on the disentanglement of scene content and photographic-effect parameters within textual instructions and remain largely confined to the 2D image plane. In contrast, ReX-Shot anchors scene content with deterministic geometry and composes parameterized photographic-effect control with viewpoint and focal-length changes.

## 2.3 Diffusion Priors for Restoration

Diffusion priors have become a strong backbone for realworld image super resolution and restoration. Representative methods leverage pretrained text-to-image diffusion models as image priors for detail recovery. Among them, StableSR [11] balances fidelity and perceptual quality with a time-aware encoder, DiffBIR [13] combines an initial reconstruction network with Stable Diffusion [42] detail synthesis, and SeeSR [12] improves semantic fidelity with degradation-aware prompts. More recent works reduce inference cost through diffusion distillation and one-step formulations. SinSR [43] uses consistency-preserving distillation to perform super resolution in a single step, AddSR [44] introduces adversarial diffusion distillation for blind super resolution, and OSEDiff [15] starts diffusion directly from the low-quality input to avoid stochastic uncertainty. TSD-SR [16] further introduces target score distillation for stronger detail recovery and faster inference.

Generative priors have also been used to compensate for incomplete 3D evidence in sparse-view reconstruction and novel view synthesis. ReconFusion [45] regularizes fewview NeRF reconstruction with diffusion-based novel-view priors, while ViewCrafter [8] combines coarse point-based geometry with a video diffusion prior to synthesize highfidelity novel views and progressively extend the covered view range. ReconX [46] constructs a global point cloud from sparse input views, uses it as a 3D structure condition for video diffusion, and then optimizes a 3D Gaussian Splatting representation from the generated sequence. For 3D Gaussian Splatting restoration, DiFiX3D+ [47] and GS-Fixer [48] use diffusion priors to repair artifacts caused by underconstrained or degraded 3D representations.

![](images/c9153a12290805338122ed42e46a5897aab81cdb8892cd1a1d9e6013c2fd56d2.jpg)  
Fig. 2. Pipeline of ReX-Shot. Given a single reference image, ReX-Shot first estimates a per-pixel point cloud and camera parameters, and then synthesizes rephotographed images with controllable viewpoint, focal length, and photographic effects. The pipeline comprises two stages. 1. Dualspace geometry and camera grounding combines pixel-space projection, which renders the estimated point cloud under the target camera, with feature-space projection, which implicitly transforms VGGT [29] and DINOv2 [31] features into camera-aware tokens through PRoPE-based crossattention [41]. 2. Generative rephotography leverages the detail prior of a one-step generative super-resolution model to recover details missing from sparse 3D resampling under the guidance of dual-space conditions, while a parameterized photographic-effect encoder injects continuous effect controls into the DiT blocks. Snowflakes and flames denote frozen and trainable modules, respectively, and ⊕ denotes concatenation

ReX-Shot draws insights from these two categories of methods for single-image rephotography. Generative priors are useful for both recovering fine details required by focallength enlargement and synthesizing missing novel-view content when explicit 3D evidence is incomplete. ReX-Shot therefore adapts a generative super-resolution model to perform target-camera detail restoration and content completion.

## 3 METHOD

Fig. 2 provides an overview of the ReX-Shot pipeline. In the following, we first formulate single-image rephotography in Section 3.1, and then introduce dual-space geometry and camera grounding in Section 3.2, comprising pixel-space explicit projection in Section 3.2.1 and feature-space implicit projection in Section 3.2.2. We subsequently present generative rephotography with a super-resolution (SR) prior in Section 3.3, and parameterized photographic-effect control in Section 3.4.

## 3.1 Single-Image Rephotography

Given a reference image $I _ { r } \in [ 0 , 1 ] ^ { H \times W \times 3 }$ , ReX-Shot synthesizes a target image $\hat { I } _ { t }$ under a new shot specification. We denote the reference and target cameras by $c _ { r } = \left( K _ { r } , T _ { r } \right)$ and $c _ { t } ~ = ~ ( K _ { t } , T _ { t } )$ , respectively. Each camera is parameterized by an intrinsic matrix K and an extrinsic matrix $T = [ R \mid { \bf \bar { \theta } } t ]$ , where R and t denote rotation and translation, respectively. The target camera $c _ { t }$ specifies the desired shot by coupling focal-length control through $K _ { t }$ with viewpoint control through $T _ { t } .$ When photographic-effect control is requested, we further use an effect condition $e _ { \phi } = \alpha ,$ , where $\phi$ denotes the photographic-effect type and $\alpha ~ \in ~ [ 0 , 1 ]$ its normalized control value. The task is therefore written as

$$
\hat { I } _ { t } = \mathcal { F } _ { \Theta } ( I _ { r } , c _ { t } , e _ { \phi } ) , \quad e _ { \phi } = \mathcal { D } \mathrm { ~ w h e n ~ n o ~ e f f e c t ~ i s ~ a p p l i e d . }\tag{1}
$$

This formulation defines rephotography as target-shot formation governed by camera geometry and parameterized photographic effects. The model must preserve scene content and spatial relationships under the target viewpoint, recover details affected by focal-length-induced resampling, and realize the requested photographic effect when provided.

ReX-Shot realizes Eq. (1) through two stages, as illustrated in Fig. 2. The first stage establishes dual-space geometry and camera grounding. In pixel space, we use VGGT [29] to estimate a per-pixel point cloud from the reference image and reproject it under the target camera to produce a coarse geometry anchor. In feature space, we design a camera-aware feature projector that implicitly transforms VGGT and DINOv2 [31] features according to the relative projective relation between the reference and target cameras. The second stage feeds these complementary conditions to a one-step diffusion model initialized from TSD-SR [16], which completes missing regions and recovers fine details. Within the second stage, we introduce a dedicated encoder that injects the effect condition into the generative backbone to control the photographic effect.

## 3.2 Dual-Space Geometry and Camera Grounding

## 3.2.1 Pixel-Space Explicit Projection

The pixel-space branch explicitly projects the reconstructed scene into the target camera. Given the reference image $I _ { r } ,$ VGGT [29] predicts its depth map $D _ { r }$ and camera parameters $( K _ { r } , T _ { r } )$ . For each reference pixel $\boldsymbol { u } = ( x , y , \dot { 1 } ) ^ { \top }$ , we back-project it to a 3D point in the world coordinate system and project the point onto the target image plane:

$$
\begin{array} { r l } & { X ( u ) = R _ { r } ^ { \top } \left( D _ { r } ( u ) K _ { r } ^ { - 1 } u - t _ { r } \right) , } \\ & { u _ { t } ( u ) = \pi ( K _ { t } \left( R _ { t } X ( u ) + t _ { t } \right) ) , } \end{array}\tag{2}
$$

where $\pi ( [ p _ { x } , p _ { y } , p _ { z } ] ^ { \top } ) = ( p _ { x } / p _ { z } , p _ { y } / p _ { z } ) ^ { \top }$ denotes perspective division. We rasterize the projected points with $Z ^ { - }$ buffering to obtain a coarse target rendering $\widetilde { I } _ { t } .$ . Because the point cloud contains only samples visible in the reference image, $\widetilde { I } _ { t }$ can become incomplete due to viewpoint-induced disocclusion and sparse due to focal-length enlargement. More importantly, unavoidable inaccuracies in single-image geometry estimation can introduce local projection offsets in ${ \ddot { I } } _ { t } .$ . Consequently, it serves only as a coarse geometry anchor for subsequent generative completion and detail recovery.

## 3.2.2 Feature-Space Implicit Projection

Pixel-space projection provides an explicit geometry anchor, but its point-wise correspondences are sensitive to local offsets caused by unavoidable geometry inaccuracies. We therefore complement it with an implicit feature-space projection branch. Specifically, given $\begin{array} { r } { \dot { I } _ { r } \ \in \ [ 0 , 1 ] ^ { H \times 1 } W \times 3 } \end{array}$ , we extract the VGGT [29] and DINOv2 [31] feature maps as

$$
\begin{array} { r l } & { F _ { v } = { \mathcal { E } } _ { \mathrm { V G G T } } ( I _ { r } ) \in \mathbb { R } ^ { \frac { H } { 1 4 } \times \frac { W } { 1 4 } \times d _ { v } } , } \\ & { F _ { d } = { \mathcal { E } } _ { \mathrm { D I N O } } ( I _ { r } ) \in \mathbb { R } ^ { \frac { H } { 1 4 } \times \frac { W } { 1 4 } \times d _ { d } } , } \end{array}\tag{3}
$$

where $d _ { v }$ and $d _ { d }$ denote the respective channel dimensions. Following the patch-based tokenization of VGGT and DI-NOv2, each spatial location in these feature maps corresponds to a 14 × 14 image patch rather than an individual pixel. Moreover, interactions among tokens at different spatial locations allow each feature to aggregate information from a broader image region. These properties make the resulting representation more robust to local projection offsets. To combine the geometry-aware scene structure encoded by VGGT with the robust visual semantics captured by DINOv2, we concatenate the two feature maps channel-wise to obtain $\begin{array} { r } { F _ { r } = [ F _ { v } ; F _ { d } ] \in \mathbb { R } ^ { \frac { H } { 1 4 } \times \frac { W } { 1 4 } \times ( d _ { v } + d _ { d } ) } } \end{array}$

Changes in viewpoint and focal length alter scene visibility and the field of view, making some reference-view features irrelevant to the target view. We therefore design a camera-aware feature projector to transform $F _ { r }$ into targetview tokens while filtering irrelevant reference content. Specifically, the projector applies a stack of transformer blocks with PRoPE cross-attention [41], which encodes the reference and target camera parameters as relative positional encodings:

$$
\begin{array} { r l } & { \boldsymbol { Q } ^ { ( \ell ) } = \boldsymbol { \mathcal { B } } _ { \psi } ^ { ( \ell ) } \left( \boldsymbol { Q } ^ { ( \ell - 1 ) } ; { \boldsymbol { F } } _ { r } , { \mathcal { P } } \right) , \quad \ell = 1 , \ldots , L _ { p } , } \\ & { \boldsymbol { \mathcal { B } } _ { \psi } ^ { ( \ell ) } = \mathrm { F F N } ^ { ( \ell ) } \circ \mathrm { C A } _ { \mathrm { P R o P E } } ^ { ( \ell ) } \circ \mathrm { S A } ^ { ( \ell ) } , \quad \hat { \boldsymbol { F } } _ { t } = \boldsymbol { Q } ^ { ( L _ { p } ) } , } \end{array}\tag{4}
$$

where $Q ^ { ( 0 ) }$ denotes a set of learnable query tokens that are randomly initialized before training, $\bar { \mathcal { P } } = \left( c _ { r } , c _ { t } \right)$ denotes the reference–target camera pair, $\mathop { B _ { \psi } ^ { ( \ell ) } }$ is the ℓ-th transformer block, and $L _ { p }$ is the number of projector blocks. The operators $\mathrm { S A } , \mathrm { C A } _ { \mathrm { P R o P E } } ^ { \bullet } ,$ and FFN denote self-attention, PRoPE cross-attention, and the feed-forward network, respectively, and $\hat { F } _ { t }$ denotes the projected target-view tokens. This transformation is performed entirely in feature space and does not require explicit scene geometry. It is therefore less sensitive to local geometry errors and provides more stable target-view guidance.

However, if we jointly train the feature projector with the generative backbone using only image-level supervision, it may fail to learn the intended camera transformation, forcing the generative backbone to locate target-camerarelevant cues directly within the reference-view features. We therefore pretrain the projector against paired target-view features $\bar { F _ { t } } ,$ which are extracted from the paired target image $I _ { t }$ using the same frozen VGGT and DINOv2 encoders as $F _ { r }$ . This direct feature-level supervision encourages $\hat { F } _ { t }$ to align with the target camera before it is used to guide generation. For N patch tokens, we minimize the cosine similarity loss

$$
\mathcal { L } _ { \mathrm { a l i g n } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( 1 - \frac { \left. \hat { F } _ { t } ^ { ( i ) } , F _ { t } ^ { ( i ) } \right. } { \left\| \hat { F } _ { t } ^ { ( i ) } \right\| _ { 2 } \left\| F _ { t } ^ { ( i ) } \right\| _ { 2 } } \right) ,\tag{5}
$$

while keeping the VGGT and DINOv2 encoders frozen. This objective trains the projector to map reference-view features to target-camera-aligned tokens.

## 3.3 Generative Rephotography with an SR Prior

Focal-length enlargement creates a detail-recovery problem closely related to super-resolution (SR). Let $\rho _ { r }$ and $\rho _ { t }$ denote the densities of available reference samples before and after reprojection, respectively. Under focal-length enlargement, the spatial coordinates are scaled by $s _ { f } ~ = ~ f _ { t } / f _ { r }$ , and the corresponding sample density becomes

$$
\rho _ { t } = \frac { \rho _ { r } } { s _ { f } ^ { 2 } } , \qquad s _ { f } = \frac { f _ { t } } { f _ { r } } > 1 .\tag{6}
$$

An SR model with scale factor s reconstructs $s ^ { 2 }$ output samples per input sample, corresponding to an output sample density

$$
\rho _ { \mathrm { H R } } = s ^ { 2 } \rho _ { \mathrm { L R } } ,\tag{7}
$$

where $\rho _ { \mathrm { L R } }$ and $\rho _ { \mathrm { H R } }$ denote the input and output sample densities of the SR model, respectively. For focal-length enlargement, setting $\rho _ { \mathrm { L R } } ~ = ~ \rho _ { t }$ and $\begin{array} { r } { s = s _ { f } } \end{array}$ gives ρ<sub>HR</sub> = $s _ { f } ^ { 2 } \rho _ { t } \stackrel { \sim } { = } \rho _ { r }$ . Therefore, focal-length enlargement and SR share the same detail-recovery requirement under spatial magnification. In our pipeline, the scene is rendered directly under the target camera, and the SR prior restores the resulting coarse rendering by recovering details missing between sparsely projected samples. Viewpoint changes additionally reveal regions that are occluded or unobserved in the reference image, extending the task from detail restoration to geometry-aware completion. We therefore adapt the onestep TSD-SR model [16] as our generative backbone to address both requirements. The adapted generator is jointly conditioned on the pixel-space rendering $\widetilde { I } _ { t }$ and the featurespace projected tokens $\hat { F } _ { t }$ . Let ${ \mathcal E } _ { \mathrm { v a e } }$ and $\mathcal { D } _ { \mathrm { v a e } }$ denote the VAE encoder and decoder, respectively, and let $\tau$ be the fixed denoising timestep. The dual-space-conditioned onestep generation process is defined as

$$
z _ { t } = \xi _ { \mathrm { v a e } } ( \widetilde { I } _ { t } ) , \qquad \hat { I } _ { t } = \mathcal { D } _ { \mathrm { v a e } } \left( z _ { t } - \epsilon _ { \theta } ( z _ { t } , \tau ; \hat { F } _ { t } ) \right) .\tag{8}
$$

We take the latent code of the pixel-space rendering $\widetilde { I } _ { t }$ as the degraded input latent $z _ { t }$ at the fixed timestep $\tau$ and use the adapted generative backbone to predict and remove the degradation residual in a single denoising step. The featurespace condition $\hat { F } _ { t }$ is injected into the transformer attention layers to provide robust target-view guidance beyond the incomplete and locally misaligned rendering.

Specifically, $\hat { F } _ { t }$ is injected through an additional attention branch in every transformer layer. In parallel to the original attention output $A _ { \ell } ^ { \mathrm { o r i g } }$ , the feature tokens are normalized and linearly mapped to additional keys and values:

$$
\begin{array} { r l } & { k _ { \ell } ^ { F } = W _ { k , \ell } ^ { F } \mathrm { N o r m } ( \hat { F } _ { t } ) , \quad v _ { \ell } ^ { F } = W _ { v , \ell } ^ { F } \mathrm { N o r m } ( \hat { F } _ { t } ) , } \\ & { A _ { \ell } ^ { F } = \mathrm { A t t n } \left( q _ { \ell } ^ { h } , [ k _ { \ell } ^ { h } ; k _ { \ell } ^ { F } ] , [ v _ { \ell } ^ { h } ; v _ { \ell } ^ { F } ] \right) , } \end{array}\tag{9}
$$

where $q _ { \ell } ^ { h } , \ k _ { \ell } ^ { h }$ , and $v _ { \ell } ^ { h }$ denote the query, key, and value corresponding to the image-latent tokens, while $k _ { \ell } ^ { F }$ and $v _ { \ell } ^ { F }$ are the feature keys and values derived from $\hat { F } _ { t } ^ { \phantom { \dagger } } ,$ , respectively. The feature keys and values are concatenated with the image keys and values, respectively, along the token dimension before attention. The two attention outputs are combined as

$$
\widetilde { A } _ { \ell } = A _ { \ell } ^ { \mathrm { o r i g } } + A _ { \ell } ^ { F } .\tag{10}
$$

This additional attention branch allows the image-latent tokens to interact directly with the projected foundationmodel feature tokens at every transformer layer, while retaining the original attention pathway to preserve the pretrained generative prior.

During training, we fine-tune the TSD-SR-initialized generator with the paired target image $I _ { t }$ as supervision. We combine mean squared error and LPIPS perceptual distance [49] with equal weights:

$$
\mathcal { L } _ { \mathrm { r e p h o t o } } = \mathcal { L } _ { \mathrm { M S E } } + \mathcal { L } _ { \mathrm { L P I P S } } .\tag{11}
$$

This objective balances pixel-level fidelity and perceptual quality.

## 3.4 Parameterized Photographic-Effect Control

Alongside camera geometry, photographic effects are key elements of photography that govern image appearance. We therefore incorporate parameterized photographic-effect control into the 3D-aware generative backbone introduced above, enabling the model to preserve content consistency across viewpoints and focal lengths. For each effect type $\phi ,$ the corresponding photographic-effect encoder consists of a positional encoding MLP (PE-MLP) and a ControlNet-style auxiliary branch [50] comprising DiT blocks [51] with the same architecture as those in the generative backbone. The PE-MLP maps the normalized control value α to an effect embedding $z _ { e }$ using Fourier features [52]:

$$
\begin{array} { r } { \gamma ( \alpha ) = \left[ \alpha , \{ \sin ( \omega _ { i } \alpha ) , \cos ( \omega _ { i } \alpha ) \} _ { i = 1 } ^ { N _ { f } } \right] , } \\ { z _ { e } = M _ { \phi } ( \gamma ( \alpha ) ) , \qquad } \end{array}\tag{12}
$$

where $M _ { \phi }$ denotes the PE-MLP and $\{ \omega _ { i } \} _ { i = 1 } ^ { N _ { f } }$ denotes a set of fixed multi-scale frequencies. The Fourier encoding provides a multi-scale representation of $\alpha ,$ allowing the PE-MLP to distinguish nearby control values while modeling smooth, nonlinear effect variations. We then tile the resulting effect embedding over the latent spatial grid and apply patch embedding to obtain the initial effect tokens $g _ { 0 } ^ { e } \in \mathbb { R } ^ { \lambda _ { e } \times d _ { h } }$ , with $N _ { e }$ and $d _ { h }$ matching the token count and hidden dimension of the main DiT branch, respectively.

The photographic-effect encoder is coupled with the main DiT layer by layer. Starting from $g _ { 0 } ^ { e } ,$ , its ℓ-th DiT block produces a guidance signal $g _ { \ell } ^ { e } .$ , which is injected after the corresponding main DiT block:

$$
\begin{array} { r l } & { g _ { \ell } ^ { e } = \mathrm { D i T B l o c k } _ { \ell } ^ { e } ( g _ { \ell - 1 } ^ { e } ) , } \\ & { h _ { \ell } = \mathrm { D i T B l o c k } _ { \ell } ( h _ { \ell - 1 } ) + g _ { \ell } ^ { e } , \qquad \ell = 1 , \dots , L _ { e } . } \end{array}\tag{13}
$$

We train separate photographic-effect encoders for color temperature, exposure, saturation, and contrast using the reconstruction objective in Eq. (11) on the corresponding effect-conditioned target images. Throughout this stage, we freeze the main generative backbone and optimize only the corresponding photographic-effect encoder.

In this way, camera geometry (viewpoint and focal length) is handled by the main generative backbone, while the photographic-effect encoder leverages this 3D-aware backbone to maintain consistent effect control across viewpoints and focal lengths.

## 4 EXPERIMENTS

We evaluate ReX-Shot from five perspectives: camerageometry control, photographic-effect control, joint control, inference efficiency, and component contributions. We first describe the implementation details in Section 4.1 and the experimental protocols in Section 4.2. We then evaluate viewpoint and focal-length control in Section 4.3, followed by photographic-effect and joint control in Section 4.4. Finally, Section 4.5 reports inference latency, and Section 4.6 analyzes the contribution of the proposed components.

## 4.1 Implementation Details

Dataset. We randomly split the 140 scenes of DL3DV-140 [1] into 112 scenes for training and 28 scenes for testing. For each training scene, we group every nine consecutive frames into a cluster, obtaining 4,415 clusters in total. Within each cluster, the center frame serves as the reference input, while each of the other eight frames provides a target view, resulting in 35,320 reference-target pairs. To simulate focal-length changes, we independently sample the focallength scale $s _ { f }$ for each target from the continuous interval [1.0, 3.0] and generate the corresponding ground truth via center cropping. For photographic-effect training, we sample color temperature from [2000, 10000] $\mathrm { K } ,$ exposure scale from [0.2, 2.0], saturation scale from [0.0, 2.0], and contrast scale from [0.5, 2.0]. Each sampled value is normalized to [0, 1] using its corresponding control range before being passed to the photographic-effect encoder. We evaluate the model on the DL3DV-140 test split and additionally on MipNeRF360 [53] and Tanks & Temples [54] to assess crossdataset generalization.

Reference View

AnySplat

LVSM

ViewCrafter

PE-Field

ReX-Shot

Target View

![](images/297fff3a96d46fbec679931eb3d81dd429887dc812f5f852ce6b57f337f6c1c0.jpg)  
Fig. 3. Qualitative comparison with novel-view-synthesis baselines under viewpoint and focal-length control. For each example, the first row shows results at the target viewpoint with the focal-length scale fixed at 1.0×. The second and third rows keep the target viewpoint fixed and enlarge the focal length by 2.0× and 3.0×, respectively. The input reference and ground-truth target images are shown in the leftmost and rightmost columns, respectively.

Training details. We implement ReX-Shot on top of Stable Diffusion 3 Medium [51] and initialize the generative backbone from the released TSD-SR [16] checkpoint. We begin by pretraining the camera-aware feature projector using frozen VGGT [29] and DINOv2 [31] feature extractors. With the pretrained projector and the TSD-SR LoRA weights as initialization, we then fine-tune the generative backbone on the DL3DV-140 training split by jointly optimizing LoRA adapters [55] for the SD3 transformer, VAE encoder, and projector branch, along with the feature-injection attention processors. This fine-tuning stage uses 4 GPUs with a batch size of 8 per GPU, a learning rate of $1 \times 1 0 ^ { - 4 } .$ , 500 warmup steps, 8 epochs, and a training resolution of 448×252 pixels. After completing this stage, we freeze the SD3 transformer, VAE, and camera-aware feature projector, and train separate photographic-effect encoders, one for each effect. These encoders use the same training resolution and optimization settings as the preceding generative-backbone fine-tuning stage.

## 4.2 Experimental Protocols

Evaluation settings. Our evaluation covers camerageometry control, photographic-effect control, joint control, inference efficiency, and component contributions. For camera-geometry control, we evaluate viewpoint and focallength changes on DL3DV-140, MipNeRF360, and Tanks & Temples using PSNR [56], SSIM [57], and LPIPS [49]. For photographic-effect control, we evaluate color temperature, exposure, saturation, and contrast, and additionally report ∆E [58] to measure color accuracy. For joint control, we demonstrate ReX-Shot’s ability to jointly control camera geometry and photographic effects within a single generation by simultaneously varying viewpoint, focal length, and color temperature. For inference efficiency, we measure end-to-end latency and decompose it into reconstruction, rendering, and effect-editing time where applicable. For component contributions, we conduct ablation studies to isolate the effects of the TSD-SR initialization and the proposed camera-aware feature projector. In each quantitative table, the best, second-best, and third-best results for each metric are highlighted with , , and , respectively.

TABLE 1  
Quantitative comparison under joint viewpoint and focal-length control.
<table><tr><td rowspan="2">Method</td><td colspan="3">DL3DV</td><td colspan="3">MipNeRF360</td><td colspan="3">Tanks &amp; Temples</td><td colspan="3">Average</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>AnySplat</td><td>16.51</td><td>0.464</td><td>0.461</td><td>14.66</td><td>0.457</td><td>0.477</td><td>13.61</td><td>0.425</td><td>0.521</td><td>14.93</td><td>0.449</td><td>0.486</td></tr><tr><td>CameraCtrl</td><td>14.45</td><td>0.391</td><td>0.480</td><td>13.71</td><td>0.400</td><td>0.604</td><td>12.69</td><td>0.411</td><td>0.545</td><td>13.62</td><td>0.401</td><td>0.543</td></tr><tr><td>LVSM</td><td>18.87</td><td>0.526</td><td>0.383</td><td>15.48</td><td>0.498</td><td>0.636</td><td>15.62</td><td>0.514</td><td>0.511</td><td>16.66</td><td>0.513</td><td>0.510</td></tr><tr><td>ViewCrafter</td><td>15.84</td><td>0.417</td><td>0.424</td><td>16.13</td><td>0.443</td><td>0.453</td><td>14.18</td><td>0.433</td><td>0.487</td><td>15.38</td><td>0.431</td><td>0.455</td></tr><tr><td>PE-Field</td><td>16.34</td><td>0.429</td><td>0.427</td><td>15.95</td><td>0.443</td><td>0.486</td><td>14.31</td><td>0.433</td><td>0.485</td><td>15.53</td><td>0.435</td><td>0.466</td></tr><tr><td>ReX-Shot</td><td>23.05</td><td>0.679</td><td>0.137</td><td>19.93</td><td>0.557</td><td>0.276</td><td>18.07</td><td>0.533</td><td>0.275</td><td>20.35</td><td>0.590</td><td>0.229</td></tr></table>

TABLE 2

Quantitative comparison under viewpoint-only control.
<table><tr><td rowspan="2">Method</td><td colspan="3">DL3DV</td><td colspan="3">MipNeRF360</td><td colspan="3">Tanks &amp; Temples</td><td colspan="3">Average</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>AnySplat</td><td>18.07</td><td>0.638</td><td>0.223</td><td>12.39</td><td>0.479</td><td>0.419</td><td>12.65</td><td>0.526</td><td>0.386</td><td>14.37</td><td>0.548</td><td>0.343</td></tr><tr><td>CameraCtrl</td><td>17.63</td><td>0.481</td><td>0.273</td><td>15.56</td><td>0.438</td><td>0.503</td><td>14.53</td><td>0.469</td><td>0.428</td><td>15.91</td><td>0.463</td><td>0.401</td></tr><tr><td>LVSM</td><td>19.82</td><td>0.564</td><td>0.210</td><td>15.74</td><td>0.459</td><td>0.523</td><td>15.86</td><td>0.510</td><td>0.388</td><td>17.14</td><td>0.511</td><td>0.374</td></tr><tr><td>ViewCrafter</td><td>19.94</td><td>0.570</td><td>0.205</td><td>17.37</td><td>0.462</td><td>0.380</td><td>15.30</td><td>0.470</td><td>0.397</td><td>17.54</td><td>0.501</td><td>0.327</td></tr><tr><td>PE-Field</td><td>19.55</td><td>0.510</td><td>0.236</td><td>18.59</td><td>0.491</td><td>0.344</td><td>16.49</td><td>0.495</td><td>0.348</td><td>18.21</td><td>0.499</td><td>0.309</td></tr><tr><td>ReX-Shot</td><td>23.13</td><td>0.703</td><td>0.122</td><td>20.18</td><td>0.548</td><td>0.245</td><td>17.87</td><td>0.544</td><td>0.244</td><td>20.39</td><td>0.598</td><td>0.204</td></tr></table>

TABLE 3

Quantitative comparison under focal-length-only control.
<table><tr><td rowspan="2">Method</td><td colspan="3">DL3DV</td><td colspan="3">MipNeRF360</td><td colspan="3">Tanks &amp; Temples</td><td colspan="3">Average</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>AnySplat</td><td>20.21</td><td>0.676</td><td>0.398</td><td>25.39</td><td>0.797</td><td>0.285</td><td>20.90</td><td>0.695</td><td>0.365</td><td>22.17</td><td>0.723</td><td>0.349</td></tr><tr><td>CameraCtrl</td><td>15.65</td><td>0.430</td><td>0.409</td><td>15.20</td><td>0.426</td><td>0.488</td><td>14.50</td><td>0.448</td><td>0.433</td><td>15.12</td><td>0.435</td><td>0.443</td></tr><tr><td>LVSM</td><td>22.16</td><td>0.628</td><td>0.259</td><td>24.37</td><td>0.689</td><td>0.279</td><td>22.98</td><td>0.681</td><td>0.247</td><td>23.17</td><td>0.666</td><td>0.262</td></tr><tr><td>ViewCrafter</td><td>16.40</td><td>0.428</td><td>0.427</td><td>19.70</td><td>0.551</td><td>0.403</td><td>15.90</td><td>0.486</td><td>0.408</td><td>17.33</td><td>0.488</td><td>0.413</td></tr><tr><td>PE-Field</td><td>17.08</td><td>0.453</td><td>0.390</td><td>18.47</td><td>0.509</td><td>0.359</td><td>16.37</td><td>0.489</td><td>0.389</td><td>17.31</td><td>0.484</td><td>0.379</td></tr><tr><td>OSEDiff</td><td>23.22</td><td>0.696</td><td>0.208</td><td>24.39</td><td>0.698</td><td>0.236</td><td>23.66</td><td>0.725</td><td>0.194</td><td>23.76</td><td>0.706</td><td>0.213</td></tr><tr><td>TSD-SR</td><td>24.02</td><td>0.703</td><td>0.166</td><td>26.19</td><td>0.717</td><td>0.165</td><td>25.37</td><td>0.743</td><td>0.146</td><td>25.19</td><td>0.721</td><td>0.159</td></tr><tr><td>ReX-Shot</td><td>30.00</td><td>0.878</td><td>0.058</td><td>28.57</td><td>0.846</td><td>0.128</td><td>28.83</td><td>0.859</td><td>0.098</td><td>29.13</td><td>0.861</td><td>0.095</td></tr></table>

Baselines. The baselines are selected according to the capability required by each protocol. For camera-geometry control, we compare with AnySplat [6], LVSM [7], CameraCtrl [10], ViewCrafter [8], and PE-Field [9] to examine whether existing novel-view-synthesis methods remain reliable when the viewpoint and focal length change simultaneously. To disentangle these two factors, we also evaluate these baselines under a viewpoint-only setting and a focallength-only setting. OSEDiff [15] and TSD-SR [16] are additionally included in the focal-length-only setting because focal-length enlargement poses a detail-recovery problem closely related to super resolution. For photographic-effect control, we compare with IP2P [17] and ICEdit [22] as general-purpose baselines because they support a broad range of appearance edits, and photographic effects can also be described using natural-language instructions. For inference efficiency, we compare ReX-Shot with representative feed-forward and generative methods. We use the official implementations and released checkpoints for all baselines with their recommended inference settings. All baselines are evaluated on the same reference-target pairs.

## 4.3 Viewpoint and Focal Length Control

We evaluate geometry control by varying viewpoint and focal length within the test-set clusters. Each cluster contains nine neighboring viewpoints, and each viewpoint is evaluated under nine uniformly sampled focal-length scales from 1.0× to 3.0×. From these reference-target pairs, we derive three evaluation settings. The viewpoint-only protocol uses samples whose focal length remains at 1.0× while the target viewpoint changes. The focal-length-only protocol keeps the viewpoint fixed to the reference view and evaluates the remaining focal-length scales. The joint viewpoint and focal length protocol uses the samples where both the target viewpoint and the focal length differ from those of the reference view. Together, these three protocols comprehensively evaluate viewpoint and focal-length control, both independently and in combination.

Fig. 3 evaluates whether state-of-the-art novel-viewsynthesis methods can extend beyond standard viewpoint transfer to handle focal-length changes, thereby supporting comprehensive camera-geometry control. The first row of each example follows the viewpoint-only protocol, where the target viewpoint changes while the focal-length scale remains fixed at 1.0×. In this case, ReX-Shot achieves visual quality comparable to strong novel-view-synthesis baselines under the single-image setting. The second and third rows are more diagnostic, with the target viewpoint fixed to that in the first row while the focal length is enlarged by 2.0× and 3.0×. AnySplat can render geometrically plausible content using its predicted explicit scene representation and camera parameters, but it cannot synthesize content in previously unobserved regions that become visible under viewpoint changes, including areas occluded or outside the reference field of view. Focal-length enlargement further produces visible gaps between projected primitives. LVSM, meanwhile, exhibits grid-like artifacts. ViewCrafter hallucinates unsupported textures and distorts object geometry, whereas PE-Field produces an incorrect field of view. These results highlight the limitations of existing novel-viewsynthesis methods in handling focal-length changes for rephotography. In contrast, ReX-Shot plausibly synthesizes content in disoccluded and previously out-of-view regions under viewpoint changes and recovers fine-grained texture details under focal-length enlargement.

![](images/ed3b2d53a24d267b3e4ca16429fda97a52362b2e01655fcca10d2cccaf522c91.jpg)  
Fig. 4. Qualitative focal-length-only comparison between TSD-SR and ReX-Shot under continuous focal-length enlargement.

The quantitative results in Tab. 1 and Tab. 2 reinforce these qualitative observations. Under joint viewpoint and focal-length control, ReX-Shot outperforms all compared methods across the three datasets, demonstrating that correct focal-length control cannot be obtained by directly applying novel-view-synthesis baselines. In the viewpointonly setting, ReX-Shot remains comparable to strong baselines such as PE-Field, ViewCrafter, and LVSM, indicating that ReX-Shot retains strong novel-view-synthesis performance while supporting focal-length control.

The focal-length-only experiment isolates focal-length control while keeping the camera viewpoint unchanged. This setting is evaluated against all novel-view-synthesis baselines as well as generative super-resolution baselines, with particular attention to OSEDiff and TSD-SR, as they deliver strong results in Tab. 3. Nevertheless, a successful method should not only recover fine-grained details during focal-length enlargement, but also preserve scene geometry and maintain consistency with the reference observation across continuous focal-length changes. We therefore select TSD-SR as a representative method and compare it with ReX-Shot in Fig. 4. TSD-SR treats focal-length enlargement as a 2D restoration problem. Its strong generative prior can enhance fine details, but without explicit geometric guidance, it may hallucinate implausible textures and compromise geometric consistency. ReX-Shot instead provides geometry- and camera-grounded guidance for focal-length control, allowing its generative prior to recover fine details while better preserving geometric consistency and scene fidelity. This advantage is reflected in the best PSNR, SSIM, and LPIPS in Tab. 3.

TABLE 4  
Quantitative comparison under photographic-effect control.
<table><tr><td>Metric</td><td>Effect</td><td>ICEdit</td><td>IP2P</td><td>ReX-Shot</td></tr><tr><td rowspan="5">PSNR↑</td><td>Color Temp. Exposure</td><td>23.28 17.25</td><td>20.23</td><td>31.72 30.18</td></tr><tr><td></td><td></td><td>16.13</td><td></td></tr><tr><td>Saturation</td><td>19.92</td><td>27.75</td><td>30.13</td></tr><tr><td>Contrast</td><td>19.86</td><td>20.64</td><td>28.59</td></tr><tr><td>Average</td><td>20.08</td><td>21.19</td><td>30.16</td></tr><tr><td rowspan="6">SSIM↑</td><td>Color Temp.</td><td>0.909</td><td>0.781</td><td>0.913</td></tr><tr><td>Exposure</td><td>0.791</td><td>0.697</td><td>0.907</td></tr><tr><td>Saturation</td><td>0.814</td><td>0.832</td><td>0.905</td></tr><tr><td>Contrast</td><td>0.791</td><td>0.759</td><td>0.897</td></tr><tr><td>Average</td><td>0.826</td><td>0.767</td><td>0.906</td></tr><tr><td></td><td>0.199</td><td>0.270</td><td></td><td>0.152</td></tr><tr><td rowspan="5">LPIPS↓</td><td>Color Temp. Exposure</td><td>0.166</td><td>0.203</td><td>0.166</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Saturation</td><td>0.293</td><td>0.129</td><td>0.169</td></tr><tr><td>Contrast</td><td>0.212</td><td>0.193</td><td>0.163</td></tr><tr><td>Average</td><td>0.218</td><td>0.199</td><td>0.163</td></tr><tr><td rowspan="6">∆E↓</td><td>Color Temp.</td><td></td><td></td><td></td></tr><tr><td></td><td>13.71</td><td>18.23</td><td>3.033</td></tr><tr><td>Exposure</td><td>17.92</td><td>18.00</td><td>3.814</td></tr><tr><td>Saturation</td><td>17.07</td><td>7.092</td><td>3.715</td></tr><tr><td>Contrast</td><td>11.79</td><td>12.16</td><td>4.099</td></tr><tr><td>Average</td><td>15.12</td><td>13.87</td><td>3.665</td></tr></table>

## 4.4 Photographic-Effect and Joint Control

We evaluate parameterized photographic-effect control against IP2P and ICEdit. The two baselines edit images through natural-language instructions. For both methods, we use the prompt adjust the {photographic\_effect} of this image to {effect\_value}, replacing the placeholders with the target effect type and value. At test time, we uniformly sample several target values from 2000 K to 10000 K for color temperature, from 0.2× to 2.0× for exposure, from 0.0× to 2.0× for saturation, and from 0.5× to 2.0× for contrast.

The qualitative results in Figs. 5–8 reveal two observations. First, their responses to numerical instructions are unstable, with effect strengths often poorly aligned with the requested values and some inputs leading to severe failure cases, as shown in the first row of Fig. 7. Second, the edits may alter the underlying scene content instead of changing only the photographic effect, as illustrated by the unintended changes to the ceiling in the fourth column of the second row in Fig. 5 and to the sky in the first row of Fig. 6. ReX-Shot instead anchors scene content with deterministic geometry and introduces a parameterized photographic-<sub>C</sub><sup>E</sup>effect encoder for continuous effect control. This design decouples scene content from photographic-effect control, <sup>2</sup>avoiding unintended content changes while enabling accu-Irate and continuous effect control for rephotography. Tab. 4 Pconfirms the visual trend, with ReX-Shot achieving the best <sup>P</sup>average PSNR, SSIM, and LPIPS, as well as a substantially <sub>h</sub><sup>o</sup>lower average ∆E than both IP2P and ICEdit.

![](images/5622c00701d1e722e91ff1519eb0dd20388663fa61040d4babf9fc8cd071fdb9.jpg)

Fig. 5. Qualitative comparison with instruction-guided image editing baselines under color-temperature control.  
![](images/38b4c8af7e3abf566400ec9e639d537f3d832212b5715149b96ae916b4c52820.jpg)  
<sup>E</sup>Fig. 6. Qualitative comparison with instruction-guided image editing baselines under exposure control.

Fig. 10 further presents an example of joint camera-<sup>h</sup>geometry and photographic-effect control. Rather than ap-X<sup>-</sup>plying these controls through independent post-processing Rsteps, ReX-Shot realizes them jointly within a unified gener-<sup>T</sup> ation process. Enabled by its 3D-aware generative backbone and parameterized photographic-effect encoder, ReX-Shot produces geometrically consistent results across viewpoints and focal lengths while faithfully realizing the requested photographic effect.

## 4.5 Inference Efficiency

Fig. 9 compares the end-to-end inference latency of ReX-Shot with feed-forward methods (LVSM and AnySplat) and generative methods (CameraCtrl, IP2P, ICEdit, ViewCrafter, and PE-Field). Where applicable, we decompose the endto-end latency into reconstruction, target-camera rendering, and photographic-effect editing. Notably, although

![](images/67475d2343fc63d5f9ade074024db0cf72e5233092f875a4b59c9a9da54d70e8.jpg)

Fig. 7. Qualitative comparison with instruction-guided image editing baselines under saturation control.  
![](images/3ea2ee1dbaedcc2e4a5fb71986cc55d5d189d4bc05e62e276d0164bb6a6bf4cb.jpg)  
Fig. 8. Qualitative comparison with instruction-guided image editing baselines under contrast control.

LVSM [7] and CameraCtrl [10] do not directly rely on 3D reconstruction in their pipelines, their camera control still requires calibrated camera parameters, which are commonly estimated together with scene structure by structurefrom-motion pipelines [59]. Leveraging the efficiency of its one-step generative backbone, ReX-Shot achieves end-toend inference latency comparable to feed-forward methods and substantially lower than that of other generative methods, while jointly supporting viewpoint, focal-length, and photographic-effect control. Moreover, after a one-time reconstruction of the reference image, the resulting scene representation can be reused across target camera and effect settings, so subsequent outputs require only rendering and effect control. Rendering alone runs at approximately 23 FPS, while rephotography with photographic-effect control runs at approximately 16 FPS, enabling near-real-time interaction.

## 4.6 Ablation Study

The ablation study evaluates the contributions of the superresolution prior and the camera-aware feature projector. Fig. 11 compares SD3 fine-tuning (SD3-FT), TSD-SR finetuning (TSD-SR-FT), and TSD-SR fine-tuning with the camera-aware feature projector (TSD-SR-FT+FP). SD3-FT often produces blurry and noisy results with noticeable local offsets. TSD-SR-FT improves detail recovery and perceptual quality, demonstrating the benefit of the super-resolution prior, but local offsets caused by geometric inaccuracies persist. TSD-SR-FT+FP further incorporates the camera-aware feature projector into TSD-SR-FT, preserving the fine-detail recovery of the TSD-SR prior while producing target-viewaligned results.

![](images/01fa6fd53d1fcb38346711ed358f8cc1ab0d84cb2d58cca929d51cc42bcbfcb0.jpg)  
Fig. 9. Inference latency comparison between ReX-Shot and representative feed-forward and generative methods. When applicable, end-to-end latency is decomposed into reconstruction, rendering for camera-geometry control, and effect editing for photographic-effect control.

TABLE 5  
Quantitative comparison across SD3-FT, TSD-SR-FT, and TSD-SR-FT+FP. Parentheses report changes relative to the preceding row.
<table><tr><td rowspan="2">Method</td><td colspan="3">Focal Length Only</td><td colspan="3">Viewpoint Only</td><td colspan="3">Joint Control</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>SD3-FT</td><td>25.42</td><td>0.764</td><td>0.141</td><td>20.82</td><td>0.590</td><td>0.216</td><td>20.69</td><td>0.566</td><td>0.230</td></tr><tr><td>TSD-SR-FT</td><td>29.64(+4.22)</td><td>0.867 (+0.103)</td><td>0.061 (-0.080)</td><td>22.30 (+1.48)</td><td>0.658 (+0.068)</td><td>0.134(-0.082)</td><td>21.85(+1.16)</td><td>0.620 (+0.054)</td><td>0.153 (-0.077)</td></tr><tr><td>TSD-SR-FT+FP</td><td>30.00 (+0.36)</td><td>0.878 (+0.011)</td><td>0.058 (-0.003)</td><td>23.13 (+0.83)</td><td>0.703 (+0.045)</td><td>0.122 (-0.012)</td><td>23.05 (+1.20)</td><td>0.679 (+0.059)</td><td>0.137 (–0.016)</td></tr></table>

![](images/f3b9c6fa7a31e353a21ed94a4b5051bc0d670bac2a905774e47e67cf57a22ebe.jpg)  
Fig. 10. Qualitative results under joint control of viewpoint, focal length, and color temperature.

SD3-FT  
TSD-SR-FT  
TSD-SR-FT+FP  
![](images/c83c5f6d1b9e58df85a2b8a229f781ef6b069d4fdf6db2dfb354da4de227d952.jpg)  
Fig. 11. Qualitative comparison across SD3-FT, TSD-SR-FT, and TSD-SR-FT+FP. The top row shows the generated images, and the bottom row shows the corresponding error maps. All error maps are visualized using the same color scale, with warmer colors indicating larger errors.

![](images/e31796d617f79e56ddfb3b40451bd37a96ac521fe623f693861311f54908587c.jpg)  
Fig. 12. PCA visualization of the projected output features, with the first three principal components mapped to RGB. The two rows correspond to two target viewpoints, and the columns show focal-length scales of 1.0×, 2.0×, and 3.0×.

Tab. 5 further shows the complementary roles of the two components. The gains between the first two rows in the columns under Focal Length Only indicate that the TSD-SR prior primarily benefits focal-length control, whereas the gains between the last two rows in the Viewpoint Only columns show that the camera-aware feature projector primarily benefits viewpoint control. These quantitative results corroborate the qualitative observations in Fig. 11.

To further illustrate the camera-aware feature projector’s ability to produce camera-aligned features, Fig. 12 visualizes its output features using PCA, with the first three principal components mapped to RGB. The two rows, corresponding to different target views, show that the projected features adapt to each target view according to its viewpoint and focal length, providing robust feature-space guidance for the generative backbone.

## 5 CONCLUSION

In this paper, we introduced single-image rephotography, the task of synthesizing new shots of the same scene from a single reference image with specified viewpoints, focal lengths, and parameterized photographic effects. We presented ReX-Shot as a unified solution to this task. To mitigate projection bias caused by imperfect single-image geometry, we combine pixel-space explicit projection with feature-space implicit projection to provide complementary target-view guidance. To recover details lost under focallength enlargement, we formulate focal-length control as a geometry-guided super-resolution problem and leverage the detail prior of a one-step generative super-resolution model. Building on the resulting 3D-aware generative backbone, we further introduce dedicated photographic-effect encoders for continuous, content-consistent effect control across viewpoints and focal lengths. Experiments demonstrate that ReX-Shot outperforms representative baselines across viewpoint, focal-length, and photographic-effect control tasks. Beyond these individual control tasks, ReX-Shot supports joint control within a unified generation process and enables near-real-time interaction through its one-step generative backbone.

Despite these strengths, the current scope of ReX-Shot is limited to focal-length enlargement, a relatively narrow range of viewpoint changes, and one photographic effect per forward pass. Future work will broaden this scope by supporting shorter focal lengths, larger viewpoint changes, and compositional control of multiple photographic effects.

## REFERENCES

[1] L. Ling, Y. Sheng, Z. Tu, W. Zhao, C. Xin, K. Wan, L. Yu, Q. Guo, Z. Yu, Y. Lu, X. Li, X. Sun, R. Ashok, A. Mukherjee, H. Kang, X. Kong, G. Hua, T. Zhang, B. Benes, and A. Bera, “DL3DV-10K: A large-scale scene dataset for deep learning-based 3d vision,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 22 160–22 169.

[2] K. Sargent, Z. Li, T. Shah, C. Herrmann, H.-X. Yu, Y. Zhang, E. R. Chan, D. Lagun, L. Fei-Fei, D. Sun, and J. Wu, “ZeroNVS: Zero-shot 360-degree view synthesis from a single image,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 9420–9429.

[3] R. Gao, A. Holynski, P. Henzler, A. Brussee, R. Martin-Brualla, P. P. Srinivasan, J. T. Barron, and B. Poole, “CAT3D: Create anything in 3d with multi-view diffusion models,” Advances in Neural Information Processing Systems, 2024.

[4] J. Y. Chung, S. Lee, H. Nam, J. Lee, and K.-M. Lee, “LucidDreamer: Domain-free generation of 3d gaussian splatting scenes,” IEEE Transactions on Visualization and Computer Graphics, vol. 31, no. 12, pp. 10 640–10 651, 2025.

[5] Z. Wang, Z. Yuan, X. Wang, Y. Li, T. Chen, M. Xia, P. Luo, and Y. Shan, “MotionCtrl: A unified and flexible motion controller for video generation,” in ACM SIGGRAPH 2024 Conference Papers. ACM, 2024.

[6] L. Jiang, Y. Mao, L. Xu, T. Lu, K. Ren, Y. Jin, X. Xu, M. Yu, J. Pang, F. Zhao, D. Lin, and B. Dai, “AnySplat: Feed-forward 3d gaussian splatting from unconstrained views,” ACM Transactions on Graphics, vol. 44, no. 6, 2025.

[7] H. Jin, H. Jiang, H. Tan, K. Zhang, S. Bi, T. Zhang, F. Luan, N. Snavely, and Z. Xu, “LVSM: A large view synthesis model with minimal 3d inductive bias,” in International Conference on Learning Representations, 2025.

[8] W. Yu, J. Xing, L. Yuan, W. Hu, X. Li, Z. Huang, X. Gao, T.-T. Wong, Y. Shan, and Y. Tian, “ViewCrafter: Taming video diffusion models for high-fidelity novel view synthesis,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[9] Y. Bai, H. Li, and Q. Huang, “Positional encoding field,” in International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id= STPO8onj9d

[10] H. He, Y. Xu, Y. Guo, G. Wetzstein, B. Dai, H. Li, and C. Yang, “CameraCtrl: Enabling camera control for video diffusion models,” in International Conference on Learning Representations, 2025. [Online]. Available: https: //openreview.net/forum?id=Z4evOUYrk7

[11] J. Wang, Z. Yue, S. Zhou, K. C. K. Chan, and C. C. Loy, “Exploiting diffusion prior for real-world image super-resolution,” International Journal of Computer Vision, vol. 132, no. 12, pp. 5929–5949, 2024.

[12] R. Wu, T. Yang, L. Sun, Z. Zhang, S. Li, and L. Zhang, “SeeSR: Towards semantics-aware real-world image super-resolution,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 25 456–25 467.

[13] X. Lin, J. He, Z. Chen, Z. Lyu, B. Dai, F. Yu, Y. Qiao, W. Ouyang, and C. Dong, “DiffBIR: Toward blind image restoration with generative diffusion prior,” in European Conference on Computer Vision, 2024, pp. 430–448.

[14] F. Yu, J. Gu, Z. Li, J. Hu, X. Kong, X. Wang, J. He, Y. Qiao, and C. Dong, “Scaling up to excellence: Practicing model scaling for photo-realistic image restoration in the wild,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 25 669–25 680.

[15] R. Wu, L. Sun, Z. Ma, and L. Zhang, “One-step effective diffusion network for real-world image super-resolution,” in Advances in Neural Information Processing Systems, 2024. [Online]. Available: https://openreview.net/forum?id=TPtXnpRvur

[16] L. Dong, Q. Fan, Y. Guo, Z. Wang, Q. Zhang, J. Chen, Y. Luo, and C. Zou, “TSD-SR: One-step diffusion with target score distillation for real-world image super-resolution,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 23 174–23 184.

[17] T. Brooks, A. Holynski, and A. A. Efros, “InstructPix2Pix: Learning to follow image editing instructions,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 18 392–18 402.

[18] K. Zhang, L. Mo, W. Chen, H. Sun, and Y. Su, “MagicBrush: A manually annotated dataset for instruction-guided image editing,” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 31 428–31 449.

[19] S. Sheynin, A. Polyak, U. Singer, Y. Kirstain, A. Zohar, O. Ashual, D. Parikh, and Y. Taigman, “Emu Edit: Precise image editing via recognition and generation tasks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 8871–8879.

[20] Q. Yu, W. Chow, Z. Yue, K. Pan, Y. Wu, X. Wan, J. Li, S. Tang, H. Zhang, and Y. Zhuang, “AnyEdit: Mastering unified high-quality image editing for any idea,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 26 125–26 135.

[21] T.-J. Fu, W. Hu, X. Du, W. Y. Wang, Y. Yang, and Z. Gan, “Guiding instruction-based image editing via multimodal large language models,” in International Conference on Learning Representations, 2024.

[22] Z. Zhang, J. Xie, Y. Lu, Z. Yang, and Y. Yang, “Enabling instructional image editing with in-context generation in large scale diffusion transformer,” in Advances in Neural Information Processing Systems, D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, Eds., vol. 38. Curran Associates, Inc., 2025, pp. 139 195–139 227. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2025/ file/cb7b55efe349d67ff3ac244aa1ae49d7-Paper-Conference.pdf

[23] I.-S. Fang, Y.-H. Han, and J.-C. Chen, “Camera settings as tokens: Modeling photography on latent diffusion models,” in SIGGRAPH Asia 2024 Conference Papers, 2024.

[24] Y. Yuan, X. Wang, Y. Sheng, P. Chennuri, X. Zhang, and S. Chan, “Generative photography: Scene-consistent camera control for realistic text-to-image synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 7920–7930.

[25] X. Qin, Z. Wang, F. Li, H. Chen, R. Pei, W. Li, and X. Cao, “CamEdit: Continuous camera parameter control for photorealistic image editing,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=2ncMTlR9nC

[26] Q. Yang, Y. Yang, Y. Zeng, X. Hu, B. Li, H. Yue, J. Yang, and P.-T. Jiang, “CameraMaster: Unified camera semantic-parameter control for photography retouching,” arXiv preprint arXiv:2511.21024, 2025.

[27] W. Yin, C. Zhang, H. Chen, Z. Cai, G. Yu, K. Wang, X. Chen, and C. Shen, “Metric3D: Towards zero-shot metric 3d prediction from a single image,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 9043–9053.

[28] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “DUSt3R: Geometric 3d vision made easy,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[29] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “VGGT: Visual geometry grounded transformer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 5294–5306.

[30] Y. Wang, J. Zhou, H. Zhu, W. Chang, Y. Zhou, Z. Li, J. Chen, J. Pang, C. Shen, and T. He, “π<sup>3</sup>: Permutation-equivariant visual geometry learning,” in International Conference on Learning Representations, 2026. [Online]. Available: https: //openreview.net/forum?id=DTQIjngDta

[31] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P.-Y. Huang, S.-W. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski,´ “DINOv2: Learning robust visual features without supervision,” Transactions on Machine Learning Research, 2024. [Online]. Available: https://openreview.net/forum?id=a68SUt6zFt

[32] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” in European conference on computer vision. Springer, 2020, pp. 405–421.

[33] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3d¨ gaussian splatting for real-time radiance field rendering,” ACM Transactions on Graphics, vol. 42, no. 4, pp. 1–14, 2023.

[34] D. Charatan, S. L. Li, A. Tagliasacchi, and V. Sitzmann, “pixelSplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 457–19 467.

[35] Y. Chen, H. Xu, C. Zheng, B. Zhuang, M. Pollefeys, A. Geiger, T.-J. Cham, and J. Cai, “MVSplat: Efficient 3d gaussian splatting from sparse multi-view images,” in European Conference on Computer Vision. Springer, 2024, pp. 370–386.

[36] H. Jiang, H. Tan, P. Wang, H. Jin, Y. Zhao, S. Bi, K. Zhang, F. Luan, K. Sunkavalli, Q. Huang, and G. Pavlakos, “RayZer: A self-supervised large view synthesis model,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 4918–4929.

[37] J. Zhou, H. Gao, V. Voleti, A. Vasishta, C.-H. Yao, M. Boss, P. Torr, C. Rupprecht, and V. Jampani, “Stable virtual camera: Generative view synthesis with diffusion models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

[38] J. Seo, K. Fukuda, T. Shibuya, T. Narihira, N. Murata, S. Hu, C.-H. Lai, S. Kim, and Y. Mitsufuji, “GenWarp: Single image to novel views with semantic-preserving generative warping,” in Advances in Neural Information Processing Systems, 2024.

[39] N. Muller, K. Schwarz, B. R¨ ossle, L. Porzi, S. R. Bul¨ o, M. Nießner,\` and P. Kontschieder, “MultiDiff: Consistent novel view synthesis from a single image,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 10 258– 10 268.

[40] E. R. Chan, K. Nagano, M. A. Chan, A. W. Bergman, J. J. Park, A. Levy, M. Aittala, S. De Mello, T. Karras, and G. Wetzstein, “GeNVS: Generative novel view synthesis with 3D-aware diffusion models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4217–4229.

[41] R. Li, B. Yi, J. Liu, H. Gao, Y. Ma, and A. Kanazawa, “Cameras as relative positional encoding,” in Advances in Neural Information Processing Systems, D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, Eds., vol. 38. Curran Associates, Inc., 2025, pp. 15 984–16 009. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2025/ file/17a7075094632c88cccdd86270ad715b-Paper-Conference.pdf

[42] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10 684–10 695.

[43] Y. Wang, W. Yang, X. Chen, Y. Wang, L. Guo, L.-P. Chau, Z. Liu, Y. Qiao, A. C. Kot, and B. Wen, “SinSR: Diffusion-based image super-resolution in a single step,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 25 796–25 805.

[44] Y. Tai, R. Xie, C. Zhao, K. Zhang, Z. Zhang, J. Zhou, and J. Yang, “AddSR: Accelerating diffusion-based blind super-resolution with adversarial diffusion distillation,” Pattern Recognition, vol. 175, p. 113012, 2026.

[45] R. Wu, B. Mildenhall, P. Henzler, K. Park, R. Gao, D. Watson, P. P. Srinivasan, D. Verbin, J. T. Barron, B. Poole, and A. Holynski, “ReconFusion: 3d reconstruction with diffusion priors,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 551–21 561.

[46] F. Liu, W. Sun, H. Wang, Y. Wang, H. Sun, J. Ye, J. Zhang, and Y. Duan, “ReconX: Reconstruct any scene from sparse views with video diffusion model,” arXiv preprint arXiv:2408.16767, 2024.

[47] J. Z. Wu, Y. Zhang, H. Turki, X. Ren, J. Gao, M. Z. Shou, S. Fidler, Z. Gojcic, and H. Ling, “DIFIX3D+: Improving 3d reconstructions with single-step diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 26 024–26 035.

[48] X. Yin, Q. Zhang, J. Chang, Y. Feng, Q. Fan, X. Yang, C.-M. Pun, H. Zhang, and X. Cun, “GSFixer: Improving 3d gaussian splatting with reference-guided video diffusion priors,” in Proceedings of the 43rd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 306, 2026. [Online]. Available: https://openreview.net/pdf/ 065e379205a0ffce09e3383c80c82c313cc5e5aa.pdf

[49] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 586–595.

[50] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3836–3847.

[51] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Mueller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel, D. Podell, T. Dockhorn, Z. English, and R. Rombach, “Scaling rectified flow transformers for high-resolution image synthesis,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 235. PMLR, 2024, pp. 12 606– 12 633.

[52] M. Tancik, P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. Barron, and R. Ng, “Fourier features let networks learn high frequency functions in low dimensional domains,” Advances in Neural Information Processing Systems, vol. 33, pp. 7537–7547, 2020.

[53] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, “Mip-NeRF 360: Unbounded anti-aliased neural radiance fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5470–5479.

[54] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, “Tanks and temples: Benchmarking large-scale scene reconstruction,” ACM Transactions on Graphics, vol. 36, no. 4, 2017.

[55] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in International Conference on Learning Representations, 2022.

[56] Q. Huynh-Thu and M. Ghanbari, “Scope of validity of PSNR in image/video quality assessment,” Electronics Letters, vol. 44, no. 13, pp. 800–801, 2008.

[57] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,”

IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.

[58] G. Sharma, W. Wu, and E. N. Dalal, “The CIEDE2000 color-difference formula: Implementation notes, supplementary test data, and mathematical observations,” Color Research & Application, vol. 30, no. 1, pp. 21–30, 2005.

[59] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 4104–4113.

Ruiqi Zhang received his B.S. degree from the School of Electronic and Optical Engineering, Nanjing University of Science and Technology, Nanjing, China, in 2022. He is currently pursuing the Ph.D. degree with the School of Electronic Science and Engineering, Nanjing University, Nanjing, China. His current research interests include computational photography and 3D reconstruction and perception.

![](images/4327b07ae126b8bff9fbd27bdaca5aeb34426ac813c75f46255dc18f04667589.jpg)

Hao Zhu is an Assistant Professor in the School of Electronic Science and Engineering, Nanjing University. He received the B.S. and Ph.D. degrees from Northwestern Polytechnical University in 2014 and 2020, respectively. He was a visiting scholar at the Australian National University. His research interests include computational photography and optimization for inverse problems.

![](images/afc261147070b617d6ad572dbf2ff4479c6e7efa299a818ff7828cfc51490c92.jpg)

![](images/8a1b7993b4b0a4c3e60e14490df3cfc991e427cedc1e4a1a1f8f77408c99edf9.jpg)

Wenhao Zhang received a B.S. degree from Nankai University, Tianjin, China. He is currently pursuing the M.S. degree with the School of Electronic Science and Engineering, Nanjing University, Nanjing, China. His research interests include 3D reconstruction and generation.

![](images/373870f7ad8a857f493f4cab267d19ef96744a935678e6776b1dd187654e335b.jpg)

![](images/f2a54e145ad4954f1976e18673286a42d5541cf111ee5a981579f13ec6f7dab2.jpg)

Junqi Shi received his B.S. and M.E. degrees in the School of Electronic Science and Engineering from Nanjing University, Nanjing, China in 2022 and 2025 respectively, where he is currently pursuing the Ph.D. degree. His current research interests include deep learning-based image/video coding, model quantization and signal representation.

Qi Zhang is currently a lead researcher with vivo Mobile Communication Co., Ltd. in Xi’an, China. Before that, he was a researcher with Tencent AI Lab. He received his Ph.D. from the School of Computer Science at Northwestern Polytechnical University in 2021. He received CCF Doctorial Dissertation Award Nominee in 2021. His research interests include 3D vision, neural rendering, Gaussian Splatting, and AIGC.

![](images/2fd48bdaf482d71577673e4d34f8017d6d672cd9733c1838fcf570b3c3425772.jpg)

Ming Lu is an Associate Researcher with the School of Electronic Science and Engineering, Nanjing University, Nanjing, China. He received his B.S. and Ph.D. degrees in the School of Electronic Science and Engineering from Nanjing University, Nanjing, China, in 2016 and 2023 respectively. His research focuses on deep learning-based image/video coding. He is a corecipient of the 2018 ACM SIGCOMM Student Research Competition Finalist, the 2020 IEEE MMSP Image Compression Grand Challenge

Best Performing Solution, and the 2023 IEEE WACV Best Algorithms Paper Award.

![](images/a5143fe6289addc44ce2303b82b285c0ee78948905884a069199fe5cb9b89f1c.jpg)

Xun Cao received the B.S. degree from Nanjing University, Nanjing, China, in 2006, and the Ph.D. degree from the Department of Automation, Tsinghua University, Beijing, China, in 2012. He held visiting positions with Philips Research, Aachen, Germany, in 2008 and Microsoft Research Asia, Beijing, from 2009 to 2010. He was a Visiting Scholar with the University of Texas at Austin, Austin, TX, USA, from 2010 to 2011. He is a Professor at the School of Electronic Science and Engineering, Nanjing

University. His current research interests include computational photography and image-based modeling and rendering.

![](images/c7a8a2dc0eba83d0084f4db02470f6ba56122e3f118241d0529b0e1668f56cab.jpg)

Zhan Ma (SM’19) is a Professor in Electronic Science and Engineering School, Nanjing University, Nanjing, Jiangsu, 210093, China. He received the B.S. and M.S. degrees from the Huazhong University of Science and Technology, Wuhan, China, in 2004 and 2006, respectively, and the Ph.D. degree from the New York University, New York, in 2011. From 2011 to 2014, he has been with Samsung Research America, Dallas, TX, and Futurewei Technologies, Inc., Santa Clara, CA, respectively. His research focuses on learning-based video communication and computational imaging. He is a co-recipient of the 2019 IEEE Broadcast Technology Society Best Paper Award, the 2020 IEEE MMSP Image Compression Grand Challenge Best Performing Solution, the 2023 IEEE WACV Best Algorithm Paper Award, and the 2023 IEEE Circuits and Systems Society Outstanding Young Author Award.