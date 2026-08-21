# AvatarDynamizer: From Static to Dynamic Human Avatars via Generative Dynamic Textures

Guoxing Sun<sup>1</sup>, Heming Zhu<sup>1</sup>, Linjie Lyu<sup>1</sup>, Pascal Fua<sup>3</sup>, Christian Theobalt<sup>1,2</sup>, Marc Habermann<sup>1,2</sup> <sup>1</sup>Max Planck Institute for Informatics, Saarland Informatics Campus <sup>2</sup>VIA Research Center <sup>3</sup>EPFL {gsun, hezhu, llyu, theobalt, mhaberma}@mpi-inf.mpg.de pascal.fua@epfl.ch https://vcai.mpi-inf.mpg.de/projects/AvatarDynamizer/

![](images/5e7d6526acc5da232c3ae99c38176785aaac9a3dd0835640c1464b063169b21b.jpg)  
Figure 1. Given a 3D static avatar from any multimodal input, e.g., a monocular image with LHM [59], we introduce a novel method, AvatarDynamizer, which converts static avatars into controllable 4D human avatars with pose-dependent surface dynamics (see the shirt).

## Abstract

Forfull-body avatars, modeling surface dynamics is crucial for overcoming the uncanny valley and achieving perceptual realism. Person-agnostic methods recover static 3D avatars from monocular images, videos, or text prompts, but their skeleton-driven animations lack realistic surface dynamics such as clothing wrinkles. In contrast, personspecific methods achieve high-quality rendering and realistic dynamics, but require expensive multi-view captures for each individual. Recent generalizable dynamic avatar methods struggle to embed surface dynamics, leading to either limited multi-view consistency or dynamic expressiveness. To this end, we propose AvatarDynamizer, a generative method that transforms an off-the-shelfstatic 3D avatar into a controllable, realistic, and multi-view-consistent 4D avatar. We introduce a novel texture-space surfacedynamics embedding and formulate avatar dynamics modeling as conditional texture generation. Our encoder– decoder representation embeds pose-dependent dynamics into dynamic texture maps, enabling compatibility with pretrained video diffusion models while decoding them into 3D Gaussians for multi-view consistent rendering. Since existing datasets are limited in scale, sequence length, or mo-

tion diversity, we collect a large-scale multi-view dataset with long sequences covering diverse skeletal motions and surface dynamics. Experiments show that our method effectively animates static avatars withfaithful surface dynamics and outperforms competing generalizable methods in visual fidelity, especially under limited dynamic training data.

## 1. Introduction

Creating faithful and controllable digital twins of real humans has long been of interest in Vision and Graphics, with broad applications in telecommunications, remote coaching, and virtual agents. Recently, substantial effort has gone into collecting large-scale datasets [12, 44] for generalizable human reconstruction from sparse views [59, 101], as well as for the generation of 3D avatars from multimodal inputs such as text [14, 48] or from unconditional noise [11, 93]. All these methods generate a static avatar, which can be animated into arbitrary poses using an underlying rigged skeleton. However, the resulting animations primarily rely on standard skinning-like deformations [43] that are approximately piecewise rigid. Realistic surface dynamics, including cloth wrinkles and induced shading effects, are still missing. This leads to an uncanny valley effect, making the avatars appear unnatural to human observers. In this work, we ask: Can we introduce faithful dynamics to off-the-shelf static 3D avatars to bridge this gap?

Traditionally, modeling avatar surface dynamics in Computer Graphics requires manual effort throughout the animation pipeline, including rigging, skinning, animation, cloth simulation, and rendering. In the machine learning domain, some works learn a mapping from body motion to 4D geometry and appearance represented as radiance/SDF fields [97], Gaussian splats [35, 54], light fields [41], or conventional 2D textures [47]. To date, these methods have achieved unprecedented rendering quality and fine-grained control. However, each character must be captured with a multi-camera system, after which a person-specific model must be trained for several days. Moreover, they remain person-specific and do not generalize across identities.

Recently, research has shifted toward generalizable ap proaches [8, 77, 87, 99], which leverage 2D foundation models [62] to generate dynamic images from an identity image and 2D pose controls. However, such 2D generation cannot produce multi-view-consistent, free-view renderings. Recent works further explore cross-identity generalization together with 3D surface dynamics modeling [21, 51, 65]. We observe that their key difference lies in where surface dynamics are embedded. Image-space methods [51, 65] embed dynamic surface details into video priors and 3D pose controls, enabling them to leverage pretrained foundation models but sacrificing 3D consistency. In contrast, texel- or Gaussian-regression methods [21] embed surface dynamics into a deterministic feed-forward regression framework, directly mapping identity and motion conditions to avatar representations. While this formulation naturally preserves 3D consistency, it prevents them from directly exploiting pre-trained 2D video generative priors. Moreover, deterministic regression struggles with modeling fine-grained surface dynamics under limited data, as motion-dependent wrinkles are high-frequency and partially stochastic: the same identity and pose may correspond to different plausible cloth states depending on unobserved physical conditions. Thus, regression-based methods can easily produce over-smoothed or nearly static appearance dynamics, even when motion conditions are provided.

These observations suggest that the key bottleneck lies in how surface dynamics are represented and embedded. A desired representation should satisfy two properties: it should be temporally and spatially consistent, while remaining compatible with powerful generative models such as image and video diffusion models. To this end, inspired by latent diffusion [62], we introduce a novel texture-space surface dynamics embedding within an encoder–decoder human representation. We explicitly embed pose-dependent surface dynamics into colored dynamic texture maps, rather than modeling them via deterministic regression or direct image-space generation. These dynamic textures serve as a generative intermediate representation: they are spatially and temporally coherent in texel space, compatible with pre-trained video diffusion models, and can be decoded into 3D Gaussians for multi-view consistent rendering.

More precisely, given a static 3D human avatar from diverse multi-modal sources, such as a monocular image, text prompt, 3D scan, or multi-view images, we aim to build a generalizable avatar dynamics model, dubbed AvatarDynamizer, which transforms the static avatar into a controllable 4D avatar with faithful surface and appearance dynamics (see also Fig. 1). At its core, our method learns to generate motion-conditioned dynamic texture maps that explicitly embed pose-dependent surface dynamics.

Our method consists of three steps: 1) multi-view data encoding and Generalized Gaussian Decoder learning, 2) dynamics generator learning, and 3) inference with a novel static avatar. We first encode multi-view videos into 2D texture videos through multi-stage optimization. Here, we non-rigidly register SMPL-X [55] to each frame and then optimize colored 2D dynamic texture maps. Next, we train our cross-identity Generalized Gaussian Decoder to take these texture maps and pose-related normal maps as input and to decode them into 3D Gaussian splats, which are rendered and supervised using multi-view images. As this step does not require temporal data, any type of multi-view human data can be used to ensure identity generalization.

To learn dynamics for the second step, multi-view video data featuring diverse motions is required, but publicly available data is scarce. Thus, we collected our own dynamic human dataset, DynaHuman, comprising 58 actors captured with 100 4K cameras, where each sequence has a total length of 27,000 frames. While significantly more diverse than any other publicly available dataset, it is still insufficient for full generalization. Therefore, we introduce our Dynamic Texture Generator, which formulates dynamics learning as a conditional image-to-image translation task in texel space, i.e., given a static color texture and posed normal maps encoding motion, generate the dynamic texture map with pose-dependent wrinkles. Importantly, it can leverage pre-trained foundational video models [3] to compensate for data scarcity. Finally, at inference time, our method recovers a corresponding static identity texture from a static 3D avatar. Our contributions are:

• A novel texture-space surface-dynamics embedding that represents 4D humans as compact 2D dynamic texture maps, together with a Generalized Gaussian Decoder for multi-view consistent rendering.

• A Dynamic Texture Generator that leverages pre-trained video diffusion priors to generate highly detailed dynamic textures conditioned on identity and motion.

• DynaHuman, a large-scale human dataset with diverse identities, camera views, and substantial motion diversity.

## 2. Related Work

Here, we review prior works on human representations, human priors, and multi-view human datasets for modeling diverse identities, body motions, and clothing dynamics. Head avatar methods [5, 17, 18, 25, 45, 49, 58, 61, 72, 73, 84, 95, 96], which focus on facial expressions, skin materials, and hair modeling, are beyond the scope of this work.

## 2.1. Human Representations

Developing effective 3D human representations is a fundamental problem for human modeling, reconstruction, and rendering. Early methods commonly rely on meshbased templates, often combined with motion capture techniques [36, 67], to improve mocap accuracy [26, 34] or capture human performances [19, 22, 66]. These representations are efficient and compatible with graphics rendering pipelines, but they struggle to capture fine-grained geometry, appearance details, and complex clothing deformation. With the development of neural rendering, diverse representations have been explored for human avatars, in cluding neural fields [64, 68, 78], point clouds [52, 83], NeRF [40, 56], NeuS [69, 91], 3D Gaussians [10, 29, 33, 47, 70, 98] and transformer-based representations [90]. These methods achieve high-quality rendering and controllability through rigged skeletons. However, such 3D representations are usually designed for person-specific reconstruction or animation, making it non-trivial to integrate them with pre-trained 2D generative pipelines for generalizable, pose dependent surface modeling. Recent works further investigate texture-based representations for human rendering and reconstruction [1, 66, 70]. GIGA [101] unprojects sparseview images into a texture map and predicts Gaussians in texel space, but it does not adopt a generative formulation. MoGA [15] learns an implicit texture latent for human inversion; however, its pose-independent design restricts it to static human generation and prevents effective modeling of dynamic surface changes. In contrast, we reinterpret the texture map as an explicit latent representation of a 3D human from an encoding–decoding perspective. This explicit latent is pose-aware, changes smoothly under pose control, and serves as a bridge between pre-trained 2D video generative priors and multi-view consistent 3D avatar rendering.

## 2.2. Human Priors

Human priors are widely used to regularize optimization, improve reconstruction, and enable generalizable generation. In motion capture, prior works use hand-crafted constraints or learned statistical priors [4, 13, 55, 67] for robust body pose estimation. For image-based reconstruction, regression-based methods learn priors to predict neural fields [2, 64, 85], multi-view images [30, 90], normal maps [46, 86], or 3D Gaussians [59, 60, 100]. While generalizing across identities and clothing, they mainly target static or frame-wise reconstruction and struggle with temporally coherent, pose-dependent surface dynamics. Diffusion-based monocular approaches such as Avatar-PopUp [39] exploit generative priors for single-view avatar reconstruction, but controllable 4D modeling with temporal wrinkles and multi-view consistency remains challenging. Another line of work adopts image or video diffusion models to represent humans in 2D image space [9, 30, 46, 81, 87, 88, 99]. These methods benefit from strong identity generalization enabled by large-scale generative priors, but they often lack multi-view consistency. Toward consistent 4D avatar generation, HDA [6] learns a generative model in the hyperspace of an animatable avatar representation [54], though its quality is limited by the fittingbased preprocessing of network weights. Most closely related to our work, Vid2AvatarPro [21] and GAS [51] aim to model generalizable human surface and appearance dynamics. Vid2AvatarPro [21] conditions on both identity and pose for dynamic avatar reconstruction, but jointly learning identity and pose control through a regression model is challenging and often results in static appearances even when pose input is provided. Moreover, directly decoding 3D Gaussians makes it difficult to leverage pre-trained foundation video models. GAS [51] learns video generation priors with a switcher to produce dynamic wrinkles for a static 3D avatar in image space, but this comes at the cost of multi-view consistency. In this work, we embed surface dynamics into texture space: a video diffusion prior models motion-dependent texture dynamics, while a reconstruction prior decodes the generated textures into 3D Gaussians for multi-view consistent rendering.

## 2.3. Human Datasets

High-quality multi-view human datasets are essential for learning generalizable and dynamic human avatars. Existing datasets mainly fall into two categories, each with distinct limitations for modeling both identity diversity and pose-dependent surface dynamics. On the one hand, dynamic datasets of individual subjects, such as DDC [23] and DUT [70], capture detailed body motions and surface deformations. These datasets are valuable for learning precise dynamics, but they contain very few identities, which limits cross-subject generalization. On the other hand, largescale identity datasets such as MVHumanNet++ [44] and DNA-Rendering [12] provide diverse human appearances, clothing styles, and body shapes. However, they generally contain static poses or very limited motion ranges, making them insufficient for learning complex dynamic surface priors such as temporal cloth wrinkling. This dichotomy motivates us to leverage pre-trained 2D video diffusion priors to bridge the gap between large-scale identity diversity and realistic human dynamics, and to collect a large-scale dataset with diverse identities and body motions.

![](images/1ff79cf2330d1a6f42e1a36a36bc1565fdda0a69cbbca4c90528df333f746357.jpg)  
Figure 2. Overview. Given a static human avatar reconstructed from multimodal inputs, e.g., multi-view images, we first perform identity fitting to obtain its coarse shape and identity texture. Our Dynamic Texture Generator predicts dynamic texture maps conditioned on identity and novel pose. Then, our Generalized Gaussian Decoder recovers Gaussian splats for faithful and view-consistent renderings.

## 3. Method

We introduce AvatarDynamizer, which transforms a static 3D human avatar from diverse input modalities, such as monocular or multi-view images, into a controllable 4D avatar with pose-dependent deformations and appearance changes, $\mathrm { e . g . }$ ., shadows and wrinkles (Fig. 2). Importantly, our renderings are inherently multi-view consistent, as the final dynamic avatar is represented using Gaussian splats. To improve 4D human encoding, we introduce a novel texture-space human representation that encodes multi-view human videos into compact dynamic textures through multi-stage optimization. We train our Generalized Gaussian Decoder to reconstruct texel-aligned 3D Gaussian splats from dynamic color and pose textures, supervised by multi-view images (Sec. 3.3). With multi-view data embedded, our Dynamic Texture Generator formulates dynamics learning as conditional video generation in texture space, using pose-related normal maps and identity textures as inputs (Sec. 3.4). This formulation enables initialization with a pre-trained video foundation model [3], supporting dynamics and identity generalization under limited data while capturing the stochastic nature of surface dynamics. At inference, we fit the coarse shape and identity texture of the static avatar, generate motion-dependent textures from the identity texture and posed normal map, and decode them into 3D Gaussians for novel-view rendering (Sec. 3.5). Before diving into the method, we review the background (Sec. 3.1) and describe the key idea (Sec. 3.2).

## 3.1. Preliminaries

Latent Video Diffusion. LVD [3] learns to generate videos by modeling the reverse diffusion process in a compressed latent space. A pre-trained 3D variational autoencoder $( \mathrm { V A E } ) \boldsymbol { \mathcal { E } } ( \boldsymbol { \mathcal { V } } ) = \mathbf { z } _ { 0 }$ first embeds a T-frame video sequence $\boldsymbol { \mathcal { V } } \in \mathbb { R } ^ { T \times H \times W \times 3 }$ into a latent $\mathbf { z } _ { 0 } .$ . With a predefined noise schedule, we compute a noisy latent $\mathbf { z } _ { t }$ at timestep t as $\mathbf { z } _ { t } = \alpha _ { t } \mathbf { z } _ { 0 } + \sigma _ { t } \boldsymbol { \epsilon } .$ , where $\epsilon \sim \mathcal { N } ( 0 , \bf { I } )$ denotes the Gaussian noise added to the latent representation. At each timestep t, a denoising network $\epsilon _ { \theta } ( \cdot )$ predicts the noise, and the training objective is formulated as:

$$
\mathcal { L } _ { L V D } = \mathbb { E } _ { \mathbf { z } _ { 0 } , \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } ) , t } \left[ \lVert \epsilon - \epsilon _ { \theta } ( \mathbf { z } _ { t } , t , \mathbf { C } ) \rVert _ { 2 } ^ { 2 } \right]\tag{1}
$$

where C represents the conditioning signals, which could be text prompts, depth maps, or normal maps, and θ refers to the trainable network weights.

Texel Gaussian Avatar. We use the parametric model SMPL-X [55] as the base template $\bar { \mathbf { V } } \in \mathbf { \mathbb { R } } ^ { N _ { \mathrm { V } } \times 3 }$ and bind 3D Gaussians to its texture [54]. The coarse shape is defined by the shape parameter $\beta _ { : }$ , cloth displacements $\mathbf { d } _ { \mathrm { c l o t h } }$ and hair displacements $\mathbf { d } _ { \mathrm { h a i r } } .$ . Given a skeletal motion M, i.e., joint rotations, as input, we animate it through LBS [43]:

$$
{ \bf V } = T _ { \mathrm { L B S } } ( \mathcal { M } , T _ { \mathrm { D } } ( \bar { \bf V } , \beta , { \bf d } _ { \mathrm { h a i r } } , { \bf d } _ { \mathrm { c l o t h } } ) ) ,\tag{2}
$$

where we deform the base mesh in the canonical pose using $T _ { \mathrm { D } } ( \cdot )$ For each valid texel i of the texture map, we parameterize the corresponding Gaussian as $\begin{array} { r l } { \mathcal { T } _ { \pmb { \mathscr { G } } _ { i } } } & { { } = } \end{array}$ $\{ \mu _ { i } , \mathbf { h } _ { i } , \mathbf { s } _ { i } , \mathbf { r } _ { i } , \alpha _ { i } \}$ , including 3D positions in world space, $\pmb { \mu } _ { i } = \mathbf { T } _ { \mathrm { L B S } } ( \mathcal { M } , \mathbf { T } _ { \mathrm { D } } ( \cdot ) + \pmb { \delta } _ { i } )$ , spherical harmonics, scaling, rotations, and opacity values, respectively. $\delta _ { i }$ denotes a motion-dependent per-Gaussian displacement that is later predicted by our Generalized Gaussian Decoder alongside the other Gaussian parameters. Given a virtual camera view matrix W, Gaussians are rendered into a 2D image with a tile-based 3DGS rasterizer [38]: $\mathcal { R } ( \pmb { \mathscr { G } } , \mathbf { W } ) = \hat { \mathcal { T } }$

## 3.2. Embedding 4D Human as Dynamic Textures

Motivation. For faithful and controllable 4D human avatars, how surface dynamics are represented and embedded is crucial. Existing representations either lack compatibility with powerful diffusion priors [15, 21] or sacrifice multi-view consistency [51, 88]. An ideal representation should support generative priors while preserving multiview consistency. Inspired by latent diffusion [62], we embed multi-view images into compact 2D textures, which are decoded by our Generalized Gaussian Decoder into texelaligned 3D Gaussians. Unlike texel-space methods such as GIGA [101], where textures serve as sparse-view fusion features for feed-forward reconstruction, we use dynamic textures as an explicit, time-varying latent space for generative surface dynamics. They are optimized from multi-view videos, generated under motion control, and decoded into 3D Gaussians for consistent rendering. This shifts texture maps from reconstruction features to a generative dynamics embedding. To turn a static avatar, represented in our compact texture space, into a dynamic one, our Dynamic Texture Generator maps static color textures to motion-dependent dynamic textures conditioned on skeletal motion. This for mulates dynamics generation as conditional video generation in texel space, enabling to leverage video diffusion priors [3, 62] to counter data scarcity. Unlike image-space methods [51, 88], our generated textures are decoded into 3D Gaussians, naturally preserving multi-view consistency. Dynamic Texture Embedding. Embedding multi-view videos into dynamic textures involves multi-stage optimization. For clarity, we omit the temporal smoothing used during batch optimization. Given multi-view images $\{ \mathcal { T } ^ { k } \} _ { k \in [ 1 , N ] }$ captured from N camera views with paired foreground segmentations $\{ { \cal S } ^ { k } \} _ { k \in [ 1 , N ] }$ of each frame, we first reconstruct point cloud geometry p [80] and run 2D marker detection [7] and triangulation to recover 3D markers m. With p and m as reference, we optimize the pose parameter M and shape parameter $\beta$ of SMPL-X [55], yield ing coarse geometric tracking. To capture fine textures, we further optimize time-varying cloth displacements $\ddot { \mathbf { d } } _ { \mathrm { c l o t h } }$ in the canonical pose on top of the coarsely tracked $\mathrm { S M P L - }$ X template, with p as the reference geometry. Next, we recover the RGB texture map $\tau$ through differentiable rasterization DR [42] from multi-view images:

$$
\operatorname* { m i n } _ { T } \sum _ { k = 1 } ^ { N } \left\| \hat { \mathcal { I } } _ { k } - \mathcal { D } \mathcal { R } ( \mathbf { V } ( \mathcal { M } , \beta , \mathbf { d } _ { \mathrm { h a i r } } , \tilde { \mathbf { d } } _ { \mathrm { c l o t h } } ) , T , \mathbf { W } _ { \mathbf { k } } ) \right\| _ { 2 } ^ { 2 }\tag{3}
$$

Thus, human information is encoded in textures $\tau$ for all subjects and frames. See the supplementary for details.

## 3.3. Generalized Gaussian Decoder

With the multi-view data compressed into dynamic textures, the decoder takes them as input to reconstruct 3D Gaussian primitives and render images accordingly. Specifically, the Generalized Gaussian Decoder $\Phi _ { \mathrm { G G D } }$ takes the texture map $\tau$ and the root-normalized normal map $\mathcal { T } _ { \bar { \mathrm { N } } }$ as inputs and predicts the texel-aligned 3D Gaussian parameters

$$
\mathcal { T } _ { \mathfrak { g } } = \Phi _ { \mathrm { G G D } } ( \mathcal { T } , \mathcal { T } _ { \bar { \mathrm { N } } } ) \ .\tag{4}
$$

We supervise $\Phi _ { \mathrm { G G D } }$ with multi-view image losses, including L1, SSIM [82], and IDMRF [79], and geometric regularization on the displacements of the Gaussians:

$$
\mathcal { L } _ { \mathrm { G G D } } = \mathcal { L } _ { \mathrm { L 1 } } + \lambda _ { \mathrm { S S I M } } \mathcal { L } _ { \mathrm { S S I M } } + \lambda _ { \mathrm { I D M } } \mathcal { L } _ { \mathrm { I D M } } + \lambda _ { \mathrm { R e g } } \mathcal { L } _ { \mathrm { R e g } } .\tag{5}
$$

$\Phi _ { \mathrm { G G D } }$ therefore lifts our texture-space surface-dynamics embedding to 3D Gaussians, enabling video diffusion priors to model dynamics in texture space while preserving multiview consistency after decoding.

## 3.4. Dynamic Texture Generator

Our texture-space embedding compresses 4D human surface dynamics as dynamic RGB texture videos, providing a compact generative space that can directly leverage pre-trained video diffusion priors. Next, we introduce the Dynamic Texture Generator $\Phi _ { \mathrm { D T G } }$ to model motiondependent surface dynamics directly in texture space. At this stage, 3D Gaussians are not involved; instead, the generated dynamic textures are later decoded by the Generalized Gaussian Decoder for multi-view consistent rendering. To faithfully model the wrinkles on the dynamic textures, unlike Vid2Avatar-Pro [21], which models the prior through regression, we formulate our generator $\Phi _ { \mathrm { D T G } }$ as a skeletal motion-conditioned and identity-conditioned video generation task in texel space. Specifically, we adopt a stable video diffusion model [3] as the base model and condition character identity on the RGB texture map $\mathcal { T } _ { \mathrm { i d } }$ capturing the character appearance in ‘A pose’, and character motion on a stack of normal maps $\mathcal { T } _ { \bar { \mathrm { N } } }$ rendered from the SMPL-X template mesh with the root translation and rotation factored out. Under this conditioning scheme, we formulate the denoising prior model $\Phi _ { \mathrm { D T G } }$ as

$$
\epsilon _ { \theta } ( \mathbf { z } _ { t } , t , \mathbf { C } ) = \Phi _ { \mathrm { D T G } } ( \mathbf { z } _ { t } , t , \mathcal { E } ( \mathcal { T } _ { \mathrm { i d } } ) , \mathcal { E } ( \{ \mathcal { T } _ { \tilde { \mathrm { N } } } ^ { j } \} _ { j \in [ 1 , T ] } ) ) ,\tag{6}
$$

which is trained with the denoising loss in Eq. 1 using dynamic textures from multiple subjects. Once we obtain generated dynamic textures $\{ \hat { T } ^ { j } \} _ { j \in [ 1 , T ] }$ that describe the character under the given motions, we feed them into the $\Phi _ { \mathrm { G G D } }$ to reconstruct the 4D dynamic Gaussian Splats.

## 3.5. Novel Avatar Dynamization

Existing methods reconstruct static avatars from text [48], monocular or multi-view images [54, 59, 100], videos [20], or 3D scans [94]. We dynamize them in the following steps. Identity Fitting. Given a static avatar, we render multiview images of it and follow Sec. 3.2 to optimize rest body pose $\mathcal { M } _ { : }$ , shape parameter $\beta ,$ cloth displacements $\mathbf { d } _ { \mathrm { c l o t h } }$ hair displacements $\mathbf { d } _ { \mathrm { { h a i r } } }$ , and identity texture ${ \mathcal { T } } _ { \mathrm { i d } } . \qquad \mathbf { D } \mathbf { y } \cdot$ namic Texture Generation. We feed the novel identity texture $\mathcal { T } _ { \mathrm { i d } }$ and control poses $\{ \mathcal { T } _ { \bar { \mathrm { N } } } ^ { t } \} _ { t \in [ 1 , M ] }$ to the Φ<sub>DTG</sub> to generate dynamic textures $\{ \hat { T } ^ { t } \} _ { t \in [ 1 , M ] }$ . For arbitrary-length inference, we use sliding-window blending of overlapping VAE latents for temporal consistency.

![](images/ef529f6b466a3d2f5831bedaf6d400654f89ae0454a73deae8d667e6068618e7.jpg)  
RDT+GGD  
GT  
RDT+GGD  
GT

Figure 3. Qualitative Comparison on DynaHuman. Given a static avatar reconstructed from a monocular image using LHM [59] or from multi-view images (MV), we compare our method with state-of-the-art generalizable surface dynamics methods. Our AvatarDynamize produces faithful wrinkles while preserving both 3D geometry and identity consistency.

Avatar Rendering. Our Φ<sub>GGD</sub> decodes {T <sup>t</sup><sub>¯</sub> }<sup>M</sup><sub>t=1</sub> and {T<sup>ˆt</sup>}<sup>M</sup> into 3D Gaussians, which are rendered with the given camera parameters using the Gaussian rasterizer [38].

## 4. Results

Datasets. We collected a large-scale multi-view human dataset named DynaHuman, which consists of 58 subjects. Each subject is captured for 27,000 frames by 100 cameras at 4K resolution. In terms of motion diversity, our dataset substantially surpasses existing ones. We isolate two subjects for evaluation. For the Φ , we use 56 subjects from

DynaHuman and 478 subjects from MVHPP [44] for better identity and clothing generalization. We use registered texture maps from DynaHuman to train the Φ<sub>DTG</sub> because of the temporal smoothness and diversity of identity and motion. In addition to DynaHuman, we also evaluate our method on public datasets, DeepCap [22], DDC [23], and MetaCap [69] as reported in Tab. 1. For each dataset, we use two subjects and compute the average metric scores.

Implementation Details. For texture embedding, we first optimize SMPL-X parameters for each frame using the point cloud and 3D markers, and then refine them by processing 120-frame chunks with 20-frame overlap. For deformation optimization, the sequence is divided into 20- frame chunks with an overlap of 4 frames. The texture resolution is set to 256, and we uniformly sample 30 to 32 cameras for texture optimization. For the $\Phi _ { \mathrm { G G D } }$ , we first train it on MVHPP for 1M steps, and then fine-tune it for another 600k steps using 90% from DynaHuman and 10% from MVHPP at 2K resolution. We use AdamW [50] with batch size 1 and learning rate $1 0 ^ { - 4 }$ . For Φ<sub>DTG</sub>, we use LoRA [28] to fine-tune [28] the 1.3B Wan video diffusion model [76] with batch size 4 and learning rate $1 0 ^ { - 4 }$

Table 1. Quantitative comparison across datasets. Each result is averaged over two subjects. Higher PSNR and SSIM are better, while lower LPIPS, FID, and FVD are better. The best and second-best results are highlighted within each dataset and each method family.
<table><tr><td>Dataset</td><td>Method</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td><td>FID↓</td><td>FVD↓</td><td>Method</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FID↓</td><td>FVD↓</td></tr><tr><td rowspan="3">DeepCap</td><td>LHM</td><td>30.61</td><td>0.715</td><td>0.201</td><td>50.10</td><td>150.12</td><td>MV</td><td>30.86</td><td>0.721</td><td>0.191</td><td>39.27</td><td>142.18</td></tr><tr><td>+GAS</td><td>30.14</td><td>0.700</td><td>0.248</td><td>57.14</td><td>223.82</td><td>+GAS</td><td>30.48</td><td>0.705</td><td>0.242</td><td>54.28</td><td>222.78</td></tr><tr><td>+VAP</td><td>31.31</td><td>0.730</td><td>0.221</td><td>101.73</td><td>206.88</td><td>+VAP</td><td>32.64</td><td>0.752</td><td>0.197</td><td>73.47</td><td>178.08</td></tr><tr><td rowspan="3"></td><td>+Ours</td><td>31.19</td><td>0.723</td><td>0.200</td><td>48.69</td><td>160.75</td><td>+Ours</td><td>31.95</td><td>0.735</td><td>0.185</td><td>37.05</td><td>134.26</td></tr><tr><td>LHM</td><td>29.65</td><td>0.696</td><td>0.206</td><td>42.64</td><td>108.00</td><td>MV</td><td>29.52</td><td>0.698</td><td>0.189</td><td>36.14</td><td>103.10</td></tr><tr><td>+GAS</td><td>29.40</td><td>0.676</td><td>0.281</td><td>44.32</td><td>167.20</td><td>+GAS</td><td>29.64</td><td>0.680</td><td>0.279</td><td>44.35</td><td>164.61</td></tr><tr><td rowspan="3"></td><td>+VAP</td><td>30.81</td><td>0.729</td><td>0.224</td><td>77.90</td><td>199.85</td><td>+VAP</td><td>31.97</td><td>0.750</td><td>0.206</td><td>64.51</td><td>165.77</td></tr><tr><td>+Ours</td><td>30.70</td><td>0.716</td><td>0.199</td><td>34.97</td><td>104.72</td><td>+Ours</td><td>31.26</td><td>0.728</td><td>0.186</td><td>30.43</td><td>93.95</td></tr><tr><td>LHM</td><td>28.11</td><td>0.704</td><td>0.173</td><td>80.19</td><td>80.19</td><td>MV</td><td>30.06</td><td>0.719</td><td>0.157</td><td>23.24</td><td>69.69</td></tr><tr><td rowspan="3">MetaCap</td><td>+GAS</td><td>29.07</td><td>0.685</td><td>0.275</td><td>100.37</td><td>100.36</td><td>+GAS</td><td>29.33</td><td>0.689</td><td>0.273</td><td>44.50</td><td>99.48</td></tr><tr><td>+VAP</td><td>30.88</td><td>0.727</td><td>0.199</td><td>115.71</td><td>115.71</td><td>+VAP</td><td>32.28</td><td>0.751</td><td>0.177</td><td>43.68</td><td>106.36</td></tr><tr><td>+Ours</td><td>30.36</td><td>0.713</td><td>0.177</td><td>25.68</td><td>60.14</td><td>+Ours</td><td>31.46</td><td>0.728</td><td>0.158</td><td>18.31</td><td>48.59</td></tr><tr><td rowspan="3">DynaHuman</td><td>LHM</td><td>25.32</td><td>0.719</td><td>0.193</td><td>31.13</td><td>75.39</td><td>MV</td><td>25.20</td><td>0.727</td><td>0.181</td><td>25.83</td><td>52.79</td></tr><tr><td>+GAS</td><td>25.74</td><td>0.723</td><td>0.254</td><td>50.03</td><td>124.51</td><td>+GAS</td><td>25.97</td><td>0.725</td><td>0.248</td><td>49.53</td><td>107.15</td></tr><tr><td>+VAP</td><td>26.81</td><td>0.751</td><td>0.211</td><td>39.34</td><td>79.01</td><td>+VAP</td><td>27.64</td><td>0.771</td><td>0.190</td><td>37.51</td><td>54.97</td></tr><tr><td></td><td>+Ours</td><td>26.43</td><td>0.729</td><td>0.202</td><td>22.58</td><td>56.72</td><td>+Ours</td><td>26.56</td><td>0.736</td><td>0.186</td><td>15.28</td><td>37.00</td></tr></table>

Evaluation Metrics. We use PSNR, SSIM [82], and LPIPS [92] to evaluate the image reconstruction quality. To evaluate generative quality at both the image and video levels, we report FID [27] and FVD [74].

## 4.1. Comparison

Competing Methods. Static methods: Initial static avatars are obtained from two input modalities: monocular and multi-view images. LHM [59] is a generalizable method that creates static human avatars from monocular images. MV denotes reconstructing static Gaussian avatars from multi-view images via gradient descent. The obtained static avatars are animated in world space using body motion, as described in Sec. 3.1. Dynamic methods: GAS [51] is a video diffusion-based model that first renders a static avatar to 2D images and then generates surface details in image space. Vid2Avatar-Pro (VAP) [21] is a regressionbased method that maps identity and pose information to 3D Gaussians in texel space.

With methods above, we evaluate how different design choices for dynamics embedding affect dynamic wrinkle generation. Tab. 1 show quantitative comparisons on four datasets, and Fig. 3 visualizes qualitative comparisons on DynaHuman subjects. Compared to LHM [59], MV leverages more input views for reconstruction, resulting in better performance on both reconstruction and generative metrics. GAS [51] adds surface dynamics in the image space, breaking both multi-view and identity consistency. It performs worst in terms of generative metrics. As a regression-based model, VAP [21] tends to regress averaged low-frequency signals, such as coarse shape deformation and lighting variations, which leads to strong reconstruction metrics. However, it struggles to regress high-frequency, motion-dependent surface dynamics and therefore produces overly smooth, nearly uniform wrinkles, resulting in worse generative metrics. Our method achieves the best generative metrics at both the image and video levels, the second-best reconstruction quality, and consistently better visual results for high-frequency surface dynamics. Since generation is probabilistic, the generated dynamics may differ from the ground truth, which can lead to high-frequency rendering differences while remaining visually plausible. In contrast, regression-based methods such as VAP [21] tend to conservatively predict low-frequency appearance changes, yielding strong reconstruction metrics but weak explicit surface dynamics. Therefore, for surface dynamics generation, generative metrics are more suitable than reconstruction metrics, as they better reflect the plausibility of high-frequency motion-dependent details.

## 4.2. Ablation Studies

Human Representation. We evaluate different representations for compressing and reconstructing multi-view human data on one training subject. We compare two representations: (1) VAP [21] with static textures (ST), and (2) our optimized dynamic reference textures (RDT) decoded by the Generalized Gaussian Decoder. VAP embeds dynamic information into a regression model, where pose controls guide the prediction of dynamic 3D Gaussians from static identity textures. While compact, it struggles to reconstruct high-frequency surface dynamics. In contrast, we embed dynamic information explicitly into optimized dynamic reference textures, which can be accurately decoded to recover the original multi-view data with acceptable storage overhead. Moreover, these dynamic textures provide a lightweight 2D generative space for training our Dynamic Texture Generator, making it easier to synthesize fine-grained motion-dependent dynamics than directly regressing 3D Gaussians.

Table 2. Quantitative Comparison on Human Representation. We compare our human representation with VAP [21] on Subject59 from the training set. Our Generalized Gaussian Decoder (GGD) with optimized reference dynamic textures (RT) achieves better image reconstruction with only a moderate storage increase.
<table><tr><td>Representation</td><td>PSNR SSIM</td><td>LPIPS</td><td>Tex/Full Storage</td></tr><tr><td>ST + Decoder (VAP)</td><td>30.36 0.808</td><td>0.160</td><td>89KB / 67.6GB</td></tr><tr><td>RDT + Decoder (Ours)</td><td>32.02 0.874</td><td>0.103</td><td>110MB / 67.6GB</td></tr></table>

Table 3. Quantitative Ablation on Training Datasets. Training with both datasets results in improved generalization and higher reconstruction accuracy.
<table><tr><td>Train Set</td><td>Test Set</td><td>PSNR LPIPS</td><td>FVD</td></tr><tr><td rowspan="2">w/o DynaHuman w/o MVHPP</td><td>DynaHuman</td><td>28.77 0.142</td><td>33.60</td></tr><tr><td>DynaHuman</td><td>29.70 0.124</td><td>25.77</td></tr><tr><td>Ours</td><td>DynaHuman</td><td>29.71 0.122</td><td>22.97</td></tr><tr><td rowspan="2">w/o DynaHuman w/o MVHPP</td><td>MVHPP</td><td>29.45 0.190</td><td></td></tr><tr><td>MVHPP</td><td>27.22 0.213</td><td></td></tr><tr><td>Ours</td><td>MVHPP</td><td>29.51 0.190</td><td>=</td></tr></table>

Influence of Training Dataset. The generalization of our Φ<sub>GGD</sub> depends on the quality and diversity of its training data. MVHPP [44] contains diverse subjects but provides fewer motion variations and slightly lower image quality. As a complement, our DynaHuman offers higher image quality and richer motion diversity but includes fewer subjects. Joint training on both datasets achieves the best reconstruction performance and generalization (see Tab. 3).

Pretrained Video Model Weights. Dynamic multi-view training data remains scarce, making pretrained video diffusion priors [3] important for learning generalizable surface dynamics. Training the Dynamic Texture Generator without a pre-trained prior (PP) would make convergence and generalization difficult, leading to poor results, as illustrated in Fig. 4 and Tab. 4. This underscores the importance of being able to leverage pre-trained priors for dynamics generation and the necessity of our texture representation.

Temporal Model vs. Image Model. To evaluate temporal modeling, we set the temporal window to 1 for the Dynamic Texture Generator, removing temporal information (TI). While per-frame quality remains similar, temporal quality degrades significantly, as reflected by the FVD in Tab. 4. Moreover, the model suffers from sudden changes in cloth appearance, as shown by the red arrows and dashed line in Fig. 4.

Generation vs. Reconstruction. Given generated textures (Ours) and reference dynamic textures (RDT), we ablate the ability of our full generation pipeline and reconstruction module in Tab. 4 and Fig. 3. While our method outperforms baseline dynamic methods, RDT serves as an upper bound by directly using optimized dynamic textures.

Table 4. Quantitative Ablation on Dynamics Prior with Subject71. Without pretrained diffusion weights, our Dynamic Texture Generator has limited performance. Removing the temporal block results in degraded temporal performance.
<table><tr><td>Baselines</td><td>PSNR SSIM</td><td>LPIPS</td><td>FID</td><td>FVD</td></tr><tr><td>w/o PP</td><td>24.70 0.700</td><td>0.240</td><td>60.42</td><td>228.21</td></tr><tr><td>w/o TI</td><td>26.61 0.748</td><td>0.187</td><td>16.91</td><td>55.20</td></tr><tr><td>Ours</td><td>26.61 0.745</td><td>0.182</td><td>13.84</td><td>37.74</td></tr><tr><td>w/RDT</td><td>29.89 0.847</td><td>0.116</td><td>8.47</td><td>20.73</td></tr></table>

![](images/2a91f5145b494895a2b363d8accd87dea0e9930594a1194cad0c5891edde473a.jpg)  
Ours

![](images/ef85ac483f38ced1d91023c6e8772fb32b42bde7907d004987466e14406ae7f4.jpg)  
GT  
Figure 4. Qualitative Ablation on Dynamics Prior. Training the Dynamic Texture Generator from scratch without pretrained diffusion weights yields limited quality. Removing the temporal block reduces temporal consistency and causes frame-to-frame variations (red arrows). The full model achieves the best generation quality and temporal consistency.

## 5. Conclusion

Incorporating dynamics into animatable and generalizable human avatars is crucial for overcoming the uncanny valley and bridging the quality gap between personalized and generalized avatars. We take an important step in this direction by proposing AvatarDynamizer, a method that transforms off-the-shelf static avatars into dynamic ones. We demonstrate that our method outperforms both static and dynamic baselines and further highlight the versatility of our approach in multi-modal settings, such as monocular or multi-view image inputs. Our main technical insight is that dynamics modeling can be formulated as a 2D dynamic texture generation and generalized Gaussian decoding problem, enabling the use of foundational video priors while ensuring multi-view-consistent rendering. Both properties are essential yet lacking in prior work. Looking ahead, relaxing the template assumption should be a key research focus to enable modeling of arbitrary clothing and potentially even topological changes.

## References

[1] Thiemo Alldieck, Gerard Pons-Moll, Christian Theobalt, and Marcus Magnor. Tex2shape: Detailed full human body geometry from a single image. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2293–2303, 2019. 3

[2] Thiemo Alldieck, Mihai Zanfir, and Cristian Sminchisescu. Photorealistic monocular 3d reconstruction of humans wearing clothing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 3

[3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 4, 5, 8

[4] Federica Bogo, Angjoo Kanazawa, Christoph Lassner, Peter Gehler, Javier Romero, and Michael J Black. Keep it smpl: Automatic estimation of 3d human pose and shape from a single image. In European conference on computer vision, pages 561–578. Springer, 2016. 3

[5] Chen Cao, Yanlin Weng, Stephen Lin, and Kun Zhou. 3d shape regression for real-time facial animation. ACM Transactions on Graphics (TOG), 32(4):1–10, 2013. 3

[6] Dongliang Cao, Guoxing Sun, Marc Habermann, and Florian Bernard. Hyper diffusion avatars: Dynamic human avatar generation using network weight space diffusion. In Thirteenth International Conference on 3D Vision, 2025. 3

[7] Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei, and Yaser Sheikh. Openpose: Realtime multi-person 2d pose estimation using part affinity fields. IEEE transactions on pattern analysis and machine intelligence, 43(1):172–186, 2019. 5

[8] Di Chang, Yichun Shi, Quankai Gao, Hongyi Xu, Jessica Fu, Guoxian Song, Qing Yan, Yizhe Zhu, Xiao Yang, and Mohammad Soleymani. Magicpose: Realistic human poses and facial expressions retargeting with identity-aware diffusion. In International Conference on Machine Learning, pages 6263–6285. PMLR, 2024. 2, 18

[9] Di Chang, Hongyi Xu, You Xie, Yipeng Gao, Zhengfei Kuang, Shengqu Cai, Chenxu Zhang, Guoxian Song, Chao Wang, Yichun Shi, et al. X-dyna: Expressive dynamic human image animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5499–5509, 2025. 3

[10] Jianchuan Chen, Jingchuan Hu, Gaige Wang, Zhonghua Jiang, Tiansong Zhou, Zhiwen Chen, and Chengfei Lv. Taoavatar: Real-time lifelike full-body talking avatars for augmented reality via 3d gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10723–10734, 2025. 3

[11] Zhaoxi Chen, Fangzhou Hong, Haiyi Mei, Guangcong Wang, Lei Yang, and Ziwei Liu. Primdiffusion: Volumetric primitives diffusion for 3d human generation. Advances in Neural Information Processing Systems, 36:13664–13677, 2023. 1

[12] Wei Cheng, Ruixiang Chen, Siming Fan, Wanqi Yin, Keyu Chen, Zhongang Cai, Jingbo Wang, Yang Gao, Zhengming Yu, Zhengyu Lin, et al. Dna-rendering: A diverse neural actor repository for high-fidelity human-centric rendering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19982–19993, 2023. 1, 3, 14, 16

[13] Andrey Davydov, Anastasia Remizova, Victor Constantin, Sina Honari, Mathieu Salzmann, and Pascal Fua. Adversarial parametric pose prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10997–11005, 2022. 3

[14] Junting Dong, Qi Fang, Zehuan Huang, Xudong Xu, Jingbo Wang, Sida Peng, and Bo Dai. Tela: Text to layer-wise 3d clothed human generation. In European Conference on Computer Vision, pages 19–36. Springer, 2024. 1

[15] Zijian Dong, Longteng Duan, Jie Song, Michael J.Black, and Andreas Geiger. MoGA: 3d Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction. In International Conference on Computer Vision (ICCV), 2025. 3, 5

[16] EasyMoCap. Easymocap - make human motion capture easier. Github, 2021. 15

[17] Xuan Gao, Jingtao Zhou, Dongyu Liu, Yuqi Zhou, and Juyong Zhang. Constructing diffusion avatar with learnable embeddings. In ACM SIGGRAPH Asia Conference Proceedings, 2025. 3

[18] Dimitrios Gerogiannis, Foivos Paraperas Papantoniou, Rolandos Alexandros Potamias, Alexandros Lattas, and Stefanos Zafeiriou. Arc2avatar: Generating expressive 3d avatars from a single image via id guidance. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10770–10782, 2025. 3

[19] Chen Guo, Xu Chen, Jie Song, and Otmar Hilliges. Human performance capture from monocular video in the wild. In 2021 International Conference on 3D Vision (3DV), pages 889–898. IEEE, 2021. 3

[20] Chen Guo, Tianjian Jiang, Xu Chen, Jie Song, and Otmar Hilliges. Vid2avatar: 3d avatar reconstruction from videos in the wild via self-supervised scene decomposition. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 5

[21] Chen Guo, Junxuan Li, Yash Kant, Yaser Sheikh, Shunsuke Saito, and Chen Cao. Vid2avatar-pro: Authentic avatar from videos in the wild via universal prior. In Proceed ings of the Computer Vision and Pattern Recognition Con ference, pages 5559–5570, 2025. 2, 3, 5, 7, 8, 14, 16, 18

[22] Marc Habermann, Weipeng Xu, Michael Zollhoefer, Gerard Pons-Moll, and Christian Theobalt. Deepcap: Monocular human performance capture using weak supervision. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020. 3, 6, 18, 21

[23] Marc Habermann, Lingjie Liu, Weipeng Xu, Michael Zollhoefer, Gerard Pons-Moll, and Christian Theobalt. Realtime deep dynamic characters. ACM Transactions on Graphics, 40(4), 2021. 3, 6, 14, 16, 18, 21

[24] Sang-Hun Han, Min-Gyu Park, Ju Hong Yoon, Ju-Mi Kang, Young-Jae Park, and Hae-Gon Jeon. High-fidelity 3d hu-

man digitization from single 2k resolution images. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12869–12879, 2023. 18

[25] Yisheng He, Xiaodong Gu, Xiaodan Ye, Chao Xu, Zhengyi Zhao, Yuan Dong, Weihao Yuan, Zilong Dong, and Liefeng Bo. Lam: Large avatar model for one-shot animatable gaussian head. arXiv preprint arXiv:2502.17796, 2025. 3

[26] Eric Hedlin, Helge Rhodin, and Kwang Moo Yi. A simple method to boost human pose estimation accuracy by correcting the joint regressor for the human3. 6m dataset. In 2022 19th Conference on Robots and Vision (CRV), pages 1–7. IEEE, 2022. 3

[27] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 7

[28] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. 7

[29] Shoukang Hu, Tao Hu, and Ziwei Liu. Gauhuman: Articulated gaussian splatting from monocular human videos. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20418–20431, 2024. 3

[30] Yangyi Huang, Ye Yuan, Xueting Li, Jan Kautz, and Umar Iqbal. Adahuman: Animatable detailed 3d human generation with compositional multiview diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13533–13543, 2025. 3

[31] Yasamin Jafarian and Hyun Soo Park. Learning high fidelity depths of dressed humans by watching social media dance videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12753– 12762, 2021. 18

[32] Yuheng Jiang, Chengcheng Guo, Yize Wu, Yu Hong, Shengkun Zhu, Zhehao Shen, Yingliang Zhang, Shaohui Jiao, Zhuo Su, Lan Xu, Marc Habermann, and Christian Theobalt. Topology-aware optimization of gaussian primitives for human-centric volumetric videos. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, New York, NY, USA, 2025. Association for Computing Machinery. 19

[33] Yuheng Jiang, Zhehao Shen, Chengcheng Guo, Yu Hong, Zhuo Su, Yingliang Zhang, Marc Habermann, and Lan Xu. Reperformer: Immersive human-centric volumetric videos from playback to photoreal reperformance. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 11349–11360, 2025. 3

[34] Hanbyul Joo, Tomas Simon, and Yaser Sheikh. Total capture: A 3d deformation model for tracking faces, hands, and bodies. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8320–8329, 2018. 3

[35] Hendrik Junkawitsch, Guoxing Sun, Heming Zhu, Christian Theobalt, and Marc Habermann. Eva: Expressive virtual avatars from multi-view videos. In SIGGRAPH 2025 Conference Papers, pages 1–11, 2025. 2

[36] Angjoo Kanazawa, Michael J Black, David W Jacobs, and Jitendra Malik. End-to-end recovery of human shape and pose. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7122–7131, 2018. 3

[37] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020. 18

[38] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler,¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4), 2023. 4, 6

[39] Nikos Kolotouros, Thiemo Alldieck, Enric Corona, Ed uard Gabriel Bazavan, and Cristian Sminchisescu. Instant 3d human avatar generation using image diffusion models. In European Conference on Computer Vision (ECCV), 2024. 3

[40] Youngjoong Kwon, Dahun Kim, Duygu Ceylan, and Henry Fuchs. Neural human performer: Learning generalizable radiance fields for human performance rendering. Advances in Neural Information Processing Systems, 34: 24741–24752, 2021. 3

[41] Youngjoong Kwon, Lingjie Liu, Henry Fuchs, Marc Habermann, and Christian Theobalt. Deliffas: Deformable light fields for fast avatar synthesis. Advances in Neural Information Processing Systems, 2023. 2

[42] Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, and Timo Aila. Modular primitives for high-performance differentiable rendering. ACM Transactions on Graphics, 39(6), 2020. 5

[43] JP Lewis, Matt Cordner, and Nickson Fong. Pose space deformation: a unified approach to shape interpolation and skeleton-driven deformation. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques, pages 165–172, 2000. 1, 4

[44] Chenghong Li, Hongjie Liao, Yihao Zhi, Xihe Yang, Zhengwentai Sun, Jiahao Chang, Shuguang Cui, and Xi aoguang Han. Mvhumannet++: A large-scale dataset of multi-view daily dressing human captures with richer annotations for 3d human digitization. arXiv preprint arXiv:2505.01838, 2025. 1, 3, 6, 8, 14, 16, 17, 18

[45] Linzhou Li, Yumeng Li, Yanlin Weng, Youyi Zheng, and Kun Zhou. Rgbavatar: Reduced gaussian blendshapes for online modeling of head avatars. In The IEEE/CVF Confer ence on Computer Vision and Pattern Recognition, 2025. 3

[46] Peng Li, Wangguandong Zheng, Yuan Liu, Tao Yu, Yangguang Li, Xingqun Qi, Xiaowei Chi, Siyu Xia, Yan-Pei Cao, Wei Xue, et al. Pshuman: Photorealistic singleimage 3d human reconstruction using cross-scale multiview diffusion and explicit remeshing. In Proceedings of the computer vision and pattern recognition conference, pages 16008–16018, 2025. 3

[47] Zhe Li, Zerong Zheng, Lizhen Wang, and Yebin Liu. Animatable gaussians: Learning pose-dependent gaussian maps for high-fidelity human avatar modeling. In Proceed-

ings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19711–19722, 2024. 2, 3

[48] Tingting Liao, Hongwei Yi, Yuliang Xiu, Jiaxiang Tang, Yangyi Huang, Justus Thies, and Michael J Black. Tada! text to animatable digital avatars. In 2024 International Conference on 3D Vision (3DV), pages 1508–1519. IEEE, 2024. 1, 5

[49] Di Liu, Teng Deng, Giljoo Nam, Yu Rong, Stanislav Pidhorskyi, Junxuan Li, Jason Saragih, Dimitris N. Metaxas, and Chen Cao. Lucas: Layered universal codec avatars. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 3

[50] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. 7

[51] Yixing Lu, Junting Dong, Youngjoong Kwon, Qin Zhao, Bo Dai, and Fernando De la Torre. Gas: Generative avatar synthesis from a single image. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12883–12893, 2025. 2, 3, 5, 7, 17

[52] Qianli Ma, Jinlong Yang, Michael J. Black, and Siyu Tang. Neural point-based shape modeling of humans in challenging clothing. In 2022 International Conference on 3D Vision (3DV), 2022. 3

[53] OpenAI. Dall·e, 2024. 14

[54] Haokai Pang, Heming Zhu, Adam Kortylewski, Christian Theobalt, and Marc Habermann. Ash: Animatable gaussian splats for efficient and photoreal human rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1165–1175, 2024. 2, 3, 4, 5

[55] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black. Expressive body capture: 3d hands, face, and body from a single image. In Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2019. 2, 3, 4, 5, 15, 17

[56] Sida Peng, Junting Dong, Qianqian Wang, Shangzhan Zhang, Qing Shuai, Xiaowei Zhou, and Hujun Bao. Animatable neural radiance fields for modeling dynamic human bodies. In ICCV, 2021. 3, 16

[57] Sida Peng, Yuanqing Zhang, Yinghao Xu, Qianqian Wang, Qing Shuai, Hujun Bao, and Xiaowei Zhou. Neural body: Implicit neural representations with structured latent codes for novel view synthesis of dynamic humans. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9054–9063, 2021. 16

[58] Shenhan Qian, Tobias Kirschstein, Liam Schoneveld, Davide Davoli, Simon Giebenhain, and Matthias Nießner. Gaussianavatars: Photorealistic head avatars with rigged 3d gaussians. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20299– 20309, 2024. 3

[59] Lingteng Qiu, Xiaodong Gu, Peihao Li, Qi Zuo, Weichao Shen, Junfei Zhang, Kejie Qiu, Weihao Yuan, Guanying Chen, Zilong Dong, et al. Lhm: Large animatable human reconstruction model for single image to 3d in seconds. In

Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 14184–14194, 2025. 1, 3, 5, 6, 7, 17, 22

[60] Lingteng Qiu, Peihao Li, Heyuan Li, Qi Zuo, Xiaodong Gu, Yuan Dong, Weihao Yuan, Rui Peng, Siyu Zhu, Xiaoguang Han, Guanying Chen, and Zilong Dong. Lhm++: An ef ficient large human reconstruction model for pose-free images to 3d. arXiv preprint arXiv:2503.10625, 2025. 3

[61] Pramod Rao, Gereon Fox, Abhimitra Meka, Mallikarjun B R, Fangneng Zhan, Tim Weyrich, Bernd Bickel, Hans-Peter Seidel, Hanspeter Pfister, Wojciech Matusik, Mohamed Elgharib, and Christian Theobalt. Lite2relight: 3daware single image portrait relighting. In ACM SIGGRAPH 2024 Conference Proceedings, 2024. 3

[62] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image¨ synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 5

[63] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015. 18

[64] Shunsuke Saito, Zeng Huang, Ryota Natsume, Shigeo Morishima, Angjoo Kanazawa, and Hao Li. Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 2304–2314, 2019. 3

[65] Ruizhi Shao, Youxin Pang, Zerong Zheng, Jingxiang Sun, and Yebin Liu. Human4dit: 360-degree human video gen eration with 4d diffusion transformer. ACM Transactions on Graphics (TOG), 43(6), 2024. 2

[66] Ashwath Shetty, Marc Habermann, Guoxing Sun, Diogo Luvizon, Vladislav Golyanik, and Christian Theobalt. Holoported characters: Real-time free-viewpoint rendering of humans from sparse rgb cameras. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1206–1215, 2024. 3

[67] Carsten Stoll, Nils Hasler, Juergen Gall, Hans-Peter Seidel, and Christian Theobalt. Fast articulated motion tracking using a sums of gaussians body model. In 2011 international conference on computer vision, pages 951–958. IEEE, 2011. 3

[68] Guoxing Sun, Xin Chen, Yizhang Chen, Anqi Pang, Pei Lin, Yuheng Jiang, Lan Xu, Jingyi Yu, and Jingya Wang. Neural free-viewpoint performance rendering under com plex human-object interactions. In Proceedings of the 29th ACM International Conference on Multimedia, pages 4651–4660, 2021. 3

[69] Guoxing Sun, Rishabh Dabral, Pascal Fua, Christian Theobalt, and Marc Habermann. Metacap: Meta-learning priors from multi-view imagery for sparse-view human per formance capture and rendering. In ECCV, 2024. 3, 6, 18, 21

[70] Guoxing Sun, Rishabh Dabral, Heming Zhu, Pascal Fua, Christian Theobalt, and Marc Habermann. Real-time free-

view human rendering from sparse-view rgb videos using double unprojected textures. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 3, 14, 16, 18

[71] Timo Teufel, Pulkit Gera, Xilong Zhou, Umar Iqbal, Pramod Rao, Jan Kautz, Vladislav Golyanik, and Christian Theobalt. Humanolat: A large-scale dataset for fullbody human relighting and novel-view synthesis. In International Conference on Computer Vision (ICCV), 2025. 14

[72] Justus Thies, Michael Zollhofer, Marc Stamminger, Christian Theobalt, and Matthias Nießner. Face2face: Real-time face capture and reenactment of rgb videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2387–2395, 2016. 3

[73] Phong Tran, Egor Zakharov, Long-Nhat Ho, Liwen Hu, Adilbek Karmanov, Aviral Agarwal, McLean Goldwhite, Ariana Bermudez Venegas, Anh Tuan Tran, and Hao Li. Voodoo xp: Expressive one-shot head reenactment for vr telepresence. ACM Transactions on Graphics, Proceedings of the 17th ACM SIGGRAPH Conference and Exhibition in Asia 2024, (SIGGRAPH Asia 2024), 12/2024, 2024. 3

[74] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 7

[75] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 18

[76] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 7

[77] Guangyuan Wang, Li Hu, Dechao Meng, Zhongyi Zhang, Peng Zhang, Mingyang Huang, Ruoshi Zhang, Ke Sun, Zhe Zhang, Xingjun Wang, Gang Cheng, and Bang Zhang. Wan-animate-2: Real-time end-to-end character animation via diffusion transformer. arXiv preprint arXiv:TODO.06009, 2026. 2, 18, 20

[78] Shaofei Wang, Marko Mihajlovic, Qianli Ma, Andreas Geiger, and Siyu Tang. Metaavatar: Learning animatable clothed human models from few depth images. In Advances in Neural Information Processing Systems, 2021. 3

[79] Yi Wang, Xin Tao, Xiaojuan Qi, Xiaoyong Shen, and Jiaya Jia. Image inpainting via generative multi-column convolutional neural networks. In Advances in Neural Information Processing Systems, pages 331–340, 2018. 5

[80] Yiming Wang, Qin Han, Marc Habermann, Kostas Daniilidis, Christian Theobalt, and Lingjie Liu. Neus2: Fast learning of neural implicit surfaces for multi-view reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 5, 14

[81] Yuhan Wang, Fangzhou Hong, Shuai Yang, Liming Jiang, Wayne Wu, and Chen Change Loy. Meat: Multiview diffusion model for human generation on megapixels with mesh

attention. In Proceedings of the Computer Vision and Pat tern Recognition Conference, pages 11297–11306, 2025. 3

[82] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 5, 7

[83] Minye Wu, Yuehao Wang, Qiang Hu, and Jingyi Yu. Multi view neural human rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1682–1691, 2020. 3

[84] Yue Wu, Xuanhong Chen, Yufan Wu, Wen Li, Yuxi Lu, and Kairui Feng. Fastavatar: Towards unified and fast 3d avatar reconstruction with large gaussian reconstruction transformers. In The Fourteenth International Conference on Learning Representations, 2026. 3

[85] Yuliang Xiu, Jinlong Yang, Dimitrios Tzionas, and Michael J. Black. ICON: Implicit Clothed humans Obtained from Normals. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13296–13306, 2022. 3

[86] Yuliang Xiu, Jinlong Yang, Xu Cao, Dimitrios Tzionas, and Michael J. Black. ECON: Explicit Clothed humans Optimized via Normal integration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 3

[87] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. In Proceed ings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1481–1490, 2024. 2, 3

[88] Yuxuan Xue, Xianghui Xie, Margaret Kostyrko, and Gerard Pons-Moll. Infinihuman: Realistic 3d human creation with precise control. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, pages 1–12, 2025. 3, 5

[89] Tao Yu, Zerong Zheng, Kaiwen Guo, Pengpeng Liu, Qiong hai Dai, and Yebin Liu. Function4d: Real-time human vol umetric capture from very sparse consumer rgbd sensors. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5746–5756, 2021. 18

[90] Zhiyuan Yu, Zhe Li, Hujun Bao, Can Yang, and Xiaowei Zhou. Humanram: Feed-forward human reconstruction and animation model using transformers. In Proceedings ofthe Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1– 13, 2025. 3

[91] Yifei Zeng, Yuanxun Lu, Xinya Ji, Yao Yao, Hao Zhu, and Xun Cao. Avatarbooth: High-quality and customizable 3d human avatar generation. arXiv preprint arXiv:2306.09864, 2023. 3

[92] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 7

[93] Weitian Zhang, Yichao Yan, Yunhui Liu, Xingdong Sheng, and Xiaokang Yang. E3gen: Efficient, expressive and editable avatars generation. In Proceedings of the 32nd ACM

International Conference on Multimedia, page 6860–6869, 2024. 1

[94] Zerong Zheng, Han Huang, Tao Yu, Hongwen Zhang, Yandong Guo, and Yebin Liu. Structured local radiance fields for human avatar modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 5

[95] Yuxiao Zhou, Menglei Chai, Alessandro Pepe, Markus Gross, and Thabo Beeler. Groomgen: A high-quality generative hair model using hierarchical latent representations. ACM Transactions on Graphics (TOG), 42(6):1–16, 2023. 3

[96] Zhenglin Zhou, Fan Ma, Hehe Fan, and Tat-Seng Chua. Zero-1-to-a: Zero-shot one image to animatable head avatars using video diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 3

[97] Heming Zhu, Fangneng Zhan, Christian Theobalt, and Marc Habermann. Trihuman: a real-time and controllable tri-plane representation for detailed human geometry and appearance synthesis. ACM Transactions on Graphics, 44 (1):1–17, 2024. 2

[98] Heming Zhu, Guoxing Sun, Christian Theobalt, and Marc Habermann. Uma: Ultra-detailed human avatars via multilevel surface alignment. ACM Transactions on Graphics, 2026. 3

[99] Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. Champ: Controllable and consistent human image animation with 3d parametric guidance. In European Conference on Computer Vision (ECCV), 2024. 2, 3

[100] Yiyu Zhuang, Jiaxi Lv, Hao Wen, Qing Shuai, Ailing Zeng, Hao Zhu, Shifeng Chen, Yujiu Yang, Xun Cao, and Wei Liu. Idol: Instant photorealistic 3d human creation from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26308– 26319, 2025. 3, 5

[101] Anton Zubekhin, Heming Zhu, Paulo Gotardo, Thabo Beeler, Marc Habermann, and Christian Theobalt. Giga: Generalizable sparse image-driven gaussian humans. International Conference on 3D Vision (3DV), 2026. 1, 3, 5

In this supplemental material, we present more qualitative (Sec. B and Sec. G) and ablative (Sec. C) results. We discuss the drawbacks of big 2D-only method and compare our method with them (Sec. I). Then we provide more details regarding the large-scale DynaHuman dataset (Sec. D). Further, we provide more implementation details regarding texture optimization (Sec. E) and network structures (Sec. F). We perform user study on comparison methods (Sec. H). We discuss the limitations of our method (Sec. J). Lastly, we carefully discuss the potential ethical issues and concerns (Sec. K).

## A. Conceptual Comparison of Dynamic Human Representation Paradigms

Fig. 6 compares three design choices for dynamic human representation. Regression-based approaches are effective for static 3D reconstruction, but their extension to dynamic humans remains difficult. Deterministic regression is not well suited for modeling fine-grained surface dynamics under limited data, as motion-dependent details such as wrinkles are high-frequency and inherently ambiguous: the same identity and pose can correspond to multiple plausible surface states depending on unobserved physical factors. As a result, these methods often produce over-smoothed or nearly static dynamics, even when motion conditions are provided. To alleviate this limitation, they typically require large-scale multi-view motion data, which is expensive to collect and often not publicly available. Another option is to condition a video model on a static human rendering, allowing the model to generate plausible dynamics from learned 2D video priors. However, since the generation is performed in image space, the results do not explicitly guarantee 3D consistency across time and viewpoints. Our representation is designed to address this gap. It leverages video priors for dynamic generation while using the Generalized Gaussian Decoder to produce a dynamic 3D human representation that remains consistent in 3D.

## B. Multi-modal Input Results

We present additional qualitative results in Fig. 5, showing that our method can generate dynamic avatars from multimodal inputs, like text [53] and 3D scans [71]. Given either textual input or a static 3D scan, we first generate the identity texture and a coarse body shape, represented by an SMPL-X template with personalized offsets. Through our proposed pipeline, we synthesize clothing dynamics induced by the subject’s motion, enriching the static rigged avatar with realistic and expressive motion-aware wrinkle details.

## C. Qualitative Ablations

Representation Ablation. We compare our dynamictexture representation against the static-texture representation of VAP [21]. Both methods have access to the full multi-view videos of the training subject. However, VAP implicitly encodes dynamics within the regression model through a static identity texture, whereas AvatarDynamizer supplies an explicit per-frame dynamic texture to the Generalized Gaussian Decoder (GGD). As shown in Fig. 7, the GGD faithfully reconstructs fine-grained wrinkles when they are explicitly represented in the dynamic texture. In contrast, the static-texture baseline results in a flat and blurry appearance.

Training Dataset Ablation. Fig. 8 presents qualitative ablations analyzing the influence of the training datasets on the Generalized Gaussian Decoder. Removing either DynaHuman or MVHPP [44] from training leads to inaccurate coarse shape estimation, uneven lighting, and ghosting artifacts. In contrast, leveraging both datasets leads to significantly improved robustness and reconstruction quality.

## D. DynaHuman Dataset

Capture setup. DynaHuman is captured in a dedicated multi-camera studio equipped with 100 synchronized, calibrated 4K cameras arranged in a dome configuration that covers the subject from all elevations. Notably, the studio is equipped with uniform lighting, which facilitates accurate photometric geometry reconstruction. As shown in Fig. 9 and Tab. 5, subjects in DynaHuman are dressed in diverse clothing styles, such as casual, formal, and sportswear, with varying degrees of tightness, which enhances the diversity of clothing deformation and dynamics.

Statistics. The dataset comprises 58 subjects, each recorded for approximately 18 minutes (27,000 frames at 25 fps) of continuous, natural full-body motion. Each subject performs motions including locomotion (walking, running, jogging), dynamic whole-body actions (jumping, squatting, turning), and expressive upper-body gestures (arm raises, waving, reaching), deliberately chosen to elicit large, rapid cloth deformations. Each frame in the recorded multi-view sequence is annotated with SMPL-X tracking, along with ground-truth geometry reconstructed using NeuS2 [80]. Additionally, we provide the corresponding texture maps as described in the main paper.

Comparison with existing datasets. As summarized in Tab. 1 of the main paper, existing datasets fall into two extremes: DDC [23] and DUT [70] provide rich clothing dynamics but cover very few identities. In contrast, MVHumanNet++ [44] and DNA-Rendering [12] include many subjects, but these subjects mostly perform slow or static motions. DynaHuman bridges this gap, featuring both scale (58 subjects) and dynamic diversity, enabling generalizable

Rendering Results

![](images/6569d2bb13ab96f971abb1d8ad1d70f1e1d66d7d417951e421e1d7e0781e1841.jpg)  
Identity Texture & Coarse Shape

![](images/9b61ca773a4d91bf754d1188211006b56d934d59f40cd6af3b8b4a4ff1983d22.jpg)

Figure 5. Multi-modal Avatar Generation. Our method transforms static avatars that are acquired from multi-modal inputs into dynamic ones. Here, we show static avatars that are created from text and 3D scans. Next, the static avatar is converted into identity texture and coarse shape. Finally, we can drive the avatar using novel skeletal motions. We highlight the different wrinkle patters and dynamics that are induced by the motion and that our method can faithfully generate.  
![](images/aef4ff066abf8e44cca0df4a365d8a8fd985ce627f04548e365c257afbadf530.jpg)  
Figure 6. Conceptual Comparison of Generalized and Dynamic Human Representations. a) Regressing 3D human dynamics directly cannot leverage foundational video priors. Thus, such methods rely on large-scale (closed-source) multi-view data. While regression-based models perform well for static reconstruction, they struggle with dynamic reconstruction. b) Leveraging a static rendering as conditioning for a video model allows generative modeling of dynamics. However, since the model operates in 2D image space, 3D consistency is not ensured. c) Our human representation unites the best of both worlds and enables generative modeling and video prior usage for dynamics while our Generalized Gaussian Decoder predicts a dynamic 3D human that is 3D consistent.

learning of realistic cloth dynamics.

## E. Texture Embedding via Multi-stage Optimization

For each frame, we first obtain point cloud geometry p and 3D markers m from multi-view images as illustrated in Sec. 3.3 of the main paper. Next, we introduce how to register SMPL-X [55] with coarse shape, body motion, and deformations.

SMPL-X Tracking. We split SMPL-X tracking into three steps. In the first step, we follow EasyMocap [16], which estimates initial SMPL-X parameters from m as

$$
\begin{array} { l } { \hat { \mathcal { M } } ^ { \mathrm { s t e p 1 } } , \boldsymbol { \hat { \beta } } ^ { \mathrm { s t e p 1 } } = \arg \underset { { \mathcal { M } , \beta } } { \mathrm { m i n } } } \\ { \lambda _ { m } \| { \bf X } ( { \bf V } ( { \mathcal { M } } , \boldsymbol { \beta } ) ) - { \bf m } \| _ { 2 } ^ { 2 } } \\ { \displaystyle ~ + \lambda _ { \mathcal { M } } \sum _ { t = 2 } ^ { T } \big \| \mathcal { M } ^ { t } - \mathcal { M } ^ { t - 1 } \big \| _ { 2 } ^ { 2 } + \lambda _ { \beta } \| \beta \| _ { 2 } ^ { 2 } , } \end{array}\tag{7}
$$

where X(·) is the marker regressor.

Table 5. Dataset Comparison. Statistics are based on the publicly available versions of the datasets (e.g., MVHPP [44] only releases part of the data). Our dataset matches or surpasses person-specific approaches [23, 56, 70] in motion diversity and number of views while providing significantly more identities. Compared to large-scale identity datasets [12, 44], our dataset has significantly higher motion diversity, views, and frames per subject. Besides, we provide high-precision geometry and registered textures. Thus, our dataset bridges an important gap in literature to cover surface dynamics.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>FramesPer Subject</td><td rowspan=1 colspan=1>Views</td><td rowspan=1 colspan=1>Identities</td><td rowspan=1 colspan=1>MotionDiversity</td><td rowspan=1 colspan=1>Geometry</td><td rowspan=1 colspan=1>Texture</td><td rowspan=1 colspan=1>4KResolution</td><td rowspan=1 colspan=1>Frames(In total)</td></tr><tr><td rowspan=1 colspan=1>DDC [23]</td><td rowspan=1 colspan=1>~26,000</td><td rowspan=1 colspan=1>~90</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>11.8M</td></tr><tr><td rowspan=1 colspan=1>DUT [70]</td><td rowspan=1 colspan=1>~35,000</td><td rowspan=1 colspan=1>~116</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>8.1M</td></tr><tr><td rowspan=1 colspan=1>ZJU [57]</td><td rowspan=1 colspan=1>~500-2,000</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>180K</td></tr><tr><td rowspan=1 colspan=1>DNA [12]</td><td rowspan=1 colspan=1>~200</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>500</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>(√)</td><td rowspan=1 colspan=1>67.5M</td></tr><tr><td rowspan=1 colspan=1>MVHPP [44]</td><td rowspan=1 colspan=1>~60</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>4,500</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>645.1M</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>27,000</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>58</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>y</td><td rowspan=1 colspan=1>156.6M</td></tr></table>

![](images/3fb5b7445dd5120533f8dfe7f248ae03ca2b5f6c1b40c28574d92c361e45aeca.jpg)  
VAP

![](images/16f28b6a1a2f04acf769a48e83ec00f0c9e6a98c26737b8d4a52ed2213e13200.jpg)  
RDT+GGD

![](images/82798ffb1c4b517b82e333da35cfeab8e1d9d66017a4daec577a2ea8dbfb2560.jpg)  
GT

![](images/a359148d54d267c874e7e35a5306d63df251aa55801ce2a6f3564cad5406a76f.jpg)  
VAP

![](images/bc451f3b6b32fb35547449af29d82b2e463e384f3c8d807ecf6ec6349430f4b1.jpg)  
RDT+GGD

![](images/6ce4324bbd5474204733e2778bca8a025b89fe4da0eee39efe837d447803f3ce.jpg)  
GT

Figure 7. Representation Ablation. The qualitative ablation over the avatar representation on the training split. Given the same multiview videos, compared with the baseline approach, i.e., VAP [21], our proposed Generalized Gaussian Decoder (GGD), which takes the reference dynamic texture (RDT) as input, reconstructs the pose-dependent wrinkles more accurately.

In the second step, we refine the SMPL-X parameters per frame with p and m for better alignment, initialized by M<sup>ˆ</sup> <sup>step1</sup> and $\hat { \boldsymbol { \beta } } ^ { \mathrm { s i e p 1 } }$

$$
\begin{array} { r } { \hat { \mathcal { M } } ^ { \mathrm { s t e p 2 } } , \boldsymbol { \hat { \beta } } ^ { \mathrm { s t e p 2 } } = \arg \underset { \mathcal { M } ^ { t } , \beta ^ { t } } { \operatorname* { m i n } } \lambda _ { m } \left\| \mathbf { X } \big ( \mathbf { V } ( \mathcal { M } ^ { t } , \beta ) \big ) - \mathbf { m } ^ { t } \right\| _ { 2 } ^ { 2 } } \\ { + \lambda _ { p } \left\| \mathbf { V } ( \mathcal { M } ^ { t } , \beta ) - \mathbf { p } ^ { t } \right\| _ { 2 } ^ { 2 } . } \end{array}\tag{8}
$$

In the third step, we smooth the SMPL-X parameters with p and m, initialized by $\hat { \mathcal { M } } ^ { \mathrm { s t e p 2 } }$ and $\hat { \boldsymbol { \beta } } ^ { \mathrm { s t e p 2 } }$

$$
\begin{array} { r l r } {  { \hat { \mathcal { M } } ^ { \mathrm { s t e p 3 } } = \arg \operatorname* { m i n } _ { \mathcal { M } } \lambda _ { m } \sum _ { t = 1 } ^ { T } \| \mathbf { X } \big ( \mathbf { V } ( \mathcal { M } ^ { t } , \beta ) \big ) - \mathbf { m } ^ { t } \| _ { 2 } ^ { 2 } } } \\ & { } & { + \lambda _ { p } \sum _ { t = 1 } ^ { T } \| \mathbf { V } ( \mathcal { M } ^ { t } , \beta ) - \mathbf { p } ^ { t } \| _ { 2 } ^ { 2 } } \\ & { } & { + \lambda _ { M } \sum _ { t = 2 } ^ { T } \| \mathcal { M } ^ { t } - \mathcal { M } ^ { t - 1 } \| _ { 2 } ^ { 2 } . \quad \quad } \end{array}\tag{9}
$$

Deformation Optimization. With optimized SMPL-X body motion M and coarse shape $\beta ,$ we optimize time-

varying cloth displacements $\tilde { \mathbf { d } } _ { \mathrm { c l o t h } }$

$$
\begin{array} { r l } { \left. { \hat { \tilde { \mathbf { d } } } _ { \mathrm { c l o t h } } = \operatorname* { m i n } _ { \tilde { \mathbf { d } } _ { \mathrm { c l o t h } } } \sum _ { t = 1 } ^ { T } \left\| \mathbf { V } ( \mathcal { M } ^ { t } , \beta , \mathbf { d } _ { \mathrm { h a i r } } , \tilde { \mathbf { d } } _ { \mathrm { c l o t h } } ^ { t } ) - \mathbf { p } ^ { t } \right\| _ { 2 } ^ { 2 } } } \\ & { + \sum _ { t = 1 } ^ { T - 1 } \left\| \mathbf { V } ( \mathcal { M } ^ { t } , \beta , \mathbf { d } _ { \mathrm { h a i r } } , \tilde { \mathbf { d } } _ { \mathrm { c l o t h } } ^ { t } ) \right. } \\ & { \left. - \mathbf { V } ( \mathcal { M } ^ { t + 1 } , \beta , \mathbf { d } _ { \mathrm { h a i r } } , \tilde { \mathbf { d } } _ { \mathrm { c l o t h } } ^ { t + 1 } ) \right\| _ { 2 } ^ { 2 } , \qquad ( \mathrm { C } ^ { t + 1 } , \beta , \mathbf { d } _ { \mathrm { c l o t h } } ^ { t + 1 } ) \right\| _ { 2 } ^ { 2 } , } \end{array}\tag{10}
$$

where $\mathbf { d } _ { \mathrm { { h a i r } } }$ comes from the coarse shape of the initial static pose.

Texture Optimization. Given the deformed human geometry from the previous step, for each frame, we then optimize the texture of the avatar, i.e., unproject the images onto the texture space, using differentiable rasterization. The exact formulation is provided in Eq. 3 in the main paper.

## F. More Implementation Details

As head modeling is beyond the main scope of body avatar dynamization in this work, we employ a head replacement strategy in VAP [21] and ours to ensure stable head rendering. Specifically, we replace the head Gaussian splats of the dynamic avatars with those from the corresponding static avatars.

![](images/c9355943eb082a1d9335b33394d2df6963bfcefeaca79e7c3accdcfa97abe5ee.jpg)  
Ours  
Subject71

![](images/4efdb51ce37a646028ab23c035d0a8f03e171b5610b7b5146449084ef5dc505e.jpg)  
w/o DynaHuman

![](images/fad4829ecf7b9fe64887024fa467f85dcacc59ba754217cd143875093548de3f.jpg)  
w/o MVHPP

![](images/34c87ee94a5b4820be975ba1ae1cbdbdec45780ac20052daa39f4a5d93ffaa76.jpg)  
GT

![](images/aa5fdb15366917be99118150b4b2dd7c473f55942c5e8d87f6484b84ee7fee17.jpg)  
Ours

![](images/34ecb94045a5b8276584ae017b8f27415c88f4eaf0a1f7e6388f2be971aec2db.jpg)  
104344  
w/o DynaHuman

![](images/36e5d16774b9192c21e20e388d2c3c9988ee3af9a433e06998f0ff7e36bca827.jpg)  
GT  
w/o MVHPP

Figure 8. Dataset Ablation. The qualitative ablation over the training datasets. Removing the DynaHuman dataset from training leads to uneven lighting, while excluding the MVHPP dataset [44] from training could cause severe artifacts. By combining our DynaHuman dataset and the MVHPP dataset, our Generalized Gaussian Decoder achieves better reconstruction.  
![](images/696162c4e5ef8832f97b899ae2596aff7e19cd26c9b2307d489c025b45d32c47.jpg)  
Subject Scans of A-Pose

![](images/77cdb8c0158a649a14b39e0b04fb7edaebe5bd557f08d8fa0677c9c3b2cb6d7f.jpg)  
Registered SMPLX & Corresponding Textures

![](images/eedefc3d5cf9f9c814e07743f9144d3baf2897fd2ae2de6b3205db97bbab9879.jpg)  
Multi-view Images

![](images/d8efb597da06636447f5b4e34bb17755780d4c431c1df29dd8bddbff1d010c9b.jpg)  
Temporal Images

Figure 9. Dataset Visualization. For each subject, we provide an A-pose 3D scan, multi-view videos, and registered and deformed SMPL X as well as dynamic texture annotations.

Evaluation. For image reconstruction evaluation, we uniformly sample every 10 frames from the sequences of the testing subjects and select four camera views that are uniformly placed around the subject. The metrics are computed at a resolution of 2K. For generative quality evaluation, we use 26,800 frames from each test subject sequence and select a camera view facing the subject at the first frame. The images are cropped and resized to 512 × 512, with the subject centered in each frame.

LHM. Since LHM [59] takes a monocular image as input, we provide a front-facing ‘A’-pose image and multi-view estimated SMPL-X [55] shape parameters β for fair comparison. We use the official ‘LHM-500M’ checkpoint for all experiments. We additionally optimize a static Gaussian avatar from LHM outputs with the same number and ordering of points as the 3D methods for head replacement.

MV. MV denotes reconstructing an initial static human avatar from multi-view images. Instead of optimizing from scratch, we feed the identity map into our Generalized Gaussian Decoder and optimize it for 5k steps with a cropped head region loss weight of 1.0 and a regularization weight of 0.01.

GAS. Since GAS [51] does not provide training code, we use the official checkpoint and perform inference in the ‘novel pose’ mode. Note that although we do not fine-tune GAS on our DynaHuman dataset, it has already been trained on several other datasets in addition to MVHPP [44], including THuman2.1 [89], 2K2K [24], TikTok [31], and additional internet videos, which are not used for training our approach.

VAP. Since the source code and training data of VAP [21] are not publicly available, we re-implement it using the same datasets, network architecture (UNet [63]), and training strategy as our Generalized Gaussian Decoder. The main difference between the re-implemented VAP and our Generalized Gaussian Decoder lies in the input representation, $\mathrm { i . e . , }$ a static identity texture versus a dynamic texture. Our re-implementation slightly differs from their original one – representing the canonical space in texture space or orthogonal projection space, and predicting Gaussian deltas versus directly predicting Gaussian parameters – to better evaluate the core difference in representation, i.e., dynamic vs. static textures. We highlight that our approach achieves better quality and generalization than VAP while requiring significantly less data, i.e., VAP was originally trained on 1000 dynamic human sequences, which are not open source.

Ours. For the Generalized Gaussian Decoder, we adopt a network architecture similar to DUT [70]. The temporal window size of our Dynamic Texture Generator is 49, with an overlap of 5 frames during inference. We fine-tune the video diffusion model for 100k steps on 4 NVIDIA H100 GPUs. We set the LoRA weight to 1.0 for LHM-created avatars and 0.8 for MV-created avatars.

RDT+GGD. In the reference dynamic texture (RDT) and Generalized Gaussian Decoder (GGD) experiment, we use reference textures optimized from multi-view images of the test subjects and feed them into the Generalized Gaussian Decoder. Thus, the model operates in a reconstruction setting, i.e., reconstructing Gaussians from the reference dynamic textures.

## G. More Qualitative Results.

Fig. 12 and Fig. 13 shows additional qualitative results on DeepCap [22], DDC [23], MetaCap [69] and in-thewild monocular images. Our method successfully generates pose-dependent wrinkles across different motions and clothing types. For in-the-wild subjects, we drive them with the same body motion. The wrinkles evolve with the motion and vary across different clothing types.

## H. User Study

Beyond qualitative and quantitative evaluations, we conduct a user study to assess human preferences on the perceptual quality of different methods. A total of 18 participants took part in the study and were asked to compare the results on five subjects according to the following questions:

• Q1. How well do the methods preserve the identity of the original subject?

• Q2. How do these methods present physically plausible surface dynamics?

• Q3. Are these results consistent across different views?

• Q4. The overall quality considering all technical aspects. As shown in Tab. 6, our method is preferred in terms of identity preservation, physical plausibility, and overall quality, and achieves the second-best score in view consistency. Although view consistency is included in the study, it is worth noting that methods based on explicit 3D primitives, including the static avatar baseline, VAP, and ours, are naturally consistent when rendering novel views.

## I. Discussions and Comparisons with Big 2Donly models.

Inspired by the success of scaling laws [37], Wan-Animate 2 [77] scales up the 2D-based paradigm exemplified by MagicPose [8], leveraging substantially larger training data and a more advanced diffusion architecture. Given a reference image and a conditional motion video, these methods generate an animated 2D video of the reference subject following the prescribed motion. However, they do not employ explicit 3D representations. As a result, their temporal and spatial consistency must be learned implicitly through attention mechanisms [75] and large-scale training data, which does not guarantee consistent appearance across time or viewpoints. Fig. 10 illustrates typical failures in maintaining temporal and spatial consistency. Fig. 11 shows additional failure cases of Wan-Animate 2 on a DynaHuman subject, including incorrect motion embedding, poseto-appearance ambiguity, inconsistent lighting, inaccurate body shape, and background artifacts. We quantitatively compare our method with Wan-Animate 2 [77] in Tab. 7, where our method consistently achieves better performance. Please refer to the main video for qualitative comparisons.

## J. Limitations.

Our method effectively transforms a static avatar into a dynamic one, producing realistic renderings with vivid surface dynamics while preserving spatial and temporal consistency. Nevertheless, several challenges remain: First, the current pipeline relies on motion tracking and geometry registration, which can be less robust for garments with extreme non-rigid deformation. Extending generalization to more complex clothing and identifying a suitable canonical space – potentially a unified 2D representation – remain future work. Second, as with most template-based human modeling methods, our representation relies on a fixed mesh template, i.e., SMPL-X, which limits its ability to explicitly model topological changes such as opening a jacket. While recent work explores Gaussian-based tracking of such changes [32], generalizable tracking across subjects is still challenging. Third, explicit texture representations may suffer from incomplete observations caused by self-occlusion during texture embedding.

<table><tr><td></td><td>Identity Consistency (Q1)</td><td>Physical Plausibility (Q2)</td><td>View Consistency (Q3)</td><td>Overall Score (Q4)</td></tr><tr><td>Static Avatar</td><td>3.944/5.0</td><td>3.289/5.0</td><td>4.089/5.0</td><td>3.644/5.0</td></tr><tr><td>+GAS</td><td>1.233/5.0</td><td>1.289/5.0</td><td>1.356/5.0</td><td>1.256/5.0</td></tr><tr><td>+VAP</td><td>3.511/5.0</td><td>2.622/5.0</td><td>3.689/5.0</td><td>2.878/5.0</td></tr><tr><td>+Ours</td><td>4.000/5.0</td><td>4.003/5.0</td><td>3.789/5.0</td><td>3.947/5.0</td></tr></table>

Table 6. User study questionnaire results. 1 = Poor, 5 = Excellent.

![](images/dcc321cd9469e1b7362f07eeb380ad85db4a69eed6f312959fa6fc98c7394e20.jpg)  
Figure 10. Failure of Wan-Animate 2. State-of-the-art 2D-only animation models fail to preserve spatial and temporal consistency. Note how the back texture and nail colors change over time (temporal inconsistency), and how facial wrinkles, clothing wrinkles, and nail shapes change across different camera views (spatial inconsistency). All results shown above are from the official Wan-Animate 2 results.

## K. Ethical Discussions

Our method enables the generation of vivid, temporally consistent surface dynamics for animatable static human avatars, while supporting a wide range of input modalities. By explicitly modeling fine-grained dynamic textures, it enhances realism, preserves subtle surface details such as wrinkles and cloth deformation, and allows for expressive and controllable avatar animation. These capabilities open up promising applications in areas such as digital communication, virtual reality, telepresence, entertainment, accessibility technologies, and personalized digital humans. In particular, the method can facilitate more natural remote interaction, immersive storytelling, and inclusive human–computer interfaces by bridging the gap between static identity capture and dynamic expressiveness.

![](images/6d4e9a9b69935f538bbfbfa0bc943beeef50f3be3c5ade837b554c857254f21d.jpg)  
Reference Image  
Results of Wan-Animate 2

Figure 11. Additional Failure of Wan-Animate 2. Given a DynaHuman subject, Wan-Animate 2 exhibits several limitations, including incorrect motion embedding, pose-to-appearance ambiguity, inconsistent lighting, distorted body shape, and background artifacts.  
Table 7. Quantitative comparison on DynaHuman. Here, we compare Wan-Animate 2 [77] with our method. Our method achieves consistent better results.
<table><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>Subject71</td><td rowspan=1 colspan=1>Subject76</td></tr><tr><td rowspan=1 colspan=1>FID    FVD</td><td rowspan=1 colspan=1>FID   FVD</td></tr><tr><td rowspan=2 colspan=1>Wan-Animate 2LHM+Ours</td><td rowspan=1 colspan=1>31.83  185.09</td><td rowspan=1 colspan=1>28.20  179.98</td></tr><tr><td rowspan=1 colspan=1>28.20  72.66</td><td rowspan=1 colspan=1>19.65  40.78</td></tr></table>

As with many powerful generative technologies, the method could be misused. For instance, it may be employed to create fabricated or misleading videos in which an individual’s identity is reconstructed from publicly available social media images and subsequently animated according to artificially designed scenarios. Such misuse raises concerns regarding consent and authenticity.

To mitigate these risks, responsible deployment is essential. Appropriate safeguards should therefore be considered prior to practical adoption. These may, for example, include robust identity verification mechanisms, explicit consent requirements for identity reconstruction, or watermarking. By integrating such technical protections along with ethical guidelines and governance measures, the technology can be directed toward socially beneficial use while minimizing potential harm.

# <sub>eep</sub>C<sup>a</sup> MOTMMI <sub>D</sub>D<sup>C</sup> <sub>eta</sub>C<sup>ap</sup>

Figure 12. Qualitative Comparisons on DeepCap, DDC and MetaCap. We show additional qualitative comparisons on public datasets, i.e. DeepCap [22], DDC [23] and MetaCap [69]. The results are shown in the order of ground truth , MV (static avatar) and our method . Our method successfully generates pose-dependent wrinkles across different motions and clothing types.

AAARRAT大TTTT AAANRA方T方方TT AAAAAA大T大大IT

![](images/5fe69324f36b3b69b602285f9c5589b3f8382c76a958f2e283c29df72d89df37.jpg)  
Figure 13. Qualitative Comparisons on In-the-wild Images. The leftmost image is the input to LHM [59]. The results are shown in the order of LHM (static avatar) , and our method . We drive them with the same body motion. Notice that how the wrinkles evolve with the motion and vary across different clothing types.