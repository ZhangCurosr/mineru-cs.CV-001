# GS-Voxel: Fitting-Free Structured Latents for Large-Scale 3DGS Generation

Ming Qian<sup>1</sup> Zijian Wang<sup>1</sup> Minchao Sun<sup>1</sup> Jincheng Xiong<sup>1,2</sup> Hang Zhang<sup>1</sup> Mu Xu<sup>1</sup> Chi Wang<sup>2</sup> Baoquan Chen<sup>3</sup> <sup>1</sup>Amap, Alibaba <sup>2</sup>Zhejiang University <sup>3</sup>Peking University

![](images/9365bfc0b5a6734973947aa537f97d4aa23a2fc730a13d9706b00940674db496.jpg)  
Figure 1. Given a large-footprint satellite-view conditioning image (1,400 m × 800 m, top-left), our GS-Voxel-based pipeline generates a large-area 3DGS scene. The five crops highlight local appearance and geometric detail.

Many scalable latent 3D generators operate on structured tensors, whereas pre-optimized 3D Gaussian Splatting (3DGS) reconstructions are unordered, spatially irregular, and vary widely in primitive count. We present GS-Voxel, a fitting-free structured latentframework, and evaluate itfor large-scale aerial 3D Gaussian scene generation. GS-Voxel deterministically converts a compatible pre-optimized 3DGS reconstruction into sparse active voxels without additional per-scene optimization, retaining the sub-voxel positions and rendering attributes ofthe selected primitives. A GS-specific factorized VAE then separately encodes voxel geometry and

## Abstract

local Gaussian attributes into sparse 3D latents whose size grows with the number of occupied voxels rather than being limited by afixed scene-wide primitive count. We train image-conditionedflow models in the GS-Voxel latent space to generate aerial 3DGS scenes. A key application enabled by GS-Voxel is large-area scene generation: overlap-aware tiled inference extends synthesis beyond a single training crop conditioned on satellite-view images. Our results show that GS-Voxel provides structured latentsfor pre-optimized aerial 3DGS reconstructions, with latent capacity that grows with the number ofoccupied voxels.

## 1. Introduction

Latent 3D generative models have advanced rapidly at the object level by separating representation learning from generative modeling [10, 35, 36, 43]. A compact structured latent can be modeled efficiently with diffusion or flow-based architectures and decoded into detailed 3D content. Extending this paradigm to real-world outdoor scenes, however, requires a representation that simultaneously preserves photorealistic appearance, accommodates large spatial extents, and remains tractable for generative learning.

We study this representation problem using large-scale aerial 3D Gaussian Splatting (3DGS) [9] as the target domain. Such reconstructions capture detailed appearance and non-mesh-friendly content such as vegetation without requiring explicit topology. Yet a pre-optimized 3DGS reconstruction is an unordered, spatially irregular set whose cardinality varies across scenes. Raw 200 m × 200 m aerial 3DGS tiles can contain more than 3.0 million primitives, far beyond the global primitive budgets common in objectcentric generation. Native 3DGS therefore does not directly provide the structured spatial tensor expected by the sparse latent architectures used in this work.

Existing Gaussian generation methods address irregularity in different ways. GaussianCube [41] refits a fixed number of Gaussians and globally rearranges them with Optimal Transport, while L3DG [28] learns latent diffusion primarily for object- and room-scale Gaussian scenes. Can3Tok [4] tokenizes a globally fixed-cardinality scenelevel 3DGS set into a fixed-size latent, whereas TripoSplat [39] learns an adaptive density distribution and demonstrates variable-budget object decoding from 33K to 262K Gaussians. Together, these works cover globally rearranged Gaussian sets, fixed-cardinality scene latents, and adaptive object-level outputs. However, they do not directly convert an already optimized, million-scale 3DGS reconstruction into a spatial latent without global refitting or primitive rearrangement.

To address this representation bottleneck, we present GS-Voxel, a fitting-free structured latent framework designed and evaluated for large-scale aerial 3D Gaussian scene generation. Here, fitting-free refers specifically to per-scene conversion: given a pre-optimized 3DGS reconstruction in the supported SH0 attribute format, constructing GS-Voxel and encoding its latent require no additional optimization of that scene, no fitting to a fixed-cardinality scene-wide Gaussian template, and no global Optimal Transport rearrangement of the input primitives. The conversion is deterministic, while the factorized VAE and image-conditioned flow models are trained once across the dataset. GS-Voxel voxelizes the input set, retains the selected primitives’ sub-voxel positions and rendering attributes, and imposes a fixed slot budget only within each active voxel. The factorized VAE separately encodes voxel geometry and local Gaussian attributes into structured sparse 3D latents. We evaluate GS-Voxel by training image-conditioned flow models in its latent space to generate aerial 3DGS scenes. We further demonstrate large-area synthesis through overlap-aware tiled inference.

## 2. Related Work

## 2.1. 3D Object Generation

Existing 3D object generation methods typically lift 2D priors into 3D via score distillation or view-conditioned image synthesis [15, 16, 22, 32]. Subsequent feed-forward and latent 3D frameworks improve efficiency [6, 12, 30, 33, 42], but they remain largely bounded by object-centric representations. More recently, latent 3D generative models [5, 31, 34– 36, 43] decouple 3D compression from generative modeling to enable high-quality object-scale synthesis. Our work follows this paradigm but develops and evaluates a structured latent representation for pre-optimized 3DGS at the scale of real-world aerial reconstructions, where the globally fixed primitive budgets common in object-level models become restrictive.

## 2.2. 3D Outdoor Scene Generation

Existing outdoor scene creation systems synthesize training data, construct controllable urban assets, or reconstruct scenes from aerial observations [3, 13, 19, 20, 23, 25, 26, 29, 37]. Sat2City [7] learns cascaded sparse-voxel latent diffusion from synthetic city assets, while Sat2City v2 [8] adapts a pretrained native structured latent model to real satellite–mesh pairs and generates textured mesh assets. Their goals and output representations differ from ours: few learn a generative model directly over pre-optimized real-world 3DGS reconstructions. XCube [27] and Earth-Crafter [17] advance learned 3D generation at larger spatial scales, with EarthCrafter expanding scenes through semanticconditioned sliding-window inference. ABot-Earth 0.5 [24] is an engineering-oriented project for satellite-conditioned 3DGS generation, emphasizing system integration, multi-LOD output, and planetary-scale deployment. Our focus is instead GS-Voxel itself: we describe how it converts preoptimized 3DGS reconstructions into sparse spatial latents without per-scene fitting and how these latents are used for image-conditioned generation.

## 2.3. Generative Modeling of 3D Gaussians

Recent work has begun to explore diffusion-based generation for 3DGS. Early approaches mainly adopt view-conditioned or view-aligned Gaussian generation. For example, DiffusionGS [2] relies on multi-view conditioning, while teacherguided approaches [21] distill priors from 2D generative models. DiffSplat [14] and DiffGS [44] further introduce image-diffusion or functional parameterizations for Gaussian generation.

Table 1. Taxonomic comparison of representative outdoor 3D scene creation systems.
<table><tr><td>Method</td><td>Training Data</td><td>Condition</td><td>Beyond Buildings</td><td>Large-Scale</td><td>Core Paradigm</td></tr><tr><td>Sat2Scene [13]</td><td>Images &amp; Height Map</td><td>Layout</td><td>X</td><td>X</td><td>Geo. Coloring</td></tr><tr><td>Sat2City [7]</td><td>Mesh</td><td>Height Map</td><td></td><td></td><td>Building Object Gen.</td></tr><tr><td>Sat2Density++ [25]</td><td>Images</td><td>Single Satellite</td><td></td><td></td><td>Feedforward</td></tr><tr><td>Sat3DGen [26]</td><td>Images</td><td>Single Satellite</td><td></td><td></td><td>Feedforward</td></tr><tr><td>UrbanWorld [29]</td><td>Mesh + UV</td><td>OSM or Layout</td><td></td><td></td><td>Multi-Stage</td></tr><tr><td>SynCity [3]</td><td>N/A (training-free)</td><td>Text Prompt</td><td></td><td></td><td>Multi-Stage</td></tr><tr><td>SkyFali-GS [11]</td><td>Images</td><td>Multi-view Satellite</td><td></td><td></td><td>Iterative Optimization</td></tr><tr><td>Orbit2Ground [40]</td><td>Images</td><td>Multi-view Satellite</td><td></td><td></td><td>Iterative Optimization</td></tr><tr><td>XCube [27]</td><td>Mesh</td><td>LiDAR Scan</td><td></td><td></td><td>3D Generative</td></tr><tr><td>EarthCrafter [17]</td><td>Mesh</td><td>Satellite + Depth</td><td></td><td></td><td>3D Generative</td></tr><tr><td>Ours</td><td>3DGS</td><td>Single Satellite-View Image</td><td></td><td></td><td>3D Generative</td></tr></table>

A particularly relevant line of work structures irregular Gaussian sets before generative modeling. Can3Tok [4] uses cross-attention from input 3DGS primitives to learned canonical queries. Its preprocessing caps each reconstruction at 100K Gaussians and then selects 40K Gaussians for every VAE input; the VAE maps this globally fixed-cardinality set to a 64×64×4 latent and decodes a fixed number of Gaussians. Can3Tok therefore demonstrates scene-level 3DGS latent modeling under a global primitive budget. GS-Voxel instead fixes capacity only within each active voxel, so the available output slots vary with the number of active voxels rather than being normalized to a common scene-wide count. This distinction matters in our evaluated aerial setting, where raw aerial 3DGS tiles can contain more than 3.0 million primitives.

GaussianCube [41] first obtains a fixed number of Gaussians through densification-constrained fitting and then rearranges them into a predefined voxel grid using Optimal Transport. L3DG [28] learns a sparse VQ-VAE over 3D Gaussians and performs latent diffusion for object- and roomscale synthesis.

TripoSplat is particularly relevant because its Density-Sampled Gaussian VAE (DeG-VAE) supports adaptive output budgets [39]. Its encoder does not ingest an optimized 3DGS parameter set; instead, it samples points from an asset surface, projects DINOv3 and FLUX.2 VAE features from multi-view renderings onto those points, and encodes the resulting feature-augmented point sets. The two methods therefore assume different inputs and construct Gaussians differently: TripoSplat learns a new object-level Gaussian distribution from surface and image evidence, whereas GS-Voxel deterministically voxelizes an existing pre-optimized 3DGS set using a fixed local slot budget. TripoSplat reports decoded budgets from 33K to 262K Gaussians; in our evaluated aerial setting, raw 200 m × 200 m aerial 3DGS tiles can contain more than 3.0 million primitives, while the default GS-Voxel decoder provides approximately 1.5 million output slots. The two methods therefore operate at different scales and allocate output capacity differently: GS-Voxel’s total slot capacity grows with the number of active voxels, but unlike TripoSplat, it does not adapt decoded density at inference time.

## 3. Method

Our method centers on GS-Voxel, a sparse voxel-aligned representation constructed deterministically from a compatible pre-optimized 3DGS reconstruction. We develop and evaluate this representation for large-scale aerial scenes. The full representation pipeline combines GS-Voxel construction with a factorized VAE that separately encodes voxel geometry and local Gaussian attributes.

## 3.1. Fitting-Free Structured Latent Pipeline

As illustrated in Fig. 2, the representation pipeline has two components. First, an explicit conversion reorganizes a compatible pre-optimized 3DGS reconstruction into sparse active voxels without additional per-scene optimization, a fixedcardinality scene-wide Gaussian template, or global rearrangement of the input primitives. Second, a two-stage VAE maps voxel geometry and Gaussian attributes to compact sparse 3D latents. The resulting latents remain aligned with the voxel grid and can represent aerial scenes with millions of unevenly distributed primitives.

![](images/6365b2a85debc80a38959b086e4963e3a91f352e6e12580596dc50ea88d28de4.jpg)  
Figure 2. Overview of our structured-latent pipeline. We first convert a compatible pre-optimized 3DGS reconstruction into GS-Voxel, a sparse representation of the retained local Gaussian parameters, and then encode it with a factorized VAE. The Geometry VAE models hierarchical subdivision and occupancy, while the Local Attribute VAE models Gaussian attributes on the decoded support.

## 3.1.1. GS-Voxel Construction from Pre-optimized 3DGS

GS-Voxel converts the variable number of Gaussians in each voxel into a fixed-width feature vector and normalizes every attribute to a common range.

A compatible pre-optimized 3DGS reconstruction is represented as an unordered set of Gaussian primitives

$$
\begin{array} { r } { \mathcal { G } = \{ g _ { i } \} _ { i = 1 } ^ { N } , \qquad g _ { i } = \left( \mathbf { x } _ { i } , \alpha _ { i } , \mathbf { f } _ { i } ^ { d c } , \mathbf { s } _ { i } , \mathbf { q } _ { i } \right) , } \end{array}\tag{1}
$$

where $\mathbf { x } _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ is the primitive center, $\alpha _ { i }$ is the opacity logit, $\mathbf { f } _ { i } ^ { d c } \in \mathbb { R } ^ { 3 }$ is the DC spherical-harmonic color coefficient, $\mathbf { s } _ { i } \in \mathbb { R } ^ { 3 }$ is the log-scale, and $\mathbf { q } _ { i } \in \mathbb { R } ^ { 4 }$ is the rotation quaternion. In the current implementation, a compatible input follows this parameterization, is provided in a known cubic normalization frame, and stores only SH degree 0, i.e., the DC color coefficient. SH0 gives view-independent color, which is suitable for our restricted aerial viewing range and keeps the per-primitive attribute vector small for multi-million-primitive scenes. The voxel-assignment rule itself does not depend on SH degree. Supporting higherorder view-dependent appearance would require adding the corresponding attribute channels and retraining the Local Attribute VAE and attribute flow.

Independently of the SH degree, native 3DGS is an unordered set, and different spatial regions contain different numbers of primitives. Sparse-convolutional latent models instead expect features attached to discrete spatial coordinates with a fixed channel dimension, so native 3DGS cannot be passed to them directly.

We therefore introduce GS-Voxel, a sparse voxel-aligned representation that reorganizes Gaussian primitives into a format compatible with sparse 3D VAEs. Inspired by the practical offline conversion in O-Voxel [36], we adopt a lightweight, fitting-free conversion without re-fitting or optimization. Unlike O-Voxel, which explicitly encodes surface geometry and material fields, GS-Voxel packs the retained primitives’ local parameters directly (details in Fig. 3).

![](images/1bc52a8a1068960546c1d65e3c13b3e3d513bdabda98e1445b9b5ff9d2876e89.jpg)  
Figure 3. Deterministic conversion from a compatible preoptimized 3DGS reconstruction to GS-Voxel.

We place the normalized scene in a cubic frame $[ \mathbf { b } _ { \mathrm { m i n } } , \mathbf { b } _ { \mathrm { m i n } } + R h ] ^ { 3 }$ and discretize it into a voxel grid of resolution $R ^ { 3 }$ with voxel size $h .$ After discarding primitives below the implementation opacity threshold, each remaining

Gaussian is assigned to a voxel according to its center,

$$
\mathbf { v } _ { i } = \mathrm { c l i p } \left( \left\lfloor { \frac { \mathbf { x } _ { i } - \mathbf { b } _ { \operatorname* { m i n } } } { h } } \right\rfloor , 0 , R - 1 \right) ,\tag{2}
$$

where $ { \mathbf { b } } _ { \mathrm { m i n } }$ is the minimum corner of the cubic frame. Let $\mathcal { G } ( v )$ denote the set of retained, above-threshold primitives whose centers fall into voxel $v .$ . Since the number of Gaussians within each voxel is variable, we convert the local unordered set into a fixed-capacity representation by sorting primitives by their opacity logits α and retaining the top- $K _ { \mathrm { i n } }$ entries:

$$
\begin{array} { r } { \boldsymbol { \tilde { \mathcal { G } } } ( \boldsymbol { v } ) = \mathrm { T o p K } _ { \alpha } ( \boldsymbol { \mathcal { G } } ( \boldsymbol { v } ) , \boldsymbol { K } _ { \mathrm { i n } } ) , \qquad n _ { \boldsymbol { v } } = \mathrm { m i n } ( | \boldsymbol { \mathcal { G } } ( \boldsymbol { v } ) | , \boldsymbol { K } _ { \mathrm { i n } } ) . } \end{array}\tag{3}
$$

The remaining slots are zero-padded when $n _ { v } < K _ { \mathrm { i n } }$ . This yields a deterministic local ordering and, more importantly, ensures a consistent channel dimension across active voxels for downstream neural compression and generative modeling. In practice, most occupied voxels contain only a few primitives, so a proper choice of $K _ { \mathrm { i n } }$ preserves local content with controlled capacity while avoiding unnecessarily large feature channels.

For each retained Gaussian, we encode its position relative to the voxel center $\mathbf { m } _ { v }$

$$
\Delta \mathbf { x } _ { v , j } = \frac { \mathbf { x } _ { v , j } - \mathbf { m } _ { v } } { h } ,\tag{4}
$$

which retains sub-voxel offsets before 8-bit quantization. We normalize each retained Gaussian’s local position and rendering attributes to [0, 1] before quantizing them into uint8 buffers:

$$
\begin{array} { r } { \mathbf { o } _ { v , j } = \Delta \mathbf { x } _ { v , j } + 0 . 5 , } \end{array}
$$

$$
\begin{array} { r } { \mathbf { c } _ { v , j } = C _ { 0 } \mathbf { f } _ { v , j } ^ { d c } + 0 . 5 , } \end{array}\tag{5}
$$

$$
p _ { v , j } = \sigma ( \alpha _ { v , j } ) ,\tag{6}
$$

(7)

$$
\mathbf { l } _ { v , j } = \frac { \mathrm { c l i p } ( \mathbf { s } _ { v , j } , s _ { \mathrm { m i n } } , s _ { \mathrm { m a x } } ) - s _ { \mathrm { m i n } } } { s _ { \mathrm { m a x } } - s _ { \mathrm { m i n } } } ,\tag{8}
$$

$$
\mathbf { r } _ { v , j } = \frac { \hat { \mathbf { q } } _ { v , j } + 1 } { 2 } ,\tag{9}
$$

where $C _ { 0 } = 0$ .28209479, $\hat { \mathbf { q } } _ { v , j }$ is the normalized quaternion, and $( s _ { \mathrm { m i n } } , s _ { \mathrm { m a x } } ) = ( - 2 4 , - 4 )$ in our implementation. The final GS-Voxel feature stored at each active voxel is

$$
\phi _ { v } = \mathrm { c o n c a t } [ \mathrm { v e c } ( \mathbf { O } _ { v } ) , \mathrm { v e c } ( \mathbf { C } _ { v } ) , \mathrm { v e c } ( \mathbf { P } _ { v } ) , \mathrm { v e c } ( \mathbf { L } _ { v } ) , \mathrm { v e c } ( \mathbf { R } _ { v } ) ]\tag{10}
$$

Here, $n _ { v }$ is only a construction-time count used for truncation and zero padding: it is not included in $\phi _ { v }$ , supplied to the VAE, or predicted by the decoder. Active voxels are those with at least one retained, above-threshold primitive, and zero-padded slots have zero opacity and do not contribute to rendering.

The conversion from pre-optimized 3DGS to GS-Voxel is explicit and lightweight: it voxelizes the primitives, retains up to $K _ { \mathrm { i n } }$ dominant entries per active voxel, and packs their normalized attributes into a sparse format. The inverse mapping dequantizes the active-voxel features, reconstructs local offsets and Gaussian parameters, and concatenates the resulting primitives over the active support. Complete forward and inverse procedures, default conversion parameters, and serialization details are provided in Sec. A of the supplementary material. Under the tested choices of voxel size and $K _ { \mathrm { i n } }$ , this explicit conversion retains high rendering fidelity, as evaluated in our experiments. GS-Voxel therefore retains the selected local primitives in a sparse voxel grid with the fixed per-voxel feature size required by latent models. We denote its fine active support at resolution $R ^ { 3 }$ by $\scriptstyle { S _ { R } }$

## 3.1.2. Two-Stage GS-Voxel VAE

We implement the GS-Voxel factorized VAE as two components: a Geometry VAE, which models voxel geometry as hierarchical subdivision and occupancy, and a Local Attribute VAE, which models Gaussian primitives only within occupied voxels. This decomposition separates support modeling from attribute reconstruction and avoids allocating attribute capacity to empty space. These two GS-Voxel-specific components are distinct from the Sparse Structure VAE used for the coarse stage of the generation pipeline.

Geometry VAE. The Geometry VAE encodes the fine support $\scriptstyle { S _ { R } }$ into a geometry latent $z _ { g }$ on a coarsened sparse lattice at $( R / 8 ) ^ { 3 }$ . Each active coordinate is augmented with a sinusoidal positional encoding before entering the encoder. The decoder predicts hierarchical child-subdivision signals whose decisions define the reconstructed fine support $\hat { S } _ { R } \mathbf { : }$

$$
\{ \hat { s } _ { l } \} _ { l = 1 } ^ { L } = D _ { g } ( z _ { g } ) .\tag{11}
$$

where $L$ is the number of hierarchy levels and $s _ { l }$ is the target child-occupancy signal at level l. Unlike TRELLIS.2, which also models additional geometric quantities such as dual vertices, our geometry stage is simplified for GS-Voxel and only reconstructs occupied support and its hierarchy. The training objective is

$$
\mathcal { L } _ { \mathrm { g e o m } } = \lambda _ { \mathrm { s u b d i v } } \sum _ { l = 1 } ^ { L } \mathrm { B C E } ( \hat { s } _ { l } , s _ { l } ) + \lambda _ { \mathrm { K L } } \mathcal { L } _ { \mathrm { K L } } .\tag{12}
$$

Local Attribute VAE. The Local Attribute VAE encodes Gaussian attributes defined on the fine GS-Voxel support; it does not learn the support itself. During generation, the attribute flow produces $z _ { a } ,$ and the Local Attribute VAE decoder maps it to Gaussian attributes on the generated $\hat { S } _ { R }$

For an active voxel v, the packed VAE input is

$$
\begin{array} { r l } & { \mathcal { U } ( v ) = \{ \mathbf { u } _ { v , j } \} _ { j = 1 } ^ { K _ { \mathrm { i n } } } , } \\ & { \mathbf { u } _ { v , j } = \left( \mathbf { o } _ { v , j } , \mathbf { c } _ { v , j } , p _ { v , j } , \mathbf { l } _ { v , j } , \mathbf { r } _ { v , j } \right) \in [ 0 , 1 ] ^ { 1 4 } , } \end{array}\tag{13}
$$

where the components are the normalized quantities defined in Eqs. (5)–(9). The encoder maps these fine-grid attributes to a compact latent $z _ { a }$ on a coarsened sparse lattice at $( R / 8 ) ^ { 3 }$ During decoding, rather than predicting a variable number of primitives within each supplied fine-grid voxel, we adopt a fixed per-voxel slot decoder that outputs

$$
\hat { \mathcal { U } } ( v ) = \{ \hat { \mathbf { u } } _ { v , j } \} _ { j = 1 } ^ { K _ { \mathrm { o u t } } } ,\tag{14}
$$

for each voxel on the supplied fine support. We apply the inverse transforms of Eqs. (5)–(9) to reconstruct sub-voxel offsets and standard 3DGS attributes before differentiable rendering. Importantly, $K _ { \mathrm { o u t } }$ need not equal $K _ { \mathrm { i n } } \colon$ supervision is applied through rendered-image consistency rather than explicit count matching, so the decoder may use a different local slot budget as long as the rendered 3DGS matches the target observations.

The attribute loss is

$$
\mathcal { L } _ { \mathrm { a t t r } } = \lambda _ { \mathrm { i m g } } \mathcal { L } _ { \mathrm { i m g } } + \lambda _ { \mathrm { K L } } \mathcal { L } _ { \mathrm { K L } } ,\tag{15}
$$

where ${ \mathcal { L } } _ { \mathrm { i m g } }$ is a rendering loss combining L1, L2, SSIM, and perceptual terms. Therefore, this stage does not reconstruct sparse support or predict an explicit primitive count; it learns fixed per-voxel Gaussian slots on a supplied fine support.

## 3.2. Generation Pipeline

For image-conditioned generation, we use a three-stage flowmatching pipeline based on TRELLIS.2 [36]. The sparsestructure flow models a coarse occupancy latent, while the geometry and attribute flows model the two GS-Voxel-specific latents.

The generation process has three stages. First, the imageconditioned sparse-structure flow predicts a coarse occupancy latent, which the Sparse Structure VAE decoder converts into support at $( R / 8 ) ^ { 3 }$ . Second, conditioned on the input image and this coarse support, the geometry flow predicts $z _ { g } ;$ the Geometry VAE decoder expands it into the hierarchical subdivision and fine active support. Third, conditioned on the input image and generated geometry, the attribute flow predicts $z _ { a } ;$ the Local Attribute VAE decoder emits $K _ { \mathrm { o u t } }$ Gaussian slots per active fine-grid voxel. The resulting parameters form a standard 3DGS primitive set.

## 3.2.1. Large-Area Scene Generation

For large-area synthesis beyond a fixed training crop, we use an overlap-aware tiled-inference procedure tailored to our three-stage latent hierarchy. Existing neighboring tiles provide inpainting constraints, while stage-wise structure, geometry, and attribute generation together with overlap blending maintain continuity across tile boundaries. At inference, each spatial tile is conditioned on the aligned patch of a supplied satellite-view image, which may be real or generated. Implementation details are provided in the supplemental material.

![](images/0b1ddb73f87a1ad513d65c68a3675e6f9056ea7bb68eb621115fef2553156598.jpg)  
Figure 4. Conditional generation pipeline. Given a satellite-view image, a coarse sparse-structure stage predicts scene support; GS-Voxelspecific geometry and attribute stages then generate fine support and Gaussian parameters, yielding a standard 3DGS primitive set.

## 4. Training Data Construction

We construct the dataset from multiple large-scale aerial 3DGS scenes. Each source scene spans hundreds of meters, has uneven terrain, and may contain floating outliers. Preprocessing proceeds in three stages. We first crop each scene into fixed-size windows, normalize each window to a unit cube, remove outliers, and convert the remaining Gaussians into GS-Voxel. We then sample a dense set of virtual cameras to render diverse aerial views. Finally, geometric and VLM-based filters select reliable renders to supervise the Local Attribute VAE. Details of the three stages follow.

Training Tile Generation and Voxel Filtering We perform sliding-window cropping on the large-scale 3DGS for each scene. Specifically, we slide an axis-aligned window of size W = 200 m with a stride $S = 1 5 0 \mathrm { m }$ (50 m overlap) across the large 3DGS, and retain only the Gaussians whose centers fall inside each window footprint. Each cropped window is then normalized to the unit cube $[ - 0 . 5 , 0 . 5 ] ^ { 3 }$ . We apply DBSCAN $( \varepsilon { = } 0 . 0 2 , m _ { \mathrm { m i n } } { = } 5 0 )$ to isolate the dominant scene support from floating outliers and retain its largest cluster. We then use the lower extent of this retained cluster as the vertical reference and translate it so that its lowest point lies at $z = - 0 . 5$ , aligning the scene support with the lower face of the unit cube. This ordering removes spatially disconnected primitives before they can create spurious voxels and prevents bottom outliers from determining the vertical alignment. Finally, following Sec. 3.1.1, the cleaned Gaussians are converted into GS-Voxel for training.

Virtual Camera Setting After normalization, we render a dense set of training views in the normalized coordinate frame. We place a virtual aerial camera rig on a 10×10 grid spanning $[ - 0 . 7 , 0 . 7 ] ^ { 2 }$ in the xy plane. The grid extends slightly beyond the scene footprint so that peripheral cameras can also capture the scene at oblique angles, and is replicated across five height layers with denser sampling and wider FOV near the ground. Here pitch is measured as the deviation from the nadir direction, so $\mathrm { p i t c h { = } } 0 ^ { \circ }$ denotes a top-down view; for non-zero pitch we additionally sweep four compass yaws so that every grid cell is observed from multiple orientations. Camera orientation is set directly from each layer’s pitch and compass yaw rather than by looking at the scene center, keeping view distributions comparable across windows of different content. To break the regularity of the rig and increase view diversity, we apply uniform random perturbations to the xy position (±15% of the grid spacing), height (±0.1), pitch (±5<sup>◦</sup>), and yaw (±30<sup>◦</sup>). The per-layer configuration is summarized in Table 2. The rig produces approximately 3.9K candidate views per window, which are then filtered as described below.

Table 2. Per-layer configuration of the virtual aerial camera rig. Pitch is measured as the deviation from the nadir direction. For $\mathrm { p i t c h { = } } 0 ^ { \circ }$ a single yaw is used (top-down view); for pitch>0<sup>◦</sup> four compass yaws $\{ 0 ^ { \circ } , 9 0 ^ { \circ } , 1 8 0 ^ { \circ } , 2 7 0 ^ { \circ } \}$ are sampled. Views/cell counts cameras per xy grid cell; Views/window assumes a 10×10 grid.
<table><tr><td>Layer</td><td>z</td><td>FOV</td><td>Pitches (deg)</td><td>Views/cell</td><td>Views/window</td></tr><tr><td>0</td><td>0.0</td><td>26°</td><td>{0, 30, 45, 60}</td><td>13</td><td>1,300</td></tr><tr><td>1</td><td>0.3</td><td>18°</td><td>{15, 30, 45}</td><td>12</td><td>1,200</td></tr><tr><td>2</td><td>0.6</td><td>14°</td><td>{0, 15, 30}</td><td>9</td><td>900</td></tr><tr><td>3</td><td>0.9</td><td>14°</td><td>{15}</td><td>4</td><td>400</td></tr><tr><td>4</td><td>1.2</td><td>14°</td><td>{0}</td><td>1</td><td>100</td></tr><tr><td colspan="6"></td></tr></table>

VLM-Based View Filtering For each window, the camera rig generates about 3,900 candidate renders. Occlusions, blind spots, and reconstruction artifacts make many views unreliable supervision (e.g., empty boundary views, oblique views through Gaussian holes, or renders with blur and floaters). We therefore apply two sequential filters: (1) a geometric coverage test using rendered alpha, discarding views with mean accumulated opacity below $\tau _ { \alpha } = 0 . 9 ;$ and (2) VLM-based quality scoring via Qwen-VL-Plus from the Qwen-VL model family [1], which evaluates the remaining renders in [0, 1] for texture and silhouette sharpness, reconstruction artifacts, and whether dark regions correspond to natural boundaries. The exact scoring prompt is provided in the supplementary material. The filtering policy retains roughly 500–1,000 reliable renders per window. These views supervise only the Local Attribute VAE and are separate from the conditioning-view pool used by the flow models.

![](images/f5a0616281b04da79f02a162f35632651f00eb789d0fa2e9b9f7f438de41b933.jpg)  
Figure 5. Examples of retained and rejected aerial renders. Filtering removes unreliable or severely degraded supervision; only the retained views supervise the Local Attribute VAE.

## 5. Experiments

## 5.1. Implementation Details

Following Sec. 4, we obtain approximately 18K samples, each covering a $2 0 0 \times 2 0 0 \mathrm { { m ^ { 2 } } }$ area. We use 800 samples for validation and the rest for training. The split is performed at the source-scene level: no source scene or spatial region is shared between training and validation.

The two GS-Voxel-specific VAEs use three spatial downsampling stages, taking inputs at $R = 2 5 6$ and producing latents at resolution 32<sup>3</sup>. The pretrained Sparse Structure VAE is reused without retraining, while the sparse-structure, geometry, and attribute conditional flows are all trained on our data. For this training, each 3D tile is paired with nine $5 1 2 \times 5 1 2$ near-orthographic bird’s-eye-view renders with slightly perturbed extrinsics. The same nine-view pool is used by all three conditional flows, with one view randomly sampled for each training instance.

We train the Geometry VAE, Local Attribute VAE, and all three conditional flows with AdamW [18], using a learning rate of $1 \times 1 0 ^ { - 4 }$ and weight decay 0.01. For the conditional flows, we use classifier-free conditioning dropout with a rate of 0.1. We use 0.3B-parameter DiTs for the geometry and attribute flows, reduced from the 1.3B architecture in TRELLIS.2 to match our smaller and less diverse dataset.

## 5.2. Representation and VAE Analysis

We evaluate the representation and VAE analyses in this subsection on 10 validation tiles from the source-sceneseparated split. Reported reconstruction metrics are aggregated over rendered views from these tiles.

Choice of $K _ { \mathrm { i n } }$ and R for GS-Voxel Construction. Table 3 evaluates the direct GS-Voxel conversion, before VAE encoding, while varying the per-voxel budget $K _ { \mathrm { i n } }$ and grid resolution R. At fixed $R = 2 5 6$ , increasing $K _ { \mathrm { i n } }$ improves reconstruction because more local primitives are retained. The gain becomes small from $K _ { \mathrm { i n } } = 1 6 ~ \mathrm { t o } ~ 3 2$ , while the per-voxel attribute width doubles from 224 to 448. At fixed

Table 3. Direct GS-Voxel conversion before VAE encoding, evaluated on 10 validation tiles. We report average retained-primitives and active-voxel counts rounded to the nearest thousand (K), together with rendered reconstruction quality. Bold settings denote our default configuration.
<table><tr><td> $R$ </td><td> $K _ { \mathrm { i n } }$ </td><td> $\mathbf { A v } \mathbf { g } .$  Retained Primitives</td><td> $\operatorname { A v g } .$  Active Voxels</td><td>SSIM ↑</td><td>PSNR ↑</td><td>LPIPS ↓</td></tr><tr><td>256</td><td>1</td><td>377K</td><td>377K</td><td>0.55</td><td>18.52</td><td>0.425</td></tr><tr><td>256</td><td>4</td><td>971K</td><td>377K</td><td>0.85</td><td>26.82</td><td>0.166</td></tr><tr><td>256</td><td>8</td><td>1,340K</td><td>377K</td><td>0.95</td><td>33.26</td><td>0.058</td></tr><tr><td>256</td><td>16</td><td>1,602K</td><td>377K</td><td>0.98</td><td>40.04</td><td>0.023</td></tr><tr><td>256</td><td>32</td><td>1,737K</td><td>377K</td><td>0.98</td><td>40.94</td><td>0.021</td></tr><tr><td>512</td><td>1</td><td>980K</td><td>980K</td><td>0.87</td><td>27.38</td><td>0.156</td></tr><tr><td>512</td><td>4</td><td>1,501K</td><td>980K</td><td>0.98</td><td>39.20</td><td>0.025</td></tr></table>

$K _ { \mathrm { i n } } = 4 ,$ , increasing R from 256 to 512 raises PSNR from 26.82 to 39.20, but also increases the average active support from approximately 377K to 980K voxels. The tested $R = 5 1 2$ $K _ { \mathrm { i n } } = 4$ setting remains slightly below the fidelity of $R \ : = \ : 2 5 6 , \ : K _ { \mathrm { i n } } = 1 6$ while using about 2.6× as many active voxels. Because each tile covers approximately $2 0 0 \mathrm { m } \times 2 0 0 \mathrm { m }$ , we adopt $R = 2 5 6$ and $K _ { \mathrm { i n } } = 1 6$ as the tested trade-off between direct-conversion fidelity and sparse-support size.

Choice of $K _ { \mathrm { o u t } }$ for Local Attribute VAE. We ablate the per-voxel output slot budget $K _ { \mathrm { o u t } }$ of the Local Attribute VAE over {1, 4, 8}. As shown in Table 4, $K _ { \mathrm { o u t } } ~ = ~ 1$ restricts local capacity and lowers reconstruction quality. Although $K _ { \mathrm { o u t } } = 8$ provides more slots, it does not improve reconstruction under the current objective. We therefore use $K _ { \mathrm { o u t } } = 4 $ , which gives the best PSNR, SSIM, and LPIPS among the tested settings.

Table 4. Choice of $K _ { \mathrm { o u t } }$ for the Local Attribute VAE, evaluated using rendered-image reconstruction quality.
<table><tr><td>Method / Setting</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td> $K _ { \mathrm { o u t } } = 1$ </td><td>22.81</td><td>0.61</td><td>0.350</td></tr><tr><td> $K _ { \mathrm { o u t } } = 4 ( \mathrm { O u r s } )$ </td><td>23.09</td><td>0.62</td><td>0.331</td></tr><tr><td> $K _ { \mathrm { o u t } } = 8$ </td><td>22.12</td><td>0.58</td><td>0.354</td></tr></table>

Factorized VAE Component Analysis. We compare our factorized VAE with a single-stage baseline that jointly compresses voxel geometry and local Gaussian attributes. The Geometry VAE and Local Attribute VAE are evaluated separately because they model different quantities. As shown in Table 5, the Geometry VAE obtains 0.99 Shape IoU for support reconstruction. The Local Attribute VAE is evaluated on the target support $\scriptstyle { S _ { R } }$ and obtains 23.09 PSNR, 0.62 SSIM, and 0.331 LPIPS.

## 5.3. Tile-Level Generative Results

For context, Tab. 6 lists cross-paper FID and KID values reported by existing methods, whose ground-truth datasets and camera protocols differ. Under our evaluation protocol, the generated and ground-truth sets each contain 15K images, with ground truth rendered from held-out source scenes in the validation split; our method obtains an FID of 28.0 and a KID of 0.020. Qualitative tile-level generation results are shown in Fig. 6.

Table 5. Separate evaluation of the Geometry VAE and Local Attribute VAE. Shape IoU measures support reconstruction; PSNR, SSIM, and LPIPS measure attribute reconstruction on the target support.
<table><tr><td>Model</td><td>Shape IoU ↑</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>Single-stage VAE</td><td>0.76</td><td>21.01</td><td>0.54</td><td>0.418</td></tr><tr><td>Geometry VAE</td><td>0.99</td><td></td><td></td><td></td></tr><tr><td>Local Attribute VAE</td><td>一</td><td>23.09</td><td>0.62</td><td>0.331</td></tr></table>

Table 6. Cross-paper reference for image-level FID/KID. Baselines use different ground-truth sets and camera protocols; values are not directly comparable and no ranking is implied.
<table><tr><td>Method</td><td>FID</td><td>KID</td></tr><tr><td>CityDreamer [37]</td><td>97.3</td><td>0.096</td></tr><tr><td>GaussianCity [38]</td><td>86.9</td><td>0.090</td></tr><tr><td>EarthCrafter [17]</td><td>69.5</td><td>0.061</td></tr><tr><td>Ours</td><td>28.0</td><td>0.020</td></tr></table>

## 5.4. Large-Area Scene Generation

As shown in Figs. 1, 7 and 8, we demonstrate large-area synthesis using constructed satellite-view conditions whose source scenes are geographically disjoint from the training data. We assemble these test conditions from rendered aerial references by spatially extending local content at a fixed ground sampling resolution. Overlap-aware tiled inference generates scene latents across the full footprint and decodes them into standard 3DGS primitives. The examples span up to 1,400 m×800 m, substantially beyond the 200 m×200 m training crop.

## 6. Conclusion

We presented GS-Voxel, a fitting-free structured latent representation, and instantiated it for large-scale aerial 3D Gaussian scene generation. Its deterministic construction organizes selected primitives from a compatible pre-optimized 3DGS reconstruction into sparse voxels, while its factorized VAE encodes occupied support separately from local Gaussian attributes. In our aerial experiments, this design handles millions of input primitives without fitting each scene to a fixed-size, scene-wide Gaussian template. We use image-conditioned flow models to generate sparse-structure, geometry, and attribute latents that decode into aerial 3DGS scenes, while overlap-aware tiled inference extends synthesis beyond a single training crop. GS-Voxel may therefore provide a basis for future work on scalable aerial 3D generation and downstream simulation.

## 7. Limitations and Future Work

GS-Voxel is a representation and factorized VAE framework for compatible pre-optimized 3DGS reconstructions; the present implementation and experiments are developed for SH0 aerial scenes.

Because training relies on 3DGS scenes reconstructed from real-world data, the approach is constrained by the scale and availability of suitable training scenes. The current image-conditioned flow models are trained with rendered bird’s-eye-view images. At inference, the conditional pipeline can accept real or generated satellite-view images, but robustness to sensor variation and domain shifts among rendered, real, and generated conditions has not yet been evaluated. Extending training and evaluation across these input domains is important future work.

GS-Voxel’s top- $K _ { \mathrm { i n } }$ local capacity can discard primitives in extremely dense voxels, while finite spatial resolution can make very thin structures harder to preserve and reconstruct. Adaptive resolution and learned local-capacity allocation are promising directions for addressing these distinct limitations.

The present study evaluates reconstruction fidelity and image-conditioned generation rather than coding efficiency, and therefore does not yet report entropy-coded bitrates or rate–distortion curves. A systematic evaluation of memory and storage reduction, together with quantized or entropycoded GS-Voxel latents, is planned for a future revision.

Beyond aerial scenes, future work will test the same fitting-free voxel-local representation on indoor scenes, street-level scenes, and object-level assets. These settings introduce different camera distributions, spatial scales, and appearance characteristics, and may require adapting the current SH0 attributes, aerial-scene normalization policy, and training data.

Finally, future work should evaluate whether generated aerial scenes are sufficiently faithful and controllable for downstream uses such as planning, simulation, or emergency-response analysis.

## References

[1] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023.

[2] Yuanhao Cai, He Zhang, Kai Zhang, Yixun Liang, Mengwe Ren, Fujun Luan, Qing Liu, Soo Ye Kim, Jianming Zhang, Zhifei Zhang, Yuqian Zhou, Yulun Zhang, Xiaokang Yang, Zhe Lin, and Alan Yuille. Baking gaussian splatting into diffusion denoiser for fast and scalable single-stage imageto-3d generation and reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 25062–25072, 2025.

[3] Paul Engstler, Aleksandar Shtedritski, Iro Laina, Christian Rupprecht, and Andrea Vedaldi. Syncity: Training-free generation of 3d worlds. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 27585–27595, 2025.

[4] Quankai Gao, Iliyan Georgiev, Tuanfeng Y. Wang, Krishna Kumar Singh, Ulrich Neumann, and Jae Shin Yoon. Can3tok: Canonical 3d tokenization and latent modeling of scene-level 3d gaussians. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9320–9331, 2025.

[5] Fangzhou Hong, Jiaxiang Tang, Ziang Cao, Min Shi, Tong Wu, Zhaoxi Chen, Shuai Yang, Tengfei Wang, Liang Pan, Dahua Lin, et al. 3dtopia: Large text-to-3d generation model with hybrid diffusion priors. arXiv preprint arXiv:2403.02234, 2024.

[6] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. In International Conference on Learning Representations, 2024.

[7] Tongyan Hua, Lutao Jiang, Ying-Cong Chen, and Wufan Zhao. Sat2city: 3d city generation from a single satellite image with cascaded latent diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 27978–27988, 2025.

[8] Tongyan Hua, Dongli Wu, Jinjing Zhu, Yinrui Ren, Zhongcheng Hong, Ying-Cong Chen, Hui Xiong, and Wufan Zhao. Sat2City v2: Native 3D city asset generation from a single satellite image, 2026.

[9] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4), 2023.

[10] Zeqiang Lai, Yunfei Zhao, Haolin Liu, Zibo Zhao, Qingxiang Lin, Huiwen Shi, Xianghui Yang, Mingxin Yang, Shuhui Yang, Yifei Feng, Sheng Zhang, Xin Huang, Di Luo, Fan Yang, Fang Yang, Lifu Wang, Sicong Liu, Yixuan Tang, Yulin Cai, Zebin He, Tian Liu, Yuhong Liu, Jie Jiang, Linus, Jingwei Huang, and Chunchao Guo. Hunyuan3d 2.5: Towards high-fidelity 3d assets generation with ultimate details, 2025.

[11] Jie-Ying Lee, Yi-Ruei Liu, Shr-Ruei Tsai, Wei-Cheng Chang, Chung-Ho Wu, Jiewen Chan, Zhenjun Zhao, Chieh Hubert Lin, and Yu-Lun Liu. Skyfall-gs: Synthesizing immersive 3d urban scenes from satellite imagery, 2025.

[12] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. In International conference on learning representations, 2024.

[13] Zuoyue Li, Zhenqiang Li, Zhaopeng Cui, Marc Pollefeys, and Martin R. Oswald. Sat2scene: 3d urban scene generation from satellite images with diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7141–7150, 2024.

[14] Chenguo Lin, Panwang Pan, Bangbang Yang, Zeming Li, and Yadong MU. Diffsplat: Repurposing image diffusion models for scalable gaussian splat generation. In The Thirteenth International Conference on Learning Representations, 2025.

[15] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution textto-3d content creation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 300–309, 2023.

[16] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9298– 9309, 2023.

[17] Shang Liu, Chenjie Cao, Chaohui Yu, Wen Qian, Jing Wang, and Fan Wang. Earthcrafter: Scalable 3d earth generation via dual-sparse latent diffusion. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7260–7268, 2026.

[18] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

[19] Pascal Muller, Peter Wonka, Simon Haegler, Andreas Ulmer,¨ and Luc Van Gool. Procedural modeling of buildings. ACM Trans. Graph., 25(3):614–623, 2006.

[20] Yoav I. H. Parish and Pascal Muller. Procedural modeling¨ of cities. In Proceedings ofthe 28th Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH 2001, Los Angeles, California, USA, August 12-17, 2001, pages 301–308. ACM, 2001.

[21] Chensheng Peng, Ido Sobol, Masayoshi Tomizuka, Kurt Keutzer, Chenfeng Xu, and Or Litany. A lesson in splats: Teacher-guided diffusion for 3d gaussian splats generation with 2d supervision. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 28707–28717, 2025.

[22] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.

[23] Ming Qian, Jincheng Xiong, Gui-Song Xia, and Nan Xue. Sat2density: Faithful density learning from satellite-ground image pairs. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 3683–3692, 2023.

[24] Ming Qian, Tianjian Ouyang, Mingchao Sun, Zijian Wang, Jincheng Xiong, Jiarong Han, Yongchang Zhang, Jiawei Zhang, Xu Wang, Yu Liu, Luyang Tang, Fei Yu, Zengye Ge, Mengmeng Du, Yuan Liu, Nianfei Fan, Song Wang, Yingliang Peng, Chunxue Jia, Yang Liu, Shiying Zeng, Haozhe Shi, Junnan Lai, Hongyu Pan, Zheng Wu, Ning Guo, Mu Xu, and Hang Zhang. ABot-Earth 0.5: Generative 3D Earth model, 2026. arXiv:2606.09967.

[25] Ming Qian, Bin Tan, Qiuyu Wang, Xianwei Zheng, Hanjiang Xiong, Gui-Song Xia, Yujun Shen, and Nan Xue. Seeing through satellite images at street views. IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(5):5692– 5709, 2026.

[26] Ming Qian, Zimin Xia, Changkun Liu, Shuailei Ma, Wen Wang, Zeran Ke, Bin Tan, Hang Zhang, and Gui-Song Xia. Sat3DGen: Comprehensive street-level 3d scene generation from single satellite image. In The Fourteenth International Conference on Learning Representations, 2026.

[27] Xuanchi Ren, Jiahui Huang, Xiaohui Zeng, Ken Museth, Sanja Fidler, and Francis Williams. Xcube: Large-scale 3d generative modeling using sparse voxel hierarchies. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 4209–4219, 2024.

[28] Barbara Roessle, Norman Muller, Lorenzo Porzi, Samuel¨ Rota Bulo, Peter Kontschieder, Angela Dai, and Matthias\` Nießner. L3DG: Latent 3d gaussian diffusion, 2024. SIG-GRAPH Asia 2024.

[29] Yu Shang, Yuming Lin, Yu Zheng, Hangyu Fan, Jingtao Ding, Jie Feng, Jiansheng Chen, Li Tian, and Yong Li. Urbanworld: An urban world model for 3d city generation. arXiv preprint arXiv:2407.11965, 2024.

[30] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In European Conference on Computer Vision, pages 1–18. Springer, 2024.

[31] Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, et al. Rodin: A generative model for sculpting 3d digital avatars using diffusion. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 4563–4573, 2023.

[32] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in neural information processing systems, 36: 8406–8441, 2023.

[33] Xinyue Wei, Kai Zhang, Sai Bi, Hao Tan, Fujun Luan, Valentin Deschaintre, Kalyan Sunkavalli, Hao Su, and Zexiang Xu. Meshlrm: Large reconstruction model for highquality meshes. arXiv preprint arXiv:2404.12385, 2024.

[34] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer. Advances in Neural Information Processing Systems, 37:121859–121881, 2024.

[35] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 21469–21480, 2025.

[36] Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, and Jiaolong Yang. Native and compact structured latents for 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14419–14429, 2026.

[37] Haozhe Xie, Zhaoxi Chen, Fangzhou Hong, and Ziwei Liu. Citydreamer: Compositional generative model of unbounded 3d cities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9666–9675, 2024.

[38] Haozhe Xie, Zhaoxi Chen, Fangzhou Hong, and Ziwei Liu. Generative gaussian splatting for unbounded 3d city generation. In Proceedings of the IEEE/CVF Conference on

Computer Vision and Pattern Recognition (CVPR), pages 6111–6120, 2025.

[39] Runjie Yan, Yan-Pei Cao, Peng Wang, Ding Liang, and Yuan-Chen Guo. Generative 3d gaussians with learned density control, 2026. TripoSplat; SIGGRAPH Conference Papers 2026.

[40] Fei Yu, Yu Liu, Luyang Tang, Mingchao Sun, Zengye Ge, Rui Bu, Yuchao Jin, Haisen Zhao, He Sun, Yangyan Li, Mu Xu, Wenzheng Chen, and Baoquan Chen. From orbit to ground: Generative city photogrammetry from extreme offnadir satellite images, 2026.

[41] Bowen Zhang, Yiji Cheng, Jiaolong Yang, Chunyu Wang, Feng Zhao, Yansong Tang, Dong Chen, and Baining Guo. Gaussiancube: A structured and explicit radiance representation for 3d generative modeling. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[42] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting. In European Conference on Computer Vision, pages 1–19. Springer, 2024.

[43] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating highquality 3d assets. ACM Transactions on Graphics (TOG), 43 (4):1–20, 2024.

[44] Junsheng Zhou, Weiqi Zhang, and Yu-Shen Liu. DiffGS: Functional gaussian splatting diffusion. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

## A. GS-Voxel Conversion Procedures

The following pseudocode gives the complete deterministic conversion used to construct GS-Voxel and recover a standard 3DGS primitive set.

Input normalization and coordinate recovery. The conversion assumes that a selected cubic 3DGS crop has been placed in the normalized frame; it does not estimate this frame from the primitive bounds. Let $\mathbf { c } \in \mathbb { R } ^ { 3 }$ and $L > 0$ be the center and side length of the crop in the original coordinate system. Before Algorithm S1, we transform the Gaussian centers and log-scales as

$$
\mathbf { x } _ { i } = \frac { \mathbf { x } _ { i } ^ { \mathrm { { w o r l d } } } - \mathbf { c } } { L } , \qquad \mathbf { s } _ { i } = \mathbf { s } _ { i } ^ { \mathrm { { w o r l d } } } - \log L ,\tag{16}
$$

where subtraction of log L is applied to all three log-scale components. This maps the crop to $[ - 0 . 5 , 0 . 5 ] ^ { 3 }$ while preserving each Gaussian’s size relative to the scene. Opacity logits, SH coefficients, and rotations are unchanged. If recovery of the original coordinate system is required, preprocessing must retain $( \mathbf { c } , L )$ ; these values are not stored in the .vxz file. Algorithm S2 returns Gaussians in the normalized frame; world-space centers and log-scales are recovered by

$$
\begin{array} { r } { { \bf x } _ { i } ^ { \mathrm { w o r l d } } = L { \bf x } _ { i } + { \bf c } , \qquad { \bf s } _ { i } ^ { \mathrm { w o r l d } } = { \bf s } _ { i } + \log L . } \end{array}\tag{17}
$$

Default configuration. For all experiments, the input is an SH0 3DGS tile normalized to $[ - 0 . 5 , 0 . 5 ] ^ { 3 }$ . We use $R = 2 5 6 ,$ voxel size $h \ = \ 1 / 2 5 6 .$ , local input capacity $K _ { \mathrm { i n } } ~ = ~ 1 6 ,$ opacity threshold $t _ { 8 } ~ = ~ 5 ~ ( \mathrm { i . e . , ~ 5 / 2 5 5 }$ after sigmoid), 8- bit attribute quantization, log-scale range $[ s _ { \mathrm { m i n } } , s _ { \mathrm { m a x } } ] =$ $[ - 2 4 , - 4 ]$ , and an out-of-bounds tolerance $\delta _ { b } = 2 / 1 2 8$ . Centers lying within $\delta _ { b }$ outside the normalized cube are clamped to boundary voxels; a larger deviation is treated as an invalid normalization. No scale-based pruning is used. The choices $R = 2 5 6$ and $K _ { \mathrm { i n } } = 1 6$ follow the fidelity–support trade-off evaluated in Table ${ } ^ { 3 ; }$ the remaining conversion settings are fixed across all experiments. A PLY input uses the standard fields $\mathbf { x } / \mathbf { y } / \mathbf { z } ,$ , f dc 0/1/2, opacity, scale $. 0 / 1 / 2$ and $\mathtt { r o t \_ 0 / 1 / 2 / 3 }$ , where opacity and scale are stored as logits and log-scales, respectively.

Serialization. We serialize GS-Voxel as a .vxz file using the open-source o voxel.io interface released with TRELLIS.2, available at https://github.com/ microsoft/TRELLIS.2/tree/main/o-voxel. O-Voxel is used only as the sparse serialization layer; the stored Gaussian feature schema is specific to GS-Voxel. For M active voxels, the file contains coordinates of shape $M \times 3$ and uint8 buffers offset, rgb, opacity, scale, and rotation with respective shapes $M \times 3 K _ { \mathrm { i n } } , M \times 3 K _ { \mathrm { i n } }$

$M \times K _ { \mathrm { i n } } , M \times 3 K _ { \mathrm { i n } } ,$ , and $M \times 4 K _ { \mathrm { i n } }$ . Our conversion utility also stores an auxiliary uint8 count buffer of shape $M \times 1$ to identify the retained slots. This buffer is not part of $\phi _ { v }$ and is not supplied to the VAE; without it, valid slots are equivalently identified by quantized opacity greater than or equal to $t _ { 8 } ,$ , because padded slots have zero opacity. All stored GS-Voxel data are contained in this single .vxz file.

Let

$$
Q _ { 8 } ( y ) = \left\lfloor 2 5 5 \mathrm { c l i p } ( y , 0 , 1 ) \right\rfloor , \qquad D _ { 8 } ( z ) = z / 2 5 5\tag{18}
$$

denote 8-bit quantization and dequantization. The opacity threshold is represented by an integer $t _ { 8 } \in [ 0 , 2 5 5 ]$ ; we use $t _ { 8 } = 5$

## Algorithm S1: 3DGS → GS-Voxel

Input: normalized SH0 Gaussian set $\mathcal { G } ;$ cubic frame $( \mathbf { b } _ { \operatorname* { m i n } } , R , h )$ ; local capacity $K _ { \mathrm { i n } } ;$ opacity threshold $t _ { 8 } ;$ log-scale range $[ s _ { \mathrm { m i n } } , s _ { \mathrm { m a x } } ]$

Output: sparse active coordinates and packed features $\mathcal { V } = \{ ( v , \phi _ { v } ) \}$

1. Verify the normalized spatial range. Clamp centers within $\delta _ { b }$ of the cube boundary and stop if any center exceeds this tolerance.

2. Compute $p _ { i } ~ = ~ \sigma ( \alpha _ { i } )$ and discard every $g _ { i }$ with $p _ { i } \leq t _ { 8 } / 2 5 5$

3. Assign each remaining primitive to $\mathbf { v } _ { i }$ with Eq. (2), and compute the corresponding voxel center.

4. Compute the normalized offset, color, opacity, scale, and rotation $\left( \mathbf { o } _ { i } , \mathbf { c } _ { i } , p _ { i } , \mathbf { l } _ { i } , \mathbf { r } _ { i } \right)$ using Eqs. (5)–(9).

5. Group primitives by voxel coordinate. For every nonempty voxel v, sort its group by decreasing opacity and retain the first $K _ { \mathrm { i n } }$ primitives.

6. Apply $Q _ { 8 }$ independently to every normalized attribute of each retained primitive.

7. Place the quantized attributes in opacity order, zeropad unused local slots to $K _ { \mathrm { i n } }$ , and concatenate the buffers into $\phi _ { v }$ as in Eq. (10).

8. Return all active coordinates and their packed features.

## Algorithm S2: GS-Voxel → 3DGS

Input: sparse GS-Voxel $\nu ;$ cubic frame $( \mathbf { b } _ { \operatorname* { m i n } } , R , h ) ;$ opacity threshold $t _ { 8 } ;$ log-scale range $\left[ s _ { \operatorname* { m i n } } , s _ { \operatorname* { m a x } } \right]$ Output: reconstructed Gaussian set $\hat { \mathcal { G } }$

1. Initialize $\hat { \mathcal { G } } \gets \mathcal { D } .$

2. For each active voxel $( v , \phi _ { v } )$ , unpack $\phi _ { v }$ into its $K _ { \mathrm { i n } }$ quantized local slots and discard slots whose quantized opacity is below $t _ { 8 }$

3. For every remaining slot, apply $D _ { 8 }$ to obtain $( \mathbf { o } , \mathbf { c } , p , \mathbf { l } , \mathbf { r } )$ , set $\bar { p } = \mathrm { c l i p } ( p , 1 0 ^ { - 4 } , 1 - 1 0 ^ { - 4 } )$ , and

## reconstruct

$$
\mathbf { x } = \mathbf { m } _ { v } + h ( \mathbf { o } - 0 . 5 ) ,
$$

$$
\mathbf { f } ^ { d c } = ( \mathbf { c } - 0 . 5 ) / C _ { 0 } ,
$$

$$
\alpha = \log ( \bar { p } / ( 1 - \bar { p } ) ) ,
$$

$$
\mathbf { s } = s _ { \mathrm { m i n } } + ( s _ { \mathrm { m a x } } - s _ { \mathrm { m i n } } ) \mathbf { l } ,
$$

$$
\mathbf { q } = \mathrm { n o r m a l i z e } ( 2 \mathbf { r } - 1 ) .
$$

4. Append $( \mathbf { x } , \alpha , \mathbf { f } ^ { d c } , \mathbf { s } , \mathbf { q } )$ to $\hat { \mathcal G }$ and return the concatenated set after all active voxels are processed.

## B. Details of Overlap-Aware Tiled Inference

Overview We generate scenes beyond one training crop using overlap-aware tiled inference. The procedure builds a large scene tile-by-tile with overlapping boundaries. For each spatial tile, the aligned patch from the supplied largefootprint satellite-view image provides conditioning at the same ground sampling resolution.

Tile Traversal and Overlap Tiles are arranged on the horizontal plane and visited in a BFS-spiral order starting from the center. Each tile spans $3 2 ^ { 3 }$ cells in the coarse sparsestructure grid, corresponding to a $2 5 6 ^ { 3 }$ fine GS-Voxel grid. Neighboring tiles overlap by eight coarse-grid cells, giving a coarse-grid stride of 24.

Overlap Handling For tiles beyond the first, overlap regions are generated via repaint-style inpainting. Stored overlap latents from previously generated neighbors are noised to the current sampling time and used as constraints, while non-overlap regions are sampled freely under the boundary context.

Seam Reduction We use a three-stage schedule consistent with the main pipeline: coarse sparse structure is propagated first, geometry then predicts the fine active support, and attribute latents are finally generated and decoded into Gaussian parameters on that support. Shared noise in overlaps and feathered accumulation are used to reduce visible seams.

Post-processing We fill small holes in the coarse sparsestructure grid by nearest-donor copying and enforce the required subdivisions before geometry generation.

Settings We use 50 Euler steps per flow-matching stage with classifier-free guidance strength 3.0. The feathering validity offset is set to two coarse-grid cells.

## C. Prompt for filtering

```markdown
# Role
You are an extremely strict visual quality inspector for 3D Gaussian Splatting
renders.
Be harsh on blurriness, but distinguish between "Scene Boundaries" and "Reconstruction
Failures".
# Key Definitions
- Artifacts (fatal): blurry/smoky blobs, ghosting floaters, or "torn paper" holes that
appear inside or over objects.
- Clean boundaries (acceptable): pure black regions at the edges or corners where the
3D model ends.
If the transition from the scene to the black area is sharp and clean, it is NOT a
failure.
# Guiding Principle
- Default for ˜100% complete but soft images: ˜0.60.
- Top scores (>0.85): ONLY for crystal clear, sharp textures.
- The "Boundary" Rule: A sharp, clean black corner/edge should only receive a minor
penalty (e.g., -0.1),
NOT a fatal ˜0.20 score.
# Visual Checklist & Scoring (use the first matching description)
--- [CRITICAL FAILURE: Bad Reconstruction] ---
- Score 0.00 IF: a giant, blurry blob or "smoky smoke" obscures more than 30% of the
important scene content.
- Score 0.25 IF: there are Internal Fatal Flaws:
A) A "torn paper" hole inside a building or road.
B) Messy, blurry "floaters" or smoky artifacts that overlap with main structures.
--- [PASS: Sharp Content, Possible Clean Boundaries] ---
- Score 0.55 IF: the image is complete (or has clean edges), but appears generally
soft, hazy, or has geometric distortions.
Details (windows, lines) are blurry.
- Score 0.70 IF: the image is structurally sound and sharp, but has a significant
clean black boundary at the edges.
The transition to black must be sharp.
- Score 0.80 IF: the image is very sharp and clear, with only a tiny, clean black
sliver at a corner.
- Score 0.90+ IF: crystal clear across the entire view with 100% scene coverage and NO
flaws.
# Output Instruction
Return ONLY valid JSON with this exact schema:
{ "score": <number in [0, 1]> }
Rules:
- Return exactly one score.
- score must be a number in [0, 1].
- No extra text, no markdown.
```

![](images/658538608659a4f93f4a4d076d2218ddeb4b3a492156354a484975b405db2c82.jpg)  
Figure 6. Tile-level 3DGS generations conditioned on rendered satellite-view images.

![](images/dc21eb970f37fca78f4eeabc635dff3908c917a161abe8531d00e4619d653f08.jpg)

![](images/5bcc9a889ba0a251e056c6eae446b10719e7e8c82db4328326f5ff914ef53aa9.jpg)  
Figure 7. Large-area 3DGS generation (1/2) from constructed satellite-view conditions.

![](images/a9b7147df71aa8f6eddc3aa168c5e9f5b5ed5f42c9d06399c5f76fab4307861a.jpg)

![](images/36dea649313ad12ddbeca695788a3f9cd69ae054a248136aeeb048b99080c76b.jpg)

![](images/c8ea81e82e5c33a096e46033b965c886c4dfba5998c6ff162c6c95cdedad625a.jpg)

![](images/12d30f30b2cf7eaebe911d6f0a51cf748b2320ac1be5af6e028afd9810eeccd3.jpg)  
Figure 8. Large-area 3DGS generation (2/2). Overlap-aware tiled inference produces a large-area 3DGS primitive set from the satellite-view condition.