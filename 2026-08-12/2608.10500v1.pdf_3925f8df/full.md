# DSAR: Dual-Stream Autoregressive Modeling of Temporal Cloth Dynamics for Photorealistic Animatable Avatars

Haozhong Xiong<sup>1</sup> , Yao Yu<sup>1</sup> <sup>⋆</sup>, Yu Zhou<sup>1</sup> <sup>∗</sup>, and Sidan Du<sup>1</sup>

Nanjing University, Nanjing, China 502024230011@smail.nju.edu.cn, {allanyu,nackzhou,coff128}@nju.edu.cn

![](images/22cfa6160e7259bc41e629c5b68e1a827d36e314bcb9feae5f23af83ade03c03.jpg)

Fig. 1: We propose DSAR, which explicitly models temporal causality in cloth dynamics through dual-stream autoregressive architecture, achieving realistic deformations on loose clothing and robust generalization to out-of-distribution poses.

Abstract. Creating photorealistic and temporally coherent animatable human avatars from RGB videos remains challenging. Current methods struggle to capture realistic cloth dynamics, producing over-smoothed appearance or severe artifacts on out-of-distribution poses. This limitation stems from a fundamental oversight: existing approaches neglect the temporal causality inherent in cloth physics, where current states emerge from previous states through temporal evolution rather than instantaneous skeletal configurations alone. Without explicit modeling of this causal structure, networks learn pose-appearance correlations instead of motion evolution, leading to poor generalization. We introduce a dual-stream autoregressive framework that explicitly models both observable geometric information and implicit internal state. The geometric stream propagates surface displacement from the previous frame, while the state stream fuses current features with historical states retrieved from a memory bank. Motion-adaptive aggregation handles spatiallyvarying dynamics, and adaptive regularization balances smoothness with flexibility. Experiments on challenging datasets demonstrate significant improvements in rendering quality, temporal consistency, and generalization to motion patterns beyond training distributions, validating that dual-stream temporal modeling enables realistic cloth dynamics.

Keywords: Animatable avatars · Neural rendering · Dual-Stream

⋆ Corresponding authors.

## 1 Introduction

The creation of photorealistic, animatable human avatars has emerged as a cornerstone technology for virtual reality, telepresence, film production, and interactive media. The ultimate goal is to capture the dynamic interplay between human motion and garment appearance—subtle wrinkles forming during arm bending, fabric momentum during rapid spinning, and gradual cloth settling after motion ceases. These temporal phenomena are fundamental to visual realism yet remain challenging for current neural rendering approaches.

Recent advances in neural rendering have enabled impressive progress in avatar creation from multi-view RGB videos. Building on foundational techniques in neural radiance fields [3,40] and 3D Gaussian Splatting [23], numerous methods [17, 24, 27, 31, 64] have achieved high-fidelity avatar reconstruction. Despite diferences in underlying representations, these methods share a common modeling paradigm: they model garment appearance as functions of current or recent skeletal poses, where pose information is derived from either single frames or concatenated across temporal window. While this design enables eficient per-frame reconstruction, it exhibits fundamental limitations in practice—oversmoothed appearance on training poses, temporal flickering during motion, and severe artifacts including unrealistic wrinkles on novel pose sequences.

We identify that this limitation stems from ignoring the temporal causal structure inherent in cloth motion. In reality, garment appearance at time t is not uniquely determined by skeletal pose alone. Consider a standing pose reached after rapid spinning versus from rest: while skeletal configurations are nearly identical, cloth states difer drastically due to motion history—the postspinning garment exhibits residual momentum and dynamic wrinkles absent in the static case. This reveals a fundamental one-to-many mapping where identical poses correspond to diferent cloth states depending on temporal context.

Analysis of real-world cloth motion reveals that resolving this ambiguity requires two complementary types of information. The first is explicit kinematic information—the observable geometric configuration (where cloth surfaces are positioned) and motion patterns (how surfaces are moving, characterized by velocity and directional momentum), which together describe the current observable state of cloth surfaces. The second is implicit internal state, encompassing latent properties such as fabric tension, deformation history, and material memory that influence cloth appearance but cannot be directly observed from surface geometry alone. Only when both are accounted for can the cloth state at time t be uniquely determined.

Skeletal poses provide only body joint positions, lacking both cloth kinematic information (geometric configuration and motion patterns) and internal state. This dual deficiency creates fundamental prediction ambiguity—without explicit cloth state feedback, networks must infer both "where and how cloth surfaces are moving" and "what internal factors govern their evolution" from skeletal motion alone, leading to the observed one-to-many mapping problem.

Various approaches have attempted to address this issue. RealityAvatar [28] and MonoHuman [70] optimize per-frame latent codes to disambiguate identical poses during training, but rely on pose-based interpolation at test time that cannot capture motion-dependent variations. InstantAvatar [21] and HumanRF [20] incorporate historical skeletal poses through concatenation or transformer aggregation to provide temporal context, but skeletal representations capture only body-centric motion—where the skeleton was but not how cloth has evolved—creating ambiguity when identical poses correspond to diferent cloth states. Test-time alignment strategies [9, 31] improve generalization by mapping novel poses back to training distributions, but remain efective primarily when test poses exhibit small distributional shifts from training data.

Despite their diversity in architectural design, these learning-based avatar rendering methods do not establish explicit temporal causality between successive cloth states, instead learning correlations between pose patterns and appearance rather than temporal evolution rules.

Physics-based cloth simulation [4, 5, 13, 29, 53] naturally incorporates temporal causality through explicit state evolution, tracking cloth configurations across timesteps to model momentum and dynamic deformations. However, these methods are fundamentally limited by human modeling capabilities—the precision of manually-crafted material models, the completeness of physical constraints, and the approximations necessary for computational tractability may fail to capture the full complexity of real-world cloth behavior. Our method addresses cloth dynamics from a complementary data-driven perspective, learning temporal evolution rules directly from multi-view photometric observations. This image-based supervision enables the network to discover dynamics from visual evidence of real-world phenomena, potentially capturing efects that extend beyond manually-modeled physics.

We address this through explicit dual-stream modeling. The geometric stream establishes cloth geometric information propagation via autoregressive conditioning on the previous frame’s predicted deformation $\varDelta \mu _ { t - 1 }$ , providing explicit feedback on cloth surface configuration and movement rather than inferring them from body pose alone. The state stream maintains a memory bank of historical temporal states that encode implicit internal state, capturing latent properties that are dificult to model explicitly but govern temporal evolution. These two streams address the dual requirements: the geometric stream tracks observable kinematics deformation, while the state stream captures implicit internal state learned from visual data.

Contributions. Our work makes the following contributions:

– We identify that temporally coherent cloth rendering requires modeling both observable kinematic information and implicit internal state, which cannot be inferred from skeletal poses alone due to the one-to-many mapping problem.

– We propose a dual-stream autoregressive architecture that explicitly addresses these dual requirements from photometric supervision: a geometric stream propagates kinematic information through autoregressive conditioning on previous frame’s cloth deformation and motion-adaptive temporal aggregation, while a state stream encodes implicit internal state through memory-based retrieval of historical temporal features.

![](images/24df2fc1f9bb3162b811193d91f1b1ea60519aeef52f2bc63e608e6b8565ec6f.jpg)  
Fig. 2: Overview. Our dual-stream framework processes a temporal window and previous deformation $\varDelta \mu _ { t - 1 }$ through two complementary pathways. The geometric stream aggregates multi-scale features via MATA. The state stream fuses aggregated features with historical states from memory bank $\mathcal { M } _ { t }$ to obtain temporal state $h _ { t } ,$ which is decoded into Gaussian deformations $\varDelta G _ { t }$ and stored for next-frame inference.

– Our framework achieves significant improvements in rendering quality, temporal consistency, and generalization to motion patterns beyond training distributions.

## 2 Related Work

Animatable Human Avatar Representation. Early reconstruction methods employ implicit functions, such as occupancy fields [16,19,37–39,50,51] and signed distance functions [10,44,52,60,63,74], or utilize explicit mesh representations [15,25,68,69] to reconstruct clothed humans from scans or depth sequences. However, these representations often struggle to model view-dependent appearance efects and complex materials, limiting photorealistic rendering quality and temporal consistency.

Building on neural rendering techniques [34, 48, 55, 56, 59], neural radiance field-based methods have revolutionized avatar creation through diferentiable volumetric rendering with view-dependent appearance. Recent advances [6, 11, 43] have enabled numerous human modeling approaches [14,27,30,32,47,57,64, 75]. HumanNeRF [67] learns pose-dependent deformations using skeletal motion priors, while Neural Body [47] associates latent codes with SMPL vertices for pose-conditioned rendering. AnimatableNeRF [46] establishes a canonical NeRF with pose refinement for pose-driven animation. TAVA [27] adopts explicit warping fields for fine-grained control. However, these implicit representations are computationally expensive, and their MLP-based architectures struggle to capture high-frequency geometric details due to spectral bias [58].

3D Gaussian Splatting [23] enables real-time rendering through explicit pointbased primitives, inspiring various avatar modeling approaches [2, 17, 18, 41, 54,

![](images/f9fdeddfcca7ea6e4dff71297ab56ea6aa3101644b822efe878aafc6edbd3793.jpg)  
Fig. 3: Motion-Adaptive Temporal Aggregation (MATA). Features are aggregated hierarchically: motion-adaptive causal attention at the deepest level $\left( f _ { \tau } ^ { L } \right)$ and lightweight convolutions $\left( \varPhi _ { a g g } \right)$ at finer levels $\big ( f _ { \tau } ^ { 1 : L - 1 } \big )$

76]. GaussianAvatar [17] binds Gaussians to UV maps with learnable features for pose-driven animation. Animatable-GS [31] employs StyleUNet [22, 62] for multi-scale deformation prediction with learnable skinning weights. To enhance geometric quality, several methods [33, 72] introduce geometric priors including as-rigid-as-possible constraints and normal consistency regularization. Despite achieving impressive rendering performance, these methods fundamentally model clothing appearance as functions of current or recent skeletal poses, formulated as Appearance $\mathbf { \dot { \tau } } _ { t } = F ( \mathrm { p o s e } _ { t - n : t } )$ , neglecting the temporal causality where current cloth states emerge from previous states through evolution.

Temporal Modeling in Neural Avatar Rendering. Various neural rendering approaches have explored temporal modeling to capture cloth dynamics beyond instantaneous pose-to-appearance mappings.

Per-frame latent encoding approaches [7, 28, 70, 73] optimize frame-specific codes to disambiguate identical poses during training, capturing variability that skeletal pose alone cannot explain. RealityAvatar [28] learns per-frame latent codes jointly with neural rendering, while MonoHuman [70] employs 4D human representation with spatio-temporal features. However, these codes are optimized independently without temporal constraints between frames. At test time, codes for novel poses must be inferred through pose-based interpolation—finding training frames with similar skeletal configurations and blending their optimized codes—which cannot capture motion-dependent variations where identical poses correspond to diferent cloth states depending on motion history.

Skeletal pose concatenation methods [20, 21, 67] incorporate historical skeletal poses as network input to provide temporal context through body motion sequences. InstantAvatar [21] concatenates multiple consecutive pose parameters through temporal encoders, while HumanRF [20] employs transformers to aggregate pose sequences for dynamic human modeling. Neural Cloth Simulation [5] processes body motion through recurrent encoders with disentangled static and dynamic branches, enabling fast cloth animation through learned temporal dynamics in latent space. However, these skeletal representations provide only body-centric motion information—they capture how the skeleton moved but not the actual cloth geometric state from previous frames. This creates ambiguity when identical poses correspond to diferent cloth states due to motion history.

Test-time alignment strategies [9,31] address generalization to novel poses by mapping out-of-distribution poses back to the training distribution. Animatable-GS [31] employs PCA-based dimensionality reduction to project test poses into the training manifold, while RANA [9] introduces explicit pose space alignment through learned transformations. These methods remain efective primarily when test poses exhibit small distributional shifts from training data, as pose similarity-based alignment cannot capture motion-dependent cloth state variations.

Despite their diversity in architectural design, these learning-based methods do not establish explicit temporal causality between successive cloth states, instead learning correlations between pose patterns and appearance rather than temporal evolution rules, limiting their ability to generalize beyond training sequences.

Physics-based cloth simulation [4, 5, 8, 13, 29, 42, 53] models garment dynamics through explicit state evolution, naturally incorporating temporal causality but bounded by manually-modeled physical constraints. Our work focuses on photorealistic avatar rendering from multi-view images, learning cloth dynamics directly from photometric observations while establishing explicit temporal causality through autoregressive geometric state propagation and distributed memory-based state retrieval.

## 3 Method

## 3.1 Preliminary

3D Gaussian Splatting. We adopt 3D Gaussian Splatting [23] as our rendering representation. Each Gaussian is parameterized by its 3D center $\mu \in \mathbb { R } ^ { 3 }$ rotation quaternion $q \in \mathbb { R } ^ { 4 }$ , anisotropic scale $s \in \mathbb { R } ^ { 3 }$ , opacity $\alpha \in \mathbb { R }$ , and spherical harmonics coeficients c for view-dependent color. The covariance matrix is factorized as $\Sigma = R S S ^ { T } R ^ { T }$ , where $R$ is the rotation matrix derived from quaternion $q ,$ and S is a diagonal scaling matrix. During rendering, 3D Gaussians are projected onto the 2D image plane and blended via diferentiable alpha compositing.

SMPL Model and Linear Blend Skinning. We utilize SMPL [35] and SMPL-X [45] as the underlying body model, parameterized by shape $\beta \in \mathbb { R } ^ { 1 0 }$ and pose $\theta \in \mathbb { R } ^ { J \times 3 }$ , where $J$ denotes the number of body joints. SMPL provides a template mesh $\mathcal { T } ( \beta )$ in canonical space and joint locations $\mathcal { I } ( \beta )$ that enable articulated motion. To transform points from canonical space to posed space, we employ Linear Blend Skinning (LBS). Given a canonical point $x _ { c }$ with skinning weights $w = \{ w _ { 1 } , \ldots , w _ { K } \}$ and bone transformation matrices $\{ B _ { 1 } , \ldots , B _ { K } \}$ derived from pose θ with K joints, the posed position is:

$$
x _ { p } = \sum _ { k = 1 } ^ { K } w _ { k } B _ { k } x _ { c }\tag{1}
$$

## 3.2 Overview

We represent the avatar using a learnable Gaussian template $G _ { \mathrm { t m p } }$ in canonical space. At each frame, the network predicts deformations $\varDelta G _ { t }$ that modify the template’s Gaussian attributes. These deformations are added to the template to obtain deformed canonical Gaussians, which are then transformed to posed space via Linear Blend Skinning and rendered through diferentiable splatting.

Our deformation network Ψ adopts a StyleUNet [22, 49, 62] architecture and learns cloth dynamics through a dual-stream autoregressive process. As shown in Fig. 2, the geometric stream extracts multi-scale features through an encoder and aggregates them temporally using Motion-Adaptive Temporal Aggregation (MATA) to produce geometric features $A _ { t }$ . The state stream fuses $A _ { t }$ with historical states from a memory bank to obtain temporal state $h _ { t } ,$ , which is decoded to predict canonical-space deformations. During training, multi-view photometric supervision and adaptive temporal regularization guide the network to discover temporal evolution patterns.

## 3.3 Learnable Gaussian Template

We model avatar dynamics by predicting deformations of a learnable Gaussian template in canonical space, which are then transformed to posed space via Linear Blend Skinning.

Template Initialization. We initialize an optimizable Gaussian template $G _ { \mathrm { t m p } }$ in canonical space by binding Gaussians to the SMPL-X surface using its UV parameterization. Each Gaussian is placed at a UV coordinate location with skinning weights inherited from nearby vertices. The number of Gaussians depends on the UV map resolution and remains fixed throughout training, ensuring spatial correspondence across frames.

Template and Deformation Factorization. We factorize the canonical Gaussians into a learnable template and frame-specific deformations:

$$
G _ { t } ^ { \mathrm { c a n o } } = G _ { \mathrm { t m p } } + \varDelta G _ { t }\tag{2}
$$

where $\varDelta G _ { t } = \left\{ \varDelta \mu _ { t } , \varDelta q _ { t } , \varDelta s _ { t } , \varDelta \alpha _ { t } , \varDelta c _ { t } \right\}$ represents deformations in Gaussian attributes. The template captures average canonical geometry and appearance, while deformations encode pose-dependent and temporally-driven variations.

Posed Space Transformation. To render under driving pose $\theta _ { t }$ , we transform the deformed canonical Gaussians to posed space via LBS. For a canonical Gaussian with center $\mu _ { c }$ and covariance $\Sigma _ { c }$ :

$$
\mu _ { p } = \sum _ { k = 1 } ^ { K } w _ { k } B _ { k } \mu _ { c } , \quad \varSigma _ { p } = J \varSigma _ { c } J ^ { T }\tag{3}
$$

where J is the rotation component of the bone transformation matrices. The posed Gaussians are rendered to the target viewpoint through splatting-based rasterization.

## 3.4 Dual-Stream Autoregressive Architecture

At each time step t, we predict deformations through two complementary autoregressive streams. The geometric stream explicitly conditions on the previous frame’s predicted geometric deformation $\varDelta \mu _ { t - 1 }$ and processes the current temporal window through motion-adaptive aggregation. The state stream maintains a memory bank of historical temporal states and fuses them with current geometric features to capture long-term temporal evolution.

Our deformation prediction conditions on the immediate previous $\varDelta \mu _ { t - 1 }$ rather than a full historical sequence. This single-frame design is suficient because $\varDelta \mu _ { t - 1 }$ was itself predicted from earlier frames through the autoregressive chain, thereby carrying cloth geometric history.

Geometric Stream. The network input consists of UV-unwrapped position maps derived from skeletal pose and gaussian template. For each frame τ in the temporal window $\{ t - n + 1 , \ldots , t \}$ , the input comprises the position map $p _ { \tau } ,$ velocity map $\varDelta p _ { \tau }$ , and the geometric information $\varDelta \mu _ { t - 1 } $ :

$$
I _ { \tau } = \mathrm { C o n c a t } [ p _ { \tau } , \varDelta p _ { \tau } , \varDelta \mu _ { t - 1 } ]\tag{4}
$$

where $p _ { \tau }$ captures surface configuration, $\varDelta p _ { \tau } = p _ { \tau } - p _ { \tau - 1 }$ captures motion dynamics, and $\varDelta \mu _ { t - 1 }$ is shared across all frames in the window, providing cloth geometric deformation from the previous frame.

We extract multi-scale features through StyleUNet encoder $\varPsi _ { \mathrm { e n c } } ,$ producing an L-level feature pyramid for each frame, where $l = 1$ denotes the finest resolution and $l = L$ the coarsest:

$$
\{ f _ { \tau } ^ { l } \} _ { \tau = t - n + 1 } ^ { t } , \quad l \in \{ 1 , \dots , L \}\tag{5}
$$

This yields n feature pyramids, one for each frame in the temporal window.

Motion-Adaptive Temporal Aggregation. Cloth exhibits spatially heterogeneous motion characteristics, requiring adaptive processing rather than uniform aggregation. At the deepest semantic level (Level L), we employ motion-adaptive temporal aggregation to process the temporal window.

We first quantify motion magnitude at each spatial location from the velocity maps to capture motion heterogeneity across diferent body regions:

$$
V _ { \tau } [ u , v ] = \lVert \varDelta p _ { \tau } [ u , v ] \rVert _ { 2 }\tag{6}
$$

where $V _ { \tau } [ u , v ]$ represents the motion magnitude at spatial location $( u , v )$ in frame τ . A lightweight network $\varPhi _ { \mathrm { m } }$ transforms motion magnitude maps into featurespace embeddings:

$$
m _ { \tau } = \phi _ { \mathrm { m } } ( V _ { \tau } ) , \quad \tau \in \{ t - n + 1 , \ldots , t \}\tag{7}
$$

where $\varPhi _ { \mathrm { m } }$ consists of lightweight convolutions that project spatial motion patterns into the feature dimension. Following [26], we incorporate motion embeddings and temporal position encodings into the deepest features:

$$
\tilde { f } _ { \tau } ^ { L } = f _ { \tau } ^ { L } + m _ { \tau } + e _ { \tau } , \quad \tau \in \{ t - n + 1 , \ldots , t \}\tag{8}
$$

where $e _ { \tau }$ are position embeddings. This motion-enhanced representation enables the network to attend adaptively based on local motion characteristics—highmotion regions emphasize recent dynamic information while low-motion regions maintain broader temporal context.

We then compute causal self-attention [1, 61] over the n motion-enhanced frames:

$$
Q = \varPhi _ { Q } \big ( \big \{ \tilde { f } _ { \tau } ^ { L } \} \big ) , \quad K = \varPhi _ { K } \big ( \big \{ \tilde { f } _ { \tau } ^ { L } \} \big ) , \quad V = \varPhi _ { V } \big ( \big \{ \tilde { f } _ { \tau } ^ { L } \} \big )\tag{9}
$$

where $\varPhi _ { Q } , \varPhi _ { K } , \varPhi _ { V }$ are projection networks. We apply causal masking to enforce temporal causality:

$$
M _ { \mathrm { c a u s a l } } [ \tau , \tau ^ { \prime } ] = \left\{ { \begin{array} { l l } { - \infty , } & { { \mathrm { i f ~ } } \tau < \tau ^ { \prime } } \\ { 0 , } & { { \mathrm { i f ~ } } \tau \geq \tau ^ { \prime } } \end{array} } \right.\tag{10}
$$

where $\tau , \tau ^ { \prime } \in \{ t - n + 1 , \ldots , t \}$ denote frame indices in the temporal window. The final attention output aggregates temporal information within the window while respecting causal constraints:

$$
A _ { t } = \mathrm { s o f t m a x } \left( \frac { Q K ^ { T } } { \sqrt { d _ { k } } } + M _ { \mathrm { c a u s a l } } \right) V\tag{11}
$$

This motion guidance enables spatially-adaptive attention patterns, where actively moving regions focus on recent frames to capture transient dynamics, while stable regions distribute attention more uniformly to maintain temporal coherence.

State Stream. While $A _ { t }$ captures surface configuration and motion patterns observable from the temporal window, cloth dynamics also depend on implicit internal state that is not directly observable from geometry alone. The state stream addresses this by maintaining a memory bank of historical temporal states that encode this implicit internal state and fusing them with current geometric features.

Specifically, we maintain a memory bank $\mathcal { M } _ { t } = \{ h _ { t - m } , \ldots , h _ { t - 1 } \}$ storing the most recent M temporal states, where each $h _ { i }$ encodes the implicit internal state from previous timesteps. At time $t ,$ we retrieve relevant temporal information from the memory bank and fuse it with the current geometric feature $A _ { t }$ through cross-attention:

$$
h _ { t } ^ { * } = \mathrm { C r o s s A t t n } ( Q = A _ { t } , K V = \mathcal { M } _ { t } )\tag{12}
$$

$$
h _ { t } = ( 1 - \lambda ) A _ { t } + \lambda \cdot h _ { t } ^ { * }\tag{13}
$$

where λ is a learnable parameter that balances current geometric information and historical temporal patterns.

After obtaining $h _ { t } .$ we update the memory bank for the next timestep:

$$
\mathcal { M } _ { t + 1 } = \left\{ h _ { t - M + 1 } , \ldots , h _ { t - 1 } , h _ { t } \right\}\tag{14}
$$

Hierarchical Aggregation and Decoding. The fused temporal state $h _ { t }$ serves as the aggregated feature at the deepest level (Level L). At shallower levels (Levels 1 to $L - 1 )$ , we use lightweight convolutional networks $\varPhi _ { \mathrm { a g g } } ^ { l }$ for eficient temporal aggregation:

$$
\begin{array} { r } { F _ { t } ^ { l } = \left\{ \begin{array} { l l } { h _ { t } , } & { l = L } \\ { \phi _ { \mathrm { a g g } } ^ { l } ( \{ f _ { \tau } ^ { l } \} _ { \tau = t - n + 1 } ^ { t } ) , } & { l \in \{ 1 , \dots , L - 1 \} } \end{array} \right. } \end{array}\tag{15}
$$

The hierarchical aggregation produces the final feature pyramid $\{ F _ { t } ^ { l } \} _ { l = 1 } ^ { L }$ , combining dual-stream fusion at the deepest semantic level with eficient convolutions at finer detail levels. The aggregated multi-scale features are fed into StyleUNet decoder $\varPsi _ { \mathrm { d e c } }$ with skip connections from the encoder, which predicts the canonical-space deformations:

$$
\varDelta G _ { t } = \varPsi _ { \mathrm { d e c } } ( \{ F _ { t } ^ { l } \} _ { l = 1 } ^ { L } )\tag{16}
$$

## 3.5 Adaptive Temporal Regularization

Uniform temporal smoothness constraints create fundamental tension: they stabilize static poses but over-smooth dynamic deformations, suppressing natural momentum efects. We resolve this through motion-magnitude-aware weighting that dynamically adapts regularization strength based on spatially-varying motion characteristics.

Motion-Magnitude-Aware Weighting. For each Gaussian i at frame t, we compute an adaptive weight based on its local motion magnitude. Let $( u _ { i } , v _ { i } )$ denote the UV coordinates of Gaussian i in the canonical template. We compute per-Gaussian adaptive weights:

$$
w _ { t } [ i ] = \exp \left( - \frac { V _ { t } [ u _ { i } , v _ { i } ] } { \tau _ { \mathrm { r e g } } } \right)\tag{17}
$$

where $V _ { t } [ u _ { i } , v _ { i } ]$ is the motion magnitude at the Gaussian’s spatial location and $\tau _ { \mathrm { r e g } }$ is a hyperparameter controlling sensitivity. This formulation achieves spatiallyvarying adaptive regularization: high weights for static regions enforce strong temporal consistency, while low weights for high-motion regions permit flexibility.

Multi-Level Regularization. We apply motion-weighted regularization at three hierarchical levels:

Feature-Level Smoothness regularizes intermediate features at each pyramid level:

$$
\mathcal { L } _ { \mathrm { f e a t } } = \sum _ { l = 1 } ^ { L } \sum _ { t } \gamma _ { l } \cdot \Vert F _ { t } ^ { l } - F _ { t - 1 } ^ { l } \Vert _ { 2 } ^ { 2 }\tag{18}
$$

where $\gamma _ { l } ~ = ~ 0 . 1$ controls regularization strength at each level. This prevents high-frequency temporal variations in learned representations that could lead to flickering artifacts.

Prediction-Level Consistency regularizes predicted deformations with motionadaptive weighting:

$$
\begin{array} { r } { \mathscr { L } _ { \mathrm { p r e d } } = \displaystyle \sum _ { t } \sum _ { i } w _ { t } [ i ] \cdot \bigg ( \| \Delta \mu _ { t } [ i ] - \Delta \mu _ { t - 1 } [ i ] \| _ { 2 } ^ { 2 } + \| \Delta q _ { t } [ i ] - \Delta q _ { t - 1 } [ i ] \| _ { 2 } ^ { 2 } } \\ { + \| \Delta s _ { t } [ i ] - \Delta s _ { t - 1 } [ i ] \| _ { 2 } ^ { 2 } \bigg ) } \end{array}\tag{19}
$$

The spatially-varying weights $w _ { t } [ i ]$ enforce strong consistency in static regions (e.g., torso during standing) while permitting flexibility in dynamic regions (e.g., sleeves during arm swinging).

Cross-Frame Position Smoothness penalizes second-order temporal diferences in posed positions:

$$
\mathrm { a c c e l } _ { t } [ i ] = \mu _ { t + 1 } ^ { \mathrm { p } } [ i ] - 2 \mu _ { t } ^ { \mathrm { p } } [ i ] + \mu _ { t - 1 } ^ { \mathrm { p } } [ i ]\tag{20}
$$

$$
\mathcal { L } _ { \mathrm { s m } } = \sum _ { t } \sum _ { i } w _ { t } [ i ] \cdot \| \mathrm { a c c e l } _ { t } [ i ] \| _ { 2 } ^ { 2 }\tag{21}
$$

where $\mu _ { t } ^ { \mathrm { p } } [ i ]$ denotes the posed position of Gaussian i at frame t. This acceleration penalty minimizes jerk for smooth trajectories while motion-adaptive weighting permits natural momentum-driven motion in active regions.

Training Objective. Our complete training loss combines multi-view photometric supervision with adaptive temporal regularization:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = { \mathcal { L } } _ { \mathrm { r e n d e r } } + \lambda _ { \mathrm { f e a t } } { \mathcal { L } } _ { \mathrm { f e a t } } + \lambda _ { \mathrm { p r e d } } { \mathcal { L } } _ { \mathrm { p r e d } } + \lambda _ { \mathrm { s m } } { \mathcal { L } } _ { \mathrm { s m } }\tag{22}
$$

The rendering loss comprises:

$$
{ \mathcal { L } } _ { \mathrm { r e n d e r } } = \lambda _ { \mathrm { r g b } } { \mathcal { L } } _ { \mathrm { r g b } } + \lambda _ { \mathrm { s s i m } } { \mathcal { L } } _ { \mathrm { s s i m } } + \lambda _ { \mathrm { l p i p s } } { \mathcal { L } } _ { \mathrm { l p i p s } }\tag{23}
$$

where $\mathcal { L } _ { \mathrm { r g b } }$ is the L1 loss between rendered and ground-truth images, $\mathcal { L } _ { \mathrm { s s i m } }$ [66] measures structural similarity, and $\mathcal { L } _ { \mathrm { l p i p s } }$ [71] captures perceptual quality. These photometric losses provide supervision from multiple camera views, enabling the network to learn accurate cloth deformations from visual observations alone without requiring explicit geometric supervision.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We evaluate our method on 4D-DRESS [65] and AvatarREX [75]. Each subject contains 5-6 motion sequences (approximately 1000 frames total, captured by 8 calibrated cameras). We train on 3-4 sequences and test on a separate held-out sequence. To further validate generalization to challenging out-of-distribution poses, we additionally evaluate on motion sequences from AMASS [36], which provides diverse motion patterns including sports, dancing, and acrobatic movements substantially diferent from training data.

Distribution Divergence Quantification. To systematically evaluate generalization under varying distribution shifts, we quantify pose distribution divergence between training and test sets using Maximum Mean Discrepancy (MMD) [12]. Given pose feature sets $\chi _ { \mathrm { t r a i n } }$ and $\mathcal { X } _ { \mathrm { t e s t } }$ extracted from SMPL-X parameters, MMD measures distributional distance via kernel embeddings:

$$
\begin{array} { r l } & { \mathrm { M M D } ^ { 2 } = \lVert \mu _ { \mathcal { X } } - \mu _ { \mathcal { Y } } \rVert _ { \mathcal { H } } ^ { 2 } } \\ & { \qquad = \mathbb { E } [ k ( x , x ^ { \prime } ) ] - 2 \mathbb { E } [ k ( x , y ) ] + \mathbb { E } [ k ( y , y ^ { \prime } ) ] } \end{array}\tag{24}
$$

where k is an RBF kernel and H is the reproducing kernel Hilbert space. We categorize test sequences as Near-Distribution (ND) with $\mathrm { M M D } \ < \ 0 . 3$ (similar motion patterns) or Far-Distribution (FD) with $\mathrm { M M D } > 0 . 6$ (substantially diferent dynamics), enabling rigorous evaluation of out-of-distribution generalization.

Baselines. We compare against HumanNeRF [67], GaussianAvatar [17], and Animatable-GS [31], representing NeRF and Gaussian Splatting paradigms. All baselines use oficial implementations trained on identical data with recommended settings.

Metrics. We evaluate rendering quality using PSNR, SSIM [66] and LPIPS [71]. Implementation Details. We employ a StyleUNet architecture with temporal window size $n = 1 0$ , feature pyramid levels L = 4, UV map resolution $H \times W = 5 1 2 \times 5 1 2$ , ∼200k Gaussians per subject, memory bank size $m = 1 0$ frames. Training uses Adam optimizer (learning rate $l r = 6 \times 1 0 ^ { - 4 } )$ . Loss weights: $\lambda _ { \mathrm { { r e n d e r } } } = 1 , \lambda _ { \mathrm { { s i m } } } = 0 . 2 , \lambda _ { \mathrm { { l p i p s } } } = 0 . 5 , \lambda _ { \mathrm { { f e a t } } } = 0 . 0 1 , \lambda _ { \mathrm { { p r e d } } } = 0 . 1 , \lambda _ { \mathrm { { s m } } } = 0 . 0 5$

## 4.2 Comparisons

Table 1 presents quantitative evaluation. Our method consistently outperforms baselines, confirming that explicit modeling of kinematic information and implicit internal state enables robust generalization.

Figure 4 demonstrates our method’s ability to capture realistic cloth dynamics. In post-rotation stopping (row 1), our method correctly models momentumdriven evolution: the jacket continues swinging after the body stops, then gradually settles. Baselines produce artifacts due to instantaneous modeling—without explicit kinematic feedback and internal state encoding, they infer appearance solely from current pose. Similar improvements appear in arm movements (row 2) and jumping (row 3), where our dual-stream architecture captures natural dynamics while baselines exhibit flickering and over-smoothing.

## 4.3 Ablation Studies

We systematically validate each component’s contribution through ablation studies. Table 2 and Figure 5 compare five variants: (a) Ground Truth, (b) Full model,

Ours

![](images/cfa9edb0c5575db8ea7f67b8041ae07364acc6b1b72aa58334d7cec9f1b4d501.jpg)

![](images/ac67b3fadcfd4b385648aa50b074688d436e1acca96892886f1d46a7c63e3849.jpg)

![](images/8cc784e16b246554f827190ec67ae88363d646981934c4e4b836244c2640baf2.jpg)  
Animatable-GS

![](images/678adf50b49d5026570f563032f18e00fda220690ee781166dd31da2cc8c7a32.jpg)  
GS Avatar

![](images/b9026a0c1f1a3acbc1dbe8aff97e6e36a0af4dad2378ded89a707bdc092931ec.jpg)  
HumanNeRF

Fig. 4: Qualitative comparison with other methods on challenging test poses. Our method better captures momentum-driven cloth dynamics and generalizes to novel poses.  
Table 1: Quantitative comparison. All methods are evaluated at 940×1280 resolution.
<table><tr><td>Method</td><td>PSNR↑ SSIM↑</td><td>LPIPS↓</td></tr><tr><td>HumanNeRF</td><td>20.0567 0.9121</td><td>0.1292</td></tr><tr><td>GaussianAvatar</td><td>26.2311</td><td>0.9575 0.0715</td></tr><tr><td>Animatable-GS</td><td>29.5138</td><td>0.9724 0.0413</td></tr><tr><td>Ours</td><td>31.0765</td><td>0.9839 0.0305</td></tr></table>

(c) $\mathrm { w / o }$ Adaptive Regularization, (d) w/o Geometric Stream (removing $\varDelta \mu _ { t - 1 }$ input), (e) $\mathrm { w / o }$ State Stream (removing memory bank and cross-attention).

Results reveal the complementary necessity of both streams, particularly on far-distribution (FD) poses. The $\mathrm { w / o }$ Geometric Stream variant (d) exhibits the most severe degradation, confirming that explicit geometric deformation feedback is essential—without $\varDelta \mu _ { t - 1 }$ and motion-adaptive aggregation, the network must infer cloth configuration and motion from body pose alone, leading to accumulated errors in position and velocity estimation. The $\mathrm { w / o }$ State Stream variant (e) also degrades significantly, demonstrating that implicit internal state cannot be reliably recovered from body pose alone.

Notably, the substantially larger performance degradation on FD versus ND sets validates that explicit temporal causality modeling is essential for achieving robust generalization beyond the training distribution.

![](images/c357ec5ebd6602a9253b95e4c930d1a11c66ac9fb11e795c8c0a79d752126010.jpg)  
(a)  
(b)  
(c)  
(d)  
(e)  
Fig. 5: Ablation study. (a) Ground Truth, (b) Full model with both streams, (c) w/o adaptive regularization, (d) w/o Geometric Stream, (e) w/o State Stream.

Table 2: Ablation study on ND (MMD=0.23) and FD (MMD=0.87) test sequences.
<table><tr><td>Split|</td><td>Variant</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td rowspan="4">ND</td><td>Full Model</td><td>31.7153</td><td>0.9802</td><td>0.0283</td></tr><tr><td rowspan="2"> $\mathrm { w } / \mathrm { o }$  Adaptive Reg.</td><td rowspan="2">29.3130</td><td>0.9680</td><td>0.0453</td></tr><tr><td rowspan="2">28.9622 0.9651</td><td rowspan="2">0.0476</td></tr><tr><td rowspan="2"> $\mathrm { w } / \mathrm { o }$  GeoStream.</td></tr><tr><td>w/o StateStream.</td><td>30.2593</td><td>0.9676</td><td>0.0386</td></tr><tr><td rowspan="4">FD</td><td rowspan="2">Full Model  $\mathrm { w } / \mathrm { o }$  Adaptive Reg.</td><td rowspan="2">30.6720 28.8778</td><td>0.9739</td><td>0.0302</td></tr><tr><td rowspan="2">0.9658</td><td rowspan="2">0.0492</td></tr><tr><td rowspan="2"> $\mathrm { w } / \mathrm { o }$ </td></tr><tr><td>GeoStream. 24.2694 w/o StateStream. 28.2286</td><td>0.9557 0.9632</td><td>0.0623 0.0511</td></tr></table>

## 5 Conclusion

We present a dual-stream autoregressive framework for temporally coherent animatable avatars. We identify that cloth rendering requires modeling both observable kinematic information and implicit internal state—dual requirements that skeletal poses alone cannot satisfy. By explicitly modeling kinematic information through autoregressive conditioning and implicit internal state through memorybased fusion, our method learns temporally coherent, photorealistic cloth dynamics directly from multi-view photometric observations.

Experiments demonstrate robust generalization to motion patterns beyond training distributions, validating that dual-stream temporal modeling is essential for realistic cloth dynamics in neural rendering.

## References

1. Bahdanau, D., Cho, K., Bengio, Y.: Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473 (2014)

2. Bao, Y., Ding, T., Huo, J., Liu, Y., Li, Y., Li, W., Gao, Y., Luo, J.: 3d gaussian splatting: Survey, technologies, challenges, and opportunities. IEEE Transactions on Circuits and Systems for Video Technology (2025)

3. Barron, J.T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., Srinivasan, P.P.: Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 5855–5864 (2021)

4. Bertiche, H., Madadi, M., Escalera, S.: Cloth3d: clothed 3d humans. In: European Conference on Computer Vision. pp. 344–359. Springer (2020)

5. Bertiche, H., Madadi, M., Escalera, S.: Neural cloth simulation. ACM Transactions on Graphics (TOG) 41(6), 1–14 (2022)

6. Chen, A., Xu, Z., Geiger, A., Yu, J., Su, H.: Tensorf: Tensorial radiance fields. In: European conference on computer vision. pp. 333–350. Springer (2022)

7. Chen, Y., Wang, X., Chen, X., Zhang, Q., Li, X., Guo, Y., Wang, J., Wang, F.: Uv volumes for real-time rendering of editable free-view human performance. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16621–16631 (2023)

8. Choi, K.J., Ko, H.S.: Stable but responsive cloth. In: ACM SIGGRAPH 2005 Courses. ACM (2005)

9. Deng, X., Zheng, Z., Zhang, Y., Sun, J., Xu, C., Yang, X., Wang, L., Liu, Y.: Ram-avatar: Real-time photo-realistic avatar from monocular videos with full-body control. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1996–2007 (2024)

10. Dong, Z., Guo, C., Song, J., Chen, X., Geiger, A., Hilliges, O.: Pina: Learning a personalized implicit neural avatar from a single rgb-d video sequence. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20470–20480 (2022)

11. Fridovich-Keil, S., Yu, A., Tancik, M., Chen, Q., Recht, B., Kanazawa, A.: Plenoxels: Radiance fields without neural networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5501–5510 (2022)

12. Gretton, A., Borgwardt, K.M., Rasch, M.J., Schölkopf, B., Smola, A.: A kernel two-sample test. Journal of Machine Learning Research 13, 723–773 (2012)

13. Grigorev, A., Black, M.J., Hilliges, O.: Hood: Hierarchical graphs for generalized modelling of clothing dynamics. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 16965–16974 (2023)

14. Guo, C., Jiang, T., Chen, X., Song, J., Hilliges, O.: Vid2avatar: 3d avatar reconstruction from videos in the wild via self-supervised scene decomposition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12858–12868 (2023)

15. Habermann, M., Xu, W., Zollhoefer, M., Pons-Moll, G., Theobalt, C.: A deeper look into deepcap. IEEE Transactions on Pattern Analysis and Machine Intelligence 45(4), 4009–4022 (2021)

16. He, T., Xu, Y., Saito, S., Soatto, S., Tung, T.: Arch++: Animation-ready clothed human reconstruction revisited. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 11046–11056 (2021)

17. Hu, L., Zhang, H., Zhang, Y., Zhou, B., Liu, B., Zhang, S., Nie, L.: Gaussianavatar: Towards realistic human avatar modeling from a single video via animatable 3d gaussians. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 634–644 (2024)

18. Hu, S., Hu, T., Liu, Z.: Gauhuman: Articulated gaussian splatting from monocular human videos. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 20418–20431 (2024)

19. Huang, Z., Xu, Y., Lassner, C., Li, H., Tung, T.: Arch: Animatable reconstruction of clothed humans. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3093–3102 (2020)

20. Işık, M., Rünz, M., Georgopoulos, M., Khakhulin, T., Starck, J., Agapito, L., Nießner, M.: Humanrf: High-fidelity neural radiance fields for humans in motion. ACM Transactions on Graphics (TOG) 42(4), 1–12 (2023)

21. Jiang, T., Chen, X., Song, J., Hilliges, O.: Instantavatar: Learning avatars from monocular video in 60 seconds. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16922–16932 (2023)

22. Karras, T., Laine, S., Aila, T.: A style-based generator architecture for generative adversarial networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4401–4410 (2019)

23. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42(4), 139–1 (2023)

24. Kocabas, M., Chang, J.H.R., Gabriel, J., Tuzel, O., Ranjan, A.: Hugs: Human gaussian splats. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 505–515 (2024)

25. Kwon, Y., Liu, L., Fuchs, H., Habermann, M., Theobalt, C.: Delifas: Deformable light fields for fast avatar synthesis. Advances in neural information processing systems 36, 40944–40962 (2023)

26. Li, D., Zhong, W., Yu, W., Pan, Y., Zhang, D., Yao, T., Han, J., Mei, T.: Pursuing temporal-consistent video virtual try-on via dynamic pose interaction. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 22648–22657 (2025)

27. Li, R., Tanke, J., Vo, M., Zollhöfer, M., Gall, J., Kanazawa, A., Lassner, C.: Tava: Template-free animatable volumetric actors. In: European Conference on Computer Vision. pp. 419–436. Springer (2022)

28. Li, Y., Zeng, Z., Pang, L., Zhang, G., Zhang, S.: Realityavatar: Towards realistic loose clothing modeling in animatable 3d gaussian avatars. arXiv preprint arXiv:2504.01559 (2025)

29. Li, Y., Du, T., Wu, K., Xu, J., Matusik, W.: Difcloth: Diferentiable cloth simulation with dry frictional contact. ACM Transactions on Graphics (TOG) 42(1), 1–20 (2022)

30. Li, Z., Zheng, Z., Liu, Y., Zhou, B., Liu, Y.: Posevocab: Learning joint-structured pose embeddings for human avatar modeling. In: ACM SIGGRAPH 2023 conference proceedings. pp. 1–11 (2023)

31. Li, Z., Zheng, Z., Wang, L., Liu, Y.: Animatable gaussians: Learning posedependent gaussian maps for high-fidelity human avatar modeling. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 19711–19722 (2024)

32. Liu, L., Habermann, M., Rudnev, V., Sarkar, K., Gu, J., Theobalt, C.: Neural actor: Neural free-view synthesis of human actors with pose control. ACM transactions on graphics (TOG) 40(6), 1–16 (2021)

33. Liu, X., Wu, C.: Vga: Reconstructing vivid 3d gaussian avatars from monocular videos. In: International Conference on Computational Visual Media. pp. 172–193. Springer (2025)

34. Lombardi, S., Simon, T., Saragih, J., Schwartz, G., Lehrmann, A., Sheikh, Y.: Neural volumes: learning dynamic renderable volumes from images. ACM Transactions on Graphics (TOG) 38(4), 1–14 (2019)

35. Loper, M., Mahmood, N., Romero, J., Pons-Moll, G., Black, M.J.: Smpl: A skinned multi-person linear model. ACM TOG (2015)

36. Mahmood, N., Ghorbani, N., Troje, N.F., Pons-Moll, G., Black, M.J.: Amass: Archive of motion capture as surface shapes. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 5442–5451 (2019)

37. Mescheder, L., Oechsle, M., Niemeyer, M., Nowozin, S., Geiger, A.: Occupancy networks: Learning 3d reconstruction in function space. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4460–4470 (2019)

38. Mihajlovic, M., Saito, S., Bansal, A., Zollhoefer, M., Tang, S.: Coap: Compositional articulated occupancy of people. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13201–13210 (2022)

39. Mihajlovic, M., Zhang, Y., Black, M.J., Tang, S.: Leap: Learning articulated occupancy of people. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10461–10471 (2021)

40. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99–106 (2021)

41. Moreau, A., Song, J., Dhamo, H., Shaw, R., Zhou, Y., Pérez-Pellitero, E.: Human gaussian splatting: Real-time rendering of animatable avatars. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 788– 798 (2024)

42. Müller, M., Heidelberger, B., Hennix, M., Ratclif, J.: Position based dynamics. Journal of Visual Communication and Image Representation 18(2), 109–118 (2007)

43. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG) 41(4), 1–15 (2022)

44. Park, J.J., Florence, P., Straub, J., Newcombe, R., Lovegrove, S.: Deepsdf: Learning continuous signed distance functions for shape representation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 165– 174 (2019)

45. Pavlakos, G., Choutas, V., Ghorbani, N., Bolkart, T., Osman, A.A., Tzionas, D., Black, M.J.: Expressive body capture: 3d hands, face, and body from a single image. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10975–10985 (2019)

46. Peng, S., Dong, J., Wang, Q., Zhang, S., Shuai, Q., Zhou, X., Bao, H.: Animatable neural radiance fields for modeling dynamic human bodies. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14314–14323 (2021)

47. Peng, S., Zhang, Y., Xu, Y., Wang, Q., Shuai, Q., Bao, H., Zhou, X.: Neural body: Implicit neural representations with structured latent codes for novel view synthesis of dynamic humans. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9054–9063 (2021)

48. Pumarola, A., Corona, E., Pons-Moll, G., Moreno-Noguer, F.: D-nerf: Neural radiance fields for dynamic scenes. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10318–10327 (2021)

49. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical image computing and computer-assisted intervention. pp. 234–241. Springer (2015)

50. Saito, S., Huang, Z., Natsume, R., Morishima, S., Kanazawa, A., Li, H.: Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 2304–2314 (2019)

51. Saito, S., Simon, T., Saragih, J., Joo, H.: Pifuhd: Multi-level pixel-aligned implicit function for high-resolution 3d human digitization. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 84–93 (2020)

52. Saito, S., Yang, J., Ma, Q., Black, M.J.: Scanimate: Weakly supervised learning of skinned clothed avatar networks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2886–2897 (2021)

53. Santesteban, I., Otaduy, M.A., Casas, D.: Snug: Self-supervised neural dynamic garments. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8140–8150 (2022)

54. Shao, Z., Wang, Z., Li, Z., Wang, D., Lin, X., Zhang, Y., Fan, M., Wang, Z.: Splattingavatar: Realistic real-time human avatars with mesh-embedded gaussian splatting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1606–1616 (2024)

55. Sitzmann, V., Martel, J., Bergman, A., Lindell, D., Wetzstein, G.: Implicit neural representations with periodic activation functions. Advances in neural information processing systems 33, 7462–7473 (2020)

56. Sitzmann, V., Zollhöfer, M., Wetzstein, G.: Scene representation networks: Continuous 3d-structure-aware neural scene representations. Advances in neural information processing systems 32 (2019)

57. Su, S.Y., Yu, F., Zollhöfer, M., Rhodin, H.: A-nerf: Articulated neural radiance fields for learning human shape, appearance, and pose. Advances in neural information processing systems 34, 12278–12291 (2021)

58. Tancik, M., Srinivasan, P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J., Ng, R.: Fourier features let networks learn high frequency functions in low dimensional domains. Advances in neural information processing systems 33, 7537–7547 (2020)

59. Tewari, A., Fried, O., Thies, J., Sitzmann, V., Lombardi, S., Sunkavalli, K., Martin-Brualla, R., Simon, T., Saragih, J., Nießner, M., et al.: State of the art on neural rendering. In: Computer graphics forum. vol. 39, pp. 701–727. Wiley Online Library (2020)

60. Tiwari, G., Sarafianos, N., Tung, T., Pons-Moll, G.: Neural-gif: Neural generalized implicit functions for animating people in clothing. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 11708–11718 (2021)

61. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)

62. Wang, L., Zhao, X., Sun, J., Zhang, Y., Zhang, H., Yu, T., Liu, Y.: Styleavatar: Real-time photo-realistic portrait avatar from a single video. In: ACM SIGGRAPH 2023 Conference Proceedings. pp. 1–10 (2023)

63. Wang, S., Mihajlovic, M., Ma, Q., Geiger, A., Tang, S.: Metaavatar: Learning animatable clothed human models from few depth images. Advances in Neural Information Processing Systems 34, 2810–2822 (2021)

64. Wang, S., Schwarz, K., Geiger, A., Tang, S.: Arah: Animatable volume rendering of articulated human sdfs. In: European conference on computer vision. pp. 1–19. Springer (2022)

65. Wang, W., Ho, H.I., Guo, C., Rong, B., Grigorev, A., Song, J., Zarate, J.J., Hilliges, O.: 4d-dress: A 4d dataset of real-world human clothing with semantic annotations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 550–560 (2024)

66. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13(4), 600–612 (2004)

67. Weng, C.Y., Curless, B., Srinivasan, P.P., Barron, J.T., Kemelmacher-Shlizerman, I.: Humannerf: Free-viewpoint rendering of moving people from monocular video. In: Proceedings of the IEEE/CVF conference on computer vision and pattern Recognition. pp. 16210–16220 (2022)

68. Xiu, Y., Yang, J., Cao, X., Tzionas, D., Black, M.J.: Econ: Explicit clothed humans optimized via normal integration. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 512–523 (2023)

69. Xiu, Y., Yang, J., Tzionas, D., Black, M.J.: Icon: Implicit clothed humans obtained from normals. In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 13286–13296. IEEE (2022)

70. Yu, Z., Cheng, W., Liu, X., Wu, W., Lin, K.Y.: Monohuman: Animatable human neural field from monocular video. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16943–16953 (2023)

71. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586–595 (2018)

72. Zheng, S., Zhou, B., Shao, R., Liu, B., Zhang, S., Nie, L., Liu, Y.: Gps-gaussian: Generalizable pixel-wise 3d gaussian splatting for real-time human novel view synthesis. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 19680–19690 (2024)

73. Zheng, Z., Huang, H., Yu, T., Zhang, H., Guo, Y., Liu, Y.: Structured local radiance fields for human avatar modeling. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 15893–15903 (2022)

74. Zheng, Z., Yu, T., Liu, Y., Dai, Q.: Pamir: Parametric model-conditioned implicit representation for image-based human reconstruction. IEEE transactions on pattern analysis and machine intelligence 44(6), 3170–3184 (2021)

75. Zheng, Z., Zhao, X., Zhang, H., Liu, B., Liu, Y.: Avatarrex: Real-time expressive full-body avatars. ACM Transactions on Graphics (TOG) 42(4), 1–19 (2023)

76. Zielonka, W., Bagautdinov, T., Saito, S., Zollhöfer, M., Thies, J., Romero, J.: Drivable 3d gaussian avatars. In: 2025 International Conference on 3D Vision (3DV). pp. 979–990. IEEE (2025)

## A Supplementary Material

In this supplementary material, we provide additional implementation details and experimental results to complement the main paper. We begin by presenting the detailed network architecture in Sec. A.1. Following that, we elaborate on our two-stage training strategy in Sec. A.2, showcase more avatar results in Sec. A.3, and present additional experiments including ablation studies and eficiency comparisons in Sec. A.4. We further provide fine-grained component ablations and temporal consistency evaluations in Sec. A.5. Finally, we discuss the limitations of our approach and outline directions for future work in Sec. A.6.

## A.1 Network Architecture

StyleUNet Architecture. Our deformation network Ψ adopts a StyleUNet [62] architecture with input channel dimension $C _ { i n } = 9$ (position map $p _ { \tau } \colon 3$ channels, velocity map $\varDelta p _ { \tau } : 3$ channels, previous deformation $\varDelta \mu _ { t - 1 } : 3$ channels). The encoder $\varPsi _ { \mathrm { e n c } }$ consists of four downsampling blocks, producing a 4-level feature pyramid with channels {64, 128, 256, 512} at resolutions $\{ 5 1 2 \times 5 1 2 , 2 5 6 \times$ 256, $1 2 8 \times 1 2 8 , 6 4 \times 6 4 7$ . Each frame τ in the temporal window is processed independently through the encoder. The decoder $\varPsi _ { \mathrm { d e c } }$ employs upsampling blocks with skip connections from the encoder to predict Gaussian deformations $\varDelta G _ { t } =$ $\{ \varDelta \mu _ { t } , \varDelta q _ { t } , \varDelta s _ { t } , \varDelta \alpha _ { t } , \varDelta c _ { t } \}$

Motion-Adaptive Temporal Aggregation (MATA). The motion embedding network $\varPhi _ { m }$ employs convolutional layers to project motion magnitude maps $\bar { V _ { \tau } }$ into feature space matching Level 4 dimension (512 channels). At the deepest level $( L = 4 )$ , causal self-attention with 8 heads aggregates motionenhanced features $\{ \tilde { f } _ { \tau } ^ { L } \} _ { \tau = t - n + 1 } ^ { t }$ to produce geometric features. At shallower levels $( l \in \{ 1 , 2 , 3 \} )$ , lightweight temporal convolutions provide eficient aggregation.

Memory Bank and State Stream. The memory bank $\mathcal { M } _ { t } = \{ h _ { t - m } , \ldots , h _ { t - 1 } \}$ stores $m = 1 0$ most recent temporal states at Level 4 resolution $( 6 4 \times 6 4 \times 5 1 2 )$ Cross-attention fuses geometric features from MATA with historical states from the memory bank, using 8 attention heads.

## A.2 Training Strategy

We employ a two-stage optimization strategy. Stage 1: Warm-up phase without autoregressive connections by setting all $\varDelta \mu _ { t - 1 } = 0$ and initializing memory bank $\mathcal { M } _ { t } = h _ { 1 }$ with loss $\mathcal { L } _ { \mathrm { s t a g e } _ { 1 } } = \mathcal { L } _ { \mathrm { r e n d e r } }$ . This establishes stable feature extraction before introducing temporal dependencies. Stage 2: Full autoregressive training with geometric stream (using predicted $\varDelta \mu _ { t - 1 } )$ and state stream (updating memory bank with $h _ { t } )$ enabled: ${ \mathcal { L } } _ { \mathrm { s t a g e } _ { 2 } } = { \mathcal { L } } _ { \mathrm { r e n d e r } } + \lambda _ { \mathrm { f e a t } } { \mathcal { L } } _ { \mathrm { f e a t } } + \lambda _ { \mathrm { p r e d } } { \mathcal { L } } _ { \mathrm { p r e d } } +$ $\lambda _ { \mathrm { s m } } \mathcal { L } _ { \mathrm { s m } }$ . We use Adam optimizer with learning rate $6 \times 1 0 ^ { - 4 }$ . Loss weights: $\lambda _ { \mathrm { { r e n d e r } } } = 1 , \lambda _ { \mathrm { { s i m } } } = 0 . 2 , \lambda _ { \mathrm { { l p i p s } } } = 0 . 5 , \lambda _ { \mathrm { { f e a t } } } = 0 . 0 1 , \lambda _ { \mathrm { { p r e d } } } = 0 . 1 , \lambda _ { \mathrm { { s m } } } = 0 . 0 5$

## A.3 Additional Avatar Results

Figure 6 showcases additional avatar reconstruction results across diferent subjects and motion sequences from our dataset. Our method successfully captures diverse clothing styles including loose jackets, flowing dresses, and multi-layer garments, demonstrating robust generalization across various scenarios. The results exhibit temporally coherent cloth dynamics with natural momentum effects, validating the efectiveness of our dual-stream autoregressive approach.

## A.4 Additional Experiments

Ablation on Temporal Window and Memory Bank Size. We investigate the joint impact of temporal window size n (in the geometric stream) and memory bank size m (in the state stream) on reconstruction quality and training eficiency. We evaluate three configurations: $( n = 5 , m = 5 ) , ( n = 1 0 , m = 1 0 )$ (our default), and $( n = 2 0 , m = 2 0 )$ on subject $\mathrm { ^ { 4 2 } J D \mathrm { - D R E S S \_ 0 0 1 7 5 \_ O u t e r \_ 2 ^ { \prime \prime } } }$ Table 3 presents the quantitative comparison.

Table 3: Ablation on temporal window and memory bank size. Comparison of diferent configurations where both temporal window size n and memory bank size m are varied jointly. Our default setting uses $n = 1 0 , m = 1 0$
<table><tr><td> $( n , m )$  PSNR↑</td><td>SSIM↑ LPIPS↓ Training (h)↓</td></tr><tr><td>(5,5) 30.2421 0.9791</td><td>0.0331 20</td></tr><tr><td>(10,10) 30.9515 0.9798</td><td>0.0311 28</td></tr><tr><td>(20, 20) 30.8193 0.9801</td><td>0.0314 35</td></tr></table>

The results show that $( n = 1 0 , m = 1 0 )$ achieves the best balance across rendering quality and computational eficiency. While $( n = 1 0 , m = 1 0 )$ and $( n = 2 0 , m = 2 0 )$ demonstrate comparable rendering performance, the larger configuration substantially increases training time without meaningful quality improvements. The smaller configuration $( n = 5 , m = 5 )$ reduces training time but slightly degrades rendering quality due to insuficient temporal context— both the geometric stream lacks suficient motion history to capture dynamics patterns, and the state stream has limited historical states for retrieving relevant temporal information. The balanced configuration $( n = 1 0 , m = 1 0 )$ provides suficient historical information for both streams to efectively model cloth dynamics, validating our choice as the optimal trade-of between temporal modeling capacity and computational cost.

Rendering Eficiency Comparison. We compare the rendering eficiency of our method against baseline approaches. Table 4 reports frames per second (FPS) measurements at $9 4 0 \times 1 2 8 0$ resolution on an NVIDIA RTX 4090 GPU. All methods use their oficial implementations with recommended settings.

![](images/b0231feb7dd5ada531961bbddc092d7cee3f84c4219b9421db1800d55eece8a5.jpg)

Fig. 6: Avatar gallery. Additional results demonstrating our method’s capability to reconstruct diverse avatars with diferent clothing styles and motion patterns. Our dual-stream architecture captures realistic cloth dynamics including momentum-driven motion and natural settling behaviors.

Table 4: Rendering eficiency comparison. FPS at 940 × 1280 resolution on RTX 4090.
<table><tr><td></td><td>HumanNeRF [67]</td><td>GaussianAvatar [17]</td><td>Animatable-GS [31]</td><td>Ours</td></tr><tr><td>FPS</td><td>0.18</td><td>35</td><td>13</td><td>10</td></tr></table>

Our method achieves fast rendering performance while providing superior rendering quality and temporal coherence through dual-stream autoregressive modeling. The additional computational cost from temporal window processing and memory bank retrieval is modest compared to the significant improvement in temporal consistency.

Impact of Camera Views. We investigate the robustness of our method under varying numbers of training views. To systematically evaluate the impact of camera view count, we artificially render diferent numbers of camera viewpoints (4, 8, and 16 views) using the ground truth texture data provided by 4D-DRESS. We evaluate on subject “4D-DRESS\_00135\_Outer\_2". Table 5 presents quantitative results, and Figure 7 shows qualitative comparisons.

Table 5: Impact of camera views. Quantitative evaluation with diferent numbers of training views.
<table><tr><td>Views</td><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td></tr><tr><td>4</td><td>30.5542</td><td>0.9791 0.0318</td></tr><tr><td>8</td><td>31.0515</td><td>0.9805 0.0313</td></tr><tr><td>16</td><td></td><td>31.2893 0.9817 0.0296</td></tr></table>

![](images/3d4425215b21aa1d591f1a37b2962169cc54e7fdb00e478a6789507a796b73eb.jpg)  
4 Views

![](images/ad8ef7f0397b43dea2805337a73449de845a71773dc9ee37741a3ed85b669a41.jpg)  
8 Views

![](images/80e97bb99353dc142fc4fc71ed7531a5128f8f2c8dc40a03c31a91a6dc582b4f.jpg)  
16 Views

![](images/7ed2c2618c1fc5c5483f0d30ad32c2640184765c0e4af5b3dddfbeda294f8ff4.jpg)  
GT  
Fig. 7: Impact of camera views. Visual comparison of reconstruction quality with diferent numbers of training views.

The results demonstrate that our method achieves robust and consistent performance across diferent camera configurations. Our dual-stream temporal modeling efectively captures cloth dynamics even with limited observational constraints, as the explicit temporal causality (through autoregressive geometric propagation and memory-based state retrieval) provides strong regularization that compensates for sparse multi-view supervision. This makes the method practical for real-world capture scenarios with sparse camera arrays.

## A.5 More Experiments

Quantitative Comparison with Temporal Consistency. In addition to PSNR and SSIM, we report Fréchet Video Distance (FVD) to evaluate temporal consistency. Table 6 compares our method with HumanNeRF, GaussianAvatar, Animatable-GS, MMLPs, and D3GA. Our method achieves the best rendering quality and the lowest FVD, indicating substantially improved temporal coherence.

Table 6: Quantitative comparison with FVD. Lower FVD indicates better temporal consistency.
<table><tr><td>Method</td><td>PSNR↑ SSIM↑ FVD↓</td></tr><tr><td>HumanNeRF</td><td>20.0567 0.9121 20.24</td></tr><tr><td>GaussianAvatar</td><td>26.2311 0.9575</td></tr><tr><td>Animatable-GS</td><td>12.34 29.5138 0.9724 6.17</td></tr><tr><td>MMLPs (CVPR 2025)</td><td>29.6843 0.9731 4.93</td></tr><tr><td>D3GA (3DV 2025)</td><td>29.7215 0.9728</td></tr><tr><td>Ours</td><td>3.38 31.0765 0.9839 0.31</td></tr></table>

Fine-Grained Component Ablation. We further evaluate individual design choices on the FD split. Table 7 shows that removing previous deformation feedback, velocity maps, causal attention, motion embeddings, or the state stream degrades both image quality and temporal consistency. Replacing the state stream with a long skeletal pose history remains substantially inferior, confirming that pose history cannot substitute for geometric feature memory. Extending the deformation-conditioning window beyond the immediate previous deformation yields negligible gain, supporting the single-step autoregressive design because $\varDelta \mu _ { t - 1 }$ already carries history through the autoregressive chain.

## A.6 Limitations and Future Work

Limitations. Our approach operates in a per-subject manner, requiring separate training for each clothed human, which limits scalability when dealing with diverse individuals and garment types. Additionally, the UV parameterization assumes continuous surface topology and cannot naturally handle clothing tearing, cutting, or topological changes. The autoregressive nature may also accumulate small errors over very long sequences, though our adaptive regularization strategy provides stabilization.

Table 7: Fine-grained component ablation on the FD split.
<table><tr><td>Variant</td><td>PSNR↑</td><td>SSIM↑1</td><td>FVD↓</td></tr><tr><td>Full Model</td><td>30.6723</td><td>0.9739</td><td>0.23</td></tr><tr><td> $\mathrm { w } / \mathrm { o } \ \varDelta \mu _ { t - 1 }$ </td><td>24.2694</td><td>0.9557</td><td>12.76</td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  velocity map  $\varDelta p$ </td><td>29.8436</td><td>0.9682</td><td>1.87</td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  both (position map only)</td><td>23.8318</td><td>0.9498</td><td>13.83</td></tr><tr><td>Bidirectional attn (no causal)</td><td>28.3341</td><td>0.9648</td><td>5.31</td></tr><tr><td>Direct cross-attn on prev. frame</td><td>29.1247</td><td>0.9661</td><td>2.14</td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  motion embedding</td><td>30.0153</td><td>0.9703</td><td>1.94</td></tr><tr><td> $\mathrm { w } / $  long pose hist.  $( \mathrm { w } / \mathrm { o }$  State Stream)</td><td>28.3174</td><td>0.9637</td><td>6.86</td></tr><tr><td> $\varDelta \mu _ { t - 1 } \ \mathrm { ( o u r s ) }$ </td><td>30.6720</td><td>0.9739</td><td>0.23</td></tr><tr><td> $\varDelta \mu _ { t - 1 : t - 3 }$ </td><td>30.7161 0.9741</td><td></td><td>0.25</td></tr><tr><td> $\varDelta \mu _ { t - 1 : t - 5 }$ </td><td>30.5247</td><td>0.9733</td><td>0.29</td></tr></table>

Future Work. Several directions could extend our framework. Exploring generalizable models that learn shared priors across subjects could eliminate persubject training while preserving personalized detail. Furthermore, extending the representation to handle topological changes would enable modeling of garment damage and complex interactions. The dual-stream principle is also conceptually transferable to other deformation-based dynamic representations, which we leave as future work.