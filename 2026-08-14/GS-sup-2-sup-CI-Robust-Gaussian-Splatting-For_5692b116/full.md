# GS<sup>2</sup>CI: Robust Gaussian Splatting For

# Snapshot Compressive Imaging via Large Vision Model Priors

Yanming Yang, Chenxi Song, Ping Wang, Xin Yuan, and Chi Zhang

Abstract—Snapshot Compressive Imaging (SCI) offers an efficient solution for high-speed video acquisition and, under exposure-time camera–scene relative motion, multi-view scene capture by compressing temporal or spatial information into a single 2D measurement. While recent studies have explored SCI for 3D scene reconstruction, existing methods struggle with significant challenges due to information loss, limited viewpoint diversity, and the computational burden of jointly optimizing 3D representations and camera poses. In this work, we propose a novel framework that reconstructs high-quality 3D scenes from a single SCI measurement by leveraging 3D Gaussian Splatting (3DGS) and the powerful priors of large-scale vision foundation models (VFMs). Our primary reconstruction combines measurementderived 3D VFM initialization with SCI-aware Gaussian optimization. After coarse-stage convergence, an auxiliary 2D VFM provides pseudo-view supervision at synthesized viewpoints for local appearance refinement. To further address the instability caused by ambiguous SCI supervision during 3DGS optimization, we introduce Opacity-Guided Splitting and Growth Regulation (OSGR), an SCI-specific densification strategy that augments split candidates using local opacity statistics, discourages losscompensating opacity inflation through mean-opacity regulation, and bounds representation growth with explicit candidate-ratio and Gaussian-count constraints. Extensive experiments across multiple benchmarks demonstrate that our method achieves the strongest overall performance, combining leading reconstruction quality and robustness to viewpoint variation with competitive computational efficiency. Our code is available at https://github.com/Westlake-AGI-Lab/GS2CI.git.

Index Terms—Snapshot Compressive Imaging, 3D Gaussian Splatting.

## I. INTRODUCTION

NAPSHOT Compressive Imaging (SCI) [1] is a cutting-S edge technique designed for high-speed data acquisition by compressing temporal information into a single 2D image. This is accomplished by modulating the incoming light using a sequence of specially designed 2D masks during the exposure process, allowing multiple temporal frames to be encoded into a single snapshot. A dedicated decoder is then required to reconstruct the original video frames from the compressed measurement. SCI provides a compact and efficient solution for capturing dynamic scenes, significantly reducing storage requirements and hardware complexity. These advantages make SCI attractive for high-speed video recording, where conventional high-speed cameras are costly and impose substantial data-throughput and storage demands.

Most existing approaches, whether model-based [1] or learning-based [2]–[9], are therefore designed with a videodecoding perspective: they aim to recover a sequence of 2D frames under limited inter-frame viewpoint variation. As a result, they struggle to generalize when the SCI measurement encodes multi-view inputs with wide baseline disparities. This performance drop can be attributed to two factors: (i) the use of relatively small and low-diversity training datasets for SCI decoders, such as DAVIS [10], and (ii) the lack of explicit modeling of essential 3D properties, including depth, occlusion, and view-dependent appearance. Consequently, although these decoders perform well on sequences with modest motion and viewpoint continuity, their effectiveness degrades under large viewpoint changes or discontinuous observations, leading to temporally and spatially inconsistent reconstructions.

During one sensor exposure, video SCI sequentially applies coding masks to temporally varying observations. When this variation is induced by rigid camera–scene relative motion, the latent frames correspond to different viewpoints and are multiplexed into one sensor readout. A single SCI camera can therefore encode multi-view information without simultaneous capture by a camera array, while the available viewpoint span is determined by the relative motion completed during the exposure. As illustrated in Fig. 1, this acquisition model provides geometric cues with simpler hardware, yet existing methods largely treat SCI as an extreme video compression scheme and do not fully exploit its underlying 3D structure.

These observations motivate a shift in perspective, from pure video reconstruction toward recovering a coherent underlying 3D scene that explains the SCI measurement. From this viewpoint, SCI can be formulated as a joint problem of compressed image decoding and 3D scene recovery, where explicit 3D priors act as consistency constraints across views or time. Recent methods follow this direction by adopting implicit or explicit 3D representations, such as Neural Radiance Fields (NeRF) [11] and 3D Gaussian Splatting (3DGS) [12], to recover spatially consistent scenes from severely underdetermined SCI inputs [13]–[16].

However, because SCI compresses multiple viewpoints through coded modulation and pixel-wise aggregation, it inevitably discards fine-grained view cues, making geometry and pose estimation particularly ill-posed. Existing 3D-SCI methods typically rely on jointly optimizing 3D scene representations and camera parameters, which is computationally intensive and sensitive to initialization. For example, NeRFbased approaches [14] often require tens of thousands of iterations per scene with costly gradient computations. When only a single SCI measurement is available, the optimization tends to overfit the modulated views and generalizes poorly to unseen viewpoints, undermining both the decoding of compressed images and the recovery of a stable 3D scene without strong priors or external guidance.

![](images/64e2f83a9ad9338d9f66d5b214b6bba0dc36471087353cbe93e2a3a5f7ee7bed.jpg)  
Fig. 1. Conceptual comparison between conventional multi-camera capture and video SCI acquisition. Under rigid camera–scene relative motion, one SCI camera sequentially modulates views observed at different instants and accumulates them into a single sensor measurement. Our method reconstructs the shared static scene and the corresponding camera trajectory from this measurement and the coding masks.

To overcome these limitations, we present a computationally efficient framework built on 3DGS [12], enhanced by vision foundation models trained on large-scale multi-view and 3D data. In our formulation, these foundation models provide strong priors on geometry, appearance, and camera motion, thereby regularizing the severely under-constrained SCI reconstruction problem. Our framework comprises a primary reconstruction stage and an auxiliary refinement stage. In the primary stage, a 3D VFM [17] uses measurement-derived proxy views to initialize camera poses and sparse scene geometry, followed by SCI-aware Gaussian optimization. After coarse-stage convergence, the auxiliary stage uses a frozen 2D VFM to construct pseudo-view targets at synthesized poses for local appearance refinement.

While 3DGS offers clear advantages in rendering speed and training efficiency, applying it directly to SCI remains challenging because a single multiplexed measurement constrains only the encoded sum of the rendered views rather than each latent view independently. Different combinations of geometry, appearance, and opacity can therefore produce similar measurement values. During joint scene-and-pose optimization, the optimizer may reduce part of the multiplexed residual by increasing the opacity of individual Gaussians before their spatial support is sufficiently resolved, producing local opacity peaks that are not explicitly addressed by the original adaptive density control mechanism [12].

To address this issue, we propose Opacity-Guided Splitting and Growth Regulation (OSGR), a densification strategy tailored to SCI. OSGR uses local opacity contrast to augment the split-candidate set, allowing concentrated responses to be reparameterized by smaller child Gaussians with distinct spatial support. The mean-opacity term softly discourages renewed opacity inflation, while the candidate-ratio and Gaussian-count constraints bound representation growth. Together, they guide refinement and stabilize densification under SCI supervision.

Extensive experiments on multiple benchmarks show that our method achieves the strongest overall performance by combining leading visual fidelity and novel-view synthesis quality with competitive reconstruction speed. Moreover, it demonstrates strong robustness to large viewpoint changes and challenging SCI settings. The main contributions of this paper are summarized as follows:

• We propose a 3D reconstruction framework from a single SCI measurement that combines measurement-derived 3D VFM geometry initialization with SCI-aware Gaussian optimization, followed by auxiliary 2D VFM pseudoview supervision for local appearance refinement.

• We develop OSGR, a customized densification strategy for 3D reconstruction from SCI, which stabilizes the joint optimization of camera poses and scene structure under severely under-constrained SCI supervision.

• Extensive experiments across multiple benchmarks demonstrate the strongest overall performance, with leading reconstruction quality and competitive efficiency. The code is publicly available.

## II. RELATED WORK

a) Video SCI Reconstruction: SCI aims to recover a video sequence from a single compressed measurement, significantly reducing acquisition cost and latency [18]. Due to its ill-posed nature, strong priors are required to constrain the solution space. Traditional methods adopt iterative optimization with handcrafted regularizers [1], [19], [20], but often suffer from slow convergence and poor scalability to high-resolution or dynamic scenes. Recent work has shifted to learningbased methods that directly map compressed measurements to videos, leveraging neural networks [3], [21], [22] to better model spatiotemporal dependencies. However, most learningbased methods rely on synthetic data and neglect 3D scene geometry, leading to geometric inconsistencies across views.

b) Geometric and Generative Priors: Recent 3D reconstruction methods can be broadly grouped into two directions. The first direction explores 3D VFMs, which learn to infer depth, camera poses, and scene geometry from large-scale image data without explicit SfM supervision [23]–[29]. A representative example is VGGT [17], which jointly predicts pose and geometry in a feed-forward end-to-end framework. The second direction leverages generative priors, particularly diffusion models, to facilitate sparse-view reconstruction [30]–[34]. These methods refine imperfect renderings with pretrained or distilled diffusion models and update the 3D representation, improving reconstruction quality and view consistency.

## III. METHOD

We present a framework for SCI-based 3D reconstruction by integrating VFMs with a 3DGS-based SCI reconstruction pipeline. The primary reconstruction uses a 3D VFM [17] to initialize geometry and camera poses before SCI-aware Gaussian optimization. An auxiliary fine stage subsequently applies 2D VFM pseudo-view supervision at synthesized viewpoints for local appearance refinement while keeping the camera trajectory estimated by the primary reconstruction fixed.

The remainder of this section is organized as follows. We begin by introducing the preliminaries of our method in Section III-A. We then describe the initial geometry and pose estimation using the 3D VFM in Section III-B. The primary SCI reconstruction and auxiliary pseudo-view refinement are subsequently presented in Section III-C and Section III-D, respectively, followed by our improved densification strategy tailored for SCI in Section III-E. An overview of the entire framework is illustrated in Fig. 2.

## A. Preliminaries

a) Snapshot Compressive Imaging Model: In a typical SCI [18] system, the compressed image Y is formed by modulating the scene through N binary masks M during the exposure time. The sensor records the accumulated exposure over these masks. The image formation process can be mathematically represented as:

$$
\mathbf { Y } = \sum _ { i = 1 } ^ { N } \mathbf { X } _ { i } \odot \mathbf { M } _ { i } + \mathbf { Z } ,\tag{1}
$$

where $Y$ is the compressed snapshot, $X _ { i }$ represents the i-th modulated image corresponding to each exposure interval, $M _ { i }$ is the binary modulation mask, Z is the measurement noise and ⊙ denotes Hadamard product. The number of masks N defines the temporal compression ratio (CR), which determines how many frames are compressed into a single snapshot. The pixel values in the binary mask are typically randomized or designed with a fixed overlap ratio. For the static-scene setting considered in this work, $\mathbf { X } _ { i }$ denotes the scene observed at camera pose T during the i-th mask interval. Multi-view parallax is produced by camera–scene relative motion within the sensor exposure; the sequentially modulated observations are accumulated into one sensor readout. If the relative pose remains fixed, the latent frames share the same viewpoint and the measurement contains no multi-view parallax.

## B. Initial Geometry Estimation

A single SCI measurement Y lacks the view-resolved RGB observations required by conventional Structure-from-Motion methods such as COLMAP [35] and by VGGT [17]. We therefore construct geometry-oriented proxy views on the original image grid from Y and the actual coding masks $\bar { \mathcal { M } } \ = \ \bar { \{ \bf M } _ { i }  \} _ { i = 1 } ^ { \bar { N } }$ . Let $\begin{array} { r c l } { a ( \mathbf { p } ) } & { = } & { \sum _ { i } \mathbf { M } _ { i } ( \mathbf { p } ) } \end{array}$ and $\mu _ { \mathcal { M } } ~ =$ $( H W ) ^ { - 1 } \textstyle \sum _ { \mathbf { p } } a ( \mathbf { p } )$ denote the per-pixel and mean measurement multiplicities, respectively.

For standard low-multiplicity acquisitions $( \mu _ { \mathcal { M } } < \tau _ { \mu }$ , with $\tau _ { \mu } = 4$ fixed throughout), we use Energy-Normalized Initialization (ENI) to compensate for first-order spatial exposure variation [3] before assigning the normalized, mask-supported samples to individual proxy views:

$$
\begin{array} { r l } & { \mathbf { \overline { { Y } } } ( \mathbf { p } ) = \left\{ \mathbf { Y } ( \mathbf { p } ) / a ( \mathbf { p } ) , \begin{array} { l l } { a ( \mathbf { p } ) > 0 , } \\ { \mathbf { 0 } , } \end{array} \right. } & { a ( \mathbf { p } ) = 0 , } \\ & { \mathbf { \widetilde { X } } _ { i } ^ { \mathrm { E N I } } = \mathcal { I } \left( \mathbf { \overline { { Y } } } \odot \mathbf { M } _ { i } ; \mathbf { M } _ { i } \right) , \qquad i = 1 , \dots , N . } \end{array}\tag{2}
$$

Here, I completes each proxy by retaining the normalized measurement samples selected by M<sub>i</sub> and interpolating the locations omitted by that mask. Linear interpolation is used inside the convex hull of the selected samples, while the remaining locations are filled by nearest-neighbor interpolation.

However, when increased compression or mask overlap yields $\mu _ { \mathcal { M } } \ge \tau _ { \mu }$ , ENI cannot separate the superposed view contributions. We therefore use Adaptive Mask-Decoding Initialization (AMDI), which estimates the N proxy views from local coding-pattern diversity through adaptive, regularized least squares. Both constructions provide geometry-oriented VGGT inputs rather than recovered frames or photometric supervision. The supplementary material details how measurement multiplicity determines the routing rule and provides the complete AMDI solver. Writing the two constructions as $\mathcal { P } _ { \mathrm { E N I } }$ and $\mathcal { P } _ { \mathrm { A M D 1 } }$ , we define the proxy sequence as:

$$
\widetilde { \mathcal { X } } = \left\{ \begin{array} { l l } { \mathcal { P } _ { \mathrm { E N I } } ( \mathbf { Y } , \mathcal { M } ) , } & { \mu _ { \mathcal { M } } < \tau _ { \mu } , } \\ { \mathcal { P } _ { \mathrm { A M D I } } ( \mathbf { Y } , \mathcal { M } ) , } & { \mu _ { \mathcal { M } } \geq \tau _ { \mu } . } \end{array} \right.\tag{3}
$$

![](images/47ce2abe74062a22387bdd20bfcc5e944738c1ed2d124f0c1c5aa2d6aeb703b6.jpg)  
Fig. 2. Pipeline overview of our 3D reconstruction framework from a single SCI measurement. Measurement-derived proxy views are processed by a 3D VFM [17] to initialize scene geometry and camera poses, followed by SCI-aware 3DGS [12] optimization augmented by our OSGR. After coarse-stage convergence, an auxiliary fine stage uses a frozen 2D VFM to construct pseudo-view targets at synthesized poses for local appearance refinement.

The resulting proxy sequence is supplied to the frozen VGGT model to estimate a 3D point set and camera parameters. Bundle adjustment (BA) is preferred when the proxyderived correspondences provide reliable constraints, because it jointly refines the 3D structure and camera parameters to improve global multi-view consistency. Since these indirect correspondences can be spatially uneven, we verify the post-BA geometry before using it to initialize 3DGS and retain a supported estimate when BA is underconstrained. We denote this complete VGGT-based initialization as follows:

$$
\left( \mathcal { Q } ^ { \star } , \{ \mathbf { K } _ { i } ^ { \star } , \mathbf { T } _ { i } ^ { ( 0 ) } \} _ { i = 1 } ^ { N } \right) = \Phi _ { \mathrm { i n i t } } \left( \widetilde { \mathcal { X } } \right) .\tag{4}
$$

Here, $\mathcal { Q } ^ { \star }$ is the selected 3D point set, $\mathbf { K } _ { i } ^ { \star }$ is the camera intrinsic matrix, and $\mathbf { T } _ { i } ^ { ( 0 ) }$ is the initial camera pose for view i. The reliability criteria are fixed a priori and applied consistently across all scenes. The treatment of the geometry before and after refinement, together with the handling of degenerate initializations, is described in the supplementary material.

## C. Pose Refinement and Scene Optimization

After obtaining the initial point cloud from the 3D VFM [17], we jointly optimize the Gaussian scene and camera poses. Although the 3D VFM provides a strong initialization, small pose errors can hinder precise reconstruction. We therefore use a hybrid objective that enforces fidelity to the observed SCI measurement and penalizes mean Gaussian opacity. With ${ \widehat { \bf Y } } ~ = ~ R ( { \mathcal G } , { \bf T } )$ denoting the rendered SCI measurement, the coarse loss is:

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } _ { \mathrm { c o a r s e } } = ( 1 - \lambda _ { \mathrm { s s i m } } ) { \mathcal { L } } _ { 1 } ( \widehat { \mathbf { Y } } , \mathbf { Y } ) + \lambda _ { \mathrm { s s i m } } { \mathcal { L } } _ { \mathrm { s s i m } } ( \widehat { \mathbf { Y } } , \mathbf { Y } ) } } \\ { ~ + \frac { \lambda _ { \mathrm { o p a c i t y } } } { G } \sum _ { i = 1 } ^ { G } \alpha _ { i } , } \end{array}\tag{5}
$$

Here, $R ( { \mathcal { G } } , \mathbf { T } )$ denotes the entire SCI imaging pipeline: the current Gaussians $\mathcal { G }$ are rendered using poses T, and the rendered views are then encoded according to Eq. (1). Moreover, $\alpha _ { i }$ denotes the opacity of the i-th Gaussian and G is the number of Gaussians at the current training step.

Each initial pose $\mathbf { T } _ { i } ^ { ( 0 ) } \in \mathrm { S E } ( 3 )$ is refined via a learnable adjustment $\Delta \mathbf { T } _ { i } \ \in \ \mathrm { S E } ( 3 )$ , yielding the final pose: $\mathbf { T } _ { i } \ =$ $\Delta \mathbf { \bar { T } } _ { i } \cdot \mathbf { T } _ { i } ^ { ( 0 ) }$ , where $\Delta { \bf T } _ { i }$ is optimized in the Lie algebra $\mathfrak { s e } ( 3 )$ Each adjustment is initialized as the identity transformation and jointly optimized with the Gaussian parameters under the coarse-stage objective in Eq. (5).

## D. Auxiliary 2D VFM Pseudo-view Refinement

After the primary coarse-stage reconstruction, we first render the current Gaussian representation at synthesized viewpoints and process each rendering with a frozen 2D VFM [30] to produce a pseudo-ground-truth appearance target for the corresponding viewpoint. Because these targets are generated independently by a 2D model, they may contain local appearance content that is inconsistent with the current 3D scene representation. To reduce the influence of such inconsistencies during optimization, we assign a small overall weight to the pseudo-view objective and modulate its photometric residuals using a soft support weight derived from Gaussian alpha accumulation. With the recovered camera trajectory fixed and the SCI consistency objective retained, this auxiliary supervision further refines the local appearance of the Gaussian representation.

We synthesize target poses from the refined camera trajectory $\{ \mathbf { T } _ { i } \} _ { i = 1 } ^ { N } .$ Given two adjacent anchor poses $\begin{array} { r l } { \mathbf { T } _ { a } } & { { } = } \end{array}$ $\left( \mathbf { R } _ { a } , \mathbf { t } _ { a } \right)$ and $\mathbf { T } _ { b } = ( \mathbf { R } _ { b } , \mathbf { t } _ { b } )$ , we interpolate their rotations on $\mathrm { S O ( 3 ) }$ and their translations linearly:

$$
\begin{array} { r l } & { \quad \mathbf { R } ( \alpha ) = \exp \left( \alpha \log \left( \mathbf { R } _ { b } \mathbf { R } _ { a } ^ { \mathsf { T } } \right) \right) \mathbf { R } _ { a } , } \\ & { \quad \mathbf { t } ( \alpha ) = ( 1 - \alpha ) \mathbf { t } _ { a } + \alpha \mathbf { t } _ { b } , } \\ & { \quad \mathbf { T } _ { \mathrm { i n t } } ( \alpha ) = \left[ \begin{array} { c c } { \mathbf { R } ( \alpha ) } & { \mathbf { t } ( \alpha ) } \\ { \mathbf { 0 } ^ { \mathsf { T } } } & { 1 } \end{array} \right] . } \end{array}\tag{6}
$$

where $\alpha \in ( 0 , 1 )$ . To generate the endpoint extrapolated poses, we extend the boundary motion by one additional step: the relative transform from the second pose to the first is applied once more before the first pose, and the relative transform from the penultimate pose to the last is applied once more after the last pose. The synthesized pose set is: $\mathcal { T } _ { \mathrm { s y n } } = \mathcal { T } _ { \mathrm { i n t } } \cup \mathcal { T } _ { \mathrm { e x t } }$

At each synthesized pose $\begin{array} { r l r } { \mathbf { T } _ { t } } & { { } \in } & { \mathcal { T } _ { \mathrm { s y n } } , } \end{array}$ the auxiliary renderer produces a pseudo-view and alpha map: $( \mathbf { X } _ { t } ^ { 0 } , \mathbf { A } _ { t } ^ { 0 } ) = \mathcal { R } _ { \operatorname { a u x } } ( \mathcal { G } , \mathbf { T } _ { t } )$ . After coarse-stage optimization is completed, we apply the frozen 2D VFM $\mathcal { F } _ { \psi }$ to obtain a refinement target: $\hat { \mathbf { X } } _ { t } = \mathcal { F } _ { \psi } ( \mathbf { X } _ { t } ^ { 0 } ; \tau _ { 0 } )$ , where $\tau _ { 0 }$ controls the refinement strength.

To construct the soft support weight, we additionally render the current fine-stage Gaussian representation at each synthesized pose as: $( \mathbf { X } _ { t } ^ { r } , \mathbf { A } _ { t } ^ { r } ) = \mathcal { R } _ { \operatorname { a u x } } ( \mathcal { G } , \mathbf { T } _ { t } )$ . We compute an alpha-based support score as: ${ \bf A } _ { t } ^ { \mathrm { s u p } } = \operatorname* { m i n } ( { \bf A } _ { t } ^ { 0 } , { \bf A } _ { t } ^ { r } )$ , where the pointwise minimum retains only the accumulated alpha supported both when the pseudo-ground-truth target is generated and at the current optimization step. We then map this score to a soft pixel-level weight:

$$
\mathbf { W } _ { t } ( \mathbf { p } ) = w _ { \operatorname* { m i n } } + ( 1 - w _ { \operatorname* { m i n } } ) \sigma \biggl ( \frac { \mathbf { A } _ { t } ^ { \operatorname* { s u p } } ( \mathbf { p } ) - \tau _ { v } } { \beta _ { v } } \biggr ) ,\tag{7}
$$

where $\sigma ( \cdot )$ is the sigmoid function, $\tau _ { v }$ is the support threshold, $\beta _ { v }$ controls the transition softness, and $w _ { \mathrm { m i n } }$ retains a residual contribution for pixels with low accumulated alpha. The resulting weight approaches one for high-support pixels and $w _ { \mathrm { m i n } }$ for low-support pixels. The corresponding hyperparameters are specified in Sec. IV-A. We apply this lightweight weighting only to the photometric component of the auxiliary pseudoview loss, while retaining a weak unweighted perceptual term. Let $d _ { t } ( \mathbf { p } ) = \ell _ { 1 } ( \mathbf { X } _ { t } ^ { r } ( \mathbf { p } ) , \hat { \mathbf { X } } _ { t } ( \mathbf { p } ) )$ denote the photometric error at pixel p:

$$
\mathcal { L } _ { \mathrm { s y n } } ^ { t } = \lambda _ { l 1 } \frac { \sum _ { \mathbf { p } } \mathbf { W } _ { t } ( \mathbf { p } ) d _ { t } ( \mathbf { p } ) } { \sum _ { \mathbf { p } } \mathbf { W } _ { t } ( \mathbf { p } ) + \epsilon } + \lambda _ { \mathrm { p e r } } \mathcal { D } _ { \mathrm { p e r } } \left( \mathbf { X } _ { t } ^ { r } , \hat { \mathbf { X } } _ { t } \right) ,\tag{8}
$$

where $\mathcal { D } _ { \mathrm { p e r } }$ denotes a perceptual distance and ϵ is a small constant for numerical stability. This weighting reduces the contribution of photometric residuals along rays with low accumulated alpha.

The final fine-stage objective is defined as:

$$
\mathcal { L } _ { \mathrm { f i n e } } = \mathcal { L } _ { \mathrm { c o a r s e } } + \lambda _ { \mathrm { s y n } } \frac { 1 } { | T _ { \mathrm { s y n } } | } \sum _ { \mathbf { T } _ { t } \in \mathcal { T } _ { \mathrm { s y n } } } \mathcal { L } _ { \mathrm { s y n } } ^ { t } ,\tag{9}
$$

where $\lambda _ { \mathrm { s y n } }$ controls the contribution of the auxiliary supervision at both interpolated and extrapolated pseudo-view poses.

## E. OSGR: Opacity-Guided Splitting and Growth Regulation

In the original 3DGS framework [12], adaptive density control (ADC) relies on duplicate and split operations to refine the Gaussian representation. While effective under standard multi-view supervision, these densification strategies become unstable in the SCI setting, where multiple views are compressed into a single measurement and the supervision is therefore severely under-constrained. In practice, standard ADC may densify weakly observed regions, causing abrupt opacity increases, photometric loss oscillations, and unstable joint scene-and-pose optimization.

To address this issue, we propose OSGR, a densification strategy tailored to SCI. OSGR incorporates local opacity statistics into candidate selection and regulates the global opacity distribution during optimization. It combines an opacity-guided splitting rule, a mean-opacity regulation term, and explicit candidate-ratio and Gaussian-count constraints.

Because a single SCI measurement constrains only the encoded sum of the rendered views, different combinations of geometry, appearance, and opacity can produce similar measurement values. The optimizer may consequently absorb part of the multiplexed residual by increasing the opacity of individual Gaussians, producing local opacity peaks during joint scene-and-pose optimization. OSGR uses these peaks as an additional signal for spatial refinement. For each Gaussian i, we compute the average opacity within its local neighborhood:

$$
\bar { \alpha } _ { i } = \frac { 1 } { K + 1 } \sum _ { j \in \mathcal { K } ( i ) \cup \{ i \} } \alpha _ { j } ,\tag{10}
$$

where $\kappa ( i )$ denotes the neighbors of the i-th Gaussian. Building upon the default split mechanism in standard ADC, a Gaussian with $\alpha _ { i } > \bar { \alpha } _ { i }$ is included in the additional split set. Splitting converts the locally concentrated opacity response into smaller child Gaussians with distinct spatial support. Their different projections across views and masks allow subsequent SCI optimization to redistribute the contribution among the children, while the child-opacity correction limits the immediate compositing change caused by the split. Meanopacity regulation discourages renewed opacity inflation, and the candidate-ratio and Gaussian-count constraints bound representation growth. The original ADC candidates remain active and are augmented by the opacity-guided candidates, as illustrated in Fig. 3.

In addition to the local splitting rule, the global meanopacity term discourages representation-wide opacity inflation during optimization, thereby providing a soft constraint on opacity values. The candidate-ratio budget limits additional candidates at each densification step, while the Gaussian-count constraint sets a hard cap on the total population.

Together, local opacity guides additional refinement, meanopacity regulation constrains opacity evolution, and the explicit growth constraints bound representation size, stabilizing densification under SCI supervision.

![](images/21c5600693029bc0a72d23943fa8ef9f628d84d43864cec7b5f4f50ad03a3f73.jpg)  
Fig. 3. Illustration of the proposed OSGR strategy. In addition to the scale- and screen-radius criteria in vanilla 3DGS ADC [12], OSGR identifies opacityinconsistent Gaussians using local neighborhood statistics and applies global opacity regulation during optimization. Best viewed by zooming in.

## IV. EXPERIMENT

To evaluate the effectiveness and generalizability of the proposed method, we conduct comprehensive experiments under diverse SCI settings on synthetic datasets and real data, covering variations in scene type, camera motion, and novel view synthesis scenarios.

## A. Experimental Settings

a) Datasets and Metrics: To ensure comprehensive evaluation, we test our method on both simulated and realworld SCI datasets. Following prior work [14], we first evaluate on simulated SCI measurements generated from widely used benchmarks: NeRF Synthetic [11], DeblurNeRF [36], DTU [37], LLFF [38] and DAVIS [10]. Unless otherwise specified, the simulated measurements use a compression ratio of 8 and a mask ratio of 0.25. For real-world evaluation, we use the real SCI data provided by SCINeRF [14]. We report SSIM, PSNR, and LPIPS as our main metrics.

b) Compared Baselines: We evaluate our method against a wide range of state-of-the-art SCI reconstruction approaches, including GAP-TV [20], PnP-FFDNet [8], PnP-FastDVDNet [9], and EfficientSCI [7]. We also compare with 3D-based state-of-the-art methods including SCINeRF [14] and SCIGS [13]. All experiments are evaluated under consistent settings for fair comparison.

c) Implementation Details: We implement our method in Nerfstudio [39] and use gsplat for differentiable 3DGS rasterization [12]. All experiments use one NVIDIA RTX 4090 with a fixed random seed of 42, and the hyperparameters remain unchanged across standard and extended evaluations. On this GPU, proxy construction and the complete VGGTbased geometry initialization take 4 minutes 45 seconds on average. Coarse training takes 53.67 minutes, while DiFix3D+ target generation and subsequent fine-stage optimization add roughly 10 minutes, bringing the total to approximately 68.4 minutes. A fresh SCINeRF [14] run under its official schedule requires 746.84 minutes (12.45 hours).

d) Optimization Protocol: Initialization and coarse stage. We apply the acquisition-aware proxy construction and VGGT-based geometry initialization described in Section III-B to every setting. The AMDI solver and the treatment of geometrically degenerate initializations are detailed in the supplementary material. The Gaussian representation and perview local SE(3) pose corrections are then jointly optimized with Adam for 20,000 iterations. The SCI consistency objective combines $\ell _ { 1 }$ and SSIM terms with weights of 0.9 and 0.1, while opacity regulation is weighted by 0.01. Pose gradients are accumulated over 25 iterations to stabilize camera refinement. The learning rates for pose corrections and Gaussian means decay exponentially from $1 0 ^ { - 2 }$ and $1 . 6 \times 1 0 ^ { - 4 }$ to $2 \times 1 0 ^ { - 5 }$ and $1 . 6 \times 1 0 ^ { - 6 }$ , respectively.

Densification. Densification is performed every 100 iterations between iterations 500 and 15,000. To constrain model growth, the Gaussian count at iteration 7,000 bounds all subsequent splitting and duplication, while opacity is reset every 3,000 iterations. OSGR augments the conventional gradient, scale, and screen-space radius cues with the proposed local opacity criterion. For duplication, opacity is revised following Revising-GS [40] to preserve transmittance; for splitting, we extend the same transmittance-preserving correction to the resulting child Gaussians.

Auxiliary fine stage. The fine stage is initialized from the final coarse-stage estimate and refines the Gaussian repre sentation for 3,000 iterations with fixed camera poses. Di-Fix3D+ [30] serves as the 2D VFM, with pseudo-view targets obtained through single-step inference at diffusion timestep t = 400. Both pseudo-view groups use an outer loss weight of 0.006 and combine $\ell _ { 1 }$ and perceptual terms with weights of 0.8 and 0.01. The alpha-based support weight is computed from the pointwise minimum of the target-construction and current-render alpha maps, with a support threshold of 0.25, transition softness of 0.08, and minimum weight of 0.10. For Table X, the evaluation poses are disjoint from those used for pseudo-view supervision.

## B. Results

a) SCI Reconstruction: Table I presents the quantitative comparison. Compared with EfficientSCI [7], the state-of-theart method for video sequence SCI, our approach achieves better multi-view consistency and higher reconstruction quality. In comparison to SCINeRF [14] and SCIGS [13], which represent the state-of-the-art in 3D modeling-based SCI, our method achieves better overall performance across most scenes. These results confirm the effectiveness of the proposed method for reconstructing high-quality 3D scenes from SCI measurements. Visual comparisons are shown in Fig. 4.

![](images/fdb2dc9be0b4887780266f4cf35159e53b0826aec2587cd11918cc37cce21736.jpg)  
Fig. 4. Qualitative evaluations of our method against state-of-the-art SCI decoding methods. Our results demonstrate the effectiveness of the proposed approach in restoring high-quality images from a single SCI measurement. The last row uses real SCI data provided by SCINeRF [14].

TABLE I  
Q C SCI R Q S S . W PSNR↑, SSIM↑, LPIPS↓. O ACHIEVES THE BEST OR SECOND-BEST PERFORMANCE ON ALMOST ALL SCENES AND METRICS, WITH PARTICULARLY NOTABLE GAINS IN PERCEPTUAL QUALITY (LPIPS) AND STRUCTURAL FIDELITY (PSNR/SSIM), DEMONSTRATING ITS EFFECTIVENESS FOR HIGH-FIDELITY SCI RECONSTRUCTION. BOLD AND UNDERLINED VALUES DENOTE THE BEST AND SECOND-BEST RESULTS, RESPECTIVELY.
<table><tr><td>Method</td><td>Vender Factory PSNR SSIM LPIPS PSNR SSIM LPIPS PSNR SSIM LPIPS PSNR SSIM LPIPS PSNR SSIM LPIPS PSNR SSIM LPIPS</td><td>Airplants</td><td>Hotdog</td><td>Tanabata</td><td>Cozy2room</td></tr><tr><td>GAP-TV</td><td colspan="5">20.00 0.3680 0.688024.05 0.5660 0.5150 22.850.4060 0.4990 22.35 0.7660 0.318020.42 0.4260 0.6250 21.770.4320 0.6030</td></tr><tr><td>PnP-FFDNet</td><td>28.70 0.9230 0.131031.75 0.8970 0.1140 27.79 0.9120 0.1820 29.00 0.9760 0.0510 29.17 0.9030 0.1190 28.980.8920 0.0980</td><td></td><td></td><td></td><td></td></tr><tr><td>FastDVDNet</td><td>29.680.9390 0.104032.530.9160 0.105028.180.9090 0.175029.930.9720 0.052029.73 0.9330 0.098030.190.9130 0.0790</td><td></td><td></td><td></td><td></td></tr><tr><td>EfficientSCI</td><td>33.17 0.9400 0.045032.870.9250 0.070030.13 0.94200.1120</td><td></td><td>30.75 0.9560 0.0460</td><td>32.30 0.9580 0.060031.470.9330 0.0480</td><td></td></tr><tr><td>SCINeRF</td><td>36.40 0.9840 0.029036.60 0.9630 0.0220</td><td>30.690.9330 0.0720</td><td>31.350.9870 0.0310</td><td></td><td></td></tr><tr><td>SCIGS</td><td>36.000.9640 0.0190</td><td></td><td>29.310.9370 0.0810</td><td>33.61 0.9630 0.037033.23 0.9490 0.0450 35.120.9580 0.0270</td><td>33.780.9200 0.0420</td></tr><tr><td></td><td></td><td>37.750.9650 0.0290</td><td>27.180.7270 0.3000</td><td></td><td></td></tr><tr><td>Ours</td><td colspan="5">38.52 0.9890 0.0098 40.040.9887 0.0156 31.36 0.9382 0.0688 31.480.97560.0217 37.78 0.98690.0093 34.200.9712 0.0482</td></tr></table>

b) Novel View Synthesis: We further evaluate novel-view synthesis from SCI inputs. As shown in Fig. 5, our method produces more coherent structures and sharper appearance than state-of-the-art SCINeRF [14] at unseen viewpoints. The visual comparison reflects the complete reconstruction pipeline, whereas Table X reports a component study of the auxiliary 2D VFM refinement, with all variants initialized from the same coarse-stage reconstruction.

## C. Additional Results on Challenging SCI Settings

To further evaluate the generalizability of our method beyond the standard benchmark settings, we consider additional static SCI scenes, complex measurements involving unbounded scenes and challenging camera trajectories, varying mask ratios, and high compression ratios. Dynamic-scene behavior is discussed separately in Sec. IV-E.

In addition to the state-of-the-art SCI-based 3D reconstruction method SCINeRF [14], we construct three controlled 3DGS-based variants within the same SCI reconstruction framework. SCI-3DGS, SCI-MCMC, and SCI-RevADC adopt the adaptive density control and associated optimization rules of the original 3DGS [12], Gaussian MCMC [41], and Revising-GS [40], respectively, whereas our method uses OSGR. All four Gaussian methods otherwise share the same SCI data-fidelity objective, VFM initialization, and camerapose refinement protocol. This construction distinguishes the contribution of OSGR from that of the shared reconstruction framework. We additionally report a separate coarsestage novel-view synthesis comparison using the same set of methods in Table II, where our method achieves the best performance across all reported metrics.

All controlled Gaussian comparisons in this subsection, including the novel-view comparison above, use only the firststage reconstructions. The second-stage pseudo-view targets are generated from and conditioned on the corresponding first-

![](images/4d0ba7460d3f9b6fe1bc5d8474b8ee41008c4376e34513dfb6bf5eb692d3e522.jpg)  
Fig. 5. Comparison of novel view synthesis results with the state-of-the-art 3D-based SCI method SCINeRF [14]. Our method produces more accurate images under both interpolated and extrapolated viewpoints, demonstrating improved generalization to unseen views. Best viewed by zooming in.  
TABLE II

AVERAGE COARSE-STAGE NOVEL-VIEW SYNTHESIS COMPARISON. GAUSSIAN METHODS ARE EVALUATED BEFORE AUXILIARY 2D VFM   
REFINEMENT; SCINERF IS AN EXTERNAL REFERENCE. HIGHER PSNR   
AND SSIM AND LOWER LPIPS INDICATE BETTER PERFORMANCE. BOLD INDICATES THE BEST RESULT IN EACH COLUMN.

<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>SCINeRF</td><td>26.61</td><td>0.9360</td><td>0.0835</td></tr><tr><td>SCI-3DGS</td><td>25.23</td><td>0.9238</td><td>0.0806</td></tr><tr><td>SCI-MCMC</td><td>25.20</td><td>0.9275</td><td>0.0932</td></tr><tr><td>SCI-RevADC</td><td>25.72</td><td>0.9312</td><td>0.0696</td></tr><tr><td>Ours</td><td>27.71</td><td>0.9494</td><td>0.0439</td></tr></table>

stage output; applying this refinement independently to each variant would therefore entangle the effect of its densitycontrol strategy with differences introduced by its own pseudoview targets. We exclude the second stage from these comparisons to preserve clear attribution.

We next consider more challenging SCI measurements that go beyond the standard bounded-scene setting. As shown in Table III, our method achieves the strongest overall results and ranks first or second in every entry against SCINeRF, SCI-3DGS, SCI-MCMC, and SCI-RevADC. This demonstrates robustness under more ambiguous and realistic SCI acquisition conditions and shows that the improvement is not specific to a single baseline or density-control strategy.

a) Complex Viewpoint Variations: We further evaluate challenging SCI scenarios involving irregular and large-scale camera motion from Mip-NeRF 360 [43]. As shown in Fig. 6, SCINeRF [14] struggles to preserve structural coherence under these viewpoint changes, leading to distortions and multi-view inconsistency. In contrast, our method produces more accurate and consistent reconstructions by exploiting the complementary geometric and image priors of the VFMs. Together with the quantitative results in Table III, this comparison shows that our method achieves the best overall performance against SCINeRF, SCI-3DGS, SCI-MCMC, and SCI-RevADC under complex viewpoint variations.

b) Robustness to Mask Ratio: We next vary the mask ratio while retaining the same coarse-stage evaluation protocol. Table IV compares SCINeRF, SCI-3DGS, SCI-MCMC, SCI-RevADC, and our method at every tested ratio from 0.125 to 0.75. The controlled Gaussian variants allow us to distinguish robustness arising from the proposed density-control design from that of the shared 3DGS reconstruction framework. Our method achieves the best performance at every mask ratio, outperforming SCINeRF and all three controlled variants, including under the more challenging extreme settings. These results demonstrate that the proposed reconstruction remains robust to changes in both the spatial coding pattern and the amount of information retained by the SCI measurement throughout the evaluated mask-ratio range.

c) High Compression Ratios: We then evaluate highcompression SCI measurements using scenes from DTU [37] and Mip-NeRF 360 [43], with the mask ratio fixed at 0.25. Table V includes SCINeRF together with SCI-3DGS, SCI-MCMC, and SCI-RevADC, enabling a consistent comparison of representation and density-control choices at CR = 16 and CR = 32. Our method achieves the best overall performance at both compression ratios, outperforming SCINeRF and all three controlled variants. The advantage remains pronounced at CR = 32, showing that the proposed priors and SCI-specific optimization remain effective when substantially more views are multiplexed into a single measurement.

d) Camera Trajectory Accuracy: Accurate camera poses are essential for geometrically consistent 3D reconstruction from SCI. We compare trajectories estimated by SCINeRF, SCI-3DGS, SCI-MCMC, SCI-RevADC, and ours on scenes from NeRF Synthetic [11] and Mip-NeRF 360 [43]. Table VI reports ATE, RPE<sub>t</sub>, RPE<sub>r</sub>, nATE, and nRPE<sub>t</sub> as complementary global and local trajectory errors. Ours achieves the lowest value for every metric among SCINeRF and the three controlled Gaussian variants.

QUANTITATIVE COMPARISON ON COMPLEX SCI MEASUREMENTS OVER DIFFERENT 3DGS STRATEGIES. WE EVALUATE ON MORE CHALLENGING SCI MEASUREMENTS, INCLUDING UNBOUNDED SCENES AND MEASUREMENTS GENERATED UNDER MORE COMPLEX CAMERA TRAJECTORIES FROM TANKS AND TEMPLES [42], MIP-NERF 360 [43] AND DTU [37]. TO ENSURE A FAIR COMPARISON OF DENSITY-CONTROL STRATEGIES, ALL GAUSSIAN VARIANTS ARE EVALUATED USING FIRST-STAGE RECONSTRUCTIONS ONLY. OUR METHOD ACHIEVES THE BEST OVERALL PERFORMANCE AMONG THESE VARIANTS AND THE STATE-OF-THE-ART BASELINE. BOLD AND UNDERLINED VALUES DENOTE THE BEST AND SECOND-BEST RESULTS, RESPECTIVELY.  
TABLE III
<table><tr><td>Method</td><td>Museum Truck PSNR SSIM LPIPS PSNR SSIM LPIPS PSNR</td><td>Bicycle Counter SSIM LPIPS PSNR SSIM LPIPS</td><td>Garden PSNR SSIM LPIPS</td><td>Buda PSNR SSIM LPIPS</td></tr><tr><td>SCINeRF</td><td>23.92 0.7280 0.5300 21.11 0.6800 0.4690</td><td>19.67 0.5600 0.4960 22.01</td><td>0.6620 0.5030 20.33</td><td>0.5580 0.5180 28.80 0.8900 0.2620</td></tr><tr><td>SCI-3DGS</td><td>25.14 0.7928 0.2942 21.68 0.7526 0.2322</td><td>19.40 0.5756 0.3879</td><td>25.18 0.7930 0.2428 20.33</td><td>0.6002 0.3159 28.44 0.8935 0.1926</td></tr><tr><td>SCI-MCMC</td><td>27.48 0.8809 0.1431 22.42</td><td>0.7938 0.1717 20.48 0.6237 0.3501</td><td>26.28 0.8430 0.1669 19.63</td><td>0.5567 0.3336 28.040.89140.1735</td></tr><tr><td>SCI-RevADC Ours</td><td>25.07 0.7904 0.2970 21.73</td><td>0.7513 0.2318 19.35 0.5720 0.3840</td><td>25.160.7917 0.2420 20.18</td><td>0.5949 0.3167 28.080.8791 0.2241</td></tr><tr><td></td><td>27.520.8809 0.1627 23.97</td><td>0.8460 0.1306 20.63 0.6389 0.3470</td><td>26.180.83770.1846</td><td>21.600.66220.2816 30.150.9231 0.1272</td></tr></table>

![](images/8e54fa5bb7070dded4e94f494bde610fb3408c0582833247a3fe42ece35cfb7e.jpg)  
Fig. 6. Qualitative results under complex viewpoint changes. The top and middle examples cover wide-range viewpoint changes, while the bottom example contains irregular sweeping motion and large transformations. From top to bottom: ground truth, SCINeRF [14], and our method.

## D. Ablation Study

Table VII separately evaluates opacity-guided splitting, mean-opacity regulation, and the Gaussian-count constraint.

Without the count constraint, a subset of runs exhausts GPU memory before 20,000 iterations, leaving no complete sixscene aggregate. Each remaining component improves all three reconstruction metrics, and Fig. 7 illustrates their combined qualitative effect.

a) Comparison of 3DGS Densification Strategies: Adaptive density control rules developed for dense multi-view supervision may behave differently when multiple views are entangled in one SCI measurement. We therefore report two complementary comparisons in Tables VIII and IX.

TABLE IV  
COARSE-STAGE BASE-VIEW RECONSTRUCTION UNDER DIFFERENT MASK RATIOS WITH N = 8 ENCODED VIEWS.
<table><tr><td>Mask Ratio</td><td>Method</td><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td><td></td></tr><tr><td rowspan="5">0.125</td><td>SCINeRF</td><td>32.31</td><td>0.9595</td><td>0.0675</td></tr><tr><td>SCI-3DGS</td><td>31.69</td><td>0.9541</td><td>0.0632</td></tr><tr><td>SCI-MCMC</td><td>32.92</td><td>0.9638</td><td>0.0495</td></tr><tr><td>SCI-RevADC</td><td>31.77</td><td>0.9551</td><td>0.0622</td></tr><tr><td>Ours</td><td>34.10</td><td>0.9713</td><td>0.0334</td></tr><tr><td rowspan="5">0.25</td><td>SCINeRF</td><td>33.65</td><td>0.9632</td><td>0.0393</td></tr><tr><td>SCI-3DGS</td><td>33.33</td><td>0.9580</td><td>0.0606</td></tr><tr><td>SCI-MCMC</td><td>34.43</td><td>0.9640</td><td>0.0477</td></tr><tr><td>SCI-RevADC</td><td>33.36</td><td>0.9572</td><td>0.0625</td></tr><tr><td>Ours</td><td>35.57</td><td>0.9751</td><td>0.0289</td></tr><tr><td rowspan="5">0.5</td><td>SCINeRF</td><td>34.38</td><td>0.9648</td><td>0.0496</td></tr><tr><td>SCI-3DGS</td><td>32.85</td><td>0.9500</td><td>0.0721</td></tr><tr><td>SCI-MCMC</td><td>33.44</td><td>0.9579</td><td>0.0588</td></tr><tr><td>SCI-RevADC</td><td>33.35</td><td>0.9548</td><td>0.0670</td></tr><tr><td>Ours</td><td>34.47</td><td>0.9668</td><td>0.0468</td></tr><tr><td rowspan="5">0.75</td><td>SCINeRF</td><td>32.94</td><td></td><td>0.0976</td></tr><tr><td>SCI-3DGS</td><td>32.23</td><td>0.9315 0.9388</td><td>0.0863</td></tr><tr><td>SCI-MCMC</td><td>32.81</td><td>0.9477</td><td>0.0731</td></tr><tr><td>SCI-RevADC</td><td></td><td></td><td></td></tr><tr><td></td><td>33.03</td><td>0.9492</td><td>0.0681</td></tr><tr><td></td><td>Ours</td><td>34.24</td><td>0.9618</td><td>0.0490</td></tr></table>

TABLE V

COARSE-STAGE BASE-VIEW RECONSTRUCTION UNDER HIGH COMPRESSION RATIOS ON SCENES FROM DTU [37] AND MIP-NERF 360 [43]. THE MASK RATIO IS FIXED AT 0.25.
<table><tr><td>Compression</td><td>Method</td><td>PSNR↑ SSIM↑ LPIPS↓</td></tr><tr><td>CR = 16</td><td>SCINeRF SCI-3DGS SCI-MCMC</td><td>21.60 0.5697 0.5281 22.33 0.6295 0.4292 23.15 0.6541 0.3709 0.4229</td></tr><tr><td></td><td>SCI-RevADC Ours</td><td>22.26 0.6274 23.65 0.6844</td></tr><tr><td></td><td>SCINeRF SCI-3DGS</td><td>0.3688 18.69 0.4647 0.6033 21.02 0.5599 0.5032</td></tr><tr><td>CR = 32 Ours</td><td>SCI-MCMC SCI-RevADC</td><td>21.28 0.5708 0.4821 20.94 0.5563 0.5034</td></tr></table>

Controlled full-20 comparison. Table VIII includes all 20 cases and is designed to isolate adaptive density control and method-specific optimization. SCI-3DGS, SCI-MCMC, SCI-RevADC, and ours use the same SCI data-fidelity objective, VFM initialization, camera-pose refinement, training budget, and evaluation data; only their adaptive density control and associated optimization rules differ, following vanilla 3DGS [12], MCMC [41], Revising-GS [40], and OSGR, respectively. All Gaussian variants are evaluated after the first stage. We omit the fine stage because its pseudo-view targets are generated from the corresponding first-stage reconstruction, so applying it independently could propagate coarsestage differences and obscure attribution. SCINeRF is included only as an external reference. Thus, SCI-MCMC measures the effect of the original MCMC rules inside the shared protocol. Under this controlled setting, ours achieves the best reconstruction quality, renders at 407.944 FPS, and completes training in 53.67 minutes. SCI-RevADC is marginally faster at 415.343 FPS and 50.24 minutes, but has substantially worse PSNR, SSIM, and LPIPS.

Matched 17-case comparison at the coarse stage. Because SCISplat<sup>†</sup> [15] has no public implementation, we reproduce its published pipeline under the same data and hardware protocol. We attempt the same 20 cases, but SCISplat<sup>†</sup> fails to initialize on three. Table IX therefore reports paired averages over the 17 successful cases, with ours recomputed on exactly the same subset before auxiliary 2D VFM refinement. Because each refinement target is generated from its corresponding first-stage output, applying the second stage separately can propagate and compound existing coarse-stage differences. We therefore compare first-stage results only, preserving direct attribution to the reconstruction method. Under this protocol, ours improves PSNR by 3.29 dB and SSIM by 0.0538 while reducing LPIPS by 0.0694 and training time by 32.16 minutes, and it yields lower error on every trajectory metric.

b) Auxiliary 2D VFM Refinement: We evaluate the auxiliary refinement from the same coarse reconstruction. As shown in Table X, unweighted 2D VFM supervision improves all aggregate metrics, particularly on extrapolated views. Alphabased support weighting further maintains or improves every metric and yields the highest PSNR and lowest LPIPS for both view sets. The 2D VFM thus supplies refinement targets, while the support weight modulates their spatial influence according to the accumulated alpha of the current Gaussian representation. Figure 7 illustrates the corresponding changes in local structure and appearance.

c) Qualitative Analysis of OSGR: We further provide both 2D and 3D visual ablations to analyze the effect of the proposed components. In the 2D rendering comparison shown in Fig. 7, removing OSGR or camera pose optimization leads to noticeably blurrier reconstructions and local structural distortions, whereas the full model recovers sharper details and more faithful appearance. These visualizations further support that OSGR improves both image-space fidelity and 3D structural consistency under the compressed SCI setting.

d) Effect of Opacity Regulation: We evaluate the meanopacity term with the remaining OSGR components fixed, including the Gaussian-count constraint. As summarized in Table VII and further illustrated by the population trajectories in the supplementary material, removing this term yields lower reconstruction quality and a substantially larger Gaussian population by the 7,000-iteration reference point. The meanopacity term provides soft regulation during active densification. After the reference iteration, the count constraint fixes the population reached at that point as the hard upper bound for subsequent splitting and duplication. Together, they enable the complete model to maintain a substantially more compact representation than the variant without opacity regulation while achieving the best reconstruction metrics.

TABLE VI  
AVERAGE CAMERA TRAJECTORY ACCURACY ON SCENES FROM NERF SYNTHETIC [11] AND MIP-NERF 360 [43]. nATE AND nRPE DENOTE NORMALIZED ATE AND NORMALIZED RPE , RESPECTIVELY. LOWER IS BETTER FOR ALL METRICS.
<table><tr><td>Method</td><td>ATE↓</td><td>nATE↓</td><td>Trans. mean↓</td><td>Rot. mean (°)↓</td><td>RPEt ↓</td><td>nRPEt ↓</td><td>RPEr (°)↓</td></tr><tr><td>SCINeRF</td><td>0.149424</td><td>0.050720</td><td>0.135576</td><td>45.7417</td><td>0.394496</td><td>0.092206</td><td>4.7359</td></tr><tr><td>SCI-3DGS</td><td>0.044590</td><td>0.014153</td><td>0.040852</td><td>4.1965</td><td>0.048161</td><td>0.015108</td><td>1.0580</td></tr><tr><td>SCI-MCMC</td><td>0.068249</td><td>0.021450</td><td>0.062590</td><td>6.0624</td><td>0.081741</td><td>0.025888</td><td>2.0724</td></tr><tr><td>SCI-RevADC</td><td>0.042689</td><td>0.013698</td><td>0.038905</td><td>3.3509</td><td>0.040692</td><td>0.013089</td><td>0.9033</td></tr><tr><td>Ours</td><td>0.042353</td><td>0.013490</td><td>0.038785</td><td>3.3018</td><td>0.039854</td><td>0.012780</td><td>0.8927</td></tr></table>

TABLE VII  
COMPONENT ABLATION ON SIX SCENES. A ✓ DENOTES AN ENABLED COMPONENT. OOM DENOTES INCOMPLETE UNCAPPED RUNS.
<table><tr><td>VFM Points</td><td>VFM Poses</td><td>Opacity Split</td><td>Opacity Reg.</td><td>Count Cap</td><td>Pose Refine</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>21.20</td><td>0.6980</td><td>0.4490</td></tr><tr><td>X</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>33.51</td><td>0.9652</td><td>0.0436</td></tr><tr><td>√</td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>30.30</td><td>0.9404</td><td>0.0777</td></tr><tr><td>√</td><td>√</td><td>x</td><td>√</td><td>√</td><td>√</td><td>33.26</td><td>0.9570</td><td>0.0629</td></tr><tr><td>√</td><td>√</td><td>√</td><td>x</td><td>√</td><td>√</td><td>33.90</td><td>0.9619</td><td>0.0604</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>√</td><td></td><td>OOM</td><td></td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>29.70</td><td>0.9090</td><td>0.1476</td></tr><tr><td>√</td><td>√</td><td>L</td><td>√</td><td>√</td><td>√</td><td>35.57</td><td>0.9751</td><td>0.0289</td></tr></table>

TABLE VIII

CONTROLLED FIRST-STAGE COMPARISON OVER ALL 20 CASES. THE FOUR GAUSSIAN VARIANTS ARE RECOMPUTED WITH THE SAME SCI OBJECTIVE, VFM INITIALIZATION, POSE-REFINEMENT PROTOCOL, AND TRAINING BUDGET; SCINERF IS AN EXTERNAL REFERENCE. FPS AND TRAINING TIME USE THE FIXED-HARDWARE BENCHMARK. BOLD VALUES INDICATE THE BEST RESULT IN EACH COLUMN.
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FPS↑</td><td>Training Time (min)↓</td></tr><tr><td>SCINeRF</td><td>27.84</td><td>0.8483</td><td>0.2215</td><td>0.372</td><td>746.84</td></tr><tr><td>SCI-3DGS</td><td>27.80</td><td>0.8619</td><td>0.1594</td><td>382.263</td><td>54.33</td></tr><tr><td>SCI-MCMC</td><td>29.05</td><td>0.8821</td><td>0.1276</td><td>284.647</td><td>59.20</td></tr><tr><td>SCI-RevADC</td><td>27.92</td><td>0.8631</td><td>0.1566</td><td>415.343</td><td>50.24</td></tr><tr><td>Ours</td><td>29.80</td><td>0.9005</td><td>0.1090</td><td>407.944</td><td>53.67</td></tr></table>

TABLE IX

COARSE-STAGE COMPARISON WITH SCISPLAT<sup>†</sup> [15], WHERE <sup>†</sup> DENOTES OUR REPRODUCTION. RECONSTRUCTION AND TRAINING RESULTS AVERAGE 17 SUCCESSFUL CASES (THREE OF 20 FAIL). BOLD INDICATES THE BETTER RESULT.
<table><tr><td></td><td colspan="4">Reconstruction and Training</td><td colspan="5">Trajectory Accuracy</td></tr><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Train (min)↓</td><td>ATE RMSE↓</td><td></td><td>nATE↓ Rotation (°)↓</td><td>nRPEt↓</td><td>RPEr (°)↓</td></tr><tr><td>SCISplat†</td><td>26.79</td><td>0.8461</td><td>0.1761</td><td>86.89</td><td>0.08023</td><td>0.04062</td><td>16.767</td><td>0.04446</td><td>1.534</td></tr><tr><td>Ours</td><td>30.08</td><td>0.8999</td><td>0.1067</td><td>54.73</td><td>0.04235</td><td>0.01349</td><td>3.302</td><td>0.01278</td><td>0.893</td></tr></table>

TABLE X

ABLATION OF AUXILIARY 2D VFM REFINEMENT AND ALPHA-BASED SUPPORT WEIGHTING OVER EIGHT SCENES. ALL VARIANTS SHARE THE SAME COARSE RECONSTRUCTION AND ARE EVALUATED AT HELD-OUT POSES DISJOINT FROM THOSE USED FOR PSEUDO-VIEW SUPERVISION.
<table><tr><td>View Set</td><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td rowspan="3">All Novel</td><td>Coarse Output</td><td>27.71</td><td>0.9494</td><td>0.0439</td></tr><tr><td>2D VFM</td><td>27.75</td><td>0.9498</td><td>0.0426</td></tr><tr><td>+ Support Weight</td><td>27.76</td><td>0.9498</td><td>0.0425</td></tr><tr><td rowspan="3">Extrapolated</td><td>Coarse Output</td><td>25.19</td><td>0.9273</td><td>0.0779</td></tr><tr><td>2D VFM</td><td>25.38</td><td>0.9295</td><td>0.0720</td></tr><tr><td>+ Support Weight</td><td>25.39</td><td>0.9297</td><td>0.0715</td></tr></table>

## E. Limitations and Future Work

Table XI evaluates two DAVIS sequences outside our staticscene formulation [10]. For SCI-3DGS, SCI-MCMC, SCI-RevADC, and ours, we report the corresponding complete finestage outputs. Ours obtains better values than SCINeRF [14], SCI-3DGS, and SCI-RevADC in all six scene–metric entries, but SCI-MCMC achieves the best value for every reported metric. Ours ranks second throughout, indicating that dynamic scenes remain a limitation of the current formulation.

One plausible OSGR-specific hypothesis concerns the use of local opacity contrast as an additional splitting cue. Independent object motion makes the static forward model misspecified: one Gaussian set cannot consistently explain all masked views multiplexed into the measurement. Under this mismatch, optimized opacity may reflect not only persistent surface support but also compensation for motion-induced residuals. A compensatory primitive supported by only a subset of views may become a local opacity outlier and satisfy OSGR’s splitting criterion despite lacking temporal support. The transmittance-preserving child update means that splitting does not by itself imply an immediate increase in accumulated opacity, while the global opacity penalty and Gaussiancount constraint limit proliferation. However, none of these mechanisms identifies whether a local opacity peak arises from static geometry or motion compensation. Repeated selection could therefore divert bounded representation capacity toward ambiguous dynamic regions and potentially preserve ghost contours or fragmented geometry after subsequent optimization. Future work will incorporate motion-aware crossview support into splitting and extend the representation to 4D Gaussians [44], thereby improving robustness in general dynamic SCI settings under complex object motion.

![](images/6a16d06365273fded6c1489071913dc0c068ea0c325ce4b377946a9632855dfa.jpg)  
Fig. 7. Qualitative ablations of camera-pose optimization, OSGR, and auxiliary 2D VFM refinement. Removing camera-pose optimization or OSGR produce broader changes in geometry and sharpness, whereas the differences associated with auxiliary refinement are concentrated in local structure and appearance. The complete configuration combines these complementary effects.

TABLE XI  
QUANTITATIVE COMPARISON ON THE BEAR AND FLAMINGO DYNAMIC SCENES FROM DAVIS [10]. ALL GAUSSIAN VARIANTS ARE EVALUATED USING THEIR COMPLETE FINE-STAGE OUTPUTS.
<table><tr><td>Scene</td><td>Method</td><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td></tr><tr><td rowspan="5">Bear</td><td>SCINeRF SCI-3DGS</td><td>23.96</td><td>0.8560 0.1940</td></tr><tr><td></td><td>24.10</td><td>0.8119 0.1671</td></tr><tr><td>SCI-MCMC</td><td>27.46</td><td>0.9072 0.0843</td></tr><tr><td>SCI-RevADC</td><td>25.16</td><td>0.8499 0.1345</td></tr><tr><td>Ours</td><td>26.90</td><td>0.8923 0.0997</td></tr><tr><td></td><td>SCINeRF SCI-3DGS</td><td>25.11 25.13</td><td>0.8240 0.2700 0.8313 0.2237</td></tr><tr><td>Flamingo</td><td>SCI-MCMC SCI-RevADC</td><td>28.90 25.20</td><td>0.9115 0.1112 0.8306</td></tr><tr><td></td><td>Ours</td><td>27.59</td><td>0.2150 0.8949 0.1396</td></tr></table>

## V. CONCLUSION

We present a framework for 3D scene reconstruction from a single SCI measurement by combining vision foundation models with a tailored 3DGS-based pipeline. Our primary reconstruction combines measurement-derived 3D VFM initialization with SCI-aware Gaussian optimization, where OSGR combines opacity-guided splitting with global opacity regulation to stabilize Gaussian growth under sparse SCI supervision. An auxiliary 2D VFM stage provides complementary pseudo-view supervision at synthesized viewpoints for local appearance refinement. Extensive experiments across diverse SCI settings demonstrate leading performance in both reconstruction and novel-view synthesis. More broadly, the modular design permits increasingly capable geometry and image foundation models to be incorporated into the corresponding initialization and refinement stages, providing a practical path toward higher-quality 3D reconstruction from compressed visual observations and more challenging SCI acquisition settings.

## REFERENCES

[1] Y. Liu, X. Yuan, J. Suo, D. J. Brady, and Q. Dai, “Rank minimization for snapshot compressive imaging,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 12, pp. 2990–3006, Dec. 2019. [Online]. Available: http://dx.doi.org/10.1109/TPAMI.2018.2873587

[2] J. Ma, X.-Y. Liu, Z. Shou, and X. Yuan, “Deep tensor ADMM-Net for snapshot compressive imaging,” in 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 10 222–10 231.

[3] Z. Cheng, R. Lu, Z. Wang, H. Zhang, B. Chen, Z. Meng, and X. Yuan, “BIRNAT: Bidirectional recurrent neural networks with adversarial training for video snapshot compressive imaging,” in Computer Vision – ECCV 2020, ser. Lecture Notes in Computer Science, vol. 12369. Springer, 2020, pp. 258–275.

[4] Z. Cheng, B. Chen, G. Liu, H. Zhang, R. Lu, Z. Wang, and X. Yuan, “Memory-efficient network for large-scale video compressive sensing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2021, pp. 16 246–16 255.

[5] L. Wang, M. Cao, Y. Zhong, and X. Yuan, “Spatial-temporal transformer for video snapshot compressive imaging,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 7, pp. 9072–9089, July 2023.

[6] P. Wang, Y. Zhang, L. Wang, and X. Yuan, “Hierarchical separable video transformer for snapshot compressive imaging,” in Computer Vision – ECCV 2024, ser. Lecture Notes in Computer Science, vol. 15139. Springer, 2025, pp. 104–122.

[7] L. Wang, M. Cao, and X. Yuan, “EfficientSCI: Densely connected network with space-time factorization for large-scale video snapshot compressive imaging,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2023, pp. 18 477–18 486.

[8] X. Yuan, Y. Liu, J. Suo, and Q. Dai, “Plug-and-play algorithms for largescale snapshot compressive imaging,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020, pp. 1444–1454.

[9] X. Yuan, Y. Liu, J. Suo, F. Durand, and Q. Dai, “Plug-and-play algorithms for video snapshot compressive imaging,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 7093– 7111, October 2022.

[10] J. Pont-Tuset, F. Perazzi, S. Caelles, P. Arbelaez, A. Sorkine-Hornung,´ and L. V. Gool, “The 2017 DAVIS challenge on video object segmentation,” 2018. [Online]. Available: https://arxiv.org/abs/1704. 00675

[11] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “NeRF: Representing scenes as neural radiance fields for view synthesis,” in Computer Vision – ECCV 2020, ser. Lecture Notes in Computer Science, vol. 12346. Springer, 2020, pp. 405–421.

[12] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3D gaussian¨ splatting for real-time radiance field rendering,” ACM Transactions on Graphics, vol. 42, no. 4, July 2023.

[13] Z. Wang, H. Yang, Y. Guo, and F. Wang, “SCIGS: 3D gaussians splatting from a snapshot compressive image,” in 2025 IEEE International Conference on Image Processing (ICIP), 2025, pp. 1013–1018.

[14] Y. Li, X. Wang, P. Wang, X. Yuan, and P. Liu, “SCINeRF: Neural radiance fields from a snapshot compressive image,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 10 542–10 552.

[15] Y. Li, X. Liu, X. Wang, X. Yuan, and P. Liu, “Learning radiance fields from a single snapshot compressive image,” 2024. [Online]. Available: https://arxiv.org/abs/2412.19483

[16] X. Li, Y. Li, X. Wang, X. Yuan, M. D. Butala, and G. Wang, “SCI-Gaussian: Optimizing 3D gaussian radiance fields from a snapshot compressive image,” in ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025, pp. 1–5.

[17] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “VGGT: Visual geometry grounded transformer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 5294–5306.

[18] X. Yuan, D. J. Brady, and A. K. Katsaggelos, “Snapshot compressive imaging: Theory, algorithms, and applications,” IEEE Signal Processing Magazine, vol. 38, no. 2, pp. 65–88, 2021.

[19] P. Yang, L. Kong, X.-Y. Liu, X. Yuan, and G. Chen, “Shearlet enhanced snapshot compressive imaging,” IEEE Transactions on Image Processing, vol. 29, pp. 6466–6481, 2020.

[20] X. Yuan, “Generalized alternating projection based total variation minimization for compressive sensing,” in 2016 IEEE International Conference on Image Processing (ICIP), 2016, pp. 2539–2543.

[21] M. Cao, L. Wang, M. Zhu, and X. Yuan, “Hybrid CNN-Transformer architecture for efficient large-scale video snapshot compressive imaging,” International Journal of Computer Vision, vol. 132, no. 10, pp. 4521–4540, 2024.

[22] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems, vol. 30, 2017, pp. 5998–6008.

[23] J. Wang, N. Karaev, C. Rupprecht, and D. Novotny, “VGGSfM: Visual geometry grounded deep structure from motion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 21 686–21 697.

[24] J. Zhang, C. Herrmann, J. Hur, V. Jampani, T. Darrell, F. Cole, D. Sun, and M.-H. Yang, “MonST3R: A simple approach for estimating geometry in the presence of motion,” 2024. [Online]. Available: https://arxiv.org/abs/2410.03825

[25] W. Jang, P. Weinzaepfel, V. Leroy, L. Agapito, and J. Revaud, “Pow3R: Empowering unconstrained 3D reconstruction with camera and scene priors,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 1071–1081.

[26] J. Yang, A. Sax, K. J. Liang, M. Henaff, H. Tang, A. Cao, J. Chai, F. Meier, and M. Feiszli, “Fast3R: Towards 3D reconstruction of 1000+ images in one forward pass,” in Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 21 924–21 935.

[27] Z. Li, R. Tucker, F. Cole, Q. Wang, L. Jin, V. Ye, A. Kanazawa, A. Holynski, and N. Snavely, “MegaSaM: Accurate, fast and robust structure and motion from casual dynamic videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 10 486–10 496.

[28] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “DUSt3R: Geometric 3D vision made easy,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 20 697–20 709.

[29] Y. Wang, J. Zhou, H. Zhu, W. Chang, Y. Zhou, Z. Li, J. Chen, J. Pang, C. Shen, and T. He, “π<sup>3</sup>: Scalable permutation-equivariant visual geometry learning,” 2025. [Online]. Available: https://arxiv.org/ abs/2507.13347

[30] J. Z. Wu, Y. Zhang, H. Turki, X. Ren, J. Gao, M. Z. Shou, S. Fidler, Z. Gojcic, and H. Ling, “DIFIX3D+: Improving 3D reconstructions with single-step diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 26 024–26 035.

[31] Z. Xu, C. Feng, Y. Li, J. Zhao, J. Yang, W. Yu, L. Yuan, and Y. Tian, “Breaking the vicious cycle: Coherent 3D gaussian splatting from sparse and motion-blurred views,” arXiv preprint arXiv:2512.10369, 2025.

[32] X. Yin, Q. Zhang, J. Chang, Y. Feng, Q. Fan, X. Yang, C.-M. Pun, H. Zhang, and X. Cun, “GSFixer: Improving 3D gaussian splatting with reference-guided video diffusion priors,” arXiv preprint arXiv:2508.09667, 2025.

[33] Z. Wang, Y. Gu, D. Zhou, and R. Xu, “FixingGS: Enhancing 3D gaussian splatting via training-free score distillation,” arXiv preprint arXiv:2509.18759, 2025.

[34] H. Wang, F. Liu, J. Chi, and Y. Duan, “VideoScene: Distilling video diffusion model to generate 3D scenes in one step,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 16 475–16 485.

[35] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,”¨ in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 4104–4113.

[36] L. Ma, X. Li, J. Liao, Q. Zhang, X. Wang, J. Wang, and P. V. Sander, “Deblur-NeRF: Neural radiance fields from blurry images,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 12 861–12 870.

[37] R. Jensen, A. Dahl, G. Vogiatzis, E. Tola, and H. Aanæs, “Large scale multi-view stereopsis evaluation,” in 2014 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2014, pp. 406–413.

[38] B. Mildenhall, P. P. Srinivasan, R. Ortiz-Cayon, N. K. Kalantari, R. Ramamoorthi, R. Ng, and A. Kar, “Local light field fusion: Practical view synthesis with prescriptive sampling guidelines,” ACM Transactions on Graphics (TOG), 2019.

[39] M. Tancik, E. Weber, E. Ng, R. Li, B. Yi, T. Wang, A. Kristoffersen, J. Austin, K. Salahi, A. Ahuja, D. Mcallister, J. Kerr, and A. Kanazawa, “Nerfstudio: A modular framework for neural radiance field development,” in Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Proceedings, ser. SIGGRAPH ’23. ACM, Jul. 2023, pp. 1–12. [Online]. Available: http://dx.doi.org/10.1145/3588432.3591516

[40] S. Rota Bulo, L. Porzi, and P. Kontschieder, “Revising densification\` in gaussian splatting,” in Computer Vision – ECCV 2024, ser. Lecture Notes in Computer Science, vol. 15121. Springer, 2025, pp. 347–362.

[41] S. Kheradmand, D. Rebain, G. Sharma, W. Sun, Y.-C. Tseng, H. Isack, A. Kar, A. Tagliasacchi, and K. M. Yi, “3D gaussian splatting as markov chain monte carlo,” in Advances in Neural Information Processing Systems, vol. 37, 2024.

[42] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, “Tanks and temples: Benchmarking large-scale scene reconstruction,” ACM Transactions on Graphics, vol. 36, no. 4, July 2017.

[43] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, “Mip-NeRF 360: Unbounded anti-aliased neural radiance fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 5470–5479.

[44] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, “4d gaussian splatting for real-time dynamic scene rendering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 310–20 320.

# $\mathrm { G S ^ { 2 } C I } \mathrm { : }$ : Robust Gaussian Splatting For Snapshot Compressive Imaging via Large Vision Model Priors Supplementary Material

Yanming Yang, Chenxi Song, Ping Wang, Xin Yuan, and Chi Zhang

## APPENDIX A

## MULTIPLEXING-AWARE GEOMETRY INITIALIZATION

VGGT [17] assumes a sequence of view-resolved RGB images, whereas SCI collapses N coded views into the single multiplexed observation Y defined in Eq. (1) in the main paper. Consequently, the raw SCI measurement does not satisfy VGGT’s input model: structures from different views are superimposed rather than individually resolved. We address this mismatch by constructing geometry-oriented proxy views from Y and the known binary coding masks $\mathcal { M } = \{ \mathbf { M } _ { i } \} _ { i = 1 } ^ { N } .$ The construction adapts to the realized mask overlap and completes each proxy by interpolating the locations omitted by its mask before the proxy is supplied to VGGT.

Specifically, let $\mathbf { X } _ { i } \ \in \ \bar { \mathbb { R } ^ { H \times W \times \bar { 3 } } }$ denote the latent RGB image of view i $, \mathbf { M } _ { i } \in \{ 0 , 1 \} ^ { H \times W }$ its corresponding coding mask shared across RGB channels, and p a pixel location. Our construction produces the sequence $\bar { \mathcal { X } } \bar { \mathbf { \Xi } } = \{ \widetilde { \mathbf { X } } _ { i } \} _ { i = 1 } ^ { N }$ of RGB proxy views on the same $H \times W$ sampling grid. These proxies are supplied exclusively to the frozen VGGT model for geometry initialization; they are not interpreted as SCI reconstructions and never serve as photometric supervision.

The mean measurement multiplicity determines whether the proxy generator uses Energy-Normalized Initialization (ENI) or Adaptive Mask-Decoding Initialization (AMDI):

$$
\mu _ { \mathcal { M } } = \frac { 1 } { H W } \sum _ { \mathbf { p } } \sum _ { i = 1 } ^ { N } \mathbf { M } _ { i } ( \mathbf { p } ) ,\tag{11}
$$

which measures the average number of latent views contributing to one measurement pixel. With $\tau _ { \mu } = 4$ fixed for all scenes, the proxy construction is selected as follows:

$$
\widetilde { \mathcal { X } } = \mathcal { P } ( \mathbf { Y } , \mathcal { M } ) = \left\{ \begin{array} { l l } { \mathcal { P } _ { \mathrm { E N I } } ( \mathbf { Y } , \mathcal { M } ) , } & { \mu _ { \mathcal { M } } < \tau _ { \mu } , } \\ { \mathcal { P } _ { \mathrm { A M D I } } ( \mathbf { Y } , \mathcal { M } ) , } & { \mu _ { \mathcal { M } } \geq \tau _ { \mu } . } \end{array} \right.\tag{12}
$$

The two constructions address complementary mixing regimes. At low multiplicity, ENI exposes mask-supported samples without fitting a local-constant multi-view inverse model. As multiplicity increases, however, exposure normalization retains only aggregate intensity and cannot separate the superposed latent-view contributions. Conversely, AMDI’s full-rank local system may require enlarged spatial support when mixing is weak. Equation (12) therefore balances crossview ambiguity against local inverse-model bias rather than identifying a scene category.

For low-multiplicity acquisitions, ENI compensates for firstorder exposure variation induced by mask overlap as follows:

$$
\begin{array} { c c c } { \displaystyle { a ( { \bf p } ) = \sum _ { i = 1 } ^ { N } { \bf M } _ { i } ( { \bf p } ) , } } \\ { \displaystyle { \overline { { \bf Y } } ( { \bf p } ) = { \bf Y } ( { \bf p } ) / a ( { \bf p } ) . } } \end{array}\tag{13}
$$

The division is applied wherever $a ( \mathbf { p } ) \mathbf { \psi } > 0$ , and we set $\overline { { \mathbf { Y } } } ( \mathbf { p } ) = 0$ when $a ( \mathbf { p } ) = 0$ . Each view-specific proxy is then obtained by remodulating the normalized measurement with M and completing the mask-omitted locations:

$$
\widetilde { \mathbf { X } } _ { i } ^ { \mathrm { E N I } } = \mathcal { I } \big ( \overline { { \mathbf { Y } } } \odot \mathbf { M } _ { i } ; \mathbf { M } _ { i } \big ) .\tag{14}
$$

For a binary mask, I treats the values at $S _ { i } = \{ \mathbf { p } : \mathbf { M } _ { i } ( \mathbf { p } ) =$ $1 \}$ as known samples, linearly interpolates missing pixels inside conv $( S _ { i } )$ , and uses nearest-neighbor extrapolation outside conv(S<sub>i</sub>). Thus, each proxy retains its mask-supported samples and is completed on the original grid.

When $\mu _ { \mathcal { M } }$ is high, more latent views overlap at each measurement pixel. Energy normalization corrects the aggregate exposure but does not separate these view contributions, so direct mask remodulation remains ambiguous. AMDI therefore exploits the spatial variation of the actual coding patterns to estimate local per-view contributions. Define m $\dot { \bf ( q ) } = [ { \bf M } _ { 1 } ( { \bf q } ) , \ldots , { \bf M } _ { N } ( \dot { \bf q } ) ] ^ { \top } \in \mathbb { R } ^ { N }$ . Within a clipped square neighborhood $\Omega _ { s } ( \mathbf { p } )$ , AMDI adopts the local-constant model $\mathbf { Y } ( \mathbf { q } ) \approx \mathbf { U } ( \mathbf { p } ) ^ { \mathsf { T } } \mathbf { m } ( \mathbf { q } ) + \mathbf { Z } ( \mathbf { q } )$ , where ${ \bf U } ( { \bf p } ) \in \mathbb { R } ^ { N \times 3 }$ stacks the unknown RGB values of the N proxies. Its normalequation terms are:

$$
\begin{array} { l } { { \displaystyle { \bf G } _ { s } ( { \bf p } ) = \sum _ { { \bf q } \in \Omega _ { s } ( { \bf p } ) } { \bf m } ( { \bf q } ) { \bf m } ( { \bf q } ) ^ { \top } } , } \\ { { \displaystyle { \bf B } _ { s } ( { \bf p } ) = \sum _ { { \bf q } \in \Omega _ { s } ( { \bf p } ) } { \bf m } ( { \bf q } ) { \bf Y } ( { \bf q } ) ^ { \top } } . } \end{array}\tag{15}
$$

Let $\begin{array} { r } { \mathbf { G } _ { \mathrm { a l l } } = \sum _ { \mathbf { a } } \mathbf { m } ( \mathbf { q } ) \mathbf { m } ( \mathbf { q } ) ^ { \mathsf { T } } } \end{array}$ be the full-image mask Gram matrix. We select the smallest numerically full-rank and wellconditioned neighborhood:

$$
\begin{array} { r l r } { s ^ { \star } ( \mathbf { p } ) = \displaystyle \operatorname* { m i n } _ { s \in \mathcal { S } } \big \{ s : \mathrm { r a n k } _ { \mathrm { n u m } } ( \mathbf { G } _ { s } ( \mathbf { p } ) ) = N , \hfill } & \\ { \displaystyle \lambda _ { \mathrm { m i n } } ( \mathbf { G } _ { s } ( \mathbf { p } ) ) \geq \tau _ { \lambda } , } & { } & \\ { \displaystyle \kappa _ { 2 } ( \mathbf { G } _ { s } ( \mathbf { p } ) ) \leq \gamma \kappa _ { 2 } ( \mathbf { G } _ { \mathrm { a l l } } ) \big \} , } & { } & \end{array}\tag{16}
$$

where $\kappa _ { 2 }$ is the spectral condition number, $\tau _ { \lambda } = 1$ , and $\gamma = 2$ Numerical rank is evaluated at the solver tolerance. Given the selected unregularized Gram matrix, a trace-normalized ridge produces the proxy stack:

$$
\begin{array} { c } { { \displaystyle \lambda _ { \mathbf { p } } ^ { \mathrm { r i d g e } } = \eta \frac { \mathrm { t r } \left( { \bf G } _ { s ^ { \star } } ( \mathbf { p } ) \right) } { N } , } } \\ { { \displaystyle \widetilde { \mathbf { X } } ^ { \mathrm { A M D I } } ( \mathbf { p } ) = \left( { \bf G } _ { s ^ { \star } } ( \mathbf { p } ) + \lambda _ { \mathbf { p } } ^ { \mathrm { r i d g e } } { \bf I } _ { N } \right) ^ { - 1 } { \mathbf { B } } _ { s ^ { \star } } ( \mathbf { p } ) . } } \end{array}\tag{17}
$$

We set $\eta = 1 0 ^ { - 3 }$ . At each pixel, the N rows of $\widetilde { \mathbf { X } } ^ { \mathrm { A M D I } } ( \mathbf { p } )$ provide the RGB values of the N proxy views. The evaluated neighborhood sizes are:

$$
\begin{array} { r } { { \cal { S } } = \{ 3 , 5 , 7 , 9 , 1 1 , 1 3 , 1 5 , 1 7 , } \\ { 2 1 , 2 5 , 3 1 , 4 1 , 5 7 , 8 1 \} . } \end{array}\tag{18}
$$

Boundary windows contain only unique in-image samples, without padding or repetition. The listed sizes cover all experiments; for larger inputs, odd window sizes grow geometrically until full-image coverage. AMDI requires an admissible fullimage system and a neighborhood satisfying Eq. (16).

For the acquisition settings evaluated in Table IV in the main paper and Table V in the main paper, Eq. (12) assigns the 8-view mask-ratio settings of 0.125 and 0.25 to ENI, those of 0.5 and 0.75 to AMDI, and the CR = 16 and $\mathrm { C R } = 3 2 $ settings at mask ratio 0.25 to AMDI. Both branches produce N complete RGB proxy views before VGGT preprocessing. The local neighborhoods introduced by AMDI are used only to construct these views; VGGT is always applied to the complete proxy sequence. The choice between ENI and AMDI is determined solely by the acquisition masks through Eq. (12), without reference imagery, scene-dependent information, or external camera calibration.

The proxy sequence is processed by VGGT’s track-based reconstruction pipeline, which establishes explicit cross-view correspondences and yields sparse multi-view estimates before and after bundle adjustment (BA):

$$
\begin{array} { r l } & { \mathcal { H } _ { \mathrm { P r e B A } } = \Phi _ { \mathrm { t r a c k } } \mathopen { } \mathclose \bgroup \left( \widetilde { \mathcal { X } } \aftergroup \egroup \right) , } \\ & { \qquad \mathcal { H } _ { \mathrm { B A } } = \mathrm { B A } \mathopen { } \mathclose \bgroup \left( \mathcal { H } _ { \mathrm { P r e B A } } \aftergroup \egroup \right) . } \end{array}\tag{19}
$$

The two estimates share the same proxy observations and correspondence initialization, while BA additionally refines the scene structure and camera parameters by minimizing multiview reprojection error. When this optimization is sufficiently constrained, the post-BA estimate is preferred because it provides the more globally consistent joint estimate of structure and cameras. The proxy views, however, are derived from a multiplexed SCI observation rather than directly observed images, and their correspondence support can be spatially nonuniform. If these correspondences do not sufficiently constrain the joint optimization, the refined solution may retain inadequate track support or image-plane coverage, or yield an unstable focal estimate. It can then be less reliable than the PreBA estimate, which preserves the geometry supported by the original tracks.

Accordingly, we first examine the post-BA reconstruction and retain it when refinement preserves complete view registration, a connected covisibility graph with sufficient multiview track support, adequate image-plane point coverage, and a stable shared-focal estimate. If these conditions are not met, we revert to the corresponding PreBA reconstruction, provided that it retains adequate view registration, multi-view track support, and image-plane point coverage. Thus, PreBA is not preferred over a valid refined solution; it prevents an underconstrained joint update from replacing geometry that remains supported by the original correspondences.

When neither candidate from the initial run satisfies these criteria, we rerun the same track-based reconstruction pipeline with a larger set of 2D tracking queries under a small, fixed set of input orderings of the same N proxy views. The larger query set covers more locations in the existing proxy images for cross-view tracking; it neither changes their spatial resolution nor introduces additional views. All retries use the same expanded query configuration and differ only in input ordering. Each retry again produces PreBA and post-BA candidates, to which we apply the same validity checks and preference for post-BA over PreBA. If no retry yields valid track-supported geometry, VGGT’s direct camera and depth predictions provide a dense initialization. This final fallback does not require persistent cross-view tracks or the trackconstrained, shared-camera BA used by the preceding pipeline. The resulting hierarchy prioritizes geometry constrained by explicit cross-view tracks and invokes the dense fallback only when such geometry cannot be recovered. All retry configurations and validity criteria are fixed across scenes and are applied without access to ground-truth imagery, groundtruth camera parameters, or final reconstruction metrics.

The retained initialization defines Q<sup>⋆</sup>, $\mathbf { K } _ { i } ^ { \star }$ , and $\mathbf { T } _ { i } ^ { ( 0 ) }$ in Eq. (4) in the main paper. Its point cloud, camera intrinsics, and poses initialize the Gaussian representation and camera parameters. The proxy views are used only for geometric initialization and never as photometric targets during SCI optimization. Measured on the single NVIDIA RTX 4090 setup used in our experiments, the complete initialization from proxy construction to the retained VGGT geometry estimate takes 4 min 45 s per scene on average.

## APPENDIX B

## ADDITIONAL IMPLEMENTATION AND ABLATION DETAILS

Beyond the Opacity-Guided Splitting and Growth Regulation (OSGR) formulation in Sec. III-E in the main paper, we compute the local opacity reference of each Gaussian using the Gaussian itself and its three nearest neighbors. Only Gaussians with a positive deviation from this local reference enter the additional opacity-driven split set. If this set exceeds 5% of the current Gaussian population, we retain the 5% with the largest positive deviations. This budget applies only to the additional opacity-driven candidates: they are merged with the conventional gradient-, scale-, and screen-radius-based split candidates, and the original high-gradient duplication rule remains unchanged. The resulting rule is relative to the local opacity distribution and therefore does not require a fixed absolute opacity threshold. We also retain the transmittancepreserving opacity correction of Revising-GS [40] for both duplication and splitting. In a controlled ablation on the six standard scenes in Table I in the main paper, applying this correction improves the overall reconstruction result in four cases; we therefore retain it for every scene to keep the OSGR update rule consistent across reconstructions.

TABLE XII  
COARSE-STAGE RUN-TO-RUN STABILITY OVER THREE INDEPENDENT RUNS. WE REPORT THE THREE-SEED MEAN AND VARIANCE FOR EACH RECONSTRUCTION METRIC.
<table><tr><td>Metric</td><td>Three-seed Mean</td><td>Variance</td></tr><tr><td>PSNR</td><td>35.5689</td><td> $0 . 0 0 1 8 9 2 5 9$ </td></tr><tr><td>SSIM</td><td>0.975100</td><td> $5 . 0 8 6 1 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>LPIPS</td><td>0.028900</td><td> $1 . 0 7 0 1 \times 1 0 ^ { - 6 }$ </td></tr></table>

The Gaussian-count constraint uses the population reached at a reference training iteration as the upper bound for subsequent splitting and duplication. This shared temporal rule adapts the numerical cap to the population accumulated by each reconstruction. Under an otherwise identical optimization protocol, we compare reference iterations of 5,000, 6,000, 7,000, 8,000, and 9,000. The 7,000-iteration setting gives the best overall result among these capped configurations and is fixed for all reported experiments. We additionally evaluate an uncapped configuration, which fails to complete the full six-scene evaluation because a subset of runs exhausts the available GPU memory before reaching the 20,000-iteration budget, as reported in Table VII in the main paper.

To quantify stochastic variation, we repeat the full coarsestage configuration in Table VII in the main paper three times with different random seeds while keeping the acquisition, initialization branch, optimization schedule, and training budget unchanged. Table XII reports the resulting mean and variance. The corresponding standard deviations are approximately 0.0435 dB for PSNR, 0.0007 for SSIM, and 0.0010 for LPIPS, indicating limited variability over the three independent runs performed with the same acquisition and optimization settings used throughout this evaluation. Table VII in the main paper, Table X in the main paper, and Table XII provide complementary controlled evidence through coarse-stage component ablations, fine-stage component ablations initialized from a shared coarse reconstruction, and runto-run variability of the complete coarse-stage configuration, respectively.

Figures 8 and 9 provide two complementary representationlevel diagnostics. Figure 8 is the 3D counterpart to the imagespace ablation in Fig. 7 in the main paper: removing OSGR produces a less coherent Gaussian structure, whereas the full model maintains a more compact representation. Figure 9 reports the corresponding population trajectories during coarse optimization. Both OSGR configurations use the same countconstraint protocol, yet the variant without opacity regulation reaches a substantially larger population by iteration 7,000. The mean-opacity term provides soft regulation during active densification, and the population reached at the reference iteration supplies the hard upper bound thereafter. The complete configuration remains close to standard adaptive density control and substantially below the variant without opacity regulation throughout the later optimization stage.

![](images/cc9864b681c6a2537a422e35c65a5a465b7d89706c52aefcb5b6f6f9f56fff41.jpg)  
(a) w/o OSGR

![](images/6e60574eb4c0e75fbdfa3c88c3f68c8d43a5266ba3f785a1efd93355aadf02b3.jpg)

![](images/7131864aaa551da30f77feb64c3ca479f673bcdad383e378690dfcdd0a345dde.jpg)

![](images/54e2211295b962c912e5e823b515b82cd1f9490debb77d625d384d9a22ffc66b.jpg)

![](images/5b11b91bdc3213c295a0f457093d787fe3c758fd0b4adc03199cdfa573cac75f.jpg)  
(b) w/ OSGR

Fig. 8. 3D qualitative ablation of OSGR, complementing the image-space comparison in Fig. 7 of the main paper. Removing OSGR yields less coherent geometry and stronger structural distortions, whereas the full model produces a more compact and geometrically consistent representation.  
![](images/ed80e13dc97eb12db6316e06c90d9accf424308a5692227612ad8c196827b30c.jpg)  
Fig. 9. Gaussian-population trajectories during coarse optimization. Ours and the variant without opacity regulation use the same count-constraint protocol, which fixes the population reached at iteration 7,000 as the upper bound thereafter. Removing the soft mean-opacity term yields a substantially larger population by the reference iteration, whereas the complete model remains close to standard 3DGS adaptive density control [12].

## APPENDIX C ADDITIONAL RESULTS

TABLE XIII  
COMPARISON OF PROXY-VIEW INITIALIZATION STRATEGIES OVER SIX MIXED CASES. BOLD VALUES INDICATE THE BEST RESULT IN EACH COLUMN.
<table><tr><td>Method</td><td>nATE (%)↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Naive  $\mathbf { Y } \odot \mathbf { M } _ { i }$ </td><td>17.1603</td><td>27.62</td><td>0.8240</td><td>0.2362</td></tr><tr><td>ENI only</td><td>18.3801</td><td>28.61</td><td>0.8471</td><td>0.2158</td></tr><tr><td>AMDI only</td><td>2.3468</td><td>31.57</td><td>0.8776</td><td>0.1629</td></tr><tr><td>Adaptive (Ours)</td><td>2.3277</td><td>31.72</td><td>0.8810</td><td>0.1611</td></tr></table>

![](images/4ae3fd0bc8edaf86793124cc549e2c7225104e00a69362b0b2f4b70d26efe81d.jpg)  
Fig. 10. Additional novel-view synthesis results on Mip-NeRF 360 [43] and Tanks and Temples [42]. Each scene group compares SCINeRF [14] in the upper row with our method in the lower row, while columns progress through unseen viewpoints along the rendering trajectory. The sequences provide a qualitative comparison of structural continuity and appearance consistency across the evaluated scenes and substantial viewpoint changes.

To directly evaluate proxy construction and isolate the routing decision, Table XIII compares a naive masked proxy, forced ENI, forced AMDI, and adaptive routing across six acquisition settings: two low-multiplicity, two high-mask-ratio, and two high-compression-ratio cases. All variants use the same VGGT backend and downstream SCI optimization, with $\tau _ { \mu } ~ = ~ 4$ and all AMDI parameters fixed. ENI is not a latent-view estimate; it forms a geometry-oriented proxy from normalized aggregate samples when overlap is limited. AMDI estimates local per-view contributions, while its smallestadmissible-window rule adapts the spatial support to local mask conditioning. Forced ENI and AMDI perform comparably in the low-multiplicity cases. Across the four strongermixing cases, AMDI improves over ENI by 4.40 dB PSNR, 0.0457 SSIM, and 0.0793 LPIPS on average. Adaptive routing gives the best aggregate reconstruction quality, reaching 31.72 dB PSNR, 0.8810 SSIM, and 0.1611 LPIPS, and also obtains the lowest nATE of 2.3277%.

Figures 10–12 provide additional qualitative examples covering novel-view synthesis, challenging camera trajectories, and extended synthetic reconstruction. Specifically, Fig. 10 compares novel-view synthesis on Mip-NeRF 360 [43] and Tanks and Temples [42], Fig. 11 examines complex geometry and irregular camera motion, and Fig. 12 presents additional synthetic-scene reconstructions.

![](images/e92dae726413d2d4d316c1704441e38cd8ff3a5843cc4757c1209a2c9ef7ba1d.jpg)  
Fig. 11. Additional qualitative results for scenes with complex geometry and irregular camera motion. In each scene group, the coded SCI measurement is shown on the left, followed by frames sampled along the camera trajectory. SCINeRF [14] and our reconstruction are presented in the upper and lower rows, respectively, enabling comparison of structural and appearance consistency across challenging viewpoint changes.

![](images/3125ce3533beaa9d5f2d00ac92cfe27db26c73f8ce13f7ee261b2068d9ca81ee.jpg)  
Fig. 12. Additional reconstruction results on extended synthetic benchmarks. For each scene, the SCI measurement is shown on the left, followed by the reference frames and the reconstructions of SCINeRF [14] and our method from top to bottom. Colored boxes mark corresponding local regions for closer inspection of structural and appearance differences.