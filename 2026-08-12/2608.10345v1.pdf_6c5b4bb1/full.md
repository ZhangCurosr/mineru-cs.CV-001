# CasDeblurGS: Cascaded 2D-to-3D Multi-View Consistency for 3D Gaussian Splatting from Two Blurry Images

Haeyun Choi<sup>1\*†</sup> , Minhyuk Jang<sup>2\*</sup> , and I-Gil Kim<sup>2</sup>

<sup>1</sup> University of Virginia, Charlottesville, VA, USA phh3ps@virginia.edu <sup>2</sup> KT R&D Center, Seoul, Republic of Korea {minhyuk.jang,i-gil.kim}@kt.com

https://haeyun-choi.github.io/Cascaded2D3D\_page/

Abstract. Free-viewpoint 3D scene media is increasingly important for immersive applications, yet practical capture often sufers from severe view sparsity and motion blur. Although neural rendering has advanced sparse-view synthesis, existing blur-aware methods typically require substantial multi-view redundancy, accurate camera poses, or costly perscene optimization. We address a stringent yet practical setting: reconstructing a coherent 3D scene from only two motion-blurred images with known intrinsics, without input-view poses, auxiliary sharp images, or per-scene test-time optimization. To this end, we propose CasDeblurGS, a cascaded framework that progressively recovers reliable cross-view information from local 2D correspondences to global 3D guidance. Stage 1 constructs locally reliable guidance through occlusionaware correspondence filtering, while Stage 2 aggregates the intermediate restorations into a provisional pose-free 3D Gaussian representation whose input-view re-renders provide dense global guidance for final restoration. The resulting views enable a more coherent 3D representation and higher-quality novel-view synthesis. Experiments on real-world and synthetic Deblur-NeRF scenes show consistent gains over strong baselines, improving PSNR by 1.19 dB and 2.11 dB, respectively. Progressive ablations, cross-view correspondence visualization, and camera reprojection analysis further demonstrate improvements in both rendering quality and multi-view geometric consistency.

Keywords: Gaussian Splatting · Deblurring · Sparse Reconstruction

## 1 Introduction

Recent advances in neural rendering, from neural radiance fields (NeRF) [24] to 3D Gaussian Splatting (3DGS) [12], have substantially improved the fidelity and eficiency of novel-view synthesis. Together with recent feed-forward reconstruction methods, which directly recover 3D representations from sparse images [2, 5, 9, 10, 38, 41], these advances are making free-viewpoint 3D media increasingly practical for applications such as XR, immersive telepresence, and digital content creation [1, 7, 11, 13, 44].

![](images/b57b3ce316ef9c2526df78804b34fb10213d3abbc08af94f33c6f1f043670d11.jpg)  
Fig. 1: (Left) Given two blurry images, CasDeblurGS progressively combines local 2D guidance and global 3D guidance for coherent 3D Gaussian reconstruction. (Right) Compared to CoherentGS [39], Difix3D+ [36], and GAURA [8], ours produces sharper details and more coherent novel views in the challenging two-view blurry setting.

Despite this progress, practical capture remains challenging. Dense multiview acquisition may be infeasible because of limited capture time, sensing conditions, or platform stability, forcing reconstruction to rely on only a few observations. When these sparse inputs are further degraded by motion blur from handheld imaging, platform vibration, or low-light exposure, image details and cross-view correspondences are simultaneously corrupted, making reliable scene recovery substantially more dificult.

Recent studies have tackled this challenge by incorporating blur-aware modeling into neural rendering or jointly optimizing image restoration and scene representation [6, 14, 22, 26, 46, 48]. However, these approaches remain dificult to deploy in practice. First, they typically rely on per-scene test-time optimization and substantial multi-view redundancy to stabilize the joint estimation of blur and scene structure, often requiring 20–40 input images. Such assumptions are dificult to satisfy when only a few blurry views are available and rapid reconstruction is desired. Second, they depend on reliable geometric initialization under blur, typically in the form of accurate camera poses or coarse scene geometry. Such initialization is already dificult to recover from blurry frames using standard structure-from-motion pipelines [27]. When both view redundancy and geometric initialization are weak, the problem becomes ill-posed, often leading to unstable and spurious geometry.

These limitations become even more severe in sparse-view settings, where blur further weakens the already limited cross-view constraints. Consequently, even methods designed for sparse-view blurry reconstruction remain fragile in the extreme two-view regime [15, 39]. Under such limited observations, the joint estimation of blur and scene structure becomes severely ill-conditioned, often yielding blurry renderings, noisy geometry, and floating artifacts.

In this work, we consider a stringent yet practical setting: reconstructing a coherent 3D scene from only two motion-blurred images with known camera intrinsics, without input-view camera poses, auxiliary sharp images, or per-scene test-time optimization. The central dificulty is that severe blur corrupts the already limited cross-view evidence required for both local correspondence estimation and global 3D aggregation. Directly transferring unreliable 2D correspondences can introduce misaligned structures, while aggregating inconsistent observations in 3D can propagate these errors into the reconstructed scene.

Our key idea is to recover reliable cross-view information progressively, from local 2D correspondences to global 3D guidance. We propose CasDeblurGS, a cascaded 2D-to-3D multi-view consistency framework. In Stage 1, a frozen stabilizer first produces alignment-friendly observations, after which our occlusionaware cross-view guidance module (OCGM) filters unreliable optical-flow correspondences through forward–backward consistency and constructs masked reference warps for intermediate restoration. In Stage 2, these restorations are aggregated by a frozen pose-free 3DGS backbone into a provisional 3D representation whose input-view re-renders provide dense global guidance for final restoration. The final restored views are then passed through the same frozen backbone to construct the output 3D representation for novel-view synthesis. At inference time, CasDeblurGS requires only the two blurry images and their intrinsics, with no camera poses, depth maps, external generative priors, or scene-specific optimization.

This progressive design combines complementary strengths of 2D and 3D guidance. Local warping selectively transfers fine cross-view evidence where correspondences are reliable, whereas the provisional 3D representation aggregates information from both views to provide dense scene-level feedback. Consequently, the cascade improves both the quality of the restored observations and their consistency for downstream 3D reconstruction.

Experiments on synthetic and real-world Deblur-NeRF scenes demonstrate consistent gains over strong blur-aware and restoration-based baselines. CasDeblurGS improves PSNR over the strongest competing results by 1.19 dB on realworld scenes and 2.11 dB on synthetic scenes. Progressive ablations, cross-view correspondence visualization, and camera reprojection analysis further demonstrate improvements in both novel-view synthesis quality and multi-view geometric consistency.

Our contributions are summarized as follows:

– We introduce CasDeblurGS, a pose-free feed-forward framework for 3D reconstruction from only two motion-blurred images with known intrinsics, without auxiliary sharp images or per-scene test-time optimization.

– We propose a cascaded 2D-to-3D guidance strategy that establishes locally reliable cross-view correspondences through OCGM and then exploits posefree 3D re-rendering to provide dense global guidance for final restoration.

– We demonstrate consistent improvements on real-world and synthetic Deblur-NeRF scenes, supported by progressive ablations, cross-view correspondence visualization, and camera reprojection analysis.

## 2 Related Work

## 2.1 Radiance Fields from Blurry Images

Recent advances in NeRF and 3D Gaussian Splatting (3DGS) have spurred research on modeling camera motion blur for sharp 3D scene reconstruction. A common strategy is to simulate the blur formation process by modeling camera motion during exposure as a continuous trajectory. Existing methods parameterize this trajectory using SE(3) interpolation for linear motion [32, 37, 48], Bézier curves for complex nonlinear paths [17], or Neural ODEs for flexible temporal transformations [18, 19]. However, such trajectory optimization typically requires substantial multi-view redundancy to jointly stabilize blur and geometry estimation. In sparse-view settings, insuficient multi-view constraints make this estimation unreliable; in the extreme two-view blurry regime, standard SfM initialization becomes infeasible and cross-view correspondences are severely weakened, leading to unstable geometry estimation.

Alternatively, some methods address sparse blurry inputs using 2D structural or semantic priors, or external generative models. For instance, HQGS [20] leverages 2D edges and semantic cues, S2Gaussian [31] resolves feature-space cross-view inconsistencies during 2D super-resolution, and CoherentGS [39] augments virtual views with deblurring networks and difusion priors. However, these methods rely heavily on local 2D cues or external priors. In extreme two-view settings with limited geometric constraints, they struggle to ensure global 3D coherence, often producing structural distortions and cross-view inconsistencies in the final renderings.

## 2.2 Feed-forward 3D Gaussian Splatting

While traditional 3DGS relies on per-scene optimization, recent generalizable feed-forward methods [45] directly predict 3D Gaussians from sparse views. Building on earlier generalizable NeRFs [3,33,42], 3DGS-based models now dominate this direction, using epipolar geometry, cost volumes, depth-aware matching, or hybrid volume-pixel representations for cross-view aggregation [2, 5, 29, 35, 38, 43]. Pose-free variants further remove explicit pose dependence through coarse-to-fine alignment or canonical-space prediction [9, 10, 41]. However, these methods largely assume sharp inputs and remain vulnerable to severe motion blur.

GAURA [8] jointly addresses feed-forward reconstruction and deblurring, but it assumes known camera poses, uses kernel-based synthetic blur, and remains unverified in the extreme two-view regime. More fundamentally, feed-forward aggregation depends on reliable cross-view correspondence cues, such as photometric consistency or epipolar features, which degrade severely under strong blur. This problem is amplified in two-view settings, where no additional observations are available to compensate for the lost constraints.

## 2.3 Multi-view Image Deblurring

Multi-view deblurring exploits geometric cues such as disparity and depth for restoration [25, 40, 49], but its reliance on correspondence estimation limits its efectiveness in extreme two-view blurry settings where high-frequency details are severely degraded and reliable camera poses and depth are unavailable.

Fundamentally, improving 2D image quality alone does not guarantee 3D consistency for novel view synthesis [16, 17, 22, 32]. Recent difusion-based approaches aim at 3D-consistent restoration [21,23,36], yet remain vulnerable when severe blur collapses geometric matching cues. Motivated by these limitations, we propose a cascaded design that progresses from flow-based local 2D guidance to global 3D constraints via pose-free 3DGS re-rendering, enabling stable reconstruction without external generative priors at inference time.

## 3 Preliminaries

## 3.1 3D Gaussian Splatting

3D Gaussian Splatting (3DGS) represents a scene using a set of anisotropic 3D Gaussians:

$$
\mathcal { G } = \{ g _ { m } \} _ { m = 1 } ^ { M } , \qquad g _ { m } = ( \pmb { \mu } _ { m } , \pmb { \mathrm { q } } _ { m } , \pmb { \mathrm { s } } _ { m } , \alpha _ { m } , \pmb { \mathrm { c } } _ { m } ) ,\tag{1}
$$

where $\pmb { \mu } _ { m } \in \mathbb { R } ^ { 3 }$ denotes the Gaussian center, $\mathbf { q } _ { m }$ the rotation, $\mathbf { s } _ { m }$ the scale, $\alpha _ { m }$ the opacity, and $\mathbf { c } _ { m }$ the appearance parameters, e.g., spherical harmonics coefficients [12, 41]. Given a target viewpoint $v _ { t }$ , the diferentiable 3DGS rasterizer renders an image as:

$$
\hat { I } _ { t } = \operatorname { R e n d e r } ( \mathcal { G } , v _ { t } ) .\tag{2}
$$

This formulation supports eficient diferentiable rendering and serves as the scene representation used throughout our pipeline.

## 3.2 Pose-free Feed-forward 3DGS

Pose-free feed-forward 3DGS reconstructs a scene directly from sparse images and their camera intrinsics without requiring input-view camera extrinsics. In particular, NoPoSplat [41] learns a mapping:

$$
\mathcal { G } = h _ { \eta } \big ( \{ ( I _ { i } , K _ { i } ) \} _ { i = 1 } ^ { N } \big ) ,\tag{3}
$$

where $I _ { i }$ and $K _ { i }$ denote the i-th input image and its camera intrinsics, respectively. NoPoSplat uses the first input view to establish a canonical coordinate frame and directly predicts the Gaussians associated with all input views in this shared space. This enables direct multi-view fusion without externally supplied input-view camera poses. The camera intrinsics are incorporated into the input representation to account for camera geometry and alleviate scene-scale ambiguity. The resulting Gaussian representation can then be rendered from a target viewpoint expressed in the same canonical frame using Eq. (2).

Stage 1: Local 2D Guidance & Deblur  
Stage 2: Global 3D Guidance & Deblur  
![](images/faecd108a606b953efad2746e616c7921e55fec0a54c12ce348e4af282692452.jpg)  
Fig. 2: Overview of CasDeblurGS. (Top) The cascade constructs local 2D guidance, then uses a frozen pose-free 3DGS backbone to provide global 3D guidance for final restoration. Final restorations feed the same backbone for novel-view synthesis. (Bottom) OCGM stabilizes inputs, estimates bidirectional RAFT flow, derives forward– backward consistency masks, and produces masked cross-view warps for Stage 1.

In this work, we instantiate the backbone in the two-view setting (N = 2) and keep $h _ { \eta }$ frozen throughout the pipeline. As detailed in Sec. 4, its canonical-space reconstruction and re-rendering capabilities are used for global 3D guidance and final novel-view synthesis, without scene-specific test-time optimization.

## 4 Method

Given two motion-blurred images and their known camera intrinsics, $\{ ( B _ { i } , K _ { i } ) \} _ { i = 1 } ^ { 2 } .$ CasDeblurGS progressively recovers reliable cross-view guidance to restore the input views and construct a pose-free 3D Gaussian representation for novel-view synthesis. No ground-truth or pre-estimated input-view camera poses are provided to the framework. As illustrated in Fig. 2, the pipeline comprises two cascaded restoration stages followed by final pose-free 3D reconstruction.

Stage 1 establishes locally reliable correspondence-based guidance and uses it to produce sharper intermediate restorations with improved cross-view consistency. Stage 2 aggregates these intermediate views into a provisional 3D representation and re-renders it at the input viewpoints, providing dense global guidance for final restoration. The final restored views are then processed by the same frozen 3DGS backbone to construct the output representation for novelview synthesis.

The stabilizer $S _ { \phi }$ [4], the RAFT-based optical-flow estimator [30], and the pose-free 3DGS backbone $h _ { \eta } \ [ 4 1 ]$ remain frozen throughout the pipeline. The 2Dand 3D-guided restoration networks, $\mathcal { D } _ { \theta } ^ { 2 D }$ and $\mathcal { D } _ { \psi } ^ { 3 D }$ , are trained ofline in a stagewise manner. All modules are fixed at inference, and CasDeblurGS therefore requires no scene-specific test-time optimization.

The key design of CasDeblurGS is the progressive construction of reliable cross-view guidance. Stage 1 suppresses unreliable local correspondences through occlusion-aware masking, whereas Stage 2 lifts the intermediate restorations into a shared 3D representation and projects the aggregated scene information back to the image plane. Algorithm 1 summarizes the complete inference pipeline. Training and implementation details are provided in Sec. 5.1 and the supplementary material.

## 4.1 Stage 1: Local 2D Guidance and Deblurring

Stage 1 constructs locally reliable correspondence-based guidance from the two input views. It transfers cross-view information only where the estimated correspondence is suficiently trustworthy. The stage consists of alignment preconditioning, occlusion-aware cross-view guidance generation, and 2D-guided restoration.

Alignment preconditioning with a Stabilizer. Directly estimating optical flow between severely motion-blurred inputs is unreliable because blur suppresses and distorts structures shared across views. We therefore first process each input using a frozen pretrained stabilizer:

$$
C _ { 1 } = S _ { \phi } ( B _ { 1 } ) , \qquad C _ { 2 } = S _ { \phi } ( B _ { 2 } ) .\tag{4}
$$

The stabilized views $( C _ { 1 } , C _ { 2 } )$ are not treated as final restorations. Instead, they serve as alignment-friendly observations from which more reliable cross-view correspondences can be estimated.

Occlusion-aware cross-view guidance. From the stabilized views, we estimate bidirectional optical flow using a frozen RAFT model [30]:

$$
F _ { 1  2 } = \mathrm { R A F T } ( C _ { 2 } , C _ { 1 } ) , F _ { 2  1 } = \mathrm { R A F T } ( C _ { 1 } , C _ { 2 } ) .\tag{5}
$$

Here, $F _ { b  r }$ denotes a flow field defined on the base-view coordinates of $b ,$ used to map content from the reference view r into view b. Not every estimated correspondence provides reliable restoration guidance. Occlusions, motion boundaries, out-of-bounds projections, and blur-induced flow errors can introduce misaligned or duplicated structures. We therefore validate each correspondence using forward–backward consistency. For a pixel location x in the base view, its mapped coordinate in the reference view is

$$
x ^ { \prime } = x + F _ { b  r } ( x ) .\tag{6}
$$

The corresponding forward–backward residual is

$$
e ( x ) = \| F _ { b  r } ( x ) + F _ { r  b } ( x ^ { \prime } ) \| _ { 2 } .\tag{7}
$$

To avoid over-rejecting valid correspondences under large displacements, we use the motion-adaptive threshold

$$
T ( \boldsymbol { x } ) = \tau + \alpha ( \lVert \boldsymbol { F } _ { b  r } ( \boldsymbol { x } ) \rVert _ { 2 } + \lVert \boldsymbol { F } _ { r  b } ( \boldsymbol { x } ^ { \prime } ) \rVert _ { 2 } ) ,\tag{8}
$$

where τ provides a base tolerance for small estimation and resampling errors, and α adjusts the tolerance according to the motion magnitude. The validity mask is defined as

$$
M _ { b  r } ( x ) = \mathcal { k } [ \mathrm { i n - b o u n d s } ( x ^ { \prime } ) ] \cdot \mathcal { k } [ e ( x ) \leq T ( x ) ] .\tag{9}
$$

Only correspondences satisfying both the in-bounds and forward–backward consistency conditions are retained. We then construct the masked cross-view warp

$$
W _ { b  r } = M _ { b  r } \odot \mathcal { W } \big ( C _ { r } , F _ { b  r } \big ) ,\tag{10}
$$

where W denotes backward warping. We refer to this combination of bidirectional flow estimation, forward–backward validation, and masked warping as the occlusion-aware cross-view guidance module (OCGM). Applying OCGM in both directions produces the guidance–mask pairs $( W _ { 1  2 } , M _ { 1  2 } )$ and $( W _ { 2  1 } , M _ { 2  1 } )$

2D-guided deblurring. For each view, we concatenate the original blurry input, the corresponding masked cross-view warp, and its validity mask:

$$
\begin{array} { r } { \tilde { S } _ { 1 } = \mathcal { D } _ { \theta } ^ { 2 D } \big ( \mathrm { c o n c a t } ( B _ { 1 } , W _ { 1  2 } , M _ { 1  2 } ) \big ) , } \\ { \tilde { S } _ { 2 } = \mathcal { D } _ { \theta } ^ { 2 D } \big ( \mathrm { c o n c a t } ( B _ { 2 } , W _ { 2  1 } , M _ { 2  1 } ) \big ) . } \end{array}\tag{11}
$$

Using the original blurry image as the base preserves image evidence that may be altered by the stabilizer, while the masked warp transfers complementary information from the other view. The explicit validity mask additionally indicates where the cross-view guidance can be trusted. The resulting intermediate restorations $( \tilde { S } _ { 1 } , \tilde { S } _ { 2 } )$ provide sharper observations with improved cross-view consistency for constructing the provisional 3D representation in Stage 2.

## 4.2 Stage 2: Global 3D Guidance and Deblurring

Stage 2 aggregates the intermediate restorations into a provisional 3D representation and projects the resulting scene-level information back to the image plane for final restoration. While Stage 1 transfers locally reliable information through correspondence-based masked warping, Stage 2 provides dense global guidance induced by a shared 3D representation.

Provisional 3D representation and re-render guidance. We feed the intermediate restorations $( \tilde { S } _ { 1 } , \tilde { S } _ { 2 } )$ and their camera intrinsics into the frozen posefree 3DGS backbone:

$$
\mathcal { G } = h _ { \eta } \big ( ( \tilde { S } _ { 1 } , K _ { 1 } ) , ( \tilde { S } _ { 2 } , K _ { 2 } ) \big ) .\tag{12}
$$

We then re-render the resulting 3D Gaussian representation at the two input viewpoints:

$$
R _ { 1 } = \operatorname { R e n d e r } ( \mathcal { G } , v _ { 1 } ) , \qquad R _ { 2 } = \operatorname { R e n d e r } ( \mathcal { G } , v _ { 2 } ) ,\tag{13}
$$

where $v _ { 1 }$ denotes the first input viewpoint in the canonical camera frame established by the pose-free backbone, and $v _ { 2 }$ is determined by the relative camera geometry inferred for the second view. Both viewpoints are internal to the backbone’s canonical-space formulation; no ground-truth or pre-estimated input-view poses are provided to CasDeblurGS. Because $\mathcal { G }$ is jointly constructed from both intermediate restorations, the re-rendered images $( R _ { 1 } , R _ { 2 } )$ encode scene information aggregated across the two views and provide dense 3D guidance over the full image plane.

3D-guided deblurring. For each view, we combine the original blurry input with its corresponding re-render guidance:

$$
\begin{array} { r } { \hat { S } _ { 1 } = \mathcal { D } _ { \psi } ^ { 3 D } \big ( \mathrm { c o n c a t } ( B _ { 1 } , R _ { 1 } ) \big ) , } \\ { \hat { S } _ { 2 } = \mathcal { D } _ { \psi } ^ { 3 D } \big ( \mathrm { c o n c a t } ( B _ { 2 } , R _ { 2 } ) \big ) . } \end{array}\tag{14}
$$

The original blurry input preserves view-specific image evidence, while the rerendered view supplies complementary scene-level structure aggregated through the shared 3D representation. Unlike the spatially selective correspondence guidance used in Stage 1, the re-rendered images provide dense guidance across the full image plane. Stage 2 thereby produces the final restored views $( \hat { S } _ { 1 } , \hat { S } _ { 2 } )$ using information jointly derived from both intermediate restorations.

## 4.3 Final 3D Representation for NVS

We apply the same frozen pose-free 3DGS backbone once more to the final restored views:

$$
\mathcal { G } ^ { \star } = h _ { \eta } \big ( ( \hat { S } _ { 1 } , K _ { 1 } ) , ( \hat { S } _ { 2 } , K _ { 2 } ) \big ) .\tag{15}
$$

The provisional representation $\mathcal { G }$ is constructed from the intermediate restorations solely to generate the re-render guidance used in Stage 2. In contrast, $\mathcal G ^ { \star }$ is constructed from the final restored views and serves as the output 3D representation for novel-view synthesis.

Given $\mathcal G ^ { \star }$ , we render novel viewpoints using the diferentiable 3DGS rasterizer. CasDeblurGS performs no per-scene or test-time optimization of $\mathcal G ^ { \star }$ . Instead, the proposed cascade supplies the frozen pose-free backbone with sharper and more mutually consistent input observations, enabling it to construct a more coherent final 3D representation.

Algorithm 1 Inference Pipeline of CasDeblurGS   
Require: Two blurry images and intrinsics $\{ ( B _ { i } , K _ { i } ) \} _ { i = 1 } ^ { 2 }$   
Ensure: Restored images $\bar { \bf \Phi } ( \hat { S } _ { 1 } , \hat { S } _ { 2 } )$ and 3D representation ${ \mathcal { G } } ^ { \star }$   
Note. All modules are frozen at inference.   
Stage 1: Local 2D Guidance and Deblurring   
1: $C _ { 1 } \gets S _ { \phi } ( B _ { 1 } )$ \triangleright stabilization   
2: $C _ { 2 }  S _ { \phi } ( B _ { 2 } )$   
3: $F _ { 1  2 }  \mathrm { R A F T } ( C _ { 2 } , C _ { 1 } )$ \triangleright optical flow   
4: $F _ { 2  1 }  \mathrm { R A F T } ( C _ { 1 } , C _ { 2 } )$   
5: $M _ { 1  2 }  \mathrm { F B M a s k } ( F _ { 1  2 } , F _ { 2  1 } )$ \triangleright validity mask   
6: $M _ { 2  1 }  \mathrm { F B M a s k } ( F _ { 2  1 } , F _ { 1  2 } )$   
7: $W _ { 1  2 }  M _ { 1  2 } \odot \mathcal { W } ( C _ { 2 } , F _ { 1  2 } )$ \triangleright masked warp   
8: $W _ { 2  1 }  M _ { 2  1 } \odot \mathcal { W } ( C _ { 1 } , F _ { 2  1 } )$   
9: $\tilde { S } _ { 1 }  \mathcal { D } _ { \theta _ { - } } ^ { 2 D } ( \mathrm { c o n c a t } ( B _ { 1 } , W _ { 1  2 } , M _ { 1  2 } ) )$ \triangleright 2D guidance   
10: $\tilde { S } _ { 2 } \gets \mathcal { D } _ { \theta } ^ { 2 D } ( \mathrm { c o n c a t } ( B _ { 2 } , W _ { 2  1 } , M _ { 2  1 } ) )$   
Stage 2: Global 3D Guidance and Deblurring   
11: $\mathcal { G }  h _ { \eta } \big ( ( \tilde { S } _ { 1 } , K _ { 1 } ) , ( \tilde { S } _ { 2 } , K _ { 2 } ) \big )$ \triangleright pose-free 3DGS backbone   
12: $R 1  \mathrm { R e n d e r } ( \mathcal { G } , v _ { 1 } )$ \triangleright re-render   
13: $R _ { 2 }  \mathrm { R e n d e r } ( \mathcal { G } , v _ { 2 } )$   
14: $\hat { S } _ { 1 }  \mathcal { D } _ { \psi } ^ { 3 D } ( \mathrm { c o n c a t } ( \hat { B } _ { 1 } , R _ { 1 } ) )$ \triangleright 3D guidance   
15: $\hat { S } _ { 2 }  \mathcal { D } _ { \psi } ^ { 3 D } ( \mathrm { c o n c a t } ( B _ { 2 } , R _ { 2 } ) )$   
Final 3D Representation for Novel View Synthesis   
16: $\mathcal { G } ^ { \star }  h _ { \eta } \big ( ( \hat { S } _ { 1 } , K _ { 1 } ) , ( \hat { S } _ { 2 } , K _ { 2 } ) \big )$   
17: return $( \hat { S } _ { 1 } , \hat { S } _ { 2 } , \mathcal { G } ^ { \star } )$

## 5 Experiments

## 5.1 Experimental Setup

Datasets and two-view evaluation protocol. We train our restoration networks on the synthetic camera-motion-blur dataset introduced in DeepDeblurRF [6] and evaluate on the camera-motion-blur subset of Deblur-NeRF [22], which comprises five synthetic scenes and ten real-world scenes. For a fair comparison, we evaluate all methods on 85 fixed tuples across 15 scenes: 25 synthetic and 60 real-world. Each tuple contains two blurry inputs and one held-out target view for novel-view synthesis. The complete input–target mappings are provided in the supplementary material. All images are resized with preserved aspect ratio and center-cropped to $2 5 6 \times 2 5 6$ to match the pose-free 3DGS backbone [41].

Compared methods. We compare CasDeblurGS with representative sparseview 3DGS methods. SE-GS [47] performs per-scene optimization for few-shot novel-view synthesis, while GAURA [8] is a generalizable feed-forward method for sparse blurry inputs. CoherentGS [39] targets sparse motion blur using video difusion priors and alternating per-scene optimization.

To test whether stronger 2D restoration before reconstruction is suficient, we construct two deblurring-then-reconstruction baselines. Specifically, we pair the same frozen pose-free 3DGS backbone used in CasDeblurGS with DAVANet [49], a stereo deblurring network, and Difix3D+ [36], a difusion-based 3D enhancement method. For these controls, the $\mathrm { ^ { 6 4 } P o s e { – } F r e e { ' } }$ and “Generalizable” refer to the integrated pipelines rather than to the restoration models alone. All methods use the same two-view protocol and input–target mappings (see Supplementary Material), with oficial implementations adapted where necessary.

Table 1: Novel-view synthesis results on real-world and synthetic Deblur-NeRF scenes. DAVANet and Difix3D+ use the same frozen pose-free 3DGS backbone as ours. Best and second-best are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Pose-Free Generalizable</td><td rowspan="2"></td><td colspan="3">Real-world</td><td colspan="3">Synthetic</td></tr><tr><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>SE-GS [47]</td><td>X</td><td>X</td><td>15.90</td><td>0.398</td><td>0.443</td><td>18.76</td><td>0.560</td><td>0.338</td></tr><tr><td>GAURA [8]</td><td>X</td><td>√</td><td>17.67</td><td>0.508</td><td>0.444</td><td>17.91</td><td>0.524</td><td>0.414</td></tr><tr><td>CoherentGS [39]</td><td>X</td><td>X</td><td>19.90</td><td>0.660</td><td>0.292</td><td>21.11</td><td>0.687</td><td>0.259</td></tr><tr><td>DAVANet [49]</td><td>√</td><td>√</td><td>19.34</td><td>0.573</td><td>0.360</td><td>19.61</td><td>0.625</td><td>0.292</td></tr><tr><td>Difix3D+ [36]</td><td>√</td><td>√</td><td>19.94</td><td>0.611</td><td>0.311</td><td>21.30</td><td>0.673</td><td>0.262</td></tr><tr><td>Ours</td><td>√</td><td>√</td><td>21.13</td><td>0.700</td><td>0.228</td><td>23.41</td><td>0.796</td><td>0.165</td></tr></table>

Implementation details. We use a pretrained NAFNet [4] as the frozen stabilizer $S _ { \phi }$ and RAFT-Large [30] as the frozen optical-flow estimator. For $h _ { \eta } ,$ we use the NoPoSplat checkpoint [41] trained on RealEstate10K with two $2 5 6 \times 2 5 6$ inputs. Both restoration networks, $\mathcal { D } _ { \theta } ^ { 2 D }$ and $\mathcal { D } _ { \psi } ^ { 3 D }$ , follow the NAFNet architecture with a base width of 64.

Stage 1 takes a 7-channel concatenation of the blurry base view, masked crossview warp, and validity mask, whereas Stage 2 takes a 6-channel concatenation of the blurry base view and its 3D re-render guidance. We train the two restoration networks stage-wise. The Stage 1 network is trained first, after which it is fixed while training the Stage 2 network.

Each stage is optimized for 400K iterations using AdamW with an initial learning rate of $1 0 ^ { - 3 }$ , weight decay of $1 0 ^ { - 3 }$ , and $\beta _ { 1 } = \beta _ { 2 } = 0 . 9$ . We use cosine learning-rate decay with a minimum learning rate of $1 0 ^ { - 7 }$ and a batch size of 32 per GPU on two NVIDIA A100 GPUs. The training objective combines an $\ell _ { 1 }$ reconstruction loss and a VGG-based perceptual loss [28,34]. For OCGM, we set the forward–backward consistency parameters to $\tau = 1 . 0$ and $\alpha = 0 . 0 1$ in all experiments. Sensitivity analysis for these parameters, runtime comparisons, and additional implementation details are provided in the supplementary material.

Evaluation metrics. We evaluate novel-view renderings against the corresponding sharp reference images using PSNR, SSIM, and LPIPS. We report the average over all fixed evaluation tuples separately for the synthetic and realworld subsets. Higher PSNR and SSIM indicate better reconstruction quality, whereas lower LPIPS indicates greater perceptual similarity.

![](images/e257ec221981dd385015962137f60785ec8750ce84a27ee20c811e13102db64b.jpg)  
Fig. 3: Qualitative novel-view synthesis comparisons on real-world and synthetic Deblur-NeRF scenes. The upper vehicle scene is real-world and the lower indoor scene synthetic. The Inputs column shows the two blurry views. For each method, the upper image shows the rendered target view, while the lower enlarges the red-boxed region.

## 5.2 Comparison with Prior Methods

Quantitative results. Table 1 compares CasDeblurGS with prior methods on the real-world and synthetic Deblur-NeRF subsets. Our method achieves the best performance across all metrics on both subsets. On real-world scenes, Cas-DeblurGS obtains 21.13 dB PSNR, 0.700 SSIM, and 0.228 LPIPS, outperforming the best competing results by 1.19 dB, 0.040, and 0.064, respectively. The gains are larger on synthetic scenes, where our method improves PSNR by 2.11 dB, SSIM by 0.109, and LPIPS by 0.094.

The gaps also reflect diferences between the baselines’ native settings and our extreme two-view regime. SE-GS addresses few-shot reconstruction from sparse posed views but does not explicitly handle blur-corrupted correspondences. GAURA is trained with 8–12 source views and uses 10 views at inference, leaving substantially less multi-view redundancy for epipolar aggregation when restricted to two inputs. CoherentGS directly targets sparse motion blur but reports configurations with 3, 6, or 9 views, leaving its alternating reconstruction and generative expansion more weakly constrained under only two inputs.

The consistent gains over DAVANet and Difix3D+, which use the same frozen NoPoSplat backbone, show that applying a strong 2D restoration model before reconstruction is insuficient. Instead, the results demonstrate the benefit of progressively incorporating reliable local correspondence cues and global 3D guidance. Moreover, CasDeblurGS remains pose-free and generalizable without per-scene optimization.

![](images/a0a2c47c813b992be4161aa2521bc6e3ace91884559371663e6845627fa07e56.jpg)  
SSIM: 0.691 | Matches: 96  
SSIM: 0.703 | Matches: 112  
SSIM: 0.666 | Matches: 77

Qualitative results. As shown in Fig. 3, SE-GS and GAURA retain substantial blur, while CoherentGS and the restoration-based baselines partially reduce blur but oversmooth details or introduce structural artifacts. These limitations are particularly visible around the vehicle grille in the real-world example and the table edges and legs in the synthetic example. In contrast, CasDeblurGS recovers sharper boundaries and structures that more closely match the groundtruth views. These results show that the cascade improves both sharpness and structural fidelity in novel-view renderings. Additional qualitative results are provided in the supplementary material.

## 5.3 Ablation Studies

Progressive contribution of the cascade. Table 2 shows consistent gains as the stabilizer, Stage 1 2D guidance, and Stage 2 3D guidance are progressively introduced. The stabilizer provides an initial improvement by producing more alignment-friendly observations. Stage 1 yields the largest incremental PSNR gain (+0.58 dB), confirming the benefit of locally reliable cross-view correspondences. Stage 2 further improves PSNR by 0.25 dB and SSIM by 0.016, indicating that dense global guidance resolves residual inconsistencies that cannot be addressed by 2D warping alone. Additional component diagnostics are provided in the supplementary material.

Table 2: Ablation on real-world scenes.
<table><tr><td></td><td>Blurry</td><td>+Stab.</td><td>+2D</td><td>+3D</td></tr><tr><td>PSNR ↑</td><td>20.07</td><td>20.30</td><td>20.88</td><td>21.13</td></tr><tr><td>SSIM ↑</td><td>0.612</td><td>0.655</td><td>0.684</td><td>0.700</td></tr></table>

Cross-view correspondence analysis. Figure 4 visualizes the progressive improvement in cross-view consistency. On BlurBasket, the number of matches increases from 69 for the blurry inputs to 77 after stabilization, 96 after Stage 1, and 112 after Stage 2. The limited gain from stabilization reflects its independent per-view processing, whereas Stage 1 establishes more reliable local correspondences through OCGM. Stage 2 further improves correspondence recovery in regions that remain challenging for 2D warping alone, such as the basket handle highlighted by the red arrows. These results show that the cascade progressively improves both local correspondence reliability and global structural consistency.

![](images/5ac36b978f30dac8726d155e3b7239f7fb92918872be78ef60c81ce0332053f9.jpg)

Stabilizer  
![](images/464905f0a78f7bb5d04a7df9b4c3c71c9ae90f53b5a086a329e22ae592836c2c.jpg)

2D-guided Deblurring  
![](images/59a5798784c8319ed8ca44831b4c61d079388b0fe75d23dc1e487f7ce5f36642.jpg)  
3D-guided Deblurring

![](images/e1b65175089199ca0016a7e5de658228676851508c4c4d3b72a51b00d397baa9.jpg)  
SSIM: 0.629 | Matches: 69

![](images/cf7523b8de1bf11e719387826a53030ca4d1d57f01bb2b8abb27dd5d263b8c47.jpg)

![](images/2c386e4daa7d8bebf8156a641380da3adf8995726e953e9744d12bb85ae428b6.jpg)  
Fig. 4: Progressive ablation on real-world BlurBasket scene. Green lines denote crossview 2D matches between input views, and red arrows highlight the basket handle.

Table 3: Camera reprojection analysis on real-world scenes.
<table><tr><td>Stage</td><td>Valid pts ↑</td><td>Reproj. err. ↓</td><td>Inliers@1px ↑</td><td>Inlier ratio ↑</td></tr><tr><td>Stage 1</td><td>95.1</td><td>0.2215</td><td>85.4</td><td>0.9168</td></tr><tr><td>Stage 2</td><td>126.3</td><td>0.2001</td><td>113.5</td><td>0.9343</td></tr></table>

Camera reprojection analysis. To examine whether Stage 2 improves crossview geometric consistency rather than merely image sharpness, we conduct the reprojection analysis reported in Tab. 3. For each restored input pair, we extract mutual SIFT matches, triangulate them using the benchmark camera projection matrices, and compute symmetric reprojection errors in both views. We report averages across the evaluation pairs for valid triangulated points, median reprojection error, one-pixel inliers, and inlier ratio. The benchmark camera poses are used only for this evaluation and are never provided to our pose-free reconstruction pipeline. Compared with Stage 1, Stage 2 produces approximately 33% more valid triangulated points and one-pixel inliers while reducing reprojection error by 9.7%. The inlier ratio increases from 0.9168 to 0.9343, indicating that 3D re-render guidance improves the geometric consistency of the restored views.

## 6 Conclusion

We presented CasDeblurGS, a cascaded framework for pose-free 3D Gaussian Splatting from two motion-blurred images with known camera intrinsics. Our method progressively recovers reliable cross-view information: Stage 1 constructs locally trustworthy 2D guidance via occlusion-aware correspondence filtering, while Stage 2 uses pose-free 3D re-rendering to provide dense global guidance for final restoration. The resulting views enable coherent 3D reconstruction with a frozen feed-forward 3DGS backbone without input-view poses, auxiliary sharp images, or per-scene test-time optimization. Experiments on real-world and synthetic Deblur-NeRF scenes demonstrate consistent gains over sparse-view reconstruction and restoration baselines. Progressive ablations, reprojection analysis, and cross-view correspondence visualization further show that the proposed cascade improves both rendering quality and multi-view geometric consistency.

Limitations and future work. Our framework assumes known intrinsics and static scenes in a two-view setting. Its correspondence guidance may deteriorate under extreme blur, large occlusions, or limited overlap. Future work will address unknown intrinsics, dynamic scenes, and broader sparse-view settings.

## Acknowledgements

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2026-25522885, Development of a World Foundation Model for Training and Deployment of Physical AI Systems).

## References

1. Bao, Y., Ding, T., Huo, J., Liu, Y., Li, Y., Li, W., Gao, Y., Luo, J.: 3d gaussian splatting: Survey, technologies, challenges, and opportunities. IEEE Transactions on Circuits and Systems for Video Technology 35(7), 6832–6852 (2025)

2. Charatan, D., Li, S.L., Tagliasacchi, A., Sitzmann, V.: pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 19457– 19467 (2024)

3. Chen, A., Xu, Z., Zhao, F., Zhang, X., Xiang, F., Yu, J., Su, H.: Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 14124–14133 (2021)

4. Chen, L., Chu, X., Zhang, X., Sun, J.: Simple baselines for image restoration. In: European conference on computer vision. pp. 17–33. Springer (2022)

5. Chen, Y., Xu, H., Zheng, C., Zhuang, B., Pollefeys, M., Geiger, A., Cham, T.J., Cai, J.: Mvsplat: Eficient 3d gaussian splatting from sparse multi-view images. In: European conference on computer vision. pp. 370–386. Springer (2024)

6. Choi, H., Yang, H., Han, J., Cho, S.: Exploiting deblurring networks for radiance fields. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 6012–6021 (2025)

7. Gao, R., Holynski, A., Henzler, P., Brussee, A., Martin-Brualla, R., Srinivasan, P., Barron, J.T., Poole, B.: Cat3d: Create anything in 3d with multi-view difusion models. arXiv preprint arXiv:2405.10314 (2024)

8. Gupta, V., Girish, R.S.V., Mukund Varma, T., Tewari, A., Mitra, K.: Gaura: Generalizable approach for unified restoration and rendering of arbitrary views. In: European Conference on Computer Vision. pp. 249–266. Springer (2024)

9. Hong, S., Jung, J., Shin, H., Han, J., Yang, J., Luo, C., Kim, S.: Pf3plat: Pose-free feed-forward 3d gaussian splatting. arXiv preprint arXiv:2410.22128 (2024)

10. Jiang, L., Mao, Y., Xu, L., Lu, T., Ren, K., Jin, Y., Xu, X., Yu, M., Pang, J., Zhao, F., et al.: Anysplat: Feed-forward 3d gaussian splatting from unconstrained views. ACM Transactions on Graphics (TOG) 44(6), 1–16 (2025)

11. Joshi, N., Carney, J., Kuo, N., Li, H., Peng, C., Brown, M.: Unconstrained large-scale 3d reconstruction and rendering across altitudes. arXiv preprint arXiv:2505.00734 (2025)

12. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics 42(4), 1–14 (2023)

13. Khan, M., Fazlali, H., Sharma, D., Cao, T., Bai, D., Ren, Y., Liu, B.: Autosplat: Constrained gaussian splatting for autonomous driving scene reconstruction. In: 2025 IEEE International Conference on Robotics and Automation (ICRA). pp. 8315–8321. IEEE (2025)

14. Lee, B., Lee, H., Sun, X., Ali, U., Park, E.: Deblurring 3d gaussian splatting. In: European Conference on Computer Vision. pp. 127–143. Springer (2024)

15. Lee, D., Kim, D., Lee, J., Lee, M., Lee, S., Lee, S.: Sparse-derf: Deblurred neural radiance fields from sparse view. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)

16. Lee, D., Lee, M., Shin, C., Lee, S.: Dp-nerf: Deblurred neural radiance field with physical scene priors. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12386–12396 (2023)

17. Lee, D., Oh, J., Rim, J., Cho, S., Lee, K.M.: Exblurf: Eficient radiance fields for extreme motion blurred images. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 17639–17648 (2023)

18. Lee, J., Kim, D., Lee, D., Cho, S., Lee, M., Lee, S.: Crim-gs: Continuous rigid motion-aware gaussian splatting from motion-blurred images. arXiv preprint arXiv:2407.03923 (2024)

19. Lee, J., Kim, D., Lee, D., Cho, S., Lee, M., Lee, W., Kim, T., Wee, D., Lee, S.: Comogaussian: Continuous motion-aware gaussian splatting from motion-blurred images. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 26415–26424 (2025)

20. Lin, X., Luo, S., Shan, X., Zhou, X., Ren, C., Qi, L., Yang, M.H., Vasconcelos, N.: Hqgs: High-quality novel view synthesis with gaussian splatting in degraded scenes. In: The Thirteenth International Conference on Learning Representations (2025)

21. Luo, Y., Zhou, S., Lan, Y., Pan, X., Loy, C.C.: 3denhancer: Consistent multi-view difusion for 3d enhancement. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 16430–16440 (2025)

22. Ma, L., Li, X., Liao, J., Zhang, Q., Wang, X., Wang, J., Sander, P.V.: Deblurnerf: Neural radiance fields from blurry images. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12861–12870 (2022)

23. Mao, Y., Wang, B., Kulkarni, N., Park, J.J.: Sir-dif: Sparse image sets restoration with multi-view difusion model. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 21620–21630 (2025)

24. Mildenhall, B., Srinivasan, P., Tancik, M., Barron, J., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In: European conference on computer vision (2020)

25. Pan, L., Dai, Y., Liu, M., Porikli, F.: Simultaneous stereo video deblurring and scene flow estimation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4382–4391 (2017)

26. Peng, C., Tang, Y., Zhou, Y., Wang, N., Liu, X., Li, D., Chellappa, R.: Bags: Blur agnostic gaussian splatting through multi-scale kernel modeling. In: European Conference on Computer Vision. pp. 293–310. Springer (2024)

27. Schonberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4104–4113 (2016)

28. Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 (2014)

29. Tang, S., Ye, W., Ye, P., Lin, W., Zhou, Y., Chen, T., Ouyang, W.: Hisplat: Hierarchical 3d gaussian splatting for generalizable sparse-view reconstruction. arXiv preprint arXiv:2410.06245 (2024)

30. Teed, Z., Deng, J.: Raft: Recurrent all-pairs field transforms for optical flow. In: European conference on computer vision. pp. 402–419. Springer (2020)

31. Wan, Y., Shao, M., Cheng, Y., Zuo, W.: S2gaussian: Sparse-view super-resolution 3d gaussian splatting. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 711–721 (2025)

32. Wang, P., Zhao, L., Ma, R., Liu, P.: Bad-nerf: Bundle adjusted deblur neural radiance fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4170–4179 (2023)

33. Wang, Q., Wang, Z., Genova, K., Srinivasan, P.P., Zhou, H., Barron, J.T., Martin-Brualla, R., Snavely, N., Funkhouser, T.: Ibrnet: Learning multi-view image-based rendering. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4690–4699 (2021)

34. Wang, X., Xie, L., Dong, C., Shan, Y.: Real-esrgan: Training real-world blind superresolution with pure synthetic data. In: International Conference on Computer Vision Workshops (ICCVW) (2021)

35. Wei, D., Li, Z., Liu, P.: Omni-scene: Omni-gaussian representation for ego-centric sparse-view scene reconstruction. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 22317–22327 (2025)

36. Wu, J.Z., Zhang, Y., Turki, H., Ren, X., Gao, J., Shou, M.Z., Fidler, S., Gojcic, Z., Ling, H.: Difix3d+: Improving 3d reconstructions with single-step difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 26024–26035 (2025)

37. Wu, R., Zhang, Z., Chen, M., Yan, Z., Zuo, W.: Deblur4dgs: 4d gaussian splatting from blurry monocular video. arXiv preprint arXiv:2412.06424 (2024)

38. Xu, H., Peng, S., Wang, F., Blum, H., Barath, D., Geiger, A., Pollefeys, M.: Depthsplat: Connecting gaussian splatting and depth. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 16453–16463 (2025)

39. Xu, Z., Feng, C., Li, Y., Zhao, J., Yang, J., Yu, W., Yuan, L., Tian, Y.: Breaking the vicious cycle: Coherent 3d gaussian splatting from sparse and motion-blurred views. arXiv preprint arXiv:2512.10369 (2025)

40. Yan, B., Ma, C., Bare, B., Tan, W., Hoi, S.C.: Disparity-aware domain adaptation in stereo image restoration. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13179–13187 (2020)

41. Ye, B., Liu, S., Xu, H., Li, X., Pollefeys, M., Yang, M.H., Peng, S.: No pose, no problem: Surprisingly simple 3D gaussian splats from sparse unposed images. In: International Conference on Learning Representations (2025)

42. Yu, A., Ye, V., Tancik, M., Kanazawa, A.: pixelnerf: Neural radiance fields from one or few images. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4578–4587 (2021)

43. Zhang, C., Xu, H., Wu, Q., Gambardella, C.C., Phung, D., Cai, J.: Pansplat: 4k panorama synthesis with feed-forward gaussian splatting. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 11437–11447 (2025)

44. Zhang, D., Li, G., Li, J., Bressieux, M., Hilliges, O., Pollefeys, M., Van Gool, L., Wang, X.: Egogaussian: Dynamic scene understanding from egocentric video with 3d gaussian splatting. In: 2025 International Conference on 3D Vision (3DV). pp. 1091–1102. IEEE (2025)

45. Zhang, J., Li, Y., Chen, A., Xu, M., Liu, K., Wang, J., Long, X.X., Liang, H., Xu, Z., Su, H., et al.: Advances in feed-forward 3d reconstruction and view synthesis: A survey. arXiv preprint arXiv:2507.14501 (2025)

46. Zhao, A., Yu, P., Zhu, Z., Wei, M.: Bsgs: Bi-stage 3d gaussian splatting for camera motion deblurring. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 8351–8359 (2025)

47. Zhao, C., Wang, X., Zhang, T., Javed, S., Salzmann, M.: Self-ensembling gaussian splatting for few-shot novel view synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4940–4950 (2025)

48. Zhao, L., Wang, P., Liu, P.: Bad-gaussians: Bundle adjusted deblur gaussian splatting. In: European Conference on Computer Vision. pp. 233–250. Springer (2024)

49. Zhou, S., Zhang, J., Zuo, W., Xie, H., Pan, J., Ren, J.S.: Davanet: Stereo deblurring with view aggregation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10996–11005 (2019)

# Supplementary Material: CasDeblurGS: Cascaded 2D-to-3D Multi-View Consistency for 3D Gaussian Splatting from Two Blurry Images

Haeyun Choi<sup>1\*†</sup> , Minhyuk Jang<sup>2\*</sup> , and I-Gil Kim<sup>2</sup>

<sup>1</sup> University of Virginia, Charlottesville, VA, USA phh3ps@virginia.edu <sup>2</sup> KT R&D Center, Seoul, Republic of Korea {minhyuk.jang,i-gil.kim}@kt.com

## Supplementary Contents

Additional Implementation Details . 2   
1.1 Training and Evaluation Data . . 2   
1.2 Frozen Modules and OCGM Implementation . 2   
1.3 Frozen 3DGS Backbone and Input-View Re-Rendering . 3   
1.4 Training Details . . 3   
1.5 Baseline Adaptation Details 4   
2 Qualitative Analysis of OCGM Thresholds . 4   
3 Additional Component Diagnostics . 6   
4 Sparse Camera Reprojection Analysis 7   
5 Runtime Analysis . 8   
6 Additional Experimental Results . 8   
7 Supplementary Video . 9

## 1 Additional Implementation Details

In this section, we provide additional implementation details for reproducibility, including dataset construction, frozen modules, stage-wise training, and baseline adaptation.

## 1.1 Training and Evaluation Data

For training and validation, we use the synthetic camera-motion-blur dataset introduced in DeepDeblurRF [3]. The training split contains 65 scenes, each with 29 viewpoints. For each viewpoint, one blurry image and its corresponding sharp target are provided. The validation split contains 10 additional scenes with the same structure. Blurred images are generated by simulating 6-DoF camera motion during exposure: multiple intermediate views are rendered along a smooth camera trajectory, averaged in linear color space, and paired with the temporally central frame as the ground-truth sharp image.

For evaluation, we use the camera-motion-blur subset of Deblur-NeRF [5], which contains five synthetic scenes and ten real-world scenes. Each scene provides motion-blurred input images and blur-free ground-truth target views for evaluation. To ensure fair and reproducible comparison in the extreme two-view setting, we use a fixed set of per-scene evaluation instances, where each instance is defined by two blurry input views and one held-out target view for novel view synthesis. The complete index mappings are provided in Tab. 1, and all compared methods are evaluated using the same instance definitions. For both training and evaluation, all images are first resized while preserving aspect ratio and then center-cropped to 256×256, following PixelSplat-style preprocessing [1].

## 1.2 Frozen Modules and OCGM Implementation

We use the width-64 NAFNet [2] pretrained on GoPro [6] as the frozen stabilizer $S _ { \phi }$ , and RAFT-Large [8] from Torchvision [9] for optical flow estimation in OCGM. For each pair of stabilized views, we estimate bidirectional optical flow and compute validity masks using forward–backward consistency together with out-of-bounds checks. A pixel is marked invalid if its mapped coordinate falls outside the image boundary or if the forward–backward residual exceeds the motion-adaptive threshold defined in the main paper. These masks suppress unreliable regions caused by occlusions, motion boundaries, and blur-induced mismatches before masked warps are constructed for Stage 1 guidance. Invalid regions are zeroed out in the warped reference image, and the resulting masked warp together with the validity mask are fed to the Stage 1 deblurring network. This results in more reliable Stage 1 guidance by removing unstable correspondences near occlusions and motion boundaries. The same bidirectional flow fields are used for both validity checking and masked warping.

## 1.3 Frozen 3DGS Backbone and Input-View Re-Rendering

We adopt the oficial NoPoSplat [13] checkpoint pretrained on RealEstate10K with two-view inputs at 256×256, and keep it frozen without further fine-tuning. During Stage 2, the intermediate restorations and their camera intrinsics are fed into the backbone to construct a canonical-space 3D Gaussian representation. We then re-render this representation to the two input viewpoints to obtain the global 3D guidance images $R _ { 1 }$ and $R _ { 2 }$ for final restoration. These rendered RGB images are used directly as Stage 2 guidance. Concretely, the first input view defines the canonical frame, and the second guidance image is rendered at the relative pose inferred by the backbone under the same canonical-space formulation. No ground-truth input-view poses are used. The same backbone is then applied to the restored views and their intrinsics to construct the output 3D representation used for novel view synthesis evaluation.

## 1.4 Training Details

Only the two deblurring networks, $\mathcal { D } _ { \theta } ^ { 2 D }$ and $\mathcal { D } _ { \psi } ^ { 3 D }$ , are optimized during training. The stabilizer $S _ { \phi }$ , RAFT, and the pose-free 3DGS backbone $h _ { \eta }$ remain frozen throughout, with no gradients propagated through them. Both trainable networks follow a NAFNet-style encoder–decoder architecture with base width 64. We use an encoder block configuration [1, 1, 1, 28], a single middle block, and a decoder configuration [1, 1, 1, 1]. Stage 1 takes a 7-channel input formed by concatenating the 3-channel blurry base view, the 3-channel masked warp, and the 1-channel validity mask, while Stage 2 takes a 6-channel input formed by concatenating the 3-channel blurry base view and the 3-channel re-render guidance. Both networks output a restored 3-channel RGB image.

Training is performed in a stage-wise manner. In Stage 1, the frozen stabilizer and RAFT module are used to construct the masked warps and validity masks, and $\mathcal { D } _ { \theta } ^ { 2 D }$ is trained to predict the intermediate restorations $( { \tilde { S } } _ { 1 } , { \tilde { S } } _ { 2 } )$ from the blurry inputs and OCGM guidance. After Stage 1 training, $\mathcal { D } _ { \theta } ^ { 2 D }$ is frozen. In Stage 2, the frozen Stage 1 network first produces $( \tilde { S } _ { 1 } , \tilde { S } _ { 2 } )$ , which are passed to the frozen pose-free 3DGS backbone to obtain the re-render guidance $( R _ { 1 } , R _ { 2 } )$ Using these re-rendered images, $\mathcal { D } _ { \psi } ^ { 3 D }$ is trained to predict the final restorations $( \hat { S } _ { 1 } , \hat { S } _ { 2 } )$

For each stage, we use AdamW with learning rate $1 \times 1 0 ^ { - 3 }$ , weight decay $1 \times 1 0 ^ { - 3 }$ , and $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 )$ . Each stage is trained for 400k iterations with cosine annealing to $\eta _ { \mathrm { m i n } } = 1 \times 1 0 ^ { - 7 }$ . We use a batch size of 32 per GPU on two NVIDIA A100 80GB GPUs, with four dataloader workers per GPU. The training objective for both stages combines an $L _ { 1 }$ reconstruction loss and a VGG19-based perceptual loss [7, 10]:

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { 1 } \mathcal { L } _ { \ell _ { 1 } } + \lambda _ { p } \mathcal { L } _ { \mathrm { p e r c } } , } \end{array}\tag{1}
$$

where $\lambda _ { 1 } = 0 . 9$ and $\lambda _ { p } = 0 . 1$ . No style loss is used.

Table 1: Per-scene evaluation index mappings for the real-world and synthetic Deblur-NeRF scenes. Each instance is defined as $\{ I _ { b l u r } \} \to I _ { t g t }$ , where $\left\{ { { I } _ { b l u r } } \right\}$ denotes the two blurry input views and $I _ { t g t }$ denotes the held-out target view for novel-view synthesis.
<table><tr><td>Scene</td><td colspan="2">Inst. 1</td><td>Inst. 2</td><td>Inst. 3</td><td>Inst. 4</td><td colspan="2">Inst. 5</td><td colspan="2">Inst. 6</td></tr><tr><td colspan="9">Real-world Scenes</td></tr><tr><td>BlurBall</td><td></td><td>{1, 2} → 0</td><td>{6, 8} → 7</td><td>{13, 15} → 14</td><td>{19, 20} → 21</td><td></td><td></td><td></td><td></td></tr><tr><td>BlurBasket</td><td></td><td>{1, 2} → 0</td><td>{10, 11} → 7</td><td>{18, 19} → 14</td><td>{22, 23} → 21</td><td>{27, 29} → 28</td><td></td><td>{38, 39} → 35</td><td>{41, 43} → 42</td></tr><tr><td>BlurBuick</td><td></td><td>{1, 2} → 0</td><td>{10, 11} → 7</td><td>{15, 18} → 14</td><td>{4, 6} → 21</td><td>{27, 33} → 28</td><td></td><td>{36, 38} → 35</td><td>{39, 43} → 42</td></tr><tr><td>BlurCoffee</td><td></td><td>{3, 8} → 0</td><td>{10, 11} → 6</td><td>{1, 17} → 12</td><td>{5, 19} → 18</td><td>{25, 26} → 24</td><td></td><td></td><td></td></tr><tr><td>BlurDecoration</td><td></td><td>{1, 2} → 0</td><td>{4, 9} → 6</td><td>{3, 17} → 12</td><td>{19, 33} → 18</td><td>{16, 25} → 24</td><td></td><td>{14, 29} → 30</td><td>{31, 40} → 36</td></tr><tr><td>BlurGirl</td><td></td><td>{1, 2} → 0</td><td>{6, 8} → 7</td><td>{9, 11} → 14</td><td>{12, 34} → 21</td><td>{18, 29} → 28</td><td></td><td>{30, 36} → 35</td><td></td></tr><tr><td>BlurHeron</td><td></td><td>{1, 11} → 0</td><td>{7, 15} → 8</td><td>{15, 27} → 16</td><td>{23, 25} → 24</td><td>{19, 33} → 32</td><td></td><td></td><td></td></tr><tr><td>BlurParterre</td><td></td><td>{2, 9} → 0</td><td>{4, 7} → 6</td><td>{11, 13} → 12</td><td>{19, 20} → 18</td><td>{23, 27} → 24</td><td></td><td>{1, 21} → 30</td><td></td></tr><tr><td>BlurPuppet</td><td></td><td>{1, 2} → 0</td><td>{10, 11} → 6</td><td>{7, 8} → 12</td><td>{15, 16} → 18</td><td>{19, 21} → 24</td><td></td><td>{27, 29} → 30</td><td>{23, 37} → 36</td></tr><tr><td>BlurStair</td><td></td><td>{1, 2} → 0</td><td>{7, 9} → 6</td><td>{8, 17} → 12</td><td>{21, 23} → 18</td><td>{16, 26} → 24</td><td></td><td>{22, 33} → 30</td><td></td></tr><tr><td colspan="10">Synthetic Scenes</td></tr><tr><td>All Scenes</td><td></td><td>{1, 2} → 0</td><td>{7, 9} → 8</td><td>{15, 17} → 16</td><td>{23, 25} → 24</td><td>{31, 33} → 32</td><td></td><td></td><td></td></tr></table>

## 1.5 Baseline Adaptation Details

All baselines are evaluated under the same fixed two-view protocol using the perscene input–target index mappings listed in Tab. 1. Methods originally designed for diferent input regimes are adapted to this protocol while following their oficial implementations and recommended hyperparameter settings as closely as possible. For methods that require camera poses, we use the benchmark poses provided with Deblur-NeRF [5].

For SE-GS [14], we optimize each scene from the same two blurry source views defined by the protocol. For GAURA [4], we follow the oficial feed-forward inference pipeline in the two-view setting. For CoherentGS [12], we retain its oficial alternating optimization framework while restricting the observations to the same two-view protocol.

For the control baselines DAVANet [15] and Difix3D+ [11], the restored outputs are passed to the same frozen pose-free 3DGS backbone [13] used in our method. Thus, the “Pose-Free” and “Generalizable” labels reported in the main paper refer to the complete pipelines formed by each restoration model together with the shared frozen 3DGS backbone, rather than to the restoration models alone.

## 2 Qualitative Analysis of OCGM Thresholds

Following the OCGM formulation in Eqs. (8)–(10) of the main paper, we qualitatively examine how the minimum tolerance τ and the motion-dependent scale factor α afect the validity mask $M _ { b  r }$ and the masked cross-view warp $W _ { b  r }$ used as Stage 1 guidance.

For each example, $C _ { b }$ denotes the stabilized base view, while $C _ { r }$ denotes the stabilized reference view. The unmasked warp $\mathcal { W } ( C _ { r } , F _ { b  r } )$ transfers content from $C _ { r }$ into the coordinates of $C _ { b }$ . OCGM retains only the regions considered geometrically reliable and provides the resulting masked warp, together with its validity mask, to the Stage 1 restoration network. The stabilizer output is therefore used only as an alignment-friendly observation: Stage 1 combines the original blurry base view with reliable complementary information from the reference view to produce a better restoration than independent stabilization alone.

![](images/c27ab17f57a3694f72c96de8ea19a4a8e819889efd6d1f1027ce01fa45b086e0.jpg)  
Fig. 1: Qualitative efects of the OCGM thresholds. In each row, the stabilized base view $C _ { b }$ provides the current-view context, while the unmasked warp $\mathcal { W } ( C _ { r } , F _ { b  r } )$ transfers complementary content from the stabilized reference view $C _ { r }$ into the baseview coordinates. The first row varies $\tau \in \{ 0 . 5 , 1 . 0 , 2 . 0 \}$ with $\alpha = 0 . 0 1$ , whereas the second varies $\alpha \in \{ 0 , 0 . 0 1 , 0 . 0 5 \}$ with $\tau = 1 . 0$ . Black regions in the masked guidance indicate rejected correspondences. The reported coverage denotes the retained fraction and is shown only to illustrate the trade-of between rejecting miswarped content and preserving useful cross-view information.

For each threshold sweep, we keep $( C _ { b } , C _ { r } )$ and their bidirectional RAFT flows fixed while varying one parameter at a time. Thus, the unmasked warp remains identical across the compared settings; only the regions retained as Stage 1 guidance change.

As shown in Fig. 1, the thresholds control the trade-of between rejecting unreliable warps and preserving valid cross-view information. In the BlurGirl example, the unmasked warp incorrectly transfers the highlighted plant tip. The permissive setting $\tau = 2 . 0$ retains this miswarped structure, whereas $\tau = 1 . 0$ rejects it while preserving most of the surrounding valid guidance. The more conservative $\tau = 0 . 5$ also removes the error but discards additional correctly warped content, as reflected by its lower coverage.

The α sweep on BlurBasket shows a similar pattern in a large-displacement region. The setting $\alpha = 0 . 0 1$ selectively suppresses the distorted chair geometry while retaining much of the surrounding useful reference content. In contrast, $\alpha = 0$ rejects additional valid chair regions, whereas $\alpha = 0 . 0 5$ admits distorted boundaries that should be excluded. Together, these examples illustrate how overly conservative thresholds reduce the amount of usable guidance, while overly permissive thresholds propagate inaccurate reference content.

We use the fixed default $\tau = 1 . 0$ and $\alpha = 0 . 0 1$ for all training and evaluation scenes. These values are not selected as per-scene optima; rather, they provide a practical scene-independent balance between filtering unreliable warps and retaining suficient guidance across diverse scenes and motion patterns. The alternative settings are shown only to illustrate OCGM’s masking behavior and do not represent separately trained models.

## 3 Additional Component Diagnostics

Table 2 and Fig. 2 provide complementary quantitative and qualitative diagnostics beyond the ablation in the main paper.

Efect of alignment preconditioning. We estimate RAFT flows and construct OCGM guidance directly from the original blurry views while reusing the same Stage 1 and Stage 2 checkpoints as the full model. This inference-time diagnostic removes the stabilizer only from the flow and guidance construction path, without retraining either restoration network. Compared with the full cascade, PSNR and SSIM decrease by 0.33 dB and 0.020, respectively. These results support the role of the stabilizer in producing alignment-friendly observations for reliable correspondence guidance, rather than serving as the final restoration module.

Efect of global 3D guidance without Stage 1. We additionally evaluate an inference time 3D-only diagnostic that bypasses the stabilizer, RAFT, OCGM, and the entire Stage 1 restoration process. The original blurry input pair is first passed to the frozen NoPoSplat backbone to produce input-view re-render guidance. The blurry inputs and corresponding re-renders are then processed by the existing Stage 2 checkpoint, and the resulting restorations are used for final reconstruction with the frozen NoPoSplat backbone.

This configuration obtains 20.26 dB PSNR and 0.627 SSIM, substantially underperforming both Stage 1 and the full cascade. The result indicates that global 3D re-render guidance derived directly from blurry inputs cannot replace the locally reliable observations established by Stage 1. Because the Stage 2 checkpoint is reused without retraining, this experiment should be interpreted as an inference-time diagnostic rather than an optimized 3D-only architecture.

Qualitative interpretation. Figure 2 further illustrates how each diagnostic configuration afects the final novel-view reconstruction. Independent stabilization improves each view separately but does not explicitly enforce cross-view consistency, leaving residual discrepancies that are dificult for the frozen pose-free 3DGS backbone to aggregate coherently. Direct RAFT estimates correspondences directly from the original blurry inputs, producing less reliable OCGM guidance. This degrades the Stage 1 restorations and consequently the 3D guidance available to Stage 2.

Stage 1 combines stabilization and OCGM to establish more reliable local correspondences and therefore yields a more coherent reconstruction. However, it lacks the global 3D feedback provided by Stage 2, leaving residual inconsistencies that cannot be resolved through local 2D warping alone. Conversely, 3D-only constructs global guidance directly from the blurry inputs, before locally reliable observations have been established. The full cascade addresses both limitations by first improving local cross-view reliability and then refining the restored views using globally aggregated 3D guidance.

Table 2: Extended component diagnostics on real-world Deblur-NeRF scenes. The first four rows reproduce the progressive ablation from the main paper; the last two are inference-time diagnostics using the trained checkpoints without retraining.
<table><tr><td>Setting</td><td>Configuration</td><td>PSNR ↑ SSIM ↑</td><td></td></tr><tr><td colspan="4">Progressive cascade</td></tr><tr><td></td><td>Blurry inputs No restoration or cross-view guidance</td><td>20.07</td><td>0.612</td></tr><tr><td>Stabilizer</td><td>Independent per-view stabilization</td><td>20.30</td><td>0.655</td></tr><tr><td>Stage 1</td><td>Stabilizer and OCGM-based local 2D guidance</td><td>20.88</td><td>0.684</td></tr><tr><td>Full cascade</td><td>Complete CasDeblurGS pipeline</td><td>21.13</td><td>0.700</td></tr><tr><td colspan="4">Additional diagnostics</td></tr><tr><td></td><td>Direct RAFT RAFT and OCGM applied directly to the original</td><td>20.80</td><td>0.680</td></tr><tr><td>3D-only</td><td>blurry views Stage 1 bypassed; Stage 2 uses blurry inputs and 3D re-renders</td><td>20.26</td><td>0.627</td></tr></table>

![](images/3d8d5ba615bf052ea40fb1649b5a51a321eeffaf62ef84d91d9199e4dbde3ba7.jpg)  
Fig. 2: Qualitative novel-view synthesis results for component diagnostics on a synthetic Deblur-NeRF scene. The Inputs column shows the two blurry views; for each reconstruction setting, the lower row enlarges the red-boxed region.

## 4 Sparse Camera Reprojection Analysis

We further describe the sparse reprojection analysis reported in the main paper. We compare the intermediate restorations produced by Stage 1 with the final restorations produced by Stage 2 using the same fixed real-world two-view pairs.

For each restored image pair, we extract OpenCV SIFT features with at most 4,000 keypoints and perform brute-force matching using the $\ell _ { 2 }$ distance. We apply a Lowe ratio threshold of 0.75 and retain only bidirectionally mutual matches. The resulting correspondences are triangulated using cv2.triangulatePoints and the benchmark camera projection matrices.

A triangulated point is considered valid if its homogeneous coordinates are finite, its homogeneous denominator is nonzero, and it has positive depth in both camera frames. For each valid point, we compute the symmetric reprojection error as the average pixel $\ell _ { 2 }$ error after reprojecting the point into the two input views.

For each evaluation pair, we measure the number of valid triangulated points, the median symmetric reprojection error, the number of points with reprojection error below one pixel, and the corresponding inlier ratio. The values reported in the main paper are obtained by averaging these per-pair measurements over the 60 real-world evaluation pairs. The reported inlier ratio is therefore the mean of the per-pair ratios, rather than the mean inlier count divided by the mean number of valid points. Benchmark camera poses are used only for this post-hoc evaluation and are never provided as inputs to CasDeblurGS.

## 5 Runtime Analysis

Although CasDeblurGS employs multiple frozen modules, all inference stages are feed-forward and require no per-scene optimization. Table 3 reports average runtime, PSNR, and SSIM across both the real-world and synthetic Deblur-NeRF subsets. CasDeblurGS improves PSNR by 4.48 dB over GAURA, the fastest baseline, while running 7.5% faster than Difix3D+ and 31.8% faster than CoherentGS, the closest competing methods in reconstruction quality. These results demonstrate a favorable trade-of between reconstruction quality and computational cost.

Table 3: Average runtime and reconstruction quality across both the real-world and synthetic Deblur-NeRF subsets. DAVANet and Difix3D+ are combined with the same frozen NoPoSplat backbone used in CasDeblurGS. Best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td>Method</td><td>Runtime (s) ↓</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>SE-GS [14]</td><td>28.48</td><td>17.33</td><td>0.479</td></tr><tr><td>GAURA [4]</td><td>25.24</td><td>17.79</td><td>0.516</td></tr><tr><td>CoherentGS [12]</td><td>96.14</td><td>20.51</td><td>0.674</td></tr><tr><td>DAVANet [15] + NoPoSplat</td><td>59.37</td><td>19.48</td><td>0.599</td></tr><tr><td>Difix3D+ [11] + NoPoSplat</td><td>70.85</td><td>20.62</td><td>0.642</td></tr><tr><td>CasDeblurGS (Ours)</td><td>65.52</td><td>22.27</td><td>0.748</td></tr></table>

## 6 Additional Experimental Results

In this section, we provide per-scene quantitative evaluations and additional qualitative comparisons on Deblur-NeRF to further support the findings in the main paper.

Per-scene quantitative results. Tables 4 and 5 present per-scene quantitative comparisons on the synthetic and real-world Deblur-NeRF datasets, respectively. Consistent with the average gains reported in the main paper, our method achieves the best average performance and remains competitive across individual scenes. These results show that the proposed cascaded 2D-to-3D guidance performs consistently across diverse scene structures and blur patterns.

Additional qualitative results. Figures 3 and 4 provide additional qualitative comparisons that further support the quantitative results. Under the challenging two-view setting with severe blur, existing methods often sufer from structural collapse or ghosting artifacts. For example, in the BlurFactory scene, our framework reconstructs the staircase with more coherent geometry and finer texture details, whereas prior 3DGS-based methods struggle to preserve its topology. In the BlurCofee scene, CoherentGS exhibits noticeable color artifacts, while Difix3D+ produces overly blurred results; by contrast, our method recovers clearer text on the notice. Overall, by combining local 2D correspondence guidance with global 3D re-render guidance, our approach improves multi-view consistency and yields higher-quality novel-view synthesis, especially in geometrically challenging regions.

## 7 Supplementary Video

The supplementary video provides an animated overview of CasDeblurGS, illustrating how local 2D correspondence guidance is progressively complemented by global 3D re-render guidance. It also presents novel-view synthesis results from our method on nine scenes spanning the real-world and synthetic Deblur-NeRF subsets, enabling temporal inspection of rendering sharpness and structural consistency beyond the still-image results in the paper.

Table 4: Quantitative results of novel view synthesis on synthetic scenes of the Deblur-NeRF dataset, reported for each individual scene. The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td>Method</td><td>Scene</td><td>PSNR ↑ SSIM ↑ LPIPS ↓</td><td></td><td>Method</td><td>Scene</td><td></td><td>PSNR ↑ SSIM ↑ LPIPS ↓</td><td></td></tr><tr><td rowspan="6">SE-GS</td><td>BlurCozy2room</td><td>20.65</td><td>0.686 0.209</td><td rowspan="6">DAVANet</td><td>BlurCozy2room BlurFactory</td><td>21.83</td><td>0.770</td><td>0.185</td></tr><tr><td>BlurFactory</td><td>17.36 0.420</td><td>0.487</td><td></td><td>16.27</td><td>0.385</td><td>0.488</td></tr><tr><td>BlurPool</td><td>22.87 0.660</td><td>0.265</td><td>BlurPool</td><td>21.53</td><td>0.607</td><td>0.270</td></tr><tr><td>BlurTanabata</td><td>15.72</td><td>0.497 0.377</td><td>BlurTanabata</td><td>18.93</td><td>0.673</td><td>0.268</td></tr><tr><td>BlurWine</td><td>17.19</td><td>0.536 0.353</td><td>BlurWine</td><td>19.50</td><td>0.691</td><td>0.247</td></tr><tr><td>Average</td><td>18.76 0.560</td><td>0.338</td><td>Average</td><td>19.61</td><td>0.625</td><td>0.292</td></tr><tr><td rowspan="6">GAURA</td><td>BlurCozy2room</td><td>17.85</td><td>0.604 0.345</td><td rowspan="6">Difix3D+</td><td>BlurCozy2room</td><td>23.60</td><td>0.801</td><td>0.138</td></tr><tr><td>BlurFactory</td><td>16.96 0.409</td><td>0.504</td><td>BlurFactory BlurPool</td><td>18.12</td><td>0.479</td><td>0.457</td></tr><tr><td>BlurPool</td><td>23.04 0.625</td><td>0.331</td><td></td><td>26.33</td><td>0.742</td><td>0.168</td></tr><tr><td>BlurTanabata</td><td>16.08 0.526</td><td>0.431</td><td>BlurTanabata</td><td>19.10</td><td>0.682</td><td>0.277</td></tr><tr><td>BlurWine</td><td>15.63</td><td>0.458 0.459</td><td>BlurWine</td><td>19.35</td><td>0.663</td><td>0.273</td></tr><tr><td>Average</td><td>17.91 0.524</td><td>0.414</td><td>Average</td><td>21.30</td><td>0.673</td><td>0.262</td></tr><tr><td rowspan="6">CoherentGS</td><td>BlurCozy2room</td><td>24.91</td><td>0.828 0.160</td><td rowspan="6">Ours</td><td>BlurCozy2room BlurFactory</td><td>25.46</td><td>0.866</td><td>0.097</td></tr><tr><td>BlurFactory</td><td>18.55 0.530</td><td>0.385</td><td></td><td>23.73</td><td>0.818</td><td>0.203</td></tr><tr><td>BlurPool</td><td>24.33 0.705</td><td>0.227</td><td>BlurPool</td><td>27.14</td><td>0.775</td><td>0.142</td></tr><tr><td>BlurTanabata</td><td>18.56 0.674</td><td>0.276</td><td>BlurTanabata</td><td>19.99</td><td>0.748</td><td>0.202</td></tr><tr><td>BlurWine</td><td>19.21</td><td>0.696 0.247</td><td>BlurWine</td><td>20.75</td><td>0.771</td><td>0.182</td></tr><tr><td>Average</td><td>21.11 0.687</td><td>0.259</td><td>Average</td><td>23.41</td><td>0.796</td><td>0.165</td></tr></table>

Table 5: Quantitative results of novel view synthesis on real-world scenes of the Deblur-NeRF dataset, reported for each individual scene. The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td>Method</td><td>Scene</td><td></td><td>PSNR ↑ SSIM ↑ LPIPS ↓</td><td></td><td>Method Scene</td><td></td><td>PSNR ↑ SSIM ↑ LPIPS ↓</td><td></td></tr><tr><td rowspan="10">SE-GS</td><td>BlurBall</td><td>18.80</td><td>0.518</td><td>0.404 0.479</td><td>BlurBall</td><td>20.74</td><td>0.598</td><td>0.342</td></tr><tr><td>BlurBasket</td><td>14.91</td><td>0.358</td><td></td><td>BlurBasket</td><td>19.54</td><td>0.578</td><td>0.378</td></tr><tr><td>BlurBuick</td><td>13.93</td><td>0.376 0.467</td><td></td><td>BlurBuick</td><td>18.46</td><td>0.602</td><td>0.340</td></tr><tr><td>BlurCoffee</td><td>18.74</td><td>0.677 0.351</td><td></td><td>BlurCoffee</td><td>23.11</td><td>0.779</td><td>0.280</td></tr><tr><td>BlurDecoration</td><td>14.10</td><td>0.281 0.495</td><td></td><td>BlurDecoration</td><td>16.88</td><td>0.457</td><td>0.411</td></tr><tr><td>BlurGirl</td><td>15.32</td><td>0.530 0.359</td><td></td><td>DAVANet BlurGirl</td><td>19.35</td><td>0.713</td><td>0.291</td></tr><tr><td>BlurHeron</td><td>15.97</td><td>0.334 0.413</td><td></td><td>BlurHeron</td><td>18.85</td><td>0.487</td><td>0.390</td></tr><tr><td>BlurParterre</td><td>16.58</td><td>0.313</td><td>0.454 0.505</td><td>BlurParterre</td><td>19.30</td><td>0.468</td><td>0.406</td></tr><tr><td>BlurPuppet</td><td>14.18</td><td>0.284</td><td></td><td>BlurPuppet</td><td>18.76</td><td>0.524</td><td>0.343</td></tr><tr><td>BlurStair</td><td>16.45</td><td>0.304 0.499</td><td></td><td>BlurStair Average</td><td>18.42</td><td>0.527</td><td>0.415</td></tr><tr><td rowspan="10"></td><td>Average</td><td>15.90</td><td>0.398</td><td>0.443 0.433</td><td>BlurBall</td><td>19.34 21.35</td><td>0.573 0.622</td><td>0.360 0.309</td></tr><tr><td>BlurBall BlurBasket</td><td>20.11 13.47</td><td>0.569</td><td></td><td>BlurBasket</td><td>19.98</td><td>0.622</td><td>0.326</td></tr><tr><td>BlurBuick</td><td>14.30</td><td>0.406 0.499 0.383 0.506</td><td></td><td>BlurBuick</td><td>18.45</td><td>0.621</td><td>0.293</td></tr><tr><td>BlurCoffee</td><td>22.53</td><td>0.800 0.310</td><td></td><td>BlurCoffee</td><td>23.74</td><td>0.815</td><td>0.203</td></tr><tr><td>BlurDecoration</td><td>16.21</td><td>0.433 0.490</td><td></td><td>BlurDecoration</td><td>17.69</td><td>0.497</td><td>0.369</td></tr><tr><td>BlurGirl</td><td>17.23</td><td>0.632</td><td></td><td>Difix3D+ BlurGirl</td><td>19.52</td><td>0.768</td><td>0.227</td></tr><tr><td>BlurHeron</td><td>17.57</td><td>0.446</td><td>0.381</td><td>BlurHeron</td><td>18.47</td><td>0.493</td><td>0.373</td></tr><tr><td>BlurParterre</td><td>18.86</td><td>0.451</td><td>0.451 0.461</td><td>BlurParterre</td><td>19.73</td><td>0.508</td><td>0.366</td></tr><tr><td>BlurPuppet</td><td>17.35</td><td>0.467</td><td>0.454</td><td>BlurPuppet</td><td>19.25</td><td>0.567</td><td>0.299</td></tr><tr><td>BlurStair</td><td>19.10</td><td>0.492</td><td>0.456</td><td>BlurStair</td><td>21.25</td><td>0.595</td><td>0.346</td></tr><tr><td>Average</td><td>17.67</td><td>0.508</td><td>0.444</td><td>Average</td><td></td><td>19.94 0.611</td><td></td><td>0.311</td></tr><tr><td rowspan="10"></td><td>BlurBall</td><td>21.75</td><td>0.672</td><td>0.303</td><td>BlurBall</td><td>22.87</td><td>0.730</td><td>0.228</td></tr><tr><td>BlurBasket</td><td>20.04</td><td>0.668 0.281</td><td></td><td>BlurBasket</td><td>21.00</td><td>0.703</td><td>0.234</td></tr><tr><td>BlurBuick</td><td>16.61</td><td>0.546</td><td>0.357</td><td>BlurBuick</td><td>20.17</td><td>0.709</td><td>0.217</td></tr><tr><td>BlurCoffee</td><td>24.78</td><td>0.826 0.247</td><td></td><td>BlurCoffee</td><td>25.21</td><td>0.859</td><td>0.156</td></tr><tr><td>BlurDecoration</td><td>16.69</td><td>0.517</td><td>0.379</td><td>BlurDecoration</td><td>17.97</td><td>0.545</td><td>0.309</td></tr><tr><td>CoherentGS BlurGirl</td><td>19.04</td><td>0.734</td><td>0.256</td><td>BlurGirl</td><td>21.56</td><td>0.843</td><td>0.158</td></tr><tr><td>BlurHeron</td><td>18.88</td><td>0.599</td><td>0.321</td><td>BlurHeron</td><td>20.42</td><td>0.651</td><td>0.260</td></tr><tr><td>BlurParterre</td><td>19.72</td><td>0.618</td><td>0.306</td><td>BlurParterre</td><td>20.41</td><td>0.586</td><td>0.281</td></tr><tr><td>BlurPuppet</td><td>19.64</td><td>0.666</td><td>0.267</td><td>BlurPuppet</td><td>19.87</td><td>0.633</td><td>0.232</td></tr><tr><td>BlurStair</td><td>21.88</td><td>0.755</td><td>0.204</td><td>BlurStair</td><td>21.82</td><td>0.736</td><td>0.209</td></tr><tr><td>Average</td><td>19.90</td><td>0.660</td><td>0.292</td><td>Average</td><td>21.13</td><td>0.700</td><td>0.228</td></tr></table>

Inputs  
SE-GS  
GAURA  
CoherentGS  
DAVANet  
Difix3D+  
Ours  
GT  
![](images/936ce9bba9a9fbf79f9a8493a7fac9b8d3feb7adc59101e83fc3ce1f46d41e2f.jpg)  
Fig. 3: Additional qualitative results of novel view synthesis on synthetic scenes of the Deblur-NeRF dataset. The Inputs column shows the two blurry views; for each method, the lower row enlarges the red-boxed region.

Inputs  
SE-GS  
GAURA  
CoherentGS  
DAVANet  
Difix3D+  
Ours  
GT  
![](images/92e2f29239cb47434acd8b041fb727b04e9feea16298b616ae49b58e42cfa80b.jpg)  
Fig. 4: Additional qualitative results of novel view synthesis on real-world scenes of the Deblur-NeRF dataset. The Inputs column shows the two blurry views; for each method, the lower row enlarges the red-boxed region.

## References

1. Charatan, D., Li, S.L., Tagliasacchi, A., Sitzmann, V.: pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 19457– 19467 (2024)

2. Chen, L., Chu, X., Zhang, X., Sun, J.: Simple baselines for image restoration. In: European conference on computer vision. pp. 17–33. Springer (2022)

3. Choi, H., Yang, H., Han, J., Cho, S.: Exploiting deblurring networks for radiance fields. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 6012–6021 (2025)

4. Gupta, V., Girish, R.S.V., Mukund Varma, T., Tewari, A., Mitra, K.: Gaura: Generalizable approach for unified restoration and rendering of arbitrary views. In: European Conference on Computer Vision. pp. 249–266. Springer (2024)

5. Ma, L., Li, X., Liao, J., Zhang, Q., Wang, X., Wang, J., Sander, P.V.: Deblurnerf: Neural radiance fields from blurry images. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12861–12870 (2022)

6. Nah, S., Kim, T.H., Lee, K.M.: Deep multi-scale convolutional neural network for dynamic scene deblurring. In: CVPR (July 2017)

7. Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 (2014)

8. Teed, Z., Deng, J.: Raft: Recurrent all-pairs field transforms for optical flow. In: European conference on computer vision. pp. 402–419. Springer (2020)

9. TorchVision maintainers and contributors: TorchVision: PyTorch’s computer vision library. https://github.com/pytorch/vision (2016)

10. Wang, X., Xie, L., Dong, C., Shan, Y.: Real-esrgan: Training real-world blind superresolution with pure synthetic data. In: International Conference on Computer Vision Workshops (ICCVW) (2021)

11. Wu, J.Z., Zhang, Y., Turki, H., Ren, X., Gao, J., Shou, M.Z., Fidler, S., Gojcic, Z., Ling, H.: Difix3d+: Improving 3d reconstructions with single-step difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 26024–26035 (2025)

12. Xu, Z., Feng, C., Li, Y., Zhao, J., Yang, J., Yu, W., Yuan, L., Tian, Y.: Breaking the vicious cycle: Coherent 3d gaussian splatting from sparse and motion-blurred views. arXiv preprint arXiv:2512.10369 (2025)

13. Ye, B., Liu, S., Xu, H., Li, X., Pollefeys, M., Yang, M.H., Peng, S.: No pose, no problem: Surprisingly simple 3D gaussian splats from sparse unposed images. In: International Conference on Learning Representations (2025)

14. Zhao, C., Wang, X., Zhang, T., Javed, S., Salzmann, M.: Self-ensembling gaussian splatting for few-shot novel view synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4940–4950 (2025)

15. Zhou, S., Zhang, J., Zuo, W., Xie, H., Pan, J., Ren, J.S.: Davanet: Stereo deblurring with view aggregation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10996–11005 (2019)