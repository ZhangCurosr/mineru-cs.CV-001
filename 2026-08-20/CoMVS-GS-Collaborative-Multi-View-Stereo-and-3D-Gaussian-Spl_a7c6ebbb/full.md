# CoMVS-GS: Collaborative Multi-View Stereo and 3D Gaussian Splatting for Surface Reconstruction

Shihan Chen<sup>1</sup>, Junjing Zhang<sup>1</sup>, Qingsong Yan<sup>1</sup>, Haibing Liu<sup>1</sup>, Haofan Ren<sup>2</sup>, and Fei Deng<sup>1</sup> <sup>1</sup>Wuhan University <sup>2</sup>Hangzhou Dianzi University

## Abstract

3D Gaussian Splatting enables eficient novel view synthesis, but accurate mesh reconstruction remains dificult in weakly observed and occluded regions, where Gaussian primitives may grow into unstable or geometrically inconsistent structures. We propose CoMVS-GS, a general surface reconstruction framework that combines Multi-View Stereo with Gaussian splatting. CoMVS-GS initializes Gaussian primitives from dense multi-view stereo points with pre-flattened scales and normal-aligned orientations, providing stronger geometric priors than sparse structure-from-motion initialization and reducing ambiguity during early optimization. It further introduces PatchMatch-3DGS Mutual Supervision, where Gaussian-rendered depths and normals initialize PatchMatch refinement, and refined PatchMatch depths supervise Gaussian optimization to improve weakly constrained geometry. For surface extraction, CoMVS-GS replaces truncated signed distance field voxel fusion with a Delaunay graph-cut meshing pipeline, reducing sensitivity to voxel resolution while preserving visibility-consistent surface evidence. Experiments on DTU, GauU-Scene V2, and MatrixCity show that CoMVS-GS remains competitive on object-level reconstruction and improves geometric accuracy and mesh compactness in outdoor scenes while maintaining high rendering quality.

Keywords: 3D Gaussian Splatting; Surface Reconstruction; Multi-View Stereo; PatchMatch;   
Delaunay Graph-Cut Meshing.

## 1 Introduction

Reconstructing accurate 3D surfaces from multi-view images is a fundamental problem in computer vision. Classical Structure from Motion (SfM) and Multi-View Stereo (MVS) pipelines [25, 26] recover geometry by matching image evidence across views, and can produce metrically reliable point clouds and meshes when suficient texture and overlap are available. In parallel, neural rendering methods have shifted the field toward optimizing scene representations through diferentiable rendering. Among them, 3D Gaussian Splatting (3DGS) [17] has become particularly attractive because it represents scenes with explicit Gaussian primitives and achieves fast training and real-time rendering. However, the original 3DGS formulation is optimized mainly for photometric view synthesis rather than surface reconstruction. Its primitives may therefore reproduce training views well while forming noisy, floating, or geometrically inconsistent structures in 3D.

Recent Gaussian-based surface reconstruction methods attempt to narrow this gap by making Gaussian primitives more surface-aware. SuGaR [11], 2DGS [13], Gaussian Surfels [7], GOF [34], Trim3DGS [8], and RaDe-GS [35] introduce diferent geometric priors or depth rasterization strategies for extracting surfaces from Gaussian representations. PGSR [4] further improves surface quality by flattening Gaussians into planar primitives and imposing multi-view photometric and reprojection constraints. These advances show that 3DGS can be adapted from a rendering representation into a geometry reconstruction framework. Nevertheless, robust mesh reconstruction remains dificult in weakly observed, heavily occluded, and geometrically under-constrained regions, especially in outdoor scenes with complex visibility, as illustrated in 3Fig. 1.

![](images/4f861539fd82f5267e5c14763d30a62efd0c7d88176f143e72d680bfb4f4be89.jpg)  
Figure 1: Qualitative comparison between PGSR and CoMVS-GS in rendered images, depth maps, and normal maps. CoMVS-GS produces flatter geometry in weakly constrained regions while preserving fine scene details.

Most 3DGS-based methods initialize Gaussians from sparse SfM point clouds [17, 11, 13, 7, 4]. While suficient for radiance-field optimization, these sparse points leave large spatial voids in regions with few reliable feature tracks, forcing geometry and surface orientation to be recovered mainly through later optimization. Classical MVS can instead generate dense point clouds with surface normal estimates before Gaussian optimization begins [26, 3], providing a stronger geometric prior for sparsely initialized and weakly observed regions.

Geometry supervision during optimization is also often weak or noisy. PGSR-style multi-view regularization relies on photometric and reprojection consistency between a reference view and selected source views [4], but distance- or direction-based neighbors may have limited actual overlap and inconsistent visibility near occlusion boundaries under irregular trajectories. Gaussian optimization therefore benefits most directly from explicit MVS depth refinement, with overlap-aware source-view filtering used as an auxiliary strategy to make multi-view regularization more reliable.

Mesh extraction introduces another source of instability. Many Gaussian reconstruction pipelines render depth maps from optimized Gaussians, fuse them into a TSDF volume [6], and extract a surface with Marching Cubes [21, 13, 4, 34]. This strategy is sensitive to voxel resolution: coarse grids over-smooth fine structures, whereas fine grids increase memory consumption and can expose holes or fragmented surfaces when rendered depths are inconsistent. The issue is especially prominent in outdoor reconstruction, although the underlying discretization sensitivity is not limited to large-scale scenes.

To address these challenges, we propose CoMVS-GS, a general MVS-enhanced Gaussian surface reconstruction framework for both object-level and outdoor scenes. Our key idea is to use MVS not as a separate final reconstruction pipeline, but as a source of dense initialization, depthlevel supervision, and visibility-consistent meshing. Specifically, MVS dense points initialize Gaussian primitives; PatchMatch-3DGS Mutual Supervision uses rendered depths and normals to initialize PatchMatch and refined PatchMatch depths to supervise 3DGS; and the final surface is recovered with depth fusion, Delaunay graph-cut reconstruction, and mesh refinement rather than TSDF voxel fusion [14, 28]. Inspired by common MVS practice, we also use SfM co-visibility with camera baseline and viewing direction to filter PGSR-style source views.

Our main contributions are summarized as follows:

• We introduce a pre-flattened and normal-aligned Gaussian initialization strategy using MVS dense point clouds. This approach explicitly embeds geometric priors into the primitives, reducing ambiguous normal recovery in standard isotropic initialization.

• We develop PatchMatch-3DGS Mutual Supervision, coupling PatchMatch depth refinement with Gaussian optimization to provide explicit depth-level supervision.

• We employ a Delaunay graph-cut meshing pipeline for TSDF-free surface extraction, reducing sensitivity to voxel resolution and improving scalability in outdoor scenes.

## 2 Related Work

## 2.1 Neural and Gaussian-Based Surface Reconstruction

Neural radiance fields optimize continuous scene representations from posed images. NeRF [23] introduced diferentiable volume rendering, while VolSDF [33], NeuS [29], and Neuralangelo [20] improve geometry with signed distance fields or multi-resolution encodings. These implicit methods provide strong modeling capacity, but usually require dense ray sampling and lengthy optimization.

3D Gaussian Splatting (3DGS) [17] replaces ray marching with eficient Gaussian rasterization. Since standard 3DGS lacks an explicit surface, later methods impose geometric priors for mesh extraction. SuGaR [11] aligns Gaussians with local surfaces, while 2DGS [13] and Gaussian Surfels [7] use flattened primitives. GOF [34] extracts surfaces from opacity fields, 3DGSR [22] adds an implicit surface branch, and RaDe-GS [35] improves depth rasterization. Their geometry still depends heavily on initialization, regularization, and meshing. PGSR [4] further flattens Gaussians and adds multi-view photometric and geometric consistency. Despite this progress, Gaussian reconstruction still faces a tension between appearance fitting and reliable geometry in weakly observed regions.

## 2.2 Multi-View Stereo and PatchMatch

Classical photogrammetric pipelines estimate camera poses with SfM and recover dense geometry with MVS. COLMAP [25, 26] combines robust SfM with pixelwise view selection for unstructured MVS. Semi-Global Matching [12] and PatchMatch stereo [1, 2] are influential dense matching paradigms; the latter estimates per-view depth and normal hypotheses from multi-view matching.

PatchMatch-based MVS improves eficiency through plane propagation, view selection, and consistency filtering. Gipuma [9] accelerates normal difusion on GPUs, ACMM [32] strengthens multi-scale consistency, and TAPA-MVS [24] targets textureless regions. OpenMVS [3] provides a practical pipeline with depth estimation, fusion, Delaunay graph-cut meshing, and mesh refinement. However, MVS remains sensitive to repetitive, or weakly textured surfaces. We use PatchMatch as an explicit geometric supervisor for Gaussian optimization, rather than treating MVS only as separate preprocessing or final reconstruction.

## 2.3 Mesh Extraction from Depth Maps

Volumetric fusion integrates depth observations into a Truncated Signed Distance Function (TSDF) [6] and extracts an iso-surface with Marching Cubes [21]; it is widely used in libraries such as Open3D [38]. Many 3DGS methods also adopt TSDF fusion over rendered depths [13, 4, 34]. However, TSDF fusion is sensitive to voxel resolution: coarse grids suppress details, whereas fine grids increase memory use and may expose holes.

Point-based and graph-based alternatives operate directly on samples and visibility. Poisson reconstruction [16] estimates an implicit indicator from oriented points. Delaunay graphcut reconstruction builds a tetrahedral complex from samples and camera rays, then infers inside/outside labels through visibility-aware optimization [18, 14]. This strategy has been integrated into high-resolution MVS pipelines [28]. We use a Delaunay graph-cut meshing pipeline [14, 28] to convert depth maps refined by PatchMatch-3DGS Mutual Supervision into a mesh, reducing the sensitivity to TSDF voxel resolution.

![](images/0b6998c17670338c276e1e620620eb49c368543664faa8e7d22641b2dfbd2fd5.jpg)  
Figure 2: Overview of CoMVS-GS. (a) MVS dense points provide pre-flattened and normalaligned Gaussian initialization without requiring a precomputed mesh. (b) PatchMatch-3DGS Mutual Supervision refines Gaussian geometry, with MVS-style filtering used for PGSR-style source views. (c) TSDF-free surface extraction through a Delaunay graph-cut meshing pipeline.

## 3 Method

Given posed images, CoMVS-GS integrates PGSR-style surface-aware optimization with explicit MVS geometry and visibility-consistent meshing, as illustrated in Fig. 2. It first initializes Gaussians from MVS dense points with pre-flattened scales and MVS-derived normal alignment, providing dense and surface-aware geometry. PatchMatch-3DGS Mutual Supervision then couples Gaussian optimization with MVS depth refinement. For the PGSR-style photometric and reprojection losses, we use an MVS-style source-view filter based on sparse-point co-visibility, camera baseline, and viewing direction. Finally, the optimized depth maps are fused into surface samples and meshed with Delaunay graph cuts, avoiding the voxel-resolution trade-of of TSDF fusion.

## 3.1 Preliminaries: Planar Gaussian Splatting

We build on the planar Gaussian representation of PGSR [4]. A scene is represented by Gaussian primitives $\mathcal { G } = \{ G _ { i } \} _ { i = 1 } ^ { N }$ , each with a center $\pmb { \mu } _ { i } ,$ opacity $o _ { i } ,$ anisotropic covariance $\Sigma _ { i \cdot }$ and spherical-harmonic color coeficients. For pixel $p ,$ visible Gaussians are depth-sorted and alpha-blended:

$$
\hat { \mathbf { C } } ( p ) = \sum _ { i \in \mathcal { R } ( p ) } T _ { i } ( p ) \alpha _ { i } ( p ) \mathbf { c } _ { i } , \quad T _ { i } ( p ) = \prod _ { j < i } \bigl ( 1 - \alpha _ { j } ( p ) \bigr ) ,\tag{1}
$$

where $\alpha _ { i } ( p )$ is the projected opacity and $T _ { i } ( p )$ is the accumulated transmittance.

PGSR flattens each Gaussian along its shortest scale axis, making it a local planar surface element. With camera-space plane normal $\mathbf { n } _ { i } ^ { c }$ and signed distance $d _ { i } ^ { c }$ , the normal map and camera-to-plane distance map are rendered with the same blending weights:

$$
\hat { \mathbf { N } } ( p ) = \sum _ { i \in \mathcal { R } ( p ) } T _ { i } ( p ) \alpha _ { i } ( p ) \mathbf { n } _ { i } ^ { c } , \quad \hat { D } ( p ) = \sum _ { i \in \mathcal { R } ( p ) } T _ { i } ( p ) \alpha _ { i } ( p ) d _ { i } ^ { c } .\tag{2}
$$

Given ray direction $\mathbf { r } ( p ) = \mathbf { K } ^ { - 1 } \tilde { p } .$ , the rendered depth is obtained by intersecting the ray with the blended plane:

$$
\hat { z } ( p ) = \frac { \hat { D } ( p ) } { \hat { \mathbf { N } } ( p ) ^ { \top } { \mathbf { r } ( p ) } } .\tag{3}
$$

## 3.2 Dense Point Cloud Initialization

While conventional planar Gaussian methods initialize primitives as isotropic spheres and rely on unguided optimization to recover surface normals, we leverage the dense MVS point cloud $\mathcal { P } _ { m v s } = \{ ( \mathbf { x } _ { i } , \mathbf { c } _ { i } , \mathbf { n } _ { i } ) \} _ { i = 1 } ^ { M }$ to introduce a pre-flattened and normal-aligned initialization. This strategy bridges spatial gaps left by sparse SfM points and embeds geometric priors into the primitives from the first iteration.

Specifically, each Gaussian is initialized as a flattened planar primitive aligned with the local MVS surface estimate:

$$
\mu _ { i } = { \bf x } _ { i } , \quad { \bf c } _ { i } ^ { S H } = \mathrm { R G B 2 S H } ( { \bf c } _ { i } ) , \quad { \bf R } _ { i } = \mathrm { A l i g n Z } ( { \bf n } _ { i } ) , \quad { \bf S } _ { i } = \mathrm { d i a g } ( s _ { x y } , s _ { x y } , \epsilon ) .\tag{4}
$$

This geometric initialization is implemented through two steps: Normal-Aligned Rotation $\mathbf { R } _ { i }$ : We establish a local orthogonal basis using the MVS-derived normal $\mathbf { n } _ { i }$ . The rotation matrix $\mathbf { R } _ { i }$ aligns the local Z-axis, i.e., the shortest axis of the planar Gaussian, with $\mathbf { n } _ { i } .$ . Pre-flattened Scale $\mathbf { S } _ { i }$ : Instead of standard isotropic scaling, we flatten the initial Gaussian by setting the scale along the normal direction to a small thickness ϵ. The planar extent $s _ { x y }$ is adaptively computed from the local neighborhood ${ \mathcal { N } } _ { i }$

By initializing Gaussians with flattened shapes and MVS-derived orientations, we reduce the reliance on ambiguous normal recovery during early optimization. This provides a betterconditioned starting point and suppresses floating artifacts in weakly observed areas.

## 3.3 PatchMatch-3DGS Mutual Supervision

## 3.3.1 Mutual Depth Refinement.

PatchMatch and 3DGS provide complementary geometric signals. PatchMatch uses multi-view matching to produce explicit depth hypotheses, but it can be sensitive to poor initialization and local matching ambiguities. Gaussian optimization provides dense rendered depth and normal maps, yet these maps may drift in weakly constrained regions when only photometric and PGSR-style regularization are used. We therefore couple the two sources of geometry in a closed loop.

During training, we periodically render the current Gaussian depth zˆ and normal $\hat { \bf N }$ using Eqs. (2)–(3). These maps initialize the plane hypotheses in PatchMatch, giving the MVS a geometry-aware starting point. PatchMatch then refines the depth maps by multi-view propagation, matching, and geometric consistency filtering, producing refined depths $z _ { p m }$ and a valid-pixel mask $\mathcal { M } _ { p m }$ . The refined depth is used as pseudo ground truth for Gaussian optimization:

$$
\mathcal { L } _ { p m } = \frac { 1 } { | \mathcal { M } _ { p m } | } \sum _ { p \in \mathcal { M } _ { p m } } \rho \big ( \hat { z } ( p ) - z _ { p m } ( p ) \big ) ,\tag{5}
$$

![](images/e9beb106ebfdf4ca5a1fdebe344e86c086de748772c3a3887cc8e15e069506e0.jpg)  
Figure 3: Illustration of PatchMatch-3DGS Mutual Supervision.

where $\rho ( \cdot )$ is a robust $\ell _ { 1 }$ penalty. This mutual supervision makes the Gaussian depth more metrically grounded, while the rendered Gaussian geometry improves the stability of subsequent PatchMatch refinement.

Fig. 3 illustrates this interaction. PatchMatch initialized only from the SfM result produces incomplete depth maps with visible holes. After 6k iterations, the Gaussian model already renders a more coherent depth and normal estimate, which provides a stronger initialization for PatchMatch. With this guidance, PatchMatch fills missing regions and corrects noisy normals on weakly observed geometry; the refined depth is then used to supervise subsequent Gaussian optimization. The final model consequently renders complete and accurate depth and normal maps.

## 3.3.2 Source-View Filtering for Multi-View Regularization.

Inspired by common MVS practice, we rank source views for the PGSR-style photometric and reprojection losses using sparse-point overlap and camera-pair geometry. For each image $I _ { i } ,$ let $\mathcal { Q } _ { i }$ be the set of SfM points visible in that image. We first compute the co-visibility ratio between two images as

$$
s _ { i j } = \frac { | \mathcal { Q } _ { i } \cap \mathcal { Q } _ { j } | } { \operatorname* { m i n } ( | \mathcal { Q } _ { i } | , | \mathcal { Q } _ { j } | ) } .\tag{6}
$$

For a reference image $I _ { r }$ , the source-view set is selected as

$$
\begin{array} { r } { S ( I _ { r } ) = \mathrm { T o p K } \{ I _ { j } \ | \ \Phi ( s _ { r j } , b _ { r j } , \theta _ { r j } ) > \tau _ { v i e w } \} , } \end{array}\tag{7}
$$

where $b _ { r j }$ and $\theta _ { r j }$ denote the camera-baseline and viewing-direction consistency, and $\Phi ( \cdot )$ is a ranking score combining overlap and camera geometry. The photometric constraint $\mathcal { L } _ { p c } ^ { f i l t }$ and geometric constraints $\mathcal { L } _ { g c } ^ { f i l t }$ are then computed over $I _ { s } \in { \mathcal { S } } ( I _ { r } )$ . Compared with using camera distance or viewing direction alone, adding sparse-point co-visibility better reflects actual image overlap under irregular camera trajectories, so the selected source views provide more useful PGSR-style supervision. We emphasize that this is an auxiliary filter for diferentiable Gaussian regularization and does not modify PatchMatch’s internal source-view selection.

Our training objective augments the PGSR losses with PatchMatch depth supervision and the filtered multi-view regularization:

$$
\mathcal { L } = \mathcal { L } _ { r g b } + \lambda _ { s } \mathcal { L } _ { s } + \lambda _ { n } \mathcal { L } _ { d n } + \lambda _ { p m } \mathcal { L } _ { p m } + \lambda _ { p c } \mathcal { L } _ { p c } ^ { f i l t } + \lambda _ { g c } \mathcal { L } _ { g c } ^ { f i l t } ,\tag{8}
$$

where $\mathcal { L } _ { s }$ is the Gaussian flattening regularizer inherited from PGSR, $\mathcal { L } _ { p m }$ is the PatchMatch depth supervision term, and $\mathcal { L } _ { p c } ^ { f i l t }$ and $\mathcal { L } _ { g c } ^ { f i l t }$ denote the PGSR multi-view losses computed with filtered source views.

## 3.4 Delaunay Graph-Cut Meshing Pipeline

After optimization, we render depth and normal maps from the final Gaussian model and fuse geometrically consistent depth pixels into surface samples. Following standard MVS filtering [27], each reference depth is checked against neighboring views using reprojection, relative-depth, and normal-angle consistency before being unprojected into a fused point cloud $\mathcal { P } _ { f u s e d } .$

Instead of accumulating all samples in a TSDF volume, we adopt visibility-consistent Delaunay graph-cut reconstruction [14]. A 3D Delaunay tetrahedralization is built from $\mathcal { P } _ { f u s e d }$ and camera centers, and camera rays provide free-space and occupied-space evidence for graph-cut labeling. The mesh is extracted from facets that separate tetrahedra with diferent labels and is then refined with image-based photometric and visibility constraints [28]. This sample-and-ray formulation operates on depth-derived samples and camera-ray visibility rather than a dense voxel grid, reducing the memory growth and voxel-size sensitivity of high-resolution TSDF fusion.

## 4 Experiments

## 4.1 Experimental Setup

Datasets and Implementation Details. We evaluate CoMVS-GS on both object-level and outdoor reconstruction. For object-level geometry, we use the standard DTU benchmark [15], which provides calibrated images and laser-scanned reference surfaces. For outdoor scenes, we construct three representative subsets: Sub-Village with 301 images from the Village scene and Sub-Technology College with 600 images from GauU-Scene V2 [31], and Sub-Small City with 986 images from MatrixCity [19]. These subsets avoid the excessive cost of full city-scale processing while retaining diverse land-cover, building layouts, thin structures, weakly textured surfaces, repeated facades, and occlusions. For all reconstruction experiments, input images are resized so that the longer side is 1600 pixels. All experiments are conducted on a GPU server equipped with an NVIDIA GeForce RTX 4090D.

Optimization Details. We optimize for 30,000 iterations with Adam. The appearance term is $\mathcal { L } _ { r g b } = 0 . 8 \mathcal { L } _ { 1 } + 0 . 2 ( 1 - \mathrm { S S I M } )$ , and the loss weights in Eq. (8) are $\lambda _ { s } = 1 0 0 , \lambda _ { n } = 0 . 0 1 5$ $\lambda _ { p m } = 0 . \dot { 3 } , \lambda _ { p c } = 0 . 1 5$ , and $\lambda _ { g c } = 0 . 0 3$ . The appearance and flattening terms are active throughout training; $\mathcal { L } _ { p m }$ starts at iteration 3,000, and $\mathcal { L } _ { d n } , \mathcal { L } _ { p c } ^ { f i l t }$ , and $\mathcal { L } _ { g c } ^ { f i l t }$ start at iteration 7,000. For $\mathcal { L } _ { p c } ^ { f i l t }$ , we use $7 \times 7$ patches, at most 102,400 samples, and a one-pixel reprojection threshold. The position learning rate decays from $4 \times 1 0 ^ { - 5 }$ to $1 . 6 \times 1 0 ^ { - 6 }$ and the scale learning rate is 0.001; other optimizer settings follow 3DGS [17]. Densification and pruning run every 200 iterations from iteration 1,000 to 15,000.

PatchMatch Configuration. We run the CUDA depth-map estimator in OpenMVS 2.3.0 [3] at iterations 3,000, 6,000, 9,000, 12,000, and 15,000, initializing each run with the current Gaussian-rendered depth and normal maps. The refined depths supervise the Gaussian model until the next update and remain fixed after the final call. We use resolution level 0 with a maximum image dimension of 1,600 pixels, two sub-resolution levels, up to eight source views, four standard and two geometric-consistency iterations, and a $9 \times 9$ matching window. The depth, normal, and photometric-cost thresholds are 0.01, 25<sup>◦</sup>, and $1 - \mathrm { N C C } = 0 . 9$ , respectively; fusion requires three agreeing views, post-processing flag 7 is enabled, and the pseudo-depth mask retains confidence values above 0.7.

Baselines & Metrics. On DTU, we adopt the standard 15-scan evaluation setting used in Geometry-Grounded Gaussian Splatting [36]. The compared methods include $\mathrm { O p e n M V S }$ [3] as a conventional photogrammetric baseline, implicit neural reconstruction methods, i.e., NeRF [23], VolSDF [33], NeuS [29], and Neuralangelo [20], as well as explicit Gaussian-based methods, including 3DGS [17], 2DGS [13], GOF [34], 3DGSR [22], RaDe-GS [35], PGSR [4], and Geometry-Grounded Gaussian Splatting [36]. We report Chamfer Distance (CD) for each scan. No runtime comparison is reported on DTU, since our focus on this benchmark is geometric accuracy.

Table 1: Quantitative comparison on the DTU dataset. We report Chamfer Distance, with the best and second-best results highlighted in bold and underlined.
<table><tr><td>Method</td><td>24</td><td>37</td><td>40</td><td>55</td><td>63</td><td>65</td><td>69</td><td>83</td><td>97</td><td>105</td><td>106</td><td>110</td><td>114</td><td>118</td><td>122 Mean</td></tr><tr><td>NeRF [23]</td><td>1.90</td><td>1.60</td><td>1.85</td><td>0.58</td><td>2.28</td><td>1.27</td><td>1.47</td><td>1.67</td><td>2.05</td><td>1.07</td><td>0.88 2.53</td><td>1.06</td><td>1.15</td><td>0.96</td><td>1.49</td></tr><tr><td>VolSDF [33]</td><td>1.14</td><td>1.26</td><td>0.81</td><td>0.49</td><td>1.25</td><td>0.70</td><td>0.72</td><td>1.29</td><td>1.18</td><td>0.70 0.66</td><td>1.08</td><td>0.42</td><td>0.61</td><td>0.55</td><td>0.86</td></tr><tr><td>NeuS [29]</td><td>1.00</td><td>1.37</td><td>0.93</td><td>0.43</td><td>1.10</td><td>0.65</td><td>0.57</td><td>1.48</td><td>1.09</td><td>0.83 0.52</td><td>1.20</td><td>0.35</td><td>0.49</td><td>0.54</td><td>0.84</td></tr><tr><td>Neuralangelo [20]</td><td>0.37</td><td>0.72</td><td>0.35</td><td>0.35</td><td>0.87</td><td>0.54</td><td>0.53</td><td>1.29</td><td>0.97 0.73</td><td>0.47</td><td>0.74</td><td>0.32</td><td>0.41</td><td>0.43</td><td>0.61</td></tr><tr><td>3DGS [17]</td><td>2.14</td><td>1.53</td><td>2.08</td><td>1.68</td><td>3.49</td><td>2.21</td><td>1.43</td><td>2.07</td><td>2.22</td><td>1.75 1.79</td><td>2.55</td><td>1.53</td><td>1.52</td><td>1.50</td><td>1.96</td></tr><tr><td>2DGS [13]</td><td>0.48</td><td>0.91</td><td>0.39</td><td>0.39</td><td>1.01</td><td>0.83</td><td>0.81</td><td>1.36</td><td>1.27</td><td>0.76 0.70</td><td>1.40</td><td>0.40</td><td>0.76</td><td>0.52</td><td>0.80</td></tr><tr><td>GOF [34]</td><td>0.50</td><td>0.82</td><td>0.37</td><td>0.37</td><td>1.12</td><td>0.74</td><td>0.73</td><td>1.18</td><td>1.29</td><td>0.68</td><td>0.77 0.90</td><td>0.42</td><td>0.66</td><td>0.49</td><td>0.74</td></tr><tr><td>3DGSR [22]</td><td>0.44</td><td>0.96</td><td>0.40</td><td>0.36</td><td>1.02</td><td>0.80</td><td>0.64 1.20</td><td></td><td>1.08</td><td>0.97</td><td>0.54 0.72</td><td>0.37</td><td>0.52</td><td>0.42</td><td>0.70</td></tr><tr><td>RaDe-GS [35]</td><td>0.43</td><td>0.75</td><td>0.35</td><td>0.37</td><td>0.81</td><td>0.74</td><td>0.74 1.19</td><td></td><td>1.20</td><td>0.65</td><td>0.61 0.84</td><td>0.35</td><td>0.66</td><td>0.46</td><td>0.68</td></tr><tr><td>PGSR [4]</td><td>0.37</td><td>0.54</td><td>0.44</td><td>0.37</td><td>0.78</td><td>0.57</td><td>0.49</td><td>1.06</td><td>0.63</td><td>0.59</td><td>0.47 0.50</td><td>0.30</td><td>0.37</td><td>0.34</td><td>0.52</td></tr><tr><td>GGGS [36]</td><td>0.37</td><td>0.50</td><td>0.27</td><td>0.31</td><td>0.81</td><td>0.43</td><td>0.42</td><td>1.05</td><td>0.64</td><td>0.52</td><td>0.32 0.58</td><td>0.30</td><td>0.31</td><td>0.33</td><td>0.48</td></tr><tr><td>OpenMVS [3]</td><td>0.44</td><td>0.69</td><td>0.28</td><td>0.32</td><td>0.80</td><td>0.67</td><td>0.50</td><td>0.72</td><td>0.72</td><td>0.68</td><td>0.41 0.57</td><td>0.29</td><td>0.39</td><td>0.49</td><td>0.53</td></tr><tr><td>CoMVS-GS</td><td>0.36</td><td>0.55</td><td>0.28</td><td>0.31</td><td>0.49</td><td>0.51</td><td>0.43</td><td>0.93</td><td>0.87</td><td>0.78</td><td>0.44 0.36</td><td>0.27</td><td>0.34</td><td>0.35</td><td>0.48</td></tr></table>

Table 2: Quantitative comparison on the Sub-Village, Sub-Technology College, and Sub-Small City. We report visual metrics, geometric metrics, and the average number of mesh faces. Dashes indicate unavailable or inapplicable metrics.
<table><tr><td rowspan="2">Method</td><td colspan="3">Visual Metrics</td><td colspan="2">Geometric Metrics</td><td rowspan="2">#Faces (×106)</td></tr><tr><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>MAE (m) ↓</td><td>RMSE (m) ↓</td></tr><tr><td>Neuralangelo [20]</td><td>24.650</td><td>0.686</td><td>0.404</td><td>0.816</td><td>1.473</td><td>33.063</td></tr><tr><td>2DGS [13]</td><td>25.569</td><td>0.799</td><td>0.277</td><td>0.853</td><td>1.408</td><td>35.954</td></tr><tr><td>RaDe-GS [35]</td><td>25.481</td><td>0.845</td><td>0.198</td><td>0.680</td><td>1.330</td><td>39.065</td></tr><tr><td>PGSR [4]</td><td>28.370</td><td>0.886</td><td>0.174</td><td>0.670</td><td>1.284</td><td>50.169</td></tr><tr><td>CityGS-X [10]</td><td>27.783</td><td>0.883</td><td>0.150</td><td>0.755</td><td>1.466</td><td>146.591</td></tr><tr><td>OpenMVS [3]</td><td></td><td></td><td></td><td>0.607</td><td>1.210</td><td>46.561</td></tr><tr><td>CoMVS-GS</td><td>27.813</td><td>0.883</td><td>0.173</td><td>0.603</td><td>1.133</td><td>8.762</td></tr></table>

For Sub-Village, Sub-Technology College, and Sub-Small City, we compare against methods that represent diferent reconstruction paradigms: Neuralangelo [20] for implicit SDF reconstruction, 2DGS [13], RaDe-GS [35], and PGSR [4] for explicit 3DGS-based surface reconstruction, CityGS-X [10] for scalable city-level Gaussian reconstruction, and OpenMVS [3] for a conventional photogrammetric pipeline. Rendering quality is evaluated on held-out images with PSNR, SSIM [30], and LPIPS [37]. We use the one-sided reference-point-to-mesh distance adopted in the outdoor evaluation protocol [5], and report MAE and RMSE. Unlike symmetric Chamfer Distance on DTU, this metric avoids penalizing reconstructed regions that are visible in the images but fall outside the reference LiDAR or simulated geometry.

## 4.2 Reconstruction Comparison

Table 1 reports the DTU Chamfer Distance results under the standard 15-scan protocol. CoMVS-GS obtains an average CD of 0.48, which is comparable to the strongest geometry-oriented baselines and clearly improves over PGSR and OpenMVS. The first row of Fig. 4 shows the qualitative comparison on DTU scan 40. PGSR fails to fully reconstruct weakly observed boundary regions and produces visible floaters, leading to a smaller surface coverage than OpenMVS and CoMVS-GS. In contrast, CoMVS-GS reconstructs a complete surface comparable <sup>I20240402151700\_0076\_Zenmuse-L1-mission\_cam</sup>to OpenMVS while preserving the advantages of Gaussian optimization. Fig. 3 further shows that, around the roof region of a DTU house model, CoMVS-GS avoids the local depression observed in PGSR and produces a flatter surface.

![](images/e9faffc5db6a1c963155b1d1439c12af21f7b11fa6f0cb7ed08b6c5756e5223d.jpg)  
Figure 4: Qualitative comparison on DTU scan 40 and Sub-Technology College. From left to right, each row shows the input image, PGSR, OpenMVS, and CoMVS-GS.

Table 2 further evaluates reconstruction on outdoor scenes. CoMVS-GS remains close to PGSR and CityGS-X in rendering metrics, ranking second in PSNR and SSIM, while achieving the best geometric accuracy with the lowest MAE and RMSE. This contrast suggests that high image-space fidelity does not necessarily translate into the most accurate surface geometry, especially in outdoor regions with occlusion and uneven view coverage. The second row of Fig. 4 illustrates this behavior on Sub-Technology College. PGSR produces floating fragmented surfaces above the scene due to insuficient geometric constraints, while OpenMVS leaves large holes in weakly textured water regions. CoMVS-GS combines MVS geometric evidence with Gaussian optimization, yielding a cleaner and more complete reconstruction without obvious floating triangles or water-surface holes.

The last column of Table 2 also shows that CoMVS-GS produces substantially more compact meshes, with only 8.762M faces on average. TSDF-based extraction typically requires highresolution voxels to preserve fine structures, which can produce dense, heavy meshes with many faces, especially in outdoor scenes. In contrast, Delaunay graph-cut meshing operates on depth-derived surface samples and camera-ray visibility, allowing CoMVS-GS to preserve reliable geometric evidence while avoiding excessive face growth from dense voxelization.

## 4.3 Ablation Studies

Impact of Dense Point Cloud Initialization. The ablation experiments are conducted on Sub-Small City. Fig. 5 shows that initializing 3DGS from dense MVS points with pre-flattened scales and normal-aligned orientations leads to flatter building facades. Sparse SfM initialization provides few reliable anchors in weakly featured or sparsely observed facade regions, making it dificult for Gaussians to converge to the correct surface. Table 3 further confirms that dense initialization improves both geometric error metrics.

PatchMatch-3DGS Mutual Supervision. PatchMatch-3DGS Mutual Supervision further stabilizes regions with limited observations. Building facades are often viewed at oblique angles and occupy only small image areas, so photometric optimization alone provides insuficient geometric constraints. The proposed mutual supervision supplies persistent depth-level guidance from refined PatchMatch maps, resulting in smoother and more planar building surfaces in Fig. 5. As shown in Table 3, this module improves surface accuracy, although the visual metrics slightly decrease compared with the variant without PatchMatch supervision. This trade-of is consistent with the observation in PGSR [4]: stronger geometric constraints restrict the freedom of Gaussians to overfit training views, which may reduce rendering quality but yields more reliable geometry.

![](images/7882555b8068d3964c5c29e8e02e780e556241f0bc244b3c35c575470863785e.jpg)  
Sparse Point Cloud Initialization

![](images/af0c30e202143fa07d21f129de3334d5b384e66ad0c7ff6041682c9a0886533b.jpg)  
w/o PatchMatch Supervision

![](images/d63a8532bc0cf8e8bce1d0777009dc5cb1baf986e43bfa5de543683a6bbd9b1f.jpg)  
TSDF Fusion

![](images/e4bd8baa7bc7273873ad18d6bf18d981c25b928b20d246100182e20759511055.jpg)  
Dense Point Cloud Initialization

![](images/8b4e059b3da939d3cf2c5742586b5692e40f6a084fbcfbd71984d162c13e91b1.jpg)  
Full Model

![](images/4acfcc26acdd47f91169cd84c859b4baa8d5c235add3d01f733a7619a9359d41.jpg)  
Delaunay Graph-Cut Meshing

Figure 5: Ablation on the Sub-Small City, showing the efects of PatchMatch supervision, Delaunay graph-cut meshing, and dense point cloud initialization.  
Table 3: Ablation of PatchMatch supervision, Delaunay graph-cut meshing, and dense point cloud initialization. We report visual metrics and geometric metrics.
<table><tr><td rowspan="2">Variant</td><td colspan="3">Visual Metrics</td><td colspan="2">Geometric Metrics</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>MAE (m) ↓</td><td>RMSE (m) ↓</td></tr><tr><td>Base Model</td><td>28.851</td><td>0.882</td><td>0.189</td><td>0.922</td><td>1.727</td></tr><tr><td>TSDF</td><td>28.994</td><td>0.898</td><td>0.157</td><td>0.543</td><td>1.161</td></tr><tr><td>w/o PatchMatch</td><td>29.341</td><td>0.899</td><td>0.161</td><td>0.579</td><td>1.133</td></tr><tr><td>SfM Initialization</td><td>28.860</td><td>0.882</td><td>0.189</td><td>0.599</td><td>1.165</td></tr><tr><td>Full Model</td><td>28.994</td><td>0.898</td><td>0.157</td><td>0.512</td><td>1.047</td></tr></table>

TSDF vs. Delaunay Graph-Cut Meshing Scalability. We also compare the Delaunay graph-cut meshing pipeline with TSDF fusion. TSDF extraction requires choosing a voxel size that balances detail preservation and completeness. In this experiment, even a relatively fine 0.4m voxel resolution still produces holes and incomplete regions in Fig. 5. Delaunay graph-cut meshing instead reasons over depth-derived surface samples and camera-ray visibility, producing a more complete mesh and achieving the best overall geometric accuracy in Table 3.

## 5 Conclusion

We presented CoMVS-GS, an MVS-enhanced 3DGS framework for surface reconstruction across object-level and outdoor scenes. By initializing Gaussians from dense MVS points, coupling PatchMatch depth refinement with Gaussian optimization, and extracting meshes through Delaunay graph cuts, CoMVS-GS improves geometric stability in weakly constrained regions without relying on TSDF voxel fusion. The method uses MVS not as an independent final reconstruction stage, but as dense initialization, iterative depth-level supervision, and visibilityconsistent surface evidence. Experiments on DTU, GauU-Scene V2, and MatrixCity show that CoMVS-GS remains competitive on object-level reconstruction and achieves accurate geometry with substantially more compact meshes. The ablation studies further verify that dense initialization, PatchMatch-3DGS Mutual Supervision, and Delaunay graph-cut meshing each contribute to the final reconstruction quality. This highlights the importance of jointly considering initialization, optimization supervision, and mesh extraction in geometry-aware Gaussian reconstruction.

Limitations and Future Work. CoMVS-GS still inherits PatchMatch’s limitations on highly specular or transparent surfaces, where reliable multi-view correspondences are dificult to establish. Periodic PatchMatch refinement also introduces additional computational overhead, especially for high-resolution outdoor scenes. Future work will explore learned matching costs and adaptive refinement schedules to further improve robustness and eficiency.

## References

[1] Connelly Barnes, Eli Shechtman, Adam Finkelstein, and Dan B. Goldman. Patchmatch: A randomized correspondence algorithm for structural image editing. ACM Transactions on Graphics, 28(3):24, 2009.

[2] Michael Bleyer, Christoph Rhemann, and Carsten Rother. Patchmatch stereo – stereo matching with slanted support windows. In BMVC, pages 14.1–14.11, 2011.

[3] Dan Cernea. Openmvs: Multi-view stereo reconstruction library. https://cdcseacave. github.io/openMVS, 2020.

[4] Danpeng Chen et al. Pgsr: Planar-based gaussian splatting for eficient and high-fidelity surface reconstruction. IEEE Transactions on Visualization and Computer Graphics, 2024.

[5] Shihan Chen, Zhaojin Li, Zeyu Chen, Qingsong Yan, Gaoyang Shen, and Ran Duan. 3d gaussian splatting for fine-detailed surface reconstruction in large-scale scene. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 15703–15710. IEEE, 2025.

[6] Brian Curless and Marc Levoy. A volumetric method for building complex models from range images. In SIGGRAPH, pages 303–312, 1996.

[7] Pinxuan Dai, Jiamin Xu, Wenxiang Xie, and Huamin Wang. High-quality surface reconstruction using gaussian surfels. In SIGGRAPH, 2024.

[8] Zongcheng Fan et al. Trim 3d gaussian splatting for accurate geometry representation. In SIGGRAPH, 2024.

[9] Silvano Galliani, Katrin Lasinger, and Konrad Schindler. Massively parallel multiview stereopsis by surface normal difusion. In ICCV, pages 873–881, 2015.

[10] Yao Gao, Huan Li, Jian Chen, Zhengxia Zou, Zhi Zhong, Deyu Zhang, Xian Sun, and Junwei Han. Citygs-x: A scalable architecture for eficient and geometrically accurate large-scale scene reconstruction. arXiv preprint arXiv:2503.23044, 2025.

[11] Antoine Gu´edon and Vincent Lepetit. Sugar: Surface-aligned gaussian splatting for eficient 3d mesh reconstruction and high-quality mesh rendering. In CVPR, pages 5354–5363, 2024.

[12] Heiko Hirschmuller. Stereo processing by semiglobal matching and mutual information. IEEE Transactions on Pattern Analysis and Machine Intelligence, 30(2):328–341, 2008.

[13] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In SIGGRAPH, 2024.

[14] Michal Jancosek and Tom´as Pajdla. Multi-view reconstruction preserving weakly-supported surfaces. In CVPR, pages 3121–3128, 2011.

[15] Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, and Henrik Aanæs. Large scale multi-view stereopsis evaluation. In CVPR, pages 406–413, 2014.

[16] Michael Kazhdan, Matthew Bolitho, and Hugues Hoppe. Poisson surface reconstruction. In SGP, pages 61–70, 2006.

[17] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):139:1– 139:14, 2023.

[18] Patrick Labatut, Jean-Philippe Pons, and Renaud Keriven. Robust and eficient surface reconstruction from range data. Computer Graphics Forum, 28(8):2275–2290, 2009.

[19] Yixuan Li et al. Matrixcity: A large-scale city dataset for city-scale neural rendering and beyond. In ICCV, pages 3205–3215, 2023.

[20] Zhaoshuo Li, Thomas M¨uller, Alex Evans, Russell H. Taylor, Mathias Unberath, Ming-Yu Liu, and Chen-Hsuan Lin. Neuralangelo: High-fidelity neural surface reconstruction. In CVPR, pages 8456–8465, 2023.

[21] William E. Lorensen and Harvey E. Cline. Marching cubes: A high resolution 3d surface construction algorithm. In SIGGRAPH, pages 163–169, 1987.

[22] Xiaoyang Lyu et al. 3dgsr: Implicit surface reconstruction with 3d gaussian splatting. In ECCV, 2024.

[23] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, pages 405–421, 2020.

[24] Andrea Romanoni and Matteo Matteucci. Tapa-mvs: Textureless-aware patchmatch multiview stereo. In ICCV, pages 10413–10422, 2019.

[25] Johannes L. Sch¨onberger and Jan-Michael Frahm. Structure-from-motion revisited. In CVPR, pages 4104–4113, 2016.

[26] Johannes L. Sch¨onberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys. Pixelwise view selection for unstructured multi-view stereo. In ECCV, pages 501–518, 2016.

[27] Shuhan Shen. Accurate multiple view 3d reconstruction using patch-based stereo for large-scale scenes. IEEE Transactions on Image Processing, 22(5):1901–1914, 2013.

[28] Hiep-Huy Vu, Patrick Labatut, Jean-Philippe Pons, and Renaud Keriven. High accuracy and visibility-consistent dense multiview stereo. IEEE Transactions on Pattern Analysis and Machine Intelligence, 34(5):889–901, 2012.

[29] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. In NeurIPS, pages 27171–27183, 2021.

[30] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004.

[31] Biao Xiong, Nan Zheng, Jiahao Liu, and Zhen Li. Gauu-scene v2: Assessing the reliability of image-based metrics with expansive lidar image dataset using 3dgs and nerf. arXiv preprint arXiv:2404.04880, 2024.

[32] Qingshan Xu and Wenbing Tao. Multi-scale geometric consistency guided multi-view stereo. In CVPR, pages 5483–5492, 2019.

[33] Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. Volume rendering of neural implicit surfaces. In NeurIPS, pages 4805–4815, 2021.

[34] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian opacity fields: Eficient and compact surface reconstruction in unbounded scenes. In NeurIPS, 2024.

[35] Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang, Xiao-Xiao Long, and Ping Tan. Rade-gs: Rasterizing depth in gaussian splatting. ACM Transactions on Graphics, 45(2):1–14, 2026.

[36] Baowen Zhang, Chenxing Jiang, Heng Li, Shaojie Shen, and Ping Tan. Geometry-grounded gaussian splatting. arXiv preprint arXiv:2601.17835, 2026.

[37] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable efectiveness of deep features as a perceptual metric. In CVPR, pages 586–595, 2018.

[38] Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. Open3d: A modern library for 3d data processing. arXiv preprint arXiv:1801.09847, 2018.