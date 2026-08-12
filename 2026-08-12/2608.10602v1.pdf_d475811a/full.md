# Gaussian Sculpting: End-to-End Controllable Surface Reconstruction via Field Optimization

KE JIAXIN, Dalian University of Technology, China   
JUNCHENG LIU, Massey University, New Zealand   
YI WANG, Dalian University of Technology, China   
ZHOUHUI LIAN, Peking University, China   
BIN LIU, Dalian University of Technology, China   
SHENGFA WANG, Dalian University of Technology, China   
XIANGJIA HE, School of Computer Science, University of Notingham Ningbo China, China

![](images/6cc6e1f2b243003467299d55cd105bb62bdf43c720574ab67f2c912866c8f7ad.jpg)  
Fig. 1. Gaussian Sculpting is an end-to-end surface reconstruction framework based on field optimization that progressively refines mesh geometry to recover fine details and eliminate erroneous structures. Compared with Gaussian opacity-field (GOF) [Yu et al. 2024b], our method produces higher-quality meshes. We also alleviate the geometric incompleteness exhibited by PGSR [Chen et al. 2024] and 2DGS [Huang et al. 2024], and avoids the floating artifacts commonly produced by NeuS [Wang et al. 2021].

3D Gaussian Splatting (3DGS) has recently enabled real-time novel view synthesis with impressive quality. However, it struggles to recover accurate surfaces under limited viewpoints and due to the inherent irregularity of Gaussian primitives. The resulting geometric errors are notoriously dificult to correct manually. To address these issues, we propose Gaussian Sculpting, a fully diferentiable end-to-end framework for high-quality surface reconstruction. Our key insight is to anchor Gaussians onto an evolving diferentiable surface, allowing them to guide signed distance field (SDF) optimization instead of extracting the surface only during post-processing. To enable stable gradient isolation during joint optimization, we design a bi-level training strategy in which the outer loop optimizes the geometry represented by the SDF, while the inner loop updates the Gaussians with the geometry fixed. We further impose constraints on Gaussian parameters to ensure consistency with the underlying surface, thereby improving both geometric and appearance fidelity during optimization. In addition, we introduce a multi-resolution subdivision scheme based on octree-like partitioning to preserve fine details while reducing memory consumption. Experiments on object-level scenes demonstrate that our method efectively removes redundant surfaces, recovers missing structures caused by limited

viewpoints, and achieves strong reconstruction quality even at relatively low resolutions.

CCS Concepts: • Computing methodologies → Mesh geometry models.

Additional Key Words and Phrases: Surface Reconstruction,Neural Implicit Representation,Field Optimization, Gaussian Splatting, End-to-end Training

## 1 Introduction

Multi-view surface reconstruction remains a challenging problem in computer graphics and vision, holding significant value for applications such as digital products [Chen et al. 2025; Li et al. 2025; Lu et al. 2025; Tang et al. 2024], robotics [Fan et al. 2025; Pan et al. 2025], and augmented/virtual reality (AR/VR) [Deng et al. 2022; Kim et al. 2025; Pechko et al. 2025; Ye et al. 2021; Zhang et al. 2021]. In recent years, 3D Gaussian Splatting (3DGS) [Kerbl et al. 2023] has become an efective method for novel view synthesis, due to its real-time rendering capability and eficient training. However, visually plausible renderings may still correspond to incomplete or unstable surfaces. Notably, downstream tasks such as game asset modeling and map construction demand more than just visual navigation. They require editable and structured geometric models. Thus, extracting meshes from Gaussian representations has emerged as a crucial research direction.

Prior methods for surface extraction from 3D Gaussians [Chen et al. 2024; Guédon and Lepetit 2024; Huang et al. 2024] treat reconstruction as a post-processing step, applying Marching Cubes (MC) [Lorensen and Cline 1998] or Poisson reconstruction [Kazhdan et al. 2006] to optimized Gaussians. Because this process is decoupled from end-to-end optimization, the resulting surface heavily depends on the Gaussian quality, yielding sensitivity to noise and poor geometric fidelity. Under limited or missing viewpoints, ambiguous Gaussian shapes and artifacts lead to incomplete geometry, which improvements in Gaussian geometry alone cannot resolve. Recent field-optimized Gaussian methods [Yu et al. 2024a,b] partially address these issues. However, due to the irregular and disordered nature of Gaussian distributions, neither the Gaussians nor their derived fields can faithfully align with a triangle mesh. This mismatch propagates discretization errors, including floating surfaces and sliver triangles [Cheng et al. 2000]. Obtaining a watertight, clean, and continuously optimizable mesh from limited-view RGB images therefore remains a challenging open problem.

To address these limitations, we reformulate surface reconstruction by optimizing a signed distance field (SDF) instead of Gaussian primitives, leveraging diferentiable Gaussian splatting for rendering supervision and integrating diferentiable surface extraction, thereby establishing a fully end-to-end optimized pipeline. Unlike prior diferentiable surface extraction methods that initialize optimization with discrete, noisy explicit SDFs, we represent the field implicitly using a multi-layer perceptron (MLP), which provides continuous geometry, improved smoothness, and better control over resolution. During optimization, surfaces are extracted from the evolving field, and Gaussians are dynamically instantiated on the resulting triangle mesh for the splatting process.

Crucially, Gaussians are not treated as independent optimization variables for correcting geometry. Instead, they serve as geometryaware rendering proxies that faithfully reflect the current surface quality and guide field optimization. To enforce this role, each Gaussian is tightly anchored to the mesh surface, with its position, orientation, and scale explicitly determined by the corresponding triangle vertices and normal vectors. Furthermore, we introduce a set of Gaussian constraints that regulate opacity, scale range, and spatial distribution, ensuring consistent alignment between Gaussian primitives and the underlying surface geometry while preventing rendering-induced artifacts.

To further stabilize optimization, we design a bi-level optimization framework in which the outer loop refines the SDF and surface geometry, while the inner loop updates Gaussian parameters with the geometry fixed. This strategy prevents rendering supervision from interfering with geometry optimization. Moreover, we adopt a progressive octree-like subdivision scheme that confines highresolution computation to reliably reconstructed surface regions. As the geometry is progressively refined through field optimization under structured Gaussian supervision, we refer to our method as Gaussian Sculpting.

Our main contributions are summarized as follows:

• We propose Gaussian Sculpting, an end-to-end framework for surface reconstruction that uses constrained Gaussians to guide SDF optimization.

• We identify the inconsistency between Gaussian rendering and surface geometry, and introduce geometry and opacity constraints to better align Gaussians with the underlying surface.

• We develop a bi-level optimization framework with gradient isolation and progressive subdivision for stable geometry refinement.

• Our method reconstructs cleaner and more complete meshes than prior NeRF-based and Gaussian-based methods, reducing floating artifacts and recovering missing regions.

## 2 Related Work

## 2.1 Traditional Surface Reconstruction

Traditional surface reconstruction aims to recover accurate geometric surfaces from sparse or noisy observations. Most methods follow a multi-view stereo (MVS) pipeline, establishing correspondences across multi-view images via patch matching [Bleyer et al. 2011] and reconstructing surfaces through triangulation or implicit surface reconstruction [Zhang et al. 2025]. The intermediate representations used in these methods include meshes, point clouds [Furukawa and Ponce 2010], depth maps [Campbell et al. 2008], and voxels [Kutulakos and Seitz 2000]. These representations exhibit inherent limitations. Point-based and surface-based methods, such as Poisson Surface Reconstruction [Kazhdan et al. 2006], rely heavily on accurate correspondence estimation. When texture is insuficient, reliable correspondence becomes dificult to establish, often leading to artifacts and missing regions. Voxel-based approaches, such as those using MC [Lorensen and Cline 1998], provide a structured representation but sufer from high memory consumption when applied to high-resolution reconstruction.

## 2.2 Neural Surface Reconstruction

With the advancement of neural networks, recent methods have explored learning-based surface reconstruction from single or multiple images in an end-to-end manner, representing shapes as point clouds [Fan et al. 2017; Lin et al. 2018], voxels [Choy et al. 2016; Xie et al. 2019], or implicit fields [Mescheder et al. 2019]. Neural Radiance Fields (NeRF) [Mildenhall et al. 2021] synthesize photorealistic novel views via alpha compositing along camera rays but struggle to recover high-fidelity surface geometry. NeuS [Wang et al. 2021] integrates signed distance functions into volume rendering, enabling more accurate surface reconstruction and improved robustness in scenes with sharp geometric transitions. Despite their success, NeRF-based methods [Li et al. 2023; Mildenhall et al. 2021; Wang et al. 2021, 2023; Yariv et al. 2021, 2023] are often limited by high inference cost and restricted representational capacity due to reliance on multi-layer perceptrons (MLPs). To address this, recent approaches decompose scene representation into points [Xu et al. 2022], voxels [Li et al. 2022], and other explicit structures, reducing dependence on MLP-based implicit modeling.

## 2.3 Gaussian Splating based Surface Reconstruction

We categorize existing work into four directions: (I) enhancing geometric adaptability, (II) extracting surfaces via post-processing, (III) improving Gaussian rendering, and (IV) leveraging Gaussians to guide geometry or field optimization.

![](images/2225b41281ef0409fe96c9a8bbf2008ce226c6c889adc05614d79fe7d1c0c648.jpg)  
Fig. 2. Pipeline of Gaussian Sculpting. In each iteration of mesh optimization, the SDF network predicts SDF values at voxel vertices, from which an intermediate mesh is extracted through diferentiable surface extraction. An inner loop independently optimizes Gaussians anchored to the mesh under Gaussian constraints. We use a detached representation to prevent rendering gradients from interfering with geometry optimization. Red-cross arrows mean those original Gaussians with SDF gradients are excluded from internal rendering. The optimized Gaussians are then used for rendering supervision with RGB and mask losses. Red arrows denote mesh-related gradients, while yellow arrows denote Gaussian-related gradients.

2D Gaussian Splatting (2DGS) [Huang et al. 2024] collapses 3D Gaussian ellipsoids into oriented planar disks, yielding viewpointconsistent geometry and a novel paradigm for fitting irregular Gaussians to smooth surfaces. Building on this, Gaussian Mesh Splatting (GaMeS) and Mani-gs [Gao et al. 2025; Waczyńska et al. 2024] redefine the parameters of Gaussians through geometric methods, introducing an adaptive triangle-aware Gaussian binding strategy. MeshGS [Choi et al. 2024] binds Gaussians to surfaces via distance constraints, balancing alignment and detail for large-scale render ing.

Regarding surface extraction post-processing, related studies provide key information, including sample points, field distributions, and depth maps from Gaussian scenes, which serve as the basis for surface extraction. SuGar [Guédon and Lepetit 2024] uses regularization to align Gaussians with surfaces and supports Poisson reconstruction via level-set sampling, but its externally defined SDF misrepresents surfaces and causes bubble artifacts. Methods exemplified by 2DGS [Huang et al. 2024] extract surfaces using MC [Lorensen and Cline 1998] or Marching Tetrahedra (MT) [Lorensen and Cline 1998] after TSDF fusion [Izadi et al. 2011]. GOF [Yu et al. 2024b] replaces TSDF with opacity fields and MT, yet sufers high cost and poor mesh quality, failing to overcome Gaussian limitations.

For rendering quality optimization, PGSR [Chen et al. 2024] introduces single-view geometric and multi-view photometric regularization terms within the Gaussian splatting framework to enhance reconstruction quality. GausSurf [Wang et al. 2024] guides 3D Gaussian optimization in texture-rich regions through patch-based multi-view stereo matching, while in texture-sparse regions it employs normal priors from pre-trained models to constrain Gaussians. Although these methods improve the eficiency of Gaussian rendering and yield better results, they still fail to resolve surface quality issues.

Now several methods combine Gaussians with SDF optimization. GSDF [Yu et al. 2024a] jointly supervises rendering and reconstruction via mutual guidance; 3DGSR [Lyu et al. 2024] loosely couples a neural SDF with 3DGS through geometric regularization; and GS-ROR2 [Zhu et al. 2025] enables bidirectional Gaussian-SDF refinement, pruning outliers and refining normals for reflective surfaces. Despite these advances, they remain constrained by noisy Gaussian estimates and inconsistent depth cues, which introduce floating artifacts and compromise geometric fidelity. Moreover, their typically decoupled training strategies weaken geometry-appearance interaction, limiting the efectiveness of joint optimization [Yu et al. 2024a].

Motivated by these limitations, we propose a fully diferentiable surface reconstruction framework that optimizes an SDF and enforces strict surface binding of Gaussian primitives.

## 3 Method

We propose an end-to-end SDF optimization framework for surface reconstruction from multi-view RGB images (Fig. 2). Our approach jointly optimizes an SDF network and Gaussian representations. Unlike methods that extract surfaces as a post-processing step, we employ a diferentiable iso-surface extraction module to derive the surface from the current SDF. Gaussians are then instantiated on the extracted mesh and used for diferentiable rendering. We further introduce Gaussian constraints to maintain consistency between Gaussian rendering and the underlying surface.

## 3.1 Field Optimization

This section introduces our implicit SDF formulation, the surface extraction process, and our surface-constrained Gaussian representation.

Implicit Field Representation: We represent the implicit SDF using an MLP that maps 3D coordinates to signed distance values and high-dimensional features.

Diferentiable Surface Extraction: We extend Flexicubes [Shen et al. 2023] to support multi-resolution grids and SDF inputs, en abling diferentiable iso-surface extraction with improved topological consistency and hierarchical adaptivity. The extraction method introduces three groups of learnable parameters: interpolation weights $\alpha \in \mathbb { R } _ { > 0 } ^ { 8 }$ and $\beta \bar { \in } \mathbb { R } _ { > 0 } ^ { 1 \bar { 2 } }$ per grid cell for dual vertex positioning, a split weight $\gamma \in \mathbb { R } _ { > 0 }$ per grid cell for controlling adaptive triangulation, and a deformation vector $\delta \in \mathbb { R } ^ { 3 }$ per grid vertex for spatial alignment. These parameters enable gradient-based optimization while improving surface fitting quality.

Surface Gaussian Model: Upon obtaining the reconstructed surface, we utilize Gaussian rendering to reflect its current quality. The original 3DGS relies on RGB supervision and often fails to establish a precise correspondence to the true object geometry. Therefore, we redesign the Gaussian parameterization. Our method initializes Gaussian attributes from mesh vertices and normals, and constrains them through color (opacity) contributions and geometric fitting (scale and distribution).

For a mesh with � triangular faces, each face $\mathcal { T } _ { f }$ is defined by three vertices $\{ \mathbf { v } _ { f , 1 } , \mathbf { v } _ { f , 2 } , \mathbf { v } _ { f , 3 } \} \subset \mathbb { R } ^ { 3 }$ . We place � Gaussians on each face. The center m $f , k \in \mathbb { R } ^ { 3 }$ of the �-th Gaussian on face $f$ is expressed as a convex combination of the face vertices:

$$
\mathbf { m } _ { f , k } = \sum _ { i = 1 } ^ { 3 } \alpha _ { f , k , i } \mathbf { v } _ { f , i } ,\tag{1}
$$

where the barycentric weights $\alpha _ { f , k , i }$ are learnable parameters satisfying $\alpha _ { f , k , i } \geq 0$ and $\begin{array} { r } { \sum _ { i = 1 } ^ { 3 } \alpha _ { f , k , i } = 1 } \end{array}$

The covariance matrix of each Gaussian on face $\mathcal { T } _ { f }$ is parameterized as $\boldsymbol { \Sigma } _ { f , k } = \mathbf { R } _ { f } \mathbf { S } _ { f , k } \mathbf { S } _ { f , k } ^ { \top } \mathbf { R } _ { f } ^ { \top }$ . The rotation matrix $\dot { \bf R } _ { f } = [ { \bf r } _ { 0 } , { \bf r } _ { 1 } , { \bf r } _ { 2 } ]$ defines an orthonormal frame, where $\mathbf { r } _ { 0 }$ is the unit normal of ${ \mathcal { T } } _ { f } :$ $\mathbf { r } _ { 1 }$ is a normalized edge direction of the face, and $\mathbf { r } _ { 2 }$ is obtained by orthogonalizing the remaining edge direction with respect to $\mathbf { r } _ { 0 }$ and $\mathbf { r } _ { 1 } .$ The scaling matrix $\mathbf { S } _ { f , k } = \mathrm { d i a g } ( s _ { 0 } , s _ { 1 } , s _ { 2 } )$ uses a small normal scale $s _ { 0 } = \varepsilon$ to flatten the Gaussian, and two tangential scales defined by:

$$
\begin{array} { r } { s _ { 1 } = \frac 1 2 \| \mathbf { v } _ { f , 2 } - \mathbf { v } _ { f , 1 } \| , \qquad s _ { 2 } = \frac 1 2 \| \mathbf { v } _ { f , 3 } - \mathbf { v } _ { f , 1 } \| . } \end{array}\tag{2}
$$

The factor $1 / 2$ provides a conservative initialization, preventing overly large Gaussians and excessive overlap during early optimization.

Scale Constraint: To ensure that the typical spatial extent of each Gaussian remains within the face-aligned minimum enclosing ellipse (MEE), we formulate the constraint as a spectral condition in the normalized space. For face ${ \mathcal { T } } _ { f } ,$ , the MEE is defined by semiaxes $s _ { 1 } , s _ { 2 }$ in the tangent frame. Let $D = \mathrm { d i a g } ( s _ { 1 } , s _ { 2 } )$ . The Gaussian covariance projected to the tangent plane is denoted as $\Sigma _ { k }$ . We normalize the covariance as:

$$
\begin{array} { r } { \tilde { \Sigma } _ { k } = D ^ { - 1 } \Sigma _ { k } D ^ { - 1 } . } \end{array}\tag{3}
$$

The Gaussian is fully contained [Calbert et al. 2023] in the MEE only if:

$$
\begin{array} { r } { \lambda _ { \operatorname* { m a x } } ( \tilde { \Sigma } _ { k } ) \leq 1 , } \end{array}\tag{4}
$$

where $\lambda _ { \operatorname* { m a x } } ( )$ denotes the largest eigenvalue of a matrix, corresponding to the maximum directional variance of the Gaussian. We define the MEE constraint loss as:

$$
\mathcal { L } _ { \mathrm { M E E } } = \operatorname* { m a x } \left( 0 , \lambda _ { \mathrm { m a x } } ( \tilde { \Sigma } _ { k } ) - 1 \right) .\tag{5}
$$

Distribution Constraint: To avoid degenerate Gaussian placements and encourage stable coverage over each triangular face, we introduce a distribution regularization on Gaussian centers in barycentric space.

I) Boundary avoidance loss $\mathcal { L } _ { b }$ discourages Gaussians from collapsing toward triangle edges:

$$
\mathcal { L } _ { b } = \frac { 1 } { { \cal F } \cdot { \cal K } \cdot 3 } \sum _ { f = 1 } ^ { F } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { 3 } \exp \bigl ( - 1 0 \alpha _ { f , k , i } \bigr ) .\tag{6}
$$

The exponential form rapidly increases the penalty near triangle boundaries.

II) Distance constraint loss $\mathcal { L } _ { d }$ prevents excessive clustering between neighboring Gaussians by encouraging a target spacing $\delta _ { \mathrm { t a r g e t } } =$ $1 / { \sqrt { K } }$ , which is empirically chosen to be proportional to the Gaussian density:

$$
\mathcal { L } _ { d } = \frac { 1 } { F \cdot K } \sum _ { f = 1 } ^ { F } \sum _ { k = 1 } ^ { K } \Bigl | \delta _ { f , k } - \delta _ { \mathrm { t a r g e t } } \Bigr | ,\tag{7}
$$

where the minimum distance for the Gaussian � is

$$
\delta _ { f , k } = \operatorname* { m i n } _ { j \neq k } \sqrt { \sum _ { i = 1 } ^ { 3 } \left( \alpha _ { f , k , i } - \alpha _ { f , j , i } \right) ^ { 2 } } .\tag{8}
$$

III) Coverage loss $\mathcal { L } _ { c }$ promotes that the Gaussians on a face span the full barycentric domain, the per-coordinate range covers most of [0, 1]. Let $r _ { f , i } =$ max<sub>�</sub> $\ : \alpha _ { f , k , i } \ : - \ :$ min<sub>�</sub> $\alpha _ { f , k , i }$ be the range of the barycentric coordinate for vertex � on face $f .$ We penalize cases where this range falls below a threshold $\tau = 0 . 8$ :

$$
\mathcal { L } _ { c } = \frac { 1 } { F \cdot 3 } \sum _ { f = 1 } ^ { F } \sum _ { i = 1 } ^ { 3 } \operatorname* { m a x } \bigl ( 0 , \ \tau - r _ { f , i } \bigr ) .\tag{9}
$$

The final loss $\mathcal { L } _ { \mathrm { D i s t } }$ is defined as

$$
\mathcal { L } _ { \mathrm { D i s t } } = \lambda _ { b } \mathcal { L } _ { b } + \lambda _ { d } \mathcal { L } _ { d } + \lambda _ { c } \mathcal { L } _ { c } ,\tag{10}
$$

where $\lambda _ { b } , \lambda _ { d } ,$ , and $\lambda _ { c }$ are balancing weights that control the contribution of each loss term.

Opacity Constraint: To maintain stable rendering supervision and prevent erroneous geometry from being hidden by low-opacity Gaussians, we remove the gradient of opacity and fix it close to 1:

$$
O _ { i } = \sigma ^ { - 1 } ( \theta ) , \quad \theta \to 1 ,\tag{11}
$$

where $\sigma ^ { - 1 } ( \cdot )$ is the inverse sigmoid function, $O _ { i }$ is the opacity parameter of the �-th Gaussian, and � is a constant close to 1.

## 3.2 Resolution Control

During initialization, we voxelize the scene at the target resolution and evaluate the SDF field. Since object surfaces occupy only a small portion of the volume, uniformly high-resolution optimization leads to substantial computational overhead. We therefore adopt an octree-like adaptive subdivision strategy [Frisken et al. 2000], using coarse voxels in empty regions and subdividing surface-related voxels into finer sub-voxels for detail reconstruction (Fig. 3). To preserve correct dual-vertex computation, adjacent voxels are subdivided consistently. We further support local refinement within user-specified bounding regions to reduce early-stage subdivision overhead. Additional details on refinement region selection and vertex remapping are provided in the supplementary materials.

![](images/798fcbad733e92e08680ce653ef21b0b87530c39f98daee68c22e582b7aed058.jpg)  
Fig. 3. Use an adaptive grid refinement method to control resolution. Left: 3D view of surface-only refinement. (a) Voxel subdivision: Targeted refinement of surface-intersecting voxels via SDF checks. (b) Refined SDF: Low-res (top) vs. high-res (botom) fields. (c) Meshes: 643 (top) vs. 1283 (botom) reconstructions.

## 3.3 Training

To prevent unreliable gradients from contaminating geometry optimization, we formulate training as a bi-level optimization problem. The outer loop optimizes the surface geometry, while the inner loop only optimizes a Gaussian scene conditioned on the current surface. At each outer iteration, we initialize a Gaussian scene � from the surface as described in Section 3.1, which remains gradientconnected to the surface parameters. We then construct a detached copy $\tilde { G }$ for inner-loop optimization. The inner loop trains $\tilde { G }$ using several steps of gradient descent under the same photometric reconstruction objective, allowing the Gaussian representation to adapt to the current surface geometry. The photometric loss [Wang et al. 2004] is defined as:

$$
\mathcal { L } _ { \mathrm { R G B } } = ( 1 - \lambda ) \mathcal { L } _ { 1 } + \lambda \mathcal { L } _ { \mathrm { D - S S I M } } ,\tag{12}
$$

computed between rendered and ground-truth images.

After inner optimization, we obtain $\tilde { G } ^ { * }$ , whose parameters are copied back to the gradient-connected Gaussian scene �. The outer loop then evaluates the same photometric loss on �, enabling gradients to update the surface geometry. To stabilize early optimization, we additionally apply a silhouette mask loss before switching to photometric supervision.

$$
\tilde { G } ^ { * } = \arg \operatorname* { m i n } _ { \tilde { G } } \mathcal { L } _ { \mathrm { R G B } } ( S , \tilde { G } ) ,\tag{13}
$$

$$
S ^ { * } = \arg \operatorname* { m i n } _ { S } \mathcal { L } _ { \mathrm { R G B } } ( S , G ) ,\tag{14}
$$

where the inner loop optimizes $\tilde { G }$ under fixed surface geometry $S ,$ and the outer loop updates � using the synchronized Gaussian scene �. For implementation details of the training framework and gradient isolation, please refer to the supplementary materials.

## 4 Experiment

## 4.1 Experiment Setings

Details: We employ an MLP network with 256 hidden units and 8 fully connected layers as the implicit representation for the SDF, incorporating skip connections, positional encoding, geometric initialization, and weight normalization. The learning rate for the SDF network is set to 0.0001. The inner Gaussian training follows the default settings of 3DGS [Kerbl et al. 2023], with 1000 iterations per internal step. All experiments are conducted on an RTX 3090 GPU with 24 GB of VRAM.

Training time. The initial surface optimization converges within a few minutes, while most computational cost comes from inner-loop Gaussian refinement. The training time for a single scene typically ranges from 6 to 18 hours, depending on scene complexity and number of optimization iterations.

Baselines: We select three categories of baselines for comparison: (I) NeRF and its SDF-based extensions [Mildenhall et al. 2021; Wang et al. 2021; Yariv et al. 2021], (II) 3D Gaussian Splatting (3DGS)–based reconstruction methods [Chen et al. 2024; Gao et al. 2024; Guédon and Lepetit 2024; Huang et al. 2024; Yu et al. 2024b], and (III) joint approaches where 3DGS is used to guide SDF optimization [Lyu et al. 2024; Yu et al. 2024a]. This selection enables a systematic evaluation of neural surface and Gaussian-based methods in isolation, and existing strategies for combining the two representations.

Datasets and Evaluation Metrics: We adopt the NeRF Synthetic [Mildenhall et al. 2021] and OmniObject3D [Wu et al. 2023] datasets as experimental benchmarks, as they provide complete ground-truth surface data for evaluating reconstruction performance in terms of noise and surface completeness. For NeRF Synthetic, we select five synthetic object scenes, each scene containing 100 training images and 200 test images, with a spatial resolution of 800 × 800 pixels per image. The OmniObject3D dataset comprises a large collection of real-world scanned objects and has been adopted in recent work, such as DiMeR [Jiang et al. 2025] and SIMECO [Wang et al. 2025], to evaluate the quality of geometric reconstruction. Following the original partitioning scheme, we select 12 representative objects from the three dificulty levels (easy, medium, and hard) for evaluation. Each object provides training images at a resolution of 800 × 800.

We evaluate the quality of the generated geometry using Chamfer Distance (CD). In particular, we sample from the complete mesh during evaluation, without neglecting challenging regions such as those with limited viewpoints. Additionally, we statistically analyze the triangle mesh quality, including maximum angle, minimum angle, radius ratio, aspect ratio, and the percentage of sliver triangles. We refer to the evaluation scripts and metrics provided by GOF [Yu et al. 2024b] and FlexiCubes [Shen et al. 2023].

## 4.2 OmniObject3D Dataset

We utilize real scanned objects from the OmniObject3D [Wu et al. 2023] dataset to evaluate the overall performance of our method, including reconstruction accuracy and geometric completeness.

Reconstruction Accuracy: The CD in Table 1 indicates that our method achieves the best performance across the vast majority of scenes. Figure 8 visualizes the results. NeRF-based methods often fail to produce clean surfaces, especially for large volumetric objects. Floaters appear inside and outside the object during extraction, e.g., in walnut or pumpkin hollows (see blue rectangle). Notably, NeuS and GSDF (built on the NeuS framework) are unstable during

Table 1. The 12 objects are shown from left to right in groups of four, following the dataset’s native dificulty grouping with increasing dificulty. Our method accurately and robustly reconstructs all objects, while others fail or have large errors on a few objects. “—” denotes reconstruction failure.
<table><tr><td> $\mathrm { C D } \left( 1 0 ^ { - 3 } \right) \downarrow$ </td><td>Banana</td><td>Dumpling</td><td>Walnut</td><td>Peanut</td><td>Teapot</td><td>Toy_plane</td><td>Handbag</td><td>Pumpkin</td><td>Toy_train</td><td>Durian</td><td>Kennel</td><td>Pan</td><td>Mean</td></tr><tr><td>NeRF</td><td>38.22</td><td>68.54</td><td>252.81</td><td>69.43</td><td>65.96</td><td>13.76</td><td>70.24</td><td>239.94</td><td>46.75</td><td>25.51</td><td>13.36</td><td>13.20</td><td>76.48</td></tr><tr><td>NeuS</td><td>5.95</td><td>27.70</td><td>51.02</td><td>12.68</td><td>29.13</td><td>5.51</td><td></td><td>44.05</td><td>8.87</td><td>19.24</td><td>7.12</td><td>11.07</td><td>20.21</td></tr><tr><td>SuGar</td><td>74.18</td><td>67.16</td><td>17.99</td><td>21.33</td><td>75.95</td><td>95.78</td><td>19.33</td><td>58.50</td><td>80.26</td><td>111.20</td><td>36.90</td><td>31.28</td><td>57.49</td></tr><tr><td>2DGS</td><td>11.70</td><td>6.88</td><td>52.37</td><td>29.26</td><td>36.75</td><td>15.63</td><td>36.63</td><td>71.45</td><td>29.96</td><td>25.05</td><td>27.55</td><td>9.18</td><td>29.37</td></tr><tr><td>PGSR</td><td>7.24</td><td>15.04</td><td>8.62</td><td>10.22</td><td>17.02</td><td>15.67</td><td>11.67</td><td>30.87</td><td>11.77</td><td>65.99</td><td>13.49</td><td>9.15</td><td>18.06</td></tr><tr><td>GOF</td><td>9.10</td><td>4.95</td><td>8.31</td><td>6.24</td><td>6.39</td><td>7.48</td><td>7.41</td><td>72.17</td><td>12.35</td><td>22.37</td><td>15.52</td><td>6.60</td><td>14.91</td></tr><tr><td>GSDF</td><td>1.60</td><td>6.64</td><td>4.72</td><td>6.76</td><td>8.14</td><td></td><td></td><td>9.71</td><td>10.56</td><td>24.77</td><td>59.18</td><td></td><td>14.68</td></tr><tr><td>Ours</td><td>4.78</td><td>11.66</td><td>6.17</td><td>7.35</td><td>6.10</td><td>7.83</td><td>14.38</td><td>8.73</td><td>8.54</td><td>21.71</td><td>5.90</td><td>5.95</td><td>9.09</td></tr></table>

Table 2. Quantitative results on the NeRF Synthetic dataset. For resolutioncontrollable methods, we test two resolutions: low (128 or original lowest) and high (method default), with resolution marked by method subscript.
<table><tr><td>CD (10−2) ↓</td><td>Method</td><td>Chair</td><td>Hotdog</td><td>Lego</td><td>Drums</td><td>Mic</td><td>Mean</td></tr><tr><td rowspan="2">Traditional</td><td>Nerf128</td><td>3.62</td><td>4.19</td><td>3.07</td><td>6.91</td><td>8.31</td><td>5.22</td></tr><tr><td>Nerf512</td><td>2.16</td><td>2.04</td><td>1.85</td><td>1.89</td><td>1.24</td><td>1.84</td></tr><tr><td rowspan="3">SDF-NeRF</td><td>VolSDF</td><td>1.18</td><td>3.22</td><td>2.26</td><td>4.03</td><td>1.14</td><td>2.37</td></tr><tr><td>NeuS128</td><td>1.34</td><td>2.12</td><td>2.56</td><td>4.10</td><td>1.01</td><td>2.23</td></tr><tr><td>NeuS512</td><td>1.62</td><td>2.02</td><td>1.58</td><td>3.79</td><td>0.78</td><td>1.96</td></tr><tr><td rowspan="7">GS-based</td><td>RelightableGaussian</td><td>3.65</td><td>3.11</td><td>1.63</td><td>2.34</td><td>1.13</td><td>2.37</td></tr><tr><td>SuGarLow</td><td>2.37</td><td>4.01</td><td>1.46</td><td>6.21</td><td>4.80</td><td>3.77</td></tr><tr><td>SuGarHigh</td><td>1.74</td><td>3.18</td><td>1.62</td><td>4.86</td><td>4.47</td><td>3.17</td></tr><tr><td>2DGS128</td><td>8.07</td><td>4.37</td><td>4.18</td><td>5.75</td><td>4.92</td><td>5.46</td></tr><tr><td>2DGS1024</td><td>1.71</td><td>2.87</td><td>4.18</td><td>4.01</td><td>2.51</td><td>3.06</td></tr><tr><td>PGSR</td><td>1.76</td><td>3.14</td><td>1.99</td><td>1.54</td><td>1.17</td><td>1.92</td></tr><tr><td>GOF</td><td>2.51</td><td>3.63</td><td>2.22</td><td>1.42</td><td>0.99</td><td>2.15</td></tr><tr><td rowspan="4">GS-guided SDF</td><td>GSDF128</td><td>1.78</td><td>2.45</td><td>2.74</td><td>4.11</td><td>7.35</td><td>3.69</td></tr><tr><td>GSDF512</td><td>1.62</td><td>3.08</td><td>1.58</td><td>3.79</td><td>7.13</td><td>3.44</td></tr><tr><td>3DGSR</td><td>1.01</td><td>1.48</td><td>1.45</td><td>0.95</td><td>1.15</td><td>1.21</td></tr><tr><td>Ours128</td><td>0.71</td><td>1.76</td><td>1.29</td><td>0.93</td><td>1.12</td><td>1.16</td></tr></table>

SDF training, leading to reconstruction failures for a few objects. Gaussian-based methods produce fewer floaters but exhibit surface incompleteness, primarily because they struggle to handle object surfaces from limited viewpoints or with low-quality Gaussians. Although GOF [Yu et al. 2024b] achieves a lower CD, it produces cluttered triangle meshes. For methods with large errors, such as SuGar [Guédon and Lepetit 2024] and 2DGS [Huang et al. 2024], the reconstructed meshes exhibit numerous outliers scattered around the main model, resulting in incorrect geometry. This is due to their suboptimal handling of datasets with a single object and masked backgrounds.

Geometric Completeness: Figure 8 also shows that our method preserves geometric continuity and completeness on the OmniObject3D dataset, especially in regions with missing viewpoints. This is because optimizing a continuous field enables indirect supervision of unobserved regions via images from other viewpoints. TSDF fusion fails to reconstruct geometry in areas with missing views; GOF merely fills the voids, but it struggles to follow the original shape.

![](images/0819ccc5c9246a7896f606c6205e13483351001a393551c020271eee47a7e357.jpg)  
Fig. 4. Mesh quality. We show magnified surface details for each surface in the direction of the arrow. Regions with geometric errors and sliver triangles are highlighted in red. Our method consistently achieves the best mesh quality as optimization progresses.

## 4.3 NeRF Synthetic Dataset

We utilize objects with diverse textures and fine structures from the NeRF Synthetic dataset for detailed evaluation. We focus on the performance of our method at diferent resolutions and the quality of the triangulated surfaces.

Diferent Resolutions: As shown in Table 2, our method achieves the best average CD among all baselines. Compared with other meth ods, our approach consistently reconstructs high-quality surfaces even at lower resolutions. At resolutions below 128, NeRF introduces significant noise due to the dificulty of threshold selection for surface extraction, and Gaussian-based methods fail to remove redundant surfaces, thereby hindering the recovery of fine geometric details. For a few objects, our reconstruction error is slightly higher, primarily due to overly fine structures or surface reflections(e.g., drum hardware and mic grille.) that hinder the field modeling and isosurface extraction process.

![](images/98bbbadfbd807e21c1e7eb1702f218613245c74a43a62407a1477f0f71ce6f0a.jpg)  
Fig. 5. Without the opacity constraint, low-opacity Gaussians accumulate around the surface, producing layered floating artifacts and bloated geometry inconsistent with the rendered appearance.

![](images/fd5f0e35f959482aec50d0aa4902cb0068053b9437c5cbfbffa17cea6b1412dd.jpg)  
Fig. 6. Comparison of the efects of the scale constraint (left) and distribution constraint (right) on Chair, Lego, and Materials. Without the scale constraint, the reconstruction exhibits missing geometry. Without the distribution constraint, the surface contains irregular geometric fluctuations.

Table 3. Ablation Studies on the NeRF Synthetic Dataset.
<table><tr><td>Settings</td><td>Opacity</td><td>Scale</td><td>Distribution</td><td>CD↓</td></tr><tr><td>Baseline</td><td></td><td></td><td></td><td>11.40</td></tr><tr><td>Only Opacity</td><td>√</td><td></td><td></td><td>2.87</td></tr><tr><td>w/o Scale</td><td>√</td><td></td><td>√</td><td>1.58</td></tr><tr><td>w/o Distribution</td><td>√</td><td>√</td><td></td><td>1.36</td></tr><tr><td>Full Model</td><td>√</td><td>√</td><td>√</td><td>1.16</td></tr></table>

Mesh Quality: Figure 4 and Figure 9 illustrate the qualitative and quantitative comparison results of triangle mesh quality. Results show that GOF [Yu et al. 2024b] and SuGar [Guédon and Lepetit 2024] produce cluttered, irregular triangle distributions, 2DGS [Huang et al. 2024] and PGSR [Chen et al. 2024] over-concentrate on right triangles and lack geometric adaptability. NeuS [Wang et al. 2021] and GSDF [Yu et al. 2024a] are prone to unstable holes and bumps. Our method generates meshes with a drastically lower fraction of sliver triangles and more regular shapes compared to all baseline approaches. The shape of triangular faces adheres more closely to the equilateral principle of Delaunay triangulation [Cheng et al. 2013].

![](images/aae6b74051f5efe8c0109edff54960b6ae7565dd039c76e6625c1419fafedb01.jpg)  
Fig. 7. (a) Neural implicit SDF yields smoother and more continuous surfaces, while discrete SDFs tend to produce noise. (b) Starting from the same initialization, training without gradient isolation leads to mutual interference between surface and Gaussian optimization, resulting in degraded surface quality.

## 4.4 Ablation Study

Gaussian Constraints: We validate the efectiveness of our Surface Gaussian Constraints on the NeRF Synthetic dataset. The results are summarized in Table 3.

I) Opacity. In Sec. 3.1, we apply an opacity constraint to fix Gaussian opacity to its maximum value. As shown in Fig. 5, we show that this constraint is critical for stable geometry optimization. Without it, erroneous regions can be represented by low-opacity Gaussians that contribute little to the rendered image. As a result, these regions receive insuficient supervision during geometric refinement and persist as floating or unstable surface artifacts.

II) Scale and Distribution. Since scale and distribution constraints provide limited benefit on severely corrupted geometry (e.g., the Trainable Opacity mesh in Fig. 5), we evaluate these two constraints with the opacity constraint enabled. As shown in Fig. 6, removing these constraints degrades reconstruction accuracy. This is because the spatial coverage of Gaussians no longer aligns with the under lying surface geometry. Over-expanded or incomplete Gaussian regions produce incorrect rendering supervision, leading to inaccurate surface optimization and degraded mesh quality.

Optimization: We qualitatively verified the improvements we made to the optimization process, and the results are summarized in Fig. 7.

I) Neural ImplicitSDF. The original diferentiable extraction method directly optimizes discrete SDF values on voxel vertices. In contrast, we parameterize the SDF with an implicit neural network. As shown in Fig. 7, the neural implicit SDF produces smoother surfaces with fewer noise.

II) Gradient isolation. In Section 3.3, we employ a gradient detachment strategy, where a detached copy of the Gaussians is optimized independently from the SDF. As shown in Fig. 7, this prevents gradient contamination and avoids propagating erroneous supervisory signals during training. .

## 5 Limitations

Our method achieves strong surface quality and geometric completeness, but several limitations remain. First, the bi-level optimization leads to long reconstruction time and heavy GPU memory usage, restricting our framework to object-level scenes. Second, Gaussian-based rendering remains brittle on reflective materials and fine-scale textures where geometry cues are ambiguous. Future work will prioritize reducing runtime and GPU memory overhead by retaining Gaussians that require no re-optimization. Additionally, rendering supervision can be improved by incorporating more expressive Gaussian formulations and stronger geometric constraints to better preserve detailed surface structures.

## 6 Conclusion

We introduced Gaussian Sculpting, a fully diferentiable framework for end-to-end surface reconstruction via SDF optimization and constrained Gaussian rendering. Our method establishes stronger consistency between rendered views and surface geometry through Gaussian constraints and bi-level optimization, enabling stable geometry refinement and reducing artifacts such as floaters, missing regions, and geometric inconsistencies. Experimental results demon strate that our framework produces complete and more topologi cally regular meshes than prior NeRF-based and Gaussian-based reconstruction approaches across diverse reconstruction scenarios.

## References

Michael Bleyer, Christoph Rhemann, and Carsten Rother. 2011. PatchMatch Stereo - Stereo Matching with Slanted Support Windows. In British Machine Vision Conference. https://api.semanticscholar.org/CorpusID:1798946

Julien Calbert, Lucas N. Egidio, and Raphaël M. Jungers. 2023. An Eficient Method to Verify the Inclusion of Ellipsoids. IFAC-PapersOnLine 56, 2 (2023), 1958–1963. doi:10.1016/j.ifacol.2023.10.1088 22nd IFAC World Congress.

Neill DF Campbell, George Vogiatzis, Carlos Hernández, and Roberto Cipolla. 2008. Using multiple hypotheses to improve depth-maps for multi-view stereo. In European conference on computer vision. Springer, 766–779.

Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, and Guofeng Zhang. 2024. Pgsr: Planar-based gaussian splatting for eficient and high-fidelity surface reconstruction. IEEE Transactions on Visualization and Computer Graphics (2024).

Yiwen Chen, Tong He, Di Huang, Weicai Ye, Sijin Chen, Jiaxiang Tang, Zhongang Cai, Lei Yang, Gang Yu, Guosheng Lin, and Chi Zhang. 2025. MeshAnything: Artist Created Mesh Generation with Autoregressive Transformers. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum? id=KGZAs8VcOM

Siu-Wing Cheng, Tamal K. Dey, Herbert Edelsbrunner, Michael A. Facello, and Shang-Hua Teng. 2000. Sliver exudation. J. ACM 47, 5 (Sept. 2000), 883–904. doi:10.1145/ 355483.355487

Siu-Wing Cheng, Tamal Krishna Dey, Jonathan Shewchuk, and Sartaj Sahni. 2013. Delaunay mesh generation. CRC Press Boca Raton.

Jaehoon Choi, Yonghan Lee, Hyungtae Lee, Heesung Kwon, and Dinesh Manocha. 2024. Meshgs: Adaptive mesh-aligned gaussian splatting for high-quality rendering. In Proceedings ofthe Asian Conference on Computer Vision. 3310–3326.

Christopher B Choy, Danfei Xu, JunYoung Gwak, Kevin Chen, and Silvio Savarese. 2016. 3d-r2n2: A unified approach for single and multi-view 3d object reconstruction. In European conference on computer vision. Springer, 628–644.

Nianchen Deng, Zhenyi He, Jiannan Ye, Budmonde Duinkharjav, Praneeth Chakravarthula, Xubo Yang, and Qi Sun. 2022. Fov-nerf: Foveated neural radi ance fields for virtual reality. IEEE Transactions on Visualization and Computer Graphics 28, 11 (2022), 3854–3864.

Haoqiang Fan, Hao Su, and Leonidas J Guibas. 2017. A point set generation network for 3d object reconstruction from a single image. In Proceedings ofthe IEEE conference on computer vision and pattern recognition (CVPR). 605–613

Qingyu Fan, Yinghao Cai, Chao Li, Wenzhe He, Xudong Zheng, Tao Lu, Bin Liang, and Shuo Wang. 2025. NeuGrasp: Generalizable Neural Surface Reconstruction with Background Priors for Material-Agnostic Object Grasp Detection. In 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 3197–3203.

Sarah F Frisken, Ronald N Perry, Alyn P Rockwood, and Thouis RJones. 2000. Adaptively sampled distance fields: A general representation of shape for computer graphics. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques. 249–254.

Yasutaka Furukawa and Jean Ponce. 2010. Accurate, Dense, and Robust Multiview Stereopsis. IEEE Transactions on Pattern Analysis and Machine Intelligence 32, 8 (2010), 1362–1376. doi:10.1109/TPAMI.2009.161

Jian Gao, Chun Gu, Youtian Lin, Zhihao Li, Hao Zhu, Xun Cao, Li Zhang, and Yao Yao. 2024. Relightable 3d gaussians: Realistic point cloud relighting with brdf

decomposition and ray tracing. In European Conference on Computer Vision. Springer, 73–89.

Xiangjun Gao, Xiaoyu Li, Yiyu Zhuang, Qi Zhang, Wenbo Hu, Chaopeng Zhang, Yao Yao, Ying Shan, and Long Quan. 2025. Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 21392–21402.

Antoine Guédon and Vincent Lepetit. 2024. Sugar: Surface-aligned gaussian splatting for eficient 3d mesh reconstruction and high-quality mesh rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 5354–5363.

Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2024. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers. 1–11.

Shahram Izadi, Richard A Newcombe, David Kim, Otmar Hilliges, David Molyneaux, Steve Hodges, Pushmeet Kohli, Jamie Shotton, Andrew J Davison, and Andrew Fitzgibbon. 2011. Kinectfusion: real-time dynamic 3d surface reconstruction and interaction. In ACM SIGGRAPH 2011 Talks. 1–1.

Lutao Jiang, Jiantao Lin, Kanghao Chen, Wenhang Ge, Xin Yang, Yifan Jiang, Yuanhuiyi Lyu, Xu Zheng, and Ying-Cong Chen. 2025. DiMeR: Disentangled Mesh Reconstruc tion Model. ArXiv abs/2504.17670 (2025). https://api.semanticscholar.org/CorpusID: 278032925

Michael Kazhdan, Matthew Bolitho, and Hugues Hoppe. 2006. Poisson surface reconstruction. In Proceedings of the fourth Eurographics symposium on Geometry processing, Vol. 7.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023. 3D Gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42, 4 (2023), 139–1.

Jonghyun Kim, Michael Stengel, Amrita Mazumdar, Tianye Li, Cheng Sun, David Luebke, and Shalini De Mello. 2025. Play4D: Accelerated and Interactive Free-viewpoint Video Streaming for Virtual Reality and Light Field Displays. In Proceedings ofthe SIGGRAPH Asia 2025 Emerging Technologies. 1–3.

Kiriakos N Kutulakos and Steven M Seitz. 2000. A theory of shape by space carving. International journal of computer vision 38, 3 (2000), 199–218.

Hai Li, Xingrui Yang, Hongjia Zhai, Yuqian Liu, Hujun Bao, and Guofeng Zhang. 2022. Vox-surf: Voxel-based implicit surface representation. IEEE Transactions on Visualization and Computer Graphics 30, 3 (2022), 1743–1755.

Yuan Li, Cheng Lin, Yuan Liu, Xiaoxiao Long, Chenxu Zhang, Ningna Wang, Xin Li, Wenping Wang, and Xiaohu Guo. 2025. CADDreamer: CAD Object Generation from Single-View Images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 21448–21457.

Zhaoshuo Li, Thomas Müller, Alex Evans, Russell H. Taylor, Mathias Unberath, Ming-Yu Liu, and Chen-Hsuan Lin. 2023. Neuralangelo: High-Fidelity Neural Surface Reconstruction. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 8456–8465.

Chen-Hsuan Lin, Chen Kong, and Simon Lucey. 2018. Learning Eficient Point Cloud Generation for Dense 3D Object Reconstruction. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 32.

William E Lorensen and Harvey E Cline. 1998. Marching cubes: A high resolution 3D surface construction algorithm. In Seminal graphics: pioneering eforts that shaped the field. 347–353.

Yuanxun Lu, Jingyang Zhang, Tian Fang, Jean-Daniel Nahmias, Yanghai Tsin, Long Quan, Xun Cao, Yao Yao, and Shiwei Li. 2025. Matrix3D: Large Photogrammetry Model All-in-One. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 11250–11263.

Xiaoyang Lyu, Yang-Tian Sun, Yi-Hua Huang, Xiuzhe Wu, Ziyi Yang, Yilun Chen, Jiangmiao Pang, and Xiaojuan Qi. 2024. 3DGSR: Implicit Surface Reconstruction with 3D Gaussian Splatting. ACM Trans. Graph. 43, 6 (2024), 1–12.

Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. 2019. Occupancy Networks: Learning 3D Reconstruction in Function Space. In Proceedings ofthe IEEE/CVFConference on Computer Vision and Pattern Recognition (CVPR). 4460–4470.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.

Shu Pan, Ziyang Hong, Zhangrui Hu, Xiandong Xu, Wenjie Lu, and Liang Hu. 2025. Russo: Robust underwater slam with sonar optimization against visual degradation. IEEE/ASME Transactions on Mechatronics (2025).

Anastasiya Pechko, Piotr Borycki, Joanna Waczyńska, Daniel Barczyk, Agata Szymańska, Sławomir Tadeja, and Przemysław Spurek. 2025. GS-Verse: Mesh-based Gaussian Splatting for Physics-aware Interaction in Virtual Reality. arXiv preprint arXiv:2510.11878 (2025).

Tianchang Shen, Jacob Munkberg, Jon Hasselgren, Kangxue Yin, Zian Wang, Wenzheng Chen, Zan Gojcic, Sanja Fidler, Nicholas Sharp, and Jun Gao. 2023. Flexible Isosurface Extraction for Gradient-Based Mesh Optimization. ACM Trans. Graph. 42, 4 (2023), 1–16.

Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. 2024. DreamGaussian: Generative Gaussian Splatting for Eficient 3D Content Creation. In International Conference on Learning Representations, B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (Eds.), Vol. 2024. 33879–33896. https://proceedings.iclr.cc/paper\_files/paper/2024/file/ 905202e21386913d8eac637c2b50f590-Paper-Conference.pdf

Joanna Waczyńska, Piotr Borycki, Sławomir Tadeja, Jacek Tabor, and Przemysław Spurek. 2024. Games: Mesh-based adapting and modification of gaussian splatting. arXiv preprint arXiv:2402.01459 (2024).

Jiepeng Wang, Yuan Liu, Peng Wang, Cheng Lin, Junhui Hou, Xin Li, Taku Komura, and Wenping Wang. 2024. GausSurf: Geometry-Guided 3D Gaussian Splatting for Surface Reconstruction. arXiv preprint arXiv:2411.19454 (2024).

Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. 2021. NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction. In Advances in Neural Information Processing Systems, M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (Eds.), Vol. 34. Curran Associates, Inc., 27171–27183. https://proceedings.neurips.cc/paper\_ files/paper/2021/file/e41e164f7485ec4a28741a2d0ea41c74-Paper.pdf

Yuqing Wang, Zhaiyu Chen, and Xiaoxiang Zhu. 2025. Learning Generalizable Shape Completion with SIM(3) Equivariance. In Advances in Neural Information Processing Systems, D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (Eds.), Vol. 38. Curran Associates, Inc., 24300–24329. https://proceedings.neurips.cc/paper\_files/paper/2025/file/ 22f61e7079302f67f633ab1fc19bf8dc-Paper-Conference.pdf

Yiming Wang, Qin Han, Marc Habermann, Kostas Daniilidis, Christian Theobalt, and Lingjie Liu. 2023. NeuS2: Fast Learning of Neural Implicit Surfaces for Multi-View Reconstruction. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV). 3295–3306.

Zhou Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. Trans. Img. Proc. 13, 4 (April 2004), 600–612. doi:10.1109/TIP.2003.819861

Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren, Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian, et al. 2023. Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 803–814.

Haozhe Xie, Hongxun Yao, Xiaoshuai Sun, Shangchen Zhou, and Shengping Zhang. 2019. Pix2Vox: Context-Aware 3D Reconstruction from Single and Multi-View Images. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 2690–2698.

Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, and Ulrich Neumann. 2022. Point-nerf: Point-based neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR). 5438– 5448.

Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. 2021. Volume rendering of neural implicit surfaces. Advances in neural information processing systems 34 (2021), 4805–4815.

Lior Yariv, Peter Hedman, Christian Reiser, Dor Verbin, Pratul P Srinivasan, Richard Szeliski, Jonathan T Barron, and Ben Mildenhall. 2023. Bakedsdf: Meshing neural sdfs for real-time view synthesis. In ACM SIGGRAPH 2023 conference proceedings. 1–9.

Weicai Ye, Hai Li, Tianxiang Zhang, Xiaowei Zhou, Hujun Bao, and Guofeng Zhang. 2021. SuperPlane: 3D plane detection and description from a single image. In 2021 IEEE Virtual Reality and 3D User Interfaces (VR). IEEE, 207–215.

Mulin Yu, Tao Lu, Linning Xu, Lihan Jiang, Yuanbo Xiangli, and Bo Dai. 2024a. Gsdf: 3dgs meets sdf for improved neural rendering and reconstruction. Advances in Neural Information Processing Systems 37 (2024), 129507–129530.

Zehao Yu, Torsten Sattler, and Andreas Geiger. 2024b. Gaussian Opacity Fields: Eficient Adaptive Surface Reconstruction in Unbounded Scenes. ACM Trans. Graph. 43, 6 (2024), 1–13.

Tianxiang Zhang, Chong Bao, Hongjia Zhai, Jiazhen Xia, Weicai Ye, and Guofeng Zhang. 2021. Arcargo: Multi-device integrated cargo loading management system with augmented reality. In 2021 IEEE Intl Confon Dependable, Autonomic and Secure Computing, Intl Confon Pervasive Intelligence and Computing, Intl Confon Cloud and Big Data Computing, Intl Confon Cyber Science and Technology Congress (DASC/PiCom/CBDCom/CyberSciTech). IEEE, 341–348.

Xinyun Zhang, Ruiqi Yu, and Shuang Ren. 2025. Neural Implicit Representations for Multi-View Surface Reconstruction: A Survey. IEEE Transactions on Visualization and Computer Graphics 31, 10 (2025), 9444–9463. doi:10.1109/TVCG.2025.3582627

Zuoliang Zhu, Beibei Wang, and Jian Yang. 2025. GS-ROR2: Bidirectional-Guided 3DGS and SDF for Reflective Object Relighting and Reconstruction. ACM Trans. Graph. 45, 1 (2025), 1–19.

PGSR

GT

Ke Jiaxin, Juncheng Liu, Yi Wang, Zhouhui Lian, Bin Liu, Shengfa Wang, and Xiangjia He

SuGaR

GOF

NeuS

GSDF

OURS

![](images/830d0bbc22744d4a3f7a3e5a8b44e7b1c128a5c0dea2198a3c9a8d56dacd0bab.jpg)

Fig. 8. Real scanned results on OmniObject3D. Interior (blue) and viewpoint-missing botom regions (red) are shown. Our method preserves geometric completeness while removing floaters.  
![](images/5bd86109974527a1f088c31a2b214a5cdf7bdf84ee96058f0f7af10a448339c1.jpg)  
Fig. 9. Quantitative comparison of the intrinsic quality of extracted meshes. Our method yields the lowest number of sliver triangles with interior angles below 10<sup>◦</sup> and more equilateral shapes. Smoother angle distributions show adaptation to geometric variations along curved boundaries and fine details. This result highlights the need for end-to-end surface reconstruction.