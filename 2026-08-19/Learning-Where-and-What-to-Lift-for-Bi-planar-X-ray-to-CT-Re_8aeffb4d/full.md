# Learning Where and What to Lift for Bi-planar X-ray-to-CT Reconstruction

Yifei Wu<sup>1</sup>, Yicheng Wu<sup>2</sup>, Qiang Ma<sup>3</sup>, Qi Chen<sup>4</sup>, Renyang Gu<sup>5</sup>, Xinyu Liu<sup>6</sup>, Yongsheng Pan<sup>1</sup>, Yong Xia<sup>1</sup>

<sup>1</sup>School of Computer Science, Northwestern Polytechnical University, Xi’an, China

<sup>2</sup>Imperial College London, London, UK

<sup>3</sup>Department of Brain Sciences, Imperial College London, London, UK

<sup>4</sup>Australian Institute for Machine Learning (AIML), Adelaide University, Adelaide, Australia

<sup>5</sup>Department of Bioengineering, Imperial College London, London, UK

<sup>6</sup>Department of Computing, Imperial College London, London, UK

## Abstract

X-ray imaging can be approximately modeled as the projection of an underlying volumetric attenuation field, with each measurement recording the accumulated attenuation along a corresponding ray path. Reconstructing a CT volume from only a few X-ray views is therefore severely ill-posed, as the projections collapse depth information and leave 3D locations of anatomical regions and their corresponding intensity distributions highly entangled and ambiguous. We observe that once the spatial organization of anatomical regions is established, estimating their CT intensities becomes substantially more tractable. Motivated by this, we propose LiftXR, an interleaved, geometry-guided framework that explicitly incorporates spatial layout recovery into CT reconstruction. Specifically, a layout lifter first generates a 3D anatomical layout from bi-planar X-rays, providing spatial guidance for an intensity renderer to reconstruct a CT volume. An anatomical parser then performs volumetric perception on the reconstruction, exploiting its spatially resolved boundary and intensity cues to recover a refined anatomical layout. This transition from projection-conditioned layout generation to reconstruction-conditioned anatomical perception allows the parsed layout to provide feedback for region-specific intensity calibration. Extensive experiments on two public datasets demonstrate that LiftXR consistently outperforms recent Xray-to-CT reconstruction methods, establishing a new state of the art. Moreover, the reconstructed CT achieves superior performance in external downstream segmentation, indicating improved anatomical fidelity. Code will be released.

## Introduction

Computed tomography (CT) reconstructs volumetric attenuation fields from multi-angle X-ray projections, typically requiring hundreds of views for accurate 3D imaging (Jung 2021). However, dense-view acquisition increases radiation exposure, limiting its applicability to longitudinal monitoring and large-scale screening (Cai et al. 2024). Sparse-view CT reconstruction reduces this burden by using only a few dozen projections (Lin et al. 2023, 2024; Shi, Pelt, and Batenburg 2026; Guo et al. 2024). bi-planar X-ray-to-CT reconstruction ofers an even more acquisition-eficient alternative, relying only on frontal and lateral radiographs that are routinely acquired in clinical practice and widely available in public datasets (Yu et al. 2025; Wang et al. 2017). Nevertheless, recovering a 3D CT volume from only two projections remains severely ill-posed. The ill-posedness originates from the severe loss of depth information during projection acquisition (Chen et al. 2025a; Wu et al. 2025). As illustrated in Fig. 1, each X-ray pixel records attenuation accumulated along a ray path, collapsing depth-dependent anatomical information into a two-dimensional measurement. We describe the underlying CT volume by an anatomical layout exhibiting the spatial extent and arrangement of organs and tissues, and a volumetric intensity field specifying their voxel-wise CT values. Because distinct 3D layouts and intensity distributions can produce similar 2D projections, the two aspects remain highly entangled under limited observations. Existing X-ray-to-CT methods (Ying et al. 2019; Kyung et al. 2023; Pan et al. 2025) have introduced geometric consistency, learnable priors, or anatomy-related regularization to facilitate CT reconstruction. However, anatomical organization is typically encoded implicitly within image features or inferred only through the reconstructed intensities, rather than being modeled as an explicit region-level 3D layout. As a result, reconstructed volumes may perform favorably on voxel-wise metrics while still exhibiting inaccurate regional extents, blurred boundaries, or inconsistent spatial relationships (Lin et al. 2025; Shi et al. 2025; Deo et al. 2025).

![](images/93475aee06761a1af6ce86edf447c486b6bc32242902f3fe220913bd1a4ba6ab.jpg)  
Figure 1: X-ray projection and CT reconstruction. The projected value p(u, v) records the accumulated attenuation µ(x) along its corresponding ray $r _ { u , v }$ . We characterize this process by an anatomical layout capturing the spatial organization of organs and tissues, and an intensity field specifying their voxel-wise CT values. Recovering both aspects from sparse X-ray views is highly ill-posed since geometric structure and intensity information are strongly entangled. Then, an oracle experiment on CT-RATE shows that reconstruction conditioned on ground-truth anatomical layouts achieves 37.04 dB PSNR and 95.07% SSIM, demonstrating the value of anatomical layout as an efective spatial constraint.

To further investigate the role of anatomical layout, we conduct an oracle study on CT-RATE (Hamamci et al. 2026), in which the reconstruction model is conditioned on ground-truth anatomical masks. The resulting reconstruction achieves a PSNR of 37.04 dB and an SSIM of 95.07%. This result demonstrates that the anatomical layout provides strong spatial constraints for CT reconstruction. By localizing major body regions in 3D space, the layout reduces the geometric uncertainty that would otherwise need to be resolved jointly with voxel-wise intensity estimation.

Motivated by these observations, we propose LiftXR, an end-to-end, geometry-guided framework that explicitly incorporates anatomical layout recovery into bi-planar X-rayto-CT reconstruction. Specifically, a layout lifter first generates an initial 3D anatomical layout from the bi-planar X-rays. This layout provides explicit spatial guidance for an intensity renderer to reconstruct a CT volume. Although imperfect, the reconstructed volume contains spatially resolved boundary and intensity cues, and a subsequent anatomical parser predicts region masks from the reconstructed CT to obtain a refined layout. Finally, an intensity calibrator uses the parsed layout to correct region-specific intensity inconsistencies. Importantly, the two anatomical layouts play fundamentally diferent roles. The lifted layout is inferred directly from bi-planar projections and serves as a generative geometric prior for volumetric reconstruction. In contrast, the parsed layout is obtained by perceiving anatomical regions from the initially reconstructed CT volume, constituting an explicit volumetric anatomical parsing task. This transition from layout generation to layout perception enables the reconstructed volume to provide feedback for geometry refinement, which subsequently guides region-specific intensity calibration.

Our contributions are threefold:

• We formulate the bi-planar X-ray-to-CT reconstruction task by explicitly modeling anatomical layout as a spatial constraint for volumetric generation. By specifying where anatomical regions are located in 3D space, the layout constrains and guides the estimation of what CT intensities they exhibit.

• We propose LiftXR, a geometry-guided framework that couples volumetric generation with anatomical perception. LiftXR integrates projection-conditioned layout generation, conditional CT rendering, reconstructionconditioned anatomical parsing, and region-specific intensity calibration, forming a unified generationperception feedback process.

• Extensive experiments on two public chest CT datasets demonstrate that LiftXR achieves state-of-the-art reconstruction performance. Its reconstructed CT volumes further yield superior downstream segmentation results, showcasing improved anatomical fidelity.

## Related Work

## X-ray-to-CT Reconstruction

X-ray-to-CT reconstruction aims to recover a 3D CT volume from one or two 2D X-ray projections, which is highly chal lenging due to the severe ambiguity introduced by the projection process. X2CT (Ying et al. 2019) first demonstrated the feasibility of reconstructing volumetric CT from bi-planar X-rays, and subsequent studies introduced geometric, generative, and structural priors to reduce this ambiguity. PerX2CT (Kyung et al. 2023) incorporates perspective projection modeling to improve spatial localization, DifuX2CT (Liu et al. 2024a) introduces a conditional difusion prior to improve volumetric consistency, and DSDF (Pan et al. 2025) separates global structure modeling from local texture synthesis to enhance reconstruction quality. However, these methods mainly optimize voxel-level appearance or represent anatomical structure implicitly, leaving explicit constraints on organ boundaries, spatial extents, and inter-organ relationships underexplored (Lin et al. 2025). LiftXR addresses this limitation by explicitly predicting anatomical structures, where predicted layouts guide CT estimation and reconstructed volumes provide cues for further layout parsing. This interleaved layout-intensity interaction constrains the solution space and promotes anatomically consistent reconstruction with clear anatomical constraints derived from bi-planar X-rays.

## Sparse-view CT Reconstruction

Sparse-view CT reconstruction aims to reduce radiation exposure by recovering CT volumes from limited X-ray projections. Analytical methods such as FBP and FDK require suficient angular sampling and often introduce streak artifacts and structural distortions under severe undersampling (Feldkamp, Davis, and Kress 1984). Traditional approaches (Yang et al. 2025; Niu et al. 2014) alleviate this ill-posedness with handcrafted regularization priors, but they typically require expensive optimization and careful parameter tuning. Deep learning methods improve reconstruction by learning efective data-driven priors from large-scale datasets (Chen et al. 2018; Lin et al. 2019). Image-domain approaches, such as FBPConvNet (Jin et al. 2017), refine degraded analytical reconstructions, whereas projection-domain approaches recover missing measurements before analytical reconstruction through DNN-based sinogram synthesis (Lee et al. 2018) or learned interpolation (Dong, Vekhande, and Cao 2019). Nevertheless, these approaches still depend on multiple projections collected along a scanning trajectory. In contrast, X-ray-to-CT reconstruction relies on only one or two radiographs, leading to a more severely under-constrained inverse problem that requires stronger anatomical priors.

## Medical Image Generation

Generative models, particularly CNN-GANs and difusion models, have been widely applied to medical image reconstruction (Song et al. 2026; Lin et al. 2026; Pan et al. 2026). CNN-GANs learn supervised input–target mappings through convolutional generators, while adversarial discriminators promote realistic textures and plausible structures (Goodfellow et al. 2014; Isola et al. 2017). They have been applied to accelerated MRI, low-dose CT, and cross-modality translation (Yang et al. 2017; Liu and Ding 2023; Armanious et al. 2020; Wu et al. 2024, 2026), ofering eficient inference and structural control. Difusion models generate highquality images through iterative denoising (Ho, Jain, and Abbeel 2020) and have been extended to MRI, sparse-view CT, and low-dose CT reconstruction (Webber and Reader 2024; Chung and Ye 2022). Despite their strong generative capability, difusion models require multi-step sampling, resulting in considerable computational overhead for 3D reconstruction and structural uncertainty (Jiang et al. 2025; Chen et al. 2025b). Therefore, we adopt a CNN-GAN framework, which provides eficient deterministic inference and better integrates with anatomy-conditioned CT reconstruction.

![](images/f391688314c2814ae45fef621eb4ea9221698a8cce1cc02b220a3f02f6eb3643.jpg)  
Figure 2: Overview of LiftXR, an interleaved, geometry-guided framework coupling volumetric generation with anatomical perception. A Layout Lifter first performs projection-conditioned anatomical generation, which guides an Intensity Renderer to reconstruct an initial CT volume. A Layout Parser then performs reconstruction-conditioned anatomical perception using the spatially resolved boundary and intensity cues. The parsed layout is subsequently fed back to an Intensity Calibrator to correct region-specific intensity distributions.

## Method

LiftXR reconstructs a CT volume from paired PA and latera X-rays by interleaving anatomical layout recovery with volumetric intensity estimation. As illustrated in Fig. 2, the two projections are first up-sampled and fused into an explicit 3D X-ray volume. A Layout Lifter predicts a coarse anatomica layout, which constrains an Intensity Renderer to generate an initial CT volume. Although imperfect, this reconstruction provides spatially resolved boundary and intensity cues for a Layout Parser to refine the anatomical layout, further regularizing regional occupancy and spatial relationships. The refined layout is then combined with the volumetric representation to guide an Intensity Calibrator in correcting region-specific intensity inconsistencies. Through this reciprocal interaction between layout and intensity, LiftXR progressively resolves geometric ambiguity and improves anatomical fidelity.

## Bi-planar X-ray Initialization

Given two PA and LAT projections, we first construct an explicit X-ray volume to bridge the gap between twodimensional observations and the target three-dimensional space. Each projection is replicated along its unobserved spatial dimension and aligned within a shared volumetric coordinate system for subsequent aggregation.

$$
V _ { \mathrm { P A } } = \mathrm { R e p l i c a t e } ( X _ { \mathrm { P A } } , D ) , V _ { \mathrm { L A T } } = \mathrm { R e p l i c a t e } ( X _ { \mathrm { L A T } } , W ) ,\tag{1}
$$

where $V _ { \mathrm { P A } } , V _ { \mathrm { L A T } } \in \mathbb { R } ^ { H \times W \times D }$ denote the lifted representations of the two views. We then aggregate these representations to obtain the X-ray volume by voxel-wise averaging

$$
V _ { \mathrm { x r a y } } = \mathrm { A v e r a g e } \left( V _ { \mathrm { P A } } , V _ { \mathrm { L A T } } \right) ,\tag{2}
$$

which preserves the ray-integrated observations from both projections in a unified representation. This strategy can naturally extend to additional projection angles by lifting each view into the same volumetric coordinate system and applying the same aggregation operation. However, a direct replication operation cannot recover the missing depth information or resolve the inherent projection ambiguity.

## Layout Lifting and Intensity Rendering

Therefore, starting from the X-ray volume $V _ { \mathrm { x r a y } }$ , LiftXR performs geometry-guided volumetric lifting to establish an initial volumetric state. Instead of directly predicting intensity from sparse projections, a Layout Lifter $\theta _ { l l }$ first estimates a 3D anatomical layout that provides explicit geometric constraints for subsequent CT reconstruction. The layout is predicted solely based on $V _ { \mathrm { x r a y } }$ as

$$
L _ { \mathrm { l i f t e d } } = \theta _ { l l } ( V _ { \mathrm { x r a y } } ) ,\tag{3}
$$

where $L _ { \mathrm { l i f t e d } }$ represents the lifted layout. By encoding organ occupancy, spatial extent, and relative arrangement, $L _ { \mathrm { l i f t e d } }$ provides anatomical configurations compatible with the biplanar observations. Conditioned on the projection-derived representation and predicted layout, the Intensity Renderer $\theta _ { i r }$ reconstructs the coarse CT volume as

$$
V _ { \mathrm { r e n d e r e d } } = \theta _ { i r } \left( V _ { \mathrm { x r a y } } \oplus L _ { \mathrm { l i f t e d } } \right)\tag{4}
$$

where $\oplus$ is the channel-wise concatenation. This layoutconditioned lifting process transforms the X-ray volume into a structured volumetric state that provides a geometry-aware initialization for subsequent refinement.

## Layout Parsing and Intensity Calibration

The lifted layout and CT volume remain imperfect due to limited information from bi-planar projections. LiftXR therefore introduces an interleaved refinement stage that parses the anatomical layout and calibrates the CT intensity field.

$$
L _ { \mathrm { p a r s e d } } = \theta _ { l p } \left( V _ { \mathrm { x r a y } } \oplus V _ { \mathrm { r e n d e r e d } } \right) ,\tag{5}
$$

where $V _ { \mathrm { x r a y } }$ preserves projection evidence and $V _ { \mathrm { r e n d e r e d } }$ provides spatially resolved boundary and attenuation cues for the layout refinement by a parser $\theta _ { l p } .$ . Conditioned on the refined layout $L _ { \mathrm { p a r s e d } }$ , the Intensity Calibrator $\theta _ { i c }$ further updates the rendered CT volume as

$$
V _ { \mathrm { c a l i b r a t e d } } = \theta _ { i c } \left( V _ { \mathrm { x r a y } } \oplus V _ { \mathrm { r e n d e r e d } } \oplus L _ { \mathrm { p a r s e d } } \right) .\tag{6}
$$

Overall, although both stages predict anatomical layouts, they solve diferent tasks. The Layout Lifter performs projection-conditioned layout generation to initialize the missing 3D geometry, whereas the Layout Parser performs reconstruction-conditioned anatomical perception to recover spatially resolved organ boundaries and regional identities.

## Interleaved Layout-Intensity Supervision

There are two aspects to supervise the anatomical layout prediction and CT reconstruction. We denote predicted layouts as $\mathcal { A } = \{ L _ { \mathrm { l i f t e d } } , L _ { \mathrm { p a r s e d } } \}$ and reconstructed volumes as $\mathcal { V } = \{ V _ { \mathrm { r e n d e r e d } } , V _ { \mathrm { c a l i b r a t e d } } \}$ . Layout supervision constrains 3D anatomical geometry, while CT supervision preserves intensity and volumetric appearance.

Anatomical Layout Supervision: Accurate anatomica layouts provide explicit constraints on organ occupancy, boundary placement, and spatial organization. We extract seven parenchymal and major thoracoabdominal regions, including tissue, bone, heart, liver, left lung, right lung, and esophagus, for spatial localization. The target anatomical layout is extracted as

$$
L _ { \mathrm { G T } } = E _ { \mathrm { s e g } } \left( V _ { \mathrm { G T } } \right) ,\tag{7}
$$

where $E _ { \mathrm { s e g } } ( \cdot )$ denotes the anatomical mask extractor. Both the lifted layout and parsed layout are supervised by

$$
\mathcal { L } _ { \mathrm { l a y o u t } } = \sum _ { a \in \mathcal { A } } \left[ \mathrm { C E } \left( a , L _ { \mathrm { G T } } \right) + \mathrm { D i c e } \left( a , L _ { \mathrm { G T } } \right) \right] ,\tag{8}
$$

where a is selected from A. This supervision guides both initial layout lifting and subsequent layout parsing toward anatomically plausible three-dimensional structures.

CT Reconstruction Supervision: While anatomical layouts provide explicit geometric constraints, they cannot fully determine region-specific intensity distributions. Therefore, both rendered and calibrated CT volumes are supervised using the ground-truth CT volume. Following (Ying et al. 2019), for each reconstructed volume $v \in \mathcal V .$ , the multi-scale feature matching loss is defined as

$$
\mathcal { L } _ { \mathrm { M S L } } ^ { ( v ) } = \Vert M S ( v ) - \mathrm { s g } \left( M S ( V _ { \mathrm { G T } } ) \right) \Vert _ { 1 } ,\tag{9}
$$

where $M S ( \cdot )$ denotes the multi-scale feature transformation, and $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operation. The CT reconstruction objective is defined as

$$
\mathcal { L } _ { \mathrm { r e c } } = \sum _ { v \in \mathcal { V } } \left[ \mathcal { L } _ { 1 } \left( v , V _ { \mathrm { G T } } \right) + \mathcal { L } _ { \mathrm { M S L } } ^ { ( v ) } + \mathcal { L } _ { \mathrm { a d v } } ^ { ( v ) } \right] ,\tag{10}
$$

where $\mathcal { L } _ { 1 }$ is the mean absolute error (MAE) loss and $\mathcal { L } _ { \mathrm { a d v } } ^ { ( v ) }$ denotes the adversarial loss that encourages reconstructed CT volumes to match the distribution of real CT volumes, following the typical setting (Ying et al. 2019; Pan et al. 2025). This objective preserves voxel-level intensities, multi-scale structural consistency, and realistic volumetric appearance throughout the lifting and calibration stages.

Overall Training Objective: The anatomical layout and CT reconstruction objectives jointly optimize the shape and intensity branches of LiftXR. The overall training objective is defined as

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { r e c } } + \lambda \times \mathcal { L } _ { \mathrm { l a y o u t } } ,\tag{11}
$$

where λ is a loss weight to balance the two aspects.

## Experiment and Result

## Implementation Details

Dataset and Pre-processing: We evaluate LiftXR on two benchmarking chest CT datasets. On the large-scale CT-RATE dataset (Hamamci et al. 2026), we randomly select 2,000 subjects for training and 300 for testing. On the LIDC-IDRI dataset, which contains 1,018 CT volumes (Armato III et al. 2011), we use 818 volumes for training and 200 for testing. Following the oficial setting, anatomical labels are generated on both datasets by using a frozen TotalSegmentator model (Wasserthal et al. 2023). Following prior X-rayto-CT studies (Ying et al. 2019; Gopalakrishnan and Golland 2022), we generate PA and Lateral radiographs from each CT volume using digitally reconstructed radiography (DRR) techniques, yielding synthetic pairs. We use the Dif-DRR (Gopalakrishnan and Golland 2022) tool<sup>1</sup>, with the setting using a source-to-detector distance of 3,000 mm and a detector pixel spacing of 1.0 mm. All CT volumes are resized to $1 2 8 \times 1 2 8 \times 1 2 8$ . The intensity range is clamped to [−1024, 1024] HU and then normalized to [0, 1].

<table><tr><td rowspan="3">Method</td><td colspan="4">CT-RATE</td><td colspan="4">LIDC-IDRI</td></tr><tr><td colspan="2">Reconstruction</td><td colspan="2">Segmentation</td><td colspan="2">Reconstruction</td><td colspan="2">Segmentation</td></tr><tr><td>PSNR (dB)↑</td><td>SSIM (%)↑</td><td>Dice (%)↑</td><td>HD95 (voxel)↓</td><td>PSNR (dB)↑</td><td>SSIM (%)↑</td><td>Dice (%)↑</td><td>HD95 (voxel)↓</td></tr><tr><td>X2CT (Ying et al. 2019)</td><td> $2 5 . 9 6 _ { \pm 1 . 0 0 }$ </td><td> $7 1 . 8 8 _ { \pm 0 . 5 1 }$ </td><td> $6 3 . 3 9 _ { \pm 0 . 7 6 }$ </td><td> $1 6 . 0 5 { \scriptstyle \pm 2 . 1 2 }$ </td><td> $2 5 . 0 6 _ { \pm 1 . 1 3 }$ </td><td> $6 9 . 4 7 _ { \pm 0 . 5 7 }$ </td><td> $5 6 . 8 0 _ { \pm 0 . 7 3 }$ </td><td> $1 9 . 8 1 _ { \pm 1 . 4 5 }$ </td></tr><tr><td>PerX2CT (Kyung et al. 2023)</td><td> $2 5 . 8 4 _ { \pm 0 . 8 4 }$ </td><td> $7 1 . 2 9 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $5 5 . 4 4 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $1 9 . 1 9 { \scriptstyle \pm 1 . 6 4 }$ </td><td> $2 5 . 3 4 { \scriptstyle \pm 0 . 8 7 }$ </td><td> $\underline { { 7 0 . 2 4 } } _ { \pm 0 . 6 0 }$ </td><td> $5 8 . 9 6 { \scriptstyle \pm 0 . 7 8 }$ </td><td> $1 8 . 2 3 { \scriptstyle \pm 1 . 0 3 }$ </td></tr><tr><td>DSDF (Pan et al. 2025)</td><td> $2 5 . 8 2 _ { \pm 0 . 9 9 }$ </td><td> $\underline { { 7 3 . 2 9 } } _ { \pm 0 . 3 9 }$ </td><td> $\underline { { 6 6 . 0 9 } } _ { \pm 0 . 7 9 }$ </td><td> $\underline { { 1 3 . 9 2 } } _ { \pm 1 . 8 1 }$ </td><td> $\underline { { 2 5 . 7 3 } } _ { \pm 0 . 9 4 }$ </td><td> $6 9 . 4 3 _ { \pm 0 . 5 8 }$ </td><td> $5 8 . 3 8 _ { \pm 0 . 8 9 }$ </td><td> $\underline { { 1 6 . 8 7 } } _ { \pm 1 . 0 1 }$ </td></tr><tr><td>GAAL (Liu et al. 2024b)</td><td> $2 2 . 8 2 { \scriptstyle \pm 1 . 2 1 }$ </td><td> $6 5 . 6 2 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $2 8 . 2 1 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $9 4 . 1 9 { \scriptstyle \pm 3 . 5 0 }$ </td><td> $2 1 . 0 2 { \scriptstyle \pm 1 . 2 3 }$ </td><td> $5 1 . 5 3 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $3 0 . 0 9 { \scriptstyle \pm 0 . 9 8 }$ </td><td> $8 5 . 5 6 { \scriptstyle \pm 4 . 9 8 }$ </td></tr><tr><td>DX2CT (Jeong, Yoo, and Chun 2025)</td><td> $2 5 . 7 6 _ { \pm 0 . 4 9 }$ </td><td> $7 0 . 4 2 _ { \pm 0 . 1 5 }$ </td><td> $6 4 . 1 1 _ { \pm 0 . 4 9 }$ </td><td> $1 4 . 9 8 _ { \pm 1 . 2 3 }$ </td><td> $2 5 . 0 6 _ { \pm 1 . 3 4 }$ </td><td> $6 9 . 2 7 _ { \pm 0 . 4 4 }$ </td><td> $5 9 . 0 2 _ { \pm 1 . 0 9 }$ </td><td> $1 7 . 4 5 { \scriptstyle \pm 1 . 4 9 }$ </td></tr><tr><td>DiffuX2CT (Liu et al. 2024a)</td><td> $\underline { { 2 6 . 0 3 } } _ { \pm 0 . 6 0 }$ </td><td> $7 2 . 8 2 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $6 5 . 5 8 { \scriptstyle \pm 0 . 5 1 }$ </td><td> $1 4 . 8 2 _ { \pm 1 . 1 2 }$ </td><td> $2 5 . 1 2 { \scriptstyle \pm 1 . 3 5 }$ </td><td> $7 0 . 1 6 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $\underline { { 5 9 . 2 2 } } _ { \pm 0 . 9 5 }$ </td><td> $1 7 . 0 7 { \scriptstyle \pm 1 . 4 0 }$ </td></tr><tr><td>LiftXR (Ours)</td><td> $\mathbf { 2 6 . 4 5 _ { \pm 1 . 0 0 } } ^ { \mathbf { * * } }$ </td><td> ${ \bf 7 6 . 4 4 _ { \pm 0 . 4 0 } } ^ { \ast \ast }$ </td><td> $\mathbf { 6 8 . 6 4 _ { \pm 0 . 8 0 } } ^ { \ast \ast }$ </td><td> $\mathbf { 1 2 . 5 1 _ { \pm 1 . 6 0 } } ^ { * * }$ </td><td> $\mathbf { 2 6 . 0 3 _ { \pm 1 . 0 8 } } ^ { \ast }$ </td><td> $\mathbf { 7 1 . 7 2 _ { \pm 0 . 6 0 } } ^ { * * }$ </td><td> $\mathbf { 6 2 . 1 6 _ { \pm 0 . 8 7 } } ^ { \pm * }$ </td><td> $\mathbf { 1 5 . 9 1 _ { \pm 1 . 1 4 } } ^ { * * }$ </td></tr></table>

Table 1: Quantitative comparisons on the CT-RATE and LIDC-IDRI datasets. Reconstruction and segmentation results are reported as mean ± standard deviation. The best and second-best results are highlighted in bold and underlined, respectively. <sup>∗</sup> and <sup>∗∗</sup> indicate that LiftXR significantly outperforms the second-best method at $p < 0 . 0 5$ and $p < 0 . 0 1$ , respectively.

Experimental Settings: We train LiftXR using the Adam optimizer with a learning rate of $2 e ^ { - 4 }$ and a batch size of 4. Based on the sensitivity analysis, λ is set to 1. $E _ { \mathrm { s e g } } ( \cdot )$ is set to TotalSegmentator to obtain the anchor organs. 100,000 iterations are used for training, taking approximately 60 hours on a single NVIDIA GeForce RTX 4090 GPU. Each model inside our LiftXR is a 3D U-Net architecture, and the total computational complexity is 6480 GFLOPs with 172.62M parameters. To ensure fair comparisons, all experiments use a fixed random seed 43 and follow identical settings. For external downstream segmentation, we mainly focus on thoracoabdominal regions, and 20 specific categories are selected, which are elaborated in the supplementary material.

## Main Results

Quantitative Analysis: Tab. 1 compares LiftXR with representative X-ray-to-CT reconstruction methods on CT-RATE and LIDC-IDRI. LiftXR achieves the best performance across all reconstruction and segmentation metrics on both datasets, demonstrating consistent efectiveness in recovering CT intensities and anatomically meaningful threedimensional structures. The improvements in PSNR are relatively moderate compared with those in anatomical metrics, as voxel-wise errors are dominated by large homogeneous regions and are less sensitive to localized corrections around organ boundaries and small structures. In contrast, the consistent SSIM improvements indicate that LiftXR better preserves local structural patterns and tissue contrast during reconstruction. These results suggest that explicit anatomical layout modeling reduces the depth ambiguity inherent in bi-planar projections and constrains the reconstruction toward anatomically plausible volumetric configurations. These gains indicate that the predicted layout explicitly constrains organ extent, shape, and spatial relationships, preventing anatomically implausible solutions that may still achieve comparable voxel-wise similarity. Meanwhile, the reconstructed CT volume provides boundary and tissue-specific intensity cues to refine uncertain layouts. The refined layout subsequently guides CT calibration to reduce tissue mixing, geometric distortions, and inconsistent intensity distributions. Through this interleaved refinement process, LiftXR progressively improves both organ occupancy and boundary accuracy.

Qualitative Analysis: Fig. 3 compares axial and coronal CT slices, sagittal segmentation overlays, and 3D anatomical renderings on the CT-RATE dataset. Existing methods generally recover the overall thoracic structure but exhibit noticeable errors in both slice-level details and volumetric anatomy. X2CT, PerX2CT, and DSDF preserve the general lung regions but produce blurred mediastinal boundaries, missing fine structures, and inaccurate organ shapes. GAAL, a multi-view reconstruction method, exhibits severe oversmoothing and substantial loss of anatomical details, likely due to the dificulty of inferring fine structures from limited projection constraints. DX2CT and DifuX2CT, difusionbased approaches, recover cleaner global structures with improved volumetric coherence, but still introduce local boundary errors and inconsistencies in inter-organ relationships. In comparison, LiftXR produces results closer to the ground truth, with clearer lung contours and more accurate vertebral and rib structures. The final row removes surrounding tissues to visualize internal anatomy, where LiftXR maintains more continuous ribs, better spine alignment, more symmetric lungs, and more plausible organ arrangements. These localized geometric improvements explain the moderate PSNR gains and the larger improvements in anatomical metrics, which are more sensitive to organ extent and boundary accuracy. The qualitative observations agree with the quantitative results, demonstrating that interleaved refinement of anatomical layout and CT intensity enables more accurate volumetric reconstruction.

Ablation Study: Tab. 2 reports the ablation results on CT-RATE. Both the layout and interleaved training contribute to reconstruction and segmentation performance, while their combination provides complementary benefits. Anchor organs establish explicit organ-level references, constraining the possible anatomical configurations during the initial lifting process and reducing large deviations in organ extent and spatial arrangement. The interleaved training further improves the reconstruction by enabling interleaved interaction between the anatomical layout and CT volume, where the estimated geometry and intensity information mutually correct residual structural and attenuation errors. While anchor organs mainly provide global anatomical guidance, the interleaved training focuses on improving local details, including organ boundaries, tissue appearance, and inter-organ relationships. Their efects demonstrate that reliable anatomical prediction together with interleaved layout-intensity refinement is efective in X-ray-to-CT reconstruction.

![](images/12c8e08a0a8145b8ece3c8368942554ca648523ac2e68dc2230de7a7420e7c9b.jpg)  
X2CT

![](images/9e27650b9c5b78a3bbb1dbeeff2b6e8c501263a90dff286dcfbf65b70a6d0c3e.jpg)  
PerX2CT

![](images/3a768bb631abf2d072fbc95e03c2653a67a437ac4033da66d72919cd8588737f.jpg)  
DSDF

![](images/1b2eaf097583590baeaa70845963e171534199e61efacec547ddadbcaf4651ac.jpg)  
GAAL

![](images/e18874c23e632e9e67abaff3a004796655055d8fd91978fa23d64751cef4c5a8.jpg)  
DX2CT

![](images/ccea76f22429fec78fdeb5be2a6a5bba907bcc56bfb358f1465c8d5bcddab22b.jpg)  
DiffuX2CT

![](images/4b0af26ac940a4a6ee0688ce148f3362f0972ced7d948a3077dca13f8c74d1d7.jpg)  
LiftXR

![](images/87aa9a8fb50664f422a15956a77f0427ab057b1a4034203bdb11f9724922f0e9.jpg)  
Ground Truth

Figure 3: Qualitative comparisons on the CT-RATE dataset. The top two rows show axial and coronal slices, while the bottom two rows present the corresponding segmentation layouts in 2D and 3D. LiftXR more faithfully preserves anatomical structures and recovers fine-grained details, such as maintaining rib continuity and reconstructing a smooth liver contour.
<table><tr><td rowspan="2">Layout</td><td rowspan="2">Interleaved Training</td><td colspan="2">Reconstruction</td><td colspan="2">Segmentation</td></tr><tr><td>PSNR (dB)↑</td><td>SSIM (%)↑</td><td>Dice (%)↑</td><td>HD95 (voxel)↓</td></tr><tr><td></td><td>=</td><td>25.68</td><td>72.89</td><td>63.54</td><td>15.45</td></tr><tr><td></td><td>√</td><td>26.07</td><td>75.34</td><td>67.03</td><td>13.31</td></tr><tr><td>√</td><td></td><td>25.95</td><td>74.10</td><td>65.45</td><td>13.45</td></tr><tr><td>√</td><td>√</td><td>26.45</td><td>76.44</td><td>68.64</td><td>12.51</td></tr></table>

Table 2: Ablation studies on the CT-RATE dataset. Best and second-best results are highlighted. Both anatomical layout and interleaved training are evaluated for both reconstruction and downstream segmentation.

Analysis of Intermediate Processes: As shown in Fig. 4, the visualization further illustrates the efectiveness of the interleaved layout-intensity refinement process in LiftXR. Starting from the initial lifted layout and rendered CT volume, the reconstructed CT provides spatially resolved boundary and intensity cues that help refine uncertain anatomical regions. Meanwhile, the refined layout introduces more accurate structural constraints for subsequent CT intensity calibration. Through this reciprocal interaction, both anatomical layouts and CT volumes are progressively improved, leading to more accurate organ localization, clearer anatomical boundaries, and more consistent intensity distributions compared with the initial reconstruction.

![](images/7dd12bb9604ba8901cf2a4ad0839db16ee3f920510c87686da19614e68ccd4ed.jpg)  
Figure 4: Intermediate visualization of the layout-intensity refinement process in LiftXR. The first row shows the initial lifted anatomical layout and rendered CT volume, the second row presents the refined layout and calibrated CT after interleaved optimization, and the third row provides the ground-truth reference on the CT-RATE dataset.

## Discussions

Efect of Layout Selection: Tab. 3 evaluates diferent anatomical supervision strategies on CT-RATE. HU-based regions (Pan et al. 2025) provide coarse tissue constraints from intensity ranges, while organ-level supervision introduces explicit anatomical boundaries and spatial relationships. Both organ anchors and full anatomical labels outperform HU-based regions, demonstrating the importance of organ-level guidance for resolving projection ambiguity. However, incorporating all possible anatomical targets from TotalSegmentator does not further improve performance since hollow and fine organs may introduce additional segmentation uncertainty, leading to sub-optimal performance.

<table><tr><td>Layout Selection</td><td>Loss Weight λ</td><td>PSNR (dB)↑</td><td>SSIM (%)↑</td></tr><tr><td>HU-based Regions</td><td></td><td>26.07</td><td>75.53</td></tr><tr><td>Anchor Organs</td><td>1</td><td>26.45</td><td>76.44</td></tr><tr><td>All Organs</td><td></td><td>26.37</td><td>76.31</td></tr><tr><td></td><td>0.1</td><td>26.28</td><td>75.70</td></tr><tr><td>Anchor Organs</td><td>1</td><td>26.45</td><td>76.44</td></tr><tr><td></td><td>10</td><td>26.09</td><td>75.46</td></tr></table>

Table 3: Sensitivity analysis of anatomical layout selection and the loss weight λ on the CT-RATE dataset.

![](images/527e1b9f915b6a6a1d7dbd52af8e0138a8d45de30fedf39d879b2a85af35efd5.jpg)  
Figure 5: Generalization analysis on real frontal and lateral X-rays on the MIMIC dataset. LiftXR synthesizes more plausible organ textures while reducing boundary distortions.

Efect of λ: Tab. 3 further investigates the balance between anatomical layout and CT supervision by varying the loss weight λ. The best performance is achieved at λ = 1, suggesting that accurate bi-planar X-ray-to-CT reconstruction requires a joint optimization of anatomical geometry and intensity. When λ is reduced, insuficient anatomical supervision weakens the geometric constraints provided by the predicted layouts, limiting the recovery of organ structures from ambiguous projections. In contrast, an excessively large λ places excessive emphasis on anatomical consistency, which may degrade intensity reconstruction. Therefore, λ = 1 provides an efective balance between anatomical guidance and CT intensity recovery and is adopted in all experiments.

Generalization to Real X-ray Images: Fig. 5 shows the results using real frontal and lateral X-rays on the MIMIC dataset (Johnson et al. 2019). Following (Ying et al. 2019; Kyung et al. 2023), we first train a CycleGAN (Zhu et al. 2017) to transform the style between real X-rays and our synthetic projections, and then reconstruct CT volumes using X2CT, DSDF, and LiftXR. In the enlarged regions, X2CT produces noticeable boundary artifacts, suggesting inaccurate anatomical recovery under real X-ray inputs. DSDF preserves more coherent global structures but still fails to recover clear textures. In contrast, LiftXR produces sharper organ boundaries and more plausible anatomical structures, benefiting from the layout guidance and interleaved layoutintensity refinement. It demonstrates that LiftXR maintains cross-domain generalization potential on real radiographs.

<table><tr><td rowspan="3">Method</td><td colspan="4">[0° , 88°]</td><td colspan="4">[0° , 92°]</td></tr><tr><td>PSNR (dB)↑</td><td>∆</td><td>SSIM (%)↑</td><td>∆</td><td>PSNR (dB)↑</td><td>∆</td><td>SSIM (%)↑</td><td>∆</td></tr><tr><td>X2CT</td><td>25.49</td><td>0.47↓</td><td>70.79</td><td>1.09↓</td><td>25.61</td><td>0.35↓</td><td>70.92</td><td>0.96↓</td></tr><tr><td>DSDF</td><td>25.20</td><td>0.62↓</td><td>72.19</td><td>1.10↓</td><td>25.18</td><td>0.64↓</td><td>72.12</td><td>1.17.↓</td></tr><tr><td>LiftXR</td><td>26.30</td><td>0.15↓</td><td>76.14</td><td>0.30↓</td><td>26.27</td><td>0.18↓</td><td>75.94</td><td>0.50↓</td></tr></table>

Table 4: Robustness evaluation under diferent angle deviations on the CT-RATE dataset. The values report the reconstruction performance and the corresponding degradation (∆) compared with the original acquisition setting.

Robustness to Projection Angle Variations: To further validate the robustness of LiftXR under realistic acquisition variations, we investigate the impact of projection angle deviations from the standard orthogonal bi-planar setting. In clinical scenarios, frontal and lateral X-rays may not be perfectly aligned due to patient positioning and acquisition conditions, introducing geometric inconsistencies between the observed projections and the ideal reconstruction configuration. As shown in Tab. 4, we perturb the projection angles to [0<sup>◦</sup>, 88<sup>◦</sup>] and [0<sup>◦</sup>, 92<sup>◦</sup>] while using the same reconstruction pipeline. LiftXR consistently achieves the best reconstruction performance under both perturbed settings, while exhibiting the smallest performance degradation compared with existing methods. This robustness to small projection perturbations is attributed to the proposed interleaved layout-intensity refinement, where anatomical layouts provide explicit geometric constraints and reconstructed CT volumes further refine the estimated layouts. Such reciprocal interaction enables LiftXR to partially compensate for geometric deviations and maintain reliable volumetric reconstruction under practical and non-ideal projection scenarios.

Limitation and Future Work: While LiftXR demonstrates promising performance in bi-planar X-ray-to-CT reconstruction, several limitations remain. First, reconstructed CT volumes are currently assessed through image quality and anatomical segmentation metrics, while their efectiveness for clinical applications requires further exploration. Second, the interleaved refinement process introduces additional computational costs, which may limit the deployment to specific applications such as mobile scenarios. Furthermore, bi-planar X-ray-to-CT reconstruction remains inherently constrained by limited information, making the recovery of highly variable or small pathological anatomical structures challenging. Future work will investigate more eficient architectures and conduct broader clinical evaluations to improve the generalizability and applicability of LiftXR.

## Conclusion

We present LiftXR, an interleaved geometry-guided framework for bi-planar X-ray-to-CT reconstruction. LiftXR transitions from projection-conditioned anatomical layout generation to reconstruction-conditioned anatomical perception, allowing the reconstructed volume to refine its geometry and the perceived anatomy to further calibrate region-specific intensities. This generation-perception coupling reduces geometric ambiguity and enables more anatomically consistent volumetric reconstruction. Experiments on two public chest CT datasets demonstrate state-of-the-art reconstruction performance and impressive downstream segmentation results, indicating superior anatomical fidelity.

## References

Armanious, K.; Jiang, C.; Fischer, M.; Küstner, T.; Hepp, T.; Nikolaou, K.; Gatidis, S.; and Yang, B. 2020. MedGAN: Medical Image Translation using GANs. Computerized Medical Imaging and Graphics, 79: 101684.

Armato III, S. G.; McLennan, G.; Bidaut, L.; McNitt-Gray, M. F.; Meyer, C. R.; Reeves, A. P.; Zhao, B.; Aberle, D. R.; Henschke, C. I.; Hofman, E. A.; et al. 2011. The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A Completed Reference Database of Lung Nodules on CT Scans. Medical Physics, 38(2): 915–931.

Cai, Y.; Wang, J.; Yuille, A.; Zhou, Z.; and Wang, A. 2024. Structure-Aware Sparse-View X-ray 3D Reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11174–11183.

Chen, H.; Hao, Z.; Guo, L.; and Xiao, L. 2025a. Mitigating Data Consistency Induced Discrepancy in Cascaded Difusion Models for Sparse-view CT Reconstruction. IEEE Transactions on Medical Imaging, 44(7): 3012–3024.

Chen, H.; Zhang, Y.; Chen, Y.; Zhang, J.; Zhang, W.; Sun, H.; Lv, Y.; Liao, P.; Zhou, J.; and Wang, G. 2018. LEARN: Learned Experts’ Assessment-Based Reconstruction Network for Sparse-Data CT. IEEE Transactions on Medical Imaging, 37(6): 1333–1347.

Chen, J.; Lin, Y.; Qin, Y.; Wang, H.; and Li, X. 2025b. Cross-View Generalized Difusion Model for Sparse-View CT Reconstruction. In International Conference on Medical Image Computing and Computer-Assisted Intervention, 140– 150. Springer.

Chung, H.; and Ye, J. C. 2022. Score-based difusion models for accelerated MRI. Medical Image Analysis, 80: 102479.

Deo, Y.; Jia, Y.; Lassila, T.; Smith, W. A.; Lawton, T.; Kang, S.; Frangi, A. F.; and Habli, I. 2025. Metrics that matter: Evaluating image quality metrics for medical image generation. arXiv preprint arXiv:2505.07175.

Dong, X.; Vekhande, S.; and Cao, G. 2019. Sinogram interpolation for sparse-view micro-CT with deep learning neural network. In Medical Imaging 2019: Physics of Medical Imaging, volume 10948, 692–698. SPIE.

Feldkamp, L. A.; Davis, L. C.; and Kress, J. W. 1984. Practical Cone-beam Algorithm. Journal of the Optical Society of America A, 1(6): 612–619.

Goodfellow, I. J.; Pouget-Abadie, J.; Mirza, M.; Xu, B.; Warde-Farley, D.; Ozair, S.; Courville, A.; and Bengio, Y. 2014. Generative Adversarial Nets. Advances in Neural Information Processing Systems, 27.

Gopalakrishnan, V.; and Golland, P. 2022. Fast Autodiferentiable Digitally Reconstructed Radiographs for Solving Inverse Problems in Intraoperative Imaging. In Workshop on Clinical Image-Based Procedures, 1–11. Springer.

Guo, T.; Liu, Y.; Zhang, P.; Liu, Y.; and Gui, Z. 2024. MAIR-Net: a sparse-view CT reconstruction network based on a combination of mixed attention and iterative optimization learning. Journal ofInstrumentation, 19(08): P08029.

Hamamci, I. E.; Er, S.; Wang, C.; Almas, F.; Simsek, A. G.; Esirgun, S. N.; Dogan, I.; Durugol, O. F.; Hou, B.; Shit, S.; et al. 2026. Generalist foundation models from a multimodal dataset for 3D computed tomography. Nature Biomedical Engineering, 1–19.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising Difusion Probabilistic Models. Advances in Neural Information Processing Systems, 33: 6840–6851.

Isola, P.; Zhu, J.-Y.; Zhou, T.; and Efros, A. A. 2017. Image-To-Image Translation With Conditional Adversarial Networks. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, 1125–1134.

Jeong, Y. S.; Yoo, H. B.; and Chun, I. Y. 2025. DX2CT: Diffusion Model for 3D CT Reconstruction from Bi or Monoplanar 2D X-ray (s). In International Conference on Acoustics, Speech and Signal Processing, 1–5. IEEE.

Jiang, H.; Imran, M.; Zhang, T.; Zhou, Y.; Liang, M.; Gong, K.; and Shao, W. 2025. Fast-DDPM: Fast Denoising Difusion Probabilistic Models for Medical Image-to-Image Generation. IEEE Journal ofBiomedical and Health Informatics, 29(10): 7326–7335.

Jin, K. H.; McCann, M. T.; Froustey, E.; and Unser, M. 2017. Deep Convolutional Neural Network for Inverse Problems in Imaging. IEEE Transactions on Image Processing, 26(9): 4509–4522.

Johnson, A. E.; Pollard, T. J.; Berkowitz, S. J.; Greenbaum, N. R.; Lungren, M. P.; Deng, C.-y.; Mark, R. G.; and Horng, S. 2019. MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports. Scientific Data, 6(1): 317.

Jung, H. 2021. Basic Physical Principles and Clinical Applications of Computed Tomography. Progress in Medical Physics, 32(1): 1–17.

Kyung, D.; Jo, K.; Choo, J.; Lee, J.; and Choi, E. 2023. Perspective Projection-Based 3d CT Reconstruction from Biplanar X-Rays. In IEEE International Conference on Acoustics, Speech and Signal Processing, 1–5. IEEE.

Lee, H.; Lee, J.; Kim, H.; Cho, B.; and Cho, S. 2018. Deep-Neural-Network-Based Sinogram Synthesis for Sparse-View CT Image Reconstruction. IEEE Transactions on Radiation and Plasma Medical Sciences, 3(2): 109–119.

Lin, T.; Li, X.; Zhuang, C.; Chen, Q.; Cai, Y.; Ding, K.; Yuille, A. L.; and Zhou, Z. 2025. Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction? arXiv preprint arXiv:2506.02093.

Lin, W.-A.; Liao, H.; Peng, C.; Sun, X.; Zhang, J.; Luo, J.; Chellappa, R.; and Zhou, S. K. 2019. DuDoNet: Dual Domain Network for CT Metal Artifact Reduction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10512–10521.

Lin, Y.; Luo, Z.; Zhao, W.; and Li, X. 2023. Learning Deep Intensity Field for Extremely Sparse-View CBCT Reconstruction. In International Conference on Medical Image Computing and Computer-Assisted Intervention, 13–23. Springer.

Lin, Y.; Sun, H.; Li, Y.; Aslam, R.; Tse, L. F.; Cheng, T.; Chui, C. S.; Yau, W. F.; Le Meur, V. R.; Amangeldy, M.; et al. 2026. Real-time reconstruction of 3D bone models via very-low-dose protocols. npj Digital Medicine, 9(1): 353.

Lin, Y.; Wang, H.; Chen, J.; and Li, X. 2024. Learning 3D Gaussians for Extremely Sparse-View Cone-Beam CT Reconstruction. In International Conference on Medical Image Computing and Computer-Assisted Intervention, 425–435. Springer.

Liu, W.; and Ding, H. 2023. Solving Low-Dose CT Reconstruction via GAN with Local Coherence. In International Conference on Medical Image Computing and Computer-Assisted Intervention, 524–534. Springer.

Liu, X.; Qiao, Z.; Liu, R.; Li, H.; Zhang, J.; Zhen, X.; Qian, Z.; and Zhang, B. 2024a. DifuX2CT: Difusion Learning to Reconstruct CT Images from Biplanar X-Rays. In European Conference on Computer Vision, 458–476. Springer.

Liu, Z.; Fang, Y.; Li, C.; Wu, H.; Liu, Y.; Shen, D.; and Cui, Z. 2024b. Geometry-Aware Attenuation Learning for Sparse-View CBCT Reconstruction. IEEE Transactions on Medical Imaging, 44(2): 1083–1097.

Niu, S.; Gao, Y.; Bian, Z.; Huang, J.; Chen, W.; Yu, G.; Liang, Z.; and Ma, J. 2014. Sparse-view x-ray CT reconstruction via total generalized variation regularization. Physics in Medicine and Biology, 59(12): 2997–3017.

Pan, Y.; Ye, Y.; Zhang, Y.; Xia, Y.; and Shen, D. 2025. Draw Sketch, Draw Flesh: Whole-Body Computed Tomography from Any X-Ray Views. International Journal of Computer Vision, 133(5): 2505–2526.

Pan, Z.; Zhou, H.; Ren, Q.; Hou, X.; Dai, J.; Liu, X.; Che, Y.; Zhao, X.; Xie, Y.; Li, Z.; et al. 2026. X2Shape: CT-free 3D multi-organ reconstruction with biplanar X-rays. Medical Image Analysis, 104085.

Shi, J.; Pelt, D. M.; and Batenburg, K. J. 2026. DM4CT: Benchmarking Difusion Models for Computed Tomography Reconstruction. arXiv preprint arXiv:2602.18589.

Shi, W.; Hu, Y.; Sun, Y.; Chang, G.; Yang, Y.; Song, Y.; Qian, H.; Wei, Z.; Zhao, L.; Li, M.; et al. 2025. Exploration of optimal thresholds for predicting the invasive nature of stage T1 lung adenocarcinoma using artificial intelligence-based 3D solid component volume segmentation. Quantitative Imaging in Medicine and Surgery, 15(1): 249–258.

Song, T.; Wu, Y.; Hu, M.; Luo, X.; Wei, L.; Wang, G.; Guo, Y.; Xu, F.; and Zhang, S. 2026. Learning Modality-Aware Representations: Adaptive Group-wise Interaction Network for Multimodal MRI Synthesis. IEEE Transactions on Medical Imaging, 45(5): 2306–2316.

Wang, X.; Peng, Y.; Lu, L.; Lu, Z.; Bagheri, M.; and Summers, R. M. 2017. ChestX-ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. In

Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2097–2106.

Wasserthal, J.; Breit, H.-C.; Meyer, M. T.; Pradella, M.; Hinck, D.; Sauter, A. W.; Heye, T.; Boll, D. T.; Cyriac, J.; Yang, S.; et al. 2023. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence, 5(5): e230024.

Webber, G.; and Reader, A. J. 2024. Difusion models for medical image reconstruction. BJR| Artificial Intelligence, 1(1): ubae013.

Wu, J.; Lin, J.; Jiang, X.; Zheng, W.; Zhong, L.; Pang, Y.; Meng, H.; and Li, Z. 2025. Dual-Domain deep prior guided sparse-view CT reconstruction with multi-scale fusion attention. Scientific Reports, 15(1): 16894.

Wu, Y.; Luo, X.; Xu, Z.; Guo, X.; Ju, L.; Ge, Z.; Liao, W.; and Cai, J. 2024. Diversified and Personalized Multi-rater Medical Image Segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11470–11479.

Wu, Y.; Song, T.; Wu, Z.; Ye, J.; Ge, Z.; Bai, W.; Chen, Z.; and Cai, J. 2026. Virtual Full-stack Scanning of Brain MRI via Imputing Any Quantised Code. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21026–21035.

Yang, G.; Yu, S.; Dong, H.; Slabaugh, G.; Dragotti, P. L.; Ye, X.; Liu, F.; Arridge, S.; Keegan, J.; Guo, Y.; et al. 2017. DA-GAN: Deep De-Aliasing Generative Adversarial Networks for Fast Compressed Sensing MRI Reconstruction. IEEE Transactions on Medical Imaging, 37(6): 1310–1321.

Yang, L.; Huang, J.; Yang, G.; and Zhang, D. 2025. CT-SDM: A Sampling Difusion Model for Sparse-View CT Reconstruction Across Various Sampling Rates. IEEE Transactions on Medical Imaging, 44(6): 2581–2593.

Ying, X.; Guo, H.; Ma, K.; Wu, J.; Weng, Z.; and Zheng, Y. 2019. X2CT-GAN: Reconstructing CT From Biplanar X-Rays With Generative Adversarial Networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10619–10628.

Yu, T.; Tian, X.; Yang, J.; He, D.; Yu, J.; Wang, X.; and Zhang, Y. 2025. SPIDER: Structure-Preferential Implicit Deep Network for Biplanar X-ray Reconstruction. arXiv preprint arXiv:2507.04684.

Zhu, J.-Y.; Park, T.; Isola, P.; and Efros, A. A. 2017. Unpaired Image-To-Image Translation Using Cycle-Consistent Adversarial Networks. In Proceedings of the IEEE International Conference on Computer Vision, 2223–2232.