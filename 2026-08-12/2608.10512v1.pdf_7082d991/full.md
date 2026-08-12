# Towards Color-Faithful Low-Light Image Enhancement via Adaptive Color Debiasing and Saturation Rectification

Zhichen Yang   
Fuzhou University   
Fuzhou, Fujian, China   
zhichenyang47@gmail.com   
Fusheng Li   
Fuzhou University   
Fuzhou, Fujian, China   
lifusheng.chn@gmail.com   
Rui Xu   
Fuzhou University   
Fuzhou, Fujian, China   
xurui.ryan.chn@gmail.com

Hui Da Fuzhou University Fuzhou, Fujian, China dahuifzu@gmail.com

Yuzhen Niu<sup>∗</sup>   
Fuzhou University   
Fuzhou, Fujian, China   
yuzhenniu@gmail.com

Ri Cheng Fuzhou University Fuzhou, Fujian, China rcheng@fzu.edu.cn

![](images/766b254503b90ce14c491d33e915139dc2d9c5fe723754062f155aef16892552.jpg)  
Figure 1: Motivation of our method. (a→b) Direct brightening makes the color shift pattern inside low-light images more visible. (a→c) Baseline enhancement improves brightness but still leaves a large chromatic gap from the well-lit image, with noticeable color bias, local under- and over-saturation. (a→d) Our method improves color faithfulness by adaptive color debiasing before backbone enhancement and gamut-harmonized saturation rectification after enhancement.

## Abstract

Low-light imaging often introduces color bias caused by the low signal-to-noise ratio and the image formation process. Although recent low-light image enhancement methods have achieved strong brightness recovery, faithful color restoration remains challenging, manifesting as overall color bias together with local under- and over-saturation. To address this issue, we propose CAGE, a cylindrical color correction framework with adaptive color debiasing and gamut-harmonized saturation rectification for color-faithful

<sup>∗</sup>Corresponding Author: yuzhenniu@gmail.com low-light image enhancement. We first introduce AdaLAB, a cylindrical adaptive LAB color space that provides a decoupled and image-specific basis for uniform color correction. Building on this color space, we further develop AdaCCT, an adaptive cylindrical color transform with forward and inverse transforms for the conversion between RGB and AdaLAB color space, as well as necessary color debiasing and saturation rectification. The forward transform suppresses embedded color bias before backbone enhancement by reorganizing the chromatic distribution through chromatic-plane shifting and scaling, while the inverse transform achieves faithful saturation rectification through out-of-gamut lightness compensation. Extensive experiments on multiple benchmarks show that CAGE achieves more faithful color restoration, specifically reduces color bias and saturation abnormality, and delivers better overall visual quality across diferent low-light enhancement backbones. The code is available at https://yangzhichen763.github.io/CAGE/.

## CCS Concepts

• Computing methodologies → Computational photography;   
Image representations; Reconstruction.

## Keywords

Low-light image enhancement; Color correction; Adaptive cylindrical color transform; Color debiasing; Saturation rectification

## 1 Introduction

Images captured under low-light conditions often sufer from degraded visibility, insuficient brightness, and distorted colors [23]. Low-light image enhancement (LLIE) aims to restore visually plausible brightness and faithful colors from these degraded observations. Although recent methods [25, 44, 62] have made steady progress in brightness enhancement, faithful color restoration remains difficult. In low-light imaging, the low signal-to-noise ratio, camera hardware limitations, and automatic in-camera processing often introduce noticeable color shift during image formation [9]. Once the image is brightened, as shown in Fig. 1(b), this embedded color shift becomes more visible, making it hard to restore brightness and faithful color simultaneously.

Most existing LLIE methods [38, 45, 54] restore images directly in the RGB space. In this color space, brightness enhancement and color restoration are tightly coupled, making stronger enhancement lead to more severe color distortion [47], e.g., color shift and saturation abnormality. To ease this coupling, recent methods decouple brightening and color restoration through Retinex-based decomposition [5, 6, 42, 43] or HSV-inspired color spaces (e.g., HSV [59] and HVI [47]). Although these designs separate lightness and chrominance (i.e., color components independent of lightness), they do not remove the chromatic disturbance already embedded in the lowlight input. Instead, this disturbance keeps propagating through the enhancement pipeline and gradually shifts the recovered color distribution away from that of a well-lit image. As illustrated in Fig. 1(c), the enhanced image therefore exhibits an overall color bias, accompanied by local under- and over-saturation.

To further decouple brightness enhancement from color restoration, another line of work revisits LLIE in color spaces such as YUV [4, 14] and Lab-like spaces [2]. These spaces not only decouple lightness and chrominance, more importantly, they also provide a more coherent basis for organizing color shift patterns. In particular, the mapping from RGB to Lab-like spaces preserves the consistency in chromatic direction and continuity of chromatic variation, al lowing colors to be reorganized in a more structured form. This advantage is especially beneficial for low-light image color restoration, where the key dificulty lies not only in decoupling lightness and chrominance, but also in reorganizing disturbed colors toward well-lit distribution. Motivated by this, we revisit color-faithful LLIE in CIELab [17] (denoted by LAB hereafter), whose coherent lightness-chrominance structure ofers a more suitable foundation for modeling and correcting low-light-induced color bias.

However, any color space alone is not enough to remove the chromatic disturbance induced by low-light imaging. As an appropriate color space can reorganize distorted colors, but cannot by itself ❶ correct the embedded color bias and ❷ rectify the abnormal saturation caused by the enhancement process. For the former, the color shift embedded in the low-light image continues to interfere with subsequent restoration and persists even after a color space transformation, leading to biased color restoration (cf. Fig. 1(c), ①). For the latter, the enhancement model tends to amplify chromatic responses in a content-dependent manner, which typically results in the coexistence of under-saturation in some regions and oversaturation in others (cf. Fig. 1(c), ② and ③). Without an explicit mechanism to regulate these behaviors, the chromatic distribution remains inconsistent, and the gap between the enhanced and well-lit images cannot be efectively reduced.

To address these issues, we propose CAGE, a cylindrical Color correction framework with Adaptive color debiasing and Gamutharmonized saturation rectification for low-light image Enhancement. Specifically, we first construct a cylindrical adaptive LAB color space, named AdaLAB, where L and AB inherit the lightness and the two opponent chromatic axes from CIELab respectively, serving as a decoupled and image-specific basis for accommodating uniform color shifts and scaling in the chromatic plane. Built on this color space, we further develop an adaptive cylindrical color transform (AdaCCT), which learns hue-shift magnitude and chromascaling intensity, and adaptively predicts hue-shift direction and lightness sensitivity to reorganize the color distribution before and after backbone enhancement. In the forward transform, AdaCCT suppresses the color bias embedded in the low-light input before backbone enhancement by reorganizing the chromatic distribution. In the inverse transform, AdaCCT rectifies saturation abnormality after backbone enhancement by converting out-of-gamut chroma surplus into lightness compensation rather than direct clipping. With this design, CAGE is compatible with diferent LLIE backbones and performs more faithful color restoration.

Our contributions are summarized as follows:

• We revisit low-light image enhancement (LLIE) from the perspective of embedded chromatic disturbance and reveal that decoupling lightness and chrominance alone cannot eliminate this disturbance, which persists through enhancement and appears as global color bias and saturation abnormality.

• We introduce AdaLAB and AdaCCT as a unified color correction design for LLIE. AdaLAB provides a coherent basis for organizing low-light chromatic variation. Built on AdaLAB, AdaCCT suppresses low-light color bias through adaptive hue shifting and chroma scaling before backbone enhancement, and rectifies post-enhancement saturation abnormality through out-of-gamut lightness compensation.

• Extensive experiments on both standard and more challenging benchmarks show that CAGE consistently improves color fidelity, saturation naturalness, and overall visual quality across diferent low-light image enhancement backbones.

## 2 Related Work

## 2.1 Low-light Image Enhancement

Existing low-light image enhancement methods mainly improve images either in a unified representation [10, 38, 45, 54] or in a decomposed manner, especially in Retinex-based frameworks [5, 18, 42, 53]. It benefits brightness recovery and structure preservation by separating illumination estimation and color correction from scene reconstruction. However, although decomposition weakens the coupling between illumination and reflectance, it does not explicitly correct the color bias embedded in low-light inputs. The embedded color bias is therefore propagated through enhancement, where amplified chromatic responses can disturb the balance between brightness and color and cause local under- or over-saturation.

![](images/75159e08d26ffb2815c12a39f20559f6f68f28cf67243527691c7994a3fa9714.jpg)  
Figure 2: Overview of our proposed framework with a plug-and-play adaptive cylindrical color transform built on the AdaLAB space. AdaCCT predicts image-adaptive cylindrical parameters for the forward and inverse transforms, reducing color bias before backbone enhancement and rectifying saturation abnormality after backbone enhancement.

To further separate brightness from chromatic content, recent methods revisit low-light enhancement in transformed spaces, but these spaces present diferent limitations. YUV [4, 14, 31] ofers a linear separation between lightness and chrominance, yielding a perceptually meaningful luminance component but insuficiently separating the two components. HSV [34, 59] provides stronger separation, but it sufers from structural artifacts, such as hue discontinuity and black-plane noise [48], which destabilize the enhancement results. HVI [47, 48] is built on HSV, whose non-diferentiable transformation makes color trajectories less smooth and less consistent under the same color bias. CIELab [2] instead ofers smoother chromatic trajectories and a more coherent organization of lowlight color shifts. However, a symmetric reversible transform only maps the biased chromatic distribution into a diferent coordinate system. It neither determines the correction direction nor adjusts the distortion magnitude, so the embedded color bias remains unresolved after the transformation.

## 2.2 Color Transform

Color transforms are widely used in image enhancement for their compact and controllable color mapping. Representative forms include tone curves [12, 24, 32], white balance [1], and lookuptable-based transforms [21, 27, 49, 50, 56]. Among them, imageadaptive 3D look-up tables (3D LUTs) [56] enable content-aware color mapping, and later works further extend this formulation with adaptive sampling [49], separable decomposition [50], pixelwise local adjustment [27], and spatial-aware parameterization [21].

However, these methods are mainly designed for general image adjustment rather than low-light-specific color restoration.

Recently, color transforms have also been introduced into lowlight enhancement pipelines [14, 47, 48]. In this setting, they are integrated into decoupled representations to facilitate brightness adjustment and chromatic correction. For example, BreaD [14] adopts YCbCr decoupling for illumination-dominated color correction, while HVI-CIDNet [47] builds a learnable transform in the HVI space. Even so, existing methods still treat color transform mainly as a supporting component, rather than a dedicated mechanism for suppressing embedded color distortion during reconstruction.

## 3 Methodology

In this section, we first give the overview of CAGE (in Sec. 3.1). We then describe the image-adaptive cylindrical parameters predicted for the AdaCCT (in Sec. 3.2). Finally, we present the forward and inverse transforms of AdaCCT and AdaLAB color space (in Sec. 3.3).

## 3.1 Overview

Following previous work [47], CAGE integrates a low-light image enhancement backbone with a forward color transform and an inverse color transform of AdaCCT, which can be formulated as:

$$
\hat { y } = G ^ { - 1 } ( F _ { \theta } ( G ( x ) ) ) ,\tag{1}
$$

where $\boldsymbol { x } \in \mathbb { R } ^ { 3 \times H \times W }$ and $\hat { y } \in \mathbb { R } ^ { 3 \times H \times W }$ denote the input low-light image and the enhanced output image, $F _ { \theta }$ denotes the enhancement backbone, and $G ( \cdot )$ and $G ^ { - 1 } ( \cdot )$ denote the forward and inverse transforms, respectively, which map the image between color spaces. Beyond simple color-space mapping, CAGE further introduces image-adaptive cylindrical parameters in the transformed color space and incorporates color correction into both the forward and inverse transforms to explicitly regulate the chromatic distribution throughout the enhancement pipeline.

Specifically, as illustrated in Fig. 2, given an input image $x ,$ CAGE first predicts image-adaptive cylindrical parameters from the downsampled input image, including lightness sensitivity vertices, chroma-scaling intensity values, and hue-shift direction vector and magnitude values. Based on these parameters, the forward transform maps the input image from RGB to the proposed AdaLAB color space and suppresses the embedded color bias before backbone enhancement. The enhancement backbone then restores visibility and structures in the color space transformed domain. Finally, the inverse transform maps the enhanced result back to RGB while rectifying out-of-gamut saturation.

## 3.2 Image-Adaptive Cylindrical Parameters

Low-light images are formed under varying illumination conditions, camera pipelines, and device responses, where the embedded chromatic disturbance typically does not follow a fixed pattern across images. To make the transform responsive to each input image and its lightness distribution, we predict all image-adaptive parameters from a downsampled input image $\boldsymbol { x } _ { \downarrow 1 2 8 } \in \mathbb { R } ^ { 3 \times 1 2 8 \times 1 2 8 }$ , which reduces the computational overhead. Specifically, a lightweight backbone extracts a compact image vector representation $z \in \bar { \mathbb { R } } ^ { D }$

$$
z = H ( x _ { \downarrow 1 2 8 } ) ,\tag{2}
$$

where $H ( \cdot )$ denotes the lightweight CNN model. Since LAB separates lightness from chromatic plane, and color correction in the chromatic plane can be expressed by directional shift and radial scaling, we then predict three groups of image-adaptive parameters based on � to parameterize a LAB-suited cylindrical color correction, including lightness sensitivity vertices, chroma-scaling intensity values, and hue-shift direction vector and magnitude values.

Lightness Sensitivity Vertices. Human perception is more sensitive to chromatic variation around moderate lightness levels [11]. Natural images generally exhibit a non-uniform distribution along the lightness axis, with pixel values concentrated more heavily in some lightness ranges than others. In low-light enhancement, both embedded color bias and saturation abnormality after enhancement vary with lightness. Therefore, a uniform parameterization on the lightness axis is suboptimal and lacks suficient flexibility to capture this non-uniformity. To adapt the transform to this nonuniformity, inspired by adaptive interval learning [49], we predict image-adaptive lightness intervals from the input image and convert them into monotonically increasing sampling vertices.

Specifically, given the compact representation �, we use a single linear layer as a lightness intervals predictor to predict � interval logits $\{ \hat { v } _ { k } \} _ { k = 1 } ^ { p } ,$ where $\hat { v } _ { k } \in \mathbb { R } .$ . These logits are normalized into nonnegative interval weights summing to 1, and accumulated to obtain the lightness sampling vertices, which can be formulated as:

$$
\tilde { v } _ { k } = \frac { \exp ( \hat { v } _ { k } ) } { \sum _ { j = 1 } ^ { p } \exp ( \hat { v } _ { j } ) } ,\tag{3a}
$$

$$
v _ { 0 } = 0 , \quad v _ { k } = v _ { k - 1 } + \tilde { v } _ { k } , \quad k = 1 , \ldots , p .\tag{3b}
$$

where � denotes the number of lightness intervals, and $\not p + 1$ denotes the number of lightness sensitivity vertices.

![](images/47cc40f03b73be484cb24c1dcaaf11dab65162dd2c085e084d552f168931af0c.jpg)  
Figure 3: Visualization of low-light color shift patterns. The residuals become more coherent in the proposed AdaLAB.

Chroma-Scaling Intensity. Low-light imaging often introduces color distortion that varies with lightness [7], causing pixels at diferent lightness levels to drift toward diferent chromatic ranges (i.e., misalignment of chromatic distributions). A single monotonic scaling curve [47] can reduce this mismatch, but it is not suitable for LAB, where the feasible chroma range is non-monotonic along the lightness axis. A 1D LUT [20] along the lightness axis provides greater flexibility, but is less well suited to low-light image enhancement, since each lightness value depends on only two adjacent samples, leaving sparsely populated high-lightness regions weakly supervised and causing scaling inconsistency across intervals.

To better align chromatic distributions across diferent lightness levels to obtain structured representations in the LAB space, we introduce a group of learned chroma-scaling intensity values with global interaction across all vertices on the lightness axis:

$$
\tilde { c } _ { k } = \sum _ { j = 0 } ^ { p } \hat { c } _ { j } e ^ { - \tau | j - k | } ,\tag{4a}
$$

$$
c _ { k } = { \mathrm { s o f t p l u s } } \left( \tilde { c } _ { k } + 1 \right) ,\tag{4b}
$$

where $\{ \hat { c } _ { k } \} _ { k = 0 } ^ { p }$ denote the learnable scalar parameters with $\hat { c } _ { k } \in \mathbb { R }$ $e ^ { - \tau | j - k | }$ is an exponentially decaying weight over the index distance $| j - k |$ , and � is a learnable temperature controlling the decay rate. softplus(·) is a smooth activation function that keeps $c _ { k }$ positive, and the additional constant 1 moves $c _ { k }$ toward unit scaling.

Hue-shift Direction and Magnitude. Within any single lowlight image, color bias typically appears as a globally coherent shift pattern rather than completely irregular pixel-wise variation [1], which is induced by the low-light imaging process [9]. As shown in Fig. 3, the color diference exhibits a more compact pattern in LAB. Since the LAB color space separates lightness from the chromatic plane, this bias pattern can be described more directly by a shift direction together with a lightness-aware shift magnitude.

Specifically, given the compact representation �, we use a linear layer as a hue-shift direction predictor $\phi _ { \mathrm { d } }$ to predict a shared twodimensional hue-shift direction vector $d = \phi _ { \mathrm { d } } ( z ) \in \mathbb { R } ^ { 2 }$ to model the global shift pattern. Meanwhile, we obtain hue-shift magnitudes $\{ m _ { k } \} _ { k = 0 } ^ { p }$ through the same global interaction as in Eq. (4), and combine them with � to obtain the vertex-wise hue-shift vectors $\{ d _ { k } \} _ { k = 0 } ^ { p } ,$ which can be formulated as:

$$
m _ { k } = \sum _ { j = 0 } ^ { p } \hat { m } _ { j } e ^ { - \tau | j - k | } ,\tag{5a}
$$

$$
d _ { k } = m _ { k } \cdot d\tag{5b}
$$

![](images/869cf35fdcabe3423accbf383df72273f3ac936ba40119b21a68ae917c420cbf.jpg)  
Figure 4: Pipeline of the forward and inverse transforms of the proposed AdaCCT. (a) The forward transform maps the input RGB image to LAB and constructs the AdaLAB representation through hue-directional chromatic debiasing and lightness-aware chroma scaling. (b) The inverse transform reconstructs the enhanced RGB image by reverting the chroma-scaling intensity and applying out-of-gamut lightness compensation before LAB-to-RGB conversion.

where $\{ \hat { m } _ { k } \} _ { k = 0 } ^ { p }$ denote the learnable parameters with $\hat { m } _ { k } \in \mathbb { R }$

## 3.3 Adaptive Cylindrical Color Transform and AdaLAB Color Space

Existing methods [47] build a symmetric color space to address lowlight color distortion. However, these reversible color transforms preserve identity throughout the transformation and do no more than reorganize the representation, leaving the embedded color distortion alone. Consequently, the embedded color bias is carried into the enhancement process, further disrupting the balance between lightness recovery and color restoration, and finally manifests as color bias and abnormal saturation in the restored image.

To address this limitation, we propose AdaCCT together with AdaLAB color space. As shown in Fig. 4, the forward AdaCCT maps the input image from RGB to AdaLAB through RGB-to-LAB con version, hue-directional chromatic debiasing, and lightness-aware chroma scaling, where AdaLAB serves as the working space for subsequent backbone enhancement. After backbone enhancement in AdaLAB, the inverse AdaCCT reconstructs the enhanced RGB image by reverting chroma scaling, applying out-of-gamut lightness compensation, and converting the result from LAB to RGB.

Hue-directional Chromatic Debiasing. To suppress the embedded color bias before backbone enhancement, we incorporate a hue-directional chroma ofset to move the color-shifted representation toward a corrected one.

Specifically, we first transform an input image � ∈ $\mathbb { R } ^ { 3 \times H \times W }$ to the LAB space, yielding a LAB map �<sub>l</sub> ∈ $\mathbb { R } ^ { 3 \times H \times W }$ , which is then separated into a lightness map $l _ { \mathrm { l } } \in \mathbb { R } ^ { 1 \times H \times W }$ and a chrominance map $u _ { 1 } ~ \in ~ \mathbb { R } ^ { 2 \times H \times \check { W } }$ . After that, we apply a linear interpolation of the vertex-wise hue-shift vectors $\bar { \{ d _ { k } \} } _ { k = 0 } ^ { p }$ on the vertices $\{ v \} ^ { p }$ according to the lightness map $l _ { \mathrm { l } }$ to obtain lightness-aware hue-shift vectors $\breve { d _ { 1 } } \in \mathbb { R } ^ { 2 \times H \times W }$ . Then, we compute a similarity-aware weight $s _ { 1 } \in [ \delta _ { 1 } , \delta _ { 2 } ] ^ { 1 \times H \times W }$ between $u _ { \mathrm { l } }$ and $d _ { \mathrm { l } }$ to account for the boundary efect of color bias. Finally, the chrominance map $u _ { \mathrm { l } }$ is shifted along the lightness-aware hue-shift vectors $d _ { \mathrm { l } }$ with similarity weight $s _ { \mathrm { l } } .$

The above process is formulated as:

$$
d _ { \mathrm { l } } = \mathrm { I n t e r p } \left( l _ { \mathrm { l } } ; \{ v _ { k } \} _ { k = 0 } ^ { p } , \{ d _ { k } \} _ { k = 0 } ^ { p } \right) ,\tag{6a}
$$

$$
s _ { \mathrm { l } } = \frac { \delta _ { 2 } - \delta _ { 1 } } { 2 } \cdot \frac { u _ { \mathrm { l } } \cdot d _ { \mathrm { l } } } { \| u _ { \mathrm { l } } \| _ { 2 } \| d _ { \mathrm { l } } \| _ { 2 } } + \frac { \delta _ { 1 } + \delta _ { 2 } } { 2 } ,
$$

$$
\tilde { u } _ { 1 } = u _ { 1 } - s _ { 1 } \cdot d _ { \mathrm { l } } ,\tag{6b}
$$

(6c)

where Interp( $l _ { 1 } ; \{ v \} ^ { p } ; \{ d \} ^ { p } )$ denotes the linear interpolation of the vertex values $\{ d \} ^ { p }$ on the vertices $\{ v \} ^ { p }$ according to the lightness map $l _ { \mathrm { l } } , \delta _ { 1 }$ and $\delta _ { 2 }$ are fixed constants defining the range of the weighted similarity $s _ { 1 } , i . e . , s _ { 1 } \in [ \delta _ { 1 } , \delta _ { 2 } ]$

Lightness-aware Chroma Scaling. To provide a more structured representation for subsequent enhancement, we reorganize colors from diferent lightness planes into a more consistent chromatic range through AdaLAB color transform.

Specifically, we first obtain the lightness-aware chroma-scaling intensity $c _ { 1 } \in \overline { { \mathbb { R } } } ^ { 1 \times H \times W }$ through a linear interpolation ofthe chromascaling intensities $\{ c _ { k } \} _ { k = 0 } ^ { p }$ on the lightness sensitivity vertices $\{ v _ { k } \} _ { k = 0 } ^ { p }$ according to the lightness map $l _ { \mathrm { l } } .$ We then multiply �˜ by � to reorganize the chrominance representation and obtain the AdaLAB chrominance map $\hat { u } _ { 1 } \in \mathbb { R } ^ { 2 \times H \times W }$ , which is formulated as:

$$
c _ { 1 } = \operatorname { I n t e r p } \left( l _ { 1 } ; \left\{ v _ { k } \right\} _ { k = 0 } ^ { p } , \left\{ c _ { k } \right\} _ { k = 0 } ^ { p } \right) ,\tag{7a}
$$

$$
\hat { u } _ { 1 } = c _ { 1 } \tilde { u } _ { 1 } .\tag{7b}
$$

We then merge the reorganized chrominance map $\hat { u } _ { \mathrm { l } }$ and lightness map $l _ { \mathrm { l } }$ into reorganized AdaLAB map $\hat { x } _ { 1 } \in \mathbb { R } ^ { 3 \times H \times W }$ for subsequent enhancement.

After backbone enhancement in the AdaLAB color space, we map the enhanced representation back to the base LAB space by reverting the lightness-aware chroma scaling. Formally, let the enhanced AdaLAB representation be $\boldsymbol { x } _ { \mathrm { h } } \in \mathbb { R } ^ { 3 \times H \times W }$ , which is separated into an enhanced lightness map $\hat { l } _ { \mathrm { h } } \in \mathbb { R } ^ { 1 \times H \times W }$ and an enhanced chrominance map $\hat { u _ { \mathrm { h } } } ^ { \mathrm { ~ \bar { ~ } } } \in \mathbb { R } ^ { 2 \times H \times \bar { W } }$ . We first obtain the lightness-aware chroma-scaling intensity $c _ { \mathrm { h } } \in \mathbb { R } ^ { 1 \times H \times W }$ in the same way as $c _ { \mathrm { l } } ,$ according to the enhanced lightness map $\hat { l } _ { \mathrm { h } }$ . We then divide $\hat { u } _ { \mathrm { h } }$ by $c _ { \mathrm { h } } + \epsilon$ to revert the chrominance representation back and obtain $\tilde { u } _ { \mathrm { h } } \in \mathbb { R } ^ { 2 \times H \times W }$ , which is formulated as:

$$
c _ { \mathrm { h } } = \mathrm { I n t e r p } \left( \hat { l } _ { \mathrm { h } } ; \{ v _ { k } \} _ { k = 0 } ^ { p } , \{ c _ { k } \} _ { k = 0 } ^ { p } \right) ,\tag{8a}
$$

$$
\tilde { u } _ { \mathrm { h } } = \frac { \hat { u } _ { \mathrm { h } } } { c _ { \mathrm { h } } + \epsilon } ,\tag{8b}
$$

where $\epsilon = 1 \times 1 0 ^ { - 1 2 }$ is a small constant for numerical stability.

Out-of-gamut Lightness Compensation. After reverting the chroma scaling, the restored LAB representation is mapped back to RGB. A key issue at this stage is that the restored color may fall outside the valid RGB gamut. Direct gamut clipping would truncate excessive chroma at the boundary and easily create visible color bias artifacts in highly saturated regions, as shown in Fig. 5.

To address this issue, we introduce out-of-gamut lightness compensation (OOGLC). In detail, we first clip the original chrominance map $\tilde { u } _ { \mathrm { h } }$ to the valid gamut according to $\hat { l } _ { \mathrm { h } } ,$ , which yields a clipped chrominance map �<sub>c</sub> $\in \mathbb { R } ^ { 2 \times H \times W }$ . The diference between � and $\tilde { u } _ { \mathrm { h } }$ reflects the amount of chroma that cannot be represented within the valid gamut. Rather than discarding this out-of-gamut chroma, we convert it into lightness compensation to obtain a lightnesssaturation balanced reconstruction. After that, $u _ { \mathrm { c } }$ is clipped to the valid gamut according to $l _ { \mathrm { h } }$ to ensure color validity. The above process can be mathematically expressed as:

$$
u _ { \mathrm { c } } = \mathrm { G a m u t C l i p p i n g } ( \tilde { u } _ { \mathrm { h } } , \hat { l } _ { \mathrm { h } } ) ,\tag{9a}
$$

$$
l _ { \mathrm { h } } = \hat { l } _ { \mathrm { h } } + \gamma \Vert u _ { \mathrm { c } } - \tilde { u } _ { \mathrm { h } } \Vert _ { 2 } ,\tag{9b}
$$

$$
\boldsymbol { u } _ { \mathrm { h } } = \mathrm { G a m u t C l i p p i n g } ( u _ { \mathrm { c } } , l _ { \mathrm { h } } ) ,\tag{9c}
$$

where GamutClipping(·, ·) clips a color along the saturation direction to the valid gamut, and � = 1.0 denotes the compensation ratio. Following previous work [47], we further introduce two customizing parameters, $\alpha _ { c }$ and $\alpha _ { l } ,$ , to adjust image saturation and brightness, respectively:

$$
\hat { y } _ { \mathrm { h } } = [ \alpha _ { \mathrm { l } } l _ { \mathrm { h } } ; \alpha _ { \mathrm { c } } u _ { \mathrm { h } } ] ,\tag{10}
$$

where $[ \cdot ; \cdot ]$ denotes channel-wise concatenation. The final valid LAB map $\hat { y } _ { \mathrm { h } }$ is then converted back to the RGB space through the inverse color space mapping, obtaining the enhanced image �ˆ.

## 3.4 Training Objectives

To constrain restoration in both the AdaLAB and RGB spaces, we supervise the enhanced AdaLAB map �ˆ<sub>AdaLAB</sub> and the restored RGB image �ˆ with their corresponding targets. Specifically, �<sub>AdaLAB</sub> is obtained by mapping the ground-truth image � to the base LAB space and applying the same lightness-aware chroma scaling predicted from the input low-light image �, without the hue-directional chroma ofset. The total loss is defined as:

$$
\boldsymbol { L } _ { \mathrm { t o t a l } } = \lambda \cdot \boldsymbol { L } ( \hat { y } _ { \mathrm { A d a L A B } } , y _ { \mathrm { A d a L A B } } ) + \boldsymbol { L } ( \hat { y } , y ) ,\tag{11}
$$

where � balances the two losses, and $L ( \cdot , \cdot )$ is the reconstruction loss of the baseline network.

## 4 Experiments

## 4.1 Experimental Setup

Implementation Details. We run all experiments on NVIDIA A40 GPUs. The backbone networks used in our method are retrained for fair comparison based on their released code. For the other comparison methods, we directly use the published results when available, and retrain them with their oficial implementations on the datasets where their results are not provided.

![](images/7eced5524da831b2c849b0d46884035f45f7bfa19cf66913ea8a34deb9783de6.jpg)  
Figure 5: Comparison of Out-of-Gamut Handling Strategies.

Baselines. To comprehensively evaluate the efectiveness of CAGE, we compare with representative low-light image enhancement methods [5, 10, 12, 14, 19, 35, 38–40, 42, 45–47, 51, 55, 58, 61]. To further verify the compatibility of CAGE with diferent enhancement backbones, we integrate it into three representative methods, including Retinexformer [5] for Retinex-based enhancement, DarkIR [10] for RGB-space enhancement, and HVI-CIDNet [47] for HSV-inspired color-space enhancement. More details of the integration for each backbone are provided in Appendix A.2.

Datasets and Metrics. We evaluate CAGE on six paired lowlight benchmarks, including LOLv1 [42], LOLv2-real [52], LOLv2- synthetic [52], SDSD-indoor [36], SDSD-outdoor [36], and SID [7]. Following common practice of [5, 47], we use PSNR [16], SSIM [41], and LPIPS [57] to evaluate the restored images on paired datasets. Additional experimental results and setups are provided in Appendix B.

## 4.2 Quantitative Results

To evaluate the efectiveness of CAGE for color-faithful low-light image enhancement, we integrate it into representative enhancement backbones and compare the resulting models with both their original baselines and other competitive methods.

Standard Benchmarks. LOLv1 and LOLv2 are standard paired benchmarks for low-light image enhancement. As shown in Table 1, CAGE consistently improves all backbones across LOLv1 and LOLv2 datasets. Compared with the original backbones, CAGE achieves an average PSNR gain of 0.98 dB / 1.23 dB / 0.81 dB on LOLv1, LOLv2-real, and LOLv2-synthetic, respectively. This improvement is achieved with only 0.07M additional parameters (about 3.5% of HVI-CIDNet) and less than 0.01G FLOPs (less than 0.1% of HVI-CIDNet), indicating that the performance gain mainly comes from improved color modeling rather than increased model capacity. Compared with RGB-based method (i.e., DarkIR), Retinex-based method (i.e., Retinexformer), and even HSV-inspired color-space enhancement (i.e., HVI-CIDNet), CAGE brings substantial improvements on all evaluated datasets. This demonstrates that explicitly reorganizing chromatic distribution provides more stable color restoration under strong enhancement.

Challenging Scenarios. SDSD and SID pose a more dificult test for low-light image enhancement due to large illumination variation and severe low-light degradation with strong color bias.

Table 1: Quantitative comparisons on LOLv1, LOLv2-real, and LOLv2-synthetic datasets.
<table><tr><td rowspan="2">Methods</td><td colspan="2">Complexity Params (M) FLOPs (G)|</td><td rowspan="2">PSNR ↑</td><td colspan="2">LOLv1</td><td colspan="3">LOLv2-real</td><td colspan="3">LOLv2-synthetic</td></tr><tr><td></td><td></td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>RetinexNet [42] BMVC&#x27;18</td><td>0.84</td><td>584.47</td><td>17.56</td><td>0.645</td><td>0.378</td><td>17.68</td><td>0.642</td><td>0.440</td><td>15.61</td><td>0.449</td><td>0.746</td></tr><tr><td>KinD [61] ACM MM&#x27;19</td><td>8.02</td><td>64.73</td><td>17.64</td><td>0.777</td><td>0.175</td><td>20.59</td><td>0.818</td><td>0.143</td><td>16.26</td><td>0.591</td><td>0.435</td></tr><tr><td>DRBN [51] CVPR&#x27;20</td><td>5.27</td><td>48.61</td><td>20.13</td><td>0.830</td><td>0.155</td><td>20.29</td><td>0.831</td><td>0.147</td><td>23.22</td><td>0.927</td><td></td></tr><tr><td>MIRNet [55] ECCV&#x27;20</td><td>31.76</td><td>785.00</td><td>26.56</td><td>0.853</td><td>0.128</td><td>22.68</td><td>0.828</td><td>0.212</td><td>25.05</td><td>0.923</td><td>0.072</td></tr><tr><td>ZeroDCE [12] CVPR&#x27;20</td><td>0.08</td><td>19.01</td><td>21.88</td><td>0.640</td><td>0.335</td><td>16.06</td><td>0.580</td><td>0.313</td><td>17.71</td><td>0.815</td><td>0.169</td></tr><tr><td>EnlightenGAN [19] TIP&#x27;21</td><td>114.35</td><td>61.01</td><td>20.00</td><td>0.691</td><td>0.317</td><td>18.23</td><td>0.617</td><td>0.309</td><td>16.57</td><td>0.734</td><td>0.220</td></tr><tr><td>LLFlow [40] AAAI&#x27;22</td><td>17.42</td><td>358.4</td><td>25.53</td><td>0.870</td><td>0.111</td><td>17.43</td><td>0.831</td><td>0.176</td><td>23.43</td><td>0.933</td><td>0.050</td></tr><tr><td>SNR-Net [46] CVPR&#x27;22</td><td>4.01</td><td>26.35</td><td>26.81</td><td>0.855</td><td>0.150</td><td>21.48</td><td>0.849</td><td>0.157</td><td>24.14</td><td>0.927</td><td>0.056</td></tr><tr><td>BreaD [14] ıJCV&#x27;23</td><td>2.02</td><td>19.85</td><td>25.32</td><td>0.843</td><td>0.155</td><td>23.69</td><td>0.869</td><td>0.156</td><td>15.97</td><td>0.746</td><td>0.261</td></tr><tr><td>FourLLIE [35] ACM MM&#x27;23</td><td>0.12</td><td></td><td>25.05</td><td>0.839</td><td>0.121</td><td>22.35</td><td>0.847</td><td>0.114</td><td>24.65</td><td>0.919</td><td>0.066</td></tr><tr><td>LLFormer [38] AAAI&#x27;23</td><td>24.52</td><td>22.52</td><td>26.12</td><td>0.827</td><td>0.166</td><td>21.65</td><td>0.816</td><td>0.205</td><td>25.26</td><td>0.923</td><td>0.061</td></tr><tr><td>QuadPrior [39] cVPR&#x27;24</td><td>1252.75</td><td>1103.20</td><td>22.85</td><td>0.800</td><td>0.201</td><td>20.59</td><td>0.811</td><td>0.202</td><td>16.11</td><td>0.758</td><td>0.114</td></tr><tr><td>CWNet [58] ICCV&#x27;25</td><td>1.23</td><td>11.30</td><td>26.38</td><td>0.864</td><td>0.120</td><td>21.65</td><td>0.860</td><td>0.136</td><td>25.50</td><td>0.937</td><td>0.044</td></tr><tr><td>Retinexformer [5] ICCV&#x27;23</td><td>1.61</td><td>17.02</td><td>26.08</td><td>0.827</td><td>0.155</td><td>21.89</td><td>0.837</td><td>0.159</td><td>24.59</td><td>0.919</td><td>0.069</td></tr><tr><td>+Ours</td><td>1.68</td><td>17.03</td><td></td><td>27.09 (+1.01) 0.871 (+0.044) 0.090 (+0.065)</td><td></td><td></td><td>24.10 (+2.21) 0.847 (+0.010) 0.104 (+0.055)</td><td></td><td></td><td>26.33 (+1.74) 0.939 (+0.020) 0.039 (+0.030)</td><td></td></tr><tr><td>DarkIR [10] CVPR&#x27;25</td><td>3.32</td><td>7.11</td><td>25.32</td><td>0.806</td><td>0.106</td><td>21.32</td><td>0.830</td><td>0.139</td><td>24.06</td><td>0.921</td><td>0.060</td></tr><tr><td>+Ours</td><td>3.39</td><td>7.12</td><td></td><td>26.03 (+0.71) 0.849 (+0.043) 0.106 (+0.000)</td><td></td><td></td><td>22.51 (+1.19) 0.860 (+0.030) 0.121 (+0.018)</td><td></td><td></td><td>24.54(+0.48) 0.923 (+0.002) 0.058 (+0.002)</td><td></td></tr><tr><td>HVI-CIDNet [47] CVPR&#x27;25</td><td>1.98</td><td>8.13</td><td>26.03</td><td>0.832</td><td>0.101</td><td>23.94</td><td>0.870</td><td>0.113</td><td>25.32</td><td>0.929</td><td>0.054</td></tr><tr><td>+Ours</td><td>2.05</td><td>8.14</td><td></td><td>27.25 (+1.22) 0.857 (+0.025) 0.083 (+0.018)</td><td></td><td></td><td>24.24 (+0.30) 0.875 (+0.005) 0.098 (+0.015)</td><td></td><td></td><td>25.52 (+0.20) 0.930 (+0.001) 0.046 (+0.008)</td><td></td></tr></table>

The best and second-best results are highlighted in bold and underline, respectively. +(-) denotes the improvement (reduction) of performance, corresponding to ↑ (↓).

![](images/35317b20dc3716519c44a469e0aac2d86716faa993043a8c111d7787101dfa6b.jpg)  
Figure 6: Qualitative comparison of representative methods and baseline methods with and without our proposed CAGE on LOLv1 dataset (first row) and LOLv2-real dataset (second row). Our CAGE empowers baseline methods to produce images with less color distortion, e.g., less saturation abnormality in the first row, less color bias in the second row.

As reported in Table 2, CAGE achieves consistent improvement on both SDSD and SID datasets. Compared with strong baselines such as DarkIR, Retinexformer, and HVI-CIDNet, CAGE improves PSNR by 1.12 dB on SDSD-indoor, 0.73 dB on SDSD-outdoor, and 0.56 dB on SID. This indicates that the proposed cylindrical transform efectively aligns chromatic distributions across lightness levels, thereby mitigating instability from spatially varying illumination. Concurrently, suppressing embedded color bias is shown to be essential for scenes with extreme lighting disparity.

## 4.3 Qualitative Results

To further verify the visual benefit of CAGE, we compare the restored images on real-world low-light scenes. The visual comparisons on LOLv1 and LOLv2-real are shown in Fig. 6. Compared with the original backbone outputs, the images enhanced by CAGE show less global color bias, more stable local saturation, and more natural chromatic transitions in dark regions and detail-rich regions. In scenes with visible color cast, CAGE variants (i.e., CAGE-Retinexformer, CAGE-DarkIR, and CAGE-CIDNet) restore more faithful neutral regions and suppress the residual tint left by the baseline backbone. In scenes with strong enhancement, CAGE also reduces washed-out colors and oversaturated regions, which leads to a more balanced visual result. More visual comparisons are provided in Appendix B.3.

## 4.4 Ablation Study

We conduct several ablation experiments on real-world dataset (i.e., LOLv2-real) to evaluate the contribution of each component in CAGE and analyze their impact on color-faithful low-light image enhancement.

Cylindrical Adaptive Parameters. Table 3 shows that the full parameter set (Table 3 ④) achieves the best results on all metrics. Removing hue-shift parameters (Table 3 ①) causes the largest drop in PSNR and SSIM, reducing PSNR by 0.4862 dB and SSIM by 0.0107, while increasing LPIPS by 0.0003. Removing chroma-scaling intensity (Table 3 ②) leads to a smaller PSNR drop of 0.2118 dB and SSIM by 0.0012, but causes the largest LPIPS increase of 0.0069. Removing lightness sensitivity (Table 3 ③) keeps PSNR close to that of the full model, but still leads to inferior SSIM and LPIPS. These results show that hue-shift correction is the key part for suppressing low-light color bias, while chroma scaling and lightness sensitivity further improve structural consistency and perceptual quality.

Table 2: Quantitative comparisons on challenging datasets.
<table><tr><td>Methods</td><td>SDSD-indoor PSNR SSIM LPIPS</td><td>SDSD-outdoor PSNR SSIM LPIPS</td><td>PSNR SSIM LPIPS</td><td>SID</td></tr><tr><td>3DLUT [56]</td><td>|24.780.718</td><td>23.290.703</td><td>16.970.591</td><td></td></tr><tr><td>SNR-Net [46]</td><td>26.130.815 0.198</td><td>19.22 0.657 0.225</td><td>21.35</td><td>0.550 0.446</td></tr><tr><td>LEDNet [63]</td><td>27.29 0.876 0.121</td><td>26.66 0.850 0.145</td><td>21.47</td><td>0.638 0.349</td></tr><tr><td>FourLLIE [35]</td><td>24.740.826 0.146</td><td>24.67 0.787 0.158</td><td>18.42</td><td>0.513 0.504</td></tr><tr><td>RetinexMamba [3]</td><td>28.440.894 0.119</td><td>28.52 0.859 0.180</td><td>22.45</td><td>0.6560.351</td></tr><tr><td>MIRNet [55]</td><td>28.640.888 0.128</td><td>28.99 0.869 0.152</td><td>21.36</td><td>0.632 0.397</td></tr><tr><td>Restormer [54]</td><td>28.490.8920.124</td><td>27.99 0.868 0.189</td><td>22.01</td><td>0.6450.361</td></tr><tr><td>MambaIR [13]</td><td>25.140.876 0.130</td><td>27.53 0.851 0.177</td><td>22.02</td><td>0.658 0.337</td></tr><tr><td>CWNet [58]</td><td>30.28 0.904 0.127</td><td>28.54 0.857 0.154</td><td>20.97</td><td>0.623 0.429</td></tr><tr><td>Retinexformer [5]</td><td>28.200.877 0.137</td><td>29.57 0.874 0.122</td><td></td><td>22.050.6360.372</td></tr><tr><td>+Ours</td><td>29.98 0.906 0.072</td><td>30.12 0.877 0.100</td><td></td><td>22.64 0.660 0.281</td></tr><tr><td>Δ</td><td>(+1.78)(+0.029)(+0.065)</td><td>(+0.55)(+0.003)(+0.022)</td><td>(+0.59)</td><td>(+0.024) (+0.091)</td></tr><tr><td>DarkIR [10]</td><td>|29.940.897 0.096</td><td>|28.810.848 0.104</td><td>|21.500.588</td><td>0.469</td></tr><tr><td>+Ours</td><td>30.36 0.899 0.080</td><td>29.36 0.858 0.101</td><td>22.33</td><td>0.639 0.305</td></tr><tr><td>Δ</td><td>(+0.42)(+0.002)(+0.016)</td><td>(+0.55)(+0.010)(+0.003)</td><td>(+0.83)</td><td>(+0.051) (+0.164)</td></tr><tr><td>HVI-CIDNet [47]</td><td>|28.740.895 0.090</td><td>|28.700.866 0.107</td><td>22.21</td><td>0.631 0.308</td></tr><tr><td>+Ours</td><td>29.91 0.900 0.083</td><td>29.80 0.877 0.097</td><td>22.46</td><td>0.653 0.286</td></tr><tr><td>Δ</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>(+1.17)(+0.005)(+0.007)</td><td>(+1.10)(+0.011) (+0.010)</td><td>(+0.25)</td><td>(+0.022)(+0.022)</td></tr></table>

The best and second-best results are highlighted in bold and underline, respectively. +(-) denotes the improvement (reduction) of performance, corresponding to ↑ (↓).

AdaLAB Color Space. Our proposed AdaLAB color space provides a more coherent basis than YUV and the HSV-inspired (i.e., HVI) space for modeling low-light-induced color shift. As shown in Table 4, AdaLAB (Table 4 ⑤) achieves the best results among all com pared spaces, reaching 24.2355 PSNR, 0.8746 SSIM, and 0.0977 LPIPS. Among the baseline spaces, LAB (Table 4 ④) gives the strongest performance, surpassing RGB (Table 4 ①) and YUV (Table 4 ②) by clear margins. Compared with HVI (Table 4 ③), LAB improves PSNR by 0.0807 dB and SSIM by 0.0006, while reducing LPIPS by 0.0079. Built on LAB, AdaLAB further improves PSNR by 0.2184 dB, SSIM by 0.0028, and reduces LPIPS by 0.0072. These results indicate that LAB provides a better basis for color debiasing by more efectively decoupling lightness and chrominance, while AdaLAB further improves chromatic restoration through lightness-aware chroma alignment. As shown in Fig. 7, AdaCCT in AdaLAB yields more faithful colors.

Gamut Harmonization. As shown in Table 8, replacing RGB gamut clipping (Table 8 ①) with LAB gamut clipping (Table 8 ②) significantly improves the results, and out-of-gamut lightness compensation (Table 8 ③) further improves them to 24.2355 PSNR, 0.8746 SSIM, and 0.0977 LPIPS. These results show that simple gamut clipping fails to recover chroma adequately, whereas outof-gamut lightness compensation achieves more faithful chroma recovery.

Table 3: Ablation of image-adaptive cylindrical parameters.
<table><tr><td>#</td><td>Hue P.</td><td>Chroma P.</td><td>Lightness P.</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>①</td><td></td><td>√</td><td>√</td><td>23.7493</td><td>0.8639</td><td>0.0980</td></tr><tr><td>②</td><td>√</td><td></td><td>√</td><td>24.0237</td><td>0.8734</td><td>0.1046</td></tr><tr><td>③</td><td>√</td><td>√</td><td></td><td>24.2201</td><td>0.8667</td><td>0.1006</td></tr><tr><td>④</td><td>√</td><td>√</td><td>√</td><td>24.2355</td><td>0.8746</td><td>0.0977</td></tr></table>

Hue P., Chroma P., and Lightness P. denote Hue-shift Direction and Magnitude, Chroma-scaling Intensity, and Lightness Sensitivity Vertices, respectively.

Table 4: Ablation of color space design.
<table><tr><td>#</td><td>Color Space</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>①</td><td>RGB color space</td><td>22.3870</td><td>0.8334</td><td>0.1585</td></tr><tr><td>②</td><td>YUV color space</td><td>23.2105</td><td>0.8630</td><td>0.1117</td></tr><tr><td>③</td><td>HVI color space</td><td>23.9364</td><td>0.8712</td><td>0.1128</td></tr><tr><td>④</td><td>CIELab color space</td><td>24.0171</td><td>0.8718</td><td>0.1049</td></tr><tr><td>⑤</td><td>AdaLAB color space</td><td>24.2355</td><td>0.8746</td><td>0.0977</td></tr></table>

Table 5: Ablation of gamut harmonization on AdaLAB.
<table><tr><td>#</td><td>Gamut Handling</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>①</td><td>RGB gamut clipping</td><td>23.4189</td><td>0.8640</td><td>0.1002</td></tr><tr><td>②</td><td>LAB gamut clipping</td><td>24.2272</td><td>0.8689</td><td>0.0996</td></tr><tr><td>③</td><td>Out-of-gamut lightness compensation</td><td>24.2355</td><td>0.8746</td><td>0.0977</td></tr></table>

![](images/09516a0f220bea7b0bf23b48676c7a08bd4294bc506a16b5efc8b5237abe78fe.jpg)  
Figure 7: Visual comparison of diferent color spaces for LLIE.

## 5 Conclusion

In this paper, we revisit low-light image enhancement from the perspective of embedded chromatic disturbance, and show that lightness-chrominance decoupling alone is insuficient to eliminate the color bias inherited from low-light inputs. To address this issue, we propose CAGE, a cylindrical color correction framework built upon the proposed AdaLAB color space and the adaptive transform AdaCCT. The forward transform suppresses the embedded color bias before backbone enhancement by reorganizing the chromatic distribution, while the inverse transform stabilizes color reconstruction through out-of-gamut lightness compensation. Extensive experiments on multiple benchmarks demonstrate that CAGE consistently improves color fidelity, alleviates both global color bias and local saturation abnormality, and enhances overall visual quality across diferent low-light enhancement backbones. These results verify that explicitly modeling and correcting chromatic disturbance in a decoupled cylindrical representation provides a more reliable solution for color-faithful low-light image enhancement.

## Acknowledgments

This work was supported in part by the National Natural Science Foundation of China under Grant 62471142, in part by the Industry-Academy Cooperation Project under Grant 2024H6006, in part by the 2025 Special Fund Project for Promoting High-Quality Development of Marine and Fishery Industries in Fujian Province under Grant FJHYF-L-2025-07-004, and in part by the Fujian Provincial Science and Technology Program Project under Grant 2026Y0090.

## References

[1] Mahmoud Afifi, Luxi Zhao, Abhijith Punnappurath, Mohamed A Abdelsalam, Ran Zhang, and Michael S Brown. 2025. Time-aware auto white balance in mobile photography. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 5038–5047.

[2] Yousef Atoum, Mao Ye, Liu Ren, Ying Tai, and Xiaoming Liu. 2020. Colorwise attention network for low-light image enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 506–507.

[3] Jiesong Bai, Yuhao Yin, Qiyuan He, Yuanxian Li, and Xiaofeng Zhang. 2024. RetinexMamba: Retinex-based Mamba for low-light image enhancement. In Proceedings of the International Conference on Neural Information Processing. Springer, 427–442.

[4] Alexandru Brateanu, Raul Balmez, Adrian Avram, Ciprian Orhei, and Cosmin Ancuti. 2025. LYT-NET: Lightweight YUV transformer-based network for low light image enhancement. IEEE Signal Processing Letters 32 (2025), 2065–2069.

[5] Yuanhao Cai, Hao Bian, Jing Lin, Haoqian Wang, Radu Timofte, and Yulun Zhang. 2023. Retinexformer: One-stage Retinex-based transformer for low-light image enhancement. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 12504–12513.

[6] Luyang Cao, Han Xu, Jian Zhang, Lei Qi, Jiayi Ma, Yinghuan Shi, and Yang Gao. 2025. Towards perfection: Building inter-component mutual correction for Retinex-based low-light image enhancement. In Proceedings of the ACM International Conference on Multimedia. 9549–9558.

[7] Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun. 2018. Learning to see in the dark. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 3291–3300.

[8] Ziteng Cui, Kunchang Li, Lin Gu, Shenghan Su, Peng Gao, Zhengkai Jiang, Yu Qiao, and Tatsuya Harada. 2022. You only need 90k parameters to adapt light: A light weight transformer for image enhancement and exposure correction, In Proceedings of the British Machine Vision Conference. arXiv preprint arXiv:2205.14871.

[9] Dan Ding, Feng Shi, and Ye Li. 2025. Low-illumination color imaging: Progress and challenges. Optics and Laser Technology 184 (2025), 112553.

[10] Daniel Feijoo, Juan C Benito, Alvaro Garcia, and Marcos V Conde. 2025. DarkIR: Robust low-light image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10879–10889.

[11] Theo Gevers, Arjan Gijsenij, Joost Van de Weijer, and Jan-Mark Geusebroek. 2012. Color in computer vision: Fundamentals and applications. John Wiley & Sons.

[12] Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou, Sam Kwong, and Runmin Cong. 2020. Zero-reference deep curve estimation for low light image enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1780–1789.

[13] Hang Guo, Jinmin Li, Tao Dai, Zhihao Ouyang, Xudong Ren, and Shu-Tao Xia. 2024. MambaIR: A simple baseline for image restoration with state-space model. In Proceedings ofthe European Conference on Computer Vision. 222–241.

[14] Xiaojie Guo and Qiming Hu. 2023. Low-light image enhancement via breaking down the darkness. International Journal ofComputer Vision 131 (2023), 48–66.

[15] Xiaojie Guo, Yu Li, and Haibin Ling. 2016. LIME: Low-light image enhancement via illumination map estimation. IEEE Transactions on Image Processing 26 (2016), 982–993.

[16] Quan Huynh-Thu and Mohammed Ghanbari. 2008. Scope of validity of PSNR in image/video quality assessment. Electronics Letters 44 (2008), 800–801.

[17] International Commission on Illumination and International Organization for Standardization. 2019. Colorimetry — Part 4: CIE 1976 L\*a\*b\* Colour Space. ISO and CIE.

[18] Hai Jiang, Ao Luo, Xiaohong Liu, Songchen Han, and Shuaicheng Liu. 2024. LightenDifusion: Unsupervised low-light image enhancement with latent-Retinex difusion models. In Proceedings ofthe European Conference on Computer Vision. 161–179.

[19] Yifan Jiang, Xinyu Gong, Ding Liu, Yu Cheng, Chen Fang, Xiaohui Shen, Jian chao Yang, Pan Zhou, and Zhangyang Wang. 2021. EnlightenGAN: Deep light enhancement without paired supervision. IEEE Transactions on Image Processing 30 (2021), 2340–2349.

[20] Hanul Kim, Su-Min Choi, Chang-Su Kim, and Yeong Jun Koh. 2021. Representative color transform for image enhancement. In Proceedings of the IEEE/CVF

International Conference on Computer Vision. 4459–4468.

[21] Wontae Kim, Keuntek Lee, and Nam Ik Cho. 2025. Lightweight and fast real-time image enhancement via decomposition of the spatial-aware lookup tables. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 11895– 11905.

[22] Chulwoo Lee, Chul Lee, and Chang-Su Kim. 2013. Contrast enhancement based on layered diference representation of 2D histograms. IEEE Transactions on Image Processing 22 (2013), 5372–5384.

[23] Chongyi Li, Chunle Guo, Linghao Han, Jun Jiang, Ming-Ming Cheng, Jinwei Gu, and Chen Change Loy. 2021. Low-light image and video enhancement using deep learning: A survey. IEEE transactions on pattern analysis and machine intelligence 44, 12 (2021), 9396–9416.

[24] Chongyi Li, Chunle Guo, and Chen Change Loy. 2021. Learning to enhance low-light image via zero-reference deep curve estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence 44 (2021), 4225–4238.

[25] Chongyi Li, Chun-Le Guo, Man Zhou, Zhexin Liang, Shangchen Zhou, Ruicheng Feng, and Chen Change Loy. 2023. Embedding fourier for ultra-high-definition low-light image enhancement. In Proceedings of the International Conference on Learning Representations.

[26] Risheng Liu, Long Ma, Jiaao Zhang, Xin Fan, and Zhongxuan Luo. 2021. Retinex inspired unrolling with cooperative prior architecture search for low-light image enhancement. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10561–10570.

[27] Junyu Lou, Xiaorui Zhao, Kexuan Shi, and Shuhang Gu. 2025. Learning pixeladaptive multi-layer perceptrons for real-time image enhancement. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 14095–14105.

[28] Kede Ma, Kai Zeng, and Zhou Wang. 2015. Perceptual quality assessment for multi-exposure image fusion. IEEE Transactions on Image Processing 24 (2015), 3345–3356.

[29] Anish Mittal, Anush Krishna Moorthy, and Alan Conrad Bovik. 2012. Noreference image quality assessment in the spatial domain. IEEE Transactions on Image Processing 21 (2012), 4695–4708.

[30] Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. 2012. Making a “completely blind” image quality analyzer. IEEE Signal Processing Letters 20 (2012), 209–212.

[31] Yuzhen Niu, Fusheng Li, Yuezhou Li, Siling Chen, and Yuzhong Chen. 2025. Adaptive luminance enhancement and high-fidelity color correction for low-light image enhancement. IEEE Transactions on Computational Imaging 11 (2025), 732–747.

[32] David Serrano-Lozano, Luis Herranz, Michael S Brown, and Javier Vazquez-Corral. 2024. NamedCurves: Learned image enhancement via color naming. In Proceedings of the European Conference on Computer Vision. 92–108.

[33] Vassilios Vonikakis, Rigas Kouskouridas, and Antonios Gasteratos. 2018. On the evaluation of illumination compensation algorithms. Multimedia Tools and Applications 77 (2018), 9211–9231.

[34] Chenxi Wang and Zhi Jin. 2023. Brighten-and-colorize: A decoupled network fo customized low-light image enhancement. In Proceedings ofthe ACM International Conference on Multimedia. 8356–8366.

[35] Chenxi Wang, Hongjun Wu, and Zhi Jin. 2023. FourLLIE: Boosting low-light image enhancement by fourier frequency information. In Proceedings ofthe ACM International Conference on Multimedia. 7459–7469.

[36] Ruixing Wang, Xiaogang Xu, Chi-Wing Fu, Jiangbo Lu, Bei Yu, and Jiaya Jia. 2021. Seeing dynamic scene in the dark: A high-quality video dataset with mechatronic alignment. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 9700–9709.

[37] Shuhang Wang, Jin Zheng, Hai-Miao Hu, and Bo Li. 2013. Naturalness preserved enhancement algorithm for non-uniform illumination images. IEEE Transactions on Image Processing 22 (2013), 3538–3548.

[38] Tao Wang, Kaihao Zhang, Tianrun Shen, Wenhan Luo, Bjorn Stenger, and Tong Lu. 2023. Ultra-high-definition low-light image enhancement: A benchmark and transformer-based method. In Proceedings ofthe AAAI Conference on Artificial Intelligence. 2654–2662.

[39] Wenjing Wang, Huan Yang,Jianlong Fu, andJiaying Liu. 2024. Zero-reference lowlight enhancement via physical quadruple priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 26057–26066.

[40] Yufei Wang, Renjie Wan, Wenhan Yang, Haoliang Li, Lap-Pui Chau, and Alex Kot. 2022. Low-light image enhancement with normalizing flow. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 36. 2604–2612.

[41] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing 13 (2004), 600–612.

[42] Chen Wei, Wenjing Wang, Wenhan Yang, and Jiaying Liu. 2018. Deep Retinex decomposition for low-light enhancement. In Proceedings of the British Machine Vision Conference.

[43] Wenhui Wu, Jian Weng, Pingping Zhang, Xu Wang, Wenhan Yang, and Jianmin Jiang. 2022. URetinex-Net: Retinex-based deep unfolding network for low-light image enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5901–5910.

[44] Rui Xu, Yuezhou Li, Yuzhen Niu, Huangbiao Xu, Yuzhong Chen, and Tiesong Zhao. 2024. Bilateral interaction for local-global collaborative perception in low-light image enhancement. IEEE Transactions on Multimedia 26 (2024), 10792– 10804.

[45] Rui Xu, Yuzhen Niu, Yuezhou Li, Huangbiao Xu, Wenxi Liu, and Yuzhong Chen. 2025. URWKV: Unified RWKV model with multi-state perspective for low-light image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21267–21276.

[46] Xiaogang Xu, Ruixing Wang, Chi-Wing Fu, and Jiaya Jia. 2022. SNR-aware lowlight image enhancement. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 17714–17724.

[47] Qingsen Yan, Yixu Feng, Cheng Zhang, Guansong Pang, Kangbiao Shi, Peng Wu, Wei Dong, Jinqiu Sun, and Yanning Zhang. 2025. HVI: A new color space for low-light image enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5678–5687.

[48] Qingsen Yan, Kangbiao Shi, Yixu Feng, Tao Hu, Peng Wu, Guansong Pang, and Yanning Zhang. 2025. HVI-CIDNet+: Beyond extreme darkness for low-light image enhancement. arXiv preprint arXiv:2507.06814 (2025).

[49] Canqian Yang, Meiguang Jin, Xu Jia, Yi Xu, and Ying Chen. 2022. AdaInt: Learning adaptive intervals for 3D lookup tables on real-time image enhancement. In Proceedings ofthe IEEE/CVFConference on ComputerVision andPattern Recognition. 17522–17531.

[50] Canqian Yang, Meiguang Jin, Yi Xu, Rui Zhang, Ying Chen, and Huaida Liu. 2022. SepLUT: Separable image-adaptive lookup tables for real-time image enhancement. In Proceedings ofthe European Conference on Computer Vision. 201–217.

[51] Wenhan Yang, Shiqi Wang, Yuming Fang, Yue Wang, and Jiaying Liu. 2020. From fidelity to perceptual quality: A semi-supervised approach for low-light image enhancement. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 3063–3072.

[52] Wenhan Yang, Wenjing Wang, Haofeng Huang, Shiqi Wang, and Jiaying Liu. 2021. Sparse gradient regularized deep Retinex network for robust low-light image enhancement. IEEE Transactions on Image Processing 30 (2021), 2072–2086.

[53] Xunpeng Yi, Han Xu, Hao Zhang, Linfeng Tang, and Jiayi Ma. 2025. Dif-Retinex++: Retinex-driven reinforced difusion model for low-light image enhancement. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025), 6823–6841.

[54] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. 2022. Restormer: Eficient transformer for high-resolution image restoration. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5728–5739.

[55] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao. 2020. Learning enriched features for real image restoration and enhancement. In Proceedings of the European Conference on Computer Vision. 492–511.

[56] Hui Zeng, Jianrui Cai, Lida Li, Zisheng Cao, and Lei Zhang. 2020. Learning image-adaptive 3d lookup tables for high performance photo enhancement in real-time. IEEE Transactions on Pattern Analysis and Machine Intelligence 44 (2020), 2058–2073.

[57] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings ofthe IEEE/CVFConference on ComputerVision andPattern Recognition. 586–595.

[58] Tongshun Zhang, Pingping Liu, Yubing Lu, Mengen Cai, Zijian Zhang, Zhe Zhang, and Qiuzhan Zhou. 2025. CWNet: Causal wavelet network for low-light image enhancement. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 8789–8799.

[59] Yu Zhang, Xiaoguang Di, Bin Zhang, Ruihang Ji, and Chunhui Wang. 2021. Better than reference in low-light image enhancement: Conditional re-enhancement network. IEEE Transactions on Image Processing 31 (2021), 759–772.

[60] Yonghua Zhang, Xiaojie Guo, Jiayi Ma, Wei Liu, and Jiawan Zhang. 2021. Beyond brightening low-light images. International Journal ofComputer Vision 129 (2021), 1013–1037.

[61] Yonghua Zhang, Jiawan Zhang, and Xiaojie Guo. 2019. Kindling the darkness: A practical low-light image enhancer. In Proceedings ofthe ACM International Conference on Multimedia. 1632–1640.

[62] Han Zhou, Wei Dong, Xiaohong Liu, Shuaicheng Liu, Xiongkuo Min, Guangtao Zhai, and Jun Chen. 2024. GLARE: Low light image enhancement via generative latent feature based codebook retrieval. In Proceedings ofthe European Conference on Computer Vision. 36–54.

[63] Shangchen Zhou, Chongyi Li, and Chen Change Loy. 2022. LEDNet: Joint lowlight enhancement and deblurring in the dark. In Proceedings of the European Conference on Computer Vision. 573–589.

This appendix presents two central aspects of CAGE: i) the design rationale of the proposed adaptive cylindrical color transform (AdaCCT) in Sec. A, and further explains the adaptive interpo lation on the lightness axis in Sec. A.1 and the integration into diferent low-light enhancement frameworks in Sec. A.2. ii) the framework generalization of the proposed color modeling with training details in Sec. B.1, including experimental environment and hyperparameter configuration, dataset details in Sec. B.2, more benchmarking results in Sec. B.3, where our proposed CAGE further demonstrates strong generalization capability and clear performance gains on no-reference low-light benchmarks, and more ablation results in Sec. B.4.

## A Design Details and Visual Analyses

Rather than preserving identity throughout a reversible transformation, AdaCCT is built to correct the embedded chromatic disturbance in low-light images. In the forward transform, the input image is mapped from RGB to AdaLAB through RGB-to-LAB conversion, hue-directional chromatic debiasing, and lightness-aware chroma scaling, so that the embedded color bias is reduced before back bone enhancement. In the inverse transform, the enhanced result is mapped back to RGB by reverting chroma scaling, applying outof-gamut lightness compensation, and converting the result from LAB to RGB. The same asymmetry is kept in the AdaLAB-space supervision, where the target $y _ { \mathrm { A d a L A B } }$ shares the lightness-aware chroma scaling predicted from the input low-light image $x ,$ but does not include the hue-directional chroma ofset.

To further unpack the design of AdaCCT, we first revisit the color-space path underlying AdaLAB. As shown in Fig. 8, the RGB to-HSV route can separate brightness from chromatic content, but the conversion also brings hue discontinuity and black-plane noise, and the cylindrical HSV form further exhibits clear low-lightness disparity. By comparison, as reflected in Fig. 8(b), the one-step RGB-to-LAB conversion preserves a more uniform and structurally coherent organization of RGB-sampled points, whereas HSV becomes non-uniform and introduces low-lightness disparity, mak ing LAB a cleaner basis for building AdaLAB and introducing the image-adaptive cylindrical parameters. The remaining parts of this section complement this overview from two aspects: the interpolation from lightness vertices to pixel-wise values, and the integration of AdaCCT into diferent enhancement frameworks.

## A.1 Lightness-aware Linear Interpolation

The lightness-aware linear interpolation can be described as the linear interpolation ofthe vertex values on the vertices according to the lightness map.

As shown in Fig. 9, given a lightness map $l \in \mathbb { R } ^ { 1 \times H \times W }$ , lightness sensitivity vertices $\{ v _ { k } \} _ { k = 0 } ^ { p } .$ , and vertex values $\{ r _ { k } \} _ { k = 0 } ^ { p } ,$ , the interpolation operator produces a dense lightness-aware map $\boldsymbol { a } \in \mathbb { R } ^ { C \times H \times W }$ which can be formulated as:

$$
a = { \mathrm { I n t e r p } } ( l ; \{ v _ { k } \} _ { k = 0 } ^ { p } , \{ r _ { k } \} _ { k = 0 } ^ { p } ) ,\tag{12}
$$

where $C = 2$ for hue-shift vectors and $C = 1$ for chroma-scaling intensity. For each pixel location $( i , j )$ , let $l _ { i j }$ denote the lightness value at that pixel. We first locate the two adjacent lightness sensitivity vertices surrounding $l _ { i j }$ along the lightness axis. The index of the lower vertex is denoted by $q _ { i j } ,$ where $q _ { i j } = \operatorname* { m a x } \{ k \mid v _ { k } \leq$ $l _ { i j } , 0 \le k \le p - 1 \}$ . Since $v _ { 0 } = 0$ and ${ v _ { p } } = 1 _ { : }$ , each lightness value lies in the interval $[ v _ { q _ { i j } } , v _ { q _ { i j } + 1 } ]$ defined by these two adjacent vertices. We then compute the relative position of $\cdot \ l _ { i j }$ within this interval using the interpolation weight $\begin{array} { r } { t _ { i j } = \frac { l _ { i j } - v _ { q _ { i j } } } { v _ { q _ { i j } + 1 } - v _ { q _ { i j } } } } \end{array}$ . Based on this relative position, the pixel-wise value is linearly interpolated from the two neighboring vertex values as $a _ { i j } = \bigl ( 1 - t _ { i j } \bigr ) r _ { q _ { i j } } + t _ { i j } r _ { q _ { i j } + 1 }$ . When $r _ { k }$ is vector-valued, the same interpolation is applied channel-wise.

With this formulation, the same operator is shared by diferent parts of AdaCCT. In the forward transform, the lightness-aware hueshift vectors $d _ { l }$ and the lightness-aware chroma-scaling intensity $c _ { l }$ are obtained as

$$
d _ { l } = \mathrm { I n t e r p } ( l _ { l } ; \{ v _ { k } \} _ { k = 0 } ^ { p } , \{ d _ { k } \} _ { k = 0 } ^ { p } ) ,\tag{13a}
$$

$$
\begin{array} { r } { c _ { l } = \mathrm { I n t e r p } ( l _ { l } ; \{ v _ { k } \} _ { k = 0 } ^ { p } , \{ c _ { k } \} _ { k = 0 } ^ { p } ) . } \end{array}\tag{13b}
$$

In the inverse transform, the same interpolation is applied to the enhanced lightness map $\hat { l } _ { h }$ to obtain

$$
{ \boldsymbol { c } } _ { h } = { \mathrm { I n t e r p } } ( { \hat { l } } _ { h } ; \{ v _ { k } \} _ { k = 0 } ^ { p } , \{ c _ { k } \} _ { k = 0 } ^ { p } ) .\tag{14}
$$

## A.2 Integration Strategies for Diference LLIE Frameworks

Fig. 10 illustrates how CAGE is integrated into three representative LLIE framework paradigms, including the lightness-chrominance joint enhancement framework, the illumination-dominated Retinexbased enhancement framework, and the chrominance-dominated HSV-inspired color-space enhancement framework. Rather than changing the main enhancement branches, CAGE adapts the original pipeline by introducing the forward transform � and inverse transform $G ^ { - 1 }$ around the enhancement process. In particular, the input image is first mapped into the transformed space for lightness and chromatic processing, and the enhanced transformed output is then mapped back to RGB, with the original joint enhancement, branch interaction, or post-reconstruction processing preserved according to the corresponding framework design.

Lightness-Chrominance Joint Enhancement Framework. RGB-space enhancement [10] designs provide a representative integration setting for this paradigm. The original RGB enhancement pipeline is replaced by the transformed-space pipeline of CAGE. The input image � is first mapped into the transformed space by the forward transform, the backbone network then enhances the transformed representation jointly, and the enhanced result is finally mapped back to RGB by the inverse transform:

$$
\boldsymbol { \hat { y } } = \boldsymbol { G } ^ { - 1 } ( \boldsymbol { F } ( \boldsymbol { G } ( \boldsymbol { x } ) ) ) ,\tag{15}
$$

where $x _ { 1 } = G ( x )$ denotes the transformed input representation, which consists of the input lightness map � and chromatic map �, and $x _ { \mathrm { h } } = F ( x _ { \mathrm { l } } )$ denotes the enhanced transformed representation, which consists of the enhanced lightness map <sup>ˆ</sup>� and chromatic map �ˆ. The final RGB image is reconstructed as $\hat { y } = G ^ { - 1 } ( x _ { \mathrm { h } } )$

However, RGB images entangle lightness and chromatic information, and the original pipeline does not account for explicit lightnesschrominance decoupling. Directly feeding decoupled lightness and chromatic components into such backbones can instead be counterproductive. To better match this processing pattern, we further introduce a pseudo Retinex-based decoupling and recomposition around the transformed-space backbone. Specifically, the transformed chromatic map is first normalized by the lightness map to obtain a pseudo reflectance-related component $u _ { 1 } = u \oslash l .$ . The enhancement backbone then takes � and � as input and produces the enhanced lightness map $\hat { l }$ and the enhanced pseudo reflectancerelated component $u _ { \mathrm { h } } .$ Before the inverse transform, the transformed chromatic map is recovered by $\hat { u } ~ = ~ u _ { \mathrm { h } }$ ⊙ <sup>ˆ</sup>�. The whole process can be formulated as:

![](images/92f2902445f83356f20bc5b183bd5131770cd18fbf419f7a07debe345d31e038.jpg)  
(a) Color Space Transformation Paths of HVI and AdaLAB

![](images/5be2e1cd7e1be2961ff9196781b9a5355dcf93b7185f01b2539fe945910b0fe4.jpg)

![](images/d8d5a9e9f959a23d0146d2dd30c5ee4f17d62a41a504cc9f10d46803fe27c3f6.jpg)

![](images/84f3cc95ed3b42aa1a6d4e842e246c6ecd5cb9e4ee554e0f480a58176672fb10.jpg)  
(b) Structural Coherence in Different Color Spaces

![](images/fffa54a32cfbcf21e5ab2b5ee3ccb19e2378365d7236d3ac768231996f75eea0.jpg)

![](images/529074f5fd68cb5e1665b389a0faf88eb8565784eb580c0b40f2a0ea60fd030f.jpg)

Figure 8: LAB serves as a more coherent basis for color-faithful low-light image enhancement. (a) HVI follows the RGB-to-HSV route and inherits the structural weakness of HSV, while AdaLAB starts from the one-step RGB-to-LAB conversion and further introduces image-adaptive cylindrical parameters. (b) Under uniform RGB sampling, HSV becomes non-uniform and cylindrical HSV shows clear low-lightness disparity, whereas LAB preserves a more coherent organization.  
![](images/05e317fad0181b7c7c07eb95ea6cd86b77fddf71f02a7a0c1fe155980e7fdbc3.jpg)  
Figure 9: Detailed pipeline of the lightness-aware interpolation operation. Given the lightness map, we first place the values at the lightness sensitivity vertices along the lightness axis, then determine the two neighboring vertices for each pixel according to the lightness map, and finally apply linear interpolation to obtain the lightness-aware value map.

$$
( l , u ) = G ( x ) ,\tag{16a}
$$

$$
( \hat { l } , \hat { u } ) = F ( [ l ; u \oslash l ] ) ,\tag{16b}
$$

$$
\hat { y } = G ^ { - 1 } ( [ \hat { l } ; \hat { u } \odot \hat { l } ] ) ,\tag{16c}
$$

where $u _ { \mathrm { l } }$ and $u _ { \mathrm { h } }$ denote the pseudo reflectance-related component before and after enhancement, respectively, ⊘ denotes element-wise division between the chromatic map and the lightness map, and ⊙ denotes element-wise multiplication for chromatic map recovery, [·; ·] denotes channel-wise concatenation.

Illumination-dominated Framework. A representative integration setting for this paradigm is given by Retinex-based enhancement frameworks [5]. In this paradigm, the enhancement pipeline is organized as a decomposition-and-recomposition process centered on lightness recovery. Specifically, the input image is first decomposed into a lightness-related component � and a reflectance-related component � via $x = l \odot u .$ . The lightness branch then focuses on brightness recovery, while the reflectance branch preserves reflectance consistency. After branch-wise enhancement, the enhanced components are recomposed into an RGB image via $\boldsymbol { \tilde { y } } = \boldsymbol { \hat { l } } \odot \boldsymbol { \hat { u } } ,$ , and a denoising network can be further appended after recomposition in frameworks with such a design.

![](images/001dcea91c58e40d28584d7d2315e5e9ce1f49bfa7d1119d7262b0ccd2601ae1.jpg)  
Figure 10: Integration strategies of CAGE in three representative low-light enhancement paradigms: (a) Lightnesschrominance joint enhancement frameworks, where CAGE replaces RGB processing with the proposed forward and inverse transforms; (b) Illumination-dominated frameworks, where CAGE replaces the original decomposition and recomposition pipeline in the transformed space; and (c) Chrominance-donimated enhancement frameworks, where CAGE is placed around the decoupled lightness and chroma branches before reconstruction.

Our proposed CAGE replaces the original decomposition with the forward transform and the original recomposition with the inverse transform, and keeping the post-reconstruction denoising network unchanged. Specifically, the input image is first mapped by � into the transformed space to obtain the lightness map and chromatic map. The transformed lightness map is then enhanced by the lightness branch $F _ { \mathrm { L } }$ with the transformed chromatic map as input, while the transformed chromatic map is enhanced by the chromatic branch $F _ { \mathrm { C } }$ . The enhanced lightness map and chromatic map are then mapped back to RGB by the inverse transform $G ^ { - 1 }$ The whole process can be mathematically expressed as:

$$
( l , u ) = G ( x ) ,\tag{17a}
$$

$$
\begin{array} { r } { ( \hat { l } , \hat { u } ) = \big ( F _ { \mathrm { L } } ( [ l ; u ] ) , F _ { \mathrm { C } } ( u ) \big ) , } \end{array}\tag{17b}
$$

$$
\boldsymbol { \tilde { y } } = \boldsymbol { G } ^ { - 1 } ( [ \boldsymbol { \hat { l } } ; \boldsymbol { \hat { u } } ] ) ,\tag{17c}
$$

where $( l , u )$ denote the transformed input with the lightness map and chromatic map, $( \hat { l } , \hat { u } )$ denote the enhanced transformed output, and $\tilde { y }$ denotes the reconstructed RGB image before the optional denoising network, [·; ·] denotes channel-wise concatenation. For frameworks with a denoising network after reconstruction [5], the reconstructed RGB image is further refined by $F _ { \mathrm { N } }$ to obtain the final

enhanced output:

$$
\hat { y } = F _ { \mathrm { N } } ( \tilde { y } ) ,\tag{18}
$$

where $\hat { y }$ denotes the final enhanced output.

Chrominance-dominated Framework. A representative integration setting for this paradigm is given by HSV-inspired colorspace enhancement frameworks [47]. In this paradigm, CAGE is inserted around the decoupled lightness and chromatic branches before reconstruction, preserving the original decoupled enhancement path while replacing only the outer transform and RGB reconstruction process. Specifically, the input image is first mapped by � into the transformed space to obtain the lightness map and chromatic map. The lightness map is then enhanced by the lightness branch $F _ { \mathrm { { L } } } ,$ while the chromatic map is enhanced by the chromatic branch $F _ { \mathrm { C } }$ with the lightness map as additional input. The enhanced lightness map and chromatic map are finally mapped back to RGB by the inverse transform. The entire process can be formulated as:

$$
( l , u ) = G ( x ) ,
$$

$$
( \hat { l } , \hat { u } ) = \big ( F _ { \mathrm { L } } ( l ) , F _ { \mathrm { C } } ( [ u ; l ] ) \big ) ,\tag{19a}
$$

(19b)

$$
\boldsymbol { \hat { y } } = \boldsymbol { G } ^ { - 1 } ( [ \boldsymbol { \hat { l } } ; \boldsymbol { \hat { u } } ] ) ,\tag{19c}
$$

where $( l , u )$ denote the transformed input with the lightness map and chromatic map, (<sup>ˆ</sup>�, �ˆ) denote the enhanced transformed output, and $\hat { y }$ denotes the reconstructed RGB image, [·; ·] denotes channelwise concatenation.

These three integration strategies provide a unified way to integrate CAGE into various low-light image enhancement frameworks, while keeping their original backbone structures unchanged.

## B More Experiments

## B.1 Training Details

Experimental Environment. The experiments were run on a server equipped with an Intel Xeon Gold 6326 CPU (16 cores @ 2.90GHz), NVIDIA A40 GPU (48GB VRAM), and 251GB system memory, running Ubuntu 20.04.6 LTS. The software stack included CUDA 11.6, Python 3.8.20, GCC 9.4.0, and PyTorch 1.13.0. Full dependency specifications is provided in the open-source release of our code.

Hyperparameter Configuration. Unless otherwise specified, we set the number of lightness intervals to $ { p } = 3 2$ , the response range of the similarity-aware weight to $\delta _ { 1 } ~ = ~ 0 . 2$ and $\delta _ { 2 } ~ = ~ 1 . 0$ the lightness compensation ratio in out-of-gamut lightness compensation to $\gamma = 1 . 0 \mathrm { : }$ , and the loss balance between the AdaLAB reconstruction loss and the RGB reconstruction loss to $\lambda = 1 . 0$ . For the final linear adjustment before LAB-to-RGB conversion, we set $\alpha _ { \mathrm { l } } = 1 . 0$ and $\alpha _ { \mathrm { c } } = 1 . 0$ by default. Following previous work [47], we use dataset-specific brightness and saturation adjustment on LOLv1 and LOLv2-Real. For LOLv1, we set $\alpha _ { \mathrm { l } } = 1 . 3$ and $\alpha _ { \mathrm { c } } = 1 . 0$ For LOLv2-Real, we set $\alpha _ { \mathrm { l } } = 1 . 1$ and $\alpha _ { \mathrm { c } } = 0 . 8$

GT Mean Evaluation. Since LOLv1 contains only 15 testing pairs, direct metric evaluation can be sensitive to global brightness fluctuation and may obscure the comparison of color restoration and structural recovery. Following common practice in low-light image enhancement [40, 47, 48, 62], we use GT mean evaluation on LOLv1 during testing. For each predicted image, we first convert both the prediction and the ground-truth image into grayscale to compute their mean luminance, and then linearly rescale the prediction to match the mean luminance of the ground truth before metric evaluation. This process can be formulated as follows:

![](images/408b922c94365b15e21a8c25bc939a25082bd383e4581bc21cd46cf7c5bfe22b.jpg)  
Figure 11: Ablation of the interval number $\mathcal { P } \cdot$

Table 6: Summary of the datasets used in the experiments.
<table><tr><td>Dataset Group</td><td>Dataset / Subset</td><td>#Train</td><td>#Test</td><td>Resolution (H × W)</td></tr><tr><td rowspan="3">Paired LOL</td><td>LOLv1</td><td>485</td><td>15</td><td>400×600</td></tr><tr><td>LOLv2-Real</td><td>689</td><td>100</td><td>400×600</td></tr><tr><td>LOLv2-Synthetic</td><td>900</td><td>100</td><td>384× 384</td></tr><tr><td rowspan="2">Paired SDSD</td><td>SDSD-indoor</td><td>1655</td><td>308</td><td>512×960</td></tr><tr><td>SDSD-outdoor</td><td>2650</td><td>500</td><td>512×960</td></tr><tr><td>Paired SID</td><td>Sony subset</td><td>2099</td><td>598</td><td>1424 × 2128</td></tr><tr><td rowspan="6">Unpaired</td><td>DICM</td><td>0</td><td>69</td><td>Various</td></tr><tr><td>LIME</td><td>0</td><td>10</td><td>Various</td></tr><tr><td>MEF</td><td>0</td><td>17</td><td>Various</td></tr><tr><td>NPE</td><td>0</td><td>8</td><td>Various</td></tr><tr><td>VV</td><td>0</td><td>24</td><td>Various</td></tr></table>

$$
\hat { y } _ { \mathrm { G T m e a n } } = \hat { y } \cdot \frac { \mu ( y ) } { \mu ( \hat { y } ) } ,\tag{20}
$$

where $\hat { y } \in \mathbb { R } ^ { 3 \times H \times W }$ denotes the enhanced image, $y \in \mathbb { R } ^ { 3 \times H \times W }$ denotes the ground-truth image, and �(·) denotes the mean grayscale operation computed by averaging all pixel values after grayscale conversion. This operation reduces unnecessary luminance variation in the final comparison and makes PSNR, SSIM, and LPIPS on LOLv1 more focused on restoration quality beyond global brightness mismatch.

## B.2 Datasets

Detailed information on all datasets is summarized in Table 6. In this appendix, we further include more visual comparisons on LOLv1 [42], LOLv2-Real [52], and LOLv2-Synthetic [52] as the main paired benchmarks to demonstrate the color-faithful restoration capability of our proposed CAGE. To further present the efectiveness of our method under more challenging paired low-light conditions, we include more visual comparisons on SDSD-indoor [36], SDSDoutdoor [36], and SID [7]. To evaluate perceptual quality on real low-light scenes without paired normal-light targets, we include DICM [22], LIME [15], MEF [28], NPE [37], and VV [33], and adopt the no-reference metrics BRISQUE [29] and NIQE [30].

## B.3 More Benchmarking Results

We provide additional comparisons on both unpaired and paired benchmarks to further present the performance of our method across diverse low-light scenes and imaging conditions, including no-reference quantitative results and visual comparisons on unpaired benchmarks, as well as more qualitative comparisons on paired benchmarks.

More Results on Unpaired Datasets. As shown in Table 7, CAGE improves perceptual quality across diferent backbone designs and datasets. Over the 15 backbone-dataset combinations built on Retinexformer, DarkIR, and HVI-CIDNet, CAGE reduces BRISQUE in 12 cases and NIQE in 14 cases. In particular, when integrated into DarkIR, CAGE reduces BRISQUE by 1.45 / 3.32 / 0.53 / 1.03 / 0.75 and NIQE by 0.124 / 0.301 / 0.020 / 0.105 / 0.130 on DICM, LIME, MEF, NPE, and VV, respectively, and achieves the lowest BRISQUE scores on DICM, LIME, MEF, and NPE. CAGE further reduces BRISQUE by 7.54 / 3.48 / 8.67 / 3.11 / 10.66 on the five datasets when integrated into HVI-CIDNet, and achieves the best NIQE of 3.249 on MEF together with competitive NIQE results on LIME and VV. These results indicate that the improved chromatic modeling in CAGE consistently benefits no-reference perceptual quality across diferent backbone designs and datasets, beyond the gains observed in paired full-reference evaluation.

More visual comparisons on DICM, LIME, and MEF are shown in Fig. 14, while additional results on NPE and VV are presented in Fig. 15. These results indicate more natural exposure adjustment, more stable color rendition, and fewer local artifacts under diverse scene illumination.

More Qualitative Comparisons on Paired Datasets. More qualitative comparisons on LOLv1 [42], LOLv2-Real [52], LOLv2- Synthetic [52], SDSD-indoor [36], SDSD-outdoor [36], and SID [7] are shown in Figs. 16–19. On these paired datasets, our method produces reconstructions with more faithful brightness restoration, cleaner structural details, and better color consistency with the reference images.

## B.4 More Ablation Studies

We further include more ablation experiments on the real-world LOL-v2-real dataset to assess the contribution of each component in CAGE and provide a more detailed analysis of their efects on color-faithful low-light image enhancement.

Number of Intervals �. We vary the number of intervals � to analyze its efect on adaptive lightness sensitivity modeling. As shown in Fig. 11, the overall performance improves as � increases from 4 to 32, and then drops when � becomes larger. The best setting is $p = 3 2 ,$ , which achieves 24.2355 dB PSNR, 0.8746 SSIM, and 0.0977 LPIPS. A small � gives an overly coarse partition on the lightness axis, while an excessively large � produces overly fine intervals and reduces the robustness of interpolation. Meanwhile, the parameter count increases only slightly from 2.039M to 2.303M as � increases from 4 to 1024. Therefore, we use � = 32 as the default setting.

Global Interaction in Vertex Values. We further compare four strategies for learning the vertex values on the lightness axis, including the sinusoidal curve used in HVI-CIDNet [47], a plain 1D look-up-table (1DLUT) [20], a 1D LUT with monotonicity, and the proposed 1D LUT with global interaction. As shown in Table 8, replacing the sinusoidal curve with a plain 1D LUT improves PSNR from 23.9560 to 24.1666, SSIM from 0.8701 to 0.8727, and reduces LPIPS from 0.1080 to 0.1015. This result shows that the vertex values from HVI-CIDNet [47], where $k \in \mathbb { Q } ^ { + }$ is a trainable parameter, � = 1 × 10<sup>−8</sup> is a small constant to avoid gradient explosion, and � is the lightness map.

Table 7: No-reference quantitative comparisons on unpaired low-light datasets.
<table><tr><td rowspan="2">Methods</td><td colspan="2">DICM</td><td colspan="2">LIME</td><td colspan="2">MEF</td><td colspan="2">NPE</td><td colspan="2">VV</td></tr><tr><td>BRISQUE ↓</td><td>NIQE↓</td><td>BRISQUE ↓</td><td>NIQE ↓</td><td>BRISQUE ↓</td><td>NIQE↓</td><td>BRISQUE ↓</td><td>NIQE↓</td><td>BRISQUE ↓</td><td>NIQE↓</td></tr><tr><td>LIME [15] TIP&#x27;17</td><td>23.36</td><td>3.812</td><td>18.98</td><td>4.085</td><td>16.34</td><td>3.558</td><td>17.30</td><td>4.196</td><td></td><td></td></tr><tr><td>KinD [61] ACM MM&#x27;19</td><td>29.11</td><td>4.139</td><td>25.21</td><td>4.762</td><td>27.51</td><td>3.875</td><td>18.09</td><td>4.165</td><td>23.41</td><td>3.027</td></tr><tr><td>ZeroDCE [12] cVPR&#x27;20</td><td>23.52</td><td>3.723</td><td>18.48</td><td>3.771</td><td>16.63</td><td>3.283</td><td>15.73</td><td>3.946</td><td></td><td></td></tr><tr><td>MIRNet [55] ECCV&#x27;20</td><td>24.00</td><td>4.738</td><td>21.04</td><td>5.005</td><td>28.28</td><td>4.549</td><td>24.59</td><td>4.556</td><td>26.12</td><td>4.084</td></tr><tr><td>ZeroDCE++ [12] TPAMr21</td><td>21.37</td><td>3.824</td><td>17.22</td><td>3.965</td><td>13.60</td><td>3.400</td><td>12.94</td><td>4.021</td><td></td><td></td></tr><tr><td>RUAS [26] CVPR&#x27;21</td><td>33.44</td><td>5.270</td><td>22.78</td><td>4.302</td><td>19.27</td><td>3.828</td><td>41.21</td><td>5.678</td><td>27.68</td><td>4.244</td></tr><tr><td>LEDNet [63] TIP&#x27;22</td><td>19.56</td><td>4.077</td><td>21.30</td><td>5.059</td><td>20.63</td><td>4.194</td><td>14.35</td><td>4.875</td><td>19.15</td><td>3.109</td></tr><tr><td>LLFlow [40] AAAr&#x27;22</td><td>24.90</td><td>3.831</td><td>26.60</td><td>5.073</td><td>26.97</td><td>3.924</td><td>19.68</td><td>4.204</td><td>28.14</td><td>3.175</td></tr><tr><td>URetinexNet [43] cVPR&#x27;22</td><td>24.27</td><td>4.152</td><td>23.22</td><td>4.240</td><td>21.45</td><td>3.791</td><td>24.42</td><td>4.693</td><td>23.26</td><td>3.003</td></tr><tr><td>UHDFour [25] ACM MM&#x27;23</td><td>33.83</td><td>9.941</td><td>37.82</td><td>4.506</td><td>30.65</td><td>4.270</td><td>19.46</td><td>4.501</td><td>138.60</td><td>46.117</td></tr><tr><td>RetinexMamba [3] ICONIP&#x27;23</td><td>16.92</td><td>4.189</td><td>19.35</td><td>4.240</td><td>19.74</td><td>4.217</td><td>18.79</td><td>4.232</td><td>12.72</td><td>3.489</td></tr><tr><td>URWKV [45] CVPR&#x27;25</td><td>20.63</td><td>4.001</td><td>19.65</td><td>4.407</td><td>25.49</td><td>4.187</td><td>16.46</td><td>4.190</td><td>22.35</td><td>3.523</td></tr><tr><td>Retinexformer [5] ıCCV&#x27;23</td><td>15.66</td><td>3.973</td><td>14.68</td><td>4.167</td><td>14.12</td><td>4.363</td><td>15.38</td><td>4.691</td><td>16.33</td><td>2.989</td></tr><tr><td>Retinexformer + Ours</td><td>19.18 (-3.52) 3.966 (+0.007)</td><td></td><td>16.01 (-1.34)</td><td>3.903 (+0.264)</td><td></td><td>12.58 (+1.54) 3.522 (+0.841)</td><td></td><td>10.53 (+4.85) 3.977 (+0.713)</td><td>17.09 (-0.76)</td><td>2.824 (+0.165)</td></tr><tr><td>DarkIR [10] CVPR&#x27;25</td><td>16.95</td><td>3.791</td><td>13.32</td><td>3.953</td><td>12.13</td><td>3.482</td><td>10.09</td><td>3.923</td><td>14.88</td><td>2.998</td></tr><tr><td>DarkIR + Ours</td><td></td><td>15.49 (+1.45) 3.667 (+0.124)</td><td>10.00 (+3.32) 3.652 (+0.301)</td><td></td><td></td><td>11.60 (+0.53) 3.463 (+0.020)</td><td>9.06 (+1.03)</td><td>3.818 (+0.105)</td><td>14.14 (+0.75)</td><td>2.868 (+0.130)</td></tr><tr><td>HVI-CIDNet [47] CVPR&#x27;25</td><td>25.50</td><td>3.571</td><td>18.64</td><td>4.052</td><td>20.52</td><td>3.701</td><td>16.99</td><td>4.118</td><td>27.35</td><td>3.313</td></tr><tr><td>HVI-CIDNet + Ours</td><td>17.96 (+7.54)</td><td>3.706 (-0.135)</td><td>15.16 (+3.48)</td><td>3.701 (+0.352)</td><td></td><td>11.85 (+8.67) 3.249 (+0.452)</td><td>13.88 (+3.11)</td><td>3.991 (+0.127)</td><td>16.69 (+10.66)</td><td>2.842 (+0.471)</td></tr></table>

The best and second-best results are highlighted in bold and underline, respectively. +(-) denotes the improvement (drop) relative to the corresponding baseline.

Table 8: Ablation of global interaction.
<table><tr><td>#</td><td>Vertex Values Learning Strategy</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td></tr><tr><td>①</td><td>AdaICF [47]</td><td>23.9560</td><td>0.8701</td><td>0.1080</td></tr><tr><td>②</td><td>1DLUT</td><td>24.1666</td><td>0.8727</td><td>0.1015</td></tr><tr><td>③</td><td>1DLUT w/ Monotonicity</td><td>23.7778</td><td>0.8734</td><td>0.1024</td></tr><tr><td>④</td><td>1DLUT w/ Global Interaction</td><td>24.2355</td><td>0.8746</td><td>0.0977</td></tr></table>

$$
\begin{array} { r } { C _ { k } ( l ) = \sqrt [ k ] { \sin \left( \frac { \pi l } { 2 } \right) } } \end{array}
$$

in LAB require a more flexible lightness-wise parameterization than a single smooth curve. Adding monotonicity to the 1D LUT further reduces PSNR to 23.7778 and gives an inferior LPIPS of 0.1024. This behavior is consistent with the non-monotonic feasible chroma range of LAB along the lightness axis, where a monotonic parameterization cannot follow the lightness-dependent chromatic variation well. The proposed 1D LUT with global interaction achieves the best results, reaching a PSNR of 24.2355 dB, 0.8746 SSIM, and 0.0977 LPIPS. Compared with the plain 1D LUT, global interaction further improves PSNR by 0.0689, SSIM by 0.0019, and reduces LPIPS by 0.0038. The gain comes from allowing each vertex value to interact with all lightness vertices rather than only two adjacent samples, which yields more consistent chromatic reorganization across lightness levels.

## B.5 Eficiency Analysis

To further quantify the eficiency overhead introduced by CAGE, we compare Retinexformer, DarkIR, and HVI-CIDNet with and without CAGE in terms of parameters, FLOPs, latency, and memory usage. Following the experimental environment described in Sec. B.1, latency and memory usage are measured on a single NVIDIA A40 GPU.

As shown in Table 9, CAGE adds 0.070M parameters and 0.005G FLOPs to all three backbones (the FLOPs are shown to three decimal places to make this small increment visible). Consistent with the analysis in the main body, the extra cost is very small, indicating that the performance gains mainly comes from improved color modeling rather than increased model capacity. Across the diferent resolutions of the representative method (i.e., HVI-CIDNet), the additional latency remains below 9 ms, while the memory overhead stays close to 7%, showing that the resolution-dependent eficiency overhead is mainly determined by the LLIE backbone rather than CAGE.

## C Human Subjective Evaluation

To further assess the perceptual improvement brought by CAGE, we include a human subjective evaluation with 12 participants and more than 1,500 individual ratings. For each low-light input, one result is randomly selected from CAGE-integrated methods (i.e., CAGE-CIDNet, CAGE-Retinexformer, and CAGE-DarkIR), and three results are randomly selected from the comparison methods. The four results are displayed in random order without method names, reducing selection and position bias. As shown in Fig. 12, participants score each result from 0 to 4 based on brightness, color, detail clarity, and noise or artifacts. The interface supports image enlargement and freely selected comparisons, allowing participants to inspect the overall enhancement and local details.

As shown in Fig. 13, integrating CAGE increases the mean subjective scores of three backbones (e.g., HVI-CIDNet, Retinexformer, and DarkIR), showing that the perceptual gain of CAGE is not restricted to a specific backbone design. Among the comparison methods, BreaD receives the highest mean score, while its outputs display noticeably higher saturation. Nevertheless, the CAGE variants still surpass BreaD, indicating that the perceptual advantage of CAGE extends beyond the visual appeal associated with pronounced saturation.

Table 9: Eficiency overhead of LLIE backbones + CAGE.
<table><tr><td>Backbone</td><td></td><td></td><td>Resolution |Variant|Params (M) FLOPs (G)</td><td>Latency (ms)</td><td>Memory (MB)</td></tr><tr><td>Retinexformer</td><td>256 × 256</td><td>Base CAGE</td><td>|1.606 17.019 1.676 (+0.070) 17.024 (+0.005)</td><td>19.42 ±1.43 24.99 ±0.17 (+5.57)</td><td>1358 1462 (+7.7%)</td></tr><tr><td>DarkIR</td><td>256 × 256</td><td>Base CAGE</td><td>|3.322 7.110 3.392 (+0.070) 7.115 (+0.005)</td><td>22.94 ±0.25 31.80 ±0.24 (+8.86)</td><td>960 1020 (+6.3%)</td></tr><tr><td rowspan="3">HVI-CIDNet</td><td>256 × 256</td><td>Base CAGE</td><td>|1.976 8.133 2.046 (+0.070) 8.138 (+0.005)</td><td>27.66 ±0.21 34.34 ±0.19 (+6.68)</td><td>906 978 (+7.9%)</td></tr><tr><td>512×512</td><td>Base CAGE</td><td>|1.976 32.532 2.046 (+0.070) 32.537 (+0.005)</td><td>79.80 ±0.39 86.63 ±0.28 (+6.83)</td><td>3610 3876 (+7.4%)</td></tr><tr><td>1024 × 1024</td><td>Base CAGE</td><td>|1.976 130.127 2.046 (+0.070)130.132 (+0.005)316.83 ±0.22 (+8.72)14966 (+7.9%)</td><td>308.11 ±0.65</td><td>13870</td></tr></table>

Latency denotes mean ± std over 100 runs. Green values denote the overhead introduced by CAGE.

![](images/507f79c7d8ce9ab36de8586a319016052e68b5f65390db6a2ac806a7b798d792.jpg)  
Figure 12: Interface used for human subjective evaluation.

![](images/c9da2c28c925c6278e766bb0de4df0e4b78a79f2dcb6ed42e36cf0734d0efffe.jpg)  
Figure 13: Distribution of subjective scores for representative LLIE methods and three backbones w/ and w/o CAGE. Red horizontal lines indicate the mean scores. Scores range from 0 to 4, with higher scores indicating better perceptual quality.

## D Limitations

Although CAGE improves color-faithful low-light image enhancement by explicitly modeling chromatic disturbance in the AdaLAB space, the shared hue-shift direction defines its main application boundary. Hue-directional chromatic debiasing predicts one imagelevel shift direction and varies its magnitude along the lightness axis. This formulation matches low-light color bias whose direction is coherent across the image and whose magnitude changes with lightness. Under mixed illumination, however, spatially separated regions with similar lightness may exhibit diferent color-shift directions. Since the current formulation assigns the same direction to these regions, residual local color bias may persist after enhancement. A future extension can incorporate local chromatic content into direction prediction, allowing regions with similar lightness to receive diferent hue-shift directions.

![](images/bdf9d8142c409198e34cd617d0082f96900bc47973acdbb4b576fba250eaed89.jpg)  
Figure 14: Visual examples on the DICM, LIME, and MEF datasets among LIME [15], KinD [61], RetinexNet [42], Enlighten-GAN [19], MIRNet [55], LEDNet [63], LLFlow [40], and three baseline methods, namely Retinexformer [5], DarkIR [10], and HVI-CIDNet [47], with and without our proposed CAGE. Our method exhibits stronger exposure balancing and color correction, consequently resulting in more natural brightness, fewer color casts, and clearer local details.

![](images/f27541b0afc76f0754819f8a3c66c1bd5f8e72dd78149ed3ed0fb90697eede2a.jpg)

Figure 15: Visual examples on the NPE and VV datasets among UHDFour [25], KinD [61], RetinexNet [42], RetinexMamba [3], LEDNet [63], LLFlow [40], URWKV [45], and three baseline methods, namely Retinexformer [5], DarkIR [10], and HVI-CIDNet [47], with and without our proposed CAGE. Our method exhibits stronger highlight restraint and color recovery, resulting in more balanced exposure, fewer blown-out regions, and more faithful scene colors.  
![](images/8b3f426a5806e77d8927cd067769580cfc2c91b3a5cc005102d4715c7a48afda.jpg)  
Figure 16: Visual examples on the LOLv1 dataset among RUAS [26], KinD [61], BreaD [14], FourLLIE [35], RetinexNet [42], KinD++ [60], URetinexNet [43], and MIRNet [55], as well as Retinexformer [5], DarkIR [10], and HVI-CIDNet [47], with and without our proposed CAGE. Our method exhibits better structure-preserving brightness recovery and color correction.

![](images/1930edd98b4314025050d325ee4847666fde01eb8c07a0d9844c5c05b7ff945f.jpg)  
Figure 17: Visual examples on the LOLv2-real dataset among KinD [61], KinD++ [60], RetinexNet [42], RUAS, IAT [8], Restormer [54], LLFlow [40], BreaD [14], and RetinexMamba [3], as well as three baseline methods, namely Retinexformer [5], DarkIR [10], and HVI-CIDNet [47], with and without our proposed CAGE. Our method yields more balanced brightness restoration and more stable color recovery.

![](images/ed490867ca2a8ddad1792eea8785b8dc68d6e9ba0181bff79ad9c04f9808e58e.jpg)  
Figure 18: Visual examples for low-light image enhancement on the LOLv2-synthetic dataset among RUAS, KinD [61], KinD++ [60], RetinexNet [42], BreaD [14], Restormer [54], FourLLIE [35], URetinexNet [43], MIRNet [55], and CWNet [58], as well as three baseline methods, namely Retinexformer [5], DarkIR [10], and HVI-CIDNet [47], with and without our proposed CAGE. Our method yields more balanced exposure and more faithful color restoration, consequently resulting in outputs with clearer local structures and fewer artifacts.

Towards Color-Faithful Low-Light Image Enhancement via Adaptive Color Debiasing and Saturation Rectification

MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil

![](images/7a12bd1328b463ee368ef4df69b16007ae9e208f22ed4b78109a39a20be6ef80.jpg)

Figure 19: Visual examples for low-light image enhancement on the SDSD-indoor, SDSD-outdoor, and SID datasets among BreaD [14], LLFormer [38], FourLLIE [35], and CWNet [58], as well as three baseline methods, namely Retinexformer [5], DarkIR [10], and HVI-CIDNet [47], with and without our proposed CAGE. In SDSD-indoor, CAGE recovers severely dark regions while maintaining a smooth intensity transition, avoiding the over-flattened appearance. In SDSD-outdoor, it preserves local contrast under strong illumination variation, preventing detail loss around bright structures. In SID, CAGE enhances fine textual content and small objects with clearer edges, reducing color contamination and structural ambiguity in challenging low-light conditions.