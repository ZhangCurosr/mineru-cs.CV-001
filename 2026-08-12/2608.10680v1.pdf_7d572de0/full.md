# Bridging Severe Cross-Modal Misalignment: End-to-End Visible-Infrared Object Detection via Explicit Feature-Domain Affine Registration

Qi Ming<sup>1</sup>, Yuyang Wang<sup>2∗</sup>, Mingjing Zhao<sup>3</sup>, Yifan Xiao<sup>4</sup>,

Zhixin Guo<sup>4</sup>, Zhiqiang Zhou<sup>5</sup>, Peng Sun<sup>6</sup>, Juan Fang<sup>1</sup>, Fuqiang Yang<sup>7</sup>, Xudong Zhao<sup>5</sup>

<sup>1</sup>Beijing University of Technology, China

<sup>3</sup>Beijing Electronics Science & Technology Institute

<sup>2</sup>Central South University, China

<sup>5</sup>Beijing Institute of Technology, China

<sup>4</sup>China Aerospace Science & Industry Corporation

<sup>6</sup>Information Support Force Engineering University, China

<sup>7</sup>Trunk Technology (Beijing) Co., Ltd., China

chaser.ming@gmail.com, killlakill2023@gmail.com

## Abstract

Visible-infrared object detection relies on complementary RGB and thermal cues, but its performance is often degraded by cross-modal spatial misalignment. Most existing methods rely on implicit feature adaptation to handle weakly misaligned scenarios, while large-offset geometric discrepancies remain insufficiently addressed. In this paper, we propose a Joint Feature-domain Registration and Detection network (JFRDet), an end-to-end visibleinfrared oriented object detector tailored for severely cross-modal geometric discrepancies. JFRDet introduces a Cross-Modal Affine Alignment (CMAA) module to estimate an image-level affine transformationfor explicit multi-level feature alignment. Note that illumination changes directly affect the reliability of RGB cues, an Illumination-Guided Complementary Fusion (IGCF) module adaptively exploits modality reliability under varying illumination conditions for cross-modal fusion. Then, an Alignment Quality-Consistency Gating (AQCG) strategy stabilizes joint optimization by modulating detection supervision according to alignment reliability and gradient consistency. We further construct DroneVehicle Misaligned (DVMA), a benchmark for evaluating visible-infrared oriented object detection under severe cross-modal geometric misalignment. The proposed JFRDet achieves 69.7% mAP on DVMA, which represents state-of-the-art (SOTA) performance. The code and dataset will be available on GitHub.

## 1. Introduction

Visible-Infrared Object Detection (VIOD) has attracted increasing attention by exploiting the complementary strengths of visible and infrared modalities [9, 18]. RGB images provide rich texture and appearance cues, yet they often encounter limitations under low-light or adverse weather conditions. In contrast, infrared images are more robust to illumination variations, making their combination particularly effective for applications such as surveillance and autonomous driving [10,13]. However, visible-infrared image pairs often exhibit spatial misalignment due to differences in sensor placement, viewpoint, or platform motion. Such cross-modal spatial misalignment, including translation, rotation, and scale variations, significantly undermines the effectiveness of cross-modal feature fusion [39]. Robust cross-modal fusion under spatial misalignment remains a major challenge.

To address this issue, recent visible-infrared detectors incorporate alignment-aware designs into the detection pipeline. Some methods [36, 38, 39] estimate crossmodal correspondences to recalibrate modality-specific features, thereby reducing fusion errors caused by misalignment. Other studies [8, 23, 35] combine explicit calibration with feature correction or region-level reasoning to further alleviate modality discrepancies. Another line of work avoids strict geometric correction and instead develops fusion modules robust to weak misalignment. Representative strategies include deformable attention and offset-guided sampling [1, 4, 5], which allow the model to adaptively aggregate cross-modal features without precise spatial correspondence.

However, existing misalignment-aware VIOD studies suffer from two key limitations. (1) Severe cross-modal misalignment remains underexplored. Most methods are designed and evaluated on aligned or weakly misaligned image pairs. As shown in Fig 1 (d), most offsets in DroneVehicle [24] fall within 1–5 pixels, offering limited evidence of robustness to severe misalignment. (2) Feature downsampling obscures weak misalignment. Repeated backbone downsampling maps small image-level offsets to barely distinguishable displacements on low-resolution feature maps, effectively creating near-aligned training conditions. Consequently, existing methods may not learn to resolve intrinsic cross-modal spatial discrepancies, limiting their robustness in severely misaligned real-world scenes.

![](images/0e5b6e7039019ea81a805cf49c951a13fd35294a971de7cec9a50fe165bf637d.jpg)  
Figure 1. Comparison of different multimodal detection paradigms. (a) Spatially aligned multimodal detection assumes well-registered inputs. (b) Weakly misaligned detection performs local pixel- or feature-level alignment before fusion. (c) Our JFRDet predicts an imagelevel affine transform and progressively applies it to multi-level features, enabling aligned fusion and detection under severe misalignment. (d) Offset distribution comparison between DroneVehicle and DVMA.

To this end, this paper proposes the Joint Feature-domain Registration and Detection Network (JFRDet), an end-toend framework for VIOD in severe misalignment scenarios. JFRDet enables registration and oriented object detection to be optimized within a unified framework. Specifically, a Cross-Modal Affine Alignment module (CMAA) is introduced to recover the global geometric relationship between modalities by establishing cross-modal correspondences and estimating an affine transformation. Then, an Illumination-Guided Complementary Fusion module (IGCF) adaptively balances visible and infrared cues according to illumination, compensating degraded RGB evidence with infrared responses while retaining useful appearance details. Next, an Alignment Quality-Consistency Gating strategy (AQCG) stabilizes joint optimization by regulating detector learning based on alignment reliability and task consistency. These components jointly enable robust visible-infrared oriented object detection under large cross-modal geometric variations. Moreover, we construct the DroneVehicle Misaligned (DVMA) benchmark to evaluate visible-infrared oriented object detection under severe cross-modal misalignment. Compared with weakly misaligned datasets, DVMA provides a more challenging setting for assessing cross-modal alignment and detection robustness, as illustrated in Fig. 1 (d).

Our main contributions can be summarized as follows:

1) This paper proposes JFRDet, an end-to-end framework that performs explicit feature-domain affine registration for accurate VIOD. JFRDet is a pioneering work to address visible-infrared oriented object detection under severe cross-modal misalignment scenarios.

2) We introduce JFRDet with CMAA for explicit affine feature registration. IGCF then performs illuminationadaptive cross-modal fusion, while AQCG coordinates registration and detection training by adjusting detection supervision according to alignment quality and optimization consistency.

3) DVMA benchmark is constructed with substantially larger cross-modal geometric discrepancies than existing visible-infrared datasets. It provides a challenging evaluation setting for assessing detection robustness under severe visible-infrared misalignment.

## 2. Related Work

## 2.1. Oriented Object Detection

Oriented object detection is widely studied in aerial remote sensing, objects exhibit arbitrary orientations and dense layouts.Early methods adapt horizontal detectors using rotation-aware proposals or RoI transformations [3, 11, 19]. Subsequent studies improve localization through progressive rotated-box refinement and feature alignment [27, 31]. Others introduce flexible object representations, such as vertex-based or point-set-based representations, to describe arbitrary oriented targets [15, 28]. Feature-adaptive strategies are also explored to improve the discrimination of densely arranged objects [6]. Another line of work reformulates angle prediction or regression losses to mitigate boundary discontinuity and improve localization consistency [29, 30, 32–34]. Despite these advances, most oriented detectors are designed for visible imagery, which limits their robustness under low-light conditions.

![](images/833039b1c2af9829189bbe1eab9c3f9356ce3671dd159944417737cdd22cb50c.jpg)  
Figure 2. Construction pipeline of the DVMA benchmark.

## 2.2. Visible-Infrared Object Detection

Visible-infrared object detection exploits the complementary properties of RGB and IR imagery. Early methods mainly focused on where and how to fuse the two modalities, ranging from deep feature fusion and multi-layer fusion to joint detection-segmentation learning [2, 12, 25]. Later works further introduced gated fusion, modality balancing, and dynamic cross-modal interaction to suppress unreliable modality responses and enhance complementary cues [26, 42, 43]. Recently, attention- and Transformerbased designs model long-range cross-modal dependencies and improve interaction in complex scenes [20–22, 40]. Other studies reduce redundant information or estimate modality confidence during aggregation [14, 41]. Despite improved robustness under low-light conditions, most methods assume well-registered or weakly misaligned image pairs and remain insufficiently robust to severe crossmodal misalignment.

## 3. Method

## 3.1. DroneVehicle Misaligned Benchmark

As illustrated in Fig. 2, DVMA is constructed from aligned image pairs in DroneVehicle [24]. The visible image is center-cropped, while the infrared image undergoes a compound transformation composed of rotation, anisotropic scaling, and translation. The rotation angle is sampled between $1 5 ^ { \circ }$ and $3 0 ^ { \circ }$ in either direction, and the horizontal and vertical scaling factors are independently sampled from [0.90, 1.10]. We retain only pairs whose transformed valid regions have an overlap IoU within [0.70, 0.80]. These transformations produce spatially varying cross-modal offsets ranging from tens to hundreds of pixels while maintaining sufficient scene overlap. Since all transformation parameters are recorded, exact cross-modal correspondences can be generated automatically at each feature resolution. Continuous correspondence coordinates are used to construct normalized ground-truth offset fields consistent with the [−1, 1] coordinate range of bilinear sampling, while discretized target locations provide matching labels. Positions mapped outside the valid feature region are masked during loss computation, and correspondence supervision is generated symmetrically in both visible-toinfrared and infrared-to-visible directions. As summarized in Table 1, compared with existing misaligned datasets, DVMA provides explicit alignment annotations for paired images, enabling supervised cross-modal registration during training.

<table><tr><td>Dataset</td><td>Images</td><td>Misaligned</td><td>Categories</td><td>Scenario</td><td>Alignment GT</td></tr><tr><td>VEDAI</td><td>1200</td><td>X</td><td>9</td><td>drone</td><td>×</td></tr><tr><td>KAIST</td><td>95328</td><td>△</td><td>3</td><td>driving</td><td>×</td></tr><tr><td>CVC14</td><td>17036</td><td>√</td><td>1</td><td>driving</td><td>×</td></tr><tr><td>FLIR</td><td>24680</td><td>√</td><td>4</td><td>driving</td><td>×</td></tr><tr><td>LLVIP</td><td>30976</td><td>×</td><td>1</td><td>surveillance</td><td>×</td></tr><tr><td>DroneVehicle</td><td>56878</td><td>△</td><td>5</td><td>drone</td><td>×</td></tr><tr><td>M3FD</td><td>8400</td><td>X</td><td>6</td><td>various</td><td>×</td></tr><tr><td>DVTOD</td><td>4358</td><td>√</td><td>3</td><td>drone</td><td>×</td></tr><tr><td>DVMA (ours)</td><td>56878</td><td>√</td><td>5</td><td>drone</td><td>√</td></tr></table>

Table 1. Comparison of multispectral object detection datasets. $\triangle$ indicates that only a subset of objects are weakly misaligned. ‘Alignment $\mathrm { G T } ^ { \mathrm { , } }$ indicates whether ground-truth cross-modal correspondences are provided for alignment.

## 3.2. Overview Architecture

The overall framework of JFRDet is illustrated in Fig. 3. Paired RGB-IR images are processed by a dual-stream backbone following [44] to extract multi-scale modalityspecific features. Taking the visible branch as reference, CMAA estimates an affine transformation and progressively aligns infrared features. IGCF then fuses the aligned features under varying illumination conditions, which are fed into an $\mathrm { S ^ { 2 } A }$ -Net head [7] for oriented prediction. To stabilize training, AQCG regulates detection supervision based on alignment reliability, reducing the negative impact of unreliable alignment on detector optimization.

## 3.3. Cross-Modal Affine Alignment

Given multi-scale visible and infrared features $F _ { r g b } ^ { s }$ and $F _ { i r } ^ { s }$ , where $s \in 1 / 4 , 1 / 8 , 1 / 1 6 , 1 / 3 2$ denotes the resolution relative to the input, the proposed CMAA establishes coarse-to-fine cross-modal correspondences using the $1 / 4 .$ $1 / 8 ,$ , and $1 / 1 6$ features. Specifically, it first identifies robust coarse correspondences $\mathcal { M } _ { c }$ under severe spatial offsets, and then refines them into more precise fine-grained correspondences $\mathcal { M } _ { f }$ . These correspondences are used to estimate an image-level affine transformation $\mathbf { T } _ { i r  r g b } \in$ $\mathbb { R } ^ { 2 \times 3 }$ , which progressively warps all infrared pyramid features to obtain aligned representations $\tilde { F } _ { i r } \ = \ \tilde { F } _ { i r } ^ { s }$ . Consequently, subsequent cross-modal fusion is conducted on geometrically aligned features rather than directly on misaligned RGB-IR representations.

![](images/81f78e4ee94728f84124ad0b0fd3ef034c9da7c384ecb41ebff3ea7179130333.jpg)  
Figure 3. Network architecture of the proposed JFRDet.

## 3.3.1 Coarse-Grained Matching

To establish robust initial correspondences under severe cross-modal spatial offsets, coarse-grained matching is performed on $F _ { r g b } ^ { 1 / 1 6 }$ and $F _ { i r } ^ { 1 / 1 6 }$ . After positional encoding and flattening, the resulting tokens are processed by alternating intra-modal self-attention and inter-modal crossattention to capture long-range context and cross-modal correspondences. The refined sequences are denoted as $\hat { X } _ { r g b } ~ = ~ \{ \hat { x } _ { r q b } ^ { i } \} _ { i = 1 } ^ { N _ { r g b } }$ and $\hat { X } _ { i r } ~ = ~ \{ \hat { x } _ { i r } ^ { j } \} _ { j = 1 } ^ { N _ { i r } }$ , where $N _ { r g b }$ and $N _ { i r }$ denote the numbers of visible and infrared tokens, respectively. A temperature-scaled similarity matrix $S \in \mathbb { R } ^ { \hat { N _ { r g b } } \times N _ { i r } }$ is then computed as

$$
S ( i , j ) = \tau ^ { - 1 } \langle W _ { r g b } \hat { x } _ { r g b } ^ { i } , W _ { i r } \hat { x } _ { i r } ^ { j } \rangle ,
$$

where $W _ { r g b }$ and $W _ { i r }$ denote learnable linear projections, $\langle \cdot , \cdot \rangle$ denotes the inner product, and τ is a temperature parameter. Row- and column-wise softmax yield bidirectional

confidence matrices:

$$
\begin{array} { r } { P _ { r g b  i r } ( i , j ) = \mathrm { S o f t m a x } _ { j } \big ( S ( i , \cdot ) \big ) , } \\ { P _ { i r  r g b } ( i , j ) = \mathrm { S o f t m a x } _ { i } \big ( S ( \cdot , j ) \big ) . } \end{array}
$$

For each visible token, we select the infrared token with the highest confidence in $P _ { r g b  i r } ;$ conversely, for each infrared token, we select the most confident visible token in $P _ { i r  r g b } .$ Candidates whose confidence is below $\theta _ { c }$ are discarded. The resulting coarse correspondence set is

$$
\begin{array} { r l r } & { } & { \mathcal { M } _ { c } = \left\{ ( i , j ) \bigg | j = \arg \operatorname* { m a x } _ { j ^ { \prime } } P _ { r g b \to i r } ( i , j ^ { \prime } ) , P _ { r g b \to i r } ( i , j ) \geq \theta _ { c } \right\} \cup } \\ & { } & { \left. \left\{ ( i , j ) \bigg | i = \arg \operatorname* { m a x } _ { i ^ { \prime } } P _ { i r \to r g b } ( i ^ { \prime } , j ) , P _ { i r \to r g b } ( i , j ) \geq \theta _ { c } \right\} . \right. } \end{array}
$$

Unlike strict one-to-one assignment, this bidirectional strategy naturally preserves one-to-many candidates at the coarse stage, improving robustness to large geometric discrepancies. In addition, invalid padded regions and boundary tokens are masked to suppress unreliable matches. The resulting $\mathcal { M } _ { c }$ serve as initial matches for the subsequent fine-grained refinement and affine transformation estimation.

## 3.3.2 Fine-Grained Matching

Based on the coarse matches $\mathcal { M } _ { c } .$ , fine-grained matching further refines each pair within local neighborhoods. Specifically, for each coarse correspondence $( i , j ) \in \mathcal { M } _ { c } ,$ three pairs of local windows centered at the matched locations are extracted in a coarse-to-fine manner, with sizes $1 \times 1 , 3 \times 3$ , and $5 \times 5 ,$ , respectively. The $1 \times 1$ and $3 \times 3$ windows are first jointly encoded through self- and crossattention, injecting coarse context into the finer neighborhood. The refined $3 \times 3$ features then guide the $5 \times 5$ windows in the same manner, progressively enhancing each coarse match from coarse contextual guidance to a more discriminative fine-scale neighborhood. Denoting the resulting local features as $\tilde { f } _ { r g b } ^ { 5 \times 5 }$ and $\tilde { f } _ { i r } ^ { 5 \times 5 }$ , their similarity matrix $S _ { f } ^ { ( i , j ) }$ are constructed as

$$
S _ { f } ^ { ( i , j ) } ( u , v ) = \tau ^ { - 1 } \langle \tilde { f } _ { r g b } ^ { 5 \times 5 } ( u ) , \tilde { f } _ { i r } ^ { 5 \times 5 } ( v ) \rangle ,
$$

and $u , v$ index the local tokens in the two $5 \times 5$ windows. A dual-softmax operation is then applied to $S _ { f } ^ { ( i , j ) }$ to obtain the local confidence matrix

$$
P _ { f } ^ { ( i , j ) } ( u , v ) = \mathrm { S o f t m a x } _ { v } \left( S _ { f } ^ { ( i , j ) } ( u , \cdot ) \right) \cdot \mathrm { S o f t m a x } _ { u } \left( S _ { f } ^ { ( i , j ) } ( \cdot , v ) \right) .
$$

For each coarse correspondence, the highest-confidence local pair is retained when its confidence exceeds the threshold $\theta _ { f } ,$ , which suppresses ambiguous responses in the local neighborhood and yields a more reliable fine-grained match. To further reduce the discretization error introduced by window-based matching, the selected visible and infrared features are concatenated and fed into a lightweight regressor that predicts coordinate offsets for both modalities. The corrected local coordinates are mapped back to image space, yielding the final refined correspondence. All refined pairs form $\mathcal { M } _ { f }$ for subsequent affine estimation.

## 3.3.3 Feature Affine Registration

Given the refined correspondence set $\mathcal { M } _ { f }$ , CMAA estimates an image-level affine transformation from the infrared modality to the visible reference. Let $\begin{array} { r l } { \mathcal { M } _ { f } ^ { + } } & { { } = } \end{array}$ $\{ ( \mathbf { p } _ { r g b } ^ { n } , \mathbf { p } _ { i r } ^ { n } , w _ { n } ) \} _ { n = 1 } ^ { N }$ denote the confidence-filtered correspondences after pixel refinement, where $ { \mathbf { p } } _ { r g b } ^ { n }$ and $\mathbf { p } _ { i r } ^ { n }$ are matched image coordinates and $w _ { n }$ is the matching confidence. The transformation $\mathbf { T } _ { i r  r g b } \in \mathbb { R } ^ { 2 \times 3 }$ is obtained through confidence-weighted affine fitting:

$$
{ \bf T } _ { i r  r g b } = \arg \operatorname* { m i n } _ { \bf T } \sum _ { n = 1 } ^ { N } w _ { n } \| \mathbf { p } _ { r g b } ^ { n } - \mathbf { T } \bar { \mathbf { p } } _ { i r } ^ { n } \| _ { 2 } ^ { 2 } ,
$$

where $\bar { \mathbf { p } } _ { i r } ^ { n } = [ x _ { i r } ^ { n } , y _ { i r } ^ { n } , 1 ] ^ { \top }$ . Low-confidence matches are discarded, and the identity mapping is used when fewer than three valid correspondences are available or when the estimated transformation is geometrically unreliable.

The estimated transformation is then applied to each infrared feature level. For a feature map of size $H _ { l } \times W _ { l }$ , we define the image-to-level coordinate scaling matrix as

$$
\mathbf { S } _ { l } = \mathrm { d i a g } \left( W / W _ { l } , H / H _ { l } , 1 \right) ,
$$

where H and W are the input dimensions. Let $\bar { \mathbf { T } } _ { i r  r g b } \in$ $\mathbb { R } ^ { 3 \times 3 }$ be the homogeneous form of $\mathbf { T } _ { i r  r g b }$ . The affine transformation in the l-th feature coordinate system is given by $\bar { \mathbf { T } } _ { i r  r q b } ^ { l } = \mathbf { S } _ { l } ^ { - 1 } \bar { \mathbf { T } } _ { i r  r g b } \mathbf { S } _ { l }$ . Since differentiable grid sampling follows an inverse mapping scheme, the aligned infrared feature is obtained as

$$
\tilde { F } _ { i r } ^ { l } = \mathcal { W } ( F _ { i r } ^ { l } , [ ( \bar { \mathbf { T } } _ { i r  r g b } ^ { l } ) ^ { - 1 } ] _ { 1 : 2 , : } ) ,
$$

where $\mathcal { W } ( \cdot , \cdot )$ denotes bilinear feature warping. The resulting pyramid $\tilde { F } _ { i r } = \{ \tilde { F } _ { i r } ^ { 1 / 4 } , \tilde { F } _ { i r } ^ { 1 / 8 } , \tilde { F } _ { i r } ^ { 1 / 1 6 } , \tilde { F } _ { i r } ^ { \bar { 1 } / 3 \bar { 2 } } \}$ is used for subsequent cross-modal fusion and oriented detection.

## 3.4. Illumination-Guided Complementary Fusion

Given the visible feature pyramid $F _ { r g b } ~ = ~ \{ F _ { r g b } ^ { l } \} _ { l \in \mathcal { L } }$ and the aligned infrared feature pyramid $\tilde { F } _ { i r } = \{ \tilde { F } _ { i r } ^ { l } \} _ { l \in \mathcal { L } } ,$ where $\begin{array} { c c l } { \mathcal { L } } & { = } & { \left\{ 1 / 4 , 1 / 8 , 1 / 1 6 , 1 / 3 2 \right\} } \end{array}$ IGCF performs illumination-adaptive fusion on geometrically aligned features. Although CMAA reduces spatial misalignment, visible features remain unreliable under poor illumination, whereas infrared features generally provide more stable responses. IGCF therefore suppresses unreliable visible responses under poor illumination while enhancing complementary infrared information.

The illumination condition is estimated directly from the visible image without additional annotation. Given the visible image $\breve { I } _ { r g b } \in \mathbb { R } ^ { H \times W \times C }$ with pixel intensities in [0, 255], the illumination score $\eta \in [ 0 , 1 ]$ ] is computed as the normalized mean intensity of $I _ { r g b }$ . A darkness-aware factor $d ( \eta )$ is then obtained by clipping $( \eta _ { 0 } - \eta ) / \eta _ { 0 }$ to $[ 0 , 1 ]$ ， where $\eta _ { 0 }$ is an illumination threshold. Thus, $d ( \eta )$ increases in darker scenes and approaches zero under sufficient illumination.

For each feature level l, IGCF first applies lightweight feature refinement to obtain the enhanced responses $E _ { r g b } ^ { l }$ and $E _ { i r } ^ { l }$ . A spatial gate is generated from the discrepancy between the two responses:

$$
G ^ { l } = d ( \eta ) \cdot \mathcal { G } ^ { l } \left( \left| E _ { r g b } ^ { l } - E _ { i r } ^ { l } \right| \right) ,
$$

where $\mathcal { G } ^ { l } ( \cdot )$ predicts a spatially adaptive gating map. The global factor $d ( \eta )$ controls the illumination-dependent suppression strength, while the local discrepancy identifies inconsistent visible regions. IGCF further models shared and complementary cues through

$$
R _ { c c } ^ { l } = \phi _ { c c } ^ { l } \left( E _ { r g b } ^ { l } \odot E _ { i r } ^ { l } , \left| E _ { r g b } ^ { l } - E _ { i r } ^ { l } \right| \right) ,
$$

where the element-wise product captures shared responses, the absolute difference represents modality-specific cues, and $\phi _ { c c } ^ { l } ( \cdot )$ adaptively integrates both. The final fused feature is

$$
\begin{array} { r } { F _ { f u s } ^ { l } = F _ { r g b } ^ { l } + \tilde { F } _ { i r } ^ { l } + ( 1 - G ^ { l } ) \odot E _ { r g b } ^ { l } + E _ { i r } ^ { l } + R _ { c c } ^ { l } . } \end{array}
$$

Invalid regions introduced by infrared feature warping are masked during infrared-related computation. In this way, IGCF preserves aligned features, regulates visible responses according to illumination and local discrepancy, and incorporates complementary infrared cues for robust detection.

## 3.5. Alignment Quality-Consistency Gating

Although CMAA aligns infrared feature before fusion, the estimated transformation $\mathbf { T } _ { i r  r g b }$ can be unreliable early in training. Strong detection supervision on imperfectly aligned features may misguide detector optimization and interfere with alignment learning. As alignment improves and becomes consistent with the detection objective, stronger detection supervision becomes beneficial. To this end, AQCG adaptively reweights the detection loss according to alignment quality and gradient consistency, suppressing unreliable supervision in the early stage and progressively promoting detector learning as alignment improves.

Let $\mathcal { L } _ { a l i g n } ^ { t }$ and $\mathcal { L } _ { d e t } ^ { t }$ denote the alignment and detection losses at iteration t. AQCG maintains an exponential moving average $\bar { \mathcal { L } } _ { a l i g n } ^ { t }$ of the alignment loss with momentum coefficient $\beta ,$ , and computes the raw alignment quality as $Q ^ { t } = \exp \Bigl ( - \bar { \mathcal { L } } _ { a l i g n } ^ { t } / ( \mathcal { L } _ { r e f } + \epsilon ) \Bigr )$ , where $\mathcal { L } _ { \boldsymbol { r e f } }$ is a stabilized reference loss after warm-up and ϵ is a small constant. A lower moving-average alignment loss therefore yields a higher $Q ^ { t }$ , indicating more reliable geometric alignment. The quality score is then remapped into a signed gating signal:

$$
Q _ { s } ^ { t } = \operatorname { t a n h } \left( \kappa ( Q ^ { t } - \theta ) \right) ,
$$

where θ is the quality threshold and κ controls the transition sharpness. $Q ^ { t } < \theta$ suppresses detection learning, whereas $Q ^ { t } > \theta$ strengthens it.

To further avoid promoting detection when the two objectives conflict, AQCG measures the gradient consistency between alignment and detection with respect to the predicted affine transformation:

$$
C ^ { t } = \cos \lrcorner \sin ( \frac { \partial \mathcal { L } _ { a l i g n } ^ { t } } { \partial \mathbf { T } _ { i r  r g b } } , \frac { \partial \mathcal { L } _ { d e t } ^ { t } } { \partial \mathbf { T } _ { i r  r g b } } ) .
$$

The gating score and detection weight are computed as

$$
s ^ { t } = ( 1 - \lambda _ { c } ) Q _ { s } ^ { t } + \lambda _ { c } C ^ { t } , \qquad \alpha _ { d e t } ^ { t } = 2 ^ { s ^ { t } } ,
$$

where $\lambda _ { c } \in [ 0 , 1 ]$ balances alignment quality and gradient consistency. Since $Q _ { s } ^ { t } , C ^ { t } \in [ - 1 , 1 ]$ , the detection weight

is bounded by $\alpha _ { d e t } ^ { t } \in [ 0 . 5 , 2 . 0 ]$ . The overall objective is

$$
\begin{array} { r } { \mathcal { L } ^ { t } = \mathcal { L } _ { a l i g n } ^ { t } + \alpha _ { d e t } ^ { t } \mathcal { L } _ { d e t } ^ { t } . } \end{array}
$$

The gating weight is detached during back-propagation and serves only as a training-time regulator. In this way, AQCG limits the influence of unreliable alignment while strengthening detection supervision when the transformation is accurate and optimization-consistent.

## 4. Experiments

## 4.1. Datasets and Evaluation Metrics

Datasets. Experiments are conducted on the proposed DroneVehicle Misaligned (DVMA) benchmark, constructed from DroneVehicle [24]. It contains 28,439 UAVcaptured visible-infrared image pairs with oriented annotations for five vehicle categories: car, freight car, truck, bus, and van. DVMA introduces controlled affine perturbations to evaluate detection robustness under substantial cross-modal spatial misalignment. The training, validation, and test sets contain 17,990, 1,469, and 8,980 image pairs, respectively.

Evaluation Metrics. Following standard oriented object detection protocols, performance is reported using m $\mathrm { A P _ { 5 0 } }$ and m $\mathrm { A P _ { 5 0 : 9 5 } }$ . Specifically, m $\mathrm { A P _ { 5 0 } }$ reports AP at an IoU threshold of 0.50, reflecting performance under a relatively tolerant localization criterion. m ${ \mathrm { A P } } _ { 5 0 : 9 5 }$ averages AP over IoU thresholds from 0.50 to 0.95 at intervals of 0.05, providing a more comprehensive measure of localization accuracy.

## 4.2. Experiment Details

All experiments are implemented using the MMDetection and MMRotate frameworks with PyTorch 2.3.0 and CUDA 11.8 on two 24-GB RTX 3090 GPUs. For fair comparison, all visible-infrared image pairs are resized to a fixed resolution of 480 × 384. AdamW is used as the optimizer with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ and a weight decay of 0.05. All models are trained for 12 epochs. During training, the ground-truth annotations from the infrared modality are used as training labels, since the infrared images provide more complete target annotations in the original dataset.

## 4.3. Results Comparisons

Quantitative comparison. Table 2 reports quantitative comparisons with representative single-modal and visibleinfrared object detectors on DVMA. Overall, infrared detectors generally outperform their visible counterparts, indicating that infrared images provide more reliable target cues under challenging imaging conditions. S<sup>2</sup>ANet using infrared input achieves the best single-modal performance with an m ${ \mathrm { A P } } _ { 5 0 : 9 5 }$ of 36.0% and an m $\mathrm { A P _ { 5 0 } }$ of 61.6%, even surpassing several visible-infrared methods. This suggests that simply introducing an additional modality does not necessarily improve detection when large cross-modal geometric discrepancies exist. Although visible-infrared methods generally benefit from complementary cues, they still lack explicit geometric correction. For example, DMM with $\mathrm { S ^ { 2 } A N e t }$ achieves 35.1% $\mathrm { m A P _ { 5 0 : 9 5 } }$ and $6 6 . 7 \% \mathrm { \ m A P _ { 5 0 } }$ but remains limited under severe misalignment. In contrast, JFRDet achieves the best overall performance with an m ${ \mathrm { A P } } _ { 5 0 : 9 5 }$ of 36.1% and an m ${ \mathrm { . A P } } _ { 5 0 }$ of 69.7%. It also obtains 66.7% and 55.4% $\mathrm { A P _ { 5 0 } }$ on the truck and freight car categories, respectively, outperforming the best competing results of 58.7% and 50.7%. These gains demonstrate that explicit affine feature registration reduces geometric discrepancies and enables more reliable cross-modal fusion.

<table><tr><td>Method</td><td>Modality</td><td>Car</td><td>Truck</td><td>Bus</td><td>Van</td><td>Freight Car</td><td> $\mathbf { m A P _ { 5 0 : 9 5 } ( \% ) }$ </td><td> $\bf { m A P _ {5 0 } ( \% ) }$ </td></tr><tr><td>RetinaNet [16]</td><td rowspan="4">RGB</td><td>65.5</td><td>19.3</td><td>55.3</td><td>12.2</td><td>13</td><td>15.5</td><td>33.1</td></tr><tr><td>R3Det [31]</td><td>77.0</td><td>35.2</td><td>73.7</td><td>24.8</td><td>17.2</td><td>22.6</td><td>45.6</td></tr><tr><td>S2ANet [7]</td><td>78.0</td><td>43.6</td><td>77.1</td><td>27.8</td><td>27.5</td><td>25.6</td><td>50.8</td></tr><tr><td>KFIoU [34]</td><td>75.6</td><td>25.1</td><td>61.6</td><td>16.5</td><td>19.5</td><td>18.6</td><td>39.6</td></tr><tr><td rowspan="4">RetinaNet [16]  $\mathrm { R } ^ { \mathrm { 3 } } \mathrm { D e t } \left[ \mathrm { 3 1 } \right]$  S2 ANet [7] KFIoU [34]</td><td rowspan="4"></td><td>85.8</td><td>30.8</td><td>51.8</td><td>10.5</td><td>16.5</td><td>20.4</td><td>39.1</td></tr><tr><td>89.8 IR</td><td>39.7</td><td>83.2</td><td>24.6</td><td>30.7</td><td>32.2</td><td>53.6</td></tr><tr><td>90.2</td><td>55.9</td><td>87.2</td><td>33.9</td><td>40.8</td><td>36.0</td><td>61.6</td></tr><tr><td>88.6</td><td>29.1</td><td>74.9</td><td>12.6</td><td>21.2</td><td>24.5</td><td>45.3</td></tr><tr><td rowspan="4"> $\mathrm { C ^ { 2 } F o r m e r + F a s t e r \ : R { \mathrm { - } } C N N \ : [ 3 7 ] }$   $\mathrm { C ^ { 2 } F o r m e r + S ^ { 2 } A N e t \ [ 3 7 ] }$   $\mathrm { D M M + F a s t e r ~ R { \mathrm { - } C N N } ~ [ 4 4 ] }$   $\mathrm { D M M } + \mathrm { S } ^ { 2 } \mathrm { A N e t } \ [ 4 4 ]$ </td><td rowspan="6">RGB+IR</td><td>89.3</td><td>28.4</td><td>45.7</td><td>31.1</td><td>21.2</td><td>17.5</td><td>43.1</td></tr><tr><td>89.9</td><td>51.6</td><td>83.5</td><td>33.7</td><td>38.8</td><td>31.4</td><td>59.5</td></tr><tr><td>78.9</td><td>56.4</td><td>78.4</td><td>50.2</td><td>48.1</td><td>32.4</td><td>62.4</td></tr><tr><td>85.7</td><td>58.7</td><td>86.6</td><td>51.7</td><td>50.7</td><td>35.1</td><td>66.7</td></tr><tr><td>85.1</td><td>52.8</td><td>78.8</td><td>45.5</td><td>45.9</td><td>28.2</td><td>61.6</td></tr><tr><td>COMO [17] JFRDet (ours)</td><td></td><td>87.9</td><td>66.7</td><td>86.8 51.8</td><td>55.4</td><td></td><td>36.1</td><td>69.7</td></tr></table>

Table 2. Comprehensive comparative experiments on the DVMA dataset. We compared the JFRDet method with both single-modal and multispectral object detectors, all employing OBB detection heads. The best results are highlighted in bold.

![](images/ab6cdc2d220fa3342798f1fe8f10ddc16cc98694f79b34357097bb86ff62994a.jpg)  
Figure 4. Qualitative comparison on DVMA at a confidence threshold of 0.3. The last column shows visible ground truth. Blue dashed circles highlight the more accurate and complete detections of JFRDet across categories.

Qualitative comparison. Figure 4 compares the qualitative results of DMM with $\mathrm { S ^ { 2 } A { \cdot } N e t }$ , COMO, and JFRDet under different illumination conditions. In low-light scenes, DMM still suffers from missed detections when RGB and infrared features are spatially misaligned, whereas JFRDet produces more complete and accurate predictions by explicitly aligning infrared features before fusion. In wellilluminated scenes, JFRDet also generates more stable oriented bounding boxes for large objects such as trucks and buses, reducing inaccurate predictions caused by unreliable cross-modal aggregation. These results demonstrate that affine feature registration effectively alleviates geometric inconsistency, while illumination-guided fusion improves the use of complementary RGB-IR cues across varying illumination conditions.

## 4.4. Ablation study

Component-wise Ablation. Table 3 reports the component-wise ablation results on the DVMA dataset. The baseline directly fuses visible and infrared features without explicit geometric correction, achieving 66.0% m $\mathrm { A P _ { 5 0 } }$ and $3 4 . 3 \% \mathrm { m A P } _ { 5 0 : 9 5 }$ . After introducing CMAA, the performance increases to 67.3% m $\mathrm { A P _ { 5 0 } }$ , indicating that affine alignment helps reduce cross-modal spatial inconsistency before feature fusion. Based on the aligned features, IGCF further improves the result to $6 8 . 0 \% \mathrm { m A P _ { 5 0 } }$ by adaptively balancing visible and infrared cues under different illumination conditions. When AQCG is introduced, the performance reaches 68.2% m $\mathrm { A P _ { 5 0 } }$ and 35.0% m ${ \mathrm { A P } } _ { 5 0 : 9 5 }$ , showing that alignment reliability is important for stable optimization. With all components integrated, JFRDet achieves the best performance, demonstrating that CMAA, IGCF, and AQCG provide complementary benefits.

<table><tr><td rowspan="2">Case</td><td colspan="3">Component</td><td colspan="2">Performance (%)</td></tr><tr><td>CMAA</td><td>IGCF</td><td>AQCG</td><td> $\mathrm { m A P _ { 5 0 } }$ </td><td> $\mathrm { m A P _ { 5 0 : 9 5 } }$ </td></tr><tr><td>1</td><td>-</td><td>-</td><td>一</td><td>66.0</td><td>34.3</td></tr><tr><td>2</td><td>√</td><td>一</td><td>一</td><td>67.3</td><td>34.4</td></tr><tr><td>3</td><td>√</td><td>√</td><td>-</td><td>68.0</td><td>34.8</td></tr><tr><td>4</td><td>√</td><td>-</td><td>√</td><td>68.2</td><td>35.0</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>69.7</td><td>36.1</td></tr></table>

Table 3. Ablation experiments on the DVMA dataset.

<table><tr><td rowspan="2">θ</td><td colspan="2">w/o GC (%)</td><td colspan="2">w/ GC (%)</td></tr><tr><td> $\mathrm { m A P _ { 5 0 } }$ </td><td> $\mathrm { m A P _ { 5 0 : 9 5 } }$ </td><td> $\mathrm { m A P _ { 5 0 } }$ </td><td> $\mathrm { m A P _ { 5 0 : 9 5 } }$ </td></tr><tr><td>0.45</td><td>68.8</td><td>35.6</td><td>69.4</td><td>36.0</td></tr><tr><td>0.50</td><td>68.6</td><td>36.0</td><td>69.7</td><td>36.1</td></tr><tr><td>0.55</td><td>68.6</td><td>35.4</td><td>69.2</td><td>36.1</td></tr><tr><td>0.60</td><td>68.7</td><td>35.5</td><td>69.0</td><td>35.6</td></tr></table>

Table 4. Effect of gradient consistency (GC) in AQCG.

<table><tr><td rowspan="2">θ</td><td colspan="4"> $\lambda _ { c } / \mathrm { m A P _ { 5 0 } }$  (%)</td></tr><tr><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td></tr><tr><td>0.45</td><td>68.9</td><td>69.1</td><td>68.5</td><td>69.4</td></tr><tr><td>0.50</td><td>68.8</td><td>69.0</td><td>69.7</td><td>69.1</td></tr><tr><td>0.55</td><td>68.6</td><td>69.2</td><td>69.2</td><td>69.2</td></tr><tr><td>0.60</td><td>68.7</td><td>69.1</td><td>68.7</td><td>69.0</td></tr></table>

Table 5. Sensitivity analysis of AQCG with varying quality thresholds θ and balancing factors $\lambda _ { c } .$

Parameter Sensitivity of AQCG. Table 4 compares AQCG with and without gradient consistency, where “w/o GC” corresponds to $\lambda _ { c } = 0$ . Introducing gradient consistency generally improves performance, indicating that measuring the agreement between alignment and detection objectives provides more reliable gating signals. Table 5 further evaluates different quality thresholds θ and balancing factors $\lambda _ { c }$ with gradient consistency enabled. The best result is obtained at $\theta = 0 . 5 0$ and $\lambda _ { c } = 0 . 3$ , reaching $6 9 . 7 \% \mathrm { m A P _ { 5 0 } }$ Performance remains relatively stable across different settings, suggesting that AQCG is not overly sensitive to a specific hyperparameter choice. However, overemphasizing either alignment quality or gradient consistency may weaken the gating effect and lead to suboptimal performance.

Visual Analysis of CMAA. Figure 5 visualizes the $1 / 4 -$ resolution feature maps from the visible and infrared branches, together with the infrared features aligned by

![](images/0a09a5213442354a711f0feeae5a42ce9a147c74c273cc959318acf5a4a3507d.jpg)  
Figure 5. Visualization of feature alignment by CMAA. From left to right, each row shows the infrared image, the corresponding 1/4-resolution infrared feature map, the infrared feature transformed by CMAA, the 1/4-resolution visible feature map, and the visible image.

CMAA. Before alignment, the response regions of the two modalities exhibit clear spatial inconsistency, where target activations are shifted and partially mismatched. Directly fusing these features may aggregate responses from different physical locations, resulting in ambiguous cross-modal representations. After applying CMAA, the infrared responses are better registered to the visible reference. The activation centers and object structures become more consistent across modalities, while misaligned background responses are reduced. This visualization further confirms that CMAA effectively corrects feature-level geometric discrepancies and provides a more reliable basis for crossmodal fusion.

## 5. Conclusion

This paper presents JFRDet, an end-to-end visibleinfrared oriented object detector for severe cross-modal geometric misalignment. Different from implicit feature adaptation under weak misalignment, JFRDet performs explicit feature-domain affine registration before crossmodal fusion, and integrates illumination-guided fusion with alignment-aware optimization. To support systematic evaluation, the DVMA benchmark is introduced to provide a challenging setting for misaligned visible-infrared oriented object detection. Experiments on DVMA show that JFRDet achieves superior performance over representative single-modal and visible-infrared detectors, while ablation studies and visual analyses further verify the effectiveness of explicit affine feature registration.

## References

[1] Chen Chen, Jiahao Qi, Xingyue Liu, Kangcheng Bin, Ruigang Fu, Xikun Hu, and Ping Zhong. Weakly misalignment-free adaptive feature alignment for uavsbased multimodal object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26836–26845, June 2024. 1

[2] Yunfan Chen, Han Xie, and Hyunchul Shin. Multi-layer fusion techniques using a cnn for multispectral pedestrian detection. IET Computer Vision, 12(8):1179–1187, 2018. 3

[3] Jian Ding, Nan Xue, Yang Long, Gui-Song Xia, and Qikai Lu. Learning roi transformer for oriented object detection in aerial images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2849– 2858, 2019. 2

[4] Haolong Fu, Jin Yuan, Guojin Zhong, Xuan He, Jiacheng Lin, and Zhiyong Li. Cf-deformable detr: An end-to-end alignment-free model for weakly aligned visible-infrared object detection. In IJCAI, pages 758–766, 2024. 1

[5] Junjie Guo, Chenqiang Gao, Fangcen Liu, Deyu Meng, and Xinbo Gao. Damsdet: Dynamic adaptive multispectral detection transformer with competitive query selection and adaptive feature fusion. In European Conference on Computer Vision, pages 464–481. Springer, 2024. 1

[6] Zonghao Guo, Chang Liu, Xiaosong Zhang, Jianbin Jiao, Xiangyang Ji, and Qixiang Ye. Beyond bounding-box: Convexhull feature adaptation for oriented and densely packed object detection. In Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, pages 8792– 8801, 2021. 2

[7] Jiaming Han, Jian Ding, Jie Li, and Gui-Song Xia. Align deep features for oriented object detection. IEEE transactions on geoscience and remote sensing, 60:1–11, 2021. 3, 7

[8] Mingzhou He, Qingbo Wu, King Ngi Ngan, Feng Jiang, Fanman Meng, and Linfeng Xu. Misaligned rgb-infrared object detection via adaptive dual-discrepancy calibration. Remote Sensing, 15(19):4887, 2023. 1

[9] Soonmin Hwang, Jaesik Park, Namil Kim, Yukyung Choi, and In So Kweon. Multispectral pedestrian detection: Benchmark dataset and baseline. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1037–1045, 2015. 1

[10] Xinyu Jia, Chuang Zhu, Minzhen Li, Wenqi Tang, and Wenli Zhou. Llvip: A visible-infrared paired dataset for lowlight vision. In 2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pages 3489–3497, 2021. 1

[11] Yingying Jiang, Xiangyu Zhu, Xiaobing Wang, Shuli Yang, Wei Li, Hua Wang, Pei Fu, and Zhenbo Luo. R2cnn: Rotational region cnn for orientation robust scene text detection. arXiv preprint arXiv:1706.09579, 2017. 2

[12] Chengyang Li, Dan Song, Ruofeng Tong, and Min Tang. Multispectral pedestrian detection via simultaneous detection and segmentation. arXiv preprint arXiv:1808.04818, 2018. 3

[13] Chenglong Li, Dan Song, Ruofeng Tong, and Ming Tang. Illumination-aware faster R-CNN for robust multispectral pedestrian detection. Pattern Recognition, 85:161–171, 2019. 1

[14] Ruimin Li, Jiajun Xiang, Feixiang Sun, Ye Yuan, Longwu Yuan, and Shuiping Gou. Multiscale cross-modal homogeneity enhancement and confidence-aware fusion for multispectral pedestrian detection. IEEE Transactions on Multimedia, 26:852–863, 2023. 3

[15] Wentong Li, Yijie Chen, Kaixuan Hu, and Jianke Zhu. Oriented reppoints for aerial object detection. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 1829–1838, 2022. 2

[16] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense object detection. In´ Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017. 7

[17] Chang Liu, Xin Ma, Xiaochen Yang, Yuxiang Zhang, and Yanni Dong. Como: Cross-mamba interaction and offsetguided fusion for multimodal object detection. Information Fusion, 125:103414, 2026. 7

[18] Jingjing Liu, Shaoting Zhang, Shu Wang, and Dimitris N Metaxas. Multispectral deep neural networks for pedestrian detection. 2016. 1

[19] Jianqi Ma, Weiyuan Shao, Hao Ye, Li Wang, Hong Wang, Yingbin Zheng, and Xiangyang Xue. Arbitrary-oriented scene text detection via rotation proposals. IEEE transactions on multimedia, 20(11):3111–3122, 2018. 2

[20] Fang Qingyun, Han Dapeng, and Wang Zhaokui. Crossmodality fusion transformer for multispectral object detection. arXiv preprint arXiv:2111.00273, 2021. 3

[21] Fang Qingyun and Wang Zhaokui. Cross-modality attentive feature fusion for object detection in multispectral remote sensing imagery. Pattern Recognition, 130:108786, 2022. 3

[22] Jifeng Shen, Yifei Chen, Yue Liu, Xin Zuo, Heng Fan, and Wankou Yang. Icafusion: Iterative cross-attention guided feature fusion for multispectral object detection. Pattern Recognition, 145:109913, 2024. 3

[23] Kechen Song, Xiaotong Xue, Hongwei Wen, Yingying Ji, Yunhui Yan, and Qinggang Meng. Misaligned visiblethermal object detection: A drone-based benchmark and baseline. IEEE Transactions on Intelligent Vehicles, 2024. 1

[24] Yiming Sun, Bing Cao, Pengfei Zhu, and Qinghua Hu. Drone-based rgb-infrared cross-modality vehicle detection via uncertainty-aware learning. IEEE Transactions on Circuits and Systems for Video Technology, 32(10):6700–6713, 2022. 2, 3, 6

[25] Jorg Wagner, Volker Fischer, Michael Herman, Sven¨ Behnke, et al. Multispectral pedestrian detection using deep fusion convolutional neural networks. In ESANN, volume 587, pages 509–514, 2016. 3

[26] Jin Xie, Rao Muhammad Anwer, Hisham Cholakkal, Jing Nie, Jiale Cao, Jorma Laaksonen, and Fahad Shahbaz Khan. Learning a dynamic cross-modal network for multispectral pedestrian detection. In Proceedings of the 30th ACM International Conference on Multimedia, pages 4043–4052, 2022. 3

[27] Xingxing Xie, Gong Cheng, Jiabao Wang, Xiwen Yao, and Junwei Han. Oriented r-cnn for object detection. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 3520–3529, 2021. 2

[28] Yongchao Xu, Mingtao Fu, Qimeng Wang, Yukang Wang, Kai Chen, Gui-Song Xia, and Xiang Bai. Gliding vertex on the horizontal bounding box for multi-oriented object detection. IEEE transactions on pattern analysis and machine intelligence, 43(4):1452–1459, 2020. 2

[29] Xue Yang, Liping Hou, Yue Zhou, Wentao Wang, and Junchi Yan. Dense label encoding for boundary discontinuity free rotation detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15819–15829, 2021. 3

[30] Xue Yang and Junchi Yan. Arbitrary-oriented object detection with circular smooth label. In European conference on computer vision, pages 677–694. Springer, 2020. 3

[31] Xue Yang, Junchi Yan, Ziming Feng, and Tao He. R3det: Refined single-stage detector with feature refinement for rotating object. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 3163–3171, 2021. 2, 7

[32] Xue Yang, Junchi Yan, Qi Ming, Wentao Wang, Xiaopeng Zhang, and Qi Tian. Rethinking rotated object detection with gaussian wasserstein distance loss. In International conference on machine learning, pages 11830–11841. PMLR, 2021. 3

[33] Xue Yang, Xiaojiang Yang, Jirui Yang, Qi Ming, Wentao Wang, Qi Tian, and Junchi Yan. Learning high-precision bounding box for rotated object detection via kullbackleibler divergence. Advances in Neural Information Processing Systems, 34:18381–18394, 2021. 3

[34] Xue Yang, Yue Zhou, Gefan Zhang, Jirui Yang, Wentao Wang, Junchi Yan, Xiaopeng Zhang, and Qi Tian. The kfiou loss for rotated object detection. arXiv preprint arXiv:2201.12558, 2022. 3, 7

[35] Maoxun Yuan, Xiaorong Shi, Nan Wang, Yinyan Wang, and Xingxing Wei. Improving rgb-infrared object detection with cascade alignment-guided transformer. Information Fusion, 105:102246, 2024. 1

[36] Maoxun Yuan, Yinyan Wang, and Xingxing Wei. Translation, scale and rotation: cross-modal alignment meets rgbinfrared vehicle detection. In European Conference on Computer Vision, pages 509–525. Springer, 2022. 1

[37] Maoxun Yuan and Xingxing Wei. C<sup>2</sup>former: Calibrated and complementary transformer for rgb-infrared object detection. IEEE Transactions on Geoscience and Remote Sensing, 62:1–12, 2024. 7

[38] Lu Zhang, Zhiyong Liu, Xiangyu Zhu, Zhan Song, Xu Yang, Zhen Lei, and Hong Qiao. Weakly aligned feature fusion for multimodal object detection. IEEE Transactions on Neural Networks and Learning Systems, 2021. 1

[39] Lu Zhang, Xiangyu Zhu, Xiangyu Chen, Xu Yang, Zhen Lei, and Zhiyong Liu. Weakly aligned cross-modal learning for multispectral pedestrian detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2019. 1

[40] Xue Zhang, Xiaohan Zhang, Jiangtao Wang, Jiacheng Ying, Zehua Sheng, Heng Yu, Chunguang Li, and Hui-Liang Shen. Tfdet: Target-aware fusion for rgb-t pedestrian detection. IEEE Transactions on Neural Networks and Learning Systems, 36(7):13276–13290, 2024. 3

[41] Tianyi Zhao, Maoxun Yuan, Feng Jiang, Nan Wang, and Xingxing Wei. Removal then selection: A coarse-to-fine fusion perspective for rgb-infrared object detection. IEEE Transactions on Intelligent Transportation Systems, 2025. 3

[42] Yang Zheng, Izzat H Izzat, and Shahrzad Ziaee. Gfd-ssd: Gated fusion double ssd for multispectral pedestrian detection. arXiv preprint arXiv:1903.06999, 2019. 3

[43] Kailai Zhou, Linsen Chen, and Xun Cao. Improving multispectral pedestrian detection by addressing modality imbalance problems. In European conference on computer vision, pages 787–803. Springer, 2020. 3

[44] Minghang Zhou, Tianyu Li, Chaofan Qiao, Dongyu Xie, Guoqing Wang, Ningjuan Ruan, Lin Mei, Yang Yang, and Heng Tao Shen. Dmm: Disparity-guided multispectral mamba for oriented object detection in remote sensing. IEEE Transactions on Geoscience and Remote Sensing, 63:1–13, 2025. 3, 7