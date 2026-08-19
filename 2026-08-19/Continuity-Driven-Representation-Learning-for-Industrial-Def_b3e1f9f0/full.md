# Continuity-Driven Representation Learning for Industrial Defect Detection

Minjong Kim mj9327@cau.ac.kr

Chung-Ang University Seoul, Republic of Korea

Hyun Jun Kim hyunjun0615@cau.ac.kr

<sup>0</sup>Jeongrae Kim <sup>2</sup>kjk632@cau.ac.kr

<sup>g</sup>Heeseung Shin hs970416@cau.ac.kr Changwon Lim <sup>8</sup>clim@cau.ac.kr

## Abstract

Industrial defect detection differs from natural-image object detection because inspection images are captured under controlled conditions and contain large normal-dominant regions with repetitive structures. Defects therefore appear as localized disruptions of otherwise predictable patterns, while conventional detectors rely mainly on sparse boundingbox supervision, resulting in weakly constrained normal-region representations. We propose a continuity-driven representation regularization framework that exploits normaldominant regions as dense auxiliary supervision. The framework introduces two detectoragnostic objectives: Multi-Continuity Loss, which combines 1D patch-sequence prediction and 2D masked spatial prediction, and Differencing Loss, which regularizes firstorder feature variation and second-order curvature between neighboring patch embeddings. Both objectives are applied with box-derived region weighting to stabilize normalregion representations while preserving defect-related discontinuities.

Experiments on two real-world industrial datasets and the public NEU-DET benchmark, using six detector architectures including YOLO-family models, MambaYOLO, and DETR, demonstrate consistent improvements over native detector baselines. In the full-data setting, the proposed regularizers improve average mAP@0.5:0.95 by up to 3.49 percentage points on Industrial Metal, 5.38 percentage points on MEA, and 5.03 percentage points on NEU-DET. Under limited-data conditions, the gains become more pronounced, with Differencing Loss achieving improvements of up to 21.07 percentage points in mAP@0.5 and 8.23 percentage points in mAP@0.5:0.95 on NEU-DET using only 25% of the training data. These results suggest that continuity-driven regularization provides an effective prior for improving industrial defect detection, particularly when annotated data are scarce.

![](images/f1af2d1fdb5204aba395f18e5bfd9ba5b685d60eef9a6edfaf293e331089c384.jpg)  
(a) Natural images

![](images/879dd3ed4b730d70c6783128b7bb1bdf2982b2050ec36f6c663b16ad5c063549.jpg)  
(b) Industrial inspection images  
Figure 1: Comparison between natural-image object detection and industrial defect detection. Industrial inspection images contain repetitive normal structures, where defects appear as localized disruptions of structural continuity.

## 1 Introduction

Automated visual inspection is a central problem in industrial quality control, where visual defects must be detected reliably under high-throughput manufacturing conditions [5, 10, 15]. Although modern object detectors such as Faster R-CNN, RetinaNet, YOLO, DETR, and their variants have been widely adopted for defect localization [3, 14, 16, 17, 31], industrial defect detection differs substantially from natural-image object detection. In parallel, industrial anomaly detection methods model normal patterns and identify deviations from them, reducing the dependence on dense defect annotations [1, 2, 4, 6, 18, 26]. However, these anomaly detection approaches are often not directly optimized for supervised bounding-box localization, whereas object detectors rely mainly on sparse defect-box supervision. This gap motivates our goal: retaining the localization capability of object detectors while exploiting normal-region structure as an additional source of representation-level supervision.

Natural image benchmarks such as MS COCO, PASCAL VOC, and Open Images contain diverse object categories, backgrounds, poses, and semantic contexts [7, 11, 13], whereas industrial inspection images are often acquired under fixed viewpoints, controlled illumination, and predefined regions of interest [5, 10, 21]. As a result, most pixels correspond to normal structures that are repetitive and spatially regular, while defects are typically small, localized, and visually subtle. Figure 1 illustrates this distinction: while natural-image detection focuses on semantic object instances under diverse contexts, industrial defect detection often requires identifying localized structural violations within repetitive normal patterns.

Industrial defects should not be regarded only as independent object instances. In many inspection scenarios, defects appear as localized disruptions of otherwise predictable normal patterns, such as scratches, contamination, or cracks on repetitive structures. Therefore, effective defect detection requires not only learning discriminative defect features, but also learning stable representations of the surrounding normal-dominant regions. However, conventional object detectors rely mainly on sparse bounding-box supervision, which provides limited constraints on normal-region representations occupying most of the image. Our continuity objectives complement sparse, defect-centric box supervision with dense feature-level constraints over normal-dominant regions, while box-derived weighting prevents defect-related discontinuities from being over-regularized. Our approach specifically targets supervised industrial inspection settings where normal regions exhibit repetitive and relatively homogeneous structures. To address this issue, we propose a continuity-driven representation regularization framework for industrial defect detection. Our key idea is to exploit normal-dominant regions as dense auxiliary supervision by encouraging predictable patch-level feature transitions while down-weighting annotated defect regions to preserve defect-related discontinuities. The proposed regularizers are applied to intermediate feature maps during training without modifying the detector architecture or native detection objectives. We introduce two independent auxiliary objectives. Multi-Continuity Loss combines 1D patch-sequence prediction and 2D masked spatial prediction to model ordered representation flow and local spatial consistency. Differencing Loss regularizes first-order feature variation and second-order curvature between neighboring patch embeddings to stabilize structural representation trends in normal regions. Both losses employ box-derived region weighting so that continuity regularization is mainly enforced on normal-dominant areas.

Our contributions are summarized as follows:

1. We formulate supervised industrial defect detection from a structural-continuity perspective, where normal regions form predictable feature fields and defects are treated as localized violations of this continuity.

2. We propose Multi-Continuity Loss, a detector-agnostic auxiliary objective that combines 1D patch-sequence prediction and 2D masked spatial prediction for continuityaware feature regularization.

3. We introduce Differencing Loss, which regularizes first-order feature variation and second-order curvature to stabilize normal-region representation trends.

4. We incorporate box-derived region weighting and validate the proposed regularizers across two real-world industrial defect datasets, six detector architectures, and datascarce training settings.

## 2 Related Work

## 2.1 Industrial Defect Detection and Normality Modeling

Deep learning-based object detectors have been widely adopted for supervised industrial defect localization because they directly predict bounding boxes and class labels for defective regions [5, 10]. CNN-based detectors, including Faster R-CNN, YOLO, and RetinaNet, formulate detection through classification and localization objectives, while Transformer-based detectors such as DETR and Deformable DETR further improve global context modeling through attention mechanisms [3, 14, 16, 17, 31]. However, industrial inspection images differ substantially from natural-image datasets because they are typically acquired under controlled conditions and contain repetitive normal structures [5, 10, 21]. As a result, defects often appear as localized disruptions of otherwise predictable normal patterns, while detector supervision remains sparse and concentrated mainly on annotated defect regions.

Industrial anomaly detection models normality through distribution- or memory-based representations [6, 18] and student–teacher feature matching [2, 26], while recent methods have also explored synthetic or diffusion-based anomaly generation [30]. These approaches primarily target anomaly scoring or localization maps, whereas our framework retains the native supervised bounding-box objective and uses continuity regularization as auxiliary representation-level supervision.

## 2.2 Patch-Level Representation and Spatial Consistency Learning

Patch-level representation learning has become an important component of industrial anomaly detection and inspection because industrial defects often appear as localized disruptions of repetitive normal structures [1, 4, 12]. Representative methods such as PaDiM, PatchCore, and ReConPatch model normal-region representations or feature consistency for anomaly localization [6, 8, 18]. More recent approaches further introduce spatial-consistency regularization to improve anomaly localization performance [9]. Smoothness priors such as total variation (TV) and Laplacian regularization have been widely used to suppress local fluctuations in images or feature maps. However, uniform smoothness may suppress localized discontinuities that provide important defect evidence. Our Differencing Loss is related to such smoothness regularization but differs in two aspects. First, it jointly regularizes firstand second-order feature variations to capture local transitions and their curvature. Second, it employs box-derived region weighting to selectively regularize normal-dominant regions while preserving defect-related discontinuities. Detailed comparisons with TV and Laplacian regularization across different training-data ratios are provided in Appendix A. Differencing Loss generally improves detection performance over these conventional smoothness regularizers, particularly under data-scarce settings.

Unlike existing patch-level approaches primarily designed for anomaly scoring or anomaly map generation, our framework targets supervised bounding-box localization. It retains the native detection objective and applies continuity-driven auxiliary regularization to intermediate feature maps, enabling existing detectors to exploit structural continuity without modifying the detector architecture.

## 2.3 Data-Efficient Industrial Inspection

Industrial defect datasets often contain limited annotated defects because defective samples are rare and costly to collect [1, 2, 4, 5, 12]. Prior approaches have explored data augmentation and synthetic anomaly generation to improve generalization under limited-data settings [19, 28, 29]. However, unrealistic transformations or synthetic anomalies may not accurately reflect real industrial defects or imaging conditions [5, 21].

Instead of generating additional defect samples, our framework exploits normal-dominant regions already present in labeled detection images. The proposed continuity regularization provides dense auxiliary supervision from local feature relationships, improving representation stability under limited-data settings without requiring synthetic anomalies or additional labels.

## 3 Method

## 3.1 Overall Framework

Let $f _ { \theta }$ denote a baseline object detector and $\mathcal { L } _ { \mathrm { d e t } }$ its native detection loss. The form of $\mathcal { L } _ { \mathrm { d e t } }$ depends on the detector architecture, such as dense detection losses for YOLO-family detectors and Hungarian matching-based losses for DETR. We do not modify the detector

![](images/adb492e4634f548003a1c7d6f84f3944e58ee10c6f2cbbffe383f215aa7b81cf.jpg)  
Figure 2: Overall framework of the proposed continuity-driven representation regularization. Intermediate detector features are regularized using Multi-Continuity Loss or Differencing Loss with boxderived region weighting during training.

architecture, detection head, or detector-specific objective. Instead, we add an auxiliary feature-level regularizer to intermediate spatial representations during training.

Figure 2 illustrates the proposed framework, where intermediate detector features are regularized using either Multi-Continuity Loss or Differencing Loss. The auxiliary modules are used only during training and introduce no inference-time overhead.

Let $s$ be the set of selected feature sources. For each source $s \in S$ , we denote the corresponding spatial feature map by

$$
F ^ { s } \in \mathbb { R } ^ { B \times C _ { s } \times H _ { s } \times W _ { s } } ,
$$

where $B , C _ { s } , H _ { s } ,$ , and $W _ { s }$ denote the batch size, channel dimension, height, and width, respectively. A lightweight projection layer $\phi _ { s }$ , implemented as a $1 \times 1$ convolution, maps each feature map to a common embedding space:

$$
Z ^ { s } = \phi _ { s } ( F ^ { s } ) , \quad Z ^ { s } \in \mathbb { R } ^ { B \times D \times H _ { s } \times W _ { s } } ,
$$

where $D$ is the embedding dimension used for the proposed regularization losses. The projected feature map $Z ^ { s }$ is used only for auxiliary regularization and is discarded during inference.

We consider two independent regularization variants: Multi-Continuity Loss and Differencing Loss. They are not optimized jointly in our main experiments. For the Multi-Continuity variant, the training objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { d e t } } + \lambda _ { m c } \mathcal { L } _ { M C } , } \end{array}
$$

whereas for the Differencing variant, the objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { d e t } } + \lambda _ { d i f f } \mathcal { L } _ { D i f f } .
$$

Both auxiliary losses are applied with the same box-derived region weighting scheme, described in Section 3.4, to emphasize normal-dominant regions while down-weighting annotated defect boxes.

## 3.2 Multi-Continuity Loss

The Multi-Continuity Loss models feature continuity from two complementary perspectives. The first term models feature transitions along a one-dimensional patch sequence derived from a two-dimensional feature map, while the second term preserves local spatial adjacency through masked-convolutional prediction in the original feature map domain.

For each selected feature source $s \in { \mathcal { S } }$ , the projected feature map $Z ^ { s }$ is used to compute both 1D and 2D continuity terms.

## 3.2.1 1D masked sequence prediction

For each projected feature map $Z ^ { s }$ , we flatten the spatial dimensions into a patch sequence:

$$
X _ { s } = [ x _ { 1 } , x _ { 2 } , \cdots x _ { N } ] , \ N = H _ { s } W _ { s } ,
$$

where $x _ { i } \in \mathbb { R } ^ { D }$ denotes the embedding of the i-th spatial location under a fixed raster-scan ordering. This ordering is used only as a computational device for local context prediction; we do not assume that images possess an intrinsic temporal order.

A lightweight one-dimensional predictor $g _ { 1 D }$ estimates each patch embedding from its masked local sequence context:

$$
\widehat { x } _ { i } = g _ { 1 D } ( \mathcal { C } _ { i } ^ { 1 D } ) ,
$$

where $\mathcal { C } _ { i } ^ { 1 D }$ denotes the allowed neighboring context around location i. The location-wise discrepancy is defined as

$$
d _ { i } ^ { 1 D } = \delta ( \widehat { x } _ { i } , x _ { i } ) ,
$$

where $\delta ( \cdot , \cdot )$ is a feature discrepancy measure such as $L _ { 1 }$ or cosine distance. This term encourages predictable feature transitions in normal regions. Collecting the discrepancies over all patch locations, we obtain a location-wise 1D continuity discrepancy sequence

$$
{ \bf d } _ { s } ^ { 1 D } = \left[ d _ { 1 } ^ { 1 D } , d _ { 2 } ^ { 1 D } , \dots , d _ { N } ^ { 1 D } \right] .
$$

For compatibility with the spatial weight map, this sequence can be reshaped back to the feature-map resolution:

$$
D _ { s } ^ { 1 D } \in \mathbb { R } ^ { H _ { s } \times W _ { s } } .
$$

## 3.2.2 2D center-masked spatial prediction

Although the 1D formulation captures ordered feature transitions, flattening may weaken the original 2D spatial structure. To address this, we introduce a 2D masked-convolutional prediction term. Given the projected feature map $Z ^ { s } \in \mathbb { R } ^ { B \times D \times H _ { s } \times W _ { s } }$ , we predict each embedding from its local spatial neighborhood using a center-masked convolutional predictor $g _ { 2 D }$ where the center location is excluded from prediction:

$$
\widehat { z } _ { u , \nu } = g _ { 2 D } ( \mathcal { N } _ { u , \nu } ^ { 2 D } \backslash z _ { u , \nu } )
$$

$$
d _ { u , \nu } ^ { 2 D } = \delta \left( \widehat { z } _ { u , \nu } , z _ { u , \nu } \right) .
$$

The corresponding discrepancy is defined as follows, and all location-wise discrepancies form the 2D continuity discrepancy map $D _ { s } ^ { 2 D }$

Unlike generic pairwise smoothness, this term encourages each normal-region representation to be predictable from its local spatial context without enforcing neighboring features to become identical. The 1D term models ordered feature transitions, while the 2D term preserves spatial adjacency. Together, they provide complementary continuity constraints. The final weighted Multi-Continuity Loss is defined in Section 3.4 using box-derived region weighting.

## 3.3 Differencing Loss

The Differencing Loss is a separate continuity-driven regularization strategy. Unlike Multi-Continuity Loss, which regularizes feature representations through context-based prediction, Differencing Loss directly constrains the variation structure of neighboring patch embeddings. It is motivated by the observation that normal-dominant regions in industrial inspection images often exhibit stable feature transitions, whereas defects or irregular structures can introduce abrupt changes in local representation trajectories.

For each selected feature source $s \in S$ , we use the same flattened patch sequence

$$
X _ { s } = [ x _ { 1 } , x _ { 2 } , \cdots x _ { N } ] , \ N = H _ { s } W _ { s } ,
$$

defined in Section 3.2. The raster-scan ordering is used only as a computational device for defining local feature transitions, rather than as an assumption that images have an intrinsic temporal structure.

## 3.3.1 First-order feature variation

The first-order difference measures the local change between adjacent patch embeddings:

$$
\Delta x _ { i } = x _ { i + 1 } - x _ { i } , \ i = 1 , \dots , N - 1 .
$$

The corresponding location-wise first-order variation term is defined as

$$
d _ { i } ^ { ( 1 ) } = \| \Delta x _ { i } \| _ { 1 }
$$

or, alternatively, by a cosine-based feature discrepancy. This term discourages abrupt feature changes in normal-dominant regions and encourages neighboring patch representations to follow stable local transitions.

## 3.3.2 Second-order curvature variation

While the first-order term controls the magnitude of local feature changes, it does not explicitly constrain how these changes evolve across neighboring locations. We therefore introduce a second-order difference:

$$
\Delta ^ { 2 } x _ { i } = x _ { i + 1 } - 2 x _ { i } + x _ { i - 1 } , \ i = 2 , \ldots , N - 1 .
$$

The corresponding second-order variation term is

$$
d _ { i } ^ { ( 2 ) } = \big | \big | \Delta ^ { 2 } x _ { i } \big | \big | _ { 1 }
$$

This term penalizes unstable curvature in feature trajectories and encourages consistent variation patterns across normal-dominant regions. Unlike uniform smoothing, the proposed Differencing Loss uses box-derived region weighting to stabilize normal regions while down-weighting annotated defect areas, preventing excessive smoothing of defect-related discontinuities. The first-order and second-order terms define location-wise discrepancy sequences $d _ { s } ^ { ( 1 ) }$ and $d _ { s } ^ { ( 2 ) }$ , and the final weighted loss is formulated in Section 3.4 using the same weighting scheme as Multi-Continuity Loss.

## 3.4 Box-Derived Region Weighting

The proposed regularizers are intended to stabilize normal-dominant representations. However, defects themselves often appear as local violations of normal continuity, and applying continuity constraints uniformly may suppress defect-sensitive evidence. We therefore construct a box-derived weight map from the available bounding-box annotations.

For each image, spatial locations outside annotated defect boxes are assigned weight 1, while locations inside defect boxes are assigned a lower weight $\gamma \in [ 0 , 1 ]$

$$
W ( u , \nu ) = { \left\{ \begin{array} { l l } { \gamma , } & { ( u , \nu ) \in B , } \\ { 1 , } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }
$$

where $\boldsymbol { B }$ denotes the union of annotated defect boxes. The weight map is resized to the spatial resolution of each selected feature map. Since no boundary dilation is applied, this scheme should be interpreted as down-weighting annotated defect regions rather than explicitly removing defect boundaries.

For a two-dimensional discrepancy map $D _ { s }$ , the weighted aggregation is

$$
\mathrm { A g g } ( D _ { s } , W _ { s } ) = \frac { \sum _ { u , \nu } W _ { s } ( u , \nu ) D _ { s } ( u , \nu ) } { \sum _ { u , \nu } W _ { s } ( u , \nu ) + \varepsilon } ,
$$

where ε is a small constant for numerical stability. For sequence-based losses, $W _ { s }$ is flattened using the same raster-scan ordering as the patch sequence. For first-order differences, the weight associated with the target or ending location is used; for second-order differences, the weight associated with the center location is used. This aligns the weight sequence with the corresponding discrepancy terms.

Using this aggregation operator, the final Multi-Continuity Loss is defined as

$$
\mathcal { L } _ { M C } = \sum _ { s \in S } \left( \beta _ { 1 D } \mathrm { A g g } ( D _ { s } ^ { 1 D } , W _ { s } ) + \beta _ { 2 D } \mathrm { A g g } ( D _ { s } ^ { 2 D } , W _ { s } ) \right) ,
$$

where ${ D } _ { s } ^ { 1 D }$ and $D _ { s } ^ { 2 D }$ denote the location-wise 1D and 2D continuity discrepancy maps, and $\beta _ { 1 D }$ and $\beta _ { 2 D }$ control their relative contributions.

Similarly, the final Differencing Loss is defined as

$$
\mathcal { L } _ { D i f f } = \sum _ { s \in \cal S } \left( \alpha _ { 1 } \mathrm { A g g } ( d _ { s } ^ { ( 1 ) } , W _ { s } ) + \alpha _ { 2 } \mathrm { A g g } ( d _ { s } ^ { ( 2 ) } , W _ { s } ) \right) ,
$$

where $d _ { s } ^ { ( 1 ) }$ and $d _ { s } ^ { ( 2 ) }$ denote the first- and second-order discrepancy terms, and $\alpha _ { 1 }$ and $\alpha _ { 2 }$ control their contributions.

This region weighting distinguishes our method from global feature smoothing. Normaldominant regions receive stronger continuity constraints, while annotated defect regions are regularized less aggressively. As a result, the proposed losses stabilize predictable normal feature fields without enforcing uniform smoothness over defect-sensitive regions.

Robustness to box-scale variation. To evaluate robustness to annotation imprecision, we vary only the size of the auxiliary mask boxes derived from the ground-truth annotations while keeping all other training settings fixed. As shown in Table A1, Differencing Loss remains effective across a range of box scales, although performance varies non-monotonically with mask size. In particular, enlarged masks maintain strong performance, with the best performance obtained at 1.5×. Competitive performance is also maintained at smaller scales, indicating that the proposed regularization is not overly sensitive to moderate variations in the auxiliary mask size. Changing the mask scale mainly controls how much surrounding normal context is down-weighted during regularization.

## 3.5 Implementation Across Detector Architectures

The proposed regularizers are designed to be detector-agnostic while requiring intermediate representations that preserve spatial adjacency. Therefore, the auxiliary objectives are applied only to spatial feature maps or spatially arranged tokens, excluding unordered representations such as DETR object query embeddings. For YOLO-family detectors and MambaYOLO, the proposed losses are applied to multi-scale feature maps (P3–P5) used as inputs to the detection head. These feature maps capture defect-related information at different spatial resolutions and provide suitable representations for continuity regularization. Although the internal layer indices differ across detector implementations, the same principle of selecting spatial representations before the detection head is consistently maintained. The original detection heads and native training objectives remain unchanged. For DETR, the regularizers are not applied to object query embeddings because they do not preserve explicit spatial neighborhood relationships. Instead, the auxiliary losses are applied to backbone or encoderstage spatial representations that maintain the two-dimensional structure of the input image. This ensures that the continuity assumption is enforced only on representations where local neighborhood relationships remain meaningful. In all detector architectures, the original detection objectives are preserved, and the proposed losses are used only as auxiliary training objectives. Since the regularizers are removed during inference, the framework introduces no additional inference-time computation or parameters. Multi-Continuity Loss and Differencing Loss are evaluated independently rather than jointly optimized within the same training run.

## 4 Experiments

## 4.1 Experimental Setup

We evaluate the proposed continuity-driven representation regularization framework on six detector architectures: YOLOv8 [23], YOLOv10 [25], YOLO11 [24], YOLOv12 [22], MambaYOLO [27], and DETR [3]. Experiments compare three training settings: the native detector baseline, baseline + Multi-Continuity Loss, and baseline + Differencing Loss. The two proposed losses are evaluated independently and are not jointly optimized. For all detectors, the original architecture, detection head, and native detection objective are preserved, and the proposed losses are introduced only during training as auxiliary feature-level regularizers.

For YOLO-family detectors and MambaYOLO, the losses are applied to multi-scale detector-head input features, while for DETR they are applied to spatial backbone or encoder representations instead of unordered object queries. Auxiliary projection and prediction modules are removed during inference and therefore introduce no additional inferencetime computation.

Hyperparameters are selected using the validation set. Unless otherwise specified, $\gamma =$ 0.3 is used for Differencing Loss. Unless otherwise stated, feature discrepancies are computed using the $L _ { 1 }$ distance throughout all experiments. All results are reported on the heldout test set.

To evaluate robustness under limited annotated data, we additionally conduct reduceddata experiments using 75%, 50%, and 25% of the original training set while keeping validation and test splits fixed. Unless otherwise stated, all detectors use the same preprocessing and evaluation protocol within each dataset.

## 4.2 Datasets and Evaluation Metrics

We evaluate the proposed framework on two real-world industrial defect detection datasets: an Industrial Metal surface inspection dataset and a membrane-electrode assembly (MEA) inspection dataset. Both datasets were collected under controlled industrial inspection conditions and contain large normal-dominant regions with localized defects, making them suitable for evaluating continuity-driven representation regularization.

The Industrial Metal dataset contains 605 images (484 train / 60 val / 61 test) with an input resolution of 1280×1280. Defect sizes range from approximately $2 9 \times 2 9$ to $2 5 8 \times 2 5 8$ pixels, corresponding to about 0.05%–4.06% of the image area. The median defect size is approximately $9 6 \times 9 6$ pixels (∼ 0.56% of the image area), and over 71% of defects occupy less than 1% of the image.

The MEA dataset contains 242 images (128 train / 57 val / 57 test). Original inspection images reach resolutions up to $1 5 { , } 2 6 0 \times 5 { , } 4 5 3$ before preprocessing. After preprocessing, defect sizes range from approximately 19 × 19 to $5 8 \times 5 8$ pixels, corresponding to about 0.02%–0.21% of the image area, with a median size of approximately $5 0 \times 5 0$ pixels (∼ 0.16%). All annotated defects occupy less than 1% of the image area, making this an extreme small-defect detection setting. Compared with the Industrial Metal dataset, the MEA dataset exhibits more homogeneous and repetitive normal structures, which align well with the continuity assumption of the proposed framework.

Representative samples are shown in Figure 2. Validation and test sets are fixed across all experiments, including the reduced-data settings in Section 4.4. We report precision, recall, mAP@0.5, and mAP@0.5:0.95 on the test set. Since industrial defects are highly localized, mAP@0.5:0.95 provides a stricter evaluation of localization quality.

## 4.3 Main Results

Table 1 summarizes the full-data performance averaged across six detector architectures. Detailed detector-wise results are provided in Appendix A. On the Industrial Metal dataset, both regularization variants improve detection performance over the native detector baselines. Multi-Continuity improves average mAP@0.5 by 1.06 percentage points and mAP@0.5:0.95 by 2.28 percentage points, while Differencing improves mAP@0.5 by 1.57 percentage points and mAP@0.5:0.95 by 3.49 percentage points.

On the MEA dataset, baseline mAP@0.5 is already close to saturation. Nevertheless, both regularizers substantially improve mAP@0.5:0.95. Multi-Continuity improves average mAP@0.5:0.95 by 5.19 percentage points, while Differencing improves it by 5.38 percentage points. This suggests that continuity-driven regularization improves localization quality under stricter IoU thresholds even when coarse detection accuracy is already high.

![](images/b0f4f65ed7e4be5101925c6d089b08647e33e535cb79fb012f4134f5a6ade85a.jpg)  
Figure 3: Industrial inspection examples used for qualitative illustration. Left: an Industrial Metal inspection image. Right: schematic illustration of the MEA inspection region and localized defec examples. The schematic is used only for visualization and is not used for training or evaluation.

Table 1: Full-data average performance across six detector architectures. Values are averaged over YOLOv8, YOLOv10, YOLO11, YOLOv12, MambaYOLO, and DETR. Improvements are reported in percentage points relative to the corresponding native detector baselines.
<table><tr><td>Dataset</td><td>Method</td><td>mAP@0.5</td><td>Δ</td><td>mAP@0.5:0.95</td><td>Δ</td></tr><tr><td rowspan="3"></td><td>Baseline</td><td>0.936</td><td>一</td><td>0.656</td><td>1</td></tr><tr><td>Industrial Metal Multi-Continuity</td><td>0.946</td><td>+1.06</td><td>0.679</td><td>+2.28</td></tr><tr><td>Differencing</td><td>0.951</td><td>+1.57</td><td>0.691</td><td>+3.49</td></tr><tr><td rowspan="3">MEA</td><td>Baseline</td><td>0.993</td><td>一</td><td>0.890</td><td>一</td></tr><tr><td>Multi-Continuity</td><td>0.994</td><td>+0.15</td><td>0.942</td><td>+5.19</td></tr><tr><td>Differencing</td><td>0.994</td><td>+0.15</td><td>0.944</td><td>+5.38</td></tr></table>

Overall, the full-data results show that the proposed auxiliary losses improve detector performance without modifying the detector architecture or inference pipeline. Differencing tends to yield larger gains on average, while Multi-Continuity also provides consistent improvements through predictive feature regularization.

## 4.4 Variant-Level Ablation under Limited Training Data

We further analyze the proposed framework under different continuity regularization strategies and training-data regimes. Since Multi-Continuity Loss and Differencing Loss are trained independently, comparisons against the native detector baseline provide a variantlevel ablation of the proposed framework.

Table 2 shows that the benefits of continuity regularization become more pronounced as the amount of training data decreases. On the Industrial Metal dataset, Differencing improves average mAP@0.5 from +1.57 percentage points in the full-data setting to +7.59 percentage points at the 25% training ratio. Similar trends are observed for mAP@0.5:0.95. On the MEA dataset, Multi-Continuity and Differencing improve average mAP@0.5:0.95 by +5.47 and +6.91 percentage points, respectively, under the 25% training setting. These results indicate that continuity-driven regularization provides effective representation-level supervision when annotated defect samples are limited.

Table 2: Variant-level ablation under reduced training data. Values denote average improvements over detector-specific baselines across six detector architectures. All improvements are reported in percentage points. MC denotes Multi-Continuity and Diff. denotes Differencing.
<table><tr><td>Dataset</td><td>Training ratio</td><td>MCΔ mAP@0.5</td><td>Diff. ∆ mAP@0.5</td><td>MC∆ mAP@0.5:0.95</td><td>Diff. ∆ mAP@0.5:0.95</td></tr><tr><td rowspan="4">Industrial Metal</td><td>100%</td><td>+1.06</td><td>+1.57</td><td>+2.28</td><td>+3.49</td></tr><tr><td>75%</td><td>+1.17</td><td>+2.17</td><td>+3.36</td><td>+4.57</td></tr><tr><td>50%</td><td>+1.86</td><td>+4.70</td><td>+3.78</td><td>+7.14</td></tr><tr><td>25%</td><td>+3.74</td><td>+7.59</td><td>+3.49</td><td>+6.76</td></tr><tr><td rowspan="4">MEA</td><td>100%</td><td>+0.15</td><td>+0.15</td><td>+5.19</td><td>+5.38</td></tr><tr><td>75%</td><td>+0.33</td><td>+0.97</td><td>+0.35</td><td>+1.51</td></tr><tr><td>50%</td><td>+3.46</td><td>+4.21</td><td>+4.53</td><td>+2.05</td></tr><tr><td>25%</td><td>+2.33</td><td>+4.54</td><td>+5.47</td><td>+6.91</td></tr></table>

Table 3: Cross-detector generality analysis. Values denote average improvements over the native detector baselines across both datasets and all training ratios. All improvements are reported in percentage points.
<table><tr><td>Detector</td><td>MC∆ mAP@0.5</td><td>Diff. ∆ mAP@0.5</td><td>MCΔ mAP@0.5:0.95</td><td>Diff. ∆ mAP@0.5:0.95</td></tr><tr><td>YOLOv8</td><td>+2.02</td><td>+3.10</td><td>+4.50</td><td>+5.45</td></tr><tr><td>YOLOv10</td><td>+1.64</td><td>+2.72</td><td>+1.70</td><td>+4.01</td></tr><tr><td>YOL011</td><td>+2.27</td><td>+4.56</td><td>+3.81</td><td>+5.48</td></tr><tr><td>YOLOv12</td><td>+1.74</td><td>+3.75</td><td>+4.89</td><td>+5.64</td></tr><tr><td>MambaYOLO</td><td>+1.10</td><td>+1.62</td><td>+2.41</td><td>+3.35</td></tr><tr><td>DETR</td><td>+1.81</td><td>+3.68</td><td>+4.04</td><td>+4.41</td></tr><tr><td>Average</td><td>+1.76</td><td>+3.24</td><td>+3.56</td><td>+4.72</td></tr></table>

The two regularization variants show complementary behavior. Multi-Continuity provides stable improvements through predictive patch-level continuity modeling, while Differencing often yields larger gains in data-scarce settings by directly constraining local feature variation trends.

Finally, Table 3 summarizes detector-wise average improvements across all datasets and training ratios. Both regularizers consistently improve YOLO-family detectors, MambaY-OLO, and DETR, supporting the detector-agnostic nature of the proposed framework.

## 4.5 Generalization on Public Industrial Dataset

To further evaluate generalization beyond the two proprietary datasets, we conduct experiments on the public NEU-DET steel surface defect dataset [20]. NEU-DET contains 1,800 images, which we split into training, validation, and test sets using a 6:2:2 ratio. We evaluate the proposed regularizers under two complementary settings. First, we use YOLOv12 at $1 2 8 0 \times 1 2 8 0$ to evaluate the method under a high-resolution industrial inspection setting with small localized defects. Second, we conduct a standardized cross-architecture evaluation across six detectors at $6 4 0 \times 6 4 0$ to assess detector-level generalization. In both settings, improvements are measured relative to the corresponding detector baseline trained under the same resolution and training protocol. We use NEU-DET as an external validation dataset to assess relative improvements rather than to claim benchmark-specific state-of-the-art performance.

Table 4: Training-ratio-wise generalization analysis on NEU-DET using YOLOv12 at $1 2 8 0 \times 1 2 8 0$ resolution. Values denote improvements over the corresponding YOLOv12 baseline in percentage points.
<table><tr><td>Training Ratio</td><td>MCΔ mAP@0.5</td><td>Diff. ∆ mAP@0.5</td><td>MCΔ mAP@0.5:0.95</td><td>Diff. ∆ mAP@0.5:0.95</td></tr><tr><td>100%</td><td>+9.42</td><td>-0.40</td><td>+5.03</td><td>+0.16</td></tr><tr><td>75%</td><td>+7.93</td><td>+16.59</td><td>+3.51</td><td>+7.60</td></tr><tr><td>50%</td><td>+8.43</td><td>+15.16</td><td>+4.30</td><td>+7.27</td></tr><tr><td>25%</td><td>+13.97</td><td>+21.07</td><td>+5.42</td><td>+8.23</td></tr><tr><td>Average</td><td>+9.94</td><td>+13.11</td><td>+4.57</td><td>+5.82</td></tr></table>

Table 5: Variant-level analysis on NEU-DET across six detector architectures at 640 × 640 resolution. Values denote average improvements over detector-specific baselines in percentage points. MC denotes Multi-Continuity and Diff. denotes Differencing.
<table><tr><td>Dataset</td><td>Training ratio</td><td>MC∆ mAP@0.5</td><td>Diff. ∆ mAP@0.5</td><td>MCΔ mAP@0.5:0.95</td><td>Diff. ∆ mAP@0.5:0.95</td></tr><tr><td rowspan="4">NEU-DET</td><td>100%</td><td>+1.75</td><td>+1.60</td><td>+1.63</td><td>+1.73</td></tr><tr><td>75%</td><td>+1.15</td><td>+1.12</td><td>+1.68</td><td>+1.27</td></tr><tr><td>50%</td><td>+3.25</td><td>+3.37</td><td>+2.35</td><td>+2.05</td></tr><tr><td>25%</td><td>+3.75</td><td>+6.23</td><td>+1.78</td><td>+3.50</td></tr></table>

We first evaluate YOLOv12 at 1280 × 1280 while progressively reducing the amount of training data and keeping the validation and test sets fixed. Table 4 summarizes the improvements over the corresponding YOLOv12 baseline. Multi-Continuity improves performance across all training ratios, with gains of up to 13.97 percentage points in mAP@0.5 and 5.42 points in mAP@0.5:0.95. Differencing provides limited improvement in the full-data setting but becomes substantially more effective as the training data decrease, reaching gains of 21.07 points in mAP@0.5 and 8.23 points in mAP@0.5:0.95 at the 25% training ratio.

To examine whether these gains generalize beyond YOLOv12, we further evaluate both regularizers across six detector architectures using a standardized input resolution of 640 × 640. Table 5 reports the average improvement over the corresponding detector-specific baselines. Both variants improve average performance across all training ratios. The gains generally become more pronounced as the amount of training data decreases, particularly for Differencing. At the 25% training ratio, Multi-Continuity improves mAP@0.5 by 3.75 points, while Differencing achieves gains of 6.23 points in mAP@0.5 and 3.50 points in mAP@0.5:0.95.

Under the full-data setting, the detector baselines already learn relatively discriminative representations, leaving less room for additional regularization. This is particularly evident for YOLOv12 at 1280 × 1280, where Differencing slightly decreases mAP@0.5 by 0.40 points while marginally improving mAP@0.5:0.95 by 0.16 points. In contrast, substantially larger gains are observed as the amount of training data decreases. This trend is also observed in the standardized cross-architecture evaluation at 640 × 640, where both regularizers yield larger average gains under reduced-data settings. These results indicate that continuity-driven representation regularization generalizes across detector architectures and is particularly effective when annotated training data are limited.

Figure 4 further provides qualitative comparisons on NEU-DET under the 100% and 25% training-data settings, allowing direct comparison of missed detections, false positives, and localization quality. Differencing recovers scratch defects missed by the baseline under both settings. Under the 25% setting, an additional detection appears on an ambiguous defect-like structure, indicating that stronger sensitivity under limited supervision may also increase responses to visually similar structures. Together with the quantitative results, these examples illustrate both the benefit and potential limitation of continuity-driven regularization under data-scarce conditions. Detailed detector-wise quantitative results are provided in Appendix A.

![](images/04d2e0a6adcb9c4e43c741c327939e4f3527530de4314780d9db5f3d0e5157f9.jpg)  
Figure 4: Qualitative comparison on NEU-DET under 100% and 25% training-data settings. Differencing recovers scratch defects missed by the baseline, while the 25% setting also shows a detection on an ambiguous defect-like structure.

## 4.6 Qualitative Analysis

To further examine how continuity-driven regularization affects learned representations, we visualize intermediate feature maps from YOLOv12 trained with and without the proposed auxiliary objective. Representative feature responses at different depths are provided in Appendix B, Figure B1. The baseline model is trained using only the standard detection loss, whereas the regularized model incorporates the proposed continuity-based regularization.

The qualitative comparison suggests that the proposed regularization does not simply smooth feature representations. Instead, it preserves normal structural patterns while producing more localized and discriminative responses around defect regions. Compared with the baseline, defect-related activations become more distinguishable from the surrounding normal background, indicating improved representational separability between normal and defective regions.

Additional continuity discrepancy visualizations are provided in Appendix B, Figure B2. Compared with the baseline, the regularized model exhibits reduced discrepancy fluctuations in homogeneous normal regions and stronger responses around localized structural changes, supporting the effectiveness of continuity-driven representation learning.

This analysis also reveals an important limitation. Strong object boundaries, abrupt geometric transitions, and lighting-induced edges can produce large continuity discrepancies even when they do not correspond to defects. Therefore, continuity discrepancy alone is not sufficient for defect localization. The proposed regularizers should be interpreted as auxiliary representation priors rather than standalone anomaly scores. This observation suggests that the proposed framework is particularly suitable for inspection settings where normal regions exhibit repetitive or homogeneous spatial structures, while additional boundary-aware weighting may be beneficial for objects with complex geometric structures.

## 5 Conclusion

We proposed a continuity-driven representation regularization framework for object-detectorbased industrial defect detection. Motivated by the observation that industrial defects often appear as localized violations of repetitive normal structures, the proposed framework uses normal-dominant regions as dense auxiliary supervision during detector training. We introduced two detector-agnostic regularizers: Multi-Continuity Loss, which encourages predictable feature transitions through 1D sequence prediction and 2D masked spatial prediction, and Differencing Loss, which constrains first-order feature variation and second-order curvature between neighboring patch embeddings. Both losses are applied with box-derived region weighting to stabilize normal-region representations while avoiding indiscriminate smoothing over annotated defect regions.

Experiments on two real-world industrial inspection datasets, the public NEU-DET benchmark, and six detector architectures show that the proposed regularizers consistently improve supervised defect localization, especially under limited training data. The improvements in mAP@0.5:0.95 indicate that continuity-driven representation learning can enhance localization quality, not merely coarse defect detection. The observed gains on both proprietary industrial datasets and the public NEU-DET dataset further suggest that the proposed regularization framework generalizes across different industrial inspection domains. These results demonstrate that structural continuity serves as an effective representation-level prior for industrial inspection images.

A limitation of the current framework is that normal object boundaries or abrupt geometric transitions may also induce large continuity discrepancies. Future work will investigate boundary-aware weighting, broader public benchmark evaluation, and extensions to anomaly detection and multimodal industrial inspection.

## References

[1] Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. MVTec AD: A comprehensive real-world dataset for unsupervised anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9592–9600, 2019.

[2] Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. Uninformed students: Student-teacher anomaly detection with discriminative latent embeddings. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4183–4192, 2020.

[3] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European Conference on Computer Vision, pages 213–229. Springer, 2020.

[4] Raghavendra Chalapathy and Sanjay Chawla. Deep learning for anomaly detection: A survey. arXiv preprint arXiv:1901.03407, 2019.

[5] Yuqi Cheng, Yunkang Cao, Haiming Yao, Wei Luo, Cheng Jiang, Hui Zhang, and Weiming Shen. A comprehensive survey for real-world industrial surface defect detection: Challenges, approaches, and prospects. Journal of Manufacturing Systems, 84: 152–172, 2026. doi: 10.1016/j.jmsy.2025.11.022.

[6] Thomas Defard, Aleksandr Setkov, Angelique Loesch, and Romaric Audigier. PaDiM: A patch distribution modeling framework for anomaly detection and localization. In International Conference on Pattern Recognition, pages 475–489. Springer, 2020.

[7] Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John Winn, and Andrew Zisserman. The PASCAL visual object classes challenge. International Journal of Computer Vision, 88(2):303–338, 2010.

[8] Jeeho Hyun, Sangyun Kim, Giyoung Jeon, Seung Hwan Kim, Kyunghoon Bae, and Byung Jun Kang. ReConPatch: Contrastive patch representation learning for industrial anomaly detection. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 2052–2061, 2024.

[9] Daehwan Kim, Hyungmin Kim, Daun Jeong, Sungho Suh, and Hansang Cho. SPACE: SPAtial-aware Consistency rEgularization for anomaly detection in industrial applications. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision, pages 7184–7194. IEEE, 2025.

[10] Ramona Kühlechner. Object detection survey for industrial applications with focus on quality control. Production Engineering, 19(6):1271–1291, 2025.

[11] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International Journal ofComputer Vision, 128(7):1956– 1981, 2020.

[12] Zhuo Li, Yuhao Yan, Xiangheng Wang, Yifei Ge, and Lin Meng. A survey of deep learning for industrial visual anomaly detection. Artificial Intelligence Review, 58(9): 279, 2025.

[13] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: Common objects in context. In European Conference on Computer Vision, pages 740–755. Springer, 2014.

[14] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings of the IEEE International Conference on Computer Vision, pages 2980–2988, 2017.

[15] Lutfun Nahar, Mohammad Awrangjeb, and Md Saiful Islam. Ai-enabled defect detection in industrial products: A comprehensive survey, key insights and future research challenges. Advanced Engineering Informatics, 69:104067, 2026.

[16] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 779–788, 2016.

[17] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R-CNN: Towards realtime object detection with region proposal networks. Advances in Neural Information Processing Systems, 28, 2015.

[18] Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, and Peter Gehler. Towards total recall in industrial anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14318– 14328, 2022.

[19] Connor Shorten and Taghi M. Khoshgoftaar. A survey on image data augmentation for deep learning. Journal ofBig Data, 6(1):1–48, 2019.

[20] Kechen Song and Yunhui Yan. A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects. Applied Surface Science, 285:858– 864, 2013. doi: 10.1016/j.apsusc.2013.09.002.

[21] Carsten Steger, Markus Ulrich, and Christian Wiedemann. Machine vision algorithms and applications. John Wiley & Sons, 2018.

[22] Yunjie Tian, Qixiang Ye, and David Doermann. YOLOv12: Attention-centric real-time object detectors, 2025.

[23] Ultralytics. Ultralytics YOLOv8. https://docs.ultralytics.com/ models/yolov8/, 2023.

[24] Ultralytics. Ultralytics YOLO11. https://docs.ultralytics.com/ models/yolo11/, 2024.

[25] Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, and Guiguang Ding. YOLOv10: Real-time end-to-end object detection, 2024.

[26] Guodong Wang, Shumin Han, Errui Ding, and Di Huang. Student-teacher feature pyramid matching for anomaly detection. In British Machine Vision Conference, 2021.

[27] Zeyu Wang, Chen Li, Huiying Xu, Xinzhong Zhu, and Hongbo Li. Mamba YOLO: A simple baseline for object detection with state space model, 2024.

[28] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. CutMix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6023–6032, 2019.

[29] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In International Conference on Learning Representations, 2018.

[30] Xinyi Zhang, Naiqi Li, Jiawei Li, Tao Dai, Yong Jiang, and Shu-Tao Xia. Unsupervised surface anomaly detection with diffusion probabilistic model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6782–6791, 2023.

[31] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable DETR: Deformable transformers for end-to-end object detection. In International Conference on Learning Representations, 2021.

Table A1: Robustness to different bounding-box scales on Industrial Metal using YOLOv12 with Differencing Loss.
<table><tr><td>Scale</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td></tr><tr><td>0.5×</td><td>0.916</td><td>0.920</td><td>0.975</td><td>0.707</td></tr><tr><td>0.7×</td><td>0.791</td><td>0.894</td><td>0.933</td><td>0.637</td></tr><tr><td>0.9×</td><td>0.905</td><td>0.825</td><td>0.936</td><td>0.618</td></tr><tr><td>1.0×</td><td>0.893</td><td>0.946</td><td>0.965</td><td>0.701</td></tr><tr><td>1.1×</td><td>0.925</td><td>0.886</td><td>0.945</td><td>0.651</td></tr><tr><td>1.3×</td><td>0.934</td><td>0.875</td><td>0.971</td><td>0.723</td></tr><tr><td>1.5×</td><td>0.947</td><td>0.958</td><td>0.986</td><td>0.744</td></tr></table>

## A Additional Quantitative Results

## A.1 Detector-wise Evaluation Results

This appendix reports the detector-wise test results corresponding to the full-data and reduceddata evaluations. The validation and test sets are fixed across all training ratios. For each detector and training ratio, deltas are computed relative to the corresponding detector-specific baseline.

Table A2: Full-data detector-wise results on Industrial Metal.
<table><tr><td>Detector</td><td>Method</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>∆mAP@0.5</td><td>∆mAP@0.5:0.95</td></tr><tr><td rowspan="3">YOLOv8</td><td>Baseline</td><td>0.937</td><td>0.886</td><td>0.944</td><td>0.660</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.974</td><td>0.937</td><td>0.974</td><td>0.702</td><td>+2.94</td><td>+4.15</td></tr><tr><td>Differencing</td><td>0.962</td><td>0.896</td><td>0.966</td><td>0.700</td><td>+2.18</td><td>+3.96</td></tr><tr><td rowspan="3">YOLOv10</td><td>Baseline</td><td>0.875</td><td>0.861</td><td>0.922</td><td>0.661</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.765</td><td>0.920</td><td>0.927</td><td>0.663</td><td>+0.51</td><td>+0.15</td></tr><tr><td>Differencing</td><td>0.968</td><td>0.759</td><td>0.913</td><td>0.688</td><td>-0.89</td><td>+2.64</td></tr><tr><td rowspan="3">YOL011</td><td>Baseline</td><td>0.985</td><td>0.848</td><td>0.945</td><td>0.659</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.891</td><td>0.890</td><td>0.937</td><td>0.669</td><td>-0.76</td><td>+0.97</td></tr><tr><td>Differencing</td><td>0.923</td><td>0.943</td><td>0.951</td><td>0.698</td><td>+0.65</td><td>+3.86</td></tr><tr><td rowspan="3">YOLOv12</td><td>Baseline</td><td>0.911</td><td>0.890</td><td>0.950</td><td>0.695</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.892</td><td>0.899</td><td>0.960</td><td>0.714</td><td>+1.04</td><td>+1.93</td></tr><tr><td>Differencing</td><td>0.893</td><td>0.946</td><td>0.965</td><td>0.701</td><td>+1.47</td><td>+0.63</td></tr><tr><td rowspan="3">MambaYOLO</td><td>Baseline</td><td>0.937</td><td>0.890</td><td>0.953</td><td>0.659</td><td>一</td><td>一</td></tr><tr><td>Multi-Continuity</td><td>0.945</td><td>0.860</td><td>0.956</td><td>0.683</td><td>+0.32</td><td>+2.37</td></tr><tr><td>Differencing</td><td>0.939</td><td>0.931</td><td>0.966</td><td>0.721</td><td>+1.25</td><td>+6.19</td></tr><tr><td rowspan="3">DETR</td><td>Baseline</td><td>0.913</td><td>0.875</td><td>0.899</td><td>0.604</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.957</td><td>0.917</td><td>0.922</td><td>0.645</td><td>+2.29</td><td>+4.10</td></tr><tr><td>Differencing</td><td>0.944</td><td>0.944</td><td>0.947</td><td>0.640</td><td>+4.77</td><td>+3.65</td></tr></table>

Table A3: Full-data detector-wise results on MEA.
<table><tr><td>Detector</td><td>Method</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>∆mAP@0.5</td><td>∆mAP@0.5:0.95</td></tr><tr><td rowspan="3">YOLOv8</td><td>Baseline</td><td>0.989</td><td>1.000</td><td>0.995</td><td>0.899</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.998</td><td>0.998</td><td>0.995</td><td>0.944</td><td>+0.06</td><td>+4.43</td></tr><tr><td>Differencing</td><td>0.989</td><td>1.000</td><td>0.994</td><td>0.956</td><td>-0.05</td><td>+5.70</td></tr><tr><td rowspan="3">YOLOv10</td><td>Baseline</td><td>0.989</td><td>0.998</td><td>0.995</td><td>0.901</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.997</td><td>1.000</td><td>0.995</td><td>0.960</td><td>+0.00</td><td>+5.90</td></tr><tr><td>Differencing</td><td>0.999</td><td>0.999</td><td>0.995</td><td>0.959</td><td>+0.00</td><td>+5.77</td></tr><tr><td rowspan="3">YOL011</td><td>Baseline</td><td>0.982</td><td>0.998</td><td>0.994</td><td>0.898</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.999</td><td>0.998</td><td>0.994</td><td>0.952</td><td>+0.05</td><td>+5.39</td></tr><tr><td>Differencing</td><td>0.999</td><td>0.999</td><td>0.995</td><td>0.961</td><td>+0.12</td><td>+6.34</td></tr><tr><td rowspan="3">YOLOv12</td><td>Baseline</td><td>0.965</td><td>1.000</td><td>0.994</td><td>0.890</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.999</td><td>0.998</td><td>0.995</td><td>0.964</td><td>+0.07</td><td>+7.39</td></tr><tr><td>Differencing</td><td>1.000</td><td>1.000</td><td>0.995</td><td>0.943</td><td>+0.07</td><td>+5.34</td></tr><tr><td rowspan="3">MambaYOLO</td><td>Baseline</td><td>0.966</td><td>1.000</td><td>0.990</td><td>0.877</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.998</td><td>0.997</td><td>0.994</td><td>0.902</td><td>+0.44</td><td>+2.53</td></tr><tr><td>Differencing</td><td>0.999</td><td>0.997</td><td>0.994</td><td>0.919</td><td>+0.41</td><td>+4.21</td></tr><tr><td rowspan="3">DETR</td><td>Baseline</td><td>0.976</td><td>0.997</td><td>0.990</td><td>0.876</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.998</td><td>0.997</td><td>0.993</td><td>0.931</td><td>+0.29</td><td>+5.52</td></tr><tr><td>Differencing</td><td>0.992</td><td>0.998</td><td>0.994</td><td>0.925</td><td>+0.37</td><td>+4.90</td></tr></table>

Table A4: Detector-wise results on the Industrial Metal dataset at the 75% training ratio.
<table><tr><td>Method</td><td>Detector</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>∆mAP@0.5</td><td>∆mAP@0.5:0.95</td></tr><tr><td rowspan="6">Baseline</td><td>YOLOv8</td><td>0.873</td><td>0.930</td><td>0.959</td><td>0.666</td><td></td><td></td></tr><tr><td>YOLOv10</td><td>0.859</td><td>0.951</td><td>0.965</td><td>0.645</td><td>1</td><td></td></tr><tr><td>YOLO11</td><td>0.939</td><td>0.860</td><td>0.934</td><td>0.623</td><td>一</td><td></td></tr><tr><td>YOLOv12</td><td>0.900</td><td>0.861</td><td>0.931</td><td>0.612</td><td>一</td><td></td></tr><tr><td>MambaYOLO</td><td>0.864</td><td>0.861</td><td>0.912</td><td>0.621</td><td>一</td><td></td></tr><tr><td>DETR</td><td>0.924</td><td>0.847</td><td>0.872</td><td>0.564</td><td>一</td><td></td></tr><tr><td rowspan="6">Multi-Continuity</td><td>YOLOv8</td><td>0.929</td><td>0.929</td><td>0.971</td><td>0.718</td><td>+1.19</td><td>+5.23</td></tr><tr><td>YOLOv10</td><td>0.887</td><td>0.919</td><td>0.951</td><td>0.642</td><td>-1.45</td><td>-0.26</td></tr><tr><td>YOL011</td><td>0.927</td><td>0.896</td><td>0.932</td><td>0.675</td><td>-0.16</td><td>+5.26</td></tr><tr><td>YOLOv12</td><td>0.912</td><td>0.901</td><td>0.936</td><td>0.657</td><td>+0.52</td><td>+4.48</td></tr><tr><td>MambaYOLO</td><td>0.963</td><td>0.838</td><td>0.946</td><td>0.638</td><td>+3.40</td><td>+1.64</td></tr><tr><td>DETR</td><td>0.969</td><td>0.875</td><td>0.908</td><td>0.603</td><td>+3.53</td><td>+3.84</td></tr><tr><td rowspan="6">Differencing</td><td>YOLOv8</td><td>0.905</td><td>0.915</td><td>0.953</td><td>0.698</td><td>-0.66</td><td>+3.21</td></tr><tr><td>YOLOv10</td><td>0.874</td><td>0.913</td><td>0.942</td><td>0.691</td><td>-2.40</td><td>+4.67</td></tr><tr><td>YOL011</td><td>0.942</td><td>0.937</td><td>0.977</td><td>0.681</td><td>+4.28</td><td>+5.84</td></tr><tr><td>YOLOv12</td><td>0.890</td><td>0.931</td><td>0.950</td><td>0.675</td><td>+1.91</td><td>+6.28</td></tr><tr><td>MambaYOLO</td><td>0.880</td><td>0.879</td><td>0.929</td><td>0.627</td><td>+1.73</td><td>+0.59</td></tr><tr><td>DETR</td><td>0.919</td><td>0.944</td><td>0.954</td><td>0.632</td><td>+8.17</td><td>+6.82</td></tr></table>

Table A5: Detector-wise results on the Industrial Metal dataset at the 50% training ratio.
<table><tr><td>Method</td><td>Detector</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>∆mAP@0.5</td><td>∆mAP@0.5:0.95</td></tr><tr><td rowspan="6">Baseline</td><td>YOLOv8</td><td>0.792</td><td>0.872</td><td>0.881</td><td>0.527</td><td>一</td><td>一</td></tr><tr><td>YOLOv10</td><td>0.812</td><td>0.730</td><td>0.858</td><td>0.562</td><td>一</td><td></td></tr><tr><td>YOL011</td><td>0.787</td><td>0.883</td><td>0.904</td><td>0.603</td><td>一</td><td></td></tr><tr><td>YOLOv12</td><td>0.854</td><td>0.785</td><td>0.868</td><td>0.519</td><td>一</td><td></td></tr><tr><td>MambaYOLO</td><td>0.869</td><td>0.880</td><td>0.908</td><td>0.575</td><td>一</td><td></td></tr><tr><td>DETR</td><td>0.865</td><td>0.889</td><td>0.876</td><td>0.538</td><td>一</td><td></td></tr><tr><td rowspan="6">Multi-Continuity</td><td>YOLOv8</td><td>0.874</td><td>0.914</td><td>0.933</td><td>0.635</td><td>+5.18</td><td>+10.75</td></tr><tr><td>YOLOv10</td><td>0.831</td><td>0.792</td><td>0.891</td><td>0.560</td><td>+3.23</td><td>-0.18</td></tr><tr><td>YOL011</td><td>0.825</td><td>0.838</td><td>0.891</td><td>0.567</td><td>-1.34</td><td>-3.62</td></tr><tr><td>YOLOv12</td><td>0.828</td><td>0.831</td><td>0.888</td><td>0.594</td><td>+2.02</td><td>+7.44</td></tr><tr><td>MambaYOLO</td><td>0.888</td><td>0.835</td><td>0.899</td><td>0.587</td><td>-0.89</td><td>+1.22</td></tr><tr><td>DETR</td><td>0.857</td><td>0.917</td><td>0.906</td><td>0.609</td><td>+2.95</td><td>+7.09</td></tr><tr><td rowspan="6">Differencing</td><td>YOLOv8</td><td>0.960</td><td>0.948</td><td>0.975</td><td>0.690</td><td>+9.35</td><td>+16.32</td></tr><tr><td>YOLOv10</td><td>0.904</td><td>0.869</td><td>0.954</td><td>0.658</td><td>+9.60</td><td>+9.56</td></tr><tr><td>YOLO11</td><td>0.876</td><td>0.905</td><td>0.939</td><td>0.593</td><td>+3.47</td><td>-1.02</td></tr><tr><td>YOLOv12</td><td>0.877</td><td>0.850</td><td>0.904</td><td>0.642</td><td>+3.62</td><td>+12.24</td></tr><tr><td>MambaYOLO</td><td>0.907</td><td>0.809</td><td>0.900</td><td>0.595</td><td>-0.75</td><td>+2.08</td></tr><tr><td>DETR</td><td>0.940</td><td>0.875</td><td>0.905</td><td>0.574</td><td>+2.93</td><td>+3.63</td></tr></table>

Table A6: Detector-wise results on the Industrial Metal dataset at the 25% training ratio.
<table><tr><td>Method</td><td>Detector</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>∆mAP@0.5</td><td>∆mAP@0.5:0.95</td></tr><tr><td rowspan="6">Baseline</td><td>YOLOv8</td><td>0.859</td><td>0.868</td><td>0.893</td><td>0.553</td><td>一</td><td></td></tr><tr><td>YOLOv10</td><td>0.825</td><td>0.819</td><td>0.847</td><td>0.561</td><td>一</td><td></td></tr><tr><td>YOL011</td><td>0.700</td><td>0.643</td><td>0.732</td><td>0.417</td><td>一</td><td></td></tr><tr><td>YOLOv12</td><td>0.576</td><td>0.646</td><td>0.646</td><td>0.326</td><td>一</td><td></td></tr><tr><td>MambaYOLO</td><td>0.812</td><td>0.716</td><td>0.797</td><td>0.405</td><td>一</td><td></td></tr><tr><td>DETR</td><td>0.937</td><td>一</td><td>0.838</td><td>0.529</td><td>一</td><td></td></tr><tr><td rowspan="6">Multi-Continuity</td><td>YOLOv8</td><td>0.818</td><td>0.915</td><td>0.910</td><td>0.575</td><td>+1.73</td><td>+2.20</td></tr><tr><td>YOLOv10</td><td>0.858</td><td>0.787</td><td>0.878</td><td>0.566</td><td>+3.13</td><td>+0.46</td></tr><tr><td>YOL011</td><td>0.803</td><td>0.858</td><td>0.850</td><td>0.504</td><td>+11.79</td><td>+8.63</td></tr><tr><td>YOLOv12</td><td>0.679</td><td>0.652</td><td>0.707</td><td>0.391</td><td>+6.13</td><td>+6.48</td></tr><tr><td>MambaYOLO</td><td>0.744</td><td>0.766</td><td>0.784</td><td>0.421</td><td>-1.31</td><td>+1.54</td></tr><tr><td>DETR</td><td>0.853</td><td>0.889</td><td>0.848</td><td>0.545</td><td>+0.98</td><td>+1.60</td></tr><tr><td rowspan="6">Differencing</td><td>YOLOv8</td><td>0.866</td><td>0.897</td><td>0.928</td><td>0.593</td><td>+3.46</td><td>+3.96</td></tr><tr><td>YOLOv10</td><td>0.838</td><td>0.815</td><td>0.878</td><td>0.546</td><td>+3.16</td><td>-1.54</td></tr><tr><td>YOLO11</td><td>0.946</td><td>0.840</td><td>0.905</td><td>0.588</td><td>+17.24</td><td>+17.07</td></tr><tr><td>YOLOv12</td><td>0.698</td><td>0.847</td><td>0.790</td><td>0.461</td><td>+14.38</td><td>+13.55</td></tr><tr><td>MambaYOLO</td><td>0.834</td><td>0.739</td><td>0.808</td><td>0.441</td><td>+1.13</td><td>+3.60</td></tr><tr><td>DETR</td><td>0.984</td><td>0.847</td><td>0.900</td><td>0.568</td><td>+6.18</td><td>+3.92</td></tr></table>

Table A7: Detector-wise results on the MEA dataset at the 75% training ratio.
<table><tr><td>Method</td><td>Detector</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>∆mAP@0.5</td><td>∆mAP@0.5:0.95</td></tr><tr><td rowspan="6">Baseline</td><td>YOLOv8</td><td>0.982</td><td>0.987</td><td>0.990</td><td>0.845</td><td>一</td><td>一</td></tr><tr><td>YOLOv10</td><td>0.981</td><td>0.986</td><td>0.989</td><td>0.842</td><td>一</td><td></td></tr><tr><td>YOL011</td><td>0.983</td><td>0.988</td><td>0.990</td><td>0.849</td><td>一</td><td></td></tr><tr><td>YOLOv12</td><td>0.981</td><td>0.998</td><td>0.993</td><td>0.860</td><td>一</td><td></td></tr><tr><td>MambaYOLO</td><td>0.945</td><td>0.952</td><td>0.951</td><td>0.810</td><td>一</td><td></td></tr><tr><td>DETR</td><td>0.962</td><td>0.968</td><td>0.969</td><td>0.825</td><td>一</td><td></td></tr><tr><td rowspan="6">Multi-Continuity</td><td>YOLOv8</td><td>0.986</td><td>0.991</td><td>0.993</td><td>0.853</td><td>+0.31</td><td>+0.80</td></tr><tr><td>YOLOv10</td><td>0.985</td><td>0.990</td><td>0.992</td><td>0.851</td><td>+0.32</td><td>+0.88</td></tr><tr><td>YOL011</td><td>0.988</td><td>0.992</td><td>0.993</td><td>0.826</td><td>+0.30</td><td>-2.25</td></tr><tr><td>YOLOv12</td><td>0.984</td><td>0.999</td><td>0.995</td><td>0.866</td><td>+0.11</td><td>+0.56</td></tr><tr><td>MambaYOLO</td><td>0.952</td><td>0.958</td><td>0.957</td><td>0.821</td><td>+0.52</td><td>+1.13</td></tr><tr><td>DETR</td><td>0.969</td><td>0.872</td><td>0.974</td><td>0.835</td><td>+0.41</td><td>+1.00</td></tr><tr><td rowspan="6">Differencing</td><td>YOLOv8</td><td>0.990</td><td>0.993</td><td>0.995</td><td>0.860</td><td>+0.51</td><td>+1.50</td></tr><tr><td>YOLOv10</td><td>0.988</td><td>0.992</td><td>0.995</td><td>0.859</td><td>+0.59</td><td>+1.66</td></tr><tr><td>YOL011</td><td>0.991</td><td>0.994</td><td>0.996</td><td>0.865</td><td>+0.59</td><td>+1.61</td></tr><tr><td>YOLOv12</td><td>0.981</td><td>1.000</td><td>0.994</td><td>0.864</td><td>+0.04</td><td>+0.35</td></tr><tr><td>MambaYOLO</td><td>0.957</td><td>0.963</td><td>0.973</td><td>0.832</td><td>+2.21</td><td>+2.22</td></tr><tr><td>DETR</td><td>0.972</td><td>0.975</td><td>0.988</td><td>0.842</td><td>+1.88</td><td>+1.71</td></tr></table>

Table A8: Detector-wise results on the MEA dataset at the 50% training ratio.
<table><tr><td>Method</td><td>Detector</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>∆mAP@0.5</td><td>∆mAP@0.5:0.95</td></tr><tr><td rowspan="6">Baseline</td><td>YOLOv8</td><td>0.974</td><td>0.979</td><td>0.947</td><td>0.845</td><td>一</td><td>一</td></tr><tr><td>YOLOv10</td><td>0.973</td><td>0.978</td><td>0.955</td><td>0.843</td><td>一</td><td></td></tr><tr><td>YOL011</td><td>0.976</td><td>0.980</td><td>0.957</td><td>0.848</td><td>一</td><td></td></tr><tr><td>YOLOv12</td><td>0.979</td><td>0.982</td><td>0.920</td><td>0.868</td><td>一</td><td></td></tr><tr><td>MambaYOLO</td><td>0.941</td><td>0.947</td><td>0.894</td><td>0.812</td><td>一</td><td></td></tr><tr><td>DETR</td><td>0.959</td><td>0.962</td><td>0.911</td><td>0.826</td><td>一</td><td></td></tr><tr><td rowspan="6">Multi-Continuity</td><td>YOLOv8</td><td>0.978</td><td>0.983</td><td>0.969</td><td>0.885</td><td>+2.25</td><td>+4.00</td></tr><tr><td>YOLOv10</td><td>0.982</td><td>0.982</td><td>0.989</td><td>0.871</td><td>+3.38</td><td>+2.83</td></tr><tr><td>YOL011</td><td>0.984</td><td>0.984</td><td>0.977</td><td>0.900</td><td>+2.01</td><td>+5.19</td></tr><tr><td>YOLOv12</td><td>0.982</td><td>0.984</td><td>0.963</td><td>0.905</td><td>+4.34</td><td>+3.74</td></tr><tr><td>MambaYOLO</td><td>0.966</td><td>0.952</td><td>0.953</td><td>0.868</td><td>+5.81</td><td>+5.59</td></tr><tr><td>DETR</td><td>0.962</td><td>0.996</td><td>0.941</td><td>0.884</td><td>+2.99</td><td>+5.85</td></tr><tr><td rowspan="6">Differencing</td><td>YOLOv8</td><td>0.978</td><td>0.983</td><td>0.989</td><td>0.862</td><td>+4.25</td><td>+1.67</td></tr><tr><td>YOLOv10</td><td>0.978</td><td>0.982</td><td>0.989</td><td>0.859</td><td>+3.38</td><td>+1.62</td></tr><tr><td>YOL011</td><td>0.980</td><td>0.985</td><td>0.990</td><td>0.899</td><td>+3.29</td><td>+5.14</td></tr><tr><td>YOLOv12</td><td>0.980</td><td>0.982</td><td>0.991</td><td>0.860</td><td>+7.11</td><td>-0.75</td></tr><tr><td>MambaYOLO</td><td>0.947</td><td>0.952</td><td>0.945</td><td>0.831</td><td>+5.04</td><td>+1.88</td></tr><tr><td>DETR</td><td>0.966</td><td>0.966</td><td>0.933</td><td>0.853</td><td>+2.17</td><td>+2.70</td></tr></table>

Table A9: Detector-wise results on the MEA dataset at the 25% training ratio.
<table><tr><td>Method</td><td>Detector</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>∆mAP@0.5</td><td>∆mAP@0.5:0.95</td></tr><tr><td rowspan="6">Baseline</td><td>YOLOv8</td><td>0.893</td><td>0.890</td><td>0.915</td><td>0.701</td><td>一</td><td></td></tr><tr><td>YOLOv10</td><td>0.891</td><td>0.889</td><td>0.883</td><td>0.701</td><td>一</td><td></td></tr><tr><td>YOL011</td><td>0.919</td><td>0.901</td><td>0.869</td><td>0.690</td><td>一</td><td>一</td></tr><tr><td>YOLOv12</td><td>0.922</td><td>0.915</td><td>0.922</td><td>0.737</td><td>一</td><td>一</td></tr><tr><td>MambaYOLO</td><td>0.885</td><td>0.899</td><td>0.904</td><td>0.699</td><td>一</td><td>一</td></tr><tr><td>DETR</td><td>0.900</td><td>0.891</td><td>0.900</td><td>0.700</td><td>一</td><td>一</td></tr><tr><td rowspan="6">Multi-Continuity</td><td>YOLOv8</td><td>0.929</td><td>0.935</td><td>0.940</td><td>0.746</td><td>+2.50</td><td>+4.45</td></tr><tr><td>YOLOv10</td><td>0.927</td><td>0.933</td><td>0.923</td><td>0.739</td><td>+3.99</td><td>+3.79</td></tr><tr><td>YOLO11</td><td>0.909</td><td>0.938</td><td>0.932</td><td>0.799</td><td>+6.27</td><td>+10.91</td></tr><tr><td>YOLOv12</td><td>0.942</td><td>0.948</td><td>0.918</td><td>0.808</td><td>-0.35</td><td>+7.09</td></tr><tr><td>MambaYOLO</td><td>0.905</td><td>0.925</td><td>0.910</td><td>0.731</td><td>+0.55</td><td>+3.24</td></tr><tr><td>DETR</td><td>0.918</td><td>0.939</td><td>0.910</td><td>0.733</td><td>+1.04</td><td>+3.31</td></tr><tr><td rowspan="6">Differencing</td><td>YOLOv8</td><td>0.939</td><td>0.951</td><td>0.973</td><td>0.774</td><td>+5.80</td><td>+7.28</td></tr><tr><td>YOLOv10</td><td>0.943</td><td>0.949</td><td>0.966</td><td>0.778</td><td>+8.31</td><td>+7.70</td></tr><tr><td>YOL011</td><td>0.958</td><td>0.962</td><td>0.937</td><td>0.740</td><td>+6.80</td><td>+4.98</td></tr><tr><td>YOLOv12</td><td>0.965</td><td>0.959</td><td>0.936</td><td>0.812</td><td>+1.42</td><td>+7.49</td></tr><tr><td>MambaYOLO</td><td>0.926</td><td>0.929</td><td>0.924</td><td>0.759</td><td>+1.93</td><td>+6.06</td></tr><tr><td>DETR</td><td>0.935</td><td>0.931</td><td>0.930</td><td>0.779</td><td>+3.00</td><td>+7.92</td></tr></table>

Table A10: Training-ratio-wise results on the NEU-DET dataset using YOLOv12 across 100%, 75%, 50%, and 25% training ratios.
<table><tr><td>Method</td><td>Training Ratio</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>∆mAP@0.5</td><td>∆mAP@0.5:0.95</td></tr><tr><td rowspan="4">Baseline</td><td>100%</td><td>0.658</td><td>0.292</td><td>0.364</td><td>0.140</td><td>一</td><td>一</td></tr><tr><td>75%</td><td>0.444</td><td>0.416</td><td>0.347</td><td>0.138</td><td></td><td>一</td></tr><tr><td>50%</td><td>0.453</td><td>0.383</td><td>0.318</td><td>0.112</td><td>一</td><td>1</td></tr><tr><td>25%</td><td>0.492</td><td>0.299</td><td>0.210</td><td>0.067</td><td>一</td><td>一</td></tr><tr><td rowspan="4">Multi-Continuity</td><td>100%</td><td>0.581</td><td>0.518</td><td>0.459</td><td>0.190</td><td>+9.42</td><td>+5.03</td></tr><tr><td>75%</td><td>0.527</td><td>0.524</td><td>0.426</td><td>0.173</td><td>+7.93</td><td>+3.51</td></tr><tr><td>50%</td><td>0.510</td><td>0.446</td><td>0.402</td><td>0.155</td><td>+8.43</td><td>+4.30</td></tr><tr><td>25%</td><td>0.331</td><td>0.498</td><td>0.350</td><td>0.121</td><td>+13.97</td><td>+5.42</td></tr><tr><td rowspan="4">Differencing</td><td>100%</td><td>0.436</td><td>0.508</td><td>0.360</td><td>0.141</td><td>-0.40</td><td>+0.16</td></tr><tr><td>75%</td><td>0.590</td><td>0.573</td><td>0.512</td><td>0.214</td><td>+16.59</td><td>+7.60</td></tr><tr><td>50%</td><td>0.567</td><td>0.517</td><td>0.470</td><td>0.185</td><td>+15.16</td><td>+7.27</td></tr><tr><td>25%</td><td>0.522</td><td>0.481</td><td>0.421</td><td>0.149</td><td>+21.07</td><td>+8.23</td></tr></table>

Table A11: Comparison with conventional smoothness regularization on the Metal and Secondary Battery datasets using YOLOv12 across different training ratios.
<table><tr><td>Dataset</td><td>Method</td><td>Training Ratio</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td></tr><tr><td rowspan="18">Metal</td><td rowspan="4">TV Loss</td><td>100%</td><td>0.9519</td><td>0.7001</td></tr><tr><td>75%</td><td>0.9259</td><td>0.6354</td></tr><tr><td>50%</td><td>0.8229</td><td>0.5516</td></tr><tr><td>25%</td><td>0.7535</td><td>0.4288</td></tr><tr><td rowspan="4">Laplacian Loss</td><td>100%</td><td>0.9528</td><td>0.7074</td></tr><tr><td>75%</td><td>0.9204</td><td>0.6200</td></tr><tr><td>50%</td><td>0.8480</td><td>0.5657</td></tr><tr><td>25%</td><td>0.7701</td><td>0.4492</td></tr><tr><td rowspan="4">Differencing Loss</td><td>100%</td><td>0.9646</td><td>0.7011</td></tr><tr><td>75%</td><td>0.9497</td><td>0.6752</td></tr><tr><td>50%</td><td>0.9037</td><td>0.6418</td></tr><tr><td>25%</td><td>0.7896</td><td>0.4612</td></tr><tr><td rowspan="6">MEA Laplacian Loss</td><td rowspan="3">TV Loss</td><td>100%</td><td>0.9938</td><td>0.9187</td></tr><tr><td>75%</td><td>0.9910</td><td>0.8508</td></tr><tr><td>50%</td><td>0.9562</td><td>0.7358</td></tr><tr><td rowspan="4"></td><td>25%</td><td>0.7800</td><td>0.5560</td></tr><tr><td>100%</td><td>0.9903</td><td>0.9151</td></tr><tr><td>75%</td><td>0.9903</td><td>0.8599</td></tr><tr><td>50% 25%</td><td>0.9809 0.9207</td><td>0.8384 0.7778</td></tr><tr><td rowspan="4"></td><td></td><td></td><td></td></tr><tr><td rowspan="3">Differencing Loss</td><td>100%</td><td>0.9950 0.9938</td><td>0.9433 0.8636</td></tr><tr><td>75% 50%</td><td>0.9912</td><td>0.8600</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>25%</td><td>0.9358</td><td>0.8123</td></tr></table>

Table A12: Full-data detector-wise results on NEU-DET.
<table><tr><td>Detector</td><td>Method</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>∆mAP@0.5</td><td>∆mAP@0.5:0.95</td></tr><tr><td rowspan="3">YOLOv8</td><td>Baseline</td><td>0.686</td><td>0.683</td><td>0.728</td><td>0.408</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.680</td><td>0.674</td><td>0.746</td><td>0.414</td><td>+1.80</td><td>+0.60</td></tr><tr><td>Differencing</td><td>0.679</td><td>0.677</td><td>0.742</td><td>0.430</td><td>+1.40</td><td>+2.20</td></tr><tr><td rowspan="3">YOLOv10</td><td>Baseline</td><td>0.674</td><td>0.657</td><td>0.702</td><td>0.404</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.684</td><td>0.640</td><td>0.694</td><td>0.411</td><td>-0.80</td><td>+0.70</td></tr><tr><td>Differencing</td><td>0.661</td><td>0.666</td><td>0.701</td><td>0.415</td><td>-0.10</td><td>+1.10</td></tr><tr><td rowspan="3">YOL011</td><td>Baseline</td><td>0.650</td><td>0.656</td><td>0.711</td><td>0.393</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.633</td><td>0.667</td><td>0.710</td><td>0.412</td><td>-0.10</td><td>+1.90</td></tr><tr><td>Differencing</td><td>0.674</td><td>0.655</td><td>0.717</td><td>0.409</td><td>+0.60</td><td>+1.60</td></tr><tr><td rowspan="3">YOLOv12</td><td>Baseline</td><td>0.661</td><td>0.656</td><td>0.708</td><td>0.393</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.638</td><td>0.683</td><td>0.702</td><td>0.398</td><td>-0.60</td><td>+0.50</td></tr><tr><td>Differencing</td><td>0.652</td><td>0.680</td><td>0.723</td><td>0.413</td><td>+1.50</td><td>+2.00</td></tr><tr><td rowspan="3">MambaYOLO</td><td>Baseline</td><td>0.672</td><td>0.710</td><td>0.744</td><td>0.426</td><td>一</td><td></td></tr><tr><td>Multi-Continuity</td><td>0.684</td><td>0.720</td><td>0.757</td><td>0.430</td><td>+1.30</td><td>+0.40</td></tr><tr><td>Differencing</td><td>0.740</td><td>0.679</td><td>0.747</td><td>0.430</td><td>+0.30</td><td>+0.40</td></tr><tr><td rowspan="3">DETR</td><td>Baseline</td><td>0.527</td><td>0.506</td><td>0.509</td><td>0.257</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.667</td><td>0.531</td><td>0.598</td><td>0.314</td><td>+8.90</td><td>+5.70</td></tr><tr><td>Differencing</td><td>0.587</td><td>0.555</td><td>0.568</td><td>0.288</td><td>+5.90</td><td>+3.10</td></tr></table>

Table A13: Detector-wise results on NEU-DET using 75% of the training data.
<table><tr><td>Detector</td><td>Method</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>∆mAP@0.5</td><td>∆mAP@0.5:0.95</td></tr><tr><td rowspan="3">YOLOv8</td><td>Baseline</td><td>0.668</td><td>0.689</td><td>0.702</td><td>0.378</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.661</td><td>0.685</td><td>0.722</td><td>0.414</td><td>+2.00</td><td>+3.60</td></tr><tr><td>Differencing</td><td>0.667</td><td>0.684</td><td>0.724</td><td>0.408</td><td>+2.20</td><td>+3.00</td></tr><tr><td rowspan="3">YOLOv10</td><td>Baseline</td><td>0.638</td><td>0.630</td><td>0.676</td><td>0.372</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.669</td><td>0.638</td><td>0.688</td><td>0.395</td><td>+1.20</td><td>+2.30</td></tr><tr><td>Differencing</td><td>0.613</td><td>0.632</td><td>0.659</td><td>0.380</td><td>-1.70</td><td>+0.80</td></tr><tr><td rowspan="3">YOL011</td><td>Baseline</td><td>0.605</td><td>0.678</td><td>0.693</td><td>0.386</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.664</td><td>0.626</td><td>0.700</td><td>0.387</td><td>+0.70</td><td>+0.10</td></tr><tr><td>Differencing</td><td>0.643</td><td>0.675</td><td>0.704</td><td>0.383</td><td>+1.10</td><td>-0.30</td></tr><tr><td rowspan="3">YOLOv12</td><td>Baseline</td><td>0.647</td><td>0.666</td><td>0.706</td><td>0.374</td><td>一</td><td></td></tr><tr><td>Multi-Continuity</td><td>0.645</td><td>0.648</td><td>0.703</td><td>0.380</td><td>-0.30</td><td>+0.60</td></tr><tr><td>Differencing</td><td>0.706</td><td>0.608</td><td>0.681</td><td>0.372</td><td>-2.50</td><td>-0.20</td></tr><tr><td rowspan="3">MambaYOLO</td><td>Baseline</td><td>0.685</td><td>0.683</td><td>0.735</td><td>0.404</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.679</td><td>0.679</td><td>0.731</td><td>0.413</td><td>-0.40</td><td>+0.90</td></tr><tr><td>Differencing</td><td>0.676</td><td>0.692</td><td>0.729</td><td>0.404</td><td>-0.60</td><td>+0.00</td></tr><tr><td rowspan="3">DETR</td><td>Baseline</td><td>0.525</td><td>0.457</td><td>0.489</td><td>0.243</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.608</td><td>0.479</td><td>0.526</td><td>0.269</td><td>+3.70</td><td>+2.60</td></tr><tr><td>Differencing</td><td>0.642</td><td>0.529</td><td>0.571</td><td>0.286</td><td>+8.20</td><td>+4.30</td></tr></table>

Table A14: Detector-wise results on NEU-DET using 50% of the training data.
<table><tr><td>Detector</td><td>Method</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>∆mAP@0.5</td><td>∆mAP@0.5:0.95</td></tr><tr><td rowspan="3">YOLOv8</td><td>Baseline</td><td>0.609</td><td>0.623</td><td>0.654</td><td>0.338</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.643</td><td>0.688</td><td>0.710</td><td>0.396</td><td>+5.60</td><td>+5.80</td></tr><tr><td>Differencing</td><td>0.666</td><td>0.684</td><td>0.709</td><td>0.386</td><td>+5.50</td><td>+4.80</td></tr><tr><td rowspan="3">YOLOv10</td><td>Baseline</td><td>0.618</td><td>0.608</td><td>0.651</td><td>0.359</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.655</td><td>0.637</td><td>0.680</td><td>0.373</td><td>+2.90</td><td>+1.40</td></tr><tr><td>Differencing</td><td>0.637</td><td>0.656</td><td>0.680</td><td>0.362</td><td>+2.90</td><td>+0.30</td></tr><tr><td rowspan="3">YOL011</td><td>Baseline</td><td>0.581</td><td>0.633</td><td>0.669</td><td>0.355</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.607</td><td>0.660</td><td>0.690</td><td>0.375</td><td>+2.10</td><td>+2.00</td></tr><tr><td>Differencing</td><td>0.592</td><td>0.672</td><td>0.673</td><td>0.366</td><td>+0.40</td><td>+1.10</td></tr><tr><td rowspan="3">YOLOv12</td><td>Baseline</td><td>0.612</td><td>0.653</td><td>0.676</td><td>0.366</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.618</td><td>0.661</td><td>0.687</td><td>0.373</td><td>+1.10</td><td>+0.70</td></tr><tr><td>Differencing</td><td>0.778</td><td>0.601</td><td>0.681</td><td>0.372</td><td>+0.50</td><td>+0.60</td></tr><tr><td rowspan="3">MambaYOLO</td><td>Baseline</td><td>0.605</td><td>0.656</td><td>0.677</td><td>0.369</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.650</td><td>0.650</td><td>0.692</td><td>0.373</td><td>+1.50</td><td>+0.40</td></tr><tr><td>Differencing</td><td>0.658</td><td>0.674</td><td>0.701</td><td>0.380</td><td>+2.40</td><td>+1.10</td></tr><tr><td rowspan="3">DETR</td><td>Baseline</td><td>0.423</td><td>0.544</td><td>0.469</td><td>0.232</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.583</td><td>0.530</td><td>0.532</td><td>0.270</td><td>+6.30</td><td>+3.80</td></tr><tr><td>Differencing</td><td>0.566</td><td>0.533</td><td>0.554</td><td>0.276</td><td>+8.50</td><td>+4.40</td></tr></table>

Table A15: Detector-wise results on NEU-DET using 25% of the training data.
<table><tr><td>Detector</td><td>Method</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>∆mAP@0.5</td><td>∆mAP@0.5:0.95</td></tr><tr><td rowspan="3">YOLOv8</td><td>Baseline</td><td>0.523</td><td>0.514</td><td>0.590</td><td>0.287</td><td>1</td><td></td></tr><tr><td>Multi-Continuity</td><td>0.598</td><td>0.621</td><td>0.660</td><td>0.335</td><td>+7.00</td><td>+4.80</td></tr><tr><td>Differencing</td><td>0.600</td><td>0.623</td><td>0.655</td><td>0.339</td><td>+6.50</td><td>+5.20</td></tr><tr><td rowspan="3">YOLOv10</td><td>Baseline</td><td>0.566</td><td>0.536</td><td>0.571</td><td>0.293</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.549</td><td>0.571</td><td>0.582</td><td>0.298</td><td>+1.10</td><td>+0.50</td></tr><tr><td>Differencing</td><td>0.637</td><td>0.656</td><td>0.680</td><td>0.362</td><td>+10.90</td><td>+6.90</td></tr><tr><td rowspan="3">YOL011</td><td>Baseline</td><td>0.518</td><td>0.562</td><td>0.579</td><td>0.299</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.667</td><td>0.573</td><td>0.636</td><td>0.317</td><td>+5.70</td><td>+1.80</td></tr><tr><td>Differencing</td><td>0.526</td><td>0.604</td><td>0.628</td><td>0.322</td><td>+4.90</td><td>+2.30</td></tr><tr><td rowspan="3">YOLOv12</td><td>Baseline</td><td>0.512</td><td>0.598</td><td>0.579</td><td>0.296</td><td></td><td>一</td></tr><tr><td>Multi-Continuity</td><td>0.680</td><td>0.568</td><td>0.626</td><td>0.316</td><td>+4.70</td><td>+2.00</td></tr><tr><td>Differencing</td><td>0.506</td><td>0.638</td><td>0.605</td><td>0.305</td><td>+2.60</td><td>+0.90</td></tr><tr><td rowspan="3">MambaYOLO</td><td>Baseline</td><td>0.681</td><td>0.523</td><td>0.552</td><td>0.255</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.686</td><td>0.514</td><td>0.565</td><td>0.267</td><td>+1.30</td><td>+1.20</td></tr><tr><td>Differencing</td><td>0.583</td><td>0.600</td><td>0.624</td><td>0.292</td><td>+7.20</td><td>+3.70</td></tr><tr><td rowspan="3">DETR</td><td>Baseline</td><td>0.441</td><td>0.420</td><td>0.393</td><td>0.196</td><td></td><td></td></tr><tr><td>Multi-Continuity</td><td>0.440</td><td>0.427</td><td>0.420</td><td>0.200</td><td>+2.70</td><td>+0.40</td></tr><tr><td>Differencing</td><td>0.443</td><td>0.458</td><td>0.446</td><td>0.216</td><td>+5.30</td><td>+2.00</td></tr></table>

![](images/003dfb0b3f5b2638ce4644e53d0672f33549e86f47f7ee63e92633aac59ba2f1.jpg)

## B Additional Qualitative Results

We provide additional qualitative analyses to further examine the effect of the proposed continuity-driven regularization on intermediate feature representations. Specifically, we compare the spatial feature responses and local continuity discrepancies between the baseline YOLOv12 detector and its continuity-regularized counterpart.

![](images/f8250e161cf93a3fdae4006f9b899126b4b8a70f7c79585ad38f184b7767c8ee.jpg)

![](images/c4c9cb47339a14a7a07f549520afb5014cbac3ee331275c84bc2159f51c688ab.jpg)

![](images/dc40699696f2622f5df5b42cac4a816ecb327f42943ecc5d2afc6beaadcea923.jpg)

![](images/6fa3aab824e97cf0e9ae4e1de9916453d049909c621f1faf84cfdc796c19eeba.jpg)

![](images/dc3d9f0dee3b0642630571e7c92572408fb6bdda67d5e3b26aba2bc66a42f0ca.jpg)  
(a) Baseline

![](images/7dfdbf911efa5a5917a769ce5adfddce0b52c1b0ef37a936d24f4370e2fb05f3.jpg)

![](images/25218b9d929d2fb7d7dc081286b2c46b2ba2b36839ea0661513e87bfaf3294f2.jpg)

![](images/2331347dad3ba3d2fd0460a75bc7e8d3ecedbc17a7da9e4bfe8031c1ec2b692e.jpg)  
(b) Continuity-regularized

Figure B1: Qualitative comparison of feature representations from YOLOv12. The baseline model produces more dispersed feature responses, whereas the proposed continuity-driven regularization yields more localized activations around defect regions while preserving the underlying normal structures.  
![](images/a0086792a8b7138f663b70728fa8948541eb79e8b6a8264f9347ec29ceda6fa6.jpg)  
(a) Baseline  
(b) Continuity-regularized  
Figure B2: Continuity discrepancy visualization from intermediate YOLOv12 feature maps. The left column shows the baseline model trained with the standard detection loss, and the right column shows the continuity-regularized model trained with the proposed auxiliary objective. Continuity discrepancy maps visualize local violations of feature continuity and are used only as diagnostic visualizations, not as detector outputs. Strong responses may occur around defect-related disruptions as well as normal object boundaries or abrupt structural transitions.

## C Training Hyperparameters

This appendix summarizes the training hyperparameters used in our experiments. We report the method-specific regularization settings and the detector-specific training configurations for YOLO-based models and DETR.

Table C1: Method-specific hyperparameters for the proposed regularization methods.
<table><tr><td>Method</td><td>Hyperparameter</td><td>Setting / Search Space</td></tr><tr><td>Baseline</td><td>Additional loss</td><td>None</td></tr><tr><td rowspan="3">Multi-Continuity</td><td>1D continuity weight</td><td> $\beta _ { \mathrm { 1 D } } \in \{ 0 , 0 . 0 2 , 0 . 0 4 , 0 . 0 6 , 0 . 0 8 \}$ </td></tr><tr><td>2D continuity weight</td><td> $\beta _ { \mathrm { 2 D } } \in \{ 0 , 0 . 0 2 , 0 . 0 4 , 0 . 0 6 , 0 . 0 8 \}$ </td></tr><tr><td>Box-region weight</td><td> $\gamma \in \{ 0 . 1 , 0 . 2 , 0 . 3 \}$ </td></tr><tr><td rowspan="2">Differencing</td><td>First-order variation weight</td><td> $\alpha _ { 1 } \in \{ 0 . 0 , 0 . 1 , 0 . 3 , 0 . 5 \}$ </td></tr><tr><td>Second-order curvature weight</td><td> $\alpha _ { 2 } \in \{ 0 . 0 , 0 . 5 , 1 . 0 , 1 . 5 , 2 . 0 \}$ </td></tr></table>

Table C2: Common training hyperparameters for YOLO-based detector experiments.  
Hyperparameter Setting   
Model weights YOLOv8m, YOLOv10m, YOLO11m, YOLOv12m   
Image size 1280   
Epochs 300   
Early stopping patience 20   
Batch size 4 for YOLOv8/10/11; 2 for YOLOv12   
Learning rate schedule lr0 = 0.01, lrf = 0.01   
Momentum / weight decay 0.937 / 0.0005   
Detection loss weights Box = 7.5; class = 0.5; DFL = 1.5   
Data augmentation Mosaic = 0.6; MixUp = 0.15; copy-paste = 0.4; scale = 0.9; horizontal flip = 0.5

Table C3: DETR-specific training and model hyperparameters.  
Hyperparameter Setting   
Model architecture DETR-R50 with ResNet-50 backbone   
Input resizing COCO-style multi-scale resizing   
Object queries 200   
Position embedding Sine   
Transformer configuration Hidden dim. = 256; encoder layers = 6; decoder layers = 6; attention heads = 8   
Feed-forward / dropout FFN dim. = 2048; dropout = 0.1   
Auxiliary decoder loss Enabled   
Learning rates Transformer/head l $\mathrm { r } = 1 \times 1 0 ^ { - 4 }$ ; backbone $\mathrm { l r } = 1 \times 1 0 ^ { - 5 }$   
Weight decay 1 × 10<sup>−4</sup>   
Epochs 300   
Early stopping patience 20   
Batch size 4   
Hungarian matching cost Class = 1; bbox = 5; GIoU = 2   
Detection loss weights CE = 1; bbox = 5; GIoU = 2; no-object coefficient = 0.1