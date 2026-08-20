RESEARCH ARTICLE OPEN ACCESS

# ADVANCEDINTELLIGENTSYSTEMS

# FD-CanKD: Frequency-Decoupled Cross-Attention Distillation as a Refinement Prior for www.advint www.advintellsyst.com Compact Object Detectors

YoungJae Cheong | Jhonghyun An

<sup>1</sup>Department of AI and Software, Gachon University, Seongnam, Republic of Korea |

Correspondence: Jhonghyun An (jhonghyun@gachon.ac.kr)

Received: Revised: Accepted:

Funding: This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Ministry of Science and ICT (MSIT), Republic of Korea (No. RS-2024-00397615).

Keywords: Object detection | YOLOv12 | knowledge distillation | cross-attention distillation | frequency-decoupled distillation

## ABSTRACT

Compact object detectors are suitable for resource-constrained visual perception, but their limited representation capacity creates an accuracy gap relative to large models. Conventional detector distillation often relies on prediction-level supervision or a single feature-alignment target, such as response, distribution, correlation, or frequency-domain matching. Frequency-Decoupled Cross-Attention Knowledge Distillation (FD-CanKD) is presented as a detector-oriented framework that transfers teacher knowledge at three complementary levels: head-level prediction supervision, relation-level non-local context transfer, and frequency-level component-selective alignment. Student features first aggregate teacher-side spatial context through cross-attention-based relation transfer, after which frequency-aware alignment preserves complementary structural and detail-sensitive cues. Under controlled Microsoft Common Objects in Context (COCO) experiments, fixed 50-epoch from-scratch comparisons show that FD-CanKD remains competitive with representative detector knowledge distillation baselines. Post-distillation continued fine-tuning further produces a stronger refinement-ready student than detector-only fine-tuning, reaching 48.87 mean average precision (mAP) at intersection-over-union thresholds from 0.50 to 0.95 $( \mathrm { m A P } _ { 5 0 : 9 5 } )$ , 65.84 $\mathrm { \ m A P } _ { 5 0 } ;$ , and 53.40 mAP after 20 additional epochs. All distillation modules are removed after training, leaving the deployed student unchanged at 19.7M parameters. The framework is instantiated and evaluated in a controlled YOLOv12 teacher–student setting as a representative compact-detector case study.

## 1 Introduction

Object detection is a fundamental task for autonomous driving, surveillance, robotics, remote sensing, and edge-based visual perception. Among eficient one-stage detectors, the You Only Look Once (YOLO) family has been widely adopted because it provides a practical balance between detection accuracy and inference eficiency. Since the original YOLO, subsequent variants have improved multi-scale prediction, feature representation, training strategies, deployment eficiency, and end-to-end detection capability [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]. Recently, YOLOv12 introduced an attention-centric eficient detector that enhances representation quality while maintaining eficient inference [12]. Beyond

YOLO-based detectors, DETR-style eficient detectors have also improved query denoising, encoder eficiency, and end-to-end detection performance [13, 14, 15].

Despite these advances, a clear accuracy and eficiency gap remains between large-scale and compact detectors. Large-scale detectors generally provide stronger representation capacity and higher detection accuracy, but they require considerable computation and memory. Compact detectors are more suitable for resource-constrained deployment, but their limited capacity often leads to degraded classification confidence, localization accuracy, and localized detection responses. Therefore, transferring informative representations from a high-capacity teacher detector to a compact student detector is an important problem for accurate and eficient object detection, regardless of the particular detector family used to instantiate the teacher and student.

Knowledge distillation (KD) provides a promising solution by transferring knowledge from a teacher model to a student model [16, 17]. Early studies mainly focused on output-level distillation and feature hints, while later methods improved the transfer process through mutual learning, decoupled log its, output correlations, hierarchical feature review, relational constraints, correlation-based alignment, and selective feature imitation [18, 19, 20, 21, 22, 23, 24, 25]. For object detection, distillation has further evolved toward detector-specific supervision, including intermediate feature transfer, foreground and global feature distillation, channel-wise response alignment, masked generative feature reconstruction, difusion-based feature refinement, instance-conditional supervision, distillation search, kernel alignment, and frequency-aware transfer [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]. Despite this broad progress, direct target matching remains a common baseline and may become restrictive when heterogeneous teacher responses are transferred to a substantially smaller student using a single alignment rule.

As illustrated in Figure 1, head-only distillation transfers the final prediction behavior of the teacher, but it does not explicitly guide the intermediate feature hierarchy. Direct feature distillation transfers feature-level information, but it commonly relies on point-wise matching between teacher and student features. Such rigid matching can overlook non-local teacher-student spatial relations and can make the student follow teacher-specific local responses. This issue is particularly important for compact detectors because teacher features contain heterogeneous information, including object-level structures, contextual responses, boundary-sensitive activations, and local variations that may not be equally transferable. In contrast, Figure 1(c) summarizes the detector-oriented integration adopted in FD-CanKD, where head-output supervision is coupled with relation-aware feature transfer and component-selective alignment.

These observations indicate that large-to-small detector distillation should not simply increase the amount of teacher supervision. Instead, teacher knowledge should be organized according to its role in dense object detection. Prediction outputs should guide the final classification and localization behavior of the student, multi-scale feature relations should transfer contextual dependencies among objects and surrounding regions, and a component-selective alignment mechanism should avoid impos ing the same matching strength on all transferred responses. In this study, frequency decomposition is used as one practical mechanism for implementing such selective alignment rather than being treated as a general deficiency of prior detector distillation methods.

Attention and relation modeling provide a natural mechanism for transferring contextual feature dependencies. Transformer attention captures long-range dependencies [37], non-local neural networks model global spatial relations beyond local convolutional neighborhoods [38], and relation networks exploit contextual interactions for object detection [39]. Based on this principle, crossattention-based non-local feature transfer allows student features to aggregate teacher-side spatial context [40]. In FD-CanKD, this relation transfer is applied to semantically corresponding multiscale detection features, allowing the student to refer to teacher responses across spatial positions rather than matching only the same spatial location. Channel and resolution diferences between the selected teacher and student features can be handled through learnable projections and spatial resampling.

![](images/3dc8455dc0e9bdf30458ceec908bb80987b64f41e2f10908023486d5ff3ca8f8.jpg)  
F I G U R E 1 | Comparison of knowledge distillation paradigms for YOLOv12 large-to-small distillation. (a) Head-only knowledge distillation (KD) provides prediction-level supervision only. (b) Direct feature KD performs point-wise feature matching between the teacher and student. (c) FD-CanKD combines head KD, relation-aware transfer, and band-specific feature alignment within a detector-oriented training pipeline [40, 49].

Frequency-domain analysis provides one useful perspective for partitioning heterogeneous detection responses. Prior studies on cross-modal distillation and feature distribution alignment show that representation inconsistency can hinder knowledge transfer [41, 42, 43]. Frequency-based representations, including scattering transforms, wavelet pooling, and frequency-domain learning, capture complementary visual information that may not be explicit in the spatial domain [44, 45, 46]. Frequency-aware distillation methods have also demonstrated that band-dependent supervision can support knowledge transfer [36, 47]. Motivated by these studies, we investigate whether band-specific alignment can complement relation-aware transfer in an intra-modal detector distillation setting. Low-frequency responses are treated as being more associated with coarse structures, whereas highfrequency responses may be more sensitive to boundaries, fine details, and local teacher-student discrepancies. These associations serve as design motivation and are evaluated through controlled ablations.

Although cross-attention-based feature distillation and frequency-decoupled knowledge transfer have been studied in prior work, their direct combination does not automatically define an efective detector distillation framework. Object detection requires simultaneous calibration of classification confidence, localization, and multi-scale dense feature representations. Therefore, this work does not simply introduce relation transfer and frequency-decoupled alignment as independent auxiliary objectives. Instead, they are organized as a detector-oriented refinement mechanism for large-to-small distillation. Teacherguided alignment is evaluated not only by its fixed-budget from-scratch behavior but also by whether it forms a better student representation before subsequent detector fine-tuning. The motivation is not that prior methods universally overlook frequency information, but that a compact detector may benefit from avoiding uniform alignment over heterogeneous transferred responses and from entering the final optimization stage with a more favorable representation.

Motivated by the above analysis, we present Frequency-Decoupled Cross-Attention Knowledge Distillation, referred to as FD-CanKD, as a detector-oriented integration framework for transferring knowledge from a high-capacity teacher to a compact student. The framework assumes that semantically corresponding multi-scale features and detector outputs can be identified between the teacher and student. It combines four components: distillation of detector-native classification and localization outputs, multi-scale cross-attention feature relation transfer, frequency decomposition of relation-enhanced features, and band-specific feature alignment. The low-frequency branch uses MSE alignment, while the high-frequency branch uses relaxed logarithmic alignment to reduce the influence of large local discrepancies. In detectors that employ Distribution Focal Loss, the corresponding DFL distributions can additionally be distilled, whereas other detector families can retain their native localization supervision. This design operationalizes componentselective supervision without changing the deployed student architecture.

Experiments on the Microsoft Common Objects in Context (COCO) dataset [48] instantiate the proposed framework using a YOLOv12-X teacher and a YOLOv12-M student. The evaluation includes diagnostic from-scratch KD comparisons, internal component analysis, frequency-alignment ablation, scale-pair sensitivity analysis, post-distillation fine-tuning, and qualitative evaluation. The fixed 50-epoch comparisons show that FD-CanKD is competitive with representative detector KD baselines and provides the strongest $\mathrm { m A P } _ { 5 0 }$ among the compared KD variants. More importantly, the post-training fine-tuning trajectory shows that FD-CanKD provides a stronger refinement trajectory than detector-only continued fine-tuning, reaching $4 8 . 8 7 \ \mathrm { m A P } _ { 5 0 : 9 5 }$ $6 5 . 8 4 \mathrm { m A P } _ { 5 0 } ,$ and $5 3 . 4 0 \mathrm { m A P } _ { 7 5 }$ after 20 additional epochs. These results indicate that the proposed relation-aware and frequencyaware alignment produces a student representation that can be refined more efectively, while the deployed model remains the original 19.7M-parameter YOLOv12-M student.

While the proposed principles are formulated at the level of general detector-oriented distillation, we adopt YOLOv12 as a controlled representative instantiation in this study, isolating the efect of the proposed alignment strategy from confounding differences across detector architectures and training pipelines.

The main contributions of this paper are summarized as follows:

We present FD-CanKD, a detector-oriented large-to-small distillation framework that integrates detector-native output supervision, cross-attention-based relation transfer, and frequency-decoupled feature alignment within a unified training-time pipeline.

We organize relation-aware and frequency-aware transfer for semantically corresponding multi-scale detection features. Instead of relying on direct point-wise matching, the compact student first aggregates teacher-side spatial context, after which the relation-enhanced features are aligned using component-dependent penalties.

∙ We formulate the framework independently of a particular detector family, provided that corresponding feature levels and prediction outputs can be selected and aligned. YOLOv12 is used as a controlled representative instantiation for empirical evaluation.

∙ We show that detector KD should be analyzed not only as a fixed-budget from-scratch training objective but also as a refinement starting point. Under the controlled 50-epoch diagnostic protocol, KD variants show modest aggregate AP diferences, whereas KD-guided refinement clearly outperforms detector-only fine-tuning.

∙ We provide a three-seed reproducibility check at the distilled before-FT checkpoint, where FD-CanKD obtains $4 8 . 4 2 \pm 0 . 0 7 \mathrm { \ m A P } _ { 5 0 : 9 5 } ,$ compared with $4 8 . 4 1 \pm \ : 0 . 1 2$ for CanKD. Continued FD-CanKD refinement reaches 48.87 $\begin{array} { r } { \operatorname* { m A P } _ { 5 0 : 9 5 } , 6 5 . 8 4 \operatorname* { m A P } _ { 5 0 } , } \end{array}$ , and $5 3 . 4 0 \mathrm { m A P } _ { 7 5 }$ .

Through controlled COCO experiments, component ablations, scale-pair analysis, continued fine-tuning, reproducibility checks, object-size-wise AP analysis, and qualitative comparisons, we show that FD-CanKD improves compact-detector refinement while preserving the original deployed YOLOv12-M architecture.

## 2 Related Work

## 2.1 Eficient Object Detection and YOLO Detectors

Object detection aims to localize and classify objects under practical computation and memory constraints. Among various detector families, YOLO-based detectors have become representative eficient detection frameworks because they provide a practical balance between accuracy and eficiency. The original YOLO formulated object detection as a single-stage regression problem [1], and subsequent variants improved multi-scale prediction, feature representation, training strategies, and deployment eficiency [2, $3 , \ 4 , \ 5 , \ 6 ]$ . Recent YOLO variants further enhanced architectural eficiency and detection accuracy through eficient feature aggregation, anchor-free prediction, programmable gradient information, end-to-end detection, and refined latency–accuracy trade-ofs [7, 8, 9, 10, 11]. YOLOv12 extends this line by introducing an attention-centric eficient detector that improves representation quality while maintaining eficient inference [12]. Recent eficient detectors have also benefited from transformerbased designs and eficient feature processing. Denoising DETR (DN-DETR) improves DEtection TRansformer (DETR) training through query denoising, while Real-Time DETR (RT-DETR) and RT-DETRv2 improve eficient end-to-end detection through efficient encoder design and training refinements [13, 14, 15]. In parallel, eficient backbone and feature aggregation designs such as Cross Stage Partial Network (CSPNet) and Eficient Layer Aggregation Network (ELAN) improve feature reuse and gradient flow [50, 51], and FlashAttention variants reduce the computational and memory overhead of attention operations [52, 53]. Multi-scale feature hierarchies and localization objectives are also important for accurate dense prediction. Feature Pyramid Networks provide representative multi-scale detection features [54], and localization losses such as Generalized Intersection over Union (GIoU), Distance-IoU (DIoU), Intersection over Union (IoU) loss, and distribution-based regression losses improve bounding box optimization [55, 56, 57, 58].

Although these advances have improved detector accuracy and eficiency, compact detectors still have limited representation capacity compared with large-scale models. This limitation is critical in resource-constrained environments, where the deployed model must retain high accuracy under strict computation and memory budgets. Therefore, this paper focuses on reducing the performance gap between a large-scale YOLOv12 teacher and a compact YOLOv12 student through detector-specific knowledge distillation.

## 2.2 Knowledge Distillation for Object Detection

Knowledge distillation transfers knowledge from a high-capacity teacher model to a compact student model and has been widely used for model compression and performance improvement [16, 17]. Early studies mainly used softened output distributions or intermediate feature hints [18]. Later methods improved the transfer process through collaborative learning, decoupled logit supervision, output correlation modeling, hierarchical feature review, relational constraints, Pearson correlation alignment, and selective feature imitation [19, 20, 21, 22, 23, 24, 25]. These studies show that efective distillation requires more than matching final outputs.

For object detection, knowledge distillation is more challenging than image classification because the student must learn both classification and localization. Detector-specific distillation has progressed along several directions. Feature-level methods transfer intermediate teacher representations to the student [26, 27], while region- and response-oriented methods emphasize foreground regions, global context, and channel-wise responses. Representative examples include Feature Knowledge Distillation (FKD) [28], Focal and Global Distillation (FGD) [29], and Channel-Wise Distillation (CWD) [30]. Reconstruction and refinement-based methods further improve feature transfer through Masked Generative Distillation (MGD) [31] and knowledge difusion [32]. Other studies exploit instance-conditional supervision, distillation search, Pearson-correlation-based Knowledge Distillation (PKD), centered kernel alignment, and semantic frequency prompting, including Instance-Conditional Knowledge Distillation (ICKD) [33], DetKDS [34], Pearson-centered kernel alignment (PCKA) [35], and FreeKD [36].

Despite these advances, many detector distillation methods still treat teacher outputs or teacher features as targets to be directly matched. Prediction-level distillation transfers the final decision behavior of the teacher, but it provides limited guidance for intermediate detection features. Direct feature distillation provides dense supervision, but it often aligns teacher and student features at corresponding spatial locations without considering teacher– student spatial dependencies. When the student capacity is much smaller than the teacher capacity, such rigid feature imitation can guide the student toward responses that are not equally transferable or necessary. This limitation motivates a distillation design that preserves prediction behavior while avoiding uniform point-wise feature matching.

## 2.3 Relation-Aware Feature Distillation

Attention mechanisms are efective for modeling long-range dependencies and have become a fundamental component in modern representation learning [37]. Non-local neural networks extend this idea to visual feature maps by capturing global spatial dependencies beyond local convolutional neighborhoods [38]. Relation Networks also show that object-level relationships and contextual interactions are useful for detection tasks [39]. These studies suggest that feature distillation can benefit from modeling spatial relations rather than relying only on independent point-wise matching.

Cross-Attention-based Non-local Knowledge Distillation (CanKD) applies this idea by allowing student feature responses to attend to teacher feature responses [40]. Instead of forcing the student to match teacher features only at identical spatial positions, this strategy enables the student to aggregate teacher-side spatial context. This property is suitable for dense prediction tasks such as object detection because multi-scale detection features contain contextual dependencies across objects, background regions, and diferent spatial scales.

However, relation modeling alone does not determine how strongly each transferred feature component should be aligned. Teacher features contain heterogeneous information, including coarse structures, boundary-sensitive responses, local details, and teacher-specific variations. These components may not be equally transferable to a compact student. Therefore, we investigate a component-selective alignment strategy that controls the strength of feature imitation after spatial context has been transferred.

## 2.4 Frequency-Aware Feature Distillation

Frequency-domain analysis provides one established way to partition feature responses into components with diferent spatial characteristics. Prior studies on cross-modal distillation and feature distribution alignment show that representation inconsistency can make knowledge transfer dificult [41, 42, 43]. Frequency-based representations, including scattering transforms, wavelet pooling, and frequency-domain learning, capture complementary visual information that may not be explicit in the spatial domain [44, 45, 46]. Existing frequency-aware distillation methods further demonstrate the feasibility of using band-dependent supervision [36, 47]. Thus, our use of frequency decomposition builds on prior work and is introduced as a component-selective alignment mechanism.

Recent frequency-decoupled distillation studies suggest that lowand high-frequency components can benefit from diferent alignment strengths, particularly when representation consistency differs across components [49]. Low-frequency responses are often associated with coarse spatial structures, whereas high-frequency responses may capture boundaries, fine local responses, and other rapidly varying details. These observations originate primarily from cross-modal settings. In this work, we treat them as a testable design motivation for intra-modal compact detector distillation rather than as a universal property of all detector features.

Building on these observations, we adopt a detector-specific bandselective design. CanKD focuses on cross-attention-based nonlocal feature transfer, while a recent feature-disentanglement cross-modal distillation method studies frequency-decoupled alignment for cross-modal representation discrepancy [49]. This study organizes these complementary principles for compact YOLOv12 detector distillation. Specifically, FD-CanKD applies relation transfer to multi-scale detection features, performs bandspecific alignment after relation enhancement, and jointly optimizes the feature-level terms with detection-head distillation for classification, box regression, and DFL distributions. The distinction lies in this detector-oriented ordering and coupling of relation transfer, band-specific alignment, and detection-head supervision.

![](images/4170606fb06e7448547fe4d91b82fa25c313e3a30efcf29f6d7b36f8aaa609d7.jpg)  
F I G U R E 2 | Overall architecture of the FD-CanKD integration framework. A frozen large-scale YOLOv12 teacher transfers knowledge to a compact YOLOv12 student [12]. The feature-level branch performs cross-attention-based non-local distillation on multi-scale detection features [40], followed by fast Fourier transform (FFT)-based decomposition into low- and high-frequency components and mean squared error (MSE) and log compressed MSE (LogMSE) alignment [49]. The prediction-level branch distills classification, box regression, and Distribution Focal Loss (DFL) outputs from the teacher head to the student head [58].

## 3 Method

## 3.1 Overall Framework

We present Frequency-Decoupled Cross-Attention Knowledge Distillation, referred to as FD-CanKD, as a detectororiented integration framework for compact detector distillation, instantiated and controlled in this study on a YOLOv12 largeto-small setting. The objective is to improve the detection accuracy of a compact YOLOv12 student by using a high-capacity YOLOv12 teacher during training, while keeping the deployed student architecture unchanged. As shown in Figure 2, the largescale YOLOv12 detector is used as a frozen teacher, and the compact YOLOv12 detector is optimized as the student. During inference, all distillation modules are removed, and only the compact student detector is deployed.

FD-CanKD transfers teacher knowledge through two complementary branches. The first branch performs head-output distillation to guide the student’s final classification and localization behavior. The second branch performs feature-level distillation on multi-scale detection features. In this branch, student features are first projected into the teacher feature space, then enhanced by cross-attention-based relation transfer, and finally subjected to band-specific alignment using a frequency decomposition. This ordering is a key design choice: the decomposition is applied after relation-aware transfer, rather than directly on raw feature maps, so that component-selective alignment operates on contextenhanced detection representations.

Building on these prior principles, the framework focuses on their detector-oriented organization for YOLOv12 distillation. Unlike feature-only distillation, the proposed training objective couples relation transfer and component-selective alignment with detection-head distillation. This coupling is important for object detection because improved intermediate representations should also be calibrated with final classification confidence, box regression, and DFL-based localization behavior. Therefore, FD-CanKD difers from direct feature matching in two aspects. First, it transfers teacher-side spatial context instead of matching only the same pixel locations. Second, it applies diferent alignment rules to decomposed components after relation transfer, rather than uniformly matching all responses.

## 3.2 Notation

Let $T _ { \mathrm { { L } } }$ and $S _ { \mathrm { { S } } }$ denote the frozen large-scale YOLOv12 teacher and the trainable compact YOLOv12 student, respectively. Given an input image �, both models generate detection head outputs and multi-scale detection features. The set of feature levels is denoted by �, corresponding to multi-scale detection features such as P3, P4, and P5. For each level $l \in \mathcal { P } , \mathbf { F } _ { l } ^ { T }$ and $\mathbf { F } _ { l } ^ { S }$ denote the teacher and student feature maps.

The teacher and student head outputs contain classification logits, box regression logits, and Distribution Focal Loss (DFL)-based regression distributions. We denote the temperature-scaled class probability distributions by $\mathbf { p } _ { l } ^ { T }$ and $\mathbf { p } _ { l } ^ { S }$ , the bounding box regression logits by $\mathbf { b } _ { l } ^ { T }$ and $\mathbf { b } _ { l } ^ { S } .$ , and the temperature-scaled DFL-based regression distributions by $\mathbf { q } _ { l } ^ { T }$ and $\mathbf { q } _ { l } ^ { S }$ . The relation-enhanced student feature is denoted by $\widehat { \mathbf { F } } _ { l } ^ { s }$

## 3.3 Head-Output Distillation

The head-output distillation branch provides task-level supervision for the student detector. Since object detection requires both category prediction and accurate localization, we distill three types of teacher head outputs: classification logits, bounding box regression logits, and DFL-based regression distributions. The head-output distillation loss is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { h e a d } } = \displaystyle \frac { 1 } { | \mathcal { P } | } \sum _ { l \in \mathcal { P } } \Big ( \beta _ { \mathrm { c l s } } D _ { \mathrm { K L } } ( \mathbf { p } _ { l } ^ { T } | | \mathbf { p } _ { l } ^ { S } ) + \beta _ { \mathrm { r e g } } | | \mathbf { b } _ { l } ^ { T } - \mathbf { b } _ { l } ^ { S } | | _ { 1 } } \\ { + \beta _ { \mathrm { d f l } } D _ { \mathrm { K L } } ( \mathbf { q } _ { l } ^ { T } | | \mathbf { q } _ { l } ^ { S } ) \Big ) , \qquad } \end{array}\tag{1}
$$

where $D _ { \mathrm { K L } } ( \cdot | | \cdot )$ denotes the Kullback-Leibler divergence. The coeficients $\beta _ { \mathrm { c l s } } , \beta _ { \mathrm { r e g } } ,$ , and $\beta _ { \mathrm { d f l } }$ balance classification, regression, and DFL distillation terms, respectively. This branch guides the student to follow the teacher’s final detection behavior, including class confidence, localization tendency, and distribution-based box regression.

## 3.4 Multi-Scale Relation-Aware Feature Transfer

Direct feature matching aligns teacher and student features at corresponding spatial locations. However, detection features contain contextual dependencies across object regions, surrounding background, and diferent object scales. To transfer such contextual information, we apply cross-attention-based relation transfer to multi-scale YOLOv12 detection features.

For each feature level �, the student feature is first projected to the teacher channel dimension using $\textbf { a } 1 \times 1$ adapter $\mathcal { A } _ { l } ( \cdot )$ The adapted student feature attends to the teacher feature and produces the relation-enhanced student feature:

$$
\begin{array} { l } { { \displaystyle { \widetilde { \bf F } _ { l } ^ { S } = \mathcal { A } _ { l } ( { \bf F } _ { l } ^ { S } ) } , } } \\ { { \displaystyle { \widetilde { \bf f } _ { l , i } ^ { S } = \frac { 1 } { N _ { l } } \sum _ { j = 1 } ^ { N _ { l } } \xi ( \widetilde { \bf f } _ { l , i } ^ { S } , { \bf f } _ { l , j } ^ { T } ) g ( { \bf f } _ { l , j } ^ { T } ) } , } } \\ { { \displaystyle \xi ( \widetilde { \bf f } _ { l , i } ^ { S } , { \bf f } _ { l , j } ^ { T } ) = \Theta ( \widetilde { \bf f } _ { l , i } ^ { S } ) ^ { \top } \phi ( { \bf f } _ { l , j } ^ { T } ) } , } \end{array}\tag{2}
$$

where $N _ { l } = H _ { l } W _ { l } ,$ , and $\widetilde { \mathbf { f } } _ { l , i } ^ { S }$ and $\mathbf { f } _ { l , j } ^ { T }$ are feature vectors at spatial positions � and $j ,$ respectively. The functions �(⋅), �(⋅), and $g ( \cdot )$ are implemented using 1×1 convolutional projections. Following CanKD [40], we use an unnormalized dot-product afinity with the factor $1 / N _ { l } ,$ rather than softmax-normalized attention. This formulation preserves direct relative response diferences across teacher positions and avoids introducing an additional probability normalization into the cross-attention non-local operation. The output vectors $\{ \widehat { \mathbf { f } } _ { l , i } ^ { S } \} _ { i = 1 } ^ { N _ { l } }$ are reshaped into $\widehat { \mathbf { F } } _ { l } ^ { S }$ . Through this operation, the student aggregates teacher-side non-local context rather than imitating only the corresponding spatial location.

Algorithm 1 Simplified FD-CanKD training flow   
1: Require: Image �, label �, frozen teacher $T _ { \mathrm { { L } } } ,$ , compact student   
$S _ { \mathrm { { S } } }$   
2: Ensure: Distilled compact student $S _ { \mathrm { { S } } }$   
3: Extract teacher outputs and features: $( \mathcal { O } ^ { T } , \{ \mathbf { F } ^ { T } \} )  T _ { \mathrm { L } } ( I )$   
4: Extract student outputs and features: $( \mathcal { O } ^ { S } , \{ \mathbf { F } ^ { S } \} )  S _ { \mathrm { s } } ( I )$   
5: Compute detection loss: $\mathcal { L } _ { \mathrm { d e t } }  \mathrm { Y O L O L o s s } ( \mathcal { O } ^ { S } , Y )$   
6: Compute head-output distillation loss: $\mathcal { L } _ { \mathrm { h e a d } } $ HeadKD( $\mathcal { O } ^ { T } , \mathcal { O } ^ { S } )$   
7: Apply relation-aware transfer: $\{ \widehat { \mathbf { F } } ^ { S } \} \gets$ RelAttn(�({�<sup>�</sup> }), {�<sup>�</sup> })   
8: Decompose {�<sup>�</sup>} and {�<sup>ˆ�</sup>} into low- and high-frequency components   
9: Compute frequency-decoupled loss: $\mathcal { L } _ { \mathrm { f d } } \gets \mathrm { M S E } _ { \mathrm { l o w } } + \mathrm { L o g M S E } _ { \mathrm { h i g h } }$   
10: Update $S _ { \mathrm { { S } } }$ using ${ \mathcal { L } } _ { \mathrm { t o t a l } }$   
11: Remove distillation modules and deploy $S _ { \mathrm { { S } } }$

## 3.5 Frequency-Decoupled Feature Alignment

Although relation-aware transfer provides spatially enriched feature guidance, the transferred representation still contains heterogeneous responses. Motivated by prior frequency-aware distillation studies, we use frequency decomposition as a practical partitioning mechanism for applying component-dependent supervision. Low-frequency responses are often associated with coarse spatial structures, whereas high-frequency responses may be more sensitive to boundaries, local activations, and teacher– student discrepancies. These associations motivate, but do not by themselves establish, the use of diferent alignment rules.

Accordingly, we decompose the teacher feature $\mathbf { F } _ { l } ^ { T }$ and the relation-enhanced student feature $\widehat { \mathbf { F } } _ { l } ^ { S }$ into low- and highfrequency components. Before the fast Fourier transform (FFT), we remove the channel-wise spatial mean as the direct-current (DC) component to reduce domination by the global feature magnitude. The frequency decomposition is defined as

$$
\begin{array} { r l } & { \mathbf { U } _ { l } ^ { m } = \operatorname { f f t s h i f t } \left( \mathcal { F } _ { 2 D } \left( \mathcal { D } ( \mathbf { F } _ { l } ^ { m } ) \right) \right) , } \\ & { \mathbf { F } _ { l , r } ^ { m } = \operatorname { R e } \left[ \mathcal { F } _ { 2 D } ^ { - 1 } \left( \operatorname { i f f t s h i f t } \left( \mathbf { U } _ { l } ^ { m } \odot \mathbf { M } _ { r } \right) \right) \right] , } \end{array}\tag{3}
$$

where � $\in \{ T , \widehat { S } \} , r \in \{ \mathrm { l o w } , \mathrm { h i g h } \} , \mathcal { D } ( \cdot )$ denotes DC filtering, $\mathcal { F } _ { 2 D } ( \cdot )$ and $\mathcal { F } _ { 2 D } ^ { - 1 } ( \cdot )$ denote the 2D FFT and inverse 2D FFT, respectively, and � is the corresponding frequency mask. The fftshift(⋅) operation moves the zero-frequency component to the center of the spectrum so that the radial masks in Equations (4) and (5) are centered at $( H _ { l } / 2 , W _ { l } / 2 )$ . Before the inverse transform, ifftshift(⋅) restores the original frequency ordering.

To define the low- and high-frequency masks compactly, we use the normalized radial frequency distance:

$$
d _ { l } ( u , v ) = \left\| \left( \frac { u - H _ { l } / 2 } { H _ { l } } , \frac { v - W _ { l } / 2 } { W _ { l } } \right) \right\| _ { 2 } .\tag{4}
$$

The low- and high-frequency masks are then defined as

$$
\mathbf { M } _ { \mathrm { l o w } } ( u , v ) = \mathbb { I } \left[ d _ { l } ( u , v ) \leq \tau \right] , \quad \mathbf { M } _ { \mathrm { h i g h } } = 1 - \mathbf { M } _ { \mathrm { l o w } } ,\tag{5}
$$

where $( u , v )$ denotes the frequency coordinate, �[⋅] is the indicator function, and � is the cutof ratio. In our implementation, $\tau =$ 0.25 is used unless otherwise specified.

The low-frequency branch is aligned with MSE to preserve broad structural correspondence. The high-frequency branch is aligned with a logarithmic mapping that limits the influence of large local discrepancies. Let �<sup>̄</sup> denote the channel-wise normalized feature map, and let $\rho ( x ) = \mathrm { s i g n } ( x ) \log ( 1 + | x | )$ . The frequencydecoupled feature distillation loss is defined as

T A B L E 1 | Controlled comparison with the locally reproduced 50-epoch from-scratch YOLOv12-M student reference and representative detector KD methods under the same YOLOv12-X to YOLOv12-M local training protocol. External feature-distillation methods are combined with the same head-output distillation branch for a fair controlled comparison.
<table><tr><td>Method</td><td>Feature KD Target</td><td>Head KD</td><td> $\mathbf { m A P } _ { 5 0 : 9 5 }$ </td><td> $\mathbf { m A P } _ { 5 0 }$ </td><td> $\mathbf { m A P } _ { 7 5 }$ </td></tr><tr><td>YOLOv12-M Student (50-epoch FS)</td><td></td><td>No</td><td>45.6</td><td>62.0</td><td>50.8</td></tr><tr><td>CWD + Head KD [30]</td><td>Channel-wise distribution</td><td>Yes</td><td>46.3</td><td>62.4</td><td>50.5</td></tr><tr><td>MGD + Head KD [31]</td><td>Masked generative feature</td><td>Yes</td><td>46.3</td><td>62.3</td><td>50.4</td></tr><tr><td>PKD + Head KD [24]</td><td>Pearson correlation</td><td>Yes</td><td>46.2</td><td>62.2</td><td>50.5</td></tr><tr><td>FGD + Head KD [29]</td><td>Focal/global feature</td><td>Yes</td><td>46.3</td><td>62.2</td><td>50.4</td></tr><tr><td>FreeKD + Head KD [36]</td><td>Frequency-domain feature</td><td>Yes</td><td>46.5</td><td>62.7</td><td>50.7</td></tr><tr><td>FD-CanKD</td><td>Relation + frequency</td><td>Yes</td><td>46.5</td><td>62.8</td><td>50.8</td></tr></table>

T A B L E 2 | Selected internal component comparison under the controlled 50-epoch from-scratch YOLOv12-X to YOLOv12-M local training protocol. The table separates relation-aware feature transfer, head-output distillation, and frequency-aware alignment.
<table><tr><td>Method</td><td>Rel.</td><td>Head</td><td>Freq.</td><td> $\mathbf { m A P } _ { 5 0 : 9 5 }$ </td><td> ${ \bf m A P } _ { 5 0 }$ </td><td> $\mathbf { m A P } _ { 7 5 }$ </td></tr><tr><td>YOLOv12-M Student (50-epoch FS)</td><td>一</td><td>一</td><td></td><td>45.6</td><td>62.0</td><td>50.8</td></tr><tr><td>CanKD [40]</td><td>Yes</td><td>No</td><td>No</td><td>46.5</td><td>62.4</td><td>50.5</td></tr><tr><td>CanKD + Head KD</td><td>Yes</td><td>Yes</td><td>No</td><td>46.3</td><td>62.4</td><td>50.5</td></tr><tr><td>FD-CanKD</td><td>Yes</td><td>Yes</td><td>Yes</td><td>46.5</td><td>62.8</td><td>50.8</td></tr></table>

Note. Rel., relation-aware feature transfer. Head, head-output distillation. Freq., frequency-aware alignment.

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { f d } } = \displaystyle \frac { 1 } { | \mathcal { P } | } \sum _ { l \in \mathcal { P } } \Big ( \gamma _ { \mathrm { l o w } } \| \bar { \mathbf { F } } _ { l , \mathrm { l o w } } ^ { T } - \bar { \mathbf { F } } _ { l , \mathrm { l o w } } ^ { \widehat { S } } \| _ { 2 } ^ { 2 } } \\ & { \qquad + \gamma _ { \mathrm { h i g h } } \| \rho ( \bar { \mathbf { F } } _ { l , \mathrm { h i g h } } ^ { T } - \bar { \mathbf { F } } _ { l , \mathrm { h i g h } } ^ { \widehat { S } } ) \| _ { 2 } ^ { 2 } \Big ) , } \end{array}\tag{6}
$$

where $\gamma _ { \mathrm { l o w } }$ and $\gamma _ { \mathrm { h i g h } }$ balance the low- and high-frequency alignment terms. This formulation corresponds to MSE alignment for low-frequency components and LogMSE alignment for highfrequency components. The resulting objective implements different penalties for broad structural correspondence and rapidly varying local discrepancies.

## 3.6 Overall Objective and Training Flow

The final objective combines the original YOLOv12 detection loss, the head-output distillation loss, and the frequency-decoupled feature distillation loss. Following the YOLOv12 training objective [12], the original detection loss consists of box regression, classification, and DFL terms. The total training objective is defined as

$$
\begin{array} { r l } & { \ \mathcal { L } _ { \mathrm { d e t } } = \alpha _ { \mathrm { b o x } } \mathcal { L } _ { \mathrm { b o x } } + \alpha _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } + \alpha _ { \mathrm { d f l } } \mathcal { L } _ { \mathrm { d f l } } , } \\ & { \mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { d e t } } + \lambda _ { \mathrm { h e a d } } \mathcal { L } _ { \mathrm { h e a d } } + \lambda _ { \mathrm { f d } } \mathcal { L } _ { \mathrm { f d } } . } \end{array}\tag{7}
$$

Here, $\lambda _ { \mathrm { h e a d } }$ and $\lambda _ { \mathrm { f d } }$ control the contribution of head-output and feature-level distillation losses, respectively.

Algorithm 1 summarizes the training flow corresponding to Figure 2. The teacher provides head-output and feature-level supervision only during training. After distillation, all trainingtime modules are removed, and only the compact YOLOv12 student is deployed. Therefore, FD-CanKD improves the compact detector through training-time supervision without changing the deployed YOLOv12-M architecture, which is important for resource-constrained intelligent vision systems.

## 4 Experiments

## 4.1 Experimental Setup

We evaluate the FD-CanKD framework on the Microsoft Common Objects in Context (COCO) dataset [48]. The main setting uses YOLOv12-X as the teacher model and YOLOv12-M as the student model. During distillation, the teacher network is frozen, and only the student network and training-time distillation modules are optimized. Unless otherwise stated, all experiments are conducted with an input resolution of 640, a batch size of 16, and 50 training epochs. The main evaluation metrics are mean average precision over IoU thresholds from 0.50 to 0.95 $( \mathbf { m A P _ { 5 0 : 9 5 } } ) _ { : }$ m $\mathrm { A P } _ { 5 0 } ,$ and $\mathrm { m A P } _ { 7 5 }$ on the COCO validation set.

The oficial YOLOv12 study reports COCO results obtained using a substantially longer 600-epoch training schedule [12]. In contrast, the objective of the present experiments is not to reproduce the oficial YOLOv12 leaderboard, but to verify the applicability and component interactions of FD-CanKD under a controlled local environment. Accordingly, the YOLOv12-X, YOLOv12-M, and YOLOv12-S reference models, representative detector KD baselines, and internal distillation variants are evaluated using the same local codebase, input resolution, batch size, 50-epoch training duration, data-processing pipeline, and validation rule. The absolute AP values reported in this paper should therefore be interpreted as internally controlled local results rather than direct reproductions of the oficial YOLOv12 values.

For fair comparison, the representative external KD comparisons and internal component ablations follow the same from-scratch local protocol. In this setting, all student-side models are trained from scratch under the same initialization policy, training schedule, data-processing pipeline, and validation rule. The 50-epoch YOLOv12-M student is used as a fixed from-scratch reference for Tables 1–4. It should not be interpreted as a fully converged standalone detector baseline. Table 5 separately evaluates continued detector optimization under a post-training analysis with its own locally evaluated student and teacher references. This separation prevents the diagnostic from-scratch KD comparisons in Tables 1–4 from being conflated with the warm-start distillation and continued fine-tuning trajectories in Table 5. We additionally evaluate post-distillation fine-tuning and a warm-start distillation protocol to examine whether teacher-guided feature alignment becomes more efective after the compact detector has already learned a stable detection representation. In the warmstart FD-CanKD setting, the student is initialized from locally trained YOLOv12-M student weights and then optimized with the proposed distillation objective under the same local training pipeline.

T A B L E 3 | Frequency-decoupled alignment ablation after relation-aware feature transfer under the controlled 50-epoch from-scratch teacher– student setting.
<table><tr><td>Method</td><td>Low</td><td>High</td><td>Low Loss</td><td>High Loss</td><td> $\mathbf { m A P } _ { 5 0 : 9 5 }$ </td><td> $\mathbf { m A P } _ { 5 0 }$ </td><td> $\mathbf { m A P } _ { 7 5 }$ </td></tr><tr><td>YOLOv12-M Student (50-epoch FS)</td><td>一</td><td>一</td><td>一</td><td>一</td><td>45.6</td><td>62.0</td><td>50.8</td></tr><tr><td>Rel. KD + Head</td><td>一</td><td>一</td><td>一</td><td></td><td>46.3</td><td>62.4</td><td>50.5</td></tr><tr><td>All-MSE</td><td>Yes</td><td>Yes</td><td>MSE</td><td>MSE</td><td>46.5</td><td>62.4</td><td>50.6</td></tr><tr><td>Low-only</td><td>Yes</td><td>No</td><td>MSE</td><td></td><td>46.5</td><td>62.6</td><td>50.7</td></tr><tr><td>High-only</td><td>No</td><td>Yes</td><td></td><td>LogMSE</td><td>46.4</td><td>62.5</td><td>50.3</td></tr><tr><td>Low + High-log</td><td>Yes</td><td>Yes</td><td>MSE</td><td>LogMSE</td><td>46.5</td><td>62.4</td><td>50.8</td></tr></table>

Note. MSE, mean squared error. LogMSE, log-compressed mean squared error.

T A B L E 4 | Scale-pair sensitivity analysis across diferent YOLOv12 teacher–student pairs under the controlled 50-epoch from-scratch local protocol.
<table><tr><td>Scale Pair</td><td> $\mathbf { T e a c h e r m A P } _ { 5 0 : 9 5 }$ </td><td> $\mathbf { S t u d e n t B a s e m A P } _ { 5 0 : 9 5 }$ </td><td> $\mathbf { F D - C a n K D \ m A P } _ { 5 0 : 9 5 }$ </td></tr><tr><td> $\mathrm { Y O L O v 1 2 \mathrm { - } L }  \mathrm { Y O L O v 1 2 \mathrm { - } M }$ </td><td>47.0</td><td>45.6</td><td>46.4</td></tr><tr><td> $\mathrm { Y O L O v 1 2 - X }  \mathrm { Y O L O v 1 2 - M }$ </td><td>49.5</td><td>45.6</td><td>46.5</td></tr><tr><td> $\mathrm { Y O L O v 1 2 - X } \to \mathrm { Y O L O v 1 2 - S }$ </td><td>49.5</td><td>40.9</td><td>41.1</td></tr></table>

Note. All AP values are obtained from the controlled local protocol described in Section 4.1.

The proposed method is implemented on top of a YOLOv12/Ultralytics-style training pipeline. For reproducibility, the main implementation hyperparameters are fixed across the controlled comparisons rather than tuned separately for each baseline or ablation. The values are selected as stable local settings for the 50-epoch controlled protocol and are kept un changed throughout the experiments. The frequency cutof is set $\tan \tau = 0 . 2 5$ , which defines a compact centered low-frequency region while leaving the remaining spectrum to the high-frequency branch. For head-output distillation, the classification, boxregression, and DFL weights are set to $\beta _ { \mathrm { c l s } } ~ = ~ 1 . 0 , \beta _ { \mathrm { r e g } } ~ = ~ 0 . 0 ,$ and $\beta _ { \mathrm { d f l } } ~ = ~ 1 . 0 $ , respectively. Thus, the main implementation uses classification and DFL distribution distillation and does not apply a separate L1 box-regression KD term, avoiding an additional box-logit constraint beyond the original YOLO detection loss. For frequency-decoupled feature alignment, the low- and high-frequency weights are set to $\gamma _ { \mathrm { l o w } } ~ = ~ 1 . 0$ and $\gamma _ { \mathrm { h i g h } } ~ = ~ 0 . 5 ,$ so that broad structural correspondence receives the full MSE penalty while detail-sensitive high-frequency discrepancies are softened. The total loss weights are set to $\lambda _ { \mathrm { h e a d } } ~ = ~ 0 . 2$ and $\lambda _ { \mathrm { f d } } = 0 . 1$ , keeping distillation as auxiliary supervision relative to the primary detection objective. The multi-scale feature weights for P3/P4/P5 are $0 . 2 5 / 1 . 0 / 1 . 0 $ , which reduces the influence of the highest-resolution P3 feature and emphasizes the more semantically stable P4/P5 detection features. The head-KD temperature is $T \ = \ 2 . 0 ;$ , and the teacher confidence threshold is 0.1 to retain broad teacher guidance while filtering extremely low-confidence predictions. The original YOLO detection loss weights are $\alpha _ { \mathrm { b o x } } = 7 . 5 , \alpha _ { \mathrm { c l s } } = 0 . 5 ,$ and $\alpha _ { \mathrm { d f l } } = 1 . 5 . \ \mathrm { A }$ 5-epoch feature warm-up and a 3-epoch head warm-up are used, with late-stage decay factors applied to stabilize distillation after the student has formed initial detection responses.

In addition to the original YOLO detection loss, FD-CanKD applies head-output distillation and feature-level distillation. The feature-level branch operates on multi-scale detection features and combines cross-attention-based relation transfer with frequency-aware alignment. After training, all distillation modules are removed, and only the compact student detector is used for inference.

## 4.2 Controlled Comparison and Ablation Protocol

We evaluate FD-CanKD through representative external comparisons and controlled ablations under the same local YOLOv12-X to YOLOv12-M protocol. Because detector distillation results are sensitive to codebase, schedule, initialization, and evaluation details, the teacher, student, representative external KD baselines, and internal ablation variants are all reproduced within the same local environment. This design allows the efect of the proposed relation-plus-frequency alignment to be compared against representative detector distillation targets without conflating it with diferences in training duration, implementation, or validation protocol.

The quantitative evaluation is organized into six parts. Tables 1– 4 are interpreted as diagnostic comparisons under a fixed 50- epoch from-scratch local protocol. They examine representative detector KD targets, selected internal components, the frequencydecoupled alignment rule, and teacher–student scale sensitivity without using extra detector-only optimization. The post-training behavior is evaluated separately in Table 5, where detector-only fine-tuning and KD-guided fine-tuning are compared from their corresponding local starting points. Random-seed variability at the distilled before-FT checkpoint is reported directly in Table 5 using the sample standard deviation over three random seeds and is discussed as a robustness check rather than as a formal significance test. Finally, Table 6 reports the same local YOLOv12-M student reference used in Table 5, together with a COCO objectsize-wise AP breakdown at the matched +20- and +30-epoch continued fine-tuning checkpoints. The reference row serves only as a common anchor, whereas the size-specific method comparison is made between CanKD and FD-CanKD at the same continued fine-tuning stage. This organization identifies whether the observed FD-CanKD diference is concentrated in a particular object-size regime and whether the trend persists across neighboring refinement stages.

T A B L E 5 | Post-distillation continued fine-tuning analysis under the controlled local protocol. The table compares detector-only continued finetuning with warm-start CanKD and FD-CanKD trajectories, together with locally evaluated student and teacher references. The distilled before-FT rows report the mean and sample standard deviation over three random seeds.
<table><tr><td>Method</td><td>Init</td><td>Stage / Epochs</td><td> $\mathbf { m A P _ { 5 0 : 9 5 } }$ </td><td> $\mathbf { m A P } _ { 5 0 }$ </td><td> $\mathbf { m A P } _ { 7 5 }$ </td></tr><tr><td rowspan="4">YOLOv12-M FT</td><td rowspan="4">一</td><td>+5 FT</td><td>46.66</td><td>63.15</td><td>50.91</td></tr><tr><td>+10 FT</td><td>46.71</td><td>63.25</td><td>50.81</td></tr><tr><td>+20 FT</td><td>46.90</td><td>63.50</td><td>51.01</td></tr><tr><td>+30 FT</td><td>47.02</td><td>63.68</td><td>51.20</td></tr><tr><td rowspan="5">CanKD</td><td rowspan="5">WS</td><td>Distilled, before FT</td><td> $4 8 . 4 1 \pm 0 . 1 2$ </td><td> $6 5 . 0 9 \pm 0 . 1 3$ </td><td> $5 2 . 7 9 \pm 0 . 1 8$ </td></tr><tr><td>+5 FT</td><td>48.50</td><td>65.23</td><td>52.76</td></tr><tr><td>+10 FT</td><td>48.58</td><td>65.38</td><td>52.86</td></tr><tr><td>+20 FT</td><td>48.75</td><td>65.56</td><td>52.94</td></tr><tr><td>+30 FT</td><td>48.80</td><td>65.65</td><td>53.19</td></tr><tr><td rowspan="5">FD-CanKD</td><td rowspan="5">WS</td><td>Distilled, before FT</td><td> $4 8 . 4 2 \pm 0 . 0 7$ </td><td> $6 5 . 0 0 \pm 0 . 1 3$ </td><td>52.85 ± 0.07</td></tr><tr><td>+5 FT</td><td>48.72</td><td>65.52</td><td>53.10</td></tr><tr><td>+10 FT</td><td>48.74</td><td>65.65</td><td>53.13</td></tr><tr><td>+20 FT</td><td>48.87</td><td>65.84</td><td>53.40</td></tr><tr><td>+30 FT</td><td>48.84</td><td>65.82</td><td>53.35</td></tr><tr><td>YOLOv12-M Student</td><td>FS</td><td>Local student (50-ep FS ref.)</td><td>45.63</td><td>62.01</td><td></td></tr><tr><td>YOLOv12-X Teacher</td><td>一</td><td>Local teacher (ref.)</td><td>49.58</td><td>66.29</td><td>50.81 54.04</td></tr></table>

Note. Init denotes the initialization used for the corresponding training or distillation stage. FS denotes from-scratch training with random initialization for 50 epochs, whereas WS denotes warm-start initialization from the locally trained 50-epoch YOLOv12-M student before the 50-epoch distillation stage. The locally evaluated student and teacher rows are references specific to this post-trainin analysis. The 45.63 local student reference is reused consistently in Table 6. The 45.6 dia nostic fromscratch reference in Tables 1–4 belongs to the fixed 50-epoch diagnostic comparison protocol and should not be used interchangeably for cross-table gain calculations. For the CanKD and FD-CanKD “Distilled, before FT” rows, values are reported as mean ± sample standard deviation over three random seeds. FT denotes continued detector fine-tuning without KD loss, and the +5/+10/+20/+30 FT rows denote additional detector fine-tuning epochs from the corresponding starting point. The YOLOv12-X teacher row is a locally evaluated reference and is not a deployed student result.

T A B L E 6 | Object-size-wise COCO AP for the local YOLOv12-M student reference and the +20- and +30-epoch continued fine-tuning checkpoints.
<table><tr><td>Method</td><td>Stage</td><td>Setting</td><td> $\mathbf { m A P } _ { 5 0 : 9 5 }$ </td><td> $\mathbf { A P } _ { S }$ </td><td> $\mathbf { A P } _ { M }$ </td><td> $\mathbf { A P } _ { L }$ </td></tr><tr><td>YOLOv12-M Student (ref.)</td><td>Reference</td><td>Local student</td><td>45.63</td><td>28.92</td><td>51.94</td><td>62.35</td></tr><tr><td>CanKD</td><td>+20 FT</td><td>Relation-aware</td><td>48.75</td><td>29.19</td><td>53.76</td><td>66.09</td></tr><tr><td>FD-CanKD</td><td>+20 FT</td><td>Relation + frequency</td><td>48.87</td><td>30.15</td><td>53.54</td><td>65.41</td></tr><tr><td>CanKD</td><td>+30 FT</td><td>Relation-aware</td><td>48.80</td><td>29.44</td><td>53.87</td><td>66.23</td></tr><tr><td>FD-CanKD</td><td>+30 FT</td><td>Relation + frequency</td><td>48.84</td><td>30.02</td><td>53.43</td><td>65.43</td></tr></table>

Note. FT denotes continued detector fine-tuning. The YOLOv12-M Student (ref.) row reproduces the same 45.63 local student reference used in Table 5 and serves only as a common anchor. It is not a continued fine-tuning checkpoint. The +20 and +30 FT rows are matched-stage comparisons between CanKD and FD-CanKD from their corresponding warm-start distillation trajectories. $\mathbf { A P } _ { S } , \mathbf { A P } _ { M } ,$ and $\mathrm { A P } _ { L }$ follow the COCO area definitions for small, medium, and large objects, respectively [48]. Two neighboring checkpoints are reported to reduce dependence on a single selected stage. All entries are evaluation results from the corresponding saved models. No additional training is performed for this analysis.

## 4.3 Overall Performance Comparison

Table 1 summarizes the from-scratch diagnostic comparison among the locally reproduced 50-epoch YOLOv12-M student reference, representative detector distillation methods, and FD-CanKD under the same local YOLOv12-X to YOLOv12-M protocol.

Under this fixed 50-epoch setting, the locally reproduced YOLOv12-M student reference obtains 45.6 $\mathrm { m A P } _ { 5 0 : 9 5 }$ , 62.0 ${ \mathrm { \ m A P } } _ { 5 0 } ,$ and 50.8 $\mathrm { \ m A P _ { 7 5 } } .$ . This value is used as a controlled from-scratch reference for comparing KD variants, not as a fully optimized detector-only baseline. CWD, MGD, PKD, FGD, and FreeKD combined with head-output distillation obtain m $\mathrm { A P } _ { 5 0 : 9 5 }$ values between 46.2 and 46.5. FD-CanKD obtains 46.5 m $\mathrm { A P } _ { 5 0 : 9 5 } ,$ while reaching 62.8 m $\mathrm { A P } _ { 5 0 }$ and 50.8 m $\therefore \mathrm { P } _ { 7 5 } .$ Although the aggregate AP diference among KD variants is modest, FD-CanKD gives the strongest $\mathrm { m A P } _ { 5 0 }$ while preserving the student-level $\mathrm { \ m A P _ { 7 5 } } .$ . Thus, Table 1 is used to analyze the controlled distillation behavior under the same training budget, while the efect of additional detector-only optimization is evaluated separately in Table 5.

## 4.4 Internal Component Ablation

Table 2 analyzes the selected internal components under the from-scratch setting.

![](images/96a35ff513a4b66c97b045a720462f2495bc4baa4788d0c5c50be1da55b07132.jpg)  
F I G U R E 3 | Qualitative comparison on cluttered and occluded scenes. From left to right in each group, the results correspond to the YOLOv12-M student, the relation-aware KD variant, and FD-CanKD. The baseline student often shows lower confidence or unstable predictions in cluttered regions, while relation-aware KD improves several object responses. FD-CanKD combines relation-aware feature transfer with band-specific alignment and shows more stable detections, including fine-grained local object responses, in the illustrated examples.

The locally reproduced 50-epoch YOLOv12-M student reference obtains $4 5 . 6 \ \mathrm { m A P } _ { 5 0 : 9 5 }$ . CanKD alone reaches 46.5, and adding head-output distillation reaches 46.3. The full from-scratch FD-CanKD also reaches 46.5 while producing the strongest $\mathrm { m A P } _ { 5 0 }$ among the internal variants and preserving m $\mathrm { A P } _ { 7 5 }$ relative to the student. These results show that relation transfer and frequencyaware alignment provide useful guidance under the controlled protocol, while the small numerical diferences among internal variants suggest that aggregate AP alone does not fully explain the qualitative behavior of the detectors.

## 4.5 Frequency-Decoupled Alignment Ablation

Table 3 analyzes alternative alignment rules for the decomposed feature components under the from-scratch setting. The locally reproduced 50-epoch YOLOv12-M student reference obtains 45.6 $\mathrm { m A P } _ { 5 0 : 9 5 } , 6 2 . 0 \mathrm { \ m A P } _ { 5 0 } ,$ and 50.8 m $\mathsf { A P } _ { 7 5 } .$ The Rel. $\mathrm { K D } + \mathrm { H e a d }$ baseline obtains $4 6 . 3 \mathrm { \ m A P } _ { 5 0 : 9 5 } , \mathrm { \ 6 2 . 4 \ m A P } _ { 5 0 } ,$ and $5 0 . 5 \mathrm { m A P } _ { 7 5 }$ . The All-MSE and Low-only settings both reach 46.5 $\mathrm { m A P } _ { 5 0 : 9 5 } ,$ , while the High-only setting reaches 46.4. The final Low + High-log setting reaches $4 6 . 5 \mathrm { \ m A P } _ { 5 0 : 9 5 } , 6 2 . 4 \mathrm { \ m A P } _ { 5 0 } ,$ , and $5 0 . 8 \mathrm { m A P } _ { 7 5 }$ . Although the aggregate AP diferences are small, the asymmetric objective preserves the student-level mA $\mathrm { \Delta } P _ { 7 5 }$ while im proving m $\mathrm { A P } _ { 5 0 : 9 5 }$ over the student reference. This supports the design choice that low-frequency responses, which tend to encode broader object-level structure and semantic context, can be directly aligned with MSE, whereas high-frequency responses, which contain boundary and local-detail variations, benefit from a softened log-compressed penalty.

## 4.6 Scale-Pair Sensitivity Analysis

Table 4 evaluates FD-CanKD across diferent YOLOv12 teacher– student scale pairs under the from-scratch protocol. In the YOLOv12-L → YOLOv12-M setting, FD-CanKD reaches $4 6 . 4 \mathrm { \ m A P } _ { 5 0 : 9 5 } ,$ compared with 45.6 for the student and 47.0 for the teacher. In the YOLOv12-X → YOLOv12-M setting, FD-CanKD reaches 46.5, compared with 45.6 for the student and 49.5 for the teacher. In the YOLOv12-X → YOLOv12-S setting, FD-CanKD reaches 41.1, compared with 40.9 for the student. These results show that the proposed distillation structure remains efective across the tested scale pairs, while the absolute AP depends on the teacher–student capacity relationship.

## 4.7 Post-Distillation Fine-Tuning and Warm-Start Results

Table 5 evaluates a post-training analysis rather than another from-scratch comparison. It compares detector-only YOLOv12-M fine-tuning, warm-start CanKD, and warm-start FD-CanKD, together with locally evaluated YOLOv12-M student and YOLOv12- X teacher references specific to this analysis. The Table 5 student reference obtains $4 5 . 6 3 \mathrm { m A P } _ { 5 0 : 9 5 } , 6 2 . 0 1 \mathrm { m A P } _ { 5 0 } ,$ , and $5 0 . 8 1 \mathrm { m A P } _ { 7 5 } ,$ while the locally evaluated teacher obtains 49.58, 66.29, and 54.04, respectively. The same 45.63 local student reference is reused in Table 6, where its size-wise scores are 28.92 $\operatorname { A P } _ { S } ,$ , 51.94 $\mathrm { A P } _ { M }$ and $6 2 . 3 5 \mathrm { A P } _ { L } .$ . These post-training references are kept separate from the 45.6 diagnostic from-scratch student reference used in Tables 1–4, so gains are interpreted within the corresponding evaluation protocol rather than across table groups.

![](images/d03a0662667354ceaa95fee95c5caed152ef4657f96f40858ffb18ca68bd4540.jpg)  
F I G U R E 4 | Qualitative comparison on challenging scale and illumination variations. From left to right in each group, the results correspond to the YOLOv12-M student, the relation-aware KD variant, and FD-CanKD. The relation-aware KD variant improves teacher–student spatial transfer, while FD-CanKD additionally applies diferent alignment rules to decomposed feature components. The examples illustrate more balanced detections under scale and illumination changes, especially for local objects that require fine spatial detail.

Continuing detector-only fine-tuning reaches $4 7 . 0 2 \mathrm { \ m A P } _ { 5 0 : 9 5 } ,$ $6 3 . 6 8 \mathrm { \ m A P } _ { 5 0 } .$ , and $5 1 . 2 0 ~ \mathrm { m A P } _ { 7 5 }$ after 30 additional epochs. Under the same +30-epoch continued fine-tuning budget, CanKD reaches 48.80, 65.65, and 53.19, whereas FD-CanKD reaches 48.84, 65.82, and 53.35. Relative to detector-only fine-tuning at +30 epochs, FD-CanKD therefore improves the reported result by 1.82 $\mathrm { m A P } _ { 5 0 : 9 5 } ,$ , 2.14 $\mathrm { m A P } _ { 5 0 } ,$ and $2 . 1 5 \mathrm { \ m A P } _ { 7 5 }$ . The strongest FD-CanKD checkpoint in Table 5 occurs at +20 epochs, reaching 48.87 $\mathrm { m A P } _ { 5 0 : 9 5 } , 6 5 . 8 4 \mathrm { m A P } _ { 5 0 } ,$ , and $5 3 . 4 0 \mathrm { m A P } _ { 7 5 } .$ These results show that the observed refinement benefit is not explained by additional detector optimization alone. Teacher-guided relation transfer and frequency-aware alignment provide a more favorable representation for subsequent detector refinement.

To assess run-to-run variability at the distilled before-FT checkpoint, we repeated both CanKD and FD-CanKD with three random seeds. CanKD obtains $4 8 . 4 1 \pm 0 . 1 2 \mathrm { m A P } _ { 5 0 : 9 5 } , 6 5 . 0 9 \pm 0 . 1 3$ $\mathrm { m A P } _ { 5 0 } $ , and $5 2 . 7 9 \pm 0 . 1 8 \mathrm { \ m A P } _ { 7 5 } ,$ , whereas FD-CanKD obtains $4 8 . 4 2 \pm 0 . 0 7 , 6 5 . 0 0 \pm 0 . 1 3$ , and $5 2 . 8 5 \pm 0 . 0 7$ , respectively. The before-FT mean diference in $\mathrm { m A P } _ { 5 0 : 9 5 }$ is therefore only 0.01 point, so the three-seed repetition is not used to claim a statistically significant before-FT superiority over CanKD. Instead, it serves as a reproducibility check: FD-CanKD reproduces a similar before-FT performance level across the tested seeds and shows lower observed sample standard deviation in $\mathrm { m A P } _ { 5 0 : 9 5 }$ and m $\mathsf { A P } _ { 7 5 } ,$ . The practical refinement benefit is therefore supported primarily by the continued fine-tuning trajectory rather than by an overstated before-FT mean separation.

## 4.8 Object-Size-Wise Analysis

To identify where the frequency-decoupled branch contributes most clearly while avoiding dependence on a single selected checkpoint, Table 6 first anchors the analysis with the same local YOLOv12-M student reference used in Table 5: 45.63 m $\mathrm { A P } _ { 5 0 : 9 5 } ,$ 28.92 $\mathrm { A P } _ { S } , 5 1 . 9 4 \mathrm { A P } _ { M }$ , and $6 2 . 3 5 \mathrm { A P } _ { L }$ . This row is a common reference only. The size-specific method comparison is made between CanKD and FD-CanKD at matched +20- and +30-epoch continued fine-tuning checkpoints. At +20 epochs, the overall m $\mathrm { A P } _ { 5 0 : 9 5 }$ diference between CanKD and FD-CanKD is modest at +0.12 point, whereas FD-CanKD improves $\mathsf { A P } _ { S }$ from 29.19 to 30.15, corresponding to a +0.96-point gain. The same direction remains at +30 epochs, where the overall gap narrows to +0.04 point but $\boldsymbol { \mathrm { A P } _ { S } }$ still improves from 29.44 to 30.02, a +0.58-point gain. In contrast, $\mathrm { A P } _ { M }$ and $\mathrm { A P } _ { L }$ do not improve under FD-CanKD at either checkpoint. Therefore, the positive CanKD-to-FD-CanKD diference is repeatedly concentrated on small-object detection rather than uniformly distributed across all object scales.

This scale-specific behavior can be explained by how small objects are represented in multi-scale detection features. A small object occupies only a limited number of spatial locations, and repeated downsampling can substantially weaken its already sparse representation. Consequently, its remaining discriminative evidence is often concentrated in local intensity transitions, object boundaries, and fine structural details, which are more strongly associated with high-frequency responses than the broader structural patterns that dominate larger objects. Small-object recognition is therefore particularly sensitive to whether such fine-detail information survives feature transformation and teacher–student alignment.

![](images/c0f83f919cf2fd1204d0356e3c1b8059261445240e439e63fe21747931f94c71.jpg)  
Relation- and frequency-aware KD

F I G U R E 5 | Qualitative comparison on large objects and context-sensitive objects. From left to right in each group, the results correspond to the YOLOv12-M student, the relation-aware KD variant, and FD-CanKD. The proposed method maintains reliable detection of dominant objects while improving the stability of context-sensitive and fine-grained object predictions in the illustrated examples.

The relation-aware transfer used in CanKD improves non-local contextual interaction by allowing the student to exploit teacherside dependencies across spatial regions. However, relation transfer alone does not explicitly distinguish between slowly varying structural responses and rapidly varying detail-sensitive responses. Under a uniform alignment rule, dominant lowfrequency structure may govern optimization, while weaker highfrequency responses associated with small boundaries or local details can receive comparatively limited emphasis. FD-CanKD instead decomposes the relation-enhanced representation into complementary frequency components and applies componentspecific alignment objectives, preventing broad structural information and detail-sensitive responses from being forced to follow an identical matching rule.

This also explains why the improvement is not uniform across object sizes. Medium and large objects occupy more spatial locations and retain broader shape, regional, and contextual cues even after downsampling, whereas small objects have much less spatial redundancy retain broader shape, regional, and contextual cues even after down. The loss of only a few informative activations can therefore substantially weaken a small object’s representation. The repeated $A P _ { S }$ advantage of FD-CanKD at both +20 and +30 epochs is thus consistent with the hypothesis that frequency-decoupled alignment helps preserve detail-sensitive teacher information that is disproportionately important for spatially small instances.

Importantly, this result should not be interpreted as evidence that all high-frequency information is beneficial or that the frequency branch alone causally determines small-object performance. High-frequency responses may also contain background clutter or noise, and Table 6 provides a two-checkpoint size-wise diagnostic rather than a controlled causal isolation of individual frequency bands. We therefore interpret the repeated $A P _ { S }$ advantage as supportive quantitative evidence consistent with the proposed frequency-decoupled design, rather than as formal statistical proof of a causal frequency efect.

## 4.9 Qualitative Results

Figures 3, 4, and 5 present qualitative comparisons among the YOLOv12-M student, the relation-aware KD variant, and FD-CanKD. In each group, the results are arranged from left to right as the baseline student, relation-aware KD, and FD-CanKD. The highlighted boxes indicate representative changes in confidence, localization, or local object response across the compared methods. Although the aggregate numerical diferences among KD variants are modest, the qualitative examples show that FD-CanKD more consistently preserves fine-grained local object responses and context-dependent detections in cluttered, occluded, or scale-varying scenes.

Figure 3 shows examples with cluttered scenes, overlapping objects, occlusion, and local objects. The baseline student detects dominant objects but often produces lower confidence predictions or unstable bounding boxes in cluttered regions. Relationaware KD improves several responses by transferring teacher-side spatial context. FD-CanKD additionally applies band-specific alignment after relation transfer and shows more stable detections, including finer local object responses, in the illustrated cases. Because these examples are qualitative, they are used to complement rather than establish the interpretation of the decomposed feature components.

Figure 4 compares results under challenging scale and illumination variations. In scenes containing distant persons, strong illumination contrast, and large foreground objects, the baseline student may miss context-sensitive objects. Relation-aware KD improves contextual object responses, while FD-CanKD additionally uses the asymmetric band-alignment objective and produces more balanced predictions for fine-grained local regions in the illustrated cases.

Figure 5 shows cases involving large objects and context-sensitive objects. The baseline student reliably detects large objects but can produce unstable responses for surrounding objects. Relationaware KD improves some local responses through spatial context transfer. The final FD-CanKD model shows more consistent behavior across dominant objects and fine-grained local object regions.

These qualitative results complement the quantitative findings in Tables 1, 2, 5, and 6. While the aggregate numerical diferences among from-scratch KD variants are small, the visual examples indicate that FD-CanKD better preserves fine-grained local object responses and context-dependent detections in challenging scenes. The object-size-wise breakdown provides additional quantitative support for this interpretation: FD-CanKD improves $\boldsymbol { \mathrm { A P } } _ { S }$ over CanKD by 0.96 point at +20 epochs and by 0.58 point at +30 epochs, while gains are not uniformly observed for medium and large objects. The repeated direction across two neighboring checkpoints reduces dependence on a single selected refinement stage.

## 4.10 Discussion

The experimental results provide six observations. First, the fromscratch comparisons in Tables 1 and 2 should be interpreted as diagnostic evaluations under the fixed 50-epoch local protocol. FD-CanKD remains competitive with representative detector KD methods while using the locally reproduced YOLOv12-M student as a common reference. Although the aggregate $\mathrm { m A P } _ { 5 0 : 9 5 }$ diferences among the KD variants are small, FD-CanKD achieves the strongest m $\mathrm { \Delta } \mathrm { P } _ { 5 0 }$ in Table 1 and preserves the student-level m $\mathrm { \nabla { \cdot } } \operatorname { P } _ { 7 5 } .$ This suggests that the proposed structure improves teacherstudent alignment rather than merely adding more supervision terms.

Second, the qualitative results help explain why FD-CanKD remains useful despite modest aggregate AP diferences. Figures 3– 5 show improved preservation of local object responses and context-dependent detections in cluttered, occluded, and scalevarying scenes. This behavior is consistent with relation-aware transfer capturing non-local spatial context and frequency-aware alignment preserving detail-sensitive cues for stable localization. Third, Table 6 provides object-size-wise evidence using the same 45.63 student reference as Table 5. At +20 epochs, FD-CanKD improves $\mathsf { A P } _ { S }$ from 29.19 to 30.15 over CanKD, a gain of 0.96 points, while the overall $\mathrm { m A P } _ { 5 0 : 9 5 }$ diference is 0.12 points. At +30 epochs, the overall diference decreases to 0.04 points, but the $\boldsymbol { \mathrm { A P } } _ { S }$ gain remains 0.58 points, from 29.44 to 30.02. Comparable improvements are not observed for medium or large objects. The repeated concentration of the positive diference in $\mathsf { A P } _ { S }$ is consistent with the motivation that frequency-aware alignment benefits boundary and fine-detail responses, although this analysis is treated as diagnostic evidence rather than a formal significance test.

Fourth, Table 5 demonstrates the benefit of KD-guided refinement over detector-only continued fine-tuning. After 30 additional epochs, detector-only fine-tuning reaches $4 7 . 0 2 \mathrm { \ m A P } _ { 5 0 : 9 5 } ,$ $6 3 . 6 8 \mathrm { \ m A P } _ { 5 0 } ,$ and $5 1 . 2 0 \mathrm { m A P } _ { 7 5 } .$ . Under the same budget, CanKD reaches 48.80, 65.65, and 53.19, whereas FD-CanKD reaches 48.84, 65.82, and 53.35. FD-CanKD obtains its strongest checkpoint after 20 additional epochs, reaching 48.87, 65.84, and 53.40, respectively. These results show that teacher-guided refinement provides a more favorable optimization trajectory than detectoronly fine-tuning.

Fifth, the three-seed experiment confirms reproducibility at the distilled before-FT checkpoint. CanKD obtains $4 8 . 4 1 ~ \pm ~ 0 . 1 2$ m $\mathrm { A P } _ { 5 0 : 9 5 } ,$ whereas FD-CanKD obtains $4 8 . 4 2 \pm 0 . 0 7$ . The 0.01- point diference is negligible and is not interpreted as statistical superiority. Instead, the lower observed variation for FD-CanKD supports that its before-FT result is reproducible across the tested seeds rather than being caused by one favorable run.

Sixth, the frequency and scale-pair ablations clarify the source and scope of the observed improvements. Component-selective frequency alignment helps preserve stricter localization performance, while the scale-pair experiments show that FD-CanKD remains efective across the evaluated teacher-student sizes under the same local protocol. Together, these findings support FD-CanKD as a structural distillation framework that prepares compact detectors for subsequent refinement.

We note that all reported results are obtained within the YOLOv12 teacher–student family. This single-family control was a deliberate design choice for isolating component efects, and validating the proposed alignment principles on other detector families (e.g., DETR-style or other YOLO variants) is left for future work.

Overall, FD-CanKD should be interpreted as a detector-oriented teacher-student alignment framework rather than an accumulation of auxiliary objectives. It organizes teacher knowledge into prediction-level, relation-level, and frequency-aware cues and applies an appropriate alignment rule to each type. Because all distillation components are removed after training, the deployed YOLOv12-M model retains its original architecture and parameter count. The primary benefit is therefore improved accuracy and refinement behavior without additional inference complexity.

## 5 Conclusion

In this paper, we presented FD-CanKD, a detector-oriented integration framework that combines head-level prediction supervision, cross-attention-based relation transfer, and frequencydecoupled alignment for controlled YOLOv12 large-to-small distillation. Unlike single-target feature matching, FD-CanKD organizes teacher knowledge into complementary prediction, relation, and frequency-aware cues and transfers them with alignment rules suited to dense object detection.

The experiments show that the locally reproduced YOLOv12-M student obtains $4 5 . 6 \ \mathrm { m A P } _ { 5 0 : 9 5 }$ as the controlled 50-epoch fromscratch reference for the diagnostic comparisons in Tables 1– 4. Under this setting, representative detector KD methods and FD-CanKD achieve similar aggregate m $\mathrm { 1 A P _ { 5 0 : 9 5 } }$ values, while FD-CanKD provides the strongest m $\mathrm { A P } _ { 5 0 }$ among the compared

KD variants and preserves the student-level $\mathrm { \ m A P _ { 7 5 } } .$ . Qualitative results further show that FD-CanKD better preserves finegrained local object responses in cluttered, occluded, and scalevarying scenes. In the separate post-distillation continued finetuning analysis ofTable 5, detector-only YOLOv12-M fine-tuning reaches $4 7 . 0 2 \ \mathrm { m A P } _ { 5 0 : 9 5 } , 6 3 . 6 8 \ \mathrm { m A P } _ { 5 0 } ,$ and $5 1 . 2 0 \mathrm { m A P } _ { 7 5 }$ after 30 additional epochs, while FD-CanKD reaches 48.84, 65.82, and 53.35 under the same +30-epoch budget and peaks at 48.87, 65.84, and 53.40 after 20 additional epochs. At the distilled before-FT checkpoint, three-seed repetition gives $4 8 . 4 1 \pm 0 . 1 2 \mathrm { m A P } _ { 5 0 : 9 5 }$ for CanKD and $4 8 . 4 2 \pm 0 . 0 7$ for FD-CanKD. Because the mean difference is only 0.01 point, these seed repetitions are interpreted as a reproducibility check rather than as evidence of a statistically significant before-FT advantage. Across the shared +20- and +30-epoch checkpoints, the object-size-wise breakdown shows that the positive diference is consistently concentrated in smallobject detection: FD-CanKD improves $\boldsymbol { \mathrm { A P } } _ { S }$ from 29.19 to 30.15 over CanKD (+0.96 point) at +20 epochs and from 29.44 to 30.02 (+0.58 point) at +30 epochs. This repeated trend provides quantitative evidence consistent with the proposed fine-detail alignment motivation while avoiding reliance on a single selected checkpoint.

These results indicate that relation-aware and frequency-aware distillation provides a more favorable refinement starting point than detector-only fine-tuning for compact object detectors. Since all auxiliary distillation modules are removed after training, the deployed model remains the original 19.7M-parameter YOLOv12- M student. All experiments in this study are conducted within a single representative teacher–student family, YOLOv12. This setting was deliberately controlled to isolate the efect of relationand frequency-aware alignment without confounding diferences in codebase, training schedule, or initialization across detector architectures. Extending FD-CanKD to other detector families remains an important direction for future work, together with adaptive distillation scheduling, instance-aware alignment, and uncertainty-guided feature selection for more robust compact detector optimization.

## Use of Generative AI

During the preparation of this manuscript, the authors used OpenAI’s ChatGPT to assist with English-language editing, sentence restructuring, and improving the clarity and readability of the manuscript. All AI-assisted content was critically reviewed and revised by the authors, who take full responsibility for the final content of the manuscript.

## Author Contributions

YoungJae Cheong: Conceptualization, Methodology, Software, Validation, Formal Analysis, Investigation, Visualization, and Writing – Original Draft. Jhonghyun An: Conceptualization, Methodology, Supervision, Project Administration, Funding Acquisition, and Writing – Review & Editing. All authors have read and approved the final manuscript.

## Acknowledgements

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Ministry of Science and ICT (MSIT), Republic of

Korea (No. RS-2024-00397615, “Development of SDV-Based Automotive Software Platform Technology Interacted with an AI Framework Required for Intelligent Vehicles”).

## Financial Disclosure

The authors declare no relevant financial interests to disclose.

## Conflicts of Interest

The authors declare no conflicts of interest.

## Data Availability Statement

This study uses the publicly available Microsoft COCO dataset for training and evaluation. The dataset can be accessed through the oficial COCO repository. No additional dataset was generated as part of this work.

## Code Availability Statement

The implementation code, training configuration, hyperparameter settings, and evaluation scripts can be made available from the corresponding author upon reasonable request.

## References

[1] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 779–788.

[2] J. Redmon and A. Farhadi, “YOLO9000: Better, faster, stronger,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 7263–7271.

[3] J. Redmon and A. Farhadi, “YOLOv3: An incremental improvement,” arXiv preprint arXiv:1804.02767, 2018.

[4] A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, “YOLOv4: Optimal speed and accuracy of object detection,” arXiv preprint arXiv:2004.10934, 2020.

[5] G. Jocher, K. Nishimura, T. Mineeva, and R. J. A. M. Vilariño, “YOLOv5,” 2020. [Online]. Available: https://github. com/ultralytics/yolov5

[6] C. Li, L. Li, Y. Geng, H. Jiang, M. Cheng, B. Zhang, Z. Ke, X. Xu, and X. Chu, “YOLOv6 v3.0: A full-scale reloading,” arXiv preprint arXiv:2301.05586, 2023.

[7] C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, “YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for realtime object detectors,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 7464–7475.

[8] Ultralytics, “YOLOv8 anchor-free bounding box prediction,” 2023. [Online]. Available: https://github.com/ultralytics/ ultralytics/issues/189

[9] C.-Y. Wang, I.-H. Yeh, and H.-Y. M. Liao, “YOLOv9: Learning what you want to learn using programmable gradient information,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 1–21.

[10] A. Wang, H. Chen, L. Liu, K. Chen, Z. Lin, J. Han, and G. Ding, “YOLOv10: Real-time end-to-end object detection,” Adv. Neural Inf. Process. Syst., vol. 37, pp. 107984–108011, 2024.

[11] G. Jocher, “YOLOv11,” 2024. [Online]. Available: https:// github.com/ultralytics

[12] Y. Tian, Q. Ye, and D. Doermann, “YOLOv12: Attentioncentric real-time object detectors,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 38, 2025.

[13] F. Li, H. Zhang, S. Liu, J. Guo, L. M. Ni, and L. Zhang, “DN-DETR: Accelerate DETR training by introducing query denoising,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 13619–13627.

[14] Y. Zhao, W. Lv, S. Xu, J. Wei, G. Wang, Q. Dang, Y. Liu, and J. Chen, “DETRs beat YOLOs on real-time object detection,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 16965–16974.

[15] W. Lv, Y. Zhao, Q. Chang, K. Huang, G. Wang, and Y. Liu, “RT-DETRv2: Improved baseline with bag-offreebies for eficient detection transformer,” arXiv preprint arXiv:2407.17140, 2024.

[16] C. Bucilua, R. Caruana, and A. Niculescu-Mizil, “Model compression,” in Proc. 12th ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD), 2006, pp. 535–541.

[17] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531, 2015.

[18] A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta, and Y. Bengio, “FitNets: Hints for thin deep nets,” arXiv preprint arXiv:1412.6550, 2014.

[19] Y. Zhang, T. Xiang, T. M. Hospedales, and H. Lu, “Deep mutual learning,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 4320–4328.

[20] B. Zhao, Q. Cui, R. Song, Y. Qiu, and J. Liang, “Decoupled knowledge distillation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 11953–11962.

[21] T. Huang, S. You, F. Wang, C. Qian, and C. Xu, “Knowledge distillation from a stronger teacher,” Adv. Neural Inf. Process. Syst., vol. 35, pp. 33716–33727, 2022.

[22] P. Chen, S. Liu, H. Zhao, and J. Jia, “Distilling knowledge via knowledge review,” in Proc. IEEE/CVFConf. Comput. Vis. Pattern Recognit. (CVPR), 2021, pp. 5008–5017.

[23] W. Park, D. Kim, Y. Lu, and M. Cho, “Relational knowledge distillation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 3967–3976.

[24] W. Cao, Y. Zhang, J. Gao, A. Cheng, K. Cheng, and J. Cheng, “PKD: General distillation framework for object detectors via Pearson correlation coeficient,” Adv. Neural Inf. Process. Syst., vol. 35, pp. 15394–15406, 2022.

[25] B. Heo, J. Kim, S. Yun, H. Park, N. Kwak, and J. Y. Choi, “A comprehensive overhaul of feature distillation,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2019, pp. 1921– 1930.

[26] Q. Li, S. Jin, and J. Yan, “Mimicking very eficient network for object detection,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 6356–6364.

[27] G.-H. Wang, Y. Ge, and J. Wu, “Distilling knowledge by mimicking features,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 11, pp. 8183–8195, 2021.

[28] L. Zhang and K. Ma, “Improve object detection with featurebased knowledge distillation: Towards accurate and eficient detectors,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2020.

[29] Z. Yang, Z. Li, X. Jiang, Y. Gong, Z. Yuan, D. Zhao, and C. Yuan, “Focal and global knowledge distillation for detectors,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 4643–4652.

[30] C. Shu, Y. Liu, J. Gao, Z. Yan, and C. Shen, “Channelwise knowledge distillation for dense prediction,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 5311– 5320.

[31] Z. Yang, Z. Li, M. Shao, D. Shi, Z. Yuan, and C. Yuan, “Masked generative distillation,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2022, pp. 53–69.

[32] T. Huang, Y. Zhang, M. Zheng, S. You, F. Wang, C. Qian, and C. Xu, “Knowledge difusion for distillation,”Adv. Neural Inf. Process. Syst., vol. 36, pp. 65299–65316, 2023.

[33] Z. Kang, P. Zhang, X. Zhang, J. Sun, and N. Zheng, “Instanceconditional knowledge distillation for object detection,” Adv. Neural Inf. Process. Syst., vol. 34, pp. 16468–16480, 2021.

[34] L. Li, Y. Bao, P. Dong, C. Yang, A. Li, W. Luo, Q. Liu, W. Xue, and Y. Guo, “DetKDS: Knowledge distillation search for object detectors,” in Proc. Int. Conf. Mach. Learn. (ICML), 2024.

[35] Z. Zhou, Y. Shen, S. Shao, L. Gong, and S. Lin, “Rethinking centered kernel alignment in knowledge distillation,” arXiv preprint arXiv:2401.11824, 2024.

[36] Y. Zhang, T. Huang, J. Liu, T. Jiang, K. Cheng, and S. Zhang, “FreeKD: Knowledge distillation via semantic frequency prompt,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 15931–15940.

[37] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” Adv. Neural Inf. Process. Syst., vol. 30, 2017.

[38] X. Wang, R. Girshick, A. Gupta, and K. He, “Non-local neural networks,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 7794–7803.

[39] H. Hu, J. Gu, Z. Zhang, J. Dai, and Y. Wei, “Relation networks for object detection,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 3588–3597.

[40] S. Sun and W. Ohyama, “CanKD: Cross-attention-based nonlocal operation for feature-based knowledge distillation,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2026, pp. 8606–8616.

[41] F. Huo, W. Xu, J. Guo, H. Wang, and S. Guo, “C2KD: Bridging the modality gap for cross-modal knowledge distillation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 16006–16015.

[42] Y. Liu, Z. Jia, and H. Wang, “EmotionKD: A cross-modal knowledge distillation framework for emotion recognition based on physiological signals,” in Proc. 31st ACM Int. Conf. Multimedia (ACM MM), 2023, pp. 6122–6131.

[43] B. Sun and K. Saenko, “Deep CORAL: Correlation alignment for deep domain adaptation,” in Proc. Eur. Conf. Comput. Vis. Workshops (ECCVW), 2016, pp. 443–450.

[44] J. Bruna and S. Mallat, “Invariant scattering convolution networks,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 35, no. 8, pp. 1872–1886, 2013.

[45] T. Williams and R. Li, “Wavelet pooling for convolutional neural networks,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2018.

[46] K. Xu, M. Qin, F. Sun, Y. Wang, Y.-K. Chen, and F. Ren, “Learning in the frequency domain,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 1740– 1749.

[47] C. Pham, V.-A. Nguyen, T. Le, D. Phung, G. Carneiro, and T.-T. Do, “Frequency attention for knowledge distillation,”

in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2024, pp. 2277–2286.

[48] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, “Microsoft COCO: Common objects in context,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2014, pp. 740–755.

[49] J. Liu, Y. Zhang, T. Huang, W. Xu, and R. Yang, “Distilling cross-modal knowledge via feature disentanglement,” in Proc. AAAI Conf. Artif. Intell., vol. 40, no. 28, 2026, pp. 23739–23747, doi: 10.1609/aaai.v40i28.39548.

[50] C.-Y. Wang, H.-Y. M. Liao, Y.-H. Wu, P.-Y. Chen, J.-W. Hsieh, and I.-H. Yeh, “CSPNet: A new backbone that can enhance learning capability of CNN,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), 2020, pp. 390–391.

[51] C.-Y. Wang, H.-Y. M. Liao, and I.-H. Yeh, “Designing network design strategies through gradient path analysis,” arXiv preprint arXiv:2211.04800, 2022.

[52] T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Ré, “FlashAttention: Fast and memory-eficient exact attention with IO-awareness,” Adv. Neural Inf. Process. Syst., vol. 35, pp. 16344–16359, 2022.

[53] T. Dao, “FlashAttention-2: Faster attention with better parallelism and work partitioning,” arXiv preprint arXiv:2307.08691, 2023.

[54] T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, “Feature pyramid networks for object detection,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 2117–2125.

[55] H. Rezatofighi, N. Tsoi, J. Gwak, A. Sadeghian, I. Reid, and S. Savarese, “Generalized intersection over union: A metric and a loss for bounding box regression,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 658– 666.

[56] Z. Zheng, P. Wang, W. Liu, J. Li, R. Ye, and D. Ren, “Distance-IoU loss: Faster and better learning for bounding box regression,” in Proc. AAAI Conf. Artif. Intell., vol. 34, no. 7, 2020, pp. 12993–13000.

[57] D. Zhou, J. Fang, X. Song, C. Guan, J. Yin, Y. Dai, and R. Yang, “IoU loss for 2D/3D object detection,” in Proc. Int. Conf. 3D Vis. (3DV), 2019, pp. 85–94.

[58] X. Li, W. Wang, L. Wu, S. Chen, X. Hu, J. Li, J. Tang, and J. Yang, “Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection,” Adv. Neural Inf. Process. Syst., vol. 33, pp. 21002–21012, 2020.