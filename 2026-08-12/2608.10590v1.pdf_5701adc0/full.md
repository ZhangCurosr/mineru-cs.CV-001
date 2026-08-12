## Rethinking Data Eficiency in Industrial Dense Prediction: Pretraining Coherence, Not Inductive Bias, Determines ViT’s Low-Data Advantage

Haoran Sui, Yaoyuan Jia

• Identified pretraining incoherence as the root cause of ViT’s low-data ineficiency, not transformer inductive bias.

• Formally model the cross-architecture feature gap via engineering-oriented measurable metrics and derive empirical MMD alignment upper bound.

• AlignBlock achieves functional compatibility conversion, enabling ViT backbones to surpass CNNs with as few as 100 samples.

• Rigorous statistical testing and a complete 2 × 2 decomposition reveal stable neck transfer and domain-dependent backbone compatibility.

• Comprehensive deployment analysis and practical decision guidelines for industrial ViT adoption.

# Rethinking Data Eficiency in Industrial Dense Prediction: Pretraining Coherence, Not Inductive Bias, Determines ViT’s Low-Data Advantage

Haoran Sui<sup>a,∗,1</sup>, Yaoyuan Jia<sup>b,1</sup>

<sup>a</sup>ZTE Corporation, Shenzhen, Guangdong Province, P.R. China

<sup>b</sup>The University of Hong Kong, Hong Kong SAR, P.R. China

## A R T I C L E I N F O

Keywords:   
Few-shot learning   
Industrial visual inspection   
Dense prediction   
Vision transformer   
Feature alignment   
Pretraining incoherence

## A BS T RA C T

Vision Transformers (ViTs) are widely believed to require more labeled data than CNNs for industrial dense prediction. Through controlled experiments on four industrial datasets, we show that the dataeficiency gap stems from pretraining incoherence, which refers to the statistical mismatch between ImageNet-pretrained ViT backbones and COCO-pretrained CNN necks, rather than from inherent self-attention deficits. We characterize the cross-architecture feature gap and propose a lightweight AlignBlock family for pyramid-level feature recalibration. Our core finding empirically identifies a data-eficiency frontier: for domain-proximal scenes with ≥ 200 samples, Swin-Graft surpasses YOLOv11x (terminal 703-shot: 0.973 vs 0.956 mAP@50); for domain-distant scenes, CNNs retain advantage (hook 141-shot: 0.900 vs 0.600 mAP@50). Grafted neck weights yield up to 2.5× the mAP of a randomly initialized neck.

## 1. Introduction

## 1.1. Industrial Dense Prediction Under Annotation Scarcity

Manufacturing visual inspection relies heavily on object detection and instance segmentation to validate assembly integrity, surface defects, and electrical component conformity. Unlike COCO public natural datasets with 118K training images, industrial datasets are limited to 150–750 annotated frames due to mandatory domain expert labeling, fixed camera rigs, constrained lighting, and proprietary production environments. Each dense prediction sample requires bounding box or pixel-wise mask labels, drastically elevating annotation overhead. YOLO series CNN detectors (Jocher et al., 2023; Wang et al., 2024) have become de facto industrial standards for their speed-accuracy tradeof and fully homogeneous end-to-end COCO pretraining: backbone, PAN neck, and detection head are jointly optimized under identical detection objectives, forming a tightly matched feature pipeline with consistent BatchNorm statistical priors.

## 1.2. The ViT-CNN Paradox in Low-Data Regimes

Hierarchical Swin Transformers (Liu et al., 2021, 2022a) achieve superior long-range multi-scale modeling on large benchmarks but introduce critical architectural incompatibility when combined with of-the-shelf CNN detection necks. Swin backbones are pre-trained on ImageNet-22K classification with LayerNorm normalization, which enforces global spatial feature statistics per single image. In contrast, YOLO PAN necks are pre-trained for COCO detection with BatchNorm, which relies on stable per-channel batch-wise activation expectations centered at zero. Directly concatenating unmodified Swin backbones to CNN necks creates a heterogeneous feature gap which may partly be attributed to ViT’s innate lack of local convolution inductive bias. The majority of existing literature claims ViTs underperform CNNs with limited labels, yet no systematic work provides quantitative metrics to characterize this crossarchitecture representation dislocation across multi-scale feature pyramids.

## 1.3. The Heterogeneous Feature Gap

We hypothesize that a key missing piece is the heterogeneous feature gap between the pretrained transformer backbone and the convolutional dense prediction head. This gap arises from three practical mismatches widely ignored in existing industrial detection research:

1. Local structural mismatch: Transformer features focus global context via self-attention, lacking fine local texture priors needed for tiny industrial cracks and defects, while CNN necks expect locally coherent feature maps.

2. Feature-statistical mismatch: ViT LayerNorm computes per-image feature statistics, CNN BatchNorm relies on batch-centered channel activations; inconsistent distribution creates information loss between backbone and neck.

3. Optimization mismatch: Freezing ViT backbone blocks loss gradients from adjusting pre-trained features to fit industrial detection targets, limiting model adaptation under limited labels. We analyze this threepart mismatch as the core root of ViT low-shot performance drop in mixed ViT-CNN industrial pipelines.

## 1.4. Research Questions

• RQ1: Can we quantitatively determine the crossover sample threshold where ViT-CNN hybrid pipelines outperform native CNN detectors, and empirically validate that domain distribution proximity, rather than raw sample volume, dominates transfer performance?

• RQ2: Can we decompose the independent performance contributions of pre-trained backbones and grafted neck layers via a standardized 2 × 2 experimental matrix across diverse industrial domains?

• RQ3: Can the proposed lightweight alignment module efectively address the three identified mismatches (spatial, statistical, and optimization), and what empirical upper bound limits its alignment capacity under extreme domain shift?

## 1.5. Proposed Solution

To address these heterogeneous mismatches, we propose AlignBlock, a lightweight plug-in bridging ViT backbones and CNN heads. It simultaneously injects local inductive bias via compact convolutions, aligns LayerNorm feature statistics to BatchNorm expectations, and preserves transformer semantics via residuals. Coupled with positionsensitive insertion and a three-stage progressive training protocol, our framework stabilizes cross-architecture optimization.

## 1.6. Contributions

1. Empirical Finding: We challenge the prevalent assumption that ViTs naturally underperform CNNs with few labels. Controlled industrial trials show the cross-model accuracy gap is primarily driven by pretraining distribution matching rather than merely transformer architecture flaws.

2. Quantitative Decomposition: Using a novel 2×2 decomposition (CNN/ViT × COCO/random neck), we isolate neck pretraining contribution from backbone compatibility, revealing a stable cross-domain neck transfer value while backbone compatibility becomes more critical under domain shift.

3. Lightweight Alignment Toolkit: We design four AlignBlock variants with minimal parameter overhead (< 1%), validated through extensive component ablation and position sensitivity analysis.

4. Industrial Decision Guidance: We provide clear scene-based model selection rules derived from domain proximity and sample-size thresholds, translating experimental findings into actionable engineering practices.

## 1.7. Paper Organization

The remainder of this paper is structured as follows. Section 2 reviews related work with practical diferentiation against existing methods. Section 3 presents the empirical motivation and a diagnostic framework for the heterogeneous feature mismatch. Section 4 details our AlignBlock framework and training strategy. Sections 5 and 6 present experimental setup and results with rigorous statistical testing. Section 7 discusses industrial applicability and practical guidelines, and Section 8 concludes the paper with limitations and future work.

## 2. Related Work

## 2.1. Few-Shot Industrial Visual Inspection

Classical industrial defect detection relied on handcrafted features (Bergmann et al., 2019) before CNN prevalence. Modern pipelines predominantly adopt COCO pretrained YOLO detectors (Bao et al., 2023) via transfer learning, achieving viable accuracy with 100–200 labeled frames. Few-shot object detection (FSOD) meta-learning methods (Wang et al., 2020; Sun et al., 2021) target natural image benchmarks but adopt homogeneous single-architecture backbones and necks, ignoring cross-architecture feature mismatch.

Critical Practical Diference: To the best of our knowledge, most prior FSOD and industrial CNN works have largely overlooked the feature statistic mismatch caused by separately pre-trained ViT and CNN components. Existing manufacturing research rarely designs pyramid-level alignment modules to fix this real-world pipeline issue widely adopted by factory engineers.

## 2.2. ViT for General Dense Prediction

DETR, Deformable DETR, and ViTDet (Carion et al., 2020; Zhu et al., 2021; Li et al., 2022) design unified transformer detection pipelines end-to-end co-pre-trained on COCO, removing pretraining mismatch by design. Swin Transformer V2 and ConvNeXt hybrids embed convolutions inside ViT blocks but maintain homogeneous pretraining without decoupled neck weights. These works primarily evaluate abundant public data and rarely analyze sparse industrial low-data transfer.

Critical Practical Diference: Native unified ViT detectors co-train backbone and head on identical COCO distribution, avoiding the mismatch we study. Our work focuses on the common industrial shortcut of separately loading public Swin and YOLO weights, providing diagnostic metrics and lightweight interface modules to alleviate the resulting performance gap.

## 2.3. Parameter-Eficient ViT Adaptation

LoRA, AdaptFormer, and ViT-Adapter (Hu et al., 2022; Chen et al., 2022, 2023) insert trainable submodules inside ViT encoder layers to fine-tune features without full retraining. ViT-Adapter adds spatial prior branches within Swin blocks but operates strictly inside the transformer, largely omitting addressing backbone-neck boundaries.

Critical Practical Diference: In-block adapters optimize internal ViT representations, but cannot resolve the LN-BN statistical gap at the ViT-CNN interface. Our Align-Block directly targets this cross-architecture boundary, complementing in-block methods.

![](images/6bda438d5737205372275fc880124bca35934665e3f2a2c34772a620bdbc65c9.jpg)  
Figure 1: Overview of the proposed low-shot industrial defect detection framework. The architecture integrates five sequential stages for robust feature representation: Step1 Input Preprocessing: normalization and augmentation of industrial images. Step2 Backbone Extraction: a Swin-Large Transformer backbone for multi-scale hierarchical feature extraction, utilizing a partially frozen training strategy to retain pretrained knowledge. Step3 Heterogeneous Feature Alignment: the proposed position-sensitive AlignBlock module designed to bridge the domain gap between Transformer and CNN features. Step4 Feature Fusion: a pretrained CNN-based PAN neck grafted for multi-stage P2 through P5 feature integration. Step5 Prediction Head: decoupled heads tailored for specific downstream tasks including detection, segmentation, and classification. Bottom Panel: Architectural details of the proposed AlignBlock Family. Four structural variants labeled a to d illustrate the injection of local inductive biases and statistical reshaping via compact convolutions, various normalization layers, and residual shortcuts.

## 2.4. Inductive Bias and Cross-Architecture Feature Fusion

CNNs’ translational equivariance serves as a strong lowdata prior, while vanilla ViTs lack built-in locality (Dosovitskiy et al., 2021). Hybrid CNN-ViT designs (Liu et al., 2022a; Ding et al., 2022) inject convolution layers into ViT blocks to supplement local bias.

Critical Practical Diference: Prior hybrid methods modify transformer internals, whereas we retain of-theshelf Swin and YOLO components and add only lightweight boundary modules. We further reveal an alignment upper bound: even perfect statistical correction cannot fully compensate for ViT’s weak local bias under extreme domain shift.

## 2.5. Cross-Architecture Feature Alignment for ViT-CNN Pipeline

Existing boundary alignment methods provide incomplete solutions at the backbone-neck interface. For instance, BEiT v2 (Peng et al., 2022) lacks multi-scale dense prediction support, while AdaptFormer (Chen et al., 2022) reduces distribution shift but fails to address spatial locality and optimization dynamics.

Unlike these partial approaches, AlignBlock comprehensively resolves three orthogonal mismatches across feature pyramids. It simultaneously integrates spatial locality via 3 × 3 depthwise convolutions, statistical recalibration via GroupNorm, and optimization stability via progressive training. Ablation studies (Section 6.5) confirm this synergy, showing a performance drop of up to 0.055 mAP when any single component is removed. Moreover, our progressive training eliminates the 30% failure rate observed in prior methods.

![](images/a01ba101b1e5648844f701195c0213fbbc073772bad874b960d44203ff74f03b.jpg)  
Figure 2: Overview of training dynamics, alignment mechanisms, and low-shot evaluation metrics. The top panels illustrate a three-phase progressive fine-tuning strategy to stabilize optimization alongside core mechanisms for feature statistical reshaping and residual semantic preservation. The bottom panel validates the framework eficacy. It features a data-eficiency frontier showcasing superior mAP performance under limited sample sizes. Comparative loss and gradient norm trajectories confirm training stability compared to the baseline, complemented by t-SNE and UMAP visualizations demonstrating distinct feature cluster separability.

Distinct from single-architecture designs like ConvNeXt (Liu et al., 2022b), AlignBlock functions exclusively as a cross-architecture adaptor, efectively integrating of-theshelf Swin and YOLO components for industrial few-shot scenarios.

## 3. Empirical Motivation and Problem Formulation

## 3.1. Pilot Study Protocol

To systematically investigate heterogeneous ViT-CNN architectures under data constraints, we designed a controlled pilot study spanning four real-world industrial datasets, three task types, and multiple data-regime levels. All experiments use identical training hyperparameters, data augmentation strategies, and evaluation protocols to ensure fair comparison.

## 3.1.1. Backbone architectures

We compare four backbone configurations:

1. yolo11x (Homogeneous CNN): The native YOLOv11x architecture with CSPNet convolutional backbone,

PAN neck, and task-specific detection head—alljointly pretrained on COCO detection end-to-end.

2. CNN-NoGraft (CNN backbone, random neck): YOLOv11x CSPNet backbone with randomly initialized PAN neck. This homogeneous but untrained neck configuration isolates the contribution of neck pretraining within the CNN family.

3. Swin-Graft (Heterogeneous, pretrained neck): A Swin-Large backbone pretrained on ImageNet-22K, connected to a COCO-pretrained YOLOv11x PAN neck via our AlignBlock modules. The neck weights are grafted from yolo11x, and only alignment blocks are randomly initialized.

4. Swin-NoGraft (Heterogeneous, random neck): Same Swin backbone & AlignBlock pipeline, but neck weights randomly initialized.

## 3.1.2. Training protocol

All models train for 300 epochs with the AdamW optimizer, a batch size of 4, and an input resolution of 640×640. The three-stage schedule was determined through preliminary pilot experiments balancing stability and convergence. The first 150 epochs freeze the backbone, followed by 15 epochs of stabilization and 135 epochs of full fine-tuning with gradual learning rate reduction (0.2× of the initial rate). Standard YOLO data augmentation is applied consistently across all comparison experiments.

## 3.2. Empirical Observations from Controlled Experiments

We condense the main empirical findings into three patterns; detailed numerical tables (Tables 1–4) are provided for reference.

Pattern 1: Performance gain for COCO-similar scenes. On terminal datasets, Swin-Graft consistently outperforms YOLOv11x across all tested sample counts (100–703), with a statistically significant advantage of +2.3% mAP@50 at 200 shots. This strongly indicates that when the heterogeneous mismatch is properly resolved, ViT does not sufer intrinsic low-data drawbacks.

Pattern 2: Performance drop for domain-distant scenes. On overhead hook images $( \mathrm { M M D } ^ { 2 } \ = \ 0 . 8 8$ vs COCO), Swin-Graft lags behind YOLOv11x by 33.3% mAP@50 at 141 shots $( p \texttt { < } 0 . 0 1 )$ ). Sample size alone does not determine transfer performance; visual similarity to COCO is the dominant factor.

Pattern 3: $2 \times 2$ decomposition reveals stable neck transfer and domain-dependent backbone compatibility. From the full experimental matrix, we observe that the gain from grafting a COCO-pretrained neck is stable across domains (ratio yolo11x / CNN-NoGraft ≈ 1.6–1.7). In contrast, the backbone-compatibility gain (CNN-NoGraft / Swin-NoGraft) increases from 1.44× on terminal (domainproximal) to 1.73× on hook (domain-distant). The total heterogeneous penalty decomposes into neck pretraining (69% on terminal, 63% on hook) and backbone compatibility (31% vs 37%), confirming that neck knowledge is cross-domain while backbone compatibility is domain-dependent.

Based on these observations, we formulate four working hypotheses:

• (H1) Domain proximity is associated with relative performance, not raw sample count.

• (H2a) Neck pretraining provides a stable multiplicative gain across domains.

• (H2b) Backbone compatibility becomes more critical under larger domain shift.

• (H3) AlignBlock reduces MMD but has an upper bound under extreme shift.

• (H4) With high COCO similarity and $\geq 2 0 0$ labels, Swin-Graft matches or exceeds CNN.

## 3.3. From Diagnostic Probes to Performance Contribution Decomposition

To bridge the qualitative mismatch description (Section 3.2) and the design of AlignBlock, we first introduce three empirical diagnostic probes that quantify the severity of

spatial, statistical, and optimization discrepancies at the feature/gradient level before alignment:

1. Statistical Gap $G _ { \mathbf { s t a t } }$ (LN vs. BN mismatch):

$$
G _ { \mathrm { s t a t } } = \mathbf { M } \mathbf { M } \mathbf { D } ^ { 2 } ( \mu _ { \mathrm { V i T } } , \sigma _ { \mathrm { V i T } } , \mu _ { \mathrm { C N N } } , \sigma _ { \mathrm { C N N } } )\tag{1}
$$

where $\mu$ and � are channel-wise mean and standard deviation aggregated over the training set.

2. Spatial Locality Gap $G _ { \mathbf { s p a t i a l } } \mathbf { : }$

$$
G _ { \mathrm { s p a t i a l } } = 1 - \frac { 1 } { N } \sum _ { i , j } \mathrm { S i m } _ { \mathrm { l o c a l } } ( F _ { \mathrm { V i T } } ( i , j ) , F _ { \mathrm { C N N } } ( i , j ) )\tag{2}
$$

3. Optimization Gap $G _ { \mathbf { o p t } }$ :

$$
G _ { \mathrm { o p t } } = \| \nabla _ { \mathrm { d e t } } \circ F _ { \mathrm { V i T } } \| _ { 2 } ^ { 2 }\tag{3}
$$

We define the performance contribution $C _ { i }$ of each bottleneck as the mAP@50 drop observed when the corresponding sub-module is individually removed from the full AlignBlock pipeline:

$G _ { \mathbf { s p a t i a l } } \mathrm { : }$ drop when removing the $3 \times 3$ depthwise convolution (locality injection).

$C _ { \mathbf { s t a t } } \mathrm { : }$ drop when removing GroupNorm (statistical recalibration).

$C _ { \mathbf { o p t } } \mathrm { : \Omega }$ drop when disabling the progressive training strategy (optimization dynamics).

To further examine whether these three bottlenecks interact in practice, we define the coupling residual � as:

$$
\eta = \Delta m A P _ { \mathrm { a l l } } - ( C _ { \mathrm { s p a t i a l } } + C _ { \mathrm { s t a t } } + C _ { \mathrm { o p t } } )\tag{4}
$$

where Δ�� $P _ { \mathrm { a l l } }$ is the total mAP drop when all three components are removed simultaneously. If � is substantially smaller than the individual terms, these bottlenecks are empirically largely non-overlapping, supporting AlignBlock’s modular design.

A supplementary analysis of the coupling residual measured in the MMD feature space is provided in the Appendix (Table A2), which shows a consistent trend with the primary mAP-based decomposition reported in Section 6.5.

## 3.4. Empirical Saturation of Alignment under Domain Shift

We model the alignment module as a measurable mapping $A : \mathcal { F } _ { V } \to \mathcal { F } _ { C } .$ In our empirical setting, we observe that AlignBlock reduces the empirical $\mathbf { M } \mathbf { M } \mathbf { D } ^ { \bar { 2 } }$ by 57–63%, but the reduction saturates at a non-zero residual (e.g., 0.18 on terminal and 0.33 on hook at P4). We attribute this empirical saturation to two practical bottlenecks: (1) the limited receptive field of small-kernel convolutions under frozen backbones, which cannot fully compensate for ViT’s global attention bias; and (2) the inherent domain gap between COCO pre-training and industrial data, which imposes a data-dependent performance floor for lightweight adapters.

## 3.4.1. Why MMD cannot be reduced to zero

Even with optimal lightweight alignment, MMD cannot approach zero due to two fundamental constraints under fixed pre-trained weights:

1. The intrinsic local inductive bias gap of self-attention, which cannot be fully compensated by small-kernel convolutions—this is empirically evidenced by the remaining high $\mathbf { M M D } ^ { 2 }$ at P4 on hook (0.33) and the degraded performance when removing the $3 \times 3$ convolution (−0.055 mAP drop), confirming that local bias injection only partially mitigates the gap;

2. Industrial data inevitably diverges from COCO, and Table 3 quantifies the consequence. The raw $\mathbf { M M D } ^ { 2 }$ for hook (0.88) is far larger than for terminal (0.45); after alignment, hook still sits at 0.33 versus terminal’s 0.18. These numbers suggest that domain distance imposes a hard floor on what lightweight alignment can achieve—aligning features only goes so far when the pre-training distribution is fundamentally mismatched.

## 4. Proposed Heterogeneous Feature Alignment Framework

## 4.1. Overall Architecture and Pipeline

Our framework combines frozen or fine-tuned Swin ViT backbone, plug-in AlignBlock modules inserted at P2–P5 pyramid layers, COCO-grafted YOLO PAN neck, and task detection or segmentation heads. Only AlignBlock layers are randomly initialized; backbone and neck pre-trained parameters are fully preserved.

![](images/7b1787cca0baf9eab512f8026ea5fc18ab14112c0df57ba32e33b99f46067899.jpg)  
Figure 3: Architectures of the four AlignBlock variants. (a) SwinAlignBlock with $3 \times 3$ depthwise convolution for local feature injection. (b) Ultra-lightweight SwinSimpleAlign built upon a single 1 × 1 convolution. (c) ConvNeXtAlign-Block with $7 \times 7$ depthwise convolution and multi-scale modulation for micro-defect detection. (d) SpatialAdaptiveAlign adopting dense spatial attention and gating mechanism to filter background interference.

Although AlignBlock reuses standard $1 \times 1$ and $3 ~ \times$ 3 convolution components, its core innovation lies in the multi-objective joint alignment design tailored for industrial heterogeneous pretraining scenarios: simultaneously injecting local spatial prior, correcting LN-BN statistical drift, and stabilizing frozen-to-fine-tuning gradient flow.

## 4.2. Core Module: AlignBlock

## 4.2.1. SwinAlignBlock (Residual Base)

Residual bottleneck with $1 \times 1$ channel projection, middle $3 \times 3$ local mixing convolution, and output projection. Residual branch preserves original ViT feature content:

$$
\begin{array} { r } { \mathrm { S w i n A l i g n B l o c k } ( x ) = \mathrm { C o n v } _ { 1 \times 1 } ^ { C _ { \mathrm { o u t } } } \Big ( \mathrm { S i L U \Big ( G N \Big ( C o n v } _ { 3 \times 3 } ^ { C _ { \mathrm { o u t } } } \Big ( } \\ { \mathrm { S i L U \Big ( G N \Big ( C o n v } _ { 1 \times 1 } ^ { C _ { \mathrm { o u t } } } ( x ) \Big ) \Big ) \Big ) \Big ) \Big ) } \\ { + \mathrm { S h o r t c u t } ( x ) \qquad } \end{array}\tag{5}
$$

## 4.2.2. SwinSimpleAlign (Ultra-Light)

Single $1 \times 1$ conv + normalization + SiLU, ∼ 0.01M parameters per scale.

$$
\mathrm { S w i n S i m p l e A l i g n } ( x ) = \mathrm { S i L U } ( \mathrm { G N } ( \mathrm { C o n v } _ { 1 \times 1 } ( x ) ) )\tag{6}
$$

## 4.2.3. ConvNeXtAlignBlock (Dense Defect Focus)

7 × 7 depthwise conv + GroupNorm + GELU + pointwise transform.

$$
\begin{array} { r } { \mathrm { C o n v N e X t A l i g n B l o c k } ( x ) = \mathrm { C o n v } _ { 1 \times 1 } \big ( \mathrm { G E L U } ( } \\ { \mathrm { G N } ( \mathrm { D W C o n v } _ { 7 \times 7 } ( x ) ) ) \big ) } \end{array}\tag{7}
$$

## 4.2.4. SpatialAdaptiveAlign (Attention-Gated)

Add spatial soft gate to base SwinAlignBlock to suppress irrelevant background activations:

$$
\mathrm { o u t } = \mathrm { m a i n } ( x ) \odot \sigma ( \mathbf { C o n v } _ { 1 \times 1 } ^ { 1 } ( x ) ) + \mathrm { s h o r t c u t } ( x )\tag{8}
$$

## 4.3. Local Inductive Bias Injection

The embedded $3 \times 3$ convolution kernel adds pixel-level locality missing from window-limited Swin self-attention, directly reducing $G _ { \mathrm { s p a t i a l } }$ as defined in Eq. (2).

## 4.4. Feature Statistical Recalibration

GroupNorm inside AlignBlock learns new target-domain statistics without batch-size dependency. Fresh normalization layers discard ImageNet LN bias. For batch size $\leq$ 4, GroupNorm replaces BatchNorm entirely. Quantitative $\mathbf { M M D } ^ { 2 }$ results confirm a 57–63% reduction in $G _ { \mathrm { s t a t } }$

## 4.5. Residual Feature Preservation

The residual shortcut preserves original ViT semantics:

$$
Z _ { \mathrm { C N N } } = Z _ { \mathrm { V i T } } + \Delta _ { \mathrm { a l i g n } } ( Z _ { \mathrm { V i T } } )\tag{9}
$$

## 4.6. Position-Sensitive Insertion Strategy

AlignBlocks are placed after backbone pyramid extraction layers, before PAN fusion. This placement is critical; Table 9 shows that boundary insertion achieves optimal mAP, whereas inserting AlignBlock inside the neck (Section 6.7) degrades performance due to corrupted feature maps.

Table 3  
Table 1  
Table 2  
Overview of industrial datasets. Metrics for hook are reported via 5-fold cross-validation.
<table><tr><td>Dataset</td><td>Task</td><td>Classes</td><td>Train</td><td>Val</td><td>Annotation type</td></tr><tr><td>terminal_cls</td><td>Classification</td><td>2</td><td>755</td><td>188</td><td>ImageFolder labels</td></tr><tr><td>terminal_det</td><td>Detection</td><td>1</td><td>703</td><td>176</td><td>Bounding boxes</td></tr><tr><td>hook</td><td>Detection</td><td>1</td><td>141</td><td>60 (5-fold CV)</td><td>Bounding boxes</td></tr><tr><td>safety-belt</td><td>Segmentation</td><td>2</td><td>178</td><td>45</td><td>Polygon masks</td></tr></table>

## 4.7. Three-Stage Progressive Training

• Phase 1 Freeze (150 epochs): Backbone frozen, only AlignBlock & neck & heads trainable with weak augmentation.

• Phase 2a Stabilize (15 epochs): Full augmentation enabled, reduced learning rate.

• Phase 2b Full Fine-tune (135 epochs): All layers unlocked with hierarchical learning rates. Training failure rate drops from 30% to zero with this protocol, directly reducing $G _ { \mathrm { o p t } }$

## 5. Experimental Setup

## 5.1. Datasets

## 5.1.1. Industrial Datasets and Qualitative Comparison

We evaluate on four real-world industrial datasets. Key characteristics are summarized in Table 1. Qualitatively, terminal images resemble COCO with natural lighting, varied angles, and multi-scale objects; hook images are captured from a low-angle perspective, resulting in densely packed tiny targets under uniform illumination; safety-belt images are intermediate, showing safety ropes on utility poles with outdoor lighting but less viewpoint variation.

All hook metrics are reported with 5-fold cross-validation on the training set (141 images) to ensure statistical reliability, with a validation set of 60 images.

## 5.1.2. Low-Data COCO Subsets

We construct controlled subsets from COCO 2017 comprising the 10 most frequent classes. Balanced training image sampling per class at 1500, 500, and 200 total sample regimes. Validation set fixed at 500 images.

## 5.2. Baselines

1. yolo11x: End-to-end COCO pre-trained homogeneous baseline.

2. CNN-NoGraft: YOLOv11x backbone with random PAN neck.

3. Swin-Graft (Ours): Swin-Large & AlignBlock, grafted COCO pre-trained neck & progressive training.

4. Swin-NoGraft: Same as Swin-Graft but with random neck.

5. DETR: Deformable DETR with ResNet-50 back bone, trained on COCO-10 subsets.

6. DINOv2: ViT-based detector with DINOv2 pretraining, fine-tuned on COCO-10 subsets.

7. ViT-Adapter (reimplemented): In-block adaptation baseline.

Experimental results on four industrial datasets across tasks, data volumes, and pretraining configurations. Metrics are reported as mean ± std over three random seeds (or 5-fold CV for hook). Bold indicates the best per row; ∗ indicates statistically significant diference $( p ~ < ~ 0 . 0 5$ , paired t-test) compared to yolo11x.
<table><tr><td>Dataset</td><td>Task</td><td>Samples</td><td>yolo11x</td><td>CNN-NoGraft</td><td>Swin-NoGraft</td><td>Swin-Graft</td></tr><tr><td colspan="7">mAP@50</td></tr><tr><td> $\mathsf { t e r m i n a l \_ c l s }$ </td><td>CLS</td><td>755</td><td> $0 . 9 4 0 \pm 0 . 0 0 4$ </td><td>0.912</td><td>1</td><td> $\mathbf { 0 . 9 6 0 \pm 0 . 0 0 4 ^ { \circ } }$ </td></tr><tr><td>terminal_det</td><td>DET</td><td>703</td><td> $0 . 9 5 6 \pm 0 . 0 0 4$ </td><td>0.602</td><td>0.785</td><td> $\mathbf { 0 . 9 7 3 \pm 0 . 0 0 5 ^ { \circ } }$ </td></tr><tr><td>terminal_det</td><td>DET</td><td>200</td><td> $0 . 9 2 9 \pm 0 . 0 0 8$ </td><td>0.548</td><td> $0 . 3 8 0 \pm 0 . 0 3 2$ </td><td> $\mathbf { 0 . 9 5 0 \pm 0 . 0 0 7 ^ { \circ } }$ </td></tr><tr><td>terminal_det</td><td>DET</td><td>100</td><td> $0 . 8 9 1 \pm 0 . 0 1 2$ </td><td>一</td><td>0.322 ± 0.038</td><td> $\mathbf { 0 . 9 0 8 \pm 0 . 0 1 4 ^ { \circ } }$ </td></tr><tr><td>hook</td><td>DET</td><td>141</td><td>0.900 ± 0.010</td><td>0.520</td><td> $0 . 3 0 0 ^ { * } \pm 0 . 0 3 5$ </td><td> $0 . 6 0 0 ^ { * } \pm 0 . 0 2 0$ </td></tr><tr><td>hook</td><td>DET</td><td>70</td><td> $\mathbf { 0 . 4 8 3 \pm 0 . 0 2 5 }$ </td><td>0.275</td><td> $0 . 1 9 5 ^ { * } \pm 0 . 0 4 2$ </td><td> $0 . 4 2 5 \pm 0 . 0 2 8$ </td></tr><tr><td>safety-belt</td><td>SEG</td><td>178</td><td> $\mathbf { 0 . 5 0 0 \pm 0 . 0 1 5 }$ </td><td>0.312</td><td> $0 . 2 2 5 ^ { * } \pm 0 . 0 3 0$ </td><td> $0 . 3 8 0 ^ { * } \pm 0 . 0 2 2$ </td></tr><tr><td colspan="7">mAP@50:95</td></tr><tr><td>terminal_det</td><td>DET</td><td>703</td><td> $0 . 6 1 2 \pm 0 . 0 0 5$ </td><td>0.385</td><td>0.452</td><td> $\mathbf { 0 . 6 3 1 \pm 0 . 0 0 6 } ^ { }$ </td></tr><tr><td>terminal_det</td><td>DET</td><td>200</td><td> $0 . 5 7 2 \pm 0 . 0 1 0$ </td><td>0.328</td><td> $0 . 1 5 5 ^ { * } \pm 0 . 0 2 5$ </td><td> $\mathbf { 0 . 5 9 8 \pm 0 . 0 0 9 ^ { \circ } }$ </td></tr><tr><td>terminal_det</td><td>DET</td><td>100</td><td> $\mathbf { 0 . 5 2 5 \pm 0 . 0 1 5 }$ </td><td></td><td> $0 . 1 1 8 ^ { * } \pm 0 . 0 2 8$ </td><td> $0 . 5 4 2 \pm 0 . 0 1 8$ </td></tr><tr><td>hook</td><td>DET</td><td>141</td><td> $\mathbf { 0 . 5 4 2 \pm 0 . 0 1 5 }$ </td><td>0.285</td><td> $0 . 1 1 2 ^ { * } \pm 0 . 0 3 0$ </td><td> $0 . 2 8 4 ^ { * } \pm 0 . 0 2 5$ </td></tr><tr><td>hook</td><td>DET</td><td>70</td><td> $\mathbf { 0 . 2 0 8 \pm 0 . 0 2 5 }$ </td><td>0.112</td><td> $0 . 0 6 8 ^ { * } \pm 0 . 0 3 0$ </td><td> $0 . 1 7 8 \pm 0 . 0 2 8$ </td></tr><tr><td>safety-belt</td><td>SEG</td><td>178</td><td> $\mathbf { 0 . 2 2 1 \pm 0 . 0 1 2 }$ </td><td>0.118</td><td> $0 . 0 7 8 ^ { * } \pm 0 . 0 2 0$ </td><td> $0 . 1 4 8 ^ { * } \pm 0 . 0 1 5$ </td></tr></table>

Domain proximity performance decomposition. ${ \mathsf { M } } { \mathsf { M } } { \mathsf { D } } ^ { 2 }$ values reflect distributional distance to COCO neck’s expected input. ∗ indicates statistically significant diference $\left( p < 0 . 0 5 \right)$
<table><tr><td>Metric</td><td>terminal_det 200</td><td>hook 141</td><td>safety-belt 178</td></tr><tr><td>MMD² to COCO (P4)</td><td>0.45</td><td>0.88</td><td>0.67</td></tr><tr><td>yolo11x mAP@50</td><td> $0 . 9 2 9 \pm 0 . 0 0 8$ </td><td> $0 . 9 0 0 \pm 0 . 0 1 0$ </td><td> $0 . 5 0 0 \pm 0 . 0 1 5$ </td></tr><tr><td>Swin-Graft mAP@50</td><td> $0 . 9 5 0 \pm 0 . 0 0 7 ^ { \ast }$ </td><td> $0 . 6 0 0 ^ { * } \pm 0 . 0 2 0$ </td><td> $0 . 3 8 0 ^ { * } \pm 0 . 0 2 2$ </td></tr><tr><td>∆ (Graft – baseline)</td><td> $+ 2 . 3 \%$ </td><td> $- 3 3 . 3 \%$ </td><td>-24.0%</td></tr><tr><td>Swin-NoGraft mAP@50</td><td> $0 . 3 8 0 ^ { * } \pm 0 . 0 3 2$ </td><td> $0 . 3 0 0 ^ { * } \pm 0 . 0 3 5$ </td><td> $0 . 2 2 5 ^ { * } \pm 0 . 0 3 0$ </td></tr><tr><td>Graft or NoGraft ratio</td><td>2.5×</td><td>2.0x</td><td>1.7×</td></tr></table>

8. BEiT v2: Vector-quantized ViT-CNN alignment method (Peng et al., 2022).

9. AdaptFormer: Lightweight adapter with learnable scaling (Chen et al., 2022).

10. CrossNorm: Cross-architecture normalization.

11. Linear Probe: Single $1 \times 1$ convolution projection baseline.

## 5.3. Implementation Details

PyTorch Ultralytics v8.3, Swin-Large from timm ImageNet-22K, YOLOv11x oficial COCO weights, AdamW optimizer, cosine annealing LR, single 24GB GPU. All statistical tests are paired t-tests over three random seeds (or 5-fold CV for hook).

## 6. Results and Analysis

## 6.1. Main Comparative Results

Table 2 and Table 3 present the primary empirical results across datasets, task types, sample volumes, and pretraining strategies.

## 6.2. 2x2 Performance Gap Decomposition

Table 4 presents the complete $2 \times 2$ matrix isolating neck pretraining value from backbone compatibility. Homogeneous = CNN backbone & CNN neck; Heterogeneous = ViT backbone & CNN neck.

Table 6  
Table 9  
Table 4  
2 × 2 decomposition of pretraining contributions.
<table><tr><td>Dataset</td><td>Samples</td><td>yolo11x (COCO)</td><td>CNN-NoGraft (Random)</td><td>Swin-Graft (COCO)</td><td>Swin-NoGraft (Random)</td></tr><tr><td colspan="6">mAP@50</td></tr><tr><td>terminal_cls</td><td>755</td><td>0.940</td><td>0.912</td><td>0.960*</td><td>一</td></tr><tr><td>terminal_det</td><td>703</td><td>0.956</td><td>0.602</td><td>0.973*</td><td> $0 . 7 8 5 ^ { * }$ </td></tr><tr><td>terminal_det</td><td>200</td><td>0.929</td><td>0.548</td><td>0.950*</td><td>0.380*</td></tr><tr><td>hook</td><td>141</td><td>0.900</td><td>0.520</td><td>0.600*</td><td> $0 . 3 0 0 ^ { * }$ </td></tr><tr><td>safety-belt</td><td>178</td><td>0.500</td><td>0.312</td><td>0.380*</td><td>0.225*</td></tr><tr><td>COCO-10</td><td>1500</td><td>0.618</td><td>0.465</td><td>0.626*</td><td>0.355*</td></tr><tr><td>COCO-10</td><td>200</td><td>0.418</td><td>0.310</td><td>0.425*</td><td>0.242*</td></tr><tr><td colspan="6">mAP@50:95</td></tr><tr><td>terminal_det</td><td>703</td><td>0.612</td><td>0.385</td><td>0.631*</td><td>0.452*</td></tr><tr><td>terminal_det</td><td>200</td><td>0.572</td><td>0.328</td><td>0.598*</td><td> $0 . 1 5 5 ^ { * }$ </td></tr><tr><td>hook</td><td>141</td><td>0.542</td><td>0.285</td><td>0.284*</td><td> ${ 0 . 1 1 2 ^ { * } }$ </td></tr><tr><td>safety-belt</td><td>178</td><td>0.221</td><td>0.118</td><td> $0 . 1 4 8 ^ { * }$ </td><td>0.078*</td></tr><tr><td>COCO-10</td><td>1500</td><td>0.368</td><td>0.279</td><td> $0 . 3 8 1 ^ { * }$ </td><td> $0 . 2 6 5 ^ { * }$ </td></tr><tr><td>COCO-10</td><td>200</td><td>0.230</td><td>0.171</td><td> $0 . 2 3 4 ^ { * }$ </td><td> $0 . 1 6 2 ^ { * }$ </td></tr></table>

Table 5

Performance on low-data COCO-10 subsets. $\mathsf { m A P @ 5 0 }$ (mean $\pm \mathsf { s t d } )$ . ∗ $p < 0 . 0 5$ vs yolo11x.
<table><tr><td>Samples</td><td>yolo11x</td><td>DETR</td><td>DINOv2</td><td>Swin-Graft</td><td>Swin-NoGraft</td></tr><tr><td>1500</td><td> $0 . 6 1 8 \pm 0 . 0 0 4$ </td><td>0.592</td><td>0.602</td><td> $\mathbf { 0 . 6 2 6 \pm 0 . 0 0 5 ^ { \ast } }$ </td><td> $0 . 3 5 5 ^ { \ast } \pm 0 . 0 2 5$ </td></tr><tr><td>500</td><td> $0 . 5 4 2 \pm 0 . 0 0 8$ </td><td>0.518</td><td>0.528</td><td> $\mathbf { 0 . 5 4 8 \pm 0 . 0 0 9 }$ </td><td> $0 . 3 0 5 ^ { \ast } \pm 0 . 0 2 8$ </td></tr><tr><td>200</td><td> $0 . 4 1 8 \pm 0 . 0 1 5$ </td><td>0.395</td><td>0.402</td><td> $\mathbf { 0 . 4 2 5 \pm 0 . 0 1 8 }$ </td><td> $0 . 2 4 2 ^ { * } \pm 0 . 0 3 5$ </td></tr></table>

Backbone capacity and architecture comparison. $\mathsf { m A P @ 5 0 }$ values are reported with standard deviations. ∗ indicates $p <$ 0.05 vs YOLOv11x.
<table><tr><td>Model</td><td>Params (Backbone)</td><td>terminal det 200</td><td>hook 141 (5-fold CV)</td></tr><tr><td>YOLOv11x</td><td>68.2M (full model)</td><td> $0 . 9 2 9 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 9 0 0 \pm 0 . 0 1 0 }$ </td></tr><tr><td>Swin-Tiny + AlignBlock</td><td>28.3M (backbone) 2.8M (ÀlignBlock)</td><td> $0 . 9 2 5 \pm 0 . 0 0 9$ </td><td> $0 . 5 5 4 ^ { * } \pm 0 . 0 2 5$ </td></tr><tr><td>Swin-Large +</td><td>197.0M (backbone)</td><td></td><td></td></tr><tr><td>AlignBlock</td><td>2.8M (AlignBlock)</td><td> $\mathbf { 0 . 9 5 2 ^ { \ast } \pm 0 . 0 0 7 }$ </td><td> $0 . 6 0 5 ^ { \ast } \pm 0 . 0 2 0$ </td></tr></table>

## 6.3. COCO-10 Generalization

Table 5 evaluates cross-domain generalization on natural image subsets, including DETR and DINOv2 baselines.

## 6.4. Backbone Scaling and Architecture Comparison

Table 6 quantifies the impact of backbone capacity and compares Swin-Tiny & AlignBlock, Swin-Large & Align-Block, and YOLOv11x. Despite modest sensitivity to scale (+0.050 mAP), ViT variants fundamentally underperform YOLOv11x by > 0.29 mAP on OOD tasks, identifying pretraining alignment—not capacity—as the binding constraint.

## 6.5. Ablation Studies and Coupling Residual �

Table 7 and 8 validate the performance contribution decomposition and the coupling residual �.

The coupling residual:

$$
\begin{array} { r l } & { \eta = \Delta \mathrm { m A P } _ { \mathrm { a l l } } - ( C _ { \mathrm { s p a t i a l } } + C _ { \mathrm { s t a t } } + C _ { \mathrm { o p t } } ) } \\ & { \quad = 0 . 1 4 8 - ( 0 . 0 5 5 + 0 . 0 4 8 + 0 . 0 4 7 ) } \\ & { \quad = - 0 . 0 0 2 } \end{array}\tag{10}
$$

Component ablation on hook 141-shot (5-fold CV). ✓=enabled, ✗=disabled. mAP@50. ∗ $p < 0 . 0 5$ vs previous row.
<table><tr><td>AlignBlock</td><td>Prog. Train</td><td>Graft Neck</td><td>GN</td><td>mAP@50</td></tr><tr><td>— (Identity)</td><td>x</td><td>x</td><td>x</td><td> $0 . 3 0 0 \pm 0 . 0 3 5$ </td></tr><tr><td>Only Linear (1×1 conv)</td><td>x</td><td>√</td><td>x</td><td> $0 . 3 8 2 \pm 0 . 0 2 8$ </td></tr><tr><td>SwinAlignBlock</td><td>x</td><td>√</td><td>x</td><td> $0 . 4 5 0 ^ { * } \pm 0 . 0 2 2$ </td></tr><tr><td>SwinAlignBlock</td><td>√</td><td>√</td><td>x</td><td> $0 . 5 5 0 ^ { * } \pm 0 . 0 2 0$ </td></tr><tr><td>SwinAlignBlock</td><td>√</td><td>√</td><td>√</td><td> $0 . 6 0 0 ^ { \ast } \pm 0 . 0 2 0$ </td></tr><tr><td>SpatialAdaptive</td><td>√</td><td>√</td><td>√</td><td> $\mathbf { 0 . 6 2 0 \pm 0 . 0 1 8 }$ </td></tr></table>

Additivity validation of the three-gap decomposition on hook 141-shot (5-fold CV). Each row independently removes one component from the full AlignBlock.
<table><tr><td>Removed Component</td><td>Target Gap</td><td>mAP@50</td><td>∆mAP</td></tr><tr><td>None (full AlignBlock)</td><td></td><td>0.600</td><td></td></tr><tr><td>Remove 3×3 Conv</td><td> $G _ { \mathsf { s p a t i a l } }$ </td><td>0.545</td><td> $- 0 . 0 5 5$ </td></tr><tr><td>Remove GroupNorm</td><td> $G _ { \mathrm { s t a t } }$ </td><td>0.552</td><td>-0.048</td></tr><tr><td>Remove Prog. Train (single-stage)</td><td> $G _ { \mathrm { o p t } }$ </td><td>0.553</td><td>-0.047</td></tr><tr><td>Remove all three</td><td> $G _ { \mathrm { s p a t i a l } } + \tilde { \mathsf { G } } _ { \mathrm { s t a t } } ^ { \mathrm { r } } + G _ { \mathrm { o p t } }$ </td><td>0.452</td><td>-0.148</td></tr></table>

Insertion position ablation on hook 141-shot (5-fold CV). ∗ $: p < 0 . 0 5$ vs best.
<table><tr><td>Position</td><td>mAP@50</td></tr><tr><td>After Index, before neck (default boundary)</td><td> $\mathbf { 0 . 6 0 0 \pm 0 . 0 2 0 }$ </td></tr><tr><td>After PAN top-down fusion</td><td> $0 . 5 2 0 ^ { * } \pm 0 . 0 2 5$ </td></tr><tr><td>At every neck skip connection</td><td> $0 . 4 9 0 ^ { * } \pm 0 . 0 3 0$ </td></tr><tr><td>Inside PAN neck (per C2f)</td><td> $0 . 5 4 0 ^ { * } \pm 0 . 0 2 2$ </td></tr></table>

which is only 1.3% of the total drop and well within standard deviation, empirically confirming that the three bottlenecks are largely non-overlapping.

## 6.6. Position Ablation

Table 9 compares four AlignBlock insertion strategies. Boundary placement (after backbone extraction, before PAN fusion) achieves the best mAP@50 of $0 . 6 0 0 \pm 0 . 0 2 0 .$ , significantly outperforming alternatives that insert after fusion or inside neck blocks $( p < 0 . 0 5 )$ .

## 6.7. Comparison with Adaptation Methods

Table 10 compares our approach against internal adapter baselines and cross-architecture boundary alignment methods.

## 6.8. Training Stability

We classify a training run as “failed” if the loss diverges to NaN or the final validation mAP@50 falls below 0.05. Under the single-stage training protocol without stabilization, 30% of runs failed by this criterion. Our three-stage progressive training achieves a 0% failure rate across all tested configurations and reduces the mAP variance across folds by more than twofold on hook.

## 6.9. Multi-Scale Channel Statistical Mismatch

To characterize the feature incompatibility, Figure 4 compares the channel-wise statistical distributions of multiscale features (P2 to P5). Raw Swin outputs, normalized by

Table 12

Comparison with adaptation methods on hook 141-shot (5- fold CV). mAP@50, total trainable parameters, and inference speed are reported. $\ast \ : p < 0 . 0 5$ vs best. AlignBlock alone adds only 2.8M parameters (< 1% of the full Swin-Large & YOLO model).
<table><tr><td>Method</td><td>Trainable Params</td><td>FPS</td><td>mAP@50</td></tr><tr><td>Swin-Graft (Ours, boundary)</td><td>58.6M†</td><td>26.2</td><td> $\mathbf { 0 . 6 0 0 \pm 0 . 0 2 0 }$ </td></tr><tr><td>BEiT v2 (boundary)</td><td>64.2M</td><td>24.8</td><td> $0 . 4 9 2 ^ { * } \pm 0 . 0 2 3$ </td></tr><tr><td>AdaptFormer (in-block)</td><td>59.3M</td><td>25.6</td><td> $0 . 4 7 6 ^ { * } \pm 0 . 0 2 4$ </td></tr><tr><td>CrossNorm</td><td>58.9M</td><td>26.0</td><td> $0 . 4 5 5 ^ { * } \pm 0 . 0 2 8$ </td></tr><tr><td>ViT-Adapter</td><td>8.3M</td><td>27.1</td><td> $0 . 4 8 0 ^ { * } \pm 0 . 0 2 5$ </td></tr><tr><td>LoRA</td><td>2.1M</td><td>27.8</td><td> $0 . 4 2 0 ^ { * } \pm 0 . 0 3 0$ </td></tr><tr><td>Linear Probe (1×1 conv)</td><td>0.8M</td><td>28.0</td><td> $0 . 3 8 2 ^ { * } \pm 0 . 0 2 8$ </td></tr><tr><td>Swin-NoGraft</td><td>58.6M†</td><td>26.5</td><td> $0 . 3 0 0 ^ { * } \pm 0 . 0 3 5$ </td></tr></table>

† Includes AlignBlock (2.8M) & YOLO neck & detection heads. AlignBlock alone adds only 2.8M parameters. The Swin Transformer backbone is frozen during training.

LayerNorm, exhibit pronounced mean and variance discrepancies relative to the BatchNorm-normalized CNN features. AlignBlock addresses this mismatch by reshaping the channel moments through GroupNorm, substantially reducing the statistical distance to the CNN neck’s expected input distribution and efectively bridging the representation gap.

![](images/6760fcd54823233899ce54fdb54618bc14ad1e2d2fa913b559e69faa2d7bb1e2.jpg)  
Figure 4: Channel-level statistical distribution comparison of multi-scale P2–P5 features. (a) Density distributions of channel means. (b) Density distributions of channel vari ances. (c) Log-scale summary of mean channel variance. (d) Statistical gap Gstat (MMD²) between feature distributions. Blue: raw unaligned Swin features; Orange: Swin features aligned via AlignBlock; Green: CNN neck features.

## 6.10. AlignBlock Input-Output Feature Calibration

As shown in Figure 5, raw Swin inputs exhibit high mean activations � and variance �, indicating background clutter. AlignBlock successfully suppresses this background energy as mean values drop near or below zero, while sharply preserving target contours across the Terminal, COCO, and Hook benchmarks.

Quantitatively, feature discrepancies measured by JS Divergence $D _ { \mathrm { J S } }$ and Cosine Dissimilarity 1–cos decrease from shallow stages such as P2 and P3 to reach a minimum at stage P4, descending onto the target convergence threshold marked by the dashed line. This confirms that AlignBlock achieves efective semantic calibration without compromising scale specific representation integrity.

Table 11  
MMD<sup>2</sup> reduction across pyramid scales and datasets (hook values from 5-fold CV).
<table><tr><td>Scale</td><td>terminal (Raw → Aligned) hook (Raw → Aligned)</td><td></td></tr><tr><td>P2</td><td> $0 . 2 1  0 . 0 9 ( - 5 7 \% )$ </td><td> $0 . 3 5  0 . 1 4 ( - 6 0 \% )$ </td></tr><tr><td>P3</td><td> $0 . 2 8  0 . 1 2 \ ( - 5 7 \% )$ </td><td> $0 . 5 2  0 . 2 1 \ ( - 6 0 \% )$ </td></tr><tr><td>P4</td><td> $0 . 4 5  0 . 1 8 ( - 6 0 \% )$ </td><td> $0 . 8 8  0 . 3 3 ( - 6 3 \% )$ </td></tr><tr><td>P5</td><td> $0 . 3 9  0 . 1 6 ( - 5 9 \% )$ </td><td> $0 . 7 4  0 . 2 8$  (-62%)</td></tr></table>

MMD<sup>2</sup> (P4) after alignment for diferent boundary methods on hook 141-shot.
<table><tr><td>Method Aligned MMD² (P4)</td></tr><tr><td>Raw Swin (no align) 0.88</td></tr><tr><td>BEiT v2 0.41</td></tr><tr><td>AdaptFormer 0.45</td></tr><tr><td>CrossNorm 0.48</td></tr><tr><td>AlignBlock (Ours) 0.33</td></tr></table>

![](images/78eb5a44eb010fa156b96df4ad52b2ae82f6a09d8963ab78117cb279ac88f122.jpg)  
Figure 5: Qualitative and quantitative comparison of feature alignment across diferent datasets. (Top) Visualization of feature maps before (“Raw Swin Input”) and after alignment (“Aligned Output”) for (a) Terminal, (b) COCO, and (c) Hook datasets. Mean (�) and standard deviation (�) of feature activations are indicated above each patch. (Bottom) Quantitative evaluation of feature discrepancy across multi-scale feature pyramid levels (P2–P5) using Jensen-Shannon Divergence $( D _ { \mathrm { J S } } ,$ , left) and Cosine Dissimilarity (1–cos, right). The horizontal dashed line denotes the target convergence threshold.

## 6.11. MMD<sup>2</sup> Distribution Distance

Table 11 reports MMD<sup>2</sup> between the pyramid features of each target dataset and the corresponding COCO features, measured before and after alignment at scales P2 through P5. AlignBlock reduces the divergence by 57 to 60 percent on terminal and by 60 to 63 percent on hook, the largest absolute reduction occurring at P4 where the hook value falls from 0.88 to 0.33. The reduction is stable across scales because AlignBlock applies the same parameterisation at every level of the pyramid.

Table 12 compares AlignBlock with three alternative boundary alignment methods at P4 on hook. AlignBlock achieves the smallest residual distance of 0.33, outperforming BEiT v2 (0.41), AdaptFormer (0.45), and RepAdapter (0.48), demonstrating superior cross-architecture alignment efectiveness.

Alignment narrows the gap but does not remove it. After alignment the residual distance on hook remains 0.33 against 0.18 on terminal, preserving the ordering of the two domains, and this ordering matches the ordering of detection accuracy reported in Section 6.1, where Swin-Graft reaches 0.950 mAP@50 on terminal but only 0.600 on hook. Feature level alignment therefore compensates for part of the distributional mismatch, while the residual discrepancy continues to limit transfer on domain distant data.

Q1  
Table 13  
Joint CKA-MMD analysis. CKA values are averaged over terminal and hook (hook from 5-fold CV).
<table><tr><td>Scale</td><td>CKA (Aligned vs CNN)</td><td> ${ \mathsf { M M D } } ^ { 2 }$  Reduction</td></tr><tr><td>P2</td><td>0.20</td><td>58.5%</td></tr><tr><td>P3</td><td>0.18</td><td>58.5%</td></tr><tr><td>P4</td><td>0.10</td><td>61.5%</td></tr><tr><td>P5</td><td>0.09</td><td>60.5%</td></tr></table>

![](images/a08c26a37eaa568327a37240172f14c1d2985b39231001552257123215066e86.jpg)  
Figure 6: $\mathbf { M M D } ^ { 2 }$ between pyramid-level Swin features and COCO-native CNN features before (light bars) and after (dark bars) AlignBlock alignment, across feature pyramid levels P2–P5 for the terminal and hook datasets.

## 6.12. CKA Representation Similarity and Joint Analysis with MMD

Given feature sets $\ b { X } \in \mathbb { R } ^ { n \times p }$ and $Y \in \mathbb { R } ^ { n \times q }$ , Centered Kernel Alignment (CKA) (Kornblith et al., 2019) is defined as:

$$
\operatorname { C K A } ( X , Y ) = { \frac { \operatorname { H S I C } ( X , Y ) } { { \sqrt { \operatorname { H S I C } ( X , X ) \cdot \operatorname { H S I C } ( Y , Y ) } } } }\tag{11}
$$

which measures angular similarity of representations, invari ant to isotropic scaling and rotation.

Maximum Mean Discrepancy (MMD) measures distributional distance:

$$
\begin{array} { r l } & { \mathbf { M M D } ^ { 2 } ( P , Q ) = \mathbb { E } _ { x , x ^ { \prime } \sim P } [ k ( x , x ^ { \prime } ) ] + \mathbb { E } _ { y , y ^ { \prime } \sim Q } [ k ( y , y ^ { \prime } ) ] } \\ & { \phantom { \frac { 1 } { 2 } } - 2 \mathbb { E } _ { x \sim P , y \sim Q } [ k ( x , y ) ] } \end{array}\tag{12}
$$

which quantifies statistical mismatch directly relevant to BatchNorm priors.

![](images/4e8e0e605aa5814a8ee721cac884a5c33a2a4eee5b000ae474dba1da200d58d1.jpg)  
Figure 7: Layer-wise CKA similarity heatmaps before and after AlignBlock integration. The heatmaps visualize the layer-wise feature similarity (L0 to L14) for three diferent scenarios: Terminal, COCO, and Hook. The top row displays the internal representational structure of the baseline architecture (Raw Swin, without AlignBlock), while the bottom row shows the altered feature dynamics after applying the proposed module (Swin & AlignBlock). By comparing the two rows, it is evident that AlignBlock subtly adjusts the hierarchical feature distributions. Instead of forcing identical representations, it performs a functional compatibility conversion, efectively bridging the domain gap and adapting the ViT outputs for the subsequent CNN neck.

By comparing the two rows, it is evident that AlignBlock subtly adjusts the hierarchical feature distributions. Instead of forcing identical representations, it performs a functional compatibility conversion, efectively bridging the domain gap and adapting the ViT outputs for the subsequent CNN neck.

## 6.13. Swin Self-Attention Heatmaps

Figure 8 visualizes the self-attention weight distributions from the deep blocks of the baseline Swin Transformer.

For specific spatial query points marked by green stars, the model exhibits a pronounced global attention bias where weights widely disperse across background regions rather than focusing tightly on local object boundaries. Interestingly, the severity of this dispersion depends heavily on scene complexity.

Q3  
Q1  
![](images/f289f4a78dd5b626727b53af6f47cb015bcd515b79f9e46c2304f5cd690b05c4.jpg)  
Q3  
Figure 8: Evolution of self-attention heatmaps across deep blocks (Block 21–24). Attention distributions are shown for three query locations (�1, �2, �3). The figure compares the attention scattering efect between the Hook dataset

(left panel) and the Terminal dataset (right panel). The progression from left to right within each panel illustrates the increasing global dispersion of attention weights in deeper network stages. Attention scattering occurs to some extent in all scenarios, but the complex background clutter and irregular textures of Hook images make the problem considerably more severe than in the relatively structured Terminal dataset. This observation points to a root cause of the mismatch at the CNN neck: pure Transformer architectures lack an inherent local inductive bias. It therefore motivates the 3×3 convolutions in AlignBlock, which re-establish local spatial consistency and suppress irrelevant background noise before the features are fed into the PAN neck.

The progression from left to right within each panel illustrates the increasing global dispersion of attention weights in deeper network stages. Attention scattering occurs to some extent in all scenarios, but the complex background clutter and irregular textures of Hook images make the problem considerably more severe than in the relatively structured Terminal dataset. This observation points to a root cause of the mismatch at the CNN neck: pure Transformer architectures lack an inherent local inductive bias. It therefore motivates the 3 × 3 convolutions in AlignBlock, which reestablish local spatial consistency and suppress irrelevant background noise before the features are fed into the PAN neck.

## 6.14. Activation Heatmap Comparison

![](images/b048b3a5f01fa7f7344e2bf790f7b94e53e7a146fc39c1a3efc7d0abef6bc70b.jpg)  
Figure 9: Feature activation map comparisons demonstrating the background suppression and target-focusing capabilities of AlignBlock. Results are shown for (a) Terminal, (b) COCO, and (c) Hook datasets across four test instances (Samples 1–4). Within each dataset, columns from left to right represent: standard CNN, baseline Swin(no Align), and Swin+Alignblock. High-activation regions (yellow or red) indicate that Swin(no Align) sufers from difuse, background-scattered responses due to global attention bias. Incorporating AlignBlock filters out irrelevant background context and sharpens feature localization directly over target regions, yielding significantly higher spatial specificity than both CNN and unaligned Swin baselines.

High-activation regions (yellow or red) indicate that Swin(no Align) sufers from difuse, background-scattered responses due to global attention bias. Incorporating Align-Block filters out irrelevant background context and sharpens feature localization directly over target regions, yielding significantly higher spatial specificity than both CNN and unaligned Swin baselines.

## 6.15. Foreground-Background Feature Separability

t-SNE and UMAP clustering jointly reveal the crossdomain separability gap. While CNN exhibits more clearly separated clusters in the visualizations, Swin-Graft achieves higher detection mAP. We attribute this counter-intuitive result to ViT’s global receptive field, which facilitates boundingbox regression and partially compensates for its slightly less discriminative classification boundaries (see Section 6.12).

![](images/88f8decc2fc9698174f2bc5239e39c053bf031362e42027af13e2f7aeb88262d.jpg)  
Figure 10: Foreground-background feature separability.

## 7. Industrial Applicability and Discussion

## 7.1. ViT-Suitable Scene Criteria

ViT & graft CNN pipeline is recommended when:

1. images share COCO visual traits;

2. task is detection or classification (not segmentation with < 200 labels).

## 7.2. Sample-Size Crossover and the Limits of Capacity

The eficacy of domain-aligned grafting exhibits a bimodal sensitivity to target distribution characteristics (Tables 2 and 4). On the distribution-proximal Terminal set, the Swin-Grafted transformer maintains robust data eficiency even under extreme scarcity. With only 100 samples, it achieves $0 . 9 0 8 \pm 0 . 0 1 4$ mAP@50, marginally surpassing the YOLOv11x baseline $( 0 . 8 9 1 \pm 0 . 0 1 2 )$ . At 200 samples, Swin-Graft not only closes this gap but reaches statistical parity with the CNN benchmark $( 0 . 9 5 0 \pm 0 . 0 0 7$ vs. $0 . 9 2 9 \pm 0 . 0 0 8 )$ . This confirms that, under favorable domain conditions, feature alignment efectively compensates for architectural inductive bias.

However, this advantage collapses under severe distribution shift on the Hook dataset $( N ~ = ~ 1 4 1 )$ . Here, convolutional priors exhibit robustness that neither increased parameterization nor alignment modules can overcome. YOLOv11x achieves $0 . 9 0 0 { \scriptstyle \pm 0 . 0 1 0 \mathrm { m A P } \ @ 5 0 }$ , leading Swin-Graft $( 0 . 6 0 0 { \pm } 0 . 0 2 0 )$ by 0.300 points—a margin that remains substantial at mAP@50:95 (0.542±0.015 vs. 0.284±0.025).

These results suggest that model capacity is not a viable recovery mechanism for domain shift. Even the highcapacity Swin-Graft, equipped with explicit distributionaware modules, remains below 0.61 mAP@50 against a 0.900 YOLO baseline. In other words, pretraining distribution coherence dictates the upper bound of generalization; architectural scaling alone cannot compensate for a fundamental mismatch between source and target visual manifolds.

Table 14  
Deployment eficiency on Jetson Xavier NX.
<table><tr><td>Method</td><td>FPS</td><td>Memory (GB)</td><td>Model Size (MB)</td></tr><tr><td>yolo11x</td><td>28.5</td><td>1.8</td><td>112</td></tr><tr><td>Swin-Graft (Ours)</td><td>18</td><td>2.8</td><td>258</td></tr><tr><td>Swin-Graft (Swin-Tiny)</td><td>22</td><td>2.1</td><td>141</td></tr></table>

Table 15

Industrial pipeline selection guidelines with MMD<sup>2</sup> thresholds.
<table><tr><td>Scenario</td><td>Recommendation MMD{2</td><td>Threshold</td><td>Rationale</td></tr><tr><td>COCO-similar, &gt;200 samples</td><td>Swin-Graft</td><td>&lt; 0.65</td><td>+1.8-2.3% mAP@50 gain (significant)</td></tr><tr><td>COCO-similar, 100–200 samples</td><td>Swin-Graft</td><td>&lt; 0.65</td><td>Advantage holds; neck pretraining critical</td></tr><tr><td>Overhead tiny targets (any)</td><td>YOLOv11x</td><td>&gt; 0.8</td><td>CNN retains large advantage (significant)</td></tr><tr><td>Segmentation, &lt;200 samples</td><td>YOLOv11x</td><td>any</td><td>Mask head amplifies mismatch</td></tr><tr><td>Extreme scarcity (&lt;70)</td><td>Either</td><td>any</td><td>Gap negligible (not significant)</td></tr></table>

## 7.3. Deployment Eficiency

Table 14 summarizes computational eficiency on NVIDIA Jetson Xavier NX.

## 7.4. Model Selection Guidelines

Table 15 translates our experimental findings into quantitative rules, including MMD<sup>2</sup> thresholds for automatic decision support. In the boundary alignment landscape, our method uniquely integrates spatial, statistical, and optimization alignment within a lightweight boundary module, offering the best trade-of between accuracy and overhead for industrial few-shot scenarios.

## 8. Limitations and Future Work

## 8.1. Methodological Limitations

1. The current AlignBlock design assumes a fixed pyramid hierarchy; dynamic insertion strategies may further improve domain adaptation.

2. This work mainly validates Swin hierarchical ViT; extending AlignBlock to DeiT, ConvNeXt and plain ViT will be conducted in future industrial multi-scenario experiments.

3. The additive mismatch decomposition is heuristic; formal interaction terms require additional controlled experiments, though our additivity test (Table 8) shows strong empirical support, and the newly introduced � coupled analysis (Table 13) confirms that coupling is minimal.

4. The hook dataset’s validation set contains 60 images, which provides reasonable statistical power; however, a larger validation set would still be beneficial for more robust conclusions.

## 8.2. Dataset and Task Limitations

1. Industrial datasets are single or dual class; multi-class complex assembly lines remain untested.

2. Few-shot experiments rely on random subsampling; cross-session generalization in production lines is not evaluated.

## 8.3. Potential Negative Societal Impact

While our work aims to improve industrial safety, overreliance on automated inspection without human oversight

Table A1  
Hyperparameter configuration.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Base LR (Phase 1)</td><td>0.01</td></tr><tr><td>Base LR (Full)</td><td>0.0005</td></tr><tr><td>Phase LR decay</td><td>0.2× base</td></tr><tr><td>Batch size</td><td>4</td></tr><tr><td>Input resolution</td><td>640×640</td></tr><tr><td>Epochs</td><td>300(150+15+135)</td></tr><tr><td>Weight decay</td><td>0.0005</td></tr><tr><td>Warmup epochs</td><td>3</td></tr><tr><td>LR schedule</td><td>Cosine annealing</td></tr><tr><td>Gradient clipping</td><td>max norm = 10.0</td></tr><tr><td>Augmentation GN groups</td><td>Mosaic 1.0, Mixup 0.1, Copy-Paste 0.3, HSV, Affine min(32, C)</td></tr></table>

## Table A2

Supplementary analysis of the coupling residual $\varepsilon _ { \mathsf { c o u p l e d } }$ measured in MMD<sup>2</sup> feature space across datasets and pyramid scales. Values are averaged over three random seeds. The percentage in parentheses indicates the proportion of total divergence $D ( A ( p _ { V } ) , p _ { C } )$ attributed to coupling. This table serves as a consistency check with the mAP-based decomposition in Section 6.5.

<table><tr><td>Dataset</td><td>P2</td><td>P3</td><td>P4</td><td>P5</td></tr><tr><td>terminal</td><td>0.023 (5.2%)</td><td>0.031 (6.8%)</td><td>0.039 (7.1%)</td><td>0.027 (5.9%)</td></tr><tr><td>hook</td><td>0.035 (7.3%)</td><td>0.041 (7.9%)</td><td>0.038 (6.2%)</td><td>0.033 (6.5%)</td></tr><tr><td>safety-belt</td><td>0.028 (6.1%)</td><td>0.032 (6.5%)</td><td>0.036 (7.0%)</td><td>0.029 (6.2%)</td></tr></table>

could miss rare critical defects. In deployment, model output should be combined with human inspectors for rare tiny defects under extreme domain shift, avoiding fully automated misdetection. Deployment must include human-in-the-loop protocols.

## 8.4. Future Work

1. Quantitative domain distance metric (e.g., FID) for a priori model selection.

2. Self-supervised pretraining (He et al., 2022; Zhou et al., 2022; Oquab et al., 2023) to reduce COCO or ImageNet dependency.

3. Edge-optimized AlignBlock with quantization for embedded deployment.

4. Expanding the hook validation set to at least 80 images for more robust statistical conclusions.

5. Develop integer quantization pipeline for AlignBlock to further reduce embedded GPU memory consumption.

## 9. Conclusion

Our results show that ViTs are not universally datahungry. Performance in ViT-CNN hybrids is governed by pretraining coherence: when domain similarity to COCO is high, Swin-Graft exceeds YOLOv11x with as few as 100 samples (statistically significant); under large domain shift, CNNs retain a clear advantage. A rigorous 2 × 2 decomposition and additive gap analysis further reveal that neck pretraining value is stable across domains, while backbone compatibility becomes the critical factor under shift. Align-Block provides functional compatibility conversion—not exact representation-space alignment—enabling the CNN neck to process ViT features efectively. The practical implication is straightforward: for deploying ViTs in data-scarce industrial environments, pretraining distribution alignment should carry more weight than architectural choice or model scale.

## References

Ba, J.L., Kiros, J.R., Hinton, G.E., 2016. Layer normalization. arXiv preprint arXiv:1607.06450.

Bao, Y., Song, K., Liu, J., Wang, Y., Yan, Y., Yu, H., Li, X., 2023. ArtificiallyR2U-Net: A novel deep learning approach for surface defect detection. Expert Systems with Applications 211, 118673.

Bergmann, P., Fauser, M., Sattlegger, D., Steger, C., 2019. MVTec AD—A comprehensive real-world dataset for unsupervised anomaly detection. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9592-9600.

Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S., 2020. End-to-end object detection with transformers. In: European Conference on Computer Vision (ECCV). Springer, Cham, pp. 213-229.

Chen, S., Ge, C., Tong, Z., Wang, J., Song, Y., Wang, J., Luo, P., 2022. AdaptFormer: Adapting vision transformers for scalable visual recognition. In: Advances in Neural Information Processing Systems (NeurIPS), vol. 35, pp. 16664-16678.

Chen, Z., Duan, Y., Wang, W., He, J., Lu, T., Dai, J., Qiao, Y., 2023. Vision transformer adapter for dense predictions. In: International Conference on Learning Representations (ICLR).

Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L., 2009. ImageNet: A large-scale hierarchical image database. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 248-255.

Ding, X., Zhang, X., Han, J., Ding, G., 2022. Scaling up your kernels to 31×31: Revisiting large kernel design in CNNs. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11963-11975.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N., 2021. An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (ICLR).

He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R., 2022. Masked autoencoders are scalable vision learners. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16000-16009.

Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., 2022. LoRA: Low-rank adaptation of large language models. In: International Conference on Learning Representations (ICLR).

Iofe, S., Szegedy, C., 2015. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In: International Conference on Machine Learning (ICML). PMLR, vol. 37, pp. 448-456.

Jia, M., Tang, L., Chen, B.C., Cardie, C., Belongie, S., Hariharan, B., Lim, S.N., 2022. Visual prompt tuning. In: European Conference on Computer Vision (ECCV). Springer, Cham, pp. 709-727.

Jocher, G., Chaurasia, A., Qiu, J., 2023. Ultralytics YOLO11. GitHub repository. https://github.com/ultralytics/ultralytics (accessed 10 January 2024).

Kornblith, S., Norouzi, M., Lee, H., Hinton, G., 2019. Similarity of neural network representations revisited. In: International Conference on Machine Learning (ICML). PMLR, vol. 97, pp. 3519-3529.

Li, Y., Mao, H., Girshick, R., He, K., 2022. Exploring plain vision transformer backbones for object detection. In: European Conference on Computer Vision (ECCV). Springer, Cham, pp. 280-296.

Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L., 2014. Microsoft COCO: Common objects in context. In: European Conference on Computer Vision (ECCV). Springer, Cham, pp. 740-755.

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B., 2021. Swin Transformer: Hierarchical vision transformer using shifted windows. In: IEEE/CVF International Conference on Computer Vision (ICCV), pp. 10012-10022.

Liu, Z., Hu, H., Lin, Y., Yao, Z., Xie, Z., Wei, Y., Ning, J., Cao, Y., Zhang, Z., Dong, L., Wei, F., Guo, B., 2022a. Swin Transformer V2: Scaling up capacity and resolution. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12009-12019.

Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S., 2022b. A ConvNet for the 2020s. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11976-11986.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.Y., Li, S.W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P., 2023. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research.

Peng, Z., Dong, L., Bao, H., Ye, Q., Wei, F., 2022. BEiT v2: Masked image modeling with vector-quantized visual tokenizers. arXiv preprint arXiv:2208.06366.

Sun, B., Li, B., Cai, S., Yuan, Y., Zhang, C., 2021. FSCE: Few-shot object detection via contrastive proposal encoding. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7352-7362.

Wang, X., Huang, T.E., Darrell, T., Gonzalez, J.E., Yu, F., 2020. Frustratingly simple few-shot object detection. In: International Conference on Machine Learning (ICML). PMLR, vol. 119, pp. 9919-9928.

Wang, C.Y., Yeh, I.H., Liao, H.Y.M., 2024. YOLOv9: Learning what you want to learn using programmable gradient information. In: European Conference on Computer Vision (ECCV). Springer, Cham, pp. 1-17.

Wightman, R., 2019. PyTorch Image Models. GitHub repository. https://github.com/rwightman/pytorch-image-models (accessed 15 January 2024).

Wu, Y., He, K., 2018. Group normalization. In: European Conference on Computer Vision (ECCV). Springer, Cham, pp. 3-19.

Yang, L., Zhang, R.Y., Li, L., Xie, X., 2023. RepAdapter: A simple and efective adapter for vision transformers. In: IEEE/CVF International Conference on Computer Vision (ICCV), pp. 16697-16706.

Zhang, G., Luo, Z., Cui, K., Lu, S., 2024. Meta-DETR: Image-level fewshot detection with inter-class correlation exploitation. IEEE Transactions on Pattern Analysis and Machine Intelligence 46(5), 3466-3482.

Zhang, H., Li, F., Liu, S., Zhang, L., Su, H., Zhu, J., Ni, L.M., Shum, H.Y., 2022. DINO: DETR with improved denoising anchor boxes for end-to-end object detection. In: International Conference on Learning Representations (ICLR).

Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., Kong, T., 2022. iBOT: Image BERT pre-training with online tokenizer. In: International Conference on Learning Representations (ICLR).

Zhu, X., Su, W., Lu, L., Li, B., Wang, X., Dai, J., 2021. Deformable DETR: Deformable transformers for end-to-end object detection. In: International Conference on Learning Representations (ICLR).

## Author Statement

Haoran Sui: Conceptualization (lead), Methodology (lead), Software (equal), Validation (equal), Formal analysis (lead), Investigation (equal), Data Curation (equal), Writing – original draft (lead), Writing – review & editing (equal), Visualization (supporting), Supervision (lead), Project administration (lead).

Yaoyuan Jia: Conceptualization (supporting), Methodology (supporting), Software (equal), Validation (equal), Investigation (equal), Data Curation (equal), Writing – original draft (supporting), Writing – review & editing (equal), Visualization (lead).