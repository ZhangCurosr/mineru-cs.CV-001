# Generative Semantic Segmentation via an Observable Semantic-Image Interface and Hierarchical Generator Evidence Alignment

Weize Cai<sup>1,†</sup> Yongqi Dong<sup>1,2,†,∗</sup> Zhida Shao<sup>1,†</sup> Zixin Fu<sup>3,†</sup>

<sup>1</sup>RWTH Aachen University, Aachen, Germany

<sup>2</sup>Delft University of Technology, Delft, The Netherlands

<sup>3</sup>Chang’an University, Xi’an, China

<sup>†</sup>Equal contribution. <sup>∗</sup>Corresponding author: yongqi.dong@rwth-aachen.de

## Abstract

Generative semantic segmentation exposes structured predictions as images, but direct color decoding is susceptible to color drift and boundary mixing, whereas latent-feature decoders that predict a separate output distribution may relegate the rendered image to an intermediate visualization. We present Semantic Prism, a conditional semantic-image generation-and-refinement framework with deterministic inference. A difusion-distilled one-step generator renders a semantic RGB image; per-pixel distances from the rendered colors to a fixed class-color codebook define an explicit probabilistic interface. Hierarchical Generator Evidence Alignment (HGEA) spatially aligns multi-level generator features and uses a zero-initialized output projection to predict an additive residual in the interface logit space, retaining the imagedefined interface as the reference for the final distribution. The interface and refined distributions further enable Contextual Interface–Hierarchy Disagreement (C-IHD), a fixed readout for ranking remaining pixel errors without an auxiliary predictor or additional forward pass. On the 500-image Cityscapes validation set, Semantic Prism achieves 72.07% mean intersection over union (mIoU), 11.39 mIoU points above direct-interface decoding, with 0.41% expected calibration error (ECE). Matched-capacity ablations over three seeds support the benefit of jointly aligned multi-level evidence. A separately trained model attains 62.22% mIoU on BDD100K, while the Cityscapes-trained model reaches 46.89% mIoU under source-frozen transfer to the Adverse Conditions Dataset with Correspondences (ACDC), without target-domain adaptation. Across all three datasets, C-IHD consistently improves the area under the precision–recall curve (AUPR) for pixelerror ranking over maximum softmax probability (MSP) on the same segmentation predictions; on ACDC, it raises AUPR from 0.6580 to 0.7557.

## Introduction

Semantic segmentation supports scene understanding by assigning a semantic class to every image pixel. Modern discriminative architectures achieve strong accuracy by mapping latent image representations directly to class logits (Xie et al. 2021; Cheng et al. 2022). Generative formulations instead cast dense prediction as the generation of a label map or semantic image (Chen et al. 2023; Lai et al. 2023; Ji et al. 2023), exposing a visible structured output that can be independently decoded and evaluated. However, preserving the semantic image as an explicit, independently evaluable predictive interface while achieving fine-grained spatial accuracy remains challenging.

Direct color decoding maintains a transparent relationship between the rendered image and the prediction, but color drift, boundary mixing, and blur can corrupt class assignments, particularly at semantic transitions and thin structures. Conversely, latent-feature decoders can recover local detail, but when they produce the final distribution through a path separate from the image-defined interface, the rendered semantic image may serve only as an intermediate visualization. Together, these limitations expose a trade-of between preserving the semantic image as an explicit independently evaluable interface and recovering fine-grained spatial accuracy. This motivates a formulation in which the semantic image defines a per-pixel class distribution that serves as the reference for the final prediction, while aligned multilevel generator evidence contributes additive corrections to its logits rather than establishing a separate prediction path.

We realize this design with Semantic Prism, a one-step semantic-image generation-and-refinement framework with deterministic inference. A difusion-distilled image translator (Parmar et al. 2024; Sauer et al. 2023) renders a semantic RGB image. At each pixel, distances from the rendered RGB value to a fixed class-color codebook define a full classprobability distribution, forming an explicit probabilistic interface. We call this interface observable in an operational sense: the pre-refinement distribution, including its top-1 label, maximum class probability, and pairwise class log-odds, is recoverable from the rendered image using the fixed codebook decoder, without access to latent generator features.

Hierarchical Generator Evidence Alignment (HGEA) complements this interface by spatially aligning features from three generator levels. Through a zero-initialized output projection, it predicts additive corrections to the interface logits rather than a separate output distribution. The imagedefined interface therefore remains an explicit probabilistic reference for the final prediction, while aligned hierarchical evidence supplies fine-grained spatial cues for correcting boundary and thin-structure errors. As an auxiliary readout, Contextual Interface–Hierarchy Disagreement (C-IHD) combines pointwise and local uncertainty with disagreement between the interface and hierarchy-refined distributions to rank remaining pixel errors. It leaves the segmentation unchanged and requires neither a trainable error predictor nor an additional model forward pass.

We evaluate Semantic Prism in three complementary settings: in-domain evaluation on Cityscapes (Cordts et al. 2016), independent in-domain training and evaluation on BDD100K (Yu et al. 2020), and source-frozen transfer from Cityscapes to the Adverse Conditions Dataset with Correspondences (ACDC) (Sakaridis, Dai, and Gool 2021). These experiments assess segmentation accuracy, boundary quality, calibration, and pixel-error ranking.

Our contributions are threefold: (1) We formulate an observable semantic-image interface whose fixed distancebased codebook decoder maps rendered RGB values to full class distributions, with top-1 labels directly recoverable from the rendered image and pairwise class log-odds available in closed form; (2) We propose HGEA, which aligns multi-level generator features and additively refines interface logits. On Cityscapes, it improves mean intersection over union (mIoU) by 11.39 points over direct-interface decoding, reaching 72.07%; matched-capacity three-seed controls support gains from joint multi-level alignment over hierarchyfree and single-level refinement; (3) We introduce C-IHD, a fixed readout that reuses the interface and refined distributions to improve pixel-error ranking, measured by the area under the precision–recall curve (AUPR), relative to a maximum softmax probability (MSP) on the same underlying predictions in each of the three evaluation settings.

## Related Work

Discriminative semantic segmentation. Modern segmentation-based predictors combine local detail, multiscale context, and global interaction through convolutional decoders (Chen et al. 2018), hierarchical transformers (Xie et al. 2021), mask classification (Cheng et al. 2022), and state-space models (Fu, Lou, and Yu 2025). Domaingeneralized methods explicitly target unseen appearance shifts through feature regularization or generative guidance (Choi et al. 2021; Li et al. 2025). In both conventional and domain-generalized semantic segmentation systems, final predictions are produced by learned heads operating on latent features. We study a complementary output construction in which the rendered semantic image, together with a fixed decoder, defines a pre-refinement probability field that can be evaluated without access to latent features.

Generative and image-form dense prediction. An alternative line of work represents dense predictions directly in image space. GSS encodes segmentation masks as RGB “maskige” images (Chen et al. 2023), SegGPT performs incontext coloring (Wang et al. 2023b), and UniGS (Qi et al. 2024) and CAM-Seg (Ahmed et al. 2025) introduce locationaware or continuous semantic-image representations. Other generative approaches employ difusion, flow, or autoregressive processes to generate or refine dense label maps (Ji et al. 2023; Lai et al. 2023; Wang et al. 2023a, 2024; Caetano et al. 2026; Yang et al. 2026; Deng et al. 2025), while generalist models formulate diverse perception tasks as conditional image generation (Geng et al. 2024; Zhao et al. 2025). Within this generalist line, Vision Banana is especially relevant to our setting because it represents semantic segmentation as an RGB image using prompt-specified class colors (Gabeur et al. 2026). Collectively, these works demonstrate the viability of image-form dense prediction and, in some cases, color-based label recovery. Semantic Prism difers by using fixed codebook distances to define a full per-pixel class distribution from the rendered RGB image and by constraining hierarchical refinement to additive corrections in the resulting interface logit space. The novelty lies in this observable probabilistic interface and same-space refinement, rather than in RGB output alone.

Generator representations and inspectable interfaces. Beyond image-form outputs, prior work has explored internal generative representations and structured intermediate interfaces. VPD (Zhao et al. 2023) and ODISE (Xu et al. 2023) couple difusion features with learned segmentation decoders, while training-free methods (Meng et al. 2026) aggregate internal features and attention maps. Concept and semantic bottlenecks expose semantically aligned intermediate variables (Koh et al. 2020; Losch, Fritz, and Schiele 2021), whereas prototype segmentors relate predictions to learned examples or parts (Sacha et al. 2023; Porta et al. 2025). Our setting difers: the rendered semantic image serves as the interface and a fixed class-color decoder maps it to an independently evaluable pre-refinement distribution. HGEA uses aligned generator features to additively correct the interface logits rather than to predict a separate output distribution.

Predictive reliability and error ranking. In addition to segmentation accuracy, predictive reliability concerns both calibration and the ranking ofprediction errors. Expected calibration error (ECE) summarizes the discrepancy between predictive confidence and empirical accuracy (Guo et al. 2017), whereas the complement of maximum softmax probability (MSP) provides a standard deterministic score for ranking errors (Hendrycks and Gimpel 2017). Dense calibration methods learn pixelwise or multivariate confidence corrections (Ding et al. 2021; Wang, Gong, and Wang 2023; Küppers et al. 2022), while neighbor-aware objectives incorporate local spatial structure during training (Murugesan et al. 2025). Failure-localization networks introduce auxiliary predictors (Rahman et al. 2022; Kwon and Kwak 2022), whereas segment- and image-level methods aggregate uncertainty for quality estimation (Rottmann et al. 2020; Guarino et al. 2026). Selective prediction evaluates such rankings through the risk–coverage trade-of (Geifman and El-Yaniv 2017). The proposed C-IHD instead targets pixel-error ranking on fixed predictions, requiring neither an auxiliary predictor nor an additional forward pass.

## Method

## Overview

Given an input image $\mathbf { x } \in \mathbb { R } ^ { H \times W \times 3 }$ on the pixel lattice $\Omega = \{ 1 , \dots , \overset { \cdot } { H } \} \times \overset { \cdot } { \{ 1 , \dots , W \} }$ , a one-step conditional generator deterministically renders a semantic RGB image s and exposes intermediate feature maps $\mathcal { H } _ { \mathbf { x } } = \{ \mathbf { h } _ { \mathbf { x } } ^ { \ell } \} _ { \ell \in \mathcal { I } }$ , where $\mathcal { I }$ indexes the selected generator levels and $\mathbf { h } _ { \mathbf { x } } ^ { \ell }$ denotes the feature map extracted at level ℓ.

Let $\mathcal { C } = \{ { \mathbf { c } _ { k } } \} _ { k = 1 } ^ { K }$ be a fixed high-separation class-color codebook, where $K$ is the number of semantic classes and $\mathbf { c } _ { k } \in \mathbb { R } ^ { 3 }$ is the RGB prototype assigned per class k. Semantic Prism forms two coupled distributions. A fixed distancebased decoder maps the rendered semantic image to an interface distribution, while HGEA uses the feature hierarchy to predict an additive residual in the same logit space.

Applied pointwise for each pixel u over Ω, the fixed decoder first maps the rendered semantic image to the interface distribution, after which HGEA applies a hierarchy-derived residual in the same class-logit space:

$$
p _ { I } ( u ) = \Pi _ { \mathcal { C } } ^ { ( \tau _ { I } ) } ( \mathbf { s } ( u ) ) = \mathrm { s o f t m a x } ( \mathbf { z } ^ { I } ( u ) ) ,\tag{1}
$$

$$
\begin{array} { r } { p _ { H } \big ( u \big ) = \mathrm { s o f t m a x } \big ( \mathbf { z } ^ { I } ( u ) + \Delta \mathbf { z } ^ { H } ( u ) \big ) . } \end{array}\tag{2}
$$

Here $\Pi _ { c } ^ { ( \tau _ { I } ) }$ denotes the fixed distance-based codebook decoder, $\check { \mathbf { z } ^ { I } }$ denotes the prototype-distance logits underlying $p _ { I }$ , and $\Delta \mathbf { z } ^ { H }$ denotes the hierarchy-derived residual. The rendered image and fixed decoder therefore determine the interface distribution and its pairwise class log-odds, while HGEA updates the same logit parameterization rather than producing the final distribution through a separate latentfeature path. For each pixel $u \in \bar { \Omega } .$ , the final label is $\hat { y } ( u ) = \arg \operatorname* { m a x } _ { k } p _ { H , k } ( u )$ . MSP and C-IHD are computed downstream of this prediction path and leave both refined distribution $p _ { H }$ and the predicted label map yˆ unchanged.

Figure 1 summarizes the complete pipeline.

## One-Step Semantic-Image Generation

We encode a label field y as a semantic target image $\mathbf { s } ^ { \star } ( u ) = \mathbf { c } _ { y ( u ) }$ and adapt the pix2pix-Turbo generator (Parmar et al. 2024; Sauer et al. 2023) using a fixed segmentation prompt. The codebook is constructed once by greedy max–min selection on a saturated RGB grid, followed by confusion-aware class assignment; the exact prototypes are reported in the supplementary material. During generator training, rendered colors and prototypes are expressed in normalized RGB coordinates. For a color v and temperature $\tau > 0$ , the fixed prototype decoder is

$$
\Pi _ { \mathcal { C } } ^ { ( \tau ) } ( \mathbf { v } ) _ { k } = \frac { \exp \left( - \| \mathbf { v } - \mathbf { c } _ { k } \| _ { 2 } ^ { 2 } / \tau \right) } { \sum _ { j } \exp \left( - \| \mathbf { v } - \mathbf { c } _ { j } \| _ { 2 } ^ { 2 } / \tau \right) } .\tag{3}
$$

The generator objective is

$$
\begin{array} { r l } & { { \mathscr L } _ { G } = 0 . 5 { \mathscr L } _ { \mathrm { r g b } } + 3 . 0 { \mathscr L } _ { \mathrm { p r o t o } } + { \mathscr L } _ { \mathrm { v a l i d } } } \\ & { ~ + 1 . 5 { \mathscr L } _ { \mathrm { m a r g i n } } + 0 . 3 { \mathscr L } _ { \mathrm { b o u n d a r y } } . } \end{array}\tag{4}
$$

Here $\mathcal { L } _ { \mathrm { r g b } }$ is a masked Smooth-L1 loss between the generated and target prototype images. For a generated color v, let $d _ { k } ^ { \mathrm { G } } ( \mathbf { v } ) \ = \ \mathbf { \bar { \lVert { v } } - \mathbf { c } _ { k } \rVert _ { 2 } ^ { 2 } }$ denote its squared distance to prototype $\mathbf { c } _ { k }$ in normalized RGB space. $\mathcal { L } _ { \mathrm { p r o t o } }$ is class-weighted cross-entropy over logits $- d _ { k } ^ { \mathrm { G } } ( \mathbf { v } ) / \tau _ { g } ,$ and $\mathcal { L } _ { \mathrm { v a l i d } }$ averages min<sub>k</sub> $d _ { k } ^ { \mathrm { G } } ( \mathbf { v } )$ . The margin loss averages [m $\begin{array} { r } { + d _ { y } ^ { \mathrm { G } } ( \mathbf { v } ) - \operatorname* { m i n } _ { j \neq y } d _ { j } ^ { \mathrm { G } } ( \mathbf { v } ) ] _ { + } } \end{array}$ , encouraging the target prototype to be closer than the nearest competitor by at least m. $\mathcal { L } _ { \mathrm { b o u n d a r y } }$ applies the same loss within a radius-2 groundtruth boundary band. Ignore pixels are excluded from all reductions. Generator supervision uses RGB coordinates normalized to $[ 0 , 1 ]$ ; thus, we set the prototype-softmax temperature to $\tau _ { g } = 0 . 0 3$ and the margin to $m = 0 . 0 2$ in normalized squared-distance units. The loss weights were fixed during source-only development and were not tuned on validations.

## Observable Semantic-Image Interface

For interface decoding, rendered colors and codebook prototypes are represented on the [0, 255] RGB scale. Because the decoder operates on squared Euclidean color distances, we set the interface temperature to $\tau _ { I } ~ = ~ 9 0 0$ in the corresponding squared-distance units. The fixed decoder maps each rendered pixel to a full class distribution using Equation (1). For each $u \in \Omega$ , let $d _ { k } ^ { \mathrm { I } } ( u ) = \lVert \mathbf { s } ( u ) - \mathbf { c } _ { k } \rVert _ { 2 } ^ { 2 }$ and $z _ { k } ^ { I } ( u ) = - d _ { k } ^ { \mathrm { I } } ( u ) / \tau _ { I } ,$ so that $p _ { I } ( u ) = \mathrm { s o f t m a x } ( \mathbf { z } ^ { I } ( u ) )$ . The rendered RGB value determines its top-1 label:

$$
\hat { y } _ { I } ( u ) = \arg \operatorname* { m a x } _ { k } p _ { I , k } ( u ) = \arg \operatorname* { m i n } _ { k } d _ { k } ^ { \mathrm { I } } ( u ) .\tag{5}
$$

For any two distinct classes $a , b \in \{ 1 , \ldots , K \}$ , the corresponding pairwise log-odds are

$$
\Lambda _ { a b } ^ { I } ( u ) = \log \frac { p _ { I , a } ( u ) } { p _ { I , b } ( u ) } = \frac { d _ { b } ^ { \mathrm { I } } ( u ) - d _ { a } ^ { \mathrm { I } } ( u ) } { \tau _ { I } } .\tag{6}
$$

Thus, given the fixed codebook and temperature, the complete interface distribution is recoverable from the rendered image without access to latent features. In RGB space, each class’s top-1 decision region is its codebook-induced Euclidean Voronoi cell (i.e., the set of colors nearest to that class prototype), while the pairwise log-odds provide closed-form relative evidence between competing classes.

## Hierarchical Generator Evidence Alignment

The rendered semantic image may contain blurred class boundaries or omit thin structures. HGEA augments the interface with spatially aligned multi-level features from the frozen generator. Let J index the selected feature levels $( | \mathcal { I } | = 3$ in our implementation). For each $\ell \in \mathcal { I } , \mathrm { ~ a ~ 1 ~ } \times 1$ projection $\mathbf { W } _ { 1 \times 1 } ^ { \ell }$ maps $\mathbf { h } _ { \mathbf { x } } ^ { \ell }$ to 24 channels, followed by group normalization (GN), SiLU activation, and bilinear resampling $( \mathcal { U } _ { \ell \to \Omega } )$ to the output lattice Ω:

$$
\widetilde { \mathbf { h } } ^ { \ell } = \mathcal { U } _ { \ell \to \Omega } \left( \mathrm { S i L U } \big ( \mathrm { G N } \big ( \mathbf { W } _ { 1 \times 1 } ^ { \ell } * \mathbf { h } _ { \mathbf { x } } ^ { \ell } \big ) \big ) \right) , \qquad \ell \in \mathcal { I } .\tag{7}
$$

The aligned feature maps are concatenated and combined with the input image, rendered semantic image, and interface distribution to predict the logit residual:

$$
\widetilde { \mathbf { h } } = \mathrm { C o n c a t } \widetilde { \mathbf { h } } ^ { \ell } ,\tag{8a}
$$

$$
\begin{array} { r } { \Delta \mathbf { z } ^ { H } = \mathcal { R } _ { \phi } \Big ( \mathrm { C o n c a t } ( \mathbf { x } , \mathbf { s } , p _ { I } , \widetilde { \mathbf { h } } ) \Big ) . } \end{array}\tag{8b}
$$

The residual head $\mathcal { R } _ { \phi }$ comprises two $3 \times 3$ Conv–GN–SiLU blocks followed by a zero-initialized $1 \times 1$ output projection that produces K class-logit residuals at each pixel.

For any two distinct classes $a \ne b ,$ define $\Lambda _ { a b } ^ { H } ( u ) ~ =$ $\log [ p _ { H , a } ( u ) / p _ { H , b } ( u ) ]$ ]. The update in Equation (2) then yields

$$
\Lambda _ { a b } ^ { H } ( u ) - \Lambda _ { a b } ^ { I } ( u ) = \Delta z _ { a } ^ { H } ( u ) - \Delta z _ { b } ^ { H } ( u ) .\tag{9}
$$

Thus, HGEA applies additive corrections to the interface log-odds rather than parameterizing an independent output distribution. The zero-initialized output projection ensures $\Delta \mathbf { z } ^ { H } ( u ) = \mathbf { 0 }$ for every $u \in \Omega$ , and hence $p _ { H } ( u ) = p _ { I } ( u )$ at initialization.

![](images/4500adcf6690166c2054e2c570a922267548d888c31476245e65521bf534ad0f.jpg)  
Figure 1: Overview of Semantic Prism. A one-step pix2pix-Turbo generator renders a semantic RGB image and exposes multi level internal features. The fixed codebook decodes the image into the interface distribution $p _ { I }$ , while HGEA aligns the feature hierarchy and predicts $\Delta \mathbf { z } ^ { H }$ , yielding the refined distribution $p _ { H }$

With the generator frozen, we train only the lightweight 190,891-parameter HGEA module using semantic supervision, boundary- and interior-focused objectives, target-aware sampling, mild class reweighting, and staged false-positive suppression. Complete objectives and training schedules are provided in the supplementary material.

## Contextual Interface–Hierarchy Disagreement

The complement of MSP from the refined distribution provides a pointwise uncertainty score. C-IHD augments this pointwise uncertainty with local context:

$$
\mathcal { U } _ { \mathrm { M S P } } ( u ) = 1 - \operatorname* { m a x } _ { k } p _ { H , k } ( u ) ,\tag{10a}
$$

$$
\mathcal { U } _ { \mathrm { l o c } } ( u ) = \mathcal { A } _ { 5 \times 5 } [ \mathcal { U } _ { \mathrm { M S P } } ] ( u ) ,\tag{10b}
$$

where $\mathcal { A } _ { 5 \times 5 }$ denotes a $5 \times 5$ local averaging operator.

To quantify the distributional change induced by HGEA, we use a normalized ρ-weighted Jensen–Shannon divergence between $p _ { I }$ and $p _ { H }$ . With $\rho ~ = ~ 0 . 8$ and $m _ { \rho } ( u ) = ( 1 -$ $\rho ) p _ { I } ( u ) + \rho p _ { H } ( u )$ , define

$$
\begin{array} { r } { \mathcal { D } _ { \mathrm { I H D } } ( u ) = \cfrac { 1 } { H _ { \mathrm { b } } ( \rho ) } \left[ ( 1 - \rho ) \mathrm { K L } ( p _ { I } ( u ) \| m _ { \rho } ( u ) ) \right. } \\ { \left. + \rho \mathrm { K L } ( p _ { H } ( u ) \| m _ { \rho } ( u ) ) \right] . } \end{array}\tag{11}
$$

Here $H _ { \mathrm { b } } ( \rho ) = - ( 1 - \rho ) \log ( 1 - \rho ) - \rho \log \rho$ is the binary entropy. Because the weighted Jensen–Shannon divergence is bounded by $H _ { \mathrm { b } } ( \rho ) , \bar { D _ { \mathrm { I H D } } } ( u ) \in [ 0 , 1 ]$ , with $\mathcal { D } _ { \mathrm { I H D } } ( u ) = 0$ if and only $\mathrm { i f } p _ { I } ( u ) = p _ { H } ( u )$

Let $\mathbf { g } ( \bar { u } ) \doteq ( \dot { \mathcal { U } } _ { \mathrm { m s p } } ( \bar { u } ) , \dot { \mathcal { U } } _ { \mathrm { l o c } } ( u ) , \mathcal { D } _ { \mathrm { I H D } } ( u ) ) ^ { \top }$ . Using componentwise mean $\mu _ { \mathrm { t r } } ,$ standard deviation $\sigma _ { \mathrm { t r } }$ estimated once from predictions on the corresponding source training set, and fixed weights $\mathbf { w } = ( 1 , 0 . 5 , \bar { 0 } . 2 ) ^ { \top }$ , the final readout is

$$
\boldsymbol { \mathcal { U } } _ { \mathrm { C - I H D } } ( \boldsymbol { u } ) = { { \mathbf { w } } ^ { \top } } \left[ \left( \mathbf { g } ( \boldsymbol { u } ) - \pmb { \mu } _ { \mathrm { t r } } \right) \boldsymbol { \mathcal { O } } \pmb { \sigma } _ { \mathrm { t r } } \right] ,\tag{12}
$$

where $\oslash$ denotes componentwise division. All statistics and weights are fixed before evaluation; source-frozen transfer reuses the source statistics unchanged. C-IHD leaves $p _ { H }$ and the predicted segmentation unchanged and requires neither a trainable error predictor nor additional model forward pass.

## Experiments

## Experimental Setup

Cityscapes (Cordts et al. 2016) serves as our primary benchmark. Using AdamW, we adapt the generator on its 2,975 training images for 100k steps (learning rate $1 0 ^ { - 5 }$ , batch size one), then freeze it and train HGEA for 48k steps (learning rate $5 \times 1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 }$ , gradient clipping 1.0). The training endpoints, interface temperature, and C-IHD coeficients are fixed without validation-set model selection, and readout statistics are estimated from the corresponding source-training predictions.

We evaluate the Cityscapes-trained model on all 500 validation images across 19 classes. We also train Semantic Prism independently on BDD100K (Yu et al. 2020) and evaluate the full 1,000-image validation set. For source-frozen transfer, the Cityscapes-trained model and fixed readout are applied unchanged to all 406 ACDC (Sakaridis, Dai, and Gool 2021) validation images, with ACDC annotations used only for evaluation.

All evaluations use 1024 × 512 outputs, common label mappings, valid-pixel masks, and the same metric implementation, without multiscale or flip test-time augmentation or postprocessing. All methods are evaluated at the same output resolution, although inference cost is not matched. We report mIoU, thin/rare-class mIoU, boundary F-score at a three-pixel tolerance (BF@3), 15-bin expected calibration error $\mathrm { ( E C E _ { 1 5 } ) }$ , Brier score, and negative log-likelihood (NLL). Pixel-error localization is evaluated using the area under the receiver operating characteristic curve (AUROC), AUPR, and the area under the risk–coverage curve (AURC), with incorrect valid pixels treated as positives. External baselines use MSP for cross-model comparison; the contribution of C-IHD is evaluated against MSP on identical $p _ { H }$

![](images/6731bf9f2daaad70ebcc8e2e8c389c7310f19cf79d9fd1a133ec252253caf61b.jpg)  
Figure 2: Observable-to-final Cityscapes predictions. Fixed-codebook decoding converts the generated semantic RGB image into the Direct Interface prediction; HGEA then produces the final Semantic Prism prediction. Aligned insets zoom in and highlight representative diferences at semantic boundaries and thin structures.

## Observable Semantic Interface: Evaluation on Cityscapes

Before hierarchical refinement, fixed-codebook decoding of the generated semantic image attains 60.68% mIoU, demonstrating that the visible output is independently evaluable. Figure 2 shows that the rendered image captures major scene regions, while HGEA corrects representative boundary and thin-structure errors without broadly disturbing correct regions. The interface has 5.69% ECE, showing that direct decodability does not by itself imply calibrated probabilities.

Table 1 compares all methods under the common evaluation protocol. Relative to direct-interface decoding, HGEA raises mIoU by 11.39 points to 72.07%, improves thin/rareclass mIoU and BF@3, and lowers ECE from 5.69% to 0.41%. Under the common-resolution protocol, Semantic Prism exceeds SegFormer-B0 in mIoU but remains below the stronger discriminative and difusion baselines.

For pixel-error localization, M2F-SwinT attains marginally higher AUROC, whereas Semantic Prism with C-IHD yields the highest AUPR. Because predictor accuracy and error prevalence difer, these cross-model rankings are descriptive; Table 5 further isolates the incremental efect of C-IHD over MSP with $p _ { H }$ held fixed.

Additional codebook comparison, Brier score and NLL, temperature sensitivity, reliability diagrams, and spatial calibration analysis are provided in the supplementary material. The next subsection uses matched refiners to test whether the HGEA gains extend beyond generic learned refinement.

## In-Domain Evaluation on BDD100K

To verify the robustness, we further train Semantic Prism independently on BDD100K and evaluate the full 1,000- image validation set (Yu et al. 2020). As shown in Table 2, the model ranks second in mIoU, thin/rare-class mIoU, and BF@3, trailing DSNet-Base by 0.14 mIoU points and 1.75 BF@3 points, while achieving the lowest ECE and Brier score among compared methods. On the same fixed $p _ { H } , { \bf C } \mathrm { - }$

IHD raises AUROC/AUPR from 0.9000/0.4395 with MSP to 0.9018/0.4481 without changing segmentation or calibration.

## Controlled Attribution of Hierarchical Evidence

To isolate hierarchical evidence from generic learned refinement, as shown in Table 3, we compare four capacitymatched refiners against the fixed Direct Interface (DI) baseline. Each refiner has approximately 0.191M trainable parameters, with counts difering by less than 0.1%, and is trained for 36k steps over three prespecified seeds using matched sample order, crops, optimization, and losses.

The Observable-Interface Refiner (OI-Ref) receives only the rendered semantic image and interface distribution p . The Capacity-Matched Flat Refiner (CM-Flat) additionally receives the input RGB image and uses the same interfacelogit residual formulation, but no generator features. Single-Level HGEA $\mathrm { ( S L \mathrm { - } H G E A _ { \mathrm { m i d } } ) }$ uses only the prespecified middle generator level, whereas Multi-Level HGEA (ML-HGEA) uses all three levels.

Across all three seeds, mIoU follows the same ordering: OI-Ref $< \mathrm { C M \mathrm { - } F l a t } < \mathrm { S L \mathrm { - } H G E A _ { \mathrm { m i d } } < M L \mathrm { - } H G E A }$ . ML-HGEA achieves 71.43 ± 0.47% mIoU and outperforms CM-Flat and $\mathrm { S L \mathrm { - } H G E A _ { \mathrm { { m i d } } } }$ by paired margins of 1.68 ± 0.11 and 0.89 ± 0.23 mIoU points. Here, ± denotes the sample standard deviation over the three seeds, computed from paired per-seed diferences for the reported margins. These consistent gains support the benefit of joint multi-level alignment over capacity-matched hierarchy-free and single-level refinement. ML-HGEA also improves thin/rare-class mIoU and BF@3 over CM-Flat while maintaining comparable ECE.

A pixel-aligned audit further shows that HGEA leaves the interface top-1 prediction unchanged on 93.48% of valid pixels. Among interface errors, it corrects 48.68%, yielding a net pixel-accuracy gain of 3.67 points. This gain is concentrated at semantic boundaries, reaching 18.85 points compared with 1.67 points in region interiors. These results indicate that HGEA selectively corrects dificult pixels while preserving most interface predictions. Additional calibration, output-parameterization, removal, and spatial-shufling analyses are provided in the supplementary material.

<table><tr><td></td><td></td><td colspan="2">Segmentation</td><td colspan="2">Calibration</td><td colspan="4">Pixel-Error Localization</td></tr><tr><td>Method</td><td>Paradigm</td><td>mIoU (%)↑ Thin/Rare (%)↑ BF@3 (%)↑ ECE15 (%)↓ AUROC↑ AUPR↑ Ent.-AUPR↑ Mar.-AUPR↑</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SegFormer-B0</td><td>Disc.</td><td>71.13</td><td>61.90</td><td>79.51</td><td>1.08</td><td>0.9340</td><td>0.4419</td><td>0.4274</td><td>0.4225</td></tr><tr><td>M2F-SwinT</td><td>Disc.</td><td>77.92</td><td>71.03</td><td>85.63</td><td>1.80</td><td>0.9459</td><td>0.4339</td><td>0.4480</td><td>0.4163</td></tr><tr><td>DDPS-B0</td><td>Iter. Diff.</td><td>74.17</td><td>67.06</td><td>82.83</td><td>1.48</td><td>0.9375</td><td>0.4291</td><td>0.4233</td><td>0.4093</td></tr><tr><td>DDP-CNXT-T</td><td>Iter. Diff.</td><td>79.10</td><td>73.33</td><td>86.87</td><td>1.51</td><td></td><td>0.94420.4115</td><td>0.4045</td><td>0.3965</td></tr><tr><td>GSS-FF-R101</td><td>Gen. Mask</td><td>75.00</td><td>68.27</td><td>85.33</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Direct Interface 1-Step Gen.</td><td></td><td>60.68</td><td>48.76</td><td>78.70</td><td>5.69</td><td></td><td></td><td></td><td></td></tr><tr><td>Semantic Prism 1-Step Gen.</td><td></td><td>72.07</td><td>63.80</td><td>81.26</td><td>0.41</td><td></td><td>0.9457 0.4812</td><td>0.4504</td><td>0.4546</td></tr></table>

Table 1: Cityscapes val500 at 1024 × 512. Pixel-error scores are MSP for external models and C-IHD for Semantic Prism. Ent.-AUPR and Mar.-AUPR use entropy and inverse top-two margin, respectively. Baselines are SegFormer (Xie et al. 2021), Mask2Former (Cheng et al. 2022), DDPS (Lai et al. 2023), DDP (Ji et al. 2023), and GSS (Chen et al. 2023).
<table><tr><td></td><td></td><td colspan="3">Segmentation</td><td colspan="2">Calibration</td><td colspan="2">Pixel-Error Localization</td></tr><tr><td>Method</td><td>Paradigm</td><td>mIoU (%)↑</td><td>T/R (%)↑</td><td>BF@3 (%)↑</td><td>ECE15↓(%)</td><td>Brier↓</td><td>AUROC↑</td><td>AUPR↑</td></tr><tr><td>DSNet-Base (Guo et al. 2024)</td><td>Disc.</td><td>62.36</td><td>50.69</td><td>74.08</td><td>3.70</td><td>0.0985</td><td>0.8435</td><td>0.3776</td></tr><tr><td>MSeg BDD-1M (Lambert et al. 2020)</td><td>Disc.</td><td>60.79</td><td>49.12</td><td>67.29</td><td>0.89</td><td>0.1003</td><td>0.8942</td><td>0.4299</td></tr><tr><td>Semantic Prism (MSP)</td><td>1-Step Gen.</td><td>62.22</td><td>50.19</td><td>72.33</td><td>0.88</td><td>0.0910</td><td>0.9000</td><td>0.4395</td></tr><tr><td>Semantic Prism (C-IHD)</td><td>1-Step Gen.</td><td>62.22</td><td>50.19</td><td>72.33</td><td>0.88</td><td>0.0910</td><td>0.9018</td><td>0.4481</td></tr></table>

Table 2: Unified evaluation on BDD100K val1000. The MSP and C-IHD rows share $p _ { H }$ and difer only in the uncertainty readout. T/R denotes thin/rare-class mIoU. Bold and underlined values denote the best and second-best results.

<table><tr><td>Model</td><td>mIoU↑</td><td>T/R↑</td><td>BF@3↑</td><td> $\mathrm { E C E 1 5 ~ \downarrow }$ </td></tr><tr><td>DI</td><td>60.68</td><td>48.76</td><td>78.70</td><td>5.69</td></tr><tr><td>OI-Ref</td><td>67.58 ±0.88</td><td>57.79 ±1.19</td><td>79.65 ±0.36</td><td>0.40 ±0.02</td></tr><tr><td>CM-Flat</td><td>69.75</td><td>60.83</td><td>79.82</td><td>0.47</td></tr><tr><td></td><td>±0.40 70.54</td><td>±0.63 61.77</td><td>±0.23 80.32</td><td>±0.14 0.39</td></tr><tr><td> $\mathbf { S L } \mathbf { - H G E A } _ { \mathrm { m i d } }$ </td><td>±0.59 71.43</td><td>±0.75 62.55</td><td>±0.56 80.26</td><td>±0.12 0.43</td></tr><tr><td>ML-HGEA</td><td>±0.47</td><td>±1.17</td><td>±1.25</td><td>±0.16</td></tr></table>

Table 3: Matched refinement ablation on Cityscapes val500. Learned controls report three-seed mean ± sample standard deviation (both in %). These checkpoints are trained separately from the canonical 72.07% result in Table 1.

<table><tr><td>Method Type mIoU↑  $\mathrm { E C E _ { 1 5 } \downarrow }$  AUROC↑ AUPR↑</td></tr><tr><td>Disc.</td><td>46.48 10.61 0.8407</td></tr><tr><td>SegFormer-B0 DDPS-B0</td><td>Diff. 48.36 12.12</td></tr><tr><td>M2F-SwinT Disc.</td><td>0.8164 0.4729 49.74 9.65 0.8965 0.5935</td></tr><tr><td>GSS-FF-R101 Gen. 37.26</td><td></td></tr><tr><td>DDP-CNXT-T Diff.</td><td>51.24 12.62 0.7109 0.3558</td></tr><tr><td></td><td>34.72 9.92 0.8829 0.6932</td></tr><tr><td>DANet-R101 Disc. DeepLabv3+_R101 Disc.</td><td>10.22 0.8611 0.6080</td></tr><tr><td>39.67 Ours (MSP)</td><td>8.48 0.8903 0.6580</td></tr><tr><td>Gen. 46.89</td><td>46.89 8.48 0.9076 0.7557</td></tr><tr><td>Ours (C-IHD) Gen.</td><td></td></tr></table>

Table 4: Source-frozen evaluation on the 406-image ACDC validation set at 1024 × 512. The MSP and C-IHD rows share $p _ { H }$ and difer only in the uncertainty readout. mIoU and ECE are in $\% ;  {  }  { \stackrel { . . } { - } }  { \stackrel { . } { \operatorname { \jmath } } }$ denotes unavailable results. Additional baselines are DANet-R101 (Fu et al. 2019) and DeepLabv3+\_R101 (Chen et al. 2018).

## Source-Frozen Evaluation on ACDC

We apply the Cityscapes-trained predictor and fixed C-IHD readout unchanged to all 406 ACDC validation images (Sakaridis, Dai, and Gool 2021) spanning fog, night, rain, and snow, without target adaptation. As shown in Table 4, Semantic Prism attains 46.89% mIoU and the lowest ECE in the comparison (8.48%), while DDP-CNXT-T attains the highest mIoU. Because predictor error rates difer, the crossmodel rankings are descriptive; the paired Semantic Prism rows isolate the readout efect. With $p _ { H }$ held fixed, C-IHD raises AUROC/AUPR from 0.8903/0.6580 to 0.9076/0.7557.

Figure 3 shows that C-IHD concentrates high risk around challenging residual errors, particularly near low-contrast boundaries, thin structures, and regions degraded by adverse weather/illumination, while generally assigning lower risk to correctly predicted areas. However, it occasionally flags correct boundaries and overlooks some high-confidence errors.

## Fixed-Prediction Reliability Analysis

To isolate readout efects, we conduct fixed-prediction ablations comparing alternative uncertainty scores with $p _ { H }$ and the segmentation predictions held fixed. Across Cityscapes, BDD100K, and ACDC, C-IHD consistently improves MSPbased pixel-error ranking. On Cityscapes val500, it raises AUROC/AUPR from 0.94502/0.47814 to 0.94572/0.48121 and reduces AURC from 4 $. 2 8 1 \times 1 0 ^ { - 3 }$ to 4 $. 2 3 5 \times 1 0 ^ { - 3 }$ . Table 5 shows that local context provides the main individual gain, while IHD complements it in the full readout despite mixed standalone efects. Paired image-level bootstrap 95% confidence intervals exclude zero for both the AUROC and AUPR gains. Results remain stable across the predeclared sensitivity grid; details are provided in the supplementary.

Ground Truth  
Semantic Prism  
Residual Error  
C-IHD Score  
![](images/6e10896ade8803e716a6112655899b7071f218718642969b0ad25897eb31488b.jpg)  
Figure 3: Source-frozen ACDC transfer evaluation. Rows show scenarios of fog, night, rain, and snow; columns show the input, ground truth, prediction, pixel-error mask, and shared-scale C-IHD risk map. Quantitative results are reported in Table 4.

<table><tr><td>Readout</td><td>AUROC↑</td><td>AUPR↑</td><td>AURC  $( 1 0 ^ { - 3 } ) \downarrow$ </td></tr><tr><td>MSP</td><td>0.94502</td><td>0.47814</td><td>4.281</td></tr><tr><td>MSP + IHD</td><td>0.94512</td><td>0.47730</td><td>4.273</td></tr><tr><td>MSP + Local-MSP</td><td>0.94558</td><td>0.48026</td><td>4.244</td></tr><tr><td>C-IHD</td><td>0.94572</td><td>0.48121</td><td>4.235</td></tr></table>

Table 5: Pixel-error-ranking readouts on the 500-image Cityscapes validation set with $p _ { H }$ held fixed. Local-MSP denotes the 5 × 5 local average of the MSP-derived uncertainty, and IHD denotes interface–hierarchy disagreement.
<table><tr><td colspan="3">(a) End-to-end inference throughput</td></tr><tr><td>Method</td><td>Inference steps↓</td><td>FPS↑ Peak VRAM (GiB)↓</td></tr><tr><td>DDP-CNXT-T</td><td>3</td><td>3.33</td></tr><tr><td>DDPS-B0</td><td>20</td><td>0.47 0.84 1.25</td></tr><tr><td>Semantic Prism</td><td>1</td><td>1.57 6.40</td></tr></table>

(b) Incremental adaptation overhead
<table><tr><td>Component</td><td>Trainable params</td><td>Core (%)</td><td>Latency [ms (%)]</td></tr><tr><td>Unmerged LoRA</td><td>9.002M</td><td>0.698</td><td>128.93 (20.30)</td></tr><tr><td>VAE skips</td><td>0.492M</td><td>0.038</td><td>1.85 (0.29)</td></tr><tr><td>HGEA</td><td>0.191M</td><td>0.015</td><td>29.54 (4.65)</td></tr><tr><td>C-IHD (fixed)</td><td>0</td><td>0</td><td>1.16 (0.18)</td></tr><tr><td>Total optimized</td><td>9.696M</td><td>0.752</td><td></td></tr></table>

Table 6: Computational cost measured on a single NVIDIA A100 80 GB GPU with batch size 1. In (b) percentages are relative to end-to-end latency; component latencies are measured independently and are not additive.

## Computational Cost

Table 6 shows that Semantic Prism achieves 1.57 FPS, approximately 1.9× the throughput of 20-step DDPS (Lai et al. 2023), but remains slower and requires more peak memory than three-step DDP-CNXT-T (Ji et al. 2023). The full adaptation setup optimizes 9.696M parameters, corresponding to 0.752% of the core model. Among the added components, unmerged low-rank adaptation (LoRA) incurs the largest measured latency overhead, while the overheads of HGEA and C-IHD correspond to 4.65% and 0.18% of endto-end latency, respectively. Additional timing analyses, including the efect of LoRA weight merging, are reported in the supplementary material.

## Conclusion

We presented Semantic Prism, a one-step generative segmentation framework with deterministic inference, in which a rendered semantic image and a fixed codebook decoder define an independently evaluable probabilistic interface. HGEA uses aligned multi-level generator evidence to predict an additive residual in the interface logit space, while C-IHD reuses the interface and refined distributions for pixelerror ranking. On Cityscapes, HGEA raises mIoU by 11.39 points over direct-interface decoding. Complementary experiments with a separately trained BDD100K model and source-frozen transfer from Cityscapes to ACDC evaluate the framework on a second in-domain benchmark and under an adverse-condition domain shift, respectively. Across all three datasets, C-IHD improves AUPR over MSP without changing the segmentation predictions.

The semantic-image interface alone does not fully characterize the learned HGEA correction. Additional limitations include the closed-set codebook, the computational cost of the generator, and limited cross-domain evidence. Overall, results indicate that image-form generative segmentation can preserve a quantitatively evaluable probability interface, improve spatial accuracy through hierarchical refinement, and enable lightweight fixed-prediction pixel-error ranking.

## References

Ahmed, M.; Hasan, Z.; Haque, S. A.; Faridee, A.-Z.; Purushotham, S.; You, S.; and Roy, N. 2025. CAM-Seg: A Continuous-Valued Embedding Approach for Semantic Image Generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, 6421– 6430.

Caetano, F.; Viviers, C.; de With, P. H. N.; and van der Sommen, F. 2026. Symmetrical Flow Matching: Unified Image Generation, Segmentation, and Classification with Score-Based Generative Models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 2498–2506.

Chen, J.; Lu, J.; Zhu, X.; and Zhang, L. 2023. Generative Semantic Segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7111–7120.

Chen, L.-C.; Zhu, Y.; Papandreou, G.; Schrof, F.; and Adam, H. 2018. Encoder–Decoder with Atrous Separable Convolution for Semantic Image Segmentation. In Proceedings of the European Conference on Computer Vision, 801–818.

Cheng, B.; Misra, I.; Schwing, A. G.; Kirillov, A.; and Girdhar, R. 2022. Masked-Attention Mask Transformer for Universal Image Segmentation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1290–1299.

Choi, S.; Jung, S.; Yun, H.; Kim, J. T.; Kim, S.; and Choo, J. 2021. RobustNet: Improving Domain Generalization in Urban-Scene Segmentation via Instance Selective Whitening. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11580–11590.

Cordts, M.; Omran, M.; Ramos, S.; Rehfeld, T.; Enzweiler, M.; Benenson, R.; Franke, U.; Roth, S.; and Schiele, B. 2016. The Cityscapes Dataset for Semantic Urban Scene Understanding. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 3213–3223.

Deng, J.; Weng, T.; Yang, T.; Luo, W.; Li, Z.; and Jiang, W. 2025. LlamaSeg: Image Segmentation via Autoregressive Mask Generation. arXiv preprint arXiv:2505.19422.

Ding, Z.; Han, X.; Liu, P.; and Niethammer, M. 2021. Local Temperature Scaling for Probability Calibration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 6889–6899.

Fu, J.; Liu, J.; Tian, H.; Li, Y.; Bao, Y.; Fang, Z.; and Lu, H. 2019. Dual Attention Network for Scene Segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3146–3154.

Fu, Y.; Lou, M.; and Yu, Y. 2025. SegMAN: Omni-scale Context Modeling with State Space Models and Local Attention for Semantic Segmentation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 19077–19087.

Gabeur, V.; Long, S.; Peng, S.; Voigtlaender, P.; Sun, S.; Bao, Y.; Truong, K.; Wang, Z.; Zhou, W.; Barron, J. T.; Genova, K.; Kannen, N.; Ben, S.; Li, Y.; Guo, M.; Yogin, S.; Gu, Y.; Chen, H.; Wang, O.; Xie, S.; Zhou, H.; He, K.; Funkhouser, T.; Alayrac, J.-B.; and Soricut, R. 2026. Image

Generators are Generalist Vision Learners. arXiv preprint arXiv:2604.20329.

Geifman, Y.; and El-Yaniv, R. 2017. Selective Classification for Deep Neural Networks. In Advances in Neural Information Processing Systems, volume 30.

Geng, Z.; Yang, B.; Hang, T.; Li, C.; Gu, S.; Zhang, T.; Bao, J.; Zhang, Z.; Li, H.; Hu, H.; Chen, D.; and Guo, B. 2024. InstructDifusion: A Generalist Modeling Interface for Vision Tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 12709–12720.

Guarino, V. E.; Winklmayr, C.; Franzen, J.; Rumberger, J. L.; Pfeufer, M.; Greven, S.; Maier-Hein, K.; Kainmueller, D.; Karg, C.; and Lüth, C. T. 2026. Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13145–13156.

Guo, C.; Pleiss, G.; Sun, Y.; and Weinberger, K. Q. 2017. On Calibration of Modern Neural Networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, 1321–1330.

Guo, Z.; Bian, L.; Huang, X.; Wei, H.; Li, J.; and Ni, H. 2024. DSNet: A Novel Way to Use Atrous Convolutions in Semantic Segmentation. arXiv preprint arXiv:2406.03702.

Hendrycks, D.; and Gimpel, K. 2017. A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. In International Conference on Learning Representations.

Ji, Y.; Chen, Z.; Xie, E.; Hong, L.; Liu, X.; Liu, Z.; Lu, T.; Li, Z.; and Luo, P. 2023. DDP: Difusion Model for Dense Visual Prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 21741–21752.

Koh, P. W.; Nguyen, T.; Tang, Y. S.; Mussmann, S.; Pierson, E.; Kim, B.; and Liang, P. 2020. Concept Bottleneck Models. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, 5338–5348.

Küppers, F.; Haselhof, A.; Kronenberger, J.; and Schneider, J. 2022. Confidence Calibration for Object Detection and Segmentation. arXiv preprint arXiv:2202.12785.

Kwon, D.; and Kwak, S. 2022. Semi-Supervised Semantic Segmentation with Error Localization Network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9947–9957.

Lai, Z.; Duan, Y.; Dai, J.; Li, Z.; Fu, Y.; Li, H.; Qiao, Y.; and Wang, W. 2023. Denoising Difusion Semantic Segmentation with Mask Prior Modeling. arXiv preprint arXiv:2306.01721.

Lambert, J.; Liu, Z.; Sener, O.; Hays, J.; and Koltun, V. 2020. MSeg: A Composite Dataset for Multi-Domain Semantic Segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Li, F.; Wang, X.; Qi, M.; Zhang, Z.; and Xu, Y. 2025. Better to Teach than to Give: Domain Generalized Semantic Segmentation via Agent Queries with Difusion Model Guidance. In Proceedings of the 42nd International Conference

on Machine Learning, volume 267 of Proceedings of Machine Learning Research, 36129–36139.

Losch, M.; Fritz, M.; and Schiele, B. 2021. Semantic Bottlenecks: Quantifying and Improving Inspectability of Deep Representations. International Journal ofComputer Vision, 129: 3136–3153.

Meng, B.; Xu, Q.; Wang, Z.; Cao, X.; Huang, L.; and Huang, Q. 2026. Making Training-Free Difusion Segmentors Scale with the Generative Power. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 35861–35871.

Murugesan, B.; Vasudeva, S. A.; Liu, B.; Lombaert, H.; Ayed, I. B.; and Dolz, J. 2025. Neighbor-Aware Calibration of Segmentation Networks with Penalty-Based Constraints. Medical Image Analysis, 101: 103501.

Parmar, G.; Park, T.; Narasimhan, S. G.; and Zhu, J. 2024. One-Step Image Translation with Text-to-Image Models. arXiv preprint arXiv:2403.12036.

Porta, H.; Dalsasso, E.; Marcos, D.; and Tuia, D. 2025. Multi-Scale Grouped Prototypes for Interpretable Semantic Segmentation. In Proceedings of the IEEE/CVF Winter Conference onApplications ofComputer Vision, 2869–2880.

Qi, L.; Yang, L.; Guo, W.; Xu, Y.; Du, B.; Jampani, V.; and Yang, M.-H. 2024. UniGS: Unified Representation for Image Generation and Segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6305–6315.

Rahman, Q. M.; Sünderhauf, N.; Corke, P.; and Dayoub, F. 2022. FSNet: A Failure Detection Framework for Semantic Segmentation. IEEE Robotics and Automation Letters, 7(2): 3030–3037.

Rottmann, M.; Colling, P.; Hack, T.-P.; Chan, R.; Hüger, F.; Schlicht, P.; and Gottschalk, H. 2020. Prediction Error Meta Classification in Semantic Segmentation: Detection via Aggregated Dispersion Measures of Softmax Probabilities. In International Joint Conference on Neural Networks, 1–9.

Sacha, M.; Rymarczyk, D.; Struski, Ł.; Tabor, J.; and Zieliński, B. 2023. ProtoSeg: Interpretable Semantic Segmentation with Prototypical Parts. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 1481–1492.

Sakaridis, C.; Dai, D.; and Gool, L. V. 2021. ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Perception. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 10765–10775.

Sauer, A.; Lorenz, D.; Blattmann, A.; and Rombach, R. 2023. Adversarial Difusion Distillation. arXiv preprint arXiv:2311.17042.

Wang, C.; Li, X.; Qi, L.; Ding, H.; Tong, Y.; and Yang, M.- H. 2024. SemFlow: Binding Semantic Segmentation and Image Synthesis via Rectified Flow. In Advances in Neural Information Processing Systems, volume 37.

Wang, D.; Gong, B.; and Wang, L. 2023. On Calibrating Semantic Segmentation Models: Analyses and an Algorithm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 23652–23662.

Wang, M.; Ding, H.; Liew, J. H.; Liu, J.; Zhao, Y.; and Wei, Y. 2023a. SegRefiner: Towards Model-Agnostic Segmentation Refinement with Discrete Difusion Process. In Advances in Neural Information Processing Systems, volume 36.

Wang, X.; Zhang, X.; Cao, Y.; Wang, W.; Shen, C.; and Huang, T. 2023b. SegGPT: Towards Segmenting Everything in Context. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 1130–1140.

Xie, E.; Wang, W.; Yu, Z.; Anandkumar, A.; Alvarez, J. M.; and Luo, P. 2021. SegFormer: Simple and Eficient Design for Semantic Segmentation with Transformers. In Advances in Neural Information Processing Systems, vol ume 34, 12077–12090.

Xu, J.; Liu, S.; Vahdat, A.; Byeon, W.; Wang, X.; and Mello, S. D. 2023. Open-Vocabulary Panoptic Segmentation with Text-to-Image Difusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2955–2966.

Yang, Y.; Zhuang, X.; Cai, Y.; Ma, C.; Bai, S.; Yao, J.; Zhang, Y.; Lin, J.; and Wang, Y. 2026. GenMask: Adapting DiT for Segmentation via Direct Mask Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 20455–20467.

Yu, F.; Chen, H.; Wang, X.; Xian, W.; Chen, Y.; Liu, F.; Madhavan, V.; and Darrell, T. 2020. BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Zhao, C.; Sun, Y.; Liu, M.; Zheng, H.; Zhu, M.; Zhao, Z.; Chen, H.; He, T.; and Shen, C. 2025. DICEPTION: A Generalist Difusion Model for Visual Perceptual Tasks. In Advances in Neural Information Processing Systems, volume 38.

Zhao, W.; Rao, Y.; Liu, Z.; Liu, B.; Zhou, J.; and Lu, J. 2023. Unleashing Text-to-Image Difusion Models for Visual Perception. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 5729–5739.

# Supplementary Material for Generative Semantic Segmentation via an Observable Semantic-Image Interface and Hierarchical Generator Evidence Alignment

Weize Cai<sup>1,†</sup> Yongqi Dong<sup>1,2,†,∗</sup> Zhida Shao<sup>1,†</sup> Zixin Fu<sup>3,†</sup>

<sup>1</sup>RWTH Aachen University, Aachen, Germany

<sup>2</sup>Delft University of Technology, Delft, The Netherlands

<sup>3</sup>Chang’an University, Xi’an, China

<sup>†</sup>Equal contribution. <sup>∗</sup>Corresponding author: yongqi.dong@rwth-aachen.de

This supplement provides implementation details, controlled analyses, reliability diagnostics, and transfer/deployment results that complement the main paper. Unless stated otherwise, all Cityscapes analyses use the same frozen generator, fixed semantic-RGB interface, and final HGEA checkpoint. We denote the Direct Interface and refined class distributions by p<sub>I</sub> and p<sub>H</sub>, respectively.

## Implementation and Evaluation Protocol Environment and Reproducibility Controls

All experiments are run with Python 3.10, PyTorch 2.5.1, TorchVision 0.20.1, and CUDA 12.1 on a single NVIDIA GPU. The one-step generator uses SD-Turbo with the pix2pix-Turbo adapter stack, implemented with difusers 0.38.0, transformers 5.12.1, peft 0.19.1, accelerate 1.14.0, and safetensors 0.8.0. Additional dependencies include NumPy 2.2.5, Pillow 12.1.0, scikitimage 0.25.2, scikit-learn 1.7.2, SciPy 1.15.3, OpenCV 4.13, and cityscapesscripts 2.2.4. The complete pinned environment is provided as a Conda specification. Training and evaluation require CUDA; no CPU-only execution path is supported. At $5 1 2 \times 5 1 2$ , a single-GPU run uses approximately 24 GB of device memory.

Training is organized into two sequential stages separated by a frozen handof: generator adaptation followed by HGEA refinement. Table S1 summarizes the core optimization settings; the stage-specific objectives and curricula are detailed in their respective subsections below.

<table><tr><td>Configuration</td><td>Generator Adaptation</td><td>HGEA</td></tr><tr><td>Total steps</td><td>100,000</td><td>48,000</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Batch size</td><td>4</td><td>1</td></tr><tr><td>Precision</td><td>FP32</td><td>FP32</td></tr><tr><td>Gradient clipping</td><td>1.0</td><td>1.0</td></tr><tr><td>Random seed</td><td>13013</td><td>38001</td></tr></table>

Table S1: Two-stage training configuration for generator adaptation and HGEA refinement.

Each stage uses a separate fixed seed (generator 13013,

HGEA 38001). Python, NumPy, and $\mathrm { P y } ^ { \prime }$ Torch, including the CUDA random-number generators, are seeded before any model or data loader is constructed, thereby controlling adapter initialization, crop sampling, and augmentation streams. Generator adaptation records the resolved configuration and checkpoints. Each HGEA checkpoint additionally stores the Python, NumPy, PyTorch, and CUDA RNG states, allowing an interrupted run to resume from the same stochastic state. Non-finite-loss handling is stage-specific: generator adaptation terminates immediately to expose divergence, whereas HGEA skips the afected optimizer update and continues from the last valid state.

## Data Pipeline and Preprocessing

Cityscapes samples are discovered by pairing each leftImg8bit image with its matching gtFine\_labelTrainIds label under the oficial {train,val}/city/ layout; images are loaded as 8-bit RGB and labels as 19-class trainId maps with 255 as the ignore index. Paired augmentation is fully synchronized: bicubic resize with antialiasing for the image and nearestneighbor resize for the label, a shared crop window, and a shared horizontal flip (probability 0.5). Training draws a scale factor uniformly from [0.5, 2.0] and, when the scaled short side would fall below the crop size, rescales up so a $5 1 2 \times 5 1 2$ crop is always feasible; evaluation instead resizes the short side to 512 and takes a fixed crop. Generator supervision converts each label crop to the fixed semantic-RGB target on the fly, mapping ignore pixels to a reserved color and encoding the generator target in [−1, 1]. Rare-target crops sample a window that maximizes coverage of the designated thin/small classes over a bounded number of attempts, falling back to a uniform crop when none qualifies; this is a sampling policy only and leaves the loss and evaluation code unchanged.

## Full-Frame Inference and Stitching

All reported outputs have resolution 1024×512. Semantic Prism uses three overlapping 512×512 windows with horizontal origins 0, 256, and 512. Each window undergoes one deterministic generator trajectory, fixed-codebook decoding, and HGEA refinement. In overlap regions, $p _ { I }$ and $p _ { H }$ are averaged and renormalized separately. Labels are decoded from the stitched $p _ { H }$ , and C-IHD reads the stitched $p _ { I }$ and $p _ { H }$ . No multiscale or flip test-time augmentation, postprocessing, or runtime ensemble is used.

## One-Step Generator Adaptation

The semantic-image generator starts from SD-Turbo with fresh pix2pix-Turbo adapters; no task-specific pix2pix-Turbo LoRA is loaded. Adaptation at 512×512 optimizes rank-8 U-Net LoRA, the U-Net input convolution, rank-4 VAE LoRA, and four VAE skip convolutions, for 9,505,160 optimized parameters. AdamW uses learning rate $1 0 ^ { - 5 }$ , weight decay $\stackrel { \cdot } { 1 } 0 ^ { - 2 } , ( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 ) , \stackrel {  } { \epsilon } = 1 0 ^ { - 8 }$ , batch size one, constant learning rate, gradient clipping at 1.0, and bfloat16 autocast for 100,000 steps. Augmentation comprises scale jitter in [0.5, 2.0], a 512×512 crop, and horizontal flipping with probability 0.5. Steps 1–20,000 use random crops; later steps attempt a rare-target crop with probability 0.5 without resetting the optimizer.

Let $\Omega = \{ u : y ( u ) \neq 2 5 5 \}$ and $E _ { k } ( u ) = \| \mathbf { s } ( u ) - \mathbf { c } _ { k } \| _ { 2 } ^ { 2 }$ in normalized RGB. The generator objective is

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { G } } = 0 . 5 \mathcal { L } _ { \mathrm { r g b } } + 3 . 0 \mathcal { L } _ { \mathrm { p r o t o } } + \mathcal { L } _ { \mathrm { v a l i d } } } \\ & { ~ + 1 . 5 \mathcal { L } _ { \mathrm { m a r g i n } } + 0 . 3 \mathcal { L } _ { \mathrm { b o u n d a r y } } . } \end{array}\tag{S1}
$$

The RGB term is Smooth-L1 over valid pixels. The prototype term is cross-entropy under the fixed codebook with temperature $\tau _ { g } ~ = ~ 0 . 0 3 ; ~ \mathcal { L } _ { \mathrm { v a l i d } }$ minimizes the distance to the nearest prototype; and the margin terms use $\begin{array} { r } { [ 0 . 0 2 + E _ { y ( u ) } - \mathrm { m i n } _ { j \neq y ( u ) } ^ { } E _ { j } ] _ { + } } \end{array}$ Boundary supervision marks both sides of four-neighbor label transitions and dilates the mask with a 5×5 square kernel. Prototype and margin terms use mean-normalized inverse-square-root classfrequency weights; designated rare classes receive a factor of 1.5 before clipping at 8 and renormalization. The generator is frozen before HGEA training.

## Fixed Semantic-RGB Interface and Codebook

The High-Separation codebook $\mathcal { C } _ { \mathrm { H S } }$ is constructed once with a greedy max–min heuristic on a saturated RGB grid and assigned to increase separation between commonly confused classes. The heuristic is not claimed to solve global maximin packing. The prototypes are fixed before training and unchanged at inference. Table S2 records the complete codebook in prediction-channel order.

Operationally, the main paper’s [0, 255] interface notation uses the following lossless 8-bit serialization. For generator output $t \in [ - 1 , 1 ]$

$$
Q _ { 8 } ( t ) = \mathrm { r o u n d } _ { \mathrm { e v e n } } \left( \mathrm { c l i p } ( 1 2 7 . 5 ( t + 1 ) , 0 , 2 5 5 ) \right) .\tag{S2}
$$

For window w, the interface is recovered solely from the rendered image:

$$
\begin{array} { l } { { E _ { w , k } ( u ) = \lVert Q _ { 8 } ( \mathbf { s } _ { w } ( u ) ) - \mathbf { c } _ { k } ^ { ( 8 ) } \rVert _ { 2 } ^ { 2 } , } } \\ { { p _ { w , k } ^ { I } ( u ) = \displaystyle \frac { \exp [ - E _ { w , k } ( u ) / \tau _ { I } ] } { \sum _ { j } \exp [ - E _ { w , j } ( u ) / \tau _ { I } ] } , \qquad \tau _ { I } = 9 0 0 . } } \end{array}\tag{S3}
$$

Here τ is measured in squared 8-bit RGB units and is distinct from the normalized-RGB supervision temperature $\tau _ { g }$

The fixed-endpoint comparison in Table S3 isolates codebook geometry: both palettes remain directly decodable, while the high-separation construction mainly improves boundary recovery.

## HGEA Architecture and Optimization

HGEA reads frozen VAE encoder levels with 128, 256, and 512 channels at 256<sup>2</sup>, 128<sup>2</sup>, and $6 4 ^ { 2 }$ for each $5 1 2 ^ { 2 }$ crop. Each is projected to 24 channels by 1×1 convolution–GN(8)– SiLU and bilinearly aligned with align\_corners=False. The 72 aligned channels are concatenated with generated RGB, input RGB, and the 19 interface probabilities. The residual mapper is

$$
\begin{array} { r l } & { \mathrm { C o n v } _ { 3 \times 3 } ( 9 7 , 9 6 ) \to \mathrm { G N } ( 8 ) \to \mathrm { S i L U } } \\ & { \quad \to \mathrm { C o n v } _ { 3 \times 3 } ( 9 6 , 9 6 ) \to \mathrm { G N } ( 8 ) \to \mathrm { S i L U } } \\ & { \quad \to \mathrm { C o n v } _ { 1 \times 1 } ( 9 6 , 1 9 ) . } \end{array}
$$

The last layer is zero-initialized. The projections and mapper contain 190,891 trainable parameters. Generator features are extracted under no\_grad; AdamW updates only HGEA in FP32 with learning rate $5 \times 1 0 ^ { - 4 }$ , weight decay $\mathrm { \dot { 1 } 0 ^ { - 4 } }$ , batch size one, constant learning rate, and gradient clipping at 1.0.

Let $q = \mathrm { s o f t m a x } ( { \mathbf { z } } ^ { I } + \Delta { \mathbf { z } } ^ { H } )$ . In addition to valid-pixel cross-entropy, training uses boundary cross-entropy, classweighted cross-entropy, and a fusion loss on non-boundary pixels with max<sub>k</sub> $p _ { I , k } \ge 0 . 9 0$ . The fusion target is the normalized mixture $0 . 2 5 p _ { I } + 0 . 7 5 q$ . A final conditional term suppresses motorcycle and pole probability on predefined safeclass pixels only for foreground-hard-negative crops containing no motorcycle ground truth. Empty eligible sets return a graph-connected zero. The fixed optimization curriculum follows.

Optimization curriculum. The 48,000-step run keeps the model and AdamW states continuous. Steps 1–8,000 use $\mathcal { L } _ { \mathrm { c e } }$ with uniform crops; steps 8,001–16,000 add $0 . 1 0 \mathcal { L } _ { \mathrm { b n d } } +$ $0 . 0 2 \mathcal { L } _ { \mathrm { f u s i o n } } .$ , still with uniform crops. Steps 16,001–32,000 replace cross-entropy by ${ \mathcal { L } } _ { \mathrm { w c e } }$ and use the target-aware mixture. Steps $3 2 , 0 0 1 { - } 4 8 , 0 0 0$ use $\mathcal { L } _ { \mathrm { w c e } } + 0 . 1 \bar { 2 } \mathcal { L } _ { \mathrm { b n d } } +$ $0 . 0 1 5 \mathcal { L } _ { \mathrm { f u s i o n } } \mathrm { { ^ - } 0 . 0 2 5 \mathcal { L } _ { \mathrm { f p } } }$ with the conservative mixture. The endpoint is fixed before validation evaluation.

Target-aware crop probabilities are 0.35 uniform, 0.25 severe recall, 0.18 thin/small, 0.12 stuf boundary, and 0.10 hard negative. Conservative probabilities are 0.45 uniform, 0.16 train/rider positive, 0.08 motorcycle positive, 0.16 foreground hard negative, 0.10 thin boundary, and 0.05 wall/fence/terrain. All phases resize the short side to 512, take a $5 1 2 \times 5 1 2$ crop, and flip horizontally with probability 0.5. No validation result selects a checkpoint.

## Evaluation Metrics

The thin/rare subset comprises wall, fence, pole, trafic light, trafic sign, terrain, rider, truck, bus, train, motorcycle, and bicycle. Adaptive ECE uses deterministic equal-mass rank bins; ignored pixels are excluded from all metrics and retained-risk calculations.

Segmentation metrics accumulate a 19×20 confusion matrix per image, where the extra prediction column absorbs any out-of-range decoded label so that an invalid prediction on a valid ground-truth pixel counts as a false negative rather than silently vanishing. Mean IoU averages per-class intersection over union over classes with nonzero support, and pixel accuracy is computed over valid pixels only. Boundary metrics operate on a semantic-boundary band obtained by four-neighbor label transitions dilated with a disk structuring element of the stated pixel radius; boundary accuracy, precision, recall, and F-score are all measured inside this band with ignore pixels excluded, and interior statistics use its complement. Codebook validity is audited by the nearestprototype Euclidean distance in 8-bit RGB, reporting the mean and 95th percentile distance and the fraction of pixels within tolerance.

<table><tr><td>Class</td><td>RGB prototype</td><td>Class</td><td>RGB prototype</td><td>Class</td><td>RGB prototype</td><td>Class</td><td>RGB prototype</td></tr><tr><td>road</td><td>(255, 36, 36)</td><td>sidewalk</td><td>(36, 180, 180)</td><td>building</td><td>(132, 180, 36)</td><td>wall</td><td>(36, 36, 255)</td></tr><tr><td>fence</td><td>(180, 255, 36)</td><td>pole</td><td>(228, 36, 255)</td><td>traffic light</td><td>(255, 180, 36)</td><td>traffic sign</td><td>(36, 132, 255)</td></tr><tr><td>vegetation</td><td>(36, 255, 36)</td><td>terrain</td><td>(255, 255, 132)</td><td>sky</td><td>(132,255, 255)</td><td>person</td><td>(228, 132, 132)</td></tr><tr><td>rider</td><td>(132, 36, 180)</td><td>car</td><td>(132, 132, 255)</td><td>truck</td><td>(255, 36, 132)</td><td>bus</td><td>(36, 255, 255)</td></tr><tr><td>train</td><td>(255, 132, 255)</td><td>motorcycle</td><td>(132, 255, 132)</td><td>bicycle</td><td>(36, 255, 132)</td><td></td><td></td></tr></table>

Table S2: Exact high-separation semantic-RGB codebook. Entries follow the row-major prediction-channel order and are fixed throughout training and inference.

<table><tr><td>Codebook</td><td>mIoU (%)↑</td><td>BF@3 (%)↑</td><td> $d _ { \mathrm { m i n } }$  ↑</td><td>Bnd. gap↑</td></tr><tr><td>CS palette</td><td>59.86</td><td>67.89</td><td>20.00</td><td>7.39</td></tr><tr><td>High-Sep. (ours)</td><td>60.68</td><td>78.70</td><td>89.04</td><td>27.12</td></tr></table>

Table S3: Fixed-endpoint Direct Interface comparison on Cityscapes val500. $d _ { \mathrm { m i n } }$ is the minimum inter-prototype distance; Bnd. gap is $d _ { 2 } - d _ { 1 }$ between the nearest and runner-up prototypes at boundary pixels. Distances use 8-bit RGB $\ell _ { 2 }$ units.

Probability metrics are computed in a single streaming pass. Top-label ECE uses B equal-width confidence bins with the reliability gap weighted by bin mass; Brier score and negative log-likelihood are averaged over all valid pixels after renormalizing each pixel distribution. Failure-ranking metrics (AUROC, AUPR, and the risk–coverage curve) are evaluated on a fixed budget of pixels per image selected by a token-seeded hash of the valid mask, so the same pixels are scored across every checkpoint and readout. The risk– coverage curve sorts pixels by ascending uncertainty and reports AURC as the mean retained risk over the sorted prefix; retained risk at coverage c (R@c) reads this curve at the corresponding prefix. Because all selective-prediction quantities reorder a fixed set of predictions, they never change the decoded segmentation.

Metric consistency check. As an implementation-level consistency check, we apply the fixed-codebook decoder and the metric definitions above to the complete Cityscapes val500 prediction set (228, 972, 480 valid pixels). With $\tau _ { I } ~ = ~ 9 0 0$ , fixed-codebook decoding yields 60.68% interface mIoU, and the corresponding HGEA predictions yield 72.07% final mIoU, matching the values reported in the main paper.

## Controlled Analysis of Hierarchical Refinement

## Matched Controls and Mechanistic Tests

Table S4 extends the main-paper matched ablation with all three single hierarchy levels and per-seed endpoints. Within each seed, learned controls share the fixed 36,000-step protocol, sample order, crops, and evaluation code; parametercount diferences are below 0.5%.

UC-Head receives the same three hierarchy levels and uses the same decoder and parameter count as ML-HGEA, but predicts final logits directly instead of an additive residual to $\mathbf { z } ^ { I }$ . It tests the constraint imposed by interface anchoring. Table S5 reports this capacity-matched endpoint.

UC-Head is 1.09 mIoU points higher in this singleseed run. Residual anchoring therefore preserves an explicit prototype-logit reference but is not, by itself, the source of the accuracy gain in this control.

Removal and shufling instead intervene on one fixed trained checkpoint over 100 validation images; they are not retrained models. Their purpose is to test whether the trained predictor uses hierarchical content and spatial correspondence, not to estimate retrained-model performance.

Table S6 answers a diferent question: removing hierarchical fields reduces mIoU by 8.14 points, and destroying their spatial correspondence reduces it by 13.54 points. The trained checkpoint therefore uses both the content and alignment of the hierarchy.

## Regional and Classwise Refinement Efects

We compare Direct Interface predictions with those produced by the fixed final HGEA checkpoint across all 500 Cityscapes validation images. The correction rate is the fraction of Direct Interface errors corrected by HGEA, whereas the regression rate is the fraction of initially correct Direct Interface predictions that become incorrect after refinement. Boundary pixels are defined using a radius-3 band around four-neighbor transitions in the ground-truth labels. Table S7 jointly reports these transition rates, regional calibration, and semantic-RGB prototype geometry.

Beyond the aggregate boundary gain reported in the main paper, classwise transition patterns are heterogeneous: vegetation has the highest correction rate (61.45%), whereas person, motorcycle, truck, and rider have negative net pixel changes. The aggregate improvement therefore does not imply uniform gains across classes.

Matched hierarchy comparison (seed 18313)
<table><tr><td>Model</td><td>Evidence</td><td>Params</td><td>mIoU (%)↑</td><td>Thin (%)↑</td><td>BF@3 (%)↑</td><td>ECE (%)↓</td><td>Brier↓</td><td>NLL↓</td></tr><tr><td>DI</td><td>Fixed prototype</td><td>0</td><td>60.68</td><td>48.76</td><td>78.70</td><td>5.69</td><td>0.144</td><td>1.143</td></tr><tr><td>OI-Ref</td><td>s, pI</td><td>≈190.9k</td><td>67.70</td><td>57.84</td><td>79.71</td><td>0.41</td><td>0.083</td><td>0.207</td></tr><tr><td>CM-Flat</td><td>x, s, pI</td><td>190,915</td><td>70.19</td><td>61.50</td><td>80.09</td><td>0.51</td><td>0.081</td><td>0.200</td></tr><tr><td> $\mathbf { S L - H G E A _ { \mathrm { f i n e } } }$ </td><td>Level 1</td><td>190,395</td><td>69.99</td><td>61.30</td><td>79.80</td><td>0.58</td><td>0.081</td><td>0.204</td></tr><tr><td> $\mathbf { S L - H G E A _ { \mathrm { { m i d } } } }$ </td><td>Level 2</td><td>190,763</td><td>71.14</td><td>62.56</td><td>80.89</td><td>0.45</td><td>0.077</td><td>0.185</td></tr><tr><td>SL-HGEAcoarse</td><td>Level 3</td><td>190,843</td><td>66.76</td><td>56.70</td><td>78.43</td><td>0.84</td><td>0.089</td><td>0.257</td></tr><tr><td>ML-HGEA</td><td>Levels 1–3</td><td>190,891</td><td>71.97</td><td>63.67</td><td>81.22</td><td>0.42</td><td>0.075</td><td>0.178</td></tr></table>

Per-seed mIoU (%)
<table><tr><td>Seed</td><td>OI-Ref</td><td>CM-Flat</td><td>SL-HGEAmid</td><td>ML-HGEA</td></tr><tr><td>18313</td><td>67.70</td><td>70.19</td><td>71.14</td><td>71.97</td></tr><tr><td>28313</td><td>68.39</td><td>69.66</td><td>70.54</td><td>71.23</td></tr><tr><td>38313</td><td>66.65</td><td>69.41</td><td>69.95</td><td>71.09</td></tr><tr><td>Mean</td><td>67.58</td><td>69.75</td><td>70.54</td><td>71.43</td></tr><tr><td>sample std.</td><td>±0.88</td><td>±0.40</td><td>±0.59</td><td>± 0.47</td></tr></table>

Table S4: Controlled refinement evidence on Cityscapes val500. The upper block is a capacity- and level-matched comparison for seed 18313; the lower block reports per-seed replication. Brier and NLL belong to the matched checkpoint, not the separately trained 72.07% canonical model.

<table><tr><td>Model</td><td>mIoU↑</td><td>Thin↑</td><td>BF@3↑</td><td>ECE↓</td><td>Brier↓</td><td>NLL↓</td></tr><tr><td>ML-HGEA</td><td>71.97</td><td>63.67</td><td>81.22</td><td>0.42</td><td>0.075</td><td>0.178</td></tr><tr><td>UC-Head</td><td>73.06</td><td>65.12</td><td>82.50</td><td>0.34</td><td>0.071</td><td>0.159</td></tr></table>

Table S5: Capacity-matched output-head parameterization on Cityscapes val500 for seed 18313. Both heads use the same hierarchy and decoder capacity.
<table><tr><td>Configuration</td><td>mIoU↑</td><td>Thin↑</td><td>BF@3↑</td><td>Brier↓</td><td>NLL↓</td></tr><tr><td>Aligned hierarchy</td><td>72.98</td><td>65.35</td><td>82.57</td><td>0.087</td><td>0.217</td></tr><tr><td>Hierarchy removed</td><td>64.84</td><td>54.52</td><td>78.40</td><td>0.123</td><td>0.390</td></tr><tr><td>Correspondence shuffled</td><td>59.44</td><td>51.36</td><td>71.77</td><td>0.170</td><td>0.451</td></tr></table>

Table S6: Fixed-checkpoint hierarchy interventions on 100 Cityscapes validation images. These are interventions on one trained model, not retrained controls.

## Observable Interface Analysis

## Sensitivity to the Interface Temperature $\tau _ { I }$

The temperature rescales interface confidence without changing the nearest-prototype decoding rule. Table S8 evaluates this efect around the fixed operating point.

The sweep requires neither retraining nor model reselection. mIoU changes by at most 0.079 points from the default and pixel disagreement remains below 0.19%, while probability quality responds more strongly to the softmax scale. BF@3 spans only 78.674–78.702%, and Brier spans 0.132– 0.160.

## Spatial Calibration and Prototype Geometry

The regional audit in Table S7 localizes the interface failure mode. Boundary ECE is 31.22% versus 2.36% in interiors; boundary colors are farther from their nearest prototype and have a smaller runner-up gap. These aligned observations connect mixed or ambiguous rendered colors to boundary miscalibration and motivate spatially aligned refinement. After HGEA, boundary ECE falls to 4.46% while boundary accuracy rises by 18.85 points. Across 10, 15, 20, 30, and 50 bins, $p _ { H }$ equal-width ECE remains in 0.403– 0.407% and adaptive ECE in 0.383–0.405%; no calibrator is fit. The 15-bin adaptive ECE is 0.405%. The corresponding one-vs-rest classwise ECE averages 0.122%; it is a diferent binary-calibration quantity and is not compared directly with top-label ECE.

## Fixed-Prediction Reliability Analysis

## C-IHD Definition and Component Analysis

For MSP uncertainty, Local-MSP, and interface– hierarchy disagreement (IHD), let g(u) = $[ \mathcal { U } _ { \mathrm { M S P } } ( \dot { u } ) , \mathcal { U } _ { \mathrm { l o c } } ( \dot { u } ) , \mathcal { D } _ { \mathrm { I H D } } ( u ) ] ^ { \top }$ . Local-MSP is $\textbf { a } ~ 5 \times 5$ spatial average of MSP uncertainty. IHD is the ρ-weighted Jensen–Shannon disagreement with $\rho = 0 . 8$ , normalized by binary entropy. The fixed Cityscapes readout is

$$
\mathcal { U } _ { \mathrm { C - I H D } } ( u ) = ( 1 , 0 . 5 , 0 . 2 ) \mathrm { D i a g } ( \pmb { \sigma } _ { \mathrm { t r } } ) ^ { - 1 } \big ( \mathbf { g } ( u ) - \pmb { \mu } _ { \mathrm { t r } } \big ) .\tag{S4}
$$

The training-prediction statistics are

$$
\begin{array} { r } { \pmb { \mu } _ { \mathrm { t r } } = ( 0 . 0 4 1 8 7 4 , 0 . 0 4 2 0 5 8 , 0 . 0 5 1 2 2 3 ) , } \\ { \pmb { \sigma } _ { \mathrm { t r } } = ( 0 . 1 1 5 4 0 2 , 0 . 1 0 6 0 8 1 , 0 . 1 7 3 9 7 1 ) . } \end{array}
$$

Validation predictions do not update these statistics or coeficients. The normalized IHD lies in [0, 1]: if a Bernoulli variable selects $p _ { H }$ with probability $\rho$ and $p _ { I }$ otherwise, the numerator is the mutual information between the selector and the sampled class and is therefore bounded by the selector entropy.

<table><tr><td></td><td colspan="3">RGB geometry</td><td colspan="2">Direct Interface  $p _ { I }$ </td><td colspan="2">Refined pH</td><td colspan="3">Pixel transition</td></tr><tr><td>Region</td><td>Mean d1</td><td>Median d1</td><td>Mean gap</td><td>Acc. (%)</td><td>ECE (%)</td><td>Acc. (%)</td><td>ECE (%)</td><td>Corr. (%)</td><td>Regr. (%)</td><td>Net (pp)</td></tr><tr><td>Global</td><td></td><td></td><td></td><td>91.40</td><td>5.69</td><td>95.07</td><td>0.41</td><td>48.68</td><td>0.57</td><td>+3.67</td></tr><tr><td>Boundary</td><td>67.56</td><td>69.99</td><td>27.12</td><td>54.92</td><td>31.22</td><td>73.77</td><td>4.46</td><td>48.03</td><td>5.09</td><td>+18.85</td></tr><tr><td>Interior</td><td>31.34</td><td>21.69</td><td>75.54</td><td>96.19</td><td>2.36</td><td>97.86</td><td>0.42</td><td>49.70</td><td>0.23</td><td>+1.67</td></tr></table>

Table S7: Regional failure anatomy on Cityscapes val500. Prototype distances are measured on continuous RGB outputs before quantization; the geometry audit covers the boundary and interior partitions. Accuracy and equal-width 15-bin ECE use the same valid pixels. Correction and regression are conditioned on initially wrong and initially correct pixels, respectively.

<table><tr><td>TI</td><td>mIoU (%)↑</td><td>Conf. (%)↑</td><td>ECE (%)↓</td><td>NLL↓</td><td>Disagr. (%)</td></tr><tr><td>225</td><td>60.702</td><td>98.576</td><td>7.170</td><td>1.787</td><td>0.139</td></tr><tr><td>450</td><td>60.698</td><td>98.140</td><td>6.732</td><td>1.502</td><td>0.068</td></tr><tr><td>675</td><td>60.690</td><td>97.640</td><td>6.235</td><td>1.300</td><td>0.030</td></tr><tr><td>900</td><td>60.681</td><td>97.089</td><td>5.691</td><td>1.143</td><td>0.000</td></tr><tr><td>1200</td><td>60.669</td><td>96.267</td><td>4.877</td><td>0.981</td><td>0.035</td></tr><tr><td>1800</td><td>60.650</td><td>94.153</td><td>2.780</td><td>0.766</td><td>0.089</td></tr><tr><td>3600</td><td>60.602</td><td>82.565</td><td>9.039</td><td>0.591</td><td>0.185</td></tr></table>

Table S8: Direct Interface temperature sweep on Cityscapes val500. Bold marks the fixed $\tau _ { I } = 9 0 0$ interface; disagreement is measured against it.

## Comparison of Fixed-Prediction Readouts

Table S9 compares six uncertainty readouts on the same fixed $p _ { H }$ predictions for Cityscapes val500. Holding the segmentation fixed isolates pixel-error ranking from changes in predictive accuracy. AURC and retained risk at coverage c (R@c) are reported in units of $1 0 ^ { - 3 }$

C-IHD obtains the lowest AURC (4.235) and the lowest retained risk at 70%, 90%, and 95% coverage, whereas Local-MSP is marginally better at 50% coverage. IHD alone, without the MSP and Local-MSP components, gives the weakest ranking. Relative to standalone MSP, C-IHD reduces AURC by $0 . 0 \bar { 4 } 6 \times 1 0 ^ { - 3 }$ , corresponding to an approximately 1.1% relative reduction in AURC.

On the same predictions, a 10,000-replicate paired imagelevel bootstrap over all 500 images gives 95% C-IHD– MSP gain intervals of $\left\lceil 0 . 3 7 0 , 1 . 0 \dot { 2 } 2 \right\rceil \times 1 0 ^ { - 3 }$ for AUROC and $[ 2 . 0 8 5 , 4 . 0 5 6 ] \times 1 0 ^ { - \dot { 3 } }$ for AUPR. These intervals quantify image-level sampling variation for the fixed checkpoint rather than seed-to-seed training variability. Together with Table S9 and Figure S1, the results support complementarity among the fixed readouts rather than a strong standalone IHD detector.

## Prespecified Sensitivity Analysis

One factor is varied at a time around the fixed default. We test $\rho ~ \in ~ \{ 0 . 2 0 , 0 . 5 0 , 0 . 6 5 , 0 . 8 0 , 0 . 9 0 , 0 . 9 5 \}$ , local weights in {0, 0.25, 0.50, 0.75, 1.00}, IHD weights in {0, 0.10, 0.20, 0.30, 0.40}, local windows in {1, 3, 5, 7, 11}, and the readout with and without training-set standardization. Across the grid, performance remains stable: AUROC ranges from 0.9451 to 0.9458, AUPR from 0.4773 to 0.4814, and AURC from $4 . 2 3 2 \times 1 0 ^ { - 3 } \mathrm { ~ t o ~ } 4 . 2 7 3 \times 1 0 ^ { - 3 }$ . Performance varies only modestly around the prespecified default;

![](images/db5198e529a064853c43202badffc16c323326869774e4d6d3e2fb6b3954df7f.jpg)

![](images/c427b9fbb090ad373f284163cb21567a3a7391d96ee7a581698992af5f90b2d3.jpg)  
Figure S1: Calibration and fixed-prediction reliability on Cityscapes val500. (a) Calibration residuals (accuracy minus confidence) for $p _ { I }$ and $p _ { H }$ . (b) Retained-risk diferences for C-IHD and Local-MSP relative to MSP on the same fixed $p _ { H } ;$ negative values denote lower risk. C-IHD yields lower retained risk over the 70–95% coverage range, whereas Local-MSP is marginally lower at 50% coverage. Because $p _ { H }$ is held fixed in (b), these diferences reflect error ranking rather than changes in the segmentation.

the sweep is descriptive and does not reselect the readout configuration.

## Transfer and Deployment Audit

## Source-Frozen ACDC Diagnostics

Table S10 compares four uncertainty readout functions on source-frozen ACDC val406 predictions, isolating errorordering quality without target-domain adaptation.

C-IHD achieves the lowest AURC (43.079 in $1 0 ^ { - 3 }$ units), improving by $6 . 6 0 9 \times 1 0 ^ { - 3 }$ over standalone MSP. Adding IHD or Local-MSP to MSP yields marginal AURC gains (0.352 and 0.406 respectively), whereas the full C-IHD combination delivers the strongest improvement in both AUROC (0.9076) and AUPR (0.7557).

<table><tr><td>Readout</td><td>AUROC↑</td><td>AUPR↑</td><td> $\mathrm { A U R C } ( \times 1 0 ^ { - 3 } ) \downarrow$ </td><td> $\mathsf { R } @ 5 0 ( \times 1 0 ^ { - 3 } ) \downarrow$ </td><td> $\mathbf { R } @ 7 0 ( \times 1 0 ^ { - 3 } ) \downarrow$ </td><td> $\mathsf { R } @ 9 0 ( \times 1 0 ^ { - 3 } ) \downarrow$ </td><td> $\mathsf { R } @ 9 5 ( \times 1 0 ^ { - 3 } ) \downarrow$ </td></tr><tr><td>MSP</td><td>0.94502</td><td>0.47814</td><td>4.281</td><td>0.66</td><td>1.96</td><td>13.77</td><td>26.14</td></tr><tr><td>Entropy</td><td>0.94235</td><td>0.45043</td><td>4.417</td><td>0.66</td><td>1.98</td><td>14.28</td><td>27.98</td></tr><tr><td>Margin</td><td>0.94456</td><td>0.45459</td><td>4.301</td><td>0.65</td><td>1.95</td><td>13.75</td><td>26.31</td></tr><tr><td>Local-MSP</td><td>0.94339</td><td>0.46672</td><td>4.355</td><td>0.61</td><td>1.90</td><td>14.59</td><td>26.88</td></tr><tr><td>IHD only</td><td>0.92958</td><td>0.30143</td><td>5.080</td><td>0.63</td><td>2.02</td><td>17.37</td><td>34.11</td></tr><tr><td>C-IHD</td><td>0.94572</td><td>0.48121</td><td>4.235</td><td>0.62</td><td>1.87</td><td>13.65</td><td>26.05</td></tr></table>

Table S9: Pixel-error ranking at fixed $p _ { H }$ on Cityscapes val500.

<table><tr><td>Readout</td><td>AUROC↑</td><td>AUPR↑</td><td> $\mathrm { A U R C } ( \times 1 0 ^ { - 3 } ) \downarrow$ </td></tr><tr><td>MSP</td><td>0.8903</td><td>0.6580</td><td>49.688</td></tr><tr><td> $\mathrm { M S P + I H D }$ </td><td>0.8910</td><td>0.6554</td><td>49.336</td></tr><tr><td>MSP + Local-MSP</td><td>0.8919</td><td>0.6657</td><td>49.282</td></tr><tr><td>C-IHD</td><td>0.9076</td><td>0.7557</td><td>43.079</td></tr></table>

Table S10: Source-frozen ACDC readout comparison.

Table S11 reports the per-condition breakdown across fog, night, rain, and snow using the C-IHD readout.

<table><tr><td colspan="4">Condition N mIoU  $( \% )$  Error (%) AUROC</td></tr><tr><td>Overall</td><td>406</td><td>46.89</td><td>AUPR 19.69 0.9076 0.7557</td></tr><tr><td>Fog</td><td>100</td><td>58.15</td><td>11.85 0.9134 0.7017</td></tr><tr><td>Night</td><td>106</td><td>27.81 35.10</td><td>0.8521 0.7711</td></tr><tr><td>Rain</td><td>100</td><td>51.01 9.18</td><td>0.9186 0.6505</td></tr><tr><td>Snow</td><td>100</td><td>44.47 22.16</td><td>0.9076 0.8032</td></tr></table>

Table S11: ACDC per-condition breakdown (source-frozen, C-IHD readout).

Night exhibits the largest segmentation degradation (27.81% mIoU) and the highest pixel error rate (35.10%), yet achieves the second-highest AUPR (0.7711). Snow reaches the highest AUPR (0.8032) despite 22.16% error rate. These elevated AUPR values partly reflect higher error prevalence rather than superior ranking quality; C-IHD remains the most efective readout across all four conditions, with AUROC ranging from 0.8521 (night) to 0.9186 (rain).

## Computational Cost and Adaptation Overhead

Table S12 reports end-to-end inference cost and adaptation accounting on one NVIDIA A100 80GB PCIe at batch size one with FP32/TF32 and no autocast.

Each timing averages 100 frames after 10 warm-ups and includes preprocessing, host-to-device transfer, model execution, and probability stitching at 1024×512. The timing audit reports 635.04 ms for the full three-window path. The one-step generator pipeline is faster than the 20-step DDPS baseline but slower and more memory-intensive than the three-step DDP-CNXT-T baseline. For the complete adaptation, HGEA contributes 0.191M trainable parameters and adds 29.54 ms (4.65% of the full path), while C-IHD adds

End-to-end inference on A100 80GB
<table><tr><td>Method</td><td>Steps/crop</td><td>FPS↑</td><td>GiB↓</td></tr><tr><td>DDPS-B0 (20-step)</td><td>60</td><td>1.55</td><td>32.47</td></tr><tr><td>DDP-CNXT-T (3-step)</td><td>9</td><td>3.61</td><td>8.08</td></tr><tr><td>GSS (surrogate)</td><td>3</td><td>3.02</td><td>29.65</td></tr><tr><td>LoRA (unmerged)</td><td>9.002M</td><td>0.698</td><td>128.93 (20.30)</td></tr><tr><td>Semantic Prism</td><td>3</td><td>1.92</td><td>29.65</td></tr></table>

<table><tr><td colspan="5">Adaptation accounting</td></tr><tr><td>Component</td><td>Trainable</td><td>Total</td><td>Core%</td><td>∆ms (share)</td></tr><tr><td>LoRA U-Net skips</td><td>8.510M</td><td>8.510M</td><td>0.660</td><td>96.56 (15.20)</td></tr><tr><td>LoRA VAE skips</td><td>0.492M</td><td>0.492M</td><td>0.038</td><td>1.85 (0.29)</td></tr><tr><td>U-Net input conv.</td><td>0</td><td>0.012M</td><td>0.001</td><td>0</td></tr><tr><td>HGEA</td><td>0.191M</td><td>0.191M</td><td>0.015</td><td>29.54 (4.65)</td></tr><tr><td>C-IHD</td><td>0</td><td>0</td><td>0</td><td>1.16 (0.18)</td></tr><tr><td>Full adaptation</td><td>9.684M</td><td>9.696M</td><td>0.752</td><td></td></tr></table>

Table S12: Computational cost audit. Timing shares relative to the 635.04-ms full three-window path.

1.16 ms (0.18%) without introducing trainable weights. Most adapted parameters belong to the 9.002M unmerged LoRA weights; a tested merged-LoRA variant failed the prespecified numerical-equivalence criterion and is therefore omitted. The GSS hard-probability surrogate remains only as a timing reference.