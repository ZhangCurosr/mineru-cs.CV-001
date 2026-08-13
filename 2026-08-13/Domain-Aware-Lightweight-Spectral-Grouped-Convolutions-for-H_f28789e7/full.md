# Domain-Aware Lightweight Spectral-Grouped Convolutions for Hyperspectral Fish Freshness Classification

Kazi Nabiul Alam

K.Alam3742@student.leedsbeckett.ac.uk

Pooneh Bagheri Zadeh P.Bagheri-Zadeh@leedsbeckett.ac.uk

Akbar Sheikh-Akbari A.Sheikh-Akbari@leedsbeckett.ac.uk

School of Built Environment, Engineering, and Computing, Leeds Beckett University. Leeds LS6 3QS, United Kingdom.

## Abstract

Hyperspectral imaging (HSI) offers nondestructive assessment of fish freshness by detecting biochemical alterations across spectral bands. However, conventional deep learning approaches do not fully address the particular characteristics of HSI data, such as spectral dominance over spatial textures, ordinal label structure, and a small number of training samples. We propose SGNet (Spectral-Grouped Network), a lightweight architecture that separates spectral and spatial feature extraction using grouped convolutions and a depthwise spatial pathway. A dual attention mechanism that couples channel-wise squeeze-and-excitation with spatial gating adaptively highlights informative features. SGNet achieves 97.8% classification accuracy and 0.64 days mean absolute error (MAE) with just 4.75M parameters when tested on our newly developed 16-day refrigerator-stored salmon fillet dataset. Ablation studies validate the contribution of each component, while comparisons demonstrate a five- to eighteen-fold parameter reduction relative to ResNet-50 and Vision Transformers. Our findings indicate that domain-aware design supports precise, real-time freshness prediction for industrial implementation.

## Introduction

<sup>1</sup>Fish freshness is a significant determinant of nutritional content, consumer safety, and mar-8ket value in the global seafood industry [1, 2]. Conventional assessment techniques, such <sup>0</sup>as sensory evaluation, microbiological testing, and chemical analysis, are variously subjec-<sup>6</sup>tive, destructive, or time-intensive, which makes them inappropriate for real-time industrial use [3, 4]. The growing body of research into sensing technologies reflects a clear demand <sup>v</sup>for automated, non-destructive, and rapid quality inspection.

X Hyperspectral imaging (HSI) has emerged as an effective tool for evaluating food quality rby integrating spatial imaging with extensive spectral data acquired across numerous con-<sup>a</sup>tiguous wavelength bands [5, 6]. An HSI system operates differently from standard RGB imaging because it is able to identify chemical breakdown products through the analysis of specific spectral bands [7, 8]. These traits make HSI especially useful for estimating fish freshness, since the quality of fish degrades gradually and continuously over time, leaving a measurable spectral signature at each stage of spoilage.

Although classical machine learning methods such as partial least squares discriminant analysis (PLS-DA) and support vector machines (SVM) have demonstrated potential for HSI-based freshness evaluation [9, 10], they depend heavily on manually designed features and prior wavelength selection [11], which limits their capacity to capture detailed spectralspatial interactions. Deep learning techniques have achieved superior results in hyperspectral image classification according to recent research [12], yet most established models were designed for remote sensing tasks. Such models do not adequately accommodate the characteristics of food quality data, which combine large spectral dimensionality with restricted sample availability and an inherently ordinal label structure. A further difficulty, frequently overlooked, is the substantial redundancy between neighbouring spectral bands; indiscriminate mixing of all bands therefore risks entangling correlated and unrelated wavelength responses, which amplifies noise and encourages overfitting under limited data regimes.

This study introduces SGNet (Spectral-Grouped Convolutional Neural Network), a framework designed around the structure of hyperspectral food data rather than adapted from a remote sensing or natural image backbone. Rather than proposing new primitives, our contribution is a principled composition of efficient operators that jointly respects spectral dominance, ordinal label structure, and severe sample scarcity, together with the analysis and dataset needed to validate it in this regime. Our contributions are as follows:

• An explicit spectral-spatial factorisation in which grouped pointwise convolution confines cross-band interaction to local channel subsets while a parallel depthwise pathway captures spatial structure, preventing premature entanglement of unrelated wavelength responses.

• A lightweight dual attention mechanism coupling channel-wise excitation with spatial gating at below 5% of the block parameters, giving selective emphasis without the data appetite of self-attention.

• A hierarchical encoder, formalised in Algorithm 1, that attains 97.8% accuracy with only 4.75M parameters and is benchmarked against widely used convolutional and transformer architectures retrained under identical conditions.

• A curated 16-day refrigerated salmon dataset with strict pack-level separation, addressing the scarcity of leakage-controlled hyperspectral freshness data, which we release to support reproducibility.

## 2 Related Work

## 2.1 Hyperspectral Imaging for Fish Freshness

HSI has been applied extensively to fish freshness assessment in recent years. Cheng et al. [1] employed HSI with least squares SVM to predict total volatile basic nitrogen (TVB-N) content in grass carp, while Yu et al. [2] combined HSI with data fusion for rapid evaluation of tilapia fillet freshness. Falahatnejad et al. [3] surveyed fish quality assessment using HSI and computer vision, distinguishing the conventional pipelines that dominate the field. Hardy et al. [13] applied K-nearest neighbours to hyperspectral data for salmon storage-day prediction, achieving 77% accuracy. Kashani Zadeh et al. [14] developed a multimode spectroscopic system that combined VIS-NIR, SWIR, and fluorescence data for salmon freshness classification, and Khoshnoudi-Nia and Moosavi-Nasab [4] predicted multiple freshness indicators in fish fillets from a single multispectral imaging system. These studies establish that HSI data carry rich freshness information, yet they predominantly rely on conventional machine learning pipelines that require human intervention to engineer features and select informative wavelengths.

## 2.2 Deep Learning for Hyperspectral Classification

Deep learning has transformed hyperspectral image analysis through its capacity for hierarchical feature extraction. Early CNN-based methods [15, 16] used 3D convolutional layers to fuse spectral and spatial information jointly, but their high computational requirements limited their practicality. Roy et al. [17] subsequently proposed HybridSN, a hybrid 3D– 2D convolutional design that reduces the cost of pure 3D models while retaining strong spectral-spatial discrimination. In parallel, attention mechanisms have proven highly effective: Hu et al. [18] introduced squeeze-and-excitation (SE) blocks for channel recalibration, and Woo et al. [19] proposed CBAM, which unifies channel and spatial attention. For HSI specifically, spectral and recurrent attention networks [12, 20] have achieved state-of-the-art results on remote sensing benchmarks. A recurring limitation is that these systems are calibrated for large-scale data collections and incur considerable computational cost, which is poorly matched to the small-sample, high-dimensional regime that characterises food quality inspection.

## 2.3 Efficient Architectures and Ordinal Modelling

Advances in efficient network design provide the primitives on which SGNet builds. Howard et al. [21] introduced depthwise separable convolutions in MobileNet, reducing computation by roughly eight to nine times relative to standard convolution. Tan and Le [22] formalised compound scaling in EfficientNet to balance depth, width, and resolution. Liu et al. [23] revisited the convolutional design space with ConvNeXt and showed that pure ConvNets can rival Vision Transformers [24], while hierarchical transformers such as Swin [25] brought locality and multi-scale structure to attention-based models. Grouped convolutions, popularised by ResNeXt [26], occupy a middle ground between standard and depthwise convolution by partitioning channels into groups. A complementary line of work addresses the ordinal nature of many prediction tasks: Cao et al. [27] proposed the rank-consistent CORAL framework for ordinal regression, and Lin et al. [28] introduced focal loss to address class imbalance, both of which are directly relevant to freshness estimation where labels evolve continuously over time. SGNet draws on these efficient primitives but arranges them according to the specific structure of hyperspectral food data rather than transplanting a remote sensing or natural image backbone unchanged.

A concise comparison of representative approaches is given in Table 1, which contrasts the input modality, the feature extraction strategy, the treatment of label ordinality, and the relative parameter scale. The table makes clear that prior HSI freshness work tends either to rely on hand-crafted features or to inherit heavyweight backbones, leaving a gap for a lightweight, domain-aware design that respects spectral dominance and ordinal structure simultaneously.

Table 1: Comparison of representative approaches to HSI-based food quality assessment and related deep architectures. “Hand-crafted” denotes manual feature or wavelength selection; “Eval.” denotes ordinality handled at evaluation time; “–” denotes not applicable.
<table><tr><td>Method</td><td>Input</td><td>Feature Strategy</td><td>Ordinal-Aware</td><td>Scale</td></tr><tr><td>Classical SVM [0, ] HSI / spectra</td><td></td><td>Hand-crafted</td><td>No</td><td></td></tr><tr><td>KNN salmon [[3]</td><td>HSI</td><td>Hand-crafted</td><td>No</td><td></td></tr><tr><td>3D-CNN [,]</td><td>HSI cube</td><td>Learned (3D)</td><td>No</td><td>Heavy</td></tr><tr><td>HybridSN []</td><td>HSI cube</td><td>Learned (3D-2D)</td><td>No</td><td>Medium</td></tr><tr><td>SE / CBAM [, ]</td><td>Image</td><td>Learned + attention</td><td>No</td><td>Medium</td></tr><tr><td>SpectralFormer [2]</td><td>HSI</td><td>Transformer</td><td>No</td><td>Heavy</td></tr><tr><td>ConvNeXt [[3]</td><td>Image</td><td>Learned (ConvNet)</td><td>No</td><td>Heavy</td></tr><tr><td>SGNet (Ours)</td><td>HSI cube</td><td>Grouped + DW + attention</td><td>Eval.</td><td>Light</td></tr></table>

![](images/3ba33fcf7556f0d0f18c88187a21058b7a45a8d71c21b589fbe61e7b4d817320.jpg)  
Figure 1: Proposed SGNet framework for hyperspectral fish freshness classification.

## 3 Methodology

## 3.1 Problem Setting and Notation

Let $\mathcal { D } = \{ ( X _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ denote a hyperspectral fish freshness dataset, where $X _ { i } \in \mathbb { R } ^ { C \times H \times W }$ represents a hyperspectral image cube with C spectral bands, and $y _ { i } \in \{ 0 , 1 , . . . , K - 1 \}$ denotes the freshness day label. Unlike standard image classification, the labels are ordinal, reflecting the continuous temporal evolution of freshness rather than a set of mutually independent categories.

The objective is to learn a mapping

$$
f _ { \theta } : \mathbb { R } ^ { C \times H \times W }  \mathbb { R } ^ { K } ,\tag{1}
$$

in which prediction errors are penalised not only in a categorical sense but also with respect to the ordinal distance between predicted and true freshness days [27].

## 3.2 Design Rationale

The hyperspectral fish freshness classification task presents three principal obstacles. First, spectral dominance means that freshness information is carried primarily by wavelengthdependent reflectance patterns rather than by spatial texture. Second, temporal smoothness requires that representations of neighbouring days remain close in feature space, so that an error of one day is treated as far less severe than an error of several days. Third, the available data are limited in number and are subject to additional sources of variation and noise. Generic CNNs that combine spectral and spatial information without regard for these constraints perform poorly, because they entangle correlated bands prematurely and overfit to spurious patterns. The proposed design therefore contains two independent processing paths that use grouped convolutions together with lightweight attention to suppress unimportant variation while performing spectral and spatial interaction in a controlled manner.

<table><tr><td colspan="3">Table 2: SGNet Architecture Overview</td></tr><tr><td>Stage</td><td>Layers</td><td>Output Size Params</td></tr><tr><td>Stem Stage 1</td><td>Spec. Proj. + Patch Emb.  $\overline { { 6 4 \times 5 6 ^ { 2 } } }$ </td><td>74K</td></tr><tr><td> $\mathrm { S G N e t } \times 3 + \mathrm { D o w n }$ </td><td> $1 2 8 \times 2 8 ^ { 2 }$ </td><td>429K</td></tr><tr><td>Stage 2</td><td>SGNet ×3 + Down  $2 5 6 \times 1 4 ^ { 2 }$ </td><td>1.31M</td></tr><tr><td>Stage 3</td><td> ${ \mathrm { S G N e t ~ } } \times 4$   $2 5 6 \times 1 4 ^ { 2 }$ </td><td>2.89M</td></tr><tr><td>Head</td><td> $\mathrm { G A P + L i n e a r }$ </td><td>4.1K</td></tr><tr><td>Total</td><td>K</td><td>4.75M</td></tr></table>

Table 2 and Figure 1 summarise the SGNet architecture.

## 3.3 Spectral Projection and Patch Embedding

We first perform a linear spectral projection with a point-wise convolution on the hyperspectral cube $\boldsymbol { X } \in \mathbb { R } ^ { C \times H \times W }$ :

$$
\begin{array} { r } { \boldsymbol { X } ^ { ( 0 ) } = \mathbf { L } \mathbf { N } \left( W _ { 0 } * \boldsymbol { X } \right) , \quad W _ { 0 } \in \mathbb { R } ^ { C _ { 0 } \times C \times 1 \times 1 } , } \end{array}\tag{2}
$$

where $C _ { 0 } = 6 4$ and LN(·) denotes channel-wise layer normalisation. This operation performs controlled spectral mixing, compressing the C-dimensional spectral signature into a learned C<sub>0</sub>-dimensional embedding while preserving full spatial resolution.

To obtain spatially efficient representations, a strided convolution is subsequently applied:

$$
\begin{array} { r } { X ^ { ( 1 ) } = \mathbf { L N } \left( W _ { 1 } * X ^ { ( 0 ) } \right) , \quad W _ { 1 } \in \mathbb { R } ^ { 6 4 \times 6 4 \times 4 \times 4 } , } \end{array}\tag{3}
$$

yielding feature maps of size $6 4 \times \frac { H } { 4 } \times \frac { W } { 4 }$ . Unlike the pooling-based patchification common in vision transformers [24], this learnable downsampling preserves spectral fidelity while enabling early spatial abstraction.

## 3.4 Spectral–Grouped CNN Block

The core computational unit is the Spectral–Grouped CNN (SGNet) block, which processes spectral and spatial information through dedicated pathways. Let $X \in \mathbb { R } ^ { D \times h \times w }$ denote the input feature map. Each block comprises four sequential operations.

## 3.4.1 Spectral Mixing

Cross-channel interaction is performed via grouped pointwise convolution:

$$
S ( X ) = \phi \left( \operatorname { L N } \left( W _ { s } ^ { ( g ) } * X \right) \right) ,\tag{4}
$$

where $W _ { s } ^ { ( g ) }$ denotes a $1 \times 1$ convolution with $g = 8$ groups, $\phi ( \cdot )$ is GELU activation, and the output is further processed by a two-layer MLP with expansion ratio 2. Grouped convolution constrains cross-band interaction to local channel subsets, preventing premature entanglement of unrelated spectral features and reducing sensitivity to noise.

## 3.4.2 Spatial Mixing

Local spatial context is aggregated independently per channel using depthwise separable convolution:

$$
\begin{array} { r } { P ( X ) = W _ { p } * \left( W _ { d } * X \right) , } \end{array}\tag{5}
$$

where $W _ { d } \in \mathbb { R } ^ { D \times 1 \times k \times k }$ is a depthwise convolution with kernel size $k = 7$ , and $W _ { p } \in \mathbb { R } ^ { D \times D \times 1 \times 1 }$ is a pointwise projection. This factorisation captures local tissue structure and spatial patterns while preserving spectral independence.

## 3.4.3 Dual Attention Gating

To adaptively emphasise informative spectral responses and spatial regions, we introduce a lightweight dual attention mechanism inspired by squeeze-and-excitation [18] and convolutional attention [19]. Channel-wise attention is computed via squeeze-and-excitation:

$$
g _ { c } ( X ) = \sigma \left( W _ { c 2 } \cdot \delta \left( W _ { c 1 } \cdot \operatorname { G A P } ( X ) \right) \right) ,\tag{6}
$$

where $\mathrm { G A P ( \cdot ) }$ denotes global average pooling, $\delta ( \cdot )$ is ReLU activation, and the bottleneck reduction ratio is $r = 4$ . Spatial attention is computed as:

$$
g _ { s } ( X ) = \sigma \left( W _ { a } * X \right) ,\tag{7}
$$

where $W _ { a } \in \mathbb { R } ^ { 1 \times D \times 1 \times 1 }$ produces a spatial attention map. The gated output combines both attention mechanisms:

$$
\tilde { X } = X \odot g _ { c } ( X ) \odot g _ { s } ( X ) ,\tag{8}
$$

where $\odot$ denotes element-wise multiplication with appropriate broadcasting. Unlike selfattention, this mechanism introduces minimal computational overhead (less than 5% of block parameters) and remains stable under limited data regimes.

## 3.4.4 Feed-Forward Network

The final output is obtained via a residual feed-forward transformation:

$$
Y = X + \mathrm { F F N } \left( \mathrm { L N } ( { \tilde { X } } ) \right) ,\tag{9}
$$

where $\mathrm { F F N } ( \cdot )$ consists of two pointwise convolutions with expansion ratio 4 and GELU activation:

$$
\mathrm { F F N } ( Z ) = W _ { 2 } \ast \phi \left( W _ { 1 } \ast Z \right) , \quad W _ { 1 } \in \mathbb { R } ^ { 4 D \times D } , W _ { 2 } \in \mathbb { R } ^ { D \times 4 D } .\tag{10}
$$

## 3.5 Hierarchical Encoder

The complete network stacks SGNet blocks across three stages with progressively increasing channel dimensions {64,128,256} and correspondingly decreasing spatial resolutions $\{ \bar { 5 6 } ^ { 2 } , 2 8 ^ { 2 } , 1 4 ^ { 2 } \}$ . Block depths are set to {3,3,4} for stages 1–3 respectively. Downsampling between stages is performed via layer-normalised strided convolution:

$$
X _ { l + 1 } = \operatorname { C o n v } _ { 2 \times 2 , s = 2 } \left( \operatorname { L N } ( X _ { l } ) \right) .\tag{11}
$$

This hierarchical design enables the model to capture freshness-related cues across multiple spatial scales, ranging from fine-grained local tissue variation to coarse global structural change indicative of degradation. The progressive structure mirrors the multi-scale philosophy of hierarchical transformers [25] while retaining the efficiency of convolutional primitives.

## 3.6 Classification Head

Global average pooling aggregates the final-stage features into a location-invariant representation:

$$
z = \mathbf { G } \mathbf { A } \mathbf { P } ( X _ { L } ) \in \mathbb { R } ^ { 2 5 6 } .\tag{12}
$$

A linear classifier produces logits for the K ordinal freshness classes:

$$
\begin{array} { r } { \hat { y } = W _ { o } z + b _ { o } , \quad W _ { o } \in \mathbb { R } ^ { K \times 2 5 6 } . } \end{array}\tag{13}
$$

The model is trained with standard cross-entropy loss, which we adopt as a deliberately simple and strong baseline; this isolates the effect of the architecture from that of the loss. We evaluate using ordinal-aware metrics including mean absolute error (MAE), accuracy (Acc), and quadratic weighted kappa (QWK), which better reflect the continuous nature of freshness evolution and penalise predictions in proportion to their distance from the ground truth. That the architecture alone yields strong ordinal behaviour under a non-ordinal loss is itself informative; we discuss the integration of an explicit rank-consistent objective [27] as a direction for future work in Section 5.

## 3.7 The SGNet Forward Pass

For clarity, Algorithm 1 collects the operations defined above into a single end-to-end procedure. The listing makes explicit the order in which spectral and spatial information are processed and shows that the two pathways remain decoupled until the dual attention stage recombines them. We regard this controlled separation, rather than the early joint mixing favoured by generic spectral-spatial networks, as the principal source of the model’s sample efficiency.

## 4 Experiments

## 4.1 Experimental Setup

Dataset and Implementation. Our curated salmon hyperspectral dataset [29] consists of N = 800 HSI cubes acquired over a 16-day refrigerated storage period (K = 16 classes), where day 6 is the labelled expiry date. Each cube comprises $C = 4 6 2$ spectral bands spanning the visible and near-infrared range. The dataset is split into training (560), validation (112), and test (128) sets with balanced class distributions. To avoid data leakage, distinct fish packs were assigned to each split (35 packs for training, 7 for validation, and 8 for testing), so that no cube from a given pack appears in more than one split.

Algorithm 1 SGNet forward pass for a hyperspectral cube.   
Require: Cube $\overline { { \boldsymbol { X } \in \mathbb { R } ^ { C \times H \times W } } }$ ; stage depths $\{ n _ { 1 } , n _ { 2 } , n _ { 3 } \} = \{ 3 , 3 , 4 \}$ ; widths {64,128,256}   
Ensure: Ordinal class logits $\hat { y } \in \bar { \mathbb { R } } ^ { K }$   
1: $X ^ { ( 0 ) } \gets \mathrm { L N } ( W _ { 0 } * X )$ ▷ spectral projection to $C _ { 0 } { = } 6 4$   
2: $X  \mathrm { L N } ( W _ { 1 } * X ^ { ( 0 ) } )$ ▷ strided patch embedding, stride 4   
3: for stage $s = 1$ to 3 do   
4: for block = 1 to $n _ { s }$ do   
5: $S \gets \phi ( \mathrm { L N } ( W _ { s } ^ { ( g ) } * X ) )$ ▷ grouped spectral mixing, $g { = } 8$   
6: $S \gets \mathbf { M L P } _ { 2 } ( S )$ ▷ expansion ratio 2   
7: $P \gets W _ { p } * \left( W _ { d } * S \right)$ ▷ depthwise spatial mixing, k=7   
8: $g _ { c } \gets \dot { \sigma } ( W _ { c 2 } \delta ( W _ { c 1 } \mathrm { G A P } ( P ) ) )$ ▷ channel attention   
9: $g _ { s } \gets \sigma ( W _ { a } * P )$ ▷ spatial attention   
10: $\tilde { X }  P \odot g _ { c } \odot g _ { s }$ ▷ dual gating   
11: $X  X + \mathrm { F F N } ( \mathrm { L N } ( \tilde { X } ) )$ ▷ residual FFN, ratio 4   
12: end for   
13: if $s < 3$ then   
14: $X \gets \mathrm { C o n v } _ { 2 \times 2 , s = 2 } ( \mathrm { L N } ( X ) )$ ▷ downsample   
15: end if   
16: end for   
17: $z \gets \mathbf { G A P } ( X )$ ▷ location-invariant embedding   
18: $\hat { y }  W _ { o } z + b _ { o }$ ▷ ordinal classification head   
19: return $\hat { y }$

Table 3: Implementation Details
<table><tr><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td></tr><tr><td>Optimiser</td><td>AdamW</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>Epochs</td><td>40</td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>Precision</td><td>FP16</td></tr><tr><td>LR schedule</td><td>Cosine</td><td>Augmentation</td><td>Random crop</td></tr></table>

Table 3 summarises the training configuration. All experiments use PyTorch 2.0 on a single NVIDIA A100 GPU. Baselines were trained under the same data splits, augmentation policy, and optimisation schedule as SGNet to ensure a fair comparison. The dataset and splits will be released to support reproducibility. Because pack identity is disjoint across splits and each cube is acquired under a fixed illumination and geometry, the discriminative signal must be carried by spectral reflectance rather than by pack-specific or acquisition artefacts.

Evaluation Metrics. We report top-1 accuracy and macro-averaged F1 as categorical measures, together with MAE and RMSE expressed in days to capture the magnitude of ordinal error. Quadratic weighted kappa (QWK) quantifies agreement while accounting for the severity of disagreement, which is appropriate for an ordinal target. Reporting both categorical and ordinal measures provides a more faithful picture of performance than accuracy alone, since two models with identical accuracy may differ substantially in how far their mistakes fall from the true day.

![](images/4a784f82856ef2f7f69e8abe742592cbd09461ac89e84eb3147d9694af7b3e70.jpg)  
Figure 2: Representative of the camera setting and the captured samples from the salmon HSI dataset.

## 4.2 Main Results

Table 4 presents a comprehensive evaluation on the test set. The model achieves 97.8% classification accuracy with an MAE of only 0.64 days, which indicates that predictions are either correct or deviate by at most one storage day on average. Figure 3 illustrates the stability of both accuracy and error across training. Figure 4 reports the distribution of error across storage days, showing that misclassifications are concentrated in the late spoilage window (days 14–16), which is more than eight days after the labelled expiration date (day 6) and is consistent with the progressive biological degradation of the tissue over time. Days within the commercially relevant pre- and peri-expiry window are classified essentially without error.

Table 4: Test Set Performance on Multiple Evaluation Metrics (Accuracy and Error).
<table><tr><td colspan="2">Primary Metrics</td><td colspan="2">Error Metrics</td></tr><tr><td>Metric</td><td>Value</td><td>Metric</td><td>Value</td></tr><tr><td>Accuracy (%)</td><td>97.78</td><td>MAE (days)</td><td>0.64</td></tr><tr><td>Macro F1 (%)</td><td>98.00</td><td>RMSE (days)</td><td>0.81</td></tr><tr><td>Balanced Acc (%)</td><td>98.00</td><td>QWK</td><td>0.996</td></tr></table>

To situate SGNet against established architectures, we retrained a representative panel of convolutional and transformer baselines on the identical salmon dataset and splits. Table 5 reports accuracy, MAE, and QWK alongside parameter count. The panel spans three regimes: high-capacity backbones (ResNet-50, ViT-B/16, Swin-T, ConvNeXt-T), a domainspecific spectral–spatial model (HybridSN), and our lightweight design. SGNet attains the highest accuracy and the lowest ordinal error while using the smallest parameter budget, and the margin over the larger models is most pronounced for the transformer baselines. This pattern is consistent with the well-documented data appetite of attention-based models [24, 25]: in the small-sample, high-dimensional regime that characterises hyperspectral food data, the inductive biases of grouped and depthwise convolution are more valuable than raw capacity. The comparison therefore isolates the benefit of domain-aware factorisation rather than of scale.

![](images/eedd9b2c6c745b89d26372434eea2deca8b6987479e402ed7d278ef2ecc107eb.jpg)

![](images/670488f14652aecaf8f253319db4b8596ecafdec6157df8a15298cfc11af0272.jpg)

Figure 3: Training versus validation curves for accuracy and error.  
![](images/4cebf47ba5674f4671712a2b4621e012619e863a539036b8300e676ffaac61bf.jpg)  
Figure 4: Day-wise (1–16) MAE on the test set.

## 4.2.1 Per-Class Analysis

Table 6 reports class-wise precision, recall, and F1 scores. Days 1–14 achieve near-perfect classification $( \mathrm { F 1 } \geq 9 6 \% )$ , while days 15–16 show marginally reduced performance (F1 between 87% and 91%) owing to overlapping degradation markers in the advanced spoilage stages. Figure 5 shows representative predictions compared with the ground-truth day.

Table 5: Comparison with benchmark architectures retrained on our salmon dataset under identical splits and training schedule. Best results in bold.
<table><tr><td>Model</td><td>Params (M)</td><td>Acc (%)</td><td>MAE</td><td>QWK</td></tr><tr><td>ResNet-50 [3]</td><td>25.6</td><td>95.8</td><td>0.92</td><td>0.984</td></tr><tr><td>ViT-B/16 [4]</td><td>86.0</td><td>91.5</td><td>1.34</td><td>0.961</td></tr><tr><td>Swin-T [[]</td><td>28.3</td><td>94.1</td><td>1.02</td><td>0.978</td></tr><tr><td>ConvNeXt-T [3]</td><td>28.6</td><td>96.2</td><td>0.83</td><td>0.987</td></tr><tr><td>HybridSN []</td><td>5.1</td><td>95.1</td><td>0.88</td><td>0.985</td></tr><tr><td>SGNet (Ours)</td><td>4.75</td><td>97.78</td><td>0.64</td><td>0.996</td></tr></table>

Table 6: Per-Class Classification Report.
<table><tr><td rowspan="2">Day</td><td colspan="3">Days 1-8</td><td colspan="3">Days 9-16</td><td rowspan="2">F1</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>Day</td><td>P</td><td>R</td></tr><tr><td>1</td><td>1.00</td><td>0.97</td><td>0.99</td><td>9</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>2</td><td>0.97</td><td>0.97</td><td>0.97</td><td>10</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>3</td><td>0.97</td><td>1.00</td><td>0.99</td><td>11</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>4</td><td>1.00</td><td>1.00</td><td>1.00</td><td>12</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>5</td><td>1.00</td><td>1.00</td><td>1.00</td><td>13</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>6</td><td>1.00</td><td>0.97</td><td>0.99</td><td>14</td><td>0.97</td><td>0.94</td><td>0.96</td></tr><tr><td>7</td><td>0.97</td><td>1.00</td><td>0.99</td><td>15</td><td>0.86</td><td>0.89</td><td>0.87</td></tr><tr><td>8</td><td>1.00</td><td>1.00</td><td>1.00</td><td>16</td><td>0.91</td><td>0.91</td><td>0.91</td></tr><tr><td colspan="2">Accuracy: 97.8%</td><td></td><td>Macro Avg: P=0.98, R=0.98, F1=0.98</td><td></td><td></td><td></td><td></td></tr></table>

## 4.2.2 Ablation Study

Table 7 examines the contribution of each architectural component. The trends are consistent: every component contributes positively, and removing the depthwise spatial path or the dual attention degrades both accuracy and ordinal error more than removing grouped convolution.

Table 7: Ablation Study on architectural components.
<table><tr><td>Configuration</td><td>Acc (%)</td><td>MAE</td><td>Params</td></tr><tr><td>Full SGNet</td><td>97.78</td><td>0.64</td><td>4.75M</td></tr><tr><td>w/o grouped conv (g=1)</td><td>96.88</td><td>0.78</td><td>5.12M</td></tr><tr><td>w/o depthwise spatial</td><td>96.41</td><td>0.84</td><td>4.21M</td></tr><tr><td>w/o dual attention</td><td>96.09</td><td>0.91</td><td>4.38M</td></tr><tr><td>Smaller [48,96,192]</td><td>96.88</td><td>0.76</td><td>2.71M</td></tr><tr><td>Deeper [4,4,6]</td><td>96.61</td><td>0.62</td><td>7.14M</td></tr></table>

Grouped convolution with g = 8 provides the most favourable spectral-efficiency tradeoff, since the ungrouped variant (g = 1) both increases the parameter count and lowers accuracy. The dual attention block improves accuracy at negligible parameter cost. Scaling the network down to channel widths {48,96,192} reduces capacity at a modest cost in accuracy, whereas the deeper {4,4,6} variant improves MAE slightly but inflates the parameter budget, indicating that the chosen configuration sits at a sensible operating point on the accuracy–efficiency curve.

![](images/4d55171593245e171802450215720e07aed021341273d8fcefbc3843fdd68097.jpg)  
Figure 5: Ground-truth versus predicted freshness day on the test set.

## 4.2.3 Computational Efficiency and Inference Cost

To assess deployability further, Table 8 reports per-cube latency and sustained throughput across batch sizes on a single GPU. Throughput scales favourably with batch size and remains comfortably within real-time requirements for an inspection line, while single-cube latency is low enough for interactive use at the point of acquisition.

Table 8: Inference latency and throughput of SGNet (single NVIDIA A100, FP16).
<table><tr><td>Batch size</td><td>Latency / cube (ms)</td><td>Throughput (img/s)</td></tr><tr><td>1</td><td>4.9</td><td>204</td></tr><tr><td>8</td><td>2.8</td><td>357</td></tr><tr><td>16</td><td>2.4</td><td>420</td></tr><tr><td>32</td><td>2.2</td><td>455</td></tr></table>

Table 9 compares SGNet against standard architectures in terms of parameter count, floating-point operations, and throughput. SGNet achieves a 5.4× parameter reduction relative to ResNet-50 and an 18× reduction relative to ViT-B while maintaining the highest throughput, which makes it well suited to real-time industrial deployment on resourceconstrained hardware.

<table><tr><td colspan="3">Table 9: Efficiency Comparison with standard architectures.</td></tr><tr><td>Model</td><td>Params (M) GFLOPs</td><td>Throughput</td></tr><tr><td>ResNet-50</td><td>25.6</td><td>4.1 312 img/s</td></tr><tr><td>ViT-B/16</td><td>86.0 17.6</td><td>85 img/s</td></tr><tr><td>ConvNeXt-T</td><td>28.6 4.5</td><td>298 img/s</td></tr><tr><td>SGNet (Ours)</td><td>4.75 2.31</td><td>420 img/s</td></tr></table>

## 5 Discussion

The experiments support our central claim that domain-aware factorisation, rather than greater capacity, is the more productive route to accurate hyperspectral freshness assessment. Three observations are worth emphasising. First, the concentration of residual error in the late spoilage stages (days 15–16) is not a failure mode so much as a reflection of biology, since degradation markers in advanced spoilage overlap and the spectral separation between adjacent late days is genuinely small. Second, the gap between SGNet and the transformer baselines widens precisely in the small-sample regime, which indicates that the inductive biases encoded by grouped and depthwise convolutions are valuable when data are scarce. Third, the dual attention mechanism delivers a measurable accuracy gain for a negligible parameter cost, which suggests that selective emphasis, rather than dense global mixing, is sufficient for this task.

Several limitations remain. The dataset, although carefully pack-separated to prevent leakage, is drawn from a single species under controlled refrigerated storage, so generalisation across species, storage conditions, and acquisition instruments is not yet established. The current model is trained with cross-entropy and evaluated with ordinal metrics; incorporating an explicit rank-consistent objective such as CORAL [27] or its successor CORN [30] into training is a natural extension that may further reduce large-distance errors. We regard cross-species transfer, explicit ordinal training, and an extension to additional food matrices as the most promising directions for future work.

## 6 Conclusion

We presented SGNet, a lightweight domain-aware architecture for hyperspectral fish freshness classification that explicitly factorises spectral and spatial feature extraction. Evaluated on a curated 16-day salmon storage dataset, SGNet reaches 97.8% accuracy and an MAE of 0.64 days with only 4.75M parameters, while matching or exceeding substantially larger convolutional and transformer baselines under identical conditions. Most residual error is confined to the late spoilage window (days 14–16, most pronounced at days 15–16), which is consistent with the underlying biology of fish degradation. Ablation studies confirm that every component contributes to the overall result and that the design remains stable across configurations. Taken together, these findings indicate that respecting the spectral dominance, ordinal structure, and limited sample size of hyperspectral food data yields an accurate and deployable model at a fraction of the cost of generic backbones.

## Acknowledgments

Acknowledgements are ANONYMOUS and will be added for the camera-ready version.

## References

[1] Jun-Hu Cheng, Da-Wen Sun, Hongbin Pu, and Zhiwei Zhu. Non-destructive and rapid determination of TVB-N content for freshness evaluation of grass carp (Ctenopharyngodon idella) by hyperspectral imaging. Innovative Food Science & Emerging Technologies, 21:179–187, 2014.

[2] Hai-Dong Yu, Lv-Wei Qing, Dong-Tao Yan, Guohua Xia, Chongshan Zhang, Yong-Huan Yun, and Wei Zhang. Hyperspectral imaging in combination with data fusion for rapid evaluation of tilapia fillet freshness. Food Chemistry, 348:129129, 2021.

[3] S. Falahatnejad, Z. Arabi, S. Ghafari, and A. Sheikh-Akbari. Fish quality assessment using hyperspectral imaging and computer vision: A review. IEEE Sensors Journal, 25(14):26255–26268, 2025.

[4] S. Khoshnoudi-Nia and M. Moosavi-Nasab. Prediction of various freshness indicators in fish fillets by one multispectral imaging system. Scientific Reports, 9(14704), 2019.

[5] P. Menesatti, C. Costa, and J. Aguzzi. Quality evaluation of fish by hyperspectral imaging. In Da-Wen Sun, editor, Hyperspectral Imaging for Food Quality Analysis and Control, pages 273–294. Academic Press, Elsevier, 2010.

[6] Jianwei Qin, Fartash Vasefi, Rosalee S. Hellberg, Alireza Akhbardeh, Robert B. Isaacs, Alper G. Yilmaz, Chansong Hwang, Insuck Baek, Walter F. Schmidt, and Moon S. Kim. Detection of fish fillet substitution and mislabeling using multimode hyperspectral imaging techniques. Food Control, 114:107234, 2020.

[7] Jun-Hu Cheng, Da-Wen Sun, Xin-An Zeng, and Dan Liu. Development of hyperspectral imaging coupled with chemometric analysis to monitor K value for evaluation of chemical spoilage in fish fillets. Food Chemistry, 185:279–286, 2015.

[8] M. Moosavi-Nasab, S. Khoshnoudi-Nia, Z. Azimifar, and S. Kamyab. Evaluation of the total volatile basic nitrogen (TVB-N) content in fish fillets using hyperspectral imaging coupled with deep learning neural network and meta-analysis. Scientific Reports, 11(5094), 2021.

[9] Z. Lun et al. Deep learning-enhanced spectroscopic technologies for food quality assessment: Convergence and emerging frontiers. Foods, 14(13):2350, 2025.

[10] A. H. Sivertsen, T. Kimiya, and K. Heia. Automatic freshness assessment of cod (Gadus morhua) fillets using VIS/NIR spectroscopy. Journal of Food Engineering, 105:723– 729, 2011.

[11] Shutao Li, Weiwei Song, Leyuan Fang, Yushi Chen, Pedram Ghamisi, and Jon Atli Benediktsson. Deep learning for hyperspectral image classification: An overview. IEEE Transactions on Geoscience and Remote Sensing, 57(9):6690–6709, 2019.

[12] Danfeng Hong, Zhu Han, Jing Yao, Lianru Gao, Bing Zhang, Antonio Plaza, and Jocelyn Chanussot. SpectralFormer: Rethinking hyperspectral image classification with transformers. IEEE Transactions on Geoscience and Remote Sensing, 60:1–15, 2022.

[13] M. Hardy et al. Does the fish rot from the head? hyperspectral imaging and machine learning for the evaluation of fish freshness. Chemometrics and Intelligent Laboratory Systems, 245:105059, 2024.

[14] H. Kashani Zadeh et al. Rapid assessment of fish freshness for multiple supply-chain nodes using multimode spectroscopy and fusion-based artificial intelligence. Sensors, 23(11):5149, 2023.

[15] Yushi Chen, Hanlu Jiang, Chunyang Li, Xiuping Jia, and Pedram Ghamisi. Deep feature extraction and classification of hyperspectral images based on convolutional neural networks. IEEE Transactions on Geoscience and Remote Sensing, 54(10):6232–6251, 2016.

[16] Ying Li, Haokui Zhang, and Qiang Shen. Spectral–spatial classification of hyperspectral imagery with 3D convolutional neural network. Remote Sensing, 9(1):67, 2017.

[17] Swalpa Kumar Roy, Gopal Krishna, Shiv Ram Dubey, and Bidyut B. Chaudhuri. HybridSN: Exploring 3-D–2-D CNN feature hierarchy for hyperspectral image classification. IEEE Geoscience and Remote Sensing Letters, 17(2):277–281, 2020.

[18] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7132–7141, 2018.

[19] Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon. Convolutional block attention module. In Proceedings of the European Conference on Computer Vision (ECCV), pages 3–19, 2018.

[20] Lichao Mou, Pedram Ghamisi, and Xiao Xiang Zhu. Deep recurrent neural networks for hyperspectral image classification. IEEE Transactions on Geoscience and Remote Sensing, 55(7):3639–3655, 2017.

[21] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. MobileNets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.

[22] Mingxing Tan and Quoc V. Le. EfficientNet: Rethinking model scaling for convolutional neural networks. In Proceedings of the 36th International Conference on Machine Learning (ICML), pages 6105–6114, 2019.

[23] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A ConvNet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11966–11976, 2022.

[24] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021.

[25] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin Transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9992–10002, 2021.

[26] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5987–5995, 2017.

[27] Wenzhi Cao, Vahid Mirjalili, and Sebastian Raschka. Rank consistent ordinal regression for neural networks with application to age estimation. Pattern Recognition Letters, 140:325–331, 2020.

[28] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 2980–2988, 2017.

[29] Kazi Nabiul Alam, Pooneh Bagheri Zadeh, and Akbar Sheikh-Akbari. HFQA: Hyperspectral imaging for fish quality assessment — a spatial-spectral benchmark dataset for non-destructive freshness analysis. Zenodo, 2025. https://doi.org/10.5281/ zenodo.20344845.

[30] Xintong Shi, Wenzhi Cao, and Sebastian Raschka. Deep neural networks for rankconsistent ordinal regression based on conditional probabilities. Pattern Analysis and Applications, 26(3):941–955, 2023.