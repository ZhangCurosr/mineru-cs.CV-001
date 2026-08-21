# AsymFeX: A Symmetry-Driven Framework for Ischemic Stroke Segmentation Across Imaging Modalities and Stroke Stages

Maunil Shah, Vaanathi Sundaresan

## Abstract

Fast and accurate segmentation of Acute Ischemic Stroke (AIS) lesions is essential for stroke prognosis and treatment planning. Non-contrast CT (NCCT), the first-line imaging modality for diagnosing ischemic infarcts, exhibits subtle infarct contrast, making manual delineation slow and labor-intensive. Motivated by this, and by the clinical practice of comparing brain hemispheres to localize infarcts, we propose a two-stage, nnU-Net-compatible 3D segmentation method. The first stage corrects head tilt to align each scan to its true anatomical mid-sagittal plane; the second applies a novel Asymmetric Feature Extraction (AsymFeX) module, comparing each voxel to its true contralateral counterpart within a local 3 × 3 × 3 neighborhood via crosshemispheric attention, feature disparity estimation, and dual-scale gating to capture both large and small infarcts. On AISD, our method achieves 0.6796 Dice, 23.53 mm HD95, and 7.69 mL AVD, significantly outperforming existing state-of-the-art methods, with clinically relevant volumetric analysis at the 70 mL thrombolysiseligibility threshold. Proof-of-concept evaluation on ATLAS v2.1 and ISLES’24 demonstrates that the same symmetry-driven design generalizes across imaging modalities and stroke time points without architectural changes, further supported by an uncertainty analysis assessing reliability under clinical deployment. Code is publicly available at https://github.com/ biomedia-lab/AIS-detection.

Index Terms—Ischemic stroke segmentation, bilateral asymmetry, cross-hemispheric attention, multimodal stroke imaging.

## 1 Introduction

Ischemic stroke is a subtype of cerebrovascular stroke that occurs due to reduced cerebral blood flow caused by thrombotic or embolic events that block the cerebral vessels, leading to insuficient brain tissue perfusion [1]. According to the GBD 2019 report, there were 7.63 million cases of ischemic stroke, accounting for 62.4% of all stroke incidents [2] [3]. During an ischemic event, brain tissue that incurs irreversible damage is called the infarct or ischemic core, while the tissue that is at risk of infarction but can potentially be salvaged is referred to as the ischemic penumbra.

Multiple imaging modalities are acquired depending on symptom onset time, patient criticality, and scanner availability, each guiding a distinct stage of reperfusion decision-making. NCCT is acquired to exclude intracranial hemorrhage or stroke mimics [4]. Within the first 24 hours, Difusion-Weighted Imaging (DWI) is the goldstandard study for detecting acute ischemia [5] [6], owing to its higher sensitivity to infarction than NCCT [7]. CT Perfusion (CTP) identifies salvageable, at-risk penumbra tissue, guiding whether a patient will benefit from reperfusion therapy [4] [7]. Together, NCCT, DWI, and CTP ofer complementary information across the acute window to inform treatment decisions. In the chronic stage, beyond three weeks post-onset, T1- weighted MRI’s high spatial resolution clearly reveals infarcts as hypointense regions [8] [9], supporting poststroke treatment and rehabilitation planning. Given the time-sensitive nature of Acute Ischemic Stroke (AIS), delivering appropriate therapy as early as possible remains critical to salvaging maximal penumbra volume.

Although AIS infarcts are most appreciable on DWI in the acute phase, NCCT remains indispensable due to its shorter acquisition time, wide availability, and status as the first-line imaging modality in most stroke assessment protocols. However, AIS infarcts exhibit very subtle contrast on NCCT relative to healthy tissue [4] [5], may suffer from artifacts and low signal-to-noise ratio, making it dificult to isolate afected tissue. Consequently, manual delineation of AIS lesions is laborious, time-consuming, and demands significant clinical expertise, motivating the need for automated segmentation methods that support rapid decision-making and treatment planning.

Deep learning (DL) has become the dominant paradigm for automated AIS segmentation, and a growing subset of recent work has attempted to exploit the brain’s bilateral anatomical symmetry, since healthy hemispheres are approximately mirror-symmetric while an infarct locally distorts this symmetry. However, existing symmetry-aware methods remain limited in at least one of the following respects: they permit attention over the entire brain or entire contralateral hemisphere rather than a localized landmark-level comparison incurring unnecessary computational cost; they operate in 2D and cannot capture volumetric lesion characteristics; they compare a scan against its mirrored copy without first correcting head tilt, so the compared regions may not be true anatomical counterparts; or they require full brain slices as input, making them incompatible with the patch-based sampling used by modern segmentation frameworks such as nnU-Net. Beyond these architectural limitations, existing methods are typically evaluated on single imaging modality, single stroke time-point data and report no measure of prediction reliability. A detailed technical critique of representative methods is provided in Section 2.

Motivated by these gaps, we propose a novel framework: a two-stage, nnU-Net-compatible 3D segmentation pipeline that first geometrically aligns each scan to its true anatomical mid-sagittal plane, then exploits the resulting hemispheric correspondence through a localized, landmark-level cross-attention mechanism. Our contributions are summarized as follows:

• A 2-stage 3D segmentation pipeline that integrates with nnUNet and incorporates stroke-specific prior knowledge via an explicit geometric tiltcorrection step, ensuring the contralateral comparison is anatomically valid.

• A novel Asymmetric Feature Extraction (Asym-FeX) module combining 3D local cross-hemispheric attention, disparity estimation, and dual-scale gating, restricting comparison to a local neighborhood around each voxel’s true contralateral landmark rather than the entire opposite hemisphere.

• A comprehensive evaluation against existing stateof-the-art and baseline methods on AISD, with extensive ablation isolating the contribution of each architectural component and volumetric analysis assessing clinical relevance.

• A proof-of-concept demonstration that the same asymmetry-driven design adapts across the acuteto-chronic stroke imaging continuum (CT perfusion and MRI).

• A comprehensive uncertainty quantification covering error-uncertainty analysis and calibration analysis on three datasets to assess prediction reliability across modalities and stroke stages.

The remainder of this paper is organized as follows. Section 2 reviews related work and Section 3 presents the proposed methodology. Section 4 describes the datasets and experimental setup, and Section 5 reports results, including state-of-the-art comparison, computational cost, volumetric analysis, ablation studies, crossmodality generalization, and uncertainty quantification, followed by discussion and future directions in Section 6.

## 2 Related Work

In this section, we present a literature survey pertaining to generic deep learning methods for fully supervised medical image segmentation, followed by methods specifically intended to segment AIS lesions by leveraging asymmetry.

Fully supervised DL for medical image segmentation has progressed markedly since the introduction of U-Net [10], with subsequent architectures building on the encoder-decoder paradigm across both 2D and 3D settings. U-Net++ [11] introduced nested skip pathways to bridge the semantic gap between encoder and decoder features, improving delineation of fine anomalous regions. Attention U-Net [12] incorporated attention gates to address low tissue contrast and shape/size heterogeneity across organs. The nnU-Net [13] instead emphasized systematic pipeline design, categorizing parameters such as voxel spacing, class proportions, learning rate, optimizer, post-processing, hardware constraints, etc. into fixed, rule-based, and empirical groups, yielding a task-agnostic framework that, despite relying on a standard U-Net backbone, remains a highly competitive segmentation baseline.

Transformer-based models quickly became a widelyadapted model for image segmentation due to their ability for efectively learning the long range dependencies. Swin-UNET [14] was the first pure transformerbased model proposed for medical image segmentation. However, hybrid models combining CNNs and transformers yields better performance as opposed to pure transformer-based models due to their ability of capturing complementary information. As such, Swin-UNETR [15] is a hybrid variant where the encoder consists of Swin transformer block while the residual blocks and the decoder blocks comprise of convolution layers and deconvolution layers respectively. TransUNet [16] proposed a hybrid encoder wherein the initial features are extracted using a CNN-based encoder followed by transformer encoder to complement the CNN features.

Recently, state-space architectures have emerged to address the computational complexity of transformerbased networks. U-Mamba [17] integrates State Space Models (SSMs), specifically the Mamba selective scan mechanism, within a U-Net framework to model longrange global dependencies. By achieving linear computational complexity (O(N)) relative to sequence length, U-Mamba overcomes the quadratic scaling limitations (O(N<sup>2</sup>)) inherent to Vision Transformers while maintaining high representative capacity.

Kolmogorov-Arnold Networks (KANs) benefit from interpretability and learnable activation functions [18]. Motivated by this, U-KAN [19] substitutes traditional Multi-Layer Perceptrons (MLPs) with tokenized KAN blocks near the network bottleneck. By leveraging KAN, the authors take a positive step in the direction of theory-backed medical image segmentation.

Methods discussed so far are general purpose medical image segmentation methods and do not utilize any stroke-specific information such as bilateral asym metry. In this section, we discuss recent methods that have attempted to leverage the brain’s anatomy-related cues and structural information to segment AIS lesions. MDAN [20] leverages hemispheric symmetry by learning representations that bring healthy tissue closer together while pushing lesion tissue apart, and integrates Mirror Feature Fusion modules along the skip connections to fuse original and mirrored features. SAN-Net [21] addresses cross-site MR heterogeneity through adaptive normalization and a symmetry-inspired augmenta tion strategy that focuses the model on a single hemi sphere to simplify lesion localization. IS-Net [22] aggregates features across multiple stages and employs a Nonlocal Parallel Decoder combining deformable convolution with self-attention to promote mask continuity and exploit hemispheric asymmetry. PAPL [23] proposes a three-stage framework that first learns represen tations of pathological asymmetry through contrastive pre-training, then carries this knowledge into end-to-end training, and finally refines mis-segmented regions using the predicted region and its counterpart across the mid sagittal line. Cl-SegNet [24] fuses transformer and CNN features eficiently and introduces a Bilateral Diference Learning (BDL) module to compare intensities between a feature map and its flipped copy. More recently, Sun et al. [25] propose a two-stage method that first identifies suspicious lesion areas by learning to replicate a maximum intensity projection map, then uses this sig nal to predict probability maps for the original scan, the asymmetry map, and the final lesion mask. Li et al. [26] propose DySym-UNet, which introduces a Symmetric Cross-Attention block to capture abnormalities between the left and right brain hemispheres.

Although existing methods have shown promising results, they carry limitations that constrain their use for 3D AIS segmentation. Several rely on attention over the entire brain (e.g., IS-Net) or entire contralateral hemisphere (e.g., DySym-UNet), which is computationally expensive and unnecessary, since detecting abnormality only requires comparison with the mirror-symmetric counterpart across the mid-sagittal line. Other methods (MDAN, PAPL, SAN-Net) are 2D and cannot cap ture 3D lesion characteristics such as surface texture or spatial continuity across slices. Cl-SegNet compares a feature map with its flipped copy without correcting head tilt beforehand, so misaligned scans yield a flipped copy that no longer corresponds to the true contralateral region, misleading the resulting feature diference. Further, SAN-Net, IS-Net, PAPL, and DySym-UNet all require full brain slices to exploit hemispheric disparity, making them incompatible with nnU-Net’s random patch-based sampling, where an individual patch may not contain both hemispheres.

Beyond these architectural constraints, we identify two further gaps common across this body of work. First, existing methods are typically validated on a single imaging modality and at a single stroke time-point data, and do not demonstrate that a symmetry-driven design generalizes across the acute-to-chronic continuum spanning from NCCT to CT perfusion to MRI. Second, none report any measure of prediction reliability or uncertainty estimates, despite its importance for clinical deployment.

The proposed method addresses gaps identified above using a robust 3D framework that integrates with nnU-Net and mimics a clinician’s approach of comparing tissue directly with its contralateral landmark across the mid-sagittal line, while addressing the validation, generalization, and reliability. Section 3 presents the proposed framework in detail.

## 3 Methodology

The proposed framework comprises two modalityagnostic stages, illustrated in Algorithm 1 and Fig. 1. Stage 1 geometrically aligns the input volume to its true mid-sagittal plane, ensuring a subsequent vertical flip yields a valid contralateral counterpart. Stage 2 then processes the aligned volume and its flip through a shared-weight encoder and a novel Asymmetric Feature Extraction (AsymFeX) module, comparing each tissue region against its contralateral landmark to identify asymmetries indicative of infarction. Both stages are defined independently of any specific imaging modality, with modality-specific adaptation and the datasets are detailed in Sections 5.5 and 4.1, respectively.

## 3.1 Stage 1: Geometric Tilt Correction

Most existing NCCT datasets [27] are not tilt corrected. To truly leverage symmetric diferences in the downstream asymmetric learning task, we first implement a pre-processing algorithm that rectifies head tilt in a CT scan and its corresponding segmentation mask. The complete workflow is illustrated in Algorithm 1.

## 3.1.1 Optimal Slice Selection

We begin with applying a brain windowing (width = 80, level = 40) and mapping the voxel values to the range [0, 255]. We then proceed to remove the nonbrain structures, such as the CT scanner head cushion, using the FSL Brain Extraction Tool [28]. Following CT cushion removal, we select an optimal axial slice that captures the maximum cross-sectional area of the brain parenchyma.

![](images/37fc299b57e1a9df577334c7df84b11019e0c79496c8ab5177af53d931c8b6fb.jpg)  
Figure 1: Illustration of Stage 2 of the proposed methodology: Stage 2 receives $V _ { f i n a l }$ and $V _ { f l i p p e d }$ from stage 1 as the input and uses AsymFex blocks along the deeper layers for leveraging the asymmetric diferences. The AsymFex comprises of 3 sub-modules, viz., Cross-Stream Alignment (CSA), Feature Disparity Estimation (FDE) and Dual-Scale Gating (DSG).

To isolate the brain structure from high-intensity skull elements, the stripped volume $V _ { \mathrm { s t r i p p e d } }$ is binarized using an intensity threshold $\tau \ = \ 2 5 0$ to reliably separate residual high-intensity skull fragments from brain parenchyma following skull-stripping. Voxels with attenuation values greater than or equal to τ are suppressed to 0, while all the remaining voxel values are mapped to 1.

For each axial slice i along the longitudinal axis of the binarized volume, a 2D morphological hole-filling operation is executed to create a solid, continuous representation of the brain structure. We then apply 2D connected component labeling using ${ \mathrm { ~ a ~ 3 ~ } } \times 3$ structural connectivity matrix to eliminate peripheral artifacts and isolate the largest contiguous connected component mask $\tilde { S } _ { i }$ . The structural score for the slice i is defined as the total area (voxel sum) of this refined tissue region:

$$
\mathrm { S c o r e } ( i ) = \sum _ { x , y } \tilde { S } _ { i } ( x , y )\tag{1}
$$

The slice index exhibiting the maximum area score is designated as the optimal slice $s _ { \mathrm { o p t } }$

$$
s _ { \mathrm { o p t } } = \arg \operatorname* { m a x } _ { i } \mathrm { S c o r e } ( i )\tag{2}
$$

## 3.1.2 Symmetry-Driven Tilt Correction

Once $s _ { \mathrm { o p t } }$ is identified, its geometric centroid ${ \cal { C } } = $ $( C _ { x } , C _ { y } )$ is computed based on the spatial coordinates of its non-zero pixels. The entire 3D volume $\left( V _ { \mathrm { s t r i p p e d } } \right)$ and the segmentation mask $( M _ { \mathrm { r a w } } )$ are then translated to C.

Using a grid search brute-force optimization paradigm, the exact head tilt angle θ is extracted from the translated optimal slice $\left( s _ { \mathrm { o p t } } \right)$ by minimizing hemispheric structural disparity. This is achieved by defining a search space for angle in the range $\phi ~ \in ~ ( - 4 5 ^ { \circ } , 4 5 ^ { \circ } )$ at increments of $0 . 0 5 ^ { \circ }$ For each candidate angle ϕ:

1. A standard 2D rotation matrix $R _ { \phi }$ is constructed, and $s _ { \mathrm { o p t } }$ is rotated about its centered origin.

2. The rotated slice is partitioned along its vertical line into left and right halves.

3. The right half is flipped horizontally and overlaid onto the left half.

4. A hemispheric disparity score, defined as the diference between the union and the intersection of the two halves, is calculated.

The optimal tilt correction angle θ is determined as the angle that minimizes this asymmetry disparity score, effectively mapping the precise true vertical symmetry line of the cranium. Mathematically, this optimization is expressed as:

```tcl
Algorithm 1 Geometric Tilt Correction.
Require: Ischemic CT Volume $V _ { \mathrm { r a w } } .$ Target Mask $M _ { \mathrm { r a w } }$
Ensure: Aligned and Cropped $( V _ { \mathrm { f i n a l } } , ~ M _ { \mathrm { f i n a l } } )$
1: $/ / \boldsymbol { \mathit { 1 } } .$ . Optimal Slice Selection $( s _ { o p t } )$
2: $\dot { V } _ { \mathrm { s t r i p p e d } }  \mathrm { F S L } \mathrm { . B E T } ( V _ { \mathrm { r a w } } )$
3: $V _ { \mathrm { b i n } } $ Binarize(V<sub>stripped</sub>, τ ) {Set $\geq \tau$ to 0 and $> 0$
to 1}
4: Best Score ← −1, s<sub>opt</sub> ← None
5: for each slice i in $V _ { \mathrm { b i n } }$ do
6: $S _ { i } \gets V _ { \mathrm { b i n } } [ : , : , i ]$
7: $S _ { \mathrm { f i l l e d } }  \mathrm { F i l l }$ Holes $\mathrm { 2 D } ( S _ { i } )$
8: $L _ { i } \gets$ Label Connected Components $( S _ { \mathrm { f i l l e d } } )$
9: $\tilde { S } _ { i } \gets$ Isolate Largest Component $( L _ { i } )$
10: Score $ \sum { \tilde { S } } _ { i }$ {Calculate component area}
11: if Score > Best Score then
12: Best Score ← Score
13: $s _ { \mathrm { o p t } } \gets V _ { \mathrm { s t r i p p e d } } [ : , : , i ]$
14: end if
15: end for
16: $/ / 2 .$ Symmetry-Driven Tilt Correction
17: $C ( C _ { x } , C _ { y } ) \gets \mathrm { C o m p u t e \_ C e n t r o i d } ( s _ { \mathrm { o p t } } )$
18: $V _ { \mathrm { t r a n s } } $ Translate Volume(V<sub>stripped</sub>, C)
19: $M _ { \mathrm { t r a n s } }  \mathrm { T r a n s l a t e \_ V o l u m e } ( M _ { \mathrm { r a w } } , C )$
20: Min Disparity $ \infty , \quad \theta  0 . 0$
21: for $\phi = - 4 5 . 0 ^ { \circ }$ to $4 5 . 0 ^ { \circ }$ with step $0 . 0 5 ^ { \circ }$ do
22: $s _ { \mathrm { r o t } } $ Rotate ${ \mathrm { S l i c e } } ( s _ { \mathrm { o p t } } , R _ { \phi } )$
23: $H _ { \mathrm { l e f t } } , H _ { \mathrm { r i g h t } } \gets \mathrm { S p l i t . V e r t i c a l { \_ A x i s } } ( s _ { \mathrm { r o t } } )$
24: $H _ { \mathrm { f l i p p e d . r i g h t } }  \mathrm { H o r i z o n t a l . F l i p } ( H _ { \mathrm { r i g h t } } )$
25: $\mathrm { D i s } \mathrm {  { ~ \hat { \ p a r i t y } ~ } }  \mathrm {  ~ | ~ { \cal H } _ { l e f t } ~ \cup ~ { \cal H } _ { f l i p p e d \bot r i g h t } | \ - \ | ~ { \cal H } _ { l e f t } ~ \cap ~  }$
$H _ { \mathrm { H i p p e d . } }$ <sub>right</sub>|
26: if Disparity < Min Disparity then
27: Min Disparity ← Disparity
28: $\theta  \phi$
29: end if
30: end for
31: $V _ { \mathrm { a l i g n e d } } $ Rotate Volume $( V _ { \mathrm { t r a n s } } , \theta )$
32: $M _ { \mathrm { a l i g n e d } }$ ← Rotate Volume $( M _ { \mathrm { t r a n s } } , \theta )$
33: $/ / \ 3 .$ Tight Cropping
34: $V _ { \mathrm { f i n a l } } , M _ { \mathrm { f i n a l } }  \mathrm { T i g h t } .$ Bound Box Crop(V<sub>aligned</sub>, M<sub>align</sub>
35: return $V _ { \mathrm { f i n a l } } , M _ { \mathrm { f i n a l } }$
```

$$
\theta = \arg \operatorname* { m i n } _ { \phi } \left( | H _ { \mathrm { l e f t } } \cup H _ { \mathrm { f l i p p e d - r i g h t } } | - | H _ { \mathrm { l e f t } } \cap H _ { \mathrm { f l i p p e d - r i g h t } } | \right)\tag{3}
$$

## 3.1.3 Tight Cropping

Following calibration, the translated 3D CT volume $\left( V _ { \mathrm { t r a n s } } \right)$ and its corresponding ground-truth segmentation mask $( M _ { \mathrm { t r a n s } } )$ are rotated by the optimal angle θ, yielding pairs $V _ { \mathrm { a l i g n e d } }$ and $M _ { \mathrm { a l i g n e d } }$ . To eliminate redundant background voxels and constrain the region of interest strictly to the cranial anatomy, a tight 3D bound ing box crop is applied, producing the finalized volume $V _ { \mathrm { f i n a l } }$ and mask $M _ { \mathrm { f i n a l } }$ . We then mirror the volume $V _ { \mathrm { f i n a l } }$ to produce $V _ { \mathrm { H i p p e d } }$ which swaps the left and right hemisphere so that each landmark becomes aligned with its counterpart on the opposite side. This flipped volume $\left( V _ { \mathrm { f l i p p e d } } \right)$ is then coupled with $V _ { \mathrm { f i n a l } }$ to create a dualstream input.

## 3.2 Stage 2: Asymmetric Deep Network Architecture

The fundamental objective of the proposed architecture is to exploit asymmetry between the two bilateral hemispheres by leveraging the tilt-corrected volume $V _ { \mathrm { f i n a l } }$ along with its horizontally flipped version $V _ { \mathrm { H i p p e d } }$ for direct comparison. The overall architecture is depicted in Fig. 1.

## 3.2.1 Shared CNN Encoder

Inspired by the architectural topology of the 3D U Net [10], the framework first passes the inputs through a five-stage convolutional encoder, denoted as E. Each stage within E contains two consecutive blocks, where each block is composed of a standard 3D convolutional layer followed by a 3D instance normalization layer and a Rectified Linear Unit (ReLU) activation function. The two input streams, the tilt-corrected volume $V _ { \mathrm { f i n a l } } \in \mathbb { R } ^ { 1 \times D \times H \times W }$ and its mirrored contralateral counterpart $V _ { \mathrm { f l i p p e d } } \in \mathbb { R } ^ { 1 \times D \times H \times W }$ , are processed independently through this encoder. The encoder E operates under a weight-sharing configuration across both streams. The choice of a pure CNN architecture for the encoder is driven by three reasons. Firstly, it extracts similar features from both inputs and maps them into a shared space. Since both hemispheres are processed using the exact same weights, any diferences found between their feature maps later in the network could be )attributed to actual tissue diferences, which could be further leveraged by the Asymmetric Feature Extraction (AsymFeX) modules (3.2.2). Secondly, our model is integrated within the nnU-Net [13] framework, which relies on random 3D patch-based sampling during training. Because an isolated 3D patch rarely captures the entire brain, the CNN encoder ensures that the network establishes a clear representation of a voxel’s local surroundings before the model attempts to evalu ate broader hemispheric diferences. Thirdly, this setup closely mimics clinical workflow where a radiologist typically inspects one hemisphere first to establish a baseline of normal anatomy before cross-referencing the opposite side to detect anomalies. Our encoder similarly extracts and preserves structural features independently before any comparison occurs.

Formally, let $\begin{array} { r l r } { { \bf E } ( V _ { \mathrm { f i n a l } } ) } & { { } = } & { \{ g _ { 1 } , g _ { 2 } , g _ { 3 } , g _ { 4 } , g _ { 5 } \} } \end{array}$ and $\mathbf { E } ( V _ { \mathrm { f l i p p e d } } ) = \{ h _ { 1 } , h _ { 2 } , h _ { 3 } , h _ { 4 } , h _ { 5 } \}$ denote the multi-scale feature maps extracted by E from $V _ { \mathrm { f i n a l } }$ and $V _ { \mathrm { H i p p e d } }$ , respectively. We then route the deep features from the last two layers, namely $( g _ { 4 } , h _ { 4 } )$ and $( g _ { 5 } , h _ { 5 } )$ , forward into the corresponding Asymmetric Feature Extraction (AsymFeX) blocks along the skip connections. We place AsymFeX only in deeper stages, where features are semantically richer and spatially broader, allowing it to capture meaningful anatomical diferences rather than shallow texture or edge information.

## 3.2.2 Asymmetric Feature Extraction (Asym-FeX)

Building on the independent features extracted by the shared encoder E, the asymmetric comparison modules are integrated at feature levels 4 and 5, which contain 256 and 320 channels, respectively. This module consists of a cascaded sequence of identical AsymFeX blocks. Within each block, three stages are performed sequentially: Cross-Stream Alignment (CSA), Feature Disparity Estimation (FDE), and Dual-Scale Gating (DSG).

(a) Cross-Stream Alignment (CSA): Standard global self-attention scales quadratically with the number of voxels $\left( \mathcal { O } ( N ^ { 2 } ) \right)$ [29], making it expensive for 3D image analysis. Moreover, it allows every voxel to attend to distant regions, which is unnecessary for our objective as we only need to compare each tissue region with its corresponding contralateral counterpart in the flipped input. Since our Stage 1 pre-processing ensures that corresponding contralateral features are nearly aligned, we restrict cross-attention between $g _ { i }$ and $h _ { i }$ to a local $3 \times 3 \times 3$ spatial neighborhood (found to be an optimal choice - refer to Section 5.4.4 for further experimental details), giving a neighborhood size of $N _ { w } = 2 7$ voxels. This reduces the attention cost from $\mathcal { O } ( N ^ { 2 } )$ to $\mathcal { O } ( N \cdot N _ { w } )$ with $N _ { w } \ll N$ , i.e. linear in the number of voxels.

The local neighborhood is motivated by the fact that even after geometric alignment, the two hemispheres may not match perfectly at the voxel level due to the brain’s natural anatomical asymmetry, interpolation effects and minor tilt-correction errors. Therefore, a true contralateral match may not lie exactly at the mirrored location, so the window provides a search region around the expected landmark which is large enough to absorb these errors (ablation conducted with diferent window sizes as specified in Section 5.4.4).

We denote a feature volume by its shape $( C , H , W , D )$ where C is the number of channels and $( H , W , D )$ are the height, width, and depth of the 3D spatial grid. We start by computing the Query (Q), Key (K) and Value $( V )$ tensors. The $Q$ is projected from the flipped stream $\left( h _ { \mathrm { i } } \right)$ while the K and V are projected from the original stream $\left( g _ { \mathrm { i } } \right)$ , each using a separate point-wise $1 \times 1 \times 1$ convolution preserving the tensor shape $( C , H , W , D )$

This can be denoted as:

$$
Q = W _ { q } ( h _ { \mathrm { i } } ) ,\tag{4a}
$$

$$
K = W _ { k } ( g _ { \mathrm { i } } ) ,\tag{4b}
$$

$$
V = W _ { v } ( g _ { \mathrm { i } } )\tag{4c}
$$

Next, the $C$ channels are partitioned into M attention heads, each operating on $d _ { k } = C / M$ channels, so that similarity is measured independently within M feature subspaces. Hence, the matrices $Q , K , V$ are reshaped to $\mathbb { R } ^ { M \times H \times W \times D \times d _ { k } }$ The local attention window spans 3 positions per axis and therefore $N _ { w }$ accumulates to 27 neighbors per voxel. Next, before gathering neighbors in K and $V ,$ we first zero-pad them by one voxel on each spatial side to include the boundary voxels as well. We then apply three successive tensor unfold operations, one along each spatial axis $( H , W , D )$ , which collects the $3 \times 3 \times 3$ block of neighbors for every voxel. More precisely, for each voxel we collect the neighbors obtained by applying the ofsets $( d x , d y , d z ) \in \{ - 1 , 0 , 1 \} ^ { 3 }$ along the three spatial axes, i.e. the voxel itself together with its immediate neighbors in every direction. Upon flat tening the three window axes yields,

$$
\tilde { K } , \tilde { V } \in \mathbb R ^ { M \times H \times W \times D \times N _ { w } \times d _ { k } }\tag{5}
$$

In other words, $\tilde { K }$ and $\tilde { V }$ make each voxel’s $N _ { w } = 2 7$ neighbors directly available along a dedicated axis, so that the subsequent attention reduces to vectorized multiplications. For head $m ,$ anchor voxel $( x , y , z )$ , and neighbor index $n \in \{ 1 , 2 , . . , N _ { w } \}$ , the unnormalized score $U _ { m }$ is calculated as the product between the $Q$ and the gathered neighbor key $\tilde { K }$ , taken over the $d _ { k }$ channels of the head:

$$
U _ { m } ( x , y , z , n ) = \sum _ { c = 1 } ^ { d _ { k } } Q ( m , x , y , z , c ) { \tilde { K } } ( m , x , y , z , n , c )\tag{6}
$$

Because the zero-padding introduces neighbors that lie outside the true feature grid for voxels near the boundary, an additive mask M removes their influence:

$$
M ( x , y , z , n ) = \left\{ { \begin{array} { l l } { 0 } & { { \mathrm { i f ~ } } ( x + d x , y + d y , z + d z ) \in { \mathcal { B } } , } \\ { C _ { { \mathrm { m a s k } } } } & { { \mathrm { o t h e r w i s e } } } \end{array} } \right.\tag{7}
$$

where B is the set of in-bounds voxel positions of the feature grid and $C _ { \mathrm { m a s k } }$ is a large negative constant $( - 1 0 ^ { 4 } )$ Next, the scores obtained in Eqn. 6 are scaled by a factor of $1 / \sqrt { d _ { k } }$ , augmented with the boundary mask $M ,$ and normalized by a softmax over the $N _ { w }$ neighbors:

$$
S _ { m } ( x , y , z , n ) = \frac { 1 } { \sqrt { d _ { k } } } U _ { m } ( x , y , z , n ) + M ( x , y , z , n )\tag{8}
$$

$$
A _ { m } ( x , y , z , n ) = \operatorname { S o f t m a x } _ { n } \left( S _ { m } ( x , y , z , n ) \right)\tag{9}
$$

where $A _ { m } ( x , y , z , n )$ represents attention weight of each neighbor for the voxel $( x , y , z )$ under the head $m .$ . The

final attention for head m $( A t t n _ { m } \in \mathbb { R } ^ { D \times H \times W \times d _ { k } } )$ is calculated as:

$$
A t t n _ { m } ( x , y , z , d _ { k } ) = \sum _ { n = 1 } ^ { N _ { w } } A _ { m } ( x , y , z , n ) \ { \tilde { V } } ( m , x , y , z , n , d _ { k } )\tag{10}
$$

Likewise, each head produces its output as the attention weighted sum of the gathered neighbor values. The perhead outputs are concatenated and passed through a final point-wise projection $W _ { o }$ to form the aligned feature volume:

$$
F _ { \mathrm { a l i g n e d } } = W _ { o } \binom { M } { | | } \boldsymbol { A } t t { n } _ { m } \rangle\tag{11}
$$

where denotes concatenation across heads. Hence, the CSA module helps the model to isolate attention to a local neighborhood of 27 voxels thereby allowing it to focus only on relevant bilaterally opposite landmarks.

(b) Feature disparity estimation (FDE): Once the CSA module captures the neighborhood information, we need a way to guide the model into learning asymmetric features. Hence, we calculate the feature disparity between $F _ { \mathrm { a l i g n e d } }$ and the encoder output for the flipped input $h _ { i } ,$ where we first apply an $L _ { 2 }$ vector normalization across the channel dimension for both the locally aligned feature map $F _ { \mathrm { a l i g n e d } }$ and the flipped contralateral feature map $h _ { i }$ . The Feature Disparity map $( D _ { \mathrm { f e a t } } )$ is then calculated as

$$
D _ { \mathrm { f e a t } } = 1 . 0 - \left( { \frac { F _ { \mathrm { a l i g n e d } } } { \| F _ { \mathrm { a l i g n e d } } \| _ { 2 } + \epsilon } } \odot { \frac { h _ { i } } { \| h _ { i } \| _ { 2 } + \epsilon } } \right)\tag{12}
$$

where $\odot$ denotes element-wise multiplication and $\epsilon =$ $1 0 ^ { - 6 }$ is added to prevent division-by-zero. The factor 1− (·) yields a disparity map. The outcome $D _ { \mathrm { f e a t } }$ exhibits higher values where the aligned and the contralateral features diverge.

(c) Dual-scale gating (DSG): Ischemic lesions can vary widely in size, ranging from large MCA (Middle Cerebral Artery) obstructions to tiny, isolated focal infarcts spanning only a few voxels. Hence, it is crucial to retain the information about tiny lesions in the deeper features.

To address this, the model processes the feature disparity map $( D _ { \mathrm { f e a t } } )$ through a parallel dual-scale gating module designed to capture both broad regional trends and sharp focal details. As shown in Fig. 1, the disparity map is split into two paths:

1. Gate Small (GS): Employs a convolution block with a kernel size of $1 \times 1 \times 1$ . Since tiny infarcts span only a few voxels, a larger kernel dilutes their signal with surrounding information; the $1 \times 1 \times 1$ kernel instead operates on each voxel independently, preserving these faint signals.

$$
P _ { \mathrm { s m a l l } } = \mathrm { C o n v } _ { 1 \times 1 \times 1 } ( D _ { \mathrm { f e a t } } )\tag{13}
$$

2. Gate Large (GL): Uses a standard $3 \times 3 \times 3$ convolution. Larger lesions span many voxels, so a $3 \times 3 \times 3$ kernel aggregates local context to capture their extent, while avoiding bigger kernels whose cubicallygrowing parameter cost tends to blur lesion boundaries.

$$
P _ { \mathrm { l a r g e } } = \mathrm { C o n v _ { 3 \times 3 \times 3 } } ( D _ { \mathrm { f e a t } } )\tag{14}
$$

The outputs from both paths are recombined via element-wise addition. We then apply Group Normalization (GN) followed by a Sigmoid activation function (σ) to generate a spatial gating map $G \in [ 0 , 1 ]$

$$
G = \sigma \left( \mathrm { G N } \left( P _ { \mathrm { l a r g e } } + P _ { \mathrm { s m a l l } } \right) \right)\tag{15a}
$$

$$
G _ { \mathrm { a l i g n e d } } = G \odot F _ { \mathrm { a l i g n e d } }\tag{15b}
$$

Next, the representation $G _ { \mathrm { a l i g n e d } }$ is concatenated with the original unaltered feature map $g _ { \mathrm { i } }$ . This safeguards rare cases where infarcts appear at bilaterally opposite points. In such cases, since both landmarks look similar, their disparity is low and the gating would suppress the signal. Including $g _ { i }$ preserves information about such lesions. We then apply a convolution to fuse the concatenated features and project the 2C channels back to $C ,$ followed by Group Normalization (GN) and Gaussian Error Linear Unit (GELU) activation function as follows:

$$
f _ { \mathrm { i } } = \mathrm { G E L U } \left( \mathrm { G N } \left( \mathrm { C o n v } _ { 3 \times 3 \times 3 } \left( \left[ G _ { \mathrm { a l i g n e d } } \parallel { g } _ { \mathrm { i } } \right] \right) \right) \right)\tag{16a}
$$

where ∥ denotes tensor concatenation along the channel axis.

## 3.2.3 CNN Decoder

The features $\left( g _ { 1 } , g _ { 2 } , g _ { 3 } , f _ { 4 } , f _ { 5 } \right)$ are passed to the decoder D to construct the final segmentation map. The flipped stream $h _ { i }$ serves only as a contralateral reference for computing inter-hemispheric disparity. Once this comparison is complete, the segmentation is produced in the original, unflipped anatomical space. The decoder progressively upsamples the feature maps using 3D transposed convolutions, recombining them with the corresponding encoder features via skip connections. The output from the final layer contains $N _ { c l s } + 1$ channels, $N _ { c l s }$ being the number of foreground classes, followed by a softmax activation function to produce per-voxel class probabilities. At inference, class predictions are then obtained by taking the argmax of the probabilities.

## 3.2.4 Loss function

During training, the loss function $( \mathcal { L } _ { \mathrm { t o t a l } } )$ used is the sum of Dice loss $\left( \mathcal { L } _ { \mathrm { D i c e } } \right)$ and Cross-Entropy loss $( \mathcal { L } _ { \mathrm { C E } } )$ This loss is implemented at all decoder layers to preserve stroke representations across all resolutions and can be

represented as follows:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \sum _ { i = 1 } ^ { 5 } \alpha _ { i } \left[ \mathcal { L } _ { \mathrm { C E } } ^ { ( i ) } ( y _ { \mathrm { p r e d } } , y _ { \mathrm { g t } } ) + \mathcal { L } _ { \mathrm { D i c e } } ^ { ( i ) } ( y _ { \mathrm { p r e d } } , y _ { \mathrm { g t } } ) \right]\tag{17}
$$

where $\alpha _ { i }$ represents the weight of the loss function at the $i ^ { t h }$ layer of the decoder D.

## 4 Experimental Setup

## 4.1 Datasets

We evaluate the proposed method on three publicly available datasets spanning the acute-to-chronic stroke imaging continuum.

## 4.1.1 Acute Ischemic Stroke Dataset (AISD)

The AISD [27] dataset comprises 397 NCCT scans acquired within 24 hours of symptom onset, with lesions manually delineated using DWI scans (acquired within 24 hours post-CT) as reference and were further reviewed by a senior doctor. In this study, we use only the NCCT scans for training and validation. We use the standard split of 345 training and 52 test scans.

## 4.1.2 Anatomical Tracings of Lesions After Stroke (ATLAS v2.1) dataset

ATLAS v2.1 [9] is a multi-site, T1-weighted MRI dataset of chronic and sub-acute stroke, comprising 1271 volumes from 44 research cohorts. The data are derived from studies that were approved by their local ethics committee and were conducted in accordance with the 1964 Declaration of Helsinki. Informed consent was obtained from all subjects. The ethics committee at the receiving site (the University of Southern California) approved the receipt and sharing of the de-identifed data, which do not contain any personal identifers. We use the 655 publicly available training volumes with their masks, and evaluate on the 300 test volumes for which v2.1 additionally released ground-truth masks that were withheld in v2.0.

## 4.1.3 Ischemic Stroke Lesion Segmentation Challenge (ISLES’24) dataset

ISLES’24 [6, 30] is a multi-center longitudinal dataset providing a pre-interventional CT trilogy (NCCT, CT-Perfusion, CT-Angiography), with lesion masks originally derived from follow-up DWI acquired 2-9 days later. This multi-center study used data from studies approved by their local ethics committees and were conducted in accordance with 1964 Declaration of Helsinki and its updated version 24. Due to defacing and rigorous anonymization, the ethics committee at the receiving site (University of Zurich) approved sharing the de-identified data. ISLES’24 targets prediction of postinterventional infarct from pre-interventional CT, but since infarcts evolve after the initial scan, the released masks do not reflect the pre-interventional state. An expert neuroradiologist therefore manually re-delineated pre-interventional ischemic core and penumbra on the CT perfusion scans using ITK-SNAP [31]. Given a mean onset-to-door time of 3h:28m [30] and the limited sensitivity of NCCT to early ischemic change at this stage [4] [5], we incorporate CT perfusion parameter maps, including Cerebral Blood Volume (CBV), Cerebral Blood Flow (CBF), Mean Transit Time (MTT) and Time-to-Maximum $( T _ { \mathrm { m a x } } )$ , alongside NCCT to improve core and penumbra visibility. As the oficial 96-case test split remains withheld, we partition the available 149 cases into 109 training and 40 test cases.

## 4.2 Implementation details

All experiments were implemented in PyTorch 2.5.1 (CUDA 12.1) within the nnU-Net [13] codebase, using two NVIDIA RTX A6000 GPUs (48 GB each). Models were trained for 160 epochs using Stochastic Gradient Descent (SGD) optimizer with a polynomial learningrate schedule (initial rate $1 \times 1 0 ^ { - 3 }$ , exponent 0.9) and batch size 6. Data augmentations include Gaussian noise $( \mu \ : = \ : 0 , \ : \sigma \ : \in \ : [ 0 . 1 , 0 . 3 ] )$ , Gaussian blur $( \mu = 0$ $\sigma \in [ 0 . 5 , 2 . 5 ] )$ , gamma transform $( \gamma \in [ 0 . 7 , 1 . 5 ] )$ , linear contrast transform $( m \in [ 0 . 7 5 , 0 . 1 2 5 ] )$ .

## 4.3 Evaluation metrics and statistical testing

Segmentation performance is quantified using three complementary metrics: the Dice similarity coeficient measures volumetric overlap between the predicted mask and ground truth, the 95th-percentile Hausdorf Distance (HD95) (mm) quantifies boundary agreement while suppressing sensitivity to small outlier regions, computed as the 95th percentile of the set of nearestneighbor distances between predicted and ground-truth surface points in both directions and lastly Absolute Volume Diference (AVD) (ml) captures the diference be tween the total predicted and the ground truth volume. To assess the statistical significance of the improvement in performance, we perform a paired Wilcoxon signedrank test on a per-case basis for each metric and each dataset, with a significance threshold of $p < 0 . 0 5$

## 5 Experiments and results

## 5.1 Comparison with existing state-ofthe-art methods

On the AISD dataset, we compare against baselines such as nnU-Net [13], Attention U-Net [12], U-Net++ [11], TransUNet [16], Swin-UNETR [14], U-Mamba [17], and

U-KAN [19], all trained within the same nnU-Net codebase for fair comparison. Additionally, existing state-ofthe-art methods are evaluated using their publicly available implementations (e.g., SAN-Net [21], Cl-SegNet [24]) or by replicating methodology in their publications when code is unavailable (MDAN [20]). For methods DySym-UNet [26]) and PAPL [23] we draw indirect comparison using their reported numbers as the AISD test split is standard and shared across all methods. Table 1 reports the results of the comparative analysis. The proposed method achieves a Dice of 0.6796, HD95 of 23.53, AVD of 7.69 and significantly outperforms all compared baselines and stroke-specific methods across Dice, HD95, and AVD $( p \ < \ 0 . 0 5$ , Wilcoxon signedrank test). Our method also outperforms PAPL [23] (Dice 0.5468) and DySym-UNet [26] (Dice 0.6183) by 24.28% and 9.91% respectively whereas U-Mamba (Enc) variant fares the lowest (Dice 0.3451). Fig. 2 shows a visual comparison against existing state-of-the-art ap proaches. U-KAN and SAN-Net miss the lesion entirely in the shown cases, while our method locates it reliably and traces the boundary more accurately than nnU-Net (3D), U-Net++, Attention U-Net, U-Mamba (Bot), and Cl-SegNet. These visual results align with Table 1: the improvement in HD95 reflects the sharper boundaries, while the lower AVD indicates more accurate lesionvolume estimation, clinically important for treatmenteligibility decisions. Our method also has the lowest Dice standard deviation, indicating more consistent performance with fewer critical failures than U-KAN and SAN-Net.

## 5.2 Computational Cost Analysis

To isolate the eficiency gain from restricting CSA to a local neighborhood, we compare it against an otherwise identical full (global) cross-attention variant, difering only in attention scope, profiled at a representative resolution of 14×32×32 (N = 14,336), C = 256 on a single NVIDIA RTX A6000 (48 GB), batch size 1. As summarized in Table 2, both variants share identical parameter counts, confirming the observed savings stem entirely from the attention mechanism rather than model capac ity. Restricting attention to a $3 \times 3 \times 3$ neighborhood reduces attention-specific FLOPs by 531× and total block FLOPs by 28.5×, cutting peak training memory from 24.6 GB to 1.6 GB (15.6×) and inference time by 11.7×. Fig. 3 further shows that global attention scales quadratically $( \mathcal { O } ( N ^ { 2 } ) )$ ) with input size, while the proposed local CSA scales linearly (O(N)), consistent with the complexity argument in Section 3.2.2.

## 5.3 Volumetric Analysis

Lesion volume plays an important role in treatment planning for AIS, with a 70 mL cutof commonly used to determine patients’ eligibility for intravenous thrombolysis and mechanical thrombectomy [32]. We adopt this cutof to stratify infarcts as small or large. As shown in Table 1, our method achieves the highest mean Dice of 0.6131 outperforming every other state-of-theart methods statistically at segmenting small lesions. Fig. 4(a) shows a Bland-Altman plot of predicted versus ground-truth lesion volume on AISD: the mean difference was −5.68 mL, with limits of agreement from −33.30 to 21.93 mL, indicating good agreement with a bias close to zero and spread of diferences remaining relatively contained within the limits of agreement. Fig. 4(b) shows the corresponding scatter plot, yielding a Pearson correlation of $r = 0 . 9 8 5 4 ~ ( p < 0 . 0 0 1 )$ , confirming a strong linear relationship between predicted and ground-truth volumes.

## 5.4 Ablation Studies

We conduct an extensive set of ablation studies on AISD to isolate the contribution of each architectural compo nent; results are summarized in Table 3.

## 5.4.1 Study I - Efectiveness of the overall framework

We evaluate three configurations relative to the full method: (i) single-channel input, no tilt correction, no AsymFeX (plain 5-layer U-Net) (ii) dual-channel input (original and flipped) without tilt correction, retaining AsymFeX, isolating Stage 1’s efect and (iii) dualchannel, tilt-corrected input with AsymFeX removed, isolating AsymFeX’s architectural contribution. Config uration (i) degrades Dice by 8.1% and HD95 by 34.5% relative to the full method, confirming the overall framework’s efectiveness. Configuration (ii) degrades Dice by 5.6% and HD95 by 52.27%, which is a sharper HD95 drop than removing AsymFeX alone, confirming tilt correction is a necessary prerequisite for exploiting the con tralateral symmetry prior. Configuration (iii) degrades Dice by 4.9% and HD95 by 27.79%. Notably, dualchannel input without AsymFeX (Table 3, row 3) improves only marginally over the single-channel baseline (Dice 0.6463 vs. 0.6246), while the full method recovers substantially more (Dice 0.6796), indicating the gain stems primarily from the proposed comparison mechanism rather than additional input alone. For all configurations, the improvements are statistically significant $\left( p < 0 . 0 5 \right)$

## 5.4.2 Study II - Placement of the AsymFeX modules

We assess two configurations to assess AsymFeX’s contribution at each encoder scale (layers 4 and 5): retain ing AsymFeX only at layer 5 (removed from layer 4), and retaining it only at layer 4 (removed from layer 5). The former degrades Dice by 3.0% and increases HD95 by 14.49%; the latter degrades Dice by 2.2% and increases

Table 1: Quantitative comparison of the proposed method against existing approaches on AISD and small lesions of AISD using 70mL as cutof value. Best result in each column is highlighted in bold. ↑: higher is better; ↓: lower is better. \* denotes statistically significant diference in performance with p-value < 0.05 using Wilcoxon signed rank test.
<table><tr><td rowspan="2">Method</td><td colspan="4">AISD (Overall)</td><td colspan="4">AISD - Small Lesions (volume ≤ 70mL)</td></tr><tr><td>Dice ↑  $( \mathrm { m e a n } \pm \mathrm { s t d . } )$ </td><td> $\overline { { { \bf H } { \bf D 9 5 } \left( \bf m \bf n \right) \downarrow } }$   $( \mathrm { m e a n } \pm \mathrm { s t d . } )$ </td><td colspan="2">AVD (mL)↓  $( \mathrm { m e a n } \pm \mathrm { s t d . } )$ </td><td>Dice ↑  $( \mathrm { m e a n } \pm \mathrm { s t d . } )$ </td><td>HD95 (mm) ↓  $( \mathrm { m e a n } \pm \mathrm { s t d . } )$ </td><td>AVD (mL)↓</td><td>(mean ± std.)</td></tr><tr><td>nnU-Net (3D) [13]</td><td> $\overline { { 0 . 5 7 6 5 \pm 0 . 2 2 4 8 } }$ </td><td> $\overline { { 4 9 . 9 3 5 0 \pm 2 7 . 5 3 2 3 } }$ </td><td> $\overline { { 1 2 . 0 0 5 5 \pm 1 6 . 9 1 3 5 } }$ </td><td></td><td> $\overline { { 0 . 4 9 6 3 \pm 0 . 2 6 7 2 } }$ </td><td> $\overline { { 5 2 . 9 7 6 7 \pm 2 8 . 9 3 0 5 } }$ </td><td></td><td> $\overline { { 9 . 7 2 7 9 \pm 1 3 . 0 7 9 3 } }$ </td></tr><tr><td>nnU-Net (2D) [13]</td><td> $0 . 5 3 9 1 \pm 0 . 2 3 5 3$ </td><td> $3 0 . 4 9 9 7 \pm 2 2 . 0 3 0 4$ </td><td></td><td> $1 1 . 4 3 4 4 \pm 1 3 . 9 0 2 8$ </td><td> $0 . 4 6 8 5 \pm 0 . 2 5 1 9$ </td><td> $3 4 . 2 8 1 9 \pm 2 3 . 8 4 2 1$ </td><td></td><td> $9 . 2 1 4 0 \pm 1 1 . 5 3 4 4$ </td></tr><tr><td>Attention U-Net [12]</td><td> $0 . 5 9 1 0 \pm 0 . 2 0 8 3$ </td><td> $3 8 . 1 0 4 9 \pm 2 8 . 0 9 6 9$ </td><td> $1 3 . 4 6 5 8 \pm 2 4 . 3 4 2 9$ </td><td></td><td> $0 . 5 1 2 7 \pm 0 . 2 6 2 9$ </td><td> $4 1 . 1 6 5 9 \pm 3 0 . 2 0 0 4$ </td><td></td><td> $9 . 0 7 1 1 \pm 1 3 . 1 3 0 2$ </td></tr><tr><td>U-Net++ [11]</td><td> $0 . 5 5 1 9 \pm 0 . 2 2 9 3$ </td><td> $4 3 . 0 0 6 9 \pm 2 6 . 1 5 2 6$ </td><td> $1 4 . 8 8 4 5 \pm 2 4 . 7 5 1 1$ </td><td></td><td> $0 . 4 6 2 0 \pm 0 . 2 5 4 0$ </td><td> $4 6 . 0 4 3 7 \pm 2 6 . 0 4 5 7$ </td><td></td><td> $1 1 . 2 3 6 5 \pm 1 3 . 3 0 8 4$ </td></tr><tr><td>TransUNet [16]</td><td> $0 . 3 5 7 7 \pm 0 . 2 3 4 6$ </td><td> $5 6 . 0 8 3 9 \pm 1 6 . 1 3 6 4$ </td><td></td><td> $2 2 . 3 2 6 2 \pm 2 6 . 4 0 5 5$ </td><td> $0 . 2 8 4 0 \pm 0 . 2 1 6 0$ </td><td> $5 7 . 7 6 8 3 \pm 1 6 . 4 3 3 6$ </td><td></td><td> $1 8 . 7 2 0 2 \pm 1 7 . 1 0 5 9$ </td></tr><tr><td>SwinUNeTR [15]</td><td> $0 . 4 8 1 3 \pm 0 . 2 2 5 5$ </td><td> $4 9 . 3 9 9 2 \pm 2 3 . 2 9 8 7$ </td><td></td><td> $1 7 . 3 2 8 9 \pm 3 0 . 6 5 5 9$ </td><td> $0 . 4 1 0 4 \pm 0 . 2 5 1 5$ </td><td> $5 1 . 5 5 5 0 \pm 2 3 . 0 1 8 0$ </td><td></td><td> $1 0 . 6 8 7 8 \pm 1 4 . 4 0 2 5$ </td></tr><tr><td>U-Mamba Bot [17]</td><td> $0 . 5 4 7 3 \pm 0 . 2 2 9 5$ </td><td> $4 5 . 0 5 2 3 \pm 2 3 . 0 6 8 1$ </td><td></td><td> $1 4 . 0 9 4 9 \pm 1 9 . 3 9 1 9$ </td><td> $0 . 4 7 3 9 \pm 0 . 2 5 9 9$ </td><td> $4 7 . 3 2 4 3 \pm 2 2 . 8 4 8 2$ </td><td></td><td> $1 0 . 7 3 3 8 \pm 1 0 . 4 7 4 6$ </td></tr><tr><td>U-Mamba Enc [17]</td><td> $0 . 3 4 5 1 \pm 0 . 2 5 6 8$ </td><td> $6 2 . 7 2 3 9 \pm 2 0 . 1 9 6 7$ </td><td> $2 7 . 0 1 0 4 \pm 2 9 . 1 1 6 3$ </td><td></td><td> $0 . 2 4 8 0 \pm 0 . 2 2 9 8$ </td><td> $6 5 . 8 8 0 1 \pm 1 9 . 9 7 7 5$ </td><td></td><td> $2 2 . 5 7 4 8 \pm 1 5 . 4 8 2 7$ </td></tr><tr><td>U-KAN [19]</td><td> $0 . 5 6 1 3 \pm 0 . 2 2 0 9$ </td><td> $3 0 . 8 6 7 2 \pm 2 3 . 8 2 6 2$ </td><td> $9 . 8 3 1 9 \pm 1 3 . 3 0 5 6$ </td><td></td><td> $0 . 4 7 5 0 \pm 0 . 2 5 9 1$ </td><td> $3 4 . 2 9 0 6 \pm 2 5 . 1 5 3 3$ </td><td></td><td> $8 . 1 0 4 0 \pm 1 0 . 8 9 7 4$ </td></tr><tr><td>MDAN [20]</td><td> $0 . 4 7 0 3 \pm 0 . 2 3 0 4$ </td><td> $4 4 . 7 1 4 2 \pm 2 1 . 7 9 8 1$ </td><td> $1 3 . 8 4 0 9 \pm 2 2 . 7 2 5 9$ </td><td></td><td> $0 . 3 9 3 3 \pm 0 . 2 5 0 8$ </td><td> $4 7 . 3 5 1 1 \pm 2 2 . 3 2 5 5$ </td><td></td><td> $8 . 9 9 1 9 \pm 1 2 . 2 3 7 2$ </td></tr><tr><td>SAN-Net [21]</td><td> $0 . 5 0 3 0 \pm 0 . 2 2 0 7$ </td><td> $3 8 . 9 8 8 7 \pm 2 4 . 1 2 3 5$ </td><td></td><td> $1 6 . 2 8 9 3 \pm 3 3 . 1 3 1 9$ </td><td> $0 . 4 4 1 2 \pm 0 . 2 5 0 5$ </td><td> $4 1 . 0 9 2 9 \pm 2 4 . 8 9 1 8$ </td><td></td><td> $8 . 3 9 3 1 \pm 1 0 . 7 9 2 5$ </td></tr><tr><td>Cl-SegNet [24] Our Method</td><td> $0 . 6 1 5 4 \pm 0 . 2 1 0 1$ </td><td> $2 9 . 2 0 7 3 \pm 2 4 . 4 4 4 4$ </td><td></td><td> $1 0 . 7 9 4 4 \pm 1 4 . 8 1 1 4$ </td><td> $0 . 5 5 1 3 \pm 0 . 2 6 5 5$ </td><td> $3 4 . 1 2 5 6 \pm 2 8 . 0 7 6 9$ </td><td></td><td> $8 . 7 9 2 1 \pm 1 1 . 2 9 6 4$ </td></tr><tr><td></td><td> $\mathbf { 0 . 6 7 9 6 \pm 0 . 1 6 2 2 ^ { \ast } }$ </td><td> $\mathbf { 2 3 . 5 3 0 2 \pm 2 1 . 7 2 2 2 ^ { \ast } }$ </td><td> $\mathbf { 7 . 6 8 8 3 \pm 1 3 . 0 8 4 3 ^ { \ast } }$ </td><td></td><td> $\mathbf { 0 . 6 1 3 1 \pm 0 . 2 3 1 7 ^ { \ast } }$ </td><td> $\mathbf { 2 6 . 0 1 9 0 \pm 2 4 . 1 6 9 7 ^ { \ast } }$ </td><td></td><td> $\mathbf { 5 . 9 9 0 0 \pm 1 0 . 7 7 7 0 ^ { \ast } }$ </td></tr></table>

![](images/91c33fe4d5f4b661e591f239b8fafaa6184b9e06bc324ade13faafc774cad939.jpg)  
Figure 2: Qualitative comparison of segmentation outputs: Visual results of comparison of our method with existing methods along with Uncertainty maps on ATLASv2.1 (row 1), ISLES’24 (row 2 - Penumbra, row 3 - Core) and AISD (row 4).

HD95 by 9.7%, indicating that asymmetric cues at both spatial scales contribute complementary information.

## 5.4.3 Study III - Efectiveness of sub-modules under AsymFeX

We evaluate three configurations, each removing or replacing one AsymFeX sub-module while retaining the other two: (i) CSA removed entirely (ii) FDE replaced with naive element-wise feature subtraction and (iii) DSG removed entirely, with performance additionally assessed via the Absolute Lesion Count Diference (ALCD) to specifically test DSG’s role in preserving small, discrete lesions. Removing CSA significantly degrades Dice by 6.32% $\left( p < 0 . 0 5 \right)$ and HD95 by 37.76%, the largest single-component drop observed, underscoring the importance of localized cross-hemispheric attention. Replacing FDE with naive subtraction degrades Dice by 2.9% and HD95 by 24.63%. Removing DSG significantly degrades Dice by 3.7% $( p \ < \ 0 . 0 5 )$ and HD95 by 36.52%. The overall mean ALCD also rises significantly $( p < 0 . 0 5 )$ from 2.19 to 3.48 (a 37.0% increase). Furthermore, applying the same 70 mL threshold we evaluate the w/o DSG configuration on small lesions (≤ 70mL), and obtain a mean dice of 0.5877, mean HD95 of 34.02 and mean ALCD of 3.48. This is a significant degradation of performance in segmenting small lesions compared to our full configuration (Table

![](images/4e19254ce2e60a01a1420d5886768a8f9d9a910f524177636ff92522e5fcdb7a.jpg)  
Figure 3: Attention cost versus input size $( N = D \times$ $H \times W )$ for full attention (red) and CSA (blue). Full attention scales quadratically $( \mathcal { O } ( N ^ { 2 } ) )$ while our CSA scales linearly $( \mathcal O ( N ) )$ .

Table 2: Cost of local (ours) vs. full cross-attention at $1 4 \times 3 2 \times 3 2 , C = 2 5 6 ( N = 1 4 , 3 3 6 )$ , batch size 1, single GPU. Ratio = global / local. FLOPs use $\mathrm { M A C } = 2$ operations.
<table><tr><td>Metric</td><td>Local (Ours)</td><td>Global</td><td>Ratio</td></tr><tr><td>Parameters</td><td>263,168</td><td>263,168</td><td>1.0×</td></tr><tr><td>Total FLOPs (G)</td><td>7.93</td><td>226.19</td><td>28.5×</td></tr><tr><td>Attention FLOPs (G)</td><td>0.41</td><td>218.67</td><td>531×</td></tr><tr><td>Inference time (ms)</td><td>6.17</td><td>72.42</td><td>11.7×</td></tr><tr><td>Training time (ms)</td><td>27.39</td><td>227.27</td><td>8.3×</td></tr><tr><td>Inference peak memory (GB)</td><td>1.25</td><td>12.36</td><td>9.9×</td></tr><tr><td>Train peak memory (GB)</td><td>1.58</td><td>24.60</td><td>15.6×</td></tr></table>

1) indicating DSG’s particular importance for correctly identifying small, discrete lesions.

We additionally conduct an additive ablation: (i) CSA alone, (ii) CSA+FDE (equivalent to $\mathrm { w / o D S G ) }$ , and (iii) the full CSA+FDE+DSG method. CSA alone degrades Dice by 5.02% and HD95 by 9.7%. Adding FDE improves Dice to 0.6544 but temporarily raises HD95 to 32.12, indicating that while FDE sharpens the disparity signal, the model responds with inflated boundary error without DSG to regulate it. Adding DSG resolves this, yielding the full model’s Dice of 0.6796 and HD95 of 23.53. Both intermediate diferences along this additive path are statistically significant, with the gap narrowing progressively as each module is added, confirming each component’s contribution to AsymFeX’s overall effectiveness.

## 5.4.4 Study IV - Selection of Spatial Neighborhood

To justify the $3 \times 3 \times 3 \ \mathrm { C S A }$ neighborhood, we evaluate two alternatives: ${ \mathrm { ~ a ~ 1 ~ } } \times { \mathrm { ~ 1 ~ } } \times { \mathrm { ~ 1 ~ } }$ window, where each voxel in Q attends only to its exact counterpart in K with no neighborhood context, and an enlarged $5 \times 5 \times 5$ window, testing whether additional context is beneficial or introduces noise from distant tissue. The $1 \times 1 \times 1$ window degrades Dice by 2.2% and HD95 by 13.15%, confirming that neighborhood context is essential to accommodate minor misalignments and anatomical shifts. The $5 \times 5 \times 5$ window degrades Dice by 3.1% and HD95 by 13.70%, likely from incorporating redundant information from distant, healthy tissue. Both diferences in Dice values are significant $( p < 0 . 0 5 )$ implying that $3 \times 3 \times 3$ configuration is thus both necessary and sufficient, indicating an eficiency-accuracy tradeof point rather than an arbitrary choice.

![](images/02b20e34835cde2e865523fe14f99b3349619368363b447593d3fe6daaaf4d17.jpg)

![](images/ab58b24265817233261d40f7092102e5cc8bcfc83126c7acfebb7646094d6d6a.jpg)  
Figure 4: Volumetric Analysis for AISD (top) Bland-Altman plot comparing predicted volume with the ground truth, (bottom) Scatter plot of the predicted volumes versus the ground truth.

![](images/ff1102ab9bc600fc29e24545f47a328aae4b5ac413af2cb18ff44d0732f9370d.jpg)

![](images/651b8ed347202ab63e99de93f9c5e485a856ec7f79c0838e7f406cdf7d6c0783.jpg)

![](images/0945e4c44714042bccd2434f5d9204e2f963ec847ce01d6455ec5ccf4167c7b5.jpg)

![](images/a0f75fe418a92478267ef7cae31f8360b41cd283baa16f40397ecaafa4277ea8.jpg)  
Figure 5: Calibration plots. Bars show the observed foreground frequency per bin, red markers show the mean predicted probability and diagonal denotes perfect calibration. Deviations from the bars indicate miscalibration. The corresponding ECE values are reported in Table 5.

Table 3: Ablation study on AISD. Best result is highlighted in bold.
<table><tr><td>Ablation Experiment</td><td>Dice ↑</td><td>HD95↓</td></tr><tr><td>Single-channel,  $\mathrm { w / o }$  tilt correction,  $\mathrm { w / o }$  AsymFex</td><td>0.6246</td><td>31.6476</td></tr><tr><td>Dual-channel,  $\mathrm { w / o }$  tilt correction, with AsymFeX</td><td>0.6415</td><td>35.8302</td></tr><tr><td>Dual-channel, with tilt correction,  $\mathrm { w } / \mathrm { o }$  AsymFeX AsymFeX only in layer 5</td><td>0.6463 0.6590</td><td>30.0690 26.9394</td></tr><tr><td>AsymFeX only in layer 4 w/o CSA</td><td>0.6648 0.6366</td><td>25.8136 32.4149</td></tr><tr><td>with CSA only w/o FDE with CSA + FDE (same as w/o DSG)</td><td>0.6454 0.6602 0.6544</td><td>27.3981 29.3264 32.1235</td></tr><tr><td>AsymFeX with neighborhood size  $\overline { { 1 ^ { 3 } } }$  AsymFeX with neighborhood size  $5 ^ { 3 }$ </td><td>0.6644 0.6585</td><td>26.6024 26.7541</td></tr><tr><td>Our Method</td><td>0.6796</td><td>23.5302</td></tr></table>

## 5.5 Modality-Specific Adaptations and Cross-Dataset Comparison Protocol

The core premise of the proposed framework is that loss of left-right hemispheric symmetry during an ischemic event comes from brain anatomy and is observable across imaging modalities and stroke timepoints, from acute hypoattenuation on NCCT, to hypoperfusion on CT perfusion, to chronic infarction on MRI. To assess whether the proposed architecture generalizes across this continuum across multiple modalities without modification, we evaluate it on ATLAS v2.1 [9] (sub-acute and chronic, T1w MRI) and ISLES’24 [6, 30] (acute, CT perfusion), adapting only the Stage 1 preprocessing to each modality’s input characteristics.

For ATLAS v2.1, a single-modality T1w dataset, the network receives the original volume and its flipped copy, identical to the primary AISD configuration. Since the dataset already had pre-processed T1-w MRI in MNI space, additional tilt-correction is not required. For ISLES’24, the tilt angle and translation are computed once from the NCCT channel and applied identically to the co-registered CBV, CBF, MTT, and $T _ { \mathrm { m a x } }$ maps, rather than re-derived per channel, ensuring all channels remain spatially consistent after alignment. The encoder then receives all five channels (NCCT, CBV, CBF, MTT, $T _ { \mathrm { m a x } } )$ alongside their corresponding flipped copies. No further architectural modification is made in either case.

Having already reported a full ablation on AISD (Section 5.4), we restrict this evaluation to a focused comparison against the five best-performing baseline methods from the AISD benchmark, together with two methods designed specifically for each target modality. As reported in Table 4, the proposed method achieves the highest Dice on the infarct class on both ATLAS and ISLES’24, provides within 0.01 Dice of the bestperforming method (nnU-Net) on the penumbra class, and attains the lowest HD95 across all compared methods on both datasets. On ATLAS, our method performs statistically better than all compared methods $( p \ <$ 0.05). Also, on ISLES’24, without any additional architectural changes beyond the input-fusion strategy described above, our method performed on par or provided better results compared to existing methods. This superior performance relative to methods purpose-built for each modality, achieved without any modality-specific architectural tuning, instead supports the generality of the proposed asymmetry-driven design across the stroke imaging continuum, which is the central claim of this proof-of-concept evaluation.

## 5.6 Perfusion Necessity on ISLES’24

Given the short mean onset-to-door time on ISLES’24 (3h:28m [30]), ischemic infarcts typically lack suficient hypoattenuation to be visible on NCCT within this interval. To verify if NCCT alone is inadequate at this time point, we segment AIS lesions using NCCT only, keeping architecture and training protocol unchanged, for our method and the two best-performing methods from Table 1. All three methods collapse to a nearidentical, near-failing Dice on NCCT alone (nnU-Net (3D): $0 . 1 7 6 6 { \pm } 0 . 1 4 9 2 ;$ Attention U-Net: $0 . 1 7 6 3 { \pm } 0 . 1 4 8 1$ ; Ours: 0.1786±0.1498), with HD95 uniformly poor across all three (159-163 mm), compared to a Dice of 0.6354 once perfusion maps are included. This near-identical collapse across architectures indicates the limitation is intrinsic to NCCT’s visibility of AIS lesions at this time point rather than any specific architectural shortcoming, confirming that CT perfusion parameter maps are necessary, not merely beneficial, for ISLES’24.

## 5.7 Uncertainty Analysis

To evaluate the reliability of the model’s predictions, we quantify predictive uncertainty using Monte Carlo (MC) dropout at inference. Dropout layers (rate 10%) placed after the final AsymFeX block in the last two encoder layers remain active during testing, and each test case is passed through 30 stochastic forward passes. We validate the resulting uncertainty estimates via two complementary analyses: an Error-Uncertainty analysis, testing whether high-uncertainty regions coincide with actual segmentation errors, and a Calibration analysis, testing whether predicted confidence matches observed correctness.

## 5.7.1 Error-Uncertainty Analysis

A reliable model should exhibit high uncertainty precisely where it errs. We evaluate this within a region of interest (ROI) defined as the union of the predicted and ground-truth masks, dilated slightly to exclude trivial background voxels. Within this ROI, each voxel’s uncertainty is treated as a predictor of segmentation error, and we report the Area Under the ROC Curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC) against the corresponding error prevalence. We also report mean uncertainty stratified by True Positive (TP), False Positive (FP), False Negative (FN), and True Negative (TN) voxels. As shown in Table 5, AUROC exceeds 0.70 and AUPRC exceeds the corresponding prevalence for every dataset, confirming that predicted uncertainty identifies erroneous voxels well above chance. Moreover, mean uncertainty is consistently higher for error categories (FP, FN) than for correct categories (TP, TN) across all datasets.

Table 4: Quantitative comparison of the proposed method against existing approaches on ATLAS v2.1 and ISLES’24. Symbol \* denotes statistically significant diference in performance with p-value $< 0 . 0 5$ using Wilcoxon signed rank test.  
(a) ATLAS v2.1
<table><tr><td>Method</td><td>Dice ↑  $( \mathrm { m e a n } \pm \mathrm { s t d . } )$ </td><td>HD95 (mm) ↓  $( \mathrm { m e a n } \dot { \pm } \mathrm { s t d . } )$ </td></tr><tr><td>nnUNet (3D) [13]</td><td> $\overline { { 0 . 6 1 3 1 \pm 0 . 2 7 1 9 } }$ </td><td> $\overline { { 2 5 . 9 9 3 1 \pm 2 5 . 8 4 7 4 } }$ </td></tr><tr><td>Attention U-Net [12]</td><td> $0 . 6 1 0 1 \pm 0 . 2 6 7 7$ </td><td> $2 6 . 4 3 5 3 \pm 2 6 . 4 7 6 8$ </td></tr><tr><td>U-Net++ [11]</td><td> $0 . 6 2 1 0 \pm 0 . 2 6 9 9$ </td><td> $2 5 . 3 8 5 7 \pm 2 7 . 4 0 4 3$ </td></tr><tr><td>U-Mamba Bot [17]</td><td> $0 . 6 1 6 8 \pm 0 . 2 6 9 4$ </td><td> $2 5 . 6 7 4 2 \pm 2 6 . 5 2 1 4$ </td></tr><tr><td>U-KAN [19]</td><td> $0 . 5 8 5 7 \pm 0 . 2 8 8 5$ </td><td> $2 4 . 0 2 9 6 \pm 2 5 . 7 5 6 8$ </td></tr><tr><td>MDAN [20]</td><td> $0 . 4 8 8 4 \pm 0 . 2 8 1 7$ </td><td>32.6047 ± 26.6804</td></tr><tr><td> $\mathrm { S A N - N e t ~ [ 2 1 ] }$ </td><td> $0 . 5 0 7 3 \pm 0 . 3 1 1 4$ </td><td> $3 3 . 6 2 2 9 \pm 3 0 . 6 9 7 5$ </td></tr><tr><td>Our Method</td><td> $\mathbf { 0 . 6 3 5 3 \pm 0 . 2 5 9 7 ^ { \ast } }$ </td><td> $\mathbf { 2 0 . 9 5 7 8 \pm 2 3 . 4 6 7 6 ^ { \ast } }$ </td></tr></table>

(b) ISLES’24 (Pre-treatment AIS segmentation)
<table><tr><td rowspan=2 colspan=1>Method</td><td rowspan=1 colspan=2>Infarct</td><td rowspan=1 colspan=4>Penumbra</td></tr><tr><td rowspan=1 colspan=1>Dice ↑ $( \mathrm { m e a n } \pm \mathrm { s t d . } )$ </td><td rowspan=1 colspan=1>HD95 (mm) ↓ $( \mathrm { m e a n } \stackrel { . } { \pm } \mathrm { s t d } . )$ </td><td rowspan=1 colspan=1>Dice ↑ $( \mathrm { m e a n } \pm { \mathrm { { \dot { s t d } } } } . )$ </td><td rowspan=1 colspan=3>HD95 (mm)↓ $( \mathrm { m e a n } \stackrel { . } { \pm } \mathrm { s t d } . )$ </td></tr><tr><td rowspan=1 colspan=1>nnUNet (3D) [13]</td><td rowspan=1 colspan=1>0.5885 ± 0.2708</td><td rowspan=1 colspan=1> $\overline { { 1 9 . 0 9 8 0 \pm 2 2 . 2 4 6 7 } }$ </td><td rowspan=1 colspan=1>0.6340 ± 0.2233</td><td rowspan=1 colspan=3> $\overline { { 2 2 . 1 3 1 4 \pm 2 3 . 5 8 3 7 } }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { A t t e n t i o n ~ U \mathrm { - N e t } \dot { \ } [ 1 2 ] }$ </td><td rowspan=1 colspan=1> $0 . 5 6 7 2 \pm 0 . 2 8 5 2$ </td><td rowspan=1 colspan=1> $2 2 . 4 3 9 9 \pm 2 5 . 0 4 3 5$ </td><td rowspan=1 colspan=1> $0 . 6 2 4 2 \pm 0 . 2 2 0 1$ </td><td rowspan=1 colspan=3> $2 7 . 7 2 7 1 \pm 2 5 . 9 9 8 4$ </td></tr><tr><td rowspan=2 colspan=1> $\mathrm { U - N e t + + \ [ 1 1 ] }$  $\mathrm { U - M a m b a ~ B o t ~ [ 1 7 ] }$ </td><td rowspan=1 colspan=1> $0 . 5 6 8 2 \pm 0 . 2 7 5 4$ </td><td rowspan=1 colspan=1> $1 9 . 6 2 0 1 \pm 2 2 . 7 1 9 1$ </td><td rowspan=1 colspan=1> $0 . 6 1 3 9 \pm 0 . 2 1 5 1$ </td><td rowspan=1 colspan=3> $2 6 . 7 0 8 2 \pm 2 5 . 1 9 1 9$ </td></tr><tr><td rowspan=1 colspan=1> $0 . 6 1 9 1 \pm 0 . 2 5 9 8$ </td><td rowspan=1 colspan=1> $1 5 . 4 2 3 8 \pm 1 9 . 4 7 7 1$ </td><td rowspan=1 colspan=1> $0 . 6 1 5 8 \pm 0 . 2 3 8 8$ </td><td rowspan=1 colspan=1> $2 3 . 0 3 5 8</td><td rowspan=1 colspan=2>\pm 2 5 . 4 4 2 4$ </td></tr><tr><td rowspan=1 colspan=1>U-KAN [19]</td><td rowspan=1 colspan=1> $0 . 5 6 8 6 \pm 0 . 2 8 1 1$ </td><td rowspan=1 colspan=1> $1 7 . 1 8 0 9 \pm 1 6 . 9 2 1 3$ </td><td rowspan=1 colspan=1> $0 . 5 8 8 9 \pm 0 . 2 2 8 0$ </td><td rowspan=1 colspan=1>20.19</td><td></td><td rowspan=1 colspan=1>19.2605</td></tr><tr><td rowspan=1 colspan=1>ISP-Net [33]</td><td rowspan=1 colspan=1> $0 . 5 2 3 0 \pm 0 . 2 8 0 1$ </td><td rowspan=1 colspan=1> $2 0 . 4 9 6 0 \pm 2 1 . 2 9 0 7$ </td><td rowspan=1 colspan=1> $0 . 5 8 8 9 \pm 0 . 2 0 8 7$ </td><td rowspan=1 colspan=3> $2 3 . 0 4 4 6 \pm 2 2 . 1 9 6 3$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { I 2 P C { - } N e t \ [ 3 4 ] }$ </td><td rowspan=1 colspan=1> $0 . 5 9 0 1 \pm 0 . 2 7 7 1$ </td><td rowspan=1 colspan=1> $1 6 . 2 5 8 6 \pm 1 9 . 3 0 4 4$ </td><td rowspan=1 colspan=1> $0 . 6 2 0 9 \pm 0 . 2 4 6 9$ </td><td rowspan=1 colspan=3> $2 2 . 9 5 5 4 \pm 2 3 . 0 7 3 8$ </td></tr><tr><td rowspan=1 colspan=1>Our Method</td><td rowspan=1 colspan=1> $\mathbf { \overline { { 0 . 6 3 5 4 } } } \pm \mathbf { 0 . 2 5 2 0 }$ </td><td rowspan=1 colspan=1> $\mathbf { 1 5 . 1 8 4 6 \ : \pm { \ : 1 7 . 2 9 9 0 } }$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 6 3 3 9 \pm 0 . 2 3 4 8 } }$ </td><td rowspan=1 colspan=3> $\mathbf { 1 3 . 8 6 4 2 \pm 1 3 . 7 7 5 5 }$ </td></tr></table>

## 5.7.2 Calibration Analysis.

To assess whether predicted probabilities match observed correctness, Fig. 5 shows reliability diagrams for each dataset (observed frequency vs. the diagonal line of perfect calibration), with the corresponding Expected Calibration Error (ECE) reported in Table 5; both are computed over the ROI defined in Section 5.7.1. AISD shows the strongest calibration among the three datasets, while ATLAS and ISLES’24 are moderately miscalibrated. The higher ECE on ISLES’24 is consistent with genuine ambiguity at the infarct-penumbra boundary, which is typically difuse rather than sharp, producing many inherently ambiguous voxels; this is further supported by the higher mean uncertainty of correctly classified voxels on ISLES’24 relative to AISD and ATLAS (Table 5). ISLES’24’s smaller training set (109 cases) could also have contributed to this miscalibration.

Taken together with the Dice results in Sections 5.5 and 5.6, AISD achieves both the highest accuracy and the sharpest boundary confidence among the three datasets, while ISLES’24 attains comparable Dice but markedly higher boundary uncertainty. This indicates that the method is not merely equally accurate across the stroke continuum, but less confidently so at earlier, perfusion-dependent time points where the underlying tissue distinction is itself more ambiguous.

Table 5: Uncertainty metrics per dataset. All values are computed within the foreground ROI. Every metric is calculated over the MC-averaged T=30 passes probabilities.
<table><tr><td rowspan="2">Metric</td><td rowspan="2">AISD</td><td rowspan="2">ATLAS v2.1</td><td colspan="2">ISLES&#x27;24</td></tr><tr><td>Infarct</td><td>Penumbra</td></tr><tr><td>AUROC</td><td>0.8102</td><td>0.7277</td><td>0.7706</td><td>0.7053</td></tr><tr><td>AUPRC</td><td>0.3678</td><td>0.3832</td><td>0.4819</td><td>0.4357</td></tr><tr><td>Prevalence</td><td>0.1296</td><td>0.1947</td><td>0.2515</td><td>0.2837</td></tr><tr><td>Mean uncertainty</td><td></td><td></td><td></td><td></td></tr><tr><td>TP</td><td>0.1719</td><td>0.2206</td><td>0.3028</td><td>0.4545</td></tr><tr><td>TN</td><td>0.0739</td><td>0.0896</td><td>0.2479</td><td>0.3070</td></tr><tr><td>FP</td><td>0.4999</td><td>0.4776</td><td>0.6566</td><td>0.6624</td></tr><tr><td>FN</td><td>0.2880</td><td>0.2425</td><td>0.4825</td><td>0.5498</td></tr><tr><td>ECE</td><td>0.0807</td><td>0.1437</td><td>0.1694</td><td>0.1660</td></tr></table>

## 6 Discussion and Conclusion

In this study, we propose a two-stage, 3D AIS segmentation pipeline that incorporates a stroke-specific prior and integrates directly with nnU-Net. At its core is AsymFeX, which combines local 3D cross-hemispheric attention, disparity estimation, and dual-scale gating, restricting comparison to a local neighborhood around each voxel’s true contralateral landmark rather than the entire opposite hemisphere, closely mirroring how a radiologist compares corresponding regions across hemispheres in practice. Across an extensive evaluation on AISD, our method achieves the best Dice, HD95, and AVD among nine baselines and three stroke-specific state-of-the-art methods, with gains statistically signifi cant via Wilcoxon signed-rank testing (Table 1). Ablation confirms that both stages of the pipeline are necessary, bypassing tilt correction alone degrades HD95 more sharply than removing AsymFeX entirely, and dualchannel tilt-corrected input without AsymFeX yields only marginal gains over a single-channel baseline, confirming that the observed improvement stems from the proposed comparison mechanism rather than additional input alone (Table 3). Volumetric analysis shows strong agreement with ground-truth lesion volume (Pearson r = 0.9854, Fig. 4), directly relevant to the 70 mL threshold used clinically to determine thrombolysis and thrombectomy eligibility.

Proof-of-concept evaluation on ATLAS v2.1 and ISLES’24 shows the same architecture, adapted only at the input level, remains competitive with modalityspecialized methods without any architectural retuning. The perfusion-necessity experiment on ISLES’24 confirms that this adaptation is not incidental: NCCT alone collapses to near-failing performance at this timepoint regardless of architecture, underscoring that imaging modality, not model design, is the binding constraint at early stroke stages. Uncertainty analysis further shows that boundary confidence and accuracy varies across the stroke continuum. AISD yields both the highest accuracy and the sharpest boundary confidence (Table 5), while ISLES’24 achieves comparable Dice with markedly higher boundary uncertainty, indicating the model is not equally confident across all time points even where it is equally accurate. Collectively, these results carry clear implications for clinical translation. The strong volumetric agreement near the 70 mL thrombolysis/thrombectomy eligibility threshold [35] suggests the method could support rapid, automated volume estimation at triage, where treatment delay directly reduces salvageable penumbra [35] [36]. Generalization across NCCT, CT perfusion, and MRI without architectural redesign is relevant given that multiple modalities could be acquired for a given patient depending on onset time or to gather all the necessary information for efective treatment planning [4], favoring a single adaptable pipeline over separate per-modality models. The uncertainty framework further ofers calibrated clinical trust rather than a bare prediction, particularly relevant given that boundary confidence degrades at earlier, more treatment-critical stroke stages.

The proposed framework assumes the two hemispheres are approximately mirror-symmetric prior to infarction, an assumption more likely violated by substantial midline shift or, in chronic cases, encephalomalacia and ventricular enlargement unrelated to the acute event, both of which may compromise Stage 1 alignment and downstream AsymFeX comparison. Also, the dual-stream encoder also incurs meaningfully greater inference cost than a single-stream baseline despite the eficiency of windowed attention (Section 5.2), which may matter in resource-constrained settings. Three future research directions follow naturally. Using uncertainty estimates potentially from reporting to an op erational role, flagging low-confidence cases for review, indicating when an additional modality is necessary, or guiding uncertainty-aware continual learning. Given the cost of manual delineation, training under weak supervision, using bounding boxes, image-level labels, or point annotations, could reduce annotation burden while retaining the asymmetry-driven comparison principle. Finally, given AIS’s time-critical nature, extending the framework to predict lesion trajectory, i.e., the rate of penumbra-to-core conversion via collateral circulation information, would connect segmentation output more directly to treatment-window decisions.

## Conclusions

In summary, this study demonstrates that an anatomically-grounded, computationally eficient asymmetry prior can deliver state-of-the-art AIS segmentation, improving Dice by 10.42% over the strongest baseline and reducing HD95 by 19.44% on AISD, while generalizing, without architectural redesign, across the acute-to-chronic stroke imaging continuum with strong volumetric agreement (Pearson r = 0.9854) and a 531× reduction in attention FLOPs relative to global crossattention, alongside reliability characteristics that vary in clinically interpretable ways across that continuum. Future directions include uncertainty-aware diagnosis, training under weak supervision and lesion trajectory prediction.

## References

[1] T. D. Musuka, S. B. Wilton, M. Traboulsi, and M. D. Hill, “Diagnosis and management of acute ischemic stroke: speed is critical,” Cmaj, vol. 187, no. 12, pp. 887–893, 2015.

[2] Q. Ding, S. Liu, Y. Yao, H. Liu, T. Cai, and L. Han, “Global, regional, and national burden of ischemic stroke, 1990–2019,” Neurology, vol. 98, no. 3, pp. e279–e290, 2022.

[3] X. Zhang, H. Lv, X. Chen, M. Li, X. Zhou, and X. Jia, “Analysis of ischemic stroke burden in asia from 1990 to 2019: based on the global burden of disease 2019 data,” Frontiers in Neurology, vol. Volume 14 - 2023, 2023. [Online]. Available: https://www.frontiersin.org/journals/ neurology/articles/10.3389/fneur.2023.1309931

[4] J. J. Heit and M. Wintermark, “Imaging selection for reperfusion therapy in acute ischemic stroke,” Current treatment options in neurology, vol. 17, no. 2, p. 4, 2015.

[5] M. Wintermark, P. C. Sanelli, G. W. Albers, J. Bello, C. Derdeyn, S. W. Hetts, M. H. Johnson, C. Kidwell, M. H. Lev, D. S. Liebeskind et al., “Imaging recommendations for acute stroke and transient ischemic attack patients: a joint statement by the american society of neuroradiology, the american college of radiology, and the society of neurointerventional surgery,” American Journal of Neuroradiology, vol. 34, no. 11, pp. E117–E127, 2013.

[6] E. de la Rosa, R. Su, M. Reyes, E. O. Riedel, H. Baazaoui, R. Wiest, F. Kofler, K. Yang, D. Robben, M. Mojtahedi et al., “Isles’24: Final infarct prediction with multimodal imaging and clinical data. where do we stand?” arXiv preprint arXiv:2408.10966, 2024.

[7] K. D. Kurz, G. Ringstad, A. Odland, R. Advani, E. Farbu, and M. W. Kurz, “Radiological imaging in acute ischaemic stroke,” European Journal of Neurology, vol. 23, no. S1, pp. 8–17, 2016. [Online]. Available: https://onlinelibrary.wiley. com/doi/abs/10.1111/ene.12849

[8] N. A. Bachtiar, B. Murtala, M. Muis, M. I. Ilyas, H. b. Abdul Hamid, S. As’ ad, J. Tammasse, A. D. Wuysang, and G. V. Soraya, “Non-contrast mri sequences for ischemic stroke: a concise overview for clinical radiologists,” Vascular Health and Risk Management, pp. 521–531, 2024.

[9] S.-L. Liew, B. P. Lo, M. R. Donnelly, A. Zavaliangos-Petropulu, J. N. Jeong, G. Barisano, A. Hutton, J. P. Simon, J. M. Juliano, A. Suri et al., “A large, curated, open-source stroke neuroimaging dataset to improve lesion segmentation algorithms,” Scientific data, vol. 9, no. 1, p. 320, 2022.

[10] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in International Conference on Medical image computing and computer-assisted intervention. Springer, 2015, pp. 234–241.

[11] Z. Zhou, M. M. Rahman Siddiquee, N. Tajbakhsh, and J. Liang, “Unet++: A nested u-net architecture for medical image segmentation,” in International workshop on deep learning in medical image analysis. Springer, 2018, pp. 3–11.

[12] O. Oktay, J. Schlemper, L. L. Folgoc, M. Lee, M. Heinrich, K. Misawa, K. Mori, S. McDonagh, N. Y. Hammerla, B. Kainz et al., “Attention u-net: Learning where to look for the pancreas,” arXiv preprint arXiv:1804.03999, 2018.

[13] F. Isensee, P. F. Jaeger, S. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnu-net: a self-configuring method for deep learning-based biomedical image segmentation,” Nature methods, vol. 18, no. 2, pp. 203–211, 2021.

[14] H. Cao, Y. Wang, J. Chen, D. Jiang, X. Zhang, Q. Tian, and M. Wang, “Swin-unet: Unet-like pure transformer for medical image segmentation,” in European conference on computer vision. Springer, 2022, pp. 205–218.

[15] A. Hatamizadeh, V. Nath, Y. Tang, D. Yang, H. R. Roth, and D. Xu, “Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images,” in International MICCAI brainlesion workshop. Springer, 2021, pp. 272–284.

[16] J. Chen, Y. Lu, Q. Yu, X. Luo, E. Adeli, Y. Wang, L. Lu, A. L. Yuille, and Y. Zhou, “Transunet: Transformers make strong encoders for medical image segmentation,” arXiv preprint arXiv:2102.04306, 2021.

[17] J. Ma, F. Li, and B. Wang, “U-mamba: Enhancing long-range dependency for biomedical image segmentation,” arXiv preprint arXiv:2401.04722, 2024.

[18] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljacic, T. Hou, and M. Tegmark, “Kan: Kolmogorov–arnold networks,” in International conference on learning representations, vol. 2025, 2025, pp. 70 367–70 413.

[19] C. Li, X. Liu, W. Li, C. Wang, H. Liu, Y. Liu, Z. Chen, and Y. Yuan, “U-kan makes strong backbone for medical image segmentation and generation,” in Proceedings of the AAAI conference on artificial intelligence, vol. 39, no. 5, 2025, pp. 4652– 4660.

[20] Q. Bao, S. Mi, B. Gang, W. Yang, J. Chen, and Q. Liao, “Mdan: Mirror diference aware network for brain stroke lesion segmentation,” IEEE Journal ofBiomedical and Health Informatics, vol. 26, no. 4, pp. 1628–1639, 2021.

[21] W. Yu, Z. Huang, J. Zhang, and H. Shan, “San-net: Learning generalization to unseen sites for stroke lesion segmentation with self-adaptive normalization,” Computers in Biology and Medicine, vol. 156, p. 106717, 2023.

[22] H. Yang, C. Huang, X. Nie, L. Wang, X. Liu, X. Luo, and L. Liu, “Is-net: automatic ischemic stroke lesion segmentation on ct images,” IEEE Transactions on Radiation and Plasma Medical Sciences, vol. 7, no. 5, pp. 483–493, 2023.

[23] J. Sun, Q. Li, Y. Liu, Y. Liu, G. Coatrieux, J.- L. Coatrieux, Y. Chen, and J. Lu, “Pathological asymmetry-guided progressive learning for acute ischemic stroke infarct segmentation,” IEEE Transactions on Medical Imaging, vol. 43, no. 12, pp. 4146–4160, 2024.

[24] H. Kuang, Y. Wang, J. Liu, J. Wang, Q. Cao, B. Hu, W. Qiu, and J. Wang, “Hybrid cnntransformer network with circular feature interaction for acute ischemic stroke lesion segmentation on non-contrast ct scans,” IEEE Transactions on

Medical Imaging, vol. 43, no. 6, pp. 2303–2316, 2024.

[25] J. Sun, G.-L. Ju, Y.-H. Qu, H.-H. Xie, H.-X. Sun, S.-Y. Han, Y.-F. Li, X.-Q. Jia, and Q. Yang, “Deep learning for segmenting ischemic stroke infarction in non-contrast ct scans by utilizing asymmetry,” Clinical Neuroradiology, vol. 36, no. 1, pp. 129–142, 2026.

[26] L. Li, H. Li, J. Fang, J. Yan, and Y. Yu, “Brain asymmetry-guided network model for infarct lesion segmentation in acute ischemic stroke,” Expert Systems with Applications, p. 130255, 2025.

[27] K. Liang, K. Han, X. Li, X. Cheng, Y. Li, Y. Wang, and Y. Yu, “Symmetry-enhanced attention network for acute ischemic infarct segmentation with noncontrast ct images,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2021, pp. 432–441.

[28] S. M. Smith, “Fast robust automated brain extraction,” Human brain mapping, vol. 17, no. 3, pp. 143–155, 2002.

[29] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[30] E. O. Riedel, E. de la Rosa, T. A. Baran, M. H. Petzsche, H. Baazaoui, K. Yang, F. A. Musio, H. Huang, D. Robben, J. O. Seia et al., “Isles’24–a real-world longitudinal multimodal stroke dataset,” arXiv preprint arXiv:2408.11142, 2024.

[31] P. A. Yushkevich, J. Piven, H. Cody Hazlett, R. Gimpel Smith, S. Ho, J. C. Gee, and G. Gerig, “User-guided 3D active contour segmentation of anatomical structures: Significantly improved eficiency and reliability,” Neuroimage, vol. 31, no. 3, pp. 1116–1128, 2006.

[32] G. Turc, P. Bhogal, U. Fischer, P. Khatri, K. Lobotesis, M. Mazighi, P. D. Schellinger, D. Toni, J. De Vries, P. White et al., “European stroke organisation (eso)-european society for minimally invasive neurological therapy (esmint) guidelines on mechanical thrombectomy in acute ischemic stroke,” Journal of neurointerventional surgery, vol. 15, no. 8, pp. e8–e8, 2023.

[33] H. Zhu, Y. Chen, T. Tang, G. Ma, J. Zhou, J. Zhang, S. Lu, F. Wu, L. Luo, S. Liu, S. Ju, and H. Shi, “Isp-net: Fusing features to predict ischemic stroke infarct core on ct perfusion maps,” Computer Methods and Programs in Biomedicine, vol. 215, p. 106630, 2022.

[Online]. Available: https://www.sciencedirect. com/science/article/pii/S0169260722000153

[34] H. Kuang, X. Tan, J. Wang, Z. Qu, Y. Cai, Q. Chen, B. J. Kim, and W. Qiu, “Segmenting ischemic penumbra and infarct core simultaneously on non-contrast ct of patients with acute ischemic stroke using novel convolutional neural network,” Biomedicines, vol. 12, no. 3, p. 580, 2024.

[35] P. Seners, N. Yuen, M. Mlynash, S. J. Snyder, J. J. Heit, M. G. Lansberg, S. Christensen, J.-F. Albucher, C. Cognard, I. Sibon et al., “Quantification of penumbral volume in association with time from stroke onset in acute ischemic stroke with large vessel occlusion,” JAMA neurology, vol. 80, no. 5, pp. 523–528, 2023.

[36] J. L. Saver, “Time is brain—quantified,” Stroke, vol. 37, no. 1, pp. 263–266, 2006.