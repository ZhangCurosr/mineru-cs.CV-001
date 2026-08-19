# Heterogeneity-Aware Deep Learning for Tumour Classification from Multiparametric MRI

Yue Xia<sup>1</sup>, Euijoon Ahn<sup>1</sup>, Tian Xia<sup>2</sup>, Yuan Yuan<sup>1</sup>, Michael Fulham<sup>2\*</sup>, Jinman Kim<sup>1\*</sup>

<sup>1</sup>School of Computer Science, The University of Sydney, Camperdown, NSW, 2006, Australia.

<sup>2</sup>Department of Molecular Imaging, Royal Prince Alfred Hospital, Camperdown, NSW, 2006, Australia.

## Abstract

Intra-tumoural heterogeneity (ITH) reflects spatially varying biological characteristics within a tumour and is considered a major determinant of tumour behaviour, prognosis, and treatment response. Radiomics and deep learning approaches have shown promise for tumour classification using multiparametric MRI (mp-MRI). However, radiomics methods depend on handcrafted features, which may not fully capture complex tumour characteristics. Most deep learning methods instead rely on whole-tumour representations or manually defined sub-regions, limiting their ability to learn scalable and biologically meaningful representations of tumour heterogeneity. To address these limitations, we propose a Heterogeneity-Aware Deep Learning Classification (HA-DLC) framework that explicitly models imaging-derived tumour sub-regions for tumour classification tasks, including lesion-type diagnosis and molecular-status prediction. HA-DLC comprises two principal components: (1) a Heterogeneous Sub-region Generation (HSG) module that generates initial pseudo-labelled tumour subregions through unsupervised clustering, followed by a Cross-Patient Sub-region Alignment (CPSA) strategy that learns soft assignments from the cluster-derived regions to a shared label space; and (2) a Dual-Stream Feature Extraction (DSFE) module that integrates local heterogeneity-aware representations with global tumour features. Given the initial clustering-derived masks, the CPSA, segmentation, feature-extraction, and classification components are jointly optimized end-to-end using soft-target segmentation and classification objectives, enabling simultaneous learning of sub-regional and tumour-level information. We evaluate HA-DLC on the LLD-MMRI2023 liver lesion dataset and the RSNA-ASNR-MICCAI 2021 Radiogenomic Brain Tumour (BraTS-RC) dataset.

Experimental results demonstrate that HA-DLC consistently outperforms stateof-the-art radiomics and deep learning baselines, highlighting the efectiveness of cross-patient sub-region alignment and dual-stream heterogeneity modelling for tumour classification from mp-MRI.

Keywords: Intra-tumoural heterogeneity, Multiparametric MRI, Tumour classification, Imaging-derived sub-regions, Radiogenomics, Deep learning

## 1 Introduction

Tumours are generally biologically heterogeneous, exhibiting spatial variation in cellu lar, molecular, and microenvironmental characteristics that may manifest as regional diferences in their imaging phenotypes. Intra-tumoural heterogeneity (ITH) refers to these within-tumour diferences, which may arise from variations in cellular composition, genomic alterations, gene expression, vasculature, necrosis, perfusion, and oxygenation, collectively reflecting the underlying tumour biology. ITH is associated with tumour behaviour, prognosis, and therapeutic response and therefore has implications for diagnosis, prognostic stratification, treatment selection, and patient management [1, 2]. Although imaging is central to tumour assessment, definitive histopathological and many molecular analyses rely on cells or tissue obtained through fine-needle aspiration, core-needle biopsy, surgical biopsy, or tumour resection [3]. However, surgical resection is not always feasible or clinically indicated, while fineneedle aspiration and core-needle biopsy sample only limited regions of a tumour. Consequently, these procedures are susceptible to sampling error and may not capture the full spatial extent of tumour heterogeneity [4].

Medical imaging provides a non-invasive, in vivo alternative for evaluating tumour status and assessing ITH. Among current imaging modalities, multiparametric magnetic resonance imaging (mp-MRI) provides complementary contrasts sensitive to regional diferences in cellularity, perfusion, and microstructure. Jain et al. [5] reported that mp-MRI appearance can serve as non-invasive imaging surrogates of ITH and correlate with genomic profiles. Multiple studies have demonstrated that tumours across various cancer types can be partitioned into distinct sub-regions on mp-MRI using computational approaches based on imaging-derived radiomic features and spatial molecular information, or through manual delineation by clinicians [6–8]. These sub-regions have been shown to correlate with underlying molecular characteristics. However, manual delineation is subjective, labour-intensive, and dificult to scale. Computational approaches such as unsupervised clustering ofer an alternative by grouping tumour voxels with similar imaging characteristics into distinct withintumour sub-regions. However, the resulting cluster labels are inherently arbitrary and may not correspond across patients, limiting their utility for learning consistent cross-patient representations of tumour heterogeneity.

Recent advances in deep learning ofer a promising approach for tumour characterisation by learning imaging features associated with tumour morphology, molecular status, or clinical outcomes. Deep learning methods have achieved strong performance across a range of MRI clinical decision-support tasks, including brain tumour segmentation and prostate cancer detection and localisation [9, 10]. Zong et al., for example, developed a deep learning model for detecting and localising malignant prostate cancer using mp-MRI [11]. Pasquini et al. employed a CNN model to predict IDH mutation status in patients with GBM [12]. Cheng et al. developed a fully automated multi-task network that jointly performed glioma segmentation and IDH-genotype prediction from multimodal MRI [13]. Although this framework combined spatial delineation with molecular prediction, it learned supervised, predefined glioma regions rather than discovering and aligning imaging-derived heterogeneity phenotypes. Despite these advances, three challenges remain. First, many deep learning classification frameworks operate on whole-tumour regions, tumour-enclosing bounding boxes, or other coarse predefined regions, which may obscure spatially distinct imaging characteristics and therefore underrepresent ITH. Second, although tumour sub-regions may provide more informative heterogeneity-aware representations, existing sub-region based approaches often depend on manual expert delineation, making them subjective, labour-intensive, and dificult to scale. Third, although unsupervised clustering can automatically generate tumour sub-regions, when each tumour is partitioned independently before the resulting regions are re-clustered at the population level, the population-level step can group or relabel patient-specific regions but cannot alter their voxel membership or spatial boundaries established by the initial patient-level clustering. This may limit the learning of consistent cross-patient heterogeneity representations.

In this study, we propose a Heterogeneity-Aware Deep Learning Classification (HA-DLC) framework that combines unsupervised sub-region generation with endto-end heterogeneity-aware learning for tumour classification from mp-MRI. HA-DLC addresses the limitations of existing whole-tumour and manually defined sub-region approaches by automatically generating tumour sub-regions and learning crosspatient-consistent heterogeneity representations. Figure 1 illustrates an overview of our proposed framework. The main contributions are as follows:

• Heterogeneous Sub-region Generation (HSG) Module: We propose an unsupervised sub-region generation strategy based on clustering algorithms that operate within the tumour region of interest (ROI), producing within-tumour pseudo-labels for use in subsequent sub-region segmentation.

• Cross-Patient Sub-region Alignment (CPSA): To enable unsupervised subregions usable for supervised heterogeneity-aware classification, we introduce a soft, learnable alignment mechanism that estimates a probability distribution mapping each patient-specific cluster to a set of shared sub-region classes. These soft correspondences convert the fixed patient-level cluster masks into aligned soft targets for supervising the sub-region segmentation networks. This reduces the efect of patient-specific cluster-label permutations and enables the classifier to learn cross-patient-comparable ITH patterns from imaging-derived sub-regions without requiring manual compartment annotations.

• Dual-Stream Feature Extraction (DSFE) Module: We propose a dual-stream architecture comprising a Heterogeneity Feature Extraction (HFE) stream, which learns sequence-specific features from the generated sub-regions, and a Global Feature Extraction (GFE) stream, which captures tumour-level volumetric context. By integrating local heterogeneity information with global contextual features, DSFE produces a comprehensive representation for tumour classification.

![](images/f395bf3311f0dfe6d6d26bd2f4695b61fb13ec29148e8cd1e41b20c46e74c8d3.jpg)  
Fig. 1 Overview of Heterogeneity-Aware Deep Learning Classification (HA-DLC) framework. The framework comprises a Heterogeneous Sub-region Generation (HSG) module, Cross-Patient Sub-region Alignment (CPSA) module, and a Dual-Stream Feature Extraction (DSFE) module for heterogeneity-aware tumour classification. n denotes the number of input MRI sequences in mp-MRI. The segmentation loss $( L _ { s e g } )$ is computed between the aligned sub-region pseudo-labels and the predicted sub-region probabilities, while the classification loss $\left( \boldsymbol { L _ { c l s } } \right)$ is computed between the ground-truth class labels and the predicted class labels.

We evaluated HA-DLC on seven-class liver lesion diagnosis using LLD-MMRI2023 [14] and MGMT promoter methylation prediction using BraTS-RC [15]. HA-DLC achieved the highest mean performance among the evaluated comparison methods. Ablation experiments assessed sub-region supervision, cross-patient soft relabelling, and global tumour context. The HSG comparisons used the same outer folds without nested configuration selection and should be interpreted as post-hoc sensitivity analyses.

## 2 Method

## 2.1 Overview

We define intra-tumoural feature heterogeneity as spatial variation in voxel-wise imaging features within the tumour ROI. For each of N mp-MRI inputs (sequences or contrast phases), $V ^ { ( n ) } ~ \in ~ \mathbb { R } ^ { D \times H \times W }$ , patient-level clustering produces a subregion map $\bar { S } ^ { ( n ) } \in \{ 0 , 1 , \ldots , K \} ^ { D \times H \times W }$ , where 0 denotes background and $1 , \ldots , K$ denote imaging-derived tumour clusters. CPSA maps each patient-specific cluster to a probability distribution over M shared sub-region classes, producing a soft pseudo-label tensor $\tilde { \mathbf { S } } ^ { ( n ) } \in \mathsf { [ 0 , 1 ] } ^ { ( M + 1 ) \times D \times H \times W }$ . This alignment changes the crosspatient correspondence of each cluster while preserving its voxel membership and spatial boundaries. The soft-aligned pseudo-labels supervise input-specific segmentation networks, whose encoder features are fused with global tumour features for classification.

As shown in Figure 1, the HFE stream within DSFE uses input-specific 3D segmentation networks to predict imaging-derived tumour sub-regions and extract local heterogeneity features from each mp-MRI input. The sub-region prediction heads are supervised by soft-aligned pseudo-label distributions generated by CPSA from the HSG clustering maps. Specifically, Lseg is the voxel-wise soft cross-entropy between the predicted sub-region distributions and the soft-aligned targets, averaged across all mp-MRI inputs. The tumour-level classification branch is supervised by Lcls, defined as the cross-entropy loss between the predicted class distribution and the ground-truth class label. HA-DLC is optimised using the joint objective Lseg +Lcls. Together, these losses encourage the learned features to encode aligned sub-regional heterogeneity while remaining discriminative for tumour-level classification.

## 2.2 Unsupervised Sub-region Label Generation

HSG generates initial patient-specific sub-region pseudo-labels within the tumour ROI using ofline unsupervised clustering performed independently for each mp-MRI input. Two methods are implemented: (i) intensity-based k-means clustering [16], for which the number of clusters k determines the sub-region granularity; and (ii) Diferentiable Feature Clustering (DFC) [17], which learns voxel features with spatial regularisation controlled by the continuity-loss coeficient $\mu .$ The resulting patient-specific label maps are subsequently aligned by CPSA.

## 2.3 Cross-Patient Sub-region Alignment Module

The Cross-Patient Sub-region Alignment (CPSA) module softly aligns patient-specific cluster labels by estimating their correspondence to a shared set of sub-region classes. One alignment network is used for each of the N mp-MRI inputs. Each network receives (i) the corresponding image volume, (ii) its raw sub-region label map generated by k-means or DFC, and (iii) the shared multi-scale global features from the GFE stream. Two modified DenseNet encoders [18], each comprising three dense blocks, separately encode the image volume and raw label map. The resulting image and labelmap feature vectors and the globally pooled GFE feature vector are each projected to 64 dimensions and concatenated into a 192-dimensional representation. This representation is passed to K cluster-specific fully connected (FC) heads, each producing M assignment logits over the shared aligned classes. Thus, for patient b and input n, the alignment network produces $\mathbf { Z } _ { b , n } \in \mathbb { R } ^ { K \times M }$ , and the complete logit tensor has shape $B \times N \times K \times M$ . A softmax over the M aligned classes converts each row of $\mathbf { Z } _ { b , n }$ into a soft cluster-to-class probability distribution.

The soft cluster-to-class assignment probabilities are obtained as

$$
A _ { b , n , j , k } = \frac { \exp ( Z _ { b , n , j , k } ) } { \displaystyle \sum _ { k ^ { \prime } = 1 } ^ { M } \exp ( Z _ { b , n , j , k ^ { \prime } } ) } , \qquad j \in \{ 1 , \ldots , K \} , \quad k \in \{ 1 , \ldots , M \} .\tag{1}
$$

![](images/7ffed33be56c53df5ee06fde5f18b69f464d94718e3c02cf83f72aa590bb23f2.jpg)  
Fig. 2 Heterogeneity feature extraction stream architecture. The stream follows a 3D residual U-Net structure with encoder and decoder branches constructed from residual blocks. Local heterogeneity features are extracted from the outputs of the encoder residual blocks. The residual block denoted by m contains convolutional layers with m output channels. The notation a@b, m denotes a kernel size of a × a × $^ { a , }$ a stride of $b \times b \times b ,$ and m output channels.

$A _ { b , n , j , k }$ represents the probability of mapping patient-specific cluster j to aligned class $k .$ The background label remains fixed and is excluded from the soft assignment. For a tumour voxel v with original cluster label $j ,$ the soft-aligned target is defined as

$$
\tilde { S } _ { b , n , k , v } = A _ { b , n , j , k } , \qquad j = S _ { b , n , v } \in \{ 1 , \ldots , K \} , \quad k \in \{ 1 , \ldots , M \} .\tag{2}
$$

Thus, all voxels belonging to the same initial cluster receive the same aligned-class probability distribution, preserving their voxel membership and spatial boundaries. For visualisation only, a discrete aligned sub-region map can be obtained as

$$
\bar { S } _ { b , n , v } = \mathop { \mathrm { a r g m a x } } _ { k \in \{ 0 , . . . , M \} } \tilde { S } _ { b , n , k , v } .\tag{3}
$$

## 2.4 Dual-Stream Feature Extraction Module

The Dual-Stream Feature Extraction (DSFE) module extracts complementary local and global representations for heterogeneity-aware tumour classification. It comprises two streams: (i) a Heterogeneity Feature Extraction (HFE) stream, which uses an input-specific 3D segmentation network for each MRI sequence or contrast phase to predict tumour sub-regions and learn local features under soft-aligned segmentation supervision; and (ii) a Global Feature Extraction (GFE) stream, which uses a global feature backbone to capture tumour-level volumetric context from the complete mp-MRI input. The resulting local and global representations are fused to perform tumourlevel classification.

## 2.4.1 Heterogeneity Feature Extraction Stream

The HFE stream consists of input-specific segmentation networks, each based on a customised 3D residual U-Net architecture [19]. Each network processes one mp-MRI input (sequence or contrast phase), predicts imaging-derived tumour sub-regions, and extracts local heterogeneity features from the encoder. The sub-region prediction heads are supervised using the soft-aligned pseudo-labels generated by CPSA.

As shown in Figure 2, each segmentation network follows an encoder–decoder structure with skip connections. The encoder contains five cascaded residual blocks that progressively extract multi-scale local features from the input volume. Between encoder blocks, max-pooling is used to reduce spatial resolution and increase the receptive field. The decoder mirrors the encoder structure by progressively upsampling the feature maps and combining them with corresponding encoder features through skip connections. These skip connections help preserve voxel-level spatial detail for sub-region prediction.

Each residual block contains a projection branch and a main convolutional branch. The projection branch adjusts the number of feature channels so that it can be added to the output of the main branch. The main branch applies two successive convolutional operations, each followed by batch normalisation and ReLU activation. The output of the first convolutional operation is added to the projected input, and the resulting intermediate feature is added again after the second convolutional operation.

The final sub-region logits are produced by ${ \mathrm { ~ a ~ } } 1 \times 1 \times 1$ convolution and converted into probabilities using softmax normalisation, where the output channels correspond to the background and shared aligned sub-region classes. In addition to segmentation, local heterogeneity features are extracted from the encoder at multiple scales. Specifically, the output of each encoder residual block is compressed using a $1 \times 1 \times 1$ convolution, followed by batch normalisation, ReLU activation, and global average pooling. The resulting multi-scale features are concatenated to form an input-specific local heterogeneity representation.

## 2.4.2 Global Feature Extraction Stream

The GFE stream captures whole-tumour contextual information complementary to the local sub-region-aware features. We use UniFormer [20] as the global backbone because its hybrid convolution–attention architecture captures both local image patterns and global volumetric context. In our implementation, a UniFormer-Small-Plus configuration is used to extract the global feature representation from the mp-MRI input.

Global features are passed through fully connected layers to form a compact embedding. This embedding is concatenated with the sub-region-aware local features from all sequence-specific segmentation networks. A final classification layer uses the combined feature vector to generate tumour-level predictions. The DSFE module therefore integrates sub-region-aware local features with global tumour context.

## 2.5 Loss

HA-DLC is optimised using equally weighted classification and segmentation losses. HSG first generates patient- and input-specific cluster maps through ofline clustering, and CPSA converts them into voxel-wise soft-aligned targets for supervising the inputspecific segmentation networks. The total objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { c l s } } + \mathcal { L } _ { \mathrm { s e g } } .\tag{4}
$$

The classification and segmentation losses are defined as

$$
\mathcal { L } _ { \mathrm { c l s } } = \mathrm { C E } ( \mathbf { y } , \hat { \mathbf { y } } ) ,\tag{5}
$$

$$
\mathcal { L } _ { \mathrm { s e g } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathrm { C E } \Big ( \tilde { \mathbf { S } } ^ { ( n ) } , \hat { \mathbf { S } } ^ { ( n ) } \Big ) ,\tag{6}
$$

where CE denotes mean cross-entropy, y and $\hat { \mathbf { y } }$ are the ground-truth and predicted tumour-class distributions, and $\tilde { \mathbf { S } } ^ { ( n ) }$ and $\hat { \mathbf { S } } ^ { ( n ) }$ are the soft-aligned target and predicted sub-region distribution for mp-MRI input $n .$ The segmentation loss is averaged over the batch and all voxels, including background. Because the soft targets remain diferentiable functions of the CPSA assignments, $\mathcal { L } _ { \mathrm { s e g } }$ jointly trains CPSA and the segmentation networks and also propagates into GFE. Meanwhile, ${ \mathcal L } _ { \mathrm { c l s } }$ supervises the local and global feature streams and the classification head. Thus, following ofline clustering, all trainable components are jointly optimised.

## 3 Experiment

## 3.1 Dataset

The LLD-MMRI2023 dataset (LLD-M) comprises 498 patients with eight multiphase MRI sequences: T2W, DWI, in-phase, out-of-phase, pre-contrast, arterial, portal venous, and delayed phases [14]. The task is to classify seven lesion types, including four benign lesions (haemangioma, abscess, cyst, and focal nodular hyperplasia) and three malignant lesions (cholangiocarcinoma, metastases, and hepatocellular carcinoma). Whole-tumour masks provided with the dataset were generated using MedSAM within a human-in-the-loop annotation pipeline [21].

The BraTS2021 Radiogenomic Classification dataset (BraTS-RC) comprises 585 cases, with four MRI sequences: T1-weighted post-contrast (T1Gd), T1-weighted precontrast (T1w), T2-weighted (T2), and fluid-attenuated inversion recovery (FLAIR). To obtain tumour segmentation masks, patient identifiers were matched with the RSNA-ASNR-MICCAI Brain Tumor Segmentation Challenge 2021 dataset [15]. These masks include three major tumour sub-region labels (enhancing tumour, necrotic/nonenhancing tumour core, and peritumoural oedema), whose union defines the whole tumour, and served as ground-truth labels for the expert-defined sub-region ablation. Cases without matching segmentation masks were excluded, resulting in 577 cases for analysis.

## 3.2 Comparison methods

We compared HA-DLC against several state-of-the-art deep learning and radiomicsbased classification methods. For the BraTS-RC dataset, comparison methods included the first-place solution from the BraTS-RC challenge [22], the Vision Graph Neural Network (ViG) framework for MGMT prediction [23], and UniFormer [20]. In addition, three widely used classification architectures, ResNet-50 [24], DenseNet-121 [18], and EficientNet-B7 [25], were evaluated as strong 3D baselines. The first-place BraTS-RC solution employed a 3D ResNet-10 backbone and constructed a 41-slice

FLAIR volume centred on the slice containing the largest brain cross-sectional area. For LLD-M, the slice index was determined from the T2W tumour mask. The ViG framework originally used fused FLAIR, T1w, and T2 pseudo-RGB images as input to a PyramidViG-S architecture. In our implementation, its input layer was adapted to incorporate all available MRI sequences: four for BraTS-RC and eight for LLD-M. For UniFormer, the UniFormer-Small-Plus architecture was implemented using the same input images as HA-DLC. As a radiomics baseline, 107 handcrafted features were extracted per ROI and MRI sequence using PyRadiomics (v3.0.1) [26]. These included shape descriptors, first-order intensity statistics, and texture features derived from the gray-level co-occurrence matrix (GLCM), gray-level size zone matrix (GLSZM), gray-level run length matrix (GLRLM), neighbouring gray tone diference matrix (NGTDM), and gray-level dependence matrix (GLDM). Features were extracted from either the whole-tumour region or each generated tumour sub-region across MRI sequences and subsequently used to train an XGBoost classifier [27].

## 3.3 Experimental settings

The provided whole-tumour masks were used to crop each tumour ROI prior to sub-region generation. HSG clustering was performed independently on each MRI sequence. For k-means clustering, the number of clusters was varied from 3 to 9. For Deep Feature Clustering (DFC), continuity-loss coeficients of 2, 0.5, and 0.001 were evaluated to control spatial continuity and the resulting sub-region granularity. A fixed random seed was used, and the best-performing configuration for each dataset was identified from the mean performance across the five folds.

We conducted two sets of ablation experiments under the same 5-fold crossvalidation protocol. The first set evaluated the efect of heterogeneity supervision by comparing diferent segmentation targets:

1. No segmentation loss: the segmentation branch and its local encoder features were retained, but the segmentation loss $L _ { s e g }$ was removed from the training objective;

2. Whole-tumour segmentation: the whole-tumour mask was used as the segmentation target without sub-region partitioning; and

3. Ground-truth sub-region segmentation (BraTS-RC only): expert-defined anatomical tumour sub-regions (enhancing tumour, necrotic/non-enhancing tumour core, and peritumoural oedema) were used as the segmentation target.

These experiments investigated whether heterogeneity-aware supervision provides additional predictive value beyond no segmentation supervision and conventional whole-tumour representations. The second set assessed the contribution of individual HA-DLC components by removing one component at a time while keeping all other settings fixed:

1. No alignment: the CPSA module was disabled and the raw clustering labels were used directly as segmentation targets, evaluating the contribution of cross-patient sub-region alignment; and

2. Local HFE only: only the HFE stream was used for classification, with the GFE stream removed, evaluating the contribution of global tumour context.

These experiments were performed on both datasets to evaluate the individual contributions of CPSA and the global GFE stream within DSFE to heterogeneityaware tumour classification.

## 3.4 Evaluation metrics

For the LLD-MMRI2023 (LLD-M) dataset, the primary evaluation metric was the average of the macro-averaged F1-score and Cohen’s kappa, following the challenge protocol [14]. This metric was chosen to account for class imbalance by jointly assessing class-wise classification performance and agreement beyond chance. For the BraTS-RC dataset, the primary evaluation metric was the area under the receiver operating characteristic curve (AUC) [15]. Accuracy was also reported as a supplementary metric to facilitate comparison across methods.

## 3.5 Implementation details

HA-DLC and all deep learning comparison methods were implemented in $\mathrm { P y }$ Torch [28]. The AdamW optimiser was used for training, with a batch size of 4 for the LLD-M dataset and 2 for the BraTS-RC dataset. To account for diferences in tumour extent along the through-plane direction, all tumour ROIs were resized to $1 2 8 \times 1 2 8 \times 1 8$ for LLD-M and $1 2 8 \times 1 2 8 \times 3 6$ for BraTS-RC. Data augmentation was applied during training, including random cropping to $1 1 2 \times 1 1 2 \times 1 6$ and $1 1 2 \times 1 1 2 \times 3 2$ for LLD-M and BraTS-RC, respectively, random flipping along the three spatial axes, and random in-plane rotations of up to $1 0 ^ { \circ }$ . The initial learning rate was set to $1 \times 1 0 ^ { - 4 }$ and was updated using the cosine learning-rate scheduler implemented in $\mathrm { P y }$ Torch Image Models (timm) [29], with five warm-up epochs and a minimum learning rate of $1 \times 1 0 ^ { - 5 }$ . All deep learning experiments were conducted on a single NVIDIA RTX 4090 GPU with 24 GB of memory. Models were trained and evaluated using 5-fold cross-validation, and the reported performance corresponds to the average across the five folds.

## 4 Results

## 4.1 Classification Performance

Table 1 presents the classification results comparing HA-DLC with state-of-the-art methods. For the LLD-M dataset, HA-DLC achieved an accuracy of 0.841, which was 0.032–0.172 higher than those of the comparison methods. In terms of the F1–kappa average, our model reached 0.819, exceeding the comparison methods by 0.041–0.209. For the BraTS-RC dataset, HA-DLC obtained an accuracy of 0.660 and an AUC of 0.686, outperforming the comparison methods by diferences of 0.036–0.068 and 0.019–0.080, respectively.

Table 1 Classification performance comparison with the evaluated methods on the LLD-MMRI2023 (LLD-M) and BraTS-RC datasets. Best results are in bold and second-best are underlined.
<table><tr><td rowspan="2">Method</td><td colspan="2">LLD-MMRI2023 (LLD-M)</td><td colspan="2">BraTS-RC</td></tr><tr><td>Accuracy</td><td>F1-Kappa average</td><td>Accuracy</td><td>AUC</td></tr><tr><td>Kaggle Leaderboard</td><td> $0 . 6 7 7 \pm 0 . 0 3 6$ </td><td> $0 . 6 2 9 \pm 0 . 0 4 8$ </td><td> $\underline { { 0 . 6 2 4 } } \pm 0 . 0 3 0$ </td><td> $0 . 6 4 0 \pm 0 . 0 3 6$ </td></tr><tr><td>ViG-S</td><td> $0 . 7 1 9 \pm 0 . 0 2 0$ </td><td> $0 . 6 8 1 \pm 0 . 0 2 1$ </td><td> $0 . 6 0 3 \pm 0 . 0 2 3$ </td><td> $0 . 6 4 7 \pm 0 . 0 2 3$ </td></tr><tr><td>ResNet-50</td><td> $0 . 7 6 1 \pm 0 . 0 2 1$ </td><td> $0 . 7 2 5 \pm 0 . 0 2 0$ </td><td> $0 . 6 2 2 \pm 0 . 0 3 1$ </td><td> $0 . 6 6 6 \pm 0 . 0 1 2$ </td></tr><tr><td>EfficientNet-B7</td><td> $0 . 6 6 9 \pm 0 . 0 5 4$ </td><td> $0 . 6 1 0 \pm 0 . 0 7 6$ </td><td> $0 . 5 9 5 \pm 0 . 0 6 3$ </td><td> $0 . 6 5 7 \pm 0 . 0 2 1$ </td></tr><tr><td>DenseNet-121</td><td> $0 . 7 3 5 \pm 0 . 0 5 7$ </td><td> $0 . 7 0 0 \pm 0 . 0 6 1$ </td><td> $0 . 6 0 3 \pm 0 . 0 2 3$ </td><td> $\underline { { 0 . 6 6 7 } } \pm 0 . 0 2 8$ </td></tr><tr><td>UniFormer-Small-Plus</td><td> $\underline { { 0 . 8 0 9 } } \pm 0 . 0 2 0$ </td><td> $\underline { { 0 . 7 7 8 } } \pm 0 . 0 2 9$ </td><td> $0 . 5 9 2 \pm 0 . 0 1 9$ </td><td> $0 . 6 0 6 \pm 0 . 0 1 5$ </td></tr><tr><td>HA-DLC (Proposed)</td><td> $\mathbf { 0 . 8 4 1 \pm 0 . 0 3 3 }$ </td><td> $\mathbf { 0 . 8 1 9 \pm 0 . 0 3 3 }$ </td><td> $\mathbf { 0 . 6 6 0 \pm 0 . 0 2 4 }$ </td><td> ${ \bf 0 . 6 8 6 \pm 0 . 0 3 6 }$ </td></tr></table>

Table 2 reports the sensitivity of HA-DLC to the HSG clustering configuration. For LLD-M, the F1–kappa average ranged from 0.791 to 0.819 across the tested configurations. K-means with k = 5 achieved the highest value of 0.819, although several alternatives yielded numerically similar values. On BraTS-RC, the AUC ranged from 0.663 to 0.686, with DFC at $\mu = 0 . 5$ achieving the highest value.

Table 2 Sensitivity of HA-DLC to HSG clustering configuration on LLD-M and BraTS-RC. All configurations were evaluated using the same outer five-fold splits without nested inner selection. Values are cross-validation validation means and population standard deviations across folds. Best result per dataset is in bold.
<table><tr><td>Clustering</td><td>Parameter</td><td>LLD-M (F1-Kappa average) BraTS-RC (AUC)</td><td></td></tr><tr><td rowspan="7">k-means</td><td>k = 3</td><td> $0 . 8 1 4 \pm 0 . 0 1 2$ </td><td> $0 . 6 7 8 \pm 0 . 0 4 8$ </td></tr><tr><td> $k = 4$ </td><td> $0 . 8 0 5 \pm 0 . 0 4 2$ </td><td> $0 . 6 6 7 \pm 0 . 0 2 1$ </td></tr><tr><td> $k = 5$ </td><td> $\mathbf { 0 . 8 1 9 \pm 0 . 0 3 3 }$ </td><td> $0 . 6 6 9 \pm 0 . 0 3 0$ </td></tr><tr><td> $k = 6$ </td><td> $0 . 7 9 9 \pm 0 . 0 3 6$ </td><td> $0 . 6 6 7 \pm 0 . 0 2 9$ </td></tr><tr><td> $k = 7$ </td><td> $0 . 8 0 9 \pm 0 . 0 4 9$ </td><td> $0 . 6 7 0 \pm 0 . 0 2 4$ </td></tr><tr><td> $k = 8$ </td><td> $0 . 7 9 1 \pm 0 . 0 2 7$ </td><td> $0 . 6 6 8 \pm 0 . 0 3 8$ </td></tr><tr><td> $k = 9$ </td><td> $0 . 7 9 3 \pm 0 . 0 4 5$ </td><td> $0 . 6 6 9 \pm 0 . 0 3 2$ </td></tr><tr><td rowspan="3">DFC</td><td> $\mu = 0 . 0 0 1$ </td><td> $0 . 8 1 6 \pm 0 . 0 2 8$ </td><td> $0 . 6 6 3 \pm 0 . 0 3 0$ </td></tr><tr><td> $\mu = 0 . 5$ </td><td> $0 . 8 0 0 \pm 0 . 0 3 1$ </td><td> ${ \bf 0 . 6 8 6 \pm 0 . 0 3 6 }$ </td></tr><tr><td> $\mu = 2$ </td><td> $0 . 8 1 4 \pm 0 . 0 3 6$ </td><td> $0 . 6 7 4 \pm 0 . 0 3 7$ </td></tr></table>

Table 3 summarises the impact of diferent sub-region definitions on radiomicsbased classification performance for the LLD-M and BraTS-RC datasets. Three region definitions were evaluated: (i) whole-tumour masks, (ii) discrete sub-region maps predicted by the trained HA-DLC model, and (iii) expert-defined anatomical tumour sub-regions (BraTS-RC only). On the LLD-M dataset, radiomics features extracted from HA-DLC-predicted sub-regions achieved an F1–kappa average of $0 . 6 1 6 \pm 0 . 0 7 1$ numerically outperforming the whole-tumour approach (0.565 ± 0.036). Similarly, for

MGMT promoter methylation prediction, radiomics features derived from HA-DLCpredicted sub-regions achieved an AUC of $0 . 5 7 6 \pm 0 . 0 4 9$ , numerically exceeding both the whole-tumour baseline $( 0 . 5 3 9 \pm 0 . 0 3 8 )$ and the expert-defined anatomical tumour sub-regions $( 0 . 5 5 8 \pm 0 . 0 3 7 )$

Table 3 XGBoost classification results using radiomics features from diferent tumour sub-regions.
<table><tr><td>Sub-region used</td><td>LLD-M (F1–Kappa average)</td><td>BraTS-RC (AUC)</td></tr><tr><td>Whole tumour</td><td> $0 . 5 6 5 \pm 0 . 0 3 6$ </td><td> $0 . 5 3 9 \pm 0 . 0 3 8$ </td></tr><tr><td>Generated</td><td> $\mathbf { 0 . 6 1 6 \pm 0 . 0 7 1 }$ </td><td> $\mathbf { 0 . 5 7 6 \pm 0 . 0 4 9 }$ </td></tr><tr><td>Expert-defined</td><td>NA</td><td> $0 . 5 5 8 \pm 0 . 0 3 7$ </td></tr></table>

## 4.2 Ablation studies

Table 4 presents the ablation study results on the LLD-M and BraTS-RC datasets. The first set of experiments evaluated the efect of diferent segmentation targets and the removal of segmentation supervision on classification performance. Removing the segmentation loss reduced the F1–kappa average to 0.768 on LLD-M and the AUC to 0.623 on BraTS-RC. Using the whole-tumour mask as the segmentation target improved the corresponding scores to 0.802 and 0.650, respectively. Replacing whole-tumour supervision with soft, aligned imaging-derived sub-region supervision further improved performance to 0.819 on LLD-M and 0.686 on BraTS-RC. On BraTS-RC, using expert-defined anatomical tumour sub-regions as the segmentation target achieved an AUC of 0.681, which was slightly lower than that obtained using the proposed imaging-derived sub-regions.

The second set of experiments evaluated the contribution of individual HA-DLC components. Disabling the CPSA module and directly using the raw patient-specific clustering labels as segmentation targets reduced the F1–kappa average to 0.784 on LLD-M and the AUC to 0.618 on BraTS-RC, supporting the contribution of crosspatient sub-region alignment. Restricting the classifier to the HFE stream alone resulted in an F1–kappa average of 0.769 on LLD-M and an AUC of 0.669 on BraTS-RC, indicating that incorporating global tumour context through the GFE stream improved the mean classification performance.

## 4.3 Qualitative and Quantitative Analysis of Predicted Sub-regions

Figure 3 illustrates the sub-region maps predicted by HA-DLC on T2-weighted images for four representative cases of hepatic metastases and four cases of hepatocellular carcinoma (HCC). The colour-coded maps suggest that the aligned sub-regions capture recurring heterogeneity patterns across patients. In metastatic lesions, HA-DLC identified distinct sub-region structures that often corresponded to intensity zones visible on the T2-weighted images, indicating heterogeneous imaging characteristics within the lesions. In contrast, HCC lesions exhibited a diferent sub-region organisation, with one sub-region occupying much of the tumour interior and another concentrated near the lesion boundary. These observations suggest that the proposed framework may identify lesion-type-associated spatial imaging patterns. Importantly, the generated sub-regions were derived entirely from imaging data without histopathological supervision; therefore, the biological significance of individual clusters requires further validation.

Table 4 Ablation study on LLD-M and BraTS-RC (5-fold cross-validation). The upper block varies the segmentation target; the lower block removes one component of HA-DLC at a time. ∆ is relative to the full HA-DLC. Best result per dataset in bold.
<table><tr><td rowspan="2">Variant</td><td colspan="2">LLD-M (F1–Kappa average)</td><td colspan="2">BraTS-RC (AUC)</td></tr><tr><td>Score</td><td>∆</td><td>Score</td><td>∆</td></tr><tr><td>HA-DLC (full)</td><td> $\mathbf { 0 . 8 1 9 \pm 0 . 0 3 3 }$ </td><td></td><td> ${ \bf 0 . 6 8 6 \pm 0 . 0 3 6 }$ </td><td></td></tr><tr><td>Segmentation target</td><td></td><td></td><td></td><td></td></tr><tr><td>w/o segmentation loss</td><td> $0 . 7 6 8 \pm 0 . 0 2 9$ </td><td>-0.051</td><td> $0 . 6 2 3 \pm 0 . 0 5 4$ </td><td>-0.063</td></tr><tr><td>Whole-tumour segmentation</td><td> $0 . 8 0 2 \pm 0 . 0 3 1$ </td><td>-0.017</td><td> $0 . 6 5 0 \pm 0 . 0 2 1$ </td><td>-0.036</td></tr><tr><td>Expert-defined sub-regions</td><td>NA</td><td></td><td> $0 . 6 8 1 \pm 0 . 0 2 2$ </td><td>-0.005</td></tr><tr><td>Component removed</td><td></td><td></td><td></td><td></td></tr><tr><td>w/o alignment</td><td> $0 . 7 8 4 \pm 0 . 0 4 7$ </td><td>-0.035</td><td> $0 . 6 1 8 \pm 0 . 0 1 6$ </td><td>-0.068</td></tr><tr><td>Local HFE only</td><td> $0 . 7 6 9 \pm 0 . 0 4 5$ </td><td>-0.050</td><td> $0 . 6 6 9 \pm 0 . 0 3 6$ </td><td>-0.017</td></tr></table>

Hepatic Metastasis Predicted Sub-region Masks for T2WI  
![](images/357ee277babff1c1f476e2cc6420a3d7ad8fdff1a804cfe727bac10f3aff2b20.jpg)  
Fig. 3 Sub-region segmentation of liver tumour on T2-weighted MRI. The top two rows show four representative hepatic metastases, with predicted sub-region maps in the first row and the corresponding T2-weighted images in the second row. The bottom two rows show four representative hepatocellular carcinomas (HCC), with predicted sub-region maps in the third row and the corresponding T2-weighted images in the fourth row. The predicted sub-regions are overlaid in semi-transparent colours, illustrating lesion-specific intra-tumoural heterogeneity patterns identified by HA-DLC.

(a)  
![](images/bfe43bbc22cdfef19a548b9bf52c73f3a0d0b7879ef158595dd1a5ee52ed54a8.jpg)  
MGMT Methylated

![](images/1966b1e44480cdc7fce7a0638b9a5fa91a3c69582dd1fdacee158b8a55e5da1c.jpg)  
(b)

![](images/7e14b42cce8592b831530edacb22eabc2c657b920f449f9cff7524728aafe171.jpg)

![](images/f1ddb92e5ebe063f1e30faa9e8361bb5db8d200c9d61948dbecf6866c2c852c7.jpg)

![](images/ac157ac66d75a96637532a3dc28c9e702354f19616ab27188b5504516153719b.jpg)  
Fig. 4 Sub-region segmentation and quantitative comparison by MGMT promoter methylation status. (a) Representative T1-weighted MRI slices of glioblastoma lesions from MGMT-methylated tumours (top row) and MGMT-unmethylated tumours (bottom row). Sub-region 1 is shown in blue and Sub-region 2 in brown. (b) Boxplots of normalised sub-region volume ratios (sub-region volume/total tumour volume), stratified by methylation status (blue boxes: unmethylated; red boxes: methylated).

The learned sub-regions were also associated with molecular status in glioblastoma. As shown in Figure 4, the heterogeneity-aware framework delineated two dominant complementary regions on T1-weighted images. The normalised volume fraction of Sub-region 1 was significantly higher in MGMT-methylated tumours than in unmethylated tumours (Welch’s two-sample t-test, $n = 5 7 8 , p = 0 . 0 0 3 6 )$ , whereas Sub-region 2 was significantly lower in the methylated cohort $( p \ : = \ : 0 . 0 0 2 4 )$ . These quantitative diferences were consistent with the qualitative examples shown in Figure 4, which illustrate diferent spatial distributions of the two sub-regions according to MGMT methylation status. Together, these findings suggest that the imaging-derived subregions identified by HA-DLC may capture tumour characteristics associated with underlying molecular phenotypes.

## 5 Discussion

Our main findings are as follows: (i) HA-DLC consistently outperformed state-ofthe-art radiomics and deep learning methods on both the LLD-M and BraTS-RC datasets, supporting the efectiveness of explicitly modelling tumour heterogeneity for both lesion classification and molecular-status prediction; (ii) performance was relatively robust to the choice of HSG clustering strategy, although the optimal clustering configuration difered between datasets, suggesting that the framework can accommodate diferent patterns of tumour heterogeneity; (iii) imaging-derived sub-region segmentation yielded superior classification performance compared with whole-tumour segmentation and, on BraTS-RC, achieved performance comparable to expert-defined tumour compartments, highlighting the value of data-driven heterogeneity representations; and (iv) the ablation studies demonstrated that the segmentation objective, CPSA module, and GFE stream within the DSFE architecture each contributed to the overall performance gains. Collectively, these findings suggest that learning cross-patient-consistent sub-regional representations of tumour heterogeneity provides complementary information beyond conventional whole-tumour analysis and can improve tumour classification from mp-MRI.

HA-DLC achieved superior performance to all evaluated comparison methods for both liver lesion classification and MGMT promoter methylation prediction (Table 1). Previous radiomics studies have demonstrated the value of incorporating tumour heterogeneity through sub-region analysis to improve classification performance [30]. Consistent with these observations, our results suggest that imaging-derived subregions capture local heterogeneity patterns that complement information contained in whole-tumour representations. This interpretation is supported by the radiomics analysis (Table 3), where features extracted from HA-DLC-predicted sub-regions consistently outperformed features extracted from whole-tumour regions and, on BraTS-RC, achieved slightly higher performance than expert-defined tumour compartments. Together, these findings indicate that explicitly modelling spatially organised imaging phenotypes within a tumour may preserve heterogeneity-related information that is particularly relevant for classification.

The clustering sensitivity analysis (Table 2) indicates that HA-DLC is relatively robust to the initial HSG configuration. On LLD-M, performance varied only slightly across clustering settings, suggesting that multiple cluster configurations can generate informative sub-region candidates. Although the performance diferences across BraTS-RC configurations were also modest, BraTS-RC showed a numerical tendency towards coarser partitions, suggesting that overly fine clustering could fragment task-relevant imaging patterns. Because the optimal configuration difered between datasets, clustering parameters were selected separately for each task. Nevertheless, the similar best-case performance achieved by k-means and DFC suggests that the efectiveness of HA-DLC is not dependent on a specific clustering algorithm. Instead, the primary benefit appears to arise from identifying, aligning, and learning from imaging-derived heterogeneous tumour regions.

Figures 3 and 4 illustrate recurring heterogeneity patterns learned by HA-DLC from imaging-derived tumour sub-regions. Despite substantial inter-patient variability in tumour appearance, corresponding sub-regions occupied similar relative tumour locations across patients, suggesting that CPSA can establish cross-patient sub-region correspondence. In BraTS-RC, MGMT promoter methylation status was associated with significant diferences in sub-region composition (Figure 4), indicating that the learned sub-regions capture information beyond global tumour morphology and intensity characteristics. Similarly, the recurring sub-region patterns observed in liver lesions (Figure 3) suggest that diferent tumour types exhibit distinct heterogeneity signatures. Together, these findings indicate that HA-DLC can identify task-relevant imaging heterogeneity patterns without requiring manual sub-region annotations. However, the biological significance of the generated sub-regions remains to be validated through histopathological or radiogenomic studies.

The ablation results (Table 4) further support the contribution of each evaluated component of HA-DLC. The segmentation-target ablation demonstrated a progressive improvement in performance when moving from no segmentation supervision to wholetumour supervision and subsequently to imaging-derived sub-region supervision. On LLD-M, the F1–kappa average increased from 0.768 without segmentation loss to 0.802 with whole-tumour supervision and further to 0.819 with sub-region supervision. Similarly, on BraTS-RC, the AUC increased from 0.623 to 0.650 and then to 0.686. Notably, imaging-derived sub-region supervision achieved performance comparable to that obtained using expert-defined anatomical BraTS tumour compartments (AUC: 0.686 versus 0.681), suggesting that efective heterogeneity-aware supervision can be obtained without manual sub-region annotation. The component-removal experiments further demonstrated the complementary roles of CPSA and the GFE stream within DSFE. Incorporating CPSA increased performance from 0.784 to 0.819 on LLD-M and from 0.618 to 0.686 on BraTS-RC, highlighting the importance of establishing consistent sub-region identities across patients. Similarly, combining the Global Feature Extraction (GFE) stream with the Heterogeneity Feature Extraction (HFE) stream improved performance from 0.769 to 0.819 on LLD-M and from 0.669 to 0.686 on BraTS-RC, demonstrating that local heterogeneity patterns and global tumour context provide complementary information for classification. Interestingly, the relative contribution of these components difered between datasets. The larger performance gain associated with GFE on LLD-M suggests that global tumour context may be particularly important for distinguishing liver lesion types, whereas the greater contribution of CPSA on BraTS-RC suggests a stronger dependence on consistent local heterogeneity patterns for MGMT promoter methylation prediction. Overall, these findings indicate that heterogeneity-aware sub-region modelling, cross-patient alignment, and multi-scale feature learning jointly contribute to improved tumour classification.

Despite the promising results, we note several limitations. First, the generated sub-regions depend on the initial clustering and alignment processes. Although the imaging-derived sub-regions achieved performance comparable to expert-defined tumour compartments on BraTS-RC, they do not necessarily correspond to established anatomical or pathological regions. This suggests that the most efective representation of tumour heterogeneity for classification may be task-dependent and does not always align with conventional tumour compartment definitions. Consequently, the biological interpretation of the learned sub-regions remains limited without histopathological or radiogenomic validation. Second, although HA-DLC demonstrated relatively stable performance across diferent clustering configurations (Table 2), the optimal clustering parameters varied between datasets. The current framework therefore requires datasetspecific selection of clustering hyperparameters, including the number of clusters for k-means and the continuity-loss coeficient for DFC. Future work will investigate adaptive strategies for automatically determining suitable clustering configurations while reducing computational overhead. Finally, the proposed framework was evaluated on two public mp-MRI datasets representing diferent tumour classification tasks. Although the consistent performance improvements observed across both datasets suggest potential generalisability, the evaluation relied on internal five-fold crossvalidation; therefore, validation on larger independent multi-centre cohorts, additional tumour types, and other imaging modalities is still required.

## 6 Conclusion

In this study, we introduced a Heterogeneity-Aware Deep Learning Classification (HA-DLC) framework for tumour classification from mp-MRI. HA-DLC combines unsupervised imaging-derived sub-region generation, cross-patient sub-region alignment, and dual-stream feature extraction to jointly capture intra-tumoural heterogeneity and global tumour context. By generating and softly aligning sub-region pseudo-labels, the framework enables heterogeneity-aware representation learning without requiring manual sub-region annotations. Experiments across two distinct MRI datasets demonstrated that HA-DLC consistently outperformed the comparison methods, supporting the value of incorporating imaging-derived heterogeneity into deep learning-based tumour classification. These findings indicate that modelling structured intra-tumoural variation can provide complementary information for both lesion-type classification and molecular-status prediction.

## References

[1] Fisher, R., Pusztai, L., Swanton, C.: Cancer heterogeneity: implications for targeted therapeutics. Br. J. Cancer 108(3), 479–485 (2013)

[2] Crippa, V., Malighetti, F., Villa, M., Graudenzi, A., Piazza, R., Mologni, L., Ramazzotti, D.: Characterization of cancer subtypes associated with clinical outcomes by multi-omics integrative clustering. Comput. Biol. Med. 162, 107064 (2023)

[3] Tsimberidou, A.M., Fountzilas, E., Nikanjam, M., Kurzrock, R.: Review of precision cancer medicine: Evolution of the treatment paradigm. Cancer Treat. Rev. 86, 102019 (2020)

[4] Makowska, Z., Boldanova, T., Adametz, D., Quagliata, L., Vogt, J.E., Dill, M.T., Matter, M.S., Roth, V., Terracciano, L., Heim, M.H.: Gene expression analysis of biopsy samples reveals critical limitations of transcriptome-based molecular classifications of hepatocellular carcinoma. J. Pathol. Clin. Res. 2(2), 80–92 (2016)

[5] Jain, R., Johnson, D.R., Patel, S.H., Castillo, M., Smits, M., Bent, M.J., Chi, A.S., Cahill, D.P.: “real world” use of a highly reliable imaging sign: “t2-flair mismatch” for identification of idh mutant astrocytomas. Neuro-Oncol. 22(7), 936–943 (2020)

[6] Parry, M.A., Srivastava, S., Ali, A., Cannistraci, A., Antonello, J., Barros-Silva, J.D., Ubertini, V., Ramani, V., Lau, M., Shanks, J., Nonaka, D., Oliveira, P., Hambrock, T., Leong, H.S., Dhomen, N., Miller, C., Brady, G., Dive, C., Clarke, N.W., Marais, R., Baena, E.: Genomic evaluation of multiparametric magnetic resonance imaging-visible and -nonvisible lesions in clinically localised prostate cancer. Eur. Urol. Oncol. 2(1), 1–11 (2018)

[7] Stoyanova, R., Pollack, A., Takhar, M., Lynne, C., Parra, N., Lam, L.L.C., Alshalalfa, M., Buerki, C., Castillo, R., Jorda, M., Ashab, H.A.-D., Kryvenko, O.N., Punnen, S., Parekh, D.J., Abramowitz, M.C., Gillies, R.J., Davicioni, E., Erho, N., Ishkanian, A.: Association of multiparametric mri quantitative imaging features with prostate cancer gene expression in mri-targeted prostate biopsies. Oncotarget 7(33), 53362–53376 (2016)

[8] Hu, L.S., D’Angelo, F., Weiskittel, T.M., Caruso, F.P., Fortin Ensign, S.P., Blomquist, M.R., et al.: Integrated molecular and multiparametric mri mapping of high-grade glioma identifies regional biologic signatures. Nat. Commun. 14(1) (2023)

[9] Dorfner, F.J., Patel, J.B., Kalpathy-Cramer, J., Gerstner, E.R., Bridge, C.P.: A review of deep learning for brain tumor analysis in mri. npj Precis. Oncol. 9(1) (2025)

[10] He, K., Gan, C., Li, Z., Rekik, I., Yin, Z., Ji, W., Gao, Y., Wang, Q., Zhang, J., Shen, D.: Transformers in medical image analysis. Intell. Med. 3(1), 59–78 (2023)

[11] Zong, W., Carver, E., Zhu, S., Schaf, E., Chapman, D., Lee, J., Bagher-Ebadian, H., Movsas, B., Wen, W., Alafif, T., Zong, X.: Prostate cancer malignancy detection and localization from mpmri using auto-deep learning as one step closer to clinical utilization. Sci. Rep. 12(1) (2022)

[12] Pasquini, L., Napolitano, A., Tagliente, E., Dellepiane, F., Lucignani, M., Vidiri, A., Ranazzi, G., Stoppacciaro, A., Moltoni, G., Nicolai, M., Romano, A., Di Napoli, A., Bozzao, A.: Deep learning can diferentiate idh-mutant from idh-wild gbm. J. Pers. Med. 11(4), 290 (2021)

[13] Cheng, J., Liu, J., Kuang, H., Wang, J.: A fully automated multimodal mri-based multi-task learning for glioma segmentation and idh genotyping. IEEE Trans. Med. Imaging 41(6), 1520–1532 (2022)

[14] Lou, M., Ying, H., Liu, X., Zhou, H.-Y., Zhang, Y., Yu, Y.: Sdr-former: A siamese dual-resolution transformer for liver lesion classification using 3d multi-phase

[15] Baid, U., Ghodasara, S., Mohan, S., Bilello, M., Calabrese, E., Colak, E., Farahani, K., et al.: The rsna-asnr-miccai brats 2021 benchmark on brain tumor segmentation and radiogenomic classification. arXiv preprint arXiv:2107.02314 (2021)

[16] Lloyd, S.: Least squares quantization in pcm. IEEE Trans. Inf. Theory 28(2), 129–137 (1982)

[17] Kim, W., Kanezaki, A., Tanaka, M.: Unsupervised learning of image segmentation based on diferentiable feature clustering. IEEE Trans. Image Process. 29, 8055– 8068 (2020)

[18] Huang, G., Liu, Z., Maaten, L., Weinberger, K.Q.: Densely connected convolutional networks. In: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR) (2017)

[19] Diakogiannis, F.I., Waldner, F., Caccetta, P., Wu, C.: ResUNet-a: A deep learning framework for semantic segmentation of remotely sensed data. ISPRS Journal of Photogrammetry and Remote Sensing 162, 94–114 (2020) https://doi.org/10. 1016/j.isprsjprs.2020.01.013

[20] Li, K., Wang, Y., Gao, P., Song, G., Liu, Y., Li, H., Qiao, Y.: Uniformer: Unified transformer for eficient spatiotemporal representation learning. In: Proc. Int. Conf. Learn. Represent. (ICLR) (2022)

[21] Ma, J., He, Y., Li, F., Han, L., You, C., Wang, B.: Segment anything in medical images. Nat. Commun. 15(1) (2024) https://doi.org/10.1038/ s41467-024-44824-z

[22] Baba, F.: Leaderboard of RSNA-MICCAI brain tumor radiogenomic classification. Kaggle (2021)

[23] Hu, M., Yang, K., Wang, J., Qiu, R.L.J., Roper, J., Kahn, S., Shu, H.-K., Yang, X.: Mgmt promoter methylation prediction based on multiparametric mri via vision graph neural network. J. Med. Imaging 11(1) (2024)

[24] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR) (2016)

[25] Tan, M., Le, Q.V.: Eficientnet: Rethinking model scaling for convolutional neural networks. In: Proc. Int. Conf. Mach. Learn. (ICML) (2019)

[26] Griethuysen, J.J.M., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V., Beets-Tan, R.G.H., Fillion-Robin, J.-C., Pieper, S., Aerts, H.J.W.L.: Computational radiomics system to decode the radiographic phenotype. Cancer Res.

77(21), 104–107 (2017)

[27] Chen, T., Guestrin, C.: Xgboost: A scalable tree boosting system. In: Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (2016)

[28] Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E.Z., DeVito, Z., Lin, Z., Desmaison, A., Antiga, L., Lerer, A.: Automatic diferentiation in pytorch. In: Proc. NeurIPS Workshop (2017)

[29] Wightman, R.: PyTorch Image Models (timm). GitHub repository (2019)

[30] Lin, M., Wynne, J.F., Zhou, B., Wang, T., Lei, Y., Curran, W.J., Liu, T., Yang, X.: Artificial intelligence in tumor subregion analysis based on medical imaging: A review. J. Appl. Clin. Med. Phys. 22(7), 10–26 (2021)