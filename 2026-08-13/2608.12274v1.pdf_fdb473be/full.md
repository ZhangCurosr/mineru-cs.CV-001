# A Neighborhood Attention Transformer Network for Enhanced 3D Segmentation of the Left Anterior Descending Artery

Rafi Ibn Sultan<sup>1</sup>, Chengyin Li<sup>2</sup>, Yiannos Demetriou<sup>1</sup>, Ahmed I. Ghanem<sup>2,3</sup>, Joshua P. Kim<sup>2</sup>, Justine Cunningham<sup>2</sup>, Hassan Bagher-Ebadian<sup>2,4,5,6</sup>, Dongxiao Zhu<sup>1,7</sup>, and Kundan S. Thind<sup>2</sup>

<sup>1</sup>Department of Computer Science, Wayne State University, Detroit, MI, 48202, USA <sup>2</sup>Department of Radiation Oncology, Henry Ford Health, Detroit, MI, 48202, USA <sup>3</sup>Department of Clinical Oncology, Alexandria University, Alexandria, 21526, Egypt <sup>4</sup>Department of Radiology, Michigan State University, E. Lansing, MI, 48824, USA <sup>5</sup>College of Osteopathic Medicine, Michigan State University, E. Lansing, MI, 48824, USA <sup>6</sup>Department of Physics, Oakland University, Rochester, MI, 48309, USA <sup>7</sup>Institute for AI and Data Science, Wayne State University, Detroit, MI, 48202, USA

Corresponding author: Rafi Ibn Sultan, Department of Computer Science, Wayne State University, Detroit, MI, 48202, USA. E-mail: rafis@wayne.edu.

## Abstract

Background: Accurate segmentation of the Left Anterior Descending (LAD) artery in 3D free-breathing, non-contrast CT is critical for cardiac dose sparing in thoracic radiotherapy. The task is inherently dificult because the LAD is extremely small, exhibits poor soft-tissue contrast, and varies substantially across patients. Even manual contours show limited inter-observer agreement on non-contrast CT, underscoring the ambiguity of the vessel boundaries.

Purpose: To develop a transformer-based segmentation framework that enhances LAD delineation in low-contrast, imbalanced CT using local-global context modeling and uncertainty-guided optimization.

Methods: We propose NA-UNETR, a 3D transformer-based segmentation model with Neighborhood Attention (NA) and Dilated NA (DiNA) blocks that jointly capture fine structural detail and long-range context. To address the limited availability of annotated LAD artery data, the model is pretrained on 1,000 CTA volumes representing general coronary anatomy and then fine-tuned using a parameter-eficient adaptation strategy on 20 free-breathing institutional CT scans of the LAD artery. A composite Diceˆa€“Focal and Hausdorf loss, dynamically balanced via homoscedastic uncertainty, improves overlap and boundary accuracy. Preprocessing included intensity clipping, contrast enhancement, and artery-centric sampling, and postprocessing refined morphology.

Results: NA-UNETR achieved comparable performance to SOTA with improved efficiency and boundary properties. It reached 45.64% Dice, 38.16 mm HD95, and 10.01 mm ASD, improving Dice by 3.10 percentage points over nnU-Net and reducing HD95 by 2.96 mm relative to Swin UNETR, while showing the strongest boundary accuracy among all models and demonstrating improved geometric and centerline stability. On

ImageCAS, it achieved 79.49% Dice, 8.89 mm HD95, and 1.02 mm ASD. Ablations confirmed that residual blocks, variable kernels, and uncertainty-weighted loss each contributed to performance stability and boundary precision.

Conclusions: NA-UNETR efectively balances local precision and global context, enabling accurate segmentation of thin, low-contrast LAD structures. Its integration of Neighborhood Attention, uncertainty-weighted loss, and LoRA-based fine-tuning demonstrates a robust and computationally eficient framework for substructure-level cardiac segmentation in radiotherapy planning.

## I. Introduction

Radiation-induced heart disease (RIHD) is a growing concern in thoracic oncology, particularly for patients undergoing radiotherapy for lung, esophageal, or breast cancers <sup>1,2</sup>. RIHD includes both acute (e.g., pericarditis) and late efects (e.g., coronary artery disease, heart failure, arrhythmias), with risks manifesting earlier than expected, sometimes within 5 years of treatment, and persisting for decades<sup>3</sup>. Among cardiac substructures, the Left Anterior Descending (LAD) artery is especially susceptible to radiation injury due to its location along the anterior interventricular groove and its role in supplying the anterior wall and most of the left ventricle<sup>4</sup>. Radiation doses as low as 10 Gy have been linked to increased coronary artery calcification and ischemic events <sup>5</sup>, while a mean heart dose greater than 10 Gy or a LAD volume receiving 15 Gy $\left( \mathrm { V _ { 1 5 G y } } \right)$ exceeding 10 percent significantly increase the risk of major adverse cardiac events in non-small cell lung cancer patients <sup>6</sup>.

Despite its clinical significance, the LAD is rarely segmented in routine radiotherapy workflows, which often rely on whole-heart doseˆa€“volume metrics<sup>7,8</sup> such as those recommended by QUANTEC<sup>9</sup>. These measures treat the heart as a uniform organ, potentially overlooking radiosensitive substructures. Studies indicate that dose to regions like the LAD<sup>10</sup>, left ventricle<sup>11</sup>, and left atrium<sup>12</sup> may better predict long-term morbidity than global heart dose. However, LAD delineation on standard free-breathing, non-contrast, nonˆa€“ECG-gated planning CT remains extremely challenging due to its small caliber, variable course, and limited soft-tissue contrast <sup>13,14,15,16</sup>. The artery often appears faint with ill-defined boundaries and is further obscured by motion artifacts<sup>17</sup>. Inter-observer studies report manual coronary artery Dice scores ranging from 0.10 to 0.53 on non-contrast CT <sup>18</sup>, underscoring the substantial disagreement even among expert readers.

Several automatic segmentation strategies have been explored for LAD segmentation. Atlas-based methods are simple to implement but depend on image registration and perform poorly on the LAD, with reported Dice Similarity Coeficients (DSC) of 0.09ˆa€“0.27 <sup>17,19,20,21</sup>. Their inability to capture anatomical variability or resolve small, low-contrast structures limits their usefulness. Deep learning approaches generally perform better but still struggle on non-contrast CT: U-Net<sup>22</sup> and nnU-Net<sup>23</sup> achieve only about 0.21 DSC for the LAD<sup>24,25</sup>, far lower than the values above 0.85 commonly reported for cardiac chambers and large vessels. This disparity stems from the intrinsic dificulty of LAD delineation on non-contrast

CT, where faint boundaries, low contrast, and motion artifacts limit the visual cues available for supervision. These factors, together with the limited availability of expert-labeled data, create a pronounced small-data bottleneck for coronary substructure segmentation. Multimodal strategies using MRI or contrast-enhanced CT improve visibility<sup>9,26,27</sup> but remain impractical for standard radiotherapy and diagnostic workflows.

While this work focuses on the LAD, insights from broader coronary artery segmentation remain relevant since the underlying challenges often overlap. Most pipelines employ convolutional neural networks (CNNs), including U-Net and its variants<sup>28,29,30</sup>, or enhanced designs with attention and multi-scale fusion<sup>31,32</sup>, as well as PSPNet-based<sup>33</sup> and ensemble approaches<sup>34</sup>. However, CNNs struggle to capture long-range anatomical dependencies that are essential for maintaining vessel continuity <sup>35,36,37</sup>. Transformer-based models address this limitation by modeling global context, allowing relationships between distant structures to be learned jointly. Yet, transformer-based approaches have rarely been explored for coronary artery segmentation and, to our knowledge, have not been evaluated for LAD segmentation in non-contrast, free-breathing planning CT, an especially dificult setting with limited annotated data.

We present NA-UNETR, a 3D transformer-based segmentation framework developed to address the small-data regime inherent to LAD segmentation. Its design integrates Neighborhood Attention (NA)<sup>38</sup> for localized feature extraction within a global transformer backbone, enabling improved representation of thin and low-contrast anatomy. To mitigate class imbalance and improve thin-structure detection, we incorporate artery-focused patch sampling and a composite loss that combines Dice-Focal and boundary-aware Hausdorf losses with homoscedastic uncertainty weighting. Crucially, NA-UNETR employs a two-stage transfer learning strategy: pretraining on a large CCTA dataset followed by parametereficient fine-tuning using LoRA on clinically curated planning CTs. This approach directly addresses the scarcity of expert-labeled LAD data and provides a principled solution to the small-data bottleneck. Together, these innovations yield a segmentation framework suited for substructure-level cardiac analysis in thoracic radiotherapy.

## II. Methods

Our task is formulated as a 3D binary segmentation problem, where the model predicts voxel-wise probabilities of the LAD artery from input CT volumes.

## II.A. Preliminaries: Neighborhood Attention (NA)

NA is a computationally eficient alternative to global self-attention<sup>38</sup> that restricts token interactions to spatially local neighborhoods while preserving contextual modeling capacity. In 3D medical volumes where the target anatomy may occupy only a minute fraction of voxels and exhibit elongated tubular geometry, unrestricted global attention can mix structurally unrelated regions and dilute weak anatomical signals within dominant background responses. By confining each query to a local $k \times k \times k$ window, NA introduces a spatial inductive bias that promotes geometrically coherent responses among adjacent voxels, improving the delineation of thin, low-contrast vascular structures.

Given an input tensor X, the NA mechanism computes attention over a local $k \times k \times k$ window centered around each query location. For each query ${ \bf q } _ { i }$ at spatial index i, attention is computed only with key-value pairs $\{ ( \mathbf { k } _ { j } , \mathbf { v } _ { j } ) | j \in \mathcal { N } ( i ) \}$ , where $\mathcal { N } ( i )$ denotes the set of neighboring indices. The output is

$$
\mathrm { N A } ( \mathbf { q } _ { i } ) = \sum _ { j \in \mathcal { N } ( i ) } \alpha _ { i j } \mathbf { v } _ { j } , \quad \alpha _ { i j } = \frac { \exp { \left( \mathbf { q } _ { i } ^ { \top } \mathbf { k } _ { j } / \sqrt { d } \right) } } { \sum _ { l \in \mathcal { N } ( i ) } \exp { \left( \mathbf { q } _ { i } ^ { \top } \mathbf { k } _ { l } / \sqrt { d } \right) } } ,\tag{1}
$$

where ${ \bf q } _ { i } , ~ { \bf k } _ { j }$ , and $\mathbf { v } _ { j }$ are query, key, and value vectors obtained via linear projections of X, and d is the dimensionality of the attention head. In practice, this attention operation is applied independently across multiple heads, where d denotes the dimensionality of each individual head.

Dilated Neighborhood Attention (DiNA). While NA efectively models local dependencies, its receptive field increases only linearly with depth. DiNA extends this formulation by sampling dilated neighborhoods, enabling the receptive field to expand progressively across layers while maintaining the underlying locality constraint. This gradual expansion supports integration of longer vessel segments across slices without resorting to fully global interactions, thereby balancing longitudinal context aggregation with local structural fidelity. The resulting output follows the same form as Eq. 1, but with the attention computed over $\mathcal { N } _ { \delta } ( i )$ instead of $\mathcal { N } ( i )$

$$
\mathrm { D i N A } _ { \delta , k } ( { \bf q } _ { i } ) = \sum _ { j \in \mathcal { N } _ { \delta } ( i ) } \alpha _ { i j } ^ { ( \delta ) } { \bf v } _ { j } , \quad \alpha _ { i j } ^ { ( \delta ) } = \frac { \exp \Big ( { \bf q } _ { i } ^ { \top } { \bf k } _ { j } / \sqrt { d } + b ( i , j ) \Big ) } { \sum _ { l \in \mathcal { N } _ { \delta } ( i ) } \exp \Big ( { \bf q } _ { i } ^ { \top } { \bf k } _ { l } / \sqrt { d } + b ( i , l ) \Big ) } .\tag{2}
$$

Here, $b ( i , j )$ denotes a learnable relative positional bias between query position i and key position $j ,$ which encodes spatial relationships within the attention window. By alternating NA and DiNA layers, the model preserves locality while exponentially expanding its receptive field without additional computational cost, enabling richer integration of short- and longrange context.

## II.B. NA-UNETR: Architecture

NA-UNETR (illustrated in Figure 1) is a 3D encoderˆa€“decoder segmentation framework with a transformer-based architecture incorporating neighborhood attention within a UNETR-style backbone <sup>39</sup>.

## II.B.1. Encoder

As shown in Figure 1, NA-UNETR processes a 3D input image $\mathbf { X } \in \mathbb { R } ^ { 1 \times H \times W \times D }$ , where $( H , W , D )$ are the spatial dimensions and the single input channel indicates a grayscale CT volume. The embedding dimension d in the encoder is set to 48. The visual encoder begins with an overlapping tokenizer consisting of two consecutive $3 \times 3 \times 3$ convolutions with strides of $2 \times 2 \times 2$ and $1 \times 1 \times 1$ , which reduce spatial resolution while introducing inductive bias and local connectivity. To enhance locality modeling, depthwise $3 \times 3 \times 3$ convolutions are also used within the tokenizer and feed-forward networks of each transformer block to preserve fine-grained anatomical details.

The encoder is divided into four sequential encoder stages followed by a bottleneck stage (left of Figure 1). Each stage contains multiple Neighborhood Attention Transformer (NAT) blocks, which form the core processing units. A residual convolution (Res-Conv) layer is placed before each NAT block to stabilize training and improve gradient flow. Within each NAT block, Neighborhood Attention captures local spatial dependencies within a defined neighborhood, while DiNA can optionally extend this receptive field by introducing a dilation factor. Each NAT block consists of an NA or DiNA layer followed by Layer Normalization (LN), a Multi-Layer Perceptron (MLP), and a second LN. The number of NAT blocks per stage (including the bottleneck stage) is empirically set to (3, 4, 6, 18, 5), and kernel sizes to (7, 7, 7, 3, 3), following <sup>38</sup>. Each stage is preceded by an overlapping downsampler (except the final one), which halves the spatial resolution and doubles the feature channels. The resulting feature maps at each stage are denoted as $( F _ { 2 } , H _ { 2 } , W _ { 2 } , D _ { 2 } ) , ( F _ { 3 } , H _ { 3 } , W _ { 3 } , D _ { 3 } )$ 2 $( F _ { 4 } , H _ { 4 } , W _ { 4 } , D _ { 4 } )$ , and $( F _ { 5 } , H _ { 5 } , W _ { 5 } , D _ { 5 } )$ , where F represents the number of channels that increase by $2 n .$ and H, W, and D decrease by 2n at each successive stage. At each stage, the refined feature map is forwarded to the next stage and retained for decoder skip connections. The final encoder output (bottleneck features) serves as the input to the decoder.

![](images/d3aec9d0cef2c225f196ff875de63ae74e7ed3052f383ffeee69967f86e1d54d.jpg)  
Figure 1: Overall architecture of the proposed NA-UNETR model. The encoder side is divided into four encoder stages, each containing multiple NAT blocks preceded by a residual convolution layer. Skip connections link encoder stages to the corresponding decoder stages, which use residual blocks and upsampling to produce the final segmentation output. Each NAT block can be configured to include DiNA in addition to standard NA, or to use only NA without dilation.

## II.B.2. Decoder

The decoder in NA-UNETR follows a symmetric U-shaped design (right of Figure 1), progressively reconstructing the spatial resolution while integrating semantic information from the encoder through skip connections. Feature maps from each encoder stage are refined by a residual block with two $3 \times 3 \times 3$ convolutions and instance normalization, preserving spatial resolution and channel dimensionality. These refined maps are then upsampled by a deconvolution (transposed convolution) layer that doubles the spatial resolution. Each upsampled feature map is concatenated with its corresponding encoder feature map at the same resolution to restore spatial detail. The concatenated features are further processed by another residual block to fuse contextual and spatial information. After all decoding stages, the resulting feature maps are aggregated through a final convolutional head composed of a $1 \times 1 \times 1$ convolution followed by a sigmoid activation, producing the voxel-wise probability map of the LAD artery. The use of skip connections, deconvolution layers, and concatenation operations ensures that the decoder combines high-level semantic understanding with low-level spatial precision to produce anatomically coherent segmentation outputs.

## II.C. Datasets

## II.C.1. Institutional Dataset: LAD-SEG

We utilize LAD-SEG, an IRB-approved dataset of free-breathing 3D CT scans from 20 lung cancer patients, each with physician-delineated ground-truth contours of the LAD artery. The scans have a spatial resolution of (1.17, 1.17, 3.0) mm and an in-plane size of $5 1 2 \times 5 1 2$ voxels, with slice counts ranging from 102 to 142. The LAD is represented by small, thin structures within the volume, with an average foreground ratio of $1 . 7 \times 1 0 ^ { - 5 }$ and an average of 540 foreground voxels per case (ranging from 312 to 1087 voxels). In terms of intensity characteristics, the foreground Hounsfield Unit (HU) range spans from -669 to 269, with the 1% and 99% intensity percentiles lying between -238 and 150. Levene’s test for homogeneity of variance, computed across three key attributes: voxel intensity, artery size, and boundary complexity, yielded a statistic of 0.1619 with a p-value of 0.8511, indicating no statistically significant variance diferences among the samples.

## II.C.2. Public Dataset: ImageCAS

We leverage the ImageCAS dataset <sup>40</sup>, a largeˆa€‘scale, publicly available benchmark We leverage the ImageCAS dataset40, a

specifically designed for coronary artery segmentation from computed tomography angiography (CTA) images. ImageCAS consists of 1,000 highˆa€‘resolution 3D CTA volumes acquired using Siemens 128ˆa€‘slice dualˆa€‘source scanners at Guangdong Provin cial Peopleˆa€™s Hospital between 2012 and 2018. Each scan (512A—512 <sup>˜</sup> A—206ˆa <sup>˜</sup> €“275 voxels, 0.29ˆa€“0.43 mm inˆa€‘plane resolution, 0.25ˆa€“0.45 mm slice spacing) includes meticulous voxelˆa€‘wise annotations of both left and right coronary arteries by two expert radiologists, with crossˆa€‘validation and adjudication by a third in case of disagreement. Only binary coronary artery masks are provided in the dataset, without further subclass labels or vesselˆa€‘type categories.

![](images/ecabeecc10cd280112c1b672c277c47e6b742a85bcda7ba84b8481694159cdda.jpg)

## II.D. Data Preprocessing

We apply a preprocessing pipeline tailored to the LAD-SEG dataset to address class im-

Figure 2: Overview of preprocessing, postprocessing, and augmentation used to address severe class imbalance and enhance vessel-tobackground contrast in non-contrast CT scans.

balance and low vessel-to-background contrast, as illustrated in Figure 2.

## II.D.1. Addressing Data Imbalance

To alleviate severe class imbalance, training patches of size (96, 96, 96) were extracted based on the typical spatial extent and morphology of coronary arteries. For each training iteration, an equal number of patches were randomly cropped around positive (LAD) and negative (background) regions in a 1:1 ratio, ensuring balanced exposure of both classes during optimization. Four random patches were generated per image in each iteration, promoting robust learning of small vessel structures while preventing bias toward background regions.

## II.D.2. Addressing Low Contrast

LAD intensity statistics in Section II.C.1. show that vessel voxels lie primarily within a narrow soft-tissue range, with most values between the 1st and 99th percentiles of −238 and 150 HU. Accordingly, we clipped CT intensities to [−200, 400] HU to standardize contrast and suppress unrelated highdensity regions. Following clipping, all intensity values were linearly rescaled to the range [0, 1]. Local contrast was randomly enhanced using a gamma adjustment factor between 1.6 and 1.8 (applied with probability 0.8), while random intensity shifts promoted invariance to scanner-specific baseline

![](images/763ba884eff55aad403f7d59f97814b29894ba2e36cbd143a885b19143c3717d.jpg)  
Figure 3: LAD artery segmentation without preprocessing (left) and with preprocessing (right). Each shows a low-contrast CT patch and its predicted mask overlaid on the ground truth (yellow). Red boxes highlight regions where preprocessing improves vessel edge clarity and continuity.

diferences. A Savitzkyˆa€“Golay filter with a window length of 5 and polynomial order of 2 was applied for smoothing, followed by random global intensity shifts of up to ±0.10 with a probability of 0.5. The qualitative impact of these preprocessing steps is illustrated in Figure 3, where the preprocessed images show improved vessel visibility and continuity in low-contrast regions compared to the non-preprocessed inputs.

## II.D.3. Geometric and Structural Augmentations

Training images and labels were randomly rotated (up to $\pi / 3 0$ radians, $\pm 6 ^ { \circ } )$ and scaled $( \mathrm { u p ~ t o } \pm 5 \% )$ while maintaining the target spatial size. These small perturbations simulated anatomical and positioning variability without distorting vessel morphology. Random Gaussian sharpening with variable kernel widths enhanced local edge details, while low-magnitude bias fields introduced smooth, gradual low-frequency intensity variations to simulate scannerinduced inhomogeneities and improve generalization to clinical data.

## II.E. Loss Function

We introduce a loss formulation that integrates a segmentation loss with a boundary-aware loss to better refine the contours of thin structures. The two components are balanced using homoscedastic uncertainty weighting <sup>41</sup>, allowing their relative contributions to adjust dynamically during training.

## II.E.1. Segmentation Loss: Dice-Focal Loss

We adopt Dice-Focal loss as the primary segmentation loss, which combines the regionˆa€‘overlap sensitivity of Dice Loss<sup>42</sup> with the hardˆa€‘example focusing of Focal Loss<sup>43</sup> making it more suitable for highly imbalanced anatomical structures. Given the predicted probability map $\hat { \mathbf Y }$ and the ground truth label map Y, the Dice component is formulated as:

$$
\mathcal { L } _ { \mathrm { D i c e } } = 1 - \frac { 2 \sum _ { n } y _ { n } \hat { y } _ { n } + \epsilon } { \sum _ { n } y _ { n } + \sum _ { n } \hat { y } _ { n } + \epsilon } ,\tag{3}
$$

where $y _ { n }$ and ${ \hat { y } } _ { n }$ denote the ground truth and prediction at voxel n, and ϵ is a small constant for numerical stability.

The focal component modulates the standard cross-entropy term by down-weighting well-classified examples and focusing on harder ones:

$$
\mathcal { L } _ { \mathrm { F o c a l } } = - \frac { 1 } { N } \sum _ { n } \alpha ( 1 - \hat { y } _ { n } ) ^ { \gamma } y _ { n } \log ( \hat { y } _ { n } ) + ( 1 - \alpha ) \hat { y } _ { n } ^ { \gamma } ( 1 - y _ { n } ) \log ( 1 - \hat { y } _ { n } ) ,\tag{4}
$$

where α balances positive and negative samples and $\gamma$ controls the focusing strength.

The combined Dice-Focal Loss is then defined as:

$$
\mathcal { L } _ { \mathrm { D i c e - F o c a l } } = \lambda _ { 1 } \mathcal { L } _ { \mathrm { D i c e } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { F o c a l } } ,\tag{5}
$$

with $\lambda _ { 1 }$ and $\lambda _ { 2 }$ controlling the contribution of each component. This loss guides the network toward high overlap with ground truth while emphasizing dificult voxels along the vessel.

## II.E.2. Boundary Loss: Hausdorf Loss

To improve boundary alignment alongside segmentation accuracy, we adopt Hausdorf Loss <sup>44</sup> based on the Hausdorf distance, which explicitly penalizes large deviations between predicted and true boundaries, improving boundary precision.

Let $\partial \hat { \mathbf Y }$ and ∂Y denote the sets of boundary points extracted from the predicted segmentation $\hat { \mathbf Y }$ and the ground truth Y. The symmetric Hausdorf distance between them is defined as:

$$
d _ { H } ( \partial { \hat { \mathbf { Y } } } , \partial { \mathbf { Y } } ) = \operatorname* { m a x } \left\{ \operatorname* { s u p } _ { x \in \partial { \hat { \mathbf { Y } } } } \operatorname* { i n f } _ { y \in \partial { \mathbf { Y } } } d ( x , y ) , \ \operatorname* { s u p } _ { y \in \partial { \mathbf { Y } } } \operatorname* { i n f } _ { x \in \partial { \hat { \mathbf { Y } } } } d ( x , y ) \right\} ,\tag{6}
$$

where $d ( x , y )$ is the Euclidean distance between points x and $y .$ . In practice, we use a diferentiable approximation:

$$
\mathcal { L } _ { \mathrm { H a u s d o r f } } = \frac { 1 } { | \partial \hat { \mathbf { Y } } | } \sum _ { x \in \partial \hat { \mathbf { Y } } } \operatorname* { m i n } _ { y \in \partial \mathbf { Y } } d ( x , y ) ^ { 2 } + \frac { 1 } { | \partial \mathbf { Y } | } \sum _ { y \in \partial \mathbf { Y } } \operatorname* { m i n } _ { x \in \partial \hat { \mathbf { Y } } } d ( x , y ) ^ { 2 } .\tag{7}
$$

## II.E.3. Loss Balancing Using Homoscedastic Uncertainty

During training, the contributions of the segmentation loss and the boundary loss are dynamically balanced using homoscedastic uncertainty weighting<sup>41</sup>. This approach enables the model to learn the relative importance of each task, eliminating the need for dificult and task-specific manual weight tuning. Two learnable logˆa€‘variance parameters, log $\sigma _ { 1 }$ and log $\sigma _ { 2 }$ , are introduced for the Dice-Focal Loss and the Hausdorf Loss, respectively; these are exponentiated and clamped to obtain the variance terms $\sigma _ { 1 } ^ { 2 }$ and $\sigma _ { 2 } ^ { 2 }$ . A zero-mean Gaussian noise term $\epsilon \sim \mathcal { N } ( 0 , \sigma _ { n } ^ { 2 } )$ with small variance was added to the Hausdorf Loss to form $\tilde { \mathcal { L } } _ { \mathrm { H a u s d o r f f } } .$ slowing convergence and encouraging exploration of both global and local boundary features, which in turn prevented overfitting to early boundary estimates and promoted smoother, more consistent boundary learning. The total loss is then computed as

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \frac { 1 } { 2 \sigma _ { 1 } ^ { 2 } } \mathcal { L } _ { \mathrm { D i c e - F o c a l } } + \frac { 1 } { 2 \sigma _ { 2 } ^ { 2 } } \tilde { \mathcal { L } } _ { \mathrm { H a u s d o r f f } } + \log \sigma _ { 1 } + \log \sigma _ { 2 } ,\tag{8}
$$

where $\mathcal { L } _ { \mathrm { D i c e - F o c a l } }$ is the combined Dice and focal Loss, $\tilde { \mathcal { L } } _ { \mathrm { H a u s d o r f } }$ is the noiseˆa€‘perturbed Hausdorf Loss, $\sigma _ { 1 } ^ { 2 }$ and $\sigma _ { 2 } ^ { 2 }$ control the relative weighting of the two losses, and the logarithmic terms act as regularizers.

## II.F. Data Postprocessing

To refine the predicted LAD masks and reduce spurious false positives, we applied a threestep postprocessing pipeline. First, only the largest connected component corresponding to the foreground was retained, ensuring that isolated noisy predictions were discarded. Next, connected components smaller than 64 voxels were removed to suppress small, irrelevant structures. Finally, residual holes within the retained LAD region were filled to produce contiguous, anatomically plausible segmentations.

## III. Implementation Details

## III.A. Training Strategy: Pretraining and Fine-tuning

We first pretrained all models on the ImageCAS dataset<sup>40</sup>, which consists of 1,000 3D CTA volumes of coronary arteries. This pretraining provides generalized encoder representations of coronary artery structures that transfer efectively to smaller downstream datasets, such as LAD artery segmentation. Let $\mathcal { D } _ { \mathrm { p r e } } = \{ ( \mathbf { X } ^ { ( i ) } , \mathbf { Y } ^ { ( i ) } ) \} _ { i = 1 } ^ { 1 0 0 0 }$ denote this dataset, where $\mathbf { X } ^ { ( i ) } \in \mathbb { R } ^ { C \times H \times W \times D }$ are the input volumes and $\mathbf { Y } ^ { ( i ) }$ are their corresponding ground-truth segmentations. The parameters of the pretrained encoder and decoder are denoted by $\theta _ { \mathrm { e n c } } ^ { * }$ and $\theta _ { \mathrm { d e c } } ^ { * }$ , respectively.

For fine-tuning on our in-house LAD-Seg dataset $\mathcal { D } _ { \mathrm { L A D } }$ , all benchmark models and our NA-UNETR were initialized with the pretrained weights $( \theta _ { \mathrm { e n c } } ^ { \ast }$ and $\theta _ { \mathrm { d e c } } ^ { * } )$ . To eficiently adapt these models, we employed Low-Rank Adaptation $\mathrm { ( L o R A ) ^ { 4 5 } }$ , a parameter-eficient fine-tuning method that adapts large models without retraining all their weights. The attention layers within the encoder were frozen, and we fine-tuned only the decoder parameters $\left( \theta _ { \mathrm { d e c } } \right)$ and the inserted LoRA adapters. Specifically, the MLP layers in each encoder block were replaced with LoRA modules of rank $r = 8$ , defined as $W  W + A B$ , where W is the original pretrained MLP weight matrix and $A \in \mathbb { R } ^ { d \times r }$ and $\boldsymbol { B } \in \mathbb { R } ^ { r \times d }$ are the trainable low-rank factors. During this fine-tuning stage, only $\theta _ { \mathrm { d e c } } , A$ , and B were updated, while all other pretrained weights remained fixed.

## III.B. Experiment Setup

We designed two experimental approaches based on the two datasets.

The common setup for both is as follows:

All models in all experimental settings were trained using our proposed loss function strategy, with a balancing factor $\alpha =$ 0.8 and focusing parameter $\gamma = 2$ for the focal loss component, where class weights were set to 0.1 for the background and 0.9 for the foreground to counter the class imbalance. The weighting coeficients $\lambda _ { 1 }$ and $\lambda _ { 2 }$ were set to $\lambda _ { 1 } = 1$ and $\lambda _ { 2 } = 1$ in all experiments based on empirical evaluation, as alternative settings (0.5/1, 1/0.5, and $2 / 1 )$ did not improve DSC or HD95. Class imbalance was handled through the focal loss parameters and class weighting described above. Optimized with the AdamW optimizer (learning

![](images/2eb1f11a63e2dd6d97c0408f686cad130c9ce1983f53f9085f5682fafaae2a61.jpg)  
Figure 4: Overview of the two experimental approaches: the top section shows pretraining on ImageCAS followed by fine-tuning on LAD-SEG, while the bottom section shows direct training on ImageCAS.

rate 10<sup>−4</sup>, weight decay 10<sup>−5</sup>). All experiments were implemented in PyTorch 2.5.1 with Python 3.9.21 and executed on a server equipped with eight NVIDIA A100 GPUs (40 GB memory each), with each model trained on a single GPU. The specific configurations for each experimental approach (Figure 4) are detailed below.

## III.B.1. Pretraining & Fine-Tuning Approach

In this approach, each model was first pretrained on a large public dataset to learn generalized feature representations and then fine-tuned on the smaller in-house dataset to tackle the limited availability of annotated LAD scans for supervised training.

ImageCAS. In the pretraining stage, models were trained on the full ImageCAS dataset, comprising 1,000 images, with 5% reserved for validation. Training was conducted for 100 epochs, with validation performed every 10 epochs to monitor performance and store the best-performing weights. Preprocessing included clipping voxel intensities to the range [−200, 500] Hounsfield units, followed by linear scaling to [0, 1]. To address class imbalance, a 1:1 random cropping strategy was applied between foreground (vessel) and background regions. Additional augmentations included random intensity shifts, random afine transformations, and other standard preprocessing operations. Each benchmark model was trained under this configuration, and the resulting pretrained weights will be made publicly available to facilitate further research.

LAD-SEG. In the fine-tuning stage, we applied a 5-fold split of the LAD-SEG dataset, with each fold serving as an independent validation set. The dataset was randomly partitioned into five approximately equal folds using a fixed random seed for reproducibility. In each iteration, four folds were used for training and one for validation, with the process repeated until every fold had served as the validation set exactly once. Model predictions were evaluated against physician-delineated ground-truth LAD contours, and final performance was reported as the average across all folds. Each model (including all baselines and NA-UNETR) was trained for 200 epochs under the same training protocol, using identical LAD-SEG preprocessing and postprocessing procedures described in subsection II.D. and subsection II.F., the same data splits, sampling strategy, loss formulation, and optimization settings.

## III.B.2. Single-Dataset Direct Training

In this approach, both training and evaluation were performed on the public dataset to assess model performance on a well-established benchmark.

ImageCAS. In the single-dataset training setting, models were trained and evaluated exclusively on the ImageCAS dataset to enable direct comparison with benchmark methods on this public dataset. Following the oficial protocol of <sup>40</sup>, we performed 4-fold splits, training for 30 epochs each split. Preprocessing steps were identical to those used in the previous approach.

## III.C. Benchmark Models

A diverse set of CNN-based and Transformer-based segmentation models that represent state-of-the-art approaches in medical image analysis were selected as benchmark models. Each model followed its oficial implementation, with dataset-specific tweaks applied (consistent with those used in our proposed model) for fair evaluation. For Transformer-based segmentation backbones (e.g., UNETR), we applied LoRA to the MLP layers within each Transformer block for parameter-eficient fine-tuning. In contrast, for CNN-based models (e.g., U-Net, U-Net++), which lack MLP components, LoRA was not used; instead, we fine-tuned only the decoder while keeping the encoder frozen.

## CNN-based Models

• U-Net<sup>22</sup>: Classic encoderˆa€“decoder with skip connections, preserving spatial details for biomedical segmentation.

• UNet++<sup>46</sup>: U-Net variant with nested dense skip connections for improved multiscale feature fusion.

• nnU-Net<sup>23</sup>: Self-configuring framework that automatically adapts preprocessing, architecture, and training to datasets.

• MedNeXt<sup>47</sup>: ConvNeXt-inspired CNN using large kernels and residual connections for eficient long-range context capture.

## Transformer-based Models

• UNETR<sup>39</sup>: Transformer-based encoderˆa€“decoder with a ViT encoder and multiscale feature aggregation in the decoder.

• Swin UNETR<sup>48</sup>: Swin Transformer backbone using shifted-window self-attention for locality and eficiency.

• Swin UNETR-V2<sup>49</sup>: Refined Swin UNETR with architectural and training improvements for stable representation learning.

• nnFormer<sup>50</sup>: Hybrid CNNˆa€“Transformer for 3D segmentation, balancing local convolutions with global attention.

## III.D. Evaluation Metrics

We assess model performance using four primary metrics: Dice Similarity Coeficient (DSC, %), centerline Dice (clDice, %), 95th-percentile Hausdorf Distance (HD95, mm), and Average Surface Distance (ASD, mm).

The Dice Similarity Coeficient (DSC) measures the volumetric overlap between the predicted segmentation map P and the ground truth mask G, defined as:

$$
\mathrm { D S C } = \frac { 2 \sum _ { j } P _ { j } G _ { j } } { \sum _ { j } P _ { j } ^ { 2 } + \sum _ { j } G _ { j } ^ { 2 } } \times 1 0 0 ,\tag{9}
$$

where $P _ { j }$ and $G _ { j }$ denote the predicted probability and ground truth value for voxel j. Higher DSC indicates better overlap.

To complement DSC for thin tubular structures, we also report centerline Dice (clDice), which evaluates topological agreement by comparing the skeletons of the prediction and ground truth. Let $S _ { P } = \mathrm { s k e l } ( P )$ and $S _ { G } = { \mathrm { s k e l } } ( G )$ denote their centerline skeletons. clDice is defined as:

$$
{ \mathrm { c l D i c e } } = { \frac { 2 T _ { \mathrm { p r e c } } T _ { \mathrm { s e n s } } } { T _ { \mathrm { p r e c } } + T _ { \mathrm { s e n s } } } } \times 1 0 0 , \quad T _ { \mathrm { p r e c } } = { \frac { | S _ { P } \cap G | } { | S _ { P } | } } , \quad T _ { \mathrm { s e n s } } = { \frac { | S _ { G } \cap P | } { | S _ { G } | } } .\tag{10}
$$

Higher clDice indicates better preservation of the vessel trajectory.

The 95th-percentile Hausdorf distance (HD95) evaluates the geometric alignment between the predicted segmentation boundary and the ground truth, defined as:

$$
{ \mathrm { H D 9 5 } } = { \mathrm { q u a n t i l e } } _ { 9 5 \% } \left( \operatorname* { m a x } _ { x \in \partial { \hat { p } } } \operatorname* { m i n } _ { y \in \partial G } \| x - y \| \right) ,\tag{11}
$$

where $\partial \hat { p }$ and ∂G denote predicted and ground-truth boundaries. Lower HD95 indicates fewer large boundary deviations.

The Average Surface Distance (ASD) quantifies mean symmetric boundary distance:

$$
\mathrm { A S D } = \frac { 1 } { \left| \partial \hat { p } \right| + \left| \partial G \right| } \left( \sum _ { x \in \partial \hat { p } } \operatorname* { m i n } _ { y \in \partial G } \left\| x - y \right\| + \sum _ { y \in \partial G } \operatorname* { m i n } _ { x \in \partial \hat { p } } \left\| y - x \right\| \right) ,\tag{12}
$$

III.D. Evaluation Metrics

with lower values indicating closer average alignment between predicted and true boundaries. To assess the statistical significance of performance diferences, we conducted the nonparametric Mann–Whitney U test on the per-case metric values across all methods. The test was applied separately to the results from both datasets (our in-house LAD-SEG dataset and the public ImageCAS dataset) for DSC, HD95, and ASD. Reported p-values indicate whether the improvements achieved by NA-UNETR over competing baselines are statistically significant $\left( p < 0 . 0 5 \right)$

![](images/444cc6fd82b28f9729d47163fbe8cd20598a0a8e971f69709eb5b7f658f08b68.jpg)  
Figure 5: Representative qualitative results on randomly selected slices from the LAD-SEG and ImageCAS validation datasets. Each row corresponds to a randomly chosen slice from a randomly selected case. The first column shows the CT image, followed by the ground truth annotation, and predictions obtained using the proposed NA-UNETR (ours) and other state-of-the-art (SOTA) segmentation models. To facilitate visual comparison of the vascular structures, the red bounding boxes indicate manually selected regions containing the target vessel and are shown as zoomed-in views of the predicted masks.

## IV. Results

## IV.A. Qualitative Results

Qualitative comparisons on randomly selected slices from both LAD-SEG and ImageCAS datasets (Figure 5) demonstrate that NA-UNETR provides clearer and more accurate delineations of the coronary arteries compared with benchmark models. Each prediction is accompanied by a zoomed-in view (red bounding box) to highlight subtle diferences in boundary quality. As shown, NA-UNETR consistently produces vessel contours that align more closely with the ground truth, whereas competing SOTA methods often exhibit fragmented boundaries, spurious predictions, or missed segments. These improvements are particularly pronounced on the LAD-SEG dataset, which is more challenging due to its low contrast and thin vessel boundaries. In such cases, NA-UNETR preserves structural continuity and reduces noise, ofering more reliable vessel localization. To further illustrate the three-dimensional structural consistency of NA-UNETR and its comparison with nnU-Net, representative axial, coronal, and sagittal views are presented in Figure 6.

## IV.B. Quantitative Results

Table 1: Quantitative results of NA-UNETR against diferent SOTA models on our institutional data LAD-SEG (both CNN-based and Transformer-based). Each entry reports the mean A<sup>ˆ</sup>± standard deviation. NA-UNETR (no-dil.) is NA-UNETR with NA blocks instead of DiNA blocks. The best results are bolded.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>DSC (%) ↑</td><td rowspan=1 colspan=1>clDice (%) ↑</td><td rowspan=1 colspan=1>HD95 (mm)↓</td><td rowspan=1 colspan=1>ASD $\mathbf { \Pi } ( \mathbf { m m } ) \downarrow$ </td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathrm { U - N e t } ^ { 2 2 } } }$ </td><td rowspan=1 colspan=1> $3 5 . 3 0 { \pm } 5 . 9 1 $ </td><td rowspan=1 colspan=1> $\overline { { 3 7 . 2 0 { \pm } 1 1 . 3 1 } }$ </td><td rowspan=1 colspan=1> $\overline { { 4 3 . 8 0 { \pm } 7 . 5 5 } }$ </td><td rowspan=1 colspan=1> $1 2 . 8 0 { \pm } 3 . 2 7 $ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { U N e t } { + } + $ </td><td rowspan=1 colspan=1> $3 5 . 5 7 { \pm } 8 . 2 4 $ </td><td rowspan=2 colspan=1> $4 2 . 4 0 { \pm } 1 0 . 4 3 \ $ </td><td rowspan=1 colspan=1> $4 0 . 9 5 { \pm } 6 . 0 5$ </td><td rowspan=1 colspan=1> $1 2 . 0 9 { \pm } 1 . 9 1 $ </td></tr><tr><td rowspan=2 colspan=1> $\mathrm { { n n U - N e t ^ { 2 3 } } }$  $\mathrm { M e d N e X t ^ { 4 7 } }$ </td><td rowspan=2 colspan=1> $4 2 . 5 4 { \pm } 2 . 9 0 $ 38.79±6.44</td><td rowspan=2 colspan=1> $4 0 . 9 1 { \pm } 1 0 . 4 9 $ </td><td rowspan=1 colspan=1> $3 9 . 6 8 { \pm } 5 . 9 2$ </td><td rowspan=1 colspan=1> $1 0 . 3 7 { \pm } 1 . 5 6 $ </td></tr><tr><td rowspan=1 colspan=1> $4 1 . 6 8 { \pm } 4 . 8 0 $ </td><td rowspan=1 colspan=1>11.78±2.56</td></tr><tr><td rowspan=4 colspan=1> $\overline { { \mathrm { U N E T R ^ { 3 9 } } } }$  $\mathrm { S w i n ~ U N E T R ^ { 4 8 } }$  $\mathrm { S w i n U N E T R - V 2 ^ { 4 9 } }$  $\mathrm { n n F o r m e r ^ { 5 0 } }$ </td><td rowspan=4 colspan=1> $\overline { { 4 2 . 1 3 { \pm } 3 . 5 5 } }$ 44.78±6.0844.11±4.54 $4 2 . 0 4 { \pm } 5 . 3 4$ </td><td rowspan=4 colspan=1> $4 2 . 0 8 { \pm } 8 . 7 0 $  $4 3 . 7 6 { \pm } 9 . 3 3$  $4 3 . 5 0 { \pm } 9 . 7 9 $  $4 1 . 8 8 { \pm } 9 . 7 1 $ </td><td rowspan=1 colspan=1>40.65±6.96</td><td rowspan=3 colspan=1>10.44±1.2610.79±2.45 $1 0 . 1 8 { \pm } 2 . 1 0 $ </td></tr><tr><td rowspan=1 colspan=1>41.12±5.98</td></tr><tr><td rowspan=1 colspan=1> $4 0 . 2 9 { \pm } 7 . 2 8 $ </td></tr><tr><td rowspan=1 colspan=1> $4 0 . 4 3 { \pm } 7 . 1 1 $ </td><td rowspan=1 colspan=1> $1 1 . 0 5 { \pm } 3 . 0 2 $ </td></tr><tr><td rowspan=1 colspan=1> $\overline { { { \mathrm { N A - U N E T R ~ } } ( \mathrm { n o - d i l . } ) } }$  $\mathrm { N A \mathrm { - } U N E T R \ ( o u r s ) }$ </td><td rowspan=1 colspan=1> $\overline { { 4 4 . 0 9 { \pm } 3 . 8 9 } }$  $\mathbf { 4 5 . 6 4 } \pm \mathbf { 4 . 8 6 }$ </td><td rowspan=1 colspan=1> $\overline { { 4 3 . 4 5 { \pm 9 . 0 6 } } }$  ${ \bf 4 4 . 3 9 2 8 . 3 8 }$ </td><td rowspan=1 colspan=1> $\overline { { 4 0 . 1 0 { \pm 9 . 3 0 } } }$  $\mathbf { 3 8 . 1 6 { \pm } 4 . 3 7 }$ </td><td rowspan=1 colspan=1> $\overline { { { \bf 9 . 5 5 } \pm { \bf 3 . 2 4 } } }$  $1 0 . 0 1 { \pm } 1 . 3 9 $ </td></tr></table>

Table 1 and Table 2 summarize the performance of NA-UNETR (ours) and NA-UNETR (no-dil.), the no-dilation variant of the model, compared with state-of-the-art CNN

Table 2: Quantitative results of NA-UNETR against diferent SOTA models on the public data ImageCAS dataset (both CNN-based and Transformer-based). NA-UNETR (no-dil.) is NA-UNETR with NA blocks instead of DiNA blocks. Each entry reports the mean $\hat { \mathsf { A } } \pm$ standard deviation. The best results are bolded.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>DSC (%) ↑ c</td><td rowspan=1 colspan=1>lDice (%) ↑</td><td rowspan=1 colspan=1>HD95 (mm) ↓</td><td rowspan=1 colspan=1>ASD (mm) ↓</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathrm { U - N e t } ^ { 2 2 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 7 3 . 4 7 { \pm } 0 . 3 2 } }$ </td><td rowspan=1 colspan=1> $\overline { { 8 2 . 0 7 { \pm } 0 . 4 9 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 0 . 6 9 { \pm 0 . 7 8 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 . 3 9 { \pm } 0 . 0 6 } }$ </td></tr><tr><td rowspan=3 colspan=1> $\mathrm { U N e t } { + } + \AA ^ { 4 6 }$  $\mathrm { { n n U - N e t ^ { 2 3 } } }$  $\mathrm { M e d N e X t ^ { 4 7 } }$ </td><td rowspan=3 colspan=1> $7 8 . 5 7 { \pm } 0 . 5 1 $  $7 7 . 9 8 { \pm } 0 . 4 2 $  $7 7 . 2 0 { \pm } 0 . 1 4 $ </td><td rowspan=1 colspan=1> $8 4 . 7 1 { \pm } 0 . 6 9$ </td><td rowspan=1 colspan=1> $9 . 4 6 { \pm } 0 . 8 4$ </td><td rowspan=1 colspan=1> $1 . 1 5 { \pm } 0 . 0 7$ </td></tr><tr><td rowspan=2 colspan=1> $8 5 . 9 1 { \pm } 0 . 5 7 $  $8 4 . 6 9 { \pm } 0 . 0 9$ </td><td rowspan=1 colspan=1> $9 . 0 0 { \pm } 0 . 6 3 $ </td><td rowspan=1 colspan=1> $1 . 1 4 { \pm } 0 . 0 5$ </td></tr><tr><td rowspan=1 colspan=1> $9 . 5 6 { \pm } 0 . 3 5 $ </td><td rowspan=1 colspan=1> $1 . 1 9 { \pm } 0 . 0 3 $ </td></tr><tr><td rowspan=4 colspan=1> $\overline { { \mathrm { U N E T R ^ { 3 9 } } } }$  $\mathrm { S w i n ~ U N E T R ^ { 4 8 } }$  $\mathrm { S w i n U N E T R - V 2 ^ { 4 9 } }$  $\mathrm { n n F o r m e r ^ { 5 0 } }$ </td><td rowspan=3 colspan=1> $7 6 . 0 9 { \scriptstyle \pm 0 . 4 1 }$  $7 7 . 7 1 { \pm } 0 . 4 6$  $7 8 . 0 3 { \pm } 0 . 4 8 $ </td><td rowspan=3 colspan=1> $\overline { { 8 2 . 5 1 { \pm } 0 . 3 8 } }$  $8 5 . 3 6 { \pm } 0 . 4 8$  $8 5 . 6 1 { \pm } 0 . 6 7$ </td><td rowspan=1 colspan=1> $9 . 5 5 { \pm } 0 . 7 6 $ </td><td rowspan=2 colspan=1> $1 . 2 2 { \pm } 0 . 0 7$  $1 . 1 6 { \pm } 0 . 0 4$ </td></tr><tr><td rowspan=1 colspan=1> $9 . 2 6 { \pm } 0 . 5 2 $ </td><td rowspan=1 colspan=1>1.16</td></tr><tr><td rowspan=1 colspan=1> $9 . 1 3 { \pm } 0 . 7 3 $ </td><td rowspan=1 colspan=1> $1 . 1 2 { \pm } 0 . 0 6$ </td></tr><tr><td rowspan=1 colspan=1> $7 4 . 9 0 { \pm } 0 . 6 2 $ </td><td rowspan=1 colspan=1> $8 2 . 7 4 { \pm } 0 . 6 3$ </td><td rowspan=1 colspan=1> $1 0 . 9 3 { \pm } 0 . 6 7$ </td><td rowspan=1 colspan=1> $1 . 3 7 { \pm } 0 . 0 8$ </td></tr><tr><td rowspan=2 colspan=1> $\overline { { \mathrm { N A - U N E T R ~ ( n o - d i l . ) } } }$  $\mathrm { N A - U N E T R }$ (ours)</td><td rowspan=1 colspan=1> $7 8 . 7 4 2 0 . 2 9$ </td><td rowspan=1 colspan=1> $\overline { { 8 6 . 1 3 { \pm 0 . 3 6 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 9 . 0 5 \pm 0 . 4 4 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 . 1 4 \pm 0 . 0 4 } }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathbf { 7 9 . 4 9 { \pm } 0 . 2 5 }$ </td><td rowspan=1 colspan=1> $\mathbf { 8 6 . 8 8 { \pm } 0 . 3 2 }$ </td><td rowspan=1 colspan=1> $\mathbf { 8 . 8 9 } \pm \mathbf { 0 . 3 0 }$ </td><td rowspan=1 colspan=1> $\mathbf { 1 . 0 2 \pm 0 . 0 3 }$ </td></tr></table>

and Transformer baselines. LAD-SEG serves as the primary evaluation under challenging non-contrast conditions, with ImageCAS included to demonstrate performance on a larger, high-contrast coronary dataset.

Axial

Coronal

Sagittal

On LAD-SEG, CNN-based models achieved limited accuracy, with Dice scores between 35–43%, HD95 values often above 40 mm, and ASD greater than 10 mm. Transformer-based methods performed better, with Swin UNETR reaching 44.78% Dice and SwinUNETR-V2 achieving an ASD of 10.18 mm. NA-UNETR obtained the strongest overall results with 45.64% Dice, 44.39% clDice, 38.16 mm HD95, and 10.01 mm ASD, representing improvements in both volumetric overlap and centerline topology. Relative to its no-dilation variant, NA-UNETR achieved higher Dice and clDice and reduced HD95, indicating that dilated NA blocks help capture longer vessel

![](images/333c72d22b206d449b03efcfcedb8d663e79b791176bc9c289aee9bc6f982288.jpg)  
Figure 6: Representative axial, coronal, and sagittal views with zoomed-in ROI insets (yellow boxes) highlighting the segmented LAD region across Ground Truth, NA-UNETR, and nnU-Net predictions.

trajectories. These gains correspond to approximately a 3% Dice improvement over nnU-Net and a 3 mm reduction in HD95 relative to Swin UNETR. Mannˆa€“Whitney U testing indicated that the diferences were not statistically significant $( p > 0 . 0 5 )$ , which is expected given the small sample size (n = 20) and the substantial inter-patient variability in LAD-SEG.

On ImageCAS, CNN-based models performed strongly, with UNet++ achieving 78.57% Dice, 84.71% clDice, 9.46 mm HD95, and 1.15 mm ASD. Transformer-based approaches such as SwinUNETR-V2 reached 78.03% Dice, 85.61% clDice, and 1.12 mm ASD. NA-UNETR surpassed both groups, achieving 79.49% Dice, 86.88% clDice, 8.89 mm HD95, and 1.02 mm ASD. Compared with its no-dilation variant, NA-UNETR showed modest but consistent gains across Dice, clDice, HD95, and ASD, indicating improved robustness even under highcontrast imaging. These gains correspond to a 1.2% Dice improvement over UNet++, a 4% reduction in HD95 compared with Swin UNETR, and roughly a 9% reduction in ASD relative to the next best model, together with the highest centerline accuracy. Mannˆa€“Whitney U testing confirmed that NA-UNETRˆa€™s improvements on ImageCAS were statistically significant $\left( p < 0 . 0 5 \right)$

## IV.C. Ablation Studies

Table 3: Ablation studies: various architecture settings.
<table><tr><td>NAT Blocks Per Stage</td><td>Residual Block</td><td>Kernel Size</td><td>DSC (%)↑</td><td>HD95 (mm)↓</td><td>ASD (mm)↓</td></tr><tr><td>(3, 4, 6, 18, 5)</td><td>√</td><td>variable</td><td>45.64</td><td>38.16</td><td>10.01</td></tr><tr><td>(2, 3, 4, 18, 5)</td><td>√</td><td>variable</td><td>44.61</td><td>41.63</td><td>11.10</td></tr><tr><td>(2, 2, 2, 2, 2)</td><td>√</td><td>variable</td><td>43.26</td><td>40.60</td><td>10.93</td></tr><tr><td>(3, 4, 6, 18, 5)</td><td>√</td><td>static</td><td>43.17</td><td>40.20</td><td>10.84</td></tr><tr><td>(3, 4, 6, 18, 5)</td><td>X</td><td>variable</td><td>43.01</td><td>42.08</td><td>11.43</td></tr><tr><td>(3, 4, 6, 18, 5)</td><td>X</td><td>static</td><td>42.28</td><td>40.67</td><td>10.80</td></tr></table>

Architecture Ablations. Ablation results in Table 3 show the efect of several key design choices in NA-UNETR. The default configuration uses a non-uniform NAT depth of (3, 4, 6, 18, 5) with residual blocks preceding each NAT block and variable kernel sizes across stages. Reducing the NAT depth to (2, 3, 4, 18, 5), as proposed in the original Neighborhood Attention work <sup>38</sup>, lowered DSC by approximately 1.0%. A uniform allocation of (2, 2, 2, 2, 2), similar to Swin UNETR<sup>48</sup>, produced an even larger reduction of about 2.4%. Removing residual blocks further decreased DSC from 45.64% to 43.01% and increased ASD, confirming their role in stabilizing local feature extraction. Using a fixed $3 \times 3 \times 3$ kernel also reduced DSC by roughly 2.5%, indicating that variable kernels better accommodate stage-specific

receptive field requirements.

Table 4: Ablation study on LoRA rank r.
<table><tr><td>Rank r</td><td>DSC (%)↑</td><td>HD95 (mm)↓</td><td>ASD (mm)↓</td></tr><tr><td>2</td><td>44.32</td><td>41.37</td><td>11.02</td></tr><tr><td>4</td><td>45.18</td><td>40.32</td><td>10.41</td></tr><tr><td>8</td><td>45.64</td><td>38.16</td><td>10.01</td></tr><tr><td>16</td><td>45.41</td><td>40.08</td><td>10.22</td></tr></table>

LoRA Rank Sensitivity. We evaluate LoRA rank $r \in \{ 2 , 4 , 8 , 1 6 \}$ while keeping all other settings fixed. As shown in Table 4, performance improves from r = 2 to r = 8, with higher DSC and lower HD95 and ASD. The results suggest that increasing the rank improves the model’s adaptation capacity for capturing fine-grained vessel structures, whereas too small a rank limits expressiveness. Increasing the rank beyond 8 does not yield further improvement and slightly degrades performance, indicating diminishing returns. Therefore, $r \ = \ 8$ is adopted in all experiments.

Table 5: Ablation on training strategy components.
<table><tr><td>Variant</td><td>DSC↑</td><td>HD95 (mm)↓</td><td>ASD (mm)↓</td></tr><tr><td>NA-UNETR (default)</td><td>45.64</td><td>38.16</td><td>10.01</td></tr><tr><td>Train on LAD-SEG only</td><td>36.39</td><td>40.72</td><td>10.90</td></tr><tr><td>Static loss weights</td><td>44.08</td><td>42.19</td><td>11.35</td></tr><tr><td>No Focal Loss</td><td>44.01</td><td>41.70</td><td>10.88</td></tr><tr><td>Segmentation loss only</td><td>43.47</td><td>42.49</td><td>11.93</td></tr><tr><td>Standard preprocessing only</td><td>43.12</td><td>40.91</td><td>11.44</td></tr><tr><td>No postprocessing</td><td>44.41</td><td>40.74</td><td>10.47</td></tr><tr><td>nnU-Net (standard preprocessing)</td><td>42.54</td><td>39.68</td><td>10.37</td></tr><tr><td>Standard Preprocessing Only</td><td>39.98</td><td>41.22</td><td>11.50</td></tr></table>

Training Strategy Ablations. Ablation results in Table 5 show that the two stage approach of pretraining on ImageCAS followed by fine tuning on LAD-SEG produced the strongest performance for NA-UNETR. Training on LAD-SEG alone reduced NA-UNETR DSC $( 4 5 . 6 4 \%  3 6 . 3 9 \% )$ , showing that pretraining provides a more stable initialization for thin vessel learning. Using fixed loss weights instead of dynamic weighting decreased DSC and increased boundary errors, and removing the boundary loss led to further degradation, indicating that both components contribute to balancing overlap and contour accuracy. Removing the focal term (Dice + Hausdorf) reduced DSC, and removing postprocessing (largest component, small object removal, hole filling) caused a modest drop, confirming these components provide incremental but non-dominant gains. Standard preprocessing reduced DSC and worsened boundary metrics for both NA-UNETR (45.64% → 43.12%) and nnU-Net $( 4 2 . 5 4 \%  3 9 . 9 8 \% )$ , with similar reductions of 2.52 and 2.56 percentage points respectively, indicating that the tailored pipeline provides consistent incremental benefits across architectures. Here, “standard preprocessing” denotes intensity clipping and linear scaling only.

## IV.D. Computational Cost Analysis

Table 6: Computational cost analysis, NA-UNETR vs Benchmark Models.
<table><tr><td>Method</td><td>Training (s/epoch)</td><td>Trainable Params. (M)</td><td>FLOPS (B)</td><td>Inference (S)</td><td>Peak VRAM (GB)</td></tr><tr><td>nnU-Net 23</td><td>17.25</td><td>8.05</td><td>399.59</td><td>0.50</td><td>2.78</td></tr><tr><td> $\mathrm { M e d N e X t ^ { 4 7 } }$ </td><td>25.98</td><td>3.91</td><td>121.17</td><td>4.68</td><td>4.91</td></tr><tr><td>UNETR 39</td><td>18.13</td><td>20.13</td><td>480.87</td><td>1.17</td><td>3.91</td></tr><tr><td>Swin UNETR 48</td><td>22.21</td><td>19.71</td><td>300.00</td><td>2.13</td><td>5.49</td></tr><tr><td>NA-UNETR</td><td>18.77</td><td>19.60</td><td>314.10</td><td>1.33</td><td>4.17</td></tr></table>

Table 6 summarizes the computational requirements of NA-UNETR relative to CNNand Transformer-based baselines. CNN models such as nnU-Net and MedNeXt generally involve fewer parameters or lower FLOPs, whereas Transformer architectures incur higher complexity. NA-UNETR has 19.6M trainable parameters and 314.1B FLOPs, comparable to Swin UNETR (19.7M, 300.0B) and notably lower than UNETR (20.1M, 480.9B). In addition, NA-UNETR achieves competitive inference latency (1.33 s per volume) with moderate peak GPU memory usage (4.17 GB), remaining more memory-eficient than Swin UNETR while maintaining strong segmentation accuracy. Training time per epoch is also comparable across models, with NA-UNETR (18.77 s) remaining close to nnU-Net (17.25 s) and UNETR (18.13 s), indicating no additional training overhead.

## V. Discussion

Accurately delineating the LAD on non-contrast CT is dificult because the vessel often blends with adjacent structures, reducing the visibility of its boundaries. Prior studies have shown that manual LAD annotations on non-contrast CT exhibit substantial inter-observer variability, with reported Dice values ranging from 0.10 to 0.53 <sup>17,18,19,20,21</sup>, and automated methods typically achieving 0.09–0.30, often requiring manual correction <sup>17,19,20,21</sup>. In this setting, NA-UNETR achieved a mean DSC of 45.6% on LAD-SEG, which compares favorably to prior reports and to the baselines evaluated in this study. The tailored preprocessing and training pipeline improved performance in the tested control settings. Under this common pipeline, NA-UNETR achieved the highest overall accuracy, consistent with a benefit from its integration of local and global context through Neighborhood Attention.

Pretraining on CTA provides a clear visualization of the coronary tree and allows the encoder to learn stable vessel features that form a useful foundation for subsequent finetuning. Although CTA difers from non-contrast CT, the decoder and the LoRA-adapted MLP layers help mitigate the modality shift and enable the model to transfer structural information while adapting to the target domain. Transformer-based models benefited from fine-tuning the MLP encoder layers together with the decoder, enabling better adaptation to non-contrast CT and yielding higher DSC than the CNN-based baselines Table 1, while clDice was less influenced since topological continuity can be maintained without additional encoder fine-tuning. Future work will incorporate explicit domain-adaptation or modalityalignment strategies to further reduce the CTA–non-contrast CT domain gap.

NA-UNETR showed favorable generalization by outperforming baselines on our LAD-SEG dataset and achieving competitive performance on the public artery dataset when trained directly on that benchmark. Across both datasets, NA-UNETR achieved stronger boundary-based metrics (HD95, ASD) in addition to higher Dice-based performance (DSC, clDice). Although boundary accuracy is particularly important for LAD segmentation due to the vessel’s small caliber and thin, elongated course, prior LAD-focused studies <sup>17,19,20,21</sup> did not report boundary-based metrics, limiting their assessment of contour quality and topological continuity.

Despite these gains, boundary-based metrics remain challenging on LAD-SEG. The thin vessel caliber and weak boundaries often produce small, fragmented predictions that inflate HD95 and ASD even when the centerline is captured accurately, a pattern consistent across all models reflecting the intrinsic visibility limits of the LAD on a small non-contrast CT dataset. NA-UNETR achieved substantially better HD95 and ASD on ImageCAS, where the CCTA images provide clearer vessel definition, highlighting the strong dependence of boundary accuracy on imaging modality. On LAD-SEG, given that reported inter-observer Dice scores range from 0.10 to 0.53 <sup>18</sup>, the achieved DSC lies within expert variability, suggesting potential utility in reducing manual editing efort and improving consistency for small, elongated vascular structures. Prospective workflow studies evaluating time savings, edit burden, and user acceptance will be required to further establish its clinical impact.

Visual inspection in Figure 5 and Figure 6 showed that most errors arose when the vessel narrowed to only a few voxels or when local contrast was very weak, leading to small gaps or minor boundary ofsets in the predictions. Although NA-UNETR shows improvements over baseline methods across multiple metrics, LAD segmentation on non-contrast CT remains fundamentally limited by the visibility of the anatomy, and model performance is ultimately bounded by the information available in the underlying images. These results represent a step forward, but the approach is not yet suitable for clinical deployment. Further refinement, larger annotated datasets, and validation across multiple institutions will be essential to establish the reliability and robustness required for routine clinical use.

## VI. Conclusion

The LAD artery’s small caliber, poor soft-tissue contrast, and limited annotated data make it one of the most challenging targets for automated segmentation in non-contrast CT. This work addresses these dificulties through neighborhood attention for geometrically coherent local-global context modeling and uncertainty-guided loss balancing for precise boundary delineation. Generalizability to the small-data regime is further strengthened through pretraining on a broader coronary dataset followed by parameter-eficient fine-tuning. Together, these contributions advance the feasibility of automated LAD delineation and lay groundwork for its integration into cardiac dose assessment workflows in thoracic radiotherapy planning.

## Acknowledgments

This work was supported in part by a grant from Varian Medical Systems (Siemens Healthineers).

## Conflict of Interest

The authors declare that they have no conflicts of interest.

## Data Availability Statement

Code has been made available at https://github.com/rafiibnsultan/NA\_UNETR.

## References

G. Nilsson, L. Holmberg, H. Garmo, O. Duvernoy, I. Sj¨ogren, B. Lagerqvist, and C. Blomqvist, Distribution of coronary artery stenosis after radiation for breast cancer, Journal of clinical oncology 30, 380–386 (2012).

C. R. Correa, H. I. Litt, W.-T. Hwang, V. A. Ferrari, L. J. Solin, and E. E. Harris, Coronary artery findings after left-sided compared with right-sided radiation treatment for early-stage breast cancer, Journal of clinical oncology 25, 3031–3037 (2007).

3 S. C. Darby et al., Risk of ischemic heart disease in women after radiotherapy for breast cancer, New England Journal of Medicine 368, 987–998 (2013).

I. Rehman, C. C. Kerndt, and A. Rehman, Anatomy, thorax, heart left anterior descending (LAD) artery, in StatPearls [Internet], StatPearls Publishing, 2023.

5 S. Patel, S. Mahmood, T. Nguyen, B. Yeap, R. Jimenez, N. Meyersohn, T. Neilan, and S. MacDonald, Comparing whole heart versus coronary artery dosimetry in predicting the risk of cardiac toxicity following breast radiation therapy, International Journal of Radiation Oncology, Biology, Physics 102, S46 (2018).

6 K. M. Atkins, D. S. Bitterman, T. L. Chaunzwa, D. E. Kozono, E. H. Baldini, H. J. Aerts, B. K. Tamarappoo, U. Hofmann, A. Nohria, and R. H. Mak, Mean heart dose is an inadequate surrogate for left anterior descending coronary artery dose and the risk of major adverse cardiac events in lung cancer radiation therapy, International Journal of Radiation Oncology\* Biology\* Physics 110, 1473–1479 (2021).

J. Song, T. Tang, J.-M. Caudrelier, J. B´elec, J. Chan, P. Lacasse, G. Aldosary, and V. Nair, Dose-sparing efect of deep inspiration breath hold technique on coronary artery and left ventricle segments in treatment of breast cancer, Radiotherapy and Oncology 154, 101–109 (2021).

8 V. A. van den Bogaard, D. S. Spoor, A. van der Schaaf, L. V. van Dijk, E. Schuit, N. M. Sijtsema, J. A. Langendijk, J. H. Maduro, and A. P. Crijns, The importance of radiation dose to the atherosclerotic plaque in the left anterior descending coronary artery for radiation-induced cardiac toxicity of breast cancer patients?, International Journal of Radiation Oncology\* Biology\* Physics 110, 1350–1359 (2021).

E. D. Morris, A. I. Ghanem, M. Dong, M. V. Pantelic, E. M. Walker, and C. K. Glide-Hurst, Cardiac substructure segmentation with deep learning for improved cardiac sparing, Medical physics 47, 576–586 (2020).

10 C. Nieder, S. Schill, P. Kneschaurek, and M. Molls, Influence of diferent treatment techniques on radiation dose to the LAD coronary artery, Radiation Oncology 2, 1–7 (2007).

11 V. A. van den Bogaard et al., Validation and modification of a prediction model for acute cardiac events in patients with breast cancer treated with radiotherapy based on threedimensional dose distributions to cardiac substructures, Journal of Clinical Oncology 35, 1171–1178 (2017).

<sup>12</sup> S. Vivekanandan et al., The impact of cardiac radiation dosimetry on survival after radiation therapy for non-small cell lung cancer, International Journal of Radiation Oncology\* Biology\* Physics 99, 51–60 (2017).

13 Z. Wang, G. Chen, D. Song, X. Xu, C. Chu, S. Zhang, H. Chai, H. Yu, X. Luan, and P. Song, Reduced contrast agent volume using a heart-rate dependent and free-breathing scanning protocol in coronary computed tomography angiography (CTA) for patients with chronic obstructive pulmonary disease (COPD), BMC Cardiovascular Disorders 25, 15 (2025).

14 E. Nicolas, N. Khalifa, C. Laporte, S. Bouhroum, and Y. Kirova, Safety margins for the delineation of the left anterior descending artery in patients treated for breast cancer, International Journal of Radiation Oncology\* Biology\* Physics 109, 267–272 (2021).

15 P. Sakyanun, K. Saksornchai, C. Nantavithya, C. Chakkabat, and K. Shotelersuk, The efect of deep inspiration breath-hold technique on left anterior descending coronary

artery and heart dose in left breast irradiation, Radiation Oncology Journal 38, 181 (2020).

16 C. Zhou, H.-P. Chan, A. Chughtai, J. Kuriakose, P. Agarwal, E. A. Kazerooni, L. M. Hadjiiski, S. Patel, and J. Wei, Computerized analysis of coronary artery disease: performance evaluation of segmentation and tracking of coronary arteries in CT angiograms, Medical Physics 41, 081912 (2014).

17 V. A. van den Bogaard, L. V. van Dijk, R. Vliegenthart, N. M. Sijtsema, J. A. Langendijk, J. H. Maduro, and A. P. Crijns, Development and evaluation of an autosegmentation tool for the left anterior descending coronary artery of breast cancer patients based on anatomical landmarks, Radiotherapy and Oncology 136, 15–20 (2019).

<sup>18</sup> F. Duane et al., A cardiac contouring atlas for radiotherapy, Radiotherapy and Oncology 122, 416–422 (2017).

19 R. Kaderka et al., Geometric and dosimetric evaluation of atlas based auto-segmentation of cardiac structures in breast cancer patients, Radiotherapy and oncology 131, 215–220 (2019).

20 E. D. Morris, A. I. Ghanem, M. V. Pantelic, E. M. Walker, X. Han, and C. K. Glide-Hurst, Cardiac substructure segmentation and dosimetry using a novel hybrid magnetic resonance and computed tomography cardiac atlas, International Journal of Radiation Oncology\* Biology\* Physics 103, 985–993 (2019).

<sup>21</sup> R. Zhou et al., Cardiac atlas development and validation for automatic segmentation of cardiac substructures, Radiotherapy and Oncology 122, 66–71 (2017).

22 O. Ronneberger, P. Fischer, and T. Brox, U-net: Convolutional networks for biomedical image segmentation, in Medical Image Computing and Computer-Assisted Intervention– MICCAI 2015, pages 234–241, Springer, 2015.

23 F. Isensee, P. F. Jaeger, S. A. Kohl, J. Petersen, and K. H. Maier-Hein, nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation, Nature methods 18, 203–211 (2021).

24 M. Saha, J. W. Jung, S.-W. Lee, C. Lee, C. Lee, and M. M. Mille, A deep learning segmentation method to assess dose to organs at risk during breast radiotherapy, Physics and Imaging in Radiation Oncology 28, 100520 (2023).

25 X. Jin, M. A. Thomas, J. Dise, J. Kavanaugh, J. Hilliard, I. Zoberi, C. G. Robinson, and G. D. Hugo, Robustness of deep learning segmentation of cardiac substructures in noncontrast computed tomography for breast cancer radiotherapy, Medical physics 48, 7172–7188 (2021).

26 N. Summerfield, E. Morris, S. Banerjee, Q. He, A. I. Ghanem, S. Zhu, J. Zhao, M. Dong, and C. Glide-Hurst, Enhancing Precision in Cardiac Segmentation for Magnetic Resonance-Guided Radiation Therapy Through Deep Learning, International Journal of Radiation Oncology\* Biology\* Physics 120, 904–914 (2024).

27 C. Li, H. Zhu, R. Ibn Sultan, H. B. Ebadian, P. Khanduri, C. Indrin, K. Thind, and D. Zhu, MulModSeg: Enhancing Unpaired Multi-Modal Medical Image Segmentation with Modality-Conditioned Text Embedding and Alternating Training, in Proceedings of the Winter Conference on Applications of Computer Vision (WACV), 2025.

28 S¸. Kaba, H. Haci, A. Isin, A. Ilhan, and C. Conkbayir, The application of deep learning for the segmentation and classification of coronary arteries, Diagnostics 13, 2274 (2023).

29 L.-S. Pan, C.-W. Li, S.-F. Su, S.-Y. Tay, Q.-V. Tran, and W. P. Chan, Coronary artery segmentation under class imbalance using a U-Net based architecture on computed tomography angiography images, Scientific reports 11, 14493 (2021).

30 A. Song, L. Xu, L. Wang, B. Wang, X. Yang, B. Xu, B. Yang, and S. E. Greenwald, Automatic coronary artery segmentation of CCTA images with an eficient featurefusion-and-rectification 3D-UNet, IEEE journal of biomedical and health informatics 26, 4044–4055 (2022).

31 C. Dong, S. Xu, D. Dai, Y. Zhang, C. Zhang, and Z. Li, A novel multi-attention, multiscale 3D deep network for coronary artery segmentation, Medical Image Analysis 85, 102745 (2023).

Y. Shen, Z. Fang, Y. Gao, N. Xiong, C. Zhong, and X. Tang, Coronary arteries segmentation based on 3D FCN with attention gate and level set function, Ieee Access 7, 42826–42835 (2019).

33 K. Chachadi, S. Nirmala, and P. G. Netrakar, Automated Coronary Artery Segmentation with 3D PSPNET using Global Processing and Patch Based Methods on CCTA Images, Cardiovascular Engineering and Technology , 1–15 (2025).

<sup>34</sup> J. Park et al., Selective ensemble methods for deep learning segmentation of major vessels in invasive coronary angiography, Medical physics 50, 7822–7839 (2023).

35 C. Li, Y. Qiang, R. I. Sultan, H. Bagher-Ebadian, P. Khanduri, I. J. Chetty, and D. Zhu, FocalUNETR: A Focal Transformer for Boundary-Aware Prostate Segmentation Using CT Images, in International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 592–602, Springer, 2023.

36 R. I. Sultan, H. Zhu, C. Li, and D. Zhu, BIPVL-SEG: bidirectional progressive visionlanguage fusion with global-local alignment for medical image segmentation, arXiv preprint arXiv:2503.23534 (2025).

37 U. Mgboh, R. I. Sultan, J. Kim, K. Thind, and D. Zhu, Fluenceformer: transformerdriven multi-beam fluence map regression for radiotherapy planning, arXiv preprint arXiv:2512.22425 (2025).

38 A. Hassani, S. Walton, J. Li, S. Li, and H. Shi, Neighborhood attention transformer, in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6185–6194, 2023.

39 A. Hatamizadeh, Y. Tang, V. Nath, D. Yang, A. Myronenko, B. Landman, H. R. Roth, and D. Xu, Unetr: Transformers for 3d medical image segmentation, in Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 574–584, 2022.

40 A. Zeng et al., ImageCAS: A large-scale dataset and benchmark for coronary artery segmentation based on computed tomography angiography images, Computerized Medical Imaging and Graphics 109, 102287 (2023).

41 A. Kendall, Y. Gal, and R. Cipolla, Multi-task learning using uncertainty to weigh losses for scene geometry and semantics, in Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7482–7491, 2018.

42 F. Milletari, N. Navab, and S.-A. Ahmadi, V-net: Fully convolutional neural networks for volumetric medical image segmentation, in 2016 fourth international conference on 3D vision (3DV), pages 565–571, Ieee, 2016.

43 T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Doll´ar, Focal loss for dense object detection, in Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017.

44 D. Karimi and S. E. Salcudean, Reducing the hausdorf distance in medical image segmentation with convolutional neural networks, IEEE Transactions on medical imaging 39, 499–513 (2019).

45 E. J. Hu, yelong shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, LoRA: Low-Rank Adaptation of Large Language Models, in International Conference on Learning Representations, 2022.

46 Z. Zhou, M. M. Rahman Siddiquee, N. Tajbakhsh, and J. Liang, Unet++: A nested u-net architecture for medical image segmentation, in Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support: 4th International Workshop, DLMIA 2018, and 8th International Workshop, ML-CDS 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 20, 2018, Proceedings 4, pages 3–11, Springer, 2018.

47 S. Roy, G. Koehler, C. Ulrich, M. Baumgartner, J. Petersen, F. Isensee, P. F. Jaeger, and K. H. Maier-Hein, Mednext: transformer-driven scaling of convnets for medical image segmentation, in International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 405–415, Springer, 2023.

48 A. Hatamizadeh, V. Nath, Y. Tang, D. Yang, H. R. Roth, and D. Xu, Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images, in International MICCAI Brainlesion Workshop, pages 272–284, Springer, 2021.

49 Y. He, V. Nath, D. Yang, Y. Tang, A. Myronenko, and D. Xu, Swinunetr-v2: Stronger swin transformers with stagewise convolutions for 3d medical image segmentation, in International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 416–426, Springer, 2023.

50 H.-Y. Zhou, J. Guo, Y. Zhang, L. Yu, L. Wang, and Y. Yu, nnformer: Interleaved transformer for volumetric segmentation, arXiv preprint arXiv:2109.03201 (2021).