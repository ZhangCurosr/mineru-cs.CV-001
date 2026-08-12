## Highlights

Lesion-Aware Adaptive Fourier Neural Operator for CT-to-PSMA PET Synthesis in Prostate Cancer

Rashmi Bhaskara, Waleed M. Almutairi, Matthew Gopaulchan, Maram Musaad Alqurashi, Francis Asamoah, Alex Ocana, Clinton D. Bahler, Oluwaseyi M. Oderinde

• We propose LAFNO for lesion-aware CT-to-PSMA PET synthesis in prostate cancer.

• CT-derived contrast and disorder proxy channels are used to condition the AFNO bottleneck.

• Lesion-aware losses improve tumor-core and peritumoral fidelity beyond whole-volume metrics.

# Lesion-Aware Adaptive Fourier Neural Operator for CT-to-PSMA PET Synthesis in Prostate Cancer

Rashmi Bhaskara<sup>a</sup>, Waleed M. Almutairi<sup>a</sup>, Matthew Gopaulchan<sup>a</sup>, Maram Musaad Alqurashi<sup>a</sup>, Francis Asamoah<sup>b</sup>, Alex Ocana<sup>c</sup>, Clinton D. Bahler<sup>d</sup> and Oluwaseyi M. Oderinde<sup>a,b,∗</sup>

<sup>a</sup>School ofHealth Sciences, Purdue University, West Lafayette, 47907, IN, USA

<sup>b</sup>Department of Radiology, Indiana University School of Medicine, Indianapolis, 46202, IN, USA

<sup>c</sup>Department ofUrology, Indiana University School ofMedicine, Indianapolis, 46202, IN, USA

<sup>d</sup>Department of Radiation Oncology, Indiana University School of Medicine, Indianapolis, 46202, IN, USA

## A R T I C L E I N F O

Keywords:   
PSMA PET   
Synthetic PET   
Prostate cancer   
Lesion-aware learning   
Adaptive Fourier neural operator   
Tumor microenvironment   
Radiomics

## A BS T RA C T

Deep learning models that synthesize PET from CT or MRI can reduce patient dose and scanner demand, but are typically optimized with global losses such as L1 or mean squared error (MSE) that treat all voxels similarly. In whole-body PSMA-PET, tumor voxels occupy only a small fraction of the volume, yet carry the clinically relevant activity signal; as a result, models can achieve high structural similarity index measure (SSIM) and peak signal-to-noise ratio (PSNR) while still underestimating lesion activity or failing to preserve tumor-specific structure. Radiomics provides biologically meaningful descriptors of tumor intensity and texture, but direct radiomics conditioning is time-consuming because it requires feature extraction from delineated lesion regions. We propose LAFNO, a Lesion-Aware Adaptive Fourier Neural Operator for CT-to-PSMA-PET synthesis that replaces high-dimensional radiomics conditioning with two eficient CT-derived proxy channels. Motivated by radiomics analysis of PSMA-avid tumorcore and peritumoral regions, LAFNO uses a contrast proxy for local density variation and a disorder proxy for local texture heterogeneity, both injected into the model bottleneck. LAFNO combines whole-volume reconstruction with lesion-level total lesion activity (TLA), tumor-core contrast, and peritumoral supervision. We evaluated LAFNO against four baseline architectures on the TCIA PSMA-PET-CT-Lesions dataset. LAFNO remained competitive on whole-volume image quality, achieving SSIM of 0.960 and 0.938 for <sup>18</sup>F- and <sup>68</sup>Ga-PSMA, respectively, while reducing per-patient TLA error to 48.3% and 64.0% for <sup>18</sup>F- and <sup>68</sup>Ga-PSMA, respectively, and achieving the highest tumor-core radiomics reproducibility across all feature classes for both tracers. Peritumoral reproducibility remained tracer-dependent, indicating that biological fidelity in synthetic PSMA-PET remains challenging.

## 1. Introduction

Prostate cancer (PCa) is the most diagnosed male malignancy in Western countries Maurer et al. (2016). In the United States, it accounts for 31% of all male cancer cases, with more than 35,000 deaths projected in 2026 Jemal (2026). Despite major therapeutic advances, many patients, particularly those with high-risk disease, still experience recurrence or progression Bouchelouche et al. (2016); Tonry et al. (2020). Accurate imaging therefore plays a critical role in diagnosis, staging, restaging, biochemical recurrence detection, and treatment planning Bouchelouche et al. (2016).

Prostate-specific membrane antigen (PSMA) is a transmembrane glycoprotein expressed by prostatic epithelial cells, with higher expression in malignant prostate tissue than in benign or normal tissue. PSMA expression increases with Gleason grade and tumor stage, making it an important biomarker in both hormone-sensitive and castration-resistant prostate cancer Kim et al. (2023). Because biochemical changes often precede anatomical changes Reddy and Robinson (2010), conventional imaging modalities such as computed tomography (CT) and magnetic resonance imaging (MRI) can be limited for early detection and disease characterization. PSMA PET/CT has therefore become increasingly important because it provides improved sensitivity for staging, restaging, and detection of metastatic disease compared with conventional imaging Siva et al. (2020); Tsechelidis and Vrachimis (2022); Caldarella et al. (2024); Dondi et al. (2024); Ishibashi-Kanno et al. (2023).

Despite its diagnostic value, PSMA PET/CT remains expensive and unevenly accessible Subramanian et al. (2023); Smith and Harper (2025). Limited tracer availability, scanner demand, insurance coverage, and cost can restrict access, particularly in underserved populations. These barriers are clinically important because delayed or incomplete imaging may afect staging accuracy, recurrence detection, and treatment planning Subramanian et al. (2023). Therefore, alternative imaging strategies that reduce dependence on radiotracer administration while preserving clinically relevant tumor information could improve accessibility and broaden the clinical impact of PSMA-targeted imaging.

Recent advances in machine learning and deep learning have enabled cross-modality medical image synthesis, including PET synthesis from anatomical imaging Dayarathna et al. (2024). Convolutional neural networks, especially U-Net-based encoder–decoder architectures, have been widely used for medical image-to-image translation because they preserve spatial structure through skip connections Ronneberger et al. (2015); Emami et al. (2020). Generative models, including GANs, Pix2Pix, CycleGAN, and difusion models, have also been explored for multimodal medical image synthesis Ghafari et al. (2022); Chai et al. (2025); Jung et al. (2024).

However, CT-to-PET synthesis in prostate cancer remains challenging because CT has limited soft-tissue contrast, particularly for small structures and subtle lesions Abdollahi et al. (2026). These lesions occupy only a tiny fraction of the voxels in a whole-body scan Abtahi et al. (2026). As a result, models trained with global reconstruction objectives and evaluated mainly using whole-volume metrics such as structural similarity index measure (SSIM), peak signal-to-noise ratio (PSNR), and mean absolute error (MAE) can become biased toward healthy background tissue and may omit or underestimate small but clinically important diagnostic hotspots Abtahi et al. (2026). In addition, the mapping from CT to PET is fundamentally one-to-many: multiple plausible uptake patterns may correspond to the same CT appearance, especially for subtle lesions and physiologic tracer activity Mahdi et al. (2026). These limitations highlight the need for synthetic PET models that move beyond global image similarity and incorporate tumor-relevant biological and anatomical signatures.

Radiomics provides a useful bridge between anatomical imaging and tumor biology because it converts standard medical images into high-dimensional, mineable quantitative features that can describe tumor intensity, shape, and texture Lambin et al. (2012); Gillies et al. (2016). These features have been explored as non-invasive imaging biomarkers across multiple cancers, including lung cancer, glioblastoma, and prostate cancer Lambin et al. (2012); Gillies et al. (2016); Almutairi et al. (2026). Recent work has also shown that radiomics can be used as a conditioning signal for tumor synthesis. Kim et al. proposed a radiomics-conditioned tumor-generation framework that uses a GAN-based model to generate tumor masks and a difusion-based model to generate tumor texture conditioned on user-specified radiomics features such as size, shape, and texture Kim et al. (2025). This supports the idea that radiomics features can provide biologically meaningful conditioning for generative medical imaging models. However, conventional radiomics extraction can be time-consuming because it requires computing numerous handcrafted features from delineated lesion regions. The extraction also depends on accurate lesion delineation, and manual or semi-automated segmentation can be time-consuming, operator-dependent, and sensitive to inter-observer variability Traverso et al. (2018); Gillies et al. (2016).

Building on this idea, we present a Lesion-aware Adaptive Fourier Neural Operator (LAFNO) for CT-to-PSMA-PET synthesis. Rather than using radiomics features directly as model inputs, we use radiomics analysis as a discovery step to identify repeatable CT patterns across all annotated PSMA-avid lesions in The Cancer Imaging Archive (TCIA) PSMA-PET-CT-Lesions dataset Jeblick et al. (2026). We then replace high-dimensional radiomics conditioning with two computationally simple CT-derived proxy channels that approximate these patterns: a contrast proxy for local density variation and a disorder proxy for local texture heterogeneity. These proxies are used to condition our model. In this way, LAFNO preserves the biological motivation of radiomics conditioning while avoiding radiomics extraction and lesion segmentation during inference.

Importantly, these proxy patterns are not defined only from the tumor core. Our analysis also includes the peritumoral region because tumor progression and therapeutic resistance are increasingly understood to depend not only on tumor cell biology, but also on continuous signaling with the surrounding tumor microenvironment (TME), including immune cells, cancer-associated fibroblasts, stromal cells, vasculature, and secreted signaling molecules Wang et al. (2021); Liu et al. (2025). The peritumoral region, the interface between tumor and adjacent normal tissue, remains less studied than the tumor core Koca et al. (2024); Zhang et al. (2023), despite evidence that it can be transcriptionally and phenotypically distinct from both tumor and healthy tissue Koca et al. (2024). In prostate cancer, pronounced intratumoral heterogeneity limits the reliability of single-region assessment Yadav et al. (2018). Incorporating peritumoral radiomics features has been shown to complement intratumoral features and improve diagnostic and classification performance beyond intratumoral analysis alone Algohary et al. (2020); Zhou et al. (2025). Notably, the diagnostic contribution of peritumoral features may depend on their spatial distance from the tumor boundary Liu et al. (2025). Therefore, our radiomics analysis explicitly includes the peritumoral region, and motivates the disorder proxy, which approximates local texture heterogeneity around tumor-adjacent tissue. Together with the contrast proxy, this allows LAFNO to condition PET synthesis on both tumor-core density variation and peritumoral texture patterns.

![](images/09ee7ec14781818ba12e21622185127a7a1564ff8c4bbe613dd7f9099b3da152.jpg)  
Figure 1: CT radiomics profiles from the tumor core to peritumoral shells. (A) firstorder\_Mean decreases outward. (B) firstorder\_Entropy increases from tumor core to peritumoral tissue and remains relatively stable.

In summary, to address the limitations of global CT-to-PET synthesis models, we use radiomics analysis to identify repeatable tumor-core and peritumoral CT patterns and translate these patterns into eficient CT-derived proxy channels. Based on this design, we propose LAFNO, a lesion-aware CT-to-PSMA-PET synthesis framework that combines proxy-conditioned image synthesis with lesion-level and peritumoral supervision. The main contributions are summarized as follows:

• We introduce CT-derived proxy conditioning to guide PET synthesis using radiomics-motivated tumor-core and peritumoral patterns.

• We propose a lesion-aware objective that supervises total lesion activity and penalizes errors in the tumor-adjacent peritumoral region.

• We evaluate synthetic PET beyond whole volume similarity using lesion-level uptake metrics and tumor/peritumoral radiomics reproducibility.

## 2. Materials and Methods

## 2.1. CT Radiomics Analysis of Tumor-Core and Peritumoral Shells

We analyzed whether CT radiomics features vary systematically from the tumor core outward. We studied 10,209 connected-component PSMA-avid tumor regions across 480 patients. For each tumor region derived from the groundtruth mask, radiomics features were extracted independently from the tumor core and the peritumoral shells at increasing distances from the tumor boundary: 0–5 mm, 5–10 mm, and 10–20 mm. Shells were constructed using a Euclidean distance transform from the tumor boundary, with tumor voxels excluded from every shell. Adjacent regions were compared using paired Wilcoxon signed-rank tests with false discovery rate correction, and the analysis was repeated after patient-level aggregation to reduce the influence of patients with many lesions.

Two spatial trends were observed (Figure 1). First, CT attenuation decreased monotonically from the tumor core to the outer peritumoral shells. This was reflected by firstorder\_Mean and firstorder\_Median, which quantify the average and central CT intensity within each region, respectively, and by ngtdm\_Contrast, which measures local gray-level contrast relative to neighboring voxels Zwanenburg et al. (2016). The firstorder\_Mean trend was significant across all adjacent shell comparisons at the lesion level and remained consistent after patient-level aggregation, indicating a robust core-to-periphery attenuation gradient.

Second, CT texture heterogeneity increased when moving from the tumor core into the peritumoral region. This was captured by firstorder\_Entropy, which measures randomness or heterogeneity in the intensity distribution

![](images/5c86645f8cdaedff2a43f1eefd869f163175cc4b504b6d14b27295e8ba481079.jpg)  
Figure 2: Representative CT-derived proxy maps showing the CT image, contrast proxy, and disorder proxy. Zoomed panels highlight the lesion neighborhood.

Zwanenburg et al. (2016). Entropy increased significantly on leaving the tumor core at both lesion and patient levels. At the lesion level, entropy peaked in the 0–5 mm shell, suggesting increased heterogeneity near the tumor boundary. At the patient level, entropy increased from the tumor core to the peritumoral tissue and then remained relatively stable across the outer shell, indicating a tumor-to-peritumoral increase in CT texture heterogeneity.

Together, these findings show a smooth core-to-periphery attenuation gradient and a tumor-to-peritumoral increase in texture. These two CT signatures directly motivate the contrast and disorder proxy channels described in Section 2.2.

## 2.2. Diferentiable CT Proxy Channels

To encode the two CT signatures, we introduce two diferentiable proxy filters, contrast and disorder, that can be computed directly from CT without requiring a ground-truth tumor mask. Representative proxy maps are shown in Figure 2. The contrast proxy approximates the core-to-periphery attenuation gradient using a Gaussian residual filter:

$$
C ( \mathbf { x } ) = \mathrm { C T } ( \mathbf { x } ) - G _ { \sigma } \ast \mathrm { C T } ( \mathbf { x } ) , \qquad \sigma = 5 \mathrm { m m }\tag{1}
$$

where � denotes the voxel location, CT(�) is the CT intensity at that voxel, �(�) is the resulting contrast proxy value, $G _ { \sigma }$ is a 3D Gaussian smoothing kernel with physical scale �, and ∗ denotes convolution. The Gaussian smoothing is implemented using separable 1D convolutions. The scale $\sigma = 5$ mm was chosen to match the spatial scale of the observed firstorder\_Mean decay.

The disorder proxy captures tumor-adjacent tissue heterogeneity via local variance over a sliding window:

$$
D ( \mathbf { x } ) = \overline { { \mathbf { C T } ^ { 2 } } } _ { w } ( \mathbf { x } ) - \left( \overline { { \mathbf { C T } } } _ { w } ( \mathbf { x } ) \right) ^ { 2 } , \qquad w = 7 . 5 \mathrm { m m }\tag{2}
$$

where $D ( \mathbf { x } )$ is the disorder proxy value at voxel �, � is the physical width of the local cubic window, $\overline { { \mathbf { C T } } } _ { w } ( \mathbf { x } )$ is the local mean CT intensity within that window, and $\operatorname { C T } _ { w } ^ { 2 } ( \mathbf { x } )$ is the local mean of squared CT intensities. The local means are implemented using uniform-filter convolution. The window size $w = 7 . 5$ mm was chosen empirically to capture local tumor–peritumoral texture transitions, while avoiding excessive sensitivity to voxel-level noise.

Comparison with the radial radiomics analysis showed that the contrast and disorder proxies capture the observed density and texture trends. Both proxies are computed on the full native-resolution CT volume and used together to highlight tumor-like CT patterns and their interaction with adjacent tissue. In this way, the proxy channels provide the model with an approximate spatial map of regions that may contain tumor-relevant behavior. Their integration into the model is described in Section 2.3.

![](images/b0663faa8560b2f84a15b6487b0a26f59b4979f50f007cf6a74ab57f7ab2fbab.jpg)  
Figure 3: Overview of the proposed LAFNO architecture with CT proxy conditioning and lesion-aware training supervision.

## 2.3. LAFNO Architecture

LAFNO is a 3D U-Net with an Adaptive Fourier Neural Operator (AFNO) bottleneck Guibas et al. (2021). The encoder applies three stride-2 convolution blocks $\mathrm { ( C o n v 3 D + B a t c h N o r m + L e a k y R e L U ) }$ , mapping a $6 4 ^ { 3 }$ CT patch to an $8 ^ { 3 }$ representation with 256 channels. The decoder mirrors the encoder with transposed convolution upsampling and skip connections, ending with a Sigmoid output.

At the $8 ^ { 3 }$ bottleneck, four AFNO blocks perform spectral channel mixing followed by a residual MLP. Before the first AFNO block, the two CT proxy channels are injected. The contrast proxy is average-pooled from $6 4 ^ { 3 }$ to $8 ^ { 3 }$ to preserve its smooth monotonic gradient, while the disorder proxy is max-pooled to preserve its localized peak character. The pooled proxies are concatenated with the 256-channel bottleneck features and projected back to 256 channels:

$$
\mathbf { b } ^ { \prime } = W _ { 2 5 8 \to 2 5 6 } [ \mathbf { b } \parallel \operatorname { A v g P o o l } _ { 8 } ( C ) \parallel \operatorname { M a x P o o l } _ { 8 } ( D ) ]\tag{3}
$$

where � $\in \mathbb { R } ^ { 2 5 6 \times 8 ^ { 3 } }$ are the bottleneck features, $W _ { 2 5 8 \to 2 5 6 }$ is a learned 1×1×1 convolution, and ‖ denotes channel concatenation. The projected features $ { \mathbf { b } } ^ { \prime }$ are then passed through the four AFNO blocks. Figure 3 shows the overall architecture.

## 2.4. Lesion-Aware Loss

The training objective combines whole-volume PET reconstruction with lesion-level activity preservation and peritumoral supervision:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { L 1 } } + \lambda _ { \mathrm { T L A } } \mathcal { L } _ { \mathrm { T L A } } + \lambda _ { c } \mathcal { L } _ { c } + \lambda _ { \mathrm { p e r i } } \mathcal { L } _ { \mathrm { p e r i } }\tag{4}
$$

Here, $\mathcal { L } _ { \mathrm { L 1 } }$ is the standard voxel-wise L1 loss. Because background voxels vastly outnumber lesion voxels, a voxel-wise loss alone tends to under-constrain small, high-uptake regions. The lesion-level term ${ \mathcal { L } } _ { \mathrm { T L A } }$ addresses this by supervising the summed activity of each connected component independently, so that small lesions are not dominated by larger lesions or by surrounding background tissue. For a set of lesions  in a patient volume, the lesion-wise activity loss is defined as

$$
\mathcal { L } _ { \mathrm { T L A } } = \frac { 1 } { \vert \mathcal { K } \vert } \sum _ { k \in \mathcal { K } } \frac { \Bigg \vert \log \left( 1 + \widehat { A } _ { k } \right) - \log \left( 1 + A _ { k } \right) \Bigg \vert } { \log \left( 1 + N _ { k } \cdot S U V _ { \operatorname* { m a x } } \right) } ,\tag{5}
$$

where $N _ { k }$ is the number of voxels in lesion �, $\mathrm { S U V } _ { \operatorname* { m a x } }$ is the global standardized uptake value (SUV) normalization ceiling, and

$$
\widehat { A } _ { k } = \sum _ { i \in k } \widehat { s } _ { i } , \qquad A _ { k } = \sum _ { i \in k } s _ { i } .\tag{6}
$$

Here, $\hat { s } _ { i }$ and $s _ { i }$ are the predicted and ground-truth SUV values within the same lesion mask. The logarithmic transform compresses the wide dynamic range of lesion uptake, and the denominator normalizes each lesion term by its maximum possible summed activity, $N _ { k } \cdot S U V _ { \operatorname* { m a x } }$ . This keeps lesion contributions on a comparable scale and weights lesions by count rather than by absolute activity magnitude. The quantities $\widehat { A } _ { k }$ and $A _ { k }$ are summed SUV values. Physical total lesion activity can be obtained by multiplying by the voxel volume, $\mathrm { T L A } _ { k } = v _ { \mathrm { v o x } } A _ { k } .$ In the loss, we use the summed-SUV form because the predicted and ground-truth activities are computed over the same lesion mask and therefore remain proportional to physical TLA. The objective is minimized when $\widehat { A } _ { k } = A _ { k }$ , which corresponds to matching physical TLA for that lesion.

The remaining two lesion-aware terms supervise tumor-core contrast and peritumoral fidelity. The tumor-contrast loss, $\mathcal { L } _ { c } .$ , applies the contrast operation from Equation 1 to the predicted and ground-truth PET volumes in SUV space and averages the absolute diference over tumor-mask voxels. This penalizes mismatch in local tumor-core uptake structure.

The peritumoral loss, $\mathcal { L } _ { \mathrm { { p e r i } } }$ , is computed in the 0–10 mm ring outside the tumor mask using an exponential distanceweighted L1 penalty. Let $R \hat { ( \mathbf { x } ) }$ denote the binary peritumoral ring mask and let $d ( \mathbf { x } )$ be the distance in millimeters from voxel � to the tumor boundary. The peritumoral weight field is defined as

$$
w ( { \bf x } ) = R ( { \bf x } ) \exp \left( - \frac { d ( { \bf x } ) } { \tau } \right) , \qquad \tau = 5 \mathrm { m m } .\tag{7}
$$

The decay parameter was set to $\tau = 5$ mm, with this value, the weight decreases to half of its boundary value from the tumor boundary. Finally, the peritumoral loss is then computed as

$$
\mathcal { L } _ { \mathrm { p e r i } } = \frac { \sum _ { \mathbf { x } } w ( \mathbf { x } ) \left| \hat { y } ( \mathbf { x } ) - y ( \mathbf { x } ) \right| } { \sum _ { \mathbf { x } } w ( \mathbf { x } ) + \epsilon } , \qquad \epsilon = 1 0 ^ { - 6 } ,\tag{8}
$$

where $\hat { y } ( \mathbf x )$ and �(�) are the predicted and ground-truth PET values. This loss gives higher importance to voxels immediately surrounding the tumor and gradually reduces the penalty for voxels farther from the tumor boundary. As a result, the model is encouraged to preserve PET structure in the tumor-adjacent region without overemphasizing distant background tissue.

Together, these terms encourage the model to preserve lesion activity, tumor-core uptake structure, and peritumoral behavior rather than optimizing whole-volume similarity alone. Loss weights were set to $\lambda _ { \mathrm { L 1 } } = 1 . 0 , \lambda _ { \mathrm { T L A } } = 0 . 0 5$ $\lambda _ { c } = 0 . 0 2$ , and $\lambda _ { \mathrm { p e r i } } = 0 . 0 5$

## 2.5. Training

All models were trained patch-wise on $6 4 ^ { 3 }$ crops for 200 epochs using a single NVIDIA A100-40 GB GPU. We used the Adam optimizer with learning rate $2 { \times } 1 0 ^ { - 4 } , \beta _ { 1 } = 0 . 5 .$ , and $\beta _ { 2 } = 0 . 9 9 9$ , followed by linear learning-rate decay from epoch 50. To address tumor sparsity, 70% of training patches were lesion-centered with random positional jitter, while 30% were sampled uniformly to match the sliding-window inference distribution. Patients were sampled uniformly before patch extraction.

## 3. Experimental Setup and Evaluation

## 3.1. Baselines

We selected one representative baseline from each major CT-to-PET synthesis training strategy: U-Net-style reconstruction, adversarial training, flow matching, and difusion-based generation.

AFNO-L1. AFNO-L1 was used as the direct architectural baseline for LAFNO. It follows the same 3D U-Net-style encoder–decoder design with AFNO blocks inserted at the bottleneck, but receives only the CT patch as input and is trained with whole-volume L1 loss. Unlike LAFNO, it does not use CT-derived proxy conditioning or lesion-aware

![](images/50c1f04d3a2f2c91bf16a50d773b66da97644fcc0d1b035b90db575e897352db.jpg)  
Figure 4: Representative reconstructions from diferent CT-to-PET synthesis models.

supervision. This baseline isolates the efect of the proposed proxy channels and lesion-aware loss while keeping the core architecture fixed.

Pix2Pix. Pix2Pix Isola et al. (2017) was used as a 3D conditional GAN baseline. The generator is a 3D U-Net that maps a 64<sup>3</sup> CT patch to a synthetic PET patch using encoder–decoder layers with skip connections, and the discriminator follows a PatchGAN design.

FlowLet. FlowLet Danese et al. (2026) was used as a flow-matching baseline. The model operates in 3D Haar wavelet space, where CT and PET volumes are decomposed into low- and high-frequency subbands. A U-Net backbone is trained to predict the flow from CT wavelet coeficients to PET wavelet coeficients, conditioned on the CT wavelet

representation and a time embedding. The predicted PET wavelet coeficients are converted back to image space using the inverse wavelet transform.

cWDM. cWDM Friedrich et al. (2024) was used as a conditional difusion baseline. The model also operates in 3D Haar wavelet space and uses a U-Net with attention blocks. CT wavelet coeficients are used as the conditioning input, and the final PET volume is reconstructed from generated PET wavelet coeficients using the inverse wavelet transform.

## 3.2. Dataset and Preprocessing

We used TCIA PSMA-PET-CT-Lesions dataset Jeblick et al. (2026), including <sup>18</sup>F-PSMA (� = 335) and $^ { 6 8 } \mathrm { G a } \cdot$ PSMA $( n = 2 0 4 )$ patients with annotated tumor masks. CT volumes were clipped to [−1000, 1000] HU and linearly scaled to [0, 1]. PET volumes were converted to body-weight standardized uptake value (SUV) and log-normalized as

$$
\mathrm { P E T } ( \mathbf { x } ) = \frac { \log ( 1 + \mathrm { S U V } ( \mathbf { x } ) ) } { \log ( 1 + S U V _ { \operatorname* { m a x } } ) } ,\tag{9}
$$

where � denotes the voxel location and $S U V _ { \operatorname* { m a x } }$ is the global cohort SUV ceiling. During evaluation, predictions were converted back to SUV using the inverse transform. The CT-derived proxy channels �(�) and �(�) were computed before resampling and normalized by their respective 99th percentiles.

## 3.3. Evaluation Metrics

Whole-volume image quality was assessed using the Structural Similarity Index Measure (SSIM), Peak Signal-to-Noise Ratio (PSNR), and Mean Absolute Error (MAE). Whole-volume SUVmax and SUVmean percentage errors were reported after converting predictions back from log-normalized to the standardized uptake value (SUV) space.

Lesion-level activity preservation was evaluated using Total Lesion Activity (TLA). Lesions were obtained from the ground-truth tumor mask using 26-connectivity connected-component labeling. For each lesion, physical TLA was computed as the sum of SUV within the lesion mask, multiplied by voxel volume, which is equivalent to SUVmean multiplied by lesion volume. This quantity is proportional to SUV⋅mL TLA because the predicted and ground-truth values are evaluated on the same lesion mask; therefore, the voxel-volume factor cancels in the percentage-error calculation. The percentage error was computed as

$$
e _ { p , k } = \frac { \widehat { \mathrm { T L A } } _ { p , k } - \mathrm { T L A } _ { p , k } } { \mathrm { T L A } _ { p , k } } \times 1 0 0 ,\tag{10}
$$

where $\mathrm { T L A } _ { p , k }$ and $\widehat { \mathrm { T L A } } _ { p , k }$ are the real and predicted TLA values for lesion � in patient �. We reported TLA error in two ways. First, lesion-level error was computed by pooling all lesions across all patients and averaging the absolute percentage error. This measures how well the model preserves lesion activity across the full lesion cohort. Second, patient-level error was computed by averaging lesion errors within each patient first and then averaging across patients. We also reported the fraction of lesions within 25%, 50%, and 75% absolute TLA error.

Radiomics reproducibility was evaluated using the Intraclass Correlation Coeficient (ICC) between radiomics features extracted from real and synthesized PET volumes. ICC was used because it measures agreement between real and predicted feature values across the test cohort, rather than only their linear association. Features were extracted from the tumor core and from the 0–10 mm peritumoral ring outside the tumor mask. For simplicity, radiomics reproducibility was summarized by feature class: first-order features (FO, � = 11), gray-level co-occurrence matrix features (GLCM, $n = 6 )$ , gray-level run-length matrix features (GLRLM, � = 4), and gray-level size-zone matrix features (GLSZM, � = 4).

## 4. Results

Figure 4 shows representative PET reconstructions from diferent models. The following sections report the quantitative results.

Whole-volume image quality across all models. The shaded row is the proposed model; bold marks the best value per column.
<table><tr><td></td><td colspan="3"> $^ { 1 8 } \mathsf { F } \mathrm { - } \mathsf { P S M A } \left( N = 4 7 \right)$ </td><td colspan="3"> $^ { 6 8 } \mathsf { G a } \mathsf { - P S M A } \ ( N = 3 0 )$ </td></tr><tr><td>Model</td><td>SSIM ↑</td><td>PSNR ↑</td><td>MAE↓</td><td>SSIM ↑</td><td>PSNR ↑</td><td>MAE↓</td></tr><tr><td>LAFNO (ours)</td><td> $0 . 9 6 0 { \scriptstyle \pm 0 . 0 1 4 }$  </td><td> $3 5 . 9 8 { \scriptstyle \pm 1 . 6 0 }$  </td><td> $0 . 0 0 3 8 { \scriptstyle \pm 0 . 0 0 1 2 }$  </td><td> $0 . 9 3 8 { \pm } 0 . 0 1 6$  </td><td> $3 5 . 1 0 { \scriptstyle \pm 1 . 2 8 }$  </td><td> $0 . 0 0 4 8 { \scriptstyle \pm 0 . 0 0 1 1 }$ </td></tr><tr><td>AFNO-L1</td><td>_  $\pm { \bf 0 . 9 6 3 } \pm { \bf 0 . 0 1 4 }$ </td><td>一  ${ 3 6 . 4 2 { \scriptstyle \pm 1 . 7 5 } }$ </td><td> $\mathbf { 0 . 0 0 3 6 { \scriptstyle \pm 0 . 0 0 1 2 } }$ </td><td> $\mathbf { 0 . 9 4 5 { \scriptstyle \pm 0 . 0 1 3 } }$ </td><td> $\pm \mathbf { 5 . 8 0 { \pm 1 . 3 6 } }$ </td><td> $\mathbf { 0 . 0 0 4 4 { \scriptstyle \pm 0 . 0 0 0 9 } }$ </td></tr><tr><td>cWDM</td><td> $0 . 9 6 1 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td> $3 6 . 2 2 { \scriptstyle \pm 1 . 8 0 }$ </td><td> $0 . 0 0 3 7 { \scriptstyle \pm 0 . 0 0 1 3 }$ </td><td> $0 . 9 3 5 { \scriptstyle \pm 0 . 0 1 5 }$ </td><td> $3 4 . 9 6 { \scriptstyle \pm 1 . 2 6 }$ </td><td> $0 . 0 0 4 9 { \scriptstyle \pm 0 . 0 0 1 0 }$ </td></tr><tr><td>FlowLet</td><td> $0 . 9 3 4 { \scriptstyle \pm 0 . 0 2 3 }$ </td><td> $3 3 . 9 8 { \pm } 1 . 4 9$ </td><td> $0 . 0 0 5 4 { \scriptstyle \pm 0 . 0 0 1 6 }$ </td><td> $0 . 9 1 0 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td> $3 1 . 9 2 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $0 . 0 0 7 1 { \scriptstyle \pm 0 . 0 0 1 2 }$ </td></tr><tr><td>Pix2Pix</td><td> $0 . 9 4 4 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $3 3 . 7 4 { \scriptstyle \pm 1 . 3 8 }$ </td><td> $0 . 0 0 5 0 { \scriptstyle \pm 0 . 0 0 1 6 }$ </td><td> $0 . 9 3 2 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td> $3 3 . 7 7 { \scriptstyle \pm 1 . 3 0 }$ </td><td> $0 . 0 0 5 5 { \scriptstyle \pm 0 . 0 0 1 2 }$ </td></tr></table>

## 4.1. Whole-volume Metrics

Table 1 reports whole-volume SSIM, PSNR, and MAE for both tracer cohorts. AFNO-L1 achieved the best global image-quality metrics on both <sup>18</sup>F-PSMA and <sup>68</sup>Ga-PSMA, while LAFNO showed slightly lower whole-volume fidelity. This trade-of is expected because LAFNO is optimized to preserve tumor-core and peritumoral information rather than maximizing global image similarity alone.

Whole-volume SUV error (% error, mean ± SD). The shaded row is the proposed model; bold marks the lowest error per column.
<table><tr><td></td><td colspan="2"> $^ { 1 8 } \mathsf { F { - } P S M A } \ ( N = 4 7 )$ </td><td colspan="2"> $^ { 6 8 } \mathsf { G a } \mathsf { - P S M A } \ ( N = 3 0 )$ </td></tr><tr><td>Model</td><td>SUVmax % err ↓</td><td>SUVmean % err ↓</td><td>SUVmax % err ↓</td><td>SUVmean % err ↓</td></tr><tr><td>LAFNO (ours)</td><td> $5 5 . 0 5 { \pm } 2 0 . 1 9$  1</td><td>_  $8 . 6 1 { \pm } 6 . 2 9$  </td><td> $2 3 . 6 0 { \scriptstyle \pm 1 7 . 6 5 }$  1</td><td>13.41±7.20</td></tr><tr><td>AFNO-L1</td><td>M  $6 2 . 9 8 { \scriptstyle \pm 1 8 . 0 1 }$ </td><td>10.12±6.39</td><td> ${ \bf 1 8 . 4 9 { \scriptstyle \pm 1 8 . 2 3 } }$ </td><td>12.18±7.01</td></tr><tr><td>cWDM</td><td> $4 6 . 1 4 { \scriptstyle \pm 2 4 . 6 5 }$ </td><td> $7 . 2 4 { \scriptstyle \pm 7 . 4 8 }$ </td><td> $3 9 . 8 4 \pm 3 5 . 8 4$ </td><td>7.52±5.93</td></tr><tr><td>FlowLet</td><td> $6 7 . 0 8 { \scriptstyle \pm 3 5 . 6 8 }$ </td><td> $9 . 9 5 { \pm } 7 . 0 5 $ </td><td> $5 9 . 6 0 { \scriptstyle \pm 1 3 . 6 1 }$ </td><td>13.26±7.10</td></tr><tr><td>Pix2Pix</td><td> $6 1 . 7 1 { \scriptstyle \pm 1 8 . 7 2 }$ </td><td>19.73±8.19</td><td> $2 4 . 1 9 { \scriptstyle \pm 1 8 . 3 1 }$ </td><td>10.84±6.85</td></tr></table>

Table 2 reports whole-volume SUVmax and SUVmean percentage errors. cWDM achieved the lowest SUVmax and SUVmean errors for <sup>18</sup>F-PSMA and the lowest SUVmean error for $^ { 6 8 } \mathrm { G a - P S M A }$ , while AFNO-L1 achieved the lowest SUVmax error for <sup>68</sup>Ga-PSMA. LAFNO remained competitive across global SUV metrics, but was not consistently the best model.

## 4.2. Total Lesion Activity (TLA)

Lesion-level TLA error. Lesion-level |err| pools all lesions; per-patient |err| averages within each patient first, then across patients. Within-�% is the per-patient fraction of lesions with |TLA error| ≤ �%. The shaded row is the proposed model; bold marks the best value per column.
<table><tr><td>Model</td><td colspan="5">18F-PSMA</td><td colspan="5"> $^ { 6 8 } \mathsf { G a - P S M A }$ </td></tr><tr><td></td><td>Lesion |err| ↓</td><td>Patient |err| ↓</td><td>≤25% ↑</td><td> $\leq 5 0 \% \uparrow$ </td><td> $\leq 7 5 \% \uparrow$ </td><td>Lesion |err| ↓</td><td>Patient |err| ↓</td><td> $\leq 2 5 \% \uparrow$ </td><td>≤50%↑</td><td> $\leq 7 5 \% \uparrow$ </td></tr><tr><td>LAFNO (ours)</td><td>52.7</td><td>48.3</td><td>24.0</td><td>50.9</td><td>81.0</td><td>54.4</td><td>64.0</td><td>13.2</td><td>31.3</td><td>61.4</td></tr><tr><td>AFNO-L1</td><td>66.2</td><td>62.2</td><td>9.8</td><td>25.9</td><td>65.3</td><td>73.9</td><td>71.7</td><td>4.1</td><td>17.3</td><td>43.4</td></tr><tr><td>cWDM</td><td>62.8</td><td>58.7</td><td>10.6</td><td>29.5</td><td>72.3</td><td>70.5</td><td>63.9</td><td>13.6</td><td>24.5</td><td>55.6</td></tr><tr><td>FlowLet</td><td>62.3</td><td>59.2</td><td>8.3</td><td>34.3</td><td>70.9</td><td>72.6</td><td>67.6</td><td>10.8</td><td>22.8</td><td>47.6</td></tr><tr><td>Pix2Pix</td><td>70.4</td><td>69.0</td><td>4.1</td><td>15.8</td><td>52.9</td><td>78.5</td><td>71.5</td><td>10.4</td><td>22.1</td><td>39.1</td></tr></table>

Table 3 reports lesion-level and patient-level TLA error. For <sup>18</sup>F-PSMA, LAFNO achieved the lowest absolute TLA error and the largest fraction of lesions with TLA error below 25%, 50%, and 75%. The mean absolute patient-level TLA error was reduced to 48.3%, compared with 58.7–69.0% for the baseline models. For $^ { 6 8 } \mathrm { G a - P S M A }$ , LAFNO, and cWDM, performance was similar in absolute TLA error, while LAFNO had more lesions per patient with TLA errors below 50% and 75%. Figure 5 shows predicted versus real per-lesion TLA on log–log axes. When errors were pooled across all lesions, LAFNO achieved the lowest overall TLA error for both tracers, with 52.7% error for <sup>18</sup>F-PSMA and 54.4% error for $^ { 6 8 } \mathrm { G a - P S M A }$ (Table 3). This indicates improved preservation of activity at the lesion level throughout the whole set of lesions. The scatter plots also show that LAFNO more closely follows the identity line, particularly for moderate and high-activity lesions.

![](images/cd74733f147157be0c81c6c00afc5d0aa756afb1fd6a5fda5bb536594f57f36b.jpg)  
Figure 5: Predicted versus real per-lesion TLA on log–log axes for <sup>18</sup>F-PSMA (top) and $^ { 6 8 } \mathsf { G a - P S M A }$ (bottom). The dashed line indicates perfect agreement; the annotation gives the mean absolute TLA error across all lesions. All models underestimate high-activity lesions, while LAFNO tracks the identity line most closely across the full range of lesion burden.

## 4.3. Radiomics Reproducibility

## Table 4

Radiomics reproducibility (ICC) between real and synthesized PET, by feature class. Higher is better. The shaded rows are the proposed model; bold marks the best value per column within each region.
<table><tr><td></td><td colspan="4">18F-PSMA</td><td colspan="4">68Ga-PSMA</td></tr><tr><td>Model</td><td>FO</td><td>GLCM</td><td>GLRLM</td><td>GLSZM</td><td>FO</td><td>GLCM</td><td>GLRLM</td><td>GLSZM</td></tr><tr><td colspan="9">Tumor core</td></tr><tr><td>LAFNO (ours)</td><td>0.38</td><td>0.62</td><td>0.78</td><td>0.71</td><td>0.27</td><td>0.36</td><td>0.54</td><td>0.47</td></tr><tr><td>AFNO-L1</td><td>0.24</td><td>0.39</td><td>0.59</td><td>0.51</td><td>0.13</td><td>0.14</td><td>0.44</td><td>0.44</td></tr><tr><td>cWDM</td><td>0.20</td><td>0.44</td><td>0.58</td><td>0.43</td><td>0.12</td><td>0.24</td><td>0.51</td><td>0.45</td></tr><tr><td>FlowLet</td><td>0.17</td><td>0.29</td><td>0.40</td><td>0.35</td><td>0.17</td><td>0.30</td><td>0.48</td><td>0.44</td></tr><tr><td>Pix2Pix</td><td>0.21</td><td>0.36</td><td>0.57</td><td>0.49</td><td>-0.03</td><td>0.08</td><td>0.34</td><td>0.37</td></tr><tr><td colspan="9">Peritumoral ring (0–10 mm)</td></tr><tr><td>LAFNO (ours)</td><td>0.59</td><td>0.58</td><td>0.85</td><td>0.70</td><td>0.48</td><td>0.33</td><td>0.58</td><td>0.56</td></tr><tr><td>AFNO-L1</td><td>0.56</td><td>0.45</td><td>0.66</td><td>0.64</td><td>0.43</td><td>0.33</td><td>0.60</td><td>0.54</td></tr><tr><td>cWDM</td><td>0.55</td><td>0.49</td><td>0.73</td><td>0.61</td><td>0.50</td><td>0.51</td><td>0.72</td><td>0.49</td></tr><tr><td>FlowLet</td><td>0.46</td><td>0.37</td><td>0.55</td><td>0.57</td><td>0.42</td><td>0.27</td><td>0.58</td><td>0.58</td></tr><tr><td>Pix2Pix</td><td>0.53</td><td>0.30</td><td>0.51</td><td>0.64</td><td>0.41</td><td>0.40</td><td>0.68</td><td>0.50</td></tr></table>

FO = first-order (�=11); GLCM = gray-level co-occurrence matrix (�=6); GLRLM = gray-level run-length matrix (�=4); GLSZM = gray-level size-zone matrix (�=4).

Table 4 reports radiomics reproducibility between real and synthesized PET using ICC by feature class. In the tumor core, LAFNO achieved the highest ICC across all feature classes for both <sup>18</sup>F-PSMA and <sup>68</sup>Ga-PSMA, indicating improved preservation of tumor radiomic structure compared with the baseline models. The strongest agreement was observed for texture feature classes such as GLRLM and GLSZM, while first-order features showed lower reproducibility, particularly for <sup>68</sup>Ga-PSMA. This suggests that although LAFNO better preserves tumor texture patterns, recovery of absolute intensity-distribution features remains limited and requires further improvement.

![](images/817c774d968719db0507257e2bcd856918e46fe773219b65f8ebe8cc1b62b7e8.jpg)

![](images/72a8122a66ab0f0963e9a8c92c88e9122e69a69b89fa2dbd20842441652ac370.jpg)  
Figure 6: Peritumoral radiomics reproducibility (ICC) by distance band from the tumor boundary, for $^ { 1 8 } \mathsf { F - P S M A }$ (left) and $^ { 6 8 } \mathsf { \bar { G } a \mathrm { - } P S M A }$ (right). The dashed curve shows the training loss weight exp(−�∕5).

In the 0–10 mm peritumoral ring, LAFNO achieved the highest ICC across all feature classes for $^ { 1 8 } \mathrm { F - P S M A }$ suggesting improved preservation of tumor-adjacent texture patterns for this tracer. For $^ { 6 8 } \mathrm { G a - P S M A }$ , the peritumoral results were more mixed, with cWDM achieving the highest ICC for FO, GLCM, and GLRLM, and FlowLet achieving the highest GLSZM ICC. Figure 8 resolves peritumoral reproducibility by distance band from the tumor boundary. The two tracers behave diferently. For <sup>18</sup>F-PSMA, the texture classes GLRLM and GLSZM remain high across the full range, while first-order and GLCM are lowest at the immediate rim and recover with distance; no single band is best for every class. For <sup>68</sup>Ga-PSMA, all four classes rise from the 0–3 mm band outward, and the texture classes peak sharply at 3–5 mm before declining. In both tracers, reproducibility at the immediate rim is not the highest despite the training weight exp(−�∕5) being largest there, indicating that the region closest to the high-contrast tumor boundary is the hardest to reproduce.

## 5. Ablation Study

All ablation experiments were conducted on the <sup>18</sup>F-PSMA cohort. We first compared two formulations of the per-lesion loss term across a range of weights to establish which better supervises lesion activity (Section 5.1). We then evaluated LAFNO using a cumulative ablation design. Table 5 shows the cumulative ablation study. Starting from AFNO-L1, we added CT-derived proxy conditioning, $\mathcal { L } _ { \mathrm { T L A } } , \mathcal { L } _ { c }$ , and $\mathcal { L } _ { \mathrm { p e r i } }$ one step at a time. Each row includes the components from the row above it, so changes between rows show the added efect of each component. The results are discussed in Sections 5.2–5.5. Throughout, the backbone, training schedule, and test split are held fixed, so that any diference is attributable to the component under test.

## 5.1. Choice of Per-Lesion Loss Formulation

We first compared two formulations of the per-lesion supervision term. The first is the lesion-wise summed-activity loss ${ \mathcal { L } } _ { \mathrm { T L A } }$ in Equation 5, which supervises each lesion independently. The second is a uniform L1 loss restricted to tumor-mask voxels, denoted ${ \mathcal { L } } _ { \mathrm { { T u m o r L 1 } } }$

$$
\mathcal { L } _ { \mathrm { T u m o r L 1 } } = \frac { 1 } { \left| \mathcal { M } \right| } \sum _ { i \in \mathcal { M } } \left| \hat { y } _ { i } - y _ { i } \right| ,\tag{11}
$$

where  is the set of tumor-mask voxels, and $\hat { y } _ { i }$ and $y _ { i }$ are the predicted and ground-truth PET values.

Each loss was trained at four weights, $\lambda \in \{ 0 . 0 5 , 0 . 1 , 0 . 5 , 1 . 0 \}$ , on top of an identical whole-volume L1 base loss, using the 3D U-Net with AFNO bottleneck backbone. Figure 7 shows that ${ \mathcal { L } } _ { \mathrm { T L A } }$ and ${ \mathcal { L } } _ { \mathrm { { T u m o r L 1 } } }$ respond very diferently as � increases. ${ \mathcal { L } } _ { \mathrm { { T u m o r L } } }$ improves sharply between $\lambda = 0 . 0 5$ and $\lambda = 0 . 1$ , with absolute TLA error falling from 51.3% to 46.8%, and then saturates, fluctuating between 45.8% and 48.4% over the remainder of the range, and its overestimation rate flattens near 22–24%. In contrast, ${ \mathcal { L } } _ { \mathrm { T L A } }$ declines steadily across the full range, from 51.0% to 42.9%, and its overestimation rate rises correspondingly from 13.9% to 36.6%.

![](images/09077e735c7e8da65bc5334ee1945352c35fd6ac982ccef1a785e1cae9367fb1.jpg)

![](images/3e147ee1c5826b295316fd4a4461c36040f2ff4e71e465e22b27ca9160ed4b47.jpg)  
Figure 7: Per-lesion loss formulations across $\lambda \in \{ 0 . 0 5 , 0 . 1 , 0 . 5 , 1 . 0 \}$ on ${ } ^ { 1 8 } \mathsf { F - P S M A }$ . (A) Per-patient overestimate rate. (B) Per-patient absolute TLA error. TumorL1 $( \mathcal { L } _ { \mathtt { T u m o r L 1 } } )$ saturates after $\lambda = 0 . 1 ;$ the TLA loss $\left( \mathcal { L } _ { \mathtt { T L A } } \right)$ continues to respond across the full range.

Both models show systematic underestimation at low lesion weight. Therefore, the increase in overestimation reflects ${ \mathcal { L } } _ { \mathrm { T L A } }$ progressively correcting this bias. $\mathcal { L } _ { \mathrm { { T u m o r L 1 } } }$ stops correcting once the tumor-mask error saturates, whereas ${ \mathcal { L } } _ { \mathrm { T L A } }$ continues to respond to lesion-level activity mismatch. Moreover, $\mathcal { L } _ { \mathrm { { T u m o r L 1 } } }$ penalizes every tumor voxel uniformly, so its gradient is dominated by large lesions and by the bulk of voxels already close to their target. In contrast, the $\mathcal { L } _ { \mathrm { T L A } } \mathrm { ^ { 2 } s }$ denominator in Equation 5, log $\left( 1 + N _ { k } \cdot S U V _ { \operatorname* { m a x } } \right)$ , converts the raw size-dependent lesion-activity error into a relative, per-lesion-normalized error. This prevents larger lesions from dominating the objective and helps stabilize gradients across lesions of diferent sizes. We therefore adopt the lesion-wise ${ \mathcal { L } } _ { \mathrm { T L A } }$ formulation. Its higher accuracy at large � comes with an increased overestimation rate, which is why the final model uses a low weight of $\lambda _ { \mathrm { T L A } } = 0 . 0 5$ and relies on the CT proxy channels and the tumor-contrast loss $\mathcal { L } _ { c }$ to improve tumor-level performance. This keeps the overestimation rate below 20% in the final model: 19.98% for <sup>18</sup>F-PSMA with 48.35% absolute TLA error, and 10.47% for $^ { 6 8 } \mathrm { G a - P S M A }$ with 64.01% absolute TLA error, as reported in Table 3.

## Table 5

Cumulative ablation on <sup>18</sup>F-PSMA cohort. Each row adds one component to the row above. ICC values are the mean over features within each class. The shaded row is the full proposed model.
<table><tr><td></td><td colspan="4">Lesion TLA</td><td colspan="2">Whole-volume</td><td colspan="4">Tumor-core ICC</td></tr><tr><td>Configuration</td><td>|err| ↓</td><td>≤25%</td><td>≤50%</td><td> $\leq 7 5 \%$ </td><td>SSIM</td><td>PSNR</td><td>FO</td><td>GLCM</td><td>GLRLM</td><td>GLSZM</td></tr><tr><td>AFNO-L1 (CT only)</td><td>62.2</td><td>9.8</td><td>25.9</td><td>65.3</td><td>0.963</td><td>36.42</td><td>0.235</td><td>0.393</td><td>0.589</td><td>0.510</td></tr><tr><td>+ proxies</td><td>56.5</td><td>13.0</td><td>35.3</td><td>72.9</td><td>0.966</td><td>37.05</td><td>0.328</td><td>0.532</td><td>0.723</td><td>0.481</td></tr><tr><td> $+ \ \mathcal { L } _ { \mathrm { T L A } }$ </td><td>50.5</td><td>19.6</td><td>51.1</td><td>78.8</td><td>0.965</td><td>36.71</td><td>0.357</td><td>0.565</td><td>0.739</td><td>0.624</td></tr><tr><td>+ Lc</td><td>47.8</td><td>25.2</td><td>54.9</td><td>80.8</td><td>0.960</td><td>35.88</td><td>0.392</td><td>0.626</td><td>0.809</td><td>0.760</td></tr><tr><td> $+ \mathcal { L } _ { \mathsf { p e r i } } \ ( \mathsf { L A F N O } )$  1</td><td>48.3</td><td>24.0</td><td>50.9</td><td>81.0</td><td>0.960</td><td>35.98</td><td>0.381</td><td>0.616</td><td>0.783</td><td>0.709</td></tr></table>

## 5.2. Efect of CT Proxy Conditioning

We first isolated the contribution of the CT-derived proxy channels by training two models under the same conditions. The first, AFNO-L1, received only the CT patch as input and was trained with plain whole-volume L1 loss. The second, AFNO-L1 + Proxy, used the same backbone as AFNO-L1, but additionally received the CT-derived contrast and disorder proxy channels of Equations 1–2. Since the backbone, decoder, skip connections, and training loss were identical between the two models, this comparison isolates the contribution of CT-derived proxy conditioning.

Adding the proxy channels reduced mean per-patient absolute TLA error from 62.2% to 56.5% and increased the fraction of lesions within 50% of their true activity from 25.9% to 35.3% (Table 5). Whole-volume image quality improved only marginally, and radiomics reproducibility improved in the tumor core. The benefit is therefore concentrated at the lesion-quantification level rather than distributed uniformly across the volume. Notably, this arm used only the plain L1 reconstruction loss, with no tumor-mask, TLA, or peritumoral supervision. This result shows that CT-derived proxy conditioning can improve lesion-level TLA estimation even without lesion segmentations or TLA-specific supervision during training. Unlike the TLA loss, which requires lesion masks to compute ground-truth lesion activity, the contrast and disorder proxies are computed directly from CT. They therefore provide tumor-relevant conditioning by highlighting regions with tumor-like density and texture behavior, even when explicit lesion annotations are unavailable.

![](images/572bc795cf79a926b2caeb74584ab4e9a605243ff42201d01396f27b0182c4ff.jpg)

![](images/c5acf9deabd62137640a2ac66bc0a18086037cde69213fe25e44aa1cc9143726.jpg)

![](images/d4b158e67c2cb040e84842bc90e70588e07616935ce125274710c8173485e7f2.jpg)

![](images/6feafec6903f63c2e58d26de3dae964d7fb0ee2ad5518473a9204b22daa94d0f.jpg)  
Figure 8: Peritumoral radiomics reproducibility (ICC) by distance band, before and after adding $\mathcal { L } _ { \mathsf { p e r i } } ,$ , for each feature class (<sup>18</sup>F-PSMA). The before/after curves overlap for first-order, GLCM, and GLRLM; the main efect is on GLSZM in the 0–3 mm band.

## 5.3. Efect of the Lesion-wise TLA Loss

We next added the lesion-wise TLA term at $\lambda _ { \mathrm { T L A } } = 0 . 0 5$ , holding the architecture and proxy conditioning fixed. The mean per-patient absolute TLA error reduced from 56.5% to 50.5%, and the fraction of lesions within 50% of true activity rose from 35.3% to 51.1% (Table 5). Whole-volume fidelity decreased slightly, reflecting the expected trade between a lesion-focused objective and global reconstruction. The lesion-level gain substantially outweighs this cost, confirming that explicit per-lesion activity supervision supplies information the proxy conditioning alone does not.

## 5.4. Efect of the Tumor-Contrast Loss

We next added the tumor-contrast term $\mathcal { L } _ { c } .$ . This improved lesion quantification further, reducing absolute TLA error to 47.8% and raising the fraction of lesions within 25% of true activity from 19.6% to 25.2%. Its clearest efect, however, was on intra-lesion structure: tumor-core radiomics reproducibility rose in every feature class, with the largest gain in GLSZM (0.624 to 0.760), indicating better recovery of zone-size distribution within the lesion (Table 5). This is consistent with $\mathcal { L } _ { c }$ supervising local density structure inside the tumor mask, which the TLA term, constraining only total activity, does not.

## 5.5. Efect of the Peritumoral Loss

Finally, we added the peritumoral term $\mathcal { L } _ { \mathrm { { p e r i } } }$ at $\lambda _ { \mathrm { p e r i } } = 0 . 0 5$ , kept deliberately small so that supervision of the surrounding band does not overpower the tumor-core terms, while leaving lesion-level and tumor-core metrics essentially unchanged (Table 5).

Resolving the efect by feature class and distance band (Figure 8) shows that the term acts narrowly rather than broadly. For first-order, GLCM, and GLRLM, peritumoral reproducibility is almost identical before and after adding $\mathcal { L } _ { \mathrm { p e r i } }$ , with the before/after curves separated by at most 0.02–0.03 ICC at any band. The one substantial efect is on GLSZM at the immediate rim: size-zone reproducibility in the 0–3 mm band rises from 0.637 to 0.849 $( \Delta = 0 . 2 1 )$ , while the same class changes little in the outer bands $( \leq 0 . 0 2 )$ . The improvement is therefore concentrated exactly where the exponential weight is largest, consistent with the loss design, but it refines a single texture property at the tumor boundary rather than improving peritumoral fidelity generally.

Peritumoral fidelity nonetheless remains the most challenging region for the model. The narrowness of this efect likely reflects the small weight assigned to $\mathcal { L } _ { \mathrm { { p e r i } } }$ , and we did observe a slight drop in tumor-core ICC, once this term was included, which is expected when balancing many competing objectives in a single loss. A more carefully designed objective that jointly penalizes the tumor core and the peritumoral region while establishing a balance between them may be required to strengthen peritumoral fidelity without compromising the core.

![](images/0fc7ecdf575467fa7974d277ad02807be6f4a6ebd95cd1ca974c4afbba1013b4.jpg)  
Figure 9: Whole-volume SUVmax percentage error on the normalized [0, 1] scale and after conversion back to SUV space, for <sup>18</sup>F-PSMA (left) and $^ { 6 8 } \mathsf { G a - P S M A }$ (right). The inverse log transform amplifies error at the high-uptake end, so SUV-scale error is consistently larger than normalized-scale error across all models.

## 6. Discussion

In this study, we show that a biologically conditioned CT-to-PET synthesis model better preserves clinically relevant PET signals than models optimized primarily for global image similarity. LAFNO was designed as a simple AFNO-based synthesis model with CT-derived proxy conditioning and lesion-aware supervision. The AFNO bottleneck was selected because Fourier-domain operators can model long-range spatial interactions and have shown strong performance in medical image translation tasks Bhaskara and Oderinde (2025). Overall, LAFNO achieved competitive whole-volume image quality while improving lesion-level accuracy and tumor-core radiomics reproducibility.

The whole-volume SSIM, PSNR, and MAE results indicate that LAFNO generates synthetic PET volumes that remain visually close to the ground-truth scans. However, whole-volume SUVmax and SUVmean errors remained relatively high. This is partly expected because the model was trained in log-normalized PET space to handle the large dynamic range of standardized uptake value (SUV) values. When predictions are converted back to SUV space, small errors in log space can become larger absolute errors after inversion, especially in high-uptake regions (Figure 9). This highlights an important limitation of evaluating synthetic PET only using global image metrics or whole-volume SUV errors.

At the lesion level, LAFNO achieved the best TLA performance among the evaluated models, but the absolute TLA error remained high. We attribute part of this residual error to tumor heterogeneity. Tumor lesions are highly heterogeneous (a feature that can challenge the reliability of localized assessments) Almutairi et al. (2025), and tumor phenotype can vary substantially across lesions and within the same patient Heppner et al. (1981). Similar CT appearances may therefore correspond to diferent PET uptake patterns depending on lesion biology, tissue state, and tracer avidity Mahdi et al. (2026). This is dificult to capture with reconstruction losses dominated by background-region accuracy, which may generate visually plausible PET images while still underestimating or misrepresenting lesion activity Abtahi et al. (2026). It is also worth noting that SUV itself is a semiquantitative measure influenced by scanner performance, reconstruction algorithms, calibration, and observer variability, as well as histologic type, tumor grade, and lesion size factors that add further variability independent of model performance Weiss and Korn (2012); Ulaner (2019); Wang et al. (2025). This concern is directly relevant to our dataset, which combines studies acquired across three PET/CT scanners and two PET tracers $( { } ^ { 1 8 } \mathrm { F - } \dot { \mathrm { P S M A } }$ and $^ { 6 8 } \mathrm { G a - P S M A } )$ with difering reconstruction parameters variations that can afect measured SUV by more than 50% according to EANM quality-control guidelines Boellaard et al. (2010). The contrast and disorder proxies were designed as simple CT-derived representations of tumor-relevant density variation and local texture heterogeneity. These proxies do not fully resolve lesion heterogeneity, but they only provide the model with additional spatial cues beyond CT intensity alone. This helps explain why LAFNO improved lesion-level TLA performance compared with the baseline models, even though absolute errors remained substantial. In addition, the efectiveness of proxy conditioning may also depend on the imaging modality. Figure 10 shows that applying the same contrast and disorder operations to a T2-weighted MRI prostate patch highlights stronger local prostate texture information than CT. This suggests that MRI-guided or hybrid CT/MRI proxy conditioning may be better suited for PSMA-PET synthesis in soft-tissue prostate disease.

![](images/5edf7daca220ca4cd454e56abe6ab946279645a73264fb287fb1d59cc293d77b.jpg)  
Figure 10: CT- and T2-weighted MRI-derived proxy maps in a prostate-region example. The same contrast and disorder operations highlight a stronger local prostate structure on MRI than on CT.

Similarly, the ICC scores indicated moderate radiomics reproducibility within the tumor core, which is because the tumor interior is also highly heterogeneous. We also observed that LAFNO performed diferently across tracers. For <sup>18</sup>F-PSMA, LAFNO improved texture-based ICC values across feature classes, whereas for $^ { 6 8 } \mathrm { G a - P S M A } .$ , ICC values were generally lower, and several feature classes remained near 0.5. A similar tracer-dependent pattern was observed in the peritumoral region. Although the peritumoral loss used an exponential distance-decay weighting scheme, recovery near the tumor boundary remained challenging, and for <sup>68</sup>Ga-PSMA, reproducibility peaked in the 3–5 mm band rather than at the immediate boundary. This behavior may be related to diferences in tracer uptake, image reconstruction, noise characteristics, or tumor boundary contrast. For example, <sup>68</sup>Ga-PSMA has a higher average positron energy than <sup>18</sup>F-PSMA, resulting in a longer positron range, reduced spatial resolution, and greater partial volume efects. These factors can degrade tracer quantification and introduce lesion localization uncertainty, which may contribute to reduced model performance. Reconstruction diferences may also afect feature reproducibility. In this study, images were acquired across diferent scanners with various time of flight corrections introducing additional variability that may have influenced model performance. Future studies should aim to minimize inter-tracer and inter-scanner diferences by implementing partial volume correction to improve tracer quantification, evaluating anatomically relevant non-PSMA imaging targets that may improve tissue contrast, and standardizing image acquisition, reconstruction, and data preparation workflows. Additional correction for distance dependent peritumoral efects may further improve model robustness.

Finally, the ablation study helps explain how each component contributed to these results. CT-derived proxy conditioning improved lesion-level TLA estimation even without lesion-specific supervision, indicating that the contrast and disorder proxies provide useful tumor-relevant spatial cues. Adding ${ \mathcal { L } } _ { \mathrm { T L A } }$ further improved lesion activity preservation by directly supervising each connected component. The tumor-contrast loss $\mathcal { L } _ { c }$ produced the clearest gain in tumor-core radiomics reproducibility, suggesting that local contrast supervision helps preserve intra-lesion structure. The peritumoral loss $\mathcal { L } _ { \mathrm { p e r i } }$ had a more localized efect, mainly improving GLSZM reproducibility near the immediate tumor boundary, while leaving most lesion-level and tumor-core metrics largely unchanged.

Several limitations should be noted. The dataset is likely biased toward visually apparent and higher-burden lesions, particularly bone-dominant metastatic disease, because these lesions are easier to identify and annotate on PSMA-PET/CT. As a result, the proxies may not fully represent subtle soft-tissue lesions, small nodal disease, or lesions with weak uptake. Future works will likely require more richly annotated datasets that distinguish lesion phenotype, tissue state, and imaging context. Necrotic, sclerotic, soft-tissue, inflammatory, and physiologic uptake regions may show diferent PET behavior even when their CT appearance overlaps. Distinguishing these tissue types and characterizing how the underlying biological signal varies across them would allow models to learn subtype-specific patterns rather than relying on a single generalized CT-to-PET mapping. Such annotations could also help identify more specific CT and PET signatures for diferent lesion groups and guide the design of future conditioning strategies. Another major limitation is that high physiologic bladder activity can introduce spillover, noise, and boundary-related uncertainty in the pelvic region, which may afect both training and evaluation near the prostate bed and adjacent soft tissues. Future datasets could reduce this source of uncertainty by using acquisition protocols that limit bladder activity, such as catheter-assisted bladder drainage or other standardized bladder-management strategies during image acquisition. Models trained on such datasets may better learn prostate-bed activity with reduced spillover from high bladder uptake.

## 7. Conclusion

In this work, we proposed LAFNO for CT-to-PSMA-PET synthesis using CT-derived proxy conditioning and lesion-aware supervision. The model was designed to preserve clinically relevant tumor information. LAFNO also maintained a competitive global image quality while improving lesion-level activity preservation and tumor-core radiomics reproducibility compared with the baseline models. However, peritumoral reproducibility and tracer-dependent diferences, especially for <sup>68</sup>Ga-PSMA, remain challenging. Future work should focus on larger annotated datasets, stronger biological conditioning, and improved modeling of tumor-adjacent tissue.

## References

Abdollahi, H., Harsini, S., Yousefirizi, F., Hatami, B., Bénard, F., Shariftabrizi, A., Alberts, I., Rahmim, A., 2026. Microenvironment at a distance: Multi-endocrine-organ radiomics to identify systemic signatures in psma-negative prostate cancer. Cancers 18, 1767.

Abtahi, M., Hong, C.M., Deveshwar, N., Nwihim, S., Larson, P.E., Hope, T.A., 2026. Fine-unetr for psma pet/ct lesion segmentation: Automated tumor quantification and overall survival stratification in prostate cancer. arXiv preprint arXiv:2606.17570 .

Algohary, A., Shiradkar, R., Pahwa, S., Purysko, A., Verma, S., Moses, D., Shnier, R., Haynes, A.M., Delprado, W., Thompson, J., et al., 2020. Combination of peri-tumoral and intra-tumoral radiomic features on bi-parametric mri accurately stratifies prostate cancer risk: a multi-site study. Cancers 12, 2200.

Almutairi, W.M., Gopaulchan, M., Bhaskara, R., Zheng, Q.H., Ocana, J.A., Langer, M., Perekattu Kuruvilla, T., Durm, G., Oderinde, O.M., 2026. Metabolic [18f]f-arag pet imaging of t cell activation: a functional complement to cell-specific immune tracers. EJNMMI Research 16, 85. doi:10.1186/s13550-026-01436-6.

Almutairi, W.M., Zheng, Q.H., Langer, M., Oderinde, O.M., 2025. Application of immunopet imaging to enhance head and neck squamous cell carcinoma clinical management. Frontiers in Oncology 15, 1644692. doi:10.3389/fonc.2025.1644692.

Bhaskara, R., Oderinde, O., 2025. Enhancing synthetic pelvic ct generation from cbct using vision transformer with adaptive fourier neural operators. Biomedical Physics & Engineering Express 11, 055013.

Boellaard, R., O’Doherty, M.J., Weber, W.A., Mottaghy, F.M., Lonsdale, M.N., Stroobants, S.G., Oyen, W.J.G., Kotzerke, J., Hoekstra, O.S., Pruim, J., Marsden, P.K., 2010. Fdg pet and pet/ct: Eanm procedure guidelines for tumour pet imaging: Version 1.0. European Journal of Nuclear Medicine and Molecular Imaging 37, 181–200. doi:10.1007/s00259-009-1297-4.

Bouchelouche, K., Turkbey, B., Choyke, P.L., 2016. Psma pet and radionuclide therapy in prostate cancer, in: Seminars in nuclear medicine, Elsevier. pp. 522–535.

Caldarella, C., De Risi, M., Massaccesi, M., Miccichè, F., Bussu, F., Galli, J., Rufini, V., Leccisotti, L., 2024. Role of 18f-fdg pet/ct in head and neck squamous cell carcinoma: current evidence and innovative applications. Cancers 16, 1905.

Chai, L., Yao, X., Yang, X., Na, R., Yan, W., Jiang, M., Zhu, H., Sun, C., Dai, Z., Yang, X., 2025. Synthesizing [18f] psma-1007 pet bone images from ct images with gan for early detection of prostate cancer bone metastases: a pilot validation study. BMC cancer 25, 907.

Danese, D., Lombardi, A., Attimonelli, M., Fasano, G., Di Noia, T., 2026. Flowlet: Conditional 3d brain mri synthesis using wavelet flow matching. Medical Image Analysis 113, 104161. URL: http://dx.doi.org/10.1016/j.media.2026.104161, doi:10.1016/j.media. 2026.104161

Dayarathna, S., Islam, K.T., Uribe, S., Yang, G., Hayat, M., Chen, Z., 2024. Deep learning based synthesis of mri, ct and pet: Review and analysis. Medical image analysis 92, 103046.

Dondi, F., Gazzilli, M., Albano, D., Rizzo, A., Treglia, G., Pisani, A.R., Palumbo, C., Rubini, D., Racca, M., Rubini, G., et al., 2024. Prognostic role of pre-and post-treatment [18f] fdg pet/ct in squamous cell carcinoma of the oropharynx in patients treated with chemotherapy and radiotherapy. Medical Sciences 12, 36.

Emami, H., Liu, Q., Dong, M., 2020. Frea-unet: frequency-aware u-net for modality transfer. arXiv preprint arXiv:2012.15397 .

Friedrich, P., Durrer, A., Wolleb, J., Cattin, P.C., 2024. cwdm: Conditional wavelet difusion models for cross-modality 3d medical image synthesis. URL: https://arxiv.org/abs/2411.17203, arXiv:2411.17203.

Ghafari, A., Sheikhzadeh, P., Seyyedi, N., Abbasi, M., Farzenefar, S., Yousefirizi, F., Ay, M.R., Rahmim, A., 2022. Generation of 18f-fdg pet standard scan images from short scans using cycle-consistent generative adversarial network. Physics in Medicine & Biology 67, 215005.

Gillies, R.J., Kinahan, P.E., Hricak, H., 2016. Radiomics: images are more than pictures, they are data. Radiology 278, 563–577.

Guibas, J., Mardani, M., Li, Z., Tao, A., Anandkumar, A., Catanzaro, B., 2021. Adaptive fourier neural operators: Eficient token mixers for transformers. arXiv preprint arXiv:2111.13587 .

Heppner, G.H., Shapiro, W.R., Rankin, J.K., 1981. Tumor heterogeneity. Pediatric Oncology 1: with a special section on Rare Primitive Neuroectodermal Tumors , 99–116.

Ishibashi-Kanno, N., Yamagata, K., Hara, T., Takaoka, S., Fukuzawa, S., Uchida, F., Bukawa, H., 2023. Prognostic prediction using maximum standardized uptake value ratio of lymph node-to-primary tumor in preoperative pet-ct for oral squamous cell carcinoma. Journal of Stomatology, Oral and Maxillofacial Surgery 124, 101489.

Isola, P.. Zhu, J.Y., Zhou, T.. Efros, A.A., 2017. Image-to-image translation with conditional adversarial networks, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1125–1134.

Jeblick, K., Schachtner, B., Mittermeier, A., Dexl, J., Wesp, P., Küstner, T., Gatidis, S., Früh, M., Fabritius, M.P., Herr, F., et al., 2026. A whole-body psma-pet/ct dataset with manually annotated tumor lesions. Scientific Data 13, 1023.

Jemal, A., 2026. Cancer statistics, 2026. CA: A Cancer Journal for Clinicians 76, e70043.

Jung, E., Kong, E., Yu, D., Yang, H., Chicontwe, P., Park, S.H., Jeon, I., 2024. Generation of synthetic pet/mr fusion images from mr images using a combination of generative adversarial networks and conditional denoising difusion probabilistic models based on simultaneous 18f-fdg pet/mr image data of pyogenic spondylodiscitis. The Spine Journal 24, 1467–1477.

Kim, J., Na, I., Ko, E.S., Park, H., 2025. Tumor synthesis conditioned on radiomics, in: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), IEEE. pp. 3635–3646.

Kim, M., Seifert, R., Fragemann, J., Kersting, D., Murray, J., Jonske, F., Pomykala, K.L., Egger, J., Fendler, W.P., Herrmann, K., et al., 2023. Evaluation of thresholding methods for the quantification of [68ga] ga-psma-11 pet molecular tumor volume and their efect on survival prediction in patients with advanced prostate cancer undergoing [177lu] lu-psma-617 radioligand therapy. European journal of nuclear medicine and molecular imaging 50, 2196–2209.

Koca, D., Abedi-Ardekani, B., LeMaoult, J., Guyon, L., 2024. Peritumoral tissue (ptt): increasing need for naming convention. British Journal of Cancer 131, 1111–1115.

Lambin, P., Rios-Velazquez, E., Leijenaar, R., Carvalho, S., Van Stiphout, R.G., Granton, P., Zegers, C.M., Gillies, R., Boellard, R., Dekker, A., et al., 2012. Radiomics: extracting more information from medical images using advanced feature analysis. European journal of cancer 48, 441–446.

Liu, J., Chen, J., Qiu, L., Li, R., Li, Y., Li, T., Leng, X., 2025. The value of intratumoral and peritumoral ultrasound radiomics model constructed using multiple machine learning algorithms for non-mass breast cancer. Scientific Reports 15, 19953.

Mahdi, M.A., Al-Shalabi, M., Elbarougy, R., Alnfrawy, E.T., Hadi, M.U., Ali, R.F., 2026. Ct-to-pet synthesis in the head–neck and thoracic region via conditional 3d latent difusion modeling. Bioengineering 13, 534.

Maurer, T., Eiber, M., Schwaiger, M., Gschwend, J.E., 2016. Current use of psma–pet in prostate cancer management. Nature Reviews Urology 13, 226–235.

Reddy, S., Robinson, M.K., 2010. Immuno-positron emission tomography in cancer models, in: Seminars in nuclear medicine, Elsevier. pp. 182–189.

Ronneberger, O., Fischer, P., Brox, T., 2015. U-net: Convolutional networks for biomedical image segmentation, in: International Conference on Medical image computing and computer-assisted intervention, Springer. pp. 234–241.

Siva, S., Udovicich, C., Tran, B., Zargar, H., Murphy, D.G., Hofman, M.S., 2020. Expanding the role of small-molecule psma ligands beyond pet staging of prostate cancer. Nature Reviews Urology 17, 107–118.

Smith, T., Harper, M., 2025. Prostate-specific membrane antigen (psma) pet-ct: Revolutionizing staging, restaging, and treatment response assessment. Ann. Urol. Oncol 8, 200–210.

Subramanian, K., Martinez, J., Huicochea Castellanos, S., Ivanidze, J., Nagar, H., Nicholson, S., Youn, T., Nauseef, J.T., Tagawa, S., Osborne, J.R., 2023. Complex implementation factors demonstrated when evaluating cost-efectiveness and monitoring racial disparities associated with [18f] dcfpyl pet/ct in prostate cancer men. Scientific Reports 13, 8321.

Tonry, C., Finn, S., Armstrong, J., Pennington, S.R., 2020. Clinical proteomics for prostate cancer: understanding prostate cancer pathology and protein biomarkers for improved disease management. Clinical Proteomics 17, 41.

Traverso, A., Wee, L., Dekker, A., Gillies, R., 2018. Repeatability and reproducibility of radiomic features: a systematic review. International Journal of Radiation Oncology\* Biology\* Physics 102, 1143–1158.

Tsechelidis, I., Vrachimis, A., 2022. Psma pet in imaging prostate cancer. Frontiers in Oncology 12, 831429.

Ulaner, G.A., 2019. Fundamentals of Oncologic PET/CT. Elsevier. doi:10.1016/C2016-0-05304-4.

Wang, G., Zhang, M., Cheng, M., Wang, X., Li, K., Chen, J., Chen, Z., Chen, S., Chen, J., Xiong, G., et al., 2021. Tumor microenvironment in head and neck squamous cell carcinoma: Functions and regulatory mechanisms. Cancer Letters 507, 55–69.

Wang, Y., Yang, Y., Li, J., Cheng, D., Xu, H., Huang, J., 2025. Dynamic fdg pet/ct imaging: Quantitative assessment, advantages and application in the diagnosis of malignant solid tumors. Frontiers in Oncology 15, 1539911. doi:10.3389/fonc.2025.1539911.

Weiss, G.J., Korn, R.L., 2012. Interpretation of pet scans: Do not take suvs at face value. Journal of Thoracic Oncology 7, 1744–1746. doi:10.1097/JTO.0b013e31827450ae.

Yadav, S.S., Stockert, J.A., Hackert, V., Yadav, K.K., Tewari, A.K., 2018. Intratumor heterogeneity in prostate cancer, in: Urologic Oncology: Seminars and Original Investigations, Elsevier. pp. 349–360.

Zhang, S., Regan, K., Najera, J., Grinstaf, M.W., Datta, M., Nia, H.T., 2023. The peritumor microenvironment: physics and immunity. Trends in cancer 9, 609–623.

Zhou, H., Xie, M., Shi, H., Shou, C., Tang, M., Zhang, Y., Hu, Y., Liu, X., 2025. Integrating multimodal imaging and peritumoral features for enhanced prostate cancer diagnosis: a machine learning approach. Plos one 20, e0323752.

Zwanenburg, A., Leger, S., Vallières, M., Löck, S., 2016. Image biomarker standardisation initiative. arXiv preprint arXiv:1612.07003 .