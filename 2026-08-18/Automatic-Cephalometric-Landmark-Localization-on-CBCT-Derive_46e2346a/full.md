# Automatic Cephalometric Landmark Localization on CBCT-Derived Digitally Reconstructed Radiographs for Skeletal Malocclusion Classification

Benjamin Hou<sup>1</sup>, Konstantinia Almpani<sup>2</sup>, Janice S. Lee<sup>2</sup>, and Zhiyong Lu<sup>1</sup>

Division of Intramural Research, National Library of Medicine <sup>2</sup> Craniofacial Anomalies and Regeneration Section, National Institute of Dental and Craniofacial Research, National Institutes of Health, Bethesda, MD, USA.

Abstract. Manual cephalometric landmark annotation is important for craniofacial assessment but is labor-intensive and dificult to scale. We introduce CephViT, a Vision Transformer-based model for automated 2D lateral cephalometric landmark localization, and evaluate its use in downstream skeletal malocclusion classification. CephViT was trained and benchmarked on a public lateral cephalogram dataset, achieving a mean radial error of 1.28 ± 1.42 mm and a successful detection rate of 92.0% at 3.0 mm. Because the private evaluation cohort consisted of 3D CBCT scans, lateral cephalogram-like digitally reconstructed radiographs (DRRs) were generated from each volume and used as 2D inputs to the landmark localization model. Landmark coordinates were normalized into a common coordinate frame, and skeletal malocclusion classification was performed using landmarks shared between the reference and DRR-based pipelines. Classification performance using DRR-localized landmarks was comparable to that obtained using manually annotated reference landmarks, with accuracies of 70.0% and 68.3%, respectively. These results support the feasibility of automated cephalometric analysis on CBCT-derived DRRs for skeletal malocclusion assessment.

Keywords: Cephalometric landmark detection · DRR · Cone-beam CT · Skeletal malocclusion · Vision Transformer

## 1 Introduction

Skeletal malocclusion is a clinically important manifestation of aberrant craniofacial development, which has been associated with a broad range of comorbidities spanning orofacial, respiratory, musculoskeletal, neuropsychological, and systemic domains [13, 17, 3, 16, 20, 2]. Early and accurate diagnosis can support growth monitoring, guide interceptive orthodontic treatment management, and help distinguish cases that may later require more complex orthopedic or surgical treatment [23, 24, 14]. Historically, standard cephalometric analysis on lateral cephalograms has long played a central role in this process by quantifying an individual’s potential degree of deviation from the normative state, as defined by cephalometric measurement values collected from individuals with no skeletal malocclusion (orthognathic). These are linear and angular measurements that are used to assess the spatial relationships of the cranial base, maxilla, mandible, and dentition, thereby helping define the anatomical basis of malocclusion, and assisting in tracking craniofacial growth and development over time [15, 4].

In modern clinical practice, Cone-Beam Computed Tomography (CBCT) is increasingly used in the craniofacial field, ofering a complete and more accurate three-dimensional (3D) assessment of craniofacial anatomy, particularly in cases involving facial asymmetry, complex skeletal discrepancies, or treatment planning beyond what can be reliably inferred from a two-dimensional (2D) cephalogram. In addition, inherent problems of standard lateral cephalograms, including head positioning errors, magnification and distortion, and superimposition of bilateral structures can significantly undermine the accuracy of landmark annotation on these images.

CBCT-based cephalometric analysis still relies heavily on expert landmark annotation despite reduced superimposition and more accurate visualization of anatomy [1]. Most automated cephalometric landmark detection models, however, have been developed for 2D lateral cephalograms [8]. Recent work has shown that deep learning is a promising approach for automated cephalometric landmark detection, but robustness, generalizability, and suficient coverage of clinically relevant landmark sets remain important challenges [18, 7, 6].

Digitally reconstructed radiographs (DRRs) provide a radiograph-like 2D representation of CBCT volumes and can therefore enable established 2D cephalometric landmark detectors to be applied to CBCT-derived data. Therefore, the aim of this study was to evaluate whether skeletal malocclusion classification can be assessed from automatically localized cephalometric landmarks on CBCTderived DRRs. A Vision Transformer-based model, CephViT, was trained on a public lateral cephalogram dataset and applied to DRRs generated from a private CBCT cohort, with downstream skeletal malocclusion classification compared against a reference pipeline based on manual CBCT landmark annotations.

## 2 Materials and Methods

This study comprised two linked components: automatic cephalometric landmark localization and downstream skeletal malocclusion classification. CephViT, a landmark localization network, was developed using the Aariz lateral cephalogram dataset [11, 10] and subsequently applied to DRRs generated from a separate CBCT dataset (Fig. 1). The CBCT dataset included both manual reference landmark annotations and skeletal malocclusion labels, enabling comparison of malocclusion classification using automatically localized versus reference landmarks. Model and accompanying code are available at:

https://huggingface.co/nlm-dir/CephViT,

https://github.com/farrell236/midas-journal-784.

![](images/eefed94231523925978fe4b156f5fa8f45fd4c95735c7c93f252486f53c51e8a.jpg)  
Fig. 1: Overview of the study pipeline. From L-to-R: CBCT volume, lateral DRR, cephalometric landmark localization using CephViT, landmark heatmap output. The localized landmark coordinates were subsequently normalized and used for downstream skeletal malocclusion classification.

## 2.1 Datasets

This retrospective study involved both public and private datasets. The private dataset was deidentified prior to analysis, and its use was approved by the institutional review board with a waiver of informed consent.

The public dataset, Aariz, consisted of 1,000 lateral cephalometric radiographs from patients aged 12–62 years [11, 10]. Images were acquired using seven imaging devices with varying resolutions and annotated with 29 cephalometric landmarks, comprising 15 skeletal, 8 dental, and 6 soft-tissue landmarks. Annotations were made in a two-phase process involving junior and senior orthodontists, with the final ground truth defined as the average of both annotations.

The private CBCT dataset, NIDCR-CARS (IRB #16-D-0040), consisted of 300 scans from 300 patients. Age and sex were available for 297 patients (134 male, 163 female), who were aged 13.6–65.4 years (mean age, 24.2 years ± 12.8). Skeletal landmarks were annotated on the CBCT volumes using a validated 3D annotation protocol [1], with 43 landmarks spanning the face, maxilla, mandible, and cranial base. Each subject was assigned a skeletal malocclusion label of Class I, Class II, Class IIIA, or Class IIIB, yielding an analysis-cohort distribution of 99, 99, 27, and 75 subjects, respectively.

## 2.2 DRR generation from CBCT

DRRs were generated from the CBCT volumes using the Siddon-Jacobs raytracing algorithm [19], which produces radiograph-like projection images by tracing rays from a virtual x-ray source to a detector plane and integrating voxel values along each ray path through the 3D volume. Formally, the DRR intensity at detector location (u, v) can be written as:

$$
D ( u , v ) = \int _ { \mathcal { L } ( u , v ) } \mu ( \mathbf { r } ) d \mathbf { r } ,\tag{1}
$$

where $D ( u , v )$ denotes the projected image intensity, $\mathcal { L } ( u , v )$ is the ray path from the virtual x-ray source to detector coordinate $( u , v )$ , and $\mu ( \mathbf { r } )$ is the voxel intensity at spatial location r in the CBCT volume. DRR generation was performed using the ITK-based Siddon-Jacobs ray-casting implementation described by Wu et al. [21]. Projection images were generated at a resolution of $5 1 2 \times 5 1 2$ pixels with detector pixel spacing of $0 . 5 1 \times 0 . 5 1$ mm and a source-to-isocenter distance of 1000 mm. To suppress contributions from air and low-density background regions, voxels with intensities below -900 were excluded during projection.

## 2.3 Automatic cephalometric landmark localization

Automatic cephalometric landmark localization was performed using a Vision Transformer (ViT)-based heatmap regression model [5]. Given an input image $I \in \mathbb { R } ^ { H \times W \times 3 }$ , the network $f _ { \theta }$ produced a set of output landmark heatmaps, denoted by H<sup>ˆ</sup> :

$$
\begin{array} { r } { \hat { H } = f _ { \theta } ( I ) , \qquad \hat { H } \in \mathbb { R } ^ { N \times h \times w } , } \end{array}\tag{2}
$$

where N denotes the number of landmarks and $h \times w$ is the heatmap resolution. The model used a ViT backbone with $1 6 \times 1 6$ patches and $5 1 2 \times 5 1 2$ pixel inputs. The classification token was discarded, and the remaining patch embeddings were reshaped into a 2D feature map. $\mathrm { ~ A ~ } 1 \times 1$ convolution projected the transformer features to 256 channels, followed by a convolutional decoder with two bilinear upsampling stages and intermediate $3 \times 3$ convolution layers with ReLU activation. A final $1 \times 1$ convolution generated one heatmap per cephalometric landmark. Supervision was formulated as heatmap regression. For each landmark $n ,$ a Gaussian target heatmap $H _ { n }$ was generated as:

$$
H _ { n } ( x , y ) = \exp \left( - \frac { ( x - x _ { n } ) ^ { 2 } + ( y - y _ { n } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) ,\tag{3}
$$

where $( x _ { n } , y _ { n } )$ is the landmark location in heatmap space and σ controls the Gaussian spread. The full set of target heatmaps is denoted by H. The model was trained by minimizing the mean squared error between the output heatmaps H<sup>ˆ</sup> and the target heatmaps H:

$$
\mathcal { L } _ { \mathrm { M S E } } = \mathrm { M S E } ( \hat { H } , H ) .\tag{4}
$$

At inference, landmark coordinates were recovered from the output heatmaps using DARK-style decoding with local refinement [22] and then mapped back to image space for evaluation.

The model was trained for 100 epochs with a batch size of 16 using AdamW, with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ and weight decay of $1 \times 1 0 ^ { - 4 }$ . Gaussian target heatmaps were generated with $\sigma = 2 . 0$ . On-the-fly augmentation included afine transforms (scale, 0.95−1.05; translation, ±3%; rotation, $\pm 8 ^ { \circ }$ ; shear, $\pm 3 ^ { \circ } )$ , random brightness and contrast adjustment, and Gaussian noise. Training was performed on an HPC cluster node equipped with an AMD EPYC 7543P CPU and an NVIDIA A100 GPU, full training completed in approximately 5 hours.

Table 1: Cephalometric landmark detection performance on the Aariz test set.
<table><tr><td rowspan="2">Method</td><td rowspan="2">MRE ± SD (mm)</td><td colspan="4">SDR (%)</td></tr><tr><td>2.0 mm</td><td>2.5 mm</td><td>3.0 mm</td><td>4.0 mm</td></tr><tr><td>Khan et al. [12]</td><td> $1 . 9 2 \pm 7 . 8 5$ </td><td>78.54</td><td>85.72</td><td>89.64</td><td>94.49</td></tr><tr><td>Khalid et al. [9]</td><td> $1 . 8 7 \pm 4 . 0 1$ </td><td>75.17</td><td>82.43</td><td>88.78</td><td>93.01</td></tr><tr><td>Khan et al. [11]</td><td> $1 . 6 9 \pm 3 . 3 6$ </td><td>81.18</td><td>87.28</td><td>90.82</td><td>94.82</td></tr><tr><td>CephViT (ours)</td><td> ${ \bf 1 . 2 8 \pm 1 . 4 2 }$ </td><td>82.10</td><td>87.90</td><td>92.00</td><td>95.70</td></tr></table>

## 2.4 Landmark-based skeletal malocclusion classification

As the reference landmarks were annotated in 3D CBCT space rather than on 2D radiographs, each subject’s 3D landmark configuration was transformed into a common anatomical frame and projected onto a lateral sagittal plane for downstream analysis.

To achieve this, a cranial-base centroid defined the origin, while bilateral craniofacial landmarks and the posterior-to-anterior palatal axis defined consistent left-right, anterior-posterior, and superior-inferior axes. Bilateral landmarks were collapsed to their midpoints where appropriate, and a final Nasion-Sella scale normalization reduced residual inter-subject size variation.

Cephalometric landmarks automatically localized on CBCT-derived DRRs were normalized in 2D by aligning each image-plane landmark configuration to a representative reference shape using a similarity transformation estimated from the 12 common cephalometric landmarks. The transform was then applied to the full localized landmark set before feature extraction.

For direct comparison between the reference and DRR-localized landmark pipelines, skeletal malocclusion classification was performed using coordinate and pairwise-distance features derived from the 12 cephalometric landmarks common to both representations: A, ANS, Ar, B, Go, Me, N, Or, PNS, Po, Pog, and Sella.

## 3 Results

## 3.1 CephViT Landmark Localization Performance

On the held-out Aariz test set, CephViT achieved a mean radial error of 1.28 $\mathrm { m m } \pm 1 . 4 2 $ mm across all 29 landmarks. Successful detection rates were 82.1%, 87.9%, 92.0%, and 95.7% at 2.0, 2.5, 3.0, and 4.0 mm, respectively. Landmarkwise results are summarized in Table 1. Compared with previously reported methods evaluated on the same benchmark split, CephViT achieved competitive localization performance, with lower mean radial error and a smaller error spread, together with higher or comparable successful detection rates across the evaluated thresholds.

## 3.2 Malocclusion Classification from Reference Landmarks

Reference landmark normalization and projection were successful for all 300 subjects, with no exclusions at this stage. The projected landmark configurations formed a coherent anatomical space; Sella, Porion, Articulare, Orbitale, and posterior nasal spine showed the lowest mean radial deviation from the cohort mean shape, whereas Menton, Pogonion, B point, Gonion, and A point showed the greatest variability. A post hoc Tukey-rule screen identified 11 potential shape outliers, although no explicit outlier exclusion was applied in the referencelandmark analysis.

Skeletal malocclusion classification was then performed using the normalized two-dimensional landmark coordinates. Performance was evaluated using a multinomial logistic regression model with mean imputation, feature standardization, balanced class weights, and 5-fold stratified cross-validation, with class-wise metrics computed from out-of-fold predictions.

Using the normalized reference landmark coordinates, skeletal malocclusion classification achieved an overall accuracy of 68.3%. Class-wise F1 scores were 0.667 for Class I, 0.768 for Class II, 0.431 for Class IIIA, and 0.703 for Class IIIB. Performance was strongest for Class II and Class IIIB, whereas Class IIIA showed the lowest precision and F1 score, consistent with greater dificulty in separating this smaller subgroup. The macro-averaged F1 score was 0.642, and the weighted-average F1 score was 0.688 (Table 2).

## 3.3 Malocclusion Classification from DRR-Localized Landmarks

For the DRR-based landmark pipeline, all 300 manifest-common subjects were retained. Automatically localized 2D landmark configurations were aligned to a representative reference configuration using a similarity transformation estimated from the 12 common cephalometric landmarks, with no post-registration case exclusion (Figure 3).

![](images/e336aba4e9c5a19cf7a3c24e00468784252b293b78529efa470897bd86cdffa5.jpg)

![](images/8999de16ca666ef0540496de6f32bbac1e1d6378b055676576fb5eef3d2beef8.jpg)  
Fig. 2: Projection and normalization of CBCT-derived reference landmarks. Left; 3D CBCT landmark configuration and lateral projection onto a 2D cephalometric plane. Right; all projected subjects after 3D anatomical normalization, with mean landmark positions shown in red.

After alignment, the most stable landmarks were posterior nasal spine, A point, Orbitale, Articulare, and Gonion, whereas the most variable were Sella, Nasion, Menton, Porion, and Pogonion. A post hoc Tukey-rule screen of subjectlevel mean landmark deviation identified 36 potential shape outliers in the aligned DRR cohort. Classification used the same stratified 5-fold cross-validation fold assignments as the 3D reference-landmark pipeline to enable a fair paired comparison.

Using the normalized DRR-localized landmark coordinates, skeletal malocclusion classification achieved 70.0% accuracy, macro-F1 of 0.647, and weighted-F1 of 0.708 under 5-fold cross-validation (Table 2). Paired significance testing was performed using an exact McNemar test, while macro- and weighted-F1 diferences were assessed using paired permutation testing with bootstrap 95% confidence intervals. No statistically significant diferences were observed for accuracy (68.3% vs. 70.0%; $p = 0 . 6 9 6 )$ , macro-F1 (0.642 vs. 0.647; $p = 0 . 8 9 4 )$ , or weighted-F1 (0.688 vs. 0.708; $p = 0 . 5 5 5 )$ , supporting comparable performance of the DRR-based method relative to the reference-landmark standard. The small numerical advantage of the DRR-localized pipeline should not be interpreted as superiority and may reflect cross-validation variability or efects of the shared 2D projection.

## 4 Discussion

This study evaluated whether CBCT-derived DRRs could support automated 2D cephalometric landmark localization and downstream skeletal malocclusion classification. This workflow is intended as a practical bridge, not a replacement for native 3D CBCT landmarking. When CBCT volumes are available but no validated 3D model exists for the full 43-landmark reference protocol, DRRs allow established 2D cephalometric landmark localizers to be applied to CBCTderived data for downstream classification. Using the same 300 subjects and identical cross-validation folds, the DRR-localized landmark pipeline achieved performance comparable to the reference pipeline based on manual 3D CBCT landmarks projected to 2D (accuracy, 70.0% vs. 68.3%; macro F1, 0.647 vs. 0.642). These findings suggest that DRR-localized cephalometric landmarks preserve suficient craniofacial geometry for skeletal subclass discrimination.

![](images/03098cd46ef42229e74e68574c112197e287a594060e8e4b1aa137e2dd648029.jpg)

![](images/54696850b7d49e777022fd424343aa3f75b6a00e9761afa32c8af23bfbc18b6e.jpg)  
Fig. 3: 2D normalization of DRR-localized cephalometric landmarks before (left) and after (right) similarity-based alignment to a common coordinate system.

Table 2: Five-fold cross-validated malocclusion classification.
<table><tr><td></td><td>Support</td><td>landmarks</td><td>Reference DRR-localized landmarks</td><td>Δ</td><td>95% CI for ∆</td><td>Paired p</td></tr><tr><td>Accuracy</td><td>300</td><td>0.683</td><td>0.700</td><td></td><td>+0.017 [-0.050, 0.083]</td><td>0.696</td></tr><tr><td>Macro F1</td><td>300</td><td>0.642</td><td>0.647</td><td></td><td>+0.005 [-0.066, 0.074]</td><td>0.894</td></tr><tr><td>Weighted F1</td><td>300</td><td>0.688</td><td>0.708</td><td></td><td>+0.020 [-0.045, 0.084]</td><td>0.555</td></tr><tr><td>Class I</td><td>99</td><td>0.667</td><td>0.691</td><td></td><td>+0.024 [-0.071, 0.119]</td><td></td></tr><tr><td>Class II</td><td>99</td><td>0.768</td><td>0.794</td><td></td><td>+0.026 [-0.052, 0.107]</td><td></td></tr><tr><td>Class IIIA</td><td>27</td><td>0.431</td><td>0.364</td><td></td><td>-0.067 [-0.223, 0.080]</td><td></td></tr><tr><td>Class IIIB</td><td>75</td><td>0.703</td><td>0.740</td><td></td><td>+0.036 [-0.065, 0.139]</td><td></td></tr></table>

Values are computed from out-of-fold predictions. ∆ denotes DRR-localized minus reference-landmark performance. Confidence intervals were estimated using paired bootstrap resampling (n = 10,000). Paired p values were computed using an exact McNemar test for accuracy and paired permutation tests for macro- and weighted-F1 $( n = 1 0 0 , 0 0 0 )$ . Class-wise F1 rows are not separately hypothesis-tested.

Automated extraction of cephalometric geometry from CBCT-derived DRRs could support scalable craniofacial phenotyping or pre-screening, particularly in settings where manual CBCT landmark annotation is impractical. Because manual 2D annotations were not available for the generated DRRs, domain transfer was assessed indirectly through downstream classification rather than direct DRR landmark error. Nevertheless, the DRR-based workflow remains dependent on image quality, landmark localization accuracy, and geometric normalization. Greater landmark dispersion and subject-level shape outliers after alignment indicate that robust localization and registration remain important failure modes to address.

Future work should validate the pipeline in larger external cohorts across CBCT acquisition protocols, fields of view, image quality, and DRR-generation settings, and assess whether image-based or hybrid image-landmark models can further improve diferential classification between skeletal Class III subgroups.

Acknowledgments. This research was supported by the Intramural Research Program of the National Institutes of Health (NIH). The contributions of the NIH authors are considered Works of the United States Government. The findings and conclusions presented in this paper are those of the authors and do not necessarily reflect the views of the NIH or the U.S. Department of Health and Human Services. This work utilized the computational resources of the NIH High-Performance Computing Biowulf cluster (https://hpc.nih.gov).

## References

1. Almpani, K., Adjei, A., Liberton, D.K., Verma, P., Hung, M., Lee, J.S.: Threedimensional cephalometric landmark annotation demonstration on human cone beam computed tomography scans. JoVE (Journal of Visualized Experiments) (199), e65224 (2023)

2. Alrashed, M., Alqerban, A.: The relationship between malocclusion and oral health-related quality of life among adolescents: a systematic literature review and meta-analysis. European Journal of Orthodontics 43(2), 173–183 (2021)

3. Ângelo, D.F., Faria-Teixeira, M.C., Mafia, F., Sanz, D., Sarkis, M., Marques, R., Mota, B., João, R.S., Cardoso, H.J.: Association of malocclusion with temporomandibular disorders: a cross-sectional study. Journal of Clinical Medicine 13(16), 4909 (2024)

4. Björk, A., Skieller, V.: Normal and abnormal growth of the mandible. a synthesis of longitudinal cephalometric implant studies over a period of 25 years. The European Journal of Orthodontics 5(1), 1–46 (1983)

5. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)

6. Gillot, M., Miranda, F., Baquero, B., Ruellas, A., Gurgel, M., Al Turkestani, N., Anchling, L., Hutin, N., Biggs, E., Yatabe, M., et al.: Automatic landmark identification in cone-beam computed tomography. Orthodontics & Craniofacial Research 26(4), 560–567 (2023)

7. Hendrickx, J., Gracea, R.S., Vanheers, M., Winderickx, N., Preda, F., Shujaat, S., Jacobs, R.: Can artificial intelligence-driven cephalometric analysis replace manual tracing? a systematic review and meta-analysis. European Journal of Orthodontics 46(4), cjae029 (2024)

8. Jiang, F., Guo, Y., Yang, C., Zhou, Y., Lin, Y., Cheng, F., Quan, S., Feng, Q., Li, J.: Artificial intelligence system for automated landmark localization and analysis of cephalometry. Dentomaxillofacial Radiology 52(1), 20220081 (2023)

9. Khalid, M.A., Khurshid, A., Zulfiqar, K., Bashir, U., Fraz, M.M.: A two-stage regression framework for automated cephalometric landmark detection incorporating semantically fused anatomical features and multi-head refinement loss. Expert Systems with Applications 255, 124840 (2024)

10. Khalid, M.A., Zulfiqar, K., Bashir, U., Shaheen, A., Iqbal, R., Rizwan, Z., Rizwan, G., Fraz, M.M.: Cepha29: Automatic cephalometric landmark detection challenge 2023. arXiv preprint arXiv:2212.04808 (2022)

11. Khalid, M.A., Zulfiqar, K., Bashir, U., Shaheen, A., Iqbal, R., Rizwan, Z., Rizwan, G., Fraz, M.M.: A benchmark dataset for automatic cephalometric landmark detection and cvm stage classification. Scientific Data 12(1), 1336 (2025)

12. Khan, R., Khalid, M.A., Zulfiqar, K., Bashir, U., Fraz, M.M.: Enhancing cephalometric landmark detection with a two-stage cascaded cnn on multi-resolution multi-modal data. In: Annual conference on medical image understanding and analysis. pp. 3–18. Springer (2024)

13. Koskela, A., Neittaanmäki, A., Rönnberg, K., Palotie, A., Ripatti, S., Palotie, T.: The relation of severe malocclusion to patients’ mental and behavioral disorders, growth, and speech problems. European Journal of Orthodontics 43(2), 159–164 (2021)

14. Lin, Z., Zhou, C., Hu, Z., Zhang, Z., Cheng, Y., Fang, B., He, H., Wang, H., Li, G., Guo, J., et al.: Expert consensus on imaging diagnosis and analysis of early correction of childhood malocclusion. International Journal of Oral Science 17(1), 21 (2025)

15. McNamara Jr, J.A.: A method of cephalometric evaluation. American journal of orthodontics 86(6), 449–469 (1984)

16. Nicot, R., Chung, K., Vieira, A.R., Raoul, G., Ferri, J., Sciote, J.J.: Condyle modeling stability, craniofacial asymmetry and actn3 genotypes: Contribution to tmd prevalence in a cohort of dentofacial deformities. PLoS One 15(7), e0236425 (2020)

17. Nowak, M., Golec, J., Golec, J., Wieczorek, A.: Associations between anteroposterior occlusal class, musculoskeletal pain patterns, and temporomandibular disorders in young adults: a cross-sectional study. Journal of Clinical Medicine 14(23), 8606 (2025)

18. Schwendicke, F., Chaurasia, A., Arsiwala, L., Lee, J.H., Elhennawy, K., Jost-Brinkmann, P.G., Demarco, F., Krois, J.: Deep learning for cephalometric landmark detection: systematic review and meta-analysis. Clinical oral investigations 25(7), 4299–4309 (2021)

19. Siddon, R.L.: Fast calculation of the exact radiological path for a three-dimensional ct array. Medical physics 12(2), 252–255 (1985)

20. Togawa, R., Ohmure, H., Sakaguchi, K., Takada, H., Oikawa, K., Nagata, J., Yamamoto, T., Tsubouchi, H., Miyawaki, S.: Gastroesophageal reflux symptoms in adults with skeletal class iii malocclusion examined by questionnaires. American journal of orthodontics and dentofacial orthopedics 136(1), 10–e1 (2009)

21. Wu, J.: Itk-based implementation of two-projection 2d/3d registration method with an application in patient setup for external beam radiotherapy. The Insight Journal (July–December 2010), https://www.insight-journal.org/ browse/publication/784, handle: 10380/3245

22. Zhang, F., Zhu, X., Dai, H., Ye, M., Zhu, C.: Distribution-aware coordinate representation for human pose estimation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 7093–7102 (2020)

23. Zhou, C., Duan, P., He, H., Song, J., Hu, M., Liu, Y., Liu, Y., Guo, J., Jin, F., Cao, Y., et al.: Expert consensus on pediatric orthodontic therapies of malocclusions in children. International Journal of Oral Science 16(1), 32 (2024)

24. Zhou, X., Chen, S., Zhou, C., Jin, Z., He, H., Bai, Y., Li, W., Wang, J., Hu, M., Cao, Y., et al.: Expert consensus on early orthodontic treatment of class iii malocclusion. International Journal of Oral Science 17(1), 20 (2025)

## Supplementary Material

## Malocclusion Classification

The most common current skeletal classification is based on the position of the maxilla and mandible relative to the anterior cranial base in the sagittal plane [A]. This classification is based on the results of the cephalometric analysis. When both the maxilla and mandible are in positions that are within the normal range of the existing cephalometric measurement normative values, close to the position of the anterior cranial base in the sagittal plane, there is no skeletal malocclusion, and the skeletal classification is Class I (orthognathic). When the maxilla is in a more anterior position and/or the mandible is in a more posterior position in relation to the anterior cranial base, the resulting skeletal malocclusion is classified as Class II, commonly described as “retrognathic” or “overbite”. When opposite relationships exist, with the maxilla in a more posterior position and/or the mandible in a more forward position in relation to the anterior cranial base the resulting malocclusion is classified as Class III, often described as “prognathic” or underbite.

![](images/ff9f9708aa7a7c5f26f965d01e5711c23374220121c441b3350a608a29265619.jpg)  
(a) Class I

![](images/2643683b6f61d5a526310ede6f8054630fe875ac35d80679ddfe207659b91a68.jpg)  
(b) Class II

![](images/c47b5b90b173d145743fb1c6e8c83d04a66024cd7d64b0ca0f82ac79d9c1d381.jpg)  
(c) Class IIIA

![](images/f08646eea520bfb80fad2e7fc9d1dbc21ce92a8fd23efae5b7a2255406ec9208.jpg)  
(d) Class IIIB  
Fig. S1: Craniofacial malocclusion classes

The Class III classification was expanded into two subgroups to more accurately describe the skeletal discrepancy with the use of two separate classifications, depending on the etiology of malocclusion: a Class IIIA classification to describe the cases where the more posterior position of the maxilla is the main contributing factor to the malocclusion, and a Class IIIB classification to describe the case where a more anterior position of the mandible is the main component. This subclassification is clinically significant and contributes to treatment planning decisions. In the case of Class II, the diferential diagnosis related to the contribution of either jaw in the skeletal discrepancy is not as clinically critical, and, therefore, no Class II subgroups were created.

[A] Tang, E. L. K., & Wei, S. H. Y. (1993). Recording and measuring malocclusion: a review of the literature. American Journal of Orthodontics and Dentofacial Orthopedics, 103(4), 344–351.