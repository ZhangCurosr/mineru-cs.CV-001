# Flow Matching Meets 3D Curvilinear Structure Segmentation in Medical Imaging

Sidi Mohamed Sid’El Moctar<sup>[0009−0009−5490−1916]</sup>, Nicolas Vitry and Hélène Bouvrais<sup>[0000−0003−1128−1322]</sup>

CNRS, Univ. Rennes, Institute of Genetics and Development of Rennes (IGDR), Rennes, France

Abstract. Segmentation of curvilinear anatomical structures in 3D medical images remains challenging due to complex topology, severe class imbalance, weak contrast, and large variations in structure morphology. While deep learning approaches for 3D curvilinear segmentation have been proposed, they are often tailored to specific anatomies or modalities, limiting generalization across clinical settings and leaving room for improvement. Recent generative models have shown the benefits of iterative prediction for structured segmentation tasks, yet difusion-based methods sufer from computationally expensive sampling, hindering their use on high-resolution 3D volumes. We present 3D-CurvSegFlow, a flow matching-based model for 3D curvilinear structure segmentation. The model learns a continuous transformation from a simple source distribution to the target vascular representation, enabling progressive refinement of complex curvilinear geometries with eficient inference. We evaluate our method on Three public challenging datasets covering distinct anatomies and modalities: portal vein, cerebral vessel, and coronary arteries. Using a common architecture and training strategy across all tasks, our method outperforms general-purpose and vessel-specific approaches, with strong preservation of thin branches and vascular continuity. This work not only advances the state-of-the-art in 3D curvilinear segmentation but also opens new avenues for eficient, generalizable, and clinically applicable methods in medical image analysis.

Keywords: 3D curvilinear structure · Flow matching · Medical image segmentation.

## 1 Introduction

Accurate segmentation of curvilinear structures in 3D medical images is essential for computational anatomy and computer-aided diagnosis. Portal, hepatic, coronary, and cerebral vessels exhibit complex branching geometries, caliber variations, and intricate topologies [12]. These characteristics are critical for understanding vascular pathologies (e.g. aneurysms, stenosis, Alzheimer’s disease), surgical planning, and minimally invasive treatments [2,12]. Yet automatic segmentation remains dificult due to several factors [12]: (i) Severe class imbalance. (vascular networks typically occupy a small fraction of the image volume), (ii)

Morphological variability across patients (branching patterns, vessel calibers, tortuosity, and spatial organization), (iii) Imaging complexities: (low contrast between vessels and parenchyma, partial volume efects, blurring boundaries, intensity distortions), and (iv) Manual annotation challenges (time-consuming, inter-annotator variability, inconsistent labels). These challenges hinder the generalization of segmentation algorithms trained on specific vascular anatomies, raising the question of whether a unified approach can achieve robust performance across diverse 3D curvilinear structures.

Over recent years, 3D deep learning has advanced curvilinear segmentation by leveraging full volumetric information. The 3D U-Net extended the U-Net model into three dimensions [6], becoming a standard for volumetric medical segmentation. Residual connections enabled deeper models [18,28], and attentionbased methods such as 3D Attention U-Net improved focus on relevant vessel regions while reducing background interference [20]. Transformer-based architectures like UNETR [10] and SwinUNETR [9] further introduced attention to capture long-range dependencies, including some specialized methods in curvilinear structure segmentation like CS<sup>2</sup>-Net [19] and Lv et al. [16]. More recently, Mamba-based architectures have emerged as an eficient alternative to Transformers by modeling long-range dependencies with linear computational complexity [17,15]. However, most methods remain highly specialized, necessitating extensive retraining and limiting generalization. Foundation models have emerged as promising solutions [25]. For example, vesselFM specifically designed for 3D blood vessel segmentation achieved zero-shot generalization across modalities, by training on curated annotations, domain-randomized synthetic data, and flow matching-generated samples [26]. Difusion-based models demonstrate that iterative refinement improves segmentation by progressively denoising predictions [13]. However, they require many sampling steps, incurring substantial computational cost, and rely on stochastic processes that introduce variability. Flow matching ofers a deterministic alternative, learning a vector field that transports an initial distribution to the target through a continuous trajectory [14,23]. This enables eficient inference with few integration steps. Although 2D flow-based methods have shown promise for progressive refinement and preservation of thin structures [21,1], their extension to 3D vascular segmentation—with increased memory and computational demands—remains largely unexplored and generalization in multiple vascular territories has not been systematically evaluated.

In this work, we introduce 3D-CurvSegFlow, a flow matching-based model for robust 3D curvilinear structure segmentation. Our approach formulates segmentation as a continuous time-dependent process, learning a velocity field that progressively transforms a noisy initialization into the final mask, conditioned on the input image. This iterative refinement enables gradual error correction and maintains structural continuity across vessel networks. The model employs a 3D time-conditioned U-Net with attention-gated skip connections and sinusoidal time embeddings. Our main contributions are:(1) The first comprehensive evaluation of conditional flow matching for 3D curvilinear structure segmentation

across multiple vascular territories, demonstrating robust generalization without task-specific modifications; (2) A time-conditioned 3D U-Net architecture with attention gating and a combined training objective that jointly optimizes trajectory learning and segmentation quality; and (3) Extensive experiments on three challenging vascular datasets, showing that iterative refinement through flowbased modeling consistently improves vessel continuity and reduces topological errors, particularly under challenging imaging conditions.

## 2 Methodology

## 2.1 Datasets

To evaluate generalization across anatomical territories and imaging modalities, we select three publicly available 3D datasets encompassing portal veins, cerebral vasculature, and coronary arteries. These datasets exhibit substantial heterogeneity in acquisition protocols, contrast enhancement, vessel caliber distributions, and annotation quality, providing a rigorous testbed for assessing robustness without task-specific adaptations.

3Dircadb (portal vein) [22]: This database comprises 19 contrast-enhanced abdominal CT scans with manual segmentations, from which we target the portal vein. Acquisitions have moderate vessel-to-parenchyma contrast and sparse vascular occupancy, typically less than 2% of the liver volume. Slice thickness ranges from 1 to 4 mm, with 74 to 260 slices per volume and an in-plane resolution of 512 × 512 pixels. We adopt a patient-wise split of 15 volumes for training and 4 for testing.

SMILE-UHURA (brain vessels) [4]: This challenge dataset provides 3D multi-slab time-of-flight MRA data acquired on a 7T Siemens MAGNETOM scanner with an isotropic resolution of 300 µm. The original collection includes 20 subjects from the StudyForrest project, but only the the training-validation dataset of 14 labelled samples was made available. We randomly partition these into 12 training and 2 test volumes. These volumes feature fine-scale Circle of Willis structures and aneurysms, requiring high sensitivity to thin vessels.

ImageCAS (coronary arteries) [29]: This large-scale benchmark for coronary artery segmentation comprises 1000 3D CTA volumes, each from a unique patient, acquired with a Siemens 128-slice dual-source scanner. Volumes have dimensions of 512 × 512 × (206−275) voxels. Coronary vessels exhibit severe caliber variations, with the presence of calcified plaques and motion artefacts. We randomly select 800 volumes for training and 200 for test.

## 2.2 Proposed Model

We formulate 3D curvilinear structure segmentation as a continuous dynamical process. Rather than predicting a mask in a single forward pass, the model learns a time-dependent velocity field that progressively transports a noisy initialization toward the ground truth, conditioned on the input image (Figure 1).

![](images/21e5d7a5812567333d054dafaa743b92cc45fb861d249d11c5d3a3182ee43fc0.jpg)  
Fig. 1. An overview of the proposed 3D-CurvSegFlow model

Let $I \in \mathbb { R } ^ { H \times W \times D }$ denote the input volume and $x _ { 1 } ~ \in ~ \{ 0 , 1 \} ^ { H \times W \times D }$ the ground-truth binary mask. We sample initial noise $x _ { 0 } \sim \mathcal { N } ( 0 , \mathbf { I } )$ independent of I. For continuous time $t \in [ 0 , 1 ]$ , we define a linear interpolation between noise and target:

$$
x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 }\tag{1}
$$

The corresponding target velocity is $\boldsymbol { u } _ { t } = \boldsymbol { x } _ { 1 } - \boldsymbol { x } _ { 0 }$ . The model learns a vector field $\nu _ { \theta } ( x _ { t } , I , t )$ parameterized by $\theta$ to predict the instantaneous change of the evolving mask. We instantiate $\nu _ { \theta }$ as a 3D time-conditioned U-Net with four encoder levels (channels: 32, 64, 128, 256), a bottleneck (512 channels), and a symmetric decoder. Each convolutional block uses $3 \times 3 \times 3$ convolutions, Group Normalization, and SiLU activations. Attention gates are applied to all skip connections to focus on vascular regions and suppress background. The total number of parameters is 23.49 million. Sinusoidal positional embeddings map scalar time t to a 256-dimensional vector, processed through a two-layer MLP and added element-wise to feature maps at each block. This allows the network to adapt its behaviour across reconstruction stages. Segmentation is obtained by integrating the learned dynamics from $t = 0$ to $t = 1$ using explicit Euler with $N = 3$ steps (vs. hundreds for difusion models), followed by a sigmoid $\hat { x } = \sigma ( x _ { N } )$ . For each iteration, we sample $t \sim \mathcal { U } ( 0 , 1 )$ and $x _ { 0 } \sim \mathcal { N } ( 0 , \mathbf { I } )$ , compute $x _ { t }$ and $u _ { t } ,$ , and update:

$$
x _ { n + 1 } = x _ { n } + \varDelta t \nu _ { \theta } ( x _ { n } , I , t _ { n } ) , \quad t _ { n } = n \varDelta t , \quad \varDelta t = 1 / N\tag{2}
$$

We use a composite loss which includes a flow matching loss:

$$
\mathcal { L } _ { \mathrm { F M } } = \frac { 1 } { 2 } \Vert \nu _ { \theta } ( x _ { t } , I , t ) - u _ { t } \Vert ^ { 2 }\tag{3}
$$

To align final reconstruction with ground truth, we apply weighted binary cross-entropy (with class weights $w _ { 1 } = 1 . 0 , w _ { 0 } = 0 . 1 5 )$ and soft Dice loss:

$$
{ \mathcal { L } } _ { \mathrm { W B C E } } = - { \frac { 1 } { V } } \sum [ w _ { 1 } y \log { \hat { x } } + w _ { 0 } ( 1 - y ) \log ( 1 - { \hat { x } } ) ]\tag{4}
$$

$$
{ \mathcal { L } } _ { \mathrm { D i c e } } = 1 - { \frac { 2 \sum y { \hat { x } } + \epsilon } { \sum y + \sum { \hat { x } } + \epsilon } }\tag{5}
$$

The total loss is $: { \mathcal { L } } = \lambda _ { \mathrm { F M } } { \mathcal { L } } _ { \mathrm { F M } } + \lambda _ { \mathrm { W B C E } } { \mathcal { L } } _ { \mathrm { W B C E } } + \lambda _ { \mathrm { D i c e } } { \mathcal { L } } _ { \mathrm { D i c e } }$

The model is optimized using AdamW with a learning rate of $5 \times 1 0 ^ { - 5 }$ and a weight decay of $1 \times 1 0 ^ { - 5 }$ . A cosine annealing scheduler gradually reduces the learning rate to $1 0 ^ { - 6 }$ . During training, data augmentation is applied through random axis flips, afine transformations (scale variation up to 10% and rotations up to $1 0 ^ { \circ } )$ , gamma intensity adjustments, and Gaussian noise injection to improve generalization. Training is performed for up to 700 epochs for 3Dircadb and SMILE-UHURA and up to 200 for ImageCAS, with early stopping based on an exponential moving average (EMA) of the validation loss (α = 0.1).

## 2.3 Metrics

Quantitative evaluation employs six metrics to assess both volumetric overlap and topological fidelity. Dice and Intersection over Union (IoU) serve as primary measures of spatial agreement, with Dice being particularly well-suited for imbalanced vascular segmentation. Precision and recall evaluated the trade-of between over-segmentation and under-segmentation, which is critical for minimizing false positives while maintaining sensitivity. To assess topological preservation, we use centerline Dice (clDice), which quantifies overlap of predicted and ground truth centerlines and is sensitive to broken branches, and 95th percentile Hausdorf Distance (HD95), which provides a robust measure of boundary accuracy that is insensitive to outliers. These topological metrics are essential for curvilinear structures where preserving continuity and branching patterns is clinically more relevant than pixel-wise overlap alone.

## 3 Results and Discussion

We compare the segmentation results of 3D-CurvSegFlow with recent state-ofthe art models. This include D<sup>2</sup>-RD-UNet (2025) for hepatic vessel segmentation [3]; FFCM-MRF (2024) and vsesselFM (2025) for cerebrovascular segmentation [7,26]; and MSFP-Net (2026) for coronary artery segmentation [24] (Table 1). Table 1 also includes the performance of : (i) models from the benchmarks associated with the aforementioned state-of-the-art methods, and (ii) general-purpose models (nnUNet, 3D-UNet and CS2-Net) trained and tested on the same images as 3D-CurvSegFlow.

Importantly, CurvSegFlow employs a single architecture and identical training strategy to segment various types of curvilinear structures across all datasets without anatomy-specific modifications. This allows the evaluation to focus on the generalizability of the proposed flow matching model. 3D-CurvSegFlow consistently achieves the best Dice, clDice and recall scores across all datasets. It also achives the best IoU, precision and HD95 scores for two out of three datasets.

This consistency across portal, cerebral, and coronary vessels is particularly noteworthy because these datasets difer substantially in imaging modality, anatomical complexity, vessel diameter distribution, and annotation characteristics. The results therefore suggest that the proposed formulation captures general properties of curvilinear structures rather than features specific to a particular vascular territory.

Table 1. Quantitative evaluation of vessel segmentation performance across three datasets. We compare 3D-CurvSegFlow to various state-of-the-art models : <sup>∗</sup> Results from [3]; <sup>†</sup> Results from [26]: few-shot task; <sup>‡</sup> Results from [7]; and <sup>•</sup> Results from [24]: 5-fold cross-validation performed. Results are reported as mean across up to six metrics, with best values highlighted in bold.
<table><tr><td>Dataset</td><td>Model</td><td>Dice↑</td><td>IoU↑</td><td>Precision↑ Recall↑ clDice↑ HD95↓</td><td></td><td></td><td></td></tr><tr><td rowspan="7">3Dircadb</td><td rowspan="7">3D U-Net [6] CS2-Net [19] nnU-Net [11] VNet* [18]</td><td>0.36</td><td>0.224</td><td>0.29</td><td>0.49</td><td>0.36</td><td>71.62</td></tr><tr><td>0.40 0.70</td><td>0.259</td><td>0.32</td><td>0.59</td><td>0.41</td><td>40.46</td></tr><tr><td></td><td>0.544</td><td>0.74</td><td>0.67</td><td>0.64</td><td>20.85</td></tr><tr><td>0.61 Att-UNet* [20]</td><td>0.439</td><td>0.66</td><td>0.60</td><td>0.60</td><td>23.59</td></tr><tr><td>0.61</td><td>0.439</td><td>0.64</td><td>0.62</td><td>0.60</td><td>24.43</td></tr><tr><td>[9] 0.61</td><td>0.439</td><td>0.68</td><td>0.60</td><td>0.62</td><td>24.82</td></tr><tr><td>[3] 0.66</td><td>0.493</td><td>0.67</td><td>0.68</td><td>0.68</td><td>23.31</td></tr><tr><td rowspan="8">SMILE-UHURA</td><td>Ours 3D U-Net [6]</td><td>0.74</td><td>0.587</td><td>0.78</td><td>0.71</td><td>0.70</td><td>18.54</td></tr><tr><td>CS2-Net [19]</td><td>0.5121</td><td>0.4330</td><td>0.5243</td><td>0.5116</td><td>0.4815</td><td>5.00</td></tr><tr><td>nnU-Net [11]</td><td>0.5118</td><td>0.4158</td><td>0.5026</td><td>0.5318</td><td>0.4804</td><td>5.70</td></tr><tr><td>GCS [5]</td><td>0.7981</td><td>0.6702</td><td>0.9238</td><td>0.7168</td><td>0.8294</td><td>3.38</td></tr><tr><td></td><td>0.7082</td><td>0.5482</td><td>0.9346</td><td>0.5789</td><td></td><td>56.47</td></tr><tr><td>GMM-MRF [30] FFCM-MRF‡</td><td>0.7644</td><td>0.6188</td><td>0.8166</td><td>0.7370</td><td></td><td>54.19</td></tr><tr><td>[7] [SAM-Med3D† [25]</td><td>0.7706</td><td>0.6267</td><td>0.8416</td><td>0.7273</td><td></td><td>57.17</td></tr><tr><td>[vesselFM† [26]</td><td>0.4659 0.3037</td><td></td><td></td><td></td><td>0.4463</td><td></td></tr><tr><td rowspan="8">ImageCAS</td><td>Ours</td><td>0.5815</td><td>0.4100</td><td></td><td></td><td>0.4272 0.8418</td><td></td></tr><tr><td>3D U-Net [6]</td><td>0.8032</td><td>0.6763</td><td>0.8780</td><td>0.7734</td><td></td><td>3.87</td></tr><tr><td>nnU-Net [11]</td><td>0.7231 0.8059</td><td>0.5718</td><td>0.6464</td><td>0.8250</td><td>0.7549</td><td>75.16</td></tr><tr><td>VNet• [18]</td><td>0.7305</td><td>0.6783</td><td>0.7866</td><td>0.8325</td><td>0.8721</td><td>19.58</td></tr><tr><td>CAS-Net• [8]</td><td>0.7077</td><td>0.575</td><td>0.7275</td><td>0.7669</td><td></td><td>23.01</td></tr><tr><td>UNETR• [i0]</td><td>0.7375</td><td>0.548 0.584</td><td>0.7419 0.7121</td><td>0.7128 0.7744</td><td></td><td>31.87</td></tr><tr><td>Swin-UNETR• [9]</td><td>0.7454</td><td>0.594</td><td>0.7189</td><td>0.7817</td><td></td><td>21.96</td></tr><tr><td>SegFormer• [27]</td><td>0.7315</td><td></td><td></td><td>0.7532</td><td></td><td>20.57 18.76</td></tr><tr><td rowspan="4"></td><td>MSFP-Net• [24]</td><td></td><td>0.577</td><td>0.7234 0.7974</td><td>0.7924</td><td></td><td>11.25</td></tr><tr><td>Ours</td><td>0.7923 0.656</td><td></td><td>0.8020</td><td>0.850</td><td>0.8793</td><td>22.62</td></tr><tr><td></td><td>0.824 0.701</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

The improvements are especially pronounced in terms of the recall-precision balance and clDice (altough no comparison done was performed on imageCAS for the latter). This observation is clinically important because vascular segmentation errors rarely originate from inaccurate delineation of large vessels. Instead, they mainly arise from missed peripheral branches and broken vessel connectivity. The higher recall indicates that the proposed model detects a larger proportion of small vessels, while the hight precision reflects that this improved detection does not come at the cost of increased false positives. Competitive clDice scores further demonstrate that these additional detections preserve the continuity of the vascular tree, rather than introducing fragmented predictions. These results are consistent with the design of the proposed model, where segmentation is formulated as a continuous transport process instead of a singlestep voxel classification problem. During integration, each refinement step takes advantage of both the image appearance and the evolving segmentation state, enabling the network to progressively correct incomplete vessel trajectories and recover thin ambiguous branches.

![](images/972251b44fd374cbd670b7a540a6dec516eee71d3ffff2b1bccb4f6a223cda45.jpg)  
Fig. 2. Qualitative comparison of 3D-CurvSegFlow, nnU-Net, CS<sup>2</sup>-Net and 3D U-Net on SMILE-UHURA, 3Dircadb, and ImageCAS. The orange circles highlight regions where predictions are incorrect.

The qualitative examples shown in Figure 2 further support the quantitative evaluation. The proposed method generates more continuous vascular trees with fewer disconnected distal branches. These diferences are particularly visible in regions afected by weak contrast or imaging artifacts, where competing methods frequently interrupt vessel trajectories or produce false positives, while 3D-CurvSegFlow maintains anatomical continuity and accurate segmentation. A similar trend is observed for cerebral vessels, where numerous fine branches recovered by the proposed model remain absent in competing predictions. Although these local improvements may represent only a limited number of voxels, they have a disproportionately large impact on vascular topology and may be clinically important for downstream applications such as surgical planning, vessel quantification, and computational vascular analysis. Notably, the annotations in the SMILE-UHURA dataset appear wider than the vessels visible in the images. Interestingly, 3D-CurvSegFlow’s predictions more accurately capture the true vessel width, outperforming the manual annotations and illustrating its enhanced capabilities. The proposed method consistently outperforms existing approaches across datasets, demonstrating robust generalization under varying imaging conditions. Improvements are observed in challenging scenarios characterized by severe class imbalance, thin vessels, and imaging artifacts. This suggests that modeling segmentation as a continuous flow better preserves vessel continuity and topology than conventional one-step prediction.

However, some limitations should be acknowledged. First, the evaluation is performed on three vascular datasets acquired using contrast-enhanced CT or high-resolution TOF-MRA. The model’s behavior on low- or non-contrast modalities remains untested. Second, training relies on fixed-size volumetric patches, which may restrict access to global anatomical context for very large vascular trees. Third, although the proposed method substantially improves topological preservation, occasional distal over-segmentation still afects boundarybased metrics such as HD95 on coronary CTA. Finally, we demonstrate computational eficiency by using only three Euler integration steps, achieving 22.3 s per volume on SMILE-UHURA during inference with a general Dice score of 0.8032. Testing 5 and 10 inference steps not only increases inference time but also degrades prediction accuracy: 5 steps: 36.4 s, Dice 0.773; 10 steps: 71.9 s, Dice 0.753. This reveals that 3 inference steps are suficient and orders of magnitude more eficient than the hundreds of steps typically required by difusion models. However, additional investigation of alternative ODE solvers and adaptive integration strategies may further improve the accuracy-inference speed trade-of.

## 4 Conclusion

This paper introduces 3D-CurvSegFlow, a flow matching-based framework for robust 3D segmentation of curvilinear anatomical structures. By formulating segmentation as a continuous dynamical process, the proposed method learns a time-conditioned vector field that progressively refines the segmentation while preserving vascular topology using only a few integration steps. Experiments on three public datasets covering diferent anatomies and imaging modalities demonstrate consistent improvements over other methods, including vessel-specifi methods (e.g. D<sup>2</sup>-RD-UNet, CS<sup>2</sup>-Net, or MSFP-Net), and recent foundation models (e.g. VesselFM). These results highlight the robustness and generalization capability of the proposed model and suggest that conditional flow matching provides an efective and computationally eficient paradigm for 3D curvilinear structure segmentation. In future work, we will investigate its extension to additional imaging modalities, larger datasets, and adaptive integration strategies to further improve scalability and boundary accuracy in complex clinical scenarios.

## References

1. Asadi, B., Wu, P., Golparvar-Fard, M., Shah, V., Hajj, R.: Fms<sup>2</sup>: Unified flow matching for segmentation and synthesis of thin structures. arXiv preprint arXiv:2603.13659 (2026)

2. Asrani, S.K., Devarbhavi, H., Eaton, J., Kamath, P.S.: Burden of liver diseases in the world. Journal of hepatology 70(1), 151–171 (2019)

3. Cavicchioli, M., Moglia, A., Garret, G., Puglia, M., Vacavant, A., Pugliese, G., Cerveri, P.: D2-rd-unet: A dual-stage dual-class framework with connectivity correction for hepatic vessels segmentation. Computers in Biology and Medicine 195, 110530 (2025)

4. Chatterjee, S., Mattern, H., Dörner, M., Sciarra, A., Dubost, F., Schnurre, H., Khatun, R., Yu, C.C., Hsieh, T.L., Tsai, Y.S., et al.: Smile-uhura challenge–small vessel segmentation at mesoscopic scale from ultra-high resolution 7t magnetic resonance angiograms. arXiv preprint arXiv:2411.09593 (2024)

5. Chen, C., Zhou, K., Wang, Z., Xiao, R.: Generative consistency for semi-supervised cerebrovascular segmentation from tof-mra. IEEE Transactions on Medical Imaging 42(2), 346–353 (2022)

6. Çiçek, Ö., Abdulkadir, A., Lienkamp, S.S., Brox, T., Ronneberger, O.: 3d u-net: learning dense volumetric segmentation from sparse annotation. In: International conference on medical image computing and computer-assisted intervention. pp. 424–432. Springer (2016)

7. Cui, Y., Huang, H., Liu, J., Zhao, M., Li, C., Han, X., Luo, N., Gao, J., Yan, D.M., Zhang, C., et al.: Ffcm-mrf: An accurate and generalizable cerebrovascular segmentation pipeline for humans and rhesus monkeys based on tof-mra. Computers in Biology and Medicine 170, 107996 (2024)

8. Dong, C., Xu, S., Dai, D., Zhang, Y., Zhang, C., Li, Z.: A novel multi-attention, multi-scale 3d deep network for coronary artery segmentation. Medical Image Analysis 85, 102745 (2023)

9. Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H.R., Xu, D.: Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images. In: International MICCAI brainlesion workshop. pp. 272–284. Springer (2021)

10. Hatamizadeh, A., Tang, Y., Nath, V., Yang, D., Myronenko, A., Landman, B., Roth, H.R., Xu, D.: Unetr: Transformers for 3d medical image segmentation. In: Proceedings of the IEEE/CVF winter conference on applications of computer vision. pp. 574–584 (2022)

11. Isensee, F., Jaeger, P.F., Kohl, S.A., Petersen, J., Maier-Hein, K.H.: nnu-net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods 18(2), 203–211 (2021)

12. Jin, Y., Pepe, A., Li, J., Gsaxner, C., Chen, Y., Puladi, B., Zhao, F.h., Pomykala, K., Kleesiek, J., Frangi, A.F., et al.: Aortic vessel tree segmentation for cardiovascular diseases treatment: Status quo. ACM Computing Surveys 57(9), 1–35 (2025)

13. Kazerouni, A., Aghdam, E.K., Heidari, M., Azad, R., Fayyaz, M., Hacihaliloglu, I., Merhof, D.: Difusion models for medical image analysis: A comprehensive survey. arXiv preprint arXiv:2211.07804 (2022)

14. Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022)

15. Liu, T., Zhang, Z., Fan, G., Li, B., Zhou, S., Xu, C., Zhao, G., Yang, F.: Mambavesselnet: a novel approach to blood vessel segmentation based on state-space models. IEEE Journal of Biomedical and Health Informatics 29(3), 2034–2047 (2024)

16. Lv, J., Zhang, L., Xu, J., Li, W., Li, G., Zhou, H.: Automatic segmentation of mandibular canal using transformer based neural networks. Frontiers in Bioengineering and Biotechnology 11, 1302524 (2023)

17. Ma, J., Li, F., Wang, B.: U-mamba: Enhancing long-range dependency for biomedical image segmentation. arXiv preprint arXiv:2401.04722 (2024)

18. Milletari, F., Navab, N., Ahmadi, S.A.: V-net: Fully convolutional neural networks for volumetric medical image segmentation. In: 2016 fourth international conference on 3D vision (3DV). pp. 565–571. Ieee (2016)

19. Mou, L., Zhao, Y., Fu, H., Liu, Y., Cheng, J., Zheng, Y., Su, P., Yang, J., Chen, L., Frangi, A.F., et al.: Cs2-net: Deep learning segmentation of curvilinear structures in medical imaging. Medical image analysis 67, 101874 (2021)

20. Oktay, O., Schlemper, J., Folgoc, L.L., Lee, M., Heinrich, M., Misawa, K., Mori, K., McDonagh, S., Hammerla, N.Y., Kainz, B., et al.: Attention u-net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999 (2018)

21. Sid’El Moctar, S.M., Ait Laydi, A., Beber, A., Braun, M., Lansky, Z., El Mourabit, Y., Bouvrais, H.: Curvsegflow: Time-conditioned flow matching for robust segmentation of curvilinear structures in noisy biomedical images. arXiv preprint arXiv:2606.21608 (2026)

22. Soler, L., Hostettler, A., Agnus, V., Charnoz, A., Fasquel, J.B., Moreau, J., Osswald, A.B., Bouhadjar, M., Marescaux, J.: 3d image reconstruction for comparison of algorithm database. URL: https://www. ircad. fr/research/data-sets/liversegmentation-3d-ircadb-01 13, 4 (2010)

23. Tong, A., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Fatras, K., Wolf, G., Bengio, Y.: Conditional flow matching: Simulation-free dynamic optimal transport. arXiv preprint arXiv:2302.00482 2(3) (2023)

24. Wang, A., Wang, L., Xu, L., Liu, J., Sun, Y., Zhang, L.: Msfp-net: a multi-scale and multi-frequency enhanced 3d network with learnable priors for coronary artery segmentation in ccta images. Biomedical Signal Processing and Control 115, 109366 (2026)

25. Wang, H., Guo, S., Ye, J., Deng, Z., Cheng, J., Li, T., Chen, J., Su, Y., Huang, Z., Shen, Y., et al.: Sam-med3d: a vision foundation model for general-purpose segmentation on volumetric medical images. IEEE Transactions on Neural Networks and Learning Systems (2025)

26. Wittmann, B., Wattenberg, Y., Amiranashvili, T., Shit, S., Menze, B.: vesselfm: A foundation model for universal 3d blood vessel segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20874–20884 (2025)

27. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M., Luo, P.: Segformer: Simple and eficient design for semantic segmentation with transformers. Advances in neural information processing systems 34, 12077–12090 (2021)

28. Yu, W., Fang, B., Liu, Y., Gao, M., Zheng, S., Wang, Y.: Liver vessels segmentation based on 3d residual u-net. In: 2019 IEEE international conference on image processing (ICIP). pp. 250–254. IEEE (2019)

29. Zeng, A., Wu, C., Lin, G., Xie, W., Hong, J., Huang, M., Zhuang, J., Bi, S., Pan, D., Ullah, N., et al.: Imagecas: A large-scale dataset and benchmark for coronary artery segmentation based on computed tomography angiography images. Computerized Medical Imaging and Graphics 109, 102287 (2023)

30. Zhang, B., Wu, Z., Liu, S., Zhou, S., Li, N., Zhao, G.: A device-independent novel statistical modeling for cerebral tof-mra data segmentation. In: Workshop on Clinical Image-Based Procedures. pp. 172–181. Springer (2019)