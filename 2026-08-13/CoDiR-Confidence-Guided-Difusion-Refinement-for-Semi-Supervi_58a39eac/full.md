# CoDiR: Confidence-Guided Difusion Refinement for Semi-Supervised Histopathology Segmentation

Hoai Nhan Pham<sup>1⋆</sup>, Dang-Nguyen Bui<sup>2⋆</sup>, Le-Van Thai<sup>1⋆</sup>, Thanh-Hiep Vo<sup>3</sup>, Lan Anh Dinh Thi<sup>4</sup>, Tien Dat Nguyen<sup>1</sup>, Duy-Dong Nguyen<sup>1</sup>, Ngoc Lam Quang Bui<sup>5</sup>, Tam Tran<sup>6(B)</sup> , and Zhi Huang<sup>7(B)</sup>

<sup>1</sup> AI VIETNAM Lab, Vietnam

<sup>2</sup> Portland Community College, USA

3 University of Science, VNU-HCM, Ho Chi Minh City, Vietnam <sup>4</sup> Hanoi University of Science and Technology, Vietnam

5 Department of Mechanical System Engineering, Jeonbuk National University, Republic of Korea

<sup>6</sup> John T. Milliken Department of Medicine, Washington University School of Medicine, Saint Louis, MO, USA

tamt@wustl.edu

<sup>7</sup> Perelman School of Medicine, University of Pennsylvania, USA zhi.huang@pennmedicine.upenn.edu

Abstract. Semi-supervised histopathology segmentation is challenging due to scarce annotations and unreliable pseudo-labels in ambiguous gland regions. To address this problem, we propose Confidence-Guided Difusion Refinement (CoDiR), a semi-supervised framework that combines a Mean Teacher segmentation model with difusion-based pseudolabel refinement. Given an unlabeled image, the teacher first produces a soft prediction, and only low-confidence regions are refined by a conditional difusion model trained to capture plausible mask structures from labeled data. The refined mask is then fused with reliable teacher predictions and used to train the student with confidence weighting and consistency regularization. On the GlaS and CRAG datasets CoDiR reaches 88.09% and 89.83% mDice with 10% labeled data, and 89.19% and 90.29% mDice with 20%, matching or exceeding the strongest published method on seven of the eight benchmark metrics. Ablations attribute the largest single contribution to the refinement module, which adds +6.36% mDice over the Mean Teacher baseline. The implementation code is publicly available at: https://github.com/vongla345/codir

Keywords: Semi-supervised learning, Histopathology segmentation, Diffusion models, Pseudo-label refinement, Foundation models, Mean Teacher.

## 1 Introduction

Medical image segmentation plays a crucial role in computer-aided diagnosis and pathological analysis by enabling accurate delineation of anatomical structures

Equal contribution

[12,17]. However, acquiring pixel-level annotations for histopathology images is costly and labor-intensive, requiring expert pathologists to label complex tissue structures with high morphological variability and staining-induced appearance changes [18,8,25].

Semi-supervised learning (SSL) alleviates annotation costs by leveraging abundant unlabeled data alongside limited labeled samples [6]. Teacher–student frameworks based on pseudo-labeling and consistency regularization have achieved strong performance in medical image segmentation [21,26,4,22]. However, inaccurate pseudo-labels may introduce structural and boundary errors that are repeatedly reinforced during training, leading to confirmation bias and degraded segmentation quality [1,11].

Meanwhile, pathology foundation models such as UNI [3] have demonstrated strong transferability across downstream pathology tasks. Building on these pretrained representations, our earlier work, UniSemAlign [24], introduced prototype and text-guided semantic alignment for semi-supervised histopathology segmentation. While it improves representation quality and feature discrimination, it constrains pseudo-labels only indirectly through feature alignment and does not explicitly refine erroneous mask regions.

To address these limitations, we propose Confidence-Guided Difusion Refinement (CoDiR), a semi-supervised histopathology segmentation framework that leverages pathology foundation representations and difusion-guided pseudo-label refinement. By selectively correcting unreliable pseudo-label regions, CoDiR provides more reliable supervision for unlabeled data. Extensive experiments on the CRAG and GlaS benchmarks demonstrate the efectiveness of the proposed approach.

Our main contributions are as follows:

– We propose CoDiR, a semi-supervised histopathology segmentation framework that combines a frozen UNI encoder, a DeepLabV3-style decoder [2], and Mean Teacher learning [23]. Where our earlier UniSemAlign framework [24] improves pseudo-label quality indirectly through semantic alignment, CoDiR acts on the mask itself, correcting unreliable regions with a learned structural prior.

– We introduce a confidence-gated difusion refinement module that selectively corrects low-confidence regions while preserving reliable teacher predictions. Compared with DifRect [11] and SDEdit-style editing [14], our approach refines only uncertain pixels with a conditional difusion prior, reducing confirmation bias and improving boundary quality.

– We evaluate CoDiR on the GlaS and CRAG benchmarks, where it achieves consistent improvements over recent semi-supervised methods.

## 2 Methodology

Overview. We propose a semi-supervised histopathology segmentation framework that integrates a Mean Teacher segmentation model with a difusion-based pseudo-label refinement module. As illustrated in Fig. 1, the framework consists of a segmentation network trained under the Mean Teacher paradigm [23] and a difusion model [7]. The segmentation model is supervised using labeled data and learns from unlabeled data via teacher-generated pseudo-labels. Meanwhile, the difusion model is trained on labeled data to model the conditional distribution $p ( y | I )$ of segmentation masks. During training, the teacher produces soft pseudo-masks for unlabeled images, and only low-confidence regions are refined by the difusion model before being fused with high-confidence predictions to form the final pseudo-label $y _ { r }$

![](images/89ee15c61c3ac447c0cebeffcbb4d4ace50dc7cc8c1b248eab1a8aaf5963a149.jpg)  
Fig. 1. Overview of CoDiR. The student segmentation network is trained with labeled supervision and pseudo-labels generated by an EMA teacher. Meanwhile, the difusion model is trained on labeled data and used within the same training loop to refine lowconfidence regions of teacher pseudo-masks. The refined outputs are then fused with high-confidence teacher predictions to produce the final pseudo-labels for unlabeled learning.

## 2.1 Segmentation Model

Visual Encoder. Given an input image $I \in \mathbb { R } ^ { B \times 3 \times H \times W }$ , a pretrained UNI ViT-L/16 [3] encoder extracts patch-level visual features:

$$
\mathbf { F } _ { i m g } = \mathrm { U N I } ( I ) \in \mathbb { R } ^ { B \times N \times d _ { i m g } }\tag{1}
$$

where $\begin{array} { r } { N = \frac { H } { 1 6 } \times \frac { W } { 1 6 } } \end{array}$ and $d _ { i m g } = 1 0 2 4$ . The CLS token is discarded, and the remaining patch tokens are reshaped into a 2D feature grid.

Decoder. The feature $\mathbf { F } _ { i m g }$ passed into a DeepLabV3-style decoder [2], which aggregates multi-scale context using ASPP and upsamples the features to produce a per-pixel segmentation logit map $\mathbf { S } \in \mathbb { R } ^ { B \times \mathbf { \hat { 1 } } \times H \times \mathbf { \hat { W } } }$

Mean Teacher. The framework adopts a Teacher-Student paradigm [23]. The Student $f _ { \theta }$ is updated via gradient descent, while the Teacher $f _ { \bar { \theta } }$ maintains an exponential moving average (EMA) of the Student weights:

$$
\bar { \theta }  \alpha \bar { \theta } + ( 1 - \alpha ) \theta\tag{2}
$$

where α is the EMA decay coeficient. The Teacher parameters are not directly optimized and serve solely as a pseudo-label generator.

## 2.2 Difusion-based Refinement

Teacher predictions provide useful supervision, but they are often unreliable in ambiguous regions and may propagate confirmation bias when used directly as pseudo-labels [11,1]. To address this issue, we introduce a difusion-based refinement module that selectively corrects low-confidence regions while preserving reliable predictions.

Difusion model training. The difusion model $\epsilon _ { \phi }$ is trained exclusively on the labeled subset $\mathcal { D } _ { l }$ to model the conditional mask distribution $p ( y \mid I )$ . Following the DDPM formulation [7], it learns to denoise corrupted segmentation masks conditioned on the input image, thereby capturing a structural prior over plausible object shapes. To mitigate overfitting under extreme label scarcity, we use a low-capacity mini-UNet and difusion-induced noise regularization; refinement is further confined to low-confidence regions to avoid corrupting reliable structures and suppress hallucinations.

Confidence-guided refinement. For an unlabeled image $I _ { u } ,$ the teacher predicts a soft pseudo-mask

$$
\hat { y } = \sigma ( f _ { \bar { \theta } } ( \mathcal { A } _ { w } ( I _ { u } ) ) ) \in [ 0 , 1 ] ,\tag{3}
$$

with confidence

$$
c _ { \mathrm { t e a c h e r } } = \operatorname* { m a x } ( \hat { y } , 1 - \hat { y } ) ,\tag{4}
$$

where low-confidence regions are identified by an adaptive threshold τ following prior confidence-based pseudo-labeling strategies [27].

To refine these regions, we adopt a partial noising strategy inspired by SDEdit [14]. The pseudo-mask is perturbed to an intermediate noise level $\alpha _ { \mathrm { n } }$ to obtain $\tilde { y } _ { \alpha _ { \mathrm { n } } }$ then denoised for K DDPM reverse steps conditioned on $I _ { u } \colon$

$$
y _ { \mathrm { d i f f } } = \mathrm { D e n o i s e } _ { K } ( \tilde { y } _ { \alpha _ { \mathrm { n } } } , I _ { u } ; \epsilon ) .\tag{5}
$$

The confidence map is thresholded to produce a binary refinement mask $m ,$ and the refined pseudo-mask is computed as

$$
y _ { r } = m \odot y _ { \mathrm { d i f f } } + ( 1 - m ) \odot \hat { y } ,\tag{6}
$$

where $m \in \{ 0 , 1 \} ^ { H \times W }$ denotes pixels with $c _ { \mathrm { t e a c h e r } } < \tau$

Confidence weighting. To suppress noisy supervision, each pixel is assigned a confidence weight based on the agreement between the teacher and difusion outputs [10]:

$$
w = c _ { \mathrm { t e a c h e r } } \cdot c _ { \mathrm { d i f f u s i o n } } ^ { \mathrm { e f f } } ,\tag{7}
$$

with

$$
c _ { \mathrm { d i f f u s i o n } } ^ { \mathrm { e f f } } = m \odot \operatorname* { m a x } ( y _ { \mathrm { d i f f } } , 1 - y _ { \mathrm { d i f f } } ) + ( 1 - m ) \odot c _ { \mathrm { t e a c h e r } } .\tag{8}
$$

Thus, difusion confidence is used in refined regions, while teacher confidence is retained elsewhere, so that reliable pixels contribute more strongly to the unsupervised loss.

## 2.3 Training Objectives

The overall objective is defined as

$$
\mathcal { L } = \frac { 1 } { 2 } \left( \mathcal { L } _ { \mathrm { s u p } } + \mathcal { L } _ { \mathrm { u n s u p } } \right)\tag{9}
$$

The supervised loss is computed on labeled data, while the unsupervised loss is applied to unlabeled samples using difusion-refined pseudo-labels.

Supervised loss. For labeled images, the binary student prediction S is supervised with the ground-truth mask using a combination of pixel-wise and region-level losses:

$$
{ \mathcal { L } } _ { \mathrm { s u p } } = \lambda _ { 1 } { \mathcal { L } } _ { \mathrm { C E } } + \lambda _ { 2 } { \mathcal { L } } _ { \mathrm { D i c e } } + \lambda _ { 3 } { \mathcal { L } } _ { \mathrm { c l D i c e } } + \lambda _ { 4 } { \mathcal { L } } _ { \mathrm { B o u n d a r y } }\tag{10}
$$

where $\mathcal { L } _ { \mathrm { c l D i c e } }$ [19] preserves gland topology, and $\mathcal { L } _ { \mathrm { B o u n d a r y } }$ [9] improves boundary accuracy.

Unsupervised loss. For unlabeled images, the student learns from the refined pseudo-mask $y _ { r }$ under confidence weighting w:

$$
\mathcal { L } _ { \mathrm { u n s u p } } ^ { \mathrm { r e f i n e } } = \mathbb { E } [ w \cdot ( \mathcal { L } _ { \mathrm { B C E } } ( \hat { y } _ { u } ^ { w } , y _ { r } ) + \mathcal { L } _ { \mathrm { D i c e } } ( \hat { y } _ { u } ^ { w } , y _ { r } ) ) ]\tag{11}
$$

To encourage augmentation-invariant representations, a consistency loss is imposed between weakly and strongly augmented student predictions:

$$
\mathcal { L } _ { \mathrm { u n s u p } } ^ { \mathrm { c o n s } } = \mathcal { L } _ { \mathrm { B C E } } ( \hat { y } _ { u } ^ { s } , \hat { y } _ { u } ^ { w } ) + \mathcal { L } _ { \mathrm { D i c e } } ( \hat { y } _ { u } ^ { s } , \hat { y } _ { u } ^ { w } )\tag{12}
$$

The full unsupervised objective is

$$
\mathcal { L } _ { \mathrm { u n s u p } } = \lambda _ { u } \mathcal { L } _ { \mathrm { u n s u p } } ^ { \mathrm { r e f i n e } } + \lambda _ { c } \mathcal { L } _ { \mathrm { u n s u p } } ^ { \mathrm { c o n s } }\tag{13}
$$

## 3 Experiments

## 3.1 Datasets

We evaluate the proposed framework on the GlaS [20] and CRAG [5] histopathology gland segmentation benchmarks, containing 165 and 213 images, respectively. For semi-supervised learning, 10% and 20% of the training data are randomly selected as labeled, with the remainder used as unlabeled. Main results use a fixed labeled split, while ablation studies report the mean ± std over three random seeds.

## 3.2 Implementation Details

All experiments are conducted on a single NVIDIA RTX PRO 4000 Blackwell GPU (24GB). Images are resized to 256×256 for training, while inference is performed on non-overlapping patches and stitched to reconstruct the full-resolution mask. Models are trained for 100 epochs using AdamW $( 1 \times 1 0 ^ { - 4 }$ , batch size 16), with EMA teacher updates $( \alpha = 0 . 9 9 )$ . We set $\lambda _ { 1 } = \lambda _ { 2 } = 1 . 0 , \lambda _ { 3 } = \lambda _ { 4 } = 0 . 5$ $\lambda _ { u } = \lambda _ { c } = 0 . 2 5$ , and use a 10-epoch warmup.

For difusion-based refinement, we use $T ~ = ~ 1 0 0$ timesteps with a partial noising ratio of $\alpha _ { \mathrm { n } } = 0 . 2$ . Low-confidence pixels are identified using a FreeMatchstyle adaptive threshold, where $c = \operatorname* { m a x } ( p , 1 - p )$ . The threshold τ is updated as an EMA of the batch mean of the maximum teacher confidence for each image and clamped to $[ \tau _ { \mathrm { m i n } } , \tau _ { \mathrm { m a x } } ] = [ 0 . 6 0 , 0 . 7 5 ]$ , where $\tau _ { \mathrm { m i n } }$ and $\tau _ { \mathrm { m a x } }$ prevent overly conservative and aggressive refinement, respectively. At inference, we use a fixed threshold of $\tau = 0 . 7 5$ . During inference, only the student network is used, and performance is evaluated using mDice and mJaccard.

## 3.3 Comparison with State of the Art

Quantitative Results. As shown in Table 1, CoDiR achieves the best result on seven of the eight benchmark metrics. Relative to the strongest competing method, our earlier UniSemAlign framework [24], CoDiR improves by +1.26% in mDice and +2.94% in mJaccard on CRAG under 10% labeling, and is on par on GlaS (−0.06% in mDice, +0.59% in mJaccard). Under 20% labeling the gains are +0.61% in mDice and +1.59% in mJaccard on GlaS, and $+ 0 . 8 9 \%$ in mDice and +2.05% in mJaccard on CRAG. The improvement is consistently larger on CRAG, whose glands are larger and less regular, so that pseudo-labels there contain more structural error for the refinement step to correct.

Qualitative Results. Fig. 2 compares predictions under the 10% labeling setting. CoDiR delineates gland boundaries with fewer spurious foreground regions than the baselines and retains narrow structures that competing methods break apart, yielding masks closer to the ground truth. Both efects are concentrated in the ambiguous areas that the refinement step targets.

Table 1. Comparison with state-of-the-art semi-supervised segmentation methods on the GlaS and CRAG datasets. The best performance is highlighted in bold, and the second-best among all compared methods is underlined. Baseline results are taken from UniSemAlign [24], which uses the same experimental protocol. The last row reports our proposed method.
<table><tr><td rowspan="2">Labeled Ratio</td><td rowspan="2">Method</td><td colspan="2">GlaS</td><td colspan="2">CRAG</td></tr><tr><td>mDice (%)</td><td>mJaccard (%)</td><td>mDice (%)</td><td>mJaccard (%)</td></tr><tr><td>100%</td><td>Fully-Supervised</td><td>89.59</td><td>81.84</td><td>92.68</td><td>86.36</td></tr><tr><td rowspan="11">10%</td><td>UAMT (MICCAI&#x27;19) [26]</td><td>78.57</td><td>64.70</td><td>77.85</td><td>63.74</td></tr><tr><td>FixMatch (NeurIPS&#x27;20) [21]</td><td>66.01</td><td>51.39</td><td>72.42</td><td>64.31</td></tr><tr><td>CPS (CVPR’21) [4]</td><td>61.70</td><td>49.34</td><td>47.10</td><td>40.77</td></tr><tr><td>CT (MIDL’22) [13]</td><td>81.75</td><td>70.44</td><td>77.33</td><td>65.56</td></tr><tr><td>XNet (ICCV’23) [28]</td><td>74.13</td><td>58.90</td><td>65.27</td><td>50.40</td></tr><tr><td>CorrMatch (CVPR’24) [22]</td><td>85.54</td><td>74.74</td><td>79.93</td><td>66.76</td></tr><tr><td>DuSSS (AAAI&#x27;25) [15]</td><td>75.07</td><td>61.46</td><td>64.25</td><td>49.99</td></tr><tr><td>CSDS (MICCAI-W&#x27;25) [16]</td><td>82.89</td><td>71.61</td><td>79.86</td><td>67.81</td></tr><tr><td>UniSemAlign (CVPRW&#x27;26) [24]</td><td>88.15</td><td>78.82</td><td>88.57</td><td>79.52</td></tr><tr><td>CoDiR (Ours)</td><td>88.09</td><td>79.41</td><td>89.83</td><td>82.46</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="11">20%</td><td>UAMT (MICCAI&#x27;19) [26]</td><td>77.09</td><td>62.72</td><td>79.76</td><td>66.33</td></tr><tr><td>FixMatch (NeurIPS&#x27;20) [21]</td><td>58.51</td><td>44.12</td><td>71.20</td><td>64.75</td></tr><tr><td>CPS (CVPR’21) [4]</td><td>75.99</td><td>63.69</td><td>52.43</td><td>45.70</td></tr><tr><td>CT (MIDL&#x27;22) [13]</td><td>85.96</td><td>76.51</td><td>81.64</td><td>70.87</td></tr><tr><td>XNet (ICCV&#x27;23) [28]</td><td>79.56</td><td>67.50</td><td>65.60</td><td>50.85</td></tr><tr><td>CorrMatch (CVPR’24) [22]</td><td>86.09</td><td>75.59</td><td>85.35</td><td>74.54</td></tr><tr><td>DuSSS (AAAI&#x27;25) [15]</td><td>79.50</td><td>66.80</td><td>71.13</td><td>57.45</td></tr><tr><td>CSDS (MICCAI-W&#x27;25) [16]</td><td>83.50</td><td>72.87</td><td>81.28</td><td>69.67</td></tr><tr><td>UniSemAlign (CVPRW&#x27;26) [24]</td><td>88.58</td><td>79.50</td><td>89.40</td><td>80.88</td></tr><tr><td>CoDiR (Ours)</td><td>89.19</td><td>81.09</td><td>90.29</td><td>82.93</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/29bfc432cfc75d0c43e55485c96f5b5f64cb2d3dccb8737c93f89b15b6f1b7a2.jpg)  
Fig. 2. Predicted masks at the 10% labeling ratio: GlaS in the upper three rows, CRAG in the lower three. Pink rectangles mark the regions where the compared methods diverge most visibly.

## 3.4 Ablation Studies

All ablation studies are conducted on the GlaS dataset under the 10% labeling setting, with results reported as mean ± std over three random seeds. Efect of Visual Backbone. Table 2 shows that UNI consistently outperforms the other visual encoders, achieving 88.09% mDice and 79.41% mJaccard. This demonstrates the advantage of using a pathology-specific visual representation for histopathology segmentation.

Table 2. Ablation study of diferent visual encoders in the Mean Teacher framework.
<table><tr><td>Backbone</td><td>mDice (%)</td><td>mJaccard (%)</td></tr><tr><td>ResNet101</td><td> $6 5 . 9 3 \pm 0 . 1 5$ </td><td> $5 1 . 1 0 \pm 0 . 1 3$ </td></tr><tr><td>ResNet200</td><td> $7 4 . 7 5 \pm 0 . 1 0$ </td><td> $6 0 . 6 4 \pm 0 . 1 1$ </td></tr><tr><td>MedCLIP</td><td> $7 2 . 6 5 \pm 0 . 1 7$ </td><td> $5 8 . 4 7 \pm 0 . 2 3$ </td></tr><tr><td>CONCH</td><td> $7 8 . 8 9 \pm 0 . 0 4$ </td><td> $6 6 . 5 5 \pm 0 . 1 4$ </td></tr><tr><td>UNI</td><td> ${ \bf 8 8 . 0 9 \pm 0 . 3 2 }$ </td><td> ${ \bf 7 9 . 4 1 \pm 0 . 4 6 }$ </td></tr></table>

Efect of Proposed Modules. Table 3 shows the incremental contribution of each CoDiR module. Difusion Refinement provides the largest gain, while Confidence Fusion and Confidence Weighting further improve the performance, resulting in 88.09% mDice and 79.41% mJaccard.

Table 3. Incremental ablation study of the proposed CoDiR modules, starting from a Mean Teacher baseline with the UNI encoder and DeepLabV3 decoder.
<table><tr><td>Configuration</td><td>mDice (%)</td><td>mJaccard (%)</td></tr><tr><td>Mean Teacher (UNI)</td><td> $8 0 . 9 5 \pm 0 . 1 2$ </td><td> $6 9 . 0 9 \pm 0 . 7 9$ </td></tr><tr><td>+ Diffusion Refinement</td><td> $8 7 . 3 1 \pm 0 . 2 6$ </td><td> $7 8 . 2 8 \pm 0 . 4 0$ </td></tr><tr><td>+ Confidence Fusion</td><td> $8 7 . 5 0 \pm 0 . 3 1$ </td><td> $7 8 . 5 4 \pm 0 . 4 7$ </td></tr><tr><td>+ Confidence Weighting</td><td> ${ \bf 8 8 . 0 9 \pm 0 . 3 2 }$ </td><td> ${ \bf 7 9 . 4 1 \pm 0 . 4 6 }$ </td></tr></table>

Efect of Confidence Threshold. As shown in Table 4, varying the confidence threshold ceiling $\tau _ { \mathrm { m a x } }$ (with $\tau _ { \mathrm { m i n } } = 0 . 6 0$ fixed) yields stable performance, with $\tau _ { \mathrm { m a x } } = 0 . 7 5$ performing best; this value is adopted for both training and inference. These results indicate that CoDiR remains stable across diferent choices of $\tau _ { \mathrm { m a x } }$

Table 4. Efect of the confidence threshold $\tau _ { \mathrm { m a x } }$
<table><tr><td>Tmax</td><td>mDice (%)</td><td>mJaccard (%)</td></tr><tr><td>0.70</td><td> $8 7 . 7 0 \pm 0 . 0 5$ </td><td> $7 8 . 8 9 \pm 0 . 1 6$ </td></tr><tr><td>0.75</td><td> $\mathbf { 8 8 . 0 9 \ : \pm { \ : 0 . 3 2 } }$ </td><td> $\mathbf { 7 9 . 4 1 \ : \pm { \ : 0 . 4 6 } }$ </td></tr><tr><td>0.80</td><td> $8 7 . 9 8 \pm 0 . 3 9$ </td><td> $7 9 . 2 9 \pm 0 . 6 2$ </td></tr></table>

Efect of Noise Level. Table 5 analyzes the impact of the noise level $\alpha _ { \mathrm { n } }$ in difusion refinement. Moderate noise $( \alpha _ { \mathrm { n } } = 0 . 2 0 )$ achieves the best performance, while both smaller and larger values lead to slight degradation. This indicates that an appropriate noise level is important for efective pseudo-label refinement.

Table 5. Sensitivity analysis of the difusion noise level $\alpha _ { \mathrm { n } }$
<table><tr><td> $\alpha _ { \mathrm { n } }$ </td><td>mDice (%)</td><td>mJaccard (%)</td></tr><tr><td>0.05</td><td> $8 7 . 0 7 \pm 0 . 0 9$ </td><td> $7 7 . 8 8 \pm 0 . 1 4$ </td></tr><tr><td>0.10</td><td> $8 7 . 9 8 \pm 0 . 3 9$ </td><td> $7 9 . 2 9 \pm 0 . 6 2$ </td></tr><tr><td>0.20</td><td> ${ \bf 8 8 . 0 9 \pm 0 . 3 2 }$ </td><td> ${ \bf 7 9 . 4 1 \pm 0 . 4 6 }$ </td></tr><tr><td>0.40</td><td> $8 7 . 6 1 \pm 0 . 6 9$ </td><td> $7 8 . 6 7 \pm 0 . 0 5$ </td></tr><tr><td>0.60</td><td> $8 7 . 8 6 \pm 0 . 3 8$ </td><td> $7 9 . 0 5 \pm 0 . 3 3$ </td></tr></table>

Efect of EMA Decay. As shown in Table 6, the performance remains relatively stable across diferent EMA decay rates. Among the evaluated settings, $d = 0 . 9 9$ achieves the best performance and is therefore adopted in all experiments.

Table 6. Sensitivity analysis of the EMA decay rate d.
<table><tr><td>d</td><td>mDice (%)</td><td>mJaccard (%)</td></tr><tr><td>0.90</td><td> $8 7 . 6 5 \pm 0 . 1 8$ </td><td> $7 8 . 7 8 \pm 0 . 1 9$ </td></tr><tr><td>0.99</td><td> ${ \bf 8 8 . 0 9 \pm 0 . 3 2 }$ </td><td> ${ \bf 7 9 . 4 1 \pm 0 . 4 6 }$ </td></tr><tr><td>0.995</td><td> $8 7 . 2 2 \pm 0 . 0 5$ </td><td> $7 8 . 1 4 \pm 0 . 1 3$ </td></tr><tr><td>0.999</td><td> $8 7 . 4 2 \pm 0 . 0 6$ </td><td> $7 8 . 3 5 \pm 0 . 1 3$ </td></tr></table>

## 4 Conclusion

In this work, we presented CoDiR, a semi-supervised histopathology segmentation framework that combines Mean Teacher learning with difusion-based pseudo-label refinement. By selectively refining low-confidence regions while preserving reliable teacher predictions, CoDiR improves segmentation performance under limited annotation. Experiments on GlaS and CRAG demonstrate consistent improvements over recent semi-supervised methods, highlighting the effectiveness of the proposed framework. However, the difusion refiner is trained on a relatively small labeled subset and evaluated only on two colorectal gland datasets, leaving its cross-organ and cross-stain generalization underexplored. Future work will focus on developing more data-eficient and structure-aware refinement strategies to improve boundary precision and enhance uncertain-region correction across diverse domains.

## References

1. Arazo, E., Ortego, D., Albert, P., O’Connor, N.E., McGuinness, K.: Pseudolabeling and confirmation bias in deep semi-supervised learning. In: 2020 International joint conference on neural networks (IJCNN). pp. 1–8. IEEE (2020)

2. Chen, L.C., Papandreou, G., Schrof, F., Adam, H.: Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587 (2017)

3. Chen, R.J., Ding, T., Lu, M.Y., Williamson, D.F., Jaume, G., Chen, B., Zhang, A., Shao, D., Song, A.H., Shaban, M., et al.: Towards a general-purpose foundation model for computational pathology. Nature Medicine (2024)

4. Chen, X., Yuan, Y., Zeng, G., Wang, J.: Semi-supervised semantic segmentation with cross pseudo supervision. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2613–2622 (2021)

5. Graham, S., Chen, H., Gamper, J., Dou, Q., Heng, P.A., Snead, D., Tsang, Y.W., Rajpoot, N.: Mild-net: Minimal information loss dilated network for gland instance segmentation in colon histology images. Medical Image Analysis 52, 199–211 (2019)

6. Han, K., Sheng, V.S., Song, Y., Liu, Y., Qiu, C., Ma, S., Liu, Z.: Deep semisupervised learning for medical image segmentation: A review. Expert Systems with Applications 245, 123052 (2024)

7. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)

8. Kapse, S., Pati, P., Das, S., Zhang, J., Chen, C., Vakalopoulou, M., Saltz, J., Samaras, D., Gupta, R.R., Prasanna, P.: Si-mil: Taming deep mil for self-interpretability in gigapixel histopathology. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11226–11237 (2024)

9. Kervadec, H., Bouchtiba, J., Desrosiers, C., Granger, E., Dolz, J., Ayed, I.B.: Boundary loss for highly unbalanced segmentation. In: International conference on medical imaging with deep learning. pp. 285–296. PMLR (2019)

10. Li, W., Lu, W., Chu, J., Tian, Q., Fan, F.: Confidence-guided mask learning for semi-supervised medical image segmentation. Computers in Biology and Medicine 165, 107398 (2023)

11. Liu, X., Li, W., Yuan, Y.: Difrect: Latent difusion label rectification for semisupervised medical image segmentation. In: International conference on medical image computing and computer-assisted intervention. pp. 56–66. Springer (2024)

12. Long, J., Shelhamer, E., Darrell, T.: Fully convolutional networks for semantic segmentation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3431–3440 (2015)

13. Luo, X., Hu, M., Song, T., Wang, G., Zhang, S.: Semi-supervised medical image segmentation via cross teaching between cnn and transformer. In: International conference on medical imaging with deep learning. pp. 820–833. PMLR (2022)

14. Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.Y., Ermon, S.: Sdedit: Guided image synthesis and editing with stochastic diferential equations. arXiv preprint arXiv:2108.01073 (2021)

15. Pan, Q., Qiao, W., Lou, J., Ji, B., Li, S.: Dusss: Dual semantic similaritysupervised vision-language model for semi-supervised medical image segmentation. arXiv preprint arXiv:2412.12492 (2024). https://doi.org/10.48550/arXiv. 2412.12492, https://arxiv.org/abs/2412.12492

16. Pham, H.H., Vu, N.L.V., Nguyen, T.H., Bagci, U., Xu, M., Le, T.N., Pham, H.H.: Learning disentangled stain and structural representations for semi-supervised histopathology segmentation. In: Studer, L., Ciompi, F., Khalili, N., Faryna, K., Faryna, K., Yeong, J., Lau, M.C., Chen, H., Liu, Z., Brattoli, B. (eds.) Proceedings of the MICCAI Workshop on Computational Pathology. Proceedings of Machine Learning Research, vol. 316, pp. 278–287. PMLR (27 Sep 2026), https://proceedings.mlr.press/v316/pham26a.html

17. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical image computing and computer-assisted intervention. pp. 234–241. Springer (2015)

18. Shen, Z., Cao, P., Yang, H., Liu, X., Yang, J., Zaiane, O.R.: Co-training with highconfidence pseudo labels for semi-supervised medical image segmentation. arXiv preprint arXiv:2301.04465 (2023)

19. Shit, S., Paetzold, J.C., Sekuboyina, A., Ezhov, I., Unger, A., Zhylka, A., Pluim, J.P., Bauer, U., Menze, B.H.: cldice-a novel topology-preserving loss function for tubular structure segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 16560–16569 (2021)

20. Sirinukunwattana, K., Pluim, J.P.W., Chen, H., Qi, X., Heng, P.A., Guo, Y.B., Wang, L.Y., Matuszewski, B.J., Bruni, E., Sanchez, U., et al.: Gland segmentation in colon histology images: The GlaS challenge contest. Medical Image Analysis 35, 489–502 (2017)

21. Sohn, K., Berthelot, D., Carlini, N., Zhang, Z., Zhang, H., Rafel, C.A., Cubuk, E.D., Kurakin, A., Li, C.L.: Fixmatch: Simplifying semi-supervised learning with consistency and confidence. Advances in neural information processing systems 33, 596–608 (2020)

22. Sun, B., Yang, Y., Zhang, L., Cheng, M.M., Hou, Q.: Corrmatch: Label propagation via correlation matching for semi-supervised semantic segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3097–3107 (2024)

23. Tarvainen, A., Valpola, H.: Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. Advances in neural information processing systems 30 (2017)

24. Van Thai, L., Nguyen, T.D., Pham, H.N., Thi, L.A.D., Nguyen, D.D., Bui, N.L.Q.: Unisemalign: Text-prototype alignment with a foundation encoder for semi-supervised histopathology segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. pp. 6803–6813 (2026)

25. Wu, J., Chen, M., Ke, X., Xun, T., Jiang, X., Zhou, H., Shao, L., Kong, Y.: Learning heterogeneous tissues with mixture of experts for gigapixel whole slide images. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 5144–5153 (2025)

26. Yu, L., Wang, S., Li, X., Fu, C.W., Heng, P.A.: Uncertainty-aware self-ensembling model for semi-supervised 3d left atrium segmentation. In: International conference on medical image computing and computer-assisted intervention. pp. 605–613. Springer (2019)

27. Zhang, C., Yang, G., Li, F., Wen, Y., Yao, Y., Shu, H., Simon, A., Dillenseger, J.L., Coatrieux, J.L.: Ctanet: confidence-based threshold adaption network for semisupervised segmentation of uterine regions from mr images for hifu treatment. IRBM 44(3), 100747 (2023)

28. Zhou, Y., Li, L., Wang, Z., Liu, G., Liu, Z., Yang, G.: Xnet v2: Fewer limitations, better results and greater universality. arXiv preprint arXiv:2409.00947 (2024). https://doi.org/10.48550/arXiv.2409.00947, https://arxiv.org/abs/2409. 00947