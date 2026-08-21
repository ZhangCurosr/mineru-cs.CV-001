# Far from the Crowd: Scalable Self-Supervised Learning via Geographic Isolation

Daniele Rege Cambrin<sup>1</sup> , Francesco Rossi<sup>1</sup> , and Mattia Varile<sup>1</sup>

AIKO, Corso Stati Uniti 45, Turin, Italy {name.surname}@aikospace.com https://www.aikospace.com/

Abstract. Self-supervised pretraining on remote sensing imagery typically treats all samples as equally informative, despite large variability in geographic and visual structure. We propose a curriculum learning strategy for self-supervised Earth observation that ranks samples by geographic isolation, a label-free proxy derived entirely from geolocation metadata already present in geospatial datasets, requiring no image decoding, no model feedback, and no manual annotation. Unlike visual complexity proxies, it scales as O(D log D) with dataset size D and is well-defined for both contrastive and reconstructive objectives.

We integrate the proposed measure into MoCoV2 and MAE pretraining and evaluate across three downstream tasks from CopernicusBench (BigEarthNet, DFC-2020, LCZ). Our curriculum reaches baseline finalepoch performance using as few as 20% of the training budget (MAE) and at most 40% (MoCo) of the training budget, and improves final downstream performance by up to +5 mAP on BigEarthNet, with gains of 1–5 points across benchmarks, matching visual-complexity curricula while reducing pre-computation cost by more than 140× (4 s vs. 568 s on SSL4EO). A CKA and efective-rank analysis further reveals that curriculum-trained encoders develop higher-dimensional, more uniformly utilized embedding spaces throughout training.

Keywords: Earth Observation · Self-Supervision · Curriculum Learning

## 1 Introduction

Self-supervised learning has become the basis of modern visual representation learning, enabling models to learn rich features from large unlabeled datasets without manual annotations [38, 41]. However, standard self-supervised training treats all samples as equally important, ignoring that some images are inherently more dificult to model due to their visual or structural complexity. Curriculum learning seeks to address this by gradually exposing the model to easier examples first, thereby stabilizing optimization and improving generalization [33, 40]. In the absence of explicit labels, defining a meaningful notion of dificulty for self-supervised signals remains an open challenge.

In the labeled setting, dificulty is often grounded in semantic complexity (provided by the labels) or model uncertainty (defined by the model’s supervised loss), but for self-supervised learning these notions are not directly available (the labels are not present, and the loss does not represent the model uncertainty). Instead, one must rely on proxies that capture intrinsic properties of the input distribution.

The vast amount of historical Earth observation data is commonly used to build large-scale unlabeled pretraining corpora, particularly valuable given the significant expertise, time, and cost required for high-quality semantic annotation. [41]. While they do not have associated labels, remote-sensed data (and geospatial data) have, in general, additional metadata: the geographic coordinates (i.e. the location) [1].

In this work, we propose a scalable, model-agnostic curriculum learning strategy for self-supervised learning derived from the coordinates: geographic isolation. This approach relies on the observation that the most geographically isolated samples are also the most visually anomalous relative to the densely sampled regions of the dataset. Additionally, compared to visual complexity, it is more scalable for large datasets since its computational cost is O(D log D + D<sup>¯</sup>k), where <sup>¯</sup>k is the mean neighborhood size, making it independent of image resolution and at least an order of magnitude cheaper than compression-based proxies at scale (a 142× speedup on SSL4EO).

Experiments on using the well-established MoCoV2 [8] and MAE [15] on SSL4EO [5,42] show that our curriculum improves convergence speed, representation quality, and downstream performance on BigEarthNet [9], DFC-2020 [28], LCZ [50] compared to uniform sampling and visual complexity curriculum.

In addition, we also provide a label-agnostic analysis of the learned feature space with central kernel alignment [21] and efective rank [29] to provide additional insight other than simply quantitative analysis on downstream tasks.

Our contributions are:

A scalable geographic isolation complexity measure that can be used in place of visual complexity;

– A benchmark of the proposed curriculum learning on both well-established contrastive (MoCoV2) and reconstructive (MAE) approaches for SSL in Earth observation during the training evolution.

– An embeddings space analysis over the training evolution to better understand the efects over the geometry composition.

For reproducibility the code is released at https://github.com/DarthReca/ far-from-the-crowd.

## 2 Related Works

This section covers the technique for self-supervised learning, curriculum learning, and their application to self-supervision. Finally, we cover XAI analysis for vision and earth observation.

## 2.1 Self-Supervised Learning

Self-supervised learning methods learn representations from unlabeled data by designing pretext tasks that proxy downstream objectives. Contrastive learning frameworks such as MoCo [8,16], SimCLR [7], BYOL [13], DINO [6,27,31] maximize agreement between diferently augmented views of the same image, while reconstruction-based methods such as MAE [15], BEiT [2] learn to reconstruct masked regions or transformations. In the geospatial domain, many unsupervised datasets such as CopernicusPretrain [43], EarthView [39] and SSL4EO [35, 42] and the relative models were created thanks to the large availability of data. These methods typically operate on large, heterogeneous collections of images but do not explicitly account for the varying dificulty of samples, instead sampling uniformly from the dataset.

Recent work has explored data-aware sampling strategies in self-supervised learning, such as curriculum-based [4,14,33] or uncertainty-driven sampling [24, 36], but these often rely on trained models or semantic priors [11, 32] that are not available at initialization. In contrast, our approach defines dificulty without access to labels or model predictions, enabling earlier and more robust curriculum scheduling.

## 2.2 Curriculum Learning

Curriculum learning aims to improve training by gradually exposing the model to increasingly dificult examples. Early work in curriculum learning focused on ordered data, such as sorting captioned images by sentence length or complexity [4], and has since been extended to automatic methods that estimate dificulty from model uncertainty or task-specific metrics [22].

In supervised settings, curriculum signals include predicted confidence, gradient norms, or losses, which can be used to select or reweight samples [22,33,40]. However, these approaches are tied to the current state of the model and can be sensitive to initialization and early noise [22, 40]. Several works have proposed more robust curriculum policies, such as self-paced learning or dynamic dificulty scheduling, but they still require access to labels or model outputs [18,33,40,46].

In the self-supervised setting, where neither labels or stable predictions are available early, designing a meaningful dificulty proxy without introducing model bias is particularly challenging. Our work sits at the intersection of these lines of research, providing a scalable, bias-free curriculum signal grounded in information-theoretic and spatial properties of the data.

## 2.3 Curriculum Learning for Self-Supervision

Curriculum learning has recently been adapted to self-supervised representation learning, where the absence of labels demands alternative notions of sample dificulty. Srinidhi and Martel [34] introduced a hardness-aware dynamic curriculum for contrastive pretraining leveraging annotator disagreement as a proxy for dificulty; however, their approach still depends on domain-specific metadata unavailable in generic large-scale datasets. u et al. [23] demonstrated that training on progressively cleaner data can induce noise robustness in self-supervised models without explicit denoising, suggesting that data ordering alone has a strong regularizing efect. Similarly, Joshi and Mirzasoleiman [20] showed that examples most beneficial for contrastive SSL are structurally diferent from those useful for supervised learning, motivating the need for SSL-specific dificulty criteria.

Beyond individual sample selection, several works have investigated scheduling policies over training. Self-paced contrastive learning [12] introduces a learnable pace parameter that progressively incorporates harder negatives as training proceeds, though it requires the model to have already converged to a meaningful representation space before the curriculum can take efect. ReSim [47] and subsequent region-based methods exploit local spatial structure to define patchlevel dificulty, but assume access to object-level cues. In contrast, our method operates entirely at the image level, deriving a dificulty score from intrinsic properties of the data enabling curriculum scheduling from the very first epoch, without any model feedback or semantic annotations.

## 3 Methodology

In this section, we describe the complexity scores employed in this study along with the curriculum strategy employed.

## 3.1 Complexity Scores

In this section, we define how the complexity scores for the curriculum are defined given the lack of labels in SSL settings using both the common visual appearance and the geographical information.

Visual Complexity A natural proxy for the visual dificulty of an unlabeled image is its information content. Intuitively, images with high spatial variability and diverse spectral patterns are harder to model than homogeneous ones. While entropy-based measures are commonly used, pixel-wise entropy ignores spatial structure, and embedding-based estimates introduce bias from pretrained representations.

We instead use lossless compression as a model-agnostic proxy for complexity, as it jointly captures spatial redundancy and spectral regularities. Given an image i and a lossless compressor c we define the compression ratio as:

$$
r = { \frac { | i | } { | c ( i ) | } }
$$

A high compression ratio indicates strong redundancy and low complexity (easy samples), whereas a low ratio reflects higher information density (hard samples). Unlike learned metrics, this proxy is independent of model inductive biases and requires no additional training or supervision and scales to hundreds of thousands of samples with negligible overhead on CPU.

![](images/51ac1b1e1da4717999c0a32ac18063c4ab974a46c25cd5068f9b90440c3434a6.jpg)  
|N<sub>r</sub>(i)| large → Low isolation

![](images/eb355f0fa8674562aa393777fec1e439a0a156cd8c368b226431c0a2417f5ac0.jpg)  
|N<sub>r</sub>(i)| small → High isolation  
Fig. 1: Given a fixed radius $r ,$ samples in densely covered areas have many neighbors and low isolation, whereas samples in sparsely covered areas have few neighbors and high isolation. This is aligned with the principle of spatial correlation.

Geographic Isolation Complexity Geospatial datasets provide implicit structural information through sample locations. According to Tobler’s first law of geography [37]:

Everything is related to everything else, but near things are more related than distant things.

so nearby samples tend to be more similar than distant ones. Consequently, geographically isolated samples are more likely to represent rare or diverse patterns (scenes that are underrepresented in the training distribution). Notably, this notion of rarity is orthogonal to visual complexity: empirically, geographic isolation and compression-ratio scores are rank-uncorrelated (Spearman ≈ 0), suggesting they capture distinct data properties. We quantify this intuition using a local density estimate. For each sample i, we define its geographic isolation score as

$$
g = - \log \left( \frac { | N _ { r } ( i ) | } { \pi r ^ { 2 } } \right)
$$

where $N _ { r } ( i )$ is the set of samples within radius $r$ of $i ,$ and $\pi r ^ { 2 }$ is the area of the neighborhood. The term inside the logarithm estimates the local sample density. Higher values of $g ( i )$ correspond to lower-density regions (i.e. more isolated and potentially more complex samples) as shown in Figure 1. The logarithm stabilizes the dynamic range and connects the measure to the notion of selfinformation [10].

To eficiently compute $N _ { r } ( i )$ , we index geographic coordinates using a Ball Tree [26], enabling sub-linear query time and scalability to large datasets.

It is possible to see the geographic distribution of the dataset and the top-100 most isolated points in Figure 2.

![](images/8ab31ad0eb0a786e8aede8105b305c0960dd263dbe44d0bd7039476a418a8529.jpg)

(a) Whole Dataset  
![](images/d2f60516ee090dcdffa8727f7fbefa5d11304e86fb6230b8f0411617417b44cd.jpg)  
(b) Top-100 Most Isolated Points  
Fig. 2: Geographical Distribution of Samples in the dataset and most geographically isolated points

## 3.2 Curriculum Building

Following a common definition in curriculum learning literature, we bin the dificulty scores into three, representing "easy," "medium," and "hard" samples [33] and then we apply a dynamic weight into the loss to each sample based on the dificulty tier [4, 22, 51].

The reweighting scheme shared across MoCo and MAE is time-dependent. After assigning each sample to a dificulty tier k (i.e. easy, medium, hard), we modulate its contribution with an annealed weight w<sub>k</sub>(t).

$$
w _ { k } ( t ) = \sigma \left( \alpha \frac { t - c _ { k } } { T } \right)
$$

where t is the current epoch, T the total number of epochs, $c _ { k }$ the activation center of tier $k ,$ α an annealing sharpness parameter. This schedule provides a soft curriculum as proposed in [45]: easy samples receive substantial weight earlier, while medium and hard samples are progressively emphasized as training advances and stabilize.

In MoCo, these weights directly scale the instance-discrimination objective so that contrastive updates are curriculum-aware at the sample level. For MAE, we use the same tier-wise weighting on the per-sample reconstruction loss. In summary, instead of employing a mean, we aggregate using a sample-weighted mean. Additionally, this strategy permits the MoCo memory bank to have a high variety of samples from the whole dataset even if they do not contribute to the model update.

## 4 Experiments

In this section, we present the experimental settings and the experimental results.   
Additional settings are provided in supplementary.

## 4.1 Experimental Settings

PreTraining For the SSL pretraining, we employed SSL4EOv1.1 Sentinel-2 subset [5] with around 250k locations around the globe. We trained a ResNet-50 [17] with MoCoV2 [8] and a ViT-Base with MAE [15] to provide an overview of two diferent well-established training and architectural paradigms. We trained both networks with a similar wall-time budget (≈12 h), which corresponds to 50 epochs for MoCoV2 and 100 epochs for MAE. The batch size was set to 128. The optimizer was SGD for MoCoV2 with a learning rate of 0.03 as in the original implementation and Adam with a learning rate of 1.5e-4 for MAE. The annealing value α was set to 8.0 after preliminary tests. The activation centers c were set to 0%, 20%, and 40% of training epochs.

Geographic Isolation The geographic isolation depends on the ray r, which was set to 50 km. The choice is linked to the dataset construction: according to SSL4EOv1.1 [5] they sampled within a 50 km radius of the world’s 10,000 most populated cities, ensuring near-global coverage while focusing on urban areas but also including ocean patches and rural areas. Additionally, if we fit a BallTree with the dataset coordinates and we query each point for the 50 km radius, we get more than one neighbor for 99.8% of the samples (vs. 97% for 20 km), confirming the sampling strategy of the paper.

Linear Probing We selected three tasks from CopernicusBench from the Sentinel 2 subset [44] evaluated according to the original metrics: two level 2 tasks (i.e. BigEarthNet-S2 and DFC-2020-S2) and one level 3 task (i.e. LCZ-S2) to test the goodness of the representation for diferent dificulties. We tested all the tasks in linear probing to better evaluate the quality of the learned representations, adding a linear layer for classification and a convolutional layer for segmentation.

Evaluation Metrics We evaluate the performance according to the original metrics of CopernicusBench [44]: multilabel classification (ML Cls) with 19 classes of BigEarthNet-S2 using mean average precision (mAP), multiclass semantic segmentation (MC Seg) with 10 classes of DFC-2020-S2 using micro

Intersection-over-Union (mIoU), and multiclass classification (MC Cls) with 17 classes of LCZ-S2 using overall accuracy (OA). We report the linear probing results across three diferent seeds, and we performed Bayesian tests [3] with a rope of 0.5 to account for significant diferences with small samples (i.e. 3 seeds). You can find all the tests in the supplementary materials.

## 4.2 Experimental Results

We evaluate representations at five equally spaced checkpoints covering 20%, 40%, 60%, 80%, and 100% of the total training budget (epochs 10, 20, 30, 40, 50 for MoCoV2; 20, 40, 60, 80, 100 for MAE) to assess not only whether final representation quality improves, but also whether convergence is accelerated. The tabulated results can be found in supplementary material.

![](images/986554672d7277712cc3b20752cc604d3fe805e0573db2d346efc17377f13f31.jpg)

![](images/bcad02e57cbeea9f33975b0b0235d68568cad11de6f7f67b41bf53373dd4495c.jpg)

![](images/188b437dfe23a8f11d4200a3124dc4bde1e7818cb87c0e496198e4ffad58e4a6.jpg)  
(a) MoCoV2 performance by epoch

![](images/9abc8bd5862cb867f39bf590b3af5547d76a0bcd0ee0f3e39acae3d762c15994.jpg)

![](images/f375e5e43d36f74decfccaa240565b484c01dbe3f936fa80d1e3c47f21912109.jpg)

![](images/dc18d0480f94bd03e8053eda1747064ff7ce829fc7d85cb5fb1848db3f5b9753.jpg)  
(b) MAE performance by epoch  
Fig. 3: Downstream linear probing performance along SSL pretraining. Results are reported for MoCoV2 and MAE using the baseline training (BASE), the compressionratio curriculum (COMP), and the geographic-isolation curriculum (GEO). The bands report the standard deviation.

As shown in Figure 3, curriculum-based pretraining consistently improves the quality of the learned representations with respect to the non-curriculum baseline across both SSL paradigms. This efect is particularly evident during the early stages of pretraining, where both curricula reach substantially higher downstream performance than the baseline after only a fraction of the total training budget. This suggests that the proposed reweighting strategy does not only improve final representations but also accelerates the emergence of transferable features.

For MoCoV2, both curriculum variants outperform the baseline on all downstream tasks during the whole training, also in early epochs. The Bayesian tests confirm this trend with probabilities higher than 95%, except for epoch 10 in DFC. On DFC-2020, the improvements are more moderate in absolute terms, likely due to saturation of the linear separability, but both curriculum strategies still provide a consistent advantage over the baseline. For MAE, the gains induced by curriculum learning are even more pronounced. The baseline shows a much slower convergence, especially on BigEarthNet-S2 and LCZ-S2, where the performance remains substantially below the curriculum variants for most of training. For DFC-2020, the baseline progressively closes part of the gap (as also indicated by the Bayesian test with probability 99%), but the curriculumbased methods remain consistently superior or comparable across all checkpoints. These results indicate that MAE benefits strongly from presenting easier samples with higher weight in the initial stages, likely because reconstruction-based pretraining is particularly sensitive to the visual complexity of the input distribution.

Comparing the two complexity scores, neither criterion dominates uniformly; Bayesian equivalence probabilities range from 70% to 90% across checkpoints. Critically, this parity is achieved at dramatically diferent costs, making geographic isolation the preferred choice when compute is constrained. This suggests the downstream benefit is not proxy-specific but reflects a general principle: any curriculum that breaks uniform sampling accelerates representation learning in the EO pretraining setting.

## 4.3 Computational Cost

Although dynamically adapting the curriculum during training is attractive in principle, pre-computing a fixed ordering provides a less adaptive but far more scalable solution — particularly suited to large datasets, repeated runs, and long training schedules.

Visual Complexity When using a compression algorithm, we can assume the worst case as a linear cost to the number of pixels P in an image, so for a dataset of size, D the cost is $O ( D \cdot N )$ . However, it is important to note that the process can be parallelized.

Geographic Isolation Complexity For geographic isolation via a BallTree, the onetime construction cost is O(D log D) and each radius query costs O(log $D + k )$ , where $k = | N _ { r } ( i ) |$ |. Summed over the full dataset, the total cost is O(D log D + $D \bar { k } )$ , which reduces to $\mathcal { O } ( D \log D )$ when $\bar { k } \ll D$ , as is typical for geographically distributed data

When using a BallTree to compute the geographic isolation, we incur (for a 2D representation) into a building cost of $O ( D \log ( D ) )$ ), while each query cost $O ( \log ( D ) + k )$ , where $k = | N _ { r } ( i ) | < < D$ , if the dataset is geographically distributed (the worst case is $k = D$ if all the points are returned). So for the dataset $D _ { \ast }$ , the whole cost to compute the geographic isolation is $O ( D \log D + D \overline { { k } } )$ , where $\overline { { k } }$ is the mean number of neighbors.

Self-Paced Learning The usage of self-paced learning is generally costly and not suited for this setting, since readapting the dificulty each epoch costs $O ( T \cdot D$ $F _ { m o d e l } )$ , where T is the number of epochs and $F _ { m o d e l }$ is the forward pass cost of the model.

Our one-shot solution can be cheaply computed without slowing the training independently of the model or the training paradigm without relying on labels. Compared to compression, it is not dependent on the image size but only on the number of samples and can be easily computed with few resources. In practice, on the full SSL4EO dataset: geographic isolation (BallTree) takes 4 s; compression ratio takes 568 s (142× slower); a single ResNet-50 forward pass over the dataset takes 379 s at batch size 128 and 34,962 s at batch size 1, two to four orders of magnitude slower than geographic isolation, and is repeated every epoch for self-paced learning.

## 4.4 Additional Experimental Results

Performance over Masking Ratio Curriculum Following previous approaches [25,48], we also evaluate how the proposed approach compares to gradually increasing the masking ratio of MAE as another scalable solution for SSL (which we name CLM hereafter).

As shown in Figure 4, the masking-ratio curriculum already improves over the fixed MAE baseline in most settings. Nevertheless, combining CLM with the proposed sample-level curricula further improves performance, particularly in the early and intermediate stages of pretraining. This indicates that tasklevel dificulty (masking ratio) and data-level dificulty (sample weighting) are orthogonal axes of curriculum design: each contributes an independent signal, and combining them is strictly better than either alone, particularly in early pretraining.

The Bayesian tests support this observation. On BigEarthNet-S2, both compression and geographic isolation improve over CLM with probabilities above 98%. On DFC-2020, the combined curricula are better at epoch 20, with probabilities above 97%, while they become statistically equivalent to CLM at later checkpoints, with equivalence probabilities between 85% and 99%, suggesting partial saturation of the downstream task. On LCZ-S2, both proposed combinations remain better than CLM with probabilities above 97%. Comparing the two sample-level proxies, GEO+CLM is generally better than or equivalent to COMP+CLM, with probabilities ranging from 80% to 99%.

Additional Metrics Analysis LCZ already balances the classes by construction, so overall accuracy is meaningful enough, while for BigEarthNet and DFC, such balance is not guaranteed by the dataset. Along with the standard metrics for the tasks proposed in CopernicusBench, we report the macro F1 scores, which let better understand the performance for rare classes as shown in Figure 5.

On BigEarthNet, both curriculum strategies substantially improve macro F1 over the baseline for both SSL paradigms, indicating that the learned representations are not only improving dominant classes but also better supporting rarer or under-represented land-cover categories (this is also confirmed by a Bayesian test with probability > 95%). For DFC, using MAE, the curriculum variants provide a clear advantage at the early checkpoints (with probability of ≈ 98%), after which the baseline progressively closes the gap and all methods converge to similar MacroIoU values (with probability of equality ≈ 98% for the last two checkpoints). This suggests that, on DFC, curriculum learning mainly accelerates convergence, while the final representations become comparable once pretraining is suficiently long. The two variants of curriculum behave similarly, and there is no clear winner (the Bayesian test confirms this finding, suggesting a probability higher than 97% of equality for most of the checkpoints).

![](images/0b44e35899eb5e0c9f2ffc9dec21361c29686b2ef499c8fdfe7529e5b405c58a.jpg)  
Fig. 4: Downstream performance across training epochs for the baseline MAE (BASE), masking ratio curriculum (CLM), and masking ratio curriculum combined with compression-based (COMP+CLM) and geographic isolation-based (GEO+CLM) sample ordering. Shaded areas denote standard deviations.

Overall, the class-balanced analysis reinforces the main experimental findings. The proposed curricula improve convergence speed and produce representations that are more robust to class imbalance, which is particularly relevant in remote-sensing datasets where rare semantic categories are common.

## 5 Embedding Space Analysis

To better understand how the embeddings space is built during the epochs, we analyze the embeddings space (of the last layer) of the test set of BigEarthNet (with around 6k samples), since it is the most semantically varied dataset, using a label-agnostic analysis, to provide a more general understanding of the model behavior.

## 5.1 Central Kernel Alignment

First, we analyze using linear Central Kernel Alignment [21] to quantify how representational geometry evolves across training and to determine whether curricula accelerate or alter the trajectory of representational convergence [30] in Figure 6.

For MoCoV2, the BASE-vs-GEO matrix shows the geographic-isolation curriculum alters the early-stage representation geometry, but its efect on the final representation becomes more moderate, so the baseline partially "catches up" in terms of which features it learns. The GEO-vs-COMP matrix, in contrast, suggests that they converge to similar representations as pretraining progresses, providing an explanation for the empirical observation that the two curricula are often statistically equivalent on downstream tasks.

![](images/625d771e9ed0fbc253cb011ef88796d3c681054eeecbe27a8e96f713c7f0c449.jpg)

(a) Macro F1 (BEN) and Macro IoU (DFC) across training epochs for MoCo-based pretraining.  
![](images/583a8dc713d00acab967ac79026934e1a53560be941e8face19dfd4c6c933be5.jpg)  
(b) Macro F1 (BEN) and Macro IoU (DFC) across training epochs for MAE-based pretraining  
Fig. 5: Macro F1 on BEN and Macro IoU on DFC across pretraining epochs for MoCo and MAE.

For MAE, both matrices are uniformly higher, indicating that MAE-based encoders, regardless of the curriculum, end up learning more similar representations than MoCo-based ones. This also suggests that generally employing a curriculum-induced training diversity enhances representation quality. The BASE-vs-GEO similarity mirrors the MoCo trend, while the GEO-vs-COMP matrix is remarkably saturated, with the lowest values appearing at the diagonal extremes (epochs 20 and 100). This suggests that, for masked reconstruction, the two curricula act through a similar mechanism (i.e. modulating the visual complexity distribution seen by the encoder) and that this mechanism dominates over the choice of proxy.

## 5.2 Efective Rank

Second, we analyze the efective rank [29] to understand how much the embeddings space is exploited [19] in Figure 7.

For MoCoV2, both compression and geo-isolation reach efective ranks well above BASE throughout training. Interestingly, the two curricula exhibit different growth dynamics: COMP grows almost monotonically across epochs and attains the highest final efective rank, whereas GEO plateaus. This suggests that the compression-based proxy yields a more gradual, sustained diversification of the representation, while the geographic-isolation proxy provides an early, strong diversification signal that the model quickly absorbs. The baseline, in contrast, grows much more slowly and never catches up within the same training budget, indicating that, without an explicit curriculum, instance discrimination tends to compress the embedding onto a narrower subspace.

![](images/a83c619b750a7fd872f9ebc592f459d3c547de40dcca99c211eaeb3097096858.jpg)

![](images/0a67a0c172df72a4eda6d69ba7141fb79325e8779627c4f83e068a9ec732fb20.jpg)

(a) MoCo  
![](images/24659f29c9214ae3f3a241936871dd9c2e7fcc02a5924f442f09eb09bac07e46.jpg)

<table><tr><td rowspan=7 colspan=1>COMP</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.88</td></tr><tr><td rowspan=2 colspan=1>0.960.97</td><td rowspan=2 colspan=1>0.900.89</td><td rowspan=1 colspan=1>0.96</td><td rowspan=2 colspan=1>0.970.97</td><td rowspan=2 colspan=1>0.860.87</td></tr><tr><td rowspan=1 colspan=1>0.95</td></tr><tr><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.82</td></tr><tr><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.76</td></tr><tr><td rowspan=2 colspan=1>20</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>60</td><td rowspan=2 colspan=1>80</td><td rowspan=2 colspan=1>100</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>GEO</td></tr></table>

(b) MAE  
Fig. 6: Central Kernel Alignment (CKA) between encoder representations at diferent pretraining epochs, computed on the BigEarthNet test set.

For MAE, the efect is qualitatively similar but markedly more unstable. The baseline exhibits large epoch-to-epoch fluctuations in efective rank (a wellknown signature of the instability known to afect masked-image modeling [15, 30, 49] when reconstruction targets are uniformly complex). Both COMP and GEO suppress these oscillations and maintain consistently higher ranks, with GEO providing the smoothest trajectory, suggesting it acts as a distributional regularizer on the reconstruction dificulty seen during early training.

![](images/43dd48d71520d231757594cfb748464082269dcca0211313d4f26122fdef34ba.jpg)  
(a) MoCo

![](images/21628943156241ad41adb447af7d55c2ffacd0c2cc8945de782d258a8efc9548.jpg)  
(b) MAE  
Fig. 7: Efective rank of the encoder representations on the BigEarthNet test set across pretraining epochs under the baseline (BASE), the compression-ratio curriculum (COMP), and the geographic-isolation curriculum (GEO).

## 6 Conclusion

Geographic isolation provides an efective curriculum signal for self-supervised Earth observation: without any model feedback, label, or image decoding, a single BallTree construction and query over coordinates (costing just 4 s for SSL4EO, versus 568 s for compression-based alternatives) sufices to substantially accelerate representation learning and improve downstream performance.

Experiments with MoCoV2 and MAE on SSL4EO-v1.1, evaluated across three downstream tasks from CopernicusBench, show that both curriculum variants consistently accelerate convergence and improve final downstream performance over uniform sampling. The gains are most pronounced in the early pretraining regime and for MAE, where reconstruction objectives are particularly sensitive to the complexity distribution of the training data. The embedding space analysis reinforces these findings: curriculum-trained encoders occupy a higher-dimensional, more isotropic representation space throughout training, and CKA analysis reveals that both proxies converge to geometrically similar solutions once pretraining is suficiently long. This robustness to proxy choice is a practically useful finding: geographic isolation and compression ratio are largely interchangeable in terms of representational outcome, yet the former is dramatically cheaper to compute. This also suggests that applying a curriculum strategy is relevant to improve the representational quality regardless of the proxy.

Future work could extend geographic isolation to multi-modal geospatial datasets (optical, SAR, and elevation); to temporal sequences where both spatial and temporal isolation are meaningful; and to continual learning settings where new acquisitions arrive incrementally and re-ranking must remain cheap.

## References

1. Ayush, K., Uzkent, B., Meng, C., Tanmay, K., Burke, M., Lobell, D., Ermon, S.: Geography-aware self-supervised learning. In: 2021 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 10161–10170 (2021). https://doi. org/10.1109/ICCV48922.2021.01002 2

2. Bao, H., Dong, L., Wei, F.: Beit: Bert pre-training of image transformers. ArXiv abs/2106.08254 (2021), https://api.semanticscholar.org/CorpusID: 235436185 3

3. Benavoli, A., Corani, G., Demšar, J., Zafalon, M.: Time for a change: a tutorial for comparing multiple classifiers through bayesian analysis. Journal of Machine Learning Research 18(77), 1–36 (2017), http://jmlr.org/papers/v18/16-305. html 8

4. Bengio, Y., Louradour, J., Collobert, R., Weston, J.: Curriculum learning. In: International Conference on Machine Learning (2009), https://api.semanticscholar. org/CorpusID:873046 3, 6

5. Blumenstiel, B., Braham, N.A.A., Albrecht, C.M., Maurogiovanni, S., Fraccaro, P.: Ssl4eo-s12 v1.1: A multimodal, multiseasonal dataset for pretraining, updated (2026), https://arxiv.org/abs/2503.00168 2, 7

6. Caron, M., Touvron, H., Misra, I., J’egou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. 2021 IEEE/CVF International Conference on Computer Vision (ICCV) pp. 9630–9640 (2021), https://api.semanticscholar.org/CorpusID:233444273 3

7. Chen, T., Kornblith, S., Norouzi, M., Hinton, G.E.: A simple framework for contrastive learning of visual representations. ArXiv abs/2002.05709 (2020), https://api.semanticscholar.org/CorpusID:211096730 3

8. Chen, X., Fan, H., Girshick, R., He, K.: Improved baselines with momentum contrastive learning (2020), https://arxiv.org/abs/2003.04297 2, 3, 7

9. Clasen, K.N., Hackel, L., Burgert, T., Sumbul, G., Demir, B., Markl, V.: reBEN: Refined bigearthnet dataset for remote sensing image analysis. In: IEEE International Geoscience and Remote Sensing Symposium (IGARSS) (2025) 2

10. Cover, T.M., Thomas, J.A.: Elements of Information Theory. Wiley (Apr 2005). https://doi.org/10.1002/047174882x, http://dx.doi.org/10.1002/ 047174882X 5

11. Cui, P., Zhang, D., Deng, Z., Dong, Y., Zhu, J.: Learning sample dificulty from pre-trained models for reliable prediction. ArXiv abs/2304.10127 (2023), https: //api.semanticscholar.org/CorpusID:258236695 3

12. Ge, Y., Chen, D., Zhu, F., Zhao, R., Li, H.: Self-paced contrastive learning with hybrid memory for domain adaptive object re-id. ArXiv abs/2006.02713 (2020), https://api.semanticscholar.org/CorpusID:219305432 4

13. Grill, J.B., Strub, F., Altch’e, F., Tallec, C., Richemond, P.H., Buchatskaya, E., Doersch, C., Pires, B.Á., Guo, Z.D., Azar, M.G., Piot, B., Kavukcuoglu, K., Munos, R., Valko, M.: Bootstrap your own latent: A new approach to self-supervised learning. ArXiv abs/2006.07733 (2020), https://api.semanticscholar.org/ CorpusID:219687798 3

14. Hacohen, G., Weinshall, D.: On the power of curriculum learning in training deep networks. In: International Conference on Machine Learning (2019), https://api. semanticscholar.org/CorpusID:102350936 3

15. He, K., Chen, X., Xie, S., Li, Y., Doll’ar, P., Girshick, R.B.: Masked autoencoders are scalable vision learners. 2022 IEEE/CVF Conference on Computer Vi-

sion and Pattern Recognition (CVPR) pp. 15979–15988 (2021), https://api. semanticscholar.org/CorpusID:243985980 2, 3, 7, 13

16. He, K., Fan, H., Wu, Y., Xie, S., Girshick, R.B.: Momentum contrast for unsupervised visual representation learning. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 9726–9735 (2019), https: //api.semanticscholar.org/CorpusID:207930212 3

17. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 770–778 (2016). https://doi.org/10.1109/CVPR.2016.90 7

18. Higuchi, T., Saxena, S., Souden, M., Tran, T.D., Delfarah, M., Dhir, C.S.: Dynamic curriculum learning via data parameters for noise robust keyword spotting. ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) pp. 6848–6852 (2021), https://api.semanticscholar. org/CorpusID:231979234 3

19. Jing, L., Vincent, P., LeCun, Y., Tian, Y.: Understanding dimensional collapse in contrastive self-supervised learning. ArXiv abs/2110.09348 (2021), https: //api.semanticscholar.org/CorpusID:239016966 12

20. Joshi, S., Mirzasoleiman, B.: Data-eficient contrastive self-supervised learning: Most beneficial examples for supervised learning contribute the least. In: International Conference on Machine Learning (2023), https://api.semanticscholar. org/CorpusID:258987598 4

21. Kornblith, S., Norouzi, M., Lee, H., Hinton, G.E.: Similarity of neural network representations revisited. ArXiv abs/1905.00414 (2019), https://api. semanticscholar.org/CorpusID:141460329 2, 11

22. Kumar, M.P., Packer, B., Koller, D.: Self-paced learning for latent variable models. In: Neural Information Processing Systems (2010), https : / / api . semanticscholar.org/CorpusID:1977996 3, 6

23. Lu, W., Zhang, J., van Assel, H., Balestriero, R.: Ditch the denoiser: Emergence of noise robustness in self-supervised learning from data curriculum. ArXiv abs/2505.12191 (2025), https://api.semanticscholar.org/CorpusID: 278740957 4

24. Mindermann, S., Brauner, J.M., Razzak, M.T., Sharma, M., Kirsch, A., Xu, W., Höltgen, B., Gomez, A.N., Morisot, A., Farquhar, S., Gal, Y.: Prioritized training on points that are learnable, worth learning, and not yet learnt. In: Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., Sabato, S. (eds.) Proceedings of the 39th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 162, pp. 15630–15649. PMLR (17–23 Jul 2022), https: //proceedings.mlr.press/v162/mindermann22a.html 3

25. Naskar, A., Wong, N.Z., Shamekh, S.: Cumolos-mae: A masked autoencoder for remote sensing data reconstruction. ArXiv abs/2508.14957 (2025), https:// api.semanticscholar.org/CorpusID:280700165 10

26. Omohundro, S.M.: Five balltree construction algorithms (2009), https://api. semanticscholar.org/CorpusID:61067117 5

27. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.Y.B., Li, S.W., Misra, I., Rabbat, M.G., Sharma, V., Synnaeve, G., Xu, H., Jégou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: Dinov2: Learning robust visual features without supervision. ArXiv abs/2304.07193 (2023), https://api.semanticscholar.org/CorpusID:258170077 3

28. Robinson, C., Malkin, K., Jojic, N., Chen, H., Qin, R., Xiao, C., Schmitt, M., Ghamisi, P., Hänsch, R., Yokoya, N.: Global land-cover mapping with weak supervision: Outcome of the 2020 ieee grss data fusion contest. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 14, 3185–3199 (2021). https://doi.org/10.1109/JSTARS.2021.3063849 2

29. Roy, O., Vetterli, M.: The efective rank: A measure of efective dimensionality. In: 2007 15th European Signal Processing Conference. pp. 606–610 (2007) 2, 12

30. Shekhar, S., Bordes, F., Vincent, P., Morcos, A.S.: Objectives matter: Understanding the impact of self-supervised objectives on vision transformer representations. ArXiv abs/2304.13089 (2023), https://api.semanticscholar.org/CorpusID: 258331753 11, 13

31. Sim’eoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., Tolan, J., Brandt, J., Couprie, C., Mairal, J., J’egou, H., Labatut, P., Bojanowski, P.: Dinov3 (2025), https://api.semanticscholar.org/CorpusID:280649654 3

32. Sorscher, B., Geirhos, R., Shekhar, S., Ganguli, S., Morcos, A.S.: Beyond neural scaling laws: beating power law scaling via data pruning. ArXiv abs/2206.14486 (2022), https://api.semanticscholar.org/CorpusID:250113273 3

33. Soviany, P., Ionescu, R.T., Rota, P., Sebe, N.: Curriculum learning: A survey. International Journal of Computer Vision 130(6), 1526–1565 (Apr 2022). https:// doi.org/10.1007/s11263-022-01611-x, http://dx.doi.org/10.1007/s11263- 022-01611-x 1, 3, 6

34. Srinidhi, C.L., Martel, A.L.: Improving self-supervised learning with hardnessaware dynamic curriculum learning: An application to digital pathology. 2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW) pp. 562–571 (2021), https://api.semanticscholar.org/CorpusID:237091233 3

35. Stewart, A.J., Lehmann, N., Corley, I.A., Wang, Y., Chang, Y., Braham, N.A.A., Sehgal, S., Robinson, C., Banerjee, A.: Ssl4eo-l: Datasets and foundation models for landsat imagery. ArXiv abs/2306.09424 (2023), https://api. semanticscholar.org/CorpusID:259187561 3

36. Swayamdipta, S., Schwartz, R., Lourie, N., Wang, Y., Hajishirzi, H., Smith, N.A., Choi, Y.: Dataset cartography: Mapping and diagnosing datasets with training dynamics. In: Webber, B., Cohn, T., He, Y., Liu, Y. (eds.) Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). pp. 9275–9293. Association for Computational Linguistics, Online (Nov 2020). https: //doi.org/10.18653/v1/2020.emnlp- main.746, https://aclanthology.org/ 2020.emnlp-main.746/ 3

37. Tobler, W.R.: A computer movie simulating urban growth in the detroit region. Economic Geography 46, 234 (1970). https://doi.org/10.2307/143141, http: //dx.doi.org/10.2307/143141 5

38. Uelwer, T., Robine, J., Wagner, S.S., Höftmann, M., Upschulte, E., Konietzny, S., Behrendt, M., Harmeling, S.: A survey on self-supervised methods for visual representation learning. Mach. Learn. 114(4) (Mar 2025). https://doi.org/10. 1007/s10994-024-06708-7, https://doi.org/10.1007/s10994-024-06708-7 1

39. Velázquez, D.A., L’opez, P.R., Alonso, S., Gonfaus, J.M., Gonzàlez, J., Richarte, G., Marín, J., Bengio, Y., Lacoste, A.: Earthview: A large scale remote sensing dataset for self-supervision. 2025 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW) pp. 1138–1147 (2025), https://api. semanticscholar.org/CorpusID:275515549 3

40. Wang, X., Chen, Y., Zhu, W.: A survey on curriculum learning. IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 4555–4576 (2021), https://api. semanticscholar.org/CorpusID:232362223 1, 3

41. Wang, Y., Albrecht, C.M., Braham, N.A.A., Mou, L., Zhu, X.X.: Self-supervised learning in remote sensing: A review. IEEE Geoscience and Remote Sensing Magazine 10(4), 213–247 (Dec 2022). https://doi.org/10.1109/mgrs.2022.3198244, http://dx.doi.org/10.1109/MGRS.2022.3198244 1, 2

42. Wang, Y., Braham, N.A.A., Xiong, Z., Liu, C., Albrecht, C.M., Zhu, X.X.: Ssl4eo-s12: A large-scale multi-modal, multi-temporal dataset for self-supervised learning in earth observation. ArXiv abs/2211.07044 (2022), https://api. semanticscholar.org/CorpusID:253510090 2, 3

43. Wang, Y., Xiong, Z., Liu, C., Stewart, A.J., Dujardin, T., Bountos, N.I., Zavras, A., Gerken, F., Papoutsis, I., Leal-Taixé, L., Zhu, X.X.: Towards a unified copernicus foundation model for earth vision. 2025 IEEE/CVF International Conference on Computer Vision (ICCV) pp. 9888–9899 (2025), https://api.semanticscholar. org/CorpusID:277065800 3

44. Wang, Y., Xiong, Z., Liu, C., Stewart, A.J., Dujardin, T., Bountos, N.I., Zavras, A., Gerken, F., Papoutsis, I., Leal-Taixé, L., Zhu, X.X.: Towards a unified copernicus foundation model for earth vision. In: 2025 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 9888–9899 (2025). https://doi.org/10.1109/ ICCV51701.2025.00922 7

45. Wang, Y., Gan, W., Yang, J., Wu, W., Yan, J.: Dynamic curriculum learning for imbalanced data classification. In: 2019 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 5016–5025 (2019). https://doi.org/10.1109/ ICCV.2019.00512 6

46. Weinshall, D., Cohen, G., Amir, D.: Curriculum learning by transfer learning: Theory and experiments with deep networks (2018), https://arxiv.org/abs/ 1802.03796 3

47. Xiao, T., Reed, C., Wang, X., Keutzer, K., Darrell, T.: Region similarity representation learning. 2021 IEEE/CVF International Conference on Computer Vision (ICCV) pp. 10519–10528 (2021), https://api.semanticscholar.org/CorpusID: 232335845 4

48. Yoon, T., Kang, D.: Currimae: curriculum learning based masked autoencoders for multi-labeled pediatric thoracic disease classification. PeerJ Computer Science 11 (2025), https://api.semanticscholar.org/CorpusID:280559310 10

49. Zhang, K., Shen, Z.: i-mae: Are latent representations in masked autoencoders linearly separable? 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) pp. 7740–7749 (2022), https://api. semanticscholar.org/CorpusID:253018584 13

50. Zhu, X.X., Hu, J., Qiu, C., Shi, Y., Kang, J., Mou, L., Bagheri, H., Haberle, M., Hua, Y., Huang, R., Hughes, L., Li, H., Sun, Y., Zhang, G., Han, S., Schmitt, M., Wang, Y.: So2sat lcz42: A benchmark data set for the classification of global local climate zones [software and data sets]. IEEE Geoscience and Remote Sensing Magazine 8(3), 76–89 (2020). https://doi.org/10.1109/MGRS.2020.2964708 2

51. Zhuang, J., Jing, X.Y., Jia, X.: Mining negative samples on contrastive learning via curricular weighting strategy. Information Sciences 668, 120534 (May 2024). https://doi.org/10.1016/j.ins.2024.120534, http://dx.doi.org/10.1016/ j.ins.2024.120534 6

<sup>1</sup> https://docs.python.org/3/library/zlib.html

## A Additional Experimental Settings

All run were performed on a single NVIDIA A6000.

## A.1 Compression Ratio

Since we are simply interested in the redundancy, we compress the image data with zlib <sup>1</sup> with a compression level of 1 for minimal computational burden.

## A.2 Linear Probing

The pretrained backbones were frozen. The batch size was set to 128. We used the AdamW optimizer with reduce learning rate on plateau for 30 epochs and a learning rate of 1e-3. We employ binary cross-entropy loss for BigEarthNet and cross-entropy loss for LCZ. The bands that are not included in the downstream tasks are zeroed (the Sentinel-2 NODATA value).

## B Full Bayesian Comparison

The results for the Bayesian tests over the main metrics are reported in Table 1, while for macro metrics in Table 3 for both MoCov2 and MAE during the training epochs. The results of the Bayesian tests over the curriculum masking variants are reported in Table 2.

## C Tabulated Results

We report the results for each seed (i.e. 123, 0, 42) for both MoCo and MAE variants for the main metrics in Table 4 and for the macro metrics in Table 6. In Table 5, we report the results for curriculum masking (CLM) variants across the same seeds.

Table 1: Bayesian Test Probability on main metrics with Rope of 0.5
<table><tr><td>Dataset</td><td>Model</td><td>Epoch (%)|</td><td>|P(left) P(right) P(rope)|P(left) P(right) P(rope)</td><td>MoCo</td><td></td><td>MAE</td><td></td></tr><tr><td>BEN</td><td>GEO vs BASE</td><td>20%</td><td>92.99</td><td>2.38 4.63</td><td>99.96</td><td>0.03</td><td>0.01</td></tr><tr><td>BEN</td><td>GEO vs BASE</td><td>40%</td><td>97.74</td><td>1.45 0.82</td><td>99.97</td><td>0.02</td><td>0.01</td></tr><tr><td>BEN</td><td>GEO vs BASE</td><td>60%</td><td>99.43</td><td>0.39 0.18</td><td>100.00</td><td>0.00</td><td>0.00</td></tr><tr><td>BEN</td><td>GEO vs BASE</td><td>80%</td><td>98.87</td><td>0.60 0.53</td><td>99.89</td><td>0.09</td><td>0.02</td></tr><tr><td>BEN</td><td>GEO vs BASE</td><td>100%</td><td>99.82</td><td>0.12 0.06</td><td>98.60</td><td>0.67</td><td>0.74</td></tr><tr><td>BEN</td><td>COMP vs BASE</td><td>20%</td><td>96.35</td><td>1.45 2.19</td><td>99.96</td><td>0.03</td><td>0.01</td></tr></table>

Continued on next page

Table 1: Bayesian Test Probability on main metrics with Rope of 0.5
<table><tr><td>Dataset</td><td>Model</td><td>Epoch (%)</td><td>P(left) P(right) P(rope)|P(left) P(right) P(rope)</td><td>MoCo</td><td></td><td></td><td>MAE</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BEN</td><td>COMP vs BASE</td><td>40%</td><td>95.87</td><td>2.64</td><td>1.48</td><td>|100.00 99.94</td><td>0.00</td><td>0.00</td></tr><tr><td>BEN</td><td>COMP vs BASE COMP vs BASE</td><td>60% 80%</td><td>99.67 99.69</td><td>0.23 0.21</td><td>0.10 0.10</td><td>99.99</td><td>0.05 0.01</td><td>0.01 0.00</td></tr><tr><td>BEN BEN</td><td>COMP vs BASE</td><td>100%</td><td>99.88</td><td>0.08</td><td>0.04</td><td>99.86</td><td>0.08</td><td>0.05</td></tr><tr><td>BEN</td><td>GEO vs COMP</td><td>20%</td><td>1.22</td><td>25.94</td><td>72.84</td><td>99.55</td><td>0.14</td><td>0.32</td></tr><tr><td>BEN</td><td>GEO vs COMP</td><td>40%</td><td>20.06</td><td>7.87</td><td>72.07</td><td>9.37</td><td>0.48</td><td>90.15</td></tr><tr><td>BEN</td><td>GEO vs COMP</td><td>60%</td><td>0.87</td><td>7.11</td><td>92.02</td><td>28.47</td><td>5.32</td><td>66.21</td></tr><tr><td>BEN</td><td>GEO vs COMP</td><td>80%</td><td>0.05</td><td>99.85</td><td>0.10</td><td>5.73</td><td>21.86</td><td>72.41</td></tr><tr><td>BEN</td><td>GEO vs COMP</td><td>100%</td><td>0.02</td><td>91.03</td><td>8.94</td><td>1.13</td><td>96.28</td><td>2.59</td></tr><tr><td>DFC</td><td>GEO vs BASE</td><td>20%</td><td>58.85</td><td>4.56</td><td>36.59</td><td>99.77</td><td>0.12</td><td>0.11</td></tr><tr><td>DFC</td><td>GEO vs BASE</td><td>40%</td><td>99.79</td><td>0.05</td><td>0.15</td><td>99.12</td><td>0.25</td><td>0.63</td></tr><tr><td>DFC</td><td>GEO vs BASE</td><td>60%</td><td>99.98</td><td>0.01</td><td>0.01</td><td>96.40</td><td>0.15</td><td>3.45</td></tr><tr><td>DFC</td><td>GEO vs BASE</td><td>80%</td><td>99.87</td><td>0.03</td><td>0.11</td><td>97.74</td><td>0.03</td><td>2.23</td></tr><tr><td>DFC</td><td>GEO vs BASE</td><td>100%</td><td>99.88</td><td>0.04</td><td>0.08</td><td>0.19</td><td>0.02</td><td>99.80</td></tr><tr><td>DFC</td><td>COMP vs BASE</td><td>20%</td><td>35.72</td><td>4.53</td><td>59.74</td><td>99.88</td><td>0.05</td><td>0.07</td></tr><tr><td>DFC</td><td>COMP vs BASE</td><td>40%</td><td>99.84</td><td>0.04</td><td>0.12</td><td>99.09</td><td>0.23</td><td>0.68</td></tr><tr><td></td><td>COMP vs BASE</td><td>60%</td><td>100.00</td><td>0.00</td><td>0.00</td><td>83.08</td><td>0.48</td><td>16.44</td></tr><tr><td>DFC</td><td>COMP vs BASE</td><td>80%</td><td>98.45</td><td>0.44</td><td>1.11</td><td>0.76</td><td>0.17</td><td>99.07</td></tr><tr><td>DFC</td><td>COMP vs BASE</td><td>100%</td><td>99.81</td><td>0.05</td><td>0.14</td><td>0.55</td><td>0.45</td><td>99.00</td></tr><tr><td>DFC DFC</td><td>GEO vs COMP</td><td>20%</td><td>3.71</td><td>0.68</td><td>95.61</td><td>97.42</td><td>0.25</td><td>2.34</td></tr><tr><td></td><td></td><td></td><td>0.06</td><td>0.03</td><td>99.92</td><td>0.20</td><td>0.07</td><td>99.74</td></tr><tr><td>DFC</td><td>GEO vs COMP</td><td>40%</td><td>29.59</td><td>0.05</td><td></td><td>0.58</td><td>0.25</td><td></td></tr><tr><td>DFC</td><td>GEO vs COMP</td><td>60%</td><td>1.93</td><td>14.56</td><td>70.36 83.51</td><td>16.67</td><td></td><td>99.17 83.26</td></tr><tr><td>DFC</td><td>GEO vs COMP</td><td>80%</td><td></td><td></td><td></td><td>1.09</td><td>0.07</td><td>98.78</td></tr><tr><td>DFC</td><td>GEO vs COMP</td><td>100%</td><td>0.13</td><td>0.04</td><td>99.83</td><td></td><td>0.13</td><td>0.00</td></tr><tr><td>LCZ</td><td>GEO vs BASE</td><td>20%</td><td>99.79</td><td>0.15</td><td>0.05</td><td>99.96</td><td>0.03</td><td></td></tr><tr><td>LCZ</td><td>GEO vs BASE</td><td>40%</td><td>98.04</td><td>1.32</td><td>0.64</td><td>99.85</td><td>0.13</td><td>0.02</td></tr><tr><td>LCZ</td><td>GEO vs BASE</td><td>60%</td><td>99.17</td><td>0.52</td><td>0.31</td><td>99.38</td><td>0.54</td><td>0.08 0.01</td></tr><tr><td>LCZ</td><td>GEO vs BASE</td><td>80%</td><td>99.53</td><td>0.27</td><td>0.19</td><td>99.94</td><td>0.05</td><td>0.01</td></tr><tr><td>LCZ</td><td>GEO vs BASE</td><td>100%</td><td>98.10</td><td>1.13</td><td>0.76</td><td>99.97</td><td>0.02</td><td>0.02</td></tr><tr><td>LCZ</td><td>COMP vs BASE</td><td>20%</td><td>99.95</td><td>0.03</td><td>0.02</td><td>99.78</td><td>0.20</td><td></td></tr><tr><td>LCZ</td><td>COMP vs BASE</td><td>40%</td><td>98.60</td><td>0.97</td><td>0.43</td><td>99.94</td><td>0.05</td><td>0.01</td></tr><tr><td>LCZ</td><td>COMP vs BASE</td><td>60%</td><td>98.66</td><td>0.79</td><td>0.55</td><td>99.34</td><td>0.57</td><td>0.09</td></tr><tr><td>LCZ</td><td>COMP vs BASE</td><td>80%</td><td>99.60</td><td>0.23</td><td>0.17</td><td>99.85</td><td>0.12</td><td>0.03</td></tr><tr><td>LCZ</td><td>COMP vs BASE</td><td>100%</td><td>99.81</td><td>0.11</td><td>0.09</td><td>99.88</td><td>0.09</td><td>0.04</td></tr><tr><td>LCZ</td><td>GEO vs COMP</td><td>20%</td><td>97.87</td><td>0.97</td><td>1.16</td><td>97.27</td><td>1.65</td><td>1.07</td></tr><tr><td>LCZ</td><td>GEO vs COMP</td><td>40%</td><td>0.52</td><td>29.90</td><td>69.58</td><td>92.06</td><td>2.04</td><td>5.90</td></tr><tr><td>LCZ</td><td>GEO vs COMP</td><td>60%</td><td>62.46</td><td>0.73</td><td>36.81</td><td>92.01</td><td>2.63</td><td>5.36</td></tr><tr><td>LCZ</td><td>GEO vs COMP</td><td>80%</td><td>9.48</td><td>4.73</td><td>85.79</td><td>99.58</td><td>0.13</td><td>0.29</td></tr><tr><td>LCZ</td><td>GEO vs COMP</td><td>100%</td><td>39.30</td><td>10.66</td><td>50.04</td><td>0.96</td><td>97.04</td><td>2.00</td></tr></table>

Table 2: Bayesian Test Probability on main metrics for Curriculum masking ratio (CLM) with Rope of 0.5
<table><tr><td>Dataset</td><td>Model</td><td>Epoch|P(left) P(right) P(rope)</td><td></td><td></td><td></td></tr><tr><td>BEN</td><td>GEO+CLM vs CLM</td><td>20%</td><td>99.91</td><td>0.06</td><td>0.02</td></tr><tr><td>BEN</td><td>GEO+CLM vs CLM</td><td>40%</td><td>99.87</td><td>0.10</td><td>0.03</td></tr><tr><td>BEN</td><td>GEO+CLM vs CLM</td><td>60%</td><td>99.99</td><td>0.01</td><td>0.00</td></tr><tr><td>BEN</td><td>GEO+CLM vs CLM</td><td>80%</td><td>99.98</td><td>0.02</td><td>0.01</td></tr><tr><td>BEN</td><td>GEO+CLM vs CLM</td><td>100%</td><td>99.61</td><td>0.28</td><td>0.11</td></tr><tr><td>BEN</td><td>COMP+CLM vs CLM</td><td>20%</td><td>100.00</td><td>0.00</td><td>0.00</td></tr><tr><td>BEN</td><td>COMP+CLM vs CLM</td><td>40%</td><td>99.96</td><td>0.02</td><td>0.02</td></tr><tr><td>BEN</td><td>COMP+CLM vs CLM</td><td>60%</td><td>99.80</td><td>0.13</td><td>0.07</td></tr><tr><td>BEN</td><td>COMP+CLM vs CLM</td><td>80%</td><td>98.06</td><td>1.19</td><td>0.75</td></tr><tr><td>BEN</td><td>COMP+CLM vs CLM</td><td>100%</td><td>99.14</td><td>0.48</td><td>0.38</td></tr><tr><td>BEN</td><td>GEO+CLM vs COMP+CLM</td><td>20%</td><td>88.98</td><td>1.08</td><td>9.94</td></tr><tr><td>BEN</td><td>GEO+CLM vs COMP+CLM</td><td>40%</td><td>99.54</td><td>0.31</td><td>0.15</td></tr><tr><td>BEN</td><td>GEO+CLM vs COMP+CLM</td><td>60%</td><td>58.85</td><td>1.47</td><td>39.68</td></tr><tr><td>BEN</td><td>GEO+CLM vs COMP+CLM</td><td>80%</td><td>97.27</td><td>1.44</td><td>1.30</td></tr><tr><td>BEN</td><td>GEO+CLM vs COMP+CLM 100%</td><td></td><td>99.79</td><td>0.09</td><td>0.12</td></tr><tr><td>BEN</td><td>CLM vs BASE</td><td>20%</td><td>99.86</td><td>0.09</td><td>0.05</td></tr><tr><td>BEN</td><td>CLM vs BASE</td><td>40%</td><td>0.34</td><td>97.66</td><td>1.99</td></tr><tr><td>BEN</td><td>CLM vs BASE</td><td>60%</td><td>99.99</td><td>0.01</td><td>0.00</td></tr><tr><td>BEN</td><td>CLM vs BASE</td><td>80%</td><td>99.96</td><td>0.03</td><td>0.01</td></tr><tr><td>BEN</td><td>CLM vs BASE</td><td>100%</td><td>16.48</td><td>29.68</td><td>53.84</td></tr><tr><td>DFC</td><td>GEO+CLM vs CLM</td><td>20%</td><td>97.68</td><td>0.45</td><td>1.87</td></tr><tr><td>DFC</td><td>GEO+CLM vs CLM</td><td>40%</td><td>1.29</td><td>1.40</td><td>97.31</td></tr><tr><td>DFC</td><td>GEO+CLM vs CLM</td><td>60%</td><td>0.24</td><td>1.68</td><td>98.08</td></tr><tr><td>DFC</td><td>GEO+CLM vs CLM</td><td>80%</td><td>0.56</td><td>1.42</td><td>98.02</td></tr><tr><td>DFC</td><td>GEO+CLM vs CLM</td><td>100%</td><td>0.15</td><td>0.67</td><td>99.19</td></tr><tr><td>DFC</td><td>COMP+CLM vs CLM</td><td>20%</td><td>98.89</td><td>0.30</td><td>0.81</td></tr><tr><td>DFC</td><td>COMP+CLM vs CLM</td><td>40%</td><td>5.01</td><td>0.98</td><td>94.01</td></tr><tr><td>DFC</td><td>COMP+CLM vs CLM</td><td>60%</td><td>13.62</td><td>1.28</td><td>85.10</td></tr><tr><td>DFC</td><td>COMP+CLM vs CLM</td><td>80%</td><td>4.83</td><td>0.59</td><td>94.57</td></tr><tr><td>DFC</td><td>COMP+CLM vs CLM</td><td>100%</td><td>2.24</td><td>0.23</td><td>97.53</td></tr><tr><td>DFC</td><td>GEO+CLM vs COMP+CLM</td><td>20%</td><td>0.01</td><td>0.16</td><td>99.83</td></tr><tr><td>DFC</td><td>GEO+CLM vs COMP+CLM</td><td>40%</td><td>0.08</td><td>0.48</td><td>99.45</td></tr><tr><td>DFC</td><td>GEO+CLM vs COMP+CLM</td><td>60%</td><td>0.31</td><td>63.84</td><td>35.85</td></tr><tr><td>DFC</td><td>GEO+CLM vs COMP+CLM</td><td>80%</td><td>0.04</td><td>1.64</td><td>98.32</td></tr><tr><td>DFC</td><td>GEO+CLM vs COMP+CLM 100%</td><td></td><td>0.10</td><td>16.67</td><td>83.23</td></tr><tr><td>DFC</td><td>CLM vs BASE</td><td>20%</td><td>94.83</td><td>0.42</td><td>4.76</td></tr><tr><td>DFC</td><td>CLM vs BASE</td><td>40%</td><td>99.39</td><td>0.07</td><td>0.54</td></tr><tr><td>DFC</td><td>CLM vs BASE</td><td>60%</td><td>29.70</td><td>0.35</td><td>69.95</td></tr><tr><td>DFC</td><td>CLM vs BASE</td><td>80%</td><td>21.49</td><td>0.29</td><td>78.22</td></tr><tr><td>DFC</td><td>CLM vs BASE</td><td>100%</td><td>0.60</td><td>0.83</td><td>98.57</td></tr></table>

Continued on next page

Table 2: Bayesian Test Probability on main metrics for Curriculum masking ratio (CLM) with Rope of 0.5
<table><tr><td>Dataset</td><td>Model</td><td></td><td>Epoch|P(left) P(right) P(rope)</td><td></td></tr><tr><td>LCZ</td><td>GEO+CLM vs CLM</td><td>20%</td><td>99.99 0.01</td><td>0.00</td></tr><tr><td>LCZ</td><td>GEO+CLM vs CLM</td><td>40%</td><td>99.99 0.01</td><td>0.00</td></tr><tr><td>LCZ</td><td>GEO+CLM vs CLM</td><td>60%</td><td>99.80 0.16</td><td>0.04</td></tr><tr><td>LCZ</td><td>GEO+CLM vs CLM</td><td>80%</td><td>98.95 0.75</td><td>0.30</td></tr><tr><td>LCZ</td><td>GEO+CLM vs CLM</td><td>100%</td><td>99.60 0.30</td><td>0.11</td></tr><tr><td>LCZ</td><td>COMP+CLM vs CLM</td><td>20%</td><td>99.84 0.13</td><td>0.03</td></tr><tr><td>LCZ</td><td>COMP+CLM vs CLM</td><td>40%</td><td>97.49 0.70</td><td>1.81</td></tr><tr><td>LCZ</td><td>COMP+CLM vs CLM</td><td>60%</td><td>99.69 0.24</td><td>0.07</td></tr><tr><td>LCZ</td><td>COMP+CLM vs CLM</td><td>80%</td><td>99.18 0.57</td><td>0.25</td></tr><tr><td>LCZ</td><td>COMP+CLM vs CLM</td><td>100%</td><td>99.46 0.40</td><td>0.14</td></tr><tr><td>LCZ</td><td>GEO+CLM vs COMP+CLM</td><td>20%</td><td>2.32 89.66</td><td>8.01</td></tr><tr><td>LCZ</td><td>GEO+CLM vs COMP+CLM</td><td>40%</td><td>100.00 0.00</td><td>0.00</td></tr><tr><td>LCZ</td><td>GEO+CLM vs COMP+CLM</td><td>60%</td><td>78.60 4.20</td><td>17.19</td></tr><tr><td>LCZ</td><td>GEO+CLM vs COMP+CLM</td><td>80%</td><td>40.34 8.07</td><td>51.59</td></tr><tr><td>LCZ</td><td>GEO+CLM vs COMP+CLM 100%</td><td></td><td>17.42 17.42</td><td>65.17</td></tr><tr><td>LCZ</td><td>CLM vs BASE</td><td>20%</td><td>99.89 0.09</td><td>0.01</td></tr><tr><td>LCZ</td><td>CLM vs BASE</td><td>40%</td><td>98.75 0.67</td><td>0.58</td></tr><tr><td>LCZ</td><td>CLM vs BASE</td><td>60%</td><td>97.61 1.79</td><td>0.60</td></tr><tr><td>LCZ</td><td>CLM vs BASE</td><td>80%</td><td>99.99 0.01</td><td>0.00</td></tr><tr><td>LCZ</td><td>CLM vs BASE</td><td>100%</td><td>6.05 18.41</td><td>75.54</td></tr></table>

Table 3: Bayesian Test Probability on Macro metrics with Rope of 0.5
<table><tr><td>Dataset</td><td>Model</td><td>Epoch (%)|</td><td></td><td>MoCo |P(left) P(right) P(rope)|P(left) P(right) P(rope)</td><td></td><td>MAE</td><td></td><td></td></tr><tr><td>BEN</td><td>GEO vs BASE</td><td>20%</td><td>96.00</td><td>2.83</td><td>1.17</td><td>99.33</td><td>0.58</td><td>0.08</td></tr><tr><td>BEN</td><td>GEO vs BASE</td><td>40%</td><td>98.32</td><td>1.27</td><td>0.40</td><td>99.95</td><td>0.04</td><td>0.01</td></tr><tr><td>BEN</td><td>GEO vs BASE</td><td>60%</td><td>93.30</td><td>4.85</td><td>1.85</td><td>99.22</td><td>0.66</td><td>0.12</td></tr><tr><td>BEN</td><td>GEO vs BASE</td><td>80%</td><td>98.04</td><td>1.24</td><td>0.72</td><td>96.10</td><td>3.17</td><td>0.74</td></tr><tr><td>BEN</td><td>GEO vs BASE</td><td>100%</td><td>97.53</td><td>1.71</td><td>0.76</td><td>95.84</td><td>2.30</td><td>1.86</td></tr><tr><td>BEN</td><td>COMP vs BASE</td><td>20%</td><td>92.36</td><td>5.16</td><td>2.48</td><td>98.16</td><td>1.59</td><td>0.25</td></tr><tr><td>BEN</td><td>COMP vs BASE</td><td>40%</td><td>97.86</td><td>1.59</td><td>0.55</td><td>99.69</td><td>0.23</td><td>0.09</td></tr><tr><td>BEN</td><td>COMP vs BASE</td><td>60%</td><td>92.50</td><td>5.33</td><td>2.17</td><td>98.70</td><td>1.08</td><td>0.22</td></tr><tr><td>BEN</td><td>COMP vs BASE</td><td>80%</td><td>98.65</td><td>0.86</td><td>0.49</td><td>96.42</td><td>2.89</td><td>0.68</td></tr><tr><td>BEN</td><td>COMP vs BASE</td><td>100%</td><td>99.20</td><td>0.59</td><td>0.21</td><td>97.91</td><td>1.35</td><td>0.73</td></tr><tr><td>BEN</td><td>GEO vs COMP</td><td>20%</td><td>81.90</td><td>4.07</td><td>14.03</td><td>74.48</td><td>15.82</td><td>9.70</td></tr><tr><td>BEN</td><td>GEO vs COMP</td><td>40%</td><td>55.45</td><td>8.07</td><td>36.48</td><td>97.99</td><td>0.70</td><td>1.31</td></tr><tr><td>BEN</td><td>GEO vs COMP</td><td>60%</td><td>47.68</td><td>32.93</td><td>19.39</td><td>89.70</td><td>2.27</td><td>8.03</td></tr><tr><td>BEN</td><td>GEO vs COMP</td><td>80%</td><td>21.55</td><td>28.58</td><td>49.87</td><td>28.75</td><td>25.87</td><td>45.38</td></tr><tr><td>BEN</td><td>GEO vs COMP</td><td>100%</td><td>1.79</td><td>92.17</td><td>6.04</td><td>0.01</td><td>99.94</td><td>0.05</td></tr><tr><td>DFC</td><td>GEO vs BASE</td><td>20%</td><td>81.19</td><td>3.33</td><td>15.47</td><td>99.95</td><td>0.04</td><td>0.01</td></tr><tr><td>DFC</td><td>GEO vs BASE</td><td>40%</td><td>96.89</td><td>0.95</td><td>2.16</td><td>98.55</td><td>0.50</td><td>0.95</td></tr><tr><td>DFC</td><td>GEO vs BASE</td><td>60%</td><td>99.94</td><td>0.03</td><td>0.03</td><td>92.70</td><td>0.54</td><td>6.76</td></tr><tr><td>DFC</td><td>GEO vs BASE</td><td>80%</td><td>99.31</td><td>0.18</td><td>0.51</td><td>99.59</td><td>0.02</td><td>0.39</td></tr><tr><td>DFC</td><td>GEO vs BASE</td><td>100%</td><td>99.65</td><td>0.15</td><td>0.20</td><td>0.10</td><td>0.05</td><td>99.85</td></tr><tr><td>DFC</td><td>COMP vs BASE</td><td>20%</td><td>86.29</td><td>2.15</td><td>11.56</td><td>99.99</td><td>0.01</td><td>0.00</td></tr><tr><td>DFC</td><td>COMP vs BASE</td><td>40%</td><td>98.43</td><td>0.47</td><td>1.10</td><td>98.75</td><td>0.37</td><td>0.88</td></tr><tr><td>DFC</td><td>COMP vs BASE</td><td>60%</td><td>99.90</td><td>0.04</td><td>0.06</td><td>69.97</td><td>0.87</td><td>29.15</td></tr><tr><td>DFC</td><td>COMP vs BASE</td><td>80%</td><td>99.55</td><td>0.18</td><td>0.27</td><td>1.13</td><td>0.14</td><td>98.72</td></tr><tr><td>DFC</td><td>COMP vs BASE</td><td>100%</td><td>99.45</td><td>0.19</td><td>0.36</td><td>0.42</td><td>0.94</td><td>98.64</td></tr><tr><td>DFC</td><td>GEO vs COMP</td><td>20%</td><td>1.11</td><td>1.36</td><td>97.53</td><td>84.92</td><td>2.10</td><td>12.97</td></tr><tr><td>DFC</td><td>GEO vs COMP</td><td>40%</td><td>2.42</td><td>2.80</td><td>94.78</td><td>1.85</td><td>0.33</td><td>97.82</td></tr><tr><td>DFC</td><td>GEO vs COMP</td><td>60%</td><td>78.95</td><td>0.09</td><td>20.96</td><td>0.23</td><td>0.03</td><td>99.74</td></tr><tr><td>DFC</td><td>GEO vs COMP</td><td>80%</td><td>0.05</td><td>96.56</td><td>3.39</td><td>95.93</td><td>0.01</td><td>4.06</td></tr><tr><td>DFC</td><td>GEO vs COMP</td><td>100%</td><td>0.34</td><td>0.00</td><td>99.66</td><td>0.63</td><td>0.15</td><td>99.22</td></tr></table>

Table 4: Results over the three seed for MoCo and MAE
<table><tr><td>MODEL Dataset Epoch (%)|</td><td></td><td>MoCo</td><td>MAE</td></tr><tr><td>BASE</td><td>BEN 20%</td><td>|51.12/49.42/50.37|</td><td>|35.48/35.71/36.31</td></tr><tr><td>BASE</td><td>BEN 40%</td><td>51.96/52.59/51.55</td><td>42.09/41.88/42.16</td></tr><tr><td>BASE</td><td>BEN 60%</td><td>52.6/51.85/52.7</td><td>39.85/39.84/39.98</td></tr><tr><td>BASE</td><td>BEN 80%</td><td>53.67/52.98/53.85</td><td>40.39/40.28/40.54</td></tr><tr><td>BASE</td><td>BEN 100%</td><td>53.88/53.27/53.78</td><td>45.42/45.06/45.39</td></tr><tr><td>COMP</td><td>BEN 20%</td><td>52.62/51.82/52.65</td><td>45.62/45.9/45.67</td></tr><tr><td>COMP</td><td>BEN 40%</td><td>55.93/55.51/56.86</td><td>47.4/47.6/47.55</td></tr><tr><td>COMP</td><td>BEN 60%</td><td>57.65/57.72/57.91</td><td>49.04/48.57/48.95</td></tr><tr><td>COMP</td><td>BEN 80%</td><td>58.49/58.34/58.47</td><td>49.28/49.31/49.36</td></tr><tr><td>COMP</td><td>BEN 100%</td><td>59.06/58.95/59.09</td><td>49.55/49.64/49.48</td></tr><tr><td>GEO</td><td>BEN 20%</td><td>52.22/51.59/52.11</td><td>47.49/47.53/47.34</td></tr><tr><td>GEO</td><td>BEN 40%</td><td>56.29/55.89/56.65</td><td>47.84/47.87/47.85</td></tr><tr><td>GEO</td><td>BEN 60%</td><td>57.29/57.58/57.63</td><td>49.1/49.23/ /49.15</td></tr><tr><td>GEO</td><td>BEN 80%</td><td>56.57/56.58/56.65</td><td>49.29/48.74/49.21</td></tr><tr><td>GEO</td><td>BEN 100%</td><td>58.49/58.43/58.54</td><td>48.08/47.61/48.11</td></tr><tr><td>BASE</td><td>DFC 20%</td><td>57.25/57.27/57.75</td><td>58.67/58.86/58.39</td></tr><tr><td>BASE</td><td>DFC 40%</td><td>57.86/57.76/57.84</td><td>60.22/60.53/60.11</td></tr><tr><td>BASE</td><td>DFC 60%</td><td>57.06/57.09/57.07</td><td>61.5/61.41/61.34</td></tr><tr><td>BASE</td><td>DFC 80%</td><td>57.55/57.66/57.67</td><td>61.68/61.53/61.71</td></tr><tr><td>BASE</td><td>DFC 100%</td><td>57.36/57.49/57.51</td><td>61.93/61.78/61.98</td></tr><tr><td>COMP</td><td>DFC 20%</td><td>57.82/57.82/57.77</td><td>61.05/60.69/60.79</td></tr><tr><td>COMP</td><td>DFC 40%</td><td>59.27/59.27/59.25</td><td>61.85/61.71/61.76</td></tr><tr><td>COMP</td><td>DFC 60%</td><td>59.01/59.06/59.02</td><td>61.96/62.12/62.09</td></tr><tr><td>COMP</td><td>DFC 80%</td><td>59.32/59.36/59.03</td><td>61.66/61.91/61.89</td></tr><tr><td>COMP</td><td>DFC 100%</td><td>59.05/59.1/59.05</td><td>61.77/61.98/62.02</td></tr><tr><td>GEO</td><td>DFC 20%</td><td>58.13/58.01/57.9</td><td>61.91/61.74/61.65</td></tr><tr><td>GEO</td><td>DFC 40%</td><td>59.35/59.37/59.32</td><td>61.96/61.84/61.92</td></tr><tr><td>GEO</td><td>DFC 60%</td><td>59.45/59.56/59.52</td><td>62.1/62.17/62.21</td></tr><tr><td>GEO</td><td>DFC 80%</td><td>58.92/59.01/58.95</td><td>62.15/62.32/62.35</td></tr><tr><td>GEO</td><td>DFC 100%</td><td>59.17/59.26/59.19</td><td>62.05/62.24/62.22</td></tr><tr><td>BASE</td><td>LCZ 20%</td><td>75.78/74.78/74.74</td><td>48.68/48.3/47.3</td></tr><tr><td>BASE</td><td>LCZ 40%</td><td>77.52/77.22/75.94|</td><td>60.18/59.18/60.1</td></tr><tr><td>BASE</td><td>LCZ 60%</td><td>78.36/77.44/78.04</td><td>61.52/60.8/58.46</td></tr><tr><td>BASE</td><td>LCZ 80%</td><td>78.94/78.48/78.52</td><td>64.9/64.72/63.94</td></tr><tr><td>BASE</td><td>LCZ 100%</td><td>79.36/78.44/78.54</td><td>70.36/69.56/69.92</td></tr><tr><td>COMP</td><td>LCZ 20%</td><td>80.34/79.06/79.18</td><td>66.24/67.96/67.8</td></tr><tr><td>COMP</td><td>LCZ 40%</td><td>82.32/81.96/82.26</td><td>72.1/72.14/72.5</td></tr><tr><td>COMP</td><td>LCZ 60%</td><td>81.48/81.74/81.8</td><td>72.56/73.1/73.62</td></tr><tr><td>COMP</td><td>LCZ 80%</td><td>82.3/82.1/82.5</td><td>73.06/73.04/74.22</td></tr><tr><td>COMP</td><td>LCZ 100%</td><td>82.52/81.96/82.02</td><td>75.12/75.44/75.44</td></tr><tr><td>GEO</td><td>LCZ 20%</td><td>82.42/81.42/82.16</td><td>70.52/70.72/72.06</td></tr><tr><td>GEO</td><td>LCZ 40%</td><td>81.92/81.4/81.9</td><td>73.68/73.0/74.1</td></tr><tr><td>GEO</td><td>LCZ 60%</td><td>82.16/82.3/82.2</td><td></td></tr><tr><td>GEO</td><td>LCZ 80%</td><td>82.3/82.46/82.46</td><td>74.64/74.82/74.62 74.98/74.72/75.94</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>GEO</td><td>LCZ</td><td>100% 82.28/82.54/82.74</td><td>73.14/73.56/74.1</td></tr></table>

Table 5: Results for Curriculum Masking (CLM) over three seeds
<table><tr><td>Model</td><td>Dataset Epoch (%)|</td><td></td><td>Performance</td></tr><tr><td>CLM</td><td>BEN</td><td>20%</td><td>40.67/40.7/40.99</td></tr><tr><td>CLM</td><td>BEN</td><td>40%</td><td>40.86/40.94/41.05</td></tr><tr><td>CLM</td><td>BEN</td><td>60%</td><td>45.11/45.24/45.31</td></tr><tr><td>CLM</td><td>BEN</td><td>80%</td><td>45.26/44.92/45.19</td></tr><tr><td>CLM</td><td>BEN</td><td>100%</td><td>45.46/45.33/44.57</td></tr><tr><td>COMP+CLM</td><td>BEN</td><td>20%</td><td>46.12/46.11/46.34</td></tr><tr><td>COMP+CLM</td><td>BEN</td><td>40%</td><td>43.48/43.53/43.77</td></tr><tr><td>COMP+CLM</td><td>BEN</td><td>60%</td><td>49.77/49.76/49.44</td></tr><tr><td>COMP+CLM</td><td>BEN</td><td>80%</td><td>48.9/49.69/48.52</td></tr><tr><td>COMP+CLM</td><td>BEN</td><td>100%</td><td>48.64/48.53/48.5</td></tr><tr><td>GEO+CLM</td><td>BEN</td><td>20%</td><td>47.04/47.15/46.95</td></tr><tr><td>GEO+CLM</td><td>BEN</td><td>40%</td><td>48.6/48.69/48.11</td></tr><tr><td>GEO+CLM</td><td>BEN</td><td>60%</td><td>50.15/50.25/50.21</td></tr><tr><td>GEO+CLM</td><td>BEN</td><td>80%</td><td>51.98/51.93/52.05</td></tr><tr><td>GEO+CLM</td><td>BEN</td><td>100%</td><td>50.98/51.1/51.1</td></tr><tr><td>CLM</td><td>DFC</td><td>20%</td><td>59.62/59.56/59.31</td></tr><tr><td>CLM</td><td>DFC</td><td>40%</td><td>61.25/61.57/61.04</td></tr><tr><td>CLM</td><td>DFC</td><td>60%</td><td>61.91/61.96/61.73</td></tr><tr><td>CLM</td><td>DFC</td><td>80%</td><td>62.15/62.01/62.05</td></tr><tr><td>CLM</td><td>DFC</td><td>100%</td><td>61.95/61.75/61.87</td></tr><tr><td>COMP+CLM</td><td>DFC</td><td>20%</td><td>60.98/61.21/60.99</td></tr><tr><td>COMP+CLM</td><td>DFC</td><td>40%</td><td>61.51/61.65/61.32</td></tr><tr><td>COMP+CLM</td><td>DFC</td><td>60%</td><td>62.27/62.1/62.14</td></tr><tr><td>COMP+CLM</td><td>DFC</td><td>80%</td><td>62.44/62.16/62.37</td></tr><tr><td>COMP+CLM</td><td>DFC</td><td>100%</td><td>62.23/61.95/62.18</td></tr><tr><td>GEO+CLM</td><td>DFC</td><td>20%</td><td>60.66/60.89/60.69</td></tr><tr><td>GEO+CLM</td><td>DFC</td><td>40%</td><td>61.26/61.46/61.11</td></tr><tr><td>GEO+CLM</td><td>DFC</td><td>60%</td><td>61.69/61.67/61.55</td></tr><tr><td>GEO+CLM</td><td>DFC</td><td>80%</td><td>62.08/61.81/61.97</td></tr><tr><td>GEO+CLM</td><td>DFC</td><td>100%</td><td>61.81/61.53/61.68</td></tr><tr><td>CLM</td><td>LCZ</td><td>20%</td><td>62.4/61.44/61.8</td></tr><tr><td>CLM</td><td>LCZ</td><td>40%</td><td>63.08/62.84/62.92</td></tr><tr><td>CLM</td><td>LCZ</td><td>60%</td><td>67.12/66.42/66.64</td></tr><tr><td>CLM</td><td>LCZ</td><td>80%</td><td>70.72/70.64/69.72</td></tr><tr><td>CLM</td><td>LCZ</td><td>100%</td><td>70.04/69.7/69.52</td></tr><tr><td>COMP+CLM</td><td>LCZ</td><td>20%</td><td>71.44/71.46/70.9</td></tr><tr><td>COMP+CLM</td><td>LCZ</td><td>40%</td><td>64.62/64.14/64.76</td></tr><tr><td>COMP+CLM</td><td>LCZ</td><td>60%</td><td>74.1/74.56/74.6</td></tr><tr><td>COMP+CLM</td><td>LCZ</td><td>80%</td><td>75.44/76.18/75.84</td></tr><tr><td>COMP+CLM</td><td>LCZ</td><td>100%</td><td>75.76/76.66/76.38</td></tr><tr><td>GEO+CLM</td><td>LCZ</td><td>20%</td><td>70.5/69.78/69.92</td></tr><tr><td>GEO+CLM</td><td>LCZ</td><td>40%</td><td>74.46/74.0/74.48</td></tr><tr><td>GEO+CLM</td><td>LCZ</td><td>60%</td><td>75.26/74.94/75.88</td></tr><tr><td>GEO+CLM</td><td>LCZ</td><td>80%</td><td>76.0/76.06/76.56</td></tr><tr><td>GEO+CLM</td><td>LCZ</td><td>100%</td><td>76.0/76.14/76.66</td></tr></table>

Table 6: Results for MAE and MoCo on Macro Metrics over three seeds
<table><tr><td>Model</td><td>Dataset Epoch|</td><td></td><td>MoCo</td><td>MAE</td></tr><tr><td>BASE</td><td>BEN</td><td>20%</td><td>|40.95/37.69/ /38.12|</td><td>16.2/17.88/ /17.09</td></tr><tr><td>BASE</td><td>BEN</td><td>40%</td><td>41.31/41.67/40.98</td><td>26.08 /25.78/ /25.73</td></tr><tr><td>BASE</td><td>BEN</td><td>60%</td><td>44.6/ /43.14/45.38</td><td>23.68 /21.57/20.59</td></tr><tr><td>BASE</td><td>BEN</td><td>80%</td><td>45.67/43.93/45.7</td><td>26.91 /22.77/22.78</td></tr><tr><td>BASE</td><td>BEN</td><td>100%</td><td>44.92 /43.87/45.29</td><td>27.53 /29.72 /28.77</td></tr><tr><td>COMP</td><td>BEN</td><td>20%</td><td>43.07/42.81/43.43</td><td>30.01 /27.76/ /32.06</td></tr><tr><td>COMP</td><td>BEN</td><td>40%</td><td>47.35/ /46.85/48.85</td><td>31.44 /31.82 /32.72</td></tr><tr><td>COMP</td><td>BEN</td><td>60%</td><td>48.32/50.32/48.88</td><td>32.36/ /32.63/ /32.19</td></tr><tr><td>COMP</td><td>BEN</td><td>80%</td><td>49.26/ /48.91/49.97</td><td>33.31 /32.12 /32.66</td></tr><tr><td>COMP</td><td>BEN</td><td>100%</td><td>50.81/51.17/51.16</td><td>33.38 /32.92/ /32.94</td></tr><tr><td>GEO</td><td>BEN</td><td>20%</td><td>44.48/43.27/44.72</td><td>30.85 /32.14/32.83</td></tr><tr><td>GEO</td><td>BEN</td><td>40%</td><td>48.55/47.16/49.07</td><td>33.47/33.88/34.23</td></tr><tr><td>GEO</td><td>BEN</td><td>60%</td><td>50.89/49.64/48.15</td><td>33.83 3/33.35/ /33.56</td></tr><tr><td>GEO</td><td>BEN</td><td>80%</td><td>49.72/49.0/49.14</td><td>32.74/ /33.01 /32.46</td></tr><tr><td>GEO</td><td>BEN</td><td>100%</td><td>49.18 /50.29/49.89</td><td>32.1/31.6/ /31.6</td></tr><tr><td>BASE</td><td>DFC</td><td>20%</td><td>41.18 /41.43/41.29</td><td>36.43/37.07/36.18</td></tr><tr><td>BASE</td><td>DFC</td><td>40%</td><td>41.6/40.96/41.49</td><td>42.72/43.43 /42.36</td></tr><tr><td>BASE</td><td>DFC</td><td>60%</td><td>40.74/40.83/41.01</td><td>44.32/44.44/44.05</td></tr><tr><td>BASE</td><td>DFC</td><td>80%</td><td>41.16/41.41/41.55</td><td>44.47/44.21/44.33</td></tr><tr><td>BASE</td><td>DFC</td><td>100%</td><td>40.97/41.27/41.43</td><td>44.91/44.53/44.72</td></tr><tr><td>COMP</td><td>DFC</td><td>20%</td><td>42.49/42.05/42.27</td><td>44.13/43.04/43.36</td></tr><tr><td>COMP</td><td>DFC</td><td>40%</td><td>43.06/ /42.88/43.12</td><td>44.88/44.22 /44.39</td></tr><tr><td>COMP</td><td>DFC</td><td>60%</td><td>43.0/43.16/43.17</td><td>44.87/44.8/44.91</td></tr><tr><td>COMP</td><td>DFC</td><td>80%</td><td>43.46/ /43.64/43.52</td><td>44.42/44.55/44.76</td></tr><tr><td>COMP</td><td>DFC</td><td>100%</td><td>43.03/ /43.24/43.17</td><td>44.46/44.55 6/44.85</td></tr><tr><td>GEO</td><td>DFC</td><td>20%</td><td>42.56/41.95/42.22</td><td>44.89/44.33/44.05</td></tr><tr><td>GEO</td><td>DFC</td><td>40%</td><td>43.04/42.99/42.97</td><td>45.05/44.5/44.56</td></tr><tr><td>GEO</td><td>DFC</td><td>60%</td><td>43.49/43.72/43.75</td><td>45.12/45.03/45.12</td></tr><tr><td>GEO</td><td>DFC</td><td>80%</td><td>42.79/43.0/42.93</td><td>44.98/45.13/45.3</td></tr><tr><td>GEO</td><td>DFC</td><td>100%</td><td>43.47/43.68/43.6</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>44.61/44.77/45.0</td></tr></table>