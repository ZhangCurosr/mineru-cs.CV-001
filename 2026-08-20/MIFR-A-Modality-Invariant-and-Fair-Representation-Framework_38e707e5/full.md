# MIFR: A Modality-Invariant and Fair Representation Framework for Skin Disease Classification

Asonyu Senge Njih<sup>1</sup>, Yvan Guifo Fodjo<sup>2</sup>, Vianney Kengne Tchendji<sup>1</sup>, Jerry Lacmou Zeutouo<sup>3</sup>, and Kerol Djoumessi<sup>4(B)</sup>

<sup>1</sup> Department of Mathematics and Computer Science, University of Dschang, Cameroon

<sup>2</sup> EFREI Research Lab, Université Paris-Panthéon-Assas, France 3 MIS, Université de Picardie Jules Verne, France

4 Hertie Institute for AI in Brain Health, University of Tübingen, Germany kerol.djoumessi-donteu@uni-tuebingen.de https://hertie.ai/

Abstract. Skin diseases represent a major global public health burden, yet machine learning tools developed to assist in their diagnosis sufer from two critical limitations: reliance on only one modality for diagnosis and systematic performance disparities across skin tones. While existing approaches address each challenge separately, this work proposes a modality-invariant framework with fair representation (MIFR) for skin disease classification. The architecture pairs clinical photographs with dermoscopic images using ViT-based encoders, projecting each input into a high-dimensional embedding space via modality-specific projection heads. The resulting model is trained with a five-component multiobjective loss including weighted cross-entropy for classification, confusion and skin-type classification losses for fairness, per-modality supervised contrastive loss for class alignment, and a modality-invariance loss for clinical and dermoscopic modality alignment. Experiments on the HIBA+Derm7pt paired dataset and the external PAD-UFES-20 and ISIC 2019 datasets showed that modality-invariant representation learning provides competitive predictive performance compare to relevant baseline models and competitive fairness on the internal dataset. t-SNE visualizations confirmed that clinical and dermoscopic embeddings of the same disease are geometrically aligned, validating the joint objectives.

Keywords: Deep representation learning · Fairness · Skin disease diagnosis · Modality invariance

## 1 Introduction

Skin diseases afect nearly one-third of the world’s population and represent a major global health burden [8]. Deep learning has brought dermatology to a new level, with CNNs and Vision Transformers matching or even surpassing dermatologists in diagnostic performance [5,16]. Yet, two major issues hinder their reliable deployment in real clinical settings. First, skin lesions are captured through heterogeneous imaging modalities, mostly clinical photographs and dermoscopic images, and relying on a single modality can be insuficient for accurate diagnosis [10, 13]. Second, most public dermatology datasets are dominated by lighter skin tones, causing models to perform significantly worse on darker skin and perpetuating healthcare disparities [4, 7]. Importantly, most existing fairness-aware methods are developed for single-modality clinical images [6], while recent multimodal foundation models do not incorporate fairness constraints [19], leaving the intersection of modality invariance and fairness largely unexplored.

Among fairness-aware approaches, FairDisCo [6] achieves skin-tone debiasing via disentanglement and contrastive learning, but operates on a single modality. FairAdaBN [18] replaces batch normalization with adaptive batch normalization, but requires sensitive-attribute labels at test time. PatchAlign [1] aligns image patches with clinical text descriptions to improve fairness, but depends on costly natural-language annotations and remains single-modal on images. In contrast, PanDerm [19] demonstrates the efectiveness of multimodal vision foundation models across clinical, dermoscopic, dermatopathological, and totalbody-photography images but it does not address demographic equity. These methods share three limitations: (i) they treat modality invariance and fairness as separate problems; (ii) most, such as FairAdaBN, require sensitive attribute labels at inference; and (iii) cross-dataset evaluation remains limited.

In this work, we propose a method that overcomes these gaps by: (1) designing a dual-encoder architecture that jointly processes clinical and dermoscopic images; (2) requiring FST annotations only for an auxiliary training objective, not at inference; (3) introducing a unified multi-objective loss that simultaneously enforces modality invariance and skin-tone fairness; and (4) rigorously evaluating cross-dataset generalization on two external benchmarks, PAD-UFES-20 and ISIC 2019, consisting of clinical and dermoscopic images respectively.

## 2 Proposed Modality-invariant and Fair Model

## 2.1 Problem Formulation

Let $\mathcal { D } _ { \mathrm { p a i r } } = \{ ( \mathbf { X } _ { i } ^ { c } , \mathbf { X } _ { i } ^ { d } , y _ { i } , s _ { i } ) \} _ { i = 1 } ^ { N }$ denote paired examples for a number of pairs $N _ { ☉ }$ , where $\mathbf { X } _ { i } ^ { c }$ and $\mathbf { X } _ { i } ^ { d }$ are clinical and dermoscopic images of the same lesion, $y _ { i } \in \{ 0 , 1 , . . . , c \}$ the disease class, and $s _ { i } \in \{ 1 , \ldots , 6 \}$ the Fitzpatrick Skin Type. Unpaired sets $\mathcal { D } _ { c }$ and $\mathcal { D } _ { d }$ are also available. The full training set consists of $D = \mathcal { D } _ { \mathrm { p a i r } } \cup \mathcal { D } _ { c } \cup \mathcal { D } _ { d }$ . We seek encoders $f _ { \theta } ^ { c } , \ f _ { \phi } ^ { d }$ to produce embeddings $Z _ { i } ^ { c }$ and $Z _ { i } ^ { d }$ from the inputs, and projection heads $g _ { \psi _ { c } } ^ { c } , \ g _ { \psi _ { d } } ^ { d }$ such that the shared embedding space $H ( Z ) \in \mathbb { R } ^ { D }$ satisfies: (i) a classifier $h _ { \omega } : Z \to Y$ produces accurate predictions $\hat { y } _ { i } = h _ { w } ( Z ) ; \mathrm { ( i i ) }$ paired embeddings $\lVert z _ { i } ^ { c } - z _ { i } ^ { d } \rVert$ are optimized for modality invariance; and (iii) the embedding z is decorrelated from sensitive attribute s for fairness. The overall architecture is illustrated in Figure 1.

![](images/312c8007999f46dd27e7ee7607feed7a9afc081496db2522703262437ce60fa2.jpg)  
Fig. 1: Overview of the proposed architecture. Clinical and dermoscopic streams are processed by independent encoders and projected into a shared embedding space via modality-specific projection heads. Four training branches jointly optimize classification $\left( L _ { \mathrm { c l s } } \right)$ , skin-tone fairness via adversarial confusion and skin-tone losses $\left( L _ { \mathrm { c o n f } } , \ L _ { s } \right)$ , cross-modality contrastive alignment $\left( L _ { \mathrm { c o n } } \right)$ 2 and paired modality invariance $( L _ { \mathrm { M I } } )$

## 2.2 Overall Architecture

The clinical encoder $f _ { \theta } ^ { c }$ and dermoscopic encoder $f _ { \phi } ^ { d }$ independently process input images (Fig. 1). Each is followed by a modality-specific projection head; a simple multi-layer perceptron with batch normalization and GELU activations, ending with L2 normalization, that maps the encoder outputs into a shared Ddimensional embedding space where the cosine distance between embeddings directly reflects their semantic similarity. These projection heads have independent parameters, producing $\mathbf { z } _ { i } ^ { c } = g _ { \psi _ { c } } ^ { c } ( f _ { \theta } ^ { c } ( x _ { i } ^ { c } ) )$ and $\mathbf { \bar { z } } _ { i } ^ { d } = g _ { \psi _ { d } } ^ { d } ( f _ { \phi } ^ { d } ( x _ { i } ^ { d } ) ) \in \mathbb { R } ^ { 5 1 2 }$ , meaning the embeddings are converted to a D-dimension. The shared embedding space Z is where all branches contribute to the final classification.

## 2.3 Training Branches and Loss Functions

The embedding space is structured by four parallel branches: target classification, skin-tone attribute, contrastive, and modality-invariant.

Target branch (BASE). The weighted cross-entropy loss used inverse-frequency class weights $( w _ { y _ { i } } \propto 1 / n _ { y _ { i } } )$ to address class imbalance:

$$
\mathcal { L } _ { \mathrm { c l s } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } w _ { y _ { i } } \log p ( y _ { i } \mid z _ { i } ) .\tag{1}
$$

Skin-tone Attribute (SA) branch. This branch removes skin-tone bias from the learned features. An auxiliary classifier $f _ { s }$ is attached to predict skin tone from those features. Training involves two competing losses that pull in opposite directions [6]. First, the confusion loss ${ \mathcal { L } } _ { \mathrm { c o n f } }$ updates only the main encoder by removing skin-tone information from the representations. It forces the classifier’s predictions to become as uncertain as possible, and is minimized when the classifier outputs equal probability for all six skin-tone classes. Second, the skin loss $\mathcal { L } _ { \mathrm { s k i n } }$ updates only the classifier. It trains the classifier to predict the true skin-tone label from the current features. This prevents the classifier from collapsing and ensures it continues providing useful feedback.

Together, these losses create an adversarial game: the encoder tries to hide skin-tone cues, while the classifier tries to detect them. Over time, the encoder learns features that are useful for disease diagnosis but contain no skin-tone information. This adversarial disentanglement mechanism is computed as:

$$
\mathcal { L } _ { \mathrm { c o n f } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log ( p _ { s } ^ { i } ) , \quad \mathcal { L } _ { \mathrm { s k i n } } = - \sum _ { k = 1 } ^ { 6 } \log f _ { s } ( z ) _ { k } .\tag{2}
$$

Contrastive branch. A supervised contrastive loss pulls same-class embeddings together and pushes diferent classes apart [6], computed separately per modality to establish intra-class compactness before cross-modal alignment:

$$
\mathcal { L } _ { \mathrm { c o n t r } } = - \frac { 1 } { | P _ { y } | } \sum _ { p \in P _ { y } } \log \frac { \exp ( \varPsi ( r , p ) / \tau ) } { \exp ( \varPsi ( r , p ) / \tau ) + \sum _ { n \in N _ { y } } \exp ( \varPsi ( r , n ) / \tau ) } ,\tag{3}
$$

where $r = H ( Z )$ is the shared embedding space, p and n are the distances between an embedding and another embedding in the same class, and in a diferent class respectively, $\varPsi ( . , . )$ is cosine similarity and $\tau > 0$ a temperature.

Modality-invariant branch. This is our core contribution to enforce modality invariance without requiring annotations. The branch employs a VICReginspired loss [2] with three complementary terms:

– Invariance term $\mathcal { L } _ { \mathrm { i n v } }$ minimizes the cosine distance between paired clinical and dermoscopic embeddings, forcing the representation to be determined by disease content rather than imaging modality:

$$
\mathcal { L } _ { \mathrm { i n v } } = \frac { 1 } { | P | } \sum _ { ( i , j ) \in P } \left( 1 - \frac { z _ { i } ^ { c } \cdot z _ { j } ^ { d } } { \| z _ { i } ^ { c } \| \| z _ { j } ^ { d } \| } \right) .\tag{4}
$$

– Variance term ${ \mathcal { L } } _ { \mathbf { v a r } }$ prevents dimensional collapse by ensuring that each embedding dimension retains suficient variance across the batch, applied separately to clinical and dermoscopic streams:

$$
\mathcal { L } _ { \mathrm { v a r } } = \frac { 1 } { 2 d } \sum _ { m \in \{ c , d \} } \sum _ { k = 1 } ^ { d } \operatorname* { m a x } \left( 0 , \gamma - \sqrt { \mathrm { V a r } ( z _ { . k } ^ { m } ) + \epsilon } \right) ,\tag{5}
$$

where $\gamma = 1 . 0$ is a target variance floor and ϵ a small stabilizer.

Table 1: Dataset statistics after filtering to the target classes (MEL, NV, BCC).
<table><tr><td>Dataset</td><td> $\boxed { \# }$  Samples</td><td></td><td>Retained Melanoma</td><td>Nevus</td><td>BCC</td></tr><tr><td>HIBA [15]</td><td>1,616</td><td>1,195</td><td>253</td><td>602</td><td>340</td></tr><tr><td>[Derm7pt [9]</td><td>2,022</td><td>1,738</td><td>504</td><td>1,150</td><td>84</td></tr><tr><td>PAD-UFES-20 [14]</td><td>2,098</td><td>1,141</td><td>52</td><td>244</td><td>845</td></tr><tr><td>ISIC 2019 [3]</td><td>33,769</td><td>25,517</td><td>5,849</td><td>15,370</td><td>4,298</td></tr></table>

– Covariance term $\mathcal { L } _ { \mathbf { c o v } }$ decorrelates embedding dimensions to reduce redundancy and increase efective capacity:

$$
\mathcal { L } _ { \mathrm { c o v } } = \frac { 1 } { 2 d } \sum _ { m \in \{ c , d \} } \sum _ { i \neq j } [ C ^ { m } ] _ { i j } ^ { 2 } ,\tag{6}
$$

where $\begin{array} { r } { C ^ { m } = \frac { 1 } { N - 1 } ( Z ^ { m } ) ^ { \top } Z ^ { m } } \end{array}$ is the batch covariance matrix.

The total MI loss combines these terms:

$$
{ \mathcal { L } } _ { \mathrm { M I } } = \lambda _ { \mathrm { i n v } } { \mathcal { L } } _ { \mathrm { i n v } } + \lambda _ { \mathrm { v a r } } { \mathcal { L } } _ { \mathrm { v a r } } + \lambda _ { \mathrm { c o v } } { \mathcal { L } } _ { \mathrm { c o v } } .\tag{7}
$$

The complete objective jointly optimizes all components:

$$
\mathcal { L } _ { \mathrm { T o t a l } } = \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { c o n f } } \mathcal { L } _ { \mathrm { c o n f } } + \lambda _ { \mathrm { s k i n } } \mathcal { L } _ { \mathrm { s k i n } } + \lambda _ { \mathrm { c o n t r } } \mathcal { L } _ { \mathrm { c o n t r } } + \lambda _ { \mathrm { M I } } \mathcal { L } _ { \mathrm { M I } } .\tag{8}
$$

At inference, the auxiliary branches are discarded, retaining only the encoders, projection heads, and target classification branch. Compared to baselines (FairDisCo and PatchAlign), the resulting model can directly operate on clinical images, dermoscopic images, or both, enabling flexible deployment in resource-constrained settings where only a single imaging modality may be available, thereby eliminating the need for separate specialized models for each modality.

## 3 Experiments

## 3.1 Datasets

We classify the images into three classes: Melanoma (MEL), Nevus (NV), and Basal Cell Carcinoma (BCC). The training dataset consisted of two paired datasets (Table 1). HIBA [15] has 1,195 images with FST annotations. Its class proportions are 21.2% MEL, 50.4% NV, and 28.4% BCC. Derm7pt [9] has 1,738 images with FST estimated using the Individual Typology Angle, following the implementations of [7]. Its class proportions are 29.0% MEL, 66.2% NV, and 4.8% BCC. For both datasets, only Melanoma, Nevus, and Basal Cell Carcinoma classes were retained, focusing on the most clinically relevant and wellrepresented classes. Data were stratified by disease class and FST. We split it into 70% training, 10% validation, and 20% test. All images were resized to

$2 2 4 \times 2 2 4$ pixels and online augmentations were applied, including random resized crop, color jitter, and horizontal flip with a given probability $( p = 0 . 8 5 )$ . For cross-dataset evaluation, we used two external benchmarks. PAD-UFES-20 [14] contains 1,141 clinical images with FST labels. Its class proportions are 4.6% MEL, 21.4% NV, and 74.0% BCC. ISIC 2019 [3] has 25,517 dermoscopic images. Its class proportions are 22.9% MEL, 60.2% NV, and 16.8% BCC.

## 3.2 Experimental Setup

Model training and hyperparameters. Based on preliminary experiments, the ViT-Small (patch16-224) model pre-trained on ImageNet was adopted as backbone encoder. Key hyperparameters were selected on the validation split, with the final configuration balancing The loss weights were set to $\lambda _ { \mathrm { c o n f } } = 0 . 3$ $\lambda _ { \mathrm { s k i n } } = 0 . 2$ , and $\lambda _ { \mathrm { c o n t r } } = 0 . 5$ , prioritizing the contrastive alignment over skin classification. The modality-invariant term was assigned a smaller weight $\lambda _ { \mathrm { M I } } =$ 0.15 after adjusting, since a larger value would have overly suppressed taskrelevant variations. For the sub-weights of the variance-invariance-covariance regularizer, following [2], we kept $\lambda _ { \mathrm { i n v } } = 1 . 0$ and $\lambda _ { \mathrm { v a r } } = 1 . 0$ to enforce strict invariance and variance preservation, while setting $\lambda _ { \mathrm { c o v } } = 0 . 0 4$ to apply only mild decorrelation pressure. The temperature $\tau = 0 . 1$ follows standard contrastive practice, and the variance floor $\gamma = 1 . 0$ ensures numerical stability. A batch size of 32 was used due to memory constraints, and AdamW optimizer [11] with separate learning rates of $8 . 5 \times 1 0 ^ { - 5 }$ for the encoders and $1 \times 1 0 ^ { - 4 }$ for the classifiers and projection heads. Weight decay of $1 \times 1 0 ^ { - 4 }$ and 50 warmup epochs were used to stabilize early training, while label smoothing of 0.01 mitigates overfitting to noisy labels. All models were trained for 500 epochs on the MatriCS platform using one Nvidia GPU, 16 CPU cores, and 128GB of host memory.

Model Evaluation. Our model is compared with two main baseline: FairDisCo [6] and PatchAlign [1]. The predictive performance is assessed via standard classification metrics: Accuracy (Acc), F1-score, and Area Under the ROC Curve (AUC). Fairness is evaluated following [6] and [12] by computing: Predictive Quality Disparity (PQD) as the ratio of minimum to maximum class-wise accuracy across skin tones; and Accuracy Gap (∆Acc) as the diference between highest and lowest group accuracies.

## 4 Results

## 4.1 Ablation Study

We first perform ablation experiments on the internal test sets to isolate the contribution of each loss component (Tab. 2). Starting from a dual-encoder baseline with classification loss only (BASE), we progressively added the skintone attribute branch (SA), contrastive branch (Contr), and Modality-Invariant branch (MI). On the internal test set, SA alone improves accuracy from 93.13% at the BASE, to 93.52%, and reduces the Acc gap from 0.0691 to 0.0562. MI alone improves the BASE PQD from 66.67% to 87.23%. The combination BASE+Contr+MI yields the lowest Acc gap (0.0265), suggesting contrastive learning with modality invariance implicitly promotes fairness even without explicit SA constraints. Removing MI from the full model increases Acc gap from 0.054 to 0.0747, confirming that modality invariance provides complementary fairness regularization.

## 4.2 Comparison with State-of-the-Art

HIBA+Derm7pt internal test set. Table 2 compares the full model against FairDisCo and PatchAlign. Our model achieves 93.33% accuracy and 92.17% F1- score, which does not surpass but remains highly competitive with FairDisCo (93.81%) and PatchAlign (93.81%). Crucially, our model reaches a PQD of 87.23%, substantially surpassing FairDisCo (66.67%) and PatchAlign (82.98%).

PAD-UFES-20 external benchmark. The full model consistently outperforms state-of-the-art methods in Acc (85.10%), AUC (91.89%), and F1-score (66.25%). This clearly surpasses FairDisCo (78.35% Acc, 83.44% AUC) and PatchAlign (75.02% Acc, 88.66% AUC). Our model prioritizes overall diagnostic accuracy and robust cross-dataset generalization, validating its real-world applicability although PatchAlign exhibits a higher PQD (0.8878) than ours (0.54). This fairness gap likely reflects distributional shift: PAD-UFES-20 is heavily skewed toward BCC (74% vs. 28.4% and 4.8% in HIBA and Derm7pt), which may interact unevenly with underrepresented skin-tone subgroups.

ISIC 2019 external benchmark. The full model achieves 76.64% Acc and 70.62% F1-score (matching FairDisCo and surpassing PatchAlign), while its AUC (80.50%) is significantly lower than BASE (86.62%). This is a predictable trade-of for modal invariance: the dermoscopic encoder regularized to align with clinical embeddings sacrifices some modality-specific discrimination for crossmodality generalizability, a modest cost given the substantial gain achieved on PAD-UFES-20. Fairness metrics were not computed on ISIC 2019 because, as a purely dermoscopic dataset, skin surface is not visible in the images, making skin tone dificult to estimate, therefore impacting fairness evaluation.

## 4.3 Embedding Space Visualizations

We produced t-SNE plots of the full model on the internal test set, making it easier to understand how the shared embedding space is organized in 2D scatter plots and how alignment takes place. With these considerations in mind, the resulting visualizations (Fig. 2) reveal interleaved clinical-dermoscopic embeddings (confirming modality invariance), and near-perfect mixing of skin tones within each cluster (confirming fairness).

Table 2: Results on internal and external datasets: ablations and state-of-the-art comparison. The best values are in bold, and the second are underlined
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="3">Performance</td><td colspan="2">Fairness</td></tr><tr><td>Acc ↑</td><td>AUC ↑</td><td>F1-Score ↑</td><td>∆Acc↓</td><td>PQD ↑</td></tr><tr><td rowspan="10">HIBA+Derm7pt</td><td>BASE</td><td>.9313</td><td>.9738</td><td>.9165</td><td>.0691</td><td>.6667</td></tr><tr><td>BASE+SA</td><td>.9352</td><td>.9813</td><td>.9226</td><td>.0562</td><td>.6667</td></tr><tr><td>BASE+Contr</td><td>.9284</td><td>.9683</td><td>.9142</td><td>.0887</td><td>.8298</td></tr><tr><td>BASE+MI</td><td>.9188</td><td>.9619</td><td>.9001</td><td>.0394</td><td>.8723</td></tr><tr><td>BASE+Contr+MI</td><td>.9043</td><td>.9664</td><td>.8883</td><td>.0265</td><td>.8511</td></tr><tr><td>BASE+SA+MI</td><td>.9236</td><td>.9560</td><td>.9053</td><td>.0546</td><td>.8871</td></tr><tr><td>BASE+SA+Contr</td><td>.9333</td><td>.9748</td><td>.9186</td><td>.0747</td><td>.8511</td></tr><tr><td>FairDisCo</td><td>.9381</td><td>.9790</td><td>.9268</td><td>.0769</td><td>.6667</td></tr><tr><td>PatchAlign</td><td>.9381</td><td>.9792</td><td>.9244</td><td>.0988</td><td>.8298</td></tr><tr><td>All</td><td>.9333</td><td>.9678</td><td>.9217</td><td>.0540</td><td>.8723</td></tr><tr><td rowspan="4">PAD-UFES-20</td><td>BASE</td><td>.7528</td><td>.8404</td><td>.5040</td><td>.2066</td><td>.6053</td></tr><tr><td>FairDisCo</td><td>.7835</td><td>.8344</td><td>.5421</td><td>.1479</td><td>.5631</td></tr><tr><td>PatchAlign</td><td>.7502</td><td>.8866</td><td>.5935</td><td>.0760</td><td>.8878</td></tr><tr><td>All</td><td>.8510</td><td>.9189</td><td>.6625</td><td>.1413</td><td>.5400</td></tr><tr><td rowspan="4">ISIC 2019</td><td>BASE</td><td>.7586</td><td>.8662</td><td>.6998</td><td></td><td></td></tr><tr><td>FairDisCo</td><td>.7560</td><td>.8335</td><td>.7050</td><td></td><td></td></tr><tr><td>PatchAlign</td><td>.7337</td><td>.8626</td><td>.6650</td><td></td><td></td></tr><tr><td>All</td><td>.7664</td><td>.8050</td><td>.7062</td><td></td><td></td></tr></table>

![](images/d5fc1a0d05710e8a055297c309077febc06c45ae6ef02ac331491e4e78acc81d.jpg)

![](images/871f7cc7da2790e3bd3518a1739df293ffb70f6ea8464253458390a880e26103.jpg)  
Fig. 2: Left: t-SNE of the full model colored by imaging modality, showing alignment of clinical and dermoscopic embeddings. Right: t-SNE colored by Fitzpatrick skin type, showing near-perfect skin-tone mixing within disease clusters.

## 5 Discussion

The dual ViT-Small design has about 45M parameters, which limited the batch size to 32 and needed separate learning rates for the encoders and the classifiers. Among the five loss weights, the three MI sub-weights $( \lambda _ { \mathrm { i n v } } , \lambda _ { \mathrm { v a r } } , \lambda _ { \mathrm { c o v } } )$ were carefully tuned: a small covariance penalty reduced redundancy without hurting quality, while high invariance and variance enforced strong invariance and prevented collapse. Because the five weights interact, they were tuned one at a time on the validation split rather than searching all combinations, which is practical but does not guarantee optimal performance. A more principled way for hyperparameter selection could lead to better performance while reducing complexity. Fairness supervision also relies on estimated Fitzpatrick skin types for Derm7pt [7], adding some label noise, and the AUC drop in ISIC 2019 shows that modality invariance costs some modality-specific detail, which was a tradeof given the fairness and generalization gains. Finally, all datasets come from International repositories, with no African skin lesion data, reflecting the broader scarcity of paired, skin-tone-labeled dermatology data from African populations. This gap is particularly relevant given that public dermatology datasets remain skewed toward lighter Fitzpatrick types (I–III) [4, 7], while African populations are concentrated in FST IV–VI, the range where prior fairness methods have historically underperformed. Adapting MIFR to African clinical datasets, such as eSkinHealth [17], represents an important direction for future work.

## 6 Conclusion

This work proposed a Modality-Invariant and Fair Representation Learning (MIFR) framework for equitable skin disease classification across clinical and dermoscopic imaging. The model demonstrates superior cross-dataset generalization, achieving an AUC improvement of over 3.2% on PAD-UFES-20 compared with the best-performing baseline (PatchAlign) and over 7.8% compared with the base model (BASE), while achieving substantial fairness gains on the internal dataset and maintaining competitive predictive performance across datasets. t-SNE visualizations confirm modality-aligned embeddings, validating that joint optimization of invariance and fairness yields more robust and equitable models. However, MIFR has two key limitations. First, it relies on paired training data, which are often scarce. Second, its dual-encoder architecture increases computational cost. Future work should investigate unpaired approaches to modality invariance representation learning and develop more eficient architectures to facilitate deployment in resource-constrained settings.

Acknowledgments. This work was granted access to HPC resources of “Plateforme MatriCS” within the University of Picardie Jules Verne. “Plateforme MatriCS” is cofinanced by the European Union with the European Regional Development Fund (FEDER) and the Hauts-De-France Regional Council among others. We thank the Hertie Foundation and the Excellence Cluster 2064 “Machine Learning – New Perspectives for Science” for supporting this work.

Disclosure of Interests. The authors declare no competing interests.

## References

1. Aayushman, Gaddey, H., Mittal, V., Chawla, M., Gupta, G.R.: Fair and Accurate Skin Disease Image Classification by Alignment with Clinical Labels. In: Linguraru, M.G., Dou, Q., Feragen, A., Giannarou, S., Glocker, B., et al. (eds.) Medical Image Computing and Computer Assisted Intervention – MIC-CAI 2024, vol. 15001, pp. 394–404. Springer Nature Switzerland, Cham (2024). https://doi.org/10.1007/978-3-031-72378-0\_37

2. Bardes, A., Ponce, J., LeCun, Y.: VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning (Jan 2022). https://doi.org/10.48550/ arXiv.2105.04906

3. Combalia, M., Codella, N.C.F., Rotemberg, V., Helba, B., Vilaplana, V., et al.: BCN20000: Dermoscopic lesions in the wild. arXiv preprint arXiv:1908.02288 (2019)

4. Daneshjou, R., Vodrahalli, K., Novoa, R.A., Jenkins, M., Liang, W., et al.: Disparities in dermatology AI performance on a diverse, curated clinical image set. Science Advances 8(32), eabq6147 (Aug 2022). https://doi.org/10.1126/sciadv.abq6147

5. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., et al.: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Jun 2021). https://doi.org/10.48550/arXiv.2010.11929

6. Du, S., Hers, B., Bayasi, N., Hamarneh, G., Garbi, R.: FairDisCo: Fairer AI in Dermatology via Disentanglement Contrastive Learning (Aug 2022). https://doi. org/10.48550/arXiv.2208.10013

7. Groh, M., Harris, C., Soenksen, L., Lau, F., Han, R., et al.: Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset. In: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). pp. 1820–1828. IEEE, Nashville, TN, USA (Jun 2021). https://doi.org/10.1109/CVPRW53098.2021.00201

8. Karimkhani, C., Colombara, D.V., Drucker, A.M., Norton, S.A., Hay, R., et al.: The global burden of scabies: A cross-sectional analysis from the Global Burden of Disease Study 2015. The Lancet Infectious Diseases 17(12), 1247–1254 (Dec 2017). https://doi.org/10.1016/S1473-3099(17)30483-8

9. Kawahara, J., Daneshvar, S., Argenziano, G., Hamarneh, G.: 7-point checklist and skin lesion classification using multi-task multi-modal neural nets. IEEE Journal of Biomedical and Health Informatics 23(2), 538–546 (2019)

10. Kemeness, C., Schandua, H., Spangenberg, A., Gupta, R., Muraguri, I., et al.: Modified Goeckerman therapy for chronic skin diseases in a resource-limited setting: A case series from Kenya. JAAD Case Reports 74, 70–74 (2026). https: //doi.org/10.1016/j.jdcr.2026.05.042

11. Kingma, D.P., Ba, J.: Adam: A Method for Stochastic Optimization (Jan 2017). https://doi.org/10.48550/arXiv.1412.6980

12. Kong, Q., Chiu, C.H., Zeng, D., Chen, Y.J., Ho, T.Y., et al.: Achieving Fairness Through Channel Pruning for Dermatological Disease Diagnosis. In: Linguraru, M.G., Dou, Q., Feragen, A., Giannarou, S., Glocker, B., et al. (eds.) Medical Image Computing and Computer Assisted Intervention – MICCAI 2024, vol. 15010, pp. 24–34. Springer Nature Switzerland, Cham (2024). https://doi.org/10.1007/ 978-3-031-72117-5\_3

13. Nawaz, A., Edinat, A., Rana, M.R.R., Ali, T., Mustafa, G., et al.: The role of multimodality in clinical disease diagnosis: Advances, challenges, and opportunities. Frontiers in Public Health Volume 14 - 2026 (2026). https://doi.org/10.3389/ fpubh.2026.1788454

14. Pacheco, A.G., Lima, G.R., Salomão, A.S., Krohling, B., Biral, I.P., et al.: PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones. Data in Brief 32, 106221 (Oct 2020). https://doi.org/ 10.1016/j.dib.2020.106221

15. Ricci Lara, M.A., Rodríguez Kowalczuk, M.V., Lisa Eliceche, M., Ferraresso, M.G., Luna, D.R., et al.: A dataset of skin lesion images collected in Argentina for the evaluation of AI tools in this population. Scientific Data 10(1), 712 (Oct 2023). https://doi.org/10.1038/s41597-023-02630-0

16. Takahashi, S., Sakaguchi, Y., Kouno, N., Takasawa, K., Ishizu, K., et al.: Comparison of Vision Transformers and Convolutional Neural Networks in Medical Image

Analysis: A Systematic Review. Journal of Medical Systems 48(1), 84 (Sep 2024). https://doi.org/10.1007/s10916-024-02105-8

17. Wang, J., Hu, X., Zhang, Y., Almamy, D., Bamba, V., et al.: eskinhealth: a multimodal dataset for neglected tropical skin diseases. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 12949–12956 (2025)

18. Xu, Z., Zhao, S., Quan, Q., Yao, Q., Zhou, S.K.: FairAdaBN: Mitigating unfairness with adaptive batch normalization and its application to dermatological disease classification (Jul 2023). https://doi.org/10.48550/arXiv.2303.08325

19. Yan, S., Yu, Z., Primiero, C., Vico-Alonso, C., Wang, Z., et al.: A multimodal vision foundation model for clinical dermatology. Nature Medicine 31(8), 2691–2702 (Aug 2025). https://doi.org/10.1038/s41591-025-03747-y