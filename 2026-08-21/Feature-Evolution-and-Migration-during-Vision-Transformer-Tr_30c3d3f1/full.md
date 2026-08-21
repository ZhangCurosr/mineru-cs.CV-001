# Feature Evolution and Migration during Vision Transformer Training<sup>∗</sup>

Joonas Järve   
Institute of Computer Science   
University of Tartu   
Tartu, Estonia   
joonas.jarve@ut.ee   
Tarun Khajuria   
Institute of Computer Science   
University of Tartu   
Tartu, Estonia   
tarun.khajuria@ut.ee

## Abstract

We present a novel view on feature evolution in Vision Transformers (ViTs) by visualizing the training process over two dimensions – network depth (layer) and training time (epochs). We employ Sparse Autoencoders (SAEs) to extract candidate sparse features from CLStoken representations and compare their activation profiles across epoch–layer pairs. This allows us to study feature-level dynamics that are not directly visible from representation-level similarity measures. Furthermore, we demonstrate how this framework of feature evolution allows us to describe feature migration, the change in the layer where a feature is most detectable during training. Our experiments show that migration is concentrated early in training, occurs more often toward earlier layers than toward deeper layers, and declines as feature organization stabilizes. We further find that deeper layers stabilize earlier and more strongly than shallow lay ers. The results show that our approach can be employed as a tool for understanding how ViTs learn and evolve.

## CCS Concepts

• Computing methodologies → Neural networks; Image representations.

## Keywords

Feature evolution, Sparse autoencoders, Linear probing, Represen tation dynamics

## 1 Introduction

Vision Transformers (ViTs) [14] have become standard backbones for visual recognition and multimodal learning, but we still have limited understanding of how their internal visual features are

Halil Ibrahim Aysel   
Institute of Computer Science   
University of Tartu   
Tartu, Estonia   
halil.ibrahim.aysel@ut.ee   
Meelis Kull   
Institute of Computer Science   
University of Tartu   
Tartu, Estonia   
meelis.kull@ut.ee

formed during training. Prior work has shown that both convolutional neural networks (CNNs) [41] and ViTs exhibit broad hierarchical structure across depth, progressing from simpler patterns in early layers toward more semantic structure in deeper layers [13, 16, 22, 25, 33]. Less is known about how it develops: when do visual features first appear, how long do they persist, and do they remain in the same layers throughout training? Existing analysis

![](images/47aa6c401feef9e0de307dde259038e85a4451e7ce8e429c628695c461fb012a.jpg)  
Figure 1: Examples of distinct feature-evolution types observed during ViT training, shown in an epoch-by-layer view. ★ represents the reference feature.

tools operate either at the level of individual neurons, which are often entangled, or at the level of whole representations, where scalar similarity scores can obscure which features are actually shared between representations. In particular, alignment measures such as CKA [38] are informative about representation-level similarity, but do not reveal whether two layers or checkpoints contain the same features [62], rendering feature evolution tracking dificult. Linear probes [1] can track predefined human-labeled attributes, but only for attributes that are already annotated. As Yun et al. [65] put it: “Probing tasks can only verify whether a certain prior structure is learned in a language model. They can not reveal the structures beyond our prior knowledge.” We therefore need feature-level methods that can track both known human-interpretable attributes and features discovered directly from model activations.

In this work, we study ViT training along two axes: time, represented by training checkpoints, and depth, represented by network layers. As a growing body of literature suggests that sparse autoencoders (SAEs) [26, 49] can recover more disentangled and often more monosemantic features than raw hidden units, partly mitigating the efects of superposition [4, 26], we base our feature discovery on SAEs. At each checkpoint and layer, we analyze the model’s representations using SAEs to extract sparse latent features from the model’s activations. Throughout the paper, we use the term feature broadly: some features are human-interpretable, while others may be model-internal factors [59] without a simple semantic label. This distinction is particularly important in vision, where meaningful internal factors need not align with natural-language categories [30].

This two-dimensional view allows us to study feature migration. We say that a feature migrates when the layer where it is most detectable changes during training. For example, a feature that is initially visible only in deeper layers may later become detectable in earlier layers (example a in Figure 1), suggesting that the model has reorganized where that information is represented. Conversely, some features may remain local and become increasingly stable as training progresses (example b in Figure 1) and some spread across layers (example c in Figure 1).

Our experiments use this framework to ask three questions. First, how do visual features emerge and stabilize across ViT layers during training? Second, do features migrate from deeper to earlier layers, or from earlier to deeper layers? Third, how do these dynamics depend on the dataset scale? We find that feature migration is most pronounced early in training, that deeper-layer features become more stable over time, and dataset scale does not influence most of the migration metrics.

Our contributions are as follows:

(1) We introduce an epoch-by-layer framework for tracking visual features in ViTs during training.

(2) We define and measure feature migration, showing how features shift across layers as training progresses.

(3) We provide empirical evidence that ViT feature organization stabilizes over training, with much of the observed migration occurring in the early stages.

The code of this work is available on GitHub <sup>1</sup>.

## 2 Related Work

Feature-level interpretation in vision models. A large body of work studies what internal representations ofvision models encode. Early visualization methods inspected highly activating image patches [45], while Network Dissection [3] quantified the alignment be tween hidden units and human-interpretable features, also known as concepts. Other concept-based methods, such as TCAV [32] and concept bottleneck models [35], analyze user-defined concepts or enforce concept-level intermediate representations. In parallel, studies of CNNs and ViTs have shown that visual representations often become more abstract with depth, moving from colors, textures, and local patterns toward object- or class-level structure [3, 13, 16, 21, 22, 25, 33, 44, 50]. These works motivate feature-level analysis, but they usually study trained models or individual layers rather than how features evolve throughout training.

Representation similarity and training dynamics. Another line of work compares internal representations across layers, checkpoints, datasets, or architectures using measures such as SVCCA [55] and CKA [38]. These methods are useful for measuring global similarity between activation spaces [7, 27, 38] and have revealed important diferences between CNNs and ViTs, including the role of residual connections and early global information flow in ViTs [16, 56]. However, global representational similarity does not identify which features are shared [36, 37, 62]. Two representations may be similar while containing diferent features, and two representations may support related features even if their activation spaces are not globally aligned. Our work therefore complements representationlevel analysis with feature-level tracking.

Sparse feature discovery. Sparse autoencoders and related dictionary learning methods decompose hidden activations into sparse latent variables [68]. In vision models, recent works [44, 50, 52, 58] have shown that such latents can correspond to meaningful visual patterns in CLIP [54], DINOv2 [51], and other ViT-based encoders. These methods are attractive because they do not require a predefined vocabulary of human concepts. At the same time, SAE latents should not be treated as guaranteed atomic or human-interpretable concepts [29, 39, 53]: they may be unstable across runs, split across related features, or represent mixtures [6, 17, 47]. We therefore treat SAE latents as candidate sparse features and evaluate their continuity through their activation profiles. Overall, studies establish SAEs as a promising tool for vision interpretability, but principally our epoch-by-layer view is not limited to the method used for feature discovery; any other method could be used as well (e.g., NMF [42], PCA, or K-SVD [61]).

Feature evolution and migration. The closest work to ours studies how representations or sparse features change during training. Prior studies have analyzed training dynamics in CNNs and ViTs [16, 33, 57], and recent SAE-based work has tracked feature dynamics in language models [64]. However, existing work does not provide an epoch-by-layer account of feature evolution in ViTs. In this paper, we study whether visual features emerge, persist, disappear, or migrate between layers during training. This gives a feature-level view of ViT learning that is not captured by either static interpretability methods or scalar representation-similarity measures. Since state-of-the-art visual backbones such as DINOv2 and CLIP-style encoders are built on the ViT architecture, understanding their feature-level training dynamics is therefore crucial.

Prior work has interpreted features, compared representations, and studied training dynamics, but to the best of our knowledge, has not tracked feature-level migration across both ViT depth and training time.

## 3 Methodology

In this section, we introduce our method for computing similarities between features when the features are not predefined with labels specifying for each image whether it has this property or not. When features are predefined, one can simply use linear probing and track the features in our proposed epoch-by-layer map.

First, we show how to use SAEs to extract candidate sparse features from CLS-token representations, and second, explain how to compare their activation profiles across epoch–layer pairs. This allows us to study feature-level dynamics that are not directly visible from representation-level similarity measures.

Let $f _ { \theta ^ { ( t ) } }$ denote a ViT backbone at training checkpoint � and let $h ^ { ( t , \ell ) } ( a ) \in \mathbb { R } ^ { d _ { \ell } }$ denote its CLS-token representation at layer ℓ for input �. Given an image dataset $\mathcal { D } = \{ a _ { i } \} _ { i = 1 } ^ { n }$ , we form a representation matrix

$$
X ^ { ( t , \ell ) } = \left[ \begin{array} { c } { x _ { 1 } = h ^ { ( t , \ell ) } ( a _ { 1 } ) ^ { \top } } \\ { \vdots } \\ { \vdots } \\ { x _ { n } = h ^ { ( t , \ell ) } ( a _ { n } ) ^ { \top } } \end{array} \right] \in \mathbb { R } ^ { n \times d _ { \ell } } .\tag{1}
$$

This representation extraction phase is illustrated in part A of Figure 2, showing that representations are extracted for each epoch-layer pair.

Now, we go over our SAE setup and explain how we get SAE features. This is illustrated in part B of Figure 2. Given activations $x \in X ^ { ( t , \ell ) }$ from a fixed layer and checkpoint, a sparse autoencoder (SAE) maps � to a sparse latent code $z \in \mathbb { R } ^ { m }$ and reconstructs it as

$$
z = \mathrm { R e L U } ( W _ { e n c } ( x - b _ { d e c } ) + b _ { e n c } ) ,\tag{2}
$$

$$
\hat { x } = W _ { d e c } z + b _ { d e c } ,\tag{3}
$$

where $W _ { e n c } \in \mathbb { R } ^ { m \times d }$ and $W _ { d e c } \in \mathbb { R } ^ { d \times m }$ are the encoder and decoder weights; $b _ { e n c }$ and $b _ { d e c }$ are the encoder and decoder biases, respectively [4, 19]. We use the high-scoring [30] BatchTopK SAE objective, which uses an explicit batch-level sparsity constraint [5, 39]

$$
\mathcal { L } ( X ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } { \| x _ { i } - ( W _ { d e c } \mathrm { B a t c h T o p K } ( z _ { i } ) + b _ { d e c } ) \| _ { 2 } ^ { 2 } } + \lambda \frac { 1 } { n } \sum _ { i = 1 } ^ { n } { \| z _ { i } \| _ { 2 } ^ { 2 } } .
$$

In our setting, each latent dimension $z _ { \mathrm { : } , k }$ is interpreted as a candidate feature. For a fixed (ℓ, �), the SAE thus defines a feature set

$$
Z ^ { ( \ell , t ) } = \left\{ z _ { : , k } ^ { ( \ell , t ) } \right\} _ { k = 1 } ^ { m } ,\tag{4}
$$

where $z _ { : , k } ^ { ( \ell , t ) } \in \mathbb { R } ^ { n }$ is the activation profile of feature � over the dataset. This allows us to distinguish representation similarity from feature similarity: the former compares whole representation spaces, whereas the latter compares matched latent features extracted from those spaces.

We want to stress that not every SAE feature is human-interpretable; they can merely be a “feature for the algorithm” as in the work of Fel et al. [16]: “Thus, in our work, a feature refers to a random scalar value extracted by a dictionary learning method on the activations.”

## 3.1 Similarity Metrics for SAE Feature Comparison

Here, we show how we measure the similarity of the extracted features across epoch-layer pairs for our time-depth analysis. Part C of Figure 2 illustrates this feature-similarity calculation. Representationlevel metrics compare the geometry of full activation spaces, but we compare individual SAE features through their activation profiles to quantify the feature-level similarity.

For brevity, let

$$
\mathcal { A } = ( \ell , t ) , \qquad \mathcal { B } = ( \ell ^ { \prime } , t ^ { \prime } )\tag{5}
$$

denote two checkpoint-layer pairs, with corresponding backbone representations $\bar { X ^ { \mathcal { A } } }$ and $\dot { X } ^ { \mathcal { B } }$ , and SAE feature sets $Z ^ { \mathcal { A } } = \{ z _ { : , k } ^ { \mathcal { A } } \} _ { k = 1 } ^ { m _ { \mathcal { A } } }$ and $Z ^ { \mathcal { B } } = \{ z _ { : , j } ^ { \mathcal { B } } \} _ { j = 1 } ^ { m _ { \mathcal { B } } }$

Transferred-feature similarity: one SAE applied across checkpoints and layers (Fixed SAE). The following is one part of our core methodology. Our first feature-level setting fixes a reference SAE trained at A and applies it to any other checkpoint-layer pair B. Let $z _ { : , k } ^ { \mathcal { B }  \mathcal { A } }$ denote the activation profile of source feature � from the SAE of A when evaluated on $X ^ { \hat { \mathcal { B } } }$ on the dataset D. This setting measures feature transportability: whether the same linear combination applied at a diferent epoch-layer pair results in a similar activation profile, i.e., instances that get a high (or low) activation at the reference epoch-layer pair, also get a high (respectively low) activation when the linear combination is applied at that other epoch-layer pair.

We compare source and transferred activations using Spearman rank correlation,

$$
\begin{array} { r } { s _ { \mathrm { s p r } } ( \boldsymbol { k } ; \mathcal { R } \right. \mathcal { B } ) = \rho _ { s } \left( \boldsymbol { z } _ { : , \boldsymbol { k } } ^ { \mathcal { R } } , \boldsymbol { z } _ { : , \boldsymbol { k } } ^ { \mathcal { B } \left. \mathcal { A } \right)}  , } \end{array}\tag{6}
$$

where $\rho _ { s } ( \cdot , \cdot )$ denotes Spearman’s rank correlation.

Cross-SAE feature similarity: one SAE per checkpoint and layer (Independent SAE). Our second feature-level setting trains an independent SAE for each checkpoint-layer pair and compares the resulting features directly. For source feature � in A and target feature � in B, we define a full cross-SAE similarity matrix using the same Spearman correlation similarity:

$$
\begin{array} { r } { S _ { k j , \mathrm { s p r } } ^ { \mathcal { R } , \mathcal { B } } = \rho _ { s } \left( z _ { : , k } ^ { \mathcal { R } } , z _ { : , j } ^ { \mathcal { B } } \right) , } \end{array}\tag{7}
$$

From this matrix, we compute row-wise best-match scores to determine the highest correlation between diferent epoch-layer pairs’ activation profiles, possibly indicating their presence.

$$
s _ { \mathrm { s p r } } ^ { \operatorname* { m a x } } ( k ; \mathcal { R } \to \mathcal { B } ) = \operatorname* { m a x } _ { j } S _ { k j , \mathrm { s p r } } ^ { \mathcal { R } , \mathcal { B } } ,\tag{8}
$$

Final similarity score as a combination of the two. Since both previously defined similarities have pros and cons, we reason as follows. Spearman correlation is robust to monotone rescalings of activation magnitude. So, fixed SAE measure directly tests whether a feature learned at A remains stable and readable in another representation. But this score does not directly measure feature identity in the target representation. Rather, it tests whether the source dictionary remains a valid readout of the target space. A low score may therefore indicate true disappearance of the feature, but it may also arise if the same feature is present in a shifted, rotated, split, or otherwise reparameterized form. This can be mitigated with the help of our proposed independent SAE measure. Independent SAE best-match score measures whether a feature from A is rediscovered by the independently trained SAE at B. This is closer to feature identity than fixed-SAE measure, since it does not assume that a single dictionary remains valid across all checkpoints and layers. Since the independent SAE score depends on both SAEs successfully learning the corresponding feature, combining it with the fixed SAE helps to achieve the desired balance between robustness to feature reparametrizations and excessive reliance on feature rediscovery. Therefore, the final method uses the maximum of the two:

$$
r _ { k } ^ { \mathcal { R }  \mathcal { B } } = \operatorname* { m a x } ( s _ { \mathrm { s p r } } ^ { \mathrm { m a x } } ( k ; \mathcal { R }  \mathcal { B } ) , s _ { \mathrm { s p r } } ( k ; \mathcal { R }  \mathcal { B } ) )\tag{9}
$$

![](images/62ef64f1559c6757f85b92fba088ed5aae75bf62e82707ffea6a33ea9059dacd.jpg)  
Figure 2: Overview of the methodology: (A) extraction of image representations from every ViT checkpoint–layer pair; (B) SAE training at each pair; and (C) computation of feature-similarity scores for one example.

After calculating the similarity score from A to each epoch-layer checkpoint, we can visualize the similarity as a heatmap and observe feature correlation across time and depth, providing insight into its evolution. Repeating this for all features � , we can describe the evolution of features from checkpoint A.

## 4 Experiments

Our depth-time view, combined with SAE-based feature evolution trajectories, allows us to investigate feature evolution with respect to where they are from, i.e., the reference epoch-layer point. We use our methodology to investigate the following questions:

(1) Do features migrate from deeper to earlier layers, or from earlier to deeper layers, or in both directions?

(2) When do features stabilize across ViT layers during training?

(3) How persistent are the features learned early in the train ing?

(4) How layer-localized is the feature evolution during training?

(5) How class-specific are the features throughout the training?

(6) Clustering SAE features based on migration, do they form interpretable clusters?

(7) Do the answers to the previous questions depend on the dataset size?

Now, we define the following metrics to answer the questions.

## 4.1 Feature Evolution Metrics

Let $S _ { \ell , t } ^ { ( c ) } \in \mathbb { R }$ denote the zero-capped positive similarity score of feature � at layer $\ell \in \{ 0 , \ldots , L - 1 \}$ and epoch $t \in \{ 0 , . . . , T \}$ . By zero-capped we mean that every negative score gets replaced by zero. For a reference point $( \ell _ { \mathrm { r e f } } , t _ { \mathrm { r e f } } )$ , we remove the self-similarity (being always 1) and define a feature-specific threshold as half of the second-highest similarity score (a feature-relative threshold helps to treat features with diferent similarity scales accordingly).

$$
\tau _ { c } = \frac { 1 } { 2 } \operatorname* { m a x } _ { ( \ell , t ) \neq ( \ell _ { \mathrm { r e f } } , t _ { \mathrm { r e f } } ) } S _ { \ell , t } ^ { ( c ) } .\tag{10}
$$

This thresholding suppresses weak background similarities and isolates the dominant similarity trajectory across time and depth to be able to capture its dynamics. We next define the above-threshold mask and filtered similarity as follows:

$$
K _ { \ell , t } ^ { ( c ) } = 1 \left[ S _ { \ell , t } ^ { ( c ) } > \tau _ { c } \right] , \qquad \widetilde w _ { \ell , t } ^ { ( c ) } = K _ { \ell , t } ^ { ( c ) } S _ { \ell , t } ^ { ( c ) } .\tag{11}
$$

Layer distribution, center, and spread. For each epoch, the surviving layer similarities are converted into a masked layer distribution

$$
P _ { \ell , t } ^ { ( c ) } = \frac { K _ { \ell , t } ^ { \left( c \right) } \exp \left( S _ { \ell , t } ^ { ( c ) } \right) } { \sum _ { j = 0 } ^ { L - 1 } K _ { j , t } ^ { \left( c \right) } \exp \left( S _ { j , t } ^ { ( c ) } \right) } ,\tag{12}
$$

with $P _ { \ell , t } ^ { ( c ) } = 0$ if no layer survives. Let �<sub>ℓ</sub> denote the numerical layer coordinate (in our case $r _ { \ell } \in \{ 0 , \ldots , 1 1 \} )$ . The layer center of feature � at epoch � and the layer spread (i.e., layer mean and standard

Feature Evolution and Migration during Vision Transformer Training

deviation estimates)

$$
m _ { t } ^ { ( c ) } = \sum _ { \ell = 0 } ^ { L - 1 } r _ { \ell } P _ { \ell , t } ^ { ( c ) } , \qquad \sigma _ { t } ^ { ( c ) } = \sqrt { \sum _ { \ell = 0 } ^ { L - 1 } \left( r _ { \ell } - m _ { t } ^ { ( c ) } \right) ^ { 2 } P _ { \ell , t } ^ { ( c ) } } .\tag{13}
$$

Low $\sigma _ { t } ^ { ( c ) }$ indicates layer-localized similarity, whereas high $\sigma _ { t } ^ { ( c ) }$ indicates spread across layers.

Emergence and disappearance. The epoch-level activity signal is computed from the filtered similarity:

$$
\alpha _ { t } ^ { ( c ) } = \frac { 1 } { L } \sum _ { \ell = 0 } ^ { L - 1 } \widetilde { w } _ { \ell , t } ^ { ( c ) } .\tag{14}
$$

The emergence epoch is the first epoch at which the feature is above threshold for $k _ { \mathrm { o n } }$ consecutive epochs:

$$
t _ { \mathrm { e m e r g e } } ^ { ( c ) } = \operatorname* { m i n } \left\{ t : \alpha _ { u } ^ { ( c ) } > 0 \mathrm { f o r } \mathrm { a l l } u \in \left\{ t , . . . , t + k _ { \mathrm { o n } } - 1 \right\} \right\} ,
$$

and if $u > T$ then we define $\alpha _ { u } ^ { ( c ) } = 0$ . The disappearance epoch is the first sustained inactive run after emergence, lasting �<sub>of</sub> epochs, after which no later sustained re-emergence occurs:

$$
t _ { \mathrm { d i s a p p e a r } } ^ { ( c ) } = \operatorname* { m i n } \left\{ t > t _ { \mathrm { e m e r g e } } ^ { ( c ) } : \alpha _ { u } ^ { ( c ) } = 0 \forall u \in \left\{ t , . . . , t + k _ { \mathrm { o f f } } - 1 \right\} \right\} ,\tag{15}
$$

where $\mathrm { i f } \ u > T$ then we define $\alpha _ { u } ^ { ( c ) } = 0$

Mean layer localization. Layer localization is measured by the normalized entropy of the layer distribution:

$$
H _ { t } ^ { ( c ) } = - \frac { 1 } { \log L } \sum _ { \ell = 0 } ^ { L - 1 } P _ { \ell , t } ^ { ( c ) } \log \left( P _ { \ell , t } ^ { ( c ) } + \varepsilon \right) , \qquad \lambda _ { t } ^ { ( c ) } = 1 - H _ { t } ^ { ( c ) } .\tag{16}
$$

Thus, $\lambda _ { t } ^ { ( c ) } \approx 1$ indicates a sharply localized layer distribution, while $\lambda _ { t } ^ { ( c ) } \approx 0$ indicates a spread distribution. The mean layer localization over valid active epochs is defined as:

$$
\overline { { \lambda } } ^ { ( c ) } = \frac { \sum _ { t \in \mathcal { V } ^ { ( c ) } } \omega _ { t } \alpha _ { t } ^ { ( c ) } \lambda _ { t } ^ { ( c ) } } { \sum _ { t \in \mathcal { V } ^ { ( c ) } } \omega _ { t } \alpha _ { t } ^ { ( c ) } } .\tag{17}
$$

(An example of a feature with low localization and high spread is given in Figure 1, part c).

Stability. To measure whether a feature stays near a stable layer, we first compute the activity- and localization-weighted layer occupancy. In our experiments we are not always using uniformly spaced epochs (for more details, look at the header of Section 4.3), so to account for non-uniform checkpoint spacing, we assign each sampled checkpoint a midpoint-based time weight equal to the interval it represents; interior weights extend to the midpoints between adjacent checkpoints, while endpoint boundaries are extrapolated by half the adjacent checkpoint gap; let $\omega _ { t }$ be that weight for epoch �

$$
Q _ { \ell } ^ { ( c ) } = \frac { \sum _ { t \in \mathcal { V } ^ { ( c ) } } \omega _ { t } \alpha _ { t } ^ { ( c ) } \lambda _ { t } ^ { ( c ) } P _ { \ell , t } ^ { ( c ) } } { \sum _ { t \in \mathcal { V } ^ { ( c ) } } \omega _ { t } \alpha _ { t } ^ { ( c ) } \lambda _ { t } ^ { ( c ) } } ,\tag{18}
$$

where $\mathcal { \boldsymbol { V } } ^ { ( c ) }$ denotes valid active epochs, excluding epochs whose layer spread exceeds the chosen maximum-width threshold $\sigma _ { m a x } .$ Intuitively, it reads as how much of the feature’s active lifetime is

spent near layer ℓ. The anchor layer is $\ell _ { \star } ^ { ( c ) } = \arg \operatorname* { m a x } _ { \ell } Q _ { \ell } ^ { ( c ) }$ and for a radius $\delta ,$ the stability score is

$$
\mathrm { S t a b i l i t y } _ { \delta } ( c ) = \sum _ { \ell : | r _ { \ell } - r _ { \ell _ { \star } ^ { ( c ) } } | \le \delta } Q _ { \ell } ^ { ( c ) } .\tag{19}
$$

High stability indicates that the feature remains close to one layer during its active lifetime. (An example of a stable feature with high mean layer localization is in Figure 1, part b).

Signed active drift. Let $\mathcal { V } _ { \mathrm { s t a r t } } ^ { ( c ) }$ and $\mathcal { V } _ { \mathrm { e n d } } ^ { ( c ) }$ denote the first and last fractions ofvalid active epochs. The start and end layer distributions are $Q _ { \mathrm { s t a r t } } ^ { ( c ) }$ and $\boldsymbol { Q } _ { \mathrm { e n d } } ^ { ( c ) }$ . The signed mean active drift is

$$
D _ { \mathrm { m e a n } } ^ { ( c ) } = \sum _ { \ell = 0 } ^ { L - 1 } r _ { \ell } Q _ { \mathrm { e n d } , \ell } ^ { ( c ) } - \sum _ { \ell = 0 } ^ { L - 1 } r _ { \ell } Q _ { \mathrm { s t a r t } , \ell } ^ { ( c ) } .\tag{20}
$$

Positive values indicate movement toward deeper layers, while negative values indicate movement toward earlier layers. (An example of feature migration towards earlier layer is in Figure 1, part a).

Feature classes. In our experiments, a feature is classified as spread if mean layer localization $\overline { { \lambda } } ^ { \left( c \right) }$ is smaller than 0.5. We classify a feature as migratory only when it’s not spread and its estimated displacement exceeds one complete layer, meaning a feature is classified as deeper-migratory if $D _ { \mathrm { m e a n } } ^ { ( c ) } > 1$ and earlier-migratory if $D _ { \mathrm { m e a n } } ^ { ( c ) } < - 1 . \ell$ A feature is classified as stable if it is not migratory nor spread, $| D _ { \mathrm { m e a n } } ^ { ( c ) } | < 0 . 5$ and Stabilit $\mathrm { y } _ { 1 } ( c ) \geq 0 . 7$ . The former condition requires the net displacement of a stable feature to remain below half a layer. This is combined with Stability $( c ) \geq 0 . 7$ , requiring at least 70% of the activity- and localization-weighted layer occupancy to lie within one layer of the feature’s anchor layer. Thus, the stability criterion captures both low net displacement and sustained concentration around a common layer. We deliberately leave a margin between the stable and migratory regimes: shifts below half a layer are considered negligible, whereas migration requires a displacement exceeding one full layer. Intermediate displacements are not assigned to either category. These cutofs are not intended as theoretically unique decision boundaries but provide conservative and interpretable operational categories in units of layers and normalized occupancy.

Feature validation. To answer the question of whether SAE features cluster based on their migratory behavior, we need to somehow label the features. We follow [52] and interpret a set of SAE features manually. Our manual efort revealed that SAE features can usually be put under one of the categories: class-specific, superclassspecific, inter-class, not human-understandable (at least not immediately). We automate the process partly as follows. We classify each feature by examining the normalized ImageNet class distribution of the images in the test set where that feature is active. If one class is clearly dominant, meaning that the normalized activation for the second-highest class is less than half of that of the highest class, the feature is labeled a class-specific feature (see the left of Figure 3). If there is no single dominant class, we check whether a small group of 2 to 10 top-ranked classes has relatively similar normalized activations and is followed by a large drop, defined as the next class having less than half the normalized activation of the lowest-ranked class in that group. This indicates that the feature may be shared by a small semantic group rather than a single class. To confirm this, we utilise the semantic hierarchy of WordNet [18]: the top classes must share an immediate hypernym. If they do, the feature is labeled a superclass feature (see the middle of Figure 3). If neither condition is satisfied, meaning a feature is not clearly associated with one class or with a tightly related group of classes, we return to our manual search and check if any of the remaining features indicate a clear pattern, i.e., whether it is an inter-class high-level feature (see the right of Figure 3).

## 4.2 Setup

ViT training. We use two ImageNet-based datasets: ImageNet-1k [11] and its balanced superclass dataset Mixed 10 adapted from Engstrom et al. [15].

We use the vanilla architecture of the ViT [14] to assess its feature evolution, as it is a primary building block of modern vision models, e.g., DINOv2 and CLIP. More precisely, we exemplify our framework on ViT-Tiny [60], but use 12 heads as suggested by [63]. Wang et al. [63] also showed that the representations of ViT-Tiny and ViT-Base models trained from scratch look similar using CKA, allowing us to generalize across ViTs with our analysis on the computationally eficient ViT-Tiny.

We trained a ViT-Tiny on ImageNet-1k following the setup from [63], using 12 transformer blocks, patch size $^ { 1 6 , }$ embedding dimension 192, and 12 attention heads. Training was performed for 300 epochs with AdamW $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ , weight decay 0.05) and a cosine learning-rate schedule [46] with 5 warmup epochs. The base learning rate was set to $1 0 ^ { - 3 }$ and scaled linearly with the global batch size as $\mathrm { l r } = 1 0 ^ { - 3 } \cdot \frac { B } { 2 5 6 } \ [ 2 ]$ . We used batch size 2048 and mixup [67] with $\alpha = 0 . 2 .$ . Input images were resized to 224 × 224, and training augmentations followed the paper-inspired recipe with random resized crops, horizontal flipping, color jitter (0.3), and RandAugment [9]. We additionally used layer-wise learning-rate decay with coeficient 0.75, mixed-precision training, and standard ImageNet normalization. To speed up the dataloader, we used FFCV [40]. For the ImageNet-1k superclass dataset Mixed 10, we used the same training recipe but with a smaller batch size 512. The achieved accuracies were for ImageNet-1k 72.55% and for Mixed 10 86.37%.

SAE training. Recent work suggests that the distinction between CLS and patch tokens is not clean: standard ViTs may already treat them diferently through normalization and other implicit mechanisms, despite nominally shared computation [48]. Darcet et al. [10] also suggest that global information can leak into patch tokens, complicating the interpretation of patch-stream features as purely local. For clarity of interpretation, we limit ourselves in this paper to the CLS tokens.

Therefore, we train SAEs on top of CLS tokens. Gao et al. [19] suggests training SAE until convergence, but in our setup, such computation was not feasible, and we fixed the number of epochs to 150 (at this point, the loss started to plateau during hyperparameter search). For ImageNet-1k, we limited the training set size via random sample to be $N _ { t r a i n } = 1 0 0 0 0 0$ uniformly distributed among 1000 classes, and for testing we used the original ImageNet-1k val idation set as a test split $( N _ { t e s t } = 5 0 0 0 0 )$ ). For Mixed 10 dataset, we used the entire training split $( N _ { t r a i n } = 7 7 0 6 0 )$ and test split $( N _ { t e s t } = 2 7 0 0 )$ . We used the same SAE settings for all the checkpoints and both datasets: batch size 4096, sparsity expansion coeficient 5, BatchTopK with $K = 3 2 ,$ , L2 regularization 0.05.

## 4.3 Results

We report the results for both datasets, if they are substantially diferent, otherwise, we show only the ImageNet-1k results, since its larger test set is less noisy. The results for the migration metrics defined in Section 4.1 are presented with a high-frequency window of [reference epoch−50; reference epoch+50] around the reference epoch and low-frequency elsewhere (i.e., every 10th epoch). For the reference epochs smaller than 50 or greater than 250, the window was set to [0, 100] and [200, 300], respectively. Reference epochs were chosen mainly from the beginning of the training, as it is when the model is learning more rapidly, and more sparsely after that, resulting in reference epochs [0, 100] ∪ {110, 120, . . . , 280} ∪ [281, 300).

Do features migrate from deeper to earlier layers, orfrom earlier to deeper layers, or in both directions? We conducted several experiments to answer this question. Firstly, we used Eq. 20 with the given thresholds from Section 4.1 and calculated the percentage of deeper migrating features among all the reference points; the resulting heatmap is depicted in Figure 4. We can see that deeper migration is relatively rare, peaking at 4%. The few visible peaks occur mostly in early checkpoints, suggesting that movement toward deeper layers is not a dominant mode of feature reorganization.

Analogously, the resulting heatmap for the percentage of features migrating towards earlier layers is given in Figure 5, which shows that, for both datasets, it is most prevalent primarily in the first part of the training. The main diference is in the layers afected: early layers in the case of ImageNet-1k and later layers in the case of Mixed 10. For clarity, features from layer 0 can still migrate towards earlier layers, as the definition 20 does not require that feature drift necessarily begin at the reference layer, but rather where it is most active, which can, in fact, be later layers.

Neither of the previous plots says much about the gravity of migration; e.g. whether migration is across 1 or even 5 layers. We depict the average of signed mean drift in Figure 6. We see that deeper layers and the later training phase are, on average, less affected by migration. On the contrary, earlier layers and training show higher migration towards earlier layers, reaching approximately −2.5 layers in the regions with strongest migration.

In Figure 9, we depict the layer center-of-mass (COM) trajectory averaged over all the features per reference point. In the figure, notably, after the first epoch, we can see migration predominantly towards deeper layers from layer 0 (but rarely exceeding 1 layer) and towards earlier layers from layer 11. The features from the last layer in epoch 300 are stable and have little deviation around the COM, whereas the first layer is less stable and can experience migration towards earlier layers. Epochs 100 and 200 show a peak at the reference location – this may be due to the localization of similarity at the reference layer and difusion or migration towards earlier layers thereafter. The peak is smoother for the first layer at epoch 200, indicating feature stabilization. The last layer at the same epochs shows more stability, and COM captures the reference layer better. Curiously, the average layer width for features referenced from the first layer increases over training, indicating that their similarity mass becomes less layer-localized.

![](images/57c60c71c3ca6506c15a329f93ae28172ba5445722b0a6beefc4c5dec3774584.jpg)

Figure 3: Feature-types based on ImageNet classes. Human-understandable SAE features are observed under 3 main categories: class-specific, superclass-specific and inter-class (see maximally activating images [66] for each feature in Appendix).  
![](images/3df4983b323f230652bcffb5f91d04db65a1297cfe57baa0b3d2e134d20f1ee4.jpg)  
Figure 4: Percentage of features migrating towards deeper layers.

When do features stabilize across ViT layers during training? This question is partially answered in the previous paragraph. We used Equation 19 with the given thresholds from Section 4.1 and counted the percentage of stable features from all the reference points; the resulting heatmap is depicted in Figure 7. Naturally, it correlates visibly with Figure 6 of average migration magnitude but in this case we can also see how many of the features are considered stable. Firstly, the deeper layers stabilize, and throughout training, earlier layers stabilize as well, but not to the same extent as the deeper layers.

How persistent are features learned early in the training? In Figure 8, the mean active lifetime of features in epochs (the mean number of epochs between emergence and disappearance) is depicted per epoch-layer pair. Interestingly, some earlier layers and epochs exhibit long persistence throughout the training, even though they are not stable, as shown in Figure 7. Surprisingly, deeper and intermediate layers in early training tend to have a short lifetime. The average active lifetime for late-training-phase features seems stable.

How layer-localized is the feature evolution during training? To measure the layer localization, we used Equation 16 with the given thresholds from Section 4.1 and counted the percentage of features that are widely spread across layers from all the reference points;

![](images/2b4f117efe75d514dacbb235d49479e41461b3dbbbed4194816ca1cdb15c41d2.jpg)

![](images/787fb3575e106f652c46c3b5e77217d1953326f373faa06742ed6a443e28ed2d.jpg)  
Figure 5: Percentage of features migrating towards earlier layers.

the resulting heatmap is depicted in Figure 10. Interestingly, widely spread features occur throughout training, mainly at the beginning and in the earlier layers. Therefore, combined with the percentage of stable features, there is no more significant migration left at the end of training.

Clustering SAE features based on migration: do they form interpretable clusters? To further investigate the feature migration patterns, we provide a layer center-of-mass (COM) analysis in Figure 11, which shows the change in COM for features from diferent reference points. As can be seen from the figure, a vast majority of the features are migrating towards earlier layers, suggesting that they start emerging in later layers in early training, then move towards earlier layers, e.g., the behavior we see in Figure 1 part (a). Moreover, layer COMs for features referenced to very early training checkpoints, i.e., epoch 1, mostly cluster together and move roughly from layers 3-5 to layers 1-3. Other reference features from later training, such as at epochs 100, 200 or 300, are mostly active after the 4th layer in the early training and move 1-3 layers up later towards the end of the training. The presented features are also labeled automatically as mentioned in Section 4.1; however, we do not observe any notable correlation between their types, e.g., class-specific, superclass-specific, or others, and their migration behavior.

![](images/9e11f9ed75cdca67fd2274331be12cb682adc22d472555a174b58b2fff6e04e1.jpg)  
Figure 6: Mean signed active drift. For each reference epoch– layer pair, feature-level quantities are averaged across features. Average signed displacement of the layer center, in layer-index units; negative values indicate movement toward earlier layers.

![](images/b4dec51a5e89f84875fb795a492c0ac748d3888dd4099042cbfb62bd1981a5a5.jpg)  
Figure 7: Percentage of stable features.

![](images/ff320e4583fffba7c47d5b5c32d431ab062c7cadd990f24f9971822ad4c2b1f8.jpg)  
Figure 8: Mean active lifetime of features, given in number of epochs.

<table><tr><td>Reference epoch</td><td>class</td><td>superclass</td><td>other</td><td>total (active)</td></tr><tr><td>1</td><td>70</td><td>0</td><td>755</td><td>825</td></tr><tr><td>100</td><td>259</td><td>22</td><td>678</td><td>959</td></tr><tr><td>200</td><td>401</td><td>35</td><td>872</td><td>1308</td></tr><tr><td>300</td><td>461</td><td>42</td><td>903</td><td>1406</td></tr></table>

Table 1: Feature-type counts by reference epochs presented in Figure 11.

How class-specific are the features throughout the training? Furthermore, we measured the label entropy [44] of our features based on the image labels for which the feature is activated, resulting in Figure 12. We find that there is a stark diference between the datasets, ImageNet-1k having high label entropy in higher layers whereas Mixed 10 having high entropy, on the contrary, in the lower layers consistently.

Additional results and experiment details are presented in the Appendix A.

## 5 Discussion and Conclusion

Our observations of long feature lifetime in the early layers and prevalent migration towards earlier layers are consistent with prior claims by Yun et al. [65] and Fel et al. [16]. More precisely, our finding that early layer features have a long active lifetime relates to Yun et al. [65], who claimed for language Transformers (BERT [12]) that, due to residual connection, we should expect “if a vital feature is learned in the beginning layers, it won’t disappear in later stages. Instead, it will be carried all the way to the end with gradually decayed weight since many more features would join along the way.” Furthermore, in Figure 9, the average layer width increased over training, which might be due to the residual connections gradually having more efect on deeper layers. A similar observation about the role of residual connections is made by Raghu et al. [56], who notice that ViTs exhibit more uniform representations across depth, make use of global information earlier, and are more strongly shaped by residual connections than CNNs. In Figure 11, we did not observe any clustering of feature types by migration, but we do see a progression across feature types over training time: the later the epoch, the more class- and superclass-specific features there are (see Table 1). This relates to the finding by [3] in CNNs that low-level features are learned earlier in training. Sharon and Dar [57] analyzed how representations change through training in CNNs and ViTs, showing that layers can evolve diferently across regimes and that deeper layers may change more substantially during prolonged fitting – as prolonged fitting was not part of our routine, we can neither confirm nor refute their finding, but on the contrary, we did see more instability and migration in earlier layers rather than deeper ones.

Our finding about the label entropy in the previous section, depicted in Figure 12 can possibly be explained by the expansion coeficient of the SAE. Since the image representations have dimensionality 192 and are mapped into a sparse layer via SAE of dimensionality 5 × 192 = 960, then SAE cannot aford to recover only single-class-specific features for ImageNet-1k. The opposite seems to be the case for the Mixed 10 dataset. Therefore, this might have implications for the selection of the SAE expansion coeficient when class-specific features are desirable.

ImageNet-1k : Average active-layer trajectory (COM, active\_mode=threshold)  
![](images/b45ba5d104e749d6b5794a0b55149f978697eb90d49c602219c4473de6b09f34.jpg)  
Figure 9: Layer center-of-mass (COM) ± standard deviation trajectories averaged over all the features of selected reference checkpoints (0,100,200 and 300). The high-frequency epochs are depicted as a solid line around the reference checkpoints (marked as x) and the low-frequency epochs as dots. On the figure below, the average layer width ± standard deviation is depicted. The averaging is carried out only on active epochs per feature and therefore color fading signifies the lack of active features in that area.

![](images/2a69572c025998f967265f8af88acd1f15bd048d1024befd4af21ad9decaea3c.jpg)  
Figure 10: Percentage of features that are widely spread across layers.

Our work can be described as feature-level readouts of weightspace training dynamics. A substantial line of work studies neural network training as a trajectory through parameter space, using tools such as interpolation from initialization to solution [23], losslandscape visualization [43], mode connectivity [20], stochastic weight averaging [28], and curvature-based analyses of optimization dynamics [8]. Our method instead studies an interpretable feature-level projection of the same training trajectory and, therefore, is complementary to this weight-space perspective. Rather than measuring distances, barriers, curvature, or connectivity between checkpoints in parameter space, we ask how the model’s movement through parameter space is reflected in the development ofinterpretable internal features. One can view our method as defining a feature-level observable on the model’s parameter trajectory: for each checkpoint and layer, we train a sparse autoencoder and compare feature activation profiles across the resulting layer-byepoch grid. Each resulting heatmap is a feature-level readout of the training trajectory: it does not track movement in parameter space directly, but instead tracks how that movement manifests as the emergence, persistence, disappearance, or displacement of features.

This positions SAE-based feature dynamics as a feature-space counterpart to weight dynamics. Weight-space analyses ask where optimization moves in the loss landscape and what geometric structure the trajectory encounters; our analysis asks what interpretable representational structure is built, preserved, or reorganized along that same trajectory. The two views therefore operate at diferent levels of description.

In conclusion, we introduced an epoch-by-layer framework for tracking visual-feature migration in ViTs during training using SAE-based features. Using our framework, we found that features migrate mainly towards earlier layers and migration is concentrated earlier in training. In addition, we found that deeper-layer features stabilize earlier and more strongly, and that average active lifetime is longer for features in early layers. Using feature-type automation, we noticed that feature type does change over training time: early checkpoints have fewer class-specific features. The broad migration patterns are qualitatively similar across ImageNet-1k and Mixed 10, although this comparison should not be interpreted as isolating dataset scale alone, since the datasets also difer in label granularity and test-set size.

![](images/f2addc062f1c19970921fc61fa62315a3d4700f6b76cdba531f6ad74a3e2de36.jpg)  
Figure 11: Feature migration analysis by layer center-of-mass (COM). Most of the features migrate towards earlier layers.

## 6 Limitations and Future Work

This work opens a new avenue of tracking feature formation and migration by defining hypothesis-relevant metrics. It remains for future investigation to test out diferent similarity metrics for tracking feature evolution – Spearman-based feature alignment can be sensitive to ties for zero activations, so, alternatives such as Top-� overlap should be explored. Currently, our SAE approach does not guarantee finding a feature across other epoch-layer pairs; this remains future work to address problematic feature discovery.

Our methodology has several limitations that suggest directions for additional future work. First, since the pattern classification is to some extent subjective, then the definitions of evolution tracking metrics may need further grounding. Second, the analysis is restricted to linearly decodable features. This restriction is nontrivial, since Fel et al. [16] provide evidence that not all features need to be linearly decodable throughout the network. Third, we study only supervised training. Other training regimes remain unexplored, including self-supervised objectives such as ViT-Tiny MAE [24], which can be trained from scratch with a reasonable computational budget. In addition, future work could study how feature trajectories change during fine-tuning. Evaluating the proposed methodology across multiple seeds, additional datasets, architectures, normalization schemes, and optimizers would help determine whether the observed feature-evolution and migration patterns generalize beyond the present setting.

![](images/2c6dca39c0f8639b72a68708c4f64f39cf5aabcaa42b92987f62b0e02433af7e.jpg)

![](images/42d9e28a9b522af441cf3a9a2f3cf5af626640082ac309bfdf996c4dc3f8fac5.jpg)  
Figure 12: The diference between two datasets in label entropy. Low label entropy indicates label-selective features, whereas high label entropy indicates features that activate across many labels.

We note that conducting a similar analysis for human-interpretable features (e.g., using linear probes with per-image attributes) would further enrich understanding of feature evolution, but is beyond this work’s scope and remains a direction for future work.

A further open issue is whether feature evolution difers between the CLS and patch streams. Our methodology can be easily adapted for patch tokens. ViT analyses have shown that patch tokens retain spatial information through most layers, whereas the final layer behaves more like token mixing or learned pooling [22]. In [31], it has been shown that in multi-object images, the CLS token decodes the primary object well, but decoding of secondary objects is worse than with object-specific patch tokens. This implies the efect of the downstream task on the representation of CLS token. These examples make CLS-stream versus patch-stream feature evolution a substantive future research question for obtaining a more complete picture.

## Acknowledgments

This work was supported by the Estonian Research Council grant PRG1604, and by the Estonian Center of Excellence in Artificial Intelligence (EXAI), funded by the Estonian Ministry of Education and Research grant TK213.

## 7 GenAI Usage Disclosure

During the preparation of this manuscript, the authors utilized Generative AI technologies to assist with identifying relevant literature, improving the readability of the text, and developing code. After using these tools, the authors reviewed and edited the resulting content and take full responsibility for the presented manuscript.

## References

[1] Guillaume Alain and Yoshua Bengio. 2016. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644 (2016).

[2] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. 2021. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254 (2021).

[3] David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba. 2017. Network dissection: Quantifying interpretability of deep visual representations. In Proceedings of the IEEE conference on computer vision and pattern recognition. 6541–6549.

[4] Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, et al. 2023. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread 2, 5 (2023), 6.

[5] Bart Bussmann, Patrick Leask, and Neel Nanda. 2024. Batchtopk sparse autoen coders. arXiv preprint arXiv:2412.06410 (2024).

[6] David Chanin, James Wilken-Smith, Tomáš Dulka, Hardik Bhatnagar, Satvik Golechha, and Joseph Bloom. 2024. A is for absorption: Studying feature splitting and absorption in sparse autoencoders. arXiv preprint arXiv:2409.14507 (2024).

[7] Laure Ciernik, Lorenz Linhardt, Marco Morik, Jonas Dippel, Simon Kornblith, and Lukas Muttenthaler. 2024. Objective drives the consistency of representational similarity across datasets. arXiv preprint arXiv:2411.05561 (2024).

[8] Jeremy M Cohen, Simran Kaur, Yuanzhi Li, J Zico Kolter, and Ameet Talwalkar. 2021. Gradient descent on neural networks typically occurs at the edge of stability. arXiv preprint arXiv:2103.00065 (2021).

[9] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. 2020. Randaug ment: Practical automated data augmentation with a reduced search space. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition workshops. 702–703.

[10] Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. 2023. Vision transformers need registers. arXiv preprint arXiv:2309.16588 (2023).

[11] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. ImageNet: A large-scale hierarchical image database. In 2009IEEE Conference on Computer Vision and Pattern Recognition. 248–255. doi:10.1109/CVPR.2009.5206848

[12] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 conference ofthe North American chapter ofthe association for computational linguistics: human language technologies, volume 1 (long and short papers). 4171–4186.

[13] Teresa Dorszewski, Lenka Tětková, Robert Jenssen, Lars Kai Hansen, and Kristof fer Knutsen Wickstrøm. 2025. From colors to classes: Emergence of concepts in vision transformers. In World Conference on Explainable Artificial Intelligence. Springer, 28–47.

[14] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR (2021).

[15] Logan Engstrom, Andrew Ilyas, Shibani Santurkar, and Dimitris Tsipras. 2019. Robustness (Python Library). https://github.com/MadryLab/robustness

[16] Thomas Fel, Louis Bethune, Andrew K Lampinen, Thomas Serre, and Katherine Hermann. 2024. Understanding visual feature reliance through the lens of complexity. Advances in Neural Information Processing Systems 37 (2024), 69888– 69924.

[17] Thomas Fel, Ekdeep Singh Lubana, Jacob S Prince, Matthew Kowal, Victor Boutin, Isabel Papadimitriou, Binxu Wang, Martin Wattenberg, Demba Ba, and Talia Konkle. 2025. Archetypal sae: Adaptive and stable dictionary learning for concept extraction in large vision models. arXiv preprint arXiv:2502.12892 (2025).

[18] Christiane Fellbaum. 1998. WordNet: An electronic lexical database. MIT press.

[19] Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jefrey Wu. 2024. Scaling and evaluating sparse autoencoders. arXiv preprint arXiv:2406.04093 (2024).

[20] Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry P Vetrov, and Andrew G Wilson. 2018. Loss surfaces, mode connectivity, and fast ensembling of dnns. Advances in neural information processing systems 31 (2018).

[21] Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A Wichmann, and Wieland Brendel. 2018. ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. In International conference on learning representations.

[22] Amin Ghiasi, Hamid Kazemi, Eitan Borgnia, Steven Reich, Manli Shu, Micah Goldblum, Andrew Gordon Wilson, and Tom Goldstein. 2022. What do vision

transformers learn? a visual exploration. arXiv preprint arXiv:2212.06727 (2022).

[23] Ian J Goodfellow, Oriol Vinyals, and Andrew M Saxe. 2014. Qualitatively characterizing neural network optimization problems. arXiv preprint arXiv:1412.6544 (2014).

[24] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. 2022. Masked autoencoders are scalable vision learners. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 16000–16009.

[25] Taicheng Huang, Zonglei Zhen, and Jia Liu. 2021. Semantic relatedness emerges in deep convolutional neural networks designed for object recognition. Frontiers in computational neuroscience 15 (2021), 625804.

[26] Robert Huben, Hoagy Cunningham, Logan Smith, Aidan Ewart, and Lee Sharkey. 2024. Sparse autoencoders find highly interpretable features in language models. In International Conference on Learning Representations, Vol. 2024. 7827–7845.

[27] Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. 2024. Position: The platonic representation hypothesis. In Forty-first International Conference on Machine Learning.

[28] Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. 2018. Averaging weights leads to wider optima and better generalization. arXiv preprint arXiv:1803.05407 (2018).

[29] Adam Karvonen, Can Rager, Johnny Lin, Curt Tigges, Joseph Bloom, David Chanin, Yeu-Tong Lau, Eoin Farrell, Callum McDougall, Kola Ayonrinde, et al. 2025. Saebench: A comprehensive benchmark for sparse autoencoders in language model interpretability. arXiv preprint arXiv:2503.09532 (2025).

[30] Adam Karvonen, Can Rager, Samuel Marks, and Neel Nanda. 2024. Evaluating sparse autoencoders on targeted concept erasure tasks. arXiv preprint arXiv:2411.18895 (2024).

[31] Tarun Khajuria, Braian Olmiro Dias, Marharyta Domnich, and Jaan Aru. 2025. Interpreting the structure of multi-object representations in vision encoders. In World Conference on Explainable Artificial Intelligence. Springer, 359–382.

[32] Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, et al. 2018. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav). In International conference on machine learning. PMLR, 2668–2677.

[33] Jinyeong Kim, Junhyeok Kim, Yumin Shim, Joohyeok Kim, Sunyoung Jung, and Seong Jae Hwang. 2025. Interpreting vision transformers via residual replacement model. arXiv preprint arXiv:2509.17401 (2025).

[34] Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

[35] Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. 2020. Concept bottleneck models. In International conference on machine learning. PMLR, 5338–5348.

[36] Neehar Kondapaneni, Oisin Mac Aodha, and Pietro Perona. 2025. Representational Similarity via Interpretable Visual Concepts. In International Conference on Learning Representations, Vol. 2025. 12850–12881.

[37] Neehar Kondapaneni, Oisin Mac Aodha, and Pietro Perona. 2026. Representational Diference Explanations. Advances in Neural Information Processing Systems 38 (2026), 71478–71527.

[38] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geofrey Hinton. 2019. Similarity of neural network representations revisited. In International conference on machine learning. PMlR, 3519–3529

[39] Patrick Leask, Bart Bussmann, Michael Pearce, Joseph Bloom, Curt Tigges, Noura Al Moubayed, Lee Sharkey, and Neel Nanda. 2025. Sparse autoencoders do not find canonical units of analysis. arXiv preprint arXiv:2502.04878 (2025).

[40] Guillaume Leclerc, Andrew Ilyas, Logan Engstrom, Sung Min Park, Hadi Salman, and Aleksander Madry. 2022. fcv. https://github.com/libfcv/fcv/. commit 3a12966.

[41] Yann LeCun, Bernhard Boser, John S Denker, Donnie Henderson, Richard E Howard, Wayne Hubbard, and Lawrence D Jackel. 1989. Backpropagation applied to handwritten zip code recognition. Neural computation 1, 4 (1989), 541–551.

[42] Daniel D Lee and H Sebastian Seung. 1999. Learning the parts of objects by non-negative matrix factorization. nature 401, 6755 (1999), 788–791.

[43] Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein. 2018. Visualizing the loss landscape of neural nets. Advances in neural information processing systems 31 (2018).

[44] Hyesu Lim, Jinho Choi, Jaegul Choo, and Stefen Schneider. 2024. Sparse autoencoders reveal selective remapping of visual concepts during adaptation. arXiv preprint arXiv:2412.05276 (2024).

[45] Mengchen Liu, Jiaxin Shi, Zhen Li, Chongxuan Li, Jun Zhu, and Shixia Liu. 2016. Towards better analysis of deep convolutional neural networks. IEEE transactions on visualization and computer graphics 23, 1 (2016), 91–100.

[46] Ilya Loshchilov and Frank Hutter. 2016. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983 (2016).

[47] Luke Marks, Alasdair Paren, David Krueger, and Fazl Barez. 2024. Enhancing neural network interpretability with feature-aligned sparse autoencoders. arXiv preprint arXiv:2411.01220 (2024).

[48] Alexis Marouani, Oriane Siméoni, Hervé Jégou, Piotr Bojanowski, and Huy V Vo. 2026. Revisiting [CLS] and Patch Token Interaction in Vision Transformers.

arXiv preprint arXiv:2602.08626 (2026).

[49] Andrew Ng et al. 2011. Sparse autoencoder. CS294A Lecture notes 72, 2011 (2011), 1–19.

[50] Matthew Lyle Olson, Musashi Hinck, Neale Ratzlaf, Changbai Li, Phillip Howard, Vasudev Lal, and Shao-Yen Tseng. 2025. Probing the representational power of sparse autoencoders in vision models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 6167–6177.

[51] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. 2023. DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193 [cs.CV]

[52] Mateusz Pach, Shyamgopal Karthik, Quentin Bouniot, Serge Belongie, and Zeynep Akata. 2025. Sparse autoencoders learn monosemantic features in vision language models. arXiv preprint arXiv:2504.02821 (2025).

[53] Kenny Peng, Rajiv Movva, Jon Kleinberg, Emma Pierson, and Nikhil Garg. 2025. Use sparse autoencoders to discover unknown concepts, not to act on known concepts. arXiv preprint arXiv:2506.23845 (2025).

[54] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.

[55] Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein. 2017. Svcca: Singular vector canonical correlation analysis for deep learning dynamics and interpretability. Advances in neural information processing systems 30 (2017).

[56] Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, and Alexey Dosovitskiy. 2021. Do vision transformers see like convolutional neural networks? Advances in neural information processing systems 34 (2021), 12116– 12128.

[57] Yuval Sharon and Yehuda Dar. 2024. How Does Perfect Fitting Afect Representation Learning? On the Training Dynamics of Representations in Deep Neural Networks. CoRR (2024).

[58] Samuel Stevens, Wei-Lun Chao, Tanya Berger-Wolf, and Yu Su. 2025. Interpretable and Testable Vision Features via Sparse Autoencoders. arXiv preprint

arXiv:2502.06755 (2025).

[59] Demian Till. 2024. Do sparse autoencoders find "true features". https://www.lesswrong.com/posts/QoR8noAB3Mp2KBA4B/do-sparseautoencoders-find-true-features

[60] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. 2021. Training data-eficient image transformers & distillation through attention. In International conference on machine learning. PMLR, 10347–10357.

[61] Romeo Valentin, Sydney M Katz, Vincent Vanhoucke, and Mykel J Kochenderfer. 2025. DB-KSVD: Scalable Alternating Optimization for Disentangling High-Dimensional Embedding Spaces. arXiv preprint arXiv:2505.18441 (2025).

[62] Johanna Vielhaben, Dilyara Bareeva, Jim Berend, Wojciech Samek, and Nils Strodthof. 2026. Beyond scalars: Concept-based alignment analysis in vision transformers. Advances in Neural Information Processing Systems 38 (2026), 71379–71414.

[63] Shaoru Wang, Jin Gao, Zeming Li, Xiaoqin Zhang, and Weiming Hu. 2023. A closer look at self-supervised lightweight vision transformers. In International conference on machine learning. PMLR, 35624–35641.

[64] Yang Xu, Yi Wang, Hengguan Huang, and Hao Wang. 2024. Tracking the feature dynamics in llm training: A mechanistic study. arXiv preprint arXiv:2412.17626 (2024).

[65] Zeyu Yun, Yubei Chen, Bruno Olshausen, and Yann LeCun. 2021. Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors. In Proceedings ofDeep Learning Inside Out (DeeLIO): The 2nd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures. 1–10.

[66] Matthew D Zeiler and Rob Fergus. 2014. Visualizing and understanding convolu tional networks. In European conference on computer vision. Springer, 818–833.

[67] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. 2017. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412 (2017).

[68] Hongyi Zheng, Hongwei Yong, and Lei Zhang. 2021. Deep convolutional dictionary learning for image denoising. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 630–641.

## A Appendix

Here we present additional results and hyperparameters that could not fit in the main text.

## A.1 The Number of Features

In Figure 13 we report the number of SAE features that activate on more than 0.1% of test samples i.e., the number of features discovered by the SAE in each checkpoint, providing a cue of representational complexity throughout the learning process. Otherwise the feature is considered dead (number of dead features is therefore 960−"Number of features").

![](images/78ba3036dd925004025cc9756b90b146d6cd9fce63123f54136aefe91feea363.jpg)

![](images/6405dcef9a4ff836e68820091e5adbc5eaf41a0ed7d44e769aab3d8320c2fbf3.jpg)  
Figure 13: Number of features found by SAE.

![](images/81b364792b8c7273b727b76490fa2f1a32838cf35819f75cc41e15c33c4dd37c.jpg)  
Figure 14: Percentage of features migrating towards deeper layers on Mixed 10.

![](images/b151c94d863039ef1723545d19d45b54cbd8750f2cfd6942d325394ca7760e3c.jpg)

Figure 15: Percentage of stable features on Mixed 10.  
![](images/623ade3e066076c14dd8305fa51bcc60d7ad96c4cfe1fe94e123c01c9197fe09.jpg)  
Figure 16: Mean active lifetime of features, given in number of epochs on Mixed 10.

![](images/8a9a1b61674740e1f1e111cea539608b9177311cd23f8469922258e800ada19c.jpg)

Figure 17: Percentage of features that are widely spread across layers on Mixed 10.  
![](images/18eb8c015cfee05a4812bd590d1a86c0659a149e1acf9a85cfdab09a69c50304.jpg)  
Figure 18: Mean signed active drift. For each reference epoch– layer pair, feature-level quantities are averaged across features. Average signed displacement of the layer center, in layer-index units; negative values indicate movement toward earlier layers on Mixed 10.

Mixed 10 : Average active-layer trajectory (COM, active\_mode=threshold)  
![](images/a43fc6418e2c496ece6e4b2488c364e574934055796191369a12d3e5fb56e09d.jpg)  
Figure 19: Layer center-of-mass (COM) ± standard deviation trajectories averaged over all the features of selected reference checkpoints (0,100,200 and 300). The high-frequency epochs are depicted as a solid line around the reference checkpoints (marked as x) and the low-frequency epochs as dots. On the figure below, the average layer width ± standard deviation is depicted. The averaging is carried out only on active epochs per feature and therefore color fading signifies the lack of active features in that area.

## A.2 Mixed 10 Additional Results

In this section, we report the Mixed 10 counterparts that were omitted from the main paper. Deeper migratory features are depicted in Figure 14, percentage of widely spread features in Figure 17, mean active lifetime of features in Figure 16, percentage of stable features in Figure 15, mean signed active drift in Figure 18 and layer center of mass trajectories in Figure 19.

## A.3 Feature Tracking Example

The results shown in the main part of the paper cover the aggregate perspective of feature evolution and migration; moreover, this methodology provides, as a by-product, a view into individual feature tracking. In Figure 20, top image, we can see the evolution pattern of a specific feature starting from epoch 1 and layer 8. The images below depict the best-matching feature in the selected following epochs and their respective evolution pattern creating a clearer understanding of where the specific feature resides. This approach allows us to see the tiny scale of learning and could be used alongside annotated features.

This also serves as a sanity check that the features matching via correlation manifest also in similar features (or at least similar topactivating images representing tube-like objects in this example).

## A.4 Computational Cost

SAE training time and computation of epoch-by-layer views for ViT-Tiny trained on ImageNet-1k on one H200 GPU took ≈ 192 hours.

Due to the strict compute budget, we only explored ViT-Tiny but in future work, training larger models or including the patch-token perspective, the analysis and training pipeline should be made online as storing all the checkpoints and activations is extremely memory-heavy. Furthermore, even though we used a combination of two SAEs per checkpoint, stronger claims would require multiseed experiments.

## A.5 SAE Loss

Since we trained all the SAEs for 150 epochs instead of training until convergence, we demonstrate the test loss of each SAE in Figure 21. We can see that the loss is higher in the deeper layers and later in training, which correlates visually with the number-offeatures analysis in Figure 13. This could be due to the complexity of representations increasing in the deeper layers and later training rendering reconstruction harder.

SAEs were trained using Adam [34] optimizer with a learning rate (LR) of 0.005, without an LR scheduler or dead-feature resampling. The inputs are ℓ<sub>2</sub>-normalized, and each decoder dictionary vector is renormalized to unit ℓ<sub>2</sub>-norm after every optimizer update, thereby removing the scale ambiguity between decoder weights and feature activations [4, 19].

![](images/c746580addce4fbc3771965d4415d7307d381412ae2d5f7cb4febac4bfc7840d.jpg)  
Figure 20: This figure depicts an SAE feature 881 (marked with ★) evolution discovered from checkpoint epoch 1, layer 8 (top figure). Figures below show the best-matching target feature at each selected checkpoint (marked with ★) and the target feature’s top-5 activating images (example image thumbnails are from the ImageNet-1k dataset).

## A.6 Comparison to CKA

To motivate the feature-similarity view, we demonstrate the representation similarity via CKA. This is a widely used measure to compare the backbone representations $X ^ { \mathcal { A } } \in \dot { \mathbb { R } } ^ { n \times d _ { \ell } }$ and $X ^ { \mathcal { B } } \in \mathbb { R } ^ { n \times d _ { \ell ^ { \prime } } }$ on the same dataset D. We use linear CKA [38] to demonstrate the diference to our approach on column-centered representations $X ^ { \mathcal { A } }$ and $X ^ { \mathcal { B } }$

$$
\operatorname { C K A } ( X ^ { \mathcal { A } } , X ^ { \mathcal { B } } ) = \frac { \Vert X ^ { \mathcal { A } ^ { \top } } X ^ { \mathcal { B } } \Vert _ { F } ^ { 2 } } { \Vert X ^ { \mathcal { A } ^ { \top } } X ^ { \mathcal { A } } \Vert _ { F } \Vert X ^ { \mathcal { B } ^ { \top } } X ^ { \mathcal { B } } \Vert _ { F } } .\tag{21}
$$

CKA provides a robust global measure of representational similarity and is invariant to orthogonal linear transformations and isotropic scaling [38]. It is therefore well-suited for comparing layers and checkpoints of diferent dimensionalities. But CKA does not imply feature similarity: two layers may have high CKA while encoding diferent features, or low CKA while still supporting related features under a diferent parameterization [62]. In ViTs, high interlayer similarity may also partly reflect residual pathways rather than truly shared semantics [16, 56]. In Figure 22, we can see an example of CKA similarity between a reference representation (epoch 1, layer 8) to every other epoch-layer pair across 12 layers and 120 epochs.

![](images/22306a6a3223ae5ae5bea045d27551dbfea49c3e03995d9019eba74bf9fd7321.jpg)  
Figure 21: SAE test loss on a log-scale across time and depth on ImageNet-1k trained ViT checkpoints.

Table 2: Efective feature-evolution hyperparameters.
<table><tr><td>Quantity</td><td>Value</td><td>Role</td></tr><tr><td> $k _ { \mathrm { o n } }$ </td><td>3</td><td>Active checkpoints for emergence</td></tr><tr><td> $k _ { \mathrm { o f f } }$ </td><td>5</td><td>Inactive checkpoints for disappearance</td></tr><tr><td> $\sigma _ { \mathrm { m a x } }$ </td><td>4</td><td>Maximum layer width</td></tr><tr><td> $f _ { \mathrm { s e } }$ </td><td>0.05</td><td>Fraction of valid active epochs at drift endpoints</td></tr><tr><td> $\delta$ </td><td>1</td><td>Anchor/reference stability radius</td></tr><tr><td> $d _ { \mathrm { m i n } }$ </td><td>1</td><td>Strict migration threshold in layers</td></tr><tr><td> $\lambda _ { \operatorname* { m i n } }$ </td><td>0.5</td><td>Spread-feature cutoff</td></tr><tr><td> $S _ { \mathrm { m i n } }$ </td><td>0.7</td><td>Minimum stability occupancy</td></tr><tr><td> $d ^ { \mathrm { m a x } }$   $a _ { \mathrm { s t a b l e } } ^ { \mathrm { a } }$ </td><td>0.5</td><td>Maximum stable net drift</td></tr></table>

On the other hand, in Figure 23, we can see the similarity of several chosen features from the same reference epoch 1 and layer 8 (i.e., features from the point, where it is most active or discoverable) to every other epoch-layer pair features using our method. CKA seems to capture the general trend but lacks the resolution of how the features from that reference checkpoint evolve and exhibit diverse patterns rather than a mere trend.

## A.7 Feature Types

Figure 3 in the main text provides the class names for the diferent feature types. Figure 24 additionally shows three maximally activating images for each class as illustrative examples.

## A.8 Additional Hyperparameters

Here we report the hyperparameters for feature evolution metrics.

For the emergence and disappearance metrics $k _ { o n }$ and $k _ { o f f }$ can mean, depending on the experiment, consecutive epochs or consecutive sampled checkpoints. In our experiments, around the high frequency window, we used only every 10th epoch, hence, $k _ { o n }$ and $k _ { o f f }$ meant consecutive sampled checkpoints. Hyperparameter values are reported in Tabel 2.

![](images/a347621b2d5042c45ce132d2d3badc2737d3148b4258b7c80c7c1b2d211e27ca.jpg)  
Figure 22: Linear CKA between the ImageNet-1k representation at epoch 1, layer 8 and every sampled epoch–layer pair.

Feature Evolution Patterns (Epoch 1, Layer 8) Maximum Spearman correlation similarity  
![](images/0e9a4f89ebbd3fbc4be7a115544f28e0c0531698f58c66e72392928ece8974b5.jpg)  
Figure 23: This figure depicts a diverse selection of SAE feature evolutions decoded from checkpoint epoch 1, layer 8 (marked with ★) and the feature’s top-5 activating images (example image thumbnails are from the ImageNet-1k dataset).

![](images/1b0a9df476d2848869ebcba136d83bf95838116a28725cab0efc13626770fb25.jpg)

![](images/4fc9e0b6fea1744d6df92ee3ff6bea742c502605fb42376b2f38121588f51d4a.jpg)  
Figure 24: Feature-type examples illustrated by maximally activating images [66]. Human-understandable SAE features are observed under 3 main categories: class-specific, superclass-specific and inter-class (Example image thumbnails are from the ImageNet-1k dataset).