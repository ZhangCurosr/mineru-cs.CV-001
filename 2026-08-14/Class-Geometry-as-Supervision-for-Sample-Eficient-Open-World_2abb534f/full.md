# Class Geometry as Supervision for Sample-Eficient Open-World Detection

Akash Rao<sup>1∗</sup>, Zhou Chen<sup>1†</sup>, Revanth Reddy Palem<sup>1†</sup>, Udhav Ramachandran<sup>2</sup>, Ruth Scimeca<sup>2</sup>,

Sathyanarayanan N. Aakur<sup>1∗</sup>

<sup>1</sup>CSSE Department, Auburn University, Auburn, AL, 36849

<sup>2</sup> Department of Veterinary Pathobiology, Oklahoma State University, Stillwater, OK, 74078 {azr0187,san0028}@auburn.edu

## Abstract

Open-world object detection requires models to recognize known categories, reject unfamiliar objects, and incorporate new classes over time. This is especially challenging in scarce-data settings such as biomedical and scientific imaging, where rare categories may have only a few annotated examples and fine-grained classes difer by subtle morphology. Prototype-based detectors are natural for this regime, but they typically learn class prototypes as independent anchors, ignoring relational structure among classes. We propose class-geometry supervision (CGS), a general framework that constrains learned prototype or class-representation spaces to preserve visual or semantic class dissimilarities estimated from training data. CGS introduces a dissimilarity-preserving objective that aligns pairwise distances among learned class representations with a target class-geometry matrix while retaining the standard task loss. We instantiate the same objective across prototype recognition, few-shot biomedical object detection, open-set detection, novel-class insertion, and OWOD adaptation on COCO. Experiments show that CGS improves sample eficiency in recognition and ova detection, substantially strengthens novel-class insertion, and improves unknown recall on COCO while retaining much of the knownclass detection performance. Ablations show that meaningful visual geometry provides the most reliable gains, while random geometry can help novel separation but is less consistent for few-shot detection. These results suggest that relational class geometry is an efective supervisory signal for building calibrated and extensible open-world detectors under limited supervision.

## Introduction

Open-world object detection requires a model to detect known categories, identify unfamiliar objects, and incorporate new classes over time. This capability is especially important in biomedical and scientific imaging, where rare categories, emerging subtypes, and expert-defined labels often appear with only a handful of annotated examples. In these settings, detection cannot rely on large-scale retraining whenever new classes arise. Instead, models must support sample-eficient open-world detection: learning known categories from scarce examples, recognizing when objects fall outside the known label set, and inserting new categories with minimal supervision. While existing open-world detectors have made substantial progress under standard benchmark protocols, they are often evaluated in data-rich regimes and provide limited guidance for scarcely labeled data.

The central dificulty is that scarce examples provide an incomplete view of each class. This is particularly problematic in fine-grained domains, where visually similar categories difer only by subtle morphology, and where unknown objects may be mistaken for rare known classes or background. Prototype-based recognition and detection ofer a natural starting point because they represent each class using a small support set. However, conventional prototype learning treats class prototypes as independent anchors: each class is pulled toward its examples and pushed away from others only through task loss. This ignores a key source of structure that is often available even when labels are scarce: classes have geometry. Some categories are visually, semantically, or morphologically close, while others are distant. Without this relational structure, prototype spaces can become poorly calibrated for open-world decisions.

In this work, we propose to use class geometry as supervision for sample-eficient open-world detection. Rather than learning prototypes solely from scarce labeled examples, we constrain the inter-class prototype space to preserve a target geometry derived from class-level visual or semantic dissimilarities. In the biomedical setting, this geometry is estimated from training examples by computing average pixel-level differences between classes, capturing coarse morphology relationships among ova; we also study language-derived dissimilarities from visual class descriptions. As shown in Fig. 1, this provides supervision over how categories should be arranged: visually similar classes are encouraged to remain nearby, while dissimilar classes are separated. We realize this principle through a dissimilarity-preserving prototype objective, which aligns pairwise distances among class prototypes with a target visual or semantic class-dissimilarity matrix, complementing the standard recognition losses. Because the constraint acts on relationships among class prototypes rather than on individual embeddings, it organizes the class manifold while preserving intra-class variation.

We evaluate class-geometry supervision across recognition, biomedical detection, and COCO open-world detection. First, we apply CGS to ProtoNet-style recognition to isolate its efect on scarce-data prototype learning (Trehan et al. 2026). Second, we integrate CGS into a prototypebased DETR detector for parasitic ova (Trehan et al. 2026), where region features are matched to support-derived class prototypes for few-shot detection, open-set detection, and novel-class insertion. Third, we evaluate CGS on the standard OWOD/COCO protocol (Gupta et al. 2022) in two regimes: frozen-head adaptation over fixed OW-DETR features and full-model fine-tuning. Across these settings, CGS improves low-shot recognition and few-shot ova detection, yields large gains in novel and generalized novel-class insertion, and improves unknown recall on COCO. The COCO results reveal a tradeof: frozen-head CGS substantially improves openworld sensitivity, while full-model fine-tuning reduces the mAP cost and preserves known-class performance.

![](images/04c5a7fb0582bae53898729018ba046796a11e4f914573b5cde595f8403946d3.jpg)  
Figure 1: Class geometry supervision. Standard prototype learning can produce poorly calibrated class anchors under scarce supervision, causing unknown objects to be misclassified. We construct a training-only class geometry D from support-crop morphology and text descriptions, then use L to align the learned prototype graph with this target geometry. The resulting space better supports unknown rejection and novel-class insertion.

Our contributions are four-fold: (i) We formulate classgeometry supervision as a mechanism for sample-eficient open-world detection, where known-class detection, unknown discovery, and novel-class insertion must operate under limited labeled data. (ii) We propose a dissimilaritypreserving prototype objective that aligns pairwise distances among learned class representations with visual or semantic class-dissimilarity structure, complementing standard matching and detection losses. (iii) We instantiate this objective across prototype recognition and biomedical open-world detection, showing gains in low-shot recognition, few-shot detection, open-set detection, and novel-class insertion. (iv) We evaluate CGS on COCO OWOD in frozen-head and full fine-tuning regimes, showing that class geometry improves unknown recall while exposing a controllable tradeof with known-class mAP.

## Related Work

Open-world and open-vocabulary object detection Object detection has evolved from restrictive closed-set models to advanced open-world (OWOD) and open-vocabulary (OVOD) paradigms capable of identifying unknown objects and incrementally learning new categories. While OWOD frameworks—building on foundational models like DETR (Carion et al. 2020) have refined the separation of knowns and unknowns through advanced probabilistic and neutralization techniques (Joseph et al. 2021; Gupta et al. 2022; Zohar, Wang, and Yeung 2023; Xi et al. 2024), OVOD models leverage large vision-language architectures like OWL-ViT (Minderer et al. 2022) and its scaled successors (Minderer, Gritsenko, and Houlsby 2023) to detect novel items via text prompts, often utilizing multi-modal classifiers for improved zero-shot adaptability (Kaul, Xie, and Zisserman 2023). However, a major limitation of both approaches is their heavy reliance on extensive labeled base data, complex pseudo-labeling, or computationally expensive end-toend training pipelines. To overcome data barriers, we utilize class-level visual and semantic geometry to make open-world detection highly sample-eficient, seamlessly integrating new object classes under limited per-class supervision.

Few-shot, prototype-based, and scarce-data detection Addressing the challenge oflimited supervision, few-shot object detection (FSOD) seeks to recognize novel objects using only a handful of support examples. Foundational work in meta-learning, such as Prototypical Networks (Snell, Swersky, and Zemel 2017), established the paradigm of classifying instances based on their distance to learned class anchors or prototypes. This prototype-based approach has been extensively adapted for detection: early advancements utilized meta-learning frameworks like Meta R-CNN (Yan et al. 2019) to aggregate support features, while subsequent research demonstrated the eficacy of two-stage fine-tuning (Wang et al. 2020) and decoupled training pipelines like DeFRCN (Qiao et al. 2021). Additionally, methods such as FSCE (Sun et al. 2021) have focused on refining the feature space through contrastive proposal encoding to better separate object classes. However, the key distinction is that prior prototype methods learn class anchors from support examples but do not explicitly constrain the relational geometry among the prototypes themselves. While few-shot detectors improve adaptation from limited support examples through meta-learning, fine-tuning, contrastive encoding, or prototype calibration, our approach structurally departs from this paradigm by supervising the prototype space using known class relationships, ensuring that scarce examples are inherently arranged in a more calibrated open-world geometry.

Class-relational, semantic, and geometry-aware representation learning. Moving beyond the assumption of independent classes, classical Multidimensional Scaling (Kruskal 1964) and modern contrastive learning paradigms (Schrof, Kalenichenko, and Philbin 2015; Khosla et al. 2020) structure representation spaces by mapping relationships to geometric constraints. Similarly, semantic structures have been explicitly modeled to improve generalization through zero-shot embeddings (Liu et al. 2021), hierarchy-aware losses (Bertinetto et al. 2020), and vision-language priors like CLIP (Radford et al. 2021) and BioCLIP (Stevens et al. 2024). While these methods demonstrate that labels contain rich relational structure beyond one-hot identity, we build on this view by applying it directly to detector prototypes. Unlike prior open-world detectors that rely on high-data endto-end training, or few-shot detectors that learn class anchors independently, we use class-level visual and semantic dissimilarities as explicit supervisory signals. By preserving this geometry during training, we structurally organize the prototype space for sample-eficient open-world detection.

## Proposed Approach

Problem Statement. We consider sample-eficient openworld detection, where a model must detect known classes, identify objects outside the current label set, and incorporate newly labeled classes over time. Let $\mathcal { C } _ { t } = \{ 1 , \dots , \bar { C _ { t } } \}$ denote the known classes at stage t, and let $\mathcal { U } _ { t }$ denote unknown classes that may appear but are not yet annotated with class identities. For each known class $c \in { \mathcal { C } } _ { t } .$ , the model receives a small support set $S _ { c } = \{ ( I _ { c , k } , b _ { c , k } ) \} _ { k = 1 } ^ { K }$ , where $I _ { c , k }$ is a support image, $b _ { c , k }$ is a support region or bounding box, and $K$ is small. $\mathbf { A }$ detector produces candidate regions or object queries $b ,$ and a region encoder maps each region to a feature $\dot { z } = f _ { \theta } ( I , b ) \in \mathring { \mathbb { R } ^ { d } }$ . Each known class is represented by a support-derived prototype $p _ { c } = | S _ { c } | ^ { - 1 } { \textstyle \sum } _ { ( I , b ) \in S _ { c } } f _ { \theta } ( I , b )$ A query region is classified by its similarity or distance to the current prototype set $\mathcal { P } _ { t } = \dot { \{ p _ { c } : c \in \mathcal { C } _ { t } \} }$ , while regions that do not match any known prototype with suficient confidence are treated as unknown. Standard prototype learning optimizes these prototypes only through a recognition or detection objective, treating them as independent class anchors. Under scarce supervision, this leaves the prototype space underconstrained: many prototype configurations can explain the limited support examples, but induce diferent behavior for unknown rejection and novel-class insertion.

We address this limitation by introducing class-level geometry supervision. Let $D \in \overline { { \mathbb { R } } } ^ { C _ { t } \times C _ { t } }$ be a target classdissimilarity matrix, where $D _ { i j }$ encodes the visual or semantic dissimilarity between classes i and $j .$ In our biomedical setting, we estimate visual geometry from training examples

$$
D _ { i j } ^ { \mathrm { v i s } } = \frac { 1 } { | S _ { i } | | S _ { j } | } \sum _ { ( I , b ) \in S _ { i } } \sum _ { ( I ^ { \prime } , b ^ { \prime } ) \in S _ { j } } \mathrm { M S E } ( \phi ( I , b ) , \phi ( I ^ { \prime } , b ^ { \prime } ) ) ,\tag{1}
$$

where $\phi ( I , b )$ denotes the cropped and resized image region corresponding to box b. This captures coarse pixel-level morphology diferences between classes. We also consider language-derived geometry $D _ { i j } ^ { \mathrm { t x t } } = 1 - \cos ( e _ { i } , e _ { j } )$ , where $e _ { i }$ and $e _ { j }$ are text embeddings of visual descriptions for classes i and $j$ . We use $D$ as a nonnegative dissimilarity matrix, not necessarily a metric; the objective only requires pairwise relational targets. The matrix $D$ is computed only from training/support information and provides a relational prior over the label space. The learned prototype space induces its own class-distance matrix $G ,$ with entries $\bar { G } _ { i j } = d ( p _ { i } , p _ { j } )$ where $d ( \cdot , \cdot )$ is a distance in the prototype space, such as Euclidean or cosine distance. To compare distances that may have diferent scales, we normalize the of-diagonal entries as $\hat { G } _ { i j } = ( G _ { i j } - \mu _ { G } ) / \sigma _ { G }$ and $\hat { D } _ { i j } = ( D _ { i j } - \mu _ { D } ) / \sigma _ { D } ,$ where $\mu _ { G } , \sigma _ { G }$ and $\mu _ { D } , \sigma _ { D }$ are computed over class pairs $i < j$ . Let $\mathcal { E } = \{ ( i , j ) : 1 \leq i < j \ \overset { \cdot } { \leq } C _ { t } \}$ . We define the class-geometry loss as

$$
\mathcal { L } _ { \mathrm { C G } } = \frac { 1 } { \vert \mathcal { E } \vert } \sum _ { ( i , j ) \in \mathcal { E } } \left( \hat { G } _ { i j } - \hat { D } _ { i j } \right) ^ { 2 } .\tag{2}
$$

The full objective is $\mathcal { L } = \mathcal { L } _ { \mathrm { t a s k } } + \lambda \mathcal { L } _ { \mathrm { C G } }$ , where $\mathcal { L } _ { \mathrm { t a s k } }$ denotes the standard prototype matching, classification, or detection loss, and λ controls class-geometry supervision strength.

What Does Class-Geometry Supervision Control? The proposed objective aligns two complete weighted graphs over the class set: a target class graph with edge weights $\hat { D } _ { i j }$ , encoding visual or semantic dissimilarity, and a learned prototype graph with edge weights $\hat { G } _ { i j } .$ , encoding distances between learned class prototypes. The following proposition shows that minimizing $\mathcal { L } _ { \mathrm { C G } }$ directly controls the average distortion between these graphs.

Proposition 1 (Prototype-graph distortion bound). If $\mathcal { L } _ { \mathrm { C G } } \leq \epsilon ,$ , then the average absolute distortion between the learned prototype graph and the target class-dissimilarity graph satisfies

$$
\frac { 1 } { | \mathcal { E } | } \sum _ { ( i , j ) \in \mathcal { E } } \left| \hat { G } _ { i j } - \hat { D } _ { i j } \right| \leq \sqrt { \epsilon } .
$$

Proposition 1 shows that $\mathcal { L } _ { \mathrm { C G } }$ is not an unconstrained regularizer: it explicitly minimizes average distortion between the learned prototype geometry and the target class geometry. While this controls average graph distortion, preserving local neighborhood topology requires suficiently small pairwise distortion for the relevant class pairs. The next proposition formalizes this margin condition.

Proposition 2 (Neighborhood-order preservation). Suppose the target class geometry indicates that class $j$ is closer to class i than class k by margin $\gamma > 0 , i . e . , \hat { D } _ { i j } + \gamma \leq \hat { D } _ { i k } .$ Ifthe learned prototype distances satisfy $| \hat { G } _ { i j } - \hat { D } _ { i j } | \leq \gamma / 2$ and $| \hat { G } _ { i k } - \hat { D } _ { i k } | \le \gamma / 2 ,$ , then the learned prototype space preserves this neighborhood relation: $\hat { G } _ { i j } \leq \hat { G } _ { i k }$

Algorithm 1: Class-Geometry Supervised Prototype Learn  
ing   
Require: Support sets $\left\{ S _ { c } \right\} _ { c = 1 } ^ { C _ { t } } ,$ feature extractor $f _ { \theta } ,$ , task   
loss $\mathcal { L } _ { \mathrm { t a s k } } .$ geometry weight λ   
Require: Class dissimilarity matrix $D \in \mathbb { R } ^ { C _ { t } \times C _ { t } }$   
Ensure: Geometry-supervised prototype set $\mathcal { P } _ { t } = \{ p _ { c } \} _ { c = 1 } ^ { C _ { t } }$   
1: Normalize of-diagonal entries of $\bar { D }$ to obtain target ${ \mathrm { g e - } }$   
ometry D<sup>ˆ</sup>: $\hat { D } _ { i j } \gets ( D _ { i j } - \mu _ { D } ) / \sigma _ { D }$   
2: for each training episode or minibatch do   
3: Compute support class prototypes:   
$p _ { c } \gets \frac { 1 } { | S _ { c } | } \sum _ { ( I , b ) \in S _ { c } } f _ { \theta } ( I , b ) , \quad \forall c \in \{ 1 , \dots , C _ { t } \}$   
4: Form learned prototype distance matrix $G$ with entries   
$G _ { i j } = d ( p _ { i } , \bar { p _ { j } } )$   
5: Normalize of-diagonal entries of $G$ to obtain $\hat { G } \colon$   
$\hat { G } _ { i j } \gets ( G _ { i j } - \mu _ { G } ) / \sigma _ { G }$   
6: Compute dissimilarity-preserving loss:   
$\mathcal { L } _ { \mathrm { C G } }  \frac { 1 } { \vert \mathcal { E } \vert } \sum _ { ( i , j ) \in \mathcal { E } } ( \hat { G } _ { i j } - \hat { D } _ { i j } ) ^ { 2 }$   
7: Update parameters by minimizing $\mathcal { L }  \mathcal { L } _ { \mathrm { t a s k } } + \lambda \mathcal { L } _ { \mathrm { C G } }$   
8: end for   
9: Use $\mathcal { P } _ { t }$ for nearest-prototype prediction, unknown rejec  
tion, or novel-class insertion   
10: return $\mathcal { P } _ { t }$

Proposition 2 gives a useful interpretation for scarce-data and incremental settings. If the target class geometry has a suficient margin between nearby and distant classes, then a low-distortion prototype space preserves that neighborhood order. Thus, newly introduced classes can be placed into an existing prototype space according to their visual or semantic relationships with known classes, rather than being learned as isolated anchors. Proofs are provided in the supplementary.

Open-World Decisions with Prototype Geometry. The propositions above show that $\mathcal { L } _ { \mathrm { C G } }$ controls the distortion between the learned prototype graph and the target classdissimilarity graph. We now describe how this geometrysupervised prototype graph is used for open-world decisions. Algorithm 1 illustrates the process. Given a query region feature $z = f _ { \theta } ( I , b )$ , we compute its distance to each knownclass prototype $p _ { c } \in \mathcal { P } _ { t }$ . The predicted known label is

$$
\hat { y } = \arg \operatorname* { m i n } _ { c \in \mathcal { C } _ { t } } d ( z , p _ { c } ) .
$$

In our implementation, the open-world score is the nearestprototype distance: $m ( z ) { = } \operatorname* { m i n } _ { c \in { \mathcal { C } } _ { t } } d ( z , p _ { c } )$ . A region is assigned to its nearest known class only if this distance is below a global validation-selected rejection threshold $\tau _ { \mathrm { u n k } } ;$ otherwise it is marked as unknown:

$$
\mathrm { u n k n o w n } ( z ) = \mathbb { I } \left[ m ( z ) > \tau _ { \mathrm { u n k } } \right] .
$$

Thus, unknown rejection depends on the calibration of distances between query features and the known prototype space. Class-geometry supervision improves this decision rule by organizing known prototypes according to visual or semantic relationships. If prototypes are learned as independent anchors, nearest-prototype distances may be poorly calibrated: visually similar classes may be separated arbitrarily, while dissimilar classes may collapse. In contrast, minimizing $\mathcal { L } _ { \mathrm { C G } }$ reduces distortion between the learned prototype graph and the target class geometry, making the known-class manifold more structured. This is especially important for open-world detection, where unknown regions are identified by their failure to fit the known prototype geometry.

The same structure supports novel-class insertion. When a set of newly labeled classes $\Delta \mathcal { C }$ becomes available at stage $t + 1$ , we compute support-derived prototypes $p _ { c }$ for $c \in \Delta \mathcal { C }$ and expand the prototype set as $\mathcal { P } _ { t + 1 } { = } \mathcal { P } _ { t } \cup \{ p _ { c } : c \in \Delta \mathcal { C } \}$ In the incremental setting, existing prototypes in $\mathcal { P } _ { t }$ are kept fixed unless explicitly fine-tuned, and only the newly introduced class prototypes are added from their limited support examples. The class-dissimilarity matrix is similarly expanded to include relationships between new and existing classes. The geometry objective then encourages new prototypes to be inserted according to their visual or semantic relationships with the previous class $\operatorname { s e t } ,$ rather than being learned as isolated anchors. This connects directly to Proposition 2: when the target geometry has a suficient margin between nearby and distant classes, low pairwise distortion preserves the intended neighborhood order after insertion.

Applying Class-Geometry Supervision. The formulation above only assumes that the model produces class prototypes or class-specific embeddings. We therefore apply the same class-geometry objective in three settings: prototype recognition, prototype-based biomedical detection, and open-world detector adaptation.

Prototype recognition. We first apply class-geometry supervision to ProtoNet-style recognition to isolate its efect on scarce-data prototype learning. Given support examples for each class, prototypes are computed as class centroids in the embedding space and query images are classified by nearest-prototype matching. In this setting, $\mathcal { L } _ { \mathrm { t a s k } }$ is the standard prototype matching loss, and $\mathcal { L } _ { \mathrm { C G } }$ shapes only the interclass prototype geometry. This experiment evaluates whether preserving visual or semantic class relationships improves recognition before introducing detection-specific factors.

Prototype-based biomedical detection. We next apply the same objective to scarce-data biomedical detection. A DETR-style detector produces object queries and region features, and class labels are assigned by matching each region feature to support-derived class prototypes. The objective combines localization, matching, and CGS supervision:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { d e t } } + \mathcal { L } _ { \mathrm { p r o t o } } + \lambda \mathcal { L } _ { \mathrm { C G } } . } \end{array}
$$

Here, $\mathcal { L } _ { \mathrm { C G } }$ does not alter the localization objective; it structures the class prototype space used for known-class recognition, unknown rejection, and novel-class insertion. We use this instantiation for few-shot ova detection, open-set detection, and incremental 15-to-20 class learning.

Open-world detector adaptation. Finally, to test whether class-geometry supervision transfers beyond the biomedical benchmark, we apply it to the OW-DETR/COCO protocol. We evaluate two adaptation regimes. In the frozen-head setting, OW-DETR detector features are fixed and only a lightweight two-layer classification head is fine-tuned with the CGS objective applied to the induced class representations. In the full fine-tuning setting, the detector is optimized end-to-end with the same class-geometry loss added to the standard OW-DETR training objective. These settings allow us to test whether relational class supervision can improve unknown discovery at the classifier level alone, and whether full model adaptation can reduce the known-class mAP tradeof while retaining the open-world benefit.

![](images/6cf83d1935781bbfc14662aad1cd380c47ccd753d3dd0834d85c949b40a7c6fc.jpg)

![](images/ce6bce893072b493f0a9a862b9e7a2e4cd9a3a056013a5bd620cac1961f40352.jpg)

![](images/7d788d7c3664d3b4feb9505aa7f0e69c6f2173693ae85c44d5de5ee825365c88.jpg)  
Figure 2: Sample eficiency across recognition and detection. CGS improves performance across three increasingly challenging settings. (a) In prototype recognition, CGS improves low-shot classification. (b) In ova detection, CGS consistently improves FSP-DETR mAP. (c) On COCO Task 1, we freeze detector features and fine-tune only the classification MLP head with K labeled examples per class; we report relative change with respect to the Task 1 full OW-DETR reference.

Implementation Details. All models use the same image preprocessing, support-set construction, and train/test splits as the corresponding baseline unless otherwise stated. For recognition, we train a ProtoNet backbone with the standard episodic matching loss and add $\mathcal { L } _ { \mathrm { C G } }$ over episode prototypes using $\lambda \in \{ 0 . 0 5 , 0 . 0 1 , 0 . 1 , 0 . 5 , 1 \}$ }. For biomedical detection, we initialize a DETR model and apply the geometry loss after training using the vanilla DETR loss, using supportderived prototypes from $K \in \{ 5 , 1 0 , 1 5 , 2 0 \}$ examples per class. Visual dissimilarities are computed from cropped and resized support regions at resolution 224 × 224; text dissimilarities use MPNet (Song et al. 2020) embeddings of Google Gemini (Gemini Team, Google 2024) generated visual descriptions. For OW-DETR/COCO, we evaluate CGS in two regimes: frozen-head adaptation, where OW-DETR features are fixed and a two-layer classification MLP is fine-tuned, and full fine-tuning. Thresholds for unknown rejection, stress weights, and normalization hyperparameters are selected on the validation split; additional details in supplementary.

## Experimental Evaluation

Tasks and Datasets. We evaluate class-geometry supervision across three complementary settings designed to test whether relational class structure improves sample-eficient open-world learning. First, we use a parasitic ova benchmark (Trehan et al. 2026) for low-shot prototype recognition and few-shot detection, where fine-grained categories must be learned from limited support crops or bounding boxes. Second, on the same ova dataset, we evaluate open-set detection and novel-class insertion, testing whether geometrysupervised prototypes improve unknown rejection and the integration of newly labeled classes. Third, we evaluate on the standard OW-DETR/COCO (Gupta et al. 2022) open-world detection protocol to test whether the same idea transfers beyond biomedical imagery, including both frozen-head adaptation and full-model fine-tuning regimes. Across these settings, we report recognition F1/accuracy, detection mAP and mAR, unknown recall, and seen/novel-class performance.

<table><tr><td rowspan="2">Method</td><td colspan="2">Few-shot Detection mAP</td><td colspan="2">Open-set Detection</td></tr><tr><td>5-shot</td><td>10-shot</td><td>mAP</td><td>mAR</td></tr><tr><td colspan="5">Generic detectors</td></tr><tr><td>DETR</td><td>0.026</td><td>0.054</td><td>N/A</td><td>N/A-</td></tr><tr><td>YOLO</td><td>0.000</td><td>0.021</td><td>N/A</td><td>N/A</td></tr><tr><td>Stable-DINO</td><td>0.002</td><td>0.011</td><td>N/A</td><td>N/A</td></tr><tr><td>FCOS</td><td>0.001</td><td>0.016</td><td>N/A</td><td>N/A</td></tr><tr><td colspan="5">Prototype / few-shot baselines</td></tr><tr><td>ProtoNet + SS</td><td>0.036</td><td>0.045</td><td>0.003</td><td>0.026</td></tr><tr><td>ProtoKD + SS</td><td>0.039</td><td>0.048</td><td>0.031</td><td>0.109</td></tr><tr><td>FSP-DETR</td><td>0.066</td><td>0.094</td><td>0.045</td><td>0.141</td></tr><tr><td colspan="2">FSP-DETR + CGS 0.076 ↑</td><td>0.102↑</td><td>|0.061 ↑</td><td>0.136↓</td></tr></table>

Table 1: Ova detection under scarce-data and open-set settings. Arrows indicate improvement/decline relative to FSP-DETR without CGS. N/A: model not tailored for task.

Metrics and Baselines. We use task-standard metrics for each evaluation setting and compare against baselines matched to each task. For prototype-based recognition, we report macro-F1 over query examples and compare against a standard ProtoNet (Snell, Swersky, and Zemel 2017) trained with the same episodic protocol but without class-geometry supervision. For ova detection, we report mean Average Precision (mAP) and mean Average Recall (mAR), comparing against generic detectors including DETR (Carion et al. 2020), YOLO (Jocher et al. 2020), Stable-DINO (Liu et al. 2023), and FCOS (Tian et al. 2019), as well as prototype/fewshot baselines such as ProtoNet+SS (Snell, Swersky, and Zemel 2017) and ProtoKD+SS (Trehan et al. 2023). Our primary controlled comparison is FSP-DETR (Trehan et al. 2026), the strongest prototype-based detector baseline, with and without CGS. For open-set detection, mAP and mAR are computed with the unknown category included, and for novel-class insertion we report mAP and mAR under novelonly, generalized novel, and seen retention settings using the same known/novel splits. For COCO open-world detection, we follow the standard OWOD protocol (Joseph et al. 2021), reporting unknown recall (U-Recall) and mAP over previously known, currently introduced, and all known classes, and compare against ORE (Joseph et al. 2021), UC-OWOD (Wu et al. 2022), OW-DETR (Gupta et al. 2022), Fast-OWDETR (Chen 2022), OCPL (Yu et al. 2023), and RE-OWOD (Zhao et al. 2024).

<table><tr><td>Setting</td><td>Method</td><td>mAP</td><td>mAR</td></tr><tr><td rowspan="4">Novel-only</td><td>ProtoNet + SS</td><td>0.0486</td><td>0.2146</td></tr><tr><td>ProtoKD + SS</td><td>0.0532</td><td>0.2134</td></tr><tr><td>FSP-DETR</td><td>0.0629</td><td>0.2595</td></tr><tr><td>FSP-DETR + CGS</td><td>0.1558↑</td><td>0.3728↑</td></tr><tr><td rowspan="4">Generalized novel</td><td>ProtoNet + SS</td><td>0.0340</td><td>0.1710</td></tr><tr><td>ProtoKD + SS</td><td>0.0520</td><td>0.1960</td></tr><tr><td>FSP-DETR</td><td>0.0569</td><td>0.2138</td></tr><tr><td>FSP-DETR + CGS</td><td>0.1348↑</td><td>0.3248 ↑</td></tr><tr><td rowspan="4">Seen retention</td><td>ProtoNet + SS</td><td>0.0033</td><td>0.0236</td></tr><tr><td>ProtoKD + SS</td><td>0.0380</td><td>0.1107</td></tr><tr><td>FSP-DETR</td><td>0.0566</td><td>0.1173</td></tr><tr><td>FSP-DETR + CGS</td><td>0.0670↑</td><td>0.1259↑</td></tr></table>

Table 2: Novel-class insertion. CGS yields the largest gains when new classes are inserted into the prototype space.

Sample Eficiency Across Tasks. We first evaluate whether class-geometry supervision improves sample eficiency across recognition, detection, and open-world adaptation. Figure 2 shows that CGS improves ProtoNet recognition and FSP-DETR ova detection across support sizes, with the largest gains in the lowest-shot regimes. This indicates that the benefit is not detection-specific: when class prototypes are estimated from few examples, preserving inter-class geometry provides additional supervision that helps organize the class space before more data are available. In ova detection, this translates into consistent mAP gains over FSP-DETR without changing the detector backbone, suggesting that CGS improves the use of scarce support examples rather than merely increasing model capacity. On COCO Task 1, we freeze OW-DETR features and fine-tune only the classification head with K labeled examples per class; Fig. 2(c) reports relative change with respect to the full OW-DETR reference. CGS consistently increases unknown recall relative to the full model, although with lower known-class mAP. Thus, CGS shifts the adaptation toward open-world sensitivity: it is most useful when the goal is not only to maximize known-class accuracy, but to improve the model’s ability to detect objects outside the current label set under limited supervision.

<table><tr><td>Variant</td><td>5-shot</td><td>10-shot</td><td>Novel-only</td></tr><tr><td>No CGS</td><td>0.0658</td><td>0.0940</td><td>0.0629</td></tr><tr><td>Random D</td><td>0.0710</td><td>0.0975</td><td>0.1657</td></tr><tr><td>Text  $D _ { \mathrm { t x t } }$ </td><td>0.0717</td><td>0.0989</td><td>0.1476</td></tr><tr><td>Visual + Text</td><td>0.0741</td><td>0.0984</td><td>0.1581</td></tr><tr><td>Visual  $D _ { \mathrm { v i s } }$ </td><td>0.0759</td><td>0.1019</td><td>0.1558</td></tr></table>

Table 3: Efect of class-geometry source on ova detection. We compare diferent target class geometries used by CGS.

Scarce-Data and Open-Set Ova Detection. Table 1 compares CGS with generic detectors, prototype/few-shot baselines, and the strongest prototype detector baseline on ova detection. In the few-shot setting, generic detectors perform poorly under limited supervision, and prototype-based methods provide stronger performance. Among these, FSP-DETR is the strongest baseline, achieving 0.066 mAP in the 5- shot setting and 0.094 mAP in the 10-shot setting. Adding class-geometry supervision improves FSP-DETR to 0.076 and 0.102 mAP, respectively. These gains indicate that CGS improves the class prototype space without changing the detector backbone, with the largest relative improvement occurring in the most data-scarce 5-shot regime. We also evaluate open-set ova detection, where the model must reject objects outside the known class set. CGS improves open-set mAP from 0.045 to 0.061 over FSP-DETR, while maintaining comparable recall. This suggests that organizing the known-class prototype manifold with visual class geometry improves the nearest-prototype distance used for unknown rejection. The slight decrease in mAR reflects a precision– recall tradeof: CGS is more selective in assigning detections as “unknown”, resulting in higher mAP at lower recall.

Novel-Class Insertion. Table 2 evaluates whether classgeometry supervision helps insert newly labeled classes into an existing prototype space. We consider three settings: novel-only, where only the newly introduced classes are evaluated; generalized novel, where all prototypes are available but performance is measured on novel classes; and seen retention, where performance is measured on previously known classes after insertion. CGS provides substantial gains in the novel-only setting, improving mAP from 0.0629 to 0.1558 and mAR from 0.2595 to 0.3728. The gains remain large in the generalized novel setting, where mAP improves from 0.0569 to 0.1348 and mAR improves from 0.2138 to 0.3248. These results indicate that CGS helps newly introduced prototypes occupy meaningful positions in the existing class manifold rather than being learned as isolated anchors. CGS also improves seen-class retention, increasing mAP from 0.0566 to 0.0670 and mAR from 0.1173 to 0.1259. This suggests that the benefit of class geometry is not limited to novel classes; organizing the prototype space also helps preserve discrimination among previously known categories.

OWOD Adaptation on COCO. Table 4 evaluates whether class-geometry supervision transfers beyond the ova benchmark to the standard OWOD protocol on COCO. Rather than treating COCO only as a closed-set mAP benchmark, we use it to measure the tradeof between known-class detection and unknown discovery. We compare OW-DETR with two CGS variants: a frozen-head setting, where only the classification head is updated over fixed OW-DETR features, and a full finetuning setting, where the entire model is optimized with the CGS objective. The results show that CGS consistently shifts the detector toward stronger open-world sensitivity. In the frozen-head setting, CGS substantially improves U-Recal over OW-DETR across Tasks 1–3, showing that reorganizing the class space at the head level is suficient to make the detector more responsive to objects outside the known label set. This comes with some loss in known-class mAP, especially on later tasks, which is expected because the backbone and localization components are fixed while the head is adapted toward a more geometry-aware class space. Full-model CGS fine-tuning provides a lower-tradeof operating point. Compared with head-only adaptation, it preserves more of the known-class mAP while still improving U-Recall on later tasks. This suggests that when the full detector is allowed to adapt, the model can partially recover known-class discrimination while retaining the open-world benefit of geometry supervision. CGS does not simply maximize mAP, but offers a transferable way to improve unknown discovery with a controllable mAP–U-Recall tradeof. Figure 3 illustrates examples to highlight this tradeof qualitatively.

![](images/2e51904269959fb5dcb6810d16d32a7dfc8a326ab59ebac2732822bf1ba3843c.jpg)

<table><tr><td rowspan="3">Method</td><td rowspan="3">Trainable Data</td><td rowspan="3"></td><td colspan="2">Task 1</td><td colspan="4">Task 2</td><td colspan="4">Task 3</td><td colspan="4">Task 4</td></tr><tr><td rowspan="2">U-Rec.</td><td rowspan="2">mAP Curr.</td><td rowspan="2">U-Rec.</td><td colspan="4">mAP</td><td rowspan="2">U-Rec.</td><td colspan="4">mAP</td><td colspan="2">mAP</td></tr><tr><td></td><td>Prev.</td><td>Curr.</td><td>Both</td><td>Prev.</td><td>Curr.</td><td></td><td>Both</td><td>Prev.</td><td>Curr. Both</td></tr><tr><td>ORE (Joseph et al. 2021)</td><td>full</td><td>full</td><td>4.9</td><td>56.0</td><td>2.9</td><td>52.7</td><td>26.0</td><td>39.4</td><td>3.9</td><td>38.2</td><td>12.7</td><td>29.7</td><td></td><td>29.6</td><td>12.4</td><td>25.3</td></tr><tr><td>UC-OWOD (Wu et al. 2022)</td><td>full</td><td>full</td><td>75.4</td><td>56.4</td><td>68.0</td><td>33.1</td><td>30.5</td><td>46.1</td><td>53.3</td><td>28.8</td><td></td><td>16.3</td><td>28.0</td><td>25.6</td><td>15.9</td><td>26.7</td></tr><tr><td>Fast-OWDETR (Chen 2022)</td><td>full</td><td>full</td><td>9.2</td><td>56.6</td><td>8.8</td><td></td><td></td><td>39.4</td><td>7.8</td><td></td><td></td><td></td><td>32.2</td><td></td><td>1</td><td>25.0</td></tr><tr><td>OCPL (Yu et al. 2023)</td><td>full</td><td>full</td><td>8.3</td><td>56.6</td><td>7.7</td><td>50.7</td><td>27.5</td><td>39.1</td><td>11.9</td><td></td><td>38.6</td><td>14.7</td><td>30.7</td><td>30.8</td><td>14.4</td><td>26.7</td></tr><tr><td>RE-OWOD (Zhao et al. 2024)</td><td>full</td><td>full</td><td>17.9</td><td>61.3</td><td>11.8</td><td>55.6</td><td>37.9</td><td>46.6</td><td>9.9</td><td></td><td>46.5</td><td>31.3</td><td>38.9</td><td>38.9</td><td>19.8</td><td>33.9</td></tr><tr><td>OW-DETR (Gupta et al. 2022)</td><td>full</td><td>full</td><td>7.5</td><td>59.2</td><td>6.2</td><td>53.6</td><td>33.5</td><td>42.9</td><td>5.7</td><td></td><td>38.3</td><td>15.8</td><td>30.8</td><td>31.4</td><td>17.1</td><td>27.8</td></tr><tr><td>Frozen OW-DETR + CGS</td><td>head</td><td>full</td><td>11.9↑</td><td>62.2↑</td><td>9.1↑</td><td>53.3↓</td><td>19.4↓</td><td>35.6↓</td><td>9.8↑</td><td></td><td>37.4↓</td><td>16.8↑</td><td>30.5↓</td><td>30.7↓</td><td>21.4↑</td><td>28.4↑</td></tr><tr><td>OW-DETR + CGS</td><td>full</td><td>full</td><td>7.5↑</td><td>59.5↑</td><td>9.8↑</td><td>54.2↑</td><td>28.2↓</td><td>40.2↓</td><td>8.2↑</td><td></td><td>41.5↑</td><td>21.2↑</td><td>34.7个</td><td>34.5↑</td><td>24.6↑</td><td>32.0↑</td></tr></table>

Table 4: OWOD evaluation on COCO. We report U-Rec. and mAP over previously known, currently known, and all known classes. Arrows show improvement or decline relative to OW-DETR. CGS results are bolded.

Source of Class Geometry. Table 3 ablates the target geometry used by CGS. All geometry variants improve over the no-CGS baseline, showing that the pairwise relational objective helps structure the prototype space under scarce supervision. Meaningful geometry is more reliable for fewshot detection: visual geometry $D _ { \mathrm { v i s } }$ gives the best 5-shot and 10-shot mAP, while text-derived geometry and visual+text geometry also improve over the baseline but do not surpass $D _ { \mathrm { v i s } }$ . This suggests that support-crop morphology provides the strongest relational signal for fine-grained ova detection. The novel-only setting shows a diferent pattern: random D (with permutated class-distance matrix $D _ { \mathrm { v i s } } )$ performs strongly, suggesting that even arbitrary pairwise spreading can help separate newly inserted prototypes from existing classes. However, random geometry is less consistent and does not encode class relationships.

![](images/c483005ff93dda048beb9489d7dcb44def7d87d262d1dfe8cdc9d2a221a7fff7.jpg)

![](images/019f15be95363b0679b825178157e9187bcd0ee09d3939a559bd9080cfcf1b73.jpg)  
Success: More known objects recovered (ball, person, glove). Original OW-DETR Ours (CGS)  
Failure: semantic confusion and extra unknown detections.  
Figure 3: Qualitative comparison on OWOD COCO. Each row compares OW-DETR (left) and OW-DETR+CGS (right). Green boxes: known-class detections; red: unknown.

## Conclusion and Future Work

We introduced class-geometry supervision, a simple relational objective for organizing prototype and classrepresentation spaces in sample-eficient open-world detection. Across recognition, ova detection, open-set detection, novel-class insertion, and COCO OWOD adaptation, CGS improves prototype calibration and strengthens open-world behavior without requiring a new detector architecture. The results show that class geometry is especially useful for novel insertion and unknown recall, though it can introduce a known-class mAP tradeof, particularly in frozen-head adaptation. Our ablations further show that the choice of target geometry matters: visual morphology provides the most reliable gains, while random geometry can help separation but lacks consistent and interpretable structure. Future work will address the broader challenge of how to learn or update class geometry reliably as new categories arrive, especially when visual and semantic geometry disagree.

Acknowledgement. This work was supported in part by the US NSF grants IIS 2348689, IIS 2348690, IIS 2615771, and the USDA award no. 2023-69014-39716.

## References

Bertinetto, L.; Mueller, R.; Tertikas, K.; Samangooei, S.; and Lord, N. A. 2020. Making Better Mistakes: Leveraging Class Hierarchies With Deep Networks.

Carion, N.; Massa, F.; Synnaeve, G.; Usunier, N.; Kirillov, A.; and Zagoruyko, S. 2020. End-to-End Object Detection with Transformers. Lecture Notes in Computer Science, 213– 229.

Chen, X. 2022. Fast OWDETR: transformerfor open world object detection. Master’s thesis, Nanyang Technological University, Singapore.

Gemini Team, Google. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530.

Gupta, A.; Narayan, S.; Joseph, K. J.; Khan, S.; Khan, F. S.; and Shah, M. 2022. OW-DETR: Open-world Detection Transformer. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Jocher, G.; Stoken, A.; Borovec, J.; Changyu, L.; Hogan, A.; Diaconu, L.; Poznanski, J.; Yu, L.; Rai, P.; Ferriday, R.; et al. 2020. ultralytics/yolov5: v3. 0. Zenodo.

Joseph, K. J.; Khan, S.; Khan, F. S.; and Balasubramanian, V. N. 2021. Towards Open World Object Detection.

Kaul, P.; Xie, W.; and Zisserman, A. 2023. Multi-modal classifiers for open-vocabulary object detection. In International Conference on Machine Learning, 15946–15969. PMLR.

Khosla, P.; Teterwak, P.; Wang, C.; Sarna, A.; Tian, Y.; Isola, P.; Maschinot, A.; Liu, C.; and Krishnan, D. 2020. Supervised Contrastive Learning. arXiv.

Kruskal, J. B. 1964. Nonmetric Multidimensional Scaling: A Numerical Method. Psychometrika, 29: 115–129.

Liu, L.; Zhou, T.; Long, G.; Jiang, J.; Dong, X.; and Zhang, C. 2021. Isometric Propagation Network for Generalized Zero-shot Learning. arXiv.

Liu, S.; Ren, T.; Chen, J.; Zeng, Z.; Zhang, H.; Li, F.; Li, H.; Huang, J.; Su, H.; Zhu, J.; and Zhang, L. 2023. Detection Transformer with Stable Matching. arXiv:2304.04742.

Minderer, M.; Gritsenko, A.; and Houlsby, N. 2023. Scaling open-vocabulary object detection. Advances in Neural Information Processing Systems, 36: 72983–73007.

Minderer, M.; Gritsenko, A. A.; Stone, A.; Neumann, M.; Weissenborn, D.; Dosovitskiy, A.; Mahendran, A.; Arnab, A.; Dehghani, M.; Shen, Z.; Wang, X.; Zhai, X.; Kipf, T.; and Houlsby, N. 2022. Simple Open-Vocabulary Object Detection with Vision Transformers. In Computer Vision – ECCV 2022, 728–755. Springer.

Qiao, L.; Zhao, Y.; Li, Z.; Qiu, X.; Wu, J.; and Zhang, C. 2021. DeFRCN: Decoupled Faster R-CNN for Few-Shot Object Detection. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 8661–8670. IEEE.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.;

Krueger, G.; and Sutskever, I. 2021. Learning Transferable Visual Models From Natural Language Supervision. arXiv.

Schrof, F.; Kalenichenko, D.; and Philbin, J. 2015. FaceNet: A unified embedding for face recognition and clustering. 815–823.

Snell, J.; Swersky, K.; and Zemel, R. S. 2017. Prototypical Networks for Few-shot Learning. arXiv.

Song, K.; Tan, X.; Qin, T.; Lu, J.; and Liu, T.-Y. 2020. MP-Net: Masked and Permuted Pre-training for Language Understanding. arXiv.

Stevens, S.; Wu, J.; Thompson, M. J.; Campolongo, E. G.; Song, C. H.; Carlyn, D. E.; Dong, L.; Dahdul, W. M.; Stewart, C.; Berger-Wolf, T.; Chao, W.-L.; and Su, Y. 2024. BioCLIP: A Vision Foundation Model for the Tree of Life. 19412– 19424.

Sun, B.; Li, B.; Cai, S.; Yuan, Y.; and Zhang, C. 2021. FSCE: Few-Shot Object Detection via Contrastive Proposal Encoding. arXiv.

Tian, Z.; Shen, C.; Chen, H.; and He, T. 2019. FCOS: Fully Convolutional One-Stage Object Detection. In Proc. Int. Conf. Computer Vision (ICCV).

Trehan, S.; Ramachandran, U.; Rao, A.; Scimeca, R.; and Aakur, S. N. 2026. Fsp-detr: few-shot prototypical parasitic ova detection. In Proceedings ofthe IEEE/CVF Winter Conference on Applications of Computer Vision, 5342–5351.

Trehan, S.; Ramachandran, U.; Scimeca, R.; and Aakur, S. N. 2023. ProtoKD: Learning from Extremely Scarce Data for Parasite Ova Recognition. In 2023 International Conference on Machine Learning and Applications (ICMLA), 683–688. IEEE.

Wang, X.; Huang, T. E.; Darrell, T.; Gonzalez, J. E.; and Yu, F. 2020. Frustratingly Simple Few-Shot Object Detection. arXiv.

Wu, Z.; Lu, Y.; Chen, X.; Wu, Z.; Kang, L.; and Yu, J. 2022. UC-OWOD: Unknown-Classified Open World Object Detection. In Lecture Notes in Computer Science, 193–210.

Xi, X.; Huang, Y.; Lin, J.; and Luo, R. 2024. KTCN: Enhancing Open-World Object Detection with Knowledge Transfer and Class-Awareness Neutralization. In IJCAI, 1462–1470.

Yan, X.; Chen, Z.; Xu, A.; Wang, X.; Liang, X.; and Lin, L. 2019. Meta R-CNN: Towards General Solver for Instance-Level Low-Shot Learning. 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 9576–9585.

Yu, J.; Ma, L.; Li, Z.; Peng, Y.; and Xie, S. 2023. Open-World Object Detection via Discriminative Class Prototype Learning. 2022 IEEE International Conference on Image Processing (ICIP).

Zhao, X.; Ma, Y.; Wang, D.; Shen, Y.; Qiao, Y.; and Liu, X. 2024. Revisiting Open World Object Detection. IEEE Transactions on Circuits and Systems for Video Technology, 34: 3496–3509.

Zohar, O.; Wang, K.-C.; and Yeung, S. 2023. Prob: Probabilistic objectness for open world object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11444–11453.

## Appendix 1

## Extended Mathematical Analysis

## What Does Class-Geometry Supervision Control?

The proposed objective can be interpreted as aligning two complete weighted graphs over the class set. The target graph has edge weights $\hat { D } _ { i j }$ , encoding visual or semantic class dissimilarity, while the learned prototype graph has edge weights $\hat { G } _ { i j }$ , encoding distances between learned class prototypes. The following proposition shows that minimizing $\mathcal { L } _ { \mathrm { C G } }$ directly controls the average distortion between these two graphs.

Proposition 3 (Prototype-graph distortion bound). Let

$$
\mathcal { L } _ { \mathrm { C G } } = \frac { 1 } { \vert \mathcal { E } \vert } \sum _ { i < j } \left( \hat { G } _ { i j } - \hat { D } _ { i j } \right) ^ { 2 }
$$

be the class-geometry loss over the class-pair set $\mathcal { E } , I f$

$$
\mathcal { L } _ { \mathrm { C G } } \leq \epsilon ,
$$

then the average absolute distortion between the learned prototype graph and the target class-dissimilarity graph is bounded as

$$
\frac { 1 } { | \mathcal { E } | } \sum _ { i < j } \left| \hat { G } _ { i j } - \hat { D } _ { i j } \right| \leq \sqrt { \epsilon } .
$$

Proof. By Cauchy–Schwarz,

$$
\frac { 1 } { | \mathcal { E } | } \sum _ { i < j } \Big | \hat { G } _ { i j } - \hat { D } _ { i j } \Big | \leq \sqrt { \frac { 1 } { | \mathcal { E } | } \sum _ { i < j } \Big ( \hat { G } _ { i j } - \hat { D } _ { i j } \Big ) ^ { 2 } } .
$$

The term inside the square root is exactly $\mathcal { L } _ { \mathrm { C G } }$ . Therefore,

$$
\frac { 1 } { \lvert \mathcal { E } \rvert } \sum _ { i < j } \left. \hat { G } _ { i j } - \hat { D } _ { i j } \right. \leq \sqrt { \mathcal { L } _ { \mathrm { C G } } } \leq \sqrt { \epsilon } .
$$

Thus, $\mathcal { L } _ { \mathrm { C G } }$ explicitly minimizes average distortion between the learned prototype graph and the target classdissimilarity graph. This makes the objective a relational graph-alignment criterion rather than an unconstrained regularizer. While Proposition 1 controls average distortion, preserving local neighborhood topology requires suficiently small pairwise distortion for the relevant class pairs. The next proposition formalizes this margin condition.

Proposition 4 (Neighborhood-order preservation). Suppose the target class geometry indicates that class j is closer to class i than class k by margin $\gamma > 0 .$

$$
\begin{array} { r } { \hat { D } _ { i j } + \gamma \le \hat { D } _ { i k } . } \end{array}
$$

Ifthe learned prototype distances satisfy

$$
\left| \hat { G } _ { a b } - \hat { D } _ { a b } \right| \le \frac { \gamma } { 2 } f o r a l l r e l e { \nu } a n t c l a s s p a i r s ( a , b ) ,
$$

then the learned prototype space preserves this neighborhood relation:

$$
\hat { G } _ { i j } \leq \hat { G } _ { i k } .
$$

Proof. Using the assumed distortion bound,

$$
\hat { G } _ { i j } \leq \hat { D } _ { i j } + \frac { \gamma } { 2 } .
$$

Since $\hat { D } _ { i j } + \gamma \le \hat { D } _ { i k }$ , we have

$$
\hat { D } _ { i j } + \frac { \gamma } { 2 } \leq \hat { D } _ { i k } - \frac { \gamma } { 2 } .
$$

Again using the distortion bound,

$$
\hat { D } _ { i k } - \frac { \gamma } { 2 } \leq \hat { G } _ { i k } .
$$

Combining these inequalities gives

$$
\hat { G } _ { i j } \leq \hat { G } _ { i k } ,
$$

which proves the claim.

Proposition 2 shows that when the target class geometry has a suficient margin between nearby and distant classes, low pairwise distortion preserves the corresponding neighborhood order in the learned prototype space. In practice, minimizing $\mathcal { L } _ { \mathrm { C G } }$ reduces average graph distortion, while empirical analyses of distance correlations and nearest-neighbor consistency can verify whether this relational structure is preserved for the learned prototypes.

Qualitative analysis. Figure 4 shows that class-geometry supervision changes the operating behavior of the detector in a consistent way. In several cases, CGS makes the model less likely to absorb unfamiliar regions into incorrect known classes, instead either recovering the correct known category or isolating ambiguous content as unknown. This is visible in examples where a bottle is no longer rejected as unknown, a bear is no longer confused with a car, and additional sportsrelated objects are correctly recognized. However, the same behavior also produces a failure mode: the model becomes more sensitive to ambiguous regions and can over-predict unknown objects, including on parts of known objects or humans. Thus, the qualitative results mirror the quantitative tradeof in Table 4: CGS improves unknown awareness, but may also reduce precision through additional unknown detections and occasional semantic confusion. Figure 5 shows similar qualitative visualizations on the few-shot Ova detection problem.

## Supplementary Implementation Details

Prototype recognition. We train ProtoNet using the same episodic train/validation/test splits as the baseline. Each episode samples $N = 1 6$ classes with $K \in \{ 5 , 1 0 , 1 5 , 2 0 \}$ support examples per class and 5 query examples per class. The encoder is Wide Resnet, trained with Adam optimizer, learning rate $1 \times 1 0 ^ { - 5 }$ , and episode size 500 for 100 epochs. CGS is applied to the episode-level class prototypes with $\lambda = \{ 0 . 0 \dot { 5 } , 0 . 0 1 , 0 . 1 , 0 . 5 , 1 \}$ , selected on the validation split.

Ova detection. For few-shot ova detection, we use the same 15 known-class / 5 held-out-class split, image preprocessing, and support construction as FSP-DETR. DETR is first trained with the standard detection objective, after which supportderived prototypes are computed from $K \in \{ 5 , 1 0 , 1 \dot { 5 } , 2 0 \}$ examples per class. CGS is applied during the prototypematching stage with $\lambda = \{ \bar { 0 . 0 5 } , 0 . 0 1 , 0 . \bar { 1 , } 0 . 5 , \bar { 1 } \}$ . Visual dissimilarities $D _ { \mathrm { v i s } }$ are computed from cropped ground-truth support regions resized to $\overline { { 2 2 4 } } \times 2 2 4$ , using average pairwise pixel MSE between classes. For text geometry, class descriptions are generated using Google Gemini and embedded using MPNet; pairwise text dissimilarity is computed as $1 - \cos ( e _ { i } , e _ { j } )$

Open-set and novel-class insertion. Unknown rejection uses the nearest-prototype distance $m ( z ) = \operatorname* { m i n } _ { c } d ( z , p _ { c } )$ with threshold $\tau _ { \mathrm { u n k } }$ selected on the validation split. For novelclass insertion, prototypes for the held-out classes are computed from their support examples and added to the existing prototype set without retraining existing class prototypes unless otherwise stated. The expanded geometry matrix includes pairwise relationships between seen and newly inserted classes.

COCO OWOD adaptation. For COCO, we follow the OW-DETR task protocol and report results using the standard Task 1–4 evaluation format. In frozen-head adaptation, OW-DETR features are fixed and only a two-layer classification MLP is fine-tuned with CGS. In full fine-tuning, CGS is added to the standard OW-DETR training objective and all model parameters are updated. For Fig. $2 ( \mathrm { c } )$ , Task 1 uses K ∈ {5, 10, 15, 25, 50, 100, 250, 500, 750, 1000} labeled examples per class and reports relative change with respect to the full OW-DETR Task 1 reference.

Random-geometry ablation. The random-geometry baseline preserves the regularization form while removing meaningful class-pair structure. We construct $D _ { \mathrm { r a n d } }$ by randomly permuting the of-diagonal entries of $D _ { \mathrm { v i s } } ,$ then symmetrizing the matrix and setting the diagonal to zero. This preserves the marginal distribution of pairwise dissimilarities while destroying their assignment to specific class pairs.

Optimization and model selection. All thresholds, CGS weights, and normalization choices are selected using the validation split. Unless otherwise specified, pairwise distances in the prototype space use squared euclidean distance, and of-diagonal entries of D and $G$ are standardized before computing $\mathcal { L } _ { \mathrm { C G } }$ . We report the best validation-selected model for each setting and use the same evaluation protocol as the corresponding baseline.

Hardware. Experiments are run on a server with Nvidia RTXA550 GPUs. Ova recognition and detection experiments use 1 GPU with 24 GB memory. COCO OWOD adaptation and full fine-tuning experiments use 4 GPUs. Training time is approximately 6 hours for ProtoNet recognition, 25 minutes for ova detection, and 10 hours for COCO adaptation.

![](images/c0a3dc40bda6230b88985afe4abc8a64cf7f2080caa926a200c77eee3b730e95.jpg)

![](images/98d37ad78b905214575fce931606b8596bd355f44e17d38f02b3c7e2e3b1ee33.jpg)  
Bottle recovered from unknown.

![](images/7ba1475358e1d768e405f58470a222e2e119156df4df615a82e31008173dc66f.jpg)  
Bear corrected from a wrong known label.

![](images/a67e4f23d2bb23c3f628ca1fabd67d4ea819456d99a70c8c24f73c4e733364cb.jpg)

![](images/4069276aa3b44739b541d71780e1953c42d3182ef8c93db289439dd052006039.jpg)

![](images/2f47db5aa7cb3ceda9f387ab455e3965fa4151119fc379351c46e72202126cb5.jpg)  
Additional known objects recovered.

![](images/5b2eccb3d28da3cc8b4a94ac5b8a27a5317046e99d5dd6ac44b3100edfeeaf54.jpg)

![](images/cb72d6da43be764e335191489537d6b950325284e3afcd88e7d0c8f49f80baa1.jpg)

![](images/7dc0930da26063574f1807002fae7d5d6b6d1ad727c28d6578f69c3fb53cd17c.jpg)

Missed known object and more unknown boxes.  
![](images/86fb150f8bddd08241082d14d5c5cc3c60140d3d04686e5d26fdced05626810a.jpg)  
Person region partially confused as unknown.

![](images/a2b5a7d54d9367ce3109de81022226dc215928ae06c8c211f65855b7fe4a8dfc.jpg)

![](images/1246fb52688d62a040c452288b9c67eaae3c12c16c86a37befd1f65cf54f98c1.jpg)  
Semantic confusion and increased unknown detections.

Figure 4: Qualitative comparison on OWOD-COCO. For each example, the left image shows original OW-DETR predictions and the right image shows predictions after class-geometry supervision (CGS). Green boxes denote known-class detections and red boxes denote unknown detections. Examples with green borders highlight representative improvements: CGS can recover known objects previously marked as unknown, correct some misclassifications, and recover additional known objects. Examples with red borders highlight the tradeof: increased open-world sensitivity can introduce extra unknown boxes, partial person-to-unknown confusion, and occasional semantic label errors.

![](images/38481cea7e970d5c6fe8e3f7b0081ecae7981337b675e0e511613b9cf4beb761.jpg)  
Figure 5: Qualitative comparison on Ova dataset. For each example, the left image shows ground truth, the middle shows FSP-Detr predictions, and the right image shows predictions after class-geometry supervision (CGS). The red bounding box denotes the ground truth ova parasite, and the green bounding boxes denote the predictions made by FSP-Detr and FSP-Detr+CGS. The examples with green borders highlight the improvement of FSP-Detr with class geometry supervision (CGS). The red borders denote the instances where CGS did not improve FSP-Detr prediction.