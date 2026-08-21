# Open-Vocabulary 3D Object Detection with Co-Distillation Discovery and Dual Guidance Robust Training

Shangbo Yuan<sup>1,‡</sup> Jie Xu<sup>2,∗</sup> Xiaofeng Zhu<sup>3</sup> Na Zhao<sup>2</sup>

<sup>1</sup> School of Computer Science and Engineering,

University of Electronic Science and Technology of China, Chengdu, China

<sup>2</sup> Information Systems Technology and Design Pillar,

Singapore University of Technology and Design, Singapore

3 School of Computer Science and Technology, Hainan University, Haikou, China

Abstract. Recently, open-vocabulary 3D object detection (3D-OVD) has gained increasing attention for its ability to detect unseen objects in 3D scenes. Existing approaches typically adopt a two-stage pipeline that first discovers novel objects using foundation models and then trains a 3D-OVD model based on these discovered objects. Although efective, this pipeline often sufers from inaccurate localization and mismatched classification during the discovery stage, which subsequently limits the performance of the model training stage. To address these limitations, we advocate for improving both the reliability of novel object discovery and the robustness of model training, and propose an innovative framework. Specifically, for reliable discovery, our co-distillation strategy distills high-quality novel objects by applying Hungarian matching over a comprehensive score that incorporates geometric consistency, structural objectness, and semantic certainty. To enhance robust model training, we further propose a dual-guidance learning scheme, incorporating a scene-awareness-guided uncertainty regularization for the regression head and an LLM-guided hierarchical alignment for the classification head, efectively mitigating the negative efects of imprecise 3D bounding boxes and semantic ambiguity. Extensive experiments on SUN RGB-D and ScanNetV2 demonstrate that our method achieves significant performance gains over state-of-the-art approaches. Code is available at https://github.com/shangboyuan/Co-3DGT.

Keywords: Robotic perception · Open-vocabulary · 3D object detection · Co-distillation · Robust training

## 1 Introduction

Open-vocabulary object detection (OVD) [12, 45, 47] aims to localize and recognize novel objects beyond the training categories in unseen scenes by employing the unbounded vocabulary of vision-language foundation models [20, 21]. Recently, OVD has achieved remarkable progress in 2D object detection tasks, largely driven by the foundation models (e.g., CLIP [28]) pre-trained on massive 2D image-text datasets [6, 11, 13, 22]. However, extending these advances to open-vocabulary 3D object detection (3D-OVD) [8, 10, 16] remains challenging due to the intrinsic gap between 2D images and 3D point clouds [44], as well as the limited availability of large-scale annotated 3D data, which hinders the training of 3D-specific foundation models.

To achieve 3D-OVD, existing methods typically follow a two-stage pipeline: they first leverage 2D foundation models to discover novel objects in 3D scenes, and then train a 3D detector using both ground-truth base objects and discovered novel objects to transfer localization and classification knowledge to the 3D domain. Depending on how novel object discovery stage is conducted, these methods can be categorized into two main types: 2D-Detection-based and 3D-Proposal-based. As illustrated in Figure 1a, 2D-Detection-based methods [24, 39] perform open-vocabulary object detection on 2D images associated with a 3D scene. Specifically, a pre-trained OV 2D detector is utilized to identify target objects (e.g., a ‘desk’ alongside its confidence score) within the 2D frame, leveraging global 2D contexts for visual recognition. Subsequently, these methods back-project the accurately detected 2D bounding boxes (bboxes) into the 3D space to form and localize the identified 3D objects. In contrast, 3D-Proposalbased methods [3,4], shown in Figure 1b, employ a class-agnostic 3D detector to generate 3D bboxes for all object proposals, which are projected onto the image plane to obtain corresponding 2D regions of interest. Vision-language models (e.g., CLIP [28]) are then used to assign open-vocabulary class labels to the image crops of these detected 3D objects. During the training stage, both the 3D bboxes and semantic labels of the discovered novel objects are employed to train a 3D-OVD model, where 3D bboxes are regressed for localization and semantic label features from CLIP are aligned for classification.

Despite the efectiveness of the two-stage pipeline, its performance is still limited by noise in the discovered novel objects, which negatively impacts the subsequent training stage that relies on them as supervision. While both 2D-Detection-based and 3D-Proposal-based methods inevitably introduce noise during the discovery stage, they sufer from two diferent major types of errors. Specifically, the 2D-Detection-based strategy often yields inaccurate 3D bboxes because the noisy back-projection process encapsulates erroneous depth measurements and extraneous points from adjacent objects into the resulting 3D point set. Thus, calculating a 3D bbox from such a noisy 3D point set leads to imprecise localization. For example, in Figure 1a, the back-projected bbox of the desk is largely imprecise because its underlying point set includes noisy and unassociated points. On the other hand, 3D-Proposal-based approaches may assign incorrect semantic labels to 3D bboxes due to occlution, as shown in Figure 1b, the desk is partially occluded by the chair, causing the cropped image of the discovered proposal to be incorrectly classified as a chair by the CLIP model. Consequently, the discovery process directly bottlenecks the subsequent training stage, which relies on the bbox parameters and semantic labels from discovered novel objects for model optimization.

![](images/94b74f183abbbab24712b71064ba430969021e1a2a5a06a44c8e8802a4969651.jpg)  
Fig. 1: Comparison between our Co-3DGT and two existing 3D-OVD paradigms. In the Novel Object Discovery stage: (a) 2D-Detection-based strategy [24, 39] back-projects 2D-OVD results into 3D space, often leading to imprecise 3D bounding boxes; (b) 3D-Proposal-based methods [3,4] project 3D proposals from classagnostic 3D detectors onto 2D plane and crop images for open-vocabulary labeling, which may introduce semantic ambiguity; (c) We propose Co-Distillation, jointly considering consider geometric constraints alongside 2D semantic and 3D localization inofrmation for more reliable novel objects. In the Training stage: (d) Previous methods typically employ equal supervision for both base and novel objects; while (e) We propose Dual Guidance Robust Training, integrating uncertainty regularization for the regression head and leveraging an LLM to achieve hierarchical semantic alignment.

Moreover, the inherent errors stemming from both 2D and 3D detectors remain largely unavoidable in practical open-vocabulary scenarios. Specifically, 2D OVDs can also introduce semantic misclassifications (i.e., classification noise) [34], while class-agnostic 3D detectors inevitably yield imprecise spatial localization (i.e., bounding box noise) [7]. However, during the subsequent training stage, existing methods typically treat the high-quality ground-truth annotations of base objects and the noisy pseudo-labels of discovered novel objects with equal supervision as presented in Figure 1d. Without accounting for the severe quality disparity between these two sources of annotation, localization accuracy of the model is impaired by the imprecise bboxes, and semantic errors from the 2D models are inadvertently propagated into the trained model. This indiscriminate training strategy consequently degrades the feature representations for both base and novel categories, leading to suboptimal detection performance.

To address the above challenges, we propose open-vocabulary 3D object detection with Co-Distillation Discovery and Dual Guidance Training (Co-3DGT), a unified framework that enhances both the reliability of novel 3D object discovery and the robustness of model training. As illustrated in Figure 1c, our co-distillation discovery module jointly leverages a 2D open-vocabulary detector and a class-agnostic 3D detector to discover novel objects from both spatial and semantic perspectives in parallel. Specifically, we measure the spatial IoU scores between semantic 2D proposals and class-agnostic 3D proposals in a scene, and then combine the 2D class confidence and 3D objectness scores to jointly solve a scene-level matching problem. This objective expects to discover the spatial consistency among 3D boxes with semantic constraints, to co-distill reliable 3D novel objects.

To mitigate the impact of inevitable noise in discovered objects, we introduce the dual-guidance training strategy as shown in Figure 1e, which comprises (i) scene-awareness-guided uncertainty regularization and (ii) large language model (LLM)-guided hierarchical alignment. Concretely, (i) we exploit the scene-aware uncertainty from base classes to regularize the uncertainties for novel classes, thus dynamically weighting the 3D bbox regression loss. This prevents the model from becoming overconfident in noise 3D bboxes, thereby ensuring stable and robust localization performance. Meanwhile, (ii) we propose an LLM-guided hierarchical labeling to enrich label supervision instead of directly using given category labels. It derives two levels of super-category labels from the original category names via an LLM to capture shared semantic and geometric properties among related classes. Then, hierarchical semantic alignment aligns 3D object features with CLIP embeddings across multiple semantic levels, making the semantic classification less sensitive to noise labels. This increases more discriminative supervision from super-categories to the original supervision, and thus helps the model to achieve robust classification.

In summary, our work makes four contributions as follows:

– We identify the performance bottleneck in the widely adopted two-stage 3D-OVD pipeline and propose a unified framework that jointly reduces noise in novel object discovery and enhances training robustness.

– We propose a Co-Distillation Discovery strategy that enforces 3D-2D consistency via a joint optimization formulation that leverages spatial and semantic scores to selectively distill reliable novel 3D objects.

– We introduce a Dual Guidance Robust Training scheme consisting of sceneawareness-guided uncertainty regularization for robust uncertainty-weighted regression, and LLM-guided hierarchical alignment for improved classification via label-augmented semantic feature alignment.

– Extensive experiments on SUN RGB-D and ScanNetV2 demonstrate that our method achieves state-of-the-art performance, with accuracy improvements of 4.71% and 9.82% $\mathrm { A P _ { 2 5 } }$ in novel-class object detection, respectively.

## 2 Related Work

3D Object Detection focuses on identifying and localizing objects in 3D space from point clouds or RGB-D data [14, 31, 40, 48, 49]. VoteNet [26] introduce an end-to-end point-based framework using Hough voting to generate anchorfree object proposals. Specifically, it employs PointNet++ [27] to extract point cloud features, where seed points vote for object centers. 3DETR [25] adopts a Transformer architecture with self-attention for global context modeling, effectively eliminating the need for hierarchical aggregation and heuristic voting schemes. Instead, it uses learnable queries and a Hungarian matching loss for direct, end-to-end detection. More recently, Cubify-Anything [19] introduced the

Cubify Transformer (CuTR) model alongside the Cubify-Anything 1M (CA-1M) dataset, further advancing indoor 3D object detection by leveraging Vision Transformer (ViT)-based [9] architectures.

Open-Vocabulary Object Detection seeks to generalize object detections beyond their seen training categories to recognize and localize previously unseen object classes by leveraging the powerful representational capabilities of vision-language pretraining models [1,15,36,37]. The seminal work of CLIP [28] establishes the foundational paradigm for this research field by learning semantically aligned vision-language representations through contrastive learning on web-scale image-text pairs. Based on this, Detic [50] integrate zero-shot recognition into detectors like Faster R-CNN [29] by replacing the fixed-vocabulary classification heads with CLIP-based classifiers that can dynamically accommodate arbitrary text object categories during inference without requiring additional fine-tuning. Grounding DINO [23] achieves tighter vision-language integration by unifying detection and phrase grounding via language-guided query mechanisms, enabling the model to learn robust associations between textual descriptions and visual regions without requiring expensive box-level supervision annotations during the pretraining phase.

Open-Vocabulary 3D Object Detection has emerged to overcome the scalability limitations of traditional closed-set detectors, which rely on exhaustive and costly 3D annotations [2, 35, 41, 51, 52]. OV-3DET [24] pioneers 3D-OVD without requiring 3D annotations by leveraging pretrained 2D detectors for semantic transfer. CoDA [3] introduces a collaborative framework that integrates novel object discovery with cross-modal alignment, which jointly exploits 3D geometry to discover potential objects and 2D semantics for their identification, enhancing the model’s ability to distinguish unknown from known instances. Addressing the challenge of semantic ambiguity, INHA [17] advances this by seeding 3D novel object proposals from 2D OVD results and introducing a multi-level hierarchical alignment across instance, category, and scene contexts to reduce semantic inconsistency. More recently, OV-Uni3DETR [39] extends open-vocabulary detection to multimodal and outdoor environments, unifying representations across point clouds and RGB images within a transformer-based architecture, demonstrating robust performance and flexibility for various modality combinations.

## 3 Method

Problem Definition. A 3D scene is represented by point cloud $\mathcal { P } = \{ \mathbf { P } \in \mathcal { C } \}$ $\mathbb { R } ^ { N \times 3 } \}$ and corresponding RGB-D images $( I ^ { \mathrm { r g b } } \in \mathring { \mathbb { R } } ^ { 3 \times \hat { H } \times W } , I ^ { \mathrm { d e p t h } } \in \mathring { \mathbb { R } } ^ { 1 \times \hat { H } \times W } )$ with known camera intrinsics $\textbf { K } \in \ \mathbb { R } ^ { \bar { 3 } \times 3 }$ , rotation and translation extrinsics $( \mathbf { R } \in \mathbb { R } ^ { 3 \times 3 } , \mathbf { t } \in \mathbb { R } ^ { 3 } )$ . Each 3D object in the scene is parameterized as $( \mathbf { b } , c )$ where b denotes the bbox defined as $\mathbf { b } = ( x , y , z , l , w , h , \theta )$ . Here, $( x , y , z )$ is the center coordinate, $( l , w , h )$ are the dimensions, and θ is the orientation around the vertical axis. The category label is $c \in { \mathcal { C } } .$ , where C is the set of object categories.

Unlike closed-set 3D object detection, open-vocabulary 3D object detection is only provided with base objects $\mathcal { O } ^ { \mathrm { b a s e } } = \{ { \bf b } _ { i } , c _ { i } \ | \ c _ { i } \in \mathcal { C } ^ { \mathrm { b a s e } } \}$ , while the goal is to detect and recognize objects from the combined category set $\mathcal { O } = \mathcal { O } ^ { \mathrm { b a s e } } \cup \mathcal { O } ^ { \mathrm { n o v e l } }$ To achieve this, a discovery stage is employed to identify potential novel objects $\mathcal { O } ^ { \mathrm { n o v e l } } = \{ \mathbf { b } _ { j } , c _ { j } \ | \ c _ { j } \in \mathcal { C } ^ { \mathrm { n o v e l } } \}$ within the scene. These discovered instances serve as additional pseudo-labels that expand the category coverage beyond C<sup>base</sup>. In the training stage, both the annotated base objects O<sup>base</sup> and the discovered novel objects O<sup>novel</sup> are jointly utilized as supervision. This enables the model to localize and recognize both base and novel objects.

![](images/df93e5399b972333a861d46361431214853fcd7312348a84fc6f829226b85622.jpg)  
Fig. 2: Framework Overview of our Co-3DGT. In Co-Distillation Discovery stage: (a) a class-agnostic 3D detector generates 3D boxes, while a 2D open-vocabulary detector produces class-aware 2D boxes which are back-projected and to co-distill reliable class-aware 3D boxes. In Dual Guidance Robust Training stage: (b) Scene-Awareness-Guided Uncertainty Regularization exploits the scene-aware uncertainty from base classes to regularize the uncertainties for novel classes, dynamically weighting the 3D bbox regression. (c) LLM-Guided Hierarchical Alignment leverage LLM to form super-categories that increase structural semantic supervision beyond original categories, promoting the model to achieve robust classification.

Framework Overview. As illustrated in Figure 2, we propose Co-3DGT to enhance the reliability of novel object discovery and the robustness of model training. Our framework consists of two stages. In the novel object discovery stage, we propose a scene-level matching problem based co-distillation discovery strategy (Sec. 3.1) that can jointly distill high-quality novel objects. In the model training stage, we propose a dual guidance robust training scheme, where a scene-awareness-guided uncertainty regularization (Sec. 3.2) and an LLM-guided hierarchical alignment (Sec. 3.3) are designed to mitigate negative efects of the unavoidable imprecise 3D bboxes and semantic ambiguity, respectively.

## 3.1 Co-Distillation Discovery

We observe that the 2D-Detection-based discovery exhibits stronger performance in capturing semantically accurate object categories but sufers from producing noise 3D bboxes, while the 3D-Proposal-based discovery is conducive to predicting precise 3D bboxes but is prone to yielding noise semantic labels. Motivated by this, we propose a co-distillation discovery approach to avoid the drawbacks of previous two pipelines, and to fully exploit the respective advantages of 3D localization and 2D semantic recognition. That is, as shown in Figure 2a, we first obtain semantically labeled 2D bounding boxes by leveraging the rich categorical knowledge of open-vocabulary 2D detectors, alongside class-agnostic 3D object proposals generated in parallel. We then introduce a bipartite matching problem, to jointly distill high-quality novel object annotations from both 2D semantic and 3D geometric perspectives.

Specifically, to lift 2D detections to 3D space, we follow the back-projection operation used in previous methods [24,39]: $P _ { k } = ( K R _ { t } ) ^ { - 1 } \cdot ( d _ { k } \cdot { \bf x } _ { k } )$ . The pixel set $\{ \mathbf { x } _ { k } \} = \{ [ u _ { k } , v _ { k } , 1 ] ^ { \top } \}$ within a 2D OVD result is back-projected to obtain the 3D point set $\{ P _ { k } \}$ using the camera intrinsic matrix $K ,$ , camera extrinsics $R _ { t } ,$ and the corresponding depth value $\{ d _ { k } \}$ . The resulting 3D bbox is then defined as the minimal cuboid enclosing the 3D point set $\{ P _ { k } \}$

Following this, let $\mathcal { O } ^ { \mathrm { 2 D } } = \{ ( \mathbf { b } _ { i } ^ { \mathrm { 2 D } } , c _ { i } ^ { \mathrm { 2 D } } , s _ { i } ^ { \mathrm { 2 D } } ) \} _ { i = 1 } ^ { M }$ denote the set of back-projected 3D objects derived from the 2D OVD results across all views. Here, $\mathbf { b } _ { i } ^ { \mathrm { 2 D } }$ represents the i-th 3D bbox lifted from the 2D OVD result, while $c _ { i } ^ { \mathrm { 2 D } }$ and $s _ { i } ^ { \mathrm { 2 D } }$ indicate its predicted semantic category label and associated confidence score, respectively. Similarly, let $\mathcal { O } ^ { \mathrm { 3 D } } = \{ ( \mathbf { \bar { b } } _ { j } ^ { \mathrm { 3 D } } , s _ { j } ^ { \mathrm { 3 D } } ) \} _ { j = 1 } ^ { N }$ represent the set of class-agnostic 3D proposals by the class-agnostic detector, where for the j-th proposal, $\mathbf { b } _ { j } ^ { \mathrm { 3 D } }$ denotes the 3D bbox and $s _ { j } ^ { \mathrm { 3 D } }$ is the foreground probability.

To achieve co-distillation discovery, we formulate the association between M 2D bounding boxes and N 3D proposals as a bipartite matching problem. To solve this matching problem via the Hungarian algorithm, we define the cost matrix $\mathbf { C } \in \mathbb { R } ^ { M \times N }$ as:

$$
\begin{array} { r } { C _ { i j } = - \mathrm { I o U } _ { i j } - 0 . 1 ( s _ { j } ^ { \mathrm { 3 D } } + s _ { i } ^ { \mathrm { 2 D } } ) . } \end{array}\tag{1}
$$

Here, Io $\operatorname { J } _ { i j }$ denotes the Intersection over Union (IoU) between the i-th 3D bbox lifted from the 2D OVD result and the j-th class-agnostic 3D proposal. To filter out geometrically unfeasible pairs, we establish a minimal IoU threshold $\tau = 0 . 1$ . Specifically, if $\mathrm { I o U } _ { i j } < \tau$ , we prevent their assignment by setting the corresponding cost $C _ { i j } = \infty$

For candidates that satisfy this minimal IoU, their exact assignment is driven by minimizing the cost function. The $\mathrm { I o U } _ { i j }$ term serves as the primary geometric alignment metric, penalizing size mismatches and location misalignments, thereby encouraging tight spatial correspondence between cross-modal candidates. The 3D foreground probability $s _ { j } ^ { \mathrm { 3 D } }$ and the 2D confidence score $s _ { i } ^ { \mathrm { 2 D } }$ quantify the spatial objectness of the 3D proposal and the semantic reliability of the 2D prediction, respectively. To preserve $\mathrm { I o U } _ { i j }$ as the primary selection criterion, we apply a weight coeficient of 0.1 to both confidence scores. Together, this composite cost drives the assignment of cross-modal pairs that are simultaneously geometrically consistent and semantically reliable, facilitating high-quality transfer of categorical knowledge from the 2D domain to the 3D representation.

By minimizing the total cost via the Hungarian algorithm, we obtain the optimal assignment set H of the matched pairs $( i , j )$ . Consequently, our codistillation strategy discovers and extracts more reliable novel objects to form

the final distilled set as follows:

$$
\mathcal { O } _ { \mathrm { C o - D } } ^ { \mathrm { n o v e l } } = \left\{ \left( \mathbf { b } _ { j } ^ { \mathrm { 3 D } } , c _ { i } ^ { \mathrm { 2 D } } \right) \vert \left( i , j \right) \in \mathcal { H } \right\} .\tag{2}
$$

Here, $\mathbf { b } _ { j } ^ { \mathrm { 3 D } }$ denotes the bbox parameters of the selected 3D proposal, and $c _ { i } ^ { \mathrm { 2 D } }$ represents the open-vocabulary semantic category transferred from its paired 2D detection. By pairing high-quality 3D geometry with holistically inferred 2D semantics, this formulation efectively constructs a more reliable pseudo-label set for novel classes, facilitating the subsequent learning process.

Given the base and discovered novel objects $\mathcal { O } = \mathcal { O } ^ { \mathrm { b a s e } } \cup \mathcal { O } _ { \mathrm { C o - D } } ^ { \mathrm { n o v e l } }$ , we train a 3D detector to transfer the geometric localization and semantic classification knowledge to the 3D-OVD model. Considering that both the class-agnostic 3D detector and the 2D open-vocabulary detector might introduce inevitable error, we propose the Dual Guidance Robust Training scheme to achieve the robust 3D-OVD model with imperfect localization and classification supervision. It consists of Scene-Awareness-Guided Uncertainty Regularization and LLM-Guided Hierarchical Alignment that will be introduced in following sections.

## 3.2 Scene-Awareness-Guided Uncertainty Regularization

Inspired by the uncertainty estimation in deep learning [18, 30], we employ the uncertainty-weighted regression loss to train the model as shown in Figure 2b. Specifically, the 3D bbox supervision for base objects (with manual annotation) and novel objects (from discovery stage) are derived from diferent processes, and they usually exhibit diferent noise levels. To this end, we treat base and novel objects as two distinct supervision sources during training and design specialized regression loss functions for each. For base objects, the bboxes $\mathbf { b } _ { i } \in \mathcal { O } ^ { \mathrm { b a s e } }$ are manually annotated, we adopt the following regression loss [30]:

$$
\mathcal { L } _ { \mathrm { R e g } } ^ { \mathrm { b a s e } } = \sum _ { i } \left\lfloor \frac { \sigma _ { i } } { \sqrt { 2 } } \right\rfloor ^ { \frac { 1 } { 2 } } \left( \frac { \sqrt { 2 } } { \sigma _ { i } } , \left| \hat { \mathbf { b } } _ { i } - \mathbf { b } _ { i } \right| + \log \sigma _ { i } \right) ,\tag{3}
$$

where $\hat { \mathbf { b } } _ { i }$ is the i-th prediction of the 3D bbox of the model for the object $\mathbf { b } _ { i } \in \mathcal { O } ^ { \mathrm { b a s e } }$ , and $\sigma _ { i }$ is the estimated uncertainty for the i-th object obtained by neural networks [18]. ⌊·⌋ denotes the stop-gradient operation.

In this formulation, the regression loss is driven by three key components. We first employ two complementary weighting terms, where the uncertainty weighting term ${ \sqrt { 2 } } / { \sigma _ { i } }$ enables the model to focus on reliable bboxes with low $\sigma _ { i } .$ and the adaptive re-weighting term $\left\lfloor \sigma _ { i } / { \sqrt { 2 } } \right\rfloor ^ { \frac { 1 } { 2 } }$ prevents the model from ignoring dificult samples. Next, log $\sigma _ { i }$ acts as a regularization term, which explicitly penalizes the network for predicting an arbitrarily large uncertainty $\sigma _ { i }$ to reduce loss. Finally, the interplay among these three components efectively maintains balanced supervision across varying uncertainty levels for base boxes.

For novel objects, the bboxes $\mathbf { b } _ { j } \in \mathcal { O } _ { \mathrm { C o - D } } ^ { \mathrm { n o v e l } }$ are generated by the class-agnostic 3D detector, and consequently contain additional localization errors. The uncertainty term log $\sigma _ { i }$ in Eq. (3) is learned primarily from precise manual annotations of base objects. As a result, the learned uncertainty estimates may not adequately capture the elevated noise level of novel bboxes. To address this issue, we improve Eq. (3) and propose the scene-awareness-guided uncertainty regularization as follows:

$$
\mathcal { L } _ { \mathrm { R e g } } ^ { \mathrm { n o v e l } } = \sum _ { j } \Biggl ( \frac { \sqrt { 2 } } { \sigma _ { j } } \bigl | \hat { \mathbf { b } } _ { j } - \mathbf { b } _ { j } \bigr | \bigl | \log \sigma _ { j } - \mathbb { E } \left[ \bigl \lfloor \log \sigma _ { i } \bigr \rfloor \right] \bigr | \Biggr ) ,\tag{4}
$$

where $\mathbb { E } \left[ \left\lfloor \log { \sigma _ { i } } \right\rfloor \right]$ represents the scene-aware inherent uncertainty predicted by the model, derived from the expectation of the stop-gradient log-uncertainty of base bboxes. To handle scenes without base objects, we maintain a moving average $\mathbb { E } ^ { \prime }$ term updated using the expectations from base-containing scenes via a standard momentum update (decay rate of 0.9) during training.

Here, the term $\left| \log \sigma _ { j } - \mathbb { E } [ \left\lfloor \log \sigma _ { i } \right\rfloor ] \right|$ serves as a soft regularization that guides the uncertainty of novel objects toward the scene-aware uncertainty. The motivation of our regression loss is that we hope the bbox uncertainty of novel objects remains at the same level as that of base objects $\mathbb { E } [ \lfloor \log \sigma _ { \mathrm { b a s e } } ] ]$ , which can be supported by the classical theory of maximum mean discrepancy (MMD). This prevents both underestimation (when log $\sigma _ { j } \leq \mathbb { E } [ \lfloor \log { \sigma _ { i } } \rfloor ] )$ and unbounded overestimation (when log $\sigma _ { j } > \mathbb { E } [ \lfloor \log { \sigma _ { i } } \rfloor ] )$ ).Notably, we omit the adaptive reweighting term $\lfloor \sigma / { \sqrt { 2 } } \rfloor ^ { \frac { 1 } { 2 } }$ from Eq. (3) since applying this term to novel bboxes with higher noise levels could amplify erroneous signals.

Overall, the 3D bbox regression loss with our scene-awareness-guided uncertainty regularization is:

$$
\mathcal { L } _ { \mathrm { R e g } } = \mathcal { L } _ { \mathrm { R e g } } ^ { \mathrm { b a s e } } + \mathcal { L } _ { \mathrm { R e g } } ^ { \mathrm { n o v e l } } .\tag{5}
$$

This scene-awareness-guided uncertainty regularization leverages scene-aware uncertainty estimates from base objects as an anchor to robustly guide the uncertainty prediction for novel objects within the same scene.

## 3.3 LLM-Guided Hierarchical Alignment

To achieve the robustness against inaccurate semantic information form the given category labels, we first propose an LLM-guided hierarchical labeling to enrich label supervision, and then conduct the hierarchical semantic alignment to achieve the reliable classification as shown in Figure 2c.

LLM-Guided Hierarchical Labeling. To mitigate the possible noise supervision from the given category labels, we employ an LLM to infer functional, geometric, and conceptual relationships among base and novel categories, to obtain the label-augmented supervision. Specifically, we define three hierarchies for each category label, i.e., top, mid, bottom. The bottom level contains the original fine-grained category labels $\mathcal { C } ^ { \mathrm { b o t } } = \mathcal { C } ^ { \mathrm { b a s e } } \cup \mathcal { C } ^ { \mathrm { n o v e l } }$ , while the top and mid levels C<sup>mid</sup>, C<sup>top</sup> are super-categories inferred by the LLM. For example, hierarchical

labels from $\mathcal { C } ^ { \mathrm { b o t } }$ is constructed as:

$$
\left\{ \begin{array} { l l } { { \mathcal { C } ^ { \mathrm { t o p } } : \mathrm { { f u r n i t u r e } } , \mathrm { { \cdot } a p p l i a n c e } , \mathrm { { \cdot } a c c e s s o r y } ^ { \mathrm { , } } , \ldots } } \\ { { \mathcal { C } ^ { \mathrm { m i d } } : \mathrm { { \ s e a t i n g } } : \mathrm { { s t o r a g e } } ^ { \mathrm { , } } : \mathrm { { c o m e s t i c } } ^ { \mathrm { , } } \mathrm { { \cdot } t o o l s } ^ { \mathrm { , } } , \mathrm { { \cdot } . . . } } } \\ { { \mathcal { C } ^ { \mathrm { b o t } } : \mathrm { { \cdot } c h a i r } ^ { \mathrm { , } } : \mathrm { { t a b l e } } ^ { \mathrm { , } } , \mathrm { { \cdot } p i l l o w } ^ { \mathrm { , } } , \mathrm { { \cdot } d o o r } ^ { \mathrm { , } } , \mathrm { { \cdot } f r i d g e } ^ { \mathrm { , } } , \ldots } } \end{array} \right.\tag{6}
$$

In our method, the mid level corresponds to functionally and geometrically similar super-categories, and the top level represents abstract ones. The top and mid levels are less sensitive to semantic ambiguity from visually similar objects, which provides structural information for subsequent semantic alignment.

Hierarchical Semantic Alignment. For the j-th semantic label, we use CLIP to encode its corresponding hierarchical labels $c _ { j } ^ { \mathrm { t o p } } , \ c _ { j } ^ { \mathrm { m i d } } , \ c _ { j } ^ { \mathrm { b o t } }$ to obtain the textual embeddings ${ \bf h } _ { j } ^ { ( 1 ) } , { \bf h } _ { j } ^ { ( 2 ) } , { \bf h } _ { j } ^ { ( 3 ) }$ , respectively. These embeddings are subsequently aligned with the $\mathrm { 3 D \mathrm { - } O V D }$ model which also produce hierarchical semantic features $\mathbf { f } _ { i } ^ { ( 1 ) } , \mathbf { f } _ { i } ^ { ( 2 ) } , \mathbf { f } _ { i } ^ { ( 3 ) }$

To obtain the predicted probability that the i-th sample belongs to the j-th category at level l, we first compute the cosine similarity score and then map it to the interval (0,1) using a sigmoid function:

$$
p _ { i j } ^ { ( l ) } = \mathrm { s i g m o i d } \left( \frac { \langle \mathbf { f } _ { i } ^ { ( l ) } , \mathbf { h } _ { j } ^ { ( l ) } \rangle } { \| \mathbf { f } _ { i } ^ { ( l ) } \| \| \mathbf { h } _ { j } ^ { ( l ) } \| } \right) ,\tag{7}
$$

and then adopt binary cross-entropy (BCE) loss:

$$
\mathcal { L } _ { \mathrm { b c e } } ^ { ( l ) } = - \frac { 1 } { N } \sum _ { j } ^ { | \mathcal { C } ^ { ( l ) } | } \sum _ { i } ^ { N } \left[ y _ { i j } ^ { ( l ) } \cdot \log ( p _ { i j } ^ { ( l ) } ) + ( 1 - y _ { i j } ^ { ( l ) } ) \cdot \log ( 1 - p _ { i j } ^ { ( l ) } ) \right]\tag{8}
$$

where $y _ { i j } ^ { ( l ) }$ denotes the indicator variable, $\mathrm { e . g . , } y _ { i j } ^ { ( l ) } = 1$ if the i-th object belongs to the j-th category at the l-th level, otherwise $\bar { y } _ { i j } ^ { ( l ) } = 0$

The overall hierarchical alignment loss is formulated as:

$$
\mathcal { L } _ { \mathrm { A l i g n } } = \mathcal { L } _ { \mathrm { b c e } } ^ { ( 1 ) } + \mathcal { L } _ { \mathrm { b c e } } ^ { ( 2 ) } + \mathcal { L } _ { \mathrm { b c e } } ^ { ( 3 ) } .\tag{9}
$$

In the hierarchical semantic alignment, the bottom level preserves fine-grained category distinctions and supervises the learning of detailed object features, while the mid and top levels mitigate the impact of noise labels by improving robustness against visually similar or ambiguous objects.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We evaluate our proposed method on two challenging large-scale indoor 3D object detection datasets: SUN RGB-D [33] and ScanNetV2 [5]. The

SUN RGB-D dataset contains about 10,335 RGB-D scenes across diverse indoor environments. Following common practice, we use 5,285 training samples covering 46 object categories, where the 10 most frequent categories are treated as base classes and the remaining 36 as novel classes. The ScanNetV2 dataset [5] consists of more than 1500 indoor scenes with 200 object categories. From the 1,201 training scenes in ScanNetV2, we select the 10 most frequent classes as base classes and the next 50 as novel classes. In addition, following OV-3DET [24], we adopt a setting where no ground-truth annotations are provided, and evaluate the top 20 most frequent unseen classes on ScanNetV2.

Baselines. We compare our method with a series of 3D-OVD methods that take point clouds as input. These baselines include Det-PointCLIP [43], Det-PointCLIPv2 [42], Det-CLIP<sup>2</sup> [46], OV-3DET [24], CoDA [3], INHA [17], Co-DAv2 [4], and OV-Uni3DETR [39].

Evaluation Metrics. We adopt the mean Average Precision (mAP) and mean Average Recall (mAR) at an IoU threshold of 0.25 for evaluation. We report the mAP for unseen classes, seen classes, and all classes, denoted as $\mathrm { A P _ { 2 5 } ^ { n o v e l } }$ $\mathrm { A P _ { 2 5 } ^ { b a s e } }$ , and $\mathrm { A P _ { 2 5 } ^ { m e a n } }$ , respectively. Similarly, the corresponding mAR are denoted as $\bar { \mathrm { A R } } _ { 2 5 } ^ { \mathrm { n o v e l } } , \mathrm { A R } _ { 2 5 } ^ { \mathrm { b a s e } }$ , and $\mathrm { A R _ { 2 5 } ^ { m e a n } }$ . In addition, under the same setting with no ground-truth annotations following OV-3DET, we adopt the $\mathrm { A P _ { 2 5 } }$ as the metric for the top 20 most frequent unseen classes.

Implementation Details. We adopt the backbone of Uni3DETR [38], including its voxel encoder and transformer decoder, as our 3D detection model. In line with common approaches [3,17,24,39,43,46], we utilize CLIP [28] to transform category labels into textual features. Following previous work [17, 24, 39], we leverage Detic as the 2D open-vocabulary detector. Additionally, we employ CuTR [19] as the 3D class-agnostic detector and ChatGPT-5 [32] to infer category relationships and generate hierarchical labels. In terms of computational cost, our full pipeline requires approximately 47 minutes and 169 minutes for novel object discovery on SUN RGB-D and ScanNetV2, respectively, using a single RTX 3090 GPU. The LLM-guided hierarchical alignment introduces negligible time cost, requiring only a single API call for the entire dataset (approximately 20 ∼ 30 seconds) to generate the two-level hierarchical labels. Although the discovery stage is moderately slower than OV-Uni3DETR (25 minutes and 103 minutes with single RTX 3090 GPU on SUN RGB-D and ScanNetV2, respectively), our approach substantially reduces the overall training time. Using four RTX 3090 GPUs, the training stage takes 490 and 309 minutes on SUN RGB-D and ScanNetV2, respectively (compared to 716 and 621 minutes for OV-Uni3DETR), since our model operates only on point cloud inputs, in contrast to OV-Uni3DETR which integrates both point cloud and image modalities.

## 4.2 Main Results

As shown in Table 1, our method achieves the highest overall performance on both SUN RGB-D and ScanNetV2 benchmarks. Notably, our model significantly improves detection of novel categories, boosting $\mathrm { A P _ { 2 5 } ^ { n o v e l } }$ to 14.37% on SUN RGB-D and 21.91% on ScanNetV2, which corresponds to considerable gains of +4.71% and +9.82% over the previous state-of-the-art approach (OV-Uni3DETR). In comparison, the improvements on base categories are more modest, with increases of +1.56% on SUN RGB-D and +2.36% on ScanNetV2. Overall, our model achieves a $\mathrm { A P _ { 2 5 } ^ { m e a n } }$ of 22.63% on SUN RGB-D and 23.75% on ScanNetV2, surpassing OV-Uni3DETR by +4.57% and +8.60%, respectively. These results demonstrate that our approach efectively enhances robustness to unseen categories, while maintaining strong performance on base categories.

Table 1: Comparison of $\mathrm { A P _ { 2 5 } }$ (%) and $\mathrm { { A R } _ { 2 5 } }$ (%) for open-vocabulary 3D object detection on SUN RGB-D and ScanNetV2 datasets.
<table><tr><td rowspan="2">Method</td><td colspan="6">SUN RGB-D</td><td colspan="6">ScanNetV2</td></tr><tr><td> $\mathrm { \Big | A P _ { 2 5 } ^ { n o v e l } ~ A P _ { 2 5 } ^ { b a s e } ~ A P _ { 2 5 } ^ { m e a n } ~ A R _ { 2 5 } ^ { n o v e l } ~ A R _ { 2 5 } ^ { b a s e } ~ A R _ { 2 5 } ^ { m e a n } \Big | A P _ { 2 5 } ^ { n o v e l } ~ A P _ { 2 5 } ^ { b a s e } ~ A P _ { 2 5 } ^ { m e a n } ~ A P _ { 2 5 } ^ { n o v e l } ~ A R _ { 2 5 } ^ { b a s e } ~ A R _ { 2 5 } ^ { b a s e } ~ A R _ { 2 5 } ^ { m e a n } ~ A R _ { 2 5 } ^ { b a s e } ~ A R _ { 2 5 } ^ { m e a n } ~ A R _ { 2 5 } ^ { b a s e } ~ }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Det-PointCLIP [43]</td><td>0.09</td><td>5.04</td><td>1.17</td><td>21.98</td><td>65.03</td><td>31.33</td><td>0.13</td><td>2.38</td><td>0.50</td><td>33.38</td><td>54.88</td><td>36.96</td></tr><tr><td>Det-PointCLIPv2 [42]</td><td>0.12</td><td>4.82</td><td>1.14</td><td>21.33</td><td>63.74</td><td>30.55</td><td>0.13</td><td>1.75</td><td>0.40</td><td>32.60</td><td>54.52</td><td>36.25</td></tr><tr><td>Det-CLIP² [46]</td><td>0.88</td><td>22.74</td><td>5.63</td><td>22.21</td><td>65.04</td><td>31.52</td><td>0.14</td><td>1.76</td><td>0.40</td><td>34.26</td><td>56.22</td><td>37.92</td></tr><tr><td>3D-CLIP [28]</td><td>3.61</td><td>30.56</td><td>9.47</td><td>21.47</td><td>63.74</td><td>30.66</td><td>3.74</td><td>14.14</td><td>5.47</td><td>32.15</td><td>54.15</td><td>35.81</td></tr><tr><td>CoDA [3]</td><td>6.71</td><td>38.72</td><td>13.66</td><td>33.66</td><td>66.42</td><td>40.78</td><td>6.54</td><td>21.57</td><td>9.04</td><td>43.36</td><td>61.00</td><td>46.30</td></tr><tr><td>INHA [17]</td><td>8.91</td><td>42.17</td><td>16.18</td><td>51.34</td><td>78.65</td><td>57.23</td><td>7.79</td><td>25.10</td><td>10.68</td><td>55.10</td><td>71.60</td><td>57.85</td></tr><tr><td>CoDAv2 [4]</td><td>9.17</td><td>42.04</td><td>16.31</td><td>43.16</td><td>71.64</td><td>49.35</td><td>9.12</td><td>23.35</td><td>11.49</td><td>64.00</td><td>72.16</td><td>65.36</td></tr><tr><td>OV-Uni3DETR [39]</td><td>9.66</td><td>48.29</td><td>18.06</td><td></td><td></td><td></td><td>12.09</td><td>30.47</td><td>15.15</td><td></td><td></td><td></td></tr><tr><td>Co-3DGT (ours)</td><td>14.37</td><td>49.85</td><td>22.63</td><td>57.70</td><td>79.36</td><td>63.08</td><td>21.91</td><td>32.83</td><td>23.75</td><td>66.30</td><td>74.21</td><td>69.33</td></tr></table>

Additionally, in the same annotation-free setting following OV-3DET [24], no ground-truth annotated base objects are available. As a result, our sceneawareness-guided uncertainty regularization becomes inapplicable. In this scenario, our model is trained using a simplified variant that retains only the LLM-guided hierarchical alignment with the novel objects discovered by Co-Distillation. Our method still achieves 36.15% $\mathrm { A P _ { 2 5 } ^ { m e a \bar { n } } }$ as shown in Table 2, improving upon OV-Uni3DETR by +10.82%. This improvement confirms the reliability of our Co-Distillation discovery and the robustness of the LLM-guided hierarchical alignment.

Moreover, Figure 3 presents qualitative examples of 3D-OVD results by our method in various indoor scenes on SUN RGB-D and ScanNetV2. Each subfigure illustrates detected 3D objects within reconstructed indoor scenes, indicating our method’s efectiveness to localize and recognize both base and novel objects.

## 4.3 Ablation Study

In Tables 3, 4 and 5, we conduct ablation studies to verify the efectiveness of each component in our framework.

Efectiveness of Co-Distillation Discovery. We first evaluate the performance of diferent discovery pipelines on the training set, focusing on the instancelevel precision and recall for novel categories (left side of Table 3). Compared to the 2D-Detection-based and 3D-Proposal-based pipelines, our Co-Distillation discovery pipeline yields notable improvements. Specifically, compared to the 3D-Proposal-based pipeline, Co-distil(scratch) improves precision by +0.97% and recall by +3.86% on SUN RGB-D, alongside gains of +4.97% and +8.18% on ScanNetV2. Incorporating the pretrained class-agnostic 3D detector, CuTR, further enhances discovery capability. Co-distil(CuTR) improves precision by +2.99% and recall by $+ 1 0 . 7 2 \%$ on SUN RGB-D, over the 3D-Proposal-based baseline, and achieves +5.89% and +12.01% respective gains on ScanNetV2.

Table 2: Detailed comparison of per-category $\mathrm { A P _ { 2 5 } }$ (%) for open-vocabulary 3D object detection on ScanNetV2 dataset across 20 unseen categories.
<table><tr><td>Method</td><td>| Mean |</td><td>toilet</td><td>bed</td><td>chair</td><td>sofa</td><td>dresser</td><td>table</td><td></td><td>cabinet bookshelf pillow</td><td></td><td>sink</td></tr><tr><td>OV-3DET [24]</td><td>18.02</td><td>57.29</td><td>42.26</td><td>27.06</td><td>31.50</td><td>8.21</td><td>14.17</td><td>2.98</td><td>5.56</td><td>23.00</td><td>31.60</td></tr><tr><td>CoDA [3]</td><td>19.32</td><td>68.09</td><td>44.04</td><td>28.72</td><td>44.57</td><td>3.41</td><td>20.23</td><td>5.32</td><td>0.03</td><td>27.95</td><td>45.26</td></tr><tr><td>CoDAv2 [4]</td><td>22.72</td><td>77.24</td><td>43.96</td><td>15.05</td><td>53.27</td><td>11.37</td><td>13.96</td><td>1.42</td><td>0.11</td><td>34.42</td><td>44.38</td></tr><tr><td>OV-Uni3DETR [39]</td><td>25.33</td><td>86.05</td><td>50.49</td><td>28.11</td><td>31.51</td><td>18.22</td><td>24.03</td><td>6.58</td><td>12.17</td><td>29.62</td><td>54.63</td></tr><tr><td>Co-3DGT (ours)</td><td>36.15</td><td>92.75</td><td>70.42</td><td>81.27</td><td>60.64</td><td>12.50</td><td>39.15</td><td>15.92</td><td>12.56</td><td>37.58</td><td>60.59</td></tr><tr><td>Method</td><td colspan="3">|bathtub refrigerator</td><td colspan="3">desk nightstand counter</td><td colspan="3">door curtain</td><td>lamp</td><td>bag</td></tr><tr><td>OV-3DET [24]</td><td></td><td>56.28</td><td>10.99</td><td>19.72</td><td>0.77</td><td>0.31</td><td>9.59</td><td>10.53</td><td>3.78</td><td>2.11</td><td>2.71</td></tr><tr><td>CoDA [3]</td><td></td><td>50.51</td><td>6.55</td><td>12.42</td><td>15.15</td><td>0.68</td><td>7.95</td><td>0.01</td><td>2.94</td><td>0.51</td><td>2.02</td></tr><tr><td>CoDAv2 [4]</td><td></td><td>55.60</td><td>24.41</td><td>20.67</td><td>20.72</td><td>0.28</td><td>13.54</td><td>0.92</td><td>4.16</td><td>4.37</td><td>9.20</td></tr><tr><td>OV-Uni3DETR [39]</td><td></td><td>63.73</td><td>14.41</td><td>30.47</td><td>2.94</td><td>1.00</td><td>19.02</td><td>19.90</td><td>12.70</td><td>5.58</td><td>13.46</td></tr><tr><td>Co-3DGT (ours)</td><td></td><td>69.52</td><td>34.57</td><td>24.23</td><td>13.15</td><td>7.01</td><td>15.99</td><td>16.86</td><td>15.99</td><td>38.61</td><td>4.35</td></tr></table>

Table 3: Ablation study of the Co-Distillation module on SUN RGB-D and Scan-NetV2. Models are evaluated using $\mathrm { A P _ { 2 5 } ^ { n o v e l , d i s c } }$ pseudo-label quality on the training samples, and $\mathrm { A P _ { 2 5 } ^ { n o v e l , d e t } }$ (%) and $\mathrm { \dot { A R } _ { 2 5 } ^ { n o v e l , d e t } }$ (%) to assess final detection performance.
<table><tr><td rowspan="3">Method</td><td colspan="4">Novel Object Discovery</td><td colspan="4">Training Result</td></tr><tr><td colspan="2">SUN RGB-D  $\mathrm { A P _ { 2 5 } ^ { n o v e l , d i s c } }$ </td><td colspan="2">ScanNetV2</td><td colspan="2">SUN RGB-D</td><td colspan="2">ScanNetV2</td></tr><tr><td></td><td></td><td> $\mathrm { A P _ { 2 5 } ^ { n o v e l , d i s c } }$ </td><td> $\mathrm { A R } _ { 2 5 } ^ { \mathrm { n o v e l , d i s c } }$ </td><td> $\mathrm { A P _ { 2 5 } ^ { n o v e l , d e t } }$ </td><td></td><td> $\mathrm { A P _ { 2 5 } ^ { n o v e l , d e t } }$ </td><td></td></tr><tr><td>2D-Detection-based</td><td>6.21</td><td>16.10</td><td>3.93</td><td>23.44</td><td>9.59</td><td>52.99</td><td>11.92</td><td>57.85</td></tr><tr><td>3D-Proposal-based</td><td>8.49</td><td>12.02</td><td>4.57</td><td>20.28</td><td>9.88</td><td>52.15</td><td>12.71</td><td>58.38</td></tr><tr><td>Co-distil(scratch)</td><td>10.45(+1.96) 19.72(+7.70)</td><td></td><td>9.35(+4.76)</td><td>28.46(+8.18)</td><td>11.76(+1.88) 54.26(+2.11)</td><td></td><td></td><td>16.68(+3.97) 63.70(+5.32)</td></tr><tr><td>Co-distil(CuTR)</td><td>10.97(+2.48) 24.33(+12.31)</td><td></td><td></td><td>10.46(+5.89) 32.29(+12.01)</td><td>12.69(+2.81) 55.41(+3.26)</td><td></td><td>18.74(+6.03) 64.19(+5.81)</td><td></td></tr></table>

Subsequently, we analyze how the discovery results from Co-Distillation with CuTR translate to the final detection performance for novel classes (right side of Table 3), where our proposed dual-guidance training strategies are not applied. Better quality in novel object discovery directly contributes to higher final training results. Relying solely on the baseline training method without our robust training methods, Co-distil(CuTR) achieves 12.69% $\mathrm { A P _ { 2 5 } ^ { n o v e l } }$ and 55.41% $\mathrm { A R _ { 2 5 } ^ { n o v e l } }$ on SUN RGB-D, as well as 18.74% $\mathrm { A P _ { 2 5 } ^ { n o v e l } }$ and 64.19% $\mathrm { A R _ { 2 5 } ^ { n o v e l } }$ on ScanNetV2. These consistent improvements across both the training set and the final detection results verify that our Co-Distillation efectively mines high-quality novel objects to boost model performance.

Efectiveness of Dual Guidance Robust Training. Based on the results of Co-Distil(CuTR) discovery, we further ablate our proposed dual guidance robust training strategies in Table 4, including scene-awareness-guided uncertainty regularization (SGUR) and LLM-guided hierarchical alignment (LGHA). Here, we focus on the performance improvements across all classes $( \mathrm { A P _ { 2 5 } ^ { m e a n } } )$ . On SUN RGB-D, applying SGUR alone improves the performance by +1.55% (achieving 22.17% $\mathrm { A P _ { 2 5 } ^ { m e a n } } )$ , while LGHA alone provides $\mathrm { ~ a ~ } + 0 . 7 3 \%$ gain (reaching 21.35% $\mathrm { A P _ { 2 5 } ^ { m e a n } } )$ . When combined, they bring a total improvement of +2.01%, yielding a final result of 22.63% $\mathrm { A P _ { 2 5 } ^ { m e a n } }$ . We observe a similar trend on ScanNetV2, where SGUR and LGHA contribute gains of +2.97% and +2.58%, respectively. Their combination achieves 23.75% $\mathrm { A P _ { 2 5 } ^ { m e a n } }$ , surpassing the baseline by +3.60%. These results demonstrate the complementary roles of SGUR and LGHA in mitigating noise and enhancing overall training robustness.

Table 4: Ablation study for dual guidance robust model training. $\mathrm { A P _ { 2 5 } ^ { m e a n } \ ( \% ) }$ on SUN RGB-D and ScanNetV2 are reported.  
Table 5: Ablation study for LLM selection in LGHA. $\mathrm { A P _ { 2 5 } ^ { m e a n } }$ (%) on SUN RGB-D and ScanNetV2 are reported.
<table><tr><td>Variants</td><td>SUN RGB-D</td><td>ScanNetV2</td></tr><tr><td>Training w/o dual guidance</td><td>20.62</td><td>20.15</td></tr><tr><td>SGUR√</td><td> $2 2 . 1 7 \ ( + 1 . 5 5 )$ </td><td> $2 3 . 1 2 \ \left( + 2 . 9 7 \right)$ </td></tr><tr><td>LGHA√</td><td> $2 1 . 3 5 \ ( + 0 . 7 3 )$ </td><td> $2 2 . 7 3 \ ( + 2 . 5 8 )$ </td></tr><tr><td> $\mathrm { S G U R } \nearrow + \mathrm { L G H A } \nearrow$ </td><td> $2 2 . 6 3 \ \AA + 2 . 0 1 \AA )$ </td><td> $2 3 . 7 5 \ \AA + 3 . 6 0 \dot { ) }$ </td></tr></table>

<table><tr><td>Models</td><td>SUN RGB-D</td><td>ScanNetV2</td></tr><tr><td>ChatGPT-5</td><td>22.63</td><td>23.75</td></tr><tr><td>Gemini-3-Flash-Thinking</td><td>21.89</td><td>23.84</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>22.71</td><td>23.34</td></tr><tr><td>Llama-3.3-70B-Instruct</td><td>21.42</td><td>23.51</td></tr></table>

![](images/5f7c30ee1fcfccba23b5b1025c23cc396b6feee1c58d35374078fce84a18a397.jpg)  
(a)

![](images/a8b31d5c55cb4da8a2047eb1f86b3527b3a13ca8bbd7da7d1c018c83412b7105.jpg)  
(b)

![](images/62704a05d0f09d849d2cbff7fa87006465697ba3b190419a017af4a603980b2b.jpg)

![](images/2630a5e175f382a7babf2782e0f4cb40c4cd2e9fe452696a40403df390844215.jpg)

(e)  
![](images/16f8c7ed45c02582dcd1f46b7b87913a06359fcc59286daae3c736e8f0c6930f.jpg)

(c)  
![](images/26a0137093d9198a59fbbcecb196eec9c6af164c66313fd1c5da733b3084a2ad.jpg)  
(f )

![](images/8c76d453a5af8c1c7cf860ec62ce4fcd7344b566c57d3ce204e52bf0f6244b87.jpg)  
(g)

(d)  
![](images/90490e66953051ce74387f148ba03bc38bc5da9ba80718c0deb83d1b213dedb6.jpg)  
(h)  
Fig. 3: Qualitative examples on SUN RGB-D (the first row) and ScanNetV2 (the second row) to visualize the 3D-OVD results by our method. The red and orange colored bboxes correspond to diferent base-class objects, while blue and green colored bboxes represent diferent novel-class objects.

Influence of Diferent LLMs. In Table 5, we investigate the impact of utilizing diferent LLMs to generate the hierarchical class labels required for LGHA. Our results indicate that changing the underlying LLM yields similar performances of $\mathrm { A P _ { 2 5 } ^ { m e a n } }$ with minimal fluctuations. This demonstrates that our LGHA strategy is robust to the choice of the LLM, as the alignment relies primarily on the rich structural and semantic hierarchy of the generated class labels rather than the specific capabilities or phrasing of a particular LLM.

## 5 Conclusion

In this work, we propose an innovative framework for 3D-OVD, focusing on reliable novel object discovery and robust model training. For reliable discovery, we introduce a Hungarian matching-based co-distillation strategy to generate high-quality novel objects by integrating 2D semantic and 3D geometric information. For robust model training, we introduce a dual guidance training method including scene-awareness-guided uncertainty regularization for robust localization and LLM-guided hierarchical alignment for mitigating semantic ambiguity in classification. Experiments show significant performance gains over state-ofthe-art methods. In future work, we plan to extend the proposed framework toward real-time 3D-OVD by incorporating temporal fusion to enable eficient perception for robotic navigation in indoor environments.

## Acknowledgments

This research was supported in part by the National Key Research & Development Program of China under Grant 2022YFA1004100, and in part by the Ministry of Education, Singapore, under its MOE Academic Research Fund Tier 2 (MOE-T2EP20124-0013).

## References

1. Alayrac, J.B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al.: Flamingo: a visual language model for few-shot learning. NeurIPS 35, 23716–23736 (2022)

2. Cao, S., Li, C., Xu, J., Li, T., Zhao, N.: Late-decoupled 3d hierarchical semantic segmentation with semantic prototype discrimination based bi-branch supervision. arXiv preprint arXiv:2511.16650 (2025)

3. Cao, Y., Yihan, Z., Xu, H., Xu, D.: Coda: Collaborative novel box discovery and cross-modal alignment for open-vocabulary 3d object detection. NeurIPS 36, 71862–71873 (2023)

4. Cao, Y., Zeng, Y., Xu, H., Xu, D.: Collaborative novel object discovery and box-guided cross-modal alignment for open-vocabulary 3d object detection. IEEE Transactions on Pattern Analysis and Machine Intelligence 47(11), 10475–10489 (2025)

5. Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T., Nießner, M.: Scannet: Richly-annotated 3d reconstructions of indoor scenes. In: CVPR. pp. 5828–5839 (2017)

6. Deng, J., Dong, W., Socher, R., Li, L., Li, K., Fei-Fei, L.: Imagenet: A large-scale hierarchical image database. In: CVPR. pp. 248–255 (2009)

7. Deng, J., Lu, J., Zhang, T.: Quantity-quality enhanced self-training network for weakly supervised point cloud semantic segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence 47(5), 3580–3596 (2025)

8. Deng, J., He, T., Jiang, L., Wang, T., Dayoub, F., Reid, I.: 3d-llava: Towards generalist 3d lmms with omni superpoint transformer. In: CVPR. pp. 3772–3782 (2025)

9. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. ICLR (2021)

10. Etchegaray, D., Huang, Z., Harada, T., Luo, Y.: Find n’propagate: Openvocabulary 3d object detection in urban environments. In: ECCV. vol. 15098, pp. 133–151 (2024)

11. Everingham, M., Van Gool, L., Williams, C.K., Winn, J., Zisserman, A.: The pascal visual object classes (voc) challenge. International Journal of Computer Vision 88(2), 303–338 (2010)

12. Gu, X., Lin, T.Y., Kuo, W., Cui, Y.: Open-vocabulary object detection via vision and language knowledge distillation. In: ICLR (2022)

13. Gupta, A., Dollar, P., Girshick, R.: Lvis: A dataset for large vocabulary instance segmentation. In: CVPR. pp. 5356–5364 (2019)

14. Han, Y., Zhao, N., Chen, W., Ma, K.T., Zhang, H.: Dual-perspective knowledge enrichment for semi-supervised 3d object detection. In: AAAI. vol. 38, pp. 2049– 2057 (2024)

15. Huang, J., Zhang, J., Jiang, K., Lu, S.: Open-vocabulary object detection via language hierarchy. In: NeurIPS. vol. 37, pp. 124951–124978 (2024)

16. Jiang, L., Shi, S., Schiele, B.: Open-vocabulary 3d semantic segmentation with foundation models. In: CVPR. pp. 21284–21294 (2024)

17. Jiao, P., Zhao, N., Chen, J., Jiang, Y.G.: Unlocking textual and visual wisdom: Open-vocabulary 3d object detection enhanced by comprehensive guidance from text and image. In: ECCV. vol. 15106, pp. 376–392 (2024)

18. Kendall, A., Gal, Y.: What uncertainties do we need in bayesian deep learning for computer vision? NeurIPS 30, 5574–5584 (2017)

19. Lazarow, J., Grifiths, D., Kohavi, G., Crespo, F., Dehghan, A.: Cubify anything: Scaling indoor 3d object detection. In: CVPR. pp. 22225–22233 (2025)

20. Li, J., Li, D., Xiong, C., Hoi, S.: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In: ICML. pp. 12888– 12900 (2022)

21. Li, L.H., Zhang, P., Zhang, H., Yang, J., Li, C., Zhong, Y., Wang, L., Yuan, L., Zhang, L., Hwang, J.N., et al.: Grounded language-image pre-training. In: CVPR. pp. 10965–10975 (2022)

22. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: ECCV. pp. 740–755 (2014)

23. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H., et al.: Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In: ECCV. vol. 15105, pp. 38–55 (2024)

24. Lu, Y., Xu, C., Wei, X., Xie, X., Tomizuka, M., Keutzer, K., Zhang, S.: Openvocabulary point-cloud object detection without 3d annotation. In: CVPR. pp. 1190–1199 (2023)

25. Misra, I., Girdhar, R., Joulin, A.: An end-to-end transformer model for 3d object detection. In: ICCV. pp. 2906–2917 (2021)

26. Qi, C.R., Litany, O., He, K., Guibas, L.J.: Deep hough voting for 3d object detection in point clouds. In: ICCV. pp. 9277–9286 (2019)

27. Qi, C.R., Yi, L., Su, H., Guibas, L.J.: Pointnet++: Deep hierarchical feature learning on point sets in a metric space. NeurIPS 30, 5099–5108 (2017)

28. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML. pp. 8748–8763 (2021)

29. Ren, S., He, K., Girshick, R., Sun, J.: Faster r-cnn: Towards real-time object detection with region proposal networks. IEEE Transactions on Pattern Analysis and Machine Intelligence 39(6), 1137–1149 (2016)

30. Seitzer, M., Tavakoli, A., Antic, D., Martius, G.: On the pitfalls of heteroscedastic uncertainty estimation with probabilistic neural networks. In: ICLR (2022)

31. Sheng, H., Cai, S., Zhao, N., Deng, B., Liang, Q., Zhao, M.J., Ye, J.: Ct3d++: Improving 3d object detection with keypoint-induced channel-wise transformer. International Journal of Computer Vision 133(7), 4817–4836 (2025)

32. Singh, A., Fry, A., Perelman, A., Tart, A., Ganesh, A., El-Kishky, A., McLaughlin, A., Low, A., Ostrow, A., Ananthram, A., et al.: Openai gpt-5 system card. arXiv preprint arXiv:2601.03267 (2025)

33. Song, S., Lichtenberg, S.P., Xiao, J.: Sun rgb-d: A rgb-d scene understanding benchmark suite. In: CVPR. pp. 567–576 (2015)

34. Tran, P.V.: Simltd: Simple supervised and semi-supervised long-tailed object detection. In: CVPR. pp. 4672–4681 (2025)

35. Wang, J., Zhao, N.: Uncertainty meets diversity: A comprehensive active learning framework for indoor 3d object detection. In: CVPR. pp. 20329–20339 (2025)

36. Wang, J., Chen, B., Kang, B., Li, Y., Xian, W., Chen, Y., Xu, Y.: Ov-dquo: Open-vocabulary detr with denoising text query training and open-world unknown objects supervision. In: AAAI. pp. 7762–7770 (2025)

37. Wang, X., Li, S., Kallidromitis, K., Kato, Y., Kozuka, K., Darrell, T.: Hierarchical open-vocabulary universal image segmentation. NeurIPS 36, 21429–21453 (2023)

38. Wang, Z., Li, Y.L., Chen, X., Zhao, H., Wang, S.: Uni3detr: Unified 3d detection transformer. NeurIPS 36, 39876–39896 (2023)

39. Wang, Z., Li, Y., Liu, T., Zhao, H., Wang, S.: Ov-uni3detr: Towards unified openvocabulary 3d object detection via cycle-modality propagation. In: ECCV. vol. 15105, pp. 73–89 (2024)

40. Wu, Y., Wang, K., Pan, Y., Zhao, N.: Ccf: Complementary collaborative fusion for domain generalized multi-modal 3d object detection. In: CVPR. pp. 18745–18754 (2026)

41. Xu, J., Zhao, N.: Stream3D: Streaming zero-shot 3d instance segmentation with multi-view noise mask filtering and manifold refining. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings. pp. 327–337 (2026)

42. Yao, L., Han, J., Liang, X., Xu, D., Zhang, W., Li, Z., Xu, H.: Detclipv2: Scalable open-vocabulary object detection pre-training via word-region alignment. In: CVPR. pp. 23497–23506 (2023)

43. Yao, L., Han, J., Wen, Y., Liang, X., Xu, D., Zhang, W., Li, Z., Xu, C., Xu, H.: Detclip: Dictionary-enriched visual-concept paralleled pre-training for open-world detection. NeurIPS 35, 9125–9138 (2022)

44. Yuan, S., Xu, J., Hu, P., Zhu, X., Zhao, N.: Graph smoothing for enhanced local geometry learning in point cloud analysis. In: AAAI. vol. 40, p. 12250–12258 (2026)

45. Zareian, A., Rosa, K.D., Hu, D.H., Chang, S.F.: Open-vocabulary object detection using captions. In: CVPR. pp. 14393–14402 (2021)

46. Zeng, Y., Jiang, C., Mao, J., Han, J., Ye, C., Huang, Q., Yeung, D.Y., Yang, Z., Liang, X., Xu, H.: Clip2: Contrastive language-image-point pretraining from real-world point cloud data. In: CVPR. pp. 15244–15253 (2023)

47. Zhang, H., Li, F., Zou, X., Liu, S., Li, C., Yang, J., Zhang, L.: A simple framework for open-vocabulary segmentation and detection. In: ICCV. pp. 1020–1031 (2023)

48. Zhao, N., Chua, T.S., Lee, G.H.: Sess: Self-ensembling semi-supervised 3d object detection. In: CVPR. pp. 11079–11087 (2020)

49. Zhao, N., Lee, G.H.: Static-dynamic co-teaching for class-incremental 3d object detection. In: AAAI. vol. 36, pp. 3436–3445 (2022)

50. Zhou, X., Girdhar, R., Joulin, A., Krähenbühl, P., Misra, I.: Detecting twentythousand classes using image-level supervision. In: ECCV. vol. 13669, pp. 350–368 (2022)

51. Zhu, X., Zhou, H., Xing, P., Zhao, L., Xu, H., Liang, J., Hauptmann, A., Liu, T., Gallagher, A.: Open-vocabulary 3d semantic segmentation with text-to-image difusion models. In: ECCV. vol. 15087, pp. 357–375 (2024)

52. Zhu, Y., Qian, J., Yang, J., Xie, J., Zhao, N.: Few-shot incremental 3d object detection in dynamic indoor environments. In: CVPR. pp. 18786–18795 (2026)