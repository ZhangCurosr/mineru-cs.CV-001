# SED-FOD: Scattering-Aware Expert Decomposition for Few-Shot Cross-Sensor SAR Object Detection

Shu Yang, Zhen Chen, Zhiyu Jiang, Yanlei Li, and Xingdong Liang

Abstract—Synthetic aperture radar (SAR) object detection is an important part of remote sensing interpretation. However, because of variations in frequency band, resolution, background clutter, and target scattering responses, the performance of existing detectors often degrades when training and testing data are acquired from different SAR domains. Although domain adaptation methods offer a promising paradigm for solving this problem, most of them mainly pursue domain-invariant feature alignment and suppress sensor-dependent scattering characteristics that are useful for object detection. This problem becomes more challenging in few-shot scenarios, where only a few fully annotated target-domain SAR images are available. To address this issue, we propose a scattering-aware shared-specific feature decomposition framework for few-shot SAR domain adaptation object detection. We decompose detection features into a shared path and several soft-gated scattering-specific expert paths. The shared path learns transferable object structural information and is used for asymmetric domain alignment, while the scatteringspecific experts adaptively compensate heterogeneous SAR responses. In addition, routing-domain auxiliary loss is introduced to encourage specific experts to capture sensor-dependent routing preferences, and an expert balancing loss is used to prevent routing collapse. Extensive experiments on four bidirectional heterogeneous SAR detection tasks between FARAD-X/FARAD-Ka and MiniSAR under different few-shot settings have been conducted and experimental results demonstrate that the proposed method achieves superior performance in both forward and reverse adaptation directions.

Index Terms—Synthetic aperture radar (SAR) object detection, few-shot domain adaptation, cross-sensor, mixture of experts.

## I. INTRODUCTION

Synthetic Aperture Radar (SAR) is an active microwave imaging sensor that utilizes pulse compression and synthetic aperture techniques to provide two-dimensional highresolution imagery. Owing to its all-day and all-weather capabilities, SAR is widely used in remote-sensing interpretation tasks [1]–[3]. SAR object detection is an essential task in SAR image interpretation for locating targets of interest, such as vehicles, ships, and aircraft, from complex backgrounds [4]– [6].

In recent years, deep-learning-based detectors have significantly improved SAR object detection performance by learning discriminative representations from large amounts of annotated data with similar statistical characteristics [7]–[9].

However, most existing SAR image object detection studies train and evaluate their model under homogeneous imaging conditions, where training and testing images are acquired from the same or similar sensors. When evaluated on images collected under different SAR imaging conditions, their performance may degrade significantly because of cross-sensor distribution discrepancies. [10]. In practical applications, SAR images are often collected by different sensors with diverse frequency bands, resolutions, imaging geometries, polarization modes, and background environments. These factors lead to significant differences in target characteristics and feature distributions [11]. Consequently, the same vehicle target may present obviously different signatures across sensors, while background structures may also produce confusing scattering patterns leading to degraded detection performance and reduced generalization ability.

Unlike conventional transfer learning methods, which usually require sufficient labeled target-domain data from target domain to finetune the model, heterogeneous SAR object detection often suffers from very limited labeled target-domain data. As a result, the distribution discrepancy between training and testing data cannot be effectively overcome [12]. To solve this problem, domain adaptation methods have been proposed. These methods usually reduce the distribution discrepancy between the source and target domains by mapping the data from the source domain and the target domain into the same feature space and aligning domain-invariant representations [13]–[15]. Domain adaptation has been proven to be an effective approach to object detection tasks and is currently widely applied in the fields of computer vision and optical remote sensing. In particular, most previous studies focus on unsupervised domain adaptation, which requires abundant unlabeled data from the target domain. Although this can reduce the high cost caused by the fact that SAR image interpretation usually requires expert knowledge such a requirement is impractical when target data are inherently scarce. This scenario often occurs when a new SAR platform captures a new batch of data and requires rapid preliminary interpretation. Therefore, developing a few-shot domain adaptation strategy is crucial for practical cross-sensor SAR object detection [16].

However, few-shot domain adaptation for SAR object detection still faces certain challenges. Most existing studies mainly pursue domain-invariant representation learning by aligning the source and target feature distributions [17]–[19]. This alignment strategy has been widely adopted in optical domain adaptation, where domain shifts are often dominated by lowlevel appearance variations such as illumination, weather and styles. As shown in Fig. 1(a), the clear-to-foggy variation in optical images (Foggy Cityscapes [20]) mainly changes lowlevel appearance, while the vehicle structure and scene layout remain largely consistent. Such domain-specific visual variations usually provide limited direct cues for target discrimination and are often treated as nuisance factors to be aligned or suppressed. In contrast, cross-sensor SAR vehicle detection presents a more challenging scenario. Fig. 1(b) shows that cross-sensor SAR images exhibit significant discrepancies in background clutter and target scattering responses. The enlarged vehicle patches in Fig. 1(c) further indicate that SAR targets across domains still share certain structural cues, while their local scattering responses vary noticeably. Therefore, the domain discrepancy in SAR imagery should not be regarded as a simple nuisance appearance shift, and directly enforcing the entire feature representation to be domain-invariant may be suboptimal. [21], [22].

![](images/f687055dd896ace83c12b78ff73615bc7671886f723b84e710940d2e359c0e8f.jpg)  
Fig. 1. Comparison between optical and cross-sensor SAR domain shifts. (a) Cityscapes and Foggy Cityscapes illustrate low-level visual variations caused by fog, which provide limited target-discriminative cues. (b) FARAD and MiniSAR show cross-sensor SAR discrepancies in clutter and target scattering responses. (c) Enlarged vehicle patches highlight shared target structures with green boxes and domain-specific scattering cues with blue dots.

This makes existing feature-level alignment methods potentially problematic from two aspects. On the one hand, domainspecific features like scattering centers, shadow responses, and resolution-dependent target structures may be weakened if incorrectly treated as noise, which reduces target objectness and leads to missed detections. On the other hand, background structures such as building edges, parking lots, sidelobelike bright points, and clutter may exhibit vehicle-like local scattering responses. Forcing source and target features to be aligned through directly minimizing domain discrepancy with only a few target samples may pull such confusing background responses toward the target feature space, resulting in false alarms. Therefore, the key challenge of few-shot SAR adaptation is not simply to remove domain-specific information, but to distinguish transferable target structures from specific discriminative information that is still useful for detection.

This observation motivates us to rethink the role of domainspecific discriminative information in few-shot SAR adaptation. Instead of simply minimizing domain discrepancy, we argue that the detection representation should be decomposed into transferable shared features and sensor-dependent scattering-specific characteristics. The domain invariant representation is expected to capture domain-transferable target structures while the scattering-specific representation should be preserved and adaptively exploited to compensate for heterogeneous SAR responses.

To overcome this issue, we propose a scattering-aware shared-specific feature decomposition framework for few-shot SAR domain adaptive object detection. The proposed method introduces a shared path and multiple soft-gated specific expert paths. The shared path is used to learn transferable target representations to guide subsequent source data selection and domain alignment while the scattering-specific representations learned by the experts will be adaptively weighted to model specific characteristics and provide compensation for object prediction. With the help of both paths, the proposed method aligns transferable structures and retains important scatteringspecific information to reduce the risk of over-alignment and over-fitting in few-shot heterogeneous SAR adaptation.

The main contributions are as follows:

• We propose a scattering-aware shared-specific feature decomposition framework for few-shot SAR domain adaptive object detection, which separates transferable domain invariant representation from sensor-dependent scattering characteristics.

• We design a soft gated scattering-specific expert compensation module, where multiple experts adaptively model heterogeneous SAR scattering variations.

• We perform source data selection and domain alignment on the shared representation, reducing the risk of overaligning and over-fitting. Extensive experiments demonstrate that the proposed method significantly outperforms state-of-the-art methods.

## II. RELATED WORK

## A. SAR Object Detection and Domain Adaptation

SAR object detection has received increasing attention because SAR sensors provide reliable imaging under diverse weather and illumination conditions [23]. Early methods mainly relied on statistical modeling and handcrafted features, among which constant false alarm rate methods and their variants were extensively studied [24]–[26]. With the development of deep learning, CNN based detectors have substantially improved detection performance in complex SAR scenes [27]– [29]. However, SAR imagery differs fundamentally from optical imagery and is characterized by sparse scattering centers, target shadows, speckle noise, and strong background clutter [30]–[32]. Target responses also vary with frequency band, spatial resolution, imaging geometry, polarization, and background conditions [16]. Consequently, detectors trained under homogeneous imaging conditions often generalize poorly to heterogeneous SAR data, particularly when only limited target domain annotations are available [33].

Domain adaptive object detection addresses this problem by reducing the distribution discrepancy between source and target domains [34]. Feature alignment methods align image level or instance level representations through adversarial learning or progressive adaptation [35]–[37]. Data enhancement methods instead generate target like samples or combine source and target content, as exemplified by transition image generation, Cross Domain CutMix, and Fourier domain adaptation [1], [38], [39]. Another line of work adopts pseudo labeling and teacher student learning to progressively refine target domain supervision [40]–[43]. Although effective, these methods typically require sufficient target samples to estimate the target distribution or construct reliable pseudo labels. Their applicability is therefore limited when a newly deployed SAR platform provides only a small number of annotated images.

## B. Few Shot Cross Sensor SAR Object Detection

Few shot object detection adapts a detector to novel scenarios using only a small set of annotated samples [44]. A common strategy first trains the detector on source data and then fine tunes selected components using limited target samples. FSDet demonstrates that simple fine tuning provides a competitive baseline [45], while AcroFOD improves adaptation through domain aware data enhancement and candidate filtering [46]. AsyFOD further introduces target guided source selection and asymmetric alignment to address the imbalance between abundant source samples and scarce target samples [17]. For SAR imagery, FsDAOD employs mutual information regularization to preserve domain related knowledge [16], while CS FSDet introduces distribution alignment and domain interaction modules for cross sensor adaptation [47]. These methods demonstrate the value of few shot adaptation when both domain discrepancy and target data scarcity are present.

Most existing approaches nevertheless emphasize domain invariant representation learning through feature alignment or enhancement. This strategy may be suboptimal for heterogeneous SAR imagery because sensor dependent variations contain not only nuisance factors but also discriminative scattering cues. Enforcing complete invariance may suppress useful scattering centers, shadow structures, and resolution dependent responses, thereby causing negative transfer. More broadly, conditional representation learning has shown that explicitly preserving task relevant conditions can improve controllability and consistency in pose guided generation, virtual dressing, and story visualization [48]–[51]. Motivated by this principle, our method decomposes the detection representation into transferable shared features and sensor dependent scattering features. Domain adaptation is performed only on the shared representation, while scattering specific experts preserve and adaptively compensate for heterogeneous SAR responses.

## III. METHOD

The overall framework of the proposed method is illustrated as Fig. 2. Following the asymmetric Adaptation Paradigm [17], source and few-shot target SAR images are first fed into the backbone to extract hierarchical features, which are then used for source selection, domain alignment, and object detection. However, directly applying selection and alignment to the whole detection feature may over-align sensor-dependent scattering responses in cross-sensor SAR imagery. To address this issue, we decompose the detection feature into an always-on shared path and multiple soft-gated scattering-specific expert paths. The shared representation is used for source selection and asymmetric alignment, while the scattering-specific experts provide adaptive compensation for object detection.

The remainder of this section is organized as follows. Section III-A formulates the few-shot cross-sensor SAR object detection problem. Section III-B introduces the proposed shared-specific feature decomposition. Section III-C presents the soft-gated scattering-specific expert compensation module. Section III-D describes shared-representation-guided source selection and domain alignment. Finally, Section III-E gives the overall optimization objective.

## A. Problem Formulation

Let $\mathcal { D } _ { s } ~ = ~ \{ ( x _ { i } ^ { s } , y _ { i } ^ { s } ) \} _ { i = 1 } ^ { N _ { s } }$ be the labeled source-domain SAR dataset, where $x _ { i } ^ { s }$ denotes a source SAR image and $y _ { i } ^ { s } ~ = ~ \{ ( b _ { i j } ^ { s } , c _ { i j } ^ { s } ) \} _ { j = 1 } ^ { M _ { i } ^ { s } }$ denotes the corresponding object annotations, including bounding boxes $b _ { i j } ^ { s }$ and class labels $c _ { i j } ^ { s } .$ The target-domain few-shot training set is denoted as $\mathcal { D } _ { t } ^ { K } \overset { } { = }$ $\{ ( x _ { i } ^ { t } , y _ { i } ^ { t } ) \} _ { i = 1 } ^ { K }$ , where only K fully annotated target-domain images are available for adaptation. In this work, K denotes the number of fully annotated target-domain images rather than the number of object instances. The source and target domains share the same object category but are collected by different scenes, leading to a significant distribution discrepancy, i.e. $P _ { s } ( x , y ) \neq P _ { t } ( x , y )$

## B. Scattering-Aware Shared-Specific Feature Decomposition

The above analysis indicates that few-shot cross-sensor SAR adaptation should not only reduce the source–target distribution discrepancy, but also handle sensor-dependent scattering variations. Therefore, the feature representation is expected to satisfy two coupled objectives. First, the detector should align transferable object structures so that vehicle-related representations can be shared across sensors, which helps reduce missed detections caused by severe domain shift. Second, the detector should preserve and adaptively exploit sensordependent scattering responses, so that useful target scattering cues are not suppressed and confusing background scatterers can be better distinguished from real vehicles. To this end, we decompose the detection feature into a shared component for domain adaptation and scattering-specific expert components for residual compensation.

![](images/30ff4da3264b60a7b3cd697a3dd26bd97800dc25dede359ed2aaabc3848c35bb.jpg)  
Fig. 2. Overall framework of the proposed method. Source and few-shot target SAR images are fed into a shared feature extractor, followed by an MoE featur learning module composed of an always-on shared path and router-weighted scattering-specific experts. The shared representation is used for asymmetric domain alignment, while the aggregated expert features provide domain-sensitive compensation for object detection.

Given an intermediate detection feature map $\mathbf { F } _ { l } \in$ $\mathbb { R } ^ { C _ { l } \times H _ { l } \times W _ { l } }$ at the l-th feature level, we decompose it into a domain-shared representation and multiple scattering-specific representations. The domain-shared path is designed to extract transferable responses from the contextual features encoded by the detector, including sensor-stable cues related to target structure, spatial configuration, and object-background relations. It is formulated as

$$
\mathbf { F } _ { l } ^ { \mathrm { s h } } = G _ { l } ^ { \mathrm { s h } } ( \mathbf { F } _ { l } ) ,\tag{1}
$$

where $G _ { l } ^ { \mathrm { s h } } ( \cdot )$ denotes the shared feature transformation.

In parallel, multiple scattering-specific experts are introduced to capture complementary residual responses. The output of the e-th expert is given by

$$
{ \bf F } _ { l , e } ^ { \mathrm { s p } } = G _ { l , e } ^ { \mathrm { s p } } ( { \bf F } _ { l } ) , \qquad e = 1 , \ldots , E ,\tag{2}
$$

where E is the number of specific experts and $G _ { l . e } ^ { \mathrm { s p } } ( \cdot )$ denotes the transformation of the e-th expert. Unlike the shared path, the specific expert paths are not explicitly constrained to produce domain-invariant representations. With independent parameters, they can learn complementary channel transformations for scattering-related residual responses associated with variations in frequency band, resolution, clutter, shadow, and scattering-center distribution.

The shared transformation and all specific expert transformations adopt the same lightweight architecture while maintaining independent parameters. Specifically, each transformation consists of two pointwise convolutions with channel expansion and projection:

$$
\boldsymbol { G } ( \mathbf { F } _ { l } ) = \mathrm { G N } _ { 2 } \left( \mathbf { W } _ { 2 } ^ { 1 \times 1 } \ast \boldsymbol { \phi } \left( \mathrm { G N } _ { 1 } \left( \mathbf { W } _ { 1 } ^ { 1 \times 1 } \ast \mathbf { F } _ { l } \right) \right) \right) ,\tag{3}
$$

where $\mathbf { W } _ { 1 } ^ { 1 \times 1 }$ expands the channel dimension from $C _ { l }$ to $r C _ { l } , \ \mathbf { W } _ { 2 } ^ { 1 \times 1 }$ projects it back to $C _ { l } , ~ \mathrm { G N } ( \cdot )$ denotes group normalization, and $\phi ( \cdot )$ is the SiLU activation function. We set the expansion ratio to $r = 2$ . Using a homogeneous architecture ensures that the complementary behaviors of different experts arise from their independently learned parameters and adaptive routing weights, rather than manually assigned expert structures.

## C. Soft-Gated Scattering-Specific Expert Compensation

Although the shared path provides transferable object structures, a single domain-specific branch is insufficient to represent the heterogeneous scattering variations in crosssensor SAR images. The sensor-dependent discrepancy may be caused by different factors, such as scattering-center distribution, shadow response, background clutter, resolution degradation, and sidelobe-like interference. These factors vary across images and local scenes, making the domain-specific residual a mixture of multiple scattering patterns rather than a single deterministic transformation.

To model such heterogeneous SAR responses, inspired by the Mixture of Experts model [52], we introduce a soft-gated expert compensation mechanism. For the l-th feature level, a routing function predicts the importance of each scatteringspecific expert according to the input feature:

$$
\mathbf { r } _ { l } = \mathrm { s o f t m a x } \left( \frac { R _ { l } ( \mathrm { G A P } ( \mathbf { F } _ { l } ) ) } { \tau } \right) ,\tag{4}
$$

where $R _ { l } ( \cdot )$ denotes the routing function, $\mathrm { G A P } ( \cdot )$ is global average pooling, $\tau$ is a temperature parameter, and $\mathbf { r } _ { l } = [ r _ { l , 1 } , \dots , r _ { l , E } ]$ represents the soft expert weights. The scattering-specific compensation feature is then obtained by

$$
\mathbf { F } _ { l } ^ { s p } = \sum _ { e = 1 } ^ { E } r _ { l , e } \mathbf { F } _ { l , e } ^ { s p } .\tag{5}
$$

To encourage the router to capture domain-related scattering preferences, we further introduce a routing-domain classifier. Given the domain label $d _ { i } \in \{ 0 , 1 \}$ , where 0 and 1 denote the source and target domains, respectively, the routing weights from selected feature levels are concatenated as

$$
\mathbf { q } _ { i } = \mathrm { C o n c a t } \left( \mathbf { r } _ { l _ { 1 } } ( x _ { i } ) , \mathbf { r } _ { l _ { 2 } } ( x _ { i } ) , \ldots , \mathbf { r } _ { l _ { m } } ( x _ { i } ) \right) ,\tag{6}
$$

where $\{ l _ { 1 } , \ldots , l _ { m } \}$ denotes the set of feature levels equipped with scattering-specific experts. A domain classifier C<sub>r</sub>(·) predicts the domain probability from the routing vector:

$$
\hat { \mathbf { d } } _ { i } = C _ { r } ( \mathbf { q } _ { i } ) .\tag{7}
$$

The routing-domain supervision is defined as

$$
\mathcal { L } _ { \boldsymbol { r } d } = \frac { 1 } { | \boldsymbol { B } | } \sum _ { i \in \boldsymbol { B } } \mathrm { C E } \left( \hat { \mathbf { d } } _ { i } , d _ { i } \right) ,\tag{8}
$$

where CE(·) denotes the cross-entropy loss and $\boldsymbol { B }$ is the training mini-batch. This auxiliary loss is not used to make the whole representation domain-invariant. Instead, it encourages the router to softly adjust the weights of scattering-specific experts according to sensor-dependent response patterns. The classifier is used only during training and is removed during inference.

The final output feature is formulated as

$$
\widetilde { \mathbf { F } } _ { l } = \mathbf { F } _ { l } + \alpha _ { l } \mathbf { F } _ { l } ^ { s h } + \beta _ { l } \mathbf { F } _ { l } ^ { s p } ,\tag{9}
$$

where $\alpha _ { l }$ and $\beta _ { l }$ control the contributions of the shared representation and the scattering-specific compensation, respectively. The original feature $\mathbf { { F } } _ { l }$ is preserved through a residual connection to maintain the basic detection representation. The shared path is mandatory for all samples and provides a stable cross-sensor feature basis, while the soft-gated scatteringspecific experts adaptively compensate for sensor-dependent SAR responses.

It is worth noting that the shared path does not participate in the expert competition. This design is particularly important in the few-shot target setting, where completely relying on routed experts may overfit the limited target-domain samples. By enforcing all samples to pass through the shared path, the detector maintains transferable object representations, while the expert paths only provide adaptive residual compensation for scattering-specific variations.

## D. Shared-Representation-Guided Domain Adaptation

After obtaining the shared and scattering-specific representations, domain adaptation is performed only on the shared representation. This design is motivated by two observations in few-shot cross-sensor SAR adaptation in Fig. 3. First, insufficient feature alignment may leave a large discrepancy between the source and target domains. In this case, the source and target features are scattered in the representation space with only limited overlap, making it difficult for the detector trained on source samples to generalize to the target sensor. Second, directly enforcing strong alignment on the entire detection feature is also risky under the few-shot target setting. Since the target distribution is estimated from only a few annotated target images, the alignment anchor can be severely biased toward limited target scattering patterns. As a result, the target features may become over-concentrated around a small number of target samples, while many source samples still remain poorly matched to the target domain. Conceptually, both cases lead to a low overlap between the source and target distributions compared with their overall feature spread.

![](images/8af4e112fb8fea414f159feca25a440714b5927151106546648132b5dadf4e22.jpg)  
Fig. 3. Motivating t-SNE visualization of source and target feature distributions under the 1-shot FARAD-Ka → MiniSAR setting. AcroFOD shows limited source–target overlap, while AsyFOD produces a more concentrated target cluster but still leaves many source samples poorly matched.

Therefore, the key is not to avoid alignment, but to constrain alignment to the transferable part of the representation. The shared path is expected to encode object structures that are relatively stable across sensors, such as target shape, spatial layout, and object-background configuration. In contrast, the scattering-specific expert paths contain sensor-dependent responses related to scattering centers, shadow, clutter, and resolution-induced structures. If the entire fused feature is directly aligned, useful scattering-specific cues may be incorrectly suppressed. Therefore, both target-guided source selection and asymmetric feature alignment are conducted in the shared feature space, while the scattering-specific expert paths are preserved for adaptive compensation.

Specifically, for each target image $\boldsymbol { x } _ { i } ^ { t }$ , its shared feature at the selected high-level feature layer is extracted and globally pooled as

$$
\mathbf { z } _ { i } ^ { t } = \mathrm { G A P } \left( \mathbf { F } ^ { s h } ( x _ { i } ^ { t } ) \right) .\tag{10}
$$

The few-shot target samples are then used to estimate the target-domain shared representation:

$$
\bar { \mathbf { z } } _ { t } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \mathbf { z } _ { i } ^ { t } ,\tag{11}
$$

where K denotes the number of labeled target-domain images available for adaptation.

Similarly, for each source image $x _ { i } ^ { s } .$ , we obtain its shared representation:

$$
\mathbf { z } _ { i } ^ { s } = \mathrm { G A P } \left( \mathbf { F } ^ { s h } ( x _ { i } ^ { s } ) \right) .\tag{12}
$$

Following the target-guided source selection strategy, the relevance between each source image and the few-shot target domain is measured in the shared feature space:

$$
d _ { i } = { \mathcal { D } } \left( \mathbf { z } _ { i } ^ { s } , { \bar { \mathbf { z } } } _ { t } \right) ,\tag{13}
$$

where $\mathcal { D } ( \cdot )$ denotes the feature discrepancy measure. In our implementation, $\mathcal { D } ( \cdot )$ is instantiated by the MMD-based discrepancy used in the target-guided source selection procedure. Source samples with smaller discrepancies are considered more relevant to the target domain and are preferentially selected for training [46]. Vitally, source selection is performed on the shared representation rather than the original detector feature. Therefore, the selected source samples are expected to share transferable object structures with the target domain instead of being selected according to sensor-specific scattering responses.

After obtaining the target-similar source samples, asymmetric feature alignment is imposed on the shared representation [17]. Given a mini-batch containing selected source samples and few-shot target samples, the shared alignment loss is formulated as

$$
\mathcal { L } _ { a l i g n } ^ { s h } = \mathcal { A } _ { a s y } \left( \{ \mathbf { z } _ { i } ^ { s } \} _ { i = 1 } ^ { n _ { s } } , \{ \mathbf { z } _ { j } ^ { t } \} _ { j = 1 } ^ { n _ { t } } \right) ,\tag{14}
$$

where $\mathcal { A } _ { a s y } ( \cdot )$ denotes the adopted asymmetric distribution alignment criterion, and $n _ { s }$ and $n _ { t }$ are the numbers of selected source samples and target samples in the mini-batch, respectively. Different from symmetric alignment that treats the two domains equally, the asymmetric alignment strategy uses the limited target-domain samples as adaptation anchors and encourages the selected source shared features to approach the target-domain distribution. In our implementation, the original asymmetric alignment criterion is applied to the shared representation rather than the entire fused detection feature.

By restricting domain adaptation to the shared feature space, the detector is encouraged to learn domain-invariant object structures while avoiding over-alignment of heterogeneous SAR scattering patterns. Meanwhile, the scattering-specific expert paths are not directly constrained by the alignment loss, allowing them to preserve and compensate sensordependent responses caused by frequency band, resolution, clutter, shadow, and scattering-center variations. This sharedspecific separation improves the robustness of few-shot crosssensor SAR object detection.

## E. Optimization Objective

The overall training objective consists of the detection loss, the shared-representation alignment loss, the routing-domain auxiliary loss, and the expert balancing loss:

$$
\mathcal { L } = \mathcal { L } _ { d e t } + \lambda _ { a l i g n } \mathcal { L } _ { a l i g n } ^ { s h } + \lambda _ { r d } \mathcal { L } _ { r d } + \lambda _ { b a l } \mathcal { L } _ { b a l } ,\tag{15}
$$

where $\lambda _ { a l i g n } , \lambda _ { r d } .$ , and $\lambda _ { b a l }$ are trade-off coefficients.

The detection loss follows the YOLOv5 [43] object detection objective:

$$
\mathcal { L } _ { d e t } = \mathcal { L } _ { b o x } + \mathcal { L } _ { o b j } + \mathcal { L } _ { c l s } ,\tag{16}
$$

where $\mathcal { L } _ { b o x } , \mathcal { L } _ { o b j }$ , and $\mathcal { L } _ { c l s }$ denote the bounding-box regression loss, objectness loss, and classification loss, respectively. The detection loss is computed on both labeled source-domain samples and the few fully annotated target-domain samples.

The shared alignment loss $\mathcal { L } _ { a l i g n } ^ { s h }$ is imposed only on the domain-shared representation. It encourages the detector to learn transferable object structures across sensors while avoiding over-alignment of scattering-specific expert features. The routing-domain loss $\mathcal { L } _ { r d }$ is applied to the routing weights of the scattering-specific experts, encouraging the router to capture sensor-dependent scattering preferences.

![](images/821acaf4169030367cffa42aa64caf38a7cbe4ba152cf8056a36ce21eab2d4f3.jpg)  
Fig. 4. Examples of SAR images and corresponding optical scenes from the datasets used in our experiments. (a), (c), and (e) are SAR images from MiniSAR, FARAD-X, and FARAD-Ka, respectively. (b), (d), and (f) show the corresponding optical scenes on Google Earth.

To prevent routing collapse, we further introduce an expert balancing loss [53]. Let $\bar { \mathbf { r } } _ { l }$ denote the average routing probability over a mini-batch at the l-th feature level:

$$
\bar { \mathbf { r } } _ { l } = \frac { 1 } { | \boldsymbol { B } | } \sum _ { i \in \mathcal { B } } \mathbf { r } _ { l } ( x _ { i } ) .\tag{17}
$$

The balancing loss is defined as

$$
\mathcal { L } _ { b a l } = \sum _ { l \in \mathcal { S } } \left\| \bar { \mathbf { r } } _ { l } - \frac { 1 } { E } \mathbf { 1 } \right\| _ { 2 } ^ { 2 } ,\tag{18}
$$

where $s$ denotes the set of feature levels equipped with scattering-specific experts, E is the number of experts, and 1 is an all-one vector. This loss encourages different experts to be sufficiently activated within a mini-batch, thus avoiding the degeneration where only a few experts dominate the routing process.

During inference, only the detector with the proposed shared-specific feature decomposition and soft-gated expert compensation is retained. The routing-domain classifier and all auxiliary losses are removed, and no target-domain labels are required.

## IV. EXPERIMENTS

## A. Datasets and Experimental Settings

Experiments are conducted on the FARAD-X, FARAD-Ka [54], and MiniSAR [55] vehicle datasets released by Sandia National Laboratories. As summarized in Table I, these datasets were acquired by different radar systems and exhibit evident discrepancies in frequency band, incidence angle, and scene clutter. Representative SAR and optical images are shown in Fig. 4. In this paper, we construct four bidirectional cross-sensor adaptation tasks: FARAD-X to MiniSAR, FARAD-Ka to MiniSAR, MiniSAR to FARAD-X, and MiniSAR to FARAD-Ka. The vehicle category is used as the only target class. Following the preprocessing strategy of

TABLE I  
IMAGING PARAMETERS AND DOMAIN CHARACTERISTICS OF THE DATASETS.
<table><tr><td>Domain</td><td>FARAD-X</td><td>FARAD-Ka</td><td>MiniSAR</td></tr><tr><td>Band</td><td>X</td><td>Ka</td><td>Ku</td></tr><tr><td>Time</td><td>2015.10</td><td>2015.08</td><td>2005.05</td></tr><tr><td>Bandwidth</td><td>3 GHz</td><td>5 GHz</td><td>3GHz</td></tr><tr><td>Center Freq.</td><td>9.6 GHz</td><td>35.6 GHz</td><td>16.8 GHz</td></tr><tr><td>Angle</td><td>26°-34°</td><td>26°-34°</td><td>26°-29°</td></tr><tr><td>Resolution</td><td>0.1 m</td><td>0.1 m</td><td>0.1 m</td></tr><tr><td>Max Range</td><td>12 km</td><td>6 km</td><td>8 km</td></tr><tr><td>Scene Type</td><td>Dense urban</td><td>Dense urban</td><td>Suburban</td></tr><tr><td rowspan="2">Location</td><td>Univ. of</td><td>Univ. of</td><td>Kirtland Air</td></tr><tr><td>New Mexico</td><td>New Mexico</td><td>Force Base</td></tr></table>

SIVED [56], the original large-scale SAR images are cropped into 512 × 512 patches, and only patches containing at least one vehicle target are retained.

For the forward tasks, the training split of FARAD-X or FARAD-Ka is used as the source domain, while 30 MiniSAR images form the target-domain candidate set and the remaining 70 images constitute a fixed test set. For the reverse tasks, the MiniSAR training split is used as the source domain, the FARAD training split forms the target-domain candidate set, and the corresponding validation and test splits are combined for evaluation. No target-domain evaluation image is used during adaptation.

For the K-shot setting, K fully annotated target images are selected from the candidate set for adaptation, where $K \in \{ 1 , 3 , 5 , 8 , 1 0 \}$ . The few-shot subsets are constructed in a nested manner [45], i.e., the samples used in a lowershot setting are included in the corresponding higher-shot setting under the same random seed. The unused images in the candidate set are not involved in training.

All experiments are implemented based on the Asy-FOD [17] framework for fair comparison with representative few-shot domain adaptation baselines. The input SAR patches are resized to $5 1 2 \times 5 1 2$ during both training and testing. The detector is trained for 300 epochs using SGD with an initial learning rate of 0.01 and a batch size of 12, including 11 source-domain images and one target-domain image. All experts are used at each detection feature level. We set $\lambda _ { r d } = 0 . 1 , \lambda _ { b a l } = 0 . 0 0 2$ , and $\tau = 1$ . The learnable coefficient α is initialized to 0.3, while β is fixed to 1. All experiments are conducted on an NVIDIA GeForce RTX 5090 GPU, and performance is evaluated using $\mathrm { \ m A P _ { 5 0 } }$

## B. Comparison with State-of-the-Art Methods

To evaluate the effectiveness of the proposed method, we compare it with representative few-shot and domain adaptation baselines under the same SAR few-shot domain adaptation protocol. Source-only denotes the detector trained only with labeled source-domain images and directly evaluated on the target domain, which serves as the reference result without target-domain adaptation. Following AsyFOD, we implement the FSDet [45] baseline on YOLOv5 by freezing the first ten layers and fine-tuning the remaining parameters using the limited target-domain samples. SSDA-YOLO [42] is included as a YOLO-based domain adaptation baseline and is adapted to the same few-shot target setting for comparison. In addition, AcroFOD [46] and AsyFOD [17] are selected as representative few-shot domain adaptive object detection methods. During reproduction, we found that the asymmetric alignment loss term in the released AsyFOD implementation was not involved in gradient back-propagation. We report both the results obtained using the released implementation, denoted as AsyFOD, and those obtained after correcting the gradient back-propagation, denoted as AsyFOD<sup>†</sup>. The basic detector, data augmentation strategy, optimizer, learning-rate schedule, and asymmetric alignment configuration are kept consistent with AsyFOD unless otherwise specified. For a fair comparison, all methods are evaluated using the same sourcedomain data, predefined target-domain few-shot subsets, and target-domain test sets.

Table II reports the mAP<sub>50</sub> results on four cross-domain adaptation tasks under different few-shot settings. The upper panel presents the FARAD-X → MiniSAR and FARAD-Ka → MiniSAR results, while the lower panel presents the MiniSAR → FARAD-X and MiniSAR → FARAD-Ka results. Since Source-only does not use any target-domain samples for adaptation, its result is independent of the number of target shots. The results show that directly transferring the source-domain detector generally leads to limited target-domain performance because of the considerable discrepancy between heterogeneous SAR sensors. Most target-adapted methods, particularly AcroFOD and AsyFOD<sup>†</sup>, improve upon Source-only in the majority of settings, demonstrating the importance of using the limited target-domain annotations for cross-domain adaptation.

Among the compared adaptation methods, AcroFOD and AsyFOD obtain strong performance by exploiting few-shot target annotations and source-domain knowledge. In particular, AsyFOD provides a competitive baseline by using targetguided source selection and asymmetric alignment. However, these methods mainly focus on sample selection or feature alignment, while sensor-dependent SAR scattering responses are not explicitly modeled. In contrast, the proposed method decomposes the detection representation into a mandatory shared path and soft-gated scattering-specific expert paths. The shared path provides transferable structural representations for source selection and domain alignment, while the scatteringspecific experts preserve and adaptively compensate heterogeneous SAR responses.

As shown in the upper panel of Table II, the proposed method achieves the best performance in all ten FARAD-to-MiniSAR settings. On FARAD-X → MiniSAR, our method improves upon AsyFOD<sup>†</sup> by 3.80, 2.10, 2.88, 3.99, and 2.35 percentage points under the 1-, 3-, 5-, 8-, and 10-shot settings, respectively. On FARAD-Ka → MiniSAR, the corresponding improvements are 5.96, 8.30, 3.31, 6.00, and 2.21 percentage points. The lower panel further evaluates the proposed method in the reverse adaptation direction when MiniSAR is used as the source domain. Different from the forward tasks, only the 80 images in the MiniSAR training split are available as source-domain training data, whereas the forward tasks use 247 FARAD-X or 251 FARAD-Ka source training images.

COMPARISON OF DIFFERENT METHODS ON BIDIRECTIONAL FEW-SHOT SAR DOMAIN ADAPTATION TASKS IN TERMS OF MAP (%).  
TABLE II
<table><tr><td rowspan="2">Methods</td><td colspan="5">(a) FARAD-X → MiniSAR</td><td colspan="5">(b) FARAD-Ka → MiniSAR</td></tr><tr><td>1-shot</td><td>3-shot</td><td>5-shot</td><td>8-shot</td><td>10-shot</td><td>1-shot</td><td>3-shot</td><td>5-shot</td><td>8-shot</td><td>10-shot</td></tr><tr><td>Source-only [43]</td><td></td><td></td><td>22.46</td><td></td><td></td><td></td><td></td><td>25.77</td><td></td><td></td></tr><tr><td>FSDet [45]</td><td>19.38</td><td>30.42</td><td>36.84</td><td>41.01</td><td>47.69</td><td>27.72</td><td>37.33</td><td>44.07</td><td>53.36</td><td>45.26</td></tr><tr><td>SSDA-YOLO [42]</td><td>23.13</td><td>28.48</td><td>39.04</td><td>40.08</td><td>39.45</td><td>29.17</td><td>39.68</td><td>46.67</td><td>46.56</td><td>50.69</td></tr><tr><td>AcroFOD [46]</td><td>27.35</td><td>36.20</td><td>51.81</td><td>61.29</td><td>64.53</td><td>30.98</td><td>41.30</td><td>55.56</td><td>64.08</td><td>66.18</td></tr><tr><td>AsyFOD [17]</td><td>29.09</td><td>36.67</td><td>49.60</td><td>56.58</td><td>63.37</td><td>37.43</td><td>36.90</td><td>52.78</td><td>60.91</td><td>58.62</td></tr><tr><td>AsyFOD† [17]</td><td>43.39</td><td>49.09</td><td>66.10</td><td>72.07</td><td>71.23</td><td>41.30</td><td>43.30</td><td>66.01</td><td>69.72</td><td>73.21</td></tr><tr><td>Our method</td><td>47.19</td><td>51.19</td><td>68.98</td><td>76.06</td><td>73.58</td><td>47.26</td><td>51.60</td><td>69.32</td><td>75.72</td><td>75.42</td></tr><tr><td rowspan="2">Methods</td><td colspan="5">(c) MiniSAR → FARAD-X</td><td colspan="5">(d) MiniSAR → FARAD-Ka</td></tr><tr><td>1-shot</td><td>3-shot</td><td>5-shot</td><td>8-shot</td><td>10-shot</td><td>1-shot</td><td>3-shot</td><td>5-shot</td><td>8-shot</td><td>10-shot</td></tr><tr><td>Source-only [43]</td><td></td><td></td><td>19.42</td><td></td><td></td><td></td><td></td><td>22.64</td><td></td><td></td></tr><tr><td>FSDet [45]</td><td>8.22</td><td>42.94</td><td>53.97</td><td>56.97</td><td>58.05</td><td>15.62</td><td>42.28</td><td>48.48</td><td>54.56</td><td>49.77</td></tr><tr><td>SSDA-YOLO [42]</td><td>17.63</td><td>25.39</td><td>22.92</td><td>16.89</td><td>16.56</td><td>16.66</td><td>20.08</td><td>22.60</td><td>24.63</td><td>15.64</td></tr><tr><td>AcroFOD [46]</td><td>22.41</td><td>50.47</td><td>44.79</td><td>61.68</td><td>46.73</td><td>19.58</td><td>44.37</td><td>44.08</td><td>54.72</td><td>56.72</td></tr><tr><td>AsyFOD [17]</td><td>21.08</td><td>41.02</td><td>42.80</td><td>44.10</td><td>59.42</td><td>11.48</td><td>34.98</td><td>58.90</td><td>51.80</td><td>56.95</td></tr><tr><td>AsyFOD† [17]</td><td>23.94</td><td>52.85</td><td>55.44</td><td>62.75</td><td>64.45</td><td>26.26</td><td>52.24</td><td>61.98</td><td>64.62</td><td>64.57</td></tr><tr><td>Our method</td><td>24.94</td><td>54.22</td><td>58.31</td><td>64.50</td><td>71.76</td><td>23.94</td><td>55.80</td><td>62.00</td><td>72.51</td><td>69.77</td></tr></table>

Note: The K-shot setting denotes the use of K fully annotated images from the target domain. AsyFOD denotes the results reproduced using the released implementation, whereas AsyFOD<sup>†</sup> denotes the reproduced results obtained after correcting the back-propagation of the asymmetric alignment loss.

TABLE III  
DOMAIN-WISE ROUTER DISTRIBUTION STATISTICS ON THE SOURCE AND TARGET IMAGE SETS. EACH EXPERT ENTRY REPORTS THE SOURCE/TARGET MEAN ROUTING WEIGHTS (%). TVD DENOTES THE TOTAL VARIATION DISTANCE BETWEEN THE SOURCE AND TARGET MEAN ROUTING DISTRIBUTIONS, AND ENTROPY DENOTES THE NORMALIZED SOURCE/TARGET ROUTING ENTROPY.
<table><tr><td>Adaptation Task</td><td>Level</td><td>Expert 1</td><td>Expert 2</td><td>Expert 3</td><td>Expert 4</td><td>TVD (%)</td><td>Entropy</td></tr><tr><td rowspan="2">FARAD-X → MiniSAR</td><td>P3</td><td>23.01/22.78</td><td>28.47/29.02</td><td>24.44/24.23</td><td>24.08/23.97</td><td>0.55</td><td>0.9973/0.9966</td></tr><tr><td>P4</td><td>24.30/24.94</td><td>24.30/23.89</td><td>26.62/28.08</td><td>24.79/23.09</td><td>2.10</td><td>0.9993/0.9977</td></tr><tr><td rowspan="2">FARAD-Ka → MiniSAR</td><td>P3</td><td>22.97/22.76</td><td>26.73/26.87</td><td>27.12/27.35</td><td>23.18/23.02</td><td>0.37</td><td>0.9977/0.9973</td></tr><tr><td>P4</td><td>25.89/25.50</td><td>24.92/24.90</td><td>25.30/25.16</td><td>23.88/24.44</td><td>0.56</td><td>0.9996/0.9997</td></tr></table>

The reverse tasks therefore provide less source-domain supervision and more limited coverage of target and background variations, constituting a more challenging adaptation setting. On MiniSAR → FARAD-X, the proposed method outperforms AsyFOD<sup>†</sup> under all five shot settings, with improvements of 1.00, 1.37, 2.87, 1.75, and 7.31 percentage points, respectively. On MiniSAR → FARAD-Ka, AsyFOD<sup>†</sup> obtains the best 1-shot result of 26.26%, whereas the proposed method achieves 23.94%. Nevertheless, our method performs best in the remaining four settings and improves upon AsyFOD<sup>†</sup> by 3.56, 0.02, 7.89, and 5.20 percentage points under the 3-, 5-, 8-, and 10-shot settings, respectively. The relatively lower result on the 1-shot MiniSAR → FARAD-Ka task suggests that, when both the source-domain training set and the target-domain adaptation set are extremely limited, the target distribution estimated from a single FARAD-Ka image may not be sufficiently representative. As more target-domain images become available, the routing preferences and shared target representation can be estimated more reliably, and the advantage of the proposed scattering-specific compensation becomes more evident. These improvements demonstrate that aligning only the shared representations while preserving scattering-specific responses reduces the risk of over-aligning beneficial, sensor-dependent cues. These results validate the effectiveness of the proposed scattering-aware shared-specific feature decomposition for few-shot SAR domain adaptive object detection. Overall, the framework proves effective not only for adapting FARAD domains to MiniSAR, but also under the more challenging reverse setting with a markedly smaller MiniSAR source-domain training set.

Fig. 5 presents qualitative detection results on MiniSAR test images under the 5-shot FARAD-X → MiniSAR setting. Each column corresponds to the same test scene, while the rows from top to bottom show the ground truth, AcroFOD, AsyFOD<sup>†</sup>, and the proposed method, respectively. Compared with AcroFOD and AsyFOD<sup>†</sup>, the proposed method produces more reliable detection results in complex SAR scenes.

In the first column, AcroFOD fails to detect the isolated vehicle near the buildings, while AsyFOD<sup>†</sup> produces additional false alarms in the surrounding background. In contrast, the proposed method correctly detects the target while suppressing background interference. In the third column, the vehicle adjacent to the building is particularly challenging because its scattering response can be confused with nearby strong background structures. Among the compared methods, only the proposed method successfully detects this target.

![](images/0b43b8924ed8d71c63d6c60d871ef9353d1d390ff46eccfc72906e57e49bbe0a.jpg)  
Fig. 5. Qualitative comparison on MiniSAR test images under the 5-shot FARAD-X → MiniSAR adaptation task. Each column represents one test scene while the rows from top to bottom correspond to the ground truth, AcroFOD, AsyFOD<sup>†</sup>, and the proposed method. Green and red boxes denote ground-truth annotations and detection results, respectively.

For the dense vehicle regions shown in the second and fifth columns, the proposed method also provides more complete detection results, indicating better robustness to densely distributed targets and cluttered SAR backgrounds. These observations suggest that the scattering-specific expert compensation helps preserve useful target-related scattering cues while reducing interference from clutter and strong background scatterers.

Fig. 6 visualizes the P3 activation responses of the shared path and four experts on two representative MiniSAR test images. The shared path produces relatively smooth structural responses, whereas the experts exhibit more differentiated patterns. In these examples, Experts 1 and 3 show relatively stronger responses to target-adjacent shadow and surrounding context, while Experts 2 and 4 emphasize more localized scatterers, clutter edges, and structural boundaries. The experts therefore present an evident grouping tendency rather than four completely independent response modes. This observation indicates that the expert paths provide complementary scattering-related cues beyond the shared representation, and also offers qualitative support for the comparable performance of the two- and four-expert configurations reported in the subsequent parameter analysis. To quantitatively complement the expert activation maps in Fig. 6, we further compare the router distributions of source- and target-domain images within the same trained model. As shown in Table III, all four experts maintain non-negligible routing weights, while their relative contributions vary across feature levels and domains. For FARAD-X → MiniSAR, a relatively larger domain-dependent variation is observed at P4, where the routing weight of Expert 3 increases from 26.62% to 28.08%, whereas that of Expert 4 decreases from 24.79% to 23.09%. The corresponding total variation distance reaches 2.10%, which is larger than that at P3. In contrast, the FARAD-Ka → MiniSAR model exhibits smaller routing variations, indicating a relatively stable expert combination with finegrained domain-dependent adjustment. Meanwhile, the normalized routing entropies remain close to one, demonstrating that the router performs soft expert reweighting rather than collapsing to a single expert. Together with the differentiated spatial responses in Fig. 6, these results show that the expert paths provide complementary representations whose relative contributions can be softly adjusted under cross-domain inputs without routing collapse. We next analyze the adaptation behavior of the shared representation by visualizing the sourcedomain and target-domain feature distributions using t-SNE, as shown in Fig. 7. For the proposed method, the visualized features are extracted from the shared path, which is used for target-guided source selection and domain alignment. For the baseline methods, we visualize the corresponding features used for alignment before the detection head. AcroFOD produces dispersed source and target features with limited overlap, indicating that insufficient alignment cannot effectively bridge the cross-sensor discrepancy. AsyFOD<sup>†</sup> brings part of the source and target features closer, but the target features are highly concentrated under the 1-shot setting, while a large portion of source features still remain away from the target cluster. This suggests that direct alignment with extremely limited target samples may be biased toward a narrow target distribution. In contrast, the proposed method obtains a more compact and better-overlapped shared feature distribution, supporting the effectiveness of performing adaptation on the shared representation while leaving scattering-specific experts for residual compensation. As the number of target shots increases, the target-domain estimate becomes more reliable, and the alignment bias can be gradually alleviated. This observation supports our motivation that domain alignment should be mainly performed on shared representations, while scattering-specific expert paths are preserved to compensate sensor-dependent SAR responses.

![](images/b2e3d750b7bff874d4dbd37ba7aec7b9e328e8eb76ce6372fc607f559ee8f8d7.jpg)  
Fig. 6. Visualization of P3 shared and expert activation responses on MiniSAR test images under the 5-shot FARAD-X → MiniSAR setting. From left to right: original image, shared path, and Experts 1–4.

![](images/5f16b17d8d65b984683384ae03f92d24334e590389c50bcdfd6088af88400ad3.jpg)  
Fig. 7. t-SNE visualization of source-domain and target-domain shared feature distributions under the 1-shot FARAD-Ka → MiniSAR adaptation task. Blue circles and orange triangles denote source-domain and target-domain samples, respectively. (a) AcroFOD. (b) AsyFOD<sup>†</sup>. (c) Ours. The proposed method shows a closer source–target distribution in the shared feature space, indicating more effective transferable representation learning for few-shot SAR domain adaptation.

## C. Ablation Study

To evaluate the contribution of each component, we conduct ablation studies on FARAD-X → MiniSAR and FARAD-Ka → MiniSAR under different few-shot settings. AsyFOD<sup>†</sup> is used as the baseline because it mainly performs adaptation through transferable feature alignment without explicitly modeling sensor-dependent scattering responses. The shared-only variant uses the proposed feature decomposition structure but retains only the shared path. We further evaluate the expertonly variant, fused-feature alignment, the shared-feature alignment loss $\begin{array} { r } { \mathcal { L } _ { a l i g n } ^ { s h } , } \end{array}$ the routing-domain loss $\mathcal { L } _ { r d } ,$ , and the expert balancing loss $\dot { \mathcal { L } } _ { b a l }$

As shown in Table IV, the shared-only variant obtains average $\mathrm { \ m A P _ { 5 0 } }$ values of 60.17% and 59.81% on the two tasks, which are close to the AsyFOD baseline. This shows that the shared path mainly preserves transferable features, while its improvement alone is limited. Applying alignment to the fused feature achieves average $\mathrm { \ m A P _ { 5 0 } }$ values of 58.05% and 60.09%, which are lower than those of the full model. This indicates that directly aligning the fused representation may suppress useful sensor-dependent responses retained by the expert paths. Moreover, removing $\mathcal { L } _ { a l i g n } ^ { s h }$ reduces the average performance to 49.53% and 52.44%, confirming the importance of aligning the shared representation.

TABLE IV  
ABLATION RESULTS OF THE PROPOSED COMPONENTS IN TERMS OF MAP<sub>50</sub> (%).
<table><tr><td rowspan="2">Methods</td><td colspan="6">FARAD-X → MiniSAR</td><td colspan="6">FARAD-Ka → MiniSAR</td></tr><tr><td>1-shot</td><td>3-shot</td><td>5-shot</td><td>8-shot</td><td>10-shot</td><td>Avg.</td><td>1-shot</td><td>3-shot</td><td>5-shot</td><td>8-shot</td><td>10-shot</td><td>Avg.</td></tr><tr><td>AsyFOD† baseline</td><td>43.39</td><td>49.09</td><td>66.10</td><td>72.07</td><td>71.23</td><td>60.38</td><td>41.30</td><td>43.30</td><td>66.01</td><td>69.72</td><td>73.21</td><td>58.71</td></tr><tr><td>Fused-feature alignment</td><td>40.41</td><td>46.17</td><td>66.73</td><td>70.10</td><td>66.82</td><td>58.05</td><td>42.64</td><td>47.30</td><td>66.58</td><td>71.98</td><td>71.94</td><td>60.09</td></tr><tr><td>Shared only</td><td>46.16</td><td>50.43</td><td>58.57</td><td>74.19</td><td>71.51</td><td>60.17</td><td>41.67</td><td>46.83</td><td>61.69</td><td>74.15</td><td>74.73</td><td>59.81</td></tr><tr><td>w/o  $\mathcal { L } _ { a l i g n } ^ { s h }$ </td><td>28.37</td><td>40.61</td><td>55.59</td><td>63.79</td><td>59.31</td><td>49.53</td><td>36.50</td><td>42.85</td><td>54.19</td><td>63.74</td><td>64.92</td><td>52.44</td></tr><tr><td>w/o  $\mathcal { L } _ { r d }$ </td><td>46.32</td><td>50.17</td><td>63.81</td><td>73.92</td><td>72.71</td><td>61.39</td><td>41.05</td><td>51.67</td><td>68.01</td><td>69.55</td><td>73.10</td><td>60.68</td></tr><tr><td>w/o  $\mathcal { L } _ { b a l }$ </td><td>45.23</td><td>51.14</td><td>68.12</td><td>74.59</td><td>71.19</td><td>62.05</td><td>28.64</td><td>51.53</td><td>69.32</td><td>72.55</td><td>74.63</td><td>59.33</td></tr><tr><td>Expert only</td><td>44.60</td><td>49.56</td><td>42.96</td><td>71.24</td><td>72.10</td><td>56.09</td><td>43.70</td><td>48.32</td><td>66.77</td><td>74.83</td><td>73.86</td><td>61.50</td></tr><tr><td>Our method (full)</td><td>47.19</td><td>51.19</td><td>68.98</td><td>76.06</td><td>73.58</td><td>63.40</td><td>47.26</td><td>51.60</td><td>69.32</td><td>75.72</td><td>75.42</td><td>63.86</td></tr></table>

The expert-only variant achieves average mA $\mathrm { \bf { \Omega } } \cdot \mathrm { \bf { P } } _ { 5 0 }$ values of 56.09% and 61.50%. Its performance is unstable on FARAD-X → MiniSAR, especially in the 5-shot setting. In comparison, the full model achieves 63.40% and 63.86% on the two tasks. These results show that shared features and expert features are complementary, and using either path alone cannot provide stable performance across different domains and shot numbers.

Without $\mathcal { L } _ { r d } .$ , the average $\mathrm { \ m A P _ { 5 0 } }$ decreases to 61.39% and 60.68%, respectively. This suggests that domain-aware routing helps the experts learn sensor-related response patterns. Removing $\mathcal { L } _ { b a l }$ reduces the average performance to 62.05% and 59.33%. The large decrease in the 1-shot FARAD-Ka setting shows that the balancing loss helps prevent the router from relying too much on only a few experts.

Overall, the full model achieves the best average performance on both tasks. The results confirm that combining the shared path, scattering-specific experts, shared-feature alignment, routing-domain supervision, and expert balancing provides more effective and stable few-shot SAR domain adaptation.

## D. Parameter Analysis

We further analyze the influence of the number of scattering-specific experts. Tables V and VI report the $\mathrm { \ m A P _ { 5 0 } }$ results with different expert numbers on FARAD-X → MiniSAR and FARAD-Ka → MiniSAR, respectively. When only one expert is used, the model degenerates into a single scattering-specific transformation, which limits its ability to represent diverse SAR scattering variations. Using multiple experts usually provides a more flexible representation than a single expert, but the benefit is not strictly monotonic under the few-shot setting. E=2 and E=4 achieve comparable results, while E=8 may introduce redundant routing uncertainty.

It can be observed that E = 2 and E = 4 achieve comparable overall performance. Specifically, E = 4 obtains the best average $\mathrm { \ m A P _ { 5 0 } }$ on FARAD-X → MiniSAR, while $E = 2$ achieves a slightly higher average result on FARAD-Ka → MiniSAR. However, the difference between E = 2 and E = 4 is small. The four-expert configuration achieves the best average performance on FARAD-X → MiniSAR while remaining comparable to the two-expert configuration on FARAD-Ka → MiniSAR. When the number of experts is further increased to $E = 8 ,$ , the performance does not consistently improve, suggesting that too many experts may introduce redundant routing uncertainty under the few-shot target-domain setting. Therefore, we set the number of scattering-specific experts to $E = 4$ by default in the main experiments.

TABLE V  
EFFECT OF THE NUMBER OF SCATTERING-SPECIFIC EXPERTS ON FARAD-X → MINISAR IN TERMS OF M $\mathrm { \ A P _ { 5 0 } }$ (%).
<table><tr><td rowspan="2">Experts E</td><td colspan="5">FARAD-X → MiniSAR</td><td rowspan="2"> $\mathbf { A v g . }$ </td></tr><tr><td>1-shot</td><td>3-shot</td><td>5-shot</td><td>8-shot</td><td>10-shot</td></tr><tr><td>1</td><td>44.97</td><td>52.21</td><td>68.35</td><td>70.49</td><td>68.77</td><td>60.96</td></tr><tr><td>2</td><td>44.54</td><td>53.48</td><td>68.41</td><td>72.14</td><td>75.46</td><td>62.81</td></tr><tr><td>4</td><td>47.19</td><td>51.19</td><td>68.98</td><td>76.06</td><td>73.58</td><td>63.40</td></tr><tr><td>8</td><td>43.12</td><td>50.04</td><td>67.68</td><td>73.35</td><td>77.48</td><td>62.33</td></tr></table>

TABLE VI

EFFECT OF THE NUMBER OF SCATTERING-SPECIFIC EXPERTS ON FARAD-KA → MINISAR IN TERMS OF $\mathrm { \bf M A P _ { 5 0 } }$ (%).
<table><tr><td rowspan="2">Experts E</td><td colspan="5">FARAD-Ka → MiniSAR</td><td rowspan="2">Avg.</td></tr><tr><td>1-shot</td><td>3-shot</td><td>5-shot</td><td>8-shot</td><td>10-shot</td></tr><tr><td>1</td><td>44.17</td><td>54.18</td><td>70.73</td><td>76.77</td><td>75.52</td><td>64.27</td></tr><tr><td>2</td><td>49.61</td><td>46.68</td><td>71.06</td><td>78.48</td><td>76.75</td><td>64.52</td></tr><tr><td>4</td><td>47.26</td><td>51.60</td><td>69.32</td><td>75.72</td><td>75.42</td><td>63.86</td></tr><tr><td>8</td><td>43.08</td><td>49.85</td><td>67.52</td><td>77.06</td><td>78.07</td><td>63.12</td></tr></table>

## V. CONCLUSION

We present a scattering-aware shared-specific feature decomposition framework for few-shot SAR domain adaptive object detection. Different from conventional featurealignment methods that mainly learn domain-invariant representations, the proposed method considers that cross-domain SAR shifts contain both transferable target structures and sensor-dependent scattering responses. To avoid over-aligning useful scattering-specific cues, the proposed framework introduces a mandatory shared path for source selection and domain alignment, and several soft-gated scattering-specific experts for adaptive residual compensation. Routing-domain auxiliary supervision and expert balancing regularization are further used to encourage domain-sensitive expert weighting and stable expert utilization.

Experiments on four bidirectional cross-sensor adaptation tasks demonstrate the effectiveness of the proposed method in both adaptation directions. Compared with $\mathbf { A s y F O D } ^ { \dagger }$ , the proposed method improves the average $\mathrm { \ m A P _ { 5 0 } }$ by 3.02, 5.16, 2.86, and 2.87 percentage points on the four tasks, respectively, and achieves the best performance in 19 of the 20 evaluated task-shot settings. The reverse-task results further show that the proposed method remains effective when MiniSAR is used as the source domain and the more cluttered FARAD domains are used as target domains, although adaptation under the extremely limited 1-shot setting remains sensitive to the representativeness of the target samples. Ablation studies confirm that aligning shared representations while preserving scattering-specific expert compensation provides more stable few-shot cross-sensor adaptation.

## REFERENCES

[1] Y. Shi, L. Du, and Y. Guo, “Unsupervised domain adaptation for sar target detection,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6372–6385, 2021.

[2] L. Kong, F. Gao, X. He, J. Wang, J. Sun, H. Zhou, and A. Hussain, “Fewshot class-incremental sar target recognition via orthogonal distributed features,” IEEE Transactions on Aerospace and Electronic Systems, vol. 61, no. 1, pp. 325–341, 2025.

[3] J. Li, C. Qu, and J. Shao, “Ship detection in sar images based on an improved faster r-cnn,” in 2017 SAR in Big Data Era: Models, Methods and Applications (BIGSARDATA), 2017, pp. 1–6.

[4] W. Li, W. Yang, Y. Hou, L. Liu, Y. Liu, and X. Li, “Saratr-x: Toward building a foundation model for sar target recognition,” IEEE Transactions on Image Processing, vol. 34, pp. 869–884, 2025.

[5] S. Wei, X. Zeng, Q. Qu, M. Wang, H. Su, and J. Shi, “Hrsid: A high-resolution sar images dataset for ship detection and instance segmentation,” IEEE Access, vol. 8, pp. 120 234–120 254, 2020.

[6] B. Zou, J. Qin, and L. Zhang, “Vehicle detection based on semanticcontext enhancement for high-resolution sar images in complex background,” IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1–5, 2022.

[7] S. Chen, R. Zhan, W. Wang, and J. Zhang, “Learning slimming sar ship object detector through network pruning and knowledge distillation,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 1267–1282, 2021.

[8] Q. Zhang, R. Cong, C. Li, M.-M. Cheng, Y. Fang, X. Cao, Y. Zhao, and S. Kwong, “Dense attention fluid network for salient object detection in optical remote sensing images,” IEEE Transactions on Image Processing, vol. 30, pp. 1305–1317, 2021.

[9] L. Huang, W. Zhao, Y. Liu, D. Yang, A. W.-C. Liew, and Y. You, “An evidential multi-target domain adaptation method based on weighted fusion for cross-domain pattern classification,” IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 10, pp. 14 218– 14 232, 2024.

[10] M. Xu, M. Wu, K. Chen, C. Zhang, and J. Guo, “The eyes of the gods: A survey of unsupervised domain adaptation methods based on remote sensing data,” Remote Sensing, vol. 14, no. 17, p. 4380, 2022.

[11] X. Zhang, S. Zhang, Z. Sun, C. Liu, Y. Sun, K. Ji, and G. Kuang, “Cross-sensor sar image target detection based on dynamic feature discrimination and center-aware calibration,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–17, 2025.

[12] S. Zhao, Z. Zhang, T. Zhang, W. Guo, and Y. Luo, “Transferable sar image classification crossing different satellites under open set condition,” IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1–5, 2022.

[13] M. Wang and W. Deng, “Deep visual domain adaptation: A survey,” Neurocomputing, vol. 312, pp. 135–153, 2018.

[14] G. Wilson and D. J. Cook, “A survey of unsupervised deep domain adaptation,” ACM Transactions on Intelligent Systems and Technology (TIST), vol. 11, no. 5, pp. 1–46, 2020.

[15] P. Singhal, R. Walambe, S. Ramanna, and K. Kotecha, “Domain adaptation: Challenges, methods, datasets, and applications,” IEEE Access, vol. 11, pp. 6973–7020, 2023.

[16] S. Zhao, Y. Kang, H. Yuan, G. Wang, H. Wang, S. Xiong, and Y. Luo, “Fsdaod: Few-shot domain adaptation object detection for heterogeneous sar image,” Science of Remote Sensing, vol. 11, p. 100202, 2025.

[17] Y. Gao, K.-Y. Lin, J. Yan, Y. Wang, and W.-S. Zheng, “Asyfod: An asymmetric adaptation paradigm for few-shot domain adaptive object detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 3261–3271.

[18] A. Kumar, P. Sattigeri, K. Wadhawan, L. Karlinsky, R. Feris, B. Freeman, and G. Wornell, “Co-regularized alignment for unsupervised domain adaptation,” Advances in neural information processing systems, vol. 31, 2018.

[19] L. Zhang, B. Zhang, B. Shi, J. Fan, and T. Chen, “Few-shot cross-domain object detection with instance-level prototype-based meta-learning,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 10, pp. 9078–9089, 2024.

[20] C. Sakaridis, D. Dai, and L. Van Gool, “Semantic foggy scene understanding with synthetic data,” International Journal of Computer Vision, vol. 126, no. 9, pp. 973–992, 2018.

[21] Y. Shi, Y. Li, L. Du, Y. Du, and Y. Guo, “Unsupervised domain adaptative sar target detection based on feature decomposition and uncertainty-guided self-training,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 17, pp. 20 265– 20 283, 2024.

[22] S. Zhao, Y. Luo, T. Zhang, W. Guo, and Z. Zhang, “A feature decomposition-based method for automatic ship detection crossing different satellite sar images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–15, 2022.

[23] Y. Li, X. Li, W. Li, Q. Hou, L. Liu, M.-M. Cheng, and J. Yang, “Sardet-100k: Towards open-source benchmark and toolkit for large-scale sar object detection,” Advances in Neural Information Processing Systems, vol. 37, pp. 128 430–128 461, 2024.

[24] M. Weiss, “Analysis of some modified cell-averaging cfar processors in multiple-target situations,” IEEE Transactions on Aerospace and Electronic Systems, vol. AES-18, no. 1, pp. 102–114, 1982.

[25] A. Frery, H.-J. Muller, C. Yanasse, and S. Sant’Anna, “A model for extremely heterogeneous clutter,” IEEE Transactions on Geoscience and Remote Sensing, vol. 35, no. 3, pp. 648–659, 1997.

[26] C. Tison, J.-M. Nicolas, F. Tupin, and H. Maitre, “A new statistical model for markovian classification of urban areas in high-resolution sar images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 42, no. 10, pp. 2046–2057, 2004.

[27] Z. Wang, L. Du, J. Mao, B. Liu, and D. Yang, “Sar target detection based on ssd with data augmentation and transfer learning,” IEEE Geoscience and Remote Sensing Letters, vol. 16, no. 1, pp. 150–154, 2019.

[28] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” Advances in neural information processing systems, vol. 25, 2012.

[29] O. Kechagias-Stamatis and N. Aouf, “Automatic target recognition on synthetic aperture radar imagery: A survey,” IEEE Aerospace and Electronic Systems Magazine, vol. 36, no. 3, pp. 56–81, 2021.

[30] C. Oliver and S. Quegan, Understanding synthetic aperture radar images. SciTech Publishing, 2004.

[31] C. V. Jakowatz, D. E. Wahl, P. H. Eichel, D. C. Ghiglia, and P. A. Thompson, Spotlight-mode synthetic aperture radar: a signal processing approach: a signal processing approach. Springer Science & Business Media, 2012.

[32] S. qi Huang, D. zhi Liu, G. qing Gao, and X. jian Guo, “A novel method for speckle noise reduction and ship target detection in sar images,” Pattern Recognition, vol. 42, no. 7, pp. 1533–1542, 2009. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S0031320309000417

[33] J. Zhou, B. Peng, J. Xie, B. Peng, L. Liu, and X. Li, “Conditional random field-based adversarial attack against sar target detection,” IEEE Geoscience and Remote Sensing Letters, vol. 21, pp. 1–5, 2024.

[34] J. Li, Z. Yu, Z. Du, L. Zhu, and H. T. Shen, “A comprehensive survey on source-free domain adaptation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 8, pp. 5743–5762, 2024.

[35] Y. Chen, W. Li, C. Sakaridis, D. Dai, and L. Van Gool, “Domain adaptive faster r-cnn for object detection in the wild,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 3339– 3348.

[36] Y. Guo, L. Du, and G. Lyu, “Sar target detection based on domain adaptive faster r-cnn with small training data size,” Remote Sensing, vol. 13, no. 21, p. 4202, 2021.

[37] B. Zou, J. Qin, and L. Zhang, “Cross-scene target detection based on feature adaptation and uncertainty-aware pseudo-label learning for high

resolution sar images,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 200, pp. 173–190, 2023.

[38] Y. Nakamura, Y. Ishii, Y. Maruyama, and T. Yamashita, “Fewshot adaptive object detection with cross-domain cutmix,” in Asian Conference on Computer Vision, 2022. [Online]. Available: https: //api.semanticscholar.org/CorpusID:251953274

[39] Y. Yang and S. Soatto, “Fda: Fourier domain adaptation for semantic segmentation,” 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4084–4094, 2020. [Online]. Available: https://api.semanticscholar.org/CorpusID:215745272

[40] Q. Cai, Y. Pan, C.-W. Ngo, X. Tian, L. yu Duan, and T. Yao, “Exploring object relation in mean teacher for cross-domain detection,” 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11 449–11 458, 2019. [Online]. Available: https://api.semanticscholar.org/CorpusID:131774688

[41] R. Ramamonjison, A. Banitalebi-Dehkordi, X. Kang, X. Bai, and Y. Zhang, “Simrod: A simple adaptation method for robust object detection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 3570–3579.

[42] H. Zhou, F. Jiang, and H. Lu, “Ssda-yolo: Semisupervised domain adaptive yolo for cross-domain object detection,” ArXiv, vol. abs/2211.02213, 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID:253370411

[43] G. Jocher et al., “ultralytics/yolov5.” [Online]. Available: https: //github.com/ultralytics/yolov5

[44] Y. Song, T.-Y. Wang, P. Cai, S. K. Mondal, and J. P. Sahoo, “A comprehensive survey of few-shot learning: Evolution, applications, challenges, and opportunities,” ACM Computing Surveys, vol. 55, pp. 1 – 40, 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID: 248798765

[45] X. Wang, T. E. Huang, T. Darrell, J. E. Gonzalez, and F. Yu, “Frustratingly simple few-shot object detection,” ArXiv, vol. abs/2003.06957, 2020. [Online]. Available: https: //api.semanticscholar.org/CorpusID:212725414

[46] Y. Gao, L. Yang, Y. Huang, S. Xie, S. Li, and W. Zheng, “Acrofod: An adaptive method for cross-domain few-shot object detection,” in European Conference on Computer Vision, 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID:252438651

[47] C. Liu, Y. He, X. Zhang, Y. Wang, Z. Dong, and H. Hong, “Cs-fsdet: A few-shot sar target detection method for cross-sensor scenarios,” Remote Sensing, vol. 17, no. 16, p. 2841, 2025.

[48] F. Shen and J. Tang, “Imagpose: A unified conditional framework for pose-guided person generation,” Advances in neural information processing systems, vol. 37, pp. 6246–6266, 2024.

[49] F. Shen, H. Ye, J. Zhang, C. Wang, X. Han, and Y. Wei, “Advancing pose-guided image synthesis with progressive conditional diffusion models,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/ forum?id=rHzapPnCgT

[50] F. Shen, X. Jiang, X. He, H. Ye, C. Wang, X. Du, Z. Li, and J. Tang, “Imagdressing-v1: Customizable virtual dressing,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 7, 2025, pp. 6795–6804.

[51] F. Shen, H. Ye, S. Liu, J. Zhang, C. Wang, X. Han, and Y. Wei, “Boosting consistency in story visualization with rich-contextual conditional diffusion models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 7, 2025, pp. 6785–6794.

[52] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, “Adaptive mixtures of local experts,” Neural computation, vol. 3, no. 1, pp. 79–87, 1991.

[53] X. Lin, J. Peng, Z. Gan, J. Zhu, and J. Liu, “Yolo-master: Moeaccelerated with specialized transformers for enhanced real-time detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 18 440–18 449.

[54] Sandia National Laboratories, “FARAD SAR public release data,” https://www.sandia.gov/radar/ pathfinder-radar-isr-and-synthetic-aperture-radar-sar-systems/ complex-data-clone/, 2015, accessed: Jul. 9, 2026.

[55] “SANDIA miniSAR complex imagery dataset,” https://www.sandia.gov/radar/ pathfinder-radar-isr-and-synthetic-aperture-radar-sar-systems/ complex-data-clone/, 2005, accessed: Jul. 9, 2026.

[56] X. Lin, B. Zhang, F. Wu, C. Wang, Y. Yang, and H. Chen, “Sived: A sar image dataset for vehicle detection based on rotatable bounding box,” Remote Sensing, vol. 15, no. 11, p. 2825, 2023.