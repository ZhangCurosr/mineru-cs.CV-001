![](images/3e1a8ae8729770358634cb7056612d26c9d0d3d45fb4461ba3f1b36fa3574053.jpg)

# GaussianDWM++: Language-Grounded 3D Gaussian Driving World Model for Unified Scene Understanding, Editing, and Multi-Modal Generation

Tianchen Deng, Xuefeng Chen, Shuang Wu, Qu Chen, Jiajun Zhu, Bo Dai, Jianfei Yang, Hesheng Wang, Senior Member, IEEE

Abstract—Driving World Models (DWMs) have recently advanced rapidly with generative models, yet most existing methods mainly focus on conditional scene generation and lack explicit 3D scene understanding, language-grounded reasoning, and controllable 4D editing capabilities. Moreover, commonly used point cloud, occupancy, or BEV representations make it difficult to achieve fine-grained alignment between textual information and the underlying 3D scene structure. To address these limitations, we propose a foundation-feature Gaussian driving world model that unifies scene understanding, language-grounded reasoning, controllable 4D editing, and multi-modal generation within a single framework. Specifically, we introduce a foundationfeature Gaussian tokenizer that directly distills Qwen visuallanguage features into 3D Gaussian primitives, building a compact open-vocabulary Gaussian semantic field. We further design a geometry-aware Gaussian adapter that combines importanceaware hierarchical selection with text-conditioned Perceiver-style cross-attention to aggregate dense Gaussian primitives into compact world tokens. To improve representation compatibility, we introduce a KL-based Gaussian–image distribution alignment objective that aligns Gaussian world tokens with foundation image tokens. Based on the aligned Gaussian representation, our framework further supports instruction-controllable scene editing, including weather-conditioned generation and dynamic vehicle manipulation, while preserving temporal, geometric, and semantic consistency. We further construct NuScenes-GSQA, a large-scale dataset containing 1000 nuScenes driving scenes represented by foundation-feature 3D Gaussians, together with Gaussian-aware QA annotations derived from NuInteract. Extensive experiments on broader driving benchmarks demonstrate that our method achieves state-of-the-art performance across scene understanding, visual grounding, planning-oriented reasoning, and controllable 4D generation tasks. We will release the code and datasets publicly on https://github.com/dtc111111/GaussianDWM.

## I. INTRODUCTION

Driving World Models (DWMs) [1]–[3] have become increasingly important for autonomous driving, owing to their ability to predict future scene evolution, synthesize realistic

GaussianDWM++ GaussianDWM Max of Existing 20 Method

Fig. 1: Overall performance on understanding and generation benchmarks. GaussianDWM++ achieves state-ofthe-art performance in both scene understanding and scene generation. Radar radii are normalized as $\mathrm { M e t r i c } _ { i } / \mathrm { M e t r i c } _ { \mathrm { m a x } }$ for understanding and $\mathrm { M e t r i c } _ { \mathrm { m i n } } / \mathrm { M e t r i c } _ { i }$ for generation, so that larger radii indicate better performance. Boxed values show the original GaussianDWM++ results.

driving scenarios, and provide simulation data for risk forecasting, route optimization, and corner-case training. Existing DWMs mainly focus on generating future observations conditioned on input data, such as images [4], videos, point clouds [5], or BEV/occupancy representations [6]. These modalities effectively describe the visual or geometric evolution of driving environments, but they are still limited in their ability to interpret, describe, and reason about the underlying scene. As a result, most existing DWMs remain difficult to query with natural language and cannot directly provide contextual understanding for tasks such as visual question answering, 2D/3D grounding, planning-oriented reasoning, or instruction-conditioned scene editing.

Recent advances in Large Language Models (LLMs) and Vision-Language Models (VLMs) [7] have demonstrated strong capabilities in semantic understanding, commonsense reasoning, and language-grounded visual perception. These developments suggest a promising direction for autonomous driving: combining the generative capability of DWMs with the scene understanding and reasoning capability of VLMs. Pioneering works such as HERMES [8] have explored this direction by unifying scene understanding and scene generation within driving world models. However, these methods typically rely on BEV, depth, or occupancy features to represent 3D spatial information. Such representations are effective for structured driving perception, but they often establish only feature-level alignment between language and space, making it difficult to achieve fine-grained correspondence between textual concepts, image observations, and explicit 3D scene geometry.

![](images/62c9777e553aec5a7bff5fd1d08460ce0bd2b72e8b3501caa0e3a0d277c5baed.jpg)  
Fig. 2: We propose the unified 3D Gaussian-based world model framework that achieves comprehensive scene understanding and scene generation for driving scenarios. It efficiently encodes complex scenes, samples task-relevant information, and handles diverse question-answering tasks. Moreover, by leveraging the extracted world knowledge, our framework guides the generative model to perform accurate spatial and temporal scene generation.

To address these limitations, we propose GaussianDWM++, a novel foundation-feature 3D Gaussian driving world model that unifies scene understanding, language-grounded reasoning, controllable 4D scene editing, and multi-modal generation within a single 3D Gaussianbased framework, as summarized in Fig. 2. Different from point-, BEV-, or occupancy-based representations, 3D Gaussian Splatting (3DGS) provides an explicit, continuous, and differentiable scene representation, where each Gaussian primitive carries spatial position, anisotropic geometry, opacity, appearance, and semantic attributes. This primitivelevel representation offers several advantages for driving world modeling. First, it naturally preserves explicit 3D spatial structure and multi-view consistency. Second, it supports differentiable rendering, enabling image-, depth-, and feature-level supervision from multi-view observations. Third, it provides a flexible carrier for foundation visuallanguage features, allowing language semantics to be directly embedded into 3D scene primitives rather than being indirectly aligned through BEV or image-level features.

Based on this representation, we first introduce a foundation-feature Gaussian tokenizer that directly distills Qwen visual-language features into 3D Gaussian primitives. Compared with the previous LangSplat-dependent language Gaussian construction, our tokenizer learns a compact and trainable open-vocabulary Gaussian semantic field from multiview observations. This design directly connects foundation visual-language features with explicit 3D Gaussian primitives, thereby improving the alignment among image observations, language semantics, and 3D spatial geometry.

However, directly feeding all Gaussian primitives into an LLM or a generative model is impractical, since a driving scene may contain tens or hundreds of thousands of Gaussians. Moreover, dense Gaussian primitives are highly redundant, and not all primitives are equally relevant to a given query or generation condition. To overcome this challenge, we design a geometry-aware Gaussian adapter to transform dense Gaussian primitives into compact world tokens. Specifically, we first introduce an importance-aware hierarchical Gaussian selection strategy that evaluates Gaussian primitives according to their geometric quality and driving relevance. We then employ a Perceiver-style cross-attention module in which the pooled text embedding modulates a small set of learned queries. These text-conditioned queries attend to the selected Gaussian tokens and aggregate them into compact LLM-compatible world tokens. This design enables efficient token extraction while preserving task-relevant 3D spatial and semantic information.

To further improve representation compatibility, we introduce a KL-based Gaussian–image distribution alignment objective. Instead of simply projecting Gaussian tokens with independent MLPs, we regularize the compact Gaussian world tokens toward the Qwen image-token distribution from the same scene. This image-guided objective transfers semantically rich foundation features to the Gaussian tokens while preserving their explicit 3D structure. In this way, the proposed adapter not only compresses dense 3DGS primitives into compact tokens, but also makes the resulting tokens more compatible with the visual representation consumed by the LLM.

In addition to the model design, we observe that existing driving language datasets are mainly built upon image-level,

BEV-level, or object-level inputs, while large-scale datasets that directly support 3DGS-based scene understanding and reasoning remain scarce. To fill this gap, we construct a largescale 3D Gaussian driving scene dataset from nuScenes [9], containing 1000 outdoor urban scenes represented with highquality 3D Gaussian primitives. The dataset provides explicit geometry, appearance, and foundation-feature representations for each scene, enabling generalizable 3D Gaussian-based scene understanding with LLMs. Furthermore, using NuInteract annotations, we build a large-scale 3DGS-based QA dataset covering scene description, visual grounding, spatial reasoning, planning-oriented reasoning, and 3D-aware language interaction. To the best of our knowledge, this is among the first large-scale 3D Gaussian datasets for outdoor urban driving scene understanding with LLMs. This dataset not only supports the training and evaluation of our framework, but also provides a useful benchmark for future 3D Gaussian-based driving world models.

Furthermore, we extend the original dual-condition generation framework to instruction-controllable 4D driving scene editing and generation. The proposed generation module leverages high-level world knowledge from the VLM together with low-level geometric and image conditions from the Gaussian scene representation. Beyond spatial and temporal scene generation, our framework supports more diverse and controllable 4D editing modes, including weather-conditioned generation and dynamic vehicle manipulation. These editing operations are performed while preserving temporal coherence, geometric consistency, and semantic fidelity, enabling the model to function not only as a scene generator, but also as an interpretable and controllable driving world simulator.

Overall, the contributions of our paper are summarized as follows:

• We propose a novel foundation-feature 3D Gaussian driving world model that unifies scene understanding, language-grounded reasoning, controllable 4D scene editing, and multi-modal generation within a single 3D Gaussian-based framework. To the best of our knowledge, this is the first unified 3D Gaussian driving world model for scene understanding and generation.

• We introduce a foundation-feature Gaussian tokenizer that directly distills Qwen visual-language features into 3D Gaussian primitives, building a compact and trainable Gaussian semantic field. We design a geometryaware Gaussian adapter that combines importanceaware hierarchical Gaussian selection with Perceiver-style cross-attention, where pooled text embeddings modulate learned queries to aggregate dense Gaussian primitives into compact world tokens.

• We propose a KL-based Gaussian–image distribution alignment objective that regularizes Gaussian world tokens toward the Qwen image-token distribution, improving their compatibility with foundation visual representations.

• We extend the generation module to instructioncontrollable 4D driving scene editing and generation, supporting weather-conditioned generation and dynamic vehicle manipulation while preserving temporal, geometric, and semantic consistency.

• We construct a large-scale 3D Gaussian driving scene dataset, NuScenes-GSQA, containing 1000 outdoor urban scenes from nuScenes represented with high-quality 3D Gaussian primitives. Using NuInteract annotations, we further build a large-scale 3DGS-based questionanswering dataset for scene description, visual grounding, spatial reasoning, planning-oriented reasoning, and 3Daware language interaction. Extensive experiments on broader driving benchmarks demonstrate the effectiveness of our framework on scene understanding and 4D generation tasks.

This paper is an extension of our previous conference version CVPR 2026 [10]. Compared with the conference paper, this journal version makes several substantial extensions. First, we replace the LangSplat-based language Gaussian construction with a foundation-feature Gaussian tokenizer that directly distills Qwen features into 3D Gaussian primitives. Second, we redesign the Gaussian tokenization module by introducing an importance-aware hierarchical sampling strategy and a Perceiver-style cross-attention adapter, in which pooled text embeddings modulate learned queries for compact and geometry-aware aggregation of dense Gaussian primitives. Third, we introduce a KL-based Gaussian–image distribution alignment objective to align Gaussian world tokens with foundation image-token distributions. Fourth, we construct a large-scale 3D Gaussian driving scene dataset and a 3DGSbased question-answering dataset to support generalizable 3D Gaussian-based scene understanding and reasoning with LLMs. Fifth, we significantly extend the generation capability from basic spatiotemporal generation to instructioncontrollable 4D scene editing, including weather control and dynamic vehicle editing. Finally, we provide more comprehensive experimental evaluations, including additional datasets, stronger baselines, more generation settings, and systematic ablation studies, to further validate the effectiveness and generalization ability of the proposed framework.

## II. RELATED WORK

Driving World Models. Driving world models [11] have attracted increasing attention in autonomous driving due to their ability to model scene evolution, predict future observations, and synthesize controllable simulation data. Existing methods mainly rely on image, video, BEV, occupancy, or point-cloud conditions for scene generation. GAIA-1 [12] introduces an autoregressive framework for driving video generation. Drive-Dreamer [4] generates realistic driving scenes conditioned on 3D structural information, demonstrating the importance of geometric guidance for autonomous driving generation. MagicDrive [13] presents a street-view synthesis framework with precise 3D controls, such as camera poses and road maps, using cross-view attention. DreamDrive [14] further combines video diffusion with hybrid Gaussian scene representations to synthesize 4D driving scenes with 3D-consistent dynamic video rendering. UniScene [6] proposes an occupancy-centric framework to unify semantic, visual, and LiDAR generation. Recent works further explore controllable generation, future prediction, closed-loop simulation, and 4D world modeling for autonomous driving.

![](images/039fbba7d70651e5525e30959b469ff7e2b090b55f7ef68d5d1ec137d7e67fc0.jpg)  
Fig. 3: System Overview. GaussianDWM++ builds a foundation-feature Gaussian field by distilling Qwen visual features into explicit 3D Gaussian primitives. A geometry-aware adapter first performs driving-aware coarse selection and attribute projection. It then combines a pooled text embedding with learned queries and aggregates the selected Gaussian tokens through Perceiver style cross-attention. A Gaussian–image distribution alignment objective regularizes the resulting world tokens toward the foundation image-token distribution. The compact Gaussian world tokens are shared by the LLM-based scene understanding branch and the dual-condition scene generation branch for high-level world knowledge.

Although these methods have achieved impressive progress in generation quality and controllability, most existing DWMs primarily focus on synthesizing future observations rather than interpreting and reasoning about the driving environment. As a result, they usually lack explicit support for language-grounded querying, 2D/3D visual grounding, planning-oriented reasoning, and instruction-conditioned scene editing. Pioneering efforts such as HERMES [8] and UniFuture [15] attempt to bridge scene understanding and scene generation by integrating VLMs or language features into driving world models. However, their spatial representations are mainly built upon BEV, depth, or occupancy features, which provide only indirect feature-level alignment between language and 3D space. In contrast, our method constructs a unified DWM upon foundation-feature 3D Gaussians, where visual-language semantics are embedded into explicit 3D primitives and further used for both scene understanding and controllable 4D generation.

Large Language Models for Autonomous Driving. Large Language Models (LLMs) and Vision-Language Models (VLMs) have demonstrated strong generalization, semantic understanding, and reasoning capabilities across a wide range of tasks, including scene understanding [16]–[20], visual question answering, visual grounding, and embodied decisionmaking. In autonomous driving, DriveGPT4 [21] processes front-view videos to predict vehicle actions and provide natural language justifications. DriveLM [22] formulates autonomous driving as graph-based visual question answering and reasoning. NuInteract [23] further integrates large vision-language models with a spatial processor using learnable queries, and provides a large-scale multi-view image-language benchmark containing dense scene captioning, visual grounding, and diverse interactive reasoning tasks. These works show that language supervision and world knowledge can significantly improve the interpretability and reasoning ability of autonomous driving systems.

HERMES [8] is closely related to our work in that it integrates scene representation with a VLM for joint scene understanding and generation, but it still adopts a BEVbased spatial representation. Different from these methods, our approach embeds foundation visual-language features into 3D Gaussian primitives and further aggregates them into compact, language-compatible world tokens, enabling unified 3D-aware reasoning, scene understanding, and multi-modal generation.

Neural Rendering and 4D Urban Scene Reconstruction. With the emergence of Neural Radiance Fields (NeRF) [24] and 3D Gaussian Splatting (3DGS) [25], neural scene representations have been widely adopted in novel view synthesis, robotic mapping and localization [26]–[31], virtual and augmented reality [32]–[37], and autonomous driving [38], [39]. For large-scale urban scenes, early NeRF-based methods, such as NSG [38], SUDS [40], ProSGNeRF [41], EmerNeRF [42], and FreeDriveRF [43], model dynamic driving environments through scene graphs, dynamic-static decomposition, optical flow, or other motion cues. These methods demonstrate the effectiveness of neural fields for reconstructing complex urban scenes, but their volumetric rendering process often results in high computational costs.

More recently, 3DGS-based methods have been proposed to improve rendering efficiency and scalability. PVG [44] introduces periodic vibration-based temporal dynamics to unify static and dynamic elements without manual annotations. Street Gaussians [45], Driving Gaussian [46], and DeSiRe-GS [47] explicitly separate dynamic foreground objects from static backgrounds for urban scene reconstruction. LESSON [48] explores teacher-guided diffusion for generating 3D Gaussian splats using only 2D supervision. STORM [49] proposes a feed-forward Transformer architecture to infer dynamic Gaussians and their velocities, enabling efficient large-scale outdoor scene reconstruction. DrivingForward [50] performs feed-forward reconstruction from sparse surroundview inputs using self-supervised pose and depth estimation. MUDG [51] and Dist-4D [52] further extend urban scene synthesis to multi-modal generation by predicting both RGB and depth observations. Although these methods significantly advance 4D reconstruction and novel view synthesis for urban scenes, they mainly treat NeRF or 3DGS as rendering representations. They generally lack language-grounded scene understanding, explicit reasoning ability, and unified integration with driving world models.

Scene Representations for World Models. Scene representation is a fundamental component of autonomous driving systems and driving world models [53], [54]. Existing representations include image/video features [1], [12], point clouds [55], BEV maps [8], [56], occupancy fields [6], [57], neural radiance fields [58], and 3D Gaussian primitives [10], [59], [60]. Image and video features preserve rich appearance information and are naturally compatible with 2D generative models, but they lack explicit 3D structure and metric spatial grounding. Point clouds provide direct geometric measurements, yet they are sparse, irregular, and usually contain limited texture and semantic information. BEV and occupancy representations are highly structured and well suited for detection, prediction, planning, and map-based reasoning, but they discretize the 3D world and may lose fine-grained multi-view appearance and continuous geometry. Neural fields provide continuous scene modeling and differentiable rendering, but they are often computationally expensive for large-scale dynamic driving environments.

Compared with these representations, 3DGS provides an explicit, continuous, and efficient primitive-based scene representation. Each Gaussian primitive carries spatial position, anisotropic geometry, opacity, appearance, and potentially semantic or foundation-model features. This makes 3DGS particularly suitable for driving world modeling, since it can simultaneously support differentiable rendering, multiview geometric consistency, efficient scene reconstruction, and primitive-level semantic grounding. Recent 3DGS-based urban scene methods [44]–[47], [49], [50] have demonstrated strong reconstruction and rendering efficiency in large-scale outdoor environments. Meanwhile, language- or foundation-featurebased 3D representations, such as LERF [61], ConceptFusion [62], OpenScene [63], LangSplat [64], and Gaussian-VLM [65] further show the potential of connecting 3D scene representations with visual-language foundation models for understanding and language-grounded interaction.

## III. METHOD

## A. Overview

We propose GaussianDWM++, a foundation-feature 4D Gaussian driving world model for unified scene understanding, language-grounded reasoning, controllable 4D editing, and multi-modal scene generation. The input to our method consists of images $\mathcal { T } = \{ I _ { v } \} _ { v = 1 } ^ { V }$ , Gaussian ellipsoids $\mathcal { G } =$ $\{ G _ { i } \} _ { i = 1 } ^ { N }$ , and query text q. Our goal is to construct a compact and semantically aligned 3D world representation that can be consumed by both large language models and generative models. The overall pipeline is illustrated in Fig. 3.

Our framework consists of four main components. First, a foundation-feature Gaussian tokenizer distills Qwen visuallanguage features into 3D Gaussian primitives, producing an open-vocabulary semantic field with explicit 3D alignment (Sec. III-B). Second, a geometry-aware adapter combines hierarchical selection, text-conditioned Perceiver-style crossattention, and Gaussian–image distribution alignment to produce compact world tokens $\mathcal { Z } _ { G }$ (Sec. III-C). These tokens are injected into the LLM with image and text tokens for scene understanding (Sec. III-D) and are shared with the generation branch for spatiotemporal generation and instructioncontrollable 4D editing (Sec. III-E).

## B. Foundation-feature Gaussian Tokenizer

The foundation-feature Gaussian tokenizer converts multiview driving observations into an explicit 3D Gaussian semantic field. Different from the conference version that relies on pre-computed LangSplat features, we directly distill Qwen visual-language features into 3D Gaussian primitives. In this way, each Gaussian primitive is not only used for differentiable rendering, but also serves as an explicit 3D carrier of openvocabulary foundation features.

3D Gaussian Scene Representation. Given a reconstructed driving scene, we represent the Gaussian field as

$$
\mathcal { G } = \{ G _ { i } = ( x _ { i } , s _ { i } , r _ { i } , o _ { i } , c _ { i } , z _ { i } ) \} _ { i = 1 } ^ { N } .\tag{1}
$$

Here, $x _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ is the Gaussian center, $s _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ is the anisotropic scale, $r _ { i } \in \mathbb { R } ^ { 4 }$ is the rotation quaternion, $o _ { i } \in \mathbb { R }$ is the opacity, $c _ { i }$ denotes the appearance attribute, and $z _ { i } \in$ $\mathbb { R } ^ { d _ { z } }$ denotes the compact foundation-feature code. For each timestamp, we transform Gaussian centers into the current ego coordinate system, so that the Gaussian field is represented under a driving-centric coordinate frame. The opacity and scale are normalized to avoid unstable primitives during feature rendering and token extraction.

Foundation Feature Extraction. For each multi-view image $I _ { v } ,$ , we use a frozen foundation visual encoder to extract dense visual-language features:

$$
\begin{array} { r } { F _ { v } = \mathcal { E } _ { F } ( I _ { v } ) . } \end{array}\tag{2}
$$

Here, $F _ { v } \in \mathbb R ^ { H ^ { \prime } \times W ^ { \prime } \times d _ { f } }$ denotes the dense foundation feature map, and $d _ { f }$ is the feature dimension. In practice, $\mathcal { E } _ { F }$ is instantiated with Qwen visual encoders. These frozen features provide semantically rich supervision for constructing the 3D Gaussian semantic field.

Foundation-Feature Distillation. To lift 2D foundation features into the 3D Gaussian field, we associate a compact feature code $z _ { i }$ with each Gaussian primitive. Since the original foundation feature dimension is high, we decode the compact code into a full-dimensional feature before rendering:

$$
\hat { f } _ { i } = { \cal D } _ { F } ( z _ { i } ) .\tag{3}
$$

The decoded feature $\hat { f } _ { i } ~ \in ~ \mathbb { R } ^ { d _ { f } }$ is then rendered to each camera view using the same Gaussian splatting process as RGB rendering. For a pixel u in camera view $v ,$ the rendered foundation feature is:

$$
\hat { F } _ { v } ( u ) = \sum _ { i \in \mathcal { N } _ { v } ( u ) } \omega _ { i , u } ^ { v } \hat { f } _ { i } .\tag{4}
$$

Here, $\mathcal { N } _ { v } ( u )$ denotes the Gaussians contributing to pixel $u ,$ $\omega _ { i , u } ^ { v }$ is the corresponding alpha-compositing weight, and $\Omega _ { v }$ is the set of valid pixels in view $v .$ We supervise the rendered features with a pixel-level L1 distillation loss:

$$
\mathcal { L } _ { \mathrm { d i s t i l l } } = \frac { 1 } { \sum _ { v = 1 } ^ { V } \left| \Omega _ { v } \right| } \sum _ { v = 1 } ^ { V } \sum _ { u \in \Omega _ { v } } \left. \hat { F } _ { v } ( u ) - F _ { v } ( u ) \right. _ { 1 } .\tag{5}
$$

This objective lifts foundation visual-language features from 2D observations into the explicit 3D Gaussian field, enabling primitive-level semantic grounding.

Compact Feature Autoencoding. To reduce storage cost, we further adopt a compact feature autoencoding strategy. Given the high-dimensional foundation feature $f _ { i } ,$ , we encode it into a low-dimensional latent code and reconstruct it through a lightweight decoder:

$$
z _ { i } = \mathcal { E } _ { z } ( f _ { i } ) , \quad \tilde { f } _ { i } = \mathcal { D } _ { F } ( z _ { i } ) .\tag{6}
$$

The reconstruction objective is a squared L2 loss:

$$
\mathcal { L } _ { \mathrm { a e } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left. \tilde { f } _ { i } - f _ { i } \right. _ { 2 } ^ { 2 } .\tag{7}
$$

Only the compact code $z _ { i }$ is stored in each Gaussian primitive. This design keeps the Gaussian semantic field compact while preserving foundation-level visual-language semantics.

The resulting foundation-feature Gaussian field explicitly couples 3D geometry, appearance, opacity, and visuallanguage features. The adapter in Sec. III-C then selects and aggregates these primitives into compact world tokens.

## C. Geometry-aware Gaussian Adapter

The foundation-feature Gaussian field in Sec. III-B contains explicit geometry, appearance, opacity, and foundation visuallanguage features. However, a driving scene usually contains tens or hundreds of thousands of Gaussian primitives, which cannot be directly consumed by an LLM. We therefore introduce a geometry-aware Gaussian adapter to convert the dense Gaussian field into a compact set of LLM-compatible world tokens. The adapter consists of three components: driving-aware coarse selection, attribute-aware Gaussian projection, and text-conditioned Perceiver-style cross-attention with Gaussian–image distribution alignment.

Driving-aware Coarse Selection. Given the foundationfeature Gaussian field, the coarse selector aims to preserve a compact yet spatially representative candidate set. Different from the previous distance-biased quality score, we introduce a driving-aware importance score that jointly considers intrinsic Gaussian quality and trajectory relevance. We first define the intrinsic quality of each Gaussian as

$$
Q _ { i } = \mathrm { N o r m } _ { [ 0 , 1 ] } \left( \log \left( 1 + \sigma \left( o _ { i } \right) \left( \left. s _ { i } \right. _ { 1 } + 1 \right) \rho _ { i } \right) \right) .\tag{8}
$$

The rotation validity term $\rho _ { i }$ is defined as

$$
\rho _ { i } = \frac { 1 } { 1 + | \| r _ { i } \| _ { 2 } - 1 | } .\tag{9}
$$

Here, $\mathrm { { N o r m } _ { [ 0 , 1 ] } }$ denotes min-max normalization within the current scene. The quality term favors Gaussians with high opacity, reasonable scale, and valid rotation, while avoiding an explicit bias toward the ego origin.

To encourage the selected Gaussians to cover regions that are important for future driving, we further define a trajectory relevance term. Let the ego trajectory be denoted as $\mathcal { P } _ { \mathrm { t r a j } } =$ $\{ p _ { j } \} _ { j = 0 } ^ { T } .$ where $p _ { j } \in \mathbb { R } ^ { 2 }$ is a waypoint on the ground plane. The distance from a Gaussian to the future trajectory is

$$
d _ { i } ^ { \mathrm { t r a j } } = \operatorname* { m i n } _ { \substack { p _ { j } \in \mathcal { P } _ { \mathrm { t r a j } } } } \left\| \pi _ { x y } \left( x _ { i } \right) - p _ { j } \right\| _ { 2 } .\tag{10}
$$

The trajectory relevance score is then computed as

$$
T _ { i } = \exp \left( - \frac { \left( d _ { i } ^ { \mathrm { t r a j } } \right) ^ { 2 } } { 2 \sigma _ { \mathrm { t r a j } } ^ { 2 } } \right) .\tag{11}
$$

Here, $\pi _ { x y } \left( \cdot \right)$ projects a 3D point onto the ground plane, and $\sigma _ { \mathrm { t r a j } }$ controls the spatial influence range of the trajectory prior. The final driving-aware score is

$$
S _ { i } = \frac { w _ { q } Q _ { i } + w _ { t } T _ { i } } { w _ { q } + w _ { t } } .\tag{12}
$$

In our implementation, $w _ { q }$ and $w _ { t }$ control the contributions of intrinsic quality and trajectory relevance, respectively. We select $K _ { c }$ candidates by combining a global top-K branch and a voxel-coverage branch:

$$
\mathcal { C } = \mathrm { T o p K } \left( \{ S _ { i } \} _ { i = 1 } ^ { N } , K _ { c } \right) \cup \mathrm { V o x e l T o p K } \left( \{ S _ { i } \} _ { i = 1 } ^ { N } , K _ { c } \right)\tag{13}
$$

The global branch preserves the most informative Gaussians, while the voxel branch enforces spatial coverage over the whole driving scene. If the union contains more than $K _ { c }$ candidates, we keep the top-ranked candidates according to $S _ { i } ;$ otherwise, we pad the candidate set with invalid tokens.

Attribute-aware Gaussian Projection. After coarse selection, each candidate Gaussian is projected into the LLM hidden space by an attribute-aware aligner. For the 3D position, we adopt a learnable Fourier embedding:

$$
\phi _ { \mathrm { p o s } } \left( x _ { i } \right) = \left[ \sin \left( 2 \pi B x _ { i } \right) , \cos \left( 2 \pi B x _ { i } \right) \right] .\tag{14}
$$

Here, B is a learnable linear projection. We decode $z _ { i }$ as $\hat { f } _ { i } = { \cal D } _ { \cal F } ( z _ { i } )$ and project position, scale, rotation, opacity, and foundation features into a shared hidden space. Denoting this attribute set as $\mathcal { P } _ { \mathrm { a t t r } } = \{ x , s , r , o , \hat { f } \}$ , we fuse the embeddings using learnable scalar gates:

$$
\tilde { h } _ { i } = \sum _ { p \in \mathcal { P } _ { \mathrm { a t t r } } } \lambda _ { p } h _ { i } ^ { p } .\tag{15}
$$

The fused embedding is further refined by a residual MLP aligner:

$$
g _ { i } = \tilde { h } _ { i } + \mathrm { M L P } \left( \mathrm { L N } \left( \tilde { h } _ { i } \right) \right) .\tag{16}
$$

The resulting embedding $g _ { i }$ is the aligned Gaussian token representation for the i-th candidate.

Text-Conditioned Perceiver-Style Cross-Attention. Although the coarse selector significantly reduces the number of Gaussians, the candidate set C is still too large for direct LLM injection. We therefore use a small set of learned queries to aggregate the candidate Gaussian tokens through crossattention. Given the input text embeddings, we first compute a pooled instruction embedding:

$$
\bar { e } _ { q } = \frac { 1 } { | T | } \sum _ { \ell \in \mathcal { T } } e _ { \ell } ,\tag{17}
$$

where $\tau$ denotes the valid text-token positions and $e _ { \ell }$ is the embedding of the ℓ-th token. Let $\bar { \{ q _ { j } ^ { \mathrm { L } } \} } _ { j = 1 } ^ { K _ { f } }$ denote $K _ { f }$ learned queries. Each query is modulated by the pooled text embedding:

$$
\tilde { q } _ { j } = q _ { j } ^ { \mathrm { L } } + W _ { t } \bar { e } _ { q } , \qquad j = 1 , \ldots , K _ { f } .\tag{18}
$$

The aligned Gaussian candidates provide the keys and values:

$$
{ \mathcal { K } } _ { G } = \{ W _ { k } g _ { i } \mid i \in { \mathcal { C } } \} , \qquad \mathcal { V } _ { G } = \{ W _ { v } g _ { i } \mid i \in { \mathcal { C } } \} .\tag{19}
$$

We then aggregate the Gaussian candidates with multi-head cross-attention:

$$
z _ { j } = \mathrm { M H A } \left( W _ { q } \tilde { q } _ { j } , { \cal K } _ { G } , { \cal \mathcal { V } } _ { G } \right) , \qquad j = 1 , \dotsc , { \cal K } _ { f } .\tag{20}
$$

The resulting query responses form the compact Gaussian world-token set:

$$
\mathcal { Z } _ { G } = \{ z _ { j } \} _ { j = 1 } ^ { K _ { f } } .\tag{21}
$$

Because the learned queries are modulated by the pooled instruction embedding, the adapter preserves a fixed token budget while aggregating Gaussian evidence that is relevant to the current query.

Gaussian–Image Distribution Alignment. To make the aggregated Gaussian world tokens more compatible with the foundation visual space used by the LLM, we introduce a KLbased distribution alignment objective. Let $\mathcal { Y } _ { I } = \{ y _ { m } ^ { I } \} _ { m = } ^ { M _ { I } }$ 1 denote the foundation image tokens from the same sample. We estimate the diagonal Gaussian statistics of the Gaussian world tokens as

$$
\mu _ { G } = \frac { 1 } { K _ { f } } \sum _ { z \in \mathcal { Z } _ { G } } z , \qquad \sigma _ { G } ^ { 2 } = \frac { 1 } { K _ { f } } \sum _ { z \in \mathcal { Z } _ { G } } \left( z - \mu _ { G } \right) ^ { 2 } .\tag{22}
$$

The target image-token statistics are computed with stopgradient:

$$
\mu _ { I } = \mathrm { s g } \left( \frac { 1 } { M _ { I } } \sum _ { m = 1 } ^ { M _ { I } } y _ { m } ^ { I } \right) .\tag{23}
$$

$$
\sigma _ { I } ^ { 2 } = \mathrm { s g } \left( { \frac { 1 } { M _ { I } } } \sum _ { m = 1 } ^ { M _ { I } } \left( y _ { m } ^ { I } - \mu _ { I } \right) ^ { 2 } \right) .\tag{24}
$$

Let $\mathcal { N } _ { G } = \mathcal { N } ( \mu _ { G } , \mathrm { d i a g } ( \sigma _ { G } ^ { 2 } + \epsilon ) )$ and $\mathcal { N } _ { I } = \mathcal { N } ( \mu _ { I } , \mathrm { d i a g } ( \sigma _ { I } ^ { 2 } +$ ϵ)). The dimension-normalized alignment loss is

$$
{ \mathcal { L } } _ { \mathrm { K L } } = { \frac { 1 } { d } } D _ { \mathrm { K L } } \left( { \mathcal { N } } _ { G } \parallel { \mathcal { N } } _ { I } \right) .\tag{25}
$$

During task training, ${ \mathcal { L } } _ { \mathrm { K L } }$ is added to the language modeling objective with weight λ and warm-up coefficient $\alpha ( t )$ It regularizes Gaussian world tokens toward the foundation image-token distribution without imposing a separate KL constraint on text tokens.

## D. Scene Understanding with Gaussian World Tokens

After obtaining the compact Gaussian world tokens from the geometry-aware Gaussian adapter in Sec. III-C, we inject them into the LLM together with multi-view image tokens and text tokens for 3D-aware scene understanding.

Input Token Construction. Given multi-view images, we first encode them into image tokens by the visual encoder of the VLM. The language instruction q is tokenized into text tokens. The compact Gaussian world tokens are produced by the geometry-aware Gaussian adapter. Then, we insert $\mathcal { Z } _ { G }$ into dedicated Gaussian token slots between the image tokens and the text instruction. The final input sequence to the LLM is defined as

$$
\mathcal { X } _ { \mathrm { i n } } = \mathrm { C o n c a t } \left( \mathcal { V } , \mathcal { Z } _ { G } , \mathcal { T } \right) .\tag{26}
$$

This construction allows the LLM to jointly attend to appearance information from image tokens, explicit 3D spatial information from Gaussian tokens, and task semantics from text tokens.

Unified Scene Understanding Formulation. We formulate all scene understanding tasks as conditional language generation, with the LLM autoregressively predicting response a from $\chi _ { \mathrm { i n } } .$ Region description uses natural language, 2D grounding uses image-plane boxes, 3D grounding uses object location and size in the ego frame, and planning outputs a decision with its justification. Thus, heterogeneous tasks share the same token format and language modeling objective.

Language Modeling Objective. During training, each sample is converted into a prefix-response pair. The prefix contains the image tokens, Gaussian world tokens, and instruction tokens, while the response contains the ground-truth answer. We optimize the model with a prefix language modeling loss:

$$
\mathcal { L } _ { \mathrm { L M } } = - \frac { 1 } { \vert \mathcal { B } \vert } \sum _ { ( q , a ) \in \mathcal { B } } \sum _ { j = 1 } ^ { \vert a \vert } \log p _ { \theta } \left( a _ { j } \mid a _ { < j } , \mathcal { X } _ { \mathrm { i n } } \right) .\tag{27}
$$

Here, B denotes a training batch, $a _ { j }$ denotes the j-th response token, and $a _ { < j }$ denotes all previous response tokens. This objective trains the LLM to generate task-specific answers conditioned on multi-view image observations, compact Gaussian world tokens, and language instructions.

Training Strategy. We train the scene understanding module in two stages. In the first stage, we freeze the LLM and optimize the foundation-feature Gaussian tokenizer and geometryaware adapter, so that the compact Gaussian world tokens become compatible with the LLM hidden space. In the second stage, we keep the Gaussian tokenizer and adapter trainable and enable LoRA adaptation on the LLM, allowing the model to better integrate image, language, and Gaussian tokens for downstream reasoning. The total loss for scene understanding is defined as

$$
{ \mathcal { L } } _ { \mathrm { u n d e r s t a n d i n g } } = { \mathcal { L } } _ { \mathrm { L M } } + \lambda _ { \mathrm { K L } } \alpha \left( t \right) { \mathcal { L } } _ { \mathrm { K L } } .\tag{28}
$$

Here, ${ \mathcal { L } } _ { \mathrm { K L } }$ is the distribution alignment loss introduced in Sec. III-C, $\lambda _ { \mathrm { K L } }$ controls its weight, and $\alpha \left( t \right)$ is a warm-up coefficient. To reduce the influence of task imbalance, we further adopt task-aware batch sampling. Let Q denote the set of training tasks and $n _ { m }$ denote the number of samples in task m. The sampling probability of task m is defined as

$$
P _ { m } = \frac { n _ { m } ^ { \alpha _ { s } } } { \sum _ { k \in \mathcal { Q } } n _ { k } ^ { \alpha _ { s } } } .\tag{29}
$$

Here, $\alpha _ { s }$ is a smoothing factor. This strategy prevents large tasks from dominating the training process and improves the stability of multi-task scene understanding.

## E. Instruction-controllable 4D Scene Generation

In this section, we introduce the generation branch of GaussianDWM++, which performs multi-modal spatiotemporal scene generation and instruction-controllable 4D editing. Different from conventional driving scene generators that mainly rely on image or BEV conditions, our generation module is conditioned on both low-level geometric observations and high-level world knowledge extracted from the Gaussianaware LLM. The low-level conditions preserve scene texture and geometry, while the high-level language condition provides semantic guidance, driving context, and editing intent.

Dual-condition RGB-D Latent Diffusion. Our generation module is built upon a denoising UNet [66] and a frozen pre-trained VAE [67]. Given an RGB image $I _ { i }$ and its corresponding depth map $D _ { i }$ , we first convert the depth map into a pseudo-RGB image by channel replication. The RGB image and the pseudo-RGB depth image are then encoded into a unified latent space:

$$
z _ { I } = \xi _ { \mathrm { v a e } } \left( I _ { i } \right) , \quad z _ { D } = \xi _ { \mathrm { v a e } } \left( D _ { i } ^ { \mathrm { r g b } } \right) .\tag{30}
$$

During decoding, the VAE decoder reconstructs the RGB image and the depth map from the generated latent variables:

$$
\hat { I } _ { i } = \mathcal { D } _ { \mathrm { v a e } } \left( z _ { I } \right) , \quad \hat { D } _ { i } = \frac { 1 } { 3 } \sum _ { c = 1 } ^ { 3 } \mathcal { D } _ { \mathrm { v a e } } \left( z _ { D } \right) _ { c } .\tag{31}
$$

Here, the three decoded channels of the depth branch are averaged to obtain the final single-channel depth prediction.

For each training sample, we randomly sample a target camera state from either the spatially shifted trajectory or the future ego trajectory. The latent variable of the target modality is denoted as d. At diffusion timestep $t ,$ we add Gaussian noise to obtain the noisy latent:

$$
d _ { t } = \alpha _ { t } d + \sigma _ { t } \epsilon .\tag{32}
$$

Here, $\epsilon \sim \mathcal { N } ( 0 , I )$ denotes Gaussian noise, and $\alpha _ { t }$ and $\sigma _ { t }$ are the noise scheduling coefficients. Following the v-prediction objective, the denoising target is defined as:

$$
v _ { t } = \alpha _ { t } \epsilon - \sigma _ { t } d .\tag{33}
$$

Low-level Conditions. To provide explicit geometric guidance for generation, we construct low-level RGB and depth condition maps from the observed Gaussian scene or projected point cloud. Given the target camera transformation, we project the surrounding 3D scene evidence into the target view and obtain sparse RGB and depth conditions. The condition $C _ { I }$ provides sparse appearance cues, while $C _ { D }$ provides geometric structure. These low-level conditions help the diffusion model preserve spatial layout and geometric consistency during novel view synthesis and future scene prediction.

High-level Language Conditions. In addition to low-level conditions, the generation branch receives high-level semantic guidance from the scene understanding module. Given $\mathcal { Z } _ { G }$ and instruction q, the LLM produces the language condition $C _ { L } = \Phi _ { \mathrm { L L M } } ( \mathcal { Z } _ { G } , q )$

Diffusion Training Objective. The denoising network takes the noisy latent, reference latent, low-level conditions, highlevel language condition, and diffusion timestep embedding as input. Editing intent, when present, is encoded as part of the instruction-dependent condition $C _ { L }$

$$
\hat { v } _ { t } = \mathcal { F } _ { \theta } \left( d _ { t } , d _ { \mathrm { r e f } } , C _ { I } , C _ { D } , C _ { L } , s _ { t } \right) .\tag{34}
$$

Here, $d _ { \mathrm { r e f } }$ denotes the reference latent, and $s _ { t }$ denotes the timestep embedding. The generation loss is defined as:

$$
\mathcal { L } _ { \mathrm { g e n } } = \mathbb { E } _ { d , \epsilon , t , s _ { t } } \left. \mathcal { F } _ { \theta } \left( d _ { t } , d _ { \mathrm { r e f } } , C _ { I } , C _ { D } , C _ { L } , s _ { t } \right) - v _ { t } \right. _ { 2 } ^ { 2 } .\tag{35}
$$

By jointly conditioning on $C _ { I } , \ C _ { D }$ , and $C _ { L }$ , the diffusion model learns to generate RGB-D observations that are geometrically consistent, semantically coherent, and controllable by language instructions.

Spatiotemporal Generation and 4D Editing. For spatial generation, we obtain the target transformation by applying a lateral or longitudinal shift to the current camera pose, and construct sparse condition maps by projecting the Gaussian scene into the target view. This enables novel view synthesis under spatial shifts such as 1m or 2m. For temporal generation, we use the future ego trajectory predicted or inferred by the scene understanding module to define the target camera state, and project the Gaussian scene evidence into future timestamps. This enables future RGB-D prediction at 1s or 2s. Beyond spatial and temporal generation, GaussianDWM++ supports instruction-controllable 4D scene editing. For weather-conditioned generation, the editing condition modifies the global appearance and visibility while preserving the underlying scene geometry. For dynamic vehicle editing, it controls object-level operations such as inserting, removing, stopping, or moving vehicles along specified trajectories. Since both low-level geometric conditions and high-level language conditions are derived from the same foundation-feature Gaussian world representation, the generated results maintain better temporal coherence, geometric consistency, and semantic faithfulness across 4D driving scenes.

## IV. EXPERIMENTS

## A. Datasets and Evaluation Metrics

We evaluate GaussianDWM++ on both scene understanding and multi-modal scene generation tasks. For public driving data, we use nuScenes [9], a large-scale autonomous driving dataset with synchronized multi-view cameras, LiDAR, ego poses, object annotations, and temporal sequences. Following common autonomous driving settings, we use six surrounding camera images as visual inputs and construct 3D Gaussian scene representations from the corresponding driving sequences. For language-grounded scene understanding, we conduct experiments on three complementary benchmarks: NuInteract, NuInstruct [68], and OmniDrive [69]. NuInteract [23] provides approximately 1.5M multi-view image– language annotations and covers region description and perception, 2D visual grounding, 3D visual grounding, and planning-oriented reasoning. NuInstruct focuses on cross-view geometric understanding through tasks such as risk-object perception, target-distance estimation, agent and ego-state prediction, target-motion prediction, and reasoning. OmniDrive evaluates holistic driving-scene understanding with scene descriptions, attention-based question answering, counterfactual reasoning, decision and planning questions, and general dialogue. Following the VGGDrive evaluation protocol, we use its scene-description and general-dialogue tasks for quantitative evaluation. In addition to these public datasets, we further construct a foundation-feature 3D Gaussian driving dataset by distilling Qwen features into reconstructed Gaussian primitives. The construction details are introduced in Sec. IV-C.

For scene understanding, we follow the task-specific evaluation protocols of the three benchmarks. On NuInteract, region description and perception are evaluated using BLEU [70], ROUGE-L [71], and CIDEr [72]; 2D visual grounding uses mAP, F1 score, and mIoU; 3D visual grounding uses precision, mAP, and F1 score; and planning is evaluated by decision accuracy. We report the arithmetic mean over these ten metrics as the overall NuInteract score. On NuInstruct, we use MAE for regression tasks, accuracy for classification tasks, mAP for risk-object perception and detection tasks, and BLEU for captioning tasks, together with the protocol’s aggregate Average\* score. On OmniDrive, we evaluate scene-description and general-dialogue outputs using BLEU, CIDEr, and ROUGE-L, and report their arithmetic mean as the average score. The corresponding results are reported in Tabs. I–III.

For scene generation, we evaluate both spatial and temporal generation. Spatial generation corresponds to novel view synthesis under shifted camera poses, while temporal generation corresponds to future scene prediction along the driving trajectory. We report FID for image generation quality and FVD for temporal video generation quality. For RGB-D generation, we additionally evaluate the visual quality of RGB outputs and the geometric consistency of depth predictions when ground-truth depth is available. For instruction-controllable 4D editing, including weather-conditioned generation and dynamic vehicle manipulation, we provide qualitative visualizations to assess semantic consistency, temporal coherence, and controllability.

## B. Implementation Details

Our training pipeline consists of three stages. In the first stage, we train the foundation-feature Gaussian tokenizer and the geometry-aware Gaussian adapter. The Gaussian tokenizer distills Qwen visual-language features into 3D Gaussian primitives using the L1 feature-distillation loss, while the compact feature autoencoder is optimized with the squared L2 reconstruction loss. In the second stage, we integrate the Gaussian world tokens with the LLM for scene understanding. We freeze the main LLM parameters at the beginning and optimize the Gaussian tokenizer, attribute projection, and cross-attention adapter to warm up Gaussian–image representation alignment. Then, we enable LoRA-based fine-tuning of the LLM and jointly optimize the Gaussian adapter and the language reasoning branch. In the third stage, we train the multi-modal scene generation branch. The generation model is built on a denoising UNet [66] and a frozen pre-trained VAE [67]. It is optimized with a v-prediction objective [73]. The lowlevel conditions provide texture and geometric guidance, while the high-level language condition provides scene-level world knowledge and editing instructions. All models are trained on 16 NVIDIA A100 GPUs. During inference, the same compact Gaussian world representation is shared by the scene understanding branch and the generation branch, enabling consistent 3D-aware reasoning and controllable 4D scene generation.

## C. NuScenes-GSQA Dataset Construction

To support 3D Gaussian-based scene understanding and controllable driving scene generation, we construct a large-scale foundation-feature 3D Gaussian dataset from nuScenes [9]. The dataset contains 1000 outdoor urban driving scenes reconstructed as 3D Gaussians. Different from conventional image- or BEV-based driving datasets, our dataset provides explicit 3D Gaussian scene representations with geometry, appearance, opacity, and foundation visual-language features, enabling LLMs to reason over structured 3D scene primitives. The constructed dataset serves as the basis for Gaussian-aware scene understanding.

Large-scale 3D Gaussian Reconstruction. We reconstruct each nuScenes scene as a 3D Gaussian field using synchronized multi-view images, LiDAR point clouds, ego poses, and camera calibration parameters. Since outdoor driving scenes are large-scale and often contain repetitive structures, dynamic objects, motion blur, and limited view overlap, directly applying conventional COLMAP-based 3DGS initialization can lead to unstable or inaccurate reconstructions. To improve geometric robustness, we initialize each scene with aggregated LiDAR points provided by nuScenes. The aggregated point cloud is transformed into a unified scene coordinate frame and used to initialize Gaussian centers, while multi-view images are used to optimize appearance, opacity, scale, and rotation. During reconstruction, we remove frames with severe motion blur, poor exposure, or extremely low cross-view overlap. For scenes with reliable depth supervision, we additionally introduce a depth constraint to improve geometric fidelity. After optimization, each reconstructed Gaussian scene is evaluated using rendering metrics such as PSNR together with visual inspection. We exclude unreliable frames and primitives identified during this quality-control process while retaining the complete set of 1000 reconstructed scenes for downstream pre-training, scene understanding, and generation.

Foundation-feature Gaussian Annotation. After obtaining high-quality 3D Gaussian reconstructions, we further inject foundation visual-language features into Gaussian primitives. Specifically, we extract dense Qwen features from multi-view images and distill them into the reconstructed Gaussian field following Sec. III-B. Each Gaussian primitive stores a compact foundation-feature code rather than the full-dimensional feature, reducing storage cost while preserving open-vocabulary semantic information. The resulting representation associates visual-language semantics with explicit 3D scene primitives, providing a unified carrier for geometry, appearance, and foundation features.

Gaussian-aware QA Dataset. To train the LLM to consume 3D Gaussian tokens, we construct a Gaussian-aware QA dataset based on NuInteract [23]. We convert the original image-language annotations into a format that explicitly includes Gaussian references. Following the training paradigm of Qwen-VL, the Gaussian token placeholder is inserted at the beginning of the user instruction. In this way, the model learns to treat Gaussian tokens as structured 3D context before answering scene understanding questions.

A simplified annotation example is shown below:

1 {   
2 "scene\_token": "e7ef871f77f44331aefdebc24ec034b7",   
3 "frame\_idx": 0,   
4 "task": "rdp",   
5 "category": "2D\_perception\_infos",   
6 "conversations": [   
7 {   
8 "from": "human",   
9 "value": "Based on <gauss>, can you estimate the   
distance between the <CAM\_FRONT> <box>(14,219),   
37,248)</box> and the ego vehicle?"   
10 },   
11 {   
12 "from": "gpt",   
13 "value": "The object is approximately 38.7 meters   
away from the ego vehicle."   
14 }   
15 ],   
16 "image": [   
17 "nuscenes/samples/CAM\_FRONT/xxx.jpg"   
18 ],   
19 "views": [   
20 "CAM\_FRONT"   
21 ],   
22 "gauss": [   
23 "gauss/scene\_0002/frame\_00000.pth"   
24 ]   
25 }

The QA annotations cover multiple scene understanding tasks, including region description and perception, 2D visual grounding, 3D visual grounding, and planning-oriented reasoning. Compared with image-only QA data, the Gaussianaware annotations explicitly bind each instruction to the corresponding 3D Gaussian scene representation, allowing the model to learn 3D-aware language grounding.

Trajectory-oriented QA Dataset. For spatial and temporal scene generation, the model needs to infer target camera poses from language instructions. We therefore construct a trajectory-oriented QA dataset to enhance trajectory understanding and future pose prediction. For spatial generation, instructions typically describe a target viewpoint, such as shifting the camera by a certain distance to the left, right, forward, or backward. The model is trained to infer the corresponding target pose, which is then used to project the Gaussian scene or point cloud into the target view to obtain sparse generation conditions.

For temporal generation, instructions describe future scene prediction, such as predicting the scene one or two seconds later. We extract 10-frame video clips from nuScenes sequences. The first 4 frames are used as historical trajectory inputs, and the remaining 6 frames are used as prediction targets. All poses are transformed into the ego coordinate system of the 5th frame, so that the model learns future motion in a normalized local frame. A simplified trajectory QA example is shown below:

```jsonl
1 {
2 "conversations": [
3 {
"from": "human",
5 "value": "There are last 4 frames of ego trajectory:
[PT, [-7.45, 3.05, 0.16, 0.00, 0.00, -0.36, 0.9
3], [-5.75, 1.97, 0.11, 0.00, 0.00, -0.28, 0.96]
, [-4.04, 0.92, 0.09, 0.00, 0.00, -0.19, 0.98],
[-2.15, 0.26, 0.04, 0.00, 0.00, -0.10, 1.00]].
Predict the future 6-frame ego trajectory."
6 },
7 {
8 "from": "gpt",
9 "value": "[PT, [0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
1.00], [2.47, 0.65, -0.06, 0.00, 0.00, 0.09, 1.0
0], [4.24, 2.32, -0.11, 0.00, 0.00, 0.12, 0.99],
[5.50, 4.08, -0.15, 0.00, 0.00, 0.13, 0.99], [7
.03, 6.45, -0.15, 0.00, 0.00, 0.13, 0.99], [8.66
, 8.91, -0.17, -0.01, 0.00, 0.13, 0.99]]"
10 }
11 ]
12 }
```

This trajectory-oriented supervision enables the LLM to predict target camera states for spatial and temporal generation. The predicted pose is then used to construct sparse RGB and depth conditions for the diffusion generator. As a result, the same dataset supports both Gaussian-aware scene understanding and trajectory-conditioned 4D scene generation.

## D. Scene Understanding

We evaluate scene understanding on NuInteract, NuInstruct, and OmniDrive, which together cover language description, cross-view perception, 2D/3D visual grounding, state prediction, and planning-oriented reasoning. The quantitative comparisons are reported in Tabs. I–III, and representative qualitative results are shown in Fig. 4.

NuInteract. Following the DriveMonkey protocol [23], we compare GaussianDWM++ with general-purpose LVLMs, driving-specific VLMs, and specialized 3D detectors. As shown in Tab. I, GaussianDWM++ achieves the best overall result, ranks first on all region-description and 2D-grounding metrics, and obtains the best mAP and F1 for 3D grounding. Its 3D precision and closed-set planning accuracy remain competitive rather than being the best. The most notable gain over the conference version is on 2D grounding mAP, which improves

![](images/19442f479a704a6d52320c128b39eee5f09ee782b87d63491f1972284181aad9.jpg)

![](images/ab699bbd28f669d166206df3c97d91f1d2b0da6ca46f275163e4c3212c240a62.jpg)

Q: What is the quantity of truck at <CAM\_FRONT>, and what are their respective locations?. The output format …

GT: [{\"bbox\_2d\": [353, 428, 452, 618], \"label\": \"truck\"}]

GaussianDWM: [{\"bbox\_2d\": [359, 423, 452, 613], \"label\": \"truck\"}]

GaussianDWM++: [{\"bbox\_2d\": [357, 427, 453, 615], \"label\": \"truck\"}]

Q: Find all car in this <CAM\_BACK>. For each car, provide its 3D bounding box. The output format …

GT: [{\"bbox\_3d\": [18.89, 0.95, 28.36, 4.43, 1.53, 1.94, 0.00, 2.48, 1.01], \"label\": \"car\"}, {\"bbox\_3d\": [25.10, 0.88, 27.91, 4.45, 1.75, 1.86, 0.00, 2.48, 1.01], \"label\": \"car\"}, {\"bbox\_3d\": [9.23, 0.78, 21.09, 5.79, 2.09, 2.13, 0.00, 0.50, 1.01], \"label\": \"car\"}, {\"bbox\_3d\": [-7.12, 0.61, 5.29, 4.83, 1.54, 1.86, 0.00, 2.48, 1.01], \"label\": \"car\"}, {\"bbox\_3d\": [7.86, 0.84, 53.58, 4.66, 1.94, 2.64, 0.00, 0.51, 1.01], \"label\": \"car\"}]

GaussianDWM: [{\"bbox\_3d\": [10.03, 0.99, 21.00, 4.70, 1.90, 1.87, 0.00, 0.50, 1.01], \"label\": \"car\"}, {\"bbox\_3d\": [-6.99, 0.71, 4.40, 4.70, 1.50, 1.80, 0.00, 2.48, 1.01], \"label\": \"car\"}]

GaussianDWM++: [{\"bbox\_3d\": [8.30, 0.87, 18.52, 4.59, 1.82, 1.86, 0.00, 0.50, 1.01], \"label\": \"car\"}, {\"bbox\_3d\": [10.00, 0.85, 52.03, 4.57, 1.73, 2.02, 0.00, 0.50, 1.01], \"label\": \"car\"}, {\"bbox\_3d\": [-6.80, 0.65, 5.17, 4.79, 1.76, 1.98, 0.00, 2.48, 1.01], \"label\": \"car\"}]

Q: What is the <CAM\_BACK\_LEFT> <box>(11,401),(473,648)</box> indicating within the scene?What is the spatial relationship between the it in relation to the ego vehicle's position?

GT: It is parked trailer. 27.1 meters is the distance from the ego car to it

GaussianDWM: It is parked trailer. The ego car is 24.0m away from it

GaussianDWM++: It is parked trailer. The ego car is 27.7 meters distant from it.

Q: What is the appropriate driving course for the ego vehicle under these conditions?   
GT: Go Straight   
GaussianDWM: Go Straight   
GaussianDWM++: Go Straight

![](images/35c9575301f48f8d6f43ea2a8e30af32683a512375621e3ad1c7ba16586d002d.jpg)  
Fig. 4: Qualitative results for scene understanding and scene generation. From top to bottom, we display the multi-view input of the current scene and the 3D Gaussian ellipsoids, the scene understanding results, and the results of temporal generation, spatial generation, weather-conditioned generation, and dynamic vehicle manipulation with varying motion speeds.

by approximately 38%. This result pattern is consistent with the main motivation of our design: distilling foundation visuallanguage features into explicit Gaussian primitives establishes finer correspondence between semantic concepts and spatial regions, while driving-aware selection and text-conditioned cross-attention preserve and aggregate task-relevant geometry into compact world tokens. Consequently, the improvement is most pronounced on tasks that require language to be accurately grounded in scene geometry, without compromising higher-level driving reasoning.

NuInstruct. Tab. II evaluates cross-view risk perception, regression, classification, and captioning under the VGGDrive protocol. GaussianDWM++ achieves the best MAE, accuracy, mAP, and aggregate result, while its BLEU score remains competitive but is not the best. In particular, it reduces MAE by approximately 22% relative to VGGDrive. NuInstruct requires the model to associate observations across views and reason about distances, motion states, and risk objects. The strong results on its geometric and prediction-oriented metrics therefore suggest that the continuous Gaussian representation preserves multi-view spatial relationships more effectively than image-level features alone. Meanwhile, the fact that the largest gains occur outside captioning indicates that the improvement primarily comes from better 3D-aware perception and reasoning rather than a stronger language prior. OmniDrive. As reported in Tab. III, GaussianDWM++ ranks first on all language metrics and improves the best previous average result by approximately 14%. Unlike NuInteract and NuInstruct, OmniDrive emphasizes holistic scene description and dialogue. The consistent improvement across these complementary settings indicates that the proposed representation is not limited to localization tasks. By anchoring foundation features to explicit 3D primitives and aligning the Gaussian world-token distribution with foundation image tokens, the model retains scene-level visual semantics while translating structured spatial evidence into language-compatible tokens. This leads to more complete and better-grounded descriptions of complex driving scenes.

The qualitative examples in Fig. 4 further support this analysis by jointly visualizing the multi-view observations, 3D Gaussian scene representation, and language-based outputs. The responses remain tied to visible entities and their spatial relationships, illustrating how the shared Gaussian representation connects geometric evidence with semantic reasoning.

## E. Scene Generation

We evaluate multi-modal scene generation on nuScenes [9], including spatial novel-view generation under shifted camera poses and temporal prediction along the future ego trajectory. Following previous works [13], [52], we interpolate the 2Hz keyframe annotations to 12Hz and adopt the DiST-S evaluation protocol. Quantitative spatial-generation results are reported in Tab. IV, while qualitative spatial and temporal results are shown in Fig. 4 and 5.

We compare with representative reconstruction- and generation-based approaches, including PVG [44], EmerNeRF [42], StreetGaussian [45], OmniRe [74], FreeVS [87], and

DiST-S [88]. As shown in Tab. IV, GaussianDWM++ achieves the best FID and FVD under all three camera-shift settings. More importantly, its advantage becomes more pronounced as the viewpoint shift increases. Under the most challenging ±4 m setting, it reduces FID by approximately 20% relative to the conference version. Large camera shifts substantially reduce view overlap and make image-only conditions increasingly ambiguous. In GaussianDWM++, the low-level RGB-D conditions projected from the explicit Gaussian scene provide persistent appearance and geometric constraints, while the high-level language condition supplies scene semantics when projected observations are sparse or incomplete. Their complementary roles explain why the method remains robust as the target view moves farther from the observed cameras.

The spatial and temporal examples in Fig. 4 further show that the generated results preserve scene layout and maintain consistent appearance across target views and future frames. Since both conditions originate from the same foundationfeature Gaussian world representation, geometric evidence and semantic guidance remain mutually consistent rather than being fused from independently constructed scene representations. The qualitative RGB-D comparison in Fig. 5 further supports this trend: GaussianDWM++ better preserves broad, continuous regions such as road surfaces under shifted viewpoints, reducing the large-area artifacts observed in GaussianDWM and reconstruction-based baselines. This behavior is consistent with the dual-condition design, where Gaussianaware high-level world knowledge provides semantic context when projected RGB-D evidence becomes sparse or incomplete, complementing the low-level appearance and geometric constraints.

The qualitative visualization in Fig. 6 and quantitative evaluation in Tab. V further assess the long-horizon generation capability of GaussianDWM++, demonstrating stable and temporally consistent world evolution over extended prediction horizons owing to the complementary dual-condition design and the incorporation of 3D world knowledge.

Overall, the quantitative results and the qualitative results establish state-of-the-art spatial generation performance and coherent future prediction. Together, they validate the dualcondition design and the use of a shared foundation-feature Gaussian representation to connect scene understanding with generation.

## F. Ablation Study

We conduct ablation studies to examine how the Gaussian scene representation, token selection and alignment strategies, and dual-condition generation design contribute to the final performance. Rather than treating these components as interchangeable sources of improvement, we analyze the capability that each component primarily affects.

Gaussian Representation and Hybrid Sampling. Tab. VI first examines the conference-stage Gaussian representation and sampling strategy. Fine-tuning is essential for adapting the language model to driving tasks, while introducing Gaussian tokens consistently improves the fine-tuned imageonly baseline across description, grounding, and planning.

![](images/9993d073c77acb8481ad19f0fde5bae04f752329fc96f0f0c252ceca801d9b94.jpg)

Fig. 5: Qualitative comparison of RGB-D NVS with a 4 m camera shift. Compared with GaussianDWM and reconstruction based baselines [44], [45], [74], [75], GaussianDWM++ better preserves broad, continuous scene regions such as road surfaces, with fewer large-area artifacts and more coherent depth structure under large viewpoint changes.
<table><tr><td></td><td></td><td></td><td colspan="3">RD&amp;P ↑</td><td colspan="3">2D VG ↑</td><td colspan="3">3D VG ↑</td><td>Plan ↑</td><td></td></tr><tr><td>Model</td><td>Year</td><td>LLM</td><td>BLEU</td><td>ROUGE-L</td><td>CIDEr</td><td>mAP</td><td>F1</td><td>mIoU</td><td>Pr</td><td>mAP</td><td>F1</td><td>Acc</td><td>Avg. ↑</td></tr><tr><td>LLaVA1.5 [76]</td><td>2024</td><td>Vicuna-7B</td><td>64.23</td><td>76.69</td><td>74.82</td><td>0.10</td><td>0.16</td><td>14.31</td><td>6.51</td><td>5.33</td><td>3.12</td><td>36.20</td><td>28.15</td></tr><tr><td>MiniCPM-V 2 [77]</td><td>2024</td><td>MiniCPM-2B</td><td>47.43</td><td>63.16</td><td>69.88</td><td>0.11</td><td>0.13</td><td>13.34</td><td>0.97</td><td>1.55</td><td>0.86</td><td>36.69</td><td>23.41</td></tr><tr><td>MiniCPM-V 2.6 [77]</td><td>2024</td><td>Qwen2-7B</td><td>47.92</td><td>69.11</td><td>70.20</td><td>0.36</td><td>0.49</td><td>18.74</td><td>1.97</td><td>1.61</td><td>0.93</td><td>36.42</td><td>24.78</td></tr><tr><td>InternVL1.5-2B [78]</td><td>2024</td><td>InternLM2-7B</td><td>67.14</td><td>81.10</td><td>79.83</td><td>14.74</td><td>17.64</td><td>55.43</td><td>28.05</td><td>21.73</td><td>12.92</td><td>53.96</td><td>43.25</td></tr><tr><td>InternVL1.5-4B [78]</td><td>2024</td><td>Phi3-4B</td><td>66.63</td><td>80.64</td><td>79.24</td><td>14.27</td><td>17.60</td><td>53.52</td><td>25.14</td><td>19.46</td><td>11.63</td><td>40.25</td><td>40.84</td></tr><tr><td>Qwen2-VL-2B [79]</td><td>2024</td><td>Qwen2-2B</td><td>67.92</td><td>80.24</td><td>78.51</td><td>17.11</td><td>20.87</td><td>57.24</td><td>12.82</td><td>10.20</td><td>6.12</td><td>45.59</td><td>39.66</td></tr><tr><td>Qwen2-VL-7B [79]</td><td>2024</td><td>Qwen2-7B</td><td>66.65</td><td>78.57</td><td>77.97</td><td>16.06</td><td>20.04</td><td>55.51</td><td>20.64</td><td>16.26</td><td>9.82</td><td>49.33</td><td>41.09</td></tr><tr><td>InternVL2-1B</td><td>2024</td><td>Qwen2-0.5B</td><td>66.89</td><td>81.00</td><td>79.59</td><td>16.70</td><td>20.21</td><td>55.94</td><td>23.36</td><td>18.35</td><td>10.94</td><td>44.08</td><td>41.71</td></tr><tr><td>InternVL2-2B</td><td>2024</td><td>InternLM2-2B</td><td>66.77</td><td>80.87</td><td>79.62</td><td>16.12</td><td>19.49</td><td>55.29</td><td>27.83</td><td>21.09</td><td>12.58</td><td>44.61</td><td>42.43</td></tr><tr><td>InternVL2-4B</td><td>2024</td><td>Phi3-4B</td><td>66.88</td><td>80.76</td><td>79.29</td><td>19.14</td><td>23.47</td><td>59.07</td><td>25.28</td><td>20.12</td><td>11.97</td><td>40.43</td><td>42.64</td></tr><tr><td>InternVL2-8B</td><td>2024 2025</td><td>InternLM2.5-7B</td><td>67.32</td><td>81.39</td><td>80.01</td><td>20.61</td><td>25.24</td><td>61.90</td><td>31.47</td><td>24.67</td><td>14.70</td><td>46.93</td><td>45.42</td></tr><tr><td>DriveMonkey [23]</td><td></td><td>InternLM2.5-7B</td><td>67.50</td><td>81.15</td><td>79.79</td><td>19.47</td><td>24.02</td><td>59.36</td><td>51.90</td><td>34.53</td><td>20.86</td><td>82.64</td><td>52.12</td></tr><tr><td>BEVFormer [56]</td><td>2022</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>44.50</td><td>23.69</td><td>1.58</td><td></td><td></td></tr><tr><td>PETR [80] CAPE [81]</td><td>2022 2023</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>55.80</td><td>31.34</td><td>20.58</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>55.02</td><td>32.94</td><td>21.33</td><td></td><td></td></tr><tr><td>GaussianDWM [10]</td><td>2026</td><td>Qwen3-8B</td><td>68.78</td><td>81.06</td><td>78.72</td><td>34.95</td><td>40.49</td><td>71.85</td><td>50.66</td><td>52.78</td><td>32.05</td><td>80.95</td><td>59.23</td></tr><tr><td>GaussianDWM++</td><td>2026</td><td>Qwen3-8B</td><td>69.81</td><td>81.75</td><td>80.42</td><td>48.22</td><td>41.92</td><td>77.83</td><td>53.32</td><td>57.19</td><td>35.13</td><td>81.52</td><td>62.71</td></tr></table>

TABLE I: Comparison on NuInteract [23]. Results cover region description and perception (RD&P), 2D/3D visual grounding (VG), and closed-set planning. Avg. is the arithmetic mean over the ten task metrics. All metrics are higher-is-better; best and second-best results are shown in bold and underlined, respectively.

<table><tr><td colspan="6">Zero-shot Models</td><td colspan="6">Fine-tuned Models</td></tr><tr><td>Dataset</td><td>Metrics</td><td>GPT-4o [82]</td><td>LLaVA-OV [83]</td><td>RoboTron [84]</td><td>Qwen2.5-VL [85]</td><td>Qwen2.5-VL [85]</td><td>InMLLM [68]</td><td>VGGT-Dist [86]</td><td>VGGT-Add [86]</td><td>VGGDrive [86]</td><td>GaussianDWM++</td></tr><tr><td rowspan="5">NuInstruct</td><td>MAE↓</td><td>9.93</td><td>87.04</td><td>19.36</td><td>24.10</td><td>4.35</td><td>9.08</td><td>3.73</td><td>3.63</td><td>3.08</td><td>2.41</td></tr><tr><td>Accuracy ↑</td><td>10.64</td><td>3.75</td><td>2.57</td><td>0.63</td><td>47.71</td><td>32.48</td><td>56.21</td><td>56.35</td><td>56.37</td><td>58.02</td></tr><tr><td>mAP ↑</td><td>0</td><td>0</td><td>0</td><td>0</td><td>6.15</td><td>21.93</td><td>28.51</td><td>30.12</td><td>37.49</td><td>42.59</td></tr><tr><td>BLEU ↑</td><td>7.08</td><td>8.55</td><td>8.06</td><td>5.56</td><td>75.75</td><td>35.20</td><td>79.23</td><td>79.45</td><td>81.13</td><td>76.40</td></tr><tr><td>Average*↑</td><td>1.95</td><td>0</td><td>0</td><td>0</td><td>31.32</td><td>20.13</td><td>40.06</td><td>40.57</td><td>42.98</td><td>43.65</td></tr></table>

TABLE II: Comparison on NuInstruct [68] under the VGGDrive evaluation protocol. MAE is lower-is-better, while the remaining metrics are higher-is-better. Average\* is (Accuracy + mAP + BLEU − MAE)/4. Best and second-best results are shown in bold and underlined, respectively.

The comparison among sampling strategies further shows that the benefit does not simply come from providing more

3D primitives. Replacing random sampling with Top-K plus uniform sampling produces the most pronounced gain in planning accuracy, with an improvement of approximately 64%. This indicates that a mixture of salient primitives and broad spatial coverage is important for retaining the global context required by driving decisions. Adding text–Gaussian similarity selection yields the best overall result and the strongest 2Dgrounding performance, whereas Top-K plus uniform sampling remains slightly better on several 3D-grounding metrics. Thus, query relevance mainly refines object-level localization, while geometry-preserving coverage remains important for metric 3D reasoning.

<table><tr><td colspan="7">Zero-shot Models</td><td colspan="5">Fine-tuned Models</td></tr><tr><td>Dataset</td><td>Metrics</td><td>GPT-4o [82]</td><td>LLaVA-OV [83]</td><td>RoboTron [84]</td><td>OmniDrive [69]</td><td>HERMES [8]</td><td>Qwen2.5-VL [85]</td><td>VGGT-Dist [86]</td><td>VGGT-Add [86]</td><td>VGGDrive [86]</td><td>GaussianDWM++</td></tr><tr><td rowspan="4">OmniDrive</td><td>BLEU ↑</td><td>10.91</td><td>16.14</td><td>20.30</td><td>38.00</td><td></td><td>37.29</td><td>37.28</td><td>37.23</td><td>37.58</td><td>42.11</td></tr><tr><td>CIDEr ↑</td><td>24.42</td><td>28.41</td><td>34.33</td><td>68.60</td><td>74.10</td><td>86.29</td><td>86.41</td><td>86.38</td><td>86.57</td><td>100.00</td></tr><tr><td>ROUGE-L ↑</td><td>22.34</td><td>22.14</td><td>23.67</td><td>32.60</td><td>32.70</td><td>34.33</td><td>34.32</td><td>34.26</td><td>34.40</td><td>38.13</td></tr><tr><td>Average ↑</td><td>19.22</td><td>22.23</td><td>26.10</td><td>46.40</td><td></td><td>52.64</td><td>52.67</td><td>52.62</td><td>52.85</td><td>60.08</td></tr></table>

TABLE III: Comparison on OmniDrive [69] under the VGGDrive evaluation protocol. Average is the arithmetic mean of BLEU, CIDEr, and ROUGE-L. All metrics are higher-is-better. Best and second-best results are shown in bold and underlined, respectively.

![](images/78c3482724816910923568e06bcb425782a062f6696dd5d91002010851f2a019.jpg)

Fig. 6: Long-horizon generation from 0–9 s.
<table><tr><td></td><td colspan="2">±1m</td><td colspan="2">±2m</td><td colspan="2">±4m</td></tr><tr><td>Method</td><td>FID ↓</td><td>FVD ↓</td><td>FID ↓</td><td>FVD ↓</td><td>FID ↓</td><td>FVD ↓</td></tr><tr><td>PVG [44]</td><td>48.15</td><td>246.74</td><td>60.44</td><td>356.23</td><td>84.50</td><td>501.16</td></tr><tr><td>EmerNeRF [42]</td><td>37.57</td><td>171.47</td><td>52.03</td><td>294.55</td><td>76.11</td><td>497.85</td></tr><tr><td>StreetGaussian [45]</td><td>32.12</td><td>153.45</td><td>43.24</td><td>256.91</td><td>67.44</td><td>429.98</td></tr><tr><td>OmniRe [74]</td><td>31.48</td><td>152.01</td><td>43.31</td><td>254.52</td><td>67.36</td><td>428.20</td></tr><tr><td>FreeVS [87]</td><td>51.26</td><td>431.99</td><td>62.04</td><td>497.37</td><td>77.14</td><td>556.14</td></tr><tr><td>DiST-S [88]</td><td>10.12</td><td>45.14</td><td>12.97</td><td>68.80</td><td>17.57</td><td>105.29</td></tr><tr><td>GaussianDWM [10]</td><td>8.36</td><td>44.50</td><td>11.27</td><td>68.17</td><td>18.81</td><td>116.40</td></tr><tr><td>GaussianDWM++</td><td>8.19</td><td>42.65</td><td>11.05</td><td>63.86</td><td>15.02</td><td>101.20</td></tr></table>

TABLE IV: Spatial novel-view generation on nuScenes. FID and FVD under ±1 m, ±2 m, and ±4 m camera shifts using the six-camera DiST-S protocol; lower is better. Best and secondbest results are shown in bold and underlined, respectively.

<table><tr><td>Method</td><td>6-View</td><td>Video</td><td>FID ↓</td><td>FVD ↓</td></tr><tr><td>BEVGen</td><td></td><td>x</td><td>25.54</td><td></td></tr><tr><td>BEVControl</td><td></td><td>X</td><td>24.85</td><td></td></tr><tr><td>DriveGAN [89]</td><td>x</td><td></td><td>73.40</td><td>502.30</td></tr><tr><td>DriveDreamer</td><td>x</td><td></td><td>52.60</td><td>452.00</td></tr><tr><td>Vista [1]</td><td>x</td><td></td><td>6.90</td><td>89.40</td></tr><tr><td>WoVoGen</td><td></td><td></td><td>27.60</td><td>417.70</td></tr><tr><td>Panacea</td><td></td><td></td><td>16.96</td><td>139.00</td></tr><tr><td>MagicDrive [13]</td><td></td><td></td><td>16.20</td><td>217.94</td></tr><tr><td>DriveDreamer-2 [4]</td><td></td><td></td><td>25.00</td><td>105.10</td></tr><tr><td>MagicDriveDiT</td><td></td><td></td><td>20.91</td><td>94.84</td></tr><tr><td>Drive-WM [3]</td><td></td><td></td><td>15.80</td><td>122.70</td></tr><tr><td>UniScene [6]</td><td></td><td></td><td>6.12</td><td>70.52</td></tr><tr><td>DiST-T [88]</td><td></td><td>V</td><td>6.83</td><td>22.67</td></tr><tr><td>GaussianDWM++ [10]</td><td></td><td></td><td>4.55</td><td>11.79</td></tr></table>

TABLE V: Temporal video generation on the nuScenes validation set. FID and FVD are lower-is-better. Best and secondbest results are shown in bold and underlined, respectively.

Foundation-Feature Tokenizer and Geometry-Aware Adapter. Tab. VII further separates the journal-version components. Introducing Gaussian–image distribution alignment produces the largest change in 2D grounding, improving mAP by approximately 39%, and also benefits most description and 3D-grounding metrics. This supports the role of the KL objective in transferring foundation image-token semantics to Gaussian world tokens. However, distribution alignment alone does not preserve the best planning result, showing that image-guided compatibility is necessary but insufficient for high-level driving reasoning. Text-conditioned cross-attention achieves the strongest 3D-grounding results, while driving-aware coarse selection performs best on the principal 2D-grounding metrics. These complementary trends agree with the adapter design: coarse selection maintains driving-scene coverage, whereas textmodulated learned queries aggregate task-relevant primitives into a fixed number of world tokens. Finally, enabling foundation-feature distillation improves planning accuracy by approximately 20% over the preceding configuration and produces the best aggregate result. Although the complete model does not rank first on every individual metric, it provides the best balance among semantic description, spatial grounding, and planning-oriented reasoning.

<table><tr><td rowspan="2">Fine-tuning</td><td rowspan="2">Gaussian</td><td rowspan="2">Sampling</td><td colspan="3">RD&amp;P ↑</td><td colspan="3">2D VG ↑</td><td colspan="3">3D VG ↑</td><td rowspan="2">Plan ↑  $\operatorname { A v g } . \uparrow$ </td><td rowspan="2"></td></tr><tr><td>BLEU</td><td>ROUGE-L</td><td>CIDEr</td><td>mAP</td><td>F1</td><td>mIoU</td><td>Pr</td><td>mAP</td><td>F1  $\operatorname { A c c }$ </td></tr><tr><td>Zero-shot</td><td>No</td><td></td><td>2.91</td><td>12.68</td><td>0.59</td><td>0.00</td><td>0.00</td><td>12.24</td><td>48.75</td><td>47.59</td><td>29.12</td><td>0.00</td><td>15.39</td></tr><tr><td>Fine-tuned</td><td>No</td><td></td><td>65.09</td><td>78.35</td><td>76.56</td><td>30.01</td><td>35.45</td><td>69.01</td><td>50.36</td><td>51.88</td><td>31.43</td><td>45.09</td><td>53.32</td></tr><tr><td>Fine-tuned</td><td>Yes</td><td>Random</td><td>66.19</td><td>79.00</td><td>76.97</td><td>33.94</td><td>39.37</td><td>71.40</td><td>50.94</td><td>52.85</td><td>32.03</td><td>49.43</td><td>55.21</td></tr><tr><td>Fine-tuned</td><td>Yes</td><td>Top-K + Uniform</td><td>68.78</td><td>81.06</td><td>78.82</td><td>33.89</td><td>39.31</td><td>71.37</td><td>51.16</td><td>52.87</td><td>32.05</td><td>80.95</td><td>58.93</td></tr><tr><td>Fine-tuned</td><td>Yes</td><td> $\mathrm { T o p - } K + \mathrm { U n i f o r m } + \mathrm { S i m i l a r i t y }$ </td><td>68.78</td><td>81.06</td><td>78.82</td><td>34.95</td><td>40.49</td><td>71.85</td><td>50.66</td><td>52.78</td><td>32.05</td><td>80.95</td><td>59.23</td></tr></table>

TABLE VI: Ablation of the Gaussian scene representation and token sampling on NuInteract. The settings and results follow the conference-stage evaluation protocol. Best and second-best results are shown in bold and underlined, respectively.
<table><tr><td rowspan="2">Foundation-Feature Distillation</td><td rowspan="2">Coarse Selection</td><td rowspan="2">Fine Adapter</td><td rowspan="2">Distribution Alignment</td><td colspan="3">RD&amp;P ↑</td><td colspan="3">2D VG ↑</td><td colspan="3">3D VG ↑</td><td rowspan="2">Plan ↑ Acc</td><td rowspan="2"> $\operatorname { A v g } . \uparrow$ </td></tr><tr><td>BLEU</td><td>ROUGE-L</td><td>CIDEr</td><td> $\mathrm { m A P }$ </td><td>F1</td><td>mIoU</td><td>Pr</td><td>mAP</td><td></td></tr><tr><td>x</td><td>Voxel Top-K</td><td>Similarity</td><td>x</td><td>68.78</td><td>81.06</td><td>78.72</td><td>34.95</td><td>40.49</td><td>71.85</td><td>50.66</td><td>52.78</td><td>32.05</td><td>80.95</td><td>59.23</td></tr><tr><td>x</td><td>Voxel Top-K</td><td>Similarity Query-Aware</td><td>√</td><td>69.97</td><td>81.47</td><td>80.05</td><td>48.59</td><td>41.95</td><td>77.56</td><td>53.39</td><td>57.18</td><td>35.22</td><td>67.02</td><td>61.24</td></tr><tr><td>x</td><td>Voxel Top-K</td><td>Cross-Attn Query-Aware</td><td>√</td><td>70.19</td><td>81.77</td><td>80.11</td><td>48.10</td><td>41.83</td><td>77.67</td><td>53.59</td><td>57.67</td><td>35.51</td><td>67.21</td><td>61.37</td></tr><tr><td>x</td><td>Driving-Aware</td><td>Cross-Attn Query-Aware</td><td>√</td><td>69.76</td><td>81.88</td><td>79.98</td><td>49.07</td><td>42.03</td><td>77.76</td><td>53.30</td><td>57.10</td><td>35.18</td><td>68.06</td><td>61.41</td></tr><tr><td>√</td><td>Driving-Aware</td><td>Cross-Attn</td><td>√</td><td>69.81</td><td>81.75</td><td>80.42</td><td>48.22</td><td>41.92</td><td>77.83</td><td>53.32</td><td>57.19</td><td>35.13</td><td>81.52</td><td>62.71</td></tr></table>

TABLE VII: Ablation of the foundation-feature Gaussian tokenizer and geometry-aware Gaussian adapter on NuInteract. Rows progressively introduce KL-based Gaussian–image distribution alignment, text-conditioned cross-attention, and drivingaware coarse selection. The last row additionally enables foundation-feature distillation. Avg. is the arithmetic mean over the ten task metrics; higher is better.

<table><tr><td>Low-Level High-level</td><td></td><td>FID ↓ FVD ↓</td></tr><tr><td>X</td><td>√</td><td></td></tr><tr><td>√</td><td>x</td><td>10.12 45.14</td></tr><tr><td>√</td><td>√</td><td>8.36 44.5</td></tr></table>

TABLE VIII: Ablation study of the dual-condition generation mechanism under a ±1 m camera shift. FID and FVD are lower-is-better. “–” denotes failure under the corresponding setting.

Dual-Condition Generation. Tab. VIII analyzes the roles of low-level geometric observations and high-level world knowledge. High-level conditioning alone fails to generate valid results because semantic guidance cannot determine the targetview geometry and appearance without a spatial anchor. Lowlevel RGB-D conditioning is sufficient to make generation feasible, but combining it with the high-level condition further reduces FID by approximately 17%. This confirms that the two conditions play complementary rather than redundant roles: projected RGB-D evidence constrains scene layout and local appearance, while Gaussian-aware language features supply semantic context for uncertain or incomplete regions. The ablation therefore explains why the full model improves perceptual quality without sacrificing geometric consistency.

## V. CONCLUSION

In this paper, we present GaussianDWM++, a foundationfeature 4D Gaussian driving world model for unified scene understanding, language-grounded reasoning, controllable editing, and multi-modal generation. By directly distilling Qwen features into 3D Gaussian primitives, our framework builds an explicit open-vocabulary Gaussian scene representation and reduces the dependence on external language Gaussian construction pipelines. To efficiently bridge dense 3D Gaussians with large language models and generative models, we further introduce a geometry-aware Gaussian adapter with importance-aware sampling, text-conditioned Perceiver-style cross-attention, and KL-based Gaussian–image distribution alignment. Based on the aligned Gaussian world representation, our model supports controllable 4D scene editing, including weather-conditioned generation and dynamic vehicle manipulation. We also construct NuScenes-GSQA with 1000 foundation-feature Gaussian driving scenes and Gaussianaware QA annotations, providing a useful benchmark for future 3D Gaussian-based driving world models. Extensive experiments demonstrate that the proposed framework effectively improves scene understanding, visual grounding, planningoriented reasoning, and controllable multi-modal generation, showing the potential of 3D Gaussian representations as a unified foundation for future driving world models.

## REFERENCES

[1] S. Gao, J. Yang, L. Chen, K. Chitta, Y. Qiu, A. Geiger, J. Zhang, and H. Li, “Vista: A generalizable driving world model with high fidelity and versatile controllability,” Advances in Neural Information Processing Systems, vol. 37, pp. 91 560–91 596, 2024.

[2] A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado, “Gaia-1: A generative world model for autonomous driving,” arXiv preprint arXiv:2309.17080, 2023.

[3] Y. Wang, J. He, L. Fan, H. Li, Y. Chen, and Z. Zhang, “Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 14 749–14 759.

[4] G. Zhao, X. Wang, Z. Zhu, X. Chen, G. Huang, X. Bao, and X. Wang, “Drivedreamer-2: Llm-enhanced world models for diverse driving video generation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 10, 2025, pp. 10 412–10 420.

[5] Z. Yang, L. Chen, Y. Sun, and H. Li, “Visual point cloud forecasting enables scalable autonomous driving,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 14 673–14 684.

[6] B. Li, J. Guo, H. Liu, Y. Zou, Y. Ding, X. Chen, H. Zhu, F. Tan, C. Zhang, T. Wang et al., “Uniscene: Unified occupancy-centric driving scene generation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 11 971–11 981.

[7] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[8] X. Zhou, D. Liang, S. Tu, X. Chen, Y. Ding, D. Zhang, F. Tan, H. Zhao, and X. Bai, “Hermes: A unified self-driving world model for simultaneous 3d scene understanding and generation,” arXiv preprint arXiv:2501.14729, 2025.

[9] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for autonomous driving,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 11 621–11 631.

[10] T. Deng, X. Chen, Y. Chen, Q. Chen, Y. Xu, L. Yang, L. Xu, Y. Zhang, B. Zhang, W. Huang, and H. Wang, “Gaussiandwm: 3d gaussian driving world model for unified scene understanding and multi-modal generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2026, pp. 10 656–10 667.

[11] L. Kong, W. Yang, J. Mei, Y. Liu, A. Liang, D. Zhu, D. Lu, W. Yin, X. Hu, M. Jia et al., “3d and 4d world modeling: A survey,” arXiv preprint arXiv:2509.07996, 2025.

[12] A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado, “Gaia-1: A generative world model for autonomous driving,” arXiv preprint arXiv:2309.17080, 2023.

[13] R. Gao, K. Chen, E. Xie, L. Hong, Z. Li, D.-Y. Yeung, and Q. Xu, “Magicdrive: Street view generation with diverse 3d geometry control,” arXiv preprint arXiv:2310.02601, 2023.

[14] J. Mao, B. Li, B. Ivanovic, Y. Chen, Y. Wang, Y. You, C. Xiao, D. Xu, M. Pavone, and Y. Wang, “Dreamdrive: Generative 4d scene modeling from street view images,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 367–374.

[15] D. Liang, D. Zhang, X. Zhou, S. Tu, T. Feng, X. Li, Y. Zhang, M. Du, X. Tan, and X. Bai, “Unifuture: A 4d driving world model for future generation and perception,” in IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2026.

[16] Y. Wang, Z. Fang, L. Zhao, and W. Chen, “Learning to tune like an expert: Interpretable and scene-aware navigation via mllm reasoning and cvae-based adaptation,” arXiv preprint arXiv:2507.11001, 2025.

[17] Y. Hong, H. Zhen, P. Chen, S. Zheng, Y. Du, Z. Chen, and C. Gan, “3d-llm: Injecting the 3d world into large language models,” Advances in Neural Information Processing Systems, vol. 36, pp. 20 482–20 494, 2023.

[18] H. Zhen, X. Qiu, P. Chen, J. Yang, X. Yan, Y. Du, Y. Hong, and C. Gan, “3d-vla: A 3d vision-language-action generative world model,” arXiv preprint arXiv:2403.09631, 2024.

[19] P. Zhang, Y. Su, P. Wu, D. An, L. Zhang, Z. Wang, D. Wang, Y. Ding, B. Zhao, and X. Li, “Cross from left to right brain: Adaptive text dreamer for vision-and-language navigation,” arXiv preprint arXiv:2505.20897, 2025.

[20] C. Zhu, Y. Lin, J. Shao, J. Lin, and Y. Wang, “Pathology-aware prototype evolution via llm-driven semantic disambiguation for multicenter diabetic retinopathy diagnosis,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 9196–9205.

[21] Z. Xu, Y. Zhang, E. Xie, Z. Zhao, Y. Guo, K.-Y. K. Wong, Z. Li, and H. Zhao, “Drivegpt4: Interpretable end-to-end autonomous driving via large language model,” IEEE Robotics and Automation Letters, 2024.

[22] C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, J. Beißwenger, P. Luo, A. Geiger, and H. Li, “Drivelm: Driving with graph visual question answering,” in European conference on computer vision. Springer, 2024, pp. 256–274.

[23] Z. Zhao, H. Fu, D. Liang, X. Zhou, D. Zhang, H. Xie, B. Wang, and X. Bai, “Extending large vision-language model for diverse interactive tasks in autonomous driving,” arXiv preprint arXiv:2505.08725, 2025.

[24] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” in ECCV, 2020.

[25] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3d gaussian¨ splatting for real-time radiance field rendering.” ACM Trans. Graph., vol. 42, no. 4, pp. 139–1, 2023.

[26] T. Deng, G. Shen, C. Xun, S. Yuan, T. Jin, H. Shen, Y. Wang, J. Wang, H. Wang, D. Wang et al., “Mne-slam: Multi-agent neural slam for mobile robots,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 1485–1494.

[27] T. Deng, G. Shen, X. Chen, S. Yuan, H. Shen, G. Peng, Z. Wu, J. Wang, L. Xie, D. Wang, H. Wang, and W. Chen, “Mcn-slam: Multi-agent collaborative neural slam with hybrid implicit neural scene representation,” arXiv preprint arXiv:2506.18678, 2025.

[28] T. Deng, G. Shen, T. Qin, J. Wang, W. Zhao, J. Wang, D. Wang, and W. Chen, “Plgslam: Progressive neural scene represenation with local to global bundle adjustment,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 657–19 666.

[29] T. Deng, W. Wu, J. He, Y. Pan, X. Jiang, S. Yuan, D. Wang, H. Wang, and W. Chen, “Vpgs-slam: Voxel-based progressive 3d gaussian slam in large-scale scenes,” arXiv preprint arXiv:2505.18992, 2025.

[30] Y. Deng, Y. Yue, J. Dou, J. Zhao, J. Wang, Y. Tang, Y. Yang, and M. Fu, “Omnimap: A general mapping framework integrating optics, geometry, and semantics,” IEEE Transactions on Robotics, 2025.

[31] Y. Deng, J. Wang, J. Zhao, J. Dou, Y. Yang, and Y. Yue, “Openobj: Open-vocabulary object-level neural radiance fields with fine-grained understanding,” IEEE Robotics and Automation Letters, vol. 10, no. 1, pp. 652–659, 2024.

[32] T. Deng, Y. Wang, H. Xie, H. Wang, R. Guo, J. Wang, D. Wang, and W. Chen, “Neslam: Neural implicit mapping and self-supervised feature tracking with depth completion and denoising,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 12 309–12 321, 2025.

[33] L. Li, S. Jia, J. Wang, Z. Jiang, F. Zhou, J. Dai, T. Zhang, Z. Wu, and J.-N. Hwang, “Human motion instruction tuning,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 17 582–17 591.

[34] R. Gu, S. Jia, Y. Ma, J. Zhong, J.-N. Hwang, and L. Li, “Mocount: Motion-based repetitive action counting,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 9026–9034.

[35] S. Zhu, G. Wang, H. Blum, Z. Wang, G. Zhang, D. Cremers, M. Pollefeys, and H. Wang, “Sni-slam++: Tightly-coupled semantic neural implicit slam,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[36] Z. Gong, X. Li, F. Tosi, Y. Zhang, S. Mattoccia, J. Wu, and M. Poggi, “Dino-slam: Dino-informed rgb-d slam for neural implicit and explicit representations,” arXiv preprint arXiv:2507.19474, 2025.

[37] Z. Gong, X. Li, F. Tosi, J. Han, S. Mattoccia, J. Cai, and M. Poggi, “Ov3r: Open-vocabulary semantic 3d reconstruction from rgb videos,” arXiv preprint arXiv:2507.22052, 2025.

[38] J. Ost, F. Mannan, N. Thuerey, J. Knodt, and F. Heide, “Neural scene graphs for dynamic scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 2856–2865.

[39] Y. Chen, T. Deng, W. Zhao, X. Wang, W. Xi, W. Chen, and J. Wang, “Snlidar: Semantic neural fields for novel space-time view lidar synthesis,” arXiv preprint arXiv:2504.08361, 2025.

[40] H. Turki, J. Y. Zhang, F. Ferroni, and D. Ramanan, “Suds: Scalable urban dynamic scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 12 375–12 385.

[41] T. Deng, Y. Wang, Y. Liu, C. Su, J. Wang, H. Wang, D. Wang, and W. Chen, “Prosgnerf: Progressive dynamic neural scene graph with frequency modulated auto-encoder in urban scenes,” arXiv preprint arXiv:2312.09076, 2023.

[42] J. Yang, B. Ivanovic, O. Litany, X. Weng, S. W. Kim, B. Li, T. Che, D. Xu, S. Fidler, M. Pavone et al., “Emernerf: Emergent spatialtemporal scene decomposition via self-supervision,” arXiv preprint arXiv:2311.02077, 2023.

[43] Y. Wen, L. Song, Y. Liu, S. Zhu, Y. Miao, L. Han, and H. Wang, “Freedriverf: Monocular rgb dynamic nerf without poses for autonomous driving via point-level dynamic-static decoupling,” arXiv preprint arXiv:2505.09406, 2025.

[44] Y. Chen, C. Gu, J. Jiang, X. Zhu, and L. Zhang, “Periodic vibration gaussian: Dynamic urban scene reconstruction and real-time rendering,” arXiv preprint arXiv:2311.18561, 2023.

[45] Y. Yan, H. Lin, C. Zhou, W. Wang, H. Sun, K. Zhan, X. Lang, X. Zhou, and S. Peng, “Street gaussians: Modeling dynamic urban scenes with gaussian splatting,” in European Conference on Computer Vision. Springer, 2024, pp. 156–173.

[46] X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M.-H. Yang, “Drivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 21 634–21 643.

[47] C. Peng, C. Zhang, Y. Wang, C. Xu, Y. Xie, W. Zheng, K. Keutzer, M. Tomizuka, and W. Zhan, “Desire-gs: 4d street gaussians for staticdynamic decomposition and surface reconstruction for urban driving scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 6782–6791.

[48] C. Peng, I. Sobol, M. Tomizuka, K. Keutzer, C. Xu, and O. Litany, “A lesson in splats: Teacher-guided diffusion for 3d gaussian splats generation with 2d supervision,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 28 707–28 717.

[49] J. Yang, J. Huang, Y. Chen, Y. Wang, B. Li, Y. You, A. Sharma, M. Igl, P. Karkus, D. Xu et al., “Storm: Spatio-temporal reconstruction model for large-scale outdoor scenes,” arXiv preprint arXiv:2501.00602, 2024.

[50] Q. Tian, X. Tan, Y. Xie, and L. Ma, “Drivingforward: Feed-forward 3d gaussian splatting for driving scene reconstruction from flexible surround-view input,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 7, 2025, pp. 7374–7382.

[51] Y. Zou, Y. Ding, C. Zhang, J. Guo, B. Li, X. Lyu, F. Tan, X. Qi, and H. Wang, “Mudg: Taming multi-modal diffusion with gaussian splatting for urban scene reconstruction,” arXiv preprint arXiv:2503.10604, 2025.

[52] J. Guo, Y. Ding, X. Chen, S. Chen, B. Li, Y. Zou, X. Lyu, F. Tan, X. Qi, Z. Li et al., “Dist-4d: Disentangled spatiotemporal diffusion with metric depth for 4d driving scene generation,” arXiv preprint arXiv:2503.15208, 2025.

[53] T. Deng, Y. Pan, S. Yuan, D. Li, C. Wang, M. Li, L. Chen, L. Xie, D. Wang, J. Wang, J. Civera, H. Wang, and W. Chen, “What is the best 3d scene representation for robotics? from geometric to foundation models,” arXiv preprint arXiv:2512.03422, 2025.

[54] S. Tu, X. Zhou, D. Liang, X. Jiang, Y. Zhang, X. Li, and X. Bai, “The role of world models in shaping autonomous driving: A comprehensive survey,” arXiv preprint arXiv:2502.10498, 2025.

[55] S. Zuo, Z. Xie, W. Zheng, S. Xu, F. Li, S. Jiang, L. Chen, Z.- X. Yang, and J. Lu, “Dvgt: Driving visual geometry transformer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 14 658–14 668.

[56] Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Q. Yu, and J. Dai, “Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[57] W. Zheng, W. Chen, Y. Huang, B. Zhang, Y. Duan, and J. Lu, “Occworld: Learning a 3d occupancy world model for autonomous driving,” in European conference on computer vision. Springer, 2024, pp. 55–72.

[58] C. Lindstrom, G. Hess, A. Lilja, M. Fatemi, L. Hammarstrand, C. Peters-¨ son, and L. Svensson, “Are nerfs ready for autonomous driving? towards closing the real-to-simulation gap,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 4461– 4471.

[59] C. Ni, G. Zhao, X. Wang, Z. Zhu, W. Qin, G. Huang, C. Liu, Y. Chen, Y. Wang, X. Zhang et al., “Recondreamer: Crafting world models for driving scene reconstruction via online restoration,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 1559–1569.

[60] G. Zhao, C. Ni, X. Wang, Z. Zhu, X. Zhang, Y. Wang, G. Huang, X. Chen, B. Wang, Y. Zhang et al., “Drivedreamer4d: World models are effective data machines for 4d driving scene representation,” in Proceedings of the computer vision and pattern recognition conference, 2025, pp. 12 015–12 026.

[61] J. Kerr, C. M. Kim, K. Goldberg, A. Kanazawa, and M. Tancik, “Lerf: Language embedded radiance fields,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 19 729–19 739.

[62] K. M. Jatavallabhula, A. Kuwajerwala, Q. Gu, M. Omama, T. Chen, A. Maalouf, S. Li, G. Iyer, S. Saryazdi, N. Keetha et al., “Conceptfusion: Open-set multimodal 3d mapping,” arXiv preprint arXiv:2302.07241, 2023.

[63] S. Peng, K. Genova, C. Jiang, A. Tagliasacchi, M. Pollefeys, T. Funkhouser et al., “Openscene: 3d scene understanding with open vocabularies,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 815–824.

[64] M. Qin, W. Li, J. Zhou, H. Wang, and H. Pfister, “Langsplat: 3d language gaussian splatting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 051–20 060.

[65] A.-M. Halacheva, J.-N. Zaech, X. Wang, D. P. Paudel, and L. Van Gool, “Gaussianvlm: Scene-centric 3d vision-language models using languagealigned gaussian splats for embodied reasoning and beyond,” arXiv preprint arXiv:2507.00886, 2025.

[66] A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts et al., “Stable video

diffusion: Scaling latent video diffusion models to large datasets,” arXiv preprint arXiv:2311.15127, 2023.

[67] J. Xing, M. Xia, Y. Zhang, H. Chen, W. Yu, H. Liu, G. Liu, X. Wang, Y. Shan, and T.-T. Wong, “Dynamicrafter: Animating open-domain images with video diffusion priors,” in European Conference on Computer Vision. Springer, 2024, pp. 399–417.

[68] X. Ding, J. Han, H. Xu, X. Liang, W. Zhang, and X. Li, “Holistic autonomous driving understanding by bird’view injected multi-modal large models,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 13 668–13 677.

[69] S. Wang, Z. Yu, X. Jiang, S. Lan, M. Shi, N. Chang, J. Kautz, Y. Li, and J. M. Alvarez, “Omnidrive: A holistic vision-language dataset for autonomous driving with counterfactual reasoning,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 22 442–22 452.

[70] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “Bleu: a method for automatic evaluation of machine translation,” in Proceedings of the 40th annual meeting of the Association for Computational Linguistics, 2002, pp. 311–318.

[71] C.-Y. Lin, “Rouge: A package for automatic evaluation of summaries,” in Text summarization branches out, 2004, pp. 74–81.

[72] R. Vedantam, C. Lawrence Zitnick, and D. Parikh, “Cider: Consensusbased image description evaluation,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 4566–4575.

[73] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Muller, H. Saini,¨ Y. Levi, D. Lorenz, A. Sauer, F. Boesel et al., “Scaling rectified flow transformers for high-resolution image synthesis,” in Forty-first international conference on machine learning, 2024.

[74] Z. Chen, J. Yang, J. Huang, R. de Lutio, J. M. Esturo, B. Ivanovic, O. Litany, Z. Gojcic, S. Fidler, M. Pavone, L. Song, and Y. Wang, “Omnire: Omni urban scene reconstruction,” in The Thirteenth International Conference on Learning Representations, 2025.

[75] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, “Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction,” arXiv preprint arXiv:2309.13101, 2023.

[76] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 26 296–26 306.

[77] Y. Yao, T. Yu, A. Zhang, C. Wang, J. Cui, H. Zhu, T. Cai, H. Li, W. Zhao, Z. He et al., “Minicpm-v: A gpt-4v level mllm on your phone,” arXiv preprint arXiv:2408.01800, 2024.

[78] Z. Chen, W. Wang, H. Tian, S. Ye, Z. Gao, E. Cui, W. Tong, K. Hu, J. Luo, Z. Ma et al., “How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites,” Science China Information Sciences, vol. 67, no. 12, p. 220101, 2024.

[79] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge et al., “Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution,” arXiv preprint arXiv:2409.12191, 2024.

[80] Y. Liu, T. Wang, X. Zhang, and J. Sun, “Petr: Position embedding transformation for multi-view 3d object detection,” in European conference on computer vision. Springer, 2022, pp. 531–548.

[81] K. Xiong, S. Gong, X. Ye, X. Tan, J. Wan, E. Ding, J. Wang, and X. Bai, “Cape: Camera view position embedding for multi-view 3d object detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 21 570–21 579.

[82] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford et al., “Gpt-4o system card,” arXiv preprint arXiv:2410.21276, 2024.

[83] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu et al., “Llava-onevision: Easy visual task transfer,” arXiv preprint arXiv:2408.03326, 2024.

[84] Z. Huang, C. Feng, F. Yan, B. Xiao, Z. Jie, Y. Zhong, X. Liang, and L. Ma, “Robotron-drive: All-in-one large multimodal model for autonomous driving,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2025, pp. 8011–8021.

[85] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin, “Qwen2.5-vl technical report,” arXiv preprint arXiv:2502.13923, 2025.

[86] J. Wang, G. Li, Z. Huang, C. Dang, H. Ye, Y. Han, and L. Chen, “Vggdrive: Empowering vision-language models with cross-view geometric grounding for autonomous driving,” arXiv preprint arXiv:2602.20794, 2026.

[87] Q. Wang, L. Fan, Y. Wang, Y. Chen, and Z. Zhang, “Freevs: Generative view synthesis on free driving trajectory,” in Proceedings of the International Conference on Learning Representations (ICLR), 2025.

[88] J. Guo, Y. Ding, X. Chen, S. Chen, B. Li, Y. Zou, X. Lyu, F. Tan, X. Qi, Z. Li, and H. Zhao, “Dist-4d: Disentangled spatiotemporal diffusion with metric depth for 4d driving scene generation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025, p. 27231–27241.

[89] S. W. Kim, J. Philion, A. Torralba, and S. Fidler, “Drivegan: Towards a controllable high-quality neural simulation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 5820–5829.