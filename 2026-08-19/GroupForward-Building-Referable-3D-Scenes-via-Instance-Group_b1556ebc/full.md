# GroupForward: Building Referable 3D Scenes via Instance-Grouped Feed-Forward Gaussian Splatting

Qijian Tian, Zimeng Wu, Xuhong Wang, Lizhuang Ma, and Xin Tan

Abstract—Simultaneously reconstructing and understanding 3D environments is essential for embodied agents. Toward this goal, feed-forward semantic 3D Gaussian Splatting (3DGS) efficiently constructs semantic scene representations from sparse multi-view observations. However, existing methods lack explicit instance discrimination and mainly support category- or phrase-based semantic queries. To this end, we propose Group-Forward, an instance-grouped feed-forward Gaussian splatting model that reconstructs geometry, appearance, instance structure, and semantics from sparse, unposed, and uncalibrated multi-view images. Unlike existing methods that attach highdimensional semantic features to each Gaussian, GroupForward learns compact instance embeddings that group Gaussians into cross-view consistent 3D instances, reformulating feed-forward semantic 3DGS from per-Gaussian semantic feature rendering to instance-level semantic aggregation and propagation. Building on these instance groups, we further propose a Referential Scene Reasoning Framework (RSRF) for complex 3D referring segmentation. RSRF constructs an instance-grouped 3D scene graph and retrieves candidate instances for a given referring expression. A vision-language model then reasons over structured instance evidence and multi-view observations to identify the referred instance among the candidates. RSRF thereby extends language interaction from simple semantic querying to complex referential scene reasoning. Experiments on semantic reconstruction and referential reasoning demonstrate the effectiveness of our instance-grouped reconstruction and reasoning framework.

Index Terms—Semantic 3D Reconstruction, 3D Gaussian Splatting, Scene Understanding, Referential Reasoning.

## I. INTRODUCTION

Embodied intelligence requires agents to perceive, understand, and execute tasks in real-world 3D environments. To support these capabilities, agents need to rapidly construct the surrounding scene from limited observations and reason about spatial relationships. Scene reconstruction can recover 3D scenes from 2D observations, providing the foundation for such spatial perception and understanding. However, geometry and appearance alone are insufficient for semantic understanding and decision-making. Therefore, 3D scene representations for downstream tasks should further incorporate semantic information and spatial relationships, extending reconstructed scenes from renderable 3D models to structured representations that support scene understanding, referential reasoning, and embodied interaction.

To achieve this goal, semantic 3D reconstruction incorporates semantic features into 3D representations, enabling semantic understanding of reconstructed scenes. Recently, 3D Gaussian Splatting [1] (3DGS) has been widely adopted due to its efficient rendering, explicit representation, and flexibility in attaching semantic attributes. However, most existing semantic 3DGS methods [2]–[6] rely on per-scene optimization, where geometry, appearance, and semantic attributes are optimized for each scene. Such a process is time-consuming and typically requires dense observations, limiting their applicability to sparse-view reconstruction and real-time interaction. This motivates feed-forward semantic 3DGS reconstruction [7], [8], which directly predicts Gaussian scene representations of geometry, appearance, and semantics from sparse input views.

Nevertheless, existing feed-forward semantic 3DGS reconstruction methods exhibit two major limitations. First, most methods formulate semantic representation as per-Gaussian feature embeddings, where high-dimensional language-aligned features are attached to individual Gaussians and rendered into 2D semantic maps. Although such representations associate Gaussians with semantic features or open-vocabulary concepts, they do not explicitly group Gaussians into object instances, making it difficult to distinguish multiple instances within the same category and preserve cross-view identity consistency. Second, existing methods mainly support category-level or phrase-based semantic queries, such as object category names or simple phrases, through semantic similarity matching. However, real-world language instructions often involve complex referring expressions, such as “the trash bin near the cabinet” or “the place in the scene where one can wash hands,” which require joint reasoning over object identities, spatial relationships, functional attributes, and contextual information. These limitations make it difficult for existing methods to support complex language-based referential reasoning.

To address these limitations, we introduce an instancegrouped scene representation into feed-forward 3DGS, where instances serve as structured entities that bridge Gaussian primitives and language reasoning. As illustrated in Fig. 1, our framework reconstructs cross-view consistent 3D instance groups from unposed and uncalibrated multi-views, discriminating instances within the same category. These instance groups further provide a structured scene representation for complex referential reasoning, grounding referring expressions to explicit 3D instance entities using geometric, appearance, semantic, and spatial cues.

Specifically, we first propose GroupForward, an instancegrouped feed-forward Gaussian splatting model. In contrast to existing methods that directly embed high-dimensional semantic features into each Gaussian, GroupForward learns lowdimensional instance embeddings with rendering-consistent instance supervision, grouping Gaussian primitives into crossview consistent 3D instances. Semantic information is then aggregated and propagated at the instance level. We extract semantic features from context views using a 2D openvocabulary model and aggregate them into a semantic prototype for each 3D instance. In the target view, rendered instance partitions are combined with these prototypes to produce viewconsistent open-vocabulary semantic predictions. GroupForward thus reformulates semantic 3D reconstruction from highdimensional per-Gaussian feature rendering to instance-level semantic aggregation and propagation.

![](images/9368b7e08d1eab938e2bd9527177062454d27e6e75841c6d40f7d05cf493ea8e.jpg)  
Fig. 1. The proposed instance-grouped formulation for feed-forward semantic 3D Gaussian Splatting builds referable scenes with low-dimensional instance embeddings rather than attaching high-dimensional semantic features, enabling complex referential reasoning beyond category- or phrase-level queries.

Moreover, we build a Referential Scene Reasoning Framework (RSRF) on the 3D instance groups produced by Group-Forward to understand complex natural-language referential expressions and achieve 3D referring segmentation. The framework constructs an instance-grouped scene graph, where each node represents a cross-view consistent 3D instance with its semantic prototype, multi-view masks, 3D position, and visibility, while edges encode spatial and geometric relations between instances. Given a referential expression, RSRF retrieves candidate instances using semantic, geometric, and graph cues and organizes the corresponding information into an evidence bundle. A vision-language model (VLM) then reasons over existing instance nodes to select the target, rather than generating new segmentation masks. The target instance and the corresponding cross-view consistent masks are directly returned. By grounding language understanding in a stable 3D instance graph, RSRF extends feed-forward semantic 3DGS from category-level semantic querying to instance-grounded scene reasoning and 3D referring segmentation.

Our main contributions are summarized as follows:

• We propose Instance-Grouped Feed-Forward Gaussian Splatting (GroupForward), which shifts feed-forward semantic 3DGS from per-Gaussian high-dimensional semantic feature rendering to instance-grouped reconstruction based on low-dimensional instance embeddings, enabling explicit instance-level scene structures from sparse input views.

• Building on the 3D instance groups, we propose a Referential Scene Reasoning Framework (RSRF) based on a 3D instance scene graph, extending semantic 3DGS from category-level or phrase-based semantic queries to complex natural-language referential reasoning and 3D referring segmentation.

• Experiments on sparse-view reconstruction, semantic reconstruction, and referential reasoning show that our method outperforms existing methods and demonstrate the effectiveness of instance-grouped feed-forward reconstruction and 3D instance-based reasoning.

## II. RELATED WORK

## A. Feed-Forward 3D Reconstruction

Typical 3D reconstruction paradigms, such as Structurefrom-Motion [9], Neural Radiance Fields [10] (NeRF), and 3D Gaussian Splatting [1] (3DGS), rely on per-scene optimization to recover scene geometry and appearance. Although these methods achieve high reconstruction quality, they often require dense observations and take minutes or even hours to reconstruct a single scene, making them less suitable for embodied intelligence [11]–[13], where 3D representations need to be rapidly constructed from sparse observations for perception, planning, and interaction. This has motivated feed-forward 3D reconstruction [14]–[21], which learns generalizable crossscene priors and directly infers 3D representations from sparse input views in a single forward pass without costly per-scene optimization. Existing methods use different 3D representations. Pointmap-based approaches mainly recover geometry from unposed sparse views [15], [16], [20], while NeRF- and 3DGS-based approaches additionally model appearance for novel-view synthesis [14], [17], [18], [21]. However, they generally do not incorporate semantics into reconstructed scenes, limiting downstream scene understanding and reasoning.

## B. Semantic 3D Reconstruction

Semantic 3D reconstruction extends typical 3D reconstruction by incorporating semantic information into 3D scene representations. Beyond geometry and appearance, it associates 3D representations such as points, radiance fields, or 3D Gaussians with semantic labels or language-aligned features, enabling semantic understanding, open-vocabulary querying, or language-guided interaction. Similar to 3D reconstruction, semantic 3D reconstruction methods initially rely on per-scene optimization. NeRF-based methods [22], [23] and 3DGS-based methods [2]–[6] attach semantic or languagealigned features to 3D representations for semantic rendering and querying. Recent studies have further explored feedforward semantic 3D reconstruction to improve efficiency and practicality. Some methods [24]–[27] incorporate semantics into pointmap-based feed-forward reconstruction models. However, point-based representations have limited appearance modeling capability for high-quality rendering and novel-view synthesis. Other methods [7], [8], [28]–[32] perform feedforward semantic reconstruction with 3D Gaussians, jointly reconstructing geometry, appearance, and semantics. Most of these methods attach high-dimensional semantic features to each Gaussian and distill rendered feature maps from 2D semantic features. This design introduces semantics into 3DGS but lacks explicit instance modeling, limiting the ability to distinguish same-category instances. Rendering highdimensional semantic features also increases computational cost and may lead to feature degradation. SIU3R [33] avoids dense feature rendering through query-based mask prediction, but its reliance on closed-set ground-truth semantic labels limits open-vocabulary capability. In contrast, GroupForward replaces high-dimensional per-Gaussian semantic features with low-dimensional instance embeddings and aggregates openvocabulary semantic prototypes at the instance level. This enables view-consistent open-vocabulary semantic prediction without ground-truth semantic labels while providing 3D instance entities for downstream referential reasoning.

## C. 3D Scene Understanding and Reasoning

Open-world scene understanding has been widely studied in 2D vision through CLIP-based methods [34], [35] and has recently been extended to 3D. SceneSplat [36] and SceneSplat++ [37] introduce language-aligned 3D Gaussian representations for open-world 3D scene understanding. However, these methods mainly support category-level or phrase-based semantic queries and struggle with complex natural-language instructions. Existing semantic 3D reconstruction methods [3], [7], [8], [23] also have this limitation, as they primarily focus on semantic querying rather than complex referential reasoning. MLLMs provide stronger language understanding and reasoning capabilities [38]–[41], but lack explicit 3D perception and thus fail to directly ground language in 3D representations. ReferSplat [42] explores referring 3D segmentation on 3DGS, while REALM [43] further leverages MLLMs for this task. However, they both rely on dense observations and per-scene optimization. In contrast, our RSRF combines feed-forward semantic reconstruction with MLLM-based reasoning to enable efficient referential scene understanding and reasoning from sparse observations.

## III. METHOD

## A. Overview

As illustrated in Fig. 2, our method consists of Group-Forward (Sec. III-B) for instance-grouped feed-forward 3D reconstruction and the Referential Scene Reasoning Framework (RSRF; Sec. III-C) for referential scene reasoning. The key insight is to use persistent 3D instance groups as intermediate entities to bridge low-level Gaussian reconstruction and high-level language reasoning. GroupForward jointly predicts cameras and Gaussian primitives with compact instance embeddings, which are then clustered into cross-view consistent 3D instance groups. Open-vocabulary features from the context views are aggregated within each group to form an instance-level semantic prototype. Built on this instancegrouped representation, RSRF organizes the instance groups into a 3D scene graph, retrieves candidate instances based on the input expression, and employs a VLM to identify target 3D entities using structured graph evidence and multi-view observations. The selected instances are then directly mapped to their cross-view masks without additional segmentation.

## B. Instance-Grouped Feed-Forward Gaussian Splatting

GroupForward predicts Gaussian primitives with compact instance embeddings in a feed-forward manner. Its design further includes gauge-aligned self-rendering, renderingconsistent instance learning, instance-level semantic aggregation and propagation, and staged curriculum training.

Feed-Forward Gaussian Splatting with Instance Embeddings. The feed-forward model uses a shared multi-view encoder followed by two decoders with complementary roles. The geometry decoder comprises camera and depth heads for geometry prediction. The Gaussian decoder also comprises two lightweight heads. The attribute head decodes pixelaligned attributes for standard Gaussians, while the instance head predicts compact embeddings for instance grouping. Specifically, each input pixel is lifted to a Gaussian primitive,

$$
\begin{array} { r } { G _ { i , p } = \left( \pmb { \mu } _ { i , p } , \alpha _ { i , p } , \mathbf { s } _ { i , p } , \mathbf { q } _ { i , p } , \mathbf { c } _ { i , p } , \mathbf { e } _ { i , p } \right) , } \end{array}\tag{1}
$$

where $\mu _ { i , p } , \alpha _ { i , p } , { \bf s } _ { i , p } , { \bf q } _ { i , p } .$ , and $\mathbf { c } _ { i , p }$ respectively denote the 3D center, opacity, scale, rotation, and spherical-harmonic appearance coefficients of the Gaussian associated with pixel p in view i. In contrast to standard 3D Gaussian primitives, we augment each primitive with an additional instance embedding ${ \bf e } _ { i , p } ~ \in ~ \mathbb { R } ^ { d _ { e } }$ . The embedding dimension $d _ { e }$ is set to be substantially lower than typical semantic feature dimensions, since $\mathbf { e } _ { i , p }$ is intended to encode instance grouping cues rather than semantics. The Gaussian decoder integrates multi-scale image features and predicted geometric cues to recover scene geometry and appearance, and the instance head combines multi-scale backbone features with predicted depth to promote embeddings aligned with object boundaries and 3D structure.

![](images/bc07e2f2b30247a27979db1db971bb6ca91803bfd4faec414d1da3bac5aaf4fc.jpg)  
Fig. 2. Overview of GroupForward and the Referential Scene Reasoning Framework (RSRF). Given sparse, unposed, and uncalibrated multi-view images, GroupForward reconstructs cross-view consistent 3D instance groups with compact instance embeddings and aggregates open-vocabulary semantics at the instance level. RSRF further organizes the instance groups into a 3D scene graph and integrates structured graph evidence with multi-view observations for VLM-based complex referential reasoning and 3D referring segmentation.

Gauge-Aligned Self-Rendering. Feed-forward Gaussian reconstruction is commonly trained with photometric supervision between observed and synthetic images rendered with ground-truth camera parameters. While providing a reliable reconstruction signal, this supervision does not account for pose estimation errors and therefore does not explicitly enforce consistency between the reconstructed scene and the predicted camera poses. To address this limitation, we introduce gaugealigned self-rendering, which supervises the reconstructed scene under self-predicted camera poses. Specifically, we construct the Gaussian scene from the context views and perform an additional stop-gradient pose inference pass over both the context and target views, producing the joint pose predictions. Since the context-only and joint pose predictions may differ in global scale, we align the latter to the former using their shared context views. Let $\mathbf { t } _ { i } ^ { c }$ and $\mathbf { t } _ { i } ^ { r }$ denote the translation vectors predicted for context view i by the contextonly reconstruction pass and the joint pose inference pass, respectively. The alignment scale is computed as

$$
\gamma = \frac { \sum _ { i \in \mathcal { C } } \| \mathbf { t } _ { i } ^ { c } \| _ { 2 } } { \sum _ { i \in \mathcal { C } } \| \mathbf { t } _ { i } ^ { r } \| _ { 2 } } ,\tag{2}
$$

where C denotes the context-view set. We scale the translations of all jointly predicted poses as $\widetilde { \mathbf { t } } _ { v } ^ { r } = \gamma \mathbf { t } _ { v } ^ { r } .$ , where $v \in \mathcal { C } \cup \mathcal { T }$ and render the context-built Gaussian scene from both context and target views under the aligned poses for photometric supervision $\mathcal { L } _ { p h o t o }$ . This objective explicitly encourages consistency between the reconstructed scene and the predicted camera poses, while ground-truth pose supervision $\mathcal { L } _ { p o s e }$ anchors pose estimation and prevents mutually compensating errors between scene geometry and camera poses.

Rendering-Consistent Instance Learning. For instance learning, we aim to group Gaussian primitives into instances based on their attached instance embeddings. A straightforward solution is to apply pixel-level contrastive learning to the pixel-aligned embeddings from the context views. However, enumerating pixel pairs is computationally expensive and may overemphasize redundant local variations. We therefore formulate direct supervision using instance centroids. From $N$ sampled context embeddings with corresponding instance labels $\{ \mathbf { e } _ { n } , y _ { n } \} _ { n = 1 } ^ { N }$ , we compute an ℓ<sub>2</sub>-normalized centroid $\bar { \mathbf { e } } _ { k }$ for each of the K instances, where $\mathbf { e } _ { n }$ and $y _ { n }$ denote the normalized embedding and instance label, respectively. The direct objective pulls embeddings toward their corresponding centroids and pushes different centroids apart by a margin m:

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { d i r e c t } } ^ { \mathrm { p u l l } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \| \mathbf { e } _ { n } - \bar { \mathbf { e } } _ { y _ { n } } \| _ { 2 } , } \\ { \displaystyle \mathcal { L } _ { \mathrm { d i r e c t } } ^ { \mathrm { p u s h } } = \frac { 1 } { K ( K - 1 ) } \sum _ { k \neq l } \left[ m - \| \bar { \mathbf { e } } _ { k } - \bar { \mathbf { e } } _ { l } \| _ { 2 } \right] _ { + } . } \end{array}\tag{3}
$$

We combine pull and push terms to form the direct objective:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d i r e c t } } = \lambda _ { \mathrm { p u l l } } \mathcal { L } _ { \mathrm { d i r e c t } } ^ { \mathrm { p u l l } } + \lambda _ { \mathrm { p u s h } } \mathcal { L } _ { \mathrm { d i r e c t } } ^ { \mathrm { p u s h } } . } \end{array}\tag{4}
$$

However, this context-space objective alone does not explicitly account for changes introduced by Gaussian rasterization, including visibility, occlusion, and alpha compositing. We therefore compute the assignment probability of Gaussian $g$ to instance k as

$$
p _ { g k } = \mathrm { s o f t m a x } _ { k } \left( - \frac { \left\| \mathbf { e } _ { g } - \bar { \mathbf { e } } _ { k } \right\| _ { 2 } } { \tau } \right) ,\tag{5}
$$

where τ is a temperature parameter. The probabilities over all K instances form the assignment vector $\mathbf { p } _ { g } = \left( p _ { g 1 } , \ldots , p _ { g K } \right)$ During training, ${ \bf p } _ { g }$ is temporarily attached to Gaussian g as an auxiliary renderable attribute and rasterized into both context and target views. Each assignment channel represents a distinct object instance rather than a semantic category. We supervise the rendered partitions against ground-truth instance masks using a dice loss $\mathcal { L } _ { \mathrm { r e n d e r } } ^ { \mathrm { d i c e } }$ [44] for region-level alignment and a focal loss $\mathcal { L } _ { \mathrm { r e n d e r } } ^ { \mathrm { f o c a l } }$ [45] for ambiguous and low-confidence instance assignments. In parallel, we render the Gaussian embeddings into the same views and sample $N _ { \mathrm { r } }$ rendered embeddings. Each sampled embedding is pulled toward its corresponding context centroid computed before rendering, while different rendered centroids are pushed by a margin m:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { r e n d e r } } ^ { \mathrm { p u l l } } = \displaystyle \frac { 1 } { N _ { \mathrm { r } } } \sum _ { r = 1 } ^ { N _ { \mathrm { r } } } { \| \widehat { \mathbf { e } } _ { r } - \bar { \mathbf { e } } _ { y _ { r } } \| _ { 2 } } , } \\ & { \mathcal { L } _ { \mathrm { r e n d e r } } ^ { \mathrm { p u s h } } = \displaystyle \frac { 1 } { K ( K - 1 ) } \sum _ { k \neq l } \left[ m - \frac { \left\| \widehat { \mathbf { e } } _ { k } - \widehat { \mathbf { e } } _ { l } \right\| _ { 2 }  _ { + } } . } \end{\right]array} \end{array}\tag{6}
$$

where $\widehat { \mathbf { e } } _ { r }$ and $y _ { r }$ denote the normalized rendered embedding and its ground-truth instance label, respectively. We combine the dice and focal losses on the rendered partitions with the pull and push losses on the rendered embeddings to form the rendering objective:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { r e n d e r } } = \lambda _ { \mathrm { d i c e } } \mathcal { L } _ { \mathrm { r e n d e r } } ^ { \mathrm { d i c e } } + \lambda _ { \mathrm { f o c a l } } \mathcal { L } _ { \mathrm { r e n d e r } } ^ { \mathrm { f o c a l } } } \\ & { ~ + \lambda _ { \mathrm { p u l l } } \mathcal { L } _ { \mathrm { r e n d e r } } ^ { \mathrm { p u l l } } + \lambda _ { \mathrm { p u s h } } \mathcal { L } _ { \mathrm { r e n d e r } } ^ { \mathrm { p u s h } } . } \end{array}\tag{7}
$$

The complete instance-learning objective is

$$
\mathcal { L } _ { \mathrm { i n s } } = \lambda _ { \mathrm { d i r e c t } } \mathcal { L } _ { \mathrm { d i r e c t } } + \lambda _ { \mathrm { r e n d e r } } \mathcal { L } _ { \mathrm { r e n d e r } } .\tag{8}
$$

The direct objective supervises instance discrimination in the context-view embedding space, while the rendering objective further enforces instance consistency in 3D space. Collectively, these objectives facilitate instance embedding learning and enforce consistency across context and target views.

Semantic Aggregation and Propagation. After obtaining discriminative Gaussian instance embeddings, we jointly cluster the embeddings from all context views using HDB-SCAN [46] at inference time. Since each embedding is attached to a pixel-aligned Gaussian, the resulting clusters directly define cross-view consistent 3D Gaussian groups. We then extract dense semantic features $\mathbf { F } _ { v }$ from each context image using a frozen 2D open-vocabulary model [35]. For instance group $k ,$ its semantic prototype is obtained by pooling the features within its context-view partitions:

$$
\mathbf { f } _ { k } ^ { \mathrm { s e m } } = \frac { \displaystyle \sum _ { v \in \mathcal { C } } \sum _ { u } M _ { v , k } ( u ) \mathbf { F } _ { v } ( u ) } { \displaystyle \sum _ { v \in \mathcal { C } } \sum _ { u } M _ { v , k } ( u ) } ,\tag{9}
$$

where $M _ { v , k } ( u ) \in \{ 0 , 1 \}$ denotes the instance mask obtained by mapping Gaussian group k to its corresponding pixels in context view v through the pixel-Gaussian correspondence, and $\mathbf { f } _ { k } ^ { \mathrm { s e m } }$ is the corresponding semantic prototype. The instance assignments are subsequently attached to the pixelaligned Gaussians and rendered into target views to obtain target-view instance partitions. Semantic features are propagated by combining the rendered partition weights with their corresponding prototypes. Consequently, high-dimensional semantic features are stored once per instance rather than for every Gaussian. The low-dimensional embeddings organize Gaussians into instance groups, while the high-dimensional semantics are aggregated and propagated at the instance level.

Staged Curriculum Training. We adopt a three-stage curriculum training strategy. In Stage 1, we freeze the instance head and optimize the shared encoder, geometry decoder, and Gaussian decoder for scene reconstruction. The reconstruction objective ${ \mathcal { L } } _ { \mathrm { r e c } }$ combines $\ell _ { 1 }$ and SSIM photometric losses with an additional LPIPS loss, as well as pose, depth, point-map, and normal supervision. In Stage 2, we freeze the shared encoder, Geometry Decoder, and attribute head. We only train the instance head using ${ \mathcal { L } } _ { \mathrm { i n s } }$ . In Stage 3, we jointly fine-tune the full model with

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { r e c } } + \mathcal { L } _ { \mathrm { i n s } } ,\tag{10}
$$

This curriculum training establishes stable geometry and appearance before introducing instance learning, thereby improving training stability and preventing early optimization collapse. Joint optimization in the final stage further facilitates coordinated adaptation of the Geometry and Gaussian decoders. In addition, the staged training also supports instance learning without ground-truth instance labels. When such labels are unavailable, we first extract instance masks independently from each view using SAM2 [47]. The depth and camera geometry learned in Stage 1 are then used to project these masks across views, discard geometrically inconsistent correspondences, and associate the remaining masks according to their projected overlap. This process establishes consistent identities across the per-view masks and provides pseudolabels for instance supervision in Stage 2.

## C. Referential Scene Reasoning Framework

The Referential Scene Reasoning Framework (RSRF) enables structured referential reasoning over instance-grouped 3D scenes and achieves 3D referring segmentation. Its design includes 3D scene graph construction, expression-guided instance retrieval, and VLM-based reasoning with structured relational evidence and multi-view observations.

3D Scene Graph over Instance Groups. We organize the reconstructed scene into an instance-level relational graph $\mathcal { G } = ( \mathcal { N } , \mathcal { E } )$ , where each node corresponds to a cross-view consistent 3D Gaussian group and each ordered edge encodes the spatial relation of one group to another:

$$
\begin{array} { r } { \mathbf { n } _ { k } = \left( \mathbf { f } _ { k } ^ { \mathrm { s e m } } , \mathbf { g } _ { k } ^ { \mathrm { 3 D } } , \left\{ \mathbf { g } _ { v , k } ^ { \mathrm { 2 D } } , \nu _ { v , k } \right\} _ { v \in \mathcal { C } } \right) , } \\ { \mathbf { e } _ { k l } = \left( d _ { k l } ^ { \mathrm { 3 D } } , \Delta h _ { k l } , r _ { k l } , \left\{ \mathbf { r } _ { v , k l } ^ { \mathrm { 2 D } } \right\} _ { v \in \mathcal { C } } \right) . } \end{array}\tag{11}
$$

Here, $\mathbf { f } _ { k } ^ { \mathrm { s e m } }$ denotes the semantic prototype of node $k ,$ while $\mathbf { g } _ { k } ^ { \mathrm { 3 D } }$ summarizes its 3D center, spatial extent, and volume. The view-dependent attributes $\mathbf { g } _ { v , k } ^ { \mathrm { 2 D } }$ and $\nu _ { v , k }$ describe its projected geometry and visibility in view $v ,$ respectively. For an ordered node pair $( k , l )$ , the edge attributes encode their 3D distance $d _ { k l } ^ { \mathrm { 3 D } }$ , relative height $\Delta h _ { k l }$ , size ratio $r _ { k l }$ , and view-dependent

TABLE I  
QUANTITATIVE COMPARISON ON SCANNET WITH TWO INPUT VIEWS, FOLLOWING THE BENCHMARK ESTABLISHED BY LSM [7] AND UNI3R [8].
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue &amp; Year</td><td colspan="4">Source View</td><td colspan="5">Target View</td></tr><tr><td>mIoU↑</td><td>Acc.↑</td><td>rel↓</td><td>τ ↑</td><td>mIoU↑</td><td>Acc.↑</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>LSeg [35]</td><td>ICLR 2022</td><td>0.4701</td><td>0.7891</td><td></td><td></td><td>0.4819</td><td>0.7927</td><td></td><td></td><td></td></tr><tr><td>NeRF-DEF [23]</td><td>NeurIPS 2022</td><td>0.4540</td><td>0.7173</td><td>27.68</td><td>9.61</td><td>0.4037</td><td>0.6755</td><td>19.86</td><td>0.6650</td><td>0.3629</td></tr><tr><td>Feature 3DGS [3]</td><td>CVPR 2024</td><td>0.4453</td><td>0.7276</td><td>12.95</td><td>21.07</td><td>0.4223</td><td>0.7174</td><td>24.49</td><td>0.8132</td><td>0.2293</td></tr><tr><td>PixelSplat [17]</td><td>CVPR 2024</td><td></td><td></td><td></td><td></td><td></td><td></td><td>24.89</td><td>0.8392</td><td>0.1641</td></tr><tr><td>LSM [7]</td><td>NeurIPS 2024</td><td>0.5034</td><td>0.7740</td><td>3.38</td><td>67.77</td><td>0.5078</td><td>0.7686</td><td>24.39</td><td>0.8072</td><td>0.2506</td></tr><tr><td>AnySplat [21]</td><td>SIGGRAPH Asia 2025</td><td></td><td></td><td>6.35</td><td>47.57</td><td></td><td></td><td>22.08</td><td>0.8118</td><td>0.2480</td></tr><tr><td>IGGT [24]</td><td>ICLR 2026</td><td>0.5626</td><td>0.8145</td><td>3.15</td><td>66.94</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Uni3R [8]</td><td>CVPR 2026</td><td>0.5403</td><td>0.8255</td><td>3.87</td><td>61.37</td><td>0.5584</td><td>0.8268</td><td>25.53</td><td>0.8727</td><td>0.1380</td></tr><tr><td>Ours</td><td></td><td>0.5905</td><td>0.8294</td><td>2.14</td><td>82.23</td><td>0.5989</td><td>0.8302</td><td>26.23</td><td>0.8948</td><td>0.1230</td></tr></table>

2D relations $\mathbf { r } _ { v , k l } ^ { \mathrm { 2 D } }$ , including projected distance, direction, and overlap. Each node additionally retains references to its associated masks and visual observations, linking the structured graph representation to the corresponding multi-view evidence.

Instance Candidate Retrieval. Given a referring expression, RSRF decomposes it into target concepts, target cardinality, anchor instances, spatial relations, and visual attributes. These query components are then used to retrieve relevant nodes from the scene graph. Semantic relevance is measured by comparing the target and anchor concepts with the instancelevel semantic prototypes and open-vocabulary labels, while geometric relevance evaluates whether the spatial relations between candidate nodes and inferred anchors are compatible with the expression. Cross-view visibility is also considered to favor candidates supported by sufficient visual evidence. The resulting semantic, geometric, and visibility scores are combined to rank the graph nodes, and the highest-scoring nodes are retained for subsequent reasoning. This stage reduces the VLM reasoning space while preserving candidates that satisfy the principal semantic and spatial constraints of the expression.

VLM Reasoning with Structured Evidence. For each retrieved candidate, we construct a node-centric evidence bundle containing its semantic and geometric attributes, relations to neighboring nodes, cross-view visibility, and multiview observations. The visual evidence includes global scene overviews, highlighted instance masks, object-centric crops, and local contexts that expose both appearance and surrounding instances. A vision-language model (VLM) reasons over the referring expression, structured graph evidence, and multiview observations to identify the target 3D instance entities from the retrieved candidates, consistent with the inferred target cardinality. By restricting the VLM to selecting among the retrieved candidate nodes, RSRF formulates referential grounding as a closed-set selection problem over reconstructed 3D instances. Since each selected node maintains explicit correspondences to its observations and masks across views, the grounded 3D entities can be directly mapped to cross-view consistent segmentations without additional mask prediction.

## IV. EXPERIMENTS

## A. Experimental Setup

Implementation Details. We implement GroupForward and RSRF in PyTorch. For GroupForward, we use VGGT [20] as its multi-view encoder and initialize both the encoder and geometry decoder with pretrained VGGT weights. The remaining prediction heads are randomly initialized. We use gsplat [48] for differentiable Gaussian rasterization. We set the instance embedding dimension to $d _ { e } = 8 ,$ substantially smaller than the 512-dimensional LSeg features and the 64- or higher-dimensional embeddings used by existing baselines [3], [7], [8]. The learning rates are $1 0 ^ { - 6 }$ for the encoder and $1 0 ^ { - 5 }$ for the decoders. We empirically set $\lambda _ { \mathrm { p u l l } } = \lambda _ { \mathrm { p u s h } } = \lambda _ { \mathrm { d i c e } } =$ $\lambda _ { \mathrm { f o c a l } } = 0 . 5 , m = 1 . 0$ , and $\lambda _ { \mathrm { d i r e c t } } = \lambda _ { \mathrm { r e n d e r } } = 1 . 0$ . Following the staged curriculum in Sec. III-B, we train each stage for approximately 50k iterations in the two-view setting, which takes approximately 30 hours on 8 × NVIDIA A100 GPUs (80GB). Since GroupForward naturally supports a variable number of input views, we further fine-tune the two-view model for 50k iterations using 2–24 input views for multi-view settings. Following LSM and Uni3R [7], [8], we also adopt LSeg [35] as the semantic model. GroupForward aggregates and propagates the extracted LSeg features at the instance level. For RSRF, we adopt Qwen3-VL-8B-Instruct [39] as the VLM and deploy it with vLLM [49]. To bound the context length, we rank scene-graph nodes by semantic similarity, geometric relevance, and multi-view visibility, and retain the top-K nodes (K = 8) as candidates for VLM reasoning. We run RSRF on a single NVIDIA A100 GPU (80GB).

Datasets and Evaluation Protocol. We follow the twoinput-view ScanNet benchmark and evaluation protocol established by LSM [7] and Uni3R [8]. We train GroupForward on the ScanNet [50] training set, with strictly nonoverlapping training and test scenes. We further evaluate more input views on ScanNet and zero-shot generalization on Replica [51]. We report mean intersection over union (mIoU) and pixel accuracy (Acc.) for semantic reconstruction via input- and novel-view semantic segmentation, absolute relative error (rel) and threshold accuracy (τ ) for geometry estimation, and peak signal-to-noise ratio (PSNR), structural similarity index measure (SSIM) [52], and learned perceptual image patch similarity (LPIPS) [53] for novel-view synthesis. For referential scene reasoning, we construct referring expressions paired with target instances and evaluate novel-view referring segmentation against the corresponding ground-truth masks. We report mIoU averaged over all queries and novel views.

TABLE II  
QUANTITATIVE COMPARISON ON SCANNET WITH VARYING NUMBERS OF INPUT VIEWS FOR NOVEL-VIEW SEMANTIC SEGMENTATION AND SYNTHESIS.
<table><tr><td rowspan="2">Method</td><td colspan="5">4 Views</td><td colspan="5">6 Views</td></tr><tr><td>mIoU↑</td><td>Acc.↑</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>mIoU↑</td><td>Acc.↑</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>LSeg [35]</td><td>0.5002</td><td>0.7982</td><td></td><td></td><td></td><td>0.5381</td><td>0.8077</td><td></td><td></td><td></td></tr><tr><td>IGGT [24]</td><td>0.5247</td><td>0.7911</td><td></td><td></td><td></td><td>0.5799</td><td>0.8131</td><td></td><td></td><td></td></tr><tr><td>AnySplat [21]</td><td></td><td></td><td>21.82</td><td>0.8204</td><td>0.1415</td><td></td><td></td><td>22.28</td><td>0.8093</td><td>0.1509</td></tr><tr><td>Feature 3DGS [3]</td><td>0.5192</td><td>0.7250</td><td>23.78</td><td>0.7779</td><td>0.2661</td><td>0.5461</td><td>0.7399</td><td>24.36</td><td>0.7905</td><td>0.2356</td></tr><tr><td>LSM [7]</td><td>0.4949</td><td>0.8002</td><td>22.85</td><td>0.8066</td><td>0.1971</td><td>0.5323</td><td>0.8076</td><td>20.80</td><td>0.7986</td><td>0.2056</td></tr><tr><td>Uni3R [8]</td><td>0.5443</td><td>0.8118</td><td>25.32</td><td>0.8438</td><td>0.1340</td><td>0.5794</td><td>0.8186</td><td>25.34</td><td>0.8441</td><td>0.1422</td></tr><tr><td>Ours</td><td>0.5868</td><td>0.8277</td><td>27.14</td><td>0.9015</td><td>0.1083</td><td>0.6513</td><td>0.8467</td><td>26.51</td><td>0.8890</td><td>0.1198</td></tr><tr><td rowspan="9">LSeg [35] IGGT [24] AnySplat [21]</td><td></td><td></td><td>8 Views</td><td></td><td></td><td></td><td></td><td>16 Views</td><td></td><td></td></tr><tr><td>mIoU↑</td><td>Acc.↑</td><td></td><td></td><td>LPIPS↓</td><td>mIoU↑</td><td>Acc.↑</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>0.5466</td><td>0.8175</td><td>PSNR↑</td><td>SSIM↑</td><td></td><td>0.5387</td><td>0.8201</td><td></td><td></td><td></td></tr><tr><td>0.5719</td><td>0.8346</td><td></td><td></td><td></td><td>0.6002</td><td>0.8596</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>21.98</td><td>0.7724</td><td>0.1722</td><td></td><td></td><td>18.92</td><td>0.7105</td><td>0.2372</td></tr><tr><td>Feature 3DGS [3] 0.5810</td><td>0.7745</td><td>24.19</td><td>0.7929</td><td>0.2239</td><td>0.5046</td><td>0.8289</td><td>23.78</td><td>0.8184</td><td>0.1812</td></tr><tr><td>LSM [7]</td><td>0.5567 0.8202</td><td>20.63</td><td>0.7888</td><td>0.2125</td><td>0.5621</td><td>0.8319</td><td>19.07</td><td>0.7813</td><td>0.2262</td></tr><tr><td></td><td>0.5880 0.8359</td><td>24.99</td><td>0.8318</td><td>0.1580</td><td>0.5813</td><td>0.8484</td><td>23.35</td><td>0.7827</td><td>0.2089</td></tr><tr><td>Uni3R [8]</td><td>0.6623 0.8669</td><td>26.01</td><td>0.8740</td><td>0.1303</td><td>0.6880</td><td>0.8893</td><td>25.71</td><td>0.8629</td><td>0.1495</td></tr></table>

TABLE III

ZERO-SHOT GENERALIZATION FROM SCANNET TO REPLICA. RESULTS ARE AVERAGED OVER 2, 4, 6, AND 8 INPUT VIEWS.
<table><tr><td>Method</td><td>mIoU↑</td><td>Acc.↑</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>LSeg [35]</td><td>0.4690</td><td>0.8410</td><td></td><td></td><td></td></tr><tr><td>LSM [7]</td><td>0.4341</td><td>0.8233</td><td>16.73</td><td>0.6539</td><td>0.2814</td></tr><tr><td>Uni3R [8]</td><td>0.4754</td><td>0.8351</td><td>23.03</td><td>0.7880</td><td>0.1276</td></tr><tr><td>Ours</td><td>0.5633</td><td>0.8769</td><td>27.26</td><td>0.8790</td><td>0.1226</td></tr></table>

TABLE IV

COMPARISON OF REFERENTIAL SCENE REASONING CAPABILITIES WITH NOVEL-VIEW REFERRING SEGMENTATION PERFORMANCE.
<table><tr><td>Method</td><td>Language Understanding</td><td>3D Output</td><td>mIoU↑</td></tr><tr><td>Uni3R [8]</td><td>Phrase / Category</td><td>√</td><td>0.2538</td></tr><tr><td>Sa2VA [40]</td><td>Referential</td><td>X</td><td>0.5012</td></tr><tr><td>Ours</td><td>Referential</td><td>√</td><td>0.6496</td></tr></table>

Baselines. We primarily compare our method with semantic 3D reconstruction methods, including feed-forward 3DGS methods [7], [8], per-scene optimized methods [3], [23], and pointmap-based semantic reconstruction method [24]. We further include semantic-only [35] and reconstruction-only methods [17], [21] for comprehensive evaluation. We use the published results for the existing benchmark. For additional comparisons, we evaluate the baselines using their official implementations and open-source pretrained weights.

## B. Comparison on Semantic 3D Reconstruction

Quantitative Results on ScanNet. Table I compares the reconstruction performance of different methods on the existing two-input-view ScanNet benchmark [7], [8]. GroupForward achieves the best performance across all reported semantic, geometric, and appearance metrics. Unlike LSM and Uni3R, which attach high-dimensional semantic features to each Gaussian, GroupForward uses compact instance embeddings and propagates semantics through instance-level prototypes. Although IGGT [24] also learns instance-level features, it does not support novel-view synthesis and underperforms Group-Forward on source-view semantic metrics. The consistent improvements demonstrate the effectiveness of our instancegrouped formulation in improving semantic reconstruction while preserving strong geometry and appearance quality.

TABLE V  
COMPARISON OF DIFFERENT SEMANTIC MODELS FOR GROUPFORWARD.
<table><tr><td>Semantic Model</td><td>mIoU↑</td><td>Acc.↑</td></tr><tr><td>LSeg [35] (default)</td><td>0.5905</td><td>0.8294</td></tr><tr><td>OpenSeg [54]</td><td>0.4536</td><td>0.6292</td></tr><tr><td>CLIP [34]</td><td>0.1630</td><td>0.3759</td></tr></table>

Results with More Input Views. Table II reports results using 4, 6, 8, and 16 input views on ScanNet with our multiview model. GroupForward consistently achieves the best performance across all input-view settings in both novel-view semantic segmentation and novel-view synthesis. These results demonstrate that the advantages of our instance-grouped formulation extend to multi-view settings.

Zero-Shot Generalization. Table III evaluates zero-shot generalization across datasets, where our multi-view model trained on ScanNet is evaluated on Replica without finetuning. Results are averaged over settings with 2, 4, 6, and 8 input views. Our GroupForward achieves the best performance across all reported semantic and appearance metrics, consistently outperforming existing methods. These results demonstrate the strong cross-dataset generalization of our

![](images/8220faeab873efdb8835d54e789163b24c131f14b760dfbf34775e36e20a0739.jpg)  
Fig. 3. Qualitative comparisons of novel-view open-vocabulary segmentation on ScanNet. Our GroupForward produces more complete semantic regions with cleaner object boundaries and fewer cross-category inconsistencies than existing methods.

method to unseen scenes.

Qualitative Results. Fig. 3 and Fig. 4 present qualitative comparisons of novel-view semantic segmentation and novelview synthesis, respectively. GroupForward produces more complete semantic regions with clearer object boundaries and fewer cross-category inconsistencies in novel views. It also better preserves scene structure and visual details in the synthesized novel views. These observations are consistent with the quantitative improvements and show that instance-level aggregation promotes view-consistent semantic propagation without compromising rendering quality.

## C. Referential Scene Reasoning

Beyond semantic 3D reconstruction, we further evaluate the proposed Referential Scene Reasoning Framework (RSRF) built upon GroupForward through both quantitative and qualitative comparisons. As summarized in Table IV, existing methods face inherent limitations in this task. Semantic reconstruction methods such as Uni3R [8] rely on semantic similarity matching and therefore support only phrase- or category-level language queries, failing to handle complex referring expressions involving spatial relations and contextual constraints. Conversely, referring segmentation models that couple a segmentation model with a VLM, such as Sa2VA [40], can interpret complex referential language but do not support novel-view synthesis. We therefore directly take the novel views as input to Sa2VA for evaluation. Moreover, Sa2VA does not natively produce 3D outputs, resulting in inconsistent segmentation results across views. In contrast, RSRF grounds complex referring expressions to the cross-view consistent 3D instance groups reconstructed by GroupForward, jointly supporting complex referential understanding, viewconsistent segmentation, and native 3D output. Consistent with these capabilities, RSRF substantially outperforms both baselines on novel-view referring segmentation.

Fig. 5 presents qualitative comparisons. In the first three rows, which involve complex referring expressions, our method correctly identifies the target instances, whereas the compared methods either identify incorrect objects or fail to produce valid segmentations. In the relatively simple case shown in the fourth row, our method still yields higher-quality novel-view segmentation together with a consistent 3D output, while Sa2VA, despite achieving comparable 2D accuracy, cannot provide a 3D result. These results demonstrate the advantage of our instance-grounded reasoning framework for referential scene reasoning over multi-view inputs.

Ours

GT  
Feature 3DGS  
LSM  
Uni3R  
AnySplat  
![](images/faf5fd2503b2142fdc21bc8460fe8bedf378732a9cfc97627aa58dd5c52ef399.jpg)  
Fig. 4. Qualitative comparison of novel-view synthesis on ScanNet. Red boxes highlight representative rendering artifacts in competing methods, while our GroupForward better preserves scene structure and fine-grained visual details.

## D. Compatibility with Different Semantic Models

As shown in Table V, our GroupForward is compatible with different semantic models, including LSeg [35], OpenSeg [54], and CLIP [34]. Following the established benchmark [7], [8], semantic quality is evaluated using 2D multiclass segmentation including an explicit other class. LSeg achieves the best performance among the evaluated models. OpenSeg is less effective at modeling the catch-all other class, resulting in lower overall scores, while CLIP is less suitable for dense pixellevel alignment and fine-grained class discrimination due to its global image-text training objective. Since GroupForward aggregates dense semantic features into instance prototypes, feature quality directly affects semantic performance. We therefore use LSeg as the default semantic model, which is also consistent with LSM [7] and Uni3R [8] for a fair comparison.

## E. Ablation Studies

We conduct ablation studies on ScanNet to evaluate the contributions of key components and the effectiveness of pseudo-label supervision. The results are reported in Table VI.

Instance Components. For instance- and semantic-related components, we separately ablate the loss design and the instance-grouped representation. Removing the renderingbased objective $\mathcal { L } _ { \mathrm { r e n d e r } }$ and retaining only the context-view objective $\mathcal { L } _ { \mathrm { d i r e c t } }$ (w/o ${ \mathcal { L } } _ { \mathrm { r e n d e r } } ,$ row 2) leads to a clear drop in mIoU, validating the importance of rendering-level supervision for instance learning. We further remove the instance embedding representation entirely (w/o instance embedding, row 3) and follow LSM and Uni3R by attaching high-dimensional semantic features to the Gaussians and supervising the rendered feature maps. The larger performance degradation highlights the advantage of our instance-grouped representation over direct high-dimensional semantic feature rendering. Meanwhile, both variants exhibit only marginal changes in appearance quality, indicating that these components mainly contribute to semantic reconstruction.

Gauge-Aligned Self-Rendering. Removing gauge-aligned self-rendering (w/o self-rendering, row 4) degrades all appearance metrics, with lower PSNR and SSIM and higher LPIPS, while semantic performance decreases slightly. These results show that gauge-aligned self-rendering improves consistency between the reconstructed scene and predicted camera poses, thereby benefiting appearance reconstruction.

Staged Curriculum Training. We remove the staged curriculum and optimize all components jointly from the beginning (w/o curriculum training, row 5). This variant causes a substantial degradation across both semantic and appearance metrics. Since instance supervision relies on sufficiently reliable geometry and poses, introducing it too early leads to unstable optimization. The result indicates that the staged curriculum is important for establishing a stable reconstruction before instance learning.

Novel View Segmentation  
![](images/7f5e4a260405d78abf1aa22149358c68e0335205e1146da19edb6333d5599156.jpg)  
Fig. 5. Qualitative comparisons of referential scene reasoning. Our method correctly grounds target instances with accurate novel-view segmentation and corresponding 3D outputs, while competing methods fail to ground the target instances or lack native 3D outputs.

TABLE VI  
ABLATION STUDIES OF GROUPFORWARD ON SCANNET. STARTING FROM THE DEFAULT FULL MODEL, WE REMOVE OR REPLACE KEY COMPONENTS TOASSESS THE CONTRIBUTIONS OF INDIVIDUAL DESIGN CHOICES AND THE EFFECTIVENESS OF PSEUDO-LABEL SUPERVISION.
<table><tr><td>#</td><td>Method</td><td>mIoU↑</td><td>Acc.↑</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>1</td><td>Ours (default)</td><td>0.5989</td><td>0.8302</td><td>26.23</td><td>0.8948</td><td>0.1230</td></tr><tr><td colspan="7">Semantic ablation</td></tr><tr><td>2</td><td>w/o  $\mathcal { L } _ { \mathrm { r e n d e r } }$ </td><td>0.5772</td><td>0.8131</td><td>26.17</td><td>0.8924</td><td>0.1237</td></tr><tr><td>3</td><td>w/o instance embedding</td><td>0.5479</td><td>0.8096</td><td>26.19</td><td>0.8929</td><td>0.1234</td></tr><tr><td></td><td></td><td colspan="3">Rendering ablation</td><td></td><td></td></tr><tr><td>4</td><td>w/o self-rendering</td><td>0.5950</td><td>0.8297</td><td>26.11</td><td>0.8917</td><td>0.1267</td></tr><tr><td></td><td></td><td colspan="3">Training ablation</td><td></td><td></td></tr><tr><td>5</td><td>w/o curriculum training</td><td>0.4669</td><td>0.7486</td><td>18.92</td><td>0.6733</td><td>0.3122</td></tr><tr><td colspan="7">Pseudo-label ablation</td></tr><tr><td>6</td><td>w/ VGGT</td><td>0.5879</td><td>0.8278</td><td>25.97</td><td>0.8862</td><td>0.1287</td></tr><tr><td>7</td><td>w/ SAM2</td><td>0.5649</td><td>0.8221</td><td>26.16</td><td>0.8911</td><td>0.1287</td></tr></table>

Pseudo-Label Supervision. To evaluate the applicability of our method when ground-truth supervision is unavailable, we separately replace ground-truth geometric and instance supervision with predicted pseudo-labels and retrain the corresponding models. Using VGGT predictions for geometric supervision (w/ VGGT, row 6) results in only a moderate degradation, suggesting that predicted geometry remains sufficiently reliable for reconstruction and instance learning. Replacing ground-truth instance labels with SAM2 pseudo masks (w/ SAM2, row 7) also preserves competitive semantic and appearance performance. Although pseudo-supervision introduces some degradation relative to the default setting, both variants still maintain strong performance against other methods in Table I. This demonstrates that our method can effectively learn from both geometric and instance pseudolabels, reducing the reliance on ground-truth annotations.

## V. CONCLUSION

We introduced GroupForward, an instance-grouped feedforward Gaussian splatting model for semantic 3D reconstruction from sparse, unposed, and uncalibrated multi-view images. Rather than attaching high-dimensional semantic features to each Gaussian, GroupForward learns compact embeddings that organize the primitives into cross-view consistent instance groups, while open-vocabulary semantics are aggregated and propagated at the instance level. Built on these instances, we further introduce the Referential Scene Reasoning Framework (RSRF), which constructs a 3D scene graph and employs a VLM with structured graph evidence and multi-view observations to identify target 3D entities from complex referring expressions. Experiments show that GroupForward consistently outperforms existing methods in semantic 3D reconstruction across different input-view settings and generalizes well to unseen scenes, while RSRF substantially improves referring segmentation with view-consistent and native 3D outputs. Ablation studies further validate the effectiveness of the proposed components. Overall, our instance-grouped formulation enables explicit instance discrimination in feedforward semantic 3DGS and extends the language interaction capability from category- or phrase-level semantic querying to complex referential scene reasoning.

## REFERENCES

[1] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3d gaussian¨ splatting for real-time radiance field rendering,” ACM Trans. Graph., vol. 42, no. 4, pp. 139:1–139:14, 2023.

[2] M. Qin, W. Li, J. Zhou, H. Wang, and H. Pfister, “Langsplat: 3d language gaussian splatting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 20 051– 20 060.

[3] S. Zhou, H. Chang, S. Jiang, Z. Fan, Z. Zhu, D. Xu, P. Chari, S. You, Z. Wang, and A. Kadambi, “Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 21 676–21 685.

[4] J.-C. Shi, M. Wang, H.-B. Duan, and S.-H. Guan, “Language embedded 3d gaussians for open-vocabulary scene understanding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 5333–5343.

[5] Y. Wu, J. Meng, H. Li, C. Wu, Y. Shi, X. Cheng, C. Zhao, H. Feng, E. Ding, J. Wang et al., “Opengaussian: Towards point-level 3d gaussianbased open vocabulary understanding,” Advances in Neural Information Processing Systems (NeurIPS), vol. 37, pp. 19 114–19 138, 2024.

[6] W. Li, Y. Zhao, M. Qin, Y. Liu, Y. Cai, C. Gan, and H. Pfister, “Langsplatv2: High-dimensional 3d language gaussian splatting with 450+ FPS,” in Advances in Neural Information Processing Systems (NeurIPS), 2025.

[7] Z. Fan, J. Zhang, W. Cong, P. Wang, R. Li, K. Wen, S. Zhou, A. Kadambi, Z. Wang, D. Xu et al., “Large spatial model: End-toend unposed images to semantic 3d,” Advances in Neural Information Processing Systems (NeurIPS), vol. 37, pp. 40 212–40 229, 2024.

[8] X. Sun, H. Jiang, L. Liu, S. Nam, G. Kang, X. Wang, W. Sui, Z. Su, W. Liu, X. Wang et al., “Uni3r: Unified 3d reconstruction and semantic understanding via generalizable gaussian splatting from unposed multiview images,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 33 280–33 290.

[9] N. Snavely, S. M. Seitz, and R. Szeliski, “Photo tourism: Exploring photo collections in 3d,” in ACM SIGGRAPH 2006 Papers, 2006, pp. 835–846.

[10] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106, 2021.

[11] Q. Gu, A. Kuwajerwala, S. Morin, K. M. Jatavallabhula, B. Sen, A. Agarwal, C. Rivera, W. Paul, K. Ellis, R. Chellappa, C. Gan, C. M. de Melo, J. B. Tenenbaum, A. Torralba, F. Shkurti, and L. Paull, “Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning,” in International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 5021–5028.

[12] Y. Yang, H. Yang, J. Zhou, P. Chen, H. Zhang, Y. Du, and C. Gan, “3d-mem: 3d scene memory for embodied exploration and reasoning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 17 294–17 303.

[13] Z. Zhu, X. Wang, Y. Li, Z. Zhang, X. Ma, Y. Chen, B. Jia, W. Liang, Q. Yu, Z. Deng, S. Huang, and Q. Li, “Move to understand a 3d scene: Bridging visual grounding and exploration for efficient and versatile embodied navigation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025, pp. 8120–8132.

[14] A. Yu, V. Ye, M. Tancik, and A. Kanazawa, “pixelnerf: Neural radiance fields from one or few images,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 4578–4587.

[15] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “Dust3r: Geometric 3d vision made easy,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 20 697–20 709.

[16] V. Leroy, Y. Cabon, and J. Revaud, “Grounding image matching in 3d with mast3r,” in European Conference on Computer Vision (ECCV), 2024, pp. 71–91.

[17] D. Charatan, S. L. Li, A. Tagliasacchi, and V. Sitzmann, “pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 19 457–19 467.

[18] Y. Chen, H. Xu, C. Zheng, B. Zhuang, M. Pollefeys, A. Geiger, T.- J. Cham, and J. Cai, “Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images,” in European Conference on Computer Vision (ECCV), 2024, pp. 370–386.

[19] Q. Tian, X. Tan, Y. Xie, and L. Ma, “Drivingforward: Feed-forward 3d gaussian splatting for driving scene reconstruction from flexible surround-view input,” in Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), vol. 39, no. 7, 2025, pp. 7374–7382.

[20] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “Vggt: Visual geometry grounded transformer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 5294–5306.

[21] L. Jiang, Y. Mao, L. Xu, T. Lu, K. Ren, Y. Jin, X. Xu, M. Yu, J. Pang, F. Zhao et al., “Anysplat: Feed-forward 3d gaussian splatting from unconstrained views,” ACM Transactions on Graphics (TOG), vol. 44, no. 6, pp. 257:1–257:16, 2025.

[22] S. Zhi, T. Laidlow, S. Leutenegger, and A. J. Davison, “In-place scene labelling and understanding with implicit scene representation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 15 838–15 847.

[23] S. Kobayashi, E. Matsumoto, and V. Sitzmann, “Decomposing nerf for editing via feature field distillation,” Advances in Neural Information Processing Systems (NeurIPS), vol. 35, pp. 23 311–23 330, 2022.

[24] H. Li, Z. Zou, F. Liu, X. Zhang, F. Hong, Y. Cao, Y. LAN, M. Zhang, G. YU, D. Zhang, and Z. Liu, “Iggt: Instance-grounded geometry transformer for semantic 3d reconstruction,” in International Conference on Learning Representations (ICLR), 2026.

[25] C. Zhou, R. Wang, F. Luo, M. D. Pese, Z. Fan, Y. Zhong, and S. Huang,´ “Ff3r: Feedforward feature 3d reconstruction from unconstrained views,” arXiv preprint arXiv:2604.09862, 2026.

[26] S. Koch, J. Wald, H. Matsuki, P. Hermosilla, T. Ropinski, and F. Tombari, “Unified semantic transformer for 3d scene understanding,” Transactions on Machine Learning Research, 2026.

[27] C. Li, X. Huang, S.-F. Chng, H. Zhan, Q. Yan, and Y. Xu, “Fast3dis: Feed-forward anchored scene transformer for 3d instance segmentation,” arXiv preprint arXiv:2603.25993, 2026.

[28] Q. Li, J. Sun, L. An, Z. Su, H. Zhang, and Y. Liu, “Semanticsplat: Feedforward 3d scene understanding with language-aware gaussian fields,” arXiv preprint arXiv:2506.09565, 2025.

[29] Y. Sheng, J. Deng, X. Zhang, Y. Zhang, B. Hua, Y. Zhang, and J. Ji, “Spatialsplat: Efficient semantic 3d from sparse unposed images,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 26 404–26 414.

[30] X. Wang, C. Lan, H. Zhu, Z. Chen, and Y. Lu, “Gsemsplat: Generalizable semantic 3d gaussian splatting from uncalibrated image pairs,” arXiv preprint arXiv:2412.16932, 2024.

[31] Q. Tian, X. Tan, J. Gong, Y. Xie, and L. Ma, “Uniforward: Unified 3d scene and semantic field reconstruction via feed-forward gaussian splatting from only sparse-view images,” arXiv preprint arXiv:2506.09378, 2025.

[32] Q. Tian, X. Tan, J. Ying, X. Wang, Y. Xie, and L. Ma, “Fleg: Feedforward language embedded gaussian splatting from any views,” arXiv preprint arXiv:2512.17541, 2025.

[33] Q. Xu, D. Wei, L. Zhao, W. Li, Z. Huang, S. Ji, and P. Liu, “Siu3r: Simultaneous scene understanding and 3d reconstruction beyond feature alignment,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 38, 2025, pp. 110 800–110 830.

[34] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International Conference on Machine Learning (ICML), 2021, pp. 8748–8763.

[35] B. Li, K. Q. Weinberger, S. Belongie, V. Koltun, and R. Ranftl, “Language-driven semantic segmentation,” in International Conference on Learning Representations (ICLR), 2022.

[36] Y. Li, Q. Ma, R. Yang, H. Li, M. Ma, B. Ren, N. Popovic, N. Sebe, E. Konukoglu, T. Gevers et al., “Scenesplat: Gaussian splatting-based scene understanding with vision-language pretraining,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 4961–4972.

[37] M. Ma, Q. Ma, Y. Li, J. Cheng, R. Yang, B. Ren, N. Popovic, M. Wei, N. Sebe, E. Konukoglu et al., “Scenesplat++: A large dataset and comprehensive benchmark for language gaussian splatting,” Advances in Neural Information Processing Systems (NeurIPS), vol. 38, 2025.

[38] OpenAI, “GPT-4o System Card,” Aug. 2024. [Online]. Available: https://openai.com/index/gpt-4o-system-card/

[39] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.

[40] H. Yuan, X. Li, T. Zhang, Z. H. Huang, S. Xu, S. Ji, Y. Tong, L. Qi, J. Feng, and M.-H. Yang, “Sa2va: Marrying sam2 with llava for dense grounded understanding of images and videos,” arXiv preprint, 2025.

[41] Y. Jin, J. Li, T. Gu, Y. Liu, B. Zhao, J. Lai, Z. Gan, Y. Wang, C. Wang, X. Tan et al., “Efficient multimodal large language models: A survey,” Visual Intelligence, vol. 3, no. 1, p. 27, 2025.

[42] S. He, G. Jie, C. Wang, Y. Zhou, S. Hu, G. Li, and H. Ding, “Refersplat: Referring segmentation in 3d gaussian splatting,” in International Conference on Machine Learning (ICML), 2025, pp. 22 456–22 467.

[43] C. Shi, M. Chen, Y. Mao, C. Yang, X. Hu, J. Ding, and Z. Yu, “Realm: An mllm-agent framework for open world 3d reasoning segmentation and editing on gaussian splatting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 16 779–16 788.

[44] F. Milletari, N. Navab, and S.-A. Ahmadi, “V-net: Fully convolutional neural networks for volumetric medical image segmentation,” in International Conference on 3D Vision (3DV), 2016, pp. 565–571.

[45] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss´ for dense object detection,” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2980–2988.

[46] L. McInnes, J. Healy, S. Astels et al., “hdbscan: Hierarchical density based clustering.” J. Open Source Softw., vol. 2, no. 11, p. 205, 2017.

[47] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Radle, C. Rolland, L. Gustafson¨ et al., “Sam 2: Segment anything in images and videos,” in International Conference on Learning Representations (ICLR), 2025.

[48] V. Ye, R. Li, J. Kerr, M. Turkulainen, B. Yi, Z. Pan, O. Seiskari, J. Ye, J. Hu, M. Tancik, and A. Kanazawa, “gsplat: An open-source library for gaussian splatting,” Journal of Machine Learning Research, vol. 26, no. 34, pp. 1–17, 2025.

[49] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for large language model serving with pagedattention,” in Proceedings of the ACM Symposium on Operating Systems Principles (SOSP), 2023.

[50] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner, “Scannet: Richly-annotated 3d reconstructions of indoor scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[51] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma et al., “The replica dataset: A digital replica of indoor spaces,” arXiv preprint arXiv:1906.05797, 2019.

[52] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE Transactions on Image Processing (TIP), vol. 13, no. 4, pp. 600–612, 2004.

[53] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 586–595.

[54] G. Ghiasi, X. Gu, Y. Cui, and T.-Y. Lin, “Scaling open-vocabulary image segmentation with image-level labels,” in European Conference on Computer Vision (ECCV), 2022, pp. 540–557.