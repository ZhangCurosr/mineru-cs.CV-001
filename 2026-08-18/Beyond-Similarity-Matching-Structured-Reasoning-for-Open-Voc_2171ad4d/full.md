# Beyond Similarity Matching: Structured Reasoning for Open-Vocabulary Referring Segmentation in 3DGS

Yizhao Wang Xinfa Wang School of Computer Science, Henan Institute School of Computer Science, Henan Institute ofScience and Technology ofScience and Technology cswyz@stu.hist.edu.cn, ORCID: wangxf@hist.edu.cn, ORCID: 0009-0003-7056-7264 0000-0002-6293-5624

Jingbo Wang Ting Li School of Computer Science, Henan Institute School of Computer Science, Henan Institute of Science and Technology of Science and Technology

Guantao Zhang Yafeng Han School of Computer Science, Henan Institute School of Computer Science, Henan Institute of Science and Technology of Science and Technology

Guohong Gao Yuhe Xia School ofComputer Science, Henan Institute School ofComputer Science, Henan Institute of Science and Technology of Science and Technology

Abstract Open-vocabulary referring segmentation in 3D Gaussian Splatting (3DGS) requires a neural model to select Gaussian primitives according to free-form language expressions. Existing 3DGS-based methods usually rely on global text-region similarity, which is weak for queries involving attributes, reference objects, spatial relations, and fine-grained parts. This often causes target-reference confusion, granularity mismatch, part-whole leakage, and relation violations. We propose QAGaussian, a query-adaptive neural reasoning framework for language-guided Gaussian primitive selection. QAGaussian first learns query-conditioned multi-scale Gaussian slots as diferentiable candidates whose receptive fields are shaped by the input expression. It then builds a relation-aware slot graph with language-conditioned edge weighting to propagate target-reference, attribute, part-whole, and contextual evidence. A granularity-adaptive router softly combines region-level, object-level, part-level, attribute-aware, and relation-aware mask branches, followed by relation-constrained refinement for spatial, part-whole, attribute, and geometric consistency. QAGaussian is pretrained only on Mosaic3D-5.6M for Gaussian-text alignment and evaluated on independent benchmarks without target-dataset fine-tuning. It achieves 47.2 Avg. mIoU and 63.2 Avg. F1, outperforming the strongest 3DGS referring baseline by 2.7 mIoU points and 2.9 F1 points. It also improves Part-mIoU from 38.6 to 43.4, Rel-mIoU from 44.4 to 50.8, and reduces target-reference confusion from 10.8 to 7.4. These results demonstrate that query-conditioned slot learning, relation-aware graph reasoning, and adaptive routing provide an efective neural modeling strategy for open-vocabulary referring segmentation in 3DGS. The code is available at https://github.com/zqeslwyz/QAGaussian.

Keywords: 3D Gaussian Splatting; Referring Segmentation; Open-Vocabulary Segmentation; Query-Conditioned Slot Learning; Graph Neural Reasoning; Adaptive Routing

## 1. Introduction

Neural vision-language models have made open-vocabulary perception increasingly practical by aligning visual representations with natural language supervision (Jia et al., 2021; J. Li et al., 2025;

Radford et al., 2021; Tschannen et al., 2025; Wang et al., 2024; X. Zhai et al., 2023). In 3D scenes, however, language-conditioned perception is not only a recognition problem. A user may ask for “the red cup to the left of the monitor”, “the handle of the kettle”, or “the chair leg near the wall”, where the correct output depends on a target category, attributes, reference objects, spatial relations, and part-whole structure. Such queries require a neural model to learn query-conditioned visual units, reason over relations among candidate units, and adapt the prediction granularity to the expression. This requirement is more demanding than closed-set 3D recognition and motivates a representation-learning formulation for open-vocabulary referring segmentation.

3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) provides an explicit primitive-based scene representation with eficient rendering, making it attractive for semantic feature learning and language-grounded scene understanding. Recent methods attach language or semantic features to Gaussian primitives (Qin et al., 2024; Y. Wu et al., 2024; S. Zhou et al., 2024), group Gaussians for mask-aware understanding (Cen et al., 2025; Ye et al., 2024a), and scale Gaussian representation learning with large 3D-language datasets (Lee et al., 2025; Y. Li et al., 2025; Ma et al., 2025). Despite this progress, most open-vocabulary 3DGS segmentation pipelines still reduce a free-form expression to a global text embedding and select regions by text-region similarity. This strategy can work for simple category-name prompts, but it lacks an explicit neural mechanism for decomposing a structured query, constructing query-specific candidates, propagating relation evidence, and selecting an appropriate output granularity.

This limitation leads to four observable failure modes. First, target-reference confusion occurs when the model activates both the referred target and the reference object because they are mentioned in the same sentence. Second, granularity mismatch appears when the expression asks for a part or an attribute-constrained subset but the model predicts a whole object or a broad region. Third, part-whole leakage occurs when a predicted part spills outside its parent object, especially for fine structures defined by PartNet, PartNet-Mobility, and ShapeNetPart (Mo et al., 2019; Xiang et al., 2020; Yi et al., 2016). Fourth, relation violation occurs when a mask is semantically plausible but inconsistent with the spatial relation in the query, a problem also observed in language-grounded 3D scene reasoning (Achlioptas et al., 2020; D. Z. Chen et al., 2020; Werby et al., 2024; Zhang et al., 2023). These failure modes indicate that Gaussian-level referring segmentation should be treated as query-conditioned neural reasoning over primitives rather than as static similarity matching.

We propose QAGaussian, a query-adaptive neural framework for open-vocabulary referring segmentation in 3DGS. The core idea is to convert millions of Gaussian primitives into a compact set of diferentiable, query-conditioned Gaussian slots, perform relation-aware reasoning over these slots, and dynamically route the prediction to the segmentation branches required by the current expression. Unlike fixed object proposals or direct primitive-text matching, the proposed slot formulation lets the language query shape candidate formation itself. The relation-aware slot graph then parameterizes target-reference, attribute, part, and contextual interactions with languageconditioned edges. Finally, a granularity-adaptive router softly fuses region-level, object-level, part-level, attribute-aware, and relation-aware mask branches, followed by relation-constrained refinement for spatial, part-whole, attribute, and geometric consistency.

QAGaussian is pretrained only on Mosaic3D-5.6M (Lee et al., 2025) for Gaussian-text semantic alignment and is evaluated on multiple independent benchmarks without target-dataset fine-tuning. It achieves an Avg. mIoU of 47.2 and an Avg. F1-score of 63.2, improving over the strongest 3DGS referring baseline ReferSplat (S. He et al., 2025) by 2.7 mIoU points and 2.9 F1 points under our unified protocol. The gains are more pronounced on complex query types: Part-mIoU improves from 38.6 to 43.4, Rel-mIoU improves from 44.4 to 50.8, and target-reference confusion decreases from 10.8 to 7.4. These results support the claim that query-conditioned slots, relation-aware graph reasoning, and adaptive routing address concrete modeling failures rather than merely adding engineering components. The main contributions are summarized as follows:

• We formulate open-vocabulary referring segmentation in 3DGS as query-conditioned Gaussian primitive selection, and introduce a multi-scale Gaussian slot learning mechanism that produces diferentiable candidates whose receptive fields are shaped by the input expression.

• We develop a relation-aware slot graph with language-conditioned edge weighting to model target-reference, attribute, part-whole, and contextual interactions among Gaussian slots, reducing target-reference confusion and relation violations.

• We propose a granularity-adaptive neural routing and consistency learning scheme that softly combines region-level, object-level, part-level, attribute-aware, and relation-aware mask branches and refines the output with spatial, part-whole, attribute, and geometric constraints.

• We conduct extensive zero-shot cross-benchmark evaluation, diagnostic analysis, and ablation studies, showing state-of-the-art overall performance and clear gains on part-level, relation-dependent, compositional, and language-robust referring segmentation.

## 2. Related Work

## 2.1 Open-Vocabulary Segmentation and Gaussian Semantic Learning

Open-vocabulary segmentation transfers language-aligned visual representations to dense prediction. In 2D vision, CLIP-based dense prediction, mask-level adaptation, unified decoders, and reasoning segmentation models enable segmentation from category names or free-form prompts (Lai et al., 2024; B. Li et al., 2022; Liang et al., 2023; J. Xu et al., 2022; C. Zhou et al., 2022; Zou, Dou, et al., 2023; Zou, Yang, et al., 2023). SAM and language-grounded variants further provide category-agnostic and text-conditioned mask proposals (Kirillov et al., 2023; F. Li et al., 2024; S. Liu et al., 2024; T. Ren et al., 2024). These methods provide strong 2D priors, but their masks must be lifted, fused, and reconciled across views before they can support 3D primitive-level prediction.

Open-vocabulary 3D methods extend language-aligned perception to point clouds, neural fields, 3D instances, and semantic maps. DFF and LERF learn language-aligned radiance fields (Kerr et al., 2023; Kobayashi et al., 2022), while OpenScene, ConceptFusion, OpenMask3D, and Open3DIS transfer 2D foundation-model knowledge to 3D scenes (Jatavallabhula et al., 2023; K. Liu et al., 2023; Nguyen et al., 2024; Takmaz et al., 2023). Recent 3D instruction-tuning and dense-captioning benchmarks further emphasize object-level and scene-level language grounding (Hu et al., 2025; Yan et al., 2025). These studies improve open-set recognition, but many of them still treat language mainly as a semantic retrieval signal rather than as a structure that should condition candidate formation and reasoning.

3DGS has recently been extended from real-time rendering to semantic understanding, editing, grouping, and open-vocabulary segmentation (Kerbl et al., 2023). Feature 3DGS, LangSplat, LEGaussians, and OpenGaussian attach distilled semantic or language features to Gaussian primitives (Qin et al., 2024; Shi et al., 2023; Y. Wu et al., 2024; S. Zhou et al., 2024). Gaussian Grouping, SAGA, GaussianCut, OpenSplat3D, PanoGS, and CAGS explore mask-aware grouping, interactive segmentation, open-vocabulary instance segmentation, panoptic understanding, and context-aware Gaussian reasoning (Cen et al., 2025; Piekenbrinck et al., 2025; Sun et al., 2025; Ye et al., 2024a, 2024b; H. Zhai et al., 2025). However, these methods mostly target category-level open-vocabulary segmentation or instance-level recognition. QAGaussian difers by learning query-conditioned Gaussian slots and performing relation-aware neural reasoning for expressions that specify attributes, references, spatial relations, and parts.

## 2.2 3D Referring Grounding and Closest 3DGS Baselines

3D referring expression grounding localizes objects in point clouds or reconstructed scenes according to natural language. ReferIt3D and ScanRefer established object-language matching benchmarks for fine-grained 3D reference resolution (Achlioptas et al., 2020; D. Z. Chen et al., 2020). InstanceRefer, 3DVG-Transformer, and TransRefer3D further improve proposal matching and contextual reasoning (D. He et al., 2021; Yuan et al., 2021; Zhao et al., 2021). Later methods such as BUTD-DETR, 3D-SPS, MVT, 3D-VisTA, and Multi3DRefer strengthen detection, language-guided point selection, multi-view fusion, 3D vision-language pretraining, and multi-target grounding (Huang et al., 2022; Jain et al., 2022; Luo et al., 2022; Zhang et al., 2023; Zhu et al., 2023). Related visual grounding and referring video object segmentation studies also show the value of semantic decomposition and memory-based target tracking for language-conditioned mask prediction (Z. Liu et al., 2025; J. Wu et al., 2025).

These grounding methods demonstrate that reference resolution requires more than category recognition, but their outputs are usually bounding boxes, point selections, or object instances. Such outputs are not naturally aligned with 3DGS primitives and are less suitable for part-level or attribute-constrained Gaussian segmentation. ReferSplat is the closest baseline because it explicitly studies referring segmentation in 3DGS and introduces spatially aware language modeling (S. He et al., 2025). OpenGaussian provides open-vocabulary Gaussian-level semantic prediction, while CAGS adds contextual Gaussian reasoning for open-vocabulary scene understanding (Sun et al., 2025; Y. Wu et al., 2024). Nevertheless, OpenGaussian mainly relies on language-feature similarity, CAGS focuses on context-aware open-vocabulary understanding rather than free-form referring segmentation, and ReferSplat emphasizes spatial-language matching without explicit query-conditioned slot learning or adaptive multi-branch routing. QAGaussian addresses these gaps by making the candidate set, relation graph, and mask granularity all conditioned on the current expression.

Relation-aware 3D scene graphs and open-vocabulary semantic graphs provide another line of structured 3D reasoning. Classic 3D scene graphs model objects, attributes, and spatial relations (Armeni et al., 2019; Wald et al., 2020; S.-C. Wu et al., 2021), while ConceptGraphs, HOV-SG, Open3DSG, and CLIP-driven scene graph generation connect 3D relations with visionlanguage models (L. Chen et al., 2024; Gu et al., 2023; Koch et al., 2024; Werby et al., 2024). These graphs are often scene-level and relatively static. Referring segmentation instead requires a compact graph that changes with the expression, since the same object can be a target, reference, contextual distractor, or parent object under diferent queries. This motivates the query-specific slot graph used in QAGaussian.

## 2.3 Slot-Based, Graph-Based, and Adaptive Neural Reasoning

The proposed framework is also related to object-centric representation learning. Slot Attention learns a set of latent slots that bind visual elements into object-centric representations (Locatello et al., 2020). This idea is useful for scenes containing multiple entities, but standard slot learning is usually image-level and query-agnostic. In contrast, QAGaussian uses the language expression to modulate slot queries and produces multi-scale Gaussian slots that may correspond to regions, objects, parts, or contextual candidates. This turns slot generation into a query-conditioned primitive selection mechanism rather than a generic object discovery module.

Graph neural networks provide a natural tool for modeling interactions among structured candidates. GCNs, GraphSAGE, and graph attention networks learn representations by propagating information through graph neighborhoods and attention-weighted edges (Hamilton et al., 2017; Kipf & Welling, 2017; Veličković et al., 2018). In language-guided 3D segmentation, the graph should not be fixed by geometry alone because the same spatial layout can imply diferent evidence under diferent relation words. QAGaussian therefore constructs a relation-aware slot graph whose edge weights depend jointly on spatial descriptors and parsed relation cues, allowing target-reference and part-whole evidence to be interpreted according to the current query.

![](images/f707d4e6b54adbe1a6ddae6874eeb7a29a9fa36bc42521ba9544aae8bbd6a44a.jpg)

![](images/e8fb717b7985194163ac407691e11cc175bfec22e504e497030464473b4dd2de.jpg)  
Figure 1: Overall framework of QAGaussian. Given a 3D Gaussian Splatting scene and a free-form referring expression, the model generates query-conditioned multi-scale Gaussian slots, performs relation-aware slot reasoning, adaptively fuses multiple mask prediction branches, and refines the result to obtain the final Gaussian-level referring segmentation mask.

Adaptive neural computation and routing mechanisms are also relevant. Mixture-of-experts models learn input-dependent combinations of specialized subnetworks (Jacobs et al., 1991; Shazeer et al., 2017), while dynamic routing in capsule networks iteratively assigns lower-level entities to higher-level capsules (Sabour et al., 2017). QAGaussian follows the general principle of input-conditioned routing, but applies it to Gaussian-level referring segmentation: the router predicts soft weights over region-level, object-level, part-level, attribute-aware, and relation-aware mask branches. This design explicitly links routing decisions to query granularity and slot relations, enabling the model to handle compositional expressions without committing to a single hard query type.

## 3. Methodology

We formulate open-vocabulary referring segmentation in 3DGS as query-conditioned primitive selection. Given a Gaussian scene $\mathcal { G } = \{ g _ { i } \} _ { i = 1 } ^ { N }$ and a free-form expression �, QAGaussian predicts a probability mask $M \in [ 0 , 1 ] ^ { N }$ over Gaussian primitives. As shown in Fig. 1, the framework follows a unified adaptive reasoning pipeline: hierarchical slot generation produces query-specific soft candidates, relation-aware graph reasoning models candidate interactions, and dynamic branch aggregation with geometric refinement yields the final mask. This formulation targets the central ambiguity of natural language queries: the referred entity may correspond to a region, an object, a part, an attribute-constrained subset, or a relation-dependent target.

The design follows two principles. First, all intermediate representations are kept at the Gaussian or slot level, so that the final prediction can be rendered from arbitrary viewpoints without converting back from boxes or point-cloud instances. Second, hard decisions are delayed until the final output stage. Candidate formation, branch selection, and refinement are all represented by soft weights, which makes the pipeline trainable end-to-end and reduces the risk of discarding the correct target early when the expression is ambiguous.

## 3.1 Query-Conditioned Hierarchical Slot Generation

The first question is how to obtain candidate regions from millions of Gaussian primitives without committing to a fixed object proposal set. A referring expression may describe a small part, a complete object, or a broader contextual region, so a single-scale candidate representation is inherently brittle. We therefore use query-conditioned slots to let the language expression decide which Gaussian groups should become candidate units.

QAGaussian first maps the Gaussian scene and the expression into a shared language-aligned feature space. Each Gaussian primitive is represented by its learned appearance feature, opacityaware geometric attributes, center position, scale, and view-consistent semantic feature distilled during pretraining. These signals are projected by the Gaussian encoder into primitive features $H _ { G } \in \mathbb { R } ^ { N \times D }$ . In parallel, the text encoder produces a global query embedding $z _ { Q }$ , and a lightweight parser extracts structured query cues

$$
\mathcal { Z } _ { Q } = \{ z _ { t a r } , z _ { a t t r } , z _ { r e l } , z _ { r e f } , z _ { p a r t } \} .\tag{1}
$$

These cues represent target, attribute, relation, reference, and part information when they are present in the expression. The parser does not provide query-type labels to the model. Instead, it only decomposes the expression into soft semantic cues, which are later used to condition slot generation, relation weighting, and branch routing.

To capture the hierarchical nature of 3D entities, we pool Gaussian primitives into multi-scale tokens $H _ { V } ^ { s }$ and modulate learnable slot queries with the global query context:

$$
H _ { V } ^ { s } = \psi _ { s } ( H _ { G } ) , \qquad Q _ { c } ^ { s } = Q _ { 0 } ^ { s } + \mathrm { M L P } _ { s } ( z _ { Q } ) , \quad s \in \{ 1 , 2 , 3 \} .\tag{2}
$$

In our implementation, $s = 1 , 2 , 3$ correspond to fine, middle, and coarse Gaussian token scales. The fine scale preserves local details for small parts and thin structures, the middle scale captures objectlevel components, and the coarse scale provides larger context for region-level and relation-dependent expressions. The conditioned queries interact with Gaussian tokens through slot attention,

$$
S ^ { s } = \mathrm { S l o t A t t n } _ { s } ( Q _ { c } ^ { s } , H _ { V } ^ { s } ) ,\tag{3}
$$

where each generated slot represents a soft Gaussian candidate. Unlike a fixed proposal, a slot is not required to correspond to a complete object. It may cover a local part, a complete instance, or a contextual region depending on the query. For slot $j ,$ its Gaussian assignment and activeness score are predicted as

$$
A _ { j , i } = \sigma ( s _ { j } ^ { \top } W _ { m } h _ { i } ) , \qquad o _ { j } = \sigma ( w _ { o } ^ { \top } s _ { j } ) .\tag{4}
$$

Here, $A _ { j , i }$ measures the soft membership of Gaussian primitive $g _ { i }$ to slot $j ,$ , and $o _ { j }$ estimates whether the slot corresponds to a meaningful candidate. Low-activeness slots are not forced to represent physical objects; they serve as inactive capacity that allows the model to use a fixed slot budget across scenes of diferent complexity. Thus, the model obtains a compact set of query-conditioned candidates whose efective receptive fields are adapted to the expression rather than fixed by a single proposal scale. This step also reduces the computational burden of later reasoning, since graph modeling and routing are performed over slots rather than millions of primitives.

## 3.2 Relation-Aware Graph Reasoning and Adaptive Routing

After candidate generation, the main dificulty shifts from finding plausible regions to selecting the correct one under linguistic constraints. For example, a target and its reference object may both be semantically compatible with the query, but only one satisfies the specified relation or part-whole structure. This motivates explicit relation modeling among slots and a router that adapts the segmentation behavior to the query type.

Slot candidates alone do not resolve expressions involving references, relations, or part-whole structures. We therefore build a lightweight graph over active slots. Nodes are selected according to slot activeness and query relevance, while edges connect spatially close slots, semantically similar slots, and slots that may satisfy relation or containment cues. For slot pair $( i , j )$ , the edge descriptor $r _ { i j }$ encodes relative position, overlap, containment, scale ratio, and feature similarity. Its query-conditioned edge weight is

$$
a _ { i j } = \sigma \left( { \mathrm { M L P } } _ { r e l } ( \left[ r _ { i j } , z _ { r e l } \right] ) \right) ,\tag{5}
$$

which allows the same spatial configuration to be interpreted diferently under diferent relation words. For example, two slots with a similar relative displacement may receive diferent weights for queries involving “next to”, “beneath”, or “in front $\mathrm { o f } ^ { \prime \prime }$ . This learnable relation weighting avoids relying only on hand-crafted geometric rules and makes the graph more flexible for open-vocabulary expressions. A graph encoder propagates these structural priors and outputs relation-enhanced slot features:

$$
H _ { Q } = \operatorname { G E n c } ( S , { \mathcal { E } } _ { Q } , { \mathcal { Z } } _ { Q } ) .\tag{6}
$$

During message passing, target-like slots receive information from reference-like and context-like slots, while distractor slots are suppressed when they are inconsistent with the query relation. This graph is query-specific: the same scene can produce diferent efective edges when the expression changes.

The updated slots are passed to a query-adaptive granularity router. The router predicts branch weights $\pi _ { b }$ over region-level, object-level, part-level, attribute-aware, and relation-aware branches. Each branch estimates the contribution of slot � as

$$
p _ { j } ^ { b } = \sigma \left( \mathrm { M L P } _ { b } ( [ h _ { j } ^ { Q } , z _ { Q } , o _ { j } ] ) \right) .\tag{7}
$$

The branches have diferent roles. The region-level branch favors broad areas and contextual descriptions, the object-level branch focuses on instance-like targets, the part-level branch emphasizes fine slots and containment cues, the attribute-aware branch strengthens compatibility with color, material, or shape descriptors, and the relation-aware branch relies more on target-reference interactions. The initial Gaussian mask is assembled by weighted slot aggregation:

$$
M _ { 0 } = \sum _ { b } \pi _ { b } \left( \sum _ { j } p _ { j } ^ { b } A _ { j } \right) .\tag{8}
$$

This separates what the expression asks for, represented by routing weights, from how the target should be segmented, represented by branch-specific slot aggregation. Because the router outputs a soft distribution rather than a single hard branch, a compositional query can simultaneously use attribute-aware and relation-aware evidence, while a part-level query can still borrow object-level context from its parent object.

## 3.3 Relation-Constrained Refinement and Learning Objectives

The routed mask provides a query-aware prediction, but local Gaussian assignments can still be noisy because 3DGS primitives are optimized for rendering rather than semantic boundaries. In particular, relation-dependent and part-level queries are sensitive to small fragments, leakage to reference objects, and masks that violate containment or spatial consistency. The refinement stage therefore introduces lightweight structural priors without replacing the mask predictor with a heavier decoder.

The initial mask $M _ { 0 }$ may contain fragmented Gaussians or candidates that violate the query relation. QAGaussian refines it with four lightweight consistency priors: spatial relation consistency, part-whole consistency, attribute compatibility, and Gaussian geometric continuity. These priors form the refinement loss

$$
\mathcal { L } _ { r e f } = \lambda _ { r e l } \mathcal { L } _ { r e l } + \lambda _ { p a r t } \mathcal { L } _ { p a r t } + \lambda _ { a t t r } \mathcal { L } _ { a t t r } + \lambda _ { g e o } \mathcal { L } _ { g e o } ,\tag{9}
$$

where unavailable cues are masked out for expressions that do not contain the corresponding structure. The spatial relation term penalizes masks that do not satisfy the parsed relation with respect to the reference slot. The part-whole term discourages a predicted part from extending outside its parent object candidate. The attribute term improves compatibility between selected Gaussians and attribute cues, and the geometric term suppresses isolated fragments by encouraging local continuity among neighboring primitives. The refinement module updates the mask in a small number of iterations and produces the final soft mask $M _ { f i n a l }$

QAGaussian is pretrained end-to-end on Mosaic3D-5.6M without fine-tuning on downstream benchmarks. The training data provides language descriptions and Gaussian-level supervision after unified conversion, which enables Gaussian-language semantic alignment before independent evaluation. Slot-level learning uses Hungarian matching between predicted slots and ground-truth Gaussian masks. The matching cost considers mask overlap, Dice distance, and slot activeness, so that meaningful slots are assigned to annotated masks and redundant slots remain unused. Matched slots are supervised by BCE and Dice mask losses, while unmatched slots are encouraged to remain inactive. The slot objective is

$$
\begin{array} { r } { \mathcal { L } _ { s l o t } = \mathcal { L } _ { m a s k } ^ { s l o t } + \beta _ { a c t } \mathcal { L } _ { a c t } . } \end{array}\tag{10}
$$

The router is trained with soft pseudo-labels derived from expression structure and mask geometry, since a single query may require multiple reasoning strategies. For instance, a query containing a part phrase and a spatial relation should not be forced into only one branch. Soft routing supervision allows the model to combine multiple reasoning modes while still learning a preference for the most relevant branch. The full objective is

$$
\mathcal { L } = \sum _ { u \in \Omega } \beta _ { u } \mathcal { L } _ { u } , \quad \Omega = \{ s l o t , r o u t e , s e g , a l i g n , r e f \} .\tag{11}
$$

Here, $\mathcal { L } _ { s e g }$ supervises the fused Gaussian mask, $\mathcal { L } _ { a l i g n }$ aligns positive slot-query pairs and separates negatives, $\mathcal { L } _ { r o u t e }$ regularizes branch selection, and $\beta _ { u }$ denotes the corresponding loss weight.

During inference, the model takes a reconstructed 3DGS scene and a free-form expression as input. The parser extracts query cues, the slot generator produces multi-scale soft candidates, the graph encoder updates their relation-aware representations, the router fuses branch-specific predictions, and the refinement module outputs $M _ { f i n a l }$ . The model keeps slot assignments and masks soft until the final stage. This avoids early commitment to an incorrect candidate and preserves the information needed by relation-constrained refinement. The discrete segmentation output is obtained by thresholding

$$
\hat { M } = \mathbb { I } ( M _ { f i n a l } > \tau ) ,\tag{12}
$$

where the threshold and all hyperparameters are fixed after pretraining and shared across independent benchmarks.

Table 1: Experimental settings and main hyperparameters.
<table><tr><td>Item</td><td>Setting</td></tr><tr><td>Pretraining dataset</td><td>Mosaic3D-5.6M</td></tr><tr><td>Hidden dimension D</td><td>256</td></tr><tr><td>Gaussian tokenization scales</td><td>3</td></tr><tr><td>Slots per scale  $( K _ { 1 } , K _ { 2 } , K _ { 3 } )$ </td><td>96 / 64 / 32</td></tr><tr><td>Total slots K</td><td>192</td></tr><tr><td>Graph neighbors per slot</td><td>16</td></tr><tr><td>Slot attention iterations</td><td>3</td></tr><tr><td>Graph encoder layers</td><td>2</td></tr><tr><td>Refinement iterations</td><td>3</td></tr><tr><td>Mask threshold τ</td><td>0.5</td></tr><tr><td>Slot activeness threshold  $\tau _ { o }$ </td><td>0.3</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Initial learning rate</td><td> $1 . 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td><td> $1 . 0 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Batch size</td><td>32</td></tr><tr><td>Training epochs</td><td>30</td></tr><tr><td>Learning rate schedule</td><td>Cosine decay</td></tr><tr><td>Warm-up epochs</td><td>2</td></tr></table>

## 4. Experiments

## 4.1 Experimental Settings

QAGaussian is pretrained on Mosaic3D-5.6M to learn open-vocabulary Gaussian-language alignment and query-conditioned Gaussian mask prediction. Unless otherwise specified, all hyperparameters are selected on the validation split of Mosaic3D-5.6M and kept fixed across all independent evaluations. We trained and evaluated on eight NVIDIA RTX 2080Ti GPUs.

For visualization, the predicted Gaussian mask can be rendered to arbitrary camera views with the original 3DGS rasterizer. Implementation details are shown in Table 1.

We compare QAGaussian with five groups of baselines: multi-view 2D projection methods, 3D referring grounding methods, open-vocabulary 3D segmentation and mapping methods, 3DGS openvocabulary segmentation methods, and 3DGS referring segmentation methods, their predictions are converted to Gaussian primitives using the unified protocol described in Sec. 4.2.

## 4.2 Datasets and Evaluation Protocol

For independent evaluation, we use benchmark splits derived from ScanRefer (D. Z. Chen et al., 2020), ReferIt3D (Achlioptas et al., 2020), Multi3DRefer (Zhang et al., 2023), ReferSplat (S. He et al., 2025), and part-level datasets including PartNet, PartNet-Mobility, and ShapeNetPart (Mo et al., 2019; Xiang et al., 2020; Yi et al., 2016). These benchmarks cover object-level, attribute-aware, part-level, relation-dependent, and compositional referring expressions. To ensure fair comparison, we include multi-view 2D projection baselines, 3D referring grounding methods, open-vocabulary 3D segmentation and mapping methods, and 3DGS-native segmentation methods.

Since these benchmarks provide heterogeneous annotations, we convert all supervision and predictions into Gaussian-level masks. For datasets without native 3DGS scenes, we reconstruct a 3DGS representation from posed RGB-D/RGB images and align the original annotation space to the Gaussian coordinate frame using camera poses and rigid registration when necessary. Point-cloud and mesh annotations are transferred to Gaussian primitives by nearest-neighbor or mesh-region assignment based on Gaussian centers. Box annotations are converted by selecting visible Gaussians whose centers fall inside the predicted or ground-truth box. For 2D masks, we render the 3DGS scene from annotated views and accumulate opacity-weighted foreground evidence for each Gaussian, followed by multi-view aggregation and thresholding. The same conversion protocol is applied to baseline predictions, allowing 2D masks, 3D boxes, point-cloud instances, and Gaussian masks to be evaluated under the same Gaussian-level metrics.

We report mIoU and F1-score as the main metrics, and further report Acc@0.25, Acc@0.50, precision, recall, Part-mIoU, Rel-mIoU, target-reference confusion rate, inference time, and memory usage in detailed analyses. Query-type labels are used only for evaluation analysis and are not provided to the model during inference. Custom 3DGS scenes collected by the authors are used only for qualitative visualization and are excluded from training, validation, hyperparameter tuning, and quantitative evaluation.

## 4.3 Comparison with State-of-the-Art Methods

Table 2 and Table 3 reports the quantitative comparison between QAGaussian and 23 representative baselines on multiple independent benchmarks. QAGaussian achieves the best overall performance, with an Avg. mIoU of 47.2 and an Avg. F1-score of 63.2. Compared with ReferSplat (S. He et al., 2025), the strongest 3DGS referring segmentation baseline under our unified protocol, QAGaussian improves Avg. mIoU by 2.7 points and Avg. F1-score by 2.9 points. The gain is more evident on Multi3DRefer, ReferSplat-style evaluation, and part-level evaluation, indicating that QAGaussian is particularly efective for complex referring expressions, spatial relations, and fine-grained parts.

Multi-view 2D projection methods, such as LISA with Fusion (Lai et al., 2024), benefit from strong 2D vision-language foundation models, but their performance is limited by cross-view inconsistency, occlusion, and target-reference ambiguity during 3D fusion. 3D referring grounding methods, such as BUTD-DETR (Jain et al., 2022) and Multi3DRefer (Zhang et al., 2023), perform well on object-centric queries, but their box-level or instance-level outputs are less suitable for Gaussian primitive-level segmentation, especially for local parts and attribute-constrained regions. Open-vocabulary 3D segmentation methods, such as Open3DIS (Nguyen et al., 2024), provide stronger open-set recognition, yet they still mainly rely on text-region similarity and therefore struggle with structured expressions requiring relation or part-whole reasoning. This limitation is consistent with recent visual grounding evidence that decomposing language semantics can reduce target-reference confusion (J. Wu et al., 2025).

Among 3DGS-native methods, CAGS (Sun et al., 2025) and ReferSplat (S. He et al., 2025) are the strongest baselines. ReferSplat performs slightly better than QAGaussian on ScanRefer and ReferIt3D, which are more object-centric. However, QAGaussian achieves clear improvements on Multi3DRefer, ReferSplat-style evaluation, and part-level evaluation. This suggests that strong spatial-language matching is suficient for many simple object-level expressions, while query-conditioned slot generation, relation-aware graph reasoning, and granularity-adaptive mask prediction are more beneficial for multi-target, relation-dependent, compositional, and part-level queries.

Overall, the results show that QAGaussian improves open-vocabulary referring segmentation in 3D Gaussian Splatting not by uniformly increasing all benchmark scores, but by better modeling the structure of complex referring expressions. This supports the motivation of our framework: Gaussian-level referring segmentation requires semantic alignment, query-adaptive candidate organization, relation reasoning, and granularity-aware mask prediction.

Table2:Quantitativecomparisonwithstate-of-the-artmethodsonindependentbenchmarks.Withineachmethodcategory,darkerandlightergreencellsindicatethebestandsecond-best <sub>ults,</sub> <sub>res</sub>p<sup>ective</sup>
<table><tr><td rowspan="2">Method</td><td colspan="2">ScanRefer</td><td colspan="2">ReferIt3D</td><td colspan="2">Multi3DRefer</td><td colspan="2">ReferSplat</td><td colspan="2">Part-level</td><td colspan="2">Avg.</td></tr><tr><td>mIoU</td><td>F1</td><td>mIoU</td><td>F1 mIoU</td><td></td><td></td><td>mIoU</td><td>F1</td><td>mIoU</td><td>F1</td><td>mIoU</td><td>F1</td></tr><tr><td colspan="9">Multi-view 2D projection methods</td><td colspan="2"></td><td></td></tr><tr><td colspan="10">G-DINO + SAM + Fusion (Kirillov et al., 2023; S. Liu et al., 2024) 54.0</td></tr><tr><td>G-SAM + Fusion (T. Ren et al., 2024)</td><td>38.4 39.8</td><td>35.2 36.5</td><td>50.8 52.0</td><td>32.0</td><td>47.0</td><td>25.4</td><td>40.5</td><td>27.0</td><td>40.8</td><td>31.6</td><td>46.6</td></tr><tr><td>CLIPSeg + Fusion (Lüddecke &amp; Ecker, 2022)</td><td>29.5</td><td>55.2 27.4</td><td>40.5</td><td>34.0 25.6</td><td>49.3 38.0</td><td>27.6 21.2</td><td>42.8 34.5</td><td>29.2 21.0</td><td>43.0 33.2</td><td>33.4 24.9</td><td>48.5 38.0</td></tr><tr><td>X-Decoder + Fusion (Zou, Dou, et al., 2023)</td><td>43.8 38.8 54.6</td><td>35.8</td><td>51.3</td><td>33.3</td><td>48.7</td><td>28.0</td><td>43.0</td><td>29.0 31.8</td><td>42.8 47.2</td><td>33.0 35.6</td><td>48.1</td></tr><tr><td colspan="10">41.5 57.0 38.0 53.6 36.2</td><td>50.9</td></tr><tr><td></td><td></td><td>3D referring grounding methods</td><td></td><td></td><td>51.0</td><td>30.4</td><td>46.0</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>60.2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10"></td></tr><tr><td>InstanceRefer (Yuan et al., 2021)</td><td>43.9 45.2 61.3</td><td>39.4 40.5</td><td>55.8 56.5</td><td>34.1 35.0</td><td>49.5 51.1</td><td>32.8 34.3</td><td>48.2 49.8</td><td>25.6 28.5</td><td>39.4 43.8</td><td>35.2 36.7</td><td>50.6 52.5</td></tr><tr><td>3DVG-Trans. (Żhao et al., 2021)</td><td>46.4</td><td>62.1 42.6</td><td>58.6</td><td>37.2</td><td>52.7</td><td>35.5 36.0</td><td>50.3</td><td>27.1</td><td>44.8</td><td>37.8</td><td>53.7</td></tr><tr><td>TransRefer3D (D. He et al., 2021) BUTD-DETR (Jain et al., 2022)</td><td>47.0</td><td>63.0 43.0 44.8</td><td>59.1</td><td>38.4</td><td>54.0 57.1</td><td>37.8</td><td>51.7 52.8</td><td>28.6 27.3</td><td>45.2 45.0</td><td>38.6 40.2</td><td>54.6</td></tr><tr><td>3D-SPS (Luo et al., 2022)</td><td>49.5 48.1</td><td>65.0</td><td>60.6</td><td>41.6</td><td>54.9</td><td>36.9</td><td>51.4</td><td>28.4</td><td>46.0</td><td></td><td>56.1</td></tr><tr><td>Multi3DRefer (Zhang et al., 2023)</td><td>63.8 49.0 64.5</td><td>43.8 44.1</td><td>59.7 60.0</td><td>39.5 44.6</td><td></td><td></td><td>53.5</td><td>29.2</td><td>46.7</td><td>39.3 41.1</td><td>55.2 57.0</td></tr><tr><td colspan="10">Open-vocabulary 3D segmentation and mapping methods</td><td></td></tr><tr><td></td><td>62.4</td><td></td><td></td><td></td><td></td><td>53.0</td><td>30.2</td><td></td><td></td><td></td><td></td></tr><tr><td>OpenScene (K. Liu et al., 2023) ConceptFusion (Jatavallabhula et al., 2023)</td><td>46.1 44.8</td><td>42.0 61.0 40.3</td><td>58.5</td><td>37.4</td><td></td><td></td><td>45.5 43.9</td><td>29.0 27.8</td><td>41.9 41.3</td><td>36.9 35.5</td><td>52.3 50.9</td></tr><tr><td>OpenMask3D (Takmaz et al., 2023)</td><td>48.4</td><td>64.2 44.9</td><td>56.8 60.7</td><td>35.9 40.2</td><td></td><td>51.6 56.2</td><td>28.8 33.1</td><td>48.6 30.8</td><td>47.5</td><td>39.5</td><td>55.4</td></tr><tr><td>Open3DIS (Nguyen et al., 2024)</td><td>50.1</td><td>65.6 46.3</td><td>62.3</td><td>42.2</td><td></td><td>58.5</td><td>34.4</td><td>50.1 33.5</td><td>49.8</td><td></td><td>57.3</td></tr><tr><td colspan="10">3DGS open-vocabulary segmentation methods</td><td>41.3</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SAGA (Cen et al., 2025) LangSplat (Qin et al., 2024)</td><td>43.2 59.6 44.6 61.0</td><td>39.9 40.9</td><td>56.0</td><td>36.6</td><td>52.0 53.2</td><td>31.5 32.8</td><td>45.6 47.2</td><td>30.1 31.6</td><td>45.4 48.8</td><td>36.3 37.5</td><td>51.7</td></tr><tr><td>OpenGaussian (Y. Wu et al., 2024)</td><td>47.5 63.4</td><td>43.6</td><td>57.4 59.7</td><td>37.8 40.8</td><td>56.1</td><td>34.7</td><td>50.2</td><td>34.9</td><td>50.4</td><td>40.3</td><td>53.5 56.0</td></tr><tr><td>OpenSplat3D (Piekenbrinck et al., 2025)</td><td>49.2</td><td>65.1 45.0</td><td>61.1</td><td>42.7</td><td>58.2</td><td>36.2</td><td>52.0</td><td>36.5</td><td>52.6</td><td>41.9</td><td>57.8</td></tr><tr><td>PanoGS (H. Zhai et al., 2025)</td><td>50.8</td><td>66.2 46.7</td><td>62.7</td><td>44.0</td><td>60.1</td><td>37.6</td><td>53.8</td><td>37.4</td><td>53.7</td><td>43.3</td><td>59.3</td></tr><tr><td>CAGS (Sun et al., 2025)</td><td>51.5</td><td>66.8 47.4</td><td>63.5</td><td>45.2</td><td></td><td>61.2</td><td>38.4 54.6</td><td>37.5</td><td>53.6</td><td>44.0</td><td>59.9</td></tr><tr><td colspan="10">3DGS referring segmentation methods</td></tr><tr><td>ReferSplat (S. He et al., 2025)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>QAGaussian</td><td>52.8 52.0</td><td>67.6 66.7</td><td>48.6 64.1 48.0 63.6</td><td>46.9 50.7</td><td>62.0 65.2</td><td>35.8 41.9</td><td>52.6 59.0</td><td>38.6 43.4</td><td>555.</td><td>44.5 47.2</td><td>60.3 63.2</td></tr></table>

![](images/e8acc8c812a5a5ed5c973bf61bf1309d0db321527d457027e7848f07823bf74d.jpg)  
“The control cabinet next to the red box”  
Figure 2: Qualitative comparison with CAGS and ReferSplat on representative 3DGS scenes. Each row shows the inpu scene, ground-truth mask, and segmentation results of diferent methods under the same referring expression. The colored query components are automatically identified by the language parser: red denotes the target object, blue denotes the spatial relation, brown denotes the reference object, and green denotes the attribute. QAGaussian produces cleaner Gaussian-level masks and better suppresses reference objects and distractors.

Table 3: Average metrics on representative independent benchmarks
<table><tr><td>Method</td><td>mIoU</td><td>Acc@0.25</td><td>Acc@0.50</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>G-SAM + Fusion (T. Ren et al., 2024)</td><td>33.4</td><td>53.1</td><td>28.8</td><td>47.6</td><td>49.4</td><td>48.5</td></tr><tr><td>LISA + Fusion (Lai et al., 2024)</td><td>35.6</td><td>55.8</td><td>31.2</td><td>50.3</td><td>51.5</td><td>50.9</td></tr><tr><td>BUTD-DETR (Jain et al., 2022)</td><td>40.2</td><td>62.0</td><td>36.9</td><td>54.8</td><td>57.5</td><td>56.1</td></tr><tr><td>Multi3DRefer (Zhang et al., 2023)</td><td>41.1</td><td>63.4</td><td>37.8</td><td>55.9</td><td>58.1</td><td>57.0</td></tr><tr><td>OpenMask3D (Takmaz et al., 2023)</td><td>39.5</td><td>61.2</td><td>36.0</td><td>54.1</td><td>56.8</td><td>55.4</td></tr><tr><td>Open3DIS (Nguyen et al., 2024)</td><td>41.3</td><td>64.0</td><td>38.2</td><td>56.4</td><td>58.2</td><td>57.3</td></tr><tr><td>OpenGaussian (Y. Wu et al., 2024)</td><td>40.3</td><td>62.5</td><td>37.0</td><td>54.7</td><td>57.4</td><td>56.0</td></tr><tr><td>CAGS (Sun et al., 2025)</td><td>44.0</td><td>67.5</td><td>40.9</td><td>58.6</td><td>61.3</td><td>59.9</td></tr><tr><td>ReferSplat (S. He et al., 2025)</td><td>44.5</td><td>68.2</td><td>41.6</td><td>59.0</td><td>61.7</td><td>60.3</td></tr><tr><td>QAGaussian</td><td>47.2</td><td>72.0</td><td>45.7</td><td>62.4</td><td>64.1</td><td>63.2</td></tr></table>

Table 4: Fine-grained diagnostic metrics on part-level queries from PartNet, PartNet-Mobility, and ShapeNetPart, and relation-dependent queries from ScanRefer, ReferIt3D, Multi3DRefer, and ReferSplat-derived splits. TR-conf. denotes the target-reference confusion rate.
<table><tr><td>Method</td><td>Part-mIoU</td><td>Rel-mIoU</td><td>TR-conf. ↓</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>G-SAM + Fusion (T. Ren et al., 2024)</td><td>29.2</td><td>35.0</td><td>17.2</td><td>44.6</td><td>41.5</td><td>43.0</td></tr><tr><td>LISA + Fusion (Lai et al., 2024)</td><td>31.8</td><td>33.4</td><td>18.9</td><td>48.6</td><td>45.9</td><td>47.2</td></tr><tr><td>BUTD-DETR (Jain et al., 2022)</td><td>27.3</td><td>40.6</td><td>15.9</td><td>43.8</td><td>46.2</td><td>45.0</td></tr><tr><td>Multi3DRefer (Zhang et al., 2023)</td><td>29.2</td><td>41.8</td><td>15.6</td><td>45.4</td><td>48.0</td><td>46.7</td></tr><tr><td>Open3DIS (Nguyen et al., 2024)</td><td>33.5</td><td>39.6</td><td>14.8</td><td>49.2</td><td>50.4</td><td>49.8</td></tr><tr><td>LangSplat (Qin et al., 2024)</td><td>31.6</td><td>37.8</td><td>15.5</td><td>48.0</td><td>49.6</td><td>48.8</td></tr><tr><td>OpenGaussian (Y. Wu et al., 2024)</td><td>34.9</td><td>41.5</td><td>13.9</td><td>49.5</td><td>51.3</td><td>50.4</td></tr><tr><td>CAGS (Sun et al., 2025)</td><td>37.5</td><td>43.2</td><td>12.1</td><td>52.8</td><td>54.5</td><td>53.6</td></tr><tr><td>ReferSplat (S. He et al., 2025)</td><td>38.6</td><td>44.4</td><td>10.8</td><td>54.8</td><td>56.0</td><td>55.4</td></tr><tr><td>QAGaussian</td><td>43.4</td><td>50.8</td><td>7.4</td><td>60.2</td><td>62.8</td><td>61.5</td></tr></table>

## 4.4 Qualitative Comparison with Existing Methods

Fig. 2 shows qualitative comparisons among CAGS, ReferSplat, and QAGaussian on diferent 3DGS scenes. Each example includes the input scene, ground-truth mask, and segmentation results produced by diferent methods. The colored words in each query denote the target object, spatial relation, reference object, and attribute, which makes the required language structure explicit.

As shown in the figure, CAGS can often find regions that are semantically related to the query, but it tends to produce coarse or incorrect masks. For example, in the query “The water cup in front of the brown bear”, CAGS incorrectly segments a large part of the reference object. In the query “The wooden stool near the wall”, it also includes nearby furniture that does not belong to the target. These results indicate that CAGS has limited ability to distinguish the referred target from reference objects and surrounding distractors.

ReferSplat generally performs better than CAGS and is more suitable for 3DGS scenes. However, it is still afected by similar or nearby objects in complex queries. In the water-cup example, ReferSplat keeps another glass-like object together with the target. In the duck example, it introduces irrelevant objects around the target. In the outdoor control-room example, it also fails to fully suppress the other similar structure. These cases show that direct language-to-Gaussian matching is not suficient for robust referring segmentation under relation-dependent expressions.

In contrast, QAGaussian produces masks that are closer to the ground truth across these examples. For relation queries, chained spatial queries, and attribute-relation queries, QAGaussian more accurately separates the target from reference objects and distractors. This demonstrates that query-conditioned multi-scale Gaussian slot generation and relation-aware reasoning are efective for open-vocabulary referring segmentation in 3D Gaussian Splatting.

## 4.5 Visualization under Diferent Query Types

Table 4 and Fig. 3 further analyze the performance under diferent types of referring expressions. Diferent from the cross-scene qualitative comparison in Sec. 4.4, this section uses multiple queries within the same complex kitchen 3DGS scene. This setting removes the influence of scene variation and directly shows whether the model can adapt its segmentation output according to the structure and granularity of the input expression. The query-type labels are used only for analysis and are not provided to the model during inference.

As shown in Table 4, QAGaussian achieves more stable results on fine-grained and relationdependent expressions. Compared with ReferSplat, QAGaussian improves Part-mIoU from 38.6 to 43.4 and Rel-mIoU from 44.4 to 50.8, while reducing the target-reference confusion rate from 10.8 to 7.4. This indicates that QAGaussian improves part-level segmentation and reduces confusion between the referred target and reference objects.

![](images/8d0d3713f03438445c7d871b4aa17938c11ae6448774d40a31a6a143b4173ea1.jpg)  
Figure 3: Visualization under diferent query types within the same complex kitchen 3DGS scene. The input scene is fixed, while the referring expression changes across object-level, attribute-aware, part-level, relation-dependent, and compositional queries. Each row shows Gaussian-level segmentation results from three views. The colored query components are automatically identified by the language parser: red denotes the target object, blue denotes the spatial relation, brown denotes the reference object, and green denotes the attribute.

Fig. 3 shows qualitative results for five typical query types in the same scene. For the object-level query “The kettle”, the model segments the whole kettle consistently across diferent views. For the attribute-aware query “The blue bottle”, the model uses the color attribute to select the correct object from multiple container-like objects. For the part-level query “The handle of the kettle”, the output focuses on the handle instead of the entire kettle, showing that the model can adjust the segmentation granularity according to the query.

For the relation-dependent query “The toaster next to the kettle”, the model needs to identify both the target and the reference object and use their spatial relation for disambiguation. The result shows that QAGaussian can localize the toaster next to the kettle instead of selecting the kettle itself or other nearby objects. For the compositional query “The blue bottle beside the sink under the shelf”, the model needs to combine attribute, target, and multiple spatial constraints. Although this query is more challenging and may still be afected by nearby objects in some views, the result remains mainly focused on the target blue bottle.

Overall, Table 4 and Fig. 3 show that QAGaussian can produce reasonable Gaussian-level masks for diferent query types. The results are especially clear for part-level, relation-dependent, and compositional expressions, where the model needs to adapt both target selection and segmentation granularity. This supports the efectiveness of query-conditioned multi-scale Gaussian slot generation, relation-aware reasoning, and adaptive routing.

Table 5: Efect of pretraining data on cross-dataset generalization. All settings are evaluated on independent benchmarks without target benchmark fine-tuning.
<table><tr><td>Pretraining setting</td><td>Avg. mIoU</td><td>Acc@0.50</td><td>Avg. F1</td></tr><tr><td>2D pseudo masks</td><td>41.2</td><td>37.4</td><td>56.3</td></tr><tr><td>Mosaic3D-1.4M</td><td>43.6</td><td>40.1</td><td>58.9</td></tr><tr><td>Mosaic3D-2.8M</td><td>45.4</td><td>42.8</td><td>61.0</td></tr><tr><td>Mosaic3D-5.6M</td><td>47.2</td><td>45.7</td><td>63.2</td></tr></table>

![](images/b8f357c3f5f987c9a6504f2a975036ebf11f6a96bd24256ea118f3182cacb48b.jpg)  
Figure 4: Performance retention under diferent query perturbations.

## 4.6 Cross-Dataset and Language Robustness Analysis

We further analyze cross-dataset generalization and language robustness. QAGaussian is pretrained only on Mosaic3D-5.6M and is evaluated on independent benchmarks. Therefore, the results reflect whether the learned Gaussian-language alignment can transfer to unseen scenes and annotations.

Table 5 reports the efect of diferent pretraining settings. Without pretraining, the model obtains an Avg. mIoU of 39.6. Using 2D pseudo masks improves the result to 41.2, but the gain is limited because the supervision is not directly aligned with Gaussian primitives. Increasing the Mosaic3D pretraining scale consistently improves cross-dataset performance, and the full Mosaic3D-5.6M setting reaches 47.2 Avg. mIoU without any target benchmark fine-tuning.

Fig. 4 evaluates robustness to language perturbations. We construct perturbed queries by synonym replacement, relation paraphrase, attribute insertion, compositional rewrite, and distractor-rich expression rewriting while keeping the referred target unchanged. The original query is used as the reference point for computing performance retention.

As shown in Fig. 4, QAGaussian retains a larger proportion of its original performance than ReferSplat under all perturbation types. The advantage is especially clear for compositional rewrites and distractor-rich expressions, where direct language-to-Gaussian matching is more likely to lose the target-reference structure. These results suggest that the proposed query-conditioned slot generation and relation-aware reasoning improve both cross-dataset transfer and language robustness.

Table 6: Ablation study of key QAGaussian components on the independent evaluation benchmarks. TR-conf. denotes the target-reference confusion rate, where lower values are better.
<table><tr><td>Variant</td><td>mIoU</td><td>Acc@0.25</td><td>Acc@0.50</td><td>Part-mIoU</td><td>Rel-mIoU</td><td>TR-conf. ↓</td><td>F1</td></tr><tr><td>Full QAGaussian</td><td>47.2</td><td>72.0</td><td>45.7</td><td>43.4</td><td>50.8</td><td>7.4</td><td>63.2</td></tr><tr><td>w/o query-cond. slots</td><td>43.8</td><td>67.4</td><td>40.9</td><td>39.0</td><td>45.7</td><td>11.9</td><td>59.5</td></tr><tr><td>w/o multi-scale slots</td><td>44.6</td><td>68.6</td><td>42.1</td><td>40.1</td><td>46.8</td><td>10.7</td><td>60.4</td></tr><tr><td>w/o relation graph</td><td>44.9</td><td>69.0</td><td>42.4</td><td>41.0</td><td>45.9</td><td>12.6</td><td>60.8</td></tr><tr><td>w/o adaptive router</td><td>45.1</td><td>69.3</td><td>42.7</td><td>40.7</td><td>47.2</td><td>10.9</td><td>61.0</td></tr><tr><td>w/o refinement</td><td>45.8</td><td>70.2</td><td>43.6</td><td>41.6</td><td>48.4</td><td>9.6</td><td>61.7</td></tr><tr><td>w/o activeness filtering</td><td>46.0</td><td>70.5</td><td>43.8</td><td>42.0</td><td>48.9</td><td>9.1</td><td>61.9</td></tr></table>

## 4.7 Ablation Study

We conduct ablation studies on the independent evaluation benchmarks defined in Sec. 4.2, without target benchmark fine-tuning, to evaluate the contribution of each key component in QAGaussian. All variants are trained and evaluated under the same setting as the full model, and only the specified component is removed or replaced. The results are reported in Table 6.

As shown in Table 6, the full QAGaussian model achieves the best performance across all metrics. Removing query-conditioned Gaussian slots causes the largest overall degradation, reducing mIoU from 47.2 to 43.8 and F1 from 63.2 to 59.5. This confirms that query-conditioned slot generation is essential for organizing Gaussian primitives into compact and query-relevant soft candidates. Without this component, the model relies more on primitive-level matching and becomes less stable under open-vocabulary referring expressions.

The multi-scale slot design mainly afects fine-grained segmentation. When it is removed, PartmIoU drops from 43.4 to 40.1, showing that diferent slot scales are useful for handling part-level and small-region targets. Removing the relation graph leads to the largest drop in Rel-mIoU and the highest TR-conf. value. This indicates that relation-aware reasoning is important for distinguishing the referred target from reference objects and nearby distractors.

The adaptive router and refinement module also provide consistent gains. Without the router, the model cannot dynamically balance object-level, part-level, attribute-aware, and relation-aware branches, leading to weaker performance on both part-level and relation-dependent expressions. Without relation-constrained refinement, the masks become less consistent with spatial, attribute, and geometric cues. Finally, removing activeness filtering causes a smaller but consistent decline, suggesting that suppressing inactive or noisy slots helps stabilize mask prediction. These results demonstrate that the proposed components are complementary and jointly contribute to robust open-vocabulary referring segmentation in 3DGS.

Fig. 5 further provides a qualitative ablation example on the query “The right glass door beneath the black roof”. The full model accurately localizes the referred glass door and suppresses surrounding windows, facade regions, and roof structures. This indicates that QAGaussian can combine the target description, positional cue, and reference object to produce a compact Gaussian-level mask.

When query-conditioned slots are removed, the predicted mask expands to nearby facade and roof-adjacent regions, suggesting that the model fails to form compact candidates conditioned on the current expression. Without relation graph reasoning, the prediction is split into several visually similar glass regions, which shows that relation-aware slot interactions are important for spatial disambiguation. Without the adaptive router, the output becomes coarser and includes more surrounding building surfaces, indicating that dynamic branch selection helps match the required segmentation granularity. This qualitative evidence is consistent with the quantitative trends in Table 6.

![](images/7df288dcaea262d9c2a8f26e067e28000c7c9b104721499a4fadfba83cae7612.jpg)  
Figure 5: Qualitative ablation example on a relation-dependent region query. The query components are automatically extracted by the language parser and visualized with diferent colors. The full QAGaussian model accurately segments the referred glass door, while removing query-conditioned slots, relation graph reasoning, or adaptive routing leads to coarse masks, activation of similar glass regions, or leakage to surrounding facade areas.

## 4.8 Eficiency Analysis

We further analyze the computational behavior of QAGaussian. Although the proposed framework introduces query-conditioned slot generation, relation-aware graph reasoning, and relation-constrained refinement, these operations are performed on a compact set of Gaussian slots rather than all Gaussian primitives. Therefore, the additional cost mainly comes from slot construction, graph message passing, and a small number of refinement iterations.

Table 7 reports the runtime and memory usage on Gaussian scenes of diferent scales. OpenGaussian is the fastest method because it mainly relies on direct open-vocabulary similarity matching, and CAGS also keeps relatively low computational cost. ReferSplat introduces spatially aware referring segmentation and therefore requires slightly more computation. QAGaussian is slower than these baselines, but the gap is moderate: on medium-scale scenes, the query time increases from 0.66s for ReferSplat to 0.82s for QAGaussian, while Avg. mIoU improves from 44.5 to 47.2.

The additional cost is consistent with the design of QAGaussian. Query-conditioned multi-scale Gaussian slot generation avoids exhaustive primitive-level reasoning, but it still needs to build soft slot assignments for each expression. The relation-aware graph encoder then updates slot representations using target-reference and spatial cues, and the refinement module improves mask consistency through several lightweight iterations. These steps increase runtime and memory slightly, but they lead to clearer gains on relation-dependent, compositional, and part-level expressions, as shown in Tables 4 and 6.

Overall, QAGaussian provides a favorable accuracy-eficiency trade-of. It is not designed to be the fastest similarity-matching method; instead, it adds a compact reasoning layer over Gaussian slots to improve open-vocabulary referring segmentation. The eficiency results show that this structured reasoning brings a 2.7-point Avg. mIoU improvement over ReferSplat with acceptable additional query time and memory usage.

Table 7: Eficiency comparison on Gaussian scenes of diferent scales. Preprocess and query time are reported for small, medium, and large scenes.
<table><tr><td>Method</td><td colspan="3">Preprocess time</td><td colspan="3">Query time</td><td colspan="3">Memory</td><td>Avg. mIoU</td></tr><tr><td></td><td>Small</td><td>Medium</td><td>Large</td><td>Small</td><td>Medium</td><td>Large</td><td>Small</td><td>Medium</td><td>Large</td><td></td></tr><tr><td>OpenGaussian (Y. Wu et al., 2024)</td><td>6.2m</td><td>8.6m</td><td>13.8m</td><td>0.28s</td><td>0.43s</td><td>0.68s</td><td>5.8GB</td><td>7.8GB</td><td>10.6GB</td><td>40.3</td></tr><tr><td>CAGS (Sun et al., 2025)</td><td>6.8m</td><td>9.4m</td><td>15.1m</td><td>0.36s</td><td>0.61s</td><td>0.92s</td><td>6.3GB</td><td>8.6GB</td><td>11.8GB</td><td>44.0</td></tr><tr><td>ReferSplat (S. He et al., 2025)</td><td>7.1m</td><td>9.8m</td><td>15.8m</td><td>0.40s</td><td>0.66s</td><td>1.05s</td><td>6.8GB</td><td>9.1GB</td><td>12.4GB</td><td>44.5</td></tr><tr><td>QAGaussian</td><td>7.4m</td><td>10.3m</td><td>16.6m</td><td>0.48s</td><td>0.82s</td><td>1.28s</td><td>7.5GB</td><td>9.8GB</td><td>13.6GB</td><td>47.2</td></tr></table>

## 5. Discussion

Efectiveness of query-conditioned Gaussian slots. The main improvement of QAGaussian comes from organizing Gaussian primitives in a query-conditioned manner rather than relying on a fixed candidate pool or direct text-region matching. In open-vocabulary referring segmentation, the same 3DGS scene may require diferent segmentation granularity under diferent expressions. For example, a query may refer to a whole object, a local part, an attribute-constrained object, or a target determined by its relation to another object. Fixed candidates are often insuficient for such flexible language structures, especially when spatial perception and contextual 3D descriptions are required (Yan et al., 2025). By generating multi-scale Gaussian slots conditioned on the input expression, QAGaussian can form compact soft candidates that are more relevant to the current query. This design explains why the model performs better on part-level, relation-dependent, and compositional expressions, and why removing query-conditioned slots leads to the largest degradation in the ablation study.

Role of relation-aware reasoning. A central challenge in 3D referring segmentation is not only recognizing the target category, but also distinguishing the target from reference objects and nearby distractors. Many expressions contain explicit spatial relations, such as “next to”, “beneath”, or “in front of”, where the reference object should guide target localization but should not be included in the final mask. Direct language-to-Gaussian similarity matching tends to confuse the referred target with the reference object or other semantically similar regions. The relation-aware slot graph in QAGaussian provides a compact structure for modeling interactions among target, reference, attribute, part, and contextual slots. This helps the model suppress reference leakage and reduce target-reference confusion, especially in relation-dependent queries. The diagnostic metrics and qualitative ablation results show that relation reasoning is particularly important for improving Rel-mIoU and reducing TR-conf.

Generalization without target benchmark fine-tuning. QAGaussian is pretrained only on Mosaic3D-5.6M for Gaussian-language semantic alignment and is evaluated on independent benchmarks. This setting is important because open-vocabulary 3D scene understanding should not depend on adapting the model to every downstream dataset. The pretraining analysis shows that larger-scale Mosaic3D pretraining improves cross-dataset performance, suggesting that Gaussianlevel language alignment can transfer across diferent scene types, annotation formats, and query distributions. Compared with using frozen 2D vision-language features or 2D pseudo masks alone, Mosaic3D pretraining provides supervision that is more directly aligned with Gaussian primitives. This supports the use of large-scale Gaussian-language pretraining as a practical foundation for open-vocabulary 3DGS segmentation.

Eficiency and practical trade-of. QAGaussian is not the fastest method among the compared approaches, since it introduces query-conditioned slot generation, relation-aware graph reasoning, adaptive routing, and relation-constrained refinement. However, these operations are performed over a compact set of slots rather than all Gaussian primitives, which keeps the additional cost moderate. The eficiency analysis shows that QAGaussian requires slightly higher query time and memory than simpler similarity-matching methods, but provides a clear improvement in segmentation accuracy, especially for complex referring expressions. This trade-of is reasonable for applications such as embodied interaction, robotic manipulation, and augmented reality, where correctly identifying the referred 3D target is often more important than minimizing a small amount of additional inference time.

Limitations and future work. Despite its efectiveness, QAGaussian still has several limitations. First, the method depends on the quality of the reconstructed 3DGS scene. If the Gaussian representation is noisy, incomplete, or inaccurate due to occlusion, transparent objects, reflective surfaces, or insuficient views, the predicted mask may also be degraded. Second, very small parts and thin structures remain challenging, because they require both high-quality geometry and fine-grained semantic alignment. Third, although the relation-aware slot graph improves spatial reasoning, highly complex multi-hop expressions, ambiguous references, and relations such as “between”, “inside”, or “aligned with” may still be dificult. Future work may explore uncertaintyaware slot selection, stronger 3D scene graph priors, interactive correction, and extension to dynamic or 4D Gaussian scenes. These directions could further improve the robustness and applicability of open-vocabulary referring segmentation in 3D Gaussian Splatting.

## 6. Conclusion

This paper presented QAGaussian, a query-adaptive framework for open-vocabulary referring segmentation in 3D Gaussian Splatting. Unlike existing methods that mainly rely on direct text-region similarity matching, QAGaussian models the internal structure of free-form referring expressions and performs Gaussian-level segmentation through query-conditioned multi-scale slot generation, relation-aware slot graph reasoning, adaptive granularity routing, and relation-constrained mask refinement. By organizing Gaussian primitives into diferentiable query-conditioned slots, the proposed framework can adapt its target selection and segmentation granularity to object-level, attribute-aware, part-level, relation-dependent, and compositional expressions.

QAGaussian is pretrained on Mosaic3D-5.6M for Gaussian-language semantic alignment and evaluated on independent benchmarks. Extensive experiments show that QAGaussian achieves stateof-the-art overall performance and provides clear gains on complex referring expressions, part-level segmentation, relation-dependent queries, language robustness, and cross-dataset generalization. Ablation studies further verify the complementary contributions of query-conditioned slots, multiscale modeling, relation-aware reasoning, adaptive routing, and refinement. Although the method introduces moderate additional computation compared with simpler similarity-matching baselines, it ofers a favorable accuracy-eficiency trade-of for language-guided 3DGS scene understanding. Future work will explore stronger multi-hop relation modeling, uncertainty-aware slot selection, interactive correction, and extension to dynamic or 4D Gaussian scenes.

## Acknowledgments

This research was supported by the Henan Province Key Research and Development Program (Grant No. 251111210800, Grant No. 261111210600).

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data Availability

Data will be made available on request.

## References

Achlioptas, P., Abdelreheem, A., Xia, F., Elhoseiny, M., & Guibas, L. (2020). Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes. European Conference on Computer Vision, 422–440.

Armeni, I., He, Z.-Y., Gwak, J., Zamir, A. R., Fischer, M., Malik, J., & Savarese, S. (2019). 3d scene graph: A structure for unified semantics, 3d space, and camera. Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 5664–5673.

Cen, J., Fang, J., Yang, C., Xie, L., Zhang, X., Shen, W., & Tian, Q. (2025). Segment any 3d gaussians. AAAI Conference on Artificial Intelligence.

Chen, D. Z., Chang, A. X., & Nießner, M. (2020). Scanrefer: 3d object localization in rgb-d scans using natural language. European Conference on Computer Vision, 202–221.

Chen, D. Z., Wu, Q., Nießner, M., & Chang, A. X. (2022). D3Net: A speaker-listener architecture for semi-supervised dense captioning and visual grounding in rgb-d scans. European Conference on Computer Vision.

Chen, L., Wu, J., Lei, Z., Liu, J., Lin, S., Wang, C., & He, G. (2024). Clip-driven open-vocabulary 3d scene graph generation via cross-modality contrastive learning. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 29250–29260.

Chen, Y., Chen, Z., Zhang, C., Wang, F., Yang, X., Wang, Y., Cai, Z., Yang, L., Liu, H., & Lin, G. (2024). Gaussianeditor: Swift and controllable 3d editing with gaussian splatting. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21476–21485.

Ding, J., Xue, N., Xia, G.-S., & Dai, D. (2022). Decoupling zero-shot semantic segmentation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11583–11592.

Ghiasi, G., Gu, X., Cui, Y., & Lin, T.-Y. (2022). Scaling open-vocabulary image segmentation with image-level labels. European Conference on Computer Vision, 540–557.

Gu, Q., Kuwajerwala, A., Morin, S., Jatavallabhula, K. M., Sen, B., Agarwal, A., Rivera, C., Paul, W., Ellis, K., Chellappa, R., Gan, C., de Melo, C. M., Tenenbaum, J. B., Torralba, A., Shkurti, F., & Paull, L. (2023). Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning. arXiv preprint arXiv:2309.16650.

Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. Advances in Neural Information Processing Systems, 1024–1034.

He, D., Zhao, Y., Luo, J., Hui, T., Huang, S., Zhang, A., & Liu, S. (2021). TransRefer3D: Entityand-relation aware transformer for fine-grained 3d visual grounding. Proceedings of the ACM International Conference on Multimedia, 2344–2352.

He, S., Jie, G., Wang, C., Zhou, Y., Hu, S., Li, G., & Ding, H. (2025). Refersplat: Referring segmentation in 3d gaussian splatting. International Conference on Machine Learning.

Hu, T., Zhang, J., Rao, Y., Zeng, D., Yu, H., & Huang, X. (2025). 3DBench: A scalable benchmark for object and scene-level instruction-tuning of 3D large language models. Neural Networks, 189, 107566.

Huang, S., Chen, Y., Jia, J., & Wang, L. (2022). MVT: Multi-view transformer for 3d visual grounding. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. Neural Computation, 3(1), 79–87.

Jain, A., Gkanatsios, N., Mediratta, I., & Fragkiadaki, K. (2022). Bottom up top down detection transformers for language grounding in images and point clouds. European Conference on Computer Vision, 417–433.

Jatavallabhula, K. M., Kuwajerwala, A., Gu, Q., Omama, M., Chen, T., Maalouf, A., Li, S., Iyer, G., Saryazdi, S., Keetha, N., Tewari, A., Tenenbaum, J. B., de Melo, C., Krishna, M., Paull, L., Shkurti, F., & Torralba, A. (2023). Conceptfusion: Open-set multimodal 3d mapping. Robotics: Science and Systems.

Jia, C., Yang, Y., Xia, Y., Chen, Y.-T., Parekh, Z., Pham, H., Le, Q. V., Sung, Y.-H., Li, Z., & Duerig, T. (2021). Scaling up visual and vision-language representation learning with noisy text supervision. International Conference on Machine Learning, 4904–4916.

Kerbl, B., Kopanas, G., Leimkuehler, T., & Drettakis, G. (2023). 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4), 139:1–139:14.

Kerr, J., Kim, C. M., Goldberg, K., Kanazawa, A., & Tancik, M. (2023). LERF: Language embedded radiance fields. Proceedings of the IEEE/CVF International Conference on Computer Vision, 19729–19739.

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. International Conference on Learning Representations.

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., & Girshick, R. (2023). Segment anything. Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 4015–4026.

Kobayashi, S., Matsumoto, E., & Sitzmann, V. (2022). Decomposing NeRF for editing via feature field distillation. Advances in Neural Information Processing Systems, 35, 23311–23330.

Koch, S., Vaskevicius, N., Colosi, M., Hermosilla, P., & Ropinski, T. (2024). Open3dsg: Openvocabulary 3d scene graphs from point clouds with queryable objects and open-set relationships. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14183–14193.

Lai, X., Tian, Z., Chen, Y., Li, Y., Yuan, Y., Liu, S., & Jia, J. (2024). LISA: Reasoning segmentation via large language model. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Lee, J., Park, C., Choe, J., Wang, Y.-C. F., Kautz, J., Cho, M., & Choy, C. (2025). Mosaic3d: Foundation dataset and model for open-vocabulary 3d segmentation. arXiv preprint arXiv:2502.02548.

Li, B., Weinberger, K. Q., Belongie, S., Koltun, V., & Ranftl, R. (2022). Language-driven semantic segmentation. International Conference on Learning Representations.

Li, F., Zhang, H., Sun, P., Zou, X., Liu, S., Yang, J., Li, C., Zhang, L., & Gao, J. (2024). Semantic-sam: Segment and recognize anything at any granularity. European Conference on Computer Vision, 194–212.

Li, J., Mo, W., Song, F., Sun, C., Qiang, W., Su, B., & Zheng, C. (2025). Supporting vision-language model few-shot inference with confounder-pruned knowledge prompt. Neural Networks, 185, 107173.

Li, Y., Ma, Q., Yang, R., Li, H., Ma, M., Ren, B., Popovic, N., Sebe, N., Konukoglu, E., Gevers, T., Van Gool, L., Oswald, M. R., & Paudel, D. P. (2025). Scenesplat: Gaussian splatting-based scene understanding with vision-language pretraining. arXiv preprint arXiv:2503.18052.

Liang, F., Wu, B., Dai, X., Li, K., Zhao, Y., Zhang, H., Zhang, P., Vajda, P., & Marculescu, D. (2023). Open-vocabulary semantic segmentation with mask-adapted CLIP. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7061–7070.

Liu, K., Zhan, F., Chen, Y., Zhang, J., Yu, Y., El Saddik, A., Lu, S., & Xing, E. P. (2023). Openscene: 3d scene understanding with open vocabularies. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 815–824.

Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H., Zhu, J., & Zhang, L. (2024). Grounding dino: Marrying dino with grounded pre-training for open-set object detection. European Conference on Computer Vision, 38–55.

Liu, Z., Wang, L., Hu, Y., & Yin, B. (2025). Memory transmission based referring video object segmentation. Neural Networks, 189, 107548.

Locatello, F., Weissenborn, D., Unterthiner, T., Mahendran, A., Heigold, G., Uszkoreit, J., Dosovitskiy, A., & Kipf, T. (2020). Object-centric learning with slot attention. Advances in neural information processing systems, 33, 11525–11538.

Lu, Y., Zhou, Y., Qiao, Y., Song, C., Liang, T., Ma, J., Wang, H., & Yin, Y. (2025). Segment then splat: Unified 3d open-vocabulary segmentation via gaussian splatting. arXiv preprint arXiv:2503.22204.

Lüddecke, T., & Ecker, A. (2022). CLIPSeg: Image segmentation using text and image prompts. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7086–7096.

Luo, J., Fu, J., Kong, X., Gao, C., Ren, H., Shen, H., Xia, H., & Liu, S. (2022). 3d-sps: Single-stage 3d visual grounding via referred point progressive selection. Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 16454–16463.

Ma, Q., Ren, B., Sebe, N., Konukoglu, E., Li, Y., Gevers, T., Van Gool, L., & Paudel, D. P. (2025). Shapesplat: A large-scale dataset of gaussian splats and their self-supervised pretraining. arXiv preprint arXiv:2408.10906.

Mo, K., Zhu, S., Chang, A. X., Yi, L., Tripathi, S., Guibas, L. J., & Su, H. (2019). Partnet: A large-scale benchmark for fine-grained and hierarchical part-level 3d object understanding. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 909–918.

Nguyen, P. D. A., Ngo, T. D., Gan, C., Kalogerakis, E., & Tran, A. (2024). Open3dis: Open-vocabulary 3d instance segmentation with 2d mask guidance. arXiv preprint arXiv:2312.10671.

Piekenbrinck, J., Schmidt, C., Hermans, A., Vaskevicius, N., Linder, T., & Leibe, B. (2025). Opensplat3d: Open-vocabulary 3d instance segmentation using gaussian splatting. arXiv preprint arXiv:2506.07697.

Qin, M., Li, W., Zhou, J., Wang, H., & Pfister, H. (2024). Langsplat: 3d language gaussian splatting. arXiv preprint arXiv:2312.16084.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. International Conference on Machine Learning, 8748–8763.

Rao, Y., Zhao, W., Chen, G., Tang, Y., Zhu, Z., Huang, G., Zhou, J., & Lu, J. (2022). DenseCLIP: Language-guided dense prediction with context-aware prompting. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18082–18091.

Ren, T., Liu, S., Zeng, A., Lin, J., Li, K., Cao, H., Chen, J., Huang, X., Chen, Y., Yan, F., Zeng, Z., Zhang, H., Li, F., Yang, J., Li, H., Jiang, Q., Yang, L., Liu, Z., & Zhang, L. (2024). Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159.

Ren, Z., Huang, Z., Wei, Y., Zhao, Y., Fu, D., Feng, J., & Jin, X. (2024). GLaMM: Pixel grounding large multimodal model. arXiv preprint arXiv:2311.03356.

Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic routing between capsules. Advances in Neural Information Processing Systems, 3856–3866.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. International Conference on Learning Representations.

Shi, J.-C., Wang, M., Duan, H.-B., & Guan, S.-H. (2023). Language embedded 3d gaussians for open-vocabulary scene understanding. arXiv preprint arXiv:2311.18482.

Sun, W., Zhou, Y., Jiao, J., & Li, Y. (2025). Cags: Open-vocabulary 3d scene understanding with context-aware gaussian splatting. arXiv preprint arXiv:2504.11893.

Takmaz, A., Fedele, E., Sumner, R. W., Pollefeys, M., Tombari, F., & Engelmann, F. (2023). Openmask3d: Open-vocabulary 3d instance segmentation. Advances in Neural Information Processing Systems.

Tschannen, M., Gritsenko, A., Wang, X., Naeem, M. F., Alabdulmohsin, I., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., Hénaf, O., Harmsen, J., Steiner, A., & Zhai, X. (2025).

SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786.

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. International Conference on Learning Representations.

Wald, J., Dhamo, H., Navab, N., & Tombari, F. (2020). Learning 3d semantic scene graphs from 3d indoor reconstructions. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3961–3970.

Wang, J., Liao, D., Zhang, Y., Xu, D., & Zhang, X. (2024). Layerwised multimodal knowledge distillation for vision-language pretrained model. Neural Networks, 175, 106272.

Werby, A., Huang, C., Batra, D., & Martín-Martín, R. (2024). Hierarchical open-vocabulary 3d scene graphs for language-grounded robot navigation. arXiv preprint arXiv:2403.17846.

Wu, J., Wu, C., Wei, Y., Xu, Q., & Gong, F. (2025). Learning contrastive semantic decomposition for visual grounding. Neural Networks, 190, 107593.

Wu, S.-C., Wald, J., Tateno, K., Navab, N., & Tombari, F. (2021). Scenegraphfusion: Incremental 3d scene graph prediction from rgb-d sequences. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7515–7525.

Wu, Y., Meng, J., Li, H., Wu, C., Shi, Y., Cheng, X., Zhao, C., Feng, H., Ding, E., Wang, J., Zeng, J., & Zhang, X. (2024). Opengaussian: Towards point-level 3d gaussian-based open vocabulary understanding. arXiv preprint arXiv:2406.02058.

Xiang, F., Qin, Y., Mo, K., Xia, Y., Zhu, H., Liu, F., Liu, M., Jiang, H., Yuan, Y., Wang, H., Yi, L., Chang, A. X., Guibas, L. J., & Su, H. (2020). PartNet-Mobility: A large-scale benchmark for articulated object understanding. ACM SIGGRAPH Asia Conference Papers, 1–16.

Xu, J., De Mello, S., Liu, S., Byeon, W., Breuel, T., Kautz, J., & Wang, X. (2022). GroupViT: Semantic segmentation emerges from text supervision. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18134–18144.

Xu, J., Liu, S., Vahdat, A., Byeon, W., Wang, X., & De Mello, S. (2023). Open-vocabulary panoptic segmentation with text-to-image difusion models. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2955–2966.

Xu, M., Zhang, Z., Wei, F., Lin, Y., Cao, Y., Hu, H., & Bai, X. (2023). Side adapter network for open-vocabulary semantic segmentation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2945–2954.

Yan, J., Xie, Y., Zou, S., Wei, Y., & Luan, X. (2025). Enhancing spatial perception and contextual understanding for 3D dense captioning. Neural Networks, 185, 107252.

Ye, M., Danelljan, M., Yu, F., & Ke, L. (2024a). Gaussian grouping: Segment and edit anything in 3d scenes. arXiv preprint arXiv:2312.00732.

Ye, M., Danelljan, M., Yu, F., & Ke, L. (2024b). Gaussiancut: Interactive segmentation via graph cut for 3d gaussian splatting. Advances in Neural Information Processing Systems.

Yi, L., Kim, V. G., Ceylan, D., Shen, I.-C., Yan, M., Su, H., Lu, C., Huang, Q., Shefer, A., & Guibas, L. (2016). A scalable active framework for region annotation in 3d shape collections. ACM Transactions on Graphics, 35(6), 1–12.

Yuan, Z., Yan, X., Liao, Y., Guo, Y., Li, G., Cui, S., & Li, Z. (2021). Instancerefer: Cooperative holistic understanding for visual grounding on point clouds through instance multi-level contextual referring. Proceedings of the IEEE/CVF International Conference on Computer Vision, 1791–1800.

Zhai, H., Li, H., Li, Z., Pan, X., He, Y., & Zhang, G. (2025). Panogs: Gaussian-based panoptic segmentation for 3d open vocabulary scene understanding. arXiv preprint arXiv:2503.18107.

Zhai, X., Mustafa, B., Kolesnikov, A., & Beyer, L. (2023). Sigmoid loss for language image pre-training. Proceedings of the IEEE/CVF International Conference on Computer Vision, 11975–11986.

Zhang, Y., Gong, Z., & Chang, A. X. (2023). Multi3drefer: Grounding text description to multiple 3d objects. Proceedings of the IEEE/CVF International Conference on Computer Vision, 15225–15236.

Zhao, L., Cai, D., Sheng, L., & Xu, D. (2021). 3DVG-Transformer: Relation modeling for visual grounding on point clouds. Proceedings of the IEEE/CVF International Conference on Computer Vision, 2928–2937.

Zhou, C., Loy, C. C., & Dai, B. (2022). Extract free dense labels from CLIP. European Conference on Computer Vision, 696–712.

Zhou, S., Chang, H., Jiang, S., Fan, Z., Zhu, Z., Xu, D., Chari, P., You, S., Wang, Z., & Kadambi, A. (2024). Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. arXiv preprint arXiv:2312.03203.

Zhu, Z., Ma, X., Chen, Y., Deng, Z., Huang, S., & Li, Q. (2023). 3d-vista: Pre-trained transformer for 3d vision and text alignment. arXiv preprint arXiv:2308.04352.

Zou, X., Dou, Z.-Y., Yang, J., Gan, Z., Li, L., Li, C., Dai, X., Behl, H., Wang, J., Yuan, L., Peng, N., Wang, L., Lee, Y. J., & Gao, J. (2023). Generalized decoding for pixel, image, and language. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 15116–15127.

Zou, X., Yang, J., Zhang, H., Li, F., Li, L., Wang, J., Wang, L., Gao, J., & Lee, Y. J. (2023). Segment everything everywhere all at once. Advances in Neural Information Processing Systems, 36.