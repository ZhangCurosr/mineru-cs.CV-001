# CoverPrune: Coverage-Driven Token Pruning for 3D VLMs via Optimal Transport

Peng Ling<sup>1</sup>, Yingda Yin<sup>2†\*</sup>, Lingting Zhu<sup>2\*</sup>, Weikai Chen<sup>2</sup>, Shengju Qian<sup>2</sup>, Zeyu Hu<sup>2</sup>, Xin Wang<sup>2</sup>, and Wenming Yang<sup>1†</sup>

<sup>1</sup> Shenzhen International Graduate School, Tsinghua University <sup>2</sup> LIGHTSPEED

Abstract. While 3D Vision-Language Models (3D VLMs) have demonstrated remarkable spatial reasoning capabilities, they sufer from massive visual token counts that create severe computational bottlenecks during inference. Existing token pruning methods primarily rely on diversity-based selection, discarding similar tokens to maximize dispersion. However, in 3D environments, this approach frequently drops representative prototype tokens in favor of outliers, breaking the multi-view consistencies and geometric structures essential for spatial reasoning. In this paper, we propose a paradigm shift for 3D VLM token pruning: from maximizing diversity to preserving visual evidence coverage. We introduce CoverPrune, a training-free framework that formulates inferencetime token pruning as an Optimal Transport (OT) problem. To overcome the intractable combinatorial subset selection inherent in this formulation, we design the Feature-Spatial-Temporal (FST) transport cost and target capacity, along with an eficient Spatial-Guided Greedy Selection (SGS) algorithm to approximate the OT objective. Furthermore, we propose CoverPrune-Lite, an accelerated variant utilizing spatially structured local matching for minimal overhead. Extensive experiments across multiple 3D visual-spatial reasoning benchmarks demonstrate that our methods achieve state-of-the-art token eficiency, maintaining robust reasoning performance even under highly aggressive pruning budgets. Visit our project website at https://github.com/Brucess/CoverPrune.

Keywords: Visual Token Pruning · VLMs · Optimal Transport

## 1 Introduction

Building visual-spatial intelligence with large vision-language models [4,12,26,34] has recently led to the emergence of 3D vision-language models (3D VLMs), which inject explicit geometric cues from videos, multi-view observations, or 3D representations into the visual token stream. While this design enables powerful spatial reasoning capabilities, it also dramatically increases the number of visual tokens processed by the model. A single input may generate hundreds or even thousands of tokens, causing inference to be dominated by the quadratic complexity of attention and the growth of KV caches. As 3D VLMs scale toward richer visual environments, visual token eficiency becomes a key bottleneck, making inference-time token pruning essential for practical deployment.

Existing inference-time pruning methods for accelerating VLMs are primarily designed for 2D image or video understanding, and are typically evaluated on benchmarks [15, 24, 28, 37] that empha size coarse semantic or event-level comprehension. In these settings, token pruning is commonly performed using either attention-based ranking or diversitybased selection. Attention-based approaches [8, 20, 39, 40] remove tokens with low early-layer attention mass, while diversity-based approaches [1, 32] treat similarity in feature space as a proxy for redundancy and remove the most mutually similar tokens to maximize dispersion among the retained set. While these strategies are efective at reducing redundancy, they are not explicitly designed to preserve the representative visual evidence required for reasoning. Attention scores can be distorted by atten tion sinks and prompt-dependent saliency, while diversity-based pruning optimizes dispersion rather than representativeness.

![](images/ace78d64f30b5b7c23b00dd4078855a117971bc789e21c0be980f4e85d551814.jpg)  
Fig. 1: CoverPrune and CoverPrune-Lite Performance. We report crossbenchmark quantitative results under varying token retention ratios, with each dimension representing the average performance retention rate relative to the full token baseline across all metrics for each benchmark; our method achieves near-zero performance loss with 10% visual tokens on general 3D tasks and retains over 90% performance with 15% visual tokens on the reasoning-heavy VSI-Bench.

These observations raise a fundamental question: what is the appropriate objective for token pruning in 3D VLMs? Existing methods [18, 19, 23] largely adopt a diversity-based perspective, where tokens

that are similar to others are treated as redundant and removed to maximize dispersion among the retained set. While this strategy can reduce redundancy when compression is mild, it becomes increasingly misaligned with representativeness under aggressive pruning. Prototype tokens that represent dominant visual patterns are, by definition, similar to many other tokens within their mode and are therefore prone to early removal, causing the retained tokens to remain diverse yet skew toward outliers rather than representative observations. This failure mode is particularly problematic in 3D visual-spatial reasoning, where tokens frequently encode repeated multi-view observations that collectively establish geometric structure. Although these observations may appear redundant in feature space, they provide complementary evidence for reconstructing spatial relationships and maintaining geometric consistency. Removing them based solely on similarity can therefore break multi-view correspondences and degrade reasoning about spatial relations such as distance and ordering.

Our key insight is that efective token pruning should preserve coverage rather than maximize diversity. Instead of selecting tokens that are maximally diferent from one another, the retained tokens should collectively cover the informative content of the original token set. At the same time, the retained tokens must remain compact to enable eficient inference. This principle naturally leads to selecting a compact set of tokens that maximizes coverage of visual evidence. Driven by this insight, we propose CoverPrune, a training-free token pruning method for 3D visual-language models that formulates pruning as a coverage-with-compactness optimization problem. Rather than maximizing diversity among retained tokens, CoverPrune selects a compact subset of prototype tokens that collectively cover the informative content of the original visual token set. Towards this end, we reinterpret token pruning through the lens of Optimal Transport (OT): the retained tokens act as prototypes that distribute their representational mass to the original tokens, and pruning aims to minimize the distortion of this coverage assignment. This perspective naturally aligns token pruning with the objective of preserving representative observations while maintaining a compact token budget.

However, translating this coverage objective into a practical inference-time pruning algorithm presents several challenges. First, the transport cost must capture 3D representativeness, modeling feature similarity together with spatial and temporal consistency to preserve geometric structure. Second, under tight pruning budgets, naive transport formulations may allocate mass unevenly, requiring mechanisms that account for token informativeness. Third, unlike classical OT where the source and target supports are fixed, our setting requires jointly selecting prototype tokens and optimizing their coverage assignment, resulting in a combinatorial subset selection problem.

CoverPrune addresses these challenges with three key designs. We introduce a Feature-Spatial-Temporal (FST) transport cost that jointly models semantic similarity, spatial proximity, and temporal coherence. We further incorporate informativeness-aware target capacities to stabilize coverage under aggressive pruning. Finally, we develop an eficient Spatial-Guided Greedy Selection (SGS) algorithm that approximates the semi-relaxed OT objective for practical inference-time token selection. We also derive a lightweight variant, CoverPrune-Lite, which approximates the OT coverage objective through spatially structured local matching for faster pruning with minimal performance loss.

Extensive experiments on multiple 3D visual-spatial reasoning benchmarks demonstrate that CoverPrune consistently improves reasoning performance over state-of-the-art (SOTA) methods, as shown in Fig. 1. Compared with existing pruning methods, CoverPrune achieves better accuracy under the same token budget and maintains strong performance even under aggressive pruning. In summary, our main contributions are as follows:

– Novel Pruning Paradigm: We introduce CoverPrune, a training-free token pruning framework for 3D VLMs that fundamentally shifts the pruning objective from maximizing token diversity to preserving visual evidence coverage. We elegantly formulate this via Optimal Transport (OT) to retain a compact yet highly representative set of visual tokens.

Tailored OT Solutions: To resolve the inherent challenges of applying OT to inference-time token pruning, we propose three key designs: a Feature-Spatial-Temporal (FST) cost that comprehensively models multidimensional token relationships, dynamic target capacities that prioritize informative tokens, and an eficient optimization algorithm.

– Lightweight Acceleration: We design CoverPrune-Lite, a highly eficient variant that approximates the OT coverage objective through spatially structured local matching, significantly reducing pruning overhead with minimal performance degradation.

– State-of-the-Art Performance: Extensive experiments across multiple 3D visual–spatial reasoning benchmarks demonstrate that CoverPrune outperforms existing state-of-the-art token pruning methods, establishing superior token robustness, especially under aggressive pruning budgets.

## 2 Related Work

## 2.1 Large Vision Language Models for 3D Understanding

Spatial understanding is increasingly framed as a key ingredient of multimodal intelligence for embodied agents and scene centric reasoning [22, 27, 35, 38, 42]. Recent VLMs provide a strong base by pairing robust 2D perception with scalable language reasoning [4,12,26,34]. Building on these pretrained backbones, a dominant paradigm introduces explicit geometry into visual tokens so that spatial reasoning can rely on 3D evidence rather than implicitly recovering structure from 2D cues. A series of works inject 3D signals derived from SfM or geometry foundation models into pre-trained LVLMs, and finetune the resulting 3D VLMs for spatial question answering tasks. SR-3D [11] incorporates 3D aware region representations to support spatially grounded language interaction. Spatial-MLLM [36] and GS-Reasoner [9] further emphasize geometry augmented tokenization and grounded reasoning, showing that explicit structure improves spatial queries under viewpoint changes. Other eforts broaden supervision and objectives for 3D and video spatial understanding, including position aware training signals and instruction aligned spatial tuning [14, 41]. Despite rapid progress in 3D VLMs, the critical issue of token explosion induced by multi-frame inputs has yet to be adequately addressed, leaving eficient inference a long-standing bottleneck for them with spatial reasoning capabilities.

![](images/2f600bfdc4da7ae0854bea750c6a0b86dd39612483a7fb8c1295f3e01ca21e3c.jpg)  
Fig. 2: Framework overview. (Left) CoverPrune serves as a training-free, plug-andplay module inserted between the visual-geometric encoder and the 3D VLM. (Right) We formulate token pruning as an Optimal Transport (OT) problem to maximize visual evidence coverage. To resolve this, we introduce three key designs: (1) a Feature-Spatial-Temporal (FST) Cost $\mathbf { C } \left( d _ { f } , d _ { x } , d _ { \tau } \right)$ to comprehensively model multidimensional token relationships; (2) an asymmetric capacity assignment to stabilize mass allocation based on token informativeness; and (3) a tractable optimization strategy to approximate the inherently NP-hard combinatorial subset selection problem.

## 2.2 Visual Token Pruning

Visual token reduction is widely studied for accelerating VLM inference in generic image and video understanding, where evaluation typically emphasizes coarse perception and caption style or short form reasoning rather than quantitative spatial grounding [15, 24, 28, 37]. One direction compresses tokens through redesigned multimodal projectors. Honeybee [6], LLaVA-UHD [16], and Token-Packer [25] reduce visual token counts before the language model, but commonly require architectural changes and end-to-end adaptation. A lighter line performs training-free pruning and mainly follows attention-based or diversity-based criteria [3]. Attention-based approaches [8,20,39,40] estimate token saliency from self attention or cross-modal attention and remove low score tokens. Diversity-based methods [1,32] reduce redundancy by feature space merging or maximizing diversity of retained set. While efective for generic understanding, these heuristics can be misaligned with spatial reasoning, since attention patterns can evolve across layers and decoding steps, and feature-driven merging can distort the representative tokens needed for complex spatial reasoning. Only a few works explicitly tailor token reduction to spatial tasks. DTC [19] compresses inputs for 3D question answering with voxel grounded token compression, EgoPrune [23] leverages SfM pose cues to align overlapping regions before filtering redundant tokens, and ToSA [18] introduces spatial awareness signals to guide safer merging.

## 3 CoverPrune

## 3.1 Preliminary: Optimal Transport

Optimal Transport (OT) [33] provides a principled way to measure how well one weighted set can be matched to another under a chosen notion of cost. Consider a source set $\{ s _ { i } \} _ { i = 1 } ^ { m }$ and a target set $\{ t _ { j } \} _ { j = 1 } ^ { n }$ , equipped with nonnegative capacities $\mathbf { u } \in \mathbb { R } _ { + } ^ { m }$ and $\mathbf { v } \in \mathbb { R } _ { + } ^ { n }$ , which are often normalized so $\mathbf { u } ^ { \top } \mathbf { 1 } = 1$ and $\mathbf { v } ^ { \top } \mathbf { 1 } = 1$ Throughout this paper, we use capacity to refer to these OT marginal vectors $( { \mathrm { i . e . } }$ , distributions and weights). Let $\mathbf { C } \in \mathbb { R } ^ { m \times n }$ be a cost matrix, where $C _ { i j }$ quantifies the cost of assigning $s _ { i }$ to $t _ { j } . \mathrm { A }$ transport plan is a nonnegative matrix $\mathbf { P } \in \mathbb { R } _ { + } ^ { m \times n }$ whose row and column sums match the prescribed capacities:

$$
\mathbf { P 1 } = \mathbf { u } , \qquad \mathbf { P } ^ { \top } \mathbf { 1 } = \mathbf { v } .\tag{1}
$$

The OT objective finds the least-cost plan:

$$
\operatorname { O T } ( \mathbf { u } , \mathbf { v } ) = \operatorname* { m i n } _ { \mathbf { P } \geq 0 } ~ \langle \mathbf { C } , \mathbf { P } \rangle ~ \mathrm { s . t . } ~ \mathbf { P } \mathbf { 1 } = \mathbf { u } , ~ \mathbf { P } ^ { \top } \mathbf { 1 } = \mathbf { v } ,\tag{2}
$$

where $\begin{array} { r } { \langle \mathbf { C } , \mathbf { P } \rangle = \sum _ { i , j } C _ { i j } P _ { i j } } \end{array}$

For eficiency, a common practice is to add an entropic regularizer with weight $\varepsilon > 0$ and solve the smoothed problem with Sinkhorn [13] iterations:

$$
\operatorname { O T } _ { \varepsilon } ( \mathbf { u } , \mathbf { v } ) = \operatorname* { m i n } _ { \mathbf { P } > 0 } \left. \mathbf { C } , \mathbf { P } \right. - \varepsilon H ( \mathbf { P } ) \ \mathrm { ~ s . t . ~ } \ \mathbf { P } \mathbf { 1 } = \mathbf { u } , \ \mathbf { P } ^ { \top } \mathbf { 1 } = \mathbf { v } ,\tag{3}
$$

where $\begin{array} { r } { H ( \mathbf { P } ) = - \sum _ { i , j } P _ { i j } ( \log P _ { i j } - 1 ) } \end{array}$

## 3.2 Problem Setup

The overview of CoverPrune is shown in Fig. 2. We first introduce the problem formulation. Let $\mathcal { T } = \{ t _ { j } \} _ { j = 1 } ^ { N }$ denote all visual tokens extracted from an input video before being fed into the backbone of a 3D VLM. Each token $t _ { i }$ is associated with a feature embedding $\mathbf { f } _ { j } \in \mathbb { R } ^ { d }$ , a 3D global coordinate $\mathbf { x } _ { j } \in \mathbb { R } ^ { 3 }$ that can be estimated via SfM or a geometry foundation model, and a timestamp $\tau _ { j }$ of its corresponding frame. Given a retention ratio $R \in \mathsf { \Gamma } ( 0 , 1 ]$ we set the pruning budget as $K = \lceil R N \rceil$ and aim to select a subset $s \subseteq \tau$ with $| S | = K$ as the visual input for subsequent decoding:

$$
\quad S ^ { \star } = \arg \operatorname* { m a x } _ { \substack { S \subseteq T , \ | S | = K } } \mathrm { ~ C o v e r } ( S ; T ) .\tag{4}
$$

To obtain a principled and computable notion of coverage, we cast prototype selection as minimizing the discrepancy between the selected set and the full token set. Concretely, we treat $s$ as a source support that should explain the target support $\tau ,$ and measure their mismatch via an OT objective. We assign nonnegative capacities to tokens in $s$ and $\tau$ , define a pairwise cost between any retained token and any original token, and compute an OT matching cost by optimizing a transport plan:

$$
{ \mathcal { L } } _ { \mathrm { O T } } ( S ; { \mathcal { T } } ) = \operatorname* { m i n } _ { \mathbf { P } \geq 0 } { \big \langle } \mathbf { C } ( S , { \mathcal { T } } ) , \mathbf { P } { \big \rangle } \quad { \mathrm { s . t . } } \quad \mathbf { P 1 } = \mathbf { u } , \ \mathbf { P } ^ { \top } \mathbf { 1 } = \mathbf { v } .\tag{5}
$$

Here, $\mathbf { C } ( S , \mathcal { T } )$ is the cost matrix with entries $C _ { i j }$ measuring the discrepancy between a retained token $s _ { i } ~ \in ~ S$ and an original token $t _ { j } ~ \in ~ \tau ;$ P denotes the transport plan; and u and v are normalized capacity vectors over $s$ and $\tau$ respectively. In our setting, choosing u to be uniform is a natural and reasonable default, since the selected prototypes serve as an unlabeled summary set without prior preference among them. In contrast, a uniform v is generally suboptimal because original tokens can vary substantially in informativeness, and treating them equally may allocate excessive mass to uninformative or noisy regions while under-emphasizing salient spatial evidence.

CoverPrune then selects the retained token set by minimizing this transport cost:

$$
\mathcal { S } ^ { \star } = \arg \operatorname* { m i n } _ { \mathcal { S } \subseteq \mathcal { T } , \ | \mathcal { S } | = K } \ \mathcal { L } _ { \mathrm { O T } } ( \mathcal { S } ; \mathcal { T } ) .\tag{6}
$$

In the following subsections, we detail how to instantiate Eq. (5)–(6) by (i) designing a multi-domain transport cost $\mathbf { C } ( S , \mathcal { T } )$ , (ii) specifying the target capacities v via an informativeness-aware reweighting scheme, and (iii) developing an eficient inference-time optimization strategy for selecting $s$ despite the underlying NP-hard combinatorial search.

## 3.3 Feature-Spatial-Temporal Transport Cost

3D visual-spatial reasoning demands information preservation beyond pure semantics. Prior pruning methods judge redundancy solely in the feature domain, risking damage to the geometric structure critical for grounding, while ignoring temporal order introduces inconsistent evidence of region observation timing and location. To address this, we design a Feature-Spatial-Temporal (FST) criterion to jointly model token relations across these three domains: (i) featurespace proximity minimizes semantic distortion during token substitution; (ii) 3D spatial proximity preserves geometric integrity for accurate object and relation grounding in reasoning; (iii) temporal consistency aligns the retained token set with the true order of spatial evidence, mitigating errors from temporal misassociation.

To operationalize this FST criterion, we quantify three pairwise discrepancies between a token s and a token t:

$$
\begin{array} { r } { d _ { f } ( s , t ) = 1 - \cos ( \mathbf { f } _ { s } , \mathbf { f } _ { t } ) , \quad d _ { x } ( s , t ) = \| \mathbf { x } _ { s } - \mathbf { x } _ { t } \| _ { 2 } , \quad d _ { \tau } ( s , t ) = \mathrm { R e L U } ( \tau _ { s } - \tau _ { t } ) , } \end{array}\tag{7}
$$

where f denotes the token embedding, x is the 3D global coordinate, and $\tau$ is the timestamp. The temporal term uses ReL $\mathrm { U } ( z ) = \operatorname* { m a x } ( z , 0 )$ to penalize covering a token observed earlier in time with one observed later, encouraging the retained set to respect the temporal order of spatial evidence. We define the transport cost between a retained token $s _ { i } \in S$ and an original token $t _ { j } \in \tau$ as a weighted sum:

$$
C _ { i j } = \lambda _ { f } \hat { d } _ { f } ( s _ { i } , t _ { j } ) + \lambda _ { x } \phi _ { \kappa } \big ( \hat { d } _ { x } ( s _ { i } , t _ { j } ) \big ) + \lambda _ { \tau } \hat { d } _ { \tau } ( s _ { i } , t _ { j } ) ,\tag{8}
$$

where $\boldsymbol { \hat { d } } ( \cdot , \cdot )$ denotes a min-max normalized discrepancy computed within the current sample, and $\lambda _ { f } , \lambda _ { x } , \lambda _ { \tau }$ control the relative importance of feature, spatial, and temporal terms. We further apply a nonlinear mapping $\phi _ { \kappa } ( x ) = \log ( 1 +$ $\kappa x ) / \log ( 1 + \kappa )$ with $\kappa > 0$ , which expands the dynamic range for near-field distances. $C _ { i j }$ serves as a unified notion of substitutability. Consequently, the transport plan favors allocating mass along semantically aligned, geometrically consistent, and temporally coherent correspondences, which directly steers the selected tokens toward globally faithful coverage.

## 3.4 Feature-Spatial-Temporal Capacity

The transport cost in Eq. (8) specifies how mass should be assigned once S is given, but efective coverage under a tight budget also depends on which target tokens deserve more coverage. We therefore introduce an FST capacity vector $\mathbf { v } \in \mathbb { R } _ { + } ^ { N }$ to parameterize the target capacity in Eq. (5), where a larger $v _ { j }$ encourages allocating more transport mass to token $t _ { j }$ . Our design follows the same FST criterion: tokens that are harder to approximate by their local neighborhood in the FST sense tend to carry more distinctive information and should be prioritized.

Specifically, for each target token $t _ { j } ,$ , we compute a local distinctiveness score by averaging its FST discrepancy to a small neighbor set $\mathcal { N } _ { n } ( t _ { j } )$ :

$$
r _ { j } = \frac { 1 } { | \mathcal { N } _ { n } ( t _ { j } ) | } \sum _ { t _ { k } \in \mathcal { N } _ { n } ( t _ { j } ) } \Big ( \alpha _ { f } \hat { d } _ { f } ( t _ { j } , t _ { k } ) + \alpha _ { x } \hat { d } _ { x } ( t _ { j } , t _ { k } ) + \alpha _ { \tau } \hat { d } _ { \tau } ( t _ { j } , t _ { k } ) \Big ) ,\tag{9}
$$

where $\mathcal { N } _ { n } ( t _ { j } )$ denotes the set of n nearest neighbors of $t _ { j }$ in 3D space, and $\alpha _ { f } , \alpha _ { x } , \alpha _ { \tau }$ are capacity weights. We then map $\{ r _ { j } \}$ to a nonnegative capacity vector and normalize it to match the pruning budget:

$$
v _ { j } = \frac { \phi ( r _ { j } ) } { \sum _ { l = 1 } ^ { N } \phi ( r _ { l } ) } ,\tag{10}
$$

where $\phi ( \cdot )$ is a monotone increasing mapping that controls how capacity concentrates on informative tokens. This construction allocates more capacity to tokens that are locally distinctive in feature, geometry, or time, preventing dense but redundant regions from dominating the coverage objective.

## 3.5 Optimization

Semi-Relaxed Optimal Transport. Classical OT in Eq. (2) assumes that both supports and their weights are fixed, and optimizes only the transport plan. In our setting, however, the source support is itself a decision variable: we seek a subset ${ \mathcal { S } } \subseteq \{ s _ { i } \} _ { i = 1 } ^ { m }$ with $| S | \le K$ and a capacity vector supported on $s$ that best matches a fixed target capacity under the OT cost. This yields a coupled discrete–continuous problem, where one must jointly select S and solve for the optimal coupling. Such formulations are generally intractable to solve exactly and are known to be NP-hard [17, 21].

To obtain an eficient solver with provable approximation behavior, an approach is to relax the strict OT marginal constraints while preserving the OT principle of minimizing transport cost. The key idea is to introduce slack on the target side so that the induced set objective becomes amenable to greedy optimization, and in particular to submodular maximization. Concretely, we adopt the semi-relaxed OT formulation [5, 7, 30]:

$$
\operatorname { S O T } ( \mathbf { u } , \mathbf { v } ) = \operatorname* { m i n } _ { \mathbf { P } \geq 0 } ~ \langle \mathbf { C } , \mathbf { P } \rangle \quad \mathrm { s . t . } \quad \mathbf { P 1 } = \mathbf { u } , ~ \mathbf { P } ^ { \top } \mathbf { 1 } \leq \mathbf { v } ,\tag{11}
$$

where $\mathbf { u } \in \mathbb { R } _ { + } ^ { m }$ denotes the source capacity and $\mathbf { v } \in \mathbb { R } _ { + } ^ { n }$ specifies per-target capacities. Compared to OT, the inequality constraint $\mathbf { P } ^ { \top } \mathbf { 1 } \leq \mathbf { v }$ permits unused capacity, which provides exactly the flexibility needed for tractable support optimization. Prior work shows that, under such relaxed Wasserstein objectives, the induced set functions for subset selection can be monotone submodular, and therefore admit greedy maximization with constant-factor approximation guarantees under a cardinality constraint [17,21]. When the relaxation is tight, semirelaxed OT recovers standard OT as a special case [21, 30]. Thus, semi-relaxed OT serves as a principled relaxation that enables greedy subset construction while remaining consistent with the OT objective at the target budget. Similar to Eq. (3), we further add an entropic regularizer to obtain a smooth objective:

$$
\mathrm { S O T } _ { \varepsilon } ( \mathbf { u } , \mathbf { v } ) = \operatorname* { m i n } _ { \mathbf { P } \geq 0 } \left. \mathbf { C } , \mathbf { P } \right. - \varepsilon H ( \mathbf { P } ) \mathrm { ~ s . t . ~ } \mathbf { P 1 } = \mathbf { u } , \mathbf { P } ^ { \top } \mathbf { 1 } \leq \mathbf { v } .\tag{12}
$$

Spatial-Guided Greedy Selection. While the semi-relaxed formulation makes subset construction algorithmically feasible, a naive greedy implementation is still computationally prohibitive for long video sequences. In particular, evaluating the marginal benefit of adding each candidate token would require re-solving a transport problem at every greedy step, resulting in excessive runtime.

We therefore propose Spatial-Guided Greedy Selection (SGS), built on a simple locality prior in 3D scenes: a token can efectively cover only tokens that are spatially nearby, since distant 3D regions typically correspond to diferent surfaces or objects and incur high transport cost. This observation motivates restricting marginal-cost evaluation to a small 3D neighborhood, which avoids repeated global OT computations from each candidate token to all target tokens. Specifically, for each candidate token t, we define its neighborhood within the target set as

$$
\begin{array} { r } { \mathcal { M } _ { g } ( t ) = \mathrm { N N } _ { g } ( t ; \mathcal { T } ) , } \end{array}\tag{13}
$$

where $\mathrm { N N } _ { g }$ returns the $g$ nearest target tokens in 3D space. At greedy step $\ell ,$ given the current selected set $S _ { \ell } .$ we solve a single semi-relaxed OT problem to

obtain a transport plan $\mathbf { P } _ { \ell }$ and compute the residual capacity on the target side:

$$
\mathbf { r } _ { \ell } = \left[ \mathbf { v } - \mathbf { P } _ { \ell } ^ { \top } \mathbf { 1 } \right] _ { + } .\tag{14}
$$

We then select the next token by minimizing a local, residual-weighted marginal cost:

$$
t _ { \ell } ^ { \star } = \arg \operatorname* { m i n } _ { t \in \mathcal { T } \setminus \mathcal { S } _ { \ell } } \sum _ { t _ { j } \in \mathcal { M } _ { g } ( t ) } r _ { \ell , j } C ( t , t _ { j } ) ,\tag{15}
$$

and update the set as

$$
S _ { \ell + 1 } = S _ { \ell } \cup \{ t _ { \ell } ^ { \star } \} ,\tag{16}
$$

where $C ( \cdot , \cdot )$ is the FST cost and v is the corresponding capacity vector.

## 4 CoverPrune-Lite: Block-Structured OT Approximation via 3D-Aware Ordering

CoverPrune constructs the retained set with a greedy procedure that repeatedly solves semi-relaxed OT. While principled, this iterative global transport optimization incurs cubic-time complexity in the number of tokens, which becomes a bottleneck for long video sequences. To further improve eficiency, we propose CoverPrune-Lite, which exploits a spatial locality prior that efective coverage is predominantly supported by spatially nearby tokens in 3D. CoverPrune-Lite therefore approximates OT-style coverage by restricting transport to local 3D neighborhoods, eliminating iterative transport solving.

## 4.1 3D-Aware Ordering and Capacity-Guided Grouping

We first build a 3D-aware ordering of all target tokens via the Morton code space-filling curve [31]. This ordering preserves spatial locality, so tokens that are adjacent in the sorted list are likely to be proximal in 3D space, enabling coherent neighborhood construction without explicit nearest-neighbor search. On top of this order, we partition tokens into K non-overlapping groups using the target FST capacity. Let $\textbf { v } \in \mathbb { R } _ { + } ^ { N }$ denote the per-token target capacity, normalized as a distribution with $\begin{array} { r } { \sum _ { j = 1 } ^ { N } v _ { j } = 1 } \end{array}$ . We traverse the Morton-ordered list and accumulate capacity until it reaches $1 / K$ , then finalize a group and start a new one, producing K groups $\{ \mathcal { G } _ { q } \} _ { q = 1 } ^ { K }$ that satisfy

$$
\sum _ { t _ { j } \in \mathcal { G } _ { q } } v _ { j } \approx \frac { 1 } { K } \qquad q = 1 , \dots , K .\tag{17}
$$

This adaptive grouping assigns approximately equal information mass to each group. Regions with many redundant tokens tend to have small per-token capacity and thus form larger groups, whereas informative regions have larger per-token capacity and form smaller, more fine-grained groups.

We then choose one prototype token from each group by minimizing the capacity-weighted transport cost within the group,

$$
s _ { q } = \arg \operatorname* { m i n } _ { t \in \mathcal { G } _ { q } } \sum _ { t _ { j } \in \mathcal { G } _ { q } } v _ { j } C ( t , t _ { j } ) ,\tag{18}
$$

and output the pruned set as ${ \cal S } = \{ s _ { q } \} _ { q = 1 } ^ { K }$

## 4.2 Connection to Coverage Objective

CoverPrune-Lite can be understood as a block-constrained variant of our OT objective (i.e., Eq. (6)), where the transport plan is restricted to be locally supported on the pre-defined groups. With the same source and target capacities u and v defined in CoverPrune, we restrict the feasible couplings to

$$
\mathcal { P } _ { \mathrm { b l k } } = \Big \{ \mathbf { P } \geq 0 \big | \mathbf { P } \mathbf { 1 } = \mathbf { u } , \ \mathbf { P } ^ { \top } \mathbf { 1 } = \mathbf { v } , \ P _ { q j } = 0 \ \mathrm { i f } \ t _ { j } \notin \mathcal { G } _ { q } \Big \} .\tag{19}
$$

Here we take a uniform source capacity over the K retained tokens. For $q =$ $1 , \ldots , K$ , so that $\textstyle \sum _ { q } u _ { q } = 1 = \sum _ { j } v _ { j }$

Given a fixed partition $\{ { \mathcal { G } } _ { q } \}$ , we then consider the constrained objective

$$
\operatorname* { m i n } _ { \boldsymbol { s } _ { q } \in \mathcal { G } _ { q } } \quad \operatorname* { m i n } _ { \mathbf { P } \in \mathcal { P } _ { \mathrm { b l k } } } \ \langle \mathbf { C } ( \boldsymbol { S } , \mathcal { T } ) , \mathbf { P } \rangle .\tag{20}
$$

When the grouping satisfies $\textstyle \sum _ { t _ { j } \in { \mathcal { G } } _ { q } } v _ { j } = 1 / K$ , the block constraint forces each prototype token $s _ { q }$ to send its entire mass $u _ { q } = 1 / K$ within $\mathcal { G } _ { q } .$ . In this case, the optimal coupling is uniquely determined as $P _ { q j } = v _ { j }$ for $t _ { j } \in \mathcal G _ { q }$ and $P _ { q j } = 0$ otherwise. Substituting this into Eq. (20) yields $\begin{array} { r } { \sum _ { q = 1 } ^ { K } \sum _ { t _ { i } \in \mathcal { G } _ { q } } v _ { j } C ( s _ { q } , t _ { j } ) } \end{array}$ , which decouples across groups and recovers exactly the within-group selection rule in Eq. (18). Therefore, CoverPrune-Lite approximates CoverPrune by enforcing a block-diagonal, locally supported transport structure, replacing iterative global OT optimization with a single pass of 3D-aware grouping and locally optimal prototype selection, and achieving O(N log N) inference-time complexity.

## 5 Experiments

## 5.1 Experimental Settings

Datasets and Benchmarks. To comprehensively evaluate our training-free CoverPrune and CoverPrune-Lite, we conduct extensive experiments on four mainstream 3D vision-language benchmarks. We first validate our method on three widely used fine-grained reasoning benchmarks: ScanQA [2], SQA3D [29], and Scan2Cap [10]. We further apply our method to VSI-Bench [38], an egocentric indoor scan-based video benchmark for complex spatial-temporal reasoning, with full evaluation across eight tasks: Object Count, Relative Distance, Relative Direction, Route Plan, Object Size, Room Size, Absolute Distance, and Appearance Order.

Table 1: Evaluation on General 3D Tasks. Vanilla baseline results from GS-Reasoner [9]. Retention ratio R is the fraction of visual tokens retained post-pruning. Per-benchmark Acc.% is the average relative performance retention across all its metrics. CoverPrune and its Lite variant consistently hit top-1/top-2 on most metrics, with strong multi-task robustness.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Retention Ratio R</td><td colspan="5">Scan2Cap</td><td colspan="6">ScanQA</td><td colspan="2">SQA3D</td></tr><tr><td>Acc.%↑</td><td>B-4↑</td><td>Rouge↑</td><td>CIDEr↑</td><td>Meteor↑</td><td>Acc.%↑</td><td>B-4↑</td><td>Rouge↑</td><td>CIDEr↑</td><td>Meteor↑</td><td>EM↑</td><td>Acc.%↑</td><td>EM↑</td></tr><tr><td>Vanilla</td><td>100%</td><td>100.00</td><td>47.60</td><td>69.20</td><td>101.00</td><td>32.10</td><td>100.00</td><td>16.20</td><td>49.20</td><td>102.60</td><td>19.80</td><td>29.90</td><td>100.00</td><td>59.90</td></tr><tr><td>VisionZip (CVPR25)</td><td rowspan="5">20%</td><td>99.29</td><td>49.02</td><td>70.96</td><td>93.48</td><td>31.81</td><td>99.85</td><td>17.68</td><td>47.77</td><td>101.19</td><td>19.71</td><td>28.36</td><td>97.96</td><td>58.68</td></tr><tr><td>FastVID (NeurIPS25)</td><td>99.71</td><td>49.15</td><td>70.85</td><td>94.78</td><td>31.90</td><td>98.85</td><td>16.60</td><td>48.17</td><td>101.30</td><td>19.73</td><td>28.56</td><td>98.01</td><td>58.71</td></tr><tr><td>DTC (CVPR25)</td><td>98.70</td><td>48.54</td><td>70.52</td><td>93.04</td><td>31.71</td><td>96.85</td><td>16.14</td><td>47.38</td><td>98.80</td><td>19.29</td><td>28.28</td><td>96.83</td><td>58.00</td></tr><tr><td>EgoPrune (arXiv25)</td><td>88.79</td><td>43.95</td><td>68.49</td><td>71.66</td><td>29.82</td><td>88.67</td><td>15.28</td><td>43.83</td><td>88.55</td><td>17.86</td><td>24.94</td><td>92.04</td><td>55.13</td></tr><tr><td>CoverPrune</td><td>101.68</td><td>50.10</td><td>71.18</td><td>99.35</td><td>32.18</td><td>100.95</td><td>16.99</td><td>48.87</td><td>103.17</td><td>20.04</td><td>29.54</td><td>99.67</td><td>59.70</td></tr><tr><td>CoverPrune-Lite</td><td></td><td>101.31</td><td>49.94</td><td>71.17</td><td>98.22</td><td>32.17</td><td>102.46</td><td>17.63</td><td>49.42</td><td>104.69</td><td>20.32</td><td>29.41</td><td>97.45</td><td>58.37</td></tr><tr><td>VisionZip (CVPR25)</td><td rowspan="5">10%</td><td>93.65</td><td>46.61</td><td>69.62</td><td>81.30</td><td>30.68</td><td>90.76</td><td>15.23</td><td>44.66</td><td>92.13</td><td>18.29</td><td>25.97</td><td>92.60</td><td>55.47</td></tr><tr><td>FastVID (NeurIPS25)</td><td>95.00</td><td>46.62</td><td>69.75</td><td>85.68</td><td>30.96</td><td>94.93</td><td>15.58</td><td>46.67</td><td>97.51</td><td>19.04</td><td>27.64</td><td>95.26</td><td>57.06</td></tr><tr><td>DTC (CVPR25)</td><td>94.33</td><td>46.60</td><td>69.66</td><td>83.42</td><td>30.87</td><td>90.81</td><td>14.84</td><td>45.01</td><td>92.50</td><td>18.24</td><td>26.52</td><td>92.60</td><td>55.47</td></tr><tr><td>EgoPrune (arXiv25)</td><td>80.69</td><td>40.23</td><td>66.73</td><td>53.93</td><td>28.38</td><td>80.66</td><td>13.66</td><td>40.39</td><td>80.01</td><td>16.42</td><td>22.72</td><td>86.86</td><td>52.03</td></tr><tr><td>CoverPrune</td><td>99.04</td><td>48.85</td><td>70.63</td><td>93.84</td><td>31.64</td><td>100.00</td><td>17.63</td><td>47.98</td><td>100.87</td><td>19.71</td><td>28.64</td><td>97.45</td><td>58.37</td></tr><tr><td>CoverPrune-Lite</td><td></td><td>99.07</td><td>48.75</td><td>70.69</td><td>93.71</td><td>31.76</td><td>98.93</td><td>17.34</td><td>47.66</td><td>100.29</td><td>19.55</td><td>28.19</td><td>96.83</td><td>58.00</td></tr></table>

Baselines. We compare CoverPrune and CoverPrune-Lite against four stateof-the-art (SOTA) training-free visual token pruning methods: two general multimodal methods, VisionZip [39] and FastVID [32], which integrate attentionbased importance estimation with feature diversity heuristics; two 3D VLMspecific methods, DTC [19] with a diversity-driven token selection strategy, and EgoPrune [23] with attention-diversity fused token merging.

Implementation Details. To validate the generalizability of our proposed method, we instantiate it on two SOTA 3D VLMs, GS-Reasoner [9] and VLM-3R [14], both augmenting visual tokens with geometric cues for 3D spatial reasoning. We keep all default base-model configurations fully unchanged for a fair, controlled comparison, including uniform 32-frame sampling, and follow the protocol in GS-Reasoner to generate all token coordinates via an estimator without using ground-truth values. Our methods are inserted immediately before the LLM prefill stage, operating directly on raw visual tokens with compatibility across diverse acceleration frameworks. We set $\lambda _ { f } = \lambda _ { x } = \lambda _ { \tau } = 1$ and $\alpha _ { f } = \alpha _ { x } = \alpha _ { \tau } = 1$ in our experiments.

## 5.2 Efectiveness Evaluation

Results on General 3D Tasks. Table 1 demonstrates that CoverPrune and CoverPrune-Lite achieve top-tier performance on nearly all reported metrics under matched token budgets, consistently ranking first or tied for first across 3D captioning and question answering tasks with well-generalized performance gains. Critically, our performance lead over competing baselines widens significantly under more aggressive pruning. However, we observe that SOTA performance on general 3D QA tasks has largely saturated, motivating us to pioneer systematic token pruning evaluation on VSI-Bench, a visual-spatial reasoning benchmark with substantially higher complexity and reasoning dificulty.

Table 2: Evaluation on VSI-Bench with GS-Reasoner as the base model. Under each ratio, the number of tokens retained by each method was kept consistent. Our proposed CoverPrune and CoverPrune-Lite consistently exhibit superior performance.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Retention Ratio R</td><td rowspan="2">Avg.</td><td colspan="4">Numerical Answer</td><td colspan="4">Multiple-Choice Answer</td></tr><tr><td>Obj. Count</td><td>Abs. Dist.</td><td>Obj. Size</td><td></td><td>Room Size Rel. Dist. Rel. Dir.</td><td></td><td>Route Plan</td><td>Appr. Order</td></tr><tr><td>Vanilla</td><td>100%</td><td>64.70</td><td>69.10</td><td>61.90</td><td>70.00</td><td>65.70</td><td>65.40</td><td>88.90</td><td>44.30</td><td>52.30</td></tr><tr><td>VisionZip (CVPR25)</td><td rowspan="6">20%</td><td>57.55</td><td>67.82</td><td>50.16</td><td>61.10</td><td>53.26</td><td>59.86</td><td>78.31</td><td>40.72</td><td>49.19</td></tr><tr><td>FastVID (NeurIPS25)</td><td>55.52</td><td>67.04</td><td>49.62</td><td>63.85</td><td>52.81</td><td>54.65</td><td>73.40</td><td>34.54</td><td>48.22</td></tr><tr><td>DTC (CVPR25)</td><td>56.57</td><td>68.55</td><td>51.03</td><td>64.80</td><td>46.91</td><td>55.07</td><td>78.99</td><td>35.57</td><td>51.62</td></tr><tr><td>EgoPrune (arXiv25)</td><td>49.44</td><td>63.72</td><td>43.53</td><td>59.34</td><td>54.06</td><td>44.51</td><td>65.84</td><td>35.57</td><td>28.96</td></tr><tr><td>CoverPrune</td><td>59.76</td><td>69.38</td><td>55.04</td><td>66.71</td><td>53.26</td><td>59.44</td><td>84.37</td><td>38.14</td><td>51.78</td></tr><tr><td>CoverPrune-Lite</td><td>59.43</td><td>68.87</td><td>54.03</td><td>66.31</td><td>57.64</td><td>57.18</td><td>82.20</td><td>37.11</td><td>52.10</td></tr><tr><td>VisionZip (CVPR25)</td><td rowspan="6">15%</td><td>53.37</td><td>66.35</td><td>44.94</td><td>60.60</td><td>50.38</td><td>50.14</td><td>74.65</td><td>36.08</td><td>43.85</td></tr><tr><td>FastVID (NeurIPS25)</td><td>54.03</td><td>66.51</td><td>47.13</td><td>61.67</td><td>52.29</td><td>49.44</td><td>72.12</td><td>35.05</td><td>48.06</td></tr><tr><td>DTC (CVPR25)</td><td>55.07</td><td>68.41</td><td>49.00</td><td>63.99</td><td>45.28</td><td>49.01</td><td>76.99</td><td>38.66</td><td>49.19</td></tr><tr><td>EgoPrune (arXiv25)</td><td>46.94</td><td>63.40</td><td>39.66</td><td>57.40</td><td>53.16</td><td>38.87</td><td>59.43</td><td>34.02</td><td>29.61</td></tr><tr><td>CoverPrune</td><td>58.27</td><td>69.42</td><td>55.10</td><td>66.46</td><td>52.81</td><td>58.03</td><td>77.13</td><td>35.57</td><td>51.62</td></tr><tr><td>CoverPrune-Lite</td><td>57.72</td><td>69.35</td><td>51.83</td><td>64.06</td><td>57.81</td><td>55.63</td><td>77.22</td><td>35.05</td><td>50.81</td></tr><tr><td>VisionZip (CVPR25)</td><td rowspan="6">10%</td><td>50.36</td><td>64.74</td><td>41.80</td><td>58.58</td><td>49.55</td><td>42.39</td><td>66.83</td><td>34.02</td><td>44.98</td></tr><tr><td>FastVID (NeurIPS25)</td><td>51.56</td><td>64.92</td><td>42.36</td><td>60.33</td><td>51.98</td><td>45.49</td><td>68.41</td><td>37.11</td><td>41.91</td></tr><tr><td>DTC (CVPR25)</td><td>51.66</td><td>66.87</td><td>44.10</td><td>61.86</td><td>44.76</td><td>48.17</td><td>70.32</td><td>34.02</td><td>43.20</td></tr><tr><td>EgoPrune (arXiv25)</td><td>44.71</td><td>63.06</td><td>35.10</td><td>53.99</td><td>52.88</td><td>37.46</td><td>54.73</td><td>34.54</td><td>25.89</td></tr><tr><td>CoverPrune</td><td>56.83</td><td>67.98</td><td>51.69</td><td>63.47</td><td>53.16</td><td>50.00</td><td>79.67</td><td>39.18</td><td>49.51</td></tr><tr><td>CoverPrune-Lite</td><td>56.94</td><td>67.13</td><td>47.91</td><td>63.19</td><td>55.76</td><td>53.10</td><td>78.43</td><td>39.69</td><td>50.32</td></tr><tr><td>VisionZip (CVPR25)</td><td rowspan="6">5%</td><td>46.10</td><td>62.87</td><td>36.38</td><td>54.59</td><td>51.11</td><td>36.76</td><td>54.68</td><td>37.11</td><td>35.28</td></tr><tr><td>FastVID (NeurIPS25)</td><td>46.76</td><td>63.19</td><td>37.21</td><td>54.59</td><td>53.19</td><td>41.41</td><td>53.73</td><td>34.54</td><td>36.25</td></tr><tr><td>DTC (CVPR25)</td><td>46.31</td><td>65.50</td><td>35.44</td><td>57.91</td><td>45.69</td><td>40.14</td><td>57.91</td><td>31.96</td><td>35.92</td></tr><tr><td>EgoPrune (arXiv25)</td><td>40.85</td><td>62.02</td><td>29.93</td><td>52.40</td><td>52.19</td><td>30.85</td><td>47.79</td><td>32.99</td><td>18.61</td></tr><tr><td>CoverPrune</td><td>52.66</td><td>66.14</td><td>41.27</td><td>60.46</td><td>51.81</td><td>46.34</td><td>73.41</td><td>36.08</td><td>45.79</td></tr><tr><td>CoverPrune-Lite</td><td>52.88</td><td>66.30</td><td>44.45</td><td>60.19</td><td>54.69</td><td>47.75</td><td>68.82</td><td>35.05</td><td>45.79</td></tr></table>

Results on 3D Spatial Reasoning. Table 2 and 3 report results on VSI-Bench [38]. Across both base models, CoverPrune and its Lite variant consistently set the highest average scores under matched token budgets, outperforming all baselines including attention-diversity fused strategies.

At 20% token retention, CoverPrune preserves 92.4% of full-token performance. Our coverage-based paradigm’s advantage amplifies under aggressive pruning: both variants show far more graceful degradation than other SOTA methods, widening the performance gap as retention ratio drops, with substantially higher scores than the strongest competitor at 10% and 5% retention. It confirms that multi-domain coverage is critical for spatial reasoning under extreme compression. We further observe a complementary performance trend across the two variants: CoverPrune-Lite tends to perform better on global layout-focused Room Size tasks, while full CoverPrune shows advantages on fine-grained Relative Direction tasks. This aligns with their respective designs: CoverPrune-Lite’s geometry-aware ordering preserves coarse scale signals, while CoverPrune’s full coverage optimization retains fine-grained relational correspondences.

Table 3: Evaluation on VSI-Bench with VLM-3R [14] as base model, where our methods outperform prior SOTA baselines on the average score and most individual tasks.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Retention Ratio R</td><td rowspan="2">Avg.</td><td colspan="4">Numerical Answer</td><td colspan="4">Multiple-Choice Answer</td></tr><tr><td>Obj. Count</td><td>Abs. Dist.</td><td>Obj. Size</td><td>Room Size</td><td>Rel. Dist.</td><td>Rel. Dir.</td><td>Route Plan</td><td>Appr. Order</td></tr><tr><td>Vanilla</td><td>100%</td><td>60.90</td><td>70.20</td><td>49.40</td><td>69.20</td><td>67.10</td><td>65.40</td><td>80.50</td><td>45.40</td><td>40.10</td></tr><tr><td>VisionZip (CVPR25)</td><td rowspan="6">10%</td><td>50.76</td><td>63.47</td><td>40.14</td><td>62.82</td><td>50.52</td><td>53.38</td><td>61.41</td><td>44.85</td><td>29.45</td></tr><tr><td>Fast VID (NeurIPS25)</td><td>49.98</td><td>63.06</td><td>41.14</td><td>62.72</td><td>44.55</td><td>52.25</td><td>60.03</td><td>44.85</td><td>31.23</td></tr><tr><td>DTC (CVPR25)</td><td>52.64</td><td>64.71</td><td>43.73</td><td>64.56</td><td>59.10</td><td>50.99</td><td>64.89</td><td>44.33</td><td>28.80</td></tr><tr><td>EgoPrune (arXiv25)</td><td>47.49</td><td>62.80</td><td>39.17</td><td>61.49</td><td>48.96</td><td>44.65</td><td>57.97</td><td>44.33</td><td>20.55</td></tr><tr><td>CoverPrune</td><td>54.69</td><td>64.90</td><td>43.30</td><td>65.38</td><td>56.25</td><td>54.65</td><td>71.75</td><td>45.88</td><td>35.44</td></tr><tr><td>CoverPrune-Lite</td><td>54.74</td><td>65.54</td><td>43.20</td><td>64.91</td><td>56.94</td><td>54.51</td><td>70.24</td><td>44.85</td><td>37.70</td></tr><tr><td>VisionZip (CVPR25)</td><td rowspan="6"></td><td>46.30</td><td>62.16</td><td>37.13</td><td>61.41</td><td>40.31</td><td>50.42</td><td>50.10</td><td>42.78</td><td>26.05</td></tr><tr><td>FastVID (NeurIPS25)</td><td>46.13</td><td>62.18</td><td>36.64</td><td>61.20</td><td>37.29</td><td>47.32</td><td>53.10</td><td>43.81</td><td>27.51</td></tr><tr><td>DTC (CVPR25)</td><td>49.83</td><td>63.84</td><td>40.43</td><td>62.79</td><td>55.73</td><td>47.46</td><td>58.64</td><td>44.33</td><td>25.40</td></tr><tr><td>EgoPrune (arXiv25)</td><td>44.22</td><td>61.86</td><td>36.24</td><td>60.19</td><td>42.40</td><td>42.25</td><td>49.26</td><td>43.30</td><td>18.28</td></tr><tr><td>CoverPrune</td><td>51.28</td><td>63.66</td><td>39.80</td><td>64.07</td><td>51.98</td><td>52.39</td><td>61.90</td><td>42.27</td><td>34.14</td></tr><tr><td>CoverPrune-Lite</td><td>51.56</td><td>64.04</td><td>40.53</td><td>64.84</td><td>53.19</td><td>51.41</td><td>60.25</td><td>43.30</td><td>34.95</td></tr></table>

Table 4: Ablation Study of CoverPrune with GS-Reasoner as the base model.  
Table 5: Eficiency analysis of diferent pruning methods.
<table><tr><td>Variant</td><td>∆Overall</td><td>Overall</td><td>Rel. Dist.</td><td>Rel. Dir.</td><td>Appr. Order</td></tr><tr><td>CoverPrune</td><td>0.00</td><td>59.76</td><td>59.44</td><td>84.37</td><td>51.78</td></tr><tr><td>w/o FST capacity</td><td>-0.18</td><td>59.59</td><td>59.01</td><td>82.73</td><td>51.46</td></tr><tr><td>w/o feature cost</td><td>-3.58</td><td>56.18</td><td>52.68</td><td>74.62</td><td>48.54</td></tr><tr><td>w/o geometry cost</td><td>-0.74</td><td>59.02</td><td>59.15</td><td>81.20</td><td>50.65</td></tr><tr><td>w/o time cost</td><td>-0.80</td><td>58.96</td><td>58.45</td><td>80.51</td><td>51.62</td></tr></table>

<table><tr><td>Methods</td><td>Tokens</td><td>Dec. Time (ms/token)</td><td>Prun. Time (s)</td><td>Memory (GB)</td><td>Rel. Acc</td></tr><tr><td>Vanilla</td><td>6272</td><td>45.5</td><td>0.0</td><td>33.5</td><td>100.00</td></tr><tr><td>DTC</td><td>628</td><td>40.7</td><td>3.47</td><td>23.8</td><td>79.85</td></tr><tr><td>CoverPrune</td><td>628</td><td>40.7</td><td>2.53</td><td>23.8</td><td>87.84</td></tr><tr><td>CoverPrune-Lite</td><td>628</td><td>40.7</td><td>0.41</td><td>23.8</td><td>88.01</td></tr></table>

## 5.3 Ablation and Analysis

Component Ablation. Table 4 presents component ablation results on VSI-Bench (R = 20%), covering the FST capacity weighting and each term in the FST cost. Removing the feature cost causes the largest degradation, confirming semantic afinity is critical for fixed-budget coverage. Removing geometry or temporal cost incurs smaller but non-negligible drop, verifying 3D and temporal cues boost relation-centric reasoning. Disabling FST capacity weighting leads to a mild consistent decline, showing suficient transport capacity benefits performance even with fixed retained tokens.

Eficiency. Table 5 compares the eficiency of our method against the SOTA 3D VLM method DTC on VSI-Bench. Dec. Time, Prun. Time, and Rel. Acc denote decoding latency, pruning time overhead, and relative accuracy normalized to Vanilla, respectively. CoverPrune improves relative accuracy while reducing pruning overhead compared with DTC, and CoverPrune-Lite further cuts pruning time by a large margin while achieving the best relative accuracy and the lowest time cost. Since all pruned methods share the same memory reservation and decode latency at this budget, the speedup mainly comes from the lightweight coverage approximation in CoverPrune-Lite, which makes coveragebased selection more practical for real-time use.

## 6 Conclusions and Future Directions

In this work, we redefine 3D VLM visual token pruning as an OT-based coverage maximization problem, departing from the prevailing diversity-driven and attention-based paradigm. We propose CoverPrune, a training-free pruning frame work built on our multi-domain FST transport cost, informativeness-aware FST target capacity, and an eficient SGS solver. We further introduce a lightweight variant, CoverPrune-Lite, which drastically reduces the pruning time overhead with minimal performance loss. Extensive experiments show our method consistently outperforms current SOTA baselines across benchmarks and base models. Looking ahead, we aim to comprehensively explore the full potential of this principled coverage-based pruning paradigm, extending its applicability to general VLMs, to deliver theoretically grounded, robust inference acceleration for universal large model deployment.

Acknowledgements. This work was partly supported by the Special Foundations for the Development of Strategic Emerging Industries of Shenzhen (No. KJ ZD20231023094700001) and the Shenzhen-Tsinghua Special Project for Fundamental & Frontier Research in Artificial Intelligence (No. AI2026018).

## References

1. Alvar, S.R., Singh, G., Akbari, M., Zhang, Y.: Divprune: Diversity-based visual token pruning for large multimodal models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 9392–9401 (2025)

2. Azuma, D., Miyanishi, T., Kurita, S., Kawanabe, M.: Scanqa: 3d question answering for spatial scene understanding. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022)

3. Baek, C., Song, J., Kim, S., Kong, K.: An empirical study of attention and diversity for adaptive visual token pruning in large vision-language models. In: The Fourteenth International Conference on Learning Representations

4. Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., Zhou, J.: Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966 1(2), 3 (2023)

5. Benamou, J., Carlier, G., Cuturi, M., Nenna, L., Peyré, G.: Iterative bregman projections for regularized transportation problems. SIAM J. Sci. Comput. 37(2), A1111–A1138 (2015). https://doi.org/10.1137/141000439, https://doi.org/ 10.1137/141000439

6. Cha, J., Kang, W., Mun, J., Roh, B.: Honeybee: Locality-enhanced projector for multimodal llm. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13817–13827 (2024)

7. Chapel, L., Alaya, M.Z., Gasso, G.: Partial optimal transport with applications on positive-unlabeled learning. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual (2020), https://proceedings.neurips.cc/paper/ 2020/hash/1e6e25d952a0d639b676ee20d0519ee2-Abstract.html

8. Chen, L., Zhao, H., Liu, T., Bai, S., Lin, J., Zhou, C., Chang, B.: An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In: European Conference on Computer Vision. pp. 19–35. Springer (2024)

9. Chen, Y., Qi, Z., Zhang, W., Jin, X., Zhang, L., Liu, P.: Reasoning in space via grounding in the world. arXiv preprint arXiv:2510.13800 (2025)

10. Chen, Z., Gholami, A., Nießner, M., Chang, A.X.: Scan2cap: Context-aware dense captioning in rgb-d scans. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3193–3203 (2021)

11. Cheng, A.C., Fu, Y., Chen, Y., Liu, Z., Li, X., Radhakrishnan, S., Han, S., Lu, Y., Kautz, J., Molchanov, P., et al.: 3d aware region prompted vision language model. arXiv preprint arXiv:2509.13317 (2025)

12. Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al.: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025)

13. Cuturi, M.: Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems 26 (2013)

14. Fan, Z., Zhang, J., Li, R., Zhang, J., Chen, R., Hu, H., Wang, K., Qu, H., Wang, D., Yan, Z., et al.: Vlm-3r: Vision-language models augmented with instruction-aligned 3d reconstruction. arXiv preprint arXiv:2505.20279 (2025)

15. Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al.: Mme: A comprehensive evaluation benchmark for multimodal large language models. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track

16. Guo, Z., Xu, R., Yao, Y., Cui, J., Ni, Z., Ge, C., Chua, T.S., Liu, Z., Huang, G.: Llava-uhd: an lmm perceiving any aspect ratio and high-resolution images. In: European Conference on Computer Vision. pp. 390–406. Springer (2024)

17. Gurumoorthy, K.S., Jawanpuria, P., Mishra, B.: SPOT: a framework for selection of prototypes using optimal transport. In: Dong, Y., Kourtellis, N., Hammer, B., Lozano, J.A. (eds.) Machine Learning and Knowledge Discovery in Databases. Applied Data Science Track - European Conference, ECML PKDD 2021, Bilbao, Spain, September 13-17, 2021, Proceedings, Part IV. Lecture Notes in Computer Science, vol. 12978, pp. 535–551. Springer (2021). https://doi.org/10.1007/978- 3-030-86514-6\_33, https://doi.org/10.1007/978-3-030-86514-6\_33

18. Huang, H.W., Chai, W., Chen, K.M., Yang, C.Y., Hwang, J.N.: Tosa: Token merging with spatial awareness. In: 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 9654–9660. IEEE (2025)

19. Huang, H.W., Chen, F.C., Chai, W., Su, C.C., Xia, L., Jung, S., Yang, C.Y., Hwang, J.N., Sun, M., Kuo, C.H.: Zero-shot 3d question answering via voxelbased dynamic token compression. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19424–19434 (2025)

20. Huang, K., Zou, H., Xi, Y., Wang, B., Xie, Z., Yu, L.: Ivtp: Instruction-guided visual token pruning for large vision-language models. In: European conference on computer vision. pp. 214–230. Springer (2024)

21. Kawano, K., Koide, S., Otaki, K.: Partial wasserstein covering. In: Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications of Artificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022. pp. 7115–7123. AAAI Press (2022).

https://doi.org/10.1609/AAAI.V36I7.20671, https://doi.org/10.1609/ aaai.v36i7.20671

22. Lee, P.Y., Je, J., Park, C., Uy, M.A., Guibas, L., Sung, M.: Perspective-aware reasoning in vision-language models via mental imagery simulation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9241–9251 (2025)

23. Li, J., Li, K., Gao, C., Li, Y., Chen, X.: Egoprune: Eficient token pruning for egomotion video reasoning in embodied agent. arXiv preprint arXiv:2507.15428 (2025)

24. Li, K., Wang, Y., He, Y., Li, Y., Wang, Y., Liu, Y., Wang, Z., Xu, J., Chen, G., Luo, P., et al.: Mvbench: A comprehensive multi-modal video understanding benchmark. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22195–22206 (2024)

25. Li, W., Yuan, Y., Liu, J., Tang, D., Wang, S., Qin, J., Zhu, J., Zhang, L.: Tokenpacker: Eficient visual projector for multimodal llm. International Journal of Computer Vision 133(10), 6794–6812 (2025)

26. Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., Yuan, L.: Video-llava: Learning united visual representation by alignment before projection. In: Proceedings of the 2024 conference on empirical methods in natural language processing. pp. 5971– 5984 (2024)

27. Ling, P., Tan, T., Lin, J., Yang, W.: Sovgaussian: Sparse-view 3d gaussian splatting for open-vocabulary scene understanding. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 5343–5351 (2025)

28. Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al.: Mmbench: Is your multi-modal model an all-around player? In: European conference on computer vision. pp. 216–233. Springer (2024)

29. Ma, X., Yong, S., Zheng, Z., Li, Q., Liang, Y., Zhu, S.C., Huang, S.: Sqa3d: Situated question answering in 3d scenes. In: International Conference on Learning Representations (2023), https://openreview.net/forum?id=IDJx97BC38

30. Peyré, G., Cuturi, M.: Computational optimal transport. Found. Trends Mach. Learn. 11(5-6), 355–607 (2019). https://doi.org/10.1561/2200000073, https: //doi.org/10.1561/2200000073

31. Samet, H.: Foundations of multidimensional and metric data structures. Morgan Kaufmann (2006)

32. Shen, L., Gong, G., He, T., Zhang, Y., Zhao, S., Ding, G., et al.: Fastvid: Dynamic density pruning for fast video large language models. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems

33. Villani, C., et al.: Optimal transport: old and new, vol. 338. Springer (2009)

34. Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al.: Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024)

35. Wang, Q., Yu, Y., Yuan, Y., Mao, R., Zhou, T.: Videorft: Incentivizing video reasoning capability in mllms via reinforced fine-tuning. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems

36. Wu, D., Liu, F., Hung, Y.H., Duan, Y.: Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems

37. Wu, H., Li, D., Chen, B., Li, J.: Longvideobench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems 37, 28828–28857 (2024)

38. Yang, J., Yang, S., Gupta, A.W., Han, R., Fei-Fei, L., Xie, S.: Thinking in space: How multimodal large language models see, remember, and recall spaces. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 10632– 10643 (2025)

39. Yang, S., Chen, Y., Tian, Z., Wang, C., Li, J., Yu, B., Jia, J.: Visionzip: Longer is better but not necessary in vision language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19792– 19802 (2025)

40. Zhang, Y., Fan, C.K., Ma, J., Zheng, W., Huang, T., Cheng, K., Gudovskiy, D.A., Okuno, T., Nakata, Y., Keutzer, K., et al.: Sparsevlm: Visual token sparsification for eficient vision-language model inference. In: International Conference on Machine Learning. pp. 74840–74857. PMLR (2025)

41. Zheng, D., Huang, S., Li, Y., Wang, L.: Learning from videos for 3d world: Enhancing mllms with 3d vision geometry priors. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems

42. Zhu, Z., Wang, X., Li, Y., Zhang, Z., Ma, X., Chen, Y., Jia, B., Liang, W., Yu, Q., Deng, Z., et al.: Move to understand a 3d scene: Bridging visual grounding and exploration for eficient and versatile embodied navigation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 8120–8132 (2025)