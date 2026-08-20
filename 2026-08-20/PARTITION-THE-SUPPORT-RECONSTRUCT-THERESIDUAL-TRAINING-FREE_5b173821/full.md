# PARTITION THE SUPPORT, RECONSTRUCT THERESIDUAL: TRAINING-FREE SPARSE ATTENTION FORVIDEO GENERATION AND WORLD MODELS

Pardis Taghavi, Reza Langari, Gaurav Pandey

Texas A&M University

{ptgh,rlangari,gpandey}@tamu.edu

## ABSTRACT

Training-free block-sparse attention can accelerate video transformers, but rowwise attention concentration does not by itself specify an executable sparse operator. Queries sharing a block route may have poorly overlapping supports, while retained attention mass alone does not determine the post-softmax error from skipped interactions. We show that partition geometry affects both pooled support and the predictability of the remaining residual from the sparse output. We introduce SparsePR, which combines Response-Coupled Partitioning with Probe-Fitted Residual Reconstruction. Sampled-query key responses form paired K/V groups, whose centroids induce query-response coordinates for shared routing. A small set of exact query rows then calibrates a call-specific affine correction from the sparse output within the output subspace observed in the probe residuals. Across four heterogeneous video generation and world models, SparsePR consistently reduces attention-reconstruction error. Ablations show that probe fitting accounts for most of this reduction, while response-coupled partitioning lowers hard-drop error and improves reconstruction under a finite probe budget. SparsePR preserves generation quality at 22.0–26.0% realized executed-pair density while achieving 1.48×–2.61× end-to-end speedups. Project page

## 1 INTRODUCTION

Video generation models (Kong et al., 2024; Wan et al., 2025; Yang et al., 2024) and video world models for physical world prediction (NVIDIA, 2025; 2026) process long spatiotemporal token sequences, making quadratic self-attention a major inference bottleneck at high resolution and long duration. Pretrained video DiTs often exhibit structured, head-dependent attention concentration (Chen et al., 2026a; Sun et al., 2026; Luo et al., 2026), motivating training-free sparse attention for accelerating pretrained models. Prior methods exploit spatiotemporal structure, estimate important blocks online, reorder tokens into executable layouts, or approximate skipped interactions (Zhang et al., 2025b; Xi et al., 2025; Xia et al., 2025; Zhang et al., 2025a; Xu et al., 2025; Yang et al., 2026; Zhou et al., 2026b; Li et al., 2026b). Yet row-wise attention concentration does not determine the support needed when queries share executable block routes, and retained attention mass does not determine the post-softmax error from skipped interactions. These issues are particularly relevant in video world models, where conditioning and prediction regimes can induce different attention structures. An executable sparse operator must therefore account for both shared-route support and the output error from skipped interactions, rather than treating sparsity as a row-wise property alone.

We analyze this challenge by separating the choices that define an executable block-sparse attention computation. The query partition determines which rows share a routing decision, the paired K/V partition determines which tokens are selected together at block granularity, the routing policy selects the cells evaluated exactly, and the skipped-interaction rule specifies how the remaining pairs affect the output. Our analysis reveals three dependencies across the evaluated models. First, perquery concentration does not determine shared-route support. Under the semantic partition, Wan2.2 requires a median support of 6.2% per query to retain 90% of the attention mass, but 22.9% when eight grouped queries share one route. In contrast, Cosmos-Predict2.5 exhibits denser support at the individual-row level. High shared-route density can therefore arise either from weak support agreement within a query group or intrinsic row density. Second, retained attention mass does not determine post-softmax accuracy. Under renormalized hard drop, the residual depends on both the omitted mass and the difference between the outputs induced by the omitted and retained supports. Routes with comparable retained mass can therefore produce different output errors. Cosmos3- Nano exhibits this sensitivity in our sparse runs, motivating direct measurement of post-softmax error rather than inferring output behavior from retained mass or exact-pair density. Third, under matched sparse execution, partition choice changes how much of the remaining residual can be rep resented by an affine function of the sparse output. Partition geometry therefore affects both the support required for shared execution and the structure of the residual left for reconstruction.

![](images/8036f03dde488c5bc604ecba69560075c7ae6eb10cda7d5934cb8e9d00613c29.jpg)  
Figure 1: Qualitative Results. Cosmos2.5-14B mdoel.

Motivated by these observations, we introduce SparsePR, a training-free sparse-attention framework that combines Response-Coupled Partitioning with Probe-Fitted Residual Reconstruction. Response-Coupled Partitioning constructs paired K/V and query partitions through a single asymmetric pass over response coordinates derived from the current attention call. Sampled queries define key-response coordinates for paired K/V grouping, and the resulting key-group centroids define query-response coordinates for shared routing. Probe-Fitted Residual Reconstruction evaluates a small set of query rows exactly and fits a call-specific affine map from the sparse output to the corresponding post-softmax residual. The fitted map estimates residual on unprobed rows. All partitioning and residual calibration are performed online without offline training. Unlike prior methods that couple query and key groups for routing or approximate skipped interactions from block-level approximations (Luo et al., 2026; Zhou et al., 2026b; Li et al., 2026b;a), SparsePR combines currentcall response partitioning with exact row calibrated post-softmax residual reconstruction.

We make three contributions. First, we characterize dependencies that shape executable blocksparse attention across four video models. Per-query concentration does not determine shared-route support, high retained attention mass does not guarantee low post-softmax error, and partition choice changes how much of the residual can be represented by an affine function of the sparse output. Second, we introduce SparsePR, which combines Response-Coupled Partitioning with Probe-Fitted Residual Reconstruction in an online, training-free sparse-attention framework. Third, we evaluate SparsePR across four models using exact-pair accounting that includes probe rows and end-to-end timing that includes online overheads. SparsePR closely matches dense benchmark quality while achieving 1.48× to 2.61× end-to-end speedups.

## 2 RELATED WORK

Sparse Attention for Video Generative Models and Video World Models. Training-free sparse attention for video DiTs reduces inference cost through structured sparse patterns, online block selection, or current-call support estimation. Sliding Tile Attention, SVG, Sparse-vDiT, Radial Attention, LVSA, and VORTA exploit spatial, temporal, layer, or head structures (Zhang et al., 2025b; Xi et al., 2025; Chen et al., 2026a; Li et al., 2026c; Glorian et al., 2026; Sun et al., 2026). AdaSpa, SpargeAttention, and XAttention estimate important blocks from the current attention call (Xia et al., 2025; Zhang et al., 2025a; Xu et al., 2025), while HASTE and ScalingAttention reduce repeated support-estimation cost through temporal mask reuse or offline topology priors (Zheng et al., 2026; Zhou et al., 2026a). VSA and VEDA learn sparse execution through end-to-end training or distillation, while VMoBA supports both sparse training and training-free inference (Zhang et al., 2026; Han et al., 2026; Wu et al., 2025). Exact dense kernels such as FlashAttention are complementary because they preserve dense attention support (Dao et al., 2022). In autoregressive video models, sparse attention has also been used for causal K/V cache compression and approximate retrieval (Samuel et al., 2026). Our setting considers full-sequence attention and studies how partitioning and skipped-interaction treatment determine executable density and post-softmax error.

![](images/b06a78374b4a5056f95490f362a0bb32b27f3317588e7698c55c9ef8682108ec.jpg)  
Figure 2: Overview of SparsePR. Response-Coupled Partitioning builds executable K/V and query groups, and Probe-Fitted Residual Reconstruction uses exact probe rows to correct the sparse output.

Token Partitioning for Executable Block Sparsity. Earlier methods for general Transformers organized tokens through LSH, online k-means, query clustering, asymmetric clustering, and learned permutations (Kitaev et al., 2020; Roy et al., 2021; Vyas et al., 2020; Daras et al., 2020; Tay et al., 2020). For video DiTs, SVG2 clusters query and key activations and permutes the resulting groups into contiguous layouts (Yang et al., 2026). DraftAttention and DFSAttn use coarse attention guidance or hierarchical ordering to expose executable block structure (Shen et al., 2025; Hu et al., 2026). AdaCluster uses different similarity criteria for queries and keys, while SVOO iteratively alternates query-aware key clustering and key-aware query clustering (Tan et al., 2026; Luo et al., 2026). Response-Coupled Partitioning instead uses a single asymmetric coupling pass. Key responses under sampled queries define paired K/V groups, nd the resulting key-group centroids induce query coordinates without alternating partition refinement.

Attention Approximation and Residual Reconstruction. Attention approximations use random features, landmarks, or sparse and low-rank approximations (Choromanski et al., 2020; Xiong et al., 2021; Chen et al., 2021). Sparse attention methods for visual generation instead approximate or recover contributions from skipped interactions. Re-ttention reuses softmax statistics across denoising steps, Rectified SpaAttn applies pooled corrections, PSA retains multiresolution K/V representations, and SVG-EAR compensates skipped semantic blocks using K/V centroids and error-aware routing (Chen et al., 2026b; Liu et al., 2025; Li et al., 2025; Zhou et al., 2026b). PISA and Sol-Attn approximate omitted contributions using blockwise Taylor expansions and routing proxy scores, respectively (Li et al., 2026b;a). SparsePR instead uses a small set of exact exact query rows to fit a map from the sparse output to the residual, without temporal reuse or offline training.

## 3 SPARSE ATTENTION FORMULATION AND OBSERVATIONS

Block-sparse attention makes sparsity executable by grouping tokens and evaluating selected query– key group pairs. The resulting operator is determined by partition construction, route selection, and the treatment of skipped interactions. These choices introduce distinct sources of approximation error. We formalize this operator and identify three structural dependencies that motivate SparsePR.

Dense and block-sparse attention. For one attention head, let $Q \in \mathbb { R } ^ { N _ { q } \times d } , K \in \mathbb { R } ^ { N _ { k } \times d }$ , and $V \in \mathbb { R } ^ { N _ { k } \times d _ { v } }$ . Dense attention computes

$$
A = \mathrm { s o f t m a x } \bigg ( \frac { Q K ^ { \top } } { \sqrt { d } } \bigg ) , \qquad O ^ { \mathrm { d e n s e } } = A V .\tag{1}
$$

A block-sparse operator partitions queries into $\mathcal { G } ^ { Q } ~ = ~ \{ G _ { a } ^ { Q } \} _ { a = 1 } ^ { C _ { q } }$ and K/V tokens into $\begin{array} { r l } { \mathcal { G } ^ { K V } } & { { } = } \end{array}$ $\{ G _ { b } ^ { K V } \} _ { b = 1 } ^ { C _ { k } }$ . Each pair $G _ { a } ^ { Q } \times G _ { b } ^ { K V }$ defines a cell. Let $z _ { a b } \in \{ 0 , 1 \}$ indicate whether the cell is evaluated exactly. To account for unequal group sizes, the routing density over query–key pairs is

$$
\rho _ { \mathrm { r o u t e } } = \frac { 1 } { N _ { q } N _ { k } } \sum _ { a = 1 } ^ { C _ { q } } \sum _ { b = 1 } ^ { C _ { k } } z _ { a b } | G _ { a } ^ { Q } | | G _ { b } ^ { K V } | .\tag{2}
$$

Under renormalized hard drop, every query $i \in G _ { a } ^ { Q }$ uses the same retained K/V groups. Let $K _ { a }$ and $V _ { a }$ collect the key and value rows from groups $G _ { b } ^ { \breve { K } V }$ with $z _ { a b } = 1$ . The sparse output is

$$
O _ { i } ^ { \mathrm { s p } } = \mathrm { s o f t m a x } \bigg ( \frac { q _ { i } K _ { a } ^ { \top } } { \sqrt { d } } \bigg ) V _ { a } .\tag{3}
$$

Collecting these rows gives $O ^ { \mathrm { s p } }$ , with post-softmax residual

$$
R = O ^ { \mathrm { d e n s e } } - O ^ { \mathrm { s p } } .
$$

Here, $\rho _ { \mathrm { r o u t e } }$ counts pairs evaluated by sparse routing. Total executed-pair density additionally includes exact probe rows, as defined in Section 4.3.

## 3.1 STRUCTURAL OBSERVATIONS ON SPARSE ATTENTION

We use semantic clustering as a reference partition, applying k-means separately to the current query and key activations, with each value token inheriting the assignment of its paired key (Yang et al., 2026). We use this partition to examine support expansion under shared routing, post-softmax error under renormalized hard drop, and the affine representability of the residual from the sparse output.

O1: Per-Query Sparsity Does Not Determine Shared-Route Support. For a target retained mass $\alpha \in ( 0 , 1 ]$ , let $D _ { i } ( \alpha )$ denote the smallest set of highest-weight key indices whose cumulative dense attention mass for query i is at least α. For m queries $i _ { 1 } , \ldots , i _ { m } \stackrel { \textstyle \cdot } { \in } G _ { a } ^ { Q }$ , define

$$
\tau _ { i } ( \alpha ) = \frac { | D _ { i } ( \alpha ) | } { N _ { k } } , \qquad \gamma _ { a } ^ { ( m ) } ( \alpha ) = \frac { | \bigcup _ { r = 1 } ^ { m } D _ { i _ { r } } ( \alpha ) | } { N _ { k } } .
$$

Here, $\tau _ { i } ( \alpha )$ measures row-wise support density, while $\gamma _ { a } ^ { ( m ) } ( \alpha )$ measures pooled support density. Because the union is formed over individual key indices, this diagnostic isolates query-side support disagreement before further expansion from selecting K/V groups. Under the semantic partition, Fig. 3(a) compares per-query and pooled support at $\alpha = 0 . 9$ . Vertical displacement above the diagonal measures support expansion from limited overlap among key supports. Wan2.2 is the clearest example: its median support increases from 6.2% per query to 22.9% after pooling. Cosmos-Predict2.5 is broad at the row level, with median support increasing from 56.5% to 77.7% after pooling. HunyuanVideo remains sparse at both levels, while Cosmos3-Nano occupies an intermediate regime. Pooled support therefore depends on both row-wise concentration and support overlap within query groups, showing why per-query sparsity alone does not characterize shared routing.

O2: Retained Attention Mass Alone Does Not Determine Output Error. For $i \in G _ { a } ^ { Q }$ , let $U _ { a }$ denote the omitted key indices and $\textstyle p _ { U , i } = \sum _ { i \in U _ { a } } A _ { i j }$ be omitted mass. Let $O _ { U , i }$ denote the output obtained by renormalizing the dense attention weights over $\mathcal { U } _ { a }$ . The residual then satisfies:

$$
R _ { i } = { \cal O } _ { i } ^ { \mathrm { d e n s e } } - { \cal O } _ { i } ^ { \mathrm { s p } } = p _ { U , i } \left( { \cal O } _ { U , i } - { \cal O } _ { i } ^ { \mathrm { s p } } \right) .\tag{4}
$$

Thus, high retained mass makes $p _ { U , i }$ small, but does not constrain $\lVert O _ { U , i } - O _ { i } ^ { \mathrm { s p } } \rVert _ { 2 }$ . Queries with similar retained mass can therefore have different residual norms. Figure 3(b) shows that error generally decreases as retained mass increases, but model-dependent variation remains at comparable mass levels. High retained attention mass alone is insufficient to characterize post-softmax error.

![](images/0d2182a577821b84764ff104e379e4445934dda42a2b177ad9bacc6dc8def2cf.jpg)

![](images/02debf59ef705c4dccd27cbf4c89a3af3b5afe4a137c9dda72418280c52d7db5.jpg)  
Figure 3: Structural observations under the semantic reference partition. (a) Per-query versus eightquery pooled support at $9 0 \%$ attention mass. Vertical displacement above the diagonal indicates support expansion under pooling. (b) p99 normalized attention-output error versus realized retained attention mass. Comparable retained mass can produce different post-softmax errors across models.

O3: Partition Geometry Changes Residual Representability Residual magnitude alone does not indicate how well the hard-drop residual can be represented by an affine function of the sparse output. Let g index a choice of query and K/V partitions, with sparse output $O _ { g } ^ { \mathrm { s p } }$ and corresponding post-softmax residual $R _ { g }$ . Define the affine feature matrix

$$
X _ { g } = [ { \bf 1 } , { \cal O } _ { g } ^ { \mathrm { s p } } ] ,
$$

and let $P _ { X _ { g } }$ denote the orthogonal projector onto co $( X _ { g } )$ . The residual decomposes as

$$
R _ { g } = \underbrace { P _ { X _ { g } } R _ { g } } _ { \mathrm { a f f i n e - e x p l a i n a b l e ~ c o m p o n e n t } } + \underbrace { ( I - P _ { X _ { g } } ) R _ { g } } _ { \mathrm { a f f i n e - o r t h o g o n a l ~ r e s i d u a l } } .\tag{5}
$$

For $X _ { g }$ , the projected term is the best least-squares approximation of $R _ { g }$ by an affine function of $O _ { g } ^ { \mathrm { s p } }$ while the orthogonal term lies outside this feature space. Computing this decomposition requires the full dense residual, and it serves as an oracle diagnostic rather than part of sparse execution.

We compare the semantic and response-coupled partitions from Section 4.1 under matched group counts, routing, renormalized hard drop, and realized pair density. We report the fraction of residual energy captured by the affine feature space, $\| P _ { X _ { g } } ^ { \bullet } R _ { g } \| _ { F } ^ { 2 } / \| \mathbf { \bar { \phi } } _ { R _ { g } } \| _ { F } ^ { 2 }$ , and the affine-orthogonal energy, $\| ( I - P _ { X _ { g } } ) R _ { g } \| _ { F } ^ { 2 }$ . Across the four models, response-coupled partitioning increases the affine-explainable fraction by 3.2 to 14.9 percentage points and reduces affine-orthogonal energy to $0 . 2 8 5 { \times } { - } 0 . 6 5 3 { \times }$ the semantic-partition baseline. The higher explainable fraction indicates greater relative alignment with the affine feature space, while the lower orthogonal energy shows that the absolute residual outside this space also decreases. Thus, at matched sparse execution, partition geometry changes both affine representability and the residual energy left outside the feature space.

## 4 METHOD

Motivated by the observations, SparsePR combines two online, training-free components. Response-Coupled Partitioning constructs executable query and K/V groups from current-call response geometry, while Probe-Fitted Residual Reconstruction uses a set of query rows to calibrate a call-specific affine correction for the sparse output. Together, they address the shared-support and residual-error effects identified in Section 3. We then describe sparse execution and exact-pair accounting.

## 4.1 RESPONSE-COUPLED PARTITIONING

An executable partition should group queries with routing behavior and paired K/V indices whose keys induce similar response profiles. Proximity in the activation space does not guarantee either property. Nearby queries can require different K/V supports, while nearby keys can respond differently under the same queries. We construct partitions in low-dimensional response coordinates derived from the current attention call. The construction is coupled and asymmetric: sampled queries define the coordinates for K/V grouping, and the resulting groups define the query coordinates.

Key-response coordinates for paired K/V groups. We group paired K/V tokens according to their key responses under sampled queries. Let $\bar { Q _ { s } } \doteq \mathbb { R } ^ { M _ { s } \times \tilde { d } }$ contain $M _ { s }$ query rows sampled from the current attention call. We define the key-response metric

$$
M _ { K } = \frac { 1 } { M _ { s } } { Q } _ { s } ^ { \top } { Q } _ { s } .
$$

For any two keys $k _ { j }$ and $k _ { \ell }$

$$
\big ( k _ { j } - k _ { \ell } \big ) ^ { \top } M _ { K } \big ( k _ { j } - k _ { \ell } \big ) = \frac { 1 } { M _ { s } } \left\| Q _ { s } \big ( k _ { j } - k _ { \ell } \big ) \right\| _ { 2 } ^ { 2 } .
$$

Thus, keys are close under $M _ { K }$ when they have similar response profiles across sampled queries. These responses correspond to pre-softmax logit profiles up to the $1 / \sqrt { d }$ attention scale. Let $F _ { K } \in$ $\mathbb { R } ^ { d \times r _ { K } }$ contain the leading $r _ { K }$ eigenvectors of $M _ { K }$ , scaled by the square root of their eigenvalues after the retained eigenvalues are normalized to unit mean. We represent key $k _ { j }$ by

$$
\begin{array} { r } { \phi _ { K } ( k _ { j } ) = F _ { K } ^ { \top } k _ { j } \in \mathbb { R } ^ { r _ { K } } . } \end{array}\tag{6}
$$

Euclidean distance in this coordinate space corresponds to a normalized rank- $- r _ { K }$ approximation of the $M _ { K }$ . Applying k-means to $\phi _ { K } ( k _ { j } )$ yields the paired K/V groups $\mathcal { G } ^ { K V }$ , with each value inheriting its key’s assignment.

Query response coordinates from K/V groups. The resulting K/V groups define the response space used to compare queries. We compute each group’s centroid in the original key space, center the centroids by their uniform mean and stack them as the rows of $\widetilde { K }$ . We then form the queryresponse metric $M _ { Q } = \widetilde { K } ^ { \top } \widetilde { K } / ( C _ { k } d )$ . Under this metric, two queries are close when they produce similar relative pre-softmax logit profiles over the nonwmpty K/V groups. Applying the same eigenvalue-weighted construction to the leading $r _ { Q }$ eigendirections of $M _ { Q }$ gives $\bar { F _ { Q } } \doteq \bar { \mathbb { R } ^ { d \times r _ { Q } } }$ . We define the query-response coordinate as

$$
\phi _ { Q } ( q _ { i } ) = \frac { F _ { Q } ^ { \top } q _ { i } } { \operatorname* { m a x } \left( \sqrt { \| F _ { Q } ^ { \top } q _ { i } \| _ { 2 } ^ { 2 } / r _ { Q } } , \epsilon \right) } \in \mathbb { R } ^ { r _ { Q } } ,\tag{7}
$$

where ϵ is a numerical stabilizer. The row-wise RMS normalization makes k-means primarily compare response direction rather than scale. Clustering $\phi _ { Q } ( q _ { i } )$ yields the query partition $\mathcal { G } ^ { Q }$ used for shared routing. Together with the preceding K/V stage, this forms a coupled, single-pass construction without alternating refinement between the K/V and query partitions.

## 4.2 PROBE-FITTED RESIDUAL RECONSTRUCTION

For each attention call, Probe-Fitted Residual Reconstruction uses $M \ll N _ { q }$ exact query rows to calibrate a residual estimator. Motivated by the affine diagnostic in Section 3.1, we model the residual as an affine function of the sparse output:

$$
x _ { i } = O _ { i } ^ { \mathrm { s p } } \in \mathbb R ^ { d _ { v } } , \qquad R _ { i } \approx b + x _ { i } B .
$$

Exact probe calibration. We select M probe indices $\mathcal { P } \subseteq \{ 1 , \ldots , N _ { q } \}$ , stratified across the query groups. Exact attention on each probe yields $R _ { p } = O _ { p } ^ { \mathrm { d e n s e } } - O _ { p } ^ { \mathrm { s p } }$ . Let $\mathbf { \bar { \boldsymbol { X } } } _ { \mathcal { P } } , \boldsymbol { R _ { \mathcal { P } } } \in \mathbb { R } ^ { M \times d _ { v } }$ stack the probe features and residuals, and let $\breve { W }$ contain query-group coverage weights. We standardize $X _ { \mathcal { P } }$ and center $R _ { \mathcal { P } }$ using the weighted probe statistics, yielding $\bar { X } _ { \mathcal P }$ and $\bar { R } _ { \mathcal { P } }$ . We estimate

$$
B _ { \lambda } = \arg \operatorname* { m i n } _ { { \cal B } } \left\| { \cal W } ^ { 1 / 2 } \left( \bar { R } _ { \mathcal { P } } - \bar { X } _ { \mathcal { P } } { \cal B } \right) \right\| _ { F } ^ { 2 } + \lambda \| { \cal B } \| _ { F } ^ { 2 } .\tag{8}
$$

Coverage weights account for the query population represented by each probe, while ridge regularization stabilizes the fit from the limited probe set. Probe selection, weighting, and standardization are further detailed in the supplementary material.

Output subspace from probe residuals. With M probes, the affine map may extrapolate into output directions poorly constrained by the observed residuals. We therefore let $\bar { \Psi _ { r } } \in \mathbb { R } ^ { d _ { v } ^ { * } \times r }$ contain the leading r right singular vectors of the weighted, centered exact probe-residual matrix $W ^ { 1 / 2 } \bar { R } _ { \mathcal { P } }$ For an unprobed query, let $\bar { x } _ { i }$ denote $x _ { i }$ standardized using the probe statistics. We predict

$$
\widehat { R } _ { i } = \mu _ { R } + \bar { x } _ { i } B _ { \lambda } \Psi _ { r } \Psi _ { r } ^ { \top } .\tag{9}
$$

Because the basis is computed from centered residuals, $\mu _ { R }$ is added separately, while $\Psi _ { r } \Psi _ { r } ^ { \top }$ restricts the feature-dependent term to output directions observed in the exact probe residuals. This restriction does not assume that the complete residual matrix is globally low rank. The final output is

$$
\begin{array} { r } { \widehat { O } _ { i } = \left\{ \begin{array} { l l } { O _ { i } ^ { \mathrm { d e n s e } } , } & { i \in \mathcal { P } , } \\ { O _ { i } ^ { \mathrm { s p } } + \widehat { R } _ { i } , } & { i \notin \mathcal { P } . } \end{array} \right. } \end{array}\tag{10}
$$

## 4.3 SPARSE EXECUTION AND EXACT-PAIR ACCOUNTING

We permute Q, K, and V into contiguous group-major layouts, with each value token following its paired key. Cell selection uses a query–key pair budget and charges each cell its true cost, $| G _ { a } ^ { \dot { Q } } | | G _ { b } ^ { K V } |$ . The selected cells and ragged group offsets are passed to FlashInfer’s variable-block sparse-attention primitive Ye et al. (2025), which evaluates the selected interactions exactly and renormalizes each query row over its retained K/V groups, as in Eq. equation 3. The output is then restored to the original query order. For grouped-query attention, K/V partition and routing metadata are shared across associated query heads. Exact probe rows are evaluated against all keys afte sparse attention. Because these evaluations are additional to the routed sparse operator, the total executed-pair density is $\rho _ { \mathrm { e x e c } } = \rho _ { \mathrm { r o u t e } } + M / N _ { q }$ . The router reserves the probe cost before selecting sparse cells. Because complete query–K/V cells are indivisible, realized density may differ slightly from the target, so we report the realized value for each configuration. Executed pair density counts exact query–key interactions, whereas reported latency includes partition construction, routing, permutation, sparse attention, probe evaluation, residual fitting, correction, and output restoration.

## 5 EXPERIMENTS

Implementation details. All experiments use BF16 on a single NVIDIA H100 GPU. Response-Coupled Partitioning uses feature ranks $r _ { K } = 4 8$ and $r _ { Q } = 6 4 $ . Probe-Fitted Residual Reconstruction uses $M = 6 4$ exact rows per query head, output rank $r = 1 6 ,$ and ridge coefficient $\lambda = 0 . 1$ . The total executed-pair target, including probe rows, is 22% for HunyuanVideo, Wan2.2, and Cosmos-Predict2.5, and 26% for Cosmos3-Nano. Other hyperparameters are shared across models. Timing includes the online costs described in Section 4.3. Additional implementation details and timing protocols are provided in the supplementary material.

Models and benchmarks. We evaluate HunyuanVideo-13B on VBench for text-to-video generation, Wan2.2-I2V-A14B on VBench++ Huang et al. (2025) for image-to-video generation, and Cosmos-Predict2.5-14B and Cosmos3-Nano-16B for physical-world generation and prediction. For the Cosmos models, we report VBench++ visual-quality metrics and PBench NVIDIA (2025) physical-world metrics under each model’s default conditioning protocol.

Evaluation protocol. Dense and sparse runs use the same conditioning inputs, preprocessing, random seeds, sampling schedule, inference steps, guidance settings, resolution, and frame count. Reproduced baselines use the same samples, and results reported by prior work are marked with †.

Metrics. We report four metric groups. Dense-reference fidelity uses PSNR, SSIM, and LPIPS over temporally aligned frames. Generation quality uses ImgQual and subject consistency from VBench or VBench++. Physical-world performance uses the PBench Quality score. Efficiency uses realized executed-pair density, end-to-end FLOPs, and end-to-end generation speedup.

## 5.1 QUALITY–EFFICIENCY TRADE-OFFS

Table 1 compares dense-reference fidelity, benchmark quality, and efficiency across four models.

Qualitative comparison. Figure 1 compares matched dense and SparsePR outputs. Additional comparisons are provided in the supplementary material.

Table 1: Quality–efficiency comparison across content-generation and physical video world models.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="3">Dense-Reference Fidelity</td><td colspan="2">VBench / VBench++</td><td rowspan="2">PBench</td><td colspan="3">Efficiency</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>ImgQual↑</td><td>SubCons↑ Quality↑</td><td>Density↓</td><td>PFLOPs↓</td><td>E2E Speedup↑</td></tr><tr><td colspan="2">Content-generation video models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Dense SpargeAttn† Zhang et al. (2025a)</td><td>24.589</td><td>0.796</td><td>0.232</td><td>0.850</td><td>0.976 0.908</td><td></td><td>100% 40.09%</td><td>612.38 389.76</td><td>1.00× 1.38×</td></tr><tr><td>HunyuanVideo</td><td>SVG2† Yang et al. (2026)</td><td>30.452</td><td>0.910</td><td>0.117</td><td>0.852</td><td>0.927</td><td></td><td>25.45%</td><td>299.02</td><td>2.30×</td></tr><tr><td>13B, 720p Text-to-Video</td><td></td><td>24.879</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>SVOO† Luo et al. (2026) SVG-EAR† Zhou et al. (2026b)</td><td></td><td>0.843</td><td>0.224</td><td>0.6793</td><td>0.9799</td><td></td><td></td><td></td><td>2.17×</td></tr><tr><td></td><td>SparsePR</td><td>31.043 31.844</td><td>0.928 0.932</td><td>0.092 0.087</td><td>0.845 0.850</td><td>0.903 0.976</td><td></td><td>22.17% 21.92%</td><td>281.86 255.95</td><td>1.93×</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>2.61×</td></tr><tr><td>Wan2.2-I2V-A14B</td><td>Dense</td><td></td><td></td><td></td><td>0.689</td><td>0.974</td><td></td><td>100%</td><td>658.46</td><td>1.00×</td></tr><tr><td></td><td>SpargeAttn† Zhang et al. (2025a)</td><td>27.140</td><td>0.883</td><td>0.116</td><td>0.680</td><td>0.958</td><td></td><td>30.15%</td><td>396.83</td><td>1.58×</td></tr><tr><td>14B active, 720p</td><td>SVG2† Yang et al. (2026)</td><td>26.562</td><td>0.861</td><td>0.138</td><td>0.668</td><td>0.959</td><td></td><td>31.28%</td><td>393.95</td><td>1.59×</td></tr><tr><td>Image-to-Video</td><td>SVOO† Luo et al. (2026)</td><td>29.678</td><td>0.913</td><td>0.095</td><td>0.7337</td><td>0.9731</td><td></td><td></td><td></td><td>1.61×</td></tr><tr><td></td><td>SVG-EAR† Zhou et al. (2026b)</td><td>29.759</td><td>0.918</td><td>0.093</td><td>0.680</td><td>0.959</td><td></td><td>23.64%</td><td>378.88</td><td>1.61×</td></tr><tr><td></td><td>SparsePR</td><td>30.658</td><td>0.907</td><td>0.044</td><td>0.687</td><td>0.973</td><td></td><td>21.97%</td><td>328.70</td><td>1.80×</td></tr><tr><td colspan="2">Physical video world models Dense</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Cosmos-Predict2.5</td><td></td><td></td><td></td><td>一</td><td>0.714</td><td>0.976</td><td>77.76</td><td>100%</td><td>526.87</td><td>1.00×</td></tr><tr><td>14B, 720p</td><td>SVG2 Yang et al. (2026) SVOO Luo et al. (2026)</td><td>20.075</td><td>0.624</td><td>0.330</td><td>0.678</td><td>0.896</td><td>76.14</td><td>28.81%</td><td>286.51</td><td>1.24×</td></tr><tr><td>Image-to-World</td><td></td><td>22.066</td><td>0.685</td><td>0.289</td><td>0.701</td><td>0.909</td><td>76.03</td><td>37.63%</td><td>315.38</td><td>1.03×</td></tr><tr><td></td><td>SVG-EAR Zhou et al. (2026b) SparsePR</td><td>25.549 26.328</td><td>0.908 0.942</td><td>0.062 0.068</td><td>0.710 0.714</td><td>0.976</td><td>77.78</td><td>29.75%</td><td>289.69</td><td>1.10×</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.976</td><td>77.75</td><td>22.14%</td><td>253.61</td><td>1.51×</td></tr><tr><td>Cosmos3-Nano</td><td>Dense</td><td></td><td></td><td></td><td>0.700</td><td>0.950</td><td>77.31</td><td>100.00%</td><td>90.01</td><td>1.00×</td></tr><tr><td>16B, 720p</td><td>SVG2 Yang et al. (2026)</td><td>22.458</td><td>0.735</td><td>0.216</td><td>0.677</td><td>0.915</td><td>75.03</td><td>37.29%</td><td>57.69</td><td>1.16×</td></tr><tr><td>Image-to-World</td><td>SVOO Luo et al. (2026)</td><td>16.642</td><td>0.573</td><td>0.381</td><td>0.707</td><td>0.962</td><td>77.59</td><td>67.32%</td><td>69.51</td><td>1.02×</td></tr><tr><td></td><td>SVG-EAR Zhou et al. (2026b)</td><td>21.167</td><td>0.709</td><td>0.261</td><td>0.658</td><td>0.872</td><td>72.85</td><td>37.18%</td><td>57.64</td><td>1.10×</td></tr><tr><td></td><td>SparsePR</td><td>24.417</td><td>0.801</td><td>0.176</td><td>0.699</td><td>0.949</td><td>77.30</td><td>25.96%</td><td>43.22</td><td>1.48×</td></tr></table>

<sup>†</sup>Reported by prior work. Reproduced under matched hardware, sequence shape, and timing protocols.

## 5.2 ABLATIONS

Partition and reconstruction. Table 2 compares partition variants and probe-fitted reconstruction. The key-response K/V variant changes only the K/V partition, while the response-coupled variant also derives the query partition from K/V centroids. For repair comparisons, probe indices and estimator settings are fixed, and errors are reported only on unprobed rows. Response-coupled partitioning reduces both mean and p99 error relative to semantic partitioning on all four models. Probe fitting provides the larger reduction, and benefits more from the response-coupled partition. Figure 4(a) summarizes the fixed-density comparison.

Density sweep. Table 3 evaluates Cosmos3-Nano across five executed-pair budgets. SparsePR achieves lowest mean and p99 error at every density, showing gains across operating points.

Probe and estimator design. With response-coupled partitioning fixed, Table 4 compares probe selection and output-subspace regularization. Query-group-stratified probes achieve lowest mean and p99 error. Probe-residual subspace projection provides a small consistent gain over ridge alone.

## 5.3 RUNTIME ANALYSIS

Figure 4(b) decomposes estimated full-generation latency on Wan2.2. SparsePR reduces latency from 1650 s to 917 s, corresponding to a 1.80× speedup, compared with 1038 s and 1.59× for SVG2. Probe-Fitted Residual Reconstruction accounts for only 1.1% of latency. Thus, the primary fidelity gain from probe fitting adds little end-to-end overhead.

Table 2: Partition and Reconstruction ablation at matched routed-pair density. Entries report mean/p99 normalized attention-output error; lower is better.
<table><tr><td>Configuration</td><td>HunyuanVideo</td><td>Wan2.2</td><td>Cosmos- Predict2.5</td><td>Cosmos3-Nano</td></tr><tr><td>Partition construction, repair disabled</td><td></td><td></td><td></td><td></td></tr><tr><td>Semantic partition</td><td>0.0887 / 0.7136</td><td>0.1634 / 1.7338</td><td>0.7903 / 7.5318</td><td>0.3590 / 3.3557</td></tr><tr><td>Key-response K/V partition</td><td>0.0851 / 0.8121</td><td>0.1560 / 1.6591</td><td>0.7686 / 7.2700</td><td>0.3409 / 3.1738</td></tr><tr><td>Response-coupled partition</td><td>0.0736 / 0.6967</td><td>0.1489 / 1.6479</td><td>0.7617 / 7.2271</td><td>0.3315 /3.1502</td></tr><tr><td>Partition-repair interaction</td><td></td><td></td><td></td><td></td></tr><tr><td>Semantic partition + probe repair</td><td>0.0527 / 0.3562</td><td>0.1041 / 0.8186</td><td>0.2622 / 0.8260</td><td>0.1720 / 0.8648</td></tr><tr><td>SparsePR</td><td>0.0330 / 0.2285</td><td></td><td>0.0707 / 0.4305 0.0954/ 0.5769</td><td>0.0822 / 0.4951</td></tr></table>

Table 3: Cosmos3-Nano sensitivity to total executed-pair density $\rho .$ Entries report mean/p99.
<table><tr><td>Method</td><td> $\rho = 1 2 \%$ </td><td> $\rho = 1 7 \%$ </td><td> $\rho = 2 2 \%$ </td><td> $\rho = 2 8 \%$ </td><td> $\rho = 3 5 \%$ </td></tr><tr><td>Semantic + hard drop</td><td></td><td></td><td></td><td></td><td>0.5469 / 4.99250.4347 / 4.00980.3590 / 3.35570.2929 / 2.77800.2354 / 2.2662</td></tr><tr><td>Response-coupled + hard drop</td><td></td><td></td><td></td><td></td><td>0.5061 /4.63450.4016 /3.74860.3315 /3.15020.2701 /2.61680.2170 /2.1421</td></tr><tr><td>Semantic + probe repair</td><td></td><td></td><td></td><td></td><td>0.1484 / 0.83770.1140 / 0.66800.1720 / 0.86480.0764 / 0.48350.0627 / 0.4183</td></tr><tr><td>SparsePR</td><td></td><td></td><td></td><td></td><td>0.1274 / 0.7245 0.0990 / 0.5771 0.0822 / 0.4951 0.0681 / 0.4222 0.0567 / 0.3669</td></tr></table>

Table 4: Probe-selection and estimator ablations with the response partitiotn, $M = 6 4 , r = 1 6 .$
<table><tr><td>Configuration</td><td>HunyuanVideo</td><td>Wan2.2</td><td>Cosmos- Predict2.5</td><td>Cosmos3-Nano</td></tr><tr><td>Probe selection</td><td></td><td></td><td></td><td></td></tr><tr><td>Random probes</td><td>0.0382 / 0.2361</td><td>0.0775 / 0.4533</td><td>0.0977 / 0.5873</td><td>0.0852 / 0.5024</td></tr><tr><td>Uniform spatiotemporal probes</td><td>0.0402 / 0.2486</td><td>0.0820 / 0.4316</td><td>0.1068/0.6283</td><td>0.0898 / 0.5359</td></tr><tr><td>Query-group-stratified probes</td><td>0.0329 / 0.2271</td><td>0.0708 / 0.4307</td><td>0.0953 / 0.5768</td><td>0.0818 / 0.4907</td></tr><tr><td>Estimator design with query-group-stratified probes</td><td></td><td></td><td></td><td></td></tr><tr><td>Ridge only</td><td>0.0340 / 0.2332</td><td>0.0734 / 0.4530</td><td>0.0995 / 0.5882</td><td>0.0837 / 0.5041</td></tr><tr><td>Ridge + probe-residual subspace</td><td>0.0329 / 0.2271</td><td></td><td>0.0708 / 0.4307 0.0953 / 0.5768</td><td>0.0818 / 0.4907</td></tr></table>

![](images/8a3a6465c8f53f65b4105769b08cdec1e996b397ffae82b4ec7d8bb474999bc3.jpg)

![](images/b3ed2ea2c67a35dfdf6622d6870b786b1fb0580213356d8dcc73d8612c5d8d7e.jpg)  
Figure 4: Partitioning and runtime analysis. (a) Error reduction from response-coupled vs. semantic partitioning at 22% density. (b) Wan2.2 full generation latency for Dense, SVG2, and SparsePR.

## ACKNOWLEDGMENTS

This work used ACES at Texas A&M University through allocation CIS250376 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by U.S. National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296 Boerner et al. (2023).

## 6 CONCLUSION

We studied training-free block-sparse attention as an executable operator in which partitioning shapes both shared support and the residual left by skipped interactions. SparsePR addresses these effects with Response-Coupled Partitioning and Probe-Fitted Residual Reconstruction, combining response-based grouping with call-specific residual correction. Across four video generation and world models, SparsePR reduces attention-reconstruction error, with probe fitting providing the largest fidelity gain and response-coupled partitioning further improving reconstruction. SparsePR closely matches dense benchmark quality while achieving 1.48× to 2.61× end-to-end speedups at 22% to 26% realized executed-pair density. These results show that effective sparse execution depends on both partition geometry and reconstruction of skipped interactions.

## REFERENCES

Timothy J. Boerner, Stephen Deems, Thomas R. Furlani, Shelley L. Knuth, and John Towns. AC-CESS: Advancing innovation: NSF’s advanced cyberinfrastructure coordination ecosystem: Services & support. In Practice and Experience in Advanced Research Computing (PEARC ’23), pp. 173–176, 2023. doi: 10.1145/3569951.3597559.

Beidi Chen, Tri Dao, Eric Winsor, Zhao Song, Atri Rudra, and Christopher Re. Scatterbrain: Uni-´ fying sparse and low-rank attention. Advances in Neural Information Processing Systems, 34: 17413–17426, 2021.

Pengtao Chen, Xianfang Zeng, Maosen Zhao, Mingzhu Shen, Wei Cheng, Gang Yu, and Tao Chen. Sparse-vdit: Unleashing the power of sparse attention to accelerate video diffusion transformers. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 2957–2965, 2026a.

Ruichen Chen, Keith Mills, Liyao Jiang, Chao Gao, and Di Niu. Re-ttention: Ultra sparse visual generation via attention statistical reshape. Advances in Neural Information Processing Systems, 38:58029–58055, 2026b.

Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. arXiv preprint arXiv:2009.14794, 2020.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Re. Flashattention: Fast and memory- ´ efficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022.

Giannis Daras, Nikita Kitaev, Augustus Odena, and Alexandros G Dimakis. Smyrf-efficient attention using asymmetric clustering. Advances in Neural Information Processing Systems, 33: 6476–6489, 2020.

Gael Glorian, Ioannis Lamprou, Zhen Zhang, Yujie Yuan, and Hongsheng Liu. Lvsa: Training-free sparse attention for long video diffusion. arXiv preprint arXiv:2605.31057, 2026.

Shihao Han, Hao Yang, Xinting Hu, Xiaofeng Mei, Yi Jiang, and Xiaojuan Qi. Veda: Scalable video diffusion via distilled sparse attention. arXiv preprint arXiv:2605.30325, 2026.

Jie Hu, Zixiang Gao, Yutong He, and Kun Yuan. Dfsattn: Dynamic fine-grained sparse attention for efficient video generation. arXiv preprint arXiv:2605.23445, 2026.

Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. IEEE Transactions on Pattern Analysis and Machine Intelli gence, 2025.

Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. arXiv preprint arXiv:2001.04451, 2020.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Haopeng Li, Yitong Li, Junsong Chen, Tian Ye, Haozhe Liu, Jincheng Yu, Duomin Wang, Ruihua Zhang, Zeke Xie, Enze Xie, et al. Sol-attn: Accelerating video generation inference via on-the-fly attention sparsification. arXiv preprint arXiv:2607.24027, 2026a.

Haopeng Li, Shitong Shao, Wenliang Zhong, Zikai Zhou, Lichen Bai, Hui Xiong, and Zeke Xie. Pisa: Piecewise sparse attention is wiser for efficient diffusion transformers. arXiv preprint arXiv:2602.01077, 2026b.

Xiaolong Li, Youping Gu, Xi Lin, Weijie Wang, and Bohan Zhuang. Psa: Pyramid sparse attention for efficient video understanding and generation. arXiv preprint arXiv:2512.04025, 2025.

Xingyang Li, Muyang Li, Tianle Cai, Haocheng Xi, Shuo Yang, Yujun Lin, Lvmin Zhang, Songlin Yang, Jinbo Hu, Kelly Peng, et al. Radial attention: o(n log n) sparse attention with energy decay for long video generation. Advances in Neural Information Processing Systems, 38:16822–16852, 2026c.

Xuewen Liu, Zhikai Li, Jing Zhang, Mengjuan Chen, and Qingyi Gu. Rectified spaattn: Revisiting attention sparsity for efficient video generation. arXiv preprint arXiv:2511.19835, 2025.

Jiayi Luo, Jiayu Chen, Jiankun Wang, Cong Wang, Hanxin Zhu, Qingyun Sun, Chen Gao, Zhibo Chen, and Jianxin Li. Attention sparsity is input-stable: Training-free sparse attention for video generation via offline sparsity profiling and online qk co-clustering. arXiv preprint arXiv:2603.18636, 2026.

NVIDIA. Pbench: A physical ai benchmark for world models, 2025. URL https:// huggingface.co/datasets/nvidia/PBench.

NVIDIA. World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062, 2025.

NVIDIA. Cosmos 3: Omnimodal world models for physical ai. arXiv preprint arXiv:2606.02800, 2026. URL https://research.nvidia.com/labs/cosmos-lab/cosmos3/ technical-report.pdf.

Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse attention with routing transformers. Transactions of the Association for Computational Linguistics, 9:53–68, 2021.

Dvir Samuel, Issar Tzachor, Matan Levy, Michael Green, Gal Chechik, and Rami Ben-Ari. Fast autoregressive video diffusion and world models with temporal cache compression and sparse attention. arXiv preprint arXiv:2602.01801, 2026.

Xuan Shen, Chenxia Han, Yufa Zhou, Yanyue Xie, Yifan Gong, Quanyi Wang, Yiwei Wang, Yanzhi Wang, Pu Zhao, and Jiuxiang Gu. Draftattention: Fast video diffusion via low-resolution attention guidance. arXiv preprint arXiv:2505.14708, 2025.

Wenhao Sun, Rong-Cheng Tu, Yifu Ding, Jingyi Liao, Zhao Jin, Shunyu Liu, and Dacheng Tao. Vorta: Efficient video diffusion via routing sparse attention. Advances in Neural Information Processing Systems, 38:7837–7863, 2026.

Haoyue Tan, Shengnan Wang, Yulin Qiao, Juncheng Zhang, Youhui Bai, Ping Gong, Zewen Jin, and Cheng Li. Adacluster: Adaptive query-key clustering for sparse attention in video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 43249–43259, 2026.

Yi Tay, Dara Bahri, Liu Yang, Donald Metzler, and Da-Cheng Juan. Sparse sinkhorn attention. In International conference on machine learning, pp. 9438–9447. PMLR, 2020.

Apoorv Vyas, Angelos Katharopoulos, and Franc¸ois Fleuret. Fast transformers with clustered attention. Advances in Neural Information Processing Systems, 33:21665–21674, 2020.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Jianzong Wu, Liang Hou, Haotian Yang, Xin Tao, Ye Tian, Pengfei Wan, Di Zhang, and Yunhai Tong. Vmoba: Mixture-of-block attention for video diffusion models. arXiv preprint arXiv:2506.23858, 2025.

Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025.

Yifei Xia, Suhan Ling, Fangcheng Fu, Yujie Wang, Huixia Li, Xuefeng Xiao, and Bin Cui. Trainingfree and adaptive sparse attention for efficient long video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15982–15993, 2025.

Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, and Vikas Singh. Nystromformer: A nystr¨ om-based algorithm for approximating self-attention. In¨ Proceedings of the AAAI conference on artificial intelligence, volume 35, pp. 14138–14148, 2021.

Ruyi Xu, Guangxuan Xiao, Haofeng Huang, Junxian Guo, and Song Han. Xattention: Block sparse attention with antidiagonal scoring. arXiv preprint arXiv:2503.16428, 2025.

Shuo Yang, Haocheng Xi, Yilong Zhao, Muyang Li, Jintao Zhang, Han Cai, Yujun Lin, Xiuyu Li, Chenfeng Xu, Kelly Peng, et al. Sparse videogen2: Accelerate video generation with sparse attention via semantic-aware permutation. Advances in Neural Information Processing Systems, 38:96965–96991, 2026.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, et al. Flashinfer: Efficient and customizable attention engine for llm inference serving. Proceedings of Machine Learning and Systems, 7, 2025.

Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, and Jianfei Chen. Spargeattention: Accurate and training-free sparse attention accelerating any model inference. arXiv preprint arXiv:2502.18137, 2025a.

Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang Ding, Ion Stoica, Zhengzhong Liu, and Hao Zhang. Fast video generation with sliding tile attention. arXiv preprint arXiv:2502.04507, 2025b.

Peiyuan Zhang, Yongqi Chen, Haofeng Huang, Will Lin, Zhengzhong Liu, Ion Stoica, Eric Xing, and Hao Zhang. Faster video diffusion with trainable sparse attention. Advances in Neural Information Processing Systems, 38:152509–152534, 2026.

Xuzhe Zheng, Yuexiao Ma, Jing Xu, Xiawu Zheng, Rongrong Ji, and Fei Chao. Haste: Trainingfree video diffusion acceleration via head-wise adaptive sparse attention. arXiv preprint arXiv:2605.14513, 2026.

Ruiliang Zhou, Xuecheng Wu, Kang He, Guangyun Han, Bin Liu, Qinqin Chen, Wende Xu, Qingjie Zhao, and Chengru Song. Scalingattention: Discovering intrinsic sparse attention topology for video diffusion transformers. arXiv preprint arXiv:2606.23019, 2026a.

Xuanyi Zhou, Qiuyang Mang, Shuo Yang, Haocheng Xi, Jintao Zhang, Huanzhi Mao, Joseph E Gonzalez, Kurt Keutzer, Ion Stoica, and Alvin Cheung. Svg-ear: Parameter-free linear compensation for sparse video generation via error-aware routing. arXiv preprint arXiv:2603.08982, 2026b.

## A PROBE-FITTED RESIDUAL RECONSTRUCTION DETAILS

This appendix derives the exact hard-drop residual and details the call-specific estimator used by Probe-Fitted Residual Reconstruction. The final estimator uses only the sparse output as its regression feature. Query-response coordinates are used for query partitioning and probe selection, but not for residual regression.

## A.1 HARD-DROP RESIDUAL AND AFFINE MODEL

For one query row $i ,$ let $s _ { i }$ and $\mathcal { U } _ { i }$ denote its retained and omitted key-index sets. Define the attention logit

$$
\ell _ { i j } = \frac { q _ { i } ^ { \top } k _ { j } } { \sqrt { d } } .\tag{A.1}
$$

The retained and omitted partition sums and value numerators are

$$
Z _ { S , i } = \sum _ { j \in S _ { i } } \exp ( \ell _ { i j } ) ,
$$

$$
N _ { S , i } = \sum _ { j \in S _ { i } } \exp ( \ell _ { i j } ) v _ { j } ,
$$

$$
Z _ { U , i } = \sum _ { j \in \mathcal { U } _ { i } } \exp ( \ell _ { i j } ) ,
$$

$$
N _ { U , i } = \sum _ { j \in \mathcal { U } _ { i } } \exp ( \ell _ { i j } ) v _ { j } .\tag{A.2}
$$

Dense attention and renormalized hard drop therefore compute

$$
O _ { i } ^ { \mathrm { d e n s e } } = { \frac { N _ { S , i } + N _ { U , i } } { Z _ { S , i } + Z _ { U , i } } } , \qquad O _ { i } ^ { \mathrm { s p } } = { \frac { N _ { S , i } } { Z _ { S , i } } } .\tag{A.3}
$$

Let

$$
p _ { U , i } = \frac { Z _ { U , i } } { Z _ { S , i } + Z _ { U , i } } , \qquad O _ { U , i } = \frac { N _ { U , i } } { Z _ { U , i } } \quad ( p _ { U , i } > 0 )\tag{A.4}
$$

denote the omitted dense-softmax mass and the output obtained by renormalizing over the omitted support. Then

$$
O _ { i } ^ { \mathrm { d e n s e } } = ( 1 - p _ { U , i } ) O _ { i } ^ { \mathrm { s p } } + p _ { U , i } O _ { U , i } ,\tag{A.5}
$$

and the exact post-softmax residual is

$$
\Big | R _ { i } = O _ { i } ^ { \mathrm { d e n s e } } - O _ { i } ^ { \mathrm { s p } } = p _ { U , i } \left( O _ { U , i } - O _ { i } ^ { \mathrm { s p } } \right) . \Big |\tag{A.6}
$$

The sparse output is observable during sparse execution, whereas $p _ { U , i }$ and $O _ { U , i }$ depend on omitted interactions. We therefore fit a call-specific affine model from the sparse output to the measured residual,

$$
x _ { i } = O _ { i } ^ { \mathrm { s p } } \in \mathbb R ^ { d _ { v } } , \qquad R _ { i } \approx b + x _ { i } B .\tag{A.7}
$$

This is an empirical local model for one active attention call, not an assumption that the omitted statistics are uniquely determined by $O _ { i } ^ { \mathrm { s p } }$ . The oracle analysis in the main paper measures how much of the residual can be represented by this affine family under different partitions.

## A.2 QUERY-GROUP-STRATIFIED EXACT PROBE SELECTION

The affine map depends on the current input, head, layer, routing pattern, and denoising step. We therefore estimate it from exact rows of the active attention call rather than from an offline dataset. For each query head, we evaluate $M = 6 4 \ll N _ { q }$ probe rows against all keys. The probe contribution to executed-pair density is $M / N _ { q } .$ . Across the evaluated sequence lengths, this fraction ranges from 0.054% to 0.145%.

Probe selection uses the query partition and the query-response coordinates from Response-Coupled Partitioning, without inspecting the dense residual. For each nonempty query group $\dot { G } _ { a } ^ { Q }$ , define

$$
c _ { a } = \frac { 1 } { | G _ { a } ^ { Q } | } \sum _ { i \in G _ { a } ^ { Q } } \phi _ { Q } ( q _ { i } ) , \qquad d _ { i } = \| \phi _ { Q } ( q _ { i } ) - c _ { a } \| _ { 2 } .\tag{A.8}
$$

Rows within each group are ordered by increasing $d _ { i } ,$ with ties broken by the original row index. A round-robin traversal first draws from distinct nonempty groups and then takes additional rows from their radial orderings until the probe budget is exhausted.

Require: Query groups $\mathcal { G } ^ { Q } \mathrm { : }$ ; query-response coordinates $\phi _ { Q } ( q _ { i } ) _ { i = 1 } ^ { N _ { q } }$ ; probe budget M   
Ensure: Probe set $\mathcal { P }$   
1: $\mathcal { P }  \emptyset$   
2: $\mathcal { A }  a : , | G _ { a } ^ { Q } | > 0$ in deterministic group order   
3: for all $a \in { \mathcal { A } }$ do   
4: $c _ { a } \gets | G _ { a } ^ { Q } | ^ { - 1 } \sum _ { i \in G _ { a } ^ { Q } } \phi _ { Q } ( q _ { i } )$   
5: $\pi _ { a }  \boldsymbol { S }$ ORTINDICESASCENDING $\displaystyle : ( i \in G _ { a } ^ { Q } , \vert \phi _ { Q } ( q _ { i } ) - c _ { a } \vert _ { 2 } )$   
6: end for   
7: t ← 1   
8: while $| \mathcal { P } | < M$ do   
9: for all a $\in { \mathcal { A } }$ do   
10: $\mathbf { i f } t \leq | G _ { a } ^ { Q } |$ and $| \mathcal { P } | < M$ then   
11: ${ \mathcal { P } } \dot { \gets } { \dot { \mathcal { P } } } \cup \pi _ { a } \dot { ( t ) }$   
12: end if   
13: end for   
14: $t \gets t + 1$   
15: end while   
16: return $\mathcal { P }$

For each probe $p \in \mathcal P$ , exact attention provides

$$
R _ { p } = O _ { p } ^ { \mathrm { d e n s e } } - O _ { p } ^ { \mathrm { s p } } .\tag{A.9}
$$

The estimator is fitted from the pairs $\{ ( x _ { p } , R _ { p } ) \} _ { p \in \mathcal { P } }$

Group-coverage weights. Let $m _ { a } = | \mathcal { P } \cap G _ { a } ^ { Q } |$ be the number of probes selected from query group a. Each probe $\check { p } \in G _ { a } ^ { Q }$ receives weight

$$
w _ { p } = \frac { | G _ { a } ^ { Q } | } { m _ { a } } .\tag{A.10}
$$

Thus, $\begin{array} { r } { \sum _ { p \in { \mathcal { P } } \cap G _ { a } ^ { Q } } w _ { p } = | G _ { a } ^ { Q } | } \end{array}$ , so groups with more selected probes do not receive disproportionate influence. These are deterministic coverage weights, not an importance-sampling correction.

## A.3 WEIGHTED AFFINE FITTING AND PROBE-RESIDUAL OUTPUT SUBSPACE

Normalize the coverage weights to unit sum,

$$
\alpha _ { p } = \frac { w _ { p } } { \sum _ { t \in \mathcal { P } } w _ { t } } .\tag{A.11}
$$

The weighted feature and residual means are

$$
\mu _ { x } = \sum _ { p \in \mathcal { P } } \alpha _ { p } x _ { p } , \qquad \mu _ { R } = \sum _ { p \in \mathcal { P } } \alpha _ { p } R _ { p } .\tag{A.12}
$$

Define the elementwise feature scale

$$
s _ { x } = \left[ \sum _ { p \in \mathcal { P } } \alpha _ { p } ( x _ { p } - \mu _ { x } ) ^ { \odot 2 } + \epsilon \right] ^ { 1 / 2 } ,\tag{A.13}
$$

where $\odot 2$ denotes elementwise squaring. The standardized feature and centered residual are

$$
\begin{array} { r } { \bar { x } _ { p } = ( x _ { p } - \mu _ { x } ) \oslash s _ { x } , \qquad \bar { R } _ { p } = R _ { p } - \mu _ { R } , } \end{array}\tag{A.14}
$$

where $\oslash$ denotes elementwise division. Stacking the probe rows gives $\bar { X } _ { \mathcal { P } } \in \mathbb { R } ^ { M \times d _ { v } }$ and $\bar { R } _ { \mathcal { P } } ~ \in$ $\mathbb { R } ^ { M \times d _ { v } }$

For the regression objective, rescale the coverage weights to have unit mean,

$$
\widetilde { w } _ { p } = \frac { M w _ { p } } { \sum _ { t \in \mathcal { P } } w _ { t } } , \qquad W = \mathrm { d i a g } \left( \{ \widetilde { w } _ { p } \} _ { p \in \mathcal { P } } \right) .\tag{A.15}
$$

This preserves their relative values while keeping the scale of the ridge coefficient less dependent on the query count.

Weighted ridge fit. The centered feature-to-residual map is

$$
B _ { \lambda } = \arg \operatorname* { m i n } _ { { \cal B } } \left\| { \cal W } ^ { 1 / 2 } \left( \bar { R } _ { \mathcal { P } } - \bar { X } _ { \mathcal { P } } { \cal B } \right) \right\| _ { F } ^ { 2 } + \lambda \| { \cal B } \| _ { F } ^ { 2 } .\tag{A.16}
$$

Its closed form is

$$
B _ { \lambda } = \left( \bar { X } _ { \mathcal P } ^ { \top } W \bar { X } _ { \mathcal P } + \lambda I \right) ^ { - 1 } \bar { X } _ { \mathcal P } ^ { \top } W \bar { R } _ { \mathcal P } .\tag{A.17}
$$

The implementation solves the regularized normal equations with a linear solver rather than explicitly forming the inverse.

Probe-residual output subspace. Let $\Psi _ { r } \in \mathbb { R } ^ { d _ { v } \times r }$ contain the leading r right singular vectors of the weighted, centered exact probe-residual matrix,

$$
W ^ { 1 / 2 } \bar { R } _ { \mathcal { P } } = U _ { R } \Sigma _ { R } V _ { R } ^ { \top } , \qquad \Psi _ { r } = ( V _ { R } ) _ { [ : , 1 : r ] } .\tag{A.18}
$$

This basis is constructed from the measured probe residuals, not from the fitted residuals. It identifies output directions observed in the exact current-call corrections.

For an unprobed row i, define $\bar { x } _ { i } = ( x _ { i } - \mu _ { x } ) \oslash s _ { x }$ . Its predicted residual is

$$
\widehat { R } _ { i } = \mu _ { R } + \bar { x } _ { i } B _ { \lambda } \Psi _ { r } \Psi _ { r } ^ { \top } .\tag{A.19}
$$

The projector restricts the feature-dependent correction to output directions observed in the probe residuals. It does not assume that the complete post-softmax residual is globally low rank. No correction-norm cap is applied. The final output is

$$
\begin{array} { r } { \widehat { O } _ { i } = \left\{ \begin{array} { l l } { O _ { i } ^ { \mathrm { d e n s e } } , } & { i \in \mathcal { P } , } \\ { O _ { i } ^ { \mathrm { s p } } + \widehat { R } _ { i } , } & { i \notin \mathcal { P } . } \end{array} \right. } \end{array}\tag{A.20}
$$

## B RESPONSE-COUPLED PARTITIONING AND SPARSE EXECUTION

## B.1 KEY-RESPONSE COORDINATES FOR PAIRED K/V GROUPS

For each K/V head, let $Q _ { s } \in \mathbb { R } ^ { M _ { s } \times d }$ contain a deterministic sample of query rows. Define

$$
M _ { K } = \frac { 1 } { M _ { s } } { Q } _ { s } ^ { \top } { Q } _ { s } .\tag{B.1}
$$

For any two keys $k _ { i }$ and $k _ { j }$ ,

$$
\big ( k _ { i } - k _ { j } \big ) ^ { \top } M _ { K } \big ( k _ { i } - k _ { j } \big ) = \frac { 1 } { M _ { s } } \left\| Q _ { s } \big ( k _ { i } - k _ { j } \big ) \right\| _ { 2 } ^ { 2 } .\tag{B.2}
$$

Thus, the metric compares pre-softmax response profiles under the sampled queries. Let $F _ { K } \in$ $\mathbb { R } ^ { d \times r _ { K } }$ contain the leading $r _ { K }$ eigendirections of $M _ { K }$ , scaled by the square roots of their eigenvalues after the retained eigenvalues are normalized to unit mean. The key-response coordinate is

$$
\begin{array} { r } { \phi _ { K } ( k _ { j } ) = F _ { K } ^ { \top } k _ { j } \in \mathbb { R } ^ { r _ { K } } . } \end{array}\tag{B.3}
$$

Applying k-means to $\flat _ { K } ( k _ { j } )$ produces the paired K/V groups. Each value $v _ { j }$ inherits the assignment of its paired key $k _ { j }$ . Values do not enter the partition feature.

## B.2 QUERY-RESPONSE COORDINATES FROM K/V GROUPS

Let $C _ { k } ^ { + }$ denote the number of nonempty K/V groups. We compute each nonempty group’s centroid in the original key space, subtract the uniform mean over nonempty groups, and stack the centered centroids as $\widetilde { K } \in \mathbb { R } ^ { C _ { k } ^ { + } \times d }$ . The query-response metric is

$$
M _ { Q } = \frac { \widetilde { K } ^ { \top } \widetilde { K } } { C _ { k } ^ { + } d } .\tag{B.4}
$$

Let $F _ { Q } \in \mathbb { R } ^ { d \times r _ { Q } }$ be obtained from the leading $r _ { Q }$ eigendirections of $M _ { Q }$ using the same eigenvalue weighting as above. The normalized query-response coordinate is

$$
\phi _ { Q } ( q _ { i } ) = \frac { F _ { Q } ^ { \top } q _ { i } } { \operatorname* { m a x } \Bigl ( \sqrt { \| F _ { Q } ^ { \top } q _ { i } \| _ { 2 } ^ { 2 } / r _ { Q } } , \epsilon \Bigr ) } \in \mathbb { R } ^ { r _ { Q } } , \qquad \epsilon = 1 0 ^ { - 6 } .\tag{B.5}
$$

Only nonempty K/V groups contribute to $M _ { Q }$ , and their centered centroids receive equal weight rather than token-count weight. Clustering $\phi _ { Q } ( q _ { i } )$ gives the query partition. The same coordinates are used to stratify probe selection, but not as residual-regression features. Together, the K/V and query stages form a coupled, single-pass construction without alternating refinement between the two partitions.

## C IMPLEMENTATION AND EVALUATION DETAILS

## C.1 GPU IMPLEMENTATION

The implementation uses warm-started GPU k-means and a fused query-response projection and RMS-normalization kernel. At the evaluated production shape, the fused query projection reduces latency from 1.6170 ms to 0.5064 ms and has relative feature error $3 . 6 8 \times 1 0 ^ { - 4 }$ against the reference implementation. Residual fitting, probe-residual SVD, projection, and correction are implemented with small batched GPU linear-algebra operations.

## C.2 SHARED HYPERPARAMETERS

All experiments use BF16 on a single NVIDIA H100 GPU. The method settings are shared across models except for the selected total executed-pair target.

Table C.1: Method configuration used in the evaluated models.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Key-response rank  $r _ { K }$ </td><td>48</td></tr><tr><td>Query-response rank  $r _ { Q }$ </td><td>64</td></tr><tr><td>Exact probes per query head  $M$ </td><td>64</td></tr><tr><td>Probe-residual output rank r</td><td>16</td></tr><tr><td>Ridge coefficient  $\bar { \lambda }$ </td><td>0.1</td></tr><tr><td>Target  $\rho _ { \mathrm { e x e c } }$  for HunyuanVideo, Wan2.2, Cosmos-Predict2.5</td><td>0.22</td></tr><tr><td>Target  $\rho _ { \mathrm { e x e c } }$  for Cosmos3-Nano</td><td>0.26</td></tr></table>

Table C.2: Probe-row fractions for the evaluated sequence lengths with $M = 6 4$
<table><tr><td>Model</td><td> $N _ { q }$ </td><td> ${ M } / { N _ { q } }$ </td></tr><tr><td>HunyuanVideo</td><td>118,800</td><td>0.054%</td></tr><tr><td>Wan2.2</td><td>75,600</td><td>0.085%</td></tr><tr><td>Cosmos-Predict2.5</td><td>84,480</td><td>0.076%</td></tr><tr><td>Cosmos3-Nano</td><td>44,160</td><td>0.145%</td></tr></table>

## C.3 TIMING PROTOCOL

Dense and sparse runs use matched checkpoints, conditioning inputs, random seeds, sampling schedules, inference steps, guidance settings, resolutions, and frame counts. Executed-pair density counts exact query–key interactions, including probe rows. Reported latency also includes response-coordinate construction, clustering, routing, permutation, sparse attention, exact probes, residual fitting, correction, and output restoration.

Error-density trade-off under identical uncapped raw-residual repair

![](images/47fa588cd094f15b5ce43e95de9986d87bfd95342fd47c4785805671e76764d1.jpg)  
Total executed-pair density (%)  
Figure D.1: Cross-model density sensitivity. Mean (top) and p99 (bottom) normalized attentionoutput error versus total executed-pair density for semantic and response-coupled partitioning under the same residual-repair configuration. The dashed line marks the 22% reference density.

## D CROSS-MODEL DENSITY SENSITIVITY

Figure D.1 compares semantic and response-coupled partitioning across total executed-pair densities from 12% to 35%, under the same residual-repair configuration. Response-coupled partitioning reduces mean error across all densities and models, with p99 error also reduced over nearly the full range. The gains persist across operating points, showing that the partitioning benefit is not specific to a single density.

## E MORE QUALITATIVE RESULTS

![](images/88165db9f52689f55399fbbd01cb6322afe3cc3c77328d2c55e42e7cf7a73947.jpg)  
Figure E.1: Qualitative comparison on Cosmos3-Nano-16B.

![](images/9a7496b40b5483144bf20e73a40f197cc7d95c197c20fdd440b9ee15f340d68e.jpg)  
Figure E.2: Qualitative comparison on Cosmos3-Nano-16B.

![](images/cfa1b8ce1680a17879ad8f27d067299b0765c73b0b4febcdf932dbdaf69a707b.jpg)  
Figure E.3: Qualitative comparison on HunyuanVideo-13B.

![](images/3dc50d04b870188f48af059953873684adcf44f0297b82ba35fb6a10f1ee4b24.jpg)  
Figure E.4: Qualitative comparison on Wan2.2-I2V-A14B.