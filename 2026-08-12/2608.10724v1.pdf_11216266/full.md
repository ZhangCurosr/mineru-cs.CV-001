# InterPruner: Interactive Structured Pruning via Taylor-Implicit Criterion and Language-Prior Modulator for Multimodal Object Detection

Qi Ming<sup>1</sup>, Zihan Yang<sup>2</sup>, Shaoguang Huang<sup>3</sup>, Si Sun<sup>4</sup>, Hanqing Zhang<sup>5</sup>, Nanqing Liu<sup>6</sup>, Jiahui Lv<sup>7</sup>, Juan Fang<sup>1</sup>, Aleksandra Pizurica<sup>8</sup>

<sup>1</sup>Beijing University of Technology, <sup>2</sup>Southwest University, <sup>3</sup>China University of Geosciences Wuhan

<sup>4</sup>Tsinghua University, <sup>5</sup>Beijing Institute of Technology, <sup>6</sup>Yunnan Normal University, <sup>7</sup>Beijing Forestry University

<sup>8</sup>Department of Telecommunications and Information Processing (TELIN), Ghent University

chaser.ming@gmail.com, yangzihan0211@163.com, Aleksandra.Pizurica@ugent.be

## Abstract

Multimodal object detection proves effective in remote sensing, especially the RGB-Infrared paradigm. The parallel feature extractors provide rich multimodal information for robust detection, yet introduce substantial channel redundancy and computational overhead. Existing pruning methods can reduce channel redundancy, but they are designed for unimodal backbones, overlooking cross-modal interactions and dynamic scene-wise redundancy. In this paper, we propose InterPruner, the first interactive structured channel pruning framework for RGB–infrared object detectors. Specifically, we first derive a Taylor-Implicit Criterion(TIC) to quantify channel importance via high-order Taylor expansion and implicit function theorem. Then, a Modality Interaction Redundancy Analyzer (MIRA) identifies redundant channels via mutual compensability assessment. Finally, a Scene-Prior Channel Anchor (SPCA) uses language priors as semantic anchors to measure channelscene relevance for dynamic channel importance estimation. Cross-modality channel pruningfor RGB-Infrared detection is yet unexplored. Extensive experiments on RGBinfrared object detection dataset demonstrate that Inter-Pruner maintains high performance with negligible degradation. Specifically, it even achieves a 0.6% mAP increase on the FLIR dataset when pruning 50% of the channels. Code will be available on GitHub to facilitate future work.

## 1. Introduction

Multimodal object detection plays an important role in remote sensing applications [8]. By integrating visual cues from RGB [30], infrared [6], depth [22] and radar [15], multimodal systems seek to enhance scene understanding under adverse conditions. Among these modalities, RGB-infrared object detection [16, 23] serves as a representative application scenario. Specifically, RGB images provide rich texture and semantic information, while infrared images offer reliable temperature cues that are insensitive to illumination changes. Existing RGB-infrared detectors employ two modality-specific branches with feature fusion modules to extract features [18, 21, 28]. However, the dual-branch architecture introduces substantial cross-modal redundancy, since many channels in the two streams contain correlated features that contribute little to detection accuracy. This redundancy causes unnecessary computation, making deployment on resource-constrained platforms challenging.

![](images/f89be78a0ec67a8911bdbe612b18fcb80e727e38dbd077d9478249b9d9aefebd.jpg)  
Figure 1. Feature visualizations of the RGB and Infrared branches before and after unimodal pruning at the P4 stage. Compared to the unpruned baseline (top), pruning the RGB branch (bottom) has a negligible impact on the RGB feature but leads to feature misalignment in the infrared branch.

Pruning offers a promising solution to mitigate this efficiency problem via unstructured [13] or structured [12] approaches. Structured pruning, in particular, operates at different granularities depending on the backbone: token-level [1] for transformers and Mamba [25], versus channel-level for convolutional neural networks [3]. Given the high resolution of remote sensing imagery, most RGB-Infrared detectors still adopt CNN backbones for their efficient multiscale feature extraction. But these channel-level pruning techniques are mainly designed for single-stream CNNs.

Directly extending single-stream pruning methods to the dual-stream setting results in performance degradation. We attribute this degradation to two factors: (1) Channel Importance is Coupled Across Modalities. The two branches are inherently coupled via cross-modal interactions. Pruning one branch can impair the other’s representation, even if the pruned channels appear unimportant in isolation. (2) Channel Importance Varies with Scenes. Pruning decisions optimal for one scene may become suboptimal under another, as the redundancy distribution changes with environmental conditions [20]. A fixed pruning criterion cannot adapt to such changes.

To overcome these limitations, we propose InterPruner, the first interactive structured channel pruning framework for RGB–infrared object detection. First, we formulate a Taylor-Implicit Criterion (TIC) that approximates the postremoval loss change through Taylor expansion and implicit function theorem to reveal each channel’s genuine contribution. Next, we employ a Modality Interaction Redundancy Analyzer (MIRA) that alternately masks each modality and checks for compensation by the other to identify redundant channels. Finally, a Scene-Prior Channel Anchor (SPCA) harnesses language priors to dynamically evaluate channel importance. These modules produce a unified importance score that guides structured channel removal, preserving critical cross-modal information while adjusting to different scenes. InterPruner bridges the gap of structured pruning for RGB-infrared object detection, which remains unexplored to the best of our knowledge.

In summary, the major contributions of our work are:

• We propose a Taylor-Implicit Criterion (TIC) that estimates channel importance via high-order loss expansion, providing a robust pruning basis for modalities.

• A Modality Interaction Redundancy Analyzer (MIRA) is designed to identify cross-modal redundant channels via compensability assessment, ensuring that only informative features are retained for detection.

• To make pruning decisions scene-aware, we introduce a Scene-Prior Channel Anchor (SPCA) that leverages scene priors, thereby enabling the model to identify redundancy across different visual contexts.

## 2. Related Work

## 2.1. Multimodal Object Detection

Multimodal detection integrates features from diverse sensors to enable robust perception. RGB-infrared detection, in particular, combines the visual cues with the thermal features, proving effective under adverse conditions [18, 28]. Various backbones have been explored for this task. Vision Transformers [16] and their adapter-based variants [24] suffer from high computational cost. Emerging mamba-based models [11] excel at global context modeling, yet their efficiency bottlenecks lie at the sequence level. In practice, CNNs [18, 28, 29] remain the dominant backbone for RGB-Infrared detection, offering a favorable balance of efficiency and representational power. Despite their effectiveness, these dual-branch architectures introduce substantial channel redundancy, as many channels across the two streams encode correlated features with limited contribution to final detection accuracy. In this paper, we aim to reduce this redundancy and computational overhead while maintaining detection performance.

## 2.2. Structured Channel Pruning

Channel pruning removes entire filters or feature maps to achieve structured sparsity and produce compact models. Early approaches prune channels with small BN scaling factors [12], and later variants further separate important from unimportant channels via regularization [7]. More recent works attempt to estimate the post-pruning loss change via influence functions [2] or unify static and dynamic pruning through bi-level optimization [3]. Furthermore, domain-specific channel pruning has also been explored. IRPruneDeXt [27] introduces wavelet-regularized channel pruning for infrared small target detection, while CSP [9] performs joint channel and space pruning for general CNNs. Despite these advances, the assumption remains that channels can be judged independently. But this assumption fails in multimodal detectors, where a channel pruned from one branch may be compensated by the other.

## 3. Methodology

## 3.1. Preliminaries

Let $x _ { i } ^ { r g b }$ and $x _ { i } ^ { i r }$ denote aligned RGB and infrared images, and $y _ { i }$ be the detection annotation. W and M denote the model weights and channel-wise binary masks respectively. Therefore, a RGB-infrared detector is formulated as:

$$
\hat { y } _ { i } = f ( x _ { i } ^ { r g b } , x _ { i } ^ { i r } ; { \bf W } , { \bf M } ) .\tag{1}
$$

For modality $m \in \{ r g b , i r \}$ , we define $\mathbf { W } _ { m } = \{ \mathbf { W } _ { m } ^ { ( l ) } \} _ { l = 1 } ^ { L }$ and $\mathbf { M } _ { m } = \{ M _ { m } ^ { ( l , c ) } \in \{ 0 , 1 \} \} _ { l = 1 , c = 1 } ^ { L , C _ { m } ^ { ( l ) } }$ , with $C _ { m } ^ { ( l ) }$ being the number of output channels in layer l. Each mask $M _ { m } ^ { ( l , c ) } = 0$ indicates that the c-th channel in the l-th layer of branch m is pruned during inference.

Given a pretrained detector, we seek an optimal compact mask Mc under pruning ratio $\rho ,$ formulated as an optimization over detection loss L and training loss $\ell \left[ 2 \right] :$

![](images/da122c61c14a4e836e36c2142320c8e1ad658ba393e23f0aed74edbced627f79.jpg)  
Figure 2. Overveiw of InterPruner: Given multi-scale channels from RGB and infrared backbones, TIC evaluates each channel’s contribution $C _ { q }$ via loss variation. MIRA then exposes cross-modal redundancy $R _ { q }$ through alternating masking and compensability assessment. In parallel, SPCA extracts scene semantics from VLM using RGB-IR pairs and prompts to produce a semantic weight w(x). These three cues are aggregated into $S _ { q }$ for channel ranking and pruning, yielding a compact detector.

$$
\begin{array} { r l } & { \displaystyle \operatorname* { m i n } _ { \widehat { \mathbf { M } } \in { \mathcal { S } } } { \mathcal { L } \big ( \mathbf { W } ^ { * } ( \widehat { \mathbf { M } } ) , \widehat { \mathbf { M } } \big ) } , } \\ & { \widehat { \mathbf { M } } \in { \mathcal { S } } } \\ & { \widehat { \mathbf { M } } \mathbf { \widetilde { \mathbf { \Lambda } } } ( \widehat { \mathbf { M } } ) = \arg \displaystyle \operatorname* { m i n } _ { \mathbf { W } } { \ell \big ( \widehat { \mathbf { M } } \odot \mathbf { W } \big ) } + R ( \mathbf { W } ) , } \end{array}\tag{2}
$$

where $R ( \cdot )$ is a regularization term, and $s$ denotes the feasible set of masks satisfying the pruning ratio $\rho .$

## 3.2. Taylor-Implicit Criterion

The original model parameters W are trained to convergence $\mathbf { W } ^ { * }$ under the mask M. Removing a single channel gives a new mask $\widetilde { \mathbf { M } }$ and its retrained optimum is $\widetilde { \mathbf { W } } ^ { * }$ Here, we define two types of loss changes. The immediate loss change $\Delta \mathcal { L } _ { e l }$ is the post-pruning loss increase without parameter update:

$$
\Delta \mathcal { L } _ { e l } = \mathcal { L } ( \mathbf { W } ^ { * } , \widetilde { \mathbf { M } } ) - \mathcal { L } ( \mathbf { W } ^ { * } , \mathbf { M } ) .\tag{3}
$$

The real loss change $\Delta \mathcal { L } _ { r l }$ is the loss increase after pruning with full retraining to convergence:

$$
\Delta \mathcal { L } _ { r l } = \mathcal { L } ( \widetilde { \mathbf { W } } ^ { * } , \widetilde { \mathbf { M } } ) - \mathcal { L } ( \mathbf { W } ^ { * } , \mathbf { M } ) .\tag{4}
$$

Directly computing $\Delta \mathcal { L } _ { r l }$ for each channel is infeasible. By subtracting Eq.3 and Eq.4, $\Delta \mathcal { L } _ { r l }$ can be rewritten as:

$$
\Delta \mathcal { L } _ { r l } = \Delta \mathcal { L } _ { e l } + \mathcal { L } ( \widetilde { \mathbf { W } } ^ { * } , \widetilde { \mathbf { M } } ) - \mathcal { L } ( \mathbf { W } ^ { * } , \widetilde { \mathbf { M } } ) .\tag{5}
$$

Since $\Delta \mathcal { L } _ { e l }$ can be directly computed from the weights $\mathbf { W } ^ { * }$ , our goal is to estimate the recovery loss without finetuning, which is defined as

$$
\Delta \mathcal { L } _ { r e c } = \mathcal { L } ( \widetilde { \mathbf { W } } ^ { * } , \widetilde { \mathbf { M } } ) - \mathcal { L } ( \mathbf { W } ^ { * } , \widetilde { \mathbf { M } } ) .\tag{6}
$$

Let $\Delta \mathbf { W } = \widetilde { \mathbf { W } } ^ { * } - \mathbf { W } ^ { * }$ denote the parameter displacement after pruning and fine-tuning. Since $\widetilde { \mathbf { W } } ^ { * }$ is a local optimum under Mf, it satisfies:

$$
\nabla _ { \mathbf { W } } \mathcal { L } ( \widetilde { \mathbf { W } } ^ { * } , \widetilde { \mathbf { M } } ) = \nabla _ { \mathbf { W } } \mathcal { L } ( \mathbf { W } ^ { * } + \Delta \mathbf { W } , \widetilde { \mathbf { M } } ) = 0\tag{7}
$$

Taylor Expansion We expand the real loss function and its gradient (Eq.7) around $\mathbf { W } ^ { * }$ in Eq.8 and Eq.9, respectively.

$$
\begin{array} { r l r } {  { \mathcal { L } ( \widetilde { \mathbf { W } } ^ { * } , \widetilde { \mathbf { M } } ) = \mathcal { L } ( \mathbf { W } ^ { * } + \Delta \mathbf { W } , \widetilde { \mathbf { M } } ) } } & { ( 8 ) } \\ & { } & { = \mathcal { L } ( \mathbf { W } ^ { * } , \widetilde { \mathbf { M } } ) + \mathbf { g } ^ { T } \Delta \mathbf { W } + \frac { 1 } { 2 } \Delta \mathbf { W } ^ { T } \mathbf { H } \Delta \mathbf { W } } \\ & { } & { + \frac { 1 } { 6 } \mathbf { T } ( \Delta \mathbf { W } , \Delta \mathbf { W } , \Delta \mathbf { W } ) + O ( \| \Delta \mathbf { W } \| ^ { 4 } ) , } \end{array}
$$

$$
\begin{array} { r l } & { \nabla _ { \mathbf { W } } \mathcal { L } ( \mathbf { W } ^ { * } + \Delta \mathbf { W } , \widetilde { \mathbf { M } } ) = 0 } \\ & { = \mathbf { g } + \mathbf { H } \Delta \mathbf { W } + \frac { 1 } { 2 } \mathbf { T } ( \Delta \mathbf { W } , \Delta \mathbf { W } ) + o ( \| \Delta \mathbf { W } \| ^ { 3 } ) . } \end{array}\tag{9}
$$

where we assume $\| \Delta \mathbf { W } \|$ is small, so higher-order terms are negligible. Here g, H, and T are the first-, second-, and third-order derivatives of L evaluated at $( \mathbf { W } ^ { * } , \widetilde { \mathbf { M } } )$

Implicit Function By the implicit function theorem, for sufficiently small $\mathbf { g } , \Delta \mathbf { W }$ admits a unique power series expansion in g, with $\Delta \mathbf { W } ^ { ( 1 ) }$ and $\Delta { \bf W } ^ { ( 2 ) }$ being the first- and second-order terms:

$$
\Delta \mathbf { W } = \Delta \mathbf { W } ^ { ( 1 ) } + \Delta \mathbf { W } ^ { ( 2 ) } + O ( \| \mathbf { g } \| ^ { 3 } ) .\tag{10}
$$

Substituting Eq.10 into Eq.9 and collecting terms by order yields:

$$
\Delta \mathbf { W } ^ { ( 1 ) } = - \mathbf { H } ^ { - 1 } \mathbf { g } ,\tag{11}
$$

$$
\Delta \mathbf { W } ^ { ( 2 ) } = - \frac { 1 } { 2 } \mathbf { H } ^ { - 1 } \mathbf { T } ( \Delta \mathbf { W } ^ { ( 1 ) } , \Delta \mathbf { W } ^ { ( 1 ) } ) .\tag{12}
$$

Thus, the second-order approximation of ∆W is

$$
\Delta { \mathbf W } \approx - { \mathbf H } ^ { - 1 } { \mathbf g } - \frac { 1 } { 2 } { \mathbf H } ^ { - 1 } { \mathbf T } ( \Delta { \mathbf W } ^ { ( 1 ) } , \Delta { \mathbf W } ^ { ( 1 ) } ) + O ( \| { \mathbf g } \| ^ { 3 } ) .\tag{13}
$$

Taylor-Implicit Criterion Combining Eq.13 with Eq.8 and eliminating higher-order terms via Eq.11–12 yields:

$$
\Delta \mathcal { L } _ { r e c } = - \frac { 1 } { 2 } \mathbf { g } ^ { T } \mathbf { H } ^ { - 1 } \mathbf { g } - \frac { 1 } { 6 } \mathbf { T } ( \mathbf { H } ^ { - 1 } \mathbf { g } , \mathbf { H } ^ { - 1 } \mathbf { g } , \mathbf { H } ^ { - 1 } \mathbf { g } )\tag{14}
$$

Finally, our Taylor-Implicit Criterion is defined as

$$
C _ { q } = \Delta \mathcal { L } _ { e l } ^ { \mathrm { q } } - \left( \frac { 1 } { 2 } \mathbf { g } _ { q } ^ { \intercal } \mathbf { H } _ { q } ^ { - 1 } \mathbf { g } _ { q } + \frac { 1 } { 6 } \mathbf { T } _ { q } ( \mathbf { H } _ { q } ^ { - 1 } \mathbf { g } _ { q } , \mathbf { H } _ { q } ^ { - 1 } \mathbf { g } _ { q } , \mathbf { H } _ { q } ^ { - 1 } \mathbf { g } _ { q } ) \right)\tag{15}
$$

This criterion estimates each channel’s importance via its contribution to the detection loss. Channels with smaller $C _ { q }$ are less critical and prioritized for pruning.

## 3.3. Modality Interaction Redundancy Analyzer

To analyze the role of each channel under cross-modal interaction, we alternately mask all channels in one modality by setting $\mathbf { M } _ { r g b } = \mathbf { 0 }$ or $\mathbf { M } _ { i r } = \mathbf { 0 }$ , and observe the loss variation of the other branch after short adaptation.

Reactivation via Short Adaptation. The discrete mask $M _ { m } ^ { ( l , c ) } \in \{ 0 , 1 \}$ controls channel retention but is nondifferentiable for gradient-based optimization. Therefore, we introduce a continuous scalar gate $z _ { q } \in [ 0 , 1 ]$ for each channel $q = ( m , l , c )$ , which scales the corresponding channel weights as $\mathbf { W } _ { m } ^ { ( l , c ) } ( z _ { q } ) = z _ { q } \cdot \mathbf { W } _ { m } ^ { ( l , c ) }$ . Here, $z _ { q } ~ = ~ 1$ preserves the channel while $z _ { q } = 0$ effectively disables it.

Take the infrared branch as the active modality for example. Each reactivation measurement starts from $\mathbf { W } ^ { * }$ , so the two modalities do not affect each other’s results. With the RGB branch disabled, the model undergoes B-step gradient adaptation on the infrared branch.

$$
\mathbf { W } ^ { b + 1 } = \mathbf { W } ^ { b } - \eta \nabla _ { \mathbf { W } } \mathcal { L } \big ( \mathbf { W } ^ { b } \big ) , b = 0 , 1 , \ldots , B - 1 ,\tag{16}
$$

where η is the short-adaptation learning rate. During this procedure, the masked RGB branch is disabled, forcing the active infrared branch to update its parameters. Here, underutilized channels originally masked by the RGB branch contribute substantially more to loss reduction. We term this reactivation and it serves as an indicator of cross-modal redundancy. Strong reactivation reveals channels that were suppressed, while essential channels show little change.

Channel Redundancy Measurement. We further quantify the loss variation during the B-batch adaptation. $G _ { q }$ is defined as the directional derivative of the detection loss with respect to $z _ { q } .$ This value reflects each channel’s importance under the current mask configuration:

$$
G _ { q } ( \mathbf { W } ) = \left. \frac { \partial \mathcal { L } ( \mathbf { W } , z _ { q } ) } { \partial z _ { q } } \right| _ { z _ { q } = 1 } = \left. \mathbf { W } _ { m } ^ { ( l , c ) } , \frac { \partial \mathcal { L } ( \mathbf { W } ) } { \partial \mathbf { W } _ { m } ^ { ( l , c ) } } \right. .\tag{17}
$$

Given a set of B batches, we compute the variation of $G _ { q }$ as redundency measurement.

$$
R _ { q } = \frac { 1 } { B } \left| \sum _ { b = 1 } ^ { B } \left( G _ { q } ( \mathbf { W } ^ { B } ) - G _ { q } ( \mathbf { W } ^ { 0 } ) \right) \right| .\tag{18}
$$

A large $R _ { q }$ indicates that the channel becomes more responsive when the opposite modality is absent. This suggests its functionality is suppressed in the original dualstream model. Conversely, a small $R _ { q }$ implies that the channel’s contribution remains largely unchanged regardless of whether the other modality is present. This indicates that it carries modality-specific information and cannot be easily compensated. Therefore, $R _ { q }$ serves as a reliable indicator of cross-modal redundancy. Channels with higher reactivation scores are more dispensable and should be prioritized for pruning.

## 3.4. Scene-Prior Channel Anchor

The relative reliability of RGB and thermal modalities varies with imaging conditions. To adapt pruning decisions accordingly, we introduce a Scene-Prior Channel Anchor (SPCA) that extracts structured scene knowledge from each image pair and converts it into channel-wise pruning weights. SPCA is implemented as a lightweight MLP and applied only to intermediate layers, where effective crossmodal fusion and task-relevant visual pattern recognition primarily occur.

Modality-aware prompt learning. For each aligned RGB-Infrared pair, we use a frozen Qwen2.5-VL model with a fixed text prompt to generate structured scene descriptions. Additionally, K learnable soft prompts $\mathrm { ~ \bf ~ P ~ } \in$ R<sup>K×dQ</sup> is also inserted into the multimodal input to make the encoding sensitive to modality reliability.

![](images/ef1e3c1fddde35bba01501bb27ec908979712f83f38951aff3879e7f8a92f75c.jpg)  
Figure 3. Comparison of the Pruning Method. While existing methods are suboptimal and fragile under varying lighting conditions, our SPCA enables adaptive pruning decisions conditioned on scene context.

We optimize P using modality-difference supervision from a frozen CLIP image encoder. Specifically, we define the RGB-Infrared discrepancy direction $\mathbf { v } _ { i } ^ { - }$ and common direction $\mathbf { v } _ { i } ^ { + }$ from the normalized CLIP embeddings of each pair. Let $\mathbf { H } _ { i } ^ { P }$ be the prompt tokens’ hidden states. Their pooled representation is projected into the CLIP space via a linear layer A to obtain $\mathbf { u } _ { i } .$ The prompt-learning objective is:

$$
\mathcal { L } _ { \mathrm { p r o m p t } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } ( 1 - \mathbf { u } _ { i } ^ { \top } \mathbf { v } _ { i } ^ { - } ) + \lambda _ { 1 } \frac { 1 } { B } \sum _ { i = 1 } ^ { B } ( \mathbf { u } _ { i } ^ { \top } \mathbf { v } _ { i } ^ { + } ) ^ { 2 } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { c o n } } ,\tag{19}
$$

which aligns prompts with RGB-thermal discrepancy, suppresses common information, and separates different pairs, with only P and A trainable.

Structured semantic encoding. With the learned prompt $\mathbf { P } ^ { * }$ , Qwen2.5-VL generates a structured description for each RGB-IR pair:

$$
\mathcal { D } _ { i } = G _ { \mathrm { Q w e n } } ( x _ { i } ^ { r } , x _ { i } ^ { t } ; \mathbf { P } ^ { * } ) = \{ d _ { i } ^ { \mathrm { e n v } } , d _ { i } ^ { \mathrm { s c e n e } } , d _ { i } ^ { \mathrm { o b j } } , d _ { i } ^ { \mathrm { t h e r m } } \} ,\tag{20}
$$

covering illumination, scene context, object-level evidence, and thermal properties. These four text fields are encoded by the CLIP text encoder and averaged along the field dimension to obtain a scene-level semantic embedding $\mathbf { z } _ { i } ~ \in$ $\mathbb { R } ^ { 7 6 8 }$

Channel-wise pruning modulation. Prompt learning and semantic description generation are performed offline before pruning calibration. The core of SPCA is to transform scene representation into channel-wise weights that modulate the pruning criterion. Let $\textit { b } = \ ( m , l )$ denote layer l in modality branch $m \in \{ r , t \}$ , and let $C _ { b }$ be its number of output channels. We implement the modulation network as a lightweight two-layer MLP that maps the

768-dimensional CLIP text embedding $\mathbf { z } _ { i }$ to channel-wise weights:

$$
\delta _ { i , b } = \operatorname { t a n h } \left( { \mathbf { W } } _ { 2 } \operatorname { R e L U } \left( { \mathbf { W } } _ { 1 } \mathbf { z } _ { i } + { \mathbf { b } } _ { 1 } \right) + { \mathbf { b } } _ { 2 } \right) \in \mathbb { R } ^ { C _ { b } } .\tag{21}
$$

To prevent the semantic module from uniformly scaling all channels in a branch, we center and normalize the residual:

$$
\widehat { \delta } _ { i , b } = \frac { \delta _ { i , b } - \mathrm { M e a n } ( \delta _ { i , b } ) } { \operatorname* { m a x } \left( 1 , \lVert \delta _ { i , b } - \mathrm { M e a n } ( \delta _ { i , b } ) \rVert _ { \infty } \right) } .\tag{22}
$$

The channel-wise semantic weights are defined as follow, where $\gamma$ controls the modulation range, yielding ${ \bf w } _ { i , b } \in$ $[ 1 - \gamma , 1 + \gamma ] ^ { C _ { b } }$

$$
\begin{array} { r } { \mathbf { w } _ { i , b } = \mathbf { 1 } + \gamma \widehat { \delta } _ { i , b } , } \end{array}\tag{23}
$$

The lightweight MLP is optimized via the detection objective with differentiable channel masks. Since physical pruning requires a fixed architecture, we average the sample-dependent weights over the calibration set:

$$
\overline { { w } } _ { m , l , c } = \frac { 1 } { N _ { \mathrm { c a l } } } \sum _ { i = 1 } ^ { N _ { \mathrm { c a l } } } w _ { i , m , l , c } .\tag{24}
$$

The averaged weights are substituted into Eq. (25) to determine the final channel ranking.

## 3.5. Unified Channel Selection

With the channel contribution $C _ { q }$ from TIC and the cross-modal redundancy $R _ { q }$ from MIRA established, we now integrate them into a unified importance score. A channel should be preserved if it is highly contributive to the detection task, and penalized if it is redundantly substitutable by the other modality. Besides, we introduce a scene-conditioned weight $w _ { i , m , l , c } ,$ predicted by our SPCA module, which adaptively adjusts the redundancy penalty according to the semantic context. The final importance score for channel c in layer l of modality branch m is formulated as:

$$
S _ { i , m , l , c } = \alpha \overline { { C } } _ { m , l , c } - \beta w _ { i , m , l , c } \overline { { R } } _ { m , l , c } ,\tag{25}
$$

where $\overline { { C } } _ { m , l , c }$ and $\overline { { R } } _ { m , l , c }$ are the normalized contribution and redundancy scores, and $\alpha , \beta$ are fixed global weights. The sample-dependent $w _ { i , m , l , c } ~ \in ~ [ 1 - \gamma , 1 + \gamma ]$ modulates the redundancy penalty: $w _ { i , m , l , c } > 1$ amplifies the penalty, making the channel more likely to be pruned, while $w _ { i , m , l , c } < 1$ reduces it, allowing retention even if redundant.

With the unified importance score $S _ { i , m , l , c } ,$ , we jointly prune the RGB and infrared backbones. For each modality, channels in each layer are sorted in ascending order of their scores, with lower scores indicating higher dispensability. Under an equal-width constraint, we prune the same number of channels from both modalities within each layer pair to preserve structural compatibility for subsequent fusion modules. To maintain robustness, we pre-select the top 10% of channels with the highest contribution scores $C _ { q }$ and exclude them from pruning. The global pruning budget is then allocated across all layers via a greedy strategy. At each step, the lowest-scored available channel across all layers is pruned, and the next channel from the same layer enters the candidate pool, until the target ratio $\rho$ is met. This enables global competition across layers while maintaining per-layer RGB/IR balance. Thus, we obtain the compact model.

<table><tr><td rowspan="2">Channel Criterion</td><td colspan="3">DroneVehicle</td><td colspan="3">FLIR</td></tr><tr><td>30%</td><td>50%</td><td>70%</td><td>30%</td><td>50%</td><td>70%</td></tr><tr><td>Norm Pruning [5]  $\ell _ { 1 }$ </td><td>67.6</td><td>66.3</td><td>46.7</td><td>64.4</td><td>56.4</td><td>41.5</td></tr><tr><td>Network Slimming [12]</td><td>67.8</td><td>66.0</td><td>64.8</td><td>49.8</td><td>49.2</td><td>44.2</td></tr><tr><td>Intra-Fusion [19]</td><td>75.8</td><td>64.6</td><td>47.9</td><td>66.6</td><td>48.3</td><td>31.3</td></tr><tr><td>BilevelPruning [3]</td><td>77.2</td><td>72.3</td><td>70.4</td><td>75.6</td><td>74.8</td><td>73.5</td></tr><tr><td>REPrune [14]</td><td>67.0</td><td>63.5</td><td>59.4</td><td>55.3</td><td>53.7</td><td>53.1</td></tr><tr><td>IRPruneDeXt [27]</td><td>76.7</td><td>76.1</td><td>63.7</td><td>73.1</td><td>72.5</td><td>45.4</td></tr><tr><td>Probabilistic Pruning [4]</td><td>74.2</td><td>71.8</td><td>65.3</td><td>70.5</td><td>67.1</td><td>58.7</td></tr><tr><td>CSP [9]</td><td>78.5</td><td>76.2</td><td>72.1</td><td>76.3</td><td>73.9</td><td>68.4</td></tr><tr><td>HOT-MKS [10]</td><td>80.4</td><td>78.0</td><td>73.4</td><td>74.1</td><td>70.3</td><td>63.3</td></tr><tr><td>InterPruner (Ours)</td><td>81.0</td><td>80.3</td><td>79.0</td><td>77.5</td><td>76.7</td><td>72.6</td></tr></table>

Table 1. Comparison of mAP<sub>50</sub> (%) across different channel importance criteria and pruning ratios on the DroneVehicle and FLIR datasets. The best results are highlighted in bold.

## 4. Experiments

## 4.1. Datasets

FLIR. FLIR [26] focuses on complex outdoor driving scenarios, comprising 10,228 images (8,862 training, 1,366 testing) with annotations for Person, Car, Bicycle. It is characterized by crowded streets, significant scale variations, and cluttered backgrounds. These conditions pose a substantial challenge to the model’s ability to preserve structural details and distinguish objects in dense environments.

DroneVehicle. DroneVehicle [17] consists of 56,878 image pairs collected by UAVs, featuring an aerial perspective. It covers five vehicle categories (Car, Truck, Bus, Van, Freight-Car) and provides oriented bounding box annotations. The dataset introduces unique challenges such as small object scales, high density, and complex background textures, setting a high standard for evaluating the adaptability of multimodal detectors in aerial surveillance scenarios.

## 4.2. Implementation Details

We evaluate all models using mAP and $\mathrm { \ m A P _ { 5 0 } } .$ , and report their FLOPs and parameter counts. We implement InterPruner using the Ultralytics YOLOv8 framework with a pretrained C2DFF model [28] on NVIDIA RTX 3090 GPUs. For TIC and MIRA, we perform 8-batch adaptation with a learning rate of $1 \times 1 0 ^ { - 3 }$ . For SPCA, the learnable 16-token Qwen2.5-VL prompt is optimized using AdamW at $1 \times 1 0 ^ { - 2 }$ , with a frozen CLIP ViT-L/14 encoder providing modality-difference supervision. The pruned detector is fine-tuned with SGD at $1 \times 1 0 ^ { - 3 }$ . The balancing coefficients α and $\beta$ are both set to 1. For baselines requiring fine-tuning, we use $1 \times 1 0 ^ { - 4 }$ for 80 epochs at 30% and 50% pruning, and $1 \times 1 0 ^ { - 3 }$ for 70% pruning.

## 4.3. Main Results

We conduct extensive experiments to validate the effectiveness of our proposed InterPruner. As seldom pruning method is specifically designed for the infrared-visible dual-modality setting, we adapt existing single-modality pruning methods to serve as baselines. This allows us to perform a comprehensive analysis and ultimately evaluate the superiority of our cross-modal pruning strategy. Specifically, we compare against L1 Norm, Network Slimming [12], Intra-Fusion [19], BilevelPruning [3], REPrune [14], IRPruneDecX [19], and CSP [9].

Comparison with Single Model Pruning Criteria. As shown in Table 1, directly applying existing single-branch pruning criteria to RGB-infrared detectors leads to performance degradation, especially under large pruning ratios. Magnitude-based methods $( \ell _ { 1 }$ Norm Pruning [5], Network Slimming [12]) perform the worst in multimodal scenarios. This confirms that single-channel weights or scaling factors cannot discern cross-modal complementarity, thus inevitably remove vital branches.

![](images/19d49698fb2e9e068b7d507ee09ae467375283b5f5b9aa925bcc469e20a3fb79.jpg)  
Figure 4. Comparison with other pruning methods on the DroneVehicle dataset in terms of m $\mathrm { A P 5 0 }$ and model size.

Recent works have explored structured pruning from various perspectives. Similarity-based approaches (Bilevel-Pruning [3], REPrune [14]) achieve 72.3% on DroneVehicle dataset at 50% pruning, demonstrating moderate performance. However, they merely focus on intra-modal geometric redundancy while completely ignoring inter-modal dependencies. Gradient-based method [10] outperforms other baselines (80.4% at 30% pruning on DroneVehicle), proving that Taylor expansion effectively retains loss-sensitive channels. Yet, localized gradient estimation proves brittle under extreme pruning, as evidenced by $\mathrm { a 6 . 8 \% \ m A P _ { 5 0 } }$ drop on FLIR dataset at the 70% ratio. Reconstructionbased methods (Intra-Fusion [19]) partially compensate for pruning-induced loss through feature reconstruction, but this comes at the cost of substantial computational overhead and limited scalability to high pruning ratios. Although some of these single-modality metrics perform well at low pruning ratios, they fundamentally lack the capacity to model cross-modal complementarity and thus fail under aggressive pruning. Therefore, InterPruner jointly assesses channel importance across modalities, achieving state-ofthe-art performance across all settings and maintaining robustness even at 70% pruning.

Comparison with the Original Model under Varying Pruning Ratios Table 2 presents the performance and complexity of InterPruner across varying pruning ratios on FLIR and DroneVehicle. As pruning intensifies, parameter and FLOP counts decline substantially, yet m $\mathrm { { A P } _ { 5 0 } }$ exhibits only negligible degradation on both datasets. This underscores InterPruner’s insensitivity to specific pruning levels, achieving a flexible accuracy-efficiency trade-off and affirming the robustness of the proposed strategy.

![](images/4b1d4c9b616b15c142922fa335b8725fc97f7a51149c82ecb96e3264f21cf0e0.jpg)

Figure 5. Feature visualizations of the RGB and Infrared branches before and after pruning under extremely dark condition. The pruned model focuses on more salient regions in both modalities, showing the effectiveness of InterPruner.
<table><tr><td>Dataset</td><td>Pruning Ratio</td><td> $\mathrm { \ m A P _ { 5 0 } }$ </td><td>(%) mAP(%)</td><td>Params (M)</td><td>FLOPs (G)</td></tr><tr><td rowspan="4">FLIR</td><td>30%</td><td>77.5</td><td>41.5</td><td>4.695</td><td>11.4</td></tr><tr><td>50%</td><td>76.7</td><td>41.4</td><td>3.696</td><td>10.4</td></tr><tr><td>70%</td><td>72.6</td><td>38.7</td><td>2.767</td><td>7.9</td></tr><tr><td>-</td><td>76.8</td><td>40.8</td><td>6.580</td><td>14.6</td></tr><tr><td rowspan="4">DroneVehicle</td><td>30%</td><td>81.0</td><td>60.4</td><td>4.446</td><td>9.5</td></tr><tr><td>50%</td><td>80.3</td><td>59.7</td><td>3.415</td><td>8.1</td></tr><tr><td>70%</td><td>79.0</td><td>58.4</td><td>2.598</td><td>7.2</td></tr><tr><td>-</td><td>80.2</td><td>59.8</td><td>6.580</td><td>14.6</td></tr></table>

Table 2. Performance comparison of pruned models at different ratios and the original model.
<table><tr><td rowspan="2"></td><td rowspan="2">TIC MIRA SPCA</td><td colspan="2">30%</td><td colspan="2">50%</td><td colspan="2">70%</td></tr><tr><td> $\mathrm { \ m A P 5 0 }$ </td><td>mAP</td><td> $\mathrm { \ m A P 5 0 }$ </td><td>mAP</td><td>mAP50 mAP</td><td></td></tr><tr><td>√</td><td></td><td>72.4</td><td>39.2</td><td>70.2</td><td>38.1</td><td>62.7</td><td>33.7</td></tr><tr><td>√</td><td>V</td><td>76.3</td><td>41.2</td><td>74.4</td><td>40.0</td><td>67.1</td><td>35.4</td></tr><tr><td>√</td><td>V √</td><td>77.5</td><td>41.5</td><td>76.7</td><td>41.4</td><td>72.6</td><td>38.7</td></tr></table>

Table 3. Ablation study of the proposed components on FLIR under different pruning ratios. Results are percentage.

## 4.3.1 Component Ablations.

The ablation results are summarized in Table 3. TIC effectively estimates channel importance using high-order expansion and establishes a moderate baseline across pruning ratios. MIRA further boosts the performance and the gain grows with sparsity. We attribute this to its explicit crossmodal modeling, which prevents the removal of channels that are individually less important but provide complementary information to the other modality. Adding SPCA improves $\mathrm { \ m A P _ { 5 0 } }$ by 5.5% at 70% pruning, and the improvement widens as pruning becomes aggressive. This suggests SPCA adapts selection to scene-dependent distributions, which is critical at high sparsity, where fixed criteria often fail. Ultimately, the full model preserves high precision across pruning ratios, confirming that TIC, MIRA, and SPCA collectively address dynamic channel importance.

<table><tr><td>TIC Approximation</td><td>30%</td><td>50%</td><td>70%</td></tr><tr><td>First-order Taylor</td><td>38.2</td><td>37.9</td><td>35.1</td></tr><tr><td>Second-order Taylor</td><td>39.4</td><td>39.1</td><td>36.4</td></tr><tr><td>Third-order Taylor</td><td>41.5</td><td>41.4</td><td>38.7</td></tr><tr><td>Params (M)</td><td>4.695</td><td>3.696</td><td>2.767</td></tr></table>

Table 4. Effect of Taylor expansion order in TIC on FLIR under different pruning ratios. Results are reported in mAP(%).

## 4.3.2 Analysis of Taylor-Implicit Criterion.

We investigate the influence of different Taylor approximation orders in TIC in Table 4. First-order Taylor only captures the immediate gradient response after channel removal, and it exhibits the poorest performance across all pruning ratios, yielding only 38.2% mAP at 30% channel pruning. While introducing the second-order term effectively improves the results by considering local curvature, it remains insufficient to model the nonlinearity inherent in our compensation-aware strategy. By incorporating the third-order term, TIC better models the nonlinear loss variation caused by channel removal and parameter compensation, leading to the best performance under all pruning ratios. These results confirm that high-order Taylor expansion outperforms lower-order alternatives by effectively modeling the loss induced by pruning and compensation, ensuring reliable channel importance estimation.

## 4.3.3 Sensitivity Analysis of Hyper-parameters in Unified Channel Selection

We ablate the trade-off ratio $\beta / \alpha$ in Eq. (25) on the FLIR dataset under three distinct pruning ratios. As shown in Fig. 6, the mAP consistently peaks at $\beta / \alpha = 1 . 0$ , reaching 41.5%, 41.4%, and 38.7% respectively. This unified peak shows the optimal ratio is pruning-agnostic. In fact, deviations from this setting cause performance degradation. A small ratio ignores compensation potential, while a large one overemphasizes it and may discard vital features, especially under limited capacity. Therefore, setting $\beta / \alpha = 1 . 0$ ensures an optimal trade-off between information retention and pruning efficiency, validating that our criterion remains effective even under aggressive pruning.

## 5. Conclusion

In this paper, we propose InterPruner, an interactive structured pruning framework for RGB-infrared object detectors. The framework first assesses channel importance via a Taylor-Implicit Criterion, then identifies cross-modal redundancy through a Modality Interaction Redundancy Analyzer, and finally leverages a Scene-Prior Channel Anchor that adapts pruning to semantic context. This pruning paradigm offers new insights for lightweight multimodal perception, with natural extensions to other resourceconstrained applications, such as heterogeneous segmentation, medical imaging, and autonomous systems, where preserving cross-modal information with minimal overhead is critical.

<table><tr><td rowspan="2"> $\beta / \alpha$ </td><td colspan="2">30%</td><td colspan="2">50%</td><td colspan="2">70%</td></tr><tr><td> $\mathrm { \ m A P _ { 5 0 } }$ </td><td> $\mathrm { m A P }$ </td><td> $\mathrm { \ m A P { 5 0 } }$ </td><td>mAP</td><td> $\mathrm { \ m A P _ { 5 0 } }$ </td><td> $\mathrm { m A P }$ </td></tr><tr><td>0.0</td><td>72.4</td><td>39.2</td><td>70.2</td><td>38.1</td><td>62.7</td><td>33.7</td></tr><tr><td>0.25</td><td>73.5</td><td>39.4</td><td>72.2</td><td>38.6</td><td>67.1</td><td>35.6</td></tr><tr><td>0.50</td><td>74.3</td><td>39.8</td><td>73.6</td><td>39.0</td><td>69.7</td><td>37.2</td></tr><tr><td>1.00</td><td>77.5</td><td>41.5</td><td>76.7</td><td>41.4</td><td>72.6</td><td>38.7</td></tr><tr><td>2.00</td><td>72.0</td><td>38.5</td><td>71.2</td><td>38.2</td><td>67.5</td><td>35.8</td></tr></table>

Table 5. Ablation study of the $\beta / \alpha$ on the FLIR dataset across different pruning ratios. Results are percentage.

![](images/3989d0dc5a339eba8ceff3d00a0d5164a11acf46abcaea05cee6a74bd235db74.jpg)  
Figure 6. Visualization of the $\beta / \alpha$ ablation study in Table 5.

## References

[1] Benjamin Bergner, Christoph Lippert, and Aravindh Mahendran. Token cropr: Faster vits for quite a few tasks, 2024. 1

[2] Hongrong Cheng, Miao Zhang, and Javen Qinfeng Shi. Influence function based second-order channel pruningevaluating true loss changes for pruning is possible without retraining, 2023. 2, 3

[3] Shangqian Gao, Yanfu Zhang, Feihu Huang, and Heng Huang. Bilevelpruning: Unified dynamic and static channel pruning for convolutional neural networks. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16090–16100, 2024. 1, 2, 6, 7

[4] Kwanhee Lee and Hyang-Won Lee. Dynamic network compression via probabilistic channel pruning. Neural Networks, 193:108080, 2026. 6

[5] Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. arXiv preprint arXiv:1608.08710, 2016. 6

[6] Ting Li, Mao Ye, Tianwen Wu, Nianxin Li, Shuaifeng Li, Song Tang, and Luping Ji. Pseudo visible feature fine-grained fusion for thermal object detection. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6710–6719, 2025. 1

[7] Yawei Li, Kamil Adamczewski, Wen Li, Shuhang Gu, Radu Timofte, and Luc Van Gool. Revisiting random channel pruning for neural network compression, 2022. 2

[8] Yuxuan Li, Xiang Li, Yunheng Li, Yicheng Zhang, Yimian Dai, Qibin Hou, Ming-Ming Cheng, and Jian Yang. Sm3det: A unified model for multi-modal remote sensing object detection, 2025. 1

[9] Yang Li, Liejun Wang, and Minchi Kuang. Csp: Channel and space pruning for compressing deep convolutional neural networks. IEEE Transactions on Multimedia, 28:4146– 4157, 2026. 2, 6

[10] Suyun Lian, Yang Zhao, Jiajian Cai, Muxin Liao, Stefan Poslad, and Jihong Pei. A lightweight pruning framework with minimal retraining using taylor expansion and multiknowledge preservation strategy. Engineering Applications ofArtificial Intelligence, 168:114004, 2026. 6, 7

[11] Chang Liu, Xin Ma, Xiaochen Yang, Yuxiang Zhang, and Yanni Dong. Como: Cross-mamba interaction and offsetguided fusion for multimodal object detection, 2024. 2

[12] Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, and Changshui Zhang. Learning efficient convolutional networks through network slimming. In 2017 IEEE International Conference on Computer Vision (ICCV), pages 2755–2763, 2017. 1, 2, 6

[13] Mingge Lu, Jingwei Sun, Junqing Lin, Zechun Zhou, and Guangzhong Sun. Lua-llm: Learning unstructured-sparsity allocation for large language models. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, pages 123455–123481. Curran Associates, Inc., 2025. 1

[14] Mincheol Park, Dongjin Kim, Cheonjun Park, Yuna Park, Gyeong Eun Gong, Won Woo Ro, and Suhyun Kim.

Reprune: Channel pruning via kernel representative selection, 2024. 6, 7

[15] Loveneet Saini, Mirko Meuter, Hasan Tercan, and Tobias Meisen. Attentivegru: Recurrent spatio-temporal modeling for advanced radar-based bev object detection. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 2406–2415. IEEE, 2025. 1

[16] Jifeng Shen, Haibo Zhan, Xin Zuo, Heng Fan, Xiaohui Yuan, Jun Li, and Wankou Yang. IRDFusion: Iterative relationmap difference guided feature fusion for multispectral object detection. Pattern Recognition, 176:113189, 2026. 1, 2

[17] Yiming Sun, Bing Cao, Pengfei Zhu, and Qinghua Hu. Drone-based rgb-infrared cross-modality vehicle detection via uncertainty-aware learning. IEEE Transactions on Circuits and Systems for Video Technology, pages 1–1, 2022. 6

[18] Zhanyan Tang, Zhihao Wu, Mu Li, Jie Wen, Bob Zhang, Yong Xu, and Jianqiang Li. Adaptive fine-grained fusion network for multimodal uav object detection. IEEE Transactions on Image Processing, 35:1870–1882, 2026. 1, 2

[19] Alexander Theus, Olin Geimer, Friedrich Wicke, Thomas Hofmann, Sotiris Anagnostidis, and Sidak Pal Singh. Towards meta-pruning via optimal transport, 2024. 6, 7

[20] Kunyu Wang, Xueyang Fu, Xin Lu, Chengjie Ge, Chengzhi Cao, Wei Zhai, and Zheng-Jun Zha. Efficient test-time adaptive object detection via sensitivity-guided pruning, 2025. 2

[21] Jiaqi Wu, Zhen Wang, Enhao Huang, Kangqing Shen, Yulin Wang, Yang Yue, Yifan Pu, and Gao Huang. Bridging the rgb-ir gap: Consensus and discrepancy modeling for textguided multispectral detection, 2026. 1

[22] Lanhu Wu, Zilin Gao, Hao Fei, Mong-Li Lee, and Wynne Hsu. Leaf-mamba: Local emphatic and adaptive fusion state space model for rgb-d salient object detection. In Proceedings of the 33rd ACM International Conference on Multimedia, MM ’25, page 1013–1022. ACM, October 2025. 1

[23] Haitian Yang, Juan Fang, Yiren Zhu, et al. Crossweaver: Towards efficient cross-modal interweaving and decoupling for weakly-aligned multispectral object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, 2026. 1

[24] Maoxun Yuan, Bo Cui, Tianyi Zhao, Jiayi Wang, Shan Fu, Xue Yang, and Xingxing Wei. Unirgb-ir: A unified framework for visible-infrared semantic tasks via adapter tuning. In Proceedings of the 33rd ACM International Conference on Multimedia, MM ’25, page 2409–2418. ACM, October 2025. 2

[25] Zheng Zhan, Zhenglun Kong, Yifan Gong, Yushu Wu, Zichong Meng, Hangyu Zheng, Xuan Shen, Stratis Ioannidis, Wei Niu, Pu Zhao, and Yanzhi Wang. Exploring token pruning in vision state space models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 1

[26] Heng Zhang, Elisa Fromont, Sebastien Lef´ evre, and Bruno\` Avignon. Multispectral fusion for object detection with cyclic fuse-and-refine blocks. In 2020 IEEE International Conference on Image Processing (ICIP), pages 1–5, October 2020. 6

[27] Mingjin Zhang, Jin Feng, Handi Yang, Jie Guo, Yunsong Li, and Xinbo Gao. Irprunedext: Efficient infrared small target detection via musical wavelet-regularized channel pruning. IEEE Transactions on Neural Networks and Learning Systems, 36(12):20229–20242, 2025. 2, 6

[28] Yue Zhang, Jinbao Chen, Jianyuan Wang, Donghao Shi, Shu Han, and Lixiao Deng. C2dff-net for object detection in multimodal remote sensing images. IEEE Transactions on Geoscience and Remote Sensing, 63:1–16, 2025. 1, 2, 6

[29] Tianyi Zhao, Maoxun Yuan, Feng Jiang, Nan Wang, and Xingxing Wei. Removal then selection: A coarse-to-fine fusion perspective for rgb-infrared object detection, 2024. 2

[30] Haodong Zhu, Wenhao Dong, Linlin Yang, Hong Li, Yuguang Yang, Yangyang Ren, Qingcheng Zhu, Zichao Feng, Changbai Li, Shaohui Lin, Runqi Wang, Xiaoyan Luo, and Baochang Zhang. Wavemamba: Wavelet-driven mamba fusion for rgb-infrared object detection, 2025. 1