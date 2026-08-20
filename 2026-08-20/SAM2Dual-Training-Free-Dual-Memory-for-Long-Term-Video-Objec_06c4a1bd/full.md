# SAM2Dual: Training-Free, Dual Memory for Long-Term Video Object Segmentation

JeongRae Kim and Changwon Lim

Abstract—Long-term video object segmentation (VOS) remains challenging due to error accumulation under extended occlusions, re-appearance, and scene changes. Although SAM2 provides strong zero-shot performance, its streaming memory can amplify drift over long horizons when recent, unreliable predictions dominate the memory state. We propose SAM2Dual, a training-free, plug-and-play inference-time enhancement that improves longvideo robustness without updating model weights. SAM2Dual introduces a Dual Memory design that explicitly separates (i) short-term memory for rapid local adaptation and (ii) long-term memory built via interval-based sampling to preserve global identity cues, combined through a gated fusion strategy. In addition, we present Text-Aware Memory (TAM), which extracts a compact word-level cue from early frames and uses text embeddings to reweight memory contributions based on semantic compatibility, supporting identity preservation when visual evidence becomes weak or ambiguous. Across long-term benchmarks, SAM2Dual consistently improves stability on long videos, raising J&F from 49.33 to 50.65 on MOSEv2 and achieving consistent gains on LVOSv2.

Index Terms—Dual memory, Text-aware memory, Long-term video object segmentation

## I. INTRODUCTION

pixel-accurate masks for target objects over time. Recent progress has been driven by two complementary directions: memory-based propagation methods that store and retrieve object information across frames, and foundation models that offer strong zero-shot generalization through large-scale pretraining. While early benchmarks and methods largely focused on short clips, recent datasets and real-world scenarios increasingly involve long, unconstrained videos. In such settings, targets may undergo prolonged occlusions, abrupt changes in pose or illumination, motion blur, camera viewpoint shifts, and even scene transitions, making errors prone to accumulate over time. Consequently, robust VOS is not simply a matter of strong per-frame segmentation; it requires maintaining consistent instance identity and recovering from long-horizon drift and uncertainty.

A central challenge in long-term VOS is error accumulation. Many state-of-the-art systems operate iteratively, where predictions at time t serve as evidence for time t+1. Under partial occlusion or visual ambiguity, the tracker may drift to distractor regions or cause an identity switch. Once an incorrect mask or corrupted feature is written into memory, subsequent propagation often amplifies the error, making recovery progressively harder. This issue is particularly severe in long sequences, where the target can disappear and reappear after many frames, or where substantial appearance changes render earlier cues unreliable. Therefore, long-term robustness calls for explicit mechanisms to mitigate memory contamination, prevent unreliable evidence from dominating future inference, and preserve identity cues across large temporal gaps. An example is shown in Fig. 1, where SAM2 gradually loses the target in a long sequence with large scale and pose variations [1].

Memory-based VOS frameworks attempt to handle temporal variation by storing features or masks from previous frames and matching them to the current frame. However, practical memory designs face an inherent trade-off: recent frames provide strong local evidence for short-term adaptation but are vulnerable to transient noise such as occlusion, while older frames preserve global identity information but can become mismatched when the object appearance changes significantly. Systems that rely on a single streaming memory are particularly sensitive to this tension [2]. If the memory is dominated by recent observations, the model may overfit to short-lived cues and drift; if the memory is overly conservative, the model may fail to adapt to legitimate appearance changes. This suggests that a monolithic memory update is an incomplete solution for long videos, and that temporal evidence should be structured explicitly to balance fast adaptation and longrange stability.

Recent Segment-Anything-style video models (e.g., SAM2- style architectures) have demonstrated impressive generalization and can be deployed without per-dataset training. Nevertheless, strong zero-shot segmentation does not automatically translate into stable long-term tracking. In challenging videos, the model must repeatedly answer not only “where is the object now?” but also “which object is it?” when multiple similar instances exist, when the target reappears after a long absence, or when only partial visual evidence is available. Purely visual matching can be insufficient in such cases, motivating the use of lightweight semantic cues as an identity anchor. Even minimal word-level semantics can provide an additional constraint that complements appearance-based evidence, especially when the visual signal is weak, intermittent, or confusable.

In this work, we propose SAM2Dual, a training-free, plugand-play inference-time framework that improves robustness on long videos without updating model weights. Our approach follows two principles: explicitly structure temporal evidence to mitigate the short-term/long-term trade-off in memory, and inject lightweight semantics to support identity preservation under uncertainty. First, we introduce a Dual Memory design that separates (i) a short-term memory emphasizing recent frames for rapid local adaptation and (ii) a long-term memory that stores interval-sampled frames to preserve global identity cues over long horizons. A gating mechanism dynamically balances these memories, reducing drift while retaining flexibility under appearance changes. Second, we present a Text-Aware Memory (TAM) module that extracts compact word-level cues from early frames and uses text embeddings to reweight memory contributions based on semantic compatibility, reinforcing identity consistency when appearance cues degrade. We evaluate SAM2Dual on a diverse suite of short- and long-term VOS benchmarks and observe consistent improvements in long-video stability while maintaining competitive performance on standard shortsequence datasets.

![](images/c890f973c22ecee5212a49ddf5d1058531068683d6f41c7633946112686ca24f.jpg)  
Fig. 1. A qualitative comparison on a long sequence (over 1,300 frames) shows that, under the same first-frame mask initialization, SAM2 progressively drift and loses the target in later frames (e.g., Frames 1016 and 1300) under large scale and pose variations, whereas the proposed approach maintains consistent segmentation throughout the video, effectively suppressing long-term drift.

## II. RELATED WORKS

## A. Foundation models for promptable video segmentation.

Segment Anything Model 2 (SAM2) extends promptable segmentation from images to videos with a streaming-memory mechanism, enabling strong zero-shot performance and practical inference-time deployment [1]. By formulating video segmentation as a prompt-driven process, SAM2 serves as a strong backbone for interactive or semi-automatic VOS pipelines, where limited user input can be propagated across frames. However, long and unconstrained videos remain challenging: temporal uncertainty (e.g., prolonged occlusion, reappearance, and scene changes) can induce drift and amplify errors through iterative propagation. In particular, when recent predictions become unreliable, streaming updates may overemphasize short-lived cues, making recovery increasingly difficult over long horizons. This motivates inference-time strategies that improve long-horizon robustness without additional training or weight updates.

## B. Long-term memory design in video object segmentation.

A dominant line of VOS research improves temporal consistency by storing and retrieving object information from previous frames. Prior works explore various memory management strategies to balance robustness and efficiency in long videos. XMem proposes a long-video VOS framework inspired by the Atkinson–Shiffrin memory model, introducing interacting memory stores and a consolidation mechanism that transfers useful information into long-term memory [2]. This design aims to reduce long-horizon performance decay while controlling memory growth over extended sequences. XMem further highlights that explicitly separating short-lived evidence from sustained identity cues is beneficial for long-term robustness. DeAOT builds on AOT-style association transformers for efficient propagation [3], [4]. Despite these advances, many long-term memory designs rely on dedicated training or architectural changes, and their integration into foundationstyle promptable backbones is not always straightforward. Our work follows this principle, but targets a different setting: we propose a training-free enhancement that can be plugged into a SAM2-style backbone at inference time, improving stability without redesigning or retraining the base model.

## C. Vision–language representations for semantic anchoring.

CLIP learns aligned image–text embeddings from largescale paired data and enables zero-shot transfer by comparing visual features with textual concepts [5]. Such vision– language representations provide a lightweight semantic signal that complements appearance-based matching when visual evidence becomes weak or ambiguous. In VOS, even a compact word-level cue can act as an identity anchor, reducing confusion among similar instances and supporting recovery after long occlusions. In our setting, we adopt JINA-CLIP [6], whose Matryoshka-style training [7] encourages compact embeddings to retain semantic information, making it suitable for training-free inference-time modulation with minimal information loss. However, purely visual matching can still be insufficient when early-frame cues are weak or the target is fine-grained, motivating auxiliary semantic constraints that stabilize identity over time. Motivated by this, we leverage CLIP-style text embeddings not as a primary segmentation driver, but as an auxiliary signal that modulates memory usage to encourage identity-consistent propagation under uncertainty.

![](images/9734e13c8683e128d3681948e970af01e2380894be78f5b0d8cec4e884229834.jpg)  
Fig. 2. Overall architecture of the proposed method. Given video frames, the model reads from short-term (ST) and long-term (LT) memories, fuses the readouts, and decodes masks for each frame. Notation: m: long-term update interval; α: ST/LT fusion gate; $\lambda _ { \mathrm { w o r d } } \mathrm { : }$ text modulation strength; g: text gate.

## III. METHOD

## A. Problem Setup and Overview

Given a video sequence $\{ x _ { t } \} _ { t = 1 } ^ { T }$ , we assume a first-frame prompt mask $y _ { 1 }$ is provided for the target object. Our goal is to predict a mask sequence $\{ \hat { y } _ { t } \} _ { t = 2 } ^ { T }$ in a streaming (online) manner without any additional training. Following the SAM2 inference paradigm [1], the first-frame prompt is converted into a memory slot $m _ { 1 }$ via a memory encoder, and each subsequent frame is processed by attending to stored memory tokens to update the segmentation prediction.

For multi-object datasets, we apply SAM2Dual in an objectwise manner: each object o maintains its own memory banks $( M _ { t , o } ^ { S T } , M _ { t , o } ^ { L T } )$ and word cue $w _ { o } ^ { \ast } .$ . Final scores are obtained by averaging over objects following the standard J&F protocol [8].

We propose SAM2Dual, an inference-time plug-in framework built on top of SAM2 [1], composed of two complementary modules: (1) Dual Memory, which restructures the baseline single streaming memory into short-term (ST) and longterm (LT) banks to mitigate long-range drift; and (2) Text-Aware Memory (TAM), which injects a lightweight semantic identity cue to modulate memory usage when visual evidence is weak or ambiguous. The overall pipeline is training-free and preserves the original SAM2 backbone and decoder.

## B. Dual Memory

The key idea of Dual Memory is to separate temporal evidence into two different time scales: a short-term memory that captures rapid local appearance changes, and a longterm memory that preserves global identity cues across large temporal gaps. Let $m _ { t , o }$ denote the memory slot constructed from frame t for object o (e.g., the key/value tokens produced by the SAM2 memory encoder). At time t, we maintain two memory banks:

$$
\begin{array} { l } { M _ { t , o } ^ { S T } = \{ m _ { t - K , o } , \dots , m _ { t - 1 , o } \} , } \\ { M _ { t , o } ^ { L T } = \{ m _ { t - m , o } , m _ { t - 2 m , o } , \dots , m _ { t - L m , o } \} . } \end{array}\tag{1}
$$

Here, K is the short-term memory length, m is the longterm sampling (update) interval, and L is the maximum number of long-term slots. Importantly, both ST and LT banks are queried using the same (frozen) SAM2 memory attention module; they share identical weights and differ only in the temporal composition of stored slots. This design enables two temporal contexts without any additional training: all SAM2 parameters remain frozen, and we only reorganize the streaming memory into ST/LT banks at inference time.

1) Short-Term Memory Read: The short-term (ST) bank stores the most recent K slots. Given the query feature $q _ { t , o }$ extracted from the current frame $x _ { t }$ for object $^ { O , }$ we obtain an ST memory response via memory attention:

$$
Z _ { t , o } ^ { S T } = \mathrm { A t t n } ( q _ { t , o } , M _ { t , o } ^ { S T } ) .\tag{2}
$$

This corresponds to explicitly defining the baseline “recentframe” memory usage as a dedicated short-term bank.

2) Long-Term Memory Update and Read: The long-term (LT) bank stores temporally distant slots at a fixed interval. To control memory growth and preserve global context, the LT bank is updated only when t is a multiple of m:

$$
M _ { t + 1 , o } ^ { \mathrm { L T } } = \left\{ \begin{array} { l l } { \mathrm { U p d a t e } \left( M _ { t , o } ^ { \mathrm { L T } } , m _ { t , o } \right) , } & { t \mathrm { m o d } m = 0 , } \\ { M _ { t , o } ^ { \mathrm { L T } } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{3}
$$

The Update(·) operator appends $m _ { t , o }$ to the LT bank and retains at most L slots (e.g., oldest-first eviction). Using the same query $q _ { t , o } ,$ we read the LT response as:

$$
Z _ { t , o } ^ { L T } = \mathrm { A t t n } ( q _ { t , o } , M _ { t , o } ^ { L T } ) .\tag{4}
$$

Intuitively, $Z _ { t , o } ^ { L T }$ provides a stable identity reference from temporally distant frames, which becomes particularly useful after long occlusions or scene transitions.

3) Gated Fusion of ST/LT Responses: The final memory response $Z _ { t , o }$ is computed by a gated linear combination of the two responses:

$$
Z _ { t , o } = \alpha Z _ { t , o } ^ { S T } + ( 1 - \alpha ) Z _ { t , o } ^ { L T } , \qquad 0 \leq \alpha \leq 1 .\tag{5}
$$

Here, α controls the contribution of short-term evidence. A larger α emphasizes recent appearance cues and quick

adaptation, while a smaller α increases reliance on long-range identity context. The fused response $Z _ { t , o }$ is passed to the SAM2 decoder to produce the mask prediction $\hat { y } _ { t , o } .$

## C. Text-Aware Memory (TAM)

Dual Memory stabilizes temporal evidence structurally, but purely visual memory can still be unreliable under prolonged occlusion, severe deformation, or insufficient initial appearance cues. TAM addresses this by introducing a compact wordlevel semantic anchor and using it to modulate memory usage based on semantic compatibility.

1) TAM Word Selection Policy: We extract a representative word for each object o using only the first N frames (here $N = 5 )$ . For each frame $t \in \{ 1 , \ldots , N \}$ , we crop the target region using the bounding box of the available mask (with a 2-pixel padding on each side), and feed the crop into BLIP v1 [9]. To encourage noun-like, word-level predictions, we run BLIP in a prompt-conditioned generation mode with a fixed prefix ‘‘a photo $_ { \mathrm { o f } } , , _ { }$ , and retrieve the top-K candidate tokens for the next word along with their confidence scores $\{ ( \tilde { w } _ { t , o } ^ { ( k ) } , c _ { t , o } ^ { ( k ) } ) \} _ { k = 1 } ^ { K }$ (here $K = 5 0 )$ . We use the ground-truth mask at $t = 1$ and predicted masks for $t \geq 2$ to define the crop.

Not all candidates are informative at the word level $( \mathrm { e . g . }$ articles or numerals). We therefore apply a lightweight lexical filter based on a small set of non-informative tokens, denoted by S (e.g., a, the, one, two). Specifically, we scan the top-K candidates in descending rank order and select the first token that is not filtered out:

$$
\begin{array} { r l r } & { } & { k ^ { * } = \operatorname* { m i n } \Bigl \{ k \ : \Bigl | \ : \tilde { w } _ { t , o } ^ { ( k ) } \notin \mathcal { S } \Bigr \} , } \\ & { } & { w _ { t , o } = \tilde { w } _ { t , o } ^ { ( k ^ { * } ) } , \qquad c _ { t , o } = c _ { t , o } ^ { ( k ^ { * } ) } . } \end{array}\tag{6}
$$

We then select the final word as the one obtained from the frame with the highest confidence:

$$
\begin{array} { r l } & { t ^ { * } = \underset { t \in \{ 1 , \dots , N \} } { \mathrm { a r g m a x } } c _ { t , o } , } \\ & { w _ { o } ^ { * } = w _ { t ^ { * } , o } . } \end{array}\tag{7}
$$

The resulting cue $\boldsymbol { w } _ { o } ^ { * }$ captures a coarse semantic concept $( \mathrm { e . g . }$ person, car) beyond fine-grained instance appearance, and is fixed after frame $N .$

## 2) TAM Gate:

a) Text embedding.: We obtain a Matryoshka text embedding $\tilde { t } _ { o } ~ \in ~ \mathbb { R } ^ { d _ { \mathrm { c l i p } } }$ from JINA-CLIP [6] using the prompt template $\therefore \{ w o r d \} ^ { \prime \prime }$ . Let $d _ { k }$ denote the channel dimension of SAM2 memory keys. To match dimensions without training, we truncate the text embedding to the first $d _ { k }$ channels: $t _ { o } = \mathrm { T r u n c } _ { d _ { k } } ( \tilde { t } _ { o } )$ , then apply $\ell _ { 2 }$ normalization, $\bar { t } _ { o } = t _ { o } / \| t _ { o } \| _ { 2 }$ <sub>2</sub>, before computing cosine similarity with memory keys.

b) Semantic similarity to memory tokens.: Each memory slot $m _ { t , o }$ consists of a set of key/value tokens produced by the SAM2 memory encoder. We flatten tokens across stored frames and spatial locations, and index them by $i \in$ $\{ 1 , \ldots , N _ { \mathrm { t o k } } \}$ . For each memory token with key vector $k _ { o , i } ,$ we compute cosine similarity using the $\ell _ { 2 } \cdot$ -normalized vectors $\bar { k } _ { o , i } = k _ { o , i } / \| k _ { o , i } \| _ { 2 }$ and $\bar { t } _ { o }$ (the truncated and normalized text embedding from above):

$$
s _ { o , i } = \bar { k } _ { o , i } ^ { \top } \bar { t } _ { o } = \frac { k _ { o , i } ^ { \top } t _ { o } } { \| k _ { o , i } \| _ { 2 } \| t _ { o } \| _ { 2 } } .\tag{8}
$$

This measures how semantically compatible each memory token is with the target word concept.

c) Text-gated reweighting of memory.: We convert the similarity score into a non-negative scalar gate using ReLU and a fixed modulation strength $\lambda _ { \mathrm { w o r d } }$ . We add a constant offset to make the modulation conservative in a residual manner: the gate is always lower-bounded by 1, so TAM never attenuates memory tokens and only upweights semantically compatible ones, reverting to the baseline behavior when the text cue is uninformative (i.e., $s _ { o , i } \leq 0 )$

$$
g _ { o , i } = 1 + \lambda _ { \mathrm { w o r d } } \cdot \mathrm { R e L U } ( s _ { o , i } ) , \qquad \lambda _ { \mathrm { w o r d } } = 0 . 1 .\tag{9}
$$

The constant term 1 ensures that when the text signal is absent (or ReL $\mathrm { U } ( s _ { o , i } ) ~ = ~ 0 )$ , the original memory contribution is preserved. We then reweight the memory key (and in practice, the value as well) using the same gate:

$$
k _ { o , i } ^ { * } = g _ { o , i } k _ { o , i } , \qquad v _ { o , i } ^ { * } = g _ { o , i } v _ { o , i } .\tag{10}
$$

d) Where TAM is applied.: We apply TAM to memory tokens before attention by reweighting key/value tokens in both ST and LT banks. TAM is enabled only after the word cue is fixed, i.e., from frame $t = N + 1$ onward (the 6th frame when $N = 5 ) !$ : we re-compute $Z _ { t , o } ^ { S T }$ and $Z _ { t , o } ^ { L T }$ using the reweighted tokens, and then fuse them by Eq. (5). Finally, the fused response is decoded to obtain $\hat { y } _ { t , o } ,$ after which we update the ST/LT memory banks for the next step. The full streaming inference procedure is summarized in Algorithm 1 in the Appendix.

## D. Hyperparameters (Reference Setting)

Unless otherwise stated, we use the reference configuration in Table I. We resize each frame so that its longer side is 1024 and initialize tracking with a first-frame mask prompt. For Dual Memory, the short-term (ST) bank retains the most recent K=7 slots, while the long-term (LT) bank stores up to $L { = } 7$ slots and is updated every m=10 frames. The ST/LT readouts are fused with coefficient α=0.63 (Eq. (5)). For TAM, we extract a fixed word cue from the first N=5 frames using the top- $K _ { \mathrm { w o r d } } { = } 5 0$ BLIP candidates per frame with a lightweight lexical filter (non-informative token set S), and apply text modulation with $\lambda _ { \mathrm { w o r d } } { = } 0 . 1$ to both memory keys and values (Eqs. (9)–(10)).

TABLE I  
REFERENCE SETTING USED FOR ALL EXPERIMENTS.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Input</td><td>long side 1024, mask@t1</td></tr><tr><td>Dual Memory</td><td> $K { = } 7 , L { = } 7 , m { = } 1 0 , \alpha { = } 0 . 6 3$ </td></tr><tr><td>TAM</td><td> $N { = } 5 , K _ { \mathrm { w o r d } } { = } 5 0 , \lambda _ { \mathrm { w o r d } } { = } 0 . 1 ,$  gate K&amp;V</td></tr></table>

Compute. All experiments were conducted on a single NVIDIA Tesla V100 GPU with 40 GB VRAM. We use Python

3.12, PyTorch 2.6, and CUDA 12.8. We use the official SAM2 pretrained checkpoint sam2\_hiera\_large.pt and do not perform any additional training or finetuning. We report OOM (out-of-memory) when a method exceeds the available GPU memory under this reference setting.

## IV. RESULTS

## A. Datasets

All experiments are conducted in a training-free manner: we perform streaming inference only and never update model weights. Table II summarizes the datasets/splits and their roles in our study.

a) Main results vs. Appendix results.: We tune hyperparameters on DAVIS train and LVOSv2 train, and report main results on MOSEv2, LVOSv2 val, and PUMaVOS val following the official protocols. We additionally report Appendix results on several splits; however, some of these were obtained before finalizing the reference hyperparameters. This is because the corresponding evaluation server became unavailable after the hyperparameter sweep, preventing reevaluation with the final setting. We therefore treat these numbers as supplementary and report them separately in the Appendix.

TABLE II  
DATASETS/SPLITS AND THEIR ROLES IN OUR STUDY (ALL INFERENCE-ONLY).
<table><tr><td>Dataset / split</td><td>Role</td><td>Used in</td></tr><tr><td>DAVIS_train [8], [10]</td><td>HP tuning</td><td>tuning only</td></tr><tr><td>DAVIS_dev [8], [10]</td><td>Appendix eval</td><td>Appendix only</td></tr><tr><td>YouTube-VOS_val [11]</td><td>Appendix eval</td><td>Appendix only</td></tr><tr><td>MOSEv2 [12]</td><td>evaluation</td><td>main &amp; Appendix</td></tr><tr><td>LVOSv1_val [13] LVOSv2_train [14]</td><td>Appendix eval</td><td>Appendix only</td></tr><tr><td></td><td>HP tuning</td><td>tuning only</td></tr><tr><td>LVOSv2_val [14]]</td><td>evaluation</td><td>main &amp; Appendix</td></tr><tr><td>PUMaVOS_val [15]</td><td>evaluation</td><td>main &amp; Appendix</td></tr></table>

TABLE III

SUMMARY STATISTICS OF THE main evaluation BENCHMARKS USED FOR THE MAIN RESULTS.
<table><tr><td>Statistic</td><td>MOSEv2</td><td>LVOSv2_val</td><td>PUMaVOS_val</td></tr><tr><td>Year</td><td>2025</td><td>2024</td><td>2023</td></tr><tr><td>Avg. frames/video</td><td>154</td><td>472</td><td>883</td></tr><tr><td>Videos</td><td>433</td><td>140</td><td>24</td></tr><tr><td>Objects</td><td>575</td><td>239</td><td>26</td></tr></table>

b) Dataset characteristics.: MOSEv2 [12] consists of complex real-world videos with multiple objects, cluttered backgrounds, and frequent occlusions, where ambiguous visual evidence can lead to identity confusion. LVOSv2 [14] targets long-term VOS with long sequences, large temporal gaps, and re-appearance events, making it suitable for assessing long-horizon drift and recovery. PUMaVOS [15] provides fine-grained and challenging annotations (e.g., detailed structures/parts), serving as an additional stress test for longvideo robustness under identity ambiguity. DAVIS [8], [10] is a high-quality short-term benchmark with pixel-accurate annotations, commonly used for hyperparameter tuning under appearance changes, fast motion, and partial occlusions. YouTube-VOS [11] is a large-scale benchmark with diverse categories and videos; we report supplementary results in the Appendix. LVOSv1 [13] is an earlier long-term VOS benchmark focusing on large temporal gaps and re-appearance events, also reported in the Appendix.

For all datasets, we follow the official evaluation protocols and report results on the standard splits provided by each benchmark.

## B. Evaluation Metrics

Following common VOS practice [8], we report the combined J&F score, which averages region similarity (J) and boundary accuracy (F).

a) Region similarity (J).: For each frame, J is defined as the intersection-over-union (IoU) between the predicted mask yˆ and the ground-truth mask $y \colon$

$$
\mathcal { I } ( \hat { y } , y ) = \frac { \left| \hat { y } \cap y \right| } { \left| \hat { y } \cup y \right| } .\tag{11}
$$

b) Boundary accuracy $( F ) .$ .: F measures contour quality using the F-measure between predicted and ground-truth boundaries. Specifically, boundaries are extracted from masks and matched within a small tolerance, then precision and recall are computed to obtain:

$$
\mathcal { F } = \frac { 2 \cdot \mathrm { P r e c i s i o n } \cdot \mathrm { R e c a l l } } { \mathrm { P r e c i s i o n } + \mathrm { R e c a l l } } .\tag{12}
$$

c) Combined score (J&F).: We compute the per-frame score ${ \frac { 1 } { 2 } } ( { \mathcal { I } } + { \mathcal { F } } )$ and average across frames. For multi-object sequences, scores are additionally averaged across objects:

$$
\mathrm { J } \& \mathrm { F } = \frac { 1 } { 2 } \left( \overline { { \mathcal { I } } } + \overline { { \mathcal { F } } } \right) ,\tag{13}
$$

where · denotes averaging over frames (and objects when applicable) [8].

## C. Quantitative results on long-term benchmarks

Table IV reports the quantitative comparison (J&F) on three long-term VOS benchmarks (MOSEv2, LVOSv2, and PUMaVOS). Conventional memory-propagation methods struggle to scale to long horizons: DeAOT runs out of memory (OOM) on all three datasets, while XMem++ and Cutie show substantial degradation [4], [15], [16], [17], [18], highlighting the practical difficulty of long-video inference under limited memory budgets. With the emergence of SAM2, the evaluation focus in promptable and foundation-style VOS has largely shifted toward SAM2-centered baselines, since earlier memory-propagation approaches are less practical under longvideo settings.

Among promptable foundation-style baselines, SAM2 provides a strong zero-shot reference (49.33 on MOSEv2, 82.1 on LVOSv2, and 81.0 on PUMaVOS). Our full model, SAM2Dual, achieves the best overall stability on long videos, improving MOSEv2 from 49.33 to 50.65 and LVOSv2 from

TABLE IV  
J&F (%) ON LONG-TERM VOS BENCHMARKS (MOSEV2, LVOSV2, AND PUMAVOS; HIGHER IS BETTER). BEST IN BOLD AND SECOND BEST UNDERLINED; OOM INDICATES OUT-OF-MEMORY.
<table><tr><td>Method</td><td>MOSEv2</td><td>LVOSv2</td><td>PUMaVOS</td></tr><tr><td>DeAOT [4]</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>XMem++ [15]</td><td>36.20</td><td>62.6</td><td>62.3</td></tr><tr><td>Cutie [16]</td><td>41.90</td><td>77.7</td><td>65.9</td></tr><tr><td>SAM2 [1]</td><td>49.33</td><td>82.1</td><td>81.0</td></tr><tr><td>SAMURAI [17]</td><td>47.82</td><td>82.1</td><td>79.4</td></tr><tr><td>HiM2SAM [18]</td><td>45.90</td><td>81.7</td><td>78.5</td></tr><tr><td>SAM2Dual(ours)</td><td>50.65</td><td>83.1</td><td>80.6</td></tr></table>

82.1 to 83.1. On PUMaVOS, SAM2Dual remains competitive at 80.6, which is slightly below SAM2 (81.0) but still above SAMURAI and HiM2SAM. Notably, SAMURAI and HiM2SAM are primarily tracking-oriented variants, and although they improve temporal tracking robustness, they do not show equally strong segmentation quality on these longterm VOS benchmarks. These results indicate that restructuring streaming memory at inference time is effective for mitigating long-horizon drift while preserving strong zero-shot performance.

We attribute the gain of SAM2Dual primarily to its ability to prevent recent, unreliable predictions from saturating the memory state, thereby reducing progressive drift under long occlusions and appearance shifts. The semantic cue from TAM provides an additional constraint that is especially helpful when visual evidence is weak or confusable, improving identity consistency via semantic compatibility.

## D. Effect on long sequences (MOSEv2)

![](images/b8caddcbd26f5a18fc6fda66a67039d1c1c2cfb92243f04bba94014eea519eae.jpg)  
Fig. 3. Per-sequence F scores ordered by sequence length.

We compare SAM2 and SAM2Dual on the top 10% longest sequences of MOSEv2 (Fig. 3 and Fig. 4). Among the sequences where score differences are observed, SAM2Dual attains higher F and J scores than the original SAM2 for most sequences. In several cases, the gains are substantial, while a smaller number of sequences exhibit degradations, indicating sequence-dependent behavior on long videos. Overall, these results suggest that the proposed Dual Memory design can help preserve object representations over extended temporal horizons.

![](images/23dd8862658e6c3fdb15cbc7febef59a1bd2878e8343e7fc65b7e1c2d2033b97.jpg)  
Fig. 4. Per-sequence J scores ordered by sequence length.

## E. Dual Memory vs. TAM

TABLE V  
J&F (%) ON MOSEV2, LVOSV2 VAL, AND PUMAVOS VAL. BEST INBOLD, SECOND BEST UNDERLINED.
<table><tr><td>Model / Setting</td><td>MOSEv2</td><td>LVOSv2_val</td><td>PUMaVOS_val</td></tr><tr><td>Dual only</td><td>50.4</td><td>82.9</td><td>80.7</td></tr><tr><td>TAM only</td><td>49.33</td><td>82.3</td><td>81.0</td></tr><tr><td>Dual + TAM</td><td>50.65</td><td>83.1</td><td>80.6</td></tr></table>

Table V analyzes the complementary roles of Dual Memory and TAM. Dual-only provides the main improvement on MO-SEv2 (50.4) and LVOSv2 val (82.9), confirming that temporal structuring is the primary driver of long-horizon stability. TAM-only remains close to the SAM2 baseline on MOSEv2 (49.33) and yields a modest improvement on LVOSv2 val (82.3). On PUMaVOS val, TAM-only reaches 81.0, indicating that semantic identity cues can be helpful in this setting, although the gain is limited and does not clearly carry over to the full model. Combining both modules produces the best results on MOSEv2 and LVOSv2 val, reaching 50.65 and 83.1, respectively. These trends indicate that Dual Memory mainly improves long-horizon stability through multi-scale temporal evidence, while TAM plays a complementary role by supporting identity-consistent retrieval when visual evidence is weak or ambiguous.

Figs. 5 and 6 provide qualitative comparisons on long videos. In each figure, rows (top to bottom) correspond to SAM2, SAM2 + Dual Memory, SAM2 + TAM, and SAM2Dual, and columns show selected frames over time. Overall, SAM2 tends to exhibit progressive drift or target loss in later frames, while SAM2Dual preserves more stable masks by combining multi-scale temporal evidence (Dual Memory) with text-aware reweighting (TAM), consistent with the complementary trends in Table V.

In Fig. 5, we compare SAM2 and our variants on a long sequence containing a herd of elephants. The video involves large scale changes, frequent inter-instance occlusions, and strong perspective shifts in later frames, making identity preservation difficult. While SAM2 produces an accurate mask in early frames (Frame 1), it gradually shrinks and deforms the mask around mid-sequence (Frames 452–466) and often fails to recover the target after re-appearance, sometimes drifting to background or other instances. Dual Memory mitigates this drift by retaining long-term appearance cues across occlusions. TAM-only, however, is limited by category-level semantics (elephant) and may suffer identity switches when multiple similar instances are present. SAM2Dual combines both modules and yields the most consistent masks throughout the

![](images/7dad6997c9b6a339594b497f1a8c2771fa4987e65790448156da8654374c6cd6.jpg)  
Fig. 5. Qualitative ablation on a long elephant sequence. Rows (top to bottom): SAM2, SAM2 + Dual Memory, SAM2 + TAM, and SAM2Dual. SAM2 progressively erodes the mask and loses the target in later frames, whereas SAM2Dual maintains stable target coverage over time.

![](images/aad4e85d6ee7197fc2c18d937d25f81014fba65ad7c9b9f9422e673754ed8e36.jpg)  
Fig. 6. Qualitative ablation on a long car sequence. Rows (top to bottom): SAM2, SAM2 + Dual Memory, SAM2 + TAM, and SAM2Dual. SAM2 progressivel erodes the mask and loses the target in later frames, whereas SAM2Dual maintains stable target coverage over time.

sequence.

In Fig. 6, the racing-track sequence exhibits repeated dust occlusions caused by a preceding vehicle, yielding very limited visual cues for the target car in early frames. The SAM2 baseline becomes unstable when dust occlusion occurs before sufficient visual evidence has been accumulated, resulting in unreliable segmentation in the mid sequence. Dual-only can also fail in this setting: when the initial appearance evidence is weak, incorrect or incomplete cues can be written into both ST and LT memories and subsequently reinforced, eventually causing the tracker to lose the target. By contrast, TAM-only benefits from the generic semantic cue (car), allowing it to follow the vehicle more stably in the early stage despite weak visual evidence; however, without explicit temporal structuring of memory, its predictions can again become unstable in later frames. SAM2Dual alleviates both issues: TAM reduces earlystage errors by providing semantic anchoring under ambiguity, and Dual Memory then preserves reliable appearance evidence over time, enabling the most consistent tracking of the target car up to the final frame despite repeated dust occlusions.

## F. Hyperparameter sensitivity and reference setting

TABLE VI  
HYPERPARAMETER SWEEP ON LVOSV2 TRAIN (J&F, %). WE VARY λ<sub>word</sub> ∈ {0.10, 0.05, 0.01}, α ∈ {0.50, 0.63, 0.75}, AND m ∈ {5, 10, 15}. BEST IS IN BOLD.
<table><tr><td> $\lambda _ { \mathrm { w o r d } }$ </td><td>α</td><td> $m { = } 5$ </td><td> $m { = } 1 0$ </td><td> $m { = } 1 5$ </td></tr><tr><td rowspan="3">0.10</td><td>0.50</td><td>92.20</td><td>92.30</td><td>92.10</td></tr><tr><td>0.63</td><td>92.50</td><td>92.80</td><td>92.40</td></tr><tr><td>0.75</td><td>92.20</td><td>92.20</td><td>92.00</td></tr><tr><td rowspan="3">0.05</td><td>0.50</td><td>92.40</td><td>92.30</td><td>92.50</td></tr><tr><td>0.63</td><td>92.00</td><td>91.60</td><td>92.40</td></tr><tr><td>0.75</td><td>92.20</td><td>91.70</td><td>92.20</td></tr><tr><td rowspan="3">0.01</td><td>0.50</td><td>92.00</td><td>92.00</td><td>92.30</td></tr><tr><td>0.63</td><td>92.00</td><td>91.50</td><td>92.10</td></tr><tr><td>0.75</td><td>91.90</td><td>91.70</td><td>91.60</td></tr></table>

Baseline (original) on LVOSv2 train: $\overline { { \mathrm { J } \& \mathrm { F } = 9 2 . 3 } }$

TABLE VII  
HYPERPARAMETER SWEEP ON DAVIS TRAIN (J&F, %). WE VARY $\lambda _ { \mathrm { w o r d } } \in \{ 0 . 1 0 , 0 . 0 5 , 0 . 0 1 \} , \alpha \in \{ 0 . 5 0 , 0 . 6 3 , 0 . 7 5 \}$ , AND $m \in \{ 5 , 1 0 , 1 5 \}$ . BEST IS IN BOLD.
<table><tr><td> $\lambda _ { \mathrm { w o r d } }$ </td><td>α</td><td> $m { = } 5$ </td><td> $m { = } 1 0$ </td><td> $m { = } 1 5$ </td></tr><tr><td rowspan="3">0.10</td><td>0.50</td><td>90.60</td><td>90.80</td><td>90.60</td></tr><tr><td>0.63</td><td>90.90</td><td>90.90</td><td>90.60</td></tr><tr><td>0.75</td><td>90.90</td><td>90.60</td><td>90.50</td></tr><tr><td rowspan="3">0.05</td><td>0.50</td><td>90.60</td><td>90.80</td><td>90.60</td></tr><tr><td>0.63</td><td>90.60</td><td>90.90</td><td>90.50</td></tr><tr><td>0.75</td><td>90.90</td><td>90.60</td><td>90.50</td></tr><tr><td rowspan="3">0.01</td><td>0.50</td><td>90.60</td><td>90.80</td><td>90.50</td></tr><tr><td>0.63</td><td>90.50</td><td>90.70</td><td>90.50</td></tr><tr><td>0.75</td><td>90.80</td><td>90.60</td><td>90.20</td></tr></table>

Baseline (original) on DAVIS train: J&F = 90.5

Tables VI and VII summarize sensitivity to the Dual Memory update interval $m ,$ , the fusion gate $\alpha ,$ and the TAM modulation strength $\lambda _ { \mathrm { w o r d } }$ on LVOSv2 train and DAVIS train, respectively. Performance varies smoothly across a broad range of settings, suggesting that SAM2Dual does not rely on fragile tuning. To ensure that the proposed method maintains performance not only on long videos but also on short videos, we select hyperparameters based on both DAVIS (short-term) and LVOSv2 (long-term) training splits. We adopt the bestperforming reference setting shared across these datasets, namely $\alpha { = } 0 . 6 3 , \lambda _ { \mathrm { w o r d } } { = } 0 . 1$ , and m=10.

G. Failure case on PUMaVOS.

![](images/6ea74eb61a02d9c80c05655ccba8b5ab3556bfe343c669e288e210b968269481.jpg)  
Fig. 7. Representative failure case on PUMaVOS with a fine-grained, partlevel target (shoe), where SAM2Dual drifts and switches identity to the full person in later frames.

Fig. 7 shows a representative failure case on PUMaVOS with a fine-grained, part-level target (shoe). While SAM2+TAM remains relatively localized around the target, both variants with Dual Memory gradually drift toward the full person in later frames $( \mathrm { e . g . }$ , Frames 436 and 625). Although Dual Memory improves overall long-horizon stability, this example suggests that the current fixed periodic memory-update policy can be less robust in certain part-level scenarios. In such cases, an inaccurate intermediate prediction may be written into long-term memory and subsequently affect later retrieval, making precise recovery more difficult. This tendency appears particularly relevant for part-level targets, where a small local region can be absorbed into a larger semantically related instance such as the whole person. As a future direction, adaptive memory-update strategies—for example, confidence-aware, consistency-aware, or eventtriggered updates instead of fixed periodic writes—may further improve robustness on challenging long-horizon cases.

## V. CONCLUSION

In this paper, we presented SAM2Dual, a training-free, plug-and-play inference-time enhancement for SAM2 that improves robustness in long-term video object segmentation without updating model weights. SAM2Dual mitigates longhorizon drift and identity instability by restructuring streaming memory across temporal scales and injecting lightweight semantic cues into memory retrieval.

We introduced a Dual Memory design that separates (i) a short-term memory for rapid local adaptation and (ii) an interval-sampled long-term memory updated every m frames to preserve global identity cues, and combine them via a simple gated fusion. We also proposed Text-Aware Memory (TAM), which extracts a compact word-level cue from early frames and uses text embeddings to reweight memory keys/values based on semantic compatibility.

Experiments show that SAM2Dual consistently improves long-video stability on challenging benchmarks, raising J&F from 49.33 to 50.65 on MOSEv2 and from 82.1 to 83.1 on LVOSv2, while maintaining competitive performance on shortsequence datasets. Overall, the results suggest that long-term failures in streaming segmentation are often driven by monolithic, recent-frame-dominated memory, and that multi-scale memory structuring with semantic reweighting provides an effective training-free remedy.

## APPENDIX A

TRAINING-FREE INFERENCE PIPELINE (ALGORITHM)   
1: Input: Video frames $\{ x _ { t } \} _ { t = 1 } ^ { T } ,$ first-frame prompt mask $\{ y _ { 1 , o } \} _ { o = 1 } ^ { O } ;$   
hyperparameters $N , \alpha , K _ { \mathrm { w o r d } } , \dot { K } , L , m$   
2: Output: Predicted masks $\{ \hat { y } _ { t , o } \} _ { t = 2 } ^ { T }$ for each object o.   
3:   
4: (0) Initialize memory:   
5: for each object o do   
6: Create slot $^ { m _ { 1 , o } }$ from $( x _ { 1 } , y _ { 1 , o } ) ;$ initialize $M _ { 1 , o } ^ { S T } \gets \{ m _ { 1 , o } \}$ and   
$M _ { 1 , o } ^ { L T } \gets \{ m _ { 1 , o } \}$   
7: Initialize word-cue buffer $B _ { o } \  \ \varnothing$ and set word\_ready ←   
false.   
8: end for   
9: for $t = 2$ to $T$ do   
10: for each object o do   
11: (1) Read ST/LT (TAM off/on):   
12: Compute query ${ { q } _ { t , o } }$ from $x _ { t } .$   
13: if $\mathbf { \bar { \Psi } } _ { t } \geq \bar { N } + \mathbf { \bar { 1 } }$ and word\_ready then   
14: Compute cosine similarity $s _ { o , i }$ and gate $g _ { O , i }$ by Eqs. (8)–(9).   
15: Reweight key/value tokens by Eq. (10) for both ST and LT;   
read with reweighted tokens (TAM on).   
16: else   
17: Read with visual-only tokens (TAM off).   
18: end if   
19: $Z _ { t , o } ^ { S T }$ $\mathrm { A t t n } ( q _ { t , o } , M _ { t - 1 , o } ^ { S T } ) ;$ $Z _ { t , o } ^ { L T }$ ←   
Att $\begin{array} { r } { \mathrm { ~ \ i ( } q _ { t , o } , M _ { t - 1 , o } ^ { L T } ) . } \end{array}$   
20: (2) Fuse: $Z _ { t , o } \gets \alpha Z _ { t , o } ^ { S T } + ( 1 - \alpha ) Z _ { t , o } ^ { L T } .$   
21: (3) Decode: $\hat { y } _ { t , o } \gets \mathrm { D e c o d e r } ( Z _ { t , o } , x _ { t } ) .$   
22: (4) Collect word cue (only for early frames):   
23: ${ \mathbf i } { \mathbf f } \ t \leq N$ then   
24: Crop by the bounding box of $\hat { y } _ { t , o }$ (2-pixel padding).   
25: Obtain top- $K _ { \mathrm { w o r d } }$ word candidates from BLIP v1 [9].   
26: Apply good\_word filter (e.g., remove a, the, one, two).   
27: Set $_ { w _ { t , o } }$ as the first candidate that passes the filter (Eq. (6)).   
28: Store $( w _ { t , o } , c _ { t , o } )$ into buffer $B _ { o } .$   
29: end if   
30: if $t = N$ then   
31: Select $\boldsymbol { w } _ { o } ^ { * }$ from $B _ { o }$ by Eq. $( 7 ) .$   
32: Embed $w _ { o } ^ { * }$ into $t _ { o }$ via JINA-CLIP [6] using the prompt   
$\mathbf { \cdots } \{ \mathbf { w o r d } \} ^  \} .$   
33: Set word\_ready ← true.   
34: end if   
35: (5) Update memory:   
36: Create slot $\mathbf { \omega } _ { m _ { t , o } }$ from $( x _ { t } , \hat { y } _ { t , o } ) .$   
37: $M _ { t , o } ^ { S T } \gets M _ { t - 1 , o } ^ { S T } \cup \{ \dot { m } _ { t , o } \} ;$ keep the last K slots.   
38: if t mod $\overset { \cdot } { \underset { \cdot } { m } } = \overset { \cdot } { 0 }$ then   
39: $M _ { t , o } ^ { L T } \gets M _ { t - 1 , o } ^ { L T } \cup \{ m _ { t , o } \}$ ; keep at most L slots.   
40: else   
41: $M _ { t , o } ^ { L T } \gets M _ { t - 1 , o } ^ { L T } .$   
42: end if   
43: end for   
44: end for   
Algorithm 1. SAM2Dual inference pipeline (training-free).

## APPENDIX B

## ADDITIONAL BENCHMARK RESULTS UNDER A PREVIOUS SETTING

Our main tables report results under a single reference configuration for fair comparison. For some benchmarks, evaluation under the current reference setting was not feasible due to dataset/benchmark constraints in our environment. For completeness, we provide the results obtained with a previous configuration. These numbers are reported for reference and are not directly comparable to the main tables.

TABLE VIII  
J&F (%) COMPARISON ON VOS BENCHMARKS UNDER A PREVIOUS SETTING.
<table><tr><td>Method</td><td>DAVIS</td><td>YouTube-VOS</td><td>MOSEv2</td><td>LVOSv1</td><td>LVOSv2</td><td>PUMaVOS</td></tr><tr><td>DeAOT [4]</td><td>82.19</td><td>86.25</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>XMem++ [15]</td><td>80.94</td><td>84.00</td><td>36.20</td><td>44.13</td><td>62.6</td><td>62.3</td></tr><tr><td>Cutie [16]</td><td>86.37</td><td>86.15</td><td>41.90</td><td>66.65</td><td>77.7</td><td>65.9</td></tr><tr><td>SAM2 [1]</td><td>88.59</td><td>88.81</td><td>49.33</td><td>76.72</td><td>82.1</td><td>81.0</td></tr><tr><td>SAMURAI [17]</td><td>88.98</td><td>88.51</td><td>47.82</td><td>75.90</td><td>82.1</td><td>79.4</td></tr><tr><td>HiM2SAM [18]</td><td>88.77</td><td>88.64</td><td>45.90</td><td>67.31</td><td>81.7</td><td>78.5</td></tr><tr><td>TAM</td><td>88.80</td><td>88.82</td><td>49.39</td><td>76.33</td><td>82.1</td><td>81.1</td></tr><tr><td>Dual Memory</td><td>89.34</td><td>88.50</td><td>50.40</td><td>78.94</td><td>82.9</td><td>80.7</td></tr><tr><td>SAM2Dual</td><td>89.40</td><td>88.56</td><td>50.57</td><td>80.41</td><td>83.0</td><td>80.8</td></tr></table>

Bold: best; Underline: second best. OOM indicates out-of-memory.

## REFERENCES

[1] Ravi, N., Gabeur, V., Hu, Y. T., Hu, R., Ryali, C., Ma, T., ... & Feichtenhofer, C. (2024). Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714.

[2] H. K. Cheng and A. G. Schwing, “XMem: Long-term video object segmentation with an Atkinson–Shiffrin memory model,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2022, pp. 640–658.

[3] Z. Yang, Y. Wei, and Y. Yang, “Associating objects with transformers for video object segmentation,” in Adv. Neural Inf. Process. Syst., vol. 34, 2021.

[4] Z. Yang and Y. Yang, “Decoupling features in hierarchical propagation for video object segmentation,” in Adv. Neural Inf. Process. Syst., vol. 35, 2022, pp. 36324–36336.

[5] A. Radford et al., “Learning transferable visual models from natural language supervision,” in Proc. Int. Conf. Mach. Learn. (ICML), vol. 139, 2021, pp. 8748–8763.

[6] A. Koukounas et al., “jina-clip-v2: Multilingual multimodal embeddings for text and images,” arXiv:2412.08802, 2024.

[7] A. Kusupati et al., “Matryoshka representation learning,” in Adv. Neural Inf. Process. Syst., vol. 35, 2022, pp. 30233–30249.

[8] F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, and A. Sorkine-Hornung, “A benchmark dataset and evaluation methodology for video object segmentation,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 724–732.

[9] J. Li, D. Li, C. Xiong, and S. Hoi, “BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation,” in Proc. Int. Conf. Mach. Learn. (ICML), 2022, pp. 12888–12900.

[10] J. Pont-Tuset, F. Perazzi, S. Caelles, P. Arbelaez, A. Sorkine-Hornung,´ and L. Van Gool, “The 2017 DAVIS Challenge on video object segmentation,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), 2017.

[11] N. Xu, L. Yang, Y. Fan, J. Yang, D. Yue, Y. Liang, B. Price, S. Cohen, and T. S. Huang, “YouTube-VOS: Sequence-to-sequence video object segmentation,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2018, pp. 603– 619.

[12] H. Ding, K. Ying, C. Liu, S. He, X. Jiang, Y.-G. Jiang, P. H. S. Torr, and S. Bai, “MOSEv2: A more challenging dataset for video object segmentation in complex scenes,” arXiv:2508.05630, 2025.

[13] L. Hong, W. Chen, Z. Liu, W. Zhang, P. Guo, Z. Chen, and W. Zhang, “LVOS: A benchmark for long-term video object segmentation,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 13480–13492.

[14] L. Hong et al., “LVOS: A benchmark for large-scale long-term video object segmentation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 1, pp. 946–961, Jan. 2026, doi: 10.1109/TPAMI.2025.3611020.

[15] M. Bekuzarov, A. Bermudez, J.-Y. Lee, and H. Li, “XMem++: Production-level video segmentation from few annotated frames,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 635–644.

[16] H. K. Cheng, S. W. Oh, B. Price, J.-Y. Lee, and A. Schwing, “Putting the object back into video object segmentation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 3151–3161.

[17] C.-Y. Yang, H.-W. Huang, W. Chai, Z. Jiang, and J.-N. Hwang, “SAMU-RAI: Adapting Segment Anything Model for zero-shot visual tracking with motion-aware memory,” arXiv:2411.11922, 2024.

[18] R. Chen, G. Sun, Y. Li, J. Qin, and L. Benini, “HiM2SAM: Enhancing SAM2 with hierarchical motion estimation and memory optimization towards long-term tracking,” arXiv:2507.07603, 2025.