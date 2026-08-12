# FADE: From Passive Verification to Active Discovery in Counterfactual Video Understanding

Fufangchen Zhao<sup>1</sup> <sup>2</sup>, Jinhu Fu<sup>1</sup> <sup>2</sup>, Jiachen Lei<sup>2</sup>, Jiahong Wu<sup>2∗</sup>, Xiangxiang Chu<sup>2</sup>, Danfeng Yan<sup>1†</sup> <sup>1</sup>State Key Laboratory of Networking and Switching Technology, BUPT <sup>2</sup> AMAP, Alibaba Group zhaofufangchen@bupt.edu.cn

![](images/1387e32e4a9f3be4ce3dce29a3c82a11f7d1617af032f05dc2a9f09a2d415bbd.jpg)  
(a) Illustrative case

![](images/4b8ee191d14db9c9b9f7cc0d57a12eddc09e491ba3dd1606bd1bf2e08b810400.jpg)

![](images/8cfcb9a3b38d053a084fed124c7765c21b5c8631b7f7160a8c16be430623af58.jpg)  
(b) Aggregate results  
Figure 1: Performance comparison between FADE and its base model as instance-specific textual anchors progressively fade. (a) Both models answer the anchor-rich MCQ correctly; after the options and then the question-specific cue are removed, only FADE consistently discovers and explains the counterfactual event. (b) Aggregate results on DualityVidQA-test and IPV-Bench. Bars report absolute scores, while lines report performance retained relative to MCQ.

## Abstract

Counterfactual video understanding evaluates whether models grasp physical and commonsense regularities. However, existing multiple-choice question (MCQ) benchmarks inadvertently leak target events through their questions and candidate options. This reduces the core challenge from active discovery to text-guided verification.

In this paper, we present FADE, an efective training framework for counterfactual discovery and explanation. Our method is built on an evidence-first, two-stage training paradigm. First, evidence-internalized supervised fine-tuning grounds the model’s predictions in decisive visual anomalies. Second, we apply a fading-anchor reinforcement learning strategy that progressively removes textual guidance, compelling the model to independently discover and explain evidence. To rigorously evaluate this capability, we also introduce an efective pipeline that converts existing MCQ datasets into aligned MCQ, open-ended question answering (OQA), and captioning tasks without requiring additional data curation.

Our simple approach yields strong results. Using Qwen3-VL-8B as the baseline, FADE achieves state-of-the-art strict paired scores across all three tasks on DualityVidQA-test and IPV-Bench, outperforming GPT-5.6. In specific, when transitioning from constrained MCQs to unconstrained OQA and captioning, our model demonstrates remarkable robustness. Its performance retention is 90.4% and 67.4% on DualityVidQAtest—substantially higher than the 48.1% and 30.7% retained by GPT-5.6. We hope this simple framework can serve as a solid baseline for future research in unconstrained counterfactual video understanding.

## Introduction

Counterfactual video understanding investigates whether video-based multimodal large language models (Video-MLLMs) capture the physical and commonsense rules of dynamic environments (Huang et al. 2026; Chen et al. 2025; Motamed et al. 2025). However, most existing benchmarks rely on multiple-choice questions (MCQ) (Huang et al. 2026; Bai, Ci, and Shou 2025). This formulation provides strong textual priors. The question and candidate options consequently act as instance-specific textual anchors that specify the event to inspect, turning an inherently open-ended discovery problem into text-guided visual verification. High MCQ accuracy may therefore overestimate counterfactual understanding: a model can confirm a supplied hypothesis without autonomously determining what is anomalous. In this paper, we ask a simple question: Can a Video-MLLM independently discover, localize, and explain a counterfactual event without being told what to lookfor?

To expose this overestimation, we progressively reduce instance-specific textual guidance from MCQ with a question and candidate options, to OQA with the question alone, and finally to captioning with only a generic instruction, as illustrated in Figure 1(a). Existing Video-MLLMs deteriorate sharply along this progression. On DualityVidQA-test (Huang et al. 2026), GPT-5.6 (Singh et al. 2025) falls from 84.0 in MCQ to 40.4 in OQA and 25.8 in captioning, while Qwen3-VL-8B (Bai et al. 2025a) drops from 57.3 to 27.3 and 16.2. Figure 2 reveals the same trend across representative open- and closed-source models, demonstrating that high MCQ scores can mask a shared dependence on text-specified hypotheses.

To bridge this gap, we introduce FADE (Fading Textual Anchors for Counterfactual Discovery and Explanation), an evidence-first, two-stage training framework that shifts counterfactual video understanding from text-guided verification toward video-driven discovery and explanation. In Stage I, evidence-internalized supervised fine-tuning (SFT) teaches the model to discover and temporally localize decisive counterfactual evidence. Building on this grounding, Stage II employs fading-anchor reinforcement learning (RL) to preserve evidence-based interpretation as questions and candidate options are progressively removed. As shown in Figure 1(b), training with FADE raises the OQA and captioning retention rates of Qwen3-VL-8B from 47.6% and 28.3% to 90.4% and 67.4%, respectively, with similarly substantial gains on IPV-Bench (Bai, Ci, and Shou 2025).

To evaluate this capability without constructing a new test set, we additionally develop an evaluation protocol that operates directly on existing public MCQ benchmarks. It preserves each video and its semantic target while reformulating the original query into three independently evaluated formats: MCQ with the question and candidate options, OQA with the question alone, and captioning with only a generic description instruction. Task-compatible verifiers then assess whether counterfactual judgments persist as instance-specific textual guidance is progressively removed.

Our contributions are threefold:

• We identify textual anchoring as an under-explored problem in counterfactual video evaluation and distinguish passive verification from active discovery.

• We propose FADE, an evidence-first training framework that combines evidence-internalized SFT with fadinganchor RL to support autonomous counterfactual discovery and explanation.

• We develop a reusable evaluation protocol that reformulates MCQ instances from existing public benchmarks into progressively less guided formats without building a new benchmark from scratch. Extensive experiments expose the textual-anchor dependence of current Video-MLLMs and demonstrate the robustness of the FADEtrained model.

## Related Works

## Counterfactual Video Understanding

Counterfactual video understanding probes whether models have genuinely acquired the real-world regularities underlying video events by exposing them to physical violations, causal interventions, or conflicts with commonsense knowledge. Early studies primarily followed the violationof-expectation paradigm, assessing intuitive physical understanding by requiring models to distinguish physically plausible events from impossible ones (Riochet et al. 2018; Bordes et al. 2025).Subsequent work, exemplified by CLEVRER and CoPhy, shifted the focus beyond event-plausibility judgment toward reasoning about temporal dynamics, causal dependencies, and counterfactual outcomes. This line of research was later extended to explaining violations of physical expectations and performing evidence-grounded and commonsense reasoning in complex real-world videos (Yi et al. 2019; Baradel et al. 2019; Dai et al. 2023; Li, Niu, and Zhang 2022). With the emergence of multimodal large language models for video understanding, recent studies have introduced broader benchmarks covering diverse violations of real-world regularities and have improved anomalous-event reasoning through subproblem decomposition, paired real and counterfactual videos, and causal-graph-guided training (Zhou et al. 2025; Bai, Ci, and Shou 2025; Huang et al. 2026; Chen et al. 2025). Nevertheless, most existing tasks provide hypotheses, targeted questions, or candidate answers that specify in advance which events the model should inspect and verify. Consequently, whether a model can move beyond explicit textual guidance to autonomously discover, localize, and explain anomalous content directly from video remains largely unexplored.

## Textual Bias in Multimodal Reasoning

Multimodal models often exploit linguistic priors arising from question formulations, answer distributions, and textual co-occurrence patterns, rather than grounding their predictions in the visual input. Early VQA studies (Agrawal et al. 2018; Cadene et al. 2019; Niu et al. 2021) exposed this vulnerability by modifying question-answer priors and subsequently sought to mitigate it through unimodal bias modeling and counterfactual causal inference, thereby reducing the direct influence of question semantics on answer prediction. This line of inquiry later expanded to video understanding.

![](images/be66351e5f4c6d84511402920714aed4f9d13d1adf9949ffd02fd10c3e5379f3.jpg)  
Figure 2: Anchor-fading profiles of representative Video-MLLMs on DualityVidQA-test (Both) under our evaluation protocol. Each panel shows one model; bars report absolute scores and lines report retention relative to MCQ. The consistent decline from MCQ to OQA and captioning reveals strong reliance on instance-specific textual anchors.

Through complementary diagnostic techniques, including visual evidence localization, modality ablation, and minimal video pairs, prior work (Li et al. 2022; Xiao et al. 2024; Krojer et al. 2025; Lim et al. 2026) has shown that high question-answering accuracy does not necessarily imply reliable video understanding: model predictions may still be driven by linguistic correlations, static visual cues, or temporally irrelevant segments rather than the evidence needed to support the answer. Moreover, TemporalBench (Cai et al. 2024) and recent analyses of bias in multiple-choice video question answering (Loginova et al. 2025) suggest that linguistic regularities and positional imbalances among answer options can introduce additional answer-selection shortcuts. Nevertheless, existing studies have largely examined whether models can ignore visual inputs or exploit dataset-level statistical biases. Much less attention has been paid to a subtler form of textual anchoring: even when a model processes the video, the question and candidate answers may predetermine what it searches for, turning open-ended anomaly discovery into the passive verification of an explicitly suggested phenomenon.

## Visual Grounding and Reinforcement Learning in Video MLLMs

Visual grounding aligns model predictions with relevant video segments, thereby grounding video reasoning in verifiable spatiotemporal evidence. NExT-GQA (Xiao et al. 2024) pioneered the joint evaluation of answer prediction and temporal evidence localization, while more recent approaches, such as Grounded-VideoLLM (Wang et al. 2024) and ED-VTG (Pramanick et al. 2025), strengthen fine-grained temporal grounding in Video MLLMs through explicit temporal representations and query-semantic enrichment. Following the introduction of GRPO (Shao et al. 2024), reinforcement learning has been increasingly adopted for video-model posttraining. Video-R1 (Feng et al. 2026) and VideoChat-R1 (Li et al. 2025) improve video understanding by targeting temporal reasoning and multi-task spatiotemporal perception, respectively, whereas Time-R1 (Wang et al. 2026) jointly optimizes reasoning and localization with verifiable temporal rewards. Related eforts further incorporate curriculum reinforcement learning to localize violation segments, as in RAVEN (Ji et al. 2025), or combine paired factual and counterfactual videos with a two-stage SFT–RL pipeline to mitigate language-prior dependence, as in DualityVidQA (Huang et al. 2026). Nevertheless, these methods typically optimize grounding or answer correctness under a fixed textual query, which specifies the event to be searched for in advance. In contrast, FADE progressively fades the guidance provided by questions and candidate answers, encouraging the model to move beyond query-conditioned evidence verification toward video-driven discovery and explanation of counterfactual events.

## FADE

As illustrated in the left and center panels of Figure 3, FADE is an evidence-first, two-stage training framework. Stage I uses evidence-internalized supervised fine-tuning (SFT) to make decisive counterfactual evidence discoverable from the model’s response representations. Building on this grounding, Stage II applies fading-anchor reinforcement learning (RL) to preserve evidence-based interpretation and explanation as instance-specific textual guidance is progressively reduced. The right panel summarizes the separate evaluation protocol described in Section Experiments; it is used only for evaluation and does not participate in FADE optimization.

## Stage I: Evidence-Internalized SFT

Given a sample $\left( V _ { i } , X _ { i } , Y _ { i } , z _ { i } , { \mathcal { T } } _ { i } \right)$ $V _ { i }$ is the video, $X _ { i }$ the textual input, $Y _ { i } = ( y _ { i , 1 } , \dots , y _ { i , L _ { i } } )$ the target response, and $z _ { i } \in \{ 0 , 1 \}$ the factual/counterfactual label, where $z _ { i } = 1$ denotes a counterfactual video. For $z _ { i } = 1$ , the normalized interval ${ \mathcal { T } } _ { i } = [ a _ { i } , b _ { i } ]$ contains the decisive event; otherwise $\mathcal { T } _ { i } = \mathcal { O }$ . We retain the autoregressive negative log-likelihood

$$
\mathcal { L } _ { \mathrm { N L L } } = - \mathbb { E } _ { i } \left[ \frac { 1 } { L _ { i } } \sum _ { t = 1 } ^ { L _ { i } } \log p _ { \theta } ( y _ { i , t } \mid V _ { i } , X _ { i } , y _ { i , < t } ) \right] ,\tag{1}
$$

where $p _ { \theta }$ is the model distribution. This term preserves response correctness and fluency, but alone does not determine which visual evidence supports the response.

![](images/d955e4d002ecfd8a051033dfb9ebf82f5136761b13a80832a284158e3f8b33bb.jpg)  
Figure 3: Overview of FADE and the separate anchor-fading evaluation protocol. FADE comprises evidence-internalized SFT for discovering and grounding anomalous evidence (left) and fading-anchor RL for interpreting and explaining it under reduced textual guidance (center). Separately, the protocol reformulates each public MCQ benchmark instance into aligned MCQ, OQA, and captioning evaluations with task-compatible verifiers (right), requiring no new test set.

Localized evidence projection. As illustrated by the evidence-projection branch in the left panel of Figure $^ { 3 , }$ we use $\mathcal { T } _ { i }$ only as privileged training supervision. Let $H _ { i } ^ { v } \ =$ $\left[ \mathbf { v } _ { i , 1 } , \ldots , \mathbf { v } _ { i , M _ { i } } \right]$ be the temporally ordered visual tokens, $\tau _ { i , m }$ the timestamp of $\mathbf { v } _ { i , m } .$ , and $S _ { i } \overset { \cdot } { = } \{ m \mid a _ { i } \leq \tau _ { i , m } < b _ { i } \}$ The target evidence prototype is

$$
\bar { \mathbf { v } } _ { i } = \left\{ \begin{array} { l l } { | S _ { i } | ^ { - 1 } { \sum _ { m \in { \mathcal { S } } _ { i } } \mathbf { v } _ { i , m } } , } & { z _ { i } = 1 , } \\ { M _ { i } ^ { - 1 } { \sum _ { m = 1 } ^ { M _ { i } } \mathbf { v } _ { i , m } } , } & { z _ { i } = 0 . } \end{array} \right.\tag{2}
$$

Thus, counterfactual responses are tied to the annotated event, whereas a global prototype avoids imposing an artificial anomaly location on factual videos. For response states $H _ { i } ^ { y } = [ \mathbf { h } _ { i , 1 } ^ { y } , \ldots , \mathbf { h } _ { i , L _ { i } } ^ { y } ]$ , we define

$$
\mathcal { L } _ { \mathrm { p r o j } } = \mathbb { E } _ { i } \left[ \frac { 1 } { L _ { i } } \sum _ { t = 1 } ^ { L _ { i } } \left. g _ { \phi } ( \mathbf { h } _ { i , t } ^ { y } ) - \mathrm { n g } ( \bar { \mathbf { v } } _ { i } ) \right. _ { 2 } ^ { 2 } \right] ,\tag{3}
$$

where $g _ { \phi }$ is a lightweight projector and ng stops gradients. This makes the decisive visual content recoverable from the response representation rather than allowing a solution based only on textual cues.

Response-conditioned evidence re-grounding. Evidence projection specifies what should support the response, but inference must recover that evidence without access to $\mathcal { T } _ { i }$ . The second blue branch in Figure 3, termed RCER, transfers interval supervision into whole-video retrieval. A span-guided branch first uses K learnable queries $Q \in \mathbb { R } ^ { K \times d }$ to produce a privileged content-and-position target:

$$
( E _ { i } ^ { g } , P _ { i } ^ { g } ) = \mathrm { A t t n } ( Q , H _ { i , S _ { i } } ^ { v } , H _ { i , S _ { i } } ^ { v } ) ,\tag{4}
$$

where $E _ { i } ^ { g } \in \mathbb { R } ^ { K \times d }$ encodes evidence content and $P _ { i } ^ { g }$ its temporal distribution. Factual videos instead use the global representation and a uniform distribution. A response-conditioned branch, which never observes $\mathcal { T } _ { i } .$ , retrieves from the complete video:

$$
( E _ { i } ^ { r } , P _ { i } ^ { r } ) = \mathrm { A t t n } ( W _ { q } \widetilde { H } _ { i } ^ { y } , H _ { i } ^ { v } , H _ { i } ^ { v } ) ,\tag{5}
$$

where $\widetilde { H } _ { i } ^ { y }$ contains K sampled and projected response states, and $W _ { q }$ maps them into retrieval queries. We align the response states, retrieved content, and temporal distributions through

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { R C E R } } = \mathbb { E } _ { i } \Big [ \alpha _ { 1 } \lVert E _ { i } ^ { g } - \widetilde { H } _ { i } ^ { y } \rVert _ { 1 } + \alpha _ { 2 } \lVert E _ { i } ^ { r } - \mathrm { s g } ( E _ { i } ^ { g } ) \rVert _ { 1 } } \\ & { \qquad + \alpha _ { 3 } D _ { \mathrm { K L } } \big ( \mathrm { s g } ( P _ { i } ^ { g } ) \lVert P _ { i } ^ { r } \big ) \Big ] . } \end{array}\tag{6}
$$

The three terms respectively internalize the privileged evidence, recover its content from the full video, and distill its temporal location. Stage I optimizes

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S F T } } = \mathcal { L } _ { \mathrm { N L L } } + \lambda _ { p } \mathcal { L } _ { \mathrm { p r o j } } + \lambda _ { e } \mathcal { L } _ { \mathrm { R C E R } } , } \end{array}\tag{7}
$$

where $\lambda _ { p }$ and $\lambda _ { e }$ balance the auxiliary objectives. Temporal annotations and auxiliary branches are used only during training; the evidence-internalized checkpoint initializes Stage II.

## Stage II: Fading-Anchor RL

Although Stage I grounds the response in visual evidence, the textual input may still reveal which event should be examined. The center panel of Figure 3 therefore progressively broadens the evidence-search space through three views of the same video:

$$
\mathcal { A } _ { i } ^ { ( 1 ) } = ( q _ { i } , \mathcal { O } _ { i } ) , \qquad \mathcal { A } _ { i } ^ { ( 2 ) } = q _ { i } , \qquad \mathcal { A } _ { i } ^ { ( 3 ) } = \emptyset ,\tag{8}
$$

corresponding to MCQ, OQA, and captioning. Here, $\mathcal { A } _ { i } ^ { ( 3 ) } =$ ∅ removes all instance-specific anchors while retaining a generic video-description instruction. We organize these views into a scafold-and-consolidate curriculum

$$
\mathcal { C } _ { 1 } : 1 \Rightarrow 2 \Rightarrow 3 , \qquad \mathcal { C } _ { 2 } : 2 \Rightarrow 3 , \qquad \mathcal { C } _ { 3 } : 3 ,\tag{9}
$$

where ⇒ is a correctness-gated transition. $\mathcal { C } _ { 1 }$ uses stronger anchors to scafold weaker-anchor behaviors; its checkpoint initializes ${ \mathcal { C } } _ { 2 } ,$ and subsequently ${ \mathcal { C } } _ { 3 } ,$ each beginning from a clean context. The model therefore first learns each harder behavior and then consolidates it after the preceding textual cues are removed.

Trajectory rewards and policy optimization. Let $c _ { i , k } ^ { ( \ell ) } \in$ $\{ 0 , 1 \}$ indicate whether trajectory k for sample i passes the verifier at level ℓ. MCQ uses exact matching, whereas OQA and captioning use task-adapted semantic verification. For curriculum ${ \mathcal { C } } _ { r } ,$ the prefix-gated progress reward is

$$
R _ { i , k } ^ { \mathrm { p r o g } } = \sum _ { \ell = r } ^ { 3 } \prod _ { j = r } ^ { \ell } c _ { i , k } ^ { ( j ) } ,\tag{10}
$$

so a weaker-anchor success is rewarded only if all preceding levels are also correct.

To discourage an “always counterfactual” shortcut, each counterfactual video $V _ { i } ^ { + }$ is paired with its factual counterpart $V _ { i } ^ { - }$ . Let $d _ { r } ( \widehat { Y } ) \in \{ 0 , 1 \}$ be the predicted direction under the starting format of ${ \mathcal { C } } _ { r } ,$ where 1 denotes counterfactual. We define

$$
R _ { i , k } ^ { \mathrm { p a i r } } = \mathbb { I } \Bigl [ d _ { r } ( \widehat { Y } _ { i , k } ^ { + } ) = 1 \ \wedge \ d _ { r } ( \widehat { Y } _ { i , k } ^ { - } ) = 0 \Bigr ] ,\tag{11}
$$

which is positive only when both directions are correct. A lightweight $R _ { i , k } ^ { \mathrm { f m t } }$ additionally ensures parseable responses.

Because the progress, pair, and format signals difer in scale and sparsity, we adopt GDPO (Liu et al. 2026) to normalize them independently. For G trajectories sampled from prompt i, the advantage is

$$
A _ { i , k } = \mathrm { N o r m } _ { B } \left[ \sum _ { c \in \{ \mathrm { p r o g , p a i r , f m t } \} } w _ { c } \frac { R _ { i , k } ^ { c } - \mu _ { i } ^ { c } } { \sigma _ { i } ^ { c } + \epsilon } \right] ,\tag{12}
$$

where $\mu _ { i } ^ { c }$ and $\boldsymbol { \sigma } _ { i } ^ { c }$ are the group statistics of component $c , w _ { c }$ is its weight, and Norm denotes batch whitening. We then apply the standard KL-regularized clipped objective with the Stage-I model as the reference policy. Thus, SFT establishes discoverable visual evidence, while RL preserves its use as textual anchors fade; neither temporal annotations nor paired videos are required at inference.

## Experiments

## Evaluation Protocol

Existing counterfactual-video benchmarks are predominantly formulated as MCQs, whose questions and candidate options specify both what to inspect and which hypotheses to verify. To diagnose this textual anchoring without collecting a new test set, we reformulate each MCQ instance from an existing public benchmark into three semantically aligned evaluations with progressively less instance-specific guidance, as shown in the right panel of Figure 3.

Given a source instance $\bar { B _ { i } } = \bar { ( } V _ { i } , q _ { i } , \bar { O _ { i } } , \bar { a _ { i } ^ { \star } } )$ , let $e _ { i } ^ { \star }$ denote the target-event semantics jointly specified by $( q _ { i } , o _ { i , a _ { i } ^ { \star } } )$ . We construct

$$
\begin{array} { r l } & { \mathcal { U } _ { i } ^ { \mathrm { M C Q } } = ( V _ { i } , q _ { i } , \mathcal { O } _ { i } ) , } \\ & { \mathcal { U } _ { i } ^ { \mathrm { O Q A } } = ( V _ { i } , q _ { i } ) , } \\ & { \mathcal { U } _ { i } ^ { \mathrm { C A P } } = ( V _ { i } , p _ { \mathrm { c a p } } ) . } \end{array}\tag{13}
$$

where $p _ { \mathrm { c a p } }$ is a generic description instruction shared across all samples. Thus, instance-specific anchors fade as $( q _ { i } , { \mathcal { O } } _ { i } ) \to q _ { i } { \bf \bar { \mu } } \to \emptyset$ . The three formats preserve the original video, target semantics, public split, and visual sampling budget, and are evaluated independently from clean contexts.

For an arbitrary evaluated model $f _ { \psi } ,$ , correctness and the reported score at level $\ell \in \{ \mathrm { M C Q } , \dot { \mathrm { O Q A } } , \mathrm { C A P } \}$ are

$$
\begin{array} { l } { { \displaystyle c _ { i } ^ { ( \ell ) } = \mathcal { I } _ { \ell } \Big ( f _ { \psi } ( \mathcal { U } _ { i } ^ { ( \ell ) } ) ; a _ { i } ^ { \star } , e _ { i } ^ { \star } , q _ { i } \Big ) \in \{ 0 , 1 \} , } } \\ { { \displaystyle S ^ { ( \ell ) } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } c _ { i } ^ { ( \ell ) } } . } \end{array}\tag{14}
$$

Here, $\mathcal { I } _ { \mathrm { M C Q } }$ uses exact option matching, $\mathcal { I } _ { \mathrm { { O Q A } } }$ uses a frozen semantic verifier, and $\mathcal { I } _ { \mathrm { C A P } }$ checks whether the target event is independently discovered and correctly described. The resulting profile $\mathbf { \bar { F } } _ { \psi } = ( S ^ { ( \mathrm { M C Q } ) } , S ^ { ( \mathrm { O Q A } ) } , S ^ { ( \mathrm { C A P } ) } )$ ) diagnoses whether video-groundedjudgments persist as textual anchors fade. Because the output space also becomes increasingly open-ended, cross-format gaps are interpreted diagnostically rather than as strict causal efects.

## Experimental Setup

Training Data. We directly use the DualityVidQA (Huang et al. 2026) training split. The SFT data contain 104,879 samples, comprising 54,879 factual and 50,000 counterfactual videos, while the RL data contain 20,000 factual– counterfactual video pairs. Because the original training annotations are MCQ-only, we supplement them with OQA answers and captions generated by Qwen3.6-Plus (Qwen Team 2026) using each question and correct option as semantic context. These references support FADE’s fading-anchor curriculum while preserving the original MCQ supervision.

Implementation Details. We adopt Qwen3-VL-8B (Bai et al. 2025a) as the base model and implement the SFT and RL phases using the LLaMA-Factory (Zheng et al. 2024) and ms-swift (Zhao et al. 2024) frameworks, respectively. The SFT phase is fine-tuned for two epochs using LoRA with a rank of $^ { \cdot } 8 ,$ with a learning rate of $5 \times 1 0 ^ { - 5 }$ . The RL phase is initialized from the SFT checkpoint, trained for one epoch with a learning rate of $1 \times 1 0 ^ { - 6 }$ , and samples 8 candidate responses per input. For both stages, we set the per-GPU batch size to 1, apply gradient accumulation over 4 steps, and utilize BF16 precision. All experiments were conducted on a cluster equipped with 8 NVIDIA H20 GPUs.

<table><tr><td rowspan="3">Model</td><td colspan="8">DualityVidQA</td><td colspan="3">IPV-Bench</td></tr><tr><td></td><td>MCQ</td><td></td><td></td><td>OQA</td><td></td><td></td><td>CAP.</td><td></td><td>MCQ</td><td>OQA CAP.</td></tr><tr><td>Real</td><td>CF</td><td>Both</td><td>Real</td><td>CF</td><td>Both</td><td>Real</td><td>CF</td><td>Both</td><td></td><td>1 1</td></tr><tr><td>Random</td><td>24.2 23.9</td><td></td><td>4.5</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>20</td><td>1</td></tr><tr><td colspan="10">Close-source VLLMs</td></tr><tr><td>GPT-4o-mini (Hurst et al. 2024)</td><td>91.8</td><td>57.4</td><td>51.9</td><td>91.7 26.7</td><td>24.7</td><td></td><td>93.2</td><td>17.7</td><td>11.3</td><td>68.4 25.6</td><td>14.1</td></tr><tr><td>GPT-4o (Hurst et al. 2024)</td><td>92.7</td><td>72.1</td><td>65.8</td><td>93.0 27.4</td><td>21.8</td><td>94.5</td><td>21.1</td><td>14.6</td><td>79.7</td><td>35.3</td><td>21.8</td></tr><tr><td>GPT-4.1 (Achiam et al. 2023)</td><td>87.3</td><td>75.5</td><td>65.6</td><td>90.5 25.1</td><td>23.4</td><td>91.2</td><td>16.3</td><td>12.2</td><td>82.6</td><td>35.6</td><td>21.3</td></tr><tr><td>GPT-5.5 (OpenAI 2026a)</td><td>95.0</td><td>83.4</td><td>82.4</td><td>96.5 38.6</td><td>35.5</td><td>97.0</td><td>24.8</td><td>22.1</td><td>89.5</td><td>40.2</td><td>24.9</td></tr><tr><td>GPT-5.6 (OpenAI 2026b)</td><td>96.8</td><td>85.5</td><td>84.0</td><td>98.2 40.7</td><td>40.4</td><td>99.0</td><td>25.8</td><td>25.8</td><td>90.6</td><td>43.1</td><td>25.5</td></tr><tr><td>Gemini-2.5 Pro (Comanici et al. 2025)</td><td>92.5</td><td>80.3</td><td>74.3</td><td>94.3 31.8</td><td>25.4</td><td>94.5</td><td>16.1</td><td>14.1</td><td>87.4</td><td>39.1</td><td>24.2</td></tr><tr><td colspan="10">Open-source VLLMs</td></tr><tr><td>Qwen2.5-vl-7B* (Bai et al. 2025b)</td><td>91.7</td><td>59.9</td><td>52.8</td><td>94.2 21.5</td><td>14.9</td><td>92.8</td><td>12.1</td><td>9.8</td><td>73.2</td><td>26.3</td><td>14.4</td></tr><tr><td>Qwen2.5-vl-32B* (Bai et al. 2025b)</td><td>95.2</td><td>55.4</td><td>51.3</td><td>96.8 32.3</td><td>29.3</td><td>97.8</td><td>12.1</td><td>11.4</td><td>77.6</td><td>29.9</td><td>17.1</td></tr><tr><td>Qwen2.5-vl-72B* (Bai et al. 2025b)</td><td>95.8</td><td>62.8</td><td>58.9</td><td>97.8 35.7</td><td>35.2</td><td>96.8</td><td>15.6</td><td>13.7</td><td>81.3</td><td>34.1</td><td>20.3</td></tr><tr><td>Qwen3-vl-8B (Bai et al. 2025a)</td><td>91.7</td><td>64.4</td><td>57.3</td><td>91.1 28.7</td><td>27.3</td><td>94.3</td><td>18.0</td><td>16.2</td><td>76.8</td><td>29.1</td><td>16.1</td></tr><tr><td>Qwen3-vl-32B (Bai et al. 2025a)</td><td>93.3</td><td>70.5</td><td>61.6</td><td>93.8 33.9</td><td>32.0</td><td>95.7</td><td>20.9</td><td>20.7</td><td>83.5</td><td>36.0</td><td>22.0</td></tr><tr><td>InternVL3.5-8B (Wang et al. 2025)</td><td>39.1</td><td>77.3</td><td>18.9</td><td>40.4 21.5</td><td>7.8</td><td>77.3</td><td>9.4</td><td>9.4</td><td>74.6</td><td>26.7</td><td>14.7</td></tr><tr><td>InternVL3.5-14B (Wang et al. 2025)</td><td>36.4</td><td>81.1</td><td>19.7</td><td>35.7 21.7</td><td>8.4</td><td>81.3</td><td>5.3</td><td>5.3</td><td>78.2</td><td>31.1</td><td>17.7</td></tr><tr><td>MiMo-VL-7B (Team et al. 2025)</td><td>74.6</td><td>76.1</td><td>52.6</td><td>81.4 14.0</td><td>11.2</td><td>91.3</td><td>11.7</td><td>8.0</td><td>75.1</td><td>26.9</td><td>14.4</td></tr><tr><td>DNA-Training-7B* (Huang et al. 2026)</td><td>95.8</td><td>80.1</td><td>76.8</td><td>96.8 48.7</td><td>46.2</td><td>97.5</td><td>28.0</td><td>25.8</td><td>85.7</td><td>43.8</td><td>25.2</td></tr><tr><td>FADE(ours)</td><td>96.4</td><td>87.7</td><td>84.6</td><td>99.2</td><td>77.8 76.5</td><td>99.8</td><td>57.2</td><td>57.0</td><td>91.2</td><td>78.6</td><td>60.2</td></tr></table>

Table 1: Results (%) under our anchor-fading evaluation protocol on DualityVidQA-test and IPV-Bench. For DualityVidQAtest, Real, CF, and Both denote factual, counterfactual, and strict paired accuracy, respectively; IPV-Bench contains only counterfactual videos. For models marked with <sup>∗</sup>, MCQ scores are taken from the original DualityVidQA study (Huang et al. 2026); all other scores are obtained using our protocol.

Benchmarks and Metrics. We apply the evaluation protocol to DualityVidQA-test (Huang et al. 2026) and IPV-Bench (Bai, Ci, and Shou 2025). DualityVidQA-test contains factual–counterfactual video pairs and reports factual accuracy (Real), counterfactual accuracy (CF), and strict paired accuracy (Both), where a pair is correct only when both instances are correctly interpreted. IPV-Bench contains only counterfactual videos and reports accuracy under each reconstructed format. Thus, the protocol reuses the original public test instances while diagnosing whether model judgments survive the removal of textual guidance.

## Main Results

Table 1 reports results with our evaluation protocol. Existing Video-MLLMs deteriorate sharply as textual guidance is removed. On DualityVidQA-test (Both), Qwen3-VL-8B drops from 57.3 in MCQ to 27.3 in OQA and 16.2 in captioning, retaining only 47.6% and 28.3% of its MCQ score. Even GPT-5.6 falls from 84.0 to 40.4 and 25.8, with retention rates of 48.1% and 30.7%. These consistent declines indicate that strong MCQ performance can reflect verification of textspecified hypotheses rather than autonomous counterfactual discovery.

The FADE-trained model achieves the best strict paired scores on DualityVidQA-test and the best scores on IPV-Bench across all three formats. On DualityVidQA-test (Both), it scores 84.6, 76.5, and 57.0 under MCQ, OQA, and captioning, retaining 90.4% and 67.4% of its MCQ performance. On IPV-Bench, it obtains 91.2, 78.6, and 60.2, compared with 90.6, 43.1, and 25.5 for GPT-5.6. The substantially flatter degradation demonstrates that FADE-trained model preserves evidence-grounded counterfactual judgments as instance-specific textual anchors fade.

<table><tr><td rowspan="2">Configuration</td><td colspan="2">DualityVidQA-test</td><td colspan="2">IPV-Bench</td></tr><tr><td colspan="2">MCQ OQA Cap.</td><td colspan="2">MCQ OQA Cap.</td></tr><tr><td>Qwen3-VL-8B</td><td colspan="2">57.3 27.3 16.2</td><td colspan="2">76.8 29.1 16.1</td></tr><tr><td>w/o SFT</td><td colspan="2">63.8 37.9 24.6</td><td colspan="2">80.2 40.5 26.7</td></tr><tr><td>w/o RL</td><td colspan="2">77.9 55.6 37.4</td><td colspan="2">85.9 57.2 38.7</td></tr><tr><td>w/o Prog. RL</td><td colspan="2">82.1 67.8 48.8</td><td colspan="2">89.1 69.5 50.8</td></tr><tr><td>FADE</td><td colspan="2">84.6 76.5 57.0</td><td colspan="2">91.2 78.6 60.2</td></tr></table>

Table 2: Ablation of FADE on DualityVidQA-test (Both) and IPV-Bench under the same evaluation protocol, showing the complementary contributions of SFT, RL, and progressive anchor fading.

## Ablation Study

From Evidence Discovery to Understanding. To investigate the individual contributions of SFT, RL, and the progressive reinforcement strategy, we construct three ablation variants: w/o SFT, which applies progressive RL directly to the base model; w/o RL, which retains only the SFT phase; and w/o Prog. RL, which jointly trains on MCQ, OpenQA, and Captioning with identical data and update steps to ablate the progressive mechanism. All variants share the same base model and evaluation protocol. For DualityVidQA-test, we exclusively report the strict Both metric.

(a) Video sequence with annotated CF segment  
![](images/400a766e00bf533e8853d0845e449eda86946655183db04d9ad37bc7db1604cb.jpg)

![](images/087bdadf4af74b711cd3e39d5dbaea4ce47993e7ca41b56f4f1235dd282fe386.jpg)

![](images/3e084c80e0c61cfc14bb446c1216cf3fcdac25a05e611ee26b62e4ca255c2925.jpg)  
(b) Response-conditioned temporal localization

![](images/e269bd2eb67883679d34279d9aa6387e60912ce4affd87f73a71e66855560fe2.jpg)

Counterfactual segment (F5–F8)  
![](images/6a7a66beb4791573b9bc5d053fb5e608ed19a65755d0851c10d568a7430a467f.jpg)

![](images/bd1d58fa2de90c24d9e2e582aa03c4a0ecdf9a7f617f1c773ae9b5b15610a3f0.jpg)

(c) Cross-format ECR (↑)  
![](images/047eedbb718dbdb98d5dbea4310d767e8cd5b3ed11b99c1d34f80376d807a466.jpg)  
Figure 4: Visualization of counterfactual evidence discovery before and after SFT. (a) Video sequence with the annotated counterfactual interval (F5–F8). (b) Frame-level localization responses of the base and SFT-tuned models across MCQ, OQA, and captioning. (c) Evidence concentration rate (ECR), showing that SFT consistently focuses on decisive evidence as textual anchors fade.

As shown in Table 2, directly applying RL yields only marginal gains, indicating that in the absence of evidenceaware initialization, the model struggles to consistently localize critical counterfactual segments. While the SFT-only variant significantly improves MCQ performance, it exhibits a substantial deficit on openQA and captioning, suggesting that evidence retrieval alone is insuficient to guarantee accurate comprehension. The non-progressive RL variant further enhances overall performance; however, its advantages remain largely contingent on scenarios with strong textual anchors. As questions and options are progressively stripped of textual cues, its performance gap relative to the full model consistently widens. The full FADE-trained model achieves the best performance across all configurations on both benchmarks, demonstrating that evidence discovery (via SFT), evidence comprehension (via RL), and robustness to weak textual cues (via progressive optimization) are mutually indispensable.

Counterfactual Evidence Discovery Capability of SFT. To investigate whether SFT efectively equips the model with the ability to identify definitive counterfactual evidence, Figure 4 presents a comparison of per-frame evidence responses between the base model and an SFT-only variant across three conditions: MCQ, OpenQA, and Caption. We quantify the concentration of model responses within the ground-truth counterfactual interval $\mathcal { T } _ { \mathrm { C F } }$ using the Evidence Concentration Rate (ECR), defined as $\begin{array} { r } { \mathrm { E } \tilde { \mathrm { C R } } = \sum _ { t \in \mathcal { T } _ { \mathrm { C F } } } { s _ { t } } / \sum _ { t } { s _ { \mathrm { i } } } } \end{array}$ <sub>t</sub>, where $s _ { t }$ denotes the localization score at frame t. The base model accurately localizes counterfactual segments only under the MCQ condition, where explicit answer options are provided; the progressive removal of textual cues causes its attention distribution to become increasingly difuse. In contrast, the SFT-tuned model consistently localizes the counterfactual intervals across all three conditions. This demonstrates that SFT shifts the model from prompt-dependent passive verification to autonomous, task-agnostic evidence discovery, thereby validating the primary design objective of Phase I: enabling the model to accurately attend to pivotal evidence.

## Conclusion

We present FADE, a two-stage training framework that shifts counterfactual video understanding from text-guided verification toward video-driven discovery. Evidence-internalized SFT first grounds model responses in decisive anomalous evidence, and fading-anchor RL then preserves its interpretation and explanation as textual guidance decreases. In addition, we introduce an anchor-fading evaluation protocol that reformulates existing public MCQ benchmarks into aligned MCQ, OQA, and captioning settings without collecting a new test set. Experiments on DualityVidQA and IPV-Bench reveal that existing Video-MLLMs degrade sharply as textual anchors fade, whereas the FADE-trained model remains substantially more robust, underscoring the importance of evaluating autonomous evidence discovery beyond constrained MCQ accuracy.

## References

Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Agrawal, A.; Batra, D.; Parikh, D.; and Kembhavi, A. 2018. Don’t just assume; look and answer: Overcoming priors for visual question answering. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, 4971– 4980.

Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.; Deng, L.; Ding, W.; Gao, C.; Ge, C.; et al. 2025a. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang,K.; Wang, P.; Wang, S.; Tang, J.; et al. 2025b. Qwen2. 5-VLTechnical Report. arXiv e-prints, arXiv–2502.

Bai, Z.; Ci, H.; and Shou, M. Z. 2025. Impossible videos. arXiv preprint arXiv:2503.14378.

Baradel, F.; Neverova, N.; Mille, J.; Mori, G.; and Wolf, C. 2019. Cophy: Counterfactual learning of physical dynamics. arXiv preprint arXiv:1909.12000.

Bordes, F.; Garrido, Q.; Kao, J. T.; Williams, A.; Rabbat, M.; and Dupoux, E. 2025. Intphys 2: Benchmarking intuitive physics understanding in complex synthetic environments. arXiv preprint arXiv:2506.09849.

Cadene, R.; Dancette, C.; Cord, M.; Parikh, D.; et al. 2019. Rubi: Reducing unimodal biases for visual question answering. Advances in neural information processing systems, 32.

Cai, M.; Tan, R.; Zhang, J.; Zou, B.; Zhang, K.; Yao, F.; Zhu, F.; Gu, J.; Zhong, Y.; Shang, Y.; et al. 2024. Temporalbench: Benchmarking fine-grained temporal understanding for multimodal video models. arXiv preprint arXiv:2410.10818.

Chen, Y.; Liu, J.; Lin, X.; and Tang, R. 2025. Counter-VQA: Evaluating and Improving Counterfactual Reasoning in Vision-Language Models for Video Understanding. arXiv preprint arXiv:2511.19923.

Comanici, G.; Bieber, E.; Schaekermann, M.; Pasupat, I.; Sachdeva, N.; Dhillon, I.; Blistein, M.; Ram, O.; Zhang, D.; Rosen, E.; et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Dai, B.; Wang, L.; Jia, B.; Zhang, Z.; Zhu, S.-C.; Zhang, C.; and Zhu, Y. 2023. X-voe: Measuring explanatory violation of expectation in physical events. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 3992–4002.

Feng, K.; Gong, K.; Li, B.; Guo, Z.; Wang, Y.; Peng, T.; Wu, J.; Zhang, X.; Wang, B.; and Yue, X. 2026. Video-r1: Reinforcing video reasoning in mllms. Advances in Neural Information Processing Systems, 38: 99114–99137.

Huang, Z.; Wen, H.; Hao, A.; Song, B.; Wu, M.; Wu, J.; Chu, X.; Lu, S.; and Wang, H. 2026. Taming Hallucinations: Boosting MLLMs’ Video Understanding via Counterfactual Video Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8153–8163.

Hurst, A.; Lerer, A.; Goucher, A. P.; Perelman, A.; Ramesh, A.; Clark, A.; Ostrow, A.; Welihinda, A.; Hayes, A.; Radford, A.; et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.

Ji, D.; Yang, Y.; Wu, H.; Ma, S.; Chen, T.; and Zhu, L. 2025. RAVEN: Robust advertisement video violation temporal grounding via reinforcement reasoning. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 6: Industry Track), 22–31.

Krojer, B.; Komeili, M.; Ross, C.; Garrido, Q.; Sinha, K.; Ballas, N.; and Assran, M. 2025. A shortcut-aware videoqa benchmark for physical understanding via minimal video pairs. arXiv preprint arXiv:2506.09987.

Li, J.; Niu, L.; and Zhang, L. 2022. From representation to reasoning: Towards both evidence and commonsense reasoning for video question-answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 21273–21282.

Li, X.; Yan, Z.; Meng, D.; Dong, L.; Zeng, X.; He, Y.; Wang, Y.; Qiao, Y.; Wang, Y.; and Wang, L. 2025. Videochatr1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958.

Li, Y.; Wang, X.; Xiao, J.; Ji, W.; and Chua, T.-S. 2022. Invariant grounding for video question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2928–2937.

Lim, G.; Shim, M.; Park, S.; Lee, J.; Lee, I.; Kim, T.; Wee, D.; and Choi, Y. 2026. Video-oasis: Rethinking evaluation of video understanding. arXiv preprint arXiv:2603.29616.

Liu, S.-Y.; Dong, X.; Lu, X.; Diao, S.; Belcak, P.; Liu, M.; Chen, M.-H.; Yin, H.; Wang, Y.-C. F.; Cheng, K.-T.; et al. 2026. Gdpo: Group reward-decoupled normalization policy optimization for multi-reward rl optimization. arXiv preprint arXiv:2601.05242.

Loginova, O.; Bezrukov, O.; Shekhar, R.; and Kravets, A. 2025. Addressing blind guessing: Calibration of selection bias in multiple-choice question answering by video language models. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), 3216–3246.

Motamed, S.; Chen, M.; Van Gool, L.; and Laina, I. 2025. Travl: A recipe for making video-language models better judges of physics implausibility. arXiv preprint arXiv:2510.07550.

Niu, Y.; Tang, K.; Zhang, H.; Lu, Z.; Hua, X.-S.; and Wen, J.-R. 2021. Counterfactual vqa: A cause-efect look at language bias. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 12700–12710.

OpenAI. 2026a. GPT-5.5 System Card. https: //deploymentsafety.openai.com/gpt-5-5/gpt-5-5.pdf. Accessed: 2026-07-29.

OpenAI. 2026b. GPT-5.6 System Card. https: //deploymentsafety.openai.com/gpt-5-6/gpt-5-6.pdf. Accessed: 2026-07-29.

Zhou, Q.; Gong, Y.; Bao, G.; Qiu, H.; Li, J.; Zhu, X.; Zhang, H.; and Zhang, Y. 2025. Reasoning is all you need for video generalization: A counterfactual benchmark with subquestion evaluation. In Findings of the Association for Computational Linguistics: ACL 2025, 2939–2957.

Pramanick, S.; Mavroudi, E.; Song, Y.; Chellappa, R.; Torresani, L.; and Afouras, T. 2025. Enrich and Detect: Video Temporal Grounding with Multimodal LLMs. In Proceedings ofthe IEEE/CVFInternational Conference on Computer Vision, 24297–24308.

Qwen Team. 2026. Qwen3.6-Plus: Towards Real World Agents.

Riochet, R.; Castro, M. Y.; Bernard, M.; Lerer, A.; Fergus, R.; Izard, V.; and Dupoux, E. 2018. Intphys: A framework and benchmark for visual intuitive physics reasoning. arXiv preprint arXiv:1803.07616.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y.; Wu, Y.; et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Singh, A.; Fry, A.; Perelman, A.; Tart, A.; Ganesh, A.; El-Kishky, A.; McLaughlin, A.; Low, A.; Ostrow, A.; Ananthram, A.; et al. 2025. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267.

Team, K.; Du, A.; Yin, B.; Xing, B.; Qu, B.; Wang, B.; Chen, C.; Zhang, C.; Du, C.; Wei, C.; et al. 2025. Kimi-vl technical report. arXiv preprint arXiv:2504.07491.

Wang, H.; Xu, Z.; Cheng, Y.; Diao, S.; Zhou, Y.; Cao, Y.; Wang, Q.; Ge, W.; and Huang, L. 2024. Grounded-videollm: Sharpening fine-grained temporal grounding in video large language models. arXiv preprint arXiv:2410.03290.

Wang, W.; Gao, Z.; Gu, L.; Pu, H.; Cui, L.; Wei, X.; Liu, Z.; Jing, L.; Ye, S.; Shao, J.; et al. 2025. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and eficiency. arXiv preprint arXiv:2508.18265.

Wang, Y.; Wang, Z.; Xu, B.; Du, Y.; Lin, K.; Xiao, Z.; Yue, Z.; Ju, J.; Zhang, L.; Yang, D.; et al. 2026. Timer1: Post-training large vision language model for temporal video grounding. Advances in Neural Information Processing Systems, 38: 83330–83364.

Xiao, J.; Yao, A.; Li, Y.; and Chua, T.-S. 2024. Can i trust your answer? visually grounded video question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13204–13214.

Yi, K.; Gan, C.; Li, Y.; Kohli, P.; Wu, J.; Torralba, A.; and Tenenbaum, J. B. 2019. Clevrer: Collision events for video representation and reasoning. arXiv preprint arXiv:1910.01442.

Zhao, Y.; Huang, J.; Hu, J.; Wang, X.; Mao, Y.; Zhang, D.; Jiang, Z.; Wu, Z.; Ai, B.; Wang, A.; Zhou, W.; and Chen, Y. 2024. SWIFT:A Scalable lightWeight Infrastructure for Fine-Tuning. arXiv:2408.05517.

Zheng, Y.; Zhang, R.; Zhang, J.; Ye, Y.; Luo, Z.; Feng, Z.; and Ma, Y. 2024. LlamaFactory: Unified Eficient Fine-Tuning of 100+ Language Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations). Bangkok, Thailand: Association for Computational Linguistics.