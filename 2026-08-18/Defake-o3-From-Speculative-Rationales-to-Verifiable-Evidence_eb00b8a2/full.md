# Defake-o3: From Speculative Rationales to Verifiable Evidence for Explainable AIGI Detection

Bowen Deng<sup>∗</sup> des\_wv@sjtu.edu.cn Shanghai Jiao Tong University Shanghai, China

## Abstract

Jiahui Zhan   
jiahuizhan@sjtu.edu.cn   
Shanghai Jiao Tong   
University   
Shanghai, China Yikun Ji   
da-kun@sjtu.edu.cn   
Shanghai Jiao Tong University Shanghai, China   
Haozhen Yan   
orion810@sjtu.edu.cn   
Shanghai Jiao Tong   
University   
Shanghai, China

Jianfu Zhang<sup>∗†</sup> c.sis@sjtu.edu.cn Shanghai Jiao Tong University Shanghai, China

The rapid progress of image generation models calls for AI-generated image (AIGI) detectors that are not only accurate but also explainable and reliable. While MLLM-based detectors can provide natural language explanations, existing methods often generate speculative rationales: they rely on vague or hallucinated artifacts, miss subtle localized flaws from the latest generators, and fail to provide evidence that can be visually verified. We present Defake-o3, an explainable AIGI detector that moves from speculative rationales to verifiable evidence. It combines interactive visual search with verifier-guided evidence alignment: the model iteratively zooms into suspicious regions to inspect fine-grained details, while an Evidence Verifier, trained from human verification annotations, provides reinforcement learning rewards that favor grounded evidence and penalize baseless claims. To support this objective, we construct GroundFake, a dataset designed for grounded explainable detection, with localized bounding-box evidence, human verifica tion based on visual grounding and artifact specificity, corrected reasoning trajectories, and valid/invalid evidence supervision. We further introduce FakeFrontier, an out-of-distribution benchmark built from real images and outputs of 10 recent generators, together with an MLLM-based protocol for evaluating evidence quality and persuasiveness. Experiments on GroundFake, FakeFrontier, and additional out-of-distribution benchmarks show that Defake-o3 im proves both detection accuracy and explanation quality, producing more localized, verifiable, and persuasive evidence.

## 1 Introduction

The rapid progress of image generation models, including autoregressive text-to-image systems [10, 45, 55, 66] and difusion-based approaches [4, 7, 16, 21, 36, 46, 47, 51, 60], has made synthetic images increasingly photorealistic and harder to distinguish from authentic photographs. Although these advances benefit creative applications, they also create serious risks for public trust by enabling the spread of visually convincing misinformation at scale. As a result, robust and trustworthy AI-generated image (AIGI) detection has become increasingly important. Traditional AIGI detection methods typically cast the task as binary classification. Built on Convolutional Neural Networks (CNNs) [42] and Vision Transformers [15], these data-driven detectors have achieved strong detection performance [9, 12, 52, 57, 58, 64], yet they remain fundamentally black-box models. In most cases, they output only an authenticity score without revealing the visual evidence behind the decision. As a result, their predictions are dificult to verify, which limits their trustworthiness and practical deployment in high-stakes settings.

![](images/f4e559a3b64efc4705bc24f6a9a830df92689101d156cef2403409fbf0483db4.jpg)  
Figure 1: Overview of Defake-o3. Compared with existing single-pass MLLM detectors, Defake-o3 iteratively zooms into suspicious regions to verify candidate artifacts and produces a structured verdict supported by global evidence and localized key evidence.

Recent studies have explored Multimodal Large Language Models (MLLMs) [2, 3, 5, 14, 32, 35] for explainable AIGI detection [26, 33, 59, 63, 65], producing natural-language rationales alongside authenticity predictions. However, these rationales are often speculative rather than supported by verifiable evidence. First, they are frequently ungrounded, relying on vague, generic, or hallucinated cues instead of precise visual artifacts, with no explicit criterion for valid evidence. Second, existing methods generally perform single-pass, coarse-grained inspection. As modern generators leave increasingly subtle and localized flaws, standard opensource MLLMs, constrained by input resolution and single-pass visual encoding, often fail to examine suspicious regions in suficient detail. Consequently, their explanations may appear plausible while remaining weakly grounded, poorly localized, and dificult to verify.

To address these limitations, we propose Defake-o3, an explainable AIGI detector that moves from speculative rationales to verifiable evidence through two complementary components: interactive visual search, which determines where and at what granularity to inspect, and verifier-guided evidence alignment, which defines what constitutes valid evidence. Defake-o3 adopts an iterative “thinking-with-images” mechanism [56, 61, 69], using a zoom-in tool to crop and magnify suspicious regions and uncover subtle artifacts missed in the full image (Fig. 1). To support this objective, we construct GroundFake, a debiased training dataset that controls image format, aspect ratio, semantic category, and aesthetic quality to reduce shortcut bias. It provides localized bounding-box evidence under strict human verification based on visual grounding and artifact specificity, together with corrected reasoning trajectories and valid/invalid evidence labels. These annotations support both supervised fine-tuning and the training of an Evidence Verifier, which serves as the reward model [13] during GRPO-based reinforcement learning [48, 50, 67] to reward grounded evidence and penalize baseless claims. For evaluation on recent generators, we introduce FakeFrontier, an out-of-distribution benchmark containing 2,000 real images and 2,000 synthetic images generated by 10 recent models, including Nano Banana [20] and Seedream 4.5 [6]. We also develop an MLLM-based protocol to evaluate both classification performance and the quality and persuasiveness of the produced evidence.

In summary, our contributions are threefold:

• We propose Defake-o3, an explainable AIGI detector that combines interactive visual search with verifier-guided evidence alignment. Iterative zoom-in inspection reveals subtle artifacts, while the Evidence Verifier guides reinforcement learning toward grounded evidence and away from baseless claims.

• We introduce GroundFake, a debiased training dataset for grounded explainable AIGI detection. It provides localized evidence verified according to explicit visual-grounding and artifact-specificity criteria, together with corrected reasoning trajectories and valid/invalid evidence labels for supervised fine-tuning and verifier training.

• We present FakeFrontier, an out-of-distribution benchmark containing real images and outputs from 10 recent generators, along with an MLLM-based protocol for evaluating both detec tion performance and evidence quality.

## 2 Related Work

Conventional Black-Box AIGI Detection: A dominant line of AIGI detection formulates the task as binary classification. CNNSpot [57] uses data augmentation to improve cross-model generalization of a standard ResNet detector. DIRE [58] and DRCT [9] exploit reconstruction discrepancies revealed by pre-trained difusion models. NPR [52] identifies synthetic images through local pixel statistics introduced by upsampling, while AIDE [64] combines frequency and semantic cues in a dual-stream architecture. Recent studies [12] further suggest that detector generalization is also sensitive to distributional bias between real and synthetic data. These methods have established strong classification baselines for AIGI detection,

but they remain fundamentally black-box detectors. In practice, they typically output only an authenticity score, without revealing which visual cues support the prediction or providing localized, visually verifiable evidence that humans can inspect. Consequently, their decisions are dificult to interpret and verify, which motivates recent eforts toward explainable AIGI detection.

Explainable Detection via MLLMs: To move beyond score-only black-box detectors, recent studies have extended Multimodal Large Language Models (MLLMs) to explainable AIGI detection, while related works also consider broader image or video forgery settings. Instruction-tuning-based methods such as FakeVLM [59] and FakeScope [34] construct large-scale visual instruction data so that MLLMs can jointly predict authenticity and generate textual explanations. IVY-FAKE [25] further unifies explainable detection across images and videos. To improve spatial grounding, methods such as LEGION [26] and FakeShield [63] augment MLLMs with artifact localization, often by leveraging external models such as SAM [27] to predict masks over suspicious artifact regions. More recently, align ment strategies based on preference optimization or reinforcement learning have been explored to improve explanation quality. AIGI-Holmes [70] adopts DPO [44] to suppress low-quality explanations, but its outputs remain coarse-grained without explicit localization. So-Fake [23] and FakeXplain [24] further combine explanation generation with bounding-box grounding under RL-style training. However, current methods still mainly optimize the similarity ofthe explanation with reference annotations, rather than explicitly learning whether a claimed piece of evidence is visually grounded and artifact-specific enough to support an AIGI verdict. As a result, even localized outputs can remain vague, spurious, or hallucinated, and existing resources still provide limited evidence-level supervision for training and evaluating truly verifiable explanations.

Dynamic Visual Reasoning and Exploration: Recent advances in MLLM-based visual reasoning have shifted from fixed, textdominant chain-of-thought toward interactive visual exploration, where models actively decide where to inspect and at what granularity. Early works such as Visual CoT [49] and LLaVA-CoT [62] made reasoning more explicit through region grounding or staged decomposition, but still relied largely on static visual inputs. Later studies treated visual search as a core reasoning primitive: V\* [61] introduced LLM-guided search for high-resolution detail grounding, Visual-RFT [37] showed that verifiable rewards can improve visual reasoning behavior, and Pixel Reasoner [56], DeepEyes [69], and Mini-o3 [30] further enabled models to zoom, inspect, and iteratively interact with images during inference. These advances are especially relevant to explainable AIGI detection, where recent generators leave increasingly subtle and localized artifacts that are hard to capture with single-pass visual encoding. However, they mainly improve where to look, rather than what should count as valid evidence for an AIGI verdict. Therefore, interactive visual search alone does not guarantee grounded, artifact-specific, and visually verifiable explanations.

## 3 GroundFake Dataset Construction

To support grounded explainable AIGI detection, we construct GroundFake, a training dataset containing 16,000 images, stepwise reasoning transcripts, and evidence annotations. For fake images, the localized key evidence is manually verified under visual ground ing and artifact specificity criteria. We further rewrite the reasoning trajectories using the verified evidence for annotation consistency. GroundFake provides two complementary supervision signals for Defake-o3: rewritten trajectories for supervised fine-tuning and valid/invalid evidence labels for Evidence Verifier training. The overall construction pipeline of GroundFake is illustrated in Fig. 2.

![](images/33e2e0c23301e6c35f8e40cce2e099c704e79ff7f2f3de6ef5ed3dea6350e86e.jpg)  
Figure 2: GroundFake construction pipeline. Starting from a balanced, bias-controlled image pool, we generate label-conditioned candidate reasoning trajectories and localized evidence, manually verify localized key evidence for fake images, and rewrite the trajectories into annotation-consistent supervision transcripts. The resulting dataset supports both SFT and Evidence Verifier training.

## 3.1 Image Collection and Bias Control

We construct a balanced and bias-controlled image pool of 16k images (8k real and 8k fake) for GroundFake. Real images are sampled from Open Images V7 [17], while fake images are collected from recent image generators and supplemented with additional FLUX.1- dev samples [28] to broaden photographic conditions. To reduce dataset-specific shortcuts, we standardize file format and align the aspect-ratio, coarse semantic-category, and aesthetic-quality distributions between the real and fake subsets, encouraging the detector to rely on generation artifacts rather than trivial source cues.

## 3.2 Label-Conditioned Candidate Annotation

At this stage, we use Gemini 3 Pro Preview [19] to produce candidate reasoning trajectories and evidence proposals conditioned on the known image label. The ground-truth label is provided only to elicit class-consistent candidate evidence and localization proposals. The prompt lists several common artifact categories in AI-generated images, including incoherent text, deformed faces or hands, distorted object structures, and violations of common sense or physical constraints, as heuristic cues for candidate generation. The model is then instructed to inspect the image region by region, with attention to fine-grained details, and to return a structured output containing a reasoning trajectory, a label-consistent conclusion, and supporting evidence. Specifically, the generated output contains three components:

• Reasoning Trajectory: A sequential list in which each step records a region ofinterest (specified by a bounding box) together with the corresponding intermediate reasoning text.

• Verdict: A label-consistent conclusion in {“real”, “fake”}, included to maintain a complete supervision format.

• Evidence: The visual and textual basis associated with the conclusion. For both real and fake images, the model provides one global evidence item that summarizes coarse scene-level coherence. For fake images, it further proposes several localized key evidence items, each paired with a bounding box that points to a candidate generation artifact.

We treat the Gemini outputs as candidate annotations. Gemini 3 Pro Preview provides useful candidate trajectories and localization proposals, but a non-trivial portion of the localized key evidence remains weakly grounded or hallucinated. We therefore conduct a subsequent human verification step on the localized key evidence for fake images before these annotations are used for trajectory rewriting or Evidence Verifier training.

## 3.3 Human Annotation and Verification

We focus human verification on the localized key evidence for fake images, because these artifact claims are the most hallucinationprone and are the annotations later used for verifier supervision. The proposed candidate key evidence may still be weakly grounded or invalid. Typical failure cases include: (1) the evidence text is irrelevant to the content inside the bounding box; (2) the text contradicts the actual visual content (e.g., correctly rendered text or normal hands are falsely described as defective); or (3) the claimed cue is not specific enough to AI generation (e.g., “smooth skin” which may also appear in heavily post-processed real photographs). We therefore ask human annotators to judge each localized key evidence item, consisting of the original image, the bounding box, and the associated text, with a binary label: Valid or Invalid. Each item is independently reviewed by three annotators under two criteria:

(1) Visual Grounding: The textual description accurately matches the visual content within the bounding box.

(2) Artifact Specificity: The described flaw is more specific to AI generation than a generic attribute that may also appear in real photographs.

We retain the vote ratio across the three annotators as a soft validity label for Evidence Verifier training, and use the retained valid evidence in the subsequent trajectory rewriting step. This yields a higher-quality subset of localized evidence by filtering out unreasonable, weakly grounded, or non-specific indicators of AIGI.

## 3.4 Reasoning Trajectory Rewriting for Annotation Consistency

Simply discarding invalid key evidence would leave the reasoning transcript inconsistent with the retained evidence annotations used for training. We therefore employ Gemini 3 Flash Preview [18] to rewrite the reasoning trajectories. Given the original trajectory and the human-verified valid evidence list, the model rewrites the intermediate reasoning text so that the resulting transcript remains consistent with the verified evidence and no longer depends on discarded hallucinated claims. The rewritten trajectories are then used as annotation-consistent supervision transcripts, providing stable training targets for tool use and final output formatting.

## 4 Defake-o3 Methodology

To address the limitations of single-pass and weakly grounded MLLM detectors, we propose Defake-o3. Defake-o3 combines interactive visual search with verifier-guided evidence alignment. At inference time, it iteratively zooms into suspicious regions and then outputs a structured verdict with global evidence and, when available, localized key evidence. Training proceeds in two stages: SFT on human-filtered GroundFake traces to learn tool use and structured evidence generation, followed by GRPO with a hybrid evidence reward that combines rule-based matching and an Evidence Verifier trained from human verification labels. An overview is shown in Fig. 3.

## 4.1 Task Formulation

Given an input image I, the Defake-o3 model $\pi _ { \theta }$ performs a multiturn interaction with the image and then produces a structured output.

Interactive Visual Search. At turn �, the model receives an observation $O _ { i } ,$ which is either the full image or a zoomed-in patch from a previous step, and chooses an action �<sub>�</sub> ∈ {Zoom $\operatorname { I n } ,$ Final Output} A Zoom In action predicts a 2D bounding box $b _ { i }$ and returns the corresponding crop as the next observation $O _ { i + 1 } . \mathrm { A }$ Final Output action terminates the interaction and triggers the final output. Final Structured Output. We denote the final output by $y \ : = \ :$ $( \hat { y } , e _ { g } , E )$ , where $\hat { y } \in \{ \mathsf { r e a l } , \mathsf { f a k e } \}$ is the verdict, $\boldsymbol { e } _ { g } = ( \boldsymbol { 0 } , t _ { g } )$ is a mandatory global evidence item summarizing image-level cues, and $E = \{ e _ { 1 } , \ldots , e _ { K } \}$ is a possibly empty set of localized key evidence items. Each $e ~ = ~ ( b , t ) ~ \in ~ E$ consists of a bounding box $b \ \ne \ 0$ and a text description � of a local artifact. For a real verdict, we set $E \ = \ \varnothing$ . For a fake verdict, the model outputs localized key evidence when clear and visually verifiable local artifacts are found. Otherwise, it may keep $E = \emptyset$ and return only the global evidence to avoid hallucinated localization. Throughout this paper, we reserve the term verifiable evidence for the localized evidence items in $E ;$ the global evidence $e _ { g }$ serves only as a contextual image-level assessment.

## 4.2 Supervised Fine-Tuning (SFT)

In the SFT stage, we bootstrap the base MLLM using the humanfiltered and trajectory-corrected traces in GroundFake. Each training sample is cast as a multi-turn interaction sequence in which the model learns to invoke the zoom-in tool, condition on the returned patches, and emit the final structured output defined in Sec. 4.1. We train the model with standard autoregressive next-token prediction over the full trace. This stage mainly initializes tool use, sequential visual context accumulation, and structured evidence generation, while evidence quality is further aligned in the subsequent RL stage.

## 4.3 Evidence Verifier Training

For grounded AIGI detection, exact matching to reference annotations, such as spatial overlap with human boxes [23, 24], is useful but insuficient as a sole reward signal. Reference annotations can be incomplete, and annotation matching alone does not explicitly penalize invalid or fabricated evidence. To provide a denser and more human-aligned signal, we train an Evidence Verifier $V _ { \phi }$ under the human verification protocol in Sec. 3.3. In GroundFake, each localized evidence item is judged by visual grounding and artifact specificity. Using these annotations, the verifier takes the original image I, the cropped local region $\mathcal { T } _ { b }$ , the bounding box �, and the evidence text � as input, and predicts a validity score $s \in [ 0 , 1 ]$ We instantiate $V _ { \phi }$ by appending a linear classification head to an MLLM and optimize it as a binary classifier with soft labels derived from annotator agreement. Specifically, if� out of � annotators mark an evidence item as valid, we set its target to $y _ { \mathrm { s o f t } } = m / M$ and train the verifier with a soft-label binary cross-entropy loss:

$$
s = V _ { \phi } ( \bar { I } , \bar { J } _ { b } , b , t ) , \quad \bar { \mathcal { L } } _ { \mathrm { v e r } } = - y _ { \mathrm { s o f t } } \log s - ( 1 - y _ { \mathrm { s o f t } } ) \log ( 1 - s ) .\tag{1}
$$

This allows $V _ { \phi }$ to capture the graded certainty of human judgments and provide a learned validity signal for the subsequent RL stage.

![](images/0fc9e536daf74bce9fdb3d5873b38ecfd067b2060149650a1de6eb7671e051be.jpg)  
Figure 3: Overview of Defake-o3. The model performs interactive visual search by iteratively zooming into suspicious regions and then outputs a structured verdict with global and localized evidence. For every correctly classified fake image, the total reward combines the classification reward with rule-based evidence matching and model-based scoring from the Evidence Verifier.

## 4.4 Reward Function Design

Our RL objective combines task correctness with localized evidence quality. We penalize unsupported evidence strongly and avoid rewarding the model for proposing many weak evidence items. Accordingly, the reward is defined on the verdict and the localized key evidence set � from Sec. 4.1.

4.4.1 Total Reward $( R _ { t o t a l } ) .$ For a final output $\boldsymbol { y } = ( \hat { y } , e _ { q } , E )$ , where $E = \left\{ e _ { 1 } , \dots , e _ { K } \right\}$ denotes the localized key evidence set, we define

$$
R _ { t o t a l } = \left\{ \begin{array} { l l } { R _ { f a i l } , } & { \mathrm { i n v a l i d f o r m a t } , } \\ { R _ { c l s } + \mathbb { I } ( \hat { y } = \mathsf { f a k e } ) R _ { e v i } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{2}
$$

where invalidformat means that at least one tool call or the final structured output fails to parse, and we set $R _ { f a i l } = - 1$ . The indicator function I(·) equals 1 when its argument is true and 0 otherwise.

The classification reward measures only verdict correctness:

$$
R _ { c l s } = \mathbb { I } ( \hat { y } = y _ { g t } ) ,\tag{3}
$$

where $y _ { g t }$ is the ground-truth label; while the finer-grained shaping of localized evidence quality is handled by $R _ { e v i }$

4.4.2 Evidence Reward $( R _ { e v i } ) .$ . We compute the evidence reward on the localized key evidence set �. We use a hybrid evidence reward that combines a coarse rule-based matching term with a verifierbased validity term:

$$
R _ { e v i } = ( 1 - \alpha _ { M } ) R _ { r u l e } + \alpha _ { M } R _ { m o d e l } ,\tag{4}
$$

where $R _ { r u l e }$ anchors the policy to the available reference annotations, while $R _ { m o d e l }$ provides a learned signal for evidence validity under the human verification protocol (see Sec. 4.3). This hybrid form avoids relying entirely on either exact annotation matching or the learned verifier alone. Unless otherwise stated, we set $\alpha _ { M } = 0 . 5 $

4.4.3 Rule-based Reward $( R _ { r u l e } )$ . Although exact matching is insuficient as a sole reward signal, it remains a useful coarse anchor when reference localized evidence is available. For the predicted localized evidence set �, we form a union mask $M _ { p r e d }$ from all predicted boxes and concatenate all evidence texts into $T _ { p r e d } .$ Likewise, for the reference localized evidence, we form $M _ { r e f }$ and $T _ { r e f } .$ If the model incorrectly classifies a real image as fake, we set $R _ { r u l e } = 0$ Otherwise, the rule-based reward is defined as:

$$
R _ { r u l e } = \lambda _ { i o u } \mathrm { I o U } ( M _ { p r e d } , M _ { r e f } ) + \lambda _ { b l e u } \mathrm { B L E U - } 2 ( T _ { p r e d } , T _ { r e f } )\tag{5}
$$

where $\lambda _ { i o u } = \lambda _ { b l e u } = 1$ by default. We use set-level matching, since reference annotations may be incomplete and multiple valid decompositions of evidence can exist for the same image. The BLEU-2 term serves as a lightweight lexical anchor.

4.4.4 Model-based Reward $( R _ { m o d e l } )$ . For each localized evidence item $e = ( b , t ) \in E ,$ we extract the corresponding crop $\mathcal { T } _ { b }$ and query the Evidence Verifier $V _ { \phi } { \mathrm { : } }$ , obtaining a score $s _ { e } = V _ { \phi } ( \tilde { J } , \tilde { J _ { b } } , b , t )$ . We then convert the verifier score into an item-level reward:

$$
r _ { m o d e l , e } = \left\{ \begin{array} { l l } { - C _ { \mathrm { i n v a l i d } , } } & { s _ { e } < \tau , } \\ { C _ { \mathrm { v a l i d } } \cdot { \frac { s _ { e } - \tau } { 1 - \tau } } , } & { s _ { e } \geq \tau , } \end{array} \right.\tag{6}
$$

where � is the acceptance threshold under the verification protocol in Sec. 4.3. This mapping assigns a fixed penalty to low-scoring evidence and only bounded positive reward to evidence whose verifier score exceeds �. When a real image is incorrectly predicted as fake, the same mapping naturally penalizes the proposed localized evidence, since such evidence is unlikely to pass the verifier threshold. Unless otherwise stated, we use $\tau = 0 . 5$ and $C _ { \mathrm { v a l i d } } = C _ { \mathrm { i n v a l i d } } = 1 .$

To avoid rewarding the model for proposing many borderline evidence items, we aggregate item-level rewards with cumulative penalties and capped positive rewards. Let

$$
\mathcal { E } ^ { - } = \{ e \mid r _ { m o d e l , e } < 0 \} , \qquad \mathcal { E } ^ { + } = \{ e \mid r _ { m o d e l , e } \geq 0 \} ,\tag{7}
$$

Table 1: Results on the GroundFake test set.
<table><tr><td>Method</td><td>Acc</td><td>F1</td><td>BLEU-1</td><td>BLEU-2</td><td>ROUGE-L</td><td>IoU</td></tr><tr><td>CNNSpot</td><td>0.976</td><td>0.976</td><td></td><td>一</td><td>1</td><td>一</td></tr><tr><td>NPR</td><td>0.838</td><td>0.830</td><td></td><td>一</td><td>一</td><td>一</td></tr><tr><td>AIDE</td><td>0.924</td><td>0.922</td><td></td><td>一</td><td>一</td><td>一</td></tr><tr><td>FakeVLM</td><td>0.857</td><td>0.854</td><td>0.105</td><td>0.045</td><td>0.110</td><td>一</td></tr><tr><td>LEGION</td><td>0.502</td><td>0.024</td><td>0.003</td><td>0.001</td><td>0.002</td><td>0.000</td></tr><tr><td>FakeShield</td><td>0.494</td><td>0.275</td><td>0.029</td><td>0.013</td><td>0.026</td><td>0.017</td></tr><tr><td>Defake-Direct (w/o RL)</td><td>0.946</td><td>0.943</td><td>0.282</td><td>0.154</td><td>0.234</td><td>0.165</td></tr><tr><td>Defake-Direct</td><td>0.958</td><td>0.957</td><td>0.333</td><td>0.195</td><td>0.252</td><td>0.262</td></tr><tr><td>Defake-CoT (w/o RL)</td><td>0.988</td><td>0.988</td><td>0.330</td><td>0.184</td><td>0.267</td><td>0.222</td></tr><tr><td>Defake-CoT</td><td>0.992</td><td>0.992</td><td>0.335</td><td>0.184</td><td>0.257</td><td>0.280</td></tr><tr><td>Defake-o3 (w/o RL)</td><td>0.972</td><td>0.971</td><td>0.319</td><td>0.179</td><td>0.250</td><td>0.241</td></tr><tr><td>Defake-o3</td><td>0.992</td><td>0.992</td><td>0.364</td><td>0.215</td><td>0.281</td><td>0.311</td></tr></table>

Acc/F1: image-level classification performance. BLEU-1/2 and ROUGE-L: evidence-level lexical overlap with the reference annotations. IoU: evidence-level spatial overlap with the reference regions.

and let $r _ { ( 1 ) } \geq r _ { ( 2 ) }$ denote the largest and second-largest reward of the evidence in $\delta ^ { + }$ , respectively. We define:

$$
R _ { m o d e l } = \underbrace { \sum _ { e \in \mathcal { E } ^ { - } } r _ { m o d e l , e } } _ { \mathrm { P e n a l t y } } + \underbrace { \mathcal { R } ( \mathcal { E } ^ { + } ) } _ { \mathrm { R e w a r d } } , \quad \mathcal { A } ( \mathcal { E } ^ { + } ) = \left\{ \begin{array} { l l } { 0 , } & { | \mathcal { E } ^ { + } | = 0 , } \\ { r _ { ( 1 ) } , } & { | \mathcal { E } ^ { + } | = 1 , } \\ { r _ { ( 1 ) } + \beta r _ { ( 2 ) } , } & { | \mathcal { E } ^ { + } | \ge 2 . } \end{array} \right.\tag{8}
$$

This aggregation reflects our preference for a small number of strong, verifiable artifacts rather than many weak ones. In our implementation, we set $\beta = 0 . 5$

## 4.5 Reinforcement Learning

Starting from the SFT-trained weights, we optimize Defake-o3 using GRPO [50, 67]. Each sampled rollout contains multiple rounds of Zoom In actions followed by a final structured output. Through trajectory-level optimization, the model learns to use the zoomin tool judiciously and to favor a small number of high-quality localized evidence items over many weak or vague ones.

## 5 Experiments

## 5.1 Experimental Setup

Implementation Details. Unless otherwise stated, all experiments are conducted on 8 A100 GPUs. We employ Qwen3-VL-8B-Instruct [3] as our base model and fine-tune it using LoRA [22], which is applied to all linear layers within the text backbone, with a rank of $r = 1 6$ and an alpha of $\alpha = 3 2$ . During the SFT stage, we train the model using a global batch size of 8 and a learning rate of $1 \times 1 0 ^ { - 4 }$ . In the RL stage, the group size is set to 16, and the learning rate is $1 \times 1 0 ^ { - 5 }$ . Following DAPO [67], we implement several improvements to the standard GRPO algorithm: (1) removing the KL divergence constraint; (2) adopting token-level loss instead of sequence-level loss; (3) applying a clip higher strategy with $\epsilon _ { \mathrm { l o w } } = 0 . 2$ and $\epsilon _ { \mathrm { h i g h } } = 0 . 2 8 ;$ and (4) utilizing overlong reward masking, a mechanism that limits sampling length without sup pressing the generation of longer reasoning paths. The Evidence Verifier shares the same base model, LoRA configuration, and SFT training settings, with the exception that its linear classification head is fully trainable.

Baselines and Ablations. We compare Defake-o3 with traditional binary classification approaches including CNNSpot [57], NPR [52], and AIDE [64]. For a fair comparison, these models are re-trained on our GroundFake training set. For MLLM-based explainable methods, we employ and test FakeVLM [59], LEGION [26], and FakeShield [63]. We evaluate FakeVLM and FakeShield using their oficially released weights, while LEGION is reproduced utilizing its oficial training code and data. To validate the efectiveness of our proposed method, we design two variants of our model for ablation studies: (1) Defake-CoT: relies solely on pure textual chainof-thought reasoning without zoom-in tool calls; and (2) Defake-Direct: directly outputs the final verdict and evidence without any preliminary reasoning steps.

## 5.2 Results on GroundFake Test Set

Table 1 reports classification (Acc/F1), evidence text overlap (BLEU-1/2 and ROUGE-L), and evidence region overlap (IoU) of state-ofthe-art AIGI detectors and Defake-o3 on the GroundFake test set. RL consistently improves Acc/F1 and IoU for all three variants. Enabled by interactive visual search for verifiable evidence, Defake-o3 achieves the strongest overall explainable performance, matching the best Acc/F1 while attaining the highest BLEU-1, BLEU-2, ROUGE-L, and IoU.

## 5.3 FakeFrontier Benchmark

Beyond the in-domain evaluation on GroundFake, we further assess out-of-distribution generalization on recent generators using FakeFrontier. FakeFrontier contains 4,000 images, evenly split between real and fake. The real subset is drawn from Open Images V7 [17] and Chameleon [64], while the fake subset is collected from outputs of 10 recent image generators, including Nano Banana [20], GPT-Image-1.5 [40], and Seedream 4.5 [6]. The Open Images V7 images used in FakeFrontier do not overlap with those used in GroundFake. Unlike GroundFake, FakeFrontier does not provide human evidence annotations and is therefore used to evaluate both classification on the latest generators and explanation quality without relying on reference boxes or texts. To evaluate explanations under this annotation-free setting, we adopt an MLLM-based protocol on 200 fake images randomly sampled from FakeFrontier. We instantiate the judge with three open-source MLLMs, Qwen3- VL-235B-A22B-Thinking [3], Kimi K2.5 [54], and GLM-4.6V [68], to reduce evaluator-specific bias. Each evidence item is assessed independently rather than as part of a bundled explanation. We consider two complementary aspects: evidence quality, which measures whether an evidence item is visually grounded and specific to AI generation, and evidence persuasiveness, which measures whether it can convince the judge to predict “fake”. Table 2 reports the MLLMbased evaluation, and Table 3 reports classification results on the full FakeFrontier benchmark.

Quality Evaluation. We first evaluate each evidence item independently with an MLLM judge. The judge assigns a score from 1 to 10, where 1 denotes evidence that is irrelevant, contradictory, or not specific to AI generation, and 10 denotes evidence that is visually grounded and highly specific to AI generation. We normalize the score to [0, 1] and report it as the Quality Score (QS). For each image, we average the normalized scores over its evidence items, and then average the resulting image scores over the evaluation set.

![](images/fc07056f8c36f321990987817a86ba854c9fc703187e745a7ff449609bdc9ef1.jpg)  
Figure 4: Qualitative examples from the GroundFake test set. Defake-o3 first performs coarse scene-level inspection and then uses targeted zoom-in steps to verify candidate artifacts, yielding a small set of localized, visually verifiable evidence items. Compared with FakeVLM [59] and FakeShield [63], its outputs are more directly grounded in explicit, image-specific artifacts. FakeShield masks are omitted for brevity.

Table 2: Evaluation with the MLLM-based protocol on the fake-image subset of FakeFrontier.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Avg #Evi</td><td colspan="3">Qwen3-VL-235B (Prior-Acc = 0.2421)</td><td colspan="3">Kimi K2.5 (Prior-Acc = 0.3526)</td><td colspan="3">GLM-4.6V (Prior-Acc = 0.0737)</td></tr><tr><td>QS</td><td>Hit@Img</td><td>Hit@Evi</td><td>QS</td><td>Hit@Img</td><td>Hit@Evi</td><td>QS</td><td>Hit@Img</td><td>Hit@Evi</td></tr><tr><td>Defake-o3</td><td>3.0804</td><td>0.5378</td><td>0.7839</td><td>0.6362</td><td>0.4334</td><td>0.7471</td><td>0.5895</td><td>0.6812</td><td>0.9045</td><td>0.8249</td></tr><tr><td>Defake-CoT</td><td>2.9397</td><td>0.4841</td><td>0.7755</td><td>0.6239</td><td>0.3811</td><td>0.7119</td><td>0.5704</td><td>0.5980</td><td>0.8961</td><td>0.7584</td></tr><tr><td>Defake-Direct</td><td>3.1910</td><td>0.3721</td><td>0.6047</td><td>0.4861</td><td>0.2877</td><td>0.5645</td><td>0.4504</td><td>0.4431</td><td>0.7085</td><td>0.5554</td></tr><tr><td>FakeVLM</td><td>3.6421</td><td>0.1268</td><td>0.3702</td><td>0.2558</td><td>0.1204</td><td>0.4526</td><td>0.3531</td><td>0.2438</td><td>0.5456</td><td>0.5092</td></tr><tr><td>FakeShield</td><td>2.2161</td><td>0.0875</td><td>0.2312</td><td>0.3296</td><td>0.0763</td><td>0.2747</td><td>0.4218</td><td>0.1393</td><td>0.2747</td><td>0.4255</td></tr></table>

Avg #Evi: average number of evidence items per image. Prior-Acc: probability that the MLLM evaluator classifies an image as fake without additional information. QS: Quality Score. Hit@Img / Hit@Evi: Image Hit Rate / Evidence Hit Rate. LEGION is excluded from the explainability evaluation, as it classifies nearly all fake images as real.

Table 3: Classification results on FakeFrontier benchmark.
<table><tr><td>Method</td><td>Real Acc</td><td>Fake Acc</td><td>Acc</td><td>F1</td></tr><tr><td>Defake-o3</td><td>0.9020</td><td>0.9446</td><td>0.9232</td><td>0.9246</td></tr><tr><td>Defake-CoT</td><td>0.8675</td><td>0.9532</td><td>0.9102</td><td>0.9137</td></tr><tr><td>Defake-Direct</td><td>0.8485</td><td>0.7826</td><td>0.8157</td><td>0.8088</td></tr><tr><td>FakeVLM</td><td>0.7127</td><td>0.7102</td><td>0.7114</td><td>0.7108</td></tr><tr><td>FakeShield</td><td>0.8135</td><td>0.4107</td><td>0.6127</td><td>0.5139</td></tr><tr><td>LEGION</td><td>0.9930</td><td>0.0045</td><td>0.5004</td><td>0.0090</td></tr><tr><td>CNNSpot</td><td>0.8200</td><td>0.4786</td><td>0.6499</td><td>0.5767</td></tr><tr><td>NPR</td><td>0.8610</td><td>0.1147</td><td>0.4891</td><td>0.1829</td></tr><tr><td>AIDE</td><td>0.7495</td><td>0.5083</td><td>0.6293</td><td>0.5775</td></tr></table>

Persuasion Evaluation. We next evaluate whether a single evidence item is strong enough to support a fake verdict. For each fake image, the judge first predicts from the image alone, which reveals its prior tendency and is reported as Prior-Acc in Table 2.

Table 4: Accuracy on other OoD test sets.
<table><tr><td>Test Set</td><td>Defake-o3</td><td>FakeVLM</td><td>LEGION</td><td>FakeShield</td><td>CNNSpot</td><td>NPR</td><td>AIDE</td></tr><tr><td>AIGI-Now</td><td>0.9180</td><td>0.8596</td><td>0.4997</td><td>0.8307</td><td>0.7730</td><td>0.6925</td><td>0.8473</td></tr><tr><td>EvalGEN</td><td>0.9872</td><td>0.9668</td><td>0.0907</td><td>0.9513</td><td>0.7660</td><td>0.2430</td><td>0.7045</td></tr><tr><td>MNW</td><td>0.8871</td><td>0.8417</td><td>0.0265</td><td>0.5072</td><td>0.6435</td><td>0.3775</td><td>0.7712</td></tr></table>

We then provide one evidence item from a detector and ask the judge to assess the same image again. Each evidence item is tested independently. Because this decision can be sensitive to prompting, we use three system prompts and average the results. We report two metrics: (1) Hit@Img, the proportion of fake images for which at least one evidence item leads the judge to output “fake”; and (2) Hit@Evi, the proportion of evidence items that lead the judge to output “fake”. For detectors that output plain text rather than structured JSON, we first use Qwen3-VL-8B-Instruct to split their outputs into independent evidence items. These metrics are complementary: QS measures the visual grounding and artifact specificity of the evidence, Hit@Img measures whether a detector can surface at least one decisive flaw for a fake image, and Hit@Evi measures how often its individual evidence items are actually useful rather than noisy or hallucinated. As shown in Tables 2 and 3, Defake-o3 achieves the highest Acc/F1 on FakeFrontier and also ranks first in QS, Hit@Img, and Hit@Evi under all three judges. Although baseline methods produce more evidence items on average, their much lower Hit@Evi suggests that many of those claims are weak or unpersuasive.

## 5.4 Results on External OoD Benchmarks

Beyond FakeFrontier, we further evaluate classification generalization on three external OoD benchmarks: AIGI-Now [11], Eval-GEN [12], and MNW [38]. For these external benchmarks, we report classification accuracy only. To maintain a strict OoD setting, we remove samples from generators that overlap with the Ground-Fake training set. As shown in Table 4, Defake-o3 achieves the best accuracy on all three datasets, reaching 0.9180 on AIGI-Now, 0.9872 on EvalGEN, and 0.8871 on MNW. It consistently outperforms both explainable MLLM baselines and black-box detectors, showing that the advantage of Defake-o3 is not limited to Ground-Fake or FakeFrontier. These results suggest that interactive visual search and verifier-guided evidence alignment improve robustness under broader distribution shifts, without sacrificing classification strength.

## 5.5 Ablation Studies

We conduct ablation studies to assess the efectiveness ofinteractive visual search and the role of the Evidence Verifier.

Efectiveness of Interactive Visual Search: Based on the results in Tables 1–3, Defake-o3 exhibits clear advantages in both classifi cation performance and explanation quality over the Defake-CoT model (which lacks the zoom-in mechanism) and the Defake-Direct model (which lacks any reasoning process).

The Role of the Evidence Verifier: Figure 5 shows that the Evidence Verifier assigns high scores to human-accepted evidence and much lower scores to human-rejected evidence or forcibly fabricated evidence on random bounding boxes (Random in Fake/Real), indicating that it captures the visual grounding and artifact specificity criteria in our annotation protocol. This learned signal leads to better reward shaping during RL. Compared with the rule-only variant $( \alpha _ { M } = 0 \mathrm { i n }$ n Table 5), the model trained with the Evidence Verifier improves IoU on GroundFake from 0.252 to 0.311, while causing only minor changes in lexical overlap with the reference annotations. Table 5 further shows that the best results require a balance between rule-based matching and verifier guidance. Using only the verifier $( \alpha _ { M } = 1 )$ degrades text-explanation performance. Although a smaller verifier weight $( \alpha _ { M } = 0 . 2 5 )$ yields a slightly higher IoU, it noticeably degrades Acc/F1. We therefore use $\alpha _ { M } = 0 . 5$ as the best overall trade-of.

## 5.6 Qualitative Results

Figure 4 illustrates how Defake-o3 turns exploratory inspection into compact, verifiable evidence. It zooms into candidate regions, retains only verifiable artifacts—such as the malformed headset logo, invalid timestamp, and implausible desk object—and discards unsupported suspicions. In contrast, FakeVLM and FakeShield provide coarser evidence that is less directly tied to image-specific artifacts.

![](images/b4166720dce538fed352d4cd17f7d62415c24de6115128f4c32d507e3f848d2b.jpg)  
Figure 5: Output value distribution of the Evidence Verifier on generated evidence.

Table 5: Ablation of reward hyperparameters evaluated on the GroundFake test set.
<table><tr><td>αM</td><td> $\lambda _ { i o u } / \lambda _ { b l e u }$ </td><td>Acc</td><td>F1</td><td>BLEU-1</td><td>BLEU-2</td><td>ROUGE-L</td><td>IoU</td></tr><tr><td>1</td><td>0</td><td>0.984</td><td>0.984</td><td>0.275</td><td>0.150</td><td>0.222</td><td>0.195</td></tr><tr><td>0.75</td><td>1</td><td>0.910</td><td>0.902</td><td>0.303</td><td>0.176</td><td>0.235</td><td>0.260</td></tr><tr><td>0.5</td><td>1</td><td>0.992</td><td>0.992</td><td>0.364</td><td>0.215</td><td>0.281</td><td>0.311</td></tr><tr><td>0.25</td><td>1</td><td>0.962</td><td>0.960</td><td>0.330</td><td>0.194</td><td>0.270</td><td>0.318</td></tr><tr><td>0</td><td>1</td><td>0.986</td><td>0.986</td><td>0.363</td><td>0.218</td><td>0.291</td><td>0.252</td></tr></table>

## 6 Conclusion

In this paper, we presented Defake-o3, an explainable AIGI detector that moves from speculative rationales to verifiable evidence through interactive visual search and verifier-guided evidence alignment. We further introduced GroundFake and FakeFrontier for grounded training and out-of-distribution evaluation. Experiments show that Defake-o3 achieves the best overall performance among compared methods, with top or tied-for-top classification results and stronger judge-based evidence quality.

## Acknowledgments

This work was supported in part by the National Natural Science Foundation of China (Grant Nos. 62302295, 62595733, and 62561160155), the Shanghai Municipal Science and Technology Major Project (Grant No. 2021SHZDZX0102). This work was also supported by Ant Group. In particular, we sincerely thank Zijuan Yu, Yan Hong, and Jun Lan from Ant Group for their valuable help.

## References

[1] Stability AI. 2024. stable-difusion-3.5-large. https://huggingface.co/stabilityai/ stable-difusion-3.5-large. Accessed: 2026-04-09.

[2] Jean-Baptiste Alayrac, Jef Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. 2022. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems 35 (2022), 23716–23736.

[3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. 2025. Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631 (2025).

[4] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, Wesam Manassra, Prafulla Dhariwal, Casey Chu, Yunxin Jiao, and Aditya Ramesh. 2023. Improving Image Generation with Better Captions. Technical Report. OpenAI. https://cdn.openai. com/papers/dall-e-3.pdf

[5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems 33 (2020), 1877–1901.

[6] ByteDance. 2025. Seedream 4.5. https://seed.bytedance.com/en/seedream4\_5. Accessed: 2026-03-31.

[7] Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Xin Jin, Liangchen Li, et al. 2025. Z-Image: An Eficient Image Generation Foundation Model with Single-Stream Difusion Transformer. arXiv preprint arXiv:2511.22699 (2025).

[8] Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. 2025. HunyuanImage 3.0 Technical Report. arXiv preprint arXiv:2509.23951 (2025). https://arxiv.org/abs/ 2509.23951

[9] Baoying Chen, Jishen Zeng, Jianquan Yang, and Rui Yang. 2024. DRCT: Difusion Reconstruction Contrastive Training towards Universal Detection of Difusion Generated Images. In Forty-first International Conference on Machine Learning.

[10] Mark Chen, Alec Radford, Rewon Child, Jefrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. 2020. Generative pretraining from pixels. In International conference on machine learning. PMLR, 1691–1703.

[11] Ruoxin Chen, Jiahui Gao, Kaiqing Lin, Keyue Zhang, Yandan Zhao, Isabel Guan, Taiping Yao, and Shouhong Ding. 2026. AlignGemini: Gen eralizable AI-Generated Image Detection Through Task-Model Alignment. arXiv:2512.06746 [cs.CV] https://arxiv.org/abs/2512.06746

[12] Ruoxin Chen, Junwei Xi, Zhiyuan Yan, Ke-Yue Zhang, Shuang Wu, Jingyi Xie, Xu Chen, Lei Xu, Isabel Guan, Taiping Yao, et al. 2025. Dual Data Align ment Makes AI-Generated Image Detector Easier Generalizable. arXiv preprint arXiv:2505.14359 (2025).

[13] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. Advances in neural information processing systems 30 (2017).

[14] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. 2023. InstructBLIP: Towards General-Purpose Vision-Language Models with Instruction Tuning. Advances in neural information processing systems 36 (2023), 49250–49267.

[15] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xi aohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020).

[16] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. 2024. Scaling rectified flow transformers for high-resolution image synthesis. In Fortyfirst international conference on machine learning.

[17] Google. 2022. Open Images V7. https://storage.googleapis.com/openimages/ web/factsfigures\_v7.html. Accessed: 2026-03-31.

[18] Google. 2025. Gemini 3 Flash Preview. https://ai.google.dev/gemini-api/docs/ models/gemini-3-flash-preview. Accessed: 2026-08-09.

[19] Google. 2025. Gemini 3 Pro Preview. https://ai.google.dev/gemini-api/docs/ models/gemini-3-pro-preview. Accessed: 2026-08-09.

[20] Google. 2025. Gemini Image – Nano Banana. https://deepmind.google/models/ gemini-image/. Accessed: 2026-03-31.

[21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising Diffusion Probabilistic Models. In Advances in Neural Information Processing Systems, Vol. 33. 6840–6851. https://papers.nips.cc/paper/2020/hash/ 4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html

[22] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. 2022. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 1, 2 (2022), 3.

[23] Zhenglin Huang, Tianxiao Li, Xiangtai Li, Haiquan Wen, Yiwei He, Jiangning Zhang, Hao Fei, Xi Yang, Xiaowei Huang, Bei Peng, and Guangliang Cheng. 2025. So-Fake: Benchmarking and Explaining Social Media Image Forgery Detection. arXiv preprint arXiv:2505.18660 (2025). arXiv:2505.18660 [cs.CV] https://arxiv. org/abs/2505.18660

[24] Yikun Ji, Yan Hong, Qi Fan, Jun Lan, Huijia Zhu, Weiqiang Wang, Liqing Zhang, and Jianfu Zhang. 2026. FakeXplain: AI-Generated Image Detection via Human-Aligned Grounded Reasoning. In ICLR. https://openreview.net/forum?id= UcpTOa8OnG

[25] Changjiang Jiang, Wenhui Dong, Zhonghao Zhang, Chenyang Si, Fengchang Yu, Wei Peng, Xinbin Yuan, Yifei Bi, Ming Zhao, Zian Zhou, and Caifeng Shan. 2025. IVY-FAKE: A Unified Explainable Framework and Benchmark for Image and Video AIGC Detection. arXiv preprint arXiv:2506.00979 (2025). arXiv:2506.00979 [cs.CV] https://arxiv.org/abs/2506.00979

[26] Hengrui Kang, Siwei Wen, Zichen Wen, Junyan Ye, Weijia Li, Peilin Feng, Baichuan Zhou, Bin Wang, Dahua Lin, Linfeng Zhang, and Conghui He. 2025. LEGION: Learning to Ground and Explain for Synthetic Image Detection. arXiv preprint arXiv:2503.15264 (2025). arXiv:2503.15264 [cs.CV] https://arxiv.org/abs 2503.15264

[27] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. 2023. Segment anything. In Proceedings ofthe IEEE/CVF international conference on computer vision. 4015–4026.

[28] Black Forest Labs. 2024. black-forest-labs/FLUX.1-dev. https://huggingface.co/ black-forest-labs/FLUX.1-dev. Accessed: 2026-03-31.

[29] Black Forest Labs. 2026. FLUX.2. https://bfl.ai/flux2. Accessed: 2026-04-09.

[30] Xin Lai, Junyi Li, Wei Li, Tao Liu, Tianjian Li, and Hengshuang Zhao. 2025. Mini-o3: Scaling up reasoning patterns and interaction turns for visual search. arXiv preprint arXiv:2509.07969 (2025).

[31] LAION-AI. 2022. LAION-Aesthetics\_Predictor V1. https://github.com/LAION-AI/aesthetic-predictor. Accessed: 2026-03-31.

[32] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. BLIP-2: Bootstrapping Language-Image Pre-Training with Frozen Image Encoders and Large Language Models. In International conference on machine learning. PMLR, 19730– 19742.

[33] Yixuan Li, Xuelin Liu, Xiaoyang Wang, Bu Sung Lee, Shiqi Wang, Anderson Rocha, and Weisi Lin. 2025. FakeBench: Probing Explainable Fake Image Detection via Large Multimodal Models. IEEE Transactions on Information Forensics and Security (2025).

[34] Yixuan Li, Yu Tian, Yipo Huang, Wei Lu, Shiqi Wang, Weisi Lin, and An derson Rocha. 2025. FakeScope: Large Multimodal Expert Model for Transparent AI-Generated Image Forensics. arXiv preprint arXiv:2503.24267 (2025). arXiv:2503.24267 [cs.CV] https://arxiv.org/abs/2503.24267

[35] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in neural information processing systems 36 (2023), 34892–34916.

[36] Xingchao Liu, Chengyue Gong, and Qiang Liu. 2022. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003 (2022).

[37] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. 2025. Visual-RFT: Visual Reinforcement Fine-Tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 2034– 2044.

[38] Microsoft. 2025. MNW Benchmark Dataset. https://github.com/microsoft/MNW/ tree/main. Accessed: 2026-03-31.

[39] Midjourney. 2023. Midjourney. https://www.midjourney.com/home. Accessed: 2026-03-31.

[40] OpenAI. 2025. GPT-Image-1.5 Model | OpenAI API. https://developers.openai. com/api/docs/models/gpt-image-1.5. Accessed: 2026-03-31.

[41] OpenAI. 2026. GPT Image 1 Model | OpenAI API. https://platform.openai.com/ docs/models/gpt-image-1. Accessed: 2026-04-09.

[42] Keiron O’shea and Ryan Nash. 2015. An introduction to convolutional neural networks. arXiv preprint arXiv:1511.08458 (2015).

[43] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2023. SDXL: Improving Latent Difusion Models for High-Resolution Image Synthesis. arXiv preprint arXiv:2307.01952 (2023).

[44] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems 36 (2023), 53728–53741.

[45] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. 2021. Zero-shot text-to-image generation. In International conference on machine learning. PMLR, 8821–8831.

[46] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. 2022. High-Resolution Image Synthesis with Latent Difusion Models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10674–10685. doi:10.1109/CVPR52688.2022.01042

[47] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. 2022. Photorealistic text-to-image difusion models with deep language understanding. Advances in neural information processing systems 35 (2022), 36479–36494.

[48] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017).

[49] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. 2024. Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning. Advances in Neural Information Processing Systems 37 (2024), 8612– 8642.

[50] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300 (2024).

[51] Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020. Denoising difusion implicit models. arXiv preprint arXiv:2010.02502 (2020).

[52] Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, Ping Liu, and Yunchao Wei. 2024. Rethinking the Up-Sampling Operations in CNN-Based Generative Network for Generalizable Deepfake Detection. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 28130–28139.

[53] ByteDance Seed Vision Team. 2025. Seedream 3.0 Technical Report. https:// seed.bytedance.com/en/public\_papers/seedream-3-0-technical-report. Accessed: 2026-04-09.

[54] Kimi Team. 2026. Kimi K2.5: Visual Agentic Intelligence. https://www.kimi.com/ blog/kimi-k2-5. Accessed: 2026-08-08.

[55] Aaron Van Den Oord, Oriol Vinyals, et al. 2017. Neural discrete representation learning. Advances in neural information processing systems 30 (2017).

[56] Haozhe Wang, Alex Su, Weiming Ren, Fangzhen Lin, and Wenhu Chen. 2025. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven rein forcement learning. arXiv preprint arXiv:2505.15966 (2025).

[57] Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A Efros. 2020. CNN-generated images are surprisingly easy to spot... for now. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 8695–8704.

[58] Zhendong Wang, Jianmin Bao, Wengang Zhou, Weilun Wang, Hezhen Hu, Hong Chen, and Houqiang Li. 2023. DIRE for Difusion-Generated Image Detection. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 22445– 22455.

[59] Siwei Wen, Junyan Ye, Peilin Feng, Hengrui Kang, Zichen Wen, Yize Chen, Jiang Wu, Wenjun Wu, Conghui He, and Weijia Li. 2025. Spot the Fake: Large Multimodal Model-Based Synthetic Image Detection with Artifact Explanation. arXiv preprint arXiv:2503.14905 (2025). arXiv:2503.14905 [cs.CV] https://arxiv. org/abs/2503.14905

[60] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Shengming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. 2025. Qwen-Image Technical Report. arXiv preprint arXiv:2508.02324 (2025).

[61] Penghao Wu and Saining Xie. 2024. V\*: Guided Visual Search as a Core Mechanism in Multimodal LLMs. In Proceedings ofthe IEEE/CVFConference on Computer Vision and Pattern Recognition. 13084–13094.

[62] Guowei Xu, Peng Jin, Ziang Wu, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. 2025. LLaVA-CoT: Let Vision Language Models Reason Step-by-Step. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV). IEEE Computer Society, Los Alamitos, CA, USA, 2087–2098.

[63] Zhipei Xu, Xuanyu Zhang, Runyi Li, Zecheng Tang, Qing Huang, and Jian Zhang. 2025. FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models. In ICLR. https://openreview.net/forum? id=pAQzEY7M03

[64] Shilin Yan, Ouxiang Li, Jiayin Cai, Yanbin Hao, Xiaolong Jiang, Yao Hu, and Weidi Xie. 2024. A Sanity Check for AI-Generated Image Detection. arXiv preprint arXiv:2406.19435 (2024).

[65] Junyan Ye, Baichuan Zhou, Zilong Huang, Junan Zhang, Tianyi Bai, Hengrui Kang, Jun He, Honglin Lin, Zihao Wang, Tong Wu, et al. 2024. LOKI: A Comprehensive Synthetic Data Detection Benchmark Using Large Multimodal Models. arXiv preprint arXiv:2410.09732 (2024).

[66] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. 2022. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789 2, 3 (2022), 5.

[67] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. 2025. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv preprint arXiv:2503.14476 (2025).

[68] Z.ai. 2025. GLM-4.6V: Open Source Multimodal Models with Native Tool Use. https://z.ai/blog/glm-4.6v. Accessed: 2026-03-31.

[69] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. 2025. DeepEyes: Incentivizing “Thinking with Images” via Reinforcement Learning. arXiv preprint arXiv:2505.14362 (2025).

[70] Ziyin Zhou, Yunpeng Luo, Yuanchen Wu, Ke Sun, Jiayi Ji, Ke Yan, Shouhong Ding, Xiaoshuai Sun, Yunsheng Wu, and Rongrong Ji. 2025. AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models. arXiv preprint arXiv:2507.02664 (2025). arXiv:2507.02664 [cs.CV] https://arxiv.org/abs/2507.02664

# Defake-o3: From Speculative Rationales to Verifiable Evidence for Explainable AIGI Detection

Supplementary Material

![](images/066be6a9fdd28379da5950b9a69eab62b1761198ac5db4d6fdeb1667622f4e42.jpg)  
Figure S1: Comparison of validation loss between Defake-o3 and Defake-CoT in the SFT stage.

![](images/77ba83fe095b59dc3c7403bb80e1d030a48f96ce4257329d198e9aa705de5a58.jpg)  
Figure S2: Average number of tool calls and other metrics during the RL stage of Defake-o3.

## A Prompts

In this section, we summarize all the prompts utilized throughout our study. The full content of each prompt is provided at the end of this supplementary material.

• Prompts for Inference: The system prompts used during infer ence by Defake-o3, Defake-CoT, and Defake-Direct are detailed in Figures S7, S8, and S9, respectively, and the unified user prompt is provided in Figure S10.

• Prompts for GroundFake Dataset Construction: The system and user prompts used to instruct Gemini 3 Pro Preview to generate evidence candidates are shown in Figures S11 and S12, and the prompt for trajectory rewriting is detailed in Figure S13.

• Prompts for FakeFrontier Benchmark: The prompt used for extracting descriptions for image generation is shown in Figure S14, the prompt for evidence quality evaluation is detailed in Figure S15, the three system prompts for persuasion evaluation are provided in Figures S16, S17, and S18, and the prompt for parsing baseline explanations is shown in Figure S19.

## B Robustness Evaluation

We evaluate the performance of our model on the FakeFrontier benchmark under various image degradations, including JPEG compression (Quality Factor = 70), Gaussian blur (� = 1), and resizing

(×0.5). As shown in Table S1, Defake-o3 exhibits strong robustness against these common visual perturbations.

## C Discussion on Training Dynamics

During the SFT stage, we consistently observed that the evaluation loss of the Defake-CoT model was significantly higher than that of Defake-o3 (see Figure S1), despite their reasoning text being identical. This indicates that the absence ofmagnified visual patches makes the reasoning process substantially more dificult.

Furthermore, during the RL training of Defake-o3, we observed that although we did not introduce any explicit reward for tool invocation, the number of zoom-in tool calls gradually increased after an initial decrease (see Figure S2). This suggests that the model, during the learning process, autonomously overcame the inertia of not invoking tools [69] and recognized the crucial role of the zoom-in tool in localizing generative flaws.

## D Ablation on Rule-based Reward

On the GroundFake test set, incorporating the Evidence Verifier into the reward function significantly improves the Intersection over Union (IoU) compared to the model relying solely on the rulebased reward, albeit with a slight decrease in textual metrics. A natural question arises: if we solely use the rule-based reward but increase the weight of the IoU component, can we achieve the same efect?

As shown in Table S2, increasing the ratio $\lambda _ { i o u } / \lambda _ { b l e u }$ under the rule-based reward only setting $( \alpha _ { M } = 0 )$ can improve the IoU to some extent. However, it still falls short of the spatial accuracy achieved when the Evidence Verifier is introduced $( \alpha _ { M } = 0 . 5 )$ . This confirms that the Evidence Verifier provides unique signals beyond what simple bounding box matching can ofer.

## E Base Model Performance

We report the performance ofthe pre-trained Qwen3-VL-8B-Instruct base model on the GroundFake dataset without any fine-tuning. During inference, we utilize the same prompt as Defake-Direct. The performances of Defake-Direct and Defake-o3 are also provided in Table S3 for comparison. As can be seen, our training pipeline brings substantial performance improvements over the general-purpose base model.

## F Inference Speed

We present a throughput comparison of the Defake-o3, Defake-CoT, and Defake-Direct methods, alongside three explainable baselines (FakeVLM, LEGION, and FakeShield) on an 8×A100 GPU platform. As shown in Table S4, Defake-o3 trades lower inference eficiency for higher detection performance and explanation quality, largely due to the sequential generation and multi-turn interaction nature of its “thinking-with-images” paradigm.

Table S1: Robustness evaluation on FakeFrontier.
<table><tr><td>Perturbation</td><td>Defake-o3</td><td>FakeVLM</td><td>LEGION</td><td>FakeShield</td></tr><tr><td>Original</td><td>0.9232</td><td>0.7114</td><td>0.5004</td><td>0.6127</td></tr><tr><td>JPEG Compression</td><td>0.8924</td><td>0.6891</td><td>0.5004</td><td>0.5952</td></tr><tr><td>Gaussian Blur</td><td>0.8746</td><td>0.6888</td><td>0.4991</td><td>0.6090</td></tr><tr><td>Resize</td><td>0.9182</td><td>0.6682</td><td>0.5004</td><td>0.5867</td></tr></table>

Table S2: Ablation of rule-based reward hyperparameters evaluated on the GroundFake test set.
<table><tr><td>αM</td><td> $\lambda _ { i o u } / \lambda _ { b l e u }$ </td><td>Acc</td><td>F1</td><td>BLEU-1</td><td>BLEU-2</td><td>ROUGE-L</td><td>IoU</td></tr><tr><td>0.5</td><td>1</td><td>0.992</td><td>0.992</td><td>0.364</td><td>0.215</td><td>0.281</td><td>0.311</td></tr><tr><td>0</td><td>1</td><td>0.986</td><td>0.986</td><td>0.363</td><td>0.218</td><td>0.291</td><td>0.252</td></tr><tr><td>0</td><td>2</td><td>0.952</td><td>0.950</td><td>0.321</td><td>0.190</td><td>0.266</td><td>0.237</td></tr><tr><td>0</td><td>4</td><td>0.974</td><td>0.973</td><td>0.341</td><td>0.202</td><td>0.286</td><td>0.258</td></tr></table>

Table S3: Performance of the base model compared to our trained models on GroundFake.
<table><tr><td>Model</td><td>Acc</td><td>F1</td><td>BLEU-1</td><td>BLEU-2</td><td>ROUGE-L</td><td>IoU</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>0.824</td><td>0.823</td><td>0.108</td><td>0.051</td><td>0.159</td><td>0.143</td></tr><tr><td>Defake-Direct</td><td>0.958</td><td>0.957</td><td>0.333</td><td>0.195</td><td>0.252</td><td>0.262</td></tr><tr><td>Defake-o3</td><td>0.992</td><td>0.992</td><td>0.364</td><td>0.215</td><td>0.281</td><td>0.311</td></tr></table>

Table S4: Throughput comparison during inference.
<table><tr><td>Method</td><td>Throughput (samples/second)</td></tr><tr><td>Defake-Direct</td><td>10.668</td></tr><tr><td>Defake-CoT</td><td>8.180</td></tr><tr><td>Defake-o3</td><td>2.606</td></tr><tr><td>FakeVLM</td><td>3.008</td></tr><tr><td>LEGION</td><td>7.398</td></tr><tr><td>FakeShield</td><td>0.535</td></tr></table>

## G More Details About GroundFake

## G.1 Image Collection and Bias Control

We collected a balanced training set of 16k images (8k real and 8k fake) and applied several bias control procedures to reduce obvious shortcut diferences between real and fake images.

Data Sources. Our training objective is to enable the model to identify subtle generative flaws within AI-generated images. However, if the quality of the generated images in the training set is too low and the errors are overly apparent, the resulting model may perform poorly on high-quality images produced by newer generators. Therefore, we aim to construct a generated image dataset that maintains a high average quality while presenting a gradient of dificulty. To achieve this, we scraped 6.5k generated images from the Internet, covering newer generators including SDXL [43], DALLE3 [4], Midjourney V5 [39], and Nano Banana [20]. For authentic images, we sampled 8k real images from OpenImageV7 [17].

Concurrently, we observed that recent generators typically produce images of high aesthetic quality (e.g., perfect composition and lighting). This tendency creates a stylistic discrepancy between the fake image subset and the real image subset, potentially inducing a tendency for the model to overfit to aesthetic features rather than genuine generative artifacts. To counteract this, we generated an additional 1.5k fake images using FLUX.1-dev [28] paired with a specialized custom LoRA [22]. This LoRA enables the model to simulate the diverse and often imperfect shooting conditions typical of authentic photographs, such as underexposure and motion blur. During this generation process, we utilized the captions of real images from OpenImageV7 as prompts, which effectively produced fake images that closely mimic the diverse styles of real photographs (see Figure S3). The final dataset comprises 8k generated images and 8k authentic images. All these images have a longer side of at least 1024 pixels.

Bias Control Procedures. AIGI detectors can easily rely on datasetspecific shortcuts (non-semantic biases, e.g., aesthetic style) instead of actual generation artifacts. We therefore applied the following procedures:

• Format Standardization: All images were converted to highquality JPG to remove trivial file format cues across diferent sources.

• Aspect Ratio Alignment: Among the collected fake images, square formats are overrepresented. To reduce the shortcut that square images are fake, we cropped fake images to follow the aspect ratio distribution of OpenImageV7.

• Category Alignment: We used Qwen3-VL-8B-Thinking [3] as a coarse classifier to group images into four broad categories: human, animal, object, and scene. At the same time, we filtered out clearly non-photorealistic content such as cartoons and digital art. The four categories were kept balanced at a ratio of 1:1:1:1.

• Aesthetic Quality Alignment: As mentioned above, recent generators often produce highly polished images, while real photographs span a wider quality range. To reduce the shortcut that very high aesthetic quality implies fake, we matched the aesthetic score distribution of the real subset to that of the fake subset using the LAION-Aesthetic-Predictor [31]. The additional FLUX.1-dev samples further broaden the fake subset with more diverse photographic conditions, including motion blur, noise, and imperfect lighting.

## G.2 Label-Conditioned Candidate Annotation

To guide the MLLM in generating high-quality explanations, we summarized several common AI generation errors that serve as strong, definitive evidence of synthetic origin. These include: (1) unrecognizable or garbled text; (2) missing, redundant, or distorted object components, including deformed faces or hands; (3) repetitive patterns, such as two identical faces or a single person holding two identical cups simultaneously; (4) anomalous lighting, such as areas appearing unnaturally bright without a clear light source; and (5) other violations of common sense or physical laws, such as objects floating in mid-air. We explicitly incorporated this information into the system prompt to make the model more attentive to these specific issues.

For each image, we employed Gemini 3 Pro Preview [19] to generate reasoning trajectories and evidence candidates. To improve annotation eficiency, we provided the ground-truth label of the image in the prompt. The system prompt used is detailed in Figure S11 and the user prompt is shown in Figure S12.

Nano Banana

Midjourney V5

![](images/7d2bfec4796ff17a89521b83c75442e43c7647fcebad3b7193f121a99fc2b431.jpg)  
Figure S3: Examples of images from the GroundFake dataset.

It is worth noting that we simultaneously instructed the model to generate Chinese explanatory text for the evidence to facilitate the annotation process, as all our human annotators are native Chinese speakers.

## G.3 Human Annotation and Verification

In this stage, the evidence candidates generated for the fake images in the previous step were further inspected and filtered by human annotators. This verification is necessary because the evidence generated by MLLMs frequently exhibits several issues: (1) The descriptive text is entirely irrelevant to the region enclosed by the bounding box; (2) The text contradicts the actual visual content within the bounding box (e.g., text or hands that are rendered correctly are falsely claimed to be deformed); (3) The evidence is insuficient or overly sensitive to be deemed a definitive generative flaw (e.g., claiming “the character’s skin is too smooth,” which frequently occurs in real images due to post-processing). Meanwhile, the human-annotated data obtained at this stage was also used to train our Evidence Verifier, which learned how to identify such invalid evidence.

During the annotation process, each piece ofevidence was marked as either “Valid” or “Invalid”. Annotators were instructed that a piece of evidence should only be marked as “Valid” if the text accurately matches the image region within the bounding box AND correctly points out an error that is exclusive to AI-generated images (i.e., not an artifact that could reasonably appear in real photographs due to lighting constraints, simple post-processing, or motion blur).

The annotation of each evidence item was strictly independent. Annotators were provided with the image, the bounding box, and the text in both English and Chinese. A total of 6 annotators participated in this process, with each piece of evidence independently evaluated by 3 of them. To ensure annotation quality, we organized the tasks into batches of approximately 1,000 evidence items. In each batch, we interspersed a certain proportion of evidence from real images, as well as forcibly generated bogus evidence on real images. We mandated that the “Valid” rate for these control items must not exceed 5% within any batch; otherwise, the entire batch was rejected and re-annotated.

Table S6: Persuasion Evaluation Results under Prompt I.
<table><tr><td rowspan="2">Method</td><td colspan="2">Qwen3-VL-235B</td><td colspan="2"> $\mathrm { K i m i K } 2 . 5$ </td><td colspan="2">GLM-4.6V</td></tr><tr><td>Hit@Img</td><td>Hit@Evi</td><td>Hit@Img</td><td>Hit@Evi</td><td>Hit@Img</td><td>Hit@Evi</td></tr><tr><td>Defake-03</td><td>0.9347</td><td>0.8825</td><td>0.8643</td><td>0.7292</td><td>0.9397</td><td>0.9527</td></tr><tr><td>Defake-CoT</td><td>0.9548</td><td>0.8684</td><td>0.8844</td><td>0.7521</td><td>0.9548</td><td>0.8974</td></tr><tr><td>Defake-Direct</td><td>0.7538</td><td>0.7165</td><td>0.6683</td><td>0.5890</td><td>0.7688</td><td>0.6803</td></tr><tr><td>FakeVLM</td><td>0.5895</td><td>0.4740</td><td>0.5474</td><td>0.4870</td><td>0.7053</td><td>0.7428</td></tr><tr><td>FakeShield</td><td>0.3166</td><td>0.5420</td><td>0.3116</td><td>0.5465</td><td>0.3266</td><td>0.5646</td></tr></table>

Table S5: Impact of removing invalid evidence during training.
<table><tr><td rowspan="2">Invalid Filtering</td><td rowspan="2">RL Training</td><td colspan="2">GroundFake</td><td colspan="3">FakeFrontier</td></tr><tr><td>Acc</td><td>IoU</td><td>Real Acc</td><td>Fake Acc</td><td>Acc</td></tr><tr><td>x</td><td>x</td><td>0.984</td><td>0.201</td><td>0.836</td><td>0.916</td><td>0.876</td></tr><tr><td>√</td><td>x</td><td>0.972</td><td>0.241</td><td>0.883</td><td>0.904</td><td>0.894</td></tr><tr><td>√</td><td>√</td><td>0.992</td><td>0.311</td><td>0.902</td><td>0.945</td><td>0.923</td></tr></table>

Excluding the control items from real images used for quality checks, there was an average of 2.397 pieces of evidence to be annotated per fake image. Under a majority voting rule, 74.99% of the evidence candidates were ultimately marked as “Valid”. Furthermore, all 3 annotators reached a unanimous consensus on 89.43% of the evidence items.

## G.4 Reasoning Trajectory Rewriting for Annotation Consistency

In this phase, we utilized Gemini 3 Flash Preview [18] to rewrite portions of the reasoning trajectories. This step ensures that the thought processes within the trajectories remain logically consistent with the filtered evidence (i.e., after the ‘Invalid‘ evidence was removed by human annotators). The system prompt used for this task is provided in Figure S13.

In the user prompt for this task, we provided the original image, the uncorrected output from Gemini 3 Pro Preview, and the human annotation results presented as a boolean list.

## G.5 Impact of Human Verification on Classification Accuracy

During the human annotation phase, evidence that did not align with human judgment was removed. The primary goal of this procedure was to align the model’s explanations with human cognitive standards. Another intriguing question is: does this filtering process also afect the model’s overall classification accuracy?

As shown in Table S5, removing human-annotated invalid evidence during the SFT stage understandably results in a higher IoU on the GroundFake test set, because the ground truth references have also been stripped of these invalid elements. Interestingly, this filtering also leads to an increase in overall accuracy on the OoD FakeFrontier benchmark. This improvement stems from a noticeable reduction in the false positive rate on real images. This demonstrates that removing invalid, overly sensitive, or hallucinated evidence during training fundamentally helps reduce the probability of the model misclassifying real images.

## H More Details About FakeFrontier

## H.1 Image Sources and Generation Methods

The 2,000 real images in the FakeFrontier benchmark are evenly sourced from OpenImageV7 [17] and Chameleon [64]. The 2,000 fake images are generated by 10 state-of-the-art generators: GPT-Image-1.5 [40], GPT-Image-1 [41], Seedream 4.5 [6], Seedream 3.0 [53], Z-Image-Turbo [7], Qwen-Image [60], Nano Banana [20], FLUX.2-pro [29], HunyuanImage 3.0 [8], and Stable Difusion 3.5 Large [1]. For each generator, we produced 200 images. Specifically, 100 images were generated using prompts derived from the captions of OpenImagesV7 samples, and the remaining 100 were generated using prompts based on Chameleon captions. We utilized Qwen3-VL-8B-Instruct to generate these captions, instructing the model to preserve the stylistic information (e.g., underexposure, motion blur) of the original real images. This approach ensures that the generated images mimic the diverse aesthetic conditions of real-world photography. The prompt used for generating image captions is detailed in Figure S14.

During the image generation process, we employed several standard resolutions: 1344 × 768, 1216 × 832, 1024 × 1024, 832 × 1216, and 768 × 1216. For each reference real image, its synthetic counterpart was generated using the standard resolution that most closely matched its aspect ratio. All other generation parameters were set to the models’ default or oficially recommended configurations.

## H.2 MLLM-based Protocol

We deployed Qwen3-VL-235B-A22B-Thinking, Kimi K2.5, and GLM-4.6V locally to evaluate the explanation quality. During inference, all models used their default generation parameters.

For the Quality Evaluation, we used the system prompt detailed in Figure S15. For the Persuasion Evaluation, since the susceptibility of Large Language Models to persuasion is highly dependent on the system prompt, we employed three distinct sets of system prompts. The final results presented in the main text are the average scores across these three prompts. Prompt I is detailed in Figure S16. The distinction in Prompt II (detailed in Figure S17) is the inclusion of common AI generation errors as background knowledge. The distinction in Prompt III (detailed in Figure S18) is a warning to the model that the provided evidence might be misleading, forcing it to rely more heavily on its autonomous judgment of the image content. This significantly increases the model’s skepticism; consequently, evidence that still manages to persuade the model under these conditions possesses remarkably high quality.

## H.3 Detailed Results of Persuasion Evaluation

We provide the raw persuasion evaluation results under each of the three individual prompts for reference, as shown in Tables S6, S7, and S8. It can be observed that the dificulty of persuading the MLLM to deliver a “fake” verdict progressively increases from Prompt I to Prompt III. This indicates that evidence capable of persuading the MLLM under the more stringent conditions of Prompt III is significantly more efective and reasonable. Moreover, across all diferent prompt configurations, Defake-o3 consistently maintains a clear advantage over the other baseline methods.

Table S7: Persuasion Evaluation Results under Prompt II.
<table><tr><td rowspan="2">Method</td><td colspan="2">Qwen3-VL-235B</td><td colspan="2"> $\mathrm { K i m i K } 2 . 5$ </td><td colspan="2">GLM-4.6V</td></tr><tr><td>Hit@Img</td><td>Hit@Evi</td><td>Hit@Img</td><td>Hit@Evi</td><td>Hit@Img</td><td>Hit@Evi</td></tr><tr><td>Defake-o3</td><td>0.8945</td><td>0.7455</td><td>0.8392</td><td>0.7080</td><td>0.9497</td><td>0.9462</td></tr><tr><td>Defake-CoT</td><td>0.9146</td><td>0.7607</td><td>0.8693</td><td>0.7197</td><td>0.9598</td><td>0.8923</td></tr><tr><td>Defake-Direct</td><td>0.6935</td><td>0.5543</td><td>0.6482</td><td>0.5449</td><td>0.7688</td><td>0.6787</td></tr><tr><td>FakeVLM</td><td>0.4158</td><td>0.2384</td><td>0.5316</td><td>0.4176</td><td>0.6632</td><td>0.6561</td></tr><tr><td>FakeShield</td><td>0.2764</td><td>0.3741</td><td>0.3116</td><td>0.4807</td><td>0.3266</td><td>0.5692</td></tr></table>

Table S8: Persuasion Evaluation Results under Prompt III.

<table><tr><td rowspan="2">Method</td><td colspan="2">Qwen3-VL-235B</td><td colspan="2">Kimi K2.5</td><td colspan="2">GLM-4.6V</td></tr><tr><td>Hit@Img</td><td>Hit@Evi</td><td>Hit@Img</td><td>Hit@Evi</td><td>Hit@Img</td><td>Hit@Evi</td></tr><tr><td>Defake-o3</td><td>0.5226</td><td>0.2806</td><td>0.5377</td><td>0.3312</td><td>0.8241</td><td>0.5759</td></tr><tr><td>Defake-CoT</td><td>0.4573</td><td>0.2427</td><td>0.3819</td><td>0.2393</td><td>0.7739</td><td>0.4855</td></tr><tr><td>Defake-Direct</td><td>0.3668</td><td>0.1874</td><td>0.3769</td><td>0.2173</td><td>0.5879</td><td>0.3071</td></tr><tr><td>FakeVLM</td><td>0.1053</td><td>0.0549</td><td>0.2789</td><td>0.1546</td><td>0.2684</td><td>0.1286</td></tr><tr><td>FakeShield</td><td>0.1005</td><td>0.0726</td><td>0.2010</td><td>0.2381</td><td>0.1709</td><td>0.1429</td></tr></table>

## H.4 Parsing Explanatory Text

Our explanation quality evaluation is conducted at the granularity of individual “evidence” items. Defake-o3 naturally outputs structured JSON with evidence separated into a list. However, baseline methods often output an unformatted block of text. Therefore, their explanations must be parsed and split before evaluation. We utilized Qwen3-VL-8B-Instruct with the prompt detailed in Figure S19 to achieve this.

## I Details About Baselines

Here we briefly introduce each baseline method, along with their training details or the source of their weights used in our experiments.

• CNNSpot [57] is a standard image classifier trained on real and ProGAN-generated images, with enhanced cross-generator generalization via preprocessing, postprocessing, and data aug mentation.

We trained it on GroundFake using the Adam optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 64 for 15 epochs.

• NPR [52] models neighboring pixel relationships induced by generator up-sampling operations and uses these local structural artifacts as a source-invariant representation for generalizable fake image detection.

We trained it on GroundFake using the Adam optimizer with a learning rate of $2 \times 1 0 ^ { - 4 }$ and a batch size of 32 for 10 epochs.

• AIDE [64] combines CLIP-based global semantic embeddings with DCT-guided high/low-frequency patch sampling and SRMbased noise features to detect AI-generated images using hybrid high-level and low-level cues.

We trained it on GroundFake using the AdamW optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 32 for 20 epochs.

• FakeVLM [59] is a specialized multimodal large language model trained on the FakeClue dataset that performs synthetic image classification and generates natural-language artifact explanations from fine-grained visual clues.

We used the pre-trained weights published on their oficial GitHub repository.

• LEGION [26] integrates a global image encoder, an MLLM, a grounding image encoder, and a pixel decoder to jointly perform synthetic image detection, artifact localization, and textual explanation.

We reproduced LEGION on their SynthScars dataset using their oficial training scripts and default parameters.

• FakeShield [63] combines a domain tag-guided explainable forgery detection module with a multimodal forgery localization module to predict image authenticity, localize manipulated regions, and explain tampering evidence in language.

We used the pre-trained weights published on their oficial GitHub repository.

## J More Qualitative Results

We provide additional qualitative results ofDefake-o3 on the Ground-Fake test set, along with corresponding outputs from FakeVLM and FakeShield for a comprehensive comparison of explanation quality.

As can be seen from the examples in Figures S4, S5, and S6, Defake-o3 is able to precisely localize specific generative flaws like unrecognizable objects, and easily identify surreal, obviously AIgenerated scenes, such as the two penguins taking a selfie with a smartphone in the bottom example of Figure S5.

In contrast, the explanations provided by the baseline methods are often extremely vague and unreliable. For instance, in 5 out of the 6 examples, FakeVLM outputs the exact same boilerplate explanation: "The image exhibits underlying characteristic inconsistencies in its features that suggest it is artificially created." This indicates that FakeVLM struggles to pinpoint explicit flaws and instead makes unreliable judgments based merely on overall image texture. Meanwhile, FakeShield always attempts to explain its verdicts from the perspective of lighting and shadows, even when the lighting in the generated images does not appear problematic to a human observer. Furthermore, in the bottom example of Figure S5, FakeShield mistakenly identifies the smartphone as a "mirror", demonstrating that its fine-tuning process may have compromised the model’s pre-existing world knowledge.

Input Image  
![](images/b095b7ec3909051f5a80ccf329e84539c7d581314f5bd8b30aa105c91c727dc5.jpg)  
Label: Fake

## FakeVLM

## Verdict: Fake

The image exhibits underlying characteristic inconsistencies in its features that suggest it is artificially created.

Input Image  
![](images/3fd19c280af75d94be8fad480947db3a598887b59f6773da5caa0e8bd55e7e54.jpg)  
Label: Fake  
The object on the coffee table, presumably a plant or a decorative item, has a distorted, melted appearance. It lacks a clear structure…

## FakeVLM

## Verdict: Fake

The image exhibits underlying characteristic inconsistencies in its features that suggest it is artificially created.

# Defake-o3

## Verdict: Fake

## Global Evidence:

The overall image has a hyper-realistic, overly smooth texture… particularly in the way the light interacts with the floor and the plants.

Local Evidence #1:  
![](images/b3285f505c9f560d41307876691dc386d552a043567bbba154ed23653dafb4be.jpg)

Local Evidence #2:  
![](images/970b7c56507cf578f13136c326b866145ec2d9b74f287b9971952283de024edc.jpg)

The small stools on the right side have nonsensical structures. The legs are thin, floating, or merging into the floor…

## FakeShield

## Verdict: Fake

The natural light coming from the outside appears to be inconsistent in terms of brightness and color temperature… The shadows on the floor and wall appear to be inconsistent with the direction and intensity of the light…

## Defake-o3

## Verdict: Fake

## Global Evidence:

The overall image has a hyper-realistic, overly smooth texture characteristic of AI generation, particularly in the reflections on the car body.

Local Evidence #1:  
![](images/8cf3a683efa98066b5d5792110e820537c42a0a97c101ed758320893a38b7898.jpg)

Local Evidence #2:  
![](images/2549c6de0c9397383dd6e0c357f6978724a8b4d071d42be8862e91d9f5672f56.jpg)

The logo on the hood is a distorted shield emblem with nonsensical internal details that do not resemble a real car manufacturer's logo.

The emblem on the front grille is a generic blue and silver circular logo that blends into the grille slats and lacks a coherent design.

## FakeShield

## Verdict: Fake

The shadows and highlights on the car's hood and the background appear inconsistent with the natural lighting in the rest of the image…

Figure S4: Additional qualitative comparison of Defake-o3 against FakeVLM and FakeShield on the GroundFake test set (Part 1).   
The bounding boxes are from the outputs of Defake-o3.

Label: Fake  
Input Image  
![](images/7853e6922bae192210647982385af6bcff83fd13a129f65f51f05eebcd05fe4f.jpg)  
Label: Fake

## FakeVLM

## Verdict: Fake

The image exhibits underlying characteristic inconsistencies in its features that suggest it is artificially created.

Input Image  
![](images/2489d522d118f0a3a48d29baadfe679ea396ff351dbda8b3f77525e968fe6991.jpg)

## FakeVLM

## Verdict: Fake

The penguins have unnatural proportions, with their heads and beaks appearing disproportionately large compared to their bodies…

## Defake-o3

## Verdict: Fake

## Global Evidence:

The overall image has a hyper-realistic, overly smooth aesthetic typical of AI generation, particularly in the lighting and the perfect composition.

Local Evidence #1:  
![](images/c3ba289347431a49925d0b87a664c4e9378cd97eed8cf771ec87e3801a57a1db.jpg)  
Local Evidence #2:

The metal hook connecting the cage to the wooden beam is nonsensical; it … floats without a proper attachment point.

![](images/dcef4f7609c1086c5be0cb4aa571ead9dd5379ba32bee3d3587fa79ba1cb0cf5.jpg)

The object attached to the left side of the cage is a distorted cluster of texture that resembles seeds or a decorative ornament but lacks a logical structure.

## FakeShield

## Verdict: Fake

The lighting on the bird does not match the rest of the scene. The bird appears to be lit from the front, while the rest of the scene suggests a light source coming from the right side, as indicated by the shadows on the cage…

## Defake-o3

## Verdict: Fake

## Global Evidence:

The overall image depicts a surreal scenario of penguins taking a selfie with a smartphone, which is biologically impossible…

Local Evidence #1:  
![](images/8b013c2d025f072a2434046b8413b3c5219bbbf0d01d1e2fdc393674954be9f9.jpg)  
Local Evidence #2:

![](images/13be5c428669fd34c7eeceaa0a9d59639dd9ee405b781566013af3885afcf61a.jpg)

The penguin is holding a smartphone with a flip-style case or stand that extends like a limb. The hand holding the phone is anatomically nonsensical…

The mouths of both penguins are open in a way that defies natural expression. The interior of the mouths appears distorted, with teeth-like structures…

## FakeShield

## Verdict: Fake

…The reflection in the mirror does not match the lighting of the actual penguin. The penguin's body is lit from the right side, casting a shadow to the left, while the reflection in the mirror shows a different angle of lighting…

Figure S5: Additional qualitative comparison of Defake-o3 against FakeVLM and FakeShield on the GroundFake test set (Part 2).

Input Image  
![](images/315d99cf906e346972a29f4a1e011bbd0e25d4bdf31d5a5727d2f0f746ae3294.jpg)  
Label: Fake

## FakeVLM

## Verdict: Fake

The image exhibits underlying characteristic inconsistencies in its features that suggest it is artificially created.

Input Image  
![](images/498a0505d36ffd9a87d5173badddd091b3b372447c62ca5c89396d70a7df14df.jpg)  
Label: Fake

## FakeVLM

## Verdict: Fake

The image exhibits underlying characteristic inconsistencies in its features that suggest it is artificially created.

## Defake-o3

## Verdict: Fake

## Global Evidence:

The overall image has a hyper-realistic, overly smooth quality typical of AI generation, particularly in the lighting and the perfect arrangement of the food.

Local Evidence #1:  
![](images/325a8780856ee1a977851e143d9bc8f392793db32f6b104fe13a14c5d03c4a76.jpg)

Local Evidence #2:  
![](images/3e373f2fd414da5ab63e8de20e3de89f18c42760473c27a0b36f32de48d06d0c.jpg)

There is a strange, white, dough-like object floating or standing on the black square dish in the center. It lacks a clear texture…

The cutlery on the plate is distorted. The handles of the spoons or utensils merge into the plate or each other in a nonsensical way…

## FakeShield

## Verdict: Real (×)

The lighting in the image is consistent, with the shadows falling in a natural direction, indicating a single light source. The colors and textures of the food and tableware are realistic…

## Defake-o3

## Verdict: Fake

## Global Evidence:

The overall image has a slightly oversmoothed, 'waxy' texture common in AI generation, particularly in the skin and the way objects blend into the background.

Local Evidence #1:  
![](images/2a16a85a0ba21d9d65638dd14db0df7a929d0a1f274a78655593a8b0d2a451fb.jpg)  
Local Evidence #2:

There is a long, straight metal rod sticking out of the pot that serves no clear purpose and defies physics, appearing to float…

![](images/0afccd22c51d387d851229cc99eab571d30db14fbd146c045077158216466742.jpg)

The control panel on the microwave appliance is distorted, with nonsensical, melted-looking buttons and dials instead of functional controls.

## FakeShield

## Verdict: Fake

The proportions of the person's body are distorted, which is especially noticeable around the waistline… The pot and the person's arm appear to have a different exposure level compared to the surrounding environment…

![](images/e5766a2adb976279b6bef0475a6eeb3535d84348c4050ce70fcf70f478149651.jpg)  
Figure S7: System Prompt for Defake-o3.

![](images/d8c319fb07a387c85c5e4da513847f21a8703bb2e20c6a02abfaf899ab7b2018.jpg)  
Figure S8: System Prompt for Defake-CoT.

![](images/0983a20494d7872ec060a8737edb64f1cc4c520ad5a3a8750f074ef35c479ea1.jpg)

Figure S9: System Prompt for Defake-Direct.  
![](images/5516ac7c3244b8fd98e653625172d145edbdaa2f8e8d5db772ec3789bf750c2d.jpg)  
Figure S10: User Prompt for Inference.

![](images/8cb16132331951d7cf642b9a93cc2d3d4c91422223553d7313490897386ccd66.jpg)

Figure S11: System Prompt for Evidence Candidate Generation (GroundFake).  
![](images/d37bebd4cdf0b62f511c7cb48eb71021e362d1828db12fe8e6350d01a380daed.jpg)  
Figure S12: User Prompt for Evidence Candidate Generation (GroundFake).

![](images/d59e6200c6266509fd5ff5d5987b6439390532a050f7a2c491e481899ee7be8b.jpg)  
Figure S13: System Prompt for Trajectory Rewriting (GroundFake).

You are an expert photo captioner. Describe the photo in natural language using at most two sentences. Include:   
1. A concise description of the image's style or shooting conditions in one sentence. For example, "Professional photography", "Photograph with natural lighting   
and high depth of field", "Amateur photo from the 2000s, in dim lighting, taken with a flash", "Amateur photo, even lighting, casual composition", "Amateur photo,   
with slight motion blur, handheld camera".   
2. The main subjects, their actions, the setting.   
Now give your caption directly.

Figure S14: Prompt for Image Captioning (FakeFrontier).  
![](images/b3f3f85dd46e440adef753d28d0049f203566f4b128b93433c8466f2502d6e24.jpg)

Figure S15: System Prompt for Quality Evaluation (FakeFrontier).  
![](images/812e7e0f0d3df66f0368b1deda5bf716597c332d871c13603a504c3e7eb07ae2.jpg)

Figure S16: Persuasion Evaluation Prompt I (FakeFrontier).  
![](images/dbe47e586bcceafc1c4d617b4d98da08093e32673152528c513772d2acea143f.jpg)  
Figure S17: Persuasion Evaluation Prompt II (FakeFrontier).

You are an expert in the field of forged image detection.   
For reference, here are some common errors found in AI-generated images:   
1. Unrecognizable text   
2. Missing, redundant, or distorted object components, such as deformed face, deformed limbs, extra or missing limbs, including inconsistency with logos or unique   
features of well-known brands   
3. Repeating patterns, such as two identical faces, one person holding two water cups, or a large number of people with unusually similar appearances   
4. Unusual lighting, such as unusually bright areas in the absence of light sources   
5. Other obvious anomalies that defy common sense or physics, such as objects floating in mid-air   
You will see an image which may be AI generated or not, along with a piece of evidence that attempts to prove that the image was generated by AI.   
Your task is to determine whether the image was generated by AI, referring to this evidence.   
Note that the evidence may be misleading, and you should make a verdict based on the actual content of the image.   
Provide your verdict and reasons; your final verdict should be given in a \*\*JSON text block\*\* like this:   
\`\`\`json   
{   
"verdict": "real" or "fake"   
}

Figure S18: Persuasion Evaluation Prompt III (FakeFrontier).  
![](images/f295fad4976ad1d2ce95fbd88b910fc4f317bfb23acef591dbdf8861821a084b.jpg)  
Figure S19: Prompt for Evidence Parsing.