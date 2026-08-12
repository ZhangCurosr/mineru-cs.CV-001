# RadFusion: Towards Threshold-Controllable Radiology Report Generation

Ying Jin Noel C. F. Codella John Corring Mu Wei Dinei Florencio Eric Horvitz

Microsoft

{ying.jin, ncodella, john.corring}@microsoft.com {muhsin.wei, dinei, horvitz}@microsoft.com

## Abstract

Automated radiology report generation is advancing rapidly in response to the shortage of radiologists, yet unlike a perception model, existing generation models offer no control over the sensitivity-specificity trade-off of their diagnostic content. Such control is essential because clinical scenarios diverge: emergency triage prioritizes sensitivity to reduce missed findings, whereas confirmatory interpretation emphasizes specificity to limit unnecessary interventions. A single fixed report can neither adapt to these scenarios nor support the ROC-based validation widely expected for regulatory clearance. We introduce RadFusion, a framework that equips report generation with threshold controllability. Our method fuses a multi-label classifier, which provides per-disease confidence scores, with a VQA-based report generator, which describes medical findings in detail; an LLM then rewrites the report so that its stated diagnoses follow the classifier's decisions at the selected threshold while staying grounded in the generator's descriptions. On MIMIC-CXR, the performance of RadFusion conforms to the classifier's ROC curve: sweeping the threshold and mapping the reports back to class labels reproduces the classifier's validated ROC performance. This conformance makes generated reports quantitatively evaluable through ROC analysis, strengthening the case for regulatory clearance, and enables operating-point selection that matches report behavior to clinical context. Moreover, combining the two model types improves diagnostic accuracy over uncontrolled generation: sensitivity increases by 6.9% at matched specificity, and specificity by 20.7% at matched sensitivity. These results show that RadFusion makes report generation clinically adaptable, quantitatively verifiable, and diagnostically more reliable.

## 1 Introduction

Interpreting medical images and communicating the findings in written reports is a central task in clinical care, and the growing shortage of radiologists makes its automation increasingly valuable. Clinical use, however, demands more than a fixed, plausible report: diagnostic behavior must adapt to diverse clinical scenarios and support regulatory validation. Perception models such as image classifiers meet these demands because their numerical confidence scores can be thresholded to trade off sensitivity against specificity and support ROC-based validation widely used in regulatory approval processes such as FDA clearance (Food, 2007). This threshold control is equivalent to choosing an operating point on the model's ROC curve, serving diverse clinical needs: emergency triage calls for low thresholds that prioritize sensitivity to reduce missed findings, whereas confirmatory assessments call for high thresholds that prioritize specificity to limit unnecessary interventions. However, perception models are constrained by predefined classes: a classifier may assign an 87% probability to pneumonia, but cannot describe its anatomical location, extent, or underlying cause. Report generation models offer complementary capabilities by producing expressive, open-vocabulary text that is standard in clinical reporting practice. However, these models lack confidence estimates or threshold-based mechanisms, making them difficult to accurately assess and adjust, and ultimately leaving a gap toward trustworthy, deployable clinical AI (Lin et al., 2024).

Radiology report generation is typically framed as an image captioning task and is advancing rapidly with recent progress in large vision–language models (Chaves et al., 2024; Hyland et al., 2023; Chen et al., 2024; Zhou et al., 2024; Zhang et al., 2025; Tu et al., 2024). Despite the improved performance, this captioning formulation yields a single, fixed report with no mechanism to adjust the diagnostic operating point. Although controllability and confidence estimation for text generation have been studied in the general domain, existing work does not control the generated text through class-specific thresholds. For example, one line of work (Dathathri et al., 2019; Yang and Klein, 2021) studies controllable text generation, which steers generation toward desired attributes at the token level, and another line of work (Kadavath et al., 2022; Xiong et al., 2024) focuses on the confidence-based self-assessment of candidate answers. To the best of our knowledge, threshold-controllable text generation remains a critical yet under-explored problem for clinical deployment and regulatory approval.

To fill this gap, we introduce RadFusion, a threshold-controllable radiology report generation framework that aligns the diagnostic content of generated reports with the ROC characteristics of a medical image classifier. RadFusion fuses three components: (1) a classification model that produces calibrated per-class confidence scores, (2) a VQA-based report generation model that produces detailed radiology reports and supports follow-up queries for additional details, and (3) an off-the-shelf large language model that rewrites the report so that its stated diagnoses are consistent with thresholdadjusted classification, while remaining grounded in the report generation model's findings to preserve clinical specificity. By varying the thresholds, the system produces reports spanning a range of diagnostic operating points, each corresponding to a distinct sensitivity-specificity trade-off. Moreover, since the output text is factually consistent with the classifier, the numerical confidence estimates serve as a complement to the textual report, enabling ROC-based regulatory validation and easing integration into automated systems.

RadFusion is a portable framework, generally applicable to a wide range of classifiers and report generation models. In this paper, we further study how various implementations of each component affect the final performance and identify the optimal configuration through systematic comparison. As a result, experiments on MIMIC-CXR (Johnson et al., 2019) show that our threshold-adjusted reports exhibit the expected concave ROC behavior, with area under the curve closely tracking that of the underlying classifier. In addition, threshold control yields improvements in diagnostic accuracy relative to uncontrolled generation, including a 6.9% increase in sensitivity at matched specificity and a 20.7% increase in specificity at matched sensitivity. Collectively, these results demonstrate three contributions of this work:

• Clinical adaptability: clinician-customizable reporting allows operating-point selection to match the clinical context.

• Regulatory validation: ROC-based evaluation of generated reports provides a viable path toward regulatory approval.

• Diagnostic accuracy: fusing complementary model types improves diagnostic accuracy over uncontrolled generation.

## 2 Related Work

Radiology report generation. Early work adapted encoder-decoder architectures to radiology report generation: R2Gen (Chen et al., 2020) introduced memory-driven transformers, R2GenCMN (Chen et al., 2022) added cross-modal memory, and CvT2DistilGPT2 (Nicolson et al., 2023) combined vision transformers with distilled decoders. Recent approaches leverage large vision-language models, including LLaVA-Rad (Chaves et al., 2024), MAIRA (Hyland et al., 2023; Bannur et al., 2024), CheXagent (Chen et al., 2024), MedVersa (Zhou et al., 2024), Libra (Zhang et al., 2025), MedGemma (Sellergren et al., 2025), and Med-PaLM M (Tu et al., 2024). Complementary strategies, including knowledge-driven modeling (Li et al., 2019; Zhang et al., 2020), prompt-based methods (Jin et al., 2024a), region-guided generation (Tanida et al., 2023), and retrieval augmentation (Sun et al.,

2025), have improved factual grounding. QRad (Jin et al.), a state-of-the-art model based on a pre-trained MedImageInsight (Codella et al., 2024) encoder, unifies the report generation and medical VQA tasks, and serves as the report generation component of our framework. All these models, however, produce a single, fixed report with no mechanism to adjust diagnostic behavior or select an operating point.

Medical visual question answering. Medical VQA offers an interface for extracting clinical information via natural language queries (Lau et al., 2018; He et al., 2020; Liu et al., 2021). Multimodal models such as Med-Flamingo (Moor et al., 2023) and LLaVA-Med (Li et al., 2024) adapt vision–language architectures to medicine, while Rad-ReStruct (Pellegrini et al., 2023a) proposed hierarchical VQA for structured reporting and RaDialog (Pellegrini et al., 2023b) combined report generation with multi-turn dialogue. Since omissions are common in radiology reports, our framework uses medical VQA to recover omitted information from the image, supplying grounded descriptions when a low threshold flips a finding to positive.

Controllable text generation. Controllable generation methods such as PPLM (Dathathri et al. 2019), FUDGE (Yang and Klein, 2021), and Classifier-free Guidance (Ho and Salimans, 2022) steer token probabilities during decoding toward desired attributes. While effective for stylistic or semantic control, these methods do not support class-specific diagnostic thresholds or operating-point selection.

Confidence estimation and calibration. Existing work on confidence estimation for LLMs includes logit-based methods using P(True) (Kadavath et al., 2022), verbalized confidence (Xiong et al., 2024; Li et al., 2025), sampling-based self-consistency (Wang et al., 2023), and the information geometry of chain-of-thought outputs (Lau et al., 2025). Calibration then aligns the predicted confidence with empirical correctness, typically through post-hoc temperature scaling (Guo et al., 2017; Platt, 1999) or proper scoring rules during training (Blasiok et al., 2023; Fröhlich and Williamson, 2024); recent task-specific methods partition the input space with uniform mass binning to approximate Bayes-optimal calibrators (Manggala et al., 2025). For radiology report generation, Wang et al. (2024) considered confidence during training by weighting losses with uncertainty, but offered no control mechanism at inference. Our framework builds on these foundations: per-class confidence scores are calibrated via temperature scaling so that thresholds map predictably to operating points, and logit-based P(Yes) extraction offers an alternative classifier implementation.

## 3 Method

## 3.1 Overview

RadFusion enables threshold-controllable report generation by conforming the diagnostic decisions of the generated report to those of a classification model operating at a chosen threshold, while preserving the generative model's clinical expressiveness. It is a portable framework that is applicable to a wide range of classifiers and report generation models. In this section, we first introduce the implementation that yields the optimal overall performance, followed by alternative designs for each component; the comparison among these implementations is discussed in Section 5.

As illustrated in Figure 1, RadFusion fuses three components:

1. Perception model (Section 3.2): Given a chest X-ray image, we use a classifier to produce numerical confidence scores $\{ \bar { P } ( C _ { i } = 1 ) \} _ { i = 1 } ^ { K }$ for K predefined disease classes. Given a threshold τ, it yields binary predictions $\hat { y } _ { i } = \mathbf { 1 } [ P ( C _ { i } = 1 ) > \tau ]$ , partitioning classes into positive (disease present) and negative (disease absent).

2. Report generation model (Section 3.3): Given the same image, a report generation model produces a free-text radiology report r describing both positive and negative findings. Compared to the classifier's binary labels, this open-vocabulary report contains richer information, including the anatomical location, severity, progression, and clinical context of each finding.

3. LLM rewriter (Section 3.4): An off-the-shelf large language model that combines the classifier's binary decisions Î with the report's detailed descriptions r, rewriting the report so that its stated diagnoses align with the thresholded classes while remaining grounded in the clinical attributes from the original report.

![](images/3fca732d76c39e69f67267e171b9ba4f587a00c07442ade2b693c2a36241d5d2.jpg)  
Figure 1: Overview of RadFusion, our threshold-controllable report generation framework. The classification model produces confidence scores for predefined disease categories, which are thresholded to binary decisions. The report generation model produces an initial report. The LLM rewriter combines both outputs so that the report's stated diagnoses conform to the binary classifications while preserving clinical detail. We use QRad, a report generation model supporting follow-up questions to obtain evidence omitted from the initial report.

By sweeping a threshold τ from 0 to 1 per class, RadFusion produces a family of reports spanning the full sensitivity-specificity spectrum. For evaluation, we assess the diagnostic quality of these reports across operating points with a closed-loop protocol detailed in Section 4.2.

## 3.2 Perception Model: MedImageInsight Classifier

Given a chest X-ray image, the classification model produces numerical confidence scores for 14 disease classes defined by CheXpert (Irvin et al., 2019), such as Enlarged Cardiomediastinum Cardiomegaly, and Lung Opacity. We study three different classification approaches. The baseline approach is based on MedImageInsight (MI2) (Codella et al., 2024), a medical image-text foundation model with a DaViT (Ding et al., 2022) image encoder (360M parameters) and a language encoder (252M parameters), pre-trained via contrastive learning (Yang et al., 2022) on over 3.5 million medical image-text pairs across 12 imaging domains. A classification model is fine-tuned and calibrated on MIMIC-CXR (Johnson et al., 2019) as described below. Its performance evaluation is discussed in Section 4.

Classifier Approach 1: Baseline. To adapt MI2 to the CheXpert classes, we fine-tune it with the Image-Text-Class Hybrid Contrastive loss (Jin et al., 2024b). This hybrid loss enables the model to jointly leverage image-class and image-text supervision to attain superior classification performance. The structured class labels ground the model on the disease classes of interest, while the free-text reports provide richer, fine-grained supervision that prevents the feature space from collapsing onto the coarse class labels alone. The loss function is defined as:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { i m a g e - t e x t } } + \lambda \mathcal { L } _ { \mathrm { i m a g e - c l a s s } } , } \end{array}\tag{1}
$$

where $\mathcal { L } _ { \mathrm { i m a g e - t e x t } }$ is the image-text contrastive loss based on softmax cross-entropy, and Limage-class is the image-class contrastive loss based on sigmoid cross-entropy (the sigmoid accommodates the multi-label setup). The embeddings for the classes are based on template prompts (e.g., “a chest

X-ray showing [CLASS NAME]"). We use class labels from VisualCheXbert (Jain et al., 2021). At inference, the class probabilities are computed as:

$$
\begin{array} { r } { P ( C _ { i } = 1 \mid I ) = \mathrm { s i g m o i d } [ ( \mathrm { c o s i n e \_ s i m i l a r i t y } ( f _ { I } ( I ) , \ f _ { T } ( t _ { i } ) ) / \tau _ { \mathrm { t e m p } } ) ] , } \end{array}\tag{2}
$$

where $f _ { I } , \ f _ { T }$ are the image and text encoders, $t _ { i }$ is the class prompt, and τtemp is a learnable temperature. Since the class prompts are fixed, we precompute $f _ { T } ( t _ { i } )$ as the weights of a single linear classification head, removing the text encoder at inference time.

The confidence scores are calibrated by temperature scaling (Guo et al., 2017), which involves learning a scalar $T$ that minimizes the Expected Calibration Error on a validation set. This lightweight post-hoc correction improves calibration without altering discriminative performance (AUC is unchanged).

Classifier Approach 2: Linear probing of QRad's encoder. An alternative implementation of the classifier trains a linear classification head on top of QRad's frozen image encoder, i.e., a linear probe. Since QRad's encoder is also initialized from MI2, this alternative differs from the MI2 classifier only by an additional generative fine-tuning step; moreover, at test time the classifier and the report generation model share the same encoder, reducing computational cost. The linear probe boosts performance on hard-to-detect classes (e.g. Pneumothorax) but slightly underperforms on easy-to-detect ones (e.g. Support Devices), with overall ROC-AUC comparable to the MI2 classifier, as shown in Section 5.

Classifier Approach 3: Single-token probability from QRad. QRad, being a hybrid report generation and medical VQA model, is trained with binary classification questions answered by single-token yes/no responses (e.g., “Is this image classified as [CLASS]? (yes/no)"). We extract the softmax probability over the [Yes] and [No] logits (Kadavath et al., 2022) as the class confidence. This alternative has the advantage of using a single model for both numerical confidence and textual report generation, but it underperforms a standalone classifier on most classes.

## 3.3 Report Generation Model: QRad Auto-VQA

Given a chest X-ray image, a report generation model produces a free-text radiology report that describes the patient's findings in clinical language. The report covers both positive findings (e.g., “mild bibasilar atelectasis") and pertinent negatives (e.g., “no pneumothorax"), along with attributes that binary labels cannot express, such as the location of a finding (right lower lobe vs. bilateral), its severity (mild, moderate, severe), progression relative to prior studies (stable, worsening, improved), and clinical context (e.g., post-surgical changes). Omission of some negative findings is common in radiology reports.

Conventional report generation models follow a direct image-to-text mapping $Y = f ( I )$ , producing a monolithic report with no mechanism to query additional details that may have been omitted. We use QRad (Jin et al.), which builds on the MI2 vision encoder and reframes report generation as a self-directed Visual Question Answering (Auto-VQA) process:

$$
Q = f _ { Q } ( I ) , \quad Y = f _ { A } ( I , Q ) ,\tag{3}
$$

where a Question Generator $f _ { Q }$ predicts a sequence of clinically relevant questions conditioned on the image, and an Answer Generator $f _ { A }$ answers each question to produce one sentence of the report. The final report is the concatenation of all answers. This Auto-VQA reframing allows RadFusion to query disease categories omitted in the initial report by composing questions and sending them to the Answer Generator $f _ { A } \mathrm { : }$ when a low threshold flips a class to positive, a follow-up query returns a grounded description of the finding with its location, severity, and context, rather than leaving the rewriter to hallucinate such details. The initial report and all follow-up answers together form the evidence pool for rewriting.

Alternative approaches: RadFusion is widely applicable to report generation models, such as UniRG-CXR (Liu et al., 2026), which leverages reinforcement fine-tuning. For models without a query interface, a separate VQA model may be required to recover omitted information. Because QRad serves both roles with a single model, we adopt QRad in this study as a simple yet effective implementation.

## 3.4 LLM Rewriting

The LLM rewriter combines the outputs of the two preceding components: the binary predictions $\{ \hat { y } _ { i } \} _ { i = 1 } ^ { K }$ from the thresholded classifier, and the evidence pool $\mathcal { E }$ consisting of the initial report r and optional follow-up answers. We use an off-the-shelf LLM such as GPT-5 (Singh et al., 2025) to rewrite the report¹ because modern LLMs have demonstrated strong instruction-following capabilities, avoiding the need to train a dedicated model for this task.

The rewriting instruction for the LLM plays a critical role in ensuring that its stated diagnoses are consistent with $\hat { y } _ { i }$ while remaining grounded in the clinical attributes (location, severity, progression, etc.) provided by $\mathcal { E }$ . Therefore, we structure the instruction to explicitly define the desired behavior of the LLM, including how to handle positive and negative classes, how to incorporate evidence from E, and how to maintain non-class content. Specifically:

• For each positive class $( { \hat { y } } _ { i } = 1 ) \colon$ if the finding is already present in the report, it is kept unchanged; if absent or contradicted, the LLM adds or corrects it using detail from $\mathcal { E } .$

• For each negative class $( \hat { y } _ { i } = 0 )$ : if the finding is mentioned as present, the LLM removes or negates it; if already absent, no change is needed.

• Non-class content (e.g., imaging quality, support devices not in the class set) is preserved.

An LLM capable of this task must have medical domain knowledge to reason over the disease class names and the input report. To help establish the correspondence between classes and their varied natural language manifestations in the report, we provide examples per class as part of the instruction. For example, we show that the class “Pleural Effusion" may appear in a report as “small bilateral pleural effusions", “fluid in the costophrenic angles", or “blunting of the costophrenic recesses". The full instruction is provided in Section E.

Alternative approaches: other LLMs as the rewriter. RadFusion can employ any capable LLM as the rewriter, and satisfactory performance depends on the model's medical knowledge, reasoning, and instruction-following capabilities. In Section 5, we compare alternatives spanning models of varied sizes, including GPT 5.4, GPT 5.4-mini, and DeepSeek V4 Pro, as well as varied levels of reasoning effort.

## 3.5 Extension to Three-Dimensional Threshold Control for Clinical Deployment

RadFusion's rewriting-based design extends naturally beyond a single binary threshold. In clinical practice, radiologists rarely operate on a strict positive/negative dichotomy: borderline findings are reported with explicit hedging (e.g., “cannot exclude early consolidation"), and the appropriate response to a finding depends not only on diagnostic confidence but also on its time sensitivity. To support such graded decision-making, we extend the threshold space to three dimensions $( T _ { a } , T _ { b } , T _ { c } )$ per class, which requires only modified rewriting instructions rather than any model retraining. The first two dimensions $( T _ { a } \leq T _ { b } )$ partition the confidence spectrum into three zones:

$P ( C _ { i } = 1 \mid I ) < T _ { a }$ : negative zone, where the report denies the finding.

$P ( C _ { i } = 1 \mid I ) > T _ { b }$ : positive zone, where the report asserts the finding.

$T _ { a } \leq P ( C _ { i } = 1 \mid I ) \leq T _ { b } .$ semi-positive zone, where the report describes the finding while explicitly flagging uncertainty and recommending further evaluation.

The third dimension, $T _ { c }$ , thresholds a per-class urgency score, motivated by the fact that clinical urgency does not always align with imaging severity (e.g., an early pneumothorax may demand immediate treatment); the score can be derived from clinical priors, patient context, or an auxiliary classifier, and findings exceeding $T _ { c }$ are emphasized as time-sensitive and actionable, independent of the $T _ { a } / T _ { b }$ axis. The system presented in Sections 3.1 and 3.4 is the special case with $T _ { a } = T _ { b }$ and $T _ { c }$ omitted, which we adopt for direct ROC-based comparison against the underlying classifier. This extension illustrates the extendability of the design: richer reporting policies, such as asserting findings conservatively (high $T _ { b } )$ while escalating asserted ones aggressively (low $T _ { c } )$ , are implemented by adjusting thresholds and instructions alone. We view clinician-tunable operating-point interfaces as a promising direction for future work.

![](images/39cb99d6b84454b61dd0255138ad143406337540a3c19b15e5dfbadaf94745e9.jpg)

![](images/3335b8072aff161c2fa9bc12ff95408073f82dc3f66940730d51cdc58775f8f0.jpg)

![](images/4eb4f07f846a4a7eead4fcf2a6639b979f4725f5408a1e0070d4313295ea9ec7.jpg)

![](images/e4b6b774b9cb7bf1104be2ca4580dc35ad19779eb8143bf04ca23e2ad44f174d.jpg)  
Figure 2: Threshold-controllable report generation on four representative classes. Cardiomegaly: common, both models perform strongly. Edema: threshold tuning is clinically valuable since mild cases often need no treatment. Atelectasis: challenging, requires fine visual details. Pleural Other: rare (1.3% prevalence). Across all four, the report ROC (red) closely tracks the classifier (blue), and the non-controlled report (green cross) falls below the curves.

## 4 Experiments

## 4.1 Dataset and Setup

We conduct experiments on MIMIC-CXR (Johnson et al., 2019), one of the largest radiology report datasets containing 227,835 studies with 377,110 chest X-ray images. Following standard practice (Chaves et al., 2024; Hyland et al., 2023), we use the Findings section as the generation target. All evaluations use the official test split (2,347 studies).

## 4.2 Evaluation Methods

To assess the threshold controllability of RadFusion, we implement a closed-loop evaluation protocol that runs from the classifier's thresholded decisions, through the rewritten report, and back to class labels, which are then compared against the original decisions for alignment. Given a threshold τ, the classifier's confidence scores are binarized into positive and negative class lists, and the LLM rewrites QRad's initial report to conform to these classifications, as described in Section 3.4. To evaluate, we convert the rewritten report back to binary class labels using GPT-5, which we found more reliable than VisualCheXbert (Jain et al., 2021), a BERT-based text classification model commonly used to extract disease classes from radiology reports. Comparing the extracted labels against the ground truth yields the true positive rate (sensitivity) and false positive rate (1 — specificity) of the rewritten reports at threshold τ.

We sweep τ from 0.0 to 1.0 with a step size of 0.1, yielding 11 operating points.2 The same τ is applied across all 13 disease classes (“No Finding" is excluded because it semantically overlaps with the absence of other classes). Connecting these operating points traces the ROC curve of the threshold-controlled reports. The ROC curve plots TPR against FPR across thresholds, and each of its points corresponds to a specific sensitivity-specificity trade-off, making ROC the natural evaluation for threshold-controllable systems. It is also the metric widely expected by regulatory approval processes for medical devices. The area under the curve (AUC) summarizes the discriminative performance across all operating points on the ROC curve. We compare both the ROC curves and their AUC among three systems (Figures 2 and 4):

• The classifier (blue curves): ROC computed directly from the classifier's continuous confidence scores, the upper bound we aim to conform to.

• Threshold-controlled reports (red curves): ROC computed by converting each thresholdadjusted report back to class labels. Each point corresponds to one operating point specified by a threshold.

• Non-controlled reports (green cross): The single operating point for the generated report without threshold adjustments.

Table 1: AUC-ROC of the three classifier implementations. MI2 FT: the fine-tuned classifier based on MedImageInsight; QRad Linear Prob: fitting a linear classification head on QRad's frozen encoder; QRad Token Logit: single-token yes/no probabilities using QRad's VQA capability. Results show AUC-ROC across the 13 disease classes, with AVG their macro average.
<table><tr><td>AUC-ROC</td><td>Enl. Car. L.O. L.L. Ede. Con. Pmn. Ate. Pmt. P.E. P.O. Fra. S.D.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>AVG</td></tr><tr><td>MI2 FT</td><td>.93</td><td>.92</td><td>.92</td><td>.85</td><td>.92</td><td>.91</td><td>.88</td><td>.90</td><td>.88</td><td>.90.90</td><td></td><td>.89</td><td>.95</td><td>.90</td></tr><tr><td>QRad Linear Prob .92</td><td></td><td>.91</td><td>.92</td><td>.89</td><td>.91</td><td>.91</td><td>.89</td><td>.91</td><td>.95</td><td>.92</td><td>.90</td><td>.87</td><td>.91</td><td>.91</td></tr><tr><td>QRad Token Logit .82</td><td></td><td>.81</td><td>.83</td><td>.73</td><td>.78</td><td>.78</td><td>.71</td><td>.74</td><td>.68</td><td>.78</td><td>.58</td><td>.56</td><td>.65</td><td>.73</td></tr></table>

## 4.3 Results and Discussion

Overall, the results demonstrate that RadFusion delivers effective threshold control: the thresholdcontrolled reports closely conform to the classifier's ROC curves, improve diagnostic accuracy over non-controlled generation, and expose a tunable sensitivity-specificity trade-off that serves diverse clinical scenarios. We highlight three key findings:

Conformance to classifier ROC. Figure 2 highlights four classes spanning a range of difficulty and prevalence: Cardiomegaly (common), Edema (threshold-sensitive), Atelectasis (hard), and Pleural Other (rare, 1.3%). Across all four, the report ROC closely tracks the classifier ROC with comparable AUC, and the concave shape confirms a monotonic sensitivity-specificity trade-off; the full 13-class evaluation (Figure 4) shows the same pattern. This close overlap validates that the threshold effectively trades off sensitivity and specificity, and that the rewritten reports conform to the classifier's decisions across the operating range.

Improved factual correctness. The ROC curves of threshold-controlled reports are above the noncontrolled reports (green crosses): at matched specificity, sensitivity improves by 6.9%; at matched sensitivity, specificity improves by 20.7%. By combining the two model types, threshold control transfers the classifier's superior discriminative performance to the generated reports.

Clinical utility. Low thresholds prioritize sensitivity (screening/triage); high thresholds prioritize specificity (confirmatory assessment). Our framework makes this trade-off explicit and tunable, replacing the fixed, opaque operating point of conventional models and offering a path toward ROC-based regulatory evaluation of generated reports.

## 5 Ablation Studies

## 5.1 Alternative Classifier Implementations

Here we compare the three classifier implementations detailed in Section 3.2: a standalone classifier fine-tuned from MI2 (our default), a linear probe of QRad's frozen encoder (sharing the encoder with report generation), and single-token yes/no probabilities extracted from QRad's VQA answers (a single model for classification and report generation). All three implementations use the same MedImageInsight pre-trained image encoder, and QRad further fine-tunes the encoder for report generation.

Table 1 reports their per-class AUC-ROC, and their ROC curves are provided in Section C. The MI2 classifier and the QRad linear probe achieve comparable overall AUC, each stronger on a different subset of classes, while the QRad token-logit variant trails on most classes. Notably, the QRad Linear Prob implementation significantly improves on hard classes like Pneumothorax, possibly because the report generation objective enhances the encoder's visual representation to better capture subtle visual cues. The MI2 FT classifier is stronger on visually obvious classes like Support Devices, possibly due to its pretraining on a large corpus of radiology images. RadFusion uses the MI2 FT classifier by default to demonstrate a common use case where separate classifiers and report generators are combined for better utility than either alone.

## 5.2 Alternative LLM Rewriters

Model size and rewriting instruction. We compare LLM rewriters across model sizes (GPT-5, GPT-5.4, and GPT-5.4-mini) and rewriting instructions, with and without the per-class example text segments (Section 3.4); by default, the instruction includes the examples. Table 2 reports the AUC-ROC of the threshold-controlled reports for each configuration. Removing the examples degrades performance: for GPT-5, the AUC-ROC on Fracture (Fra.) drops from .89 to .73 and the macro average from .90 to .88. This confirms that grounding the class-text correspondence with concrete examples is important for faithful rewriting, consistent with our analysis in Section 3.4. We further observe that a newer model generation does not necessarily outperform its predecessor under the same prompt: GPT-5.4 does not exceed GPT-5. This suggests that different models may require specifically optimized instructions for the optimal performance (the instruction was tuned on GPT-5).

Table 2: Comparison of LLM rewriter implementations across model sizes and rewriting instructions. By default, the instruction includes the per-class example text segments; “w/o examples" removes them. The table shows AUC-ROC of the threshold-controlled reports across the 13 disease classes with AVG their macro average. Within each model group (separated by midrules), the higher AVG is in bold.
<table><tr><td>Model</td><td>Reasoning Enl.</td><td></td><td>Car.</td><td>L.O.</td><td>L.L.</td><td>Ede.</td><td>Con.</td><td>Pmn.</td><td>Ate.</td><td>Pmt.</td><td>P.E.</td><td>P.O.</td><td>Fra.</td><td>S.D.</td><td>AVG</td></tr><tr><td>GPT-5</td><td>default</td><td>.93</td><td>.92</td><td>.92</td><td>.84</td><td>.92</td><td>.91</td><td>.88</td><td>.90</td><td>.88</td><td>.90</td><td>.90</td><td>.89</td><td>.93</td><td>.90</td></tr><tr><td>GPT-5 w/o examples default</td><td></td><td>.88</td><td>.92</td><td>.91</td><td>.84</td><td>.91</td><td>.91</td><td>.88</td><td>.90</td><td>.88</td><td>.90</td><td>.90</td><td>.73</td><td>.89</td><td>.88</td></tr><tr><td>GPT-5.4</td><td>medium</td><td>.93</td><td>.92</td><td>.92</td><td>.75</td><td>.92</td><td>.91</td><td>.88</td><td>.90</td><td>.88</td><td>.90</td><td>.90</td><td>.89</td><td>.94</td><td>.90</td></tr><tr><td>GPT-5.4-mini</td><td>medium</td><td>.93</td><td>.92</td><td>.92</td><td>.82</td><td>.90</td><td>.90</td><td>.88</td><td>.90</td><td>.84</td><td>.90</td><td>.90</td><td>.88</td><td>.93</td><td>.89</td></tr></table>

Table 3: Comparison of LLM rewriter implementations across model providers (OpenAI and DeepSeek) and reasoning efforts, using the QRad Linear Prob classifier as the perception model for additional context. The pattern is consistent with Table 2, and reasoning effort is vital especially for smaller models. The table shows AUC-ROC of the threshold-controlled reports across the 13 disease classes, with AVG their macro average. Within each model group (separated by midrules), the higher AVG is in bold. DS-V4-Pro denotes DeepSeek V4 Pro (Xu et al., 2026).
<table><tr><td>Model</td><td>Reasoning</td><td>Enl.</td><td>Car.</td><td>r. L.O. L.L. Ede. Con. Pmn. Ate. Pmt. P.E. P.O. Fra. S.D.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>AVG</td></tr><tr><td>GPT-5</td><td>default</td><td>.92</td><td>.91</td><td>.91</td><td>.85</td><td>.91</td><td>.90</td><td>.88</td><td>.90</td><td>.94</td><td>.91</td><td>.89</td><td>.86</td><td>.89</td><td>.90</td></tr><tr><td>GPT-5.4</td><td>default</td><td>.92</td><td>.91</td><td>.91</td><td>.73</td><td>.91</td><td>.90</td><td>.88</td><td>.90</td><td>.94</td><td>.91</td><td>.87</td><td>.84</td><td>.85</td><td>.88</td></tr><tr><td>GPT-5.4</td><td>medium</td><td>.92</td><td>.91</td><td>.91</td><td>.73</td><td>.91</td><td>.90</td><td>.89</td><td>.90</td><td>.94</td><td>.91</td><td>.89</td><td>.86</td><td>.88</td><td>.89</td></tr><tr><td>GPT-5.4-mini default</td><td></td><td>.86</td><td>.89</td><td>.83</td><td>.58</td><td>.77</td><td>.63</td><td>.67</td><td>.85</td><td>.65</td><td>.85</td><td>.85</td><td>.56</td><td>.85</td><td>.76</td></tr><tr><td>GPT-5.4-mini medium</td><td></td><td>.92</td><td>.91</td><td>.91</td><td>.81</td><td>.90</td><td>.90</td><td>.87</td><td>.90</td><td>.88</td><td>.90</td><td>.89</td><td>.85</td><td>.88</td><td>.89</td></tr><tr><td>DS-V4-Pro</td><td>default</td><td>.91</td><td>.91</td><td>.89</td><td>.77</td><td>.89</td><td>.89</td><td>.85</td><td>.89</td><td>.87</td><td>.89</td><td>.87</td><td>.85</td><td>.85</td><td>.87</td></tr></table>

Model provider and reasoning effort. Table 3 extends the comparison across model providers (OpenAI and DeepSeek) and reasoning efforts, using the QRad Linear Prob classifier as the perception model for additional context. The results exhibit a consistent pattern with Table 2, and further show that reasoning effort is vital, especially for smaller models. Notably, both GPT-5.4 and GPT-5.4-mini experience significant degradation on Lung Lesion (L.L.), but raising the reasoning effort to medium improves GPT-5.4-mini from .58 to .81 on this class.

## 6 Conclusion

We introduced RadFusion, a threshold-controllable radiology report generation framework that fuses a classifier, a VQA-based report generator, and an LLM rewriter. By conforming the report's diagnostic content to the classifier's threshold-adjusted predictions, RadFusion produces reports whose performance tracks the classifier's ROC curve while preserving rich clinical detail, so that a single threshold selects the operating point along the sensitivity-specificity spectrum. On MIMIC-CXR, this conformance improves sensitivity by 6.9% at matched specificity and specificity by 20.7% at matched sensitivity over non-controlled generation. To our knowledge, this is the first framework to achieve threshold control in report generation, making generated reports clinically adaptable, quantitatively evaluable, and diagnostically more accurate, and offering a practical path toward regulatory validation and clinician-customizable AI-assisted radiology.

## 7 Broader Impact

RadFusion can improve the safety and clinical utility of AI-assisted radiology. By making the sensitivity-specificity trade-off explicit and tunable, it lets clinicians match model behavior to a clinical context (high sensitivity for screening, high specificity for confirmatory assessment) rather than relying on a fixed, opaque operating point, enabling responsible deployment across settings with diverse risk tolerances. Because each report is paired with the classifier's per-class confidence scores, the framework also supports the ROC-based evaluations widely expected in regulatory pathways such as FDA device authorization Food (2007) and eases integration into rule-based clinical decision systems. Beyond radiology, conforming generative text to a perception model's structured decisions applies to any domain where generated text must make decisions with tunable risk tolerance, such as autonomous systems, content moderation, and legal compliance.

Limitations and risks. RadFusion relies on the quality of all three components: classifier errors propagate into the rewritten report, the report generator may omit findings, and the LLM rewriter may introduce subtle linguistic artifacts. The rewriting instruction is template-based and tuned for chest X-ray findings; our ablations show that its effectiveness varies across LLMs, so it would require re-tuning for a different rewriter, imaging modality, or clinical domain. Threshold control also operates over a predefined set of disease classes, and findings outside this set are not subject to threshold adjustment. Finally, our evaluation relies on automatic label extraction from reports, which may itself introduce error. We emphasize that our system is intended to assist, not replace, radiologist review.

## References

Shruthi Bannur, Kenza Bouzid, Daniel C Castro, Anton Schwaighofer, Sam Bond-Taylor, Maximilian Ilse, Fernando Pérez-García, Valentina Salvatelli, Harshita Sharma, Felix Meissen, et al. MAIRA-2: Grounded Radiology Report Generation. arXiv preprint arXiv:2406.04449, 2024.

Jaroslaw Blasiok, Parikshit Gopalan, Lunjia Hu, and Preetum Nakkiran. When does optimizing a proper loss yield calibration? Advances in Neural Information Processing Systems, 36:72071– 72095, 2023.

Juan Manuel Zambrano Chaves, Shih-Cheng Huang, Yanbo Xu, Hanwen Xu, Naoto Usuyama, Sheng Zhang, Fei Wang, Yujia Xie, Mahmoud Khademi, Ziyi Yang, Hany Hassan Awadalla, Julia Gong, Houdong Hu, Jianwei Yang, Chunyuan Li, Jianfeng Gao, Yu Gu, Cliff Wong, Mu-Hsin Wei, Tristan Naumann, Muhao Chen, Matthew P. Lungren, Serena Yeung-Levy, Curtis P. Langlotz, Sheng Wang, and Hoifung Poon. Towards a clinically accessible radiology foundation model: open-access and lightweight, with automated evaluation. 2024.

Zhihong Chen, Yan Song, Tsung-Hui Chang, and Xiang Wan. Generating radiology reports via memory-driven transformer. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, 2020.

Zhihong Chen, Yaling Shen, Yan Song, and Xiang Wan. Cross-modal memory networks for radiology report generation. arXiv preprint arXiv:2204.13258, 2022.

Zhihong Chen, Maya Varma, Jean-Benoit Delbrouck, Magdalini Paschali, Louis Blankemeier, Dave Van Veen, Jeya Maria Jose Valanarasu, Alaa Youssef, Joseph Paul Cohen, Eduardo Pontes Reis et al. Chexagent: Towards a foundation model for chest x-ray interpretation. arXiv preprint arXiv:2401.12208, 2024.

Noel C. F. Codella, Ying Jin, Shrey Jain, Yu Gu, Ho Hin Lee, Asma Ben Abacha, Alberto Santamaria-Pang, Will Guyman, Naiteek Sangani, Sheng Zhang, Hoifung Poon, Stephanie Hyland, Shruthi Bannur, Javier Alvarez-Valle, Xue Li, John Garrett, Alan McMillan, Gaurav Rajguru, Madhu Maddi, Nilesh Vijayrania, Rehaan Bhimai, Nick Mecklenburg, Rupal Jain, Daniel Holstein, Naveen Gaur, Vijay Aski, Jenq-Neng Hwang, Thomas Lin, Ivan Tarapov, Matthew Lungren, and Mu Wei. Medimageinsight: An open-source embedding model for general domain medical imaging, 2024.

Sumanth Dathathri, Andrea Madotto, Janice Lan, Jane Hung, Eric Frank, Piero Molino, Jason Yosinski, and Rosanne Liu. Plug and play language models: A simple approach to controlled text generation. arXiv preprint arXiv:1912.02164, 2019.

Mingyu Ding, Bin Xiao, Noel Codella, Ping Luo, Jingdong Wang, and Lu Yuan. DaViT: Dual attention vision transformers. In European conference on computer vision, pages 74–92. Springer, 2022.

US Food. Drug administration. statistical guidance on reporting results from studies evaluating diagnostic tests-guidance for industry and fda staff. Food and Drug Administration, Center for Devices and Radiological Health Diagnostic Devices Branch, Division of Biostatistics, Office of Surveillance and Biometrics, 2007.

Christian Fröhlich and Robert C Williamson. Scoring rules and calibration for imprecise probabilities. arXiv preprint arXiv:2410.23001, 2024.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In International Conference on Machine Learning, pages 1321–1330, 2017.

Xuehai He, Yichen Zhang, Luntian Mou, Eric Xing, and Pengtao Xie. PathVQA: 30000+ questions for medical visual question answering. arXiv preprint arXiv:2003.10286, 2020.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Stephanie L Hyland, Shruthi Bannur, Kenza Bouzid, Daniel C Castro, Mercy Ranjit, Anton Schwaighofer, Fernando Pérez-García, Valentina Salvatelli, Shaury Srivastav, Anja Thieme, et al. MAIRA-1: A specialised large multimodal model for radiology report generation. arXiv preprint arXiv:2311.13668, 2023.

Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, et al. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In Proceedings of the AAAl conference on artificial intelligence, pages 590–597, 2019.

Saahil Jain, Akshay Smit, Steven QH Truong, Chanh DT Nguyen, Minh-Thanh Huynh, Mudit Jain, Victoria A Young, Andrew Y Ng, Matthew P Lungren, and Pranav Rajpurkar. Visualchexbert: addressing the discrepancy between radiology report labels and image labels. In Proceedings of the Conference on Health, Inference, and Learning, pages 105–115, 2021.

Haibo Jin, Haoxuan Che, Yi Lin, and Hao Chen. Promptmrg: Diagnosis-driven prompts for medical report generation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2607– 2615, 2024a.

Ying Jin, Noel C Codella, Yanbo Xu, Yu Gu, Mu Wei, Haoquan Fang, Thomas Lin, Paul Vozila, and Jenq-Neng Hwang. Qrad: Enhancing radiology report generation by captioning-to-vqa reframing. In The Second Workshop on GenAI for Health: Potential, Trust, and Policy Compliance.

Ying Jin, Zhuoran Zhou, Haoquan Fang, and Jenq-Neng Hwang. Daug: Diffusion-based channel augmentation for radiology image retrieval and classification. In Advancements In Medical Foundation Models: Explainability, Robustness, Security, and Beyond, 2024b.

Alistair EW Johnson, Tom J Pollard, Seth J Berkowitz, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Roger G Mark, and Steven Horng. Mimic-cxr, a de-identified publicly available database of chest radiographs with free-text reports. Scientific data, 6(1):317, 2019.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

Gregory Kang Ruey Lau, Hieu Dao, and Bryan Kian Hsiang Low. Uncertainty quantification for MLLMs. In ICLR Workshop: Quantify Uncertainty and Hallucination in Foundation Models: The Next Frontier in Reliable AI, 2025.

Jason J. Lau, Soumya Gayen, Asma Ben Abacha, and Dina Demner-Fushman. A dataset of clinically generated visual questions and answers about radiology images. In Scientific Data, 2018.

Changlin Li, Zhihui Li, Zongyuan Ge, and Mingjie Li. Knowledge driven temporal activity localization. Journal of Visual Communication and Image Representation, 64:102628, 2019.

Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. LLava-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day. Advances in Neural Information Processing Systems, 36 2024.

Yibo Li, Miao Xiong, Jiaying Wu, and Bryan Hooi. Conftuner: Training large language models to express their confidence verbally. arXiv preprint arXiv:2508.18847, 2025.

Zichao Lin, Shuyan Guan, Wending Zhang, Huiyan Zhang, Yugang Li, and Huaping Zhang. Towards trustworthy LLMs: a review on debiasing and dehallucinating in large language models. Artificial Intelligence Review, 2024.

Bo Liu, Li-Ming Zhan, Li Xu, Lin Ma, Yan Yang, and Xiao-Ming Wu. SLAKE: A semanticallylabeled knowledge-enhanced dataset for medical visual question answering. arXiv preprint arXiv:2102.09542, 2021.

Qianchu Liu, Sheng Zhang, Guanghui Qin, Yu Gu, Ying Jin, Sam Preston, Yanbo Xu, Sid Kiblawi, Wen-wai Yim, Tim Ossowski, et al. Scaling medical imaging report generation with multimodal reinforcement learning. arXiv preprint arXiv:2601.17151, 2026.

Putra Manggala, Atalanti A. Mastakouri, Elke Kirschbaum, Shiva Kasiviswanathan, and Aaditya Ramdas. QA-calibration of language model confidence scores. In The Thirteenth International Conference on Learning Representations, 2025.

Michael Moor, Qian Huang, Shirley Wu, Michihiro Yasunaga, Yash Dalmia, Jure Leskovec, Cyril Zakka, Eduardo Pontes Reis, and Pranav Rajpurkar. Med-flamingo: a multimodal medical few-shot learner. In Machine Learning for Health, ML4H@NeurIPS 2023, 10 December 2023, New Orleans, Louisiana, USA, pages 353–367. PMLR, 2023.

Aaron Nicolson, Jason Dowling, and Bevan Koopman. Improving chest x-ray report generation by leveraging warm starting. Artificial intelligence in medicine, 144:102633, 2023.

Chantal Pellegrini, Matthias Keicher, Ege Özsoy, and Nassir Navab. Rad-restruct: A novel vqa benchmark and method for structured radiology reporting. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 409–419. Springer, 2023a.

Chantal Pellegrini, Ege Özsoy, Benjamin Busam, Nassir Navab, and Matthias Keicher. Radialog: A large vision-language model for radiology report generation and conversational assistance. arXiv preprint arXiv:2311.18681, 2023b.

John Platt. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In Advances in Large Margin Classifiers. MIT Press, 1999.

Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri, Atilla Kiraly, Madeleine Traverse, Timo Kohlberger, Shawn Xu, Fayaz Jamil, Cían Hughes, Charles Lau, et al. Medgemma technical report. arXiv preprint arXiv:2507.05201, 2025.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

Liwen Sun, James Jialun Zhao, Wenjing Han, and Chenyan Xiong. Fact-aware multimodal retrieval augmentation for accurate medical radiology report generation. In Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics, 2025.

Tim Tanida, Philip Müller, Georgios Kaissis, and Daniel Rueckert. Interactive and explainable regionguided radiology report generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7433–7442, 2023.

Tao Tu, Shekoofeh Azizi, Danny Driess, Mike Schaekermann, Mohamed Amin, Pi-Chuan Chang Andrew Carroll, Charles Lau, Ryutaro Tanno, Ira Ktena, et al. Towards Generalist Biomedical AI. NEJM AI, 1(3):AIoa2300138, 2024.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations, 2023.

Yixin Wang, Zihao Lin, Jianzong He, Min Zhang, Ling Wang, Shuwen Li, Jingyu Li, S. Kevin Zhou, and Zhiqiang He. Trust it or not: Confidence-guided automatic radiology report generation. Neurocomputing, 2024.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. Can LLMs express their uncertainty? an empirical evaluation of confidence elicitation in LLMs. In International Conference on Learning Representations, 2024.

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly efficient milliontoken context intelligence. arXiv preprint arXiv:2606.19348, 2026.

Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Bin Xiao, Ce Liu, Lu Yuan, and Jianfeng Gao. Unified Contrastive Learning in Image-Text-Label Space, 2022.

Kevin Yang and Dan Klein. FUDGE: Controlled text generation with future discriminators. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3511–3535, 2021.

Xi Zhang, Zaiqiao Meng, Jake Lever, and Edmond SL Ho. Libra: Leveraging temporal images for biomedical radiology analysis. In Findings of the Association for Computational Linguistics: ACL 2025, pages 17275–17303, 2025.

Yixiao Zhang, Xiaosong Wang, Ziyue Xu, Qihang Yu, Alan Yuille, and Daguang Xu. When radiology report generation meets knowledge graph. Proceedings of the AAAI Conference on Artificial Intelligence, 2020.

Hong-Yu Zhou, Julián Nicolás Acosta, Subathra Adithan, Suvrankar Datta, Eric J Topol, and Pranav Rajpurkar. Medversa: A generalist foundation model for medical image interpretation. arXiv preprint arXiv:2405.07988, 2024.

## A Sample Clinical User Interface

For additional context, we provide a sample user interface that demonstrates how RadFusion assists the daily radiology reporting process. As shown in Figure 3, the radiograph and an automatically generated report are presented to the user. Alongside them, a set of threshold sliders for the disease classes is shown, each annotated with the model's predicted confidence. When the radiologist adjusts a slider, the corresponding operating point is highlighted on that class's ROC curve, giving the user immediate context on the implied sensitivity-specificity trade-off. The adjusted report is displayed on the bottom right, where hovering the mouse over each text segment (color-underlined) reveals the matching disease class, its predicted confidence, and the current threshold. In this example, the user tuned down the threshold for Edema, which turns the class positive (third sentence, orange underlined) in the adjusted report.

![](images/7bc48d02ef66024f13ba94a88eda768bdef1732ee17b7182926089eac0995cbc.jpg)  
Figure 3: A sample clinical user interface for RadFusion. The radiograph and the automatically generated report are shown to the user, together with per-class threshold sliders annotated with the model's predicted confidence. Adjusting a slider highlights the corresponding operating point on the class's ROC curve, conveying the implied sensitivity-specificity trade-off. The adjusted report (bottom right) links each color-underlined text segment to its disease class, predicted confidence, and threshold on hover. Here, lowering the Edema threshold turns the class positive (third sentence, orange underlined)

## B Threshold-Control Evaluation on All Disease Classes

Figure 4 presents the full threshold-control evaluation on all 13 disease classes, extending the four representative classes highlighted in the main text (Figure 2). For each class, we compare the classifier ROC (blue), computed directly from the classifier's continuous confidence scores, against the ROC of the threshold-controlled reports (red), obtained by sweeping τ from 0.0 to 1.0 and converting each rewritten report back to class labels, as described in Section 4.2. The green cross marks the single operating point of the non-controlled report. Across all classes, the report ROC closely tracks the classifier ROC, and the non-controlled report falls below the curves, confirming that the conformance and factual-correctness improvements reported in Section 4 hold consistently across the full label set rather than only the highlighted classes.

## C Comparison of Classifier Implementations

Figure 5 shows the per-class ROC curves of the two strongest classifier implementations compared in Section 5.1: the fine-tuned MI2 classifier (MI2 FT) and a linear probe of QRad's frozen encoder (QRad Linear Prob). These curves provide the full operating-point view underlying the summary AUC-ROC reported in Table 1. Consistent with the table, the two track each other closely across most classes, with QRad Linear Prob pulling ahead on hard classes such as Pneumothorax and MI2 FT stronger on visually obvious classes such as Support Devices.

## D Threshold-Control Evaluation with the Alternative Classifier

The threshold-control evaluation in Section B uses the default MI2 FT classifier as the perception model. Here we repeat it with the QRad Linear Prob classifier (Section C), an alternative implementation that shares its encoder with the report generator. Figure 6 shows the per-class threshold-control evaluation across all 13 disease classes, with GPT-5 as the rewriter. As with the default classifier, the report ROC (red) closely tracks the classifier ROC (blue), and the non-controlled report (green cross) falls below the curves, confirming that RadFusion's threshold conformance holds across the underlying classifier.

## E LLM Rewriting Instruction

Figure 7 shows the full instruction prompt provided to the LLM rewriter. The prompt consists of three parts: (1) a task description specifying the rewriting objective, (2) a template dictionary providing example positive and negative text segments for each of the 14 CheXbert disease classes, and (3) worked examples illustrating the rewriting rules for different cases. At inference, the placeholder {INPUT} is replaced with the original report and the threshold-adjusted classification results.

![](images/59861d22ec43a78fdce004c1dbc92a831c7823849e57773bdddfa3a3105841ab.jpg)  
Figure 4: Threshold-controllable report generation on all 13 disease classes. High overlap between classifier ROC (blue) and report ROC (red) confirms conformance. The non-controlled report (green cross) falls below the curves. Ground-truth labels are obtained from the reference reports via VisualCheXbert.

![](images/e98e684a2fb3191b4a9ac29d2e3f1b3928caa0caa74a83d32250287f904d9a6b.jpg)

![](images/4507e5498bd0f358926eec1a5b4e42022eac75da041dd5d52d4ee6a5b9a9ef67.jpg)

![](images/10ffdf6ce2a1429e2f0558ffb547723deff65daf1da6b4094ea46ddd532f5bf3.jpg)

![](images/d98e1f04c62cc1ac4f1ebd4dbf38859644c3a05ff3b681235c8e8336e7337328.jpg)

![](images/a664781a45fcc879532f4d967065b10c649e1cf9957a5fff5db32b9cc27577ee.jpg)

![](images/34228d6533d3076bfd62503169aec23fb80454ca6f7d4f94841b8ffd9879ce4b.jpg)

![](images/71dca290efcba295ebff90284d0ac3638f126a3207d6aff3e392a69ce09d197d.jpg)

![](images/1e772724b87275a86867ff08d35e3361ef7937336ca87cbf909295e4dc4792c9.jpg)

![](images/b927e1d336c80303b09739ac17cd25b733628129bf3e3bf4efb067f44f28194f.jpg)

![](images/8a7a4f354837f597cb54d9fe480643779bcab791654551e1935b1c6afefdba15.jpg)

![](images/7e38b0cbaba6575d2d1cc58fc886d5ca2595d200f0338832d3d63d911a9fce7c.jpg)

![](images/394828bb6ef749975036ef0acb848f8df11d06aae8d5ad22f889fca585bca1aa.jpg)

![](images/ff110a3f2e8ae428da4f84a1f1757fa3674d25e8355189a19df75d7785d827ff.jpg)  
Figure 5: Per-class ROC curves of the fine-tuned MI2 classifier (MI2 FT) and a linear probe of QRad's frozen encoder (QRad Linear Prob) across the 13 disease classes. The two perform comparably across most classes, with QRad Linear Prob stronger on hard classes such as Pneumothorax and MI2 FT stronger on visually obvious classes such as Support Devices.

![](images/5a325711e1d53f51370e88baba35c898c56b4e822ef301015b373b674ec9cb33.jpg)

![](images/64ea21b827f82b4177141e9e2689ccefe42d6e0b70d1b50d253e856d212cca80.jpg)

![](images/e2895d5d8e30e1c86ef4b3990a97b880d14cf8658f21c77b62e0926ee78f85b4.jpg)

![](images/1e0a8abdad70be59e9e06e7c18c94875e7d9338e58147d8d353b44013c1488b3.jpg)

![](images/e5d7f2c214225943ec71ab6104c9b10f076b32fbed13eef4e7becf8bf52e6985.jpg)

![](images/bc785b346f14d2c13b0c31d7ef26b1f6391ce6592a666581bcd60f5bba1b0f36.jpg)

![](images/5ea77c16f6fb6216c100fb234bcc129e3d3c96deec4a8ac0e0dc7f024229da78.jpg)

![](images/9eb6833f7ec8e518f7614dd12bb0eab88749e6ecbcbc02ae3120e4975f1a82ac.jpg)

![](images/233c30649b58b97636d0c5a4866bcd228729136ddc5881d02ab09d2bd17ce942.jpg)

![](images/025288106354e9e506136bf8c2f163b2bad7313f60676224b0d90615e5ebfec2.jpg)

![](images/7a0b651d29c4d05bb35a68d9e1010f6cebb9c750dbeb5bf1e1be3123b253c809.jpg)

![](images/38eeb273215e4e10d40ea5fc0634872015a5ea43707dc38c2ac74473f53891f7.jpg)

![](images/6190ad639845d56b92386c7a4c5d1e3a02915b8f322fe4748f00158e9ccddf77.jpg)  
Figure 6: Threshold-control evaluation on all 13 disease classes using the QRad Linear Prob classifier as the perception model. As with the default MI2 FT classifier, the report ROC (red) closely tracks the classifier ROC (blue), and the non-controlled report (green cross) falls below the curves. Ground-truth labels are obtained from the reference reports via VisualCheXbert.

System Instruction:   
You are given an input chest x-ray radiology report and the classification   
result of 14 CheXbert classes, which includes both positive and negative   
diseases.   
Modify the report so that it matches each class in the CheXbert classification   
results. Add text segments or sentences to the report if the class is not   
mentioned in the report, or change/remove sentences in the report so that it   
agrees with the classification.   
Your modified report should be a valid Findings section of Chest X-ray   
radiology report, with information like the location, severity of medical   
findings based on the original report.   
When overwriting the report with the positive and negative classes, consider   
using text segments from the following templates, and fit them into the report   
as appropriate.   
Template Dictionary (abbreviated; full version includes all 14 classes):   
{   
"Cardiomegaly": {   
"positive": ["The heart is moderately enlarged.",   
"Mild enlargement of the cardiac silhouette.", ...],   
"negative": ["The cardiac silhouette is normal.",   
"The heart is normal in size.", ...]   
},   
"Pneumothorax": {   
"positive": ["small apical pneumothorax", ...],   
"negative": ["no pneumothorax", ...]   
}, ….   
}   
Rewriting Rules:   
Example 1: If "Fracture" is a positive class:   
- If the original report indicates Fracture is positive, do nothing.   
- If the original report indicates Fracture is negative, replace that   
sentence with a positive text segment.   
- If the original report doesn't mention Fracture, choose a positive text   
segment and add it to the report.   
Example 2: If "Support Devices" is a negative class:   
- If the original report doesn't indicate the existence of any support   
device, do nothing.   
- If the original report mentions any kind of support device exists, remove   
it.   
The input and output report are the Findings section of a radiology report.   
Make sure all positive classes are identifiable in the modified report.   
Return the modified version as JSoN with "output" as the key.   
Only return the JSoN without any explanation. Double check and make sure you   
didn't forget any classes in the lists, and didn't make unnecessary changes:   
{INPUT}  
Figure 7: The instruction prompt for LLM-based report rewriting. The LLM receives the original report, threshold-adjusted classification results, and a template dictionary of example text segments for 14 CheXbert disease classes. The full template dictionary contains positive and negative text segments for all classes: Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, and Support Devices.