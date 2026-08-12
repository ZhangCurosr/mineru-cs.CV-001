# Rethinking LLM Verification: Evidence Structure, Uncertainty, and Selective Refinement

ma Ranjan<sup>1</sup>, Kunal Tilaganji<sup>2,\*</sup>, Aditya Koul <sup>1,\*</sup>, Anurag Mahipal<sup>1,\*</sup>, Dashpreet Singh<sup>1,\*</sup>, Hriday Rana<sup>1,\*</sup>, Manan Jain Sidharth Gupta<sup>1,\*</sup>, Ajo Babu George <sup>3</sup>, Vineeth Balasubramanian <sup>2</sup>,

Nagarajan Natarajan<sup>2</sup>, Amit Sharma<sup>2</sup>

<sup>1</sup>Indian Institute of Technology Jammu, <sup>2</sup>Microsoft Research,

<sup>3</sup>SCB Dental College and Hospital, Cuttack,

Equal contribution

## Abstract

Large language models (LLMs) often rely on shortcuts rather than systematic reasoning, raising safety concerns in medical applications. Allowing models to abstain when uncertain improves reliability but introduces a coverage–accuracy tradeoff. We propose a two-stage framework for medical hypothesis verification in multiple-choice settings that manages this tradeoff through targeted ontology grounding, applied only when the model abstains. We show that abstention is not random but reflects genuine uncertainty, with abstained predictions associated with lower confidence. Across two frontier models (GPT-5.5, accessed via the Azure OpenAI API, and DeepSeek-R1), the proposed framework improves question-level accuracy by 9.6 percentage points (82.9% to 92.5%) and hypothesis-level accuracy by 4.2 percentage points (92.0% to 96.2%). Our experiments conducted on MedReason and MedQA show that abstention can be repurposed as a control signal for selective reasoning refinement, achieving knowledge-graph-level performance without explicit knowledge graph construction.

Keywords: Medical Reasoning, Uncertainty Estimation, Abstention, Knowledge Graphs, Ontology guidance

## 1 Introduction

Recent advances in large language models (LLMs) have significantly improved performance on complex medical reasoning tasks. However, despite strong benchmark performance, LLM-based systems remain unreliable in safety-critical settings where errors can have serious consequences. Systematic evaluations reveal persistent failure modes in clinical reasoning (Xie et al., 2025), including anchoring bias and a negative correlation between reasoning length and correctness (Moëll et al., 2025). A randomised clinical trial found no improvement in physician diagnostic reasoning when LLMs were used as decision support (Goh et al., 2024), highlighting a gap between benchmark performance and clinical utility.

A key challenge is that LLMs frequently rely on shortcuts rather than systematic reasoning (Arcuschin et al., 2025; Lewis-Lim et al., 2025), and when forced to commit to an answer, they may do so with unwarranted confidence. Abstention — allowing a model to declare uncertainty rather than commit to a potentially wrong answer — has been proposed as a safety mechanism (Wen et al., 2025). However, abstention introduces a coverageaccuracy tradeoff: a model that abstains too freely sacrifices utility, while one that abstains too rarely provides false assurance. A key open question is whether abstention is a meaningful signal of genuine uncertainty, or merely an arbitrary refusal (Guo and Yan, 2026).

While prior work has explored retrieval augmentation and knowledge-grounded medical reasoning, existing approaches typically rely on always-on retrieval or curated knowledge graphs. In contrast, we treat abstention itself as a meaningful uncertainty signal and use it to trigger targeted ontology grounding only when the model is uncertain. This allows the verifier to selectively incorporate structured medical knowledge through SNOMED CT without requiring a curated retrieval pipeline or external knowledge for every prediction.

In this work, we study medical hypothesis verification in a multiple-choice setting, where each option is independently evaluated as TRUE or FALSE, augmented with an abstention option (UNKNOWN). We propose a two-stage verification framework in which abstention-triggered UN-KNOWN predictions are selectively refined using SNOMED CT grounding, enabling uncertainty resolution while preserving the efficiency and simplicity of standalone verification.

Our contributions are as follows:

1. We show that abstention is not random — on two frontier models (GPT-5.5 and DeepSeek-R1), UNKNOWN predictions are associated with genuinely lower confidence, validating abstention as a reliable uncertainty signal.

2. We propose a two-stage verification framework that uses abstention to trigger targeted SNOMED CT grounding, improving question-level accuracy by 9.6 percentage points (82.9% to 92.5%, $p = 3 . 4 7 \times 1 0 ^ { - 2 1 } )$ and hypothesis-level accuracy by 4.2 percentage points (92.0% to 96.2%, $p \ = \ 5 . 3 9 \ \times$ 10<sup>−29</sup>).

3. We show that the framework matches knowledge-graph-grounded retrieval in accuracy without requiring a curated knowledge base, providing a practical path to reliable medical verification.

4. We further show that verification performance depends strongly on evidence formulation and grounding strategy, with ontology-guided refinement substantially improving uncertainty resolution over standalone verification.

In summary, We show that abstention can be repurposed as a control signal for selective reasoning refinement, achieving knowledge-graph-level performance without explicit knowledge graph construction.

## 2 Related Work

Prior work on improving LLM has focused on Knowledge-graph based grounding, Chain-of-Thought reasoning, and its variants like Tree-of-Thought and the abstention mechanism.

Structured knowledge grounding. Knowledgegraph-augmented LLMs have demonstrated consistent improvements in factual accuracy and reduction of hallucination in medical reasoning tasks. Wu et al. (2025) construct step-by-step reasoning chains by mapping clinical question-answer pairs to paths in a validated medical knowledge graph, producing traces that connect concepts via verified inferential relations. Yun et al. (2025) introduces stepwise, guideline-verified process rewards that enforce alignment between intermediate reasoning steps and established clinical guidelines, showing that outcome-level supervision is insufficient to ensure medically valid reasoning processes.

Su and Wu (2025) propose MedOnto-RAG, which integrates ontology-grounded retrieval into medical question answering pipelines. Despite their effectiveness, such approaches require curated, domain-specific knowledge bases that are expensive to construct and are not readily transferable across clinical settings or question types.

Chain-of-thought and contrastive reasoning. Wei et al. (2022) demonstrates that eliciting intermediate reasoning steps through chain-of-thought prompting, it substantially improves LLM performance on multi-step tasks. Chia et al. (2023) extend this by augmenting demonstrations with invalid reasoning examples, showing that contrastive exposure guides models away from systematic errors. Wang et al. (2023) shows that sampling multiple diverse reasoning paths and selecting the majority answer further improves robustness. Separately, contrastive evaluation, where a model selects among multiple candidate hypotheses rather than evaluating each independently, has been widely observed to yield higher accuracy than independent verification, with the performance gap attributed to the comparative pressure that multiple options exert on model reasoning (Long et al., 2026; Sharma and Jain, 2026). Long et al. (2026) explore debate-based reasoning for improving model reliability, though their approach differs from our focus on uncertainty-driven refinement.

Uncertainty and abstention. LLMs are frequently overconfident in medical settings, assigning high confidence to incorrect predictions (Kim et al., 2025). Abstention, withholding a prediction under uncertainty, has been proposed as a safety mechanism to reduce the clinical impact of overconfident errors (Wen et al., 2025).Guo and Yan (2026) demonstrate empirically that noise and ambiguity degrade LLM reliability in medical reasoning, and that the extent of abstention varies considerably across task formulations, knowledge settings, and model architectures.

These observations motivate a unified view of abstention and evidence grounding: rather than treating abstention as a terminal failure and evidence structure as a fixed property, we ask whether abstention can serve as a dynamic signal for targeted evidence augmentation. This framing connects the calibration concerns of the uncertainty literature with the structured grounding approaches of the knowledge graph literature and forms the basis of our two-stage verification framework.

## 3 Methodology

## 3.1 Task formulation

We investigate the conditions under which LLMs abstain and how this behaviour depends on evidence structure.

Formally, we model verification as a function:

$$
( y _ { i } , c _ { i } ) = \mathcal { V } ( Q , O _ { i } , E )
$$

where $Q$ is the question, $O _ { i }$ is a candidate hypothesis, and $E$ denotes the evidence used for reasoning. The classical hypothesis testing scenario is when the output is binary

$$
y _ { i } \in \{ \mathrm { T R U E } , \mathrm { F A L S E } \}
$$

and $c _ { i } \in [ 0 , 1 ]$ is a self-reported confidence measure in the output $y _ { i }$

We consider behaviour on curated and verified knowledge graph as the gold standard, and we compare how abstention differs under different types of structure E. Specifically, we analyze

• the trade-off between accuracy and coverage

• the relationship between confidence and correctness

• whether abstention is selectively triggered in difficult cases

To accommodate the possibility of an LLM to abstain, we extend the decision space to allow abstention by introducing an UNKNOWN option so that

$$
y _ { i } \in \{ \mathrm { T R U E } , \mathrm { F A L S E } , \mathrm { U N K N O W N } \}
$$

## 3.2 Datasets

MedReason Medreason (Wu et al., 2025) is a large-scale medical QA dataset of question-answer pairs, each accompanied by a knowledge-graphgrounded reasoning trace constructed by mapping expert-written clinical explanations to paths in a validated medical knowledge graph. We use 1000 questions extracted from MedReason in two kinds of evidence scenarios - using implicit knowledge of the LLM and using MedReason KG-grounded traces as the structured evidence condition.

MedQA (Jin et al., 2020) is a multiple-choice QA dataset drawn from USMLE licensing examinations. We use English USMLE subset as a second evaluation benchmark. MedQA does not contain reasoning traces.

Both MedReason and MedQA are publicly available research datasets.

SNOMED-CT has been used as an ontological grounding, and is accessed via BioPortal under its open API terms

MedReason and MedQA are publicly available research datasets, and SNOMED CT is accessed via BioPortal under its open API terms.

For MedReason, we sample 1000 questions randomly, yielding 3996 hypotheses across 4- option questions. For MedQA, we use 1000 randomly sampled questions from the official English USMLE test set, yielding 4000 hypotheses. No training data is used.

## 3.3 Models

Experiments were conducted using the Azure OpenAI API (GPT-5.5) and the DeepSeek API (DeepSeek-R1). Total API cost was approximately 20 USD. These models represent distinct training paradigms. GPT-5.5 is a closed RLHF-trained model, while DeepSeek-R1 is an open reinforcement-learning-based reasoning model with native chain-of-thought generation.

## 3.4 Abstention under Different Evidence Regimes

We evaluate hypothesis verification under two evidence conditions:

World knowledge (implicit). The model evaluates each option using only its parametric knowledge, with no external evidence provided. The verifier is instructed to Use medically valid reasoning and assign a label $y _ { i } \in$ {TRUE, FALSE, UNKNOWN} with confidence score $c _ { i }$

KG-grounded reasoning traces (structured). The model evaluates each option using an externally provided KG-grounded reasoning trace from MedReason as the sole evidence source. These traces are dataset annotations constructed from a validated medical knowledge graph and are therefore treated as fixed external evidence rather than model-generated reasoning. The verifier is explicitly prohibited from using external medical knowledge beyond the provided trace. (“Do NOT use external medical knowledge beyond what is provided. Treat the reasoning trace as your primary evidence.”), constraining it to reason only within the provided trace.

For both conditions, we report overall accuracy, conditional accuracy (over non-UNKNOWN predictions), coverage (proportion of non-UNKNOWN predictions), and mean confidence. We analyse how the evidence structure affects the coverageaccuracy tradeoff and whether abstention is selective or arbitrary under each regime. Experiments are conducted on MedReason and MedQA with both models.

## 3.5 Ontology-grounded 2-stage uncertainty refinement

We propose a two-stage verification pipeline that uses model abstention as a signal for targeted ontological grounding. The pipeline takes as input a question Q and its options O, and proceeds as follows (illustrated in Figure 1):

Step 1: Reasoning trace generation. Unlike the KG-grounded setting in Section 3.4, the reasoning trace here is generated by the model’s parametric knowledge; rather than retrieved from an external curated knowledge source or paths. The model is given the question and all options and asked to generate a detailed step-by-step medical reasoning trace without committing to a final answer. All options are visible at this step, so the model can reason over the relevant medical concepts.

Step 2: Independent hypothesis verification. Each option is evaluated independently against the generated reasoning trace, one option at a time, with no other options visible:

$$
( y _ { i } ^ { ( 1 ) } , c _ { i } ^ { ( 1 ) } ) = \mathcal { V } ( Q , O _ { i } , R )\tag{1}
$$

where $R$ is the generated reasoning trace, and $y _ { i }$ and $c _ { i }$ as defined in the case of abstention. The verifier is constrained to reason only from the provided trace. If no option is labelled UNKNOWN, the Stage 1 verdicts are returned as the final output.

Step 3: SNOMED CT retrieval. If any option receives UNKNOWN in Step 2, SNOMED CT is queried via the BioPortal API for each uncertain option, retrieving concept definitions and synonyms to construct an ontology-grounded context E<sub>SNOMED</sub>.

Step 4: Ontology-grounded re-evaluation. All options are re-evaluated individually using both the original reasoning trace and the SNOMED context:

$$
( y _ { i } ^ { ( 2 ) } , c _ { i } ^ { ( 2 ) } ) = \mathcal { V } ( Q , O _ { i } , R \cup E _ { \mathrm { S N O M E D } } ^ { ( i ) } )\tag{2}
$$

where $E _ { \mathrm { S N O M E D } } ^ { ( i ) }$ is non-empty only for options that were initially UNKNOWN. Non-uncertain options are re-evaluated with the reasoning trace alone. Each option remains evaluated in isolation throughout.

A key design property is that SNOMED grounding is applied locally triggered only by uncertain options but re-evaluation is global: all options are reassessed, allowing the retrieved ontological context to induce corrections across the full option set. This selective refinement introduces structured knowledge only where the model signals uncertainty, without requiring a pre-built knowledge oronh

![](images/5160313cb4b8cde8e0747fd9833fdc7372e9911469d035d0e8d14d1b79e12411.jpg)  
Figure 1: Ontology Grounded 2-stage Verification

## 3.6 Statistical testing.

To assess the significance of Stage 1 to Stage 2 improvements, we use McNemar’s test for paired binary outcomes (McNemar, 1947), with continuity correction, computed using statsmodels v0.14.6. We additionally report 95% confidence intervals for the difference in proportions using the standard paired-proportion method. All experiments are conducted as single deterministic evaluation runs with fixed prompts and decoding settings (default temperature = 0.1 on GPT-5.5). Reported confidence intervals quantify uncertainty over the finite evaluation set using paired-proportion statistics, rather than variance across multiple random seeds or repeated trials.

## 4 Results

We analyze how abstention, confidence, and accuracy interact under different evidence regimes.

## 4.1 Coverage-Accuracy Tradeoff under Abstention

Table 1 reports verification accuracy under binary (T/F) and abstention-enabled (T/F/U) evaluation for GPT-5.5 and DeepSeek-R1 across evidence conditions and datasets at k=1 independent hypothesis verification. Here, k = 1 denotes single-sample verification, i.e., each hypothesis is evaluated using one independently generated reasoning trajectory without self-consistency sampling or majority voting.

Under world knowledge, abstention is rare, and T/F/U accuracy closely matches the forced binary baseline (Table 1), indicating that models commit to an answer in almost all cases. In contrast, KGgrounded reasoning induces substantially higher abstention rates, accompanied by improved conditional accuracy. This shows that abstention becomes selective under structured evidence, reflecting genuine uncertainty rather than arbitrary refusal. Overall, structured evidence encourages cautious decision-making, while world-knowledge reasoning leads to near-universal commitment.

## 4.2 Confidence as an Uncertainty Signal

We analyze whether model-reported confidence provides a meaningful signal of uncertainty and can support abstention decisions. Specifically, we examine both how confidence behaves across predictions under different evidence conditions and whether it can be used to distinguish correct from incorrect outputs.

Figure 3 shows that confidence provides a meaningful separation between correct and incorrect predictions, particularly under KG-grounded reasoning. Structured evidence leads to sharper confidence distributions, with higher confidence on correct answers and reduced residual probability on errors.

![](images/fa8783a9f4056c2330842f04ce85f66df435c0869e25f0be109eff4320e0bd5f.jpg)

![](images/31832c40b802b63d1e116ca0003756a4dc6a78e29654b5e389a8028c17f3b903.jpg)  
Figure 2: Calibration and discriminative performance of confidence for MedReason (world knowledge, $k = 1 )$ Left: reliability diagram showing deviation from perfect calibration (ECE = 0.085). Right: ROC curve (AUC = 0.69) indicating that confidence retains meaningful discriminative ability in distinguishing correct from incorrect predictions. Despite imperfect calibration, confidence remains a useful signal for distinguishing correct from incorrect predictions.

This improvement in separation directly supports the use of confidence as a selective abstention signal, as uncertainty becomes more localized to genuinely difficult cases under structured evidence

In contrast, world-knowledge settings exhibit weaker separation. Incorrect predictions retain higher residual probability on the correct option and show increased entropy, indicating more diffuse and less decisive reasoning. As a result, confidence is less discriminative in distinguishing correct from incorrect predictions. These patterns suggest that structured evidence produces sharper and more informative confidence signals, while implicit reasoning yields noisier and less reliable confidence distributions.

This pattern is reflected quantitatively in Figure 2: although confidence values do not perfectly match empirical accuracy, higher-confidence predictions are consistently more likely to be correct. As a result, confidence remains a useful signal for identifying uncertain cases.

We observe similar trends for DeepSeek-R1 (Figure 4), with additional calibration results provided in Appendix A (Figures 5 - 7)

Taken together, these results show that although confidence is not numerically accurate, it remains a consistent and useful signal of uncertainty. This supports its use as a trigger for abstention and targeted refinement in the proposed pipeline.

## 4.3 Two-Stage Uncertainty Refinement

We evaluate whether model abstention can be exploited as a signal for targeted refinement, and whether selectively augmenting uncertain cases with ontological evidence improves verification per-

![](images/bec454b4b6088fe5099a2f26b18e4679764b6303eabcbe453c47f170dab9c2ba.jpg)  
Figure 3: Confidence distributions across evidence conditions for GPT-5.5 at k = 1. Each row corresponds to a different setting (MedReason KG-grounded, MedQA world knowledge, MedReason world knowledge). Left: distribution of P(chosen) for correct vs incorrect predictions. Middle: P(correct option) when the model is incorrect. Right: entropy of the predicted distribution. KG-grounded reasoning yields the clearest separation between correct and incorrect predictions, with higher confidence on correct answers, minimal residual probability assigned to the correct option when incorrect, and lower entropy. In contrast, world-knowledge settings exhibit weaker separation, higher residual probability on errors, and increased entropy, indicating more diffuse and less decisive reasoning. These patterns suggest that structured evidence produces sharper and more discriminative confidence signals.

## formance.

Table 2 reports accuracy, conditional accuracy, and coverage across three settings: implicit world knowledge (Imp), self-generated reasoning trace verification (Self, Stage 1), and ontology-grounded re-evaluation (Ont, Stage 2)

Stage 1: reasoning trace as structured internal evidence. Generating a reasoning trace before verification produces a substantial improvement over the implicit baseline across both models and datasets. On MedReason, GPT-5.5 improves from 87.8% to 92.0% (+4.2 pp), while DeepSeek-R1 improves from 84.3% to 91.1% (+6.7 pp). Similar gains are observed on MedQA. This improvement is accompanied by a slight reduction in coverage, indicating that the model abstains more selectively when reasoning is explicitly structured. These results suggest that reasoning traces act as a form of internal evidence structuring, improving both decision quality and uncertainty localization.

Table 1: Effect of abstention on verification accuracy. T/F = forced binary decision; T/F/U = abstention permitted. Cond. Acc = accuracy over non-UNKNOWN predictions; Coverage = proportion of non-UNKNOWN predictions.
<table><tr><td>Dataset</td><td>Model</td><td>T/F Acc</td><td>T/F/U Acc</td><td>Cond. Acc</td><td>Cov. %</td></tr><tr><td>MedR (World)</td><td>GPT-5.5 DS-R1</td><td>88.9 84.3</td><td>87.8 84.3</td><td>89.7 84.5</td><td>97.9 99.7</td></tr><tr><td>MedR (KG)</td><td>GPT-5.5 DS-R1</td><td>95.6 93.7</td><td>92.9 90.6</td><td>96.0 94.0</td><td>96.8 96.4</td></tr><tr><td>MedQA (World)</td><td>GPT-5.5 DS-R1</td><td>95.5 85.8</td><td>95.1 86.9</td><td>95.6 86.9</td><td>99.4 100.0</td></tr></table>

Stage 2: targeted ontological refinement Applying SNOMED CT grounding only to uncertain cases yields further gains across all settings. On MedReason, GPT-5.5 improves from 92.0% to 96.2% (+4.2 pp), while DeepSeek-R1 improves from 91.1% to 93.4% (+2.3 pp). Coverage increases to near-complete levels, indicating that most abstentions are successfully resolved. Gains are smaller on MedQA, reflecting both higher Stage 1 baselines and fewer abstentions, leaving less scope for refinement.

Comparison to KG-grounded reasoning The final Stage 2 performance matches or exceeds KG-grounded verification without requiring curated reasoning traces. On MedReason, GPT-5.5 achieves 96.2% accuracy compared to 92.9% for KG-trace verification, and DeepSeek-R1 achieves 93.4% compared to 90.6%. This demonstrates that selective, on-demand grounding can recover or surpass the benefits of externally curated knowledge structures, while avoiding their construction cost and rigidity.

## 4.4 Ablation and Analysis

Ablation: re-evaluation without ontological grounding. The key question is whether the Stage 2 accuracy gains arise from the incorporation of SNOMED CT ontological content, or simply from re-evaluating the same reasoning trace.

To isolate these effects, we perform an ablation in which Stage 2 is triggered for questions containing at least one UNKNOWN prediction, but no SNOMED context is provided. In this setting, the model re-evaluates all options using only the original reasoning trace, without access to external ontology information. Experiments are conducted on a subset of 100 MedReason questions (400 hypotheses) using GPT-5.5.

Table 2: Two-stage uncertainty refinement results. Imp = world knowledge baseline (T/F/U); Self = self-generated reasoning trace (Stage 1); Ont = ontology-grounded re-evaluation (Stage 2). Cond = conditional accuracy; Cov = coverage; ∆ = improvement over previous stage (pp); CI = 95% confidence interval; $p =  { \mathbf { M } }  { \mathrm { c N e m a r } } ^ { \cdot }  { \mathrm { s } }$ test with continuity correction. Ablation (Reeval) shows re-evaluation without SNOMED on 100 MedReason questions (GPT-5.5) — not significant (n.s.). All results are from single deterministic runs; confidence intervals reflect paired evaluation uncertainty rather than seed variance.
<table><tr><td>Data</td><td>Model</td><td>Type</td><td>Acc</td><td>Cond (SNOMED)</td><td>Cov</td><td>∆</td><td>CI (95%)</td><td>p</td></tr><tr><td rowspan="6">MedReason</td><td rowspan="4">GPT-5.5</td><td>Imp (baseline)</td><td>87.8</td><td>89.7</td><td>97.9</td><td></td><td></td><td></td></tr><tr><td>Self</td><td>92.0</td><td>94.2</td><td>97.6</td><td>+4.2</td><td>[+3.3, +5.1]</td><td> $< 1 0 ^ { - 1 0 }$ </td></tr><tr><td>Ont</td><td>96.2</td><td>96.6</td><td>99.5</td><td>+4.2</td><td>[+3.5, +4.9]</td><td> $< 1 0 ^ { - 1 0 }$ </td></tr><tr><td>KG-Trace</td><td>92.9</td><td>96.0</td><td>96.8</td><td></td><td></td><td></td></tr><tr><td rowspan="4">DeepSeek-R1</td><td>Imp</td><td>84.3</td><td>84.5</td><td>99.7</td><td></td><td></td><td></td></tr><tr><td>Self</td><td>91.1</td><td>91.4</td><td>99.6</td><td>+6.7</td><td>[+5.7, +7.7]</td><td> $< 1 0 ^ { - 1 0 }$ </td></tr><tr><td>Ont</td><td>93.4</td><td>93.4</td><td>100.0</td><td>+2.3</td><td>[+1.5, +3.1]</td><td> $4 . 9 2 \times 1 0 ^ { - 9 }$ </td></tr><tr><td>KG-Trace</td><td>90.6</td><td>94.0</td><td>96.4</td><td></td><td></td><td></td></tr><tr><td rowspan="6">MedQA</td><td rowspan="2">GPT-5.5</td><td>Imp</td><td>95.1</td><td>95.6</td><td>99.4</td><td></td><td></td><td></td></tr><tr><td>Self</td><td>97.4</td><td>98.8</td><td>98.6</td><td>+2.4</td><td>[+1.6, +3.1]</td><td> $2 . 0 5 \times 1 0 ^ { - 1 0 }$ </td></tr><tr><td rowspan="2"></td><td>Ont</td><td>98.4</td><td>98.5</td><td>99.9</td><td>+1.1</td><td>[+0.7, +1.4]</td><td> $1 . 2 0 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Imp</td><td>86.9</td><td>86.9</td><td>100.0</td><td></td><td></td><td></td></tr><tr><td rowspan="2">DeepSeek-R1</td><td>Self</td><td>94.2</td><td>94.6</td><td>99.6</td><td>+7.3</td><td>[+6.1, +8.5]</td><td> $\begin{array} { c } { { < 1 0 ^ { - 1 0 } } } \\ { { < 1 0 ^ { - 1 0 } } } \end{array}$ </td></tr><tr><td>Ont</td><td>96.0</td><td>96.0</td><td>100.0</td><td>+1.8</td><td>[+1.3, +2.2]</td><td></td></tr><tr><td rowspan="2" colspan="2">Ablation (GPT-5.5, MedR, n=100q)</td><td>Self</td><td>91.8</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Reeval</td><td>92.0</td><td></td><td></td><td></td><td>+0.2 [−0.2, +0.7]</td><td>1.00 (n.s.)</td></tr></table>

Re-evaluation without SNOMED yields a negligible improvement of +0.2 percentage points (91.8% to 92.0%), with only a single hypothesis changing from incorrect to correct. In contrast, the full ontology-grounded pipeline improves accuracy by +4.2 percentage points on the same task.

The substantial gap between these conditions indicates that repeated evaluation alone is insufficient to explain the observed gains, and that the addition of structured ontological grounding plays a central role in resolving uncertainty.

## 4.5 Analysis of SNOMED CT Contribution

The ablation in Section 4.4 shows that re-evaluation without ontological grounding yields negligible gain (+0.2 pp), while the full Stage 2 pipeline yields +4.2 pp, implicating SNOMED CT content as the primary driver. We verify this claim at the prediction level by categorising retrieval quality and examining resolution outcomes for all hypotheses that triggered Stage 2.

Retrieval quality. Retrieved SNOMED definitions are classified as plausible (semantically related to the query), irrelevant (demonstrably unrelated, e.g., a dental term returned for a radiological query), or failed (error or null). As shown in Table 3, retrieval is plausible in over 92% of Stage 2

cases for both models.
<table><tr><td>Model</td><td>n</td><td>Plaus.</td><td>Irrel.</td><td>Failed</td></tr><tr><td>GPT-5.5</td><td>95</td><td>89 (93.7%)</td><td>2 (2.1%)</td><td>4 (4.2%)</td></tr><tr><td>DeepSeek-R1</td><td>14</td><td>13 (92.9%)</td><td>1 (7.1%)</td><td>0 (0.0%)</td></tr></table>

Table 3: SNOMED CT retrieval quality across Stage 2 hypotheses (MedReason). n = hypotheses receiving UNKNOWN in Stage 1. Plaus. = Plausible; Irrel. = Irrelevant.

Resolution outcomes. Table 4 shows that over 90% of successful Stage 2 resolutions occur when retrieval is plausible, while fixes under irrelevant or failed retrieval account for under 10% in both models. This bounds the re-evaluation confound: cases fixed without useful SNOMED content represent a genuine but minor effect in which the structured re-evaluation context alone resolves uncertainty. Remaining Stage 2 errors fall into two categories: image-dependent questions where visual information is absent from both the trace and ontology, and management questions requiring treatment guideline knowledge not encoded in SNOMED CT. Representative examples of each resolution pattern are provided in Appendix C.

Evidence utilisation. Among cases fixed with plausible SNOMED retrieval, explicit reference to the retrieved content in the final reasoning differs markedly by model: 23.7% (14/59) for GPT-5.5 versus 80.0% (8/10) for DeepSeek-R1. This divergence reflects generation style rather than differential ontological influence: GPT-5.5 integrates external evidence implicitly, while DeepSeek-R1’s chain-of-thought generation produces more explicit attribution.

<table><tr><td>Model</td><td>SNOMED Quality</td><td>Fixed</td><td>Still Wrong</td></tr><tr><td>GPT-5.5</td><td>Plausible Irrel./Failed</td><td>59 (92.2%) 5 (7.8%)</td><td>30 (96.8%) 1 (3.2%)</td></tr><tr><td>DeepSeek-R1</td><td>Plausible Irrel./Failed</td><td>10 (90.9%) 1 (9.1%)</td><td>3 (100%) 0 (0.0%)</td></tr></table>

Table 4: Stage 2 resolution outcomes by SNOMED retrieval quality (MedReason). Fixed: resolved correctly in Stage 2 after UNKNOWN in Stage 1. Still wrong: remains incorrect after Stage 2. Irrel. = Irrelevant. Row percentages within each column.

## 5 Conclusion

We studied the coverage-accuracy tradeoff introduced by abstention in medical hypothesis verification under multiple-choice settings, and showed that abstention behaviour depends critically on the evidence structure. KG-grounded reasoning traces produce selective, well-calibrated abstention models that abstain more but with higher conditional accuracy, while world-knowledge models commit indiscriminately regardless of confidence. Building on this finding, we proposed a two-stage uncertainty refinement pipeline that treats abstention as a meaningful signal for targeted ontological grounding via SNOMED CT. The pipeline improves accuracy significantly across two models and three datasets, with each stage contributing independently. An ablation study confirms that the improvement is attributable to ontological grounding rather than re-evaluation alone. The final pipeline matches KG-grounded conditional accuracy on MedReason without requiring a curated knowledge base, providing a practical path to reliable medical verification in settings where structured knowledge resources are unavailable.

These results suggest a shift in how uncertainty is used in LLM systems. Rather than treating abstention as a failure mode, it can be leveraged as a control signal for selective refinement. By combining structured reasoning with targeted ontological grounding, the proposed framework achieves knowledge-graph-level performance without requiring curated knowledge bases. This points toward a broader paradigm in which uncertainty is not avoided, but actively used to guide efficient and reliable reasoning.

## Limitations

Performance depends on the quality of SNOMED CT retrieval, which may vary across domains; we refer to Section 4.5 for a detailed analysis of this effect.

In general, over-reliance on model abstention as a safety signal in clinical settings is a risk.

## References

Iván Arcuschin, Jett Janiak, Robert Krzyzanowski, Senthooran Rajamanoharan, Neel Nanda, and Arthur Conmy. 2025. Chain-of-thought reasoning in the wild is not always faithful. In Workshop on Reasoning and Planningfor Large Language Models.

Yew Ken Chia, Guizhen Chen, Luu Anh Tuan, Soujanya Poria, and Lidong Bing. 2023. Contrastive chain-of-thought prompting. arXiv preprint arXiv:2311.09277.

Ethan Goh, Robert Gallo, Jason Hom, Emily Strong, Yingjie Weng, Hannah Kerman, and 1 others. 2024. Large language model influence on diagnostic reasoning: a randomized clinical trial. JAMA Network Open, 7:e2440969.

Kevin H. Guo and Chao Yan. 2026. Clear: Revealing how noise and ambiguity degrade reliability in llms for medicine. Preprint, arXiv:2605.01011.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. 2020. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. arXiv preprint arXiv:2009.13081.

Yubin Kim and 1 others. 2025. Medical hallucinations in foundation models and their impact on healthcare. arXiv preprint arXiv:2503.05777.

Samuel Lewis-Lim, Xingwei Tan, Zhixue Zhao, and Nikolaos Aletras. 2025. Analysing chain of thought dynamics: Active guidance or unfaithful post-hoc rationalisation? In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 29838–29853, Suzhou, China. Association for Computational Linguistics.

Erping Long and 1 others. 2026. Model confrontation and collaboration: A debate intelligence framework for enhancing medical reasoning in large language models. Cell Reports Medicine, 7(1):102547.

Quinn McNemar. 1947. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2):153–157.

Birger Moëll, Fredrik Sand Aronsson, and Sanian Akbar. 2025. Medical reasoning in LLMs: an in-depth analysis of DeepSeek R1. Frontiers in Artificial Intelligence, 8:1616145.

Vinay Sharma and Manish Jain. 2026. Enhancing reasoning accuracy in large language models during inference time. arXiv preprint arXiv:2603.21301.

Ke Su and Shanshan Wu. 2025. Ontological reasoning mechanism for medical knowledge. In 2025 2nd International Conference on Smart Healthcare and Wearable Intelligent Devices (SHWID 2025), pages 155–160, Kuala Lumpur, Malaysia. ACM.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems.

Bingbing Wen and 1 others. 2025. Know your limits: A survey of abstention in large language models. Transactions of the Association for Computational Linguistics.

Juncheng Wu, Wenlong Deng, Xingxuan Li, Sheng Liu, Taomian Mi, Yifan Peng, Ziyang Xu, Yi Liu, Hyunjin Cho, Chang-In Choi, Yihan Cao, Hui Ren, Xiang Li, Xiaoxiao Li, and Yuyin Zhou. 2025. Medreason: Eliciting factual medical reasoning steps in llms via knowledge graphs. Preprint, arXiv:2504.00993.

Weidi Xie and 1 others. 2025. Quantifying the reasoning abilities of LLMs on clinical cases. Nature Communications, 16:9799.

Jaehoon Yun, Jiwoong Sohn, Jungwoo Park, Hyunjae Kim, Xiangru Tang, Yanjun Shao, Yonghoe Koo, Minhyeok Ko, Qingyu Chen, Mark Gerstein, Michael Moor, and Jaewoo Kang. 2025. Med-PRM: Medical reasoning models with stepwise, guideline-verified process rewards. arXiv preprint arXiv:2506.11474.

## A Additional Confidence and Calibration Results

This appendix provides additional results supporting the analysis in Section 4.2. All results include abstention (UNKNOWN), allowing us to examine how confidence relates to correctness and uncertainty.

![](images/5d0e094d424b41c5a88ac8bd5c81be468718494181c214ca72aa10e2fa1deb51.jpg)  
Figure 4: Confidence distributions across evidence conditions for Deepseek-R1 at k=1.

![](images/5c4a0eef778d87fdfe35ed70ae6101751d003c2116d7ef3c51f579071933392f.jpg)  
Figure 5: reliability diagram, and ROC curve for DeepSeek-R1 on MedReason (world knowledge, k=1).

## A.1 MedReason (World Knowledge, DeepSeek-R1)

We replicate the world-knowledge setting using DeepSeek-R1 to verify that the observed. Confidence behaviour is not model-specific.

Figure 5 shows that correct predictions are associated with higher confidence, while incorrect and UNKNOWN predictions occur at lower confidence levels. Confidence values do not always match the observed accuracy, but higher-confidence predictions are still more likely to be correct. The ROC curve further confirms that confidence remains a useful signal for distinguishing correct from incorrect predictions.

## A.2 MedReason (Reasoning Trace)

We next consider the reasoning-trace setting (Stage 1), where the model evaluates each option based on its own generated reasoning.

Figure 6 shows that confidence becomes more separated between correct and incorrect predictions compared to the implicit setting, consistent with the increase in conditional accuracy observed in Section 4.2. UNKNOWN predictions remain concentrated in low-confidence regions, indicating that abstention reflects uncertainty rather than random

![](images/27d9a72ba644d3e98298ccab5bb4a1e67c8fcab8789bccee95f630cfd5bb02ca.jpg)  
Figure 6: reliability diagram, and ROC curve for DeepSeek-R1 under reasoning-trace verification (MedReason, k=1).

![](images/3d176aaa97caf9ec66f888f65033aee148e3ed25c1a7c12eb782ceb1ef926d39.jpg)  
Figure 7: reliability diagram, and ROC curve for DeepSeek-R1 on MedQA (world knowledge, k=1).

behaviour.

## A.3 MedQA (World Knowledge)

We finally evaluate whether the same trends hold on MedQA.

Figure 7 shows that the relationship between confidence and correctness remains consistent: correct predictions tend to have higher confidence, while incorrect and UNKNOWN predictions occur at lower confidence levels. As in Section 4.2, confidence becomes increasingly concentrated at high values, leading to near-complete coverage and fewer abstentions.

## A.4 Summary

Across models, datasets, and evidence conditions, the same pattern is observed: confidence is not perfectly aligned with observed accuracy, but higherconfidence predictions are more likely to be correct. UNKNOWN predictions consistently occur in lowconfidence regions, supporting the interpretation of abstention as a meaningful signal of uncertainty.

## B Prompt Templates

## B.1 Overview

We employ a two-stage prompting framework corresponding to (i) reasoning trace generation and (ii) uncertainty-aware verification with optional ontology grounding. Prompts are instantiated with task-specific variables (question, options, and retrieved concepts). All outputs are constrained to valid JSON for deterministic parsing.

## B.2 Stage 1: Reasoning Trace Generation

The model first generates a structured reasoning trace over all candidate options without committing to a final answer.

## Instruction:

You are a medical reasoning system. Generate a detailed, step-by-step medical reasoning for the following question. Do NOT select a final answer.

Rules:

• Use medically valid reasoning.

• Consider all options.

• Do not reveal which option is correct.

• Be logically consistent.

Input:   
Question: {question}   
Options: {options}   
Output (JSON):   
{   
"reasoning": "Step-by-step reasoning   
covering all options"   
}

## B.3 Stage 1: Hypothesis Verification

Each option is evaluated independently using only the generated reasoning trace. External knowledge is disallowed, and the model may abstain via the UNKNOWN label.

Prompt:

You are a medical reasoning verifier.   
Evaluate whether the following option is cor  
rect using ONLY the provided reasoning trace.   
Rules: - Do NOT use external medical knowl  
edge. - Treat the reasoning trace as complete   
knowledge. - TRUE = supported by the reason  
ing. - FALSE = contradicted or not supported.   
- UNKNOWN = cannot be inferred confidently.   
- Be logically consistent. - Assign probabilities   
that sum to 1.   
Return ONLY valid JSON.   
Input: Reasoning Trace: {reasoning}   
Question: {question}   
Option: {option}   
Output: { "label": "TRUE/FALSE/UN-  
KNOWN", "reasoning": "...", "probabilities":   
{"TRUE": p1, "FALSE": p2, "UNKNOWN":   
p3} }

## B.4 Stage 2: Ontology-Grounded Re-evaluation

If any option is labeled UNKNOWN, SNOMED CT definitions are retrieved for those options and used to re-evaluate all options jointly.

## Prompt:

You are a medical reasoning system.   
Re-evaluate all options using: 1. The original   
reasoning trace 2. SNOMED CT definitions   
for uncertain options   
Rules: - Use medically valid reasoning. - Com  
pare all options before deciding. - TRUE = cor  
rect answer (exactly one). - FALSE = incorrect.   
- UNKNOWN = still uncertain. - SNOMED   
context may help re-assess all options. - Be   
logically consistent.   
Return ONLY valid JSON.   
Input: Question: {question}   
Options: {options}   
Reasoning Trace: {reasoning}   
SNOMED Context: {snomed\_definitions}   
Output: { "A": {"label": "...", "confidence":   
...}, "B": {"label": "...", "confidence": ...}, ... }

## B.5 Implementation Details

All experiments use deterministic decoding (default temperature = 0.1). Prompts are fixed across datasets and models. SNOMED CT concepts are retrieved via the BioPortal API and inserted only for options initially labeled UNKNOWN.

## C SNOMED CT Resolution Examples

We provide three representative examples illustrating the resolution patterns discussed in Section 4.5.

Example 1 — Explicit utilisation (GPT-5.5). Question: Not true about tubercular bacilli. Option: Gram positive. Stage 1 abstained because the reasoning trace noted that mycobacteria have structural similarities to Gram-positive bacteria but do not stain reliably with Gram stain, creating genuine ambiguity. SNOMED retrieved Gram-positive diplococcus — plausible but imprecise. Stage 2 explicitly engaged this: “The supplied SNOMED definition references Gram-positive classification, which combined with the reasoning trace’s note on structural similarity, allows a definitive determination that the option is misleading — mycobacteria are not reliably Gram-positive.” Correctly resolved as FALSE.

Example 2 — Implicit utilisation (GPT-5.5). Question: Rubrospinal tract influences. Option: Posture and balance. SNOMED retrieved Posture. The final reasoning correctly concluded that posture and balance are primarily mediated by vestibulospinal and reticulospinal pathways without explicitly citing SNOMED. Whether the retrieved concept anchored the reasoning or whether the model resolved this from the trace alone cannot be determined from the output text.

Example 3 — Retrieval failure, correct resolution (GPT-5.5). Question: Which view is best for viewing hollow viscus perforation. Option: Left lateral. SNOMED returned Lateral openbite – left — a dental term unrelated to radiological positioning. Stage 2 explicitly noted: “The SNOMED CT definition provided appears erroneous and is disregarded,” and resolved the hypothesis correctly from the reasoning trace alone. This exemplifies the re-evaluation effect: the structured re-evaluation context contributes to resolution even when the retrieved definition is inapplicable.