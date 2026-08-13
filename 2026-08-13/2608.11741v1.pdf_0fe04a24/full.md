# JieZi: A Large-Scale Expert-Audited Dataset and Benchmark for Ancient Chinese Character Exegesis

Ran Li<sup>∗</sup> South China University of Technology Guangzhou, China eeran0@mail.scut.edu.cn

Junle Liu South China University of Technology Guangzhou, China junle\_liu@foxmail.com

Huiguo He South China University of Technology Guangzhou, China hehuiguo@scut.edu.cn

Hiuyi Cheng South China University of Technology Guangzhou, China eechenghiuyi1@mail.scut.edu.cn

Jiahuan Cao South China University of Technology Guangzhou, China eejiahuancao@mail.scut.edu.cn

Lianwen Jin<sup>†</sup> South China University of Technology Guangzhou, China eelwjin@scut.edu.cn

## Abstract

The scholarly exegesis of ancient Chinese characters demands integrating visual observation, linguistic analysis, and historical context. However, existing computational approaches focus narrowly on subtasks such as character recognition and retrieval, lacking the structured datasets and benchmarks required for comprehensive scholarly analysis. To address this limitation, we introduce Ancient Chinese Character Exegesis (ACCE), a vision-language question answering (VQA) task that models the scholarly exegesis process. ACCE is organized into four progressive levels: basic character identification, glyph-form analysis, meaning exegesis, and diachronic evolution analysis. To support this task, we construct two complementary resources. JieZi-Dataset is the first large-scale, expert-audited VQA training dataset for ACCE, comprising over 500K QA pairs. It is constructed via a pipeline that reduces factual errors by constraining generation with expert-designed templates and source-text references. Human verification is further applied at each key stage to ensure scholarly accuracy. JieZi-Bench is an evaluation benchmark aligned with the exegesis process, constructed and verified by human experts to ensure evaluation reliability. It consists of four levels with reference answers curated from authoritative lexicographic works held separate from the training data. Experiments on multimodal large language models show that current models perform well on basic identification but struggle with glyph analysis, semantic reasoning, and diachronic understanding. Fine-tuning on JieZi-Dataset substantially improves performance across all four levels. Code and dataset are available at https://github.com/Ran00w/JieZi.

## CCS Concepts

• Computing methodologies → Artificial intelligence.

## Keywords

Ancient Chinese Character Exegesis, Vision-Language Benchmark, Paleographic Dataset

## 1 Introduction

Ancient Chinese characters represent one of the oldest continuously used writing systems, carrying irreplaceable historical and cultural heritage [15, 23, 28, 49]. As a core undertaking in ancient Chinese philology, the exegesis of individual ancient glyphs involves analyzing visual form, interpreting semantic function, and tracing diachronic evolution. This process demands years of specialized training, cross-referencing authoritative dictionaries, and reconciling divergent scholarly views [12, 44, 58]. It remains laborintensive, subjective, and dificult to scale when processing the massive volume of unearthed artifacts and manuscripts [7, 42, 60].

![](images/4b735ebd970436b190cb906af72082a43ef52c237acd00cb9720341c7e76d198.jpg)  
Figure 1: Comparison between the Exegesis task and the traditional Recognition task.

Artificial intelligence, particularly vision-language modeling, ofers a promising path to assist in this analysis [2, 3]. However, three fundamental challenges hinder progress toward comprehensive, scholar-grade exegesis. First, existing works mainly focus on narrow subtasks such as glyph recognition, image retrieval, and single-label classification [14, 30, 32]. These tasks address only fragments of the full exegetical workflow, failing to formalize the whole scholarly exegesis process. Second, current MLLMs lack domainspecific paleographic knowledge, causing frequent hallucinations when applied to ancient character analysis [5, 47, 54]. Mitigating this deficiency requires large-scale, expert-verified training data; however, the high cost of manual annotation [12, 45] and the unreliability of fully automated generation make neither approach alone practicable or scalable. Third, there is no benchmark that systematically evaluates the full scope of scholarly exegesis, leaving the capabilities of current MLLMs for comprehensive character analysis largely unclear.

To address these challenges, we take the first step toward computational modeling of the scholarly exegesis workflow for ancient Chinese characters. As illustrated in Fig. 1, we formulate Ancient Chinese Character Exegesis (ACCE), a novel vision language task that structures the scholarly exegesis process into four progressive levels: basic character identification, glyphform analysis, meaning exegesis, and diachronic evolution analysis, grounded in established principles of traditional philol ogy [29, 44, 59, 60]. To support ACCE, we construct two complementary resources.

First, we construct JieZi-Dataset, the first large-scale, expertaudited VQA training dataset for ACCE. Leveraging an authoritative etymological dictionary [21], we design an expert-in-the-loop pipeline that constrains LLM generation with expert-designed QA templates and dictionary source text, mitigating hallucination while reducing expert efort to template design and stage-wise verification. This pipeline provides over 500K reliable QA pairs and 130K glyph images across multiple script types.

Second, we construct JieZi-Bench, a scholar-grounded evaluation benchmark for ACCE. Its relatively small scale (approximately 8K QA pairs) enables complete construction and verification by human experts, thereby ensuring high evaluation reliability. The reference answers are curated from authoritative lexicographic sources and strictly separated from the training data to prevent leakage. The benchmark is organized into four progressive levels, each aligned with the core dimensions of ACCE. Experiments on diverse MLLMs show that current models perform well on basic identification but struggle with deeper exegetical tasks such as glyph decomposition, semantic reasoning, and diachronic analy sis. Fine-tuning on JieZi-Dataset yields substantial improvements across all four levels, confirming the value of domain-specific training data for this field. In summary, our contributions are as follows:

• We take the first step toward computational ancient character exegesis (ACCE) by formalizing it as a VQA task with four progressive levels of scholarly analysis.

• We construct JieZi-Dataset, a collection of 500k expert-audited VQA pairs. We also built the entirely expert-curated JieZi-Bench, providing a standardized evaluation framework for the ACCE task. They were generated by an expert-in-the-loop pipeline that mitigates LLM hallucination via source-grounded constrained generation.

• We benchmark representative MLLMs across four progressive levels with explicit reliability metrics, establishing strong baselines and identifying key challenges in deeper scholarly exegesis.

## 2 Related work

## 2.1 Ancient Chinese Character Datasets

Recent datasets have assembled large-scale glyph collections for ancient Chinese character recognition, covering tens of thousands of images across oracle bone inscriptions (OBI) [51, 53], historical handwritten scripts [4, 61], and modern printed characters [66].

These resources have substantially advanced OCR-oriented research. However, most of them focus on a single script type with annotations limited to character-class labels. Several eforts have begun to enrich annotation dimensions beyond class labels. EV-OBC [22] introduced a cross-era dataset spanning intermediate scripts such as Small Seal. ACCID [16] provided radical-level structural annotations for OBIs, and ACCP [52] provided structural and component labels covering characters from multiple eras. OBI Component 20 [24] provided components of OBIs with expert annotations. More recently, PD-OBS [39] and OracleSage [25] took a further step by bridging OBI images with natural-language semantics. Despite this progress, such eforts remain confined to oracle bone script, where a significant portion of glyphs remain undeciphered and scholarly consensus on interpretation varies considerably [32]. In contrast, later scripts such as Bronze, Small Seal, and Clerical are supported by substantially more established and reliable scholarly resources.

In summary, existing datasets either provide only sparse symbolic labels or are limited to a single script type with debatable annotations. No dataset ofers expert-audited, multi-dimensional natural-language annotations across multiple script types.

## 2.2 Ancient Chinese Character Evaluation

General vision-language benchmarks such as MMMU [62] and DocVQA [34] have been instrumental in advancing Multimodal Large Language Models (MLLMs) on document and scene understanding, but they contain no ancient character imagery and therefore cannot assess model capabilities in this domain. Within Chinese cultural heritage, C<sup>3</sup>-Bench [11] provides a comprehensive evaluation of classical Chinese cultural knowledge, and MCS-Bench [33] together with AC-EVAL [57] have advanced the assessment of classical text comprehension. However, these benchmarks operate at the passage or knowledge level and do not evaluate single-character visual understanding. In the ancient script domain, OBI-Bench [13] and Oracle-Bench [40] have contributed evaluation frameworks for oracle bone research, including tasks such as fragment matching and visual captioning. However, they focus specifically on oracle bone archaeological scenarios rather than the multi-dimensional exegesis of individual characters across script types. Overall, current benchmarks either lack ancient character imagery entirely or assess only recognition accuracy within a single script type, leaving no systematic evaluation for multi-dimensional character exegesis.

## 3 Task Definition

Overview. Ancient Chinese Character Exegesis (ACCE) is a visionlanguage question answering task. Given an ancient Chinese glyph image and a question � ∈ Q about the glyph, the model generates a natural language answer. The question space Q covers four analytical levels: basic information, glyph form, glyph meaning, and diachronic evolution. ACCE is grounded in established principles of Chinese paleography [29, 46, 59, 60] and structures the scholarly exegesis workflow into four progressive levels.

• L1: Basic Information. This level identifies the core attributes of a glyph through two fundamental tasks. Character Recognition (CHAR) maps the ancient glyph to its modern standard Chinese counterpart. Script Classification (SCRC) identifies the historical script type, such as Oracle Bone, Bronze, Seal, Clerical, or Regular.

Table 1: Comparison of dataset coverage across ACCE dimensions. MM denotes whether the dataset provides aligned multimodal evidence beyond plain image-level labels.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">L1</td><td colspan="5">L2</td><td rowspan="2">L3</td><td colspan="2">L4</td><td rowspan="2">MM</td></tr><tr><td>CHAR</td><td>SCRC</td><td>STRC</td><td>COMR</td><td>COMF</td><td>COMI</td><td>FORC</td><td>ORIM COME</td><td>EVOI</td></tr><tr><td>OBIMD [31]</td><td></td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>OracleSage [25]</td><td></td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>√</td></tr><tr><td>ACCP [52]</td><td></td><td></td><td></td><td></td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>PD-OBS [39]</td><td></td><td>V</td><td>X</td><td>V</td><td>X</td><td>X</td><td>X</td><td>V</td><td>X</td><td>X</td><td>√</td></tr><tr><td>JieZi-Dataset (Ours)</td><td></td><td></td><td>L</td><td>L</td><td></td><td>L</td><td>」</td><td>¬</td><td>V</td><td>L</td><td>L</td></tr></table>

• L2: Glyph Form. This level analyzes the internal structure of a glyph. A glyph is decomposed into visual units called components, each with specific structural relationships, functions, and meanings [6]. It encompasses five tasks. Structure Classification (STRC) describes the spatial layout of components. Component Recognition (COMR) identifies the individual components present. Component Function (COMF) explains the functional role ofeach component. Component Interpretation (COMI) explicates the semantic significance of each component. Formation Classification (FORC) categorizes the character according to the Six Writings taxonomy, including Pictophonetic Characters, Pictographs, and other formation principles.

• L3: Glyph Meaning. This level evaluates the semantic content of a glyph. Ancient Chinese characters carry meaning through abstraction, extension, and historical reinterpretation. It focuses on Original Meaning (ORIM), which explains the foundational semantic value of a character within its historical context.

• L4: Diachronic Evolution. This level evaluates how a character evolved across historical periods. Characters transform following systematic patterns in graphic simplification, stylistic regularization, and structural reorganization [44]. It examines the temporal dimension through two tasks. Component Evolution (COME) tracks and explains how specific components changed across script types. Evolution Interpretation (EVOI) provides a holistic analysis of why a character evolved over time, contextualizing these changes within broader patterns of writing system development.

Compared to traditional recognition tasks [26, 64, 66], ACCE better reflects the goals of paleographic analysis and provides a more realistic testbed for scholar-aligned ancient character understanding.

## 4 Datasets

To support the study ofAncient Chinese Character Exegesis (ACCE), we construct two complementary resources: (1) a large-scale training dataset JieZi-Dataset, and (2) a high-reliability evaluation benchmark JieZi-Bench. Both are built through a shared multistage pipeline (Fig. 2) but adopt distinct quality control strategies: JieZi-Dataset prioritizes scale through stage-wise spot-checking, while JieZi-Bench ensures evaluation reliability through exhaustive expert verification of every instance.

## 4.1 Data Sources

JieZi-Dataset draws from two sources. The primary source is a high-resolution scanned edition of Hanzi Yuanliu Dazidian [21], a modern etymological dictionary compiled and reviewed by domain experts. The scanned volume exceeds 2K pages and 5M tokens, covering more than 13K characters with rich descriptions ofglyph form, meaning, and diachronic evolution across multiple script stages. To increase glyph diversity, we further incorporate samples from public datasets, including ACCP [52] and MegaHan97K [66]. To ensure annotation consistency, we retain only images whose character identities and script categories align with the corresponding entries.

JieZi-Bench is sourced from four classical and modern lexicographic works: Kangxi Dictionary, Shuowen Jiezi, Shuowen Jiezi Zhu, and Revised Mandarin Chinese Dictionary. These dictionaries are selected because their explanations are mutually verifiable and cover characters absent from the training data, preventing data leakage. To ensure suficient authoritative evidence, we first apply an empirical token-length threshold of 200 to retain the top 20% most informative entries across all four dictionaries. We then refine the results through expert verification, yielding 1,024 glyph images paired with multi-source explanations.

## 4.2 Construction Pipeline

The construction of both resources follows a three-stage pipeline, as illustrated in Fig. 2. The key diferences lie in the data collection strategy and the granularity of human verification at each stage.

Stage 1: Collection. For the JieZi-Dataset, we digitize the scanned dictionary through OCR and glyph extraction. We use a prompt-based OCR pipeline built on Gemini-2.5-Pro [17] to extract textual descriptions from the dictionary. We separately train a YOLOv11 [27] detector on over 2K annotated samples to localize glyph images and classify their script types, covering multiple historical periods and variant forms. Since OCR achieves only 95% accuracy, and errors predominantly occur in rare and archaic glyphs that are most critical to ACCE, we manually correct all OCR outputs and extracted glyphs to ensure accuracy, requiring approximately 1,000 hours of human efort. After correction, rule-based normalization produces a coarse-grained alignment between glyph images and their corresponding explanatory text. For external datasets, we use Hanzi Yuanliu Dazidian as the reference anchor: ACCP images are retained only when their character and script labels match dictionary entries; MegaHan97K [66] images are selected for characters covered by the dictionary, restricted to historic document sources to maintain our focus on ancient scripts. For the JieZi-Bench, data collection centers on cross-dictionary compilation. We merge entries from the four lexicographic sources, apply the tokenlength threshold described above, and have experts manually verify every retained entry to confirm factual accuracy and cross-source consistency.

![](images/ef4518be94d9104bd4656291994c6cd4183995475dffc600cb1c3fafbc094d11.jpg)  
Figure 2: Data generation pipeline of JieZi-Dataset and JieZi-Bench.

Stage 2: Structurization. Both resources undergo LLM-based structured extraction to convert raw textual entries into standardized metadata records. We employ a prompt-based extraction pipeline with Gemini-3-Flash [18] to parse each entry into fields covering Basic Information, Glyph Form, Meaning, and Diachronic Evolution, as illustrated in Fig. 2. The extraction prompts for JieZi-Dataset and JieZi-Bench are provided in the supplementary materials. The verification granularity difers between the two resources. For the JieZi-Dataset, we audit 10% of the extracted records to ensure the model does not alter, omit, or fabricate content from the original text. For JieZi-Bench, every extracted record is manually checked and revised by experts to guarantee correctness.

Stage 3: VQA Generation. We design QA templates grounded in real research scenarios from Chinese paleography. For each glyph image, we randomly generate 5 to 10 question-answer pairs, ensur ing at least one question from each of the four ACCE levels (L1–L4) to maintain balanced coverage across all subtasks. Post-processing includes removing duplicate images and QA pairs within each task, reformatting lengthy answers into structured markdown for clarity, and verifying image-content alignment. For JieZi-Dataset, we randomly sample 5K instances for manual inspection. For JieZi-Bench, every QA pair is manually checked and revised to ensure evaluation reliability.

Expert-in-the-Loop Quality Assurance. Human verification is integrated throughout the pipeline rather than applied as a single final step. At each stage, expert involvement ensures that errors do not propagate downstream. The two resources adopt complementary verification strategies that reflect their distinct roles: JieZi-Dataset employs stage-wise spot-checking to balance scale with quality, while JieZi-Bench applies exhaustive verification at every stage to maximize evaluation reliability.

![](images/fe140a69227a34ab7ee4b037f3909d2c2ea503fb6f3796bad693dd191869f3fa.jpg)  
Figure 3: Distribution of glyph stages in JieZi-Dataset and JieZi-Bench.

## 4.3 Data Statistics

We report the key statistics of both resources. JieZi-Dataset comprises approximately 13K unique characters, 130K glyph images spanning six script stages (Oracle Bone, Bronze, Warring States, Seal, Clerical, and Regular), and over 500K expert-audited QA pairs covering all ten ACCE subtasks. JieZi-Bench, constructed independently from separate lexicographic sources, contains 1,024 glyph images and approximately 8K QA pairs, with every instance verified by human experts.

Task coverage. A notable advantage ofJieZi-Dataset is its complete coverage of all ACCE dimensions. As shown in Tab. 1, existing datasets address at most four of the ten subtasks, and only OracleSage [25] and PD-OBS [39] provide aligned multimodal evidence beyond image-level labels. In contrast, JieZi-Dataset is the first resource to support the full exegesis workflow, encompassing basic identification (L1), glyph-form analysis (L2), meaning exegesis (L3), and diachronic evolution (L4) within a unified multimodal framework.

![](images/96137e70d03aa24b42f31e08d2853a0f2eab625503f6aa3fd846fb9de46b3238.jpg)  
Figure 4: Distribution of token lengths across metadata en tries in the dataset.

Annotation richness. In addition to broad task coverage, JieZi Dataset provides substantially richer per-character annotations than prior resources. As shown in Fig. 4, the median per-entry metadata length is 781 tokens and the mean reaches 1,002 tokens, with the majority of entries falling in the 512 to 1,024 token range. In contrast, existing datasets typically rely on single categorical labels or short phrases. Each character in JieZi-Dataset, however, is accompanied by detailed, multi-dimensional textual descriptions, making it suitable for training generative models.

Character and script diversity. We further assess whether the dataset adequately represents the diversity of real-world ancient texts. Fig. 5 plots the character-frequency distribution of a representative classical Chinese corpus [43]. Although this corpus exhibits a pronounced long-tail pattern, JieZi-Dataset covers a substantial portion of both high-frequency and lower-frequency characters, thereby ensuring broad applicability to downstream tasks. Fig. 3 further presents the glyph-stage composition ofJieZi-Dataset and JieZi-Bench. Both resources span all six script stages rather than concentrating on a single type, with Seal and Bronze scripts constituting a significant proportion. Notably, 12.3% of glyph images are sourced from real historical documents with natural degradation such as erosion and stains, rather than clean dictionary renderings. In the construction of JieZi-Bench, we introduced some splits to better reflect the generalization ability of JieZi-Dataset. First, 25% of the images in JieZi-Bench belong to unseen (character, script) pairs that do not appear in JieZi-Dataset. Second, 12.5% of the characters in JieZi-Bench are entirely unseen during training. Third, 25.2% of the components in JieZi-Bench are unseen in JieZi-Dataset.

## 5 Experiment

## 5.1 Experimental Setup

To comprehensively evaluate the ACCE task, we benchmark existing SOTA MLLMs, including closed-source commercial models such as GPT-5.4 [38] and open-source models such as Qwen3.5-397ba17b [41]. In addition, we validate the efectiveness of our highquality training data on Qwen3.5-2B, Qwen3.5-4B and Qwen3.5-9B. The training is performed on 8 Ascend 910B NPUs. More training details are presented in the supplementary material.

## 5.2 Evaluation Metrics

We use diferent metrics for closed-form and open-form questions.

• Acc (Accuracy): For tasks with categorical outputs (CHAR, SCRC, STRC, FORC), we use accuracy.

• F1-Score: For component identification and functional description tasks (COMR, COMF, COME), we use character-level F1- score [50], which better reflects real-world exegesis scenarios, where each glyph is treated as a single instance and component predictions may be only partially correct.

![](images/a8b9d522abb9a0415e2854cf4d28fa4ca43d516e590d5257d6784f5b5feef0f3.jpg)  
Figure 5: Coverage ofJieZi-Dataset over character frequencies in a classical Chinese corpus.

• BERTScore: For open-ended natural-language generation tasks (ORIM, COMI), we use BERTScore [65] to measure semantic similarity between model outputs and references.

• LLM-as-a-Judge: For EVOI, an open-ended generation task, we adopt an LLM-as-a-Judge protocol. The judge evaluates responses along two dimensions: (1) Fact Alignment, measuring consistency with the reference answer; (2) Scholarly Expression, assessing the use of appropriate domain-specific terminology. Full prompts and the human validation study are provided in the supplementary material.

## 5.3 Main Results

We benchmark several general MLLMs against models fine-tuned on JieZi-Dataset to evaluate the necessity of domain-specific data. The ablation study of our proposed pipeline and dataset is present in the supplementary material. Tab. 2 reveals the following insights:

General MLLMs show varying performance across all ACCE levels. Most models perform moderately on categorical tasks (e.g., SCRC 40–77%) but struggle with fine-grained analysis (e.g., CHAR 10–37%). Performance deteriorates further on L4, where even the best non-fine-tuned model scores below 40% on EVOI. This indicates that general pretraining fails to encode the structured paleographic knowledge required for exegesis.

Models trained on more Chinese data perform notably better. Among non-fine-tuned models, Doubao-Seed-2.0-pro and Kimi-K2.5 rank as the top two across nearly all subtasks. Both models are developed by Chinese technology companies and are likely trained on richer Chinese and classical-text data. In contrast, GPT-5.4 lags far behind despite its strong general capabilities (CHAR: 10.4% vs. Doubao’s 36.6%). This suggests that domain-relevant data coverage, rather than model scale, is a primary bottleneck for ACCE.

Domain-specific fine-tuning consistently improves performance, and the improvement increases with model capacity. Fine-tuning on JieZi-Dataset improves performance across all subtasks. Even the lightweight 2B model outperforms Gemini-3.1-Pro and GPT-5.4 on structural parsing (e.g., COMR and FORC), and the 9B model achieves SOTA results across the board. Performance further improves from 2B to 9B, with larger gains on deeper reasoning tasks (e.g., COME: +4.9 from 2B to 4B, +12.2 from 4B to 9B), suggesting that further scaling remains a promising direction.

Table 2: Main results on JieZi-Bench across four paleographic levels: L1 (Basic Info: CHAR, SCRC), L2 (Glyph Form: STRC, COMR, COMF, COMI, FORC), L3 (Meaning: ORIM), and L4 (Evolution: COME, EVOI-FAC/SCE). All metrics are scaled to 0-100.
<table><tr><td rowspan="3">Method</td><td colspan="2">L1</td><td colspan="5">L2</td><td>L3</td><td colspan="3">L4</td></tr><tr><td>CHAR↑</td><td>SCRC↑</td><td>STRC↑</td><td>COMR↑</td><td>COMF↑</td><td>COMI↑</td><td>FORC↑</td><td>ORIM↑</td><td>COME↑</td><td>EVOI FAC↑</td><td>SCE↑</td></tr><tr><td>Closed-source MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.4 Thinking [38]</td><td>10.4</td><td>39.2</td><td>50.4</td><td>13.8</td><td>11.9</td><td>10.3</td><td>33.1</td><td>59.9</td><td>12.4</td><td>10.2</td><td>16.8</td></tr><tr><td>Gemini 3.1 Pro [20]</td><td>29.9</td><td>76.6</td><td>65.7</td><td>31.9</td><td>29.9</td><td>24.8</td><td>56.1</td><td>62.8</td><td>30.3</td><td>22.3</td><td>27.1</td></tr><tr><td>Claude opus 4.6 [1]</td><td>22.9</td><td>65.8</td><td>63.5</td><td>28.0</td><td>26.3</td><td>22.3</td><td>50.6</td><td>59.3</td><td>26.7</td><td>19.8</td><td>26.3</td></tr><tr><td>Doubao-Seed-2.0-pro [8]</td><td>36.6</td><td>68.5</td><td>69.8</td><td>37.8</td><td>34.4</td><td>29.9</td><td>59.9</td><td>63.2</td><td>36.1</td><td>31.7</td><td>38.7</td></tr><tr><td>Open-source MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Kimi-K2.5 [47]</td><td>31.5</td><td>70.1</td><td>68.8</td><td>39.2</td><td>36.6</td><td>31.5</td><td>58.1</td><td>66.6</td><td>36.4</td><td>28.3</td><td>34.6</td></tr><tr><td>GLM-4.6V [48]</td><td>18.2</td><td>49.7</td><td>48.9</td><td>21.8</td><td>20.4</td><td>18.2</td><td>42.2</td><td>48.2</td><td>20.9</td><td>15.5</td><td>20.2</td></tr><tr><td>Qwen3.5-397b-a17b [41]</td><td>26.2</td><td>67.1</td><td>69.4</td><td>32.1</td><td>29.9</td><td>26.5</td><td>54.4</td><td>59.5</td><td>28.3</td><td>22.2</td><td>30.7</td></tr><tr><td>Qwen3.5-2B [41]</td><td>18.9</td><td>22.6</td><td>31.6</td><td>17.3</td><td>13.7</td><td>17.0</td><td>30.7</td><td>62.3</td><td>12.9</td><td>7.9</td><td>20.2</td></tr><tr><td>Qwen3.5-2B + JieZi-Dataset</td><td>41.8+22.9</td><td>61.7+39.1</td><td>72.7+41.1</td><td>42.6+25.3</td><td>32.7+19.0</td><td>41.3+24.3</td><td>63.3+32.6</td><td>65.7+3.4</td><td>28.3+15.4</td><td>27.9+20.0</td><td>36.7+16.5</td></tr><tr><td>Qwen3.5-4B [41]</td><td>3.6</td><td>6.6</td><td>1.0</td><td>4.5</td><td>3.9</td><td>4.3</td><td>5.9</td><td>6.5</td><td>3.6</td><td>2.0</td><td>3.6</td></tr><tr><td>Qwen3.5-4B + JieZi-Dataset</td><td>49.0+45.4</td><td>67.6+61.0</td><td>74.1+73.1</td><td>47.2+42.7</td><td>37.9+34.0</td><td>42.4+38.1</td><td>64.8+58.9</td><td>65.9+59.4</td><td>33.2+29.6</td><td>28.5+26.5</td><td>37.2+33.6</td></tr><tr><td>Qwen3.5-9B [41]</td><td>22.3</td><td>51.0</td><td>40.2</td><td>27.4</td><td>24.6</td><td>26.1</td><td>49.1</td><td>64.7</td><td>22.4</td><td>15.0</td><td>23.5</td></tr><tr><td>Qwen3.5-9B + JieZi-Dataset</td><td>52.6+30.3</td><td>80.1+29.1</td><td>74.7+34.5</td><td>48.1+20.7</td><td>38.4+13.8</td><td>45.7+19.6</td><td>66.5+17.4</td><td>66.7+2.0</td><td>45.4+23.0</td><td>32.5+17.5</td><td>40.6+17.1</td></tr></table>

## 5.4 Generalization Analysis

We analyze the robustness and generalization of the fine-tuned model on JieZi-Bench, which includes Unseen Characters (UC) and Unseen Glyphs (UG). Tab. 3 reports results for Qwen3.5-9B fine-tuned on JieZi-Dataset across Bronze, Seal, and Regular scripts.

Exegesis remains robust despite recognition failures. CHAR falls to near-zero on unseen subsets (e.g., Seal UC: 1.9%, Bronze UG: 5.7%), indicating that exact character identification does not generalize to novel glyphs. However, structural parsing metrics show no comparable collapse: from All to UG, COMF decreases by only 3.5 points on Bronze (14.6→11.1), 2.1 on Seal (51.9→49.8), and 7.8 on Regular (69.6→61.8). This decoupling confirms that training on JieZi-Dataset induces transferable paleographic knowledge: the model derives structural and semantic understanding from visual form rather than relying on overfitting to seen character identities.

Older scripts remain the most challenging. Across nearly all metrics, performance decreases from Regular to Seal to Bronze (e.g., CHAR: 82.2→51.5→18.3; COMR: 85.1→65.5→17.4). The greater visual variance and structural abstraction of earlier scripts pose a persistent challenge for future research.

## 6 Conclusion

In this work, we introduce Ancient Chinese Character Exegesis (ACCE), a vision-language task that structures paleographic analysis into four progressive levels: basic information, glyph form, meaning, and diachronic evolution. To support this task, we construct JieZi-Dataset, a large-scale expert-audited dataset with approximately 500K QA pairs derived from authoritative etymological sources, and JieZi-Bench, a scholar-grounded evaluation benchmark aligned with the same four-level structure. Experiments show that current MLLMs perform reasonably well on basic identification but struggle with deeper exegetical tasks such as glyph decomposition and diachronic reasoning. Fine-tuning on JieZi-Dataset yields substantial improvements across all four levels, confirming the critical role of domain-specific data. Our work contributes the first resource covering the complete exegesis workflow across multiple script types, establishing a standardized foundation and a reproducible baseline for computational paleography. Building on this foundation, this efort facilitates broader exploration at the intersection of artificial intelligence and ancient Chinese character studies, paving the way for more robust, interpretable, and domain-aligned analytical tools.

Table 3: Generalization results (%) across Bronze, Seal, and Regular scripts. UC: Unseen Characters. UG: Unseen Glyphs. All: Full split. Metrics are scaled to 0-100.
<table><tr><td>Metric</td><td colspan="3">Bronze</td><td colspan="3">Seal</td><td colspan="3">Regular</td></tr><tr><td></td><td>UC n=26</td><td>UG n=87</td><td>All n=252</td><td>UC n=53</td><td>UG n=56</td><td>All n=342</td><td>UC n=31</td><td>UG n=53</td><td>All n=219</td></tr><tr><td>CHAR</td><td>7.7</td><td>5.7</td><td>18.3</td><td>1.9</td><td>1.8</td><td>51.5</td><td>6.5</td><td>37.7</td><td>82.2</td></tr><tr><td>SCRC</td><td>56.2</td><td>54.8</td><td>51.7</td><td>90.6</td><td>87.5</td><td>88.3</td><td>93.5</td><td>94.3</td><td>98.6</td></tr><tr><td>FORC</td><td>48.1</td><td>42.9</td><td>48.5</td><td>82.5</td><td>81.7</td><td>78.4</td><td>87.9</td><td>90.1</td><td>87.7</td></tr><tr><td>STRC</td><td>57.7</td><td>54.0</td><td>58.7</td><td>86.8</td><td>85.7</td><td>81.6</td><td>93.5</td><td>92.5</td><td>93.2</td></tr><tr><td>COMR</td><td>20.1</td><td>12.8</td><td>17.4</td><td>64.0</td><td>63.2</td><td>65.5</td><td>60.8</td><td>71.2</td><td>85.1</td></tr><tr><td>COMF</td><td>16.3</td><td>11.1</td><td>14.6</td><td>50.8</td><td>49.8</td><td>51.9</td><td>57.5</td><td>61.8</td><td>69.6</td></tr><tr><td>COMI</td><td>20.1</td><td>12.8</td><td>16.9</td><td>64.0</td><td>63.2</td><td>64.7</td><td>49.5</td><td>56.1</td><td>76.7</td></tr><tr><td>ORIM</td><td>61.8</td><td>60.1</td><td>61.1</td><td>68.1</td><td>67.6</td><td>68.7</td><td>74.5</td><td>74.8</td><td>75.2</td></tr><tr><td>COME</td><td>15.5</td><td>9.5</td><td>12.8</td><td>46.6</td><td>46.0</td><td>48.3</td><td>45.6</td><td>53.3</td><td>63.8</td></tr><tr><td>FAC</td><td>10.6</td><td>6.9</td><td>12.7</td><td>33.0</td><td>32.6</td><td>41.2</td><td>42.7</td><td>46.2</td><td>54.2</td></tr><tr><td>SCE</td><td>19.2</td><td>14.7</td><td>22.6</td><td>47.2</td><td>46.4</td><td>49.5</td><td>64.5</td><td>61.3</td><td>67.4</td></tr></table>

## References

[1] Anthropic. 2026. Claude Opus 4.6 System Card. https://www-cdn.anthropic. com/14e4fb01875d2a69f646fa5e574dea2b1c0f7b5.pdf

[2] Yannis Assael, Thea Sommerschield, Alison Cooley, Brendan Shillingford, John Pavlopoulos, Priyanka Suresh, Bailey Herms, Justin Grayston, Benjamin May nard, Nicholas Dietrich, et al. 2025. Contextualizing ancient texts with generative neural networks. Nature 645, 8079 (2025), 141–147.

[3] Yannis Assael, Thea Sommerschield, Brendan Shillingford, Mahyar Bordbar, John Pavlopoulos, Marita Chatzipanagiotou, Ion Androutsopoulos, Jonathan Prag, and Nando De Freitas. 2022. Restoring and attributing ancient texts using deep neural networks. Nature 603, 7900 (2022), 280–283.

[4] Nija Babu and A Soumya. 2019. Character recognition in historical handwritten documents–a survey. In 2019 international conference on communication and signal processing (ICCSP). IEEE, 0299–0304.

[5] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. 2025. Qwen3-VL Technical Report. arXiv:2511.21631 [cs.CV] https://arxiv.org/abs/2511.21631

[6] Françoise Bottéro. 1996. Review of The Origin and Early Development of the Chinese Writing System, by William G. Boltz. Journal ofthe American Oriental Society 116, 3 (1996), 574–577. https://doi.org/10.2307/605196

[7] Françoise Bottéro and Christoph Harbsmeier. 2008. The "Shuowen Jiezi" Dictionary and the Human Sciences in China. Asia Major (2008), 249–271.

[8] ByteDance Seed. 2026. Seed2.0 Model Card. https://seed.bytedance.com/seed2 [8]

[9] ByteDance Seed Team. 2026. Seed 2.0 Oficial Launch. Oficial blog post. https://seed.bytedance.com/en/blog/seed2-0-%E6%AD%A3%E5%BC%8F% E5%8F%91%E5%B8%83 Accessed: 2026-04-09

[10] Jiahuan Cao, Yang Liu, Peirong Zhang, Yongxin Shi, Kai Ding, and Lianwen Jin. 2025. TongGu-VL: Advancing Visual-Language Understanding in Chinese Classical Studies through Parameter Sensitivity-Guided Instruction Tuning. In Proceedings of the 33rd ACM International Conference on Multimedia. 11111– 11120.

[11] Jiahuan Cao, Yongxin Shi, Dezhi Peng, Yang Liu, and Lianwen Jin. 2024. C<sup>3</sup>Bench: A Comprehensive Classical Chinese Understanding Benchmark for Large Lan guage Models. arXiv:2405.17732 [cs.CL] https://arxiv.org/abs/2405.17732

[12] Diego Chapinal-Heras and Carlos Díaz-Sánchez. 2023. A review of AI applications in Human Sciences research. Digital Applications in Archaeology and Cultural Heritage 30 (2023), e00288.

[13] Zijian Chen, Tingzhu Chen, Wenjun Zhang, and Guangtao Zhai. 2024. OBI Bench: Can LMMs aid in study of ancient script on oracle bones? arXiv preprint arXiv:2412.01175 (2024).

[14] Yang Chi, Fausto Giunchiglia, Chuntao Li, and Hao Xu. 2024. Ancient Chinese Glyph Identification Powered by Radical Semantics. In Findings ofthe Association for Computational Linguistics: ACL 2024, Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand, 12065–12074. https://doi.org/10.18653/v1/2024.findings-acl.718

[15] Wiebke Denecke, Wai-yee Li, and Xiaofei Tian. 2017. The Oxford handbook of classical Chinese literature (1000 BCE-900 CE). Oxford University Press.

[16] Xiaolei Diao, Daqian Shi, Jian Li, Lida Shi, Mingzhe Yue, Ruihua Qi, Chuntao Li, and Hao Xu. 2023. Toward zero-shot character recognition: a gold standard dataset with radical-level annotations. In Proceedings ofthe 31stACMInternational Conference on Multimedia. 6869–6877.

[17] Google Gemini Team. 2025. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities. arXiv preprint arXiv:2507.06261 (2025). https://arxiv.org/abs/2507.06261

[18] Google DeepMind. 2025. Gemini 3 Flash Model Card. https://storage.googleapis. com/deepmind-media/Model-Cards/Gemini-3-Flash-Model-Card.pdf.

[19] Google DeepMind. 2026. Gemini 3.1 Flash-Lite Model Card. https://deepmind. google/models/model-cards/gemini-3-1-flash-lite

[20] Google DeepMind. 2026. Gemini 3.1 Pro Model Card. https://deepmind.google/ models/model-cards/gemini-3-1-pro

[21] Yankui Gu. 2023. Hanzi Yuanliu Dazidian [Dictionary of Chinese Character Etymology]. Language and Culture Press, Beijing, China.

[22] Haisu Guan, Jinpeng Wan, Yuliang Liu, Pengjie Wang, Kaile Zhang, Zhebin Kuang, Xinyu Wang, Xiang Bai, and Lianwen Jin. 2024. An open dataset for the evolution of oracle bone characters: EVOBC. arXiv preprint arXiv:2401.12467 (2024).

[23] Patrick Heinrich et al. 2020. Language modernization in the Chinese character cultural sphere: China, Japan, Korea and Vietnam. In The Cambridge handbook oflanguage standardization. Cambridge University Press, 576–596.

[24] Zhikai Hu, Yiu-ming Cheung, Yonggang Zhang, Peiying Zhang, and Pui-ling Tang. 2024. Component-level oracle bone inscription retrieval. In Proceedings of the 2024 International Conference on Multimedia Retrieval. 647–656.

[25] Hanqi Jiang, Yi Pan, Junhao Chen, Zhengliang Liu, Yifan Zhou, Peng Shu, Yiwei Li, Huaqin Zhao, Stephen Mihm, Lewis C Howe, et al. 2024. OracleSage: Towards unified visual-linguistic understanding of oracle bone scripts through cross modal knowledge fusion. arXiv preprint arXiv:2411.17837 (2024).

[26] Runhua Jiang, Yongge Liu, Boyuan Zhang, Xu Chen, Deng Li, and Yahong Han. 2023. OraclePoints: A Hybrid Neural Representation for Oracle Character. In Proceedings ofthe 31st ACM International Conference on Multimedia. 7901–7911. https://doi.org/10.1145/3581783.3612534

[27] Glenn Jocher and Jing Qiu. 2024. Ultralytics YOLO11. https://github.com/ ultralytics/ultralytics

[28] David N Keightley. 1996. Art, ancestors, and the origins of writing in China. Representations 56 (1996), 68–95.

[29] Guolong Lai. 2019. On [Can] and [Xie]: Two Diferent Approaches to the Interpretation of Ancient Chinese Characters, Form-Oriented and Integrated Phonology-Form-Semantics. Bulletin of the Jao Tsung-I Academy of Sinology 6, 1 (2019), 187–224.

[30] Bang Li, Donghao Luo, Yujie Liang, Jing Yang, Zengmao Ding, Xu Peng, Boyuan Jiang, Shengwei Han, Dan Sui, Peichao Qin, et al. 2024. Oracle bone inscriptions multi-modal dataset. arXiv preprint arXiv:2407.03900 (2024).

[31] Bang Li, Jing Yang, Yujie Liang, Xiaobin Hu, Zengmao Ding, Xu Peng, Shengwei Han, Peichao Qin, Donghao Luo, Taisong Jin, et al. 2026. OBIMD: A Multi-modal Dataset for Contextual Interpretation of Oracle Bone Inscriptions. Scientific Data (2026).

[32] Jing Li, Xueke Chi, Qiufeng Wang, Dahan Wang, Kaizhu Huang, Yongge Liu, and Cheng-Lin Liu. 2024. A comprehensive survey of oracle character recognition: challenges, benchmarks, and beyond. arXiv:2411.11354 [cs.CV] https://arxiv. org/abs/2411.11354

[33] Yang Liu, Jiahuan Cao, Hiuyi Cheng, Yongxin Shi, Kai Ding, and Lianwen Jin. 2025. MCS-Bench: A Comprehensive Benchmark for Evaluating Multimodal Large Language Models in Chinese Classical Studies. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 10435–10492.

[34] Minesh Mathew, Dimosthenis Karatzas, and C.V. Jawahar. 2021. DocVQA: A Dataset for VQA on Document Images. In Proceedings ofthe IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). 2200–2209.

[35] Ian R. McKenzie, Alexander Lyzhov, Michael Martin Pieler, Alicia Parrish, Aaron Mueller, Ameya Prabhu, Euan McLean, Xudong Shen, Joe Cavanagh, Andrew George Gritsevskiy, Derik Kaufman, Aaron T. Kirtland, Zhengping Zhou, Yuhui Zhang, Sicong Huang, Daniel Wurgaft, Max Weiss, Alexis Ross, Gabriel Recchia, Alisa Liu, Jiacheng Liu, Tom Tseng, Tomasz Korbak, Najoung Kim, Samuel R. Bowman, and Ethan Perez. 2023. Inverse Scaling: When Bigger Isn’t Better. Transactions on Machine Learning Research (2023).

[36] Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing.

[37] Catherine Olsson et al. 2022. In-context Learning and Induction Heads. Transformer Circuits.

[38] OpenAI. 2026. GPT-5.4 Thinking System Card. https://deploymentsafety.openai. com/gpt-5-4-thinking/gpt-5-4-thinking.pdf

[39] Kaixin Peng, Mengyang Zhao, Haiyang Yu, Teng Fu, and Bin Li. 2025. Interpretable Oracle Bone Script Decipherment through Radical and Pictographic Analysis with LVLMs. arXiv preprint arXiv:2508.10113 (2025).

[40] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Jiapeng Wang, Yifan Zhang, Zhuoma GongQue, Chong Sun, Yida Xu, Yadong Xue, et al. 2025. V-Oracle: Making progressive reasoning in deciphering oracle bones for you and me. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 20124–20150.

[41] Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents. https: //qwen.ai/blog?id=qwen3.5

[42] Zhongwei Shen. 2020. A phonological history ofChinese. Cambridge University Press.

[43] Yongxin Shi, Chongyu Liu, Dezhi Peng, Cheng Jian, Jiarong Huang, and Lianwen Jin. 2023. M5HisDoc: A large-scale multi-style Chinese historical document analysis benchmark. Advances in Neural Information Processing Systems 36 (2023), 78483–78495.

[44] Adam D Smith. 2017. Early Chinese manuscript writings for the name of the Sage Emperor Shun, and the legacy of Warring States-period orthographic variation in early Chinese received texts. Early China 40 (2017), 63–88.

[45] Thea Sommerschield, Yannis Assael, John Pavlopoulos, Vanessa Stefanak, Andrew Senior, Chris Dyer, John Bodel, Jonathan Prag, Ion Androutsopoulos, and Nando De Freitas. 2023. Machine learning for ancient languages: A survey. Computational Linguistics 49, 3 (2023), 703–747.

[46] Ken-ichi Takashima. 2021. Some methodological issues in reading oracle-bone inscriptions: In particular reference to the Huayuanzhuang Locus East Collection. Bulletin ofChinese Linguistics 14, 1 (2021), 1–41.

[47] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, S. H. Cai, Yuan Cao, Y. Charles, H. S. Che, Cheng Chen, et al. 2026. Kimi K2.5: Visual Agentic Intelligence. arXiv:2602.02276 [cs.CL] https://arxiv.org/abs/2602.02276

[48] V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, et al. 2025. GLM-4.5V and GLM-4.1V Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcemen

Learning. arXiv:2507.01006 [cs.CV] https://arxiv.org/abs/2507.01006

[49] UNESCO. [n. d.]. Chinese Oracle-Bone Inscriptions. https://www.unesco.org/ en/memory-world/chinese-oracle-bone-inscriptions.

[50] C. J. van Rijsbergen. 1979. Information Retrieval. Butterworths, London.

[51] Mei Wang and Weihong Deng. 2022. Oracle-MNIST: a realistic image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:2205.09442 (2022).

[52] Pengjie Wang, Kaile Zhang, Xinyu Wang, Shengwei Han, Yongge Liu, Lianwen Jin, Xiang Bai, and Yuliang Liu. 2024. Puzzle Pieces Picker: Deciphering Ancient Chinese Characters with Radical Reconstruction. In Document Analysis and Recognition – ICDAR 2024 (Lecture Notes in Computer Science, Vol. 14804). Springer, 169–187. https://doi.org/10.1007/978-3-031-70533-5\_11

[53] Pengjie Wang, Kaile Zhang, Xinyu Wang, Shengwei Han, Yongge Liu, Jinpeng Wan, Haisu Guan, Zhebin Kuang, Lianwen Jin, Xiang Bai, et al. 2024. An open dataset for oracle bone character recognition and decipherment. Scientific Data 11, 1 (2024), 976.

[54] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, et al. 2025. InternVL3.5: Advanc ing Open-Source Multimodal Models in Versatility, Reasoning, and Eficiency. arXiv:2508.18265 [cs.CV] https://arxiv.org/abs/2508.18265

[55] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. 2025. InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efi ciency. arXiv preprint arXiv:2508.18265 (2025).

[56] Jason Wei, Najoung Kim, Yi Tay, and Quoc V Le. 2023. Inverse Scaling Can Become U-Shaped. In EMNLP.

[57] Yuting Wei, Yuanxing Xu, Xinru Wei, Simin Yang, Yangfu Zhu, Yuqing Li, Di Liu, and Bin Wu. 2024. AC-EVAL: Evaluating Ancient Chinese Language Understand ing in Large Language Models. In Findings ofthe Association for Computational Linguistics: EMNLP 2024, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Association for Computational Linguistics, Miami, Florida, USA, 1600– 1617. https://doi.org/10.18653/v1/2024.findings-emnlp.87

[58] Crispin Williams. 2014. Scribal variation and the meaning of the houma and wenxian covenant texts’imprecation ma yi fei shi. Early China 37 (2014), 101–179.

[59] Qiu Xigui. 1985. On the Methods of Studying Ancient Chinese Script. Early China 11 (1985), 301–316.

[60] Wen Xing. 2011. Paleographic, Historical, and Intellectual History Approaches to Warring States Manuscripts Written on Bamboo Slips: A Review Article. Early China 33 (2011), 233–262. https://doi.org/10.1017/S0362502800000298

[61] Tai-Ling Yuan, Zhe Zhu, Kun Xu, Cheng-Jun Li, Tai-Jiang Mu, and Shi-Min Hu. 2019. A large Chinese text dataset in the wild. Journal ofComputer Science and Technology 34, 3 (2019), 509–521.

[62] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. 2024. MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 9556–9567.

[63] Biao Zhang, Zhongtao Liu, Colin Cherry, and Orhan Firat. 2024. When Scaling Meets LLM Finetuning: The Efect of Data, Model and Finetuning Method. In ICLR.

[64] Chongsheng Zhang, Ruixing Zong, Shuang Cao, Yi Men, and Bofeng Mo. 2020. AI-Powered Oracle Bone Inscriptions Recognition and Fragments Rejoining. In Proceedings ofthe Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI-20). 5309–5311. https://doi.org/10.24963/ijcai.2020/779

[65] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. 2020. BERTScore: Evaluating Text Generation with BERT. In International Conference on Learning Representations.

[66] Yuyi Zhang, Yongxin Shi, Peirong Zhang, Yixin Zhao, Zhenhua Yang, and Lianwen Jin. 2025. MegaHan97K: A large-scale dataset for mega-category Chinese character recognition with over 97K categories. Pattern Recognition 167 (2025), 111757.

## Supplementary Material

This supplementary material provides additional details that could not be included in the main paper due to space constraints. First, Sec. A provides scholarly background on the ACCE task formulation and illustrative examples. Second, Sec. B presents further details on dataset construction, including data distribution, prompt templates, and quantitative verification statistics. Finally, Sec. C provides additional experimental details and the complete results on JieZi-Bench.

## A Task Formulation Details

## A.1 Scholarly Background of ACCE

Ancient character exegesis. In Chinese paleography, exegesis refers to the scholarly practice of interpreting an ancient glyph by jointly analyzing its visual form, internal structure, semantic content, and historical evolution [29, 59, 60]. As one of the old est continuously used writing systems, ancient Chinese characters carry irreplaceable value for research in archaeology, history, and historical linguistics [15, 28], making their systematic interpretation a long-standing scholarly priority. Unlike modern character recognition, which maps a glyph to a single Unicode label, exegesis requires the scholar to explain why a character has its particular form, meaning, and evolutionary trajectory. In practice, scholarly exegesis spans four complementary analytical dimensions [6, 44], which are examined jointly and iteratively rather than in a fixed sequence:

• Identification. Identifying the character and determining which historical script period it belongs to.

• Structural Analysis. Analyzing the glyph’s internal structure by decomposing it into components and examining how each component contributes to the character’s form and meaning.

• Semantic Interpretation. Interpreting the character’s original semantic value in its historical context, informed by the structural analysis.

• Diachronic Tracing. Tracing how both form and meaning evolved across successive script periods, and explaining the motivations behind these diachronic changes.

This multi-faceted, knowledge-intensive analysis constitutes the standard methodology taught in Chinese philology programs and practiced in archaeological and linguistic research.

Challenges in exegesis. Exegesis is particularly challenging because it demands the simultaneous integration of heterogeneous knowledge types. These include visual pattern recognition for identifying glyph components, linguistic knowledge for understand ing component functions, semantic reasoning for inferring origi nal meaning, and historical knowledge for explaining diachronic change. Each requires diferent cognitive skills and a distinct body ofexpertise. Moreover, ancient glyphs exhibit high visual variability. The same character may appear markedly diferent across script periods (e.g., OBI, Bronze, Seal, Clerical), and diferent characters may share visually similar components, making identification and structural analysis error-prone even for trained scholars. The deeper analytical dimensions are also interrelated: a structural misjudgment (e.g., misidentifying a component) propagates into incorrect semantic and evolutionary interpretations. This interdependence among form, meaning, and evolution is why holistic exegesis cannot be reduced to a single classification step, although each dimension can still be evaluated independently. Training a human expert in character exegesis typically requires years of graduate-level study, and the analysis of a single character may involve cross-referencing multiple authoritative dictionaries and reconciling conflicting scholarly interpretations [12, 42].

![](images/9a1082d6e6a3a98791a2aa66cbbf46a728fddeb96dd7346404c4e883f48999b2.jpg)  
Figure 6: Examples of comprehensive scholarly exegesis in complex paleographic scenarios.

Formalization as a computational task. Despite its importance, exegesis has received limited attention from the computational community. Existing work on ancient Chinese characters focuses predominantly on recognition and retrieval [14, 32], which corresponds to only the most elementary step of the full exegetical workflow. To our knowledge, no prior work has attempted to formalize the complete, multi-stage scholarly process as a computational task. ACCE addresses this gap by decomposing exegesis into four analytical levels that mirror these dimensions: identifying a character and its script period (L1: Basic Information), analyzing internal structure and component functions (L2: Glyph Form), interpreting the original semantic value (L3: Glyph Meaning), and tracing diachronic evolution (L4: Diachronic Evolution). The levels are ordered from basic to deeper analysis for clarity, but they are not a strict pipeline: each is independently answerable from the glyph image, while the deeper levels (L2–L4) remain conceptually interrelated. It enables researchers to construct targeted supervision, apply stage-specific evaluation metrics, and identify precisely which aspects of exegesis current models can and cannot handle.

<table><tr><td rowspan=1 colspan=1>COMR</td></tr><tr><td rowspan=1 colspan=1>Question: What character is this glyph?                  Image :a</td></tr><tr><td rowspan=1 colspan=1>Answer: This is charactersī (司).                        L1: Basic</td></tr><tr><td rowspan=1 colspan=1>SCRC</td></tr><tr><td rowspan=1 colspan=1>Question: What is the script type of this glyph?                   金Image :(OBI, Bronze, Warring state, Seal, Clerical, Regular, Cursive)</td></tr><tr><td rowspan=1 colspan=1>Answer: This is a Bronze script.                          L1: Basic</td></tr><tr><td rowspan=1 colspan=1>STRC</td></tr><tr><td rowspan=1 colspan=1>Image验Question: What is the structure of this glyph?</td></tr><tr><td rowspan=1 colspan=1>Answer: Left-right structure.                         L2: Form</td></tr><tr><td rowspan=1 colspan=1>COMR</td></tr><tr><td rowspan=1 colspan=1>Image :验Question: What is the component on the left?</td></tr><tr><td rowspan=1 colspan=1>Answer:言(yán)                                      L2: Form</td></tr><tr><td rowspan=1 colspan=1>COMF</td></tr><tr><td rowspan=1 colspan=1>Question: What is the function of the component &#x27;贝&#x27;?              盟(Semantic, Phonetic, Symbol)Image :</td></tr><tr><td rowspan=1 colspan=1>Answer: Semantic                                     L2: Form</td></tr><tr><td rowspan=1 colspan=1>COMI</td></tr><tr><td rowspan=1 colspan=1>Question: What is the interpretation of the component &#x27;贝&#x27;?          男Image :</td></tr><tr><td rowspan=1 colspan=1>Answer: It resembles the form of a shell currency, signifying    L2: Formvaluables.</td></tr><tr><td rowspan=1 colspan=1>FORC</td></tr><tr><td rowspan=1 colspan=1>Question: Which of the Six Scripts does this glyph belongto?(Pictogram, Ideograms, Ideogrammic compounds, Semantic- Image :示phonetic, Derivative, Borrow)</td></tr><tr><td rowspan=1 colspan=1>Answer: Pictogram                                   L2: Form</td></tr><tr><td rowspan=1 colspan=1>ORIM</td></tr><tr><td rowspan=1 colspan=1>Question: What is the original meaning of this glyph?         Image :晔</td></tr><tr><td rowspan=1 colspan=1>Answer: It originally refers to the seasons of the year.       L3: Meaning</td></tr><tr><td rowspan=1 colspan=1>COME</td></tr><tr><td rowspan=1 colspan=1>Question: What is the type of evolution for the component &#x27;厂&#x27;in       Image :this glyph? (Retention, Corruption, Simplification, Elaboration)</td></tr><tr><td rowspan=1 colspan=1>Answer: Corruption                                  L4: Evolution</td></tr><tr><td rowspan=1 colspan=1>EVOI</td></tr><tr><td rowspan=1 colspan=1>Question: What is the evolution of the character in this glyph? Image :@</td></tr><tr><td rowspan=1 colspan=1>Answer: Historically, it evolved from an OBI depicting achild in the belly, to a Bronze Script showing a babygestating in a placenta, and finally to a Seal Script formed     L4: Evolutionby&#x27;勺&#x27; (wrapping) and &#x27;&#x27;(fetus). The modern standardcharacter presents a semi-enclosed structure, where theouter radical wraps around the inner component.</td></tr></table>

Figure 7: Examples of Question-Answering pairs across the ten fine-grained sub-tasks within the ACCE framework.

![](images/6802bbf92022235aa1a392bf186c48ccf89970ce7f7d9365e52fb5db6730c7d1.jpg)  
Figure 8: Examples of responses for diferent tasks from diferent models. Red text indicates error messages, and blue text indicates correct messages.

## A.2 Examples of Tasks

To provide a more intuitive understanding of the ACCE task, we present qualitative examples from theJieZi-Dataset. Fig. 7 illustrates the ten fine-grained subtasks. While these subtasks assess individual capabilities, the ultimate goal of ACCE is holistic interpretation. Fig. 6 presents comprehensive scholarly exegesis in real scenarios, where models are challenged with complex paleographic phenomena such as severe structural corruption and visually similar but semantically distinct glyph forms. As demonstrated, a high-quality exegesis successfully resolves these challenges by synthesizing all four progressive levels (L1 to L4) into a coherent, domain-aligned explanation. Fig. 8 contrasts the responses of the fine-tuned model with those of baseline models on representative subtasks.

## B Dataset Construction and Quality Control B.1 Data Distribution and Metrics

JieZi-Dataset. To illustrate the data diversity and comprehensiveness, Tab. 4 presents the subtask coverage distribution within the JieZi-Dataset. Unlike the individually targeted test queries in JieZi-Bench, the training data often features comprehensive QAs that simultaneously address multiple analytical dimensions, mirroring the holistic nature of real-world scholarly exegesis. Consequently, the sum of subtask frequencies exceeds the total number of unique QA pairs.

Table 4: Coverage of ACCE subtasks within the JieZi-Dataset. Since a single comprehensive QA pair may simultaneously address multiple subtasks, the cumulative count across all subtasks exceeds the total number of unique QA pairs (500K).
<table><tr><td>Level</td><td>Subtask</td><td>QAs Covering Subtask</td></tr><tr><td rowspan="2">L1: Basic</td><td>CHAR</td><td>175,875</td></tr><tr><td>SCRC</td><td>175,875</td></tr><tr><td rowspan="5">L2: Form</td><td>STRC</td><td>175,875</td></tr><tr><td>COMR</td><td>175,875</td></tr><tr><td>COMF</td><td>221,558</td></tr><tr><td>COMI</td><td>221,558</td></tr><tr><td>FORC</td><td>312,984</td></tr><tr><td>L3: Meaning</td><td>ORIM</td><td>175,875</td></tr><tr><td rowspan="2">L4: Evolution</td><td>COME</td><td>221,558</td></tr><tr><td>EVOI</td><td>210,360</td></tr><tr><td>Overall</td><td>Unique QA Pairs</td><td>505,867</td></tr></table>

Table 5: Distribution of Each Task in JieZi-Bench
<table><tr><td>Task</td><td>Metric</td><td>Question Count</td></tr><tr><td>CHAR</td><td>Accuracy</td><td>1024</td></tr><tr><td>SCRC</td><td>Accuracy</td><td>1024</td></tr><tr><td>STRC</td><td>Accuracy</td><td>1024</td></tr><tr><td>FORC</td><td>Accuracy</td><td>1024</td></tr><tr><td>ORIM</td><td>BERTScore</td><td>1024</td></tr><tr><td>EVOI</td><td>LLM-as-a-Judge</td><td>1024</td></tr><tr><td>COMR</td><td>F1-Score + Accuracy</td><td>1852</td></tr><tr><td>COMF</td><td>F1-Score + Accuracy</td><td>1852</td></tr><tr><td>COMI</td><td>F1-Score + BERTScore</td><td>1852</td></tr><tr><td>COME</td><td>F1-Score + Accuracy</td><td>1852</td></tr><tr><td>Overall</td><td>Unique QA Pairs</td><td>7996</td></tr></table>

JieZi-Bench. Tab. 5 summarizes the question distribution and corresponding evaluation metrics across the ACCE tasks in JieZi-Bench. Notably, the four component-related subtasks (COMR, COMF, COMI, and COME) share the same set of test questions (1,852 per subtask), because each question requires the model to analyze all four component dimensions simultaneously. Within this shared question set, each subtask is evaluated with a distinct metric.

## B.2 Prompt for Structurization

Raw dictionary entries are written in dense, unstructured prose that interleaves character identity, structural analysis, semantic explanation, and evolutionary commentary in a single paragraph. Stage 2 (Structurization) aims to distill this unstructured text into compact, structuredJSON records aligned with the four ACCE levels (L1–L4), retaining only the information relevant to exegesis while discarding editorial remarks, cross-references, and other content not directly useful for downstream VQA generation.

This structurization is non-trivial because ancient Chinese characters exhibit highly heterogeneous complexity. A simple pictograph may require only a brief structural note, whereas a phonosemantic compound spanning multiple script periods may involve dozens of components, variant forms, and evolutionary branches. The prompt must therefore accommodate this wide range of complexity within a single unified schema without losing critical details for complex entries or generating spurious fields for simple ones.

Fig. 9 presents the full prompt used in this stage. Its design addresses three key requirements: (1) Schema formatting. The prompt specifies a strict JSON schema with all required field names and value types, ensuring machine-parseable outputs without post-hoc reformatting. (2) Sourcefidelity. The model is explicitly instructed to extract only from the provided text and to leave fields empty rather than fabricate content, which is critical for mitigating hallucination on rare or ambiguous entries. (3) Adaptive granularity. The schema naturally accommodates entries of varying complexity: simple characters produce compact records, while complex multi-period entries expand as needed without requiring separate templates.

## B.3 Templates for VQA generation

Stage 3 (VQA Generation) transforms the structured metadata from Stage 2 into natural-language VQA pairs suitable for supervised fine-tuning. The generated QA pairs are designed to simulate the progressive reasoning process of domain experts, reflecting the layered analytical workflow of scholarly exegesis rather than isolated factual retrieval.

We design four prompt templates corresponding to the four ACCE levels (L1–L4) and adopt a dual-track generation strategy. For factual subtasks with well-defined answer patterns (e.g., character identity at L1, structure type at L2), QA pairs are produced by directly instantiating the template with the corresponding metadata fields, requiring no LLM reasoning. For reasoning-intensive subtasks that must synthesize multiple fields into coherent explanations (e.g., component-level interpretation at L2, semantic analysis at L3, diachronic tracing at L4), LLM generation is necessary. However, general-purpose LLMs lack the specialized paleographic knowledge required for ACCE, and unconstrained generation often produces plausible-sounding but factually incorrect answers.

Our templates address this by (1) injecting the verified structured metadata as dynamic context, constraining the LLM to reason from the provided evidence rather than its parametric knowledge, and (2) guiding the model through the expert analytical workflow (identification → decomposition → interpretation → evolution tracing). Each template also includes multiple question phrasings per subtask to promote syntactic diversity. Fig. 10 presents a representative template.

## B.4 Quantitative Verification Details

To make the quality-control process more transparent, we report the verification scope, sampling strategy, and revision statistics for each stage of our pipeline.

![](images/dac0d5081838ee53e66756e6ba8334b8382c91761cd303396223f3916f39f8c8.jpg)  
Figure 9: Prompt template used for structured metadata extraction from dictionary entries.

## <Template> VQA generation

Please output a structured JSON based on the image. Required fields: glyph type, character formation method, structure, special structure, components, original meaning, diachronic glyph evolution.

Please generate a standard answer that can be used directly for SFT from this single-character image.

Your output must satisfy all requirements below:

2. Use exactly these fields: glyph type, character formation method, structure, special structure, components, original meaning, diachronic glyph evolution.

3. The field "glyph type" must be consistent with the sample label "{glyph type}".

4. "structure" and "components" must follow the actual form in the current image stage.

5. "original meaning" and "diachronic glyph evolution" must be grounded in context, especially historical divergence, later forms, and simplification merges.

6. For simplified-form images, modern mainstream meaning is allowed for "original meaning", but the source chain must be explicit in "diachronic glyph evolution".

7. If weak hints conflict with the image, trust the image.

8. "original meaning" must align with the candidate list first; only minor paraphrase is allowed when strongly supported.

9. If "structure" is "single component", "special structure" must be a short description; otherwise it must be an empty string.

10. If the same component appears multiple times in different positions, split it into separate keys with position suffixes.

![](images/32442f080944447ed6140873f4fb1236a6698992ab721aaa49becea3ffbe3ea6.jpg)  
Figure 10: Prompt template for VQA generation. The prompt integrates to generate high-quality SFT data.

Stage 1: Collection. For JieZi-Dataset, Stage 1 mainly involves digitization, including OCR correction and alignment of the digitized content with the original dictionary pages. Since the goal at this stage is only to ensure fidelity to the source dictionary rather than paleographic interpretation, the full scanned volume (over 2,000 pages) was checked by general annotators without requiring domain expertise. The main error types were OCR omissions and misrecognition of variant character forms. While errors were found in less than 1% of entries, this full manual verification remains indispensable: as noted in the main paper, OCR errors disproportionately concentrate on rare and archaic glyphs, which are precisely the characters most critical to ACCE. Without exhaustive correction, even a small number of such errors would propagate into downstream structuring and QA generation, silently degrading the most informative entries in the dataset.

For JieZi-Bench, the source entries were also fully checked at the collection stage. Errors requiring correction were found in approximately 10% of entries, a notably higher rate than that ofJieZi-Dataset. This is expected for two reasons: (1) JieZi-Bench integrates entries from multiple authoritative dictionaries, and cross-source alignment and deduplication introduce additional inconsistencies absent from the single-source Dataset; (2) as a benchmark intended for evaluation, JieZi-Bench applies stricter acceptance criteria for entry completeness and accuracy, causing more entries to be flagged for revision. All identified errors were corrected before proceeding to the next stage.

Stage 2: Structurization. At Stage 2, raw dictionary entries were converted into structured metadata records by an LLM-based extraction pipeline. Given the scale ofJieZi-Dataset (over 500K entries), exhaustive manual verification was impractical at this stage. Instead, to verify dataset quality, we randomly sampled 10% of the extracted metadata records. Experts inspected each sampled record by comparing it directly with the source text. Errors were found in approximately 1% of the inspected subset. The most common errors were missing structured fields and cross-source confusion. For example, the pipeline might misattribute a semantic explanation originally associated with the Oracle Bone form to the Bronze Inscription field of the same character.

For JieZi-Bench, all structured metadata records were exhaustively checked by domain experts. Errors requiring correction were found in approximately 5% of entries; all identified errors were corrected before proceeding to VQA generation.

Stage 3: VQA Generation and Verification. At Stage 3, VQA pairs were generated from the verified metadata by an LLM guided by expert-designed templates. The use of templates is essential because unconstrained LLM generation would sufer from three problems: (1) format inconsistency, where question phrasing and answer granularity vary unpredictably, hindering standardized evaluation; (2) uneven subtask coverage, where the LLM tends to favor subtasks it handles well while underrepresenting harder ones such as COME and EVOI; (3) hallucination risk, where the LLM may fab ricate etymological explanations or evolutionary paths absent from the source dictionaries. Our template-guided approach addresses all three issues. Each template prescribes the question format and expected answer scope for a given subtask, ensuring uniform coverage and consistent granularity. The LLM then generates naturallanguage VQA pairs grounded in the metadata fields specified by the template, combining the scalability of LLM generation with the controllability of structured constraints.

Table 6: Training settings and hyper-parameters for diferent model scales.
<table><tr><td>Setting</td><td>Qwen3.5-2B</td><td>Qwen3.5-4B</td><td>Qwen3.5-9B</td></tr><tr><td>Batch size</td><td>128</td><td>128</td><td>64</td></tr><tr><td>Learning rate</td><td>1e-5</td><td>1e-5</td><td>1e-5</td></tr><tr><td>LR schedule</td><td>cosine</td><td>cosine</td><td>cosine</td></tr><tr><td>Warmup ratio</td><td>0.05</td><td>0.05</td><td>0.05</td></tr><tr><td>Weight decay</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Max length</td><td>2048</td><td>2048</td><td>1024</td></tr><tr><td>Epochs</td><td>5</td><td>5</td><td>5</td></tr><tr><td>Precision</td><td>bf16</td><td>bf16</td><td>bf16</td></tr><tr><td>Hardware</td><td>8 910B</td><td>8 910B</td><td>8 910B</td></tr><tr><td>Training time</td><td>96 NPU hours</td><td>192 NPU hours</td><td>384 NPU hours</td></tr></table>

Despite these safeguards, template-guided generation introduces its own errors. The generation process can produce image-content mismatches (e.g., pairing a glyph image with an answer describing a diferent script period), and natural-language answers may deviate from the source text in phrasing or scope. Moreover, VQA-level verification covers broader dimensions than Stage 2, including not only field-level correctness but also the naturalness of questionanswer formulations and the appropriateness of paired images, which increases the likelihood of flagging issues.

For JieZi-Dataset, we randomly inspected 5,000 VQA instances. Since the underlying metadata had already been verified at Stage 2, the sampling ratio was reduced accordingly. Errors were found in approximately 3% of the inspected subset; the slightly higher rate compared to Stage 2 reflects the new error sources and broader verification scope described above rather than upstream propagation.

For JieZi-Bench, all VQA pairs were manually checked, since benchmark reliability is critical for evaluation. Errors requiring correction were found in approximately 10% of entries, consistent with the pattern observed at earlier stages: the benchmark’s stricter acceptance criteria and cross-source complexity lead to a higher correction rate.

Overall, JieZi-Dataset adopts random spot-checking at Stages 2 and 3 to balance scale and quality, whileJieZi-Bench uses exhaustive manual verification at all stages. These combined strategies ensure that JieZi-Bench achieves expert-level reliability for evaluation, while JieZi-Dataset maintains suficient quality for training at scale.

## C Experimental Details and Full Results

## C.1 Additional Training Details

Table 6 summarizes the hyperparameter settings for instruction tuning. All experiments were conducted on 8 Ascend 910B NPUs.

## C.2 Ablation Studies

Tab. 7 reports the impact of each data construction stage on Qwen3.5- 2B. The first row corresponds to the few-shot base model without fine-tuning. We select CHAR, COMF, COMI, and EVOI as representative subtasks, as they span all four ACCE levels and cover both classification and generation objectives.

Table 7: Ablation study on diferent data construction stages. Struct.: structured metadata; VQA: After VQA generation; Open-src: VQA with open-source data augmentation.
<table><tr><td>Struct. VQA Open-src</td><td></td><td></td><td>CHAR↑</td><td>COMF↑</td><td>COMI ↑</td><td colspan="2">EVOI</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>FAC ↑</td><td>SCE ↑</td></tr><tr><td></td><td></td><td></td><td>18.9</td><td>13.7</td><td>17.0</td><td>7.9</td><td>20.2</td></tr><tr><td></td><td></td><td></td><td>27.0</td><td>29.7</td><td>24.1</td><td>15.4</td><td>26.3</td></tr><tr><td></td><td></td><td></td><td>35.3</td><td>31.4</td><td>37.1</td><td>25.6</td><td>31.9</td></tr><tr><td></td><td></td><td>1</td><td>41.8</td><td>32.7</td><td>41.3</td><td>27.9</td><td>36.7</td></tr></table>

Each stage contributes a distinct gain profile. Structured metadata produces the largest improvement on the classification-oriented COMF (+16.0), since explicit component-function labels ofer dense supervision that directly matches this subtask. In contrast, gains on generative subtasks remain moderate (COMI +7.1, FAC +7.5). VQA reformatting reverses this pattern: the generative subtasks benefit most (COMI +13.0, FAC +10.2), whereas COMF improves by only +1.7. This contrast suggests that recasting structured records as natural-language QA pairs trains the model to articulate multi-step reasoning rather than simply retrieve labels.

Open-source augmentation, in turn, primarily strengthens visual robustness (CHAR +6.5) and scholarly expression (SCE +4.8). Factual subtasks show smaller gains (COMF +1.3, FAC +2.3), confirming that their performance is bottlenecked by knowledge rather than visual diversity. Taken together, the three stages address complementary dimensions of factual grounding, reasoning articulation, and visual robustness. Their consistent, non-overlapping improvements validate the necessity of each stage in the proposed pipeline.

## C.3 Human Validation of the LLM-as-a-Judge

To rigorously validate the reliability ofour LLM-as-a-Judge protocol for the Evolution Interpretation (EVOI) task, we conducted a human-LLM agreement study. We randomly sampled 200 responses in total across 4 representative models to ensure a diverse distribution of response qualities. Three human experts with backgrounds in Chinese paleography independently evaluated these responses on a scale of 1 to 5, focusing on two dimensions: Fact Alignment (FAC) and Scholarly Expression (SCE).

We computed the Pearson (�) and Spearman (�) correlation coefficients between the automated LLM judge scores and the human ground truth. As shown in Tab. 9, the LLM judge exhibits strong correlations (> 0.60) with expert evaluations across both metrics. The results confirm that the LLM judge aligns closely with human scholarly standards, demonstrating its efectiveness and reliability as an automated evaluation metric for ancient character exegesis.

BERTScore validation. We further validate BERTScore, used for the open-ended ORIM and COMI tasks, under the same expert protocol. Three paleography experts scored 200 responses from four models on a 1–5 scale, and we computed Pearson and Spearman correlations between BERTScore and averaged human ratings.

Table 8: Correlation between human expert ratings and BERTScore on the ORIM and COMI tasks.
<table><tr><td>Task</td><td>Pearson (r) Spearman (ρ)</td><td></td></tr><tr><td>ORIM</td><td>0.71</td><td>0.71</td></tr><tr><td>COMI</td><td>0.73</td><td>0.72</td></tr></table>

Table 9: Correlation between human expert ratings and LLM judge scores on the EVOI task.
<table><tr><td>Metric</td><td></td><td>Pearson (r) Spearman (ρ)</td></tr><tr><td>FAC</td><td>0.68</td><td>0.71</td></tr><tr><td>SCE</td><td>0.61</td><td>0.64</td></tr></table>

As shown in Tab. 8, BERTScore achieves strong human correlation (Pearson ≥ 0.71) on both tasks. Together with the LLM-as-a-Judge validation in Tab. 9, these results confirm that both automated metrics provide suficient reliability for comparative evaluation of open-ended exegetical responses.

Judging prompts. We provide the full prompts used in our LLM-based judging protocol for EVOI. Fig. 11 presents the prompt for evaluating Fact Alignment, while Fig. 12 presents the prompt for evaluating Scholarly Expression. We used Doubao-seed-2-0- lite-260215 [9] as our LLM-as-a-judge model.

## C.4 Main Results on JieZi-Bench and Analysis

Due to space limitations in the main text, we present the complete experimental results in Table 10, including additional models not shown in the main paper. Below, we provide a detailed analysis of the key findings.

Scaling law holds within model families. Tab. 10 includes additional model variants not reported in the main paper, enabling within-family comparisons among general MLLMs. Across all families, larger variants consistently outperform smaller ones. Gemini 3.1 Pro surpasses Gemini 3.1 Flash on every metric (e.g., CHAR: 29.9 vs. 25.7; SCRC: 76.6 vs. 56.5), Qwen3.5-397b-a17b outperforms 35b-a3b (e.g., SCRC: 67.1 vs. 61.4), and InternVL3.5-8B consistently exceeds its 4B counterpart (e.g., ORIM: 58.0 vs. 24.3). These withinfamily comparisons, together with the cross-family results in the main paper, confirm that the general scaling trend holds for ACCE.

Domain-specialized pretraining does not guarantee ACCE proficiency. tonggu-vl-2B [10] is a model specifically fine-tuned on massive ancient Chinese document corpora for tasks such as text recognition and reading comprehension. Despite this domain specialized pretraining, it scores below 5% on most ACCE subtasks (e.g., CHAR: 3.1, STRC: 0.2, COMR/COMF/COMI: 0.0) and achieves only 12.9 on ORIM. This reveals a critical distinction: recognizing and reading ancient texts is fundamentally diferent from performing scholarly exegesis, which requires structured reasoning about glyph form, component function, semantic origin, and diachronic evolution. The failure of tonggu-vl demonstrates that ACCE poses genuinely novel challenges beyond the reach of existing domainspecific models, thereby validating both the task formulation and the necessity ofJieZi-Dataset as a dedicated training resource. Conversely, once equipped with JieZi-Dataset, even a 9B model outperforms general MLLMs that are one to two orders of magnitude larger (e.g., Kimi-K2.5 170B, GLM-4.6V 107B, Qwen3.5-397b) on nearly every subtask, further confirming that high-quality domainspecific data is a far more efective lever than model scale alone for knowledge-intensive tasks like ACCE.

![](images/f47a7c708c5cac5ca42c6de7a179cb9e26d5d8512edeb1de100cc74a950137f7.jpg)  
Figure 11: Prompt template for evaluating fact alignment within our LLM-as-a-judge protocol.

Table 10: Main results on JieZi-Bench across four paleographic levels: L1 (Basic Info: CHAR, SCRC), L2 (Glyph Form: STRC, COMR, COMF, COMI, FORC), L3 (Meaning: ORIM), and L4 (Evolution: COME, EVOI-FAC/SCE). All metrics are scaled to 0-100.
<table><tr><td rowspan="2">Method</td><td colspan="2">L1</td><td colspan="5">L2</td><td colspan="2">L3</td><td colspan="2">L4</td></tr><tr><td>CHAR↑</td><td>SCRC↑</td><td>STRC↑</td><td>COMR↑</td><td>COMF↑</td><td>COMI↑</td><td>FORC↑</td><td>ORIM↑</td><td>COME↑</td><td>EVOI</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>FAC↑</td><td>SCE↑</td></tr><tr><td>Closed-source MLLMs</td><td></td><td>39.2</td><td>50.4</td><td>13.8</td><td>11.9</td><td>10.3</td><td>33.1</td><td>59.9</td><td>12.4</td><td>10.2</td><td>16.8</td></tr><tr><td>GPT-5.4 Thinking [38]</td><td>10.4 29.9</td><td>76.6</td><td>65.7</td><td>31.9</td><td>29.9</td><td>24.8</td><td>56.1</td><td>62.8</td><td>30.3</td><td>22.3</td><td>27.1</td></tr><tr><td>Gemini 3.1 Pro [20] Gemini 3.1 Flash [19]</td><td>25.7</td><td>56.5</td><td>53.5</td><td>24.6</td><td>22.5</td><td>19.1</td><td>44.8</td><td>54.1</td><td>23.2</td><td>17.5</td><td>23.1</td></tr><tr><td>Claude opus 4.6 [1]</td><td>22.9</td><td>65.8</td><td>63.5</td><td>28.0</td><td>26.3</td><td>22.3</td><td>50.6</td><td>59.3</td><td>26.7</td><td>19.8</td><td>26.3</td></tr><tr><td>Doubao-Seed-2.0-pro [8]</td><td>36.6</td><td>68.5</td><td>69.8</td><td>37.8</td><td>34.4</td><td>29.9</td><td>59.9</td><td>63.2</td><td>36.1</td><td>31.7</td><td>38.7</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Open-source MLLMs Kimi-K2.5 [47] (170B)</td><td>31.5</td><td>70.1</td><td>68.8</td><td>39.2</td><td>36.6</td><td>31.5</td><td>58.1</td><td>66.6</td><td>36.4</td><td>28.3</td><td>34.6</td></tr><tr><td>GLM-4.6V [48] (107B)</td><td>18.2</td><td>49.7</td><td>48.9</td><td>21.8</td><td>20.4</td><td>18.2</td><td>42.2</td><td>48.2</td><td>20.9</td><td>15.5</td><td>20.2</td></tr><tr><td>Qwen3.5-397b-a17b [41]</td><td>26.2</td><td>67.1</td><td>69.4</td><td>32.1</td><td>29.9</td><td>26.5</td><td>54.4</td><td>59.5</td><td>28.3</td><td>22.2</td><td>30.7</td></tr><tr><td>Qwen3.5-35b-a3b [41]</td><td>26.6</td><td>61.4</td><td>66.5</td><td>32.4</td><td>29.3</td><td>27.3</td><td>57.8</td><td>65.2</td><td>31.0</td><td>19.6</td><td>25.6</td></tr><tr><td>InternVL3.5-8B [55]</td><td>00.1</td><td>09.0</td><td>14.6</td><td>0.0</td><td>0.0</td><td>0.0</td><td>21.4</td><td>58.0</td><td>0.0</td><td>24.8</td><td>0.1</td></tr><tr><td>InternVL3.5-4B [55]</td><td>0.0</td><td>0.0</td><td>0.9</td><td>0.0</td><td>0.0</td><td>0.0</td><td>8.2</td><td>24.3</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>tonggu-vl-2B [10]</td><td>3.1</td><td>05.9</td><td>0.2</td><td>0.0</td><td>0.0</td><td>0.0</td><td>4.6</td><td>12.9</td><td>0.0</td><td>1.1</td><td>0.2</td></tr><tr><td>Qwen3.5-2B [41]</td><td>18.9</td><td>22.6</td><td>31.6</td><td>17.3</td><td>13.7</td><td>17.0</td><td>30.7</td><td>62.3</td><td>12.9</td><td>7.9</td><td>20.2</td></tr><tr><td>Qwen3.5-2B + JieZi-Dataset</td><td>41.8+22.9</td><td>61.7+39.1</td><td>72.7+41.1</td><td>42.6+25.3</td><td>32.7+19.0</td><td>41.3+24.3</td><td>63.3+32.6</td><td>65.7+3.4</td><td>28.3+15.4</td><td>27.9+20.0</td><td>36.7+16.5</td></tr><tr><td>Qwen3.5-4B [41]</td><td>3.6</td><td>6.6</td><td>1.0</td><td>4.5</td><td>3.9</td><td>4.3</td><td>5.9</td><td>6.5</td><td>3.6</td><td>2.0</td><td>3.6</td></tr><tr><td>Qwen3.5-4B + JieZi-Dataset</td><td>49.0+45.4</td><td>67.6+61.0</td><td>74.1+73.1</td><td>47.2+42.7</td><td>37.9+34.0</td><td>42.4+38.1</td><td>64.8+58.9</td><td>65.9+59.4</td><td>33.2+29.6</td><td>28.5+26.5</td><td>37.2+33.6</td></tr><tr><td>Qwen3.5-9B [41]</td><td>22.3</td><td>51.0</td><td>40.2</td><td>27.4</td><td>24.6</td><td>26.1</td><td>49.1</td><td>64.7</td><td>22.4</td><td>15.0</td><td>23.5</td></tr><tr><td>Qwen3.5-9B + JieZi-Dataset</td><td>52.6+30.3</td><td>80.1+29.1</td><td>74.7+34.5</td><td>48.1+20.7</td><td>38.4+13.8</td><td>45.7+19.6</td><td>66.5+17.4</td><td>66.7+2.0</td><td>45.4+23.0</td><td>32.5+17.5</td><td>40.6+17.1</td></tr></table>

Remark on the anomalous behavior of Qwen3.5-4B. As shown in Tab. 10, few-shot Qwen3.5-4B scores extremely low across all eleven ACCE subtasks (e.g., CHAR: 3.6, STRC: 1.0, ORIM: 6.5), performing substantially below both the smaller 2B and the larger 9B variants. We verified that all three models share identical inference pipelines, prompts, and decoding parameters, ruling out implementation errors. Manual inspection ofthe 4B outputs (Fig. 13) reveals that the model tends to reproduce content from the few-shot demonstrations rather than generating answers grounded in the current query, producing outputs that bear no meaningful correspondence to the target glyph.

Since Qwen3.5’s training details are not publicly disclosed, we can only hypothesize about the root cause based on observed behavior. We identify two plausible contributing factors. (1) Copying bias in few-shot in-context learning (ICL): The model reproduces demonstration content instead of reasoning about the current query. This behavior directly aligns with prior findings that LLMs can develop a copying bias under in-context learning, reproducing surface patterns from examples rather than inducing the underlying task [36, 37]. (2) U-shaped scaling: The 2B (functional) → 4B (col lapsed) → 9B (functional) pattern is consistent with the U-shaped scaling phenomenon, where medium-sized models are drawn toward easier competing behaviors instead of the target task [35, 56].

After fine-tuning on JieZi-Dataset, Qwen3.5-4B fully recovers to the expected scaling order, achieving the largest absolute improvements among the three model scales (e.g., STRC: +73.1, ORIM: +59.4 vs. 2B’s +41.1 for STRC and +3.4 for ORIM, and 9B’s +34.5 for STRC and +2.0 for ORIM). This confirms that the bottleneck is not model capacity but misaligned few-shot behavior, which domain-specific supervision efectively overrides [63].

## C.5 Baseline Comparison

To evaluate whether the performance gains from JieZi-Dataset stem from domain-specific reasoning rather than mere knowledge exposure, we compare supervised fine-tuning (SFT) against three alternative training-free strategies on Qwen3.5-9B: (1) few-shot incontext learning (ICL) with top-3 retrieved examples, (2) retrievalaugmented generation (RAG), and (3) dictionary-context prompting. All baselines use the same model backbone, JieZi-Bench split, decoding settings, and evaluation metrics as Tab. 10. In ACCE, questions are generic templates that carry no character-specific information; the character identity is conveyed entirely by the glyph image, making a no-image baseline inapplicable to this task.

For few-shot ICL, we retrieve the top-3 training examples using only the query glyph image, without using benchmark labels or reference answers. For RAG, the model receives top-3 retrieved training metadata and QA records as context but is not fine-tuned. Dictionary-context prompting provides the retrieved dictionarystyle context directly in the prompt. For both few-shot ICL and RAG, we use SigLIP2-SO400M-patch14-384 as the image retriever: the query glyph and training glyphs are encoded into visual embeddings, and the top-3 nearest training examples are selected by cosine similarity.

![](images/ff607e2f747a1a6f29e630ba54746ebb62753e59800538081a9ef2ffa02bfca3.jpg)  
Figure 12: Prompt template for evaluating scholarly expression within our LLM-as-a-judge protocol.

As shown in Tab. 11, SFT on JieZi-Dataset outperforms all alternatives across all subtasks. Few-shot ICL with retrieved examples provides substantial improvement over the base model, indicating that in-context paleographic examples are informative. However, RAG and dictionary-context prompting, which inject the same paleographic knowledge in diferent forms, still fall far short of SFT. This confirms that the performance gain stems from learning domain-specific reasoning patterns through supervised fine-tuning, not merely from exposure to paleographic knowledge in any form.

![](images/a8e24f4617463dc652c4a76060e825292cb3c228c260a0e3265c660a4dc43239.jpg)  
Figure 13: An example of anomalous output from few-shot Qwen3.5-4B, illustrating the copying bias.

## C.6 Data Scaling Analysis

To examine whether the full 500K-scale training set is necessary or whether comparable performance is achievable with fewer QA pairs, we train Qwen3.5-2B with nested 25%, 50%, 75%, and 100% subsets of JieZi-Dataset. The subsets are strictly nested: the 25% subset is contained in the 50% subset, which is contained in the 75% subset. All runs use the same hyperparameters as Tab. 6 and are evaluated on the unchanged JieZi-Bench.

As shown in Tab. 12, the macro average increases monotonically from 36.5 at 25% to 46.8 at 100% (+10.3). While 25% already yields substantial gains over the base model (+13.3), the full set still brings clear improvements on most subtasks (e.g., SCRC 33.3→61.7, COMI 25.5→41.3), confirming that the full 500K scale provides real marginal benefit rather than redundant volume.

## C.7 Paraphrase Robustness Test

To test whether the fine-tuned model overfits to fixed template question patterns rather than learning genuine exegetical ability, we conduct a controlled paraphrase robustness test. For each instance across all subtasks, we replace the original question with two independently paraphrased versions while keeping the glyph image, reference answer, model (Qwen3.5-2B fine-tuned on JieZi-Dataset), decoding settings, and evaluation metrics unchanged. This isolates the efect of question wording from all other factors. The two paraphrases are generated by prompting Gemini to rewrite each template question with diferent lexical choices and sentence structures while preserving the query intent.

Table 11: Comparison of SFT on JieZi-Dataset against alternative training-free strategies on Qwen3.5-9B, evaluated on JieZi-Bench. All methods use the same backbone, decoding settings, and evaluation metrics.
<table><tr><td rowspan="2">Method</td><td colspan="2">L1</td><td colspan="5">L2</td><td>L3</td><td colspan="3">L4</td></tr><tr><td>CHAR↑</td><td>SCRC↑</td><td>STRC↑</td><td>COMR↑</td><td>COMF↑</td><td>COMI↑</td><td>FORC↑</td><td>ORIM↑</td><td>COME↑</td><td>FAC↑</td><td>SCE↑</td></tr><tr><td>Qwen3.5-9B</td><td>22.3</td><td>51.0</td><td>40.2</td><td>27.4</td><td>24.6</td><td>26.1</td><td>49.1</td><td>64.7</td><td>22.4</td><td>15.0</td><td>23.5</td></tr><tr><td>+ Few-shot ICL (top-3)</td><td>21.4</td><td>56.7</td><td>46.9</td><td>27.6</td><td>24.5</td><td>18.8</td><td>51.5</td><td>53.8</td><td>26.1</td><td>14.5</td><td>19.3</td></tr><tr><td>+ RAG</td><td>30.3</td><td>50.9</td><td>34.2</td><td>31.8</td><td>27.8</td><td>25.2</td><td>58.3</td><td>57.5</td><td>30.3</td><td>19.2</td><td>21.8</td></tr><tr><td>+ Dict-context prompting</td><td>22.2</td><td>37.0</td><td>49.9</td><td>27.4</td><td>21.4</td><td>18.9</td><td>58.3</td><td>53.8</td><td>25.9</td><td>15.6</td><td>18.3</td></tr><tr><td>+ JieZi-Dataset (SFT)</td><td>52.6</td><td>80.1</td><td>74.7</td><td>48.1</td><td>38.4</td><td>45.7</td><td>66.5</td><td>66.7</td><td>45.4</td><td>32.5</td><td>40.6</td></tr></table>

Table 12: Data scaling analysis on Qwen3.5-2B with nested subsets of JieZi-Dataset, evaluated on JieZi-Bench.
<table><tr><td rowspan="2">Training Data</td><td colspan="2">L1</td><td colspan="5">L2</td><td>L3</td><td colspan="3">L4</td></tr><tr><td>CHAR↑</td><td>SCRC↑</td><td>STRC↑</td><td>COMR↑</td><td>COMF↑</td><td>COMI↑</td><td>FORC↑</td><td>ORIM↑</td><td>COME↑</td><td>FAC↑</td><td>SCE↑</td></tr><tr><td>Qwen3.5-2B (base)</td><td>18.9</td><td>22.6</td><td>31.6</td><td>17.3</td><td>13.7</td><td>17.0</td><td>30.7</td><td>62.3</td><td>12.9</td><td>7.9</td><td>20.2</td></tr><tr><td>+ 25% JieZi-Dataset</td><td>34.0</td><td>33.3</td><td>67.7</td><td>36.5</td><td>30.5</td><td>25.5</td><td>45.0</td><td>56.1</td><td>23.0</td><td>21.1</td><td>28.5</td></tr><tr><td>+ 50% JieZi-Dataset</td><td>37.6</td><td>43.5</td><td>66.4</td><td>37.7</td><td>30.6</td><td>26.0</td><td>60.4</td><td>56.5</td><td>23.9</td><td>23.5</td><td>30.5</td></tr><tr><td>+ 75% JieZi-Dataset</td><td>40.6</td><td>52.1</td><td>67.0</td><td>40.1</td><td>31.4</td><td>28.1</td><td>62.3</td><td>59.9</td><td>28.0</td><td>25.8</td><td>33.2</td></tr><tr><td>+ 100% JieZi-Dataset</td><td>41.8</td><td>61.7</td><td>72.7</td><td>42.6</td><td>32.7</td><td>41.3</td><td>63.3</td><td>65.7</td><td>28.3</td><td>27.9</td><td>36.7</td></tr></table>

As shown in Tab. 13, the fine-tuned model remains stable across paraphrased questions. Although some subtasks show moderate drops (e.g., STRC −7.0, ORIM −7.1), the overall performance remains substantially above the base model across all variants. This indicates that the performance gains are not mainly driven by surface-level template pattern matching but by learned glyph-grounded exegeti cal content.

Table 13: Paraphrase robustness test on Qwen3.5-2B fine-tuned with JieZi-Dataset. “Original” uses the standard template questions; “Prompt $2 ^ { \mathfrak { s } }$ and “Prompt ${ \mathfrak { 3 } } ^ { \mathfrak { n } }$ are independently paraphrased versions. Avg. Δ reports the mean score change across the two paraphrased variants relative to the original.
<table><tr><td rowspan="2">Question</td><td colspan="2">L1</td><td colspan="5">L2</td><td>L3</td><td colspan="3">L4</td></tr><tr><td>CHAR↑</td><td>SCRC↑</td><td>STRC↑</td><td>COMR↑</td><td>COMF↑</td><td>COMI↑</td><td>FORC↑</td><td>ORIM↑</td><td>COME↑</td><td>FAC↑</td><td>SCE↑</td></tr><tr><td>Original</td><td>41.8</td><td>61.7</td><td>72.7</td><td>42.6</td><td>32.7</td><td>41.3</td><td>63.3</td><td>65.7</td><td>28.3</td><td>27.9</td><td>36.7</td></tr><tr><td>Prompt 2</td><td>44.6</td><td>58.0</td><td>63.9</td><td>42.5</td><td>33.0</td><td>39.6</td><td>64.2</td><td>57.3</td><td>41.1</td><td>25.8</td><td>29.8</td></tr><tr><td>Prompt 3</td><td>42.6</td><td>59.8</td><td>67.6</td><td>34.6</td><td>31.6</td><td>33.7</td><td>56.8</td><td>59.9</td><td>33.2</td><td>21.0</td><td>44.5</td></tr><tr><td>Avg. ∆</td><td>+1.8</td><td>-2.8</td><td>-7.0</td><td>-4.1</td><td>-0.4</td><td>-4.7</td><td>-2.8</td><td>-7.1</td><td>+8.9</td><td>-4.5</td><td>+0.5</td></tr></table>