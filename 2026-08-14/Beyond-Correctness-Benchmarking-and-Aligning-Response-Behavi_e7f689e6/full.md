# Beyond Correctness: Benchmarking and Aligning Response Behaviors in Hybrid-Thinking MLLMs

Xinming Wang<sup>1,2,5,∗</sup>, Weinong Wang<sup>2,†</sup>, Hongming Yang<sup>2</sup>, Yansong Lin<sup>3</sup>, Zheng Ruan<sup>2</sup>, Shangpin Peng<sup>2,4</sup>, Qiming Peng<sup>2</sup>, Nan Qiao<sup>2</sup>, Fengyuan Lu<sup>2</sup>, Guoqing Ma<sup>2</sup>, Marito Li<sup>2</sup>, Songyang Zhang<sup>2</sup>, Saiyong Yang<sup>2</sup>, Han Hu<sup>2</sup>, Yonglong Tian<sup>2</sup>, Xu-Yao Zhang<sup>1,†</sup>

<sup>1</sup>Institute of Automation, Chinese Academy of Sciences

<sup>2</sup>Large Language Model Department, Tencent

<sup>3</sup>University of Electronic Science and Technology of China

<sup>4</sup>Hong Kong University of Science and Technology

<sup>5</sup>Zhongguancun Academy

<sup>∗</sup>This work was done during an internship at Tencent., <sup>†</sup>Corresponding authors.

wangxinming2024@ia.ac.cn, {weinongwang,hmingyang}@tencent.com, xyz@nlpr.ia.ac.cn

Hybrid-thinking multimodal large language models (MLLMs) allow a single model to alternate between deliberative thinking and latency-eficient non-thinking inference. Although these modes difer in reasoning budget, their delivered responses should satisfy the same user-facing standard. Correctness alone may not characterize this response quality; we therefore evaluate task accuracy and response-pattern failures as complementary outcomes. We study this gap through responsepattern alignment: whether thinking and non-thinking interfaces preserve acceptable final-response behavior. We introduce PatternEval, a failure-enriched diagnostic benchmark comprising 2,415 multimodal prompts spanning visual perception and grounding, structured image understanding, and multimodal knowledge reasoning. PatternEval tests four recurrent failures: chain-of-thought leakage, response repetition, logical contradiction, and performative reasoning. Response-pattern failures are widespread across models from diferent providers, with non-thinking inference exhibiting substantially higher failure rates and thereby creating systematic misalignment between thinking and non-thinking interfaces. Motivated by this diagnosis, we develop PatternRM, a response-level reward model, and PatternRL, which introduces pattern-specific penalties during reinforcement learning. Experiments on Qwen3-VL-4B and Qwen3-VL-8B show that incorporating pattern-specific penalties into reinforcement learning can mitigate cross-mode misalignment while incurring a marginal task performance trade-of. Together, PatternEval and PatternRL provide an evaluation-and-training framework for aligning user-visible response patterns across hybrid-thinking interfaces.

Date: Jul 28, 2026

## 1 Introduction

Multimodal large language models (MLLMs) increasingly serve both reasoning-intensive tasks and latencysensitive interactions. Modern systems therefore expose hybrid-thinking interfaces: a deliberative thinking mode allocates additional test-time computation, while a non-thinking mode returns a direct answer under tighter latency and token budgets [3, 10, 12, 32]. This interface makes reasoning efort controllable without maintaining separate models, but it also creates diferent user-facing behaviors under diferent efort, which should remain reliable under the same deployment standard.

The trade-of between capability and eficiency does not justify a trade-of in final-response quality. Regardless of the inference route, an answer should remain grounded, coherent, appropriately concise, and free of unintended process narration. Correctness-driven post-training can steer models toward expected answers without directly constraining all of these response properties: generated text may still expose reasoning-like traces, repeat content, retain incompatible statements, or make unsupported factual claims [7, 13, 26, 45]. We call a mode-dependent diference in such user-visible behavior response-pattern misalignment. We therefore ask a question complementary to answer correctness: when the inference mode changes, does the model preserve an acceptable and mode-consistent response pattern?

To this end, we introduce PatternEval, a failure-enriched diagnostic benchmark comprising 2,415 multimodal prompts drawn from nine constituent categories. These categories are organized into three broad task families: visual perception and grounding, OCR and structured-image understanding, and multimodal knowledge reasoning. PatternEval deliberately concentrates dificult prompts that expose response-pattern failures, allowing it to stress-test whether models preserve acceptable user-facing behavior under matched thinking and non-thinking interfaces. It captures four recurrent failures: chain-of-thought leakage, response repetition, logical contradiction, and performative reasoning. Despite substantial advances in overall model capability, these gains do not transfer uniformly across inference interfaces. As shown in Figure 1, frontier models still exhibit pronounced thinking–non-thinking misalignment, with non-thinking Trigger rates consistently higher and gaps reaching 48.64%. This suggests that strong task performance does not necessarily guarantee stable response behavior across diferent patterns, exposing an interface-dependent robustness gap that conventional accuracy metrics may overlook.

![](images/79097f8b235674853336a49ef46b7e12798330b180e4062db30569e7675b5267.jpg)  
(a) Pattern Misalignment of Flagship Models under Hybrid Thinking

![](images/99b98c6850bb5fc9fb0a9bb38f2e1ace8b8e37b07d18243feeb2f8a40b2e4df3.jpg)  
(b) The Composition of PatternEval  
Figure 1 Response-pattern misalignment and PatternEval. (a) Response-pattern misalignment in flagship models under hybrid-thinking configurations. (b) Composition of PatternEval.

These outcomes motivate the addition of response-pattern objectives to the post-training process, which does not directly constrain every property of the delivered response [1, 7, 8, 35]. We therefore train PatternRM to recognize the four response-pattern failures and incorporate its category-specific penalties into PatternRL. On Qwen3-VL-4B and Qwen3-VL-8B, PatternRL reduces non-thinking Trigger relative to correctness-only BaseRL by 13.08 and 14.35 percentage points, while aggregate accuracy changes by less than one percentage point. Further experiments show that PatternRL can also be integrated into broader task training, consistently mitigating response-pattern failures across diverse settings and increasing the proportion of usable model responses without materially compromising task performance.

We highlight the following contributions.

• We formulate response-pattern alignment as a requirement complementary to answer correctness and introduce PatternEval, a failure-enriched multimodal challenge set that stress-tests four user-visible response

failures under hybrid-thinking interfaces.

• Through matched-mode evaluation across diverse hybrid-thinking MLLMs, we identify a consistent thinking– non-thinking misalignment: non-thinking mode exhibits higher Trigger rates, with substantial gaps persisting even among frontier models

• We translate response-pattern evaluation into an explicit post-training objective through PatternRM and PatternRL. Compared with correctness-only BaseRL, PatternRL substantially reduces response-pattern failures while preserving aggregate accuracy, and further improves response usability when integrated into broader task training.

## 2 Related Work

Controlling hybrid thinking and reasoning. Chain-of-thought prompting and self-consistency showed that eliciting or sampling intermediate reasoning can improve answer accuracy [15, 43, 44]. Reasoning-specialized models then made deliberation explicit test-time computation [3, 12], while Qwen3 and GLM-4.5 expose thinking and direct-response modes within a single model [10, 32]. Work on controlling this computation follows three routes: learned routing or mode tokens decide whether to reason [5, 37]; budget-aware control and adaptive long/short policies regulate how much reasoning to allocate [23, 46, 50]; and data-centric or architecture-level designs seek stronger behavioral separation [40, 41]. Direct answers can remain competitive under tight budgets, so longer deliberation is not uniformly preferable [24]. However, these methods primarily optimize accuracy, eficiency, or routing rewards rather than the delivered response. Reasoning may still leak into non-thinking outputs, and a policy can satisfy a non-thinking reward while deliberating visibly [7]. Thus, controlling when and how much to reason leaves unresolved whether the selected mode produces a behaviorally appropriate final response.

Response-pattern evaluation beyond correctness. This post-routing question connects reasoning control to evaluation beyond correctness. General benchmarks assess robustness, calibration, truthfulness, safety, and instruction following [19, 20, 33, 53]; multimodal suites add expert reasoning, hallucination, visual consistency, and trustworthiness [6, 11, 18, 22, 38, 48, 51]. Instruction-following evaluation asks whether outputs satisfy explicit constraints, while studies of text degeneration, repetition, and hallucination expose recurrent responselevel failures [13, 26, 45]. Preference datasets and reward-model benchmarks extend evaluation to helpfulness, visual faithfulness, safety, and judgment quality [16, 17, 25, 47], but aggregate scores can obscure specific failures. Model-based evaluation scales such judgments, yet introduces length, self-preference, position, and superficial-reflection biases [4, 21, 30, 39, 52]. PatternEval instead treats the response pattern itself as the evaluation object, separating leakage, repetition, contradiction, and unsupported performative reasoning under an auditable taxonomy. PatternRM then converts these failure labels into training signals, closing the loop from post-routing diagnosis to policy optimization.

## 3 PatternEval

## 3.1 Benchmark Construction and Composition

Construction Process. PatternEval is deliberately constructed as a failure-enriched diagnostic benchmark from a broad candidate pool of image–prompt pairs spanning visual perception and grounding, OCR and structured-image understanding, and multimodal knowledge reasoning. Its construction follows three sequential steps. First, we generate rollout responses for the candidate prompts and identify recurrent user-visible failures, with particular attention to dificult non-thinking responses. These observations provide empirical anchors for the benchmark. Second, we expand from the initial cases to related tasks, prompt structures, and response forms, covering broader manifestations of each failure pattern rather than isolated examples. Third, we apply pattern-oriented sampling and constituent-category balancing: categories dominated by deterministic answer-format checks are down-sampled, categories with meaningful response variation are preferentially retained, and high-volume source categories are capped to prevent them from dominating aggregate results. The resulting 2,415 image–prompt pairs concentrate evaluation on conditions that expose the targeted failures while retaining broad multimodal task coverage. PatternEval therefore measures conditional robustness on a selected stress test rather than natural task or failure prevalence in deployment.

PatternEval Composition. As summarized in Table 1, PatternEval contains 2,415 prompts drawn from nine constituent categories and organized into three task families. VG (visual perception and grounding) is the largest family, comprising 1,181 prompts (48.9%) across OOD perception, content recognition, grounding, and factuality. These tasks emphasize accurate visual interpretation and faithful grounding of responses in image content, while also covering cases in which familiar recognition patterns may not transfer reliably. OS (OCR and structured-image understanding) contains 579 prompts (24.0%), spanning both text-centric OCR tasks and chart understanding. This family introduces structure-sensitive inputs for which successful responses require not only extracting local visual elements, but also preserving their spatial, tabular, or quantitative relations. The remaining 655 prompts (27.1%) constitute KR (multimodal knowledge reasoning), including STEM, knowledge, and general reasoning tasks that require integrating visual evidence with domain knowledge or multi-step inference. Although the three families difer in size, their combination deliberately covers perception-heavy, structure-sensitive, and knowledge-intensive settings. This breadth enables PatternEval to examine whether responsepattern failures are confined to particular task types or persist across heterogeneous multimodal demands under matched rea

Table 1 Composition of PatternEval across three task families. Subtotals report family-level prompt counts.
<table><tr><td>Constituent category Number</td></tr><tr><td>VG Visual perception and grounding</td></tr><tr><td>OOD perception 524 Content recognition 350</td></tr><tr><td>Grounding 251</td></tr><tr><td>Factuality 56</td></tr><tr><td>− Subtotal 1,181</td></tr><tr><td>OS OCR and structured-image understanding</td></tr><tr><td>OCR 299</td></tr><tr><td>Chart understanding 280</td></tr><tr><td>− Subtotal 579</td></tr><tr><td>KR Multimodal knowledge reasoning</td></tr><tr><td>STEM 200</td></tr><tr><td>Knowledge 185</td></tr><tr><td>Reasoning 270</td></tr><tr><td>− Subtotal 655</td></tr><tr><td>Total 2,415</td></tr></table>

## 3.2 Evaluation Protocol

![](images/6d0e21810f1f48c0c3fd1ca1fc04b3b2bf106b0d8e034b14bbb7e370cbf250aa.jpg)  
Figure 2 Representative response-pattern annotations. Each column contrasts an acceptable final response with one that triggers the corresponding PatternEval label.

Response-pattern taxonomy. PatternEval targets four recurrent failures identified through qualitative inspection of multimodal rollout traces: chain-of-thought leakage, response repetition, logical contradiction, and performative reasoning. These four failure patterns occur in user-facing responses yet remain dificult for correctness verifiers to detect. We use the following operational definitions:

• Chain-of-thought leakage (CoT): internal-process traces, draft analysis, self-correction, or explicit thinking markers appear in the final user-visible answer. A concise, polished explanation is excluded. Because generated rationales need not faithfully reveal a model’s internal causal process [36], this is an observable response-style label rather than evidence that private internal states were exposed.

• Response repetition (Rep): the response unnecessarily repeats the same sentence, semantic content, structure, or conclusion in a way that harms eficiency or readability; necessary restatement and limited emphasis are excluded.

• Logical contradiction (Con): the final answer retains mutually incompatible claims about the same object, value, or conclusion under the same conditions without resolving the conflict.

• Performative reasoning (PR): the response presents an unsupported reasoning wrapper, cites evidence that does not entail its conclusion, or stages analysis without information gain. In multimodal tasks, relevant support should refer to observable objects, text, quantities, positions, relations, chart values, or paths.

Taken together, CoT and Rep are the dominant response-pattern failures and primarily capture undesirable user-visible form, whereas Con and PR require semantic and multimodal consistency judgments. The four labels are not mutually exclusive. Because contradictions and unsupported claims often appear within leaked deliberation, we apply a CoT-priority attribution rule: evidence already attributed to CoT is not counted again as Con or PR, while an independent contradiction or unsupported justification outside the leaked reasoning trace may still trigger the corresponding label. Rep may likewise co-occur with CoT when it reflects a distinct repetitive pattern. Consequently, the four pattern rates are coupled and should be interpreted jointly.

Evaluation settings. For each image–prompt pair $x = ( v , q )$ with reference answer g, model M produces an output under inference mode $m \in \{ \mathrm { N T } , \mathrm { T } \}$ :

$$
\tilde { y } _ { m } \sim M ( \cdot \mid x , m ) ,\tag{1}
$$

where NT and T denote non-thinking and thinking inference, respectively. If the interface exposes separate reasoning and final-response channels, we define $\tilde { y } _ { m } = ( r _ { m } , y _ { m } )$ , where $r _ { m }$ is the optional reasoning-channel output and $y _ { m }$ is the user-facing response. We evaluate only $y _ { m } ;$ any separately returned $r _ { m }$ is excluded from both correctness and pattern evaluation. None of the evaluated non-thinking interfaces returns a separate reasoning channel. This notation concerns only observable model outputs and does not assume private internal states.

Each response $y _ { m }$ is independently evaluated by a verifier judge and a pattern judge. The verifier judge assigns

$$
c ( x , y _ { m } , g ) = \mathcal { I } _ { \mathrm { v e r } } ( x , y _ { m } , g ) \in \{ 0 , 1 \} ,\tag{2}
$$

where $c = 1$ denotes a correct answer. We instantiate $\mathcal { I } _ { \mathrm { v e r } }$ with Qwen3-Max and apply the task-specific evaluation criteria of the corresponding source dataset.

The pattern judge assigns

$$
\begin{array} { r } { \mathbf { b } ( x , y _ { m } ) = \mathcal { I } _ { \mathrm { p a t } } ( x , y _ { m } ) = [ b _ { \mathrm { c o t } } , b _ { \mathrm { r e p } } , b _ { \mathrm { c o n } } , b _ { \mathrm { p r } } ] \in \{ 0 , 1 \} ^ { 4 } . } \end{array}\tag{3}
$$

We instantiate $\mathcal { I } _ { \mathrm { p a t } }$ with Seed-2.0-Pro. The four labels indicate chain-of-thought leakage, response repetition, logical contradiction, and performative reasoning, respectively, with $b _ { k } = 1$ indicating that the corresponding failure is present. The complete pattern-judge prompt is provided in Section E.

For a set of N prompts, define the bad pattern trigger indicator as

$$
b _ { \mathrm { a n y } } ( x , y _ { m } ) = \mathbf { 1 } \biggl [ \operatorname* { m a x } _ { k } b _ { k } ( x , y _ { m } ) = 1 \biggr ] .\tag{4}
$$

We then compute

$$
\operatorname { A c c } _ { m } = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } c ( x _ { i } , y _ { i , m } , g _ { i } ) , \qquad \operatorname { T r i g g e r } _ { m } = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } b _ { \mathrm { a n y } } ( x _ { i } , y _ { i , m } ) .\tag{5}
$$

Thus, higher Acc indicates better task performance, whereas lower Trigger indicates fewer undesirable user-facing response patterns.

Because both modes are evaluated on the same prompt set, we define their accuracy and response-pattern gaps as $\Delta _ { \mathrm { a c c } } = \mathrm { A c c } _ { \mathrm { T } } - \mathrm { A c c } _ { \mathrm { N T } }$ and $\Delta _ { \mathrm { p a t } } = \mathrm { T r i g g e r } _ { \mathrm { N T } } - \mathrm { T r i g g e r } _ { \mathrm { T } }$ . A positive $\Delta _ { \mathrm { a c c } }$ indicates an accuracy advantage for thinking inference, while positive $\Delta _ { \mathrm { p a t } }$ indicates a higher aggregate pattern-failure rate under non-thinking inference. The sign of $\Delta _ { \mathrm { p a t } }$ records the direction of this interface gap, and $| \Delta _ { \mathrm { p a t } } |$ records its magnitude. We report both mode-specific Trigger rates together with $\Delta _ { \mathrm { p a t } }$ , since the same gap may arise when both modes have either low or high failure rates.

## 3.3 Meta-Judge Analysis

Pattern-judge calibration-set construction. To calibrate the automatic pattern judge, we construct a separately annotated pattern-judge calibration set containing 2,500 fixed responses. These responses are not sampled directly from PatternEval. Instead, we collect rollouts generated by model variants obtained from the supervised fine-tuning (SFT), MixRL, and OPD stages of Hy3-Vision on a separate test set drawn from the same source tasks as PatternEval. This design evaluates every candidate judge on the same responses while covering output distributions produced by diferent post-training stages and avoiding overlap with the benchmark samples used for model evaluation.

We use Seed-2.0-Pro, Kimi-K2.6, and Qwen3.5-397B as the initial judges. Each judge assigns binary labels for the four failure types. We then divide the rollout responses into consensus samples, for which all three judges produce the same labels, and disputed samples, for which their predictions difer. The two strata are sampled at an approximate 7:3 ratio, respectively, so that the resulting dataset contains both representative cases and dificult cases that distinguish judge capability. During filtering, we additionally control the predicted bad pattern trigger rate at approximately 70%, ensuring suficient positive examples for reliable failure-label-level evaluation. The selected 2,500 responses are subsequently annotated by human annotators under the same four-label rubric, and the human annotations serve as the reference labels.

Meta-judge evaluation and results. Response-pattern labels are less mechanically verifiable than final-answer correctness and therefore require an explicitly calibrated judging protocol. We evaluate five candidate judges on the same 2,500 fixed responses against their human reference annotations. As shown in Table 2, each judge predicts the four binary pattern labels, and we report failure-label-level precision, recall, and F1 together with their macro-averages.

Table 2 Meta-judge calibration (%). Overall averages over the four failure labels; bold and underlined values indicate the best and second-best results, respectively.
<table><tr><td rowspan="2">Judge</td><td rowspan="2">Input</td><td colspan="3">CoT.</td><td colspan="3">Rep.</td><td colspan="3">Con.</td><td colspan="3">PR.</td><td colspan="3">Overall</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>GPT-5.5</td><td>Image+text Text only</td><td>98.5 98.4</td><td>92.4 92.2</td><td>95.3 95.2</td><td>83.5 83.2</td><td>92.9 93.8</td><td>87.9 88.2</td><td>46.3 47.5</td><td>85.5 88.9</td><td>60.1 61.9</td><td>54.4 55.2</td><td>79.3 75.0</td><td>64.5 63.6</td><td>70.7 71.1</td><td>87.5 87.5</td><td>77.0 77.2</td></tr><tr><td>Seed-2.0-Pro</td><td>Image+text Text only</td><td>98.6 98.6</td><td>84.8 84.7</td><td>91.2 91.1</td><td>83.5 83.4</td><td>91.5 91.6</td><td>87.3 87.3</td><td>53.7 55.6</td><td>74.8 67.5</td><td>62.5 61.0</td><td>50.4 44.8</td><td>74.1 68.1</td><td>60.0 54.0</td><td>71.6 70.6</td><td>81.3 78.0</td><td>75.3 73.4</td></tr><tr><td>Hy3</td><td>Text only</td><td>98.7</td><td>83.0</td><td>90.2</td><td>84.2</td><td>80.7</td><td>82.4</td><td>48.8</td><td>69.8</td><td>57.4</td><td>49.5</td><td>57.1</td><td>53.0</td><td>70.3</td><td>72.7</td><td>70.8</td></tr><tr><td>Kimi-K2.6</td><td>Image+text Text only</td><td>97.7 97.7</td><td>89.1 87.7</td><td>93.2 92.4</td><td>87.1 86.5</td><td>76.8 79.2</td><td>81.6 82.7</td><td>55.8 49.4</td><td>70.1 65.0</td><td>62.1 56.1</td><td>58.4 64.1</td><td>40.7 36.0</td><td>48.0 46.1</td><td>74.8 74.4</td><td>69.2 67.0</td><td>71.2 69.3</td></tr><tr><td>Qwen3.5-397B</td><td>Image+text Text only</td><td>97.6 97.5</td><td>87.3 88.5</td><td>92.2 92.8</td><td>86.4 83.2</td><td>84.6 88.6</td><td>85.5 85.8</td><td>44.7 52.0</td><td>75.2 77.8</td><td>56.1 62.3</td><td>36.2 55.6</td><td>61.6</td><td>45.6 33.5 41.8</td><td>66.2 72.1</td><td>77.2 72.1</td><td>69.9 70.7</td></tr></table>

As shown in Table 2, CoT leakage and repetition are judged more reliably because they often exhibit explicit surface-level cues, whereas contradiction and performative reasoning require finer-grained semantic assessment, including cross-sentence consistency, visual grounding, and distinguishing evidence-supported analysis from merely reasoning-like language. Accordingly, the latter two labels yield lower and more variable precision and recall across judges. Since the operational judge is used for both benchmark evaluation and training-data filtering, we consider recall alongside precision and F1 to reduce undetected response-pattern failures. GPT-5.5 achieves the highest overall F1, while Seed-2.0-Pro with image access attains the best contradiction F1 and directly grounds its decisions in the visual input. Balancing failure-label-level performance, multimodal grounding, recall, and inference cost, we select Seed-2.0-Pro with image access as the operational pattern judge for PatternEval.

## 4 Experiments

## 4.1 Experimental Setup

Models. We evaluate 25 model configurations spanning the Qwen series, Kimi, Seed, Claude, Mimo, and GPT families. Each configuration is evaluated under matched thinking and non-thinking modes. We follow the oficial default inference settings and vary only the available reasoning control, while keeping the remaining exposed decoding parameters fixed. For externally hosted models, we use the reasoning modes or efort controls provided by their APIs. Full nine-constituent-category results are reported in Section B.

Protocol. For each configuration, we generate thinking and non-thinking responses on the same 2,415 PatternEval prompts, enabling paired comparisons across inference modes. Qwen3-Max evaluates task correctness according to the protocol and reference answers of each source task, whereas Seed-2.0-Pro, with access to the input image, evaluates the four response patterns. We report PatternEval Acc, Trigger, the per-label failure rates for CoT, Rep, Con, and PR, the VG, OS, and KR task-family Trigger rates, and the matched-mode diferences $\Delta _ { \mathrm { a c c } }$ and $\Delta _ { \mathrm { p a t } }$

## 4.2 Main Results

Table 3 reports the PatternEval results across models under thinking and non-thinking inference. Based on these results, we draw the following key takeaways.

Takeaway 1: Response-pattern gaps persist across model families on PatternEval. Every evaluated pair has a positive response-pattern gap across open- and closed-source models, dense and mixture-of-experts architectures, and parameter scales. Although several frontier models substantially reduce the gap between inference modes, $\Delta _ { \mathrm { p a t } }$ exceeds 20 percentage points for 17 of the 25 pairs. This cross-model consistency shows that stronger task capability alone does not ensure response-pattern alignment under dificult conditions.

Takeaway 2: Non-thinking inference often exposes deliberation-style content. CoT leakage is the most prominent failure mode and constitutes the dominant source of bad-pattern triggers. Under our operational definition, non-thinking responses may contain process narration, redundant intermediate analysis, selfcorrection, or repeated reasoning fragments in the user-visible answer. The label characterizes observable response form independently of any claim about a model’s private internal computation.

Takeaway 3: Pattern failures are unevenly distributed. Taking the arithmetic mean over the 50 model–mode rows, CoT leakage and response repetition have marginal trigger rates of 12.75% and 8.49%, respectively, compared with 4.72% for performative reasoning and 2.74% for logical contradiction. This imbalance is more pronounced under non-thinking inference. Performative reasoning and logical contradiction are detected less often and may also co-occur with more salient failures such as CoT leakage; because the taxonomy uses a CoT-priority attribution rule, the marginal category rates should not be interpreted as independent prevalence estimates.

Table 3 Performance and response-pattern failures under thinking and non-thinking inference. All values are percentages; signed gaps are percentage points. Light-blue and light-orange cells denote the best and worst results, respectively, computed separately within each inference mode.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Mode</td><td colspan="2">Mode Gap</td><td colspan="2">Overall Results</td><td colspan="4">Pattern Failures</td><td colspan="3">Task-family Trigger</td></tr><tr><td>∆acc</td><td>∆pat</td><td>Acc↑</td><td>Trigger↓</td><td>CoT↓</td><td>Rep↓</td><td>Con↓</td><td>PR↓</td><td>VG↓</td><td>OS↓</td><td>KR↓</td></tr><tr><td>Qwen3.5-0.8B</td><td>Non-thk Thk</td><td>-2.09</td><td>6.95</td><td>25.22 23.13</td><td>75.94 68.99 46.42</td><td>32.22 13.50 28.24</td><td>38.92 29.94 25.55</td><td>26.87 34.49</td><td>36.81 47.83</td><td>84.32 75.08</td><td>50.34 53.62</td><td>83.51 71.60</td></tr><tr><td>Qwen3.5-4B</td><td>Non-thk Thk</td><td>6.37</td><td>24.23</td><td>45.96 52.33 50.15</td><td>22.19 36.19</td><td>11.18 25.22</td><td>10.60</td><td>7.62 3.11</td><td>10.60 6.50</td><td>50.68 25.51</td><td>19.31 8.45</td><td>62.75 28.40</td></tr><tr><td>Qwen3.5-9B</td><td>Non-thk Thk</td><td>9.05</td><td>21.36</td><td>59.20</td><td>14.83 29.32</td><td>8.41</td><td>19.05 6.30</td><td>5.13 1.45</td><td>5.01 3.44</td><td>40.17 16.95</td><td>13.45 4.31</td><td>49.16 20.34</td></tr><tr><td>Qwen3.5-27B</td><td>Non-thk Thk</td><td>4.98</td><td>24.01</td><td>59.07 64.05</td><td>5.31 33.00</td><td>23.44 2.94</td><td>15.65 1.78</td><td>0.91 0.17</td><td>2.48 1.16</td><td>30.93 5.51</td><td>7.41 1.21</td><td>45.80 8.58</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>Non-thk Thk</td><td>8.05</td><td>26.63</td><td>57.02 65.07</td><td>6.37 28.45</td><td>25.13 3.87</td><td>17.97 2.50</td><td>2.48 0.62</td><td>3.02 1.08</td><td>36.02 7.94</td><td>12.59 1.55</td><td>45.65 7.86</td></tr><tr><td>Qwen3.5-122B-A10B</td><td>Non-thk Thk</td><td>7.10</td><td>22.59</td><td>59.20 66.30</td><td>5.86 29.32</td><td>23.23 4.18 25.71</td><td>15.36 2.09</td><td>1.08 0.17</td><td>1.74 1.05</td><td>30.85 5.16</td><td>8.62 0.86</td><td>41.68 11.57</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>Non-thk Thk</td><td>6.97</td><td>25.09</td><td>61.96 68.93</td><td>4.23</td><td>2.98</td><td>15.65 1.12</td><td>0.66 0.17</td><td>1.12 0.83</td><td>30.51 4.75</td><td>9.48 0.69</td><td>44.73 6.42</td></tr><tr><td>Qwen3.6-27B</td><td>Non-thk Thk</td><td>6.39</td><td>26.88</td><td>57.33 63.72 56.22</td><td>32.81 5.93 32.80</td><td>26.14 3.69</td><td>18.31 2.28</td><td>1.66 0.33</td><td>2.57 1.08</td><td>35.54 6.80 37.12</td><td>10.69 1.55 10.34</td><td>47.48 8.24</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>Non-thk Thk</td><td>7.68</td><td>26.90</td><td>63.90</td><td>5.90 26.77</td><td>26.25 3.58</td><td>17.27 1.91</td><td>2.36 0.42</td><td>2.15 1.41</td><td>6.12</td><td>1.21</td><td>44.89 9.71</td></tr><tr><td>Qwen3.7-Max</td><td>Non-thk Thk Non-thk</td><td>5.23</td><td>21.21</td><td>67.07 72.30 66.28</td><td>5.56 27.82</td><td>22.28 3.98</td><td>16.79 2.70 17.50</td><td>0.46 0.17</td><td>0.58 0.75</td><td>29.69 6.54 29.24</td><td>13.26 2.25 13.99</td><td>33.33 6.73 37.52</td></tr><tr><td>Qwen3.7-Plus</td><td>Thk</td><td>4.69</td><td>21.23</td><td>70.97</td><td>6.59</td><td>23.13 4.85</td><td>3.90</td><td>0.58 0.21</td><td>0.66 0.70</td><td>7.63</td><td>1.73</td><td>9.04</td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>Non-thk Thk Non-thk</td><td>0.66</td><td>30.88</td><td>48.21 48.87 53.21</td><td>41.54 10.66 35.03</td><td>32.03 0.54 25.83</td><td>20.43 2.90 21.31</td><td>3.58 2.37 1.49</td><td>4.22 6.60 2.07</td><td>42.11 15.15 41.09</td><td>24.66 2.93 12.76</td><td>55.62 9.47 43.88</td></tr><tr><td></td><td>Thk Non-thk</td><td>2.54</td><td>30.21</td><td>55.75 61.33</td><td>4.82 44.43</td><td>0.35 42.07</td><td>1.53 11.59</td><td>1.10 0.37</td><td>2.63 0.91</td><td>6.27 44.24</td><td>1.38 24.14</td><td>5.53 62.75</td></tr><tr><td>Kimi-K2.5</td><td>Thk Non-thk</td><td>5.73</td><td>38.29 48.64</td><td>67.06 66.47</td><td>6.14 49.63</td><td>3.55 47.39</td><td>1.00 16.74</td><td>1.09 0.54</td><td>1.34 0.50</td><td>6.69 47.92 1.41</td><td>2.76 28.45</td><td>8.19 71.45</td></tr><tr><td></td><td>Thk Non-thk</td><td>6.72</td><td></td><td>73.19 59.49</td><td>0.99 29.63</td><td>0.13 24.28</td><td>0.13 20.77</td><td>0.27 0.71</td><td>0.49 3.34</td><td>41.38</td><td>0.34 10.52</td><td>0.85 25.31</td></tr><tr><td>Seed-1.6-Vision</td><td>Thk Non-thk</td><td>2.24</td><td>20.35</td><td>61.73 66.85</td><td>9.28 6.94 3.13</td><td>2.84 4.51</td><td>3.55 0.75</td><td>1.63 1.34</td><td>4.05 1.04</td><td>14.10 7.31 3.91</td><td>3.10 3.62 0.86</td><td>5.97 9.28 3.77</td></tr><tr><td>Seed-2.0-Pro Seed-2.1-Pro</td><td>Thk Non-thk</td><td>6.13 13.10</td><td>3.81 9.06</td><td>72.98 70.44</td><td>11.37 2.31</td><td>2.09 8.90 1.72</td><td>0.25 3.05 0.34</td><td>0.21 1.25 0.05</td><td>0.67 0.54 0.39</td><td>12.83 2.14</td><td>3.45 2.24</td><td>15.88 2.67</td></tr><tr><td>Claude-Opus-4.6</td><td>Thk Non-thk</td><td></td><td>29.59</td><td>83.54 59.45</td><td>33.18 3.59</td><td>30.35 1.45</td><td>8.79 0.23</td><td>1.23 0.56</td><td>1.27 1.49</td><td>39.48 4.83</td><td>5.37 1.22</td><td>47.12 3.84</td></tr><tr><td>Claude-Opus-4.8</td><td>Thk Non-thk Thk</td><td>8.18 6.58</td><td>18.51</td><td>67.63 61.12 67.70</td><td>22.06 3.55</td><td>18.93 1.93</td><td>5.11 0.48</td><td>1.18 0.44</td><td>0.89 1.01</td><td>29.01 4.24</td><td>4.68 1.56</td><td>25.16 4.18</td></tr><tr><td>Mimo-v2-Omni</td><td>Non-thk Thk</td><td>0.31</td><td>12.83</td><td>57.21 57.52</td><td>22.85 10.02</td><td>16.37 3.50</td><td>3.74 1.30</td><td>3.66 1.94</td><td>4.03 5.10</td><td>27.23 13.99</td><td>14.19 2.71</td><td>22.63 9.48</td></tr><tr><td>Mimo-v2.5</td><td>Non-thk Thk</td><td>0.42</td><td>13.88</td><td>55.03 55.45</td><td>24.92 11.04</td><td>16.47 3.41</td><td>3.87 2.34</td><td>5.03 2.13</td><td>4.83 6.52</td><td>34.22 15.70</td><td>8.65 2.29</td><td>22.63 10.47</td></tr><tr><td>GPT-5 Nano</td><td>Non-thk Thk</td><td>13.64</td><td>23.14</td><td>30.72 44.36</td><td>35.67 12.53</td><td>2.12 0.17</td><td>7.38 1.16</td><td>9.83 1.08</td><td>28.20 11.20</td><td>40.92 19.78</td><td>22.97 2.42</td><td>37.46 8.41</td></tr><tr><td>GPT-5.4</td><td>Non-thk Thk</td><td>8.88</td><td>2.86</td><td>60.66 69.54</td><td>5.34 2.48</td><td>1.20 0.00</td><td>0.91 0.62</td><td>0.79 0.04</td><td>2.82 1.82</td><td>5.68 3.56</td><td>1.21 0.86</td><td>8.40 1.98</td></tr><tr><td>GPT-5.5</td><td>Non-thk Thk</td><td>15.70</td><td>4.26</td><td>59.20 74.90</td><td>5.04 0.78</td><td>0.29 0.00</td><td>0.42 0.18</td><td>0.75 0.14</td><td>4.00 0.46</td><td>7.32 1.17</td><td>1.21 0.52</td><td>4.31 0.34</td></tr></table>

## 4.3 Further Analysis

We further investigate two factors that may shape the observed response-pattern gap: model capability and response length. Specifically, we examine whether stronger models narrow the discrepancy between thinking and non-thinking inference, and how response length interacts with inference mode and answer correctness in determining pattern-failure incidence.

Model capability. The Qwen3.5 series shows that higher PatternEval Acc within a model family is not accompanied by a smaller cross-mode Trigger gap. From 4B to 397B-A17B, PatternEval Acc increases from 45.96% to 61.96% under non-thinking inference and from 52.33% to 68.93% under thinking inference. Over the same range, thinking-mode Trigger falls from 22.19% to 4.23%. Non-thinking Trigger initially decreases from 46.42% at 4B to 29.32% at 27B, but then remains between 28.45% and 33.00% among the larger models. Consequently, the cross-mode Trigger gap stays between 21.36 and 26.63 percentage points from 4B onward. Within this series, scale, PatternEval Acc, and the cross-mode Trigger gap follow distinct descriptive trends.

![](images/6321978a70385ac0864123746ad0ce68d773e06f87ed1ad2cd433e8360ec7679.jpg)

![](images/cecb7626969f975cb5f51a57d15f651760a5cbac95f3026395e2601ccf5aa50e.jpg)  
Figure 3 Model capability and response-pattern failures across the Qwen3.5 series. Shaded annotations report the matched thinking–non-thinking gaps in percentage points.

Failure-prone constituent categories. The radar in Figure 4 provides a category-level view of non-thinking Trigger across the nine constituent task categories. Reasoning, STEM, and OOD perception form some of the largest peaks for multiple models, whereas content recognition and chart understanding generally exhibit lower trigger rates. This tendency is not uniform, however, and the relative ordering of categories varies substantially across models. In particular, models with similar aggregate Trigger can arrive at that average through diferent profiles: failures may be concentrated in a small number of highly susceptible categories or distributed more broadly across the benchmark. Conversely, models with diferent overall Trigger can still show comparable vulnerability on individual categories. These diferences suggest that a single aggregate score may obscure localized response-pattern weaknesses and motivate reporting category-level results alongside the overall metric.

![](images/7ada91ce17cec993e757645824c04ced4b3327950d46ad1bcd3c31eb51de5600.jpg)  
Figure 4 Constituent-category susceptibility under non-thinking. Each polygon reports one model’s Trigger across nine constituent task categories.

Response length, correctness, and Trigger. Figure 5 characterizes the relationship between response length and Trigger using model-by-correctness aggregates. Across the plotted

points, average response length is positively associated with Trigger under both inference settings, with Pearson correlations of $r = 0 . 6 4$ in non-thinking mode and $r = 0 . 8 4$ in thinking mode. Thus, aggregates containing longer responses also tend to exhibit more user-visible response-pattern failures, with this relationship appearing stronger under thinking inference. This diference is descriptive rather than causal: response length may reflect other factors, such as task dificulty, uncertainty, repetition, or extended but unsuccessful reasoning, rather than directly producing the observed failures. The connected correct–incorrect points provide a complementary within-model comparison. For the same model and inference mode, the incorrect-response aggregate is generally both longer and more failure-prone than the corresponding correct-response aggregate, suggesting that the overall trend is not driven solely by diferences in typical response length across models.

![](images/9be2e564cbaf27a00d4e8f0f67adf0e6b5bc3275b8b0d2cafe0afc4380e2fedc.jpg)  
Figure 5 Model-level response length versus Trigger. Connected markers pair correct and incorrect aggregates; dashed lines show mode-specific correlations.

Figure 6 complements this aggregate view by unfolding the response-level length distribution. After responses are divided into six length-based bins, Trigger increases monotonically or near-monotonically across all four mode–correctness groups, indicating that the model-level association is not driven solely by a few unusually verbose models. Within comparable length ranges, incorrect responses remain more failure-prone than correct responses, while non-thinking responses consistently exhibit higher Trigger than thinking responses. The separation is largest in the longest sextile, where Trigger reaches approximately 86% and 56% for incorrect and correct non-thinking responses, compared with 52% and 22% under thinking inference. The high rate among long but correct non-thinking responses is particularly notable, showing that answer correctness alone does not guarantee a well-aligned delivered response. Together, the two views identify long, incorrect non-thinking responses as the highest-observed Trigger regime, while indicating that response length, correctness, and inference mode retain distinct associations with pattern failures.

![](images/3277c84cbc49b26603c0794ddcf6a30c8f0aedf1ce0c1514b274867e32df41dc.jpg)  
Figure 6 Trigger across response-length sextiles. Line style denotes correctness and color denotes inference mode.

## 5 Pattern-Aware Post-Training

On PatternEval, all 25 model configurations have higher non-thinking Trigger point estimates. To target these failures during reinforcement learning, we train PatternRM, a reward model that approximates the pattern judge, and optimize the non-thinking policy using complementary signals from the task verifier and PatternRM. This objective preserves the primary correctness reward while directly penalizing the four operational PatternEval labels.

## 5.1 PatternRM Training

Training-data construction. We construct the PatternRM supervision corpus from a heterogeneous pool of 90K responses, comprising 60K responses drawn from four existing rollout collections and 30K re-rollouts generated by three additional policies. Each response is independently annotated by Kimi-K2.6, Seed-2.0-Pro, and Qwen3.5-397B with a four-dimensional binary label indicating the presence of CoT leakage, response repetition, logical contradiction, and performative reasoning. To improve annotation reliability, we retain only instances for which all three judges agree on the complete four-label vector, yielding 52,344 unique examples. We then oversample 5,234 selected instances to improve the balance of the training mixture, resulting in 57,578 SFT instances. Among them, 34,547 use thinking-format judgment targets and 23,031 use direct non-thinking targets. Additional details on data sources, sampling, filtering, and target formatting are provided in Section D.1.

Model training. We initialize PatternRM from Qwen3.5-27B and train it through supervised fine-tuning to predict the four response-pattern labels. Since PatternRM is repeatedly invoked to provide reward signals during reinforcement learning, inference eficiency is important. We therefore adopt a text-only configuration that evaluates the generated response without taking the associated image as input.

Performance evaluation. We evaluate PatternRM on the pattern-judge calibration set. As shown in Table 4, direct-prediction PatternRM attains the highest macro-F1 among the PatternRM variants (71.3% versus 70.2%) and avoids the additional decoding required by its thinking counterpart. This comparison reflects predictive performance and decoding cost rather than measured end-to-end latency. Since PatternRM repeatedly scores rollouts during reinforcement learning, we adopt the direct-prediction variant as the reward model for PatternRL.

Table 4 PatternRM judge performance comparison (%). Bold and underline denote the best and second-best results among non-seed judges, respectively.
<table><tr><td rowspan="2">Judge</td><td rowspan="2">Input</td><td colspan="3"> $\mathsf { c o T } .$ </td><td colspan="3">Rep.</td><td colspan="3">Con.</td><td colspan="3">PR.</td><td colspan="3">Overall</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td></td><td>R</td><td>F1</td><td>P</td><td>F1</td></tr><tr><td rowspan="2">Seed-2.0-Pro</td><td>Image+text</td><td>98.6</td><td>84.8</td><td>91.2</td><td>83.5</td><td>91.5</td><td>87.3</td><td>53.7</td><td>74.8</td><td>62.5</td><td>50.4</td><td>74.1</td><td>60.0</td><td>71.6</td><td>81.3</td><td>75.3</td></tr><tr><td>Text only</td><td>98.6</td><td>84.7</td><td>91.1</td><td>83.4</td><td>91.6</td><td>87.3</td><td>55.6</td><td>67.5</td><td>61.0</td><td>44.8</td><td>68.1</td><td>54.0</td><td>70.6</td><td>78.0</td><td>73.4</td></tr><tr><td rowspan="2">Qwen3.5-27B, thinking</td><td>Image+text</td><td>97.0</td><td>88.3</td><td>92.4</td><td>81.3</td><td>88.1</td><td>84.6</td><td>45.7</td><td>75.5</td><td>56.9</td><td>48.0</td><td>48.0</td><td>48.0</td><td>68.0</td><td>75.0</td><td>70.5</td></tr><tr><td>Text only</td><td>97.4</td><td>88.6</td><td>92.8</td><td>79.2</td><td>88.2</td><td>83.5</td><td>53.8</td><td>66.7</td><td>59.5</td><td>63.2</td><td>29.3</td><td>40.0</td><td>73.4</td><td>68.2</td><td>69.0</td></tr><tr><td rowspan="2">Qwen3.5-27B, direct</td><td>Image+text</td><td>97.8</td><td>86.4</td><td>91.7</td><td>75.8</td><td>79.6</td><td>77.7</td><td>33.9</td><td>48.7</td><td>40.0</td><td>29.4</td><td>72.6</td><td>41.8</td><td>59.2</td><td>71.8</td><td>62.8</td></tr><tr><td>Text only</td><td>98.0</td><td>87.6</td><td>92.5</td><td>57.8</td><td>92.3</td><td>71.1</td><td>36.7</td><td>62.4</td><td>46.2</td><td>30.2</td><td>82.3</td><td>44.2</td><td>55.7</td><td>81.2</td><td>63.5</td></tr><tr><td>PatternRM, thinking</td><td>Text only</td><td>96.8</td><td>88.2</td><td>92.1</td><td>89.5</td><td>78.1</td><td>83.4</td><td>55.0</td><td>61.5</td><td>58.1</td><td>59.0</td><td>40.5</td><td>48.0</td><td>75.1</td><td>67.1</td><td>70.2</td></tr><tr><td>PatternRM, direct</td><td>Text only</td><td>97.6</td><td>89.2</td><td>93.3</td><td>88.0</td><td>80.2</td><td>83.9</td><td>60.0</td><td>51.0</td><td>55.1</td><td>58.0</td><td>48.5</td><td>52.8</td><td>75.9</td><td>67.2</td><td>71.3</td></tr></table>

## 5.2 PatternRL

Reward design. The training reward consists of a verifier reward for answer correctness and an auxiliary pattern reward for response quality. The verifier extracts the final answer from the last \boxed{} expression and applies normalized answer matching and symbolic equivalence checking, while unresolved cases are evaluated by GPT-oss-120B, producing a binary score $s _ { \mathrm { v e r } } \in \{ 0 , 1 \}$ . For the four PatternRM predictions $\hat { b } _ { p } ^ { \mathrm { R M } }$ we use

$$
w _ { \mathrm { C o T } } = w _ { \mathrm { R e p } } = 0 . 0 5 , \qquad w _ { \mathrm { C o n } } = w _ { \mathrm { P R } } = 0 . 0 2 .\tag{6}
$$

To reduce evaluation cost, let $z \sim$ Bernoulli(0.6) denote whether PatternRM is invoked for a rollout. The auxiliary reward is

$$
s _ { \mathrm { p a t } } = - z \operatorname* { m i n } \left( 0 . 1 , \sum _ { p } w _ { p } \hat { b } _ { p } ^ { \mathrm { R M } } \right) \in [ - 0 . 1 , 0 ] .\tag{7}
$$

The final reward is

$$
r = \mathrm { c l i p } ( s _ { \mathrm { v e r } } + s _ { \mathrm { p a t } } , 0 , 1 ) .\tag{8}
$$

Therefore, incorrect responses always receive zero reward, whereas correct responses receive a score between 0.9 and 1, allowing the auxiliary reward to penalize undesirable response patterns without overriding the primary correctness objective.

Training details. We initialize the policies from the Qwen3-VL-4B-Instruct and Qwen3-VL-8B-Instruct models and train them with GRPO [34] on a multimodal RL mixture of 44,200 prompts, approximately balanced across multimodal math, multimodal logic, and document understanding. Most samples are drawn from OpenMMReasoner-RL-74K [49], excluding its virl39k subset, and are supplemented with data from WeMath, MMK12, ThinkLite-VL-Hard-11K, PuzzleVQA, AlgoPuzzleVQA, TextbookQA, ChartQA, and InfographicVQA [2, 9, 14, 27–29, 31, 42]. The 4B and 8B models are trained for 600 and 400 steps, respectively. We define BaseRL as the correctness-only GRPO baseline trained with the same setup and verifier reward but without $s _ { \mathrm { p a t } }$ . The reward configuration and available training parameters are provided in Sections D.2 and D.3; the complete 8B configuration is unavailable.

## 5.3 PatternRL Results

Table 5 PatternRL results on PatternEval. All values are percentages; gaps are percentage points. For each backbone, the thinking result is shown as a fixed reference for the non-thinking settings; thinking-mode results after reinforcement learning are not reported.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Setting</td><td rowspan="2">Gap to T ref.</td><td colspan="2">Overall Results</td><td colspan="4">Pattern Failures</td><td colspan="3">Task-family Trigger</td></tr><tr><td>Acc↑</td><td>Trigger↓</td><td>CoT↓</td><td>Rep↓</td><td>Con↓</td><td>PR↓</td><td>VG↓</td><td>OS↓</td><td>KR↓</td></tr><tr><td rowspan="4">Qwen3-VL-4B</td><td>Thinking</td><td></td><td>45.20</td><td>21.78</td><td>9.98</td><td>11.68</td><td>5.38</td><td>13.62</td><td>26.95</td><td>6.72</td><td>25.80</td></tr><tr><td>Non-thinking</td><td>22.44</td><td>45.71</td><td>44.22</td><td>25.80</td><td>33.00</td><td>12.96</td><td>20.12</td><td>49.26</td><td>20.35</td><td>56.18</td></tr><tr><td>+ BaseRL</td><td>29.77</td><td>48.05</td><td>51.55</td><td>33.83</td><td>43.77</td><td>13.83</td><td>20.99</td><td>56.37</td><td>27.59</td><td>63.97</td></tr><tr><td>+ PatternRL</td><td>16.69</td><td>47.30</td><td>38.47</td><td>24.51</td><td>18.84</td><td>11.01</td><td>15.61</td><td>42.31</td><td>17.41</td><td>50.08</td></tr><tr><td rowspan="4">Qwen3-VL-8B</td><td>Thinking</td><td></td><td>47.55</td><td>15.98</td><td>6.21</td><td>8.16</td><td>4.06</td><td>10.06</td><td>21.86</td><td>4.14</td><td>15.88</td></tr><tr><td>Non-thinking</td><td>20.79</td><td>49.37</td><td>36.77</td><td>23.44</td><td>25.34</td><td>9.65</td><td>15.90</td><td>41.13</td><td>15.52</td><td>47.63</td></tr><tr><td>+ BaseRL</td><td>28.35</td><td>50.57</td><td>44.33</td><td>31.50</td><td>36.27</td><td>9.91</td><td>17.25</td><td>49.46</td><td>17.41</td><td>56.95</td></tr><tr><td>+ PatternRL</td><td>14.00</td><td>50.71</td><td>29.98</td><td>17.23</td><td>15.24</td><td>10.85</td><td>12.05</td><td>35.53</td><td>10.69</td><td>36.95</td></tr></table>

Results on PatternEval. As shown in Table 5, correctness-only BaseRL improves task accuracy but aggravates non-thinking response-pattern failures on both Qwen3-VL backbones. In contrast, PatternRL substantially reduces the overall Trigger rate while preserving comparable accuracy, with improvements spanning most failure types and task families. These results show that explicitly optimizing response patterns can mitigate the degradation in quality introduced by correctness-only reinforcement learning without compromising task performance.

Table 6 Accuracy across ten reasoning and document-understanding benchmarks. Bold and underlined values denote the best and second-best results within each model scale, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="5">Math Reasoning</td><td colspan="3">Logic Reasoning</td><td colspan="5">Document Understanding</td><td rowspan="2">Overall Acc.</td></tr><tr><td>MathVista MathVerse</td><td></td><td>DynaMath</td><td>WeMath</td><td> ${ \mathsf { A v g } } .$ </td><td>LogicVista VisuLogic</td><td></td><td> ${ \mathsf { A } } { \mathsf { u g } } .$ </td><td>AI2D ChartQA DocVQA InfoVQA</td><td></td><td></td><td></td><td> ${ \mathsf { A } } { \mathsf { u g } } .$ </td></tr><tr><td colspan="10">Qwen3-VL-4B</td><td colspan="7"></td></tr><tr><td>Non-thinking</td><td>82.28 86.35</td><td>69.91 73.08</td><td>73.93 79.73</td><td>88.35 91.99</td><td>78.62</td><td>81.32</td><td>28.29</td><td>57.62</td><td>54.80 89.19 90.10</td><td>80.30</td><td>84.12 75.96</td><td></td><td>80.56</td><td>83.54</td><td>75.83</td></tr><tr><td>+ BaseRL + PatternRL</td><td></td><td>70.21</td><td></td><td></td><td>82.79</td><td>89.56</td><td>25.69</td><td>56.08</td><td></td><td>84.73 85.42</td><td></td><td></td><td>81.72 78.86</td><td>83.13 82.17</td><td>77.89 76.02</td></tr><tr><td></td><td>85.56</td><td></td><td>77.54</td><td>86.06</td><td>79.84</td><td>88.46</td><td>23.70</td><td></td><td>89.80</td><td></td><td></td><td>74.60</td><td></td><td></td><td></td></tr><tr><td colspan="10">Qwen3-VL-8B 77.71 85.71</td><td colspan="7"></td></tr><tr><td>Non-thinking + BaseRL</td><td>83.07</td><td>62.61 72.09</td><td>75.66</td><td></td><td>89.49</td><td></td><td>26.61</td><td></td><td>56.16 87.85</td><td></td><td>78.45</td><td>91.90 91.23</td><td>85.75</td><td>85.99</td><td>76.71</td></tr><tr><td>+ PatternRL</td><td>85.17 86.61</td><td></td><td>79.98</td><td>91.57</td><td>82.20</td><td>89.01</td><td>25.23</td><td>57.12</td><td>89.76 88.98</td><td>84.34 84.30</td><td></td><td></td><td>87.49</td><td>88.20</td><td>79.59</td></tr><tr><td></td><td></td><td>68.33</td><td>77.85</td><td>90.63</td><td>80.86</td><td>86.26</td><td>25.38</td><td>55.82</td><td></td><td></td><td></td><td>92.54</td><td>88.05</td><td>88.47</td><td>78.89</td></tr></table>

Accuracy on general reasoning tasks. As shown in Table 6, incorporating the response-pattern objective introduces an accuracy trade-of relative to correctness-only BaseRL. The efect is more pronounced for Qwen3-VL-4B, where PatternRL yields lower average accuracy across all three task categories. In contrast, the 8B model largely preserves its task performance, with only limited declines in math and logic reasoning and a slight improvement in document understanding. Overall, although the degradation is substantially smaller at the larger model scale, PatternRL still weakens aggregate task accuracy.

## 5.4 Discussion

Limitations of RL-stage pattern alignment. Although PatternRL consistently reduces response-pattern failures, it does not eliminate them entirely, as shown in Table 5. This residual gap suggests that undesirable response patterns cannot be fully corrected through a lightweight auxiliary reward applied only during reinforcement learning. Such behaviors may already be embedded in the model through noisy, inconsistent, or pattern-biased data encountered during earlier training stages. Consequently, more fundamental improvements may require curating higher-quality supervision and incorporating response-pattern constraints during midtraining or supervised fine-tuning, before these behaviors become reinforced by subsequent correctness-oriented optimization.

Capacity dependent correctness–pattern trade-off. The cost of introducing pattern-aware optimization also depends on model capacity. As shown in Table 6, the 4B model exhibits a substantially larger accuracy decline than the 8B model, particularly on math and other reasoning-intensive tasks. One plausible explanation is that PatternRM restricts part of the non-thinking policy’s efective rollout space: exploratory, verbose, or partially self-correcting trajectories may help a smaller model reach the correct answer, while also being more likel to trigger response-pattern penalties. Larger models can more readily produce concise and well-structured solutions without relying on these behaviors and therefore better tolerate the additional constraint. PatternRL thus introduces an inherent trade-of between the verifier signal, which rewards successful problem solving, and the PatternRM signal, which constrains how that solution is expressed; balancing the two objectives may need to be adjusted according to model capacity and task dificulty.

## 6 Conclusion

We study response-pattern alignment in hybrid-thinking MLLMs and introduce PatternEval to evaluate four user-visible failures across matched thinking and non-thinking interfaces. Our results reveal a consistent mode-dependent gap: non-thinking inference produces substantially more response-pattern failures, even in frontier models. To mitigate this issue, we develop PatternRM and PatternRL, which reduce non-thinking failures while largely preserving task accuracy. Overall, our findings show that controllable reasoning efort should not come at the cost of unstable response behavior, and that explicitly optimizing response patterns provides a practical path toward more reliable hybrid-thinking MLLMs.

## References

[1] Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete problems in ai safety. arXiv preprint arXiv:1606.06565, 2016.

[2] Yew Ken Chia, Vernon Toh, Deepanway Ghosal, Lidong Bing, and Soujanya Poria. Puzzlevqa: Diagnosing multimodal reasoning challenges of language models with abstract visual patterns. In Findings of the Association for Computational Linguistics: ACL 2024, pages 16259–16273. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.findings-acl.962. URL https://aclanthology.org/2024.findings-acl.962/.

[3] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[4] Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B. Hashimoto. Length-controlled alpacaeval: A simple way to debias automatic evaluators. In Conference on Language Modeling, 2024.

[5] Gongfan Fang, Xinyin Ma, and Xinchao Wang. Thinkless: Llm learns when to think. In Advances in Neural Information Processing Systems, volume 38, 2025.

[6] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, Rongrong Ji, Caifeng Shan, and Ran He. Mme: A comprehensive evaluation benchmark for multimodal large language models. In Advances in Neural Information Processing Systems, volume 38, 2025.

[7] Siyuan Gan, Jiaheng Liu, Boyan Wang, Tianpei Yang, Runqing Miao, Yuyao Zhang, Fanyu Meng, Junlan Feng, Linjian Meng, Jing Huo, and Yang Gao. Thinking-based non-thinking: Solving the reward hacking problem in training hybrid reasoning models via reinforcement learning. arXiv preprint arXiv:2601.04805, 2026. doi: 10.48550/arXiv.2601.04805. URL https://arxiv.org/abs/2601.04805.

[8] Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. arXiv preprint arXiv:2210.10760, 2022.

[9] Deepanway Ghosal, Vernon Toh, Yew Ken Chia, and Soujanya Poria. Algopuzzlevqa: Diagnosing multimodal reasoning challenges of language models with algorithmic multimodal puzzles. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 9615–9632. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.naacl-long.486. URL https://aclanthology.org/2025.naacl-long.486/.

[10] GLM-4.5 Team. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471, 2025.

[11] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. arXiv preprint arXiv:2310.14566, 2023.

[12] Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, et al. Skywork open reasoner 1 technical report. arXiv preprint arXiv:2505.22312, 2025.

[13] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id= rygGQyrFvH.

[14] Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 4999–5007, July 2017. URL https://openaccess.thecvf.com/content\_cvpr\_2017/html/Kembhavi\_Are\_You\_ Smarter\_CVPR\_2017\_paper.html.

[15] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in Neural Information Processing Systems, 35:22199–22213, 2022.

[16] Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, Noah A. Smith, and Hannaneh Hajishirzi. Rewardbench: Evaluating reward models for language modeling. In Findings of the Association for Computational Linguistics: NAACL

2025, pages 1755–1797, 2025. doi: 10.18653/v1/2025.findings-naacl.96. URL https://aclanthology.org/2025. findings-naacl.96/.

[17] Lei Li, Zhihui Xie, Mukai Li, Shunian Chen, Peiyi Wang, Liang Chen, Yazheng Yang, Benyou Wang, Lingpeng Kong, and Qi Liu. Vlfeedback: A large-scale ai feedback dataset for large vision-language models alignment. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 6227–6246, 2024. doi: 10.18653/v1/2024.emnlp-main.358. URL https://aclanthology.org/2024.emnlp-main.358/.

[18] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023.

[19] Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110, 2022.

[20] Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. arXiv preprint arXiv:2109.07958, 2021.

[21] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-eval: Nlg evaluation using gpt-4 with better human alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522, 2023. doi: 10.18653/v1/2023.emnlp-main.153. URL https: //aclanthology.org/2023.emnlp-main.153/.

[22] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023.

[23] Feng Luo, Yu-Neng Chuang, Guanchu Wang, Hoang Anh Duy Le, Shaochen Zhong, Hongyi Liu, Jiayi Yuan, Yang Sui, Vladimir Braverman, Vipin Chaudhary, and Xia Hu. Autol2s: Auto long-short reasoning for eficient large language models. In Findings of the Association for Computational Linguistics: ACL 2026, pages 16836–16858, 2026. doi: 10.18653/v1/2026.findings-acl.831. URL https://aclanthology.org/2026.findings-acl.831/.

[24] Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, and Matei Zaharia. Reasoning models can be efective without thinking. arXiv preprint arXiv:2504.09858, 2025.

[25] Saumya Malik, Valentina Pyatkin, Sander Land, Jacob Morrison, Noah A. Smith, Hannaneh Hajishirzi, and Nathan Lambert. Rewardbench 2: Advancing reward model evaluation. arXiv preprint arXiv:2506.01937, 2025.

[26] Potsawee Manakul, Adian Liusie, and Mark J. F. Gales. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9004–9017, 2023. doi: 10.18653/v1/2023.emnlp-main.557. URL https: //aclanthology.org/2023.emnlp-main.557/.

[27] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2263–2279. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022.findings-acl.177. URL https://aclanthology.org/2022.findings-acl.177/.

[28] Minesh Mathew, Viraj Bagal, Rubén Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V. Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706, 2022. URL https://openaccess.thecvf.com/content/WACV2022/html/Mathew\_InfographicVQA\_ WACV\_2022\_paper.html.

[29] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, Ping Luo, Yu Qiao, Qiaosheng Zhang, and Wenqi Shao. Mmeureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025. URL https://arxiv.org/abs/2503.07365.

[30] Arjun Panickssery, Samuel R. Bowman, and Shi Feng. Llm evaluators recognize and favor their own generations. arXiv preprint arXiv:2404.13076, 2024.

[31] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Jiapeng Wang, Zhuoma GongQue, Shanglin Lei, Yifan Zhang, Zhe Wei, Miaoxuan Zhang, Runfeng Qiao, Xiao Zong, Yida Xu, Peiqing Yang, Zhimin Bao, Muxi Diao, Chen Li, and Honggang Zhang. We-math: Does your large multimodal mode

achieve human-like mathematical reasoning? In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 20023–20070, Vienna, Austria, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.acl-long.983. URL https://aclanthology.org/2025. acl-long.983/.

[32] Qwen Team. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[33] Paul Röttger, Fabio Pernisi, Bertie Vidgen, and Dirk Hovy. Safetyprompts: A systematic review of open datasets for evaluating and improving large language model safety. arXiv preprint arXiv:2404.05399, 2024.

[34] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[35] Joar Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward hacking. arXiv preprint arXiv:2209.13085, 2022.

[36] Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. arXiv preprint arXiv:2305.04388, 2023.

[37] Jiaqi Wang, Kevin Qinghong Lin, James Cheng, and Mike Zheng Shou. Think or not? selective reasoning via reinforcement learning for vision-language models. In Advances in Neural Information Processing Systems, volume 38, 2025.

[38] Junyang Wang, Yuhang Wang, Guohai Xu, Jing Zhang, Yukai Gu, Haitao Jia, Jiaqi Wang, Haiyang Xu, Ming Yan, Ji Zhang, and Jitao Sang. Amber: An llm-free multi-dimensional benchmark for mllms hallucination evaluation. arXiv preprint arXiv:2311.07397, 2023.

[39] Qian Wang, Zhanzhi Lou, Zhenheng Tang, Nuo Chen, Xuandong Zhao, Wenxuan Zhang, Dawn Song, and Bing sheng He. Assessing judging bias in large reasoning models: An empirical study. arXiv preprint arXiv:2504.09946, 2025.

[40] Shouren Wang, Wang Yang, Xianxuan Long, Qifan Wang, Vipin Chaudhary, and Xiaotian Han. Demystifying hybrid thinking: Can llms truly switch between think and no-think? arXiv preprint arXiv:2510.12680, 2025.

[41] Shouren Wang, Wang Yang, Chuang Ma, Debargha Ganguly, Vikash Singh, Chaoda Song, Xinpeng Li, Xianxuan Long, Vipin Chaudhary, and Xiaotian Han. Path-lock expert: Separating reasoning mode in hybrid thinking via architecture-level separation. arXiv preprint arXiv:2604.27201, 2026.

[42] Xiyao Wang, Zhengyuan Yang, Chao Feng, Hongjin Lu, Linjie Li, Chung-Ching Lin, Kevin Lin, Furong Huang, and Lijuan Wang. Sota with less: Mcts-guided sample selection for data-eficient visual reasoning self-improvement. In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://papers.nips.cc/paper\_ files/paper/2025/hash/ac3cea0be817ebac21299b77fd114ddf-Abstract-Conference.html.

[43] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. International Conference on Learning Representations, 2023.

[44] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022.

[45] Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, and Jason Weston. Neural text generation with unlikelihood training. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=SJeYe0NtvH.

[46] Hao Wen, Xinrui Wu, Yi Sun, Feifei Zhang, Liye Chen, Jie Wang, Yunxin Liu, Yunhao Liu, Ya-Qin Zhang, and Yuanchun Li. Budgetthinker: Empowering budget-aware llm reasoning with control tokens. arXiv preprint arXiv:2508.17196, 2025.

[47] Michihiro Yasunaga, Luke Zettlemoyer, and Marjan Ghazvininejad. Multimodal rewardbench: Holistic evaluation of reward models for vision language models. arXiv preprint arXiv:2502.14191, 2025.

[48] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502, 2023.

[49] Kaichen Zhang, Keming Wu, Zuhao Yang, Bo Li, Kairui Hu, Bin Wang, Xingxuan Li, and Lidong Bing. Openmmreasoner: Pushing the frontiers in multimodal reasoning with an open and general recipe. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19276–19286, 2026. URL https://openaccess.thecvf.com/content/CVPR2026/html/Zhang\_OpenMMReasoner\_Pushing\_the\_ Frontiers\_in\_Multimodal\_Reasoning\_with\_an\_Open\_CVPR\_2026\_paper.html.

[50] Ruiqi Zhang, Changyi Xiao, and Yixin Cao. Long or short cot? investigating instance-level switch of large reasoning models. arXiv preprint arXiv:2506.04182, 2025.

[51] Yichi Zhang, Yao Huang, Yifan Wang, Yitong Sun, Chang Liu, Zhe Zhao, Zhengwei Fang, Huanran Chen, Xiao Yang, Xingxing Wei, Hang Su, Yinpeng Dong, and Jun Zhu. Unveiling trust in multimodal large language models: Evaluation, analysis, and mitigation. arXiv preprint arXiv:2508.15370, 2025.

[52] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36, 2023.

[53] Jefrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

## A Appendix Roadmap

The appendix is organized as follows. Section B provides the constituent-category PatternEval results for all 25 model configurations, and Section C reports the calibration analysis used to select the operational pattern judge. Section D documents the PatternRM supervision corpus and the available PatternRL training configuration. Finally, Section E reproduces the complete operational pattern-judge prompt in English and in the original Chinese. This roadmap is intended to make the supplementary material navigable without introducing a separate table of contents.

## B Additional Results

## B.1 Category-wise Results

As shown in Table 7, positive cross-mode Trigger gaps occur across multiple constituent task categories rather than only one task family. Some of the largest descriptive diferences appear in OOD perception and knowledge-intensive reasoning, whereas content recognition and chart understanding often show smaller gaps. These category-level point estimates characterize PatternEval and do not by themselves identify the cause of a mode diference.

Table 7 also shows that scale does not uniformly reduce every constituent-category Trigger value. Within several model families, thinking-mode Trigger declines with scale, while non-thinking values remain elevated or vary non-monotonically in categories such as OOD perception, STEM, and general reasoning. This descriptive pattern motivates category-specific analysis rather than a single aggregate interpretation.

Table 7 Constituent-category Trigger under thinking and non-thinking inference. Values are percentages; lower is better. Categories are grouped into the three task families of PatternEval (Table 1).
<table><tr><td rowspan=1 colspan=14>VG                  os                KRModel               ModeOOD↓Rec.↓Grd.↓Fact.↓OCR↓Chart↓STEM↓Know.↓Reas.↓</td></tr><tr><td rowspan=2 colspan=14>Non-thk 98.10 81.1463.20 69.09 58.33  41.79  94.50   79.46  78.15Qwen3.5-0.8BThk  86.86 74.8652.00 69.09 58.33  48.57  91.50  74.05  55.19</td></tr><tr><td rowspan=1 colspan=1>86.86</td><td rowspan=1 colspan=2>74.86 52.00</td><td rowspan=1 colspan=1>69.09</td><td rowspan=1 colspan=5></td><td rowspan=1 colspan=1>55.19</td></tr><tr><td rowspan=2 colspan=4>Non-thkQwen3.5-4BThk</td><td rowspan=1 colspan=1>83.05</td><td rowspan=1 colspan=2>28.8621.20</td><td rowspan=1 colspan=1>14.55</td><td rowspan=1 colspan=5>17.00 21.79  77.50  43.78</td><td rowspan=1 colspan=1>64.81</td></tr><tr><td rowspan=1 colspan=1>44.38</td><td rowspan=1 colspan=2>16.004.80</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=5>6.33  10.71  25.50  25.41</td><td rowspan=1 colspan=1>32.59</td></tr><tr><td rowspan=2 colspan=4>Non-thkQwen3.5-9BThk</td><td rowspan=1 colspan=1>72.76</td><td rowspan=1 colspan=2>12.8614.00</td><td rowspan=1 colspan=1>21.82</td><td rowspan=1 colspan=5>16.67  10.00  59.50  32.43</td><td rowspan=1 colspan=1>52.96</td></tr><tr><td rowspan=1 colspan=1>27.81</td><td rowspan=1 colspan=2>11.144.80</td><td rowspan=1 colspan=1>5.45</td><td rowspan=1 colspan=5>2.67  6.07   21.50   13.04</td><td rowspan=1 colspan=1>24.44</td></tr><tr><td rowspan=2 colspan=4>Non-thkQwen3.5-27BThk</td><td rowspan=1 colspan=1>54.29</td><td rowspan=1 colspan=2>6.86 18.80</td><td rowspan=1 colspan=1>16.36</td><td rowspan=1 colspan=1>10.00</td><td rowspan=1 colspan=4>4.64  65.50  16.22</td><td rowspan=1 colspan=1>51.48</td></tr><tr><td rowspan=1 colspan=1>9.14</td><td rowspan=1 colspan=2>2.29 2.80</td><td rowspan=1 colspan=1>3.64</td><td rowspan=1 colspan=1>1.67</td><td rowspan=1 colspan=4>0.72   8.50   1.63</td><td rowspan=1 colspan=1>13.38</td></tr><tr><td rowspan=2 colspan=4>Non-thkQwen3.5-35B-A3BThk</td><td rowspan=1 colspan=1>65.14</td><td rowspan=1 colspan=1>6.86</td><td rowspan=1 colspan=1>20.00</td><td rowspan=1 colspan=1>16.36</td><td rowspan=1 colspan=1>13.67</td><td rowspan=1 colspan=4>11.43  60.00  17.84</td><td rowspan=1 colspan=1>54.07</td></tr><tr><td rowspan=1 colspan=1>15.25</td><td rowspan=1 colspan=1>2.29</td><td rowspan=1 colspan=1>2.00</td><td rowspan=1 colspan=1>1.82</td><td rowspan=1 colspan=1>1.67</td><td rowspan=1 colspan=1>1.43</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>8.50   3.78</td><td rowspan=1 colspan=1>10.23</td></tr><tr><td rowspan=2 colspan=4>Non-thkQwen3.5-122B-A10BThk</td><td rowspan=1 colspan=1>55.62</td><td rowspan=1 colspan=1>5.43</td><td rowspan=1 colspan=1>16.40</td><td rowspan=1 colspan=1>21.82</td><td rowspan=1 colspan=1>10.00</td><td rowspan=1 colspan=1>7.14</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>58.50  15.14</td><td rowspan=1 colspan=1>47.41</td></tr><tr><td rowspan=1 colspan=2>D-1Y10DThk</td><td rowspan=1 colspan=1>9.63</td><td rowspan=1 colspan=1>1.43</td><td rowspan=1 colspan=1>2.00</td><td rowspan=1 colspan=1>1.82</td><td rowspan=1 colspan=2>0.67  1.07</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>11.50   4.35</td><td rowspan=1 colspan=1>16.67</td></tr><tr><td rowspan=2 colspan=2>Qwen3.5-397B-A17B</td><td rowspan=1 colspan=2>Non-thk</td><td rowspan=1 colspan=1>58.67</td><td rowspan=1 colspan=1>2.00</td><td rowspan=1 colspan=1>15.60</td><td rowspan=1 colspan=1>10.91</td><td rowspan=1 colspan=5>11.67  7.14  61.50   9.19</td><td rowspan=1 colspan=1>56.67</td></tr><tr><td rowspan=1 colspan=3>Thk</td><td rowspan=1 colspan=1>8.57</td><td rowspan=1 colspan=1>1.43</td><td rowspan=1 colspan=1>2.40</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=5>1.00  0.36   4.00   2.70</td><td rowspan=1 colspan=1>10.78</td></tr><tr><td rowspan=2 colspan=3>Qwen3.6-27B</td><td rowspan=1 colspan=2>Non-thk</td><td rowspan=1 colspan=1>65.14</td><td rowspan=1 colspan=1>8.31</td><td rowspan=1 colspan=1>16.00</td><td rowspan=1 colspan=1>14.55</td><td rowspan=1 colspan=2>13.33</td><td rowspan=1 colspan=3>7.86   63.00  16.76</td></tr><tr><td rowspan=1 colspan=1>Thk</td><td rowspan=1 colspan=1>12.26</td><td rowspan=1 colspan=1>2.00</td><td rowspan=1 colspan=1>2.80</td><td rowspan=1 colspan=1>3.64</td><td rowspan=1 colspan=1>1.67</td><td rowspan=1 colspan=4>1.43   8.00   3.24</td><td rowspan=1 colspan=1>11.85</td></tr><tr><td rowspan=2 colspan=3>Qwen3.6-35B-A3B</td><td rowspan=1 colspan=1>Non-thk</td><td rowspan=1 colspan=1>69.52</td><td rowspan=1 colspan=1>5.43</td><td rowspan=1 colspan=1>17.20</td><td rowspan=1 colspan=1>20.00</td><td rowspan=1 colspan=1>12.00</td><td rowspan=1 colspan=4>8.57  59.00  14.05</td><td rowspan=1 colspan=1>55.56</td></tr><tr><td rowspan=1 colspan=1>Thk</td><td rowspan=1 colspan=1>10.56</td><td rowspan=1 colspan=1>2.29</td><td rowspan=1 colspan=1>3.20</td><td rowspan=1 colspan=1>1.82</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=2>1.79</td><td rowspan=1 colspan=2>10.50   5.41</td><td rowspan=1 colspan=1>12.12</td></tr><tr><td rowspan=2 colspan=4>Non-thkQwen3.7-MaxThk</td><td rowspan=1 colspan=1>50.29</td><td rowspan=1 colspan=1>6.59</td><td rowspan=1 colspan=1>18.80</td><td rowspan=1 colspan=1>29.09</td><td rowspan=1 colspan=1>13.33</td><td rowspan=1 colspan=2>13.19</td><td rowspan=1 colspan=2>29.50   8.15</td><td rowspan=1 colspan=1>53.33</td></tr><tr><td rowspan=1 colspan=1>9.94</td><td rowspan=1 colspan=1>1.43</td><td rowspan=1 colspan=1>7.20</td><td rowspan=1 colspan=1>3.64</td><td rowspan=1 colspan=1>3.00</td><td rowspan=1 colspan=2>1.43</td><td rowspan=1 colspan=2>6.50   4.35</td><td rowspan=1 colspan=1>8.52</td></tr><tr><td rowspan=2 colspan=4>Non-thkQwen3.7-PlusThk</td><td rowspan=1 colspan=1>52.19</td><td rowspan=1 colspan=1>3.71</td><td rowspan=1 colspan=1>15.60</td><td rowspan=1 colspan=1>34.55</td><td rowspan=1 colspan=1>14.33</td><td rowspan=1 colspan=2>13.62</td><td rowspan=1 colspan=2>31.50  15.22</td><td rowspan=1 colspan=1>57.25</td></tr><tr><td rowspan=1 colspan=1>14.48</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>3.60</td><td rowspan=1 colspan=1>3.64</td><td rowspan=1 colspan=1>1.33</td><td rowspan=1 colspan=2>2.15</td><td rowspan=1 colspan=2>10.50   3.26</td><td rowspan=1 colspan=1>11.90</td></tr><tr><td rowspan=2 colspan=4>Non-thkQwen3-VL-30B-A3BThk</td><td rowspan=1 colspan=1>69.20</td><td rowspan=1 colspan=1>12.29</td><td rowspan=1 colspan=1>31.60</td><td rowspan=1 colspan=1>52.73</td><td rowspan=1 colspan=1>30.33</td><td rowspan=1 colspan=2>18.57</td><td rowspan=1 colspan=1>76.65</td><td rowspan=1 colspan=1>25.95</td><td rowspan=1 colspan=1>60.67</td></tr><tr><td rowspan=1 colspan=1>Thk</td><td rowspan=1 colspan=1>30.19</td><td rowspan=1 colspan=1>4.00</td><td rowspan=1 colspan=1>1.60</td><td rowspan=1 colspan=1>5.45</td><td rowspan=1 colspan=1>4.67</td><td rowspan=1 colspan=2>1.07</td><td rowspan=1 colspan=1>9.00</td><td rowspan=1 colspan=1>10.81</td><td rowspan=1 colspan=1>8.89</td></tr><tr><td rowspan=2 colspan=3>Qwen3-VL-32B</td><td rowspan=1 colspan=1>Non-thk</td><td rowspan=1 colspan=1>70.75</td><td rowspan=1 colspan=1>10.57</td><td rowspan=1 colspan=1>24.40</td><td rowspan=1 colspan=1>29.09</td><td rowspan=1 colspan=1>15.00</td><td rowspan=1 colspan=2>10.36</td><td rowspan=1 colspan=1>59.00</td><td rowspan=1 colspan=1>16.85</td><td rowspan=1 colspan=1>51.11</td></tr><tr><td rowspan=1 colspan=1>Thk</td><td rowspan=1 colspan=1>12.83</td><td rowspan=1 colspan=1>2.86</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>7.27</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=2>2.14</td><td rowspan=1 colspan=1>3.89</td><td rowspan=1 colspan=1>8.11</td><td rowspan=1 colspan=1>4.85</td></tr><tr><td rowspan=2 colspan=3>Kimi-K2.5</td><td rowspan=1 colspan=1>Non-thk</td><td rowspan=1 colspan=1>82.86</td><td rowspan=1 colspan=1>2.86</td><td rowspan=1 colspan=1>23.20</td><td rowspan=1 colspan=1>34.55</td><td rowspan=1 colspan=1>35.67</td><td rowspan=1 colspan=3>11.79  82.00</td><td rowspan=1 colspan=1>27.03</td><td rowspan=1 colspan=1>72.96</td></tr><tr><td rowspan=1 colspan=1>Thk</td><td rowspan=1 colspan=1>11.89</td><td rowspan=1 colspan=1>1.72</td><td rowspan=1 colspan=1>4.00</td><td rowspan=1 colspan=1>1.85</td><td rowspan=1 colspan=1>4.00</td><td rowspan=1 colspan=3>1.43   4.10</td><td rowspan=1 colspan=1>1.63</td><td rowspan=1 colspan=1>15.67</td></tr><tr><td rowspan=2 colspan=3>Kimi-K2.6</td><td rowspan=1 colspan=1>Non-thk</td><td rowspan=1 colspan=1>83.05</td><td rowspan=1 colspan=1>3.15</td><td rowspan=1 colspan=1>30.80</td><td rowspan=1 colspan=1>74.55</td><td rowspan=1 colspan=1>37.67</td><td rowspan=1 colspan=3>18.57  85.50</td><td rowspan=1 colspan=1>32.43</td><td rowspan=1 colspan=1>87.78</td></tr><tr><td rowspan=1 colspan=1>Thk</td><td rowspan=1 colspan=1>2.44</td><td rowspan=1 colspan=1>0.29</td><td rowspan=1 colspan=1>1.20</td><td rowspan=1 colspan=1>1.82</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=5>0.00   0.00   1.08   1.47</td></tr><tr><td rowspan=2 colspan=3>Seed-1.6-Vision</td><td rowspan=1 colspan=1>Non-thk</td><td rowspan=1 colspan=1>76.57</td><td rowspan=1 colspan=1>6.63</td><td rowspan=1 colspan=1>20.00</td><td rowspan=1 colspan=1>21.82</td><td rowspan=1 colspan=1>11.67</td><td rowspan=1 colspan=5>9.29   33.50  15.76  25.79</td></tr><tr><td rowspan=1 colspan=1>Thk</td><td rowspan=1 colspan=1>21.33</td><td rowspan=1 colspan=1>6.34</td><td rowspan=1 colspan=1>10.80</td><td rowspan=1 colspan=1>9.09</td><td rowspan=1 colspan=1>2.67</td><td rowspan=1 colspan=5>3.57   7.50   7.07   3.97</td></tr><tr><td rowspan=2 colspan=3>Seed-2.0-Pro</td><td rowspan=1 colspan=1>Non-thk</td><td rowspan=1 colspan=1>10.86</td><td rowspan=1 colspan=1>1.15</td><td rowspan=1 colspan=1>8.80</td><td rowspan=1 colspan=1>5.45</td><td rowspan=1 colspan=1>2.67</td><td rowspan=1 colspan=5>4.64  10.50   8.15   9.13</td></tr><tr><td rowspan=1 colspan=1>Thk</td><td rowspan=1 colspan=1>3.62</td><td rowspan=1 colspan=1>1.73</td><td rowspan=1 colspan=1>8.40</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>1.67</td><td rowspan=1 colspan=5>0.00   5.50   1.63   3.97</td></tr><tr><td rowspan=2 colspan=4>Seed-2.1-Pro        Non-thkThk</td><td rowspan=1 colspan=1>19.62</td><td rowspan=1 colspan=1>3.17</td><td rowspan=1 colspan=1>13.60</td><td rowspan=1 colspan=1>5.45</td><td rowspan=1 colspan=1>2.67</td><td rowspan=1 colspan=5>4.29  19.50  10.33  17.06</td></tr><tr><td rowspan=1 colspan=1>1.39</td><td rowspan=1 colspan=2>0.29 5.65</td><td rowspan=1 colspan=1>1.89</td><td rowspan=1 colspan=6>3.33  1.07   4.52   1.66   1.80</td></tr><tr><td rowspan=2 colspan=4>Non-thkClaude-Opus-4.6Thk</td><td rowspan=1 colspan=1>75.92</td><td rowspan=1 colspan=2>6.12 15.32</td><td rowspan=1 colspan=1>12.73</td><td rowspan=1 colspan=6>7.41  3.21   59.80  11.24  62.75</td></tr><tr><td rowspan=1 colspan=1>8.62</td><td rowspan=1 colspan=2>1.17 2.82</td><td rowspan=1 colspan=1>12.73</td><td rowspan=1 colspan=6>1.35  1.07   0.56   1.70   8.26</td></tr><tr><td rowspan=2 colspan=4>Claude-Opus-4.8    Non-thkThk</td><td rowspan=1 colspan=3>57.61 2.04 10.48</td><td rowspan=1 colspan=1>10.91</td><td rowspan=1 colspan=6>7.41   1.79   30.65   3.37  36.44</td></tr><tr><td rowspan=1 colspan=3>7.74  0.87 2.42</td><td rowspan=1 colspan=7>5.45  0.67  2.50   1.52   0.56   8.91</td></tr><tr><td rowspan=2 colspan=7>Non-thk 39.85 15.1916.06Mimo-v2-OmniThk  24.04 2.22 8.29</td><td rowspan=1 colspan=7>34.55 9.67  19.06  24.00   9.19  30.86</td></tr><tr><td rowspan=1 colspan=7>9.09  3.33  1.98   7.00   4.32  14.87</td></tr><tr><td rowspan=1 colspan=14>Non-thk 44.38 20.6330.45Mimo-v2.5Thk  30.10 3.15 3.79  3.64  2.67  1.87   3.50   3.91  20.31</td></tr><tr><td rowspan=2 colspan=4>GPT-5 Nano        Non-thkThk</td><td rowspan=1 colspan=10>77.90 12.6410.00 7.27  30.77  14.64  55.00  25.41  32.71</td></tr><tr><td rowspan=1 colspan=10>42.29 0.86 2.00  5.45  2.01  2.86   7.00   5.95  11.15</td></tr><tr><td rowspan=2 colspan=4>Non-thkGPT-5.4Thk</td><td rowspan=1 colspan=10>9.71  3.43 0.80  3.64  1.67  0.71   12.00   6.49   7.04</td></tr><tr><td rowspan=1 colspan=10>4.95  3.71 0.80  1.82  1.33  0.36   3.00   1.08   1.85</td></tr><tr><td rowspan=1 colspan=14>GPT-5.5            Non-thkThk   1.61  1.16 0.40  1.82  1.01  0.00   0.51   0.55   0.00</td></tr></table>

## C Meta-Judge Analysis

## C.1 Calibration Protocol

The calibration pipeline separates candidate discovery from benchmark scoring. During candidate discovery, Seed-2.0-Pro, Kimi-K2.6, and Qwen3.5-397B independently annotate responses along the four PatternEval failure labels. Complete three-judge agreement defines the consensus stratum, while any disagreement defines the disputed stratum; the two strata are sampled at an approximate 7:3 ratio, with the predicted positive rate controlled near 70%. The selected 2,500 fixed responses then receive human reference annotations under the same rubric. After reference labeling, we compare five candidate judge systems—with image and text-only variants where available—on the same responses using per-label precision, recall, and F1. This fixed-response design supports controlled judge comparison, while the reference set remains separate from the 25-configuration benchmark evaluation.

## C.2 Failure-Label Calibration Behavior

The results in Table 2 show a consistent dificulty gap across the nine judge–input configurations. CoT leakage F1 ranges from 90.2% to 95.3%, and repetition F1 ranges from 81.6% to 88.2%. In contrast, contradiction F1 ranges from 56.1% to 62.5%, and performative reasoning F1 ranges from 41.8% to 64.5%. Image access improves the reported aggregate F1 for Seed-2.0-Pro and Kimi-K2.6, while GPT-5.5 achieves the highest aggregate F1 overall. Seed-2.0-Pro with image access attains the highest contradiction F1 (62.5%) and directly inspects visual evidence; we therefore use it as the operational pattern judge.

## D PatternRL Training

## D.1 Supervision Corpus

PatternRM is initialized from Qwen3.5-27B. Its initial corpus draws 15K examples from each of four completed rollout collections. Within each collection, trajectories are partitioned into three step-count buckets and sampled at a 2:3:5 ratio with question-level deduplication. Quality filtering excludes the NED-only subset, retains half of the overlong cases, removes responses shorter than five characters, and removes entirely Chinese responses to English prompts. Prompts associated with filtered responses are then re-rolled out with Qwen3-VL-32B-Instruct, Qwen3.5-4B-Think, and Kimi-K2.6-NoThink, contributing 10K examples per policy. The resulting pool contains 90K responses.

Kimi-K2.6, Seed-2.0-Pro, and Qwen3.5-397B independently assign the four PatternEval labels, and only unanimous four-label vectors are retained. For the retained targets, thinking-format examples use Kimi-K2.6’s thinking content as the rationale field, whereas non-thinking-format examples use the reasoning in its answer field. Consensus filtering yields 52,344 unique examples. Reusing 5,234 instances produces 57,578 SFT instances in total, comprising 34,547 thinking-format and 23,031 non-thinking-format targets. The incomplete A20B PatternRM-fusion trajectory collection is excluded from these counts.

## D.2 Reward Configuration

PatternRL converts the four PatternRM decisions into the category-weighted auxiliary reward defined in Equations (6) and (7). Logical contradiction and performative reasoning each carry weight 0.02, while CoT leakage and repetition each carry weight 0.05. PatternRM is invoked independently with probability 0.6 for each rollout; otherwise its auxiliary contribution is zero. Simultaneous penalties accumulate until the PatternRM subreward reaches its floor of −0.1. The subreward is added to the outcome reward and the fused scalar is clipped to [0, 1] as specified in Equation (8); consequently, −0.1 is the auxiliary-reward floor, whereas zero is the final fused-reward floor. The configuration reported below corresponds to a single run; multi-seed training manifests are unavailable.

## D.3 Training Parameters

The training configuration used for the reported PatternRL run is summarized in Table 8.

Table 8 PatternRL training configuration. Summary of the optimization hyperparameters, rollout settings, sequencelength budgets, reward configuration, and distributed training setup used for PatternRL.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Optimization algorithm</td><td>GRPO</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Clip ratio</td><td>0.2</td></tr><tr><td>KL coefficient</td><td>0</td></tr><tr><td>Entropy coefficient</td><td>0</td></tr><tr><td>Training batch size</td><td>128</td></tr><tr><td>PPO mini-batch size</td><td>128</td></tr><tr><td>Micro-batch size per GPU 1</td><td></td></tr><tr><td>Max prompt length</td><td>2,048</td></tr><tr><td>Max response length</td><td>16,384</td></tr><tr><td>Rollouts per prompt</td><td>8</td></tr><tr><td>Temperature</td><td>1.0</td></tr><tr><td>Top-p</td><td>1.0</td></tr><tr><td>Training epochs</td><td>5</td></tr><tr><td>Training steps</td><td>600</td></tr><tr><td>Actor parallelism</td><td> $\mathrm { T P } = 2 , \mathrm { P P } = 1$ </td></tr><tr><td>Rollout parallelism</td><td>TP = 2</td></tr><tr><td>Random seed</td><td>42</td></tr></table>

## E PatternEval Judge Prompt

The meta-judge receives the user question, any associated image, and the model response, and independently assigns the four response-pattern labels defined in Section 3.2. The image is not serialized into the text placeholders reproduced below; it is supplied separately as the multimodal image input associated with the same request. The complete textual prompt is reproduced below for transparency. We provide a faithful English translation first, followed by the original Chinese prompt used in our evaluation. Formatting has been adapted for typesetting, but the decision rules, priority policy, examples, required output order, and input placeholders are unchanged. Exact API serialization, image preprocessing, and model-version metadata remain part of the release manifest required for independent reproduction.

## E.1 English Version

## Response-Pattern Meta-Judge Prompt (English Translation)

You are a strict analyzer of undesirable patterns in model responses. Given the user question, any relevant image, and the model response, determine whether the response exhibits each of the following four undesirable patterns. For every pattern, output a binary decision (1 or 0) and a one-sentence justification.

## I. Decision Principles

1. Base the judgment only on the question, any relevant image, and the model response. Every decision must be grounded in observable text in the model response.

2. Use the image to verify image-related descriptions, counts, localization, recognition, and spatial relations in the response. However, do not mark a pattern merely because the answer is incorrect.

3. Factual errors, visual-recognition errors, calculation errors, and incomplete answers are not undesirable response patterns by themselves. Mark 1 only when the error manifests as one of the named patterns below.

4. Judge the four labels independently; multiple labels may be positive. Mark 1 only when there is explicit textual evidence; otherwise mark 0.

5. Length, many steps, or detailed explanation alone do not constitute an undesirable pattern. Do not mark 1 merely because a response is long or contains extensive reasoning.

## II. Decision Order (Important)

## Apply the following fixed order:

1. First judge chain-of-thought leakage: read the full response and check whether it exposes internal deliberation that should remain hidden, including self-correction, tentative wavering, or process-oriented narration that lays out the solution process from the beginning, according to Category 1.

2. Next judge response repetition independently of whether leakage is present, according to Category 2.

3. Finally judge logical contradiction and performative reasoning. First determine whether content that appears contradictory or superficially reasoned is actually part of exposed internal deliberation, such as self-correction, repeated trial, process-level wavering, or draft-like reasoning. If so, attribute that content to chain-of-thought leakage and do not count it again as contradiction or performative reasoning; mark those categories 0 for that content. An exception applies when the final delivered answer contains a separate hard contradiction or an independently unsupported shell of reasoning. Logical contradiction concerns only unresolved, mutually incompatible claims retained in the final delivered content. Performative reasoning concerns analysis that adds no valid support for the conclusion, or staged analysis on a low-reasoning task, when chain-of-thought leakage is not the better explanation.

## III. Four Undesirable Patterns

## 1. Chain-of-Thought Leakage

Definition. The model directly exposes intermediate reasoning, internal self-talk, tentative analysis, task decomposition, selfcorrection, or draft-like thinking that should remain hidden, thereby reducing the naturalness of the response or violating the expected output style.

Mark 1 only when there is explicit evidence of exposed internal reasoning in at least one of the following forms:

• First-person process narration: phrases such as “let me look first,” “I will analyze this first,” “let me think,” “let me check,” “I need to judge this,” or “I need to confirm this first”; or reader-guiding phrases such as “let us analyze these one by one,” “let us analyze step by step,” “let us first look at,” and similar uses of “let me,” “let us,” or “we first.” These expressions narrate how the model is looking or thinking and lead the reader through the model’s working process. Once such process narration appears, it counts as leakage even if followed by bullet points or section headings.

• Self-correction, reversal, wavering, or repeated trial: traces such as “no, let me look again,” “what I said earlier was wrong; it should be,” “wait, let me reconsider,” or “try A first; no, then try B.”

• Exposed internal decisions or drafts: phrases such as “end of thinking,” “now confirmed,” “my preliminary judgment is,” or “let me run through it mentally,” as well as internal decisions, execution strategies, task decomposition, draft reasoning, or residual markers such as think.

Mark 0 in the following cases, provided that none of the above traces is present:

• User-facing structured solution or analysis: sections such as “Analysis,” “Step 1,” and “Step 2,” or objective statements such as “to determine the conclusion, first analyze . . . ,” are not leakage when the response is a polished explanation directed to the user, without first-person process narration, self-correction, wavering, or exposed internal decisions. If such analysis is unsupported, classify it as performative reasoning rather than leakage.

• The user explicitly requests reasoning, steps, a draft, or a method, and the response provides a polished, user-facing derivation, tutorial, procedure, or solution.

• A direct conclusion with a brief justification or concise supporting points.

• A short elimination of alternatives, comparison, or user-facing explanation of multiple perspectives or conditional branches.

Positive example. The question asks for an object’s orientation. The response says, “Let me look at this image first. It seems to face left—no, looking again, it should face right,” exposing repeated tentative inspection.

Negative example. The question asks for the orientation of a piece of furniture. The response contains an “Analysis” section and objectively derives the orientation in Step 1 and Step 2 without first-person self-talk or self-correction. This is not leakage; if the derivation provides no valid image evidence, it is performative reasoning.

## 2. Response Repetition

Definition. The response unnecessarily repeats the same sentence, meaning, structure, or conclusion to a degree that clearly harms information efficiency, readability, or answer quality.

Mark 1 in the following cases:

• The same conclusion, recommendation, or disclaimer is repeated consecutively or frequently.

• The same information or conclusion is restated across multiple sections—for example, analysis, verification, summary, and final answer—without adding information, even if the wording changes.

• Paragraphs are highly homogeneous and merely paraphrase one another.

• The reasoning circles back, repeatedly lists the same candidates or clues, or repeatedly rejects and recomputes the same result beyond what normal emphasis requires, producing a mechanical or cyclic pattern.

Mark 0 in the following cases:

• Limited repetition used to emphasize a key point.

• Necessary local repetition in an enumeration or stepwise structure when the items differ substantively.

• A single useful self-correction or one concluding restatement that naturally connects the reasoning to the final answer without creating obvious redundancy.

Positive example. When asked for a painting’s name, the response gives the title and artist in a heading, then immediately repeats the same title, artist, and year in a “Basic Information” list without adding information.

Negative example. When asked for the answer to a problem, the response provides a derivation and closes with one sentence restating the final answer.

## 3. Logical Contradiction

Definition. In the final delivered content, the response gives two claims about the same object, value, or conclusion that cannot both be true under the same conditions. Both claims remain presented as valid information: the response does not choose between them, invalidate either one, or distinguish their conditions. A positive decision requires locating two specific, mutually incompatible passages, A and B.

A hard contradiction must satisfy all of the following conditions: A and B concern the same object, quantity, or conclusion; they use the same conditions, time, and measurement convention; they cannot both be true; the model does not state that one has been withdrawn or is merely an alternative; and both remain active claims in the answer.

Mark 1 for any of the following four forms:

1. Mutually exclusive final conclusions. The same question, object, or answer position receives two incompatible final conclusions without a choice. For example, the response first identifies an artifact as a bronze zun but ends with “Final answer: bronze ding,” or first states that A is correct and B violates the conditions but ultimately selects B.

2. Incompatible attributes of the same object. The response assigns mutually exclusive values of the same attribute to one object and retains both. For example, it calls block b horizontal in a Klotski puzzle and later instructs the user to move the vertical block b; or it says a function is odd and symmetric about the origin, then says it is symmetric about the y-axis and therefore even.

3. Incompatible values of the same quantity. The response retains two conflicting values for the same number, date, coordinate, count, angle, area, or temperature. For example, it uses two different base areas without a correction; states that an image contains six circles but itemizes three on the left and four on the right for a total of seven; or gives point P first as (2, 3) and later as (3, 2).

4. An unretracted derivation conflicts with the final answer. The reasoning explicitly derives and retains X, but the final answer gives incompatible Y without abandoning the earlier result. For example, the steps identify 8 and 9 as mandatory waypoints but the final waypoint list omits them; or the derivation obtains x = 3, verifies it by substitution, and then gives x = 5 as the final answer.

Mark 0 in the following cases:

• Corrected, abandoned, or resolved alternatives. An earlier statement is explicitly rejected, corrected, or abandoned, or one candidate is selected from several and only one final conclusion remains. Process-level wavering of this kind belongs to chain-of-thought leakage.

• Hedging or uncertainty. Expressions such as “possibly,” “probably,” “more likely,” “tends to be,” or “cannot rule out” indicate alternatives or confidence differences rather than two definite incompatible conclusions. For example, “possibly late Shang, though early Western Zhou cannot be ruled out.”

• Different conditions, times, viewpoints, or conventions are distinguished. For example, the result is X under Assumption 1 and Y under Assumption 2; or an object is on the left from the observer’s viewpoint but on the right relative to its own facing direction.

• A single wrong answer, visual error, or calculation error without an internal contradiction. This category does not evaluate correctness. Mark 0 if the answer is wrong but does not retain two mutually incompatible claims about the same object, quantity, or conclusion.

Imprecision, equivalent restatement, unit conversion, repetition, rough reasoning, or performative reasoning without a locatable pair of incompatible claims. Examples include calling an altocumulus cloud a “mackerel sky” or stating that one meter equals 100 centimeters.

Key distinction. A hard contradiction retains two propositions about the same object attribute, quantity, or conclusion that cannot both be true. Process-level wavering, hedging, visual or factual errors, distinguished conditions, and resolved alternatives do not qualify.

Positive example. When asked which position from the left contains the white-haired person wearing glasses, the response identifies the target as item 4 in its list but concludes “therefore, the fifth.” It retains two incompatible answers, 4 and 5, without resolving them.

Negative example. When asked which animal appears in an image, the response consistently answers “a golden retriever” and adds supporting details such as long fur and golden coloring.

## 4. Performative Reasoning

Definition. The response presents the appearance of analysis, derivation, argument, or evidential support, but that material adds no valid information that bears on the final conclusion. It performs being well-supported rather than providing support.

Valid information gain means an observation, comparison, count, elimination, disambiguation, calculation, or verification that can support the conclusion. For image tasks, valid support must refer to clearly observable content, such as specific objects, text, colors, quantities, positions, relations, chart values, or paths. It may compare size, distance, occlusion, direction, shape, or state; reason from visible paths, walls, nodes, arrows, table rows or columns, and flow branches; deduplicate or match people and objects across images; or read text, legends, coordinates, labels, and values. Generic phrases such as “as can be seen from the image,” “considering the overall layout,” “based on visual characteristics,” or “after comprehensive judgment” usually constitute performative reasoning when they identify no concrete visible content.

Mark 1 only when the response adopts an analytical, inferential, argumentative, or evidence-bearing posture and exhibits one of the following forms:

1. Unsupported reasoning wrapper. The response uses the form of reasoning or evidence without substantive support. Typical signs include “first,” “second,” “therefore,” or “in summary” connecting only generic assertions; merely restating the question or conclusion; claiming to reason “from the image” without naming any concrete element, position, count, text, color, direction, occlusion, or relation; or invoking frameworks such as semantics, structure, and context without connecting them to specific evidence. For example, when asked to identify a plant, the response says that leaf shape, growth habit, and overall visual characteristics establish the species but identifies none of those visible characteristics.

Evidence that does not support the conclusion. The response offers apparently relevant analysis or observations that do not entail the conclusion, or merely packages an answer that could be read directly from the image or was already supplied by the user. Typical cases include analyzing A but concluding B; providing background that cannot establish the answer; identifying something from generic impressions such as “looks similar,” “the style fits,” or “the shape is close” rather than discriminative evidence; adding analysis to a low-reasoning task without any new observation, comparison, count, elimination, disambiguation, or verification; restating a user-provided route without comparing position, roads, distance, reachability, or obstacles; or calling maze nodes “key” or “mandatory” without checking wall and path connectivity or excluding alternatives. This also includes identification answers padded with biography, work lists, general stylistic history, typical roles, or other background that does not explain how the image establishes the identity through a face, inscription, signature, emblem, or other discriminative detail.

## Mark 0 in the following cases:

• A direct answer or brief explanation that does not stage an analysis. Merely saying “based on the image” without naming an image element is treated as a direct answer.

Image analysis that cites concrete visible evidence supporting the conclusion, even if brief, approximate, verbose, or imperfect. Examples include “the lower-right button reads ‘Submit,’ so it is the submit button” and comparing the water levels and vessel shapes of two cups. The evidence must be sufficiently discriminative, such as visible text, an inscription or signature, an exact count or position, or a comparable state. Generic terms such as “similar shape,” “matching style,” “technological feel,” “flame-like core,” or “similar colors” do not qualify as valid evidence.

• A genuine image-connectivity analysis that reaches the wrong result because of a visual error. Mark 0 only when the response actually traces whether adjacent cells connect through walls or passages and uses that connectivity to rule out alternatives. Merely naming cells or nodes as “key” or “mandatory,” or listing their locations without proving connectivity or excluding

alternatives, remains performative reasoning.

• The user explicitly requests justification, analysis, stepwise reasoning, or evidence, and the analysis is not entirely empty; for example, an elimination argument explaining why option C is selected.

• A visual, calculation, or factual error when the cited evidence genuinely bears on the conclusion; for example, counting real objects in the image but missing or miscounting one.

• A cautious statement that the image is unclear and the conclusion cannot be confirmed, followed by a tentative answer or refusal to assert one.

Key distinction. Direct answers, concrete image evidence, valid reasoning, and chain-of-thought leakage receive 0 for this category. Analysis that only restates the question or conclusion, stacks transitions and generic assertions, or claims image evidence without identifying a concrete visible detail receives 1. Evidence may be mistaken yet still be genuine support; do not mark 1 merely because the conclusion is wrong.

Positive example. When asked which button submits a form, the response says that the interface layout, button semantics, and operation path must be considered together and therefore chooses the lower-right button, without naming its text, color, or icon. Negative example. When asked how many apples appear in an image, the response says that one is on the left and two are on the plate on the right, for a total of three.

## IV. Required Output

Follow the exact order and format below. Do not output unrelated content and do not omit any category. For each category, first output 1 or 0, then give a one-sentence justification. When a category is positive, identify the key evidence; for logical contradiction, identify both mutually incompatible passages A and B.

1. Chain-of-thought leakage: 1 / 0   
Reason:   
2. Response repetition: 1 / 0   
Reason:   
3. Logical contradiction: 1 / 0   
Reason:   
4. Performative reasoning: 1 / 0   
Reason:   
V. Input   
Question:   
{question}   
Model response:   
{response}

## E.2 Chinese Version

## Response-Pattern Meta-Judge Prompt (Chinese Original)

你是一个严格的模型回复坏模式分析器。请结合用户问题、相关图片（若有）和模型回答，判断模型回答是否存在以下四类坏模式，并为每一类给出1 或0 的判定和一句话理由。

## 一、判定原则

1. 只依据问题、相关图片（若有）和模型回答来判断，判定要落在模型回答的可观察文本上。

2. 要结合图片核验模型回答里和图像有关的描述、计数、定位、识别、空间关系是否成立；但不要因为答得对不对本身就判坏模式。

3. 事实错误、看图错误、算错、答得不全，这些本身不算坏模式，只有当它们表现为下面某一类可命名的坏模式时才判1。

4. 四个标签各自独立判断，允许同时成立；只有存在明确文本证据时才判1，否则判0。

5. 回答篇幅长、步骤多、解释详细，本身都不是坏模式，不能只因为篇幅长或推理多就判1。

## 二、判定顺序（重要）

## 请按下面的固定顺序判断：

1. 先判思维链外溢：通读回答，看是否暴露了本该隐藏的内部思考、自我纠错、试探摇摆，或把解题过程从头摊开的过程腔（按第1 类标准）。

2. 再判病态重复：独立判断，和是否外溢无关（按第2 类标准）。

3. 再判逻辑性冲突和伪装推理：先看那些看起来像矛盾或像空泛推理的内容，是否其实来自暴露出来的内部思考过程（自我纠错、反复试探、过程摇摆、草稿式推演）。若是，则这部分归思维链外溢、不再计入逻辑性冲突或伪装推理（这两项就该内容判0），除非回答在最终交付内容里另有独立的硬矛盾或独立的空壳推理。逻辑性冲突只针对最终交付内容里成对并列、无法同时成立、且未取舍的硬矛盾；伪装推理只针对推理对结论没有有效信息增量、或在低推理负担任务上表演式分析，且该问题不能被思维链外溢更好地解释时。

## 三、四类坏模式

## 1. 思维链外溢

定义：模型把本该隐藏的中间推理过程、内部自我对话、试探性分析、任务分解、自我纠错或草稿式思考，直接暴露在最终回复里，影响回答的自然度或不符合输出预期。

判1 必须有明确的内部思考外露痕迹，落在下面任一类上：

• 第一人称过程腔：让我先看看、我先分析一下、让我想想、我来看看、这里我需要判断一下、我得先确认一下，以及我们来逐一分析、我们来逐步分析、我们先看、我们可以先等以让我、我们来、我们先等开头、把怎么看怎么想讲出来、带着读者一步步做题的表述。这类一旦出现，即使后面分点、带小标题，也算外溢。

自我纠错、推翻、摇摆、反复试探：不对再看、我前面说错了应该是、等一下重新看、先试A 不对再试B 等中途否定再改或来回试探的痕迹。

• 暴露内部决策或草稿：结束思考、现在确认、我的初步判断是、先在脑子里过一遍等内部决策、执行策略、任务分解的过程，或残留think 这类思考、推理标签。

## 判0 的情形（没有上面这些痕迹就判0）：

面向用户的结构化解题或分析：设了分析过程、步骤一步骤二等板块、把推导分点展开，或以要确定某结论需先分析某某这类非第一人称的客观陈述开头，只要整体是面向用户、成品式地讲题或讲分析（解题步骤、逐项说明、逐图说明），没有让我、我们来这类过程腔、没有自我纠错摇摆、没有暴露内部决策或草稿，就不算外溢。这类回答若分析空转无有效依据，应归伪装推理，不归外溢。

• 用户明确要求展示推理、步骤、草稿，且回答是面向用户的成品式步骤、推导或方法说明（教程、操作流程、解题步骤）。

• 直接交付结论加简短依据或要点。

• 简短的候选排除、比较，或面向用户的多角度、条件分支说明。

示例（判1）：问某物体朝向。回答写让我先看看这张图，嗯好像朝左，不对再看一下应该是朝右，把自己来回试探的过程写了出来。

示例（判0）：问推断某家具朝向。回答设了分析过程板块，按步骤一步骤二客观地把朝向推导讲给用户，没有第一人称自言自语或自我纠错，则不算外溢（若推导无有效据图依据，归伪装推理）。

## 2. 病态重复

定义：模型在回复里不必要地重复同一句话、同一语义、同一结构或同一结论，这种重复已经明显影响信息效率、阅读体验或回答质量。

## 判1 的情形：

• 同一结论、同一建议、同一免责声明被连续或高频重复，反复表达同一个意思。

• 同一个信息或结论在多个段落（比如分析、验证、总结、最终答案）反复重申，即使分属不同小节、措辞略有变化，只要没有新增信息，也算重复。

• 段落之间内容高度同质，只是换少量措辞重新说一遍。

• 推理兜圈、候选和线索反复罗列、同一计算反复否定重来等，超出正常强调需要，呈机械化、循环化倾向。

## 判0 的情形：

• 为了强调重点做的有限重复。

• 枚举或分步结构里必要的局部重复，但各项信息有实质差异。

• 一次性的有效自我纠错、首尾呼应式的一次复述等自然承接，没有造成明显的冗余堆叠。

示例（判1）：问某画的名字。回答先在标题里给出画名和作者，紧接着又用基本信息列表把同样的画名、作者、年代再列一遍，没有新增信息。

示例（判0）：问某题答案。回答给出推导后，结尾用一句话重申最终答案作为收尾，只重复了一次。

## 3. 逻辑性冲突

定义：模型回答在最终交付内容里，针对同一对象的同一属性、同一个量的取值、或同一个结论，给出两个无法同时为真的说法，且两者都被当作有效信息并列保留，没有择一、没有声明其中一个作废、也没有用条件区分。这是硬矛盾，判1 时必须能成对定位互斥的两处具体文字A 和B。

构成硬矛盾要同时满足：A、B 针对同一对象或同一量或同一结论；处在同一条件、同一时间、同一口径下；逻辑上无法同时为真；模型没有说明其中一个已作废或只是备选；A、B 都在回答里作为有效信息并列保留。

## 判1 的四类情形（要能指出A、B 两处具体文字）：

1. 同一最终结论并列互斥：对同一问题、同一对象或同一答案位置给出两个无法同时成立的最终结论，且没有择一。例如问藏品名称，先说是青铜尊，最后又写最终答案青铜鼎；又如先说A 正确、B 不符合条件，最终答案却选B。

2. 同一对象属性互斥：对同一对象的同一属性给出无法同时成立的判断，且两处都保留使用。例如华容道里先说b 是横向木块，移动步骤里又写移动竖向木块b；又如先说某函数是奇函数关于原点对称，又说它关于y 轴对称所以是偶函数。

3. 同一量取值互斥：对同一个数值、日期、坐标、数量、角度、面积、温度给出两个无法同时成立的取值，且没有说明修正关系。例如同一底面积先写成一个值后又写成另一个值且都在用；又如说图中共6 个圆，分项却给出左3 右4 共7 个，总数口径互斥；又如点P 坐标先写成(2, 3) 后又写成(3, 2)。

4. 推导结论与最终答案互斥且推导未被放弃：推导中明确得到并保留X，最终答案却给出与X 互斥的Y，且没有声明前文作废。例如步骤里说8、9 是必经点，最终必经点列表却不含8、9；又如推导得x 等于3 并代入验证成立，最终答案却写x 等于5。

## 判0 的情形：

• 已修正、已放弃或已择一：先出现一种说法，随后明确否定、修正、放弃，或在多个候选中明确选定，最终只保留唯一结论（这类过程摇摆归思维链外溢）。

• 模糊、推测、对冲表达：用可能、大概、更像、倾向于、不排除等表示候选或置信度差异，没有形成两个确定互斥的结论。例如说可能是商代晚期，也不排除西周早期。

• 条件、时间、视角、口径不同且已区分：例如按假设一结果为X、按假设二结果为Y；又如从观察者视角看在左侧、按物体自身朝向看在右侧。

答案错误、看图错误、计算错误，但内部没有成对互斥的并列说法：本类不判断答案对错，看错图、数错数、算错值，只要内部没有对同一对象或同一量或同一结论并列保留两个互斥说法，就判0。

• 表达不严谨、同义改写、单位换算、重复说明、推理粗糙、伪装推理，但指不出两处并列保留且无法同时成立的具体说法。例如说是高积云即鱼鳞云（分类层级或俗称），又如说1 米也就是100 厘米（等价换算）。

关键区分：硬矛盾是最终交付内容里对同一对象属性、同一量取值或同一结论并列保留两个无法同时为真的命题，判1；过程摇摆、模糊措辞、看错算错事实错、条件区分或已择一，都判0。

示例（判 ）：问从左到右数白头发戴眼镜的人是第几个。回答列表里写第 项是目标人物，结尾却写所以是第 个，对同一答案并列保留4 和5 两个互斥结论且没有取舍。

示例（判0）：问图中是什么动物。回答从头到尾只说是金毛犬，并补充长毛、金黄色等依据，没有互相否定的说法。

## 4. 伪装推理

定义：模型表面上呈现出分析、推导、论证或证据支撑的样子，但这些内容没有为最终结论提供有效信息增量，只是在制造结果有理有据的表象。

有效信息增量指能支持结论的观察、比较、计数、排除、消歧、计算或核验。在图片任务中，有效增量必须落在图中明确可见的内容上，例如：指出具体物体、文字、颜色、数量、位置；比较多个对象的大小、距离、遮挡、方向、形状、状态；依据图中可见路径、墙体、节点、箭头、表格行列、流程分支判断；对多图人物或物体做去重、匹配、排除；读取图中文字、图例、坐标、标签、数值并用于支撑结论。如果只是套用从图中可以看出、结合整体布局、根据视觉特征、综合判断等话术，却没有指出任何具体可见内容，通常属于伪装推理。

## 判1 的两类情形（仅当回答已经摆出分析、推导、论证、证据支撑的姿态时）：

1. 一是无依据的推理包装：表面用了推理或证据支撑的表达，却没有提供任何能支持结论的实质依据。典型表现：用首先、其次、因此、综上、可见等连接词但中间只是空泛表态；只是重复题干、改写结论、复述用户已给信息；图片任务里声称根据图中某某、从视觉上判断、结合空间布局，却不指出具体元素、位置、数量、文字、颜色、方向、遮挡或关系；堆砌从语义结构上下文三方面看等分析框架却不连接到题面或图中具体信息。例如问图中植物是什么，回答说结合叶片形态、生长习性和整体视觉特征可判断是某植物，却不指出任何具体可见特征。

2 二是依据不支撑结论：给出一些看似相关的分析或观察，但推不出最终结论，或只是在包装一个本可直接读取、直接观察或用户已给出的答案。典型表现：分析的是A 结论却落到B；前文只是背景介绍推不出答案；识别题靠看起来像、风格符合、造型接近等泛泛印象，而不是能确认身份或对象的真实依据；低推理负担任务里答案本可直接观察读取定位，额外分析没有新增观察、比较、计数、排除、消歧或核验；路线规划只复述用户已给顺序，没有比较位置、道路、距离、可达性或障碍；迷宫或路径题只说某点是关键节点、必经节点，却没有用墙体、通道连通性或替代路线排除来证明。例如认人题只凭穿着正式、气质沉稳像某类演员就下身份结论；又如计数题用小狗有四条腿毛茸茸来包装共4 只这个数。这里还包括识别鉴定类的背景包装：给出名称或身份后主要堆砌生平、作品列表、风格通论、常演角色、历史介绍等与如何从图中确认无关的背景，却不指出图中能锁定结论的可辨识特征（面部、铭文、款识、标志性细节）。

## 判0 的情形：

• 直接作答或只做简短说明，没有展开分析来包装结论。注意只说结合图片而不指出图中具体要素，按直接作答处理。

• 图片分析有有效可见依据：指出了图中具体可见内容并用来支撑结论，即使简短、概括、啰嗦、不完美。例如说右下角按钮写着提交所以是提交按钮；又如比较两杯水位和容器形状判断哪杯更多。注意：这里要求的是能唯一锁定结论的具体可辨识细节，例如读出文字、铭文、款识，给出确切的数量、位置、可比较的状态；如果只给造型接近、风格符合、科技感、火焰核心、画风色彩相近、看着像这类泛泛特征词，不算有效依据，应按判1 处理。

• 据图做了真实连通性推进但看错（看图错误，判0）：仅当模型确实逐格交代了墙体或通道是否连通、相邻格能否通行，并据此排除替代路线（即给出了能支撑路径结论的连通性核验），只是把某段墙体或通道看错时，才算看图错误判0。如果只是泛泛点名某些格子、节点为关键点或必经点、或只复述各数字所在位置，而没有用连通性或排除来证明，仍按上面的依据不支撑结论判1。

• 用户明确要求说明理由、分析过程、逐步判断或给出依据，且分析不是完全空转。例如用排除法说明为什么选C。

• 看错、算错、事实错，但依据和结论之间存在真实支撑关系。例如基于图中真实对象计数，只是漏数或数错。

• 看不清图片时的谨慎说明：明确表示图片模糊、无法确认，并给出谨慎结论或拒绝断言。

关键区分：直接答题、有效图像依据、有效推理、思维链外溢都判0；看似在分析实则重复题干、复述结论、堆连接词、泛泛表态，或声称结合图中具体特征却不给任何具体可见细节，判 ；依据可能错但确实在尝试基于图片或题面支撑结论的，判 。不要因为回答长、步骤多、语气像推理就判1，也不要因为结论错就判1。

示例（判 ）：问图中哪个按钮是提交按钮。回答说需要结合界面布局、按钮语义和操作路径综合判断，因此是右下角那个，没有指出按钮文字、颜色或图标等具体可见依据。

示例（判0）：问图中有几个苹果。回答说左侧1 个、右侧盘子里2 个、一共3 个，给出了具体的图像观察和计数依据。

## 四、输出要求

请严格按下面顺序和格式输出，不要输出无关内容，不要省略任何类别。每一类先给1 或0 的判定，再用一句话说明理由（判1 时点出关键依据，逻辑性冲突要点出互斥的A、B 两处）。

<sub>1.</sub> 思维链外溢：<sub>1</sub> / <sub>0</sub>

理由：

<sub>2.</sub> 病态重复：<sub>1 / 0</sub>

理由：

<sub>3.</sub> 逻辑性冲突：<sub>1 / 0</sub>

理由：

<sub>4.</sub> 伪装推理：<sub>1</sub> / <sub>0</sub>

理由：

五、输入

问题：

{question}

模型回答：

{response}