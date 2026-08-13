# Context Blindness in DPO: Mitigating Object Hallucination in MLLMs via Context-Calibrated Preference Optimization

Byungoh Ko<sup>1</sup> , Jinyoung Park<sup>2</sup>, Jongha Kim<sup>2</sup>, Jeehye Na<sup>2</sup>, Jaewon Cho<sup>2</sup>, and Hyunwoo J. Kim<sup>2</sup> <sup>⋆</sup>

<sup>1</sup> Korea University, Seoul, Republic of Korea ko990128@korea.ac.kr <sup>2</sup> KAIST, Daejeon, Republic of Korea hyunwoojkim@kaist.ac.kr

Abstract. Multimodal large language models (MLLMs) have made rapid progress, yet they still exhibit object hallucination, generating plausible but incorrect descriptions that are inconsistent with the visual input. Direct Preference Optimization (DPO) mitigates this by training models to prefer non-hallucinated responses over hallucinated ones, and recent eforts further enrich the preference data with relevant context. However, it remains unclear whether DPO actually leverages such context. To investigate this, we propose Contextual Preference Gain (CPG), a simple metric that measures how much a model’s preference strengthens when relevant context is provided. We find that higher CPG consistently corresponds to lower hallucination, yet standard DPO and its variants exhibit only limited CPG, indicating that they underutilize contextual information and thus remain prone to hallucination. To address this, we propose Context-Calibrated DPO (C<sup>2</sup>-DPO), which directly maximizes CPG while preserving the original preference ordering. Across multiple benchmarks, C<sup>2</sup>-DPO substantially reduces hallucination without compromising general reasoning, relatively reducing the Object HalBench hallucination rate of Qwen2-VL-Instruct-2B by 36%. Code is available at https://github.com/mlvlab/C2-DPO

Keywords: Multimodal Large Language Models · Object Hallucination · Preference Optimization

## 1 Introduction

Large language models (LLMs) [2, 14, 30, 32, 49] have shown remarkable performance across a wide range of language tasks. Building on these advances, multimodal large language models (MLLMs) [5,12,22,25,26,45] integrate visual encoders and multimodal training data, enabling unified understanding of images and text. Recent MLLMs achieve impressive results on core vision-language tasks such as image captioning [1, 24] and visual question-answering [3, 15, 28, 40, 51].

![](images/a3074cb749292d64acd1f6b1d662c3ce3b01d7873709a6e17a5e6c89ececea1f.jpg)  
Fig. 1: Motivation of C<sup>2</sup>-DPO. A model should yield higher preference scores when richer context is provided. However, standard DPO shows minimal change between full and degraded contexts, revealing context-blind behavior. C<sup>2</sup>-DPO amplifies this gap, enabling models to better leverage contextual information and reduce hallucination.

Despite the progress, MLLMs remain prone to object hallucination, where the model generates semantically plausible descriptions that are not grounded in the visual input. This issue undermines the factual reliability of multimodal reasoning. To mitigate object hallucination, recent works have applied preference optimization, which trains the model to prefer non-hallucinated responses rather than hallucinated responses. Early studies [33,52,53] primarily focused on constructing high-quality preference data for preference optimization [37]. More recently, a follow-up study [36] enriches the input with non-hallucinated descrip tions to strengthen contextual grounding, thereby making preferred and dispreferred responses more discriminative and improving alignment performance.

While these approaches modify the preference dataset for better optimization, a more fundamental question remains unanswered: “Does DPO encourage a model to leverage relevant context and learn contextual preference?” Motivated by the information-theoretic view that relevant information reduces uncertainty [7, 41], we argue that efective preference optimization should strengthen the model’s preference for the correct response when richer contextual information is available. Structurally, DPO optimizes the preference margin for a single fixed input, so its gradient never rewards a larger margin when richer context is supplied. To investigate this, we introduce Contextual Preference Gain (CPG), a metric that measures how much the model’s preference strengthens when an additional helpful context is provided. Using CPG, we identify two notable observations. First, CPG is strongly and negatively correlated with hallucination rate: models with higher CPG consistently achieve lower hallucination rates. Second, standard DPO and its variants show CPG distributions concentrated near zero, suggesting that their learned preferences are context-blind. Together, these findings indicate that existing preference optimization approaches inadequately exploit contextual information for grounding.

To address this limitation, we propose Context-Calibrated Direct Preference Optimization (C<sup>2</sup>-DPO), a context-aware preference alignment framework designed to amplify grounding signals for mitigating object hallucination in MLLMs. C<sup>2</sup>-DPO explicitly models how helpful context influences the relative preference between preferred and dispreferred responses, and encourages higher preference scores of the input with the additional relevant context. Specifically, a calibration term directly rewards the preference margin obtained with context over the one obtained without it, turning contextual grounding into an explicit training signal rather than an emergent side efect. By calibrating preference scores with respect to contextual information, $\mathrm { C ^ { 2 } { - } D P O }$ enables MLLMs to better leverage input context and reduce object hallucination.

Comprehensive experiments across diverse benchmarks demonstrate that $\mathrm { C ^ { 2 } } \mathrm { - }$ DPO efectively reduces object hallucinations while preserving general reasoning and instruction-following capabilities. On Object Halbench [38], hallucination rates decrease by 36% and 60% in response and mention level, respectively, with consistent improvements observed on HallusionBench [16]. Furthermore, C<sup>2</sup>-DPO maintains its general reasoning performance on ScienceQA, MM-Vet, and TextVQA [28, 40, 51], indicating that mitigating object hallucination does not come at the cost of overall downstream task performance. Beyond the standard DPO, we further show that our context-aware preference gain formulation can be seamlessly incorporated into other preference-optimization variants such as SimPO [29] and RDPO [35], consistently yielding similar gains. Additionally, experiments in text-only settings confirm that $\bar { \mathrm { C } } ^ { 2 } { - } \mathrm { D P O }$ enhances factual alignment even without visual inputs, suggesting that context-aware preference modeling ofers broad benefit beyond multimodal domains.

Our contributions are summarized as follows:

– We introduce Contextual Preference Gain (CPG), a diagnostic metric that quantifies how preference margins change with additional grounding information, and reveal a strong correlation between CPG and hallucination rate, exposing a context blindness phenomenon in existing DPO-style objectives. – We propose $\mathrm { C ^ { 2 } { - } D P O }$ , a general preference alignment framework that explicitly optimizes contextual preference gain and integrates seamlessly with various preference optimization methods such as DPO, SimPO, and RDPO.

– We demonstrate the efectiveness and generality of $\mathrm { C ^ { 2 } { - } D P O }$ through extensive experiments across multimodal and text-only benchmarks, achieving substantial reduction in object hallucination while preserving the general reasoning performance.

## 2 Related Works

## 2.1 Hallucination in MLLMs

Multimodal large language models (MLLMs) often sufer from the “object hallucination problem”, where generated responses appear natural but deviate from input conditions [38, 54]. To mitigate this, several inference-time approaches based on Contrastive Decoding (CD) [23] have been proposed [8,10,19,21], controlling the model behavior during generation by contrasting multiple outputs. Still, CD-based approaches remain post-hoc remedies that fail to fundamentally correct the model’s intrinsic behavior, yielding unsatisfactory performance. Also, they require additional inference cost as multiple forwardings are required during generation. In parallel, Preference Optimization (PO) methods aim to directly train MLLMs to be less hallucinatory [17,33,36,50,52,53]. However, prior works focus on constructing better preference datasets, overlooking context blindness, a key limitation that prevents MLLMs from fully leveraging multimodal cues. Crucially, these methods alter what data the model sees but leave the optimization objective unchanged, so the enriched context is not guaranteed to be exploited during alignment. We identify this issue and propose a solution that further enhances the alignment with the same training data.

## 2.2 Direct Preference Optimization (DPO)

Reinforcement Learning from Human Feedback (RLHF) aligns model behavior with human preferences [9, 34, 39, 42]. By optimizing models toward preference data, RLHF has proven efective in reducing hallucination in MLLMs [17,33,36, 50, 52, 53]. Among its variants, Direct Preference Optimization (DPO) [37] has gained significant attention for its simplicity, removing the need for an explicit reward model. However, recent studies have revealed several limitations of DPO, including length bias [29], suboptimality of reference models [29, 46, 48], overfitting [4], and inaccurate reward estimation [47]. These eforts refine how preference is modeled for a fixed input, $e . g .$ , by reshaping the reward margin or removing the reference model. Far less attention has been paid to how the preference should respond when the input context itself varies, which is precisely the regime where multimodal grounding matters. In this work, we highlight another overlooked issue, context blindness, where MLLMs trained with existing preference objectives fail to fully utilize or comprehend input contexts. To address this, we introduce a new DPO-based training objective that explicitly strengthens contextual understanding in multimodal alignment, ofering a principled approach to further mitigate hallucination, complementary to dataset-scaling strategies.

## 3 Preliminaries

We briefly summarize Direct Preference Optimization for mitigating object hallucination in multimodal large language models (MLLMs).

Object hallucination in MLLMs refers to the generation of text that does not align with the visual content of the input image. To address this, preference alignment such as Direct Preference Optimization (DPO) [37] has emerged as a promising approach for mitigating the object hallucination problem. The preference alignment method is designed to train a policy model $\pi _ { \theta }$ to reflect human preferences over candidate responses. Given an input x and a pair of responses $( y _ { w } , y _ { l } )$ where $y _ { w }$ is the preferred sample while y<sub>l</sub> represents the dispreferred sample, DPO optimizes the model to assign higher preference to $y _ { w }$ than $y _ { l }$

One of the representatives of preference modeling is the Bradley–Terry (BT) framework [6], which formulates pairwise preferences as probabilistic outcomes based on diferences in underlying rewards. In the BT model, the probability of preferring $y _ { w }$ over $y _ { l }$ under context x is defined as

$$
p \left( y _ { w } \succ y _ { l } \mid x \right) = \sigma \left( r \left( x , y _ { w } \right) - r \left( x , y _ { l } \right) \right) ,\tag{1}
$$

where $\sigma ( \cdot )$ denotes the sigmoid function and $r \left( x , y \right)$ is the reward associated with response $y$ under context x.

Unlike the BT model, which requires an explicit reward function, DPO introduces an implicit reward $\hat { r } _ { \theta }$ parameterized by the policy model itself:

$$
\hat { r } _ { \theta } \left( x , y \right) = \beta \log \frac { \pi _ { \theta } \left( y \mid x \right) } { \pi _ { \mathrm { r e f } } \left( y \mid x \right) } ,\tag{2}
$$

where $\pi _ { \theta }$ is the trainable policy model, $\pi _ { \mathrm { r e f } }$ is a frozen reference model, and $\beta > 0$ controls the regularization strength. Substituting this implicit reward into the BT formulation yields the DPO loss:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { D P O } } ( x , y _ { w } , y _ { l } ) = - \log p ( y _ { w } \succ y _ { l } \mid x ) } \\ & { \qquad = - \log \sigma \left( \hat { r } _ { \theta } \left( x , y _ { w } \right) - \hat { r } _ { \theta } \left( x , y _ { l } \right) \right) } \\ & { \qquad = - \log \sigma \left( \beta \log \frac { \pi _ { \theta } \left( y _ { w } \mid x \right) } { \pi _ { \mathrm { r e f } } \left( y _ { w } \mid x \right) } - \beta \log \frac { \pi _ { \theta } \left( y _ { l } \mid x \right) } { \pi _ { \mathrm { r e f } } \left( y _ { l } \mid x \right) } \right) . } \end{array}\tag{3}
$$

Thus, DPO can be viewed as maximizing the probability that the model’s pairwise preferences align with human judgments, while avoiding the need for an explicit reward model.

Following other DPO works [17,33,36,50,52,53] for mitigating object hallucination, we consider $\boldsymbol { x } = ( v , q , c )$ as the input, where v denotes the input image, $q$ is the text query, and c represents the auxiliary image description helpful for understanding the input image $( e . g . ,$ caption). Here, we treat a non-hallucinated response as the preferred response $y _ { w }$ and a hallucinated response as the dispreferred response $y _ { l }$ . Notably, the DPO objective in Eq. (3) treats x as a monolithic input and is agnostic to the role of c: it never explicitly requires that providing c alter the preference between $y _ { w }$ and $y _ { l }$ . This gap is precisely what we analyze in Section 4.2. For simplicity, since each training instance always includes $( y _ { w } , y _ { l } )$ we denote the DPO loss as $\mathcal { L } _ { \mathrm { D P O } } ( x )$ hereafter.

## 4 Method

## 4.1 Overview

Our work aims to optimize multimodal large language models (MLLMs), via preference optimization, to generate outputs that align more closely with the input context. Most existing approaches [17,33,36,50,52,53] modify the preference dataset, either by rewriting sentences with an external model or by appending enriched image descriptions to the input prompt. Such modifications make the contrast between hallucinated and non-hallucinated objects more explicit, helping the model understand the preference gap.

![](images/6596c77043459224be7cad8e5597068f4112591e3481d6eb66e8dff01ae7f920.jpg)

![](images/43b8d5e5cdadec0d386ac9c6af82b4be0fe6a55059603d3f26decb8a2a1b8eb8.jpg)  
Fig. 2: Correlation between CPG and hallucination rates. We analyze the relationship between Contextual Preference Gain (CPG) and hallucination rate across multiple models. On both Object HalBench (left) and AMBER (right), CPG shows a strong negative correlation with hallucination rate, suggesting that increased CPG is closely associated with improved grounding.

However, recent approaches pay little attention to whether the model genuinely grounds this preference gap between preferred (non-hallucinated) and dispreferred (hallucinated) samples, leveraging the input context such as images or captions. We hypothesize that efective preference optimization should lead MLLMs to increasingly favor preferred response $y _ { w }$ over dispreferred response y<sub>l</sub> as additional relevant context information is richer. In other words, the preference gap between $y _ { w }$ and $y _ { l }$ becomes larger as richer contexts (e.g., captions) are given, while maintaining the preference ranking between them. This perspective suggests that the model should not only distinguish responses, but also calibrate the strength of this distinction according to the richness of context. With the preliminary experiments, we found that the standard DPO fails to satisfy these properties, revealing the context blindness.

To address this challenge, we present Context-Calibrated Direct Preference Optimization $\left( \mathrm { C ^ { 2 } - D P O } \right)$ , a preference optimization method with a simple calibration term. Our calibration encourages the model to prefer responses grounded in rich context over degraded context by maximizing the contextual preference gain through a contrastive learning objective. This lets multimodal large language models better leverage input context during preference alignment, thereby mitigating object hallucination.

## 4.2 Observations

To take a deep dive into the preference alignment of the standard preference optimization, we measure the Contextual Preference Gain (CPG) according to the richness of the input. We first introduce the preference score as a measure of alignment strength via implicit reward margin between a preferred response

![](images/2cb0caa845278d1dc611baa804a3247daff9e02b49837a2e6078560296d67a91.jpg)  
(a) DPO

![](images/e8ca8017c100b2518049d7bc1359631669a285902b7705c4b44242c1f83bef45.jpg)  
(b) SimPO

![](images/71beb64824ae31ac52ee0b71ecb8d4694270ac2dd30d806818f1e5005fe16128.jpg)  
(c) RDPO  
Fig. 3: Distributions of Contextual Preference Gain (CPG) for three preference optimization methods: DPO, SimPO, and RDPO. In all cases, the CPG values are highly concentrated near zero, indicating that removing contextual information leads to only minimal changes in preference scores. This demonstrates that existing preference optimization methods are largely context-blind.

and a dispreferred response:

$$
\begin{array} { r } { \varDelta \hat { r } _ { \theta } ( x , y _ { w } , y _ { l } ) = \hat { r } _ { \theta } ( x , y _ { w } ) - \hat { r } _ { \theta } ( x , y _ { l } ) , } \end{array}\tag{4}
$$

which represents how strongly the model prefers the non-hallucinated response $y _ { w }$ over the hallucinated one $y _ { l }$ given input x.

Contextual Preference Gain (CPG). Intuitively, providing additional relevant context should help a model more confidently distinguish a preferred response from a dispreferred one. As a result, the model should exhibit a stronger preference for the correct and grounded response when richer contextual information is available. This intuition aligns with the information-theoretic view that relevant information reduces uncertainty [7,41]. Motivated by this idea, we define Contextual Preference Gain (CPG) as the change in preference score when context is enriched. Given an input x and its degraded counterpart $x ^ { \prime }$ , CPG is defined as

$$
\mathrm { C P G } ( x , x ^ { \prime } ) = \varDelta \hat { r } _ { \theta } ( x , y _ { w } , y _ { l } ) - \varDelta \hat { r } _ { \theta } ( x ^ { \prime } , y _ { w } , y _ { l } ) .\tag{5}
$$

A positive CPG indicates that richer or more reliable input leads to a stronger preference for $y _ { w }$ over $y _ { l }$

We evaluate CPG under a context-removal setting where $\boldsymbol { x } ^ { \prime } = ( v , q , \emptyset )$ corresponds to the input with the auxiliary caption c removed. As shown in ${ \mathrm { F i g } } -$ ure 2, CPG exhibits a strong negative correlation with hallucination rates on Object HalBench [38] and AMBER [44], where models with higher CPG consistently achieve lower hallucination rates. This suggests that CPG captures a model’s ability to leverage contextual information for grounding. However, as shown in Figure 3, existing preference optimization methods (DPO, SimPO, and RDPO [29, 35, 37]) show CPG distributions highly concentrated near zero, indicating that their learned preferences are largely insensitive to contextual information. Given the strong association between CPG and hallucination rate, this context-blind behavior reveals a fundamental limitation in existing preference optimization methods.

![](images/5f9a983ac3780383a3b9b71a568836863c83e007f8a2924b66f30f812f654be0.jpg)  
Fig. 4: Context-Calibrated Direct Preference Optimization via Contextual Preference Calibration $\left( \mathbf { C } ^ { 2 } \mathbf { - D P O } \right)$ . Given a full context x and a degraded context $x ^ { \prime } , \mathrm { ~ C ^ { 2 } { - } D P O }$ calibrates the preference score by (1) applying the standard DPO loss on x, (2) contrasting preference scores between x and $x ^ { \prime }$ via a Contextual Preference Calibration loss to maximize contextual preference gain, and (3) applying DPO on $x ^ { \prime }$ to preserve correct preference ordering under degraded context. This joint objective encourages the model to strengthen preference for grounded responses when richer context is available, efectively reducing hallucination.

## 4.3 Contextual Preference Calibration

Section 4.2 shows that the CPG indicates how well a model grounds its responses in context, and thus how prone it is to hallucinate. Standard preference optimization leaves the CPG concentrated near zero, a failure we term context blindness. A context-calibrated model should instead satisfy the ordering

$$
\underbrace { \varDelta \hat { r } _ { \theta } ( x , y _ { w } , y _ { l } ) } _ { \mathrm { f u l l ~ c o n t e x t } } > \underbrace { \varDelta \hat { r } _ { \theta } ( x ^ { \prime } , y _ { w } , y _ { l } ) } _ { \mathrm { d e g r a d e d ~ c o n t e x t } } > 0 ,\tag{6}
$$

where each term is the preference score from Eq. (4) under the corresponding context. The first inequality requires the preference score under the full context to exceed the degraded-context one, which is exactly a positive CPG. The second requires it to stay positive on its own, so that $y _ { w }$ remains favored over y<sub>l</sub> even without the auxiliary cues. Together, these require the model to preserve the preference ranking across contexts while maximizing the gap as the context becomes richer.

We formulate these desiderata as a primary objective subject to two constraints, taking the full-context DPO loss as the objective and the two inequalities of Eq. (6) as the constraints:

$$
\operatorname* { m i n } _ { \theta } \ \mathcal { L } _ { \mathrm { D P O } } ( x ) \quad \mathrm { s . t . } \quad \mathrm { C P G } ( x , x ^ { \prime } ) \geq 0 , \qquad \Delta \hat { r } _ { \theta } ( x ^ { \prime } , y _ { w } , y _ { l } ) \geq 0 .\tag{7}
$$

Rather than solving Eq. (7) directly, we relax each constraint into a diferentiable surrogate that decreases monotonically with its margin, scaled by coeficients $\lambda _ { c } , \lambda _ { u } > 0$ . We derive the two surrogates below.

Contextual Preference Calibration Loss. For the first constraint, we contrast the two preference scores with a binary NCE-style objective, treating the full context x as the positive and the degraded context $x ^ { \prime }$ as the negative:

$$
\begin{array} { r l } & { \mathcal { L } _ { c } ( x , x ^ { \prime } ) = - \log \frac { \exp { \left( \varDelta \hat { r } _ { \theta } \left( x , y _ { w } , y _ { l } \right) \right) } } { \exp { \left( \varDelta \hat { r } _ { \theta } \left( x , y _ { w } , y _ { l } \right) \right) } + \exp \left( \varDelta \hat { r } _ { \theta } \left( x ^ { \prime } , y _ { w } , y _ { l } \right) \right) } } \\ & { \quad \quad = - \log \sigma \left( \varDelta \hat { r } _ { \theta } \left( x , y _ { w } , y _ { l } \right) - \varDelta \hat { r } _ { \theta } \left( x ^ { \prime } , y _ { w } , y _ { l } \right) \right) } \\ & { \quad \quad = - \log \sigma \left( \mathrm { C P G } ( x , x ^ { \prime } ) \right) . } \end{array}\tag{8}
$$

Minimizing $\mathcal { L } _ { c }$ monotonically increases the CPG, driving the full-context preference score above the degraded-context one. Since a grounded response is favored more strongly as the context grows richer, the model is encouraged to leverage the additional context rather than ignore it.

Preference under the degraded context. For the second constraint, we apply the DPO objective to the degraded context x′:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D P O } } ( x ^ { \prime } ) = - \log \sigma ( \varDelta \hat { r } _ { \theta } ( x ^ { \prime } , y _ { w } , y _ { l } ) ) , } \end{array}\tag{9}
$$

a monotone surrogate that pushes the degraded-context score positive, keeping $y _ { w }$ favored even when the auxiliary cues are removed. Anchoring this score prevents $\mathcal { L } _ { c }$ from degenerately widening the CPG by lowering $\varDelta \hat { r } _ { \theta } ( x ^ { \prime } , y _ { w } , y _ { l } )$ rather than improving grounding.

Training objective. Combining the primary objective in Eq. (7) with the two surrogate terms gives our full training objective:

$$
\mathcal { L } _ { \mathrm { C } ^ { 2 } \mathrm { - D P O } } ( x , x ^ { \prime } ) = \mathcal { L } _ { \mathrm { D P O } } ( x ) + \lambda _ { c } \mathcal { L } _ { c } ( x , x ^ { \prime } ) + \lambda _ { u } \mathcal { L } _ { \mathrm { D P O } } ( x ^ { \prime } ) ,\tag{10}
$$

where $\lambda _ { c } , \lambda _ { u } > 0$ weight the two surrogates against the full-context DPO loss. Jointly optimizing the three terms encourages the model to satisfy the ordering in Eq. (6), thereby increasing CPG and reducing object hallucination. This allows MLLMs to better leverage the input context, improving robustness to hallucination while maintaining a stable preference across input conditions.

## 5 Experiments

In this section, we empirically investigate the efectiveness of $\mathrm { C ^ { 2 } { - } D P O }$ in enhancing context awareness and reducing hallucinations. We introduce the experimental setup in Sec. 5.1, present the main results in Sec. 5.2, conduct ablation studies in Sec. 5.3, and provide a qualitative analysis in Sec. 5.4.

## 5.1 Experimental Setup

Models and training settings. We evaluate our method on two representative MLLMs of diferent scales: LLaVA-v1.5-7B [25] and Qwen2-VL-Instruct-2B [45].

For a fair comparison, we adopt the same training data as C-DPO [36], using the datasets released for LLaVA-v1. $. 5 – 7 \mathrm { B ^ { 3 } }$ and Qwen2-VL-Instruct-2B<sup>4</sup>, respectively. Each preference data consists of an image, a query, and an auxiliary caption, paired with a non-hallucinated (preferred) and a hallucinated (dispreferred) response. For the degraded input $x ^ { \prime } { \mathrm { . } }$ we remove the auxiliary caption while keeping the image and query, following the setting in Section 4.2; results under alternative degradations (e.g., noisy images, randomly substituted captions) are reported in the supplementary material. We set $\beta = 0 . 1$ , with $( \lambda _ { c } , \lambda _ { u } )$ set to (0.3, 0.5) for LLaVA-v1.5-7B and (0.5, 0.3) for Qwen2-VL-Instruct-2B, selected from the [0.3, 0.5] range based on the sensitivity analysis in Section 5.3. All models are trained for one epoch using LoRA [18] with AdamW [27] at a learning rate of $2 \times 1 0 ^ { - 6 }$ . Training is conducted with a global batch size of 64. Evaluation benchmarks. We evaluate the hallucination rates and general multimodal reasoning ability of our C<sup>2</sup>-DPO approach across a wide range of benchmarks. To evaluate hallucination, we use Object HalBench [38], AMBER [44], and HallusionBench [16]. To assess general multimodal reasoning performance, we further evaluate on ScienceQA [28], MM-Vet [51], and TextVQA [40].

Baselines. We compare C<sup>2</sup>-DPO with a wide range of state-of-the-art methods. Specifically, VCD [21], OPERA [19], and DoLa [10] apply contrastive decoding strategies, whereas HA-DPO [52], POVID [53], CLIP-DPO [33], RLAIF-V [50], TPO [17], and C-DPO [36] adopt preference optimization approahces. We additionally include a vanilla-DPO baseline, obtained by training with the original DPO [37] objective on the C-DPO dataset [36], without using any auxiliary captions. Further details on training, datasets, and evaluation protocols are provided in the supplementary material.

## 5.2 Main Results

Comparison with baselines. As shown in Table 1, C<sup>2</sup>-DPO delivers consistent and substantial improvements over prior methods across a wide range of hallucination benchmarks. On Object HalBench, our approach achieves the strongest performance among all compared methods. For Qwen2-VL-Instruct-2B, C<sup>2</sup>-DPO reduces response-level and mention-level hallucinations to 1.6 and 1.0, outperforming C-DPO [36] by large margins and yielding a 36% and 60% relative reduction, respectively. A similar trend holds on AMBER, where $\mathrm { C ^ { 2 } { - } D P O }$ achieves a hallucination score of 16.1, lower than both vanilla-DPO (39.9) and C-DPO (17.5). The improvements generalize to the larger model as well: on LLaVA-v1.5- 7B, our method attains a response-level hallucination of 4.8 and a mention-level score of 2.7, along with clear reductions across all AMBER dimensions. These hallucination reductions come without sacrificing general multimodal reasoning performance, as C<sup>2</sup>-DPO maintains strong results on ScienceQA, MM-Vet, and

Table 1: Comparison of hallucination mitigation methods in MLLMs. The best and second-best results are highlighted in bold and underline, respectively. Al comparisons are conducted under identical model size and architecture. "Rsp." and "Men." denote response-level and mention-level hallucination degrees, while "Hal." and "Cog." represent the Hallucination Score and Cognitive Score, respectively.
<table><tr><td rowspan="3">Model</td><td rowspan="3">Method</td><td colspan="6">Hallucination Benchmarks</td><td colspan="3">General Benchmarks</td></tr><tr><td colspan="2">Object HalBench</td><td colspan="2">AMBER</td><td>HallusionBench</td><td></td><td colspan="2">[ScienceQA MM-Vet TextVQA</td><td></td></tr><tr><td>Rsp. ↓</td><td>Men. ↓</td><td>CHAIR ↓ Hal. ↓ Cog. ↓</td><td></td><td></td><td>Question Acc. ↑</td><td>Image Acc. ↑ Overall ↑</td><td></td><td>Acc. ↑</td></tr><tr><td rowspan="10">LLaVA-v1.5-7B [25]</td><td>Base</td><td>52.7</td><td>28.0</td><td>8.4</td><td>35.5</td><td>4.0</td><td>46.86</td><td>66.8</td><td>31.0</td><td>58.2</td></tr><tr><td>VCD [21]</td><td>51.3</td><td>25.9</td><td>9.1</td><td>39.8</td><td>4.2</td><td>一</td><td>68.7</td><td>29.8</td><td>56.1</td></tr><tr><td>OPERA [19]</td><td>45.3</td><td>22.9</td><td>6.5</td><td>28.5</td><td>3.1</td><td></td><td>68.2</td><td>30.3</td><td>58.2</td></tr><tr><td>DoLa [10]</td><td>44.0</td><td>25.1</td><td>6.2</td><td>27.7</td><td>2.9</td><td></td><td>67.5</td><td>30.8</td><td>56.6</td></tr><tr><td>HA-DPO [52]</td><td>37.0</td><td>20.9</td><td>6.7</td><td>30.9</td><td>3.3</td><td>47.74</td><td>69.7</td><td>30.6</td><td>56.7</td></tr><tr><td>POVID [53]</td><td>33.4</td><td>16.6</td><td>5.3</td><td>28.7</td><td>3.0</td><td>46.59</td><td>68.8</td><td>31.8</td><td>56.6</td></tr><tr><td>CLIP-DPO [33]</td><td></td><td></td><td>3.7</td><td>16.6</td><td>1.3</td><td></td><td>67.6</td><td></td><td>56.4</td></tr><tr><td>RLAIF-V [50]</td><td>7.8</td><td>4.2</td><td>2.8</td><td>15.7</td><td>0.9</td><td>35.43</td><td>68.2</td><td>29.9</td><td>55.1</td></tr><tr><td>TPO [17]</td><td>5.6</td><td>3.2</td><td>3.6</td><td>20.5</td><td>1.6</td><td>40.12</td><td>67.1</td><td>25.7</td><td>55.3</td></tr><tr><td>vanilla-DPO [37] C-DPO [36]</td><td>27.4 5.9</td><td>15.9 3.3</td><td>5.4 3.0</td><td>25.5 14.9</td><td>2.5 1.3</td><td>47.83 47.48</td><td>69.3 69.4</td><td>32.0 33.4</td><td>58.2</td></tr><tr><td>C2-DPO</td><td></td><td></td><td></td><td></td><td>13.8 1.2</td><td></td><td>47.03</td><td></td><td>33.4</td><td>58.2 58.2</td></tr><tr><td rowspan="4">Qwen2-VL-Instruct-2B [45]</td><td></td><td>4.8</td><td>2.7 8.3</td><td>2.8</td><td>35.5</td><td>2.7</td><td>51.11</td><td>69.5 76.9</td><td></td><td></td></tr><tr><td>Base</td><td>16.1 6.0</td><td>3.6</td><td>6.2 4.2</td><td>39.9</td><td>3.2</td><td>51.37</td><td>77.0</td><td>49.9 49.8</td><td>78.2</td></tr><tr><td>vanilla-DPO [37]</td><td>2.5</td><td>1.6</td><td>2.7</td><td>17.5</td><td>0.8</td><td>51.55</td><td>77.3</td><td>48.2</td><td>78.4 78.4</td></tr><tr><td>C-DPO [36] C2-DPO</td><td>1.6</td><td>1.0</td><td>2.7</td><td>16.1</td><td>0.9</td><td>52.08</td><td>77.2</td><td>49.1</td><td>78.4</td></tr></table>

Table 2: Ablation on loss components. We ablate the contribution of each loss component in C<sup>2</sup>-DPO.
<table><tr><td rowspan="2">Model</td><td colspan="3">Method</td><td colspan="2">Object HalBench</td><td colspan="3">AMBER</td></tr><tr><td></td><td> $\mathcal { L } _ { \mathrm { D P O } } ( x ) ~ \mathcal { L } _ { c } ( x , x ^ { \prime } ) ~ \mathcal { L } _ { \mathrm { D P O } } ( x ^ { \prime } )$ </td><td></td><td>Rsp. ↓</td><td>Men. ↓</td><td>CHAIR ↓ Hal. ↓ Cog. ↓</td><td></td><td></td></tr><tr><td rowspan="4">LLaVA-v1.5-7B</td><td>√</td><td></td><td></td><td>5.9</td><td>3.3</td><td>3.0</td><td>14.9</td><td>1.3</td></tr><tr><td>√</td><td></td><td>√</td><td>7.1</td><td>4.1</td><td>3.1</td><td>15.9</td><td>1.2</td></tr><tr><td>√</td><td>√</td><td></td><td>5.9</td><td>3.2</td><td>3.1</td><td>15.0</td><td>1.3</td></tr><tr><td>√</td><td>√</td><td>√</td><td>4.8</td><td>2.7</td><td>2.8</td><td>13.8</td><td>1.2</td></tr><tr><td rowspan="4">Qwen2-VL-Instruct-2B</td><td>√</td><td></td><td></td><td>2.5</td><td>1.6</td><td>2.7</td><td>17.5</td><td>0.8</td></tr><tr><td>√</td><td></td><td>√</td><td>3.3</td><td>2.6</td><td>2.7</td><td>22.5</td><td>1.1</td></tr><tr><td>√</td><td>√</td><td></td><td>3.5</td><td>2.3</td><td>3.2</td><td>18.6</td><td>1.0</td></tr><tr><td>√</td><td>√</td><td>√</td><td>1.6</td><td>1.0</td><td>2.7</td><td>16.1</td><td>0.9</td></tr></table>

TextVQA. Overall, these results highlight that $\mathrm { C ^ { 2 } { - } D P O }$ provides a scalable and model-agnostic method for mitigating object hallucination in MLLMs.

## 5.3 Ablations and Further Analysis

In this section, we provide a comprehensive analysis of C<sup>2</sup>-DPO through ablations and additional studies. We first examine the contribution of each loss component, and then show that our contextual preference calibration generalizes beyond DPO to variants such as SimPO [29] and RDPO [35], and even to text-only LLM settings. We further analyze the learning dynamics of CPG and its behavior under varying context lengths, assess robustness to noisy context, and study the sensitivity to loss coeficients, collectively highlighting the robustness and versatility of our approach.

Table 3: Efectiveness of combining the proposed Contextual Preference Calibration with SimPO and RDPO. The results show that Contextual Preference Calibration generalizes well across preference optimization variants efectively.
<table><tr><td rowspan="2">Method</td><td colspan="2">Object HalBench</td><td colspan="2">AMBER</td></tr><tr><td>Rsp. ↓</td><td>Men. ↓</td><td>CHAIR ↓Hal. ↓Cog. ↓</td><td></td></tr><tr><td>SimPO</td><td>3.9</td><td>2.6</td><td>2.9 17.8</td><td>1.0</td></tr><tr><td>C²-SimPO</td><td>3.3</td><td>2.0</td><td>3.0</td><td>14.4 0.8</td></tr><tr><td>RDPO</td><td>3.4</td><td>2.3</td><td>2.9</td><td>38.7 2.5</td></tr><tr><td>C2-RDPO</td><td>1.7</td><td>1.2</td><td>2.5</td><td>29.0 1.6</td></tr></table>

Table 4: Performance on AlpacaEval 2 for LLMs. Applying our Contextual Preference Calibration consistently improves both DPO and SimPO.
<table><tr><td rowspan="2">Method</td><td colspan="2">AlpacaEval 2 [13]</td></tr><tr><td>LC (%) ↑ WR (%) ↑</td><td></td></tr><tr><td>DPO</td><td>24.1</td><td>27.4</td></tr><tr><td>C2-DPO</td><td>25.9</td><td>29.7</td></tr><tr><td>SimPO</td><td>33.5</td><td>36.5</td></tr><tr><td>C²-SimPO</td><td>34.1</td><td>37.4</td></tr></table>

Ablation on loss components. We analyze how the two components of the $\mathrm { C ^ { 2 } \mathrm { - } }$ DPO objective, $ { \mathcal { L } } _ { c } ( x , x ^ { \prime } )$ and $\mathcal { L } _ { \mathrm { D P O } } ( x ^ { \prime } )$ , afect hallucination on Object HalBench and AMBER. As shown in Table 2, optimizing either component alone leads to weaker performance. In contrast, jointly optimizing both losses produces a clear synergistic efect, yielding substantial improvements over the original baseline.

Efects on diferent DPO variants. To examine whether the Contextual Preference Calibration approach is specific to DPO or can generalize to various preference optimization schemes, we apply our formulation to two representative alternatives: SimPO [29] and RDPO [35]. As shown in Table 3, incorporating our formulation—denoted as C<sup>2</sup>-SimPO and $\mathrm { C ^ { 2 } { - } R D P O }$ —consistently reduces hallucination rates. These results indicate that contextual preference gain is orthogonal to the underlying preference-learning objective and can be seamlessly integrated with diferent optimization recipes. This general applicability suggests our method provides a broadly compatible improvement to preference-based multimodal alignment, rather than being limited to a DPO-specific remedy.

Applicability on text-only LLMs. To assess whether our Contextual Preference Calibration (CPC) strategy remains efective in text-only LLM settings, we conduct experiments following the experimental protocol of SimPO [29]. Using Qwen2.5-Instruct-1.5B [45], we assess performance on the widely used instruction-following benchmark, AlpacaEval 2 [13]. To adapt CPC to the textonly scenario, we instantiate it following $\operatorname { E q }$ . 10 by replacing both x and $x ^ { \prime }$ using only the query, i.e., $x = q , x ^ { \prime } = \emptyset$ . As shown in Table 4, CPC yields consistent improvements across both DPO and SimPO-based models, highlighting the versatility of our formulation and suggesting that CPC provides a unified and efective strategy for improving reliability in preference optimization.

Learning dynamics of contextual preference gain. Figure 5 illustrates how Contextual Preference Gain (CPG) evolves during training for C<sup>2</sup>-DPO compared to the baseline, C-DPO. C-DPO exhibits near-zero or even negative CPG throughout training, indicating that it fails to leverage contextual signals. In contrast, C<sup>2</sup>-DPO shows a clear and steady increase in CPG. This pattern suggests that C<sup>2</sup>-DPO becomes progressively more sensitive to input context during preference optimization, which in turn enables the model to better align its choices with the provided context and leads to reduced object hallucination. Contextual preference gain according to the context lengths. Figure 6 illustrates how Contextual Preference Gain (CPG) evolves as the amount of nonhallucinated contextual information increases. The top figure reports the average CPG as a function of the number of sentences added to the input. We observe a clear upward trend: CPG steadily increases with more sentences, indicating that richer contextual grounding amplifies the preference diference between grounded and hallucinated responses. The bottom figure provides a complementary analysis at the word level, showing that CPG also grows consistently as additional words are incorporated. Together, these results demonstrate that stronger contextual signals yield greater preference separation, highlighting the sensitivity of CPG to the granularity and richness of added context.

![](images/36ae4816acf0070db1e13ac296a19053d493f43d4a74babb4be8a8014a881f42.jpg)

![](images/26918ddc1e2d9092dfc4173e3db2ddad5f4581c84cc3d457330de368a7974708.jpg)

Fig. 5: Contextual Preference Gain (CPG) comparison between C<sup>2</sup>-DPO and C-DPO. Our C<sup>2</sup>-DPO steadily increases CPG throughout training, while C-DPO stays near or below zero, indicating its limited ability to utilize contextual signals.  
![](images/22d41bc830558892c2fe99816d1690aa646566901a985edbb7b6e7ce7ebaa161.jpg)  
Fig. 6: Average Contextual Preference Gain (CPG) at the sentence (top) and word (bottom) level. CPG increases consistently as more relevant context is added, indicating that richer contextual signals lead to stronger preference separation.

Robustness to noisy context. We further evaluate robustness when the auxiliary context is noisy. During training, we randomly mask the auxiliary captions c with probability $p \in \{ 0 . 0 , 0 . 3 , 0 . 5 \}$ . Table 5 shows that $\mathrm { C ^ { 2 } { - } D P O }$ consistently achieves lower hallucination rates than C-DPO across all masking levels on Object HalBench. Importantly, the performance gap widens as the masking probability increases, suggesting that contextual preference calibration enables the model to remain robust even when contextual information is partially corrupted. Sensitivity analysis on loss coeficients. Table 6 presents the sensitivity analysis for $\lambda _ { c }$ and $\lambda _ { u } ,$ which control the strength of $ { \mathcal { L } } _ { c } ( x , x ^ { \prime } )$ and $\mathcal { L } _ { \mathrm { D P O } } ( x ^ { \prime } )$ The results show that the losses are most efective with weights in the 0.3 to 0.5 range, while 0.1 provides comparatively weaker influence. This narrow region of strong performance remains consistent across both models, indicating that C<sup>2</sup>-DPO works reliably without extensive hyperparameter tuning.

Table 5: Robustness to context quality.
<table><tr><td rowspan="2">Method Noise p</td><td colspan="2">Object HalBench</td></tr><tr><td>Rsp. ↓</td><td>Men. ↓</td></tr><tr><td>C-DPO C²-DPO</td><td>2.5</td><td>1.6</td></tr><tr><td>C-DPO</td><td>1.6</td><td>1.0</td></tr><tr><td>0.3 C²-DPO</td><td>9.6 8.5</td><td>5.2</td></tr><tr><td></td><td></td><td>5.0</td></tr><tr><td>C-DPO 0.5</td><td>13.0</td><td>7.2</td></tr><tr><td>C²-DPO</td><td>8.4</td><td>4.6</td></tr></table>

Table 6: Sensitivity analysis on loss coefficients.
<table><tr><td rowspan="2">Model</td><td rowspan="2">λ</td><td rowspan="2">Value</td><td colspan="2">Object HalBench</td></tr><tr><td>Rsp. ↓</td><td>Men. ↓</td></tr><tr><td rowspan="5">LLaVA-v1.5-7B</td><td rowspan="3">λc</td><td>0.1</td><td>7.4</td><td>4.3</td></tr><tr><td>0.3</td><td>4.8</td><td>2.7</td></tr><tr><td>0.5</td><td>5.5</td><td>3.1</td></tr><tr><td rowspan="3">λu</td><td>0.1</td><td>5.5</td><td>3.2</td></tr><tr><td>0.3</td><td>5.6</td><td>3.1</td></tr><tr><td>0.5</td><td>4.8</td><td>2.7</td></tr><tr><td rowspan="5">Qwen2-VL-Instruct-2B</td><td rowspan="3">λc</td><td>0.1</td><td>4.1</td><td>2.6</td></tr><tr><td>0.3</td><td>2.0</td><td>1.3</td></tr><tr><td>0.5</td><td>1.6</td><td>1.0</td></tr><tr><td rowspan="3"> $\lambda _ { u }$ </td><td>0.1</td><td>2.4</td><td>1.5</td></tr><tr><td>0.3</td><td>1.6</td><td>1.0</td></tr><tr><td>0.5</td><td>2.0</td><td>1.3</td></tr></table>

## 5.4 Qualitative Analysis

Figure 7 shows qualitative comparisons between the baseline model and ours. The baseline exhibits hallucinations (“a hot dog with mustard”). Although it correctly identifies the object as a hot dog, it mistakenly describes it as having mustard, revealing a failure to attend to fine-grained visual details. In contrast, C<sup>2</sup>-DPO produces an accurate description (“the hot dog is covered in chili”), showing that it inspects such details more carefully. It also identifies additional cues, such as a “bottle of water”, which the baseline overlooks. This reflects our objective, which calibrates preference toward grounded evidence rather than frequent co-occurrences, a common source of hallucination.

Figure 8 further illustrates how each model uses contextual information provided alongside the image. CPG reveals this directly: the baseline C-DPO shows a low score, indicating that it underutilizes the key cue $( e . g . , \mathrm { \bar { \it ~ a ~ } }$ bus stop on the left side of the image”). In contrast, our C<sup>2</sup>-DPO attains a much higher CPG score, correctly referencing “the bus $s t o p ^ { \gamma }$ and grounding its response in the described surroundings. Here CPG tracks observable behavior: its higher value coincides with explicit use of the caption, illustrating at the instance level what Section 4.2 shows in aggregate. By enabling stronger context-aware reasoning, C<sup>2</sup>-DPO mitigates hallucinations more faithfully in multimodal understanding.

## 6 Conclusion

In this work, we investigated whether existing preference optimization methods truly leverage contextual information for grounding, and found that they do not. To quantify this, we introduced Contextual Preference Gain (CPG), which revealed two key findings: CPG is strongly and negatively correlated with hallucination rate, and existing DPO-style methods exhibit context-blind behavior with CPG concentrated near zero. Motivated by this, we proposed Context-Calibrated DPO (C<sup>2</sup>-DPO), which directly maximizes CPG through a simple contrastive regularization while preserving correct preference ordering under degraded context. Extensive experiments demonstrate that C<sup>2</sup>-DPO substantially reduces object hallucination without compromising general reasoning ability, and generalizes to other preference optimization variants and text-only settings. Beyond object hallucination, our results suggest that explicitly modeling how context shapes preference is a broadly useful principle for preference optimization. As C<sup>2</sup>-DPO relies only on the contrast between a full and a degraded input, it extends naturally to other modalities and tasks where context can be defined, which we leave for future work.

![](images/3057dd4b41382e22751c9b514cb46698567ad2731f01e4b36b7a1cbc8a978667.jpg)  
Fig. 7: Qualitative example on detailed captioning. We compare the outputs from models trained with the baseline and our $\mathrm { C ^ { 2 } { - } D P O }$ . Red, green, and blue denote hallucinated details, correct details, and additional correct details captured only by our model, respectively. Visualization is done on AMBER benchmark with LLaVA-1.5-7B.

![](images/984cc1100d856a9e1154430e0d25c0004bdb72faa8f29fca906221cd9b87ba36.jpg)  
Fig. 8: Qualitative comparison of context utilization between our $\mathbf { C } ^ { 2 } \mathbf { - D P O }$ and the baseline C-DPO. Yellow and green highlights mark context-relevant details from the provided caption that are correctly reflected in the positive response. The bar chart (right) reports the corresponding Contextual Preference Gain (CPG), which is substantially higher for C<sup>2</sup>-DPO.

Acknowledgment. This work was supported by the InnoCORE program of the Ministry of Science and ICT (N10250156, 30%), Institute of Information & communications Technology Planning & Evaluation(IITP) grant funded by the Korean government(MSIT) (RS-2025-25442149, LG AI STAR Talent Development Program for Leading Large-Scale Generative AI Models in the Physical AI Domain, 30%), and the Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korean government(MSIT) (No. RS-2024-00457882, AI Research Lab Project, 40%).

## References

1. Agrawal, H., Desai, K., Wang, Y., Chen, X., Jain, R., Johnson, M., Batra, D., Parikh, D., Lee, S., Anderson, P.: Nocaps: Novel object captioning at scale. In: ICCV (2019)

2. Anthropic: Claude 3.7 sonnet. https://www.anthropic.com/news/claude-3-7- sonnet (2025)

3. Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C.L., Parikh, D.: Vqa: Visual question answering. In: ICCV (2015)

4. Azar, M.G., Guo, Z.D., Piot, B., Munos, R., Rowland, M., Valko, M., Calandriello, D.: A general theoretical paradigm to understand learning from human preferences. In: AISTATS (2024)

5. Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., et al.: Qwen technical report. arXiv:2309.16609 (2023)

6. Bradley, R.A., Terry, M.E.: Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika (1952)

7. Burgin, M.: The essence of information: Paradoxes, contradictions, and solutions. In: Electronic Conference on Foundations of Information Science: The nature of information: Conceptions, misconceptions, and paradoxes (FIS 2002). Retrieved September (2002)

8. Chen, Z., Zhao, Z., Luo, H., Yao, H., Li, B., Zhou, J.: Halc: Object hallucination reduction via adaptive focal-contrast decoding. In: ICML (2024)

9. Christiano, P.F., Leike, J., Brown, T., Martic, M., Legg, S., Amodei, D.: Deep reinforcement learning from human preferences. In: NeurIPS (2017)

10. Chuang, Y.S., Xie, Y., Luo, H., Kim, Y., Glass, J., He, P.: Dola: Decoding by contrasting layers improves factuality in large language models. In: ICLR (2024)

11. Cui, G., Yuan, L., Ding, N., Yao, G., He, B., Zhu, W., Ni, Y., Xie, G., Xie, R., Lin, Y., et al.: Ultrafeedback: Boosting language models with scaled ai feedback. In: ICML (2024)

12. Dai, W., Li, J., Li, D., Tiong, A., Zhao, J., Wang, W., Li, B., Fung, P.N., Hoi, S.: Instructblip: Towards general-purpose vision-language models with instruction tuning. In: NeurIPS (2023)

13. Dubois, Y., Galambosi, B., Liang, P., Hashimoto, T.B.: Length-controlled alpacaeval: A simple way to debias automatic evaluators. In: COLM (2024)

14. Google DeepMind: Gemini 2.5. https://blog.google/technology/googledeepmind/gemini-model-thinking-updates-march-2025/ (2025)

15. Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., Parikh, D.: Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In: CVPR (2017)

16. Guan, T., Liu, F., Wu, X., Xian, R., Li, Z., Liu, X., Wang, X., Chen, L., Huang, F., Yacoob, Y., et al.: Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In: CVPR (2024)

17. He, L., Chen, Z., Shi, Z., Yu, T., Shao, J., Sheng, L.: A topic-level self-correctional approach to mitigate hallucinations in mllms. arXiv:2411.17265 (2024)

18. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. In: ICLR (2022)

19. Huang, Q., Dong, X., Zhang, P., Wang, B., He, C., Wang, J., Lin, D., Zhang, W., Yu, N.: Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation. In: CVPR (2024)

20. Kuznetsova, A., Rom, H., Alldrin, N., Uijlings, J., Krasin, I., Pont-Tuset, J., Kamali, S., Popov, S., Malloci, M., Kolesnikov, A., et al.: The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. IJCV (2020)

21. Leng, S., Zhang, H., Chen, G., Li, X., Lu, S., Miao, C., Bing, L.: Mitigating object hallucinations in large vision-language models through visual contrastive decoding. In: CVPR (2024)

22. Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., et al.: Llava-onevision: Easy visual task transfer. TMLR (2024)

23. Li, X.L., Holtzman, A., Fried, D., Liang, P., Eisner, J., Hashimoto, T.B., Zettlemoyer, L., Lewis, M.: Contrastive decoding: Open-ended text generation as optimization. In: ACL (2023)

24. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: ECCV (2014)

25. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. In: CVPR (2024)

26. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. In: NeurIPS (2023)

27. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: ICLR (2017)

28. Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.W., Zhu, S.C., Tafjord, O., Clark, P., Kalyan, A.: Learn to explain: Multimodal reasoning via thought chains for science question answering. In: NeurIPS (2022)

29. Meng, Y., Xia, M., Chen, D.: Simpo: Simple preference optimization with a reference-free reward. In: NeurIPS (2024)

30. Meta AI: Meta-ai. the llama 4 herd: The beginning of a new era of natively multimodal ai innovation. https://ai.meta.com/blog/llama- 4- multimodalintelligence/ (2025)

31. OpenAI: Gpt-4o mini: advancing cost-eficient intelligence. https://openai.com/ index/gpt-4o-mini-advancing-cost-efficient-intelligence/ (2024)

32. OpenAI: Hello gpt-4o. https://openai.com/index/hello-gpt-4o/ (2024)

33. Ouali, Y., Bulat, A., Martinez, B., Tzimiropoulos, G.: Clip-dpo: Vision-language models as a source of preference for fixing hallucinations in lvlms. In: ECCV (2024)

34. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.: Training language models to follow instructions with human feedback. In: NeurIPS (2022)

35. Park, R., Rafailov, R., Ermon, S., Finn, C.: Disentangling length from quality in direct preference optimization. In: ACL Findings (2024)

36. Peng, S., Yang, S., Jiang, L., Tian, Z.: Mitigating object hallucinations via sentence-level early intervention. In: ICCV (2025)

37. Rafailov, R., Sharma, A., Mitchell, E., Manning, C.D., Ermon, S., Finn, C.: Direct preference optimization: Your language model is secretly a reward model. In: NeurIPS (2023)

38. Rohrbach, A., Hendricks, L.A., Burns, K., Darrell, T., Saenko, K.: Object hallucination in image captioning. In: EMNLP (2018)

39. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.: Proximal policy optimization algorithms. arXiv:1707.06347 (2017)

40. Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., Rohrbach, M.: Towards vqa models that can read. In: CVPR (2019)

41. Soni, J., Goodman, R.: A mind at play: how Claude Shannon invented the information age. Simon and Schuster (2017)

42. Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., Christiano, P.F.: Learning to summarize with human feedback. In: NeruIPS (2020)

43. Wang, H., Xiong, W., Xie, T., Zhao, H., Zhang, T.: Interpretable preferences via multi-objective reward modeling and mixture-of-experts. arXiv:2406.12845 (2024)

44. Wang, J., Wang, Y., Xu, G., Zhang, J., Gu, Y., Jia, H., Wang, J., Xu, H., Yan, M., Zhang, J., et al.: Amber: An llm-free multi-dimensional benchmark for mllms hallucination evaluation. arXiv:2311.07397 (2023)

45. Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al.: Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv:2409.12191 (2024)

46. Wu, J., Wang, X., Yang, Z., Wu, J., Gao, J., Ding, B., Wang, X., He, X.: Alphadpo: Adaptive reward margin for direct preference optimization. In: ICML (2025)

47. Xiao, T., Yuan, Y., Zhu, H., Li, M., Honavar, V.G.: Cal-dpo: Calibrated direct preference optimization for language model alignment. In: NeurIPS (2024)

48. Xu, H., Sharaf, A., Chen, Y., Tan, W., Shen, L., Van Durme, B., Murray, K., Kim, Y.J.: Contrastive preference optimization: Pushing the boundaries of llm performance in machine translation. In: ICML (2024)

49. Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al.: Qwen3 technical report. arXiv:2505.09388 (2025)

50. Yu, T., Zhang, H., Li, Q., Xu, Q., Yao, Y., Chen, D., Lu, X., Cui, G., Dang, Y., He, T., et al.: Rlaif-v: Open-source ai feedback leads to super gpt-4v trustworthiness. In: CVPR (2025)

51. Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang, X., Wang, L.: Mm-vet: Evaluating large multimodal models for integrated capabilities. In: ICML (2024)

52. Zhao, Z., Wang, B., Ouyang, L., Dong, X., Wang, J., He, C.: Beyond multimodal hallucinations: Enhancing lvlms through hallucination-aware direct preference optimization. In: ICME (2025)

53. Zhou, Y., Cui, C., Rafailov, R., Finn, C., Yao, H.: Aligning modalities in vision large language models via preference fine-tuning. arXiv:2402.11411 (2024)

54. Zhou, Y., Cui, C., Yoon, J., Zhang, L., Deng, Z., Finn, C., Bansal, M., Yao, H.: Analyzing and mitigating object hallucination in large vision-language models. In: ICLR (2024)

## Appendix

## A Gradient Analysis

In this section, we analyze the gradient of the contextual preference calibration loss $\mathcal { L } _ { c }$ to characterize its training dynamics. We first show that the gradient factorizes into a calibration direction and an adaptive per-sample weight, and then demonstrate that the resulting update admits an information-theoretic interpretation in terms of the pointwise mutual information (PMI) between a response and the auxiliary context.

Gradient decomposition and training dynamics. By the definition of CPG in Eq. (5), the calibration loss in Eq. (8) can be written compactly as $\begin{array} { r l } { \mathcal { L } _ { c } ( x , x ^ { \prime } ) = } \end{array}$ log $\overset { \cdot } { \sigma } ( \mathrm { C P G } ( x , x ^ { \prime } ) )$ . Diferentiating with respect to the model parameters θ yields

$$
\begin{array} { r } { \nabla _ { \theta } \mathcal { L } _ { c } ( x , x ^ { \prime } ) = - \underbrace { \sigma ( - \mathrm { C P G } ( x , x ^ { \prime } ) ) } _ { \mathrm { a d a p t i v e ~ w e i g h t } } \cdot \underbrace { \nabla _ { \theta } \mathrm { C P G } ( x , x ^ { \prime } ) } _ { \mathrm { c a l i b r a t i o n ~ d i r e c t i o n } } . } \end{array}\tag{10}
$$

The two factors play distinct roles. The calibration direction $\nabla _ { \theta } \mathrm { C P G } ( x , x ^ { \prime } )$ increases the preference margin under the full context x while decreasing it under the degraded context $x ^ { \prime } { \mathrm { . } }$ so ascent along this direction directly maximizes CPG. The scalar weight $\sigma ( - \mathrm { C P G } ( x , x ^ { \prime } ) ) \in ( 0 , 1 )$ adjusts the magnitude of the update for each sample. It stays close to 1 when the $\mathrm { C P G }$ is small or negative, which corresponds to the context-blind regime identified in Section 4.2, and becomes small once the CPG grows large. As a result, $\mathcal { L } _ { c }$ focuses the training signal on samples where the model fails to exploit the auxiliary context, and its influence gradually vanishes as calibration improves. A similar weighting appears in the standard DPO gradient, where the update is scaled by how strongly the model currently violates the target ordering.

Connection to pointwise mutual information. We next show that the calibration direction can be understood in information-theoretic terms. Let $\boldsymbol { x } = ( v , q , c )$ be the full input and $\boldsymbol { x } ^ { \prime } = ( \boldsymbol { v } , \boldsymbol { q } )$ its context-degraded counterpart. For a response $y ,$ the conditional PMI between y and the context c under the policy $\pi _ { \theta }$ is defined as

$$
\operatorname { p m i } _ { \theta } ( y ; c \mid v , q ) = \log { \frac { \pi _ { \theta } ( y , c \mid v , q ) } { \pi _ { \theta } ( y \mid v , q ) \pi _ { \theta } ( c \mid v , q ) } } = \log { \frac { \pi _ { \theta } ( y \mid x ) } { \pi _ { \theta } ( y \mid x ^ { \prime } ) } } ,\tag{11}
$$

where the second equality holds because the joint distribution factorizes as $\pi _ { \boldsymbol { \theta } } ( y , c \mid v , q ) = \pi _ { \boldsymbol { \theta } } ( y \mid v , q , c ) \pi _ { \boldsymbol { \theta } } ( c \mid v , q ) .$ , and $\pi _ { \boldsymbol { \theta } } ( c \mid v , q )$ denotes the likelihood the model assigns to the context given the image and query. In other words, the PMI is the log-likelihood ratio of the response with and without the auxiliary context, and it measures how much conditioning on c raises the log-probability of $y .$

For brevity, we write $\varDelta _ { \theta } ( x )$ for the preference score $\varDelta \hat { r } _ { \theta } ( x , y _ { w } , y _ { l } )$ in Eq. (4). Substituting the implicit reward $\begin{array} { r } { \hat { r } _ { \theta } ( x , y ) \ = \ \beta \log \frac { \pi _ { \theta } ( y | x ) } { \pi _ { \mathrm { r e f } } ( y | x ) } } \end{array}$ into $\mathrm { C P G } ( x , x ^ { \prime } ) =$ $\varDelta _ { \theta } ( x ) - \varDelta _ { \theta } ( x ^ { \prime } )$ , grouping the log-probability terms by response, and applying Eq. (11), we obtain

$$
\mathrm { C P G } ( x , x ^ { \prime } ) = \beta \underbrace { \left[ \mathrm { p m i } _ { \theta } ( y _ { w } ; c \mid v , q ) - \mathrm { p m i } _ { \theta } ( y _ { l } ; c \mid v , q ) \right] } _ { \triangleq \boldsymbol { \mathrm { A p m i } } _ { \theta } } - \beta \ \mathrm { \varDelta p m i } _ { \mathrm { r e f } } ,\tag{12}
$$

where $\varDelta \mathrm { p m i } _ { \mathrm { r e f } }$ is the same PMI gap computed under the reference model. This term is constant in $\theta$ since $\pi _ { \mathrm { r e f } }$ is frozen. Combining Eq. (12) with Eq. (10) then gives

$$
\nabla _ { \theta } \mathcal { L } _ { c } ( x , x ^ { \prime } ) = - \beta \sigma ( - \mathrm { C P G } ( x , x ^ { \prime } ) ) \nabla _ { \theta } \left[ \mathrm { p m i } _ { \theta } ( y _ { w } ; c \mid v , q ) - \mathrm { p m i } _ { \theta } ( y _ { l } ; c \mid v , q ) \right] .\tag{13}
$$

Interpretation. Equation (13) shows that each gradient step on $\mathcal { L } _ { c }$ raises the PMI between the context and the preferred response while lowering it for the dispreferred one. Since the per-sample PMI is the pointwise integrand of the conditional mutual information $I _ { \theta } ( Y ; C \mid V , Q ) = \operatorname { \mathbb { E } } [ \operatorname { p m i } _ { \theta } ( y ; c \mid v , q ) ]$ ], minimizing $\mathcal { L } _ { c }$ amounts to optimizing a contrastive surrogate of this quantity. Rather than maximizing the expected PMI itself, which would encourage dependence on the context regardless of response quality, the loss widens the PMI gap between $y _ { w }$ and $y _ { l }$ and thus makes the context informative specifically about the preferred response. Standard DPO behaves diferently. It maximizes a likelihoodratio margin at a fixed context, so it is agnostic to how much of that margin comes from $c ,$ and its loss can in fact be driven to zero without the context contributing to the preference at all. This observation is consistent with the near-zero CPG distributions reported in Section 4.2.

## B Efect of Alternative Context Degradations

The main paper constructs the degraded input by removing the auxiliary caption, which corresponds to $x _ { \mathrm { n o - c a p } } ^ { \prime } = ( v , q , \emptyset )$ . In this section, we examine whether the proposed Contextual Preference Calibration remains efective when the degraded input is constructed diferently. We consider three alternatives. The first variant $x _ { \mathrm { n o - i m g } } ^ { \prime } \ = \ ( \emptyset , q , c )$ removes the image while keeping the caption. The second variant $x _ { \mathrm { n o i s y - i m g } } ^ { \prime } = ( \tilde { v } , q , c )$ replaces the image v with v˜, a copy corrupted by additive Gaussian noise. The third variant $x _ { \mathrm { r a n d - c a p } } ^ { \prime } = ( v , q , \tilde { c } )$ replaces the caption with one randomly drawn from another training sample. All experiments use Qwen2-VL-Instruct-2B [45] as the base model.

Table A reports the results. On Object HalBench [38], every variant clearly outperforms the baseline C-DPO [36], which indicates that the benefit of Contextual Preference Calibration does not hinge on one particular form of degradation. On AMBER [44], however, the caption-side variants $x _ { \mathrm { r a n d - c a p } } ^ { \prime }$ and $x _ { \mathrm { n o - c a p } } ^ { \prime }$ yield larger improvements than the image-side variants $x _ { \mathrm { n o i s y - i m g } } ^ { \prime }$ and $x _ { \mathrm { n o - i m g } } ^ { \prime } .$ We attribute this gap to the asymmetric roles of the two inputs. The image provides the primary grounding context while the auxiliary caption serves as a secondary one, so degrading the image removes the very signal that responses should be grounded in and weakens the calibration contrast. Degrading the caption instead isolates the contribution of the auxiliary context and preserves the image as a stable anchor, which explains why $x _ { \mathrm { n o - c a p } } ^ { \prime }$ achieves the best overall results and supports our design choice in the main paper.

Table A: $\mathbf { C } ^ { 2 } \mathbf { - D P O }$ performance under alternative context removal.
<table><tr><td rowspan="2">Method</td><td colspan="3">Object HalBench AMBER</td></tr><tr><td>Rsp. ↓</td><td>Men. ↓</td><td>Hal. ↓</td></tr><tr><td>Base</td><td>16.1</td><td>8.3</td><td>35.5</td></tr><tr><td>vanilla-DPO</td><td>6.0</td><td>3.6</td><td>39.9</td></tr><tr><td>C-DPO</td><td>2.5</td><td>1.6</td><td>17.5</td></tr><tr><td>C2-DPO w/  $x _ { \mathrm { n o i s y - i m g } } ^ { \prime }$ </td><td>1.7</td><td>1.1</td><td>18.1</td></tr><tr><td>C2-DPO w/  $x _ { \mathrm { n o - i m g } } ^ { \prime }$ </td><td>1.6</td><td>1.1</td><td>18.2</td></tr><tr><td>C2-DPO w/  $x _ { \mathrm { r a n d - c a p } } ^ { \prime }$ </td><td>1.6</td><td>1.0</td><td>17.1</td></tr><tr><td>C2-DPO w/  $x _ { \mathrm { n o - c a p } } ^ { \prime }$ </td><td>1.6</td><td>1.0</td><td>16.1</td></tr></table>

## C Training Details

In this section, we provide the details for reproducing our full training pipeline.

## C.1 Training Dataset

We use the SENTINEL preference dataset introduced in C-DPO [36], which provides in-domain, context-aware preference data specifically designed for mitigating object hallucination in MLLMs. Unlike synthetic or heavily engineered preference collections, SENTINEL constructs discriminative preference data by automatically generating non-hallucinated auxiliary context, which enables the model to distinguish non-hallucinated responses from hallucinated ones.

Each data sample consists of $\left( v , q , c , y _ { w } , y _ { l } \right)$ , where v is the image, q the query, c an auxiliary non-hallucinated visual description, and $( y _ { w } , y _ { l } )$ denote the preferred and dispreferred responses. Following prior work, we use the dataset exactly as released without any modification, which ensures a fair comparison with existing methods. The dataset contains approximately 8.6k preference pairs for LLaVA-v1.5-7B [25] and 12k pairs for Qwen2-VL-Instruct-2B [45].

## C.2 Implementation Details

All models are trained for one epoch using LoRA-based preference optimization. Table B summarizes the full set of optimization and calibration hyperparameters

used across all experiments. Training is performed on 4 NVIDIA RTX 4090 GPUs using PyTorch 2.5.1 with CUDA 12.1, and ZeRO stage-2 is enabled for memory-eficient distributed execution.

Table B: Training hyperparameters used in our experiments.
<table><tr><td>LLaVA-v1.5-7B</td><td>Qwen2-VL-Instruct-2B</td></tr><tr><td>Global batch size</td><td>64</td></tr><tr><td>Learning rate</td><td>2e-6</td></tr><tr><td>Learning rate scheduler</td><td>cosine</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Weight decay</td><td>0</td></tr><tr><td>Epochs</td><td>1</td></tr><tr><td>Trainable parameters</td><td>LLM linear layers only</td></tr><tr><td>LoRA r</td><td>128</td></tr><tr><td>LoRA α</td><td>256</td></tr><tr><td>Memory optimization</td><td>ZeRO stage 2</td></tr><tr><td> $\beta$ </td><td>0.1</td></tr><tr><td></td><td>0.5</td></tr><tr><td> $\lambda _ { c }$   $\lambda _ { u }$ </td><td></td></tr></table>

## C.3 Algorithmic Description

For clarity and reproducibility, we provide the full pseudo-code of our $\mathrm { C ^ { 2 } { - } D P O }$ training procedure in Algorithm 1. The algorithm summarizes how full and degraded contexts are used, how preference scores are computed, and how the combined objective is optimized.

## D Experimental Setup for CPG Analysis

For the CPG analysis in Section 4.2, we train DPO, SimPO, and RDPO [29,35, 37] on the SENTINEL dataset from C-DPO [36], using the Qwen2-VL-Instruct-2B [45] model as the base model. Each method is trained with its own objective, as summarized in Table C.

After training, we compute CPG over 2,000 examples randomly sampled from the training split. For this evaluation, all preference scores are measured with the DPO implicit reward

$$
\hat { r } _ { \theta } ( x , y ) = \beta \log \frac { \pi _ { \theta } ( y \mid x ) } { \pi _ { \mathrm { r e f } } ( y \mid x ) } ,\tag{14}
$$

regardless of the objective used during training. This choice provides a single objective-agnostic measure of preference strength, so the reported CPG diferences reflect what each method learns rather than how its reward is defined.

Algorithm 1 $\mathrm { C ^ { 2 } { - } D P O }$ Training   
Require: Policy model $\pi _ { \theta } ,$ reference model $\pi _ { \mathrm { r e f } }$ (frozen)   
Require: Dataset $\mathcal { D } { = } \{ ( v , q , c , y _ { w } , y _ { l } ) \}$   
Require: Hyperparameters $\beta , \ \lambda _ { c } , \ \lambda _ { u }$   
1: for each minibatch from D do   
2: Compute preference scores under $x = ( v , q , c )$ and $x ^ { \prime } = ( v , q , \emptyset ) ;$   
$\begin{array} { r } { \Delta \hat { r } _ { \theta } ( x , y _ { w } , y _ { l } ) \gets \beta \log \frac { \pi _ { \theta } ( y _ { w } \mid x ) } { \pi _ { \mathrm { r e f } } ( y _ { w } \mid x ) } - \beta \log \frac { \pi _ { \theta } ( y _ { l } \mid x ) } { \pi _ { \mathrm { r e f } } ( y _ { l } \mid x ) } } \end{array}$   
$\begin{array} { r } { \Delta \hat { r } _ { \theta } ( x ^ { \prime } , y _ { w } , y _ { l } ) \gets \beta \log \frac { \pi _ { \theta } ( y _ { w } \mid x ^ { \prime } ) } { \pi _ { \mathrm { r e f } } ( y _ { w } \mid x ^ { \prime } ) } - \beta \log \frac { \pi _ { \theta } ( y _ { l } \mid x ^ { \prime } ) } { \pi _ { \mathrm { r e f } } ( y _ { l } \mid x ^ { \prime } ) } } \end{array}$   
3: Compute losses:   
$\mathcal { L } _ { \mathrm { D P O } } ( x ) \gets - \log \sigma ( \varDelta \hat { r } _ { \theta } ( x , y _ { w } , y _ { l } ) )$   
$\mathcal { L } _ { c } ( x , x ^ { \prime } ) \gets - \log \sigma \big ( \varDelta \hat { r } _ { \theta } ( x , y _ { w } , y _ { l } ) - \varDelta \hat { r } _ { \theta } ( x ^ { \prime } , y _ { w } , y _ { l } ) \big )$   
$\mathcal { L } _ { \mathrm { D P O } } ( x ^ { \prime } ) \gets - \log \sigma \big ( \Delta \hat { r } _ { \theta } ( x ^ { \prime } , y _ { w } , y _ { l } ) \big )$   
4: Update $\pi \theta$ by minimizing the combined objective:   
$\mathcal { L } _ { \mathrm { C } ^ { 2 } \mathrm { - D P O } } ( x , x ^ { \prime } ) = \mathcal { L } _ { \mathrm { D P O } } ( x ) + \lambda _ { c } \mathcal { L } _ { c } ( x , x ^ { \prime } ) + \lambda _ { u } \mathcal { L } _ { \mathrm { D P O } } ( x ^ { \prime } )$   
5: end for   
6: return π<sub>θ</sub>

## E Evaluation Benchmark Details

We provide a detailed description of the evaluation benchmarks used in our study.

Object HalBench. Object HalBench [38] is a benchmark designed to measure object hallucination in image-captioning tasks for MLLMs. It leverages the image captions and segmentation annotations from the COCO 2014 dataset [24].

AMBER. AMBER [44] is a multi-dimensional benchmark for evaluating hallucination in MLLMs without relying on an external LLM judge. The dataset contains 1,004 images paired with task-specific prompt templates that assess three major hallucination types: existence (object presence), attribute(object state), and relational(object-object relations) hallucinations.

HallusionBench. HallusionBench [16] consists of a total of 1129 questions on diverse topics(food, math, geometry, statistics, etc.) and 455 visual-question control pairs while focusing on evaluating fine-grained visual misinterpretation and reasoning errors. Each sample is annotated with a gold answer and clear hallucination indicator, enabling systematic measurement of whether a model introduces nonexistent objects, misreads attributes, or relies on language priors.

ScienceQA. ScienceQA [28] is a multimodal question-answering benchmark designed to evaluate the scientific reasoning capabilities of MLLMs across diverse domains. The dataset contains over 21,000 multiple-choice questions covering natural science, language science, and social science, with many samples requiring reasoning over diagrams, charts, images, and textual contexts.

Table C: Preference optimization (PO) and Contextual Preference Calibration (CPC) objectives.
<table><tr><td colspan="3">Method Implicit reward êθ(x, y)</td><td>PO Objective LPo(x)</td><td>CPC Objective  $ { \mathcal { L } } _ { c } ( x , x ^ { \prime } )$ </td></tr><tr><td>DPO</td><td></td><td> $\beta \log { \frac { \pi _ { \theta } ( y \mid x ) } { \pi _ { \mathrm { r e f } } ( y \mid x ) } }$ </td><td> $- \log \sigma ( \hat { r } _ { \theta } ( x , y _ { w } ) - \hat { r } _ { \theta } ( x , y _ { l } ) )$ </td><td rowspan="3"></td></tr><tr><td>SimPO</td><td></td><td></td><td> $\begin{array} { r l r l r } { \displaystyle \frac { \beta } { | y | } \log \pi \theta ( y \mid x ) } & { } & & { - \log \sigma \big ( \hat { r } \theta ( x , y _ { w } ) - \hat { r } \theta ( x , y _ { l } ) - \gamma \big ) } & { - \log \sigma \big ( \Delta \hat { r } \theta ( x , y _ { w } , y _ { l } ) - \Delta \hat { r } \theta ( x ^ { ' } , y _ { w } , y _ { l } ) \big ) } \end{array}$ </td></tr><tr><td>RDPO</td><td></td><td> $\beta \log { \frac { \pi _ { \theta } ( y \mid x ) } { \pi _ { \mathrm { r e f } } ( y \mid x ) } } + \alpha | y |$ </td><td>− log σ(τθ(x, yw) − τθ(x, yl))</td></tr></table>

MM-Vet. MM-Vet [51] contains 200 images and 218 questions to evaluate MLLMs on integrated vision-language capabilities, rather than isolated tasks. The benchmark defines six core VL capabilities: recognition, OCR, knowledge, spatial awareness, language generation, and math.

TextVQA. TextVQA [40] is a visual question-answering benchmark focused on scene-text reasoning, where models are required to read and reason about textual content embedded in images to generate correct answers. It comprises 45,336 questions across 28,408 images sourced from the OpenImages dataset [20], filtered to ensure the presence of scene text.

## F Ablation Study Details

This section provides additional details for the ablation experiments reported in Table 3 and Table 4.

Extension to other preference optimization methods. In applying Contextual Preference Calibration to other preference optimization methods, no additional modification is required beyond replacing the components of the original $\mathrm { C ^ { 2 } { - } D P O }$ formulation. Concretely, we take the $\mathrm { C ^ { 2 } { - } D P O }$ objective in Eq. (9) and substitute (i) $\mathcal { L } _ { \mathrm { D P O } }$ with each method’s PO objective, and (ii) the implicit reward used inside $\mathcal { L } _ { c }$ with the corresponding formulation summarized in Table C. Applicability to text-only LLMs. We apply Contextual Preference Calibration to text-only preference optimization by defining the full and degraded contexts as $x = q$ and $x ^ { \prime } = \varnothing$ . Following the $\mathrm { C ^ { 2 } { - } D P O }$ formulation in Eq. (9), we substitute $\mathcal { L } _ { \mathrm { D P O } }$ with each method’s native preference optimization loss and compute $\mathcal { L } _ { c }$ using that method’s implicit reward, as listed in Table C. No further modification is required. Training follows the SimPO [29] protocol, where UltraFeedback [11] is reannotated using ArmoRM [43]. We use Qwen2.5-Instruct-1.5B [45] as the policy model. Evaluation is conducted on AlpacaEval 2 [13] using GPT-4o-mini [31] as the evaluator, and we report raw win rate (WR) and the length-controlled win rate (LC).

## G Additional Qualitative Examples

This section provides additional qualitative examples that complement the analyses in Figure 7 and Figure 8 of the main paper. Figure A expands upon the hallucination mitigation comparison shown in Figure 7, demonstrating how our method improves grounding and reduces both object-level and attribute-level hallucinations across diverse scenarios. Figure B provides extended cases corresponding to the context-utilization analysis in Figure 8, illustrating how changes in CPG are reflected in the model’s qualitative behavior when auxiliary descriptions are added or removed.

![](images/e744e658f78ef8b6a0ee5db0ccbb3b446306e35109e2cba6f87781c02b5b6f4a.jpg)  
Fig. A: Additional Qualitative examples of image-conditioned description generation.

![](images/d6ff80285d4a0ddb806e6ed9967bdfc1d43f36fd6b69a898c02f27bd08632408.jpg)

![](images/b0742b82918fa527670e8af9dc1dab4693071666838d9c44d18f5f88de2def00.jpg)

![](images/1d49c5d82e5ecc20f88440fd9f10af54b858598f729bd16b7ffafed70429d6b1.jpg)

![](images/b9c4017c878933b74afc318cce433f664e67bfe9d5719fb7af26db945c5fbad4.jpg)

![](images/91f9a0870d7e5d2b6ea3b8a61b6ece8fef1e9cc5471e31fd83935fb6fe2d414d.jpg)

![](images/5787acec3357e6dc822bab4af992dae65cd3b0130f222f8a6fafff7defcc5f9e.jpg)

![](images/226a33c5628ba0c27662fcad3c62d8436efb5694dfe2d67babed8b2e699c2336.jpg)

![](images/54bc6fd7b961732d4092838e0e9857c5468cbddfc78cdf364115a8a8d9cab585.jpg)

![](images/14d28412b275561806570822a819a80e2bc6d0cf3d545aac3fad6cf554c6965a.jpg)

![](images/46f14cfc99b4302172a432c86b0cab1b5fb33ae93ae50bd4bc758a6916f66447.jpg)

![](images/2259f5dd9258b0684882aa13c174ad0efff6c6b8630ec36f94cb0cb6a4df1ca8.jpg)  
Fig. B: Additional qualitative examples demonstrating how our C<sup>2</sup>-DPO model leverages additional context (caption) more efectively than the baseline C-DPO model.

![](images/f901d8d1b5898d1f42c174be7f4ec878e78abe6e069372f4646754418f92fbe6.jpg)