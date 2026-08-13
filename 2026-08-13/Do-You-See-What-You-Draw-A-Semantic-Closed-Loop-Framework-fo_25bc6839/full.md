# Do You See What You Draw? A Semantic Closed-Loop Framework for Holistic Evaluation of Unified Multimodal Models

Hao Zhang Hangzhou Institute for Advanced Study, University of Chinese Academy of Sciences 12210801@mail.sustech.edu.cn

Zhijiang Tang Hangzhou Institute for Advanced Study, University of Chinese Academy of Sciences tangzhijiang24@mails.ucas.ac.cn

Jiaxin Qi Computer Network Information Center, Chinese Academy of Sciences jxqi@cnic.cn

Jianqiang Huang Computer Network Information Center, Chinese Academy of Sciences jqhuang@cnic.cn

## Abstract

As Large Vision-Language Models increasingly aim to integrate visual generation and understanding within a single parameter space, evaluating such structural unification in a cohesive manner remains a critical challenge. Current evaluation protocols predominantly treat generative and discriminative capabilities as separate tasks, leaving a gap in system-level evaluation for unified multimodal models (UMMs). In this work, we propose Self-Generative-Understanding (SGU), a novel, annotation-free evaluation framework that probes the integrated capabilities of unified models through a semantic closed-loop challenge. Without requiring new annotations, SGU leverages the dual understanding-and-generation abilities of UMMs by asking them to first perceive an image and produce a textual description, subsequently reconstruct a visual context based on that description, and finally perform reasoning over the self-generated output. This pipeline provides a zerocost testbed that yields an integrated performance score specifically tailored for evaluating UMMs as unified systems. Extensive experiments show that even highperforming UMMs often struggle to reason over their own generated contexts, revealing limitations that are not captured by separate evaluations of understanding or generation alone. Our work provides a complementary holistic evaluation framework and offers a foundation for benchmarking the development of nextgeneration unified multimodal models.

## 1 Introduction

The evolution of Artificial General Intelligence (AGI) increasingly calls for models that go beyond passive perception and can integrate multimodal understanding with visual generation [6, 26]. Motivated by the intuition that the ability to generate visual content is closely related to understanding visual semantics, Multimodal Artificial Intelligence is transitioning from specialized, task-specific experts toward Unified Multimodal Models (UMMs) [13, 26, 28, 32]. By jointly training on generation and understanding objectives, UMMs consolidate discriminative reasoning and generative synthesis within a shared model, aiming to support both perception and creation in an integrated manner.

Despite the rapid architectural convergence toward unification, current evaluation protocols still primarily assess generation and understanding as separate capabilities [13, 14], as illustrated in

![](images/38f72f9eec175db39bda4d7841851ee3bbfc00c66ed9c12c57b4e91cfcee3cde.jpg)  
Figure 1: Illustration of traditional UMM evaluation and our Self-Generative-Understanding (SGU) framework. (a) Existing evaluations often assess generation $( \mathcal { M } _ { G } )$ and understanding $( \mathcal { M } _ { U } )$ separately, producing capability-specific scores such as $s _ { g }$ and $s _ { u } .$ . These metrics remain valuable for component-wise diagnosis but do not directly provide a system-level view of how a UMM behaves when understanding and generation are jointly involved. (b) SGU provides a complementary closed loop evaluation framework tailored to UMMs. It requires the model to understand an input image, generate a textual representation, reconstruct a visual context from that representation, and reason over the reconstructed image, yielding an outcome-based system-level score $s _ { \mathrm { u m m } }$

Figure 1(a). Typically, generative ability is quantified using distributional or text-image alignment metrics such as FID [12] and CLIPScore [11], while understanding ability is evaluated on visual question answering or multimodal reasoning benchmarks. These component-wise evaluations are important for diagnosing specific strengths and weaknesses. However, they do not directly provide a unified assessment of how well a UMM performs when understanding and generation are both involved within the same model. This question is difficult to answer by naively aggregating heterogeneous generation and understanding metrics. For example, two UMMs may exhibit opposite capability profiles, with one stronger in generation and the other stronger in understanding. Separate scores can reveal these differences, but they do not directly indicate whether the model can function well when both abilities must interact within a unified process. This motivates a complementary evaluation perspective that treats the UMM as an integrated system rather than only as a collection of separately measured components.

To address this gap, we propose Self-Generative-Understanding (SGU), a semantic closed-loop evaluation framework explicitly designed for UMMs. SGU leverages the dual understanding-andgeneration abilities of UMMs by asking the model to first perceive an input image and generate a textual description, then reconstruct a visual context based on that description, and finally answer image-grounded questions using the reconstructed image, as shown in Figure 1(b). Since the downstream questions are grounded in the original image, successful reasoning on the reconstructed image provides an outcome-based signal of whether the model can effectively carry out the integrated understanding-and-generation process. Any weakness in understanding, representation, or generation will be reflected in the final performance. The resulting SGU score is therefore a system-level performance score tailored to UMMs.

SGU offers three advantages. (1) Integrated capability evaluation. The closed-loop process activates multiple capabilities of UMMs, including image understanding, textual representation, visual generation, and downstream reasoning, without relying on external judge models or additional generation systems. (2) System-level sensitivity. The final score reflects the combined effect of these stages: information loss in understanding, intermediate representation, generation, or self-reasoning will reduce the final performance. Thus, SGU evaluates whether the UMM succeeds as an integrated system, while stage-wise analyses can be used for further diagnosis. (3) Annotation-free scalability. SGU can be instantiated on existing image-grounded reasoning benchmarks, such as MMBench [19], by reusing their original questions and ground-truth answers, requiring no new annotations.

Extensive experiments reveal a clear gap between isolated benchmark performance and integrated multimodal behavior in current UMMs. Despite strong results on individual tasks, many models suffer substantial performance degradation under the SGU loop, exposing limitations that are not captured by separate evaluations of understanding or generation alone. Stage-wise replacement experiments further help analyze which parts of the loop contribute to the degradation, while additional studies examine robustness to prompt variations, text bottlenecks, and potential shortcut behaviors. These findings show that SGU provides both a complementary system-level score and diagnostic insights for evaluating unified multimodal models.

Our contributions are summarized as follows:

• We identify the need for a complementary system-level evaluation framework for UMMs, beyond separate generation and understanding metrics.

• We propose SGU, a semantic closed-loop evaluation framework tailored for UMMs, providing an annotation-free and model-internal system-level assessment of integrated understanding-and-generation behavior.

• We conduct extensive experiments and in-depth analyses of representative UMMs, providing quantitative evaluations and diagnostic insights for assessing unified multimodal behavior.

## 2 Related Work

## 2.1 Unified Multimodal Models

Multimodal intelligence has rapidly evolved from task-specific systems toward Unified Multimodal Models (UMMs), which aim to integrate visual understanding and generation within a single model [13, 14, 29]. Existing UMMs can be broadly categorized into several representative paradigms. The first line of work adopts autoregressive modeling with unified or discrete visual tokenization, allowing text and image tokens to be processed within a shared sequence modeling framework. Representative examples include Chameleon [24], Emu3 [27], Janus-Pro [4], and X-Omni [8]. More recent models such as Show-o2 [33] further incorporate improved tokenization or scheduling strategies to enhance the balance between understanding and generation. The second paradigm integrates diffusion or flow-based generation modules with multimodal reasoning backbones, aiming to improve image generation quality while preserving understanding capabilities [14, 17, 36]. Representative models such as BLIP3-o [2], BAGEL [5], UniWorld [15], and OmniGen2 [31] leverage large-scale multimodal training or interleaved data to support both visual reasoning and generation. Another line of work maintains a strong reasoning backbone while connecting it to image generation modules through learnable interfaces or routing mechanisms [23]. This design can preserve language and vision reasoning ability while enabling generation through specialized components. In parallel, proprietary systems such as GPT-Image [22] and Gemini 2.5 Flash Image [10] further demonstrate the rapid progress of unified understanding-generation systems.

## 2.2 Benchmarks for Unified Multimodal Models

Evaluation of UMMs remains challenging because current methodologies typically assess multimodal understanding and generation separately. Understanding capabilities are generally evaluated through question-answering (QA) benchmarks, such as MME [7] and MMMU [34, 35]. Conversely, generation performance often relies on reference-based benchmarks such as GenEval [9]. Recent advancements, such as GIRBench [13], have extended this scope to visual editing, while VQAScore [16] leverages QA accuracy as a proxy for evaluating image generation quality. For unified evaluation, UmniBench [18] assesses generation and editing performance by utilizing the model’s intrinsic understanding.

These benchmarks provide valuable component-level insights. However, they do not directly provide a system-level evaluation of how understanding and generation interact within a UMM. In addition, existing approaches often rely on references or external model-based judges, which may introduce additional biases. Building on existing VQA-style benchmarks, we propose SGU as a complementary semantic closed-loop framework for evaluating UMMs as integrated systems, without requiring new annotations or third-party judge models.

![](images/c0250611ccc662c9b9213e3a41f5e56d9bc14842e380e8e9bd060d947cf909a2.jpg)  
Figure 2: Overview of the Self-Generative-Understanding (SGU) protocol. Given a VQA triplet $( v , q , a )$ , the UMM first generates a textual description $t _ { g }$ from the original image $v ,$ reconstructs a visual context vˆ from $t _ { g } ,$ and then answers the original question q based on vˆ to obtain the SGU score $s _ { \mathrm { u m m } } .$ . Each stage is executed in an isolated stateless session, with no memory or intermediate features carried across stages.

## 3 Method

## 3.1 Preliminaries

Unified Multimodal Model (UMM). Formally, we define a UMM M as a probabilistic model designed to process and generate visual and textual modalities within a unified model. While current UMMs often share a common parameter space, their functional capabilities can be viewed through conditional projections targeting different output modalities [26, 28]. We denote the visual generation function as $\mathcal { M } _ { G }$ and the understanding function as $\mathcal { M } _ { U }$

UMM evaluation. The core challenge in UMM evaluation is to define a scoring function $s$ that can assess the model as an integrated system. Given a multimodal dataset $\mathcal { D } = \{ ( v , \overline { { t } } ) \}$ , where v denotes visual input and t denotes textual input, a system-level score can be formalized as

$$
s _ { \mathrm { u m m } } = \boldsymbol { S } ( \boldsymbol { M } , \mathcal { D } ) .\tag{1}
$$

Such a score should satisfy two desiderata. (1) Unified assessment. The score should provide a scalar signal for comparing different UMMs at the system level, complementing component-wise evaluations. (2) Integrated sensitivity. The score should reflect the combined effect of understanding and generation, such that weakness in either capability, or in their interaction within the model, can affect the final performance.

Current evaluation protocols. Existing evaluations commonly assess generation and understanding using separate metrics. Formally, given a generation benchmark $\mathcal { D } _ { G }$ and an understanding benchmark $\mathcal { D } _ { U }$ , current methods can be viewed as producing a tuple of heterogeneous scores:

$$
\begin{array} { r } { s _ { \mathrm { u m m } } = ( s _ { g } , s _ { u } ) , \quad s _ { g } =  { \mathcal { S } } _ { G } (  { \mathcal { M } } _ { G } ,  { \mathcal { D } } _ { G } ) , \quad s _ { u } =  { \mathcal { S } } _ { U } (  { \mathcal { M } } _ { U } ,  { \mathcal { D } } _ { U } ) , } \end{array}\tag{2}
$$

where $\mathit { S } _ { \mathit { G } }$ denotes generation-oriented metrics such as FID [12] and CLIPScore [11], and $\mathit { S } _ { U }$ denotes understanding-oriented metrics such as VQA accuracy [16].

These component-wise evaluations are valuable for diagnosing specific capabilities. At the same time, their heterogeneous nature makes direct aggregation non-trivial. This motivates a complementary system-level perspective for understanding how a UMM behaves when understanding and generation are jointly involved.

## 3.2 Our Evaluation: Self-Generative-Understanding

We propose Self-Generative-Understanding (SGU), an annotation-free semantic closed-loop framework for system-level evaluation of UMMs. SGU is designed to evaluate whether a UMM can perform understanding and generation in an integrated manner by asking the model to operate on its own intermediate outputs. Given a data sample $( v , t ) \in \mathcal { D }$ , SGU enforces a three-stage process: (1) Visual description. The understanding function $\mathcal { M } _ { U }$ perceives the original visual input v and generates a textual description $t _ { g } . \ ( 2 )$ Visual generation. Based solely on the self-generated description $t _ { g } ,$ the generation function $\mathcal { M } _ { G }$ synthesizes a proxy visual context vˆ. (3) Selfunderstanding. The model then uses $\mathcal { M } _ { U }$ to resolve the original textual task on the synthesized image vˆ and produces the final response t<sup>ˆ</sup>. The process can be formulated as

$$
t _ { g } = \mathcal { M } _ { U } ( v ) , \quad \hat { v } = \mathcal { M } _ { G } ( t _ { g } ) , \quad \hat { t } = \mathcal { M } _ { U } ( \hat { v } , t ) .\tag{3}
$$

## 3.3 SGU score

To instantiate SGU, we adopt VQA-style settings, where each question is grounded in the original image and paired with a ground-truth answer. A data sample is formalized as a triplet $( v , q , a )$ consisting of an image v, a question q, and a ground-truth answer a. Based on Eq. (3), the SGU score is calculated as

$$
s _ { \mathrm { u m m } } = \mathbb { E } _ { ( v , q , a ) \sim \mathcal { D } } \left[ \mathbb { I } \left( \mathrm { M a t c h } \left( \mathcal { M } _ { U } ( \mathcal { M } _ { G } ( \mathcal { M } _ { U } ( v ) ) , q ) , a \right) \right) \right] ,\tag{4}
$$

where $\mathbb { I } ( \cdot )$ is the logical indicator function and Match(·) denotes answer matching. For multiplechoice questions, Match checks whether the predicted option matches the ground truth. For openform questions with deterministic answers, it applies parsing and normalization, such as case normal ization, punctuation removal, and numerical matching, before comparison. This avoids relying on naive strict string matching for free-form outputs.

What does SGU measure? The SGU score is an outcome-based system-level evaluation of whether a UMM can effectively operate through its own understanding-and-generation process. Since each question and answer are grounded in the original image, successful answering on the reconstructed image reflects the model’s integrated multimodal capability under this process. Thus, task accuracy serves as an operational proxy for evaluating the overall success of the SGU loop. As the score aggregates the effects of different stages, stage-wise analyses can further help interpret the source of performance changes, as discussed in Section 4.

Relative SGU score. Since $s _ { \mathrm { u m m } }$ is influenced by the model’s base understanding ability, we introduce a relative SGU score normalized by the base understanding score $s _ { \mathrm { b a s e } } ,$ which measures $\mathrm { \Delta V Q A }$ performance on the original images and serves as a model-specific upper-bound reference:

$$
s _ { \mathrm { b a s e } } = \mathbb { E } _ { ( v , q , a ) \sim \mathcal { D } } \left[ \mathbb { I } \big ( \mathrm { M a t c h } \left( \mathcal { M } _ { U } ( v , q ) , a \right) \big ) \right] , \quad s _ { \mathrm { u m m , r } } = \frac { s _ { \mathrm { u m m } } } { s _ { \mathrm { b a s e } } } .\tag{5}
$$

The relative score reflects how much performance is preserved after the model is required to operate through its own understanding-and-generation loop.

## 4 Experiments

## 4.1 Implementation

Datasets. We evaluate SGU on a diverse set of representative multimodal benchmarks covering general reasoning, perception, mathematical reasoning, and text-centric visual understanding. These datasets allow us to examine the system-level behavior of UMMs across different VQA-style tasks.

• MMStar [3], a comprehensive benchmark designed to assess general multimodal reasoning across diverse visual and textual scenarios.

• MMBench [19], which focuses on evaluating core vision–language understanding abilities through carefully curated multiple-choice questions.

• MathVista [20], targeting mathematical and logical reasoning grounded in visual contexts.

• OCR-VQA [21], emphasizing text recognition and semantic understanding in visually rich, text-heavy images.

UMMs. We evaluate six representative unified multimodal models across diverse architectures to analyze their system-level behavior under the SGU protocol.

Table 1: Results of various unified multimodal models on diverse VQA benchmarks under the SGU protocol. For each dataset, we report direct VQA accuracy on the original image $( s _ { \mathrm { b a s e } } )$ and SGU score $( s _ { \mathrm { u m m } } )$ , where $s _ { \mathrm { b a s e } }$ serves as a model-specific upper-bound reference for interpreting $s _ { \mathrm { u m m } } .$
<table><tr><td rowspan="2">Model</td><td colspan="2">MMStar</td><td colspan="2">MMBench</td><td colspan="2">MathVista</td><td colspan="2">OCR-VQA</td><td colspan="2">Avg</td></tr><tr><td>Original Sbase</td><td>SGU Summ</td><td>Original Sbase</td><td>SGU Summ</td><td>Original Sbase</td><td>SGU Summ</td><td>Original Sbase</td><td>SGU Summ</td><td>Original Sbase</td><td>SGU Summ</td></tr><tr><td>UPPER BOUND</td><td>66.47</td><td></td><td>86.13</td><td></td><td>70.60</td><td></td><td>81.33</td><td></td><td>76.13</td><td></td></tr><tr><td>UniWorld-V1</td><td>60.67</td><td>38.33</td><td>86.13</td><td>71.97</td><td>67.30</td><td>37.40</td><td>81.33</td><td>28.67</td><td>73.86</td><td>44.09</td></tr><tr><td>Janus-Pro-7B</td><td>47.80</td><td>39.27</td><td>76.86</td><td>68.20</td><td>41.30</td><td>32.50</td><td>69.27</td><td>37.45</td><td>58.82</td><td>44.36</td></tr><tr><td>Show-o2-7B</td><td>55.13</td><td>43.00</td><td>83.92</td><td>73.95</td><td>50.70</td><td>38.90</td><td>63.43</td><td>30.34</td><td>63.30</td><td>46.55</td></tr><tr><td>Ovis-U1</td><td>60.60</td><td>43.07</td><td>83.71</td><td>75.10</td><td>68.90</td><td>41.80</td><td>79.00</td><td>34.71</td><td>73.05</td><td>48.67</td></tr><tr><td>BAGEL-7B</td><td>66.47</td><td>42.67</td><td>85.54</td><td>76.38</td><td>70.60</td><td>39.50</td><td>74.55</td><td>53.35</td><td>74.29</td><td>52.98</td></tr><tr><td>OmniGen2</td><td>54.93</td><td>43.07</td><td>82.84</td><td>73.66</td><td>63.50</td><td>40.40</td><td>79.39</td><td>56.59</td><td>70.17</td><td>53.43</td></tr></table>

• Janus-Pro-7B [4] adopts an autoregressive framework that decouples visual encoding to mitigate conflicts between multimodal understanding and generation.

• BAGEL-7B [5] is a decoder-only foundation model trained on massive interleaved imagetext data, subsequently exhibiting strong emergent multimodal reasoning capabilities across benchmarks.

• UniWorld-V1 [15] is a unified world model that learns consistent spatiotemporal representations for both perception and synthesis within a shared and continuous latent space.

• Show-o2-7B [33] integrates discrete diffusion scheduling with autoregressive modeling to enhance generation fidelity while maintaining semantic understanding.

• OmniGen2 [31] supports arbitrary multimodal input-output sequences with a focus on generalized instruction following in complex scenarios.

• Ovis-U1-3B [25] is a lightweight unified model that enables evaluation at a smaller parameter count with lower inference cost.

Evaluation setup. For each dataset sample $( v , q , a )$ , we follow the SGU protocol by first feeding the input image v into the unified model to generate a textual description $t _ { g } ,$ then reconstructing an image vˆ from $t _ { g } .$ , and finally performing visual question answering on $( \hat { v } , \bar { q } )$ to obtain the predicted answer t<sup>ˆ</sup>. The SGU score $s _ { \mathrm { u m m } }$ is computed as the average accuracy over all samples.

To ensure fair and leakage-free evaluation, all stages are executed in a stateless manner: the same unified model is reused across stages, but each stage runs in an isolated session with no memory or intermediate features preserved across steps. Throughout the evaluation, no external judge models or human annotators are introduced; all stages are performed solely by the tested unified model, resulting in a model-internal system-level evaluation protocol.

## 4.2 Results Analysis

## Q1. Does SGUprovide a system-level view beyond isolated evaluations?

A1. Table 1 shows a consistent drop from direct VQA accuracy on the original image $( s _ { \mathrm { b a s e } } )$ to the SGU score $( s _ { \mathrm { u m m } } )$ . This drop reflects the additional challenge introduced when a UMM must operate through its own understanding-and-generation process. Importantly, models with similar $s _ { \mathrm { b a s e } }$ can exhibit very different SGU scores. For example, on $\mathrm { O C R - V Q A }$ , UniWorld-V1 achieves $s _ { \mathrm { b a s e } } { = } 8 1 . 3 3$ but drops to $s _ { \mathrm { u m m } } { = } 2 8 . 6 7$ , while OmniGen2 has a comparable $s _ { \mathrm { b a s e } } { = } 7 9 . 3 9$ but retains a much higher $s _ { \mathrm { u m m } } { = } 5 6 . 5 9$ . This indicates that isolated understanding performance alone does not fully characterize how well a UMM functions when understanding and generation are jointly involved.

The normalized ratio $s _ { \mathrm { u m m , r } } ,$ defined in Eq. (5) and visualized in Figure 3, further provides a modelspecific view of how much performance is preserved under the SGU loop. Datasets such as MathVista and OCR-VQA generally exhibit larger drops, suggesting that visually grounded reasoning and textcentric perception are more challenging to maintain through the integrated process. Overall, Table 1 and Figure 3 show that SGU complements component-wise evaluation by providing a system-level view of UMM behavior.

Table 2: Stage-wise replacement effects on SGU score $( \Delta s _ { \mathrm { u m m } } )$ replaced − baseline) on the constructed subsets. MV: MathVista; OCR: OCR-VQA.
<table><tr><td rowspan="2">Model</td><td colspan="2">Understand repl.</td><td colspan="2">Generation repl.</td></tr><tr><td>MV</td><td>OCR</td><td>MV</td><td>OCR</td></tr><tr><td>UniWorld-V1</td><td>-4.16</td><td>-2.52</td><td>13.64</td><td>38.89</td></tr><tr><td>Janus-Pro-7B</td><td>1.88</td><td>-3.03</td><td>3.54</td><td>21.21</td></tr><tr><td>Show-o2-7B</td><td>-1.74</td><td>1.01</td><td>0.00</td><td>29.30</td></tr><tr><td>Ovis-U1-3B</td><td>1.51</td><td>2.02</td><td>6.57</td><td>34.85</td></tr><tr><td>BAGEL-7B</td><td>-1.60</td><td>-0.50</td><td>5.05</td><td>13.14</td></tr><tr><td>OmniGen2</td><td>3.93</td><td>-1.51</td><td>11.61</td><td>12.12</td></tr></table>

Table 3: Prompt-sensitivity ablation under SGU on the constructed MMBench subset with OmniGen2. We vary one stage at a time and report the mean SGU score across two runs. Und./Gen. Prompt denote stage prompts; Var-1/Var-2 denote alternative templates.
<table><tr><td>Setting</td><td colspan="2">Und. prompt Gen. prompt</td><td>Summ</td></tr><tr><td>Default</td><td>Default</td><td>Default</td><td>73.74</td></tr><tr><td>Var-1 (Und)</td><td>Var1</td><td>Default</td><td>73.48</td></tr><tr><td>Var-2 (Und)</td><td>Var2</td><td>Default</td><td>73.99</td></tr><tr><td>Var-1 (Gen)</td><td>Default</td><td>Var1</td><td>75.51</td></tr><tr><td>Var-2 (Gen)</td><td>Default</td><td>Var2</td><td>73.48</td></tr></table>

To further compare SGU with conventional evaluation signals, we compute an isolated average score by aggregating commonly used component-level indicators, including direct VQA accuracy $s _ { \mathrm { b a s e } }$ and CLIPScore [11] between the original image and the reconstructed image or generated description. As shown in Figure 5, the resulting ranking trend is broadly consistent with SGU. This suggests that SGU captures capability patterns reflected by existing metrics, while providing a single outcome-based score for evaluating the integrated understanding-and-generation process.

![](images/e3033a9a9e822c89c473638f760daf884f38a2a31e17fac6137bdd87d87b4a6c.jpg)

## Q2. How does SGU reflect and diagnose the joint effects of understanding and generation?

Figure 3: Circular bar plot of $s _ { \mathrm { u m m , r } } .$

## A2. The SGU loop involves three key compo-

nents: image-to-text understanding, image generation, and final VQA reasoning. The final VQA reasoning ability is reflected by the direct VQA reference score $s _ { \mathrm { b a s e } } ,$ while the effects of the first two stages are further analyzed through stage-wise replacement experiments. Specifically, we replace the image-to-text understanding stage with Qwen3-VL-8B [1], and replace the image-generation stage with Qwen-Image-2512 [30], while keeping the rest of the loop unchanged. We focus on MathVista and OCR-VQA, where SGU gaps are most pronounced, and use fixed-size stratified subsets that preserve the original question-type distribution, with the same split reused for baseline and replacement runs; details are provided in Appendix E.

Table 2 shows that both interventions change SGU performance, confirming that the score is affected by both understanding and generation. Replacing the generation stage yields larger positive gains across models, suggesting that visual reconstruction is a major bottleneck in the current SGU loop. However, this does not mean that SGU only measures generation: if the image-to-text stage fails to extract sufficient visual information, a stronger generator cannot recover what is missing. Consistently, understanding replacement also affects the score, though sometimes with smaller or even negative changes, suggesting that stronger isolated understanding does not always produce intermediate descriptions that are more compatible with a model’s own generation and final reasoning behavior.

We further verify the role of the understanding-side intermediate representation by directly performing QA on generated captions without the image-generation stage. On OCR-VQA with OmniGen2, the score drops from 79.39 using the original image to 60.05 using captions alone, and further to 56.59 under the full SGU loop. This shows that imperfect information extraction in the captioning stage already degrades performance, while generation introduces additional challenges. Moreover, even after replacing the generation stage, most models retain a residual gap to the direct-VQA reference s<sub>base</sub>, as shown in Figure 6. Together, these results show that SGU provides a holistic performance signal, while stage-wise and caption-only analyses help explain how understanding and generation jointly shape the final score.

![](images/3341054133b7832354baf6689511ef9bea267b384b40e070422e3c6c99e14110.jpg)

(a) Case of weak understanding  
![](images/bf2876c8b740880fb5d4a0063ae1d7c4ed73e0b53678284d6a3cfc0e5b9e1944.jpg)  
(b) Case of weak generation  
Figure 4: Qualitative case studies under the SGU protocol. The examples illustrate different loop outcomes: one model successfully performs both understanding and generation, leading to the correct answer, while another produces an incorrect answer when visual information is incompletely extracted (Case a) or the reconstructed image deviates from the intended content (Case b).

## Q3. Is SGU sensitive to understanding and generation prompts?

A3. We assess prompt sensitivity by rephrasing the internal prompts used in the understanding or generation stage, while keeping the rest of the SGU loop unchanged for controlled comparison. We consider three understanding prompts:

• DEFAULT: Describe this image in detail

• VAR1: Generate a detailed description of the visual content in this image.

• VAR2: Analyze the image and provide a thorough description capturing all key elements.

We also consider three generation prompts:

• DEFAULT: Generate an image based on the following description: {description}.

• VAR1: Using the model-generated description directly as the generation input

• VAR2: Generate a high-quality, detailed image of {description}.

We change only one stage at a time, resulting in four settings, and run each setting twice with different random seeds, reporting the average accuracy. Table 3 reports the results on OmniGen2 with our constructed MMBench subset, where all prompt variants stay close to the default score, suggesting that SGU is insensitive primarily to reasonable prompt choices within the loop and provides a stable signal under such perturbations.

## Q4. Is SGU endogenous, and can models exploit shortcut signals?

A4. SGU is model-internal: all stages are performed by the tested UMM itself under a fixed protocol, without external judge models or human evaluators. To prevent hidden state leakage, each stage is executed in an isolated stateless session, and only the explicit intermediate artifact is passed forward to support the subsequent stage.

![](images/3ad23a46f92b5ed91514818fdb4bb99f06227cca78a33c130bcc9965dd6d62c8.jpg)  
Figure 5: Comparison of ranking performance across average SGU score $s _ { \mathrm { u m m } } ,$ average normalized ratio $s _ { \mathrm { u m m , r } } ,$ and an isolated average aggregated from $C L I P { - } T$ , CLIP-I, and $s _ { \mathrm { b a s e } }$

![](images/33249b5c46a412e25911c7c3c6a00500111e83d31ab69c634cdb0dcab0eadedb.jpg)  
Figure 6: Visualization of the baseline SGU accuracy $s _ { \mathrm { u m m } } .$ , the gain from replacing the generation stage, and the residual gap to the direct VQA reference score $s _ { \mathrm { b a s e } } ,$ evaluated on our constructed OCR-VQA subset.

Table 4: Image-swap check on the constructed MMBench subset. We pair models with similar direct $\mathrm { \Delta V Q A }$ accuracy $s _ { \mathrm { b a s e } }$ . For each target model A (tested for potential shortcut exploitation), $s _ { \mathrm { u m m } } = \operatorname { A c c } ( A \to A )$ evaluates $A$ on images generated by itself, while $s _ { \mathrm { c r o s s } } = \operatorname { A c c } ( B \to A )$ evaluates the paired model B on images generated by A. We report $s _ { \Delta } = s _ { \mathrm { c r o s s } } - s _ { \mathrm { u m m } } .$
<table><tr><td>Pair</td><td>Target</td><td> $\pmb { s } \pmb { \triangle }$ </td><td> $\pmb { s } _ { \mathrm { b a s e } }$ </td><td> $s _ { \mathrm { u m m } }$ </td><td> $\pmb { s } _ { \mathrm { c r o s s } }$ </td></tr><tr><td>Ovis ↔ UniWorld</td><td>Ovis</td><td>-0.82</td><td>86.36</td><td>82.32</td><td>81.50</td></tr><tr><td rowspan="3">Show-o2 ↔ BAGEL</td><td>UniWorld</td><td>+0.34</td><td>84.85</td><td>75.76</td><td>76.10</td></tr><tr><td>Show  $^ { . 0 2 }$ </td><td>-2.77</td><td>84.34</td><td>76.77</td><td>74.00</td></tr><tr><td>BAGEL</td><td>+3.04</td><td>83.33</td><td>76.76</td><td>79.80</td></tr><tr><td rowspan="2">OmniGen2 ↔ Janus</td><td>OmniGen2</td><td>-3.06</td><td>81.82</td><td>73.74</td><td>70.68</td></tr><tr><td>Janus</td><td>+5.83</td><td>78.79</td><td>67.17</td><td>73.00</td></tr></table>

To probe whether loop-generated visual outputs embed model-specific shortcut cues that disproportionately benefit self-evaluation, we perform an image-swap test on the previously constructed high-scoring MMBench subset. We first pair UMMs with similar direct VQA accuracy so their base competence is comparable. For each target model, we compare two evaluations under the same questions. In $S e l f ,$ the target answers on images it generates itself, whereas in Cross, its paired model answers on the target’s generated images. As summarized in Table 4, Cross remains broadly comparable to $S e l f$ across pairs, with only minor and non-systematic differences, suggesting no consistent shortcut signals that materially influence SGU performance.

## 5 Conclusion, Limitations, and Future Work

This paper introduces Self-Generative-Understanding (SGU), an endogenous closed-loop evaluation protocol designed for Unified Multimodal Models. By reusing standard VQA benchmarks, SGU evaluates whether a model can operate through its own understanding-and-generation process and complete tasks grounded in the original images, providing a complementary system-level view rather than replacing component-wise evaluations. Experiments across four benchmarks and six representative UMMs show that SGU captures system-level behavior that is not fully reflected by isolated evaluations, while additional analyses further explain how the integrated process affects the final score and reveals model-specific bottlenecks.

Limitations and Future Work. SGU is a system-level evaluation signal, not a standalone theoretical criterion for UMM alignment. Its score naturally aggregates effects from multiple stages, so finegrained failure attribution should be performed with stage-wise analyses and component-level metrics.

The current implementation also uses text as an intermediate representation, which may introduce information bottlenecks for detail-intensive tasks and fine-grained visual evidence. Future work will explore richer intermediate representations, broader task instantiations, and adapting SGU into a training objective using closed-loop feedback.

## References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[2] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, Le Xue, Caiming Xiong, and Ran Xu. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset, 2025. URL https://arxiv.org/abs/2505.09568.

[3] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large vision-language models?, 2024. URL https://arxiv.org/abs/2403.20330.

[4] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling, 2025. URL https://arxiv.org/abs/2501.17811.

[5] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining, 2025. URL https://arxiv.org/abs/2505.14683.

[6] Zane Durante, Qiuyuan Huang, Naoki Wake, Ran Gong, Jae Sung Park, Bidipta Sarkar, Rohan Taori, Yusuke Noda, Demetri Terzopoulos, Yejin Choi, Katsushi Ikeuchi, Hoi Vo, Li Fei-Fei, and Jianfeng Gao. Agent ai: Surveying the horizons of multimodal interaction, 2024. URL https://arxiv.org/abs/2401.03568.

[7] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, Rongrong Ji, Caifeng Shan, and Ran He. Mme: A comprehensive evaluation benchmark for multimodal large language models, 2025. URL https://arxiv.org/abs/2306.13394.

[8] Zigang Geng, Yibing Wang, Yeyao Ma, Chen Li, Yongming Rao, Shuyang Gu, Zhao Zhong, Qinglin Lu, Han Hu, Xiaosong Zhang, Linus, Di Wang, and Jie Jiang. X-omni: Reinforcement learning makes discrete autoregressive image generative models great again, 2025. URL https://arxiv.org/abs/2507.22058.

[9] Dhruba Ghosh, Hanna Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment, 2023. URL https://arxiv.org/abs/2310.11513.

[10] Google DeepMind, 2025. URL https://developers.googleblog.com/en/introducing-gemini-2-5- flash-image/.

[11] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning, 2022. URL https://arxiv.org/abs/ 2104.08718.

[12] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems, volume 30, 2017.

[13] Hongxiang Li, Yaowei Li, Bin Lin, Yuwei Niu, Yuhang Yang, Xiaoshuang Huang, Jiayin Cai, Xiaolong Jiang, Yao Hu, and Long Chen. Gir-bench: Versatile benchmark for generating images with reasoning, 2025. URL https://arxiv.org/abs/2510.11026.

[14] Yongyuan Liang, Wei Chow, Feng Li, Ziqiao Ma, Xiyao Wang, Jiageng Mao, Jiuhai Chen, Jiatao Gu, Yue Wang, and Furong Huang. Rover: Benchmarking reciprocal cross-modal reasoning for omnimodal generation, 2025. URL https://arxiv.org/abs/2511.01163.

[15] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, Yatian Pang, and Li Yuan. Uniworld-v1: High-resolution semantic encoders for unified visual understanding and generation, 2025. URL https:// arxiv.org/abs/2506.03147.

[16] Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation. In European Conference on Computer Vision, pages 366–384. Springer, 2024.

[17] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2023. URL https://arxiv.org/abs/2210.02747.

[18] Kai Liu, Leyang Chen, Wenbo Li, Zhikai Chen, Zhixin Wang, Renjing Pei, Linghe Kong, and Yulun Zhang. Umnibench: Unified understand and generation model oriented omni-dimensional benchmark, 2025. URL https://arxiv.org/abs/2512.17196.

[19] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player?, 2024. URL https://arxiv.org/abs/2307.06281.

[20] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts, 2024. URL https://arxiv.org/abs/ 2310.02255.

[21] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 947–952, 2019. doi: 10.1109/ICDAR. 2019.00156.

[22] OpenAI, 2025b. URL https://openai.com/index/image-generation-api/.

[23] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, Ji Hou, and Saining Xie. Transfer between modalities with metaqueries, 2025. URL https://arxiv.org/abs/2504.06256.

[24] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models, 2025. URL https://arxiv.org/abs/2405.09818.

[25] Guo-Hua Wang, Shanshan Zhao, Xinjie Zhang, Liangfu Cao, Pengxin Zhan, Lunhao Duan, Shiyin Lu, Minghao Fu, Xiaohao Chen, Jianshan Zhao, Yang Li, and Qing-Guo Chen. Ovis-u1 technical report, 2025. URL https://arxiv.org/abs/2506.23044.

[26] Maijunxian Wang. From understanding the world to intervening in it: A unified multi-scale framework for embodied cognition, 2025. URL https://arxiv.org/abs/2503.00727.

[27] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, Yingli Zhao, Yulong Ao, Xuebin Min, Tao Li, Boya Wu, Bo Zhao, Bowen Zhang, Liangdong Wang, Guang Liu, Zheqi He, Xi Yang, Jingjing Liu, Yonghua Lin, Tiejun Huang, and Zhongyuan Wang. Emu3: Next-token prediction is all you need, 2024. URL https://arxiv.org/abs/2409.18869.

[28] Hongyang Wei, Baixin Xu, Hongbo Liu, Cyrus Wu, Jie Liu, Yi Peng, Peiyu Wang, Zexiang Liu, Jingwen He, Yidan Xietian, Chuanxin Tang, Zidong Wang, Yichen Wei, Liang Hu, Boyi Jiang, William Li, Ying He, Yang Liu, Xuchen Song, Eric Li, and Yahui Zhou. Skywork unipic 2.0: Building kontext model with online rl for unified multimodal model, 2025. URL https://arxiv.org/abs/2509.04548.

[29] Jingxuan Wei, Caijun Jia, Xi Bai, Xinglong Xu, Siyuan Li, Linzhuang Sun, Bihui Yu, Conghui He, Lijun Wu, and Cheng Tan. Ggbench: A geometric generative reasoning benchmark for unified multimodal models, 2026. URL https://arxiv.org/abs/2511.11134.

[30] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. URL https://arxiv.org/abs/2508.02324.

[31] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation, 2025. URL https://arxiv.org/abs/2506.18871.

[32] Ji Xie, Trevor Darrell, Luke Zettlemoyer, and XuDong Wang. Reconstruction alignment improves unified multimodal models, 2025. URL https://arxiv.org/abs/2509.07295.

[33] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Show-o2: Improved native unified multimodal models, 2025. URL https://arxiv.org/abs/2506.15564.

[34] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi, 2024. URL https://arxiv.org/abs/2311.16502.

[35] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15134–15186, 2025.

[36] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model, 2024. URL https://arxiv.org/ abs/2408.11039.

## A Datasets and Evaluation Benchmarks

We evaluate SGU on four representative multimodal benchmarks covering general perception, visual reasoning, mathematical reasoning, and text-centric visual understanding. These datasets provide diverse VQA-style tasks for examining the system-level behavior of UMMs under the SGU protocol.

## A.1 Dataset Descriptions

MMStar [3]. MMStar is a high-quality vision-language benchmark curated to emphasize visiondependent questions. It is designed to reduce the influence of language priors by filtering samples that can be answered reliably without the image, and covers diverse perception-centric skills such as fine-grained recognition, spatial relations, and multi-object reasoning.

MMBench [19]. MMBench is a broad multimodal evaluation suite targeting general visual perception and reasoning. Its questions span recognition and attributes, spatial understanding, commonsense and social reasoning, and multi-step visual reasoning, providing a comprehensive testbed for overall VQA competence under a unified format.

MathVista [20]. MathVista focuses on visually grounded mathematical reasoning, where solving requires jointly interpreting visual evidence, such as geometry diagrams, plots, tables, and charts, and performing mathematical inference. This benchmark is sensitive to errors in visual reconstruction, since small distortions in the reconstructed image may affect the evidence needed for correct reasoning.

Table 5: Dataset splits and primary evaluation focus used in this work.
<table><tr><td>Dataset</td><td>Split</td><td>Type</td><td>Primary focus</td></tr><tr><td>MMStar</td><td>Val</td><td>General</td><td>Vision-dependent reasoning; perception</td></tr><tr><td>MMBench</td><td>Dev</td><td>General</td><td>Visual perception; general reasoning</td></tr><tr><td>MathVista</td><td>Test-mini</td><td>Math &amp; charts</td><td>Geometric structure; mathematical reasoning</td></tr><tr><td>OCR-VQA</td><td>Val</td><td>Text-in-image</td><td>OCR; visual text understanding</td></tr></table>

OCR-VQA [21]. OCR-VQA evaluates reading and reasoning over text embedded in images, including scene text and document-style content. Questions often require precise recognition of characters or words and their context, making it useful for analyzing failures involving text understanding and visual text reconstruction.

## A.2 Summary of Data Usage

Table 5 summarizes the splits and primary evaluation focus for each dataset used in SGU.

## B Implementation Details

This section provides the inference configurations and prompts used to execute the SGU protocol. All experiments are inference-only evaluations without model training or fine-tuning. We run the SGU pipeline on NVIDIA A800 GPUs with 80GB memory; the exact runtime varies across models and datasets because each sample requires captioning, image generation, and final VQA inference. More expensive ablations are conducted on fixed-size subsets to reduce compute while keeping controlled comparisons.

## B.1 Model Inference Configurations

To ensure fair comparison across UMMs, we use a consistent inference protocol whenever possible, while following each model’s native generation interface.

• Understanding phase (I → T). For the image-to-text stage, we use deterministic decoding by setting do\_sample to False. The output length is capped with max\_new\_tokens set to 256, which provides a unified resource constraint and standardizes the textual input to the subsequent generation stage.

• Generation phase (T → I). Generation configurations differ across model architectures. For reproducibility, we fix the global random seed to 42. When applicable, we use a Classifier-Free Guidance (CFG) scale of 5.0. Specific settings follow each model’s native architecture:

– Autoregressive architecture (Janus-Pro-7B): We generate images at a resolution of 512 × 512. The generation process uses 576 image tokens with multinomial sampling and a temperature of 1.0.

– Diffusion andflow-based architectures (BAGEL, OmniGen2, Ovis, UniWorld, Showo2): We use 30 inference steps and generate images at 512 × 512 resolution. When supported by the model, the CFG scale is set to 5.0.

• VQA phase. For the final question-answering stage, we use deterministic decoding by setting do\_sample to False. The output length is restricted with max\_new\_tokens set to 64, since VQA answers are typically concise.

## B.2 Prompt Engineering

The prompts used in each phase of the SGU cycle are listed below. The placeholders {description}, {Question}, and {Options} denote the generated caption, the input question, and the candidate answer options, respectively.

Standard protocol. For the main experiments, we use the following default prompts:

• Phase 1: Understanding (I → T)

<image> Describe this image in detail.

• Phase 2: Generation (T → I)

Generate an image based on the following description: {description}.

• Phase 3: Evaluation (VQA)

– Multiple-choice questions:   
{Question} {Options} Answer with the option letter   
directly.   
– Open-form questions:   
{Question} Answer the question directly.

Prompt variants for sensitivity analysis. For the prompt-sensitivity experiment, we vary one stage at a time while keeping the rest of the SGU pipeline unchanged. The prompt variants are:

• Understanding variants (I → T):

– Var-1 (Und.):   
Describe the visual content of this image in detail.   
– Var-2 (Und.):   
Analyze the image and provide a thorough description   
capturing all key elements.

• Generation variants (T → I):

– Var-1 (Gen.):   
{description}   
– Var-2 (Gen.):   
Generate a high-quality, detailed image of {description}.

## B.3 Stateless Pipeline Execution

SGU is executed in a stateless manner across stages. The captioning, generation, and final VQA steps are run as independent inference calls. Only the explicit intermediate artifact, i.e., the generated caption or reconstructed image, is passed to the next stage; no hidden states, KV-cache, or conversation history are carried over.

Stage isolation. For each stage, we create a fresh model invocation with a clean input payload. We do not reuse the same chat session or inference context across stages, and we avoid accumulating messages in a multi-turn dialog. This ensures that the only information available at each step is what is explicitly provided in that step’s input.

Inputs to each stage. (i) Captioning takes the original image v and the captioning prompt, and outputs a description $t _ { g } .$ . (ii) Generation takes only the text prompt formed from $t _ { g }$ and produces a reconstructed image vˆ. (iii) VQA scoring takes only vˆ and the question q, and returns the answer used to compute $s _ { \mathrm { u m m } } ;$ the original image v and the intermediate caption $t _ { g }$ are not provided to the final VQA stage.

## C Additional Metrics and Theoretical Details

## C.1 Auxiliary Metrics for Isolated Assessment

In addition to the end-to-end SGU score, we report several commonly used isolated metrics to characterize individual components. These metrics are not used as the SGU objective, but serve as auxiliary references for analysis.

FID (Fréchet Inception Distance) [12]. FID measures the distributional distance between two image sets by comparing their feature statistics in a pretrained Inception network. Given feature activations with empirical means and covariances $\left( \mu _ { r } , \Sigma _ { r } \right)$ for real images and $( \mu _ { g } , \Sigma _ { g } )$ for generated images, FID is defined as

$$
\mathrm { F I D } = \Vert \mu _ { r } - \mu _ { g } \Vert _ { 2 } ^ { 2 } + \mathrm { T r } \Big ( \Sigma _ { r } + \Sigma _ { g } - 2 \big ( \Sigma _ { r } \Sigma _ { g } \big ) ^ { 1 / 2 } \Big ) ,\tag{6}
$$

where $\operatorname { T r } ( \cdot )$ denotes the matrix trace.

CLIPScore [11]. We use CLIP embeddings to quantify (i) visual similarity between the original and reconstructed images, and (ii) text–image similarity between the model-generated caption and the original image. Let $f _ { I } ( \cdot )$ and $f _ { T } ( \cdot )$ denote the CLIP image and text encoders, and let $\langle \cdot , \cdot \rangle$ denote cosine similarity. We compute

$$
\mathrm { C L I P - I } = \left. { \frac { f _ { I } ( I ) } { \| f _ { I } ( I ) \| } } , { \frac { f _ { I } ( I ^ { \prime } ) } { \| f _ { I } ( I ^ { \prime } ) \| } } \right. ,\tag{7}
$$

$$
\mathrm { C L I P - T } = \left. \frac { f _ { I } ( I ) } { \| f _ { I } ( I ) \| } , \frac { f _ { T } ( C ) } { \| f _ { T } ( C ) \| } \right. ,\tag{8}
$$

where I is the original image, $I ^ { \prime }$ is the reconstructed image, and $C$ is the intermediate caption produced at the understanding stage.

## C.2 Direct-VQA Reference for SGU

We use direct VQA accuracy on the original image as a model-specific upper-bound reference for interpreting the SGU score:

$$
s _ { \mathrm { b a s e } } = \mathbb { E } _ { ( v , q , a ) \sim \mathcal { D } } \left[ \mathbb { I } \left( \mathrm { M a t c h } \left( \mathcal { M } _ { U } ( v , q ) , a \right) \right) \right] .\tag{9}
$$

Compared with direct VQA, SGU introduces an additional understanding–generation loop before the final answer is produced. The model must first convert the original image into an intermediate textual representation, reconstruct a visual context from $\mathbf { i t } ,$ and then answer the original question using the reconstructed image. Therefore, under the intended semantic-preservation setting, $s _ { \mathrm { b a s e } }$ serves as a natural reference for the best performance the model can achieve without the additional loop.

Proposition (upper-bound reference under semantic preservation). Assume that the SGU loop does not introduce new task-solving semantic evidence beyond what is extracted from the original image by the model’s understanding pathway. Under this assumption, direct VQA accuracy on the original image serves as an upper-bound reference for the SGU score:

$$
s _ { \mathrm { u m m } } \leq s _ { \mathrm { b a s e } } .\tag{10}
$$

Proof. We provide an abstraction-level argument based on semantic information preservation.

Semantic abstraction. Let $m _ { U } ( v )$ denote the semantic information extracted from image v by the visual understanding pathway of the unified model $\mathcal { M } ,$ viewed as a mapping from the visual modality to a semantic space. Similarly, let $m _ { G } ( t )$ denote the semantic information realized in the generated image from text t by the generation pathway of the model.

For a data sample $d = ( v , q , a ) \in \mathcal { D }$ , let $\mathcal { H } _ { v }$ denote the complete semantic space contained in image v. We define the minimal semantic set required to correctly answer the query as

$$
\mathcal { B } _ { d } : = \Big \{ s \subseteq \mathcal { H } _ { v } \mid \mathrm { M a t c h } \left( \mathcal { M } _ { U } ( s , q ) , a \right) = 1 \Big \} .\tag{11}
$$

By construction, $B _ { d } \subseteq { \mathcal { H } } _ { v }$

Correctness sets. We define the sets of correctly answered samples under direct VQA and SGU as

$$
\mathcal { G } _ { \mathrm { b a s e } } : = \Big \{ ( v , q , a ) \in \mathcal { D } \ \big | \ \mathrm { M a t c h } \left( \mathcal { M } _ { U } ( v , q ) , a \right) = 1 \Big \} ,\tag{12}
$$

$$
\mathcal { G } _ { \mathrm { S G U } } : = \Big \{ ( v , q , a ) \in \mathcal { D } \big | \ \mathrm { M a t c h } \big ( \mathcal { M } _ { U } ( \mathcal { M } _ { G } ( \mathcal { M } _ { U } ( v ) ) , q ) , a \big ) = 1 \Big \} .\tag{13}
$$

To show $s _ { \mathrm { u m m } } \leq s _ { \mathrm { b a s e } } .$ , it suffices to show

$$
\mathcal G _ { \mathrm { S G U } } \subseteq \mathcal G _ { \mathrm { b a s e } } .\tag{14}
$$

Consider any sample $d ^ { * } = ( v ^ { * } , q ^ { * } , a ^ { * } ) \in \mathcal { G } _ { \mathrm { S G U } }$ . By definition, the semantic information preserved after the understanding–generation composition,

$$
m _ { G } \circ m _ { U } ( v ^ { * } ) ,\tag{15}
$$

is sufficient to contain the semantic evidence $\boldsymbol { B } _ { d ^ { * } }$ required to answer $q ^ { * }$ correctly.

Under the semantic-preservation assumption, the generation process cannot introduce new tasksolving semantic evidence beyond what is encoded in $m _ { U } ( v ^ { * } )$ . Therefore,

$$
\left( m _ { G } \circ m _ { U } ( v ^ { * } ) \right) \cap \mathcal { B } _ { d ^ { * } } \subseteq m _ { U } ( v ^ { * } ) \cap \mathcal { B } _ { d ^ { * } } .\tag{16}
$$

Since $d ^ { \ast } \in \mathcal { G } _ { \mathrm { S G U } }$ , the SGU loop preserves sufficient information for the correct answer, which implies

$$
{ \cal B } _ { d ^ { * } } = \left( m _ { G } \circ m _ { U } ( v ^ { * } ) \right) \cap { \cal B } _ { d ^ { * } } .\tag{17}
$$

Combining the two relations gives

$$
B _ { d ^ { * } } \subseteq m _ { U } ( v ^ { * } ) \cap B _ { d ^ { * } } \subseteq B _ { d ^ { * } } .\tag{18}
$$

Thus,

$$
{ \cal B } _ { d ^ { * } } = m _ { U } ( v ^ { * } ) \cap { \cal B } _ { d ^ { * } } ,\tag{19}
$$

which implies

$$
B _ { d ^ { * } } \subseteq m _ { U } ( v ^ { * } ) .\tag{20}
$$

Therefore, the original visual understanding representation $m _ { U } ( v ^ { * } )$ already contains sufficient semantic information to answer $q ^ { * }$ correctly, implying $d ^ { \ast } \in \mathcal { G } _ { \mathrm { b a s e } }$

Since $d ^ { * }$ is arbitrary, we conclude

$$
\mathcal { G } _ { \mathrm { S G U } } \subseteq \mathcal { G } _ { \mathrm { b a s e } } ,\tag{21}
$$

and therefore

$$
s _ { \mathrm { u m m } } \leq s _ { \mathrm { b a s e } } .\tag{22}
$$

The above argument provides a semantic-space interpretation of why $s _ { \mathrm { b a s e } }$ is a useful upper-bound reference for SGU. In practice, rare cases may occur where the reconstructed image makes a previously incorrect direct-VQA sample easier to answer. Therefore, we use $s _ { \mathrm { b a s e } }$ as a model-specific reference rather than as an unconditional guarantee for every individual sample.

Figure 7 visualizes this abstraction. In case (a), the understanding–generation composition retains the information needed to answer the question, so the model remains correct after the SGU loop. In case (b), part of the relevant visual information is lost during the loop, leading to an incorrect final answer. This illustrates how SGU reflects the integrated behavior of understanding, intermediate representation, and generation.

## D More SGU Case Examples

Figure 8 provides additional qualitative examples of the full SGU loop. These cases illustrate that the final grounded task can be answered correctly only when the model performs well across the integrated process, including image understanding, intermediate description, visual generation, and final reasoning. Weaknesses in any stage may affect the final answer and are therefore reflected in the SGU evaluation result.

![](images/bf9e7d113ac119364f4fc1693ead30a077aa3746b520d3db9bf5d79ff7286ac8.jpg)  
(a) Semantic-level Abstraction: Good UMM

![](images/bb35f49e17784b7f432d02385aba31613c09433b2d59db48ee1781c387bfa0a2.jpg)  
(b) Semantic-level Abstraction: Weak UMM

Figure 7: Abstraction-level illustration of SGU success and failure. (a) The understanding–generation composition retains the information needed for the target question. (b) Part of the information is lost during the loop, causing the reconstructed image to no longer support the correct answer.  
![](images/57c94fa13f71f2c3db5186fd8096683934770d6cde4e05c330f57fcde2daf4f9.jpg)  
Figure 8: Additional SGU case examples showing the complete loop from the input VQA sample to the intermediate caption, reconstructed image, and final answer.

## E Subset Construction for Ablations

Our ablation studies require repeatedly running the full SGU loop, including captioning, image generation, and self-VQA, under controlled modifications. This is substantially more expensive than isolated evaluation. We therefore construct fixed-size subsets for ablations. These studies do not aim to estimate absolute benchmark performance; instead, they analyze relative changes induced by interventions, such as stage-wise replacement or prompt variation. To ensure comparability, all ablation settings are evaluated on the same fixed subset split, and each subset is first evaluated with the standard SGU protocol as the baseline before applying modifications.

Table 6: Ablation subsets constructed with a fixed random seed (42). For category-labeled datasets, we stratify by dataset categories; for MathVista, we stratify by question format; for OCR-VQA, we uniformly sample due to missing type labels.
<table><tr><td>Dataset</td><td>Subset size N</td><td>Stratification signal</td><td>Preserved composition</td></tr><tr><td>MMStar</td><td>198</td><td>Category label</td><td>6 categories, 33 each</td></tr><tr><td>MMBench</td><td>198</td><td>Category label</td><td>20 categories, proportional</td></tr><tr><td>MathVista</td><td>200</td><td>Question format</td><td>MCP 54% / non-MCP 46%</td></tr><tr><td>OCR-VQA</td><td>200</td><td>N/A</td><td>Uniform sample</td></tr></table>

Table 7: SGU results on the constructed fixed-size subsets. For each dataset, we report direct VQA accuracy on the original image $( s _ { \mathrm { b a s e } } )$ and SGU score $( s _ { \mathrm { u m m } } )$ .
<table><tr><td rowspan="2">Model</td><td colspan="2">MMStar</td><td colspan="2">MMBench</td><td colspan="2">MathVista</td><td colspan="2">OCR-VQA</td><td colspan="2">Avg</td></tr><tr><td>Original Sbase</td><td>SGU Summ</td><td>Original Sbase</td><td>SGU  $s _ { \mathrm { u m m } }$ </td><td>Original Sbase</td><td>SGU Summ</td><td>Original Sbase</td><td>SGU Summ</td><td>Original Sbase</td><td>SGU Summ</td></tr><tr><td>UPPER-BOUND REF.</td><td colspan="2">64.65</td><td colspan="2">86.36</td><td colspan="2">69.19</td><td colspan="2">82.32</td><td colspan="2">75.63</td></tr><tr><td>Janus-Pro-7B</td><td>41.92</td><td>33.84</td><td>78.79</td><td>67.17</td><td>41.92</td><td>33.84</td><td>68.69</td><td>36.87</td><td>57.83</td><td>42.93</td></tr><tr><td>UniWorld-V1</td><td>56.06</td><td>37.37</td><td>84.85</td><td>75.76</td><td>63.64</td><td>38.38</td><td>79.29</td><td>30.30</td><td>70.96</td><td>45.45</td></tr><tr><td>Show-o2-7B</td><td>53.03</td><td>42.93</td><td>84.34</td><td>76.77</td><td>47.98</td><td>41.41</td><td>63.13</td><td>32.32</td><td>62.12</td><td>48.36</td></tr><tr><td>Ovis-U1-3B</td><td>60.61</td><td>44.95</td><td>86.36</td><td>82.32</td><td>68.69</td><td>45.45</td><td>78.28</td><td>33.84</td><td>73.48</td><td>51.64</td></tr><tr><td>BAGEL-7B</td><td>64.65</td><td>40.40</td><td>83.33</td><td>76.77</td><td>69.19</td><td>41.41</td><td>72.22</td><td>49.49</td><td>72.35</td><td>52.02</td></tr><tr><td>OmniGen2</td><td>50.51</td><td>38.89</td><td>81.82</td><td>73.74</td><td>59.60</td><td>38.89</td><td>82.32</td><td>57.07</td><td>68.56</td><td>52.15</td></tr></table>

Stratified sampling. We build each subset via stratified random sampling with a fixed seed (42), preserving the type composition of the corresponding full split when such labels are available. For MMStar and MMBench, we stratify by the provided category labels. For MathVista, which does not provide a unified fine-grained category taxonomy in our setup, we stratify by question format, i.e., multiple-choice versus non-multiple-choice. For OCR-VQA, where questions are open-ended and no category labels are available, we uniformly sample from the split.

Subset specifications. Table 6 summarizes the subset size and stratification granularity used for each dataset. For MMStar and MMBench, we use 198 instances and preserve category distribution; for MMStar this yields a balanced subset across its six categories (33 each). For MathVista, we sample 200 instances while matching the MCP/non-MCP ratio (54% vs. 46%). For OCR-VQA, we uniformly sample 200 instances using the same seed. These constructed subsets are used only for controlled ablations, including stage-wise replacement, prompt sensitivity, and shortcut-cue checks. All main results are reported on the full evaluation splits, while the subsets provide an efficient and consistent testbed for analyzing how specific factors affect SGU within the complete loop.

Representativeness check. To verify that these fixed-size subsets are suitable for ablations, we also run the full SGU protocol on each constructed subset and report the results in Table 7. We observe that the subset-based results broadly follow the same trends as the full-benchmark results in Table 1, including the overall model ordering and the score distribution across datasets. This indicates that the subsets preserve the key evaluation behaviors of the original benchmarks and provide a consistent basis for controlled ablation analyses.

## F Additional Experimental Results

This section reports supplementary quantitative results omitted from the main paper for brevity. We include full results of the stage-wise replacement study on the MathVista and OCR-VQA subsets, as well as intermediate CLIP-based signals collected on the full benchmark evaluation.

Table 8: Stage-wise replacement on the constructed MathVista subset. We report the baseline SGU score and the score after replacing either the captioning or generation stage.
<table><tr><td>Model</td><td>Base</td><td>Cap.</td><td>Gen.</td></tr><tr><td>UniWorld-V1</td><td>38.38</td><td>34.34</td><td>52.02</td></tr><tr><td>Janus-Pro-7B</td><td>33.84</td><td>36.87</td><td>37.88</td></tr><tr><td>Show-o2-7B</td><td>41.41</td><td>39.39</td><td>41.41</td></tr><tr><td>Ovis-U1-3B</td><td>45.45</td><td>46.46</td><td>52.02</td></tr><tr><td>BAGEL-7B</td><td>41.41</td><td>39.90</td><td>46.46</td></tr><tr><td>OmniGen2</td><td>38.89</td><td>42.93</td><td>50.51</td></tr></table>

Table 9: Stage-wise replacement on the constructed OCR-VQA subset. We report the baseline SGU score and the score after replacing either the captioning or generation stage.
<table><tr><td>Model</td><td>Base</td><td>Cap.</td><td>Gen.</td></tr><tr><td>UniWorld-V1</td><td>30.30</td><td>27.78</td><td>69.19</td></tr><tr><td>Janus-Pro-7B</td><td>36.87</td><td>33.84</td><td>58.08</td></tr><tr><td>Show-o2-7B</td><td>32.32</td><td>33.33</td><td>61.62</td></tr><tr><td>Ovis-U1-3B</td><td>33.84</td><td>35.86</td><td>68.69</td></tr><tr><td>BAGEL-7B</td><td>49.49</td><td>48.99</td><td>62.63</td></tr><tr><td>OmniGen2</td><td>57.07</td><td>55.56</td><td>69.19</td></tr></table>

Table 10: CLIP-T similarity between each model-generated caption and the original image across the full evaluation datasets. Higher is better.
<table><tr><td>Model</td><td>MMStar</td><td>MMBench</td><td>MathVista</td><td>OCR-VQA</td></tr><tr><td>Janus-Pro-7B</td><td>0.2841</td><td>0.2765</td><td>0.3102</td><td>0.3499</td></tr><tr><td>BAGEL-7B</td><td>0.2817</td><td>0.2703</td><td>0.3081</td><td>0.3158</td></tr><tr><td>UniWorld-V1</td><td>0.2820</td><td>0.2749</td><td>0.3026</td><td>0.3555</td></tr><tr><td>Show-o2-7B</td><td>0.2790</td><td>0.2704</td><td>0.3050</td><td>0.3189</td></tr><tr><td>OmniGen2</td><td>0.2794</td><td>0.2710</td><td>0.3038</td><td>0.3515</td></tr><tr><td>Ovis-U1-3B</td><td>0.2822</td><td>0.2778</td><td>0.3040</td><td>0.3497</td></tr></table>

Table 11: CLIP-I similarity between the reconstructed image and the original image across the full evaluation datasets. Higher is better.
<table><tr><td>Model</td><td>MMStar</td><td>MMBench</td><td>MathVista</td><td>OCR-VQA</td></tr><tr><td>Janus-Pro-7B</td><td>0.7126</td><td>0.7736</td><td>0.6494</td><td>0.6373</td></tr><tr><td>BAGEL-7B</td><td>0.7213</td><td>0.7577</td><td>0.6873</td><td>0.6881</td></tr><tr><td>UniWorld-V1</td><td>0.5790</td><td>0.6849</td><td>0.5058</td><td>0.3454</td></tr><tr><td>Show-o2-7B</td><td>0.6712</td><td>0.7316</td><td>0.6340</td><td>0.4752</td></tr><tr><td>OmniGen2</td><td>0.6946</td><td>0.7430</td><td>0.6860</td><td>0.6894</td></tr><tr><td>Ovis-U1-3B</td><td>0.7042</td><td>0.7515</td><td>0.6831</td><td>0.4909</td></tr></table>

## F.1 Full Results for Stage-wise Replacement Study

Table 8 and Table 9 provide the complete stage-wise replacement results on the constructed MathVista and OCR-VQA subsets. For each model, we report the baseline SGU score and the SGU score after replacing either the captioning, i.e., image-to-text understanding, stage or the image-generation stage, while keeping all other stages unchanged.

## F.2 Intermediate CLIP-T and CLIP-I Signals on Full Benchmarks

To connect SGU with conventional component-level evaluation signals, we additionally report intermediate similarity metrics. Specifically, we compute CLIP-T between the model-generated caption and the original image, and CLIP-I between the reconstructed image and the original image, both on the full evaluation sets. These metrics are not used to compute SGU, but serve as auxiliary references for interpreting the captioning and reconstruction stages.

## G Additional Discussion on Robustness, Information Bottlenecks, and Score Interpretation

This section provides additional discussion on robustness, information bottlenecks, and score interpretation in the SGU protocol. We focus on three issues: randomness in the evaluation pipeline, the role of the intermediate textual representation, and whether SGU is dominated by a single stage such as captioning or generation.

Random seeds and evaluation robustness. In our experiments, we use consistent random seeds across all models to ensure fair comparison under the same evaluation protocol. We do not explicitly report a full random-seed sweep, since running the complete SGU loop is computationally expensive. Instead, our prompt-sensitivity analysis introduces structured perturbations to the framework by varying the understanding and generation prompts while keeping the rest of the pipeline unchanged. As shown in Table 3, the results remain close to the default setting under these perturbations, suggesting that SGU provides a stable signal under moderate implementation changes, including stochastic factors controlled by the fixed inference setup.

Table 12: Text-bottleneck analyses on OCR-VQA. Left: caption-length sensitivity under caption-only QA with Janus-Pro-7B. Right: caption-only QA comparison with OmniGen2.
<table><tr><td colspan="2">Caption Length</td><td colspan="2">Caption-only QA</td></tr><tr><td>Setting</td><td>Acc.</td><td>Setting</td><td>Acc.</td></tr><tr><td>Janus-Pro-7B, 256 tokens</td><td>49.50</td><td>Original image QA (sbase)</td><td>79.39</td></tr><tr><td rowspan="2">Janus-Pro-7B, 512 tokens</td><td>49.53</td><td>Caption-only QA</td><td>60.05</td></tr><tr><td></td><td>Full SGU (summ)</td><td>56.59</td></tr></table>

Intermediate text bottleneck. The intermediate caption in SGU is a compact textual representation of the model’s visual understanding, rather than a lossless encoding of the image. This bottleneck should be interpreted as part of the evaluated process. If the model fails to extract or express important visual information in the captioning stage, this reflects limitations in its understanding and representation ability under the SGU protocol. In our default setting, the model is explicitly instructed to describe the image in detail. Therefore, missing information in the generated caption is not only an artifact of the framework, but also an informative signal of how well the model can form a general visual representation.

One possible alternative is to condition the captioning stage on the downstream question. While this may reduce information loss for a specific query, it changes the nature of the evaluation: the intermediate representation becomes question-specific rather than a general description of the image. SGU instead adopts a question-agnostic captioning stage to evaluate whether a UMM can first form a general visual description and then use its own generated content to support downstream reasoning. This design better matches our goal of evaluating the model as an integrated system, rather than optimizing the intermediate caption for a particular question.

Caption length. To examine whether the text bottleneck is mainly caused by the caption token limit, we conduct a caption-only QA analysis with different maximum caption lengths. As shown in Table 12, increasing the caption limit from 256 to 512 tokens on OCR-VQA with Janus-Pro-7B changes caption-only QA accuracy only marginally, from 49.50 to 49.53. This suggests that, under our current setting, the observed degradation is not simply determined by the 256-token limit. Instead, it also depends on how effectively the model understands, organizes, and expresses visual information in the intermediate representation.

Understanding and generation bottlenecks. SGU should not be interpreted as being determined by a single bottleneck. Stage-wise replacement results in Section F.1 show that replacing the generation stage often leads to larger improvements, indicating that visual reconstruction is an important bottleneck for current UMMs. However, this does not mean that SGU only measures generation. If the captioning stage fails to extract sufficient visual information, a stronger generator cannot recover information that is absent from the intermediate representation. Conversely, even a good caption may still lead to performance degradation if the model fails to reconstruct a faithful visual context or reason correctly over the reconstructed image.

To further examine the role of the understanding-side intermediate representation, we perform captiononly QA, where the model answers the question directly from its generated caption without the image-generation stage. As shown in Table 12, on OCR-VQA with OmniGen2, direct VQA on the original image achieves 79.39, caption-only QA obtains 60.05, and the full SGU loop obtains 56.59.

The drop from original-image QA to caption-only QA shows that the captioning and reasoning path already introduces substantial degradation, while the further drop under full SGU reflects additional challenges from visual generation and reconstruction. These results support our view that SGU evaluates the combined effects of understanding, intermediate representation, generation, and final reasoning, rather than serving as a metric for any single component alone.

Interpretation of the SGU score. The SGU score is intended as an outcome-based system-level evaluation signal. It does not replace component-wise understanding or generation metrics, nor does it independently attribute errors to a specific stage. Instead, it evaluates whether the UMM can complete the full closed-loop process using its own outputs. Component-level metrics and stage-wise analyses remain useful for diagnosis, while SGU provides a complementary view of whether the integrated understanding-and-generation process succeeds on grounded downstream tasks.