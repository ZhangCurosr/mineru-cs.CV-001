# Advancing MLLM-based UAV Image Understanding and Reasoning: A Benchmark and a Training-Free Multi-Agent System

Haoyu Zhang , Shuoxun Zhang, Peng Ye, Lin Zhang, Jiakang Yuan, Shenghong Yi, Yuening Wang, Tao Chen , Senior Member, IEEE

Abstract—Multimodal Large Language Model (MLLM)-based UAV aerial image understanding and reasoning is essential for aerial intelligence yet poses distinct challenges arising from extreme scale variation, arbitrary camera orientations, and high object density. Despite growing interest, existing evaluations remain fragmented across individual datasets and narrow tasks, leaving a critical gap in unified assessment of UAV understanding and reasoning capabilities. To fill this gap, we construct UAVQA-Bench, a benchmark of 1,500 human-annotated QA pairs drawn from 13 public UAV datasets, covering 6 capability dimensions and 16 tasks in both multiple-choice and visual grounding formats. Systematic evaluation of a broad range of open-source and closed-source MLLMs as well as agent-based systems on UAVQA-Bench identifies three key failure modes: domain-toolset mismatch, unchecked error propagation, and static reasoning. Motivated by these findings, we propose UAV-MAS, a trainingfree multi-agent system for MLLM-based UAV aerial image understanding and reasoning, comprising a Domain-Specific Perception Engine (DSPE) that routes queries to task-appropriate visual tools, a Context-Aware Iterative Refinement module (CAIR) that validates intermediate reasoning to curb error accumulation, and a Difficulty-Aware Adaptive Search mechanism (DAAS) that adjusts search depth to question difficulty. UAV-MAS with a 32B open-source MLLM achieves 77.0% overall accuracy on UAVQA-Bench, surpassing Gemini 3 Pro by 4.0%, while the 8B variant improves 8.7% over its base model.

Index Terms—UAV aerial image analysis, multi-agent systems, multimodal large language models, visual reasoning

## I. INTRODUCTION

Unmanned Aerial Vehicles (UAVs) have evolved into indispensable assets across a spectrum of critical applications, from disaster rescue and precision agriculture to urban traffic patrol and infrastructure inspection. In these high-stakes scenarios, the ability to accurately perceive and interpret complex environments is essential for supporting sophisticated decisionmaking. Although UAVs provide a unique bird’s-eye view that enables comprehensive situational awareness, this perspective introduces a formidable set of perceptual challenges: extreme scale variations resulting from altitude fluctuations, arbitrary object orientations inconsistent with ground-level priors, and a prevalence of tiny, densely packed targets amidst cluttered backgrounds. Consequently, advancing UAV visual perception to handle these intricate dynamics is not merely an optimization task but a fundamental prerequisite for reliable aerial artificial intelligence.

![](images/d7f642bd62635cf02df4d4312c6d0ef93c3c30a96d2215a3380a9d6b5cb5074d.jpg)  
Fig. 1. Comparison of MLLM-based UAV aerial image understanding and reasoning paradigms. (a) Domain-Specific Models handle narrow tasks but fail on complex combinatorial queries. (b) General-Purpose MAS suffer from toolset domain gaps and error propagation in linear static reasoning. (c) UAV-MAS (Ours) addresses all three failure modes via DSPE, CAIR, and DAAS.

Despite significant progress, current UAV perception methods (for example, UAV-DETR [1] for detection and SMDE [2] for depth estimation) remain confined to their respective task domains. As illustrated in Figure 1(a), these task-specific models exhibit two critical limitations when facing comprehensive scenarios. First, they fail to autonomously handle complex combinatorial tasks, as they lack the mechanisms to integrate fragmented visual cues across multi-stage processes such as “Locating the highest building”, which requires sequentially binding depth estimation with object detection. Second, they are completely devoid of the cognitive capacity for high-level reasoning and decision-making, rendering them powerless against open-ended problems like “Assessing if a specific area is suitable for emergency landing”. Generally speaking, existing approaches perform well as specialized sensors but lack the holistic ’eyes’ and logical ’brain’ for complex tasks.

TABLE I  
COMPARISON OF UAVQA-BENCH WITH EXISTING BENCHMARKS. #TASKS REPRESENTS THE TOTAL NUMBER OF SUBTASKS. REGION-LEVEL REFERS TO TASKS WITH PRECISE LOCATION ANNOTATIONS, E.G., BOUNDING BOXES. AND RELATION REFERS TO RELATIONS BETWEEN REGION-LEVEL OBJECTS. <sup>∗</sup> DENOTES HUMAN VERIFICATION AFTER MLLM ANNOTATION GENERATION. RULE-BASED DENOTES AUTOMATIC GENERATION FROM STRUCTURED METADATA WITHOUT AI MODELS. UAVQA-BENCH COVERS WIDE TASK SCOPES AND PROVIDES COMPREHENSIVE FULLY HUMAN-ANNOTATED QUESTIONS, OPTIONS, ANSWERS AND GROUNDING BOUNDING BOXES FOR THE EVALUATION OF MLLMS IN THE UAV DOMAIN.
<table><tr><td rowspan="2">Benchmarks</td><td rowspan="2">Domain</td><td rowspan="2">Response Format</td><td rowspan="2">#Tasks</td><td colspan="3">Scope</td><td rowspan="2">Annotation</td></tr><tr><td>Scene-Level</td><td>Region-Level</td><td>Relation</td></tr><tr><td>MMBench [3]</td><td>General</td><td>QA</td><td>20</td><td>√</td><td>x</td><td>√</td><td>Human</td></tr><tr><td>RSVQA [4]</td><td>Remote Sensing</td><td>QA</td><td>5</td><td>V</td><td>x</td><td>x</td><td>Rule-based</td></tr><tr><td>VRSBench [5]</td><td>Remote Sensing</td><td>QA</td><td>12</td><td>V</td><td>√</td><td>X</td><td>GPT-4V*</td></tr><tr><td>XLRS-Bench [6]</td><td>Remote Sensing</td><td>QA</td><td>16</td><td>√</td><td>V</td><td>√</td><td>GPT-40*</td></tr><tr><td>VisDrone [7]</td><td>UAV</td><td>Bounding Box</td><td>4</td><td>x</td><td>√</td><td>x</td><td>Human</td></tr><tr><td>UAVDT [8]</td><td>UAV</td><td>Bounding Box</td><td>3</td><td>x</td><td>√</td><td>x</td><td>Human</td></tr><tr><td>UrbanVideo-Bench [9]</td><td>UAV</td><td>QA</td><td>16</td><td>√</td><td>X</td><td>x</td><td>Gemini-1.5*</td></tr><tr><td>UAVQA-Bench</td><td>UAV</td><td>QA &amp; Bounding Box</td><td>16</td><td>√</td><td>√</td><td>√</td><td>Human</td></tr></table>

To overcome these limitations, recent Multimodal Large Language Models (MLLMs) offer a promising direction. To advance MLLM-based UAV aerial image understanding and reasoning, we first introduce UAVQA-Bench, a comprehensive and fully human-annotated benchmark. As summarized in Table I, previous comprehensive multimodal benchmarks are largely built on general and remote sensing images, while existing UAV benchmarks still provide only partial coverage of the problem space. In addition, they often rely heavily on automated annotation pipelines, which may compromise data quality. As a comparison, our UAVQA-Bench covers 6 capability dimensions and 16 tasks over 1,500 carefully curated samples drawn from 13 diverse public UAV datasets, and adopts a closed-ended, objectively scorable evaluation protocol that enables reliable and reproducible assessment. All questions, candidate options, answers, and grounding annotations are produced and verified by human annotators, ensuring higher correctness, consistency, and contextual faithfulness.

With the dataset established, a systematic evaluation of a broad range of MLLMs and MLLM-based multi-agent systems is conducted. Initial findings suggest that these methods are generally capable of processing diverse problem types within UAV scenarios, demonstrating an acceptable baseline in general aerial image understanding and reasoning. In detail, closed-source methods show an overall performance advantage. Among open-source methods, larger models generally perform better in aggregate, whereas smaller models can still be stronger on specific tasks such as visual grounding. Despite this functional baseline, our further analysis reveals three recurring failure modes as shown in Figure 1(b): (i) domaintoolset mismatch, where standard vision tools trained on ground-level data degrade on aerial patterns; (ii) unchecked error propagation, where early-stage tool errors cascade through multi-step chains without self-correction; and (iii) static reasoning, where fixed linear strategies fail to adapt to varying aerial task complexity.

To address the above challenges, we propose UAV-MAS, a training-free multi-agent system for MLLM-based UAV aerial image understanding and reasoning. UAV-MAS integrates three targeted components. First, the Domain-Specific Perception Engine (DSPE) equips the system with specialized aerial operators and a precise tool selection mechanism to overcome domain-toolset mismatch on UAV imagery. Second, the Context-Aware Iterative Refinement (CAIR) strategy introduces hierarchical verification that actively detects and corrects reasoning errors, preventing noise accumulation in multistep inference. Third, the Difficulty-Aware Adaptive Search (DAAS) mechanism dynamically adjusts reasoning level based on task complexity, balancing computational efficiency with thoroughness. Experiments demonstrate strong robustness and reasoning accuracy: UAV-MAS with 32B MLLMs achieves 4.0% higher Overall Accuracy on UAVQA-Bench than the closed-source Gemini 3 Pro, while relying solely on trainingfree open-source models.

In summary, our main contributions are:

• We introduce UAVQA-Bench, a comprehensive, fully human-annotated benchmark for UAV aerial image understanding and reasoning. It covers 6 capability dimensions across 16 tasks (in multiple-choice and visual grounding formats), with 1,500 samples drawn from 13 diverse public UAV datasets. We systematically evaluate a broad range of open/closed-source MLLM-based methods on UAVQA-Bench.

• We propose UAV-MAS, a training-free multi-agent system for MLLM-based UAV aerial image understanding and reasoning. It integrates a Domain-Specific Perception Engine (DSPE) for aerial-adapted tool use, a Context-Aware Iterative Refinement (CAIR) strategy for errorresilient multi-step reasoning, and a Difficulty-Aware Adaptive Search (DAAS) mechanism for complexityadaptive inference.

• Extensive experiments show that UAV-MAS achieves state-of-the-art performance, surpassing the powerful closed-source Gemini 3 Pro by 4.0% in Overall Accuracy on UAVQA-Bench using only training-free 32B MLLMs, while offering a more efficient accuracy–cost trade-off than brute-force test-time scaling at comparable accuracy.

## II. RELATED WORK

## A. UAV Aerial Image Analysis

Recent advances in deep learning have spurred the development of specialized models tailored for diverse analysis tasks in the UAV domain, achieving impressive results. These tasks include detection [1], [10]–[14], counting [15]–[17], depth estimation [2], [18]–[20]. Representative detection methods include UAV-DETR [1], an end-to-end detector tailored to small objects in drone images; global–local feature fusion with semantic guidance [13]; and detection-oriented exposure correction for nighttime drone views [14]. Recent methods such as FLDet [21] and TGCADNet [22] respectively emphasize lightweight detection and text-guided contextual modeling for small objects in UAV scenes. Although effective under their respective settings, these methods remain optimized for predefined categories and specific perception tasks. Rex-Omni [23] leverages the strong generalization of MLLMs to enable openworld object detection. It reformulates detection as nextpoint prediction with quantized coordinates and achieves competitive zero-shot performance. For depth estimation, Shi et al. [20] recover scene depth from consecutive UAV frames by combining affine patch transformations with geometric constraints, targeting the narrow-baseline conditions common in aerial sequences.

Despite these advances, existing UAV aerial image analysis models remain confined to isolated tasks and cannot integrate diverse visual capabilities for multi-step reasoning or adaptive decision-making in real-world scenarios.

## B. UAV Aerial Image Understanding Benchmarks

Existing benchmarks for evaluating UAV aerial image understanding and reasoning exhibit notable limitations. Typical datasets such as VisDrone [7], [24] and DOTA [25] primarily serve narrow perception primitives like detection and tracking, falling short of assessing higher-order cognitive capabilities. UAVScenes [26] extends to multi-modal perception and six task types including segmentation, yet its focus remains on perceptual primitives rather than comprehensive reasoning. RSVQA [4] provides rule-generated QA for satellite and aerial orthophotos, but does not target low-altitude UAV scenes. More recent remote-sensing benchmarks, such as VRS-Bench [5] and XLRS-Bench [6], remain dominated by nadir-view perspectives and task designs that fail to capture extreme scale variation, oblique viewpoints, and dense spatial layouts.

Recent benchmarks including UrbanVideo-Bench [9] have attempted to construct more comprehensive evaluation frameworks. However, they generally suffer from incomplete task coverage and incompatible settings, along with reliance on automated annotation pipelines, resulting in data quality that falls short of human-annotated standards.

## C. Multi-Agent System

Multi-agent systems powered by MLLMs have recently gained traction as a framework for tackling complex, multistep visual reasoning tasks, where collaborative agents decompose high-level queries and coordinate visual analysis with reasoning [27]–[36]. ReAct [37] introduces a prompting framework that interleaves reasoning and actions, enabling language models to reason and act jointly. It can reduce hallucination and improve performance in tasks like question answering and decision-making. DyFo [27] adopts a static toolkit architecture and builds a multimodal interaction framework based on ReAct, and utilizes the MCTS search algorithm to achieve a method that enhances the fine-grained visual understanding capability of large-scale multimodal models without training. PyVision [28] adopts the Model Context Protocol (MCP) toolkit as the interaction interface, follows the ReAct paradigm to implement iterative cycles of multi-step reasoning and tool invocation, and utilizes the MCTS search algorithm to enhance the adaptability and interpretability of complex visual reasoning tasks.

Although effective in their domains, these methods are illsuited for UAV aerial image understanding due to groundbiased vision tools, error propagation from tool chaining, and rigid planning that fails to adapt to aerial task complexity— motivating the need for a resilient and dynamically adaptive multi-agent framework tailored to the aerial domain.

## III. UAVQA-BENCH

To address the limitations in existing UAV aerial image understanding and reasoning evaluation platforms, we introduce “UAVQA-Bench”, a comprehensive benchmark covering multi-task, multi-scenario, and multi-scale settings. As shown in table I, UAVQA-Bench features comprehensive scopes including scene-level, region-level and relation between regional objects, which ensures a holistic evaluation beyond the scope of existing UAV datasets. All annotations are provided by highly educated human annotators under an inspection-revision process. Compared to synthetic annotations in benchmarks like VRS-Bench [5] and UrbanVideo-Bench [9], human-crafted annotations demonstrate superior quality in terms of correctness, consistency, and contextual understanding over synthetic outputs from large models. UAVQA-Bench assesses 6 key capabilities of UAV agent systems and encompasses 16 distinct tasks, systematically examining UAV agents’ multi-dimensional understanding and reasoning abilities in complex environments. Notably, our evaluation adopts a closed-ended QA (multiple-choice) and grounding box matching approach, which simplifies assessment and enhances reproducibility by eliminating the ambigu ity inherent in open-ended responses, while ensuring reliable evaluation through standardized answer spaces.

## A. Dataset Construction and Data Sources

UAVQA-Bench comprises 1,500 samples, each pairing a high-quality UAV image with a carefully constructed question–answer instance. To ensure task relevance and visual diversity, we manually selected images from 13 publicly available UAV datasets: AU-AIR [38], WebUAV-3M [39], VisDrone-DET2019 [40], Semantic Drone [41], DroneVehicle [42], UAVDT [8], VDD [43], UDD [44], UAVid [45], WildUAV [46], HazyDet [47], AnimalDrone [48], and UAV123 [49], with their task-level provenance summarized in table II. Dataset construction followed a multi-stage quality-control process involving seven volunteers with backgrounds in computer vision: two selected suitable images and annotated bounding boxes where necessary, two constructed the questions, answer options, and ground-truth answers, and the remaining three independently reviewed the image quality, bounding-box annotations, question and option quality, and answer correctness. For tasks requiring distractor options, three distractors were selected from six LLM-generated candidates. Only samples receiving unanimous approval were retained; the remaining samples were revised or removed, followed by random spot checks of the resulting dataset.

![](images/e19d5aae0f46a080ff544aac93b9ac3129bdf40f2c23d758cc62b39eb60eab18.jpg)  
Fig. 2. Overview of UAVQA-Bench. The benchmark assesses 6 key capabilities through 16 distinct tasks, collectively forming a comprehensive evaluation framework.

TABLE II DATA SOURCES OF EACH TASK.
<table><tr><td rowspan=1 colspan=2>Capability</td><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>Data Source</td></tr><tr><td rowspan=3 colspan=2>Existence Detection</td><td rowspan=1 colspan=1>Scene Presence</td><td rowspan=1 colspan=1>WebUAV-3M, VisDrone-DET2019, Semantic Drone</td></tr><tr><td rowspan=1 colspan=1>on</td><td rowspan=1 colspan=1>Conditional Presence</td><td rowspan=1 colspan=1>AU-AIR, WebUAV-3M, Semantic Drone</td></tr><tr><td rowspan=1 colspan=1>Multi-Target Presence</td><td rowspan=1 colspan=1>AU-AIR, WebUAV-3M, VisDrone-DET2019, Semantic Drone, UAVDT, HazyDet</td></tr><tr><td rowspan=1 colspan=2>Category Recognition</td><td rowspan=1 colspan=1>Regional Classification</td><td rowspan=1 colspan=1>AU-AIR, VisDrone-DET2019</td></tr><tr><td rowspan=3 colspan=2>Quantity Awareness</td><td rowspan=1 colspan=1>Scene Counting</td><td rowspan=1 colspan=1>AU-AIR, WebUAV-3M, Semantic Drone, AnimalDrone</td></tr><tr><td rowspan=1 colspan=1>Regional Counting</td><td rowspan=1 colspan=1>AU-AIR, WebUAV-3M, VisDrone-DET2019, Semantic Drone, AnimalDrone</td></tr><tr><td rowspan=1 colspan=1>Number Comparison</td><td rowspan=1 colspan=1>AU-AIR, WebUAV-3M, VisDrone-DET2019, Semantic Drone, UAVDT, HazyDet, AnimalDrone, UAV123</td></tr><tr><td rowspan=3 colspan=2>Fine-grained AttributePerception</td><td rowspan=1 colspan=1>Regional Attribute Recognition</td><td rowspan=1 colspan=1>AU-AIR, VisDrone-DET2019, DroneVehicle</td></tr><tr><td rowspan=1 colspan=1>Function Recognition</td><td rowspan=1 colspan=1>VisDrone-DET2019, VDD, UDD</td></tr><tr><td rowspan=1 colspan=1>Safety Landing</td><td rowspan=1 colspan=1>VisDrone-DET2019</td></tr><tr><td rowspan=3 colspan=2>Spatial RelationshipUnderstanding</td><td rowspan=1 colspan=1>Spatial Relation</td><td rowspan=1 colspan=1>AU-AIR, VisDrone-DET2019, Semantic Drone</td></tr><tr><td rowspan=1 colspan=1>Height Comparison</td><td rowspan=1 colspan=1>VisDrone-DET2019, UAVDT, VDD, UDD, UAV123</td></tr><tr><td rowspan=1 colspan=1>Distance Comparison</td><td rowspan=1 colspan=1>AU-AIR, VisDrone-DET2019, UAVDT, VDD, UDD, UAVid, WildUAV</td></tr><tr><td rowspan=2 colspan=2>Visual Grounding</td><td rowspan=1 colspan=1>Simple Object Grounding</td><td rowspan=1 colspan=1>AU-AIR, VisDrone-DET2019</td></tr><tr><td rowspan=1 colspan=1>Complex Semantic Grounding</td><td rowspan=1 colspan=1>AU-AIR, WebUAV-3M</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Highest Object Grounding</td><td rowspan=1 colspan=1>WebUAV-3M, VDD, Semantic Drone</td></tr></table>

For region-level tasks, we manually annotate the selected images with precise bounding boxes. We then design taskspecific question templates with diverse linguistic expressions to reduce the risk of models exploiting superficial patterns. Candidate answers for multiple-choice questions include plausible visual or semantic distractors, while all questions, options, ground-truth answers, and grounding annotations are manually authored and rigorously cross-verified. For tasks requiring spatial outputs, such as visual grounding, coordinates are represented as $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \}$ , where $( x _ { 1 } , y _ { 1 } )$ and $( x _ { 2 } , y _ { 2 } )$ denote the top-left and bottom-right corners, respectively. All coordinates are normalized by image width and height and rounded to three decimal places for consistent spatial representation.

## B. Capability Dimensions

As illustrated in Figure 2, UAVQA-Bench establishes 6 core capabilities comprising 16 diverse tasks, ranging from basic perception to high-level reasoning.

• Existence Detection (ED) evaluates the agent’s ability to identify object presence under varying conditions, including Scene Presence, Conditional Presence, and Multi-Target Presence.

• Category Recognition (CR) focuses on fundamental semantics through Regional Classification of detected entities.

![](images/caa522f252ce8d41266e9f2586ebafb3d08870a5e40e1da591f273496bb72f3d.jpg)  
(a) Capability composition

![](images/ae1505c5fe7a5c47024d0be060df8c3fbd7ab5c76713f5fd1a494fab3fa6aa9f.jpg)  
(b) Target spatial distribution

![](images/8d416500e984d90811d815bcdc2433e3da54a0b1a7733b7cdd3e99a68b657cab.jpg)  
(c) High-frequency vocabulary

![](images/b78ac25dd2e8072e960a6a0eceb666c7f81b4bffe4fef1912faba30340046404.jpg)  
(d) Question token lengths

![](images/922f1591f12316b06eef0262c5f14127f7dbc3a0bf6223532f1ccfabc0b96e2a.jpg)  
(e) Image resolutions

![](images/39fe8176187b7544ce57c12cb78e4f14f173b1121073c9ece16558e4f7ddb942.jpg)  
(f) Bounding-box areas

![](images/cf02c92ce15efc631296f35487c53df2bc0f9cb9130a5062686d87f25c665e95.jpg)  
(g) Bounding-box shapes  
Fig. 3. Dataset statistics of UAVQA-Bench. (a) Composition of the six capability dimensions. (b) Spatial distribution of target centers. (c) High-frequency vocabulary. (d) Question token-length distribution. (e) Image-resolution distribution. (f) Bounding-box area relative to image area. (g) Bounding-box aspect ratio distribution.

• Quantity Awareness (QA) needs precise counting and numerical logic via Scene/Regional Counting and Number Comparison.

• Fine-grained Attribute Perception (FAP) assesses the agent’s granular understanding of entity properties and scene affordances. It encompasses Attribute Recognition (e.g., identifying colors and orientations), Function Recognition of specific objects, and Safety Landing analysis, which requires the model to evaluate surface conditions to identify secure touchdown zones for the UAV.

• Spatial Relationship Understanding (SRU) examines 3D spatial reasoning from a distinctive aerial perspective. Beyond identifying Spatial Relations between discrete entities or their relative positioning to the UAV, this dimension includes Height Comparison and Distance Comparison. Here, “higher” denotes greater physical elevation in 3D space rather than a smaller image-plane vertical coordinate, while distance denotes the camera-to-object range. Agents must therefore reason about relative elevation and depth in complex, multi-level environments.

• Visual Grounding (VG) tests the precise alignment between complex linguistic queries and visual regions. Spanning from Simple Object Grounding to Complex Semantic Grounding with multi-step reasoning, this capability evaluates the agent’s cross-modal comprehension. Notably, it includes Highest Object Grounding, which challenges the model to identify and localize the vertically most prominent entities, such as the tallest building in a dense urban scene.

## C. Evaluation Protocol and Metrics

Most UAVQA-Bench tasks adopt a multiple-choice format, whereas visual grounding tasks require a predicted bounding box. A multiple-choice instance is counted as correct when the model selects the ground-truth option. For visual grounding, the model is explicitly prompted to return coordinates in the same normalized format used by the annotations, and a prediction is counted as correct when its Intersection over Union (IoU) with the ground-truth box reaches or exceeds 0.5.

We report two complementary aggregate metrics: Overall Accuracy (OA) and Average Accuracy (AA). Let T = 16 denote the number of tasks, and let $m _ { i }$ and $c _ { i }$ denote the total and correctly answered samples in task $i ,$ respectively. The metrics are defined as

$$
\mathrm { O A } = \frac { \sum _ { i = 1 } ^ { T } c _ { i } } { \sum _ { i = 1 } ^ { T } m _ { i } } \times 1 0 0 \% ,\tag{1}
$$

$$
\operatorname { A A } = { \frac { 1 } { T } } \sum _ { i = 1 } ^ { T } { \frac { c _ { i } } { m _ { i } } } \times 1 0 0 \% .\tag{2}
$$

OA measures sample-weighted performance over the complete benchmark, while AA assigns equal weight to every task. Reporting both metrics therefore captures overall robustness without allowing tasks with more samples to obscure performance differences across capability dimensions.

## D. Dataset Statistics

Figure 3 provides a unified statistical overview of UAVQA-Bench. The benchmark maintains a relatively balanced composition across the six capability dimensions (Figure 3(a)), covering both foundational perception and more challenging reasoning-oriented tasks. Target objects are broadly distributed across the image plane rather than concentrated in a small set of locations (Figure 3(b)). The word cloud in Figure 3(c) further summarizes the benchmark’s lexical coverage across object categories, attributes, spatial relationships, and actions. Non-content words, such as prepositions and formatting tokens, are removed so that font size directly reflects the relative frequency of semantically meaningful terms.

![](images/2751a5aa35da7ca7ad39739c207c648d56fd91ae312761f1b7d1116bceab8395.jpg)  
Fig. 4. Overview of UAV-MAS. (a) The overall pipeline of UAV-MAS. (b) The Domain-Specific Perception Engine (DSPE) tailored for UAV perception. (c) The Context-Aware Iterative Refinement (CAIR) strategy to mitigate tool invocation noise. (d) The Difficulty-Aware Adaptive Search (DAAS) mechanism for dynamic reasoning exploration

UAVQA-Bench also presents substantial diversity in both textual and visual inputs: question lengths range from short queries to long compositional reasoning prompts, while image resolutions span from low-resolution to ultra-high-resolution aerial imagery (Figure 3(d) and (e)). The bounding-box statistics in Figure 3(f) and (g) further reveal the fine-grained perception challenge. Specifically, 49.2% of the boxes occupy only 0.1%–1% of the image area, and another 17.5% occupy less than 0.1%; by contrast, just 0.1% cover more than 50% of an image. Near-square boxes, with width-to-height ratios between 3:4 and 4:3, form the largest group, although the complete distribution spans a wide range of object shapes. Together, these statistics demonstrate diversity in capability coverage, target location and geometry, language, and image resolution, making UAVQA-Bench a challenging testbed for multimodal agents in UAV scenarios.

## IV. METHOD

Evaluating MLLM-based methods on UAVQA-Bench (Table III) exposes three key failures: domain-toolset mismatch, error propagation, and static reasoning. To address these, we propose UAV-MAS (Figure 4), a training-free multi-agent system comprising three core modules. First, the Domain-Specific Perception Engine (DSPE) provides aerial-targeted tools to overcome domain mismatch. Second, Context-Aware Iterative Refinement (CAIR) utilizes step-level verification to correct unreliable feedback and halt error propagation. Third, Difficulty-Aware Adaptive Search (DAAS) replaces static reasoning with complexity-adaptive exploration by pruning paths based on query difficulty. Together, these modules enable UAV-MAS to achieve robust aerial image understanding and reasoning performance without task-specific training.

## A. Domain-Specific Perception Engine

While general-purpose Multimodal Large Language Models (MLLMs) generalize well to common scenes, aerial images in UAV understanding and reasoning tasks expose structural challenges that standard models cannot adequately handle: extreme scale variation, arbitrary object orientation, and highdensity small object clutter. Generic models also lack priors for aerial-specific reasoning subtasks (e.g., distinguishing groundlevel from rooftop targets, counting tiny objects under dense occlusion). To address these gaps, we design the Domain-Specific Perception Engine (DSPE): a modular suite of aerial-targeted tools combined with a per-tool activation mechanism for lightweight scheduling.

Domain-Specific Toolkit. The toolkit comprises 5 core tools targeting the principal visual bottlenecks in overhead imagery: 1) Context-Aware Zooming employs a soft-boundary mechanism that retains edge context during cropping to enhance local resolution while preserving semantic continuity, particularly important for aerial targets that often span only a few pixels at typical UAV altitudes. 2) Fine-Grained Explicit Description converts implicit visual features (e.g., textures, behaviors) into structured text via instruction following, supporting downstream reasoning. 3) Distance Estimation utilizes Depth Anything 3 [50] to generate pseudo-depth maps for reconstructing 3D structures from monocular imagery, distinguishing ground-level from rooftop targets based on relative depth discontinuities. 4) Semantic Grounding locates individual targets based on explicit instructions $\left( \mathrm { e . g . , \ ^ { 6 6 } I } \right.$ ocate the red vehicle”) for precise spatial positioning. 5) Open-Vocabulary Detection with De-hallucination addresses hallucination patterns specific to dense aerial scenes through a two-stage verification. First, prior to localizing, an MLLMbased existence verification confirms whether the target category is present at all, preventing the model from forcibly generating bounding boxes for absent targets. Second, to address the issue where MLLMs systematically hallucinate bounding boxes at regular intervals during autoregressive decoding, we apply a heuristic filter. It removes a detection $c _ { i }$ only when the equidistant condition $| { c } _ { i + 1 } - 2 { c } _ { i } + { c } _ { i - 1 } | < \delta$ holds continuously across multiple consecutive detections, preserving legitimate near-uniform layouts (e.g., a row of parked vehicles) while suppressing fabricated repetitive patterns.

Per-Tool Activation. In aerial image analysis, tool selection is query-type-dependent: counting tasks require detection while spatial reasoning tasks require depth estimation rather than description. Small open-source MLLMs frequently fail when asked to jointly select and configure multiple tools from a single prompt, due to context overflow or hallucinated activations. To address this, each tool is assigned a dedicated lightweight agent $( \mathrm { A g e n t _ { T S } } )$ responsible for a single binary decision: “Should this tool be activated for the current imagequery pair?” If activated, $\mathrm { { A g e n t } _ { T S } }$ additionally generates the required execution parameters.

Together, the 5 aerial-targeted tools and per-tool activation ensure that the base MLLM receives enriched, task-relevant perceptual inputs without needing to manage complex multitool scheduling—directly addressing the domain gap in UAV aerial image understanding and reasoning.

## B. Context-Aware Iterative Refinement (CAIR)

In aerial image QA, tool feedback (e.g., from object detectors) is inherently noisy due to occlusion, degraded image quality, and the small scale of targets in overhead imagery. To prevent such errors from corrupting the accumulated reasoning state, we propose the Context-Aware Iterative Refinement (CAIR) strategy. CAIR augments the standard ReAct loop with a step-level verification stage: after each tool call, a dedicated agent assesses whether the new evidence is reliable and consistent, and selectively updates the answer and accumulated key clues accordingly.

Global Reasoning Track. The Strategic Reasoning Agent $( \bf { A g e n t } _ { S R } )$ maintains a ReAct [37] loop over the dialogue history $\mathcal { H } _ { t - 1 }$ of the current trace, selecting actions $A _ { t }$ from the tool set provided by DSPE and accumulating tool feedback $F _ { t } \colon$

$$
\mathcal { H } _ { t } = \mathcal { H } _ { t - 1 } | | \{ A _ { t } , F _ { t } \} .\tag{3}
$$

Step-level Verification. At each reasoning step, the Perceptual Verification Agent $( \mathbf { A g e n t _ { P V } } )$ receives only the current step’s data: image I, query $Q ,$ action $A _ { t } .$ , and tool feedback $F _ { t } .$ , without access to the accumulated history. This isolation prevents historical anchoring bias and produces a draft answer $\hat { a } _ { t }$ and a compact evidence summary $\hat { C } _ { t }$ . The Contextual Integration Agent $( \mathbf { A g e n t } _ { \mathbf { C I } } )$ does not receive the raw tool output $F _ { t } \mathbf { ; }$ ; instead, it compares $( \hat { a } _ { t } , \hat { C } _ { t } )$ with the original image–query context and the previous state $( a _ { t - 1 } , C _ { t - 1 } )$ . The image I and query $Q$ are persistent shared context and are omitted from repeated agent-call notation for brevity. The resulting state update is

Algorithm 1 Context-Aware Iterative Refinement (CAIR)   
1: Input: Task Query ${ \overline { { Q , } } }$ Image I, Iteration Limit $\overline { { T } }$   
2: Output: Single Trace Final Answer $a _ { \mathrm { t r a c e } }$   
3: Initialize: $\mathcal { H } _ { 0 }  \{ Q , I \} , a _ { 0 }  \nabla \mathbf { L } \mathbf { M } ( Q , I )$   
4: $C _ { 0 } \gets \emptyset , \ : T _ { \mathrm { o p t } } \gets \mathrm { T o o l S e l e c t i o n } ( Q , I , \mathcal { T } )$   
5: for $t = 1$ to T do   
6: Phase $\imath \imath$ Global Reasoning (ReAct)   
7: $A _ { t } \gets \mathrm { A g e n t } _ { \mathrm { S R } } ( \mathcal { H } _ { t - 1 } , \mathcal { T } _ { \mathrm { o p t } } )$   
8: $F _ { t } \gets \mathrm { E x e c u t e } ( A _ { t } )$   
9: Get $\mathcal { H } _ { t }$ based on Eq. 3   
10: Phase 2: Step-level Verification   
11: $\hat { a } _ { t } , \hat { C } _ { t } \gets \mathrm { A g e n t } _ { \mathrm { p V } } ( I , Q , A _ { t } , F _ { t } )$   
12: $( a _ { t } , C _ { t } ) \gets \mathrm { A g e n t } _ { \mathrm { C I } } ( \hat { a } _ { t } , \hat { C } _ { t } , a _ { t - 1 } , C _ { t - 1 } )$ via Eq. 4   
13: end for   
14: Phase 3: Answer Arbitration   
15: $a _ { \mathrm { e n d } } , C _ { \mathrm { e n d } } \gets \mathrm { A g e n t _ { L A } } ( \mathcal { H } _ { T } )$   
16: $a _ { \mathrm { t r a c e } } \gets \mathrm { A g e n t } _ { \mathrm { C I } } ( a _ { \mathrm { e n d } } , C _ { \mathrm { e n d } } , a _ { T } , C _ { T } )$   
17: Return $a _ { \mathrm { t r a c e } }$

$$
\begin{array} { r } { ( a _ { t } , C _ { t } ) = \left\{ \begin{array} { l l } { ( \hat { a } _ { t } , \hat { C } _ { t } ) } & { \mathrm { i f ~ t r u s t e d ~ a n d ~ } \hat { a } _ { t } \neq a _ { t - 1 } , } \\ { ( a _ { t - 1 } , C _ { t - 1 } | | \hat { C } _ { t } ) } & { \mathrm { i f ~ t r u s t e d ~ a n d ~ } \hat { a } _ { t } = a _ { t - 1 } , } \\ { ( a _ { t - 1 } , C _ { t - 1 } ) } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \end{array}\tag{4}
$$

where || denotes text-level concatenation, $a _ { t }$ is the current best answer, and $C _ { t }$ is a structured text record of accumulated key clues. Because the extracted clues are tightly coupled with the specific answer hypothesis, the state is fully replaced when the draft answer changes to avoid evidence contamination between conflicting hypotheses; when it remains consistent, new clues are safely appended to accumulate evidence.

Answer Arbitration. Upon completion of the reasoning loop, the Long-Chain Answer Agent $( \mathbf { A g e n t } _ { \mathbf { L A } } )$ synthesizes an answer $a _ { \mathrm { e n d } }$ and a clue summary $C _ { \mathrm { e n d } }$ from the accumulated history $\mathcal { H } _ { T }$ of that trace. The Contextual Integration Agent $( \mathbf { A g e n t } _ { \mathbf { C I } } )$ then selects the more reliable result between $a _ { \mathrm { e n d } }$ and the final iterative state $a _ { T }$ as the trace output $a _ { \mathrm { t r a c e } }$

CAIR’s step-level verification protects the answer–evidence state within each trace. During DAAS, every child receives a copy of its parent’s state and maintains an independent history, so feedback on an unreliable or pruned branch never enters a sibling branch. The complete procedure is summarized in Algorithm 1.

## C. Difficulty-Aware Adaptive Search (DAAS)

While CAIR mitigates noise at each reasoning step, it follows a fixed linear chain that may miss better reasoning paths for complex aerial queries. In UAV visual question answering, query difficulty varies substantially: a simple presence check requires few reasoning steps, while a compositional query (e.g., counting objects satisfying multiple spatial and attribute conditions) demands deeper, multi-step exploration. Applying a fixed pruning threshold across all queries wastes computation on easy queries and prematurely cuts off exploration on hard ones. To address this, we propose the Difficulty-Aware Adaptive Search (DAAS) strategy. Unlike standard beam search, which applies a single fixed pruning threshold and selects the final-node answer, DAAS derives a per-query pruning threshold from estimated query difficulty and selects the optimal path by global coherence across all nodes. DAAS governs tree expansion through three phases: Adaptive Initialization, Consistency-Guided Branch Control, and Optimal Path Selection.

TABLE III  
COMPARISON WITH OTHER METHODS ON OUR UAVQA-BENCH. GEN. AND SPEC. DENOTE THE USE OF THE ORIGINAL GENERAL-PURPOSE TOOLKIT AND OUR DOMAIN-SPECIFIC TOOLKIT, RESPECTIVELY, BOTH EQUIPPED WITH QWEN3-VL AS THE BACKBONE. OA AND AA DENOTE OVERALL ACCURACY AND AVERAGE ACCURACY, RESPECTIVELY. THE GLM4.6V RESULT UNDER UAV-MAS IS A SUPPLEMENTAL BACKBONE TEST. HUMAN AVG. REPORTS THE AVERAGE PERFORMANCE OF 10 PARTICIPANTS. ALL VALUES ARE REPORTED IN PERCENTAGES (%). BOLD INDICATES THE BEST MODEL PERFORMANCE, EXCLUDING THE HUMAN BASELINE.
<table><tr><td>Category</td><td>Method</td><td>Model</td><td>OA</td><td>AA</td><td>ED</td><td>CR</td><td>QA</td><td>FAP</td><td>SRU</td><td>VG</td></tr><tr><td colspan="3">Human Avg. (10 participants)</td><td>87.87</td><td>86.14</td><td>84.33</td><td>90.40</td><td>85.43</td><td>92.50</td><td>80.80</td><td>83.39</td></tr><tr><td rowspan="7">Open Source</td><td rowspan="7">Qwen3-VL [51]</td><td>Qwen3-VL 8B Instruct Qwen3-VL 8B Thinking</td><td>61.73 52.60</td><td>59.63 50.88</td><td>72.33 79.00</td><td>52.80 50.40</td><td>64.57 64.86</td><td>46.50 44.50</td><td>41.60 39.60</td><td>80.00 26.91</td></tr><tr><td>Qwen3-VL 30BA3B Instruct</td><td>61.13</td><td>59.72</td><td>75.00</td><td>54.40</td><td>58.57</td><td>51.50</td><td>43.20</td><td>75.64</td></tr><tr><td></td><td>62.00</td><td>60.03</td><td></td><td></td><td>69.43</td><td>53.50</td><td>56.00</td><td>50.91</td></tr><tr><td>Qwen3-VL 30BA3B Thinking</td><td></td><td></td><td>78.33</td><td>52.00</td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-VL 32B Instruct</td><td>69.60</td><td>68.63</td><td>80.00</td><td>67.20</td><td>67.71</td><td>58.00</td><td>59.60</td><td>79.27</td></tr><tr><td>Qwen3-VL 32B Thinking</td><td>68.27</td><td>67.06</td><td>77.33</td><td>63.20</td><td>67.14</td><td>57.00</td><td>58.40</td><td>79.27</td></tr><tr><td>InternVL3.5 [32] InternVL3.5 38B</td><td></td><td>40.53 39.18 55.13 55.73</td><td>58.00 77.33</td><td></td><td>36.00 66.40</td><td>53.71 61.14</td><td>36.50 47.20 60.50 51.20</td><td>3.63 17.82</td></tr><tr><td>GLM4.6V [52] Qwen3.5 [53]</td><td>GLM4.6V 9B Qwen3.5 9B</td><td>65.27 63.80</td><td>63.57</td><td>72.00</td><td>57.60</td><td>67.71</td><td>51.00</td><td>56.40</td><td>76.73</td></tr><tr><td></td><td>ChatGPT 5.2 Pro</td><td>57.13</td><td>62.81 56.72</td><td>65.67 77.67</td><td>59.20 60.80</td><td></td><td>65.14 71.43</td><td>57.5 50.80 58.50 67.20</td><td>78.55 4.73</td></tr><tr><td rowspan="2">Closed Source VLM</td><td>ChatGPT [54]</td><td>ChatGPT 5.2</td><td>53.20</td><td>52.30</td><td>75.67</td><td>57.60</td><td>69.43</td><td>46.50</td><td>58.80</td><td>5.82</td></tr><tr><td>Gemini [55]</td><td>Gemini 3 Pro Gemini 3 Flash</td><td>73.00 70.73</td><td>71.25 68.78</td><td>79.67 82.33</td><td>60.80 59.20</td><td>76.29 71.71</td><td>68.50 60.50</td><td>60.80 56.40</td><td>81.45 82.55</td></tr><tr><td rowspan="7">Agent Based Method</td><td>DyFo [27]</td><td>Qwen3-VL 8B Qwen3-VL 32B</td><td>46.47 51.67</td><td>48.31 52.81</td><td>61.00 70.00</td><td>64.00 64.00</td><td>48.86 57.14</td><td>63.00 61.50</td><td>35.20 58.40</td><td>17.82 5.82</td></tr><tr><td rowspan="6">Qwen-Agent [56]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-VL 8B-Gen.</td><td>60.87</td><td>59.11</td><td>75.67</td><td>52.00</td><td>61.43</td><td>52.00</td><td>41.20 51.60</td><td>72.36 73.09</td></tr><tr><td>Qwen3-VL 8B-Spec. Qwen3-VL 32B-Gen.</td><td>61.13</td><td>60.46</td><td>71.00</td><td>60.00</td><td>56.57</td><td>50.50 72.50</td><td>58.40</td><td>66.91</td></tr><tr><td>Qwen3-VL 32B-Spec.</td><td>71.13</td><td>71.33</td><td>74.33</td><td>78.40</td><td>77.43</td><td></td><td></td><td>72.00</td></tr><tr><td></td><td>69.40</td><td>69.63</td><td>63.67</td><td>71.20</td><td>71.43</td><td>71.50</td><td>68.00</td><td></td></tr><tr><td>Qwen3-VL 8B-Gen.</td><td>44.80</td><td>44.20</td><td>64.00</td><td>48.80</td><td>63.14</td><td></td><td>51.50 35.20</td><td>2.55</td></tr><tr><td rowspan="4">PyVision [28]</td><td>Qwen3-VL 8B-Spec.</td><td>50.87</td><td>49.16</td><td>62.67</td><td></td><td>45.60</td><td>63.71</td><td>44.50</td><td>39.20</td><td>39.27</td></tr><tr><td>Qwen3-VL 32B-Gen.</td><td>50.60</td><td>51.74</td><td>63.67</td><td>68.00</td><td>69.71</td><td></td><td>67.50</td><td>41.20</td><td>0.36</td></tr><tr><td>Qwen3-VL 32B-Spec.</td><td>52.33</td><td>52.53</td><td>62.67</td><td>58.40</td><td></td><td>61.43</td><td>59.00</td><td>46.40</td><td>27.27</td></tr><tr><td>Qwen3-VL 8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>66.80</td><td>80.36</td></tr><tr><td rowspan="3">Ours</td><td rowspan="3">UAV-MAS</td><td>Qwen3-VL 32B</td><td>70.47 77.00</td><td>68.94 76.03</td><td>77.67 77.00</td><td>61.60 70.40</td><td>69.71 80.57</td><td>57.50</td><td></td><td>84.00</td></tr><tr><td></td><td></td><td>67.67</td><td>75.67</td><td>56.80</td><td>70.86</td><td>75.00 54.00</td><td>69.20</td><td>77.09</td></tr><tr><td>GLM4.6V 9B Qwen3.5 9B</td><td>69.67 73.67</td><td>73.81</td><td>74.00</td><td>70.40</td><td>74.57</td><td>70.50</td><td>71.60 67.20</td><td>81.81</td></tr></table>

Adaptive Initialization. Before searching, the Score Agent $( \mathbf { A g e n t _ { S C } } )$ assigns the base model’s direct response an initial score $S _ { \mathrm { i n i t } } \in [ 0 , 1 0 ]$ as a coarse difficulty indicator. We use a fixed mapping: $S _ { \mathrm { i n i t } } \in [ 0 , 3 ] , [ 4 , 8 ]$ , and [9, 10] correspond to τ = 2, 4, and 6, respectively. Larger thresholds avoid unnecessary expansion for high-confidence queries, while smaller thresholds preserve exploration for low-confidence queries.

Consistency-Guided Branch Control. We manage tree expansion through a Consecutive Consistency check. During expansion, the system generates W candidate successor states via CAIR using temperature-based sampling, where $W$ controls branching width. The Score Agent $( \mathbf { A g e n t _ { S C } } )$ evaluates each candidate’s quality to output a new score $S ^ { \prime }$ . Branch continuation is decided by coherence over consecutive steps:

• Prune Branch: We prune only if both the current node and the new candidate fall below the threshold simultaneously:

$$
\mathrm { P r u n e } \iff ( S ^ { \prime } < \tau ) \wedge ( S < \tau ) .\tag{5}
$$

This allows a single low-confidence step to persist if its adjacent step remains high-confidence. The key insight is that intermediate tool calls in aerial reasoning (e.g., a zoom step returning a blurry sub-patch before a subsequent description step integrates the result) may transiently score low without indicating a dead branch. Requiring two consecutive low-confidence steps before pruning reduces false terminations while discarding genuinely unproductive paths.

• Expand Branch: If at least one step in the consecutive pair maintains high confidence, we create a new node $N ^ { \prime }$ and continue exploration from it.

Optimal Path Selection. Tree expansion follows branchlocal recursion and terminates when a valid final answer is generated or the maximum depth is reached. A final answer is valid only if it matches the required task format, namely a candidate option for multiple-choice tasks or a parseable normalized box for visual grounding. Paths that reach the depth limit without a valid answer are discarded. Upon completion, we select the optimal valid path by evaluating global coherence across all nodes rather than relying solely on the final-node score. To mitigate path-length bias commonly associated with average scoring across long chains, $\mathrm { { A g e n t } _ { S C } }$ is explicitly prompted to assign strict penalties to uninformative or meandering intermediate steps. The optimal path ${ \mathcal { P } } ^ { * }$ is thus selected by maximizing the average node score:

$$
\mathcal { P } ^ { * } = \arg \operatorname* { m a x } _ { \mathcal { P } _ { i } } \left( \frac { 1 } { \lvert \mathcal { P } _ { i } \rvert } \sum _ { N \in \mathcal { P } _ { i } } S ( N ) \right) .\tag{6}
$$

Subsequently, the valid answer stored at the leaf of ${ \mathcal { P } } ^ { * }$ is selected as the global final answer $a _ { \mathrm { f i n a l } } { \mathrm { : } }$

$$
a _ { \mathrm { f i n a l } }  a _ { \mathrm { l e a f } } ( \mathcal { P } ^ { * } ) .\tag{7}
$$

If every branch is pruned or reaches the depth limit without a valid answer, DAAS returns the base model’s initial response a<sub>0</sub> as a conservative fallback. In summary, DAAS focuses computation on aerial queries that require multi-step spatial or attribute reasoning by deriving a per-query pruning threshold and selecting the valid path with the highest global coherence. The complete procedure is summarized in Algorithm 2. Implementation details and agent configurations are provided in the supplementary material.

## V. EXPERIMENT

## A. Main Results

The quantitative results on UAVQA-Bench are presented in Table III. Built upon Qwen3-VL 8B and 32B models, our proposed UAV-MAS demonstrates substantial performance gains over both vanilla baselines and existing agent frameworks. Specifically, UAV-MAS-8B and UAV-MAS-32B surpass their respective Instruct counterparts by margins of 8.7% and 7.4% in Overall Accuracy. This substantial improvement underscores the efficacy of our design in bridging the domain gap that limits standard MLLMs. Furthermore, compared to general-purpose multi-agent frameworks utilizing identical models, UAV-MAS achieves a 5.9% lead over the strongest baseline, validating the effectiveness of our designs. Remarkably, UAV-MAS-32B outperforms the closed-source Gemini 3 Pro by 4.0%, showing that a domain-specialized open-source multi-agent design can compete effectively with proprietary general-purpose models.

Algorithm 2 Difficulty-Aware Adaptive Search (DAAS)   
1: Input: Image I, Query Q, Max Depth D, Max Width W   
2: Output: Final Answer a<sub>final</sub>   
3: $\begin{array} { r } { a _ { 0 } \gets \mathrm { V L M } ( Q , I ) , S _ { \mathrm { i n i t } } \gets \mathrm { A g e n t } _ { \mathrm { S C } } ( Q , I , a _ { 0 } ) } \end{array}$   
4: $\tau  g ( S _ { \mathrm { i n i t } } )$ using the fixed three-level map   
5: $N _ { \mathrm { r o o t } }  \{ \mathcal { H } _ { 0 } , a _ { 0 } , C _ { 0 } , S _ { \mathrm { i n i t } } , d = 0 \} , \nu  \emptyset$   
6: Function Search $( N \{ \mathcal { H } , a , C , S , d \} )$   
7: for $k = 1$ to W do   
8: Sample one CAIR step to obtain $\mathcal { H } ^ { \prime } , a ^ { \prime } , C ^ { \prime }$   
9: $S ^ { \prime } \gets \mathrm { A g e n t } _ { \mathrm { S C } } ( \mathcal { H } ^ { \prime } - \mathcal { H } )$   
10: if $S ^ { \prime } < \tau \land S < \tau$ then   
11: continue // prune two consecutive weak steps   
12: end if   
13: $N ^ { \prime } \gets \{ \mathcal { H } ^ { \prime } , a ^ { \prime } , C ^ { \prime } , S ^ { \prime } , d + 1 \}$   
14: if Final(a<sup>′</sup>) ∧ Valid(a<sup>′</sup>) then   
15: $\mathcal { V }  \mathcal { V } \cup \{ \mathcal { P } ( N ^ { \prime } ) \}$   
16: else if $d + 1 = D$ then   
17: if Valid(a<sup>′</sup>) then   
18: $\mathcal { V }  \mathcal { V } \cup \{ \mathcal { P } ( N ^ { \prime } ) \}$   
19: end if   
20: else   
21: Search $( N ^ { \prime } )$   
22: end if   
23: end for   
24: End Function   
25: Search $( N _ { \mathrm { r o o t } } )$   
26: if $\nu = \emptyset$ then   
27: a<sub>final</sub> $ a _ { 0 }$ //fallback for pruned/invalid searches   
28: else   
29: Select ${ \mathcal { P } } ^ { * }$ from V using Eq. 6   
30: $a _ { \mathrm { f i n a l } }  a _ { \mathrm { l e a f } } ( \mathcal { P } ^ { * } )$ via Eq. 7   
31: end if   
32: Return $a _ { \mathrm { f i n a l } }$  
TABLE IV

MODULE-LEVEL ABLATION OF UAV-MAS ON UAVQA-BENCH.BASELINE: QWEN3-VL 8B INSTRUCT. OA: OVERALL ACCURACY (%).
<table><tr><td>Method</td><td>DSPE</td><td>CAIR</td><td>DAAS</td><td>OA</td></tr><tr><td>Baseline</td><td></td><td></td><td></td><td>61.73</td></tr><tr><td>Exp1</td><td>V</td><td></td><td></td><td>64.53</td></tr><tr><td>Exp2</td><td>√</td><td>1</td><td></td><td>68.03</td></tr><tr><td>Exp3</td><td>√</td><td></td><td></td><td>68.27</td></tr><tr><td>Full</td><td>√</td><td></td><td></td><td>70.47</td></tr></table>

## B. Ablation Study

1) Module-Level Ablation: Table IV presents an incremental analysis of each module. Starting from the vanilla Qwen3- VL 8B baseline (61.73%), equipping it with DSPE alone (Exp1) raises accuracy to 64.53%, confirming the necessity of domain-specific perception operators. When CAIR is further added (Exp2), performance jumps substantially to 68.03%, reflecting its effectiveness in suppressing error propagation across reasoning steps. Adding DAAS instead of CAIR (Exp3) also yields a strong gain to 68.27%, demonstrating that adaptive search independently contributes to accurate reasoning.

![](images/b687ccbc49737a8dad211ee853730c901b413d5d5d9a07e0554e1ba6513466a7.jpg)  
Fig. 5. Visual comparison of UAV-MAS with other methods. All experiments are based on Qwen3-VL 8B except Gemini 3 Pro.

TABLE V  
WITHIN-MODULE DESIGN ABLATION OF UAV-MAS. BASELINE: QWEN3-VL 8B INSTRUCT. OA: OVERALL ACCURACY (%).
<table><tr><td>Module</td><td>Variant</td><td>OA (%)</td></tr><tr><td>Baseline</td><td>-</td><td>61.73</td></tr><tr><td rowspan="2">DSPE</td><td>w/o Per-Tool Activation</td><td>69.60</td></tr><tr><td>w/o Distributed  $\mathrm { { A g e n t } _ { T s } }$ </td><td>68.07</td></tr><tr><td rowspan="3">CAIR</td><td>w/o CAIR</td><td>68.27</td></tr><tr><td>w/o  $\mathbf { A g e n t } _ { \mathrm { C I } }$ </td><td>70.27</td></tr><tr><td>w/o  $\mathrm { \bf A g e n t { \mathrm { _ { P V } } } }$ </td><td>69.53</td></tr><tr><td rowspan="2">DAAS</td><td>w/o DAAS w/o Difficulty Awareness</td><td>68.03</td></tr><tr><td>w/o Pruning</td><td>69.87 68.73</td></tr><tr><td>Full</td><td>-</td><td>70.47</td></tr></table>

Integrating all three modules achieves the highest score of 70.47%, indicating that DSPE, CAIR, and DAAS address complementary bottlenecks and together produce a synergistic effect.

2) Within-Module Design Ablation: Table V examines the internal design choices of each module against the full system (70.47%). For DSPE, disabling Per-Tool Activation—which feeds all available tools directly into the agent—causes accuracy to drop to 69.60%, highlighting the critical importance of actively selecting task-relevant tools. Furthermore, collapsing the parallel, distributed $\mathrm { { A g e n t } _ { T S } }$ into a single centralized agent degrades performance to $6 8 . 0 7 \%$ , demonstrating that in domain-specific scenarios, smaller, less capable models require highly fine-grained tool management to operate effectively. For CAIR, completely removing the module yields 68.27%. Ablating $\mathbf { A g e n t } _ { \mathrm { C I } }$ alone still results in a 0.20% drop, showing that contextual integration is needed to reconcile verification signals; ablating $\mathrm { \Delta A g e n t { _ { P V } } }$ causes a more pronounced decline to 69.53%, highlighting that independent perceptual verification is the more critical of the two agents. For DAAS, removing the tree-search mechanism entirely (w/o DAAS) yields 68.03%. Disabling difficulty awareness (69.87%) or the pruning mechanism (68.73%) also incurs clear penalties, demonstrating that both adaptive resource allocation and branch filtering are essential for high-quality search.

3) DAAS Efficiency Ablation: To assess the computational role of DAAS, Table VI reports average per-query results on one NVIDIA H200 GPU. Compared with UAV-MAS without

DAAS, the full method introduces additional search computation while improving OA from 68.03% to 70.47%. More importantly, at a similar accuracy level, Majority Vote@3 applied to UAV-MAS without DAAS reaches 70.73% OA but requires 265.29 s, 36.39 MLLM calls, and 5.07 tool calls per query. Full UAV-MAS obtains 70.47% OA using 112.23 s, 25.60 MLLM calls, and 2.99 tool calls, reducing latency by 57.7% while requiring fewer model and tool calls. These results show that DAAS provides a better accuracy–cost trade-off than brute-force repeated sampling through targeted exploration.

TABLE VI  
EFFICIENCY ABLATION OF DAAS ON ONE NVIDIA H200 GPU. MV3DENOTES MAJORITY VOTE@3.
<table><tr><td>Method</td><td></td><td>OA (%) Time (s)</td><td>MLLM Calls</td><td> Tool Calls</td></tr><tr><td>UAV-MAS w/o DAAS</td><td>68.03</td><td>88.43</td><td>12.13</td><td>1.69</td></tr><tr><td>UAV-MAS w/o DAAS + MV3</td><td>70.73</td><td>265.29</td><td>36.39</td><td>5.07</td></tr><tr><td>UAV-MAS</td><td>70.47</td><td>112.23</td><td>25.60</td><td>2.99</td></tr></table>

## C. Visual Comparison

Figure 5 presents two representative cases where baselines fail and our UAV-MAS succeeds, collectively demonstrating all three modules. In both cases, DSPE’s per-tool activation first selects a task-relevant tool subset rather than activating all available tools indiscriminately. In Figure 5(a), the extreme top-down viewpoint creates pose ambiguity: the shown opensource baselines misidentify cyclists as persons standing on the lawn. The DESCRIBE tool returns a high-confidence observation that conflicts with the prior belief; CAIR’s steplevel verification agent detects this conflict and updates the reasoning state accordingly, yielding the correct answer. This demonstrates that step-level verification is essential for correcting perceptual bias before it propagates through the reasoning chain. In Figure 5(b), the challenge is semantic distraction, manifesting in two distinct failure modes: the shown opensource baselines over-count to 8 by including parking-lot vehicles, while Gemini 3 Pro hallucinates a non-existent onroad vehicle and returns 4. DAAS assigns a low confidence score to the distractor-contaminated detection; two consecutive low scores then trigger branch pruning, and the search continues along alternative paths that correctly isolate the 3 onroad vehicles. This demonstrates that consecutive-confidence pruning effectively rejects misleading evidence and selects the globally more reliable reasoning trace.

TABLE VII  
CROSS-DATASET GENERALIZATION ON THE CHOICE SUBSET. ALL VALUES ARE PERCENTAGES (%).
<table><tr><td>Method</td><td>OA</td><td>ILC</td><td>SII</td><td>CID</td><td>AttR</td><td>AssR</td><td>CSR</td></tr><tr><td>Qwen3-VL-8B Inst.</td><td>71.82</td><td>75.00</td><td>72.14</td><td>75.00</td><td>65.00</td><td>42.50</td><td>86.67</td></tr><tr><td>Qwen3-VL-8B Think.</td><td>69.09</td><td>70.00</td><td>68.57</td><td>83.33</td><td>47.50</td><td>47.5078.33</td><td></td></tr><tr><td>UAV-MAS-8B</td><td></td><td>75.23 85.00</td><td></td><td></td><td></td><td>77.14 76.67 65.00 50.0091.67</td><td></td></tr></table>

## D. Cross-Dataset Generalization on CHOICE

To evaluate cross-dataset generalization, we test UAV-MAS-8B on the independent CHOICE remote-sensing benchmark [57], using only generic image zooming and description tools, and compare it with the Qwen3-VL-8B Instruct and Thinking variants. We retain 440 questions whose answer formats are compatible with our evaluation and exclude Referring Expression Segmentation (RES) from Cross-instance Discernment because it requires pixel-level masks. Following the CHOICE taxonomy, ILC, SII, CID, AttR, AssR, and CSR denote Image-level Comprehension, Single-instance Identification, Cross-instance Discernment, Attribute Reasoning, Assessment Reasoning, and Common Sense Reasoning, respectively. As shown in Table VII, UAV-MAS-8B achieves the highest Overall Accuracy of 75.23%, outperforming the two baselines by 3.41 and 6.14 percentage points, respectively. These results demonstrate the effectiveness and generalizability of our method beyond the UAVQA-Bench distribution and UAV-specific perception tools.

## VI. LIMITATIONS AND FUTURE WORK

Despite its encouraging performance, UAV-MAS has two main limitations. First, its iterative multi-agent reasoning and tool invocation introduce non-negligible inference latency. Although employing lightweight models and the DAAS strategy reduces unnecessary computation, the system remains substantially slower than a single forward pass. Consequently, it is currently more suitable for offline image analysis at a ground station than for real-time on-board inference. Second, its performance still has room for improvement. CAIR mitigates the propagation of local errors during iterative reasoning but cannot eliminate them entirely. In some cases, the model changes a correct answer to an incorrect one, fails to recognize critical visual information, or cannot objectively estimate the benefit of each reasoning step.

Future work will address these limitations from both efficiency and performance perspectives. For efficient deployment, we will investigate parallel agent execution, reusable visual-feature caching, more aggressive early-exit and dynamic-routing mechanisms, and model compression to reduce redundant computation and enable real-time onboard processing. To improve performance, we will explore stronger fine-grained visual perception, uncertaintyaware answer preservation and rollback mechanisms, and better-calibrated process-level verification for estimating stepwise reasoning gains. These directions are expected to make

UAV-MAS faster, more reliable, and more practical for realworld UAV applications.

## VII. CONCLUSION

In this paper, we introduce UAVQA-Bench, a comprehensive, fully human-annotated benchmark for UAV aerial image understanding and reasoning. It covers 6 capability dimensions across 16 tasks (in multiple-choice and visual grounding formats), with 1,500 samples drawn from 13 diverse public UAV datasets. We systematically evaluate a broad range of open-source and closed-source MLLMs as well as agent-based systems on UAVQA-Bench. Based on the results, we present UAV-MAS, a novel training-free multiagent system designed to address the unique challenges of visual perception and reasoning in UAV scenarios. By integrating a Domain-Specific Perception Engine with advanced reasoning strategies—Context-Aware Iterative Refinement and Difficulty-Aware Adaptive Search—our system effectively overcomes the limitations of existing specialized models and general-purpose agents. Experiments demonstrate that UAV-MAS achieves state-of-the-art performance, enabling a 32Bparameter open-source model to surpass Gemini 3 Pro on UAVQA-Bench. These results underscore the potential of specialized multi-agent architectures in domain-specific applications. Future work will focus on optimizing real-time onboard inference and expanding the system’s capabilities to dynamic video understanding tasks.

## REFERENCES

[1] H. Zhang, K. Liu, Z. Gan, and G.-N. Zhu, “Uav-detr: efficient end-to-end object detection for unmanned aerial vehicle imagery,” arXiv preprint arXiv:2501.01855, 2025.

[2] L. Madhuanand, F. Nex, and M. Y. Yang, “Self-supervised monocular depth estimation from oblique uav videos,” ISPRS journal of photogrammetry and remote sensing, vol. 176, pp. 1–14, 2021.

[3] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu et al., “Mmbench: Is your multi-modal model an all-around player?” in European conference on computer vision. Springer, 2024, pp. 216–233.

[4] S. Lobry, D. Marcos, J. Murray, and D. Tuia, “RSVQA: Visual question answering for remote sensing data,” IEEE Transactions on Geoscience and Remote Sensing, vol. 58, no. 12, pp. 8555–8566, 2020.

[5] X. Li, J. Ding, and M. Elhoseiny, “Vrsbench: A versatile vision-language benchmark dataset for remote sensing image understanding,” Advances in Neural Information Processing Systems, vol. 37, pp. 3229–3242, 2024.

[6] F. Wang, H. Wang, Z. Guo, D. Wang, Y. Wang, M. Chen, Q. Ma, L. Lan, W. Yang, J. Zhang et al., “Xlrs-bench: Could your multimodal llms understand extremely large ultra-high-resolution remote sensing imagery?” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 14 325–14 336.

[7] D. Du, P. Zhu, L. Wen, X. Bian, H. Lin, Q. Hu, T. Peng, J. Zheng, X. Wang, Y. Zhang et al., “Visdrone-det2019: The vision meets drone object detection in image challenge results,” in Proceedings of the IEEE/CVF international conference on computer vision workshops, 2019, pp. 0–0.

[8] D. Du, Y. Qi, H. Yu, Y. Yang, K. Duan, G. Li, W. Zhang, Q. Huang, and Q. Tian, “The unmanned aerial vehicle benchmark: Object detection and tracking,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 370–386.

[9] B. Zhao, J. Fang, Z. Dai, Z. Wang, J. Zha, W. Zhang, C. Gao, Y. Wang, J. Cui, X. Chen et al., “Urbanvideo-bench: Benchmarking vision-language models on embodied intelligence with video data in urban spaces,” arXiv preprint arXiv:2503.06157, 2025.

[10] S. Chakrabarty, “Yolo26: An analysis of nms-free end to end framework for real-time object detection,” arXiv preprint arXiv:2601.12882, 2026.

[11] B. Gao, J. Tong, X. Chen, H. Yu, and Z. Li, “Dfir-detr: Frequency domain enhancement and dynamic feature aggregation for cross-scene small object detection,” arXiv preprint arXiv:2512.07078, 2025.

[12] A. Khanpour, T. Wang, A. Vahidi-Shams, W. Ectors, F. Nakhaie, A. Taheri, and C. Claudel, “Uav-based intelligent traffic surveillance system: Real-time vehicle detection, classification, tracking, and behavioral analysis,” arXiv preprint arXiv:2509.04624, 2025.

[13] Y. Chen, Z. Ye, H. Sun, T. Gong, S. Xiong, and X. Lu, “Global–local fusion with semantic information-guidance for accurate small object detection in UAV aerial images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–15, 2025, art. no. 4701115.

[14] Y. Xi, W. Jia, Q. Miao, J. Feng, J. Ren, and H. Luo, “Detection-driven exposure-correction network for nighttime drone-view object detection,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1– 14, 2024, art. no. 5605014.

[15] M. Maimaitijiang, H. Ghimire, S. Thapa, M. M. Billah, S. Sehgal, M. Singh, S. Kaushal, K. Poudel, S. Subedi, U. U. R. Janjua et al., “Wheatai v1. 0: An ai-powered high throughput wheat phenotyping platform,” arXiv preprint arXiv:2601.08863, 2026.

[16] D. E. Kharismawati and T. Kazic, “Maizestandcounting (masc): Automated and accurate maize stand counting from uav imagery using image processing and deep learning,” arXiv preprint arXiv:2510.07580, 2025.

[17] O. Youme, J. M. Dembele, E. C. Ezin, and C. Cambier, “Panoptic segmentation of environmental uav images: Litter beach,” arXiv preprint arXiv:2508.15985, 2025.

[18] P. Chen, T. Ouyang, K. Luo, W. Hong, and X. Chen, “Codrone: Autonomous drone navigation assisted by edge and cloud foundation models,” IEEE Internet of Things Journal, 2025.

[19] Y. Lin, B. Xue, M. Zhang, S. Schofield, and R. Green, “Generalization evaluation of deep stereo matching methods for uav-based forestry applications,” arXiv preprint arXiv:2512.03427, 2025.

[20] X. Shi, T. Tang, J. Chen, S. Lv, and Y. Liu, “Precise depth estimation by calculating affine transformation parameters,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–15, 2024.

[21] S. Wang, K. Liu, J. Huang, and X. Li, “FLDet: Faster and lighter aerial object detector,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 5, pp. 4450–4463, 2025.

[22] F. Sun, D. Cheng, P. Zheng, T. Song, L. Chen, and Q. Kou, “TGCADNet: Text-guided context-aware detection via CLIP for small objects in UAV scenes,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 36, no. 6, pp. 7846–7859, 2026.

[23] Q. Jiang, J. Huo, X. Chen, Y. Xiong, Z. Zeng, Y. Chen, T. Ren, J. Yu, and L. Zhang, “Detect anything via next point prediction,” arXiv preprint arXiv:2510.12798, 2025.

[24] Y. Cao, Z. He, L. Wang, W. Wang, Y. Yuan, D. Zhang, J. Zhang, P. Zhu, L. Van Gool, J. Han et al., “Visdrone-det2021: The vision meets drone object detection challenge results,” in Proceedings of the IEEE/CVF International conference on computer vision, 2021, pp. 2847–2854.

[25] G.-S. Xia, X. Bai, J. Ding, Z. Zhu, S. Belongie, J. Luo, M. Datcu, M. Pelillo, and L. Zhang, “Dota: A large-scale dataset for object detection in aerial images,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 3974–3983.

[26] S. Wang, S. Li, Y. Zhang, S. Yu, S. Yuan, R. She, Q. Guo, J. Zheng, O. K. Howe, L. Chandra et al., “Uavscenes: A multi-modal dataset for uavs,” arXiv preprint arXiv:2507.22412, 2025.

[27] G. Li, J. Xu, Y. Zhao, and Y. Peng, “Dyfo: A training-free dynamic focus visual search for enhancing lmms in fine-grained visual understanding,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 9098–9108.

[28] S. Zhao, H. Zhang, S. Lin, M. Li, Q. Wu, K. Zhang, and C. Wei, “Pyvision: Agentic vision with dynamic tooling,” arXiv preprint arXiv:2507.07998, 2025.

[29] C. Scofield, “Multi-agent constraint factorization reveals latent invariant solution structure,” arXiv preprint arXiv:2601.15077, 2026.

[30] Z. Ke, Y. Ming, A. Xu, R. Chin, X.-P. Nguyen, P. Jwalapuram, S. Yavuz, C. Xiong, and S. Joty, “Mas-orchestra: Understanding and improving multi-agent reasoning through holistic orchestration and controlled benchmarks,” arXiv preprint arXiv:2601.14652, 2026.

[31] R. R. Rodriguez Jr, “Agent identity uri scheme: Topology-independent naming and capability-based discovery for multi-agent systems,” arXiv preprint arXiv:2601.14567, 2026.

[32] W. Wang, Z. Gao, L. Gu, H. Pu, L. Cui, X. Wei, Z. Liu, L. Jing, S. Ye, J. Shao et al., “Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency,” arXiv preprint arXiv:2508.18265, 2025.

[33] Q. Ma, C. Guo, Z. Tian, S. Wang, J. Xiao, Y. Yue, and Z. Zhang, “Paper2rebuttal: A multi-agent framework for transparent author response assistance,” arXiv preprint arXiv:2601.14171, 2026.

[34] A. Adimulam, R. Gupta, and S. Kumar, “The orchestration of multiagent systems: Architectures, protocols, and enterprise adoption,” arXiv preprint arXiv:2601.13671, 2026.

[35] H. Lee, “Motion-to-response content generation via multi-agent ai system with real-time safety verification,” arXiv preprint arXiv:2601.13589, 2026.

[36] S. Hui, D. Yanfeng, H. Ma, C. Xu, K. Jin, L. Zu, C. Zhong, G. Wang, W. Cai et al., “Agentgc: Evolutionary learning-based lossless compression for genomics data with llm-driven multiple agent,” arXiv preprint arXiv:2601.13559, 2026.

[37] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao, “React: Synergizing reasoning and acting in language models,” in The eleventh international conference on learning representations, 2022.

[38] I. Bozcan and E. Kayacan, “Au-air: A multi-modal unmanned aerial vehicle dataset for low altitude traffic surveillance,” in 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020, pp. 8504–8510.

[39] C. Zhang, G. Huang, L. Liu, S. Huang, Y. Yang, X. Wan, S. Ge, and D. Tao, “Webuav-3m: A benchmark for unveiling the power of millionscale deep uav tracking,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 7, pp. 9186–9205, 2022.

[40] D. Du, P. Zhu, L. Wen, X. Bian, H. Lin, Q. Hu, T. Peng, J. Zheng, X. Wang, Y. Zhang et al., “Visdrone-det2019: The vision meets drone object detection in image challenge results,” in Proceedings of the IEEE/CVF international conference on computer vision workshops, 2019.

[41] U. Graz, “Semantic drone dataset,” 2019. [Online]. Available: http://dronedataset.icg.tugraz.at/

[42] Y. Sun, B. Cao, P. Zhu, and Q. Hu, “Drone-based rgb-infrared crossmodality vehicle detection via uncertainty-aware learning,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 10, pp. 6700–6713, 2022.

[43] W. Cai, K. Jin, J. Hou, C. Guo, L. Wu, and W. Yang, “Vdd: Varied drone dataset for semantic segmentation,” Journal of Visual Communication and Image Representation, vol. 109, p. 104429, 2025.

[44] Y. Chen, Y. Wang, P. Lu, Y. Chen, and G. Wang, “Large-scale structure from motion with semantic constraints of aerial images,” in Chinese Conference on Pattern Recognition and Computer Vision (PRCV). Springer, 2018, pp. 347–359.

[45] Y. Lyu, G. Vosselman, G.-S. Xia, A. Yilmaz, and M. Y. Yang, “Uavid: A semantic segmentation dataset for uav imagery,” ISPRS journal of photogrammetry and remote sensing, vol. 165, pp. 108–119, 2020.

[46] H. Florea, V.-C. Miclea, and S. Nedevschi, “Wilduav: Monocular uav dataset for depth estimation tasks,” in 2021 IEEE 17th International Conference on Intelligent Computer Communication and Processing (ICCP). IEEE, 2021, pp. 291–298.

[47] C. Feng, Z. Chen, X. Li, C. Wang, J. Yang, M.-M. Cheng, Y. Dai, and Q. Fu, “Hazydet: Open-source benchmark for drone-view object detection with depth-cues in hazy scenes,” arXiv preprint arXiv:2409.19833, 2024.

[48] P. Zhu, T. Peng, D. Du, H. Yu, L. Zhang, and Q. Hu, “Graph regularized flow attention network for video animal counting from drones,” IEEE Transactions on Image Processing, vol. 30, pp. 5339–5351, 2021.

[49] M. Mueller, N. Smith, and B. Ghanem, “A benchmark and simulator for uav tracking,” in European conference on computer vision, vol. 7, 2016.

[50] H. Lin, S. Chen, J. H. Liew, D. Y. Chen, Z. Li, G. Shi, J. Feng, and B. Kang, “Depth anything 3: Recovering the visual space from any views,” arXiv preprint arXiv:2511.10647, 2025.

[51] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, W. Ge, Z. Guo, Q. Huang, J. Huang, F. Huang, B. Hui, S. Jiang, Z. Li, M. Li, M. Li, K. Li, Z. Lin, J. Lin, X. Liu, J. Liu, C. Liu, Y. Liu, D. Liu, S. Liu, D. Lu, R. Luo, C. Lv, R. Men, L. Meng, X. Ren, X. Ren, S. Song, Y. Sun, J. Tang, J. Tu, J. Wan, P. Wang, P. Wang, Q. Wang, Y. Wang, T. Xie, Y. Xu, H. Xu, J. Xu, Z. Yang, M. Yang, J. Yang, A. Yang, B. Yu, F. Zhang, H. Zhang, X. Zhang, B. Zheng, H. Zhong, J. Zhou, F. Zhou, J. Zhou, Y. Zhu, and K. Zhu, “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.

[52] V. Team, W. Hong, W. Yu, X. Gu, G. Wang, G. Gan, H. Tang, J. Cheng, J. Qi, J. Ji, L. Pan, S. Duan, W. Wang, Y. Wang, Y. Cheng, Z. He, Z. Su, Z. Yang, Z. Pan, A. Zeng, B. Wang, B. Chen, B. Shi, C. Pang, C. Zhang, D. Yin, F. Yang, G. Chen, J. Xu, J. Zhu, J. Chen, J. Chen, J. Chen, J. Lin, J. Wang, J. Chen, L. Lei, L. Gong, L. Pan, M. Liu, M. Xu, M. Zhang, Q. Zheng, S. Yang, S. Zhong, S. Huang, S. Zhao, S. Xue, S. Tu, S. Meng, T. Zhang, T. Luo, T. Hao, T. Tong,

W. Li, W. Jia, X. Liu, X. Zhang, X. Lyu, X. Fan, X. Huang, Y. Wang, Y. Xue, Y. Wang, Y. Wang, Y. An, Y. Du, Y. Shi, Y. Huang, Y. Niu, Y. Wang, Y. Yue, Y. Li, Y. Zhang, Y. Wang, Y. Wang, Y. Zhang, Z. Xue, Z. Hou, Z. Du, Z. Wang, P. Zhang, D. Liu, B. Xu, J. Li, M. Huang, Y. Dong, and J. Tang, “Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning,” 2025. [Online]. Available: https://arxiv.org/abs/2507.01006

[53] Q. Team, “Qwen3.5: Accelerating productivity with native multimodal agents,” February 2026. [Online]. Available: https://qwen.ai/blog?id= qwen3.5

[54] OpenAI, “Introducing gpt-5.2,” https://openai.com/index/ introducing-gpt-5-2/, 2025, accessed: 2026-01-28.

[55] Google DeepMind, “Gemini 3 pro and gemini 3 flash models,” https: //ai.google.dev/, 2025, accessed: 2026-01-15.

![](images/5a7a2790a2727d7ed680b89724b5fa06fdb143f6aa35f2b6255411b2172b62cf.jpg)

[56] QwenLM, “Qwen-agent: An agent framework based on qwen,” https: //github.com/QwenLM/Qwen-Agent, 2024.

[57] X. An, J. Sun, Z. Gui, and W. He, “Choice: Benchmarking the remote sensing capabilities of large vision-language models,” in Advances in Neural Information Processing Systems, vol. 38, 2025. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2025/hash/befe25a01cf4dbe9635e85f835d31250-Abstract-Datasets and Benchmarks Track.html

![](images/fe06085d32adfa739d4f6c0290beecf962213189661fbd084e3ee61adda03509.jpg)  
and ICCV.  
Haoyu Zhang received the B.S. degree in electronic science and technology from the School of Information Science and Technology, Fudan University, Shanghai, China, in 2025. He is currently pursuing the Ph.D. degree with the College of Future Information Technology, Fudan University. His research interests include computer vision, multimodal large models, and intelligent agent systems.

![](images/2814938f15b9eaec017d22d305e8b95cc7828d5bbf86590add5b8cd6b70bf044.jpg)

Lin Zhang received the B.S. degree in electronic engineering from Fudan University, Shanghai, China, in 2022, where he is currently pursuing the Ph.D. degree with the College of Future Information Technology. His main research interests include computer vision and transfer learning.

Jiakang Yuan is currently pursuing the Ph.D. degree in electronic engineering with the College of Future Information Technology, Fudan University. He received the bachelor’s degree in electronic engineering from Fudan University in 2022. His research interests include multimodal reasoning, multi-agent systems, and spatial intelligence. He has published papers in leading journals and conferences, including CVPR, ICCV, ECCV, NeurIPS, and IEEE TPAMI, and serves as a reviewer for journals and conferences including IEEE TIP, IEEE TCSVT, CVPR, ECCV,

![](images/386382e25da70da8914e89ff7c062086d87d85d6568595b9bfbb70e90ccc9fc4.jpg)

Shenghong Yi received the B.S. degree in Intelligent Science and Technology from Fudan University, Shanghai, China, in 2026. He is currently working toward the Ph.D. degree at the College of Future Information Technology. His research interests include computer vision and multimodal large models.

![](images/1408853b859202cf13189cb14b81c83068b99e839fb0eeeb87e3690206bac35c.jpg)

Shuoxun Zhang received the B.S. degree in Communication Engineering from Shanghai University, Shanghai, China, in 2023. He is currently pursuing a master’s degree at the College of Future Information Technology, Fudan University. His main research interests include multimodal large models and intelligent agent systems.

![](images/449dd47339b1a407b8b87fcf70cfacfd2547a61f826b9c49073714c9129b5481.jpg)

Yuening Wang is currently pursuing the bachelor’s degree in Electronic Information Science and Technology at the College of Future Information Innovation, Fudan University, Shanghai, China, and is expected to graduate in June 2027.

![](images/9a7b390521ad41b64bc299fd486f8f975dc610e74a8a9e5d4e9154be71ab7def.jpg)

Peng Ye is currently a postdoctoral fellow at MM-Lab of The Chinese University of Hong Kong and a scientific research advisor of Shanghai AI Laboratory. He received the Ph.D. degree from Fudan University, Shanghai, China. His research interests include autonomous agents, (M)LLMs, foundation models, and efficient design and optimization. He has published papers in leading journals and conferences, including IEEE TPAMI, IJCV, CVPR, ICCV, ECCV, NeurIPS, ICML, ACM MM, and ICME. He also serves as a reviewer or program committee member for journals and conferences including IEEE TPAMI, IJCV, CVPR, ECCV, ICCV, and NeurIPS.

![](images/8021ec76e97afef9769374f20f85bf47f76d42530fb5020498c28ebdcbc32077.jpg)

Tao Chen (Senior Member, IEEE) received the Ph.D. degree in information engineering from Nanyang Technological University, Singapore, in 2013. He was a Research Scientist with the Institute for Infocomm Research, A\*STAR, Singapore, from 2013 to 2017, and a Senior Scientist with the Huawei Singapore Research Center from 2017 to 2018. He is currently a Professor with the College of Future Information Technology, Fudan University, Shanghai, China. His main research interests include efficient computer vision, multimodal visual analysis, large

VLM compression, and their applications in embodied robot intelligence, scene understanding, and reconstruction. He has published over 200 papers in international journals and conferences such as IEEE TPAMI, IEEE TIP, IJCV, CVPR, and NeurIPS. He has served in area chair and senior program committee roles for conferences such as AAAI, ICLR, and PRCV. He received the IJCAI 2025 Distinguished Paper Award.

# Supplementary Material for: Advancing MLLM-based UAV Image Understanding and Reasoning: A Benchmark and a Training-Free Multi-Agent System

## APPENDIX A MORE DETAILS ABOUT UAVQA-BENCH

## A. Question Templates

To ensure the linguistic diversity of UAVQA-Bench, we develop a variety of question templates for each task. This section presents representative templates that define the interaction logic for the agent.

1) Existence Detection (ED):

## • Scene Presence

◦ Is any {object} present here?

◦ Is any {object} visible in the image?

◦ Is there any {object} in the scene?

◦ Does this image contain any {object}?

◦ Can you see any {object} in this picture?

## • Conditional Presence

◦ Is any {object} present {condition}?

◦ Is a/an {object} {condition} in the image?

◦ Is there any {object} {condition} in this image?

◦ Does the image contain a/an {object} {condition}?

◦ Can you see any {object} {condition} in this picture?

## • Multi-Target Presence

◦ Which of the listed objects can be seen in the image?

◦ Does the image show any of the objects listed below?

◦ Which of the following objects are present in the image?

◦ From the options below, which objects appear in the picture?

◦ Among the following choices, which ones are present in the photo?

2) Category Recognition (CR):

## • Regional Classification

◦ Identify the object found in $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \}$

◦ Classify the object in region $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \}$

◦ Which class does the object in $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \}$ belong to?

◦ What is the category of the object located at $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} ?$

◦ Label the region $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \}$ with the correct object category.

3) Quantity Awareness (QA):

## • Scene Counting

◦ Provide the quantity of {object}.

◦ Count all the {object} in this picture.

◦ How many {object} are in this image?

◦ How many instances of {object} can you see?

◦ Can you count the number of {object} present?

## • Regional Counting

◦ Provide the quantity of {object} {condition}.

◦ Count all the {object} {condition} in this picture.

◦ What is the total number of {object} visible {condition}?

◦ How many {object} appear inside this bounding box $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} ?$

◦ What is the total number of {object} in the bounding box $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} ?$

◦ Can you count the number of {object} within the bounding box $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} ?$

## • Number Comparison

◦ Do $\{ { \mathrm { o b j e c t } } \} _ { A }$ outnumber $\{ \mathrm { o b j e c t } \} _ { B } ?$

◦ Do $\{ { \mathrm { o b j e c t } } \} _ { A }$ exceed $\{ \mathrm { o b j e c t } \} _ { B }$ in quantity?

◦ Are $\{ \mathrm { o b j e c t } \} _ { A }$ less abundant than $\{ \mathrm { o b j e c t } \} _ { B } ?$

◦ $\mathrm { A r e \ \{ o b j e c t \} } _ { A }$ more numerous than $\{ \mathrm { o b j e c t } \} _ { B } ?$

◦ Is it true that $\{ \mathrm { o b j e c t } \} _ { A }$ is fewer than $\{ \mathrm { o b j e c t } \} _ { B } ?$

4) Fine-grained Attribute Perception (FAP):

## • Regional Attribute Recognition

◦ What color is the {object} in $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} ?$

◦ Identify the color of the {object} in $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \}$

◦ Which color does the {object} in $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \}$ have?

◦ Identify the driving direction (relative to the camera) of the {object} in $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \}$

◦ What is the driving direction (relative to the camera) of the {object} located at $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} ?$

## • Function Recognition

◦ What is the key function of the {object} bounded by $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} ?$

◦ Which functional category does the {object} in $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \}$ belong to?

◦ The {object} within the bounding box $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \}$ falls under which functional category?

## • Safety Landing

◦ Identify the safest landing area for a drone among the given options.

◦ Among the options below, where is the best place for a drone to land?

◦ Select the optimal landing spot for a drone from the following choices.

◦ Which location is most ideal for a drone landing in the options provided?

◦ From the following, choose the most appropriate site for a drone to land.

5) Spatial Relationship Understanding (SRU):

## • Spatial Relation

◦ Relative to the object in $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { A }$ , where is the object in $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { B } ?$

◦ As seen by the object in $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { A }$ , in what direction is the object in $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { B } ?$

◦ From the viewpoint of the object in $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { A } ,$ what is the direction to the object in $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { B } ?$

◦ Assuming the UAV’s forward direction corresponds to the top of the image, at which clock position is the object in $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \}$ located?

◦ Given that the top of the image is treated as the front of the UAV, what is the clock position of the object within $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \}$ relative to the UAV?

## • Height Comparison

◦ Which between $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { A }$ and $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \}$ is determined to be at a higher position? ◦ Between the two positions $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { A }$ and $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { B }$ , which one is judged to be higher in the frame?

◦ Please judge which object is higher in the frame between $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { A }$ and $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { B } \}$

◦ Can you judge which contains an object at a higher position between $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { A }$ $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { B } ?$

◦ Please evaluate the two positions $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { A }$ and $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { B } .$ , and indicate which one is higher in the vertical direction.

## • Distance Comparison

◦ Which between $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { A }$ and $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} B$ is measured to be closer to the shooting point?

◦ Between $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { A }$ and $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} .$ <sub>B</sub>, which object has a smaller depth value, meaning it is closer?

◦ Please evaluate the two positions $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { } ~$ <sub>A</sub> and $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { B }$ , and indicate which one is closer to the drone.

◦ Please judge which object is closer to the drone between $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { A }$ and $\{ < x _ { 1 } > < y _ { 1 } > < x _ { 2 } > < y _ { 2 } > \} _ { B } \}$

6) Visual Grounding (VG):

## • Simple Object Grounding

◦ Locate the {object}.

◦ Point out the {object}.

◦ Could you point out the {object}?

◦ Draw the position of a/an {object}.

◦ Can you show me where the {object} is?

## • Complex Semantic Grounding

◦ Please give me the location of {semantics}.

◦ Please indicate the position of {semantics}.

◦ Can you point out the location of {semantics}?

◦ Could you tell me the location for {semantics}?

◦ In the current image, where can I find {semantics}?

## • Highest Object Grounding

◦ Find the area with the tallest building.

◦ Which area in the image has the tallest building?

◦ Judge which building area in the image has the highest floors.

## Existence Detection

## Scene Presence

Regional Counting  
![](images/c474e6646b56070db1ebf1b68361884c66b3576a20caf539d1e724aa20121e46.jpg)

Q: Is there any truck in the scene?

O: ["Yes", "No"]

## Conditional Presence

![](images/a589f54f20af66ce40d4705eca26bd09f4ddaaefe687ddd964d6d44dc73cf0a0.jpg)

A: "Yes"

Q: Does this image contain any buses moving left?

O: ["Yes", "No"]

A: "No"

## Multi-Target Presence

Q: Among the following choices, which ones are present in the photo?

O: ["Human", "Car", "Truck", "Bus", "Bike", "Cab"]

A: ["Human", "Car", "Bike"]

![](images/5436eee3553caf444fb6d090ba85555b064be6195198e4f5f59ea975dcea7519.jpg)

## Category Recognition

## Regional Classification

Q: Label the region {<0.818><0.766><0.828><0.790>} with the correct object category.

O: ["Number 8", "Number 5", "Number 3", "Number 2"]

A: "Number 3"

![](images/046117bddf8c3157a737fa505a1fc90fe0958e12e9cf43e47ddc19d2b5e7a113.jpg)

## Scene Counting

## Quantity Awareness

![](images/10e314fd78d98ff0d64d3630ae54768d5c1fb6ae724389388742370b10ae0a6e.jpg)

Q: How many cars can you see?

![](images/62ca6900f3224762a467c8a843ae8a939a8bb78f5b67bb45ab91729754ed7c51.jpg)

O: ["1", "2", "3", "4"]

## Number Comparison

Q: How many vehicles are in the bounding box {<0.411><0.187> <0.706><0.557>}?

A: "3"

![](images/447c983d8bad7315dd5516a58e58ab734858f8db1466e0663c0f767cdef40f53.jpg)

O: ["3", "5", "7", "9"]

A: "3"

Q: Are there more red cable cars than blue cable cars?

O: ["Yes", "No"]

A: "Yes"

Fig. 1. Sample instances for tasks in Existence Detection, Category Recognition, and Quantity Awareness.

## Fine-grained Attribute Perception

## Regional Attribute Recognition

Q: What color is the car in {<0.324><0.151><0.340><0.186>}? O: ["Black", "Silver", "Green",

"Yellow"]

A: "Green"

![](images/ee75e5494cb1c36c93b2ec219235b894c1e179ce483304aa4b2f963fa0e1a74d.jpg)

## Function Recognition

Q: Which functional category does the space $\mathrm { i n } \{ < 0 . 8 9 2 > < 0 . 0 9 0 > < 0 . 9 0 9 > < 0 . 1 6 0 > \}$ belong to?

O: ["Restaurant", "Clinic", "Bank", "Hotel"]

A: "Bank"

![](images/e5f6633ee5eed2a22aae175033d7b290e733221e5630b12274329b95f2971b1a.jpg)

## Safety Landing

Q: Among the options below, where is the best place for a drone to land?

$$
0 \colon [ " \{ < 0 . 5 6 9 > < 0 . 5 3 6 > < 0 . 5 9 2 > < 0 . 6 0 4 > \} " ,
$$

$$
" \{ < 0 . 0 6 6 > < 0 . 1 3 1 > < 0 . 0 8 8 > < 0 . 1 7 4 > \} " ,
$$

$$
" \{ < 0 . 5 1 1 > < 0 . 1 0 5 > < 0 . 5 3 1 > < 0 . 1 4 1 > \} " ,
$$

$$
\ " \{ < 0 . 4 4 7 > < 0 . 7 7 9 > < 0 . 4 6 9 > < 0 . 8 3 8 > \} " ]
$$

$$
\mathrm { A } { : } ~ " \{ <  0 . 0 6 6 { > } { < } 0 . 1 3 1 { > } { < } 0 . 0 8 8 { > } { < } 0 . 1 7 4 { > }  \} "
$$

![](images/8f38c4b365bc9057aeb0318c9926fe524f9bea909e10bee4c169d78dc373e2c0.jpg)

## Spatial Relation

Q: Relative to the object in $\{ < 0 . 4 8 4 > < 0 . 3 3 7 >$

$$
\{ < 0 . 4 0 0 > < 0 . 6 2 4 > < 0 . 4 7 8 > < 0 . 8 0 2 > \}
$$

![](images/35221c52d8e1427ab0fa590b2768baacaebb47c61261b85d203ea8ce79d0dd05.jpg)

A: "Below"

## Height Comparison

Q: Between the two positions {<0.492><0.388><0.580><0.502>} and

{<0.532><0.626><0.618><0.821>}, which one is judged to be higher in the frame?

$$
\ " \{ < 0 . 5 3 2 > < 0 . 6 2 6 > < 0 . 6 1 8 > < 0 . 8 2 1 > \} \ " ]
$$

$$
\mathrm { A } { : } ~ " \{ < 0 . 4 9 2 > < 0 . 3 8 8 > < 0 . 5 8 0 > < 0 . 5 0 2 > \} "
$$

![](images/ec5cce2c78b4401fa758b48d9a585b98d19ad249a12e9170b201def8eccbe0c9.jpg)

## Distance Comparison

Q: Between {<0.250><0.436><0.372><0.534>} and {<0.482><0.803><0.635><0.998>} which is measured to be closer to the shooting point?

$$
\ " \{ < 0 . 4 8 2 > < 0 . 8 0 3 > < 0 . 6 3 5 > < 0 . 9 9 8 > \} " ]
$$

A: "{<0.482><0.803><0.635><0.998>}"

![](images/3a69989f30222af385e36552f83d9356c24b7241566257e90c0af1436ee841db.jpg)  
Fig. 2. Sample instances for tasks in Fine-grained Attribute Perception and Spatial Relationship Understanding.

7) Template Visualizations: In this section, we present additional examples from UAVQA-Bench. Figure 1 provides sample instances and visual prompts for the tasks of Existence Detection, Category Recognition, and Quantity Awareness. Figure 2 illustrates examples for Fine-grained Attribute Perception and Spatial Relationship Understanding. Finally, Figure 3 showcases samples for the Visual Grounding tasks. In these samples, “Q”, “O”, and “A” denote the Question, Options, and Ground-truth Answer, respectively. For clarity, the sample figures include bounding boxes in different colors: red boxes indicate incorrect candidate answers, green boxes denote the correct answer, and blue boxes mark the object or region referred to in the question. These bounding boxes are shown only for visualization in the paper and are not present in the actual images provided to the model during inference.

## Visual Grounding

## Simple Object Grounding

![](images/b6efae7a08a1503084cba7ea95434760c2a83e1e10c26e5a76e575b073b75cbf.jpg)

Q: Could you tell me the location for a yellow car?

## Highest Object Grounding

![](images/6a5768a81c978ec87321da0df95c065f8326384f36f73a15cfe0ff7b0c6296ba.jpg)

O: []

$$
\mathrm { A } { : } ^ { \ " } \{ < 0 . 6 4 4 > < 0 . 0 4 6 > < 1 . 0 0 0 > < 0 . 6 7 2 > \} ^ { \ " }
$$

$$
\mathrm { A } ; \ " \{ < 0 . 4 1 2 > < 0 . 5 8 2 > < 0 . 4 5 1 > < 0 . 6 5 2 > \ " \} 
$$

## Complex Semantic Grounding

Q: Where can I find the Chinese knot decorating the street light pole on the right side of the road?

O: []

A: "{<0.639><0.224><0.658><0.305>}"

![](images/f912e0bb2b5e91d1768890fed4111bfcae9e05753d5ddae1483051dc83477791.jpg)  
Fig. 3. Sample instances for tasks in Visual Grounding.

## APPENDIX B

## MORE DETAILS OF OUR METHOD

In this section, we provide the implementation settings and agent configurations used in our experiments.

## A. Implementation Details

All MLLM calls use a temperature of 0.7 and a maximum of 8,192 new tokens. For DAAS, the maximum depth is $D = 5 ,$ and the search width is set adaptively to $W = \mathrm { m i n } ( 3 , | \mathcal { T } _ { \mathrm { o p t } } | )$ , where $| \mathcal { T } _ { \mathrm { o p t } } |$ is the number of tools selected by DSPE. Direct MLLM baselines use their default system prompts with the same image–query input structure, temperature, and token budget. All experiments are conducted on NVIDIA H200 GPUs.

## B. Agent Configurations

We detail the configuration of each agent in our framework in Table I. We utilize the Qwen3-VL series as the backbone MLLM. Specifically, we employ the Instruct variant for agents that require precise execution of specialized instructions (e.g., $\mathrm { A g e n t _ { T S } , A g e n t _ { S R } ) } .$ , and the Thinking variant for the Long-Chain Answer Agent $( \mathrm { A g e n t _ { L A } } )$ to leverage its enhanced capabilitie in complex reasoning and synthesis. It is worth noting that the Instruct and Thinking models share the identical architectura design, differing only in their parameter weights which are tuned for instruction compliance and deep reasoning, respectively.

TABLE I  
AGENT CONFIGURATIONS AND MODEL TYPES USED IN UAV-MAS.
<table><tr><td>Agent Role</td><td>Model Type</td></tr><tr><td>Tool Select  $\mathrm { { A g e n t \ ( A g e n t _ { T S } ) } }$ </td><td>Qwen3-VL Instruct</td></tr><tr><td>Strategic Reasoning Agent  $( \mathrm { A g e n t } _ { \mathrm { S R } } )$ </td><td>Qwen3-VL Instruct</td></tr><tr><td>Perceptual Verification Agent  $( \mathrm { { A g e n t } _ { P V } ) }$  Contextual Integration Agent  $( \mathrm { A g e n t } _ { \mathrm { C I } } )$ </td><td>Qwen3-VL Instruct</td></tr><tr><td>Long-Chain Answer Agent  $( \mathrm { A g e n t _ { L A } } )$ </td><td>Qwen3-VL Instruct</td></tr><tr><td>Score Agent  $( \mathrm { A g e n t } _ { \mathrm { S C } } )$ </td><td>Qwen3-VL Thinking Qwen3-VL Instruct</td></tr></table>