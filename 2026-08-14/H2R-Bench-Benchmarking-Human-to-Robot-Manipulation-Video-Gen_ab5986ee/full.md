# H2R-Bench: Benchmarking Human-to-Robot Manipulation Video Generation in World Models

Dingyi Rong<sup>1,2</sup>, Yue Shi<sup>2†</sup>, Chaofan Ma<sup>1‡</sup>, Jiezhang Cao<sup>1</sup>, Zongrui Wang<sup>1,2</sup>, Zeyu Zhang<sup>1,2</sup>, Yao Mu<sup>1,2</sup>, Guangtao Zhai<sup>1,2†</sup>, Ning Liu<sup>1†</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>Shanghai Artificial Intelligence Laboratory

## Abstract

Large-scale manipulation data is essential for robot learning, yet collecting robot demonstrations remains expensive and difficult to scale. Meanwhile, abundant egocentric human manipulation videos provide rich behavioral experiences, but transferring them across embodiments remains challenging due to diferences between human hands and robotic end-efectors. Recent advances in video world models ofer a promising pathway to synthesize robot-centric manipulation videos from human observations, while their cross-embodiment transfer capability remains largely unexplored. Therefore, we introduce H2R-Bench, a benchmark for evaluating cross-embodiment human-to-robot manipulation video generation, where models transform egocentric human demonstrations into robot manipulation videos under specified embodiments. Each benchmark instance contains a human demonstration video, target embodiment constraints, and source-grounded annotations covering task goals, action events, functional contacts, and object responses. H2R-Bench evaluates generated videos through five dimensions, including goal-state completion, action-event completion, functional contact transfer, embodiment correctness, and general video quality. We benchmark eleven stateof-the-art video generation models across six manipulation families and two robot embodiments. Our evaluation reveals that current video world models remain limited in human-torobot manipulation transfer: even leading models often fail in embodiment consistency, functional interaction, and task execution. H2R-Bench provides a systematic diagnostic framework for evaluating whether video world models can bridge the human-to-robot embodiment gap and convert human manipulation observations into robot-centric training resources. Project page: https://rongdingyi.github.io/H2R-Bench/

## Introduction

Recent progress in robot learning has shown the importance of large-scale demonstrations across diverse tasks, objects, and environments (Brohan et al. 2022; Zitkovich et al. 2023; O’Neill et al. 2024; Black et al. 2024; Zhang et al. 2026b). However, collecting robot manipulation videos remains expensive, requiring specific hardware, teleoperation interfaces, calibrated cameras, safety constraints, and repeated physical execution. In contrast, egocentric human videos are abundant and naturally capture everyday object afordances, hand-object contacts, manipulation intent, and physical state changes (Damen et al. 2022; Grauman et al. 2022; Hoque et al. 2025; Ma et al. 2026b,a). This asymmetry has motivated human-to-robot and cross-embodiment generation studies, which explore video generation as a bridge between abundant human demonstrations and scarce robot demonstrations (Xie et al. 2026; Song et al. 2025; Zhang et al. 2026a). In this view, a generated robot video serves as an intermediate representation of a target robot execution conditioned on a human demonstration. The key challenge is not merely generating a visually plausible robot video, but preserving the manipulation evidence in the source demonstration while adapting the execution to a diferent embodiment. As illustrated in Figure 1, evaluating such videos requires criteria beyond frame-level realism and temporal coherence. A valid human-to-robot video should preserve the task intent, required actions, and functional interactions implied by the source demonstration.

![](images/ac7b6a0d8319f017e5a21256aee8a5a5ebcc1e31abc81c774ba79978d9b0f283.jpg)  
Figure 1: Comparison between existing evaluation and H2R-Bench. Given a human demonstration and a target embodiment instruction, video world models generate robot manipulation videos. Existing video benchmarks such as WorldModelBench (Li et al. 2026a) assess overall video plausibility, rates both videos highly, whereas our H2R-Bench diagnoses transfer through goal, action, contact, and embodiment.

<table><tr><td>Benchmark</td><td colspan="3">Settings</td><td colspan="4">Evaluation</td></tr><tr><td></td><td>I2V RV H2R</td><td></td><td></td><td>VQ</td><td></td><td>Goal Action Cont. Emb.</td><td></td></tr><tr><td>VBench</td><td>X</td><td>×</td><td>×</td><td>√ X</td><td>×</td><td>×</td><td>×</td></tr><tr><td>WorldModelBench</td><td>5</td><td>△</td><td>X</td><td>V △ √</td><td>△</td><td>×</td><td>×</td></tr><tr><td>RBench</td><td>√</td><td>√</td><td>X</td><td>J</td><td>√</td><td>×</td><td>△</td></tr><tr><td>RoboWM-Bench</td><td>√</td><td>√</td><td>×</td><td>△ √</td><td>√</td><td>×</td><td>×</td></tr><tr><td>RoboTrustBench</td><td>√</td><td>√</td><td>X</td><td>√</td><td>√ √</td><td>×</td><td>△</td></tr><tr><td>H2R-Bench (ours)</td><td>√</td><td>L</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 1: Comparison of H2R-Bench and existing videogeneration benchmarks across evaluation capabilities. “I2V”, “RV”, and “H2R” denote image-to-video generation, robotvideo evaluation, and human-to-robot transfer, respectively. Evaluation dimensions include visual quality, goal completion, action completion, functional contact transfer, and embodiment consistency. ✓, △, and × indicate full, partial, and no support, respectively.

As summarized in Table 1, existing evaluation protocols only partially capture this ability. General video-generation benchmarks mainly measure visual fidelity, temporal consistency, text-video alignment, and low-level motion quality (Huang et al. 2024; Liu et al. 2024; Sun et al. 2025; Ji et al. 2024). Recent world-model and robotics-oriented benchmarks further examine physical plausibility, action completeness, robot structure, executability, or trustworthiness in manipulation videos (Li et al. 2026a; Bansal et al. 2025; Han et al. 2026; Deng et al. 2026; Jiang et al. 2026; Li et al. 2026c). However, these benchmarks typically condition generation on text, images, initial robot states, or robot-centric scenarios and do not evaluate cross-embodiment manipulation transfer from human demonstrations to robot embodiments. As a result, existing benchmarks cannot diagnose H2R-specific failures, such as incorrect embodiment replacement or interactions that are inconsistent with the source demonstration.

We introduce H2R-Bench, a benchmark for evaluating source-conditioned human-to-robot video transfer. Unlike existing evaluations that focus on generating plausible robot videos from language prompts or robot-centric states, H2R-Bench investigates whether models can transform an egocentric human demonstration into a robot manipulation video under a specified embodiment. Given a source human video, H2R-Bench evaluates whether the generated video preserves the demonstrated task goal, required action events, functional interactions, and object-state changes, while realizing the execution with a robot morphology and end-efector consistent with the target embodiment. H2R-Bench covers six task families defined by the physical state change that determines success: rigid-object rearrangement, mechanism actuation, insertion and assembly, deformable-object configuration, bulk-material transfer, and surface or material transformation. Each demonstration is paired with a human-verified structured task specification, initialized by a multimodal language model and reviewed against the source video, describing the intended goal, required action events, manipulated entities, relevant state transitions, and source-side contact evidence, thereby defining what must be preserved without prescribing a specific robot trajectory.

H2R-Bench scores five complementary dimensions: goal completion, action-event completion, functional contact transfer, embodiment correctness, and task-agnostic video quality. Their weighted aggregate H2RCore emphasizes contact and embodiment while retaining a smaller contribution from video quality. We evaluate 11 representative video generation models, including five proprietary models (Seedance et al. 2026; Wan et al. 2025; Team et al. 2025a; Google 2025; xAI 2026) and six open-source models (Wan et al. 2025; Ha-Cohen et al. 2026; Wu et al. 2025; Li et al. 2026b; Team et al. 2025b; Song et al. 2025). Although M5 occupies a narrow range of 0.73–0.81, H2RCore spans 30.0–84.6 and has weak rank association with M5 $( \rho = 0 . 1 4 )$ . Three human raters validate the transfer metrics, with within-scene Spearman ρ = 0.883 and aggregate Pearson $r = 0 . 9 3 0$ between human and automated transfer subscores.

Our contributions are summarized as follows:

• We introduce a source-relative evaluation setting for human-to-robot video transfer, requiring generated videos to preserve the task goal and functional interaction of a specific human demonstration while replacing the human actor with a specified robot embodiment.

• We introduce H2R-Bench, spanning six manipulation families and two target embodiments, with structured annotations of task goals, action events, object-state evolution, and functional contact.

• We develop a transfer-aware evaluation protocol that separately measures goal-state completion, action-event completion, functional contact transfer, embodiment correctness, and task-agnostic video quality, and use it to reveal a persistent mismatch between visual quality and valid robot transfer in current video generation models.

## Related Work

## Video Generation and World Models for Robotics

Video generation has advanced toward controllable models with stronger temporal coherence, visual consistency, and interaction modeling (Blattmann et al. 2023; Bar-Tal et al. 2024; Kong et al. 2024; Wan et al. 2025; Wang et al. 2026c). Recent studies further view these generators as predictive world models of object dynamics, state transitions, and interaction outcomes (Brooks et al. 2024; Bruce et al. 2024; Wang et al. 2026a,b). This perspective is relevant to robotics, where generated videos can represent future observations and provide scalable data beyond costly robot demonstrations.

## Human-to-Robot Video Transfer

Egocentric human videos provide scalable supervision because they expose task goals, object afordances, hand–object interactions, and state changes (Grauman et al. 2022; Damen et al. 2022; Hoque et al. 2025; Ma et al. 2026b). Prior work transfers this information through cross-embodiment visual representations and reward learning (Nair et al. 2022; Ma et al. 2022b, 2023b), or through afordance, trajectory, and latent-action abstractions for downstream manipulation (Bahl et al. 2023; Bharadhwaj et al. 2024; Chen et al. 2025a; Ye et al. 2025; Chen et al. 2025b; Li et al. 2025c; Zheng et al. 2026). A complementary line directly constructs robot-oriented observations or data through visual translation, editing, rendering, retargeting, and generative modeling (Smith et al. 2019; Lepert, Fang, and Bohg 2025; Li et al. 2025a,b). Mitty and H2R-Grounder generate robot videos from human interaction videos (Song et al. 2025; Ci et al. 2025), while Human2Robot learns action transfer from paired human–robot demonstrations (Xie et al. 2026). These studies motivate evaluating how faithfully video generators transfer manipulation evidence across embodiments.

![](images/71ce2831d732ac32018bd1f5f11eca102d8792479027849be06be9a7cfdd6d0a.jpg)  
Figure 2: Overview of H2R-Bench. The benchmark curates egocentric human manipulation demonstrations, conditions video generators on each model’s supported source interface and target embodiment, and evaluates the resulting robot videos with transfer-aware metrics. The lower panels summarize task coverage and model capability profiles across both target embodiments.

## Evaluation of Video World Models for Robotics

Video-generation benchmarks assess perceptual quality, temporal consistency, text–video alignment, motion, and compositional correctness (Huang et al. 2024; Liu et al. 2024; Sun et al. 2025; Ji et al. 2024). Recent world-model and robotics benchmarks additionally examine physical commonsense, object-state changes, robot appearance, action completeness, executability, and trustworthiness (Bansal et al. 2025; Li et al. 2026a; Han et al. 2026; Deng et al. 2026; Jiang et al. 2026; Li et al. 2026c; Xia et al. 2026). They nevertheless evaluate videos against text, initial visual conditions, or robot-centric tasks. Human-to-robot transfer instead requires a robot video to remain faithful to a particular human demonstration while realizing a requested embodiment. This source-relative setting jointly requires task goal, required action events, functional contact, object response, and target-embodiment realization; existing benchmarks evaluate only subsets of these properties.

## H2R-Bench

## Task Formulation

H2R-Bench evaluates video transfer from a human demonstration to a specified robot embodiment. Given an egocentric human manipulation video and a target embodiment, the model generates a robot video of the same task. The source video is treated as evidence of what happened, not simply as a visual style reference. A successful output should retain the task goal, the actions needed to reach it, and the visible interaction between the actor and the objects. We do not require pose or trajectory imitation: a parallel-jaw gripper and a dexterous hand may solve the task diferently, as long as the new strategy remains compatible with the requested morphology and the source task.

Each case is

$$
x _ { i } = ( V _ { i } ^ { h } , e _ { i } , p _ { i } ) ,\tag{1}
$$

where $V _ { i } ^ { h }$ is the human source video, $e _ { i }$ is the target embodiment, and $p _ { i }$ is the generation prompt. The prompt names the task goal and target embodiment and asks the model to preserve the source scene, objects, and interaction. It does not contain evaluation weights, evidence-frame indices, or per-metric checks. Those fields are kept in a separate annotation used only for scoring. Model m receives the visual input supported by its interface, denoted by $\mathcal { Z } _ { i , m } ^ { h } ;$ : the full source video when video conditioning is available, or an ordered subset of source frames otherwise. It generates

$$
V _ { i , m } ^ { r } = G _ { m } \mathopen { } \mathclose \bgroup \left( \mathcal { Z } _ { i , m } ^ { h } , p _ { i } \aftergroup \egroup \right) .\tag{2}
$$

Models without visual source conditioning are outside this source-conditioned setting.

## Benchmark Construction

H2R-Bench uses 120 egocentric clips from the EgoDex test split (Hoque et al. 2025) as source videos. We select clips with visible task-relevant entities, interactions, and state changes, and form 240 transfer cases by pairing each of 120 source videos with two target embodiments: a parallel-jaw gripper and a dexterous hand. The sources are evenly distributed across six manipulation families: rigid-object transport or rearrangement (F1), articulated-mechanism actuation (F2), insertion or attachment (F3), deformable-object configuration (F4), bulk-material transfer or mixing (F5), and surface or material change (F6).

Qwen3.7-Plus produces an annotation for each 5s source clip from the video. It records the initial and final states, relevant objects and tools, required action events, and source-side contact evidence. For each target embodiment, the annotation also specifies functional contact regions, manipulation modes, expected object responses, and supporting or bimanual roles when applicable. All benchmark annotations are manually verified against their source videos.

## Native-Interface Generation Protocol

All models use the same source cases, target embodiments, task specifications, and evaluation criteria. We evaluate each model with its strongest publicly documented sourceconditioning interface. Video-conditioned models receive the full source clip. Image/frame-conditioned models receive ordered source frames up to the interface limit, with the temporal endpoints retained whenever the interface allows more than one image. The wording of the task and embodiment prompt is kept aligned across models.

No target-robot reference image is used in the main setting. The target is specified in text, which tests whether the model can replace the human actor while preserving the source interaction. We study robot reference images in a separate ablation below. Generated videos remain in their native resolution, duration, and frame rate. At evaluation time, we uniformly sample a fixed number of frames for each metric, so every model is judged with the same evidence budget.

## Transfer-Aware Evaluation

H2R-Bench reports five complementary scores for each generated video. M1–M4 measure whether the source manipulation has been transferred correctly, covering the intended fina state, required action events, functional robot–object contact, and the requested robot embodiment. M5 separately measures task-agnostic video quality. For M1–M4, three MLLM judges independently assess the prescribed visual evidence using a shared 0–4 rubric, where 0 indicates absent or contradictory evidence and 4 indicates clear and complete satisfaction. Judge scores are normalized to [0, 1] and averaged.

M1: Goal-State Completion. Each case defines a set of weighted predicates describing the final state required for task success. Depending on the task, these predicates may specify a spatial or containment relation, a mechanism state, an attachment, a deformation, a material distribution, or a surface change. Judges receive 25 uniformly sampled frames: the sequence provides context, while the final frames determine whether each predicate has been satisfied. The weighted average of the normalized predicate scores gives $S _ { \mathrm { g o a l } }$

M2: Action-Event Completion. Each case also specifies the action events required to complete the task, together with their relative weights. Using the 25 sampled frames, judges determine how clearly each event is carried out. Typical events include grasping, inserting, releasing, pouring, wiping, and folding. The resulting weighted average defines $S _ { \mathrm { a c t i o n } }$ . This metric concerns whether the required operations occur, rather than whether the robot reproduces the human pose, trajectory, or timing.

M3: Functional Contact Transfer. M3 examines whether the generated robot establishes an interaction that is functionally equivalent to the one demonstrated by the human. Judges compare 25 source frames with 25 generated frames, together with a contact specification derived from the source video. The assessment considers the contacted functional region, whether contact is visibly established, the manipulation mode, whether the object response follows from that contact, and whether the interaction is feasible for the requested embodiment. The robot may use a diferent grasp or trajectory, provided that the contact serves the same manipulation function. Non-applicable dimensions are omitted when computing $S _ { \mathrm { c o n t a c t } }$

M4: Embodiment Correctness. M4 assesses whether the generated actor consistently matches the requested robot embodiment. The rubric separates robot presence from morphology: judges examine whether a robot is visible, whether human hands remain, whether the requested embodiment class is used, whether the end-efector type is correct, and whether the robot structure remains consistent over time. This distinction is important because a visible robot arm may still carry an incorrect gripper or hand. If no robot is present, or a human performs the main manipulation, $S _ { \mathrm { e m b } }$ is set to zero.

M5: Video Quality. Unlike M1–M4, M5 is computed without reference to the task annotation. It averages four normalized components: imaging quality, estimated framewise with MUSIQ (Ke et al. 2021); aesthetic quality, obtained from the LAION linear predictor over normalized CLIP ViT-L/14 features (Radford et al. 2021; LAION-AI 2022); temporal stability, measured from adjacent-frame diferences; and motion smoothness, measured by the reconstruction error of AMT-S midpoint interpolation (Li et al. 2023). Their arithmetic mean defines $S _ { \mathrm { v i d e o } }$

The primary benchmark score combines the five normalized components defined above:

$$
\begin{array} { r l } & { \mathrm { H 2 R C o r e } = 1 0 0 \big ( 0 . 1 5 S _ { \mathrm { g o a l } } + 0 . 1 5 S _ { \mathrm { a c t i o n } } } \\ & { \qquad + 0 . 3 0 S _ { \mathrm { c o n t a c t } } + 0 . 3 0 S _ { \mathrm { e m b } } } \\ & { \qquad + 0 . 1 0 S _ { \mathrm { v i d e o } } \big ) . } \end{array}\tag{3}
$$

<table><tr><td rowspan="3">Model</td><td colspan="6">Parallel-Jaw Gripper</td><td colspan="6">Dexterous Hand</td></tr><tr><td colspan="5">Component Metrics</td><td colspan="2">Aggregate</td><td colspan="3">Component Metrics</td><td></td><td>Aggregate</td></tr><tr><td>Goal Comp.</td><td>Action Comp.</td><td>Contact Transfer</td><td>Embod. Correct.</td><td>Video Quality</td><td>H2R Core</td><td>Goal Comp.</td><td>Action Comp.</td><td>Contact Transfer</td><td>Embod. Correct.</td><td>Video Quality</td><td>H2R Core</td></tr><tr><td colspan="2">Video-conditioned generation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Seedance 2.0</td><td>0.725</td><td>0.813</td><td>0.776</td><td>0.768</td><td>0.793</td><td>77.3</td><td>0.744</td><td>0.832</td><td>0.855</td><td>0.911</td><td>0.799</td><td>84.6</td></tr><tr><td>Wan2.7</td><td>0.706</td><td>0.791</td><td>0.766</td><td>0.772</td><td>0.796</td><td>76.5</td><td>0.718</td><td>0.804</td><td>0.835</td><td>0.910</td><td>0.795</td><td>83.1</td></tr><tr><td>Kling-V3</td><td>0.710</td><td>0.807</td><td>0.751</td><td>0.707</td><td>0.798</td><td>74.5</td><td>0.707</td><td>0.800</td><td>0.819</td><td>0.885</td><td>0.802</td><td>81.7</td></tr><tr><td colspan="2">Frame-conditioned generation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Mitty-EPIC14B</td><td>0.581</td><td>0.668</td><td>0.598</td><td>0.585</td><td>0.732</td><td>61.5</td><td>0.587</td><td>0.684</td><td>0.598</td><td>0.392</td><td>0.732</td><td>56.1</td></tr><tr><td>Veo 3.1</td><td>0.725</td><td>0.797</td><td>0.533</td><td>0.100</td><td>0.783</td><td>49.6</td><td>0.715</td><td>0.816</td><td>0.642</td><td>0.227</td><td>0.793</td><td>57.0</td></tr><tr><td>Grok Imagine Video</td><td>0.661</td><td>0.729</td><td>0.443</td><td>0.268</td><td>0.792</td><td>50.1</td><td>0.678</td><td>0.729</td><td>0.469</td><td>0.198</td><td>0.804</td><td>49.2</td></tr><tr><td>LTX-2.3</td><td>0.473</td><td>0.545</td><td>0.292</td><td>0.012</td><td>0.773</td><td>32.1</td><td>0.520</td><td>0.592</td><td>0.377</td><td>0.132</td><td>0.780</td><td>39.8</td></tr><tr><td>SkyReels-V3-R2V</td><td>0.448</td><td>0.610</td><td>0.256</td><td>0.004</td><td>0.787</td><td>31.5</td><td>0.441</td><td>0.607</td><td>0.341</td><td>0.026</td><td>0.789</td><td>34.6</td></tr><tr><td>Wan2.2</td><td>0.492</td><td>0.639</td><td>0.258</td><td>0.000</td><td>0.769</td><td>32.4</td><td>0.512</td><td>0.653</td><td>0.286</td><td>0.000</td><td>0.766</td><td>33.7</td></tr><tr><td>LongCat</td><td>0.472</td><td>0.583</td><td>0.243</td><td>0.000</td><td>0.790</td><td>31.0</td><td>0.428</td><td>0.545</td><td>0.298</td><td>0.020</td><td>0.793</td><td>32.0</td></tr><tr><td>HunyuanVideo 1.5-I2V</td><td>0.535</td><td>0.549</td><td>0.184</td><td>0.005</td><td>0.806</td><td>30.0</td><td>0.499</td><td>0.555</td><td>0.185</td><td>0.041</td><td>0.808</td><td>30.7</td></tr></table>

Table 2: Main-evaluation results by target embodiment. M1–M5 measure goal completion, action completion, contact transfer, embodiment correctness, and Video Quality; H2RCore aggregates all five metrics on a 0–100 scale. Rows are ordered by the sum of the two H2RCore scores within each conditioning group. Bold and underlined entries indicate the best and second-best result in each column.

For M1–M4, the three judge scores are averaged for each video. Dataset-level component scores are then computed over videos, and H2RCore is obtained from the component means. We assign 0.30 each to contact and embodiment, which account for 60% of the score: valid transfer requires both a functionally supported interaction and the requested robot morphology, and equal weights avoid privileging one requirement over the other. Goal and action completion each receive 0.15. These terms preserve the source task, but neither is suficient for transfer because a human-led or wrongembodiment video may still show the correct outcome and actions. Video quality receives 0.10, so visual polish contributes to the score without compensating substantially for failures in contact or embodiment. Supplementary B.5 reports sensitivity to alternative weight choices.

## Experiments

We evaluate 11 current video generators on the 240 transfer cases using their native source-conditioning interfaces. Beyond the leaderboard, we examine how the target embodiment changes performance, whether a target-robot reference image helps, and whether generic video quality tracks transfer quality. Human scores and qualitative examples provide an additional check on the automated metrics.

## Experimental Setup

Benchmark Instances. The main evaluation contains 240 transfer cases constructed from 120 human source videos. Each source is paired with both target embodiments, so the two conditions are evaluated on matched demonstrations. The sources are evenly distributed across the six manipulation families, with 20 sources per family.

Models. We evaluate five proprietary models (Seedance 2.0 (Seedance et al. 2026), Wan2.7 (Wan et al. 2025), Kling-

V3 (Team et al. 2025a), Veo 3.1 (Google 2025), Grok Imagine Video (xAI 2026)), and six open-source models (Wan2.2 (Wan et al. 2025), LTX-2.3 (HaCohen et al. 2026), HunyuanVideo 1.5-I2V (Wu et al. 2025), SkyReels-V3-R2V (Li et al. 2026b), LongCat (Team et al. 2025b), Mitty-EPIC14B (Song et al. 2025)). Together they cover full-video and frameconditioned generation models.

Judges and Metrics. Gemini 3.5 Flash, Qwen3.7-Plus, and GPT-5.4 independently score M1–M4, and we average the three judgments. The metrics use 25 uniformly sampled frames from each video. M3 receives 25 frames from both the source and generated videos. M5 averages normalized MUSIQ imaging quality, CLIP-based aesthetic quality, adjacent-frame temporal stability, and AMT-S interpolation consistency. H2RCore determines the transfer ranking. M1– M5 remain visible in the tables to distinguish task, contact, and embodiment failures.

Native-Interface Protocol. Seedance 2.0, Wan2.7, and Kling-V3 receive the source video. The other models receive ordered source frames within their interface limits. All models use the same task and embodiment information.

## Main Results

Table 2 compares all 11 models under both target embodiments. The three video-conditioned systems occupy the top positions. Seedance 2.0 ranks first with H2RCore scores of 77.3 for the Parallel-Jaw Gripper and 84.6 for the Dexterous Hand, followed by Wan2.7 (76.5 and 83.1) and Kling-V3 (74.5 and 81.7). Among these models, goal and action scores are similar; most of the separation comes from contact transfer and embodiment correctness.

The metric breakdown explains why broad task recognition is not enough. Veo 3.1 attains the best Parallel-Jaw

<table><tr><td>Metric</td><td>Mean change (Hand – Gripper)</td><td>Hand higher (models)</td></tr><tr><td>Goal-State Completion (M1)</td><td>+0.002</td><td>6/11</td></tr><tr><td>Action-Event Completion (M2)</td><td>+0.008</td><td>8/11</td></tr><tr><td>Functional Contact Transfer (M3)</td><td>+0.055</td><td>11/11</td></tr><tr><td>Embodiment Correctness (M4)</td><td>+0.047</td><td>8/11</td></tr><tr><td>Video Quality (M5)</td><td>+0.004</td><td>8/11</td></tr><tr><td>H2RCore (0–100)</td><td>+3.3</td><td>9/11</td></tr></table>

Table 3: Efect of target embodiment across all 11 models. Mean change reports the average score diference between the Dexterous Hand and Parallel-Jaw Gripper. “Hand higher” reports the number of models with a positive diference.
<table><tr><td>Model Ref.</td><td>Goal Action Contact Embod. Quality Core</td></tr><tr><td>Parallel-Jaw Gripper</td></tr><tr><td>No 0.710 0.807 0.751 0.707 0.79874.5 Kling-V3 0.577 0.479 0.779 61.0 77.3</td></tr><tr><td>Yes 0.673 0.760 No 0.725 0.813 Seedance 2.0 Yes 0.711 0.794</td></tr><tr><td>0.776 0.768 0.793 0.674 0.595 0.784</td></tr><tr><td>No 0.706 0.791 0.766 0.772 0.796</td></tr><tr><td>76.5 Wan2.7 Yes 0.714 0.811 0.871 0.875 0.781 83.1</td></tr><tr><td>Dexterous Hand</td></tr><tr><td>No 0.707 0.800 0.819 0.885 0.802 81.7</td></tr><tr><td>Kling-V3</td></tr><tr><td>Yes 0.629 0.734 0.772 0.728 0.799 73.4 No 0.832 0.855 0.911 0.799 84.6</td></tr><tr><td>0.744 Seedance 2.0</td></tr><tr><td>Yes 0.736 0.824 0.876 0.753 0.794 80.2 83.1</td></tr><tr><td>No 0.718 0.804 0.835 0.910 0.795 Wan2.7</td></tr><tr><td>Yes 0.712 0.819 0.905 0.873 0.789 84.2</td></tr></table>

Table 4: Efects of target-robot reference images for three video-conditioned models. “No” and “Yes” indicate generation without and with a target-robot reference image.

Gripper M1 score (0.725) and a Dexterous Hand M2 score of 0.816, yet its M4 scores are only 0.100 and 0.227. HunyuanVideo 1.5-I2V achieves the highest M5 for both targets (0.806 and 0.808), but its contact score stays near 0.185 and its embodiment score near zero. Both models often preserve a recognizable task or polished appearance without completing the requested robot transfer.

Within the video-conditioned subgroup, Seedance’s lead over Wan2.7 is small but consistent. The paired H2RCore diference is +0.79 with a 95% confidence interval of [+0.08, +1.48] for the gripper and +1.52 with [+0.92, +2.16] for the hand. Comparisons with Kling-V3 are farther from zero (Supplementary Table S5).

Human Evaluation. Three human raters independently score M1–M4 for all 11 models under both target embodiments on five sampled source scenes per task family, totaling 660 generated videos. Figure 3 compares within-scene rankings computed from the corresponding transfer subscore. The macro-average Human–MLLM correlation is $\rho ~ = ~ 0 . 8 8 3 .$ showing that the automated evaluation largely preserves human model preferences within a shared task. M5 relies on established metrics that have been independently validated in prior work, and is therefore excluded from the humanagreement analysis.

![](images/18dea6c54e33b82adda0a852382e2ec3e0f18a21eb7fc07a92e8bf9a33375870.jpg)  
Figure 3: Agreement between human and MLLM evaluators. Human raters and MLLMjudges rank generated videos based on the transfer score aggregated from M1–M4. MLLM-based evaluation closely aligns with human judgments, with Spearman correlations above 0.8 across evaluators.

![](images/45ad507cc7c0e9cefba1f5b0bf9d1f8ba77c5fb970b476a7f7a4e59d9adf4dfb.jpg)  
Figure 4: VBench Video Quality versus H2RCore. Unlike VBench, H2RCore better diferentiates the models.

## Embodiment Efects

Gripper vs. Dexterous Hand. Table 3 compares the same 120 sources under the two target embodiments. Nine of the 11 models score higher with the Dexterous Hand, with an average H2RCore gain of 3.3 points. Goal, action, and Video Quality change little, whereas contact transfer rises by 0.055 for all 11 models and embodiment correctness by 0.047 across eight. The efect is especially clear for the three videoconditioned models: the Dexterous Hand raises H2RCore by 7.3 points for Seedance, 6.6 for Wan2.7, and 7.2 for Kling-V3. Its morphology is closer to the human actor in the source video, which is consistent with better preservation of contact during actor replacement. This advantage is not universal, however: Grok Imagine Video and Mitty-EPIC14B both score lower with the hand.

![](images/5d7af810d4c3349424c25629e79e92119113c67e66e743d348bc18470f22da0f.jpg)  
Figure 5: Qualitative H2R transfer results with the shared source video and prompt. The unscored top row is the human source; the generated rows show representative stages of each output. Metric strips report per-video M1–M5 and H2RCore.

Robot Reference Image. Table 4 shows that a target-robot reference image does not provide a uniform benefit. Wan2.7 gains the most. For the gripper, contact transfer rises from 0.766 to 0.871 and embodiment correctness from 0.772 to 0.875, increasing H2RCore from 76.5 to 83.1. Its Dexterous Hand score improves more modestly, from 83.1 to 84.2. Kling-V3 and Seedance move in the opposite direction. Kling loses 13.5 H2RCore points for the gripper and 8.3 for the hand; Seedance loses 8.8 and 4.4. Video Quality changes by only a few hundredths in every setting. The reference image therefore afects adherence to the robot appearance and source interaction in a model-dependent way, without materially changing presentation quality.

## Transfer Diagnostics

VBench vs. H2RCore. Figure 4 compares the VBenchbased Video Quality score with H2RCore across 22 model– embodiment pairs. Video Quality lies between 0.73 and 0.81, while H2RCore spans 30.0 to 84.6. Their rank association is weak (Spearman ρ = 0.14). HunyuanVideo 1.5-I2V leads Video Quality score but sits near the bottom of H2RCore, whereas Mitty-EPIC14B has the lowest Video Quality score and substantially stronger overall scores. The comparison shows that Video Quality alone does not reliably recover the benchmark ranking across models.

Supplementary Figure S1 provides a more detailed breakdown of the discrepancy. Human-led manipulation and contact-region mismatch are prevalent among image/frameconditioned outputs, while end-efector mismatch remains common for the Parallel-Jaw Gripper. The diagnostic categories distinguish failures caused by actor leakage, unsupported contact, and incorrect morphology.

Qualitative Examples. Figure 5 compares three models on the same scoop-and-dump demonstration. Seedance 2.0 maintains visible robot contact with the scoop and transfers the ice into the cup. HunyuanVideo 1.5-I2V leaves the human as the active manipulator, preserving the action without transferring it to the robot. Veo 3.1 produces a plausible interaction with the wrong end efector. The four metrics capture complementary evidence: M1 checks the outcome, M2 the required events, M3 visible robot–scoop contact, and M4 the requested embodiment. A correct final state alone does not establish successful transfer.

## Conclusion

We introduced H2R-Bench, a benchmark for evaluating whether video world models can transfer egocentric human manipulation demonstrations into videos of specified robot embodiments. Across six manipulation families and two target embodiments, H2R-Bench evaluates current video world models through five complementary dimensions, including task goals, action dynamics, functional contact, embodiment consistency, and video quality. Our evaluation of eleven representative video generation models reveals a substantial gap between visual generation quality and embodied transfer capability, suggesting that current video world models still struggle to capture the requirements of crossembodiment manipulation transfer. We hope H2R-Bench will facilitate the development of video world models that bridge the human-to-robot embodiment gap, enabling the transformation of abundant human manipulation observations into reliable robot-centric training resources.

## References

Bahl, S.; Mendonca, R.; Chen, L.; Jain, U.; and Pathak, D. 2023. Afordances from human videos as a versatile representation for robotics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13778–13790.

Bansal, H.; Lin, Z.; Xie, T.; Zong, Z.; Yarom, M.; Bitton, Y.; Jiang, C.; Sun, Y.; Chang, K.-W.; and Grover, A. 2025. Videophy: Evaluating physical commonsense for video generation. In International Conference on Learning Representations, volume 2025, 102075–102121.

Bar-Tal, O.; Chefer, H.; Tov, O.; Herrmann, C.; Paiss, R.; Zada, S.; Ephrat, A.; Hur, J.; Liu, G.; Raj, A.; et al. 2024. Lumiere: A space-time difusion model for video generation. In SIGGRAPH Asia 2024 Conference Papers, 1–11.

Bharadhwaj, H.; Mottaghi, R.; Gupta, A.; and Tulsiani, S. 2024. Track2act: Predicting point tracks from internet videos enables generalizable robot manipulation. In European Conference on Computer Vision, 306–324. Springer.

Black, K.; Brown, N.; Driess, D.; Esmail, A.; Equi, M.; Finn, C.; Fusai, N.; Groom, L.; Hausman, K.; Ichter, B.; et al. 2024. pi\_0: A Vision-Language-Action Flow Model for General Robot Control. arXiv preprint arXiv:2410.24164.

Blattmann, A.; Rombach, R.; Ling, H.; Dockhorn, T.; Kim, S. W.; Fidler, S.; and Kreis, K. 2023. Align your latents: Highresolution video synthesis with latent difusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 22563–22575.

Brohan, A.; Brown, N.; Carbajal, J.; Chebotar, Y.; Dabis, J.; Finn, C.; Gopalakrishnan, K.; Hausman, K.; Herzog, A.; Hsu, J.; et al. 2022. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817.

Brooks, T.; Peebles, B.; Holmes, C.; DePue, W.; Guo, Y.; Jing, L.; Schnurr, D.; Taylor, J.; Luhman, T.; Luhman, E.; et al. 2024. Video generation models as world simulators. OpenAI Blog, 1(8): 1.

Bruce, J.; Dennis, M. D.; Edwards, A.; Parker-Holder, J.; Shi, Y.; Hughes, E.; Lai, M.; Mavalankar, A.; Steigerwald, R.; Apps, C.; et al. 2024. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning.

Chen, H.; Sun, B.; Zhang, A.; Pollefeys, M.; and Leutenegger, S. 2025a. Vidbot: Learning generalizable 3d actions from in-the-wild 2d human videos for zero-shot robotic manipulation. In Proceedings of the Computer Vision and Pattern Recognition Conference, 27661–27672.

Chen, Y.; Ge, Y.; Li, Y.; Ge, Y.; Ding, M.; Shan, Y.; and Liu, X. 2025b. Moto: Latent motion token as the bridging language for robot manipulation. In Proceedings of the IEEE/CVF international conference on computer vision.

Ci, H.; Liu, X.; Yang, P.; Song, Y.; and Shou, M. Z. 2025. H2R-Grounder: A Paired-Data-Free Paradigm for Translating Human Interaction Videos into Physically Grounded Robot Videos. arXiv preprint arXiv:2512.09406.

Damen, D.; Doughty, H.; Farinella, G. M.; Furnari, A.; Kazakos, E.; Ma, J.; Moltisanti, D.; Munro, J.; Perrett, T.; Price, W.; et al. 2022. Rescaling egocentric vision: Collection, pipeline and challenges for epic-kitchens-100. International Journal ofComputer Vision, 130(1): 33–55.

Deng, Y.; Pan, Z.; Zhang, H.; Li, X.; Hu, R.; Ding, Y.; Zou, Y.; Zeng, Y.; and Zhou, D. 2026. Rethinking Video Generation Model for the Embodied World. arXiv preprint arXiv:2601.15282.

Google. 2025. Introducing Veo 3.1 and Advanced Capabilities in Flow. Google Oficial Blog. Released October 15, 2025.

Grauman, K.; Westbury, A.; Byrne, E.; Chavis, Z.; Furnari, A.; Girdhar, R.; Hamburger, J.; Jiang, H.; Liu, M.; Liu, X.; et al. 2022. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 18995–19012.

HaCohen, Y.; Brazowski, B.; Chiprut, N.; Bitterman, Y.; Kvochko, A.; Berkowitz, A.; Shalem, D.; Lifschitz, D.; Moshe, D.; Porat, E.; et al. 2026. LTX-2: Eficient Joint Audio-Visual Foundation Model. arXiv preprint arXiv:2601.03233.

Han, X.; Zhu, B.; Hu, S.; Li, F. M.; Carrington, P.; Zimmermann, R.; and Chen, J. 2026. OSCBench: Benchmarking Object State Change in Text-to-Video Generation. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 30867– 30884.

Hoque, R.; Huang, P.; Yoon, D. J.; Sivapurapu, M.; and Zhang, J. 2025. Egodex: Learning dexterous manipulation from large-scale egocentric video. arXiv preprint arXiv:2505.11709.

Huang, Z.; He, Y.; Yu, J.; Zhang, F.; Si, C.; Jiang, Y.; Zhang, Y.; Wu, T.; Jin, Q.; Chanpaisit, N.; et al. 2024. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21807–21818.

Ji, P.; Xiao, C.; Tai, H.; and Huo, M. 2024. T2vbench: Benchmarking temporal dynamics for text-to-video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5325–5335.

Jiang, F.; Chen, Y.; Xu, K.; Liu, Y.; Wang, H.; Shen, Z.; Lu, J.; Huang, S.; Wang, Y.; Xie, C.; et al. 2026. RoboWM-Bench: A Benchmark for Evaluating World Models in Robotic Manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4455–4460.

Ke, J.; Wang, Q.; Wang, Y.; Milanfar, P.; and Yang, F. 2021. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, 5148–5157.

Kong, W.; Tian, Q.; Zhang, Z.; Min, R.; Dai, Z.; Zhou, J.; Xiong, J.; Li, X.; Wu, B.; Zhang, J.; et al. 2024. Hunyuan-

video: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603.

LAION-AI. 2022. LAION-Aesthetics Predictor V1. https: //github.com/LAION-AI/aesthetic-predictor.

Lepert, M.; Fang, J.; and Bohg, J. 2025. Masquerade: Learning from in-the-wild human videos using data-editing. arXiv preprint arXiv:2508.09976.

Li, D.; Fang, Y.; Chen, Y.; Yang, S.; Cao, S.; Wong, J.; Luo, M.; Wang, X.; Yin, H.; Gonzalez, J.; et al. 2026a. Worldmodelbench: Judging video generation models as world models. Advances in Neural Information Processing Systems, 38.

Li, D.; Fei, Z.; Li, T.; Dou, Y.; Chen, Z.; Yang, J.; Fan, M.; Xu, J.; Wang, J.; Gu, B.; et al. 2026b. Skyreels-v3 technique report. arXiv preprint arXiv:2601.17323.

Li, G.; Lyu, Y.; Liu, Z.; Hou, C.; Zhang, J.; and Zhang, S. 2025a. H2r: A human-to-robot data augmentation for robot pre-training from videos. arXiv preprint arXiv:2505.11920.

Li, H.; Wang, J.; Mei, Z.; Majumdar, A.; Chen, J.; and Zhu, B. 2026c. RoboTrustBench: Benchmarking the Trustworthiness of Video World Models for Robotic Manipulation. arXiv preprint arXiv:2606.01600.

Li, H.; Zhang, I.; Ouyang, R.; Wang, X.; Zhu, Z.; Yang, Z.; Zhang, Z.; Wang, B.; Ni, C.; Qin, W.; et al. 2025b. Mimicdreamer: Aligning human and robot demonstrations for scalable vla training. arXiv preprint arXiv:2509.22199.

Li, Q.; Deng, Y.; Liang, Y.; Luo, L.; Zhou, L.; Yao, C.; Zeng, L.; Feng, Z.; Liang, H.; Xu, S.; et al. 2025c. Scalable vision-language-action model pretraining for robotic manipulation with real-life human activity videos. arXiv preprint arXiv:2510.21571.

Li, Z.; Zhu, Z.-L.; Han, L.-H.; Hou, Q.; Guo, C.-L.; and Cheng, M.-M. 2023. Amt: All-pairs multi-field transforms for eficient frame interpolation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9801–9810.

Liu, Y.; Cun, X.; Liu, X.; Wang, X.; Zhang, Y.; Chen, H.; Liu, Y.; Zeng, T.; Chan, R.; and Shan, Y. 2024. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 22139–22149.

Ma, C.; Mao, Z.; Yang, Y.; Zeng, F.; Shi, Y.; Zhou, Y.; Cao, X.; and Yao, J. 2026a. Reason, Then Re-reason: Crossview Revisiting Improves Spatial Reasoning. arXiv preprint arXiv:2606.11683.

Ma, C.; Yang, Y.; Ju, C.; Zhang, F.; Zhang, Y.; and Wang, Y. 2023a. AttrSeg: Open-Vocabulary Semantic Segmentation via Attribute Decomposition-Aggregation. In Advances in Neural Information Processing Systems (NeurIPS).

Ma, C.; Yang, Y.; Wang, Y.; Zhang, Y.; and Xie, W. 2022a. Open-vocabulary Semantic Segmentation with Frozen Vision-Language Models. In British Machine Vision Conference (BMVC).

Ma, J.; Zhang, E.; Yang, H.; Li, D.; Xu, C.; Wang, G.; and Wang, H. 2026b. Robot Learning from Human Videos: A Survey. arXiv preprint arXiv:2604.27621.

Ma, Y. J.; Kumar, V.; Zhang, A.; Bastani, O.; and Jayaraman, D. 2023b. Liv: Language-image representations and rewards for robotic control. In International Conference on Machine Learning, 23301–23320. PMLR.

Ma, Y. J.; Sodhani, S.; Jayaraman, D.; Bastani, O.; Kumar, V.; and Zhang, A. 2022b. Vip: Towards universal visual reward and representation via value-implicit pre-training. arXiv preprint arXiv:2210.00030.

Mao, Z.; Yang, Y.; Ma, C.; Jiang, D.; Yao, J.; Zhang, Y.; and Wang, Y. 2025. SaFiRe: Saccade-Fixation Reiteration with Mamba for Referring Image Segmentation. In Advances in Neural Information Processing Systems (NeurIPS).

Nair, S.; Rajeswaran, A.; Kumar, V.; Finn, C.; and Gupta, A. 2022. R3m: A universal visual representation for robot manipulation. arXiv preprint arXiv:2203.12601.

O’Neill, A.; Rehman, A.; Maddukuri, A.; Gupta, A.; Padalkar, A.; Lee, A.; Pooley, A.; Gupta, A.; Mandlekar, A.; Jain, A.; et al. 2024. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), 6892–6903. IEEE.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748–8763. PmLR.

Seedance, T.; Chen, D.; Chen, L.; Chen, X.; Chen, Y.; Chen, Z.; Chen, Z.; Cheng, F.; Cheng, T.; Cheng, Y.; et al. 2026. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148.

Shi, Y.; Rong, D.; Chen, C.; Ma, C.; Ni, B.; and Zhang, W. 2025. DARF: Depth-Aware Generalizable Neural Radiance Field. Displays, 88: 102996.

Shi, Y.; Shi, R.; Xiong, Y.; Ni, B.; and Zhang, W. 2026. CEI-3D: Collaborative Explicit-Implicit 3D Reconstruction for Realistic and Fine-Grained Object Editing. arXiv preprint arXiv:2603.11810.

Smith, L.; Dhawan, N.; Zhang, M.; Abbeel, P.; and Levine, S. 2019. Avid: Learning multi-stage tasks via pixel-level translation of human videos. arXiv preprint arXiv:1912.04443.

Song, Y.; Liu, C.; Mao, W.; and Shou, M. Z. 2025. Mitty: Difusion-based Human-to-Robot Video Generation. arXiv preprint arXiv:2512.17253.

Sun, K.; Huang, K.; Liu, X.; Wu, Y.; Xu, Z.; Li, Z.; and Liu, X. 2025. T2v-compbench: A comprehensive benchmark for compositional text-to-video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, 8406–8416.

Team, K.; Chen, J.; Ci, Y.; Du, X.; Feng, Z.; Gai, K.; Guo, S.; Han, F.; He, J.; He, K.; et al. 2025a. Kling-Omni Technical Report. arXiv preprint arXiv:2512.16776.

Team, M. L.; Cai, X.; Huang, Q.; Kang, Z.; Li, H.; Liang, S.; Ma, L.; Ren, S.; Wei, X.; Xie, R.; et al. 2025b. Longcat-video technical report. arXiv preprint arXiv:2510.22200.

Wan, T.; Wang, A.; Ai, B.; Wen, B.; Mao, C.; Xie, C.-W.; Chen, D.; Yu, F.; Zhao, H.; Yang, J.; et al. 2025. Wan: Open

and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314.

Wang, D.; Li, R.; Han, F.; Ma, C.; Song, W.; Wang, S.; Wang, Y.; Xin, Y.; Liu, H.; Zhang, Z.; Ding, S.; Wang, T.; Cheng, Z.; Lin, T.; Jin, C.; Yu, K.; Chen, J.; Wang, W.; Wei, Z.; and Wang, J. 2026a. DeepGen 1.0: A Lightweight Unified Multimodal Model for Advancing Image Generation and Editing. arXiv preprint arXiv:2602.12205.

Wang, D.; Ma, C.; Han, F.; Wu, S.; Song, W.; Wang, Y.; Zhang, Z.; Wang, T.; Wang, S.; Wei, Z.; and Wang, J. 2026b. UniReason 1.0: A Unified Reasoning Framework for World Knowledge Aligned Image Generation and Editing. arXiv preprint arXiv:2602.02437.

Wang, D.; Wei, R.; Shi, Y.; Xu, C.; Chen, S.; Luo, D.; Yang, T.; Yang, X.; Sui, W.; Qin, Y.; Tang, R.; and Mu, Y. 2026c. Rein3D: Reinforced 3D Indoor Scene Generation with Panoramic Video Difusion Models. arXiv preprint arXiv:2604.10578.

Wu, B.; Zou, C.; Li, C.; Huang, D.; Yang, F.; Tan, H.; Peng, J.; Wu, J.; Xiong, J.; Jiang, J.; et al. 2025. Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870.

xAI. 2026. Grok Imagine API. Oficial model release and API documentation.

Xia, D.; Shi, Y.; Mu, Y.; Ji, H.; Ma, C.; Zhou, Y.; Chen, H.; Liu, Y.; Cao, J.; and Zhai, G. 2026. RoboProcessBench: Benchmarking Process-Aware Understanding in Vision-Language Robotic Manipulation. arXiv preprint arXiv:2606.13040.

Xie, S.; Cao, H.; Weng, Z.; Xing, Z.; Chen, H.; Shen, S.; Leng, J.; Wu, Z.; and Jiang, Y.-G. 2026. Human2robot: Learning robot actions from paired human-robot videos. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 11078–11086.

Yang, Y.; Ma, C.; Ju, C.; Zhang, F.; Yao, J.; Zhang, Y.; and Wang, Y. 2024a. Multi-Modal Prototypes for Open-World Semantic Segmentation. International Journal ofComputer Vision (IJCV).

Yang, Y.; Ma, C.; Mao, Z.; Yao, J.; Zhang, Y.; and Wang, Y. 2025. MoMa: Modulating Mamba for Adapting Image Foundation Models to Video Recognition. In Proceedings of the International Conference on Machine Learning (ICML).

Yang, Y.; Ma, C.; Yao, J.; Zhong, Z.; Zhang, Y.; and Wang, Y. 2024b. ReMamber: Referring Image Segmentation with Mamba Twister. In European Conference on Computer Vision (ECCV).

Yang, Y.; Zhuang, X.; Cai, Y.; Ma, C.; Bai, S.; Yao, J.; Zhang, Y.; Lin, J.; and Wang, Y. 2026. GenMask: Adapting DiT for Segmentation via Direct Mask Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Ye, S.; Jang, J.; Jeon, B.; Joo, S. J.; Yang, J.; Peng, B.; Mandlekar, A.; Tan, R.; Chao, Y.-W.; Lin, B. Y.; et al. 2025. Latent action pretraining from videos. In International Conference on Learning Representations, volume 2025, 28213–28239.

Zhang, J.; Chen, X.; Chen, A.; Lv, C.; Li, D.; Zhou, G.; Yin, H.; Yuan, H.; Li, H.; Li, J.; et al. 2026a. Qwen-RobotWorld Technical Report: Unifying Embodied World

Modeling through Language-Conditioned Video Generation. arXiv preprint arXiv:2606.17030.

Zhang, Y.; Dong, W.; Shi, Y.; Liang, Y.; Gao, J.; Yang, Q.; Lyu, Y.; Liang, Z.; Liu, Y.; Xu, C.; Guo, X.; Sui, W.; Jin, Y.; Yang, X.; Xu, Y.; and Mu, Y. 2026b. R3DP: Real-Time 3D-Aware Policy for Embodied Manipulation. arXiv preprint arXiv:2603.14498.

Zheng, R.; Niu, D.; Xie, Y.; Wang, J.; Xu, M.; Jiang, Y.; Castañeda, F.; Hu, F.; Tan, Y. L.; Fu, L.; et al. 2026. Egoscale: Scaling dexterous manipulation with diverse egocentric human data. arXiv preprint arXiv:2602.16710.

Zitkovich, B.; Yu, T.; Xu, S.; Xu, P.; Xiao, T.; Xia, F.; Wu, J.; Wohlhart, P.; Welker, S.; Wahid, A.; et al. 2023. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, 2165– 2183. PMLR.

## A Dataset Construction and Annotation Quality Control

## A.1 Source Selection and Task Stratification

H2R-Bench is built from egocentric source clips selected from the EgoDex test split (Hoque et al. 2025) and balanced over six physical-state manipulation families. We retain clips only when the manipulated entities, a task-relevant state transition, and suficient visual evidence for the interaction can be identified. Target embodiments are paired after source selection: the parallel-jaw gripper and dexterous-hand conditions therefore use exactly the same human source evidence.

The taxonomy is designed for the benchmark rather than inherited from EgoDex activity labels. It groups recurring embodied-manipulation problems by the physical state change that defines task completion, following the emphasis on diverse object interactions in robot-learning and egocentric-manipulation datasets (Brohan et al. 2022; O’Neill et al. 2024; Hoque et al. 2025). The evaluation dimensions follow the evidence needed to determine whether that state change is reproduced by the requested robot. Table S1 summarizes both design choices.

## A.2 Clip Grounding and Structured Annotation

For each benchmark clip, Qwen3.7-Plus generates a structured annotation from the 5-second source video and 32 uniformly sampled frames, including the first and last frame. This annotation preserves functional information such as contact, support, release, and expected object response while avoiding a prescribed robot trajectory. Every annotation is manually checked against the source video. Corrections are incorporated before the annotations are used for evaluation.

## A.3 Annotation Acceptance Criteria

This stage determines whether a source clip has suficient evidence to support the benchmark evaluation. Each annotation is stored in a fixed JSON schema containing the task family, goal checks, required action events, interaction requirements, and embodiment strategies. We verify that these fields are complete and internally consistent before accepting a case.

Each case receives an explicit validity decision. We exclude clips when occlusion, ambiguous object states, or unclear interactions make the task goal or required contact impossible to assess reliably. For accepted cases, remaining minor ambiguities are recorded as notes for later analysis but do not change the evaluation protocol or oficial benchmark set.

## B Additional Experimental Details

## B.1 Diagnostic Failure Rates

On the 120-source main evaluation set, Figure S1 aggregates diagnostic failures identified from the structured M2– M4 outputs. A failure is recorded when the corresponding judge-averaged diagnostic component is below 0.5; categories are non-exclusive because one generation can violate embodiment, contact, and action requirements simultaneously. In this native-interface collection, image/frameconditioned systems frequently retain human-led manipulation and exhibit weak visible contact support or an incorrect contact region or mode. The three video-conditioned systems have lower observed failure rates, but this descriptive comparison does not identify an interface efect because model and interface vary together. Parallel-Jaw Gripper transfer still shows frequent end-efector mismatch, while required action events are sometimes omitted for both target embodiments. These patterns demonstrate the value of reporting the transfer components alongside H2RCore: the benchmark identifies whether a failure arises from action completion, functional contact, or target embodiment rather than reducing all errors to a single quality score.

M3 additionally treats source-scene or task-entity substitution as a hard failure: generic-looking contact in a diferent scene is not contact transfer from the source demonstration. The source-grounding decision is made independently for each generated video before the contact dimensions are aggregated.

## B.2 Matched Source-Conditioning Ablation

To test whether sparse image evidence can substitute for a source video, we compare Seedance 2.0’s native videoconditioned run with a nine-image setting that provides nine chronologically ordered, uniformly sampled source frames. This ablation uses 24 matched sources from the main evaluation set, with four sources from each task family. Both target embodiments are evaluated for every source, giving 48 transfer cases. Replacing the complete source video with nine images preserves much of the final-state and broad action evidence, but substantially degrades functional contact and embodiment realization. H2RCore falls by 27.6 points for the Parallel-Jaw Gripper and 41.4 points for the Dexterous Hand, while Video Quality changes by less than 0.02 in both settings (Table S3). Qualitative inspection and M4 diagnoses show that the nine-image interface frequently reproduces the human manipulation rather than reliably replacing the human actor with the requested robot.

## B.3 Performance Across Task Families

On the 120-source main evaluation set, aggregate scores conceal substantial task dependence. Figure S2 separates H2RCore by manipulation family and target embodiment. Averaged over models, deformable-object configuration (F4) remains the strongest family for both targets, while the other interaction families exhibit lower aggregate transfer scores. This ordering suggests that dificulty is governed less by visible deformation itself than by whether success requires precise, localized functional contact and a partially hidden state transition. The panels also expose model-specific preferences: Seedance and Wan2.7 are comparatively stable across families, whereas Grok Imagine Video and Veo 3.1 show substantially larger task-dependent variation. A model can therefore appear competitive on visible final states in selected tasks while remaining unreliable for a particular interaction class.

Table S10 provides the per-task-family breakdown on the main evaluation set for the six manipulation families. Each family stresses a diferent aspect of human-to-robot transfer:

<table><tr><td>Element</td><td>Embodied-manipulation con- cern</td><td>H2R-Bench operationalization</td><td>Design origin</td></tr><tr><td colspan="4">Task families: task-defining physical state changes</td></tr><tr><td>F1: Rigid rearrange- ment</td><td>Object pose, support, or con- tainment</td><td>Place or transport a rigid object into the demonstrated spatial relation.</td><td>Rigid transport and placement tasks.</td></tr><tr><td>F2: Mechanism actu- ation</td><td>State of an articulated mecha- nism</td><td>Open, close, toggle, press, or rotate a task- relevant mechanism.</td><td>Interaction with articulated ob- jects.</td></tr><tr><td>F3: Insertion and as- sembly</td><td>Connection, fit, or attachment relation</td><td>Establish or remove a constrained connec- tion between entities.</td><td>Precision alignment and con- strained contact.</td></tr><tr><td>F4: Deformable con- figuration</td><td>Non-rigid shape or configura- tion</td><td>Produce the demonstrated fold, bend, com- pression, or shape change.</td><td>Deformable-object manipula- tion.</td></tr><tr><td>F5: Bulk-material transfer</td><td>Distribution or containment of material</td><td>Pour, scoop, transfer, or mix material be- tween regions or containers. Produce a visible local change through wip-</td><td>Many-particle and material- flow manipulation. Tool-mediated local transfor-</td></tr><tr><td>F6: Surface/material transformation</td><td>Local surface condition or ma- terial integrity</td><td>ing, spreading, cutting, peeling, or related interaction.</td><td>mation.</td></tr><tr><td colspan="4">Evaluation dimensions: evidence required for valid H2R generation</td></tr><tr><td>M1: Goal-State Completion</td><td>Was the demonstrated task state reached?</td><td>Score weighted predicates over the visible final state.</td><td>Task-success and outcome evaluation in robotic video</td></tr><tr><td>M2: Action-Event Completion</td><td>Were the required manipula-</td><td>Score completion of source-derived action</td><td>benchmarks. Action-completeness evalua- tion in robotic video bench-</td></tr><tr><td>M3: Functional Con-</td><td>Did robot contact support the</td><td>events. Evaluate contact region, establishment,</td><td>marks. Contact-mediated manipu- lation and H2R interaction</td></tr><tr><td>tact Transfer M4: Embodiment</td><td>source-consistent object re- sponse? Did the requested robot per-</td><td>mode, temporal object response, and embodiment-compatible strategy. Evaluate robot presence, human absence,</td><td>grounding. Robot-structure and embodi-</td></tr><tr><td>Correctness</td><td>form the manipulation?</td><td>embodiment category, end-effector subtype, and temporal structure. Measure imaging quality, aesthetic quality,</td><td>ment compliance.</td></tr><tr><td>M5: Video Quality</td><td>Is the generated video visually and temporally well formed?</td><td>temporal stability, and motion smoothness.</td><td>Task-agnostic video-generation quality.</td></tr></table>

Table S1: Design rationale for the H2R-Bench task taxonomy and evaluation dimensions. The six families are organized by taskdefining physical state changes rather than semantic activity labels; M1–M5 cover task realization, source-relative interaction, target embodiment, and presentation quality.

rigid transport, mechanism actuation, insertion and assembly, deformable configuration change, bulk-material transfer, and surface transformation. Parallel-Jaw Gripper and Dexterous Hand targets are reported in separate column groups, and row colors identify the task families.

## B.4 Attribute-Based Analysis

Using the same 120-source main evaluation set, task-family averages describe the dominant physical state change, but they do not expose whether dificulty is associated with the interaction interface or the number of required actions. Figure S3 therefore groups the evaluations using annotation fields available across the benchmark: direct manipulation versus tool use, and tasks with two–three versus four or more required events. The splits reveal systematic variation that family-level means can conceal. They are descriptive diagnostics rather than causal comparisons, because task attributes naturally co-occur.

## B.5 Statistical Reporting

Table S4 reports percentile 95% confidence intervals (CIs) for H2RCore on the 120-source main evaluation set. The intervals are obtained by stratified bootstrap resampling of source tasks within each task family.

Interpreting paired comparisons. All displayed paired intervals exclude zero under the stated bootstrap procedure. The smallest separation is Seedance–Wan2.7 for the Parallel-Jaw Gripper target.

Weight sensitivity. We recompute embodiment-specific rankings using equal weights (0.20, 0.20, 0.20, 0.20, 0.20), a larger quality weight (0.15, 0.15, 0.25, 0.25, 0.20), and a transfer-only variant (0.15, 0.15, 0.35, 0.35, 0). Kendall rank correlation with the default ranking is between 0.927 and 1.000 across these settings, indicating that the aggregate ordering is stable under these alternatives.

## B.6 Detailed Evaluation Protocol

All MLLM-based metrics use 25 uniformly sampled frames from each input video and expose only the annotation fields required for the metric. Judges return structured JSON containing per-item scores, evidence-frame indices, and rationales; scripts normalize scores and perform every aggregation. Let c denote a generated video, $m \in \mathcal { I }$ one of the three judges, and $[ z ] _ { 4 } = z / 4$ the normalization of an integer score $\bar { z } \in \{ 0 , 1 , 2 , \dot { 3 } , 4 \}$ . A score of 0 denotes absent or contradicted evidence, 1 weak evidence, 2 partial completion, 3 mostly correct evidence with a minor defect or ambiguity, and 4 clear, complete, and stable evidence.

<table><tr><td>Stage</td><td>Output</td><td>Quality control</td></tr><tr><td>Source curation</td><td>Curated clips across six families</td><td>Identifiable manipulated entities, visible task-relevant state change, and sufficient interaction evidence.</td></tr><tr><td>Clip grounding</td><td>32 ordered source frames per clip</td><td>First and last frames are retained; annotations identify initial state, contact, core transition, and final state when visible.</td></tr><tr><td>Structured annotation</td><td>Goals, events, milestones, interac- tion, and embodiment strategies</td><td>Strict JSON schema with canonical event identifiers and a fixed six-family taxonomy.</td></tr><tr><td>Completion and validation</td><td>Schema-complete case records</td><td>Only missing fields are completed; existing non-empty fields are preserved; schema and type mismatches are rejected.</td></tr><tr><td>Manual verification</td><td>All benchmark annotations</td><td>Every record is checked against its source video and corrected before evaluation.</td></tr><tr><td>Uncertainty accounting</td><td>Validity and ambiguity fields</td><td>Every record has a validity decision; notes document residual un- certainty without changing the official evaluation set.</td></tr></table>

Table S2: Dataset construction and annotation quality controls. Source-side annotations define the benchmark specification; detailed frame-level evidence and uncertainty fields are not exposed to generation models.

![](images/7a11e5035a51c3fac91ee3a5ef0eb4358c981e3da6836900c992ba276e7ffdcf.jpg)  
Figure S1: Failure-type rates derived from structured M2–M4 diagnostics on the 120-source main evaluation set. Each bar reports the percentage of evaluated videos in the corresponding conditioning group whose judge-averaged diagnostic component is below 0.5. Categories are non-exclusive. Video-conditioned models show lower observed rates of human-led manipulation and contact failures, whereas Parallel-Jaw Gripper transfer remains sensitive to end-efector mismatch and both embodiments retain action-event failures.

<table><tr><td>Target</td><td>Input</td><td>M1</td><td>M2 M3</td><td>M4</td><td>M5</td><td>Core</td></tr><tr><td>Parallel-Jaw Video</td><td></td><td>0.737 0.850 </td><td>0.709</td><td>0.598</td><td>0.793</td><td>70.9</td></tr><tr><td>Gripper</td><td>9 images</td><td>0.744 0.848</td><td>0.284</td><td>0.104</td><td>0.783</td><td>43.4</td></tr><tr><td>Dexterous</td><td>Video</td><td></td><td>0.768 0.857 0.861 0</td><td>0.898</td><td>0.799</td><td>85.1</td></tr><tr><td>Hand</td><td></td><td></td><td>9 images 0.753 0.856 0.313 0.079 0.78243.7</td><td></td><td></td><td></td></tr></table>

Table S3: Matched source-conditioning ablation for Seedance 2.0 on 24 sources. Both target embodiments are evaluated for every source. “Video” uses the full source clip; “9 ordered images” uses uniformly sampled chronological frames.

Judge configuration. M1–M4 use Gemini 3.5 Flash (gemini-3.5-flash), Qwen3.7-Plus (qwen3.7-plus), and GPT-5.4 (gpt-5.4) at temperature 0. The three judges contribute equally to each MLLM-based metric.

M1: Goal-State Completion. Each case defines a set of weighted final-state predicates $\mathcal { G } _ { c } .$ A predicate may specify a spatial relation, mechanism state, attachment relation, material distribution, deformation, or local surface change. Judge m receives 25 uniformly sampled generated frames and assigns $r _ { g } ^ { m } \in \{ 0 , \ldots , 4 \}$ to every $g \in { \mathcal G } _ { c }$ based on the state visible at the end of the sequence. Thus, M1 evaluates the final state without discarding the preceding evidence that shows whether it was reached and preserved. With annotation weight $w _ { g } ,$ the per-judge score is

![](images/42d823d797b00586bad594b2d971dd3e43323adb4e3dbc7890b78a447cbe412e.jpg)

![](images/c47699bc8b194393119b9aab947abe114bb59d1401e7c87a0a9cbe7e1fc0c8dd.jpg)  
Figure S2: Task-family H2RCore profiles across the six manipulation families. Each panel compares Parallel-Jaw Gripper and Dexterous Hand transfer for the same 11 models. Higher is better, and all panels share a 0–100 scale, exposing both taskdependent dificulty and embodiment-dependent model preferences.

<table><tr><td>Model</td><td>Parallel-Jaw Gripper Core [95% CI]</td><td>Dexterous Hand Core [95% CI]</td></tr><tr><td>Kling-V3</td><td>74.5 [73.6, 75.3]</td><td>81.7 [80.8, 82.6]</td></tr><tr><td>Seedance 2.0</td><td>77.3 [76.5, 78.1]</td><td>84.6 [84.0, 85.3]</td></tr><tr><td>Wan2.7</td><td>76.5 [75.7, 77.4]</td><td>83.1 [82.3, 83.9]</td></tr><tr><td>Grok Imagine</td><td>50.1 [49.0, 51.1]</td><td>49.2 [48.3, 50.1]</td></tr><tr><td>Veo 3.1</td><td>49.6 [48.9, 50.4]</td><td>57.0 [56.1, 57.8]</td></tr><tr><td>Hunyuan-1.5</td><td>30.0 [28.8, 31.2]</td><td>30.7 [29.5, 31.8]</td></tr><tr><td>LTX-2.3</td><td>32.1 [31.0, 33.2]</td><td>39.8 [38.6, 40.9]</td></tr><tr><td>Mitty-EPIC</td><td>61.5 [60.5, 62.5]</td><td>56.1 [55.2, 57.0]</td></tr><tr><td>SkyReels-V3</td><td>31.5 [30.5, 32.6]</td><td>34.6 [33.6, 35.6]</td></tr><tr><td>LongCat</td><td>31.0 [29.9, 32.1]</td><td>32.0 [30.9, 33.2]</td></tr><tr><td>Wan2.2</td><td>32.4 [31.4, 33.3]</td><td>33.7 [32.7, 34.7]</td></tr></table>

Table S4: Main-evaluation H2RCore point estimates and percentile 95% confidence intervals. Intervals summarize variation across source tasks under the five-metric aggregation rule.
<table><tr><td>Comparison</td><td>Gripper ∆ Core [95% CI]</td><td>Hand ∆ Core [95% CI]</td></tr><tr><td></td><td>Seedance − Wan2.7 +0.79 [+0.08, +1.48] +1.52 [+0.92, +2.16]</td><td></td></tr><tr><td>Seedance — Kling</td><td></td><td>+2.82 [+2.04, +3.61] +2.89 [+2.15, +3.73]</td></tr><tr><td>Wan2.7 – Kling</td><td></td><td>+2.04 [+1.40, +2.66] +1.37[+0.66, +2.16]</td></tr></table>

Table S5: Paired source-task H2RCore diferences for the leading video-conditioned models. Positive values favor the left model; percentile 95% CIs pair the same source tasks.

$$
S _ { \mathrm { g o a l } } ^ { m } = \frac { \sum _ { g \in { \mathcal { G } } _ { c } } w _ { g } [ r _ { g } ^ { m } ] _ { 4 } } { \sum _ { g \in { \mathcal { G } } _ { c } } w _ { g } } .\tag{S1}
$$

We additionally report weighted predicate coverage at strict thresholds,

$$
C _ { \mathrm { g o a l } } ^ { m } ( \tau ) = \frac { \sum _ { g \in \mathcal { G } _ { c } } w _ { g } \mathbf { 1 } [ r _ { g } ^ { m } \geq \tau ] } { \sum _ { g \in \mathcal { G } _ { c } } w _ { g } } , \qquad \tau \in \{ 3 , 4 \} ,\tag{S2}
$$

which distinguishes partially reached outcomes from clearly completed ones.

M2: Action-Event Completion. Each annotation contains weighted required events $\mathcal { E } _ { c }$ . Judges inspect the 25 uniformly sampled generated frames and assign every event e a completion score $a _ { e } ^ { m } \in \{ 0 , \ldots , 4 \}$ . Event completion is the weighted mean

$$
S _ { \mathrm { a c t i o n } } ^ { m } = \frac { \sum _ { e \in { \mathcal { E } } _ { c } } w _ { e } [ a _ { e } ^ { m } ] _ { 4 } } { \sum _ { e \in { \mathcal { E } } _ { c } } w _ { e } } .\tag{S3}
$$

We also report weighted event coverage at scores $\geq 3$ and = 4. M2 summarizes the visible completion of the required action events across the generated sequence.

M3: Functional Contact Transfer. M3 evaluates whether the robot realizes the source interaction functionally rather than merely reproducing a similar object trajectory. A contact specification is derived from the source annotation and 25 uniformly sampled source frames. It identifies manipulated entities, functional contact regions, manipulation modes, expected object responses, support or bimanual roles when relevant, and target-embodiment constraints. The judge compares this specification and the 25 source frames with 25 generated frames along five dimensions: (i) contact-region transfer, whether the robot acts on the same or a functionally equivalent region; (ii) contact establishment, whether visible contact is sustained enough to control the object; (iii)

![](images/853679c29a459fd0611b29e776fa58d8cd7b9b6988c2842b5eb0216ef0b1512c.jpg)

![](images/0b46b790282382c6ec0311ac63287a0533086e36ffde6ff5326e4b36251cae5c.jpg)  
Figure S3: H2RCore comparison across annotation-derived task attributes on the 120-source main evaluation set. The left panel separates direct manipulation from tool use; the right groups tasks by the number of required action events. Bars show mean H2RCore over the evaluated model outputs for each target embodiment, values are printed above the bars, and parenthetical counts give the number of source scenes in each attribute group. Both panels use the same vertical scale.

manipulation-mode transfer, whether the robot uses a compatible functional mode such as grasping, pinching, pushing, pulling, stabilizing, or scooping; (iv) temporally supported object response, whether the expected state change visibly follows robot contact or control; and (v) embodimentcompatible contact strategy, whether the visible strategy is compatible with the requested gripper or dexterous hand. The fourth item is visual-temporal evidence, not a claim that contact physically caused the response; M4 separately evaluates the robot’s visible structure and identity.

Before these dimensions are scored, the judge verifies source grounding: the generated video must preserve the source scene and camera context together with task-relevant objects, tools, and containers, allowing only task-required state changes and embodiment replacement. A substantial scene or task-entity substitution is a hard failure and receives zero M3 credit; this rule prevents generic contact in a diferent scene from being counted as source-contact transfer.

Applicability is specified by the case annotation and is shared by all judges. Let $\mathcal { C } _ { c }$ denote this fixed subset of dimensions and let $\mathsf { \bar { h } } _ { q } ^ { m } \in \{ 0 , \dots , 4 \}$ be judge m’s score for dimension q. A dimension marked as non-applicable is excluded for every judge rather than assigned a neutral score. The per-judge M3 score is

$$
S _ { \mathrm { c o n t a c t } } ^ { m } = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { i f ~ s o u r c e ~ g r o u n d i n g ~ f a i l s , } } \\ { \displaystyle { \frac { 1 } { | \mathcal { C } _ { c } | } \sum _ { q \in \mathcal { C } _ { c } } [ h _ { q } ^ { m } ] _ { 4 } } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{S4}
$$

Evidence indices are validated independently within the source and generated 25-frame sequences; consequently, source and generated frame identifiers cannot be conflated during judging.

M4: Embodiment Correctness. M4 uses 25 uniformly sampled generated frames and the requested embodiment specification only. Judges score five dimensions: robot-actor presence (p), absence of human hands or arms (u), broad embodiment-category match (b), end-efector correctness (e), and structural consistency over time (s). The presence term assesses visible robotic embodiment irrespective ofsubtype; category and end-efector are deliberately separate, so a visible dexterous hand can receive presence credit while receiving low morphology credit when the parallel-jaw gripper is requested. Let $p ^ { \overleftarrow { m } } , u ^ { m } , b ^ { m } , e ^ { m } , s ^ { m } \in \{ 0 , \ldots , 4 \}$ denote the five scores from judge m. A judge-marked hard failure, triggered when no robotic actor is visible or when human hands perform the principal manipulation, receives zero. Otherwise, the per-judge score is

$$
\begin{array} { c } { { S _ { \mathrm { e m b } } ^ { m } = 0 . 2 0 [ p ^ { m } ] _ { 4 } + 0 . 2 0 [ u ^ { m } ] _ { 4 } + 0 . 2 0 [ b ^ { m } ] _ { 4 } } } \\ { { + 0 . 2 5 [ e ^ { m } ] _ { 4 } + 0 . 1 5 [ s ^ { m } ] _ { 4 } . } } \end{array}\tag{S5}
$$

The larger end-efector weight reflects that the distinction between the parallel-jaw gripper and dexterous hand is central to the benchmark, while the hard-failure rule prevents a stable but incorrect human actor from receiving credit through the remaining dimensions.

M5: Video Quality. M5 is intentionally task-agnostic. Let $I _ { 1 } , \ldots , I _ { T }$ be the decoded frames of generated video $c ,$ and let MAE average absolute RGB diferences over all pixels and channels. Framewise components traverse the decoded sequence, while temporal components use adjacent pairs or interpolation triplets. For MUSIQ, frames retain their aspect ratio and are downsampled only when the longer side exceeds 512 pixels; the aesthetic predictor uses the standard 224-pixel CLIP center crop. Imaging quality is the mean framewise output of the MUSIQ model trained on SPAQ (Ke et al. 2021), normalized by 100:

$$
v _ { c , \mathrm { I Q } } = \frac { 1 } { 1 0 0 T } \sum _ { t = 1 } ^ { T } \mathrm { M U S I Q } ( I _ { t } ) .\tag{S6}
$$

Aesthetic quality is the mean output of the LAION linear aesthetic predictor applied to normalized CLIP ViT-L/14 image features (Radford et al. 2021; LAION-AI 2022), normalized by 10:

$$
v _ { c , \mathrm { A Q } } = \frac { 1 } { 1 0 T } \sum _ { t = 1 } ^ { T } \mathrm { L A I O N } \Bigg ( \frac { \mathrm { C L I P } ( I _ { t } ) } { \| \mathrm { C L I P } ( I _ { t } ) \| _ { 2 } } \Bigg ) .\tag{S7}
$$

Temporal stability is one minus the mean normalized diference between consecutive frames:

$$
v _ { c , { \mathrm { T F } } } = 1 - { \frac { 1 } { 2 5 5 ( T - 1 ) } } \sum _ { t = 1 } ^ { T - 1 } { \mathrm { M A E } } ( I _ { t } , I _ { t + 1 } ) .\tag{S8}
$$

For motion smoothness, AMT-S (Li et al. 2023) predicts each odd frame $\widehat { I } _ { 2 k + 1 }$ from its two neighboring even frames at interpolation time 0.5. With K valid triplets,

$$
v _ { c , \mathrm { M S } } = 1 - \frac { 1 } { 2 5 5 K } \sum _ { k = 0 } ^ { K - 1 } \mathrm { M A E } ( I _ { 2 k + 1 } , \widehat { I } _ { 2 k + 1 } ) .\tag{S9}
$$

Each component is clipped to [0, 1] before the four values are averaged:

$$
S _ { \mathrm { v i d e o } , c } = { \frac { 1 } { 4 } } \sum _ { q \in \{ \mathrm { I Q } , \mathrm { A Q } , \mathrm { T F } , \mathrm { M S } \} } v _ { c , q } .\tag{S10}
$$

M5 therefore measures the presentation quality of a video without rewarding an incorrect manipulation or embodiment.

Judge, dataset, and overall aggregation. For $X \in$ {goal, action, contact, emb}, we average the three judge scores for each video and then average across the evaluated set D:

$$
S _ { X } = \frac { 1 } { | \mathcal { D } | } \sum _ { c \in \mathcal { D } } \left( \frac { 1 } { | \mathcal { I } | } \sum _ { m \in \mathcal { I } } S _ { X , c } ^ { m } \right) , \qquad | \mathcal { I } | = 3 .\tag{S11}
$$

All tables retain the component scores because they diagnose distinct failure modes. H2RCore aggregates the five datasetlevel component scores:

$$
\begin{array} { r l } & { \mathrm { H 2 R C o r e } = 1 0 0 \big ( 0 . 1 5 S _ { \mathrm { g o a l } } + 0 . 1 5 S _ { \mathrm { a c t i o n } } } \\ & { \qquad + 0 . 3 0 S _ { \mathrm { c o n t a c t } } + 0 . 3 0 S _ { \mathrm { e m b } } } \\ & { \qquad + 0 . 1 0 S _ { \mathrm { v i d e o } } \big ) . } \end{array}\tag{S12}
$$

This aggregation prioritizes contact transfer and embodiment correctness, while assigning a smaller contribution to Video Quality. For human-agreement analyses, where raters score M1–M4 but not M5, we use the transfer subscore $1 0 0 ( 0 . 2 0 S _ { \mathrm { g o a l } } + 0 . 2 0 S _ { \mathrm { a c t i o n } } + 0 . 3 0 S _ { \mathrm { c o n t a c t } } + 0 . 3 0 S _ { \mathrm { e m b } } )$

## B.7 Human–Automatic Agreement

Three human raters score M1–M4 on the same random sample of 660 videos described in the main paper. We compare the final human scores with the averaged MLLM scores for records with complete paired evaluations. Pearson correlation is computed across videos for each transfer metric. The corresponding coeficients are 0.791 for M1, 0.818 for M2, 0.880 for M3, and 0.877 for M4 (Table S6). Combining M1– M4 with the transfer-subscore weights defined above gives $r = 0 . 9 3 0$ . Within-scene model-ranking agreement is reported separately in Figure 3.

## B.8 Inter-Judge Agreement

We assess agreement among Gemini, Qwen, and GPT on the 120-source main evaluation set for M1–M4. For each metric and judge pair $( p , q )$ , we compute the Pearson correlation $r _ { p q }$ between their per-video scores. Scores are computed from the structured per-item outputs using the arithmetic definitions in Section B.6; M5 is excluded because it does not use MLLM judges.

<table><tr><td>Score</td><td>M1</td><td>M2</td><td>M3</td><td>M4</td><td>Transfer Subscore</td></tr><tr><td>Pearson r</td><td>0.791</td><td>0.818</td><td>0.880</td><td>0.877</td><td>0.930</td></tr></table>

Table S6: Human–automatic agreement for paired human and MLLM evaluations. M1–M4 are compared per video; the transfer subscore combines these four metrics without M5.

<table><tr><td>Judge Pair</td><td>M1</td><td>M2</td><td>M3</td><td>M4</td></tr><tr><td>Gemini / Qwen</td><td>0.695</td><td>0.691</td><td>0.741</td><td>0.874</td></tr><tr><td>Gemini / GPT</td><td>0.582</td><td>0.636</td><td>0.726</td><td>0.891</td></tr><tr><td>Qwen / GPT</td><td>0.656</td><td>0.690</td><td>0.809</td><td>0.892</td></tr></table>

Table S7: Pairwise inter-judge agreement on the 120-source main evaluation set. Each cell reports Pearson correlation r between per-video scores.

## C Model Descriptions and Implementation Setups

We evaluate 11 representative video generators through their publicly available hosted interfaces or local implementations. This section records the generation configuration used by H2R-Bench rather than attempting to equate heterogeneous provider-side sampling controls. The main comparison uses the strongest documented source-conditioning interface available for each model. All systems receive an aligned casespecific English prompt that specifies the task outcome and either a parallel-jaw gripper or a dexterous hand; its presentation is adapted only to the input limits of the corresponding interface. The prompt does not expose evaluation-only annotations. Unless stated otherwise, no target-robot appearance reference is supplied in the main experiment; target-reference results are reported separately in Table 4.

## C.1 Computing Infrastructure

Locally hosted generation, video preprocessing, M5 computation, and result aggregation were performed on a workstation running Ubuntu 24.04.2 LTS with two Intel Xeon Gold 6144 processors (16 physical cores and 32 hardware threads), 251 GiB of system memory, and four NVIDIA GeForce RTX 4090 GPUs with 24 GiB of reported memory per GPU. The system used NVIDIA driver 580.173.02, CUDA 13.0, and cuDNN 9.19.0. The software environment used Python 3.10.16, PyTorch 2.11.0, torchvision 0.26.0, Transformers 4.57.1, Difusers 0.36.0, Accelerate 1.12.0, OpenCV 4.13.0, NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3, scikit-learn 1.5.0, Decord 0.6.0, and FFmpeg 6.1.1. Commercial video generators and the three MLLM judges were accessed through provider-hosted APIs; their server-side hardware is not disclosed to us.

## C.2 Commercially Hosted Models

Seedance 2.0. We use the oficial Volcengine Ark interface in its video-conditioned mode. Each request provides the complete source clip through the supported public-video interface together with the case-specific target-embodiment prompt. The main setting uses no robot reference image.

<table><tr><td>Model</td><td>Interface</td><td>Source input</td></tr><tr><td>Seedance 2.0</td><td>Video</td><td>Full clip</td></tr><tr><td>Wan2.7</td><td>Video</td><td>Full clip</td></tr><tr><td>Kling-V3</td><td>Video</td><td>Full clip</td></tr><tr><td>Veo 3.1</td><td>Frame</td><td>First + last</td></tr><tr><td>Grok Imagine Video</td><td>Frame</td><td>Up to 7 frames</td></tr><tr><td>HunyuanVideo 1.5-I2V</td><td>Frame</td><td>Ordered frame(s)</td></tr><tr><td>LTX-2.3</td><td>Frame</td><td>Ordered frame(s)</td></tr><tr><td>Mitty-EPIC14B</td><td>Frame</td><td>Ordered frame(s)</td></tr><tr><td>SkyReels-V3-R2V</td><td>Frame</td><td>Ordered frame(s)</td></tr><tr><td>LongCat</td><td>Frame</td><td>Ordered frame(s)</td></tr><tr><td>Wan2.2</td><td>Frame</td><td>First frame</td></tr></table>

Table S8: Source-conditioning interfaces in the main experiment. Frame inputs are sampled in temporal order from the source demonstration. No model receives a target-robot reference image.

Wan2.7. We use the provider’s video-conditioned interface, supplying the source demonstration and the common target-embodiment prompt. This setting tests direct retargeting from temporally ordered human manipulation evidence without a target-robot image.

Kling-V3. We use Kling’s video-conditioned generation interface with the full source demonstration and the common prompt. The requested robot morphology is specified in text only in the main experiment.

Veo 3.1. We use the supported image-conditioned interface, providing the first and last source frames in temporal order rather than the source video. The prompt identifies the task and requested embodiment; no target-robot reference image is included.

Grok Imagine Video. We use the image-conditioned interface with multiple source frames sampled in temporal order (up to seven uniformly spaced frames when supported). The model receives the common task and embodiment prompt and no robot appearance reference.

## C.3 Open-Source or Locally Hosted Models

HunyuanVideo 1.5-I2V. We run the image-to-video implementation with source frame conditioning and the targetembodiment prompt. The source video itself is not passed to the model.

LTX-2.3. We use the available image/frame-conditioned generation mode, supplying source frame evidence and the common prompt. We retain the provider output for evaluation rather than rendering a target robot externally.

Mitty-EPIC14B. We use the locally hosted image/frameconditioned implementation with source frame inputs and the same target-embodiment prompt used across models.

SkyReels-V3-R2V. We use the available image/frameconditioned interface with temporally ordered source frames and the shared prompt.

LongCat. We use the locally hosted image/frameconditioned configuration with source frame inputs and the shared prompt. No target-robot reference image is supplied.

<table><tr><td>Model</td><td>Decoded Size</td><td>Frames @ fps</td></tr><tr><td>Seedance 2.0</td><td> $1 2 8 0 \times 7 2 0$ </td><td>121 @ 24</td></tr><tr><td>Wan2.7</td><td> $1 2 8 0 \times 7 2 0$ </td><td>150 @ 30</td></tr><tr><td>Kling-V3</td><td> $1 2 8 0 \times 7 2 0$ </td><td>121 @ 24</td></tr><tr><td>Veo 3.1</td><td> $1 2 8 0 \times 7 2 0$ </td><td>144 @ 24</td></tr><tr><td>Grok Imagine Video</td><td> $1 2 8 0 \times 7 2 0$ </td><td>121 @ 24</td></tr><tr><td>HunyuanVideo 1.5-I2V</td><td> $1 2 8 0 \times 7 2 0$ </td><td>121 @ 24</td></tr><tr><td>LTX-2.3</td><td> $1 2 8 0 \times 7 2 0$ </td><td>120 @ 24</td></tr><tr><td>Mitty-EPIC14B</td><td> $1 2 8 0 \times 7 2 0$ </td><td>37 @ 8</td></tr><tr><td>SkyReels-V3-R2V</td><td> $1 2 8 0 \times 7 2 0$ </td><td>121 @ 24</td></tr><tr><td>LongCat</td><td> $8 3 2 \times 4 8 0$ </td><td>80 @ 15</td></tr><tr><td>Wan2.2</td><td>720P area</td><td>121 @ 24</td></tr></table>

Table S9: Native decoded output profiles in the main experiment. “Frames @ $\mathrm { f p s } ^ { \prime \prime }$ reports decoded frame count and frame rate. Wan2.2 preserves source aspect ratio within its 720P-area setting; audio is removed before evaluation.

Wan2.2. We use the local Wan2.2-TI2V-5B implementation. Its one-image input budget is filled by the first frame of the source demonstration; the full source clip is not supplied. Parallel-Jaw Gripper and Dexterous Hand generations use their corresponding case prompts and the same seed for a paired source clip. The generation profile produces 121 frames at 24 fps (approximately 5 seconds) and preserves the source aspect ratio within the model’s 720P-area setting.

## C.4 Visual Inputs and Output Profiles

For every frame-conditioned interface, source images are kept in chronological order. When an interface accepts one source image, we use the first source frame; when it accepts multiple source images, we use uniformly spaced frames including the temporal endpoints, up to the interface input budget. Veo 3.1 receives the first and last frames, and Grok Imagine Video receives up to seven uniformly spaced frames. The video-conditioned systems receive the complete source clip. Table S9 reports the decoded profiles of the outputs retained for evaluation. These native profiles are not normalized before generation; comparability is instead established by the fixed evaluation frame budgets described below.

## C.5 Output Handling

We retain completed provider or local outputs in their native generation format and do not post-process clips to insert, render, or remove an actor. For comparable evaluation, every input video is decoded and uniformly sampled into 25 frames for M1–M4; M3 uses one 25-frame sequence from the source and one from the generated video. M5 is computed on the generated video itself. Thus, the interface diference is explicit at generation time while the evaluation evidence budget is fixed across models. Tables S8 and S9 summarize the source-conditioning interfaces and native output profiles.

## C.6 Prompt Examples

The visual condition is supplied through each model’s native interface, while the text specifies the shared task, scene, contact, and embodiment constraints. Figure S5 presents abridged generation prompts for the same dexterous-hand case, the embodiment clause, the structured annotation input, and an M3 judge example. Source-video annotations are evaluation-only and are not exposed to the video generators. Complete case-specific prompts, annotation templates, judge prompts, and JSON schemas are included in the benchmark release.

## D Limitations

H2R-Bench evaluates visible evidence of human-to-robot transfer, not physical executability or downstream policy performance. Its 120 EgoDex sources and two target embodiments cover only part of the variation found in manipulation settings. The current benchmark is also limited to short clips; extending it to longer demonstrations may benefit from eficient spatiotemporal modeling (Yang et al. 2025). The native-interface comparison also combines model capability with diferences in source-conditioning interfaces. Finally, sampled visual evidence and MLLM judgments can remain uncertain under occlusion, subtle contact, or severe generation artifacts despite multi-judge aggregation and human validation. Future extensions could use dense visual grounding for ambiguous task entities (Ma et al. 2022a, 2023a; Mao et al. 2025; Yang et al. 2024a,b, 2026). Geometry-aware novel-view synthesis may provide complementary visual evidence (Shi et al. 2025), and fine-grained 3D reconstruction may support object-geometry diagnostics (Shi et al. 2026).

## E Human Evaluation Interface

Figure S4 shows the interface used to collect human M1–M4 scores. Each task group contains five shared source scenes from one model and task family under both target embodiments, yielding ten videos. Annotators inspect the source and generated clips side by side, verify the requested embodiment, and score the source-derived criteria on a common 0–4 scale. Scores are stored separately for each annotator, and automatic MLLM judgments are not displayed.

![](images/efed6b491f74479dc46db7c6f3b40cdc244b411c3741498f6202f71c644d93f7.jpg)  
Figure S4: Human-evaluation interface for H2R-Bench. The page pairs each source demonstration with its generated robot video and presents the requested embodiment and expandable M1–M4 criteria. The sidebar records task assignment and progress over balanced ten-video groups.

<table><tr><td rowspan=2 colspan=14>Parallel-Jaw Gripper                           Dexterous HandGoal    Action Contact Embod.                             Contact Embod.</td></tr><tr><td rowspan=1 colspan=1>Contact</td><td rowspan=1 colspan=1>Embod.</td><td rowspan=1 colspan=3>VideoGoalAction</td><td rowspan=1 colspan=1>Contact</td><td rowspan=1 colspan=1>Embod.</td><td rowspan=1 colspan=2>Video</td></tr><tr><td rowspan=1 colspan=5>Task Model     Completion Completion Tr</td><td rowspan=1 colspan=1>ansfer C</td><td rowspan=1 colspan=1>orrect. Q</td><td rowspan=1 colspan=3>uality Core Completion Completion Tr</td><td rowspan=1 colspan=1>ansfer C</td><td rowspan=1 colspan=1>orrect. Q</td><td rowspan=1 colspan=2>uality Core</td></tr><tr><td rowspan=1 colspan=5>Kling-V3       0.736   0.837</td><td rowspan=1 colspan=1>0.782</td><td rowspan=1 colspan=1>0.735</td><td rowspan=1 colspan=3>0.803 77.1    0.738   0.837</td><td rowspan=1 colspan=1>0.847</td><td rowspan=1 colspan=1>0.886</td><td rowspan=1 colspan=1>0.807</td><td rowspan=1 colspan=1>83.7</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>Seedance 2.0    0.762   0.857</td><td rowspan=1 colspan=1>0.836</td><td rowspan=1 colspan=1>0.824</td><td rowspan=1 colspan=3>0.798 82.1    0.775   0.853</td><td rowspan=1 colspan=1>0.880</td><td rowspan=1 colspan=1>0.908</td><td rowspan=1 colspan=1>0.802</td><td rowspan=1 colspan=1>86.1</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>Wan2.7        0.744   0.843</td><td rowspan=1 colspan=1>0.824</td><td rowspan=1 colspan=1>0.776</td><td rowspan=1 colspan=3>0.793 79.7   0.777   0.845</td><td rowspan=1 colspan=1>0.890</td><td rowspan=1 colspan=1>0.913</td><td rowspan=1 colspan=1>0.799</td><td rowspan=1 colspan=1>86.4</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>Grok Imagine   0.608   0.704</td><td rowspan=1 colspan=1>0.555</td><td rowspan=1 colspan=1>0.430</td><td rowspan=1 colspan=3>0.806 57.3   0.656   0.742</td><td rowspan=1 colspan=1>0.561</td><td rowspan=1 colspan=1>0.368</td><td rowspan=1 colspan=1>0.811</td><td rowspan=1 colspan=1>57.0</td></tr><tr><td rowspan=1 colspan=1>id</td><td rowspan=1 colspan=4>Veo 3.1        0.751   0.791</td><td rowspan=1 colspan=1>0.574</td><td rowspan=1 colspan=1>0.197</td><td rowspan=1 colspan=3>0.790 54.2   0.780   0.830</td><td rowspan=1 colspan=1>0.653</td><td rowspan=1 colspan=1>0.305</td><td rowspan=1 colspan=1>0.800</td><td rowspan=1 colspan=1>60.9</td></tr><tr><td rowspan=1 colspan=1>R</td><td rowspan=1 colspan=4>Hunyuan-1.5    0.520   0.446</td><td rowspan=1 colspan=1>0.188</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=3>0.812 28.2   0.483   0.429</td><td rowspan=1 colspan=1>0.209</td><td rowspan=1 colspan=1>0.027</td><td rowspan=1 colspan=1>0.814</td><td rowspan=1 colspan=1>28.9</td></tr><tr><td rowspan=1 colspan=1>FI</td><td rowspan=1 colspan=4>LTX-2.3        0.424   0.550</td><td rowspan=1 colspan=1>0.344</td><td rowspan=1 colspan=1>0.018</td><td rowspan=1 colspan=3>0.784 33.3   0.455   0.558</td><td rowspan=1 colspan=1>0.422</td><td rowspan=1 colspan=1>0.170</td><td rowspan=1 colspan=1>0.791</td><td rowspan=1 colspan=1>40.9</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>Mitty-EPIC     0.522   0.591</td><td rowspan=1 colspan=1>0.540</td><td rowspan=1 colspan=1>0.575</td><td rowspan=1 colspan=3>0.733 57.5   0.528   0.640</td><td rowspan=1 colspan=1>0.591</td><td rowspan=1 colspan=1>0.402</td><td rowspan=1 colspan=1>0.733</td><td rowspan=1 colspan=1>54.6</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>SkyReels-V3    0.408   0.527</td><td rowspan=1 colspan=1>0.267</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=3>0.792 29.9    0.376   0.489</td><td rowspan=1 colspan=1>0.333</td><td rowspan=1 colspan=1>0.061</td><td rowspan=1 colspan=1>0.795</td><td rowspan=1 colspan=1>32.8</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>LongCat       0.412</td><td rowspan=1 colspan=1>0.502</td><td rowspan=1 colspan=1>0.208</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=3>0.799 27.9   0.394   0.476</td><td rowspan=1 colspan=1>0.306</td><td rowspan=1 colspan=1>0.032</td><td rowspan=1 colspan=1>0.798</td><td rowspan=1 colspan=1>31.2</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Wan2.2        0.377</td><td rowspan=1 colspan=1>0.547</td><td rowspan=1 colspan=1>0.229</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.776 28.5</td><td rowspan=1 colspan=1>0.422</td><td rowspan=1 colspan=1>0.553</td><td rowspan=1 colspan=1>0.249</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.774</td><td rowspan=1 colspan=1>29.8</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Kling-V3</td><td rowspan=1 colspan=2>0.641</td><td rowspan=1 colspan=1>0.792</td><td rowspan=1 colspan=1>0.686</td><td rowspan=1 colspan=1>0.682</td><td rowspan=1 colspan=1>0.802 70.6</td><td rowspan=1 colspan=1>0.666</td><td rowspan=1 colspan=1>0.796</td><td rowspan=1 colspan=1>0.770</td><td rowspan=1 colspan=1>0.885</td><td rowspan=1 colspan=1>0.804</td><td rowspan=1 colspan=1>79.6</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Seedance 2.0</td><td rowspan=1 colspan=2>0.760</td><td rowspan=1 colspan=1>0.847</td><td rowspan=1 colspan=1>0.737</td><td rowspan=1 colspan=1>0.742</td><td rowspan=1 colspan=1>0.797 76.4</td><td rowspan=1 colspan=1>0.735</td><td rowspan=1 colspan=1>0.854</td><td rowspan=1 colspan=1>0.828</td><td rowspan=1 colspan=1>0.927</td><td rowspan=1 colspan=1>0.801</td><td rowspan=1 colspan=1>84.5</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Wan2.7</td><td rowspan=1 colspan=2>0.695</td><td rowspan=1 colspan=1>0.783</td><td rowspan=1 colspan=1>0.704</td><td rowspan=1 colspan=1>0.771</td><td rowspan=1 colspan=1>0.794 74.4</td><td rowspan=1 colspan=1>0.676</td><td rowspan=1 colspan=1>0.795</td><td rowspan=1 colspan=1>0.778</td><td rowspan=1 colspan=1>0.913</td><td rowspan=1 colspan=1>0.799</td><td rowspan=1 colspan=1>80.8</td></tr><tr><td rowspan=1 colspan=1>m</td><td rowspan=1 colspan=1>Grok Imagine</td><td rowspan=1 colspan=2>0.646</td><td rowspan=1 colspan=1>0.729</td><td rowspan=1 colspan=1>0.491</td><td rowspan=1 colspan=1>0.527</td><td rowspan=1 colspan=1>0.791 59.1</td><td rowspan=1 colspan=1>0.683</td><td rowspan=1 colspan=1>0.770</td><td rowspan=1 colspan=1>0.573</td><td rowspan=1 colspan=1>0.344</td><td rowspan=1 colspan=1>0.807</td><td rowspan=1 colspan=1>57.4</td></tr><tr><td rowspan=1 colspan=1>m</td><td rowspan=1 colspan=3>Veo 3.1        0.670</td><td rowspan=1 colspan=1>0.754</td><td rowspan=1 colspan=1>0.508</td><td rowspan=1 colspan=1>0.049</td><td rowspan=1 colspan=1>0.785 45.9</td><td rowspan=1 colspan=2>0.691   0.824</td><td rowspan=1 colspan=1>0.620</td><td rowspan=1 colspan=1>0.151</td><td rowspan=1 colspan=1>0.793</td><td rowspan=1 colspan=1>53.8</td></tr><tr><td rowspan=1 colspan=1>h</td><td rowspan=1 colspan=3>Hunyuan-1.5    0.433</td><td rowspan=1 colspan=1>0.517</td><td rowspan=1 colspan=1>0.178</td><td rowspan=1 colspan=1>0.014</td><td rowspan=1 colspan=1>0.807 28.1</td><td rowspan=1 colspan=2>0.399   0.481</td><td rowspan=1 colspan=1>0.202</td><td rowspan=1 colspan=1>0.123</td><td rowspan=1 colspan=1>0.809</td><td rowspan=1 colspan=1>31.0</td></tr><tr><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=3>LTX-2.3       0.478</td><td rowspan=1 colspan=1>0.514</td><td rowspan=1 colspan=1>0.278</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=3>0.774 31.0   0.523   0.613</td><td rowspan=1 colspan=1>0.328</td><td rowspan=1 colspan=1>0.157</td><td rowspan=1 colspan=1>0.786</td><td rowspan=1 colspan=1>39.4</td></tr><tr><td rowspan=1 colspan=1>PE</td><td rowspan=1 colspan=3>Mitty-EPIC     0.515</td><td rowspan=1 colspan=1>0.588</td><td rowspan=1 colspan=1>0.537</td><td rowspan=1 colspan=1>0.579</td><td rowspan=1 colspan=3>0.746 57.5   0.507   0.609</td><td rowspan=1 colspan=1>0.540</td><td rowspan=1 colspan=1>0.393</td><td rowspan=1 colspan=1>0.746</td><td rowspan=1 colspan=1>52.2</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>SkyReels-V3    0.454</td><td rowspan=1 colspan=1>0.552</td><td rowspan=1 colspan=1>0.223</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=3>0.790 29.7   0.425   0.559</td><td rowspan=1 colspan=1>0.303</td><td rowspan=1 colspan=1>0.011</td><td rowspan=1 colspan=1>0.794</td><td rowspan=1 colspan=1>32.1</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>LongCat        0.454</td><td rowspan=1 colspan=1>0.579</td><td rowspan=1 colspan=1>0.243</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=3>0.792 30.7    0.403   0.577</td><td rowspan=1 colspan=1>0.275</td><td rowspan=1 colspan=1>0.011</td><td rowspan=1 colspan=1>0.794</td><td rowspan=1 colspan=1>31.2</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Wan2.2        0.427</td><td rowspan=1 colspan=1>0.607</td><td rowspan=1 colspan=1>0.234</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=3>0.763 30.2    0.526   0.672</td><td rowspan=1 colspan=1>0.264</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.762</td><td rowspan=1 colspan=1>33.5</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Kling-V3       0.683</td><td rowspan=1 colspan=1>0.748</td><td rowspan=1 colspan=1>0.748</td><td rowspan=1 colspan=1>0.735</td><td rowspan=1 colspan=3>0.794 73.9   0.680   0.739</td><td rowspan=1 colspan=1>0.796</td><td rowspan=1 colspan=1>0.907</td><td rowspan=1 colspan=1>0.799</td><td rowspan=1 colspan=1>80.4</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Seedance 2.0    0.722</td><td rowspan=1 colspan=1>0.792</td><td rowspan=1 colspan=1>0.834</td><td rowspan=1 colspan=1>0.839</td><td rowspan=1 colspan=3>0.789 80.8   0.737   0.832</td><td rowspan=1 colspan=1>0.867</td><td rowspan=1 colspan=1>0.907</td><td rowspan=1 colspan=1>0.793</td><td rowspan=1 colspan=1>84.7</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Wan2.7        0.623</td><td rowspan=1 colspan=1>0.727</td><td rowspan=1 colspan=1>0.742</td><td rowspan=1 colspan=1>0.785</td><td rowspan=1 colspan=3>0.789 73.9   0.660   0.762</td><td rowspan=1 colspan=1>0.813</td><td rowspan=1 colspan=1>0.912</td><td rowspan=1 colspan=1>0.792</td><td rowspan=1 colspan=1>81.0</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Grok Imagine   0.599</td><td rowspan=1 colspan=1>0.688</td><td rowspan=1 colspan=1>0.422</td><td rowspan=1 colspan=1>0.320</td><td rowspan=1 colspan=3>0.788 49.4   0.603   0.650</td><td rowspan=1 colspan=1>0.398</td><td rowspan=1 colspan=1>0.299</td><td rowspan=1 colspan=1>0.807</td><td rowspan=1 colspan=1>47.8</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Veo 3.1</td><td rowspan=1 colspan=1>0.687</td><td rowspan=1 colspan=1>0.764</td><td rowspan=1 colspan=1>0.504</td><td rowspan=1 colspan=1>0.072</td><td rowspan=1 colspan=3>0.780 46.9   0.619   0.770</td><td rowspan=1 colspan=1>0.597</td><td rowspan=1 colspan=1>0.240</td><td rowspan=1 colspan=1>0.791</td><td rowspan=1 colspan=1>53.8</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Hunyuan-1.5</td><td rowspan=1 colspan=1>0.462</td><td rowspan=1 colspan=1>0.448</td><td rowspan=1 colspan=1>0.214</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=3>0.803 28.1    0.409   0.469</td><td rowspan=1 colspan=1>0.193</td><td rowspan=1 colspan=1>0.022</td><td rowspan=1 colspan=1>0.805</td><td rowspan=1 colspan=1>27.7</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=2>LTX-2.3</td><td rowspan=1 colspan=1>0.426</td><td rowspan=1 colspan=1>0.470</td><td rowspan=1 colspan=1>0.276</td><td rowspan=1 colspan=1>0.046</td><td rowspan=1 colspan=3>0.773 30.8   0.510   0.516</td><td rowspan=1 colspan=1>0.343</td><td rowspan=1 colspan=1>0.195</td><td rowspan=1 colspan=1>0.779</td><td rowspan=1 colspan=1>39.3</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Mitty-EPIC     0.571</td><td rowspan=1 colspan=1>0.634</td><td rowspan=1 colspan=1>0.597</td><td rowspan=1 colspan=1>0.602</td><td rowspan=1 colspan=2>0.737 61.4   0.578</td><td rowspan=1 colspan=1>0.664</td><td rowspan=1 colspan=1>0.590</td><td rowspan=1 colspan=1>0.470</td><td rowspan=1 colspan=1>0.737</td><td rowspan=1 colspan=1>57.8</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>SkyReels-V3    0.223</td><td rowspan=1 colspan=1>0.511</td><td rowspan=1 colspan=1>0.204</td><td rowspan=1 colspan=1>0.011</td><td rowspan=1 colspan=1>0.785 25.3</td><td rowspan=1 colspan=1>0.304</td><td rowspan=1 colspan=1>0.522</td><td rowspan=1 colspan=1>0.279</td><td rowspan=1 colspan=1>0.011</td><td rowspan=1 colspan=1>0.787</td><td rowspan=1 colspan=1>29.0</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>LongCat</td><td rowspan=1 colspan=1>0.482</td><td rowspan=1 colspan=1>0.586</td><td rowspan=1 colspan=1>0.238</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.786 31.0</td><td rowspan=1 colspan=1>0.365</td><td rowspan=1 colspan=1>0.474</td><td rowspan=1 colspan=1>0.271</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.789</td><td rowspan=1 colspan=1>28.6</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Wan2.2</td><td rowspan=1 colspan=1>0.423</td><td rowspan=1 colspan=1>0.528</td><td rowspan=1 colspan=1>0.194</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.768 27.8</td><td rowspan=1 colspan=1>0.416</td><td rowspan=1 colspan=1>0.598</td><td rowspan=1 colspan=1>0.251</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.767</td><td rowspan=1 colspan=1>30.4</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Kling-V3</td><td rowspan=1 colspan=1>0.795</td><td rowspan=1 colspan=1>0.862</td><td rowspan=1 colspan=1>0.847</td><td rowspan=1 colspan=1>0.690</td><td rowspan=1 colspan=1>0.773 78.7</td><td rowspan=1 colspan=1>0.804</td><td rowspan=1 colspan=1>0.857</td><td rowspan=1 colspan=1>0.892</td><td rowspan=1 colspan=1>0.853</td><td rowspan=1 colspan=1>0.776</td><td rowspan=1 colspan=1>85.0</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Seedance 2.0</td><td rowspan=1 colspan=1>0.770</td><td rowspan=1 colspan=1>0.821</td><td rowspan=1 colspan=1>0.837</td><td rowspan=1 colspan=1>0.801</td><td rowspan=1 colspan=1>0.770 80.7</td><td rowspan=1 colspan=1>0.833</td><td rowspan=1 colspan=1>0.869</td><td rowspan=1 colspan=1>0.897</td><td rowspan=1 colspan=1>0.910</td><td rowspan=1 colspan=1>0.778</td><td rowspan=1 colspan=1>87.5</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Wan2.7</td><td rowspan=1 colspan=1>0.799</td><td rowspan=1 colspan=1>0.818</td><td rowspan=1 colspan=1>0.852</td><td rowspan=1 colspan=1>0.778</td><td rowspan=1 colspan=1>0.775 80.9</td><td rowspan=1 colspan=1>0.789</td><td rowspan=1 colspan=1>0.824</td><td rowspan=1 colspan=1>0.877</td><td rowspan=1 colspan=1>0.893</td><td rowspan=1 colspan=1>0.777</td><td rowspan=1 colspan=1>85.1</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Grok Imagine</td><td rowspan=1 colspan=1>0.762</td><td rowspan=1 colspan=1>0.783</td><td rowspan=1 colspan=1>0.566</td><td rowspan=1 colspan=1>0.220</td><td rowspan=1 colspan=1>0.767 54.4</td><td rowspan=1 colspan=1>0.752</td><td rowspan=1 colspan=1>0.793</td><td rowspan=1 colspan=1>0.629</td><td rowspan=1 colspan=1>0.131</td><td rowspan=1 colspan=1>0.776</td><td rowspan=1 colspan=1>53.7</td></tr><tr><td rowspan=1 colspan=1>m</td><td rowspan=1 colspan=2>Veo 3.1</td><td rowspan=1 colspan=1>0.778</td><td rowspan=1 colspan=1>0.819</td><td rowspan=1 colspan=1>0.670</td><td rowspan=1 colspan=1>0.240</td><td rowspan=1 colspan=1>0.766 58.9</td><td rowspan=1 colspan=1>0.785</td><td rowspan=1 colspan=1>0.864</td><td rowspan=1 colspan=1>0.783</td><td rowspan=1 colspan=1>0.387</td><td rowspan=1 colspan=1>0.778</td><td rowspan=1 colspan=1>67.6</td></tr><tr><td rowspan=1 colspan=1>e</td><td rowspan=1 colspan=2>Hunyuan-1.5</td><td rowspan=1 colspan=1>0.608</td><td rowspan=1 colspan=1>0.634</td><td rowspan=1 colspan=1>0.138</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.796 30.7</td><td rowspan=1 colspan=1>0.566</td><td rowspan=1 colspan=1>0.644</td><td rowspan=1 colspan=1>0.164</td><td rowspan=1 colspan=1>0.013</td><td rowspan=1 colspan=1>0.800</td><td rowspan=1 colspan=1>31.5</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=2>LTX-2.3</td><td rowspan=1 colspan=1>0.485</td><td rowspan=1 colspan=1>0.595</td><td rowspan=1 colspan=1>0.334</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.753 33.8</td><td rowspan=1 colspan=1>0.540</td><td rowspan=1 colspan=1>0.615</td><td rowspan=1 colspan=1>0.462</td><td rowspan=1 colspan=1>0.102</td><td rowspan=1 colspan=1>0.760</td><td rowspan=1 colspan=1>41.8</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Mitty-EPIC</td><td rowspan=1 colspan=1>0.699</td><td rowspan=1 colspan=1>0.753</td><td rowspan=1 colspan=1>0.635</td><td rowspan=1 colspan=1>0.554</td><td rowspan=1 colspan=1>0.693 64.4</td><td rowspan=1 colspan=1>0.697</td><td rowspan=1 colspan=1>0.752</td><td rowspan=1 colspan=1>0.644</td><td rowspan=1 colspan=1>0.378</td><td rowspan=1 colspan=1>0.694</td><td rowspan=1 colspan=1>59.3</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>SkyReels-V3</td><td rowspan=1 colspan=1>0.540</td><td rowspan=1 colspan=1>0.716</td><td rowspan=1 colspan=1>0.292</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.774 35.3</td><td rowspan=1 colspan=1>0.491</td><td rowspan=1 colspan=1>0.695</td><td rowspan=1 colspan=1>0.448</td><td rowspan=1 colspan=1>0.048</td><td rowspan=1 colspan=1>0.774</td><td rowspan=1 colspan=1>40.4</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>LongCat</td><td rowspan=1 colspan=1>0.409</td><td rowspan=1 colspan=1>0.596</td><td rowspan=1 colspan=1>0.250</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.778 30.4</td><td rowspan=1 colspan=1>0.408</td><td rowspan=1 colspan=1>0.514</td><td rowspan=1 colspan=1>0.349</td><td rowspan=1 colspan=1>0.045</td><td rowspan=1 colspan=1>0.783</td><td rowspan=1 colspan=1>33.5</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Wan2.2</td><td rowspan=1 colspan=1>0.627</td><td rowspan=1 colspan=1>0.796</td><td rowspan=1 colspan=1>0.305</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.753 38.0</td><td rowspan=1 colspan=1>0.599</td><td rowspan=1 colspan=1>0.760</td><td rowspan=1 colspan=1>0.325</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.751</td><td rowspan=1 colspan=1>37.6</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Kling-V3</td><td rowspan=1 colspan=1>0.659</td><td rowspan=1 colspan=1>0.796</td><td rowspan=1 colspan=1>0.742</td><td rowspan=1 colspan=1>0.749</td><td rowspan=1 colspan=1>0.804 74.6</td><td rowspan=1 colspan=2>0.667   0.820</td><td rowspan=1 colspan=1>0.803</td><td rowspan=1 colspan=1>0.901</td><td rowspan=1 colspan=1>0.809</td><td rowspan=1 colspan=1>81.5</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Seedance 2.0</td><td rowspan=1 colspan=1>0.665</td><td rowspan=1 colspan=1>0.819</td><td rowspan=1 colspan=1>0.676</td><td rowspan=1 colspan=1>0.627</td><td rowspan=1 colspan=1>0.795 69.3</td><td rowspan=1 colspan=1>0.710</td><td rowspan=1 colspan=1>0.828</td><td rowspan=1 colspan=1>0.844</td><td rowspan=1 colspan=1>0.903</td><td rowspan=1 colspan=1>0.800</td><td rowspan=1 colspan=1>83.5</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Wan2.7</td><td rowspan=1 colspan=1>0.680</td><td rowspan=1 colspan=1>0.808</td><td rowspan=1 colspan=1>0.769</td><td rowspan=1 colspan=1>0.766</td><td rowspan=1 colspan=1>0.795 76.3</td><td rowspan=1 colspan=1>0.690</td><td rowspan=1 colspan=1>0.826</td><td rowspan=1 colspan=1>0.841</td><td rowspan=1 colspan=1>0.913</td><td rowspan=1 colspan=1>0.796</td><td rowspan=1 colspan=1>83.3</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Grok Imagine</td><td rowspan=1 colspan=1>0.697</td><td rowspan=1 colspan=1>0.778</td><td rowspan=1 colspan=1>0.277</td><td rowspan=1 colspan=1>0.046</td><td rowspan=1 colspan=1>0.802 39.8</td><td rowspan=1 colspan=2>0.754   0.772</td><td rowspan=1 colspan=1>0.315</td><td rowspan=1 colspan=1>0.025</td><td rowspan=1 colspan=1>0.814</td><td rowspan=1 colspan=1>41.2</td></tr><tr><td rowspan=1 colspan=1>k</td><td rowspan=1 colspan=2>Veo 3.1</td><td rowspan=1 colspan=1>0.732</td><td rowspan=1 colspan=1>0.828</td><td rowspan=1 colspan=1>0.486</td><td rowspan=1 colspan=1>0.025</td><td rowspan=1 colspan=1>0.791 46.6</td><td rowspan=1 colspan=2>0.710   0.817</td><td rowspan=1 colspan=1>0.613</td><td rowspan=1 colspan=1>0.137</td><td rowspan=1 colspan=1>0.800</td><td rowspan=1 colspan=1>53.4</td></tr><tr><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=2>Hunyuan-1.5</td><td rowspan=1 colspan=1>0.620</td><td rowspan=1 colspan=1>0.704</td><td rowspan=1 colspan=1>0.206</td><td rowspan=1 colspan=1>0.018</td><td rowspan=1 colspan=1>0.811 34.7</td><td rowspan=1 colspan=2>0.557   0.666</td><td rowspan=1 colspan=1>0.188</td><td rowspan=1 colspan=1>0.011</td><td rowspan=1 colspan=1>0.815</td><td rowspan=1 colspan=1>32.5</td></tr><tr><td rowspan=1 colspan=1>F5</td><td rowspan=1 colspan=2>LTX-2.3</td><td rowspan=1 colspan=1>0.449</td><td rowspan=1 colspan=1>0.541</td><td rowspan=1 colspan=1>0.236</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.786 29.8</td><td rowspan=1 colspan=2>0.522   0.656</td><td rowspan=1 colspan=1>0.353</td><td rowspan=1 colspan=1>0.091</td><td rowspan=1 colspan=1>0.788</td><td rowspan=1 colspan=1>38.9</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Mitty-EPIC     0.564</td><td rowspan=1 colspan=1>0.731</td><td rowspan=1 colspan=1>0.676</td><td rowspan=1 colspan=1>0.643</td><td rowspan=1 colspan=1>0.741 66.4</td><td rowspan=1 colspan=2>0.580   0.719</td><td rowspan=1 colspan=1>0.667</td><td rowspan=1 colspan=1>0.385</td><td rowspan=1 colspan=1>0.741</td><td rowspan=1 colspan=1>58.4</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>SkyReels-V3    0.427</td><td rowspan=1 colspan=1>0.627</td><td rowspan=1 colspan=1>0.278</td><td rowspan=1 colspan=1>0.016</td><td rowspan=1 colspan=1>0.793 32.6</td><td rowspan=1 colspan=2>0.421   0.634</td><td rowspan=1 colspan=1>0.344</td><td rowspan=1 colspan=1>0.024</td><td rowspan=1 colspan=1>0.795</td><td rowspan=1 colspan=1>34.8</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>LongCat</td><td rowspan=1 colspan=1>0.413</td><td rowspan=1 colspan=1>0.564</td><td rowspan=1 colspan=1>0.240</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=3>0.796 29.8   0.437   0.572</td><td rowspan=1 colspan=1>0.287</td><td rowspan=1 colspan=1>0.010</td><td rowspan=1 colspan=1>0.800</td><td rowspan=1 colspan=1>32.0</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Wan2.2</td><td rowspan=1 colspan=1>0.418</td><td rowspan=1 colspan=1>0.613</td><td rowspan=1 colspan=1>0.263</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=3>0.779 31.1   0.481   0.621</td><td rowspan=1 colspan=1>0.310</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.774</td><td rowspan=1 colspan=1>33.6</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Kling-V3</td><td rowspan=1 colspan=1>0.747</td><td rowspan=1 colspan=1>0.807</td><td rowspan=1 colspan=1>0.703</td><td rowspan=1 colspan=1>0.654</td><td rowspan=1 colspan=1>0.800 72.0</td><td rowspan=1 colspan=2>0.688   0.751</td><td rowspan=1 colspan=1>0.805</td><td rowspan=1 colspan=1>0.877</td><td rowspan=1 colspan=1>0.801</td><td rowspan=1 colspan=1>80.1</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Seedance 2.0</td><td rowspan=1 colspan=1>0.668</td><td rowspan=1 colspan=1>0.744</td><td rowspan=1 colspan=1>0.738</td><td rowspan=1 colspan=1>0.775</td><td rowspan=1 colspan=1>0.792 74.5</td><td rowspan=1 colspan=2>0.677   0.755</td><td rowspan=1 colspan=1>0.817</td><td rowspan=1 colspan=1>0.913</td><td rowspan=1 colspan=1>0.800</td><td rowspan=1 colspan=1>81.4</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Wan2.7</td><td rowspan=1 colspan=1>0.699</td><td rowspan=1 colspan=1>0.764</td><td rowspan=1 colspan=1>0.705</td><td rowspan=1 colspan=1>0.754</td><td rowspan=1 colspan=1>0.789 73.6</td><td rowspan=1 colspan=2>0.716   0.771</td><td rowspan=1 colspan=1>0.810</td><td rowspan=1 colspan=1>0.915</td><td rowspan=1 colspan=1>0.800</td><td rowspan=1 colspan=1>82.1</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Grok Imagine</td><td rowspan=1 colspan=1>0.652</td><td rowspan=1 colspan=1>0.693</td><td rowspan=1 colspan=1>0.344</td><td rowspan=1 colspan=1>0.064</td><td rowspan=1 colspan=1>0.795 40.4</td><td rowspan=1 colspan=2>0.617   0.645</td><td rowspan=1 colspan=1>0.328</td><td rowspan=1 colspan=1>0.014</td><td rowspan=1 colspan=1>0.805</td><td rowspan=1 colspan=1>37.2</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Veo 3.1</td><td rowspan=1 colspan=1>0.731</td><td rowspan=1 colspan=1>0.825</td><td rowspan=1 colspan=1>0.456</td><td rowspan=1 colspan=1>0.010</td><td rowspan=1 colspan=3>0.784 45.2   0.706   0.793</td><td rowspan=1 colspan=1>0.582</td><td rowspan=1 colspan=1>0.138</td><td rowspan=1 colspan=1>0.794</td><td rowspan=1 colspan=1>52.0</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Hunyuan-1.5</td><td rowspan=1 colspan=2>0.517   0.503</td><td rowspan=1 colspan=1>0.182</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.807 28.8</td><td rowspan=1 colspan=2>0.502   0.572</td><td rowspan=1 colspan=1>0.157</td><td rowspan=1 colspan=1>0.052</td><td rowspan=1 colspan=1>0.808</td><td rowspan=1 colspan=1>30.5</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=4>LTX-2.3        0.574   0.600</td><td rowspan=1 colspan=1>0.284</td><td rowspan=1 colspan=1>0.007</td><td rowspan=1 colspan=1>0.766 34.0</td><td rowspan=1 colspan=2>0.572   0.592</td><td rowspan=1 colspan=1>0.355</td><td rowspan=1 colspan=1>0.076</td><td rowspan=1 colspan=1>0.776</td><td rowspan=1 colspan=1>38.2</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Mitty-EPIC</td><td rowspan=1 colspan=2>0.615   0.709</td><td rowspan=1 colspan=1>0.601</td><td rowspan=1 colspan=1>0.557</td><td rowspan=1 colspan=1>0.740 62.0</td><td rowspan=1 colspan=2>0.632   0.721</td><td rowspan=1 colspan=1>0.558</td><td rowspan=1 colspan=1>0.318</td><td rowspan=1 colspan=1>0.740</td><td rowspan=1 colspan=1>54.0</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>SkyReels-V3</td><td rowspan=1 colspan=2>0.636   0.725</td><td rowspan=1 colspan=1>0.272</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.785 36.4</td><td rowspan=1 colspan=2>0.631   0.740</td><td rowspan=1 colspan=1>0.338</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.787</td><td rowspan=1 colspan=1>38.6</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>LongCat</td><td rowspan=1 colspan=2>0.662   0.671</td><td rowspan=1 colspan=1>0.278</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.790 36.2</td><td rowspan=1 colspan=2>0.559   0.654</td><td rowspan=1 colspan=1>0.299</td><td rowspan=1 colspan=1>0.022</td><td rowspan=1 colspan=1>0.791</td><td rowspan=1 colspan=1>35.7</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>Wan2.2        0.679   0.746</td><td rowspan=1 colspan=1>0.319</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=3>0.772 38.7   0.630   0.716</td><td rowspan=1 colspan=1>0.314</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.771</td><td rowspan=1 colspan=1>37.3</td></tr></table>

Table S10: Main-evaluation metric breakdown by task family for all 11 evaluated models. Parallel-Jaw Gripper and Dexterous Hand targets are reported separately. Video Quality is the task-family mean of M5 and contributes 0.10 to H2RCore. Overall values in Table 2 are computed from the unrounded per-video scores.

<table><tr><td>Metric</td><td>Evidence</td><td>Computation</td></tr><tr><td>M1: Goal</td><td>predicates.</td><td>25 generated frames and final-stateWeighted mean of normalized 0–4 predicate scores.</td></tr><tr><td>M2: Action</td><td>events.</td><td>25 generated frames and required Weighted mean of normalized event-completion scores.</td></tr><tr><td>M3: Contact</td><td>frames, and the contact specifica- tion.</td><td>25 source frames, 25 generated Mean of applicable contact dimensions; zero when source grounding fails.</td></tr><tr><td>M4: Embodiment</td><td>specification.</td><td>25 generated frames and the target Weighted mean of five embodiment dimensions; zero on a hard failure.</td></tr><tr><td>M5: Quality</td><td>Decoded generated video.</td><td>Mean of MUSIQ, CLIP aesthetic, temporal-stability, and AMT-S scores.</td></tr></table>

Table S11: Metric implementation summary. Detailed definitions, evidence rules, diagnostic components, and per-video formulas are given in Section B.6.

# Generation Prompts and Structured Annotation Input

## Video-Conditioned Generation

Input: complete source demonstration video.   
[VIDEO EDIT]   
Generate a 5-second edit from the reference clip. Keep the source camera, scene, lighting,   
background, object identities, layout, required actions, contact timing, and visible final state. Replace visible human hands, wrists, sleeves, and forearms with continuous robotic arms ending in five-finger dexterous hands. Objects may move only through visible robotic contact, stable support, or physically plausible pushing. Avoid scene or object substitution, residual human hands,   
invented actions, floating objects, missing   
contact, unstable final states, and warped robot geometry.

## Embodiment-Specific Replacement

Parallel-Jaw Gripper: Replace visible human manipulators with continuous robotic arms ending in rigid two-jaw parallel grippers. The jaws open and close laterally. Do not show human skin, gloves, soft or dexterous fingers, extra fingers, or claw grippers.

Dexterous Hand: Replace visible human manipulators with continuous robotic arms ending in five-finger dexterous hands, each with one opposable thumb and four articulated mechanical fingers. Do not show human skin, gloves, parallel-jaw or claw grippers, missing fingers, or extra fingers.

## M3 Judge Prompt

Role: strict visual evaluator for Functional   
Contact Transfer.   
Evidence: source frames 1-25 and generated frames 1-25, numbered independently.   
Judge only visible evidence. Do not infer occluded contact or unseen state changes, and use the case specification as a checklist rather than evidence. First apply the source-grounding gate: fail only for substantial scene or task-entity substitution. If grounding passes, score contact-region   
transfer, contact establishment, manipulation-mode transfer, temporally supported object response, and embodiment-compatible contact strategy on the 0-4 rubric. The robot need not copy the human pose or trajectory. Do not score final-goal completion, action coverage, generic robot appearance, or video quality. Return one valid JSON object with concise evidence-frame references and reasons.

## Frame-Conditioned Generation

Use the input frames as visual boundary conditions for the same manipulation. Preserve the source camera, scene, lighting, background, object identities, layout, required actions, and visible final state. Replace visible human hands, wrists, sleeves, and forearms with continuous robotic arms ending in five-finger dexterous hands. Maintain visible robot-object contact and physically plausible object motion. Avoid residual human hands, scene changes, invented actions, floating objects, missing contact, unstable final states, and warped robot geometry.

## Structured Source-Video Annotation

Use only visible evidence and record uncertainty instead of inferring occluded contact or unseen state changes. Identify the task, scene, initial and final states, manipulated entities, and task family. For each required event, record its temporal span, hand roles, contact mode and region, state transition, success evidence, and uncertainty. Select keyframes for the initial state, first contact, core transition, and final state. Return valid JSON containing transfer requirements, permissible adaptations, and invalid outcomes for both target embodiments.

## Structured Judge Output

Each applicable dimension contains {"applicable": true, "score\_0\_to\_4": 0, source/generated evidence frames, and a concise reason}.

Figure S5: Prompt formats used for generation, source-video annotation, and MLLM judging. The first row contrasts video- and frame-conditioned generation instructions; the second shows the embodiment clause and evaluation-only annotation input; and the third gives an abridged M3 judge prompt with its structured output.