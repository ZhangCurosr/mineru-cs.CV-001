# Embodied Multimodal Grounding for Open-Vocabulary Mobile Manipulation via Semantic 3D Gaussian Splating

Huosen Ou The Hong Kong University of Science and Technology (Guangzhou) Guangzhou, Guangdong, China hou061@connect.hkust-gz.edu.cn

Dongni Song   
The Hong Kong University of Science and Technology (Guangzhou) Guangzhou, Guangdong, China   
dsong549@connect.hkust-gz.edu.cn   
Tao Zhou ✉   
Midea Group   
Foshan, Guangdong, China   
zhoutao42@midea.com Yuncong Wang   
The Hong Kong University of Science and Technology (Guangzhou) Guangzhou, Guangdong, China   
ywang321@connect.hkust-gz.edu.cn   
Yiding Ji<sup>∗✉</sup>   
The Hong Kong University of Science   
and Technology (Guangzhou)   
Guangzhou, Guangdong, China   
The Hong Kong University of Science   
and Technology   
Hong Kong, China   
jiyiding@hkust-gz.edu.cn

## Abstract

Embodied mobile manipulation requires language, visual observations, three-dimensional scene structure, and action feasibility to be aligned before execution. We study open-vocabulary target grounding with few-shot manipulation in local household workspaces and present an embodied multimodal grounding framework that integrates active multi-view Semantic 3D Gaussian Splatting (Semantic-3DGS), reachability-aware base positioning, and a difusion-based vision-language-action policy. A task-driven local Semantic-3DGS serves as a shared interface across active sensing, language-conditioned 3D localization, obstacle-aware scene reasoning, base preparation, and semantic conditioning of the action model. To preserve pretrained action priors, the 3D semantic cues are injected only into the late action-expert blocks. In expanded 50- trial real-robot evaluations against representative vision-languageaction (VLA) approaches, the full system achieves 60% long-horizon success compared with 40% for PointVLA and 28% for DexVLA, and reaches 74% success in heavily cluttered manipulation compared with 52% for the single-view variant and 46% for PointVLA. It also maintains 75% success under a 75 cm height shift and eliminates photo-induced false grasps. These results indicate that explicit, refreshable 3D semantic grounding can improve robustness under clutter, occlusion, viewpoint variation, and embodiment constraints.

## CCS Concepts

• Computing methodologies → Vision for robotics; Reconstruction; • Computer systems organization → Robotic autonomy.

## Keywords

embodied multimedia, multimodal grounding, multimodal fusion, multimedia and language, mobile manipulation, semantic 3D Gaussian splatting, vision-language-action

## ACM Reference Format:

Huosen Ou, Dongni Song, Yuncong Wang, Tao Zhou, and Yiding Ji. 2026. Embodied Multimodal Grounding for Open-Vocabulary Mobile Manipulation via Semantic 3D Gaussian Splatting. In Proceedings of the 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 9 pages. https: //doi.org/10.1145/3767308.3836325

## 1 Introduction

Recent multimodal models have substantially advanced languageguided perception and action, yet robust embodied execution in real environments remains dificult. In household mobile manipulation scenarios, a robot must identify the object referred to by a language instruction, ground it in three-dimensional space, prepare a feasible body configuration, and execute under clutter, occlusion, viewpoint variation, and embodiment constraints. This makes mobile manipulation a representative problem of embodied multimedia understanding, where language, imagery, geometry, and robot state must be fused into grounded action.

This work focuses on two practical failure modes. First, many vision-language-action (VLA) systems still rely heavily on 2D appearance cues, making target grounding brittle under partial occlusion, clutter, or appearance-only distractors such as a tablet displaying a photo-realistic target. Second, even when the target is correctly localized, manipulation may still fail if the mobile base is poorly positioned relative to the arm workspace, particularly in long-horizon tasks or when the target height changes.

We therefore propose an embodied multimodal grounding framework centered on a compact and refreshable Semantic-3DGS to address the challenge of cross-modal spatial alignment. The representation is not used only as a static 3D container: the same targetconditioned local field serves as a shared interface for active-view scoring, Gaussian-level language localization, obstacle-aware rendering, target-relative pose extraction, and semantic conditioning of the manipulation policy. We initialize geometry from VGGT [18], distill CLIP/DINO semantics with mask-aware regularization inspired by Feature Splatting [17], compute a reachability-aware base stance, and condition a pretrained difusion VLA [20] through late-block semantic adapters. This design aims to add explicit 3D grounding while preserving pretrained visuomotor priors.

Our scope is open-vocabulary target grounding with few-shot manipulation, rather than zero-shot acquisition of arbitrary robot skills. Test objects are held-out instances, while the embodiment-specific manipulation policy is adapted using 10 real demonstrations per task. The local Semantic-3DGS is constructed before manipulation and refreshed when grounding fails; it is not continuously optimized inside the low-level control loop. Our Late-Block Semantic Injection is a multimodal fusion mechanism bridging heterogeneous perception and reactive control. Instead of early injection, which degrades pretrained priors, we distill DINO/CLIP features into a local Semantic-3DGS and inject dense 3D semantics only into final difusion experts, preserving action priors while adding 3D grounding. This preserves established continuous action priors while adding robust spatial grounding. Hence, our framework is not a loose combination of existing modules: the same object-centric Semantic-3DGS is used as a shared interface for active view scoring, Gaussian-level language localization, obstacle-aware rendering, PCA semantic tokenization, target-relative pose conditioning, and zero-initialized late-block adaptation.

Contributions. Our contributions are threefold. First, we formulate local-area mobile manipulation as an embodied multimodal grounding problem and construct a target-driven Semantic-3DGS from four actively acquired wrist-camera images. Second, we use this local field as a shared perception-to-action interface for active sensing, language grounding, obstacle-aware reasoning, targetrelative pose extraction, and late-block conditioning of a difusion action expert, preserving earlier pretrained action priors. Third, we couple explicit 3D grounding with reachability-aware stance preparation and validate the resulting system in real-robot few-shot manipulation under long-horizon execution, height shifts, photo deception, and heavy clutter, including expanded 50-trial studies with confidence intervals and runtime profiling.

## 2 Related Work

## 2.1 Embodied Mobile Manipulation

Embodied mobile manipulation of robots couples locomotion, perception, and contact-rich actions. Previous works have investigated quadruped manipulation, whole-body coordination, and long horizon loco-manipulation [1, 3, 5, 6, 9–11, 13–16, 22–25]. These studies show that legged platforms extend manipulation beyond fixed tabletop settings by enabling flexible approach, posture adaptation, and operation in cluttered spaces.

Compared with static-arm manipulation, however, mobile manipulation is more sensitive to embodiment constraints such as base placement, body height, arm workspace limits, self-occlusion, and nearby obstacles. In practice, failure often stems from weak coordination between perception and action preparation rather than from the arm policy alone. Our work follows this systems perspective and connects task-conditioned 3D grounding, base positioning, and manipulation through a shared local semantic representation.

## 2.2 3D Grounding and Multimodal Fusion

A key multimedia challenge in embodied interaction is aligning language, visual observations, and geometry into a representation useful for action. Recent works have explored point-cloudconditioned VLA models [8], open-vocabulary mobile manipulation [21], language-aware 3D Gaussian representations [17, 26] and 4D Gaussian splatting for dynamic scene modeling [12]. Compared with purely 2D grounding, 3D representations provide more stable cross-view localization and better support reasoning about target pose and nearby obstacles, especially under clutter and occlusion.

Our work aligns with this technical direction, but emphasizes a task-driven local representation rather than dense full-scene optimization or persistent global mapping. Under our settings, a refreshable Semantic-3DGS is then constructed from only four actively selected wrist-camera views and the resulting field is reused across target localization, obstacle-aware reasoning, reachability-aware stance preparation, and downstream VLA conditioning. In this way, the 3D representation serves as a common multimodal interface instead of an isolated perception output.

## 2.3 Vision-Language-Action Policies

VLA models and difusion-based robotic manipulation policies provide scalable interfaces for embodied control [2, 4, 7, 19, 20, 27]. Large pretrained backbones ofer transferable visuomotor priors, while difusion-based action experts improve action generation and temporal consistency. These advances have substantially improved language-conditioned manipulation.

However, when explicit 3D scene cues are absent, a policy must jointly infer object identity, spatial relation, viewpoint suficiency, and action feasibility from limited observations. This makes end-toend policies vulnerable to clutter, occlusion, viewpoint variation, and appearance-only distractors. Our method is complementary to this line of work: we retain a pretrained difusion-based VLA backbone and inject explicit Semantic-3DGS cues only into late action-expert blocks. This design preserves earlier pretrained action priors while supplying target-centered 3D semantics and obstacleaware geometry for execution.

## 3 Method

## 3.1 System Overview and Compute Split

The full system is shown in Figure 1. After receiving a language instruction �, the robot performs: (1) active local multi-view observation, (2) Semantic-3DGS construction and open-vocabulary 3D localization, (3) reachability-aware base repositioning, and (4) Semantic-3DGS-conditioned VLA manipulation.

The platform of this work is a Unitree Go2 Edu quadruped equipped with standing and crouching modes, a Unitree 4D L1 LiDAR, an onboard Jetson Orin NX, and an Alicia-D 6-DoF arm with an RGB camera near the gripper. Arm joint targets, including the gripper, are streamed at 30 Hz through ROS. Semantic-3DGS perception and VLA inference run of-board on an RTX 4090 workstation, while low-level robot control and the base posture policy run onboard. The current system targets quasi-static household manipulation rather than fast dynamic interaction.

![](images/5f66299d47fa6f3aeb7021e75dbf3f0c65cfcacc594eb0e46af9898841a4f08c.jpg)

Figure 1: Overview of the proposed system. The robot patrols by waypoint navigation, switches to task mode after receiving instruction $l ,$ actively acquires four local wrist-camera views, constructs a local Semantic-3DGS, estimates the target pose $^ { W } T _ { \mathrm { o b j } } ,$ repositions the base to a feasible stance $( x ^ { \star } , y ^ { \star } , \psi ^ { \star } , h ^ { \star } )$ , and executes a semantics-conditioned VLA manipulation policy.  
![](images/119ca437a666aaf4111c811f61368ada7c0d3232c78a3aac116afdfda867fc35.jpg)  
Figure 2: Robot platform: a Unitree Go2 Edu quadruped with a 6-DoF arm and an arm-mounted RGB camera for local perception and manipulation.

## 3.2 Representation and Notation

Let $\mathcal { F } _ { W }$ and $\mathcal { F } _ { B }$ denote the world and base frames, and let $^ { W } T _ { B , t } \in$ ��(3) be the current base pose. A multi-view image bufer is ${ \boldsymbol { \mathcal { T } } } =$

$\{ I ^ { ( i ) } \} _ { i = 1 } ^ { N } .$ , with $N = 4$ for the full system and $N = 1$ for Ours Single-View. The local Semantic-3DGS i

$$
\boldsymbol { \mathcal { G } } = \left\{ \boldsymbol { g } _ { k } \right\} _ { k = 1 } ^ { K } , \qquad \boldsymbol { g } _ { k } = ( \mu _ { k } , \boldsymbol { \Sigma } _ { k } , c _ { k } , \alpha _ { k } , f _ { k } ^ { C } , f _ { k } ^ { D } ) ,\tag{1}
$$

where $\mu _ { k } , \Sigma _ { k } , c _ { k } , \alpha _ { k }$ denote Gaussian geometry, color, and opacity, while $f _ { k } ^ { C }$ and $f _ { k } ^ { D }$ are CLIP-aligned and DINO semantic features.

## 3.3 Active Multi-view Semantic-3DGS

3.3.1 Active View Acquisition. From the first wrist-camera image $I ^ { ( 1 ) }$ , we extract a target phrase $q ( l )$ , compute a language relevance map, and combine SAM masks with coarse VGGT geometry to obtain a rough 3D target support. Candidate wrist-camera poses $\mathcal { N } _ { \mathrm { c a n d } }$ are scored by

$$
\begin{array} { r } { J ( v ) = \lambda _ { \mathrm { c o v } } C _ { \mathrm { s e m } } ( v ) + \lambda _ { \mathrm { p a r } } C _ { \mathrm { p a r } } ( v ) - \lambda _ { \mathrm { m o v e } } C _ { \mathrm { m o v e } } ( v ) , } \end{array}\tag{2}
$$

where $C _ { \mathrm { { s e m } } }$ estimates the expected semantic coverage of the coarse target support, $C _ { \mathrm { p a r } }$ rewards complementary viewpoints, and $C _ { \mathrm { m o v e } }$ discourages unnecessary wrist motion. The terms are normalized before combination, and only IK-feasible candidates are retained. The next view is

$$
v _ { n + 1 } = \underset { v \in \mathcal { V } _ { \mathrm { c a n d } } } { \arg \operatorname* { m a x } } J ( v ) .\tag{3}
$$

## Active Multi-view Semantic-3DGS for Language Grounding

![](images/c9ab8647851f4112716b984f9be0357ff9c9c7252fcbc0ea9be63410a52380e1.jpg)  
Figure 3: Active multi-view Semantic-3DGS for language grounding. Target-driven view acquisition builds a compact local semantic 3D representation from four wrist-camera images.

The robot base remains fixed during sensing; a standard IK controller moves the wrist camera until four images are collected. No VLA policy is used for view planning.

3.3.2 Geometry Initialization and Semantic Distillation. Given $\boldsymbol { \mathcal { I } } , \mathrm { a }$ pretrained VGGT model [18] predicts camera parameters and dense geometry,

$$
\{ ( \boldsymbol { \hat { g } } ^ { ( i ) } , \boldsymbol { \hat { D } } ^ { ( i ) } , \boldsymbol { \hat { P } } ^ { ( i ) } ) \} _ { i = 1 } ^ { N } = \Phi _ { \mathrm { V G G T } } ( \boldsymbol { Z } ) ,\tag{4}
$$

which initializes the local Gaussian field. For each view, we extract CLIP and DINOv2 feature maps and SAM masks. Mask-aware average pooling refines the CLIP features, and rendered Gaussian features are optimized by cosine alignment:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { { f e a t } } } = \displaystyle \sum _ { i , u } \left( 1 - \cos ( \hat { F } _ { C } ^ { ( i ) } ( u ) , \tilde { F } _ { C } ^ { ( i ) } ( u ) ) \right) } \\ & { \quad \quad \quad \quad + \left. \lambda _ { D } \displaystyle \sum _ { i , u } \left( 1 - \cos ( \hat { F } _ { D } ^ { ( i ) } ( u ) , F _ { D } ^ { ( i ) } ( u ) ) \right) , \right. } \end{array}\tag{5}
$$

with $\lambda _ { D } = 0 . 1$

3.3.3 Open-Vocabulary 3D Localization. The target phrase is encoded by a frozen CLIP text encoder to obtain $e ^ { + }$ . With generic negative prompts $\{ e _ { j } ^ { - } \} _ { j = 1 } ^ { m }$ , each Gaussian receives a language relevance score

$$
s _ { k } = \mathrm { s o f t m a x } \Big ( \tau [ \cos ( f _ { k } ^ { C } , e ^ { + } ) , \cos ( f _ { k } ^ { C } , e _ { 1 } ^ { - } ) , . . . , \cos ( f _ { k } ^ { C } , e _ { m } ^ { - } ) ] \Big ) _ { 1 }\tag{6}
$$

For the selected Gaussian support $\mathcal { K } = \{ k \mid s _ { k } > \delta \}$ , the object position is estimated by

$$
\mathcal { P } _ { \mathrm { o b j } } ^ { W } = \frac { \sum _ { k \in \mathcal { K } } w _ { k } \mu _ { k } } { \sum _ { k \in \mathcal { K } } w _ { k } } , \qquad w _ { k } = \alpha _ { k } \operatorname { t r a c e } ( \Sigma _ { k } ) ^ { - 1 } .\tag{7}
$$

A stable 6D pose $w _ { T _ { \mathrm { o b i } } }$ is estimated from the selected support using PCA, with optional ICP refinement when a template is available. The explicit 3D support reduces dependence on a single appearance cue and exposes both target and nearby obstacle geometry.

## 3.4 Reachability-aware Base Posture Control

After localization, the object position is transformed to the base frame:

$$
\boldsymbol { p } _ { \mathrm { o b j } } ^ { B } = ( ^ { W } T _ { B , t } ) ^ { - 1 } \boldsymbol { p } _ { \mathrm { o b j } } ^ { W } = [ x _ { \mathrm { o b j } } , y _ { \mathrm { o b j } } , z _ { \mathrm { o b j } } ] ^ { \top } .\tag{8}
$$

We define a pre-manipulation stance

$$
x ^ { \star } = x _ { \mathrm { o b j } } - d _ { x } , ~ y ^ { \star } = y _ { \mathrm { o b j } } - \mathrm { s i g n } ( y _ { \mathrm { o b j } } ) d _ { y } , ~ \psi ^ { \star } = \mathrm { a t a n 2 } ( y _ { \mathrm { o b j } } , x _ { \mathrm { o b j } } ) ,\tag{9}
$$

with $d _ { x } = 0 . 3 5$ m and $d _ { y } = 0 . 2 0$ m, and select the height mode by

$$
h ^ { \star } = \left\{ \begin{array} { l l } { h ^ { \mathrm { c r o u c h } } , } & { z _ { \mathrm { o b j } } < 0 . 3 0 \mathrm { m } , } \\ { h ^ { \mathrm { s t a n d } } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{10}
$$

A proximal policy optimization (PPO) policy outputs leg-joint residuals and the stand/crouch switch. The controller is trained in Isaac Lab built on NVIDIA Isaac Sim with domain randomization; only the base posture policy is trained in the simulations.

## 3.5 Semantic-3DGS-Conditioned VLA Manipulation Policy

We build on a DexVLA-style policy with a Qwen2-VL backbone [19] and ScaleDP action expert [27]. From instruction � and G, we aggregate: (1) a language-conditioned target heatmap �<sub>�</sub>, (2) an obstacleaware occupancy cue $O _ { t }$ , (3) a three-channel PCA rendering $P _ { t }$ of the distilled per-Gaussian semantic field, and (4) a target-relative pose vector $r _ { t } = [  { p _ { \mathrm { o b j } } } ^ { B } , h _ { t } ]$ ∈ R<sup>4</sup>. The visual cues and RGB observation are encoded as

$$
z _ { t } ^ { \mathrm { i m g } } = E _ { \mathrm { i m g } } ( \mathrm { c o n c a t } ( I _ { t } , H _ { t } , O _ { t } , P _ { t } ) ) \in \mathbb { R } ^ { 1 2 8 } ,\tag{11}
$$

and the pose cue as $z _ { t } ^ { \mathrm { p o s e } } = E _ { \mathrm { p o s e } } ( r _ { t } ) \in \mathbb { R } ^ { 1 2 8 }$ . Their sum $\boldsymbol { z } _ { t } ^ { \mathrm { s e m } }$ is injected only into the final five difusion blocks:

$$
\mathcal { B } = \{ L - 4 , L - 3 , L - 2 , L - 1 , L \} ,\tag{12}
$$

$$
h _ { \ell } \gets h _ { \ell } + A _ { \ell } ( \mathrm { P r o j } ( z _ { t } ^ { \mathrm { s e m } } ) ) , \qquad \ell \in \mathcal { B } ,\tag{13}
$$

where Proj is zero-initialized and �<sub>ℓ</sub> is a lightweight MLP adapter. The VLM backbone and all pretrained difusion blocks remain frozen; only the semantic encoder, projection layers, late-block adapters, and embodiment-specific action head are trained. We use 10 real demonstrations per task, an action chunk of 15 steps, and two recent observation frames.

## 4 Experiments

## 4.1 Model Training, Deployment, and Evaluation Protocol

Only the reachability-aware base posture controller is trained in Isaac Lab/Isaac Sim; the manipulation policy is trained ofline from real demonstrations. Each trajectory records wrist RGB, robot pro prioception, and arm joint targets at 30 Hz. During deployment, the full system acquires a four-view RGB bufer before manipulation, whereas Ours Single-View uses only the initial image.

Unless otherwise noted, each setting uses 30 real-robot trials with randomized initial object poses. The long-horizon and cluttered banana-to-bowl evaluations are expanded to 50 trials per method. A trial is successful only when the complete instruction is finished within the time budget (60 s for local multi-task manipulation and 180 s for the long-horizon task) without unsafe behavior. For the 50- trial studies, we additionally report success counts, time standard deviations, and 95% Wilson confidence intervals.

Fair baseline adaptation. All baselines use the same robot embodiment and the same 10 teleoperated demonstrations per task. We also keep the train/test split, wrist-camera setup, action interface, navigation waypoints, base-control assistance, initial scene distributions, and evaluation budgets fixed. Each method retains its native perception representation. Thus, the comparisons mainly reflect diferences in grounding and manipulation representations rather than diferences in demonstrations, navigation support, or stance preparation.

## 4.2 Tasks

Next, we evaluate five groups of real-robot tasks. Few-shot multitask manipulation contains bottle-to-basket, banana-to-book, and ordered toy/banana/bottle placement into a white ceramic bowl. Test objects are unseen instances in task-specific fine-tuning, while semantically related object families may appear in the 10 demonstration trajectories. Long-horizon manipulation requires opening a drawer, retrieving a banana, closing the drawer, navigating to a black chair, and placing the banana. Height adaptability varies target platform height by 30, 60, and 75 cm. Photo deception replaces a real banana with a photo-realistic banana displayed on a tablet. Cluttered banana-to-bowl manipulation places the target among multiple physical distractors and frequent occlusions.

Table 1: Long-horizon performance over 50 trials per method. CI denotes a 95% Wilson interval.
<table><tr><td>Method</td><td>Success</td><td>95% CI</td><td>Avg. time (s)</td></tr><tr><td>DexVLA</td><td>14/50 (28%)</td><td>[17.5, 41.7]</td><td> $1 7 8 . 6 \pm 1 8 . 4$ </td></tr><tr><td>PointVLA</td><td>20/50 (40%)</td><td>[27.6, 53.8]</td><td> $1 6 1 . 8 \pm 1 7 . 6$ </td></tr><tr><td>Ours w/o Base-RL</td><td>11/50 (22%)</td><td>[12.8, 35.2]</td><td> $2 0 4 . 1 \pm 2 1 . 3$ </td></tr><tr><td>Ours (full)</td><td>30/50 (60%)</td><td>[46.2, 72.4]</td><td> $1 4 0 . 7 \pm 1 5 . 9$ </td></tr></table>

Table 2: Height adaptation across platform height ofsets (cm).
<table><tr><td>Method</td><td>0→30</td><td> $0 \to 6 0$ </td><td>0→75</td></tr><tr><td>DexVLA</td><td>48%</td><td>33%</td><td>23%</td></tr><tr><td>PointVLA</td><td>58%</td><td>46%</td><td>35%</td></tr><tr><td>Ours w/o Base-RL</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td>Ours (full)</td><td>80%</td><td>78%</td><td>75%</td></tr></table>

## 4.3 Results

4.3.1 Few-shot Multi-task Manipulation. Figures 5 and 8 summarize the few-shot task settings and results. The full model reaches the highest average success rate (81.7%), compared with 64.0% for PointVLA and 37.7% for DexVLA. The advantage becomes larger in cluttered and multi-step settings, supporting the use of explicit 3D semantic grounding beyond easy single-object cases.

4.3.2 Long-horizon Manipulation. Figure 7 illustrates the multistage execution, and Table 1 reports the expanded 50-trial evaluation. The complete system completes 30/50 trials (60%) compared with 20/50 (40%) for PointVLA and 14/50 (28%) for DexVLA. Removing the Base-RL module reduces the success rate to 22%, validating that mere correct grounding is insuficient when the stance is not prepared for the arm workspace.

4.3.3 Height Adaptation. Figure 6 shows the height-shift settings, while Table 2 reports the corresponding results. The full system remains between 75% and 80% success, while the variant without Base-RL fails when the initial stance is retained.

4.3.4 Photo Deception. Table 3 replaces the real target with a photo-realistic distractor. RGB-only DexVLA and the single-view ablation exhibit high false-grasp rates, while PointVLA and the full method reject the flat-screen distractor. The full model also improves real-object success, indicating more stable localization beyond distractor rejection.

![](images/f162d7f4d1abd2b38516c6f5727bc616a62739f1f07c49f7da1b37213d2700f7.jpg)  
Figure 4: Semantic-3DGS-conditioned VLA manipulation. A lightweight semantic injector conditions the final five frozen action-expert blocks through trainable adapters while preserving pretrained action priors.

![](images/900e9d2a2503a614a69150a28d7caa4baef011de0988e49c528368b175ea0ae8.jpg)  
Figure 5: Few-shot multi-task manipulation scenarios. Test objects are unseen instances in task-specific fine-tuning, while semantically related object families may appear in the few-shot demonstrations.

4.3.5 CluteredandOccludedManipulation. The clutter benchmark stresses both target visibility and nearby obstacle geometry. Specifically, Table 4 illustrates that active multi-view Semantic-3DGS raises success from 52% for the single-view variant to 74%, while improving the collision-free rate from 70% to 88% and reducing false grasps from 18% to 6%. Relative to PointVLA, the full system improves success by 28 percentage points.

Component ablations. Separate 30-trial clutter ablations isolate the main representation and conditioning choices. A VGGT pointmap variant reaches 58% success. Removing CLIP/DINO semantic features yields 60%, and removing the obstacle-occupancy cue yields 65%. All-block semantic injection reaches 68%, compared with 74% for the full system. Collision-free execution and false-grasp rate degrade in the same ablations, indicating that the gains do not arise from active sensing or base repositioning alone.

![](images/bada6d8def2e9f88f3f51ce41d55a43b8e6b7ddaf76789b21e0f59171b3adaee.jpg)

Figure 6: Height-adaptation settings. Targets are placed at diferent platform heights to test stance selection and reachabilityaware manipulation.  
![](images/bea512b706aadd1cf6db2603a1b45a3bcf04ca8862c5e6fa61151af7c6a9ff77.jpg)  
Figure 7: Long-horizon manipulation sequence. The robot opens the top drawer, retrieves the banana, closes the drawer, navigates to the target area, and places the banana on the black chair.

![](images/d59d820f1a34b93725b0bffb7a4b543721da887d2ecbeec39021d2e8ae8217bf.jpg)  
Figure 8: Success rates on the few-shot multi-task manipulation benchmark. (1) Place the banana on the book. (2) Place the bottle into the basket. (3) Place the stufed toy, banana, and bottle into the bowl in the specified order.

![](images/1ad13fdb8ee7f60955f35a1f57fcf9f0aabd84989dd8ad760bbc476985106781.jpg)  
Figure 9: Photo-deception setup. The robot should manipulate the real object while rejecting a photo-realistic distractor on a flat screen.

Table 3: Robustness to photo deception.
<table><tr><td>Method</td><td>Real success ↑ Photo false-grasp ↓</td><td></td></tr><tr><td>DexVLA</td><td>78%</td><td>76%</td></tr><tr><td>PointVLA</td><td>80%</td><td>0%</td></tr><tr><td>Ours Single-View</td><td>76%</td><td>70%</td></tr><tr><td>Ours (full)</td><td>88%</td><td>0%</td></tr></table>

Table 4: Cluttered banana-to-bowl benchmark over 50 trials per method. CI denotes a 95% Wilson interval.
<table><tr><td>Method</td><td>Success</td><td>95% CI</td><td>Collision-free</td><td>False grasp</td><td>Avg. time (s)</td></tr><tr><td>DexVLA</td><td>13/50 (26%)</td><td>[15.9, 39.6]</td><td>22/50 (44%)</td><td>16/50 (32%)</td><td> $2 7 . 9 \pm 3 . 1$ </td></tr><tr><td>PointVLA</td><td>23/50 (46%)</td><td>[33.0, 59.6]</td><td>31/50 (62%)</td><td>10/50 (20%)</td><td> $3 0 . 7 \pm 3 . 4$ </td></tr><tr><td>Ours Single-View</td><td>26/50 (52%)</td><td>[38.5, 65.2]</td><td>35/50 (70%)</td><td>9/50 (18%)</td><td> $2 9 . 5 \pm 3 . 2$ </td></tr><tr><td>Ours (full)</td><td>37/50 (74%)</td><td>[60.4, 84.1]</td><td>44/50 (88%)</td><td>3/50 (6%)</td><td> $3 3 . 2 \pm 3 . 6$ </td></tr></table>

![](images/412d5479a40034662b0e372cac702194bf2c29c79b23ea65ad62b989d45de0d6.jpg)  
Figure 10: Cluttered banana-to-bowl benchmark with physical distractors and strong target occlusion.

Table 5: Late-block versus all-block semantic injection.
<table><tr><td>Variant</td><td>Avg. success</td><td>Chunk latency</td></tr><tr><td>All-block</td><td>75%</td><td>175 ms</td></tr><tr><td>Late-block (5)</td><td>82%</td><td>80 ms</td></tr></table>

4.3.6 Late-block Injection and Runtime. Late-block injection provides a better success/latency trade-of than modifying all actionexpert blocks (Table 5). Figure 11 further shows the block-sensitivity trend: earlier or broader intervention disrupts the pretrained action prior more strongly than conditioning only the late blocks. Runtime is reported at three levels: one-time grounding latency, per-actionchunk online latency, and independently measured full-task wallclock time. We keep these measurements separate: the one-time grounding costs and per-chunk online latencies are not arithmetically summed to estimate task duration. Full-task wall-clock time is measured independently from the start of active sensing to completion of placement and already includes repeated action-chunk inference and communication, physical arm/base motion, grasping, transfer, placement, and settling.

Active multi-view sensing adds about 3.7 s to the clutter-task wall-clock time (33.2 s versus 29.5 s for Single-View), while improving success by 22 points and collision-free execution by 18 points. This quantifies the robustness–latency trade-of of the up-front grounding stage rather than implying real-time 3D reconstruction inside the manipulation servo loop.

Scope and limitations. Our system targets quasi-static local household manipulation where a short active grounding stage is acceptable. Semantic-3DGS construction and VLA inference currently run on an of-board GPU, and the local representation is built before manipulation or refreshed after grounding failure rather than updated inside the low-level servo loop. We therefore do not claim fast dynamic interaction or zero-shot acquisition of arbitrary manipulation skills; the evaluated setting is open-vocabulary target grounding with few-shot embodiment-specific manipulation.

![](images/f2d719f55eeeb5ddf82a5cd952263b319e037be6e3476dceeff00bb761acc726.jpg)

![](images/b63a83854b66631ac7ae8ad8d5ed240912caa676452b0f8999d8e80db927fcd5.jpg)  
Figure 11: Block sensitivity analysis for semantic-3DGS injection: late-block conditioning provides a better success– eficiency trade-of than injecting semantic cues into earlier or broader portions of the action expert.

Table 6: Runtime profile. One-time stages execute before manipulation; online values are measured per action chunk.
<table><tr><td>Component</td><td>Mean latency</td></tr><tr><td>Four-view wrist motion + capture</td><td>16.00 s</td></tr><tr><td>VGGT pose/depth initialization</td><td>0.62 s</td></tr><tr><td>Semantic-3DGS feature update</td><td>1.21 s</td></tr><tr><td>Semantic rendering + localization</td><td>0.34 s</td></tr><tr><td>VLA action-chunk inference ROS/WiFi communication</td><td>0.08 s / chunk</td></tr><tr><td>Full clutter task (ours)</td><td>0.05 s / chunk  $3 3 . 2 \pm 3 . 6 \ s$ </td></tr></table>

## 5 Conclusion

We presented an embodied multimodal grounding framework for open-vocabulary target grounding with few-shot mobile manipu lation of robots. A refreshable Semantic-3DGS provides a shared interface across active sensing, language-conditioned localization, obstacle-aware geometry, reachability-aware stance preparation, and late-block VLA conditioning. Expanded real-robot evaluations show improved robustness under long-horizon execution, height variation, photo-realistic distractors, and heavy clutter. Future research will investigate lighter onboard representations, adaptive view planning, and broader unseen-object generalization.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grants 62303389 and 62373289. It was also supported by National Key Research and Development Program of China under Grant 2025YFB4713002. Additional support came from Guangdong Scientific Research Platform and Project Scheme under Grant 2024KTSCX039, Guangzhou-HKUST(GZ) Joint Fund ing Program under Grant 2024A03J0618, the Youth Talent Support Program ofGuangdong Provincial Association for Science and Technology under Grant SKXRC2025463 and Guangdong Provincial Key Lab of Integrated Communication, Sensing and Computation for Ubiquitous Internet of Things (grant number 2023B1212010007).

## References

[1] Philip Arm, Mayank Mittal, Hendrik Kolvenbach, and Marco Hutter. 2024. Pedip ulate: Enabling Manipulation Skills using a Quadruped Robot’s Leg. In 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 5717–5723. doi:10.1109/ICRA57147.2024.10611307

[2] Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. 2025. �<sub>0.5</sub>: a Vision-Language-Action Model with Open-World Generalization. In Proceedings of The 9th Conference on Robot Learning (Proceedings of Machine Learning Research, Vol. 305). PMLR, 17–40. https://proceedings.mlr.press/v305/ black25a.html

[3] Xuxin Cheng, Ashish Kumar, and Deepak Pathak. 2023. Legs as Manipulator: Pushing Quadrupedal Agility Beyond Locomotion. In 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 5106–5112. doi:10.1109/ ICRA48891.2023.10161470

[4] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burch fiel, Russ Tedrake, and Shuran Song. 2025. Difusion policy: Visuomotor policy learning via action difusion. The International Journal ofRobotics Research 44, 10-11 (2025), 1684–1704. doi:10.1177/02783649241273668

[5] Ioannis Dadiotis, Arturo Laurenzi, and Nikos Tsagarakis. 2023. Whole-Body MPC for Highly Redundant Legged Manipulators: Experimental Evaluation with a 37 DoF Dual-Arm Quadruped. In 2023 IEEE-RAS 22nd International Conference on Humanoid Robots (Humanoids). IEEE, 1–8. doi:10.1109/Humanoids57100.2023. 10375215

[6] Zhengmao He, Kun Lei, Yanjie Ze, Koushil Sreenath, Zhongyu Li, and Huazhe Xu. 2024. Learning Visual Quadrupedal Loco-Manipulation from Demonstrations. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 9102–9109. doi:10.1109/IROS58592.2024.10802742

[7] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P. Foster, Pannag R. Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. 2025. OpenVLA: An Open-Source Vision-Language-Action Model. In Proceedings ofThe 8th Conference on Robot Learning (Proceedings ofMachine Learning Research, Vol. 270). PMLR, 2679–2713. https://proceedings.mlr.press/v270/kim25c.html

[8] Chengmeng Li, Junjie Wen, Yaxin Peng, Yan Peng, and Yichen Zhu. 2026. PointVLA: Injecting the 3D World Into Vision-Language-Action Models. IEEE Robotics and Automation Letters 11, 3 (2026), 2506–2513. doi:10.1109/LRA.2026. 3653303

[9] Minghuan Liu, Zixuan Chen, Xuxin Cheng, Yandong Ji, Ri-Zhao Qiu, Ruihan Yang, and Xiaolong Wang. 2025. Visual Whole-Body Control for Legged Loco-Manipulation. In Proceedings of The 8th Conference on Robot Learning (Proceedings of Machine Learning Research, Vol. 270). PMLR, 234–257. https://proceedings.mlr. press/v270/liu25b.html

[10] Xin Liu, Bida Ma, Chenkun Qi, Yan Ding, Nuo Xu, Zhaxizhuoma, Guorong Zhang, Pengan Chen, Kehui Liu, Zhongjie Jia, Chuyue Guan, Yule Mo, Jiaqi Liu, Feng Gao, Jiangwei Zhong, Bin Zhao, and Xuelong Li. 2026. MLM: Learning Multi-Task Loco-Manipulation Whole-Body Control for Quadruped Robot With Arm. IEEE Robotics and Automation Letters 11, 1 (2026), 81–88. doi:10.1109/LRA.2025.3632087

[11] Hisayoshi Muramatsu, Keigo Kitagawa, Jun Watanabe, Yuika Yoshimoto, and Ryohei Hisashiki. 2025. A Mobile Quad-Arm Robot ARMS: Wheeled-Legged Tripedal Locomotion and Loco-Manipulation. Journal ofRobotics and Mechatronics 37, 2 (2025), 489–499. doi:10.20965/jrm.2025.p0489

[12] Huosen Ou and Yiding Ji. 2025. A Pose-Free Approach for 4D Gaussian Splatting to Reconstruct Dynamic Scenes. In 2025 11th International Conference on Control, Decision and Information Technologies (CoDIT), Vol. 1. IEEE, 2934–2939. doi:10. 1109/CoDIT66093.2025.11321396

[13] Yutao Ouyang, Jinhan Li, Yunfei Li, Zhongyu Li, Chao Yu, Koushil Sreenath, and Yi Wu. 2025. Long-Horizon Locomotion and Manipulation on a Quadrupedal Robot with Large Language Models. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 11157–11164. doi:10.1109/IROS60139. 2025.11246632

[14] Guoping Pan, Qingwei Ben, Zhecheng Yuan, Guangqi Jiang, Yandong Ji, Shoujie Li, Jiangmiao Pang, Houde Liu, and Huazhe Xu. 2025. RoboDuet: Learning a Cooperative Policy for Whole-Body Legged Loco-Manipulation. IEEE Robotics and Automation Letters 10, 5 (2025), 4564–4571. doi:10.1109/LRA.2025.3551230

[15] Tifanny Portela, Gabriel B. Margolis, Yandong Ji, and Pulkit Agrawal. 2024. Learning Force Control for Legged Manipulation. In 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 15366–15372. doi:10.1109/ICRA57147. 2024.10611066

[16] Yihao Qin, Yuanfei Wang, Hang Zhou, Peiran Liu, Hao Dong, and Yiding Ji. 2026. IPD: Boosting Sequential Policy with Imaginary Planning Distillation in Ofline Reinforcement Learning. In Proceedings ofthe 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS). IFAAMAS, 726–734. doi:10.65109/BELB5985

[17] Ri-Zhao Qiu, Ge Yang, Weijia Zeng, and Xiaolong Wang. 2025. Language-Driven Physics-Based Scene Synthesis and Editing via Feature Splatting. In Computer Vision – ECCV 2024 (Lecture Notes in Computer Science, Vol. 15099). Springer Nature Switzerland, 368–383. doi:10.1007/978-3-031-72940-9\_21

[18] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rup precht, and David Novotny. 2025. VGGT: Visual Geometry Grounded Transformer. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 5294–5306. doi:10.1109/CVPR52734.2025.00499

[19] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, andJunyang Lin. 2024. Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution. (2024). arXiv:2409.12191 [cs.CV] https://arxiv.org/abs/ 2409.12191

[20] Junjie Wen, Yichen Zhu, Jinming Li, Zhibin Tang, Chaomin Shen, and Feifei Feng. 2025. DexVLA: Vision-Language Model with Plug-In Difusion Expert for General Robot Control. In Proceedings of The 9th Conference on Robot Learning (Proceedings of Machine Learning Research, Vol. 305). PMLR, 3094–3114. https: //proceedings.mlr.press/v305/wen25b.html

[21] Sriram Yenamandra, Arun Ramachandran, Karmesh Yadav, Austin S. Wang, Mukul Khanna, Theophile Gervet, Tsung-Yen Yang, Vidhi Jain, Alexander Clegg, John M. Turner, Zsolt Kira, Manolis Savva, Angel X. Chang, Devendra Singh Chaplot, Dhruv Batra, Roozbeh Mottaghi, Yonatan Bisk, and Chris Paxton. 2023. HomeRobot: Open-Vocabulary Mobile Manipulation. In Proceedings ofThe 7th Conference on Robot Learning (Proceedings of Machine Learning Research, Vol. 229). PMLR, 1975–2011. https://proceedings.mlr.press/v229/yenamandra23a.html

[22] Naoki Yokoyama, Alex Clegg, Joanne Truong, Eric Undersander, Tsung-Yen Yang, Sergio Arnaud, Sehoon Ha, Dhruv Batra, and Akshara Rai. 2024. ASC: Adaptive Skill Coordination for Robotic Mobile Manipulation. IEEE Robotics and Automation Letters 9, 1 (2024), 779–786. doi:10.1109/LRA.2023.3336109

[23] Chen Yu, Weinan Zhang, Hang Lai, Zheng Tian, Laurent Kneip, and Jun Wang. 2023. Multi-Embodiment Legged Robot Control as a Sequence Modeling Problem. In 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 7250–7257. doi:10.1109/ICRA48891.2023.10161034

[24] Haichao Zhang, Haonan Yu, Le Zhao, Andrew Choi, Qinxun Bai, Yiqing Yang, and Wei Xu. 2025. Learning Multi-Stage Pick-and-Place With a Legged Mobile Manipulator. IEEE Robotics and Automation Letters 10, 11 (2025), 11419–11426. doi:10.1109/LRA.2025.3608425

[25] Jiazhao Zhang, Nandiraju Gireesh, Jilong Wang, Xiaomeng Fang, Chaoyi Xu, Weiguang Chen, Liu Dai, and He Wang. 2024. GAMMA: Graspability-Aware Mobile Manipulation Policy Learning Based on Online Grasping Pose Fusion. In 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 1399–1405. doi:10.1109/ICRA57147.2024.10610125

[26] Yuhang Zheng, Xiangyu Chen, Yupeng Zheng, Songen Gu, Runyi Yang, Bu Jin, Pengfei Li, Chengliang Zhong, Zengmao Wang, Lina Liu, Chao Yang, Dawei Wang, Zhen Chen, Xiaoxiao Long, and Meiqing Wang. 2024. GaussianGrasper: 3D Language Gaussian Splatting for Open-Vocabulary Robotic Grasping. IEEE Robotics and Automation Letters 9, 9 (2024), 7827–7834. doi:10.1109/LRA.2024. 3432348

[27] Minjie Zhu, Yichen Zhu, Jinming Li, Junjie Wen, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, Yaxin Peng, Feifei Feng, and Jian Tang. 2025. Scaling Difusion Policy in Transformer to 1 Billion Parameters for Robotic Manipulation. In 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 10838–10845. doi:10.1109/ICRA55743.2025.11128074