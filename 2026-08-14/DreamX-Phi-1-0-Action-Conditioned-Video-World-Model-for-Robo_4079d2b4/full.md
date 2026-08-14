# DreamX-Phi 1.0: Action-Conditioned Video World Model for Robotic Manipulation

![](images/bdb65852d0776fa5f445cc3cdd0fe4d1472fc2a2f093b0581f2fb4bffaae7cf1.jpg)

DreamX Team

We present DreamX-Phi 1.0, an action-conditioned video world model for robotic manipulation that, given an observed frame, a language instruction, and a prescribed action sequence comprising end-efector poses and gripper states, predicts the resulting future observations. Yet realism alone does not guarantee faithfulness: a convincing rollout can still move the wrong arm or lose the manipulated object. To ensure the prediction respects each arm’s commanded path, we inject per-arm SE(3) transformations into attention via PRoPE-style geometric encoding, preserving arm identity and rigid-motion structure. Action control alone does not fully constrain scene geometry or the evolution of small manipulated objects. We therefore add a lightweight depth branch for scene-level geometry and use SAM3 masks with a frozen V-JEPA teacher to maintain object consistency throughout grasping. We further distill the multi-step generator into a few-step student via distribution-matching distillation for eficient deployment. At the time of writing, DreamX-Phi achieves first place on Track 1 and second place on Track 2 of the WorldArena 2.0 Challenge. Our model and code will be publicly available.<sup>1</sup>

GitHub: github.com/AMAP-ML/DreamX-Phi Date: August 13, 2026

## 1 Introduction

World models provide a scalable way to evaluate candidate actions before physical execution. By predicting future scene evolution from the current observation and a proposed action sequence, they support planning over imagined outcomes (Yang et al., 2024; Wu et al., 2024). Modern video generators ofer powerful priors over appearance and motion, but photorealistic predictions are not necessarily faithful to the conditioning actions. Given a fixed initial observation, distinct prescribed trajectories should induce the corresponding robot motions and object interactions while leaving action-irrelevant scene content unchanged. Without this sensitivity, a model may produce a convincing video yet deviate from the commanded motion, miss the target object, or confuse grasping with release.

A key challenge is to design an action-conditioned interface that preserves the spatial structure of robot motion and relates it to the corresponding changes in the image. Existing approaches commonly encode actions as compact tokens or feature-wise modulation (Wu et al., 2024; Zhu et al., 2025; Huang et al., 2025). While compact, these representations do not explicitly preserve the rigid-body geometry of end-efector motion or indicate where the commanded motion should appear in the generated video. A more structured representation can encode both aspects explicitly: SE(3) trajectories describe how the robot moves in 3D, while dense motion cues indicate where and how that motion appears in the image (Miyato et al., 2024; Li et al., 2025a; Chen et al., 2026). Correct robot motion alone, however, does not guarantee a faithful rollout; prediction fidelity also depends on preserving scene geometry and the state of the manipulated object as the interaction unfolds. This motivates combining geometry-aware action conditioning with dense supervision of the resulting scene evolution.

To address these challenges, we introduce DreamX-Phi 1.0, a geometry-aware, action-conditioned video world model based on Wan2.2-TI2V-5B (Team Wan, 2025; Wan Team, 2025). Rather than compressing the prescribed action sequence into a generic control embedding, DreamX-Phi preserves the SE(3) structure of end-efector motion through PRoPE-based attention and relates it to the corresponding image dynamics through action-derived motion cues (Miyato et al., 2024; Li et al., 2025a; Chen et al., 2026). Its learning objective captures the physical consequences of interaction by regularizing scene geometry and object-centric temporal consistency. This encourages action-faithful rollouts in which the manipulated object retains its visual identity and evolves coherently over time. For eficient deployment, DMD post-training distills the iterative generation process into a few-step sampler (Yin et al., 2024b,a).

![](images/fa93902e3bff10e8329ab03b3c547ee31bd4f36a1d5a7926ec12977b2ddda115.jpg)  
Figure 1 Overview of DreamX-Phi 1.0, an action-conditioned video world model that predicts future observations from a single frame and prescribed bimanual actions.

To support both general video prediction and action-conditioned dynamics, we curate a heterogeneous training corpus from egocentric video and robot interaction data spanning real and simulated environments, as detailed in Section 3. Action-agnostic clips broaden the model’s visual and motion priors, while temporally aligned robot trajectories ground future predictions in control. For RoboTwin, we process the training videos with DreamX-Refiner to improve their resolution and visual quality. Evaluations on WorldArena 1.0 and 2.0 demonstrate strong performance (Shang et al., 2026a,b). In the fixed WorldArena 2.0 leaderboard snapshot, our submissions rank first on Track 1 and tie for second on Track 2. DreamX-Phi also achieves an ofline EWMScore-P of 76.88 on WorldArena 1.0 Track 1.

## Our contributions are threefold:

• We introduce a geometry-aware action representation that connects prescribed robot actions to their visual consequences. By combining structured SE(3) trajectories with image-space motion cues, it encodes both the commanded 3D end-efector motion and where its efects should appear in the predicted video.

• We propose manipulation-aware supervision that focuses learning on scene geometry and the state of the manipulated object. By emphasizing depth structure and object evolution, it encourages physically coherent robot–object interactions throughout the predicted rollout.

• Extensive evaluations on WorldArena 1.0 and 2.0 demonstrate the efectiveness of DreamX-Phi. In the fixed WorldArena 2.0 snapshot, our submissions rank first on Track 1 and tie for second on Track 2; our model also achieves an ofline EWMScore-P of 76.88 on WorldArena 1.0 Track 1.

## 2 Related Work

Action-Conditioned Video World Models. Large video generators such as Wan and Cosmos 3 provide scalable visual priors for physical dynamics (Team Wan, 2025; NVIDIA, 2026). Building on such priors, interactive models predict how a scene evolves under an external control signal. UniSim, iVideoGPT, and DreamX-World study controllable visual dynamics across broad domains, while AVID adapts pretrained video difusion through a learned action interface (Yang et al., 2024; Wu et al., 2024; DreamX Team et al., 2026; Rigter et al., 2025). Robot-specific models make the conditioning contract more concrete: IRASim aligns robot trajectories with video frames, Vid2World introduces causal generation with action guidance, and HMA learns heterogeneous action–video dynamics (Zhu et al., 2025; Huang et al., 2025; Wang et al., 2025). These methods establish video prediction as a learned simulator, but action fidelity remains dificult: a visually plausible rollout may still move the wrong arm or produce an incorrect object response. DreamX-Phi focuses on this prescribed-action setting, predicting future observations from a given bimanual trajectory rather than generating the actions themselves.

World Action Models for Robotics. Current world action models (WAMs) connect video and control in three main ways. The first injects low-dimensional action tokens or adapters into a video generator, as in IRASim, Vid2World, and HMA (Zhu et al., 2025; Huang et al., 2025; Wang et al., 2025). The second jointly models visual futures and actions, enabling the video model to act as a policy or planner, as in UVA, WorldVLA, LingBot-VA, DreamZero, and Cosmos Policy (Li et al., 2025b; Cen et al., 2025; Li et al., 2026; Ye et al., 2026; Kim et al., 2026b). The third converts robot motion into a spatially aligned condition: OSCAR renders kinematic skeletons, Robot-Factored World Models render robot geometry from commands, and FlowWAM represents actions with optical flow (Wu and Gao, 2026; Kim et al., 2026a; Chen et al., 2026). These interfaces trade compactness for structure: token-based controls are general but geometrically implicit, whereas rendered or flow-based controls localize motion in the image but do not directly preserve the continuous rigid-body trajectory of each arm.

Structured Control and Physical-Consistency Supervision. Geometry-aware attention provides a direct way to retain rigid-motion structure. GTA inserts relative SE(3) transformations into attention, while projective relative positional encoding applies known camera geometry to queries, keys, values, and outputs (Miyato et al., 2024; Li et al., 2025a). Geometry alone, however, constrains the commanded robot motion rather than the full scene response. Depth supervision can organize scene geometry, object masks can prevent small contact regions from being overwhelmed by a uniform generative objective (Lipman et al., 2023), and predictive video features can regularize object evolution (Guo et al., 2026; Carion et al., 2025; Assran et al., 2025). DreamX-Phi combines these complementary signals: arm-specific PRoPE preserves the continuous SE(3) trajectory, robot-only flow supplies an image-aligned motion cue, and depth, SAM3 mask weighting, and frozen V-JEPA relational supervision target scene and object consistency. The central design choice is therefore to preserve the structure of the commanded action while separately supervising its visual and physical consequences.

## 3 Data Curation

Data Sources. Reliable action-conditioned prediction depends not only on the diversity of robot motions, but also on broad visual coverage and consistent alignment between observations and control signals. We therefore construct the corpus from three complementary sources: action-free egocentric video, real-robot demonstrations, and simulated robot trajectories. Together, these sources span everyday visual dynamics, physically executed manipulation, and controlled variation in task and scene configuration. Table 1 summarizes their scale. Throughout curation, we preserve the camera structure provided by each source: single-view recordings remain single-view examples, whereas synchronized multi-view observations are combined as described below.

Curation and Normalization. The raw robot collections contain behaviors that are poorly matched to the manipulation setting studied in this report. We remove trajectories dominated by mobile-base motion, dexterous-hand operation, or stationary segments, while deliberately retaining failed task executions because they expose informative failure modes and non-ideal interaction dynamics. After removing mobile-base and stationary segments, the filtered AgiBot imitation-learning split contains 178.7 hours. For sources with action annotations, we normalize observations, instructions, robot states, and actions into a common LeRobot v2.1 representation, providing a consistent interface across otherwise heterogeneous datasets.

Table 1 Data sources used to construct the curated corpus. RoboTwin is reported in clips because its duration is not available.
<table><tr><td>Source</td><td>Domain</td><td>Data volume</td></tr><tr><td>Ego4D (Grauman et al., 2022)</td><td>Egocentric video</td><td>3,700 h</td></tr><tr><td>AgiBot World 2026 (AgiBot World Team, 2026)</td><td>Real robot</td><td>1,900 h</td></tr><tr><td>InternData-A1 (Tian et al., 2025) Cosmos3-DROID (Khazatsky et al.,</td><td>Real / simulated robot</td><td>78 h real; 3,747 h simulated</td></tr><tr><td>2024; NVIDIA Corporation, 2026)</td><td>Real robot</td><td>350 h</td></tr><tr><td>RoboCOIN (Wu et al., 2025)</td><td>Real robot</td><td>618 h</td></tr><tr><td>RoboTwin 2.0 (Chen et al., 2025)</td><td>Simulated robot</td><td>25,000 action-annotated clips</td></tr></table>

Action-Agnostic Pretraining. The action-agnostic pool includes every retained video, regardless of whether the source also provides action annotations. A single camera stream remains a single-view example; when synchronized streams from multiple cameras are available, we spatially concatenate them into a unified multi view video. This view-adaptive organization preserves the information available in each source and makes the curated corpus applicable to both single-view and multi-view downstream settings. It also exposes the model to the complementary appearance statistics and motion patterns found in egocentric video, real-robot operation, and simulation.

Action-Conditioned Fine-Tuning. The action-conditioned pool is restricted to videos with synchronized action annotations. Each video is paired with its corresponding robot trajectory and the annotations available in the common representation, so that visual and control streams remain temporally consistent. Its RoboTwin component contains 25,000 bimanual clips, spanning both clean and randomized variants. Before these clips enter the curated pool, we apply our video refinement model, DreamX-Refiner, to super-resolve the RoboTwin videos; the resulting high-resolution clips provide the visual data used in this phase.

## 4 Method

## 4.1 Overview

Given an observed RGB frame x<sub>0</sub>, a language instruction c, and a prescribed bimanual action trajectory a<sub>1:T</sub> containing end-efector poses and gripper states, our goal is to model the conditional distribution

$$
p _ { \theta } ( \mathbf { x } _ { 1 : T } \mid \mathbf { x } _ { 0 } , \mathbf { a } _ { 1 : T } , \mathbf { c } )\tag{1}
$$

using a Wan2.2-TI2V-5B video difusion transformer. The latent of the first frame provides the visual context, and the future-frame latents are learned under a flow-matching objective (Lipman et al., 2023). The resulting design combines structured action conditioning with auxiliary geometric and object-centric supervision, while retaining DMD as a post-training route to few-step inference. As illustrated in Figure 2, the framework is organized into three parts: action-conditioned video prediction, training supervision, and few-step post-training.

## 4.2 PRoPE Control for Bimanual Actions

World action models commonly encode actions as low-dimensional tokens or additive features and inject them through concatenation, modulation, or cross-attention (Wu et al., 2024; Zhu et al., 2025; Huang et al., 2025; NVIDIA, 2026). These generic interfaces are flexible, but they leave the rigid-body relations within an end-efector trajectory to be inferred implicitly. PRoPE provides a more structured alternative by inserting known relative transforms directly into self-attention. The transforms afect both attention weights and value aggregation while remaining invariant to the global coordinate frame (Li et al., 2025a). We incorporate this mechanism as a residual branch, so geometric control augments rather than replaces the pretrained generative path (DreamX Team et al., 2026). Because robot end-efector commands form ordered SE(3) trajectories, this group-action formulation is a natural interface for action conditioning.

![](images/47b4497dd33f8a71b21be72bec8599beca65aed9106a3f50a00e73c50b414bb3.jpg)  
Figure 2 Overview of the DreamX-Phi 1.0 framework. (1) Given an observed RGB frame, a language instruction, and an externally specified bimanual action trajectory, a Wan2.2-TI2V-5B video difusion transformer predicts the future RGB video. Arm-grouped PRoPE and a robot-only optical-flow cue provide complementary geometric and image-plane action conditioning. (2) During training, SAM3-derived masks reweight the RGB objective, Depth Anything 3 (DA3) depth maps provide targets for the depth-latent objective, and a frozen V-JEPA teacher supplies object-relationa supervision. (3) DMD with adversarial training distills the multi-step teacher into a few-step student.

Our use of PRoPE difers from its original camera setting: an end efector is not treated as a physical camera. Instead, we reuse only the group-action attention mechanism to expose relative rigid transforms along an ordered robot trajectory. Adapting the camera formulation requires three corresponding changes: all arms must be expressed in a shared coordinate system, each arm must retain a persistent portion of the attention representation, and the gripper state must be injected separately from the SE(3) transform.

Action Representation. At frame t, arm k is described by position $\mathbf { p } _ { t } ^ { k }$ , quaternion $\mathbf { q } _ { t } ^ { k }$ (converted to $\mathbf { R } _ { t } ^ { k } )$ and gripper value $g _ { t } ^ { k }$ . We first construct its end-efector frame and then express that frame relative to the initial pose of arm 1:

$$
\mathbf { G } _ { t } ^ { k } = \left[ \begin{array} { l l } { \mathbf { R } _ { t } ^ { k } } & { \mathbf { p } _ { t } ^ { k } } \\ { \mathbf { 0 } ^ { \top } } & { 1 } \end{array} \right] , \qquad \bar { \mathbf { G } } _ { t } ^ { k } = ( \mathbf { G } _ { 1 } ^ { 1 } ) ^ { - 1 } \mathbf { G } _ { t } ^ { k } .\tag{2}
$$

This construction places all arms in a common reference frame. We then normalize translations using a single motion-amplitude factor,

$$
\gamma = \underset { k , t } { \operatorname* { m a x } } \left\| \bar { \mathbf { p } } _ { t } ^ { k } - \bar { \mathbf { p } } _ { 1 } ^ { k } \right\| _ { 2 } , \quad \quad s _ { \gamma } = \left\{ \gamma , \gamma > \epsilon , \quad \quad \widetilde { \mathbf { G } } _ { t } ^ { k } = \left[ \bar { \mathbf { R } } _ { t } ^ { k } \quad \bar { \mathbf { p } } _ { t } ^ { k } / s _ { \gamma } \right] . \right.\tag{3}
$$

Because $\gamma$ measures motion amplitude rather than absolute workspace size, the resting distance between the arms does not dominate the scale. We next invert the normalized frames, $\mathbf { A } _ { t } ^ { k } = ( \widetilde { \mathbf { G } } _ { t } ^ { k } ) ^ { - 1 }$ , and temporally align both $\mathbf { A } _ { t } ^ { k }$ and $g _ { t } ^ { k }$ with the VAE latent frames. In the two-arm setting, this yields $\mathbf { \bar { A } } \in \mathbb { R } ^ { 2 \times T _ { \mathrm { l a t } } \times 4 \times 4 }$ and $\mathbf { g } \in \mathbb { R } ^ { 2 \times T _ { \mathrm { l a t } } }$ . A missing arm is represented by identity poses with $g _ { t } ^ { k } = 0$

Geometric Attention. Each transformer block contains a parallel attention branch with dedicated query, key, value, and output projections, conditioned on A and g. We adopt an identity intrinsic matrix, ${ \bf K } = { \bf I } _ { 3 }$ so the PRoPE projection matrix reduces to $\mathbf { P } _ { t } ^ { k } = \mathbf { A } _ { t } ^ { k }$ (Li et al., 2025a). We partition the attention heads into fixed contiguous groups $\{ \mathcal { H } _ { k } \}$ , with one group assigned to each arm. For token i at latent frame $n ( i )$ and head $h \in \mathcal { H } _ { k }$ , we define $\mathbf { D } _ { i } = \mathbf { I } _ { d _ { h } / 4 } \otimes \mathbf { A } _ { n ( i ) } ^ { k }$ and apply the following token-wise transforms:

$$
\begin{array} { r } { { \bf Q } _ { i } ^ { \prime } = { \bf D } _ { i } ^ { \top } { \bf Q } _ { i } , \qquad { \bf K } _ { i } ^ { \prime } = { \bf D } _ { i } ^ { - 1 } { \bf K } _ { i } , \qquad { \bf V } _ { i } ^ { \prime } = { \bf D } _ { i } ^ { - 1 } { \bf V } _ { i } , \qquad { \bf Q } _ { i } ^ { \mathrm { a c t } } = { \bf D } _ { i } \left[ \mathrm { A t t n } ( { \bf Q } ^ { \prime } , { \bf K } ^ { \prime } , { \bf V } ^ { \prime } ) \right] _ { i } . } \end{array}\tag{4}
$$

All patches associated with the same frame and arm share D<sub>i</sub>. Consequently, a token pair $( i , j )$ is coupled through the relative motion $\mathbf { D } _ { i } \mathbf { D } _ { j } ^ { - 1 }$ rather than through an absolute coordinate frame.

Gripper and Residual. Gripper opening is scalar-valued and therefore cannot be represented as an SE(3) element. We inject it after the inverse geometric map as a per-arm bias on the corresponding attention heads:

$$
\mathbf { b } _ { t } ^ { k } = \mathbf { W } _ { g } g _ { t } ^ { k } + \mathbf { b } _ { g } , \qquad \mathbf { o } _ { t , u , h } ^ { \mathrm { a c t } }  \mathbf { o } _ { t , u , h } ^ { \mathrm { a c t } } + \mathbf { b } _ { t } ^ { k } , \quad h \in \mathcal { H } _ { k } ,\tag{5}
$$

broadcast over the spatial locations u of all heads in $\mathcal { H } _ { k }$ . The resulting heads are concatenated, projected back to the model width, and added to the pretrained self-attention output. Both the gripper adapter and this output projection are initialized to zero, keeping the residual branch silent until it is updated during training.

## 4.3 Auxiliary Depth Supervision for 3D Consistency

An RGB prediction objective can capture appearance and motion without explicitly constraining surface ordering, object extent, or contact geometry. To supply this missing geometric signal during training, we draw on the depth adaptation design of X-WAM (Guo et al., 2026) and add a lightweight auxiliary depth branch. Let d denote a depth video aligned with the RGB sequence. We replicate each single-channel depth map across the channel dimension to obtain a pseudo-RGB input, then encode it with the same frozen video VAE used for RGB. This produces the latent depth target $\mathbf { z } ^ { d } = \mathcal { E } ( \mathbf { d } )$ , which is predicted by a branch attached to the tail of the RGB transformer.

Concretely, for an RGB transformer with N blocks, we replicate its final M blocks $( M < N )$ to form the auxiliary branch, leaving the first $N - M$ blocks as a shared trunk. The trunk output initializes both pathways, and each replicated depth block is initialized from its pretrained RGB counterpart. At every adapted layer $j ,$ cross-attention allows the depth pathway to read the corresponding RGB representation:

$$
\mathbf { h } _ { d } ^ { j } = \operatorname { D e p t h B l o c k } _ { j } \left( \mathbf { h } _ { d } ^ { j - 1 } ; \mathbf { K } _ { \mathrm { r g b } } ^ { j } , \mathbf { V } _ { \mathrm { r g b } } ^ { j } \right) , \qquad j = 1 , \dotsc , M .\tag{6}
$$

Here, ${ \bf K } _ { \mathrm { r g b } } ^ { j }$ and $\mathbf { V } _ { \mathrm { r g b } } ^ { j }$ are obtained from the RGB branch at layer $j .$ The connection is deliberately one-way: the depth branch can consume RGB features, but the RGB branch never consumes depth features. This asymmetric design leaves the RGB forward computation unchanged and therefore keeps depth prediction optional at inference.

A dedicated output head maps the final depth tokens to ${ \widehat { \mathbf { z } } } ^ { d }$ . We supervise this prediction with a latent-space mean-squared error,

$$
\mathcal { L } _ { \mathrm { d e p t h } } = \frac { 1 } { \left| \mathbf { z } ^ { d } \right| } \left\| \widehat { \mathbf { z } } ^ { d } - \mathbf { z } ^ { d } \right\| _ { 2 } ^ { 2 } .\tag{7}
$$

Thus, unlike the RGB generation pathway, the depth branch is supervised directly in latent space rather than treated as a separate noisy difusion sequence. The auxiliary objective encourages the shared representation to encode stronger geometric structure during training, without introducing a depth-related requirement at deployment.

## 4.4 Object-Centric Physical Consistency

The flow-matching RGB objective averages errors over all valid future tokens. In manipulation videos, however, the robot arm and manipulated object often occupy only a small fraction of the frame, allowing the static background to dominate the contact-local errors that determine whether an interaction is physically plausible. A rollout can therefore appear photorealistic even when the gripper misses or penetrates the object, the object does not respond to contact, or its shape and state change abruptly after a grasp. PRoPE constrains the commanded arm motion but, on its own, does not enforce a coherent object response. To address these complementary aspects, we use a SAM3 mask to focus the RGB objective on the manipulated object and a frozen V-JEPA teacher to regularize its spatiotemporal evolution (Carion et al., 2025; Assran et al., 2025). Together, these signals encourage predicted object motion to remain coupled to arm contact, rather than rewarding global visual quality alone.

Object-Aware Supervision. Ofline SAM3 processing provides a binary mask video for the manipulated object. The mask is used only for supervision: SAM3 is not fine-tuned jointly with the model, and no mask is required at inference. After projection onto the latent grid, token i is assigned $m _ { i } \in \{ 0 , 1 \}$ , and its normalized weight is defined as

$$
\widetilde { w } _ { i } = 1 + ( \lambda _ { m } - 1 ) m _ { i } , \qquad w _ { i } = \frac { \widetilde { w } _ { i } } { \frac { 1 } { | \mathcal { V } | } \sum _ { j \in \mathcal { V } } \widetilde { w } _ { j } } , \qquad \mathcal { L } _ { \mathrm { r g b } } ^ { \mathrm { o b j } } = \frac { 1 } { | \mathcal { V } | } \sum _ { i \in \mathcal { V } } w _ { i } \ell _ { i } ^ { \mathrm { F M } } ,\tag{8}
$$

where V denotes the set of valid future tokens and $\lambda _ { m } > 1$ is the object-to-background ratio before normalization. The mean-weight normalization stabilizes the overall loss scale as the mask area changes, while clips without a valid mask retain uniform weights. As a result, contact-local errors—such as a missed object displacement or a contact-induced deformation—remain influential despite the much larger static background.

V-JEPA Alignment for Physical Consistency. Mask reweighting operates on local flow-matching errors and does not, by itself, determine whether an object follows a coherent trajectory through contact. Similar frame-level errors may still conceal temporal drift, an inconsistent grasp state, or object motion that is decoupled from the arm. We therefore use a frozen V-JEPA teacher to constrain relations among object features across both space and time. For each sample b, we select a temporally stratified set of masked teacher tokens $\mathcal { T } _ { b }$ , capped at $M _ { \mathrm { m a x } }$ , and interpolate the corresponding video-model hidden tokens to the same coordinates. Let $M _ { b } = | \mathcal { T } _ { b } |$ , and let the normalized projected student and teacher features be $\mathbf { S } _ { b } , \mathbf { Q } _ { b } \in \mathbb { R } ^ { M _ { b } \times d }$ Rather than matching feature coordinates directly, we align their Gram matrices so that the student is not tied to the teacher’s feature basis:

$$
\ell _ { \mathrm { J E P A } } ^ { ( b ) } = \frac { 1 } { M _ { b } ^ { 2 } } \left. \mathbf { S } _ { b } \mathbf { S } _ { b } ^ { \intercal } - \mathbf { Q } _ { b } \mathbf { Q } _ { b } ^ { \intercal } \right. _ { 1 } .\tag{9}
$$

The pairwise relational objective discourages object identity, shape, and state from drifting across the contact interval. In this sense, the mask-weighted RGB term identifies where prediction accuracy is most important, whereas V-JEPA constrains how the manipulated object evolves over time; the two objectives jointly promote physically consistent arm–object interaction. To keep this supervision stable, a sample contributes to the relational loss only when its mask provides enough tokens and its flow-matching noise is not too large. Specifically,

$$
r _ { b } = \mathbb { I } [ M _ { b } \ge M _ { \mathrm { m i n } } ] \mathbb { I } [ \sigma _ { b } \le \sigma _ { \mathrm { m a x } } ] , \qquad \mathcal { L } _ { \mathrm { J E P A } } = \frac { \sum _ { b } r _ { b } \ell _ { \mathrm { J E P A } } ^ { ( b ) } } { \mathrm { m a x } ( 1 , \sum _ { b } r _ { b } ) } ,\tag{10}
$$

Samples that fail either gate contribute zero. The teacher remains frozen throughout training. For eligible samples, the projector receives gradients; during an initial projector-only phase, gradients stop at the video-model hidden state and are subsequently opened linearly to the trainable video-model parameters.

## 4.5 Few-Step Post-Training

To reduce the number of denoising evaluations, we distill the multi-step generator following DMD2 $\left( \mathrm { Y i n } \right.$ et al., 2024a). Let $G _ { \eta }$ denote the $N .$ -step student, let $\mathbf { y } = ( \mathbf { x } _ { 0 } , \mathbf { a } _ { 1 : T } , \mathbf { c } )$ collect its conditions, and let $\widetilde { \mathbf { z } } _ { 0 }$ be the student’s clean future-video latent prediction at a sampled denoising step. In contrast to the text-conditioned image-generation setting, y includes both the observed frame and the temporally aligned, prescribed bimanual action trajectory. The student, frozen teacher, and online fake-score denoiser all receive the same ${ \bf y } ;$ the adversarial classification head instead operates on the denoiser’s bottleneck features.

For a noise level τ sampled from the DMD distribution, let $q _ { \eta , \tau } ( \cdot \mid \mathbf { y } )$ and $p _ { \mathrm { d a t a } , \tau } ( \cdot \mid \mathbf { y } )$ denote the conditional marginals produced by applying the same Wan forward-noising process to the student and real future-video latents, respectively. Distribution matching is then expressed as the following KL objective:

$$
{ \mathcal { L } } _ { \mathrm { D M D } } ( \eta ) = \mathbb { E } _ { { \mathbf { y } } \sim p _ { \mathrm { d a t a } } ( { \mathbf { y } } ) , \tau } [ D _ { \mathrm { K L } } ( q _ { \eta , \tau } ( \cdot \mid { \mathbf { y } } ) \parallel p _ { \mathrm { d a t a } , \tau } ( \cdot \mid { \mathbf { y } } ) ) ] .\tag{11}
$$

We complement this distribution-matching term with the noised non-saturating GAN objective. Given $( \mathbf { z } _ { 0 } ^ { \star } , \mathbf { y } ) \sim p _ { \mathrm { d a t a } }$ and a sampled GAN noise level $u ,$ we obtain $\mathbf { z } _ { u } ^ { r }$ and $\mathbf { z } _ { u } ^ { f }$ by independently applying the same Wan forward-noising process at u to $\mathbf { z } _ { \mathrm { 0 } } ^ { \star }$ and $\widetilde { \mathbf { z } } _ { 0 } .$ , respectively. Let $D ( \cdot , u ; \mathbf { y } ) \in ( 0 , 1 )$ denote the classification head’s conditional probability that a latent is real. The generator and discriminator objectives are

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { a d v } } ^ { G } = - \mathbb { E } \big [ \log D ( \mathbf { z } _ { u } ^ { f } , u ; \mathbf { y } ) \big ] , } \\ & { \mathcal { L } _ { \mathrm { a d v } } ^ { D } = - \mathbb { E } \big [ \log D ( \mathbf { z } _ { u } ^ { r } , u ; \mathbf { y } ) + \log \big ( 1 - D ( \mathbf { z } _ { u } ^ { f } , u ; \mathbf { y } ) \big ) \big ] . } \end{array}\tag{12}
$$

The student is therefore optimized with

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s t u d e n t } } ^ { \mathrm { D M D } } = \mathcal { L } _ { \mathrm { D M D } } + \lambda _ { \mathrm { a d v } } \mathcal { L } _ { \mathrm { a d v } } ^ { G } , \qquad \lambda _ { \mathrm { a d v } } \geq 0 . } \end{array}\tag{13}
$$

The fake-score denoiser and its adversarial classification head are updated in a separate auxiliary step. Finally, the few-step student uses the same fixed N-step denoising schedule during training and inference. Under backward simulation, earlier student steps generate the input to a sampled step without gradient tracking; the conditioning tuple y remains fixed across all steps, and gradients propagate only through the sampled step.

## 5 Evaluation

We evaluate DreamX-Phi 1.0 along two complementary axes: the fidelity of its predicted visual rollouts and its utility as a learned environment for policy training. WorldArena 2.0 is our primary benchmark: Track 1 evaluates video prediction conditioned on language instructions or robot actions, while Track 2 measures whether a policy optimized through interaction with the submitted world model succeeds in held-out simulator episodes. We additionally report WorldArena 1.0 Track 1 results to contextualize performance against earlier world models (Shang et al., 2026a,b).

## 5.1 Datasets

Both benchmarks use evaluation sets curated and released by the WorldArena organizers from RoboTwin 2.0 trajectories (Chen et al., 2025; Shang et al., 2026a,b). WorldArena 2.0 Track 1 contains 1,000 episodes. Each provides an initial RGB observation together with a language instruction and a robot action trajectory, and the model predicts the subsequent rollout conditioned on either signal. WorldArena 1.0 Track 1 follows the Clean-50 protocol, covering 50 manipulation tasks with 10 held-out episodes per task.

WorldArena 2.0 Track 2 evaluates whether the learned dynamics are useful beyond open-loop video prediction. The submitted world model serves as the rollout environment for optimizing a $\pi _ { 0 . 5 }$ policy using an organizerprovided initialization and a fixed reward model (Physical Intelligence et al., 2025). The resulting policy is then evaluated on held-out Adjust Bottle episodes in RoboTwin 2.0 (Shang et al., 2026b).

## 5.2 Metrics

For Track 1, we report EWMScore-P together with the 15 normalized component metrics exposed by the oficial leaderboards. The metrics span visual quality, temporal dynamics, content consistency, physical interaction, 3D structure, and conditioning fidelity, while EWMScore-P summarizes overall performance by averaging the component scores. WorldArena 2.0 additionally caps Dynamic Degree, Flow Score, and Motion Smoothness by their ground-truth reference values before aggregation. Track 2 is evaluated by the policy success rate on held-out Adjust Bottle episodes.

## 5.3 WorldArena 2.0 Results

Because the leaderboard is continuously updated, all WorldArena 2.0 results in this report are taken from the oficial snapshot at commit cb8f9c2, dated August 12, 2026.<sup>1</sup> We report both tracks under the checkpoint name DreamX-Phi-1.0-FDM-0730.<sup>2</sup> We report the complete leaderboard Top 3 together with selected opensource reference models evaluated in the benchmark paper; all displayed values come from the leaderboard snapshot rather than the paper tables. Tables 2 and 3 report the two tracks, respectively.

Table 2 WorldArena 2.0 Track 1 leaderboard comparison at the August 12, 2026 snapshot. We report the oficial Top 3 and selected open-source reference systems. Scores use a 0–100 scale, and EWMScore-P averages the 15 component scores. Boldface and underlining denote the best and second-best results among the displayed systems, respectively.  
(a) Visual quality, motion quality, and content consistency
<table><tr><td rowspan="2">Model</td><td colspan="3">Visual Quality</td><td colspan="3">Motion Quality</td><td colspan="3">Content Consistency</td></tr><tr><td>Image Quality</td><td>Aesthetic Quality</td><td>JEPA Similarity</td><td>Dynamic Degree</td><td>Flow</td><td>Motion Score Smoothness</td><td>Subject Consistency</td><td>Background Consistency</td><td>Photometric Consistency</td></tr><tr><td>DreamX-Phi- 1.0-FDM-0730</td><td>63.25</td><td>43.38</td><td>92.93</td><td>22.90</td><td>5.81</td><td>63.26</td><td>71.93</td><td>84.52</td><td>14.29</td></tr><tr><td>Alpha-World</td><td>64.59</td><td>43.83</td><td>92.96</td><td>22.83</td><td>5.80</td><td>63.13</td><td>71.76</td><td>83.85</td><td>14.84</td></tr><tr><td>FlowWAM-FiveAges</td><td>64.31</td><td>44.00</td><td>92.54</td><td>22.76</td><td>5.80</td><td>62.67</td><td>71.94</td><td>83.69</td><td>14.96</td></tr><tr><td>GigaWorld-0</td><td>50.57</td><td>43.30</td><td>44.41</td><td>22.59</td><td>5.79</td><td>62.73</td><td>56.11</td><td>79.77</td><td>3.60</td></tr><tr><td>Vidar</td><td>45.33</td><td>42.75</td><td>54.76</td><td>18.16</td><td>5.60</td><td>63.63</td><td>53.36</td><td>69.95</td><td>10.96</td></tr><tr><td>WoW</td><td>49.37</td><td>40.63</td><td>73.25</td><td>22.27</td><td>5.76</td><td>62.28</td><td>65.55</td><td>83.41</td><td>10.73</td></tr><tr><td>Ctrl-World</td><td>53.78</td><td>38.65</td><td>78.70</td><td>23.15</td><td>5.79</td><td>63.70</td><td>70.63</td><td>85.05</td><td>14.21</td></tr><tr><td>IRASim</td><td>45.18</td><td>31.43</td><td>62.15</td><td>18.44</td><td>5.49</td><td>59.64</td><td>48.02</td><td>65.70</td><td>10.44</td></tr></table>

(b) Physics adherence, 3D accuracy, controllability, and overall score
<table><tr><td rowspan="2">Model</td><td colspan="2">Physics Adherence</td><td colspan="2">3D Accuracy</td><td colspan="2">Controllability</td><td rowspan="2">EWM Score-P</td></tr><tr><td>Interaction Quality</td><td>Trajectory Accuracy</td><td>Depth Accuracy</td><td>Perspectivity</td><td>Instruction Following</td><td>Semantic Alignment</td></tr><tr><td>DreamX-Phi- 1.0-FDM-0730</td><td>57.36</td><td>57.15</td><td>98.55</td><td>82.24</td><td>61.62</td><td>90.53</td><td>60.65</td></tr><tr><td>Alpha-World</td><td>57.18</td><td>49.22</td><td>97.14</td><td>82.38</td><td>60.58</td><td>91.84</td><td>60.13</td></tr><tr><td>FlowWAM-FiveAges</td><td>53.74</td><td>49.76</td><td>98.99</td><td>80.52</td><td>58.06</td><td>92.06</td><td>59.72</td></tr><tr><td>GigaWorld-0</td><td>47.46</td><td>15.24</td><td>77.22</td><td>74.98</td><td>51.22</td><td>85.92</td><td>48.06</td></tr><tr><td>Vidar</td><td>43.04</td><td>16.99</td><td>81.55</td><td>67.70</td><td>45.96</td><td>87.24</td><td>47.13</td></tr><tr><td>WoW</td><td>49.87</td><td>21.39</td><td>80.08</td><td>76.24</td><td>52.95</td><td>87.67</td><td>52.10</td></tr><tr><td>Ctrl-World</td><td>52.72</td><td>45.24</td><td>94.50</td><td>76.56</td><td>52.52</td><td>88.45</td><td>56.24</td></tr><tr><td>IRASim</td><td>37.88</td><td>22.68</td><td>89.68</td><td>51.56</td><td>38.18</td><td>88.03</td><td>44.97</td></tr></table>

Table 3 WorldArena 2.0 Track 2 results at the August 12, 2026 snapshot. Values are Adjust Bottle success rates (%) for the oficial Top 3 and selected open-source reference systems; higher is better. Our submission is highlighted. Boldface and underlining denote the best and second-best results among the displayed systems, respectively.
<table><tr><td>Model</td><td>Adjust Bottle</td></tr><tr><td>WOVR-PLUS</td><td>68.75</td></tr><tr><td>DreamX-Phi-1.0-FDM-0730</td><td>67.19</td></tr><tr><td>Lute</td><td>67.19</td></tr><tr><td>CtrlWorld</td><td>62.50</td></tr><tr><td>IRASim</td><td>61.33</td></tr><tr><td>RoboScape</td><td>60.74</td></tr><tr><td>OpenSora</td><td>60.16</td></tr><tr><td>Cosmos-Predict-2.5 (action)</td><td>59.38</td></tr><tr><td>iVideoGPT</td><td>56.25</td></tr></table>

On the complete 31-entry Track 1 leaderboard, our entry ranks first with an EWMScore-P of 60.65. On Track 2, DreamX-Phi-1.0-FDM-0730 achieves a 67.19% Adjust Bottle success rate and ties for second place in the full snapshot. Alpha-World and FlowWAM-FiveAges complete the Track 1 Top 3, while WOVR-PLUS leads Track 2 and Lute shares second place with our submission. These rankings are snapshot-specific and do not represent the final challenge standings.

Figure 3 shows qualitative Track 1 rollouts predicted by DreamX-Phi-1.0-FDM-0730 under both RoboTwin 2.0 evaluation settings. The model keeps the arms, grippers, and manipulated objects coherent across the rollout, and the same behavior holds when backgrounds, textures, lighting, and distractor layouts are randomized.

(a) Clean scenes  
![](images/790a33152680d0b46177a3354716752f8d3d64b5f3e69e5685d3dcbbc456307e.jpg)

(b) Domain-randomized scenes  
![](images/3dcda89889bbce07a1097d34e1e4bc4eab462be4c88f1302217aead11b01a4da.jpg)  
Figure 3 Qualitative WorldArena 2.0 Track 1 rollouts from DreamX-Phi 1.0. Each row is a predicted episode, with frames sampled in temporal order from left to right. (a) Standard RoboTwin 2.0 scenes. (b) Domain-randomized scenes with varied backgrounds, textures, lighting, and distractor objects.

## 5.4 WorldArena 1.0 Results

Table 4 places our ofline WorldArena 1.0 evaluation alongside the oficial leaderboard Top 3 and selected open-source reference models from the snapshot at commit 483dfcc, dated July 15, 2026.<sup>3</sup> DreamX-Phi-1.0- FDM-0730 was evaluated ofline on the WorldArena 1.0 Track 1 test set and is not an entry in this pinned leaderboard snapshot.<sup>4</sup>

Table 4 WorldArena 1.0 Track 1 comparison. Our ofline evaluation of DreamX-Phi-1.0-FDM-0730 is shown alongside the oficial leaderboard Top 3 and selected open-source reference systems from the July 15, 2026 snapshot. Scores use a 0–100 scale, and EWMScore-P averages the 15 component scores. Higher is better; boldface and underlining denote the best and second-best results among the displayed systems, respectively.  
(a) Visual quality, motion quality, and content consistency
<table><tr><td rowspan="3">Model</td><td colspan="3">Visual Quality</td><td colspan="3">Motion Quality</td><td colspan="3">Content Consistency</td></tr><tr><td>Image Quality</td><td>Aesthetic Quality</td><td>JEPA Similarity</td><td>Dynamic Degree</td><td>Flow Score</td><td>Motion Smoothness</td><td>Subject Consistency</td><td>Background Consistency</td><td>Photometric Consistency</td></tr><tr><td>DreamX-Phi-</td><td>55.72</td><td>40.87</td><td>92.73</td><td>88.71 100.00</td><td></td><td>92.22</td><td>82.40</td><td>88.96</td><td>10.72</td></tr><tr><td>1.0-FDM-0730</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UNIS</td><td>53.94</td><td>40.79</td><td>90.60</td><td>73.70</td><td>86.02</td><td>95.51</td><td>79.05</td><td>86.44</td><td>2.13</td></tr><tr><td>SisyphusWorld</td><td>45.57</td><td>38.38</td><td>96.58</td><td>76.27</td><td>99.82</td><td>91.42</td><td>82.11</td><td>86.52</td><td>5.69</td></tr><tr><td>BWM-Fast</td><td>51.22</td><td>40.15</td><td>97.87</td><td>69.58</td><td>75.11</td><td>94.33</td><td>81.42</td><td>90.17</td><td>3.08</td></tr><tr><td>GigaWorld-0</td><td>48.81</td><td>40.15</td><td>38.65</td><td>39.45</td><td>30.89</td><td>77.34</td><td>73.62</td><td>88.15</td><td>12.84</td></tr><tr><td>Genie Envisioner</td><td>28.18</td><td>26.39</td><td>33.81</td><td>32.41</td><td>19.91</td><td>69.17</td><td>74.53</td><td>87.54</td><td>16.04</td></tr><tr><td>TesserAct</td><td>35.45</td><td>41.30</td><td>39.94</td><td>56.49</td><td>26.28</td><td>81.80</td><td>84.57</td><td>92.82</td><td>20.31</td></tr><tr><td>RoboMaster Vidar</td><td>36.05 41.75</td><td>37.87 39.87</td><td>29.83</td><td>41.74</td><td>16.07 17.03</td><td>70.91</td><td>82.91</td><td>90.80 78.24</td><td>28.28</td></tr><tr><td>Cosmos-Predict 2.5</td><td></td><td></td><td>61.10</td><td>21.05</td><td></td><td>79.11</td><td>70.37</td><td></td><td>20.57</td></tr><tr><td>(text) Cosmos-Predict 2.5</td><td>65.49</td><td>44.33</td><td>46.69</td><td>21.62</td><td>17.55</td><td>70.93</td><td>66.19</td><td>73.98</td><td>35.39</td></tr><tr><td>(action)</td><td>49.48</td><td>34.14</td><td>90.65</td><td>18.96</td><td>9.95</td><td>65.66</td><td>67.24</td><td>72.72</td><td>45.21</td></tr><tr><td>CtrlWorld</td><td>42.44</td><td>37.05</td><td>92.77</td><td>41.82</td><td>33.57</td><td>77.34</td><td>83.56</td><td>90.30</td><td>12.88</td></tr><tr><td>CogVideoX</td><td>36.23</td><td>36.30</td><td>94.84</td><td>31.47</td><td>21.77</td><td>73.30</td><td>80.97</td><td>88.38 78.69</td><td>33.79</td></tr><tr><td>IRASim</td><td>41.85</td><td>32.86</td><td>93.72</td><td>21.60</td><td>11.66</td><td>64.31</td><td>72.82</td><td></td><td>45.70</td></tr></table>

(b) Physics adherence, 3D accuracy, controllability, and overall score
<table><tr><td rowspan="2">Model</td><td colspan="2">Physics Adherence</td><td colspan="2">3D Accuracy</td><td colspan="2">Controllability</td><td rowspan="2">EWM Score-P</td></tr><tr><td>Interaction Quality</td><td>Trajectory Accuracy</td><td>Depth Accuracy</td><td>Perspectivity</td><td>Instruction Following</td><td>Semantic Alignment</td></tr><tr><td>DreamX-Phi- 1.0-FDM-0730</td><td>77.90</td><td>58.98</td><td>93.17</td><td>96.30</td><td>84.92</td><td>89.68</td><td>76.88</td></tr><tr><td>UNIS</td><td>87.30</td><td>41.89</td><td>85.25</td><td>98.84</td><td>93.86</td><td>89.35</td><td>73.64</td></tr><tr><td>SisyphusWorld</td><td>71.98</td><td>44.58</td><td>94.85</td><td>90.34</td><td>82.68</td><td>89.18</td><td>73.06</td></tr><tr><td>BWM-Fast</td><td>79.88</td><td>44.89</td><td>86.41</td><td>97.14</td><td>90.22</td><td>89.15</td><td>72.71</td></tr><tr><td>GigaWorld-0</td><td>56.20</td><td>17.07</td><td>63.78</td><td>78.06</td><td>56.14</td><td>86.64</td><td>53.85</td></tr><tr><td>Genie Envisioner</td><td>22.74</td><td>2.63</td><td>86.83</td><td>54.32</td><td>20.36</td><td>85.98</td><td>44.06</td></tr><tr><td>TesserAct</td><td>62.70</td><td>16.39</td><td>73.85</td><td>84.26</td><td>66.80</td><td>88.68</td><td>58.11</td></tr><tr><td>RoboMaster</td><td>54.80</td><td>12.35</td><td>83.77</td><td>77.94</td><td>51.14</td><td>87.78</td><td>53.48</td></tr><tr><td>Vidar</td><td>60.84</td><td>21.26</td><td>79.77</td><td>81.88</td><td>61.54</td><td>88.46</td><td>54.86</td></tr><tr><td>Cosmos-Predict 2.5 (text)</td><td>65.14</td><td>11.60</td><td>70.71</td><td>98.22</td><td>64.70</td><td>86.34</td><td>55.93</td></tr><tr><td>Cosmos-Predict 2.5</td><td>60.46</td><td>27.49</td><td>90.35</td><td>84.68</td><td>60.54</td><td>89.86</td><td>57.83</td></tr><tr><td>(action) CtrlWorld</td><td>62.62</td><td>48.20</td><td>93.25</td><td>83.66</td><td>67.68</td><td>88.68</td><td>63.72</td></tr><tr><td>CogVideoX</td><td>66.96</td><td>34.79</td><td>91.09</td><td>85.46</td><td>75.58</td><td>89.70</td><td>62.71</td></tr><tr><td>IRASim</td><td>62.76</td><td>35.92</td><td>93.50</td><td>84.16</td><td>65.40</td><td>89.43</td><td>59.63</td></tr></table>

The fixed WorldArena 1.0 leaderboard is led by UNIS (73.64), followed by SisyphusWorld (73.06) and BWM-Fast (72.71). In the same 15-metric format, our ofline DreamX-Phi-1.0-FDM-0730 result reaches 76.88, 3.24 points above the leading oficial entry in this snapshot.

## 6 Limitations

Our evaluation is limited to WorldArena and RoboTwin, with Track 2 covering only the Adjust Bottle task, so generalization to other tasks, embodiments, and real robots remains unverified. The leaderboard scores evaluate the full system and therefore do not isolate the contribution of individual components. Finally, DreamX-Phi predicts videos from externally provided actions rather than generating actions itself. Track 2 shows that the model can serve as a rollout environment for training a separate policy, but it does not evaluate DreamX-Phi as a closed-loop controller.

## 7 Conclusion

We presented DreamX-Phi 1.0, an action-conditioned video world model for robotic manipulation. The model addresses a central challenge in bimanual video prediction: a generated rollout must follow the prescribed motion of each arm while preserving scene geometry and the state of the manipulated object. To this end, DreamX-Phi combines arm-specific SE(3) conditioning with auxiliary depth and object-centric supervision. On the August 12, 2026 WorldArena 2.0 snapshot, our submission ranked first among 31 Track 1 entries with an EWMScore-P of 60.65. In Track 2, a policy trained using the submitted world model as its rollout environment achieved a 67.19% success rate on Adjust Bottle, tying for the second-highest score in the same snapshot. These results demonstrate strong system-level performance in video prediction and world-model-based policy training, while matched ablations are still needed to quantify the contribution of each component.

## 8 Future Work

The current DreamX-Phi 1.0 is formulated as a Forward Dynamics Model (FDM), which predicts future observations from an externally provided action sequence and does not generate actions itself. We will extend this framework to other model formulations. In particular, we plan to develop a joint World Action Mode that generates future video and robot action trajectories together. Training these outputs jointly should help align each proposed action with its predicted visual consequences. We will evaluate the resulting model in terms of video quality, action accuracy, action–video consistency, and closed-loop task success.

## Authors

Team Members. Team members are listed alphabetically by last name (and by first name where last names are identical). The ordering does not indicate relative contributions.

Rui Chen, Xiangxiang Chu, Geng Li, Jifan Li, Qingfeng Shi, Datao Tang, Jing Tang, Jun Wang, Pengfei Zhang.

## References

AgiBot World Team. AgiBot World 2026. Hugging Face dataset, 2026. URL https://huggingface.co/datasets/agibot-world /AgiBotWorld2026.

Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, et al. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025. URL https://arxiv.org/abs/2506.09985.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. SAM 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025. URL https://arxiv.org/abs/2511.16719.

Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, Deli Zhao, and Hao Chen. WorldVLA: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539, 2025. URL https://arxiv.org/abs/2506.21539.

Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. RoboTwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025. URL https://arxiv.org/abs/2506.18088.

Yixiang Chen, Peiyan Li, Yuan Xu, Qisen Ma, Jiabing Yang, Kai Wang, Jianhua Yang, Dong An, He Guan, Gaoteng Liu, Jianlou Si, Jun Huang, Jing Liu, Nianfeng Liu, Yan Huang, and Liang Wang. FlowWAM: Optical flow as a unified action representation for world action models. arXiv preprint arXiv:2607.13017, 2026. URL https://arxiv.org/abs/2607.13017.

DreamX Team, Yancheng Bai, Rui Chen, Xiangxiang Chu, et al. DreamX-World 1.0: A general-purpose interactive world model. arXiv preprint arXiv:2606.16993, 2026. URL https://arxiv.org/abs/2606.16993.

Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4D: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. URL https://arxiv.org/abs/2110.07058.

Jun Guo, Qiwei Li, Peiyan Li, Zilong Chen, Nan Sun, Yifei Su, Heyun Wang, Yuan Zhang, Xinghang Li, and Huaping Liu. Unified 4D world action modeling from video priors with asynchronous denoising. arXiv preprint arXiv:2604.26694, 2026 URL https://arxiv.org/abs/2604.26694.

Siqiao Huang, Jialong Wu, Qixing Zhou, Shangchen Miao, and Mingsheng Long. Vid2World: Crafting video difusion models to interactive world models. arXiv preprint arXiv:2505.14357, 2025. URL https://arxiv.org/abs/2505.14357.

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. DROID: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024. URL https://arxiv.org/abs/2403.12945.

Byungjun Kim, Taeksoo Kim, Hyunsoo Cha, and Hanbyul Joo. Robot-factored world models via robot rendering. arXiv preprint arXiv:2607.22535, 2026a. URL https://arxiv.org/abs/2607.22535.

Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, and Jinwei Gu. Cosmos Policy: Fine-tuning video models for visuomotor control and planning. In International Conference on Learning Representations, 2026b. URL https://arxiv.org/abs/2601.16163.

Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, Yujun Shen, and Yinghao Xu. Causal world modeling for robot control. arXiv preprint arXiv:2601.21998, 2026. URL https: //arxiv.org/abs/2601.21998.

Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, and Angjoo Kanazawa. Cameras as relative positional encoding. Advances in Neural Information Processing Systems, 38, 2025a. URL https://arxiv.org/abs/2507.10496.

Shuang Li, Yihuai Gao, Dorsa Sadigh, and Shuran Song. Unified video action model. arXiv preprint arXiv:2503.00200, 2025b. URL https://arxiv.org/abs/2503.00200.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2210.02747.

Takeru Miyato, Bernhard Jaeger, Max Welling, and Andreas Geiger. GTA: A geometry-aware attention mechanism for multi-view transformers. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2310.10375.

NVIDIA. Cosmos 3: Omnimodal world models for physical AI. arXiv preprint arXiv:2606.02800, 2026. URL https: //arxiv.org/abs/2606.02800.

NVIDIA Corporation. Cosmos3-DROID. Hugging Face dataset, 2026. URL https://huggingface.co/datasets/nvidia/Cosmo s3-DROID.

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π<sub>0.5</sub>: A vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025. doi: 10.48550/arXiv.2504.16054. URL https://arxiv.org/abs/2504.16054.

Marc Rigter, Tarun Gupta, Agrin Hilmkil, and Chao Ma. AVID: Adapting video difusion models to world models. Reinforcement Learning Journal, 6:737–764, 2025. URL https://arxiv.org/abs/2410.12822.

Yu Shang, Zhuohang Li, Yiding Ma, Weikang Su, Xin Jin, Ziyou Wang, Lei Jin, Xin Zhang, Yinzhou Tang, Haisheng Su, et al. WorldArena: A unified benchmark for evaluating perception and functional utility of embodied world models. arXiv preprint arXiv:2602.08971, 2026a. URL https://arxiv.org/abs/2602.08971.

Yu Shang, Yinzhou Tang, Yiding Ma, Zhuohang Li, Lei Jin, Weikang Su, Xin Jin, Zhaolu Wang, Ziyou Wang, Xin Zhang, et al. WorldArena 2.0: Extending embodied world model benchmarking on modality, functionality and platform. arXiv preprint arXiv:2605.17912, 2026b. URL https://arxiv.org/abs/2605.17912.

Team Wan. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. URL https://arxiv.org/abs/2503.20314.

Yang Tian, Yuyin Yang, Yiman Xie, Zetao Cai, Xu Shi, Ning Gao, Hangxu Liu, Xuekun Jiang, Zherui Qiu, Feng Yuan, et al. InternData-A1: Pioneering high-fidelity synthetic data for pre-training generalist policy. arXiv preprint arXiv:2511.16651, 2025. URL https://arxiv.org/abs/2511.16651.

Wan Team. Wan2.2-TI2V-5B. Model card and released model weights, 2025. URL https://huggingface.co/Wan-AI/Wan2.2-T I2V-5B.

Lirui Wang, Kevin Zhao, Chaoqi Liu, and Xinlei Chen. Learning real-world action-video dynamics with heterogeneous masked autoregression. arXiv preprint arXiv:2502.04296, 2025. URL https://arxiv.org/abs/2502.04296.

Jialong Wu, Shaofeng Yin, Ningya Feng, Xu He, Dong Li, Jianye Hao, and Mingsheng Long. iVideoGPT: Interactive VideoGPTs are scalable world models. Advances in Neural Information Processing Systems, 37, 2024. URL https://arxiv.org/abs/2405 .15223.

Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, Zhe Li, Kai Zhu, Hongyu Wu, Yiheng Liu, et al. RoboCOIN: An open-sourced bimanual robotic data collection for integrated manipulation. arXiv preprint arXiv:2511.17441, 2025. URL https://arxiv.org/abs/2511.17441.

Zhuoyuan Wu and Jun Gao. OSCAR: Omni-embodiment action-conditioned world model for robotics. arXiv preprint arXiv:2606.04463, 2026. URL https://arxiv.org/abs/2606.04463.

Sherry Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Leslie Kaelbling, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. In International Conference on Learning Representations, 2024. URL https: //arxiv.org/abs/2310.06114.

Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi Jim Fan, and Joel Jang. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026. URL https://arxiv.org/abs/2602.15922.

Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Frédo Durand, and William T. Freeman. Improved distribution matching distillation for fast image synthesis. In Advances in Neural Information Processing Systems, volume 37, pages 47455–47487, 2024a. URL https://arxiv.org/abs/2405.14867.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman, and Taesung Park. One-step difusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6613–6623, 2024b. URL https://arxiv.org/abs/2311.18828.

Fangqi Zhu, Hongtao Wu, Song Guo, Yuxiao Liu, Chilam Cheang, and Tao Kong. IRASim: A fine-grained world model for robot manipulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9834–9844, 2025. URL https://arxiv.org/abs/2406.14540.