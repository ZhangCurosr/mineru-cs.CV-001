# DreamFly: Causal Memory and Receding-Horizon Difusion Planning for Aerial Vision-Language Navigation

Yan Deng<sup>∗1</sup> and Fei Xu<sup>2</sup>

<sup>1</sup> School of Electronic Information Engineering, Xi’an Technological University, Xi’an, China

<sup>2</sup> School of Computer Science and Engineering, Xi’an Technological University, Xi’an, China

## Abstract

Aerial vision-language navigation (VLN) requires an embodied agent to integrate visual evidence over time, plan future actions, and determine when it has reached a navigation goal under partial observability. Although recent VLA models ofer a promising perception-to-action paradigm, adapting them to aerial navigation remains challenging. Policies conditioned primarily on the current observation may lose track of previously observed landmarks; single-step action prediction provides limited lookahead; and implicit termination through action generation may be unreliable.

To address these challenges, we propose DreamFly, a difusion-based aerial VLN framework built on Dream-VLA. DreamFly first introduces a causally aligned historical memory that augments the current visual representation with evidence drawn exclusively from observations preceding the current decision step, thereby enabling temporal reasoning without leaking future information. Leveraging the bidirectional difusion backbone, we further formulate navigation as receding-horizon difusion planning, where the policy jointly predicts a K-step action chunk but executes only the first action before replanning from the next observation. This plan-K, execute-one strategy treats predicted future actions as auxiliary planning targets while preserving closed-loop visual feedback. Finally, we introduce LiteStop, a lightweight termination module that estimates the stop probability directly from the action logits at the initial all-mask state, thereby decoupling explicit termination from action generation.

Together, these components establish a causal closed-loop cycle of observation, memory retrieval, difusion planning, termination assessment, execution, and memory update. Experiments on the OpenFly benchmark demonstrate consistent improvements in both seen and unseen environments. DreamFly achieves 32.04%/29.46% SR and 28.22%/23.54% SPL on the test-seen/testunseen splits, respectively, outperforming all compared methods on both metrics while also attaining the lowest navigation error. These results highlight the efectiveness of jointly modeling historical context, future action structure, and explicit termination for aerial vision-language navigation.

Keywords: Aerial Vision-Language Navigation; Vision-Language-Action Models; Difusion Policy; Long-Term Visual Memory

## 1 Introduction

Vision-and-Language Navigation (VLN) requires embodied agents to interpret natural-language instructions and navigate to specified goals using sequentially acquired visual observations [1]. In contrast to static visual recognition and single-step action classification, VLN is inherently a partially observable, closed-loop sequential decision problem: an action taken at the current step afects the agent’s subsequent state and visual observation, while newly acquired environmental evidence updates its estimates of task progress, spatial context, and direction of travel. A reliable VLN policy therefore requires more than vision–language grounding at an isolated decision step. It must maintain temporal consistency across historical visual evidence, the current observation, planned future actions, and executed actions throughout its interaction with the environment.

Aerial vision-language navigation extends VLN to unmanned aerial vehicles (UAVs), requiring agents to navigate autonomously through large-scale 3D environments by following natural-language instructions [2]. However, aerial VLN cannot be treated as a direct transfer of conventional ground based VLN to a flying platform. Ground agents typically move on approximately two-dimensional traversable surfaces, whereas UAVs must jointly coordinate horizontal displacement, vertical motion, and viewpoint adjustment. Changes in altitude afect not only the UAV’s spatial position but also the spatial extent of the visible scene, the apparent scale of landmarks, the level of visible detail, and obstacle visibility, thereby tightly coupling horizontal and vertical navigation decisions [3]. Moreover, urban aerial environments often span large areas and contain complex 3D structures and substantial visual occlusion, such that a single egocentric observation captures only a limited portion of the surrounding scene. Consequently, local decision errors can compound over time. These characteristics impose stringent requirements on historical context modeling, 3D spatial reasoning, cross-modal grounding, and online error correction in aerial VLN [2], [3], [4].

Early VLN methods commonly relied on sequence-to-sequence architectures, typically implemented as recurrent policies, to autoregressively predict the next navigation action from the language instruction, current visual observation, and recurrent hidden state [1], [5]. With advances in vision–language pretraining and Transformer-based architectures, subsequent studies developed recurrent vision–language Transformers, history-aware representations, and explicit environmental memories to support reasoning across successive decision steps. For example, Recurrent VLN-BERT maintains a recurrent state representation, whereas HAMT employs a hierarchical Transformer to encode navigation history, thereby facilitating multimodal reasoning over extended trajectories [6], [7]. In aerial VLN, recent approaches have further explored multidirectional view selection, bird’seye-view feature maps, keyframe selection, and compression of historical visual observations to mitigate restricted egocentric visibility, redundant observations, and dificulties in tracking naviga tion state [3], [4], [8]. Together, these studies underscore the importance of historical information for navigation. However, historical modeling depends not only on how past observations are represented, but also on precisely when they become available to the policy during closed-loop interaction.

More recently, vision–language–action (VLA) models have emerged as a promising paradigm for language-conditioned end-to-end control. By formulating actions as discrete tokens or continuous values, VLA models adapt pretrained vision–language models to generate robot actions conditioned on language instructions and visual observations [9], [10]. Action chunking captures short-horizon temporal structure by jointly predicting a sequence of actions, while difusion-based policies model multimodal action distributions through iterative denoising and can be deployed with receding-horizon replanning [11]. Dream-VLA, for example, employs a difusion Transformer to jointly predict action chunks, thereby capturing dependencies among actions over a short future horizon [12]. Recent studies of aerial navigation have likewise explored end-to-end VLA policies, difusion-based action modeling, and joint world–action prediction, highlighting the potential of explicitly predicting future states or action sequences for navigation in complex 3D environments [13], [14], [15], [16]. Several of these approaches seek to enhance planning by forecasting future world states or visual observations. However, how to exploit dependencies among predicted future actions over a short horizon while retaining closed-loop replanning from newly acquired observations

remains underexplored.

Despite recent advances in historical modeling, 3D reasoning, and multi-step action prediction, three temporal aspects of aerial VLN remain insuficiently addressed within a unified closed-loop decision process. First, an online historical memory requires an explicit temporal boundary. Here, causal refers to temporally ordered information access rather than causal relations between variables: at decision step t, historical information is restricted to observations acquired before step t. Accord ingly, the historical memory $M _ { < t }$ is read before the current observation is written to it, preventing current-step information from entering the historical branch. Applying this read-before-write order ing during both training and deployment gives $M _ { < t }$ a well-defined temporal boundary and ensures consistent information availability across the two phases. This temporal notion of causality should not be conflated with covariate shift between expert and policy-induced state distributions in imi tation learning [17]. It does not mitigate the covariate shift arising from policy execution; rather, it constrains when observations enter the historical memory.

Second, multi-step action prediction should provide lookahead for the current decision without sacrificing closed-loop execution. Replanning after each executed action allows the agent to exploit dependencies among predicted future actions without committing to an open-loop action chunk, thereby retaining responsiveness to newly acquired observations [11]. Third, termination carries markedly diferent consequences from ordinary motion actions. Motion errors can often be corrected through subsequent actions, whereas a premature stop irreversibly ends the navigation episode. Treating termination and motion actions under the same prediction and calibration mechanism can therefore obscure the asymmetric risks of premature and delayed stopping [18]. Together, these considerations motivate the explicit coordination of historical information, future-action planning, and termination within a temporally consistent closed-loop decision process.

To address these issues, we propose DreamFly, a temporally consistent, closed-loop aerial VLN framework built on discrete difusion-based action generation. First, DreamFly introduces a causally aligned historical memory constructed exclusively from visual observations acquired before the current decision step. Under a read-before-write protocol, $M _ { < t }$ is held fixed during decision step t and updated with the current observation only after the current decision, for use at subsequent steps. The resulting historical representation is fused with the current visual features through a lightweight memory-conditioning module, allowing the policy to reason jointly over the language instruction, historical visual evidence, and current observation.

Building on the memory-conditioned representation, DreamFly performs receding-horizon diffusion planning by jointly predicting a chunk of K discrete actions that comprises the current action and captures dependencies among actions over a short future horizon. During online inference, DreamFly follows a plan-K, execute-one strategy: it executes only the first action and replans upon receiving the next observation. In this way, the future positions in the action chunk serve as auxiliary prediction targets for the current decision, while execution remains closed loop.

Finally, we introduce LiteStop, a lightweight module that explicitly predicts termination. Rather than treating termination as an ordinary motion action, LiteStop estimates the probability of stopping from the action-token logit grid $\mathbf { H } _ { t } ^ { ( 0 ) }$ extracted from the difusion policy at its initial all-mask state. LiteStop is trained separately while the navigation policy remains frozen, providing a dedicated termination objective without modifying the learned motion policy. Together, causally aligned historical memory, receding-horizon difusion planning, and explicit termination form a temporally consistent, closed-loop navigation process, as illustrated in Fig. 1.

We make the following three main contributions:

1. We introduce a causally aligned historical memory for aerial VLN with an explicit temporal information boundary. DreamFly constructs its historical memory exclusively from observations acquired before the current decision step and adopts a read-before-write protocol that prevents current-step information from entering the historical branch. A lightweight gated cross-attention module selectively integrates task-relevant historical evidence into the current visual representation, thereby supporting temporally consistent reasoning under partial observability.

![](images/1dd0216ee39c5c9cd6cf790720fa3bc409002642550a53ffc1f7fbf846b42363.jpg)  
Figure 1: DreamFly at a Glance. At decision step t, DreamFly conditions on the language instruction, current aerial observation, and a historical memory containing only observations acquired before step t to generate a K-step action plan. DreamFly executes only the first action, acquires a new observation, and replans at the next decision step, while LiteStop separately estimates the probability of termination. Dashed actions denote planned future actions that are not executed at the current decision step.

2. We propose receding-horizon difusion planning that exploits dependencies among predicted future actions while preserving closed-loop control. DreamFly jointly predicts the current action and a short sequence of future actions as a chunk of K discrete actions, using valid-prefix supervision and horizon-aware loss weighting to emphasize the immediately executable action. During inference, DreamFly executes only the first action and replans upon receiving the next observation, allowing the predicted future actions to serve as auxiliary planning targets while preserving closed-loop execution.

3. We develop LiteStop, an explicit and decoupled termination module for aerial VLN. Rather than coupling termination with motion-action generation, LiteStop estimates the probability of stopping from the action logits produced by the difusion policy at its initial all-mask state and is trained separately while the navigation policy remains frozen. This decoupling provides a dedicated termination objective without modifying the learned motion policy.

## 2 Related Work

## 2.1 Aerial Vision-Language Navigation

Aerial VLN requires a UAV to navigate toward specified goals in large-scale 3D environments by following natural-language instructions and relying on sequentially acquired egocentric observations. Early studies focused primarily on task formulation and benchmark construction. AVDN intro duced an interactive UAV navigation setting based on asynchronous multi-round dialogue, enabling the agent to request and use additional linguistic guidance during navigation [19]. AerialVLN subsequently established a setting for instruction-guided UAV navigation in continuous urban envi ronments, together with systematic data-collection and evaluation protocols [2]. Subsequent bench marks have progressively extended aerial VLN to larger-scale and more realistic settings. CityNav extends aerial navigation to city-scale target localization using real-world urban aerial data and a combination of visual and geographic cues [20]. OpenUAV incorporates continuous flight trajecto ries, realistic flight dynamics, and complex task scenarios, providing a setting closer to practical UAV control than discrete simulated navigation [21]. OpenFly introduces an automated data-generation pipeline that integrates rendered and real-world scenes, thereby broadening the range of training trajectories and visual domains [4]. More recently, AirNav constructs a large-scale benchmark from real-world urban aerial data paired with diverse natural-language navigation instructions [22]. Col lectively, these eforts have expanded aerial VLN from early simulated formulations to continuous, large-scale, and increasingly realistic navigation settings.

Building on these benchmarks, recent aerial VLN methods have primarily pursued two complementary directions: structured spatial and historical modeling, and reasoning with foundation models. One line of work explicitly structures spatial relations and observation histories to support navigation in large-scale environments. STMR projects instruction-relevant visual semantics into a semantic–topological–metric representation and uses a large language model to reason over this representation and predict navigation actions [23]. OpenFly-Agent selects informative keyframes from long observation histories to provide compact visual context for each navigation decision [4]. CityNavAgent uses hierarchical semantic planning to decompose complex instructions into subgoals at multiple levels of granularity while maintaining a global topological memory of explored trajecto ries [20]. HETT combines a historical grid representation with a two-stage, coarse-to-fine decision process that links global target localization to local action selection [24]. LookasideVLN incorporates directional semantics and lookahead path reasoning to improve landmark selection and spatial reasoning in complex urban environments [25]. Collectively, these methods demonstrate the value of structured spatial representations and accessible navigation histories for multi-step decision making in large-scale aerial environments.

A complementary line of work leverages vision-language foundation models and large language models to enhance cross-modal understanding and high-level reasoning. The navigation model accompanying OpenUAV employs a multimodal language model to jointly process multi-view ob servations, task descriptions, and auxiliary information, generating flight trajectories hierarchically [21]. FlightGPT combines supervised fine-tuning with reinforcement learning to strengthen vision language reasoning and navigation decision making [26]. FineCog-Nav decomposes the navigation process into fine-grained modules for language understanding, perception, memory, imagination, and decision making, investigating how coordinated foundation-model reasoning can support zeroshot UAV navigation [27]. Overall, prior work has made substantial progress in data scale, spatial representation, historical-context modeling, and foundation-model reasoning. However, how to coordinate the distinct temporal roles of historical context, multi-step action planning, and navigation termination within a unified closed-loop decision process remains underexplored.

## 2.2 Historical Context Modeling and Memory in VLN

Because each visual observation captures only a partial view of the environment, VLN policies must use historical information to retain spatial context from earlier observations, account for prior actions, and track instruction-following progress. Early approaches commonly summarized navigation history in recurrent states. Recurrent VLN-BERT summarizes navigation history in a compact recurrent state within a cross-modal Transformer rather than retaining explicitly addressable repre sentations of individual past observations [6]. In contrast, HAMT uses a hierarchical Transformer to encode the full history of panoramic observations, retaining detailed temporal context, although the amount of historical input and potential redundancy can increase as the trajectory grows [7]. These approaches illustrate two diferent points in the trade-of between compact representation and detailed history preservation.

To support explicit storage and retrieval of past visual evidence, other studies developed spatially structured memory representations. Structured Scene Memory organizes visual and geometric cues acquired during navigation into an explicit scene representation and employs an adaptive collection-and-read mechanism to support reasoning over time [28]. GridMM projects historical observations onto a dynamically expanding egocentric bird’s-eye-view grid and aggregates instructionrelevant visual evidence across spatial regions [29]. These approaches organize historical evidence through scene connectivity or relative spatial location, making spatial relations explicit for multistep reasoning and planning. Their memory construction is consequently coupled with the spatial organization or projection mechanism adopted by the navigation policy.

The characteristics of aerial VLN, including long flight trajectories, frequent viewpoint changes, and substantial visual redundancy, have further motivated the development of compact representations of observation history. OpenFly-Agent employs adaptive frame-level token sampling to retain informative historical visual content while reducing redundancy among consecutive aerial observations [4]. LongFly compresses historical images into a fixed-length context representation through slot-based aggregation and integrates the resulting representation with spatiotemporal trajectory encoding and prompt-guided multimodal fusion for navigation decision-making [8]. These studies highlight frame selection and context compression as practical strategies for limiting the growth of historical input. However, these mechanisms focus primarily on what historical information is retained and how much of it is preserved, leaving the temporal boundary that determines when an observation becomes available to the historical branch less explicitly characterized.

## 2.3 Multi-Step Action Prediction and Generation

Research on action chunking has examined not only how to jointly predict multiple future actions but also how to execute and update action chunks under closed-loop control. ACT uses a Transformer to predict an action sequence and applies temporal ensembling to aggregate overlapping predictions made at successive decision steps, thereby promoting temporally consistent continuous control [30]. Bidirectional Decoding employs a test-time strategy designed to balance consistency within an action chunk against responsiveness to newly observed states, thereby supporting adaptive closed-loop execution [31]. Real-Time Chunking addresses the deployment of difusion- and flow-based policies by asynchronously generating the next action chunk while the current chunk is being executed, thereby reducing control delays caused by inference latency [32].

Difusion-based policies provide a generative framework for multi-step action prediction. Diffusion Policy models continuous action trajectories through iterative denoising and deploys the predicted trajectories within a receding-horizon control loop with online replanning [11]. In discrete action spaces, Dream-VL introduces a difusion-based vision–language backbone with bidirectional attention, while Dream-VLA extends this backbone through continual pretraining on robotic data and jointly denoises multiple masked action tokens [33].

Beyond direct action generation, another line of work augments planning with predicted future states or learned world dynamics. DreamVLA integrates compact world knowledge into multi-step action generation [12], WorldVLN models latent world–action transitions [14], and ImagineUAV pre dicts future visual observations to guide trajectory generation [15]. Collectively, these approaches demonstrate the value of modeling both future actions and possible environmental evolution for multi-step planning. However, how discrete difusion policies can exploit short-horizon structure among jointly predicted actions in aerial VLN, while executing only the current action and replan ning from each new observation, remains comparatively underexplored.

## 3 Method

## 3.1 Problem Formulation

Given a natural-language instruction $I = ( w _ { 1 } , w _ { 2 } , \ldots , w _ { N _ { I } } )$ , an aerial agent starts from an initial pose $s _ { 0 } = ( p _ { 0 } , \psi _ { 0 } )$ in a 3D environment, where $p _ { 0 } = ( x _ { 0 } , y _ { 0 } , z _ { 0 } )$ denotes its position and ψ<sub>0</sub> denotes its heading. At each decision step t, the agent receives an egocentric RGB observation $O _ { t }$ and selects an action $a _ { t }$ from a discrete action space A. Executing $a _ { t }$ moves the agent to a new pose $s _ { t + 1 }$ and produces the next observation $O _ { t + 1 }$

The action space consists of Stop, forward motion with diferent step sizes, left and right turns, vertical movement, and lateral movement. The navigation episode terminates when the agent selects Stop or reaches the maximum number of decision steps. Given the ground-truth destination $p ^ { * }$ , an episode is considered successful if the Euclidean distance between the agent’s final position $p _ { T }$ and the destination is less than 20 m:

$$
\lVert p _ { T } - p ^ { * } \rVert _ { 2 } < 2 0 \ \mathrm { m } .
$$

## 3.2 Framework Overview

Fig. 2 illustrates the overall architecture of DreamFly, an aerial vision-language navigation framework built upon Dream-VLA [33]. We adopt Dream-VLA as the policy backbone because its perception-to-action formulation aligns naturally with aerial VLN, directly mapping visual observations and language instructions to executable actions. Compared with autoregressive VLA back bones, Dream-VLA employs bidirectional difusion modeling to support action chunking and parallel action prediction, providing a suitable foundation for receding-horizon planning.

Unlike the original VLA setting, aerial navigation requires retaining historical visual evidence and continually replanning as new observations arrive. DreamFly therefore introduces three key designs. First, a causally aligned historical memory $M _ { < t }$ is constructed only from observations acquired before the current decision step and conditions the current visual representation, preserving a well-defined temporal information boundary while retaining previously observed landmarks. Second, DreamFly performs receding-horizon difusion planning by jointly predicting a K-step action chunk. The predicted future actions provide short-horizon planning structure without being committed to consecutive execution. If termination is not triggered, only the first action is executed before the policy replans from the next observation. Third, LiteStop estimates termination from the initial all-mask action logits and provides a separate pre-action termination decision. Together, these components establish a temporally consistent closed-loop process that integrates historical reasoning, receding-horizon future-action planning, and explicit termination control.

![](images/6da855943c85aeb79ab1c490f59c6dba7d316e08f78a88072bfc96b68657c33b.jpg)  
Figure 2: Overview of the proposed DreamFly framework. At each decision step, DreamFly conditions action prediction on the current observation, navigation instruction, and a causally aligned memory constructed from prior observations. The Dream-VLA backbone predicts a K-step action chunk through bidirectional difusion, while LiteStop estimates termination from the initial all-mask action logits. If termination is not triggered, only the first action in the predicted chunk is executed before replanning from the next observation. Information derived from the current observation can enter the historical memory only for subsequent decisions.

## 3.3 Causally Aligned Historical Memory

During continuous aerial navigation, previously observed landmarks and scene cues may leave the current field of view while remaining relevant to subsequent decisions. Retaining all past obser vations would preserve such information, but would also introduce substantial redundancy and cause the visual input to grow with trajectory length. To maintain useful historical evidence under a bounded input budget, DreamFly compresses instruction-relevant information from past observations into a fixed-capacity historical memory. A short candidate FIFO and active tracks are maintained internally to accumulate evidence across recent observations, while only the long-term memory slots are exposed to the navigation policy.

At decision step $t ,$ let $O _ { \tau }$ denote the observation acquired at step τ . The historical memory available to the policy is defined as

$$
M _ { < t } = \mathcal { F } _ { \mathrm { m e m } } \left( I , \left( O _ { \tau } \right) _ { \tau < t } \right) ,
$$

where $\mathcal { F } _ { \mathrm { m e m } }$ denotes the fixed memory-construction operator. This definition constrains only the historical branch: the current observation $O _ { t }$ remains directly available for action prediction, while information derived from $O _ { t }$ can enter the historical memory only from subsequent decision steps. Instruction-conditioned candidate construction. For each observation, a frozen CLIPSeg dense router and a frozen OWLv2 region router extract complementary visual candidates conditioned on the complete navigation instruction I. To accommodate the finite text context of the routers without truncating the instruction, the full instruction is covered by fixed overlapping token windows. CLIPSeg provides dense instruction-relevant responses together with a visual feature grid, whereas OWLv2 proposes spatially localized regions. Rather than combining features from the two models, OWLv2 regions are represented directly in the frozen CLIPSeg visual feature space. For an OWLv2 region $b ,$ its visual representation is obtained as

$$
\mathbf { f } ( b ) = \mathrm { N o r m } \left( \sum _ { g } \mathrm { a r e a } \left( G _ { g } \cap b \right) \mathbf { v } _ { g } \right) ,
$$

where g indexes the CLIPSeg grid cells, $G _ { g }$ denotes the spatial region of the g-th cell, ${ \mathbf { v } } _ { g }$ is its normalized visual feature, and Norm(·) denotes $L _ { 2 }$ normalization. CLIPSeg candidates directly use their corresponding grid features. Candidates are subsequently consolidated only when they exhibit suficient spatial overlap and visual similarity.

Evidence-driven long-term memory. The resulting candidates are associated with active tracks over recent observations according to visual and spatial consistency, allowing evidence for the same visual content to accumulate across observations. Candidates receiving repeated and stable support become eligible for persistent promotion, whereas candidates without suficient repeated support may enter memory through single-observation promotion when they satisfy the confidence, regionvalidity, score-separation, and novelty criteria. These complementary mechanisms preserve both stable cross-observation evidence and informative content that may be visible only briefly.

Eligible candidates are ranked by their write utility, and at most two long-term write candidates are selected at each decision step. For each selected candidate, if a compatible slot is found, that slot is updated with the candidate; otherwise, the candidate is assigned to an empty slot or, when the memory is full, replaces an existing slot according to the retention criterion.

Each valid slot maintains an anchor and an optional prototype. The anchor stores the visual feature of the concrete instance with the highest historical write utility, whereas the prototype represents accumulated compatible evidence once stable repeated support has been established. A candidate admitted through single-observation promotion may therefore form a valid anchor-only slot; such a slot is promoted to include a prototype when it is subsequently matched by a persistent write supported by stable cross-observation evidence.

At decision step t, the memory adapter represents the policy-facing memory as

$$
M _ { < t } = \left\{ \left( \mathbf { r } _ { < t } ^ { j } , \mu _ { < t } ^ { j } \right) \right\} _ { j = 1 } ^ { 1 6 } , \qquad \mathbf { r } _ { < t } ^ { j } = \left[ \mathbf { e } _ { < t } ^ { j , \mathrm { a n c } } ; \mathbf { e } _ { < t } ^ { j , \mathrm { p r o } } ; \rho _ { < t } ^ { j } ; \log \left( 1 + \delta _ { < t } ^ { j } \right) \right] ,
$$

where $j$ indexes the memory slots, $\mu _ { < t } ^ { j } \in \{ 0 , 1 \}$ indicates slot validity, ${ \bf e } _ { < t } ^ { j , \mathrm { a n c } }$ and ${ \bf e } _ { < t } ^ { j , \mathrm { p r o } }$ denote the anchor and prototype features, respectively, $\rho _ { < t } ^ { j } \in \{ 0 , 1 \}$ indicates prototype presence, and $\delta _ { < t } ^ { j }$ is the number of decision steps since the slot was last updated. An anchor-only slot remains valid; when no prototype is available, its prototype component is zero-filled and $\rho _ { < t } ^ { j } = 0$ . For an invalid slot with $\boldsymbol { \mu } _ { < t } ^ { j } = 0$ , the complete input representation $\mathbf { r } _ { < t } ^ { j }$ is zero-filled and is excluded from the key/value dimension of cross-attention by the slot-validity mask.

Memory-conditioned visual representation. DreamFly conditions the current visual representation on historical memory rather than appending memory slots as additional multimodal tokens. Let

$$
\mathbf { Z } _ { t } = \operatorname { P r o j e c t o r } \left( \operatorname { V i s i o n E n c o d e r } ( O _ { t } ) \right)
$$

denote the current image tokens. Each slot representation $\mathbf { r } _ { < t } ^ { j }$ is mapped by a memory slot encoder $\phi ( \cdot )$ , and the resulting slot embeddings are stacked in slot order:

$$
\mathbf { E } _ { < t } = \left[ \begin{array} { c } { \phi ( \mathbf { r } _ { < t } ^ { 1 } ) } \\ { \vdots } \\ { \phi ( \mathbf { r } _ { < t } ^ { 1 6 } ) } \end{array} \right] .
$$

Using the current image tokens as queries and the historical slot embeddings as keys and values, DreamFly retrieves historical context through

$$
\begin{array} { r } { \mathbf { C } _ { t } = \mathrm { M H A } \left( \mathbf { Z } _ { t } W _ { Q } , \mathbf { E } _ { < t } W _ { K } , \mathbf { E } _ { < t } W _ { V } ; \pmb { \mu } _ { < t } \right) , } \end{array}
$$

where $W _ { Q } , W _ { K }$ , and $W _ { V }$ are learned projection matrices, $\pmb { \mu } _ { < t } = ( \mu _ { < t } ^ { 1 } , \dots , \mu _ { < t } ^ { 1 6 } )$ is the slot-validity mask applied along the key/value slot dimension, and $\mathbf { C } _ { t }$ denotes the retrieved historical context. All valid slots participate in the masked cross-attention, allowing the current visual tokens to adaptively retrieve relevant historical evidence without hand-crafted top-k selection.

The retrieved context is incorporated into the current visual tokens through a gated residual connection:

$$
\mathbf { G } _ { t } = 1 + \operatorname { t a n h } \left( \mathbf { Z } _ { t } W _ { G } + \mathbf { b } _ { G } \right) , \qquad \widetilde { \mathbf { Z } } _ { t } = \mathbf { Z } _ { t } + \mathbf { M } _ { \mathrm { i m g } } \odot \mathbf { G } _ { t } \odot \left( \mathbf { C } _ { t } W _ { O } \right) ,
$$

where $\mathbf { G } _ { t }$ is the learned gate, $W _ { G }$ and $\mathbf { b } _ { G }$ are its parameters, $W _ { O }$ is the residual output projection, $\mathbf { M } _ { \mathrm { i m g } }$ is the valid-image-token mask, and $\odot$ denotes element-wise multiplication. The residual output projection $W _ { O }$ and the gating parameters $W _ { G }$ and $\mathbf { b } _ { G }$ are zero-initialized. Since $W _ { O } = 0$ at initialization, the adapter initially implements the identity mapping $\widetilde { \mathbf Z } _ { t } = \mathbf Z _ { t }$ , while the initial gate value is $\mathbf { G } _ { t } = 1$ . The resulting memory-conditioned tokens $\widetilde { \mathbf { Z } } _ { t }$ eare then processed by the Dream-VLA ebackbone together with the navigation instruction for action prediction.

Causal alignment across training and deployment. Training and closed-loop deployment share the same frozen memory-construction mechanism and temporal visibility boundary, but operate on diferent observation histories. Training uses prefix memories constructed along expert trajectories, whereas deployment initializes a fresh online memory for each rollout and updates it from observations actually acquired by the agent. Consequently, the resulting memory states may difer, while the causal constraint remains unchanged: the memory $M _ { < t }$ used at step t contains only information acquired before that decision step.

## 3.4 Receding-Horizon Difusion Planning

Predicting only the next action provides an immediate control decision but does not explicitly represent the short-horizon action structure surrounding that decision. DreamFly therefore extends single-step prediction to a fixed-horizon discrete action chunk and exploits the bidirectional Dream VLA backbone to jointly model the current action together with $K - 1$ near-future actions. The predicted chunk serves as a structured short-horizon planning representation rather than an openloop command sequence. At each decision step, DreamFly generates the complete chunk. LiteStop may terminate the episode before action execution; otherwise, at most the leading action is handled, and the policy replans after a non-terminal motion produces new visual feedback.

Discrete action-chunk formulation. Given the memory-conditioned visual representation $\widetilde { \mathbf { Z } } _ { t }$ and the navigation instruction $I ,$ DreamFly predicts a length-K discrete action chunk

$$
\begin{array} { r } { \hat { \mathbf { a } } _ { t } = \left[ \hat { a } _ { t } ^ { 0 } , \hat { a } _ { t } ^ { 1 } , \dots , \hat { a } _ { t } ^ { K - 1 } \right] , \qquad \hat { a } _ { t } ^ { h } \in \mathcal { A } , } \end{array}
$$

where A is the discrete aerial action space and K is the planning horizon. The first slot $\hat { a } _ { t } ^ { 0 }$ corresponds to the current control decision, whereas the remaining slots $\hat { a } _ { t } ^ { 1 : K - 1 }$ represent short-horizon future actions predicted under the same current context. These future predictions form part of the structured short-horizon plan but are not committed commands to be executed consecutively.

Let V denote the full Dream-VLA vocabulary, and let

$$
\chi : { \mathcal { A } } \hookrightarrow \mathcal { V }
$$

denote the injective mapping from each discrete action to its dedicated action token. Its image $\chi ( \mathcal { A } ) \subset \mathcal { V }$ forms the dedicated action-token set. This distinction allows us to use environmentlevel actions in the planning formulation while explicitly identifying their corresponding vocabulary targets during model training and difusion generation.

Joint masked action learning. During training, the action chunk is supervised by the sufix beginning at decision step t of the stored action sequence. Let $( a _ { 0 } ^ { \star } , \ldots , a _ { T - 1 } ^ { \star } )$ denote the length-T action sequence stored for a training trajectory. At a decision step $t \in \{ 0 , \ldots , T - 1 \}$ , the number of available actions within the prediction horizon is

$$
L _ { t } = \operatorname* { m i n } ( K , T - t ) , \qquad v _ { t , h } = \mathbb { I } [ h < L _ { t } ] ,
$$

where $v _ { t , h } \in \{ 0 , 1 \}$ indicates whether the h-th action slot belongs to the available supervision prefix. To preserve a fixed-length template near the end of a trajectory, positions beyond this prefix are padded with the Stop action. We denote the padded target by

$$
\bar { a } _ { t , h } ^ { \star } = \left\{ { { a } _ { t + h } ^ { \star } } , \quad h < L _ { t } , \right.
$$

where $a _ { \mathrm { s t o p } } \in { \mathcal { A } }$ denotes the Stop action. The validity mask ensures that Stop tokens appended beyond the length-T supervision sequence solely to complete the fixed K-slot template contribute no training loss. Any Stop token already contained in $\mathbf { a } ^ { \star } = ( a _ { 0 } ^ { \star } , \ldots , a _ { T - 1 } ^ { \star } )$ remains a valid supervised target.

Importantly, extending the prediction horizon does not expose future visual observations to the policy. At decision step t, the model is conditioned on the current visual representation $\widetilde { \mathbf { Z } } _ { t }$ and instruction $I ;$ future observations $O _ { t + 1 } , \dots , O _ { t + K - 1 }$ eare not provided. Supervision for future action slots is provided only by the corresponding action targets.

During training, all K action positions supplied to the bidirectional backbone are replaced with [MASK]. Only the valid action prefix is used for supervision, while synthetic tail padding is excluded by the valid-prefix mask. DreamFly then jointly predicts all action slots in a single bidirectional forward pass. Thus, action-policy training does not unroll iterative denoising; during inference, iterative discrete difusion progressively resolves the masked action slots.

At decision step t, let

$$
\mathbf { z } _ { t , h } \in \mathbb { R } ^ { | \mathcal { V } | } , \qquad h = 0 , \dotsc , K - 1 ,
$$

denote the shifted full-vocabulary logit vector at the sequence position that predicts the target token $\chi \left( \hat { a } _ { t , h } ^ { \star } \right)$ for the h-th action slot.

Let $\mathcal { U } _ { t }$ denote the valid non-action and non-padding positions in the shifted sequence, and let $q _ { t , h }$ denote the shifted sequence position whose logits form $\mathbf { z } _ { t , h }$ . We define the deterministic geometric kernel

$$
\kappa _ { i j } = \left\{ \begin{array} { l l } { 0 , } & { i = j , } \\ { \displaystyle \frac { \beta _ { \mathrm { c a r } } } { 2 } ( 1 - \beta _ { \mathrm { c a r } } ) ^ { \left| i - j \right| - 1 } , } & { i \neq j , } \end{array} \right.
$$

and compute the context coeficient as

$$
c _ { t , h } = \operatorname* { m a x } \left( 1 0 ^ { - 6 } , \sum _ { i \in \mathcal { U } _ { t } } \kappa _ { i , q _ { t , h } } \right) , \qquad \beta _ { \mathrm { c a r } } = 0 . 1 .
$$

All action slots, including unsupervised tail slots, are excluded from $\mathcal { U } _ { t }$ . Let B denote a minibatch of decision-step samples, with trajectory identities omitted for clarity. The action objective is

$$
\begin{array} { r l r } {  { \mathcal { L } _ { \mathrm { a c t } } = \frac { \displaystyle \sum _ { t \in \mathcal { B } } \sum _ { h = 0 } ^ { K - 1 } v _ { t , h } c _ { t , h } \gamma ^ { h } \mathrm { C E } _ { \mathcal { V } } ( \mathbf { z } _ { t , h } , \chi ( \bar { a } _ { t , h } ^ { \star } ) ) } { \displaystyle \sum _ { t \in \mathcal { B } } \sum _ { h = 0 } ^ { K - 1 } v _ { t , h } c _ { t , h } \gamma ^ { h } } , } } \end{array}
$$

where $0 < \gamma \leq 1$ controls horizon-dependent weighting; values below one progressively emphasize near-term action targets. The validity mask $v _ { t , h }$ restricts the objective to positions contained in the length-T supervision sequence. Following Dream-VLA, the cross-entropy is evaluated over the complete vocabulary V. The restriction to the dedicated action-token set $\chi ( \mathcal { A } )$ is applied only during difusion generation to guarantee executable action outputs.

Discrete difusion generation. At inference time, DreamFly generates the action chunk through iterative discrete difusion. Let $\mathbf { m } _ { t } ^ { ( s ) }$ denote the abstract state of the K action slots at denoising step $s ;$ fixed formatting tokens in the assistant scafold are omitted from this notation. Generation begins from

$$
\mathbf { m } _ { t } ^ { ( 0 ) } = \left[ \left[ \mathsf { M A S K } \right] , \ldots , \left[ \mathsf { M A S K } \right] \right] ,
$$

and proceeds through

$$
{ \bf m } _ { t } ^ { ( 0 ) }  { \bf m } _ { t } ^ { ( 1 ) }  \cdots  { \bf m } _ { t } ^ { ( S ) } , \qquad \Pi _ { \cal A } ( { \bf m } _ { t } ^ { ( S ) } ) = \hat { \bf a } _ { t } ,
$$

where $S$ denotes the number of difusion steps and $\Pi _ { \mathcal { A } } ( \cdot )$ decodes the K action slots through $\chi ^ { - 1 }$ This notation distinguishes the predicted action chunk from the complete assistant scafold, which also contains fixed formatting tokens.

We use the monotonic origin sampler inherited from Dream-VLA. Let $\{ \xi _ { s } \} _ { s = 0 } ^ { S }$ be linearly spaced from 1 to $\varepsilon _ { \mathrm { d i f f } } = 1 0 ^ { - 3 }$ . At denoising step $s \in \{ 0 , \ldots , S - 1 \}$ , each unresolved action slot is independently transferred from [MASK] to a sampled action token with probability

$$
\omega _ { s } = \left\{ \begin{array} { l l } { 1 - \frac { \xi _ { s + 1 } } { \xi _ { s } } , } & { s < S - 1 , } \\ { 1 , } & { s = S - 1 . } \end{array} \right.
$$

Conditional on transfer, the replacement token is sampled from the model distribution after suppressing all logits outside $\chi ( \mathcal { A } )$ . Resolved action slots are not remasked, and the final denoising step resolves all remaining masks.

Unlike fixed left-to-right autoregressive decoding, the bidirectional backbone allows unresolved action slots to share the available context, and multiple slots may be resolved within the same denoising step. Resolved action tokens can subsequently provide context for slots that remain unresolved. Before sampling at each unresolved position, logits outside $\chi ( \mathcal { A } )$ are suppressed; consequently, every transferred token belongs to the dedicated action-token set.

The initial all-mask forward also provides the planning representation consumed by LiteStop. Let $\mathbf { z } _ { t , h } ^ { ( 0 ) } \in \mathbb { R } ^ { | \mathcal { V } | }$ denote the shifted full-vocabulary logit vector aligned with the h-th action position at the initial denoising step. We retain the corresponding action-token slices as

$$
\mathbf { H } _ { t } ^ { ( 0 ) } = \left[ \mathrm { S l i c e } _ { \boldsymbol { \chi } ( \boldsymbol { A } ) } \left( \mathbf { z } _ { t , 0 } ^ { ( 0 ) } \right) ; \ldots ; \mathrm { S l i c e } _ { \boldsymbol { \chi } ( \boldsymbol { A } ) } \left( \mathbf { z } _ { t , K - 1 } ^ { ( 0 ) } \right) \right] \in \mathbb { R } ^ { K \times | \boldsymbol { A } | } .
$$

Here, ${ \mathrm { S l i c e } } _ { \chi ( { \mathcal { A } } ) }$ extracts the action-token coordinates in the fixed action-ID order. The representation is extracted during the initial denoising forward, while all K action positions remain masked and before any action token is transferred. It therefore records the policy responses at all K action positions under the same all-mask planning state. Importantly, retaining $\bar { \mathbf { H } } _ { t } ^ { ( 0 ) }$ does not terminate or shorten difusion generation; the complete action chunk is generated before LiteStop is evaluated. Receding-horizon execution. After the complete action chunk has been generated, LiteStop determines whether navigation should terminate before any chunk action is executed. If LiteStop is triggered, the episode terminates without executing an action from $\hat { \mathbf { a } } _ { t } .$ . Otherwise, the leading action $\hat { a } _ { t } ^ { 0 }$ is handled according to the original action-level semantics. If $\hat { a } _ { t } ^ { 0 } = a _ { \mathrm { s t o p } }$ , the episode terminates through the action-level Stop condition; otherwise, the selected motion is executed. The remaining predictions $\hat { a } _ { t } ^ { 1 } , \dots , \hat { a } _ { t } ^ { K - 1 }$ are discarded rather than cached for subsequent execution.

After a non-terminal motion is executed, the agent receives the next observation $O _ { t + 1 }$ and replans a complete action chunk. The historical memory available at the next decision step follows the same causal definition introduced in Sec. 3.3:

$$
M _ { < t + 1 } = \mathcal { F } _ { \mathrm { m e m } } \left( I , \left\{ O _ { \tau } \right\} _ { \tau \leq t } \right) .
$$

Thus, information derived from $O _ { t }$ may influence the historical context at step $t + 1$ only if it is retained by the memory-construction mechanism, whereas the newly acquired observation $O _ { t + 1 }$ remains a separate current visual input and is not simultaneously exposed through the historical memory branch. DreamFly therefore follows a receding-horizon strategy that plans K actions at each decision step and executes at most the leading action before replanning from newly observed visual evidence.

## 3.5 LiteStop: Decoupled Termination Control

Termination has a qualitatively diferent consequence from ordinary motion decisions: an erroneous motion may be corrected through subsequent replanning, whereas premature termination immedi ately ends the episode. We therefore introduce LiteStop, a lightweight auxiliary head that derives an additional decision-level termination signal from the frozen policy’s initial all-mask planning response. LiteStop is optimized separately from the action policy: it consumes the frozen policy’s planning logits, but does not update policy parameters or modify the original action-level Stop semantics.

All-mask termination representation. We reuse the initial planning representation ${ \bf H } _ { t } ^ { ( 0 ) } \in  { \bf \Psi }$ $\mathbb { R } ^ { K \times | \mathcal { A } | }$ defined in Sec. 3.4. It contains the shift-aligned raw logits over the dedicated action-token set $\chi ( \mathcal { A } )$ for all K planning positions. LiteStop maps this complete response grid to a scalar Stop logit:

$$
\ell _ { t } ^ { \mathrm { s t o p } } = g _ { \mathrm { s t o p } } \Big ( \mathbf { H } _ { t } ^ { ( 0 ) } \Big ) = W _ { 2 } \mathrm { S i L U } \Big ( W _ { 1 } \mathrm { L N } \Big ( \mathrm { v e c } \Big ( \mathbf { H } _ { t } ^ { ( 0 ) } \Big ) \Big ) + \mathbf { b } _ { 1 } \Big ) + b _ { 2 } , \qquad p _ { t } ^ { \mathrm { s t o p } } = \sigma \Big ( \ell _ { t } ^ { \mathrm { s t o p } } \Big ) .
$$

Here, vec(·) vectorizes the logit grid, LN(·) denotes layer normalization, and $g _ { \mathrm { s t o p } }$ contains the learnable LayerNorm and MLP parameters. Using the complete $K \times | { \mathcal { A } } |$ grid allows LiteStop to exploit the policy’s short-horizon planning state rather than relying only on the first-position Stop logit.

Frozen-policy supervision. Let $a _ { t } ^ { \star }$ denote the expert action at decision step t. The binary Stop target is

$$
y _ { t } ^ { \mathrm { s t o p } } = \mathbb { I } [ a _ { t } ^ { \star } = a _ { \mathrm { s t o p } } ] .
$$

A Stop appearing only at a future chunk position or in synthetic tail padding therefore does not afect the current label. The supervision uses neither geometric success nor terminal metadata, and thus calibrates the frozen policy’s action-level Stop tendency rather than learning an independent goal-reached classifier.

For a training batch B, we minimize

$$
{ \mathcal { L } } _ { \mathrm { s t o p } } = - { \frac { 1 } { | \mathcal { B } | } } \sum _ { t \in \mathcal { B } } \left[ \lambda _ { + } y _ { t } ^ { \mathrm { s t o p } } \log p _ { t } ^ { \mathrm { s t o p } } + \left( 1 - y _ { t } ^ { \mathrm { s t o p } } \right) \log \left( 1 - p _ { t } ^ { \mathrm { s t o p } } \right) \right] ,
$$

where $\lambda _ { + } = 4 . 0$ is the fixed positive-class weight. The complete navigation policy, including its visual, memory, and action-planning components, remains frozen, and only LiteStop is optimized. During training, $\mathbf { H } _ { t } ^ { ( 0 ) }$ is extracted by a single all-mask bidirectional forward pass without unfolding iterative difusion, matching the all-mask representation semantics used at deployment.

Pre-action termination. During inference, $\mathbf { H } _ { t } ^ { ( 0 ) }$ is cached from the initial all-mask denoising forward. The frozen policy nevertheless completes the full difusion process and obtains the complete decoded action chunk before LiteStop is evaluated. LiteStop therefore requires no additional backbone forward pass, but it is not a difusion early-exit mechanism. This ordering preserves the frozen policy’s complete generation path and confines LiteStop to pre-action termination control; computational early exit is outside the scope of the present design.

Given a fixed threshold $\eta _ { \mathrm { s t o p } }$ , the LiteStop decision is

$$
d _ { t } ^ { \mathrm { s t o p } } = \mathbb { I } \Big [ p _ { t } ^ { \mathrm { s t o p } } \geq \eta _ { \mathrm { s t o p } } \Big ] .
$$

The final termination decision combines LiteStop with the frozen policy’s action-level Stop condition:

$$
d _ { t } ^ { \mathrm { t e r m } } = d _ { t } ^ { \mathrm { s t o p } } \vee \mathbb { I } \big [ \hat { a } _ { t } ^ { 0 } = a _ { \mathrm { s t o p } } \big ] .
$$

If $d _ { t } ^ { \mathrm { s t o p } } = 1$ , the episode terminates before any action from the current chunk is executed. Otherwise, the first action $\hat { a } _ { t } ^ { 0 }$ is handled under the receding-horizon protocol from Sec. 3.4. If $\hat { a } _ { t } ^ { 0 } = a _ { \mathrm { s t o p } } ,$ the episode terminates through the retained action-level Stop condition; if neither termination condition is satisfied, the selected motion is executed and the agent acquires a new observation before replanning the complete action chunk. LiteStop therefore provides an additional pre-action termination pathway rather than vetoing the frozen policy’s action-level Stop condition.

Overall closed-loop operation. Taken together, the three components form a causal recedinghorizon control loop. At decision step t, DreamFly combines the current observation $O _ { t }$ with historical memory $M _ { < t }$ to obtain the memory-enhanced visual representation $\widetilde { \mathbf { Z } } _ { t }$ . Together with ethe navigation instruction I, this representation conditions the action policy, which generates a complete K-step action chunk $\hat { \mathbf { a } } _ { t }$ while retaining the initial all-mask planning representation $\mathbf { H } _ { t } ^ { ( 0 ) }$ for LiteStop. After the complete action chunk has been generated and before any action from the chunk is executed, LiteStop determines whether the episode should terminate. If termination is not triggered, the leading action $\hat { a } _ { t } ^ { 0 }$ is handled according to the original action-level semantics.

The historical memory available to the current decision contains only information derived from observations preceding $O _ { t }$ . Information extracted from $O _ { t }$ may afect the historical memory only from subsequent decisions onward and can therefore first become accessible through $M _ { < t + 1 }$ if it is retained by the memory-construction mechanism. After a non-terminal motion produces a new observation $O _ { t + 1 }$ , DreamFly repeats the same process using $O _ { t + 1 }$ as the current visual input together with the corresponding causal memory prefix.

The components are optimized in stages rather than through a single joint objective. Historical memory construction remains fixed while the memory-conditioned action policy is trained with the horizon-aware action objective $\mathcal { L } _ { \mathrm { a c t } }$ . After the navigation policy has been trained, the complete policy is frozen and only LiteStop is optimized with $\mathcal { L } _ { \mathrm { s t o p } }$ . Consequently, LiteStop training does not back-propagate into the visual, memory-adaptation, or action-planning components. This staged design integrates causal historical context, short-horizon difusion planning, and decoupled termination calibration within a unified closed-loop navigation policy.

## 4 Experiments

## 4.1 Experimental Setup

## 4.1.1 Datasets

We conduct our experiments based on the OpenFly dataset. Before training, we apply four standardization and augmentation steps to the released data. First, we correct the 8-D action vector associated with Forward 6m to its canonical encoding. Second, approximately 190,000 decision steps with non-standard action labels are remapped to the corresponding canonical actions, with −1 mapped to Go Up and −2 to Go Down. Third, we remove the pre-packaged historical keyframes and retain only the current RGB observation at each decision step. Fourth, We additionally store $[ \Delta x , \Delta y , \Delta z , \mathrm { y a w } ]$ as provenance metadata. This metadata is not used as an input to either historical-memory construction or policy inference. These relative poses are not used as additional policy inputs during closed-loop inference. The resulting training set contains 20 subsets with 85,785 trajectories and 1,356,622 decision steps.

Our evaluation covers eight AirSim/UE environments with 1,796 trajectories. The test-seen split contains UE BigCity and six AirSim urban environments, totaling 1,392 trajectories, whereas the test-unseen split contains 404 trajectories from UE SmallCity. This evaluation therefore covers both Unreal Engine and AirSim environments and assesses navigation performance in seen environments as well as generalization to an unseen urban scene. As shown in Fig. 3, the training data exhibit a pronounced imbalance toward forward actions, while the test-seen and test-unseen splits show diferent distributions of initial goal distances.

![](images/8288994ed0043a935e385477c2eb6c93f78f587de573006b4f6cd6e6c4154599.jpg)  
(a)

![](images/e272699cfa0ac9bea10d1afc951e0f7512c2d5618512338fb70f80909b8223e7.jpg)  
(b)  
Figure 3: Dataset statistics: (a) action distribution in the training set; (b) distributions of initial goal distances on the test-seen and test-unseen splits.

## 4.1.2 Evaluation Metrics.

Following the evaluation protocol of OpenFly and prior aerial VLN studies, we adopt four standard metrics: navigation error (NE), success rate (SR), oracle success rate (OSR), and success weighted by path length (SPL). NE measures the average Euclidean distance between the UAV’s final stopping position and the ground-truth target position, with a lower value indicating more accurate goal localization. SR measures the proportion of successful trajectories, where a trajectory is considered successful if the UAV stops within 20 m of the target. OSR measures the proportion of trajectories that come within 20 m of the target at any point during navigation, regardless of the final stopping position, and therefore reflects whether the agent successfully reaches the target vicinity. SPL jointly evaluates navigation success and path eficiency by weighting each successful trajectory according to the ratio between the shortest-path distance and the actual traveled path length. Higher values of SR, OSR, and SPL indicate better performance.

## 4.1.3 Implementation Details

The Dream-VLA backbone is fine-tuned with all-linear LoRA $( r = 3 2 , \alpha = 1 6 )$ , while the memoryfusion adapter is trained jointly and the base projector remains frozen. We use an action-chunk length of $K = 4 .$ , horizon decay $\gamma = 0 . 7$ , and CAR reweighting probability $p = 0 . 1$ . Training uses AdamW with a learning rate of $1 \times 1 0 ^ { - 4 }$ and batch size 8 for up to 10,000 optimization steps. The navigation checkpoint at step 5,000 is used to train LiteStop and for closed-loop evaluation. Inference uses 12 discrete difusion steps.

The historical memory contains 16 long-term slots with 512-dimensional features. Memory is fused with the current visual tokens using a gated cross-attention module with dimension 512 and eight attention heads.

LiteStop operating-point selection. We evaluate $\eta _ { \mathrm { s t o p } } ~ \in ~ 0 . 5 0 , 0 . 6 5 , 0 . 8 0$ using the step-500 LiteStop checkpoint on a balanced 64-trajectory calibration set (8 per environment), disjoint from the final evaluation split. Based on the overall trade-of in Table 1, We select $\eta _ { \mathrm { s t o p } } = 0 . 5 0$ because it yields the smallest OSR–SR gap and a slightly lower NE than $\eta _ { \mathrm { s t o p } } = 0 . 8 0$ while preserving the same SR. Unlike $\eta _ { \mathrm { s t o p } } = 0 . 8 0$ , it also produces nontrivial LiteStop interventions.

Table 1: Pilot selection of the LiteStop termination threshold. $N _ { \mathrm { L S } }$ is the number of episodes terminated by LiteStop itself, out of the 64 pilot episodes. The selected operating point is marked by †.
<table><tr><td colspan="3">SR↑ SPL ↑ OSR-SR↓ NE↓  $\eta _ { \mathrm { s t o p } }$   $N _ { \mathrm { L S } }$ </td></tr><tr><td> $0 . 5 0 ^ { \dagger }$  0.266 0.229</td><td>0.188</td><td>53.07 45/64</td></tr><tr><td>0.65 0.250</td><td>0.202 0.250</td><td>51.42 19/64</td></tr><tr><td>0.80 0.266 0.230</td><td>0.219</td><td>53.20 0/64</td></tr></table>

Table 2: Performance comparison on the seen and unseen splits.
<table><tr><td rowspan="2">Method</td><td colspan="4">test-seen</td><td colspan="4">test-unseen</td></tr><tr><td>NE↓</td><td>SR↑</td><td>OSR↑</td><td>SPL↑</td><td> $\mathrm { N E \downarrow }$ </td><td>SR↑</td><td>OSR↑</td><td>SPL↑</td></tr><tr><td>Random</td><td>65.67m</td><td>13.51%</td><td>18.75%</td><td>9.72%</td><td>59.99m</td><td>15.35%</td><td>23.76%</td><td>11.31%</td></tr><tr><td>Action Sampling</td><td>62.78m</td><td>15.95%</td><td>26.51%</td><td>13.67%</td><td>55.27m</td><td>20.54%</td><td>32.67%</td><td>17.22%</td></tr><tr><td>Seq2Seq</td><td>54.44m</td><td>24.35%</td><td>61.93%</td><td>19.35%</td><td>47.69m</td><td>26.49%</td><td>61.88%</td><td>19.62%</td></tr><tr><td>CMA</td><td>313.03m</td><td>7.97%</td><td>69.32%</td><td>6.26%</td><td>230.05m</td><td>5.69%</td><td>73.02%</td><td>3.92%</td></tr><tr><td>AerialVLN</td><td>176.29m</td><td>16.52%</td><td>65.66%</td><td>14.63%</td><td>161.19m</td><td>9.65%</td><td>68.07%</td><td>7.93%</td></tr><tr><td>OpenFly-Agent</td><td>122.89m</td><td>22.63%</td><td>52.73%</td><td>20.42%</td><td>163.87m</td><td>14.11%</td><td>62.38%</td><td>12.49%</td></tr><tr><td>DreamFly(Ours)</td><td>44.87m</td><td>32.04%</td><td>46.77%</td><td>28.22%</td><td>45.29m</td><td>29.46%</td><td>46.78%</td><td>23.54%</td></tr></table>

## 4.2 Experiment Result

As shown in Table 2, we compare DreamFly with six baselines: Random, Action Sampling, Seq2Seq, CMA, AerialVLN, and OpenFly-Agent. Random uniformly samples an action from the discrete action space at each step and executes it until either $\mathrm { S t o p } _ { \mathrm { t o k } }$ is sampled or the maximum number of navigation steps is reached. Action Sampling follows the same procedure but samples actions according to their empirical distribution in the training set, providing a stronger stochastic baseline that reflects the action prior of the dataset. For OpenFly-Agent, we directly evaluate the oficially released checkpoint in the same test environment to avoid introducing discrepancies by reproducing its original training and historical-observation pipeline under our standardized data protocol. All other learning-based baselines are trained on our processed training data for the same number of optimization steps.

As shown in Figure 4, the non-zero success rates of the stochastic baselines should not be interpreted as evidence of efective goal-directed navigation. To examine the influence of the initial state, we further partition the test trajectories according to whether their initial goal distance is within the 20 m success radius and report the conditional success rates for the two groups. Both Random and Action Sampling exhibit substantially higher success rates when the agent is initialized within the success radius than when it starts outside this region. This pronounced gap indicates that their non-zero overall success rates are strongly influenced by favorable initial configurations, rather than reflecting navigation policies that consistently drive the agent toward the target.

## 4.3 Ablation Studies

To evaluate the contribution of each component, we conduct both progressive and leave-one-out ablation studies, as reported in Table 3. Starting from the Dream-VLA baseline, introducing the causally aligned historical memory consistently improves navigation performance, while receding horizon difusion planning further improves the results by jointly modeling the current action and short-horizon future actions. Incorporating LiteStop yields the strongest overall performance. Conversely, removing Memory, Action Chunk, or LiteStop from the complete framework results in performance degradation. These consistent trends indicate that historical context modeling, future action planning, and explicit termination provide complementary benefits to DreamFly.

![](images/b635aaea0cf25b1f4539984a5dc9d1816b8d9e24acc7ad7465d8af832d664f34.jpg)  
Figure 4: Conditional success rates of baseline methods under diferent initial goal distances. Trajectories are partitioned according to whether the initial goal distance is within the 20 m success radius. The stochastic baselines exhibit substantially higher success rates when initialized inside the success region, highlighting the strong influence of favorable initial configurations on their non-zero overall success rates.

Table 3: Ablation study of diferent experimental settings.
<table><tr><td>Experiment</td><td>NE↓</td><td>SR↑</td><td>OSR↑</td><td>SPL↑</td></tr><tr><td>Dream-VLA (baseline)</td><td>67.82m</td><td>21.55%</td><td>42.32%</td><td>16.09%</td></tr><tr><td>+ Causal Memory</td><td>48.93m</td><td>24.11%</td><td>48.22%</td><td>19.85%</td></tr><tr><td> $\mathrm { w / o }$  Memory</td><td>62.02m</td><td>19.60%</td><td>23.89%</td><td>18.76%</td></tr><tr><td> $\mathrm { w / o }$  Chunk</td><td>46.72m</td><td>27.73%</td><td>38.42%</td><td>23.77%</td></tr><tr><td>w/o LiteStop</td><td>50.69m</td><td>26.61%</td><td>55.18%</td><td>22.29%</td></tr><tr><td>DreamFly(Ours)</td><td>44.97m</td><td>31.46%</td><td>46.77%</td><td>27.17%</td></tr></table>

To further examine how these components contribute under diferent navigation distances, we partition the test trajectories into three groups according to their initial shortest-path distance. As shown in Fig. 5, LiteStop provides its largest gain in the shortest-distance group, where successful navigation depends more directly on recognizing the appropriate termination point. Historical memory contributes more prominently at intermediate distances, while for larger initial distances, memory and action-chunk planning provide complementary benefits by maintaining cross-step visual context and introducing short-horizon future-action structure into each replanning step.

The contribution of LiteStop decreases as the initial distance increases, which is consistent with termination becoming relevant only after the agent reaches the vicinity of the goal. In this regard, the gap between OSR and SR provides a complementary view of termination behavior, since OSR reflects whether the trajectory reaches the success region whereas SR additionally requires successful termination within it. Overall, the distance-wise results further reveal the distinct roles of historical memory, receding-horizon difusion planning, and LiteStop in the DreamFly framework.

Never reached goalReached but missed stopCorrect stop

![](images/d9d37116f013bc5e66f64a336256aa9f4363fa3c2962a9c377d64bfd443233a0.jpg)

![](images/bb58926329eef4f25af1d0f5c3fbf2a91732fa44e6b1b50f45f460a3960f72b2.jpg)

![](images/c83bedea0038328d6df730be45ecfd2db65f17ede070c8e27e2bbf49edf5138b.jpg)

![](images/b37a07a126773b35dbdbbede4a0b8b57356d041813f12cd6d7cbf8c8d397e301.jpg)  
Figure 5: Component-wise analysis across diferent initial 3D Euclidean distance(s) to the goal. Test trajectories are partitioned into three groups according to their initial shortest-path distances to examine how the contribution of each DreamFly component varies with navigation distance.

## 4.4 Qualitative Analysis

Fig. 6 qualitatively illustrates how the three proposed components afect closed-loop navigation. In the first example, DreamFly steadily approaches the target and reduces the navigation error from 78.9 m to 10.9 m, whereas removing historical memory results in much slower progress and a final error of 43.5 m. This comparison shows that retaining previously observed visual evidence helps maintain a consistent reference to the target under changing viewpoints. In the second example, DreamFly follows a coherent sequence of actions and reaches the target with an error of 2.2 m, while removing action-chunk prediction leads to an early deviation and eventually fails with an error of 58.1 m, highlighting the benefit of incorporating short-horizon future-action structure into the current decision. In the third example, both variants approach the target region, but the model without LiteStop continues moving after entering the success region and drifts away, ending at 23.0 m. DreamFly instead terminates successfully at 12.5 m, demonstrating the importance of explicitly modeling task completion.

Overall, the examples reveal complementary failure modes addressed by the proposed components: historical memory preserves cross-step visual context, receding-horizon planning improves action consistency, and LiteStop prevents unnecessary motion after reaching the goal region.

## 5 Conclusion

In this work, we present DreamFly, a difusion-based framework for aerial vision-language navigation. Building upon the Dream-VLA backbone, we revisit aerial navigation from the perspective of temporal decision making and identify three essential capabilities: retaining informative historical observations, anticipating future actions while preserving closed-loop feedback, and reliably determining when navigation should terminate. To this end, DreamFly integrates causally aligned historical memory, receding-horizon difusion planning, and LiteStop for explicit termination into a unified closed-loop framework. The historical memory provides the policy with accumulated visual evidence under a strict causal prefix constraint, while difusion-based action-chunk predic tion enables joint modeling of the current action and short-horizon future actions. By following a plan-K, execute-one strategy, DreamFly uses future-action predictions as planning variables while replanning after every executed action, thereby preserving closed-loop feedback. LiteStop further decouples termination from action generation by estimating the stop probability from the initial all-mask action logits.

Extensive experiments on OpenFly validate the efectiveness of the proposed framework. DreamFly achieves the best NE, SR, and SPL on both the test-seen and test-unseen splits, reaching 32.04%/29.46% SR and 28.22%/23.54% SPL, respectively. The consistent performance improvements across both splits further demonstrate that DreamFly remains efective when navigating previously unseen environments.

Despite the consistent improvements observed in simulation, the current evaluation of DreamFly is still limited to simulated environments. Future work will focus on deploying the framework on physical UAV platforms to assess its real-world navigation capability and robustness under sensing noise, environmental disturbances, and sim-to-real domain shifts.

## References

[1] P. Anderson et al., “Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Jun. 2018, pp. 3674–3683. doi: 10.1109/CVPR.2018.00387.

![](images/af16393f0f26700f95c1f2f43ce97b66ce0b50d737c72e29d2f4df4aa642342c.jpg)

## Instruction:

Proceed directly ahead towards a large building with gray and white colors, featuring vertical windows. Then, slightly turn right and move forward to it. Finally, slightly turn left and head straight towards it.

![](images/23da4d955b612a143d215885d2467554d3ab4334c5ab88f2f766f988032e37cc.jpg)

## Instruction:

The seguence of actions and observations can be described as follows: Move forward towards a medium-sized, aray building showcasing a multi-story structure with prominent windows and classic architecture. Shift left to face a medium-sized, white building featuring rectangular windows and a flat roof. Advance forward to it. Slightly turn right and then head straight towards a medium-to-large, dark gray or black urban multi-story building with faintly illuminated windows, glowing light at the center window, a textured brick façade, and occupying the central focus of the frame

## Instruction:

Proceed straight until you reach a large beige building characterized by rectangular windows and detailed cornices, then slightly turn right and move ahead towards another large beige building notable for its decorative detailing along the roofline and classic architectural design; this structure is a large, historic, or classical-style building with multiple stories.  
![](images/b243f62d684b9228df96516cc750155619a89e4eb5340b8a503a7be65ad3ccd2.jpg)  
Figure 6: Qualitative comparison between DreamFly and its ablated variants on representative OpenFly trajectories. The three examples illustrate the efects of historical memory, recedinghorizon action planning, and LiteStop, respectively. Numbers indicate the distance to the navigation goal.

[2] S. Liu, H. Zhang, Y. Qi, P. Wang, Y. Zhang, and Q. Wu, “AerialVLN: Vision-and-Language Navigation for UAVs,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV), Oct. 2023, pp. 15 338–15 348. doi: 10.1109/ICCV51070.2023.01411.

[3] G. Zhao, G. Li, J. Pan, and Y. Yu. “Aerial Vision-and-Language Navigation with Grid-based View Selection and Map Construction.” arXiv: 2503.11091 [cs.CV], pre-published.

[4] Y. Gao et al., “OpenFly: A Comprehensive Platform for Aerial Vision-Language Navigation,” in The Fourteenth International Conference on Learning Representations, 2026.

[5] X. Wang et al., “Reinforced Cross-Modal Matching and Self-Supervised Imitation Learning for Vision-Language Navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 6629–6638.

[6] Y. Hong, Q. Wu, Y. Qi, C. Rodriguez-Opazo, and S. Gould, “VLN⟳BERT: A Recurrent Vision-and-Language BERT for Navigation,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2021, pp. 1643–1653. doi: 10.1109/CVPR46437.2021.00169.

[7] S. Chen, P.-L. Guhur, C. Schmid, and I. Laptev, “History Aware Multimodal Transformer for Vision-and-Language Navigation,” in Advances in Neural Information Processing Systems, M. Ranzato, A. Beygelzimer, Y. Dauphin, P. S. Liang, and J. W. Vaughan, Eds., vol. 34, Curran Associates, Inc., 2021, pp. 5834–5847.

[8] W. Jiang et al., “LongFly: Long-Horizon UAV Vision-and-Language Navigation with Spatiotemporal Context Integration,” arXiv:2512.22010, 2025. [Online]. Available: https://arxiv. org/abs/2512.22010

[9] B. Zitkovich et al., “RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control,” in Proceedings of The 7th Conference on Robot Learning, PMLR, Dec. 2, 2023, pp. 2165–2183.

[10] M. J. Kim et al., “OpenVLA: An Open-Source Vision-Language-Action Model,” in Proceedings of the 8th Conference on Robot Learning, vol. 270, PMLR, 2025, pp. 2679–2713.

[11] C. Chi et al., “Difusion policy: Visuomotor policy learning via action difusion,” The International Journal of Robotics Research, vol. 44, no. 10–11, pp. 1684–1704, 2025. doi: 10.1177/02783649241273668.

[12] W. Zhang et al., “DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge,” in Advances in Neural Information Processing Systems 38, 2025.

[13] P. Xu, Z. Deng, J. Deng, Z. Gu, and S. Wan, “AerialVLA: A Vision-Language-Action Model for UAV Navigation via Minimalist End-to-End Control,” arXiv:2603.14363, 2026. [Online]. Available: https://arxiv.org/abs/2603.14363

[14] B. Zhao et al., “WorldVLN: Autoregressive World Action Model for Aerial Vision-Language Navigation,” arXiv:2605.15964, 2026. [Online]. Available: https://arxiv.org/abs/2605.15964

[15] X. Liu, J. Huang, S. Xia, B. Liu, J. Cui, and J. Yang, “ImagineUAV: Aerial Vision-Language Navigation via World-Action Modeling and Kinodynamic Planning,” arXiv:2606.01205, 2026. [Online]. Available: https://arxiv.org/abs/2606.01205

[16] X. Zhu et al., “FSD-VLN: Fast-Slow Dual-System Modeling for Aerial Long-Horizon Vision-Language Navigation,” arXiv:2607.08359, 2026. [Online]. Available: https://arxiv.org/abs/ 2607.08359

[17] S. Ross, G. Gordon, and D. Bagnell, “A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning,” in Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, vol. 15, PMLR, 2011, pp. 627–635.

[18] J. Xiang, X. Wang, and W. Y. Wang, “Learning to Stop: A Simple yet Efective Approach to Urban Vision-Language Navigation,” in Findings of the Association for Computational Linguistics: EMNLP 2020, Association for Computational Linguistics, 2020, pp. 699–707. doi: 10.18653/v1/2020.findings-emnlp.62.

[19] Y. Fan, W. Chen, T. Jiang, C. Zhou, Y. Zhang, and X. E. Wang. “Aerial Vision-and-Dialog Navigation.” arXiv: 2205.12219 [cs.CV], pre-published.

[20] J. Lee et al. “CityNav: A Large-Scale Dataset for Real-World Aerial Navigation.” arXiv: 2406. 14240 [cs.CV], pre-published.

[21] X. Wang et al. “Towards Realistic UAV Vision-Language Navigation: Platform, Benchmark, and Methodology.” arXiv: 2410.07087 [cs.CV], pre-published.

[22] H. Cai et al. “AirNav: A Large-Scale UAV Vision-and-Language Navigation Dataset with Natural and Diverse Instructions.” arXiv: 2601.03707 [cs.CL], pre-published.

[23] Y. Gao, Z. Wang, P. Han, L. Jing, D. Wang, and B. Zhao. “Exploring Spatial Representation to Enhance LLM Reasoning in Aerial Vision-Language Navigation.” arXiv: 2410.08500 [cs.RO], pre-published.

[24] X. Ding, J. Gao, C. Pan, W. Wang, and J. Qin. “History-Enhanced Two-Stage Transformer for Aerial Vision-and-Language Navigation.” arXiv: 2512.14222 [cs.CV], pre-published.

[25] Y. Ning et al., “LookasideVLN: Direction-Aware Aerial Vision-and-Language Navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 32 441–32 450.

[26] H. Cai et al., “FlightGPT: Towards Generalizable and Interpretable UAV Vision-and-Language Navigation with Vision-Language Models,” in Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng, Eds., Suzhou, China: Association for Computational Linguistics, Nov. 2025, pp. 6659–6676, isbn: 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.338.

[27] D. Shao et al., “FineCog-Nav: Integrating Fine-grained Cognitive Modules for Zero-shot Multimodal UAV Navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, 2026, pp. 1325–1334.

[28] H. Wang, W. Wang, W. Liang, C. Xiong, and J. Shen, “Structured Scene Memory for Vision-Language Navigation,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2021, pp. 8451–8460. doi: 10.1109/CVPR46437.2021.00835.

[29] Z. Wang, X. Li, J. Yang, Y. Liu, and S. Jiang, “GridMM: Grid Memory Map for Vision-and-Language Navigation,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV), Paris, France: IEEE, Oct. 1, 2023, pp. 15 579–15 590, isbn: 979-8-3503-0718-4. doi: 10.1109/ICCV51070.2023.01432.

[30] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware,” in Proceedings of Robotics: Science and Systems XIX, Daegu, Republic of Korea, 2023. doi: 10.15607/RSS.2023.XIX.016.

[31] Y. Liu, J. I. Hamid, A. Xie, Y. Lee, M. Du, and C. Finn. “Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling,” arXiv.org, Accessed: Aug. 12, 2026. [Online]. Available: https://arxiv.org/abs/2408.17355v4

[32] K. Black, M. Galliker, and S. Levine, “Real-Time Execution of Action Chunking Flow Policies,” in Advances in Neural Information Processing Systems, vol. 38, Curran Associates, Inc., 2025, pp. 33 383–33 407. doi: 10.52202/085713-1122.

[33] J. Ye et al. “Dream-VL & Dream-VLA: Open Vision-Language and Vision-Language-Action Models with Difusion Language Model Backbone.” arXiv: 2512.22615 [cs.CV], pre-published.