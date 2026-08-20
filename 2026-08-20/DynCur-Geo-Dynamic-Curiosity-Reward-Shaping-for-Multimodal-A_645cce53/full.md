# DynCur-Geo: Dynamic Curiosity Reward Shaping for Multimodal Active Geo-Localization

Yiming Sun<sup>1</sup>, Yang Zhang<sup>1</sup>, Pengfei Zhu<sup>1</sup>

<sup>1</sup>School of Automation, Southeast University, Nanjing 210096, China

## Abstract

Active geo-localization enables low-altitude UAVs to search for specified targets from limited local aerial observations, supporting time-sensitive applications such as search and rescue and emergency inspection. However, multimodal target cues, restricted views, and sparse feedback make it dificult to balance exploration with target convergence. Existing curiosity-driven methods assign a fixed intrinsic-reward weight throughout search, which can continue rewarding novelty after the agent nears the target and induce detours. We propose DynCur-Geo, a dynamic curiosity framework that adjusts prediction-error intrinsic reward according to remaining target distance. A distance-aware gate encourages early exploration and shifts the policy toward goal-directed behavior near the target, while potential-based reward shaping supplies dense progress guidance. Experiments across multimodal, cross-scene, disaster-afected, and long-range settings show consistent gains over active geo-localization baselines.

## Introduction

Low-altitude UAVs can search large areas when direct access is dificult or rapid response is required. In search and rescue or emergency inspection, however, the target may be absent from the initial view, so the UAV must gather evidence, choose movements, and localize the target before the search budget is exhausted. This distinguishes active geo-localization (AGL) from conventional visual geolocalization, where a query is matched against a geotagged reference set or mapped to coordinates [3, 5, 11, 12, 34, 35, 36]. AGL instead poses localization as sequential decisions over local aerial observations, making an eficient learned policy essential [16, 20, 22].

Targets may be specified by a previous aerial crop, a ground-view photograph, or a short text description. GOMAA-Geo [22] extends AGL to this goal-modalityagnostic setting, where the policy must align these cues with local aerial observations. The cue and local views may differ strongly in appearance and viewpoint, directional evidence can remain ambiguous for several steps, and every exploratory action consumes budget.

Curiosity-driven exploration helps under sparse feedback. GeoExplorer [16] uses action-state prediction error as an intrinsic reward and combines it with the external reward using a fixed curiosity weight throughout search. Although this encourages informative transitions, prediction error measures novelty rather than goal relevance, so it can preserve exploratory pressure after the policy enters the target neighborhood, where convergence is more valuable than unfamiliar states.

![](images/0ffe8b4d97ef430e63003aa55b4343f2aaddb58e0111a848963d0219481f2b3b.jpg)  
Figure 1: Motivation for distance-gated curiosity. GeoExplorer uses a fixed curiosity weight that can weaken latestage convergence by rewarding novelty throughout search (orange). Our dynamic gate weights intrinsic reward more strongly far from the target and less near it, promoting exploration followed by goal-directed convergence (blue–green).

To address this limitation, we propose DynCur-Geo, a dynamic curiosity framework for multimodal AGL. Fig. 1 contrasts the fixed curiosity weight of GeoExplorer [16] with our trajectory-dependent design. Broad exploration is useful early, whereas later decisions should prioritize stable progress. We implement this transition with distancegated curiosity reward shaping: intrinsic reward is weighted more far from the target and downweighted as the policy approaches it. We also add potential-based reward shaping (PBRS) [1, 18] to provide dense target-progress feedback and reinforce goal-directed convergence.

Our main contributions are summarized as follows:

• We introduce DynCur-Geo, a dynamic-curiosity framework for multimodal active geo-localization that uses multimodal target cues and observation features within one sequential search policy.

• We develop a distance-aware curiosity gate that adapts prediction-error intrinsic reward along the trajectory, supporting broad exploration far from the target and precise convergence near it.

• We integrate potential-based target-progress shaping with dynamic curiosity and demonstrate consistent gains over existing methods across multimodal, cross-scene, disaster-afected, and long-range search settings.

## Related Work

Visual and Multimodal Geo-Localization. Visual geolocalization emphasizes representation learning for retrieval or coordinate estimation. Place-recognition models learn descriptors for large reference databases [3, 5], while crossview methods match images from ground cameras, drones, and satellites [11, 14, 27, 29, 30, 31, 35, 36]. Recent work considers limited fields of view, orientation uncertainty, drone-view queries, object-level targets, broader geographic coverage, and multimodal queries [8, 12, 13, 21, 32, 33, 34]. Contrastive and vision-language models further connect visual observations with geographic or textual representations [8, 21], and multimodal UAV localization extends queries with text, prompts, depth, point clouds, or multiple image views [13, 32].

These works motivate multimodal localization representations, but their protocols difer from AGL. Retrieval methods compare a query against a reference database or map; AGL observes only the current aerial patch, chooses the next move before seeing future patches, and is judged by whether it reaches the target cell within budget. Thus, the direct baselines are active-search policies under the same movement protocol.

Active Geo-Localization and Aerial Navigation. AiR-Loc [20] abstracts search-and-rescue-style aerial search as a reinforcement learning problem in which a policy localizes an aerial-view target from partial glimpses. GOMAA-Geo [22] extends the formulation to multimodal goal cues and combines cross-modal alignment, supervised pretraining, and actor-critic policy learning for aerial, ground-view, and text goals. GeoExplorer [16] adds action-state dynamics modeling and next-state prediction error as intrinsic reward. These closest AGL baselines use the same controlled grid-based setting; our work studies how prediction-error curiosity should be weighted as the policy moves from broad exploration to target convergence.

Related aerial and embodied navigation tasks study language instruction following, continuous motion, or reaching visible goals [2, 15, 37]. In contrast, AGL uses grid-based target localization, with performance measured by success rate and final grid distance under a fixed movement budget.

Curiosity and Reward Shaping. Intrinsic motivation methods use prediction error, novelty, reachability, or uncertainty to encourage exploration when external rewards are sparse or delayed [4, 6, 7, 19, 23, 24, 26, 28]. In visual reinforcement learning, prediction-error curiosity learns a dynamics model and rewards hard-to-predict transitions. This fits AGL’s repetitive local aerial patches and late target visibility, where sparse success reward gives little early feedback.

However, curiosity is not inherently task-aligned: exploration should increase the chance of reaching the target, not merely visit surprising states. PBRS [1, 18] adds dense progress feedback through a potential function while preserving policy invariance under standard assumptions. Our method uses prediction-error curiosity for exploration and combines a distance gate with PBRS to tie it to target convergence.

## Method

Problem Setup. We formulate multimodal active geolocalization as a goal-conditioned sequential decision problem on a discretized aerial region. The region is divided into $K \times K$ cells, and each cell p corresponds to an aerial image patch $I _ { p } .$ For each search episode, the policy first receives a target cue $^ { g , }$ which may be an aerial image, a ground-view image, or a text description. The cue is associated with a target cell $p _ { g } ,$ but the target location is hidden during inference. Starting from cell $p _ { 0 }$ , the policy observes only the aerial patch at its current cell $p _ { t }$ , selects an action $a _ { t }$ from $\mathcal { A } = \left\{ \mathrm { u p } \right\}$ , right, down, left}, and moves to the next cell if the action is legal. A search episode is successful if the policy reaches $p _ { g }$ within the budget B.

At each episode start, the policy receives only the target cue and the observation at $p _ { 0 } ;$ after each action, it receives the resulting local observation. The target cell is used only during training to construct dynamics targets and rewards. During inference, the policy does not evaluate target distance, reward terms, or an oracle planner; it acts from the learned representation, history, and legal-action mask. The resulting training pipeline and reward construction are summarized in Fig. 2.

Multimodal Feature Representation. Let m denote the modality of the target cue g. Following the feature representation used in multimodal AGL [16, 22], Sat2Cap [9] encodes aerial images for both aerial target cues and search observations, while the CLIP ViT-B/32 [21] image and text encoders represent ground-view and text targets. We denote the corresponding frozen modality encoder by $E _ { m }$ , which maps the target cue to a shared 512-dimensional target embedding:

$$
q = E _ { m } ( g ) , \quad m \in \{ \mathrm { a e r i a l } , \mathrm { g r o u n d } , \mathrm { t e x t } \} .\tag{1}
$$

For search observations, the same aerial encoder $E _ { a }$ maps the current patch $I _ { p _ { t } }$ to an observation feature $v _ { t } = E _ { a } ( I _ { p _ { t } } )$ . We augment this feature with a lightweight sinusoidal gridposition encoding and denote the resulting location-aware feature as $\tilde { v } _ { t } .$ . Given a learned action embedding $e ( a _ { i } )$ , the target, observation, and action features up to step t form the causal input sequence

$$
Z _ { t } = [ q , \tilde { v } _ { 0 } , e ( a _ { 0 } ) , \tilde { v } _ { 1 } , \ldots , e ( a _ { t - 1 } ) , \tilde { v } _ { t } ] .\tag{2}
$$

We reuse the pretrained causal Transformer $F _ { \theta }$ from Geo-Explorer [16] as the sequence model. With input projections from the 512-dimensional feature space and learned action tokens, it maps the causal sequence to a state representation

$$
h _ { t } = F _ { \theta } ( Z _ { t } ) ,\tag{3}
$$

where θ denotes the model parameters. The policy therefore combines multimodal target and observation features with the action sequence instead of relying on the current patch alone.

![](images/19275b549f086586faf0db6540ae6e7293edf86865dea41022e50c3c45fb5f91.jpg)  
Figure 2: Training pipeline and reward construction. Target cues and local aerial observations form a trajectory history processed by a frozen GeoExplorer dynamics model. Prediction error provides intrinsic reward, whose weight is reduced near the target and combined with external progress and PBRS rewards for PPO training.

Action-State Dynamics Modeling. The adopted GeoExplorer [16] action-state model was pretrained on generated trajectories. At each step, it predicts two targets: a multilabel action vector $y _ { t } .$ , where each entry indicates whether the corresponding movement reduces the grid distance to the goal, and the next-state feature after the action-conditioned transition. The dynamics-modeling loss is

$$
\mathcal { L } _ { \mathrm { D M } } = \mathcal { L } _ { \mathrm { a c t } } + \alpha \mathcal { L } _ { \mathrm { s t a t e } } ,\tag{4}
$$

where $\mathcal { L } _ { \mathrm { a c t } }$ is the action prediction loss, $\mathcal { L } _ { \mathrm { s t a t e } }$ is the statefeature prediction loss, and α balances the two terms. We use binary cross-entropy for $\mathcal { L } _ { \mathrm { a c t } }$ and mean-squared error for $\mathcal { L } _ { \mathrm { s t a t e } }$ . This objective trains the sequence model to encode both goal-directed action preferences and action-conditioned changes in aerial observations.

During PPO [25] training, the pretrained sequence model is frozen and used as a stable history encoder and dynamics model. Let $\hat { v } _ { t + 1 }$ denote the predicted next-state feature and $\tilde { v } _ { t + 1 }$ denote the observed next-state feature after executing $a _ { t }$ . The prediction error is

$$
e _ { t } = \left\| \hat { v } _ { t + 1 } - \tilde { v } _ { t + 1 } \right\| _ { 2 } ^ { 2 } .\tag{5}
$$

This error is the transition novelty signal used for intrinsic reward.

Distance-Gated External and Intrinsic Rewards. The reinforcement-learning stage follows the distance-based external reward commonly used in AGL, which gives dense feedback according to whether an action moves the policy toward the target. We use this reward as the task-progress signal and write it as

$$
r _ { \mathrm { e x } , t } = \left\{ \begin{array} { l l } { r _ { g } } & { p _ { t + 1 } = p _ { g } } \\ { r _ { p } } & { \Delta ( p _ { t + 1 } , p _ { g } ) < b _ { t } , } \\ { r _ { n } } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{6}
$$

where $p _ { t + 1 }$ is the cell after action $a _ { t } , r _ { g }$ is the goal-reaching reward, $r _ { p }$ is the progress reward, and $r _ { n } < 0$ penalizes nonprogress moves. The function $\Delta ( p , p _ { g } )$ measures distancebased progress between cell p and target cell $p _ { g } ,$ , while $b _ { t }$ records the best accepted progress score along the current trajectory. This term follows the prior dense AGL reward design and uses the known target location only during training.

The intrinsic reward is derived from the frozen dynamics model:

$$
r _ { \mathrm { i n } , t } = \psi ( e _ { t } ) ,\tag{7}
$$

where ψ(·) normalizes prediction errors to a scale comparable with the external reward. The resulting signal gives dense feedback when target progress is hard to infer from local observations, but it is goal-agnostic and therefore does not distinguish target-helpful novelty from target-irrelevant novelty. We therefore gate curiosity by the remaining target distance. Let $d _ { 1 } ( p , p _ { g } )$ be the Manhattan distance between cell p and the target cell. We first define a normalized distance ratio

$$
\rho _ { t } = \mathrm { c l i p } \left( \frac { d _ { 1 } ( p _ { t } , p _ { g } ) } { \operatorname* { m a x } ( d _ { 1 } ( p _ { 0 } , p _ { g } ) , 1 ) } , 0 , 1 \right) ,\tag{8}
$$

where $p _ { t }$ is the current cell, p<sub>0</sub> is the start cell, and the denominator is lower bounded by one for numerical stability. The distance ratio then defines a family of curiosity gates,

$$
\lambda _ { t } = \lambda _ { \operatorname* { m i n } } + ( 1 - \lambda _ { \operatorname* { m i n } } ) f ( \rho _ { t } ) ,\tag{9}
$$

where $\lambda _ { \mathrm { m i n } }$ is the lower bound of the curiosity weight and $f ( \rho _ { t } )$ controls the distance schedule. Eq. (9) assigns a larger curiosity weight far from the target and reduces that weight near the target. The final model uses a linear gate, $f ( \rho _ { t } ) = \rho _ { t } .$ The ablation study also compares a constant gate $f ( { \overset { \cdot } { \rho } } _ { t } ) = 0 ,$ a sine gate $f ( \rho _ { t } ) \stackrel { \cdot } { = } \sin ( \pi \rho _ { t } / \stackrel { \cdot } { 2 } )$ , and a power gate $f ( \rho _ { t } ) = \rho _ { t } ^ { 2 }$ The special case $\lambda _ { t } = 1$ denotes ungated curiosity.

Potential-Based Reward Shaping. The second reward component is potential-based reward shaping (PBRS), which supplies smooth distance feedback without changing the inference-time inputs. For a $K \times K$ grid, the maximum Manhattan distance is $d _ { \operatorname* { m a x } } = 2 ( K - 1 )$ . We define the potential function

$$
\Phi ( p ) = - \frac { d _ { 1 } ( p , p _ { g } ) } { d _ { \operatorname* { m a x } } } ,\tag{10}
$$

and the shaping reward

$$
r _ { \Phi , t } = \beta \left( \gamma \Phi ( p _ { t + 1 } ) - \Phi ( p _ { t } ) \right) ,\tag{11}
$$

where $\beta$ controls the strength of shaping and γ is the discount factor. The final training reward is

$$
r _ { t } = r _ { \mathrm { e x } , t } + \lambda _ { t } r _ { \mathrm { i n } , t } + r _ { \Phi , t } .\tag{12}
$$

This wraps the reward design into three training-only signals: the prior distance-based external reward supplies task feedback, the gated intrinsic reward encourages exploration when target evidence is still sparse, and PBRS supplies dense target-progress shaping. At inference time, none of these reward terms or target-distance quantities is evaluated.

Policy Learning and Inference. With the frozen history encoder $F _ { \theta }$ , we train an actor-critic policy using PPO. The actor outputs a distribution $\pi ( a | h _ { t } )$ over actions $\ i \in { \mathcal { A } } ,$ and the critic estimates the value of $h _ { t } .$ . Let $M _ { t } ( a ) \in \{ 0 , 1 \}$ be the legal-action mask at step t, where invalid boundary actions have value zero. The masked policy is

$$
\bar { \pi } ( a | h _ { t } ) = \frac { M _ { t } ( a ) \pi ( a | h _ { t } ) } { \sum _ { a ^ { \prime } \in \mathcal { A } } M _ { t } ( a ^ { \prime } ) \pi ( a ^ { \prime } | h _ { t } ) } .\tag{13}
$$

During inference, model parameters are fixed and the reward terms are not evaluated. The policy updates its history after each observation and selects the highest-probability legal action,

$$
a _ { t } = \arg \operatorname* { m a x } _ { a \in A , M _ { t } ( a ) = 1 } \bar { \pi } ( a | h _ { t } ) ,\tag{14}
$$

until it reaches the target cell or exhausts the search budget. Thus, the reported gains come from the policy learned under the training reward rather than from test-time access to target distance.

## Experiments

Experimental Setup. The main evaluation follows the protocol established by prior AGL benchmarks [16, 20, 22]. The search region is discretized into a 5×5 grid with initial Manhattan distances $C = \{ 4 , 5 , 6 , 7 , 8 \}$ . We report success rate (SR) and final grid distance (SG), using greedy action selection with a legal-action mask. For N tasks, terminal cells $p _ { i } ^ { T }$

and goals $p _ { i } ^ { g }$ ,

$$
\mathrm { S R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } [ p _ { i } ^ { T } = p _ { i } ^ { g } ] ,\tag{15}
$$

$$
\mathrm { S G } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } d _ { 1 } ( p _ { i } ^ { T } , p _ { i } ^ { g } ) ,
$$

where 1[·] is the indicator function and $d _ { 1 }$ is Manhattan distance. Thus SR is the fraction of searches that reach the goal within budget and SG is the mean terminal distance (lower is better). Tables 1 and 2 follow the MM-GAG setting; the additional studies use a unified $B = 1 2 , C = 4 , \dotsc , 8$ protocol with five trials per distance. The four-scenario $B =$ 12 result is the unweighted macro-average over MASA [17], MM-GAG, xBD-pre, and xBD-disaster [10].

Our final training pool combines 137 MASA images and 47 MM-GAG images. The model is trained for 480k PPO environment steps, dynamically sampling a start cell, goal cell, and initial distance $C \in \{ 1 , \ldots , \bar { 8 } \}$ on each image rather than using fixed routes.

Evaluation uses the same source families as the AGL baselines: MASA for aerial-goal search, MM-GAG for aerial, ground-view, and text goals, GeoExplorer’s SwissView100 and SwissViewMonuments splits [16] for geographicdomain and unseen-target transfer, and xBD for disaster transfer. SwissView100 contains 100 Swiss aerial regions, while SwissViewMonuments contains 15 landmark or atypical-scene regions with aerial and ground-view cues. For xBD, we use an 800-pair test subset: xBD-pre uses pre-disaster targets and observations, whereas xBD-disaster keeps the pre-disaster target cue and searches post-disaster observations.

We compare with Random, PPO [25] and DiT reimplementations, AiRLoc [20] where applicable, GOMAA-Geo [22], and GeoExplorer [16]. Random samples legal actions uniformly; PPO uses the current observation without a history encoder; DiT follows ofline optimal start–goal trajectories with a goal prefix and four actions [16, 22]. GOMAA-Geo supports all modalities, and GeoExplorer is the closest curiosity-driven baseline. In compact tables, GOMAA denotes GOMAA-Geo. Retrieval approaches [3, 5, 11, 35, 36] are omitted because they retrieve from a reference collection rather than act under the sequential protocol.

MM-GAG Multimodal Results. Tables 1 and 2 report MM-GAG results for aerial-image, ground-view-image, and text cues; PPO, DiT, and AiRLoc apply only to aerial-target comparison. Our method achieves the highest average SR and lowest average SG in all three modalities, raising average SR over the strongest baseline from 0.5336 to 0.6170 for aerial targets, from 0.5523 to 0.6391 for ground-view targets, and from 0.5472 to 0.6247 for text targets.

The advantage is most pronounced at medium and long initial distances: several baselines remain stronger at $C = 4 ,$ whereas our method leads clearly from $C = 6 \mathsf { t o } \bar { C } = 8$ . This pattern shows that the distance gate and PBRS are particularly efective when search requires broad exploration followed by accurate convergence; the lower SG further indicates closer endpoints on unsuccessful long-horizon episodes.

<table><tr><td>Method</td><td>C = 4 C = 5 C y = 6 C = 7 C = 8 Avg.</td></tr><tr><td>Random PPO DiT A AiRLoc</td><td>0.14470.09360.05530.01700.02980.0681 0.17450.14470.15740.24680.27230.1991 0.16600.22550.27660.2681 0.19150.2255 0.56600.48940.48940.43830.31910.4604</td></tr><tr><td></td><td>GOMAA-Geo0.35740.38720.55320.62980.74040.5336 GeoExplorer 0.34890.37450.55740.68510.67660.5285 DynCur-Geo 0.2681 0.3702 0.6809 0.8426 0.9234 0.6170</td></tr><tr><td></td><td>GOMAA-Geo0.34890.37450.56170.70640.77020.5523 G GeoExplorer 0.4043 0.3745 0.53190.6681 0.71490.5387 DynCur-Geo 0.2766 0.4468 0.6511 0.9106 0.9106 0.6391</td></tr><tr><td></td><td>GOMAA-Geo0.34470.42130.57450.65960.73620.5472 T GeoExplorer 0.3957 0.3660 0.4553 0.5660 0.6085 0.4783 DynCur-Geo 0.2681 0.4298 0.6596 0.8511 0.9149 0.6247</td></tr></table>

Table 1: Success rate (SR) on MM-GAG [22] under $5 \times 5 ,$ $B = 1 0$ , and $C = 4 , \ldots , 8 .$ A/G/T denote aerial-image, ground-view, and text targets. PPO, DiT, and AiRLoc are evaluated only for aerial-image targets.

<table><tr><td>Method</td><td>C = 4 C = 5 = 6 C = 7 C = 8 Avg.</td></tr><tr><td>Random PPO DiT A AiRLoc</td><td>3.14043.56173.84684.33624.90213.9574 3.53313.56843.80953.92324.51193.6545 3.25963.30213.04683.39154.47663.4953 1.09791.29791.30641.53622.07231.4621 GOMAA-Geo 2.43401.95741.31060.96600.65531.4647 GeoExplorer 2.2213 1.9702 1.3277 0.9021 0.8681 1.4579</td></tr><tr><td></td><td>DynCur-Geo 2.5447 1.85530.84260.35320.23831.1668 GOMAA-Geo2.38301.83401.21700.65960.62131.3430</td></tr><tr><td></td><td>G GeoExplorer 2.0170 1.9106 1.3277 0.8596 0.8936 1.4017 DynCur-Geo 2.47661.7447 0.8681 0.2000 0.2383 1.1055 GOMAA-Geo 2.30641.82981.3021 0.7915 0.74041.3940 T GeoExplorer 2.1191 2.0383 1.5915 1.2596 1.2851 1.6587</td></tr></table>

Table 2: Final grid distance (SG) on MM-GAG under $5 \times 5 ,$ $B = 1 0 ,$ , and $C = 4 , \ldots , 8 .$ . A/G/T denote aerial-image, ground-view, and text targets. PPO, DiT, and AiRLoc are evaluated only for aerial-image targets.

![](images/ad02bac40042e1ff64d87eca416fbc8a4a895d2535848de3151317d8cfe09695.jpg)  
(a) Multimodal Target-Guided Search

![](images/453e2524c56b4e40b449f078f43398fd625fbf45236b801341e590a739a78723.jpg)  
(b) Comparison of Search Trajectories  
Figure 3: Qualitative visualization of multimodal active geo-localization and test-time policy trajectories. In (a), blue, orange, and green target borders and their corresponding paths denote aerial-image, ground-view-image, and text cues, respectively. In (b), path colors identify DynCur-Geo (blue), GOMAA-Geo (orange), and GeoExplorer (green). Blue squares and yellow circles mark the start and goal cells

Because start–goal pairs are dynamically resampled during training, MM-GAG also tests unseen routes across all cue types; MASA, SwissView, and xBD further cover held-out scenes and appearance shifts.

Parameter Sensitivity. We vary the distance-gate lower bound $\lambda _ { \mathrm { m i n } }$ and PBRS coeficient β on nine evaluation settings. Table 4 shows that $\lambda _ { \operatorname* { m i n } } = 0 . 4 0 5 \mathrm { a n d } \beta = 0$ .10 achieve the best joint SR/SG result. Both sweeps peak at interior settings: weaker or stronger gating, and removing or increasing PBRS, worsen at least one metric. This supports balanced distance-gated exploration and progress shaping.

Qualitative Path Visualization. Fig. 3 illustrates both multimodal cue generality and trajectory quality. Our method makes broader early progress and becomes goal-directed near the target, whereas competing policies often revisit nearby cells or drift after plausible local moves. The comparison is consistent with the quantitative distance-wise pattern: the benefit of the proposed policy is most visible when early exploratory moves must be followed by a decisive final approach.

Cross-Scene and Disaster Transfer. Table 3(a) shows the best SR/SG pair for our method across MASA, SwissView, and xBD. The xBD-disaster setting uses a pre-disaster target cue with post-disaster search observations (Fig. 4), while SwissViewMonuments tests landmark-centered heldout scenes. Our method improves both metrics on the two xBD splits, rather than trading more successes for worse failed trajectories, and the SwissView results extend the gain to held-out landmark cues.

(a) Cross-scene, landmark, and disaster transfer
<table><tr><td rowspan="2">Task</td><td colspan="3">SR↑/SG↓</td></tr><tr><td>GOMAA</td><td>GeoExplorer</td><td>DynCur-Geo</td></tr><tr><td>MASA/A</td><td>0.5640/1.3600</td><td>0.5600/1.2560</td><td>0.5880/1.1440</td></tr><tr><td>xBD-pre/A</td><td>0.5427/1.4132</td><td>0.5080/1.5555</td><td>0.5852/1.2922</td></tr><tr><td>xBD-dis./A</td><td>0.5345/1.4194</td><td>0.5129/1.5452</td><td>0.5856/1.2839</td></tr><tr><td>Swiss100/A</td><td>0.5044/1.5692</td><td>0.4700/1.5568</td><td>0.5772/1.3444</td></tr><tr><td>SwissMon./A</td><td>0.4596/1.6878</td><td>0.4604/1.7598</td><td>0.5208/1.5139</td></tr><tr><td>SwissMon./G</td><td>0.4464/1.8333</td><td>0.4593/1.7311</td><td>0.5084/1.5098</td></tr></table>

(b) 10 × 10 long-range extension
<table><tr><td>Method</td><td colspan="5">SR by initial distance C</td></tr><tr><td></td><td>14</td><td>15</td><td>16</td><td>17</td><td>18</td></tr><tr><td>GOMAA</td><td>0.5350</td><td>0.4750</td><td>0.6250</td><td>0.7500</td><td>0.7600</td></tr><tr><td>GeoExplorer</td><td>0.3350</td><td>0.3850</td><td>0.3350</td><td>0.2200</td><td>0.2250</td></tr><tr><td>DynCur-Geo</td><td>0.5400</td><td>0.6750</td><td>0.7550</td><td>0.8850</td><td>0.8850</td></tr><tr><td colspan="6">(c) Enlarged-grid summary</td></tr><tr><td>Setting</td><td></td><td>Method</td><td>SR↑</td><td></td><td>SG↓</td></tr><tr><td>8 × 8, B = 24</td><td></td><td>GOMAA</td><td>0.6790</td><td></td><td>1.3450</td></tr><tr><td>C = 10-14</td><td></td><td>GeoExplorer</td><td>0.3680</td><td></td><td>3.0460</td></tr><tr><td></td><td></td><td>DynCur-Geo</td><td>0.7460</td><td></td><td>0.8590</td></tr><tr><td>10 × 10, B = 32</td><td></td><td>GOMAA</td><td>0.6290</td><td></td><td>1.8010</td></tr><tr><td>C = 14-18</td><td></td><td>GeoExplorer</td><td>0.3000</td><td></td><td>4.6770</td></tr><tr><td></td><td></td><td>DynCur-Geo</td><td>0.7480</td><td></td><td>0.7360</td></tr><tr><td>25 × 25, B = 60</td><td></td><td>GOMAA</td><td>0.1800</td><td></td><td>13.5080</td></tr><tr><td></td><td></td><td>GeoExplorer</td><td>0.0820</td><td></td><td>15.5840</td></tr><tr><td>C = 12-48</td><td></td><td>DynCur-Geo</td><td>0.2040</td><td></td><td>12.3120</td></tr></table>

Table 3: Transfer and long-range results. Panel (a) reports SR/SG under 5 × 5, B = 12; GOMAA denotes GOMAA-Geo. Panel (b) reports SR and panel (c) reports SR/SG on the MASA aerial test split. The 25 × 25 distances in panel (c) use a step size of four.

(a) Distance-gate lower bound $\lambda _ { \mathrm { m i n } }$
<table><tr><td> $\lambda _ { \mathrm { m i n } }$ </td><td>0.100</td><td>0.250</td><td>0.405</td><td>0.650</td><td>0.900</td></tr><tr><td>Avg. SR↑</td><td>0.4813</td><td>0.5009</td><td>0.5765</td><td>0.5089</td><td>0.4582</td></tr><tr><td>Avg. SG↓</td><td>1.9150</td><td>1.7052</td><td>1.2885</td><td>1.6526</td><td>1.9135</td></tr><tr><td colspan="6">(b) PBRS coefficient β</td></tr><tr><td>β</td><td>0.00</td><td>0.05</td><td>0.10</td><td>0.15</td><td>0.20</td></tr><tr><td>Avg. SR↑</td><td>0.5535</td><td>0.5134</td><td>0.5765</td><td>0.4933</td><td>0.5307</td></tr><tr><td>Avg. SG↓</td><td>1.3290</td><td>1.5409</td><td>1.2885</td><td>1.7120</td><td>1.4880</td></tr></table>

Table 4: Sensitivity to reward-shaping parameters under 5 × 5, $B = 1 0 ,$ , and $C = 4 , \dots , 8$ , macro-averaged over nine MASA, MM-GAG, SwissView, and xBD evaluation settings.

Long-Range Extension. Table 3(b,c) further evaluates enlarged MASA aerial grids. On 10 × 10 with $B = 3 2$ and $C \overset { \cdot } { = } \{ 1 4 , 1 5 , 1 6 , 1 7 , \overset { \cdot } { 1 8 } \}$ , our method obtains 0.7480 SR and 0.7360 SG, compared with 0.6290 and 1.8010 for GOMAA-Geo, and leads every distance bin. Our method also retains the highest SR on $8 \times 8$ and $2 5 \times 2 5$ , although all methods decline in the most demanding setting. Its consistently lower SG shows that failed episodes end closer to the target across

![](images/0472df6b90f2aeeb77d41b8ff0e857a0efcfc9fa5eff915ce52eae8715d5a83a.jpg)  
(b) Post-Disaster Search  
Figure 4: Qualitative xBD cross-disaster visualization. Yellow boxes indicate pre-disaster target regions and red boxes indicate the corresponding post-disaster regions. The lower panels show the pre-disaster target cue and the post-disaster search scene with a test-time policy trajectory; blue squares and yellow circles mark the start and goal cells.

grid scales.

Unified B = 12 Main Comparison. Table 5 compares MASA, MM-GAG, xBD-pre, and xBD-disaster under $\bar { B } =$ 12. Our method achieves the highest mean SR (0.6129) and leads all four settings. This comparison separates the gain from any single benchmark: the improvement appears on the standard aerial setting, multimodal aerial-target evaluation, and both disaster-related conditions under one common budget. The consistent ranking across aerial, multimodal, and disaster-transfer settings suggests that the reward design improves the search policy itself, rather than only exploiting a particular target modality or scene split. Notably, the gains on xBD-pre and xBD-disaster are close, indicating that the learned policy does not depend on unchanged visual appearance between the target cue and search scene.

Budget Sensitivity. Table 6 sweeps the search budget from B = 4 to B = 12. With $C = 4 , \ldots , 8$ , the smallest budgets leave some distance groups geometrically unreachable.

From B = 8 onward, our method consistently attains the highest SR and lowest SG. In particular, SR rises from 0.5326 at B = 8 to 0.6129 at B = 12, while SG decreases from 1.2760 to 1.2128, which confirms that more available actions improve both successful localization and the quality of failed episodes. At B = 12, this corresponds to a 0.0527 SR gain and a 0.2196 SG reduction relative to the strongest competing entry for each metric. This budget trend is important because smaller budgets mainly test whether the policy avoids wasteful wandering, whereas larger budgets test whether early exploration can be converted into final convergence. The simultaneous SR and SG improvements therefore indicate that dynamic weighting does not merely produce more isolated successes; it also leaves unsuccessful trajectories closer to the target.

<table><tr><td>Method</td><td>MASA</td><td>MM- GAG</td><td></td><td>xBD-pre xBD-dis.</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>Random</td><td>0.1000</td><td>0.0868</td><td>0.0925</td><td>0.0925</td><td>0.0929</td></tr><tr><td>PPO</td><td>0.2280</td><td>0.1991</td><td>0.1712</td><td>0.1676</td><td>0.1915</td></tr><tr><td>DiT</td><td>0.2280</td><td>0.2255</td><td>0.2409</td><td>0.2365</td><td>0.2327</td></tr><tr><td>GOMAA-Geo</td><td>0.5800</td><td>0.5464</td><td>0.5612</td><td>0.5534</td><td>0.5602</td></tr><tr><td>GeoExplorer</td><td>0.5640</td><td>0.5421</td><td>0.5233</td><td>0.5372</td><td>0.5417</td></tr><tr><td>DynCur-Geo</td><td>0.6200</td><td>0.6357</td><td>0.5973</td><td>0.5984</td><td>0.6129</td></tr></table>

Table 5: Unified B = 12 SR on MASA/A, MM-GAG/A, xBD-pre/A, and xBD-disaster/A; Avg. denotes their macroaverage.

(a) Success rate (SR)
<table><tr><td>Method</td><td> $B = 4$ </td><td> $B = 6$ </td><td> $B = 8$ </td><td> $B = 1 0$ </td><td> $B = 1 2$ </td></tr><tr><td>GOMAA</td><td>0.0440</td><td>0.1791</td><td>0.4891</td><td>0.5424</td><td>0.5602</td></tr><tr><td>GeoExplorer</td><td>0.0459</td><td>0.1842</td><td>0.4805</td><td>0.5281</td><td>0.5417</td></tr><tr><td>DynCur-Geo</td><td>0.0366</td><td>0.1790</td><td>0.5326</td><td>0.5917</td><td>0.6129</td></tr><tr><td colspan="6">(b) Final grid distance (SG)</td></tr><tr><td>Method</td><td>B = 4</td><td>B = 6</td><td>B = 8</td><td>B = 10</td><td> $B = 1 2$ </td></tr><tr><td>GOMAA</td><td>3.2143</td><td>1.9513</td><td>1.4263</td><td>1.4138</td><td>1.4324</td></tr><tr><td>GeoExplorer</td><td>3.2261</td><td>1.9501</td><td>1.4453</td><td>1.4550</td><td>1.4984</td></tr><tr><td>DynCur-Geo</td><td>2.8507</td><td>1.8584</td><td>1.2760</td><td>1.2304</td><td>1.2128</td></tr></table>

Table 6: Budget sensitivity (SR/SG), macro-averaged over MASA/A, MM-GAG/A, xBD-pre/A, and xBD-disaster/A.

<table><tr><td>Ext. Int. √</td><td>Gate PBRS C 三</td><td>4 C 二 5 C 二 6 C 二 7 C = 8 0.3617 0.4326 0.5801 0.7305 0.7773</td></tr><tr><td colspan="3">V 0.3248 √ 1 0.15180.10640.07940.0752 0.0468 √ √ 1 0.25250.34180.58160.81840.8369 Const 0.37870.39720.52200.59290.5787 L V 7 V Const 0.30500.40990.59860.78160.8426 √ Sin V 0.34890.42410.47800.40280.2780 √ √ Sin 0.31770.37020.48790.47660.4596 小 」 Power 0.28790.34750.44960.59860.6468 V Power 0.25250.3291 0.5191 0.68090.7660 V V Line 0.26950.37160.57870.84820.8426 L 7 √ Line √ 0.27090.41560.66380.86810.9163</td></tr></table>

Table 7: Reward ablation on MM-GAG/A/G/T (5 × 5, $B =$ 10); each C column averages A/G/T.

Reward and Gate Ablation. Table 7 evaluates reward combinations on MM-GAG over aerial-image, ground-view, and text cues. Each C column averages SR over the three target forms under 5 × 5, B = 10. External reward is essential, while the complete linear-gate and PBRS formulation is strongest at $C = 6$ to $C = 8$ , improving from 0.2709 at $C = 4$ to 0.9163 at $C = 8 ;$ intrinsic reward alone falls to 0.0468 at $C = 8$ . PBRS raises the mean SR for every gated schedule, but the linear gate remains best after shaping and leads at $C = 6$ to $C = 8$ . This pattern indicates that PBRS supplies broadly useful progress feedback, whereas the gate determines how efectively curiosity is allocated over the search horizon. In other words, curiosity is beneficial when it is treated as a progress-aware training signal, not as a static exploration bonus.

## Conclusion

This paper studied how curiosity should be allocated in multimodal active geo-localization, where sparse feedback, limited local observations, and heterogeneous target cues make search ineficient. We introduced distance-gated curiosity reward shaping, which modulates prediction-error intrinsic reward by remaining target distance and combines it with potential-based progress shaping. This design preserves exploratory behavior when the target is far away while reducing novelty-seeking pressure near the goal. Across multimodal, cross-scene, disaster-afected, and enlarged-grid settings, the resulting policy improves both success rate and final grid distance over the evaluated active geo-localization baselines. These results suggest that curiosity is most useful for AGL when conditioned on task progress, and they motivate future extensions to continuous UAV dynamics, richer map scales, and real deployment constraints.

## References

[1] Adamczyk, J.; Makarenko, V.; Tiomkin, S.; and Kulkarni, R. V. 2025. Bootstrapped Reward Shaping. Proceedings of the AAAI Conference on Artificial Intelligence, 39(15): 15302–15310.

[2] Anderson, P.; Wu, Q.; Teney, D.; Bruce, J.; Johnson, M.; Sünderhauf, N.; Reid, I.; Gould, S.; and van den Hengel, A. 2018. Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, 3674– 3683.

[3] Arandjelović, R.; Gronat, P.; Torii, A.; Pajdla, T.; and Sivic, J. 2016. NetVLAD: CNN Architecture for Weakly Supervised Place Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 5297–5307.

[4] Bellemare, M. G.; Srinivasan, S.; Ostrovski, G.; Schaul, T.; Saxton, D.; and Munos, R. 2016. Unifying Count-Based Exploration and Intrinsic Motivation. In Advances in Neural Information Processing Systems, 1471–1479.

[5] Berton, G.; Mereu, R.; Trivigno, G.; Masone, C.; Csurka, G.; Sattler, T.; and Caputo, B. 2022. Deep

Visual Geo-Localization Benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5396–5407.

[6] Burda, Y.; Edwards, H.; Pathak, D.; Storkey, A.; Darrell, T.; and Efros, A. A. 2019. Large-Scale Study of Curiosity-Driven Learning. In Proceedings of the International Conference on Learning Representations.

[7] Burda, Y.; Edwards, H.; Storkey, A.; and Klimov, O. 2019. Exploration by Random Network Distillation. In Proceedings ofthe International Conference on Learning Representations.

[8] Cepeda, V. V.; Nayak, G. K.; and Shah, M. 2023. Geo-CLIP: CLIP-Inspired Alignment between Locations and Images for Efective Worldwide Geo-localization. In Advances in Neural Information Processing Systems, 8690–8701.

[9] Dhakal, A.; Ahmad, A.; Khanal, S.; Sastry, S.; Kerner, H.; and Jacobs, N. 2024. Sat2Cap: Mapping Fine-Grained Textual Descriptions from Satellite Images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 533–542.

[10] Gupta, R.; Goodman, B.; Patel, N.; Hosfelt, R.; Sajeev, S.; Heim, E.; Doshi, J.; Lucas, K.; Choset, H.; and Gaston, M. 2019. Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 10–17.

[11] Hu, S.; Feng, M.; Nguyen, R. M. H.; and Lee, G. H. 2018. CVM-Net: Cross-View Matching Network for Image-Based Ground-to-Aerial Geo-Localization. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 7258–7267.

[12] Ji, Y.; He, B.; Tan, Z.; and Wu, L. 2025. Game4Loc: A UAV Geo-Localization Benchmark from Game Data. Proceedings of the AAAI Conference on Artificial Intelligence, 39(4): 3913–3921.

[13] Ji, Y.; He, B.; Tan, Z.; and Wu, L. 2025. MMGeo: Multimodal Compositional Geo-Localization for UAVs. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 25165–25175.

[14] Liu, L.; and Li, H. 2019. Lending Orientation to Neural Networks for Cross-View Geo-Localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5624–5633.

[15] Liu, S.; Zhang, H.; Qi, Y.; Wang, P.; Zhang, Y.; and Wu, Q. 2023. AerialVLN: Vision-and-Language Navigation for UAVs. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 15338– 15348.

[16] Mi, L.; Béchaz, M.; Chen, Z.; Bosselut, A.; and Tuia, D. 2025. GeoExplorer: Active Geo-localization with Curiosity-Driven Exploration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 6122–6131.

[17] Mnih, V. 2013. Machine Learning for Aerial Image Labeling. Ph.D. thesis, University of Toronto (Canada).

[18] Ng, A. Y.; Harada, D.; and Russell, S. 1999. Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping. In Proceedings of the 16th International Conference on Machine Learning, 278–287.

[19] Pathak, D.; Agrawal, P.; Efros, A. A.; and Darrell, T. 2017. Curiosity-Driven Exploration by Self-Supervised Prediction. In Proceedings of the 34th International Conference on Machine Learning, 2778– 2787.

[20] Pirinen, A.; Samuelsson, A.; Backsund, J.; and Åström, K. 2023. Aerial View Localization with Reinforcement Learning: Towards Emulating Search-and-Rescue. arXiv:2209.03694.

[21] Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; Krueger, G.; and Sutskever, I. 2021. Learning Transferable Visual Models from Natural Language Supervision. In Proceedings of the 38th International Conference on Machine Learning, 8748–8763.

[22] Sarkar, A.; Sastry, S.; Pirinen, A.; Zhang, C.; Jacobs, N.; and Vorobeychik, Y. 2024. GOMAA-Geo: GOal Modality Agnostic Active Geo-localization. In Advances in Neural Information Processing Systems, 104934–104964.

[23] Savinov, N.; Raichuk, A.; Marinier, R.; Vincent, D.; Pollefeys, M.; Lillicrap, T.; and Gelly, S. 2019. Episodic Curiosity through Reachability. In Proceedings of the International Conference on Learning Representations.

[24] Schmidhuber, J. 1991. A Possibility for Implementing Curiosity and Boredom in Model-Building Neural Controllers. In Proceedings ofthe International Conference on Simulation ofAdaptive Behavior: From Animals to Animats, 222–227. MIT Press/Bradford Books.

[25] Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and Klimov, O. 2017. Proximal Policy Optimization Algorithms. arXiv:1707.06347.

[26] Sekar, R.; Rybkin, O.; Daniilidis, K.; Abbeel, P.; Hafner, D.; and Pathak, D. 2020. Planning to Explore via Self-Supervised World Models. In Proceedings of the 37th International Conference on Machine Learning, 8583–8592.

[27] Shi, Y.; Liu, L.; Yu, X.; and Li, H. 2019. Spatial-Aware Feature Aggregation for Image-Based Cross-View Geo-Localization. In Advances in Neural Information Processing Systems, 10090–10100.

[28] Stadie, B. C.; Levine, S.; and Abbeel, P. 2015. Incentivizing Exploration in Reinforcement Learning with Deep Predictive Models. arXiv:1507.00814.

[29] Vo, N. N.; and Hays, J. 2016. Localizing and Orienting Street Views Using Overhead Imagery. In European Conference on Computer Vision, 494–509.

[30] Workman, S.; Souvenir, R.; and Jacobs, N. 2015. Wide-Area Image Geolocalization with Aerial Reference Imagery. In Proceedings of the IEEE International Conference on Computer Vision, 3961–3969.

[31] Yan, Q.; Zheng, J.; Reding, S.; Li, S.; and Doytchinov, I. 2022. CrossLoc: Scalable Aerial Localization Assisted by Multimodal Synthetic Data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 17358–17368.

[32] Zhang, X.; Shen, Z.; Cao, S.-Y.; Bai, X.; Li, Y.; Han, Z.; Wu, Z.; Ming, Q.; and Shen, H.-L. 2026. Learning Better UAV-Based Cross-View Object Geo-Localization from Multi-Modal Prompts: MoP-UAV Benchmark and MoPT Framework. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 12843– 12851.

[33] Zheng, Y.; Ying, J.; Duan, H.; Li, C.; Zhang, Z.; Liu, J.; Liu, X.; and Zhai, G. 2026. GeoX-Bench: Benchmarking Cross-View Geo-Localization and Pose Estimation Capabilities of Large Multimodal Models. Proceedings of the AAAI Conference on Artificial Intelligence, 40(16): 13485–13493.

[34] Zheng, Z.; Wei, Y.; and Yang, Y. 2020. University-1652: A Multi-View Multi-Source Benchmark for Drone-Based Geo-Localization. In Proceedings of the 28th ACM International Conference on Multimedia, 1395–1403.

[35] Zhu, S.; Shah, M.; and Chen, C. 2022. TransGeo: Transformer Is All You Need for Cross-View Image Geo-Localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1162–1171.

[36] Zhu, S.; Yang, T.; and Chen, C. 2021. VIGOR: Cross-View Image Geo-Localization beyond One-to-One Retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3640–3649.

[37] Zhu, Y.; Mottaghi, R.; Kolve, E.; Lim, J. J.; Gupta, A.; Fei-Fei, L.; and Farhadi, A. 2017. Target-Driven Visual Navigation in Indoor Scenes Using Deep Reinforcement Learning. In Proceedings of the IEEE International Conference on Robotics and Automation, 3357–3364.

## Supplementary Materials

## Scope and Diagnostic Protocol

The supplementary material reports additional trajectorylevel results for the fixed evaluation protocols used in the main paper. It includes route visualizations, route-dynamics summaries, cross-target evaluations, enlarged-grid stress tests, and implementation details for evaluation-time rollout. Aggregate values are reported when they identify the scope of a route set or connect a qualitative pattern to the corresponding evaluator. The analyses use the same data families as the main paper: MASA, MM-GAG, GeoExplorer’s SwissView100 and SwissViewMonuments splits, and xBD.

The material is organized to separate aggregate evidence from route-level interpretation. Table S1 and Tables S4– S8 provide the main quantitative additions: shared-route descriptors, training-pool selection, distance-stratified results, development hyperparameter checks, and enlargedgrid stress tests. Figures S1–S10 then explain how these scores arise from successful routes, persistent failure modes, matched same-task contrasts, realized reward traces, and step-level reward distributions. Figures S11–S14 extend the same analysis to multimodal targets, xBD pre/post search, Swiss ground-view targets, and larger search regions.

## Recorded Route Bank

The principal trajectory bank contains 2,205 SwissView-Monuments aerial-goal trajectories. It consists of 735 shared task keys evaluated by DynCur-Geo, GOMAA-Geo, and GeoExplorer. A key fixes the aerial scene, start cell, target cell, initial Manhattan distance, and repeat index, so all three trajectories within a key solve the same navigation problem with the same target cue. The bank uses a $5 \times 5$ grid, action budget $B = 1 0$ , initial distances $C \in \{ 4 , 6 , 8 \}$ , a fixed taskbank seed, greedy action selection, and legal-action masking. It contains 375 tasks at $C = 4 , 3 0 0 \mathrm { a t } C = 6$ , and 60 at $C = 8$ for each policy. Successful episodes terminate on reaching the target; unsuccessful episodes exhaust the ten-action budget. This supplement-only diagnostic bank is distinct from the $B = 1 2$ SwissViewMonuments transfer evaluation reported in the main paper.

The route bank is used for controlled path inspection. When methods are compared, the image, start cell, target cell, initial distance, repeat index, and action budget are identical. Unless another evaluator is specified, success rates, paired counts, and case frequencies in this supplement refer to these 735 shared tasks. Route galleries are grouped by behavior and matched outcome.

## Route Descriptors

For a route $\tau = ( p _ { 0 } , \dots , p _ { T } )$ , path eficiency is $C / T ,$ , capped naturally at one because C is the shortest grid distance. A revisited step enters a cell seen earlier in the same route. An immediate backtrack satisfies $p _ { t } = p _ { t - 2 }$ . Progress and regress steps respectively reduce and increase $d _ { 1 } ( p _ { t } , p _ { g } )$ . The target neighborhood is the set of cells with distance at most one. Post-entry drift counts distance-increasing actions after first entering this neighborhood.

(a) Outcome and path-eficiency descriptors
<table><tr><td>Method</td><td>C</td><td>Tasks</td><td>SR</td><td>SG</td><td>Eff.</td></tr><tr><td rowspan="3">DynCur-Geo</td><td>4</td><td>375</td><td>0.2507</td><td>2.5867</td><td>0.5279</td></tr><tr><td>6</td><td>300</td><td>0.6067</td><td>1.0867</td><td>0.7927</td></tr><tr><td>8</td><td>60</td><td>0.8500</td><td>0.4000</td><td>0.9600</td></tr><tr><td rowspan="3">GOMAA-Geo</td><td>4</td><td>375</td><td>0.2507</td><td>2.5653</td><td>0.5179</td></tr><tr><td>6</td><td>300</td><td>0.5900</td><td>1.0800</td><td>0.7782</td></tr><tr><td>8</td><td>60</td><td>0.6833</td><td>0.8667</td><td>0.9333</td></tr><tr><td rowspan="3">GeoExplorer</td><td>4</td><td>375</td><td>0.3333</td><td>2.2453</td><td>0.5464</td></tr><tr><td>6</td><td>300</td><td>0.4733</td><td>1.5400</td><td>0.7587</td></tr><tr><td>8</td><td>60</td><td>0.6000</td><td>1.1333</td><td>0.9200</td></tr></table>

<table><tr><td colspan="5">(b) Route-dynamics descriptors</td></tr><tr><td>Method</td><td>C</td><td>Revisit</td><td>Prog.</td><td>Reg.</td></tr><tr><td rowspan="3">DynCur-Geo</td><td>4</td><td>0.1903</td><td>0.6346</td><td>0.3654</td></tr><tr><td>6</td><td>0.0915</td><td>0.8420</td><td>0.1580</td></tr><tr><td>8</td><td>0.0500</td><td>0.9600</td><td>0.0400</td></tr><tr><td rowspan="3">GOMAA-Geo</td><td>4</td><td>0.1912</td><td>0.6307</td><td>0.3693</td></tr><tr><td>6</td><td>0.1216</td><td>0.8351</td><td>0.1649</td></tr><tr><td>8</td><td>0.0967</td><td>0.9233</td><td>0.0767</td></tr><tr><td rowspan="3">GeoExplorer</td><td>4</td><td>0.1648</td><td>0.6609</td><td>0.3391</td></tr><tr><td>6</td><td>0.1337</td><td>0.8023</td><td>0.1977</td></tr><tr><td>8</td><td>0.1283</td><td>0.9033</td><td>0.0967</td></tr></table>

Table S1: Distance-stratified descriptors on the 735 shared diagnostic tasks. SR is exact-hit success rate and SG is terminal grid distance. Ef. abbreviates path eficiency; revisit, progress, and regress are route-level means. These values characterize the route bank used for the following visual analysis.

Successful routes are grouped as eficient success $( T =$ C), detour success, or late convergence. Failed routes are grouped as post-entry drift, revisit/ping-pong, approachthen-regress, non-convergent, or partial progress. These deterministic descriptors support consistent visualization across figures. Blue squares denote starts, yellow circles targets, and yellow crosses failed terminal cells. Route colors are fixed throughout: blue for DynCur-Geo, orange for GOMAA-Geo, and green for GeoExplorer.

Each single-policy route panel places the target photograph in a compact side cue rail; failed panels also include the final-cell photograph. Each matched three-policy task uses one shared target photograph beside the route set. The photographs are grid-cell crops aligned with the aerial scene, while the full $5 \times 5$ route image remains uncropped and unobstructed. Failures are interpreted relative to the requested target cell rather than as generic spatial coverage.

Table S1 summarizes the distance-conditioned route statistics used by the galleries. ${ \mathrm { A t ~ } } C = 4 ,$ the main diferences occur near the target. $\mathrm { A t } C = 6 .$ , our method has similar SG to GOMAA-Geo with higher eficiency and fewer revisits. $\mathrm { A t } { \cal C } = 8$ , our method reaches 0.8500 SR with 0.9600 path eficiency, compared with 0.6833 and 0.6000 SR for the two baselines. The three distance groups have diferent sample counts, so comparisons are made within each stratum.

![](images/b864bbf370ec1c8177d395048c7921d3174ce263d3adfaafdc6464c0bd9cdee4.jpg)  
Figure S1: Distance-stratified route gallery for DynCur-Geo. Each row contains two successes followed by two failures at the same initial distance. Every side cue rail shows the target photograph; failed panels also show the final-cell photograph. S/F denotes success/failure, T is path length for a success, and the pair on a failure is minimum/final distance.

## Successful Route Mechanisms

Fig. S2 contains 12 recorded successes from three behavior groups. The gallery separates three ways in which a successful policy can use its action budget. Eficient successes follow a shortest path and provide the cleanest visible expression of target-directed search. Detour successes temporarily preserve or increase target distance but recover within the action budget. Late-convergence successes spend most of the route outside the target neighborhood and enter it near termination.

Across all 327 successful routes of our method in the shared bank, mean path eficiency is 0.9265, the mean revisit ratio is 0.0086, and immediate backtracks average 0.0795 per route. The corresponding successful-route eficiencies are 0.9046 for GOMAA-Geo and 0.9033 for GeoExplorer. The qualitative panels show the same pattern: most routes of our method make monotonic progress, with only a small number of corrective moves in the detour and late-convergence rows.

The eficient row shows target-directed motion with little route correction. The detour row contains temporary regress actions followed by recovery and termination. The late-convergence row shows broader motion while the target remains distant, followed by contraction near the end of the route. These patterns are consistent with the reward design, but a successful path alone does not isolate individual reward terms; the stepwise reward diagnostics below report the corresponding traces.

Of the 254 eficient successes in the recorded bank, 75 occur at C = 4, 131 at C = 6, and 48 at C = 8. Detour and late-convergence successes appear at all recorded distances. The resulting behavior is not a fixed explore-then-exploit schedule. Because the distance weight is state dependent, a regress action can raise the intrinsic weight again, and convergence can occur at diferent points in the route.

## Failure Taxonomy and Mechanism Boundaries

Fig. S4 contains 20 failures across five route categories. The examples include short-, medium-, and long-distance routes; urban, mountain, forest, shoreline, and water-dominated scenes; and both near-target and globally divergent endings. Target and terminal cues are shown together so that each failure can be interpreted as an endpoint error, a loop, or a global search error.

The full recorded bank contains 408 failures of our method. Post-entry drift accounts for 220 routes (53.9% of failures), and revisit/ping-pong accounts for 133 (32.6%). Together they cover 86.5% of failures. The remaining failures comprise 36 approach-then-regress, 12 non-convergent, and 7 partial-progress routes. Many failed routes reduce distance substantially before losing the correct endpoint.

Post-entry drift. These routes reach distance one and subsequently move away. In the top row of Fig. S4, the trajectory enters the goal neighborhood, but the policy selects an adjacent visually plausible cell or leaves the neighborhood after a local ambiguity. Distance gating reduces intrinsic weight near the target while preserving a small residual exploration weight. Final-cell confirmation is handled by the next policy action under the fixed action budget.

![](images/0810beaddbce19b3eff95dfad17c5768459c94c5b003b8825389fd7b861e64aa.jpg)  
Figure S2: Successful-route atlas for DynCur-Geo: four eficient successes, four detour successes, and four late-convergence successes. C is initial distance, S marks success, and T is path length. Each panel pairs the route image with its target-cell photograph in a compact side cue rail.

![](images/94e280f24a13c498e59a4259766c5a68482c06da5ef3a0ecb4a8b44590d2b722.jpg)

![](images/96a94d404c8a8972a70a0fb2a845584e3c953eb73a0e1debf6aead5f298707fc.jpg)

![](images/f45f7f74eea4fcf9c2c48489446f901681f25e63f119a76f39f3c02c6ad550b3.jpg)

![](images/631d12db4c612d6cd22a14110c0dde0e3064ec0dbcfc9e50f2ec6feef8a9c709.jpg)  
Figure S3: Aggregate route-shape descriptors on all 735 shared diagnostic tasks per policy. DynCur-Geo denotes the proposed method, matching the main-paper terminology. The method has slightly higher path eficiency and progress-step ratio, and slightly lower revisit and regress ratios, than the two baselines.

Revisit and ping-pong. These failures repeatedly enter previously visited cells or reverse the immediately preceding action. Prediction-error curiosity is model-relative rather than a direct count of spatial coverage, and repeated observations can still have nonzero prediction error. Recurrent history reduces, but does not eliminate, repeated-cell behavior.

The distance weight changes the magnitude of the intrinsic term but does not prohibit return transitions.

Approach then regress. These routes reduce distance to at most half of the initial value and later give back at least two cells of progress. When a route moves away from the target, the remaining-distance ratio and intrinsic weight rise again. This behavior supports recovery from an incorrect local hypothesis, but it also makes endpoint retention depend on the learned visual state.

![](images/ea90ecf584a4d03c09de2434746c276d7b3820d93359d5c283f66b83de9b4104.jpg)  
Figure S4: Failure atlas for DynCur-Geo. Rows show four examples each of post-entry drift, revisit/ping-pong, approach-thenregress, non-convergence, and partial progress. Each compact side cue rail compares target and final-cell photographs, while a yellow cross marks the final cell in the full route view. F marks failure; the numeric pair is minimum/final target distance.

Non-convergence and partial progress. Non-convergent routes finish at least as far away as they started; partialprogress routes improve SG without entering the target neighborhood. All 19 such failures in the recorded bank occur at C = 4. This concentration matches the short-range weakness in Table S1: when the target is initially close, a single wrong endpoint decision consumes a large fraction of the budget.

The paired target cues expose several plausible ambiguities, including repeated road blocks, homogeneous forest, water boundaries, and snow-covered terrain. These observations are used as qualitative context for route-level errors; quantitative claims are made from route descriptors and aggregate evaluator outputs.

## Matched Same-Task Contrasts

The paired bank permits controlled qualitative comparison with scene, target cue, start cell, target cell, distance, repeat, and action budget fixed. Table S2 reports every success combination across the 735 shared keys. The proposed policy succeeds alone on 89 tasks, while at least one baseline succeeds on 224 tasks where it fails. The two directions are both visualized.

Fig. S5 displays representative proposed-only successes with matched task columns. The visible advantage follows two recurring geometries: avoiding a repeated local loop and retaining a promising target-neighborhood approach. Because every column fixes the task and target cue across policies, path diferences reflect policy behavior under the

![](images/3749d51424d2223b4e5181f3118259df2d7794ee78292c89171355206c1c6c66.jpg)  
Figure S5: Four same-task cases in which DynCur-Geo succeeds and both baselines fail, shown as matched task columns and method rows. The columns cover C = 4, 6, 8 while giving each route panel suficient space. Start, target, and budget are identical within a column; the route panels omit target-photo insets so that the paths remain unobstructed. $\bar { S } / T$ gives success and path length; F/SG gives failure and terminal distance.

<table><tr><td>Successful methods</td><td>C = 4</td><td> $C = 6$ </td><td>C = 8</td><td>Total</td></tr><tr><td>None</td><td>155</td><td>29</td><td>0</td><td>184</td></tr><tr><td>GeoExplorer only</td><td>57</td><td>15</td><td>1</td><td>73</td></tr><tr><td>GOMÁA-Geo only</td><td>44</td><td>41</td><td>2</td><td>87</td></tr><tr><td>GOMAA-Geo + GeoExplorer</td><td>25</td><td>33</td><td>6</td><td>64</td></tr><tr><td>DynCur-Geo only</td><td>40</td><td>39</td><td>10</td><td>89</td></tr><tr><td>DynCur-Geo +</td><td>29</td><td>40</td><td>8</td><td>77</td></tr><tr><td>GeoExplorer DynCur-Geo +</td><td>11</td><td>49</td><td>12</td><td>72</td></tr><tr><td>GOMAA-Geo All three</td><td>14</td><td>54</td><td>21</td><td>89</td></tr></table>

Table S2: Success-combination counts on shared diagnostic tasks. Each row lists the methods that hit the target within $B = 1 0 ;$ omitted methods failed on the same task.

same start–goal pair.

At C = 8, the routes of our method show the clearest longrange pattern: exploration remains active while the target is distant, and the distance weight decreases near the final cells. The corresponding traces appear in the stepwise reward diagnostics below.

Fig. S6 shows the complementary cases. Stronger longdistance averages can coexist with individual scenes in which a baseline retains the correct local hypothesis while our method loops, overshoots, or exits the target neighborhood.

These contrasts separate global progress from local target confirmation. A baseline may approach the correct neighborhood while matching a visually plausible distractor, whereas a successful route must preserve the target-conditioned hypothesis through the final transition.

The paired pattern frequencies show the same distance dependence. Complete three-policy failure falls from 41.3% at C = 4 to 9.7% at C = 6 and zero at $C = 8 ,$ while allthree success rises from 3.7% to 18.0% and 35.0%. Longer routes provide more visible progress transitions, whereas short routes concentrate the decision around target-cell confirmation.

![](images/7d9dab1ff5be9eb6d9f5dedb996f2dcf9d610a28ed3f922ab2948f4d7e4cb2f9.jpg)  
Figure S6: Four matched cases in which DynCur-Geo fails and at least one baseline succeeds, again shown as matched task columns and method rows without target-photo insets. The layout and annotations match Fig. S5.

## Stepwise Reward Diagnostics

The training reward is

$$
r _ { t } = r _ { \mathrm { e x } , t } + \lambda _ { t } r _ { \mathrm { i n } , t } + r _ { \Phi , t } ,\tag{S1}
$$

where the linear distance-adjustment function is

$$
\rho _ { t } = \mathrm { c l i p } \bigg ( \frac { d _ { 1 } ( p _ { t } , p _ { g } ) } { \operatorname* { m a x } ( d _ { 1 } ( p _ { 0 } , p _ { g } ) , 1 ) } , 0 , 1 \bigg )\tag{S2}
$$

$$
\lambda _ { t } = \lambda _ { \operatorname* { m i n } } + ( 1 - \lambda _ { \operatorname* { m i n } } ) \rho _ { t } \qquad \lambda _ { \operatorname* { m i n } } = 0 . 4 0 5 .\tag{S3}
$$

The extrinsic term gives the terminal hit a positive reward, gives an accepted progress transition a smaller positive reward, and penalizes invalid stays, revisits, and non-progress moves. In the recorded implementation, this accepted-progress branch uses the configured progress comparator, with the final checkpoint using the squared Euclidean cell comparator for that branch; Manhattan grid distance is still used for the displayed distance traces, for $\lambda _ { t } ,$ and for the PBRS potential. The running best-progress state is initialized from the start–goal distance and, in the reported configuration, is committed after a reward-positive transition rather than after every displayed Manhattan decrease.

The intrinsic term is computed from the frozen environment-modeling transformer’s next-state prediction error,

$$
r _ { \mathrm { i n } , t } = 0 . 2 5 \left[ 2 \frac { \mathrm { M S E } ( \hat { h } _ { t + 1 } , h _ { t + 1 } ) - 0 . 8 } { 0 . 1 } - 1 \right] ,\tag{S4}
$$

where $\hat { h } _ { t + 1 }$ and $h _ { t + 1 }$ are predicted and target hidden representations. The shaping term uses the following potentialbased form.

$$
\begin{array} { c l c r } { { r _ { \Phi , t } = \beta ( \gamma \Phi ( s _ { t + 1 } ) - \Phi ( s _ { t } ) ) } } \\ { { \Phi ( s _ { t } ) = - \displaystyle \frac { d _ { 1 } ( p _ { t } , p _ { g } ) } { 2 K - 2 } , } } \end{array}\tag{S5}
$$

with $\beta = 0 . 1 0 , \gamma = 0 . 9 3$ and grid width $K = 5$ for the standard setting. Target distance and reward terms are used for training and for the diagnostic plots. At test time, action selection uses the policy state and the legal-action mask. Recorded trajectories are replayed with their task targets to recover the reward decomposition along each sequence.

Fig. S7 compares successes and failures at matched initial distances. In the three shortest-path successes, target distance decreases monotonically and the distance weight contracts on every transition. For $C = 8 ,$ , it falls from 0.9256 after the first action to 0.405 at the target. The mean absolute raw intrinsic reward is 0.4323, while the mean absolute gated contribution is 0.3038. Across seven full-reward cases (six displayed traces plus the complete-reward control in Fig. S9), the realized gated-to-raw magnitude ratio ranges from 0.405 to 1.0 and averages 0.7645.

![](images/a72cceaba73db81808e83479e366ed88401446ceac74b9cfc4f48ef258b88999.jpg)  
Figure S7: Matched-distance stepwise diagnostics. Each row compares a success (left) with a failure (right) at the same C. Route panels include the exact target cue. Curves show remaining distance, the distance gate, and realized reward components. Successful routes contract the gate toward 0.405; failures show that regress actions reactivate the gate.

The three reward terms have diferent roles. External reward gives the dominant sign for target progress and the terminal hit. The intrinsic term is scene- and transitiondependent: it can be positive or negative after normalization and is not itself a target-direction signal. Potential shaping is small but directionally consistent. Over the same seven recorded full-reward cases, it averages (+0.0154) on 40 progress transitions and (-0.0076) on 18 regress transitions.

Fig. S8 shows the same decomposition after pooling transitions by realized movement type. Progress and terminal steps keep positive mean total reward, whereas regress steps remain negative despite a larger average distance weight. The mean distance weight is 0.7369 on progress transitions, 0.8995 on regress transitions, and 0.4050 at terminal hits.

Thus, moving away from the target can reopen exploration pressure, but the total reward sign still favors target-directed motion.

The failures show where the same mechanism can lose the endpoint. In the $C = 6$ failure, distance first falls from six to three, then oscillates and rises to eight; the distance weight falls to 0.7025 and returns to one as the route diverges. The $C = 4$ failure makes one useful first move and then spends most of the budget increasing distance despite the fixed target cue. The $C = 8$ failure preserves several correct transitions before entering a local oscillation. These rows separate early target-hypothesis selection at short range from target-hypothesis retention after a longer approach.

When a route moves away from the target, the distance weight increases again. This can help escape an incorrect local hypothesis, but it can also sustain motion toward a visually salient alternative. Mean total reward remains positive on progress transitions and negative on regress transitions, so the target-progress sign is preserved even when the trajectory later diverges. At the same action index, success and failure can carry diferent distance weights because their remaining distances difer.

![](images/e2d40ac579c093657f5aade852b31a7c884872f1ec23cf363482ec4da0d1a44f.jpg)

![](images/d69ff037881abac306e14fc6faa4343e975d23106cbc7bde757e6d8252ea3511.jpg)

![](images/ddc8ba10ea4e4e9a6848e951ec5408483635e4175105ad67f95579f42f7afa71.jpg)

![](images/2880b8b136b98fa6993c634066a4f0415c244637114b23f910586953c82ea913.jpg)  
Figure S8: Step-level reward distribution on the seven recorded full-reward diagnostic routes used for the mechanism plots. The summary covers 58 transitions and groups them by whether the executed action reduces target distance, increases target distance, or reaches the terminal target cell. The plot decomposes each transition into external reward, gated intrinsic contribution, PBRS shaping, and total reward; it is a diagnostic decomposition rather than a full-route-bank estimate.

The six traces quantify this split. The successful $C =$ 4, 6, 8 routes have mean distance weights 0.628, 0.653, and 0.665, and retain 0.800, 0.729, and 0.703 of the absolute raw intrinsic magnitude. The corresponding failure means are 0.985, 0.881, and 0.740, because regress motion repeatedly raises the weight. Mean total reward on regress actions remains negative in all three failures (−1.027, −0.777, and −0.833). These values describe the displayed diagnostics; the next subsection uses the full bank.

Fig. S9 compares independently trained controls on one fixed task. All three policies initially reduce target distance. The complete-reward policy survives several corrections and reaches the target on the tenth action. Without the distance weight, the route reaches distance one at step five but oscillates and finishes at distance two. Without PBRS, the route reaches distance one at step nine and then moves away. This case isolates one training factor at a time, while the aggregate ablation in the main paper reports the corresponding average efect.

## Population-Level Route Dynamics

Fig. S10 extends the six displayed traces to the complete bank. At C = 8, our method follows an almost deterministic contraction through step seven and reaches a cumulative hit rate of 0.850 by step ten, compared with 0.683 for GOMAA-Geo and 0.600 for GeoExplorer. At C = 6, its mean distance falls below both baselines by step four and remains lowest at termination. $\begin{array} { r } { \operatorname { A t } C = 4 . } \end{array}$ early contraction stalls after step four and mean distance begins to rise. In this controlled bank, longer routes often provide several unambiguous progress transitions, whereas short routes expose target confirmation quickly.

The state-conditioned curve isolates the near-goal bottleneck. For our method, the probability of a progress action is 1.000 at remaining distance eight, 0.914 at six, 0.770 at four, and 0.431 at distance one; the complementary regress probability at distance one is 0.569. Among 547 routes that reach distance one, 59.8% subsequently hit the target, while 47.7% leave the target neighborhood at least once. The two events can both occur because a route may exit and later recover. The image-cluster intervals summarize sampling variation within the trajectory bank.

## Cross-Domain Targets and Enlarged Search Regions

The shared Swiss aerial bank isolates route geometry under one target form. This section reports additional results in which the target modality, search image domain, grid size, or action horizon changes while the evaluator remains target conditioned.

## Target/Search Evaluation Contracts

All cross-domain experiments in this section use the same evaluator contract. A target cue identifies the requested destination before rollout, and the policy searches the corresponding observation stream under a fixed action budget. The cue may be an aerial crop, a ground photograph, a text-grounded target representation, or a pre-disaster target image. These forms change the evidence used to specify the goal, but success is still counted only when the trajectory reaches the target grid cell.

The displayed target cue defines the task instance and is encoded through the target-conditioned input pipeline used by the evaluator. It is not refreshed as a new observation at every step. Each route image shows the realized greedy rollout after legal-action masking. Aggregate tables summarize

Complete reward success

![](images/4b7494e176ed2761972470f2b1dcc43c6db9b540b703506a76e0fc0e4aed30da.jpg)

![](images/9a307dc722c806abd38a2040ddf39d75f24c0d88425c89fb2297d0d0cae25398.jpg)

![](images/40893c8bf2ec789543ddca0693e7fa00efa57f7d3ea7f901134572293c273d37.jpg)

![](images/2d5272b54a2f2aad4b0ecd623939da1cbb50305b14e6c390cdd461ff5155432e.jpg)

![](images/ceed9ba6f20cc43d965eb8b9e2bb91b588a488d7374d7a9ea428824adb9a11f4.jpg)

![](images/a1a8d873f2d73fcf1c0037fae3905492b4a75647d1749164eea161824d12b1fe.jpg)  
Figure S9: Controlled route comparison on one fixed C = 6 task. The complete-reward policy reaches the target; the separately trained policies without the distance gate or without PBRS both finish at SG = 2. Only one training factor changes in each comparison.

![](images/01efe4a9cba049cb7baca7570237fe34e29eeff7c70cce402551d567ce00a571.jpg)

![](images/8490834353e202a5f72830595e7bb2e436e73faa505cf1004602bde98f1498ee.jpg)

![](images/982e50afd7af476f02286f8be3d677f11958a618cc02a619044922ad992fb1c4.jpg)

![](images/085dca2137e98def394a6795273ae78ab02ecbfd662ad05fbb1244dc421baa7a.jpg)

![](images/9b5ad09afcd62c8f5ba43626a0363ae10a0792fc2d697f6a9eb55abde12959d6.jpg)

![](images/815490f07eecb2708ecd8bc1e1a8d507f1981c4114d957ff4509583695f4f2cf.jpg)  
Figure S10: Population dynamics over all 2,205 recorded routes. Bands and error bars are 95% percentile intervals from 2,000 deterministic bootstrap resamples clustered by aerial image. Top: mean remaining distance by initial distance. Bottom: cumulative exact-hit rate, progress probability conditioned on current remaining distance, and outcomes after first reaching distance one. Curves summarize the full trajectory bank.

success under a protocol, while route figures show the corresponding endpoint decisions, local drift, and domain-shift efects.

Aerial-to-aerial tasks mainly test spatial search once the target appearance is directly available. MM-GAG adds multimodal target specification. SwissViewMonuments groundto-aerial tasks test cross-view endpoint confirmation. xBD pre/post tasks keep the target fixed while changing the searched image domain. Equal aggregate SR can correspond to diferent route-level behavior.

Fig. S11 specifies the target evidence, the searched image stream, and the input path used by the evaluator. The subsequent route figures then show the realized greedy rollout under that fixed contract. In Fig. S12, the xBD pair is held constant except for the search domain. In Fig. S13, MM-GAG remains within one aerial family, whereas SwissViewMonuments requires ground-to-aerial endpoint confirmation. In Fig. S14, the stress test increases grid size and action horizon.

## xBD Pre/Post Same-Task Analysis

The xBD reproduction uses the fixed pre/post evaluation task bank defined for the main experiments. Each setting contains episodes under $5 \times 5 , B = 1 2 , C = 4 , \ldots , 8 ,$ five repeats, greedy actions, and task seed 20260516. The target is always a pre-disaster cell. The search image is pre-disaster in xBD-pre and post-disaster in xBD-disaster, so the task tests whether a policy trained for target-conditioned search remains stable when the search observation changes after damage.

Fig. S11 specifies the target evidence and searched image stream. Fig. S12 fixes the task tuple and shows how the realized route changes when only the search domain changes. Aggregate xBD scores can remain nearly unchanged while individual paired tasks reverse outcome.

For all cross-domain cases, the evaluator uses the fixed target cue to define termination and metrics, keeping the panels comparable to the Swiss route bank.

The aggregate behavior is nearly invariant to the image shift. SR/SG are 0.5973/1.3018 before disaster and 0.5984/1.2813 after disaster. Across the 20,000 paired tasks, 9,353 succeed in both domains, 5,439 fail in both, 2,593 succeed only before disaster, and 2,615 succeed only after disaster. The discordant counts almost cancel at the aggregate level; Fig. S12 shows both directions of task-level reversal under the same start, target, distance, repeat, and budget.

Route dynamics are also stable. Successful path eficiency is 0.9304 before disaster and 0.9322 after disaster; revisit ratios are 0.1882 and 0.1888. Among episodes that reach distance one, 73.16% and 73.32% subsequently hit the exact target, while 34.65% and 34.75% leave the target neighborhood at least once. Under this reproduction split, appearance change modifies individual decisions without increasing aggregate drift.

## Aerial and Ground-View Target Routes

Fig. S13 separates two target-confirmation regimes. In MM-GAG, the aerial cue and search image share a view family, so the policy must retain the target hypothesis across a larger stitched scene. The MM-GAG evaluator uses new target-conditioned tasks on the indexed image set rather than a held-out-image split. In SwissViewMonuments, the target is a ground photograph and the search image is aerial; the displayed rows span castle, waterfront, civic-building, and dam/mountain landmarks. This setting adds cross-view endpoint confirmation after global progress.

Table S3 shows a sharper endpoint problem for the Swiss ground-target setting. Its revisit ratio is 0.2176 versus 0.1732 for MM-GAG aerial targets, and 46.40% of episodes that reach distance one later exit the neighborhood versus 36.24%. Because the rows difer in both dataset and target modality, the comparison is descriptive rather than a controlled modality ablation.

## Long-Range Stress Test

The aggregate SR of our method is 0.746, 0.748, and 0.204 on $8 \times 8 , 1 0 \times 1 0$ , and $2 5 \times 2 5$ , respectively, compared with 0.679, 0.629, and 0.180 for GOMAA-Geo. The largest grid remains dificult for all methods. On $2 5 \times 2 5$ , SR varies strongly by distance stratum, rising from 0.02 at $C = 1 2$ to 0.58 at $C = 4 8$ for our method. The comparison is made under the same grid, budget, and distance setting for each row.

## Additional Quantitative Evidence

This section reports additional numerical studies associated with the main-paper protocol. The tables cover training-pool selection, distance-stratified results for the unified scenarios, development hyperparameters, and enlarged-grid stress tests. They use the same evaluation definitions unless a separate grid size, budget, or distance range is stated.

## Training-Pool Selection

The training image pool is not a fixed route list. For each sampled image, training episodes are generated online by resampling a start cell, a goal cell, and an initialdistance constraint. Dynamic sampling yields many targetconditioned navigation episodes from the same image pool, while evaluation task banks remain fixed. Table S4 reports the controlled training-pool study. Average SR/SG values are macro-averaged over nine evaluation settings: MASA/A, MM-GAG/A/G/T, SwissView100/A, SwissView Monuments/A/G, xBD-pre/A, and xBD-disaster/A.

The best aggregate result comes from the 184-image MASA+MM-GAG pool. MASA provides the base aerial search distribution, while MM-GAG adds target-modality diversity. Adding SwissView100 increases the image count but reduces the macro-average. The final pool uses the smaller MASA+MM-GAG combination.

## Distance-Stratified Main Scenarios

The main paper reports the unified B = 12 comparison as scenario-level averages. Table S5 adds the distance decomposition for the same rows of our method. Short-distance cases are not always easier: at C = 4 and $C = 5 ,$ many failures are endpoint-confirmation errors near the target, while longer settings provide more room for repeated progress transitions before the final cell is reached.

d. Both fail C=5

c. Both success C=7  
Target Aerial Patch  
![](images/ddbd9bb80f419cf3154eed45dbc10abac63ed8e67236f7ca70f07423337ebbdd.jpg)

Aerial Search Area  
![](images/1d2cc0d8055fbf5a6b4e2071ea1d51c5b9ac0fcf7f9cfe362d4674bbee6355a1.jpg)

Target Ground View with Text Description  
![](images/632b4bc8bb026a9bf997d9465de78952bbdd74bcfe10a94bc5b315274ba2f1f0.jpg)  
The clock tower is topp ed with a pointed roof a nd features a clock face on its front side.

Search Area  
a. MASA  
![](images/e02043339d39389b81a4b3659b928c841f49b5037a653b3d45ccaf5ebbccf857.jpg)

![](images/0b3a72232732ff5135048ed905e8b102ebd08fd612034b34ee13c8167393a02d.jpg)

Aerial Search Area  
![](images/658c6bc8135a0d54e42920016a6e7a9cb0f3fa589df25ebd55ada8fe61664c73.jpg)  
c. SwissView  
Pre-Disaster Target Image

## b. MM-GAG

![](images/29abd98341265e49771b9c9afa4f9d0a1064bd5ebd5edac3c71981953fb796aa.jpg)

Post-Disaster Target Image  
![](images/0c8511be95860e9db70f0d050e21dd2e5d91f87abb0b6331af398b57c60e50cc.jpg)  
d. xBD

Figure S11: Target/search protocol illustrations for aerial, MM-GAG, SwissView, and xBD inputs. The target cue is fixed before rollout, and the policy searches the corresponding observation stream.  
a. Pre-only success C=8  
![](images/b5eade06a5407f24165c13689718c395fe539d95b4d01b0d896623af82b190de.jpg)

![](images/91a068c9c13560f7903dbec861c312c96013613384023922fe876c586caaa456.jpg)

![](images/ed09b13a4f441dd452ddea9695f42e86b0d9b3385b488a9be7eafab7189e6ab5.jpg)

b. Post-only success C=6  
![](images/29a159e0e5eac7d950a4bb1c55306b3658c873989694e91762894ef0070b4577.jpg)

![](images/3cb083438cd6f399f956598845b469ed5c85f6e6c857b80a4a3c553d2dbe460d.jpg)

![](images/a3bb620ab7232136bf9933c01eefe39f8f314ecb9a6554a5ea411f91759c79d2.jpg)

![](images/3b437f297a2b06792f237a2e994d1af0a3af6dc08749b82120fdb33eb8a3be00.jpg)

![](images/b4decf6a8301a9abb4957798796d5c1d6220eb8f8495deb379914957443bc9f0.jpg)  
Figure S12: Recorded xBD routes on four exactly matched tasks. Panels (a)–(d) show pre-only success, post-only success, success under both domains, and failure under both, respectively. Each panel pair fixes pair, start, pre-disaster target, C, repeat, and budget; only the search image changes from pre-disaster (left route) to post-disaster (right route). Target-cue thumbnails are omitted so the route panels remain large and unobstructed.

![](images/2557d83bbc2c1fcc3086dea9f21d927dc69745fd9a22d2650f1d3bd3e84831d8.jpg)  
Figure S13: Recorded cross-modal routes. Each row pairs an MM-GAG aerial target and route (left) with a SwissViewMonuments ground-photo target and aerial route (right). Cases cover long- and short-distance successes, near misses, and larger terminal errors. Swiss targets span castle, waterfront, civic-building, and dam/mountain scenes. The MM-GAG evaluator uses all 47 indexed samples; the Swiss rows use the controlled ground-to-aerial route set described in this supplement.

<table><tr><td>Recorded evaluator</td><td>Episodes</td><td>SR</td><td>SG</td><td>Succ. eff.</td><td>Revisit</td><td>Progress</td><td>Hit after d = 1</td><td>Exit after d = 1</td></tr><tr><td>xBD pre, aerial target</td><td>20,000</td><td>0.5973</td><td>1.3018</td><td>0.9304</td><td>0.1882</td><td>0.7585</td><td>0.7316</td><td>0.3465</td></tr><tr><td>xBD disaster, pre-disaster target</td><td>20,000</td><td>0.5984</td><td>1.2813</td><td>0.9322</td><td>0.1888</td><td>0.7599</td><td>0.7332</td><td>0.3475</td></tr><tr><td>MM-GAG, aerial target</td><td>1,175</td><td>0.6357</td><td>1.1319</td><td>0.9278</td><td>0.1732</td><td>0.7720</td><td>0.7477</td><td>0.3624</td></tr><tr><td>SwissMon ground, max80</td><td>1,380</td><td>0.5420</td><td>1.4457</td><td>0.8714</td><td>0.2176</td><td>0.7124</td><td>0.6997</td><td>0.4640</td></tr></table>

Table S3: Route-level descriptors recomputed from every saved trajectory in four evaluators. Succ. ef. is shortest distance divided by path length on successful episodes. The last two columns condition on first reaching distance one. The rows have diferent dataset scopes and are diagnostic rather than a controlled modality ablation.

## Development Hyperparameter Checks

Table S6 reports development checks for entropy, the distance lower bound, PBRS, training budget, and validation-distance selection. These rows are not separate baselines.

## Detailed Enlarged-Grid Results

Fig. S14 summarizes the enlarged-grid stress tests. Tables S7 and S8 give the numerical decomposition. The 8 × 8 and 10 × 10 settings show gains over both baselines across most

distance strata. The 25 × 25 setting remains dificult for all methods, but our method has the best aggregate SR/SG and higher SR at several larger distance bins.

## Discussion and Limitations

## Supported Claims

The route evidence supports three claims. Recorded successful routes have high eficiency, low revisit ratios, and few immediate reversals relative to their displayed target cues. The realized distance weight attenuates intrinsic reward as remaining distance falls and increases it again when a route diverges. Potential shaping changes sign with progress versus regress transitions. The complete-bank curves further show high progress probability when the target is distant and lower progress probability in the immediate target neighborhood.

![](images/90f99e0c234ed5a736a621571e1c531d23ea8209b089f8c182c82e44b9c8e184.jpg)

![](images/6fc8544389bc58b2012cfc987abd8967dc957721f481f65768310047cf3870ee.jpg)

![](images/b5ea30343688e99a891ac7c47637ddba6becb3709b031efb848ede6815eabc3d.jpg)

Figure S14: Formal MASA enlarged-grid results. Panels (a)–(b) report per-distance SR for ${ 8 \times 8 ( B = 2 4 , C = 1 0 , \dots , 1 4 ) }$ and $2 5 \times 2 5 ( B = 6 0 , C = 1 2 , 1 6 , \dots , 4 8 )$ ; panel (c) reports aggregate SR for $8 \times 8 , 1 0 \times 1 0$ , and $2 5 \times 2 5$ , where $1 0 \times 1 0$ uses B = 32 and $C = 1 4 , \dotsc , 1 8 .$
<table><tr><td>Training pool</td><td>Images</td><td>Avg. SR</td><td>Avg. SG</td></tr><tr><td>MASA+MM-GAG</td><td>184</td><td>0.6033</td><td>1.2543</td></tr><tr><td>MASA+SwissView100</td><td>237</td><td>0.5684</td><td>1.4191</td></tr><tr><td>MM-GAG+SwissView100</td><td>147</td><td>0.5497</td><td>1.7869</td></tr><tr><td>MASA+MM-</td><td>284</td><td>0.5232</td><td>1.7952</td></tr><tr><td>GAG+SwissView100</td><td></td><td></td><td></td></tr><tr><td>SwissView100</td><td>100</td><td>0.5103</td><td>2.0490</td></tr><tr><td>MASA</td><td>137</td><td>0.4443</td><td>2.1380</td></tr><tr><td>MM-GAG</td><td>47</td><td>0.4328</td><td>2.0731</td></tr></table>

Table S4: Training-pool selection under the unified evaluator. The final pool uses MASA+MM-GAG: 137 MASA images and 47 MM-GAG images. The comparison uses dynamically sampled navigation episodes from each image pool.

(a) Success rate by scenario and initial distance
<table><tr><td>Scenario</td><td> $\operatorname { A v g } .$ </td><td> $C = 4$ </td><td> $C = 5$ </td><td> $C = 6$ </td><td> $C = 7$   $C = 8$ </td></tr><tr><td>MASA/A</td><td>0.6200</td><td>0.2400</td><td>0.4000</td><td>0.6400 0.8800</td><td>0.9400</td></tr><tr><td>MM-GAG/A</td><td>0.6357</td><td>0.2979</td><td>0.3915</td><td>0.6894 0.8766</td><td>0.9234</td></tr><tr><td>xBD-pre/A</td><td>0.5973</td><td>0.3033</td><td>0.4065</td><td>0.6230 0.8115</td><td>0.8423</td></tr><tr><td>xBD-dis./A</td><td>0.5984</td><td>0.2923</td><td>0.4085</td><td>0.6290 0.8180</td><td>0.8443</td></tr></table>

(b) Final grid distance by scenario and initial distance
<table><tr><td>Scenario</td><td> $\operatorname { A v g } .$ </td><td> $C = 4$ </td><td> $C = 5$ </td><td> $C = 6$   $C = 7$ </td></tr><tr><td>MASA/A</td><td>1.1360</td><td>2.6800</td><td>1.5200 1.0000</td><td>0.2400 0.2400</td></tr><tr><td>MM-GAG/A</td><td>1.1319</td><td>2.5021</td><td>1.6894 0.8681</td><td>0.3617 0.2383</td></tr><tr><td>xBD-pre/A</td><td>1.3018</td><td>2.4470</td><td>1.8755 1.1565</td><td>0.5125 0.5175</td></tr><tr><td>xBD-dis./A</td><td>1.2813</td><td>2.4475</td><td>1.8685 1.0945</td><td>0.4960 0.5000</td></tr></table>

Table S5: Distance-stratified DynCur-Geo results for the unified 5×5, B = 12, $C = 4 , \ldots , \ell$ 8 evaluator. Panel (a) reports SR and panel (b) reports SG. The same fixed task-bank protocol is used as in the unified main comparison.

The shared-bank SR values are close for the three methods, but the paired counts show diferent route geometries. our method contributes additional successes on long or loopprone tasks, while baseline successes remain visible on several short-range endpoint cases.

(a) Entropy coeficient c<sub>e</sub>
<table><tr><td></td><td>0.0025</td><td>0.0050</td><td>0.0075</td><td>0.0100</td></tr><tr><td>Avg. SR</td><td>0.5497</td><td>0.5765</td><td>0.4344</td><td>0.4729</td></tr><tr><td>Avg. SG</td><td>1.4166</td><td>1.2885</td><td>2.0582</td><td>1.7864</td></tr></table>

(b) Distance lower bound $\lambda _ { \mathrm { m i n } }$
<table><tr><td></td><td>0.250</td><td>0.350</td><td>0.405</td><td>0.500</td><td>0.650</td></tr><tr><td>Avg. SR</td><td>0.5009</td><td>0.4861</td><td>0.5765</td><td>0.4987</td><td>0.5089</td></tr><tr><td>Avg. SG</td><td>1.7052</td><td>1.6788</td><td>1.2885</td><td>1.6200</td><td>1.6526</td></tr></table>

(c) PBRS coeficient β
<table><tr><td></td><td>0.00</td><td>0.05</td><td>0.10</td><td>0.15</td><td>0.20</td></tr><tr><td>Avg. SR</td><td>0.5535</td><td>0.5134</td><td>0.5765</td><td>0.4933</td><td>0.5307</td></tr><tr><td>Avg. SG</td><td>1.3290</td><td>1.5409</td><td>1.2885</td><td>1.7120</td><td>1.4880</td></tr></table>

(d) Training and checkpoint selection
<table><tr><td>Budget</td><td>240k</td><td>480k</td><td>720k</td></tr><tr><td>Avg. SR</td><td>0.4923</td><td>0.5765</td><td>0.5673</td></tr><tr><td>Avg. SG</td><td>1.6221</td><td>1.2885</td><td>1.2851</td></tr><tr><td>Val. C</td><td>4-8</td><td>6-8</td><td>7-8</td></tr><tr><td>Avg. SR</td><td>0.4282</td><td>0.5334</td><td>0.5765</td></tr><tr><td>Avg. SG</td><td>2.1324</td><td>1.4900</td><td>1.2885</td></tr></table>

Table S6: Additional development checks under the standard 5 × 5, B = 10, $C = 4 , \ldots , 8$ evaluator. The selected setting uses $c _ { e } = 0 . 0 0 5 , \lambda _ { \mathrm { m i n } } = 0 . 4 0 5 ,$ $\beta = 0 . 1 0$ , and 480k PPO steps.

(a) 8 × 8, B = 24, MASA/A
<table><tr><td>Method</td><td>SR/SG</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td></tr><tr><td>GOMAA-Geo</td><td>0.679/1.345</td><td>0.480</td><td>0.525</td><td>0.700</td><td>0.870</td><td>0.820</td></tr><tr><td>GeoExplorer</td><td>0.368/3.046</td><td>0.385</td><td>0.385</td><td>0.340</td><td>0.415</td><td>0.315</td></tr><tr><td>DynCur-Geo</td><td>0.746/0.859</td><td>0.510</td><td>0.660</td><td>0.790</td><td>0.845</td><td>0.925</td></tr></table>

<table><tr><td colspan="7">(b) 10 × 10,  $B = 3 2 ,$  MASA/A</td></tr><tr><td>Method</td><td>SR/SG</td><td>14</td><td>15</td><td>16</td><td>17</td><td>18</td></tr><tr><td>GOMAA-Geo</td><td>0.629/1.801</td><td>0.535</td><td>0.475</td><td>0.625</td><td>0.750</td><td>0.760</td></tr><tr><td>GeoExplorer</td><td>0.300/4.677</td><td>0.335</td><td>0.385</td><td>0.335</td><td>0.220</td><td>0.225</td></tr><tr><td>DynCur-Geo</td><td>0.748/0.736</td><td>0.540</td><td>0.675</td><td>0.755</td><td>0.885</td><td>0.885</td></tr></table>

Table S7: Per-distance SR for the $8 \times 8$ and $1 0 \times 1 0$ enlargedgrid stress tests. Numeric headers give the initial distance C; the SR/SG column reports the aggregate pair for each grid.

(a) 25 × 25, B = 60, C = 12, . . . , 28
<table><tr><td>Method</td><td>SR/SG</td><td>12</td><td>16</td><td>20</td><td>24</td><td>28</td></tr><tr><td>GOMAA-Geo</td><td>0.180/13.508</td><td>0.080</td><td>0.020</td><td>0.060</td><td>0.060</td><td>0.060</td></tr><tr><td>GeoExplorer</td><td>0.082/15.584</td><td>0.020</td><td>0.060</td><td>0.060</td><td>0.020</td><td>0.040</td></tr><tr><td>DynCur-Geo</td><td>0.204/12.312</td><td>0.020</td><td>0.040</td><td>0.020</td><td>0.060</td><td>0.040</td></tr></table>

<table><tr><td colspan="5">(b) 25 × 25, B = 60, C = 32, . . . , 48</td></tr><tr><td>Method</td><td>32</td><td>36</td><td>40</td><td>44</td><td>48</td></tr><tr><td>GOMAA-Geo</td><td>0.080</td><td>0.220</td><td>0.260</td><td>0.240</td><td>0.720</td></tr><tr><td>GeoExplorer</td><td>0.140</td><td>0.120</td><td>0.120</td><td>0.200</td><td>0.040</td></tr><tr><td>DynCur-Geo</td><td>0.140</td><td>0.220</td><td>0.360</td><td>0.560</td><td>0.580</td></tr></table>

Table S8: Per-distance SR for the 25×25 enlarged-grid stress test. Numeric headers give the initial distance C; panel (a) also lists aggregate SR/SG for each method, and panel (b) continues the larger distance strata.

## Why Failures Persist

Distance gating regulates curiosity during search, but neargoal target confirmation remains the most sensitive stage. More than half of recorded failures enter the target neighborhood and then drift. Across all policies, 55.8–60.7% of routes that first reach distance one subsequently hit the target, and 45.1–48.6% leave that neighborhood at least once. A nonzero curiosity floor preserves adaptability near the goal, but the fixed action budget still requires exact final-cell resolution.

Two behavior patterns account for most remaining errors. Recurrent history reduces but does not eliminate repeatedcell behavior; the 133 revisit/ping-pong failures concentrate in this category. When a route moves away from the target, the intrinsic weight rises again. This can support recovery from an incorrect local hypothesis, but also places greater demand on the visual state representation.

## Evidence Boundaries

The principal trajectory bank uses SwissViewMonuments aerial targets, $C ~ \in ~ \{ 4 , 6 , 8 \}$ , and one fixed task seed. The cross-domain section adds ground-target and disaster routes plus enlarged-grid aggregates with their own budgets and dataset scopes. The success and failure atlases cover behavior categories, while the aggregate tables provide the corresponding frequency summaries.

The taxonomy uses explicit thresholds. Its two dominant geometries are reaching distance one before moving away, and revisiting or immediately reversing. Image-level explanations are treated as qualitative interpretations rather than encoder-level attribution.

## Reproducibility Details

The Swiss route panels are reconstructed from recorded sequences over $1 5 \bar { 0 } 0 \times 1 5 0 0$ aerial images. Route views are uncropped; footers report the target cell and, when applicable, the terminal cell. Cross-domain route panels follow the same convention for xBD, MM-GAG, and Swiss ground-toaerial cases.

Case selection follows deterministic route descriptors: four cases per success category, four per failure category, and matched tasks across contrast figures. The Swiss route set contains 118 displayed route-panel records, and the crossdomain route set adds 16 evaluator panels. Reward figures replay fixed routes with task targets for diagnostic decomposition, population curves use all 2,205 sequences with 2,000 fixed-seed aerial-image-cluster bootstraps, and enlarged-grid results are reported as aggregate and per-distance stress tests.

The experiments use existing datasets rather than introducing a new dataset. MASA, MM-GAG, SwissViewMonuments, and xBD follow the task definitions in the main paper. Fixed task banks specify image id, start cell, target cell, distance stratum, repeat index, grid size, and action budget; route images are rendered from saved trajectories after evaluation.

Route figures expose search geometry, endpoint drift, and same-task reversals. Aggregate tables summarize evaluator outputs across the corresponding route sets.

## Experimental Configuration

Unless otherwise stated, the standard evaluator uses a $5 \times 5$ grid, budget $B \ = \ 1 2 .$ distance strata $C = 4 , \ldots , 8 ,$ five fixed task repeats per distance, greedy decoding, legal-action masking, and task-bank seed 20260516. The xBD pre/post analysis uses the fixed pre/post task bank defined for the main experiments. The enlarged-grid stress tests use the protocols in Fig. $S 1 4 \colon 8 \times 8$ with ${ \overline { { B } } } = 2 4$ and $C = 1 0 , \ldots , 1 \bar { 4 } , 1 0 \times 1 0$ with $B = 3 2$ and $C = 1 4 , \ldots , 1 8$ , and $2 5 \times 2 5$ with $B = 6 0$ and $C = 1 2 , 1 6 , \dots , 4 8$

The final PPO configuration uses actor and critic learning rates of $1 0 ^ { - 4 }$ , discount factor $\gamma = 0 . 9 3$ , four PPO epochs per update, clipping parameter 0.2, and learning-rate decay 0.9999. The state-action transformer is loaded from its pretrained checkpoint and kept frozen during PPO. The reported training seed is 321. The 480k value is the PPO budget, and the reported model row uses the validation-selected checkpoint. The reward follows the main paper, including the distance-dependent intrinsic weight with floor 0.405 and the potential-based shaping term.

Reported GPU experiments were run on a Linux x86\_64 workstation with four NVIDIA GeForce RTX 3080 GPUs, 20 GB memory per GPU. The main software stack was Python 3.11, PyTorch 2.0.1, CUDA 11.7 runtime packages, torchvision 0.15.2, Transformers 4.39.2, NumPy 1.25.2, pandas 2.0.3, SciPy 1.14.1, scikit-learn 1.4.0, and Matplotlib 3.8.2.

Training and evaluation randomness are separated. The reported checkpoint uses the fixed training seed above, while the evaluator uses task-bank seed 20260516 for start/target tuples. Route-atlas case selection is deterministic, and the population bands use a fixed 2,000-replicate aerial-imagecluster bootstrap; these intervals summarize variation within the saved route bank rather than across independently trained seeds.

## Implementation-Level Reproducibility

Each inference episode is an ordered patch sequence initialized with the target embedding and the first observed search patch. After every selected action, the evaluator appends the next search-patch embedding. The frozen transformer receives this sequence together with visited patch indices and previous action tokens. The PPO actor and critic operate on the resulting hidden state, so the action head uses transformer features rather than raw patch tokens alone.

Before action selection, a legal-action mask removes outward moves at grid boundaries and renormalizes the remaining action probabilities. Formal evaluation uses greedy decoding and stops only after reaching the target patch or exhausting the budget. The policy does not receive target distance, reward values, success flags, or diagnostic categories at test time.

Training rewards remain outside the inference input. The extrinsic term rewards exact hits and target-progress transitions while penalizing repeats, non-progress, and invalid stays. The intrinsic term is the mean-squared next-state prediction error between predicted and target hidden representations, multiplied by the distance-dependent weight. The potential term uses the change in normalized distance potential between consecutive states.