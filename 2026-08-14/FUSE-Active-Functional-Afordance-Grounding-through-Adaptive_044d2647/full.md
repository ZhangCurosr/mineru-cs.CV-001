# FUSE: Active Functional Afordance Grounding through Adaptive Semantic–Geometric Evidence Acquisition

Zhou Chen<sup>1</sup>, Sathyanarayanan N. Aakur<sup>1</sup>

<sup>1</sup>CSSE Department Auburn University, Auburn, AL, 36849 {zzc0053,san0028}@auburn.edu

## Abstract

Embodied agents must often identify and interact with objects based on their function rather than their identity, requiring them to actively acquire observations that reveal discriminative functional evidence. Existing afordance grounding methods operate from fixed viewpoints and lack mechanisms for deciding where to look when functional cues are occluded or incomplete. We introduce Active Functional Afordance Grounding, a new task in which an agent sequentially explores a scene to identify and spatially ground an object satisfying a functional query. To address this problem, we propose FUSE, an adaptive semantic–geometric evidence acquisition framework that combines explicit uncertainty-driven exploration with a learned amortized planner to eficiently select informative viewpoints. We further introduce a Habitat-based benchmark for evaluating active functional grounding. Experiments show that FUSE achieves the highest observed nonoracle grounding performance while reducing computation by 1.33× relative to fully explicit exploration, and remains efective across multiple afordance knowledge sources.

## Introduction

Functional understanding of objects and their properties is central to embodied tasks such as long-horizon planning and problem solving (Ahn et al. 2022; Singh et al. 2023; Zhu et al. 2023; Zitkovich et al. 2023). A functional concept is not fully captured by recognizing a familiar object (Carion et al. 2025) or predicting a grasp from a fixed view (Mahler et al. 2017; Fang et al. 2020; Sundermeyer et al. 2021). Instead, the same intent must support several coupled capabilities: inferring what properties a suitable object should possess, identifying candidate objects in a partially observed scene, selecting observations that resolve ambiguity, and producing a grounding that supports downstream interaction (Chen, Lin, and Aakur 2025; Chen et al. 2025b; Nguyen et al. 2020; Murali et al. 2020). As illustrated in Fig. 1, a request such as “cut a piece of rope” or “scoop food from a container” may generate several plausible functional hypotheses, while the evidence needed to distinguish among them may be occluded, poorly localized, or unavailable from the initial viewpoint. We term this task active functional afordance grounding: selecting sensing actions to determine and spatially ground which scene object satisfies a functional intent, without a predefined target class or identity. Candidate labels are only query-induced hypotheses, not search targets; the grounded object must be resolved from evidence acquired across viewpoints before physical execution and verification.

![](images/4204060ff42010c50de7d5d40cf8414791481591c57501090b8ab1ff79a2708c.jpg)  
Figure 1: Active functional afordance grounding. Passive grounding predicts from a fixed observation and may commit before suficient evidence is available. Active grounding acquires additional observations to reduce semantic–geometric uncertainty until a functionally suitable object is grounded.

This formulation places diferent requirements on perception than existing active or afordance-centered tasks. Passive afordance grounding predicts task-relevant objects, parts, or interaction regions from the available observation and may therefore commit before suficient evidence has been acquired (Chen, Lin, and Aakur 2025; Chen et al. 2025b; Sawatzky et al. 2019; Li et al. 2022). Active recognition and object search acquire additional observations, but typically seek a predefined category or instance (Lei, Jiang, and Daniilidis 2026; Chaplot et al. 2020; Batra et al. 2020). Taskoriented view planning can improve visibility or grasp confidence, yet generally begins with an already selected target object or manipulation hypothesis (Shi et al. 2025; Tong et al. 2026). In contrast, active functional afordance grounding (illustrated in Fig. 1) requires the complete sensing process: forming candidate hypotheses, identifying which semantic and geometric evidence remains unresolved, and acquiring

![](images/8ca2b025ac3385f2aa1a8bb6e1e1978d12988a3b899b423e2bd3c012962a8e84.jpg)  
Figure 2: Overview of FUSE. A functional query is first converted into candidate object hypotheses and grounded using semantic (SAM3) and geometric (SSNR) evidence. FUSE adapts between fast amortized planning and explicit exploration to acquire informative observations until no further improvement is observed, returning the best reliable functional grounding.

## observations until a suitable target can be reliably grounded.

To address this challenge, we introduce FUSE (Functional Understanding through Semantic–Geometric Exploration), an adaptive evidence-acquisition framework for active functional afordance grounding. FUSE is built on the principle that sensing should reduce the unresolved semantic and geometric evidence needed to ground a functional target, which we refer to asfunctional uncertainty. As illustrated in Fig. 2, FUSE combines two complementary planning modes: (1) an amortized planner that rapidly predicts which sensing action is most likely to reduce functional uncertainty, and (2) an explicit planner that evaluates candidate actions using semantic and geometric predictions from an incrementally refined spatial scene representation. When the amortized predictions are confident, FUSE avoids costly search and representation updates; when candidate actions are ambiguous or the task evidence remains unresolved, it invokes explicit planning. By adaptively switching between fast prediction and deliberate search, FUSE preserves reliable functional grounding while reducing the computational cost of active perception.

The contributions of this paper are four-fold: (i) we introduce, define, and formulate active functional afordance grounding as a new sequential perception task in which a functional query must guide target discovery and spatial grounding under partial observability; (ii) we develop and will release a Habitat-based benchmark and evaluation protocol that operationalizes the task through functional queries, controlled sensing trajectories, standardized stopping and success criteria, sensing budgets, oracle controls, and evaluation tracks spanning static, canonical-view, VLM-active, search-based, amortized, and adaptive methods; (iii) we present FUSE, a confidence-gated adaptive evidence-acquisition framework that alternates between fast amortized planning and explicit semantic–geometric search over an incrementally refined spatial scene representation; and (iv) through controlled evaluations, we show that taskconditioned semantic–geometric evidence acquisition improves active grounding across knowledge sources.

## Related Work

Functional Afordance Grounding and Task-Oriented Manipulation. Recent work has increasingly focused on grounding functional language queries to objects, object parts, and grasp configurations for embodied manipulation. Neuro-symbolic afordance grounding methods such as CRAFT and CRAFT-E (Chen, Lin, and Aakur 2025; Chen et al. 2025b) combine visual perception with structured functional reasoning to identify objects satisfying task-level afordance queries from a single observation. Foundationmodel-based approaches extend this capability using large vision–language models and multimodal reasoning (Yu et al. 2025; Ahn et al. 2022), while task-oriented grasping methods leverage afordance prediction to optimize grasp synthesis or next-best-view selection for manipulation (Zhang et al. 2023; Tong et al. 2026; Lei, Jiang, and Daniilidis 2026; Shi et al. 2025). These approaches have substantially improved functional reasoning from available observations; however, they generally assume that suficient visual evidence has already been observed. In contrast, Active Functional Affordance Grounding explicitly addresses the preceding perception problem: determining what additional observations should be acquired before reliable functional grounding.

Active Recognition, Search, and Next-Best-View Perception. Active perception methods improve recognition, localization, and scene understanding by selecting informative sensing actions. Prior work has explored active object recognition (Paletta and Pinz 2000), object search and navigation (Chaplot et al. 2020), next-best-view planning for reconstruction and inspection (Strong et al. 2024), and active grasping systems that reposition the camera to improve grasp estimation (Lei, Jiang, and Daniilidis 2026; Shi et al. 2025). More recently, spatial world models and 3D Gaussian Splatting have enabled active exploration through explicit scene representations (Kerbl et al. 2024; Chen et al. 2025a). These methods optimize sensing for a known recognition, localization, reconstruction, or manipulation objective, where the target is specified a priori or the objective is generic scene coverage. In contrast, Active Functional Afordance Grounding must first infer candidate functional targets from a language query and then actively acquire observations until suficient evidence has been accumulated for grounding.

![](images/80ce4527948354fcf8f6c7c9949c86b8c3bc49e5c67c81364d8ae0abad959a72.jpg)  
(a) Input Observation

![](images/f6b24be0522411b94cdfc4d3676b9d8ec8bdd3f8616dbd8ef5d28ec55d120bdc.jpg)  
(b) Semantic Evidence

![](images/621915c11796196ebea40964d5b463634be1452b06cec49c4f0020ad87071bc6.jpg)  
(c) Geometric Uncertainty

![](images/a86109bbb893b09314f2c7cfebee8f144d02280cbfe5f1aac5a8886b855bd572.jpg)  
(d) Fused Evidence Map  
Figure 3: Visualization of FUSE’s semantic–geometric evidence acquisition. SAM3 highlights candidate functional objects, SSNR estimates geometric uncertainty, and their fusion produces the evidence map used to select the next viewpoint.

Afordance-Guided Active Grasping and Adaptive Planning. Several works actively reposition the camera to improve grasp synthesis, manipulation success, or afordance estimation (Lei, Jiang, and Daniilidis 2026; Shi et al. 2025; Tong et al. 2026), while recent embodied systems combine learned policies with explicit planning or world models to balance eficiency and accuracy during decision making (Chen et al. 2025a; Assran et al. 2023; Ha and Schmidhuber 2018). These approaches typically assume that the target object has already been identified and focus on improving its perception or selecting an appropriate manipulation strategy. In contrast, FUSE performs adaptive evidence acquisition before manipulation, using a confidence-gated combination of amortized planning and explicit semantic–geometric reasoning to determine which candidate object satisfies the functional query and to continue exploration until grounded.

## Proposed Approach

Problem Formulation. Given a functional query (q), an afordance-generation module first produces a set of candidate object hypotheses $\mathcal { O } _ { q } { = } \{ o _ { 1 } , \ldots , o _ { N } \}$ that may satisfy the requested function. Given an initially partial observation $( x _ { 0 } )$ and a set of feasible sensing actions (A), the goal of active functional afordance grounding is to identify and spatially ground an object $o ^ { * } \in \mathcal { O } _ { q }$ without being given its identity in advance. At each step t, the agent selects an action $a _ { t } \in { \mathcal { A } }$ , acquires a new observation $x _ { t + 1 }$ , and integrates the resulting evidence into its current scene state $B _ { t + 1 }$ . This evidence may capture semantic compatibility, object localization, visibility, and geometric structure. The process terminates when a candidate object $o ^ { * }$ can be grounded with suficient confidence under the task-specific grounding criterion $\mathcal { V } ( o ^ { * } , \mathcal { O } _ { q } , \mathcal { B } _ { T } )$ . We formulate sensing as minimizing unresolved functional uncertainty while accounting for the cost of active perception:

$$
\operatorname* { m i n } _ { T , a _ { 1 : T } , o ^ { * } } \quad \mathcal { U } _ { \mathrm { f u n c } } ( \mathcal { O } _ { q } , \mathcal { B } _ { T } ) + \lambda \sum _ { t = 1 } ^ { T } C ( a _ { t } , \mathcal { B } _ { t } ) ,\tag{1}
$$

subject to $\mathcal { V } ( o ^ { * } , \mathcal { O } _ { q } , B _ { T } ) \ge \tau$ . Here, $\mathcal { U } _ { \mathrm { f u n c } }$ represents the unresolved semantic and geometric evidence required to ground the functional hypotheses induced by $q , C$ captures sensing and scene-representation update costs, and τ defines the required grounding confidence. Unlike next-best-view planning, the objective is not to reconstruct the entire scene, but to acquire evidence for reliable functional grounding.

Functional Uncertainty In our framework, functional uncertainty is represented as a spatial field conditioned on the candidate object hypotheses $\mathcal { O } _ { q }$ . Rather than assigning uncertainty to a predefined target object, the field identifies image regions in which the evidence required for grounding the functionally relevant candidates remains unresolved. This scene-centric formulation is appropriate because the target identity is not known in advance and must emerge through active evidence acquisition. At time t, we decompose the functional uncertainty into semantic-geometric components:

$$
\mathbf { U } _ { t } ^ { \mathrm { f u n c } } ( { \mathcal { O } _ { q } } ) = \mathbf { U } _ { t } ^ { \mathrm { s e m } } ( { \mathcal { O } _ { q } } ) + \mathbf { U } _ { t } ^ { \mathrm { g e o } } .\tag{2}
$$

The semantic component captures where the current observation provides insuficient grounding evidence for the candidate hypotheses. The object hypotheses $\mathcal { O } _ { q }$ are supplied to SAM3 (Carion et al. 2025). For each candidate $o _ { i } \in { \mathcal { O } } _ { q } ,$ SAM3 predicts a mask $m _ { t , i }$ with score $s _ { t , i } .$ . The corresponding score is assigned to all pixels within the mask, and the candidate maps are aggregated to produce a dense semantic evidence map $\mathbf { S } _ { t } ( \mathcal { O } _ { q } )$ . After normalization to [0, 1], semantic uncertainty is estimated as ${ \bf U } _ { t } ^ { \mathrm { s e m } } ( \mathcal { O } _ { q } ) = 1 - { \bf \dot { S } } _ { t } ( \bar { \mathcal { O } } _ { q } )$

The geometric component captures regions that are poorly explained by the current spatial scene representation. Given the observed image $x _ { t }$ and the corresponding rendered image $\hat { x } _ { t } .$ , we compute a dense structural signal-to-noise ratio (SSNR) map from discrepancies between their image gradients. High SSNR indicates strong agreement between the observation and rendering, whereas low SSNR indicates incomplete or inaccurately modeled structure. We invert and normalize this map to obtain $\mathbf { U } _ { t } ^ { \mathrm { g e o } }$ , assigning larger uncertainty to regions with weaker reconstruction agreement. The resulting map jointly localizes where candidate-relevant semantic evidence is weak and where additional geometric observation may be useful. Figure 3 illustrates this for a functional query $" \mathrm { s i p } ^ { , \mathrm { \bullet } }$ where the target object is a cup. A sensing action is considered informative when the consequent observation is expected to reduce this uncertainty while strengthening the evidence needed for reliable functional grounding. Spatial Evidence Representation. We instantiate the scene state $B _ { t }$ as a persistent 3D spatial evidence representation that accumulates observations acquired during exploration. Reliable functional grounding requires evidence to be integrated across viewpoints in a common coordinate frame, rather than evaluated independently or retained only as an observation sequence. This spatial organization enables the agent to relate task-relevant evidence across camera poses, reason about visibility and occlusion, predict observations from candidate viewpoints, and localize where geometric evidence remains incomplete. Its purpose is not complete scene reconstruction; instead, it retains the spatial information needed to support functional grounding and subsequent sensing decisions. We implement this representation using an incrementally refined 3D Gaussian Splatting (3DGS) model (Kerbl et al. 2023). The representation is initialized from a small set of seed observations registered through structure-from-motion.

At time $t ,$ the current scene representation model $( \mathcal { G } _ { t } )$ encodes the scene using a set of 3D Gaussians with position, appearance, scale, orientation, and opacity parameters. Given a camera pose $p _ { t } ,$ , the renderer produces a predicted observation $\hat { x } _ { t } = \mathcal { R } ( \mathcal { G } _ { t } , p _ { t } )$ , where $\mathcal { R }$ denotes diferentiable Gaussian rendering. Comparing the predicted image $\hat { x } _ { t }$ with the acquired observation $x _ { t }$ yields the SSNR-based geometric uncertainty used in the functional uncertainty map. After executing a sensing action and observing $x _ { t + 1 }$ , the new image is registered to the current representation and may be incorporated through incremental 3DGS refinement:

$$
\mathcal { D } _ { t + 1 } = \mathcal { D } _ { t } \cup \{ ( x _ { t + 1 } , p _ { t + 1 } ) \} ; \mathcal { G } _ { t + 1 } = \operatorname { U p d a t e } ( \mathcal { G } _ { t } , \mathcal { D } _ { t + 1 } ) ,\tag{3}
$$

where $\mathcal { D } _ { t + 1 }$ denotes all observations registered thus far. The updated representation improves predictions from nearby viewpoints and reduces uncertainty in newly observed regions. It provides the explicit spatial substrate needed to evaluate candidate viewpoints, compare predicted and observed evidence, and support subsequent sensing decisions.

Functional Evidence Exploration. Explicit evidence exploration uses the semantic and geometric components of functional uncertainty to determine where the agent should observe next. At time t, the SAM3 semantic evidence map $\mathbf { S } _ { t } ( \mathcal { O } _ { q } )$ identifies regions likely to contain the candidate objects associated with the functional query, while the SSNRbased geometric uncertainty map $\mathbf { U } _ { t } ^ { \mathrm { g e o } }$ identifies regions that remain poorly explained by the current spatial evidence representation $\dot { \mathcal { G } } _ { t }$ . Although functional uncertainty is defined using low semantic confidence, action selection uses positive semantic evidence to restrict exploration to candidaterelevant regions, while geometric uncertainty favors regions whose structure remains under-observed. We combine these maps to obtain a functional evidence-acquisition map:

$$
\mathbf { E } _ { t } ( \mathcal { O } _ { q } ) = \mathbf { S } _ { t } ( \mathcal { O } _ { q } ) + 1 - \mathrm { N o r m } \left( \frac { \| \nabla x _ { t } \| ^ { 2 } } { \| \nabla x _ { t } - \nabla \hat { x } _ { t } \| ^ { 2 } + \epsilon } \right)\tag{4}
$$

where the gradient ratio is computed locally at each pixel and normalized spatially to [0, 1] using Norm(·), such that larger values of the inverted term indicate greater geometric uncertainty. Hence, Equation 4 acts as an acquisition surrogate that balances candidate relevance with geometric uncertainty to explore regions that are both semantically promising and geometrically under-observed. For each feasible sensing action $a \ \in \ A _ { t }$ , we define an associated image region Ω(a) that the corresponding camera motion is expected to reveal more clearly. Horizontal actions use the corresponding left or right image half, while tilt actions use the upper or lower half, respectively. We evaluate each action by averaging the evidence-acquisition map over its associated region:

$$
Q _ { t } ( a ; \mathcal { O } _ { q } ) = \frac { 1 } { | \Omega ( a ) | } \sum _ { u \in \Omega ( a ) } \mathbf { E } _ { t } ( \mathcal { O } _ { q } ) ( u ) .\tag{5}
$$

The explicit exploration policy then selects $\begin{array} { r l } { a _ { t } ^ { * } } & { { } = } \end{array}$ arg $\operatorname* { m a x } _ { a \in \mathcal { A } _ { t } } Q _ { t } ( a ; \mathcal { O } _ { q } )$ . This produces task-directed sensing: SAM3 evidence guides the agent toward regions associated with functionally relevant object hypotheses, while SSNR uncertainty encourages views that improve incomplete or poorly modeled scene structure. After executing $a _ { t } ^ { * }$ , the agent acquires a new observation $x _ { t + 1 }$ and registers it to the spatial evidence representation. $\mathcal { G } _ { t }$ is refined using the new view (Eqn. 3), after which the predicted observation, SSNR map, SAM3 evidence map, and evidence-acquisition map are recomputed. Updating the representation at every step is necessary for explicit exploration because the geometric uncertainty estimate must reflect all observations accumulated so far. Otherwise, regions that have already been suficiently observed may remain uncertain and attract the policy.

Evidence accumulation and termination. At each visited viewpoint, we compute a fused target score $r _ { t , i } = s _ { t , i } + d _ { t , i }$ for each candidate, where $s _ { t , i }$ is its SAM3 grounding score and $d _ { t , i }$ is a mask-level depth-proximity score. We track the highest fused score $r _ { t } = \operatorname* { m a x } _ { i } r _ { t , i }$ and its associated candidate and viewpoint. If $r _ { t }$ improves upon the best observed score, the best grounding is updated and the patience counter is reset; otherwise, the counter is incremented. Exploration terminates after $K$ consecutive non-improving observations and returns the best grounding encountered.

Amortized Evidence Planning. Although explicit functional evidence exploration provides reliable sensing decisions, repeatedly refining the 3D spatial evidence representation, rendering candidate viewpoints, and recomputing semantic–geometric uncertainty incur substantial computational cost. To reduce this overhead, we learn an amortized evidence planner that directly predicts the value of candidate sensing actions from the current observation, a candidate object hypothesis $o _ { i } \in \mathcal { O } _ { q } ,$ and the candidate action, without explicitly updating the spatial representation. When multiple candidate hypotheses are available, the hypothesis whose text embedding is most compatible with the current image is selected as the object context for the planner. The planner is trained using sampled scene transitions labeled by the measured change in SAM3 evidence, geometric uncertainty, visibility, localization, and proximity after each action. During inference, it produces values for the feasible sensing actions in a single forward pass, approximating explicit evidence planning for faster sensing decision-making.

FUSE: Adaptive Evidence Acquisition. Explicit evidence exploration provides reliable sensing decisions but requires repeated spatial-representation refinement and uncertainty recomputation. Amortized evidence planning avoids this cost by directly predicting the values of feasible sensing actions, but its predictions may become ambiguous in unfamiliar or poorly observed states. FUSE combines both modes through confidence-gated adaptive evidence acquisition. At each step, the amortized planner predicts values $\hat { Q } _ { t } ( \boldsymbol { a } ; \boldsymbol { o } _ { i } )$ for the feasible actions and converts them into an action distribution whose entropy measures decision ambiguity. At each step, the current observation is registered with COLMAP. When entropy is below $\eta ,$ FUSE selects the highest-valued amortized action and defers refinement. Otherwise, it jointly refines the 3D spatial representation over all registered observations and invokes explicit semantic–geometric exploration using ${ \mathcal { O } } _ { q } .$ This provides a lightweight fast–deliberative strategy in which routine sensing decisions use amortized inference, while uncertain decisions trigger more costly explicit evaluation. Algorithm 1 summarizes the complete procedure. FUSE begins with a sparsely initialized 3DGS representation and tracks the strongest semantic-depth-proximity score across $\mathcal { O } _ { q }$ by balancing amortized and explicit planning until the grounding score fails to improve for K steps or the sensing budget T is exhausted. It then returns the grounding configuration and candidate label associated with the best viewpoint.

<table><tr><td>Method</td><td>Success Rate ↑</td><td>Avg. IoU ↑</td><td></td><td>Avg. Steps ↓ Target Pixel Ratio ↑</td><td>Center Score ↑</td><td>Distance Score ↑</td><td>Final SAM3 Score ↑</td></tr><tr><td colspan="8">Passive and Semi-Static Grounding</td></tr><tr><td>Static</td><td>42.00</td><td>41.25</td><td>0.00</td><td>17.21</td><td>40.85</td><td>24.11</td><td>48.20%</td></tr><tr><td>Canonical View</td><td>32.00</td><td>31.36</td><td>1.00</td><td>22.83</td><td>29.91</td><td>51.13</td><td>28.22%</td></tr><tr><td colspan="8">Active Baselines</td></tr><tr><td>Random Active</td><td>56.00</td><td>55.03</td><td>12.20</td><td>24.83</td><td>42.08</td><td>50.16</td><td>58.17%</td></tr><tr><td>VLM Active†</td><td>65.00</td><td>64.01</td><td>7.65</td><td>40.13</td><td>56.73</td><td>64.51</td><td>58.15%</td></tr><tr><td colspan="8">Proposed Semantic-Geometric Planning Variants</td></tr><tr><td>Explicit Exploration</td><td>70.00</td><td>68.77</td><td>8.30</td><td>36.38</td><td>51.87</td><td>58.29</td><td>64.20%</td></tr><tr><td>Amortized Planner</td><td>66.00</td><td>64.98</td><td>8.50</td><td>31.04</td><td>48.73</td><td>48.17</td><td>62.69%</td></tr><tr><td>FUSE</td><td>72.00</td><td>70.91</td><td>9.32</td><td>34.96</td><td>49.52</td><td>56.97</td><td>66.61%</td></tr><tr><td colspan="8">Oracle-Label Reference</td></tr><tr><td>Oracle-Label FUSE</td><td>77.00</td><td>75.86</td><td>9.98</td><td>39.72</td><td>51.70</td><td>59.91</td><td>72.36%</td></tr></table>

Table 1: Active functional afordance grounding results. Unless otherwise stated, all methods use CRAFT-E as the afordance knowledge source. Best results in bold, and second-best underlined. <sup>†</sup>VLM receives the oracle object label and feasible actions.

Implementation Details. The spatial evidence representation is initialized from four seed RGB observations using COLMAP (Schonberger and Frahm 2016) followed by incremental 3D Gaussian Splatting optimization (Kerbl et al. 2023). Each functional query q is mapped to candidate hypotheses $\mathcal { O } _ { q }$ using neurosymbolic models, large language models, oracle object knowledge, or direct verb conditioning, and SAM3 provides pixel-level semantic masks and scores. For termination, each SAM3 score is summed with a masklevel depth-proximity score obtained by normalizing target depth between 0.3 m and 1.2 m, and exploration stops after K consecutive non-improving fused scores. The amortized planner uses CLIP’s visual backbone followed by a threelayer MLP that combines the image feature with candidateobject and action embeddings to predict action values. The MLP is trained to predict measured post-action evidence gain (sum of normalized post-action changes in semantic evidence, geometric certainty, visibility, localization, and proximity), and action entropy (normalized by log $| \mathcal { A } _ { t } | )$ is computed from a unit-temperature softmax over predicted action values. It is trained on 416 trajectories from two scenes disjoint from evaluation using Adam, learning rate $1 0 ^ { - 3 }$ , batch size 64, and 20 epochs. η was set to 0.8 using a grid search over five randomly sampled development episodes. The stopping patience $K = 5$ and sensing budget $\mathbf { \dot { \boldsymbol { T } } = 3 0 }$ are fixed evaluation-protocol choices applied consistently to all active methods. Additional details are in the supplementary.

## Experimental Evaluation

Benchmark and Evaluation Setup. We construct a Habitat-based (Savva et al. 2019; Szot et al. 2021; Puig et al. 2023) benchmark for active functional afordance grounding using procedurally assembled tabletop scenes. The benchmark evaluates 100 independently randomized scene instances drawn from six held-out scene configurations. In each episode, a target object and 7–14 distractors are randomly sampled and placed in the workspace, and the camera is initialized from a random viewpoint such that at least one valid target is partially occluded. The evaluation therefore varies target identity, distractor composition, object layout, occlusion, and initial viewpoint while testing on scene configurations disjoint from those used to train the amortized planner. For episodes with multiple valid targets, a prediction is considered correct if it satisfies the success criterion for any valid target; viewpoint-quality metrics are computed with respect to the matched target. The active sensing space is defined over a ring graph (with fixed elevation) with 36 discrete viewpoints, together with camera tilt-up and tilt-down actions. 3DGS-based methods are initialized from four fixed seed RGB observations.

Metrics and Evaluation Protocol. Each method returns the best viewpoint encountered during exploration together with its grounding prediction. We evaluate grounding accuracy using Success Rate, defined as the percentage of episodes in which the highest-scoring SAM3 mask at the returned viewpoint achieves an IoU greater than 0.5 with a valid ground-truth target. We also report Average IoU between the predicted and target masks and Average Steps, defined as the number of active sensing actions executed after initialization. To measure viewpoint quality, we report four additional metrics. Target Pixel Ratio is the min–max normalized target-pixel count at the returned viewpoint over all reachable viewpoints. Center Score is based on the normalized distance between the target-mask centroid and the image center, with 1 as perfect centering. Distance Score is the inverted, min– max normalized camera–target distance over the reachable viewpoints (larger values indicate a closer view). Final SAM3 Score, computed by querying with the ground-truth label, is reported as a secondary quality metric.

Algorithm 1: FUSE: Adaptive Evidence Acquisition   
Require: Query $q ,$ action set ${ \mathcal { A } } ,$ entropy threshold η, $\mathrm { p a } -$   
tience K, maximum steps $T$   
Ensure: Best observed grounding, candidate label, and   
viewpoint   
1: Generate candidate object hypotheses $\mathcal { O } _ { q }$ from $q$   
2: Initialize $\mathcal { G } _ { 0 }$ and registered seed-view set $\mathcal { D } _ { 0 }$   
3: $r _ { \mathrm { b e s t } }  - \infty , k _ { \mathrm { f a i l } }  0$   
4: for $t = 0$ to $T - 1$ do   
5: Observe $x _ { t }$ at viewpoint $v _ { t }$   
6: Register $x _ { t }$ and $\mathcal { D } _ { t } { \dot {  } } \mathcal { D } _ { t - 1 } \cup \{ ( x _ { t } , v _ { t } ) \}$   
7: Obtain SAM3 masks and scores $\{ ( m _ { t , i } , s _ { t , i } ) { : } o _ { i } { \in } \mathcal { O } _ { q } \}$   
8: Compute depth scores $d _ { t , i }$ over each mask $m _ { t , i }$   
9: $r _ { t , i } \gets s _ { t , i } + d _ { t , i }$   
10: $r _ { t } \gets \operatorname* { m a x } _ { o _ { i } \in \mathcal { O } _ { q } } r _ { t , i }$ and $o _ { t } \gets \arg \operatorname* { m a x } _ { o _ { i } \in \mathcal { O } _ { q } } r _ { t , i }$   
11: if $r _ { t } > r _ { \mathrm { b e s t } }$ then   
12: $r _ { \mathrm { b e s t } }  r _ { t }$   
13: o<sub>best</sub> $ o _ { t } ,$ v<sub>best</sub> $ v _ { t }$   
14: $k _ { \mathrm { f a i l } }  0$   
15: else   
16: $k _ { \mathrm { f a i l } }  k _ { \mathrm { f a i l } } + 1$   
17: end if   
18: if $k _ { \mathrm { f a i l } } \geq K$ then break   
19: Select object context $\tilde { o } _ { t } \in \mathcal { O } _ { q }$ for amortized planning   
20: Predict values $\hat { Q } _ { t } ( \boldsymbol { a } ; \tilde { \boldsymbol { o } } _ { t } )$ for $a \ \in \ A _ { t }$ and compute   
action entropy $H _ { t }$   
21: if $H _ { t } \leq \eta$ then   
22: $a _ { t } \gets \arg \operatorname* { m a x } _ { a \in \mathcal { A } _ { t } } \hat { Q } _ { t } \big ( a ; \tilde { o } _ { t } \big )$   
23: else   
24: Refine $\mathcal { G } _ { t }$ jointly over $\mathcal { D } _ { t }$   
25: Compute $\bar { \mathbf { E } } _ { t } ( \mathcal { O } _ { q } )$ and explicit scores $Q _ { t } ( a ; \mathcal { O } _ { q } )$   
26: $a _ { t } \gets \arg \operatorname* { m a x } _ { a \in \mathcal { A } _ { t } } Q _ { t } ( a ; \mathcal { O } _ { q } )$   
27: end if   
28: Execute $a _ { t }$   
29: end for   
30: return grounding of $O _ { \mathrm { b e s t } }$ at $v _ { \mathrm { b e s t } }$

Baselines. All core baselines use CRAFT-E (Chen et al. 2025b) to generate candidate object hypotheses. Static performs grounding from the initial observation without camera motion. Canonical View executes one predefined motion toward the closest feasible bird’s-eye viewpoint and grounds from that observation. Random Active follows a random walk over the sensing graph, testing whether movement alone improves grounding. VLM Active provides Gemini (Comanici et al. 2025) with the current observation, oracle target label, and feasible actions, allowing the VLM to control the camera until termination. Explicit Exploration recomputes SAM3 semantic evidence and SSNR geometric uncertainty using an incrementally refined 3DGS representation at every step. Amortized Planner selects actions using only the learned evidence planner. FUSE adaptively switches between amortized planning and explicit exploration according to action entropy. Finally, Oracle-Label FUSE supplies the ground-truth object label, isolating errors due to candidate-hypothesis generation when afordance-hypothesis generation is perfect. We also evaluate with alternative afordance knowledge sources, including Gemini (Comanici et al. 2025), Llama 3.3 70B (Grattafiori et al. 2024), Qwen 3.6 Plus (Qwen Team 2026), GPT-5.4 (OpenAI 2025), Claude Sonnet 4.6 (Anthropic 2026), and CRAFT (Chen, Lin, and Aakur 2025). This analysis measures whether the benefits of adaptive semantic–geometric exploration persist across diferent upstream functional-reasoning mechanisms.

<table><tr><td>Knowledge Source</td><td>Success ↑</td><td>IoU ↑</td><td>Steps ↓</td><td>Target Ratio ↑</td><td>SAM3 Score ↑</td></tr><tr><td>CRAFT</td><td>53.00%</td><td>52.16%</td><td>9.99</td><td>36.78%</td><td>66.79%</td></tr><tr><td>GPT</td><td>62.00%</td><td>61.06%</td><td>9.35</td><td>33.42%</td><td>61.44%</td></tr><tr><td>LLAMA</td><td>62.00%</td><td>61.00%</td><td>9.7</td><td>34.63%</td><td>58.98%</td></tr><tr><td>Qwen</td><td>64.00%</td><td>63.05%</td><td>9.89</td><td>37.44%</td><td>61.61%</td></tr><tr><td>Sonnet</td><td>69.00%</td><td>67.94%</td><td>10.01</td><td>37.78%</td><td>66.76%</td></tr><tr><td>Gemini</td><td>72.00%</td><td>70.86%</td><td>9.67</td><td>35.51%</td><td>69.26%</td></tr><tr><td>CRAFTE</td><td>72.00%</td><td>70.91%</td><td>9.32</td><td>34.96%</td><td>66.61%</td></tr><tr><td>Oracle-Label</td><td>77.00%</td><td>75.86%</td><td>9.98</td><td>39.72%</td><td>72.36%</td></tr></table>

Table 2: Robustness to diferent knowledge sources. The planner, stopping rule, and sensing budget are held fixed.

Active Functional Grounding. Table 1 compares FUSE with passive, active, and oracle-assisted methods on the proposed active functional afordance grounding benchmark. We observe four main trends. First, active perception substantially improves functional grounding under partial observability. Static grounding achieves only 42% success, while moving once to a canonical viewpoint reduces performance to 32%, indicating that no single predefined view consistently reveals the evidence needed for reliable grounding. Random Active improves success to 56% using 12.20 camera actions on average, showing that additional viewpoints are beneficial even without task-directed sensing. Second, task-relevant sensing is considerably more efective than unguided exploration. Despite using more camera actions on average, Random Active remains 14 and 16 percentage points below Explicit Exploration and FUSE in success rate, respectively. VLM Active reaches 65% success using fewer actions and an oracle target label, demonstrating that a foundation model can perform coarse task-directed exploration. However, it still remains below FUSE despite this additional supervision. VLM Active obtains high Target Pixel Ratio and Center Score, but its Final SAM3 Score is nearly identical to Random Active (58.15% vs. 58.17%) and substantially below FUSE (66.61%). This suggests that moving toward and centering the target does not necessarily produce the semantic and geometric evidence required for reliable grounding. Third, the proposed planning variants provide complementary accuracy–computation trade-ofs. Explicit Exploration improves the success rate from 56% to 70% over Random

<table><tr><td>Variant</td><td>Success ↑</td><td>IoU ↑</td><td>Steps ↓</td><td>Target Ratio ↑</td><td>Center ↑</td></tr><tr><td>Random Active</td><td>56.00</td><td>55.03</td><td>12.20</td><td>24.83</td><td>42.08%</td></tr><tr><td>Semantic Only</td><td>53.00</td><td>52.13</td><td>6.98</td><td>29.48</td><td>47.60%</td></tr><tr><td>Geometry Only</td><td>65.00</td><td>63.95</td><td>15.36</td><td>32.48</td><td>45.57%</td></tr><tr><td>Amortized Only</td><td>66.00</td><td>64.98</td><td>8.50</td><td>31.04</td><td>48.73%</td></tr><tr><td>FUSE</td><td>72.00</td><td>70.91</td><td>9.32</td><td>34.96</td><td>49.52%</td></tr></table>

Table 3: Ablation studies on semantic evidence, geometric uncertainty, and adaptive planning. All variants use the same afordance knowledge source and stopping criterion.

Active, while requiring substantially fewer camera actions (8.30 vs. 12.20). This result supports the efectiveness of the proposed semantic– geometric acquisition surrogate. The Amortized Planner reaches 66% success with only 8.50 actions and without repeated spatial-representation updates, showing that the sensing policy can be approximated efficiently. FUSE adaptively combines these modes and obtains the highest observed non-oracle success rate (72%), IoU (70.91%), and Final SAM3 Score (66.61%). Finally, FUSE provides a favorable accuracy–computation trade-of relative to Explicit Exploration. On paired episodes, it preserves comparable grounding performance (72% vs. 70%; exact McNemar p=0.688) while reducing end-to-end runtime by 1.33×. FUSE invokes explicit exploration for only 33.41% of decisions, requiring 2.97 3DGS refinements per episode on average; relative to Amortized Planning, this selective fallback improves success from 66% to 72% and IoU from 64.98% to 70.91%. Full paired statistics and stage-wise latency results are provided in the supplementary.

Robustness to Afordance Knowledge Sources. Table 2 evaluates FUSE while varying only the upstream knowledge source used to generate candidate object hypotheses. FUSE remains efective across symbolic, vision–language, and large language model sources, showing that its active evidence acquisition strategy is not tied to a particular afordance reasoner. CRAFT-E achieves 72% success, matching Gemini for the highest non-oracle success rate, while Claude Sonnet reaches 69%; the Oracle-label variant reaches 77%. CRAFT-E also obtains the strongest non-oracle IoU among the evaluated sources. Paired analysis shows that CRAFT-E significantly outperforms CRAFT and GPT, while its differences from Gemini, Sonnet, and the Oracle-label reference are not statistically distinguishable. Performance varies more substantially for weaker knowledge sources. CRAFT achieves 53% success, while Llama, Qwen, and GPT obtain 62–64%. Because the planner, stopping criterion, and action budget are fixed, these diferences primarily reflect whether the knowledge source proposes suitable functional hypotheses. Active exploration can improve the visibility and grounding of a proposed candidate, but it cannot recover a functionally appropriate target that is absent from or poorly represented in the hypothesis set. The two stages therefore play complementary roles: the knowledge source determines what objects may satisfy the query, whereas FUSE determines where to acquire evidence for grounding them. A high Final SAM3 Score does not necessarily imply correct functional grounding, since a visible and segmentable object may correspond to an incorrect functional hypothesis.

<table><tr><td>Component</td><td>Variant</td><td>Success↑</td><td>IoU↑</td><td>Steps↓</td></tr><tr><td rowspan="2">Semantic Evidence</td><td>CLIP (Patch-level)</td><td>34.00</td><td>32.67</td><td>17.11</td></tr><tr><td>SAM3 (Pixel-level)</td><td>70.00</td><td>68.84</td><td>8.35</td></tr><tr><td rowspan="4">Stopping Criterion</td><td>SAM3 (K = 3)</td><td>64.00</td><td>62.94</td><td>5.34</td></tr><tr><td>SAM3 (K = 5)</td><td>70.00</td><td>68.84</td><td>8.35</td></tr><tr><td>SAM3+Depth Proximity (K = 3)</td><td>69.00</td><td>67.94</td><td>6.30</td></tr><tr><td>SAM3+Depth Proximity  $( K = 5 )$ </td><td>72.00</td><td>70.91</td><td>9.32</td></tr></table>

Table 4: Implementation ablations evaluating semantic evidence granularity and grounding-based stopping criteria.

Ablation Studies. Table 3 evaluates the individual contributions of semantic evidence, geometric uncertainty, and adaptive planning. Semantic evidence alone is insuficient for reliable active grounding: despite using only 6.98 camera actions on average, the SAM3-driven Semantic Only variant achieves 53% success, below Random Active (56%) and substantially below FUSE (72%). Thus, the gains cannot be attributed to SAM3 alone; pixel-level semantic grounding must be combined with geometric exploration and uncertaintyaware planning. In contrast, Geometry Only improves success to 65% by actively revealing previously unseen regions, demonstrating that reducing geometric uncertainty provides a stronger exploration signal than semantic confidence alone. Combining semantic and geometric evidence improves success to 70% under Explicit Exploration, confirming their complementary roles during viewpoint selection. FUSE further increases success to 72% by adaptively invoking explicit reasoning when the amortized planner is uncertain.

Efect of Granularity and Termination Conditions. Table 4 evaluates key implementation choices in FUSE. Within the full semantic–geometric planning pipeline, replacing pixel-level SAM3 grounding with patch-level CLIP evidence reduces success from 70% to 34%. This result reflects the benefit of more precise semantic evidence when combined with geometric exploration; as shown in Table 3, SAM3- based semantic exploration alone reaches only 53% success. Incorporating depth proximity into the grounding-based stopping criterion further improves success by approximately 5 percentage points over using SAM3 alone. The depthproximity term reduces premature termination at distant or weakly resolved views, complementing SAM3 by favoring observations with the target at a more useful distance.

## Conclusion and Future Work

We introduced Active Functional Afordance Grounding, a new embodied perception task that requires sequentially acquiring observations to identify and spatially ground objects satisfying a functional query. To address this challenge, we proposed FUSE, an adaptive semantic–geometric evidence acquisition framework that combines explicit uncertaintydriven exploration with a learned amortized planner. Experiments demonstrate improved grounding success over passive, unguided-active, and VLM-controlled baselines while reducing computation relative to fully explicit exploration. Future work will validate FUSE on real robotic platforms and extend it to dynamic environments, richer manipulation tasks, and long-horizon agents that jointly learn functional reasoning, active perception, and action execution.

Acknowledgement. This work was supported in part by the US NSF grants IIS 2348689, IIS 2348690, IIS 2615771, and the USDA award no. 2023-69014-39716.

## References

Ahn, M.; Brohan, A.; Brown, N.; Chebotar, Y.; Cortes, O.; David, B.; Finn, C.; Fu, C.; Gopalakrishnan, K.; Hausman, K.; et al. 2022. Do As I Can, Not As I Say: Grounding Language in Robotic Afordances. arXiv preprint arXiv:2204.01691.

Anthropic. 2026. Claude 3.6 Sonnet Model. Web Release.

Assran, M.; Duval, Q.; Misra, I.; Bojanowski, P.; Vincent, P.; Rabbat, M.; LeCun, Y.; and Ballas, N. 2023. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 15619–15629.

Batra, D.; Gokaslan, A.; Kembhavi, A.; Maksymets, O.; Mottaghi, R.; Savva, M.; Toshev, A.; and Wijmans, E. 2020. Objectnav revisited: On evaluation of embodied agents navigating to objects. arXiv preprint arXiv:2006.13171.

Carion, N.; Gustafson, L.; Hu, Y.-T.; Debnath, S.; Hu, R.; Suris, D.; Ryali, C.; Alwala, K. V.; Khedr, H.; Huang, A.; et al. 2025. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719.

Chaplot, D. S.; Gandhi, D. P.; Gupta, A.; and Salakhutdinov, R. R. 2020. Object Goal Navigation using Goal-Oriented Semantic Exploration. In Advances in Neural Information Processing Systems, volume 33, 4247–4258. Curran Associates, Inc.

Chen, Z.; Kundu, S.; Baweja, H. S.; and Aakur, S. N. 2025a. EASE: Embodied Active Event Perception via Self-Supervised Energy Minimization. IEEE Robotics and Automation Letters.

Chen, Z.; Lin, J.; and Aakur, S. N. 2025. CRAFT: A Neuro-Symbolic Framework for Visual Functional Afordance Grounding. In 19th International Conference on Neurosymbolic Learning and Reasoning.

Chen, Z.; Lin, J.; Bulgin, C.; and Aakur, S. N. 2025b. CRAFT-E: A Neuro-Symbolic Framework for Embodied Affordance Grounding. arXiv preprint arXiv:2512.04231.

Comanici, G.; Bieber, E.; Schaekermann, M.; Pasupat, I.; Sachdeva, N.; Dhillon, I.; Blistein, M.; Ram, O.; Zhang, D.; Rosen, E.; et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Fang, H.-S.; Wang, C.; Gou, M.; and Lu, C. 2020. GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Grattafiori, A.; Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.; Vaughan, A.; et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Ha, D.; and Schmidhuber, J. 2018. World Models.

Kerbl, B.; Kopanas, G.; Leimkühler, T.; and Drettakis, G. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics, 42(4).

Kerbl, B.; Meuleman, A.; Kopanas, G.; Wimmer, M.; Lanvin, A.; and Drettakis, G. 2024. A hierarchical 3d gaussian representation for real-time rendering of very large datasets. ACM Transactions On Graphics (TOG), 43(4): 1–15.

Lei, B.; Jiang, W.; and Daniilidis, K. 2026. ActiveGrasp: Information-Guided Active Grasping with Calibrated Energy-based Model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 42418–42429.

Li, P.; Tian, B.; Shi, Y.; Chen, X.; Zhao, H.; Zhou, G.; and Zhang, Y.-Q. 2022. TOIST: Task Oriented Instance Segmentation Transformer with Noun-Pronoun Distillation. In Oh, A. H.; Agarwal, A.; Belgrave, D.; and Cho, K., eds., Advances in Neural Information Processing Systems.

Mahler, J.; Liang, J.; Niyaz, S.; Aubry, M.; Laskey, M.; Doan, R.; Liu, X.; Aparicio Ojea, J.; and Goldberg, K. 2017. Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics. In Robotics: Science and Systems XIII. Robotics: Science and Systems Foundation.

Murali, A.; Liu, W.; Marino, K.; Chernova, S.; and Gupta, A. 2020. Same Object, Diferent Grasps: Data and Semantic Knowledge for Task-Oriented Grasping. In Conference on Robot Learning.

Nguyen, T.; Gopalan, N.; Patel, R.; Corsaro, M.; Pavlick, E.; and Tellex, S. 2020. Robot Object Retrieval with Contextual Natural Language Queries. In Proceedings ofRobotics: Science and Systems. Corvalis, Oregon, USA.

OpenAI. 2025. GPT-5 System Card. Technical report, OpenAI.

Paletta, L.; and Pinz, A. 2000. Active object recognition by view integration and reinforcement learning. Robotics and Autonomous Systems, 31(1-2): 71–86.

Puig, X.; Undersander, E.; Szot, A.; Cote, M. D.; Partsey, R.; Yang, J.; Desai, R.; Clegg, A. W.; Hlavac, M.; Min, T.; Gervet, T.; Vondrus, V.; Berges, V.-P.; Turner, J.; Maksymets, O.; Kira, Z.; Kalakrishnan, M.; Malik, J.; Chaplot, D. S.; Jain, U.; Batra, D.; Rai, A.; and Mottaghi, R. 2023. Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots.

Qwen Team. 2026. Qwen3.6-Plus: Towards Real World Agents.

Savva, M.; Kadian, A.; Maksymets, O.; Zhao, Y.; Wijmans, E.; Jain, B.; Straub, J.; Liu, J.; Koltun, V.; Malik, J.; Parikh, D.; and Batra, D. 2019. Habitat: A Platform for Embodied AI Research. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV).

Sawatzky, J.; Souri, Y.; Grund, C.; and Gall, J. 2019. What object should i use?-task driven object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7605–7614.

Schonberger, J. L.; and Frahm, J.-M. 2016. Structure-frommotion revisited. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4104– 4113.

Shi, Y.; Wen, D.; Chen, G.; Welte, E.; Liu, S.; Peng, K.; Stiefelhagen, R.; and Rayyes, R. 2025. VISO-Grasp: visionlanguage informed spatial object-centric 6-DoF active view planning and grasping in clutter and invisibility. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 14931–14938. IEEE.

Singh, I.; Blukis, V.; Mousavian, A.; Goyal, A.; Xu, D.; Tremblay, J.; Fox, D.; Thomason, J.; and Garg, A. 2023. Prog-Prompt: Generating Situated Robot Task Plans using Large Language Models. In 2023 IEEE International Conference on Robotics and Automation (ICRA), 11523–11530.

Strong, M.; Lei, B.; Swann, A.; Jiang, W.; Daniilidis, K.; and III, M. K. 2024. Next Best Sense: Guiding Vision and Touch with FisherRF for 3D Gaussian Splatting. arXiv.

Sundermeyer, M.; Mousavian, A.; Triebel, R.; and Fox, D. 2021. Contact-GraspNet: Eficient 6-DoF Grasp Generation in Cluttered Scenes. In 2021 IEEE International Conference on Robotics and Automation (ICRA), 13438–13444.

Szot, A.; Clegg, A.; Undersander, E.; Wijmans, E.; Zhao, Y.; Turner, J.; Maestre, N.; Mukadam, M.; Chaplot, D.; Maksymets, O.; Gokaslan, A.; Vondrus, V.; Dharur, S.; Meier, F.; Galuba, W.; Chang, A.; Kira, Z.; Koltun, V.; Malik, J.; Savva, M.; and Batra, D. 2021. Habitat 2.0: Training Home Assistants to Rearrange their Habitat. In Advances in Neural Information Processing Systems (NeurIPS).

Tong, Z.; Dong, W.; Zhang, C.; and Zhang, H. 2026. GCNGrasp-VP: Afordance-Guided View Planning for Eficient Task-Oriented Grasping. arXiv preprint arXiv:2606.19091.

Yu, C.; Wang, H.; Shi, Y.; Luo, H.; Yang, S.; Yu, J.; and Wang, J. 2025. Seqaford: Sequential 3d afordance reasoning via multimodal large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 1691–1701.

Zhang, X.; Wang, D.; Han, S.; Li, W.; Zhao, B.; Wang, Z.; Duan, X.; Fang, C.; Li, X.; and He, J. 2023. Afordance-Driven Next-Best-View Planning for Robotic Grasping. In Conference on Robot Learning, 2849–2862. PMLR.

Zhu, Y.; et al. 2023. Vima: General robot manipulation with multimodal prompts. In International Conference on Learning Representations (ICLR).

Zitkovich, B.; Yu, T.; Xu, S.; Xu, P.; Xiao, T.; Xia, F.; Wu, J.; Wohlhart, P.; Welker, S.; Wahid, A.; Vuong, Q.; Vanhoucke, V.; Tran, H.; Soricut, R.; Singh, A.; Singh, J.; Sermanet, P.; Sanketi, P. R.; Salazar, G.; Ryoo, M. S.; Reymann, K.; Rao, K.; Pertsch, K.; Mordatch, I.; Michalewski, H.; Lu, Y.; Levine, S.; Lee, L.; Lee, T.-W. E.; Leal, I.; Kuang, Y.; Kalashnikov, D.; Julian, R.; Joshi, N. J.; Irpan, A.; Ichter, B.; Hsu, J.; Herzog, A.; Hausman, K.; Gopalakrishnan, K.; Fu, C.; Florence, P.; Finn, C.; Dubey, K. A.; Driess, D.; Ding, T.; Choromanski, K. M.; Chen, X.; Chebotar, Y.; Carbajal, J.; Brown, N.; Brohan, A.; Arenas, M. G.; and Han, K. 2023. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. In Proceedings of The 7th Conference on Robot Learning, volume 229 of Proceedings ofMachine Learning Research, 2165–2183. PMLR.

<table><tr><td>Method</td><td>Success ↑</td><td>Wall Time (h) ↓</td><td>Speedup ↑</td></tr><tr><td>Amortized Planner</td><td>66.00</td><td>3.95</td><td>2.57×</td></tr><tr><td>Explicit Exploration</td><td>70.00</td><td>10.17</td><td>1.00×</td></tr><tr><td>FUSE</td><td>72.00</td><td>7.62</td><td>1.33×</td></tr></table>

Table 5: Accuracy–eficiency comparison over 100 CRAFT-E evaluation episodes. Speedup is measured relative to Explicit Exploration under the same sensing budget and stopping criterion.

## Appendix 1

## Computational Speed Up Analysis

Table 5 compares the grounding accuracy and wall-clock cost of the three semantic–geometric planning variants over 100 evaluation episodes. Explicit Exploration requires 10.17 hours because it repeatedly refines the 3DGS representation and recomputes semantic and geometric evidence at each sensing step. The Amortized Planner reduces evaluation time to 3.95 hours, providing a 2.57× speedup, but its success rate decreases from 70% to 66%. This indicates that direct action-value prediction substantially reduces computation, but cannot fully reproduce explicit reasoning in ambiguous or poorly observed states.

FUSE provides a more favorable accuracy–eficiency trade-of. It achieves 72% success while requiring 7.62 hours, reducing wall-clock cost by 25.1% relative to Explicit Exploration and yielding a 1.33× speedup. Importantly, this reduction is achieved while improving success by 5 percentage points over explicit planning and 8 points over amortized planning. FUSE is slower than the fully amortized policy because uncertain decisions trigger 3DGS refinement and explicit semantic–geometric evaluation; however, these selective deliberative updates recover and exceed the accuracy of full explicit exploration without incurring its cost at every step. These results support the central design of FUSE: amortized inference eficiently handles routine sensing decisions, while explicit reasoning is reserved for states in which its additional computational cost provides the greatest benefit.

## Eficiency Analysis

To identify the source of the computational savings, we additionally profiled FUSE over 800 sampled evaluation episodes. This profiling set is larger than the 100-episode paired benchmark and is used only for stage-wise latency and branch-frequency analysis. Calls per episode therefore need not match the camera-action averages in the main evaluation. All GPU timings were measured with device synchronization before and after each timed operation. Calls and latency were first accumulated per episode and then averaged over the profiling set.

Table 6 shows that FUSE routes 66.59% ofonline planning decisions through the amortized branch and invokes explicit exploration for only 33.41%. Consequently, the spatial representation is refined only 2.97 ± 3.41 times per episode on average, rather than at every online decision. This confirms that the entropy gate is actively reducing the frequency of costly spatial reasoning rather than merely adding an amortized predictor on top of an otherwise unchanged explicit pipeline. Amortized action prediction contributes only 0.01% of total latency, with an average cost of 0.0027 seconds per call. The learned planner therefore introduces negligible computational overhead. In contrast, SAM3 grounding is the dominant cost, accounting for 58.94% oftotal runtime, followed by view registration at 14.71%, explicit evidence computation at 13.32%, and 3DGS refinement at 8.16%. Initialization, depth-based stopping, and camera execution account for the remaining 4.87%. The latency breakdown also explains why FUSE can be faster than Explicit Exploration despite occasionally taking more sensing actions. FUSE does not primarily save time by shortening trajectories. Instead, it replaces most explicit planning decisions with a millisecond-scale amortized forward pass and refines the 3DGS representation only when action entropy indicates ambiguity. The resulting reduction in representation-update frequency produces the reported 1.33× end-to-end speedup while retaining performance comparable to fully Explicit Exploration.

## Paired Statistical and Runtime Analysis Paired Statistical Protocol

All planning variants were evaluated on the same 100 episode definitions, enabling paired comparisons at the episode level. For binary grounding success, we report the exact McNemar test, the paired success-rate diference, a 95% percentilebootstrap confidence interval computed from 10,000 paired resamples, and the number of episodes uniquely solved by each method. For continuous metrics, we report the paired mean diference and its 95% paired-bootstrap confidence interval. We additionally use the two-sided Wilcoxon signedrank test and apply Holm correction within each metric across the two comparisons:

FUSE versus Explicit Exploration and FUSE versus Amortized

Planning. The bootstrap intervals quantify efect size uncertainty,

whereas the Wilcoxon tests assess paired distributional differences. We do not interpret an efect as statistically reliable solely from an uncorrected test.

The logged steps\_T values count online sensing actions. The four initialization actions required by 3DGS-based methods are included in the total camera-action counts reported in the main paper. Adding the same four initialization actions to FUSE and Explicit Exploration does not alter their paired action diference.

## Paired Grounding Success

Table 7 shows that FUSE preserves comparable grounding success relative to fully Explicit Exploration. FUSE solves four episodes that Explicit Exploration misses, while Explicit Exploration uniquely solves two, yielding a twopoint paired diference whose confidence interval includes zero (p = 0.688). Thus, the current evaluation does not establish a statistically distinguishable success advantage over fully explicit reasoning. Instead, the primary benefit of FUSE in this comparison is computational: it retains comparable grounding performance while avoiding repeated spatial-representation refinement. Relative to Amortized Planning, FUSE improves success by six percentage points and uniquely solves eight episodes, compared with two episodes uniquely solved by the amortized variant. The exact McNemar test is not significant at the conventional 0.05 level $( p = 0 . 1 0 9 )$ , reflecting the small number of discordant outcomes. Nevertheless, the four-to-one asymmetry in uniquely solved episodes is consistent with the intended role of explicit fallback: recovering dificult cases in which amortized action prediction alone is insuficient.

Table 6: FUSE latency and adaptive-planning statistics over 800 evaluation episodes with diferent knowledge sources.
<table><tr><td>Component</td><td>Calls/Ep.</td><td>Time/Call (s)</td><td>Time/Ep. (s)</td><td>Share (%)</td></tr><tr><td>Initialization</td><td>1.00</td><td>13.2707</td><td>13.2707</td><td>4.56</td></tr><tr><td>SAM3 grounding</td><td>9.74</td><td>17.6225</td><td>171.6019</td><td>58.94</td></tr><tr><td>Depth and stopping</td><td>9.74</td><td>0.0061</td><td>0.0598</td><td>0.02</td></tr><tr><td>Amortized planning</td><td>8.74</td><td>0.0027</td><td>0.0233</td><td>0.01</td></tr><tr><td>View registration</td><td>9.74</td><td>4.3974</td><td>42.8280</td><td>14.71</td></tr><tr><td>3DGS refinement</td><td>2.97</td><td>8.0051</td><td>23.7851</td><td>8.16</td></tr><tr><td>Explicit evidence computation</td><td>9.74</td><td>3.9826</td><td>38.7810</td><td>13.32</td></tr><tr><td>Camera execution</td><td>18.48</td><td>0.0444</td><td>0.8201</td><td>0.28</td></tr><tr><td>Total</td><td></td><td></td><td>291.1699</td><td>100</td></tr><tr><td>Amortized branch rate</td><td colspan="4">66.59%</td></tr><tr><td>Explicit branch rate</td><td colspan="4"> $3 3 . 4 1 \%$ </td></tr><tr><td>3DGS refinements per episode</td><td colspan="4"> $2 . 9 7 \pm 3 . 4 1$ </td></tr></table>

Table 7: Paired grounding-success analysis over the shared 100 evaluation episodes. “FUSE only” and “Other only” count discordant episodes uniquely solved by each method. Confidence intervals are computed from 10,000 paired bootstrap resamples.
<table><tr><td>Comparison</td><td>FUSE</td><td>Other</td><td>∆</td><td>95% CI</td><td>FUSE only</td><td>Other only</td><td>McNemar p</td></tr><tr><td>Explicit Exploration</td><td>72%</td><td>70%</td><td>+2 pp</td><td>[−3, 7] PP</td><td>4</td><td>22</td><td>0.688</td></tr><tr><td>Amortized Planning</td><td>72%</td><td>66%</td><td>+6 pp</td><td>[0, 12] PP</td><td>8</td><td></td><td>0.109</td></tr></table>

## Paired Continuous-Metric Analysis

The continuous metrics reinforce the interpretation of the success results. Relative to Explicit Exploration, FUSE produces similar IoU, target exposure, centering, and geometric quality: all corresponding confidence intervals include zero, and none of the adjusted tests is significant. FUSE does use 1.02 additional online actions on average $( p _ { \mathrm { H o l m } } = 0 . 0 1 2 )$ This does not contradict the measured runtime advantage because the additional decisions are predominantly handled through inexpensive amortized inference rather than costly 3DGS refinement. FUSE therefore trades a small increase in sensing actions for substantially lower representationupdate cost. Relative to Amortized Planning, FUSE improves mean IoU by 0.059, with a paired-bootstrap interval of [0.001, 0.119]. The Holm-adjusted Wilcoxon result is not significant, so we treat this as suggestive rather than conclusive evidence of an IoU improvement. More diagnostic viewpoint metrics show clearer gains. FUSE improves the robust target exposure measure, Target Pixel Ratio<sub>90</sub>, by 0.082 $( p _ { \mathrm { H o l m } } = 0 . 0 3 2 )$ , and improves the geometric score by 0.088 $( p _ { \mathrm { H o l m } } = 0 . 0 0 4 ) .$ . These results indicate that the explicit fallback helps FUSE acquire viewpoints with stronger target exposure and geometric evidence than the amortized planner alone. FUSE uses 0.82 additional online actions relative to Amortized Planning. Although the Wilcoxon test identifies a distributional diference, the paired-bootstrap interval for the mean diference includes zero. The combined evidence therefore suggests that FUSE occasionally explores longer, but uses those additional observations to improve geometric coverage and recover episodes that amortized planning alone fails to ground. The logged best\_nav\_score was constant at −1 for all methods and episodes and was therefore excluded from Table 8 and inferential analysis.

Paired Analysis of Functional Knowledge Sources We evaluate each knowledge source on the same 100 episodes while holding the FUSE planner, grounding model, stopping criterion, and action budget fixed. Table 9 reports paired success diferences relative to CRAFT-E. Confidence intervals are obtained from 10,000 paired bootstrap resamples, and exact McNemar p-values are Holm-adjusted across all knowledge-source comparisons. CRAFT-E significantly improves success over CRAFT (+19 pp, adjusted $\scriptstyle p = 0 . 0 0 2 )$ and GPT (+10 pp, adjusted p=0.038). Its observed improvements over Llama and Qwen do not remain significant after correction, while its performance is statistically indistinguishable from Claude Sonnet, Gemini, and the Oraclelabel reference. In particular, CRAFT-E and Gemini both achieve 72% success, whereas the Oracle reaches 77%. Online sensing actions do not difer significantly across knowledge sources, indicating that the success diferences primarily reflect candidate-hypothesis quality rather than unequal exploration budgets.

## Implementation Details

## Environment and Evaluation Episodes

Experiments are implemented in Habitat-Sim v0.3.3 and Habitat-Lab v0.3.3, using the Habitat-compatible AI2- THOR (iTHOR) scene dataset. The benchmark uses six evaluation scene configurations that are disjoint from the two configurations used to train the amortized planner. We evaluate all methods on the same fixed set of 100 episode definitions. Each definition stores the scene configuration, functional query, valid target objects, distractor objects, object poses, initial camera pose, and random seed. Each episode contains one sampled target and 7–14 sampled distractors. Object layouts, including any additional object placements, are fixed within each scene configuration.

Table 8: Paired continuous-metric comparisons over 100 episodes. Diferences are computed as FUSE minus the comparison method. For Online Actions, positive diferences indicate that FUSE uses more sensing actions; for all other metrics, positive diferences favor FUSE. The final column reports Holm-adjusted two-sided Wilcoxon signed-rank p-values.
<table><tr><td>Comparison</td><td>Metric</td><td>FUSE</td><td>Other</td><td>Paired ∆</td><td>95% CI</td><td>Holm p</td></tr><tr><td rowspan="7">Explicit Exploration</td><td>IoU</td><td>0.709</td><td>0.688</td><td>+0.021</td><td>[-0.026, 0.071]</td><td>0.809</td></tr><tr><td>Online Actions</td><td>9.32</td><td>8.30</td><td>+1.02</td><td>[0.31, 1.79]</td><td>0.012</td></tr><tr><td>Target Pixel Ratio</td><td>0.350</td><td>0.364</td><td>-0.014</td><td>[−0.065, 0.035]</td><td>0.629</td></tr><tr><td>Target Pixel Ratio90</td><td>0.561</td><td>0.547</td><td>+0.014</td><td>[−0.050, 0.078]</td><td>0.407</td></tr><tr><td>Best Target Score</td><td>0.586</td><td>0.564</td><td>+0.022</td><td>[−0.011, 0.056]</td><td>0.487</td></tr><tr><td>Center Score</td><td>0.495</td><td>0.519</td><td>-0.024</td><td>[−0.067, 0.021]</td><td>0.628</td></tr><tr><td>Geometric Score</td><td>0.570</td><td>0.583</td><td>-0.013</td><td>[-0.073, 0.045]</td><td>0.854</td></tr><tr><td rowspan="7">Amortized Planning</td><td>IoU</td><td>0.709</td><td>0.650</td><td>+0.059</td><td>[0.001, 0.119]</td><td>0.809</td></tr><tr><td>Online Actions</td><td>9.32</td><td>8.50</td><td>+0.82</td><td>[−0.34, 1.90]</td><td>0.020</td></tr><tr><td>Target Pixel Ratio</td><td>0.350</td><td>0.310</td><td>+0.039</td><td>[-0.005, 0.086]</td><td>0.076</td></tr><tr><td>Target Pixel Ratio90</td><td>0.561</td><td>0.479</td><td>+0.082</td><td>[0.023, 0.143]</td><td>0.032</td></tr><tr><td>Best Target Score</td><td>0.586</td><td>0.556</td><td>+0.030</td><td>[-0.020, 0.081]</td><td>0.167</td></tr><tr><td>Center Score</td><td>0.495</td><td>0.487</td><td>+0.008</td><td>[-0.041, 0.057]</td><td>0.628</td></tr><tr><td>Geometric Score</td><td>0.570</td><td>0.482</td><td>+0.088</td><td>[0.026, 0.150]</td><td>0.004</td></tr></table>

Table 9: Paired success analysis over the shared 100 episodes. CRAFT-E is the reference. ∆ denotes CRAFT-E minus the comparison source, and unique wins are reported as CRAFT-E/comparison.
<table><tr><td>Source</td><td>Success</td><td>∆</td><td>95% CI</td><td>Unique wins</td><td>Adj. p</td></tr><tr><td>CRAFT</td><td>53%</td><td>+19 pp</td><td>[10,28]</td><td>23/4</td><td>0.002</td></tr><tr><td>GPT</td><td>62%</td><td>+10 pp</td><td>[4,17]</td><td>11/1</td><td>0.038</td></tr><tr><td>Llama</td><td>62%</td><td>+10 pp</td><td>[3, 18]</td><td>13/3</td><td>0.106</td></tr><tr><td>Qwen</td><td>64%</td><td>+8 pp</td><td>[1, 15]</td><td>11/3</td><td>0.229</td></tr><tr><td>Claude Sonnet</td><td>69%</td><td>+3 pp</td><td>[−4,10]</td><td>8/5</td><td>1.000</td></tr><tr><td>Gemini</td><td>72%</td><td>0 pp</td><td> $[ - 6 , 6 ]$ </td><td>5/5</td><td>1.000</td></tr><tr><td>Oracle label</td><td>77%</td><td>−5pp</td><td>[−11,1]</td><td>2/7</td><td>0.539</td></tr></table>

## Camera and Action Space

For each scene configuration, we construct a scene-specific elliptical camera trajectory in the horizontal plane. The ellipse is parameterized by a scene-dependent center and major and minor axes, selected to provide collision-free views of the workspace. Each trajectory is uniformly discretized into 36 camera positions at $\bar { 1 0 ^ { \circ } }$ angular intervals. Camera height is scene-specific and ranges from 1.4 to 1.6 m. At each position, the nominal camera orientation faces the ellipse center in the horizontal plane. The scene-specific trajectory parameters are specified in the accompanying code submission.

Each trajectory position is paired with three discrete downward pitch angles, $\{ 1 5 ^ { \circ } , 3 0 ^ { \circ } , 4 5 ^ { \circ } \}$ , yielding a discrete viewpoint state space. An action is represented by a joint displacement $( \bar { \Delta } p , \Delta r )$ , where $\Delta p \in \{ - 2 , - 1 , 0 , + 1 , + 2 \}$ denotes motion along the elliptical trajectory in units of viewpoint nodes, and $\Delta r \in \{ - 1 , 0 , + 1 \}$ denotes a change in pitch level. The available actions comprise a no-op action, one- and two-node horizontal motions, one-level pitch adjustments, and their horizontal–pitch combinations. Actions that would produce an invalid pitch level are masked.

For explicit exploration, horizontal action components are associated with the corresponding left or right image region, whereas pitch-up and pitch-down components are associated with the upper and lower image regions, respectively. For coupled actions, we combine the evidence associated with their horizontal and vertical components.

## 3D Gaussian Splatting

For each episode, all 3DGS-based methods use the same four initialization observations and camera motions. These observations are excluded from the online budget T, ineligible as returned output viewpoints, and are included in the reported end-to-end runtime. Camera poses and the initial sparse reconstruction are obtained using COLMAP v3.13.0. For initialization, we extract features from the initialization observations, perform exhaustive matching, and run COLMAP’s incremental mapper. During mapping, focal length, principal point, and extra camera parameters are held fixed.

For each subsequent observation, we extract features and apply sequential matching with an overlap of 10. We enable guided matching, set the maximum number of matches to 32,768, use a SIFT ratio threshold of 0.9, and set the maximum SIFT descriptor distance to 0.8. The observation is registered into the existing sparse model, followed by point triangulation and bundle adjustment. The camera intrinsics remain fixed during incremental bundle adjustment; all other COLMAP settings use the defaults of v3.13.0.

We use the oficial implementation of 3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al. 2023), commit 54c035f7834b. Initial optimization uses 1000 iterations. Incremental updates use 400 iterations after each explicit-planning observation. The main 3DGS settings are:

• image resolution: $6 4 0 \times 4 8 0 ;$

• spherical-harmonic degree: 3;

• position learning rate: $1 . 6 \times 1 0 ^ { - 4 } \mathrm { t o } 1 . 6 \times 1 0 ^ { - 6 } ;$

• feature learning rate: $2 . 5 \times 1 0 ^ { - 3 } ,$

• opacity learning rate: $2 . 5 \times 1 0 ^ { - 2 } ;$

• scale learning rate: $5 . 0 \times 1 0 ^ { - 3 } ;$

• rotation learning rate: $1 . 0 \times 1 0 ^ { - 3 } ;$

• densification interval: 100 iterations;

• densification start/end iterations: 500 / 15,000;

• opacity-reset interval: 3,000 iterations.

FUSE does not update the 3DGS representation on amortized-planning steps. When explicit planning is invoked, optimization resumes from the most recently updated representation.

## Semantic and Geometric Evidence

We use SAM3 with the sam3.pt checkpoint to obtain dense semantic evidence from $6 4 0 \times 4 8 0$ RGB observations. Given the candidate object labels for a functional query, SAM3 is applied independently to each label. For candidate label i, let $m _ { t , i , j }$ and $s _ { t , i , j }$ denote the j-th predicted mask and its confidence score at time t, respectively. We construct the semantic evidence map by assigning each mask its confidence score and taking the pixelwise maximum over all candidate labels and predicted masks:

$$
S _ { t } ( u ) = \operatorname* { m a x } _ { i , j } s _ { t , i , j } \mathcal { k } [ u \in m _ { t , i , j } ] .
$$

This preserves the strongest candidate-relevant semantic evidence at each pixel for subsequent viewpoint selection.

To measure geometric uncertainty, we compare the current RGB observation with the corresponding rendering from the current 3DGS representation. Both images are converted to grayscale, and gradient magnitudes are computed using $3 \times 3$ Sobel filters. Let $I _ { t }$ and $R _ { t }$ denote the observed and rendered images, respectively, and let $\mathcal { N } _ { 1 1 } ( u )$ denote an $1 1 \times 1 1$ local window centered at pixel u. We define

$$
\begin{array} { r l } & { \displaystyle \boldsymbol { A } _ { t } ( \boldsymbol { u } ) = \sum _ { \boldsymbol { v } \in \mathcal { N } _ { 1 1 } ( \boldsymbol { u } ) } \| \nabla I _ { t } ( \boldsymbol { v } ) \| _ { 2 } ^ { 2 } . } \\ & { \displaystyle B _ { t } ( \boldsymbol { u } ) = \sum _ { \boldsymbol { v } \in \mathcal { N } _ { 1 1 } ( \boldsymbol { u } ) } \left( \| \nabla I _ { t } ( \boldsymbol { v } ) \| _ { 2 } - \| \nabla R _ { t } ( \boldsymbol { v } ) \| _ { 2 } \right) ^ { 2 } . } \end{array}
$$

The local structural signal-to-noise ratio is then

$$
\mathrm { S S N R } _ { t } ( u ) = 1 0 \log _ { 1 0 } \left( \frac { A _ { t } ( u ) + \epsilon } { B _ { t } ( u ) + \epsilon } \right) , \qquad \epsilon = 1 0 ^ { - 8 } .
$$

We use reflection padding at image boundaries. We clip SSNR values to [−30, 30] dB and linearly normalize them to [0, 1]. Geometric uncertainty is defined as

$$
U _ { t } ^ { \mathrm { g e o } } ( u ) = \alpha \left[ 1 - \frac { \mathrm { c l i p } ( \mathrm { S S N R } _ { t } ( u ) , - 3 0 , 3 0 ) + 3 0 } { 6 0 } \right] ,
$$

where $\alpha = 0 . 8$ in all experiments. If either the observed image or its corresponding rendering is unavailable, we set the SSNR map to zero.

We fuse semantic evidence and geometric uncertainty as

$$
E _ { t } ( u ) = w _ { s } S _ { t } ( u ) + w _ { g } U _ { t } ^ { \mathrm { g e o } } ( u ) ,
$$

where $w _ { s } = w _ { g } = 1$ in all experiments.

## Depth-Proximity Stopping Rule

For each candidate region $m _ { t , i } ,$ we compute the mean of valid positive depth values within the region using the Habitat depth observation. Depth values are converted to meters using a scale factor of 1,000. A candidate region is considered valid only if it contains at least 500 pixels. Let $\bar { z } _ { t , i }$ denote its mean depth. The depth-proximity score is

$$
d _ { t , i } = 1 - \mathrm { c l i p } \left( \frac { \bar { z } _ { t , i } - 0 . 3 } { 1 . 2 - 0 . 3 } , 0 , 1 \right) .
$$

If a candidate region contains no valid depth value, we set $d _ { t , i } = 0$

We combine semantic confidence and depth proximity as

$$
r _ { t , i } = \lambda _ { s } s _ { t , i } + \lambda _ { d } d _ { t , i } ,
$$

where $\lambda _ { s } = 0 . 7$ and $\lambda _ { d } = 0 . 3$ in all experiments. We retain the viewpoint associated with the largest grounding score observed so far. Exploration terminates after $K = 5$ consecutive observations that do not improve the best observed score, or after $T = 3 0$ online actions.

## Amortized Planner Architecture

The planner uses the frozen CLIP ViT-B/16 image and text encoders. The RGB images are resized to $2 2 4 \times 2 2 4$ and processed using the standard CLIP normalization. The image and text encoders each produce 512-dimensional features. At each step, the object context is selected as the candidate label whose text embedding has the highest cosine similarity with the current image embedding. Actions are represented using 32-dimensional learned embeddings.

The image, object, and action features are directly concatenated. They are processed by a three-layer MLP with two hidden layers:

$$
1 0 5 6 \to 5 1 2 \to 5 1 2 \to 1 .
$$

The hidden layers use ReLU activations. CLIP remains frozen during training.

## Planner Training Targets

We collect 416 trajectories from scene configurations disjoint from evaluation, comprising 3479 transitions. Each trajectory is initialized from a randomly sampled viewpoint and follows a random rollout of 5–13 actions sampled uniformly from the discrete action set. We randomly split the resulting transitions into 90% training and 10% validation sets.

For a transition $( x _ { t } , a _ { t } , x _ { t + 1 } )$ , we define the utility target as

$$
y _ { t } = w _ { \mathrm { s e m } } \Delta s _ { t } + w _ { \mathrm { u n c } } \Delta u _ { t } + w _ { \mathrm { i o u } } \Delta q _ { t } + w _ { \mathrm { v i s } } \Delta v _ { t } + w _ { \mathrm { g e o } } \Delta g _ { t } .
$$

Individual terms are defined as

$$
\begin{array} { r l } & { \Delta s _ { t } = s _ { t + 1 } - s _ { t } , \qquad \Delta u _ { t } = u _ { t } - u _ { t + 1 } . } \\ & { \Delta q _ { t } = q _ { t + 1 } - q _ { t } , \qquad \Delta v _ { t } = v _ { t + 1 } - v _ { t } . } \\ & { \qquad \Delta g _ { t } = g _ { t + 1 } - g _ { t } . } \end{array}
$$

Here, $s _ { t }$ denotes the maximum SAM3 prompt confidence, $u _ { t }$ denotes the mean 3DGS uncertainty over the nonzero semantic-evidence region, $q _ { t }$ denotes the IoU between the highest-confidence SAM3 region and the ground-truth target mask, $v _ { t }$ denotes target visibility normalized by the scenespecific maximum target pixel count, and $g _ { t }$ denotes normalized inverse camera-to-target distance. We use $w _ { \mathrm { s e m } } = 1 . 0 $ $w _ { \mathrm { u n c } } = 0 . 5 , \ : w _ { \mathrm { i o u } } = 1 . 0 , \ : w _ { \mathrm { v i s } } = 0 . 2$ , and $w _ { \mathrm { g e o } } ~ = ~ 1 . 0$ Targets are clipped to [−1, 1].

The planner is trained using mean-squared error and Adam for 20 epochs, with a learning rate $1 0 ^ { - 3 }$ and batch size 64. We use no weight decay or gradient clipping. We save the model after the final epoch; validation loss is monitored but is not used for checkpoint selection.

## Confidence Gate

For predicted action values $\hat { Q } _ { t } ( a )$ , action probabilities are computed using a softmax with temperature $\tau = 0 . 1$ . We use normalized entropy:

$$
H _ { t } = - \frac { 1 } { \log | \mathcal { A } _ { t } | } \sum _ { a \in \mathcal { A } _ { t } } p _ { t } ( a ) \log p _ { t } ( a ) .
$$

FUSE uses amortized planning when $H _ { t } \ \leq \ 0 . 8$ and explicit exploration otherwise. The threshold was selected from $\left.  \dot { \ } 0 . 6 , 0 . \dot { 7 } , 0 . 8 , 0 . 9 \right\}$ using 5 validation episodes from scene configurations disjoint from the evaluation set. Across the evaluation set, 66.59% of decisions use the amortized branch and 33.41% use explicit exploration. FUSE performs 2.97 3DGS updates per episode on average.

## Baseline Settings

All methods use the same fixed episode definitions, candidate hypotheses, SAM3 model, stopping rule, and online action budget unless otherwise stated.

Random Active. Actions are sampled uniformly from the feasible action set. We use a base random seed of 2 and set the episode seed to $2 + e$ for episode $e ,$ which controls random action sampling and environment initialization. Results are reported over the same fixed set of evaluation episodes used for all methods.

Canonical View. Grounding is performed from a fixed topdown view located above and oriented toward the scene center.

VLM Active. We use Gemini-3.1-Flash-Lite with temperature 0 for active viewpoint selection. At each step, the model receives the current RGB observation and an aligned depth visualization, the candidate object labels associated with the target verb, the current discrete loop position and pitch, the feasible actions, and the six most recent action-state records. The camera moves on a discrete elliptical loop around the scene, with horizontal actions {stay, move\_left\_10, move\_right\_10, move\_left\_20, move\_right\_20} and pitch actions {pitch\_keep, pitch\_up, pitch\_down}. The available downward pitch angles are $1 5 ^ { \circ } , 3 0 ^ { \circ }$ , and $4 5 ^ { \circ }$

The model returns a schema-constrained JSON decision specifying a stop/continue status, one horizontal action, one pitch action, and predicted target visibility, centering, and reachability. It is instructed to stop only when a candidate object is visible, centered, and suficiently close for manipulation; otherwise, it continues searching or refining the view. An episode terminates when the model outputs stop while predicting that the target is visible, or after 30 decision steps. Grounding is then performed on the final RGB observation. Invalid or failed model responses trigger a conservative nomotion continue action. API latency is excluded from runtime.

Oracle-Label FUSE. Only the generated candidate set is replaced by the ground-truth object label. No oracle masks, poses, actions, or stopping signals are used.

## Metric Computation

For each target object in each scene, we precompute statistics over the candidate camera viewpoints, including the maximum visible target-pixel count $P _ { \mathrm { m a x } }$ and the minimum and maximum camera-to-target distances, $D _ { \mathrm { m i n } }$ and $D _ { \mathrm { m a x } } .$ These statistics are used only for metric normalization.

At the returned viewpoint, i.e., the highest-scoring viewpoint encountered during exploration, SAM3 is prompted with the candidate object set associated with the target verb. We select its highest-scoring predicted instance and compute IoU between its mask and the ground-truth semantic mask of the target object. Success is defined as $\mathrm { I o U } \geq 0 . 5 ;$ if SAM3 returns no instance mask, IoU is set to zero.

Target Pixel Ratio and Distance Score are computed as

$$
\mathrm { P i x e l R a t i o } = \mathrm { c l i p } \left( \frac { P - P _ { \mathrm { m i n } } } { P _ { \mathrm { m a x } } - P _ { \mathrm { m i n } } + \epsilon } , 0 , 1 \right) ,
$$

$$
\mathrm { D i s t a n c e S c o r e } = \mathrm { c l i p } \left( \frac { D _ { \mathrm { m a x } } - D } { D _ { \mathrm { m a x } } - D _ { \mathrm { m i n } } + \epsilon } , 0 , 1 \right) .
$$

where $P$ is the target-mask pixel count and D is the Euclidean distance between the returned camera and target positions. Center Score is one minus the target-mask centroid’s distance to the image center, normalized by the center-to-corner distance.

Finally, Final SAM3 Score is the confidence of the highestscoring SAM3 instance queried with the ground-truth object label. It is a secondary view-quality metric; Success and IoU use ground-truth semantic masks.

## Runtime Protocol

Runtime is measured per episode and includes initializationview acquisition, COLMAP initialization and incremental view registration, initial and incremental 3D Gaussian Splatting (3DGS) optimization, SAM3 inference, evidence computation, action selection, camera-pose execution, and finalview evaluation. For methods without 3D reconstruction, runtime includes their corresponding view-selection and grounding operations.

Runtime excludes environment construction and scene loading, model loading, and external hosted-API latency. Episodes are executed sequentially on a single worker, and we report end-to-end runtime over the same 100 evaluation episodes.

## Hardware and Software

Experiments are conducted on a workstation with:

• CPU: AMD Ryzen Threadripper PRO 5995WX (64 cores);

• system memory: 503 GB;

• GPU: four NVIDIA RTX A5500 GPUs, each with 24 GB memory;

• operating system: Ubuntu 24.04.4 LTS;

• NVIDIA driver: 580.173.02;

• simulation stack: Habitat-Sim 0.3.3 and Habitat-Lab 0.3.3 (Python 3.9.23, PyTorch 2.5.1+cu121, CUDA 12.1);

• grounding stack: facebook/sam3 (Python 3.12.13,PyTorch 2.10.0+cu128, CUDA 12.8);

• reconstruction stack: COLMAP 3.13.0 (Python 3.8.20,PyTorch 2.1.2+cu118, CUDA 11.8).

The amortized planner requires approximately 0.10 hours (6 minutes) to train and uses 0.46 GB peak GPU memory.