# OrthoSkillVLA: Continual Skill Learning via Gradient-Informed Skill Subspace Adaptation

Jiaqi Wang1,2, Zhou Fang1,2, Qiongfeng Shi3, and Yi Zhou1,2\*

1 School of Computer Science and Engineering, Southeast University, China 2 Key Laboratory of New Generation Artificial Intelligence Technology and Its Interdisciplinary Applications, Ministry of Education, China

3 School of Electronic Science and Engineering, Southeast University, China {jiaqi.wang, 220245032, qiongfeng}@seu.edu.cn, yizhou.szcn@gmail.com https://github.com/Jiaqi-Wangx/0rthoSkillVLA

Abstract. Pretrained Vision-Language-Action models provide a strong foundation for robot learning, but sequentially adapting them to diverse skills can perturb the representations and velocity mappings used by previous skills, leading to catastrophic forgetting. Architecture-based approaches improve retention by isolating skills but lead to increased inference footprint. Recent subspace-constrained methods restrict parameter updates in an orthogonal subspace to minimize interference but impose a unified constraint on the entire model. We analyze the distinct roles of internal VLA components and identify two VLA-specific challenges. First, the VLM maintains broad semantic representations, making it vulnerable to capacity exhaustion, whereas the ActionHead refines semantics into localized velocity patterns that are highly sensitive to perturbations. Second, the final velocity decoder serves as a readout layer. Freezing it forms an output-stage expressivity bottleneck, while updating it risks overwriting previous velocity mappings. To this end, we propose OrthoSkillVLA, a parameter-efficient framework for continual skill learning in pretrained VLA models without demonstration replay. Given the representation heterogeneity, we impose separate subspace constraints on the VLM and ActionHead, preserving reusable semantic capacity while protecting localized velocity patterns. For the output layer, we introduce a lightweight feature-aware MoE decoder, where each skill is allocated a compact expert and a training-free router selects the expert according to feature-space affinity. Extensive simulated and real-world evaluations, together with ablations, demonstrate that OrthoSkillVLA better preserves prior skills while acquiring new ones.

Keywords: Vision-Language-Action Models · Continual Skill Learning

## 1 Introduction

Recent Vision-Language-Action (VLA) models are pretrained to predict conditional velocity fields, enabling them to capture diverse and multimodal ac-

\* Corresponding author.

tion distributions [5,15]. While these models exhibit strong multi-task adaptation capabilities [28], they still struggle to continually acquire skills [29]. In fitting significantly varying behavioral patterns, the parameter updates can interfere with established representations of previously learned skills, leading to distorted velocity-field predictions and severe performance degradation. This challenge compromises the scalable deployment of VLA models, motivating continual learning frameworks that support lifelong skill acquisition [27,12].

Existing architecture-based approaches alleviate forgetting by isolating skills into separate computational paths [17,25,27]. However, such isolation often couples retention with skill-dependent architectural growth. As skills accumulate, these added components introduce a growing footprint and auxiliary computation, which can be undesirable for resource-constrained real-time robotic control [3]. Gradient-projection-based methods constrain parameter updates to subspaces approximately orthogonal to previously occupied directions, thereby protecting existing knowledge [14,4]. These subspace-constrained methods maintain a constant architectural footprint after continual skill acquisition, offering a highly practical alternative for real-world deployment of VLA models.

However, directly transplanting subspace-constrained methods to VLA models is insufficient for two VLA-specific reasons. First, existing methods typically impose uniform constraints across the network, failing to account for the distinct roles of three core components of VLA architectures: the Vision-Language Model (VLM), the ActionHead and the final velocity decoder. The VLM encodes reusable semantic representations across skills, whereas the ActionHead operates on more localized velocity patterns, where even small perturbations can corrupt the final action predictions. Empirically, this representational heterogeneity induces an inherent trade-off: a stringent constraint required by the ActionHead rapidly consumes the VLM's finite adaptation capacity, while a relaxed one that preserves the VLM's plasticity fails to protect delicate velocity patterns. Such a mismatch is largely absent in vision or language continual learning [24,22,13], but becomes critical for VLA models where semantic grounding and velocity generation are tightly coupled. Second, the final velocity decoder is not simply another hidden layer but a critical output stage. As the final readout from latent representations to velocity fields, updating the shared decoder can directly overwrite previously learned velocity mappings. Freezing it avoids such disruption but shifts the burden of fitting heterogeneous velocity fields onto the preceding adapters. Since these adapters are themselves confined to orthogonal subspaces for stability, forcing them to compensate for a rigid output layer further intensifies the stability-plasticity tension.

In this work, we propose OrthoSkillVLA, a parameter-efficient continual learning framework for pretrained VLA models. OrthoSkillVLA estimates skill feature subspaces from accumulated gradients and injects new skill knowledge into the orthogonal complement of previously occupied directions. Crucially rather than uniformly constraining all modules, this orthogonal adaptation is applied in a module-aware manner. We assign separate subspace budgets to the VLM and the ActionHead, thereby balancing long-term semantic plasticity with high-precision velocity pattern retention. For the final velocity decoder, which forms an output-stage expressivity bottleneck, we exclude it from orthogonal fine-tuning and introduce a lightweight Feature-Aware MoE Decoder. Each new skill is assigned a compact expert to preserve velocity-field expressivity, while a training-free projection-based router selects experts according to feature-space affinity derived from gradient-informed skill bases. Our contributions are summarized as follows:

— We conduct empirical studies and reveal two specific challenges of pretrained VLA models: module-level representational heterogeneity and output-stage expressivity bottleneck.

We introduce a module-level subspace budgeting strategy, enabling effective skill knowledge retention while preserving semantic plasticity for subsequent skill learning. For the velocity decoder, we design a Feature-Aware MoE Decoder with a training-free projection-based routing mechanism, which maintains velocity decoding expressivity with minimal architectural overhead.

We validate OrthoSkillVLA on simulated and real-world environments, showing consistent improvements over subspace-constrained baselines in priorskill retention and final success rates.

## 2 Related Work

## 2.1 Pretrained Vision-Language-Action Models

Vision-Language-Action (VLA) models pretrained on large-scale datasets [16,9] serve as powerful foundations for end-to-end robotic control [6,31,1]. Modern VLA models have increasingly converged on flow-matching paradigms [10].

By predicting continuous velocity fields, pioneering models like π0.5 [5] and GR00T [15] successfully capture highly complex action distributions, resulting in precise and highly coherent trajectory generation. Driven by these robust generative objectives, recent advancements such as X-VLA [28] and SmolVLA [19] have successfully introduced streamlined architectures and structural miniaturizations. These compact yet capable models significantly alleviate the computational burden on physical platforms, providing ideal testbeds for real-world deployment. However, maintaining this rigid architectural efficiency while sequentially acquiring diverse skills remains a formidable challenge.

## 2.2 Continual Learning for Robotics and VLA Models

Continual learning in robotic manipulation aims to enable embodied agents to sequentially acquire new skills without forgetting previously mastered ones. Prior robotic continual learning methods mainly rely on replay, skill libraries, or hierarchical skill representations [29,21,26], whereas our work focuses on replay-free continual adaptation of pretrained flow-matching VLA models. Recent VLAoriented approaches further mitigate forgetting through adapter expansion [17], expert routing, evolving skill spaces [25], or atomic skill decomposition [27].

![](images/9e9d50019a1b021cbb1ba870618e39d4e935190fb3612aa56c8d53bf17763c38.jpg)  
Fig. 1. Overview of OrthoSkillVLA. For the VLM and ActionHead, OrthoSkillVLA constrains new skill learning to gradient-informed orthogonal subspaces with modulespecific budgets, preserving reusable semantic capacity while protecting localized velocity patterns. For the final velocity decoder, it allocates a compact expert for each skill, and selects experts through a training-free projection router based on featurespace affinity.

While these mechanisms improve skill isolation, they often require explicit routing modules, growing skill-specific components, or additional skill decomposition procedures, which may increase structural complexity and inference overhead as skills accumulate.

Another line of work mitigates forgetting by constraining the optimization trajectory rather than expanding model architecture. Gradient-projection and subspace-constrained methods estimate important feature or gradient subspaces of prior tasks [18] and restrict subsequent updates to low-interference directions [22,8,14,4]. These methods provide a natural foundation for parameterefficient continual adaptation, but their model-level or unified constraints do not directly account for the module-level heterogeneity and the low-dimensional velocity decoding bottleneck of flow-matching VLA policies.

## 3 Preliminary

Continual Skill Learning for Pretrained VLA. We consider a replay-free continual skill learning setting where an embodied agent sequentially masters a repertoire of skills $\mathbf { \bar { \mathcal { S } } } = \{ S ^ { \bar { 1 } } , \ldots , S ^ { K } \}$ without access to historical datasets. At stage $k ,$ the model only accesses demonstrations $D ^ { k }$ of the current skill $S ^ { k }$ , and is expected to acquire $S ^ { k }$ while preserving its performance on prior skills $S ^ { < k }$ During inference, the policy should execute all learned skills without relying on oracle skill identities. As the underlying policy, we consider a pretrained VLA model composed of a Vision-Language Model (VLM) for semantic feature extraction, an ActionHead for predicting latent velocity states, and a final velocity decoder that maps these representations to physical velocity fields [5,28].

First-Order Interference and Gradient-Informed Subspaces. To characterize interference between skills, consider a linear transformation $\boldsymbol { y } = \boldsymbol { x } \mathbf { W } + \boldsymbol { b }$ inside the VLA model, where $\mathbf { W } \in \mathbb { R } ^ { d _ { \mathrm { i n } } \times d _ { \mathrm { o u t } } }$ . After learning a new skill, the weight update ∆W perturbs the output of previous skills by x∆W. A sufficient first-order condition for preserving the response on historical features is $x \varDelta \mathbf { W } \approx 0 , \forall x \in \mathcal { X } ^ { < k }$ indicating that updates for new skills should preferably lie in directions orthogonal to the feature subspace occupied by previous skills [8,14].

For a linear layer, the gradient of the loss with respect to W can be written as $\nabla \mathbf { w } \mathcal { L } = x ^ { \top } \delta .$ where δ is the back-propagated error. Therefore, the accumulated gradients after learning a skill $S ^ { k }$ serve as a reliable proxy for the feature subspace occupied by $S ^ { k }$ [18].

## 4 Method

In this section, we present OrthoSkillVLA, a parameter-efficient continual learning framework for pretrained VLA models. We first introduce the module-level subspace estimation and adaptation strategy to minimize the first-order interference in high-dimensional hidden layers of the VLA model. Then we present the design of the Feature-Aware MoE Decoder and the training-free routing mechanism based on feature affinity to provide sufficient output flexibility in velocity decoding.

## 4.1 Skill Subspace Estimation with Module-Aware Budgets

For each adapted layer, we maintain a compact orthonormal basis set $\mathbf { M } _ { k - 1 } \in$ $\mathbb { R } ^ { d _ { \mathrm { i n } } \times \sum _ { i = 1 } ^ { k - 1 } }$ mi to span the feature subspace occupied by all previously learned skills, where $m _ { i }$ is the number of basis vectors allocated for skill $S ^ { i }$ . Given the accumulated gradient $\mathbf { G } _ { k }$ of the current skill, we project it onto the orthogonal complement of ${ \bf M } _ { k - 1 }$ to filter novel feature directions:

$$
\hat { \mathbf { G } } _ { k } = \mathbf { G } _ { k } - \operatorname { P r o j } _ { \mathbf { M } _ { k - 1 } } ( \mathbf { G } _ { k } ) \quad \mathrm { w h e r e } \quad \operatorname { P r o j } _ { \mathbf { X } } ( Y ) = \mathbf { X } \mathbf { X } ^ { \top } Y ,\tag{1}
$$

followed by Singular Value Decomposition (SVD) on the residual gradient: $\hat { \mathbf { G } } _ { k } =$ $\mathbf { U } _ { k } \pmb { \Sigma } _ { k } \mathbf { V } _ { k } ^ { \top }$ . Retaining all novel feature directions would aggressively consume the finite representational space, leaving restricted plasticity for future skill acquisition and unnecessary storage overhead. To determine an appropriate $m _ { k }$ for the current skill, an energy threshold $\epsilon \in ( 0 , 1 )$ is applied to guarantee that the filtered novel features $\mathbf { U } _ { k , 1 : m _ { k } }$ reconstruct the current skill feature space with sufficient fidelity:

$$
\frac { \| \mathrm { P r o j } _ { \mathbf { U } _ { k , 1 : m _ { k } } } ( \hat { \mathbf { G } } _ { k } ) \| _ { F } ^ { 2 } + \| \mathrm { P r o j } _ { \mathbf { M } _ { k - 1 } } ( \mathbf { G } _ { k } ) \| _ { F } ^ { 2 } } { \| \mathbf { G } _ { k } \| _ { F } ^ { 2 } } \geq \epsilon\tag{2}
$$

where $\| \cdot \| _ { F }$ denotes the Frobenius norm. Then the basis set is updated accumulatively: $\mathbf { M } _ { k } = [ \mathbf { M } _ { k - 1 } , \mathbf { U } _ { k , 1 : m _ { k } } ]$

Acting as a subspace budget, a higher energy threshold leads to stronger retention of previously learned skills but reduced capacity for future skill acquisition. The VLM and ActionHead play distinct roles in flow-matching VLA models. Pretrained on diverse multimodal datasets, the VLM part maintains dense representations that require a relatively broader but reusable semantic subspace, making it vulnerable to capacity exhaustion. The ActionHead, on the other hand, refines these broad semantics into specific velocity representations, rendering its dominant feature directions highly localized and sensitive to perturbations. The representational heterogeneity between them necessitates differentiated capacity requirements during new skill acquisition. We therefore adopt separate module-level budgets: a relatively relaxed threshold $\epsilon _ { \mathrm { V L M } }$ for the VLM to preserve long-term plasticity and a more stringent threshold $\epsilon _ { \mathrm { H e a d } }$ for the ActionHead to ensure precise retention of velocity patterns.

## 4.2 Gradient-Informed Orthogonal Low-Rank Adaptation

At the beginning of acquiring skill $S ^ { k }$ , the historical bases ${ \bf M } _ { k - 1 }$ summarize the feature space occupied by previously learned skills. The optimization for $S ^ { k }$ is then constrained to the orthogonal complement of ${ \bf M } _ { k - 1 }$ in a parameter-efficient manner. For each adapted linear layer, with the parameter update parameterized as $\varDelta \mathbf { W } _ { k } = \mathbf { A B }$ , its column space is bounded by the down-projection matrix A. Therefore, before fine-tuning on $S ^ { k }$ , we compute the initial gradient $\mathbf { G } _ { k } ^ { \prime }$ and project it away from the historical bases:

$$
\hat { \mathbf { G } } _ { k } ^ { \prime } = \mathbf { G } _ { k } ^ { \prime } - \operatorname { P r o j } _ { \mathbf { M } _ { k - 1 } } ( \mathbf { G } _ { k } ^ { \prime } )\tag{3}
$$

To maximize the plasticity for the current skill, we initialize A with the top-r left singular vectors of $\hat { \mathbf { G } } _ { k } ^ { \prime }$ . During fine-tuning for skill $S ^ { k }$ , only the up-projection matrix B is optimized while A remains frozen, ensuring that all updates are confined to the orthogonal subspace. This mechanism allows the VLA model to sequentially integrate diverse velocity patterns while minimizing interference with previously learned skills. After learning $S ^ { k }$ , the low-rank adapters are merged into the base model and then perform subspace estimation as described in Section 4.1.

## 4.3 Feature-Aware MoE Velocity Decoder

In VLA models, the velocity decoder $\mathbf { W } _ { \mathrm { b a s e } } \in \mathbb { R } ^ { d _ { \mathrm { i n } } \times d _ { \mathrm { a c t i o n } } }$ serves as the final readout from latent representations to executable velocity fields. Unlike hidden layers, whose representations can be further transformed by subsequent modules this layer directly determines the generated trajectory precision. Freezing the decoder would avoid disrupted velocity mappings, but create an optimization bottleneck. In this case, the optimization burden of fitting heterogeneous velocity fields will be shifted onto the preceding LoRA adapters. Crucially, these adapters are themselves confined to orthogonal subspaces to ensure stability. Forcing them to compensate for a rigid output layer further exacerbates the tension between stability and plasticity.

To provide the necessary flexibility of learning velocity mappings, we introduce a Feature-Aware Mixture-of-Experts Decoder. For each new skill $S ^ { k }$ , we allocate a lightweight skill-specific expert $\mathbf { W } _ { k }$ , and freeze the pretrained decoder:

$$
v _ { k } = x _ { k } ( \mathbf { W } _ { \mathrm { b a s e } } + \mathbf { W } _ { k } )\tag{4}
$$

where $x _ { k }$ is the latent representation and $v _ { k }$ is the predicted velocity. By freezing $\mathbf { W } _ { \mathrm { b a s e } }$ and exclusively optimizing the skill-specific $\mathbf { W } _ { k }$ , the behavioral patterns of diverse skills are captured in isolated parameter spaces, providing the outputlayer expressivity where it is most needed.

A challenge in such MoE architectures is the routing mechanism during inference. Rather than introducing a complex trainable router that might itself be prone to forgetting, we propose a training-free gradient-informed routing strategy based on feature-space affinity. According to Section 4.1, after learning skill $S ^ { k }$ , the accumulated gradient of $\mathbf { W } _ { k }$ provides a reliable approximation of the space occupied by the velocity representation $x _ { k }$ . We consolidate the left singular vectors of $\bar { \bf G } _ { k } ^ { \mathrm { D e c } }$ with the corresponding expert $\mathbf { W } _ { k }$ inside the model to support training-free routing. As a result, a test representation $x$ belonging to skill $S ^ { k }$ will naturally yield the highest projection magnitude onto its corresponding basis $\mathbf { U } _ { k } ^ { \mathrm { D e c } }$ . Consequently, the expert selection is determined by:

$$
k ^ { * } = \arg \operatorname* { m a x } _ { i = 1 , \dots , K } | | x \mathbf { U } _ { i } ^ { \mathrm { D e c } } | | ^ { 2 }\tag{5}
$$

Since this projection-based routing can be efficiently vectorized, and the decoder parameters constitute a negligible fraction of the total model size (less than 0.01% in our implementation), the MoE Decoder introduces minimal inference latency and memory footprint, ensuring real-time deployability. The training procedure is summarized in Algorithm 1.

## 5 Experiments

In this section, we evaluate the continual skill learning capability of OrthoSkil-IVLA in simulated and real-world environments. We adopt pretrained X-VLA [28] as our base model due to its lightweight architecture (\~ 0.9B) and effective adaptation capabilities proven on diverse simulation benchmarks and real-world applications.

![](images/a75f2475ace5284a06dbcd385c473ac5b77cd1f50032ae916399e95bf88ec74b.jpg)

Algorithm 1 OrthoSkillVLA: Continual Skill Learning Framework   
1: Input: Skills $\mathcal { S } = \{ S ^ { 1 } , \ldots , S ^ { K } \}$ , pretrained VLA model weights $\theta _ { 0 } ,$ adapted layers   
L, module-level energy threshold $\epsilon _ { \mathrm { V L M } }$ and $\epsilon _ { \mathrm { H e a d } } .$ LoRA rank r.   
2: Initialize: Current $\mathrm { V L A }$ model from $\theta _ { 0 } ;$ Historical bases $\{ \mathbf { M } _ { 0 } ^ { \ell } \} _ { \ell \in \mathcal { L } }  \varnothing .$   
3: for each skill $S ^ { k } \in { \mathcal { S } }$ do   
4: Orthogonal Skill Subspace Adaptation   
5: Compute initial gradients $\{ \mathbf { G } _ { k } ^ { \prime , \ell } \} _ { \ell \in \mathcal { L } }$ on dataset $D ^ { k }$ and project it to the or  
thogonal complement of $\{ \mathbf { M } _ { k - 1 } ^ { \ell } \} _ { \ell \in \mathcal { L } } ~ ( \mathrm { E q . ~ 3 } )$   
6: Initialize and freeze down-projection $\{ \mathbf { A } _ { k } ^ { \ell } \} _ { \ell \in \mathcal { L } }$ with the top r singular vectors   
7: Allocate velocity-decoder expert $\mathbf { W } _ { k }$ and freeze $\mathbf { W } _ { \mathrm { b a s e } } \ ( \mathrm { E q . \ 4 } )$   
8: Optimize up-projection $\{ \mathbf { B } _ { k } ^ { \ell } \} _ { \ell \in \mathcal { L } }$ and expert $\mathbf { W } _ { k }$   
9: Merge LoRA updates to obtain $\theta _ { k }$ and keep $\mathbf { W } _ { k }$ isolated   
10: Module-Aware Skill Subspace Estimation   
11: Accumulate gradient $\{ \mathbf { G } _ { k } ^ { \ell } \} _ { \ell \in \mathcal { L } }$ and $\mathbf { G } _ { k } ^ { \mathrm { { D e c } } }$ over dataset $D ^ { k }$   
12: Filter novel features of current skill $( \mathrm { E q . ~ 1 } )$   
13: Update historical bases $\{ \mathbf { M } _ { k } ^ { \ell } \} _ { \ell \in \mathcal { L } }$ using module-level energy thresholds €vLM   
and €Head (Eq. 2)   
14: Bind decoder-level routing basis $\mathbf { U } _ { k } ^ { \mathrm { D e c } }$ with expert $\mathbf { W } _ { k }$  
Fig. 2. Simulation results under skill-incremental continual learning. (a) Average success rate on seen skills. (b) Skill-wise success matrices across learning phases.

## 5.1 Simulated Experiments

Experimental Setup. We perform our simulated evaluation on LIBERO [11], a lifelong robot learning benchmark. While LIBERO offers diverse tasks, its original task-incremental protocols (Spatial, Object) primarily vary object identities or placements while maintaining similar motion patterns. Such settings provide limited action distribution shifts, failing to fully expose the gradient conflicts in sequentially fitting heterogeneous velocity fields.

To this end, we introduce a more challenging skill-incremental evaluation protocol by reorganizing the LIBERO-100 suite into skill-specific task sets that emphasize pronounced discrepancies in action distributions. Specifically, we define three skills, OpenClose, Turn, and PickPlace, each comprising 2—4 tasks to ensure substantial differences in motion patterns, comparable task counts, and the exclusion of compound skills. To mitigate the influence of skill difficulties on our evaluation, we evaluate the average performance across three orderings.

Table 1. Comparison of continual learning metrics across different methods in the simulated skill-incremental setting. The best results are highlighted in bold.
<table><tr><td>Method</td><td>FWT↑</td><td>NBT↓</td><td>AUC↑</td><td>Final SR(%) ↑</td></tr><tr><td>SeqLoRA [20]</td><td> $0 . 9 2 \pm 0 . 0 4$ </td><td> $0 . 8 9 \pm 0 . 0 5$ </td><td> $0 . 5 8 \pm 0 . 0 3$ </td><td> $3 2 . 4 4 \pm 3 . 3 1$ </td></tr><tr><td>IncLoRA [20]</td><td> $0 . 9 0 \pm 0 . 0 3$ </td><td> $0 . 8 3 \pm 0 . 0 6$ </td><td> $0 . 5 8 \pm 0 . 0 1$ </td><td> $3 4 . 1 1 \pm 1 . 7 0$ </td></tr><tr><td>EWC [7]</td><td> $0 . 7 3 \pm 0 . 0 2$ </td><td> $0 . 6 8 \pm 0 . 0 4$ </td><td> $0 . 4 6 \pm 0 . 0 4$ </td><td> $2 5 . 1 7 \pm 4 . 3 8$ </td></tr><tr><td>OLoRA [22]</td><td> $0 . 8 8 \pm 0 . 1 0$ </td><td> $0 . 8 5 \pm 0 . 1 0$ </td><td> $0 . 5 4 \pm 0 . 0 7$ </td><td> $3 0 . 5 0 \pm 4 . 3 4$ </td></tr><tr><td>KeepLoRA [14]</td><td> $0 . 9 0 \pm 0 . 0 5$ </td><td> $0 . 4 6 \pm 0 . 1 6$ </td><td> $0 . 7 2 \pm 0 . 0 2$ </td><td> $5 6 . 6 1 \pm 7 . 2 2$ </td></tr><tr><td>OrthoSkillVLA</td><td> $\mathbf { 0 . 9 4 \ : \pm { \ : 0 . 0 3 } }$ </td><td> $\mathbf { 0 . 1 3 \ : \pm { \ : 0 . 0 6 } }$ </td><td> $\mathbf { 0 . 8 8 \ : \pm { \ : 0 . 0 1 } }$ </td><td> $\mathbf { 8 3 . 5 0 \pm 1 . 4 2 }$ </td></tr></table>

To comprehensively evaluate OrthoSkillVLA, we compare it against representative replay-free continual learning and PEFT baselines: SeqLoRA [20], which sequentially fine-tunes LoRA adapters on each skill; IncLoRA [20], which reinitializes a new adapter for each skill without explicit interference mitigation; EWC [7], which penalizes changes to parameters estimated to be important for prior skills; OLoRA [22], which encourages LoRA updates for different skills to be orthogonal through an auxiliary loss; and KeepLoRA [14], which fine-tunes LoRA adapters in an orthogonal subspace with a shared energy threshold for both the VLM and ActionHead. The final velocity decoder is frozen to avoid corruption of previously learned velocity mappings.

Let $R _ { i , j }$ denote the average success rate on skill Sj after the model completes training on skill $S ^ { i }$ . For a sequence of K skills, to quantitatively assess continual learning capabilities, in addition to Final SR, where the final average success rate is evaluated, we also consider the following metrics:

Forward Transfer (FWT) measures the model's learning proficiency on newly introduced skills, defined as $\begin{array} { r } { \mathrm { F W T } ~ = ~ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } R _ { k , k } } \end{array}$ . Negative Backward Transfer (NBT) quantifies the severity of catastrophic forgetting to previously learned skills, defined as $\begin{array} { r } { \mathrm { N B T } = \frac { 1 } { K - 1 } \sum _ { k = 1 } ^ { K - 1 } } \end{array}$ NBTk, where $\mathrm { N B T } _ { k } =$ $\begin{array} { r } { \frac { 1 } { K - k } \sum _ { i = k + 1 } ^ { K } ( R _ { k , k } - R _ { i , k } ) } \end{array}$ . Area Under the Curve (AUC) assesses the model's performance stability throughout the entire learning process, calculated as AUC = $\textstyle { \frac { 1 } { K } } \sum _ { k = 1 } ^ { K } { \mathrm { A U C } } _ { k }$ , where $\begin{array} { r } { \mathrm { A U } \bar { \mathrm { C } } _ { k } = \frac { 1 } { K - k + 1 } \sum _ { i = k } ^ { K } R _ { i , k } } \end{array}$

Experimental Results. Table 1 reports the main continual learning results averaged over three skill orderings. SeqLoRA and IncLoRA achieve high FWT, but their large NBT and low Final SR indicate severe forgetting after sequential adaptation. EWC mitigates forgetting partially, yet its strong regularization substantially weakens new-skill acquisition, leading to low FWT. OLoRA provides only limited gains, suggesting that soft orthogonality is insufficient for fitting heterogeneous velocity fields. KeepLoRA serves as the strongest subspaceconstrained baseline, substantially improving retention over standard PEFT methods. Nevertheless, its unified constraint remains sub-optimal for VLA architectures. OrthoSkillVLA further reduces NBT from 0.46 to 0.13 and improves Final SR from 56.61% to 83.50%, demonstrating that Module-Aware Subspace Budgeting and Feature-Aware MoE decoder are both critical for continual skill learning.

Table 2. Real-world success rates during continual skill learning phases for 20 trials. We compare the KeepLoRA [14] baseline and OrthoSkillVLA.
<table><tr><td rowspan="3">Training Phase</td><td colspan="4">KeepLoRA [14]</td><td colspan="4">OrthoSkillVLA</td></tr><tr><td>Flip</td><td>Pick</td><td>Push</td><td>Press</td><td>Flip</td><td>Pick</td><td>Push</td><td>Press</td></tr><tr><td>Phase 1: Learn Flip</td><td>16</td><td></td><td></td><td></td><td>19</td><td></td><td></td><td></td></tr><tr><td>Phase 2: Learn Pick</td><td>14</td><td>18</td><td></td><td></td><td>18</td><td>20</td><td></td><td></td></tr><tr><td>Phase 3: Learn Push</td><td>11</td><td>13</td><td>15</td><td></td><td>15</td><td>17</td><td>19</td><td></td></tr><tr><td>Phase 4: Learn Press</td><td>11</td><td>12</td><td>15</td><td>19</td><td>16</td><td>16</td><td>17</td><td>20</td></tr></table>

Implementation Details. In our implementation, we use the AdamW optimizer with a batch size of 16 and a cosine annealing learning rate scheduler with a peak learning rate of $5 \times 1 0 ^ { - 5 }$ after 1000 warmup steps. For each individual skill, the acquisition process typically requires approximately 40,000 steps to reach convergence. We adopt separated thresholds with $\epsilon _ { \mathrm { V L M } } = 0 . 9 9$ and $\epsilon _ { \mathrm { H e a d } } =$ 0.9999 and defer the analysis to Section 6.2.

## 5.2 Real-World Experiments

Experimental Setup. To further validate the effectiveness of OrthoSkillVLA beyond simulated benchmarks, we conduct real-world experiments under a protocol analogous to the simulated setting. We design four manipulation skills—Flip, Pick, Push, and Press—that span distinct action distributions. Flip emphasizes large end-effector orientation changes when reorienting a cup upright, while Pick involves relatively stable pose control for placing an apple onto a plate. Push requires salient end-effector displacement to push a tripod onto a sponge, whereas Press demands precise end-effector control to sequentially press two buttons without closing the fingers.

The real-world experiments are conducted on a 7-DoF xArm equipped with a 6-DoF Inspire dexterous hand and an Orbbec 336 wrist-mounted camera. For each skill, we collect 50 expert demonstrations and the model is trained to predict the velocity fields in the normalized action space, with image observations resized to 224 × 224. During evaluation, each skill is evaluated for 20 independent trials to compute the success rate.

Results and Analysis. Table 2 reports the lower-triangular success matrix during real-world continual skill learning. Compared with KeepLoRA, OrthoSkillVLA maintains more stable performance as new skills are sequentially introduced, while KeepLoRA exhibits more degradation on previously learned skills. After completing all four skills, OrthoSkillVLA achieves an average final success rate of 86.25%, outperforming KeepLoRA by 15.0%. The improvement mainly comes from better retention of earlier skills, while the later skills can still be acquired effectively. These results demonstrate that OrthoSkillVLA better balances stability and plasticity on manipulation tasks with heterogeneous action distributions.

Table 3. Ablation on module-aware budgeting and MoE velocity decoding.
<table><tr><td>Threshold</td><td>Decoder</td><td>FWT↑</td><td>NBT↓</td><td> $\mathrm { A U C \uparrow }$ </td><td>Final SR (%) ↑</td></tr><tr><td>Unified</td><td>Frozen</td><td> $0 . 9 0 \pm 0 . 0 5$ </td><td> $0 . 4 6 \pm 0 . 1 6$ </td><td> $0 . 7 2 \pm 0 . 0 2$ </td><td> $5 6 . 6 1 \pm 7 . 2 2$ </td></tr><tr><td>Separated</td><td>Frozen</td><td> $0 . 9 1 \pm 0 . 0 6$ </td><td> $0 . 4 2 \pm 0 . 0 5$ </td><td> $0 . 7 5 \pm 0 . 0 4$ </td><td> $6 0 . 4 4 \pm 3 . 1 0$ </td></tr><tr><td>Unified</td><td>MoE</td><td> $0 . 9 2 \pm 0 . 0 3$ </td><td> $0 . 3 3 \pm 0 . 1 5$ </td><td> $0 . 7 9 \pm 0 . 0 3$ </td><td> $6 7 . 9 4 \pm 4 . 4 0$ </td></tr><tr><td>Separated</td><td>MoE</td><td> $0 . 9 4 \pm 0 . 0 3$ </td><td> $0 . 1 3 \pm 0 . 0 6$ </td><td> $0 . 8 8 \pm 0 . 0 1$ </td><td> $8 3 . 5 0 \pm 1 . 4 2$ </td></tr></table>

## 6 Ablation Studies

## 6.1 Disentangling Module-Aware Budgeting and MoE Decoding

To disentangle the effects of module-aware subspace budgeting and feature-aware MoE decoding, we compare four variants under the same protocol in our simulated experiments: (i) Unified-Frozen, which applies a single energy threshold to both the VLM and the ActionHead and freezes the velocity decoder, i.e., the KeepLoRA [14] baseline; (ii) Separated-Frozen, which allocates relatively more budget to the ActionHead but still freezes the velocity decoder; (iii) Unified-MoE, which keeps the unified threshold but equips the model with the MoE velocity decoder; and (iv) Separated-MoE, the full OrthoSkillVLA.

As shown in Table 3, when the velocity decoder is frozen, the separated threshold strategy provides slight improvements in NBT and Final SR, which is expected since a more stringent threshold for the ActionHead better preserves previously learned velocity representations. In contrast, introducing the MoE decoder brings larger gains across all metrics, as skill-specific experts provide isolated velocity-mapping flexibility at the critical output stage. Finally, the combination of both designs in OrthoSkillVLA achieves the best performance, reducing NBT from 0.46 to 0.13 and improving Final SR from 56.61% to 83.50%, which validates that module-aware budgeting and feature-aware MoE decoding are both critical for continual skill learning. We further analyze the distinct roles of the VLM and ActionHead in VLA models and the routing mechanisms of the Feature-Aware MoE Decoder in Section 6.2 and Section 6.3, respectively.

## 6.2 Validating Module-Aware Budgeting Strategy

To quantify the representation heterogeneity between the VLM and ActionHead, we measure the Subspace Occupancy Ratio (SOR), defined as the fraction of accumulated basis dimensions over the total input dimension of adapted layers after continual skill acquisition: $\begin{array} { r } { \mathrm { S O R } = \frac { \sum _ { \ell = 1 } ^ { L } \sum _ { k = 1 } ^ { \mathbf { \hat { K } } } m _ { k } ^ { \ell } } { \sum _ { \ell = 1 } ^ { L } d _ { \mathrm { i n } } ^ { \ell } } } \end{array}$

![](images/1e0a490f6155c0e591d73df06167500909773778588b19c8e4dab15f4360552b.jpg)  
Fig. 3. Evolution of SOR under varying unified energy thresholds.

Table 4. Ablation on the separated energy threshold strategy with fixed $\epsilon _ { \mathrm { V L M } } = 0 . 9 9$  
Table 5. Ablation on the separated energy threshold strategy with fixed $\epsilon _ { \mathrm { H e a d } } = 0 . 9 9 9 9$
<table><tr><td>€Head</td><td>Final SR (%)</td><td>Final SOR (%) VLM ActionHead</td></tr><tr><td>0.99</td><td>59.83</td><td>22.57 2.17</td></tr><tr><td>0.999</td><td>80.67</td><td>21.09 7.57</td></tr><tr><td>0.9999</td><td>84.83</td><td>23.07 18.85</td></tr></table>

<table><tr><td>€VLM Final SR (%)</td><td>Final SOR (%) VLM ActionHead</td></tr><tr><td>0.999 85.50</td><td>53.73 22.72</td></tr><tr><td>0.99 84.83</td><td>23.07 18.85</td></tr><tr><td>0.9 66.17</td><td>4.35 18.83</td></tr></table>

As illustrated in Fig. 3, applying a unified ε yields different SOR evolutions between VLM and ActionHead. The VLM consistently occupies a much larger proportion of the subspace compared to the ActionHead. This observed imbalance leads to a dilemma: knowledge retention in the ActionHead requires a stringent threshold that may aggressively exhaust the VLM's finite capacity, necessitating our module-aware budgeting strategy.

To further validate this strategy, we separately vary $\epsilon _ { \mathrm { H e a d } }$ and $\epsilon _ { \mathrm { V L M } }$ while fixing the other, and report both the final success rate and the final SOR to characterize the performance-capacity trade-off between them. With $\epsilon _ { \mathrm { V L M } }$ fixed to 0.99 in Table 4, relaxing $\epsilon _ { \mathrm { H e a d } }$ to 0.99 causes a significant drop in Final SR, indicating that insufficient subspace budget for ActionHead fails to preserve the localized velocity patterns. Further tightening $\epsilon _ { \mathrm { H e a d } }$ to 0.9999 substantially improves final performance to 84.83% while keeping a moderate subspace occupancy of 23.07%. Under this sufficient budget for ActionHead, further allocating capacity to VLM brings a marginal gain of 0.67% in Final SR but dramatically exhausts up to 53.73% of the VLM's representational space, as shown in Table 5.

## 6.3 Analysis of Training-Free Routing Mechanism

Regarding the routing mechanism, Fig. 4(a) presents the routing confusion matrix, compiled across all numerical integration steps during inference. The mechanism achieves highly accurate skill identification, yielding a reliable 98.9% for OpenClose and Turn, and a robust 91.5% for PickPlace. To quantify the impact of routing accuracy on final performance, Fig. 4(b) compares our proposed mechanism against Oracle Routing, where the ground-truth skill identity is explicitly provided.

(a) Routing Confusion Matrix

![](images/5a44f726623f601964cca2fe180b2ff5c1756ac11413f5da573b2da695fefb37.jpg)  
(b) Success Rate Comparison

![](images/e0079adba4513fbb3ac507eea6c2ad7ce5b56e86ba074431437c614c62390f53.jpg)  
Fig. 4. Evaluation of the training-free routing mechanism on routing accuracy (a) and success rate comparison with oracle routing (b) in our simulated benchmark.

The results indicate that our training-free router performs closely to this theoretical upper bound, achieving an average success rate of 83.5% compared to the oracle's 86.2%. The slight performance gap is primarily observed in the PickPlace skill, which aligns with its relatively lower routing accuracy shown in the confusion matrix. Overall, these findings validate the reliability of our feature-aware routing, demonstrating that it robustly distinguishes skills while preserving both inference efficiency and velocity expressivity.

## 7 Conclusion

Continual skill learning in pretrained VLA models requires preserving the representations and velocity mappings used by prior skills while adapting to diverse new skills. We show that this challenge is shaped by two VLA-specific factors, namely the heterogeneous subspace-budget requirements of the VLM and ActionHead, and the output-stage expressivity bottleneck of the final velocity decoder. Guided by these observations, OrthoSkillVLA combines module-aware subspace budgeting with a lightweight feature-aware decoder, and experiments validate that both module-specific protection and decoder flexibility are essential for stable continual skill acquisition.

## 8 Acknowledgements

This work was supported by the National Natural Science Foundation of China (Grant No 62476054) and the Southeast University Interdisciplinary Research Program for Young Scholars (2024FGC1007).

## References

1. Bu, Q., Cai, J., Chen, L., et al.: Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. In: IROS. IEEE (2025)

2. Cadene, R., Aliberts, S., Capuano, F., et al.: Lerobot: An open-source library for end-to-end robot learning. arXiv:2602.22818 (2026)

3. Fang, Z., Wang, J., Zhou, Y., Shi, Q.: Probeflow: Training-free adaptive flow matching for vision-language-action models. arXiv:2603.17850 (2026)

4. Gu, H., Luo, M.L., Zhou, Z.H., et al.: Spectral imbalance causes forgetting in low-rank continual adaptation. arXiv:2602.00722 (2026)

5. Intelligence, P., Black, K., Brown, N., et al.: π0.5: A vision-language-action model with open-world generalization. arXiv:2504.16054 (2025)

6. Kim, M.J., Pertsch, K., Karamcheti, S.a.: Openvla: An open-source visionlanguage-action model. In: CoRL. pp. 2679–2713. PMLR (2025)

7. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al.: Overcoming catastrophic forgetting in neural networks. PNAS 114(13), 3521–3526 (2017)

8. Liang, Y.S., Li, W.J.: Inflora: Interference-free low-rank adaptation for continual learning. In: CVPR. pp. 23638–23647 (2024)

9. Lim, J.J.: Droid: A large-scale in-the-wild robot manipulation dataset. In: RSS. RSS Foundation (2024)

10. Lipman, Y., Chen, R.T., Ben-Hamu, H., et al.: Flow matching for generative modeling. In: ICLR (2023)

11. Liu, B., Zhu, Y., Gao, C., et al.: Libero: Benchmarking knowledge transfer for lifelong robot learning. NeurIPS 36, 44776–44791 (2023)

12. Liu, H., Kim, C., Liu, B., et al.: Pretrained vision-language-action models are surprisingly resistant to forgetting in continual learning. arXiv:2603.03818 (2026)

13. Luo, M.L., Zhou, Z.H., Wei, T., et al.: Lada: Scalable label-specific clip adapter for continual learning. In: ICML. pp. 41604–41619. PMLR (2025)

14. Luo, M.L., Zhou, Z.H., Zhang, Y.L., et al.: KeeploRA: Continual learning with residual gradient adaptation. In: ICLR (2026)

15. NVIDIA, J.B., Castañeda, F., et al.: GR00T N1: An open foundation model for generalist humanoid robots. arXiv:2503.14734 (2025)

16. O'Neill, A., Rehman, A., Maddukuri, A.a.: Open X-Embodiment: Robotic learning datasets and RT-X models. In: ICRA. pp. 6892–6903. IEEE (2024)

17. Römer, R., Zhang, Y., et al.: Clare: Continual learning for vision-language-action models via autonomous adapter routing and expansion. arXiv:2601.09512 (2026)

18. Saha, G., Garg, I., Roy, K.: Gradient projection memory for continual learning. In: ICLR (2021)

19. Shukor, M., Aubakirova, D., Capuano, F., et al.: Smolvla: A vision-language-action model for affordable and efficient robotics. arXiv: 2506.01844 (2025)

20. Wallis, P., Allen-Zhu, Z., Li, Y., et al.: Lora: Low-rank adaptation of large language models. In: International Conference on (2021)

21. Wan, W., Zhu, Y., et al.: Lotus: Continual imitation learning for robot manipulation through unsupervised skill discovery. In: ICRA. pp. 537–544. IEEE (2024)

22. Wang, X., Chen, T., Ge, Q., et al.: Orthogonal subspace learning for language model continual learning. In: EMNLP. pp. 10658–10671 (2023)

23. Wang, X., Han, Z., Liu, Z., et al.: Lifelong language-conditioned robotic manipulation learning. In: AAAI. vol. 40, pp. 18629–18637 (2026)

24. Wang, Z., Zhang, Z., Ebrahimi, S., et al.: Dualprompt: Complementary prompting for rehearsal-free continual learning. In: ECCV. pp. 631–648. Springer (2022)

25. Wu, Y., Wang, G., Yang, Z., et al.: Continually evolving skill knowledge in vision language action model. arXiv:2511.18085 (2025)

26. Xu, J., Nie, X.: Speci: Skill prompts based hierarchical continual imitation learning for robot manipulation. IEEE Trans. Cogn. Dev. Syst. 18(2), 488–503 (2026)

27. Zhang, L., Tang, T., Zhan, Z., et al.: Atomicvla: Unlocking the potential of atomic skill learning in robots (2026)

28. Zheng, J., Li, J., Wang, Z., et al.: X-vla: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. arXiv:2510.10274 (2025)

29. Zheng, Z., Cai, J.F., Wu, X.M., et al.: imanip: Skill-incremental learning for robotic manipulation. In: ICCV. pp. 13890–13900 (2025)

30. Zhou, Y., Barnes, C., Lu, J., et al.: On the continuity of rotation representations in neural networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5745–5753 (2019)

31. Zitkovich, B., Yu, T., Xu, S., et al.: Rt-2: Vision-language-action models transfer web knowledge to robotic control. In: CoRL. pp. 2165–2183. PMLR (2023)

## A Details of Simulated Experiments

## A.1 The Skill-Incremental Setting

This work studies catastrophic forgetting in pretrained VLA models when they continually acquire diverse manipulation skills. To better simulate the distributional shifts during continual skill learning, we reorganize LIBERO [11] benchmark into a skill-incremental setting. We start from LIBERO-100 and filter out compound tasks that involve multiple manipulation primitives, e.g. “open the top drawer of the cabinet and put the bowl in it". The remaining 82 short-horizon tasks are then grouped into OpenClose, PickPlace, and Turn according to their dominant behavioral patterns, with 8, 72, and 2 candidate tasks, respectively To avoid an evaluation dominated by the much larger PickPlace group, we select a compact and more balanced subset while preserving diversity in scenes and interacted objects. The final setting contains 4 OpenClose tasks, 4 PickPlace tasks, and 2 Turn tasks.

Table 6. Task instructions of the three skills. See Figures 6–8 for execution snapshots.
<table><tr><td>Skill</td><td>Task instruction</td></tr><tr><td>OpenClose</td><td>close the bottom drawer of the cabinet open the bottom drawer of the cabinet close the top drawer of the cabinet open the microwave</td></tr><tr><td>PickPlace</td><td>put the black bowl at the back on the plate pick up the alphabet soup and put it in the basket pick up the book and place it in the left compartment of the caddy pick up the white mug and place it to the right of the caddy</td></tr><tr><td>Turn</td><td>turn on the stove turn off the stove</td></tr></table>

To mitigate order-specific bias, as shown in Table 7, we evaluate all methods under three orderings and report the mean and standard deviation in the main paper. During evaluation, each task is executed for 50 independent rollouts, and the success rate of a skill is computed by averaging over all tasks belonging to that skill. A qualitative comparison of model performance under these three orderings is further provided in Fig. 5.

## A.2 More Implementation Details

Starting checkpoint. Following X-VLA [28], which absorbs embodiment heterogeneity with domain-specific soft prompts and action-related input/output projections, we treat LIBERO as a new embodiment domain before continual skill learning. We extend the domain dimension for LIBERO by one and initialize the new entries with the average of existing pretrained domains. Then we warm up only these newly introduced parameters on LIBERO-Goal, LIBERO-Spatial, and LIBERO-Object, with the shared backbone frozen. The warm-up tasks are disjoint from our LIBERO-100 skill-incremental evaluation and serve only to obtain an aligned LIBERO-domain initialization. All compared methods start from the same checkpoint.

Table 7. Skill orders used in the simulated continual learning experiments.
<table><tr><td>Order ID</td><td>Skill orderings</td></tr><tr><td>Ordering 1</td><td> $O p e n C l o s e  P i c k P l a c e  T u r n$ </td></tr><tr><td>Ordering 2</td><td> $O p e n C l o s e  T u r n  P i c k P l a c e$ </td></tr><tr><td>Ordering 3</td><td> $P i c k P l a c e  O p e n C l o s e  T u r n$ </td></tr></table>

![](images/3e8bb91e27094dda5b229ebbe3d6b26a5d2e28336529ada7c0c594554af55d6c.jpg)  
Fig. 5. Qualitative comparison of model performance under the three skill orderings used in the simulated continual learning experiments.

Demonstrations. We use the LIBERO datasets released with the X-VLA [28] pretrained model and convert them into the LeRobot [2] format for training. Each demonstration contains two visual observations from a wrist-view camera and a third-person camera. Following the implementation of X-VLA, the effective action space for LIBERO control is represented as a 10-dimensional vector, consisting of 3 dimensions for end-effector translation, 6 dimensions for the 6D rotation representation [30], and 1 dimension for gripper control. The remaining 10 action channels in the original X-VLA interface are reserved for bimanual embodiments and are not used in our experiments.

Architectural footprint analysis. For the MoE velocity decoder, each newly acquired skill stores a skill-specific decoder expert together with its associated routing basis. Let $d _ { h }$ denote the hidden size, $d _ { a }$ the action dimension, and $n _ { \mathrm { b a s i s } }$ the number of orthonormal vectors retained for routing. The additional per-skill decoder state consists of the expert weights and the routing basis, resulting in $d _ { h } d _ { a } + d _ { h } n _ { \mathrm { b a s i s } }$ stored floating-point values. After learning K skills, the total decoder-specific growth is therefore $O ( K d _ { h } ( d _ { a } + n _ { \mathrm { b a s i s } } ) )$ , leaving the dominant VLM and ActionHead computation unchanged. In our experiments, with the hidden size of 1024 and the number of orthonormal vectors $n _ { \mathrm { b a s i s } }$ bounded by the action dimension $d _ { a } = 2 0$ , the additional per-skill decoder state is approximately 0.0046% of the 0.9B-parameter base model.

Low-rank adaptation target. We apply orthogonal low-rank adaptation only to eligible high-dimensional linear layers in the VLM and ActionHead, where both the input and output dimensions are no smaller than the model hidden size of 1024.

We summarize the hyperparameters in Table 8. The accompanying code repository provides access to both the common starting checkpoint and the LeRobot-formatted datasets.

Table 8. Training configuration and baseline-specific settings used in the simulated experiments.
<table><tr><td>Setting</td><td>Item</td><td>Value</td></tr><tr><td rowspan="9">Common</td><td>Optimizer</td><td>AdamW</td></tr><tr><td>Peak learning rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Warmup steps</td><td>1000</td></tr><tr><td>Scheduler</td><td>cosine annealing</td></tr><tr><td>Batch size</td><td>16</td></tr><tr><td>Precision</td><td>bfloat16</td></tr><tr><td>Epochs per skill</td><td>15</td></tr><tr><td>Approx. steps per skill</td><td>35~40k</td></tr><tr><td>LoRA rank</td><td>64</td></tr><tr><td rowspan="2">EWC [7]</td><td>Fisher estimation samples</td><td>All training samples</td></tr><tr><td>Regularization weight</td><td>λ = 30,000</td></tr><tr><td>O-LoRA [22]</td><td>Orthogonality Loss weight</td><td>λ = 100</td></tr><tr><td>KeepLoRA [14]</td><td>Energy threshold</td><td>€ = 0.999</td></tr></table>

## A.3 The Effect of LoRA Rank

To investigate the influence of the LoRA rank r, we evaluate our framework's continual learning performance across $r \in \{ 3 2 , 6 4 , 9 6 \}$ in Table 9.

Table 9. Ablation on the impact of the LoRA rank.
<table><tr><td>Rank</td><td>FWT↑</td><td>NBT↓</td><td>AUC↑</td><td>Final SR(%) ↑</td></tr><tr><td>32</td><td> $0 . 9 3 \pm 0 . 0 1$ </td><td> $0 . 1 7 \pm 0 . 0 3$ </td><td> $0 . 8 6 \pm 0 . 0 2$ </td><td> $7 9 . 5 0 \pm 3 . 0 6$ </td></tr><tr><td>64</td><td> $0 . 9 4 \pm 0 . 0 3$ </td><td> $0 . 1 3 \pm 0 . 0 6$ </td><td> $0 . 8 8 \pm 0 . 0 1$ </td><td> $8 3 . 5 0 \pm 1 . 4 2$ </td></tr><tr><td>96</td><td> $0 . 9 5 \pm 0 . 0 1$ </td><td> $0 . 2 9 \pm 0 . 0 5$ </td><td> $0 . 8 4 \pm 0 . 0 2$ </td><td> $7 4 . 0 6 \pm 3 . 8 6$ </td></tr></table>

As r increases, FWT shows a mild upward trend, indicating that a larger adaptation subspace provides slightly stronger plasticity for newly introduced skills. This is expected because the down-projection matrix A is initialized from the principal directions of the residual gradient, and a larger rank preserves more candidate update directions for fitting the current skill. However, the benefit saturates quickly, suggesting that most useful adaptation directions are already captured by a moderate rank. In contrast, an excessively large rank substantially degrades retention. When r increases to 96, NBT rises from 0.13 to 0.29 and the final success rate drops from 83.50% to 74.06%. This indicates that including lowenergy tail directions may introduce noisy or weakly constrained updates, which reduces the effectiveness of orthogonal adaptation and interferes with previously learned skills. Overall, r = 64 provides the best trade-off, achieving the highest AUC and final success rate while maintaining lower forgetting than both smaller and larger ranks. We therefore use r = 64 as the default setting in our main experiments.

## B Details of Real-World Experiments

For the real-world experiments, we evaluate our framework on four manipulation skills, including Flip, Pick, Push, and Press. These skills cover distinct manipulation patterns, including large end-effector orientation changes, stable graspand-place behavior, salient end-effector displacement, and precise contact-rich control.

We collect 40 demonstrations for every skill. Different from the end-effector in simulated experiments, the control command for the Inspire dexterous hand is 6-dimensional, resulting in a 15-dimensional action vector. During training, the RGB observations recorded at 480 × 480 resolution are resized to 224 × 224 and the actions are normalized using min-max statistics computed from the collected demonstrations. We provide execution snapshots to qualitatively visualize the four skills in Fig. 9.

![](images/737b4253babd69b9d35dd817062827ac0f68e9271a3039b14d4c6664926058e4.jpg)  
Fig. 6. Visualization of the tasks in skill OpenClose.

![](images/d0b443fba9856f3d015abe1d7c74bbb24ca3c312a5c7060c36bc6854065f03d5.jpg)  
Pick up the white mug and place it to the right of the caddy.

Fig. 7. Visualization of the tasks in skill PickPlace.  
![](images/e48c2c03fb33c95054c4f6fcde7bfdff095f60e8156e1ba56db406d298a851e8.jpg)  
Fig. 8. Visualization of the tasks in skill Turn.

![](images/02129abfcc799dc975b25f8bf239ff8d2b66587e52fce1acc139031b3c190e60.jpg)  
Fig. 9. Visualization of the four real-world skills, with each row showing keyframes from an autonomous rollout of Flip, Pick, Push, and Press.