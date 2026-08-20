# Decision-Metric Alignment in Latent World Models: Diagnostics and Action-Conditioned Objectives for MPC Planning

Jiawei Wang<sup>1∗†</sup>, Yushen Zuo<sup>1∗</sup>, Ke Rui<sup>1∗</sup>, Yichun Feng<sup>2</sup>, Minglei Li<sup>1†</sup>

<sup>1</sup>Simple AI, Beijing, China

<sup>2</sup>University of Chinese Academy of Sciences, Beijing, China

## Abstract

JEPA-style latent world models can use Euclidean distance to a goal latent as the cost for model-predictive control (MPC). Strong decoding of task variables, however, does not guarantee that this particular cost ranks candidate action sequences by real task progress. We call the latter property decision-metric alignment. We introduce Plan-Real Spearman, which measures latent–real rank agreement on random plans, and CEM-stage Spearman, which measures the same agreement as cross-entropy-method (CEM) search concentrates its proposal. We analyze suficient conditions under which latent distance preserves real-cost rankings, identifying encoder distortion, terminal rollout error, and candidate margins as the controlling quantities. Guided by the observed empirical alignment gap, DA-LeWM augments LeWM with inverse-dynamics and demonstration-conditioned goal-action heads. Across all our experiments, DA-LeWM accelerates convergence and achieves higher online success than LeWM, while probe scores remain similar. These results show that action-conditioned objectives improve the geometry used by Euclidean-cost, CEM-based latent MPC.

## 1 Introduction

Model predictive control on a learned latent world model is a principled recipe for robotic manipulation: roll out the world model along candidate action sequences, select the lowestcost sequence, execute, and replan (Maes et al. 2026; Ha and Schmidhuber 2018; Hafner et al. 2019, 2020; Hansen, Su, and Wang 2024). When the world model operates in a JEPA-style learned latent space (LeCun 2022; Assran et al. 2023), the planning cost is typically the squared Euclidean distance between the predicted latent trajectory and an encoded goal observation, dispensing with the need for an explicit reward function.

This recipe exhibits a striking pattern: small changes to the training objective induce large and often counterintuitive changes in online planning success, while linear probes for state, action, reward, and value remain nearly unchanged across the same variants. The dominant question asked of a latent world model, “what does the latent encode?”, is therefore incomplete for planning.

The missing property is geometric rather than informational. The planner’s cost depends on Euclidean distances between latents, so for the ranking of candidate plans to track real-world outcomes, the latent’s metric structure must be compatible with the structure of action consequences in the environment. We formalize this as the distinction between information suficiency (whether task-relevant quantities can be decoded from the latent) and decision-metric alignment (whether the latent cost ranks candidate plans consistently with their environmental outcomes). The two are independent: as Figure 1 illustrates, a latent can be informationsuficient yet geometrically misaligned.

To make decision-metric alignment testable, we introduce two diagnostics. Plan-Real Spearman measures the rank correlation between latent costs and real costs over random candidate plans. CEM-stage Spearman extends it to the converged elite candidates the agent actually executes. We analyze suficient conditions under which latent distance preserves real-cost rankings (Section 4). The analysis identifies encoder distortion, terminal rollout error, and candidate margins as the controlling quantities, motivating stage-wise evaluation as CEM concentrates its proposal. We treat SIGReg measurements as empirical collapse diagnostics, not as certificates of the analytical encoder constants. Guided by the observed empirical alignment gap, we then study action-conditioned auxiliary supervision through DA-LeWM (Decision-Aligned LeWM), which augments LeWM with two lightweight auxiliary heads, an inverse-dynamics head and a goal-conditioned action head. Across all our experiments, DA-LeWM accelerates convergence and achieves higher online success than LeWM, while linear-probe scores remain similar.

Our key contributions are as follows:

• We distinguish information suficiency from decisionmetric alignment and introduce Plan-Real Spearman and CEM-stage Spearman to measure the latter.

• We analyze suficient conditions for rank preservation, identifying encoder distortion, terminal rollout error, and candidate margins as the controlling quantities.

• Guided by the observed alignment gap, we introduce DA-LeWM with action-conditioned auxiliary supervision. It improves latent–real rank agreement in our geometry diagnostics and, across all four environments, accelerates convergence and achieves higher online success than LeWM despite similar linear-probe scores.

![](images/ba1aeeccc3d833d3901603437a62fde2114f81e96f1ecd7717725dc0894712c6.jpg)  
Figure 1: Information suficiency does not imply decision-metric alignment. In each panel, the left schematic places candidate plans around the goal latent $z _ { g } ,$ while the right plot compares their real costs (horizontal) with their latent Euclidean costs (vertical). In (a), the two costs induce the same ordering. In (b), candidate 1 is truly low-cost but lies far from $z _ { g } ,$ , whereas candidate 3 has higher real cost but lies close. Latent distance therefore reverses their preference even though candidate identit and rank remain decodable.

## 2 Background and Problem Setup

## JEPA Latent World Models

We build on LeWM (Maes et al. 2026), a JEPA-style world model for robotic manipulation. An encoder $f _ { \theta }$ maps an observation $o _ { t }$ to a latent embedding $z _ { t } = f _ { \theta } ( o _ { t } )$ via a Vision Transformer backbone followed by a linear projector. An autoregressive predictor $g _ { \phi }$ estimates future latents conditioned on actions:

$$
\hat { z } _ { t + 1 } = g _ { \phi } ( z _ { t - H : t } , \ a _ { t - H : t } ) .\tag{1}
$$

Training minimizes a prediction loss together with Sketched Isotropic Gaussian Regularization (SIGReg), introduced by LeJEPA (Balestriero and LeCun 2025) and adopted by LeWM:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { W M } } = \mathcal { L } _ { \mathrm { p r e d } } + \lambda _ { \mathrm { s i g } } \mathcal { L } _ { \mathrm { S I G R e g } } . } \end{array}\tag{2}
$$

SIGReg projects embeddings onto random unit directions and minimizes the one-dimensional Epps–Pulley statistic against a standard Gaussian, encouraging the aggregate embedding distribution to match an isotropic Gaussian and discouraging trivial collapse. This is a distribution-level target: it does not bound encoder-Jacobian singular values or imply local metric isotropy, injectivity, or bi-Lipschitz behavior.

## MPC with Latent Goal Distance

At test time, MPC is realized by the cross-entropy method (CEM) (Rubinstein 1999). Given an initial observation $o _ { 0 }$ and a goal observation $o _ { g }$ , the planner solves

$$
a ^ { * } = \arg \operatorname* { m i n } _ { a _ { 0 : H } } \| \widehat { z } _ { H } - z _ { g } \| ^ { 2 } , \quad z _ { g } = f _ { \theta } ( o _ { g } ) ,\tag{3}
$$

where $\hat { z } _ { H }$ is obtained by rolling out $g _ { \phi }$ under candidate action sequences. CEM iteratively refits its proposal distribution to the top-E elite candidates with lowest latent cost.

Goal acquisition at evaluation time. For evaluation, $o _ { g }$ is drawn from a held-out demonstration trajectory at a fixed ofset ∆ ahead of the initial observation. This guarantees that the goal latent lies in the data distribution and is reachable by some action sequence. In deployment, the goal image would typically come from a single human demonstration or sub-goal generation. Our analysis is independent of the goal-acquisition mechanism.

## Auxiliary Objectives

A standard avenue for improving latent representations is to add auxiliary prediction heads jointly trained with the world model (Schwarzer et al. 2021; Laskin, Srinivas, and Abbeel 2020; Gelada et al. 2019). We consider inversedynamics, goal-conditioned action, reward-proxy, and valueproxy heads (full losses are in Supplementary Section A). Such objectives are usually judged by probes or downstream control. We instead ask how they afect the planner’s fixed Euclidean cost.

## 3 Information Suficiency vs. Decision-Metric Alignment

## Definitions

A latent space $z = f _ { \theta } ( o )$ is ϵ-information-suficient for task quantities {q<sub>k</sub>} if probe-specific regressors $\hat { q } _ { k } = h _ { k } ( x _ { k } )$ attain held-out error at most ϵ. We test state, inverse-action, reward, value, and goal-action with linear probes. Supplementary Section A specifies each input $x _ { k }$

A latent space is decision-metric aligned with respect to a planning cost $c _ { \mathrm { l a t } } \ \mathrm { i f } ,$ for any pair of candidate action sequences $\mathbf { a } ^ { ( i ) } , \mathbf { a } ^ { ( j ) }$ with corresponding real-environment costs ${ c } _ { \mathrm { r e a l } } ^ { ( i ) } , { c } _ { \mathrm { r e a l } } ^ { ( j ) }$

$$
c _ { \mathrm { l a t } } ( \mathbf { a } ^ { ( i ) } ) < c _ { \mathrm { l a t } } ( \mathbf { a } ^ { ( j ) } ) \quad \iff \quad c _ { \mathrm { r e a l } } ( \mathbf { a } ^ { ( i ) } ) < c _ { \mathrm { r e a l } } ( \mathbf { a } ^ { ( j ) } ) .\tag{4}
$$

Decision-metric alignment is an ordinal property, not a numerical one. It requires the latent cost to correctly rank candidates rather than to be numerically equal to any reward function. The two properties are logically independent. As Figure 1 illustrates, a latent can encode all task-relevant information while arranging it in a geometry where Euclidean distance ranks candidates poorly.

## Plan-Real and CEM-stage Spearman

For each (start, goal) pair sampled from a held-out dataset with ofset $\Delta ,$ , we draw $N = 6 \bar { 4 }$ candidate action sequences from the CEM proposal distribution, score each candidate by the world model’s latent cost $c _ { \mathrm { l a t } } ^ { ( i ) } = \Vert \hat { z } _ { H } ^ { ( i ) } - z _ { g } \Vert ^ { 2 }$ and by environment-rollout cost $c _ { \mathrm { r e a l } } ^ { ( i ) } = d _ { \mathrm { t a s k } } ( s _ { H } ^ { ( i ) } , s _ { g } )$ , and compute the Spearman rank correlation between the two across the N candidates. On PushT, $d _ { \mathrm { t a s k } } = \| s _ { H } - s _ { g } \| _ { 2 }$ over all seven simulator coordinates (agent and block pose, and agent velocity), rather than thresholded online success. Plan-Real Spearman is the mean over $n = 3 0$ pairs. Spearman correlation lies in $[ - 1 , 1 ] \colon + 1$ denotes identical rankings, 0 no monotone association, and −1 reversed rankings. Exact ties receive their average rank. If either cost vector is constant, the pair-level correlation is undefined. We exclude that pair from the mean and record the number of defined pairs. Ranking is the relevant quantity because CEM updates its proposal from relative candidate order rather than calibrated costs.

CEM does not act on random plans: it iteratively concentrates candidates toward low-cost regions, and a latent cost that ranks random plans well may still fail when candidates are near-optimal. CEM-stage Spearman measures the same correlation at three CEM stages: random (iteration 0), mid (iteration 15), and elite (the final top-E set). Random-stage agreement tests global coarse ordering. Mid-stage agreement tests the ordering that shapes proposal updates. Elite-stage agreement tests local discrimination among near-optimal candidates.

## 4 DA-LeWM: A Decision-Aligned Latent World Model

## Method

We propose DA-LeWM, which augments the base LeWM training with two lightweight auxiliary heads. The full training objective is

$$
\mathcal { L } _ { \mathrm { D A - L e W M } } = \mathcal { L } _ { \mathrm { W M } } + \alpha \mathcal { L } _ { \mathrm { i n v } } + \beta \mathcal { L } _ { \mathrm { g o a l } } ,\tag{5}
$$

where $\mathcal { L } _ { \mathrm { i n v } } = \| h _ { \mathrm { i n v } } ( z _ { t } , z _ { t + 1 } ) - a _ { t } \| ^ { 2 }$ is the inverse-dynamics loss and $\mathcal { L } _ { \mathrm { g o a l } } = \| h _ { \mathrm { g o a l } } ( z _ { t } , z _ { g } ) - a _ { t } \| ^ { 2 }$ is the goal-conditioned action loss. Both heads are small MLPs with two hidden layers of width 256. We set $\alpha = 0 . 1$ and $\beta = 0 . 1$ as the cross-environment default. Sensitivity to $\beta$ is reported in Supplementary Section C.

Goal-action supervision. Training uses four-frame demonstration clips at raw ofsets 0, 5, 10, 15. For $k \in \{ 0 , 1 , 2 \} , h _ { \mathrm { g o a l } } ( z _ { t + 5 k } , z _ { t + 1 5 } )$ is supervised by the concatenated, normalized five-raw-action block $a _ { t + 5 k : t + 5 k + 4 } .$ The goal is 15, 10, or 5 raw steps ahead. The target reflects demonstrated progress, not an optimizer-computed globally optimal action. With multimodal demonstrations, squared-error training may average incompatible actions. The head shapes the representation and is not used as a policy.

Rationale. The inverse-dynamics loss encourages the latent transition to retain information predictive of $a _ { t }$ . The goal-conditioned action head provides analogous structure relative to the goal latent. An exploratory all-heads checkpoint using the state-derived proxies in Supplementary Section A is retained as an implementation-specific comparison. Because those proxies are not the environment’s reward and value targets, this checkpoint neither isolates an R/V efect nor tests correctly specified reward- or value-aware training in general.

## Suficient-Condition Analysis

We complement the empirical diagnostics with an analytical account of when latent-cost ranking can be expected to track real-cost ranking. The goal is to identify the principal error sources, not to prove a tight rank-correlation bound.

Setup and assumptions. We use the notation of Section 2. For compactness, $f _ { \theta } ( s )$ denotes encoding the rendered observation $o = R ( s )$ . The encoder does not consume simulator state directly. Let T denote the environment dynamics, and write $s _ { H } ( \mathbf { a } )$ and $\hat { z } _ { H } ( { \bf a } )$ for the real and predicted terminal states under plan a. Real and latent costs are $c _ { \mathrm { r e a l } } ( \mathbf { a } ) = \| s _ { H } ( \mathbf { a } ) - s _ { g } \| _ { \Sigma }$ and $c _ { \mathrm { l a t } } ( \mathbf { a } ) = \| \hat { z } _ { H } ( \mathbf { a } ) - z _ { g } \| _ { 2 } ,$ respectively, for some PSD metric Σ. The planner uses the squared latent norm, but positive squaring preserves every candidate ranking, so the analysis uses the unsquared norm. The PushT diagnostics instantiate $\Sigma = I$ on the full sevendimensional state.

Assumption 1 (Pointwise terminal-rollout consistency). For every horizon-H candidate plan in the analysis, terminal rollout error satisfies $\lVert \hat { z } _ { H } ( \mathbf { a } ) - f _ { \theta } ( s _ { H } ( \mathbf { a } ) ) \rVert _ { 2 } \leq \epsilon _ { H }$

Assumption 2 (Encoder bi-Lipschitz). There exist constants $0 < \mu _ { f } \leq L _ { f }$ such that, for the terminal-state/goal pairs induced by candidate plans in the analysis,

$$
\mu _ { f } \| s - s ^ { \prime } \| _ { \Sigma } \leq \| f _ { \theta } ( s ) - f _ { \theta } ( s ^ { \prime } ) \| _ { 2 } \leq L _ { f } \| s - s ^ { \prime } \| _ { \Sigma } .
$$

SIGReg empirically discourages representational collapse, but by itself does not establish Assumption 2 or a lower bound on $\mu _ { f }$ . We therefore state bi-Lipschitz behavior as an assumption and measure collapse-related surrogates separately. Training losses are empirical diagnostics, not proofs that these assumptions hold of distribution.

## Pointwise cost-approximation bound.

Proposition 1 (Two-sided cost-approximation bound). Under Assumptions 1 and 2,for any horizon-H plan a,

$$
\mu _ { f } c _ { r e a l } ( { \bf a } ) - \epsilon _ { H } \leq c _ { l a t } ( { \bf a } ) \leq L _ { f } c _ { r e a l } ( { \bf a } ) + \epsilon _ { H } .\tag{6}
$$

The bracket width $( L _ { f } - \mu _ { f } ) c _ { \mathrm { r e a l } } + 2 \epsilon _ { H }$ decomposes into an encoder distortion term and a predictor error term. Our experiments measure collapse-related and rollout-error diagnostics, but do not identify these constants.

Proof sketch. The reverse triangle inequality and Assumption 1 give $\left| c _ { \mathrm { l a t } } - \| f _ { \boldsymbol \theta } ( s _ { H } ) - z _ { g } \| _ { 2 } \right| \leq \epsilon _ { H }$ . Assumption 2 (and $z _ { g } = f _ { \theta } ( s _ { g } ) )$ gives $\| f _ { \theta } ( s _ { H } ) - z _ { g } \| _ { 2 } \in [ \mu _ { f } c _ { \mathrm { r e a l } } , L _ { f } c _ { \mathrm { r e a l } } ]$ □

![](images/d749b0193067c4761f6918e1baa4336da5bde99489a0aab9a5081f57cba1b4da.jpg)  
Figure 2: Plan-Real Spearman measurement procedure. This operationalizes the candidate-order comparison in Figure 1. For each held-out pair k, the same N = 64 action sequences are evaluated by the world model and the environment, yielding paired latent- and real-cost vectors. Spearman gives a pair-level rank correlation $\rho _ { k } ;$ Plan–Real averages the defined correlations among $n = 3 0$ sampled pairs. CEM-stage Spearman reuses the paired scoring procedure with candidates from the selected CEM population.

Corollary 1 (Margin-implied rank preservation). Under the assumptions of Prop. 1, for any pair with $c _ { r e a l } ^ { ( i ) } < c _ { r e a l } ^ { ( j ) } ,$ $\mu _ { f } c _ { r e a l } ^ { ( j ) } > L _ { f } c _ { r e a l } ^ { ( i ) } + 2 \epsilon _ { H }$ implies $c _ { l a t } ^ { ( i ) } < c _ { l a t } ^ { ( j ) }$ . If every pair of N distinct candidates satisfies this, Plan-Real Spearman is +1. (Proof: subtract the upper bound at i from the lower bound at j.)

The condition requires (i) a low-distortion encoder $( L _ { f } / \mu _ { f } \to 1 )$ and (ii) non-vanishing candidate margins $\dot { \Delta _ { i j } } = \dot { c _ { \mathrm { r e a l } } } ^ { ( j ) } - c _ { \mathrm { r e a l } } ^ { ( i ) }$ . The latter can be small on contact-rich tasks under random sampling and can shrink further as CEM concentrates its proposal. This motivates measuring rank agreement at random, intermediate, and elite stages rather than assuming that one sampling regime is representative.

Empirical consistency check. Fit $c _ { \mathrm { l a t } } ~ \approx ~ \kappa c _ { \mathrm { r e a l } }$ and let $\eta _ { i } = | c _ { \mathrm { l a t } } ^ { ( i ) } - \kappa c _ { \mathrm { r e a l } } ^ { ( i ) } |$ . Kendall’s $\tau _ { a }$ averages pairwise concordance signs (ties count as zero). The soft-margin rate p is the fraction of distinct-real-cost pairs satisfying $\kappa \Delta _ { i j } > \eta _ { i } + \eta _ { j } .$ Thus $\tau _ { a } \geq 2 p - 1$ without real-cost ties (tie-aware form in Supplementary Section F). On PushT, p tracks Plan-Real Spearman with pooled Pearson +0.895 over 100 pairs (Supplementary Section F).

## The role of inverse-dynamics training.

Claim 1 (Inverse-action mechanism). Inverse-actionprediction encourages action-relevant latent transitions: the latent displacement $z _ { t + 1 } ~ - ~ z _ { t }$ must contain enough information to recover $a _ { t } .$ If the latent transition geometry is locally well-conditioned, neither degenerate along action-relevant directions nor anisotropically dilated relative to the data distribution, this constraint manifests as an increased rank correlation between latent displacement magnitude and action magnitude. Empirically, we observe that inverse-action training increases this global rank correlation from −0.03 for the LeWM baseline to +0.38for inverse-only and +0.43 for DA-LeWM. Inverse-only produces the largest single increment in Plan-Real Spearman among the measured ablations.

The claim is mechanistic, not a theorem. Low held-out inverse-prediction error is an empirical diagnostic rather than an assumption of Proposition 1, and it does not force norm alignment: action information could in principle be encoded directionally. We report latent-displacement-vs-action-norm rank correlation in Supplementary Section B as a falsifiable signature of the mechanism.

Implications. Encoder distortion $( L _ { f } / \mu _ { f } )$ and terminal rollout error $( \epsilon _ { H } )$ are the two analytical quantities. SIGReg and predictor training are empirical interventions associated with their diagnostics, not certificates of the constants. The inverse-action loss is instead mediated by Claim 1. Reward/value decodability is absent from the bound. We retain the exploratory all-heads checkpoint as an implementationspecific comparison using the proxy R/V targets described in Supplementary Section A. It is not a general test of rewardor value-aware world-model training.

## 5 Experiments

## Setup

We evaluate on four environments: PushT, Reacher, Cube, and TwoRoom, spanning planar object pushing, continuouscontrol reaching, contact-rich manipulation, and visual navigation. Dataset and task details are provided in Supplementary Section A.

Evaluation protocol. All methods use matched training and planning budgets. Online success is measured under closed-loop, receding-horizon MPC over 50 episodes for each of 3 evaluation seeds. Full training, CEM, and replanning details are provided in Supplementary Section A.

Variants compared. We compare five variants. LeWM is the SIGReg-regularized baseline with no auxiliary heads. No-SIGReg removes the regularizer and serves as the collapse control. Inverse-only adds only the inverse-action head.

![](images/52f7a512fa7339c59c35aeae6d4b662c569c8ee20f06b997e6cbab12201330d7.jpg)  
Figure 3: Probe accuracy vs. online success on PushT (3 evaluation seeds). Probe scores are nearly identical across the four non-collapsed variants, while online success difers sharply.

<table><tr><td rowspan="2">Task</td><td colspan="2">Success (%)</td><td colspan="2">Plan-Real Sp</td></tr><tr><td>LeWM</td><td>No-SIG</td><td>LeWM</td><td>No-SIG</td></tr><tr><td>PushT</td><td> $4 9 . 3 \pm 1 2 . 2$ </td><td> $2 . 0 \pm 2 . 0$ </td><td>+0.280</td><td>+0.031</td></tr><tr><td>TwoRoom</td><td> $9 8 . 0 \pm \ : \ : 2 . 0$ </td><td> $4 1 . 3 \pm 6 . 1$ </td><td>+0.549+0.012</td><td></td></tr><tr><td>Reacher</td><td> $8 2 . 0 \pm . 2 . 0$ </td><td> $1 0 . 7 \pm 8 . 1$ </td><td>+0.504 +0.001</td><td></td></tr></table>

Table 1: Efect of removing SIGReg across three environments (3 evaluation seeds, mean ± std).

All-heads adds inverse, goal-action, reward-proxy, and valueproxy heads, whereas DA-LeWM uses only the inverse and goal-action heads. We treat all-heads as exploratory because its reward/value targets are implementation-specific. Supplementary Section C reports sensitivity to the goal-action weight.

Evaluation sequence. We first use No-SIGReg to establish the collapse sanity check. We then compare non-collapsed PushT variants across information probes, Plan-Real Spearman, held-out action geometry, and online success. Next, CEM-stage Spearman tracks rank agreement throughout planning. Finally, short-budget and ten-epoch evaluations test control across tasks before Table 5 places DA-LeWM alongside published baselines.

## SIGReg Preserves Cost-Surface Variation

We audit cost-surface variation by the ratio of maximum to minimum latent cost over a random candidate population. Removing SIGReg contracts this ratio from 3−30× to ≈ 1.005×. A ratio near one means that the planner receives almost the same objective for every plan. Table 1 shows that this flattening coincides with near-zero Plan-Real Spearman and large online-success drops on PushT, TwoRoom, and Reacher.

Cube is handled separately in Section 5, because random no-contact candidates provide too few distinct real costs for a supported Spearman estimate.

## Non-Collapsed Variants Separate Information from Planning Geometry

We next compare LeWM, inverse-only, all-heads, and DA-LeWM after excluding the collapsed control. Their state, action, and goal-action probe scores vary by at most 0.03 in $R ^ { 2 } \left( { \mathrm { F i g u r e } } 3 \right)$ . In contrast, Plan-Real Spearman changes from +0.280 for LeWM to +0.410–+0.420 for action-supervised variants (Table 2), while online success spans 43 percentage points (49.3%→92.7%). This comparison isolates decisionmetric alignment from information suficiency rather than conflating either property with representation collapse. Table 2 retains No-SIGReg only as the collapse reference. For each of the 30 held-out start–goal pairs, Spearman ranks 64 candidates. The table reports the mean, and “Positive pairs” counts how many pair-level correlations exceed zero.

Inverse-action mechanism. On PushT, inverse-only raises Plan-Real Spearman from +0.280 to +0.420 and online success from 49.3% to 64.0%. DA-LeWM retains a similar +0.412 Spearman and reaches 92.7% success. Thus inverseonly accounts for most of the measured ranking lift, while the combined objective is associated with the additional online lift.

The checkpoint-level change is accompanied by the heldout transition signature in Claim 1: correlation between latent- and action-displacement magnitudes rises from −0.03 for LeWM to +0.38 for inverse-only and +0.43 for DA-LeWM, versus −0.13 without SIGReg. Because MPC continues to use Euclidean goal distance rather than an auxiliaryhead output, this evidence concerns the geometry exposed to the planner, not an added test-time cost.

Supplementary Section B gives the corresponding values for all variants.

## CEM-stage Spearman: Global Ranking Lift and Elite Saturation on PushT

Table 3 follows CEM iterations 0/15/29, corresponding to the random, mid, and elite stages. The random stage measures global candidate ordering. The later snapshots test ordering after the proposal has adapted to the model’s own cost. This stage-wise audit operationalizes the candidate-margin term in the analysis instead of relying on a single global correlation.

![](images/489b84c8501347f234437ad08a7d7105b6b04b83347ab8f2bfb346c49dc089aa.jpg)  
Figure 4: Online success throughout ten training epochs (3 evaluation seeds per task, mean ± std). Shading shows one standard deviation. Annotations report DA-LeWM minus LeWM averaged over epochs (Avg. ∆) and at epoch 10 $( \Delta _ { 1 0 } )$

<table><tr><td>Model</td><td>Plan-Real Sp</td><td>Positive pairs (out of 30)</td></tr><tr><td>LeWM</td><td>+0.280</td><td>24</td></tr><tr><td>No-SIGReg</td><td>+0.031</td><td>14</td></tr><tr><td>Inverse-only</td><td>+0.420</td><td>30</td></tr><tr><td>All-heads</td><td>+0.410</td><td>30</td></tr><tr><td>DA-LeWM</td><td>+0.412</td><td>29</td></tr></table>

Table 2: Full-state Plan-Real Spearman on PushT (n = 30).

<table><tr><td>Model</td><td>Random</td><td>Mid</td><td>Elite</td></tr><tr><td>LeWM</td><td>+0.403</td><td>+0.227</td><td>+0.036</td></tr><tr><td>No-SIGReg</td><td>+0.029</td><td>-0.017</td><td>-0.102</td></tr><tr><td>Inverse-only</td><td>+0.523</td><td>+0.261</td><td>-0.089</td></tr><tr><td>All-heads</td><td>+0.515</td><td>+0.249</td><td>-0.010</td></tr><tr><td>DA-LeWM</td><td>+0.536</td><td>+0.253</td><td>-0.011</td></tr></table>

Table 3: Full-state CEM-stage Spearman on PushT (n = 15 held-out pairs, CEM budget 300×30, top-30 elites).

Relative to LeWM, the three action-supervised variants produce consistent random-stage gains of +0.112 to +0.133 (paired $p { \leq } 0 . 0 1 2 )$ , with DA-LeWM highest at +0.536. These variants remain at +0.249–+0.261 mid-stage versus +0.227 for LeWM. Elite-stage correlations are near zero for every variant, indicating shared local saturation. Supplementary Section E reports paired significance and local-geometry checks. Tables 2 and 3 therefore answer complementary questions. The former tests independently sampled plans, while the latter follows the distributions visited by the optimizer.

## Short-Budget Control Across Environments

Having established the PushT ranking efect, we test whether it translates to control under the same one-epoch budget. Table 4 shows that DA-LeWM improves PushT by 43.4 pp and Cube by 10.6 pp, while Reacher remains within its strong baseline band.

<table><tr><td>Model</td><td>PushT</td><td>Reacher</td><td>Cube</td></tr><tr><td>LeWM</td><td> $4 9 . 3 \pm 1 2 . 2$ </td><td> $8 2 . 0 \pm 2 . 0$ </td><td> $6 2 . 7 \pm 4 . 2$ </td></tr><tr><td>No-SIGReg</td><td> $2 . 0 \pm \ : 2 . 0$ </td><td> $1 0 . 7 \pm 8 . 1$ </td><td> $5 2 . 7 \pm 1 . 9$ </td></tr><tr><td>Inverse-only</td><td> $6 4 . 0 \pm . 7 . 2$ </td><td> $8 2 . 7 \pm 3 . 1$ </td><td> $6 8 . 0 \pm 4 . 0$ </td></tr><tr><td>DA-LeWM</td><td> ${ \bf 9 2 . 7 \pm . 1 . 2 }$ </td><td> ${ \bf 8 4 . 0 \pm \Delta 3 . 5 }$ </td><td> ${ \bf 7 3 . 3 \pm 1 . 2 }$ </td></tr></table>

Table 4: Cross-environment online success (matched training budget, 3 evaluation seeds for all variants.)

Diagnostic scope. Reacher variants stay within [+0.49, +0.53], consistent with strong baseline rank agreement. For Cube, exact no-contact ties leave too few distinct real costs for a supported Plan-Real estimate. Its No-SIGReg latent-cost range still contracts from 4× to 1.07×, while online success drops from 62.7% to 52.7% (Table 4). We treat these as collapse and online-control evidence, not as a Cube Spearman or global bi-Lipschitz claim. Sensitivity to the goal-action weight β is in Supplementary Section C.

## Learning Dynamics under a Larger Budget

We next increase the training budget to ten epochs and compare LeWM with DA-LeWM on all four tasks. Figure 4 reports the resulting learning curves, with complete per-epoch statistics in Supplementary Section H.

The learning curves test whether the large one-epoch gains are merely an early-training efect. DA-LeWM enters the high-success regime earlier and maintains higher success throughout training on all four tasks. Its advantage therefore persists as LeWM receives more optimization, supporting accelerated convergence rather than a favorable single checkpoint.

Table 5 then asks whether faster learning also yields strong absolute performance against LeWM (Maes et al. 2026), PLDM (Sobal et al. 2025), and DINO-WM (Zhou et al. 2025). Relative to LeWM, DA-LeWM improves success by 2.7 pp on PushT, 1.3 pp on Reacher, 6.7 pp on Cube, and 9.0 pp on TwoRoom. It ranks first on PushT (98.7%) and Reacher (87.3%). On Cube, it surpasses PLDM and remains within 5.3 pp of DINO-WM. On TwoRoom, it reaches 96.0%, within 1.0 pp of PLDM and 4.0 pp of DINO-WM while remaining

9.0 pp above LeWM. Thus the faster learning is accompanied by stronger control performance rather than only an earlier optimization advantage.

The auxiliary heads are used only during training and are discarded at evaluation. LeWM and DA-LeWM therefore use the same inference-time MPC procedure and incur the same planning-time computation. Their separation along the learning curves isolates a representation-learning benefit: actionconditioned supervision makes the latent geometry useful to the unchanged planner earlier in training. Together, Figure 4 and Table 5 show improvements in both learning eficiency and control performance across task regimes.

<table><tr><td>Method</td><td></td><td></td><td></td><td>PushT Reacher Cube TwoRoom</td></tr><tr><td>Random</td><td>2</td><td>10</td><td>48</td><td>0</td></tr><tr><td>PLDM (Sobal et al. 2025)</td><td>78</td><td>78</td><td>65</td><td>97</td></tr><tr><td>DINO-WM (Zhou et al. 2025)</td><td>74</td><td>79</td><td>86</td><td>100</td></tr><tr><td>LeWM (Maes et al. 2026)</td><td>96</td><td>86</td><td>74</td><td>87</td></tr><tr><td>DA-LeWM (ours)</td><td>98.7</td><td>87.3</td><td>80.7</td><td>96.0</td></tr></table>

Table 5: Online success comparison (%). DA-LeWM values are means over 3 evaluation seeds.

## 6 Related Work

JEPA and latent world models. Joint-embedding predictive architectures learn visual representations by predicting in embedding space rather than reconstructing pixels (LeCun 2022; Assran et al. 2023; Bardes et al. 2024). Masked autoencoding and self-distillation provide complementary selfsupervised routes (He et al. 2022; Caron et al. 2021). LeWM adapts joint-embedding prediction to action-conditioned visual dynamics for manipulation, PLDM learns latent dynamics for ofline reward-free control, and DINO-WM plans over frozen pretrained features (Maes et al. 2026; Sobal et al. 2025; Zhou et al. 2025). These works establish capable representations and predictors for latent planning. Linear probes assess what is decodable (Alain and Bengio 2017). Our question is whether Euclidean distance to a goal latent induces the correct ordering over candidate action sequences.

Auxiliary objectives in RL and world models. Inverse dynamics has been used for curiosity and actionrelevant representation learning (Pathak et al. 2017; Zhang et al. 2021; Gelada et al. 2019), while contrastive and self-predictive objectives improve pixel-based RL (Laskin, Srinivas, and Abbeel 2020; Schwarzer et al. 2021). Demonstration-conditioned goal-action prediction relates to goal-conditioned imitation and hindsight relabeling (Ding et al. 2019; Andrychowicz et al. 2017). Here these established objectives provide action-conditioned supervision. We test how that supervision reshapes the metric structure consumed by fixed-cost latent MPC.

MPC and planning with learned models. PETS plans under learned rewards, while Dreamer learns behavior through reward-driven latent imagination (Chua et al. 2018; Hafner et al. 2020, 2021, 2023). MuZero and TD-MPC learn valueequivalent dynamics, and DayDreamer extends world-model control to physical robots (Schrittwieser et al. 2020; Hansen, Su, and Wang 2022, 2024; Wu et al. 2023). Our setting isolates a complementary requirement: when Euclidean goal distance is itself the planning cost, accurate prediction or decoding need not imply correct relative ordering. We measure that ordering globally and as CEM concentrates its proposal.

Decision-aware and bisimulation-based representations. Bisimulation metrics ground state similarity in reward and transition consequences (Ferns, Panangaden, and Precup 2004; Castro 2020; Zhang et al. 2021). Value equivalence provides an analogous criterion for model predictions (Grimm et al. 2020). Plan-Real and CEM-stage Spearman are planner-aligned empirical counterparts in this spirit: they audit the fixed metric actually consumed by the planner rather than learning a new task-specific metric.

## 7 Conclusion

We study when Euclidean goal distance provides a reliable objective for latent MPC, separating information suficiency from decision-metric alignment. Plan-Real and CEM-stage Spearman operationalize this distinction, while the suficientcondition analysis identifies encoder distortion, terminal rollout error, and candidate margins as the quantities controlling rank preservation. Across the non-collapsed variants, similar linear-probe scores coexist with substantial diferences in Plan-Real agreement and online success, showing that this geometric distinction matters for control. Guided by this empirical gap, DA-LeWM uses action-conditioned supervision to improve the cost geometry exposed to the planner. Across all four environments, DA-LeWM accelerates convergence and achieves higher online success than LeWM. Latent world models used for control should therefore be evaluated both for what they encode and for the ordering induced by the planning cost they expose.

Limitations. Our evidence is limited to four short-horizon simulated tasks, LeWM-family checkpoints with ViT-Tiny, Euclidean goal costs, and CEM. Each configuration has one training run, so uncertainty covers evaluation variation rather than training-initialization variation. Plan-Real requires simulator rollouts and loses statistical power under exact realcost ties. We fix α = 0.1 and test neither its sensitivity nor gradient-based planning. Generalization to robots, partial observability, larger backbones, other planners, and learned rewards, goal costs, or positive-semidefinite latent metrics remains open.

Generative-AI disclosure. Generative AI assisted language editing and code review. It was not a component of the method or evaluation, and the authors verified all resulting text and code.

## References

Alain, G.; and Bengio, Y. 2017. Understanding Intermediate Layers Using Linear Classifier Probes. In International Conference on Learning Representations (ICLR) Workshop Track.

Andrychowicz, M.; Wolski, F.; Ray, A.; Schneider, J.; Fong, R.; Welinder, P.; McGrew, B.; Tobin, J.; Abbeel, P.; and Zaremba, W. 2017. Hindsight Experience Replay. In Advances in Neural Information Processing Systems 30 (NeurIPS).

Assran, M.; Duval, Q.; Misra, I.; Bojanowski, P.; Vincent, P.; Rabbat, M.; LeCun, Y.; and Ballas, N. 2023. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 15619–15629.

Balestriero, R.; and LeCun, Y. 2025. LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. arXiv preprint arXiv:2511.08544.

Bardes, A.; Garrido, Q.; Ponce, J.; Chen, X.; Rabbat, M.; LeCun, Y.; Assran, M.; and Ballas, N. 2024. Revisiting Feature Prediction for Learning Visual Representations from Video. arXiv preprint arXiv:2404.08471.

Caron, M.; Touvron, H.; Misra, I.; Jégou, H.; Mairal, J.; Bojanowski, P.; and Joulin, A. 2021. Emerging Properties in Self-Supervised Vision Transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 9650–9660.

Castro, P. S. 2020. Scalable Methods for Computing State Similarity in Deterministic Markov Decision Processes. In Proceedings ofthe 34th AAAI Conference on Artificial Intelligence, 10069–10076.

Chua, K.; Calandra, R.; McAllister, R.; and Levine, S. 2018. Deep Reinforcement Learning in a Handful of Trials Using Probabilistic Dynamics Models. In Advances in Neural Information Processing Systems 31 (NeurIPS).

Ding, Y.; Florensa, C.; Abbeel, P.; and Phielipp, M. 2019. Goal-Conditioned Imitation Learning. InAdvances in Neural Information Processing Systems 32 (NeurIPS).

Ferns, N.; Panangaden, P.; and Precup, D. 2004. Metrics for Finite Markov Decision Processes. In Proceedings of the 20th Conference on Uncertainty in Artificial Intelligence (UAI), 162–169. AUAI Press.

Gelada, C.; Kumar, S.; Buckman, J.; Nachum, O.; and Bellemare, M. G. 2019. DeepMDP: Learning Continuous Latent Space Models for Representation Learning. In Proceedings of the 36th International Conference on Machine Learning (ICML), volume 97 of Proceedings ofMachine Learning Research, 2170–2179. PMLR.

Grimm, C.; Barreto, A.; Singh, S.; and Silver, D. 2020. The Value Equivalence Principle for Model-Based Reinforcement Learning. In Advances in Neural Information Processing Systems 33 (NeurIPS).

Ha, D.; and Schmidhuber, J. 2018. Recurrent World Models Facilitate Policy Evolution. In Advances in Neural Information Processing Systems 31 (NeurIPS).

Hafner, D.; Lillicrap, T.; Ba, J.; and Norouzi, M. 2020. Dream to Control: Learning Behaviors by Latent Imagination. In Proceedings of the 8th International Conference on Learning Representations (ICLR).

Hafner, D.; Lillicrap, T.; Fischer, I.; Villegas, R.; Ha, D.; Lee, H.; and Davidson, J. 2019. Learning Latent Dynamics for Planning from Pixels. In Proceedings ofthe 36th International Conference on Machine Learning (ICML), volume 97 of Proceedings ofMachine Learning Research, 2555–2565. PMLR.

Hafner, D.; Lillicrap, T.; Norouzi, M.; and Ba, J. 2021. Mastering Atari with Discrete World Models. In Proceedings of the 9th International Conference on Learning Representations (ICLR).

Hafner, D.; Pasukonis, J.; Ba, J.; and Lillicrap, T. 2023. Mastering Diverse Domains through World Models. arXiv preprint arXiv:2301.04104.

Hansen, N.; Su, H.; and Wang, X. 2022. Temporal Diference Learning for Model Predictive Control. In Proceedings of the 39th International Conference on Machine Learning (ICML), volume 162 of Proceedings of Machine Learning Research, 8387–8406. PMLR.

Hansen, N.; Su, H.; and Wang, X. 2024. TD-MPC2: Scalable, Robust World Models for Continuous Control. In The Twelfth International Conference on Learning Representations (ICLR).

He, K.; Chen, X.; Xie, S.; Li, Y.; Dollár, P.; and Girshick, R. 2022. Masked Autoencoders Are Scalable Vision Learners. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 16000–16009.

Laskin, M.; Srinivas, A.; and Abbeel, P. 2020. CURL: Contrastive Unsupervised Representations for Reinforcement Learning. In Proceedings of the 37th International Conference on Machine Learning (ICML), volume 119 of Proceedings of Machine Learning Research, 5639–5650. PMLR.

LeCun, Y. 2022. A Path Towards Autonomous Machine Intelligence. Position paper, version 0.9.2, OpenReview.

Maes, L.; Le Lidec, Q.; Scieur, D.; LeCun, Y.; and Balestriero, R. 2026. LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. arXiv preprint arXiv:2603.19312.

Park, S.; Frans, K.; Eysenbach, B.; and Levine, S. 2025. OGBench: Benchmarking Ofline Goal-Conditioned RL. In International Conference on Learning Representations (ICLR).

Pathak, D.; Agrawal, P.; Efros, A. A.; and Darrell, T. 2017. Curiosity-Driven Exploration by Self-Supervised Prediction. In Proceedings of the 34th International Conference on Machine Learning (ICML), volume 70 of Proceedings of Machine Learning Research, 2778–2787. PMLR.

Rubinstein, R. Y. 1999. The Cross-Entropy Method for Combinatorial and Continuous Optimization. Methodology and Computing in Applied Probability, 1(2): 127–190.

Schrittwieser, J.; Antonoglou, I.; Hubert, T.; Simonyan, K.; Sifre, L.; Schmitt, S.; Guez, A.; Lockhart, E.; Hassabis, D.; Graepel, T.; Lillicrap, T.; and Silver, D. 2020. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. Nature, 588(7839): 604– 609.

Schwarzer, M.; Anand, A.; Goel, R.; Hjelm, R. D.; Courville, A.; and Bachman, P. 2021. Data-Eficient Reinforcement Learning with Self-Predictive Representations. In Proceedings of the 9th International Conference on Learning Representations (ICLR).

Sobal, V.; Zhang, W.; Cho, K.; Balestriero, R.; Rudner, T. G. J.; and LeCun, Y. 2025. Stress-Testing Ofline Reward-Free Reinforcement Learning: A Case for Planning with Latent Dynamics Models. In 7th Robot Learning Workshop: Towards Robots with Human-Level Abilities.

Tassa, Y.; Doron, Y.; Muldal, A.; Erez, T.; Li, Y.; Casas, D. d. L.; Budden, D.; Abdolmaleki, A.; Merel, J.; Lefrancq, A.; Lillicrap, T.; and Riedmiller, M. 2018. DeepMind Control Suite. arXiv preprint arXiv:1801.00690.

Wu, P.; Escontrela, A.; Hafner, D.; Abbeel, P.; and Goldberg, K. 2023. DayDreamer: World Models for Physical Robot Learning. In Proceedings of the 6th Conference on Robot Learning (CoRL), volume 205 of Proceedings ofMachine Learning Research, 2226– 2240. PMLR.

Zhang, A.; McAllister, R.; Calandra, R.; Gal, Y.; and Levine, S. 2021. Learning Invariant Representations for Reinforcement Learning without Reconstruction. In Proceedings ofthe 9th International Conference on Learning Representations (ICLR).

Zhou, G.; Pan, H.; LeCun, Y.; and Pinto, L. 2025. DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning. In Proceedings ofthe 42nd International Conference on Machine Learning (ICML), volume 267 of Proceedings of Machine Learning Research, 79115–79135. PMLR.

## Appendix

## A Experimental Details

We provide here the full details necessary to reproduce all experiments. The implementation is built on top of the opensource LeWM codebase (Maes et al. 2026), with auxiliary heads and decision losses added in a modular fashion that does not modify the base world model architecture.

## Backbone Architecture

The encoder is a Vision Transformer (ViT-Tiny) backbone with patch size 14 and image resolution 224 × 224. The CLS token is projected to a latent embedding of dimension 192. The predictor is a 6-layer Transformer with 16 attention heads, head dimension 64, MLP dimension 2048, dropout 0.1, and embedding dropout 0. History length is 3 frames and the predictor outputs one future latent per forward pass. All variants share this exact backbone configuration. The only diferences are in the auxiliary head set and the loss weights.

## Auxiliary Head Architecture

Each auxiliary head is a 2-layer MLP with hidden dimension 256, GELU activations, and LayerNorm, followed by a final linear layer to the prediction target. Concretely:

• Inverse-action head: input $[ z _ { t } , z _ { t + 1 } ] \in \mathbb { R } ^ { 2 \cdot 1 9 2 }$ , output $\hat { \boldsymbol { a } } _ { t } \in \mathbb { R } ^ { d _ { \boldsymbol { a } } }$

• Goal-conditioned action head: input $[ z _ { t } , z _ { g } ] \in \mathbb { R } ^ { 2 \cdot 1 9 2 }$ output $\hat { \boldsymbol { a } } _ { t } \in \mathbb { R } ^ { d _ { a } }$

• Reward proxy head: input $[ z _ { t } , a _ { t } ]$ , scalar output.

• Value proxy head: input $z _ { t } ,$ , scalar output.

Here $d _ { a } = 5 d _ { \mathrm { r a w } }$ , where $d _ { \mathrm { r a w } }$ is the per-step action dimension. For the all-heads configuration we instantiate all four heads. The canonical no-R/V implementation used for the PushT DA-LeWM checkpoint instantiates only the inverseaction and goal-action heads (DecisionHeadsNoRV). For compatibility with the original cross-environment training jobs, the Reacher, Cube, and TwoRoom learning-curve checkpoints retain unused reward/value modules but set both corresponding loss weights to zero. Thus these checkpoints share the DA-LeWM training objective, although their parameter-initialization stream is not byte-identical to the no-R/V implementation.

Reward and value proxy targets. For the stored all-heads checkpoint, the implementation used

$$
\begin{array} { l } { r _ { t } = - \| s _ { t } ^ { [ : 2 ] } - s _ { g } ^ { [ : 2 ] } \| _ { 2 } } \\ { \quad \quad - 0 . 1 d _ { \cal L } ( s _ { t } ^ { [ 2 ] } , s _ { g } ^ { [ 2 ] } ) , } \\ { V _ { t } = \displaystyle \sum _ { k = t } ^ { T - 1 } \gamma ^ { k - t } r _ { k } , \qquad \gamma = 0 . 9 9 . } \end{array}\tag{7}
$$

Here $d _ { \angle }$ is wrapped angular distance. Neither target is an environment reward nor optimal-policy value. Under the stored PushT state layout, these coordinates do not match the intended block-pose fields, and no corrected all-heads checkpoint was trained. We therefore retain this result only as an exploratory, implementation-specific ablation. It cannot determine whether correctly specified R/V supervision helps or hurts, or explain the gap causally.

## Training Configuration

The matched-budget ablations train every variant for one epoch. The learning-dynamics comparison uses matched 100-epoch training schedules for LeWM and DA-LeWM and analyzes the first ten saved checkpoints. Optimizer, data, and model settings are otherwise identical within each comparison.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Weight decay</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Batch size</td><td>128</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Mixed precision</td><td>bf16</td></tr><tr><td>Training schedule</td><td>1 epoch (ablations), 100 epochs (first 10 analyzed)</td></tr><tr><td>Train/val split</td><td>0.9/0.1</td></tr><tr><td>Training randomness</td><td>Matched across variants</td></tr><tr><td>SIGReg weight  $\lambda _ { \mathrm { s i g } }$ </td><td>0.09</td></tr><tr><td>SIGReg knots</td><td>17</td></tr><tr><td>SIGReg projections</td><td>1024</td></tr><tr><td>Inverse weight α</td><td>0.1</td></tr><tr><td>Goal-action weight  $\beta$ </td><td>0.1</td></tr><tr><td>Reward weight (all-heads only)</td><td>0.05</td></tr><tr><td>Value weight (all-heads only)</td><td>0.01</td></tr><tr><td>Reward/value discount γ</td><td>0.99</td></tr><tr><td>Goal strategy</td><td>clip-final-state</td></tr></table>

The objective-level diference between all-heads and DA-LeWM is the presence or absence of reward/value losses. The implementation detail above records whether zeroweight modules are retained. The base world model loss $( \mathcal { L } _ { \mathrm { p r e d } } + \lambda _ { \mathrm { s i g } } \mathcal { L } _ { \mathrm { S I G R e g } } )$ is identical across all variants except No-SIGReg, where the SIGReg term is dropped.

## Computing Infrastructure

Training and evaluation use one NVIDIA A100-SXM4- 80GB or A800-SXM4-80GB GPU per process (80 GB memory, driver 535.129.03) on x86-64 Linux 5.4 cluster nodes with Intel Xeon Platinum 8362 host CPUs. The software environment is Python 3.10.20, PyTorch 2.7.1 (cu118), Lightning 2.6.1, Hydra 1.3.2, stable-worldmodel 0.0.6, stable-pretraining 0.1.6, dm-control 1.0.40, MuJoCo 3.8.0, OGBench 1.2.1, NumPy 2.2.6, SciPy 1.15.3, and scikit-learn 1.7.2. Training uses bfloat16 mixed precision. Diagnostics and summary statistics use float32 model inference and double-precision host-side aggregation where required.

## Datasets

All data are simulated trajectories from established public environments. This work introduces no new dataset and uses no human subjects or personally identifiable information. PushT and TwoRoom exercise contact-rich manipulation and obstacle-constrained navigation, Reacher supplies a standard continuous-control test from the DeepMind Control Suite (Tassa et al. 2018), and Cube supplies an ofline goalconditioned manipulation task from OGBench (Park et al. 2025). The exact dataset identifiers are:

<table><tr><td>Task</td><td>Dataset identifier</td><td>Provenance</td></tr><tr><td>PushT</td><td>pusht_expert_train</td><td>LeWM release</td></tr><tr><td>Reacher</td><td>dmc/reacher_random</td><td>Control Suite / LeWM</td></tr><tr><td rowspan="2">Cube</td><td>ogbench/cube_single_</td><td>OGBench /</td></tr><tr><td>expert</td><td>LeWM</td></tr><tr><td>TwoRoom</td><td>tworoom</td><td>LeWM release</td></tr></table>

Training follows the four-frame, frame-skip-5 construction in Section 4. The 25-row goal clip below is evaluation-only. All tasks use the same 0.9/0.1 trajectory-level train/validation split. The code package records dataset identifiers and preprocessing.

Task definitions. The diagnostic state, raw action dimension, online success rule, and terminal diagnostic cost are:
<table><tr><td>Task</td><td>Diagnostic state / cost</td><td> $d _ { \mathrm { r a w } }$ </td><td>Online success</td></tr><tr><td rowspan="3">PushT</td><td>agent xy, block</td><td>2</td><td>∥∆(agent xy, block  $x y ) \rvert \rvert _ { 2 } < 2 0$ </td></tr><tr><td>xy/angle, agent</td><td></td><td>and wrapped</td></tr><tr><td>velocity (7-D), L2</td><td></td><td>|∆ block angle|  $< \pi / 9$ </td></tr><tr><td></td><td>TwoRoom agent xy, L2</td><td>2</td><td> $\| \Delta x y \| _ { 2 } < 1 6$  pixels</td></tr><tr><td>Reacher</td><td>2-D joint position, L2</td><td>2</td><td> $\| \Delta q \| _ { \infty } < 0 . 0 5 \mathrm { r a d i a n s }$ </td></tr><tr><td>Cube</td><td>cube xyz, L2</td><td>5</td><td> $\| \Delta x y z \| _ { 2 } \le 0 . 0 4 \mathrm { m }$ </td></tr></table>

Here $d _ { \mathrm { r a w } }$ is the per-raw-step action dimension. Each model action block concatenates five raw actions. All listed L2 costs are unweighted, simulator-only terminal distances and are not exposed to the planner.

## Evaluation Configuration

Online evaluation is performed by initializing 50 vectorized environments per evaluation seed and running CEM-based planning. Each reported mean aggregates $3 \times 5 0 ~ = ~ 1 5 0$ episodes. Short-budget ablations and ten-checkpoint learning curves each use 3 evaluation seeds per task. Task-specific sweeps are evaluated separately and are not pooled.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Episodes per evaluation seed</td><td>50</td></tr><tr><td>Short-budget evaluation repeats Learning-curve evaluation repeats</td><td>3 seeds per task 3 seeds per task, task</td></tr><tr><td>Goal clip</td><td>sweeps separate</td></tr><tr><td> $( { \tt g o a l \_ o f f s e t \_ s t e p s } )$ </td><td>25 rows (endpoint +24)</td></tr><tr><td>Evaluation horizon (eval_budget)</td><td>50 raw steps</td></tr><tr><td>Planning horizon H</td><td>5 action blocks</td></tr><tr><td>Receding-horizon stride</td><td>5 action blocks</td></tr><tr><td>Action block size (frame-skip)</td><td>5 raw actions</td></tr><tr><td>CEM samples N (online)</td><td></td></tr><tr><td></td><td>300</td></tr><tr><td>CEM iterations K (online)</td><td>30</td></tr><tr><td>CEM elite size E (online) CEM proposal variance scale</td><td>30 1.0</td></tr></table>

The dataset loader uses the end-exclusive slice [start, start + 25), so the goal is the row at index $\mathrm { s t a r t } + 2 4 .$ The five-block planning and receding horizons each execute 25 raw actions. These parameters are identical across environments. Environment-specific callables (state setters, goal setters) are implemented as configured in the codebase and are not changed across variants.

## Plan-Real and CEM-stage Spearman

Diagnostic Spearman correlations use a separate evaluation procedure that reuses the same trained checkpoints. We sample $n \ = \ 3 0$ (start, goal) pairs per checkpoint from the held-out validation split. For each pair we draw $N = 6 4$ candidate action sequences from $\sqrt ( 0 , I )$ in normalized action-block space, compute latent costs from the world model, and roll out each candidate in the environment to compute real costs. For PushT, both Plan-Real and CEM-stage diagnostics use $c _ { \mathrm { r e a l } } = \| s _ { H } - s _ { g } \| _ { 2 }$ on the unweighted seven-dimensional simulator state (agent position, block position and angle, and agent velocity), recorded as $\mathtt { p u s h t \_ f u l l \_ s t a t e \_ l 2 \_ v 1 }$

CEM-stage Spearman uses the same CEM trajectory as the online MPC controller $( N = 3 0 0$ samples × 30 iterations with the top 30 as the elite set, $n _ { \mathrm { p a i r s } } = 1 5 )$ and reports Spearman at iteration 0 (random), iteration 15 (mid), and the last iteration (elite). At the elite stage each pair’s Spearman is computed over the top-30 candidates of the converged CEM distribution. At the random/mid stages it is computed over min $( N , \operatorname* { m a x } ( 2 K , 6 4 ) ) = 6 4$ candidates uniformly subsampled from the $N = 3 0 0$ population. This budget is identical for every checkpoint reported in Table 3 and Appendix C. All Spearman calculations use average ranks for exact ties. Diagnostic artifacts also record defined and undefined pair counts.

## Information Probes

Linear probes are fit on frozen latents from the trained world model. Each probe is an afine ridge regressor (closed-form least squares with $\ell _ { 2 }$ coeficient $1 \bar { 0 } ^ { - 4 }$ and an intercept). No hidden layer or nonlinear activation is used. The exact input– target pairs are: $z _ { t } \mapsto s _ { t }$ (state), $[ z _ { t } , z _ { t + 1 } ] \mapsto a _ { t }$ (inverse action), $[ z _ { t } , z _ { g } ] \mapsto a _ { t }$ (goal action), $[ z _ { t } , a _ { t } ] \mapsto r _ { t }$ (reward proxy), and $z _ { t } \mapsto V _ { t }$ (value proxy). The R/V targets are the legacy proxies defined above. Probe $R ^ { 2 }$ is reported on a fixed held-out 20% split. The probe definition and split are identical across model variants, so diferences in $R ^ { 2 }$ reflect properties of the frozen latent rather than probe capacity.

<table><tr><td>Model</td><td>Success</td><td>State  $R ^ { 2 }$ </td><td>Action  $R ^ { 2 }$ </td><td>Goal  $R ^ { 2 }$ </td></tr><tr><td>LeWM</td><td> $4 9 . 3 \pm 1 2 . 2$ </td><td>0.90</td><td>0.86</td><td>0.78</td></tr><tr><td>No-SIGReg</td><td> $2 . 0 \pm \ : 2 . 0$ </td><td>-6.14</td><td>0.14</td><td>-0.67</td></tr><tr><td>Inverse-only</td><td> $6 4 . 0 \pm . 7 . 2$ </td><td>0.89</td><td>0.88</td><td>0.77</td></tr><tr><td>All-heads</td><td> $7 1 . 3 \pm \ : 4 . 2$ </td><td>0.89</td><td>0.88</td><td>0.78</td></tr><tr><td>DA-LeWM</td><td> $\mathbf { 9 2 . 7 \pm . 1 . 2 }$ </td><td>0.90</td><td>0.89</td><td>0.80</td></tr></table>

Table 6: Information probes and online success on PushT (3 evaluation seeds). Across the four non-collapsed methods, probe scores change by less than 0.03 in $R ^ { \frac { \mathbf { \lambda } } { 2 } }$ while online success spans 43 percentage points. No-SIGReg is shown as a degenerate sanity check.

## B Geometric Signature of Inverse-Action Training

We measure the global rank correlation between latent displacement magnitude $| | \Delta z | |$ and action magnitude $\| a \|$ over 38,400 held-out transitions. Inverse-action training changes a near-zero baseline correlation into a moderate positive correlation, a falsifiable empirical signature of Claim 1. We do not assert this is a logical consequence of the inverse loss alone: the inverse head can in principle recover actions through directional not norm structure. The figure reports what we observe under the actual training procedure.

## C Additional Ablations

## Per-Method Probes and Online Success on PushT

Figure 3 (main text) plots the four non-degenerate methods. Table 6 below restates the same numbers and additionally lists the No-SIGReg degenerate baseline as a sanity check. Probes detect the outright information loss of No-SIGReg $( R ^ { 2 }$ negative or near zero) but do not separate the four methods that train without representation collapse.

## Goal-Action Weight Sensitivity

We sweep the goal-action weight $\beta$ on PushT online success and report Plan-Real / CEM-stage Spearman for $\beta = 0 . 3$ alongside $\beta = 0 . 1$

<table><tr><td> $\beta$ </td><td>PushT online (%)</td></tr><tr><td>0.03</td><td> $6 3 . 3 \pm 1 2 . 2$ </td></tr><tr><td>0.1</td><td> ${ \bf 9 2 . 7 \pm . 1 . 2 }$ </td></tr><tr><td>0.3</td><td> $8 9 . 3 \pm \ : \ : 3 . 1$ </td></tr></table>

Spearman at $\begin{array} { r l r } { \beta } & { { } = } & { 0 . 3 } \end{array}$ on PushT. Plan-Real Spearman i $\mathrm { \textrm { s } + 0 . 4 4 8 \ : \pm \ : 0 . 2 0 7 } \ : \ : \mathrm { ( 2 9 / 3 0 }$ positive pairs), and CEM-stage Spearman is (random, mid, elite) = $\left( + 0 . 5 4 9 , + 0 . 2 \bar { 3 } 2 , - \bar { 0 } . 0 3 8 \right)$ . Relative to LeWM, the Plan-Real and random-stage gains are significant (paired $p { = } 0 . 0 0 2$ for both), whereas mid and elite are not $\left( { p = 0 . 9 0 / 0 . 3 3 } \right)$ .

Additional same-budget initialization. A separately initialized $\beta ~ = ~ 0 . 1$ checkpoint gives Plan-Real Spearman $+ 0 . 4 3 6 \pm 0 . 2 2 4 ( 2 9 / 3 0$ positive pairs) and CEM-stage Spearman $\left( + 0 . 5 4 2 , + 0 . 2 3 \bar { 7 } , + 0 . 0 2 \bar { 8 } \right)$ at random/mid/elite. Its Plan-Real and random-stage gains over LeWM are significant (paired $p < 0 . 0 0 1$ and $p { = } 0 . 0 0 2 \rangle$ , whereas mid and elite are not $( p = 0 . 8 5 / 0 . 9 1 )$ . Together, the two additional checks reproduce global alignment improvement and shared local saturation.

Cross-environment behavior. On Reacher, $\beta = 0 . 3$ regresses to 78.0% (vs. 84.0% at $\beta = 0 . 1$ 1 and $8 2 . 0 \% \mathrm { L e W M }$ baseline). Reacher per-variant Spearman numbers including $\beta = 0 . 3$ are reported in Appendix G. The Reacher regression motivates the conservative cross-environment default $\beta = 0 . 1$

## D Scope of the Suficient-Condition Diagnostics

Proposition 1 is a suficient-condition analysis, not a claim that its constants are identified by the training losses. In particular, empirical average prediction loss does not establish the pointwise terminal-rollout constant $\epsilon _ { H }$ , and finite heldout pairs cannot certify global bi-Lipschitz constants $\mu _ { f }$ and $L _ { f }$ . We therefore do not attach numerical estimates to those constants. Instead, the paper tests observable implications at three levels: SIGReg collapse via latent-cost dynamic range and tie-aware Plan-Real Spearman on PushT, TwoRoom, and Reacher (main text), action-conditioned local geometry via Figure 5, and the rank-preservation implication via the softmargin check in Supplementary Section F. These measurements are diagnostics consistent with the analysis, not proofs that its assumptions hold out of distribution.

## E Elite-Neighborhood Local Latent Geometry

This appendix details the elite-neighborhood diagnostic referenced in Section 5. The diagnostic was originally designed to discriminate between two candidate mechanisms for an apparent elite-stage Spearman gap: (i) gradients from the R/V losses distort the local latent metric in the elite neighborhood (gradient interference), or (ii) R/V heads encode global task-progress scalars that are uninformative once candidates are near-optimal but do not distort local geometry (global-progress encoding). After the same-budget reevaluation in Table 3 and Supplementary Section C showed that elite-stage Spearman is uniformly weak across the five main variants and two additional DA-LeWM checks (with paired tests vs LeWM all non-significant), the elite-stage gap no longer requires a mechanistic explanation. The diagnostic results below remain useful: they provide direct evidence that all four head configurations share the same weakly-informative local latent metric in the elite neighborhood (Pearson(log d<sup>z</sup>, log d<sup>s</sup>) ≈ 0.08 across variants), which independently corroborates the elite-stage uniformity reported in Table 3.

Procedure. For each of $n { = } 1 5$ held-out (start, goal) pairs from the PushT validation set we run CEM with 300 candidates, 30 iterations and elite size $E = K = 3 0$ (matching the planner setting) and retain the final-iteration elite set. For every elite candidate i we record

![](images/40dccc8bc845905940889a8ed6effad7d3486c3019bdf484356223a582274499.jpg)  
Figure 5: Global Spearman correlation between $| | \Delta z | |$ and ∥a∥ on 38,400 held-out PushT transitions. Values are read directly from the saved diagnostic output. No synthetic points are shown.

$z _ { i } \in \mathbb { R } ^ { 1 9 2 }$ , the predicted final-step embedding produced by the JEPA rollout (the latent the planner scores against $z _ { g } )$ , and

$s _ { i } \in \mathbb { R } ^ { 7 }$ , the real PushT final-step state after rolling out the action sequence in the simulator.

We then compute pairwise distances within the elite set, $d _ { i j } ^ { z } = \lVert z _ { i } - z _ { j } \rVert _ { 2 }$ and $d _ { i j } ^ { s } = \| s _ { i } - s _ { j } \| _ { \Sigma }$ , with $\Sigma = \mathrm { d i a g } ( \sigma ^ { - 2 } )$ and σ the per-coordinate state standard deviation on the training set (diagonal Mahalanobis). The local geometric ratio is $\rho _ { i j } = d _ { i j } ^ { z } / d _ { i j } ^ { s }$ . A locally isotropic latent has roughly constant $\rho _ { i j }$ . Gradient interference would produce directiondependent contraction with elevated $\rho _ { i j }$ variance.

Results. Table 7 reports per-pair statistics aggregated over the n = 15 pairs.
<table><tr><td>Variant</td><td>ρ mean</td><td> $\rho \mathrm { C V }$ </td><td>Log spread</td></tr><tr><td>LeWM (no DS)</td><td></td><td>2.13 ± 0.86 0.710 ± 0.200</td><td> $1 . 8 7 \pm 0 . 2 6$ </td></tr><tr><td>Inverse-only</td><td> $2 . 9 0 \pm 1 . 0 8$ </td><td> $0 . 7 1 4 \pm 0 . 1 7 2$ </td><td> $2 . 0 3 \pm 0 . 3 6$ </td></tr><tr><td>All-heads</td><td> $2 . 9 2 \pm 1 . 2 0$ </td><td>0.685 ± 0.116</td><td> $1 . 9 8 \pm 0 . 4 1$ </td></tr><tr><td>DA-LeWM  $( \beta { = } 0 . 1 )$ </td><td></td><td>2.90 ± 1.04 0.767 ± 0.118</td><td> $2 . 1 2 \pm 0 . 2 0$ </td></tr></table>

<table><tr><td>Variant</td><td>Pearson(log d² , log d°)</td></tr><tr><td>LeWM (no DS)</td><td> $+ 0 . 0 9 2 \pm 0 . 0 6 7$ </td></tr><tr><td>Inverse-only</td><td> $+ 0 . 0 8 0 \pm 0 . 0 8 4$ </td></tr><tr><td>All-heads</td><td> $+ 0 . 0 8 4 \pm 0 . 0 6 5$ </td></tr><tr><td>DA-LeWM  $( \beta { = } 0 . 1 )$ </td><td> $+ 0 . 0 5 0 \pm 0 . 0 6 6$ </td></tr></table>

Table 7: Elite-neighborhood local latent geometry on PushT $( K = 3 0$ elites/pair, n = 15 pairs). Mean ± std across pairs. CV is $\sigma ( \rho ) / \mu ( \rho )$ . Log spread i $\mathrm { { s } } \log ( { p _ { 9 5 } / p _ { 5 } } )$

Across the four variants, the coeficient of variation spans 0.685–0.767 and log spread spans 1.87–2.12 in the elite neighborhood. In particular, all-heads does not difer significantly from inverse-only, DA-LeWM, or LeWM on either statistic. This provides no evidence for the R/V-specific

gradient-interference hypothesis: if the proxy R/V losses produced a detectable increase in local anisotropy, all-heads should difer from the R/V-free configurations. Table 8 reports the paired contrasts.
<table><tr><td>Contrast</td><td></td><td>∆CV (t)</td><td>∆ log spread (t)</td></tr><tr><td>All-heads — Inverse-only</td><td></td><td>−0.029 (−0.73)</td><td>-0.049 (−0.87)</td></tr><tr><td>All-heads – DA-LeWM</td><td></td><td>-0.081 (−1.85)</td><td>-0.135 (−1.09)</td></tr><tr><td>All-heads – LeWM</td><td></td><td>−0.024 (−0.47)</td><td>+0.119 (+0.85)</td></tr><tr><td></td><td>DA-LeWM – Inverse-only</td><td>+0.053 (+0.99)</td><td>+0.086 (+0.87)</td></tr><tr><td> $\mathrm { D A - L e W M - L e W M }$ </td><td></td><td>+0.057 (+0.93)</td><td>+0.255 (+2.50)</td></tr><tr><td>Inverse-only – LeWM</td><td></td><td>+0.004 (+0.06)</td><td>+0.169 (+1.34)</td></tr></table>

Table 8: Paired comparisons ofelite-neighborhood local geometry on PushT (n= 15 pairs, same pair indices across variants). Positive numbers indicate the first variant has higher anisotropy. Parentheses report paired t. All all-heads contrasts are non-significant. Among the remaining contrasts, only DA-LeWM versus LeWM log spread is significant $( p = 0 . 0 2 6 )$

The Pearson correlation Pearson(log d<sup>z</sup><sub>ij</sub>, log $d _ { i j } ^ { s } )$ is uniformly low (+0.050–+0.092) across all four variants, showing weak within-elite association between latent and realstate distance regardless of which auxiliary heads are present. This is consistent with the full-state elite-stage results in Table 3: elite correlations remain near zero and the paired test against LeWM is non-significant for every main configuration. The isolated DA-LeWM–LeWM log-spread contrast does not support an R/V-specific efect because all three contrasts involving the all-heads checkpoint are non-significant.

Implementation. The diagnostic is implemented in decision/eval\_elite\_local\_geometry.py and re-uses the CEM inner loop and dataset / scaler / env wiring of the behavioral Plan-Real Spearman pipeline. Total wallclock for the four variants on a single 80 GB A100/A800 GPU is ≈9 minutes (n=15 pairs, 30 CEM iterations × 300 samples per iteration), so the diagnostic is essentially free relative to the CEM-stage Spearman evaluation it complements.

## F Soft-Margin Consistency Check for Corollary 1

## Procedure. Implementation:

decision/eval\_corollary\_margin\_check.py. For each PushT validation pair we sample $N = 6 4$ random plans (matching the Plan-Real Spearman protocol), compute $c _ { \mathrm { l a t } } ^ { ( i ) }$ via JEPA and the full-state PushT cost $c _ { \mathrm { r e a l } } ^ { ( i ) } = \| s _ { H } ^ { ( i ) } - s _ { g } \| _ { 2 }$ via environment rollout, fit $\begin{array} { r } { \kappa = \sum _ { i } c _ { \mathrm { l a t } } ^ { ( i ) } c _ { \mathrm { r e a l } } ^ { ( i ) } / \sum _ { i } ( c _ { \mathrm { r e a l } } ^ { ( i ) } ) ^ { 2 } } \end{array}$ (OLS through origin), and define residuals $\eta _ { i } = | c _ { \mathrm { l a t } } ^ { ( i ) } - \kappa c _ { \mathrm { r e a l } } ^ { ( i ) } | .$ The soft margin condition is κ $\Delta _ { i j } > \eta _ { i } + \eta _ { j }$ for the ordered pair with $c _ { \mathrm { r e a l } } ^ { ( i ) } < c _ { \mathrm { r e a l } } ^ { ( j ) }$ . We define p as the fraction of such distinct-real-cost pairs satisfying the condition. Kendall’s $\tau _ { a }$ is the mean concordance sign over all unordered candidate pairs, with tied pairs contributing zero. If q is the fraction of pairs with distinct real costs, the tie-aware bound is $\tau _ { a } \geq q ( 2 p - 1 )$ . Here $q = 1$ for the reported full-state PushT pairs.

Results. Table 9 reports per-variant means over $n = 2 0$ pairs.
<table><tr><td>Variant</td><td> $\rho _ { s }$ </td><td> $\tau _ { a }$ </td></tr><tr><td>LeWM (no DS) 0.288 ± 0.073 +0.338 ± 0.206 No-SIGReg Inverse-only</td><td> $0 . 2 2 1 \pm 0 . 0 5 2 + 0 . 0 5 3 \pm 0 . 2 0 9 + 0 . 0 3 7 \pm 0 . 1 4 2$   $0 . 3 1 4 \pm 0 . 0 8 8 + 0 . 4 4 8 \pm 0 . 2 0 7 + 0 . 3 2 4 \pm 0 . 1 5 4$   $0 . 3 1 7 \pm 0 . 0 9 3 + 0 . 4 5 7 \pm 0 . 2 3 6 + 0 . 3 3 1 \pm 0 . 1 7 8$ </td><td> $+ 0 . 2 3 9 \pm 0 . 1 4 8$ </td></tr><tr><td>All-heads DA  $( \beta { = } 0 . 1 )$ </td><td> $0 . 3 2 5 \pm 0 . 0 9 1 + 0 . 4 7 9 \pm 0 . 2 2 8 + 0 . 3 4 7 \pm 0 . 1 7 4$ </td><td></td></tr><tr><td></td><td>Variant 2p−1</td><td>κ</td></tr><tr><td></td><td>LeWM (no DS)</td><td> $- 0 . 4 2 4 \pm 0 . 1 4 6 ~ + 1 . 9 3$ </td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>No-SIGReg</td><td> $- 0 . 5 5 7 \pm 0 . 1 0 3 + 0 . 0 0 3$ </td></tr><tr><td></td><td></td><td> $- 0 . 3 7 1 \pm 0 . 1 7 6 \ + 2 . 2 8$ </td></tr><tr><td></td><td>Inverse-only</td><td></td></tr><tr><td></td><td>All-heads</td><td> $- 0 . 3 6 7 \pm 0 . 1 8 7 \ + 2 . 3 2$ </td></tr><tr><td></td><td>DA  $( \beta { = } 0 . 1 )$ </td><td> $- 0 . 3 5 1 \pm 0 . 1 8 2 \ + 2 . 3 4$ </td></tr></table>

Table 9: Corollary 1 soft-margin consistency check on PushT $( N = 6 4$ random plans/pair, $n = 2 0$ pairs, mean ± std across pairs). $\tau _ { a } \geq 2 p - 1$ holds for every pair.

Across the five variant means, p tracks $\rho _ { s }$ with Pearson correlation +0.9996 (Spearman rank +1.0). Pooling all $5 \times 2 0 =$ 100 (variant, pair) points gives Pearson $( p , \rho _ { s } ) = + 0 . 8 9 5$ $( p = 3 . 5 4 \times \bar { 1 0 } ^ { - 3 6 } ) ^ { \cdot }$ . The Kendall lower bound $\tau _ { a } \geq 2 p - 1$ holds for all 100 pairs. The empirical gap $\tau _ { a } - ( 2 p - 1 )$ is $0 . 6 7 0 \pm 0 . 0 8 3$ . Thus the soft condition tracks ranking consistently while remaining a conservative, loose lower bound.

## G Cross-Environment Spearman Scope

This appendix collects the behavioral Plan-Real Spearman and CEM-stage Spearman measurements on Reacher and explains why the same summary is not meaningful on contactsparse Cube, supporting the cross-environment discussion in Section 5. Table 2 and Table 3 in the main body cover PushT in detail.

## Reacher

Reacher exhibits a high baseline alignment (LeWM Sp = +0.504). Auxiliary objectives leave behavioral Spearman essentially unchanged, consistent with limited empirical headroom in this diagnostic. These measurements do not estimate the constants or tightness of Proposition 1.
<table><tr><td>Variant</td><td>Behavior</td><td>Random</td><td>Mid</td><td>Elite</td></tr><tr><td>LeWM</td><td> $+ 0 . 5 0 4 \pm 0 . 2 4 4$ </td><td>+0.468</td><td>+0.446</td><td>+0.220</td></tr><tr><td>No-SIGReg</td><td> $+ 0 . 0 0 1 \pm 0 . 1 3 5$ </td><td>-0.015</td><td>-0.018</td><td>-0.045</td></tr><tr><td>Inverse-only</td><td> $+ 0 . 5 0 6 \pm 0 . 2 5 4$ </td><td>+0.486</td><td>+0.457</td><td>+0.111</td></tr><tr><td>DA</td><td> $( \beta = 0 . 1 ) ~ + 0 . 5 2 3 \pm 0 . 2 1 7$ </td><td>+0.492</td><td>+0.460</td><td>+0.231</td></tr><tr><td>DA</td><td> $( \beta = 0 . 3 ) ~ + 0 . 5 1 4 \pm 0 . 2 4 2$ </td><td>+0.486</td><td>+0.467</td><td>+0.097</td></tr></table>

The variation across alignment-improving variants is within sampling noise $( n = 3 0$ pairs, std ∼ 0.22), and online success similarly varies by less than 3 percentage points across the same variants.

## Cube: Tie-Dominated Real Costs

Random Cube plans often fail to contact the object. Consequently, many candidates have exactly the same terminal object position and real cost. Some 64-candidate populations contain a tie block of size 63, and fully constant real-cost vectors also occur in both behavioral and CEM-stage populations. Average-rank Spearman is undefined for a constant vector and is low-support when only one or two distinct costs remain.

We therefore do not report Cube Spearman as a crossvariant metric. The appropriate conclusion is narrower: random-plan rank agreement is not informative in this contact-sparse regime. Cube comparisons in the paper use online success, while the tied-cost frequency is itself reported as a limitation of Plan-Real and CEM-stage diagnostics.

## H Training Dynamics and Checkpoint Selection

We evaluate checkpoints from epochs 1 through 10 of matched 100-epoch schedules for LeWM and DA-LeWM. Training randomness is matched across the two methods, and online evaluation uses 50 episodes for each of 3 evaluation seeds per task. Task-specific sweeps are evaluated separately and are not pooled. Intermediate checkpoints are non-monotonic, especially on Reacher and Cube. We therefore use epoch 10 as a common, pre-specified endpoint and do not select a separate best epoch for each method. The complete per-epoch means and sample standard deviations underlying Figure 4 are included in the accompanying results file.