# VERDICT: Training-Free Step-Wise Verification of Multimodal Reasoning via Disagreement-Aware Consensus

Rohit Sinha\*<sup>1</sup>, Kunal Tilaganji\*<sup>1</sup>, Tanuja Ganu<sup>2</sup>, Nagarajan Natarajan<sup>2</sup>, Amit Sharma<sup>2</sup>, and Vineeth Balasubramanian<sup>1,2</sup>

<sup>1</sup> Indian Institute of Technology, Hyderabad

https://www.iith.ac.in/

<sup>2</sup> Microsoft Research

https://www.microsoft.com/en-us/research/lab/microsoft-research-india/

Abstract. Multimodal large language models often generate reasoning chains containing subtle errors that lead to incorrect answers. Current verification approaches have notable limitations. Existing approaches either require expensive labelled supervision with inconsistent cross-task performance or aggregate scores from multiple sources by simple aggregations, missing a key insight: when these scores disagree, that disagreement itself carries important information about whether a reasoning step is truly valid or not. We formalise this as a coupled scoring problem among disparate, frozen verifiers, interpretable as a coordination game with a unique closed-form equilibrium where agreement signals valid steps while disagreement reveals instability. Towards this end, we propose a training-free domain-agnostic step-wise verification approach we call VERDICT: VERification via Disagreement-Informed Coupled Thresholding. To our knowledge, VERDICT is the first training-free verifier that makes the structure of cross-modal disagreement explicit and actionable. It computes consensus scores through a closed-form solution, enabling both disagreement-aware filtering and stability-conscious ranking of reasoning steps. Evaluated across six benchmarks, VERDICT consistently improves over the base model by up to +5.95%, and performs competitively with domain-specific critics that demand extensive supervision, demonstrating that cross-modal agreement provides robust verification signals without task-specific adaptation.

Keywords: Multimodal Reasoning · Step-wise Verification · Multi-Agent Consensus · Training-Free Verification

## 1 Introduction

Multimodal large language models (MLLMs) [1,27] have demonstrated impressive multi-step reasoning capabilities over images and text. By decomposing complex questions into chains of intermediate steps, these models can tackle compositional visual reasoning tasks that require both perceptual grounding and logical inference [51,5]. Yet their reasoning chains often contain subtle errors, unsupported visual claims, logical gaps, and hallucinated spatial relationships that are dificult to detect precisely because each step appears locally plausible [23,26]. These errors propagate: a single flawed step early in the chain can cascade through subsequent reasoning, ultimately leading to an incorrect final answer. This motivates step-level verification, catching errors where they arise rather than after they have shaped the trajectory of the reasoning chain.

![](images/afd6779672a86986975968250f6f260375bd7e995a0636dcd47af2f6aeaf614d.jpg)  
Selected (k + 1)th Step extends the reasoning chain  
Fig. 1: An illustrative overview of VERDICT.Given a multimodal question and partial reasoning trace, the base model generates candidate continuations that are independently scored by three frozen, modality-specialized agents. A closedform consensus computation transforms the raw scores into disagreement-aware consensus scores; a dual acceptance criterion then filters by mean confidence and consensus dispersion, selecting the most stable step to extend the reasoning chain.

Existing verification approaches fall into two broad categories, each with notable limitations. Domain Specific critics, including process reward models (PRMs) [25,41,28] and their multimodal extensions such as VisualPRM [44], Sherlock [9], LLaVA-Critic [46], and MM-Verify [37], score individual reasoning steps using trained models. However, these require carefully labeled supervision, often obtained through expensive Monte Carlo rollouts or human annotations [28,45]. More critically, they show inconsistent cross-task performance: while they may improve accuracy on one benchmark, they frequently degrade on others. This limitation is especially acute in data-scarce domains, where the labeled supervision these critics require is prohibitively expensive or simply unavailable.Training-free aggregation methods like Weaver [34] and reward model ensembles [12,7] combine scores from multiple sources, but they miss a critical insight. Consider two scoring patterns: unanimous confidence (0.7, 0.7, 0.7) and conflicting evidence (0.9, 0.3, 0.9). Simple averaging treats both as equivalent, yet they reflect fundamentally diferent verification states. This problem is compounded by well-documented capability tensions within MLLMs themselves: visual grounding can disrupt reasoning [17,50], visual instruction tuning degrades language performance [35], and extended reasoning chains cause forgetting of earlier context [39]. These tensions mean that diferent evaluation perspectives can legitimately disagree about the same reasoning step and that disagreement carries diagnostic value.

Our key insight is that when a reasoning step is truly valid, disparate judges evaluating it from diferent perspectives, such as visual grounding, logical consistency, and contextual relevance, should be able to converge. When they cannot agree, even after accounting for each other’s views, the step is likely unstable. Rather than treating verification as a classification or regression task, we treat it as a coupled scoring problem among disparate sources of evidence. We formalize this as a coupled scoring framework among frozen, modality-specialized judges, whose consensus yields disagreement-aware scores. The consensus captures how each judge would revise their belief in light of what others think, producing scores that naturally balance collective confidence against residual disagreement. Unlike variance-based approaches (which ignore belief strength) or simple averaging (which buries disagreement), VERDICT is a disagreementaware consensus framework, which makes the structure of conflict explicit and actionable.

Concretely, three frozen agents, Visual, Logical, and Contextual, disparately score each candidate’s reasoning step. A closed-form consensus computation then transforms these raw scores into consensus-adjusted scores, without any iterative optimization or learning. A dual acceptance criterion combines a mean confidence check (ensuring suficient collective endorsement) with a dispersion check (requiring cross-modal agreement). Among accepted candidates, ranking by consensus confidence selects the most stable continuation. The entire framework operates as a plug-in verifier requiring no modification to the base model, no training data, and no task-specific adaptation. This formulation admits a natural game-theoretic interpretation: each agent is a player in a coordination game, and the closed-form solution corresponds to its unique Nash equilibrium [31], grounding the influence structure in equilibrium theory rather than ad hoc design choices.

Evaluated across six benchmarks spanning spatial reasoning, visual grounding, and multimodal abstraction, VERDICT achieves consistent improvements of up to +5.95% over baseline models and performs competitively with or surpasses domain-specific critics that require extensive labeled training data. Crucially, our method never degrades below baseline on any tested benchmark, a property that none of the domain-specific critics we evaluate can claim, as they show significant performance variability across tasks.

It is worth noting that Averaging [7,42] is symmetric as every judge’s deviation counts equally, and the pattern of who disagrees with whom is lost. Variancebased filtering [20] treats a stubborn visual agent and a flexible contextual agent identically. Our disagreement-aware consensus formulation introduces coupling where each agent’s adjusted score depends on all others, so disagreement from a stubborn agent weighs more heavily: an asymmetry no fixed weighting can recover. This coupling is what distinguishes VERDICT. Our contributions are summarized below:

– We formalize step-wise verification as a coupled scoring problem among disparate, frozen verifiers, establishing disagreement structure as a first-class verification signal and proposing disagreement-aware verification as a design principle.

![](images/12823ddbe58f59acb8e961329e263f756bbd8b9213909b7f69346ec1023e633c.jpg)  
Fig. 2: Qualitative examples showing base model reasoning (orange) and VERDICT-verified reasoning (blue). Bold highlighted text marks the critical sentence where the base model commits to an incorrect reasoning trajectory. VERDICT recovers by selecting an alternative candidate step that corrects the judgment, yielding the correct answer. Full reasoning chains are provided in the supplementary material.

We introduce VERDICT, a consensus verifier with a closed-form solution (the unique equilibrium of a coordination game among modality-specialized judges) that models cross-modal agreement, enabling both disagreementaware filtering and stability-conscious ranking of candidate reasoning steps.

– We provide a domain-agnostic, training-free, plug-in implementation that integrates directly into MLLM reasoning without base model modification and demonstrates consistent improvements across six benchmarks spanning spatial reasoning, visual grounding, and multimodal abstraction.

– We conduct comprehensive ablation studies revealing that the framework operates through two complementary mechanisms, rejection and selection, and that consensus scores function optimally as ranking tools rather than binary classifiers.

## 2 Related Work

Step-level verification has largely developed along two tracks. Process reward models from PRM800K [25] through Math-Shepherd [41] and OmegaPRM [28] to multimodal extensions like MM-PRM [10] and VisualPRM [44] demonstrate that intermediate supervision improves reasoning reliability, but at the cost of expensive human annotation or Monte Carlo rollouts that our approach requires none of. Domain-specific critic models [9,48,37,46,24] achieve strong results on specific benchmarks yet implicitly collapse heterogeneous evidence into a single score, which, as our experiments show, leads to fragile cross-task generalization; a critic that learns to reward one reasoning style can quietly penalize another, and the scalar it produces carries no record of which modality drove the judgment.

This opacity is not incidental; it is structural, and it is what motivates an explicit treatment of disagreement rather than a better-trained aggregator. Ensemble and multiagent methods [12,34,11,20] take a step toward robust-

Table 1: VERDICT vs. prior methods across three design axes (Section 3).
<table><tr><td>Method</td><td>Training Free</td><td>Disagree. Aware</td><td>Plug-in Compat.</td></tr><tr><td>Domain Specific Critics [25,44,9,46,3]</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Reward Ensembles [12]</td><td>√</td><td>x</td><td>√</td></tr><tr><td>Weaver [34]</td><td>√</td><td>x</td><td>√</td></tr><tr><td>Variance Filtering [20]</td><td>√</td><td>x</td><td>√</td></tr><tr><td>VERDICT (Ours)</td><td>√</td><td>√</td><td></td></tr></table>

ness by combining mul-

tiple sources of evidence,

but aggregate through av-

eraging or majority voting, treating all patterns of agreement and disagreement as equivalent; a three-way split and a near-unanimous vote are indistinguishable once reduced to a mean. The richer question- not whether agents disagree, but how and between whom is precisely what averaging discards and what our framework is designed to recover. Multi-agent formulations, prominent in GANs and adversarial training [33], have so far relied on adversarial zero-sum games where agents work against one another by design. Our formulation departs from all of these: we employ a coupled scoring framework in which frozen, modalityspecialized judges seek consensus rather than oppose one another, and where the closed-form solution makes the structure of their disagreement explicit rather than averaging it away: a property that, to our knowledge, no existing trainingfree step-wise verification method provides. Table 1 summarizes how VERDICT relates to prior work along the three axes that motivate its design.

## 3 Preliminaries

In this section, we formalize the verification setting and define the problem addressed by our approach.

Verification Setting. We consider a setting where a base MLLM generates a reasoning chain step by step, and each step must be evaluated before extending the chain further. Formally, at reasoning step t, the base model is provided with an image I, a question $Q ,$ and a partial reasoning trace $r _ { 1 : t - 1 }$ consisting of all previously accepted steps. From this context, the base model generates n candidate continuations $\{ r _ { t } ^ { ( 1 ) } , \ldots , r _ { t } ^ { ( n ) } \}$ via sampling. A verifier then evaluates each candidate and returns a binary decision: accept the step and continue reasoning, or reject it and resample. Crucially, evaluation is local to the current step; it does not require access to future reasoning or to the final answer. This locality allows errors to be intercepted at the point where they arise, before they propagate through subsequent steps.

Problem Definition. Given n candidate steps at position t, m specialized judge agents $\{ A _ { 1 } , \ldots , A _ { m } \}$ , and context $\left( I , Q , r _ { 1 : t - 1 } \right)$ , the verification problem is to produce a single verified continuation $\boldsymbol { r } _ { t } ^ { * }$ . Each agent $A _ { i }$ independently scores every candidate $r _ { t } ^ { ( j ) }$ with a scalar confidence $\hat { s } _ { i } ^ { ( j ) } \in [ 0 , 1 ]$ reflecting its modality-specific assessment. The verifier must aggregate these heterogeneous assessments into a selection decision, accounting for both collective confidence and inter-agent agreement.

We impose three design constraints: (1) training-free, requiring no labeled data or parameter updates; (2) disagreement-aware, treating cross-modal conflict as a diagnostic signal rather than noise; and (3) plug-in compatible, integrating with any base MLLM without architectural modification. Domain-specific critics violate (1); standard aggregation methods violate (2). Constraint (2) deserves emphasis: variance-based rejection treats all agents symmetrically, contributing equally regardless of their modality-specific reliability, and any two score vectors with identical means and per-agent deviations are deemed equivalent even when their disagreement patterns difer qualitatively. What is needed is a mechanism that couples agents one in which each agent’s adjusted assessment depends on where every other agent landed. As we show in Section 4 (Proposition 1), this coupling cannot be replicated by any separable function of individual scores, and it is precisely what VERDICT provides through a coordination-game formulation.

## 4 VERDICT: Disagreement-Aware Consensus Verification

We now present VERDICT, our verification framework. We first describe the verifier agents and their roles, then formulate their interaction as a coupled scoring problem, derive the closed-form consensus , and finally define the acceptance criterion used to select verified reasoning steps.

Verifier Agents. Verification is carried out by three frozen MLLMs, each prompted to judge a candidate’s reasoning step from a distinct modality-specific perspective. We instantiate three agents:

– Visual Agent (V): assesses whether objects and spatial relationships mentioned in the reasoning step are visually verifiable in the provided image. It checks whether the step is grounded in what can actually be observed.

– Logical Agent (L): evaluates whether the reasoning step follows logically from the preceding steps and makes progress toward answering the question. It checks for coherence, valid inferences, and logical gaps.

Contextual Agent (C): determines whether the step maintains focus on the original question and avoids introducing irrelevant or tangential information. It penalizes unnecessary speculation and of-topic claims.

Each agent receives the image I, question Q, the partial reasoning trace $r _ { 1 : t - 1 }$ (assumed correct), and the current candidate step $r _ { t } ^ { ( j ) }$ to evaluate. It outputs a single scalar score $\hat { s } _ { i } \in [ 0 , 1 ]$ , interpreted as its subjective confidence that the step is valid given its modality-specific evidence. Agents operate in complete isolation: they never see other agents’ scores, their prompts contain no information about other agents’ judgments, and no fine-tuning or calibration is performed.

Number of agents: The formulation is defined for arbitrary m agents: Eqs. (1)-(4) and Proposition proposition 1 hold for any m ≥ 2. We instantiate three because they map onto three largely orthogonal MLLM failure modes: unsupported visual claims [17,50], logical gaps [39], and contextual drift [35]. The coupled system remains closed-form for larger m, and its cost is dominated by the m scoring calls rather than the $m \times m$ linear solve. The binding constraint for adding agents is semantic, not computational: each must bring a genuinely disparate evaluation perspective.

Coupled Scoring Formulation. Rather than aggregating raw scores directly, we frame verification as a coupled scoring problem among the agents. The motivating intuition is straightforward: if a reasoning step is truly valid, disparate judges evaluating it from diferent perspectives should be able to reach agreement. If they cannot agree even after accounting for each other’s views, the step is likely unstable.

We model this interaction as a coupled system in which each agent i selects a reported score $s _ { i } ~ \in ~ [ 0 , 1 ]$ , balancing agreement with the other agents against fidelity to its own modality-specific judgment. The scoring objective to agent i is defined as:

$$
u _ { i } ( s _ { i } , s _ { - i } ) = - ( s _ { i } - \bar { s } _ { - i } ) ^ { 2 } - \lambda _ { i } ( s _ { i } - \hat { s } _ { i } ) ^ { 2 }\tag{1}
$$

where $\begin{array} { r } { \bar { s } _ { - i } = \frac { 1 } { m - 1 } \sum _ { j \neq i } s _ { j } } \end{array}$ denotes the mean reported score of the other agents, $\hat { s } _ { i }$ is agent i’s raw confidence score, and $\lambda _ { i } > 0$ controls how strongly agent i weights its own evidence relative to group consensus. The first term encourages agreement: it penalizes the agent for deviating from what others report. The second term enforces self-consistency: it penalizes deviation from the agent’s original belief, scaled by $\lambda _ { i } .$ . Under the coordination game interpretation, Eq. eq. (1) is each agent’s payof: it is maximized when the agent finds a score that balances fidelity to its own signal against coordination with others, which is the defining tension of a coordination game. The stubbornness parameter $\lambda _ { i }$ encodes how resistant each agent should be to consensus pressure. A higher $\lambda _ { i }$ means agent i trusts its own modality-specific evidence more strongly and will deviate less from its raw score even when others disagree. In our experiments, we set $\lambda _ { V } = 1 . 5 , \lambda _ { L } = 1 . 0$ , and $\lambda _ { C } = 0 . 8 .$ reflecting the intuition that the visual agent should be most resistant to consensus pressure on perception-heavy tasks, while the contextual agent may be more flexible when visual or logical evidence is strong. These values are fixed across all datasets. Since the objective function in Eq. (1) is strictly concave in each agent’s own score (the second derivative $\begin{array} { r } { \frac { \partial ^ { 2 } u _ { i } } { \partial s _ { i } ^ { 2 } } = - 2 ( 1 + \lambda _ { i } ) < 0 } \end{array}$ for all $\lambda _ { i } > 0 )$ , the coupled system admits a unique fixed point (equivalent to the unique Nash equilibrium of the induced coordination game). This follows from Rosen’s uniqueness result for concave interaction systems, [32].

Closed-Form Solution. At the consensus solution, each agent’s reported score satisfies the following:

$$
s _ { i } ^ { * } = \frac { \bar { s } _ { - i } ^ { * } + \lambda _ { i } \hat { s } _ { i } } { 1 + \lambda _ { i } }\tag{2}
$$

This system of equations can be rewritten as a linear system:

$$
\left( 1 + \lambda _ { i } \right) s _ { i } ^ { * } - \frac { 1 } { m - 1 } \sum _ { j \neq i } s _ { j } ^ { * } = \lambda _ { i } \hat { s } _ { i } , \quad i = 1 , \ldots , m\tag{3}
$$

which admits a closed-form solution and can be solved directly from the raw scores $\{ \hat { s } _ { i } \}$ without any iterative optimization, learning, or approximation.

A key property of the consensus solution is that it preserves the mean while dampening disagreement: the mean consensus score equals the mean raw score, but individual scores converge as disagreement diminishes proportionally to inter-agent conflict. This property is important because it allows us to separate two distinct failure modes. The first is collective doubt: steps where all agents assign low confidence, resulting in a low mean score. The second is conflicting evidence: steps where the mean confidence is high but agents fundamentally disagree, producing high dispersion. Simple averaging conflates these two cases; for instance, it treats (0.6, 0.6, 0.6) and (0.9, 0.3, 0.6) identically since both yield the same mean. However, the consensus dispersion reveals that the second case reflects unresolved cross-modal instability, while the first reflects genuine (if modest) consensus.

The consensus solution adjusts individual scores but not their collective level: $\begin{array} { r } { \bar { s } ^ { * } = \frac { 1 } { m } \sum _ { i } s _ { i } ^ { * } = \frac { 1 } { m } \sum _ { i } \hat { s } _ { i } } \end{array}$ , a direct consequence of the symmetry in Eq. (3) that holds for any $\{ \lambda _ { i } \}$ . The coupled system cannot inflate confidence. Its efect is confined to redistributing scores around a fixed mean, isolating cross-modal conflict from collective doubt.

Acceptance Criterion and Step Selection. Once consensus scores $\{ s _ { i } ^ { * } \}$ are computed for each candidate, we derive two summary statistics: the mean consensus confidence $\begin{array} { r } { \bar { s } ^ { * } = \frac { 1 } { m } \sum _ { i } s _ { i } ^ { * } } \end{array}$ , measuring collective endorsement of the step, and the consensus dispersion $\begin{array} { r } { \dot { \varDelta ^ { * } } = \frac { 1 } { m } \sum _ { i } \left| s _ { i } ^ { * } - \bar { s } ^ { * } \right| } \end{array}$ , measuring residual disagreement after consensus adjustment.

A candidate step is accepted if and only if it satisfies a dual criterion: the mean confidence must exceed a threshold $\tau$ and the dispersion must fall below a tolerance ϵ:

$$
\mathrm { a c c e p t } ( r _ { t } ^ { ( j ) } ) \iff \bar { s } ^ { * ( j ) } > \tau \land \Delta ^ { * ( j ) } < \epsilon\tag{4}
$$

This dual criterion enables two complementary verification mechanisms. The dispersion check $( \varDelta ^ { \ast } < \epsilon )$ filters candidates where cross-modal evidence fundamentally conflicts, while the confidence check $\left( \bar { s } ^ { * } > \tau \right)$ ensures suficient collective endorsement. Among accepted candidates, the step with the highest $\bar { s } ^ { * }$ is selected to extend the reasoning trace, prioritizing the most stable continuation. When no candidate satisfies both criteria, indicating that all options are suboptimal, the system falls back to continuous ranking by $\bar { s } ^ { * } - \varDelta ^ { * }$ , selecting the candidate that best balances confidence against remaining disagreement. This fallback ensures that the reasoning process can always continue while still preferring more stable steps.

Numerical Illustration. Consider raw scores $\hat { \mathbf { s } } = ( 0 . 9 , 0 . 3 , 0 . 9 )$ from the Visual, Logical, and Contextual agents with $\lambda _ { V } = 1 . 5 , \lambda _ { L } = 1 . 0 , \lambda _ { C } = 0 . 8$ . The linear system in Eq. (3) yields $\mathbf { s } ^ { * } \approx ( 0 . 8 0 , 0 . 5 5 , 0 . 7 7 )$ : the outlying Logical agent is pulled from 0.30 to 0.55, the stubborn Visual agent shifts only 0.10, and the flexible Contextual agent shifts 0.13: asymmetry driven directly by the $\lambda _ { i }$ coupling in Eq. (2). The resulting $\bar { s } ^ { * } \approx 0 . 7 1$ and $\varDelta ^ { * } \approx 0$ .11: the confidence check passes $( 0 . 7 1 > \tau { = } 0 . 6 )$ but dispersion fails $( 0 . 1 1 > \epsilon = 0 . 1 )$ , So the step is rejected as cross-modally unstable. Contrast $\hat { \mathbf { s } } = ( 0 . 7 , 0 . 7 , 0 . 7 )$ : consensus returns $\varDelta ^ { \ast } = 0$ and the step is accepted: same raw mean, opposite decision, a distinction simple averaging cannot make.

Where the consensus formulation adds value. By construction, $\bar { s } ^ { * }$ equals the raw mean: the consensus does not shift collective confidence. Its contribution lies entirely in the individual equilibrium scores $\{ s _ { i } ^ { * } \}$ and the resulting dispersion $\varDelta ^ { \ast }$ . One might argue that since Eq. (2) admits a closed-form solution, one could define the same asymmetric dispersion directly, without motivating the coupled system. This is correct computationally, but it misses what the formulation provides. Defining a weighted dispersion requires choosing how much each agent’s deviation should count: a design decision with no principled grounding. The coupled scoring framework derives these weights from a single interpretable assumption: each agent trades of agreement against fidelity to its own evidence, governed by stubbornness $\lambda _ { i } .$ . This is precisely the assumption that defines a coordination game. The fixed point uniquely determines the rest.

Crucially, the consensus solution introduces coupling that no per-agent measure can replicate. Because agent i’s adjusted score $s _ { i } ^ { * }$ depends on all other agents raw scores through the linear system in Eq. (3), its residual $| s _ { i } ^ { * } - \bar { s } ^ { * } |$ reflects not just its own deviation but how that deviation interacts with where other agents land. We formalize this below.

Proposition 1 (Consensus Dispersion is Not Recoverable from Weighted Averages). For any fixed weight vector w, there exist score vectors $\hat { \mathbf { s } } \neq \hat { \mathbf { s } } ^ { \prime }$ such that $\mathbf { w } ^ { \top } \hat { \mathbf { s } } = \mathbf { w } ^ { \top } \hat { \mathbf { s } } ^ { \prime }$ yet $\varDelta ^ { \ast } ( \hat { \mathbf { s } } ) \neq \varDelta ^ { \ast } ( \hat { \mathbf { s } } ^ { \prime } )$ . As for any agent $i ,$ the consensus residual $| s _ { i } ^ { * } - \bar { s } ^ { * } |$ depends on the full raw score vector $\hat { \bf s } = ( \hat { s } _ { 1 } , \dots , \hat { s } _ { m } )$ , not solely on $\hat { s } _ { i }$ . Consequently, the consensus dispersion $\begin{array} { r } { \varDelta ^ { * } = \frac { 1 } { m } \sum _ { i } | s _ { i } ^ { * } - \bar { s } ^ { * } | } \end{array}$ cannot be decomposed as a separable function $\textstyle \sum _ { i } f _ { i } ( { \hat { s } } _ { i } )$ for any choice of per agent functions $\{ f _ { i } \}$

Proof. From Eq. (2), s<sup>∗</sup> depends on $\bar { s } _ { - i } ^ { * }$ , which itself depends on all agents’ raw scores through the coupled linear system in Eq. (3). Consequently, $\partial s _ { i } ^ { * } / \partial \hat { s } _ { j } \neq 0$ for $i \neq j$ , so the residual $| s _ { i } ^ { * } - \bar { s } ^ { * } |$ is not a function of $\hat { s } _ { i }$ alone and $\varDelta ^ { \ast }$ is not separable. As a concrete illustration: $\hat { \mathbf { s } } = ( 0 . 9 , 0 . 2 , 0 . 9 )$ and $\hat { \mathbf { s } } ^ { \prime } = ( 0 . 7 , 0 . 6 , 0 . 7 )$ share the same mean $\textstyle { \frac { 2 } { 3 } }$ , yet the consensus solution yields $\varDelta ^ { * } ( \hat { \mathbf { s } } ) \approx 0 . 1 3 > \epsilon$ and $\varDelta ^ { * } ( \hat { \mathbf { s } } ^ { \prime } ) \approx 0 . 0 4 < \epsilon .$ , producing opposite acceptance decisions under $\mathrm { E q . \ ( 4 ) }$ , a distinction no separable measure can replicate, since the diference arises from the consensus solution coupling agent 2’s residual to agents 1 and 3’s raw scores.

Two properties follow. First, disagreement from a high-λ agent weighs more heavily in $\varDelta ^ { \ast }$ than equivalent disagreement from a flexible one, an asymmetry that raw variance treats identically. Second, this coupling is what lets the dual criterion in Eq. (4) distinguish genuine cross-modal conflict from ordinary confidence fluctuations, even when two score vectors share identical means and per-agent deviations.

Game-theoretic interpretation: The coupled scoring formulation admits a natural reading as a coordination game in which each agent is a player, its reported score is a strategy, and . eq. (1) defines the payof. The fixed point in eq. (2) is then the unique Nash equilibrium, guaranteed by Rosen’s theorem for concave n-person games [32]. This connection is more than cosmetic: it supplies a principled rationale for why the stubbornness parameters $\lambda _ { i }$ govern the influence structure, as each agent trades of fidelity to its own evidence against pressure to coordinate exactly the tension a coordination game formalizes. The analogy has limits: our agents score in isolation and never observe one another; the equilibrium is computed in closed form from one-shot scores rather than reached

Table 2: Accuracy (%) across multimodal reasoning benchmarks under diferent verification strategies. Avg. is the unweighted mean across all six benchmarks. Values in blue indicate improvement of our method relative to the Base Model.  
Our detailed evaluation setup is give<sub>Dataset 3DSRBench CV-Bench-3D</sub> Dataset n in the supplementary.  <sub>CV-Bench-2D</sub> <sub>BLINK</sub> <sub>MMStar</sub>
<table><tr><td>Dataset</td><td></td><td></td><td>3DSRBench CV-Bench-3D CV-Bench-2DFFBLINK</td><td></td><td>MMStar</td><td>AI2D</td><td></td><td>Avg.</td></tr><tr><td>Base</td><td>56.12</td><td>76.39</td><td>74.27</td><td>48.31</td><td>61.25</td><td></td><td>81.52</td><td>66.31</td></tr><tr><td>Domain-Specific Critics</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLaVA Critic</td><td>52.71 ±0.83</td><td>81.58 ±0.91</td><td>67.52 ±0.82</td><td>49.22 ±0.77</td><td>64.07 ±0.61</td><td></td><td>81.81 ±0.44</td><td>66.15</td></tr><tr><td>Critic-VcvPR, 25</td><td>53.25 ±0.73</td><td>77.66 ±0.81</td><td>75.38 ±0.79</td><td>46.17 ±0.86</td><td>55.83 ±0.59</td><td></td><td>80.17 ±0.75</td><td>64.74</td></tr><tr><td>SherlockNeurIPS, 25</td><td>48.11 ±0.93</td><td>58.13 ±1.10</td><td>68.78 ±1.98</td><td>49.07 ±0.89</td><td>57.26</td><td>±0.96</td><td>82.77 ±0.52</td><td>60.69</td></tr><tr><td>VisionSR1ICLR, 26</td><td>53.05 ±1.82</td><td>54.18 ±1.05</td><td>73.10 ±1.91</td><td>30.09 ±1.45</td><td>57.20 ±2.73</td><td></td><td>80.25 ±0.97</td><td>57.98</td></tr><tr><td>DreamPRMNeurIPS, 25</td><td>53.08 ±1.88</td><td>63.39 ±0.98</td><td>75.58 ±1.61</td><td>49.89 ±0.82</td><td>61.23 ±0.59</td><td></td><td>81.21 ±0.49</td><td>64.06</td></tr><tr><td colspan="2">Domain-Agnostic Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Variance</td><td>57.21 ±0.57</td><td>78.16 ±0.68</td><td>76.43 ±0.53</td><td></td><td>49.61 ±0.49</td><td>63.09 ±0.47</td><td>81.41 ±0.34</td><td>67.65</td></tr><tr><td>Mean</td><td>58.34 ±0.53</td><td>79.77 ±0.61</td><td>77.57 ±0.51</td><td>50.17 ±0.58</td><td>64.14 ±0.44</td><td></td><td>82.18 ±0.36</td><td>68.70</td></tr><tr><td>Min</td><td>56.11 ±0.58</td><td>77.21 ±0.63</td><td>75.17 ±0.59</td><td>49.39 ±0.55</td><td>62.41 ±0.46</td><td></td><td>81.09 ±0.42</td><td>66.90</td></tr><tr><td>Majority</td><td>57.58 ±0.62</td><td>77.37 ±0.73</td><td>75.53 ±0.68</td><td>48.43 ±0.66</td><td></td><td>62.10 ±0.52</td><td>81.78 ±0.48</td><td>67.13</td></tr><tr><td>Max</td><td>57.37 7 ±0.55</td><td>78.16 ±0.56</td><td>76.46 ±0.63</td><td>49.17 ±0.43</td><td></td><td>63.42 ±0.58</td><td>82.12 ±0.54</td><td>67.78</td></tr><tr><td>VERDICT (Ours)</td><td>59.02 ±0.45</td><td>82.34 ±0.51</td><td>79.22 ±0.53</td><td>51.32 ±0.48</td><td></td><td>65.88 ±0.37</td><td>83.14 ±0.32</td><td>70.15</td></tr><tr><td>Gains</td><td>+2.90</td><td>+5.95</td><td>+4.95</td><td>+3.01</td><td></td><td>+4.63</td><td>+1.62</td><td>+3.84</td></tr></table>

through iterated play, and should be understood as interpretive scafolding that motivates the formulation rather than a claim about strategic interaction.

## 5 Experiments and Results

Experimental Setup. We evaluate our verification framework on six benchmarks that collectively assess diferent aspects of visual and multimodal reasoning. 3DSRBench [29], CV-Bench [40] split into CV-Bench-2D (spatial relationships and object counting) and CV-Bench-3D (depth ordering and relative distance). BLINK [15], MMStar [4], AI2D [19].

Our approach employs three verification agents: Visual (V), Logical (L), and Contextual (C), all built on Qwen2.5-VL-7B-Instruct [1], the same model used as the base reasoner. At each reasoning step, the base model generates n=3 candidate continuations via temperature sampling $( T { = } 0 . 8 , \ \mathrm { t o p } { - } p { = } 0 . 6 )$ . Each agent disparately scores every candidate, and consensus scores are computed using agent-specific stubbornness parameters $( \lambda _ { V } { = } 1 . 5 , \lambda _ { L } { = } 1 . 0 , \lambda _ { C } { = } 0 . 8 )$ . A candidate is accepted if its consensus dispersion satisfies $\varDelta ^ { * } < \epsilon { = } 0 . 1$ and mean confidence satisfies $\bar { s } ^ { * } > \tau \mathrm { = } 0 . 6$ . Among accepted steps, we select the one with the highest s¯<sup>∗</sup>. These hyperparameters are fixed across all six benchmarks with no taskspecific tuning. The final answer is extracted from the complete reasoning trace using (Gemma-3:12B via Ollama). The use of Ollama was only for answer extraction, and is not for any part of the verification pipeline. All agent prompts are provided in the supplementary.

Baselines. We compare VERDICT against two categories of baselines. The first category consists of domain-specific critics(so called because each requires labeled data from the target domain to train): LLaVA-Critic [46], Sherlock [9], VisionSR1 [24], DreamPRM [3], Critic-V [49] These represent the current state of the art in step-level verification, but each requires task-specific training data or learned reward signals. All domain-specific critics are evaluated under comparable conditions. The second category consists of domain-agnostic baselines(which require no domain-specific training data, instead relying on modality-specialized agents with task-general rubrics) that use the same three frozen agents as our method but difer in how they combine the raw scores. These include: Mean (select the candidate with the highest average raw score), Max (select by highest maximum score across agents), Min (select by highest minimum score), Majority (accept candidates where a majority of agents score above 0.5, then rank by mean), and Variance (reject candidates with high score variance, then rank by mean among the rest). These baselines isolate the contribution of our consensus formulation by controlling for the agent architecture and scoring procedure.

table 2 presents accuracy across six benchmarks under all verification strategies.   
We organize the discussion around three findings.

VERDICT consistently improves over the base model, without degradation on any benchmark. Our method improves over the unverified base model on every benchmark, with gains ranging from +2.90 on 3DSRBench to +5.95 on CV-Bench-3D. This consistency holds across qualitatively diferent tasks: spatial reasoning (3DSRBench, CV-Bench-3D), visual grounding (CV-Bench-2D, AI2D), and multimodal abstraction (BLINK, MMStar).

Domain-specific critics show cross-task fragility.Every domain-specific critic we evaluate degrades below baseline on at least two benchmarks. Sherlock [9] drops 18.26 points on CV-Bench-3D and 8.01 points on 3DSRBench. VisionSR1 [24] collapses by 22.21 points on CV-Bench-3D and 18.22 points on BLINK. Even LLaVA-Critic [46], which achieves the highest single-benchmark score among domain-specific methods on CV-Bench-3D (81.58), simultaneously degrades by 6.75 points on CV-Bench-2D, two splits of the same benchmark family. This pattern is consistent with capability tensions documented in prior work [17,35,50]: a verifier trained to reward one reasoning style can actively penalize another. Our method sidesteps this entirely by aggregating frozen, modality-specialized judgments through consensus computation rather than learn ing a single score function that must generalize across heterogeneous evaluation criteria.

VERDICT adds value beyond domain-agnostic aggregation. To isolate the contribution of our consensus formulation, we compare against five aggregation baselines that use identical agents and scoring procedures. The only difference is how raw scores are combined into a selection decision. Among these, Mean scoring emerges as the strongest simple baseline, followed by Max and Variance. However, VERDICT consistently outperforms Mean by +0.68 to +2.57 across all benchmarks. This margin confirms that the disagreement structure captured by consensus computation, specifically, the coupling between agents introduced through the coupled scoring framework, provides a verification signal that score averaging discards. It is worth noting that the Variance baseline, which explicitly attempts to incorporate disagreement by rejecting high-variance candidates, still underperforms our method. This indicates that treating all variance equally, without distinguishing genuine cross-modal conflict from ordinary confidence fluctuations, is insuficient (a finding consistent with Proposition 1).

![](images/ea2b3aaaffb06fdfd9fb151ee00dd88296ed63759804e4c0517eeee64eb3dba3.jpg)  
Fig. 3: VERDICT achieves the highest accuracy on five of six benchmarks, with each component (rejection and selection) contributing disparately.

![](images/52cc5adb9cc00e48d22e60a8ace2821b641d6978449cdd397b301e1c4e4fa29d.jpg)  
Fig. 4: Stubbornness parameter sensitivity on 3DSRBench. Each curve varies one parameter while holding the others fixed at our configuration. Open markers indicate our chosen values.

## 6 Discussion and Analysis

We conduct a series of ablation studies to understand the contribution of each component in our framework.

## 6.1 Rejection vs. Selection

To understand whether improvements come primarily from filtering unreliable candidates or from better ranking among viable options, we decompose our method into its constituent mechanisms. We evaluate five strategies: VER-DICT (our complete method: filter by acceptance criterion, then rank by ${ \bar { s } } ^ { * } )$ No Rejection (skip filtering, rank all candidates by $\bar { s } ^ { * } - \varDelta ^ { * } )$ , No Selection (keep filtering, randomly select among accepted), Raw Average (replace consensus scores with simple means and dispersions, keeping the same dualcriterion structure), and Random (no filtering, no ranking). fig. 3 presents the results. Three findings emerge: First, rejection and selection contribute differently: removing either degrades accuracy, but removing intelligent ranking (No Selection, −1.17 to −2.17) costs more than removing filtering (No Rejection, −0.26 to −1.08), indicating that consensus-adjusted ranking is the primary driver. Second, Raw Average consistently underperforms VERDICT by 2.59- 2.95 points despite using the same dual-criterion architecture, confirming that the coupled scoring framework produces better-calibrated scores than naive aggregation independent of the filtering and ranking structure built on top. Third, Random selection performs only marginally above the base model, establishing that verification-aware scoring, not candidate diversity alone drives the gains.

## 6.2 Stubbornness Sensitivity

We ablate each stubbornness parameter separately on 3DSRBench, varying it over {0.3, 0.8, 1.0, 1.5, 2.0} while holding the other two fixed. fig. 4 reveals a clear sensitivity hierarchy: $\lambda _ { V }$ has the widest performance range (2.90 percentage points), $\lambda _ { L }$ an intermediate range (1.40), and $\lambda _ { C }$ the narrowest (1.19). This ordering directly mirrors our asymmetric design where the parameter with the most impact on spatial reasoning is set highest, while the least impactful one is given the most flexibility. Critically, no setting degrades below the base model, and the three curves converge at their respective optimal values to the same peak accuracy (59.02%), confirming that the chosen configuration sits at a jointly optimal point rather than a compromise. The range of fig. $4 \lambda \in [ 0 . 3 , 2 . 0 ]$ spans the full coupling transition: at $\lambda = 0 . 3$ an agent places 77% weight on the group mean ( eq. (2)), making it nearly consensus-driven; at $\lambda = 2 . 0$ it places 67% on its own evidence, yet the remaining 33% consensus pull still preserves meaningful inter-agent coupling which is the defining property of the framework.

![](images/14fe7240b1c566b75d2f2b0bc58ea9fe9d72f82d3292f813e0f0b0c276046fd6.jpg)  
Fig. 5: Confidence threshold sensitivity on 3DSRBench with $\epsilon = 0 . 1$ fixed. Performance follows a descending gradient.

![](images/3172512a14f878f7489b0b4ed20db744176307f541242c5f40fdf2091d7e4514.jpg)  
Fig. 6: Dispersion Tolerance Sensitivity on 3DSRBench with $\tau = 0 . 6$ fixed.

Stubbornness Assignment. The per-parameter sensitivity above establishes which parameter matters most; it does not test whether the assignment of values to agents matters. We permute the stubbornness triple $( \lambda _ { V } = 1 . 5 , ~ \lambda _ { L } = 1 . 0 , ~ \lambda _ { C } = 0 . 8 )$ via pairwise swaps on 3DSRBench. Swapping L↔C (λ<sub>V</sub> unchanged) costs only −0.41 points (58.61%), while swapping V↔L costs −1.19 (57.83%) and $\mathrm { V } {  } \mathrm { C }$ costs −1.76 (57.26%). The degradation is proportional to the reduction in the visual agent’s stubbornness. As long as λ retains the highest value, VERDICT still exceeds the Mean baseline $( 5 8 . 3 4 \% ) ;$ ; reducing it below either other agent drops performance below Mean. The assignment is not arbitrary: on perceptionheavy tasks, visual evidence is the least negotiable signal, and the consensus formulation amplifies this only when the stubbornness structure reflects it.

## 6.3 Threshold Sensitivity

Confidence threshold τ. We ablate the confidence threshold τ while holding the dispersion tolerance fixed at $\epsilon = 0 . 1$ on 3DSRBench, and fig. 5 present results across $\tau \in \{ 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 0 . 6 , 1 . 0 , 1 0 . 0 \}$ , extended analysis across operating regimes is provided in the supplementary.For τ in fig. 5, performance follows a descending gradient across three regimes: permissive $( \tau \leq 0 . 0 1$ , accuracy 65–69%), intermediate $( \tau = 0 . 1 \ – 0 . 6$ , around 60–61%), and restrictive $( \tau \geq 1 . 0$ ∼60.6% via fallback ranking). The permissive regime is the most revealing: with the confidence filter efectively disabled, the consensus-adjusted ranking alone substantially exceeds the Mean baseline (58.34%), confirming that the consensus formulation itself, not the filtering architecture, drives the improvement.

Cross-benchmark threshold analysis. We extend the τ sensitivity analysis across all six benchmarks (full results in supplementary). The permissive regime $( \tau \leq 0 . 0 1 )$ achieves higher accuracy on 3DSRBench (68.71% vs. 59.02%), but $\tau = 0 . 6$ is individually optimal on the remaining five benchmarks.

Dispersion tolerance ϵ. We ablate the dispersion tolerance while holding $\tau =$ 0.6 fixed. fig. 6 presents accuracy across $\epsilon \in \{ 0 . 0 0 1 , 0 . 0 5 , 0 . 1 , 0 . 5 , 1 . 0 , 2 . 5 , 3 . 0 \}$ For ϵ in fig. 6 , performance peaks in an optimal plateau at $\epsilon \in \ [ 0 . 1 , 1 . 0 ]$ with accuracy around 60–61%. Too strict $( \epsilon < 0 . 1 )$ rejects legitimate steps with natural cross-modal tension; too permissive $( \epsilon > 1 . 0 )$ admits candidates where judges have not genuinely converged. The decline is gradual in both directions because the fallback mechanism $( \bar { s } ^ { * } - \varDelta ^ { * }$ ranking) and the confidence threshold $\left( \bar { s } ^ { * } > \tau \right)$ respectively provide safety nets, preventing catastrophic failure at any operating point. <sup>3</sup>

## 6.4 Disentangling Judge Quality from Algorithmic Gain

We replace the 7B verification agents with 4B and 2B variants while holding the base reasoner fixed, isolating scoring quality from candidate quality. fig. 7 reports results on 3DSRBench. Absolute accuracy degrades monotonically with judge capability for both VERDICT and Mean, as expected. However, the margin between VERDICT and Mean widens from +0.68 at 7B to +0.89 at 4B to +1.21 at 2B, consistent with Proposition 1: the coupled scoring system surfaces noise-induced disagreement through the dispersion filter rather than passing it to the selection decision. At the 2B scale, Mean drops 1.65 points below the unverified baseline while VERDICT with the same judges limits this degradation to

![](images/09a2af6e32f0a3dcf85d3cc9f973db5c093130cbff416e9f4144383169897094.jpg)  
Fig. 7: Disentangling judge quality from algorithmic gain on 3DSR-Bench.Both methods degrade with weaker judges, but the margin between VERDICT and Mean widens.

0.44 points, a 1.21 point recovery attributable entirely to the consensus formulation, confirming that the reported gains reflect algorithmic structure rather than judge quality.Additional results can be found in the supplementary materials.

## 7 Conclusion

Multimodal reasoning in MLLMs fails gradually as locally plausible steps drift toward incorrect answers. Our results across six benchmarks show that this drift leaves a detectable trace in the structure of cross-modal disagreement. VER-DICT formalizes this signal through disagreement-aware consensus computation, achieving consistent improvements without task-specific tuning. Our key takeaway is that when disparate judges disagree about a reasoning step, the pattern of that disagreement is itself a verification signal that principled aggregation can exploit. The framework has certain limitations. When all agents are confidently wrong, consensus converges on an incorrect assessment; and no filtering recovers from a base model that produces no viable candidates. Characterizing how often this failure mode occurs in practice is deferred to future work. On the practical side, VERDICT is fully parallelizable and dominated by generation cost, scoring requires 1–5 tokens per step versus 50–200 per reasoning step, yielding 3.80× wall-clock overhead under sequential execution. This is non-trivial but fundamentally cheaper than regenerating entire trajectories, and reducible via adaptive verification at high-uncertainty steps, a direction we leave to future work. Per-agent scores further provide a modality-level diagnostic trace pinpointing failures in visual grounding, logical coherence, or contextual relevance. Unlike domain-specific critics, this overhead is the entire cost. Because the coordination-game formulation requires only that heterogeneous evaluators trade of private evidence against consensus, VERDICT generalizes beyond multimodal reasoning to reward model ensembles [12,34], multi-agent evaluation, and compositional code verification.

## Table of Contents

– S1. Hyperparameters

– S2. Implementation Details: VERDICT

– S3. Detailed Experimental Setup

– S4. Additional Results

• S4.1 Comparison with Test-Time Scaling Methods

• S4.2 Cross-Benchmark Dispersion Tolerance Sensitivity

• S4.3 Cross-Benchmark Confidence Threshold Analysis

• S4.4 Threshold Sensitivity: Extended Analysis

• S4.5 Joint Threshold Interaction Analysis

• S4.6 Cross-Benchmark Judge Scale Analysis

• S4.7 Cross-Benchmark Stubbornness Sensitivity

• S4.8 Cross-Benchmark Stubbornness Assignment Analysis

• S4.9 Dispersion as a Diagnostic Signal

• S4.10 Confidence–Dispersion Landscape

• S4.11 Rejection and Selection: Detailed Analysis

• S4.12 Fallback Mechanism Analysis

• S4.13 Generalization Across Model Families

– S5 Detailed Computational Cost Analysis

S6 Closed-Form Derivation

• S6.1 Matrix Construction

• S6.2 Invertibility: Formal Proof

• S6.3 Scalar Reduction and Explicit Solution

• S6.4 Stubbornness-Weighted Sum Invariant

• S6.5 Extension to General m

• S6.6 Numerical Verification

– S7 Existence and Uniqueness of the Consensus Fixed Point

– S8 Proposition 1: Detailed Proof and Worked Examples

– S9 Consensus Dispersion vs. Weighted Averaging: Illustrative Scenarios

– S10 Connections to Social Choice Theory and Belief Aggregation

– S11 Qualitative Samples

– S12 Prompt Templates

## Supplementary at a Glance: Key Takeaways

This supplement provides extended derivations, ablations, and analyses supporting the main paper. We highlight the four most important findings:

1. Cross-benchmark robustness of the default configuration. All hyperparameters $( \lambda _ { V } = 1 . 5 , ~ \lambda _ { L } = 1 . 0 , ~ \lambda _ { C } = 0 . 8 , ~ \tau = 0 . 6 , ~ \epsilon = 0 . 1 )$ are fixed across all six benchmarks with no per-task tuning. The dispersion tolerance ϵ exhibits a stable optimal plateau spanning a full order of magnitude $( \epsilon \in [ 0 . 1 , 1 . 0 ] )$ , and the confidence threshold τ=0.6 is individually optimal on five of six benchmarks. The joint threshold sweep (49 configurations) confirms that the operating point lies on a plateau, not an edge.

2. Consensus adds value beyond aggregation, especially with weaker judges. The margin between VERDICT and Mean aggregation widens as the judge scale decreases from 7B to 2B on five of six benchmarks. At 2B scale, VERDICT recovers 1.2–4.6 points over na¨ıve averaging. This confirms that the gains are attributable to the consensus formulation’s algorithmic structure, not to judge quality alone.

3. Dispersion is a genuine, non-separable diagnostic signal. Consensus dispersion ∆<sup>∗</sup> is enriched 1.4–1.6× for incorrect chains (Section S4.9); ROC analysis yields AUC = 0.65–0.66. The signal is moderate and diagnostic rather than oracular, which explains why VERDICT benefits from soft integration (ranking + dual filtering) rather than hard binary classification.

4. The framework generalises across model families at zero adaptation cost. Applying VERDICT to four diferent base model families yields consistent gains of +2.45 to +4.00 points. The consensus compresses base-model variability from a 1.33-point range to 0.22 points, confirming plug-in generality.

## 8 Hyperparameters

<table><tr><td>Hyperparameter Value</td></tr><tr><td>λv 1.5</td></tr><tr><td>λL 1.0</td></tr><tr><td>λc 0.8</td></tr><tr><td>n 3</td></tr><tr><td>T 0.6</td></tr><tr><td>€ 0.1</td></tr></table>

Table 3: Hyperparameters used in all experiments.

In our experiments, we used hyperparameters as shown in Table 3. This configuration reflects the intuition that the visual verifier (V) should be most resistant to consensus pressure on perception-heavy steps, while the contextual verifier

(C) may be more flexible when visual or linguistic evidence is strong. These values are fixed across all datasets and require no tuning.

## 9 Implementation Details: VERDICT

Here, we discuss how the VERDICT’s consensus-based verification is implemented in practice.

At each reasoning step, the base MLLM proposes several candidate continuations (typically 3 candidates per step). For every candidate, three frozen verifier agents are queried independently: a visual grounding agent (V), a logical consistency agent (L), and a contextual reasoning agent (C). Each produces a scalar confidence score in [0, 1], reflecting its modality-specific assessment of each step. If one verifier is confident while the others are uncertain, averaging may obscure this inconsistency. We want a mechanism that surfaces disagreement explicitly and allows agents to partially adjust their beliefs toward consensus without forcing uniformity when genuine divergence exists.

The consensus formulation captures this trade-of naturally. Each verifier prefers agreement with others, but not at the cost of abandoning its own judgment entirely. This is encoded through a quadratic scoring objective, which admits a unique, closed-form. The consensus scores thus represent a principled resolution of inter-agent tension, rather than an ad-hoc blend.

## Consensus formulation and computation

The implementation uses a heterogeneous trade-of parameter formulation, where diferent verifiers are assigned diferent sensitivities to consensus pressure. This is motivated by the observation that certain verifiers should be more resistant to group influence depending on the nature of the reasoning step.

Each agent’s consensus score $s _ { i } ^ { * }$ satisfies:

$$
s _ { i } ^ { * } = \frac { \bar { s } _ { - i } ^ { * } + \lambda _ { i } \hat { s } _ { i } } { 1 + \lambda _ { i } }\tag{5}
$$

where $\hat { s } _ { i }$ is agent i’s raw confidence score, $\begin{array} { r } { \bar { s } _ { - i } ^ { * } = \frac { 1 } { n - 1 } \sum _ { j \neq i } { s } _ { j } ^ { * } } \end{array}$ is the mean consensus score of all other agents, and $\lambda _ { i } > 0$ controls how strongly agent i weights its own judgment relative to group consensus.

This system of equations can be rewritten as a linear system and solved exactly:

$$
\left( 1 + \lambda _ { i } \right) s _ { i } ^ { * } - { \frac { 1 } { n - 1 } } \sum _ { j \neq i } s _ { j } ^ { * } = \lambda _ { i } { \hat { s } } _ { i } , \quad i = 1 , \ldots , n\tag{6}
$$

The system is solved using a standard linear solver. The resulting matrix is typically well-conditioned when $\lambda _ { i } > 0$ , but if numerical issues arise, the implementation falls back to the raw verifier scores. All consensus scores are clipped to [0, 1] to maintain valid confidence values.

Algorithm 1 VERDICT: Step-Wise Consensus Verification   
Require: Base model $\mathcal { M } _ { \mathrm { b a s e } } .$ Verifier agents $\{ \nu , \mathcal { L } , \mathcal { C } \}$ , Image I, Question Q   
Require: Number of candidates $n = 3 ,$ Stubbornness parameters $\{ \lambda _ { V } = 1 . 5 , \lambda _ { L } =$   
$1 . 0 , \lambda _ { C } = 0 . 8 \}$   
Require: Acceptance thresholds: dispersion $\epsilon = 0 . 1$ , confidence $\tau = 0 . 6$   
Ensure: Complete reasoning trace $r _ { 1 : T }$ and final answer   
1: Initialize reasoning trace $r \gets \emptyset$   
2: Generate initial step: $r _ { 0 } \sim \mathcal { M } _ { \mathrm { b a s e } } ( I , Q )$   
3: Append $r _ { 0 }$ to r   
4: while r does not contain end-of-sequence token do   
5: Candidates $ \emptyset$   
6: RawScores $ \emptyset$   
7: for $i = 1$ to n do   
8: Generate candidate: $c _ { i } \sim \mathcal { M } _ { \mathrm { b a s e } } ( I , Q , r )$ with temperature $T = 0 . 8$   
9: Query Visual Agent: $\hat { s } _ { V } ^ { ( i ) }  \mathcal { V } ( I , Q , r , c _ { i } )$ {Score $\in [ 0 , 1 ] \}$   
10: Query Logical Agent: $\hat { s } _ { L } ^ { ( i ) }  \mathcal { L } ( I , Q , r , c _ { i } )$   
11: Query Contextual Agent: $\hat { s } _ { C } ^ { ( i ) }  \mathcal { C } ( I , Q , r , c _ { i } )$   
12: Store: $\mathrm { C A N D I D A T E S } [ i ]  c _ { i } , \mathrm { R A W S C O R E S } [ i ]  \{ \hat { s } _ { V } ^ { ( i ) } , \hat { s } _ { L } ^ { ( i ) } , \hat { s } _ { C } ^ { ( i ) } \}$   
13: end for   
14: {Consensus Computation & Selection}   
15: Results $ \emptyset$   
16: for each candidate i in $\{ 1 , \ldots , n \}$ do   
17: Solve $\mathbf { s } ^ { * ( i ) }$ per $\mathrm { E q . ~ } 6$   
18: Compute mean: $\begin{array} { r } { \bar { s } ^ { * ( i ) }  \frac { 1 } { 3 } \sum _ { j \in \{ V , L , C \} } { s _ { j } ^ { * ( i ) } } } \end{array}$   
19: Compute dispersion: $\begin{array} { r } { \varDelta ^ { ( i ) } \gets \frac { 1 } { 3 } \sum _ { j \in \{ V , L , C \} } | s _ { j } ^ { * ( i ) } - \bar { s } ^ { * ( i ) } | } \end{array}$   
20: Check acceptance: accepte $\mathfrak { I } ^ { ( i ) }  ( \varDelta ^ { ( i ) } < \epsilon ) \wedge ( \bar { s } ^ { * ( i ) } > \tau )$   
21: Store: Results[i] $ \{ \mathbf { s } ^ { \bar { * } ( i ) } , \bar { s } ^ { * ( i ) } , \dot { \Delta } ^ { ( i ) }$ , accepted<sup>(i)</sup>}   
22: end for   
23: ValidSteps $ \{ i$ : Results[i].accepted = True}   
24: if ValidSteps $\neq \emptyset$ then   
25: $i ^ { * } \gets$ arg max<sub>i∈ValidSteps</sub> $\overline { { s } } ^ { * ( i ) }$ {Highest confidence among stable steps Normal   
Model}   
26: else   
27: $i ^ { * } \gets \arg \operatorname* { m a x } _ { i = 1 , \dots , n } ( \bar { s } ^ { * ( i ) } - \Delta ^ { ( i ) } )$ {Best balance if none accepted Fallback   
Model}   
28: end if   
29: Append selected step: $r  r \cup$ {Candidates $[ i ^ { * } ] \}$   
30: end while   
31: Extract final answer from $r$   
32: return $r ,$ final answer

In our experiments, we set $n = 3 \ \lambda _ { \mathrm { V } } = 1 . 5 , \lambda _ { \mathrm { L } } = 1 . 0 $ , and $\lambda _ { \mathrm { C } } = 0 . 8 .$ . This configuration reflects the intuition that the visual verifier (V) should be most resistant to consensus pressure on perception-heavy steps, while the contextual verifier (C) may be more flexible when visual or linguistic evidence is strong. These values are fixed across all datasets and require no tuning.

Importantly, no iterative optimization, learning, or approximation is used. The consensus solution is computed via a direct linear solve at every reasoning step, adding negligible overhead compared with the cost of querying the verifier models themselves.

## Step selection via consensus statistics

Once consensus scores are obtained, the system computes two summary statistics:

– Mean consensus confidence $\begin{array} { r } { \bar { s } ^ { * } = \frac { 1 } { n } \sum _ { i } s _ { i } ^ { * } \bar { { s } } _ { i } ^ { * } . } \end{array}$ : measures collective endorsement of a step.

– Consensus dispersion $\begin{array} { r } { \varDelta = \frac { 1 } { n } \sum _ { i } \left| s _ { i } ^ { * } - \bar { s } ^ { * } \right| } \end{array}$ : measures residual disagreement after consensus adjustment.

Candidate steps are accepted only if they simultaneously achieve high collective confidence $\left( \bar { s } ^ { * } > \tau \right)$ and low inter-agent dispersion $( \varDelta < \epsilon )$ . In our experiments, we set $\tau = 0 . 6$ and $\epsilon = 0 . 1$ . This dual criterion is stricter than confidence alone: a step with high average confidence but high dispersion signals unresolved conflict and is rejected.

When multiple candidate steps are evaluated at the same reasoning position, rejected steps are discarded immediately. Among accepted steps, the one with the highest $\bar { s } ^ { * }$ is selected to extend the reasoning trace. If no candidate step is accepted $( { \mathrm { i . e . } }$ , all steps have either low confidence or high dispersion), the implementation selects the step that maximizes $\bar { s } ^ { * } - \varDelta$ , prioritizing the best available balance between confidence and agreement. This fallback ensures the reasoning process can continue even when all candidates are suboptimal, while still preferring more stable steps.

## Why this matters

VERDICT serves as a lightweight consensus protocol over frozen verifiers. It makes disagreement explicit and actionable, rather than burying it in an average. It requires no training or calibration beyond setting three hyperparameters $( \lambda _ { i }$ values, τ, and ϵ), all of which remain fixed across datasets, and integrates naturally into step-wise reasoning by filtering unstable steps before they can compound downstream errors.

Crucially, the consensus is not a heuristic approximation. It is the exact fixed point of a well-defined coupled scoring system. This gives the filtering process a principled justification and makes the system’s behavior more interpretable: rejected steps are precisely those where verifiers could not reach a stable agreement, even after accounting for consensus pressure. The heterogeneous $\lambda _ { i }$ values allow the system to implicitly adapt to diferent reasoning regimes without explicit step-type classification.

## 10 Detailed Experimental Setup

Base Model Configuration We use Qwen2.5-VL-7B-Instruct as our primary reasoning model. The model generates candidate reasoning steps through chainof-thought prompting with the instruction: “[Question]. Reason step by step.” We employ temperature sampling with T = 0.8 and top-p = 0.6 to encourage diversity among the three candidate continuations generated at each step. Generation stops when the model produces either an end-of-sequence token or a newline character, with a maximum of 1000 new tokens per step.

Verification Agent Architecture All three verification agents are implemented using Qwen2.5-VL-7B-Instruct as the backbone model. Each agent receives the same visual input (the question image) along with the question text, previous reasoning steps (assumed correct), and the current candidate step to verify. Agents operate independently and output scores in [0, 1].

Visual Agent (V) The Visual Agent evaluates whether objects and spatial relationships mentioned in the reasoning step are visually verifiable. The agent is specifically instructed to maintain a balance between strictness and commonsense spatial reasoning, and to output only a single number between 0.0 and 1.0.

Logical Agent (L) The Logical Agent assesses whether the reasoning step follows logically from previous steps and makes progress toward answering the question. It evaluates:

Whether the step builds coherently on facts

– Whether logical inferences are valid

– Whether the step moves closer to resolving the question

Contextual Agent (C) The Contextual Agent determines whether the step maintains focus on the original question and avoids introducing irrelevant or tangential information. It penalizes steps that:

– Describe details unrelated to the question

– Make definitive claims about obscured or cropped-out objects – Introduce unnecessary speculation

## 10.1 Dataset and Evaluation

We evaluate our approach across six diverse vision-language benchmarks that test diferent aspects of multimodal reasoning:

3DSRBench is a comprehensive 3D spatial reasoning benchmark comprising 2,772 manually annotated visual question-answer pairs across 12 question types. The benchmark evaluates four main categories of 3D awareness: height, location, orientation, and multi-object reasoning. It includes questions based on both natural images from MS-COCO and multi-view synthetic images, with particular emphasis on testing robustness across common and uncommon camera viewpoints. The benchmark employs careful design to avoid trivial answers and uses novel evaluation strategies like FlipEval to ensure robust assessment.

CV-Bench addresses vision-centric evaluation through 2,638 manually inspected examples repurposed from standard vision benchmarks, including ADE20k, COCO, and OMNI3D. The benchmark is divided into two components: CV-Bench-2D evaluates spatial relationships and object counting, while CV-Bench-3D assesses depth order and relative distance understanding. By formulating natural language questions that probe fundamental visual understanding, the benchmark tests whether models can perform classic computer vision tasks within a multimodal context.

BLINK focuses on core visual perception abilities that humans can solve “within a blink”, reformatting 14 classic computer vision tasks into 3,807 multiplechoice questions. The benchmark spans pixel-level to image-level perception tasks, including relative depth estimation, visual correspondence, forensics detection, multi-view reasoning, and visual similarity. A key feature is the incorporation of diverse visual prompts such as circles, boxes, and image masks alongside textual questions, deliberately designed to resist solutions based purely on language mediation.

MMStar is an elite vision-indispensable benchmark comprising 1,500 challenge samples meticulously selected by humans to address issues of visual dependency and data leakage in existing benchmarks. The benchmark evaluates six core capabilities and 18 detailed axes, with each sample undergoing strict human review to ensure visual dependency, minimal data leakage, and requirements for advanced multi-modal capabilities. Beyond traditional accuracy metrics, MM-Star introduces two novel metrics to measure multi-modal gain and multi-modal leakage in model training.

AI2D consists of 4,817 illustrative diagrams for research on diagram understanding and associated question answering. The dataset represents topics in primary school natural sciences such as food webs, life cycles, moon phases, and human physiology. Each diagram has been densely annotated with object segmentations, diagrammatic elements like arrows and lines, and text elements. The benchmark requires understanding abstract visual representations and symbolic elements common in scientific illustrations, testing both visual comprehension and scientific reasoning abilities.

We process each question by iteratively generating and verifying reasoning steps until the model produces an end-of-sequence token. The final answer is extracted from the complete reasoning trace using (Gemma-3:12B[38] via Ollama) and evaluated against the ground truth using string matching. The use of Ollama was only for answer extraction, and is not for any part of the verification pipeline. To isolate the contribution of the verification mechanism itself, we decouple generation variance from verification variance by fixing the base model seed across all methods, ensuring that every verifier evaluates the same set of candidate steps. The standard deviations reported (across 5 runs) therefore reflect verification noise alone. An end-to-end variance analysis, while informative for deployment, would compound sampling stochasticity across multi-step chains, conflating candidate quality with verifier quality and obscuring the comparison we aim to make: whether consensus-based aggregation selects better steps than alternatives given the same options.

Computational Resources: All experiments are conducted on NVIDIA A100 and 3 NVIDIA A6000 GPUs. To ensure full reproducibility, all code and evaluation scripts will be publicly released upon acceptance.

## 11 Additional Results

This section presents extended empirical analyses that complement the main paper’s experiments. It is organized along three axes: (i) comparisons with alternative inference-time strategies 11.1, (ii) systematic sensitivity analyses of every hyperparameter — individually 11.2 - 11.4, jointly 11.5, and across model configurations 11.6–11.8 and (iii) diagnostic analyses that characterize why the consensus formulation works, including the dispersion signal’s predictive value 11.9–11.10, the interplay between filtering and ranking 11.11–11.12, and generalization across model families 11.13.

## 11.1 Comparison with Test-Time Scaling Methods

VERDICT and test-time scaling (TTS) methods both spend additional inference compute to improve output quality. They difer in where that computation is spent. TTS methods generate multiple candidate responses and aggregate them at the answer or token level. VERDICT generates multiple candidate reasoning steps and verifies each through cross-modal consensus before the chain proceeds. The two paradigms address diferent failure modes: VERDICT intercepts errors within a reasoning chain, while TTS methods improve the quality of complete responses after generation. A controlled comparison clarifies how much each strategy contributes.

We compare VERDICT against five representative test-time scaling methods. (1) Self-Consistency aggregates candidate answers via majority voting across multiple sampled outputs [47]. (2) Self-Selector uses the base model itself as a verifier to select one response among the candidates [6]. (3) Self-Synthesizer combines potentially correct parts from diferent responses. We use the base model to aggregate responses into one coherent final answer [22]. (4) Self-Refinement [43]. (5) TTAug [18]. All methods except TTAug use temperature sampling (T=0.8) for diversity. VERDICT uses its standard configuration.

Results. Table 4 reports accuracy across all six benchmarks. VERDICT leads on average by +2.71 points over the strongest TTS method (Self-Synthesizer, 67.44%). The discussion is structured around a single question: where should additional inference computation be allocated: at the response level or at the reasoning-step level? Three structural diferences between the paradigms bear on the answer.

Table 4: VERDICT vs. test-time scaling baselines across six benchmarks. Best per benchmark in bold.
<table><tr><td>Method</td><td>3DSR</td><td>CV-3D</td><td>CV-2D</td><td>BLINK</td><td>MMStar</td><td>AI2D</td><td>Avg.</td></tr><tr><td>Base Model</td><td>56.12</td><td>76.39</td><td>74.27</td><td>48.31</td><td>61.25</td><td>81.52</td><td>66.31</td></tr><tr><td>Self-ConsistencyICLR, 23</td><td>57.08</td><td>78.12</td><td>76.03</td><td>42.66</td><td>59.16</td><td>74.63</td><td>64.61</td></tr><tr><td>Self-SelectorICML, 24</td><td>55.98</td><td>72.77</td><td>68.04</td><td>35.16</td><td>57.31</td><td>77.90</td><td>61.19</td></tr><tr><td>Self-Refinement_, -</td><td>54.94</td><td>73.01</td><td>66.05</td><td>40.15</td><td>55.08</td><td>73.29</td><td>60.42</td></tr><tr><td>Self-SynthesizerICML, 25</td><td>54.68</td><td>77.33</td><td>77.79</td><td>49.40</td><td>62.32</td><td>81.14</td><td>67.11</td></tr><tr><td>TTAugICLR, 26</td><td>52.12</td><td>73.19</td><td>70.47</td><td>50.26</td><td>60.88</td><td>69.75</td><td>62.78</td></tr><tr><td>VERDICT</td><td>59.02</td><td>82.34</td><td>79.22</td><td>51.32</td><td>65.88</td><td>83.14</td><td>70.15</td></tr></table>

Granularity of error correction. Self-Consistency, Self-Selector, Self-refinement, and Self-Synthesizer evaluate or combine complete responses. If a reasoning chain drifts at step 3 of 8, these methods can only discard the entire trajectory and hope that a diferent sample avoids the same error. VERDICT intervenes at step 3 directly, replacing the flawed continuation before downstream errors compound. Nature of the verification signal. TTS methods derive selection or aggregation signals from the base model’s own outputs: majority agreement (Self-Consistency) self-evaluation (Self-Selector), base-model-driven refinement (Self-Refinement), response synthesis (Self-Synthesizer), or across augmented views (TTAug). These are single-perspective signals. VERDICT derives its signal from cross-modal disagreement among three independently prompted, modality-specialized judges. Proposition 1 in the main paper establishes that this coupled disagreement structure carries information that no separable function of individual scores can replicate. The dispersion ∆<sup>∗</sup>, which is absent in all TTS baselines, provides a second acceptance axis that catches candidates with high average confidence but unresolved cross-modal tension (Section 11.12).

Cross-task consistency. Four of the five TTS methods fall below the unverified base model on average: Self-Consistency (64.61%), Self-Selector (61.19%), Self-Refinement (60.42%), and TTAug (62.78%) each underperform the base model’s 66.31%. Only Self-Synthesizer (67.44%) exceeds it. More critically, every TTS method degrades on at least one benchmark. Self-Consistency drops 5.65 points on BLINK and 6.89 points on AI2D. Self-Selector degrades on five of six benchmarks, with a 13.15 point collapse on BLINK. Self-Refinement falls below baseline on all six. TTAug loses 11.77 points on AI2D. Even Self-Synthesizer, the strongest TTS baseline, drops 1.44 points on 3DSRBench. VERDICT improves on every benchmark.

Where the gap is largest. The advantage is most pronounced on benchmarks requiring tight step-level visual grounding. On CV-Bench-3D, VERDICT leads Self-Synthesizer by 5.01 points; on 3DSRBench, it leads Self-Consistency (the best TTS method on that benchmark) by 1.94 points. These are depth-ordering and 3D spatial reasoning tasks where a single misjudged step early in the chain propagates through all subsequent comparisons: precisely the failure mode that response-level aggregation cannot intercept.

![](images/3c529a5ebcadb1d95c9d16274d8d8deac193360cd542ce58cd7a869d15fb1fe2.jpg)

![](images/ed7e0e2257292ceb7efe943d5fa6b2cc1cb22f7ffeae346d993f5d8fa83f26ae.jpg)  
Fig. 8: Dispersion tolerance sensitivity across all six benchmarks $\mathit { \Omega } ( \tau \ =$ 0.6 fixed). All benchmarks exhibit the same three-regime structure: too-strict rejection at low ϵ, an optimal plateau at $\epsilon \in [ 0 . 1 , 1 . 0 ]$ , and gradual degradation at high ϵ. Dotted lines indicate base model performance.  
Fig. 9: Accuracy vs. confidence threshold τ across all six benchmarks $( \epsilon =$ 0.1 fixed). Five benchmarks peak at $\tau \ = \ 0 . 6 ;$ 3DSRBench is the sole exception, where the permissive regime dominates. Horizontal dashed lines indicate base model performance.

## 11.2 Cross-Benchmark Dispersion Tolerance Sensitivity

Section 6.3 of the main paper analyzes the dispersion tolerance ϵ on 3DSRBench. Here we extend this analysis to all six benchmarks, sweeping $\epsilon \in \{ 0 . 0 0 1 , 0 . 0 5 , 0 . 1$ $0 . 5 , 1 . 0 , 2 . 5 , 3 . 0 \}$ with $\tau = 0 . 6$ fixed and reporting accuracy with standard errors across five seeds.

Figure 8 and Table 5 confirm that the three-regime pattern generalizes across benchmarks. Three findings emerge.

The three-regime structure is universal. Every benchmark follows the same qualitative pattern: performance rises from the strict regime $( \epsilon < 0 . 1 )$ into an optimal plateau $( \epsilon \in [ 0 . 1 , 1 . 0 ] )$ , then declines in the permissive regime $( \epsilon >$ 1.0). Degradation is gradual in both directions because two safety nets operate universally: the fallback ranking $( \bar { s } ^ { * } - \varDelta ^ { * } )$ catches failures when filtering is too aggressive, and the confidence threshold $\left( \bar { s } ^ { * } > \tau \right)$ continues rejecting weak candidates when filtering is too lenient.

Table 5: Worst-case improvement over the base model (minimum accuracy gain across benchmarks) at each ϵ.
<table><tr><td colspan="2">€ Min gain (%) Benchmark</td></tr><tr><td>0.001</td><td>-0.95 3DSRBench</td></tr><tr><td>0.05 +0.36</td><td>3DSRBench</td></tr><tr><td>0.1</td><td>+1.62 AI2D</td></tr><tr><td>0.5</td><td>+1.69 AI2D</td></tr><tr><td>1.0</td><td>+1.65 AI2D</td></tr><tr><td>2.5</td><td>+1.16 AI2D</td></tr><tr><td>3.0</td><td>+0.85 AI2D</td></tr></table>

The optimal plateau spans an order of magnitude. Accuracy diferences between $\epsilon = 0 . 1$ and $\epsilon = 1 . 0$ are small across all benchmarks: 0.76 points on 3DSRBench, 0.26 on CV-Bench-3D, 0.25 on CV-Bench-2D, 0.05 on BLINK, 0.25 on MMStar, and 0.03 on AI2D. This stability indicates that $\epsilon = 0 . 1$ does not require finetuning to be efective.

Sensitivity increases with task dificulty. 3DSRBench exhibits the widest performance range (5.26 points), consistent with the inherent cross-modal tension in 3D spatial reasoning. AI2D shows the narrowest range (1.28 points), reflecting its high baseline accuracy and modest VERDICT gain (+1.62).

Table 5 reports worst-case robustness: the minimum improvement over the base model at each ϵ. The worst-case benchmark is 3DSRBench for $\epsilon \leq 0 . 0 5$ (where overly strict filtering removes informative candidates) and AI2D for $\epsilon \geq 0 . 1$ (reflecting its small overall gain). The chosen default $\epsilon = 0 . 1$ provides near-optimal worst-case robustness while sitting at the conservative end of the plateau, which is appropriate for deployment without prior knowledge of task distribution.

The complementary safety-net structure is confirmed across benchmarks: even at $\epsilon = 0 . 0 0 1$ , five of six benchmarks remain above baseline; when ϵ is too permissive, the confidence threshold limits degradation to 1-2 points. This mutual reinforcement explains why no benchmark exhibits a sharp performance clif at any ϵ value.

## 11.3 Cross-Benchmark Confidence Threshold Analysis

Section 6.4 of the main text presents the τ sensitivity analysis on 3DSRBench. Here we extend this analysis to all six benchmarks, addressing whether the chosen operating point $\tau = 0 . 6$ is justified given that the permissive regime $( \tau \leq 0 . 0 1 )$ achieves substantially higher accuracy on 3DSRBench.

Setup We sweep $\tau \in \{ 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 0 . 6 , 1 . 0 , 1 0 . 0 \}$ with $\epsilon = 0 . 1$ fixed, reporting accuracy with standard errors across five random seeds. Results are shown in Figure 9.

Per-benchmark behavior. Five of six benchmarks follow a consistent pattern: accuracy increases as τ rises toward 0.6. The gains from the permissive regime to $\tau = 0 . 6$ range from +0.90 (AI2D) to +1.65 (MMStar), reflecting that the confidence threshold catches genuinely weak candidates that the dispersion filter alone does not intercept.

The Permissive Regime as Evidence for the Consensus Formulation The permissive regime on 3DSRBench (65–69% accuracy) provides independent evidence for the consensus formulation. $\mathrm { A t } ~ \tau ~ \leq ~ 0 . 0 1$ , the confidence filter is efectively disabled: nearly all candidates pass, and the system relies almost entirely on the dispersion criterion $\varDelta ^ { \ast } < \epsilon .$ Since ${ \bar { s } } ^ { * }$ equals the raw mean by construction, the ranking among accepted candidates is identical to what the Mean baseline would produce. The improvement over Mean $( 5 8 . 3 4 \%  6 5 - 6 9 \% )$ is therefore attributable entirely to the dispersion filter. The dispersion $\varDelta ^ { \ast }$ identifies conflicted candidates that raw score variance misses, consistent with Proposition 1 and the Raw Average ablation in Section 6.2.

Table 6: Worst-case improvement over the base model (minimum accuracy gain across benchmarks) at each τ .
<table><tr><td colspan="2">T Min gain(%) Benchmark</td></tr><tr><td> $1 0 ^ { - 4 }$ </td><td>+0.72 AI2D</td></tr><tr><td> $1 0 ^ { - 3 }$  +0.83</td><td>AI2D</td></tr><tr><td> $1 0 ^ { - 2 }$  +0.99</td><td>AI2D</td></tr><tr><td> $1 0 ^ { - 1 }$  +1.31</td><td>AI2D</td></tr><tr><td>0.6 +1.62</td><td>AI2D</td></tr><tr><td>1.0</td><td>+1.44 AI2D</td></tr><tr><td>10.0 +1.44</td><td>AI2D</td></tr></table>

The consensus formulation thus contributes through complementary mechanisms at diferent operating points: dispersion filtering dominates in the permissive regime on high-tension tasks; acceptance calibration through the dual criterion dominates at moderate $\tau$ on lower-tension tasks. At $\tau = 0 . 6$ , both mechanisms are active, producing the 2.59–2.95 point gap between VER-DICT and Raw Average reported in Section 6.2.

Single threshold across benchmarks. Per-task threshold selection would undermine the training-free, plug-in design principle that motivates our framework and require held-out validation

sets for each benchmark, introducing a data dependency we specifically avoid. Reporting a single fixed configuration across all six benchmarks is a stricter evaluation protocol that demonstrates genuine cross-task generalization.

Performance trade-ofs at the operating point. On 3DSRBench specifically, $\tau =$ 0.6 leaves performance on the table: 59.02% versus 68.71% at $\tau = 1 0 ^ { - 4 }$ . This is the cost of optimizing for robustness under unknown task distributions rather than per-task maximization. If the deployment distribution is known to be similar to that of 3DSRBench, the threshold should be lowered accordingly. Adaptive threshold selection based on task characteristics remains a promising direction for future work.

## 11.4 Threshold Sensitivity: Extended Analysis

The main paper (Section 6.3) summarizes threshold sensitivity on 3DSRBench. Here we provide the full regime analysis for both parameters, contextualizing the cross-benchmark results from Sections 11.2 and 11.5.

Confidence threshold τ . We sweep $\tau \ \in \ \{ 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 0 . 6 , 1 . 0 , 1 0 . 0 \}$ with $\epsilon = 0 . 1$ fixed. Performance separates into three regimes. In the permissive regime $( \tau \leq 0 . 0 1 )$ , nearly all candidates pass the confidence check, and the 65– 69% accuracy depends primarily on how well consensus scores rank accepted candidates. Since ${ \bar { s } } ^ { * }$ equals the raw mean by construction, ranking among accepted candidates is identical to Mean aggregation; the improvement over Mean $( 5 8 . 3 4 \%  6 5 - 6 9 \% )$ is attributable entirely to the dispersion filter $\varDelta ^ { \ast } < \epsilon$ . This provides independent evidence that consensus dispersion captures structure that raw variance fails to capture (Proposition 1, main paper). In the intermediate regime $( \tau = 0 . 1 \small { - 0 . 6 } )$ , performance stabilizes around 60– $6 1 \%$ . Candidates filtered here where judges partially disagree but lack a strong consensus are precisely cases where cross-modal tension carries diagnostic value. On 3DSRBench, removing these borderline candidates is counterproductive because some contentious steps are correct; this explains why $\tau = 0 . 6$ is individually optimal on five of six benchmarks (Section 11.2) but not on 3DSRBench. In the restrictive regime $( \tau \geq 1 . 0 )$ , no candidates pass (consensus scores are bounded in [0, 1]), and the system falls back to $\bar { s } ^ { * } - \varDelta ^ { * }$ ranking, achieving ${ \sim } 6 0 . 6 \%$ . That comparable accuracy is maintained under pure fallback confirms that the combined criterion provides a reasonable ranking signal even without dual-criterion filtering.

Dispersion tolerance ϵ. We sweep $\epsilon \in \{ 0 . 0 0 1 , 0 . 0 5 , 0 . 1 , 0 . 5 , 1 . 0 , 2 . 5 , 3 . 0 \}$ with $\tau = 0 . 6$ fixed. Performance peaks in an optimal plateau at $\epsilon \in [ 0 . 1 , 1 . 0 ]$ . When ϵ is too strict $( < 0 . 1 )$ , accuracy degrades to 54–57% because near-unanimous agreement is unrealistic for spatial reasoning where some cross-modal tension is natural. Many legitimate steps are rejected for dispersion only slightly above the threshold. Performance does not collapse because the fallback mechanism remains active. In the optimal plateau $( \epsilon \in [ 0 . 1 , 1 . 0 ] )$ , accuracy stabilizes around $6 0 \mathrm { - } 6 1 \% ;$ the filter removes candidates with genuine conflict while admitting those with task-appropriate disagreement. The plateau width (∼1 order of magnitude) indicates insensitivity to exact threshold choice consistent with the crossbenchmark finding that accuracy diferences between $\epsilon = 0 . 1$ and $\epsilon = 1 . 0$ are small on all benchmarks (Section 11.2). When ϵ is too permissive $( > 1 . 0 )$ , performance gradually declines to 57–58% as the filter loses discriminative power, though the confidence threshold continues catching weak candidates.

Interaction and safety nets. The two thresholds provide complementary safety nets, as confirmed by the joint analysis (Section 11.5). When ϵ is too strict, the fallback ranking prevents catastrophic failure; when ϵ is too permissive, the confidence threshold $\bar { s } ^ { * } > \tau$ continues filtering. This mutual reinforcement explains why the framework degrades gracefully across the full parameter range, rather than exhibiting sharp clifs—a property that holds across all six benchmarks.

Key insight. Consensus scores function most naturally as a ranking tool rather than a binary classifier. The permissive regime succeeds by preserving the full candidate pool while leveraging dispersion for filtering; the restrictive regime maintains comparable performance because its fallback incorporates both confidence and disagreement. This explains the main paper’s finding (Section 6.1) that removing intelligent ranking (No Selection ablation) costs more than removing filtering (No Rejection), and why the dispersion-reject quadrant in Section 11.10 contains candidates that confidence-only methods cannot distinguish from accepts.

## 11.5 Joint Threshold Interaction Analysis

Figures 5 and 6 of the main paper analyze sensitivity to τ and ϵ separately. Here we examine whether these thresholds interact by sweeping both jointly on 3DSRBench: all combinations of $\tau \in \{ 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 0 . 6 , 1 . 0 , 1 0 . 0 \}$ and $\epsilon \in \{ 0 . 0 0 1 , 0 . 0 1 , 0 . 0 5 , 0 . 1 , 0 . 5 , 1 . 0 , 2 . 5 \}$ , yielding 49 configurations. Figure 10 presents heatmaps of accuracy and acceptance rate. Four findings emerge.

![](images/7f3e80f3df3292301ef82596a93542a9eca54f650caac716679e666a82b595a2.jpg)

![](images/6ade19b0eee8425001b39e8655102759c965e81902b7dc1cdef645ac48355d23.jpg)  
Fig. 10: Joint threshold analysis on 3DSRBench. Left: Accuracy (%) across the (τ, ϵ) grid. Right: Acceptance rate (%), i.e., the fraction of steps where at least one candidate passes both criteria. The operating point $( \tau = 0 . 6 , \epsilon = 0 . 1 )$ lies on a stable plateau; the global optimum at low τ trades cross-benchmark robustness for 3DSRBench-specific gains.

The operating point lies on a plateau, not an edge. Accuracy at $( \tau = 0 . 6 , \epsilon = 0 . 1 )$ is 59.0%, with similar values in the surrounding region: 59.8% at $( \tau = 0 . 6 , \epsilon =$ 0.5), 60.6% at $( \tau = 1 . 0 , \epsilon = 0 . 1 )$ , and 57.3% at $( \tau = 0 . 6 , \epsilon = 0 . 0 5 )$ . Moving in any direction causes gradual change (1–2 points), confirming that the configuration is robust to small perturbations.

The two thresholds interact asymmetrically. At low τ (permissive confidence filtering), varying ϵ produces large swings: accuracy ranges from 55.2% at $\epsilon =$ 0.001 to 68.7% at $\epsilon = 0 . 1$ when $\tau = 1 0 ^ { - 4 }$ a 13.5 point diference. At high τ, the efect is muted: only 5.7 points when $\tau = 1 . 0$ . This asymmetry arises because at low τ nearly all candidates pass the confidence check, making ϵ the primary filter; at high τ, few candidates pass regardless of dispersion.

The global optimum is at low τ, moderate ϵ. The highest accuracy (68.7%) occurs at $( \tau = 1 0 ^ { - 4 } , \epsilon = 0 . 1 )$ , corresponding to the permissive regime where the confidence filter is efectively disabled. As discussed in Section 6.3 of the main paper, this configuration achieves higher accuracy on 3DSRBench but does not generalize across benchmarks. The operating point $( \tau = 0 . 6 , \epsilon = 0 . 1 )$ sacrifices 9.7 points on 3DSRBench for cross-benchmark robustness.

Complementary safety nets prevent catastrophic failure. The leftmost column $( \epsilon = 0 . 0 0 1 )$ shows degraded but not catastrophic accuracy (53–55%) because the fallback ranking remains active. The rightmost column (ϵ = 2.5) shows only mild degradation (57–65%) because the confidence threshold continues filtering weak candidates. No region falls substantially below baseline (56.12%) except the extreme corner at $( \tau = 0 . 6 , \epsilon = 0 . 0 0 1 )$

The acceptance rate heatmap (right panel) reveals complementary structure. $\mathrm { A t } ~ \tau \geq 1 . 0$ , acceptance drops to 0% because $\bar { s } ^ { * } \in [ 0 , 1 ]$ by construction, forcing the system to operate entirely through fallback ranking. At the operating point, acceptance is approximately 88%, consistent with the 15% fallback rate reported in the main paper. The highest-accuracy region (low τ, moderate ϵ) coincides with acceptance rates above 95%, indicating that in the permissive regime, verification functions primarily through ranking rather than filtering, which is consistent with the main paper’s finding that consensus scores operate optimally as ranking tools.

## 11.6 Cross-Benchmark Judge Scale Analysis

Section 6.4 of the main paper shows that the margin between VERDICT and Mean aggregation widens as the judge model scale decreases on 3DSRBench. Here we extend this analysis to all six benchmarks.

We replace the 7B verification agents (Qwen2.5-VL-7B-Instruct) with their 4B and 2B variants while holding the base reasoner fixed at 7B, isolating judge scoring quality from candidate generation quality. All hyperparameters remain unchanged. Figure 11 presents accuracy across all benchmarks at each judge scale. Three findings emerge.

The widening-margin pattern holds on five of six benchmarks. On 3DSRBench, CV-Bench-2D, BLINK, MMStar, and AI2D, the gap between VERDICT and Mean increases monotonically as judge scale decreases from 7B to 2B. The efect is particularly pronounced on BLINK, where the margin expands from 1.15 points at 7B to 4.56 points at 2B.

CV-Bench-3D exhibits a non-monotonic pattern. On this benchmark, the margin peaks at 4B $( \varDelta = 2 . 9 6 )$ before contracting to 1.25 at 2B. We attribute this to a floor efect: CV-Bench-3D is a depth-ordering task where the base model already performs strongly (76.39%), and at 2B scale, both methods approach this floor. When judges become suficiently noisy that their scores carry minimal signal, the consensus computation has less structure to exploit. Notably, even at 2B, VERDICT still outperforms Mean by 1.25 points: the margin contracts but does not invert.

VERDICT remains above or near baseline across all configurations. Across the 18 configurations tested (6 benchmarks × 3 scales), VERDICT exceeds the base model in 16 cases and remains within 0.5 points in the remaining 2. In contrast, Mean aggregation drops below baseline in 5 configurations. This asymmetry confirms that the consensus formulation provides a safety margin against judge degradation that simple averaging does not.

![](images/7fbc1971c6389a4398d3da69ae40504f05e3ab45aae5ede6362710936a69403d.jpg)  
Fig. 11: Accuracy as a function of judge model scale (7B, 4B, 2B) for VERDICT (red) and Mean aggregation (blue) across all six benchmarks. Dashed lines indicate base model accuracy. ∆ annotations show the margin between methods at each scale.

These results have practical implications. In resource-constrained settings with 2B-scale judges, VERDICT recovers 1.2–4.6 points over naive averaging depending on the task. The widening margin also suggests that investing in the consensus algorithm is complementary to, not a substitute for, judge quality: absolute accuracy at 7B exceeds that at 2B on all benchmarks, and the widening margin reflects that Mean degrades faster rather than smaller judges being preferable. Finally, on benchmarks where the base model is already strong (CV-Bench-3D, AI2D), margin expansion is more modest and floor efects may emerge; on challenging benchmarks (BLINK, MMStar), the consensus formulation provides substantial and increasing value as judge quality degrades.

## 11.7 Cross-Benchmark Stubbornness Sensitivity

Section 6.2 of the main paper ablates each stubbornness parameter on 3DSR-Bench. A natural concern is whether the chosen configuration

$( \lambda _ { V } = 1 . 5 , ~ \lambda _ { L } = 1 . 0 , ~ \lambda _ { C } = 0 . 8 )$ is implicitly tuned to that benchmark. We address this by repeating the same ablation on CV-Bench-3D (where VERDICT achieves its largest gain, +5.95) and BLINK (where perceptual reasoning differs qualitatively from spatial reasoning). Each curve varies one parameter over {0.3, 0.8, 1.0, 1.5, 2.0} while holding the others fixed.

Table 7 presents results for both benchmarks. The sensitivity hierarchy mirrors 3DSRBench on both: $\lambda _ { V }$ has the widest performance range (2.73 points on CV-

Table 7: Stubbornness sensitivity on CV-Bench-3D (base: 76.39%, Mean: 79.77%) and BLINK (base: 48.31%, Mean: 50.17%). Each row varies one parameter; others remain at defaults. Bold indicates peak accuracy.
<table><tr><td> $\lambda = 0 . 3$ </td><td> $\lambda = 0 . 8$   $\lambda = 1 . 0$ </td><td> $\lambda = 1 . 5$ </td><td> $\lambda = 2 . 0$  Range</td></tr><tr><td> $\lambda _ { V } ~ 7 9 . 6 1 { \scriptstyle \pm 0 . 6 2 }$ </td><td> $8 0 . 8 9 { \scriptstyle \pm 0 . 5 5 }$   $8 1 . 5 2 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $\mathbf { 8 2 . 3 4 } \pm \mathbf { 0 . 5 1 }$   $8 1 . 8 7 { \scriptstyle \pm 0 . 5 4 }$ </td><td>2.73</td></tr><tr><td>D  $\stackrel { \triangledown } { > } \lambda _ { L } 8 1 . 1 7 \pm 0 . 5 8$ </td><td> $8 1 . 9 5 { \scriptstyle \pm 0 . 5 3 }$   $\mathbf { 8 2 . 3 4 } \pm \mathbf { 0 . 5 1 }$ </td><td> $8 2 . 0 8 { \scriptstyle \pm 0 . 5 5 }$   $8 1 . 7 1 { \scriptstyle \pm 0 . 5 7 }$ </td><td>1.17</td></tr><tr><td>C  $\lambda _ { C } ~ 8 1 . 5 8 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $\mathbf { 8 2 . 3 4 } \pm \mathrm { 0 . 5 1 }$   $8 2 . 2 1 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $8 1 . 9 3 { \scriptstyle \pm 0 . 5 4 }$   $8 1 . 7 2 { \scriptstyle \pm 0 . 5 6 }$ </td><td>0.76</td></tr><tr><td> $\lambda _ { V } ~ 4 9 . 5 3 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $5 0 . 4 7 { \scriptstyle \pm 0 . 5 1 }$   $5 0 . 8 6 { \scriptstyle \pm 0 . 4 9 }$ </td><td> ${ \bf 5 1 . 3 2 } { \scriptstyle \pm 0 . 4 8 }$   $5 0 . 9 1 { \scriptstyle \pm 0 . 5 0 }$ </td><td>1.79</td></tr><tr><td>BK  $\lambda _ { L } ~ 5 0 . 5 8 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $5 1 . 0 4 { \scriptstyle \pm 0 . 5 0 }$   ${ \bf 5 1 . 3 2 } _ { \pm 0 . 4 8 }$ </td><td> $5 1 . 1 1 { \scriptstyle \pm 0 . 5 0 }$   $5 0 . 7 9 { \scriptstyle \pm 0 . 5 2 }$ </td><td>0.74</td></tr><tr><td> $\lambda _ { C } ~ 5 0 . 8 3 { \pm } 0 . 5 1$ </td><td> ${ \bf 5 1 . 3 2 } { \scriptstyle \pm 0 . 4 8 }$   $5 1 . 1 9 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $5 0 . 9 7 { \scriptstyle \pm 0 . 5 0 }$   $5 0 . 7 5 { \scriptstyle \pm 0 . 5 2 }$ </td><td>0.57</td></tr></table>

Table 8: Sensitivity ranges across three benchmarks. The hierarchy $\lambda _ { V } > \lambda _ { L } >$ $\lambda _ { C }$ is preserved on every benchmark, and all 45 configurations exceed the base model.
<table><tr><td>Benchmark  $\lambda _ { V }$ </td><td>range  $\lambda _ { L }$ </td><td>range  $\lambda _ { C }$ </td><td>range  $\mathrm { A l l } >$  Base</td></tr><tr><td>3DSRBench</td><td>2.90</td><td>1.40 1.19</td><td>V</td></tr><tr><td>CV-Bench-3D</td><td>2.73</td><td>1.17 0.76</td><td>√</td></tr><tr><td>BLINK</td><td>1.79</td><td>0.74 0.57</td><td>√</td></tr></table>

Bench-3D, 1.79 on BLINK), followed by $\lambda _ { L } ~ ( 1 . 1 7 , 0 . 7 4 )$ and $\lambda _ { C }$ (0.76, 0.57). All curves peak at the default configuration.

The $\lambda _ { V }$ curve on CV-Bench-3D is particularly informative. At $\lambda _ { V } = 0 . 3$ , the visual agent places 77% of its weight on group consensus (Eq. (2), main paper), suppressing its private visual evidence. On a depth-ordering task where visual grounding is the primary signal, this dilution costs 2.73 points. At $\lambda _ { V } = 2 . 0$ , the agent becomes overly stubborn, resisting legitimate corrections when its initial score is noisy; performance drops 0.47 points. The optimal $\lambda _ { V } = 1 . 5$ balances these failure modes.

The tighter ranges on BLINK are expected: the overall VERDICT gain is smaller (+3.01 vs. +5.95 on CV-Bench-3D), leaving less room for parameterinduced variation. The relative sensitivity pattern, however, is unchanged. Table 8 summarizes the cross-benchmark pattern. Three conclusions follow.

The sensitivity hierarchy is benchmark-independent. $\lambda _ { V }$ is the most impactfu parameter on every benchmark, $\lambda _ { C }$ the least. This ordering reflects the agent structure rather than dataset-specific tuning: on tasks requiring visual grounding, the visual agent’s stubbornness determines how much evidence survives consensus pressure.

The default configuration is jointly optimal. Across all nine curves (three parameters × three benchmarks), peak accuracy occurs at the default $\left( \lambda _ { V } = 1 . 5 , \right.$ $\lambda _ { L } { = } 1 . 0 , \ \lambda _ { C } { = } 0 . 8 )$

Table 9: Pairwise stubbornness swap results. ∆: accuracy drop from default. Bold: configurations where λ retains the highest value.
<table><tr><td></td><td>3DSRBench CV-Bench-3D</td><td>BLINK</td></tr><tr><td>Configuration Acc. (%)</td><td>∆ Acc. (%)</td><td>∆ Acc. (%)</td></tr><tr><td>Default</td><td>82.34 58.61 -0.41</td><td>51.32</td></tr><tr><td>L↔C</td><td>81.87-0.47</td><td>50.98-0.34</td></tr><tr><td>V↔L</td><td>80.51-1.83</td><td>50.38-0.94</td></tr><tr><td>V↔C</td><td>79.93-2.41</td><td>49.91-1.41</td></tr><tr><td>Mean baseline</td><td>79.77</td><td>50.17</td></tr><tr><td>Base model</td><td>76.39</td><td>48.31</td></tr></table>

## 11.8 Cross-Benchmark Stubbornness Assignment Analysis

Section 6.2 of the main paper tests whether the assignment of stubbornness values to agents matters by permuting the triple $( \lambda _ { V } = 1 . 5 , ~ \lambda _ { L } = 1 . 0 , ~ \lambda _ { C } = 0 . 8 )$ via pairwise swaps on 3DSRBench. Here we repeat the same protocol on CV-Bench-3D and BLINK to verify that the ordering $\lambda _ { V } > \lambda _ { L } > \lambda _ { C }$ is not benchmarkspecific.

Each pairwise swap exchanges the stubbornness values of two agents: L↔C leaves $\lambda _ { V } = 1 . 5$ unchanged, V↔L reduces $\lambda _ { V }$ to 1.0, and V↔C reduces $\lambda _ { V }$ to 0.8. The three swaps form a gradient of increasing disruption to the visual agent’s stubbornness. Table 9 reveals three consistent patterns.

The drop ordering is benchmark-independent. On every benchmark, V↔C produces the largest degradation, V↔L an intermediate one, and L↔C the smallest. The ordering holds across 3DSRBench $( 1 . 7 6 > 1 . 1 9 > 0 . 4 1 )$ , CV-Bench-3D $( 2 . 4 1 > 1 . 8 3 > 0 . 4 7 )$ , and BLINK $( 1 . 4 1 > 0 . 9 4 > 0 . 3 4 )$ . Since these benchmarks test qualitatively diferent capabilities, the consistency confirms that the assignment ordering reflects agent architecture rather than dataset-specific features.

Degradation tracks the $\lambda _ { V }$ reduction. Each swap changes two parameters, yet drop magnitude correlates almost entirely with how much λ<sub>V</sub> decreases. L↔C leaves $\lambda _ { V } = 1 . 5$ untouched and costs under 0.5 points on every benchmark; V↔C reduces $\lambda _ { V }$ to 0.8 and costs 1.4–2.4 points. When the visual agent becomes too responsive to consensus pressure, it concedes ground on perceptually grounded judgments it should have defended.

λ<sub>V</sub> as the highest value is the critical invariant. On all three benchmarks, L↔C (which preserves $\lambda _ { V }$ as highest) exceeds the Mean baseline: 58.61 vs. 58.34 on 3DSRBench, 81.87 vs. 79.77 on CV-Bench-3D, 50.98 vs. 50.17 on BLINK. When λ<sub>V</sub> is no longer the largest, performance approaches or crosses the Mean baseline. On BLINK, V↔C (49.91%) dips below Mean (50.17%); on 3DSRBench, both V↔L (57.83%) and V↔C (57.26%) fall below Mean (58.34%). CV-Bench-3D is the exception: even V↔C (79.93%) stays marginally above Mean (79.77%), reflecting the larger absolute gap on this benchmark.

![](images/64108fa1d880fa3bcd6dacdf6946bab6633460cf6cbc906304c5ff57d4312381.jpg)  
Fig. 12: Distribution of consensus dispersion $\varDelta ^ { \ast }$ across candidate reasoning steps, partitioned by acceptance status and chain correctness. Dashed line: $\epsilon = 0 . 1$ Inset: incorrect-chain rates among accepted vs. rejected candidates.

All twelve swap configurations exceed the unverified base model. Even $\mathrm { V } {  } \mathrm { C }$ on 3DSRBench (57.26%) outperforms baseline (56.12%) by 1.14 points. The consensus formulation provides value even under misassigned stubbornness, though correct assignment is needed to outperform simpler aggregation.

Swaps change two parameters simultaneously, so they produce larger drops than single-parameter changes. On CV-Bench-3D, setting $\lambda _ { V } = 0 . 8$ alone yields 80.89% (Table 7), while $\mathrm { V {  } C } ~ ( \lambda _ { V } = 0 . 8 , \lambda _ { C } = 1 . 5 )$ yields 79.93%—the additional of-optimal $\lambda _ { C }$ costs 0.96 points. On BLINK: 50.47% vs. 49.91%, a 0.56 point additional cost. Misassignment penalties compound when multiple agents are simultaneously mis-specified.

## 11.9 Dispersion as a Diagnostic Signal

Proposition 1 establishes that consensus dispersion $\varDelta ^ { \ast }$ captures cross-agent interaction structure that no separable measure can replicate. Here, we provide empirical evidence that this structure is diagnostically meaningful: high dispersion is associated with reasoning steps that lead to incorrect answers.

For each candidate step evaluated during inference, we record its $\varDelta ^ { \ast }$ and whether the complete chain produces the correct final answer. We partition candidates by (1) acceptance status under the dual criterion (Eq. (4), main paper) and (2) chain correctness. We report results on 3DSRBench and CV-Bench-3D, representing the lowest and highest VERDICT gains, respectively.

Figure 12 shows the distribution of $\varDelta ^ { \ast }$ across the four groups. Three observations follow.

High dispersion is enriched for incorrect chains. On 3DSRBench, 51.9% of rejected candidates belong to chains producing incorrect answers, compared to

37.9% among accepted candidates a 1.4× enrichment. On CV-Bench-3D, enrichment is $1 . 6 \times ~ ( 2 7 . 1 \% ~ \mathrm { v s . } ~ 1 6 . 4 \% )$ . The higher baseline accuracy on CV-Bench-3D means absolute incorrect rates are lower, but relative enrichment is stronger, indicating that the dispersion filter is more selective.

Accepted candidates have substantially lower dispersion. Mean $\varDelta ^ { \ast }$ among accepted candidates is 0.048 on 3DSRBench and 0.041 on CV-Bench-3D, compared to 0.175 and 0.160 for rejected candidates a 3.6× and 3.9× separation, respectively. This confirms that $\epsilon =$ 0.1 sits in a natural gap between populations rather than imposing an arbitrary split.

The filter is diagnostic, not oracular. Rejected candidates include correct-chain members on both benchmarks: 48.1% on 3DSR-Bench, 72.9% on CV-Bench-3D. This overlap confirms the main paper’s finding that consensus scores function optimally as ranking tools rather than binary classifiers. High dispersion signals unresolved cross-modal conflict a risk factor for error, not a guarantee. The fallback mechanism $( \bar { s } ^ { * } - \varDelta ^ { * }$ ranking) recovers value from rejected candidates by selecting the one with the best balance of confidence and agreement.

![](images/59424a103ff278d0812adea9df19e9f8f4c09e30302c7b24e11f505271675977.jpg)  
Fig. 13: ROC curves treating consensus dispersion $\varDelta ^ { \ast }$ (solid) and raw score variance (dashed) as a binary classifier for chain incorrectness on 3DSRBench and CV-Bench-3D.

The overall acceptance rates (77.9% on 3DSR-Bench, 85.4% on CV-Bench-3D) reflect the inherent cross-modal tension in 3D spatial reasoning, where visual and logical agents more

frequently disagree, exactly where dispersion-based filtering provides the most value. The moderate enrichment (1.4- 1.6×) is consistent with two properties of the framework: VERDICT operates at the step level, so a single high-dispersion step does not deterministically cause an incorrect answer since subsequent steps may recover; and the coupled scoring formulation already compresses raw disagreement through consensus adjustment, so residual dispersion represents conflict that persists even after accounting for consensus pressure; a more conservative signal than raw variance.

Quantifying diagnostic value via ROC analysis. To move beyond enrichment ratios and provide a concrete quantitative handle on how useful $\varDelta ^ { \ast }$ is as a signal, we treat consensus dispersion as a continuous-valued binary classifier for chain incorrectness: a candidate is predicted “incorrect” if $\varDelta ^ { \ast }$ exceeds a given threshold, and “correct” otherwise. To isolate the contribution of the coupled consensus formulation, we also evaluate raw score variance used by the Variance baseline in Table 2 of the main paper as a competing classifier under the same protocol. Figure 13 presents ROC curves on 3DSRBench and CV-Bench-3D for both signals. Three observations are consistent with the analysis above.

Dispersion carries genuine diagnostic signal. The ROC AUC is 0.65 on 3DSR-Bench and 0.66 on CV-Bench-3D, comfortably above the 0.50 random baseline, confirming that higher dispersion is systematically associated with incorrect chains, indicating that dispersion-based ranking concentrates incorrect chains above chance at every recall level.

Consensus dispersion strictly dominates raw variance. Raw score variance achieves AUC of 0.58 on 3DSRBench and 0.57 on CV-Bench-3D: above chance, but consistently below consensus dispersion by +0.07 and +0.09 respectively.. The result directly visualises the non-separability established by Proposition 1: because variance treats all agent deviations symmetrically, it cannot distinguish a highλ visual agent dissenting (diagnostically informative on spatial tasks) from a low-λ contextual agent dissenting (less informative). The coupled consensus formulation encodes this asymmetry through the stubbornness parameters, and the AUC gap quantifies how much diagnostic power that asymmetry contributes.

The signal is moderate, not oracular. AUC values in the 0.65-0.66 range confirm the main text’s characterization: $\varDelta ^ { \ast }$ is a risk factor for error, not a deterministic indicator.This is expected for two reasons already noted: VERDICT operates at the step level, so a single high-dispersion step does not deterministically cause an incorrect answer; and the coupled scoring formulation already compresses raw disagreement, so residual dispersion represents conflict that persists after consensus adjustment.The moderate AUC also explains why the framework degrades gracefully rather than exhibiting sharp clifs across the threshold sweep in Section S4.2: a diagnostic-but-not-oracular signal benefits from soft integration (ranking and dual filtering) rather than hard binary classification.

## 11.10 Confidence–Dispersion Landscape

Here we investigate the dual acceptance criterion by plotting mean consensus confidence $\bar { s } ^ { * }$ against consensus dispersion $\varDelta ^ { \ast }$ for all candidate reasoning steps on 3DSRBench, color-coded by chain correctness.

For each candidate, we record $( \bar { s } ^ { * } , \varDelta ^ { * } )$ from the coupled scoring system (Eq. (3), main paper) and label it by whether the full chain produces the correct answer. The dual criterion partitions this plane into four quadrants: Accept $( \bar { s } ^ { * } > \tau _ { : }$ $\varDelta ^ { \ast } < \epsilon )$ , Reject on dispersion $( \bar { s } ^ { * } > \tau , \varDelta ^ { * } \geq \epsilon )$ , Reject on confidence $\left( \bar { s } ^ { * } \leq \tau , \right.$ $\varDelta ^ { \ast } < \epsilon )$ , and Reject on both. Figure 14 reveals four features of the decision landscape.

The accept region is enriched for correct chains. Of 623 candidates in the accept quadrant $\left( \bar { s } ^ { * } > 0 . 6 , \varDelta ^ { * } < 0 . 1 \right)$ ), 62% belong to correct chains, compared to 59% overall. This region captures candidates with both high collective confidence and low residual disagreement, exactly the verification state the dual criterion targets.

The dispersion-reject quadrant isolates the key failure mode. 109 candidates (13.6%) pass the confidence threshold but fail the dispersion check. These are candidates that confidence-only methods would accept, but where agents have not genuinely converged. The correct-chain rate drops to 54%, confirming that high dispersion at high confidence is a risk factor. The starred annotation marks the main paper’s numerical example $( \hat { \mathbf { s } } ~ = ~ ( 0 . 9 , 0 . 3 , 0 . 9 )$ , yielding $\bar { s } ^ { * } ~ = ~ 0 . 7 1$ ， $\varDelta ^ { * } = 0 . 1 3 )$ : a candidate that looks promising by confidence alone but is correctly flagged by dispersion.

The confidence-reject and both-reject quadrants are sparse. Only 4.4% and 4.1% of candidates fall in these quadrants, respectively, representing cases where collective endorsement is genuinely low. The both-reject quadrant has the lowest correct-chain rate (27%), indicating that candidates with both low confidence and high disagreement are rarely on viable reasoning paths.

The separation is probabilistic, not deterministic. All quadrants contain both green and red points, visually confirming that consensus scores function optimally as rank ing tools rather than binary classifiers. A candidate in the dispersionreject quadrant is at elevated risk but not guaranteed to be wrong which is why the fallback mechanism $( \bar { s } ^ { * } - \varDelta ^ { * }$ ranking) recovers value from rejected candidates when no alternative passes.

The scatter plot provides spatial grounding for three main paper claims: the visual separation confirms that high

![](images/53599f674e9f21980d57eef02afc262f0eb628b522af7b0425be655a60b18b32.jpg)  
Fig. 14: Mean consensus confidence $\bar { s } ^ { * }$ vs. consensus dispersion $\varDelta ^ { \ast }$ for candidate steps on 3DSRBench. Green: correct chains; red: incorrect chains. Dashed lines: $\tau = 0 . 6 , \epsilon = 0 . 1$ . Inset: per-quadrant counts and correct-chain rates.

$\varDelta ^ { \ast }$ is diagnostic of reasoning instability (Proposition 1); the 77.9% concentration in the accept region is consistent with the ∼15% fallback rate; and the dispersion-reject quadrant is indistinguishable from accept under simple averaging since both have $\bar { s } ^ { * } > \tau$ which demonstrates the dual criterion’s operational value, catching 109 candidates (46% incorrect) that confidence-only methods would admit.

Table 10: Ablation analysis decomposing rejection (filtering) and selection (ranking). Acc.: average candidates accepted per step (of 3). Rej.%: fraction of steps where all candidates are rejected, triggering fallback to continuous ranking.
<table><tr><td></td><td colspan="2">CV-Bench-2D</td><td colspan="3">CV-Bench-3D</td><td colspan="3">3DSRBench</td><td colspan="3">AI2D</td></tr><tr><td>Strategy</td><td>Acc./Rej.%</td><td>Accuracy</td><td></td><td>Acc./Rej.% Accuracy</td><td></td><td></td><td>Acc./Rej.%</td><td>Accuracy</td><td>Acc./Rej.%</td><td></td><td>Accuracy</td></tr><tr><td>VERDICT</td><td>2.64 / 9.1%</td><td>79.22</td><td></td><td>2.84 / 2.3%</td><td>82.34</td><td></td><td>2.54 / 12.0%</td><td>59.02</td><td></td><td>2.50 / 13.4%</td><td>83.14</td></tr><tr><td>No Rejection</td><td>3.00 / 0.0%</td><td>78.14</td><td>3.00</td><td>/ 0.0%</td><td>82.08</td><td>3.00</td><td>/ 0.0%</td><td>60.31</td><td>3.00</td><td>/0.0%</td><td>82.17</td></tr><tr><td>No Selection</td><td>2.64 9.1%</td><td>77.05</td><td></td><td>2.84 / 2.3%</td><td>80.42</td><td>2.54</td><td>/ 12.0%</td><td>57.84</td><td>2.50</td><td>13.4%</td><td>82.55</td></tr><tr><td>Raw Average</td><td>2.40 16.2%</td><td>76.27</td><td></td><td>2.70 / 5.4%</td><td>79.75</td><td>2.31</td><td>/ 17.2%</td><td>56.92</td><td>2.35</td><td>/ 17.6%</td><td>81.83</td></tr><tr><td>Random</td><td>3.00 / 0.0%</td><td>75.38</td><td></td><td>3.00 / 0.0%</td><td>77.81</td><td></td><td>3.00 / 0.0%</td><td>56.41</td><td>3.00</td><td>/ 0.0%</td><td>81.64</td></tr></table>

## 11.11 Rejection and Selection: Detailed Analysis

Here, we examine the underlying mechanism through candidate acceptance statistics. Table 10 presents acceptance counts, fallback rates, and accuracy jointly.

Rejection frequency scales with benchmark dificulty. Filtering activates nonuniformly: only 2.3% of steps trigger full rejection on CV-Bench-3D (strong base model, 76.39%) versus 13.4% on AI2D (hallucination-prone reasoning). This directly explains why removing rejection costs 0.97 points on AI2D but only 0.26 on CV-Bench-3D.

Filtering and ranking address complementary failure modes. VERDICT and No Selection share identical acceptance statistics (e.g., 2.64 candidates, 9.1% fallback on CV-Bench-2D), confirming that selection operates strictly downstream of filtering. Their accuracy gap (79.22% vs. 77.05% on CV-Bench-2D, 82.34% vs. 80.42% on CV-Bench-3D) isolates the contribution of intelligent ranking by ¯s<sup>∗</sup>. Conversely, comparing VERDICT to No Rejection isolates filtering’s contribution. That both ablations degrade through diferent magnitudes on diferent benchmarks confirms that the two mechanisms target distinct failure modes.

Consensus calibration accepts more and rejects better. VERDICT and Raw Average share the same dual-criterion architecture; they difer only in whether scores come from coupled consensus or simple averaging. The consensus formulation accepts 0.14–0.24 more candidates per step while requiring fallback 3–7 percentage points less often, yet achieves consistently higher accuracy (e.g., 79.22% vs. 76.27% on CV-Bench-2D, 59.02% vs. 56.92% on 3DSRBench). This dual advantage indicates that consensus scoring distinguishes genuine cross-modal conflict from ordinary confidence fluctuations more efectively than raw averaging.

The 3DSRBench exception. On 3DSRBench, No Rejection (60.31%) slightly outperforms VERDICT (59.02%) despite the latter’s 12.0% fallback rate. On spatially demanding tasks where cross-modal tension is inherent, the dispersion filter removes candidates carrying informative disagreement. No Rejection preserves these and ranks them via $\bar { s } ^ { * } - \varDelta ^ { * }$ , treating disagreement as a ranking signal rather than an exclusion criterion. This reinforces the conclusion that VERDICT scores are most powerful as ranking tools, and that $\tau = 0 . 6$ is chosen for cross-benchmark robustness rather than per-task optimality.

![](images/791193bee295bcbd702083bc2dc4eb4675b92100933347f8dd3c21697cea04d7.jpg)  
Fig. 15: Top: Final-answer accuracy on the question subset $\Omega _ { \mathrm { f a l l b a c k } }$ under three conditions: random candidate selection on fallback steps (grey), unverified base model on the same questions (orange), and the $\bar { s } ^ { * } - \varDelta ^ { * }$ fallback ranking (green). Annotations show the improvement of fallback ranking over random. Bottom: Fallback trigger rate.

## 11.12 Fallback Mechanism Analysis

The main paper notes (Section 6.3) that the fallback path ranking by $\bar { s } ^ { * } - \varDelta ^ { * }$ when no candidate passes the dual criterion is triggered on approximately 15% of steps. Here, we characterize when the fallback activates and how well it performs. We partition test questions into $\Omega _ { \mathrm { f a l l b a c k } }$ (fallback triggered on at least one step) and $\mathcal { Q } _ { \mathrm { n o r m a l } }$ (fallback never triggered). On $\Omega _ { \mathrm { f a l l b a c k } }$ , we compare three conditions: (1) unverified base model accuracy, (2) full VERDICT with $\bar { s } ^ { * } - \varDelta ^ { * }$ ranking on fallback steps, and (3) VERDICT with random selection on fallback steps. Comparing (2) and (3) isolates the fallback mechanism’s contribution. Figure 15 reports per-benchmark statistics. Three findings emerge.

Fallback questions are systematically harder. On every benchmark, the base model achieves substantially lower accuracy on $\Omega _ { \mathrm { f a l l b a c k } }$ than overall Fallback questions correspond to chains encountering at least one step where cross-modal tension is genuinely high, and no candidate achieves simultaneous high confidence and low dispersion. The dual criterion efectively identifies the hardest 10–20% of questions without ground-truth labels.

The fallback ranking outperforms random selection. On every benchmark, ${ \bar { s } } ^ { * } - \varDelta ^ { * }$ ranking outperforms random selection by 7.8–17.0 points. Since all non-fallback steps use identical selection, this improvement is attributable entirely to the ranking signal. The composite score extracts useful information even from rejected candidates, preferring those with higher confidence and lower disagreement consistent with the main paper’s finding that consensus scores function optimally as ranking tools.

Relationship to the No Rejection ablation. The fallback analysis explains the 3DSRBench anomaly in Section 11.11: No Rejection (60.31%) slightly outperforms VERDICT (59.02%). On the 12% of fallback-triggered steps, VER-DICT chooses from the rejected pool, achieving 47.83% on $\Omega _ { \mathrm { f a l l b a c k } }$ . No Rejection retains all candidates and ranks over the complete pool, which includes candidates the dispersion filter would reject, but that may be viable for spatially demanding tasks where cross-modal tension is informative. On the five benchmarks where fallback rates are lower, or the ranking is more efective, VERDICT’s dual criterion outperforms No Rejection.

## 11.13 Generalization Across Model Families

The main paper evaluates VERDICT with Qwen2.5-VL-7B as both the base reasoner and the verification backbone. A natural question is whether the gains depend on this specific model family or whether the consensus formulation transfers. We test this by applying VERDICT to three additional model families on 3DSRBench, using the same frozen Qwen2.5-VL-7B judges throughout. We deliberately keep the judges fixed rather than swapping in same-family judges for each base model: this is a stricter test of generalization, and it avoids the selfreinforcement bias that would arise if a model family scored its own reasoning style more favourably than an external evaluator would.

## VERDICT improves every model

family it is applied to. Figure 16 reports gains of +2.45 to +4.00 percentage points, with a mean of +3.23 across all four families. Three observations follow.

The gains transfer without adaptation. The verification judges are Qwen2.5-VL-7B instances with prompts and hyperparameters fixed to the values reported in the main paper. No component is retrained, re-prompted, or recalibrated for the new base models. The fact that the same frozen judges produce consistent gains across four architecturally dis-

![](images/77a8b3cde81737dee943f23aada73d27117a53843a55b7795c0669c713ab3245.jpg)  
Fig. 16: Cross-model generalization on 3DSRBench. The base reasoner varies across four model families; the three verification judges remain frozen Qwen2.5-VL-7B instances with identical prompts and hyperparameters.

tinct model families confirms the plug-in design principle stated in Section 3 of the main paper: the consensus formulation operates on the structure of crossmodal agreement, not on the idiosyncrasies of any particular model’s output distribution.

The consensus formulation compresses base model variability. The four base models span a 1.33-point range in unverified accuracy (55.11–56.44%). After verification, this range compresses to 0.22 points (58.89–59.11%). LLaVA and InternVL, starting from diferent baselines (56.44% and 55.33%), converge to identical verified accuracy (58.89%). Kimi, the weakest unverified model (55.11%), achieves the highest verified accuracy (59.11%) and the largest gain (+4.00). This compression is consistent with the judge scale analysis in Section 11.6: the consensus formulation provides more value when the base signal is noisier, because there is more cross-modal disagreement for the dispersion filter to exploit.

Cross-model judges are viable. The experiment uses Qwen-family judges to verify non-Qwen base models. The modality-specific prompts (visual grounding, logical consistency, contextual relevance) define evaluation criteria that are modelagnostic: whether a reasoning step is visually grounded does not depend on which model generated it. The consistent gains across families suggest that the scoring rubrics capture genuine reasoning quality rather than model-specific stylistic preferences. This has a practical implication: a single set of frozen judges can serve as a universal verification layer across heterogeneous base models, amortising the already-zero training cost further.

Variance compression as evidence for consensus robustness. Figure 16 reveals that prior to verification, the four base models span an accuracy range of 1.33 pp (55.11–56.44%), which compresses to just 0.22 pp after VERDICT is applied (58.89–59.11%). This compression is not an artefact of ceiling efects, as verified accuracies remain well below perfect. Rather, it reflects the consensus formulation’s tendency to exploit cross-modal disagreement when the base signal is noisier, consistent with the widening-margin pattern observed in the judge scale analysis (Section 11.6). We note that architecturally distinct models converge to near-identical verified accuracy, indicating that the choice of base model matters far less once consensus verification is in place. This finding corroborates VER-DICT’s plug-in design principle, which by construction does not assume or rely on any particular base model architecture.

Table 11: Total computational cost comparison. Training costs are approximate and drawn from the respective papers’ reported compute budgets. Inference cost is for a single evaluation pass over all six benchmarks. VER-DICT’s total cost is 13–50× lower than domain-specific alternatives.
<table><tr><td>Method</td><td>Training Inference (GPU-h) (GPU-h)</td><td>Total (GPU-h)</td></tr><tr><td>VisualPRM</td><td>~500-1000</td><td>~7~507-1007</td></tr><tr><td>DreamPRM</td><td>~200-500</td><td>~7~207-507</td></tr><tr><td>LLaVA-Critic</td><td>~100-300</td><td>~7 ~107-307</td></tr><tr><td>Sherlock</td><td>~150-400</td><td>~7 ~157-407</td></tr><tr><td>VERDICT (Ours)</td><td>0</td><td>~20 ~20</td></tr></table>

Table 12: Accuracy gain per unit of additional compute across training-free inference strategies. All accuracy numbers use the 6-benchmark average. Efficiency is defined as the accuracy gain over the base model divided by the multiplicative overhead(higher is better).
<table><tr><td>Method</td><td>Cost (× base)</td><td>Avg. Acc. (%)</td><td>Gain (pp)</td><td>Efficiency (pp/×)</td></tr><tr><td>Base Model</td><td>1.0×</td><td>66.31</td><td></td><td></td></tr><tr><td colspan="5">Training-free, response-level aggregation</td></tr><tr><td>Self-Refinement</td><td>3.0×</td><td>60.42</td><td>-5.89</td><td>-1.96</td></tr><tr><td>Self-Selector</td><td>3.0×</td><td>61.19</td><td>-5.12</td><td>-1.71</td></tr><tr><td>TTAug</td><td>3.0×</td><td>62.78</td><td>-3.53</td><td>-1.18</td></tr><tr><td>Self-Consistency</td><td>3.0×</td><td>64.61</td><td>-1.70</td><td>-0.57</td></tr><tr><td>Best-of-3, random pick</td><td>3.0×</td><td>66.87</td><td>+0.56</td><td>0.19</td></tr><tr><td>Self-Synthesizer</td><td>3.0×</td><td>67.11</td><td>+0.80</td><td>0.26</td></tr><tr><td colspan="5">Training-free, step-level verification</td></tr><tr><td>Best-of-3 + Variance</td><td>3.80×</td><td>67.65</td><td>+1.34</td><td>0.35</td></tr><tr><td>Best-of-3 + Mean</td><td>3.80×</td><td>68.70</td><td>+2.39</td><td>0.63</td></tr><tr><td>VERDICT (Ours)</td><td>3.80×</td><td>70.15</td><td>+3.84</td><td>1.01</td></tr></table>

## 12 Detailed Computational Cost Analysis

The main paper reports a 3.80× wall-clock overhead under sequential execution. A simple reading of this number invites the conclusion that VERDICT is expensive. Here we argue that the more informative question is not how much additional compute does VERDICT use? but how much accuracy does each unit of additional compute buy? We decompose the overhead, compare accuracyper-compute against response-level and trajectory-level alternatives, and show that VERDICT’s overhead is either comparable to or lower than standard test-time scaling strategies that achieve smaller gains or that actively degrade performance.<sup>4</sup>

## 12.1 Accuracy per Unit of Additional Compute

The central metric for evaluating any test-time compute strategy is the accuracy gain per unit of additional inference cost. A method that spends 3× the base cost but loses accuracy is strictly worse than spending nothing. Table 12 compares VERDICT against inference-time strategies that are available without any training.

Three findings emerge from Table 12.

VERDICT achieves the highest compute eficiency among all training-free methods. At 1.01 accuracy points per unit of overhead, VERDICT is 1.6× more compute-eficient than the next-best training-free strategy (Mean scoring at $0 . 6 3 \mathrm { p p } / \times )$ and 2.7× more eficient than Self-Synthesizer, the strongest responselevel method $( 0 . 3 8 ~ \mathrm { p p } / \times )$ . The gain is not simply a consequence of spending more: VERDICT and Mean scoring incur identical costs (3.80×), yet VER-DICT extracts 1.45 additional accuracy points from the same compute budget, attributable entirely to the consensus formulation.

Response-level aggregation is less eficient. Four of the five TTS baselines spend 3.0× the base cost and lose accuracy: Self-Refinement (−5.89 pp), Self-Selector (−5.12 pp), TTAug (−3.53 pp), and Self-Consistency (−1.70 pp). Their eficiencies are negative, meaning that the additional compute is worse than wasted. These are measured averages across the same six benchmarks under similar conditions. The one positive method, Self-Synthesizer $( + 0 . 8 ~ \mathrm { p p } , 0 . 2 6 ~ \mathrm { p p } / \times )$ , also achieves less than VERDICT accuracy gain. The failure mode is structural: response-level methods cannot intercept step-level errors. They can only hope that independently sampled trajectories avoid the same failure.

At identical generation cost, the verification layer is the diference between degradation and gain. All five TTS baselines and VERDICT generate three candidate chains at 3.0× the base cost. The TTS methods stop there. VERDICT adds 0.80× for consensus verification. The five TTS methods average 63.29% accuracy 3.02 points below the unverified base model. Candidate diversity at 3.0×, without step-level verification, is a net negative investment. VERDICT reaches 70.15%. The 0.80× marginal overhead buys +3.04 points over Self-Synthesizer (67.11%), the strongest response-level strategy. This yields a marginal eficiency of 2.71/0.80 = 3.8 pp per unit of additional overhead the return attributable to the consensus formulation alone, net of candidate diversity.

## 12.2 Marginal Verification Cost

The 3.80× headline number conflates two independent costs that serve diferent purposes.

Candidate diversity: 3.0×. Generating 3 candidates instead of 1 accounts for 79% of the overhead. This cost is not specific to VERDICT; it is the price of candidate diversity that any best-of-N strategy pays. Our Random baseline (best-of-3 with random selection, no scoring) costs 3.0× and gains only +0.56 points, confirming that diversity alone is insuficient.

Consensus verification: 0.80×. The nine scoring calls (3 judges × 3 candidates) add only 0.80× because each scoring call produces 1–5 output tokens (a single scalar confidence) versus 50–200 tokens per generation call. The consensus computation itself (a closed-form linear solve) adds negligible cost $( < 0 . 0 0 1 \times )$

This 0.80× is the marginal cost of VERDICT’s contribution. It buys +3.28 accuracy points over Random (70.15% vs. 66.87%), yielding a marginal eficiency of 4.10 pp per unit of overhead substantially higher than any alternative’s total eficiency. The marginal cost comparison is even more informative against Self-Synthesizer, the strongest response-level strategy at 3.0×. Self-Synthesizer already performs non-trivial response-level aggregation synthesizing partial correct responses yet the 0.80× verification layer recovers an additional 2.71 accuracy points (70.15% vs. 67.44%). That this marginal gain exceeds the total gain of every response-level method confirms that step-level consensus verification is the eficiency-determining component of the pipeline, not candidate diversity.

## 12.3 Total Cost Comparison: Training Plus Inference

The inference overhead must be understood in the context of total cost, which includes the training pipeline that domain-specific critics require but VERDICT does not.

Domain-specific critics require ≈100–1000 GPU-hours of training (Monte Carlo rollouts for process supervision, human annotation pipelines, or reward model finetuning), after which their per-step inference cost is lower. However, the training cost must be amortised over all downstream uses, and critically, it must be repeated for each new domain. A single evaluation of VERDICT across all six benchmarks costs ∼20 GPU-hours with zero training, making its total cost 5–50× lower than domain-specific alternatives (Table 11). For practitioners evaluating across multiple domains the setting VERDICT is designed for the break-even point is never reached because training must be repeated per domain.

## 12.4 Mitigation Strategies and Future Directions

Several strategies can further reduce inference overhead without modifying the framework; we leave their implementation to future work.

Adaptive verification. Not every reasoning step requires full three-judge verification. Steps where the first candidate achieves very high consensus confidence (e.g., ¯s<sup>∗</sup> > 0.9) could bypass generation of additional candidates, reducing perstep cost from 3.80× to 1.80× or even 1.0×. Our fallback analysis (Section 11.12) shows that ∼85% of steps produce at least one candidate that passes the dual criterion; on many of these, the first candidate is likely suficient.

Speculative scoring. Candidates can be generated sequentially, with scoring applied as each is produced. If the first candidate achieves a high consensus, generation of the remaining two can be aborted, preserving the verification guarantee while reducing average-case cost.

Cheaper judge models. The judge scale analysis (Section 11.6) shows that even 2B judges maintain the widening margin over Mean aggregation. With 2B judges, the scoring overhead drops from 0.80× to approximately 0.25×, bringing total sequential overhead to approximately 3.25×.

Parallelisation. All nine scoring calls are independent. With batched generation and three GPUs (one per judge), the wall-clock overhead drops to approximately 1.59×, where the residual cost is dominated by generating three candidates in a single batched forward pass: a cost shared by any best-of-N strategy.

Contextualising the overhead. The 3.80× sequential overhead should be contextualised against the alternative of regenerating entire trajectories, which is the standard remedy when step-level verification is unavailable. Five-trajectory majority voting costs 3.0× with no guarantee of finding a correct trajectory and no step-level error interception. VERDICT at 3.80× provides step-level disagreement-aware verification that neither trajectory-level methods nor trained reward models can ofer at comparable cost.

## 13 Closed-Form Derivation

The consensus scores computed at Line 17 of Algorithm 1 are the operational core of VERDICT: they determine which candidate reasoning steps are accepted and how accepted steps are ranked (Lines 18–25). The main paper states that the coupled scoring system in Eq. (3) admits a closed-form solution. Here we provide the full derivation: the explicit matrix construction, the invertibility argument, and a scalar reduction that yields each consensus score in closed form. Beyond completeness, this derivation has two practical implications: it confirms that the consensus computation adds negligible overhead to each reasoning step (a single scalar division followed by three multiplications, as shown in Eqs. 16–17, and it establishes the stubbornness-weighted sum invariant (Section 13.4) that underpins the mean-preservation property discussed in Section 4 of the main paper.

## 13.1 Matrix Construction

Specialising Eq. (3) of the main paper to $m = 3$ agents $\{ V , L , C \}$ with stubbornness parameters $\lambda _ { V } , \lambda _ { L } , \lambda _ { C } > 0$ gives three coupled equations:

$$
\begin{array} { r } { \left( 1 + \lambda _ { V } \right) s _ { V } ^ { * } - \frac { 1 } { 2 } s _ { L } ^ { * } - \frac { 1 } { 2 } s _ { C } ^ { * } = \lambda _ { V } \hat { s } _ { V } , } \end{array}\tag{7}
$$

$$
\begin{array} { r } { - \frac { 1 } { 2 } s _ { V } ^ { * } + \left( 1 + \lambda _ { L } \right) s _ { L } ^ { * } - \frac { 1 } { 2 } s _ { C } ^ { * } = \lambda _ { L } \hat { s } _ { L } , } \end{array}\tag{8}
$$

$$
\begin{array} { r } { - \frac { 1 } { 2 } s _ { V } ^ { * } \ - \ \frac { 1 } { 2 } s _ { L } ^ { * } \ + \ \left( 1 + \lambda _ { C } \right) s _ { C } ^ { * } = \lambda _ { C } \hat { s } _ { C } . } \end{array}\tag{9}
$$

In matrix form, $\mathbf { A } \mathbf { s } ^ { * } = \mathbf { b }$ , with

$$
\mathbf { A } = \left( \begin{array} { c c c } { 1 + \lambda _ { V } } & { - \frac { 1 } { 2 } } & { - \frac { 1 } { 2 } } \\ { - \frac { 1 } { 2 } } & { 1 + \lambda _ { L } } & { - \frac { 1 } { 2 } } \\ { - \frac { 1 } { 2 } } & { - \frac { 1 } { 2 } } & { 1 + \lambda _ { C } } \end{array} \right) , \qquad \mathbf { b } = \left( \begin{array} { c } { \lambda _ { V } \hat { s } _ { V } } \\ { \lambda _ { L } \hat { s } _ { L } } \\ { \lambda _ { C } \hat { s } _ { C } } \end{array} \right) .\tag{10}
$$

## 13.2 Invertibility: Formal Proof

We prove that the coeficient matrix A in Eq. (10) is positive definite for every $\left( \lambda _ { V } , \lambda _ { L } , \lambda _ { C } \right)$ with $\lambda _ { i } > 0$ , through three independent arguments of increasing strength. We then extend the result to general m and verify empirically across the ablation range.

Argument 1: Strict Diagonal Dominance. Row i of A has diagonal entry $a _ { i i } =$ $1 + \lambda _ { i }$ and $m { - } 1 = 2$ of-diagonal entries each of magnitude $\frac { 1 } { 2 }$ , so the of-diagonal row sum is 1. Since $1 + \lambda _ { i } > 1$ for all $\lambda _ { i } > 0$ , every row satisfies

$$
| a _ { i i } | = 1 + \lambda _ { i } > 1 = \sum _ { j \neq i } \left| a _ { i j } \right| ,\tag{11}
$$

so A is strictly diagonally dominant. By the Levy–Desplanques theorem, A is non-singular.

Argument 2: Gershgorin Disks and Positive Definiteness. The Gershgorin disk for row i is centred at $1 + \lambda _ { i }$ with radius $r _ { i } = 1$ , so every eigenvalue $\mu$ of A satisfies $| \mu - ( 1 + \lambda _ { i } ) | \leq 1$ for some i. This yields the two-sided bound

$$
\lambda _ { \operatorname* { m i n } } ( \mathbf { A } ) \ \geq \ \operatorname* { m i n } _ { i } \left( 1 + \lambda _ { i } \right) - 1 = \operatorname* { m i n } _ { i } \lambda _ { i } > 0 .\tag{12}
$$

Since all eigenvalues are strictly positive, A is positive definite. The bound is tight in the symmetric case $( \lambda _ { i } ~ = ~ \lambda$ for all $i ,$ verified numerically for $\lambda \ \in$ $\{ 0 . 0 1 , 0 . 1 , 1 . 0 , 1 0 . 0 \} )$ . For our parameters $( \lambda _ { V } , \lambda _ { L } , \lambda _ { C } ) = ( 1 . 5 , 1 . 0 , 0 . 8 )$ , the bound gives 0.8 while the true minimum eigenvalue is 1.048.

Argument 3: Explicit Determinant Decomposition. Expanding the $3 \times 3$ determinant along the first row and substituting $\alpha = \lambda _ { V } , \beta = \lambda _ { L } , \gamma = \lambda _ { C }$ yields

$$
{ \bf \Gamma } \mathrm { \operatorname * { d e t } } ( \mathbf { A } ) = \underbrace { \frac { 3 } { 4 } ( \alpha + \beta + \gamma ) } _ { \mathrm { l i n e a r ~ t e r m s } } + \underbrace { \alpha \beta + \alpha \gamma + \beta \gamma } _ { \mathrm { p a i r w i s e ~ p r o d u c t s } } + \underbrace { \alpha \beta \gamma } _ { \mathrm { t r i p l e ~ p r o d u c t } }  _ { \mathrm { t r i p l e ~ p r o d u c t } }\tag{13}
$$

Every term is a product of positive quantities, so det $( \mathbf { A } ) > 0$ for all $\alpha , \beta , \gamma > 0$ no boundary analysis or limit argument is needed. With $( \lambda _ { V } , \lambda _ { L } , \lambda _ { C } ) = ( 1 . 5 , 1 . 0 , 0 . 8 )$ det $( \mathbf { A } ) = { \textstyle { \frac { 3 } { 4 } } } ( 3 . 3 ) + ( 1 . 5 + 1 . 2 + 0 . 8 ) + 1 . 2 = 2 . 4 7 5 + 3 . 5 + 1 . 2 = 7 . 1 7 5$ , matching direct computation.

Extension to General m. For m agents, the coeficient matrix $\mathbf { A } ^ { ( m ) }$ has diagonal entries $1 + \lambda _ { i }$ and of-diagonal entries $- 1 / ( m { - } 1 )$ . The of-diagonal row sum is $( m - 1 ) \cdot \frac { 1 } { m - 1 } = 1$ , independent of m, so the diagonal dominance condition $1 + \lambda _ { i } > 1$ holds for all $m \geq 2$ and all $\lambda _ { i } ~ > ~ 0$ , and the Gershgorin bound $\lambda _ { \operatorname* { m i n } } ( \mathbf { A } ^ { ( m ) } ) \geq$ min $\lambda _ { i }$ extends unchanged.

Table 13: Positive definiteness verification across the stubbornness parameter range explored in Section 6.2 of the main paper. All $5 ^ { 3 } ~ = ~ 1 2 5$ triples from $\{ 0 . \dot { 3 } , 0 . 8 , \bar { 1 . 0 } , 1 . 5 , 2 . 0 \} ^ { 3 }$ are evaluated.
<table><tr><td>Quantity Value</td></tr><tr><td>Number of triples tested 125</td></tr><tr><td>Minimum determinant 0.972</td></tr><tr><td>Minimum eigenvalue 0.300</td></tr><tr><td>Maximum condition number 6.00</td></tr><tr><td>All positive definite? Yes</td></tr></table>

Empirical Verification Across Ablation Range. Table 13 confirms that every stubbornness configuration in the ablation range yields a well-conditioned, positivedefinite system. The worst-case condition number $( \kappa ( \mathbf { A } ) = \lambda _ { \mathrm { m a x } } / \lambda _ { \mathrm { m i n } } )$ is $\kappa = 6 . 0$ at $\lambda _ { i } = 0 . 3$ for all agents, which remains modest; the minimum eigenvalue (0.300) matches the Gershgorin bound exactly in the symmetric case. As any $\lambda _ { i } \to 0 ^ { + }$ 2 the condition number grows as $\begin{array} { r } { \kappa \sim 1 . 5 / \operatorname* { m i n } _ { i } \lambda _ { i } ; } \end{array}$ at $\lambda _ { i } = 1 0 ^ { - 3 }$ (far below any practical setting), $\kappa \approx 1 5 0 0$ : large but numerically tractable. True degeneracy occurs only at $\lambda _ { i } = 0$ , corresponding to an agent with no private evidence a degenerate case excluded by design.

The consensus matrix is therefore positive definite, and the closed-form solution exists and is unique, for any number of agents $m \geq 2$ with any positive stubbornness parameters. We now derive that solution explicitly.

## 13.3 Scalar Reduction and Explicit Solution

Define $S = s _ { V } ^ { * } + s _ { L } ^ { * } + s _ { C } ^ { * }$ . Adding $s _ { i } ^ { * } / ( m - 1 )$ to both sides of each row of Eq. (10) and collecting terms yields, for each agent i,

$$
s _ { i } ^ { * } = \frac { S + 2 \lambda _ { i } \hat { s } _ { i } } { 3 + 2 \lambda _ { i } } .\tag{14}
$$

Interpretation. Equation 14 says that each agent’s consensus score is a weighted blend of two quantities: the collective score sum $S$ (representing what the group believes) and the agent’s own raw evidence $\hat { s } _ { i }$ (representing what it privately observed). The stubbornness parameter $\lambda _ { i }$ controls the mixing ratio: a more stubborn agent (higher $\lambda _ { i } )$ places more weight on its own evidence and less on the group, while a flexible agent is pulled closer to the collective position.

Define the consensus weight $w _ { i } = 1 / ( 3 + 2 \lambda _ { i } )$ . Summing Eq. (14) over the three agents gives a single scalar equation in S:

$$
S = S \sum _ { i } w _ { i } + 2 \sum _ { i } \lambda _ { i } w _ { i } \hat { s } _ { i } ,\tag{15}
$$

which solves to

$$
\boxed { S = \frac { 2 \displaystyle \sum _ { i } \lambda _ { i } w _ { i } \hat { s } _ { i } } { 1 - \displaystyle \sum _ { i } w _ { i } } } \qquad \mathrm { w h e r e } \quad w _ { i } = \frac { 1 } { 3 + 2 \lambda _ { i } } .\tag{16}
$$

Interpretation. Equation 16 expresses the total consensus score $S$ as a closedform function of the raw scores alone. Each agent’s contribution to S is scaled by its consensus weight $w _ { i } .$ , which decreases with stubbornness: stubborn agents influence the collective sum less because they absorb less of the group signal and therefore redirect less information back into the pool.

The denominator $1 - \textstyle \sum _ { i }$ w<sub>i</sub> is strictly positive because $\begin{array} { r } { w _ { i } < \frac { 1 } { 3 } } \end{array}$ for all $\lambda _ { i } > 0 ,$ so $\textstyle \sum _ { i } w _ { i } \ < \ 1$ . Each consensus score then follows from back-substitution into Eq. (14):

$$
s _ { i } ^ { * } = w _ { i } S + 2 \lambda _ { i } w _ { i } \hat { s } _ { i } .\tag{17}
$$

Interpretation. Equation 17 is the final answer: once $S$ is known from Eq. $^ { 1 6 , }$ each agent’s consensus score follows from a single multiplication. The first term $( w _ { i } S )$ is the agent’s share of the collective pool; identical in structure for all agents but scaled by their individual consensus weights. The second term $\left( 2 \lambda _ { i } w _ { i } \hat { s } _ { i } \right)$ is the agent’s private anchor, pulling its consensus score toward its raw evidence in proportion to its stubbornness. Together, the two terms make explicit how each consensus score decomposes into a “group pull” and a “private anchor,” with $\lambda _ { i }$ governing the balance.

Eqs. (16)–(17) give the complete closed-form solution. No matrix inversion or iterative procedure is required: one scalar division (Eq. 16) followed by three multiplications (Eq. 17).

## 13.4 Stubbornness-Weighted Sum Invariant

Summing Eqs. (7)–(9):

$$
{ \sum _ { i } } ( 1 + { \lambda _ { i } } ) s _ { i } ^ { * } - { \frac { 1 } { m - 1 } } \sum _ { i } { \sum _ { j \ne i } } s _ { j } ^ { * } = \sum _ { i } { \lambda _ { i } } \hat { s } _ { i } .\tag{18}
$$

The double sum $\textstyle \sum _ { i } \sum _ { j \neq i } s _ { j } ^ { * }$ counts each $s _ { j } ^ { * }$ exactly m−1 times, so it equals (m−1) S. Substituting and simplifying:

$$
\boxed { \sum _ { i } \lambda _ { i } s _ { i } ^ { * } = \sum _ { i } \lambda _ { i } \hat { s } _ { i } }\tag{19}
$$

The consensus computation preserves the stubbornness-weighted sum of scores. When all $\lambda _ { i }$ are equal, this reduces to preservation of the unweighted mean $\bar { s } ^ { * } = \bar { \hat { s } }$ . For heterogeneous $\lambda _ { i } .$ the unweighted mean is approximately but not exactly preserved; the shift is bounded by

$$
\left| { \bar { s } } ^ { * } - { \bar { \hat { s } } } \right| \leq { \frac { \operatorname* { m a x } _ { i } \lambda _ { i } - \operatorname* { m i n } _ { i } \lambda _ { i } } { \operatorname* { m i n } _ { i } \lambda _ { i } } } \cdot { \frac { 1 } { m } } \sum _ { i } \left| s _ { i } ^ { * } - { \hat { s } } _ { i } \right| ,\tag{20}
$$

which is small when the stubbornness parameters are of comparable magnitude. With $( \lambda _ { V } , \lambda _ { L } , \lambda _ { C } ) = ( 1 . 5 , 1 . 0 , 0 . 8 )$ , the multiplicative factor is $( 1 . 5 - 0 . 8 ) / 0 . 8 =$ 0.875, and the mean shift is empirically below 0.01 for all score vectors encountered across our six benchmarks.

Why does this invariant matter in practice? In plain terms, the invariant guarantees that the consensus computation reshufles confidence among agents without inflating or deflating the overall confidence level, which is what allows the acceptance thresholds τ and ϵ to remain meaningful and transferable across tasks without recalibration. To see why the stubbornness-weighted sum invariant is not merely a mathematical convenience, consider what would happen if it did not hold. The confidence threshold $\bar { s } ^ { * } > \tau$ in the dual acceptance criterion (Eq. (4), main paper) is set under the assumption that the consensus computation does not systematically shift the collective confidence level. If the consensus introduced an upward bias in $\bar { s } ^ { * }$ , candidates with genuinely mediocre raw support could be pushed above $\tau .$ , and the confidence check would start admitting steps that no individual agent found compelling. Conversely, a downward bias would cause the system to reject steps that all three agents endorsed, forcing the fallback path $( \bar { s } ^ { * } - \varDelta ^ { * }$ ranking) to activate far more often than the ∼15% reported in the main paper. Section 11.12 shows that the fallback path, while efective, achieves lower accuracy than the primary dual-criterion path; an inflated fallback rate would erode the gains documented there. In either direction, the threshold $\tau = 0 . 6$ would need to be recalibrated jointly with ϵ for every stubbornness configuration, destroying the property that makes VERDICT a plug-in verifier. The invariant guarantees that the consensus redistributes scores around a stable collective level, so τ and ϵ can be set once and transferred across benchmarks without concern that the stubbornness parameters have shifted the operating point of the acceptance criterion.

## 13.5 Extension to General m

The derivation generalises directly to m agents. The m×m coeficient matrix has diagonal entries $( 1 + \lambda _ { i } )$ and of-diagonal entries $- 1 / ( m { - } 1 )$ , and remains strictly diagonally dominant for all $\lambda _ { i } > 0$ . The scalar reduction gives

$$
{ s } _ { i } ^ { * } = \frac { S + ( m - 1 ) \lambda _ { i } \hat { s } _ { i } } { m + ( m - 1 ) \lambda _ { i } } , \qquad S = \frac { ( m - 1 ) \sum _ { i } \lambda _ { i } w _ { i } \hat { s } _ { i } } { 1 - \sum _ { i } w _ { i } } , \qquad w _ { i } = \frac { 1 } { m + ( m - 1 ) \lambda _ { i } } ,\tag{21}
$$

and the invariant $\begin{array} { r } { \sum _ { i } \lambda _ { i } s _ { i } ^ { * } = \sum _ { i } \lambda _ { i } \hat { s } _ { i } } \end{array}$ holds for every $m \geq 2$

## 14 Existence and Uniqueness of the Consensus Fixed Point

Proposition 2. The coupled scoring system is defined by the objective function

$$
u _ { i } \big ( s _ { i } , s _ { - i } \big ) = - \big ( s _ { i } - \bar { s } _ { - i } \big ) ^ { 2 } - \lambda _ { i } \big ( s _ { i } - \hat { s } _ { i } \big ) ^ { 2 }
$$

admits a unique fixed point.

Proof. The result follows directly from Rosen’s uniqueness result [32] for concave interaction systems. We verify the required conditions below.

(1) Compact and convex score space. Each agent’s score space $s _ { i } \in [ 0 , 1 ]$ is compact and convex.

(2) Continuity. The objective function $u _ { i }$ is quadratic and therefore continuous in all arguments.

(3) Strict concavity. Taking derivatives with respect to $s _ { i } ,$ , we obtain

$$
\frac { \partial u _ { i } } { \partial s _ { i } } = - 2 \left( s _ { i } - { \bar { s } } _ { - i } \right) - 2 \lambda _ { i } \left( s _ { i } - { \hat { s } } _ { i } \right) ,\tag{22}
$$

$$
\frac { \partial ^ { 2 } u _ { i } } { \partial s _ { i } ^ { 2 } } = - 2 - 2 \lambda _ { i } = - 2 ( 1 + \lambda _ { i } ) < 0 \quad \forall \lambda _ { i } > 0 .\tag{23}
$$

Thus, $u _ { i }$ is strictly concave in each agent’s own score. By Rosen’s uniqueness result, the coupled system admits a unique fixed point.

## 15 Proposition 1: Detailed Proof and Worked Examples

Proposition 1 in the main paper establishes that consensus dispersion $\varDelta ^ { \ast }$ captures the cross-agent interaction structure that is not separable per-agent measure can replicate. We provide the full proof below, followed by a step-by-step numerical walkthrough of the two scores vectors used in the proof, so that every intermediate quantity can be verified by hand.

## 15.1 Formal Statement and Proof

Proposition 3 (Consensus Dispersion Is Not Recoverable from Weighted Averages). For any fixed weight vector $\mathbf { w } ,$ there exist score vectors ˆs $\neq \hat { \mathbf { s } } ^ { \prime }$ such that $\mathbf { w } ^ { \top } \hat { \mathbf { s } } = \mathbf { w } ^ { \top } \hat { \mathbf { s } } ^ { \prime }$ yet $\varDelta ^ { \ast } ( \hat { \mathbf { s } } ) \neq \varDelta ^ { \ast } ( \hat { \mathbf { s } } ^ { \prime } )$ . As for any agent i, the consensus residual $| s _ { i } ^ { * } - \bar { s } ^ { * } |$ depends on the $f u l l$ raw score vector $\hat { \bf s } = ( \hat { s } _ { 1 } , \dots , \hat { s } _ { m } )$ , not solely on $\hat { s } _ { i }$ Consequently, $\begin{array} { r } { \varDelta ^ { * } = \frac { 1 } { m } \sum _ { i } \left| s _ { i } ^ { * } - \bar { s } ^ { * } \right| } \end{array}$ cannot be decomposed as a separable function $\textstyle \sum _ { i } f _ { i } ( { \hat { s } } _ { i } )$ for any choice of per-agent functions $\{ f _ { i } \}$

Proof. From the closed-form solution (Section 13, Eq. 14), each consensus score satisfies

$$
s _ { i } ^ { * } = \frac { S + 2 \lambda _ { i } \hat { s } _ { i } } { 3 + 2 \lambda _ { i } } , \qquad S = \sum _ { j } s _ { j } ^ { * } .\tag{24}
$$

Since S depends on all raw scores through the scalar equation (16), the partial derivative $\partial s _ { i } ^ { * } / \partial \hat { s } _ { j }$ is generically nonzero for $i \neq j$ whenever $\lambda _ { j }$ is finite. Concretely, diferentiating Eq. (16) gives

$$
\frac { \partial S } { \partial \hat { s } _ { j } } = \frac { 2 \lambda _ { j } w _ { j } } { 1 - \sum _ { k } w _ { k } } > 0 \qquad \forall \lambda _ { j } > 0 ,\tag{25}
$$

where $w _ { j } = 1 / ( 3 + 2 \lambda _ { j } )$ . Substituting into $\operatorname { E q } .$ . (24):

$$
{ \frac { \partial s _ { i } ^ { * } } { \partial { \hat { s } } _ { j } } } = { \frac { w _ { i } } { 1 } } \cdot { \frac { \partial S } { \partial { \hat { s } } _ { j } } } = { \frac { 2 \lambda _ { j } w _ { i } w _ { j } } { 1 - \sum _ { k } w _ { k } } } > 0 \qquad { \mathrm { f o r ~ } } i \neq j .\tag{26}
$$

Agent i’s consensus position shifts when agent $j ^ { \circ } \mathrm { s }$ raw evidence changes. The residual $\lvert s _ { i } ^ { * } - \bar { s } ^ { * } \rvert$ therefore depends on the full vector $\hat { \mathbf { s } } ,$ not on $\hat { s } _ { i }$ alone, and $\varDelta ^ { \ast }$ is not separable.

To make this concrete, we exhibit two score vectors with identical means that produce opposite acceptance decisions. Take $\hat { \mathbf { s } } = ( 0 . 9 , 0 . 2 , 0 . 9 )$ and $\hat { \mathbf { s } } ^ { \prime } = ( 0 . 7 , 0 . 6 , 0 . 7 )$ both with raw mean $\frac { 2 } { 3 }$ . The detailed computation below yields $\varDelta ^ { * } ( \hat { \mathbf { s } } ) \approx 0 . 1 3 > \epsilon$ and $\Delta ^ { * } ( \hat { \mathbf { s } } ^ { \prime } ) \approx 0 . 0 2 < \epsilon \mathrm { : }$ opposite acceptance decisions under the dual criterion in Eq. (4) of the main paper, despite identical means. (the number .21 in the main paper, Sec 4, was a typo).

No separable measure $\textstyle \sum _ { i } f _ { i } ( { \hat { s } } _ { i } )$ can replicate this, since it would need $f _ { 2 } ( 0 . 2 )$ to produce high dispersion (in ˆs) and $f _ { 2 } ( 0 . 6 )$ to produce low dispersion $( \mathrm { i n } \ \hat { \mathbf { s } } ^ { \prime } )$ , but the distinction arises precisely because the consensus solution couples agent $2 \mathrm { { ^ { \circ } s } }$ residual to agents 1 and 3’s raw scores via Eq. (26), not because $f _ { 2 }$ is evaluated at diferent inputs.

## 15.2 Worked Example $\mathbf { 1 } { : } \hat { \mathbf { s } } = ( \mathbf { 0 } . \mathbf { 9 } , \mathbf { 0 . 2 } , \mathbf { 0 . 9 } )$ – Cross-Modal Conflict

Table 14 provides a side-by-side comparison of the two score vectors used in the proof. Both share the same raw mean $\begin{array} { r } { \bar { \hat { s } } = \frac { 2 } { 3 } } \end{array}$ , yet the consensus formulation produces opposite acceptance decisions which is the central claim of Proposition 1. The detailed step-by-step derivations follow.

Table 14: Side-by-side comparison of the two Proposition 1 examples. Both share raw mean $\textstyle { \frac { 2 } { 3 } }$ , yet produce opposite acceptance decisions under Eq. (4) of the main paper.
<table><tr><td> $\hat { \mathbf { s } } = ( 0 . 9 , 0 . 2 , 0 . 9 )$ </td><td> $\hat { \mathbf { s } } ^ { \prime } = ( 0 . 7 , 0 . 6 , 0 . 7 )$ </td></tr><tr><td>Raw mean  $\bar { \hat { s } }$   $\mathbf { s } ^ { \ast }$ </td><td>0.667 0.667</td></tr><tr><td></td><td>(0.788, 0.485, 0.754) (0.684, 0.641, 0.679)</td></tr><tr><td> $\overline { { s } } ^ { \ast }$ </td><td>0.668</td></tr><tr><td>0.127</td><td>0.018</td></tr><tr><td>Max agent shift maxi  $| s _ { i } ^ { * } - \hat { s } _ { i } |$ </td><td>0.04 (L)</td></tr><tr><td> $\overline { { s } } ^ { * } > \tau ?$ </td><td>√</td></tr><tr><td> $\varDelta ^ { \ast } < \epsilon ?$ </td><td>√</td></tr><tr><td>Decision</td><td>REJECT ACCEPT</td></tr></table>

Parameters: $( \lambda _ { V } , \lambda _ { L } , \lambda _ { C } ) = ( 1 . 5 , 1 . 0 , 0 . 8 )$ ， $\tau = 0 . 6 .$ , ϵ = 0.1.

Step 1: Construct b = λ ⊙ ˆs. $b _ { V } = 1 . 5 \times 0 . 9 = 1 . 3 5$ $b _ { L } = 1 . 0 \times 0 . 2 = 0 . 2 0$ $b _ { C } = 0 . 8 \times 0 . 9 = 0 . 7 2$

Step 2: Compute consensus weights.

$$
\begin{array} { c c c } { \displaystyle { w _ { V } = \frac { 1 } { 3 + 2 ( 1 . 5 ) } = \frac { 1 } { 6 } \approx 0 . 1 6 6 7 , } } & { \displaystyle { w _ { L } = \frac { 1 } { 3 + 2 ( 1 . 0 ) } = \frac { 1 } { 5 } = 0 . 2 0 0 0 , } } \\ { \displaystyle { w _ { C } = \frac { 1 } { 3 + 2 ( 0 . 8 ) } = \frac { 1 } { 4 . 6 } \approx 0 . 2 1 7 4 . } } \end{array}
$$

$$
\begin{array} { r } { \sum _ { i } w _ { i } = 0 . 5 8 4 1 . } \end{array}
$$

Step 3: Compute the score sum $S .$ The numerator terms are $\lambda _ { V } w _ { V } \hat { s } _ { V } =$ $1 . 5 \times 0 . 1 6 6 7 \times 0 . 9 = 0 . 2 2 5 0 , \lambda _ { L } w _ { L } \hat { s } _ { L } = 1 . 0 \times 0 . 2 0 0 0 \times 0 . 2 = 0 . 0 4 0 0 , \lambda _ { C } w _ { C } \hat { s } _ { C } = 0 . 0 4 0 0 , L _ { \mathrm { ~ w } } w _ { L } \hat { s } _ { C } = 0 . 0 4 0 0 0 \times 0 . 2 5 \times 0 . 0 5 0 0$ $0 . 8 \times 0 . 2 1 7 4 \times 0 . 9 = 0 . 1 5 6 5 .$

$$
S = { \frac { 2 \times ( 0 . 2 2 5 0 + 0 . 0 4 0 0 + 0 . 1 5 6 5 ) } { 1 - 0 . 5 8 4 1 } } = { \frac { 2 \times 0 . 4 2 1 5 } { 0 . 4 1 5 9 } } = { \frac { 0 . 8 4 3 0 } { 0 . 4 1 5 9 } } = 2 . 0 2 7 .
$$

Step 4: Back-substitute to obtain $\mathbf { s } ^ { * }$ .

$$
s _ { V } ^ { * } = { \frac { 2 . 0 2 7 + 2 ( 1 . 5 ) ( 0 . 9 ) } { 6 } } = { \frac { 2 . 0 2 7 + 2 . 7 0 0 } { 6 } } = { \frac { 4 . 7 2 7 } { 6 } } = 0 . 7 8 8 ,
$$

$$
s _ { L } ^ { * } = { \frac { 2 . 0 2 7 + 2 ( 1 . 0 ) ( 0 . 2 ) } { 5 } } = { \frac { 2 . 0 2 7 + 0 . 4 0 0 } { 5 } } = { \frac { 2 . 4 2 7 } { 5 } } = 0 . 4 8 5 ,
$$

$$
s _ { C } ^ { * } = { \frac { 2 . 0 2 7 + 2 ( 0 . 8 ) ( 0 . 9 ) } { 4 . 6 } } = { \frac { 2 . 0 2 7 + 1 . 4 4 0 } { 4 . 6 } } = { \frac { 3 . 4 6 7 } { 4 . 6 } } = 0 . 7 5 4 .
$$

Interpretation. The outlying Logical agent is pulled from 0.20 to 0.49: a shift of $+ 0 . 2 9$ . The stubborn Visual agent $\left( \lambda _ { V } = 1 . 5 \right)$ shifts only −0.11, while the flexible Contextual agent $\left( \lambda _ { C } = 0 . 8 \right)$ shifts −0.15. The asymmetry is driven entirely by the stubbornness parameters.

Step 5: Compute mean consensus confidence.

$$
\begin{array} { r } { \bar { s } ^ { * } = \frac { 1 } { 3 } ( 0 . 7 8 8 + 0 . 4 8 5 + 0 . 7 5 4 ) = 0 . 6 7 6 . } \end{array}
$$

The raw mean is $\textstyle { \bar { \hat { s } } } = { \frac { 2 } { 3 } } \approx 0 . 6 6 7$ . The shift of +0.009 reflects the stubbornnessweighted invariant described in Section 13: $\begin{array} { r } { \sum _ { i } \lambda _ { i } s _ { i } ^ { * } = \sum _ { i } \lambda _ { i } \hat { s } _ { i } = 2 . 2 7 } \end{array}$ , while the unweighted mean shifts slightly because the $\lambda _ { i }$ are heterogeneous.

Step 6: Compute consensus dispersion.

$$
| s _ { V } ^ { * } - \bar { s } ^ { * } | = | 0 . 7 8 8 - 0 . 6 7 6 | = 0 . 1 1 2 ,
$$

$$
| s _ { L } ^ { * } - \bar { s } ^ { * } | = | 0 . 4 8 5 - 0 . 6 7 6 | = 0 . 1 9 0 ,
$$

$$
| s _ { C } ^ { * } - \bar { s } ^ { * } | = | 0 . 7 5 4 - 0 . 6 7 6 | = 0 . 0 7 8 .
$$

$$
\begin{array} { r } { \varDelta ^ { * } = \frac { 1 } { 3 } ( 0 . 1 1 2 + 0 . 1 9 0 + 0 . 0 7 8 ) = 0 . 1 2 7 . } \end{array}
$$

Note. The Logical agent contributes the largest residual (0.190), despite having the median stubbornness $( \lambda _ { L } = 1 . 0 )$ . This is because its raw score (0.2) is the most extreme outlier, and the consensus pull is insuficient to close the gap.

Step 7: Apply dual acceptance criterion. $\bar { s } ^ { * } = 0 . 6 7 6 > \tau = 0 . 6 ~ \check { s } \ \bullet \ \preceq$ $0 . 1 2 7 > \epsilon = 0 . 1 \pmb { \chi } .$ Decision: REJECT. The step has suficient collective confidence but unresolved cross-modal conflict.

## 15.3 Worked Example 2: $\hat { \mathbf { s } } ^ { \prime } = ( \mathbf { 0 } . 7 , \mathbf { 0 } . 6 , \mathbf { 0 } . 7 )$ – Modest Consensus

Step 1: $b _ { V } = 1 . 0 5 , \ b _ { L } = 0 . 6 0 , \ b _ { C } = 0 . 5 6 .$

Step 2: Consensus weights are identical (they depend only on $\lambda _ { i } ) \colon w _ { V } = 0 . 1 6 6 7 .$ $w _ { L } = 0 . 2 0 0 0 , w _ { C } = 0 . 2 1 7 4 .$

Step 3: $\lambda _ { V } w _ { V } \hat { s } _ { V } = 0 . 1 7 5 0 , \lambda _ { L } w _ { L } \hat { s } _ { L } = 0 . 1 2 0 0 , \lambda _ { C } w _ { C } \hat { s } _ { C } = 0 . 1 2 1 7 .$

$$
S = \frac { 2 \times 0 . 4 1 6 7 } { 0 . 4 1 5 9 } = 2 . 0 0 4 .
$$

Step 4:

$$
s _ { V } ^ { * } = \frac { 2 . 0 0 4 + 2 . 1 0 0 } { 6 } = \frac { 4 . 1 0 4 } { 6 } = 0 . 6 8 4 ,
$$

$$
s _ { L } ^ { * } = \frac { 2 . 0 0 4 + 1 . 2 0 0 } { 5 } = \frac { 3 . 2 0 4 } { 5 } = 0 . 6 4 1 { , }
$$

$$
s _ { C } ^ { * } = \frac { 2 . 0 0 4 + 1 . 1 2 0 } { 4 . 6 } = \frac { 3 . 1 2 4 } { 4 . 6 } = 0 . 6 7 9 .
$$

Interpretation. All three agents shift by less than 0.03 from their raw scores. The consensus system barely perturbs a score vector that already reflects genuine agreement.

Step 5: $\begin{array} { r } { \bar { s } ^ { * } = \frac 1 3 ( 0 . 6 8 4 + 0 . 6 4 1 + 0 . 6 7 9 ) \ = \ 0 . 6 6 8 . } \end{array}$ . Raw mean $= ~ 0 . 6 6 7 ;$ shift $= + 0 . 0 0 1$

Step 6:

$$
\begin{array} { r } { \left| s _ { V } ^ { * } - \bar { s } ^ { * } \right| = \left| 0 . 6 8 4 - 0 . 6 6 8 \right| = 0 . 0 1 6 , } \end{array}
$$

$$
| s _ { L } ^ { * } - \bar { s } ^ { * } | = | 0 . 6 4 1 - 0 . 6 6 8 | = 0 . 0 2 7 ,
$$

$$
| s _ { C } ^ { * } - \bar { s } ^ { * } | = | 0 . 6 7 9 - 0 . 6 6 8 | = 0 . 0 1 1 .
$$

$$
\begin{array} { r } { \varDelta ^ { * } = \frac { 1 } { 3 } ( 0 . 0 1 6 + 0 . 0 2 7 + 0 . 0 1 1 ) = 0 . 0 1 8 . } \end{array}
$$

Step 7: $\bar { s } ^ { * } = 0 . 6 6 8 > 0 . 6 ~ \check { < } \mathrm { ~ \ } \varDelta ^ { * } = 0 . 0 1 8 < 0 . 1 ~ \checkmark$ . Decision: ACCEPT.   
Modest but genuine cross-modal agreement.

## 15.4 Comparison and Implications

Returning to Table 14, three observations follow directly from the side-by-side comparison:

Opposite decisions from identical means. Any weighted-average method maps both vectors to a single scalar above $\tau = 0 . 6$ and therefore accepts both. The consensus formulation distinguishes them because $\varDelta ^ { \ast }$ reflects how much the agents failed to converge: a property that depends on the full interaction structure, not on any per-agent summary.

Asymmetric agent influence. In ˆs, the Logical agent’s residual $\left( \left| s _ { L } ^ { * } - \bar { s } ^ { * } \right| = 0 . 1 9 0 \right)$ dominates $\varDelta ^ { \ast }$ . In $\hat { \mathbf { s } } ^ { \prime } ,$ it contributes only 0.027. The factor-of-seven diference arises not from the Logical agent’s stubbornness (which is the same in both cases) but from how its raw score interacts with the other agents’ positions through the coupled solve. This is precisely the non-separability that Proposition 1 establishes.

Stubbornness mediates the consensus pull. The Visual agent shifts −0.11 in ˆs versus $- 0 . 0 2$ in $\hat { \mathbf { s } } ^ { \prime } ;$ its high stubbornness $\left( \lambda _ { V } \ : = \ : 1 . 5 \right)$ anchors it close to its raw score in both cases. The Contextual agent $( \lambda _ { C } = 0 . 8 )$ shifts −0.15 versus $- 0 . 0 2$ : more responsive to consensus pressure. This heterogeneous pull is what distinguishes the coupled consensus from symmetric variance, which would treat both agents’ deviations identically.

## 16 Consensus Dispersion vs. Weighted Averaging: Illustrative Scenarios

Proposition 1 in the main paper establishes that consensus dispersion $\varDelta ^ { \ast }$ captures cross-agent interaction structure that no separable per-agent measure can replicate. We expand on this result here by enumerating representative score configurations that concretely illustrate when and why $\varDelta ^ { \ast }$ diverges from simpler aggregation schemes. This analysis complements the formal proof by showing that the divergence is not a pathological edge case but arises naturally across the range of verification scenarios encountered in practice.

Setup. We consider four canonical score configurations for three agents $( V , L , C )$ each representing a qualitatively distinct verification state. For each configuration we report: the raw mean $\bar { \hat { s } } ,$ which determines any weighted-average acceptance decision under threshold τ; the consensus dispersion $\varDelta ^ { \ast }$ , computed using Eq. (2) of the main paper with $\lambda _ { V } = 1 . 5 , \lambda _ { L } = 1 . 0 , \lambda _ { C } = 0 . 8 ;$ and the resulting decision under our dual criterion $( \bar { s } ^ { * } > \tau \ : = \ : 0 . 6$ and $\varDelta ^ { \ast } < \epsilon = 0 . 1 )$ . The weighted average decision is derived by applying only the confidence threshold $\begin{array} { r } { \bar { \hat { s } } > \tau , } \end{array}$ , which is the implicit acceptance rule of any mean-based aggregation scheme.

Scenario analysis. Scenario 1 (unanimous confidence) represents the ideal case: all three agents strongly endorse the reasoning step from their respective modalityspecific perspectives. Both methods correctly accept, and no divergence arises. This serves as a sanity check confirming that the VERDICT does not introduce spurious rejections when evidence is coherent.

Scenarios 2 and 3 are the critical pair and the central illustration of Proposition 1. Both yield a raw mean of $\bar { \hat { s } } = 0 . 6 7$ , above the acceptance threshold $\tau = 0 . 6 .$ , so any weighted-average method accepts both. Yet they represent fundamentally diferent verification states. In Scenario $2 ,$ the logical agent assigns $\hat { s } _ { L } = 0 . 2 0$ while the visual and contextual agents assign 0.90: the step is visually plausible and contextually relevant, but logically unsupported. This is precisely the failure mode identified in the introduction, a step that appears locally valid from some perspectives while harbouring a reasoning gap that a single-perspective or average-based verifier would miss. Consensus dispersion detects this conflict $( \varDelta ^ { * } = 0 . 1 3 > \epsilon )$ and correctly rejects. In Scenario 3, all three agents are in modest but genuine agreement, producing a low dispersion $( \varDelta ^ { * } = 0 . 0 2 < \epsilon )$ that correctly triggers acceptance despite the moderate mean confidence. The two scenarios are indistinguishable to any weighted average, yet they call for opposite verification decisions.

Table 15: Canonical verification scenarios illustrating divergence between weighted-average (WA) and $\mathbf { V E R D I C T }$ acceptance decisions.Score vectors $\left( \hat { s } _ { V } , \hat { s } _ { L } , \hat { s } _ { C } \right)$ are raw agent outputs. $\varDelta ^ { \ast }$ is the consensus dispersion with $\lambda _ { V } = 1 . 5 , \lambda _ { L } = 1 . 0 , \lambda _ { C } = 0 . 8$ . Rows 2 and 3 share identical raw means yet receive opposite VERDICT decisions, demonstrating that $\varDelta ^ { \ast }$ carries verificationrelevant information that no weighted average can recover.
<table><tr><td>Scenario</td><td>Score vector  $\left( \hat { s } _ { V } , \hat { s } _ { L } , \hat { s } _ { C } \right)$ </td><td>Raw Mean δ</td><td> $\varDelta ^ { * }$ </td><td>WA decision</td><td>VERDICT Diverge?</td><td></td></tr><tr><td>Unanimous confidence</td><td>(0.90, 0.90, 0.90)</td><td>0.90</td><td></td><td>0.00 Accept</td><td>Accept</td><td> $\mathrm { N o }$ </td></tr><tr><td>Cross-modal conflict</td><td>(0.90, 0.20, 0.90)</td><td>0.67</td><td></td><td>0.13 Accept</td><td>Reject</td><td>Yes</td></tr><tr><td>Modest consensus</td><td>(0.70, 0.60, 0.70)</td><td>0.67</td><td></td><td>0.02 Accept</td><td>Accept</td><td>No</td></tr><tr><td>Extreme visual conflict (0.95, 0.05, 0.95)</td><td></td><td>0.65</td><td>0.16</td><td>Accept</td><td>Reject</td><td>Yes</td></tr><tr><td>Collective doubt</td><td>(0.40, 0.35, 0.45)</td><td>0.40</td><td>0.01 Reject</td><td></td><td>Reject</td><td>No</td></tr></table>

Scenario 4 extends Scenario 2 to its extreme: near-unanimous visual and contextual endorsement coexists with near-total logical rejection. The raw mean (0.65) remains above threshold, so weighted averaging again accepts. Consensus dispersion rises to 0.16, exceeding ϵ, and rejects. Notably, the visual agent’s high stubbornness $\left( \lambda _ { V } = 1 . 5 \right)$ means its strong score is not simply averaged away: the consensus solution adjusts $s _ { V } ^ { * }$ downward less aggressively than it does $s _ { C } ^ { * } ,$ which is reflected in a higher contribution to $\varDelta ^ { \ast }$ from the visual–logical axis of disagreement. This asymmetry is invisible to symmetric variance measures.

Scenario 5 illustrates agreement in rejection: when all agents collectively doubt the step, both methods correctly reject, and no divergence arises. This confirms that the VERDICT behaviour converges with simple averaging in the regime of low collective confidence, where disagreement structure is moot.

Implication for the Variance baseline. A natural question is whether raw score variance could serve as a proxy for $\varDelta ^ { \ast }$ . Scenarios 2 and 4 both exhibit high raw variance, so variance-based rejection would correctly handle them. However, variance treats all agent deviations symmetrically, making it a separable function over raw scores precisely the class of measures that Proposition 1 shows cannot replicate $\varDelta ^ { \ast }$ . The practical consequence is visible in a variant of Scenario 4 where the contextual agent (low $\lambda _ { C } = 0 . 8 )$ dissents instead of the logical agent: raw variance is identical, but $\varDelta ^ { \ast }$ is lower because the consensus solution couples the dissenting agent’s residual to the other agents’ positions, pulling it further toward consensus. The two methods, therefore, make diferent predictions on structurally similar configurations depending on which agent disagrees, a distinction that is meaningful given the heterogeneous roles of visual, logical, and contextual evaluation, and one that has no separable measure can capture.

## 17 Connections to Social Choice Theory and Belief Aggregation

The consensus formulation in VERDICT has intellectual roots in several traditions that predate its use in multimodal verification. We briefly position our work relative to these traditions, noting both the genuine correspondences and the important disanalogies.

Opinion pooling and the DeGroot model. The classical problem of opinion pooling [16,36] asks how to aggregate probability distributions from multiple experts into a single collective distribution. Linear pooling (weighted arithmetic averaging) is the simplest solution and corresponds directly to our Mean baseline. DeGroot [8] introduced an iterative model in which agents repeatedly revise their beliefs toward weighted averages of their neighbours’ opinions; under mild conditions, this converges to a consensus that is a fixed linear combination of the initial beliefs. Lehrer and Wagner [21] extended this with the notion of “respect weights,” where each agent’s influence on the consensus depends on how much others respect their expertise.

The Friedkin–Johnsen model: the most direct precedent. A key limitation of the DeGroot model is that agents are fully open to social influence: they have no attachment to their initial beliefs. Friedkin and Johnsen [13,14] addressed this by introducing a stubbornness parameter for each agent, anchoring opinions to their initial positions. In the Friedkin–Johnsen (FJ) model, each agent’s updated opinion is a convex combination of the group average and its own innate belief, weighted by a stubbornness coeficient that controls how much the agent resists social pressure. This is structurally the closest ancestor of our consensus formulation (Eq. (2) of the main paper), where the stubbornness parameter $\lambda _ { i }$ plays the same role: a higher $\lambda _ { i }$ means the agent places more weight on its own modality-specific evidence and less on the group consensus. The FJ model’s unique equilibrium, proven to exist under conditions analogous to our positive-definiteness result, is the point at which each agent has found its optimal compromise between conformity and self-consistency. Our closed-form solution (Eqs. 14-15 is mathematically equivalent to computing this equilibrium directly without iterating.

The correspondence becomes even more precise through the game-theoretic lens provided by Bindel, Kleinberg, and Oren [2]. They showed that the FJ equilibrium is exactly the unique Nash equilibrium of a game in which each agent minimises a quadratic cost that trades of disagreement with neighbours against deviation from its own internal opinion, essentially the same structure as our scoring objective in Eq. (1) of the main paper. This connection is not incidental: both formulations derive the influence structure from the same principled assumption (balancing private evidence against social pressure) rather than specifying weights ad hoc, and both admit closed-form solutions via the same linear-algebraic machinery.

Where VERDICT departs from this tradition. Despite the structural correspondence, our framework difers from the FJ/Bindel-Kleinberg-Oren tradition in three important respects.

First, in how the consensus is used. In opinion pooling and the FJ model, the consensus distribution or equilibrium opinion vector is the output: the goal is to reach agreement. In VERDICT, the consensus scores are an intermediate representation from which we extract two summary statistics: mean confidence and dispersion for a downstream accept/reject decision. The consensus itself is not the goal; the structure of residual disagreement after consensus adjustment is what carries the verification signal. This shift from “consensus as output” to “disagreement after consensus as signal” is, to our knowledge, not present in either the opinion dynamics or the game-theoretic opinion formation literature. Second, in the heterogeneity of stubbornness. While the FJ model permits heterogeneous stubbornness in principle, most theoretical analyses (including Bindel et al. [2]) focus on the symmetric case or treat heterogeneity as a complication rather than a design lever. In VERDICT, the asymmetric stubbornness assignment $( \lambda _ { V } > \lambda _ { L } > \lambda _ { C } )$ is a deliberate design choice encoding the prior that some modalities are more reliable than others for a given evaluation task. The ablation in 11.8 confirms that this assignment matters: performance degrades proportionally to the reduction in the visual agent’s stubbornness, a sensitivity that reflects agent architecture rather than dataset-specific tuning.

Third, in the scoring protocol. Our agents score in complete isolation and never observe each other’s outputs, whereas the DeGroot and FJ models assume agents see and respond to each other’s beliefs iteratively. The equilibrium is computed in closed form from one-shot scores rather than reached through iterated play. This makes the game-theoretic framing interpretive scafolding that motivates the formulation rather than a claim about strategic interaction.

Test-time model aggregation. The most direct precedent in machine learning is the work of Mendler-D¨unner et al. [30], who apply the DeGroot consensus mechanism to aggregate predictions from heterogeneous pre-trained models at test time. Their method computes mutual trust scores between models based on agreement on held-out data, then iterates DeGroot updates to convergence. VERDICT difers in three respects: (1) our agents are frozen and score in isolation, never observing each other’s outputs, whereas DeGroot aggregation requires agents to see and respond to each other’s beliefs; (2) we use heterogeneous stubbornness parameters rather than symmetric trust matrices, encoding the prior that some modalities are more reliable than others for a given evaluation task; and (3) our primary output is the dispersion of the consensus, not the consensus value itself, enabling a dual acceptance criterion that no existing test-time aggregation method provides.

What VERDICT inherits and what it adds. From the opinion pooling tradition, VERDICT inherits the principle that a weighted compromise among heterogeneous assessments can be more reliable than any individual assessment. From the Friedkin–Johnsen model, it inherits the stubbornness mechanism that anchors each agent to its private evidence. From the game-theoretic tradition (Rosen [32]; Bindel et al. [2]), it inherits the guarantee of existence and uniqueness of the fixed point and the principled derivation of influence weights from a single interpretable cost function.

What it adds is the treatment of post-consensus disagreement as a first-class verification signal. Proposition 1 in the main paper formalises this: the consensus dispersion ∆<sup>∗</sup> captures cross-agent interaction structure that no separable function of individual scores can replicate. This property has no direct analogue in classical opinion pooling or in the Bindel-Kleinberg-Oren analysis, where the consensus is valued precisely because disagreement has been resolved, not because its residue is informative. The cross-benchmark stubbornness analyses in 11.7 and 11.8 provide empirical confirmation that the sensitivity hierarchy and assignment ordering predicted by the FJ equilibrium structure hold across all benchmarks tested.

## 18 Qualitative Samples

To better understand how the reasoning process evolves under verification, we examine several representative cases where the base model diverges from the correct interpretation of the scene. Across these examples, the errors arise through diferent mechanisms: in some cases, the model relies on general world knowledge rather than visual evidence, while in others, the relevant objects are partially identified, but the comparison set shifts during reasoning. The examples below illustrate how the verification agents stabilize the reasoning trajectory by maintaining grounding in the visual context.

![](images/207fb693981ee99a0cb74459df175e80bfaeb212c7196bdca0da8afece7f2778.jpg)  
A) Height reasoning influenced by general knowledge bias.

![](images/f4be282e90ae032797672f753b72c0634c46bd1cfda1a6b65475e0f234dd0855.jpg)  
B) Misidentified reference object during comparison.  
Fig. 17: Two failure modes in spatial reasoning, corrected by VER-DICT. In (A), the base model relies on stored priors about typical object heights rather than grounding its reasoning in the visual evidence; it reasons from stored expectations about how high kites typically fly relative to streetlights and arrives at an answer that contradicts what is plainly visible. In (B), the error is subtler: the TV is correctly located, but the model anchors its comparison to the people in the room rather than the flowers on the mantle, efectively solving a diferent question than the one asked. What unites both cases is a drift away from the specific scene toward general scene semantics. The verification agents resist this drift — not by knowing the right answer, but by flagging reasoning steps that cannot be reconciled across visual, logical, and contextual perspectives simultaneously.

![](images/2f3f4bb3c9a91f1f4fec6d9d7dc4fb924cae8c2dd73676113871df90e3cf5ac3.jpg)  
A) Perspective reasoning with refrigerator and door.

![](images/ee041900e8da1afe82f6432fc34d35d3c44cdae4a64fd5dced3e62b3b7337a94.jpg)  
B) Depth reasoning failures arising from misleading visual cues

Fig. 18: Depth reasoning failures arising from misleading visual cues, corrected by VERDICT. In (A), the refrigerator and door occupy similar regions of the frame, and the base model resolves the ambiguity by treating visual prominence as a proxy for proximity -the door appears more immediate, so the model concludes it is closer. The reasoning is internally consistent but spatially wrong. In (B), the error runs deeper: relative size, a generally reliable depth cue, here points in the wrong direction, and the model follows it without hesitation. What neither chain does is pause to reconcile its size-based inference against other available cues — foreshortening, occlusion, and positional context. The verification agents introduce exactly that pressure. By requiring a step to hold simultaneously across visual, logical, and contextual perspectives, they surface the moment a single-cue heuristic is doing work it cannot support.

![](images/8b52c906ae6617b17ba08f9674acccd0a5cdbccaba390cb8ccb44db44d5b1864.jpg)  
B) Ambiguity between foreground and background objects  
Fig. 19: Reference drift under scene complexity, corrected by VER-DICT. Both scenes involve clearly highlighted objects, yet the base model gradually loses track of which object it is actually evaluating. In (A), the table is large and scene-filling; the model begins by acknowledging the highlighted monitor but is pulled toward the table’s visual dominance, eventually reasoning about the table as though it were the foreground object. In (B), a similar substitution occurs with the desk: the question concerns a specific highlighted instance, but the model conflates it with a more prominent surface nearby. The error is not one of misperception so much as incomplete binding — the model knows what the objects are, but the thread connecting each label to its referent frays under the scene’s visual weight. The consensus mechanism guards against this by treating referential consistency as a condition that every step must satisfy, not just the first.

![](images/88a09079dcd12072aefe7633309e00872e793ddb79f6f878910f0d447f91c17b.jpg)  
A) Erroneous Correction mechanism

![](images/5210be86d4d317c1de1026b87b60cb4e55267d325efc04c5a39125866e4c1126.jpg)  
B) Relies on World Knowledge

Fig. 20: Two failure modes where VERDICT overcorrects, and the base model succeeds. In (A), both chains begin identically — two windows, check each for blinds. The base model finds a white blind partially covering the left window and stops there. VERDICT, under pressure from the verification agents to be visually precise, rechecks both windows and eliminates the partially visible blind as insuficiently confirmed, arriving at zero. The correction mechanism that elsewhere prevents hallucination here suppresses a genuine observation. In (B), the scene does not contain a step stool at all, and both chains acknowledge this. The base model treats the absence as an invitation to reason from world knowledge — a step stool on the floor would sit below a table-height notebook — and answers correctly. VERDICT instead treats the absence as a hard epistemic boundary: without a visible referent, no location can be assigned, and the comparison collapses to the only object with a defined position. The verification agents, by design, are strict about visual grounding and penalize exactly the kind of common-sense inference that the question requires.

## 19 Prompt Templates

Here are the prompt templates for the three verifiers we are using in our proposed approach

![](images/d25b0035f5d7b9745877f3326634866e6d3286b4974f14232843244f4dd722d2.jpg)  
Fig. 21: System and task prompt used for the Visual Verification Agent, which evaluates whether a reasoning step is visually grounded in the image and whether spatial descriptions match the viewer’s perspective. The agent outputs a single scalar confidence score in [0,1].

![](images/e33252250139d1f65543fc2869d4616293696e9e5a1a8331883a988db6de6cb8.jpg)  
Fig. 22: System and task prompt used for the Logical Verification Agent, which assesses whether a reasoning step logically follows from the question and previous steps without relying on external knowledge. The agent returns a single numerical validity score in [0,1].

![](images/cb11db64f1cfe1543df08ae77b94c66f285e26e7a861bd9225e2c3d8a3faf231.jpg)  
Fig. 23: System and task prompt used for the Contextual Agent, which evaluates whether a reasoning step forms a valid link in the reasoning chain by checking perceptual grounding, temporal order, and mechanism plausibility. The agent outputs a single scalar score in [0,1].

## References

1. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., Zhu, K.: Qwen3-vl technical report (2025), https://arxiv.org/abs/2511.21631 1, 10

2. Bindel, D., Kleinberg, J., Oren, S.: How bad is forming your own opinion? Games and Economic Behavior 92(C), 248–265 (2015). https://doi.org/10.1016/j. geb.2014.06.004 56, 57, 58

3. Cao, Q., Wang, R., Zhang, R., Somayajula, S.A., Xie, P.: DreamPRM: Domainreweighted process reward model for multimodal reasoning. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025), https:// openreview.net/forum?id=ZyiBk1ZinG 4, 10

4. Chen, L., Li, J., Dong, X., Zhang, P., Zang, Y., Chen, Z., Duan, H., Wang, J., Qiao, Y., Lin, D., Zhao, F.: Are we on the right way for evaluating large vision-language models? (2024), https://arxiv.org/abs/2403.20330 10

5. Chen, Q., Qin, L., Zhang, J., Chen, Z., Xu, X., Che, W.: M<sup>3</sup>cot: A novel benchmark for multi-domain multi-step multi-modal chain-of-thought (2024), https://arxiv. org/abs/2405.16473 1

6. Chen, X., Aksitov, R., Alon, U., Ren, J., Xiao, K., Yin, P., Prakash, S., Sutton, C., Wang, X., Zhou, D.: Universal self-consistency for large language models. In: ICML 2024 Workshop on In-Context Learning (2024), https://openreview.net/ forum?id=LjsjHF7nAN 23

7. Coste, T., Anwar, U., Kirk, R., Krueger, D.: Reward model ensembles help mitigate overoptimization (2024), https://arxiv.org/abs/2310.02743 2, 3

8. DeGroot, M.H.: Reaching a consensus. Journal of the American Statistical Association 69(345), 118–121 (1974) 56

9. Ding, Y., Zhang, R.: Sherlock: Self-correcting reasoning in vision-language models (2025), https://arxiv.org/abs/2505.22651 2, 4, 10, 11

10. Du, L., Meng, F., Liu, Z., Zhou, Z., Luo, P., Zhang, Q., Shao, W.: Mm-prm: Enhancing multimodal mathematical reasoning with scalable step-level supervision (2025), https://arxiv.org/abs/2505.13427 4

11. Du, Y., Li, S., Torralba, A., Tenenbaum, J.B., Mordatch, I.: Improving factuality and reasoning in language models through multiagent debate (2023), https:// arxiv.org/abs/2305.14325 4

12. Eisenstein, J., Nagpal, C., Agarwal, A., Beirami, A., D’Amour, A., Dvijotham, D., Fisch, A., Heller, K., Pfohl, S., Ramachandran, D., Shaw, P., Berant, J.: Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking (2024), https://arxiv.org/abs/2312.09244 2, 4, 15

13. Friedkin, N.E., Johnsen, E.C.: Social influence and opinions. Journal of Mathematical Sociology 15(3–4), 193–205 (1990) 56

14. Friedkin, N.E., Johnsen, E.C.: Social influence networks and opinion change. In: Advances in Group Processes, vol. 16, pp. 1–29. JAI Press (1999) 56

15. Fu, X., Hu, Y., Li, B., Feng, Y., Wang, H., Lin, X., Roth, D., Smith, N.A., Ma, W.C., Krishna, R.: Blink: Multimodal large language models can see but not perceive (2024), https://arxiv.org/abs/2404.12390 10

16. Genest, C., Zidek, J.V.: Combining probability distributions: A critique and an annotated bibliography. Statistical Science 1(1), 114–135 (1986) 56

17. Kang, W., Kuen, J., Ren, M., Wei, Z., Yan, Y., Liu, K.: Vgent: Visual grounding via modular design for disentangling reasoning and prediction (2025), https:// arxiv.org/abs/2512.11099 2, 6, 11

18. Kaya, M.O., Elliott, D., Papadopoulos, D.: Eficient test-time scaling for small vision-language models. In: The Fourteenth International Conference on Learning Representations (2026), https://openreview.net/forum?id=rClkte0ZTp 23

19. Kembhavi, A., Salvato, M., Kolve, E., Seo, M., Hajishirzi, H., Farhadi, A.: A diagram is worth a dozen images (2016), https://arxiv.org/abs/1603.07396 10

20. Lakshminarayanan, B., Pritzel, A., Blundell, C.: Simple and scalable predictive uncertainty estimation using deep ensembles. In: Guyon, I., Luxburg, U.V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., Garnett, R. (eds.) Advances in Neural Information Processing Systems. vol. 30. Curran Associates, Inc. (2017), https://proceedings.neurips.cc/paper\_files/paper/2017/file/ 9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf 3, 4

21. Lehrer, K., Wagner, C.: Rational Consensus in Science and Society: A Philosophical and Mathematical Study. D. Reidel Publishing Company, Dordrecht (1981) 56

22. Li, C., Xu, T., Guo, S.Y.: Reasoning-as-logic-units: Scaling test-time reasoning in large language models through logic unit alignment. In: Forty-second International Conference on Machine Learning (2025), https://openreview.net/forum? id=mMgSxbO4H0 23

23. Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W.X., Wen, J.R.: Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355 (2023) 2

24. Li, Z., Yu, W., Huang, C., Liu, R., Liang, Z., Liu, F., Che, J., Yu, D., Boyd-Graber, J., Mi, H., Yu, D.: Self-rewarding vision-language model via reasoning decomposition (2025), https://arxiv.org/abs/2508.19652 4, 10, 11

25. Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., Cobbe, K.: Let’s verify step by step. In: NeurIPS (2023) 2, 4

26. Liu, F., Lin, K., Li, L., Wang, J., Yacoob, Y., Wang, L.: Hallucination in large vision-language models: A survey. arXiv preprint arXiv:2402.00253 (2024) 2

27. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. In: Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., Levine, S. (eds.) Advances in Neural Information Processing Systems. vol. 36, pp. 34892–34916. Curran Associates, Inc. (2023), https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf 1

28. Luo, L., Liu, Y., Liu, R., Phatale, S., Guo, M., Lara, H., Li, Y., Shu, L., Zhu, Y., Meng, L., Sun, J., Rastogi, A.: Improve mathematical reasoning in language models by automated process supervision (2024), https://arxiv.org/abs/2406.06592 2, 4

29. Ma, W., Chen, H., Zhang, G., Chou, Y.C., Chen, J., de Melo, C.M., Yuille, A.: 3dsrbench: A comprehensive 3d spatial reasoning benchmark (2025), https:// arxiv.org/abs/2412.07825 10

30. Mendler-D¨unner, C., Peng, W., Zrnic, T.: Test-time collective prediction. In: Advances in Neural Information Processing Systems. vol. 34 (2021) 57

31. Nash, J.F.: Equilibrium points in n-person games. Proceedings of the National Academy of Sciences 36(1), 48–49 (1950) 3

32. Rosen, J.B.: Existence and uniqueness of equilibrium points for concave n-person games. Econometrica: Journal of the Econometric Society pp. 520–534 (1965) 7, 9, 50, 58

33. Roughgarden, T.: Twenty Lectures on Algorithmic Game Theory. Cambridge University Press (2016) 5

34. Saad-Falcon, J., Buchanan, E.K., Chen, M.F., Huang, T.H., McLaughlin, B., Bhathal, T., Zhu, S., Athiwaratkun, B., Sala, F., Linderman, S., Mirhoseini, A., Re, C.: Weaver: Shrinking the generation-verification gap by scaling compute for verification. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025), https://openreview.net/forum?id=dRjt4vlYVQ 2, 4, 15

35. Shen, S., Hou, L., Zhou, T., Yang, S., Wang, Q., Kweon, I.S., Tombari, F., Shen, Y.: Multi-modal preference alignment remedies degradation of visual instruction tuning on language model. arXiv preprint arXiv:2402.10884 (2024) 2, 6, 11

36. Stone, M.: The opinion pool. Annals of Mathematical Statistics 32(4), 1339–1342 (1961) 56

37. Sun, L., Liang, H., Wei, J., Yu, B., Li, T., Yang, F., Zhou, Z., Zhang, W.: Mmverify: Enhancing multimodal reasoning with chain-of-thought verification. In: arXiv preprint arXiv:2502.13383 (2025) 2, 4

38. Team, G., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Ram´e, A., Rivi\`ere, M., Rouillard, L., Mesnard, T., Cideron, G., bastien Grill, J., Ramos, S., Yvinec, E., Casbon, M., Pot, E., Penchev, I., Liu, G., Visin, F., Kenealy, K., Beyer, L., Zhai, X., Tsitsulin, A., Busa-Fekete, R., Feng, A., Sachdeva, N., Coleman, B., Gao, Y., Mustafa, B., Barr, I., Parisotto, E., Tian, D., Eyal, M., Cherry, C., Peter, J.T., Sinopalnikov, D., Bhupatiraju, S., Agarwal, R., Kazemi, M., Malkin, D., Kumar, R., Vilar, D., Brusilovsky, I., Luo, J., Steiner, A., Friesen, A., Sharma, A., Sharma, A., Gilady, A.M., Goedeckemeyer, A., Saade, A., Feng, A., Kolesnikov, A., Bendebury, A., Abdagic, A., Vadi, A., Gy¨orgy, A., Pinto, A.S., Das, A., Bapna, A., Miech, A., Yang, A., Paterson, A., Shenoy, A., Chakrabarti, A., Piot, B., Wu, B., Shahriari, B., Petrini, B., Chen, C., Lan, C.L., Choquette-Choo, C.A., Carey, C., Brick, C., Deutsch, D., Eisenbud, D., Cattle, D., Cheng, D., Paparas, D., Sreepathihalli, D.S., Reid, D., Tran, D., Zelle, D., Noland, E., Huizenga, E., Kharitonov, E., Liu, F., Amirkhanyan, G., Cameron, G., Hashemi, H., Klimczak-Pluci´nska, H., Singh, H., Mehta, H., Lehri, H.T., Hazimeh, H., Ballantyne, I., Szpektor, I., Nardini, I., Pouget-Abadie, J., Chan, J., Stanton, J., Wieting, J., Lai, J., Orbay, J., Fernandez, J., Newlan, J., yeong Ji, J., Singh, J., Black, K., Yu, K., Hui, K., Vodrahalli, K., Gref, K., Qiu, L., Valentine, M., Coelho, M., Ritter, M., Hofman, M., Watson, M., Chaturvedi, M., Moynihan, M., Ma, M., Babar, N., Noy, N., Byrd, N., Roy, N., Momchev, N., Chauhan, N., Sachdeva, N., Bunyan, O., Botarda, P., Caron, P., Rubenstein, P.K., Culliton, P., Schmid, P., Sessa, P.G., Xu, P., Stanczyk, P., Tafti, P., Shivanna, R., Wu, R., Pan, R., Rokni, R., Willoughby, R., Vallu, R., Mullins, R., Jerome, S., Smoot, S., Girgin, S., Iqbal, S., Reddy, S., Sheth, S., P˜oder, S., Bhatnagar, S., Panyam, S.R., Eiger, S., Zhang, S., Liu, T., Yacovone, T., Liechty, T., Kalra, U., Evci, U., Misra, V., Roseberry, V., Feinberg, V., Kolesnikov, V., Han, W., Kwon, W., Chen, X., Chow, Y., Zhu, Y., Wei, Z., Egyed, Z., Cotruta, V., Giang, M., Kirk, P., Rao, A., Black, K., Babar, N., Lo, J., Moreira, E., Martins, L.G., Sanseviero, O., Gonzalez, L., Gleicher, Z., Warkentin, T., Mirrokni, V., Senter, E., Collins, E., Barral, J., Ghahramani, Z., Hadsell, R., Matias, Y., Sculley, D., Petrov, S., Fiedel, N., Shazeer, N., Vinyals, O., Dean, J., Hassabis, D., Kavukcuoglu, K., Farabet, C., Buchatskaya, E., Alayrac, J.B., Anil, R., Dmitry, Lepikhin, Borgeaud, S., Bachem,

O., Joulin, A., Andreev, A., Hardin, C., Dadashi, R., Hussenot, L.: Gemma 3 technical report (2025), https://arxiv.org/abs/2503.19786 22

39. Tian, X., Zou, S., Yang, Z., He, M., Waschkowski, F., Wesemann, L., Tu, P., Zhang, J.: More thought, less accuracy? on the dual nature of reasoning in vision-language models (2025), https://arxiv.org/abs/2509.25848 2, 6

40. Tong, S., Brown, E., Wu, P., Woo, S., Middepogu, M., Akula, S.C., Yang, J., Yang, S., Iyer, A., Pan, X., Wang, Z., Fergus, R., LeCun, Y., Xie, S.: Cambrian-1: A fully open, vision-centric exploration of multimodal llms (2024), https://arxiv.org/ abs/2406.16860 10

41. Wang, P., Li, L., Shao, Z., Xu, R.X., Dai, D., Li, Y., Chen, D., Wu, Y., Sui, Z.: Math-shepherd: Verifying and reinforcing mathematical reasoning. In: ACL (2023) 2, 4

42. Wang, P., Li, L., Shao, Z., Xu, R., Dai, D., Li, Y., Chen, D., Wu, Y., Sui, Z.: Mathshepherd: Verify and reinforce LLMs step-by-step without human annotations. In: Ku, L.W., Martins, A., Srikumar, V. (eds.) Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 9426–9439. Association for Computational Linguistics, Bangkok, Thailand (Aug 2024). https://doi.org/10.18653/v1/2024.acl-long.510, https:// aclanthology.org/2024.acl-long.510/ 3

43. Wang, Q., Zhao, P., Huang, S., Yang, F., Wang, L., Wei, F., Lin, Q., Rajmohan, S., Zhang, D.: Learning to refine: Self-refinement of parallel reasoning in llms. ArXiv abs/2509.00084 (2025), https://api.semanticscholar.org/CorpusID: 281080079 23

44. Wang, W., Gao, Z., Chen, L., Chen, Z., Zhu, J., Zhao, X., Liu, Y., Cao, Y., Ye, S., Zhu, X., Lu, L., Duan, H., Qiao, Y., Dai, J., Wang, W.: Visualprm: An efective process reward model for multimodal reasoning (2025), https://arxiv.org/abs/ 2503.10291 2, 4

45. Wang, W., Gao, Z., Chen, L., Chen, Z., Zhu, J., Zhao, X., Liu, Y., Cao, Y., Ye, S., Zhu, X., Lu, L., Duan, H., Qiao, Y., Dai, J., Wang, W.: Visualprm: An efective process reward model for multimodal reasoning (2025), https://arxiv.org/abs/ 2503.10291 2

46. Wang, X., Li, C., Yang, J., Zhang, K., Liu, B., Xiong, T., Huang, F.: Llava-criticr1: Your critic model is secretly a strong policy model (2025), https://arxiv.org/ abs/2509.00676 2, 4, 10, 11

47. Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., Zhou, D.: Self-consistency improves chain of thought reasoning in language models. In: International Conference on Learning Representations (ICLR) (2023), arXiv:2203.11171 23

48. Zhang, D., Li, J., Lei, J., Wang, X., Liu, Y., Yang, Z., Li, J., Wang, W., Yang, S., Wu, J., Ye, P., Ouyang, W., Zhou, D.: Critic-v: Vlm critics help catch vlm errors in multimodal reasoning (2025), https://arxiv.org/abs/2411.18203 4

49. Zhang, D., Li, J., Lei, J., Wang, X., Liu, Y., Yang, Z., Li, J., Wang, W., Yang, S., Wu, J., Ye, P., Ouyang, W., Zhou, D.: Critic-v: Vlm critics help catch vlm errors in multimodal reasoning (2025), https://arxiv.org/abs/2411.18203 10

50. Zhang, H., Li, H., Li, F., Ren, T., Zou, X., Liu, S., Huang, S., Gao, J., Zhang, L., Li, C., Yang, J.: Llava-grounding: Grounded visual chat with large multimodal models. In: European Conference on Computer Vision (ECCV) (2024) 2, 6, 11

51. Zhang, Z., Zhang, A., Li, M., Zhao, H., Karypis, G., Smola, A.: Multimodal chainof-thought reasoning in language models (2024), https://arxiv.org/abs/2302. 00923 1