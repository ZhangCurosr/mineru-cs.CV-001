# Interactive Whole Slide Images for RL-based Tumour Segmentation

Mohamad Mohamad<sup>1\*</sup> Francesco Ponzio<sup>2</sup> Maxime Gassier<sup>3</sup>

Nicolas Pote<sup>3</sup> Xavier Descombes<sup>1</sup>

<sup>1</sup>Universite C´ ote d’Azur, Inria, CNRS, I3S, INSERM, IBV, Sophia Antipolis, Franceˆ <sup>2</sup>Department of Control and Computer Engineering, Politecnico di Torino, Turin, Italy <sup>3</sup>Department of Pathology, Bichat Hospital, Assistance Publique–Hopitaux de Paris, Paris, Franceˆ

## Abstract

Whole-slide image (WSI) analysis remains computationally challenging due to the extremely large spatial resolution of slides and the sparse distribution of tumour regions. We propose an end-to-end reinforcement learning framework for sequential tumour segmentation directly on WSIs. Instead of treating the slide as a predefined collection of candidate patches, we formulate the WSI itself as a hierarchical multi-resolution environment through which an agent navigates using movement, zooming, and tumour selection actions. The agent jointly processes local observations and a global thumbnail representation within an actor-critic architecture trained using proximal policy optimization (PPO). Experiments on pulmonary adenocarcinoma WSIs demonstrate the feasibility of direct sequential tumour segmentation on full slides, achieving comparable coarse segmentation quality relative to patch-based approaches operating at similar magnification levels, while reducing inference time to a few seconds per slide. We further analyse the impact of environment design and action-space granularity. Our results suggest that modelling WSIs as interactive environments provides a promising direction for RL-based computational pathology.

## 1 Introduction

Whole-slide images (WSIs) [18] are high-resolution digital scans of histological tissue slides widely used in clinical diagnosis and computational pathology. Due to their extremely large spatial resolution, WSIs are commonly represented as multi-resolution pyramids that enable efficient navigation across magnification levels. In clinical practice, pathologists sequentially explore these pyramidal slides through spatial navigation and zooming, focusing attention on diagnostically relevant tissue regions while avoiding exhaustive inspection of the entire slide.

From a computational perspective, the scale of WSIs makes dense slide-level processing prohibitively expensive. Consequently, most deep learning approaches decompose WSIs into smaller image patches that are processed independently or aggregated across the slide [28]. While effective, such approaches largely discard the sequential and interactive nature of slide exploration, often requiring exhaustive processing of many redundant or diagnostically irrelevant regions.

Reinforcement learning (RL) provides a natural framework for modelling sequential decision-making in interactive visual environments. Recent RL-based approaches for WSI analysis have primarily focused on selecting informative patches from predefined candidate tile sets for downstream classification tasks [37, 12, 35]. Earlier methods even constrained the interaction space to pre-extracted image tiles rather than the WSI itself [25, 31]. Although these approaches can improve computational efficiency, they do not directly model the hierarchical WSI representation as an interactive spatial environment. Treating the WSI pyramid itself as the environment would enable clinically faithful and computationally efficient slide navigation, naturally reflecting the coarse-to-fine strategy pathologists employ in practice.

In this work, we investigate end-to-end RL for sequential tumour segmentation directly on full WSIs. We formulate the pyramidal WSI representation itself as a hierarchical multi-resolution environment through which an agent navigates using movement, zooming, and tumour-selection actions. The proposed actor-critic agent jointly processes local observations and a global slide overview while progressively constructing a tumour segmentation mask through sequential interaction with the WSI pyramid. We evaluate the proposed framework on pulmonary adenocarcinoma WSIs and analyse the effects of environment design, action-space granularity, reward formulation, and training strategies. Despite operating under constrained sequential interaction, the proposed agent achieves comparable segmentation performance while substantially improving inference efficiency relative to patch-based baselines.

The main contributions of this work are summarised as follows:

• We introduce a reinforcement learning framework for sequential tumour segmentation that models the full WSI pyramid as an interactive multi-resolution environment, and train an end-to-end actor-critic on pulmonary adenocarcinoma WSIs.

• We perform extensive analyses of reward design, action-space granularity, architectural choices, and training strategies, providing practical insights into RL-based WSI segmentation.

• We compare the proposed framework against patch-level UNI-based classification pipelines across both segmentation performance and inference efficiency.

## 2 Related Works

## 2.1 RL in medical images

RL has gained increasing attention in medical image analysis as a principled framework for modelling sequential and spatial decision-making processes [14]. While applications span a broad range of tasks, we focus here on localisation, detection, and segmentation.

Early work formulated anatomical landmark detection as a navigation problem, in which a separate agent per landmark iteratively traverses 2D and 3D volumes via discrete actions, with the reward defined as the change in Euclidean distance to the target [11, 29, 6, 1]. Extensions of this paradigm introduced a multi-agent setting with shared backbone weights for implicit knowledge sharing [29] and uncertainty estimation to flag out-of-distribution images for manual review [6]. This line of work was later extended to organ and lesion localisation [24, 21].

Subsequent work explored tighter integration of RL with deep learning architectures. RL-based prelocalisation was combined with UNet-based segmentation for catheter detection [34], while multi-agent formulations were adopted for iterative and interactive segmentation refinement [17]. RL has also been applied to learn adaptive masking strategies for self-supervised pretraining, improving representation quality and downstream segmentation performance [33].

Segmenting small regions of interest remains challenging when localisation and segmentation are decoupled. A collaborative framework combining RL-based localisation and refinement with UNet-based segmentation was proposed to address this for small targets [32]. More recently, segmentation has been formulated as an iterative per-pixel decision process in which pixel labels are progressively refined under a learned policy [19].

![](images/892f5daad33b5cd8930c76cdab4539898ca28ab20c7e8bfe5e906e5bd7252f9a.jpg)  
Figure 1: Agent framework for sequential WSI tumour segmentation. A) WSIs are formulated as interactive multi-resolution environments. At each episode reset, a WSI and its tumour mask are sampled. The agent sequentially navigates the image pyramid through movement, zooming, and selection actions, progressively constructing an accumulated tumour segmentation mask (yellow). B) Actor–critic architecture jointly processing global and local observations through a shared backbone encoder followed by feature fusion and policy/value heads. C) State representation composed of a global overview and the current local patch. D) Reward computation based on improvements in overlap between accumulated selections and the ground-truth tumour mask. E) Action space composed of navigation and tumour-selection actions.

## 2.2 RL in Histopathology

Current work applying RL to histopathological images analysis differs in formulation and application, yet shares a common objective: reducing computational cost by selectively attending to diagnostically relevant regions in sparse WSIs.

Early approaches investigated end-to-end trained agents operating directly on image patches [25, 31]. HER2 IHC scoring was formulated as a sequential decision-making problem, where an agent iteratively selects a limited number of regions within a tile before producing a final tile-level prediction, with an inhibition-ofreturn mechanism to discourage repeatedly attending to previously visited regions [25]. Similarly, an agent for breast cancer classification on pre-cropped histopathology images was proposed, sequentially aggregating information from selected regions to refine predictions over time [31].

More recent work has shifted towards pretrained feature extractors and weakly supervised multi-instance learning (MIL) pipelines, where RL primarily serves as a patch selection strategy [36, 35, 37, 12]. These methods typically represent WSIs using predefined candidate patches or low-resolution feature maps, from which RL agents sequentially select informative regions. Despite differences in formulation, the learned action space remains largely restricted to spatial patch selection, while zooming and magnification transitions are predefined or externally controlled rather than explicitly learned within the policy itself.

While these approaches improve computational efficiency through selective region processing, they do not directly model the hierarchical WSI pyramid as an interactive environment. More recently, RL environments for histopathological analysis built on top of Gym [5] and TorchRL [4] have been introduced, facilitating the development and evaluation of RL agents for WSI-based tasks [20, 22]. To the best of our knowledge, our work is the first to formulate WSI tumour segmentation as an end-to-end sequential decision-making process operating directly on the full slide pyramid as the environment.

## 3 Segmentation Agent and Environment

We formulate the task as a Markov Decision Process (MDP) defined by the tuple $( \mathcal { S } , \mathcal { A } , \mathcal { R } , \gamma )$ , where $\boldsymbol { \mathcal { S } }$ denotes the state space, A the action space, R the reward function, and $\gamma \in [ 0 , 1 ]$ the discount factor. At each timestep t, the agent observes the current state $s _ { t }$ , executes an action $a _ { t }$ , and receives a reward $r _ { t }$ from the environment. The discounted return is defined as the cumulative discounted sum of future rewards over an episode:

$$
G _ { t } = \sum _ { k = t } ^ { T } \gamma ^ { k - t } r _ { k } ,\tag{1}
$$

where $T$ denotes the terminal timestep. The objective of the agent is to learn a policy $\pi ( \boldsymbol { a } _ { t } \mid \boldsymbol { s } _ { t } )$ that maximises the expected return $\mathbb { E } _ { \pi } [ G _ { t } ]$ while efficiently identifying diagnostically relevant regions of the WSI.

Similar to earlier end-to-end RL approaches [31, 25], our method directly trains an agent through sequential interaction with histopathology data. At the same time, like more recent WSI-level methods [12, 35, 37], our framework operates directly on whole-slide images rather than isolated image regions. However, in contrast to prior work, we formulate the WSI itself as the interactive environment, rather than treating it as a predefined set of candidate patches. Specifically, the slide is modelled as a spatial multi-resolution pyramid through which the agent sequentially navigates for tumour segmentation. In the following subsections, we detail the different components of the proposed framework.

## 3.1 Environment

At each environment reset, a WSI together with its corresponding tumour mask is sampled to define the current episode. Let I denote a WSI with corresponding ground-truth mask M, and let p denote the fixed observation size. We further denote by W and H the width and height of the slide at the highest-resolution pyramid level $( l = 0 )$ .

The environment is modelled as a spatial multi-resolution pyramid composed of levels $\{ 0 , \ldots , n \}$ together with a thumbnail representation, from which observations can be sampled. Adjacent pyramid levels differ by a downsampling factor of 2, such that an observation of size $p \times p$ at level l corresponds to a lower-resolution representation of a $( 2 p ) \times ( 2 p )$ region at level l − 1. While the observation size remains fixed across all levels, the size of the explorable spatial domain increases across the pyramid. Specifically, the spatial dimensions at level l are defined as:

$$
W _ { l } = \frac { W } { 2 ^ { l } } , \qquad H _ { l } = \frac { H } { 2 ^ { l } } ,\tag{2}
$$

where the maximum pyramid depth n is chosen such that:

$$
W _ { n } \geq p , H _ { n } \geq p , \mathrm { w h i l e } W _ { n + 1 } < p \ \mathrm { o r } \ H _ { n + 1 } < p .\tag{3}
$$

holds. Consequently, the pyramid depth varies across WSIs and depends on the chosen observation size $p .$ The only exception to the fixed $\times 2$ scaling occurs between the thumbnail representation and the coarsest pyramid level n.

The environment dynamics allow the agent to navigate both spatially and across pyramid levels by zooming in and out. Spatial movement is defined relative to the observation size $p ,$ where spatial transitions apply horizontal and vertical displacements $( \Delta x , \Delta y )$ scaled by the current observation dimensions. For example, a displacement of $( - 1 , 1 )$ shifts the observation window by one patch size left and downward, respectively, while varying movement magnitudes are also permitted. In addition to navigation, the environment maintains an accumulated selection mask corresponding to regions predicted as tumorous by the agent. Both the selection masks and ground-truth masks are propagated across pyramid levels to preserve multi-scale consistency. Figure 1A summarises the overall environment dynamics.

State $s _ { t } .$ At timestep t, the state consists of a pair of observations:

$$
s _ { t } = ( o _ { \mathrm { c u r r e n t } } ^ { \prime } , o _ { \mathrm { g l o b a l } } ^ { \prime } ) ,\tag{4}
$$

where $O _ { \mathrm { c u r r e n t } }$ denotes the current local observation of size $p \times p$ centred at spatial coordinates $( x , y , l )$ within pyramid level $l ,$ and $\boldsymbol { O } _ { \mathrm { g l o b a l } }$ denotes the thumbnail representation providing global contextual information. Both views are augmented with the accumulated selection mask generated from previous tumour predictions during the episode, producing the masked observations $O _ { \mathrm { c u r r e n t } } ^ { \prime }$ and $O _ { \mathrm { g l o b a l } } ^ { \prime }$ (Figure 1C).

Reward $r _ { t }$ . Several RL approaches for medical image localisation and segmentation employ reward signals based on inter-step improvements of localisation or segmentation metrics [24, 19, 11, 20]. However, directly using signed metric differences $( \mathbf { e . g . ~ } s i g n ( \Delta ( I o U ) ) )$ may encourage unstable behaviour or reward exploitation, particularly in tumour segmentation settings where individual selections contribute unevenly to the final segmentation quality.

We therefore define the reward using the positive improvement in accumulated Intersection-over-Union (IoU):

$$
r _ { t } = R ( s _ { t } , a _ { t } ) = \operatorname* { m a x } \big ( \Delta ( \mathrm { I o U } ) \times S ^ { \mathrm { I o U } _ { t - 1 } } , 0 \big ) ,\tag{5}
$$

where $S \geq 1$ is a scalar controlling the relative importance of late-stage refinement during an episode, and

$$
\Delta ( \mathrm { I o U } ) = \mathrm { I o U } _ { t } - \mathrm { I o U } _ { t - 1 } .\tag{6}
$$

with $\mathrm { I o U } _ { t }$ denotes the IoU between the ground-truth mask M and the union of all regions selected by the agent up to timestep t (Figure 1D). When $S = 1$ , reward contributions depend solely on the absolute IoU improvement $\Delta ( \mathrm { I o U } )$ , independently of the segmentation stage. This formulation avoids explicit penalties for incorrect selections while still discouraging them implicitly, since inaccurate selections reduce the maximum achievable IoU and consequently limit future reward accumulation. In practice, this encourages exploratory behaviour without suppressing selection actions in slides containing small tumour regions.

The reward is only computed following selection actions, while navigation actions receive zero reward. Reward computation may additionally be performed using lower-resolution pyramid representations of the masks to reduce computational overhead.

Observability and pyramid depth. In the current setting, the exploration depth of the pyramid is constrained to the thumbnail representation together with four pyramid levels $\left\{ n , n - 1 , n - 2 , n - 3 \right\}$ . Combined with the global thumbnail observation, this restriction approximately preserves observability of the environment without requiring an explicit memory module. Removing either the global overview or the constrained pyramid depth would instead transform the problem into a partially observable Markov decision process (POMDP), potentially motivating recurrent architectures such as LSTMs similar to [31, 25]. Although recurrent memory mechanisms may further improve performance and are widely used in RL, our primary objective is to evaluate whether long-horizon WSI interaction can generalise across unseen patients without such mechanisms. Incorporating explicit memory modules would introduce an additional layer of architectural and optimisation complexity that falls outside the scope of the present study. We therefore defer its integration to future work.

## 3.2 Agent

The agent architecture consists of three main components: a feature extractor, a feature fusion stage, and separate actor-critic heads. The two image observations composing the state $O _ { \mathrm { c u r r e n t } } ^ { \prime }$ and ${ \cal O } _ { \mathrm { g l o b a l } } ^ { \prime }$ are independently processed using a shared-weight backbone encoder, producing latent feature representations $v _ { \mathrm { c u r r e n t } }$ and $v _ { \mathrm { g l o b a l } }$ . The resulting feature vectors are concatenated into a joint representation v, which is subsequently processed by separate actor and critic heads. The actor predicts a probability distribution over actions, $\pi ( . | s _ { t } )$ , while the critic estimates the state-value function, $V _ { \pi } ( s _ { t } )$ . An overview of the architecture is shown in Figure 1B.

## Actions.

The action space is divided into two categories: navigation actions and decision actions (Figure 1E). Since the agent performs tumour segmentation directly through interaction with the WSI, decision actions correspond to tumour selection over the current observation. In the current formulation, this is represented by a single Select action, which marks the current region as tumourous. Navigation actions consist of spatial movement and zooming operations within the image pyramid. Zooming actions allow transitions across pyramid levels, while spatial actions translate the observation window relative to the current patch size. Let m denote the relative movement factor controlling the translation magnitude. For each value of $m .$ four directional actions corresponding to horizontal and vertical movement are defined. We investigate two configurations: $m = 0 . 2 5$ yielding four directional actions plus zoom-in, zoom-out, and select (Agent-7), and $m \in \{ 0 . 1 , 0 . 5 , 1 \}$ yielding fifteen actions (Agent-15).

Backbone architecture. We employ a lightweight ResNet-family backbone [13] as a feature encoder for both local and global observations, following prior WSI RL approaches based on convolutional patch representations [37, 35, 36]. While the framework remains backbone-agnostic and extensible to transformer architectures, ResNet encoders provide a favourable trade-off between representational capacity and computational efficiency for long-horizon RL training.

Generalisation in visual RL remains an active research problem [16, 8, 9], particularly under limited environment diversity. Since training directly on full WSIs substantially increases complexity while limiting the number of unique training environments, we investigate dropout regularisation as a strategy to improve robustness and mitigate overfitting.

Agent Training. Earlier work on WSI analysis trained agents using REINFORCE [25, 31], while more recent methods [35, 37, 12] adopt PPO [27], which we follow throughout this study. At each timestep, the agent samples an action $a _ { t } \sim \pi _ { \theta } ( \cdot | s _ { t } )$ , receives reward $r _ { t } .$ , and transitions to the next state $s _ { t + 1 }$ . PPO alternates between rollout collection under a fixed policy $\theta _ { k }$ and policy optimisation using an advantage estimate $A _ { t }$ The clipped surrogate objective is defined as:

$$
L ^ { \mathrm { C L I P } } ( \boldsymbol { \theta } ) = \mathbb { E } _ { t } \left[ \operatorname* { m i n } \left( \hat { r } _ { t } ( \boldsymbol { \theta } ) A _ { t } , \ \mathrm { c l i p } ( \hat { r } _ { t } ( \boldsymbol { \theta } ) , 1 - \epsilon , 1 + \epsilon ) A _ { t } \right) \right]\tag{7}
$$

where $\hat { r } _ { t } ( \theta ) = \pi _ { \theta } ( a _ { t } | s _ { t } ) / \pi _ { \theta _ { k } } ( a _ { t } | s _ { t } )$ denotes the probability ratio between the current and previous policy. The full PPO objective additionally incorporates a value loss and entropy bonus:

$$
L ^ { \mathrm { P P O } } ( \boldsymbol { \theta } ) = \mathbb { E } _ { t } \left[ L _ { t } ^ { \mathrm { C L I P } } ( \boldsymbol { \theta } ) - c _ { 1 } \left( V _ { w } ( s _ { t } ) - V _ { t } ^ { \mathrm { t a r g e t } } \right) ^ { 2 } + c _ { 2 } \mathcal { H } [ \pi _ { \boldsymbol { \theta } } ] ( s _ { t } ) \right]\tag{8}
$$

![](images/1d036b0ba8d7e390dbd2bd2547c1b7cf5178f6b154e28250a5c62be4708f8f4c.jpg)  
Figure 2: Reward shaping analysis. Left: Normalised reward landscapes for different scaling parameters S as a function of consecutive IoU values $( \mathrm { I o U } _ { t - 1 } , \mathrm { I o U } _ { t } )$ . Increasing S shifts the reward towards improvements obtained at higher IoU regimes, encouraging late-stage refinement. Right: Training dynamics for different values of S. Lower values favour broader region selection and higher recall early during training, whereas larger values promote more conservative refinement with improved precision.

where $V _ { w }$ denotes the critic, H the entropy bonus, and $c _ { 1 } , c _ { 2 }$ weighting coefficients.

Incorporating batch normalisation and dropout into RL training can introduce optimisation instabilities due to discrepancies between training and evaluation behaviour [23, 3, 2]. Prior WSI RL approaches [36, 35, 37] largely avoid such issues by freezing the backbone and using it solely as a feature extractor. In contrast, our backbone is trained jointly with the agent, motivating the use of MDR [23], a recently proposed method for stable RL optimisation under batch normalisation and dropout regularisation. We compare MDR against standard PPO training in the ablation study (Section 4.3).

## 4 Experiments

We conduct ablation studies on the main components of the proposed agent and environment. In particular, we analyse the effect of the reward scaling parameter S (Section 4.2), architectural choices related to dropout and MDR-based training stabilisation (Section 4.3), and the impact of episode horizon T and action-space granularity m (Section 4.4). We further compare the proposed framework against a patch-level classification baseline in terms of both segmentation performance and inference efficiency on the evaluation cohort (Section 4.5).

Unless otherwise specified, all experiments use a ResNet18 backbone, the Agent-7 configuration, $S = 4$ max timesteps $T = 8 0$ , and an observation size of $p = 1 1 2$ . All experiments are conducted across three random seeds. Additional implementation details and hyperparameter settings are provided in Appendix C.

## 4.1 Dataset

Experiments were conducted on a pulmonary adenocarcinoma dataset collected at Hopital Pasteur for tumourˆ segmentation. The original cohort consisted of 94 patients with one or more WSIs, resulting in 106 annotated slides. The cohort is balanced across sex, with a mean patient age of 65.

![](images/b384141b499f2acbee8bbe1dc2d64c049fe6242a10cf1bec55da083665fce97a.jpg)  
Figure 3: Training stability and generalisation analysis. Left: MDR stabilises PPO training and substantially improves convergence and segmentation performance compared to standard PPO. Middle: Effect of dropout regularisation across different agent and backbone configurations. Dropout improves evaluation performance, particularly for higher-capacity agents, although a noticeable train–test gap remains. Right: Extending the training cohort substantially reduces the generalisation gap and improves evaluation performance. Training steps are doubled for agents trained on the extended cohort.

Each slide was annotated independently by two pathologists, one junior and one senior, using QuPath. Tumour regions were delineated precisely so as to exclude artefacts, non-tumoural areas, and adjacent tissue such as parenchyma and extratumoural stroma. The resulting tumour regions vary considerably in size, occupying $2 3 . 7 \pm 1 1 . 3 \%$ of the whole slide and $4 5 . 3 \pm 1 9 . 3 \%$ of the tissue area on average. This variability spans a wide range: measured as a fraction of the whole slide, 9 slides contain less than 10% tumour, 62 less than 25%, and 104 less than 50%; measured as a fraction of tissue area, the corresponding counts are 2, 15, and 67 slides.

The dataset was split at the patient level into training and test cohorts, with 65 patients (71 WSIs) used for training and 29 patients (35 WSIs) reserved for testing. We refer to this subset as Original-Cohort. To increase the number of environments available during RL training, we further extended the training cohort by incorporating additional WSIs from the same training patients, increasing the number of training slides from 71 to 334 (+263 WSIs). Since manual annotations were unavailable for these additional slides, tumour masks were generated automatically using UNI-ENS, a patch-level ensemble baseline described in Section 4.5, and were used exclusively during training.

Evaluation is performed only on the pathologist-annotated test slides of Original-Cohort, ensuring that pseudo-labels influence the training set alone and not the validity of reported results. Unless otherwise specified, ablation studies use Original-Cohort, while final results are obtained using the extended training set.

## 4.2 Reward Ablation

The reward formulation introduced in Section 3.1 uses improvements in IoU as the learning signal while introducing a scaling parameter S to modulate the contribution of late-stage segmentation refinement. To analyse the effect of this parameter on both optimisation and learned behaviour, we evaluate the agent using

Table 1: Effect of episode horizon T and action-space granularity on segmentation performance. Performance improves from short to moderate horizons before saturating, while Agent-15 largely benefits the low-budget setting.
<table><tr><td rowspan="2"> $T$ </td><td colspan="3"> $\overline { { \mathrm { A g e n t } - 7 } }$ </td><td colspan="3">Agent-15</td></tr><tr><td>Dice↑</td><td>NSD@512↑</td><td>NSD@2048↑</td><td>Dice↑</td><td>NSD@512↑</td><td>NSD@2048↑</td></tr><tr><td>20</td><td>0.842</td><td>0.147</td><td>0.484</td><td>0.873</td><td>0.179</td><td>0.555</td></tr><tr><td>40</td><td>0.887</td><td>0.188</td><td>0.585</td><td>0.890</td><td>0.206</td><td>0.609</td></tr><tr><td>60</td><td>0.893</td><td>0.201</td><td>0.612</td><td>0.887</td><td>0.209</td><td>0.612</td></tr><tr><td>80</td><td>0.894</td><td>0.206</td><td>0.617</td><td>0.888</td><td>0.209</td><td>0.613</td></tr></table>

four values of $S \in \{ 1 , 4 , 1 6 , 2 5 6 \}$

The main distinction lies between the case $S = 1$ and larger values of S. When $S = 1$ , rewards depend solely on the absolute IoU improvement at each step. In contrast, larger values increasingly favour improvements obtained at higher existing IoU values, thereby emphasising late-stage refinement during an episode. Although the reward remains globally maximised at a final IoU of 1 independently of S, the resulting navigation and selection strategies differ (see Appendix A for additional discussion on learned behaviour).

Figure 2 illustrates the effect of varying S. Increasing S shifts higher reward regions towards improvements occurring at already high IoU values. The corresponding training curves reveal a clear precision–recall trade-off: lower values of S initially favour broader recall-oriented behaviour, whereas larger values encourage more conservative selection strategies with higher precision. Intuitively, imprecise selections reduce the maximum achievable IoU during subsequent refinement steps, thereby favouring precision over recall when late-stage improvements are strongly rewarded. In contrast, low recall remains recoverable through future selections and therefore imposes a weaker constraint on long-term reward accumulation.

Very large values of S may additionally produce disproportionately high rewards, which can negatively affect optimisation stability. Overall, the IoU and precision trends suggest that moderate values of $S > 1$ (e.g., S=4 or S=16) provide a better optimisation trade-off, encouraging refinement-oriented behaviour while maintaining stable training dynamics.

## 4.3 Generalization and MDR

Figure 3 (left) illustrates the effect of MDR on training stability. Without MDR, PPO training of batchnormalisation-based architectures exhibits severe instability, with IoU stagnating near 0.3 throughout training. Incorporating MDR resolves this instability, establishing MDR as an essential component for jointly training architectures containing batch normalisation and dropout within the proposed RL framework.

Figure 3 (middle) shows the effect of dropout regularisation across agent and backbone configurations. Dropout consistently reduces the train–evaluation gap and improves evaluation IoU across all three configurations, with the benefit appearing more pronounced for higher-capacity agents. Nevertheless, a substantial generalisation gap remains when training on the limited Original-Cohort, motivating the cohort extension.

Figure 3 (right) shows that extending the training cohort substantially narrows the train–evaluation gap across both IoU and reward, yielding improvements that exceed those from dropout regularisation alone. Agents trained on the extended cohort require longer training horizons (doubled step budget) but converge to notably higher evaluation performance. These results highlight the importance of environmental diversity for training generalisable RL agents directly on full WSIs.

## 4.4 Action-space and horizon analysis

We ablate agent performance on the training cohort with respect to the maximum episode horizon T. An episode terminates either when 95% of the tumour area has been selected or when the maximum number of allowed steps is reached. Table 1 reports the performance of Agent-7 and Agent-15 for $T \in \{ 2 0 , 4 0 , 6 0 , 8 0 \}$ using Dice and Normalized Surface Dice (NSD), where NSD measures boundary agreement between predicted and ground-truth masks within a fixed tolerance (reported at 512 and 2048 pixels); standard deviations are omitted for readability.

Table 2: Segmentation performance on the evaluation set. ± denotes standard deviation across three seeds.
<table><tr><td>Method</td><td>Dice↑</td><td>Recall↑</td><td></td><td></td><td>Precision↑ NSD@512↑ NSD@2048↑</td></tr><tr><td colspan="6">Patch-based methods</td></tr><tr><td>UNI-1024</td><td> $\overline { { 0 . 8 8 \pm . 0 0 4 } }$ </td><td> $\overline { { 0 . 8 5 \pm . 0 3 0 } }$ </td><td> $\overline { { 0 . 9 1 \pm . 0 3 0 } }$ </td><td> $0 . 7 6 \pm . 0 4 1$ </td><td> $\overline { { 0 . 8 8 \pm . 0 4 6 } }$ </td></tr><tr><td>UNI-2048</td><td> $0 . 8 4 \pm . 0 2 5$ </td><td> $0 . 9 6 \pm . 0 0 7$ </td><td> $0 . 7 5 \pm . 0 4 4$ </td><td> $0 . 5 9 \pm . 0 2 7$ </td><td> $0 . 7 4 \pm . 0 3 2$ </td></tr><tr><td>UNI-4096</td><td> $0 . 8 1 \pm . 0 3 4$ </td><td> $0 . 9 3 \pm . 0 6 0$ </td><td> $0 . 7 4 \pm . 1 0 0$ </td><td> $0 . 6 1 \pm . 0 0 9$ </td><td> $0 . 7 4 \pm . 0 6 6$ </td></tr><tr><td>UNI-ENS</td><td> $0 . 9 0 \pm . 0 1 5$ </td><td> $0 . 9 3 \pm . 0 0 6$ </td><td> $0 . 8 7 \pm . 0 3 0$ </td><td> $0 . 9 1 \pm . 0 3 6$ </td><td> $0 . 9 1 \pm . 0 3 6$ </td></tr><tr><td>UNI-HR</td><td> $0 . 8 9 \pm . 0 0 8$ </td><td> $0 . 9 2 \pm . 0 1 1$ </td><td> $0 . 8 8 \pm . 0 2 4$ </td><td> $0 . 7 1 \pm . 0 3 0$ </td><td> $0 . 8 5 \pm . 0 3 5$ </td></tr><tr><td colspan="6">Sequential methods</td></tr><tr><td>Agent-7-Res18</td><td> $\overline { { 0 . 8 0 \pm . 0 0 4 } }$ </td><td> $\overline { { 0 . 8 7 \pm . 0 1 8 } }$ </td><td> $\overline { { 0 . 7 6 \pm . 0 1 4 } }$ </td><td> $\overline { { 0 . 1 4 \pm . 0 0 5 } }$ </td><td> $\overline { { 0 . 4 4 \pm . 0 1 0 } }$ </td></tr><tr><td>Agent-7-Res50</td><td> $0 . 8 3 \pm . 0 0 3$ </td><td> $0 . 8 9 \pm . 0 0 5$ </td><td> $0 . 8 1 \pm . 0 0 6$ </td><td> $0 . 1 6 \pm . 0 0 3$ </td><td> $0 . 4 8 \pm . 0 3 9$ </td></tr><tr><td>Agent-15-Res50</td><td> $0 . 8 4 \pm . 0 0 5$ </td><td> $0 . 8 8 \pm . 0 0 2$ </td><td> $0 . 8 2 \pm . 0 0 7$ </td><td> $0 . 1 9 \pm . 0 0 3$ </td><td> $0 . 5 6 \pm . 0 0 8$ </td></tr><tr><td>Human-7</td><td>0.91</td><td>0.86</td><td>0.96</td><td>0.26</td><td>0.72</td></tr></table>

Performance consistently decreases when limiting the episode horizon to $T = 2 0 .$ , indicating that short trajectories restrict the agent’s ability to refine tumour regions. In contrast, evaluation performance largely plateaus for larger values of T, suggesting that many slides can already be segmented within fewer than 60 interaction steps and that later selections contribute primarily to refinement.

The performance drop at low horizons is less pronounced for Agent-15 than for Agent-7, likely due to its larger and more flexible action space, which enables broader spatial exploration within fewer interaction steps. At longer horizons, Agent-15 performs comparably to Agent-7, suggesting that the larger action space may require additional training to exploit the extended step budget. Notably, NSD metrics improve more substantially than Dice with increasing T, suggesting that additional steps primarily contribute to boundary refinement rather than coarse tumour coverage. Additional qualitative comparisons of the learned navigation strategies are provided in Appendix B.

## 4.5 Results

We compare the proposed RL agent against patch-based deep learning baselines built on UNI [7], a selfsupervised foundation model pretrained on over 100,000 histopathology WSIs. Three classifiers are trained using patches corresponding to effective spatial regions of 1024 ×1024, 2048 ×2048, and $4 0 9 6 \times 4 0 9 6$ pixels at the highest slide magnification, resized to $2 2 4 \times 2 2 4$ . These models are denoted UNI-1024, UNI-2048, and UNI-4096, respectively. Final segmentation masks are obtained through patch aggregation followed by connected-component filtering. We additionally report an ensemble model (UNI-ENS) obtained through patch-wise majority voting across the three classifiers.

We additionally evaluate a hierarchical coarse-to-fine baseline (UNI-HR) that combines the UNI-4096 and UNI-1024 models. A coarse segmentation is first obtained with UNI-4096, after which the predicted boundary is used to identify locations for high-resolution refinement with UNI-1024. Four $1 0 2 4 \times 1 0 2 4$ patches are sampled on each side of the predicted boundary, spanning the inner and outer boundary regions. Only non-overlapping patches are extracted and fed to UNI-1024. The resulting fine-scale predictions are then used to refine the coarse segmentation mask.

The RL agents are trained on the extended cohort, while evaluation is performed exclusively on the pathologist-annotated evaluation cohort. We report three agent configurations spanning two backbones and two action spaces. Agent-7-Res18 and Agent-7-Res50 share the seven-action space but use ResNet18 and

Table 3: Inference time per slide on the evaluation set. The proposed RL agent substantially improves efficiency. ± is across slides and seeds (Mean) or across seeds (Best/Worst-case, computed per-seed).
<table><tr><td>Method</td><td>Mean Time (s)↓ Worst-case (s)↓ Best-case (s)↓</td><td></td><td></td></tr><tr><td colspan="4">Patch-based methods</td></tr><tr><td>UNI-1024</td><td> $2 2 9 . 1 \pm 1 6 4 . 4$ </td><td> $\overline { { 8 2 9 . 1 \pm 2 6 6 . 6 } }$ </td><td> $\overline { { 7 9 . 6 \pm 1 . 7 } }$ </td></tr><tr><td>UNI-2048</td><td> $1 8 7 . 9 \pm 1 1 8 . 1 $ </td><td> $5 3 8 . 3 \pm 1 0 . 4$ </td><td> $6 4 . 1 \pm 2 . 9$ </td></tr><tr><td>UNI-4096</td><td> $3 6 . 8 \pm 4 0 . 2$ </td><td> $1 3 8 . 7 \pm 3 . 0 $ </td><td> $1 0 . 0 \pm 0 . 4$ </td></tr><tr><td>UNI-HR</td><td> $1 3 0 \pm 1 1 0 . 2$ </td><td> $4 3 3 \pm 1 4 . 7$ </td><td> $3 6 . 6 \pm 1 . 6$ </td></tr><tr><td colspan="4">Sequential methods</td></tr><tr><td>Agent-7-Res18</td><td> ${ \bf 3 . 0 \pm 1 . 5 }$ </td><td> ${ \bf 6 . 0 \pm 1 . 3 }$ </td><td> $\overline { { { \bf 1 . 0 \pm 0 . 2 } } }$ </td></tr><tr><td> $\mathrm { A g e n t - } 7 { \cdot } \mathrm { R e s } 5 0$ </td><td> ${ \bf 3 . 6 \pm 1 . 5 }$ </td><td> ${ \bf 6 . 5 \pm 0 . 1 }$ </td><td> ${ \bf 1 . 2 \pm 0 . 1 }$ </td></tr><tr><td> $\mathrm { A g e n t - } 1 5 \mathrm { - } \mathrm { R e s } 5 0$ </td><td> ${ \bf 3 . 1 \pm 0 . 7 }$ </td><td> ${ \bf 5 . 1 \pm 0 . 6 }$ </td><td> ${ \bf 1 . 8 \pm 0 . 1 }$ </td></tr></table>

ResNet50 backbones, respectively, allowing assessment of the effect of backbone capacity. Agent-15-Res50 retains the ResNet50 backbone while expanding the action space to fifteen actions, allowing assessment of the effect of finer action granularity. Implementation details are provided in Appendix C. We further include a constrained human baseline (Human-7), where the annotator interacts with the slide using the same seven-action space and episode horizon as the agent while having access to the ground-truth segmentation mask.

Table 2 reports segmentation performance across Dice, Recall, Precision, and NSD metrics. Among the UNI-based approaches, UNI-1024 achieves the strongest single-model performance across Dice and NSD metrics, while UNI-ENS obtains the best overall performance. The hierarchical UNI-HR baseline reaches a Dice score comparable to UNI-ENS and improves boundary quality over its UNI-4096 starting point, with NSD@512 increasing from 0.61 to 0.71 and NSD@2048 from 0.74 to 0.85. This confirms that explicit high-magnification refinement can improve boundary agreement, although UNI-HR does not match UNI-ENS or UNI-1024 in boundary quality.

Among the sequential methods, Agent-7-Res18 and Agent-7-Res50 achieve Dice scores comparable to UNI-4096 and UNI-2048, respectively, despite operating under substantially stronger interaction constraints. Agent-15-Res50 is the strongest sequential agent across all reported metrics, achieving a Dice score of 0.84 and an NSD@2048 of 0.56. Its improvement over the seven-action agents suggests that the finer action space can provide a measurable benefit when combined with a sufficient training budget. Since the agents are restricted to navigating only the upper levels of the WSI pyramid, their explored spatial scale is most comparable to UNI-4096.

The largest performance gap appears in NSD metrics, indicating that most agent errors occur near tumour boundaries. This behaviour is expected given the coarse spatial granularity imposed by patch-level selection actions and the limited depth of the WSI pyramid. Nevertheless, the improvement from NSD@512 to NSD@2048 suggests that many errors remain spatially close to the true boundaries rather than corresponding to large localisation failures. This observation is further supported by Figure 4, where most discrepancies between the predicted and ground-truth masks remain localised near tumour boundaries. Notably, the constrained human annotator, despite having access to the ground-truth mask, achieves an NSD@512 of only 0.26 under the same seven-action constraints, increasing substantially to 0.72 at NSD@2048. This mirrors the agents’ behaviour and suggests that the boundary gap is predominantly structural, imposed by the coarse action space and pyramid constraints, rather than reflecting a fundamental limitation of the learned policy.

We additionally evaluate inference efficiency on the evaluation cohort. Table 3 reports average, best-case, and worst-case processing times per slide (Appendix D). For UNI-based methods, extraction is optimised by directly sampling patches from appropriate magnification levels and parallelising CPU tiling with GPU inference. Classification, mask aggregation, and post-processing times are omitted as they contribute minimally relative to patch extraction.

![](images/bfe0112fa1c8657d270c172f1d19ae336ce75c5481182062a93352f965040e18.jpg)  
Figure 4: Segmentation results on evaluation slides. Predictions from Agent-7 ResNet18 (left) and Agent-7 ResNet50 (right) are shown for tumours of varying size and morphology. Red contours denote the groundtruth tumour boundaries, while the blue contours correspond to the final agent segmentation obtained from the accumulated selections at the end of the episode.

Processing larger spatial regions substantially reduces inference time for UNI-based models by decreasing the number of extracted patches. However, processing time remains highly dependent on slide size, leading to large discrepancies between average and worst-case runtimes. Even UNI-4096 requires more than 30 seconds on average and exceeds two minutes in the worst case. The hierarchical UNI-HR baseline is slower still, averaging 130 seconds per slide and exceeding seven minutes in the worst case, since its boundary refinement stage adds a second round of high-magnification patch extraction. In contrast, RL agents process slides within 3 seconds while exhibiting substantially lower runtime variance across slides (worst case at 6s and best at 1s). Relative to UNI-1024, this corresponds to mean speedups of approximately 64–76× across the three agent configurations, with a Dice reduction of 0.05–0.08.

Overall, the proposed RL formulation demonstrates that direct sequential tumour segmentation on full WSIs is feasible under constrained interaction settings. While the agents remain less accurate than patch-based UNI approaches, particularly around fine tumour boundaries, they achieve substantially lower inference times and more consistent processing costs across slides.

## 5 Discussion

The results demonstrate the feasibility of direct sequential tumour segmentation on full WSIs under real-time inference constraints, while highlighting how design choices influence the trade-off between segmentation quality and computational efficiency. Improved generalisation under increased training environment diversity further suggests that scaling the number of training environments is a more impactful direction than architectural regularisation alone.

Several limitations remain. The current environment constrains exploration to four pyramid levels and a thumbnail, limiting access to finer multi-scale information. The agents operate without explicit memory mechanisms, restricting information aggregation across observations and magnification levels during longhorizon interaction. Finally, the study is monocentric and limited to pulmonary adenocarcinoma, although the proposed formulation is applicable to other organs, tumour types, and clinical centres.

These limitations point to concrete next steps. At the agent level, incorporating explicit memory mechanisms would enable cross-location and cross-scale context aggregation during long-horizon interaction. At the environment level, removing the pyramid-depth constraint would move towards fully unconstrained slide navigation. At the evaluation level, extending beyond a single centre and tumour type would test the generalisability of the formulation across organs and clinical sites. Finally, on the training side, scaling through higher-capacity and transformer-based backbones, pathology foundation models such as UNI [7], larger datasets, and recent advances in self-supervised and contrastive RL [10] represent promising avenues for learning more robust navigation directly from large-scale WSI environments.

## 6 Conclusion

We presented an end-to-end reinforcement learning framework for sequential tumour segmentation directly on whole-slide images modelled as interactive multi-resolution environments. The proposed formulation enables agents to navigate WSIs through spatial movement and zooming while performing segmentation under constrained interaction budgets. Experiments on pulmonary adenocarcinoma WSIs demonstrate the feasibility of direct sequential segmentation on full slides, achieving comparable coarse segmentation performance together with substantially improved inference efficiency relative to patch-based baselines. Beyond segmentation performance, our study provides insights into the role of reward design, action-space granularity, optimisation stability, and environment diversity in RL-based computational pathology.

## 7 Acknowledgement

This work has been supported by the French government, through the France 2030 investment plan managed by the Agence Nationale de la Recherche, as part of the “UCA DS4H” project, reference ANR-17-EURE-0004.

## References

[1] Amir Alansary et al. Evaluating reinforcement learning agents for anatomical landmark detection. Medical Image Analysis, 53:156–164, 2019.

[2] Aditya Bhatt et al. Crossnorm: Normalization for off-policy td reinforcement learning. CoRR, abs/1902.05605, 2019.

[3] Aditya Bhatt et al. Crossq: Batch normalization in deep reinforcement learning for greater sample efficiency and simplicity. In The Twelfth International Conference on Learning Representations, 2024.

[4] Albert Bou et al. Torchrl: A data-driven decision-making library for pytorch, 2023.

[5] Greg Brockman et al. Openai gym. CoRR, abs/1606.01540, 2016.

[6] James Browning et al. Uncertainty aware deep reinforcement learning for anatomical landmark detection in medical images. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2021, pages 636–644, 2021.

[7] Richard J Chen et al. Towards a general-purpose foundation model for computational pathology. Nature Medicine, 2024.

[8] Karl Cobbe et al. Quantifying generalization in reinforcement learning. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings ofthe 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 1282–1289. PMLR, 2019.

[9] Karl Cobbe et al. Leveraging procedural generation to benchmark reinforcement learning. In Proceedings of the 37th International Conference on Machine Learning, ICML’20, 2020.

[10] Benjamin Eysenbach et al. Contrastive learning as goal-conditioned reinforcement learning, 2023.

[11] Florin C. Ghesu et al. An artificial agent for anatomical landmark detection in medical images. In Medical Image Computing and Computer-Assisted Intervention - MICCAI 2016, pages 229–237, 2016.

[12] Tarun Gogisetty et al. Sequential attention-based sampling for histopathological analysis. In Advances in Neural Information Processing Systems, volume 38, pages 155532–155561, 2025.

[13] Kaiming He et al. Deep residual learning for image recognition, 2015.

[14] Mingzhe Hu et al. Reinforcement learning in medical image analysis: Concepts, applications, challenges, and future directions, 2022.

[15] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017.

[16] Heinrich Kuttler et al. The nethack learning environment, 2020.¨

[17] Xuan Liao et al. Iteratively-refined interactive 3d medical image segmentation with multi-agent reinforcement learning. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9391–9399, 2020.

[18] Geert Litjens et al. A survey on deep learning in medical image analysis. Medical Image Analysis, 42:60–88, 2017.

[19] Yunxin Liu et al. Pixel level deep reinforcement learning for accurate and robust medical image segmentation. Scientific Reports, 15(1):8213, 2025.

[20] Zhi-Bo Liu et al. Histogym: A reinforcement learning environment for histopathological image analysis, 2024.

[21] Gabriel Maicas et al. Deep reinforcement learning for active breast lesion detection from dce-mri. In Medical Image Computing and Computer Assisted Intervention - MICCAI 2017, pages 665–673, 2017.

[22] Mohamad Mohamad et al. Investigating Reinforcement Learning for Histopathological Image Analysis. In SciTePress Digital Library, pages 369–375, February 2025. BIOIMAGING is part of BIOSTEC, the 18th International Joint Conference on Biomedical Engineering Systems and Technologies.

[23] Mohamad Mohamad et al. Mode-dependent rectification for stable ppo training, 2026.

[24] Fernando Navarro et al. Deep reinforcement learning for organ localization in ct. In Proceedings of the Third Conference on Medical Imaging with Deep Learning, volume 121 of Proceedings of Machine Learning Research, pages 544–554, 2020.

[25] Talha Qaiser et al. Learning where to see: A novel attention model for automated immunohistochemical scoring. IEEE Transactions on Medical Imaging, 38(11):2620–2631, 2019.

[26] John Schulman et al. High-dimensional continuous control using generalized advantage estimation. In Proceedings of the International Conference on Learning Representations (ICLR), 2016.

[27] John Schulman et al. Proximal policy optimization algorithms, 2017.

[28] Zhuchen Shao et al. Transmil: Transformer based correlated multiple instance learning for whole slide image classification, 2021.

[29] Athanasios Vlontzos et al. Multiple landmark detection using multi-agent reinforcement learning. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2019, pages 262–270, 2019.

[30] Xiyue Wang et al. Retccl: Clustering-guided contrastive learning for whole-slide image retrieval. Medical Image Analysis, 83:102645, 2023.

[31] Bolei Xu et al. Look, investigate, and classify: A deep hybrid attention method for breast cancer classification. CoRR, abs/1902.10946, 2019.

[32] Feilong Xu et al. Rl-coseg: A reinforcement learning-based collaborative localization and segmentation framework for medical image. Expert Systems with Applications, 298:129661, 2026.

[33] Gang Xu et al. Adaptive-masking policy with deep reinforcement learning for self-supervised medical image segmentation. In 2023 IEEE International Conference on Multimedia and Expo (ICME), pages 2285–2290, 2023.

[34] Hongxu Yang et al. Deep q-network-driven catheter segmentation in 3d us by hybrid constrained semi supervised learning and dual-unet. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2020, pages 646–655, 2020.

[35] Boxuan Zhao et al. Rlogist: Fast observation strategy on whole-slide images with deep reinforcement learning, 2022.

[36] Tingting Zheng et al. Learning how to detect: A deep reinforcement learning method for whole-slide melanoma histopathology images. Computerized Medical Imaging and Graphics, 108:102275, 2023.

[37] Tingting Zheng et al. Dynamic policy-driven adaptive multi-instance learning for whole slide image classification, 2024.

## A Reward-Induced Behaviour

As discussed in Section 4.2, $S > 1$ scales rewards towards late-stage IoU improvements, making precise delimitation more valuable than early coarse coverage and directly shaping the selection strategy learned by the agent, since each selection affects the maximum achievable reward for all future steps.

Figures 5 and 6 show representative episodes for agents trained under different values of S compared against the constrained human strategy. The behavioural distinction is especially pronounced in Figure 6, where irregularly shaped tumours are present.

Agents trained with $S = 1$ exhibit high tolerance to precision sacrifice, favouring early selection of large boxes that rapidly cover substantial tumour area even at the cost of including significant non-tumour regions. This maximises short-term recall but accumulates precision errors that limit future refinement quality. Agents trained with $S > 1$ show substantially lower tolerance to such precision sacrifice — they allocate a larger fraction of the step budget to traversing tumour boundaries through multiple smaller selections before committing to broader coverage, trading coverage speed for delimitation quality.

This contrast is most visible for $S = 2 5 6$ in Figure 6, where the agent spends a substantial portion of its budget traversing the irregular tumour boundary before expanding coverage, such trajectories resemble the constrained human strategy, particularly during early episode stages.

![](images/a1c7f2927ab2410041189a9f6e49b2a6f12211350135e627e039f36265bd9ee0.jpg)  
Figure 5: Navigation strategies learned under different reward scaling values S ∈ {1, 4, 16, 256} together with the constrained human strategy shown on a training-cohort slides. Episode progression is shown from left to right, with the final timestep corresponding to the end of the episode.

![](images/0275ade335b8d4220c9d175bf860be1da9e579ede7028e9646b83bccc88933f0.jpg)  
Figure 6: Additional navigation episodes for agents trained with $S \in \{ 1 , 2 5 6 \}$

## B Episode Horizon and Learned Behaviour

Figure 7 illustrates the effect of the episode horizon T on both training dynamics and learned segmentation policies. Under constrained budgets (T = 20), agents favour large coarse selections that rapidly increase tumour coverage while sacrificing boundary precision, producing visibly blockier masks. Increasing the step budget progressively shifts the learned policy towards finer boundary refinement, with smoother and more accurate final masks emerging at T = 60 and T = 80, these behaviours are consistent with the NSD improvement observed in Table 1.

The effect of movement granularity is most pronounced at low horizons. Under T = 20, Agent-15 achieves substantially higher training IoU and reward than Agent-7, and qualitatively produces more refined segmentations. The additional movement options (e.g. m=1, m=0.5) enable more flexible spatial navigation within fewer steps, allowing a larger fraction of the episode budget to be spent on selection rather than movement. Agent-7, by contrast, compensates through larger coarse selections to maximise coverage under the same budget.

As T increases, the performance gap between agents progressively decreases in both training curves and qualitative segmentation quality. Although Agent-15 provides finer navigation capability, its larger action space increases optimisation complexity, and finer displacements such as m = 0.1 which could in principle enable more precise boundary delimitation but appears underutilised within the same training budget.

![](images/97d15590b00c1873171706dbc63d76956db983b431d1a5fbe74e16ca19c6010e.jpg)  
Figure 7: Effect of episode horizon T and movement granularity on training dynamics and learned segmentations. Top: training IoU and reward curves for Agent-7 and Agent-15. Bottom: final segmentations masks obtained under different interaction budgets on training slides.

## C Hyperparameters and Implementation details

Experiments use ImageNet-pretrained ResNet18 and RetCCL [30] pretrained ResNet50 together with dropout regularisation following [23]. Local and global image features are concatenated before being processed by independent actor and critic heads. For ResNet18, the concatenated representation is processed using two hidden layers of size 1024 in both the actor and critic networks. For ResNet50, a single hidden layer of size 4096 is used.

Table 4: Training and optimisation hyperparameters.
<table><tr><td>Hyperparameter</td><td>extended-cohort</td><td>original-cohort</td></tr><tr><td>Episode horizon (T)</td><td>80</td><td>80</td></tr><tr><td>Observation size (p)</td><td>112</td><td>112</td></tr><tr><td>Parallel environments</td><td>16</td><td>16</td></tr><tr><td>Rollout steps per environment</td><td>1000</td><td>500</td></tr><tr><td>Discount factor γ</td><td>0.99</td><td>0.99</td></tr><tr><td>GAE parameter λ</td><td>0.95</td><td>0.95</td></tr><tr><td>Value loss coefficient  $c _ { 1 }$ </td><td>1.0</td><td>1.0</td></tr><tr><td>Entropy coefficient  $c _ { 2 }$ </td><td> $4 \times 1 0 ^ { - 2 }$ </td><td> $4 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Batch size</td><td>256</td><td>128</td></tr><tr><td>Learning rate</td><td> $4 \times 1 0 ^ { - 5 }$ </td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>clip epsilon €</td><td>0.2</td><td>0.2</td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Gradient clipping</td><td>1.0</td><td>1.0</td></tr><tr><td>Dropout rate</td><td>0.10</td><td>0.10</td></tr><tr><td>Advantage normalization</td><td>Yes</td><td>Yes</td></tr><tr><td>Input normalization (ImageNet mean/std)</td><td>No</td><td>No</td></tr><tr><td>Reward normalization</td><td>No</td><td>No</td></tr><tr><td>PPO epochs</td><td>4</td><td>4</td></tr><tr><td>Total training steps</td><td>6M</td><td>3M</td></tr><tr><td> $\mathbf { M D R } \left( \alpha _ { 1 } , \alpha _ { 2 } \right)$ </td><td>(2,2)</td><td>(2,2)</td></tr></table>

Training is performed using PPO with Adam optimisation and Generalized Advantage Estimation (GAE) [15, 26]. Linear scheduling is additionally applied to the learning rate, entropy coefficient, and PPO clipping parameter throughout training. Hyperparameter tuning in RL can significantly affect final performance. Table 4 summarises the main training and optimisation hyperparameters yielding the best results. Notably, a relatively high initial entropy coefficient was found to be important for encouraging sufficient early exploration in the WSI environment. For Agent-15-Res50 specifically, the training budget was increased from 6M to 8M steps as it requires more training.

Patch-based UNI baselines use frozen UNI feature representations extracted from image patches resized to $2 2 4 \times 2 2 4$ . A lightweight two-layer MLP classifier is trained on top of the extracted features using cross-entropy loss and Adam optimisation with a learning rate of $1 0 ^ { - 5 }$ . Training for more than 6 epochs was found to overfit, and all classifiers were therefore trained for 4 epochs using balanced binary sampling and a 90/10 train–validation split. Final WSI segmentation masks are obtained through patch aggregation followed by connected-component filtering.

## D Inference time analysis details

To ensure a fair comparison, the inference time analysis for all models was conducted using the same hardware configuration. Specifically, each experiment was executed on a single NVIDIA Tesla V100 GPU (32 GB VRAM), 6 CPU cores of an Intel Xeon Gold 6126 processor, and 384 GB of system memory.

The best-case and worst-case scenarios correspond to WSIs requiring the shortest and longest processing times, respectively. For patch-based methods, processing time is primarily influenced by slide dimensions, as larger slides generate a greater number of patches that must be analyzed. For agent-based methods, processing time is additionally affected by the slide size through its impact on the image pyramid structure, including the pyramid depth and the availability of image representations at different downsampling levels. Consequently, larger WSIs generally require more computational effort and longer processing times.