# FineX: Fine-Grained Action Recognition with Cross-Attentive Latent Sparse Experts

Imtiaz Ul Hassan<sup>1⋆</sup> , Tasweer Ahmad<sup>1⋆</sup> , Nik Bessis<sup>1</sup> , and Ardhendu Behera<sup>1†</sup>

Computer Science Department, Edge Hill University, Ormskirk L39 4QP, United Kingdom

Abstract. Fine-grained human action recognition (FHAR) must distinguish visually similar actions that difer mainly in body configuration, timing, or local appearance. RGB representations retain visual context but often suppress joint-level geometry, whereas skeleton representations encode kinematics but discard dense spatial detail. We introduce FineX, which factorizes fine-grained cues into RGB appearance, poseheatmap geometry, and skeletal-graph topology. Pairwise cross-attention enables symmetric, stream-preserving information exchange, followed by a streamwise latent sparse Mixture-of-Experts that routes each representation to a content-dependent subset of shared experts, regularized by a load-balancing objective. FineX achieves state-of-the-art results on Gym99, Gym288, and Diving48. On the long-tailed Gym288, it raises mean class accuracy from 68.6% to 76.2% (+7.6 points) without textual supervision or large-scale vision–language pre-training, demonstrating the benefit of structured visual–pose–graph fusion and conditional expert refinement for FHAR.

Keywords: Fine-Grained Action Recognition · Human Behavior Analysis · Multimodal Action Recognition · Sparse Mixture of Experts

## 1 Introduction

Fine-grained human action recognition (FHAR) must distinguish action categories that share the same scene, actor, objects, and coarse motion, but difer in subtle body configurations or short temporal phases. Benchmarks like Fine-Gym [24] and Diving48 [17] emphasize this: classes often difer only in sub-action order, number of rotations, in-air posture, or slight limb-trajectory variations. Such cues are easily lost with global video descriptors and standard spatiotemporal pooling. FHAR therefore requires representations that retain local body evidence, capture temporal execution, and remain sensitive to fine structural diferences among visually similar actions.

Recent work advances FHAR by injecting human pose into video recognition. Pose-guided transformers align joint locations with visual tokens to improve pose–video interaction [38], while pose-enhanced vision-language models treat pose as an additional modality to adapt large pre-trained representations [39]. Recent multimodal methods further exploit RGB-skeleton collaboration and score-level fusion for more discriminative features [3]. In parallel, skeleton-based methods focus on fine-grained structural modeling via graph reasoning, decomposition, or hierarchical representation learning [5,7], highlighting the importance of body structure for FHAR. However, they still treat “pose” as a single auxiliary representation; joint coordinates, heatmap volumes or skeletal graphs, thus merging distinct fine-grained signals and limiting efective fusion.

We argue that fine-grained action evidence is best understood along three complementary axes. First, RGB appearance captures texture, body orientation, object or equipment contact, and local visual dynamics, but is biased by background context and loses joint-level precision after spatio-temporal aggregation. Second, dense pose heatmaps preserve joint and limb locations in the image plane, providing spatially grounded body configuration while suppressing appearance variation. Third, skeletal graphs encode inter-joint relations and temporal kinematic structure, but discard dense visual and image-plane context. These representations are not interchangeable: heatmaps retain where the body configuration appears, while graphs encode how body parts are topologically and temporally related. Collapsing them into a single pose stream removes a useful distinction for FHAR.

This observation motivates FineX, a cross-attentive, sparse-expert framework for fine-grained action recognition whose representation design explicitly follows the three evidence axes above. We use an R(2+1)D backbone [30] for RGB appearance, a PoseC3D pose-heatmap model with a SlowOnly-R50 backbone [7] for dense spatial pose, and ST-GCN++ [6] for skeletal graph topology. A lightweight fusion module then combines the three streams in two stages. First, pairwise cross-attention lets each stream query the other two while preserving stream identity, so appearance, heatmap, and graph features can exchange evidence without collapsing into a single concatenated representation. Second, a streamwise latent sparse Mixture-of-Experts (MoE) applies conditional nonlinear refinement to each cross-attended stream. Rather than being bound to fixed modalities, a shared expert bank is sparsely selected based on each representation’s content, allowing cue-dependent specialisation to emerge from data.

FineX difers from prior multimodal FHAR methods in both representation and fusion. Rather than treating pose as an auxiliary RGB signal [38, 39] or fusing RGB and skeleton features only at the score level [3], FineX separates dense spatial pose from skeletal topology and performs symmetric cross-stream reasoning over all three evidence sources. Unlike dense fusion layers that share one transformation across instances, its sparse expert module refines features per instance, which is crucial because FHAR classes and even samples within a class can depend on diferent cues: limb angles, body twists, rotation counts, transient poses, or appearance–pose interactions.

Contributions. (1) We formulate FHAR as three-stream evidence decomposition into RGB appearance, dense pose geometry, and skeletal topology. (2) We introduce symmetric pairwise cross-attention that preserves stream identity while exchanging complementary evidence. (3) We propose streamwise latent sparse expert routing for content-adaptive refinement without fixed modality– expert assignments. (4) FineX sets a new state-of-the-art on Gym99, Gym288, and Diving48; on Gym288, it achieves 94.3% Top-1 and 76.2% MCA, improving the previous best MCA by 7.6 points.

## 2 Related Work

Fine-grained video representation. Modern action-recognition methods learn spatio-temporal representations via segment aggregation [31,35], 3D or factorised convolutions [4,9,10,30], and video transformers [1,2,8,13,16,19,29]. These models work well when actions difer in scene, object interaction, or coarse motion, but FHAR demands sensitivity to shorter, more local execution patterns. Recent FHAR methods thus add stronger discriminative mechanisms on top of RGB features: DSFNet [27] selects informative temporal segments, MANet [14] focuses on motion-aware representations, and TAG-Head [12] augments a video backbone with relational reasoning. While they improve fine-grained discrimination, they still rely mainly on RGB appearance and implicit motion, so detailed body geometry and explicit joint structure must be inferred indirectly from pixels.

Pose and skeleton-based action recognition. Pose-based methods reduce appearance and background variation by directly modelling body configuration. Graph-based approaches represent joints and bones as a spatio-temporal graph, starting with ST-GCN [34] and extending to adaptive or stronger variants such as 2s-AGCN [26] and ST-GCN++ [6]. HiOD [5] further improves fine-grained skeleton recognition via hierarchical-aware feature disentanglement. A complementary line uses keypoint heatmaps instead of graph coordinates: PoseC3D [7] renders joints and limbs as spatio-temporal heatmap volumes and applies 3D convolutions, gaining robustness to pose noise while preserving image-plane spatial structure. These representations encode diferent inductive biases: graph models emphasise joint connectivity and kinematic relations, whereas heatmap models preserve dense spatial evidence near joints and limbs. Nevertheless, most pipelines either choose one representation or lump both into a single pose modality, leaving their complementarity under-explored.

Multimodal fine-grained action recognition. Multimodal FHAR methods increasingly pair RGB video with pose or semantic supervision. PGVT [38] uses pose-guided temporal and spatial attention to link body joints with video tokens. PeVL [39] injects pose into a vision–language framework via cross-modal refinement and semantic contrastive loss, extending video–text models such as ActionCLIP [32] and X-CLIP [21]. MFCF [3] strengthens RGB–skeleton collaboration through skeleton-guided localisation, contrastive learning, and score-level fusion. These works show the value of combining appearance and body structure, but fusion is usually restricted to RGB and a single pose-derived stream. They do not jointly model dense pose geometry and explicit skeletal topology as separate, interacting streams, and their pairwise or late fusion lacks a shared mechanism for all streams to selectively query one another.

![](images/a5efad4fc5e74e12fcfb57e34b3ec84504b253a8bb5eb01efb351b7904dacfda.jpg)  
Fig. 1: FineX architecture. Frozen RGB, pose-heatmap, and skeletal-graph backbones provide pre-cached features, which are projected to a shared space. Pairwise cross-attention contextualises each stream using the other two, after which a shared sparse MoE routes every refined stream representation to its top-k latent experts. The routed outputs are mean-pooled for classification.

Adaptive experts and long-tailed video recognition. Mixture-of-Experts models enable conditional computation by selecting diferent transformations per input. In long-tailed video recognition, prior work tackles imbalance using class-aware reconstruction and specialised prediction heads [23], while MEID [15] uses multiple experts with internal distillation to improve recognition on imbalanced classes. These approaches highlight the benefit of expert diversity but rely on a single visual representation and do not examine routing over multiple pose and appearance streams. FineX instead couples symmetric pairwise cross-attention with a shared latent expert bank: cross-attention exchanges evidence across RGB, pose-heatmap, and skeletal-graph representations, and sparse routing conditionally refines each stream. This diferentiates FineX from both two-stream multimodal fusion and single-representation expert models.

## 3 Methodology

Fig. 1 overviews FineX. Three frozen backbones encode RGB appearance, poseheatmap geometry, and skeletal topology. Their features are projected to a common space, contextualised by pairwise cross-attention, and conditionally refined through a shared sparse MoE before classification.

Problem formulation. Consider $S$ video clips $\mathcal { D } = \{ ( \mathcal { V } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { S }$ , where $y _ { i } \in$ $\{ 1 , \ldots , C \}$ is the class label of clip $\nu _ { i }$ . For each clip, three frozen backbones produce features $\mathbf { f } _ { m } = B _ { m } ( \nu _ { i } )$ for $m \in \mathcal { M } = \{ r , s , g \}$ . Here, $r , s , g$ denote RGB, pose heatmap, and skeletal graph, with $\mathbf { f } _ { r } , \mathbf { f } _ { s } \in \mathbb { R } ^ { 5 1 2 }$ and $\mathbf { f } _ { g } \in \mathbb { R } ^ { 2 5 6 }$ . FineX learns $\mathcal { F } _ { \theta } : ( \mathbf { f } _ { r } , \mathbf { f } _ { s } , \mathbf { f } _ { g } ) \mapsto \hat { \mathbf { y } } \in \mathbb { R } ^ { C }$ , where $\hat { \mathbf { y } }$ are the predicted class logits.

Frozen stream representations. The RGB stream uses $\mathrm { R } ( 2 + 1 ) \mathrm { D } { - } 3 4$ [30] initialised from IG-65M [11]; the pose-heatmap stream uses PoseC3D with a SlowOnly-R50 backbone [7]; and the skeletal stream uses ST-GCN++ [6]. Their penultimate-layer features are pre-cached before fusion training, decoupling the fusion mechanism from further backbone updates and letting the three heterogeneous architectures interface via fixed-dimensional representations.

## 3.1 Pairwise Cross-Attention Fusion

Direct concatenation obscures stream-specific contributions. Instead, we map each feature to a shared dimension $D \colon \mathbf { t } _ { m } ^ { ( 0 ) } = \phi _ { m } ( \mathbf { f } _ { m } ) , \ m \in \mathcal { M } .$ In layer $\ell \in$ $\{ 1 , \ldots , L \}$ , each modality $m \in \mathcal { M }$ is contextualised by the other two modalities and is denoted $\mathbf { C } _ { m } ^ { ( \ell - 1 ) }$

$$
\mathbf { C } _ { m } ^ { ( \ell - 1 ) } = \bigl [ \mathbf { t } _ { m ^ { \prime } } ^ { ( \ell - 1 ) } \bigr ] _ { m ^ { \prime } \in \mathcal { M } \backslash \{ m \} } \in \mathbb { R } ^ { 2 \times D } .\tag{1}
$$

Each stream queries this context and is updated using post-normalised multihead residual attention:

$$
\mathbf { t } _ { m } ^ { ( \ell ) } = \mathrm { L N } \Bigl [ \mathbf { t } _ { m } ^ { ( \ell - 1 ) } + \mathrm { M H A } ^ { ( \ell ) } \Bigl ( \mathbf { t } _ { m } ^ { ( \ell - 1 ) } , \mathbf { C } _ { m } ^ { ( \ell - 1 ) } , \mathbf { C } _ { m } ^ { ( \ell - 1 ) } \Bigr ) \Bigr ] .\tag{2}
$$

The context is the source for both keys and values, but separate learned projections are used in every head: $\mathbf { Q } _ { m } ^ { ( \ell , h ) } = \mathbf { t } _ { m } ^ { ( \ell - 1 ) } \mathbf { W } _ { Q } ^ { ( \ell , h ) } , \mathbf { K } _ { m } ^ { ( \ell , h ) } = \mathbf { C } _ { m } ^ { ( \ell - 1 ) } \mathbf { W } _ { K } ^ { ( \ell , h ) }$ ， $\mathbf { V } _ { m } ^ { ( \ell , h ) } = \mathbf { C } _ { m } ^ { ( \ell - 1 ) } \mathbf { W } _ { V } ^ { ( \ell , h ) }$ . Thus $\mathbf { Q } \in \mathbb { R } ^ { 1 \times d _ { h } }$ and K, $\mathbf { V } \in \mathbb { R } ^ { 2 \times d _ { h } }$ , where $d _ { h } = D / H$ Head h computes

$$
\alpha _ { m } ^ { ( \ell , h ) } = \mathrm { s o f t m a x } \left( \frac { \mathbf { Q } _ { m } ^ { ( \ell , h ) } ( \mathbf { K } _ { m } ^ { ( \ell , h ) } ) ^ { \top } } { \sqrt { d _ { h } } } \right) , \qquad \mathbf { a } _ { m } ^ { ( \ell , h ) } = \boldsymbol { \alpha } _ { m } ^ { ( \ell , h ) } \mathbf { V } _ { m } ^ { ( \ell , h ) } .\tag{3}
$$

The H heads are concatenated and projected by $\mathbf { W } _ { O } ^ { ( \ell ) }$ . Parameters are shared across stream updates within a layer, but each stream supplies a diferent query; hence the three outputs retrieve diferent complementary evidence. Stacking L layers yields $\mathcal { T } ^ { ( L ) } = \dot { \{ \mathbf { t } _ { r } ^ { ( L ) } , \mathbf { t } _ { s } ^ { ( L ) } , \mathbf { t } _ { g } ^ { ( L ) } \} }$

## 3.2 Streamwise Latent Sparse Mixture-of-Experts (MoE)

Diferent fine-grained instances rely on diferent cues, so a single dense transformation is too restrictive. FineX instead introduces a shared sparse MoE separately to each cross-attended stream. Experts are latent rather than modalityspecific: RGB, heatmap, and graph representations share one expert bank, with routing conditioned on each representation’s content.

For sample b and stream $m ,$ let $\mathbf { x } _ { b , m } = \mathbf { t } _ { b , m } ^ { ( L ) } \in \mathbb { R } ^ { D }$ denote the final cross attention feature representation to be routed through the sparse MoE. A biasfree linear router predicts logits over $N$ experts:

$$
\mathbf { z } _ { b , m } = \mathbf { W } _ { R } \mathbf { x } _ { b , m } \in \mathbb { R } ^ { N } .\tag{4}
$$

The active set $\mathcal { A } _ { b , m } = \mathrm { T o p K } _ { k } ( \mathbf { z } _ { b , m } )$ contains k experts, whose weights are renormalised as

$$
\pi _ { b , m , i } = \frac { \exp ( z _ { b , m , i } ) } { \sum _ { j \in \mathcal { A } _ { b , m } } \exp ( z _ { b , m , j } ) } , \qquad i \in \mathcal { A } _ { b , m } .\tag{5}
$$

Each expert is a bottleneck feed-forward network,

$$
E _ { i } ( \mathbf { x } ) = \mathbf { W } _ { 2 } ^ { ( i ) } \mathrm { D r o p o u t } \Big ( \mathrm { G E L U } ( \mathbf { W } _ { 1 } ^ { ( i ) } \mathbf { x } ) \Big ) , \quad \mathbf { W } _ { 1 } ^ { ( i ) } \in \mathbb { R } ^ { d _ { e } \times D } , ~ \mathbf { W } _ { 2 } ^ { ( i ) } \in \mathbb { R } ^ { D \times d _ { e } } ,\tag{6}
$$

where $d _ { e }$ is the expert bottleneck dimension. Sparse refinement and stream aggregation are

$$
\tilde { \mathbf { t } } _ { b , m } = \sum _ { i \in \mathcal { A } _ { b , m } } \pi _ { b , m , i } E _ { i } ( \mathbf { x } _ { b , m } ) , \qquad \mathbf { u } _ { b } = \frac { 1 } { | \mathcal { M } | } \sum _ { m \in \mathcal { M } } \tilde { \mathbf { t } } _ { b , m } .\tag{7}
$$

Finally, $\hat { \mathbf { y } } _ { b } = \mathbf { W } _ { C } \mathrm { D r o p o u t } ( \mathrm { L N } ( \mathbf { u } _ { b } ) ) + \mathbf { b } _ { C }$ . Cross-attention therefore controls evidence exchange, whereas sparse routing selects the nonlinear refinements applied to each contextualised stream.

## 3.3 Training Objective

FineX is trained with label-smoothed cross-entropy and an auxiliary load balancing loss for sparse routing, preventing the router from collapsing onto a few experts. For each mini-batch $B ,$ , let $y _ { b }$ and $\hat { \mathbf { y } } _ { b }$ be the ground-truth label and prediction for video $b ,$ and compute the loss as:

$$
\mathcal { L } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \mathrm { C E } _ { \epsilon } ( \hat { \mathbf { y } } _ { b } , y _ { b } ) + \lambda _ { \mathrm { l b } } \mathcal { L } _ { \mathrm { l b } } .\tag{8}
$$

Let $\mathcal { R } = \{ ( b , m ) : b = 1 , \ldots , B , m \in \mathcal { M } \}$ be the 3B routing decisions, $\mathbf { p } _ { b , m } =$ softmax $\left( \mathbf { z } _ { b , m } \right)$ , and $\mathcal { A } _ { b , m }$ the active top-k set. Following standard sparse-MoE training [25], we define

$$
\begin{array} { r l } & { f _ { i } = \displaystyle \frac { 1 } { | \mathcal { R } | } \sum _ { ( b , m ) \in \mathcal { R } } \mathbb { 1 } [ i \in \mathcal { A } _ { b , m } ] , \quad q _ { i } = \displaystyle \frac { 1 } { | \mathcal { R } | } \sum _ { ( b , m ) \in \mathcal { R } } p _ { b , m , i } , } \\ & { \mathcal { L } _ { \mathrm { l b } } = N \displaystyle \sum _ { i = 1 } ^ { N } f _ { i } q _ { i } . } \end{array}\tag{9}
$$

Here $f _ { i }$ measures expert selection frequency and $q _ { i }$ supplies a diferentiable signal through the full router distribution. Their product penalises experts that simultaneously attract excessive assignments and probability mass. We use $\epsilon = 0 . { \dot { } }$ 1 and $\lambda _ { \mathrm { l b } } = 0 . 1$

Table 1: Comparison with SOTA. Top-1 accuracy and Mean Class Accuracy (MCA) in % on Gym99/Gym288, and Top-1 accuracy on Diving48. R → RGB, T → Text, P → Pose.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Venue</td><td rowspan="2">Input</td><td colspan="2">Gym99</td><td colspan="2">Gym288</td><td>Diving48</td></tr><tr><td>Top-1</td><td>MCA</td><td>Top-1 MCA</td><td>Top-1</td></tr><tr><td rowspan="9">TSN [31] I3D [4] TRNms [40] ST-GCN++† [6] TSM [18]</td><td>TPAMI&#x27;18 CVPR&#x27;17</td><td>R</td><td>74.8</td><td>61.4</td><td>68.3</td><td>26.5</td><td></td></tr><tr><td></td><td>R</td><td>75.6</td><td>64.4</td><td>66.1</td><td>28.2</td><td></td></tr><tr><td>ECCV&#x27;18</td><td>R</td><td>79.5</td><td>68.8</td><td>73.1</td><td>32.0</td><td></td></tr><tr><td>ACM MM&#x27;22</td><td>P</td><td>94.2</td><td>91.9</td><td>85.9</td><td>61.3</td><td>85.9</td></tr><tr><td>ICCV&#x27;19</td><td>R</td><td>80.4</td><td>70.6</td><td>73.5</td><td>34.8</td><td></td></tr><tr><td>CVPR&#x27;22</td><td>P</td><td>94.4</td><td>92.0</td><td>87.2</td><td>61.4</td><td>82.8</td></tr><tr><td>ICCV&#x27;19</td><td>R</td><td>93.9</td><td>90.6</td><td>86.8</td><td>51.2</td><td></td></tr><tr><td>CVPR&#x27;21</td><td>R</td><td>93.8</td><td>90.6</td><td>89.6</td><td>61.9</td><td>81.8</td></tr><tr><td>TOMM&#x27;24</td><td>R</td><td>90.3</td><td>86.3</td><td>84.9</td><td>47.6</td><td>85.0</td></tr><tr><td>MANet [14]]</td><td>CIS&#x27;25</td><td>R</td><td></td><td></td><td></td><td></td><td>81.8</td></tr><tr><td>TAG-Head [12]</td><td>ICPR&#x27;26</td><td>R</td><td>95.6</td><td>93.8</td><td>92.2</td><td>68.6</td><td></td></tr><tr><td>HiOD [5]</td><td>ICCV&#x27;25</td><td>P</td><td>96.3</td><td></td><td></td><td></td><td></td></tr><tr><td>PGVT [38]</td><td>WACV&#x27;24</td><td>R+P</td><td>96.7</td><td>91.6</td><td>91.0</td><td>63.6</td><td>91.3</td></tr><tr><td>MFCF [3]</td><td>BMVC&#x27;25</td><td>R+P</td><td>96.0</td><td></td><td></td><td></td><td></td></tr><tr><td>PeVL [39] Ours</td><td>CVPR&#x27;24</td><td>R+T+P R+P</td><td>97.0 97.1</td><td>91.8 96.4</td><td>91.8 94.3</td><td>64.0 76.2</td><td>92.5 92.9</td></tr></table>

<sup>†</sup> ST-GCN++ and PoseC3D (SlowOnly backbone) were trained from scratch using the pose data extracted by our pipeline.

## 4 Experiments

## 4.1 Experimental Setup

Datasets and metrics. Gym99/Gym288 [24] contain 99/288 gymnastics classes with 20K/23K training and 8.5K/9.6K test clips; Gym288 is highly long-tailed. Diving48 [17] has 16.1K training and 2.3K test clips from 48 dive classes. We use oficial splits and report Top-1 on all datasets and MCA on Gym99/Gym288. Pose and training protocol. We use PySKL HRNet keypoints [6, 28] for Gym99, and D-FINE [22] plus ViTPose-L [33] for Gym288 and Diving48, keeping the largest person per frame. Stage 1 adapts the three backbones independently; Stage 2 freezes them and trains FineX on pre-cached features to isolate fusion gains. Unless noted, D=512, L=3, H=8, N=16, d =256, and k=8. The fusion module is trained for up to 80 epochs with Adam (base learning rate $3 \times 1 0 ^ { - 4 }$ weight decay $1 0 ^ { - 3 } )$ , cosine scheduling, gradient clipping, label smoothing, and Eq. (9). Further implementation details are in the supplement.

## 4.2 Comparison with State-of-the-Art (SOTA)

Table 1 compares FineX with representative RGB, pose-based, multimodal, and vision–language methods. We report Top-1 accuracy for all datasets and MCA for Gym99/Gym288, where class imbalance makes sample-averaged accuracy inadequate. FineX achieves the best performance on all benchmarks, with the largest gains on Gym288, the most fine-grained and long-tailed setting.

Gym288: Gym288 is the most diagnostic benchmark in Table 1 because it combines 288 fine-grained classes with a highly long-tailed distribution. FineX achieves 94.3% Top-1 and 76.2% MCA, surpassing the strongest prior MCA result, TAG-Head [12], by +7.6 points and the vision–language baseline PeVL [39] by +12.2 points. The MCA gain is crucial: Top-1 is dominated by frequent classes, whereas MCA weights classes equally and is more sensitive to tail performance. This is reflected in the Top-1–MCA gap: TAG-Head and PeVL reach high Top-1 accuracy but still have gaps of 23.6 and 27.8 points, respectively. FineX reduces this gap to 18.1 points, the smallest among methods reporting both metrics on Gym288, indicating better class-balance rather than improvements limited to head classes.

These results support FineX’s core design. RGB-only methods such as TQN [37] and TAG-Head [12] ofer strong appearance and motion cues but lose jointlevel spatial detail after spatio-temporal aggregation. Pose-only models such as PoseC3D and ST-GCN++ preserve body structure but lack appearance context to distinguish visually similar actions. Multimodal methods such as PGVT [38] and PeVL [39] improve over single-stream baselines, yet do not explicitly disentangle dense pose geometry from skeletal graph topology. FineX fuses these complementary signals with stream-preserving cross-attention and then applies content-adaptive sparse expert refinement, achieving the largest gains where single-stream evidence is weakest. The per-class analysis in Fig. 4 shows that these gains are concentrated on rare and mid-frequency classes.

Gym99: On Gym99, FineX reaches 97.1% Top-1 and 96.4% MCA (Table 1). Its Top-1 gains are modest, +0.1 over PeVL [39] and +0.8 over HiOD [5], indicating that Gym99 is nearing saturation around 96–97% Top-1. MCA remains more revealing: FineX outperforms PeVL by +4.6 and TAG-Head [12] by +2.6. Its Top-1–MCA gap is only 0.7, smaller than for other methods with both metrics reported, showing that FineX improves recognition consistently across classes rather than only on easy ones. Because Gym99 is less long-tailed than Gym288, these gains indicate that the proposed cross-modal decomposition also enhances fine-grained discrimination in a more balanced setting.

Diving48: FineX achieves 92.9% Top-1 accuracy, surpassing the vision–language baseline PeVL [39] by +0.4 and the strongest unimodal pose baseline ST-GCN++ by +7.0. Diving48 reduces scene-context bias: most clips share a similar aquatic setting and difer mainly in take-of mechanics, somersault and twist count, body configuration, and entry posture. In this regime, textual semantics or scene priors add little; the key signal is fine-grained motion and body structure. FineX fits this setting because pose heatmaps preserve image-plane joint geometry, skeletal graphs capture inter-joint kinematics, and RGB features add complementary appearance cues. These results show that structured visual–pose–graph fusion can match or exceed vision–language alignment when recognition hinges on subtle motion execution rather than semantic context.

Across all datasets, the comparison suggests that FineX improves not by simply increasing model scale, but by aligning the fusion architecture with the evidence required by FHAR. The largest gains appear where fine-grained cues are most dificult to recover from any single stream: long-tailed categories in Gym288 and motion-structured classes in Diving48.

Table 2: Component ablation. Cross-attention exchanges evidence across streams, while sparse MoE performs streamwise conditional refinement. Full FineX uses both.
<table><tr><td>Component</td><td></td><td>Gym99</td><td>Gym288</td><td>Diving48</td></tr><tr><td>Cross-attn.</td><td>Sparse MoE</td><td>Top-1 MCA</td><td>Top-1 MCA</td><td>Top-1</td></tr><tr><td>×</td><td></td><td>95.3 95.2</td><td>92.4 73.6</td><td>92.1</td></tr><tr><td>√</td><td>X</td><td>95.8 95.2</td><td>92.7 74.1</td><td>92.1</td></tr><tr><td>了</td><td>√</td><td>97.1 96.4</td><td>94.3 76.2</td><td>92.9</td></tr></table>

## 4.3 Ablation Studies

We run ablations on all three benchmarks to quantify the impact of FineX’s main design choices: pairwise cross-attention, streamwise sparse expert routing, cross-stream complementarity, load balancing, and the number of active experts. Unless noted, each experiment varies only one factor while keeping the rest of the training protocol fixed.

Efect of cross-attention and sparse experts. Table 2 isolates the two core components of FineX by comparing: (i) MoE only, where cross-attention is removed, and stream representations are routed directly through the sparse expert layer; (ii) cross-attention only, where the MoE is replaced by a shared dense transformation; and (iii) the full model. Both components help and have complementary efects. On Gym288, removing cross-attention reduces performance from 94.3/76.2 to 92.4/73.6 Top-1/MCA, while removing the sparse MoE gives 92.7/74.1. Cross-attention yields the larger individual gain, consistent with its role in exchanging evidence across RGB, pose-heatmap, and skeletal-graph streams. The sparse MoE then adds further improvement, indicating that conditional refinement remains useful after the streams are contextualised.

The same pattern appears on Gym99 and Diving48. On Gym99, both ablated variants reach 95.2 MCA, while the full model achieves 96.4, indicating that the modules are jointly beneficial even on a less long-tailed benchmark. On Diving48, both single-component variants obtain 92.1 Top-1, whereas FineX reaches 92.9. Thus, neither cross-attention nor sparse routing alone fully exploits the finegrained kinematic structure of diving actions; their combination yields the most consistent gains across datasets.

Cross-stream complementarity. Table 3 assesses whether the three evidence streams are redundant or complementary. Single-stream models are strong but dataset-dependent: pose heatmaps perform best on Gym99, RGB on Gym288 and Diving48, and skeletal graphs remain competitive but drop on the long-tailed Gym288. Thus, no single representation dominates across all fine-grained settings. Pairwise fusion clearly outperforms single streams, especially on Gym288: the best single stream reaches 89.1 Top-1 and 68.0 MCA, while the RGB+skeletal graph attains 93.6 Top-1 and 74.8 MCA. This gain shows that appearance and graph-based kinematics provide complementary cues: RGB supplies visual context and local motion, while the skeletal graph encodes inter-joint structure and temporal body dynamics. The RGB+skeletal graph is also the strongest pair on all three benchmarks, indicating that these two streams are the least redundant.

The full three-stream model further improves over the best pair on every dataset. Gains are small on Gym99, where performance is near saturation, but larger on Gym288 MCA (+1.4) and Diving48 Top-1 (+1.2). This demonstrates that the pose-heatmap stream adds spatial evidence even when it is not the best standalone modality. Heatmaps preserve image-plane joint and limb locations that the skeletal graph abstracts away, and this signal becomes useful when cross-attention conditions it on RGB appearance and graph topology. Overall, the results support FineX’s key motivation: dense pose geometry and skeletal topology should be treated as distinct evidence sources rather than merged into a single pose modality.

Table 3: Cross-stream complementarity R → RGB, P → Pose-Heatmap, $\textrm { S } $ Skeletal Graph. We compare single-stream, pairwise, and full three-stream fusion. The strongest pair is shaded, while full model is in Bold.
<table><tr><td colspan="2">Stream</td><td colspan="2">Gym99</td><td colspan="2">Gym288</td><td>Diving48</td></tr><tr><td>R P</td><td>S</td><td>Top-1</td><td>MCA</td><td>Top-1</td><td>MCA</td><td>Top-1</td></tr><tr><td>√ ×</td><td>×</td><td>93.6</td><td>91.5</td><td>89.1</td><td>68.0</td><td>87.5</td></tr><tr><td>X X</td><td>V</td><td>94.1</td><td>92.1</td><td>85.6</td><td>62.8</td><td>86.2</td></tr><tr><td>X √</td><td>X</td><td>94.6</td><td>92.5</td><td>86.3</td><td>67.7</td><td>82.7</td></tr><tr><td>× √</td><td>V</td><td>95.6</td><td>94.0</td><td>87.9</td><td>68.5</td><td>87.4</td></tr><tr><td>√ √</td><td>×</td><td>96.3</td><td>94.9</td><td>93.2</td><td>74.3</td><td>89.6</td></tr><tr><td>√ X</td><td>√</td><td>96.8</td><td>95.9</td><td>93.6</td><td>74.8</td><td>91.7</td></tr><tr><td>√ V</td><td>V</td><td>97.1</td><td>96.4</td><td>94.3</td><td>76.2</td><td>92.9</td></tr></table>

Table 4: Efect of the load-balance loss coeficient $\lambda _ { \mathrm { l b } }$ in $\operatorname { E q . }$ (8). $( \lambda _ { \mathrm { l b } } = 0 \to \mathrm { C E } )$  
Table 5: Efect of top-k expert selection with $N { = } 1 6$ experts. $\scriptstyle ( k = 1 6 \to$ Dense)

<table><tr><td rowspan="2"> $\lambda _ { \mathrm { l b } }$ </td><td colspan="2">Gym99</td><td colspan="2">Gym288</td><td>Diving48</td></tr><tr><td> $\scriptstyle { \overline { { \mathrm { ~ T o p } - 1 } } }$ </td><td>MCA</td><td>Top-1</td><td>MCA</td><td>Top-1</td></tr><tr><td rowspan="3">0 0.05</td><td>96.9</td><td>96.0</td><td>94.1</td><td>75.2</td><td>92.3</td></tr><tr><td>97.0</td><td>96.1</td><td>94.1</td><td>75.8</td><td>92.5</td></tr><tr><td>97.1</td><td>96.4</td><td>94.3</td><td>76.2</td><td>92.9</td></tr><tr><td>0.1 0.2</td><td>97.0</td><td>96.1</td><td>94.2</td><td>75.7</td><td>92.6</td></tr></table>

<table><tr><td rowspan="2">k</td><td colspan="2">Gym99</td><td colspan="2">Gym288</td><td>Diving48</td></tr><tr><td>Top-1</td><td>MCA</td><td>Top-1</td><td>MCA</td><td>Top-1</td></tr><tr><td>1</td><td>96.2</td><td>95.2</td><td>93.3</td><td>73.7</td><td>90.2</td></tr><tr><td>4</td><td>97.0</td><td>96.1</td><td>94.0</td><td>74.6</td><td>91.9</td></tr><tr><td>8</td><td>97.1</td><td>96.4</td><td>94.3</td><td>76.2</td><td>92.9</td></tr><tr><td>16</td><td>97.1</td><td>96.2</td><td>94.1</td><td>75.1</td><td>92.2</td></tr></table>

Efect of load-balancing. Table 4 analyses the coeficient $\lambda _ { \mathrm { l b } }$ in Eq. (8). Adding the load-balancing term improves all metrics over cross-entropy alone, with the largest gain on Gym288 MCA (75.2 to 76.2 at $\lambda _ { \mathrm { l b } } = 0 . 1 )$ , while Top-1 changes only slightly. This matches MCA’s higher sensitivity to class-wise failures and Top-1’s focus on frequent classes, so balanced expert utilisation matters most in the long-tailed setting. Performance peaks at $\lambda _ { \mathrm { l b } } = 0 . 1$ across all benchmarks. Increasing $\lambda _ { \mathrm { l b } }$ to 0.2 hurts performance (Gym288 MCA: 76.2 to 75.7), suggesting that regulariser must be strong enough to avoid expert under-utilisation but not so strong that it impedes specialisation from sparse routing. We, therefore, set $\lambda _ { \mathrm { l b } } = 0 . 1$ in all main experiments.

Efect of top-k expert routing. Table 5 varies the number of active experts per stream while fixing the expert pool to $N = 1 6$ . Setting $k = 1$ gives hard single-expert routing, and k = 16 activates all experts, removing sparsity. Performance improves from k = 1 to $k = 8$ (Gym288 MCA 73.7→76.2), indicating that single-expert routing is too restrictive for FHAR, where multiple fine-grained cues must be combined within a stream. Dense activation is also suboptimal: increasing k from 8 to 16 reduces Gym288 MCA from 76.2 to 75.1 and Diving48 Top-1 from 92.9 to 92.2. With all experts active, the router cannot enforce sparse, specialised transformations, and each token is processed by the full expert bank, weakening the intended conditional computation. The best trade-of is at k = 8, which keeps a soft mixture of experts with sparse, contentdependent routing. We adopt k = 8 as the default. Further ablation results for varying numbers of experts are provided in supplementary material.

![](images/e2f27f2d980b9916bab75b659b1693dae40b9fca5da49746c19c97b1b86afde7.jpg)  
(a) RGB

![](images/8962f9cb495fa66672b3394022849a9a755cdb3fa37c7dedad72c1ba8fb79ac8.jpg)  
(b) Pose heatmap

![](images/eb6b337f93d2e57ff26e440b221a3da9888317fe802de64a1617a4ec04335572.jpg)  
(c) Skeleton graph

![](images/f3ed789c61bb616882a33ba811b4e804faa683cfc09ee8b8c6b0c2d5642d0ebb.jpg)  
(d) FineX  
Fig. 2: t-SNE visualisation of feature embeddings on Gym288. Embeddings from (a) RGB R(2+1)D, (b) pose-heatmap PoseC3D–SlowOnly, (c) skeletal-graph ST-GCN++, and (d) the final FineX representation, for 10 randomly selected classes. Single-stream embeddings exhibit noticeable class overlap, while FineX forms more compact and better separated clusters.

## 4.4 Qualitative Analysis

Embedding separability. Fig. 2 shows t-SNE embeddings [20] for 10 randomly chosen classes for Gym288, illustrating the limits of single evidence streams. RGB features capture appearance and local motion but confuse classes with similar visual context and diferent execution details. Pose-heatmap features preserve image-plane joint locations but can merge classes with similar instantaneous poses. Skeletal-graph features model inter-joint topology and temporal kinematics but discard dense visual context useful for distinguishing visually similar phases. FineX yields tighter within-class clusters and clearer inter-class separation on all three datasets. This matches the roles of the two fusion stages: pairwise cross-attention reduces single-stream ambiguity by conditioning each representation on the other two streams, and streamwise sparse MoE provides content-adaptive nonlinear refinement. While t-SNE is only a qualitative projection and not a standalone metric, its geometry aligns with the quantitative results in Tables 1 and 3, indicating that the fused representation is more discriminative than any individual stream. Furthermore, t-SNE visualisation for Gym99 and Diving48 is provided in supplementary section.

Expert routing on Diving48. Fig. 3 analyses expert routing at both coarse and fine-grained levels. Fig. 3(a) shows a Sankey diagram that aggregates dominantexpert assignments by take-of type, with flow width indicating the number of routed tokens. The routing pattern follows motion structure: board-facing Reverse and Inward take-ofs rely on a smaller subset of dominant experts than Forward and Backward. Importantly, no expert is exclusive to a single take-of type. Instead, experts are shared across categories, capturing reusable motion cues rather than simply memorising high-level labels.

Fig. 3(b–c) then focuses on three randomly chosen fine-grained classes per take-of type (each with at least 20 validation clips), summarising routing over the five most frequently used experts plus an Others group. Even within this restricted expert pool, routing remains class-specific: distinct fine-grained classes under the same take-of type are typically dominated by diferent experts from the pool. Inward classes predominantly route to E2 and E7, with a minor fraction going to Others. Forward trafic is split mainly between E1 and Others, with a smaller contribution from E6. Reverse mainly routes to E15, then E6, with a small portion in Others. Backward is dominated by E2, with a smaller share in Others and a few clips in E1. Clips belonging to the same class form compact, largely well-separated clusters, with only occasional outliers beyond their class boundary. Altogether, the three panels indicate that take-of type restricts the subset of experts a clip uses, but does not reduce routing to a single shared expert; fine-grained classes within the same take-of type still preferentially route to diferent experts within that subset.

![](images/06ac665ab0aeb41b86b4b2571fda7ab0e649ec095ae43706e2dbbbea501a5fa4.jpg)  
Fig. 3: Expert routing on Diving48 at both coarse and fine-grained resolutions. (a) Sankey diagram summarising dominant expert selections by take-of type, with flow width indicating the number of routed tokens. (b–c) t-SNE visualisation of concatenated stream-level routing weights, with three fine-grained classes per take-of type (up to 150 clips each); the embedding is identical in both panels, difering only in colour/fill. (b) Colour encodes take-of type, and marker shape encodes fine-grained class. (c) Fill indicates the dominant expert (the five most frequently selected experts shown individually, all others grouped as Others); edge colour shows take-of type, and dashed outlines indicate the approximate cluster boundary for each take-of type.

Long-tail per-class behaviour. Fig. 4 further analyses Gym288 by sorting classes from rarest to most frequent by number of training samples. The frequency histogram confirms the benchmark’s severe long-tailed distribution, with many categories containing at most 11 training instances. Thus MCA is a more informative metric than Top-1 accuracy, as it measures class-wise recognition instead of being dominated by frequent categories.

The single-stream backbones reach similar MCA and saturate near 61%: R(2+1)D at 61.8%, SlowOnly at 61.4%, and ST-GCN++ at 61.3%. FineX boosts MCA to 76.2%, a +14.4 gain over the best single-stream model. This improvement is strongly frequency-dependent: FineX achieves the largest gains on rare classes (+24–28), followed by mid-frequency classes (+8–10), with smaller but consistent gains on frequent classes (+3–8).

![](images/198e8fa8ade58c9e03e148b0b26cf39b2a2f808d49e9e62a025942963547cbda.jpg)  
Fig. 4: Per-class accuracy on Gym288 sorted by training frequency. Top: class-frequency histogram. Bottom: per-class accuracy of FineX vs. the three singlestream backbones. FineX gives the largest gains on rare classes.

This pattern is key. For frequent classes, each stream sees enough data to learn discriminative cues, so fusion adds only modest gains. For rare classes, relying on a single representation is fragile: RGB may miss joint-level structure, pose heatmaps may miss topological relations, and skeletal graphs may miss visual context. FineX addresses by fusing appearance, dense pose geometry, and skeletal topology before sparse expert refinement. Thus, long-tail analysis reinforces the main finding on Gym288: FineX boosts not only average accuracy but also class-balanced performance for scarce training. The same complementarity yields robustness to noisy pose: PoseC3D’s heatmaps better tolerate joint noise than raw coordinates [7], while dual-pose decomposition adds strength over individual stream. Crucially, cross-attention weighting attenuates corrupted pose queries, suppressing unreliable streams rather than propagating their errors.

## 4.5 Computational Eficiency

FineX separates backbone representation learning from cross-modal fusion. In Stage 1, the three modality-specific backbones are fine-tuned individually and then frozen; R(2+1)D-34, PoseC3D with SlowOnly-R50 Backbone, and ST-GCN++ together have 67.7M parameters. In Stage 2, we train only the 7.5Mparameter fusion module on pre-cached features. Because it operates on compact feature vectors $( \mathbf { f } _ { r } , \mathbf { f } _ { s } \in \mathbb { R } ^ { 5 1 2 }$ and $\mathbf { f } _ { g } \in \mathbb { R } ^ { 2 5 6 } )$ , its cost is independent of the backbones’ spatial and temporal resolution, making Stage 2 training lightweight and avoiding repeated backpropagation through the video backbones.

Comparison with parameter-eficient video adaptation. Table 6 compares FineX with AIM [36] on Diving48 in terms of total parameters, Stage 2 tunable parameters, single-clip GFLOPs and Top-1 accuracy. As in previous adapter-based work, the Tunable column counts only parameters updated in

Table 6: Diving48 Eficiency comparison. Tunable parameters refer Stage2 fusion parameters. GFLOPs are measured for single-clip inference. R→RGB, P→Pose-Heatmap.
<table><tr><td>Method</td><td>Total (M)</td><td>Tunable (M)</td><td>GFLOPs</td><td>Top-1</td><td>Input</td></tr><tr><td>AIM ViT-B/16 [36]</td><td>97</td><td>11</td><td>809</td><td>88.9</td><td>R</td></tr><tr><td>AIM ViT-L/14 [36]</td><td>341</td><td>38</td><td>3,736</td><td>90.6</td><td>R</td></tr><tr><td>FineX (ours)</td><td>75.2</td><td>7.5</td><td>682</td><td>92.9</td><td>R+P</td></tr></table>

Stage 2. FineX has 75.2M total parameters (three frozen backbones plus fusion), fewer than AIM ViT-B/16 (97M) and far fewer than AIM ViT-L/14 (341M). For inference, FineX uses 682 GFLOPs per clip, versus 809 GFLOPs for AIM ViT-B/16 and 3,736 GFLOPs for AIM ViT-L/14.

Despite invoking three backbones, FineX incurs only 682 GFLOPs per clip, which is 5.5× less than AIM ViT-L/14 (3,736) and below AIM ViT-B/16, and since three streams share no data dependency, they can execute in parallel on separate devices, without adding wall-clock latency. It also attains higher Diving48 accuracy than both AIM variants with fewer tunable parameters and lower GFLOPs, showing that improvements come from cross-modal decomposition and eficient fusion of experts rather than simply scaling model capacity.

## 5 Conclusion

We introduced FineX, a cross-attentive sparse expert framework for fine-grained human action recognition. It models fine-grained evidence as three complementary streams: RGB appearance, pose-heatmap geometry, and skeletal-graph topology, while preserving stream identity. Instead of simple concatenation or late fusion, FineX preserves stream identity, enables pairwise cross-stream reasoning, and streamwise latent sparse expert routing for content-adaptive refinement. Cross-attention chooses what evidence to exchange between streams, and sparse experts determine how each resulting representation is transformed.

Experiments on Gym99, Gym288, and Diving48 show that this structured fusion is consistently efective. On the long-tailed Gym288 benchmark, FineX achieves 94.3% Top-1 accuracy and 76.2% MCA, improving the previous best MCA by +7.6 and the strongest vision-language baseline by +12.2, with the largest gains on rare and mid-frequency classes. This suggests that complementary visual, pose, and graph cues ofset the weaknesses of any single stream under data scarcity. On Gym99 and Diving48, FineX also reaches state-of-theart performance, indicating that the design benefits both balanced fine-grained discrimination and motion-structured recognition.

Importantly, FineX obtains these gains without textual supervision or largescale vision-language pre-training, indicating that, for FHAR, explicitly modeling visual and pose structure can rival scaling semantic supervision. However, FineX has limitations: it needs three backbone forward passes at inference. Future work could reduce computation via shared or lightweight pose representations, improve robustness to noisy keypoints, and test whether language priors add complementary gains for visually ambiguous classes.

## Acknowledgments

This work was supported by UK Research and Innovation (UKRI) - Engineering and Physical Sciences Research Council (EPSRC) under Grant EP/X028631/1 (ATRACT Project).

## References

1. Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., Schmid, C.: Vivit: A video vision transformer. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 6836–6846 (2021)

2. Bertasius, G., Wang, H., Torresani, L.: Is space-time attention all you need for video understanding? In: Icml. vol. 2, p. 4 (2021)

3. Bian, X., Chang, D., Yang, Y., Chen, L., Ma, Z.: Multimodal feature collaboration and fusion for fine-grained action recognition. In: Proceedings of the British Machine Vision Conference (BMVC) (2025)

4. Carreira, J., Zisserman, A.: Quo vadis, action recognition? a new model and the kinetics dataset. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2017)

5. Chang, H., Ren, P., Zhang, H., Xie, L., Chen, H., Yin, E.: Hierarchical-aware orthogonal disentanglement framework for fine-grained skeleton-based action recognition. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 11252–11261 (2025)

6. Duan, H., Wang, J., Chen, K., Lin, D.: Pyskl: Towards good practices for skeleton action recognition. In: Proceedings of the 30th ACM International Conference on Multimedia. pp. 7351–7354 (2022)

7. Duan, H., Zhao, Y., Chen, K., Lin, D., Dai, B.: Revisiting skeleton-based action recognition. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2969–2978 (2022)

8. Fan, H., Feichtenhofer, C., Malik, J.: Multiscale vision transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2021)

9. Feichtenhofer, C.: X3d: Expanding architectures for eficient video recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2020)

10. Feichtenhofer, C., Fan, H., Malik, J., et al.: Slowfast networks for video recognition. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2019)

11. Ghadiyaram, D., Tran, D., Mahajan, D.: Large-scale weakly-supervised pretraining for video action recognition. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12046–12055 (2019)

12. Hassan, I.U., Bessis, N., Behera, A.: TAG-Head: Time-aligned graph head for plugand-play fine-grained action recognition. In: Proceedings of the International Conference on Pattern Recognition (ICPR) (2026), to appear; arXiv:2604.11498

13. Li, K., Wang, Y., Gao, P., Song, G., Liu, Y., Li, H., Qiao, Y.: Uniformer: Unified transformer for eficient spatiotemporal representation learning. In: Proceedings of the International Conference on Learning Representations (ICLR) (2022)

14. Li, X., Yang, W., Wang, K., Wang, T., Zhang, C.: Manet: Motion-aware network for video action recognition. Complex & Intelligent Systems (2025)

15. Li, X., Xu, H.: Meid: mixture-of-experts with internal distillation for long-tailed video recognition. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 37, pp. 1451–1459 (2023)

16. Li, Y., Wu, C.Y., Fan, H., Mangalam, K., Xiong, B., Malik, J., Feichtenhofer, C.: Mvitv2: Improved multiscale vision transformers for classification and detection. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4804–4814 (2022)

17. Li, Y., Li, Y., Vasconcelos, N.: Resound: Towards action recognition without representation bias. In: Proceedings of the European conference on computer vision (ECCV). pp. 513–528 (2018)

18. Lin, J., Gan, C., Han, S.: Tsm: Temporal shift module for eficient video understanding. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2019)

19. Liu, Z., Ning, J., Cao, Y., Wei, Y., Zhang, Z., Lin, S., Hu, H.: Video swin transformer. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3202–3211 (2022)

20. van der Maaten, L., Hinton, G.: Visualizing data using t-SNE. Journal of Machine Learning Research 9, 2579–2605 (2008)

21. Ni, B., Li, J., Wang, S., et al.: Expanding language-image pretrained models for general video recognition. In: Proceedings of the European Conference on Computer Vision (2022)

22. Peng, Y., Li, H., Wu, P., Zhang, Y., Sun, X., Wu, F.: D-fine: Redefine regression task of detrs as fine-grained distribution refinement. In: Proceedings of the International Conference on Learning Representations (ICLR). vol. 2025, pp. 44015–44031 (2025)

23. Perrett, T., Sinha, S., Burghardt, T., Mirmehdi, M., Damen, D.: Use your head: Improving long-tail video recognition. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2415–2425 (2023)

24. Shao, D., Zhao, Y., Dai, B., Lin, D.: Finegym: A hierarchical video dataset for fine-grained action understanding. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2616–2625 (2020)

25. Shazeer, N., Mirhoseini, A., Malinowski, K., Davis, A., Le, Q., Hinton, G., Dean, J.: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In: Proceedings of the International Conference on Learning Representations (2017)

26. Shi, L., Zhang, Y., Cheng, J., Lu, H.: Two-stream adaptive graph convolutional networks for skeleton-based action recognition. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12026–12035 (2019)

27. Sun, B., Ye, X., Yan, T., Wang, Z., Li, H., Wang, Z.: Dsfnet: Discriminative segment focus network for fine-grained video action recognition. ACM Transactions on Multimedia Computing, Communications, and Applications 20(7) (2024)

28. Sun, K., Xiao, B., Liu, D., Wang, J.: Deep high-resolution representation learning for human pose estimation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5693–5703 (2019)

29. Tong, Z., Song, Y., Wang, J., Wang, L.: Videomae: Masked autoencoders are dataeficient learners for self-supervised video pre-training. Advances in neural information processing systems 35, 10078–10093 (2022)

30. Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., Paluri, M.: A closer look at spatiotemporal convolutions for action recognition. In: Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. pp. 6450–6459 (2018)

31. Wang, L., Xiong, Y., Wang, Z., et al.: Temporal segment networks for action recognition in videos. IEEE Transactions on Pattern Analysis and Machine Intelligence 41(11), 2740–2755 (2018)

32. Wang, M., Xing, J., Liu, Y.: Actionclip: A new paradigm for video action recognition. arXiv preprint arXiv:2109.08472 (2021)

33. Xu, Y., Zhang, J., Zhang, Q., Tao, D.: Vitpose: Simple vision transformer baselines for human pose estimation. In: Advances in Neural Information Processing Systems (NeurIPS). vol. 35, pp. 38571–38584 (2022)

34. Yan, S., Xiong, Y., Lin, D.: Spatial temporal graph convolutional networks for skeleton-based action recognition. In: Proceedings of the AAAI conference on artificial intelligence. vol. 32 (2018)

35. Yang, C., Chen, Y., Zhang, L., et al.: Temporal pyramid network for action recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2020)

36. Yang, T., Zhu, Y., Xie, Y., Zhang, A., Chen, C., Li, M.: Aim: Adapting image models for eficient video action recognition. In: Proceedings of the International Conference on Learning Representations (2024)

37. Zhang, C., Gupta, A., Zisserman, A.: Temporal query networks for fine-grained video understanding. In: Proceedings of the ieee/cvf conference on computer vision and pattern recognition. pp. 4486–4496 (2021)

38. Zhang, H., et al.: Pgvt: Pose-guided video transformer for fine-grained action recognition. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (2024)

39. Zhang, H., Leong, M.C., Li, L., Lin, W.: Pevl: Pose-enhanced vision-language model for fine-grained human action recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18857–18867 (2024)

40. Zhou, B., et al.: Temporal relational reasoning in videos. In: Proceedings of the European Conference on Computer Vision (2018)