# Inter-X++: A Comprehensive Benchmark for Multimodal Human-Human Interaction Analysis

Liang Xu, Chengqun Yang, Zili Lin, Xintao Lv, Yichao Yan, Xin Jin, Zhibo Chen, Xiaokang Yang, Fellow, IEEE and Wenjun Zeng, Fellow, IEEE

Abstract—The capability to perceive and synthesize humanhuman interactions (HHIs) is fundamental to developing intelligent digital human systems and understanding humans as social beings. However, existing datasets and modeling approaches are fundamentally constrained by low-fidelity kinematics, the omission of dexterous hand gestures and a severe lack of rich multimodal annotations. Furthermore, fragmented interaction representations and inconsistent evaluation protocols also impede fair and rigorous benchmarking. To systematically address these bottlenecks, we present Inter-X++, a comprehensive and largescale benchmark designed to empower versatile HHI analysis. Captured via a novel hybrid motion capture system, Inter-X++ provides 11,388 high-fidelity interaction sequences and over 8.1M frames, featuring precise whole-body movements and detailed finger articulations. Meanwhile, we enrich the data foundation with multifaceted annotations, including hierarchical fine-grained textual descriptions, interaction categories, causal interaction orders, the relationship and personality of the subjects, as well as vertex-level contact maps and physically regularized constraints. Leveraging these elaborate annotations, we formulate a unified testing ground comprising four categories of downstream tasks that symmetrically span both generative and perceptive paradigms. To eliminate benchmarking ambiguities, we systematically standardize the interaction representations and evaluation protocols. Finally, we go beyond dataset construction to propose OpenHHI, a single and unified HHI representation and modeling framework that jointly optimizes interaction reconstruction and semantic understanding. Extensive experiments reveal that OpenHHI achieves state-of-the-art performance on both downstream generation and perception tasks. This definitively proves that our unified representation successfully bridges interaction understanding and generation simultaneously, solidifying Inter-X++ as a robust foundation for future HHI advancements.

Index Terms—Human-human interaction, Multimodal benchmark, Unified representation

## I. INTRODUCTION

interactions (HHI) is fundamental to the development of intelligent digital human systems, which have extensive applications across visual surveillance, AR/VR [1], interactive gaming, and robotics [2], [3]. Nevertheless, modeling humanhuman dynamics remains profoundly challenging due to the inherent complexity and vast diversity of interaction patterns, coupled with severe mutual and self-occlusions. Although impressive strides have been made in both perception paradigms, i.e., skeleton-based interaction recognition [4], [5], and generative tasks, i.e., action/text-conditioned interaction synthesis [6], [7], existing methods often yield sub-optimal performance. This bottleneck primarily stems from the critical absence of a comprehensive benchmark capable of holistically capturing the multimodal nature of real-world HHIs.

The advancement of human-human interaction analysis is accompanied by the construction of dedicated interaction datasets [7]–[11], as listed in Tab. I. However, existing HHI datasets remain insufficient for comprehensive HHI modeling, exhibiting critical limitations at both the interaction data and multimodal annotation levels [7]–[13]. At the data level, highfidelity kinematic precision, diverse interaction patterns, dexterous hand articulations, and physically constrained modalities are indispensable. Previous attempts [7], [11], [14] estimating human motion via multi-view RGB cameras often suffer from estimation errors and artifacts, leading to inaccurate interaction modeling. Furthermore, small-scale datasets [8]– [10] inevitably restrict both data diversity and the range of interaction categories. The dexterous hand gestures play important roles for HHIs such as “shaking hands”, “grabbing”, “waving”, etc. However, to the best of our knowledge, there is no large-scale dataset providing high-fidelity finger movements for general human-human interactions. Additionally, most prior works overlook physically constrained interactions, which hold direct significance for downstream applications such as human-robot interaction.

Building upon the interaction data foundation, comprehensive multimodal annotations are also crucial yet underexplored in facilitating diverse downstream tasks, which encompass rich textual descriptions, interaction categories, causal interaction order, alongside interpersonal relationships and personalities. Text-driven HHI generation [7], [15] has attracted much attention due to its significant practical applicability. Within this context, interaction categories can serve as the simplest semantic annotations, complemented by hierarchical textual descriptions that describe the interaction dynamics across multiple granularities. For instance, a single interaction sequence can be broadly described as a simple “one person approaches the other and embraces her/him”, but it can also be richly detailed by incorporating part-level semantics. While simple descriptions offer a higher degree of semantic abstraction, fine-grained text annotations enable controllable interaction generation and better spatiotemporal alignment between motion and text modalities. Recent works on human reaction generation [16] have demonstrated that interaction order annotations are of practical significance for delineating causal HHIs. Moreover, the intimacy level and social relationships between individuals, together with their personalities, intuitively affect the interaction patterns, which should be considered for emotionally intelligent HHIs.

![](images/6316d6c428b88fa8c052387643e3d5340c08fb81741e00bb729e854d6d49f495.jpg)  
Fig. 1. Overview of Inter-X++. We present Inter-X++ as a comprehensive benchmark for multimodal human-human interaction analysis structured across three dimensions of interaction data, multimodal annotations and unified representation. For interaction data, Inter-X++ contains high-fidelity MoCap results with vertex-level contact annotations while incorporating physical constraints. For multimodal annotations, we enrich Inter-X++ with hierarchical text descriptions interaction categories, causal interaction order, relationship and personality annotations. We also propose a unified HHI representation named OpenHHI that is universally applicable to both generative and perceptive downstream tasks.

Beyond the aforementioned data and annotation limitations, fragmented interaction representations and inconsistent evaluation protocols are also significantly impeding the broader advancement of HHI analysis. Different works adopt different motion parameterizations and metric spaces, making experimental comparisons less reliable and often obscuring the actual impact of the representation itself. Moreover, the lack of a unified benchmark and standardized evaluation setting prevents a fair assessment of how different HHI models perform across generation and perception tasks.

To systematically address the identified limitations spanning data, annotations, representations, and evaluation protocols, we introduce Inter-X++ as a comprehensive benchmark to empower and advance versatile human-human interaction analysis, as illustrated in Fig. 1. Data acquisition leverages a novel hybrid motion capture system, combining an optical scheme for precise whole-body kinematics with an occlusion-robust inertial solution for dexterous hand gestures. The resulting dataset encompasses 40 daily interaction categories, featuring 11,388 sequences and over 8.1M frames from 89 distinct subjects. On top of the data foundation, we further enrich the multimodal annotations with hierarchical fine-grained text descriptions, semantic interaction categories, causal interaction order, relationship and personality labels, as well as vertexlevel contact and physically regularized interaction annotations, substantially broadening the utility of the benchmark.

With our proposed high-precision HHI data and the versatile annotations, we propose a unified testing ground comprising four categories of downstream tasks spanning generative and perceptive applications: 1) Hierarchical texts enable controllable text-driven human interaction generation [7], [15] and interaction captioning tasks [17], [18], while simultaneously supporting a systematic investigation into the correlation between textual description granularity and the fidelity of generated motions; 2) Interaction categories facilitate actionconditioned human interaction generation [6] together with the human interaction recognition tasks [4], [5]; 3) Interaction order enables the causal human reaction generation [19], [20] and the causal interaction order inference tasks, i.e., detecting the perpetrator in surveillance scenarios; 4) Relationship and personality make the stylized interaction generation [21] and the personality assessment possible.

To ensure rigorous benchmarking, we first systematically evaluate the impact of different interaction representations on experimental outcomes. Based on this analysis, we standardize our metric space by unifying all experiments using the continuous 6D rotation representation, while concurrently aligning the evaluation protocols across all eight downstream tasks. Furthermore, we present a single, unified HHI representation and model, termed OpenHHI, for both interaction understanding and generation tasks. Built upon the shared interaction representation, OpenHHI bridges multiple downstream tasks within a single cohesive modeling framework. Extensive experiments demonstrate the strong effectiveness of the proposed representation and model, yielding substantial gains across diverse tasks and establishing state-of-the-art performance on most evaluation metrics of the Inter-X++ benchmark.

Our contributions can be summarized as follows:

• We collect a large-scale and most comprehensive humanhuman interaction dataset, featuring high-fidelity human body movements, diverse interaction patterns, expressive finger articulations and all rigorously augmented by vertex-level contact annotations and physical constraints.

• We complement Inter-X++ with hierarchical textual descriptions, semantic interaction categories, causal interaction order, relationship and personality, thereby empowering a broad spectrum of downstream applications;

• Building upon the multimodal annotations, we define four categories of distinct downstream tasks, with each explicitly featuring both generative and perceptive variants.

• We present a comprehensive analysis and unification of human-human interaction representations, alongside a standardized evaluation protocol for downstream tasks.

• We propose a unified human-human interaction representation called OpenHHI, which simultaneously facilitates both interaction understanding and generation tasks. Furthermore, it achieves substantial performance improvements across the proposed downstream tasks, establishing state-of-the-art results on multiple benchmarks.

A preliminary version of this work [15] has been published in CVPR 2024 as a conference paper. In this extended version, we introduce substantial improvements in the following key aspects: Firstly, we drastically expand the annotation types and granularity of the existing dataset. We extend the original monolithic fine-grained textual annotations into hierarchical descriptions, enabling the investigation of how varying text granularities influence and control interaction motion generation. Furthermore, we upgrade the causal reasoning annotations to not only capture the interaction order of HHIs but also provide individual-specific textual descriptions, thereby facilitating a decoupled analysis of the two interacting subjects. Two novel modalities of vertex-level contact annotations and physically regularized HHI data are also supplemented to enhance the close-proximity and physical realism of the dataset and to support the development of physically aware interaction modeling. Secondly, to circumvent the lack of standardized benchmarks arising from fragmented representations and evaluation protocols across the literature, we systematically analyze the impact of different interaction representations on experimental outcomes and unify the evaluation protocols. Finally, we propose OpenHHI, a unified HHI representation capable of simultaneously supporting both generative and perceptive tasks. The VQ-VAE-quantized interaction features are processed through a ViT encoder and subsequently channeled into a dual-branch decoder. One branch is dedicated to interaction reconstruction, while the other introduces a novel interaction captioning objective for semantic understanding. The joint optimization of these parallel branches effectively enforces the learning of a cohesive representation, which delivers remarkable performance gains across diverse downstream applications and sets multiple new state-of-the-art results. Collectively, these systematic upgrades in HHI data, annotations, representations, and evaluation protocols establish a robust foundation for the continuous advancement of humanhuman interaction understanding and generation.

## II. RELATED WORK

In this section, we organize the related works into three aspects. We begin with an overview of existing human motion and human-human interaction (HHI) datasets in Sect. II-A. We then systematically review the major HHI tasks with their developments in Sect. II-B. Finally, we summarize the evaluation protocols and benchmarking settings adopted in HHI research in Sect. II-C.

## A. Human Motion and Interaction Datasets

Compared to RGB video streams, 3D human motion representations provide a high-level, efficient, privacy-friendly, and illumination-invariant modality [27]. The evolution of human motion datasets [23], [28]–[32] has fundamentally driven progress in human motion understanding, with early large-scale benchmarks primarily relying on discrete action labels [23], [33], laying the foundation for discriminative motion modeling. To capture fine-grained semantics and facilitate open-vocabulary analysis, subsequent datasets incorporated rich natural language descriptions [30], [31], [34], enabling cross-modal alignment between kinematics and text. Beyond isolated motions, recent efforts emphasize contextaware modeling to address real-world complexities. Specifically, datasets featuring synchronized audio signals [35] have been introduced for rhythmic and speech-driven behavior analysis. Simultaneously, the proliferation of datasets with explicit 3D scene and object geometries [36]–[38] provides crucial physical grounding, which is indispensable for advancing robust human-environment interactions and real-world humancentric tasks.

TABLE I  
DATASET COMPARISONS. A COMPREHENSIVE COMPARISON BETWEEN INTER-X++ AND EXISTING HUMAN-HUMAN INTERACTION DATASETS. SEQUENCES: TOTAL NUMBER OF INTERACTION CLIPS; FRAMES: TOTAL NUMBER OF 3D MOTION FRAMES; TEXTS: THE NUMBER OF TEXTUAL ANNOTATIONS; SCHEME: MOTION DATA ACQUISITION STRATEGY; MODALITY: KINEMATIC REPRESENTATION (“SKEL.” DENOTES SKELETON). ADDITIONALLY, HANDS, ASYN., REL.&PST., PHYSICAL, AND CONTACT INDICATE THE INCLUSION OF ARTICULATED HAND GESTURES, ASYMMETRY ANNOTATIONS, INTERPERSONAL RELATIONSHIPS AND PERSONALITIES, PHYSICAL PLAUSIBILITY, AND EXPLICIT CONTACT ANNOTATIONS, RESPECTIVELY. <sup>∗</sup> DENOTES THE INTEGRATION OF UPGRADED HIERARCHICAL LANGUAGE DESCRIPTIONS.
<table><tr><td>Dataset</td><td>Year</td><td>Sequences</td><td>Frames</td><td>Texts</td><td>Scheme</td><td>Modality</td><td>Hands</td><td>Asyn.</td><td>Rel.&amp; Pst.</td><td>Physical</td><td>Contact</td></tr><tr><td>UMPM [8]</td><td>2011</td><td>36</td><td>400K</td><td>x</td><td>MoCap</td><td>Skel.</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>SBU Kinect [9]</td><td>2012</td><td>300</td><td>7.5K</td><td>x</td><td>RGB+D</td><td>Skel.</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>You2Me [22]</td><td>2020</td><td>42</td><td>77K</td><td>x</td><td>RGB+D</td><td>Skel.</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>NTU120 [23]</td><td>2019</td><td>8,276</td><td>462K</td><td>x</td><td>RGB+D</td><td>Skel.</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Chi3D [10]</td><td>2020</td><td>373</td><td>63K</td><td>x</td><td>MoCap</td><td>SMPL-X</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>ExPI [24]</td><td>2022</td><td>115</td><td>30K</td><td>x</td><td>mRGB</td><td>Skel.</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Hi4D [11]</td><td>2023</td><td>100</td><td>11K</td><td>x</td><td>mRGB</td><td>SMPL</td><td>x</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>InterHuman [7]</td><td>2023</td><td>6,022</td><td>1.7M</td><td>16,756</td><td>mRGB</td><td>SMPL</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>ReMoCap [25]</td><td>2024</td><td>87</td><td>275.7K</td><td>x</td><td>mRGB</td><td>Skel.</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Inter-X [15]</td><td>2024</td><td>11,388</td><td>8.1M</td><td>34,164</td><td>MoCap</td><td>SMPL-X</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td></tr><tr><td>InterDance [26]</td><td>2025</td><td></td><td></td><td>x</td><td>MoCap</td><td>SMPL-X</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Embody 3D [14]</td><td>2025</td><td></td><td></td><td>√</td><td>mRGB</td><td>SMPL-X</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Inter-X++</td><td>2026</td><td>11,388</td><td>8.1M</td><td>102,492*</td><td>MoCap</td><td>SMPL-X</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Beyond single-person motion capture, a variety of humanhuman interaction datasets have been introduced [7], [9]– [11], [14], [25], [26], encompassing diverse scales, modalities, and functionalities, as shown in Tab. I. Notably, the recent InterHuman dataset [7] significantly advances this domain by providing large-scale HHI sequences paired with semantic text. While existing benchmarks typically compromise on kinematic precision or holistic detail, our preliminary work introduced the Inter-X dataset [15] to address these limitations through high-fidelity motions, fine-grained text, and synchronized hand articulation. In this extension, we significantly elevate Inter-X by incorporating a richer spectrum of modalities. Specifically, we augment the dataset with hierarchical textual descriptions, detailed contact annotations, and enhanced physical realism, thereby establishing a substantially more comprehensive and physically grounded foundation for HHI modeling.

## B. Human-Human Interaction Tasks

Human-human interaction (HHI) constitutes a fundamental component of social behavior understanding and plays a critical role in a wide range of human-centric applications, including virtual and augmented reality [1] and human-robot interaction [2], [3], [39].

1) HHI Generation: Human motion generation aims to synthesize kinematic sequences that are both physically plausible and behaviorally diverse, conditioned on various control modalities. Recent advancements have predominantly focused on single-person generation driven by discrete action labels [6], [40]–[42], free-form textual descriptions [30], [34], [43]–[45], and acoustic signals [46]. Beyond the synthesis of isolated individuals, recent efforts have further expanded toward complex multi-person interactions [6], [7], [47], [48]. in2In [49] and ComMDM [50] leverage pre-trained motion priors derived from single-person motion generation to facilitate two-person interaction synthesis. TIMotion [51] jointly accounts for temporal modeling and interaction mixing to achieve efficient HHI generation. InterMask [52] employs a VQ-VAE to discretize individual motions into 2D tokens, modeling HHIs through a generative masked modeling framework. Human-X [1] systematically addresses real-time responsiveness, physical feasibility, and safety constraints in human interactions to empower immersive human-machine collaboration. Anchored on proximal interactive poses for versatile interaction animation, Ponimator [53] successfully transfers interaction knowledge from motion capture data to open-world scenarios. Furthermore, Interact2Ar [54] utilizes an autoregressive diffusion model to synthesize full-body HHIs, naturally encompassing fine-grained hand articulations.

2) Human Reaction Generation: Emerging works have begun to explore human reaction generation, aiming to synthesize plausible responsive behaviors conditioned on antecedent human motions [25], or more generalized inputs such as video sequences [55], [56] and interlocutor speech [57]. Specifically, InterFormer [19] introduces an interaction transformer architecture equipped with both temporal and spatial attention mechanisms for human reaction generation. HIG [20] proposes an annotation-free approach to automatically learn the underlying dynamics between actors and receivers. To ensure physical realism, PhysReaction [58] focuses on generating physically plausible human reactions in real time, while ARFlow [59] directly models the continuous actionto-reaction mapping via flow matching [60]. Notably, Re-GenNet [16] constructs the first multi-setting human-reaction generation framework, incorporating explicit annotations for interaction orders. Extending beyond pure human-human scenarios, iHuman [61] establishes an online reaction generation system for human-humanoid interactions. Addressing realtime interaction constraints, Ready-to-React [62] formulates an online policy that generates reactive motions for each individual based on past observations, and ReMoGen [63] synthesizes real-time, high-quality, and coherent reactions from heterogeneous interaction data. Furthermore, HERO [55] extracts reaction generation priors directly from videos, encompassing human-human, human-animal, and human-scene interactions. Similarly, EgoReAct [56] processes streaming egocentric videos to synthesize spatially grounded and realistic human reactive motions. ReactMotion [57] introduces a novel task of generating listener body reactions driven by speaker utterances. By providing explicit temporal-order annotations and equipping each participant with participant-specific textual descriptions, Inter-X++ further advances the development of human reaction generation.

![](images/cca27eed9193321a99fe08ddd05a481479b54ef2a33cbc2fcbcef97c09d0efbb.jpg)  
Fig. 2. The Inter-X++ whole-body hybrid MoCap system. (a). Spatial integration and alignment of the optical MoCap suit and the IMU-based gloves is achieved via a rigid triangular bracket of reflective markers; (b). Detailed view of the reflective marker configuration; (c). Temporal synchronization of the human body and hand kinematics within the unified whole-body motion capture system.

3) HHI Recognition: Skeleton-based human action recognition has been extensively investigated [64]–[68]. As a challenging sub-field, human interaction recognition [4], [5], [69]– [72] further demands precise modeling of semantic and spatiotemporal correlations among individuals. Beyond discrete action classification, human kinematics inherently encode rich biometric and psychological cues [73]. For instance, gait recognition [74] seeks to identify individuals through distinct motion patterns, while other studies [75], [76] leverage body movements as predictive indicators of personality traits. Featuring large-scale action-motion and text-motion alignments, our Inter-X++ dataset directly drives advancements in human action recognition. More importantly, it establishes a novel foundation for quantitatively assessing interpersonal relationships and individual personalities from interactive motions.

4) HHI Reconstruction: Another research direction focuses on recovering high-precision HHIs from images or videos [77]. Hi4D [11] provides a robust benchmark tailored for close human interactions, featuring rich, interaction-centric annotations. CloseMoCap [78] and AvatarPose [79] attempt to reconstruct close-contact interaction sequences from multiple calibrated cameras. Addressing the more challenging monocular setting, MultiPhys [80] successfully recovers robust and physically aware multi-person interactive motions. To mitigate the severe occlusions inherent in interacting scenarios, Dual-Human [81] leverages semantic reaction priors to assist in inferring occluded regions, thereby significantly enhancing reconstruction quality. Furthermore, CloseApp [82] introduces a dual-branch optimization framework that exploits human appearance, proxemics, and physical constraints to effectively resolve visual ambiguities.

## C. HHI Representation and Benchmarking

Human motion and HHI analysis remain hindered by fragmented representations and the lack of a unified evaluation protocol. Early works such as HumanML3D [34] adopted carefully designed yet redundant kinematic representations, and HumanTOMATO [44] further extended this formulation to whole-body motion. More recent frameworks, including STMC [83] and MotionStreamer [84], have introduced substantially different representational formats. A similar issue persists in HHI modeling: InterGen [7] follows the HumanML3D formulation and defines both canonical and noncanonical representations, whereas Inter-X [15] adopts a direct 6D rotation parameterization. This inconsistency undermines fair comparison and benchmark rigor. To address it, we analyze how motion representations affect generation performance and establish a unified protocol based on a standardized representation for fair and consistent benchmarking.

## III. THE INTER-X++ DATASET

We introduce Inter-X++, a large-scale dataset designed for comprehensive multimodal human-human interaction analysis. In this section, we first introduce the data capturing system with postprocessing to curate human-human interaction data in Sect. III-A. We then detail the dataset composition, including motion data, action category labels, hierarchical textual descriptions, interaction order, contact annotations, and relationship and personality attributes of the subjects in Sect. III-B. These annotations significantly enrich the modality spectrum of human-human interaction data and support a more detailed understanding and modeling of human-human interactions.

## A. Dataset Capturing System

For interaction data acquisition, we impose two key requirements on the capturing setup: ensuring high-fidelity motion capture and enabling the capture of dexterous finger movements, which are crucial for a wide range of daily interaction scenarios such as handshaking and rock-paper-scissors. Many existing human motion datasets rely on multi-view RGB-based technologies [7], [23] to extract human motions from RGB videos. Although such pipelines preserve the natural image information, these datasets suffer from severe occlusions and penetrations (especially for close human-human interactions), and the subtle finger movements are also difficult to obtain precisely. Other approaches employ inertial MoCap suits to record human motions [85]. Despite their convenience and lack of environmental constraints, our experiments show that error accumulation can introduce substantial inaccuracies in multi-person interaction capture, particularly in the relative displacement between individuals. Thus, we choose the optical motion capture system for its high precision to obtain the body movements. Additionally, we further employ inertial gloves to capture the finger gestures, which are inherently robust to hand flipping, rapid hand motions and occlusions. The overview of the dataset capturing system is illustrated in Fig. 2.

We deploy the OptiTrack MoCap system [86] with 20 PrimeX 22 infrared cameras recording at the resolution of 2048×1088 and 120 FPS. The dimensions of our MoCap venue are 8.5 meters in length, 5.4 meters in width, and 3.3 meters in height, which are sufficient to cover most daily human-human interactions. The optical motion capture scheme ensures an error of ±0.15 mm, which is significantly lower than that of the multi-view RGB cameras or inertial MoCap suits. More importantly, the system demonstrates strong robustness in capturing multi-person interactions, especially for poses involving close physical contact, and remains reliable even when some reflective markers are occluded. To capture the dexterous hand gestures and achieve robustness to hand rotations, rapid movements, and occlusions, we adopt the inertial solution of the commercial Noitom Perception Neuron Studio (PNS) gloves [87]. The subtle finger movements can be captured in real-time, disregarding the self-occlusion and occlusion with the other person during the interactions.

For each capture session, two volunteers are randomly paired to wear the OptiTrack MoCap suits with 41 reflective markers and the inertial gloves as depicted in Fig. 2(a),(b). Both of them are carefully calibrated for the hybrid opticalinertial setup, respectively, before they perform the interactions. We provide timecodes for the OptiTrack MoCap system and the PNS gloves using Tentacle Sync so that the body and hands can be temporally synchronized. For each recording batch, we arrange five action categories with five repetitions to ensure variability, which improves efficiency and ensures the continuity of the volunteers’ actions. Taking handshaking as an example, the two participants are encouraged to perform different handshaking patterns, such as single-hand and double-hand handshakes, varying the interpersonal distance, or interacting while standing or sitting on chairs, in order to increase the diversity of the dataset. The volunteers pause for several seconds between two interaction snippets to improve capture quality and ease the subsequent temporal segmentation. We also recalibrate the PNS inertial gloves frequently between recording batches to mitigate the error accumulation.

## B. Dataset Taxonomy

This subsection first presents the processing pipeline for the captured human-human interaction data and then introduces the procedures for further enriching the data with multimodal annotations, yielding 11,388 pairs of SMPL-X [88] interaction sequences, 40 semantic action categories with diverse action/reaction patterns, 102,492 hierarchical text descriptions, interaction order labels, contact annotations and the relationship for 59 groups and personality for 89 volunteers. Representative characteristics of the Inter-X++ dataset are illustrated in Fig. 3.

![](images/7ece61e0e6c90559f1aae01ac8d4107cc8e6b7c51e5e8a518d9cc89ccc94ac43.jpg)  
Fig. 3. Motion characteristics of Inter-X++. Our proposed Inter-X++ dataset is distinguished by the 1) Precise contact for close HHIs; 2) Dexterous hand gestures; 3) Diverse action and reaction patterns.

1) Interaction Data: To obtain the full-body human motion data, the crux of the postprocessing is the alignment between the body poses from the OptiTrack MoCap system and the finger gestures from the inertial gloves. Temporally, we retrieve the overlapping intervals between the body pose and the hand pose sequences based on the millisecond-level timestamps. Spatially, these two modalities are naturally integrated through the shared wrist rotation obtained from the triangular locating bracket attached to the hand dorsum. Following the temporal and spatial alignment, we derive the whole-body HHI skeleton sequences. Since the dataset is collected in batches, we further perform temporal segmentation to ensure that each sequence corresponds to an atomic interaction. We recruit annotators to manually identify the start and end frames of each atomic interaction. The segmentation results are carefully collected and verified, and the long batch-captured sequences are then trimmed into atomic segments accordingly.

We adopt the SMPL-X [88] parametric model due to its high expressiveness in capturing intricate human body poses and fine-grained hand articulations, as well as its broad applicability to downstream tasks. Formally, a sequence of N frames is parameterized by body pose parameters $\theta \in \mathbb { R } ^ { N \times 5 5 \times 3 }$ , shape parameters $\beta \in \dot { \mathbb { R } } ^ { N \times \dot { 1 0 } }$ and translation parameters $t \in \mathbb { R } ^ { N \times 3 }$ The shape parameters $\beta$ are initialized using the height and weight of each volunteer, following the methodology in [89]. Subsequently, a carefully tuned optimization algorithm is employed to fit the SMPL-X parameters to the captured MoCap keypoints by minimizing the following objective function:

$$
E ( \theta , t ) = \lambda _ { 1 } \frac { 1 } { N } \sum _ { j \in \mathcal { I } } \lambda _ { p } \vert \vert \pmb { J } _ { j } ( \mathbb { M } ( \theta , t ) ) - \pmb { g } _ { j } \vert \vert _ { 2 } ^ { 2 } + \lambda _ { 2 } \vert \vert \theta \vert \vert _ { 2 } ^ { 2 } ,\tag{1}
$$

where $\mathcal { I }$ denotes the joint set, M represents the SMPL-X parametric model, $J _ { j }$ is the regressor function for joint j, g indicates the skeleton captured by the MoCap system. The terms $\lambda _ { 1 } , \lambda _ { 2 }$ and $\lambda _ { p }$ are different optimization weights, with $\lambda _ { p }$ specifically designed to assign varying degrees of importance to different body parts. Please refer to the supplementary materials for comprehensive implementation details.

2) Interaction Categories: We curate the interaction categories by drawing upon the established HHI datasets [7], [10], [23] and leveraging large language models [90]. Ultimately, we figure out 40 daily interaction categories, representing, to the best of our knowledge, one of the most comprehensive behavioral taxonomies in the HHI domain. To ensure data quality, volunteers are instructed to execute the interaction both naturally and diversely. Data collection is conducted in a controlled, enclosed MoCap studio environment. Volunteers are instructed to remain relaxed and perform the interactions according to their habitual behavioral patterns. To ensure high-fidelity data capture and prevent fatigue, a brief rest period of several seconds was allocated between consecutive trials. Meanwhile, the diversity is manifested across three dimensions: 1) action variations, e.g., raising the left, right, or both hands for “raising hands”; 2) reaction variations, e.g., resisting, stepping back, or falling when “being pushed”; and 3) heterogeneous initial body states, e.g., standing, sitting, crouching, or lying down. Furthermore, each interaction is performed five times to maximize intra-class variability.

3) Hierarchical Text Descriptions: Natural language exhibits inherent flexibility, allowing for varying levels of abstraction and granularity. For the same interaction sequence, the descriptions may vary as “Handshake with each other” or “Shake hands with their right hands”, where the latter specifies particular body parts. Highly complex and finegrained textual descriptions yield more precise motion representations and facilitate finer control over interaction synthesis; conversely, simpler text prompts offer a higher degree of semantic abstraction. Inter-X [15] originally provided highly detailed textual annotations with extensive human body partlevel descriptions. Specifically, we develop an annotation tool based on [91], allowing annotators to freely manipulate the 3D viewpoint, including scaling and 360-degree rotation to observe interaction details from any perspective. For every interaction sequence, three independent annotators are tasked with providing fine-grained descriptions encompassing: 1) coarse full-body movements, 2) finger articulations, and 3) relative spatial orientations between the participants. The raw textual data was subsequently refined using GPT-3.5 [90] to rectify typographical errors and validated through manual spotchecking. Consequently, the resulting descriptions average ∼35 words, substantially exceeding prior benchmarks and underscoring the exceptional semantic density of our dataset.

However, we empirically find that the state-of-the-art interaction generation models still struggle to achieve consistent alignment between the textual inputs and the synthetic interactions. Wu et al. [92] points out that overly long, detailed descriptions may distract the generative model from capturing the global semantics of the motions. To achieve a higher degree of semantic abstraction, we uniformly render 12 frames per interaction sequence and provide these visual inputs, coupled with the three corresponding raw detailed descriptions, to GPT-5.1 [93] for text simplification, thereby synthesizing a concise version of the action narrative. We further define the action category as the most simplified textual representation. Ultimately, providing texts with hierarchical levels of abstraction empowers diverse conditional motion generation tasks and their corresponding evaluations.

4) Causal Interaction Order: Exploring the causal relationships inherent in social HHI behaviors, where the action of one individual precipitates a reaction from another, is crucial for advancing the understanding of HHIs [9]. To model this, volunteers are asked to explicitly annotate the causal order of the active and reactive participants within every atomic interaction clip. For highly symmetric interactions, such as handshaking, the individual who extends their hand first is typically designated as the actor.

Furthermore, beyond merely determining the causal order for each interaction clip, we aim to acquire more fine-grained and role-specific action descriptions for both the actor and the reactor, respectively. To achieve this, we uniformly sample 12 rendered frames from each interaction sequence alongside the initial global textual descriptions, and then feed them into GPT-5.1 [93], prompting the model to generate distinct action descriptions for each participant. By providing distinct action narratives for each individual, we effectively augment the holistic HHI descriptions, enabling more fine-grained, textdriven control over individual participants.

5) Contact Annotations: The annotation of contact regions is crucial for a deeper understanding of close human-human interactions. However, acquiring precise contact areas remains highly challenging. Leveraging our high-precision OptiTrack optical MoCap system, we directly employ a simple yet efficient distance-based method to compute these regions. Specifically, we define the contact area as locations where the inter-subject distance is less than 0.05 m, thereby providing fine-grained, vertex-level contact annotations.

6) Relationship & Personality: Exploring the correspondence between HHI and their relationship & personality is a niche [75], [76] where the essence lies in the disentanglement of the personality factors from body motions. We employ the standard Big-Five Personality Model [94], utilizing the NEO Five-Factor Inventory [95] to evaluate their personalities of openness, conscientiousness, extraversion, agreeableness, and neuroticism. Additionally, participants assessed their mutual familiarity on a four-level scale and identified their social rela tionships into five types: strangers, friends, romantic partners, schoolmates, or family.

7) Physical Plausibility: To enhance the physical plausibility of Inter-X++, we input the kinematics-based motions into a physics simulator and utilize the imitation policy PHC [96] trained with reinforcement learning to drive humanoid agents to imitate these motions, thus correcting the motions to conform to physical constraints. However, directly using the original PHC trained only on AMASS results in insufficient success rates and motion jittering during imitation. Therefore, we fine-tuned PHC on Inter-X++. Similar to MultiPhys [80], the policy is trained in a single-agent environment during finetuning, which helps correct physical artifacts such as foot sliding. During the evaluation phase, we correct unreasonable interactions through having different agents sharing the same simulation environment simultaneously, allowing the physics engine to directly apply physical constraints between them, such as preventing interpenetration, thus solving physical artifacts in multi-person interactions. Finally, the corrected motions are converted back to the SMPL-X to achieve more accurate and physically plausible multi-person motions.

## IV. TASK TAXONOMY

The high-fidelity HHI sequences within our dataset, characterized by complex hand articulations, introduce unprecedented challenges to current benchmarks. Leveraging our comprehensive and versatile annotations, we further formulate several novel downstream tasks oriented toward real-world applications. Formally, a given HHI sequence is denoted as $\pmb { m } = \langle \pmb { x } , \pmb { y } \rangle$ , paired with its corresponding annotations: the discrete action category $\mathbf { \xi } _ { l _ { a } , \ l }$ textual description $\mathbf { \xi } _ { l _ { t } , \mathrm { ~ ~ } }$ causal interaction order ${ l } _ { c } ,$ interpersonal relationship $\displaystyle { l _ { r } , }$ , and individual personalities $l _ { p } = \langle l _ { p _ { x } } , l _ { p _ { y } } \rangle$

## A. Text-Related Tasks

a) Text-conditioned HHI generation: While text-driven single-person motion generation has been extensively explored across various benchmarks [30], [34], extending this capability to multi-person scenarios remains challenging. Equipped with hierarchical textual annotations, our dataset unlocks novel paradigms for controllable HHI generation. Concurrently, it introduces unprecedented challenges, particularly in synthesizing dexterous hand articulations and precisely aligning part-level textual descriptions with interactive kinematics. We also systematically evaluate the generation performance by conducting comparative experiments driven by hierarchical textual descriptions of varying granularity. Formally, this task is defined as learning a generative mapping function $F _ { t 2 m } \colon$

$$
F _ { t 2 m } ( l _ { t } ) \mapsto m .\tag{2}
$$

b) HHI captioning: Beyond conventional discrete action recognition, HHI captioning has emerged as a novel task [17], [18] that aims to synthesize comprehensive natural language descriptions directly from HHI sequences. This paradigm fundamentally reinforces the cross-modal alignment between complex kinematics and text, enabling the automated generation of diverse and semantically coherent captions. Formally, this task can be formulated as:

$$
F _ { m 2 t } ( { \pmb m } ) \mapsto l _ { t } .\tag{3}
$$

## B. Action-Related Tasks

a) Action-conditioned HHI generation: Conditioned on a discrete action label, the generative mapping $F _ { a 2 m } ( \cdot )$ aims to synthesize diverse and physically plausible HHI sequences [6], [41], [42]. Empowered by our Inter-X++ dataset, this task is substantially elevated to model high-fidelity interactive behaviors that encompass dexterous finger kinematics. Formally, it is defined as:

$$
F _ { a 2 m } ( l _ { a } ) \mapsto m .\tag{4}
$$

b) HHI recognition: Skeleton-based HHI recognition is essential for downstream applications such as visual surveillance [4], [5]. We posit that the integration of fine-grained hand articulations will significantly enhance the discriminative capacity of existing recognition frameworks. Formally, this task is defined as:

$$
F _ { m 2 a } ( m ) \mapsto l _ { a } .\tag{5}
$$

## C. Interaction-Order-Related Tasks

a) Human reaction generation: Despite its extensive applicability in immersive AR/VR and interactive gaming, human reaction synthesis [19], [20] remains a comparatively underexplored domain. The explicit annotation of actor-reactor causal orders provides a critical foundation for investigating role asymmetry and precisely modeling responsive dynamics in HHIs. Formally, this process is defined as:

$$
F _ { c 2 m } ( l _ { c } , l _ { a } , \pmb { x } ) \mapsto \pmb { y } .\tag{6}
$$

b) Causal interaction order inference: The causal interaction order inference mapping, denoted as $F _ { m 2 c } ( \cdot )$ , aims to explicitly distinguish the active instigator (actor) from the passive responder (reactor) within a given interaction sequence. This capability significantly benefits downstream applications such as intelligent surveillance and automated sports analytics:

$$
F _ { m 2 c } ( { \pmb m } ) \mapsto l _ { c } .\tag{7}
$$

## D. Relationship- and Personality-Related Tasks

a) Stylized HHI generation: Interpersonal relationships and individual personality traits offer powerful, high-level stylization priors for customized interaction synthesis. The substantial diversity of participants in our dataset, coupled with extended motion sequences for each subject, robustly supports the learning of these stylistic nuances. Formally, this generation task is formulated as:

$$
F _ { s 2 m } ( l _ { r } , l _ { p } , l _ { a } ) \mapsto m .\tag{8}
$$

b) HHI-based personality assessment: While prior studies [75], [76] have established body movements as viable predictors of individual personality, we leverage our dataset to propose the novel task of joint personality and relationship assessment. This quantitative evaluation of behavioral kinematics holds substantial potential for disciplines such as education, medical diagnostics, and sports analytics. Specifically, the assessment mapping is defined as:

$$
F _ { m 2 s } ( { \pmb m } ) \mapsto \{ l _ { r } , l _ { p } \} .\tag{9}
$$

## V. OPENHHI: UNIFIED REPRESENTATION AND MODEL

To establish a more effective interaction representation and a rigorous evaluation benchmark, we first conduct a systematic study of widely used interaction representations in Sect. V-A. Then, we introduce a unified interaction encoder which can serve both perceptive and generative tasks in Sect. V-B.

## A. Human-Human Interaction Representation

The heterogeneity of interaction representations across existing HHI methods confounds cross-method comparisons, making it difficult to disentangle improvements attributable to model design from those arising from the representation itself. To support systematic studies and standardized evaluation of HHI models, we provide the same interaction motions in five representative forms: axis-angle, quaternion, rotation matrix, continuous 6D rotation [97], and the non-canonical representation adopted by InterGen [7]. We will analyze these representations to identify the best-performing choice, which is then used consistently throughout all subsequent experiments to ensure fair and uniform comparisons.

1) Rotation representations: We adopt the original SMPL-X motions and represent the joint rotations using axis-angle, quaternion, rotation matrix, or continuous 6D rotation [97]. In addition to the rotation features, each representation includes the global root translation t and its acceleration t<sup>˙</sup>. Taking the 6D rotation representation as an example, the motion feature is formulated as

$$
\begin{array} { r } { \pmb { x } ^ { \mathrm { 6 D } } = [ \pmb { r } ^ { \mathrm { 6 D } } , \pmb { t } , \dot { \pmb { t } } ] , } \end{array}\tag{10}
$$

where $r ^ { \mathrm { 6 D } }$ contains the 6D rotations of all SMPL-X joints. The four representations therefore share the same global trajectory information and differ only in their rotation parameterization.

2) Non-canonical representation: Following InterGen [7], we consider the non-canonical representation for HHI as:

$$
\pmb { x } = [ j _ { g } ^ { p } , j _ { g } ^ { v } , j ^ { r } , c ^ { f } ] ,\tag{11}
$$

where $j _ { g } ^ { p }$ and $j _ { g } ^ { v }$ denote global joint positions and velocities, $j ^ { r }$ denotes local joint rotations in the 6D rotational format, and $c ^ { f }$ denotes binary foot-ground contact states.

## B. Unified Interaction Encoder

Generative and perceptive HHI tasks place complementary demands on interaction features. Generation requires a representation that retains fine-grained body poses, temporal dynamics, and interpersonal contact geometries so that the original interaction can be faithfully recovered, whereas understanding favors compact features organized by highlevel interaction semantics. Training separate encoders for the two regimes duplicates computation and prevents the two forms of supervision from benefiting each other. We therefore introduce a unified HHI representation that learns a single representation for both HHI generation and understanding. Its core design consists of an interaction VQ-VAE, a shared ViT encoder, and two complementary training branches for interaction reconstruction and semantic understanding.

1) Unified interaction tokenizer: Given a paired interaction m in the 6D rotational representation, we first use a VQ-VAE to compress the high-dimensional motion sequence into a compact motion-aware latent space:

$$
z _ { q } = \mathcal { Q } ( \mathcal { E } _ { \mathrm { v q } } ( { m } ) ) , \qquad h = \mathcal { E } _ { \mathrm { v i t } } ( z _ { q } ) ,\tag{12}
$$

where ${ \mathcal E } _ { \mathrm { v q } }$ and Q denote the VQ-VAE encoder and codebook quantizer, respectively, and ${ \mathcal E } _ { \mathrm { v i t } }$ is the shared interaction encoder. The output h serves as the unified HHI representation for both perceptive and generative tasks. During the training phase, we employ two branches for simultaneous supervision, comprising a reconstruction branch and an understanding branch. These two branches are entirely separate, and we will introduce them in detail below.

2) Interaction reconstruction branch: The reconstruction branch encourages h to retain the detailed kinematics required by generation. A ViT decoder $\mathcal { D } _ { \mathrm { r e c } }$ maps the unified representation back to the quantized latent space, and the VQ-VAE decoder $\mathcal { D } _ { \mathrm { v q } }$ reconstructs the paired interaction:

$$
\begin{array} { r } { \widehat { z } _ { q } = \mathcal { D } _ { \mathrm { r e c } } ( h ) , \widehat { m } = \mathcal { D } _ { \mathrm { v q } } ( \widehat { z } _ { q } ) . } \end{array}\tag{13}
$$

Following the standard VQ-VAE reconstruction objective, we supervise this branch solely in the motion space:

$$
{ \mathcal { L } } _ { \mathrm { r e c } } = \ell _ { 1 } ( m , { \widehat { \pmb { m } } } ) ,\tag{14}
$$

where $\ell _ { 1 }$ denotes the element-wise reconstruction loss between the input and reconstructed interaction motions.

3) Interaction understanding branch: Reconstruction alone mainly focuses on low-level kinematic fidelity and provides limited supervision for learning the semantics of an interaction. We therefore introduce an interaction captioning branch that uses the fine-grained textual descriptions as semantic supervision. Specifically, an autoregressive text decoder takes the unified interaction representation h as its condition and predicts the corresponding description token by token. The understanding objective is defined as the captioning loss:

$$
\mathcal { L } _ { \mathrm { u n d } } = - \sum _ { k = 1 } ^ { K } \log p ( w _ { k } \mid w _ { < k } , \boldsymbol { h } ) ,\tag{15}
$$

where $w _ { k }$ is the k-th caption token and $w _ { < k }$ denotes the preceding tokens. By requiring h to support interaction description generation, this branch encourages the unified representation to capture high-level action semantics in addition to motion details.

4) Joint optimization and downstream transfer: We jointly optimize the shared interaction encoder and the two complementary decoders using the overall objective

$$
\mathcal { L } _ { \mathrm { a l l } } = \omega _ { \mathrm { r e c } } \mathcal { L } _ { \mathrm { r e c } } + \omega _ { \mathrm { u n d } } \mathcal { L } _ { \mathrm { u n d } } ,\tag{16}
$$

where $\omega _ { \mathrm { r e c } }$ and $\omega _ { \mathrm { u n d } }$ balance the two objectives. During joint optimization, both losses are back-propagated through the shared interaction encoder, whereas each decoder is updated by its corresponding objective. In this manner, reconstruction prevents the representation from discarding fine-grained motion information, while caption generation encourages it to encode the action semantics conveyed by the textual descriptions. The resulting h therefore provides a common representation that supports both interaction generation and understanding.

Once jointly pretrained, OpenHHI serves as a shared interaction backbone for all downstream perceptive and generative tasks. The unified encoder maps interaction motions into a unified token space, eliminating the need to learn separate motion representations for different task families. For generative tasks, conditional models operate in this shared space and recover paired motions through the pretrained reconstruction and VQ-VAE decoders. For perceptive tasks, the same interaction tokens are supplied to lightweight classification, regression, or language heads. Only these newly introduced downstream modules need to be optimized. By sharing a single encoder and representation space across generative and perceptive tasks, OpenHHI transfers both fine-grained kinematic information and high-level interaction semantics, leading to consistent performance improvements across the downstream tasks.

## VI. EXPERIMENTS

In this section, we conduct extensive experiments on four key components of Inter-X++: 1) Human-Human Interaction Representation, where we systematically compare and unify different interaction parameterizations; 2) Human-Human Interaction Imitation, where we evaluate physics-based motion regularization on Inter-X++; 3) Unified Interaction Model, where we benchmark OpenHHI across both generative and perceptive downstream tasks; and 4) Hierarchical Texts, where we investigate how textual descriptions at different levels of granularity affect interaction generation quality and motionlanguage alignment. We first describe the common experimental setup and evaluation protocol, followed by detailed quantitative and qualitative analyses of each component.

TABLE II  
EXPERIMENTAL COMPARISON OF THE HUMAN-HUMAN INTERACTION REPRESENTATIONS ON INTER-X++. WE EVALUATE FIVE INTERACTION REPRESENTATIONS UNDER A UNIFIED EXPERIMENTAL PROTOCOL. ± INDICATES 95% CONFIDENCE INTERVAL AND → MEANS THE CLOSER THE BETTER. BOLD INDICATES BEST RESULTS AND UNDERLINE INDICATES SECOND BEST RESULTS.
<table><tr><td rowspan="2">Method</td><td colspan="3">R Precision↑</td><td rowspan="2">FID ↓</td><td rowspan="2">MM Dist↓</td><td rowspan="2">Diversity→</td><td rowspan="2">MModality ↑</td></tr><tr><td>Top 1</td><td> $\mathrm { T o p } ~ 2 $ </td><td> $\mathrm { T o p } \ 3$ </td></tr><tr><td>Real</td><td> $0 . 4 4 8 ^ { \pm 0 . 0 0 3 }$ </td><td> $0 . 6 3 9 ^ { \pm 0 . 0 0 3 }$ </td><td> $0 . 7 4 3 ^ { \pm 0 . 0 0 3 }$ </td><td> $0 . 0 0 1 ^ { \pm 0 . 0 0 0 1 }$ </td><td> $3 . 6 5 2 ^ { \pm 0 . 0 0 9 }$ </td><td> $9 . 6 6 2 ^ { \pm 0 . 0 7 8 }$ </td><td></td></tr><tr><td>Axis-angle</td><td> $0 . 3 0 2 ^ { \pm 0 . 0 0 4 }$ </td><td> $0 . 4 6 4 ^ { \pm 0 . 0 0 4 }$ </td><td> $0 . 5 7 0 ^ { \pm 0 . 0 0 4 }$ </td><td> $3 . 7 0 2 ^ { \pm 0 . 0 8 5 5 }$ </td><td> $4 . 8 0 7 ^ { \pm 0 . 0 1 6 0 }$ </td><td> $8 . 4 5 2 ^ { \pm 0 . 0 5 7 }$ </td><td> $2 . 5 5 9 ^ { \pm 0 . 0 7 8 }$ </td></tr><tr><td>Quaternion</td><td> $0 . 3 4 9 ^ { \pm 0 . 0 0 5 }$ </td><td> $0 . 5 3 3 ^ { \pm 0 . 0 0 5 }$ </td><td> $0 . 6 5 2 ^ { \pm 0 . 0 0 5 }$ </td><td> $\mathbf { 0 . 7 4 8 ^ { \pm 0 . 0 3 2 3 } }$ </td><td> $4 . 2 9 3 ^ { \pm 0 . 0 1 3 1 }$ </td><td> $\mathbf { 9 . 7 9 7 ^ { \pm 0 . 0 6 4 } }$ </td><td> $\mathbf { 2 . 8 9 1 ^ { \pm 0 . 0 9 1 } }$ </td></tr><tr><td>Rotation matrix</td><td> $0 . 1 0 5 ^ { \pm 0 . 0 0 1 }$ </td><td>0.193±0.003</td><td> $\overline { { 0 . 2 6 7 ^ { \pm 0 . 0 0 2 } } }$ </td><td> $9 1 . 0 7 4 ^ { \pm 0 . 1 4 5 1 }$ </td><td> $\overline { { 1 0 . 1 9 0 ^ { \pm 0 . 0 1 2 0 } } }$ </td><td> $3 . 3 8 5 ^ { \pm 0 . 0 5 3 }$ </td><td> $1 . 3 7 0 ^ { \pm 0 . 0 3 7 }$ </td></tr><tr><td>6D rotation</td><td> $\mathbf { 0 . 3 8 9 ^ { \pm 0 . 0 0 4 } }$ </td><td> $\mathbf { 0 . 5 7 7 ^ { \pm 0 . 0 0 5 } }$ </td><td> $\mathbf { 0 . 6 8 9 ^ { \pm 0 . 0 0 4 } }$ </td><td> $1 . 1 9 5 ^ { \pm 0 . 0 3 2 0 }$ </td><td> $\mathbf { 3 . 9 6 3 ^ { \pm 0 . 0 1 7 2 } }$ </td><td> $9 . 9 0 6 ^ { \pm 0 . 0 7 5 }$ </td><td> $2 . 6 9 5 ^ { \pm 0 . 0 9 0 }$ </td></tr><tr><td>Non-canonical</td><td> $0 . 3 5 3 ^ { \pm 0 . 0 0 4 }$ </td><td> $0 . 5 3 6 ^ { \pm 0 . 0 0 4 }$ </td><td> $0 . 6 4 6 ^ { \pm 0 . 0 0 3 }$ </td><td> $\overline { { 1 . 2 8 0 ^ { \pm 0 . 0 4 6 7 } } }$ </td><td> $4 . 3 4 0 ^ { \pm 0 . 0 1 3 8 }$ </td><td> $\overline { { 9 . 0 2 8 ^ { \pm 0 . 0 7 3 } } }$ </td><td> $2 . 8 5 5 ^ { \pm 0 . 1 0 0 }$ </td></tr></table>

TABLE III

QUANTITATIVE RESULTS OF INTERACTION IMITATION ON THE INTER-X++ DATASET. PHC\* MEANS THAT WE FINE-TUNE THE PHC-BASE [96] MODEL ON THE INTER-X++ DATASET WITH DIFFERENT EPOCHS.
<table><tr><td>Method</td><td>Succ(%)↑</td><td> $E _ { \mathrm { g - m p j p e \downarrow } }$ </td><td> $E _ { \mathrm { m p j p e } } \downarrow$ </td><td> $\mathrm { E _ { a c c \downarrow } }$ </td><td> $\mathrm { E _ { v e l \downarrow } }$ </td></tr><tr><td>PHC-Base [96]</td><td>76.79</td><td>70.26</td><td>70.19</td><td>14.51</td><td>13.34</td></tr><tr><td>PHC* (Epoch 6,000)</td><td>80.17</td><td>61.68</td><td>61.97</td><td>11.92</td><td>11.87</td></tr><tr><td>PHC* (Epoch 7,500)</td><td>80.21</td><td>65.29</td><td>63.64</td><td>12.33</td><td>12.27</td></tr><tr><td> $\mathrm { P H C ^ { * } } \ ( \mathrm { E p o c h ~ } 9 , 0 0 0 )$ </td><td>81.30</td><td>58.58</td><td>61.95</td><td>11.72</td><td>11.75</td></tr><tr><td> $\mathrm { P H C ^ { * } \ ( E p o c h \ 1 0 , 5 0 0 ) }$ </td><td>81.20</td><td>59.73</td><td>62.76</td><td>12.11</td><td>12.06</td></tr></table>

## A. Experimental Setup

Following HumanML3D [34] and InterHuman [7], we split Inter-X++ into training, test, and validation sets with a ratio of 0.8, 0.15, and 0.05, respectively. While single-person motion sequences are typically canonicalized with respect to the first frame, we retain the global translations of both participants to preserve their relative spatial configuration. For diffusionbased models [98], we employ 1,000 diffusion steps during training and a five-step DDIM sampler [99] during inference. All models are trained using NVIDIA A100 GPUs.

For the interaction representation study, we instantiate the same text-conditioned interaction generation model with the five motion representations introduced in Sect. V-A, including axis-angle, quaternion, rotation matrix, 6D rotation, and the non-canonical representation of InterGen [7]. To isolate the effect of the representation, we keep the network architecture, data split, training schedule, and all other configurations identical across variants. Since their outputs lie in different feature spaces, all generated outputs are projected into a unified 6D rotation space [41] for standardized evaluation. For interaction imitation, we build upon the implementation of PHC [96]. We directly evaluate the original PHC-Base policy trained on AMASS as the baseline and further train it on individual Inter-X++ motion sequences in a single-agent simulation environment, following MultiPhys [80]. At inference time, two independently controlled humanoid agents are placed in a shared simulation environment, where contact and collision constraints are jointly enforced to reduce foot sliding, motion jittering, and inter-person penetration. For the unified interaction model, we establish a comprehensive benchmark covering all eight generative and perceptive downstream tasks, evaluate representative baseline methods, and further conduct experiments using our proposed shared OpenHHI encoder. For the hierarchical text study, we evaluate text-conditioned HHI generation using descriptions at different levels of granularity.

## B. Evaluation Protocol

We summarize the adopted evaluation metrics according to task type. For text-conditioned HHI generation, interaction representation, and hierarchical text experiments, we follow HumanML3D [34] and InterHuman [7] and report Top-1/2/3 R-Precision, Frechet Inception Distance (FID) [100], Multi-Modal Distance (MM Dist), Diversity, and MModality. R-Precision and MM Dist evaluate motion-language alignment, FID measures the distributional discrepancy from real interactions, and Diversity and MModality characterize overall and condition-specific motion variation, respectively. For actionconditioned HHI generation, human reaction generation, and stylized HHI generation, we report FID, action recognition accuracy, Diversity, and MModality, where the accuracy is computed using a ST-GCN [64] evaluator to measure condition consistency. Following PHC [96], interaction imitation is evaluated using the success rate (Succ), global and root-relative mean per-joint position errors $( E _ { \mathrm { g - m p j p e } }$ and $E _ { \mathrm { m p j p e } } )$ , acceleration error $( E _ { \mathrm { a c c } } ) _ { \mathrm { : } }$ , and velocity error $( E _ { \mathrm { v e l } } )$ . For perceptive tasks, HHI recognition uses Top-1 and Top-5 accuracy, causal interaction order inference uses binary classification accuracy, HHI captioning uses BLEU [101], ROUGE [102], CIDEr [103], and BERTScore [104], and personality assessment reports the coefficient of determination $( R ^ { 2 } )$ for each personality trait. Higher R-Precision, classification accuracy, captioning scores, $R ^ { 2 }$ , and Succ indicate better performance, whereas lower FID, MM Dist, and imitation errors are preferred; for Diversity and MModality, → denotes proximity to the real-data statistics

TABLE IV  
EXPERIMENTAL RESULTS OF TEXT-CONDITIONED HHI GENERATION ON THE INTER-X++ DATASET, WHERE ± INDICATES 95% CONFIDENCE INTERVALAND → MEANS THE CLOSER THE BETTER. \* DENOTES OUR REPRODUCED RESULTS BASED ON THE OFFICIAL CODEBASE.
<table><tr><td rowspan="2">Method</td><td colspan="3">R Precision↑</td><td rowspan="2">FID ↓</td><td rowspan="2">MM Dist↓</td><td rowspan="2">Diversity→</td><td rowspan="2">MModality ↑</td></tr><tr><td>Top 1</td><td>Top 2</td><td>Top 3</td></tr><tr><td>Real</td><td> $0 . 4 4 8 ^ { \pm 0 . 0 0 3 }$ </td><td> $0 . 6 3 9 ^ { \pm 0 . 0 0 3 }$ </td><td> $0 . 7 4 3 ^ { \pm 0 . 0 0 3 }$ </td><td> $0 . 0 0 1 ^ { \pm 0 . 0 0 0 1 }$ </td><td> $3 . 6 5 2 ^ { \pm 0 . 0 0 9 }$ </td><td> $9 . 6 6 2 ^ { \pm 0 . 0 7 8 }$ </td><td></td></tr><tr><td>TEMOS [43]</td><td> $0 . 0 9 2 ^ { \pm 0 . 0 0 3 }$ </td><td> $0 . 1 7 1 ^ { \pm 0 . 0 0 3 }$ </td><td> $0 . 2 3 8 ^ { \pm 0 . 0 0 2 }$ </td><td> $2 9 . 2 5 8 ^ { \pm 0 . 0 6 9 4 }$ </td><td> $6 . 8 6 7 ^ { \pm 0 . 0 1 3 }$ </td><td> $4 . 7 3 8 ^ { \pm 0 . 0 7 8 }$ </td><td> $0 . 6 7 2 ^ { \pm 0 . 0 4 1 }$ </td></tr><tr><td>T2M [34]</td><td> $0 . 1 8 4 ^ { \pm 0 . 0 1 0 }$ </td><td> $0 . 2 9 8 ^ { \pm 0 . 0 0 6 }$ </td><td> $0 . 3 9 6 ^ { \pm 0 . 0 0 5 }$ </td><td> $5 . 4 8 1 ^ { \pm 0 . 3 8 2 0 }$ </td><td> $9 . 5 7 6 ^ { \pm 0 . 0 0 6 }$ </td><td> $5 . 7 7 1 ^ { \pm 0 . 1 5 1 }$ </td><td> $2 . 7 6 1 ^ { \pm 0 . 0 4 2 }$ </td></tr><tr><td>MDM [42]</td><td> $0 . 2 0 3 ^ { \pm 0 . 0 0 9 }$ </td><td> $0 . 3 2 9 ^ { \pm 0 . 0 0 7 }$ </td><td> $0 . 4 2 6 ^ { \pm 0 . 0 0 5 }$ </td><td> $2 3 . 7 0 1 ^ { \pm 0 . 0 5 6 9 }$ </td><td> $9 . 5 4 8 ^ { \pm 0 . 0 1 4 }$ </td><td> $5 . 8 5 6 ^ { \pm 0 . 0 7 7 }$ </td><td> $3 . 4 9 0 ^ { \pm 0 . 0 6 1 }$ </td></tr><tr><td>MDM(GRU) [42]</td><td> $0 . 1 7 9 ^ { \pm 0 . 0 0 6 }$ </td><td> $0 . 2 9 9 ^ { \pm 0 . 0 0 5 }$ </td><td> $0 . 3 8 7 ^ { \pm 0 . 0 0 7 }$ </td><td> $3 2 . 6 1 7 ^ { \pm 0 . 1 2 2 1 }$ </td><td> $9 . 5 5 7 ^ { \pm 0 . 0 1 9 }$ </td><td> $7 . 0 0 3 ^ { \pm 0 . 1 3 4 }$ </td><td> $\overline { { 3 . 4 3 0 ^ { \pm 0 . 0 3 5 } } }$ </td></tr><tr><td>ComMDM [50]</td><td> $0 . 0 9 0 ^ { \pm 0 . 0 0 2 }$ </td><td> $0 . 1 6 5 ^ { \pm 0 . 0 0 4 }$ </td><td> $0 . 2 3 6 ^ { \pm 0 . 0 0 4 }$ </td><td> $2 9 . 2 6 6 ^ { \pm 0 . 0 6 6 8 }$ </td><td> $6 . 8 7 0 ^ { \pm 0 . 0 1 7 }$ </td><td> $4 . 7 3 4 ^ { \pm 0 . 0 6 7 }$ </td><td> $0 . 7 7 1 ^ { \pm 0 . 0 5 3 }$ </td></tr><tr><td>InterGen [7]</td><td> $0 . 2 0 7 ^ { \pm 0 . 0 0 4 }$ </td><td> $0 . 3 3 5 ^ { \pm 0 . 0 0 5 }$ </td><td> $0 . 4 2 9 ^ { \pm 0 . 0 0 5 }$ </td><td> $5 . 2 0 7 ^ { \pm 0 . 2 1 6 0 }$ </td><td> $9 . 5 8 0 ^ { \pm 0 . 0 1 1 }$ </td><td> $7 . 7 8 8 ^ { \pm 0 . 2 0 8 }$ </td><td> $\mathbf { 3 . 6 8 6 ^ { \pm 0 . 0 5 2 } }$ </td></tr><tr><td>InterMask* [52]</td><td> $0 . 3 8 9 ^ { \pm 0 . 0 0 4 }$ </td><td> $0 . 5 7 7 ^ { \pm 0 . 0 0 5 }$ </td><td> $0 . 6 8 9 ^ { \pm 0 . 0 0 4 }$ </td><td> $1 . 1 9 5 ^ { \pm 0 . 0 3 2 0 }$ </td><td> $3 . 9 6 3 ^ { \pm 0 . 0 1 7 2 }$ </td><td> $9 . 9 0 6 ^ { \pm 0 . 0 7 5 }$ </td><td> $2 . 6 9 5 ^ { \pm 0 . 0 9 0 }$ </td></tr><tr><td>OpenHHI (Ours)</td><td> $\mathbf { 0 . 4 3 7 ^ { \pm 0 . 0 0 5 } }$ </td><td> $\mathbf { 0 . 6 2 7 ^ { \pm 0 . 0 0 4 } }$ </td><td> $\mathbf { 0 . 7 3 2 ^ { \pm 0 . 0 0 5 } }$ </td><td> $\mathbf { 0 . 5 7 3 ^ { \pm 0 . 0 2 0 9 } }$ </td><td> $\mathbf { \frac { 1 } { 3 . 5 9 9 ^ { \pm 0 . 0 1 7 } } }$ </td><td> $\mathbf { 9 . 6 8 5 ^ { \pm 0 . 0 8 5 } }$ </td><td> $1 . 7 9 6 ^ { \pm 0 . 0 7 1 }$ </td></tr></table>

![](images/43fb9cff6d8ce6f0b163244d4252078b76ee82b0c00fc1c3c60d8d0d608d9a89.jpg)  
"Two people stand face to face. They raise their arms high and slowly lower them, giving each other a thumbs up." "The two people face each other and step forward, giving each other three high-fives with both hands."

Fig. 4. Visualization results of text-conditioned human-human interaction generation on the Inter-X++ dataset. Text is provided as the input condition, yielding interaction sequences as the generated output.

and ↑ denotes greater variation. Unless otherwise specified, stochastic generation evaluations are repeated 20 times and reported with 95% confidence intervals, while MModality for text-conditioned generation is evaluated five times.

## C. Human-Human Interaction Representation

To systematically investigate the effect of HHI representations, we train an identical text-conditioned HHI generation model using axis-angle, quaternion, rotation matrix, 6D rotation, and non-canonical representations, strictly keeping the network architecture, data splits, training schedule, and evaluation protocol unchanged. Because these parameterizations reside in distinct feature spaces, all generated outputs are projected into a unified 6D rotation space for standardized evaluation. As reported in Tab. II, the 6D rotation representation achieves the best overall performance, attaining the highest Top-1, Top-2, and Top-3 R-Precision scores alongside the lowest MM-Dist. Although the quaternion representation yields superior FID and MModality scores, its retrieval accuracy and cross-modal semantic alignment remain consistently inferior to those of the 6D rotation. Furthermore, while the non-canonical representation shows competitive potential, it remains less effective overall, and both axis-angle and rotation matrix parameterizations suffer from substantial performance degradation. These empirical results demonstrate that the continuous and compact nature of the 6D parameterization strikes an optimal balance among optimization stability, generation fidelity, and semantic consistency. Consequently, we adopt the 6D rotation representation across all subsequent experiments.

## D. Human-Human Interaction Imitation

The results in Tab. III highlight the benefit of adapting the imitation policy to interaction data. The limited transferability of PHC-Base indicates that motion priors learned from individual human motions are insufficient for reproducing the coordinated dynamics of two-person interactions. In-domain optimization consistently raises the success rate while reducing both spatial tracking errors and temporal motion errors. Among the evaluated checkpoints, the model trained for 9,000 epochs ranks first across every metric, demonstrating more accurate global and root-relative tracking together with smoother motion reproduction. Extending training to 10,500 epochs yields a slight decline rather than further gains, suggesting that performance has already saturated. We therefore select the 9,000 epoch checkpoint for physics-based refinement of the interaction motions in Inter-X++.

## E. Unified Human-Human Interaction Model

In this section, we first establish a unified benchmark for the proposed eight downstream tasks and provide representative methods as baselines under a consistent evaluation protocol.

TABLE V  
EXPERIMENTAL RESULTS OF ACTION-CONDITIONED INTERACTIONGENERATION ON THE INTER-X++ DATASET.
<table><tr><td>Method</td><td>FID↓</td><td> $\operatorname { A c c . \uparrow }$ </td><td> $\mathrm { D i v . } $ </td><td> $\mathrm { M u l t i m o d . } $ </td></tr><tr><td>Real</td><td> $0 . 2 8 1 ^ { \pm 0 . 0 0 2 }$ </td><td> $0 . 9 9 0 ^ { \pm 0 . 0 0 0 0 }$ </td><td> $1 2 . 8 9 0 ^ { \pm 0 . 0 2 8 }$ </td><td> $2 2 . 3 9 1 ^ { \pm 0 . 1 9 5 }$ </td></tr><tr><td>Action2Motion [40]</td><td> $2 0 . 2 9 5 ^ { \pm 1 2 . 0 8 1 }$ </td><td> $0 . 7 6 6 ^ { \pm 0 . 0 0 0 3 }$ </td><td> $1 1 . 5 8 1 ^ { \pm 0 . 0 2 4 }$ </td><td> $1 5 . 3 4 5 ^ { \pm 0 . 2 4 5 }$ </td></tr><tr><td>ACTOR [41]</td><td> $9 . 3 9 2 ^ { \pm 0 . 8 1 6 }$ </td><td> $0 . 8 5 5 ^ { \pm 0 . 0 0 0 3 }$ </td><td> $1 1 . 5 9 4 ^ { \pm 0 . 0 2 9 }$ </td><td> $1 5 . 3 2 7 ^ { \pm 0 . 1 9 5 }$ </td></tr><tr><td>MDM [42]</td><td> $1 2 . 4 2 6 ^ { \pm 2 . 5 8 4 }$ </td><td> $0 . 8 9 6 ^ { \pm 0 . 0 0 0 4 }$ </td><td> $1 3 . 4 9 2 ^ { \pm 0 . 0 3 3 }$ </td><td> $\mathbf { 2 2 . 0 4 2 ^ { \pm 0 . 1 5 3 } }$ </td></tr><tr><td>MDM(GRU) [42]</td><td> $3 5 . 0 0 3 ^ { \pm 7 . 8 7 6 }$ </td><td> $0 . 7 1 \dot { 6 } ^ { \pm 0 . 0 0 0 6 }$ </td><td> $1 2 . 5 7 9 ^ { \pm 0 . 0 3 8 }$ </td><td> $1 6 . 4 5 6 ^ { \pm 0 . 1 0 0 }$ </td></tr><tr><td>Actformer [6]</td><td> $8 . 0 6 7 ^ { \pm 0 . 6 5 3 }$ </td><td> $0 . 9 4 5 ^ { \pm 0 . 0 0 0 7 }$ </td><td> $1 2 . 5 1 2 ^ { \pm 0 . 0 5 }$ </td><td> $1 6 . 1 8 7 ^ { \pm 0 . 1 8 9 }$ </td></tr><tr><td>OpenHHI (Ours)</td><td> $\overline { { 4 . 2 1 5 ^ { \pm 0 . 2 8 4 } } }$ </td><td> $\mathbf { 0 . 9 6 9 ^ { \pm 0 . 0 0 0 4 } }$ </td><td> $\mathbf { 1 2 . 8 7 3 ^ { \pm 0 . 0 3 5 } }$ </td><td> $1 7 . 1 8 4 ^ { \pm 0 . 1 4 2 }$ </td></tr></table>

TABLE VI

EXPERIMENTAL RESULTS OF HUMAN REACTION GENERATION BASED ON ACTION LABELS ON THE INTER-X++ DATASET.
<table><tr><td>Method</td><td>FID↓</td><td> $\operatorname { A c c . \uparrow }$ </td><td> $\mathrm { D i v . } $ </td><td>Multimod.→</td></tr><tr><td>Real</td><td> $0 . 2 6 0 ^ { \pm 0 . 0 0 2 1 }$ </td><td>0.988±0.0000</td><td> $1 2 . 1 1 5 ^ { \pm 0 . 0 3 1 }$ </td><td> $2 1 . 4 9 8 ^ { \pm 0 . 1 3 1 }$ </td></tr><tr><td>MDM [42]</td><td> $6 . 7 4 7 ^ { \pm 0 . 3 1 5 3 }$ </td><td> $0 . 9 0 3 ^ { \pm 0 . 0 0 0 1 }$ </td><td> $1 2 . 2 6 4 ^ { \pm 0 . 0 5 1 }$ </td><td> $1 9 . 6 8 1 ^ { \pm 0 . 2 3 4 }$ </td></tr><tr><td>MDM(GRU) [42]</td><td> $1 9 . 9 6 8 ^ { \pm 1 . 1 7 0 0 }$ </td><td> $0 . 7 5 2 ^ { \pm 0 . 0 0 0 3 }$ </td><td> $1 2 . 3 5 1 ^ { \pm 0 . 0 4 9 }$ </td><td> $1 8 . 0 5 6 ^ { \pm 0 . 1 5 6 }$ </td></tr><tr><td>RAIG [20]</td><td> $6 . 3 7 2 ^ { \pm 0 . 2 1 5 4 }$ </td><td> $0 . 9 0 8 ^ { \pm 0 . 0 0 0 1 }$ </td><td> $1 2 . 3 3 0 ^ { \pm 0 . 0 6 0 }$ </td><td> $2 0 . 0 7 1 ^ { \pm 0 . 2 9 9 }$ </td></tr><tr><td>AGRoL [105]</td><td> $4 . 3 8 6 ^ { \pm 0 . 2 1 8 6 }$ </td><td> $0 . 9 2 5 ^ { \pm 0 . 0 0 0 1 }$ </td><td> $1 2 . 2 0 4 ^ { \pm 0 . 0 4 2 }$ </td><td> $\mathbf { \frac { 1 } { 2 0 . 1 9 9 ^ { \pm 0 . 2 2 6 } } }$ </td></tr><tr><td>OpenHHI (Ours)</td><td> $\mathbf { 3 . 1 7 2 ^ { \pm 0 . 1 6 4 5 } }$ </td><td> $\mathbf { 0 . 9 4 7 ^ { \pm 0 . 0 0 0 1 } }$ </td><td> $\mathbf { i 2 . 1 3 2 ^ { \pm 0 . 0 3 9 } }$ </td><td> $1 8 . 8 7 3 ^ { \pm 0 . 1 8 5 }$ </td></tr></table>

We then apply OpenHHI to every task as a shared HHI encoder and interaction representation, introducing only lightweight downstream modules for different prediction objectives. This comprehensive evaluation examines whether a single unified representation can transfer consistently across diverse conditioning modalities and both generative and perceptive tasks.

1) Text-conditioned Interaction Generation: We compare seven representative text-to-motion and interaction generation methods: TEMOS [43], T2M [34], MDM [42], MDM-GRU [42], ComMDM [50], InterGen [7], and InterMask [52]. The single-person models are extended to synthesize paired motions, and all methods are evaluated under the unified protocol. As shown in Tab. IV, InterMask yields the best results among the compared baselines for all three R-Precision scores, FID, MM Dist, and Diversity, whereas InterGen produces the highest MModality. The substantial gaps between earlier baselines and InterMask also show that jointly modeling the two participants is essential for text-conditioned HHI generation. OpenHHI further delivers substantial overall improvements over these competing baselines across the principal retrieval and distributional metrics. The consistent gains demonstrate that its unified HHI representation supports more accurate textinteraction alignment and higher-quality motion generation while retaining realistic motion diversity.

The visualization examples in Fig. 4 illustrate that models trained on Inter-X++ can express detailed hand movements and coordinated two-person dynamics. For interactions such as “Wave”, “High five” and “Thumb up”, the generated sequences exhibit clearer limb articulation and plausible relative positioning than the counterparts trained on InterHuman, whose lack of dexterous hand annotations leads to ambiguous gestures and more visible penetration artifacts. Additional visual comparisons and video results are provided in the supplementary.

2) Action-conditioned Interaction Generation: Inter-X++ provides 40 semantic categories for action-conditioned HHI generation. We adapt Action2Motion [40], ACTOR [41], MDM [42], MDM-GRU [42], and Actformer [6] to humanhuman interactions and evaluate them using the same data split as in text-conditioned generation. As shown in Tab. V, Actformer obtains the lowest FID and highest recognition accuracy among the baselines, whereas MDM-GRU and MDM more closely match different real-motion variation statistics. This split outcome highlights the difficulty of balancing motion fidelity, condition consistency, and generation diversity. OpenHHI delivers substantial overall improvements over these existing baselines, particularly in distribution fidelity and action consistency, while preserving motion variation close to the real data. These gains show that the unified HHI representation transfers effectively to discrete action control.

TABLE VII  
EXPERIMENTAL RESULTS OF SKELETON-BASED HUMAN-HUMAN INTERACTION RECOGNITION ON THE INTER-X++ DATASET.
<table><tr><td>Method</td><td>Top-1 (%)</td><td>Top-5 (%)</td></tr><tr><td>ST-GCN [64]</td><td>64.62</td><td>90.16</td></tr><tr><td>2s-AGCN [65]</td><td>75.22</td><td>93.73</td></tr><tr><td>HD-GCN [68]</td><td>77.40</td><td>94.73</td></tr><tr><td>CTR-GCN [67]</td><td>82.19</td><td>96.72</td></tr><tr><td>MS-G3D [66]</td><td>83.30</td><td>97.09</td></tr><tr><td>OpenHHI (Ours)</td><td>84.88</td><td>97.13</td></tr></table>

TABLE VIII

EXPERIMENTAL RESULTS OF HUMAN-HUMAN INTERACTION CAPTIONING ON THE INTER-X++ DATASET.
<table><tr><td>Method</td><td>Bleu@1↑</td><td>Bleu@4↑</td><td>Rouge ↑</td><td>Cider ↑</td><td>BertScore ↑</td></tr><tr><td>RAEs [106]</td><td>28.6</td><td>9.7</td><td>34.1</td><td>25.9</td><td>10.2</td></tr><tr><td>Seq2Seq [107]</td><td>53.8</td><td>18.5</td><td>45.2</td><td>61.9</td><td>27.1</td></tr><tr><td>SeqGAN [108]</td><td>45.4</td><td>14.1</td><td>36.8</td><td>52.3</td><td>21.4</td></tr><tr><td>TM2T [17]</td><td>56.8</td><td>21.6</td><td>48.2</td><td>75.5</td><td>32.7</td></tr><tr><td>OpenHHI (Ours)</td><td>70.7</td><td>35.4</td><td>48.9</td><td>49.1</td><td>35.1</td></tr></table>

3) Human Reaction Generation: Human reaction generation predicts the reactor’s motion from the observed motion of the actor. We evaluate adapted versions of MDM [42] and MDM-GRU [42] together with the RAIG [20] and AGRoL [105]. Among the baselines as shown in Tab. VI, AGRoL achieves the best results across the evaluation metrics, confirming the value of explicitly modeling the dependency between the initiating and reactive motions. OpenHHI consistently improves upon the baselines across all evaluation criteria. The considerable gains indicate that its unified HHI representation effectively captures the coupling between an initiating motion and its corresponding response, producing reactions that are both condition-consistent and diverse.

4) Human Interaction Recognition: We evaluate ST-GCN [64], 2s-AGCN [65], HD-GCN [68], CTR-GCN [67], and MS-G3D [66] using only the skeleton-joint stream, without ensembling bone or motion streams. As shown in Tab. VII, MS-G3D obtains the best Top-1 and Top-5 accuracy among the compared baselines. The remaining recognition errors indicate that distinguishing interactions with similar global trajectories requires sensitivity to fine-grained gestures and actor-reactor variations. OpenHHI further improves interaction recognition performance over these traditional skeleton-based baselines. The gains demonstrate that joint reconstruction and language supervision produce a unified HHI representation that is more discriminative for closely related interaction categories.

![](images/edda10655a0fe5fb4b9b35ed45cdda74f1d04a73ea6aa5f32bb45188a91f77c5.jpg)  
Fig. 5. Qualitative results of human-human interaction captioning on Inter-X++. Each example presents a sampled interaction sequence together with its ground-truth description and the caption generated by OpenHHI. The predictions coherently describe the coordinated behaviors of both participants, including their roles, body movements, and interaction details.

TABLE IX  
EXPERIMENTAL RESULTS OF CAUSAL INTERACTION ORDER INFERENCE ON THE INTER-X++ DATASET.
<table><tr><td>Method</td><td>Accuracy (%)</td></tr><tr><td>ST-GCN [64]</td><td>62.3</td></tr><tr><td>2s-AGCN [65]</td><td>68.2</td></tr><tr><td>HD-GCN [68]</td><td>70.6</td></tr><tr><td>CTR-GCN [67]</td><td>74.5</td></tr><tr><td>MS-G3D [66]</td><td>76.8</td></tr><tr><td>OpenHHI (Ours)</td><td>79.1</td></tr></table>

TABLE X

EXPERIMENTAL RESULTS OF ACTION-CONDITIONED STYLIZED HUMANINTERACTION GENERATION ON THE INTER-X++ DATASET.
<table><tr><td>Method</td><td>FID↓</td><td>Acc.↑</td><td>Div.→</td><td>Multimod.→</td></tr><tr><td>Real</td><td>0.281±0.002</td><td> $0 . 9 9 0 ^ { \pm 0 . 0 0 0 0 }$ </td><td> $1 2 . 8 9 0 ^ { \pm 0 . 0 2 8 }$ </td><td>22.391±0.195</td></tr><tr><td>Action2Motion [40]</td><td> $\overline { { 2 1 . 1 8 2 ^ { \pm 1 3 . 3 1 9 } } }$ </td><td> $0 . 7 3 7 ^ { \pm 0 . 0 0 0 5 }$ </td><td> $\overline { { 1 1 . 4 9 2 ^ { \pm 0 . 0 3 2 } } }$ </td><td> $1 4 . 9 3 4 ^ { \pm 0 . 2 5 8 }$ </td></tr><tr><td>ACTOR [41]</td><td> $9 . 7 9 6 ^ { \pm 0 . 8 6 2 }$ </td><td> $0 . 8 6 7 ^ { \pm 0 . 0 0 0 3 }$ </td><td> $1 1 . 8 6 2 ^ { \pm 0 . 0 3 9 }$ </td><td> $1 5 . 1 7 4 ^ { \pm 0 . 2 4 5 }$ </td></tr><tr><td>MDM [42]</td><td> $1 1 . 7 6 2 ^ { \pm 1 . 8 5 4 }$ </td><td> $0 . 9 1 2 ^ { \pm 0 . 0 0 0 2 }$ </td><td> $1 3 . 0 2 5 ^ { \pm 0 . 0 2 8 }$ </td><td> $\mathbf { 2 1 . 7 4 2 ^ { \pm 0 . 1 0 6 } }$ </td></tr><tr><td>MDM(GRU) [42]</td><td> $3 1 . 6 8 8 ^ { \pm 4 . 4 9 2 }$ </td><td> $0 . 7 5 3 ^ { \pm 0 . 0 0 0 6 }$ </td><td> $\overline { { 1 2 . 2 5 9 ^ { \pm 0 . 0 3 9 } } }$ </td><td> $1 6 . 2 7 1 ^ { \pm 0 . 2 0 6 }$ </td></tr><tr><td>Actformer [6]</td><td> $8 . 5 4 4 ^ { \pm 0 . 6 8 4 }$ </td><td> $0 . 9 3 2 ^ { \pm 0 . 0 0 0 6 }$ </td><td> $1 2 . 1 \dot { 1 } 6 ^ { \pm 0 . 0 6 2 }$ </td><td> $1 6 . 1 2 2 ^ { \pm 0 . 1 8 3 }$ </td></tr><tr><td>OpenHHI (Ours)</td><td> $\mathbf { 5 . 2 8 6 ^ { \pm 0 . 4 2 1 } }$ </td><td> $\mathbf { 0 . 9 5 1 ^ { \pm 0 . 0 0 0 4 } }$ </td><td> $\mathbf { 1 2 . 9 1 4 ^ { \pm 0 . 0 4 4 } }$ </td><td> $1 7 . 0 1 8 ^ { \pm 0 . 1 6 2 }$ </td></tr></table>

5) Human interaction captioning: Human interaction captioning maps paired motion sequences to natural language descriptions. Following the text-generation split, we adapt RAEs [106], Seq2Seq [107], SeqGAN [108], and TM2T [17] to two-person inputs. As depicted in Tab. VIII, TM2T provides the best baseline results across the language evaluation metrics, while the recurrent autoencoder performs substantially worse, reflecting its difficulty in retaining long-range interaction dynamics. The attention-based and adversarial alternatives improve over RAEs but still leave considerable room for fine-grained interaction description. OpenHHI achieves the best BLEU@1, BLEU@4, ROUGE, and BERTScore results, demonstrating stronger lexical overlap and semantic similarity. However, its CIDEr score of 49.1 remains below the 75.5 achieved by TM2T, indicating that further improvements are needed in consensus-oriented caption generation. The qualitative examples in Fig. 5 further demonstrate that the generated captions are semantically coherent and accurately reflect the fine-grained interaction details between the two participants.

TABLE XI  
THE $R ^ { 2 }$ RESULTS (%) OF THE PERSONALITY ASSESSMENT ON THE INTER-X++ DATASET.
<table><tr><td>Method</td><td>Openness</td><td>Conscientiousness</td><td>Extraversion</td><td>Agreeableness</td><td>Neuroticism</td></tr><tr><td>ST-GCN [64]</td><td>21.16</td><td>25.38</td><td>34.91</td><td>23.67</td><td>13.02</td></tr><tr><td>2s-AGCN [65]</td><td>23.46</td><td>31.27</td><td>38.72</td><td>24.88</td><td>13.57</td></tr><tr><td>HD-GCN [68]</td><td>25.92</td><td>33.19</td><td>41.33</td><td>26.83</td><td>14.29</td></tr><tr><td>CTR-GCN [67]</td><td>27.78</td><td>35.41</td><td>43.52</td><td>29.43</td><td>15.63</td></tr><tr><td>MS-G3D [66]</td><td>28.36</td><td>37.88</td><td>46.23</td><td>29.07</td><td>16.35</td></tr><tr><td>OpenHHI (Ours)</td><td>30.74</td><td>40.12</td><td>48.67</td><td>31.05</td><td>18.21</td></tr></table>

6) Causal order inference: Causal order inference determines which participant initiates an interaction and which participant responds. We formulate it as a binary classification and evaluate the same five graph-based backbones used for interaction recognition. As shown in Tab. IX, MS-G3D obtains the best accuracy among these baselines. OpenHHI surpasses the compared methods and yields a clear improvement in causal order inference. This result suggests that the unified HHI representation retains directional interaction cues and participant-specific dynamics that are essential for identifying the actor and reactor.

7) Stylized human interaction generation: For stylized generation, we augment Action2Motion [40], ACTOR [41], MDM [42], MDM-GRU [42], and Actformer [6] with the relationship familiarity level as a style code. Among the baselines in Tab. X, Actformer achieves the lowest FID and highest recognition accuracy, whereas MDM more closely approaches the real-data motion variation statistics. The outcome suggests that maintaining interaction semantics while expressing relationship-dependent styles remains challenging for the evaluated generators. OpenHHI achieves substantial improvements in both distribution fidelity and condition consistency while preserving realistic motion variation. This balanced performance indicates that the unified HHI representation retains the underlying interaction category while allowing relationship cues to modulate how an interaction is performed.

8) Personality assessment: Personality assessment predicts five personality traits from interaction motions. To prevent identity leakage, we split the training, validation, and test data by participant identity and formulate each trait as a regression target. Among the graph-based baselines in Tab. XI, MS-

TABLE XII  
ABLATION OF THE RELATIVE WEIGHTING BETWEEN THE RECONSTRUCTION AND INTERACTION-UNDERSTANDING OBJECTIVES IN OPENHHI FORLEVEL-3 TEXT-CONDITIONED HHI GENERATION. WE FIX $\omega _ { \mathrm { r e c } } = 1$ AND VARY $\rho = \omega _ { \mathrm { u n d } } / \omega _ { \mathrm { r e c } }$ . INTERMASK\* DENOTES OUR REPRODUCED BASELINEUSING THE OFFICIAL IMPLEMENTATION. BOLD AND UNDERLINED VALUES INDICATE THE BEST AND SECOND-BEST RESULTS, RESPECTIVELY.
<table><tr><td rowspan="2">ρ</td><td colspan="3">R Precision↑</td><td rowspan="2">FID ↓</td><td rowspan="2">MM Dist↓</td><td rowspan="2">Diversity→</td><td rowspan="2">MModality ↑</td></tr><tr><td>Top 1</td><td>Top 2</td><td>Top 3</td></tr><tr><td>Real</td><td> $0 . 4 4 8 ^ { \pm 0 . 0 0 3 }$ </td><td> $0 . 6 3 9 ^ { \pm 0 . 0 0 3 }$ </td><td> $0 . 7 4 3 ^ { \pm 0 . 0 0 3 }$ </td><td> $0 . 0 0 1 ^ { \pm 0 . 0 0 0 1 }$ </td><td> $3 . 6 5 2 ^ { \pm 0 . 0 0 9 }$ </td><td> $9 . 6 6 2 ^ { \pm 0 . 0 7 8 }$ </td><td></td></tr><tr><td>InterMask* [52]</td><td> $0 . 3 8 9 ^ { \pm 0 . 0 0 4 }$ </td><td> $0 . 5 7 7 ^ { \pm 0 . 0 0 5 }$ </td><td> $0 . 6 8 9 ^ { \pm 0 . 0 0 4 }$ </td><td> $1 . 1 9 5 ^ { \pm 0 . 0 3 2 0 }$ </td><td> $3 . 9 6 3 ^ { \pm 0 . 0 1 7 2 }$ </td><td> $9 . 9 0 6 ^ { \pm 0 . 0 7 5 }$ </td><td> $\mathbf { 2 . 6 9 5 ^ { \pm 0 . 0 9 0 } }$ </td></tr><tr><td>0.1</td><td> $0 . 4 2 2 ^ { \pm 0 . 0 0 5 }$ </td><td> $0 . 6 1 3 ^ { \pm 0 . 0 0 4 }$ </td><td> $0 . 7 2 2 ^ { \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 3 6 6 ^ { \pm 0 . 0 1 8 3 } }$ </td><td> $3 . 6 6 0 ^ { \pm 0 . 0 1 8 }$ </td><td> $9 . 6 1 6 ^ { \pm 0 . 0 8 5 }$ </td><td> $1 . 9 0 1 ^ { \pm 0 . 0 5 7 }$ </td></tr><tr><td>0.5</td><td> $\mathbf { 0 . 4 3 7 ^ { \pm 0 . 0 0 5 } }$ </td><td> $\mathbf { 0 . 6 2 7 ^ { \pm 0 . 0 0 4 } }$ </td><td> $0 . 7 3 2 ^ { \pm 0 . 0 0 5 }$ </td><td> $0 . 5 7 3 ^ { \pm 0 . 0 2 0 9 }$ </td><td> $\mathbf { 3 . 5 9 9 ^ { \pm 0 . 0 1 7 } }$ </td><td> $\mathbf { 9 . 6 8 5 ^ { \pm 0 . 0 8 5 } }$ </td><td> $\overline { { 1 . 7 9 6 ^ { \pm 0 . 0 7 1 } } }$ </td></tr><tr><td>1.0</td><td> $0 . 4 2 9 ^ { \pm 0 . 0 0 5 }$ </td><td> $0 . 6 2 4 ^ { \pm 0 . 0 0 5 }$ </td><td> $\mathbf { 0 . 7 3 3 ^ { \pm 0 . 0 0 5 } }$ </td><td> $0 . 4 8 0 ^ { \pm 0 . 0 1 9 6 }$ </td><td> $3 . 6 0 2 ^ { \pm 0 . 0 2 1 }$ </td><td> $9 . 7 4 8 ^ { \pm 0 . 0 8 4 }$ </td><td> $1 . 7 8 4 ^ { \pm 0 . 0 5 0 }$ </td></tr><tr><td>2.0</td><td> $\overline { { 0 . 4 0 7 ^ { \pm 0 . 0 0 4 } } }$ </td><td> $\overline { { 0 . 5 9 6 ^ { \pm 0 . 0 0 4 } } }$ </td><td> $0 . 7 0 5 ^ { \pm 0 . 0 0 3 }$ </td><td> $\overline { { 0 . 9 9 8 ^ { \pm 0 . 0 2 7 9 } } }$ </td><td> $\overline { { 3 . 8 0 5 ^ { \pm 0 . 0 1 3 } } }$ </td><td> $9 . 6 9 3 ^ { \pm 0 . 1 0 1 }$ </td><td> $1 . 8 2 9 ^ { \pm 0 . 0 6 0 }$ </td></tr></table>

TABLE XIII

QUANTITATIVE RESULTS OF OPENHHI FOR TEXT-CONDITIONED HHI GENERATION USING HIERARCHICAL DESCRIPTIONS OF DIFFERENT GRANULARITIES. $\mathrm { ^ { * } R E A L } ^ { \mathrm { * } }$ REPORTS THE STATISTICS OF GROUND-TRUTH MOTIONS PAIRED WITH THE CORRESPONDING TEXT LEVEL.
<table><tr><td rowspan="2">Texts</td><td rowspan="2">Method</td><td colspan="3">R Precision↑</td><td rowspan="2">FID ↓</td><td rowspan="2">MM Dist↓</td><td rowspan="2">Diversity→</td><td rowspan="2">MModality ↑</td></tr><tr><td>Top 1</td><td>Top 2</td><td>Top 3</td></tr><tr><td rowspan="2">Level-1</td><td>Real</td><td> $0 . 3 9 4 ^ { \pm 0 . 0 0 3 }$ </td><td> $0 . 5 8 4 ^ { \pm 0 . 0 0 4 }$ </td><td> $0 . 6 9 4 ^ { \pm 0 . 0 0 3 }$ </td><td> $0 . 0 0 1 ^ { \pm 0 . 0 0 0 1 }$ </td><td> $3 . 7 1 9 ^ { \pm 0 . 0 1 1 }$ </td><td> $9 . 0 1 3 ^ { \pm 0 . 0 7 2 }$ </td><td></td></tr><tr><td>OpenHHI</td><td> $\mathbf { 0 . 3 7 6 ^ { \pm 0 . 0 0 4 } }$ </td><td> $\mathbf { 0 . 5 4 9 ^ { \pm 0 . 0 0 4 } }$ </td><td> $\mathbf { 0 . 6 4 7 ^ { \pm 0 . 0 0 3 } }$ </td><td> $\mathbf { 2 . 6 5 4 ^ { \pm 0 . 0 5 1 6 } }$ </td><td> $\mathbf { 4 . 1 8 7 ^ { \pm 0 . 0 1 8 } }$ </td><td> $\mathbf { 8 . 0 6 0 ^ { \pm 0 . 0 7 0 } }$ </td><td> $2 . 0 8 6 ^ { \pm 0 . 0 7 4 }$ </td></tr><tr><td rowspan="2">Level-2</td><td>Real</td><td> $0 . 3 9 8 ^ { \pm 0 . 0 0 4 }$ </td><td> $0 . 5 9 1 ^ { \pm 0 . 0 0 4 }$ </td><td> $0 . 6 9 9 ^ { \pm 0 . 0 0 3 }$ </td><td> $0 . 0 0 1 ^ { \pm 0 . 0 0 0 1 }$ </td><td> $3 . 7 0 4 ^ { \pm 0 . 0 1 1 }$ </td><td> $8 . 8 1 9 ^ { \pm 0 . 0 7 6 }$ </td><td></td></tr><tr><td>OpenHHI</td><td> $\mathbf { 0 . 3 8 4 ^ { \pm 0 . 0 0 4 } }$ </td><td> $\mathbf { 0 . 5 6 6 ^ { \pm 0 . 0 0 4 } }$ </td><td> $\mathbf { 0 . 6 6 9 ^ { \pm 0 . 0 0 4 } }$ </td><td> $\mathbf { 2 . 7 7 4 ^ { \pm 0 . 0 6 2 5 } }$ </td><td> $\mathbf { 4 . 0 5 3 ^ { \pm 0 . 0 1 5 } }$ </td><td> $\mathbf { 8 . 4 0 1 ^ { \pm 0 . 0 7 6 } }$ </td><td> $2 . 0 3 6 ^ { \pm 0 . 0 5 7 }$ </td></tr><tr><td rowspan="2">Level-3</td><td>Real</td><td> $0 . 4 4 8 ^ { \pm 0 . 0 0 3 }$ </td><td> $0 . 6 3 9 ^ { \pm 0 . 0 0 3 }$ </td><td> $0 . 7 4 3 ^ { \pm 0 . 0 0 3 }$ </td><td> $0 . 0 0 1 ^ { \pm 0 . 0 0 0 1 }$ </td><td> $3 . 6 5 2 ^ { \pm 0 . 0 0 9 }$ </td><td> $9 . 6 6 2 ^ { \pm 0 . 0 7 8 }$ </td><td></td></tr><tr><td>OpenHHI</td><td> $\mathbf { 0 . 4 3 7 ^ { \pm 0 . 0 0 5 } }$ </td><td> $\mathbf { 0 . 6 2 7 ^ { \pm 0 . 0 0 4 } }$ </td><td> $\mathbf { 0 . 7 3 2 ^ { \pm 0 . 0 0 5 } }$ </td><td> $\mathbf { 0 . 5 7 3 ^ { \pm 0 . 0 2 0 9 } }$ </td><td> $\mathbf { 3 . 5 9 9 ^ { \pm 0 . 0 1 7 } }$ </td><td> $\mathbf { 9 . 6 8 5 ^ { \pm 0 . 0 8 5 } }$ </td><td> $1 . 7 9 6 ^ { \pm 0 . 0 7 1 }$ </td></tr></table>

G3D performs best on most personality dimensions, while CTR-GCN gives the highest Agreeableness score. OpenHHI consistently improves the prediction results across all five personality dimensions. These gains suggest that its unified HHI representation encodes not only categorical interaction semantics but also subtle behavioral patterns associated with individual motion style. However, the comparatively low $R ^ { 2 }$ values confirm that personality cues are considerably subtler and more challenging than categorical action semantics.

9) Balancing Reconstruction and Understanding Objectives: We study the balance between the interaction reconstruction and understanding objectives in Eq. (16) by fixing $\omega _ { \mathrm { r e c } } = 1$ and varying $\rho = \omega _ { \mathrm { u n d } } / \omega _ { \mathrm { r e c } }$ over $\{ 0 . 1 , 0 . 5 , 1 . 0 , 2 . 0 \}$ As shown in Tab. XII, a small $\rho$ favors motion fidelity but weakens motion-language alignment, whereas an overly large $\rho$ degrades distributional quality by suppressing motion details. The setting $\rho = 0 . 5$ provides the best overall balance across R-Precision, MM Dist, and Diversity, confirming the complementary roles of the two objectives; we therefore adopt it as the default for all the downstream tasks.

## F. Hierarchical Texts for Generation

We examine how textual granularity affects HHI generation by evaluating OpenHHI with three description levels under identical settings. Level-1 uses short descriptions simplified from the original annotations, Level-2 provides more complete summaries of the overall interaction, and Level-3 specifies participant movements, body-part actions, and spatial relations. This comparison isolates the effect of increasingly detailed language conditions. As shown in Tab. XIII, Level-3 achieves the highest R-Precision, indicating that fine-grained descriptions enable the model to learn more accurate motion-language alignment. In contrast, shorter descriptions omit participantspecific and spatial details, increasing the ambiguity between a prompt and its valid motion realizations. The qualitative results in Fig. 6 further show that detailed prompts provide stronger control over individual actions and relative positioning, whereas simpler prompts impose fewer constraints and therefore allow greater generation diversity.

## VII. CONCLUSION AND LIMITATIONS

Inter-X++ brings accurately captured whole-body and hand motions, physics-refined interaction sequences, and a broad annotation suite into one resource for HHI research. Its hierarchical narratives, action semantics, causal role order, contact cues, and social attributes expose complementary aspects of how two people move, respond, and relate to one another, supporting evaluation across versatile downstream tasks. Our extensive experiments examine this resource from multiple angles: the interaction representation study establishes a consistent basis for comparison, physical imitation validates the benefit of interaction-domain adaptation, and hierarchicaltext evaluation reveals how linguistic detail trades generation diversity for controllability. Results throughout the benchmark further show that OpenHHI can reuse one interaction encoder across varied objectives while retaining both kinematic and semantic information. Together, our data, annotations, unified representation and systematic evaluation provide a solid basis for promoting in-depth research works on HHI analysis.

Limitations. Our work has some limitations in the following aspects. 1) Facial expressions. Inter-X++ focuses on body and

![](images/e6b29861dbaec2f8d57bba2a721ba74d3bdec79d404100bc663b39a5b4d20bc8.jpg)  
Level-1: They exchange playful shoulder taps using their right hands.

![](images/cc83a5153b48b5a3159a36a88e731c6a6c8cd3f2c50ffd409a7af817d7149456.jpg)

![](images/8adb8a8ef06c8904a7c67acdeb1ff90bade1fa531b337413f225d8807a48c1c4.jpg)  
Level-1: Two individuals stand facing each other and exchange a thumbs up.  
Level-2: Both people exchange playful shoulder taps using their right hands.

![](images/db988f735a649a8b7163170e3b378b5adc327b0a27350fc0dc1a04af8d10d139.jpg)  
Level-2: They stand face to face. They raise their arms high and slowly lower them, giving each other a thumbs up

![](images/6cae17b78fd68f43cc9b6df04cc8cac8630015c06660d3449850e42511c18630.jpg)  
Level-3: One person hits the left shoulder of the other person with his/her right hand, and then the other person reciprocates by hitting that person's left shoulder with his/her right hand.

![](images/c6da7331bb0067fa9203f539a3018064d4c7e1e761286db9c95ab7a14229bd2f.jpg)  
Level-3: The two individuals stand opposite one another. The initial person extends his/her right hand and lifts it over his/her head, offering a thumbs up to the second individual. The second individual extends his/her right hand and proceeds to shake it in an upward and downward motion.

Fig. 6. Qualitative comparison of HHI generation conditioned on hierarchical descriptions from Inter-X++. Each column presents motions generated for the same interaction at three levels of textual granularity. Coarser prompts permit broader motion variations, whereas finer descriptions provide more precise control.

hand motions captured in an indoor MoCap environment and does not include facial behavior, as non-professional performers may not consistently produce expressions that match the enacted interactions. Future data collection with professional actors or in more natural settings could better capture the relationship between facial emotion and interactive motion. 2) Interaction duration. The current dataset consists primarily of atomic interactions rather than long, compositional sequences with frequent behavioral transitions. Extending it to continuous scenarios would support the study of more complex social dynamics. 3) Hierarchical text-motion alignment. Our experiments across different description levels provide only an initial investigation of how linguistic granularity affects motion generation; more comprehensive alignment objectives, evaluation protocols, and conditioning strategies remain to be explored. 4) Physical interaction. Although we provide and evaluate physics-based HHI imitation, applying these findings to embodied settings, particularly human-robot interaction, requires further study of contact dynamics, safety constraints, and real-world deployment.

## REFERENCES

[1] K. Ji, Y. Shi, Z. Jin, K. Chen, L. Xu, Y. Ma, J. Yu, and J. Wang, “Towards immersive human-x interaction: A real-time framework for physically plausible motion synthesis,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2025, pp. 10 173–10 183.

[2] Z. Liu, J. Ge, M. Xiong, J. Gu, B. Tang, W. Jing, and S. Chen, “It takes two: Learning interactive whole-body control between humanoid robots,” arXiv preprint arXiv:2510.10206, 2025.

[3] W.-J. Huang, Y.-Y. Zhang, Y.-L. Wei, Z.-W. Xia, J. Tan, Y.- M. Li, Z. Zhao, and W.-S. Zheng, “Learning whole-body humanhumanoid interaction from human-human demonstrations,” arXiv preprint arXiv:2601.09518, 2026.

[4] Y. Pang, Q. Ke, H. Rahmani, J. Bailey, and J. Liu, “Igformer: Interaction graph transformer for skeleton-based human interaction recognition,” in Proc. Eur. Conf. Comput. Vis. Springer, 2022, pp. 605–622.

[5] H. Duan, M. Xu, B. Shuai, D. Modolo, Z. Tu, J. Tighe, and A. Bergamo, “c: Towards skeleton-based action recognition in the wild,” in Proc. Int. Conf. Comput. Vis., 2023, pp. 13 634–13 644.

[6] L. Xu, Z. Song, D. Wang, J. Su, Z. Fang, C. Ding, W. Gan, Y. Yan, X. Jin, X. Yang et al., “Actformer: A gan-based transformer towards general action-conditioned 3d human motion generation,” in Proc. Int. Conf. Comput. Vis., 2023, pp. 2228–2238.

[7] H. Liang, W. Zhang, W. Li, J. Yu, and L. Xu, “Intergen: Diffusionbased multi-human motion generation under complex interactions,” arXiv preprint arXiv:2304.05684, 2023.

[8] N. Van der Aa, X. Luo, G.-J. Giezeman, R. T. Tan, and R. C. Veltkamp, “Umpm benchmark: A multi-person dataset with synchronized video and motion capture data for evaluation of articulated human motion and interaction,” in Proc. Int. Conf. Comput. Vis. Worksh. IEEE, 2011, pp. 1264–1269.

[9] K. Yun, J. Honorio, D. Chattopadhyay, T. L. Berg, and D. Samaras, “Two-person interaction detection using body-pose features and multiple instance learning,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. IEEE, 2012, pp. 28–35.

[10] M. Fieraru, M. Zanfir, E. Oneata, A.-I. Popa, V. Olaru, and C. Sminchisescu, “Three-dimensional reconstruction of human interactions,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2020, pp. 7214–7223.

[11] Y. Yin, C. Guo, M. Kaufmann, J. J. Zarate, J. Song, and O. Hilliges, “Hi4d: 4d instance segmentation of close human interaction,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2023, pp. 17 016–17 027.

[12] T. Baltrusaitis, C. Ahuja, and L.-P. Morency, “Multimodal machineˇ learning: A survey and taxonomy,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 41, no. 2, pp. 423–443, 2018.

[13] P. Xu, X. Zhu, and D. A. Clifton, “Multimodal learning with transformers: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., 2023.

[14] C. McLean, M. Meendering, T. Swartz, O. Gabbay, A. Olsen, R. Jacobs, N. Rosen, P. de Bree, T. Garcia, G. Merrill et al., “Embody 3d: A large-scale multimodal motion and behavior dataset,” arXiv preprint arXiv:2510.16258, 2025.

[15] L. Xu, X. Lv, Y. Yan, X. Jin, S. Wu, C. Xu, Y. Liu, Y. Zhou, F. Rao, X. Sheng et al., “Inter-x: Towards versatile human-human interaction analysis,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2024, pp. 22 260–22 271.

[16] L. Xu, Y. Zhou, Y. Yan, X. Jin, W. Zhu, F. Rao, X. Yang, and W. Zeng, “Regennet: Towards human action-reaction synthesis,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2024, pp. 1759–1769.

[17] C. Guo, X. Zuo, S. Wang, and L. Cheng, “Tm2t: Stochastic and tokenized modeling for the reciprocal generation of 3d human motions and texts,” in Proc. Eur. Conf. Comput. Vis. Springer, 2022, pp. 580– 597.

[18] B. Jiang, X. Chen, W. Liu, J. Yu, G. Yu, and T. Chen, “Motiongpt: Human motion as a foreign language,” arXiv preprint arXiv:2306.14795, 2023.

[19] B. Chopin, H. Tang, N. Otberdout, M. Daoudi, and N. Sebe, “In-

teraction transformer for human reaction generation,” IEEE Trans. Multimedia, 2023.

[20] M. Tanaka and K. Fujiwara, “Role-aware interaction generation from textual description,” in Proc. Int. Conf. Comput. Vis., 2023, pp. 15 999– 16 009.

[21] K. Aberman, Y. Weng, D. Lischinski, D. Cohen-Or, and B. Chen, “Unpaired motion style transfer from video to animation,” ACM Trans. Graph., vol. 39, no. 4, pp. 64–1, 2020.

[22] E. Ng, D. Xiang, H. Joo, and K. Grauman, “You2me: Inferring body pose in egocentric video via first and second person interactions,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2020, pp. 9890–9900.

[23] J. Liu, A. Shahroudy, M. Perez, G. Wang, L.-Y. Duan, and A. C. Kot, “Ntu rgb+ d 120: A large-scale benchmark for 3d human activity understanding,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 42, no. 10, pp. 2684–2701, 2019.

[24] W. Guo, X. Bie, X. Alameda-Pineda, and F. Moreno-Noguer, “Multiperson extreme motion prediction,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2022, pp. 13 053–13 064.

[25] A. Ghosh, R. Dabral, V. Golyanik, C. Theobalt, and P. Slusallek, “Remos: Reactive 3d motion synthesis for two-person interactions,” arXiv preprint arXiv:2311.17057, 2023.

[26] R. Li, Y. Zhang, Y. Zhang, Y. Zhang, M. Su, J. Guo, Z. Liu, Y. Liu, and X. Li, “Interdance: Reactive 3d dance generation with realistic duet interactions,” arXiv preprint arXiv:2412.16982, 2024.

[27] L. Xu, C. Lan, W. Zeng, and C. Lu, “Skeleton-based mutually assisted interacted object localization and human action recognition,” IEEE Trans. Multimedia, 2022.

[28] C. Ionescu, D. Papava, V. Olaru, and C. Sminchisescu, “Human3.6m: Large scale datasets and predictive methods for 3d human sensing in natural environments,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 36, no. 7, pp. 1325–1339, 2013.

[29] N. Mahmood, N. Ghorbani, N. F. Troje, G. Pons-Moll, and M. J. Black, “Amass: Archive of motion capture as surface shapes,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2019, pp. 5442–5451.

[30] J. Lin, A. Zeng, S. Lu, Y. Cai, R. Zhang, H. Wang, and L. Zhang, “Motion-x: A large-scale 3d expressive whole-body human motion dataset,” arXiv preprint arXiv:2307.00818, 2023.

[31] Y. Zhang, J. Lin, A. Zeng, G. Wu, S. Lu, Y. Fu, Y. Cai, R. Zhang, H. Wang, and L. Zhang, “Motion-x++: A large-scale multimodal 3d whole-body human motion dataset,” arXiv preprint arXiv:2501.05098, 2025.

[32] W. Zhu, X. Ma, D. Ro, H. Ci, J. Zhang, J. Shi, F. Gao, Q. Tian, and Y. Wang, “Human motion generation: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 4, pp. 2430–2449, 2023.

[33] A. R. Punnakkal, A. Chandrasekaran, N. Athanasiou, A. Quiros-Ramirez, and M. J. Black, “Babel: bodies, action and behavior with english labels,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2021, pp. 722–731.

[34] C. Guo, S. Zou, X. Zuo, S. Wang, W. Ji, X. Li, and L. Cheng, “Generating diverse and natural 3d human motions from text,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2022, pp. 5152–5161.

[35] R. Li, S. Yang, D. A. Ross, and A. Kanazawa, “Ai choreographer: Music conditioned 3d dance generation with aist++,” 2021.

[36] O. Taheri, N. Ghorbani, M. J. Black, and D. Tzionas, “Grab: A dataset of whole-body human grasping of objects,” in Proc. Eur. Conf. Comput. Vis., 2020, pp. 581–600.

[37] M. Hassan, D. Ceylan, R. Villegas, J. Saito, J. Yang, Y. Zhou, and M. J. Black, “Stochastic scene-aware motion prediction,” in Proc. Int. Conf. Comput. Vis., 2021, pp. 11 374–11 384.

[38] Z. Wang, Y. Chen, T. Liu, Y. Zhu, W. Liang, and S. Huang, “HUMAN-ISE: Language-conditioned human motion generation in 3d scenes,” in Proc. Adv. Neural Inform. Process. Syst., 2022.

[39] J. Chen, X. Li, J. Cao, Z. Zhu, W. Dong, M. Liu, Y. Wen, Y. Yu, L. Zhang, and W. Zhang, “Rhino: Learning real-time humanoidhuman-object interaction from human demonstrations,” arXiv preprint arXiv:2502.13134, 2025.

[40] C. Guo, X. Zuo, S. Wang, S. Zou, Q. Sun, A. Deng, M. Gong, and L. Cheng, “Action2motion: Conditioned generation of 3d human motions,” in ACM Multimedia. ACM, 2020, pp. 2021–2029.

[41] M. Petrovich, M. J. Black, and G. Varol, “Action-conditioned 3d human motion synthesis with transformer vae,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2021, pp. 10 985–10 995.

[42] G. Tevet, S. Raab, B. Gordon, Y. Shafir, D. Cohen-Or, and A. H. Bermano, “Human motion diffusion model,” arXiv preprint arXiv:2209.14916, 2022.

[43] M. Petrovich, M. J. Black, and G. Varol, “TEMOS: Generating diverse human motions from textual descriptions,” in Proc. Eur. Conf. Comput Vis., 2022, pp. 480–497.

[44] S. Lu, L.-H. Chen, A. Zeng, J. Lin, R. Zhang, L. Zhang, and H.-Y. Shum, “Humantomato: Text-aligned whole-body motion generation,” arXiv preprint arXiv:2310.12978, 2023.

[45] M. Zhang, Z. Cai, L. Pan, F. Hong, X. Guo, L. Yang, and Z. Liu, “Motiondiffuse: Text-driven human motion generation with diffusion model,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 6, pp. 4115–4128, 2024.

[46] T. Ao, Z. Zhang, and L. Liu, “Gesturediffuclip: Gesture diffusion model with clip latents,” ACM Trans. Graph., 2023.

[47] Z. Wu, Y. Sun, Y. Chen, X. Gu, R. Liu, and J. Chen, “Intermamba: Efficient human-human interaction generation with adaptive spatiotemporal mamba,” IEEE Transactions on Visualization and Computer Graphics, 2025.

[48] Z. Geng, Z. Hayder, B. Miao, J. Liu, W. Liu, and A. Mian, “Disentangled hierarchical vae for 3d human-human interaction generation,” arXiv preprint arXiv:2603.00144, 2026.

[49] P. Ruiz-Ponce, G. Barquero, C. Palmero, S. Escalera, and J. Garc´ıa-Rodr´ıguez, “in2in: Leveraging individual information to generate human interactions,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2024, pp. 1941–1951.

[50] Y. Shafir, G. Tevet, R. Kapon, and A. H. Bermano, “Human motion diffusion as a generative prior,” arXiv preprint arXiv:2303.01418, 2023.

[51] Y. Wang, S. Wang, J. Zhang, K. Fan, J. Wu, Z. Xue, and Y. Liu, “Timotion: Temporal and interactive framework for efficient humanhuman motion generation,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2025, pp. 7169–7178.

[52] M. G. Javed, C. Guo, L. Cheng, and X. Li, “Intermask: 3d human interaction generation via collaborative masked modeling,” arXiv preprint arXiv:2410.10010, 2024.

[53] S. Liu, C. Guo, B. Zhou, and J. Wang, “Ponimator: Unfolding interactive pose for versatile human-human interaction animation,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2025, pp. 12 068–12 077.

[54] P. Ruiz-Ponce, S. Escalera, J. Garc´ıa-Rodr´ıguez, J. Deng, and R. A. Potamias, “Interact2ar: Full-body human-human interaction generation via autoregressive diffusion models,” arXiv preprint arXiv:2512.19692, 2025.

[55] C. Yu, W. Zhai, Y. Yang, Y. Cao, and Z.-J. Zha, “Hero: Human reaction generation from videos,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2025, pp. 10 262–10 274.

[56] L. Zhang, Z. Li, T. Li, Z. Cao, R. Xu, X. Long, W. Wang, J. Wang, Y. Liu, W. Wang et al., “Egoreact: Egocentric video-driven 3d human reaction generation,” arXiv preprint arXiv:2512.22808, 2025.

[57] C. Luo, B. Wu, B. Li, J. Ren, R. Bai, R. Qu, L. Shen, and B. Ghanem, “Reactmotion: Generating reactive listener motions from speaker utterance,” arXiv preprint arXiv:2603.15083, 2026.

[58] Y. Liu, C. Chen, C. Ding, and L. Yi, “Physreaction: Physically plausible real-time humanoid reaction synthesis via forward dynamics guided 4d imitation,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 3771–3780.

[59] W. Jiang, J. Wang, K. Ji, B. Jia, S. Huang, and Y. Shi, “Arflow: Human action-reaction flow matching with physical guidance,” arXiv preprint arXiv:2503.16973, 2025.

[60] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” arXiv preprint arXiv:2210.02747, 2022.

[61] Y. Liu, C. Chen, and L. Yi, “Interactive humanoid: Online full-body motion reaction synthesis with social affordance canonicalization and forecasting,” arXiv preprint arXiv:2312.08983, 2023.

[62] Z. Cen, H. Pi, S. Peng, Q. Shuai, Y. Shen, H. Bao, X. Zhou, and R. Hu, “Ready-to-react: Online reaction policy for two-character interaction generation,” arXiv preprint arXiv:2502.20370, 2025.

[63] Y. Ye, Y. Xu, Q. Sun, X. Zhu, Y. Sun, and Y. Ma, “Remogen: Real-time human interaction-to-reaction generation via modular learning from diverse data,” arXiv preprint arXiv:2604.01082, 2026.

[64] S. Yan, Y. Xiong, and D. Lin, “Spatial temporal graph convolutional networks for skeleton-based action recognition,” in Proc. Assoc. Advance. Artif. Intell. AAAI Press, 2018, pp. 7444–7452.

[65] L. Shi, Y. Zhang, J. Cheng, and H. Lu, “Two-stream adaptive graph convolutional networks for skeleton-based action recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2019, pp. 12 026–12 035.

[66] Z. Liu, H. Zhang, Z. Chen, Z. Wang, and W. Ouyang, “Disentangling and unifying graph convolutions for skeleton-based action recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2020, pp. 143– 152.

[67] Y. Chen, Z. Zhang, C. Yuan, B. Li, Y. Deng, and W. Hu, “Channelwise topology refinement graph convolution for skeleton-based action recognition,” in Proc. Int. Conf. Comput. Vis., 2021, pp. 13 359–13 368.

[68] J. Lee, M. Lee, D. Lee, and S. Lee, “Hierarchically decomposed graph convolutional networks for skeleton-based action recognition,” in Proc. Int. Conf. Comput. Vis., 2023, pp. 10 444–10 453.

[69] Z. Wang, K. Ying, J. Meng, and J. Ning, “Human-to-human interaction detection,” in International Conference on Neural Information Processing. Springer, 2023, pp. 120–132.

[70] T. Wu, R. He, G. Wu, and L. Wang, “Sportshhi: A dataset for humanhuman interaction detection in sports videos,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2024, pp. 18 537–18 546.

[71] J. Wei, T. Zhou, Y. Yang, and W. Wang, “Nonverbal interaction detection,” in European Conference on Computer Vision. Springer, 2024, pp. 277–295.

[72] L. Li, S. Jia, and J.-N. Hwang, “Multiple human motion understanding,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 8, 2026, pp. 6297–6305.

[73] A. Vinciarelli and G. Mohammadi, “A survey of personality computing,” IEEE Transactions on Affective Computing, vol. 5, no. 3, pp. 273–291, 2014.

[74] C. Wan, L. Wang, and V. V. Phoha, “A survey on gait recognition,” ACM Computing Surveys (CSUR), vol. 51, no. 5, pp. 1–35, 2018.

[75] F. Durupinar, M. Kapadia, S. Deutsch, M. Neff, and N. I. Badler, “Perform: Perceptual approach for adding ocean personality to human motion using laban movement analysis,” ACM Trans. Graph., vol. 36, no. 1, pp. 1–16, 2016.

[76] D. Delgado-Gomez, A. E. Mas´ o-Besga, D. Aguado, V. J. Rubio,´ A. Sujar, and S. Bayona, “Automatic personality assessment through movement analysis,” Sensors, vol. 22, no. 10, p. 3949, 2022.

[77] H.-S. Fang, J. Li, H. Tang, C. Xu, H. Zhu, Y. Xiu, Y.-L. Li, and C. Lu, “Alphapose: Whole-body regional multi-person pose estimation and tracking in real-time,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 6, pp. 7157–7173, 2022.

[78] Q. Shuai, Z. Yu, Z. Zhou, L. Fan, H. Yang, C. Yang, and X. Zhou, “Reconstructing close human interactions from multiple views,” ACM Transactions on Graphics (TOG), vol. 42, no. 6, pp. 1–14, 2023.

[79] F. Lu, Z. Dong, J. Song, and O. Hilliges, “Avatarpose: Avatar-guided 3d pose estimation of close human interaction from sparse multi-view videos,” in European Conference on Computer Vision. Springer, 2024, pp. 215–233.

[80] N. Ugrinovic, B. Pan, G. Pavlakos, D. Paschalidou, B. Shen, J. Sanchez-Riera, F. Moreno-Noguer, and L. Guibas, “Multiphys: Multi-person physics-aware 3d motion estimation,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2024, pp. 2331–2340.

[81] Q. Fang, Y. Fan, Y. Li, J. Dong, D. Wu, W. Zhang, and K. Chen, “Capturing closely interacted two-person motions with reaction priors,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2024, pp. 655– 665.

[82] B. Huang, C. Li, C. Xu, D. Lu, J. Chen, Y. Wang, and G. H. Lee, “Reconstructing close human interaction with appearance and proxemics reasoning,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2025, pp. 17 475–17 485.

[83] M. Petrovich, O. Litany, U. Iqbal, M. J. Black, G. Varol, X. Bin Peng, and D. Rempe, “Multi-track timeline control for text-driven 3d human motion generation,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2024, pp. 1911–1921.

[84] L. Xiao, S. Lu, H. Pi, K. Fan, L. Pan, Y. Zhou, Z. Feng, X. Zhou, S. Peng, and J. Wang, “Motionstreamer: Streaming motion generation via diffusion-based autoregressive model in causal latent space,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2025, pp. 10 086– 10 096.

[85] J. Lu, C.-H. P. Huang, U. Bhattacharya, Q. Huang, and Y. Zhou, “Humoto: A 4d dataset of mocap human object interactions,” arXiv preprint arXiv:2504.10414, 2025.

[86] “Optitrack,” https://optitrack.com/.

[87] “Noitom,” https://noitom.com/.

[88] G. Pavlakos, V. Choutas, N. Ghorbani, T. Bolkart, A. A. Osman, D. Tzionas, and M. J. Black, “Expressive body capture: 3d hands, face, and body from a single image,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2019, pp. 10 975–10 985.

[89] S. Pujades, B. Mohler, A. Thaler, J. Tesch, N. Mahmood, N. Hesse, H. H. Bulthoff, and M. J. Black, “The virtual caliper: Rapid creation of¨ metrically accurate avatars from 3d measurements,” IEEE transactions on visualization and computer graphics, vol. 25, no. 5, pp. 1887–1897, 2019.

[90] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al., “Language models are few-shot learners,” Proc. Adv. Neural Inform. Process. Syst., vol. 33, pp. 1877–1901, 2020.

[91] “Aitviewer,” https://eth-ait.github.io/aitviewer/.

[92] B. Wu, J. Xie, K. Shen, Z. Kong, J. Ren, R. Bai, R. Qu, and L. Shen, “Mg-motionllm: A unified framework for motion comprehension and generation across multiple granularities,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2025, pp. 27 849–27 858.

[93] OpenAI. (2025, Nov.) Gpt-5.1: A smarter, more conversational chatgpt. OpenAI. [Online]. Available: https://openai.com/index/gpt-5-1/

[94] J. S. Wiggins, The five-factor model of personality: Theoretical perspectives. Guilford Press, 1996.

[95] R. R. McCrae and P. T. Costa Jr, “A contemplated revision of the neo five-factor inventory,” Personality and individual differences, vol. 36, no. 3, pp. 587–596, 2004.

[96] Z. Luo, J. Cao, K. Kitani, W. Xu et al., “Perpetual humanoid control for real-time simulated avatars,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2023, pp. 10 895–10 904.

[97] Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li, “On the continuity of rotation representations in neural networks,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Computer Vision Foundation / IEEE, 2019, pp. 5745–5753.

[98] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Proc. Adv. Neural Inform. Process. Syst., vol. 33, pp. 6840– 6851, 2020.

[99] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.

[100] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” in NIPS, 2017, pp. 6626–6637.

[101] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “Bleu: a method for automatic evaluation of machine translation,” in Proceedings of the 40th annual meeting of the Association for Computational Linguistics, 2002, pp. 311–318.

[102] C.-Y. Lin, “Rouge: A package for automatic evaluation of summaries,” in Text summarization branches out, 2004, pp. 74–81.

[103] R. Vedantam, C. Lawrence Zitnick, and D. Parikh, “Cider: Consensusbased image description evaluation,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 4566–4575.

[104] T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, “Bertscore: Evaluating text generation with bert,” arXiv preprint arXiv:1904.09675, 2019.

[105] Y. Du, R. Kips, A. Pumarola, S. Starke, A. Thabet, and A. Sanakoyeu, “Avatars grow legs: Generating smooth human motion from sparse tracking inputs with diffusion model,” arXiv preprint arXiv:2304.08577, 2023.

[106] T. Yamada, H. Matsunaga, and T. Ogata, “Paired recurrent autoencoders for bidirectional translation between robot actions and linguistic descriptions,” IEEE Robotics and Automation Letters, vol. 3, no. 4, pp. 3441–3448, 2018.

[107] M. Plappert, C. Mandery, and T. Asfour, “Learning a bidirectional mapping between human whole-body motion and natural language using deep recurrent neural networks,” Robotics and Autonomous Systems, vol. 109, pp. 13–26, 2018.

[108] Y. Goutsu and T. Inamura, “Linguistic descriptions of human motion with generative adversarial seq2seq learning,” in 2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2021, pp. 4281–4287.

# Inter-X++: A Comprehensive Benchmark for Multimodal Human-Human Interaction Analysis

Supplementary Material

## VIII. DATASET AND ANNOTATION DETAILS

All participants signed informed consent forms authorizing data collection and academic use of the dataset. The released records are de-identified and contain no direct identifiers. To construct the definition of action categories of Inter-X++, we first manually collected candidate categories by referring to established human-human interaction datasets [7], [10], [23] and then leveraged large language models [90] to assist in expanding and consolidating the candidate set. The candidates were subsequently subjected to manual screening, semantic deduplication, and feasibility verification in the motion-capture environment. This process resulted in the 40 common daily human-human interaction categories listed in Tab. XIV. We focus specifically on direct interactions between two people and exclude object-mediated interactions in which the participants interact primarily through a manipulated object.

For the textual annotations, we developed an annotation tool based on [91], allowing the rendered interaction sequences to be inspected from freely adjustable viewpoints. Each interaction sequence was independently annotated by three annotators, who were instructed to describe the coarse full-body movements, fine-grained hand and finger articulations, relative spatial configurations, and participant-specific behaviors. GPT-3.5 [90] was subsequently used only to refine linguistic expression and correct typographical errors. We then conducted extensive manual screening to verify motion-text consistency, participant roles, temporal order, and the accuracy of fine-grained body-part descriptions; ambiguous or lowquality annotations were revised or removed.

## IX. BROADER IMPACTS

Inter-X++ can facilitate both generative and perceptive HHI research, with potential applications in digital-human creation, AR/VR, interactive gaming, animation, and humanrobot interaction. Its multimodal annotations and standardized evaluation protocols may also improve the reproducibility and comparability of HHI methods. Meanwhile, motion patterns, personality traits, and interpersonal relationships may contain sensitive information, and related models could be misused for intrusive surveillance or behavioral profiling. Since the dataset is collected in a controlled environment and may contain demographic or cultural biases, downstream users should follow appropriate privacy and licensing requirements, evaluate generalization carefully, and conduct additional safety validation before deployment in real-world or embodied systems.

## X. SMPL-X OPTIMIZATION DETAILS

We represent each N-frame motion sequence using the SMPL-X [88] parametric model. Specifically, the sequence is parameterized by body pose parameters $\boldsymbol { \theta } \in \mathrm { \overline { { \mathbb { R } } } ^ { N \times 5 5 \times \mathrm { \bar { 3 } } } }$ , global translations $t \in \mathbb { R } ^ { N \times 3 }$ , and shape parameters $\beta \in \mathbb { R } ^ { \overline { { N } } \times 1 0 }$

We initialize the shape parameters of each subject from the corresponding height and weight, following [89]. To accurately recover whole-body motion while preserving fine-grained hand articulation, we then fit SMPL-X to the captured MoCap skeletons using a two-stage optimization procedure.

In the first stage, we optimize the body pose parameters while excluding the finger poses. The joint fitting term aligns the SMPL-X joints with the corresponding joints of the captured skeleton:

$$
\mathbb { E } _ { j } = \frac { 1 } { N } \sum _ { i = 0 } ^ { N - 1 } \sum _ { j \in \mathcal { I } } \left\| \pmb { J } _ { j } ^ { i } \big ( \mathbb { M } ( \theta _ { b } , t ) \big ) - \pmb { g } _ { j } ^ { i } \right\| _ { 2 } ^ { 2 } ,\tag{17}
$$

where $\mathcal { I }$ denotes the joint set, M is the SMPL-X parametric model, $J _ { j } ^ { i }$ denotes the joint regressor for joint j at frame $i ,$ $\theta _ { b }$ represents the pose parameters excluding the fingers, and g denotes the skeleton captured by the MoCap system. To suppress frame-to-frame jitter and improve temporal coherence, we further introduce the following smoothing term:

$$
\mathbb { E } _ { s m o o t h } = \frac { 1 } { N - 1 } \sum _ { i = 0 } ^ { N - 1 } \sum _ { j \in \mathcal { I } } \| \pmb { J } _ { j } ^ { i + 1 } - \pmb { J } _ { j } ^ { i } \| _ { 2 } ^ { 2 }\tag{18}
$$

In addition, we apply a pose regularization term to discourage the optimized body poses from deviating excessively:

$$
\mathbb { E } _ { r } = \lVert \theta _ { b } \rVert _ { 2 } ^ { 2 }\tag{19}
$$

The complete objective for the first stage is therefore defined as:

$$
\mathbb { E } _ { 1 } = \lambda _ { j } \mathbb { E } _ { j } + \lambda _ { s m o o t h } \mathbb { E } _ { s m o o t h } + \lambda _ { r } \mathbb { E } _ { r } ,\tag{20}
$$

where we set $\lambda _ { j } , \lambda _ { s m o o t h } , \lambda _ { r } = 1 , 0 . 1 , 0 . 0 1 ,$

In the second stage, we introduce the finger pose parameters and jointly refine the complete whole-body pose. Because accurate hand articulation is particularly important for close human-human interactions, we separate the hand-related terms from the remaining body terms and assign them different optimization weights. The second-stage objective is formulated as:

$$
\mathbb { E } _ { b } = \lambda _ { j } \mathbb { E } _ { j } + \lambda _ { s m o o t h } \mathbb { E } _ { s m o o t h } + \lambda _ { r } \mathbb { E } _ { r } ,\tag{21}
$$

$$
\mathbb { E } _ { h } = \lambda _ { j _ { h } } \mathbb { E } _ { j _ { h } } + \lambda _ { s m o o t h _ { h } } \mathbb { E } _ { s m o o t h _ { h } } + \lambda _ { r _ { h } } \mathbb { E } _ { r _ { h } } ,\tag{22}
$$

$$
\mathbb { E } _ { 2 } = \mathbb { E } _ { b } + \mathbb { E } _ { h } ,\tag{23}
$$

where $\mathbb { E } _ { b }$ and $\mathbb { E } _ { h }$ denote the body and hand objectives, respectively, and $\mathbb { E } _ { 2 }$ is their combination. We set $\lambda _ { j } , \lambda _ { s m o o t h } , \lambda _ { r } =$ 1, 0.1, 0.01 for the body part and $\begin{array} { r l } { \lambda _ { j _ { h } } , \lambda _ { s m o o t h _ { h } } , \lambda _ { r _ { h } } } & { { } = } \end{array}$ 10, 0.01, 0.001 for fingers.

## XI. DETAILS OF EVALUATION METRICS

## A. Metrics for Interaction Generation

Following prior human motion generation benchmarks [7], [34], [40]–[42], we evaluate generated interactions in taskspecific feature spaces. For text-conditioned generation, a motion encoder and a text encoder are trained with a contrastive objective to map paired motions and descriptions into a shared latent space. For action-conditioned interaction generation, human reaction generation, and stylized interaction generation, we use a pretrained ST-GCN [64] evaluator to extract motion features and predict action labels.

TABLE XIV  
THE 40 COMMON DAILY HUMAN-HUMAN INTERACTION CATEGORIES INCLUDED IN THE INTER-X++ DATASET.
<table><tr><td>A01: Hug</td><td>A02: Handshake</td><td>A03: Wave</td><td>A04: Grab</td></tr><tr><td>A05: Hit</td><td>A06: Kick</td><td>A07: Posing</td><td>A08: Push</td></tr><tr><td>A09: Pull</td><td>A10: Sit on leg</td><td>A11: Slap</td><td>A12: Pat on back</td></tr><tr><td>A13: Point finger at</td><td>A14: Walk towards</td><td>A15: Knock over</td><td>A16: Step on foot</td></tr><tr><td>A17: High-five</td><td>A18: Chase</td><td>A19: Whisper in ear</td><td>A20: Support with hand</td></tr><tr><td>A21: Finger-guessing</td><td>A22: Dance</td><td>A23: Link arms</td><td>A24: Shoulder to shoulder</td></tr><tr><td>A25: Bend</td><td>A26: Carry on back</td><td>A27: Massage shoulder</td><td>A28: Massage leg</td></tr><tr><td>A29: Hand wrestling</td><td>A30: Chat</td><td>A31: Pat on cheek</td><td>A32: Thumb up</td></tr><tr><td>A33: Touch head</td><td>A34: Imitate</td><td>A35: Kiss on cheek</td><td>A36: Help up</td></tr><tr><td>A37: Cover mouth</td><td>A38: Look back</td><td>A39: Block</td><td>A40: Fly kiss</td></tr></table>

Frechet Inception Distance.´ FID [100] measures the distributional discrepancy between real and generated interaction features. A lower FID indicates that the generated interactions more closely match the real-data distribution.

R-Precision. R-Precision evaluates motion-text alignment. For each generated interaction, its ground-truth description is mixed with 31 randomly selected mismatched descriptions. All candidate descriptions are ranked according to their distances from the generated motion in the shared embedding space. Top-k R-Precision reports the probability that the groundtruth description appears among the top k retrieved candidates, where we report Top-1, Top-2, and Top-3 results. Higher values indicate better semantic consistency between the generated interaction and the text condition.

MultiModal Distance (MM Dist). MM Dist is the average Euclidean distance between the embeddings of generated interactions and their paired textual descriptions. It directly measures cross-modal alignment in the shared motion-text space; a lower value indicates that the generated motions better match the input descriptions.

Action Recognition Accuracy (Acc.). For action-conditioned generation tasks, we use the pretrained ST-GCN evaluator to predict the action category of each generated interaction. Accuracy measures the proportion of predictions that match the conditioning action labels, with a higher value indicating better action consistency.

Diversity. Diversity measures the overall variation of generated interactions across all conditions. Given two independently sampled sets of motion features $\{ v _ { i } \} _ { i = } ^ { S _ { d } }$ and $\{ v _ { i } ^ { \prime } \} _ { i = 1 } ^ { { S _ { d } } } ,$ it is computed as

$$
\mathrm { D i v e r s i t y } = \frac { 1 } { S _ { d } } \sum _ { i = 1 } ^ { S _ { d } } \| v _ { i } - v _ { i } ^ { \prime } \| _ { 2 } .\tag{24}
$$

We compute Diversity for the real motions in the same manner and use the resulting value as a reference. For results marked with →, better performance is indicated by a generated-motion Diversity value closer to the real-motion reference.

Multimodality (MModality). MModality quantifies the intracondition variability of generated interactions. Given C conditions, we independently sample two sets of $S _ { l }$ motions under each condition and compute

$$
\mathrm { M M o d a l i t y } = \frac { 1 } { C S _ { l } } \sum _ { c = 1 } ^ { C } \sum _ { i = 1 } ^ { S _ { l } } \| v _ { c , i } - v _ { c , i } ^ { \prime } \| _ { 2 } .\tag{25}
$$

Here, $v _ { c , i }$ and $v _ { c , i } ^ { \prime }$ denote the feature representations of the paired motion samples generated under condition c. A higher MModality value indicates greater variability among interactions generated from the same condition. For results marked with →, performance is assessed relative to the corresponding real-motion reference, with closer values indicating better agreement.

## B. Metrics for Interaction Imitation

Following PHC [96], we evaluate physics-based interaction imitation using one success metric and four tracking-error metrics. Success rate (Succ) is the percentage of reference sequences that the simulated humanoids track to completion without triggering the standard failure or early-termination criterion; a higher value is better. Global MPJPE $( E _ { \mathrm { g - m p j p e } } )$ measures the mean Euclidean distance between reference and simulated joint positions in the global coordinate system and therefore includes global translation errors. MPJPE $( E _ { \mathrm { m p j p e } } )$ computes the joint-position error after removing the global root translation, focusing on relative pose accuracy. Acceleration error $( E _ { \mathrm { a c c } } )$ and velocity error $\left( E _ { \mathrm { v e l } } \right)$ measure the average discrepancies between the reference and simulated joint accelerations and velocities, respectively. Lower values for all four error metrics indicate more accurate and temporally coherent imitation.

## C. Metrics for Perceptive Tasks

Interaction Recognition and Causal-Order Accuracy. For HHI recognition, Top-1 and Top-5 accuracy measure whether the ground-truth interaction category is the highest-scoring prediction or is included among the five highest-scoring predictions, respectively. For causal interaction order inference, binary accuracy is the proportion of sequences for which the actor-reactor order is correctly identified. Higher values indicate better recognition performance.

Captioning Metrics. We evaluate motion-to-text interaction captioning using complementary lexical and semantic metrics. BLEU@1 and BLEU@4 [101] measure unigram and up-tofour-gram precision, together with a brevity penalty, between generated and reference descriptions. ROUGE [102] emphasizes recall-oriented lexical overlap and measures how much salient reference content is preserved. CIDEr [103] computes consensus agreement using TF–IDF-weighted n-grams across the reference captions. BERTScore [104] matches contextualized token embeddings and is therefore more sensitive to semantic similarity than exact word overlap. Higher values are preferred for all captioning metrics.

Personality Assessment. We report the coefficient of determination $( R ^ { 2 } )$ independently for Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism. Given groundtruth scores $y _ { i } ,$ predictions ${ \hat { y } } _ { i } ,$ and the ground-truth mean ${ \bar { y } } ,$ it is defined as

$$
R ^ { 2 } = 1 - \frac { \sum _ { i } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } } { \sum _ { i } ( y _ { i } - \bar { y } ) ^ { 2 } } .\tag{26}
$$

An $R ^ { 2 }$ of 1 indicates perfect prediction, 0 corresponds to predicting the mean, and a negative value indicates performance worse than the mean predictor.

## D. Repeated Evaluation and Result Interpretation

Unless otherwise specified, stochastic generation evaluations are repeated 20 times with different random seeds, and we report the mean together with the 95% confidence interval. MModality for text-conditioned generation is evaluated five times because each evaluation requires multiple samples for every text condition. In the result tables, ↑ and ↓ indicate that higher and lower values are preferred, respectively, whereas → indicates that proximity to the real-data statistic is preferred.

## XII. ADDITIONAL IMPLEMENTATION DETAILS

a) Data representation and preprocessing: We use the official Inter-X++ training, validation, and test splits, with all motion sequences represented at 30 fps. Each two-person interaction of length $T$ is stored as a tensor in R<sup>T</sup> <sup>×56×12</sup>. After separating the two actors, each frame is represented by a 56×6 feature matrix consisting of the continuous 6D rotations of 55 SMPL-X joints and one root-trajectory entry. The global coordinate system is retained to preserve the relative spatial configuration between the participants.

For Stage 1, training samples are extracted as 64-frame windows with a temporal stride of 10 frames. For Stage 2, sequences shorter than 24 frames are excluded, motion lengths are quantized in units of four frames, and sequences are cropped and padded or truncated to a maximum length of 150 frames. The original Inter-X++ captions are used, with a maximum length of 35 tokens.

## A. Stage 1: Caption-Aware Unified Motion Tokenizer

a) Tokenizer architecture: The first stage employs a twodimensional convolutional residual VQ-VAE with parameters shared between the two actors. Two strided encoder blocks reduce the temporal resolution by a factor of four and aggregate the 56 skeletal entries into five spatial groups. The encoder– decoder width and code dimension are set to 512. Vector quantization uses a codebook of 1,024 embeddings updated by an exponential moving average with a decay rate of 0.99.

We introduce two-layer pre-normalization Transformers immediately before and after vector quantization. Both modules use 512-dimensional features, eight attention heads, a feedforward network width of 2,048, GELU activations, and a dropout rate of 0.1. Their residual branches are zero-initialized to preserve the initial behavior of the VQ-VAE.

b) Caption-aware objective: The latent features of the two actors are concatenated and mean-pooled to condition a single-layer GRU caption decoder with a hidden dimension of 512. The decoder is trained with teacher forcing and tokenlevel cross entropy. The complete Stage 1 objective is

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { V Q } } = \mathcal { L } _ { \mathrm { r e c } } + 0 . 0 2 \mathcal { L } _ { \mathrm { c o m m i t } } + \mathcal { L } _ { \mathrm { p o s } } } \\ & { ~ + 1 0 0 \mathcal { L } _ { \mathrm { v e l } } + 5 \mathcal { L } _ { \mathrm { b o n e } } + 0 . 0 1 \mathcal { L } _ { \mathrm { g e o } } } \\ & { ~ + 5 0 0 \mathcal { L } _ { \mathrm { f o o t } } + 0 . 5 \mathcal { L } _ { \mathrm { c a p } } . } \end{array}\tag{27}
$$

Here, the reconstruction and commitment terms supervise motion tokenization; the position, velocity, bone-length, geodesic, and foot-contact terms enforce kinematic consistency; and $\mathcal { L } _ { \mathrm { c a p } }$ provides interaction-level semantic supervision.

c) Optimization: The tokenizer is optimized using AdamW with a learning rate of $2 \times 1 0 ^ { - 4 } , \ : \beta = ( 0 . 9 , 0 . 9 9 )$ and zero weight decay. Gradients are clipped to a maximum norm of 1.0. Training is conducted for 50 epochs with a batch size of 128.

## B. Stage 2: Text-Conditioned Masked Interaction Transformer

Our text-conditioned interaction generator builds upon InterMask [52] and is adapted to the unified motion tokenizer. The Stage 1 tokenizer remains fixed during this stage. The two actor-specific token sequences are concatenated using a learned separation token. The vocabulary consists of 1,024 motion codes and three special symbols for masking, padding, and actor separation. Each 512-dimensional code embedding is projected into a 384-dimensional Transformer space and augmented with spatial and temporal positional encodings.

a) Interaction Transformer and text conditioning: The masked interaction Transformer comprises six blocks, each with six attention heads, a feed-forward network width of 1,024, and a dropout rate of 0.2. The two actors are modeled jointly through self-attention and cross-actor attention. Textual conditions are encoded using a frozen CLIP ViT-L/14@336px backbone. A trainable two-layer text Transformer with a width of 768, eight attention heads, and a feed-forward network width of 2,048 projects the text representation into the motion-Transformer space. The text condition is dropped with a probability of 0.1 during training to enable classifier-free guidance.

b) Training: The masking ratio follows a cosine schedule, and cross-entropy loss is evaluated at the selected token positions. Interaction masking is applied with a probability of 0.2 to encourage the model to infer one participant from the motion of the other. The masked interaction Transformer is trained from random initialization using AdamW with a learning rate of $2 \times 1 0 ^ { - 4 } , \beta = ( 0 . 9 , 0 . 9 9 )$ , and a weight decay of $1 0 ^ { - 5 }$ . Training is conducted for 500 epochs with a batch size of 128.

## C. Inference and Evaluation

a) Sampling: Generation begins from fully masked twoactor token sequences. Tokens are iteratively sampled, and low-confidence predictions are re-masked according to a cosine schedule. Unless otherwise specified, we use 20 refinement steps, categorical sampling with a temperature of 1.0, and a classifier-free guidance scale of 2. The final actor-specific token sequences are decoded independently by the shared VQ decoder.

b) Evaluation protocol: We evaluate on the Inter-X++ test split with a batch size of 32 using an independently trained and frozen text–motion evaluator. Text inputs are represented using 300-dimensional GloVe embeddings and part-of-speech features, while text and motion are mapped into a shared 512-dimensional embedding space. We report Matching Score, R-Precision at ranks 1–3, FID, Diversity, and Multimodality. Following the protocol described above, stochastic evaluations are repeated 20 times and reported with 95% confidence intervals, whereas Multimodality is evaluated five times. Diversity is computed using 300 randomly sampled motion pairs. For Multimodality, we sample 100 captions, generate 30 motions for each caption, and perform 10 random pairwise comparisons.

c) Software: The implementation uses Python 3.9.13, PyTorch 1.13.1, and CUDA 11.7. Training is performed using a single-process, single-GPU pipeline.

## XIII. ADDITIONAL QUALITATIVE RESULTS ON INTER-X++

This section presents additional qualitative results for the hierarchical textual annotations, text-conditioned human-human interaction generation, and motion-to-text interaction captioning on Inter-X++. As illustrated in Fig. 7, the three annotation levels exhibit progressively increasing semantic granularity, ranging from concise action summaries to detailed descriptions of participant-specific movements, body-part dynamics, and spatial relations. The examples in Fig. 8 demonstrate how textual conditions govern the actions, relative configurations, and coordinated motions of the two participants, whereas Fig. 9 highlights representative failure modes. Moreover, Fig. 10 compares the captions predicted by OpenHHI with the corresponding ground-truth descriptions, providing a qualitative assessment of motion–language semantic correspondence.

![](images/4082e17124cc9bd661a15d005ab2aecd994782ff6a57bf04e7f85a3d02d6a17b.jpg)  
Fig. 7. Examples of the hierarchical textual annotations in Inter-X++. For each interaction, Level-1 gives a concise action summary, Level-2 describes the overall interaction in greater detail, and Level-3 additionally specifies participant-specific movements, body parts, and spatial relations.

Level-1: One of them pulls another forward, they fall, and get supported   
Level-2: One of them pulls the other forward by the hand, causing them to fall, and then supports them with both hands   
Level-3: One individual grasps the right hand of the other individual using his/her right hand and proceeds to draw him/her   
ahead, resulting in the second individual toppling onto the first individual. Subsequently, the first individual provides   
assistance to the second individual using both hands Level-1: They stand together, one raises both hands, the other a scissor pose   
Level-2: They stand side by side, one raising both hands while the other raises a hand in a scissor-hand pose   
Level-3: The two people stand side by side, one person raises both hands, and the other person raises his/her right hand in a scissor-hand pose. Level-1: One individual approaches another and gives them a hug   
Level-2: One individual approaches and hugs the other from the left side   
Level-3: One person hugs the other person from the left side, while the other person continues to stamp his/her feet and raises his/her left hand. Level-2: One man squats down, hugs the other's thighs, and carries him on his back while the other wraps his arms around his shoulders.   
Level-3: Both men stand. One person walks up to the other, squats down, hugs the other person's thighs with both hands, and carries him on his back. The other man puts his arms around his shoulders. Level-1: The first guy pulls the second guy towards him.   
Level-2: One of them pulls the other to the left by holding their left hand.   
Level-3: The two men stand facing each other, one holds the other's left arm with both hands, pulling him back, and the other is pulled for a few steps

![](images/de50a129d18e3206473874f63957676a70fd98627cb265b7678dcd05fc76df42.jpg)  
"Two people sit face to face, placing their right elbows on the table to compete in arm wrestling. One person quickly wins."

![](images/e39a4107cf2e717f0698cb1c31e661d138ac1e34feb18a1719ca6e4839cf72e8.jpg)

![](images/aaa5e941f9be81eee26f2a4a159d6c634961a3fdb0a066df59c8278bb945f3aa.jpg)  
"The first guy steps on the second guy with his right foot, and the second guy pulls his leg back."  
"Two people sit down to play rock, paper, scissors. One person uses his/her left hand to make a rock sign, then uses his/her left hand to make a rock sign again, and finally uses his/her right hand to make a rock sign. The other person uses his/her right hand to make a rock sign, and then uses his/her right hand to make a paper sign gesture."

![](images/9deb1738b0b863ee09ce6828532dceb29efda4c18417a0840561383181105865.jpg)

![](images/c83aef366e90de64ab8ac07898bbf29ae4ecef46ca756f2a845eecd9794eea28.jpg)  
"Two people stand side by side, comparing thumbs by extending their right hands. One person extends his/her right hand, while the other person responds with a clenched fist."

![](images/eb6bf2c598aff4ccc25e37da03b2a38300cb340b33de0208103e8f5ff42a10b3.jpg)  
"One person hugs the other person's shoulders from behind, and the other person places his/her hands on top of the person's hands."  
"The first person stands in front of the second person, raises his/her hands to his/her chest, points his/her fingers at the second person, and the second person raises his/her hands and puts them on his/her chest."

![](images/e60a8f8f10a7a00297eb50674e0d1bfd23316d75ffef86287e1d41aa2176ec94.jpg)  
"The two people face each other and step forward, giving each other three high-fives with both hands."

![](images/6dfd1c40b29545c3957b6ebafacb1003427db75df6430d277e2c7c928fd6c8ac.jpg)  
"One person walks with his/her arms down, while suddenly the other person, who was sitting, stands up and starts chasing the first person. The first person starts running in a small circle, and the second person follows. raising his/her arms while chasing. Eventually, both of them stop."

Fig. 8. Additional qualitative results of text-conditioned human-human interaction generation on Inter-X++. Each example shows a generated interaction sequence together with its input description.  
![](images/c2d5ad684392dd913af9201ba65a7b60091bf89adaa78c72813a323ac749921e.jpg)  
"Two people stand facing each other, with one person extending his/her right foot and forcefully stepping on the other person's right foot.'

![](images/e85ce665449c8c59f0cf8d05575df98652c9dec3113dd78e1eb5ab23b8b8f7b2.jpg)

![](images/ba37a1c022c50621b7b6267aac56adc7de4201c627501e6e36848c87c7ef03ed.jpg)  
"The first person runs towards the second person and bumps his/her left arm with his/her own right elbow."  
"A single individual moves ahead and, upon reaching the right side of another individual, the second person stretches out both hands to obstruct his/her way. Subsequently, the initial person shifts to the left and proceeds with his/her forward movement.'

![](images/d8376896e1cd61a250bacd460865829d0271d14a90afe9e53156123362d651ec.jpg)  
"One person grabs the left hand of the other person who is crouching with both hands and pulls him/her forward. causing him/her to take a few steps forward."

Fig. 9. Representative failure cases of text-conditioned human-human interaction generation. The top row highlights frames with inaccurate or physically implausible contacts using red boxes, while the bottom row illustrates inaccurate grounding of spatial and body-part constraints, which are emphasized in red in the input descriptions.

<table><tr><td rowspan=2 colspan=2>GT: Two people stand facing each other, and the first person raises his/her right hand to touch the second person&#x27;s head.</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>OpenHHI: The first person crouches on the ground while the second person stand behind him, place his hand on the firstperson&#x27;s shoulder. Then the first person stands up and carries the second person.</td><td rowspan=1 colspan=1>OpenHHI: Two people stand face each other and the first person raise his right hand to touch the head of the second person.</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>GT: Two individuals are sitting and they raise their right hands to twist each other&#x27;s wrists, with one person overpoweringthe other.</td><td rowspan=1 colspan=1>GT: One person sits on a chair while the other person massages and pats his/her shoulders from behind. Eventually, the personbeing massaged stands up.</td></tr><tr><td rowspan=1 colspan=1>OpenHHI: Two people sit face each other extend their right hand and start arm wrestling. After a period of struggle oneperson emerges as the winner.</td><td rowspan=1 colspan=1>the first person&#x27;s right shoulder then clenches his fist and lightly taps him</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>GT: One person pulls the right arm of the other person with his/her left hand, causing the other person to be pulled alongand take a few steps.</td><td rowspan=1 colspan=1>GT: A standing person uses both hands to massage              , while the other person lightly pats the firstperson&#x27;s right shoulder with their left hand.</td></tr><tr><td rowspan=1 colspan=1>OpenHHI: One person pulls the other person&#x27;s right arm with his left hand and they walk forward together.</td><td rowspan=1 colspan=1>OpenHHI: One person stands while the other person sits on a chair. The stand person extends both hand bend down andsoftly taps the sit person&#x27;s right leg.</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>GT: A person approaches another person and slowly slaps his/her upper back with his/her right hand. The person beingslapped raises his/her hands and embraces the person who slapped him/her.</td><td rowspan=1 colspan=1>GT: Two people stand face to face. One person crouches down and slaps his/her legs, prompting the other person to bend over.</td></tr><tr><td rowspan=1 colspan=1>OpenHHI: One person walks up to the other person and hugs him. The other person reciprocates the gesture by also raiseshis right hand.</td><td rowspan=1 colspan=1>OpenHHI: One person crouches down and the other person bends down to massage his leg.</td></tr></table>

Fig. 10. Additional qualitative results of human-human interaction captioning on Inter-X++. Each example presents a sampled interaction sequence together with its ground-truth description and the caption generated by OpenHHI.