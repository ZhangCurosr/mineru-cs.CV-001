# ForeSightGuide: An Anticipatory Framework toward Accurate and Low-Redundancy Guidance for the Visually Impaired

Zhiyuan Wang<sup>1</sup>, Xu Li<sup>1</sup>, Shikang Guo<sup>1</sup>, Wei Meng<sup>1</sup>, Quan Liu<sup>1</sup>, Jie Zuo<sup>1∗</sup>

<sup>1</sup>School of Information Engineering, Wuhan University of Technology

122 Luoshi Road, Wuhan 430070, China

{wangzy , 357021, guoshikang\_2022, weimeng,quanliu,zuojie}@whut.edu.cn

## Abstract

Electronic travel aids are pivotal for the independent mobility of the visually impaired. While Vision-Language Models (VLMs) ofer rich environmental understanding, they often sufer from excessive false positives in dynamic scenarios, leading to cognitive overload. To address this, we present Fore-SightGuide, an anticipatory assistive guidance framework that couples semantic scene understanding with predictive hazard assessment. Unlike reactive systems, ForeSightGuide leverages the reasoning capabilities of VLMs to anticipate obstacle motion, efectively filtering out non-threatening objects to provide concise, actionable guidance. To validate our approach, we introduce a novel dataset captured in complex, dynamic real-world trafic scenes, designed to benchmark predictive capabilities. Extensive experiments on both public benchmarks and our proposed dataset demonstrate that ForeSightGuide achieves state-of-the-art performance. Notably, it significantly mitigates information overload by reducing redundant alerts to 0.299 per guidance output while maintaining a low missedhazard rate of 0.112, proving its eficacy for safe walking assistance.

## INTRODUCTION

According to the World Health Organization (WHO), at least 2.2 billion people worldwide have near or distance vision impairment (Organization 2026). Visual impairment can reduce walking ability and increase mobility-related safety risks, especially in unfamiliar environments with dense pedestrians and obstacles. Therefore, safe and efective assistive guidance is essential for supporting the independent mobility and social participation of blind and visually impaired persons (BVIPs) (Lee et al. 2025). However, in real-world dynamic scenes, BVIPs still face complex risks caused by moving pedestrians, vehicles, temporary obstacles, and delayed hazard perception.

In recent years, the rapid advancement of sensor technology, artificial intelligence, and computer vision has driven remarkable breakthroughs in electronic travel aids (ETAs), which act as pivotal assistive tools by providing environmental guidance and navigation solutions for BVIPs (Hayamizu et al. 2026). The core technologies of ETA systems for environmental perception, positioning, and navigation can be categorized into non-vision-based technologies and visionbased technology (VBT) (Kim 2025). Non-vision-based technologies often rely on pre-deployed dedicated infrastructure, resulting in high costs, poor adaptability to unfamiliar environments, and limited scalability. By contrast, VBT avoids such infrastructure requirements by leveraging wearable or portable visual devices and deep learning algorithms. The evolution from explicit feature extraction to end-to-end models has significantly enhanced ETAs’ environmental perception capabilities (Patel and Parmar 2022; Tapu, Mocanu, and Zaharia 2020; Zhao, Wang, and Li 2026). Owing to its superior environmental adaptability, strong scalability, and comprehensive perceptual capability, VBT has gradually become a dominant technical route in the research and development of modern ETA systems.

Although existing ETAs have made substantial progress, safe walking assistance for BVIPs in real-world dynamic scenes remains far from solved. Unlike static indoor navigation or route-level path planning, daily walking often requires users to deal with moving pedestrians, bicycles, vehicles, temporary obstacles, and trafic rules simultaneously. In such situations, an assistive system should not only identify what objects exist in the scene, but also determine whether these objects will afect the user’s future path, whether they require immediate attention, and how the warning should be expressed. If a system reports every detected object or only reacts to the current frame, it may either miss future risks or overwhelm the user with unnecessary information. Therefore, practical guidance systems need both semantic scene understanding and predictive hazard assessment.

The emergence of Vision-Language Models (VLMs) brings new opportunities for assistive guidance for BVIPs. VLMs can convert complex visual information into naturallanguage descriptions and instructions, helping visually impaired users understand their surrounding environments through accessible verbal feedback. Recent studies have explored VLMs and multimodal language models for assistive navigation, showing their potential in context-aware instruction generation and walking assistance. However, directly applying VLMs to assistive guidance for BVIPs also introduces new risks. Since general-purpose VLMs tend to describe visible content comprehensively,they may produce redundant alerts in dynamic scenes, while recent threat-oriented reasoning work suggests that safety-critical perception should focus on objects that pose real potential risks (He et al. 2026).In Fig. 1,we conducted tests on various models under representative outdoor road conditions.For BVIPs, such outputs can increase cognitive load and interfere with timely decision-making. Moreover, VLMs still have limitations in spatial-geometric reasoning and future motion understanding, which may lead to vague or unsafe guidance when obstacles are moving.

To address the core challenges of guidance VLMs for visually impaired users, this paper proposes ForeSightGuide, a novel hybrid fast-slow response guidance framework that integrates precise perception, fast-response safe path planning, and VLM-based deliberative decision-making. In this framework, the safe path planner serves as a rapid reaction layer for time-critical obstacle avoidance, while the VLM functions as a slower reasoning layer for high-level scene understanding and guidance decisions. Specifically, Fore-SightGuide extracts semantic information and predicts the future motion trends of obstacles to construct a predictive bird’s-eye-view (BEV) map. Based on this map, the system estimates the threat levels of diferent obstacles and selectively focuses on those that may pose future risks. Unlike purely reactive systems, ForeSightGuide emphasizes motion anticipation and filters out non-threatening or irrelevant objects, thereby reducing redundant alerts while maintaining sensitivity to potential hazards. The reliability of the system is validated with real-world environmental data. Our main contributions are as follows:

• ForeSightGuide leverages historical visual observations to predict obstacle motion trends instead of relying only on the current frame. It provides the VLM with a dynamic and positional context prefix, making guidance alerts more accurate.

• Leveraging its predictive capability, ForeSightGuide filters out obstacles that do not pose a real threat through a mask-based mechanism, thereby reducing redundant VLM responses and lowering cognitive load for visually impaired users.

• ForeSightGuide adopts a hybrid fast-slow response strategy. Our proposed predictive APF method provides highfrequency obstacle avoidance, while the VLM generates low-frequency semantic reminders, combining timely physical guidance with concise verbal feedback.

## RELATED WORK

In this section, we review studies related to ForeSightGuide, including electronic travel aids, obstacle-avoidance methods, and VLM-based assistive guidance for visually impaired people.

Electronic Travel Aids. Traditional ETAs include ultrasonic or infrared sensing devices, smart canes, wearable cameras, haptic feedback devices, and infrastructure-based localization systems (Dakopoulos and Bourbakis 2010; Tapu, Mocanu, and Zaharia 2020).Representative systems such as the EyeCane provide distance feedback through auditory or haptic signals (Maidenbaum et al. 2014). NavCog uses smartphone localization and Bluetooth beacons to support indoor guidance (Ahmetovic et al. 2016). Although these systems are useful for obstacle awareness and localization, they are often limited by semantic understanding, or dependence on pre-installed infrastructure.

Obstacle Avoidance and Path Planning. Safe walking assistance for visually impaired people requires reliable route selection and timely obstacle avoidance. Early systems usually relied on graph-search or heuristic planning algorithms to generate safe routes, while recent works further consider semantic maps, user preferences, and dynamic obstacles (Najjar, Al-Issa, and Hosny 2022). For example, Goswami et al. propose a floor-plan-based localization and guidance system that combines semantic SLAM with A\*-based motion command generation (Goswami et al. 2024). Surougi and McCann propose MinD, an optimization-based local pathplanning method for dynamic outdoor scenes that considers critical moving objects and predicted collision time (Surougi and McCann 2023). Na et al. further introduce a Dynamic Walking Corridor Generation method for crowded environments by combining social force models and convex optimization (Na et al. 2025). These methods provide important foundations for safe motion planning.

VLM-Based Assistive Guidance. The emergence of VLMs has brought new opportunities for blind walking assistance. By converting complex visual information into natural-language descriptions, VLMs can help visually impaired users perceive their surrounding environments and receive more informative guidance than traditional ETA systems that mainly provide simple alerts or haptic signals. Early work such as SeeWay demonstrated the promise of visuallanguage fusion for intuitive assistive guidance (Yang et al. 2022).VLM-based systems are better suited to complex semantic scenes because they can explain what is happening rather than only signaling the presence of an obstacle.

Recent studies have further applied multimodal models to visually impaired assistance in real-world urban scenes.Chao et al. propose an automated context-aware support system for individuals with visual impairment in outdoor urban environments (Chao et al. 2025). Their method uses VideoLLaMA3 with accessibility-oriented prompting and post-processing to generate spatially and temporally aware guidance. This work shows the potential of VLMs in real-world urban scenes. It also points out that traditional metrics such as BLEU and ROUGE may fail to evaluate navigation instructions that are semantically similar but lexically diferent.This motivates the use of semantic similarity for evaluating assistive guidance.

WalkVLM is the most relevant prior work to ours because it focuses on VLM-based walking assistance for visually impaired people and introduces the Walking Awareness Dataset (WAD) (Yuan et al. 2025). WAD contains 12,000 videoannotation pairs collected from diverse real-world walking scenes.This makes WAD a valuable benchmark for training and evaluating VLM-based walking assistance.WalkVLM adopts CoT-based hierarchical planning and decomposes the task into perception, comprehension, and decision stages. In the perception stage, it uses Detic, an open-vocabulary object detector, to localize scene objects and filters the detected objects by size and confidence score.

![](images/f00ce898194d6a5eab2ac15bb540ac974f8be56d03f8076848f9ec2a3b693f54.jpg)  
Figure 1: Example of redundant VLM guidance in a dynamic crossing scene. General-purpose VLMs may report distant or non-threatening objects, which increases cognitive load for BVIPs. ForeSightGuide filters objects by motion and risk, and only reports obstacles that may afect the user’s path.

Despite these advances, existing VLM-based assistive guidance still has limitations in dynamic scenes. Many methods rely mainly on current observations and do not explicitly predict whether an obstacle will afect the user in the near future. As a result, guidance may become delayed, or the system may describe non-threatening objects and increase cognitive load. Compared with prior path-planning and VLMbased guidance methods, ForeSightGuide emphasizes predictive hazard assessment and concise decision-making. It uses historical visual observations to predict obstacle motion, constructs a predictive BEV map, and filters irrelevant objects before VLM generation. In this way, ForeSightGuide connects risk estimation with semantic guidance,aiming to provide accurate and eficient assistance for BVIPs.

## Methodology

ForeSightGuide is a guidance framework designed for BVIPs, as shown in Fig. 2. It is built around three stages: environmental perception, safe path planning, and guidance decision-making. Given visual observations, the system combines depth estimation and semantic segmentation to build a predictive bird’s-eye-view (BEV) map. Historical trajectories of dynamic obstacles are used to estimate their future motion, allowing the system to reason about potential risks before they occur. Based on this predictive representation, we further propose a predictive artificial potential field method to generate a safe reference path that accounts for the future positions of dynamic obstacles.

To support both real-time control and high-level guidance, ForeSightGuide operates at two decision frequencies.

The safe path module runs at a higher frequency and continuously provides reference motion for the ETA. During the lower-frequency VLM decision process, the safe path results are used to control the ETA between VLM outputs. The VLM then generates both guidance alerts and navigation instructions, so that the user receives concise semantic feedback while the ETA still provides timely physical assistance. This design further reduces cognitive load by avoiding frequent verbal updates. In our implementation, the ETA is an electronic extra limb with three degrees of freedom, which provides traction assistance for visually impaired users.

## Environmental Perception and Risk Filtering

The environmental perception module provides dynamic scene priors for safe path planning and VLM-based guidance decision-making. For each egocentric image $I _ { t } \in \mathbb { R } ^ { \breve { 3 } \times H \times W }$ we first extract a structured perceptual state:

$$
X _ { t } = \mathcal { P } ( I _ { t } )\tag{1}
$$

where $\mathcal { P } ( \cdot )$ denotes the perception function. Specifically, YOLO11 is used for object detection and instance segmentation, producing object categories, bounding boxes, and instance masks. Depth Pro (Bochkovskiy et al. 2025) is then used to estimate the absolute depth map $\boldsymbol { Z _ { t } } \in \mathbb { R } ^ { H \times \tilde { W } }$ . Based on the camera intrinsic matrix, the RGB-D information is converted into a 3D point cloud and further projected onto a 2D bird’s-eye-view (BEV) grid. The resulting perceptual state X<sub>t</sub> contains the category, mask, BEV position, and distance of each detected object.

At time T, the current egocentric image $I _ { \mathrm { c u r r } }$ is first converted into a structured perceptual state by the perception function $\mathcal { P } ( \cdot )$ . Conditioned on the historical perceptual states, the module then extracts static and dynamic obstacle information:

![](images/28fba202d94fd6d567ad1433a27639a63df2d63a0e110a4af4dd6b7f78a3e963.jpg)  
Figure 2: Overview of ForeSightGuide. The framework predicts obstacle motion, filters non-threatening objects, and provides both high-frequency safe path planning and low-frequency VLM-based guidance for BVIPs.

$$
O _ { \mathrm { s t a t i c } } , T _ { \mathrm { d y n a m i c } } = y ( I _ { \mathrm { c u r r } } \mid X _ { \mathrm { h i s t } } ) ,\tag{2}
$$

where $X _ { \mathrm { h i s t } } ~ = ~ \{ X _ { t } \} _ { t = T - L } ^ { T - 1 }$ denotes the structured perceptual states from the previous L frames. The temporal reasoning function $y ( \cdot )$ performs object association across frames, separates static and dynamic obstacles, and predicts future motion for dynamic obstacles. The outputs $O _ { \mathrm { s t a t i c } }$ and $T _ { \mathrm { d y n a m i c } }$ denote the static obstacle set and dynamic obstacle set, respectively. The static obstacle set is defined as

$$
O _ { \mathrm { s t a t i c } } = \{ P _ { s } , c _ { s } \} ,\tag{3}
$$

where $P _ { s } = ( x _ { s } , y _ { s } )$ and $c _ { s }$ denote the current BEV position and category of static obstacle $s .$

For each dynamic obstacle $d ,$ let $\{ P _ { d } ^ { t } \} _ { t = T - L } ^ { T }$ denote its observed BEV trajectory. A Kalman filter is used to predict future positions and velocities:

$$
\{ \hat { P } _ { d } ^ { t ^ { \prime } } , \hat { V } _ { d } ^ { t ^ { \prime } } \} _ { t ^ { \prime } = T + 1 } ^ { T + N } = f _ { \mathrm { k f } } ( P _ { d } ^ { T - L : T } ) ,\tag{4}
$$

where $\hat { P } _ { d } ^ { t ^ { \prime } }$ and $\hat { V } _ { d } ^ { t ^ { \prime } }$ denote the predicted position and velocity at future time $t ^ { \prime } ,$ , and N is the prediction horizon. The dynamic obstacle set is then defined as:

$$
T _ { \mathrm { d y n a m i c } } = \left\{ P _ { d } , V _ { d } , \hat { \mathbf { P } } _ { d } ^ { T + 1 : T + N } , \hat { \mathbf { V } } _ { d } ^ { T + 1 : T + N } , c _ { d } \right\} ,\tag{5}
$$

where $P _ { d } , V _ { d } ,$ , and $c _ { d }$ denote the current position, velocity, and category of dynamic obstacle d.

By integrating static obstacles, dynamic obstacles, the module constructs a predictive BEV map:

$$
M _ { \mathrm { p r e d } } = \left\{ O _ { \mathrm { s t a t i c } } , T _ { \mathrm { d y n a m i c } } \right\}\tag{6}
$$

Based on the predictive BEV map, we filter the perceived objects before VLM decision-making. The filtering is performed according to the obstacle distance, future position and velocity direction. Static obstacles are retained only if they fall within the predefined hazard range. Dynamic obstacles are retained only if their predicted trajectories enter the user’s safety range or move toward the user’s walking direction. Objects that remain outside the safety range or move away from the user are treated as non-threatening.

The filtering process is formulated as

$$
I _ { \mathrm { m a s k } } , P _ { \mathrm { p r e f i x } } = \mathcal { F } ( M _ { \mathrm { p r e d } } , I _ { \mathrm { c u r r } } ) ,\tag{7}
$$

where $\mathcal F ( \cdot )$ denotes the object filtering function. Given the predictive BEV map $M _ { \mathrm { p r e d } }$ , the function first identifies the objects that may afect the user. For objects judged as nonthreatening, their instance masks obtained from YOLO11 segmentation are applied to the current image $I _ { \mathrm { c u r r } } ,$ producing the masked image $I _ { \mathrm { m a s k } }$ . For the retained objects, their category, current distance, motion trend, and predicted future position are organized into the perceptual information prefix $P _ { \mathrm { p r e f i x } }$

The retained objects are further ranked by urgency before being sent to the VLM. This process removes irrelevant visual content from the image and provides structured dynamic context, helping the VLM generate more focused guidance.

## Safe Path Planning

Existing obstacle avoidance methods for visually impaired users often rely on the current scene, making them less reliable in dynamic environments.We design a predictive artificial potential field (APF) method that uses the predicted trajectories of dynamic obstacles to generate safe path. The front segment of this path is converted into directional commands, allowing the electronic extra limb to provide physical guidance.

As shown in Algorithm 1, the planner uses a default forward attractive force to provide a basic forward tendency. The attractive gain ξ controls the balance between forward motion and obstacle avoidance: a larger ξ reduces the relative influence of repulsive forces and results in a path with less lateral deviation. Static obstacles and predicted dynamic obstacles generate repulsive forces to push the path away from potential hazards.

Algorithm 1: Predictive Artificial Potential Field Path Plan  
ning   
Input: Filtered static obstacle set $O _ { \mathrm { s t a t i c } } ^ { * }$ , filtered dynamic   
obstacle set $T _ { \mathrm { d y n a m i c } } ^ { * }$   
Parameter: Step length s, number of candidate repulsion   
sources $n ,$ attractive gain ξ   
Output:Planed safe path $Y _ { \mathrm { p l a n } }$   
1: Let $k = 0 .$   
2: Set default forward attractive force $\mathbf { F } _ { \mathrm { a t t } } = \xi \mathbf { e } _ { \mathrm { f } } .$   
3: while $k < M$ do   
4: Set $\mathbf { F } _ { \mathrm { S } } ( \mathbf { q } _ { k } ) = 0 , \mathbf { F } _ { \mathrm { D } } ( \mathbf { q } _ { k } ) = 0 .$   
5: for all $s \in O _ { \mathrm { s t a t i c } } ^ { * }$ do   
6: Compute ${ \bf F } _ { \mathrm { s } } ( { \bf q } _ { k } )$ via Eq. 8.   
7: $\mathbf { F } _ { \mathrm { S } } ( { \bar { \mathbf { q } } } _ { k } ) \gets \mathbf { F } _ { \mathrm { S } } ( { \bar { \mathbf { q } } } _ { k } ) + { \bar { \mathbf { F } _ { \mathrm { s } } } } ( \mathbf { q } _ { k } )$   
8: end for   
9: for all $d \in T _ { \mathrm { d y n a m i c } } ^ { * }$ do   
10: Compute $\mathbf { F } _ { \mathrm { d } } ( \mathbf { q } _ { k } )$ via Eq. 9.   
11: $\mathbf { F } _ { \mathrm { D } } ( \bar { \mathbf { q } } _ { k } ) \gets \dot { \mathbf { F } _ { \mathrm { D } } } ( \bar { \mathbf { q } } _ { k } ) + \bar { \mathbf { F } _ { \mathrm { d } } } ( \mathbf { q } _ { k } ) .$   
12: end for   
13: Calculate total force: $\mathbf { F } _ { \mathrm { t o t a l } , k } = \mathbf { F } _ { \mathrm { a t t } } + \mathbf { F } _ { \mathrm { S } } ( \mathbf { q } _ { k } ) +$   
$\mathbf { F } _ { \mathrm { D } } ( \mathbf { q } _ { k } )$   
14: Update next path point: $\begin{array} { r } { \mathbf q _ { k + 1 } = \mathbf q _ { k } + s \cdot \frac { \mathbf F _ { \mathrm { t o t a l } , k } } { \| \mathbf F _ { \mathrm { t o t a l } , k } \| } . } \end{array}$   
15: $k \gets k + 1 .$   
16: end while   
17: return $Y _ { \mathrm { p l a n } } = \{ q _ { 0 } , q _ { 1 } , \dots , q _ { M } \} .$

The planner takes the filtered obstacle sets from the perception module as input, including the static obstacle set $O _ { \mathrm { s t a t i c } } ^ { * }$ and the dynamic obstacle set $T _ { \mathrm { d y n a m i c } } ^ { * } .$

Static obstacle Repulsive force is generated based on the position of screened static objects, with the repulsive force increasing as the distance decreases:

$$
\mathbf { F } _ { \mathrm { s } } ( \mathrm { q } _ { k } ) = { \frac { \mathrm { q } _ { k } - \mathrm { p } _ { s } } { \lVert \mathrm { q } _ { k } - \mathrm { p } _ { s } \rVert ^ { 3 } } }\tag{8}
$$

where $\mathrm { p } _ { s }$ is the coordinate on the BEV map of the static object.

The predicted trajectory and speed information are fused to generate repulsive force for approaching dynamic obstacles. $\mathbf { A } \mathbf { s }$ shown in Fig. 3, at each iteration, the dynamic repulsion sources are updated according to the predicted obstacle positions at the corresponding future time step.The formula for each obstacle is as follows:

$$
\mathbf { F } _ { d } ( \mathrm { q } _ { k } ) = { \frac { 1 } { n } } \cdot \sum _ { \tau = k } ^ { k + n } { \frac { \lVert \mathbf { V } _ { \tau , d } ^ { \prime } \rVert } { \lVert \mathbf { q } _ { k } - \mathbf { p } _ { \tau , d } ^ { \prime } \rVert ^ { 3 } } } \cdot ( \mathrm { q } _ { k } - \mathrm { p } _ { \tau , d } ^ { \prime } )\tag{9}
$$

where n represents the number of repulsive sources to consider when calculating repulsion; $\tau$ is the continuous future sequence index starting from current position;p $\% _ { \tau , d } ^ { \prime }$ is the predicted trajectory coordinate of dynamic object d at time $\tau ,$ aligning repulsion sources with obstacle motion states; $\Vert \mathbf { V } _ { \tau , d } ^ { \prime } \Vert$ is the modulus of dynamic object $d \mathrm { { s } }$ predicted speed at time τ. The summation from $\tau = k$ to $k + n$ constructing a dynamic bufer zone, enhancing capability to avoid approaching obstacle.

![](images/49f33f9b31d87708a1c93832e7abec9e70717d9afd737c7df76f66dafcf36bc7.jpg)  
Figure 3: Illustration of the predictive artificial potential field method. The planner uses a default forward attractive force to maintain walking direction, while predicted obstacle positions are treated as candidate repulsion sources to push the reference path away from future hazards.

The APF step size is temporally aligned with the trajectory prediction interval, defined as $s = v \Delta t .$ , where v is the typical walking speed of BVIPs and $\Delta t$ is the frame interval. The next path point is then generated as

$$
\mathrm { q } _ { k + 1 } = \mathrm { q } _ { k } + s \cdot \frac { \mathbf { F } _ { \mathrm { t o t a l } , k } } { \| \mathbf { F } _ { \mathrm { t o t a l } , k } \| }\tag{10}
$$

where $\mathbf { F } _ { \mathrm { t o t a l } , k }$ is the total force at the k-th iteration. After reaching the maximum iteration number, the generated reference waypoints are converted into directed line segments for ETA control, providing physical guidance for BVIPs.

## Guidance VLM

ForeSightGuide uses Qwen2VL-7B as the VLM backbone to generate high-level guidance for visually impaired users. Direct reasoning on raw images often lacks accurate spatial awareness and future risk perception. As a result, the model may focus on visually salient but non-threatening objects and generate redundant alerts.

To address this issue, ForeSightGuide provides the VLM with the user demand $U$ and two perception-derived inputs: the masked image $I _ { \mathrm { m a s k } }$ and the perceptual information prefix $P _ { \mathrm { p r e f i x } }$ . The masked image removes non-threatening visual content using instance masks, while the perceptual information prefix provides structured spatial and dynamic cues for the retained obstacles. These inputs enhance the VLM’s spatial awareness and motion understanding, leading to more accurate and safety-relevant guidance.

The decision generation process is formulated as

$$
O _ { \mathrm { v l m } } = f _ { \mathrm { v l m } } ( U , I _ { \mathrm { m a s k } } , P _ { \mathrm { p r e f i x } } ) ,\tag{11}
$$

where $O _ { \mathrm { v l m } }$ denotes the generated guidance output. The output contains both guidance alerts and navigation instructions, which provide concise warnings and actionable walking suggestions for visually impaired users.

![](images/27bd9e32e89a0b79d0060bafdc4391ff2269a9174d1221d804c0d198d28ab87f.jpg)  
Figure 4: Representative examples of ForeSightGuide in dynamic indoor and outdoor scenes. The upper row shows the generated navigation instructions and guidance alerts, the middle row shows instance-level obstacle perception, and the lower row shows the predictive BEV map with the planned safe path and future obstacle trajectories.

## Experiments

## Experimental Setup

Dataset. We evaluate ForeSightGuide on the Walking Awareness Dataset (WAD) and our self-built dynamic-scene dataset. WAD contains 12,000 real-world video-annotation pairs for blind walking assistance, with 1,007 reminder samples for testing (Yuan et al. 2025). Our self-built dataset contains 140 video-annotation pairs collected from complex indoor and outdoor dynamic scenes. Each sample corresponds to one decision-making moment and includes both Guidance Alerts and Navigation Instructions. Initial annotations were generated with GPT-5.4 using the WAD style and then manually refined to ensure accuracy, consistency and suitability for BVIPs.

Metrics. We adopt two types of metrics to evaluate the generated guidance. First, ROUGE-1, ROUGE-2, and ROUGE-L are used as reference text overlap metrics, mainly for comparison with existing VLM-based walking assistance studies. Second, we introduce a semantic similarity metric based on sentence embeddings, because correct guidance may use diferent wording from the reference annotation. We use MiniLM-L12-v2 (Wang et al. 2020)to encode the generated guidance and ground truth into sentence-level embeddings, and compute their cosine similarity:

$$
S _ { \mathrm { s e m } } = \cos \left( \mathbf { e } _ { \mathrm { g e n } } , \mathbf { e } _ { \mathrm { g t } } \right) ,\tag{12}
$$

where $\mathbf { e } _ { \mathrm { g e n } }$ and $\mathbf { e } _ { \mathrm { g t } }$ denote the embeddings of the generated guidance and the ground truth, respectively. Compared with simple word overlap, this metric better reflects whether the generated guidance is semantically consistent with the reference.

## Experimental Results

ForeSightGuide uses Qwen2VL-7B as its backbone.It detects pedestrians, bikes, and vehicles from first-person images, predicts their future trajectories in BEV space, and generates a safe reference path to avoid potential conflicts. It then uses the masked image and perceptual information prefix to guide the VLM in producing concise guidance alerts and navigation instructions. Representative examples in indoor and outdoor dynamic scenes are shown in Fig. 4.

Comparison with VLM Baselines . We compare Fore-SightGuide with general VLMs and WAD-fine-tuned VLMs to evaluate the efectiveness of the proposed perceptionguided decision strategy. All fine-tuned models are trained only on the WAD training set, and then evaluated on two test sets: the WAD test set with 1,007 reminder samples and our self-built test set with 140 decision samples. ForeSight-Guide uses the open-source Qwen2VL-7B model fine-tuned on WAD as its VLM backbone, without additional finetuning on our dataset. The quantitative results are shown in Table 1.

The results show that ForeSightGuide achieves the best overall performance among the WAD-fine-tuned baselines. On WAD, it obtains the highest ROUGE-1, ROUGE-L, and semantic similarity scores, with only a slight decrease in ROUGE-2 compared with Qwen2VL (7B)\*. On our selfbuilt dataset, ForeSightGuide outperforms Qwen3VL (8B)\*, improving ROUGE-1 from 0.380 to 0.424, ROUGE-2 from 0.133 to 0.161, ROUGE-L from 0.278 to 0.329, and semantic similarity from 0.731 to 0.749. Since ForeSightGuide uses WAD-fine-tuned Qwen2VL-7B and is not further trained on our dataset, the improvement mainly comes from the mask-filtered image and perceptual information prefix, which provide focused visual evidence and spatial-temporal cues for dynamic assistive guidance. It is also worth noting that Qwen3VL (8B)\* performs lower than Qwen2VL (7B)\* on the WAD test set, but achieves better results on our selfbuilt dataset. This indicates that strong fitting to a single benchmark does not necessarily imply better generalization. Compared with WAD, our dataset further requires the model to generate both guidance alerts and navigation instructions in complex dynamic scenes, where the stronger multimodal reasoning ability of Qwen3VL (8B)\* may become more beneficial.

Table 2 further evaluates redundant alerts and missed important obstacles. This experiment is conducted on 107 samples selected from our self-built dataset. GPT-5.5 misses the fewest important obstacles, but it produces many redundant alerts, with 186 redundant outputs in total and 1.74 redundant outputs per decision. Qwen3VL (8B) shows a similar redundancy problem and misses more important obstacles. In contrast, ForeSightGuide reduces redundant alerts to 32 in total and 0.299 per decision, corresponding to an 82.8% reduction compared with GPT-5.5 and an 83.2% reduction compared with Qwen3VL (8B). Although ForeSightGuide misses slightly more important obstacles than GPT-5.5, it greatly reduces unnecessary verbal feedback while keeping the missed obstacle rate much lower than Qwen3VL (8B). This demonstrates that the proposed masking and predictive perception mechanisms help the VLM focus on truly relevant obstacles, reducing redundant outputs without severely sacrificing safety.

Real-time tests. We further conduct a real-time walking test to evaluate ForeSightGuide in practical use. In this experiment, users carry a first-person camera and walk through real indoor and outdoor environments without manual intervention. During walking, we record the first-person video frames, fast response results, VLM navigation instructions, and guidance alerts. The fast response module runs continuously, while the VLM generates high-level instructions and reminders every 3 seconds. For evaluation, the recorded scenes and system outputs are provided to ChatGPT, which judges whether each fast response and VLM instruction is correct and whether important guidance alerts are missed. As shown in Table 3, the test evaluates 753 fast-response cases and 186 VLM decision cases. The average fast response interval is 650 ms, including image transmission, network delay, and model inference time. The fast response accuracy reaches 71.32% and the VLM instruction accuracy reaches 84.41%, while the average number of missed important alerts is 0.069. These results indicate that ForeSightGuide can operate continuously during real walking, but its real-time robustness still has room for improvement.

Table 1: Quantitative comparison on the WAD dataset and our dataset. The symbol \* denotes models fine-tuned on WAD.
<table><tr><td>Model</td><td colspan="4">WAD Dataset</td><td colspan="4">Our Dataset</td></tr><tr><td></td><td>ROUGE-1</td><td>ROUGE-2</td><td>ROUGE-L</td><td>Similarity</td><td>ROUGE-1</td><td>ROUGE-2</td><td>ROUGE-L</td><td>Similarity</td></tr><tr><td>Gemma 4 (2B)</td><td>0.091</td><td>0.023</td><td>0.073</td><td>0.420</td><td>0.228</td><td>0.022</td><td>0.179</td><td>0.549</td></tr><tr><td>DeepSeek (7B)</td><td>0.100</td><td>0.010</td><td>0.074</td><td>0.468</td><td>0.220</td><td>0.023</td><td>0.164</td><td>0.525</td></tr><tr><td>Qwen3VL (8B)</td><td>0.101</td><td>0.013</td><td>0.072</td><td>0.488</td><td>0.233</td><td>0.032</td><td>0.183</td><td>0.571</td></tr><tr><td>Qwen2VL (7B)</td><td>0.117</td><td>0.023</td><td>0.089</td><td>0.485</td><td>0.239</td><td>0.033</td><td>0.182</td><td>0.558</td></tr><tr><td>Qwen2VL (7B)*</td><td>0.436</td><td>0.296</td><td>0.398</td><td>0.598</td><td>0.300</td><td>0.064</td><td>0.250</td><td>0.666</td></tr><tr><td>Qwen3VL (8B)*</td><td>0.415</td><td>0.261</td><td>0.372</td><td>0.593</td><td>0.380</td><td>0.133</td><td>0.278</td><td>0.731</td></tr><tr><td>ForeSightGuide</td><td>0.439</td><td>0.293</td><td>0.401</td><td>0.611</td><td>0.424</td><td>0.161</td><td>0.329</td><td>0.749</td></tr></table>

Table 2: Comparison of redundant alerts and missed important obstacles. The first value denotes the total number of cases, and the value after the slash denotes the average number per decision output.
<table><tr><td>Model</td><td>Redundant</td><td>Missed</td></tr><tr><td>GPT-5.5</td><td>186 / 1.74</td><td>6 / 0.056</td></tr><tr><td>Qwen3VL (8B)</td><td>190 / 1.78</td><td>42 / 0.392</td></tr><tr><td>Ours</td><td>32 / 0.299</td><td>12 / 0.112</td></tr></table>

Table 3: Real-time walking test results.
<table><tr><td>Metric</td><td>Result</td></tr><tr><td>Fast response interval</td><td>650 ms</td></tr><tr><td>Fast response accuracy</td><td>71.32% (537/753)</td></tr><tr><td>VLM instruction accuracy</td><td>84.41% (157/186)</td></tr><tr><td>Missed important alerts (avg.)</td><td>0.069</td></tr></table>

Ablation Study. We further conduct ablation experiments on our self-built dataset to evaluate the contributions of the perceptual information prefix and mask-based filtering.As shown in Table 4, the full model achieves the best performance on all metrics. Removing both the perceptual prefix and the mask-based filtering causes the largest degradation, which indicates that the two components are complementary in improving guidance quality. In particular, removing the perceptual prefix leads to a larger drop than removing the mask alone, suggesting that structured obstacle information contributes more to semantic alignment and instruction generation than visual masking alone. Nevertheless, the mask-based filtering still provides clear gains by suppressing irrelevant visual content and reducing distraction for the VLM. Overall, the ablation results verify the efectiveness of both design choices in ForeSightGuide.

Table 4: Ablation study on the perceptual information prefix and mask-based filtering.
<table><tr><td>Setting</td><td>ROUGE-1</td><td>ROUGE-2</td><td>ROUGE-L</td><td>Similarity</td></tr><tr><td>w/o</td><td></td><td></td><td></td><td></td></tr><tr><td>Prefix+Mask</td><td>0.300</td><td>0.064</td><td>0.250</td><td>0.666</td></tr><tr><td>w/o Prefix</td><td>0.353</td><td>0.129</td><td>0.288</td><td>0.673</td></tr><tr><td>w/o Mask</td><td>0.383</td><td>0.154</td><td>0.320</td><td>0.682</td></tr><tr><td>Full Model</td><td>0.424</td><td>0.161</td><td>0.329</td><td>0.749</td></tr></table>

## Conclusion

This paper presents ForeSightGuide, a VLM-based assistive guidance framework for BVIPs in dynamic walking scenes. By using historical visual observations to estimate future obstacle motion, ForeSightGuide provides the VLM with a perceptual information prefix and a mask-filtered image. This helps the VLM focus on safety-relevant objects, improve guidance accuracy, and reduce redundant alerts. Experiments on WAD, our self-built dataset, and real-time walking tests show the efectiveness and preliminary feasibility of the proposed framework. Despite these results, ForeSight-Guide still has several limitations. First, the current motion prediction relies on a simple Kalman filter, which is insuficient for modeling complex pedestrian trajectories and social interactions in crowded scenes. Second, the scale of our selfbuilt dataset is still limited, and larger real-world datasets are needed to better evaluate generalization across diverse mobility scenarios. Third, depth estimation introduces considerable computational cost, which afects real-time performance on resource-limited devices. Future work will explore stronger trajectory prediction models, larger-scale data collection, and more eficient perception modules for practical deployment.

## Acknowledgments

This work was supported in part by the National Natura Science Foundation of China under Grant 52275029 and the Natural Science Foundation of Hubei Province under Grant 2025AFD639 and Grant 2025AFB119, and in part by the Science and Technology Plan Project of Hubei Province under Grant 2025CSA117, and in part by the Wuhan Municipal Natural Science Foundation Exploration Program (Chenguang Program) under Grant 2026040301020038, and in part by the Fundamental Research Funds for the Central Universities.

## References

Ahmetovic, D.; Gleason, C.; Ruan, C.; Kitani, K. M.; Takagi, H.; and Asakawa, C. 2016. NavCog: A Navigational Cognitive Assistant for the Blind. In Proceedings ofthe 18th International Conference on Human-Computer Interaction with Mobile Devices and Services, 90–99.

Bochkovskiy, A.; Delaunoy, A.; Germain, H.; Santos, M.; Zhou, Y.; Richter, S. R.; and Koltun, V. 2025. Depth Pro: Sharp Monocular Metric Depth in Less Than a Second. In International Conference on Learning Representations.

Chao, A.; Maquiling, E.; Chao, E.; Sanjeev, R.; Bossen, T.; and Greer, R. 2025. Automated Context-Aware Navigation Support for Individuals with Visual Impairment Using Multimodal Language Models in Urban Environments. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, 6745–6752.

Dakopoulos, D.; and Bourbakis, N. G. 2010. Wearable Obstacle Avoidance Electronic Travel Aids for Blind: A Survey. IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews), 40(1): 25–35.

Goswami, R. G.; Sinha, H.; Amith, P. V.; Hari, J.; Krishnamurthy, P.; Rizzo, J.; and Khorrami, F. 2024. Floor Plan Based Active Global Localization and Navigation Aid for Persons With Blindness and Low Vision. IEEE Robotics and Automation Letters, 9(12): 11058–11065.

Hayamizu, Y.; DeFazio, D.; Mehta, H.; Altaweel, Z.; Choe, J.; Lin, C.; Juettner, J.; Xiao, F.; Blackburn, J.; and Zhang, S. 2026. From Woofs to Words: Towards Intelligent Robotic Guide Dogs with Verbal Communication. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 38560–38569.

He, Y.; Chen, W.; Gan, X.; Wang, S.; Wang, H.; and Tan, Y. 2026. Attention to Threat-Relevant Objects: Reasoning Detection in Autonomous Driving via Multimodal Large Language Models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 4690–4698.

Kim, I. J. 2025. Recent Advancements in Indoor Electronic Travel Aids for the Blind or Visually Impaired: A Comprehensive Review of Technologies and Implementations. Universal Access in the Information Society, 24: 173–193.

Lee, J.-Y.; Jeong, Y.; Choi, S.-M.; and Seo, B. 2025. Assistive Guidance System Based on Online Path Structure Recognition for the Visually Impaired. In Proceedings of

the IEEE/RSJ International Conference on Intelligent Robots and Systems, 17178–17185.

Maidenbaum, S.; Hanassy, S.; Abboud, S.; Buchs, G.; Chebat, D.-R.; Levy-Tzedek, S.; and Amedi, A. 2014. The “EyeCane”, a New Electronic Travel Aid for the Blind: Technology, Behavior and Swift Learning. Restorative Neurology and Neuroscience, 32(6): 813–824.

Na, Q.; Zhou, H.; Fu, Z.; Yang, L.; and Frisoli, A. 2025. Dynamic Walking Corridor Generation for Visually Impaired Navigation Using Social Force Models and Convex Optimization. In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 6366– 6371.

Najjar, A. B.; Al-Issa, A. R.; and Hosny, M. 2022. Dynamic Indoor Path Planning for the Visually Impaired. Journal of King Saud University - Computer and Information Sciences, 34(9): 7014–7024.

Organization, W. H. 2026. Blindness and Vision Impairment. Patel, K.; and Parmar, B. 2022. Assistive Device Using Computer Vision and Image Processing for Visually Impaired: Review and Current Status. Disability and Rehabilitation: Assistive Technology.

Surougi, H. R.; and McCann, J. A. 2023. Real-Time Optimisation-Based Path Planning for Visually Impaired People in Dynamic Environments. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, 1831–1840.

Tapu, R.; Mocanu, B.; and Zaharia, T. 2020. Wearable Assistive Devices for Visually Impaired: A State of the Art Survey. Pattern Recognition Letters, 137: 37–52.

Wang, W.; Wei, F.; Dong, L.; Bao, H.; Yang, N.; and Zhou, M. 2020. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. In Advances in Neural Information Processing Systems, volume 33.

Yang, Z.; Yang, L.; Kong, L.; Wei, A.; Leaman, J.; Brooks, J.; and Li, B. 2022. SeeWay: Vision-Language Assistive Navigation for the Visually Impaired. In 2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC), 52–58. IEEE. Best Paper Finalist.

Yuan, Z.; Zhang, T.; Zhu, Y.; Zhang, J.; Deng, Y.; Jia, Z.; Luo, P.; Duan, X.; Zhou, J.; and Zhang, J. 2025. WalkVLM: Aid Visually Impaired People Walking by Vision Language Model. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 9845–9854.

Zhao, Y.; Wang, S.; and Li, J. 2026. LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, 34994–35002.