# Explainable AI-Powered Framework for Video-Based Skill Assessment in Cataract Surgery

Mohammad Javad Ahmadi¹ and Hamid D. Taghirad¹\*

1\*Applied Robotics and AI Solutions (ARAS), Faculties of Electrical and Computer Engineering, K.N. Toosi University of Technology, Tehran, Iran.

\*Corresponding author(s). E-mail(s): taghirad@kntu.ac.ir; Contributing authors: mjahmadi@email.kntu.ac.ir;

## Abstract

Persistent shortages in the surgical workforce and inherent limitations of traditional training methods highlight the necessity of automated, data-driven approaches in surgical education. This study addresses these challenges by introducing a novel, explainable AI-powered framework for automated skill assessment, specifically focusing on cataract surgery. We present the world's largest dataset of cataract surgery videos, comprising 2,000 recordings. Additionally, we propose an AI-powered analytical framework that employs advanced computer vision and signal-processing techniques to automatically evaluate surgical videos to derive objective, quantitative performance indicators that complement or potentially replace subjective scoring methods. A significant advantage of our framework over previous methods lies precisely in its explainability of outputs, elevating it beyond merely an opaque skill classification tool. Through experimental analysis of 83 cataract surgery videos, we demonstrate that the automatically computed metrics exhibit strong correlations with expert-based subjective evaluations, achieving up to 87% accuracy in surgical skill assessment. Each metric was individually examined, and expert surgeons provided subjective ratings using the newly introduced Capsulorhexis Skill Assessment System (CSAS). These subjective assessments were compared with ten objective motionbased metrics extracted through our framework. The results indicated a robust correlation between subjective ratings and automated indicators, underscoring the framework's capacity to accurately model surgical expertise.

Keywords: Surgical Education, Skill Assessment, Cataract Surgery, Computer Vision, Dataset, Video Analysis, Motion Analysis

## 1 Introduction

Global demand for surgical services is significantly increasing; however, access to these procedures remains uneven. Notably, low- and middle-income countries face a shortfall of 143 million essential surgeries each year [1]. Expectations indicate that every country will require at least 5,000 surgical procedures per 100,000 people annually by 2030 [2]. Addressing this gap necessitates a significant expansion in healthcare infrastructure and personnel, including doubling the surgical workforce over the next fifteen years [2].

Nevertheless, current educational and assessment methods in surgical training, such as the Master-Apprentice Model (MAM), are inadequate to achieve these objectives. These conventional methods face several challenges, notably the reliance on subjective performance evaluations rather than objective, measurable standards, which often leads to biased and inconsistent assessments [3]. Furthermore, the high workload of trainers limits the time available for providing detailed and comprehensive feedback [4].

The growing gap between surgical needs and available resources, combined with the limited integration of advanced technology in training, has resulted in a lack of skilled surgeons, longer patient wait times, increased surgical errors, and patient deaths [5]. Research indicates that early-career surgeons are nearly nine times more likely to commit procedural errors compared to experienced surgeons [6], underscoring the need for improved training and assessment methods [7, 8]. By implementing enhanced training programs, healthcare systems could prevent approximately 42,000 hospital readmissions annually, save up to \$3.62 million in costs [9], and mitigate 67% of cases leading to adverse events due to inadequate instrument handling skills [10].

In this context, advancements in artificial intelligence (AI) present promising opportunities to enhance surgical procedures and transform training paradigms. One approach is the recording and analysis of surgical videos. Unlike sensor and kinematic data, which require expensive equipment, video recordings are readily available in most operating rooms through devices like surgical microscopes. Beyond their accessibility, surgical videos provide a comprehensive view of the entire process, capturing the surgeon's movements, the patient's condition, and the progress of the surgical workflow [11].

Furthermore, integrating surgical recordings into training programs has become an invaluable resource for enhancing practical skills and academic research in the surgical field [11-13]. Novice surgeons report that reviewing videos and receiving feedback helps them prepare for surgeries, improve their anatomical knowledge, and refine their skills by analyzing past performances [13].

Despite the recognized benefits of surgical video materials, their widespread use remains limited due to challenges such as the time-consuming and expensive review of extensive video archives [14-16]. These findings highlight the need for advanced systems that streamline surgical video management, review, and analysis. Implementing intelligent platforms to create annotated video libraries could enhance training programs and improve clinical outcomes.

While current research efforts in automated surgical video analysis frequently focus on identifying operative phases and workflows, data-driven methodologies for objective video-based surgical skill assessment remain limited [17]. Current research predominantly utilizes deep neural network architectures to analyze surgical videos by processing them either as individual frames or as short video segments. These neural networks are designed to extract spatiotemporal features from video that correspond to distinct skill levels.

Funke et al. [18] proposed an end-to-end deep-learning pipeline that classifies surgical skills from short video clips using an inflated 3-D CNN and a Temporal Segment Network. Their evaluation was conducted on JIGSAWS [19], a bench-top simulation dataset, not footage from the real operations. Although the reported accuracies exceeded 95%, this performance was achieved only for a three-class classification. Moreover, the reliance on simulated data and the absence of interpretable feedback constrain the approach's immediate clinical relevance.

Kitaguchi et al. [20] trained a 3-D CNN on 40-second clips from 74 videos of laparoscopic colorectal procedures labeled with Endoscopic Surgical Skill Qualification System (ESSQS) scores. Their model achieved a mean accuracy of 75% when classifying surgical performance into low, medium, or high skill levels. However, the moderate accuracy, use of a small and selectively sampled dataset, and reliance on a black-box three-tier output limit the method's robustness and its utility for providing meaningful, detailed educational feedback.

Zia et al. [21] proposed RP-Net-V2, a CNN-LSTM architecture that segments robotic-assisted radical prostatectomies into 12 steps and evaluates performance through task-based efficiency metrics, achieving a Jaccard index of 0.85 and strong correlations with expert scores. However, the model's reliance on raw RGB frames leads to frequent confusion between visually similar tasks, the class imbalance is left unaddressed, a simple running-window median filter shifts some predicted boundaries by over a minute, and the ground-truth labels originate from a single annotator.

Lajkó et al. [22] proposed a 2D video-based skill assessment method using sparse optical flow and standard classifiers on the JIGSAWS dataset, achieving around 80% accuracy. However, performance remained below kinematics-based methods, and the approach relied on manually selected tool regions, limiting automation.

Although deep learning models have achieved promising results in medical applications, their widespread use in sensitive clinical settings remains constrained by several issues, including the black box nature of most neural networks, the demand for extensive expert-annotated datasets, and limited capacity for delivering detailed, actionable feedback to surgeons.

To address these challenges, this paper introduces an AI-powered framework that integrates explainable methods for objective and automated skill evaluation in surgical videos. Although the framework is applicable to a wide range of surgical procedures, its effectiveness should be independently evaluated in each context. As a proof of concept, we focus on cataract surgery, specifically the capsulorhexis phase, as our primary case study. This choice is motivated by the fact that cataract surgery is among the most frequently performed surgeries worldwide [23].

Cataract remains the primary cause of preventable vision loss and significant visual impairment, particularly in developing nations [24, 25]. Phacoemulsification is the globally preferred surgery for cataract removal, making it a vital component of the residency educational curriculum for developing surgical competency [26]. Surgical competency requires deep knowledge, dexterity, micro-surgical skills, and dedication [27].

Creating a circular opening in the anterior surface of the cataracted lens, called capsulorhexis, is a fateful step in phacoemulsification cataract surgery. It's a significant challenge for first-year residents [28, 29]. Trainee surgeons require years of training and evaluation, with evolving methods and tools for learning and assessment. Rubrics, with their structured approach, are the conventional tools in skill assessments.

Several rubrics are available for evaluating cataract surgery, such as the International Council of Ophthalmology's Ophthalmology Surgical Competency Assessment Rubric (ICO-OSCAR) [30], Objective Assessment of Skills in Intraocular Surgery (OASIS) [31], Global Rating Assessment of Skills in Intraocular Surgery (GRASIS) [32], and Objective Structured Assessment of Cataract Surgical Skill (OSACSS) [33]. However, these rubrics typically require direct trainer supervision, leading to high costs, time-consuming processes, and potential biases.

Our proposed framework overcomes these limitations by automating the assessment process, eliminating the need for direct trainer supervision. This reduces both costs and time, while minimizing the potential for biases, providing a more efficient and objective method for evaluating surgical skills.

In the following sections, we present our newly developed cataract surgery video dataset, the largest in the world in terms of the number of data, and comprehensive annotations. We then describe our method for intelligent, automated, and objective skill evaluation, detailing the framework's architecture and operational steps. In the Results section, we report the performance of the proposed framework on 83 cataract surgery videos, accompanied by extensive analyses that underscore both the significance of this dataset and the method's effectiveness. We also discuss each objective metric in depth to illuminate various dimensions of skill assessment. Finally, in the Conclusion, we summarize the key findings and outline potential directions for future research.

## 2 Methods

In this section, we present a large-scale surgical video dataset designed to support interdisciplinary AI research and serve as ground truth for validating our automated skill assessment systems. We then introduce a novel AI-powered framework that seamlessly processes these videos, generating interpretable metrics of surgical performance and advancing the reliability of automated skill evaluation in clinical practice.

## 2.1 Dataset Crafting Description

Cataract surgery is performed with precise instruments under a surgical microscope, while an attached camera captures the operation from the surgeon's perspective. The ARAS Group and Farabi Eye Hospital jointly developed an infrastructure for the systematic acquisition of these surgical videos.

From 2021 to 2024, more than 2,500 videos were recorded at up to 60 frames per second with a resolution of 1920×1080 pixels. The video selection process involved rigorous exclusion criteria that filtered out videos with technical flaws such as substandard image quality, and low resolution.

Having established a high-quality dataset, the next step is to select an appropriate method for skill-based annotation. In the domain of surgical video analysis, two principal methods have been utilized to annotate surgeon skill. The first method involves identifying individual surgeons and estimating their proficiency based on metrics such as academic rank and accumulated surgical experience, as utilized in the Cataract-101 dataset [34].

In contrast, the second approach relies on expert review of surgical videos, where evaluators assign skill scores according to predefined standards. This expert-based approach generally provides a more accurate and unbiased evaluation of surgical performance than identity-based methods.

This study introduces a video-based skill annotation method that segments surgical procedures into distinct phases, each evaluated with specialized criteria. In our case study, video segments corresponding to the capsulorhexis phase were extracted from the recorded procedures, and detailed annotations were assigned to these segments using a new system developed to standardize the review process of capsulorhexis videos.

The proposed assessment framework was validated through a comprehensive review of existing surgical skill assessment systems [35-39], carried out in collaboration with two experienced surgeons. Based on this review, we introduced the Capsulorhexis Skill Assessment System (CSAS). Table 1 details the 11 indicators of CSAS, incorporating seven subjective and four objective metrics: rhexis position, shape, size, and time. Notably, the indicators "Commencement of Flap" and “Circular Completion" follow the ICO-OSCAR [39] guidelines, whereas “Tissue Handling", “Instrument Handling", and “Microscope Use" have been adapted from the GRASIS [40] scoring system.

In this study, 83 preprocessed surgical videos have been independently reviewed and rated by three trained medical professionals using CSAS subjective indicators. This review process resulted in crafting the cataract surgical video skill assessment dataset (CVSAD-83) [41]. Each indicator was rated on a five-point scale, where higher scores indicate greater proficiency. The reliability of these annotations was confirmed by an intraclass correlation coefficient (ICC) of 0.79 and a Cronbach's alpha coefficient of 0.93, indicating high inter-rater reliability and internal consistency. To facilitate the categorization of surgeon skills according to the Dreyfus Model [42], an overall score (OS) was computed for each video by averaging the scores of all indicators. This OS, ranging from 1 to 5, is a reliable metric for skill classification.

## 2.2 AI-Powered Objective Skill Assessment Framework

This paper introduces an innovative framework designed to enhance the explainability of AI-powered systems for automatic surgical skill assessment through video analysis. The framework begins with a computer vision algorithm that tracks and segments surgical tools and relevant tissues, extracting pixel-level motion information related to these elements. Subsequently, this raw pixel data is processed through a post-processing stage, converting it into actionable information for surgical skill assessment. Finally, an objective evaluation system is developed, which evaluates the surgeon's skills based on motion data and formulated criteria extracted from the logic of subjective indices.

<table><tr><td>Indicator</td><td>Novice</td><td colspan="3">Advanced Beginner</td><td colspan="3">Competent</td></tr><tr><td>Instrument Handling</td><td>Description Repeated, Abrupt,</td><td>Score 1-2</td><td>Description Selected/Occasional</td><td>Score 3-4</td><td>Description Fine and Smooth Movements; No</td><td></td><td>Score 5</td></tr><tr><td></td><td>Awk- ward/Bizarre, and Harsh Movements; Endless Insertion/Entry and Exit</td><td></td><td>Inappropriate Movements</td><td></td><td>Inappropriate Movements</td><td></td><td></td></tr><tr><td>Motion</td><td>Unsure Surgical Plan; Needless in- doubt Movements</td><td>1-2</td><td>Certain Surgical Plan; Occasional/s- elected Unnecessary Movements</td><td>3-4</td><td>Maximum Effective Movements; No Unnecessary Movements</td><td></td><td>5</td></tr><tr><td>Commencement flap</td><td>Tentative Chases rather than Con- trolled; Numerous disruptions in the cortex</td><td>1-2</td><td>Flap Pull-up After 2–3 Tries; Subtle Cortex Disruptions</td><td>3-4</td><td>No Cortex disruption</td><td>Delicate and Controlled Approach;</td><td>5</td></tr><tr><td>Tissue Handling</td><td>Using Unnecessary Force; Damage to Cornea and Conjunctiva</td><td>1-2</td><td>Suitable Tissue Interactions; Unin- tentional/Accidental Tissue Damage</td><td>3-4</td><td>Excellent Tissue Tissue Damage</td><td>Interactions; No</td><td>5</td></tr><tr><td>Microscope Use</td><td>Multiple Recentering; Multiple Refo cus</td><td>1-2</td><td>Few attempts to recenter/refocus</td><td>3-4</td><td>View</td><td>Keeps Eye Centered; Good Focused</td><td>4-5</td></tr><tr><td>Circular Completion</td><td>Not Able to Achieve Circular Rhexis; Entrance to Dangerous Regions (Extend to the Periphery)</td><td>1-2</td><td>Difficulty Achieving Rhexis</td><td>3-4</td><td>Rapid, Unaided Rhexis</td><td>Completion of</td><td>5</td></tr><tr><td>Adverse Events</td><td>Rhexis Tear; Entering Hazardous Areas; Unable to Recognize Adverse events. Inappropriate Reaction</td><td>√</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Position of Rhexis Shape of Rhexis</td><td>&gt;1mm Irregular Shape and Margins</td><td>1-2 1-2</td><td>0.4-1 mm Semi-Circular Shape; Minor Irregu-</td><td>3-4 3-4</td><td>&lt; 0.4 mm</td><td></td><td>5 4-5</td></tr><tr><td>Size of Rhexis</td><td></td><td></td><td>larities along Edges</td><td></td><td>Perfectly Circular</td><td></td><td></td></tr><tr><td>Time</td><td>&lt; 4.25 mm or &gt; 6.25 mm</td><td>1-2 1-3</td><td>4.25–4.75 mm or 5.75–6.25 mm</td><td>3-4</td><td></td><td>4.75–5.75 mm (5–5.5 ± 0.25 mm)</td><td>5</td></tr><tr><td></td><td>&gt; 75 seconds</td><td></td><td></td><td></td><td>≤75 seconds</td><td></td><td>4-5</td></tr></table>

aint pns Sss ns Ss : is

## 2.2.1 Motion Data Extraction and Processing

In this section, we present a detailed overview of the methods used to extract and process video-based motion data. To accomplish this, our proposed framework integrates state-of-the-art computer vision algorithms for segmentation and tracking, such as SAM-2 [43] or YOLO-11 [44].

In this study, pre-trained models developed in our previous studies [45–47] were utilized to segment the capsulorhexis instrument and the corneal tissue in the videos of CVSAD-83 dataset. The motion data obtained through tracking these components provides the essential data to evaluate surgical skills.

Following the segmentation-tracking stage, the frame-by-frame coordinates of the capsulorhexis tool-tip and the corneal center are identified, forming the fundamental motion data for analysis. However, to ensure the resulting measurements are meaningful for skill evaluation, a proper referencing system must be established. Without appropriate referencing, raw positional measurements may lead to inaccurate conclusions about skill performance.

Absolute referencing, in which each component's position is determined relative to a fixed point in the frame (e.g., the top-left corner), can be misleading. Rapid shifts in a tool's location with respect to this fixed point may arise from patient eye movements or changes in the microscope's field of view, rather than a lack of surgical precision.

To address this limitation, a relative referencing scheme provides a more reliable measure for skill-related analysis by representing the tool-tip coordinates in relation to the target tissue (in this case, the corneal center). Transitioning to a relative reference system effectively translates sudden absolute movements into smoother, more interpretable variations in the tool-tip's position, allowing for a more accurate assessment of surgical performance.

After extracting these relative positions, the data must be scaled to account for varying magnifications in the surgical microscope across different videos. One practical method is to measure the corneal diameter in pixels using computer vision techniques and then convert these measurements to physical units based on the known corneal diameter (11.71 ± 0.42 mm [48]). Although this scaled data may not precisely replicate real-world dimensions, it serves its primary purpose of capturing the essential motion trends indicative of surgical proficiency.

Finally, higher-order motion features such as velocity, acceleration, and jerk are derived from the scaled relative positions for objective skill evaluation in subsequent analytical steps. By focusing on the dynamics of tool movement rather than on absolute positional data, this approach provides a more reliable framework for subsequent analytic processes to distinguish between different levels of surgical expertise.

## 2.2.2 Objective Metrics for Automated Surgical Skill Assessment

Following the discussion on motion data extraction and processing methods, this section presents the objective metrics for automated surgical skill assessment. These

metrics were derived through expert consultation and by adapting the principles from subjective assessment systems.

1. Procedure Time (PT): The total time for completing the surgery, $T ,$ is computed by subtracting the start time from the end time, as in Equation 1:

$$
\mathrm { P T } = t _ { \mathrm { e n d } } - t _ { \mathrm { s t a r t } } .\tag{1}
$$

2. Path Lenrth (PL): This metric is computed by summing the magnitudes of the tool's relative positional coordinates. Specifically, if $( x _ { i } , y _ { i } )$ represents the position at the i-th time instant, then the path length is given by Equation 2:

$$
\mathrm { P L } = \sum _ { i = 1 } ^ { n } \sqrt { x _ { i } ^ { 2 } + y _ { i } ^ { 2 } } ,\tag{2}
$$

where n is the total number of sampled points in the recorded trajectory.

3. Number of Alternating (Back-and-Forth) Movements (NAM): This metric reflects the count of tool direction changes. A change in the sign of the velocity along any axis indicates a reversal in tool motion. The total number of such movements is calculated as in Equation 3:

$$
{ \mathrm { N A M } } = \sum _ { i = 1 } ^ { n - 1 } \delta ( \operatorname { s i g n } ( v _ { i + 1 } ) - \operatorname { s i g n } ( v _ { i } ) ) ,\tag{3}
$$

where sign(·) extracts the velocity direction along either x or y axis, and $\delta ( \cdot )$ is 1 if a sign change is detected, otherwise 0.

4. Motion Difficulty (MD): Abrupt changes in acceleration indicate higher jerk, which are associated with reduced smoothness [49]. The overall difficulty is quantified by summing the magnitude of the jerk vector, as given by Equation 4:

$$
\mathrm { M D } = \sum _ { i = 1 } ^ { n } \sqrt { j _ { x _ { i } } ^ { 2 } + j _ { y _ { i } } ^ { 2 } } ,\tag{4}
$$

where $j _ { x _ { i } }$ and $j _ { y _ { i } }$ are the jerk components along the x and y axes, respectively. 5. Energy Consumption (EC): By assuming the needle mass is negligible and constant, the kinetic energy concept $\scriptstyle { \frac { 1 } { 2 } } m v ^ { 2 }$ can be used to estimate the total energy. Let $( v _ { x _ { i } } , v _ { y _ { i } } )$ be the velocity components at time i. The EC is estimated using Equation 5:

$$
\mathrm { E C } = \sum _ { i = 1 } ^ { n } \frac { 1 } { 2 } ( v _ { x _ { i } } ^ { 2 } + v _ { y _ { i } } ^ { 2 } ) .\tag{5}
$$

6. Number of Local Peaks (NLP): The local peaks in speed, acceleration, or jerk profiles are counted using computational methods such as the find\_peaks Python function in scipy.signal. Let NLP denote the total number of detected peaks across these profiles, formalized as in Equation 6:

$$
\mathrm { N L P } = \mathrm { P e a k s } ( \mathrm { s p e e d } ) + \mathrm { P e a k s } ( \mathrm { a c c e l e r a t i o n } ) + \mathrm { P e a k s } ( \mathrm { j e r k } ) ,\tag{6}
$$

where Peaks(·) returns the count of local peaks in the corresponding signal.

7. Total Force (TF): Inspired by the physical relation Force $\mathbf { \mu } = m \times a$ and assuming negligible needle mass, the total force is estimated by summing the magnitude of the acceleration vector. Equation 7 defines Total Force:

$$
\mathrm { T F } = \sum _ { i = 1 } ^ { n } \sqrt { a _ { x _ { i } } ^ { 2 } + a _ { y _ { i } } ^ { 2 } } .\tag{7}
$$

8. Tissue Interaction (TI): Sharp deviations from the mean acceleration can indicate sudden forces that might harm tissue. By comparing local acceleration peaks with the mean, an interaction index (TI) is computed as indicated in Equation 8:

$$
\mathrm { T I } = \sum _ { i = 1 } ^ { n } \left| \sqrt { a _ { x _ { i } } ^ { 2 } + a _ { y _ { i } } ^ { 2 } } - \mu _ { a } \right| ,\tag{8}
$$

where $\mu _ { a }$ is the average magnitude of acceleration.

9. Speed Frequency Smoothness (SFS): The mean frequency magnitude in the fast Fourier transform (FFT) of the speed signal is used to indicate frequency smoothness. Equation 9 details the computation:

$$
\mathrm { S F S } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left| \mathrm { F F T } ( v _ { i } ) \right| ,\tag{9}
$$

where FFT(·) denotes the FFT of the velocity at each sampled interval.

10. Spectral Entropy (SE): The distribution of the signal's energy across frequencies is captured by the spectral entropy, described in Equation 10:

$$
\mathrm { S E } = - \sum _ { f } P ( f ) \log P ( f ) ,\tag{10}
$$

where $\begin{array} { r } { P ( f ) = \frac { | \mathrm { { F F T } } ( x _ { f } ) | ^ { 2 } } { \sum _ { j } | \mathrm { { F F T } } ( x _ { j } ) | ^ { 2 } } } \end{array}$ is the normalized power spectral density of the signal.

In the subsequent section, it will be demonstrated how these metrics, which were developed from subjective indices and motion data extracted from surgical videos, correlate with expert surgeons' subjective assessments in the CVSAD-83 dataset. An overview of our proposed framework is presented in Figure 1.

## 3 Results

This section presents and discusses the findings of three primary analyses conducted on the CVSAD-83 dataset. First, we examine the distribution of subjective skill scores offering insights into typical performance levels and areas for targeted improvement. Next, we assess the correlation between ten automated computed objective metrics and the overall subjective score (OS) assigned by expert surgeons, thereby evaluating the reliability of automated indicators as proxies for clinical expertise. Finally, we explore each metric's discriminative power by clustering surgeons into different skill groups, comparing the results with ground-truth expert labels.

![](images/d7be854e64bc580ac57cf6ef5f3b48b81386683a2a8811ee4508c5dd87e24376.jpg)  
Fig. 1: Overview of the proposed framework.

The box plots in Figure 2 illustrate the distribution of scores for the subjective indicators of the CSAS evaluation system, which were used to evaluate the 83 videos from the CVSAD dataset, in addition to the overall score (OS). In each plot, the box spans the interquartile range (IQR), with the median and mean values denoted by the horizontal lines and whiskers marking the full data range. Notably, most indicators exhibit median scores near or above 3.75, indicating that the majority of assessed performances cluster within the upper half of the 1–5 scale.

This insight into the distribution of performances helps trainers develop targeted interventions that can systematically improve surgical outcomes by addressing specific weak points discovered through the CSAS framework. For instance, the scores for Microscope Use and Tissue Handling appear slightly higher than those for Motion and Circular Completion, indicating that surgeons generally excel in maintaining focus and handling ocular tissues while having more varied performance on tasks requiring precise movement control.

Following the subjective score distribution analysis, we explored the relationship between ten objective metrics generated by our automated assessment framework and the expert-assigned Overall Scores (OS). Establishing strong correlations between objective indicators and subjective assessments is crucial for validating the reliability of our framework.

Figure 3 presents scatter plots for each metric against OS, with the horizontal axis representing the expert-assigned score (OS) value and the vertical axis showing the corresponding objective metric output. Data points are labeled by their video IDs, and a regression line highlights the general trend in each plot. A consistent inverse relationship emerges across all ten objective indicators, reflecting that surgeons assigned higher OS values typically exhibit lower measured values for objective metrics. A detailed and thorough analysis of the charts corresponding to each of the 10 objective indicators is presented as follows:

![](images/3f0852cd6a48496bd077139010cafd1c0f3c703407c28138fd0125d40fa87684.jpg)

![](images/d21deebb516e4dfb60244ad5689935e32a9c0206e4e9eb09917d5ba91acb56dd.jpg)

![](images/443b7db6e313e5034b712742738d4bbb4775f632f9e4e48779682b054619ff1d.jpg)

![](images/befb9895be3093c79df9b6aad34d51881b5939b121ec570377ba7593583f1ff0.jpg)

![](images/4167f7e2d5b481fcb3fec39d26e78617aa34e84130432cb1023adecb5363769a.jpg)

![](images/b1a9292ff45a49182029715050ce0facd9bc059b7bec92886974d781f32abace.jpg)

![](images/dc21553caa9dab32a73013b7299f13bbb0a36fbc8a6b654a415fc6b22c8cfe4e.jpg)

![](images/5b4e84945dcb7ab24304376612d6520ac6737a270eeda5394ee8931b2baf41da.jpg)  
Fig. 2: Overview of CSAS scores and overall performance from 83 CVSAD cases.

1. Procedure Time (PT): As depicted in Figure 3(a), a significant inverse correlation was observed between Procedure Time (PT) and subjective skill scores, with a regression slope of –60.057. As skill scores increased, PT decreased, indicating that more proficient surgeons complete procedures more efficiently, likely due to enhanced technical dexterity and optimized decision-making. These findings highlight the value of PT as a metric for surgical competence, emphasizing its potential to improve patient outcomes and increase operating room efficiency.

2. Path Length (PL): Analysis of the PL metric underscores a negative regression slope of -272.946 in its correlation with subjective surgical skill scores (Figure 3(b)). The shorter path lengths observed at higher skill levels suggest that experienced surgeons execute movements more efficiently, limiting unnecessary travel. This streamlined motion is particularly valuable during delicate procedures like capsulorhexis, where minimizing extra instrument movement enhances precision and patient outcomes.

3. Number of Alternating Movements (NAM): Figure 3(c) shows that the NAM exhibits a strong negative slope (-1840.795) when regressed against subjective surgical skill scores. Surgeons with higher proficiency perform fewer directional reversals, implying more controlled, deliberate motions and purposeful instrument handling. Reducing rapid back-and-forth movements enhances procedural efficiency and lessens the potential for tissue damage. These improvements in motor planning and execution are especially valuable in complex operations like capsulorhexis, where even minor inefficiencies can compromise patient outcomes.

![](images/f5b258fdee9fad0b7af901fd2bcdce7107beda946c7ed048985cb8f6417e267a.jpg)  
(a) Procedure Time (PT)

![](images/32ed35e1e978b256af1bcdee55dd6353ab7dbe9ef41cd2494a3d1fb9bd250b42.jpg)  
(b) Path Length (PL)

![](images/d1b598fe59cf1badb4e865f0e998b1d4418fa1d6a693935c2e9d56fcc39d6f56.jpg)  
(c) Number of Alternating Movements (NAM)

![](images/b6dc951f0dd10bff8d9731dc61794a646bb10c5d702e5c85a2042c7cceb906a9.jpg)  
(d) Motion Difficulty (MD)

![](images/58dd2ffaa46b2636a22fe24ae8dc6f8a64073a56e0046de9178089416e490b32.jpg)  
(e) Energy Consumption (EC)

![](images/d833cf2150f75f3de3b4fe24824be6afc7afd390fbb7d43aac8e845fefdd0b4c.jpg)  
(f) Number of Local Peaks (NLP)

![](images/a717f9dc5f5f64848fa1834171a544515c9a4b94dfe9ec77153eb776a4cc19df.jpg)  
(g) Total Force (TF)

![](images/b01fa6f44ac28e2d5f8a750b9c48b2b367ced69494ca061beb006cb7a358deb6.jpg)  
(h) Tissue Interaction (TI)  
Figure 3 continues on the next page.

![](images/58654505e340d9f7cfbafd4b456678f782d9ec4478a5cc321a8c8570b2e7c7bf.jpg)  
(a) Speed Frequency Smoothness (SFS)

![](images/307548829b6cf098f08cc5b6a618f5d2320373b6a0d76c67b78e44b08f76c8f0.jpg)  
(b) Spectral Entropy (SE)  
Fig. 3: Correlation plots for different metrics.

4. Motion Difficulty (MD): In Figure 3(d), the analysis of the MD metric demonstrates a significant inverse relationship with subjective surgical skill scores, as indicated by a regression slope of -44.944. This result reveals that higher skill levels are associated with smoother movements, thereby minimizing abrupt changes in acceleration. This reduction in motion difficulty is critical for decreasing tissue trauma and achieving optimal surgical outcomes [49].

5. Energy Consumption (EC): The analysis in Figure 3(e) reveals a negative regression slope of -0.227, indicating that higher skill corresponds with lower energy consumption. This relationship suggests that proficient surgeons execute more efficient movements that minimize unnecessary kinetic energy expenditure. Consequently, such energy efficiency is a marker of refined motor control.

6. Number of Local Peaks (NLP): The regression analysis depicted in Figure 3(f) shows that the NLP is strongly negatively correlated with subjective surgical skill scores (slope: -1298.269). This finding suggests that more proficient surgeons achieve more fluid and controlled instrument movements, marked by fewer abrupt fluctuations. Enhanced motor control and precision, as evidenced by reduced NLP values, are vital in minimizing tissue trauma and improving surgical outcomes.

7. Total Force (TF): The analysis of the TF in Figure 3(g) reveals a strong inverse correlation (slope: -25.394) with subjective skill scores. Surgeons with higher proficiency achieve more fluid and controlled instrument handling, reducing unnecessary force application.

8. Tissue Interaction (TI): The regression analysis in Figure 3(h) shows a negative slope of -0.059, implying that increased surgical proficiency is associated with reduced TI values. Such reduction and consistency in motion reduce the likelihood of abrupt force fluctuations that could damage tissue, underscoring the importance of uniform and controlled acceleration patterns in ensuring successful outcomes in sensitive surgical interventions.

9. Speed Frequency Smoothness (SFS): In the frequency domain, abrupt and inadequately planned movements are typically represented by high-frequency coefficients, while smoother movements correspond to lower-frequency coefficients [50]. The analysis depicted in Figure 3(a) shows that the SFS metric is inversely correlated with subjective surgical skill scores (regression slope: -0.171). Lower SFS values, associated with higher proficiency, indicate smoother speed profiles with fewer high-frequency fluctuations.

10. Spectral Entropy (SE): The regression analysis in Figure 3(b) shows that the spectral entropy (SE) metric exhibits a negative slope of -0.729. This result implies that higher-skill surgeons demonstrate lower SE values, meaning that the motion signal's energy is more tightly focused in particular frequency bands. This energy distribution pattern indicates consistent instrument movements and underscores improved motor control and precision.

These correlations confirm the reliability of the proposed framework in replicating expert-based evaluations. By employing video-derived metrics as automated proxies for real-world competence, this approach enables consistent and scalable skill assessments across various surgical contexts.

In the final stage of our analysis, multiple clustering algorithms, including AggAverage, AggComplete, AggSingle, Agglomerative, BayesianGMM, Birch, BisectingK-Means, GaussianMixture, KMeans, MiniBatchKMeans, and Spectral, were employed to partition surgeons into two distinct groups based on their Overall Score (OS) as determined by expert subjective evaluation. The group exhibiting higher OS values was designated as “Expert (E)," whereas the lower-scoring group was labeled “Intermediate (I)." This OS-based clustering served as the ground truth for subsequent comparisons.

Subsequently, the same clustering techniques were applied independently to ten objective performance metrics derived from our intelligent assessment framework. Consistent with the inverse relationships observed in Figure 3, lower values of these metrics corresponded to the “Expert" cluster, while higher values mapped to the “Intermediate" cluster.

Subsequently, each objective metric's clustering was compared against the OSbased clustering to assess how accurately it could distinguish between “Expert" and "Intermediate" surgeons. The formula below measures this accuracy by quantifying the overlap of surgeon IDs in the E and I clusters derived from each objective metric with the corresponding E and I clusters obtained via OS-based clustering. Thus, it provides a straightforward indicator of how well each objective metric aligns with the expert-derived groupings:

$$
\mathrm { A c c u r a c y } \ = \frac { | \mathcal { E } _ { \mathrm { O S } } \cap \mathcal { E } _ { \mathrm { m e t r i c } } \mid + | \mathcal { Z } _ { \mathrm { O S } } \cap \mathcal { Z } _ { \mathrm { m e t r i c } } \mid } { | \mathcal { E } _ { \mathrm { O S } } | + | \mathcal { Z } _ { \mathrm { O S } } | }\tag{11}
$$

where ${ \mathcal { E } } _ { \mathrm { O S } }$ and $\boldsymbol { \mathcal { T } } _ { \mathrm { O S } }$ are the sets of surgeons labeled “Expert" and “Intermediate" by the OS-based clustering, and $\mathcal { E } _ { \mathrm { m e t r i c } }$ and $\mathcal { T } _ { \mathrm { m e t r i c } }$ are the corresponding sets obtained from the metric-based clustering.

Table 2 summarizes the highest accuracy achieved for each indicator. For each objective metric, the table lists the clustering method that produced the optimal accuracy when compared with OS-based labels, along with the corresponding threshold ranges (minimum and maximum values) for both the “Expert" and “Intermediate" clusters. Among all the objective indicators, NAM clustered via the Agglomerative method exhibited the greatest alignment with the surgeon groupings derived from subjective scoring

In the case of the Overall Score, the table reports the aggregated thresholds across all methods, with an observed overlap in the range of 3.23 to 3.78. This overlap reflects the inherent variability in subjective assessments when consolidated over multiple clustering approaches.

Table 2: Top clustering accuracies and threshold ranges for classifying surgeons into Expert and Intermediate groups.
<table><tr><td>Indicator</td><td>Method</td><td>Accuracy (%)</td><td>Thresholds for [E], [1]</td></tr><tr><td>PT</td><td>KMeans</td><td>82</td><td>[17-93], [97-288]</td></tr><tr><td>PL</td><td>KMeans</td><td>84</td><td>[69-505], [544-1327]</td></tr><tr><td>NAM</td><td>Agglomerative</td><td>87</td><td>[404-3795], [4063-9135]</td></tr><tr><td>MD</td><td>Spectral</td><td>85</td><td>[14-60], [63-217]</td></tr><tr><td>EC</td><td>GaussianMixture</td><td>82</td><td>[0.06-0.39], [0.40-1.42]</td></tr><tr><td>NLP</td><td>Spectral</td><td>85</td><td>[350-1582], [1616-6115]</td></tr><tr><td>TF</td><td>Spectral</td><td>85</td><td>[8-34], [36-123]</td></tr><tr><td>TI</td><td>GaussianMixture</td><td>75</td><td>[0.05-0.21], [0.22-0.58]</td></tr><tr><td>SFS</td><td>Agglomerative</td><td>81</td><td>[0.21-0.52], [0.54-1.10]</td></tr><tr><td>SE</td><td>Spectral</td><td>84</td><td>[5.84-7.37], [7.40-8.66]</td></tr><tr><td>Overall Score (OS)</td><td>All Methods</td><td></td><td>[2.39-3.78], [3.23-4.88]</td></tr></table>

Overall, our findings emphasize the critical role of precise hand-eye coordination and muscle control in surgical performance. These results align closely with previous research highlighting the importance of motion smoothness, controlled force application, and careful instrument handling for successful surgical outcomes [51–54].

Additionally, our study represents a significant advancement beyond prior investigations, such as Kim et al. [55], by examining a more comprehensive set of skill indicators derived from detailed motion data, coupled with a rigorously annotated dataset. Our analysis clearly demonstrates that expert surgical performance depends heavily on the effective coordination of visual perception, hand movements, and muscular precision.

## 4 Conclusion

In this paper, we introduced an explainable AI framework designed to evaluate surgical skills in cataract procedures automatically. The objective framework presented here offers an accurate and scalable approach for assessing surgical proficiency. This approach has strong potential to complement or even replace existing subjective assessment systems like ICO-OSCAR. Furthermore, it can play a valuable role in refining surgical training programs by providing targeted, objective feedback, thereby systematically guiding trainees toward improved surgical outcomes.

The study stands out for several key achievements. First, we crafted the world's largest repository of cataract surgery videos, encompassing 2,000 recordings, of which 83 were meticulously annotated using the newly established Capsulorhexis Skill Assessment System (CSAS). This dataset enabled a reliable benchmark for subjective (expert-driven) and objective (motion-based) measurements, thereby bridging a longstanding gap in AI-driven surgical education research. Notably, it surpasses JIGSAWS [19] by capturing real surgical scenes rather than simulated tasks, and it improves upon Cataract-101 [34] and Cataract-1K [56] through a structured, multi-indicator annotation process conducted by three independent physicians.

Second, our approach proposed a data-driven pipeline that extracts clinically meaningful motion signals from the raw video stream. Through advanced computer vision algorithms for segmentation and tracking, we converted pixel-level motion data into quantifiable metrics that thoroughly reflect technical precision, smoothness, and tissue handling. This capability addresses the need for objective, reproducible indicators of surgical expertise.

Third, and most critically, we demonstrated that our ten computed motion-based metrics, ranging from path length to tissue interaction measures, exhibit robust correlations with expert-derived capsulorhexis skill ratings, achieving up to 87% accuracy in skill classification. This level of performance indicates that automated, objective measures can match or even exceed traditional subjective evaluations, challenging the notion that high-quality surgical skill assessments must rely solely on human observers.

Fourth, our focus on explainability stands as a cornerstone achievement. By translating raw computational outputs into clinically interpretable indices, we offer not just an accuracy-driven tool but also a transparent framework readily understood by surgical educators and trainees. Such interpretability fosters trust in AI and provides targeted feedback, allowing surgeons to pinpoint, for example, whether excessive force, abrupt acceleration, or instrument misalignment underlies lower performance scores.

Despite these promising results, certain limitations persist. The current validation focuses heavily on the capsulorhexis phase, and additional studies are essential to ensure generalizability across other surgical tasks and clinical environments. The varying complexities of multi-institutional settings also underscore the necessity for broader validation. Furthermore, while the proposed metrics operate efficiently in retrospective analyses, the challenges of real-time feedback integration, such as processing speed, user interface design, and integration within the clinical workflow, remain outstanding.

Future directions for this research thus include (1) broadening the scope to capture other phases of cataract surgery (e.g., phacoemulsification and intraocular lens placement) and extending to different surgical specialties, (2) conducting large-scale, multi-center validation studies to account for diverse patient populations and healthcare infrastructures, (3) developing real-time coaching systems that combine motion analytics with intraoperative feedback to foster faster learning curves, and (4) incorporating advanced simulator and virtual reality platforms, allowing trainees to benefit from immediate, data-driven metrics in a safe, controlled setting, (5) undertaking a clinical study to assess and model the learning curve of surgical trainees employing this method, and (6) leveraging outputs from objective metrics and threshold-based clustering to automatically generate personalized skill reports, complete with strengths, weaknesses, and targeted suggestions, by integrating large language models (LLMs). Ultimately, such efforts may help alleviate global workforce shortages and democratize access to advanced surgical training by providing scalable, objective, and explainable AI-based tools.

## References

[1] Center, V.U.M.: Global Surgery Facts and Figures. Vanderbilt University Medical Center. Accessed: April 28, 2024 (2024)

[2] UCSF Department of Surgery: Global Surgery 2030: evidence and solutions for achieving health, welfare, and economic development. Last accessed: March 1, 2025 (Accessed 2025). https://surgery.ucsf.edu/sites/default/files/umbraco/ media/8065752/Overview\_GS2030.pdf

[3] Augestad, K.M., Butt, K., Ignjatovic, D., Keller, D.S., Kiran, R.: Video-based coaching in surgical education: a systematic review and meta-analysis. Surgical Endoscopy 34(2), 521–535 (2019) https://doi.org/10.1007/s00464-019-07265-0

[4] MBZUAI: AI-Driven Surgical Skill Optimization. https://mbzuai.ac.ae/ the-node/ai-talks/ai-talk/ai-driven-surgical-skill-optimization/. Accessed: 2024-12-18 (2024)

[5] Nathan, M., al.: Intraoperative adverse events can be compensated by technical performance in neonates and infants after cardiac surgery: A prospective study. The Journal of Thoracic and Cardiovascular Surgery 142(5), 1098–11075 (2011) https://doi.org/10.1016/j.jtcvs.2011.07.003

[6] Papaspyros, S.C., Javangula, K.C., Prasad Adluri, R.K., O'Regan, D.J.: Briefing and debriefing in the cardiac operating room. analysis of impact on theatre team attitude and patient safety. Interactive CardioVascular and Thoracic Surgery 10(1), 43–47 (2010) https://doi.org/10.1510/icvts.2009.217356

[7] Tevis, S.E., Kennedy, G.D.: Postoperative complications and implications on patient-centered outcomes. Elsevier BV (2013). https://doi.org/10.1016/j.jss. 2013.01.032 . http://dx.doi.org/10.1016/j.jss.2013.01.032

[8] Semel, M.E., Lipsitz, S.R., Funk, L.M., Bader, A.M., Weiser, T.G., Gawande, A.A.: Rates and patterns of death after surgery in the United States, 1996 and 2006. Elsevier BV (2012). https://doi.org/10.1016/j.surg.2011.07.021 . http://dx. doi.org/10.1016/j.surg.2011.07.021

[9] Weiser, T.G., al.: An estimation of the global volume of surgery: a modelling strategy based on available data. The Lancet 372(9633), 139–144 (2008) https: //doi.org/10.1016/s0140-6736(08)60878-8

[10] Regenbogen, S.E., Greenberg, C.C., Studdert, D.M., Lipsitz, S.R., Zinner,

M.J., Gawande, A.A.: Patterns of technical error among surgical malpractice claims. Annals of Surgery 246(5), 705–711 (2007) https://doi.org/10.1097/sla. 0b013e31815865f8

[11] Gallant, J., Brelsford, K., Sharma, S., Grantcharov, T., Langerman, A.: Patient perceptions of audio and video recording in the operating room. Annals of Surgery 276(6), 1057–1063 (2021)

[12] Levin, M., McKechnie, T., Kruse, C.C., Aldrich, K., Grantcharov, T.P., Langerman, A.: Surgical data recording in the operating room: a systematic review of modalities and metrics. British Journal of Surgery 108(6), 613–621 (2021)

[13] Theator: Three Benefits of Recording Surgical Videos (2024). https://theator.io/ blog/benefits-of-recording-surgical-videos/

[14] Cheikh Youssef, S., Haram, K., Noël, J., Patel, V., Porter, J., Dasgupta, P., Hachach-Haram, N.: Evolution of the digital operating room: the place of video technology in surgery. Springer (2023). https://doi.org/10.1007/ s00423-023-02830-7. http://dx.doi.org/10.1007/s00423-023-02830-7

[15] Awshah, S., Bowers, K., Eckel, D.T., Diab, A.F., Ganam, S., Sujka, J., Docimo, S., DuCoin, C.: Current trends and barriers to video management and analytics as a tool for surgeon skilling. Springer (2024). https://doi.org/10.1007/ s00464-024-10754-6 . http://dx.doi.org/10.1007/s00464-024-10754-6

[16] Eckhoff, J.A., Rosman, G., Altieri, M.S., Speidel, S., Stoyanov, D., Anvari, M., Meier-Hein, L., März, K., Jannin, P., Pugh, C., Wagner, M., Witkowski, E., Shaw, P., Madani, A., Ban, Y., Ward, T., Filicori, F., Padoy, N., Talamini, M., Meireles, O.R.: SAGES consensus recommendations on surgical video data use, structure, and exploration (for research in artificial intelligence, clinical quality improvement, and surgical education). Springer (2023). https://doi.org/10.1007/ s00464-023-10288-3 . http://dx.doi.org/10.1007/s00464-023-10288-3

[17] Ahmadi, M.J., Allahkaram, M.S., Abdi, P., Mohammadi, S.-F., D. Taghirad, H.a.: Image processing and machine vision in surgery and its training. Journal of Control 17(2) (2023) https://doi.org/10.61186/joc.17.2.25 http://joc.kntu.ac.ir/article-1-999-en.pdf

[18] Funke, I., Mees, S.T., Weitz, J., Speidel, S.: Video-based surgical skill assessment using 3D convolutional neural networks. Springer (2019). https://doi.org/ 10.1007/s11548-019-01995-1. http://dx.doi.org/10.1007/s11548-019-01995-1

[19] Gao, Y., Vedula, S.S., Reiley, C.E., Ahmidi, N., Varadarajan, B., Lin, H.C., Tao, L., Zappella, L., Béjar, B., Yuh, D.D., Chen, C.C.G., Vidal, R., Khudanpur, S., Hager, G.D.: The jhu-isi gesture and skill assessment working set (jigsaws): A surgical activity dataset for human motion modeling. In: Modeling and Monitoring of Computer Assisted Interventions (M2CAI) – MICCAI Workshop (2014)

[20] Kitaguchi, D., Takeshita, N., Matsuzaki, H., Igaki, T., Hasegawa, H., Ito, M.: Development and Validation of a 3-Dimensional Convolutional Neural Network for Automatic Surgical Skill Assessment Based on Spatiotemporal Video Analysis. American Medical Association (AMA) (2021). https: //doi.org/10.1001/jamanetworkopen.2021.20786. http://dx.doi.org/10.1001/ jamanetworkopen.2021.20786

[21] Zia, A., Guo, L., Zhou, L., Essa, I., Jarc, A.: Novel evaluation of surgical activity recognition models using task-based efficiency metrics. Springer (2019). https://doi.org/10.1007/s11548-019-02025-w . http://dx.doi. org/10.1007/s11548-019-02025-w

[22] Lajkó, G., Nagyné Elek, R., Haidegger, T.: Endoscopic Image-Based Skill Assessment in Robot-Assisted Minimally Invasive Surgery. MDPI AG (2021). https: //doi.org/10.3390/s21165412 . http://dx.doi.org/10.3390/s21165412

[23] Phacoemulsification Devices Market Analysis - US, Canada, Germany, UK, China - Size and Forecast 2024-2028. Technavio. Accessed: April 28, 2024 (2023)

[24] Flaxman, S.R., Bourne, R.R.A., Resnikoff, S., Ackland, P., Braithwaite, T., Cicinelli, M.V., Das, A., Jonas, J.B., Keeffe, J., Kempen, J.H., Leasher, J., Limburg, H., Naidoo, K., Pesudovs, K., Silvester, A., Stevens, G.A., Tahhan, N., Wong, T.Y., Taylor, H.R., Bourne, R., Ackland, P., Arditi, A., Barkana, Y., Bozkurt, B., Braithwaite, T., Bron, A., Budenz, D., Cai, F., Casson, R., Chakravarthy, U., Choi, J., Cicinelli, M.V., Congdon, N., Dana, R., Dandona, R., Dandona, L., Das, A., Dekaris, I., Del Monte, M., Deva, J., Dreer, L., Ellwein, L., Frazier, M., Frick, K., Friedman, D., Furtado, J., Gao, H., Gazzard, G., George, R., Gichuhi, S., Gonzalez, V., Hammond, B., Hartnett, M.E., He, M., Hejtmancik, J., Hirai, F., Huang, J., Ingram, A., Javitt, J., Jonas, J., Joslin, C., Keeffe, J., Kempen, J., Khairallah, M., Khanna, R., Kim, J., Lambrou, G., Lansingh, V.C., Lanzetta, P., Leasher, J., Lim, J., Limburg, H., Mansouri, K., Mathew, A., Morse, A., Munoz, B., Musch, D., Naidoo, K., Nangia, V., Palaiou, M., Parodi, M.B., Pena, F.Y., Pesudovs, K., Peto, T., Quigley, H., Raju, M., Ramulu, P., Rankin, Z., Resnikoff, S., Reza, D., Robin, A., Rossetti, L., Saaddine, J., Sandar, M., Serle, J., Shen, T., Shetty, R., Sieving, P., Silva, J.C., Silvester, A., Sitorus, R.S., Stambolian, D., Stevens, G., Taylor, H., Tejedor, J., Tielsch, J., Tsilimbaris, M., Meurs, J., Varma, R., Virgili, G., Wang, Y.X., Wang, N.-L., West, S., Wiedemann, P., Wong, T., Wormald, R., Zheng, Y.: Global causes of blindness and distance vision impairment 1990–2020: a systematic review and meta-analysis. Lancet Glob. Health 5(12), 1221–1234 (2017)

[25] Khairallah, M., Kahloun, R., Bourne, R., Limburg, H., Flaxman, S.R., Jonas, J.B., Keeffe, J., Leasher, J., Naidoo, K., Pesudovs, K., Price, H., White, R.A., Wong, T.Y., Resnikoff, S., Taylor, H.R., Vision Loss Expert Group of the Global Burden of Disease Study: Number of people blind or visually impaired by cataract worldwide and in world regions, 1990 to 2010. Invest. Ophthalmol. Vis. Sci.

[26] Lacmanović Lončar, V.: The resident surgeon phacoemulsification learning curve at clinical department of ophthalmology, sestre milosrdnice university hospital center. Acta Clin. Croat., 549–554 (2016)

[27] Neufeld, A., Hanson, L.L., Pettey, J.: Teaching in the operating room: trends in surgical skills transfer in ophthalmology. Ann. Eye Sci. 2, 41–41 (2018)

[28] Bharucha, K.M., Adwe, V.G., Hegade, A.M., Deshpande, R.D., Deshpande, M.D., Kalyani, V.K.S.: Evaluation of skills transfer in short-term phacoemulsification surgery training program by international council of ophthalmology -ophthalmology surgical competency assessment rubrics (ICO-OSCAR) and assessment of efficacy of ICO-OSCAR for objective evaluation of skills transfer. Indian J. Ophthalmol. 68(8), 1573–1577 (2020)

[29] Prakash, G., Jhanji, V., Sharma, N., Gupta, K., Titiyal, J.S., Vajpayee, R.B.: Assessment of perceived difficulties by residents in performing routine steps in phacoemulsification surgery and in managing complications. Can. J. Ophthalmol. 44(3), 284–287 (2009)

[30] Golnik, K.C., Beaver, H., Gauba, V., Lee, A.G., Mayorga, E., Palis, G., Saleh, G.M.: Cataract surgical skill assessment. Ophthalmology 118(2), 427–15 (2011)

[31] Cremers, S.L., Ciolino, J.B., Ferrufino-Ponce, Z.K., Henderson, B.A.: Objective assessment of skills in intraocular surgery (OASIS). Ophthalmology 112(7), 1236– 1241 (2005)

[32] Cremers, S.L., Lora, A.N., Ferrufino-Ponce, Z.K.: Global rating assessment of skills in intraocular surgery (GRASIS). Ophthalmology 112(10), 1655–1660 (2005)

[33] Saleh, G.M., Gauba, V., Mitra, A., Litwin, A.S., Chung, A.K.K., Benjamin, L.: Objective structured assessment of cataract surgical skill. Arch. Ophthalmol. 125(3), 363–366 (2007)

[34] Schoeffmann, K., Taschwer, M., Sarny, S., Münzer, B., Primus, M.J., Putzgruber, D.: Cataract-101: video dataset of 101 cataract surgeries. In: César, P., Zink, M., Murray, N. (eds.) Proceedings of the 9th ACM Multimedia Systems Conference, MMSys 2018, Amsterdam, The Netherlands, June 12-15, 2018, pp. 421–425. ACM, ??? (2018). https://doi.org/10.1145/3204949.3208137 . https: //doi.org/10.1145/3204949.3208137

[35] Adwe, V., Bharucha, K., Hegade, A., Deshpande, R., Deshpande, M., Kalyani, V.S.: Evaluation of skills transfer in short-term phacoemulsification surgery training program by international council of ophthalmology -ophthalmology surgical competency assessment rubrics (ico-oscar) and assessment of efficacy of ico-oscar

for objective evaluation of skills transfer. Indian Journal of Ophthalmology 68(8), 1573 (2020) https://doi.org/10.4103/ijo.ijo\_2058\_19

[36] Dean, W.H., Murray, N.L., Buchan, J.C., Golnik, K., Kim, M.J., Burton, M.J.: Ophthalmic simulated surgical competency assessment rubric for manual smallincision cataract surgery. Journal of Cataract and Refractive Surgery 45(9), 1252– 1257 (2019) https://doi.org/10.1016/j.jcrs.2019.04.010

[37] Puri, S., Sikder, S.: Cataract surgical skill assessment tools. Journal of Cataract and Refractive Surgery 40(4), 657–665 (2014) https://doi.org/10.1016/j.jcrs. 2014.01.027

[38] Saleh, G.M.: Objective structured assessment of cataract surgical skill. Archives of Ophthalmology 125(3), 363 (2007) https://doi.org/10.1001/archopht.125.3.363

[39] Golnik, K.C., Beaver, H., Gauba, V., Lee, A.G., Mayorga, E., Palis, G., Saleh, G.M.: Cataract surgical skill assessment. Ophthalmology 118(2), 427–4275 (2011) https://doi.org/10.1016/j.ophtha.2010.09.023

[40] Cremers, S.L., Lora, A.N., Ferrufino-Ponce, Z.K.: Global rating assessment of skills in intraocular surgery (grasis). Ophthalmology 112(10), 1655–1660 (2005) https://doi.org/10.1016/j.ophtha.2005.05.010

[41] Ahmadi, M.J., Gandomi, I., Abdi, P., et al.: Cataract-lmm large-scale multisource multi-task benchmark for deep learning in surgical video analysis. Scientific Data 13, 1189 (2026) https://doi.org/10.1038/s41597-026-07464-0

[42] Dreyfus, S.E.: The Five-Stage Model of Adult Skill Acquisition. Accessed: 2025-03-04 (2004). https://www.bumc.bu.edu/facdev-medicine/files/2012/03/ Dreyfus-skill-level.pdf

[43] Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K.V., Carion, N., Wu, C.-Y., Girshick, R., Dollár, P., Feichtenhofer, C.: SAM 2: Segment Anything in Images and Videos (2024). https://arxiv.org/abs/2408.00714

[44] Khanam, R., Hussain, M.: YOLOv11: An Overview of the Key Architectural Enhancements (2024). https://arxiv.org/abs/2410.17725

[45] Ahmadi, M.J., Allahkaram, M.S., Rashvand, A., Lotfi, F., Abdi, P., Motaharifar, M., Mohammadi, S.F., Taghirad, H.D.: Aras-farabi experimental framework for skill assessment in capsulorhexis surgery. In: 2021 9th RSI International Conference on Robotics and Mechatronics (ICRoM), pp. 385–390. IEEE, ??? (2021). https://doi.org/10.1109/icrom54204.2021.9663494. http://dx.doi.org/10. 1109/ICRoM54204.2021.9663494

[46] Lafouti, M., Ahmadi, M.J., Allahkaram, M.S., Gandomi, I., Lotfi, F., Mohammadzadeh, M., Abdi, P., Taghirad, H.D.: Surgical instrument tracking for capsulorhexis eye surgery based on siamese networks. In: 2022 10th RSI International Conference on Robotics and Mechatronics (ICRoM), pp. 196–201. IEEE, ???(2022). https://doi.org/10.1109/icrom57054.2022.10025355. http://dx.doi. org/10.1109/ICRoM57054.2022.10025355

[47] Gandomi, I., Vaziri, M., Ahmadi, M.J., Reyhaneh Hadipour, M., Abdi, P., Taghirad, H.D.: A deep dive into capsulorhexis segmentation: From dataset creation to sam fine-tuning. In: 2023 11th RSI International Conference on Robotics and Mechatronics (ICRoM), pp. 675–681. IEEE, ??? (2023). https://doi.org/10. 1109/icrom60803.2023.10412370. http://dx.doi.org/10.1109/ICRoM60803.2023. 10412370

[48] R??fer, F., Schr??der, A., Erb, C.: White-to-white corneal diameter: Normal values in healthy humans obtained with the orbscan ii topography system. Cornea 24(3), 259–261 (2005) https://doi.org/10.1097/01.ico.0000148312.01805.53

[49] Hwang, H., Lim, J., Kinnaird, C., Nagy, A.G., Panton, O.N.M., Hodgson, A.J., Qayumi, K.A.: Correlating motor performance with surgical error in laparoscopic cholecystectomy. Surgical Endoscopy 20(4), 651–655 (2005) https://doi.org/10. 1007/s00464-005-0370-8

[50] Soleymani, A., Sadat Asl, A.A., Yeganejou, M., Dick, S., Tavakoli, M., Li, X.: Surgical skill evaluation from robot-assisted surgery recordings. In: 2021 International Symposium on Medical Robotics (ISMR), pp. 1–6. IEEE, ??? (2021). https://doi.org/10.1109/ismr48346.2021.9661527. http://dx.doi.org/10. 1109/ISMR48346.2021.9661527

[51] Kletz, S., Schoeffmann, K., Leibetseder, A., Benois-Pineau, J., Husslein, H.: Instrument Recognition in Laparoscopy for Technical Skill Assessment, pp. 589– 600. Springer, ??? (2019). https://doi.org/10.1007/978-3-030-37734-2\_48 . http: //dx.doi.org/10.1007/978-3-030-37734-2\_48

[52] Nau, P., Worden, E., Lehmann, R., Kleppe, K., Mancini, G.J., Mancini M.L., Ramshaw, B.: Global assessment of surgical skills (gass): validation of a new instrument to measure global technical safety in surgical procedures. Surgical Endoscopy 37(10), 7964–7969 (2023) https://doi.org/10.1007/ s00464-023-10116-8

[53] Aghazadeh, F., Zheng, B., Tavakoli, M., Rouhani, H.: Motion smoothness-based assessment of surgical expertise: The importance of selecting proper metrics. Sensors 23(6), 3146 (2023) https://doi.org/10.3390/s23063146

[54] Hutchinson, K., Chen, K., Alemzadeh, H.: Towards Interpretable Motion-level Skill Assessment in Robotic Surgery. arXiv (2023). https://doi.org/10.48550/ ARXIV.2311.05838 . https://arxiv.org/abs/2311.05838

[55] Kim, T.S., O'Brien, M., Zafar, S., Hager, G.D., Sikder, S., Vedula, S.S.: Objective assessment of intraoperative technical skill in capsulorhexis using videos of cataract surgery. International Journal of Computer Assisted Radiology and Surgery 14(6), 1097–1105 (2019) https://doi.org/10.1007/s11548-019-01956-8

[56] Ghamsarian, N., El-Shabrawi, Y., Nasirihaghighi, S., Putzgruber-Adamitsch, D., Zinkernagel, M., Wolf, S., Schoeffmann, K., Sznitman, R.: Cataract-1k dataset for deep-learning-assisted analysis of cataract surgery videos. Scientific Data 11(1) (2024) https://doi.org/10.1038/s41597-024-03193-4