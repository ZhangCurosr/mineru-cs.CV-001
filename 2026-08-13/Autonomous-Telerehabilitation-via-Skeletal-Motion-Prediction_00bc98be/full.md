# Autonomous Telerehabilitation via Skeletal Motion Prediction and Joint-Level Performance Assessment

Lara Pereira, Joao Ruivo Paulo, Pedro Santos and Paulo Peixoto˜

Institute of Systems and Robotics

Department of Electrical and Computer Engineering, University of Coimbra, Portugal {lara.pereira,peixoto,jpaulo}@isr.uc.pt

Abstract—Autonomous rehabilitation systems must not only recognize human motion but also provide structured feedback to support users without continuous therapist supervision. This paper presents a telerehabilitation pipeline that integrates skeleton-based exercise quality assessment and short-term motion prediction into a two-module system operating on markerfree RGB video. A self-attentive Bidirectional LSTM performs exercise quality classification using MMD-NCA metric learning, while a graph-based motion prediction module computes perjoint position errors between predicted and observed poses, generating spatially localized deviation signals. Each module is evaluated independently on established benchmarks: the classifier achieves 96.45% mean-class accuracy on squat sequences from the PROZIS dataset, and the adopted STARS predictor achieves a mean MPJPE of 75.8 mm at 560 ms on Human3.6M, outperforming graph and recurrent baselines across all prediction horizons. The framework is designed for eventual deployment in assistive robotics and home-based rehabilitation contexts; end-to-end integration and clinical validation are important directions for future work. By combining motion recognition and prediction in a single system, this work contributes a step toward autonomous, feedback-driven telerehabilitation, for more accessible and scalable rehabilitation solutions.

Index Terms—telerehabilitation, exercise quality assessment, human motion prediction, human–robot interaction, skeletonbased action recognition

## I. INTRODUCTION

In the past decade, healthcare has undergone a significant transformation driven by digital technologies, particularly in the domain of physical rehabilitation [1]. With the rising prevalence of musculoskeletal conditions and the growing demand for long-term care, telerehabilitation has emerged as a scalable solution that enables patients to perform therapy remotely, reducing travel costs and increasing accessibility [2].

However, current telerehabilitation systems largely rely on passive video guidance and lack the ability to provide structured, personalized feedback. This limitation restricts their effectiveness, as patients may perform exercises incorrectly without correction, potentially reducing therapeutic outcomes. In contrast, human therapists continuously observe, evaluate, and guide patient movements, highlighting the need for automated systems that can detect movement deviations and communicate them to the user, even though fully replicating the therapist’s clinical reasoning remains a long-term goal.

In the context of human–robot interaction and assistive robotics, this challenge translates into enabling intelligent systems, such as companion or social robots, to perceive, interpret, and respond to human motion in a meaningful and user-centered manner. For such systems to support rehabilitation effectively, they must not only assess whether an exercise is performed correctly but also anticipate how the movement should evolve over time. This predictive capability allows the system to compare expected and observed motion, enabling spatially localized, joint-level feedback that can be directly communicated to the user.

To address this, we propose a machine learning-based pipeline for telerehabilitation support that operates on marker-free RGB video. The framework integrates exercise quality assessment and short-term motion prediction, enabling the system to generate both holistic (correct / incorrect) labels and spatially localized deviation signals at the joint level. Each module is validated on established benchmarks, and their combined design is intended to support future deployment in human-centered robotic assistance scenarios.

The contributions of this work are:

• An integrated two-module pipeline for telerehabilitation support that combines skeleton-based exercise quality assessment and short-term motion prediction from marker-free RGB video. The primary contribution is the integration of holistic quality classification and jointlevel deviation signals into a single system design, intended as a component of a future human–robot rehabilitation system.

• Evaluation of MMD-NCA distributional metric learning on an annotated rehabilitation exercise dataset, achieving 96.45% mean-class accuracy on squat sequences.

• A joint-level position error (JPE) formulation and colorcoded visualization scheme for spatially localized rehabilitation feedback.

• Characterization of data requirements for generalizing skeleton-based quality classifiers across exercise types.

We position this work as establishing the technical feasibility of the individual perception and feedback-generation components required for an autonomous rehabilitation coach, rather than as a complete, evaluated human-robot interaction system. Integration with a robot or interactive front-end, and evaluation with users and therapists, are identified as the necessary next steps toward the human-centered scenario motivating this work.

## II. RELATED WORK

Automated assessment of rehabilitation exercise quality from skeleton or video data has been studied using both rulebased and learning-based approaches. Deep learning methods have largely replaced traditional signal processing techniques because of their ability for end-to-end feature learning and adaptability to inter-subject variability. A key challenge is that human motion sequences of the same exercise type can vary substantially in length and tempo, requiring architectures or preprocessing strategies that are robust to these temporal variations.

Bidirectional LSTMs have shown strong performance on sequential pose data by capturing both forward and backward temporal context. A persistent challenge in motion sequence comparison is that direct distance measures do not account for temporal misalignment, while Dynamic Time Warping (DTW) [3], [4] addresses alignment but struggles with subtle temporal variations and scales poorly to high-dimensional data [4], [5].

Metric learning approaches address this by learning an embedding in which motion sequences from the same class cluster together. Hadsell et al. [6] used Siamese networks with contrastive loss, but they only exploit pairwise relationships. Triplet loss [7] improved upon this by considering anchor, positive, and negative samples jointly, but it requires careful negative mining and sensitive margin hyperparameter tuning. Neighborhood Components Analysis (NCA) and its distributional extension, Maximum Mean Discrepancy NCA (MMD-NCA) [5], eliminate these fixed margins by normalizing over negative classes in the batch. This progression provides a robust foundation for handling variable-length action recognition without the drawbacks of explicit timewarping algorithms.

In motion prediction, with a continued focus on skeleton data, early Recurrent Neural Network (RNN)-based approaches to human motion prediction established key architectural patterns. For example, Martinez et al. [8] framed short-term prediction as a sequence-to-sequence problem, using separate encoders and decoders to improve alignment between training and inference protocols. However, Graph Convolutional Networks (GCNs) were subsequently proposed for skeleton-based understanding by Yan et al. [9], proving highly effective at encoding joint connectivity via learnable adjacency matrices. Building on this, Mao et al. [10] established a strong baseline using Discrete Cosine Transform (DCT)-domain graph networks (LTD) with explicit trajectory dependencies. Recent advancements have focused on increasing the efficiency and diversity of these predictions. Architectures like Space-Time-Separable GCN (STS-GCN) [11] introduced factorized spatial and temporal processing to reduce computational overhead, while methods such as Spatial-

Temporal Anchor-Based Sampling (STARS) [12] incorporated generative modeling to capture the inherent multimodal nature of future human motion.

Despite rapid advancements in individual tasks, few works integrate both action recognition and motion prediction into a single pipeline for rehabilitation feedback. This gap motivates the present work.

## III. PIPELINE

The proposed approach consists of three main modules, as shown in Fig. 1. An off-the-shelf human pose estimator (MediaPipe or OpenPose) converts an input RGB sequence into a time series of 3D skeleton joint coordinates. The skeleton is normalized relative to the hip joint at each frame, removing absolute position and making the representation invariant to subject placement within the camera frame. Variable-length repetition sequences are either subsampled to a maximum length or zero-padded to a minimum length before being passed to the two processing modules. Interpolation was also evaluated but yielded marginally inferior results (≤2 percentage points in mean-class accuracy) compared to zeropadding.

The normalized skeleton sequence is processed by two parallel modules. The exercise quality classifier (Section IV) operates on full-repetition sequences and produces a binary holistic quality label (correct or incorrect) for each repetition. The motion predictor (Section V) uses a short context window of 10 frames (400 ms) and predicts the next 5–25 frames (200–1000 ms). The predicted joint positions are compared to the observed ground truth using a perjoint Euclidean error, generating a spatially localized error map. This map is thresholded into four color categories: green (low error), yellow (moderate), orange (elevated), and red (high deviation), and rendered as an overlay on a 3D skeleton visualization. Joints irrelevant to the exercise under assessment (e.g., arms during a gait task) are masked to focus feedback on clinically relevant body parts.

A key design requirement of the proposed system is robustness to variations in exercise tempo, which are common in real-world rehabilitation scenarios, especially among older adults and users with limited mobility. To ensure consistent performance across different movement speeds, the BiL-STM module uses a self-attention mechanism that aggregates variable-length sequences into a fixed-length representation, enabling tempo-invariant assessment. In parallel, the graphbased prediction models use trainable temporal adjacency matrices to implicitly learn speed-adaptive motion patterns directly from data, eliminating the need for explicit temporal normalization or dynamic time warping. Together, these design choices support reliable performance and feedback across diverse user behaviors and execution styles.

## IV. EXERCISE QUALITY CLASSIFICATION

Exercise quality takes the 3D skeletons as input with the goal of classifying an exercise as correctly or incorrectly executed. Below we describe the adopted approach for this action recognition task.

![](images/978fb199cf2abf48564d1274ebfecb11b781eeca63d779bff748c88c1d9a9dde.jpg)  
Fig. 1. High-level pipeline of the proposed method.

## A. Layer-Normalised Bidirectional LSTM

Here, we adopted the Coskun et al.’s [5] approach using a BiLSTM with a self attention mechanism was chosen. The core building block is a layer-normalized LSTM (LNLSTM) that performs forward and backward passes over the input pose sequence $X ~ = ~ ( x _ { 1 } , \ldots , x _ { n } )$ , where $\boldsymbol { x } _ { t } ~ \in ~ \mathbb { R } ^ { J \times 3 }$ is the pose at timestep t. Layer normalization replaces batch normalization inside the LSTM cell, which is known to degrade performance in standard LSTM implementations [5]. The normalized hidden state is computed as:

$$
g _ { t } ^ { m } = \frac { 1 } { H } \sum _ { j } ^ { H } g _ { t } ^ { c , j } , \quad v _ { t } = \sqrt { \frac { 1 } { H } \sum _ { j } ^ { H } ( g _ { t } ^ { c , j } - g _ { t } ^ { m } ) ^ { 2 } }\tag{1}
$$

$$
\begin{array} { r } { h _ { t } = g _ { t } ^ { o } \odot \operatorname { t a n h } \Bigl ( \frac { \gamma } { v _ { t } } \odot ( g _ { t } ^ { c } - g _ { t } ^ { m } ) + \beta \Bigr ) } \end{array}\tag{2}
$$

where $\boldsymbol { g } _ { t } ^ { c }$ and $g _ { t } ^ { o }$ are the cell memory and output gate, $H { = } 1 2 8$ is the number of hidden units, and $\gamma , \beta \in \mathbb { R } ^ { H }$ are learnable scale and bias parameters. The forward and backward LNLSTM outputs are concatenated at each timestep as $\boldsymbol { s } _ { t } ~ = ~ \left[ s _ { t , f } ; ~ s _ { t , b } \right]$ , producing the sequence of states $S =$ $\{ s _ { 1 } , \ldots , s _ { n } \}$

## B. Self-Attention Mechanism

In the context of human motion sequences, certain poses intuitively convey more information than others. To prioritize the most relevant postures, we use a self-attention mechanism that, based on the work of Lin et al. [13], assigns a scalar importance weight to each timestep:

$$
\begin{array} { r } { r = W _ { s _ { 2 } } \operatorname { t a n h } ( W _ { s _ { 1 } } S ^ { T } ) , \quad a _ { i } = - \log \left( \frac { e ^ { r _ { i } } } { \sum _ { j } e ^ { r _ { i , j } } } \right) } \end{array}\tag{3}
$$

with weight matrices $W _ { s _ { 1 } } ~ \in ~ \mathbb { R } ^ { 2 0 0 \times 1 0 }$ and $W _ { s _ { 2 } } \in \mathbb { R } ^ { 1 0 \times 1 }$ where r has values in the range of [0,1] and is used to give different weights to $a _ { i } .$

The attended, fixed-length sequence embedding $S _ { C } = $ $\boldsymbol { r } \cdot \boldsymbol { a } \in \mathbb { R } ^ { 1 2 8 }$ is then passed to a classification head consisting of: $\mathrm { F C } ( 3 2 0 )  \mathrm { D r o p o u t } ( 0 . 5 )  \mathrm { B a t c h N o r m }  \mathrm { F C } ( 3 2 0 ) $ $\mathrm { B a t c h N o r m }  \mathrm { F C } ( 1 2 8 )  \mathrm { B a t c h N o r m }  L _ { 2 } \mathrm { - N o r m }$ , with ReLU activations on all but the final fully connected (FC) layer.

All square weight matrices are initialized with random orthogonal matrices; remaining matrices use a uniform distribution $( \mu { = } 0 , \sigma { = } 0 . 0 0 1 )$ .

## C. MMD-NCA Loss Function

The architecture uses a variation of the NCA, based on the Maximum Mean Discrepancy (MMD). The MMD-NCA loss [5] operates at the distributional level, measuring divergence between class embedding distributions via the MMD. Given sample sets $X { = } \{ x _ { 1 } , \ldots , x _ { m } \}$ and $Y = \{ y _ { 1 } , \dots , y _ { n } \}$ and a mixture of K Gaussian kernels k:

$$
\begin{array} { l } { { \displaystyle { \bf M } { \bf M } { \bf D } [ k , X , Y ] ^ { 2 } = \frac { 1 } { m ^ { 2 } } \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { m } k ( x _ { i } , x _ { j } ^ { \prime } ) } \ ~ } \\ { { \displaystyle ~ - \frac { 2 } { m n } \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } k ( x _ { i } , y _ { j } ) } \ ~ } \\ { { \displaystyle ~ + \frac { 1 } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } k ( y _ { i } , y _ { j } ^ { \prime } ) } \ ~ } \end{array}\tag{4}
$$

To differentiate between correct and incorrect exercise executions, the network uses the MMD-NCA loss proposed by Coskun et al. [5]. This function effectively reduces the overlap between category distributions in the embedding space while ensuring that samples from the same distribution remain as close as possible. Specifically, the final loss encourages the anchor embedding distribution to match the positive class distribution, while being separated from M randomly sampled negative classes:

$$
\mathcal { L } _ { \mathrm { M M D - N C A } } = \frac { \exp ( - \mathrm { M M D } [ k , f ( X ) , f ( X ^ { + } ) ] ) } { \sum _ { j = 1 } ^ { M } \exp ( - \mathrm { M M D } [ k , f ( X ) , f ( X _ { c j } ^ { - } ) ] ) }\tag{5}
$$

where X and $X ^ { + }$ denote motion samples belonging to the same action category, respectively. The term X. $X _ { c j } ^ { - }$ represents samples from one of M distinct, randomly selected categories $c _ { j } \in C$ , and $f ( \cdot )$ denotes the network’s mapping to the lowdimensional embedding space.

## D. Training

The model is trained with a batch size of 32 using the Adagrad optimizer and an initial learning rate of $1 0 ^ { - 3 }$ . A learning rate scheduler is applied, reducing the rate by a factor of 0.8 with a patience of 7 epochs across approximately 1000 training sequences. Binary cross-entropy serves as the classification objective alongside the MMD-NCA embedding loss. Training ran for up to 1000 epochs (with convergence typically observed before the 500th epoch) on an NVIDIA RTX 2060 GPU (≈4.2 s per epoch).

## V. SHORT-TERM MOTION PREDICTION

To further quantify the evaluation of human movement in a rehabilitation setting, another framework was proposed: a network that could estimate or predict the next x frames based on an average model, then compare them to the ground truth, allowing feedback at the joint level rather than just through an overall scoring function. For this purpose, two graph-based networks were chosen.

## A. Problem Formulation

Let $\mathbf { X } _ { \mathrm { i n } } = [ X _ { 1 } , \dots , X _ { T } ]$ denote the observed pose history, where $X _ { i } \in \mathbb { R } ^ { 3 \times V }$ encodes the 3D coordinates of V joints for the time sequence $i = 1 , . . . , T$ . The goal is to predict $\mathbf { X _ { \mathrm { o u t } } } = [ X _ { T + 1 } , \dots , X _ { T + K } ]$ , the poses for the next K frames. Both models use $T { = } 1 0$ context frames (400 ms) and are evaluated across two distinct prediction horizons: short-term predictions from K=2 to K=10 (80–400 ms), and long-term predictions from K=14 to K=25 (560–1000 ms).

## B. Space-Time-Separable GCN (STS-GCN)

STS-GCN [11] encodes the full observation as a spatiotemporal graph $\mathcal { G } = ( \nu , \mathcal { E } )$ with $T V$ nodes. The key insight is to factorize the full spatio-temporal adjacency matrix $A ^ { s t } \in$ $\mathbb { R } ^ { V T \times T V }$ into separate spatial and temporal components:

$$
\mathcal { H } ^ { l + 1 } = \sigma ( A ^ { s - l } A ^ { t - l } \mathcal { H } ^ { l } W ^ { l } )\tag{6}
$$

where $A _ { s } \in \mathbb { R } ^ { V \times V }$ captures joint-joint relations, $A _ { t } \in \mathbb { R } ^ { T \times T }$ captures time-time relations, and $W ^ { l }$ are the trainable convolutional weights at layer l. This factorization limits jointtime cross-talk, reducing the number of learnable parameters while preserving the expressiveness of the full space-time representation. Four encoder layers with batch normalization and residual connections are used. A Temporal Convolutional Network (TCN) decodes the latent representation into predicted future joint coordinates. Training minimized Mean Per Joint Position Error (MPJPE) loss using Adam (learning rate 0.001, decay factor 0.1 with patience 5 epochs), batch size 256, for 30 epochs on an RTX 2060 (20–30 minutes total).

## C. Spatial-Temporal Anchor-Based Sampling (STARS)

Following Xu et al. [12], we adopt the STARS framework, which extends the STS-GCN backbone with a generative model that decomposes the latent code into a stochastic component $z \sim p ( z )$ and a set of K learnable deterministic anchors $a = \{ a _ { k } \} _ { k = 1 } ^ { K }$ . Anchors are further factorized into $K _ { s }$ spatial anchors $\{ a _ { i } ^ { s } \}$ (controlling movement direction and posture) and $K _ { t }$ temporal anchors $\{ a _ { j } ^ { t } \}$ (controlling frequency and speed), yielding $K _ { s } \times K _ { t }$ compositional predictions:

$$
\hat { Y } _ { k } = \mathcal G ( a _ { i } ^ { s } + a _ { j } ^ { t } , z , X )\tag{7}
$$

The backbone operates in the DCT frequency domain using 8 STS-GCN layers (four standard and four pruned) with alternating channel dimensions $( 3 \to 1 2 8 \to 6 4 \to 1 2 8 \to 6 4 \to 1 2 8 \to 6 4 \to 1 2 8 \to 3 )$ . It utilizes cross-layer adjacency sharing and kinematic-tree-guided spatial pruning of the spatial adjacency matrix.

Three loss terms are combined during training: a reconstruction loss (minimizing the best prediction against ground truth), a diversity-enhancing loss (preventing anchor collapse), and motion constraint losses (history reconstruction error, pose prior, limb loss, and angle loss).

The learning rate is set to 0.001 and decays linearly after epoch 100. In this work, we use $K { = } 1$ to produce a single predicted trajectory per context window.

## D. Joint-Level Position Error

To generate spatially and temporally localized feedback, we compute a per-joint, per-frame position error (JPE). By omitting the averaging over both joints and time present in standard MPJPE, we isolate the specific error for each articulation at every individual time-step:

$$
\mathcal { L } _ { \mathrm { J P E } } ( v , k ) = \| \hat { x } _ { v k } - x _ { v k } \| _ { 2 }\tag{8}
$$

where $\hat { x } _ { v k }$ and $x _ { v k }$ are the predicted and observed 3D positions of joint v at timestep k. This yields a scalar error for each joint in each predicted frame, which is then mapped to four color thresholds for visualization. Joints not relevant to the exercise being assessed are masked in the final overlay.

## VI. DATASETS

To evaluate the proposed telerehabilitation framework, we consider multiple datasets that capture complementary aspects of the problem, including controlled motion prediction benchmarks and real-world rehabilitation exercise data. This combination allows us to assess both the technical performance of the underlying models and their applicability to practical, user-centered rehabilitation scenarios.

Specifically, we employ the Human3.6M dataset to benchmark motion prediction under standardized conditions, enabling comparison with prior work, and the CMU Motion Capture dataset to evaluate the robustness of the action recognition and metric learning components across diverse motion categories. In addition, we use the PROZIS Challenge dataset, which contains annotated fitness exercises performed by users with varying levels of expertise, to assess performance in a realistic telerehabilitation context.

Together, these datasets provide a comprehensive evaluation framework, spanning controlled laboratory settings and real-world exercise variability, and supporting the validation of both system-level performance and its relevance to assistive rehabilitation applications.

## A. Human3.6M

Human3.6M [14] is a large-scale 3D human motion capture dataset recorded at 50 Hz with a Vicon system, comprising 7 subjects performing 15 everyday actions. We use a 22-joint subset consistent with prior work [10], [11]. Two parameterizations are evaluated: 3D Cartesian coordinates (requiring no additional preprocessing) and 3D Euler angles converted to exponential map format using the preprocessing pipeline of Jain et al. [15], which avoids gimbal locking artifacts. Five subjects are used for training, one for validation, and one for testing.

## B. PROZIS Challenge Dataset

The PROZIS Challenge dataset [16], [17] is a proprietary collection of annotated fitness exercise videos gathered in 2019.

It covers five exercise types: squats, sit-ups, push-ups, jumping jacks, and burpees, performed by participants with varying fitness levels. Each repetition is labeled as correctly or incorrectly executed. 3D skeletal landmarks were extracted from video and normalized relative to the hip joint. Repetition boundaries were manually annotated by segmenting each sequence into up and down phases.

The squat class has the largest sample count (approximately 950 sequences), while other exercise types range from about 70 to 300 sequences. Sequences of variable length are zero-padded or subsampled to a fixed target length and split 70/15/15 into training, validation, and test sets. An important limitation is that the dataset lacks joint index metadata and explicit kinematic tree information, which prevents the use of graph-based architectures and restricts this dataset to the sequence-level BiLSTM classifier. This constraint is also why the classification and motion-prediction modules are evaluated on separate benchmarks in Section VII rather than jointly on a single dataset: PROZIS cannot currently support the graph-based predictor, and Human3.6M contains no annotated rehabilitation exercises. Closing this gap is identified as a primary direction for future work (Section VIII).

## C. CMU Dataset

The CMU Graphics Lab Motion Capture Database [18] is used to benchmark the action recognition and metric learning baselines. The raw sequence data is preprocessed by downsampling the original 120 Hz capture rate to 30 Hz and removing six redundant joints. To prevent gimbal locking artifacts during training, the remaining 3D Euler angles are converted to an exponential map representation [5], [19]. Of the 38 available action categories, a balanced split is used, with 19 categories for training and the remaining 19 distinct categories for testing.

## VII. RESULTS

This section evaluates the proposed telerehabilitation framework in terms of technical performance and suitability for real-world, user-centered rehabilitation scenarios. Given the system’s dual nature, we assess the exercise quality classification and motion prediction modules independently, as each was trained and evaluated on separate benchmarks optimized for its respective task. For the classification component, we focus on metrics that address class imbalance and decision-making reliability, while for motion prediction, we evaluate spatial accuracy across varying temporal horizons.

## A. Evaluation Metrics

Since the action recognition task is a classification task and the PROZIS dataset is unbalanced, it is important to select metrics that accurately represent the overall performance of the method. Therefore, we report mean-class accuracy (mCA) - the average per-class accuracy, which gives equal weight to both classes regardless of sample count - and the false positive rate (FPR) at fixed true positive rate (TPR) thresholds of 90%, 80%, and 70%, following the protocol proposed in [5]. A high-quality classifier exhibits low, tightly clustered FPR values across all three thresholds. For motion prediction, we report MPJPE in millimeters.

B. Baseline Method Selection for Exercise Quality Classification

Before applying MMD-NCA to the PROZIS classification task (Section VII.C), we summarize the baseline comparison presented by Coskun et al. [5], which motivates this choice of metric learning objective over temporal-alignment and pairwise/triplet-based alternatives. Table I presents FPR results on Human3.6M and CMU Mocap, comparing the MMD-NCA loss with established temporal alignment and metric learning baselines. MMD-NCA achieves an FPR-90 of 32.66 on CMU and 38.42 on Human3.6M. Compared to DTW-based methods, the FPR reduction varies considerably by dataset and threshold, ranging from approximately 2.6 percentage points (vs. DCTW, Human3.6M FPR-80) to 18.8 percentage points (vs. CTW, CMU FPR-70), with larger and more consistent gains on CMU Mocap. Compared to tripletbased approaches, the reduction is more modest: roughly 6–8.5 percentage points on CMU Mocap, but only 0.8– 4.4 percentage points on Human3.6M. The tight clustering of FPR values across the 90%, 80%, and 70% thresholds further indicates a well-calibrated classifier. Coskun et al. [5] further report that MMD-NCA achieves higher F1 score and Normalized Mutual Information than Triplet, Triplet+GOR, and N-Pair losses across embedding dimensions from 16 to

TABLE I  
FPR (%) AT FIXED TPR THRESHOLDS, AS REPORTED BY COSKUN et al. [5] - LOWER IS BETTER
<table><tr><td rowspan="2">Method</td><td colspan="3">CMU Mocap</td><td colspan="3">Human3.6M</td></tr><tr><td>90%</td><td>80%</td><td>70%</td><td>90%</td><td>80%</td><td>70%</td></tr><tr><td>DTW [20]</td><td>47.48</td><td>42.92</td><td>37.62</td><td>49.64</td><td>47.96</td><td>44.38</td></tr><tr><td>MDDTW [21]</td><td>44.60</td><td>39.07</td><td>34.04</td><td>49.72</td><td>45.87</td><td>44.51</td></tr><tr><td>CTW [22]</td><td>46.02</td><td>40.96</td><td>39.11</td><td>47.63</td><td>43.10</td><td>42.18</td></tr><tr><td>GDTW [23]</td><td>45.61</td><td>39.95</td><td>35.24</td><td>46.06</td><td>42.72</td><td>40.04</td></tr><tr><td>DCTW [24]</td><td>40.56</td><td>38.83</td><td>26.95</td><td>41.39</td><td>39.18</td><td>36.71</td></tr><tr><td>Triplet [25]</td><td>39.72</td><td>33.82</td><td>28.77</td><td>42.78</td><td>40.11</td><td>36.01</td></tr><tr><td>Triplet+GOR [26]</td><td>40.32</td><td>33.97</td><td>27.78</td><td>42.03</td><td>37.61</td><td>33.95</td></tr><tr><td>N-Pair [27]</td><td>40.11</td><td>32.35</td><td>26.16</td><td>40.46</td><td>39.56</td><td>36.52</td></tr><tr><td>MMD-NCA [5]</td><td>32.66</td><td>25.66</td><td>20.29</td><td>38.42</td><td>36.54</td><td>33.13</td></tr></table>

256 on both CMU and Human3.6M, reinforcing the FPRbased comparison above.

## C. Exercise Quality Classification on PROZIS Dataset

For the squat class the model achieves 96.45% mCA, with training and validation loss curves converging smoothly before epoch 200 and no signs of overfitting (see Fig. 2). The self-attention mechanism focuses on the most discriminative phases of the repetition, particularly the lowest point of the squat descent, consistent with the attention behavior reported in [5] for action retrieval. For the remaining exercises (situps, push-ups, and jumping jacks), the sample counts varied significantly, ranging from 70 to 300 sequences. Sit-ups showed mild underfitting and unstable training curves, while push-ups and jumping jacks failed to converge meaningfully, with validation accuracy collapsing to the class prior.

These failures are due to data scarcity and class imbalance rather than architectural limitations. The squat results show that the model is effective when sufficient data are available. Empirically, reliable convergence appears to require at least several hundred balanced sequences per exercise class. Inference per repetition takes approximately 20ms after extraction, which is compatible with per-repetition deployment once skeleton extraction is complete.

## D. Motion Prediction Backbone Selection

Table II summarizes the MPJPE benchmark as presented in [12] for STARS and STS-GCN, together with LTD-50-25 and ConvSeq2Seq, presented in [11], to motivate this architectural choice. Our contribution is the application of this predictor within the joint-level feedback pipeline described in Section VII.E.

Table II presents the MPJPE across all 15 Human3.6M actions. STARS consistently outperforms all baselines at every prediction horizon. For short-term prediction, STARS achieves 56.9 mm compared to 65.8 mm for STS-GCN, an improvement of 8.9 mm. For long-term prediction, STARS achieves 108.4 mm compared to 117.0 mm. Notably, STARS also outperforms LTD-50-25, which uses 50 context frames, while our method uses only 10 frames, demonstrating the advantage of the anchor-based generative formulation for long-horizon prediction. As presented in [12], STARS outperforms STS-GCN consistently across individual action categories, not only on average; the largest gains occur in high-variability, non-periodic actions such as Discussion and Posing, where the anchor-based generative formulation better captures multimodal upper-body motion, while periodic actions such as Walking show smaller but consistent improvements.

![](images/b575142e3e4c5a10c94b77924c18e6326e4ac3da55b82c062be68271ddeabdaa.jpg)

![](images/43be62e57d45d29c68e5d3b5aa266ee4b13388c783ce86ae2b70181fb967df5d.jpg)

![](images/3b4bf67276c0479fbb75c2f2a8b10f0bdc951d4bca9e2b858085262db305ffd3.jpg)

![](images/f3b0a3f75e873b1f48658fc522df1e754bc12c7ac175d8d26d49b0b6683fb0f3.jpg)  
(b) Sit-ups

(a) Squats  
![](images/724a65d24411399594c90c6bb3913eef340eac65b4c63566311603bbdb0b190c.jpg)

![](images/bad9b27c26526ed6adde19aaa4d3a26a9ff4f664bd1829880ea28abc2b0f46d0.jpg)

![](images/58c11a06d7ff999e4b031244d8f916b9537abc98735488efce06c721690bdba6.jpg)  
(c) Push-ups

![](images/c5c8aa78cfd2f48eb13e7ccbb0040fc2d0fb5f62a91f0654c1b4fddb69783d9b.jpg)  
(d) Jumping Jacks  
Fig. 2. Training and validation accuracy and loss curves on the PROZIS dataset. The squat class (a) shows smooth convergence due to a high sample count, while classes with fewer samples, such as push-ups (c) and jumping jacks (d), have difficulty converging.

TABLE II  
AVERAGE MPJPE (MM) ON HUMAN3.6M [11], [12] - LOWER IS BETTER
<table><tr><td rowspan="2">Method</td><td colspan="2">Short-term (ms)</td><td colspan="4">Long-term (ms)</td></tr><tr><td>80 160</td><td>320 400</td><td>560</td><td>720</td><td>880</td><td>1000</td></tr><tr><td>ConvSeq2Seq</td><td>16.6</td><td>33.3 53.9 61.2</td><td>90.7</td><td>104.7</td><td>116.7</td><td>124.2</td></tr><tr><td>LTD-50-25</td><td>11.2 23.4</td><td>47.9 58.9</td><td>79.6</td><td>93.6</td><td>105.2</td><td>112.4</td></tr><tr><td>STS-GCN</td><td>13.5 27.7</td><td>54.4 65.8</td><td>85.0</td><td>98.3</td><td>108.9</td><td>117.0</td></tr><tr><td>STARS</td><td></td><td>10.0 21.8 45.7 56.9</td><td>75.8</td><td>89.3</td><td></td><td>100.8 108.4</td></tr></table>

The choice of input parameterization significantly affects prediction accuracy: using raw 3D Cartesian coordinates instead of the exponential map representation reduces shortterm accuracy by approximately 23% and long-term accuracy by approximately 34% [11]. The exponential map avoids gimbal locking but requires inverse and forward kinematics functions, which add substantial preprocessing complexity. For the rehabilitation application, where prediction horizons are intentionally short and feedback resolution is at the joint rather than the sub-centimeter level, the Cartesian representation is a pragmatically acceptable trade-off.

## E. Qualitative Feedback Analysis

The per-joint error signal was qualitatively evaluated on a walking sequence from Human3.6M, using STS-GCN with 3D Cartesian input and a 25-frame prediction horizon (approximately 1 second), as shown in Fig. 3. During correctly performed linear walking, leg joints remained mostly in the green and yellow error bands for the first 15 frames, with error accumulating in the final frames as expected due to the open-loop nature of the prediction. This demonstrates that the feedback mechanism correctly highlights deviations only when they become genuinely significant, rather than producing noisy false alerts throughout the sequence.

Robustness to execution speed was also evaluated by subsampling the same walking sequence to 50% and 25% of its original frame rate, which increases the spatial distance between joints across consecutive frames. The predictor adapted its output to the altered temporal scale without retraining or explicit dynamic time warping, consistent with the tempoinvariance design goal described in Section III.

Context quality is essential: sequences that begin with a static pose result in significantly higher joint errors in the initial frames, as shown in Fig. 4. This occurs because the predictor lacks an active motion signal to condition on and defaults to a near-stationary prediction, which diverges immediately once the subject starts moving. Error rapidly decreases once sufficient active-movement context accumulates. This motivates buffering at least 10 frames (400 ms) of active movement before activating the feedback overlay, and suggests that triggering the feedback only after the first full repetition begins is a practical deployment strategy.

End-to-end inference time from preprocessed skeleton input to rendered 3D overlay is 2–5 s in Python on an RTX 2060, making the system suitable for per-repetition rather than per-frame feedback. Optimization via C++ or ONNX deployment is left for future work.

## VIII. CONCLUSION

We presented a two-module deep learning pipeline that combines exercise quality assessment with joint-level deviation signaling to support telerehabilitation. The pipeline uses a self-attentive BiLSTM classifier using MMD-NCA metric learning for holistic exercise quality classification, and a graph-based short-term motion predictor for spatially localized feedback. Evaluated on established benchmarks, the classifier achieves 96.45% mean-class accuracy on squat sequences from the PROZIS dataset, while the STARS predictor achieves 75.8 mm MPJPE at 560 ms on Human3.6M, outperforming graph and recurrent baselines across all prediction horizons. Both modules operate on marker-free RGB video without specialized hardware. The primary contribution is the integration of these two complementary signals – holistic quality labels and per-joint deviation maps – into system design intended for deployment on assistive and social robots.

The current work has four limitations that scope the present contribution honestly and define directions for future research.

First, the two modules are trained and evaluated on separate datasets, and no end-to-end evaluation of the complete recognition–prediction–feedback loop has been conducted. As noted in Section VI, this is in part a current data constraint: PROZIS lacks the kinematic-tree information required by the graph-based predictor, and Human3.6M contains no rehabilitation exercises or quality annotations. Second, the joint-level error signal (Section V.D) is derived from motionprediction error rather than therapist-annotated deviation ground truth. We present the JPE signal as a candidate feedback mechanism, with clinical validation identified as essential future work. Third, end-to-end inference takes 2– 5 seconds, restricting the system to per-repetition rather than frame-level or continuous feedback.

This study establishes the feasibility of the system by validating its individual components—a necessary first step before testing the complete setup with users, therapists, and robotic hardware. Building on this foundation, future work will focus on end-to-end validation for rehabilitation exercises. We plan to improve anatomical accuracy by integrating kinematic constraints, validate our joint-level feedback against physiotherapist-annotated data, and prospective evaluation with users and clinicians in a human-robot interaction setting.

## ACKNOWLEDGMENTS

This work was funded by national funds through FCT - Foundation for Science and Technology, I.P., under the grant UID/00048/2025 (DOI: 10.54499/UIDB/00048/2025).

## REFERENCES

[1] L. Rigamonti, U. V. Albrecht, C. Lutter, M. Tempel, B. Wolfarth, D. A. Back, and D. A. Back, “Potentials of digitalization in sports medicine: A narrative review,” Current Sports Medicine Reports, vol. 19, pp. 157–163, 4 2020.

[2] Y. Liao, A. Vakanski, and M. Xian, “A deep learning framework for assessing physical rehabilitation exercises,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 28, pp. 468–477, 2 2020.

[3] T. K. Vintsyuk, “Speech discrimination by dynamic programming,” Cybernetics, vol. 4, no. 1, pp. 52–57, 1968, russian original: Kibernetika 4(1):81–88 (1968).

[4] E. J. Keogh and M. J. Pazzani, “Derivative dynamic time warping,” Proceedings, pp. 1–11, 4 2001. [Online]. Available: /doi/pdf/10.1137/1.9781611972719.1?download=true

[5] H. Coskun, D. J. Tan, S. Conjeti, N. Navab, and F. Tombari, “Human motion analysis with deep metric learning,” Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), vol. 11218 LNCS, pp. 693–710, 8 2018. [Online]. Available: http://arxiv.org/abs/1807.11176

![](images/f95833a252ec1740b6e5966505687eedcbd272706c773c7b03439554407b24fe.jpg)

![](images/ebc90c8bdb7cb71fa1d542650ddd85dab041540a7e9aa2532cffe68dbb500778.jpg)

![](images/3700767e9428229a5770a026a3f4dcafab55188b85ed436c0b8327182cf0853e.jpg)

![](images/3a5b30a9703f208bd77214c1222af2d4dc131592252b31bbe0218cfcf24015d8.jpg)

![](images/d00fd1136d446f7d5e57a602cf830c0ca15d7c52335f5d80cd1a4f0de0aa599c.jpg)  
Fig. 3. Qualitative visual feedback generated by STS-GCN for a walking sequence. Five evenly spaced frames sampled from a continuous 25-frame (≈1s) prediction horizon are shown. The predicted skeleton is color-coded by per-joint position error (green: low; yellow: moderate; red: high) against the ground truth (dotted line). Accuracy remains high initially before open-loop error naturally accumulates in the later frames.

![](images/4d430df92893103134a0b707c465f220e113e873c43430b8a44288f2d5306a95.jpg)

![](images/67d1a77e705b2aa211eb291d65afe04e3f25d4def1c20d027899d61510660e70.jpg)

![](images/c3f662a2044ecaf577561a94eb4bdd1507c971db02e3a440e006beb3bc07d1e0.jpg)

![](images/ad00df6afb7741cf92ed9ef82dbd952f61b259ada365ebbc145fd35b34936caa.jpg)

![](images/8c8442efefc9829157bff3ff59b5ee6b671ed36370420430285fbe653f127416.jpg)  
Fig. 4. Effect of context quality on joint-level feedback. Five evenly spaced frames from a 25-frame (≈1s) prediction horizon are shown for a walking sequence that begins from a stationary pose. High initial errors (orange/red joints) decrease rapidly once active motion context accumulates, motivating a minimum buffer of 10 frames (400 ms) of active movement before activating the feedback overlay.

[6] R. Hadsell, S. Chopra, and Y. LeCun, “Dimensionality reduction by learning an invariant mapping,” Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, vol. 2, pp. 1735–1742, 2006. [Online]. Available: https://ieeexplore.ieee.org/document/1640964

[7] F. Schroff, D. Kalenichenko, and J. Philbin, “Facenet: A unified embedding for face recognition and clustering,” Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, vol. 07-12-June-2015, pp. 815–823, 10 2015. [Online]. Available: https://ieeexplore.ieee.org/document/7298682

[8] J. Martinez, M. J. Black, and J. Romero, “On human motion prediction using recurrent neural networks,” Proceedings - 30th IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, vol. 2017-January, pp. 4674–4683, 5 2017. [Online]. Available: http://arxiv.org/abs/1705.02445

[9] S. Yan, Y. Xiong, and D. Lin, “Spatial temporal graph convolutional networks for skeleton-based action recognition,” 32nd AAAI Conference on Artificial Intelligence, AAAI 2018, pp. 7444–7452, 1 2018. [Online]. Available: http://arxiv.org/abs/1801.07455

[10] W. Mao, M. Liu, M. Salzmann, and H. Li, “Learning trajectory dependencies for human motion prediction,” Proceedings of the IEEE International Conference on Computer Vision, pp. 9488–9496, 7 2020. [Online]. Available: http://arxiv.org/abs/1908.05436

[11] T. Sofianos, A. Sampieri, L. Franco, and F. Galasso, “Spacetime-separable graph convolutional network for pose forecasting,” Proceedings of the IEEE International Conference on Computer Vision, pp. 11 189–11 198, 10 2021. [Online]. Available: http: //arxiv.org/abs/2110.04573

[12] S. Xu, Y. X. Wang, and L. Y. Gui, “Diverse human motion prediction guided by multi-level spatial-temporal anchors,” Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), pp. 251–269, 2022.

[13] Z. Lin, M. Feng, C. N. dos Santos, M. Yu, B. Xiang, B. Zhou, and Y. Bengio, “A structured self-attentive sentence embedding,” 5th International Conference on Learning Representations, ICLR 2017 - Conference Track Proceedings, 3 2017. [Online]. Available: http://arxiv.org/abs/1703.03130

[14] C. Ionescu, D. Papava, V. Olaru, and C. Sminchisescu, “Human3.6m: Large scale datasets and predictive methods for 3d human sensing in natural environments,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 36, pp. 1325–1339, 2014.

[15] A. Jain, A. R. Zamir, S. Savarese, and A. Saxena, “Structural-rnn: Deep learning on spatio-temporal graphs,” Proceedings of the IEEE

Computer Society Conference on Computer Vision and Pattern Recognition, vol. 2016-December, pp. 5308–5317, 4 2016. [Online]. Available: http://arxiv.org/abs/1511.05298

[16] J. Batista, “Prozis challenge,” https://www.isr.uc.pt/index.php/projects/ past-projects?task=showprojects.show%28%29&idProject=219, 2019, online; accessed 2025.

[17] B. Ferreira, P. Menezes, and J. Batista, “Transformers for workout video segmentation,” Proceedings - International Conference on Image Processing, ICIP, pp. 3470–3474, 2022. [Online]. Available: https://ieeexplore.ieee.org/document/9897194

[18] “Carnegie mellon university - cmu graphics lab - motion capture library.” [Online]. Available: https://mocap.cs.cmu.edu/

[19] D. J. Sutherland, H.-Y. Tung, H. Strathmann, S. De, A. Ramdas, A. Smola, and A. Gretton, “Generative models and model criticism via optimized maximum mean discrepancy,” 5th International Conference on Learning Representations, ICLR 2017 - Conference Track Proceedings, 1 2021. [Online]. Available: http://arxiv.org/abs/1611. 04488

[20] T. K. Vintsyuk, “Speech discrimination by dynamic programming,” Cybernetics, vol. 4, pp. 52–57, 1 1968. [Online]. Available: https://link.springer.com/article/10.1007/BF01074755

[21] J. Mei, M. Liu, Y.-F. Wang, and H. Gao, “Learning a mahalanobis distance based dynamic time warping measure for multivariate time series classification.”

[22] F. Zhou and F. D. L. Torre, “Canonical time warping for alignment of human behavior.”

[23] F. Zhou and F. De la Torre, “Generalized canonical time warping,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 38, no. 2, pp. 279–294, 2016.

[24] G. Trigeorgis, M. A. Nicolaou, B. W. Schuller, and S. Zafeiriou, “Deep canonical time warping for simultaneous alignment and representation learning of sequences,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, pp. 1128–1138, 5 2018.

[25] F. Schroff, D. Kalenichenko, and J. Philbin, “Facenet: A unified embedding for face recognition and clustering,” Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, vol. 07-12-June-2015, pp. 815–823, 10 2015. [Online]. Available: https://ieeexplore.ieee.org/document/7298682

[26] X. Zhang, F. X. Yu, S. Kumar, and S. F. Chang, “Learning spread-out local feature descriptors,” IEEE International Conference on Computer Vision, vol. 2017-October, pp. 4605–4613, 12 2017.

[27] K. Sohn, “Improved deep metric learning with multi-class n-pair loss objective,” Neural Information Processing Systems, 2016.