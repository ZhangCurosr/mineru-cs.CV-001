# Graph Neural Assisted Actor-Critic for Latency-Efficient Edge Vision System

Alam Noor<sup>∗</sup>, Luis Almeida<sup>†</sup>, Kai Li<sup>‡</sup>, Jiyan Wu<sup>§</sup>, Miguel Gutierrez Gait´ an´ <sup>¶</sup> and Eduardo Tovar<sup>∗</sup> <sup>∗</sup>CISTER Research Center, Porto, Portugal. <sup>†</sup>Instituto de Telecomunicac¸oes, Faculdade de Engenharia,˜ Universidade do Porto, Portugal. <sup>‡</sup>Department of Information Technology, Kennesaw State University, USA. <sup>§</sup>OmniVision Technologies Inc. <sup>¶</sup>Department of Electrical Engineering, Pontificia Universidad Catolica de Chile,´ Santiago 7820436, Chile.

Abstract—UAV on-board vision systems are widely used for different activities, including monitoring in no-fly zones. In this case, the vision-equipped UAV streams a video to a ground server where an operator assists its activities. The latency of video transmission has a profound impact on the effectiveness of the operator assistance. However, most techniques available for video transmission still incur significant latency costs. In this paper, we propose a graph convolutional neural network-assisted (GCN-Assisted A2C) deep reinforcement learning (DRL) system model to find the optimal pixel-correlated area of a suspicious object. We combine the Lagrangian dual form with gradient descent to prevent lack of convergence and over- and under-penalization constraint violation during latency optimization. The proposed system model sends a sub-group pixel-correlated area of the frame from the UAV to the server rather than the transmission of the whole video frame. The proposed framework utilizes the GCN model to explore hidden representations of featurecorrelated groups of pixels. Moreover, the GCN supervises the A2C model, which selects a subgroup to enhance transmission latency, thus supervising the training of UAV actions in A2C. Experimental results show that GCN-assisted A2C reduces video frame transmission latency together with false detection rate in UAV vision systems over other DRL and state-of-the-art models.

Index Terms—Latency Optimization; GCN, Reinforcement Learning; UAVs; Mobile Edge Computing.

## I. INTRODUCTION

Unmanned Aerial Vehicles (UAVs) are used for a wide spectrum of mobile vision applications, such as augmented reality [1], trajectory planning [2], or visual surveillance systems (e.g., to detect intruders in no-fly zones) [3]–[5]. Many of these applications rely on real-time computer vision and require UAVs to detect both large and small objects in dynamic environments with high accuracy. For this purpose, an edge computing approach is common, having the UAVs stream video to a server in a ground station for assistance, getting important data from the server after preprocessing to identify objects with high accuracy [6] [7]. However, the transmission of video frames with high resolution imposes long delays that hinder real-time operation. Similarly, doing local video processing onboard the UAV is highly constrained due to limited computing resources, imposing long delays and/or low video resolution [8].

A promising solution to combine low latency and high accuracy detection following the edge computing paradigm is to offload to an edge server equipped with GPUs running deep learning models the processing of just specific areas of the video frames representing regions-of-interest (RoI) [9]– [11]. This reduces the amount of information to be transferred, leading to lower transmission time, thus reducing the latency of the Edge support. However, user-side parameters like image resolution and encoding rate and server-side parameters like neural network architecture have a significant impact on video analysis if we do not consider how the pixels of the RoI correlate with the pixels of the video frame background environment [10].

Wang et al. [9] studied a split-based object detection model (EdgeDuet) between UAVs and the Edge. The detection of small objects is offloaded to the server using an RoI of the frame and content-prioritized tiles. The RoI of the frame transmitted with pixel blocks potentially contains small objects of high resolution. The rest of the frame has large objects that can be detected with low resolution onboard the UAV. However, EdgeDuet is based on tile selection, with tiles having an area that is unrelated to the size of the small objects, thus potentially causing the transmission of more information than strictly needed, incurring unnecessarily longer transmission latency. Moreover, as shown in Fig.1, the use of tiles may also hinder the accuracy of server detection.

Yang et al. [11] introduced the Flexpatch model that allows for the detection of small objects with a small delay while also addressing occlusion, changes in object appearance, and the presence of additional objects. However, the Flexpatch model is based on optical flow with static cameras, thus unsuited for moving cameras onboard UAVs.

In this work we aim at managing the sizes of the subarea video frames to reduce the amount of information to be transferred and the associated transmission delay (Fig. 2). The key of the system model is a Graph Convolutional Network (GCN) that selects from a video frame the RoI of small objects and the group of neighboring pixel regions that are correlated with the centroid of the RoI. Then, it is important to select the group of pixels-correlated regions from the output of the GCN module that optimizes the latency subject to satisfying the accuracy of object identification.

For this purpose, we propose a novel GCN-assisted Advantage Actor Critic Deep Reinforcement Learning model (GCN-assisted-A2C) that is fast, accurate, and flexible to accommodate the movement of the onboard UAV camera. Our objective is to efficiently optimize and control the transmission delay in order to accurately identify intruding UAVs in a no-fly zone. We use the YOLO-Nano model to define the bounding box for detected UAVs. The GCN uses the bounding box center to find the group of pixels-correlated regions with a strong relationship with the RoI centroid. The A2C module of the proposed GCN-assisted-A2C model selects the actual group of pixels-correlated regions to be sent to the server. Finally, the server finds the object and sends back the information of accuracy and latency for the GCN-assisted-A2C model to optimize the latency efficiently, reducing the transmission delay by selecting a minimum group of pixelscorrelated regions that allows meeting a given target accuracy.

![](images/46ee1fc9f314d431792e9427e3e6ec8134b515600ce676c7a22258803c9b037f.jpg)  
Fig. 1. EdgeDuet non-optimal region selection for the transmission of small objects to an Edge server.

![](images/568ac29f516eec42ee367a0538656b79cb2acb18198550b7ac026261f27f608b.jpg)  
Fig. 2. Onboard UAV processing video to detect suspicious UAVs.

In summary, the main contribution of this paper is a novel GCN-assisted A2C model that encompasses the following two parts:

• A GCN that uses the bounding boxes of RoI produced by an onboard YOLO-Nano algorithm and creates clusters of feature-correlated groups of pixels and regions of hidden pixel relationships. The GCN model is trained with a SLIC segmentation of frames as ground truth for the groups of pixel regions.

• An A2C that uses the GCN as a state based on a group of pixels correlation values of future predictions to process the output, take action, and transmit it to the server. The A2C reward is based on the Lagrangian dual form for checking the sensitivity of the objective function. The aim is to reduce the risk of temporal difference learning error estimation for both the critic and the actor loss to update the policy. We use dual gradient descent to update the λ parameter and avoid lack of latency convergence and over- and under-penalization constraint violation during latency optimization.

We trained the proposed GCN-assisted-A2C model on a YouTube dataset to assess its real-world performance and compared it with three benchmark models: FlexPatch, Edge-Duet, and DRL. The proposed model achieved an inference latency of 45 ms, approximately half of what was achieved with the competing approaches. It also achieved an average precision of 70.72%, with a mean Intersection over Union (IoU) of 60.30%, indicating more accurate RoI selection and higher overall detection performance when compared against the benchmark methods.

Beyond the proposed GCN-assisted A2C, this work uses the following technologies in the end-to-end video pipeline:

• UAV side: An onboard camera combined with YOLO-Nano inference for real-time object detection. When the detection confidence falls below a predefined threshold, region-of-interest (RoI) image crops are extracted and transmitted to the edge server using an intra-coded image encoding scheme (H.264 Intra-only), enabling efficient and robust transmission of independent RoI frames.

• Edge server side: ESRGAN to regenerate missing contextual information, FFDNet for image denoising, Zero-DCE for low-light enhancement, followed by a largescale YOLO model for accurate object detection and refinement.

The rest of the paper is organized as follows. Section II discusses related work, Section III presents the mathematical system modeling, Section IV introduces the GCN-assisted A2C system model, and Section V validates the proposed model with an experimental study and discusses the results. Finally, Section VI concludes the paper.

## II. RELATED WORK

Monitoring no-fly zones for suspicious flying objects, particularly UAVs, can be achieved by resorting to a legal UAV that captures and streams live video from the air. However, this approach requires a significant volume of live video transmission from the UAV to an Edge server, consuming computing resources and incurring high transmission latency [12] [13]. To address these challenges, the previous system models might be categorized in different classes based on their primary approach to minimizing the transmission latency and improving the performance of real-time detection of UAVs.

Several researchers introduced low-latency video streaming system models to reduce video transmission latency and improve real-time detection. Dong et al. [8] introduced an ultralow latency frame delivery model using the UDP-based QUIC protocol and the widely compatible TS format, achieving latency below 200ms. Wang et al. [14] studied feature-based video transmission, using Lagrangian dual decomposition to optimize the transmission latency and semantic image segmentation for video feature selection. Qu et al. [15] studied UAVswarm-aided modeling based on a disaster response platform for the video transmission from UAVs to the ground as air-toground coordination, while Khan et al. [16] presented a public safety communication network model that uses observation UAVs to receive video streams from ground users in an affected area and transmit them to a nearby supporting ground station.

Some works focused on combining edge and cloud computation to optimize latency. Anurag et al. [17] studied edge and cloud collaborative work to use heavy deep learning models at the cloud and small models at the edge to fuse their predictions to optimize latency. Liu et al. [18] designed a multi-approach model that reduces latency by extending splitting-merging streaming and multi-link retransmission to reduce the packet loss. Yaqoob et al. [19] presented a DRL-based soft actorcritic model to improve the quality of experience of video transmission latency from a UAV to a ground server. Meanwhile, Song et al. [20] introduced an AI-driven, multipath transmission of a UAV live streaming system to optimize bandwidth aggregation and transmission reliability, while Wu et al. introduced a multiobjective control strategy learning for a UAV using a soft actor-critic DRL algorithm to manage the unstable transmission quality problem of videos in a dynamic topology network [21].

Other approaches focus on detecting objects efficiently by RoI and object-centric models to transmit video to relevant areas. Zhang et al. [10] worked on an instance segmentation technique executed on the Edge (EdgeIS) to segment an RoI and transfer it to a mobile device with low latency. The EdgeIS model detects objects and tracks their motions with mask transferring. Moreover, EdgeIs models utilize the contour-instructed edge inference acceleration scheme to reduce latency. Wang et al. [9] studied EdgeDuet, a split-based object detection framework between UAVs and the Edge, where regions of interest (RoIs) containing small objects are transmitted to an Edge server using content-prioritized tiles, while the remaining regions are processed onboard at lower resolution. Although effective, EdgeDuet relies on tile-based RoIs whose areas are not aligned with the actual size of small objects, potentially leading to unnecessary data transmission, increased latency, and reduced detection accuracy. Yang et al. [11] introduced Flexpatch, an RoI-based approach that enables low-latency detection of small objects by tracking and transmitting adaptive patches corresponding to regions of interest, while being robust to occlusion and appearance changes; however, Flexpatch relies on optical flow with static cameras, making it unsuitable for moving cameras onboard UAVs.

Among the works referred to above, only a few aim at reducing the video transmission latency for the specific purpose of flying objects real-time detection [8]–[11]. Some of the works [9]–[11], though, still stress the use of the network channel bandwidth transmitting areas of the video frames of no interest for flying object detection. This fact limits their capacity to reduce video latency and packet losses since there will be more packets transmitted than strictly needed.

In our work we pursue an approach based on identifying regions of interest in the video frames and transmit just these RoIs, leaving to the Edge server the task of accurately identifying small objects in those RoIs, even with poor video quality or environmental effects. The works that best relate to ours are EdgeDuet [9] and Flexpatch [11], which we will use for comparison. Our work is, to the best of our knowledge, the only one that trades off Average Precision (AP) with latency, using an A2C DRL approach to get the lowest latency (by means of selecting the smallest RoIs) that allows meeting a given AP target.

## III. MATHEMATICAL SYSTEM MODELING

We consider a sequence of encoded video frames $\{ F _ { t } \}$ of spatial size $W \times H$ pixels captured by a UAV. A light onboard detector (e.g., YOLO-Nano) produces at times t a bounding box $b _ { t } = ( x _ { c } , y _ { c } , \zeta _ { W } , \zeta _ { H } )$ with center $( x _ { c } , y _ { c } )$ and size $( \zeta _ { W } , \zeta _ { H } ) . \ \zeta _ { W }$ and $\zeta _ { H }$ are the width and height of the detected bounding box. The transmitter chooses an action $E _ { t }$ that parameterizes a region of interest $f _ { t } ( E _ { t } )$ centered at $( x _ { c } , y _ { c } )$ with scale factors $\alpha _ { t }$ and $\beta _ { t }$ (Eq. 1).

$$
\begin{array} { r l } & { f _ { t } ( E _ { t } ) = } \\ & { \left[ x _ { c } - \displaystyle \frac { \alpha _ { t } \zeta _ { W } } { 2 } , x _ { c } + \displaystyle \frac { \alpha _ { t } \zeta _ { W } } { 2 } \right] \times \left[ y _ { c } - \displaystyle \frac { \beta _ { t } \zeta _ { H } } { 2 } , y _ { c } + \displaystyle \frac { \beta _ { t } \zeta _ { H } } { 2 } \right] } \end{array}\tag{1}
$$

Let $| f _ { t } | = \alpha _ { t } \beta _ { t } \zeta _ { W } \zeta _ { H }$ be the RoI area in pixels. For symmetric expansion, we consider $\alpha _ { t } = \beta _ { t } = 1 + 2 E _ { t }$ , where $E _ { t }$ is the scaling factor that determines the cropping region of the bounding box size, with $E _ { \mathrm { m i n } } \le E _ { t } \le E _ { \mathrm { m a x } }$

In practice, each frame $F _ { t }$ will contain several sub-areas of interest (i.e., the RoIs). Let $\mathcal { F } _ { t } = \{ f _ { t , 1 } , \ldots , f _ { t , n } \}$ be the set of such sub-areas in which $f _ { t , n }$ denotes the n-th sub-area of frame $F _ { t }$ and N the set size. Once the N RoIs are defined, they are transmitted through the communication channel. Of particular importance to the transmission time is the RoIs area, which is given by $| f _ { t , n } | = ( 1 + 2 E _ { t , n } ) ^ { 2 } \zeta _ { W } \zeta _ { H }$ , because it determines the amount of information to be transmitted. Without loss of generality, we consider that each RoI is transmitted inside one communication packet of size $S _ { t , n }$ as given by Eq. 2, where $\rho$ (bits/pixel) captures the codec rate at the chosen quality and $\Psi _ { \mathrm { f r a m e } }$ aggregates headers and perpacket overhead.<sup>1</sup>

$$
S _ { t , n } ~ { = } ~ \underbrace { \rho \left| f _ { t , n } \right| } _ { \mathrm { c o n t e n t ~ p a y l o a d } } + \underbrace { \Psi _ { \mathrm { f r a m e } } } _ { \mathrm { h e a d e r s / o v e r h e a d } }\tag{2}
$$

In real-time the communication between the UAV and the Edge server is affected by packet loss, which in turn triggers retransmissions, impacting the RoIs transmission times. This loss-induced extension of the transmission times must be considered in the target delay constraint [22].

Therefore, we define the expected transmission time per packet $T _ { t , n }$ as given by Eq. 3 [23] [24], where R (bits/s) is the wireless link transmission rate, min $R T T$ is the roundtrip time, and ${ { q } _ { t , n } }$ is the per-packet success probability (after PHY/MAC). We also consider a stop-and-wait ARQ retransmissions protocol with a NACK packet of size B bits.

$$
\begin{array} { r l r } {  { T _ { t , n } = \underbrace { ( \frac { S _ { t , n } } { R } + \frac { \operatorname* { m i n } { R T T } } { 2 } ) } _ { \mathrm { f i r s t ~ a t t e m p t } } + } } \\ & { } & { \qquad + \underbrace { \frac { ( 1 - q _ { t , n } ) } { q _ { t , n } } } _ { \mathrm { e x p e c t e d ~ r e t r i e s } } \cdot \underbrace { ( \frac { B } { R } + \operatorname* { m i n } { R T T } + \frac { S _ { t , n } } { R } ) } _ { \mathrm { e a c h \ r e t r y } } , } \end{array}\tag{3}
$$

## A. Accuracy Constraints Strategy

We now construct the constraints for our framework regarding detection accuracy and sub-area frame size, emphasizing the importance of carefully selecting appropriate frame cropping area scales for UAV detection accuracy and how it greatly influences performance. The necessity for careful subarea frame scale selection arises from two primary factors: first, real-world UAVs come in a wide range of sizes, often unfamiliar to the visual system; and second, the distances between objects and the camera may vary, potentially lacking prior information. We aim to achieve sufficiently high accuracy $( \mathcal { A } _ { t , n } )$ on the server side when receiving sub-area frame $f _ { t , n } .$

We consider the server side accuracy $\boldsymbol { \mathcal { A } } _ { t , n }$ to be a function of RoI size as shown in (4). Note that $\mathcal { P } \{ \cdot \}$ is a probability of each transmitted sub-area, namely the likelihood that the server correctly detects an intruding UAV given the received RoI, i.e., $\mathcal { P } \{$ {correct detection $f _ { t , n } \}$

We use (4) to estimate on the inspecting UAV side the accuracy on the server side and decide on the need to scale the RoI before transmission so that a desired server side accuracy threshold $\mathcal { A } _ { t h r e s h o l d }$ is met. This procedure is shown in (5) in which we scale up the sub-area frame choosing an optimal scaling factor $E _ { t , n } ^ { ( k + 1 ) }$ iteratively until the threshold is overcome, starting from $E _ { t , n } ^ { ( 0 ) } = 0$ for $k = 0$

$$
\mathcal { A } _ { t , n } ^ { ( k ) } = \mathcal { P } ( | f _ { t , n } ^ { ( k ) } | ) = \mathcal { P } \Big ( ( 1 + 2 E _ { t , n } ^ { ( k ) } ) ^ { 2 } \zeta _ { W } \zeta _ { H } \Big ) .\tag{4}
$$

$$
\mathcal { A } _ { t , n } ^ { ( k + 1 ) } = \left\{ \begin{array} { l l } { \mathcal { P } ( | f _ { t , n } ^ { ( k + 1 ) } | ) , } & { \mathrm { i f ~ } \mathcal { A } _ { t , n } ^ { ( k ) } \leq A _ { t h r e s h o l d } , } \\ { \mathcal { A } _ { t , n } ^ { ( k ) } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{5}
$$

The scaling factor is updated as $E _ { t , n } ^ { ( k + 1 ) } = V _ { E } \cdot E _ { t , n } ^ { ( k ) }$ where $V _ { E }$ is computed proportionally to the correlation of the group of pixels of the sub-area with respect to its center.

Moreover, $\mathcal { P } \{ \cdot \}$ must be optimal to obtain maximum accuracy and low latency. A retrieval process based on $E _ { t , n } ,$ treated as a ranking or sorting method for different sub-area frames, does not work because it is non-differentiable. In addition, linear optimization cannot be applied, since constant changes in the input distance $\left| f _ { t , n } \right|$ cause the $\boldsymbol { \mathcal { A } } _ { t , n }$ values to be discontinuous [25]. In order to obtain an optimal $\mathcal { P } \{ \cdot \}$ , groups of pixel correlations between regions must be considered, since correlations between center and neighboring pixels are essential. This approach also reduces latency by enabling a one-time prediction of a sub-area of the frame rather than relying on a sequence of actions. The GNN primarily focuses on prediction and determining the optimal decision for a single occurrence. Using the GNN algorithm to identify the area according to pixel correlations results in a more significant decrease in latency compared to linear optimization. This is because linear optimization is based only on $E _ { t , n }$ , which does not guarantee higher precision in the first transmission of subarea frames and is vulnerable to selecting inappropriate regions of the frame with low pixel correlations.

## B. Problem Formulation

The above analysis shows that the number of transmissions of sub-area frames $f _ { t , n }$ in $T _ { t , n }$ is critical to maintaining the delay below the desired threshold of a single full-frame transmission delay. According to the [26], the total transmission time of frames can be expressed as the sum of all single subarea frame delays, which is given in (6).

$$
\begin{array} { r l } {  { \sum _ { n = 1 } ^ { N } T _ { t , n } = \sum _ { n = 1 } ^ { N } ( \frac { S _ { t , n } } { R } + \frac { \operatorname* { m i n } R T T } { 2 } +  } } \\ & { \quad  + \frac { ( 1 - q _ { t , n } ) } { q _ { t , n } } ( \frac { B } { R } + \operatorname* { m i n } R T T + \frac { S _ { t , n } } { R } ) ) } \end{array}\tag{6}
$$

The optimization problem to minimize the sum of the total delay can be stated as follows. Given the estimated packet sizes $S _ { t , n }$ for each transmission at send rate R and the oneway propagation/processing delay $\frac { \operatorname* { m i n } R T T } { 2 }$ , the objective is to achieve minimum transmission latency by adapting each cropped sub-area $f _ { t , n }$ via a variable scaling factor $E _ { t , n }$ per packet. The design also accounts for the number of sub-area transmissions and leverages an optimal $\mathcal { P } ( . )$ (pixel-correlationbased precision probability), so that the server-side accuracy $\boldsymbol { \mathcal { A } } _ { t , n }$ for each received sub-area frame is high and stable. Formally,

$$
{ \bf O P } : \quad \arg \operatorname* { m i n } _ { \{ f _ { t , n } \} _ { n = 1 } ^ { N } } \ \sum _ { n = 1 } ^ { N } T _ { t , n }\tag{7}
$$

$$
\mathrm { s u b j e c t ~ t o : } \quad \mathcal { A } _ { t , n } \geq A _ { \mathrm { t h r e s h o l d } } , \quad n = 1 , \ldots , N .
$$

where

$$
\begin{array} { r l } & { T _ { t , n } = \left( \cfrac { S _ { t , n } } { R } + \cfrac { \operatorname* { m i n } R T T } { 2 } \right) + } \\ & { \qquad + \cfrac { 1 - q _ { t , n } } { q _ { t , n } } \quad \left( \cfrac { B } { R } + \operatorname* { m i n } R T T + \cfrac { S _ { t , n } } { R } \right) , } \\ & { S _ { t , n } = \rho | f _ { t , n } | + \Psi _ { \mathrm { f r a m e } } , } \\ & { A _ { t , n } = \mathcal { P } ( | f _ { t , n } | ) , } \\ & { E _ { t , n } ^ { k + 1 } = V _ { E } \cdot E _ { t , n } ^ { k } , \quad E _ { t , n } ^ { 0 } = 0 } \end{array}
$$

## C. Complexity

Problem (OP) induces a constrained Markov decision process (MDP) with continuous action variable $E _ { t , n }$ and a nonconvex, data-driven accuracy function $\boldsymbol { A } _ { t , n } ( \cdot )$ . Exact dynamic programming is intractable. Therefore, we adopt a model-free primal–dual actor–critic method that learns policies minimizing the expected delay while satisfying accuracy constraints. In the simplified case where $E _ { t , n }$ takes values on a discrete grid of K levels, and both $\boldsymbol { \mathcal { A } } _ { t , n }$ and $T _ { t , n }$ are monotone in $E _ { t , n } ,$ each sub-area frame can be optimized independently in $O ( N K )$ by scanning the grid. However, when temporal correlations, joint constraints, and stochastic packet success are considered, the problem becomes a high-dimensional constrained MDP (CMDP), making reinforcement learning methods more appropriate.

## IV. PROPOSED GCN-ASSISTED DRL SYSTEM MODEL

This section presents the model of the GCN-assisted A2C system. The proposed model employs GCN to extract the areas of features that have a strong relationship with the object center, as well as the strong correlation between groups of pixels to facilitate future prediction. While A2C exploits the learning outcomes of GCN to train the action of latency optimization and accuracy constraint satisfaction.

In the previous section, we formulated the transmission optimization problem (OP) by defining the action as the RoI scaling factor $E _ { t , n }$ , which directly adjusts the size of the cropped sub-area frame $f _ { t , n }$ . While this abstraction provides a tractable CMDP formulation, it does not exploit correlations between pixels inside the bounding box, nor does it account for complex feature distributions that arise under different UAV sizes, lighting conditions, and motion. To address these limitations, we extend the baseline model with a GCN-assisted A2C framework. In this setting, the GCN predicts correlated pixel groups $\mathbb { S } _ { i }$ that summarize feature relationships in the frame, and these groups are mapped into bounding boxes $\mathbb { B } _ { t , n }$ centered around correlation-weighted centroids. Each $\mathbb { B } _ { t , n }$ corresponds to an RoI that can be expressed as $f _ { t , n }$ for some effective scaling factor $E _ { t , n } .$ . In other words, the GCN refines the baseline $E _ { t , n }$ -based action by constraining the A2C policy to choose feature-correlated regions that are both latency-efficient and accuracy-preserving. The following presents the proposed GCN–A2C solution model in detail.

## A. CMDP Formulation and Information Flow

At each time step t, the CMDP state is defined as $s _ { t } ~ =$ $\left( \phi _ { t } ^ { \mathrm { d e t } } , \mathbf { g } _ { t } , q _ { t } , m i n R T T _ { t } , \lambda _ { t } \right)$ , where $\phi _ { t } ^ { \mathrm { d e t } }$ are detector features (bounding boxes, scores), g is a GCN embedding summarizing pixel-group correlations around $( x _ { c } , y _ { c } )$ , and $\lambda _ { t }$ is the CMDP multiplier for latency–accuracy constraints.<sup>2</sup>

In the proposed system model, the feature vector (V) for GCN obtained from pixel-group sets $\mathbb { S } = \{ \mathbb { S } _ { i } \} _ { i = 1 } ^ { Z }$ to form the GCN embedding $\mathbf { g } _ { t }$ . That is, $\mathbf { g } _ { t } = \mathbb { V }$ provides a compact encoding of correlations between disjoint pixel groups, which the A2C agent uses to guide decision-making.

The CMDP action corresponds to selecting a bounding box $\mathbb { B } _ { t , n }$ from the GCN-predicted correlated pixel group ${ \mathbb S } _ { i } ^ { * }$ . Each bounding box $\mathbb { B } _ { t , n }$ defines a sub-area RoI in the original OP formulation: $\mathbb { B } _ { t , n } \ \equiv \ f _ { t , n }$ . Thus, the GCN transforms pixel correlations into bounding-box actions that the A2C policy transmits to the server, balancing latency $T _ { t , n }$ and accuracy $\boldsymbol { \mathcal { A } } _ { t , n }$ . After transmission, the server returns the observed accuracy (AP or IoU proxy) and measured delay, which update the estimated accuracy $\mathcal { A } _ { s }$ and success probability $q _ { t }$ online.

## B. GCN-A2C for Decision Making

We use an A2C strategy to optimize latency for solving the OP problem formulated from (2) to (6) [27]. Figure 3 illustrates a framework with three primary modules. The first is the actor module, which takes an action $\{ \mathbb { B } _ { t , n } , \mathbb { S } _ { i } \}$ . Here, the GCN predicts correlated pixel groups $\mathbb { S } _ { i } ,$ , and the corresponding bounding box $\mathbb { B } _ { t , n }$ defines the cropped sub-area. Formally, each $\mathbb { B } _ { t , n }$ corresponds to an RoI $f _ { t , n }$ with scaling factor $E _ { t , n } , \mathrm { i . e . , } \mathbb { B } _ { t , n } \equiv f _ { t , n }$ . Without GCN, the A2C actor would need to search exhaustively over pixel groups, leading to high computation. The pixel-group shape adapts to the variance or distribution around the centroid and has width $\zeta _ { W }$ and height $\zeta _ { H }$ . The critic module evaluates $\{ T _ { t , n } , \mathcal { A } _ { t , n } \}$ , i.e., the transmission delay per sub-area packet and the accuracy (mAP proxy) achieved at the server. The policy is improved by combining A2C training with GCN predictions: the GCN records correlations in pixel groups and provides future predictions, which periodically update the actor to select featurecorrelated regions that minimize latency while ensuring $\boldsymbol { \mathcal { A } } _ { t , n }$ constraints are met. This guarantees that the policy $\pi ( a _ { t } | s _ { t } )$ adapts to varying UAV sizes and pixel correlations. The size of each pixel group is determined by $\begin{array} { r } { \Theta = \sqrt { \frac { K } { M _ { i } } } } \end{array}$ , where K is the total number of correlated pixels near the object center, and $M _ { i } = | \mathbb { S } _ { i } |$ is the cardinality of the subset used by the GCN to guide actor decisions. Exhaustively evaluating groups of size $M _ { i }$ requires $\begin{array} { r } { \binom { K } { M _ { i } } = \frac { K ! } { M _ { i } ! ( K - M _ { i } ) ! } \stackrel { \bullet } { = } O ( K ^ { M _ { i } } ) } \end{array}$ , making bruteforce search infeasible. Instead, the GCN learns to predict future feature-correlated actions for A2C, reducing complexity while satisfying $\boldsymbol { \mathcal { A } } _ { t , n } .$ The proposed system not only selects pixel groups but also captures stronger correlations between the center and neighboring hidden pixels. In the next section, we detail the structure of the GCN model.

## C. Hidden Pixels Relation Exploration

This section presents a detailed graphic representation along with A2C of the proposed graph neural network structure for higher accuracy and low latency, shown in Fig. 3. To find the high correlations between the pixels of an image for optimal sub-area frame selection, we need to divide the frame into different partitions. The frame partitioning process is a critical and fundamental stage within any low-level vision system to find the region of interest where pixels have higher correlations with each other. Segmentation refers to the procedure of dividing a frame into distinct areas that do not overlap, with the objective of ensuring that each zone shows correlations and that the separation between areas is abrupt. Different methods for pixel correlations have been published in the past. However, it remains a challenge to identify a specific approach that shows consistency throughout numerous lighting conditions or low-visibility conditions. In addition, CNN performs convolution operations in smaller, more uniform areas and obtains features at the pixel level. However, GCN performs convolution operations in large areas with unevenly correlated pixels and gets features at the aggregated pixel level. The GCN method uses the average of all pixels in a set to give strong edge weights to areas with similar correlations and weak or no edge weights to areas with no correlations [2] [27].

![](images/ea8c520908e3671cdf13451d0585136f9d7a2d4d73a166595bf2e833a91614a9.jpg)  
Fig. 3. The figure illustrates the proposed framework, indicated by several color-coded parts. The left-side black box encloses the proposed model region, which symbolizes the core framework of the GCN-assisted A2C. The blue box shows the actor who selected the action. While the critic’s role in action-value evaluations. The green box is the GCN future state prediction, which regulates actor behavior to optimize latency and minimize errors.

1) State: Let $\begin{array} { r c l } { \mathbb { S } } & { = } & { \{ \mathbb { S } _ { i } \} _ { i = 1 } ^ { Z } } \end{array}$ represent the set of aggregate pixel groups forming the state, where $\begin{array} { r l } { \mathbb { S } _ { i } } & { { } = } \end{array}$ $\bar { \{ \mathbf { Y } _ { i } ^ { j } \} } _ { j = 1 } ^ { M _ { i } } , \qquad \bar { M } _ { i } = \left| \mathbb { S } _ { i } \right|$ . Here, $\mathbb { S } _ { i }$ denotes the i-th group of correlated pixels and $M _ { i }$ is its cardinality. The pixel groups are disjoint, i.e., $\mathbb { S } _ { i } \cap \mathbb { S } _ { j } = \emptyset$ for all $i \neq j$ . Moreover, the total number of pixels is preserved: $\begin{array} { r } { \zeta _ { H } \times \zeta _ { W } = \sum _ { i = 1 } ^ { Z } M _ { i } } \end{array}$ . Each group is summarized by its mean feature vector, yielding the state representation, which is given in (8):

$$
\begin{array} { l } { { \displaystyle { \mathbb { V } } = [ { \bf V } _ { 1 } , { \bf V } _ { 2 } , \ldots , { \bf V } _ { Z } ] ^ { T } } } \\ { { \displaystyle \quad = \left[ \frac { 1 } { M _ { 1 } } \sum _ { j = 1 } ^ { M _ { 1 } } { \bf Y } _ { 1 } ^ { j } , \ldots , \frac { 1 } { M _ { Z } } \sum _ { j = 1 } ^ { M _ { Z } } { \bf Y } _ { Z } ^ { j } \right] ^ { T } } , } \end{array}\tag{8}
$$

where $\mathbf { Y } _ { i } ^ { j }$ is the j-th pixel feature in group $\mathbb { S } _ { i } .$ . Thus, $\mathbb { V }$ is a column vector of dimension $Z ,$ with each entry being the average feature of a pixel group. This representation captures the correlation structure of the aggregated pixel regions, which serve as the input nodes for the GCN. After processing the initial state, the GCN predicts future correlation patterns, which guide action selection by cropping the corresponding bounding box region. A focal classification loss ${ \mathfrak { L } } _ { c l s }$ is used to train the GCN for UAV pixel correlation prediction, which is given in (9):

$$
\mathfrak { L } _ { c l s } = \lambda _ { 1 } \mathfrak { L } _ { t c r p } + \lambda _ { 2 } \mathfrak { L } _ { t c r n } ,\tag{9}
$$

where $\mathfrak { L } _ { t c r p }$ measures correlation consistency among pixels inside the UAV bounding box, while $\mathfrak { L } _ { t c r n }$ penalizes spurious correlations in background regions outside the UAV. The loss $\mathfrak { L } _ { t c r n }$ suppresses background noise near the UAV centroid, allowing the GCN to emphasize UAV-related pixels. Here, $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are tunable weighting coefficients used to balance the influence of $\mathfrak { L } _ { t c r p }$ and $\mathfrak { L } _ { t c r n }$ during training.

## D. A2C for $T _ { f }$ Optimization with $\boldsymbol { \mathcal { A } } _ { t , n }$ Satisfaction

1) Action: In the previous section, we proposed a GCN method to capture correlations between pixels by partitioning images into groups of pixels. Using this foundation, we now leverage the correlation data to define the action in the A2C framework, which specifies the region to be transmitted. Let the action be a bounding box $\mathbb { B } _ { t , n }$ associated with a correlated pixel group $\mathbb { S } _ { i } ^ { * }$ predicted by the GCN. The goal is for the GCN to identify highly correlated pixel groups that can be mapped into compact rectangular or square regions. To achieve this, we define a correlation-weighted centroid $( x _ { c } , y _ { c } )$ based on the coordinates (x<sub>Φ</sub>, y<sub>Φ</sub>) of pixels $\Phi \in \mathbb { S } _ { i } ^ { * }$ with weights ${ \mathfrak { C } } _ { \Phi } { \mathrm { : } }$

$$
x _ { c } = \frac { \sum _ { \Phi \in \mathbb { S } _ { i } ^ { * } } \mathfrak { C } _ { \Phi } \cdot x _ { \Phi } } { \sum _ { \Phi \in \mathbb { S } _ { i } ^ { * } } \mathfrak { C } _ { \Phi } } , \quad y _ { c } = \frac { \sum _ { \Phi \in \mathbb { S } _ { i } ^ { * } } \mathfrak { C } _ { \Phi } \cdot y _ { \Phi } } { \sum _ { \Phi \in \mathbb { S } _ { i } ^ { * } } \mathfrak { C } _ { \Phi } } .\tag{10}
$$

In (10) ${ \mathfrak { C } } _ { s }$ denotes the correlation weight of pixel $s ,$ highlighting its contribution to the centroid. Using the centroid, the bounding box action $\mathbb { B } _ { t , n }$ is defined as:

$$
\mathbb { B } _ { t , n } : \left[ x _ { c } - \frac { \zeta _ { W } ^ { * } } { 2 } , y _ { c } - \frac { \zeta _ { H } ^ { * } } { 2 } , x _ { c } + \frac { \zeta _ { W } ^ { * } } { 2 } , y _ { c } + \frac { \zeta _ { H } ^ { * } } { 2 } \right] ,\tag{11}
$$

In (11) $\zeta _ { W } ^ { * }$ and $\zeta _ { H } ^ { * }$ denote the width and height of the correlated region. These dimensions are adaptive, depending on the variability of the pixel-group distribution around the centroid. This flexibility improves the expressiveness of the bounding box and ensures that correlated regions are captured accurately.

2) Reward: The A2C training reward $( r _ { t } ^ { \mathrm { A 2 C } } )$ determined by the totals of sub-area frames transferred to the server within the action on the group of pixel regions as $\left\{ \mathbb { S } _ { i } ^ { \psi } \mid \mathbb { S } _ { i } ^ { * } , \mathbb { B } _ { t , n } \right\}$ , as ${ \mathbb S } _ { i } ^ { * }$ represents the correlated group of GCN-trained pixels. The action $\mathbb { B } _ { t , n }$ is taken by the UAV, and the GCN-predicted group of pixels transits from $\mathbb { S } _ { i } ^ { * }$ to $\mathbb { S } _ { i }$ . However, the reward can only be optimized when $\mathcal { A } _ { t , n } ~ \geq A _ { t h r e s h o l d }$ accuracy is considered for positive detection of the object on the server. It is essential to achieve optimal transmission latency and avoid penalization for constraint violation. It might be possible to put a condition for $\left\{ \mathbb { S } _ { i } ^ { \psi } \mid \mathbb { S } _ { i } ^ { * } , \mathbb { B } _ { t , n } \right\}$ preserving the feature-correlated region that can be achieved by Lagrangian dual form, as given in (12):

$$
\mathcal { L } ( \mathbb { S } _ { i } ^ { \psi } , \mathbb { S } _ { i } ^ { * } , \mathbb { B } _ { t , n } , \lambda ) = \operatorname* { m i n } _ { \forall i _ { \mathtt { S } _ { i } ^ { * } } \in [ 1 , Z ] } T _ { t , n } + \lambda ( A _ { \mathrm { t h r e s h o l d } } - A _ { t , n } ) ,\tag{12}
$$

where, λ shows how sensitive the objective function is to changes in the value of $\begin{array} { r } { \mathcal { A } _ { t , n } \ \geq \ A _ { \mathrm { t h r e s h o l d } } . } \end{array}$ , the accuracy constraints. By using this approach, we are able to ensure that minimizing the objective function will only occur if the accuracy requirements are achieved. Otherwise, OP is penalized and stops optimizing more during $\mathcal { A } _ { t , n } < A _ { \mathrm { t h r e s h } }$ <sub>old</sub>. The fixed λ creates a lack of convergence and over- and underpenalization constraint violation during latency optimization. To address these issues, we use dual gradient descent to dynamically update the λ as mentioned in (13):

$$
\lambda  \lambda + \eta ( A _ { \mathrm { t h r e s h o l d } } - A _ { t , n } ) ,\tag{13}
$$

where $\eta$ is the learning rate, which is initially set to 0.01 with a weight λ of 1. The η dynamically adjusts λ during constraint violation. So $r _ { t } ^ { \mathrm { A 2 C } } \left\{ \bar { \mathbb { S } } _ { i } ^ { \psi } \ \backslash \ \mathbb { S } _ { i } ^ { * } , \mathbb { B } _ { t , n } \right\}$ of the training of A2C is determined by offloaded group of pixel sets to the server and is given in (14) by

$$
r _ { t } ^ { \mathrm { A 2 C } } \left\{ \mathbb { S } _ { i } ^ { \psi } \mid \mathbb { S } _ { i } ^ { * } , \mathbb { B } _ { t , n } \right\} = \operatorname* { m i n } _ { \forall i _ { \mathbb { S } _ { i } ^ { * } } \in \left[ 1 , Z \right] } T _ { t , n } .\tag{14}
$$

We incorporate the reward into the temporal difference learning (δ<sub>t</sub>) error for the critic and actor loss as mentioned in (15):

$$
\delta _ { t } = r _ { t } ^ { \mathrm { A 2 C } } \left\{ \mathbb { S } _ { i } ^ { \psi } \mid \mathbb { S } _ { i } ^ { * } , \mathbb { B } _ { t , n } \right\} + \gamma V ^ { * } ( \Phi _ { t + 1 } ) - V ^ { * } ( \Phi _ { t } ) ,\tag{15}
$$

where, $V ^ { * } ( \Phi _ { t + 1 } )$ is the update estimated state of $\Phi _ { t }$ that is the old estimated state of $V ^ { * } ( \Phi _ { t } )$ . The critic loss is evaluated by squaring the $\left( \delta _ { t } \right)$ to ensure how well the $V ^ { * } ( \Phi )$

While the actor loss is based on the advantage function to regulate the policy $\left( \pi _ { \theta } ( a _ { t } \mid \Phi _ { t } ) \right)$ and map the states $\left( \Phi _ { t } \right)$ to the actions $\left( \boldsymbol { a } _ { t } \right)$ as given $\operatorname { a s } - \log \pi _ { \boldsymbol { \theta } } ( a _ { t } \mid \Phi _ { t } ) \cdot \delta _ { t } . - \log \pi _ { \boldsymbol { \theta } } ( a _ { t } \mid \Phi _ { t } )$ takes action under the current policy π<sub>θ</sub> for the state $\Phi _ { t }$ and takes the product of the likelihood of action with $\delta _ { t }$ to update the policy using (16):

$$
\mathcal { L } _ { l o s s } = - \log \pi _ { \boldsymbol { \theta } } ( a _ { t } \mid \Phi _ { t } ) \cdot \delta _ { t } + \mathcal { W } _ { c } \cdot \left( \delta _ { t } \right) ^ { 2 } ,\tag{16}
$$

where ${ \mathcal { W } } _ { c }$ shows the weighted factor for the critic loss. $\mathcal { L } _ { l o s s }$ defines that the actor is maximizing the advantage and critic is minimizing the $\delta _ { t }$ error with current updates within the constraints defined by $\mathbf { A } 2 \mathbf { C }$ reward $r _ { t } ^ { \mathrm { A 2 C } } \left\{ \mathbb { S } _ { i } ^ { \psi } \mid \mathbb { S } ^ { * } { } _ { i } , \mathbb { B } _ { t , n } \right\}$

3) Policy: The policy $\pi _ { \theta } ( a _ { t } | \Phi _ { t } )$ represents the A2C actor network that maps the CMDP state Φ<sub>t</sub> to a probability distribution over actions $a _ { t }$ . Each action corresponds to selecting a bounding box $\mathbb { B } _ { t , n }$ , which defines the ROI $f _ { t , n }$ to be transmitted. The policy parameters θ are updated using the actor loss in (16), which leverages the advantage signal $\delta _ { t }$ to encourage actions that reduce latency $T _ { t , n }$ while maintaining the required accuracy $\boldsymbol { \mathcal { A } } _ { t , n }$

## V. EXPERIMENTAL WORK

## A. Implementation of GCN-assisted A2C Model

This section presents an overview of the results for onboard detection of intruding UAVs through the proposed GCNassisted A2C model. The proposed GCN-assisted A2C model is built on a 64-bit Ubuntu 22.04 Linux workstation using Python 3.9 and PyTorch with a GPU of A300. In this study, we used YouTube videos for UAV-to-UAV monitoring. There are a total of 100 clips, all of which are 240 to 1080 resolution videos. These videos feature a variety of backgrounds, such as clouds, buildings, mountains, and sea, as well as fast movement, out-of-focus, and a range of sizes. The UAVs seen in each video come in a range of sizes, from large to small. We selected UAVs with a size range of 300mm to 1200mm to train and validate our proposed model. Furthermore, the UAVs cruise at speeds between 50 and 100 miles per hour, occasionally stopping altogether. We used all the validation videos from the YouTube stream for model training, testing and validation. Specifically, we train the model on 70 of the videos, using 15 of the remaining for testing and the remaining 15 for validation.

The GCN-assisted A2C is trained for 1 iteration per epoch to find the correlation between the different regions in every frame, and the GCN-assisted A2C is trained for 100 epochs to extract the pixel feature-correlated regions to send it to the server. GCN-assisted A2C is analyzed every 5 epochs using a validation function during the training process. This function finds the training loss and the number of iterations at which convergence occurs. The GCN-assisted A2C model with the lowest cost, denoted as the sub-optimum, is then used to assess the test data.

The GCN-assisted A2C used multiple layers that are organized into the GCN layers and multi-layer perceptron layers.

The GCN layers module consists of three GCN multi-head layers, each containing two GCN layer edge softmax heads. The first GCN multi-head layer utilizes heads that apply a linear transformation f to convert input pixel features into output-correlated pixel features. Additionally, another linear transformation is used to convert input pixel features into output pixel feature-correlated features. The second layer undergoes a transformation where 128 input features are converted to 64 output pixel features, and 128 input pixel features are converted to 1 output pixel feature in a region that is correlated. The third layer of the GCN model performs a similar transformation by changing 256 input layer pixel features into 64 output pixel features and 256 input pixel features into 1 output pixel feature-correlated group of regions. The A2C module has two linear layers, in which the first layer maps 256 input pixel features to 128 output pixel features. For the final prediction, the second layer maps 128 input features to output feature-correlated group regions. It does this by using (10) and (11) for the centroid.

The calculation of the policy gradients after multiplying by the benefits is known as policy loss. The Unmanned Aerial Vehicle (UAV) activities undergo training in the Advantage Actor-Critic (A2C) module for a total of 1500 steps, using a discount factor of 0.99. The loss function is calculated by taking the average of the squared differences between the estimated values and the actual returns. The loss calculation assigns a coefficient of 0.5 to the value function.

## B. Ablation Study

In Fig. 4, we can see how well the proposed GCN-assisted A2C system model and different DRL algorithms work in terms of transmission latency rate during training. We found that the A2C, DDPG, and DQN models mostly do the same thing, which is to minimize transmission latency. However, the synchronous nature of the A2C model leads to higher performance than the deterministic nature of the DDPG and DQN models. The frame area’s importance and the dynamical changes in the UAV’s size due to different movements make A2C more stable than the DDPG and DQN models. In addition, cropping the frame area makes a clear action space, which might be one reason why A2C is generally more useful than DDPG. The frame area cropping creates priority fluctuation that leads A2C to better decision-making. It’s hard for the DDPG and DQN models to make the best decision when there are a lot of different environmental factors and frame area cropping variability problems. However, to adopt and find feature correlations and dependencies between different regions of the frame still reduces the performance of A2C. We proposed a GCN-assisted A2C system model, which ensures and preserves critical contextual features of groups of neighboring pixels in regions where A2C lacks the ability to deal with such dependencies. A2C processes and transmits a single sub-area frame entity and makes it difficult to find the nuanced interplay between different sub-areas of frame parts or sizes of UAVs, variations, or environmental factors. Moreover, GCN assists the A2C to extract informative features of the UAV using aggregation and propagation of features across graph nodes to make better predictions. Similarly, the GCN-assested A2C system model reduces the high dimensionality focus to the specific interdependent area to efficiently handle energy constraints or bandwidth limitations. Without GCN, A2C struggles with scalability, and it leads to high computation, and might be complex architectures needed. In addition, GCN-assisted A2C is more robust to noises or missing pixel feature information to aggregate neighboring nodes feature information to improve predictions and reduce the transmission latency. If a portion of a frame is corrupt or unclear with environmental effects, then GCN leverages the hidden feature pixel information of the surrounding group of pixels, improves the transmission latency, and preserves the prediction performance constraints.

![](images/b0f956b9e5b642f44c59393bd19e0180743e31c1a3990c9a42541c55cfe89ac8.jpg)  
Fig. 4. The ablation study of different RL algorithms performance using loss using data that have across different sizes of UAVs and dynamic movements with different environmental effects.

Moreover, in the ablation study, we used different DRL models (DQN, DDPG, and A2C) and studied required evaluation metric performance, like latency for each video frame transmission, overall AP, and mean IoU. The latency of the GCN-assisted A2C is much lower than that of the other benchmark DRL algorithms. The overall FPS transmission latency of the GCN-assisted A2C is 45 ms, which is much lower than the A2C without GCN, which has an FPS transmission latency of 170 ms. The GCN-assisted A2C achieved optimal latency due to the focus on the feature-correlated group of pixel regions to recommend to the A2C for optimal action. Moreover, Fig. 5 shows the AP and mean IoU of the GCN-assisted A2C and the three variants of the ablation study. The GCN-assisted A2C is the most effective across all the algorithms to transmit the optimal area of the frame that is utilized by the edge server. The GCN-assisted A2C achieved (AP @ 70.72%) and (mean IoU @ 60.30%) to improve (AP @ 62.80%) and (mean IoU @ 30.60%) over A2C. While A2C achieved (AP @ 43.44%) and (mean IoU @ 45.91%), which is higher in ratio than (AP @ 42.20%) and (mean IoU @ 60.50%) over the DDPG algorithm. Similarly, DDPG achieved (AP @ 30.55%) and (mean IoU @ 28.61%), and DQN with a very low score of (AP @ 17.30%)

![](images/04edc7ce305252204e4936e634fb07dbbc6145d4ee8397e1b24ce9aeda257308.jpg)  
Fig. 5. With the proposed GNN-A2C (GCN-assisted A2C) and different DRL algorithms, we can check the average (30FPS) IoU and tracking AP accuracy.

## C. State-of-the-art (SOTA) comparisons

Fig. 6 shows the statistical performance of different state-ofthe-art (SOTA) models, i.e., FlexPatch, EdgeDuet, and other DRL, to visualize the transmission latency in the range of cumulative distributed function (CDF) parameters. We found that the proposed GCN-assisted A2C had the best transmission latency of 50 ms with a standard deviation of 12 ms when compared to other SOTA models. FlexPatch came in second with a transmission latency of 90 ms and a standard deviation of 15 ms. Meanwhile, EdgeDuet achieved a video frame transmission latency of 110 ms, and its standard deviation is 20ms. In contrast, other DRL models achieve a video frame transmission latency of 177 ms, with a standard deviation of 33 ms. We can see from the CDF of the GCN-assisted A2C that the model works better for achieving the best video frame transmission latency across a range of communication factors.

At the same time, we examined how well the GCN-assisted A2C worked regarding transmission delay, AP, and meanIoU accuracy when compared to FlexPatch and EdgeDuet, using different video resolutions as shown in Fig. 7. In Fig. 7(a), we show how accurately we can find suspicious UAVs from start to finish, using AP and meanIoU accuracy. Similarly, in Fig. 7(b), the mean IoU over end-to-end latency is compared. The GCN-assisted A2C model outperforms previous models. For example, FlexPatch AP accuracy performance is 60.10% with meanIoU accuracy results of 48.42% and achieved minimum latency results of 90.10 ms. The FlexPatch adds overhead when scenes change rapidly or unpredictably, and it is highly challenging to select the optimal context-dependent features. EdgeDuet achieved AP accuracy of 42.35% with meanIoU accuracy results of 39.52% and achieved minimum latency results of 110.25ms. This model also uses many highresolution tiles at the edge server and is unsuitable for realtime or low-bandwidth scenarios due to offloading frame tiles to the edge server. The GCN-assisted A2C model is better at handling quickly changing or unpredictable scenes because it uses groups of pixel areas as nodes and their connections to understand how they depend on each other. The GCNassisted A2C model reached the best accuracy of 70.72% AP, with meanIoU accuracy results of 360.30%, and improved the transmission delay to 50ms. The GCN-assisted A2C model works well in real-time or low-bandwidth situations because it can handle parts of the frame by grouping together related pixel areas.

![](images/61dbb0fd780d7c690515b2f8c788e57df49c815265ed4568b5e5957099632837.jpg)  
Fig. 6. The effectiveness of GCN-assisted A2C over SOTA models to optimize the transmission latency.

Fig. 7(c) shows the AP accuracy and meanIoU of the GCNassisted A2C when transmitting videos of different frame sizes. The GCN-assisted A2C consistently achieves higher AP meanIoU accuracy at the server side. With the decrease in frame resolutions, GCN-assisted A2C accuracy drops upto 50 AP, which is our desired threshold to have a minimum of 50 AP at the server side.

## D. Evaluation Results

We have qualitatively and quantitatively shown the expected trade-off between AP and latency improvement for three different UAV sizes under various circumstances. Each environment has three visual situations, with each subplot representing the AP and transmission latency. The blue bars show latency, whereas the olive-colored bars represent AP. Together, they exhibit both precision and efficiency. Overlaid visuals in each subfigure illustrate corresponding visual inputs related to the environmental condition. This shows how complex the agents experiences are. Fig. 8 shows that smaller UAVs, similar backgrounds, and partial cloudiness can affect latency and reduce server-side AP. These unique features have a major effect on the efficiency of transmission in the other DRL models. The proposed GCN-assisted A2C keeps an AP above 50 while ensuring the best transmission speed in all conditions and sizes of UAVs during testing of the algorithms in three different conditions—cloudy, complex, and clear sky. This hypothetical situation motivates the advancement of smarter feature-correlated region prediction algorithms. The results indicate that GCN-assisted A2C consistently does better than the basic models in every situation, achieving a much higher AP with less delay. This benefit is especially evident in cloudy or messy situations, such as cloudy and complex situations, where classic models like DQN and DDPG do not perform well. GCN-assisted A2C, on the other hand, is robust to environmental changes and visual effects, making it more stable and suitable for a wider range of applications. In addition, GCN-assisted A2C continues to outperform its competitors in ideal situations such as clear skies. This demonstrates that it performs well in both challenging and normal circumstances. These findings show that GCN-assisted A2C could be effective for making real-time predictions based on vision when transmission speed and precision are important.

![](images/2fb9461362ca5472a7c14362d86a95c9ffe857ae93e17d630e9f4b097e296a88.jpg)  
(a)

![](images/e609e18d26390c27fee4ed519c8774219403965675bca99ba2bbf8b879b7d56a.jpg)  
(b)

![](images/dd0347f47f026dc3f47b40827ba62941445d757eff8c8589f0ead3d30f19f3de.jpg)  
(c)  
Fig. 7. Performance of different models under UAV-to-server video sub-area of frame transmission. In (a), the AP and meanIoU accuracy of finding suspiciou UAVs from end-to-end are shown. In (b), the meanIoU over end-to-end latency is compared. And in (c), the performance of GCN-assisted A2C is shown over different input video frame resolutions.

## VI. CONCLUSION

This article proposed a GCN-assisted A2C model that uses data from the UAV detection nano YOLO algorithm, focusing on the center of the detected object and the surrounding pixels. The design caters to scenarios in which on-board UAV detection accuracy is less than 50. We designed the detector model to identify suspicious UAVs in no-fly zones and pinpoint their location, utilizing the onboard UAV cameras. Next, we proposed using a GCN-assisted A2C to enhance the significance of certain pixel areas in the images by examining how each area connects to the center of the UAV, creating groups based on these connections, and calculating a total score with A2C. The nano detector model results emphasize that the GCN-assisted A2C model requires transmitting the frames to the server or not. We show that using the GCNassisted A2C model can minimize the transmission latency and optimize until the AP is more than 50 on the server side. We conducted studies using authentic on-board UAV camera data and with different DRL models for the ablation study and SOTA models. Our GCN-assisted A2C model shows improved transmission latency and AP by using GCN to group related pixels and understand hidden pixel relationships, which helps the A2C make better decisions in different situations like clear skies, complex backgrounds, and partial clouds.

## ACKNOWLEDGMENTS

## REFERENCES

[1] A. Hitchcock and K. Sung, “Multi-view augmented reality with a drone,” in Proceedings of the 24th ACM Symposium on Virtual Reality Software and Technology, ser. VRST ’18. New York, NY, USA: Association for Computing Machinery, 2018. [Online]. Available: https://doi.org/10.1145/3281505.3283397

[2] K. Li, W. Ni, X. Yuan, A. Noor, and A. Jamalipour, “Exploring graph neural networks for joint cruise control and task offloading in uav-enabled mobile edge computing,” in 2023 IEEE 97th Vehicular Technology Conference (VTC2023-Spring), 2023, pp. 1–6.

[3] A. Noor, K. Li, A. Ammar, A. Koubaa, B. Benjdira, and E. Tovar, “A hybrid deep learning model for uavs detection in day and night dual visions,” in 2021 IEEE Third International Conference on Cognitive Machine Intelligence (CogMI), 2021, pp. 221–231.

[4] F. U. M. Ullah, M. S. Obaidat, A. Ullah, K. Muhammad, M. Hijji, and S. W. Baik, “A comprehensive review on vision-based violence detection in surveillance videos,” ACM Comput. Surv., vol. 55, no. 10, feb 2023. [Online]. Available: https://doi.org/10.1145/3561971

[5] K. Rezaee, M. R. Khosravi, and M. S. Anari, “Deep-transfer-learningbased abnormal behavior recognition using internet of drones for crowded scenes,” IEEE Internet of Things Magazine, vol. 5, no. 2, pp. 41–44, 2022.

[6] A. Galanopoulos, J. A. Ayala-Romero, D. J. Leith, and G. Iosifidis, “Automl for video analytics with edge computing,” in IEEE INFOCOM 2021 - IEEE Conference on Computer Communications, 2021, pp. 1–10.

[7] R. Xu, S. Razavi, and R. Zheng, “Edge video analytics: A survey on applications, systems and enabling techniques,” IEEE Communications Surveys & Tutorials, vol. 25, no. 4, pp. 2951–2982, 2023.

[8] Y. Dong, L. Song, R. Xie, and W. Zhang, “Ultra-low latency, stable, and scalable video transmission for free-viewpoint video services,” IEEE Transactions on Broadcasting, vol. 68, no. 3, pp. 636–650, 2022.

![](images/1fde0e9f3b2452e97b60e4e6641cfc54184acbce9a8a17d54fda14784dac50f0.jpg)  
Fig. 8. We evaluated three different sizes of UAVs—small (S), medium (M), and large (L)—under various conditions, such as clear sky, complex background, and partially cloudy.

[9] X. Wang, Z. Yang, J. Wu, Y. Zhao, and Z. Zhou, “Edgeduet: Tiling small object detection for edge assisted autonomous mobile vision,” in IEEE INFOCOM 2021 - IEEE Conference on Computer Communications, 2021, pp. 1–10.

[10] J. Zhang, X. Huang, J. Xu, Y. Wu, Q. Ma, X. Miao, L. Zhang, P. Chen, and Z. Yang, “Edge assisted real-time instance segmentation on mobile devices,” in 2022 IEEE 42nd International Conference on Distributed Computing Systems (ICDCS), 2022, pp. 537–547.

[11] K. Yang, J. Yi, K. Lee, and Y. Lee, “Flexpatch: Fast and accurate object detection for on-device high-resolution live video analytics,” in IEEE INFOCOM 2022 - IEEE Conference on Computer Communications, 2022, pp. 1898–1907.

[12] J. Han, Y.-F. Ren, A. Brighente, and M. Conti, “Rango: A novel deep

learning approach to detect drones disguising from video surveillance systems,” ACM Trans. Intell. Syst. Technol., vol. 15, no. 2, feb 2024. [Online]. Available: https://doi.org/10.1145/3641282

[13] M. A. Khan, E. Baccour, Z. Chkirbene, A. Erbad, R. Hamila, M. Hamdi, and M. Gabbouj, “A survey on mobile edge computing for video streaming: Opportunities and challenges,” IEEE Access, vol. 10, pp. 120 514–120 550, 2022.

[14] Y. Wang, J. Xu, and W. Ji, “A feature-based video transmission framework for visual iot in fog computing systems,” in 2019 ACM/IEEE Symposium on Architectures for Networking and Communications Systems (ANCS), 2019, pp. 1–8.

[15] C. Qu, P. Drefahl, W. Guo, and H. Wang, “Autonomous video transmission and air-to-ground coordination in uav-swarm-aided disaster

response platform,” in Digital Human Modeling and Applications in Health, Safety, Ergonomics and Risk Management, V. G. Duffy, Ed. Cham: Springer Nature Switzerland, 2024, pp. 339–355.

[16] N. Khan, A. Ahmad, A. Wakeel, Z. Kaleem, B. Rashid, and W. Khalid, “Efficient uavs deployment and resource allocation in uav-relay assisted public safety networks for video transmission,” IEEE Access, vol. 12, pp. 4561–4574, 2024.

[17] A. Ghosh, S. Iyengar, S. Lee, A. Rathore, and V. N. Padmanabhan, “React: Streaming video analytics on the edge with asynchronous cloud support,” in Proceedings of the 8th ACM/IEEE Conference on Internet of Things Design and Implementation, ser. IoTDI ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 222–235. [Online]. Available: https://doi.org/10.1145/3576842.3582385

[18] Z. Liu and Y. Jiang, “Design and implementation for a uav-based streaming media system,” Ad Hoc Networks, vol. 156, p. 103443, 2024. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S1570870524000544

[19] A. Yaqoob, Z. Yuan, and G.-M. Muntean, “A uav-centric improved soft actor-critic algorithm for qoe-focused aerial video streaming,” IEEE Transactions on Vehicular Technology, vol. 73, no. 9, pp. 13 498–13 512, 2024.

[20] C. Song, B. Han, X. Ji, Y. Li, and J. Su, “Ai-driven multipath transmission: Empowering uav-based live streaming,” IEEE Network, vol. 38, no. 2, pp. 202–210, 2024.

[21] D. Wu, L. Wang, M. Liang, Y. Kang, Q. Jiao, Y. Cheng, and J. Li, “Uavassisted real-time video transmission for vehicles: A soft actor–critic drl approach,” IEEE Internet of Things Journal, vol. 11, no. 8, pp. 14 710– 14 726, 2024.

[22] J. Wu, R. Tan, and M. Wang, “Streaming high-definition real-time video to mobile devices with partially reliable transfer,” IEEE Transactions on Mobile Computing, vol. 18, no. 2, pp. 458–472, 2019.

[23] P. Yu, F. Chen, J. Wang, P. Chen, J. Cai, and M. Yang, “Adaptive antipacket loss strategy for real-time video streaming,” in 2022 IEEE 8th International Conference on Computer and Communications (ICCC), 2022, pp. 2063–2068.

[24] J. Wu, B. Cheng, M. Wang, and J. Chen, “Energy-efficient bandwidth aggregation for delay-constrained video over heterogeneous wireless networks,” IEEE Journal on Selected Areas in Communications, vol. 35, no. 1, pp. 30–49, 2017.

[25] K. He, Y. Lu, and S. Sclaroff, “Local descriptors optimized for average precision,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 596–605.

[26] A. Alhilal, T. Braud, B. Han, and P. Hui, “Nebula: Reliable low-latency video transmission for mobile cloud gaming,” in Proceedings of the ACM Web Conference 2022, ser. WWW ’22. New York, NY, USA: Association for Computing Machinery, 2022, p. 3407–3417. [Online]. Available: https://doi.org/10.1145/3485447.3512276

[27] K. Li, W. Ni, X. Yuan, A. Noor, and A. Jamalipour, “Deep-graph-based reinforcement learning for joint cruise control and task offloading for aerial edge internet of things (edgeiot),” IEEE Internet ofThings Journal, vol. 9, no. 21, pp. 21 676–21 686, 2022.