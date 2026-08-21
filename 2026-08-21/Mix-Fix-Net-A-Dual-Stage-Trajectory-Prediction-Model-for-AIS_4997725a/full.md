# Mix&Fix-Net: A Dual-Stage Trajectory Prediction Model for AIS and Vision-Derived Vessel Data<sup>⋆</sup>

Md Mahmuddun Nabi Murad<sup>a</sup>, Bora San Turgut<sup>b,1</sup>, Yasin Yilmaz<sup>a</sup>

<sup>a</sup>Department of Electrical Engineering, University of South Florida, Tampa, FL, 33647, USA <sup>b</sup>College of Transportation and Logistics, Istanbul University, Istanbul, Turkey

## Abstract

Vessel trajectory prediction is critical for maritime safety and accident prevention. While most existing trajectory prediction models rely on Automatic Identification System (AIS) data due to its precision and availability, small vessels mostly operate without AIS, resulting in a significant monitoring gap. To address this, we propose Mix&Fix-Net, a dual-stage mixer-based trajectory prediction model designed to handle vessel trajectory time-series data derived from both AIS and (non-AIS) vision data. Our architecture integrates a Primary Trajectory Predictor with a Residual Trajectory Adjuster, enabling more refined trajectory prediction. Additionally, we introduce a new video-based dataset derived from webcam streams, from which vessel trajectories are extracted to represent non-AIS data. Extensive evaluations on both AIS and non-AIS datasets across six metrics (mean squared error, mean absolute error, symmetric mean absolute percentage error, final displacement error, Frechet distance, and average Euclidean distance) demonstrate that Mix&Fix-Net consistently outperforms existing baselines across most metrics and datasets. Keywords: Vessel Trajectory Prediction, Maritime Transportation, Video Dataset,

Non-AIS-based Trajectory Prediction

## 1. Introduction

More than three-quarters of global trade occurs using marine transport, and this portion is increasing daily due to the modernization of the marine transport system [1]. This increasing demand is also making the system vulnerable to potential hazards. According to [2], an average of 2,700 maritime accident insurance claims are made to Allianz Global Corporate & Specialty (AGCS) per year. Additionally, the study in [3] reports that approximately 50% of maritime accidents go unreported. These accidents not only cause direct damage to vessels but also result in severe environmental consequences [4]. According to [5, 6], the most frequent accidents are collision-type accidents. Developing an accurate trajectory prediction model is therefore essential to mitigate such accidents and prevent potential hazards.

The traditional trajectory prediction methods utilize the Automatic Identification System (AIS) to locate and estimate the positions of the vessels [7]. However, not all vessels have AIS in their system, as AIS is required for vessels of more than or equal to 300 gross tonnage [8]. Many smaller vessels, which are not subjected to this requirement, do not have AIS due to the additional installation cost. In this case, it is dificult to monitor the smaller vessels. In addition, the oficer on the watch continuously observes the trafic around the ship during the voyage and maneuvers the ship according to situational demands. Nevertheless, even when such vessels are visible to the naked eye, it is still challenging to discern their intentions. Especially in busy maritime trafic areas, where there are many smaller vessels, such as ports, it becomes challenging for the crew or port authority to carefully monitor all surrounding smaller vessels and analyze the risk beforehand. These limitations also highlight the need for a robust trajectory prediction model that remains efective regardless of AIS availability.

Several prior studies have proposed prediction models, but most rely heavily on AIS data, with only a few exploring alternatives. For instance, the model in [9] utilizes both the vessel images and AIS data, yet it does not address the previously mentioned challenge of employing an alternative to AIS. The radar image is proposed as an alternative to AIS in [10]. However, it is not plausible to access the radar image for all types of vessels. Several studies have also explored fusing AIS with other data sources. Ship-radar and AIS data are fused to extract various trajectory characteristics, such as speed, direction, turning rate, and acceleration [11]. AIS is combined with video for harbor vessel monitoring [12], and with ship images for maritime computer vision tasks [13]. AIS is also fused with discretized meteorological features [14] and multi-source environmental data [15] to improve prediction accuracy. However, all of these approaches still depend on AIS as the primary modality. To overcome these limitations and provide a solution applicable to both AIS- and non-AIS-equipped systems, we propose Mix&Fix-Net, a novel dualstage MLP-mixer-based model designed for vessel trajectory prediction.

Vessel trajectory prediction is fundamentally a multivariate time-series prediction task. While RNNs (e.g., LSTM, GRU), TCNs, and Transformers are common in time-series prediction, they often sufer from sequential bottlenecks [16, 17]. Recent literature shows that MLP-mixer-based architectures can outperform these models by eficiently mixing data in multiple directions [18, 17]. Moreover, MLP-Mixer-based models demonstrate strong performance in both long-term and short-term forecasting. Motivated by these advantages, we adopt the MLP-Mixer core in Mix&Fix-Net to ensure high predictive accuracy in the trajectory prediction task. Alongside the model design, we introduce a video-based dataset, from which vessel trajectories are extracted, as a non-AIS alternative in scenarios where AIS data are unavailable.

Our contributions can be summarized as follows:

• We propose a novel MLP-mixer-based model, Mix&Fix-Net, that is applicable to both AIS and non-AIS data to predict the trajectory of the vessels.

• We also propose a video-based dataset sourced from an open-access real-time webcam stream for studying vessel trajectory prediction through a practical, easy-to-obtain, non-AIS data modality.

• We provide an in-depth performance analysis for the proposed method by comparing it with existing benchmark methods in terms of six metrics.

## 2. Related Work

Research on vessel trajectory prediction has become increasingly popular due to its vast applications, such as maritime trafic flow control and accident prevention. Data availability has supported developments in this field. Most of the vessel trajectory prediction studies in the literature are based on AIS data [19, 20, 21, 22]. Due to the easy access to AIS data, numerous studies, using statistical methods [23] and deep learning-based methods, have been performed to predict the trajectory of the vessels. For instance, a transformer-based model, Seaformer, is proposed in [24] to predict the trajectory of the vessel utilizing the AIS data. Similarly, A probabilistic approach for trajectory prediction is proposed in [25].

However, AIS latency and data loss negatively afect the accuracy and efectiveness of the vessel intention recognition process [26]. On the other hand, not all vessels are equipped with AIS, and the holistic picture of maritime trafic is not obtained without the localized information of all types of vessels. Therefore, these studies are not complete in this respect. Visual data from widespread cameras on vessels and in ports can greatly contribute to complementing AIS data for vessel trajectory prediction.

Study in [27] shows the importance of trajectory prediction in avoiding bridge-ship collisions. They propose a deep learning-based system to avoid bridge-ship collision using multi-channel video data. They integrate the visible-spectrum camera with infrared thermal imaging to address the challenge of trajectory prediction in diferent illumination conditions. In [9], a model based on LSTM and a Graph attention neural network is proposed for vessel trajectory identification and estimation. They also propose a multi-feature-point tracking system utilizing the vessel’s movement and orientation in the video. However, the accuracy of trajectory prediction based on video information largely depends on the video quality, while the video quality varies depending on the weather. In [13], a novel dataset fusing various images of ships taken in various weather conditions and their corresponding AIS information is proposed. In addition to the video images, multi-sensor information is also utilized to predict the trajectory of the vessels. In [10], a multi-task sequence-to-sequence network is proposed for trajectory prediction using the radar image. In addition to radar and visual data, a recent line of research fuses ship-shore VHF voice communication with AIS to enhance maritime intelligent perception. In [28], AIS trajectories are linked with the navigational intentions expressed in VHF speech, associating spoken intentions with vessel tracks via speech recognition and ship-name similarity matching. Additionally, AIS data is integrated with VHF voice feature recognition to predict ship berthing-behavior intentions under noisy inland-waterway conditions [29]. While these voice-AIS methods add a valuable semantic, intention-level cue, they still depend on AIS as the primary modality and cannot monitor vessels operating without AIS. Our method instead operates on either AIS or vision-derived (non-AIS) trajectories independently, directly addressing this monitoring gap.

With the developments in computer vision technology, the use of cameras as an electronic navigation aid on ships is expected to increase. In parallel, AIS-based trajectory prediction remains critical, as AIS provides precise and reliable vessel position data. Therefore, we propose a novel MLP-mixer-based model that predicts vessel trajectories using historical data derived from either AIS or video sources.

## 3. Proposed Method

Our objective is to predict the trajectory of the vessels based on their previous positions. We formulate the trajectory prediction problem in the following way.

Let a multivariate time series $\ b X _ { L } \in \mathbb { R } ^ { L \times C _ { i } }$ provides the previous L observations of a vessel at time step t. Our goal is to predict the next T positions, denoted by $\hat { \pmb { Y } } _ { T } \in \mathbb { R } ^ { T \times C _ { o } }$ , while the corresponding ground truth is given by $\pmb { Y } _ { T } \in \mathbb { R } ^ { T \times C _ { o } }$ . Here, $C _ { i }$ denotes the number of input features that include both positional and non-positional information, whereas $C _ { o }$ refers to the number of output features that correspond to the positional coordinates. In this work, the time series corresponding to $X _ { L }$ can be derived from either AIS or non-AIS modality. The overall framework of our method is shown in Figure 1, which illustrates how time-series trajectories are obtained from the non-AIS data, while the AIS data is directly used as time-series. The detailed architecture of our proposed trajectory prediction model, Mix&Fix-Net, is shown in Figure 2.

![](images/49f8735f926ffb01b1bc55612a93567dddce518f0646486a1180f7ccea872197.jpg)  
Figure 1: Overall methodological framework of our vessel trajectory prediction pipeline. The framework supports two alternative input modalities: non-AIS (webcam video) processed through a vessel detector and tracker, or AIS data. Mix&Fix-Net operates on either the non-AIS or the AIS path independently (the two modalities are not fused or used simultaneously). The selected modality produces time-series trajectories that are passed through a data preprocessing stage and fed into Mix&Fix-Net to predict the trajectory.

## 3.1. Overall Modular Architecture of Mix&Fix-Net

Mix&Fix-Net is a dual-stage MLP-mixer-based model. In the first stage, it predicts the coarse trajectory of the vessel using the Primary Trajectory Predictor module, while refining the prediction in the second stage with the Residual Trajectory Adjuster module.

Primary Trajectory Predictor: The Primary Trajectory Predictor comprises zero-padding of the input, the Base Trajectory Predictor module, and the instance denormalization. Initially, we append T zero-padding entries to the end of the input $\pmb { X } _ { L } \in \mathbb { R } ^ { L \times C _ { i } }$ to obtain $\pmb { X } _ { P } \in \mathbb { R } ^ { N \times C _ { i } }$ , where $N = L + T$ . These padded values serve as placeholders for additional information during model processing, which is particularly beneficial when $T > L$ . Subsequently, the Base Trajectory Predictor processes $X _ { P }$ to generate the initial coarse trajectory prediction $\pmb { Y } _ { P } \in \mathbb { R } ^ { T \times C _ { o } }$ $Y _ { P }$ then undergoes instance denormalization [30], resulting final coarse trajectory prediction $Y _ { P } \in \mathbb { R } ^ { T \times C _ { o } }$

![](images/05e439ef7f1c836d9b67f97f26465a740c1b0845a31d670e2d2621655917d78e.jpg)  
Figure 2: Architecture of our proposed Mix&Fix-Net, an MLP-mixer-based model, for vessel trajectory prediction.

Instance denormalization facilitates the Base Trajectory Predictor’s ability to handle input distribution shifts, with statistical parameters (mean and standard deviation) computed from $X _ { L } .$

Residual Trajectory Adjuster: The Residual Trajectory Adjuster refines the coarse prediction by estimating the residual error. We concatenate the positional features of $X _ { L }$ with $\underline { { { \boldsymbol { Y } } _ { P } } } ,$ forming $X _ { e } \in \mathbb { R } ^ { N \times C _ { o } }$ , where $N = L + T$ . The Error Predictor then predicts the error $Y _ { e } \in \mathbb { R } ^ { T \times C _ { o } }$ . The final trajectory prediction $\hat { \pmb { Y } } _ { T } \in \mathbb { R } ^ { T \times C _ { o } }$ is computed as,

$$
\hat { \pmb { Y } } _ { T } = \pmb { Y } _ { P } + \pmb { Y } _ { e }\tag{1}
$$

This dual-stage design with coarse trajectory prediction followed by residual refinement improves prediction accuracy by efectively modeling both the primary trajectory and fine-grained adjustments.

## 3.1.1. Base Trajectory Predictor and Error Predictor Modules

The architectures of the Base Trajectory Predictor and Error Predictor modules are identical, difering only in the number of input features. The Base Trajectory Predictor module receives $C _ { i }$ features as input, whereas the Error Predictor module receives $C _ { o }$ features. Aside from this distinction, both modules consist of an embedding layer, multiple mixer modules, and a head module. Below, we demonstrate the sequential data processing of the Base Trajectory Predictor.

Initially, the input $\pmb { X } _ { P } \in \mathbb { R } ^ { N \times C _ { i } }$ of the Base Trajectory Predictor is embedded to $\pmb { X } _ { d } \in \mathbb { R } ^ { C _ { i } \times N \times d }$ employing an embedding layer, where d denotes the embedding dimension for each channel. Subsequently, $X _ { d }$ is processed by an MLP-Mixer module (Mixer-1) and outputs $X _ { m _ { 1 } }$ . Then $X _ { m _ { 1 } }$ is processed by the second MLP-Mixer module (Mixer-2), resulting in $X _ { m _ { 2 } }$ . We then add $X _ { m _ { 2 } }$ to $X _ { m _ { 1 } }$ and $X _ { d }$ through residual connections and perform layer normalization LN( ):

$$
\begin{array} { r } { \pmb { X _ { m } } = \pmb { \mathcal { L } } \pmb { N } ( \pmb { X _ { m _ { 2 } } } + \pmb { X _ { m _ { 1 } } } + \pmb { X _ { d } } ) \in \mathbb { R } ^ { C _ { i } \times N \times d } . } \end{array}\tag{2}
$$

Later, we permute $X _ { m } ,$ and obtain ${ \pmb X } _ { h } \in \mathbb { R } ^ { N \times C _ { i } \times d }$ . Finally, $X _ { h }$ undergoes the head module to obtain the initial coarse prediction $Y _ { p } \in \mathbb { R } ^ { T \times C _ { o } }$

In the following subsections, we provide a detailed description of the embedding layer, Mixer module, and Head module.

## 3.1.2. Embedding Layer

Embedding is performed to encode the data into a higher-dimensional space using a linear layer. Here, the embedding layer transforms $\pmb { X } _ { P } \in \mathbb { R } ^ { N \times C _ { i } }$ into a higher dimensional representation $X _ { d } \in \mathbb { R } ^ { N \times C _ { i } \cdot d }$ . The embedding layer $\mathcal { L } _ { e m b }$ is defined as,

$$
\mathcal { L } _ { e m b } : \mathbb { R } ^ { 1 \times C _ { i } } \longrightarrow \mathbb { R } ^ { 1 \times ( C _ { i } \cdot d ) } ,\tag{3}
$$

where d is a hyperparameter. After the linear transformation, each embedding vector is reshaped and permuted as follows,

$$
\mathbb { R } ^ { 1 \times C _ { i } } \xrightarrow { \mathrm { \mathrm { ~ e m b e d } } } \mathbb { R } ^ { 1 \times ( C _ { i } \cdot d ) } \xrightarrow { \mathrm { \mathrm { ~ r e s h a p e } } } \mathbb { R } ^ { 1 \times C _ { i } \times d } \xrightarrow { \mathrm { \ p e r m u t e } } \mathbb { R } ^ { C _ { i } \times 1 \times d }\tag{4}
$$

to obtain a d-dimensional representation for each channel.

## 3.1.3. Mixer Module

Both the Base Trajectory Predictor and Error Predictor modules employ two sequential mixer modules (Mixer-1 and Mixer-2) to facilitate multidirectional data mixing. Each mixer module also contains two distinct mixer blocks: the temporal mixer and the embedding mixer. The architecture of the temporal mixer and embedding mixer is similar to the token-mixing MLP described in [31]. Below, we describe the sequential data processing in the first mixer module (Mixer-1).

Initially, the input $\pmb { X } _ { d } \in \mathbb { R } ^ { C _ { i } \times N \times d }$ is normalized using layer-normalization, and permuted to obtain the dimension $\mathbb { R } ^ { d \times C _ { i } \times N }$ . This rearrangement ensures that the last dimension of the data corresponds to $N ,$ along which temporal mixing is subsequently applied.

Temporal Mixer: The temporal mixer mixes data along the temporal dimension N. It comprises two linear layers connected by a non-linear GELU activation. The dimensional transformation of the data in the temporal mixer is shown as,

$$
\mathbb { R } ^ { { \cdots } \times N } { \xrightarrow { \mathrm { L i n e a r } } } \mathbb { R } ^ { { \cdots } \times ( N \cdot t _ { f } ) } { \xrightarrow { \mathrm { G E L U } } } \mathbb { R } ^ { { \cdots } \times ( N \cdot t _ { f } ) } { \xrightarrow { \mathrm { L i n e a r } } } \mathbb { R } ^ { { \cdots } \times N }\tag{5}
$$

Here, the first linear layer mixes information along the N dimension and expands it by a factor of $t _ { f }$ , a hyperparameter. The final linear layer re-transforms the expanded dimension $( N \cdot t _ { f } )$ to the original N dimension.

The output of the temporal mixer, with dimension $\mathbb { R } ^ { d \times C _ { i } \times N }$ , is permuted and normalized using layer normalization to obtain R<sup>Ci×N×d</sup>. The permutation operation ensures that the last dimension of the data corresponds to $d ,$ along which the embedding mixing is performed subsequently.

Embedding Mixer: The Embedding mixer mixes data along the embedding dimension d. Similar to the temporal mixer, the embedding mixer also comprises two linear layers connected by a non-linear GELU activation. Dimensional transformation of the data in the embedding mixer is shown as,

$$
\mathbb { R } ^ { \cdots \times d } \xrightarrow { \mathrm { L i n e a r } } \mathbb { R } ^ { \cdots \times ( d \cdot d _ { f } ) } \xrightarrow { \mathrm { G E L U } } \mathbb { R } ^ { \cdots \times ( d \cdot d _ { f } ) } \xrightarrow { \mathrm { L i n e a r } } \mathbb { R } ^ { \cdots \times d }\tag{6}
$$

Here, the initial linear layer mixes the information along the d dimension and expands it by a factor of $d _ { f }$ , which is a hyperparameter. The final linear layer projects the expanded dimension $( d \cdot d _ { f } )$ back to the original dimension d.

To obtain the final output $\pmb { X _ { m _ { 1 } } } \in \mathbb { R } ^ { C _ { i } \times N \times d }$ from Mixer -1, the output of the embedding mixer module is added to its input, ensuring residual learning.

## 3.1.4. Head Module

The input to the Head module is denoted as $X _ { h } \in \mathbb { R } ^ { N \times C _ { i } \times d }$ , where each temporal instance, with shape R<sup>1×Ci×d</sup>, contains an embedding vector of length d for each channel. The objective of the head module is to transform each temporal representation from $\mathbb { R } ^ { 1 \times C _ { i } \times d }$ into $\mathbb { R } ^ { 1 \times C _ { i } }$ . Therefore, we first flatten the last two dimensions of R<sup>1×Ci×d</sup>, and subsequently perform a linear transformation using a linear layer:

$$
\mathbb { R } ^ { 1 \times C _ { i } \times d } \xrightarrow { \mathrm { ~ F l a t t e n } } \mathbb { R } ^ { 1 \times ( C _ { i } \cdot d ) } \xrightarrow { \mathrm { ~ L i n e a r } } \mathbb { R } ^ { 1 \times C _ { i } } .\tag{7}
$$

By performing these operations on the N temporal data points of $X _ { h } \in \mathbb { R } ^ { N \times C _ { i } \times d }$ , we obtain data with dimension $\mathbb { R } ^ { N \times C _ { i } }$ . Since our ultimate goal is to predict the T temporal instances with $C _ { o }$ features, we select the last T temporal data instances from N and retain the last $C _ { o }$ features from $C _ { i } ,$ resulting in the final output $Y _ { p } \in \mathbb { R } ^ { T \times C _ { o } }$

## 3.2. Data Collection

To evaluate the trajectory prediction performance of our model, we employ three benchmark datasets: one non-AIS dataset and two AIS-based datasets. For the non-AIS dataset, we introduce the Video-based Vessel Route (VVR) dataset <sup>2</sup>, in which vessel trajectories are extracted from video frames and represented as time-series data. For the AIS data, we utilize the Managing Maritime Movements (M3) Data Challenge dataset [32]<sup>3</sup>. Since M3 is not publicly available, we supplement our evaluation with a second, publicly available AIS dataset obtained from [33]. Figure 3 illustrates the vessel trajectory mappings for these datasets within their respective study areas. The following sections describe the collection and acquisition processes for these datasets in detail.

![](images/0c8471b8cc263af32eef4d05e78cc2afa6cc70f540af3aabcac53141556cde86.jpg)  
(a) VVR

![](images/ac357242872544019fd6f52a2e7b5e3b586a817f5d7bdcf02d8590b552da9ce9.jpg)  
(b) M3

![](images/21d670afce568fa8e3342756b3bf130dc97ecf866c9c4f214c81ffd3904dd2bc.jpg)  
(c) TampaBay

Figure 3: Vessel trajectory maps of the study areas for the VVR, M3, and TampaBay datasets.  
![](images/90576e7c35af9785ab2674f907d54b902c26b680dac0b7204f6236e2c22d704c.jpg)  
Figure 4: Vessel tracking in the VVR dataset in diferent weather conditions. Each vessel is assigned a new vessel-ID.

## 3.2.1. Collection of non-AIS data

Our proposed model predicts vessel trajectories based on data provided in a time-series format. Consequently, the model is applicable to any domain where sequential time-series data can be obtained. While AIS data is inherently sequential and can be directly adopted for the prediction task, our model is also evaluated using non-AIS data. Specifically, we employ non-AIS time series that are generated from videos. Unlike AIS data, video-based data requires stable recordings to capture vessel motion. Publicly available surveillance streams often fail to meet this requirement due to limited coverage or moving cameras. To overcome this, we selected a real-time webcam feed from the Canal do Porto de Santos, an estuary in Santos, Brazil, which continuously records vessel trafic from a fixed viewpoint [34]. The dataset consists of six distinct video recordings spanning three days, totaling 24 hours of video at a resolution of 1366×768 and a frame rate of 30 fps. The videos capture vessels of varying sizes, ranging from small boats to large cargo ships, and span diverse environmental conditions throughout the day. This variability enhances the diversity of the dataset, which can be used to assess algorithm robustness. To transform video footage into time-series data, we utilize the following procedure:

• Vessel Detection: First, a pre-trained YOLO11 model is fine-tuned using manually annotated vessel images from the first two videos. This tuned model is then used to detect vessels within the video frames. The model generates a bounding box for each detected vessel, and the bounding box center is used as the vessel’s position.

• Tracking: Using the fine-tuned detector, we first detect vessels on the remaining four videos and track them using the ByteTrack algorithm. Tracker assigns a unique vessel ID to each tracked trajectory. Examples of vessel tracking are shown in Figure 4. This tracking process links positions across video frames to specific timestamps.

As a result, we obtain a comprehensive time-series dataset containing vessel IDs, timestamps, and corresponding xy coordinates (center of bounding boxes). The values of x and y range from 0 to 1, where (0, 0) corresponds to the top-left corner and (1, 1) corresponds to the bottomright corner on the video frame. The details of the Video-based Vessel Route (VVR) dataset are shown in Table 1. Detailed implementations of the YOLO model and ByteTracking are provided in Appendix A.

## 3.2.2. Acquisition of AIS datasets

Managing Maritime Movements (M3) dataset: The M3 dataset is a synthetic time series dataset that provides latitude and longitude coordinates for 10,000 vessels in an area with latitude from 14.817 to 22.300 and longitude from -69.375 to -62.930. The initial features include profile ID, vessel ID, timestamp, latitude, and longitude. The profile ID ranges from 1 to 10 and corresponds to the portfolio that governs how trajectories are generated. The time frame of the dataset is from January 14, 2022, to January 16, 2022. The M3 dataset contains two sets: the training set, comprising trajectories for 8,000 vessels, and the test set, reserved for data challenge submissions, which includes trajectories for 2,000 vessels. We utilize the training set of the data challenge to train and validate our proposed model.

TampaBay dataset: We collect a real-world AIS dataset covering the period from January 01, 2024, to January 20, 2024, via the marinecadastre.gov website. The raw data is filtered to focus on the Tampa Bay area, specifically within the geographic bounds of $2 7 . 2 7 ^ { \circ }$ to 28<sub>.</sub>12<sup>◦</sup> Latitude and $- 8 3 . 4 0 ^ { \circ } \ \mathrm { t o } \ - 8 2 . 2 8 ^ { \circ }$ Longitude. The dataset includes key features such as Maritime Mobile Service Identity (MMSI), timestamp, vessel type, latitude, and longitude, where MMSI and vessel type serve as functional equivalents to the Vessel ID and Profile ID features used in the M3 dataset. In total, the TampaBay dataset comprises 981 unique vessel IDs and 26 distinct vessel types.

## 3.3. Dataset Preprocessing

We perform a series of preprocessing steps on the time-series trajectories of the TampaBay, M3, and VVR datasets before training our trajectory prediction model. These steps include removing duplicate time instances, interpolating missing values, and performing feature engineering. Below, we demonstrate these steps in a sequential order.

## 3.3.1. Removing duplicate time instances

We identify several repeated data points with identical timestamps in the trajectory of the M3 and TampaBay datasets. To resolve this issue, we retain only the first data point and remove all subsequent repeated data points along the trajectory. This step is not applied to the VVR dataset because it contains no repeated data points.

## 3.3.2. Interpolating missing values

We observe that the tracker model fails to detect the vessels on certain video frames. As a result, the vessels’ trajectories lack uniform time sampling. Similarly, the M3 and TampaBay time series datasets also miss several temporal data instances. Therefore, we apply linear interpolation to estimate and fill the missing data points. After interpolating the missing values, the sampling periods are 1/30 second for the VVR dataset, and 1 minute for the M3 and TampaBay datasets. Since the sampling rate of the VVR dataset is excessively fine-grained, predicting a trajectory of a certain length requires a lot of data points, which makes the computation expensive. To mitigate this, we downsample the VVR dataset by a factor of 10, resulting in a final sampling period of 1/3 second.

## 3.3.3. Adding additionalfeatures

Initially, the M3 and TampaBay include profile ID, vessel ID, timestamp, latitude, and longitude, while the VVR dataset includes the same features with the exception of the profile ID. We transform the timestamp into day (D), hour (H), and minute (M) values. We then compute the sine and cosine values of D, H, and M by the following formulas:

$$
M _ { \mathrm { { s i n } } } = \sin \biggl ( { \frac { 2 \pi M } { 6 0 } } \biggr )
$$

$$
M _ { \mathrm { { c o s } } } = \mathrm { { c o s } } \Bigg ( \frac { 2 \pi M } { 6 0 } \Bigg )\tag{8}
$$

$$
H _ { \mathrm { s i n } } = \mathrm { s i n } \Bigg ( \frac { 2 \pi H } { 2 4 } \Bigg )
$$

$$
H _ { \mathrm { c o s } } = \cos \left( { \frac { 2 \pi H } { 2 4 } } \right)\tag{9}
$$

$$
D _ { \mathrm { s i n } } = \mathrm { s i n } \bigg ( \frac { 2 \pi D } { 7 } \bigg )
$$

$$
D _ { \mathrm { c o s } } = \cos \left( { \frac { 2 \pi D } { 7 } } \right)\tag{10}
$$

We also incorporate displacement $D _ { d i s p }$ as a feature, defined as the distance traveled between consecutive time steps. Consequently, the final features for the VVR are $M _ { s i n } , M _ { c o s } , H _ { s i n } , H _ { c o s }$ $D _ { s i n } , D _ { c o s } , D _ { d i s p } ,$ , x-coordinate (longitude) and y-coordinate (latitude), while the M3 and TampaBay additionally include the profile ID. Therefore, the VVR dataset consists of 9 features, whereas the M3 and TampaBay consist of 10 features.

## 4. Experiments

This section presents the experimental framework, including dataset splitting, dataset normalization, evaluation metrics, baselines, training configurations, results, and ablation studies.

## 4.1. Dataset Splitting

In trajectory prediction literature, two primary splitting strategies exist: (1) extracting sequences via a rolling window before distributing them into sets [35, 36], and (2) partitioning raw data by vessel ID or time prior to sequence extraction [37, 38]. The former approach carries a risk of data leakage because overlapping segments of the same trajectory may appear in both training and testing sets. Some studies use hybrid variants of these methods to handle varying testing conditions [39]. In this work, we evaluate our model using both experimental settings to ensure a robust analysis of generalization:

Setting-1: To strictly prevent data leakage and evaluate the model’s ability to generalize to unseen vessels, we first partition the dataset into training, validation, and test sets with a ratio of 3:1:1, based on vessel IDs. For the TampaBay dataset, we maintain this ratio within each vessel profile and for each day; if a profile lacks suficient vessel IDs to satisfy this ratio, it is excluded. For the M3 dataset, the ratio is maintained within each profile, while for the VVR dataset, it is applied across all vessel IDs. Following the split, we extract observation-prediction pairs from each vessel’s trajectory using a one-step stride rolling window. To focus on active movement, we exclude pairs where the standard deviation of coordinates in the observation window is less than 0 001. This setting ensures the model is evaluated on entirely new trajectories not encountered during training.

Setting-2: We first apply the same rolling window and zero-motion filtering to each trajectory independently. These filtered pairs are then randomly distributed into training, validation, and testing sets with a 3:1:1 ratio.

In both settings, we set the observation and prediction window lengths for all datasets to 60 and 120, respectively, aligning with the criteria of the M3 data challenge. This choice also considers the trade-of with the minimum required trajectory length (L + T), as many trajectories are shorter. Additionally, reducing the window size negatively afects the model performance. A detailed justification for the selected window lengths is provided in Appendix C.2. Total sequence counts for both configurations are reported in Table 2.

Table 1: Details of the VVR dataset. The dataset consists of six video recordings. Vessels are manually annotated in two videos to train the vessel detector, while the remaining four videos are used to convert vessel trajectories into time series representations. Times reported in the table correspond to local time.
<table><tr><td>S/N</td><td>Recording Date</td><td>Start time</td><td>Duration</td><td>Purpose</td></tr><tr><td>0</td><td>2/12/2025</td><td>13:02:00</td><td>6 Hours</td><td>Detector training</td></tr><tr><td>1</td><td>11/19/2024</td><td>8:01:00</td><td>4 Hours</td><td>Detector training</td></tr><tr><td>2</td><td>11/19/2024</td><td>12:17:00</td><td>4 Hours</td><td>Trajectory Prediction</td></tr><tr><td>3</td><td>11/19/2024</td><td>16:26:00</td><td>38 Minutes</td><td>Trajectory Prediction</td></tr><tr><td>4</td><td>11/20/2024</td><td>8:12:00</td><td>4 Hours</td><td>Trajectory Prediction</td></tr><tr><td>5</td><td>2/12/2025</td><td>7:50:00</td><td>5 Hours</td><td>Trajectory Prediction</td></tr></table>

Table 2: Number of observation-prediction sequence pairs.
<table><tr><td rowspan="2"></td><td colspan="3">Setting-1</td><td colspan="3">Setting-2</td></tr><tr><td>M3</td><td>TampaBay</td><td>VVR</td><td>M3</td><td>TampaBay</td><td>VVR</td></tr><tr><td>Train</td><td>367039</td><td>133740</td><td>119312</td><td>369845</td><td>200719</td><td>117421</td></tr><tr><td>Validation</td><td>127752</td><td>49651</td><td>37361</td><td>123282</td><td>66940</td><td>39140</td></tr><tr><td>Test</td><td>121618</td><td>60817</td><td>39029</td><td>123282</td><td>67046</td><td>39141</td></tr><tr><td>Total</td><td>616409</td><td>244208</td><td>195702</td><td>616409</td><td>334705</td><td>195702</td></tr></table>

## 4.2. Dataset Normalization

We normalize both the training and test sets using the mean and standard deviation (znormalization) derived from the training set. To be more precise, the input to our model $X _ { L }$ denotes the normalized input. Since the predicted output $\hat { Y } _ { T }$ is compared against the normalized ground truth $Y _ { T }$ during model training, $\hat { Y } _ { T }$ also represents the normalized prediction. However, when evaluating model performance, we denormalize both $Y _ { T }$ and $\hat { Y } _ { T }$ to their original scale, denoted as $\pmb { Y } \in \mathbb { R } ^ { T \times C _ { o } }$ and $\hat { \pmb { Y } } \in \mathbb { R } ^ { T \times C _ { c } }$ <sup>o</sup> , respectively. Here, $C _ { o }$ indicates the number of the output features, which corresponds to either xy-coordinates or the latitude-longitude position of the vessel.

While AIS and VVR datasets utilize diferent coordinate systems (geographic vs. imageplane), they are unified through z-normalization to focus on motion dynamics. A detailed discussion on the semantic consistency and coordinate-agnostic nature of our model is provided in Appendix D.

## 4.3. Evaluation Metrics

To comprehensively evaluate the performance of our model, we employ six evaluation metrics, as outlined in [40]: MSE, MAE, SMAPE, FDE, FD, and AED. MSE (Mean Squared Error) measures the average squared distance between predicted and true positions at each time step, whereas MAE (Mean Absolute Error) captures the average absolute diference between them. SMAPE (Symmetric Mean Absolute Percentage Error) quantifies the relative diference between the predicted and true positions. FDE (Final Displacement Error) measures the Euclidean distance between the predicted and true final positions, while FD (Frechet Distance) captures the maximum distance between the predicted and true trajectories. Finally, AED (Average Euclidean Distance) computes the mean Euclidean distance between predicted and true positions across the predicted time steps. We compute the six metrics for each predicted sequence by the following formulas,

$$
\mathrm { M S E } = \frac { 1 } { T } \sum _ { i = 1 } ^ { T } \left. \hat { \mathbf { y } } _ { i } - \mathbf { y } _ { i } \right. _ { 2 } ^ { 2 }
$$

$$
\mathrm { S M A P E } = \frac { 1 } { T } \sum _ { i = 1 } ^ { T } \frac { \Vert \hat { \mathbf { y } } _ { i } - \mathbf { y } _ { i } \Vert _ { 1 } } { ( \Vert \hat { \mathbf { y } } _ { i } \Vert _ { 1 } + \Vert \mathbf { y } _ { i } \Vert _ { 1 } ) / 2 }
$$

$$
\mathrm { M A E } = \frac { 1 } { T } \sum _ { i = 1 } ^ { T } \| \hat { \pmb { y } } _ { i } - \pmb { y } _ { i } \| _ { 1 }
$$

$$
\mathrm { F D } = \operatorname* { m a x } _ { i \in [ 1 , T ] } \| \hat { \pmb { y } } _ { i } - \mathbf { y } _ { i } \| _ { 2 }
$$

$$
\mathrm { F D E } = \| \hat { \mathbf { y } } _ { T } - \mathbf { y } _ { T } \| _ { 2 }
$$

$$
\mathrm { A E D } = \frac { 1 } { T } \sum _ { i = 1 } ^ { T } \Vert \hat { \pmb { y } } _ { i } - \pmb { y } _ { i } \Vert _ { 2 }
$$

where, ∥ ∥<sub>2</sub> and $\lVert . \rVert _ { 1 }$ refer to L2 norm and L1 norm, respectively, $\hat { \mathbf { y } } _ { i } \in \mathbb { R } ^ { 1 \times 2 }$ denotes the $i -$ th predicted position in the predicted trajectory ${ \hat { Y } } ,$ and $\mathbf { y } _ { i } \in \mathbb { R } ^ { 1 \times 2 }$ denotes the corresponding true position in the ground-truth trajectory $Y .$ The final value for each metric is obtained by averaging again over the total number of sequences.

## 4.4. Baselines

The number of publicly available codes for existing trajectory prediction models is limited. Most existing models are adapted from widely used architectures that have demonstrated strong performance in sequential forecasting tasks, as trajectory prediction is closely aligned with time series forecasting. Recently, Transformer, MLP-mixer, and RNN-based models have achieved notable success in time series forecasting. Therefore, as baselines for our model, we select four variants that have shown superior performance in forecasting: P\_sLSTM [41], PatchTST [42], DLinear [16], and WPMixer [18]. In addition, we include a domain-specific trajectory prediction model, FusionRNN [38].

Among these, P\_sLSTM is an sLSTM-based model that incorporates patching and was originally proposed for time series forecasting. To enhance the efectiveness of Transformer models for forecasting, PatchTST introduces patching into the Transformer architecture, achieving superior performance compared to other Transformer-based variants. DLinear is a simple yet efective MLP-based model that challenges the Transformer-based models by questioning their performance for the forecasting task. Recently, MLP-Mixer-based models have often outperformed Transformer variants. WPMixer is one such model, extending the MLP-mixer framework by utilizing wavelet decomposition in combination with the patching technique. FusionRNN [38] is a fusion-based RNN architecture specifically proposed for AIS-based vessel trajectory prediction, which demonstrated state-of-the-art performance on its corresponding real-world AIS datasets.

## 4.5. Training and Hyperparameters

During training, model optimization is carried out with the Adam optimizer and MSE loss function. All experiments on the VVR and TampaBay datasets are conducted using a single NVIDIA GeForce RTX 4090 GPU. Given that the M3 dataset is approximately three times larger than the VVR dataset, training is accelerated by utilizing two NVIDIA A100 GPUs. All experiments on the VVR and TampaBay datasets used a batch size of 128, while those on the M3 dataset used a batch size of 512.

For hyperparameter tuning, we utilize Optuna, a hyperparameter tuning framework. The search space includes: embedding dimension d ∈ {16 32 64 128 256}, embedding mixer factor

$d _ { f } \in \{ 1 , 3 , 5 , 7 \}$ , temporal mixer factor $t _ { f } \in \{ 1 , 3 , 5 , 7 \}$ , and learning rate ranging from 0.01 to 0.00001. The maximum number of epochs is fixed at 50 with an early-stopping patience of 10 for all experiments. The detailed hyperparameter settings are reported in Table A.10 in the Appendix.  
Table 3: Performance comparison under data split setting-1. For each model, the best results are reported. The bestperforming model is highlighted in bold, while the second-best is underlined.
<table><tr><td rowspan="11">VΛR</td><td></td><td>Mix&amp;Fix-Net</td><td>WPMixer</td><td>P_sLSTM</td><td>PatchTST</td><td>Dlinear</td><td>FusionRNN</td></tr><tr><td>MSE</td><td>5.22E-04</td><td>7.76E-04</td><td>6.71E-04</td><td>8.18E-04</td><td>9.83E-04</td><td>9.22E-04</td></tr><tr><td>MAE</td><td>1.58E-02</td><td>1.61E-02</td><td>1.58E-02</td><td>1.66E-02</td><td>2.05E-02</td><td>2.35E-02</td></tr><tr><td>SMA.</td><td>4.60E-02</td><td>5.22E-02</td><td>4.78E-02</td><td>5.44E-02</td><td>6.31E-02</td><td>7.01E-02</td></tr><tr><td>FDE</td><td>2.69E-02</td><td>3.05E-02</td><td>2.95E-02</td><td>3.27E-02</td><td>3.99E-02</td><td>3.60E-02</td></tr><tr><td>FD</td><td>3.21E-02</td><td>3.29E-02</td><td>3.13E-02</td><td>3.36E-02</td><td>4.06E-02</td><td>3.96E-02</td></tr><tr><td>AED</td><td>1.28E-02</td><td>1.30E-02</td><td>1.30E-02</td><td>1.35E-02</td><td>1.63E-02</td><td>1.88E-02</td></tr><tr><td rowspan="7">MM</td><td>MSE</td><td>1.38E-03</td><td>2.95E-03</td><td>4.19E-03</td><td>3.16E-03</td><td>3.69E-03</td><td>5.60E-03</td></tr><tr><td>MAE</td><td>2.60E-02</td><td>3.11E-02</td><td>4.24E-02</td><td>3.20E-02</td><td>3.51E-02</td><td>4.02E-02</td></tr><tr><td>SMA.</td><td>2.89E-03</td><td>1.83E-03</td><td>1.07E-02</td><td>1.79E-03</td><td>3.45E-03</td><td>2.03E-02</td></tr><tr><td>FDE</td><td>4.28E-02</td><td>6.36E-02</td><td>8.15E-02</td><td>6.80E-02</td><td>7.23E-02</td><td>5.41E-02</td></tr><tr><td>FD</td><td>5.08E-02</td><td>6.44E-02</td><td>8.42E-02</td><td>6.86E-02</td><td>7.41E-02</td><td>6.03E-02</td></tr><tr><td>AED</td><td>2.07E-02</td><td>2.46E-02</td><td>3.47E-02</td><td>2.55E-02</td><td>2.79E-02</td><td>3.17E-02</td></tr><tr><td>MSE</td><td>2.69E-03</td><td>3.19E-03</td><td>2.92E-03</td><td>3.28E-03</td><td>3.18E-03</td><td>3.66E-03</td></tr><tr><td rowspan="6">Taay</td><td>MAE</td><td>3.88E-02</td><td>3.87E-02</td><td>4.02E-02</td><td>4.08E-02</td><td>4.27E-02</td><td>4.92E-02</td></tr><tr><td>SMA.</td><td>9.18E-04</td><td>9.20E-04</td><td>9.51E-04</td><td>9.64E-04</td><td>1.00E-03</td><td>1.16E-03</td></tr><tr><td>FDE</td><td>5.53E-02</td><td>6.15E-02</td><td>6.20E-02</td><td>6.34E-02</td><td>6.77E-02</td><td>6.46E-02</td></tr><tr><td>FD</td><td>6.33E-02</td><td>6.57E-02</td><td>6.63E-02</td><td>6.66E-02</td><td>6.99E-02</td><td>7.08E-02</td></tr><tr><td>AED</td><td>3.04E-02</td><td>3.07E-02</td><td>3.19E-02</td><td>3.23E-02</td><td>3.37E-02</td><td>3.83E-02</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 4.6. Results

The quantitative performance of our Mix&Fix-Net is compared with other baselines under both data-splitting settings. Table 3 and Table 4 show the results for setting-1 and setting-2, respectively. For the M3 dataset, performance is evaluated for each profile and then averaged across profiles. All baselines are reproduced using their publicly available code, with an extensive hyperparameter search using optuna, following their recommended ranges for the hyperparameters.

Setting-1 (Table 3) provides a rigorous and realistic evaluation of generalization, as the model is tested on entirely unseen vessels. As shown, Mix&Fix-Net almost consistently outperforms the baselines, confirming that it maintains superior predictive power even when memorization is not possible. In contrast, Setting-2 (Table 4) uses random data splitting, under which the performance improvement of our model is considerably larger. However, this setting inherently carries a risk of data leakage due to overlapping trajectory segments between splits; its results should therefore be interpreted as a measure of the model’s capacity to memorize and represent complex spatial patterns rather than to generalize to new vessels. For visualization, examples of randomly selected predictions are shown in Figure 5 and Figure A.6. It is observed that the predicted trajectories of our model follow the ground-truth more accurately than the other baselines. However, the predicted trajectories exhibit some noise, which is analyzed in detail in Appendix B.

Table 4: Performance comparison under data split setting-2. For each model, the best results are reported. The bestperforming model is highlighted in bold, while the second-best is underlined. Note that this setting involves a risk of temporal data leakage due to overlapping trajectory segments; consequently, these results do not reflect the realistic generalization performance on unseen vessels.
<table><tr><td></td><td></td><td>Mix&amp;Fix-Net</td><td>WPMixer</td><td>P_sLSTM</td><td>PatchTST</td><td>Dlinear</td><td>FusionRNN</td></tr><tr><td rowspan="11">VΛR</td><td>MSE</td><td>3.86E-06</td><td>7.38E-04</td><td>5.33E-04</td><td>8.33E-04</td><td>1.16E-03</td><td>9.43E-06</td></tr><tr><td>MAE</td><td>1.45E-03</td><td>1.59E-02</td><td>1.45E-02</td><td>1.72E-02</td><td>2.20E-02</td><td>2.58E-03</td></tr><tr><td>SMA.</td><td>4.08E-03</td><td>5.04E-02</td><td>4.40E-02</td><td>5.47E-02</td><td>6.76E-02</td><td>7.47E-03</td></tr><tr><td>FDE</td><td>1.64E-03</td><td>3.01E-02</td><td>2.68E-02</td><td>3.29E-02</td><td>4.15E-02</td><td>3.13E-03</td></tr><tr><td>FD</td><td>3.72E-03</td><td>3.25E-02</td><td>2.85E-02</td><td>3.41E-02</td><td>4.23E-02</td><td>5.35E-03</td></tr><tr><td>AED</td><td>1.16E-03</td><td>1.29E-02</td><td>1.19E-02</td><td>1.39E-02</td><td>1.75E-02</td><td>2.06E-03</td></tr><tr><td>MSE</td><td>8.39E-05</td><td>2.82E-03</td><td>3.83E-03</td><td>2.90E-03</td><td>3.87E-03</td><td>6.93E-04</td></tr><tr><td rowspan="6">MM</td><td>MAE</td><td>7.24E-03</td><td>2.82E-02</td><td>4.03E-02</td><td>3.03E-02</td><td>3.46E-02</td><td>1.75E-02</td></tr><tr><td>SMA.</td><td>4.13E-04</td><td>1.46E-03</td><td>7.08E-03</td><td>1.58E-03</td><td>2.72E-03</td><td>3.64E-03</td></tr><tr><td>FDE</td><td>1.10E-02</td><td>5.98E-02</td><td>8.00E-02</td><td>6.46E-02</td><td>7.15E-02</td><td>2.84E-02</td></tr><tr><td>FD</td><td>1.54E-02</td><td>6.05E-02</td><td>8.16E-02</td><td>6.52E-02</td><td>7.25E-02</td><td>3.33E-02</td></tr><tr><td>AED</td><td>5.90E-03</td><td>2.25E-02</td><td>3.30E-02</td><td>2.41E-02</td><td>2.77E-02</td><td>1.40E-02</td></tr><tr><td>MSE</td><td>4.72E-04</td><td>2.58E-03</td><td>1.44E-03</td><td>2.59E-03</td><td>2.88E-03</td><td>9.49E-04</td></tr><tr><td rowspan="6">Taay</td><td>MAE</td><td>1.76E-02</td><td>3.40E-02</td><td>2.66E-02</td><td>3.37E-02</td><td>3.92E-02</td><td>2.53E-02</td></tr><tr><td>SMA.</td><td>4.14E-04</td><td>8.00E-04</td><td></td><td></td><td>9.12E-04</td><td></td></tr><tr><td>FDE</td><td></td><td></td><td>6.23E-04</td><td>7.89E-04</td><td></td><td>6.06E-04</td></tr><tr><td>FD</td><td>2.50E-02 3.08E-02</td><td>5.59E-02 5.89E-02</td><td>4.40E-02 4.67E-02</td><td>5.57E-02 5.84E-02</td><td>6.39E-02 6.61E-02</td><td>3.35E-02</td></tr><tr><td>AED</td><td>1.39E-02</td><td>2.70E-02</td><td>2.13E-02</td><td>2.68E-02</td><td>3.09E-02</td><td>3.89E-02 1.99E-02</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Among the baselines, the best-performing model after Mix&Fix-Net is FusionRNN in terms of memorization capability (Setting-2). However, FusionRNN is outperformed by WPMixer and P\_sLSTM in terms of generalization capability (Setting-1). The VVR time-series dataset, generated by tracking vessels in videos, inherently contains noise, as illustrated in Figure 5. This noise limits the efectiveness of models that rely on decomposition, particularly WPMixer.

![](images/0f36b40f24dc11c6f4cf931259f3cba27618ef693e43406474a49c3100ac43b9.jpg)

![](images/c88c220aa768122628b2e4152866e6426232ea25c33be0ad1615ba46fdb9cc9b.jpg)

![](images/03a481c309ed7dfca1bcc97a1c3ecf6f0256c365292eadef2ba9bed0dba0559b.jpg)

![](images/ae6669d4954cdddce119a9424cd63cd311b92a14cf409e1385e53600f5f5342c.jpg)

![](images/fa8d0972b5c4bf83907526914c473e3dce708a3ef751a7536f6c9b893d5d503a.jpg)

![](images/0087e11c4127e59ea38f346c515814a35fdd9564b3a965dba7090ee8c9b6940e.jpg)

![](images/b38ee3bb315b72b703376373dfd6a7c83756941b3d65e77e3dec5aee0618e9dd.jpg)

![](images/6737a2a67e05863420f8c92a36990a5d80a28fa93f42c181a74710b5c53f4f03.jpg)

![](images/031669d330e486e0f0cb60a56d26c4c8ab56e6006abca42c11850dda37504bf3.jpg)

![](images/c34751243f28fd56608ae573f1c41068e471b22bf24902facad3cfe6d407e2ff.jpg)

![](images/936f66f338401fe3f162403226ace0c04588f2b3e98b699a331d4bbe1adb43e7.jpg)

![](images/705ee0f6c310825a975594c170190fb5ab76c2235308153612fc9eb92a895ed5.jpg)

![](images/d44ba2908d1c07b927bd9639eb5d20c48d88a75bb581dc6cd3b1c3c5ed249a8f.jpg)

![](images/a438dd407403f90be6e582fce08f25178fa29be418b72d2521110299c5134237.jpg)

![](images/ff83460e6007e7bf2811c854a5b159fe086bf19b934948cfb20f2d3fd14defda.jpg)  
Figure 5: Visual comparison of the prediction among diferent models. Trajectories are randomly selected. The first two rows refer to the VVR dataset, while the last two rows refer to the M3 dataset. Predicted trajectories for the TampaBay dataset are shown in Figure A.6 in the Appendix.

In contrast, the M3 dataset is synthetic and noise-free. Therefore, WPMixer benefits from decomposition on this dataset compared to P\_sLSTM. Additionally, the simple MLP-based model DLinear performs the worst on almost all datasets and settings, despite its competitive or even superior performance compared to transformer-based models in standard time-series forecasting tasks.

## 4.7. Computational Robustness and Eficiency

We evaluate all models with four diferent random seed values under the data-split setting-2, and report the average and the standard deviation values in Table 5. The results indicate that our model is more robust compared to the other models.

Additionally, we present the computational complexity of our model in Table 6, measured by the average train iteration time, test iteration time, and GFLOPs (Giga Floating Point Operations) on the VVR dataset. While iteration time is a machine-dependent metric, GFLOPs is machineindependent. For training and testing, the batch size and number of iterations are identical for all models. Details of the machine specifications and hyperparameters are provided in Section 4.5.

The results in Table 6 demonstrate that although our model incurs a significantly higher GFLOPs value than the competing methods, the train and test times remain comparable across all models. Several factors contribute to this increased GFLOPs value. Unlike WPMixer, P\_sLSTM, and PatchTST, our approach does not utilize the patching technique. Furthermore, we incorporate input padding at the beginning of the model, which triples the size of the data being processed. We also introduce the Residual Trajectory Adjuster module at the end of our model to refine coarse predictions. However, GPU parallelization enables eficient handling of these additional arithmetic operations. Additionally, as demonstrated in the ablation study (Section 5), both the padding step and the Residual Trajectory Adjuster significantly enhance the accuracy of trajectory prediction, thereby justifying the additional computational overhead.

Further, our method is suitable for real-time deployment, as demonstrated in this video <sup>4</sup>. To assess practical deployability, we measure inference latency on a single NVIDIA RTX 4090: each test batch (size 128) takes 0.030 s, i.e., about 0.23 ms per trajectory prediction (over 4,000 predictions/s), each spanning a 120-step horizon. The high GFLOPs thus does not translate into a proportional latency cost. This throughput is well within the shore-station and port-authority scenarios we target, which use GPU-class hardware. For more constrained on-board or edge settings, a lightweight variant with a smaller embedding dimension d (Figure C.7) reduces GFLOPs substantially at a modest accuracy cost. The accuracy–eficiency trade-of therefore remains favorable, as the practical cost is the wall-clock latency, which stays within real-time.

## 5. Ablation Study

In this section, we present ablation studies on the contributions of individual modules. Additional ablation studies on the role of non-positional features and the sensitivity analysis of the hyperparameters are demonstrated in Appendix C. For all experiments, data-splitting setting-2 is employed with independent hyperparameter tuning.

## 5.1. Impact of diferent modules in Mix&Fix-Net

In this section, we evaluate the impact of diferent modules on the overall performance of our model. We conduct experiments with eight cases, where in each case, we remove or disable a diferent module. For fairness, the hyperparameters are tuned separately for every case. The results, presented in Table 7, report the percentage change of six metrics relative to the base case (Case-1). The base case (Case-1) corresponds to our proposed Mix&Fix-Net model.

In Case-2, the Residual Trajectory Adjuster is removed. Removing it increases the error across all metrics, with MSE, MAE, SMAPE, FDE, FD, and AED rising by about 2–14 percent. In Case-4, the embedding mixer is removed, which severely degrades performance, with MSE increasing by 322 percent.

To mitigate distributional shift in time series, RevIN (Reversible Instance Normalization) was introduced in [30]. It applies instance normalization to the input at the beginning of the model and instance denormalization to the output at the end. However, in our model, we employ only the instance denormalization after the Base Trajectory Predictor module. This design choice reflects the fact that vessel trajectory prediction depends not only on the trajectory pattern but also on the vessel’s absolute position in the sea area. Instance normalization eliminates this localized positional information. Therefore, we omit instance normalization but retain instance denormalization. The reason for retaining the instance denormalization is to restore the scale and distribution of the predicted trajectory. This ensures that the output remains consistent with the original spatial domain. This efect is validated in Case-5 and Case-6. In Case-5, when both normalization and denormalization are applied, performance degrades significantly. In Case-6, when both are removed, performance also declines, though less severely. These results show that denormalization alone is beneficial while normalization is detrimental for this task.

Table 5: Robustness comparison. Each model is evaluated using four random seed values.
<table><tr><td></td><td></td><td colspan="2">Mix&amp;Fix-Net</td><td colspan="2">WPMixer</td><td colspan="2">P_sLSTM</td></tr><tr><td></td><td></td><td>Mean</td><td>Std.</td><td>Mean</td><td>Std.</td><td>Mean</td><td>Std.</td></tr><tr><td rowspan="5">VVR</td><td>MSE</td><td>4.12E-06</td><td>2.09E-07</td><td>7.65E-04</td><td>1.89E-05</td><td>5.33E-04</td><td>4.51E-06</td></tr><tr><td>MAE</td><td>1.46E-03</td><td>3.29E-05</td><td>1.61E-02</td><td>1.69E-04</td><td>1.43E-02</td><td>1.46E-04</td></tr><tr><td>SMAPE</td><td>4.11E-03</td><td>9.49E-05</td><td>5.12E-02</td><td>5.68E-04</td><td>4.37E-02</td><td>2.59E-04</td></tr><tr><td>FDE</td><td>1.64E-03</td><td>2.39E-05</td><td>3.09E-02</td><td>5.94E-04</td><td>2.67E-02</td><td>9.26E-05</td></tr><tr><td>FD AED</td><td>3.71E-03</td><td>4.89E-05</td><td>3.35E-02</td><td>7.89E-04</td><td>2.84E-02</td><td>1.05E-04</td></tr><tr><td rowspan="7"></td><td></td><td>1.17E-03</td><td>2.62E-05</td><td>1.31E-02</td><td>1.33E-04</td><td>1.17E-02</td><td>1.02E-04</td></tr><tr><td>MSE</td><td>1.60E-04</td><td>1.14E-04</td><td>2.94E-03</td><td>8.35E-05</td><td>3.67E-03</td><td>2.97E-04</td></tr><tr><td>MAE</td><td>9.14E-03</td><td>2.75E-03</td><td>2.97E-02</td><td>1.41E-03</td><td>3.92E-02</td><td>1.08E-03</td></tr><tr><td>SMAPE</td><td>4.85E-04</td><td>1.21E-04</td><td>1.45E-03</td><td>3.92E-05</td><td>6.83E-03</td><td>6.98E-04</td></tr><tr><td>FDE</td><td>1.46E-02</td><td>5.21E-03</td><td>6.11E-02</td><td>1.29E-03</td><td>7.75E-02</td><td>1.94E-03</td></tr><tr><td>FD</td><td>1.98E-02</td><td>5.86E-03</td><td>6.18E-02</td><td>1.31E-03</td><td>7.94E-02</td><td>1.79E-03</td></tr><tr><td>AED</td><td>7.51E-03</td><td>2.29E-03</td><td>2.38E-02</td><td>1.06E-03</td><td>3.21E-02</td><td>9.81E-04</td></tr><tr><td rowspan="6">TampaB.</td><td>MSE</td><td>4.61E-04</td><td>2.57E-05</td><td>2.60E-03</td><td>2.08E-05</td><td>1.50E-03</td><td>6.15E-05</td></tr><tr><td>MAE</td><td>1.79E-02</td><td>4.02E-04</td><td>3.41E-02</td><td>1.38E-04</td><td>2.71E-02</td><td>5.12E-04</td></tr><tr><td>SMAPE</td><td>4.21E-04</td><td>1.00E-05</td><td>8.03E-04</td><td>3.77E-06</td><td>6.34E-04</td><td>1.26E-05</td></tr><tr><td>FDE</td><td>2.43E-02</td><td>7.81E-04</td><td>5.59E-02</td><td>1.36E-04</td><td>4.49E-02</td><td>9.50E-04</td></tr><tr><td>FD</td><td>3.04E-02</td><td>7.92E-04</td><td>5.90E-02</td><td>1.47E-04</td><td>4.75E-02</td><td>9.07E-04</td></tr><tr><td>AED</td><td>1.40E-02</td><td>3.22E-04</td><td>2.71E-02</td><td>1.06E-04</td><td>2.17E-02</td><td>4.06E-04</td></tr><tr><td colspan="8"></td></tr><tr><td rowspan="9"></td><td></td><td colspan="2">PatchTST Mean</td><td colspan="2">DLinear Mean</td><td colspan="2">FusionRNN Mean</td></tr><tr><td>MSE</td><td>8.32E-04</td><td>Std.</td><td></td><td>Std.</td><td></td><td>Std.</td></tr><tr><td>MAE</td><td>1.72E-02</td><td>5.34E-06 5.52E-05</td><td>1.16E-03 2.20E-02</td><td>1.65E-07 6.66E-05</td><td>1.02E-05 2.58E-03</td><td>9.76E-07 1.19E-04</td></tr><tr><td>SMAPE</td><td>5.46E-02</td><td>1.96E-04</td><td>6.77E-02</td><td>2.64E-04</td><td>7.49E-03</td><td>3.41E-04</td></tr><tr><td>FDE</td><td>3.29E-02</td><td>1.03E-04</td><td>4.15E-02</td><td>1.23E-05</td><td>3.14E-03</td><td>1.55E-04</td></tr><tr><td>FD</td><td>3.41E-02</td><td>1.45E-04</td><td>4.23E-02</td><td>1.23E-05</td><td>5.39E-03</td><td>1.76E-04</td></tr><tr><td>AED</td><td>1.39E-02</td><td>4.93E-05</td><td>1.75E-02</td><td>4.78E-05</td><td>2.06E-03</td><td>9.65E-05</td></tr><tr><td>MSE</td><td>3.00E-03</td><td>6.52E-05</td><td>3.95E-03</td><td>1.30E-04</td><td>1.90E-03</td><td>1.38E-03</td></tr><tr><td rowspan="7"></td><td>MAE</td><td>3.09E-02</td><td>6.84E-04</td><td>3.53E-02</td><td>1.25E-03</td><td>2.68E-02</td><td>1.10E-02</td></tr><tr><td>SMAPE</td><td>1.58E-03</td><td>8.96E-05</td><td>3.20E-03</td><td>8.63E-04</td><td>9.18E-03</td><td>5.80E-03</td></tr><tr><td>FDE</td><td>6.55E-02</td><td>1.31E-03</td><td>7.28E-02</td><td>1.94E-03</td><td>3.63E-02</td><td>9.99E-03</td></tr><tr><td>FD</td><td>6.62E-02</td><td>1.30E-03</td><td>7.48E-02</td><td>4.08E-03</td><td>4.34E-02</td><td>1.20E-02</td></tr><tr><td>AED</td><td>2.47E-02</td><td>5.57E-04</td><td>2.83E-02</td><td>1.04E-03</td><td>2.15E-02</td><td>8.82E-03</td></tr><tr><td>MSE</td><td>2.97E-03</td><td>1.75E-04</td><td>2.88E-03</td><td>2.09E-06</td><td>8.72E-04</td><td>5.39E-05</td></tr><tr><td rowspan="7">TampaB.</td><td>MAE</td><td>3.69E-02</td><td>1.85E-03</td><td>3.93E-02</td><td>2.83E-04</td><td>2.45E-02</td><td>5.61E-04</td></tr><tr><td>SMAPE</td><td>8.67E-04</td><td>4.47E-05</td><td>9.12E-04</td><td>5.69E-06</td><td>5.86E-04</td><td>1.36E-05</td></tr><tr><td>FDE</td><td>5.89E-02</td><td>9.98E-04</td><td>6.41E-02</td><td>4.39E-04</td><td>3.22E-02</td><td>8.79E-04</td></tr><tr><td>FD</td><td>6.20E-02</td><td>1.16E-03</td><td>6.64E-02</td><td>4.24E-04</td><td>3.77E-02</td><td>8.48E-04</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AED</td><td>2.93E-02</td><td>1.50E-03</td><td>3.10E-02</td><td>2.11E-04</td><td>1.92E-02</td><td>4.52E-04</td></tr></table>

Table 6: Computational complexity measured by average iteration time (in seconds) for NVIDIA 4090 GPU and GFLOPs for the VVR dataset.
<table><tr><td></td><td>Train Time</td><td>Test time</td><td>GFLOPs</td></tr><tr><td>Mix&amp;Fix-Net</td><td>0.086</td><td>0.030</td><td>1136</td></tr><tr><td>WPMixer</td><td>0.011</td><td>0.005</td><td>11</td></tr><tr><td>P_sLSTM</td><td>0.029</td><td>0.010</td><td>3</td></tr><tr><td>PatchTST</td><td>0.010</td><td>0.006</td><td>35</td></tr><tr><td>Dlinear</td><td>0.008</td><td>0.004</td><td>0.03</td></tr><tr><td>FusionRNN</td><td>0.008</td><td>0.003</td><td>14</td></tr></table>

Table 7: Relative change in six metrics with respect to the base case (Case-1). All values are reported as percentages, while the corresponding absolute values are shown in Table C.12. Symbol ♠ in the Head column refers to our proposed head layer design, while symbol ♢ refers to the regular head layer design for time series forecasting models. The "D" in the RevIN column refers only to denormalization, whereas "ND" refers to both normalization and denormalization. RTA refers to the Residual Trajectory Adjuster module.
<table><tr><td>Case</td><td>RTTA</td><td>Em x</td><td>X </td><td>Head</td><td>Pading</td><td>RIN</td><td>AMSE</td><td>AAE</td><td>ASMAPE</td><td>ADE</td><td>AFD</td><td>AED</td></tr><tr><td>1</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td>D</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>2</td><td>X</td><td>√</td><td>√</td><td></td><td>√</td><td>D</td><td>14</td><td>5</td><td>4</td><td>7</td><td>2</td><td>5</td></tr><tr><td>3</td><td>√</td><td>√</td><td>X</td><td></td><td>√</td><td>D</td><td>157906</td><td>4562</td><td>3932</td><td>5312</td><td>2365</td><td>4565</td></tr><tr><td>4</td><td>√</td><td>X</td><td>√</td><td>4</td><td>√</td><td>D</td><td>322</td><td>131</td><td>140</td><td>163</td><td>87</td><td>131</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td>ND</td><td>231</td><td>67</td><td>66</td><td>98</td><td>44</td><td>68</td></tr><tr><td>6</td><td>√</td><td>√</td><td>√</td><td>P</td><td>√</td><td>=</td><td>39</td><td>23</td><td>26</td><td>27</td><td>14</td><td>22</td></tr><tr><td>7</td><td>√</td><td>√</td><td>√</td><td>4</td><td>X</td><td>D</td><td>208</td><td>99</td><td>98</td><td>128</td><td>63</td><td>101</td></tr><tr><td>8</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td>D</td><td>495</td><td>180</td><td>193</td><td>235</td><td>114</td><td>182</td></tr></table>

In Case-7, we remove the padding step. The results show a clear drop in performance, confirming the importance of padding in our architecture.

Finally, the head layer of Mix&Fix-Net difers from that of other time series forecasting models such as WPMixer, P\_sLSTM, and PatchTST, which all share a common head design. To evaluate this diference, we replace our head with WPMixer’s head in Case-8. The results also show performance degradation. In WPMixer, the head transformation is defined as $\mathbb { R } ^ { \cdots \times ( P \cdot d ) } \xrightarrow { L i n e a r } \mathbb { R } ^ { \cdots \times T }$ , where P and T are the number of patches and predicted sequences, and d is the embedding dimension. This transformation is applied per channel, making it suitable for standard time series prediction where each channel is predicted separately. However, in the trajectory prediction task, the prediction of x-coordinate channel is not performed solely, because the value of the x-coordinate depends on the value of the y-coordinate. The details of our head layer are described in Section 3.1.4.

From Table 7, it is evident that the most critical component of our model is the Temporal Mixer module, as shown in Case-3. The Temporal Mixer captures temporal dependencies among data points. Without it, the model cannot learn the trajectory pattern, leading to a dramatic loss in performance.

## 6. Diference with the Existing Models

Although vessel trajectory prediction can be formulated as a time series forecasting problem, our proposed Mix&Fix-Net departs from conventional forecasting architectures in several key ways. These diferences are necessary to handle the trajectory prediction task, as validated in our ablation study Section 5.1.

In the traditional Head layer (♢), all future data points are projected channel-wise. On the contrary, our Head layer (♠) projects both positional channels for each future data point at the same time.

Additionally, padding the input has not been explored before for the time series forecasting task. In the ablation study, we show the efectiveness of padding in the trajectory prediction.

Employing both reversible instance normalization and denormalization is a well-established approach in time series forecasting. However, in our model, we only employ denormalization. In Table 7, we demonstrate that employing RevIN-normalization degrades the trajectory prediction performance.

Moreover, we employ a dual-stage mixer architecture, where the initial mixer predicts the coarse prediction, and the following mixer refines the earlier coarse prediction. This approach is also novel and efective in terms of the refinement stage.

## 7. Limitations and Observations

Despite the predictive accuracy of Mix&Fix-Net, certain limitations regarding video-based data must be acknowledged. Our VVR dataset was recorded from the broadcast webcam stream on the Internet, resulting in relatively low-resolution footage. Additionally, the dataset’s temporal coverage is limited. Moreover, the trajectories extracted from videos exhibit less smoothness than AIS data due to pixel-level tracking noise and perspective distortion inherent in fixed-camera observations. Consequently, these predicted trajectories may not always strictly conform to idealized physical vessel kinematics. Future research will explore integrating temporal smoothing techniques, such as smoothness regularization, with homography-based coordinate mapping to better align image-plane observations with realistic geographical motion patterns.

The findings in our work have significant practical implications for governmental authorities worldwide. National coast guards and navies can benefit from the proposed framework through enhanced surveillance and monitoring capabilities for small vessels operating without AIS. Furthermore, vision-based trajectory prediction can support environmental monitoring activities conducted by environmental and oceanographic institutions. The model can also contribute to trafic optimization and digital maritime infrastructure initiatives led by maritime administrations through improved predictive analytics for vessel movement patterns.

## 8. Conclusion

Vessel trajectory prediction is a complex task as the trajectory of the vessel is influenced by various factors such as ocean currents, weather conditions, and navigational constraints at sea. The widespread availability of AIS data has enabled substantial research in the trajectory prediction domain. However, not all vessels are equipped with AIS, and obstacles along the trajectory, such as restricted zones or dynamic hazards, cannot be identified solely through AIS data. To address these challenges, we propose Mix&Fix-Net, a novel dual-stage MLP-mixer-based model that is applicable to both AIS and non-AIS data. Additionally, we introduce VVR, a video-based dataset developed specifically for vessel trajectory prediction. Our model is evaluated on both the non-AIS dataset (VVR) and the AIS datasets (M3, TampaBay) using six metrics: MSE,

MAE, SMAPE, FDE, FD, and AED, under two diferent data-splitting settings. Experimental results demonstrate that Mix&Fix-Net consistently outperforms baseline methods across most metrics and datasets. Through a detailed ablation study, we further examine the contribution of individual modules within our model and highlight its advantages over conventional time series forecasting-based models. These findings underscore the potential of Mix&Fix-Net as a robust and efective framework for advancing vessel trajectory prediction.

## Acknowledgments

The authors would like to thank Awwab Ahmed, Nguyen Hong, and Emily Wyatt for their eforts in annotating vessels for the VVR dataset.

This work was supported by the U.S. National Institute of Food and Agriculture under Grant 2023-67019-38829. Additionally, Bora San Turgut was supported by the Scientific and Technological Research Council of Türkiye (TUBITAK) during this work.

## References

[1] R. E. Schnurr, T. R. Walker, Marine transportation and energy use, Reference Module in Earth Systems and Environmental Sciences (2019) 1–9.

[2] Z. Zhao, X. Liu, L. Feng, M. Grifoll, H. Feng, Causation Analysis of Marine Trafic Accidents Using Deep Learning Approaches: A Case Study from China’s Coasts, Systems 13 (4) (2025) 284.

[3] M. Hassel, B. E. Asbjørnslett, L. P. Hole, Underreporting of maritime accidents to vessel accident databases, Accident Analysis & Prevention 43 (6) (2011) 2053–2063.

[4] H. Wang, Z. Liu, X. Wang, T. Graham, J. Wang, An analysis of factors afecting the severity of marine accidents, Reliability Engineering & System Safety 210 (2021) 107513.

[5] H. Li, X. Ren, Z. Yang, Data-driven bayesian network for risk analysis of global maritime accidents, Reliability Engineering & System Safety 230 (2023) 108938.

[6] S. Liao, J. Weng, Z. Zhang, Z. Li, F. Li, Probabilistic modeling of maritime accident scenarios leveraging bayesian network techniques, Journal of Marine Science and Engineering 11 (8) (2023) 1513.

[7] C.-H. Yang, C.-H. Wu, J.-C. Shao, Y.-C. Wang, C.-M. Hsieh, AIS-based intelligent vessel trajectory prediction using bi-LSTM, IEEE Access 10 (2022) 24302–24315.

[8] International Maritime Organization, International Convention for the Safety of Life At Sea (SOLAS), Consolidated Edition, Chapter V, Regulation 19 (2024).

[9] W. Luo, Y. Xia, T. He, Video-Based Identification and Prediction Techniques for Stable Vessel Trajectories in Bridge Areas, Sensors 24 (2) (2024) 372.

[10] P. Dijt, P. Mettes, Trajectory prediction network for future anticipation of ships, in: Proceedings of the 2020 international conference on multimedia retrieval, 2020, pp. 73–81.

[11] J. Lei, Y. Sun, Y. Wu, F. Zheng, W. He, X. Liu, Association of AIS and radar data in intelligent navigation in inland waterways based on trajectory characteristics, Journal of Marine Science and Engineering 12 (6) (2024) 890.

[12] W. He, W. He, J. Lei, S. Wan, Z. Wang, Multi-source perception data fusion of vessels in visual occlusion scenarios: Leveraging prior knowledge of vessel motion, Engineering Applications of Artificial Intelligence 156 (2025) 111118.

[13] E. Gülsoylu, P. Koch, M. Yildiz, M. Constapel, A. P. Kelm, Image and AIS Data Fusion Technique for Maritime Computer Vision Applications, in: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops, 2024, pp. 859–868.

[14] P. Huang, Q. Chen, D. Wang, M. Wang, X. Wu, X. Huang, Tripleconvtransformer: A deep learning vessel trajectory prediction method fusing discretized meteorological data, Frontiers in Environmental Science 10 (2022) 1012547.

[15] S. Lin, Y. Jiang, F. Hong, L. Xu, H. Huang, B. Wang, Hdformer: A transformer-based model for fishing vessel trajectory prediction via multi-source data fusion, Ocean Engineering 320 (2025) 120309.

[16] A. Zeng, M. Chen, L. Zhang, Q. Xu, Are transformers efective for time series forecasting?, in: Proceedings of the AAAI conference on Artificial Intelligence, Vol. 37, 2023, pp. 11121–11128.

[17] S.-A. Chen, C.-L. Li, N. Yoder, S. O. Arik, T. Pfister, TSMixer: An All-MLP Architecture for Time Series Forecasting, Transactions on Machine Learning Research (2023).

[18] M. M. N. Murad, M. Aktukmak, Y. Yilmaz, Wpmixer: Eficient multi-resolution mixing for long-term time series forecasting, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39, 2025, pp. 19581–19588.

[19] L. Wang, J. Zhang, G. Jin, X. Dong, STID-Mixer: A Lightweight Spatio-Temporal Modeling Framework for AIS-Based Vessel Trajectory Prediction, Eng 6 (8) (2025) 184.

[20] K. Takahashi, K. Zama, N. F. Hiroi, Ship trajectory prediction using AIS data with TransFormer-based AI, in: 2024 IEEE Conference on Artificial Intelligence (CAI), IEEE, 2024, pp. 1302–1305.

[21] W. Wang, W. Xiong, X. Ouyang, L. Chen, Tptrans: Vessel trajectory prediction model based on transformer using AIS data, ISPRS International Journal of Geo-Information 13 (11) (2024) 400.

[22] T. A. Volkova, Y. E. Balykina, A. Bespalov, Predicting ship trajectory based on neural networks using AIS data, Journal of Marine Science and Engineering 9 (3) (2021) 254.

[23] L. P. Perera, C. G. Soares, et al., Ocean vessel trajectory estimation and prediction based on extended Kalman filter, in: The Second International Conference on Adaptive and Self-Adaptive Systems and Applications, Citeseer, 2010, pp. 14–20.

[24] L. Sigillo, A. Marzilli, D. Moretti, E. Grassucci, C. Greco, D. Comminiello, Sailing the Seaformer: A Transformer-Based Model for Vessel Route Forecasting, in: 2023 IEEE 33rd International Workshop on Machine Learning for Signal Processing (MLSP), IEEE, 2023, pp. 1–6.

[25] G. Spadon, J. Kumar, D. Eden, J. van Berkel, T. Foster, A. Soares, R. Fablet, S. Matwin, R. Pelot, Multi-path long-term vessel trajectories forecasting with probabilistic feature fusion for problem shifting, Ocean Engineering 312 (2024) 119138.

[26] Q. Chen, C. Xiao, Y. Wen, M. Tao, W. Zhan, Ship intention prediction at intersections based on vision and bayesian framework, Journal of Marine Science and Engineering 10 (5) (2022) 639.

[27] S. Gu, X. Zhang, J. Zhang, A full-time deep learning-based alert approach for bridge–ship collision using visible spectrum and thermal infrared cameras, Measurement Science and Technology 34 (9) (2023) 095907.

[28] Y. Chen, X. Qi, C. Huang, J. Zheng, A data fusion method for maritime trafic surveillance: The fusion of AIS data and VHF speech information, Ocean Engineering 311 (2024) 118953.

[29] L. Bai, X. Zhang, R. Yao, Y. Lin, Q. Mei, Prediction of ship berthing behavior intentions using AIS integration and maritime VHF voice feature selection and recognition, Ships and Ofshore Structures (2025) 1–19.

[30] T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi, J. Choo, Reversible instance normalization for accurate time-series forecasting against distribution shift, in: International Conference on Learning Representations, 2021.

[31] I. O. Tolstikhin, N. Houlsby, A. Kolesnikov, L. Beyer, X. Zhai, T. Unterthiner, J. Yung, A. Steiner, D. Keysers, J. Uszkoreit, et al., Mlp-mixer: An all-mlp architecture for vision, Advances in neural information processing systems 34 (2021) 24261–24272.

[32] Center for Accelerating Operational Eficiency (CAOE), Managing Maritime Movements (M3) Hackathon, https://caoe.asu.edu/2024/07/25/ managing-maritime-movements-m3-hackathon/, accessed: 2025-08-30 (July 2024).

[33] Bureau of Ocean Energy Management (BOEM) and National Oceanic and Atmospheric Administration (NOAA), Ship trajectory data set, https://hub.marinecadastre.gov/ pages/vesseltraffic, accessed: 2026-02-18 (2024).

[34] Webcamtaxi, Canal do Porto de Santos, live webcam stream, São Paulo, Brazil (2025). URL https://www.webcamtaxi.com/en/brazil/sao-paulo-state/ canal-do-porto-de-santos-cam.html

[35] F. Yang, C. He, Y. Liu, A. Zeng, L. Hu, Vessel trajectory prediction based on automatic identification system data: multi-gated attention encoder decoder network, Journal of Marine Science and Engineering 12 (10) (2024) 1695.

[36] Y. Li, Q. Yu, Z. Yang, Vessel trajectory prediction for enhanced maritime navigation safety: A novel hybrid methodology, Journal of Marine Science and Engineering 12 (8) (2024) 1351.

[37] N. Evmides, M. P. Michaelides, H. Herodotou, Vessel trajectory prediction with deep learning: Temporal modeling and operational implications, Journal of Marine Science and Engineering 13 (8) (2025) 1439.

[38] I. Slaughter, J. L. Charla, M. Siderius, J. Lipor, Vessel trajectory prediction with recurrent neural networks: An evaluation of datasets, features, and architectures, Journal of Ocean Engineering and Science 10 (2) (2025) 229–238.

[39] Z. Guo, H. Qiang, X. Peng, Vessel trajectory prediction using vessel influence long shortterm memory with uncertainty estimation, Journal of Marine Science and Engineering 13 (2) (2025) 353.

[40] H. Li, H. Jiao, Z. Yang, AIS data-driven ship trajectory prediction modelling and analysis based on machine learning and deep learning methods, Transportation Research Part E: Logistics and Transportation Review 175 (2023) 103152.

[41] Y. Kong, Z. Wang, Y. Nie, T. Zhou, S. Zohren, Y. Liang, P. Sun, Q. Wen, Unlocking the power of LSTM for long term time series forecasting, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39, 2025, pp. 11968–11976.

[42] Y. Nie, N. H. Nguyen, P. Sinthong, J. Kalagnanam, A Time Series is Worth 64 Words: Long-term Forecasting with Transformers, in: International Conference on Learning Representations, 2023.

[43] Ultralytics, Multi-Object Tracking with Ultralytics YOLO, https://docs. ultralytics.com/modes/track/#tracker-arguments, accessed: 2026-03-08 (2026).

## Appendix A. Vessel Detection and Tracking Implementation Details

Our Video-based Vessel Route (VVR) dataset consists of six video recordings, detailed in Table 1. The first two videos are used to train and evaluate the vessel detection model, while the remaining four videos are used to generate the vessel’s time-series trajectories. Although the pretrained YOLOv11 model (YOLOv11x variant with pretrained weights) can detect ships or vessels, we observe frequent missed detections for small vessels in our recordings. Therefore, we fine-tune the pretrained model on our domain-specific data.

YOLO Fine-Tuning: The first two videos are converted into image frames. All vessels appearing in these frames were manually annotated with bounding boxes. The total number of image frames is split into a train and a test set with a 7:3 ratio. YOLO is then fine-tuned and evaluated on the train and test sets, respectively. To fine-tune it, we employ an initial learning rate of 0.0002, a batch size of 64, and 100 epochs. Additionally, we employ basic augmentation (e.g., scaling, flipping, color shifting) during training; augmentation parameters are shown in Table A.8.

Multi-Object Tracking: To track vessels, we use the ByteTrack algorithm with the finetuned YOLO model. The experimental settings of the tracker are shown in Table A.9. The tracker assigns a unique vessel ID to each detected object across frames. The center coordinates of the bounding boxes are extracted to generate the vessel trajectory time-series data.

Table A.8: Data augmentation parameters used for YOLOv11 fine-tuning on the VVR dataset.
<table><tr><td>Parameter</td><td>Description</td><td>Value</td></tr><tr><td>hsv_h</td><td>Image HSV-Hue augmentation (fraction)</td><td>0.015</td></tr><tr><td>hsv_s</td><td>Image HSV-Saturation augmentation (fraction)</td><td>0.7</td></tr><tr><td>hsv_v</td><td>Image HSV-Value augmentation (fraction)</td><td>0.4</td></tr><tr><td>translate</td><td>Image translation (+/- fraction)</td><td>0.1</td></tr><tr><td>scale</td><td>Image scale (+/- gain)</td><td>0.5</td></tr><tr><td>fliplr</td><td>Image flip left-right (probability)</td><td>0.5</td></tr><tr><td>degrees</td><td>Image rotation (+/- deg)</td><td>0.0</td></tr><tr><td>shear</td><td>Image shear (+/- deg)</td><td>0.0</td></tr><tr><td>flipud</td><td>Image flip up-down (probability)</td><td>0.0</td></tr></table>

Table A.9: Experimental settings of the ByteTrack algorithm. Parameter definitions follow the standard implementation in [43].
<table><tr><td>Parameter</td><td>Description</td><td>Value</td></tr><tr><td>track_high_thresh</td><td>Threshold for the first association (high score)</td><td>0.5</td></tr><tr><td>track_low_thresh</td><td>Threshold for the second association (low score)</td><td>0.1</td></tr><tr><td>new_track_thresh</td><td>Threshold for initiating a new track</td><td>0.5</td></tr><tr><td>track_buffer</td><td>Number of frames to retain a lost track</td><td>600</td></tr><tr><td>match_thresh</td><td>Intersection over Union (IoU) matching threshold</td><td>0.95</td></tr></table>

Table A.10: Hyperparameter settings of the proposed model across benchmark datasets under both experimental settings. Here, L and T denote the lookback window length and the prediction window length, respectively. Additionally, d, t , $d _ { f } ,$ drop. refer to the embedding dimension, temporal expansion factor, embedding expansion factor, and dropout value, respectively.
<table><tr><td>Setting</td><td>Data</td><td>L</td><td>T</td><td>Initial lr.</td><td>Batch</td><td> $d$ </td><td> $t _ { f }$ </td><td> $d _ { f }$ </td><td>drop.</td><td>Epochs</td><td>Patience</td></tr><tr><td rowspan="3">1</td><td>VVR</td><td>60</td><td>120</td><td>8.13E-05</td><td>128</td><td>16</td><td>5</td><td>2</td><td>0.05</td><td>50</td><td>10</td></tr><tr><td>M3</td><td>60</td><td>120</td><td>0.004887</td><td>512</td><td>16</td><td>5</td><td>1</td><td>0.05</td><td>50</td><td>10</td></tr><tr><td>TampaBay</td><td>60</td><td>120</td><td>6.72E-05</td><td>128</td><td>16</td><td>1</td><td>3</td><td>0.05</td><td>50</td><td>10</td></tr><tr><td rowspan="3">2</td><td>VVR</td><td>60</td><td>120</td><td>0.001172</td><td>128</td><td>256</td><td>5</td><td>5</td><td>0</td><td>50</td><td>10</td></tr><tr><td>M3</td><td>60</td><td>120</td><td>0.001392</td><td>512</td><td>256</td><td>3</td><td>5</td><td>0</td><td>50</td><td>10</td></tr><tr><td>TampaBay</td><td>60</td><td>120</td><td>8.18E-05</td><td>128</td><td>128</td><td>5</td><td>2</td><td>0.1</td><td>50</td><td>10</td></tr></table>

![](images/d2afb5a5393bdc0d44694b750fbe2b27ee37e198b5cc7f563fbaa9500c3fdc0b.jpg)  
Figure A.6: Visual comparison of the predicted trajectory among diferent models for the TampaBay dataset. Trajectories are randomly selected

## Appendix B. Analysis of Prediction Smoothness

As observed in Figure 5 and Figure A.6, the trajectories predicted by Mix&Fix-Net exhibit slightly more local irregularity than those of some baselines. This behavior arises from two distinct factors, one data-related and one architectural:

First, the VVR dataset is inherently noisy due to abrupt bounding-box variations from the detector and tracker, and since the training data itself is noisy, this noise propagates into the predictions of all models.

Second, even on the cleaner TampaBay and M3 datasets, our predictions show slight irregularity, and this is due to architectural design. The head layer of our model predicts the position at each future timestep individually from a dedicated hidden representation:

$$
\mathbb { R } ^ { 1 \times ( C _ { i } \cdot d ) } \xrightarrow { \mathrm { L i n e a r } } \mathbb { R } ^ { 1 \times C _ { i } } .\tag{B.1}
$$

Here, $\mathbb { R } ^ { 1 \times ( C _ { i } \cdot d ) }$ represents the hidden feature vectors for time step t, and $\mathbb { R } ^ { 1 \times C _ { i } }$ represents the corresponding output transformation at time step t. From the $C _ { i }$ features, we only consider the last two features $( C _ { o } = 2 )$ as our final positional coordinates $( \mathrm { x } / \mathrm { y }$ or lat/lon.). Conversely, the baselines derive all T predicted positions from a single shared hidden representation. For example, the head layer of WPMixer can be represented as,

$$
\mathbb { R } ^ { \cdots \times ( P \cdot d ) } \xrightarrow { \mathrm { L i n e a r } } \mathbb { R } ^ { \cdots \times T }\tag{B.2}
$$

where P and d refer to the patch number and embedding dimension, respectively. Here, the same hidden representation R<sup>···×(P·d)</sup> is shared across all T predicted time steps of a given channel, and a single linear projection produces the full output sequence at once. Since this shared representation implicitly couples adjacent outputs, this yields globally smooth curves. But our per-timestep design does not impose this coupling.

The benefit of our design, however, is significant: because each predicted point has its own hidden representation, our predicted trajectories follow the ground-truth curvature (especially along sharp turns) far more accurately than the baselines, as visible in Figure 5 and Figure A.6. This benefit is also confirmed quantitatively in Table 7 (Case-1 vs. Case-8), where replacing our head with the conventional head (time-series model) increases MSE, MAE, SMAPE, FDE, FD, and AED by 495%, 180%, 193%, 235%, 114%, and 182%, respectively. Thus, although the predicted trajectory appears slightly noisier, it tracks the true trajectory much more accurately, which is the primary objective of trajectory prediction.

## Appendix C. Additional Ablation Studies

This section presents the impact on non-positional features and hyperparameter sensitivity analysis. For all experiments, dataset split setting-2 is employed, and the hyperparameter is tuned using Optuna.

## Appendix C.1. Impact ofnon-positionalfeatures

The VVR dataset comprises nine features: $M _ { s i n }$ $M _ { c o s }$ $H _ { s i n } .$ $H _ { c o s }$ $D _ { s i n }$ $D _ { c o s }$ $D _ { d i s p }$ , xcoordinate, and y-coordinate. In contrast, the M3 dataset includes an additional feature, Profile-ID. To assess the influence of non-positional features, we evaluated the model using three configurations: positional features only $( C _ { i } = 2 )$ , the feature set excluding day-of-week encodings $( C _ { i } = 7 )$ , and the full feature set $( C _ { i } = 9 )$ . The results, presented in Table C.11, show that while positional data provides a baseline, the inclusion of auxiliary features significantly reduces error. Notably, the transition from $C _ { i } = 7$ to $C _ { i } = 9$ demonstrates that even with limited temporal coverage, the day-of-week features provide beneficial high-dimensional context for the model. This benefit does not rely only on capturing the periodic behavioral patterns; rather, these sinusoidal encodings function as expressive temporal positional encodings, providing a unique and structured temporal stamp for each data point.

Table C.11: Impact of the non-positional features. C denotes the number of input features: $C _ { i } = 2$ (positional only), $C _ { i } = 7$ (all features excluding $D _ { s i n } / D _ { c o s } ) ,$ , and $C _ { i } = 9$ (full feature set).
<table><tr><td></td><td> $C _ { i }$ </td><td> $2$ </td><td>7</td><td>9</td></tr><tr><td rowspan="5">VVR</td><td>MSE</td><td>2.16E-05</td><td>4.63E-06</td><td>3.86E-06</td></tr><tr><td>MAE</td><td>3.50E-03</td><td>1.49E-03</td><td>1.45E-03</td></tr><tr><td>SMAPE</td><td>1.02E-02</td><td>4.18E-03</td><td>4.08E-03</td></tr><tr><td>FDE</td><td>4.92E-03</td><td>1.70E-03</td><td>1.64E-03</td></tr><tr><td>FD</td><td>7.08E-03</td><td>3.79E-03</td><td>3.72E-03</td></tr><tr><td></td><td>AED</td><td>2.82E-03</td><td>1.19E-03</td><td>1.16E-03</td></tr></table>

## Appendix C.2. Sensitivity Analysis and Temporal Scale Justification

In our model, one of the critical hyperparameters is the embedding dimension d. To assess its sensitivity, we evaluate our model using $d \in \{ 1 6 , 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$ . As shown in Figure C.7, performance metrics generally improve as d increases. This occurs because the model captures more fine-grained information with higher embedding dimensions. While performance improves gradually with larger d, this comes at the cost of significantly higher GFLOPs. Considering this trade-of, we restrict d to values not exceeding 256.

Additionally, Figure C.8 shows the performance variation with observation window $L \in$ {40<sub>,</sub> 60<sub>,</sub> 80<sub>,</sub> 100}. It is observed that model’s prediction performance is improved with increasing L. However, with the increasing L, the minimum required trajectory length $( L + T )$ is also increased. Additionally, the number of trajectories with smaller length is considerable in the histogram plot, shown in Figure C.9. Considering this trade-of, and following the criteria of the M3 data challenge, we set the observation and prediction lengths to 60 and 120, respectively.

Moreover, the same window length $( L = 6 0 , T = 1 2 0 )$ corresponds to diferent physical time spans (approximately 3 hours for AIS vs. 60 seconds for video(180 points)). However, our model is designed to be scale-agnostic by focusing on the relative temporal dynamics within a sequence. Since the model is trained and evaluated independently for each modality, it learns the “behavioral physics” appropriate to each domain, such as long-term navigational trends in AIS and short-term kinematic maneuvers in video. Additionally, the coverage area within an image is very small compared to that of the AIS data. Moreover, the vessel’s trajectory across the river is very small compared to its movement along the river. Therefore, it is essential to use a suficiently small sampling period to capture suficient motion in the VVR data while rejecting noise arising from the high-frequency raw sampling rate (30 FPS).

## Appendix D. Semantic Consistency of Coordinate Systems

A key challenge in multi-modal trajectory prediction is the semantic diference between coordinate systems. In this study, AIS data utilizes geographic spherical coordinates (Latitude/Longitude), while the VVR dataset uses normalized image-plane coordinates (xy). We justify the unified treatment of these data types based on three factors. First, the VVR camera remains stationary with a fixed orientation, ensuring that image-plane coordinates provide a stable and consistent spatial reference frame throughout the dataset. Second, the use of z-normalization (as detailed in Section 4.2) maps both coordinate systems to a non-dimensional scale, allowing the Mix&Fix-Net to focus on relative motion dynamics rather than absolute spatial metrics. Finally, the model is trained and evaluated independently on AIS and non-AIS datasets. This allows the model to adapt to diferent multi-modal datasets. The high performance across both modalities confirms that the architecture is robust and coordinate-agnostic.

![](images/60a9a8ac2e1b85eaf0e80f3186ce22253bd418528da95018773097e6227556fc.jpg)  
Figure C.7: Impact of embedding dimension d on prediction performance and computational cost for the VVR dataset.

![](images/d9d4761d1616ef5fb3e21ed6ea84a07a2bfc0bdec2673411e78292053ce35577.jpg)  
Figure C.8: Impact of observation window L on prediction performance for the VVR dataset.

![](images/9e7772dc9b4ca728db8173176ffaf45984306bafe3ee0dbbf9f780162d87bbe5.jpg)  
(a) VVR

![](images/f922b266dcab66edd722e7a26d35e3851fe78cd5d14445685811f44e4b8dd890.jpg)  
(b) TampaBay  
Figure C.9: Histogram of the Vessel trajectory length.

Table C.12: Absolute values of six metrics for each ablation case, corresponding to the relative percentage changes reported in Table 7. Symbol ♠ in the Head column refers to our proposed head layer design, while symbol ♢ refers to the regular head layer design for time series forecasting models. The "D" in the RevIN column refers only to denormalization, whereas "ND" refers to both normalization and denormalization. RTA refers to the Residual Trajectory Adjuster module.
<table><tr><td>Case</td><td>RTA</td><td>Embx</td><td>Tm&#x27;x</td><td>Head</td><td>Padding</td><td>RIN</td><td>MSE</td><td>MAE</td><td>SMAPE</td><td>FDE</td><td></td><td>AED</td></tr><tr><td>1</td><td>√</td><td>√</td><td>√</td><td>à</td><td>√</td><td>D</td><td>3.86E-06</td><td>1.45E-03</td><td>4.08E-03</td><td>1.64E-03</td><td>3.72E-03</td><td>1.16E-03</td></tr><tr><td>2</td><td>×</td><td>√</td><td>√</td><td></td><td>√</td><td>D</td><td>4.41E-06</td><td>1.52E-03</td><td>4.26E-03</td><td>1.75E-03</td><td>3.79E-03</td><td>1.22E-03</td></tr><tr><td>3</td><td>√</td><td>√</td><td>X</td><td></td><td>√</td><td>D</td><td>6.10E-03</td><td>6.77E-02</td><td>1.64E-01</td><td>8.88E-02</td><td>9.17E-02</td><td>5.40E-02</td></tr><tr><td>4</td><td>√</td><td>X</td><td>√</td><td></td><td>√</td><td>D</td><td>1.63E-05</td><td>3.35E-03</td><td>9.79E-03</td><td>4.31E-03</td><td>6.96E-03</td><td>2.67E-03</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td>ND</td><td>1.28E-05</td><td>2.42E-03</td><td>6.79E-03</td><td>3.25E-03</td><td>5.35E-03</td><td>1.95E-03</td></tr><tr><td>6</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td>-</td><td>5.37E-06</td><td>1.78E-03</td><td>5.13E-03</td><td>2.08E-03</td><td>4.24E-03</td><td>1.42E-03</td></tr><tr><td>7</td><td>√</td><td>√</td><td>√</td><td></td><td>X</td><td>D</td><td>1.19E-05</td><td>2.89E-03</td><td>8.09E-03</td><td>3.74E-03</td><td>6.07E-03</td><td>2.32E-03</td></tr><tr><td>8</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td>D</td><td>2.30E-05</td><td>4.06E-03</td><td>1.19E-02</td><td>5.50E-03</td><td>7.97E-03</td><td>3.26E-03</td></tr></table>