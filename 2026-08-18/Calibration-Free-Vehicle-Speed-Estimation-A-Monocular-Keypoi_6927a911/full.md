# Calibration-Free Vehicle Speed Estimation: A Monocular Keypoint-Template Approach

Gaofeng Su   
Department of Civil & Environmental Engineering   
The University of California, Berkeley   
Email: koufongso@berkeley.edu   
Keya Li   
Department of Civil, Architectural and Environmental Engineering   
The University of Texas, Austin   
Email: keya\_li@utexas.edu

Raja Sengupta, Ph.D. Department of Civil & Environmental Engineering The University of California, Berkeley Email: rajasengupta@berkeley.edu

Kara M. Kockelman, Ph.D., P.E<sup>\*</sup>   
Department of Civil, Architectural and Environmental Engineering   
The University of Texas, Austin   
512-471-0210   
Email: kkockelm@mail.utexas.edu

<sup>\*</sup>Corresponding author

## ABSTRACT

This paper proposes a calibration-free framework for reliably and efectively estimating vehicle speeds from monocular videos, without relying on roadway features, camera calibration, or roadway-feature-based reference objects. The proposed framework estimates vehicle speeds using a 36-keypoint vehicle template and a homography matrix updated at each frame. A YOLO-based keypoint detection module is trained on diverse datasets, and two estimation strategies are compared: keypoint-only tracking and warped optical flow with dense spatial aggregation. Speed is referred by projecting displacements into metric space using the homography, with validation conducted on over 400 video clips from road-side and over-head datasets, covering speeds from 30 to 100 mph. The method achieves reliable speed estimation on the V13 and BrnoCompSpeed datasets, with the warped optical flow method delivering MAEs of 15.0% and 9.7%, respectively, and 77.9% and 93.1% of estimates within ±20% error. After applying a 10% trim to remove edge-of-frame outliers, performance improves to 11.7% and 7.6% MAE, with within-±20% accuracy rising to 85.3% and 95.4%. This work addresses key limitations of existing vision-based approaches and enables low-cost and eficient speed enforcement using portable devices such as dashcams and smartphones, thereby supporting citizen-based enforcement programs for trafic safety.

## INTRODUCTION

Estimating vehicle speed accurately is a fundamental topic in Intelligent Transportation Systems (ITS) and plays a critical role in trafic enforcement and improving road safety. Active and automated enforcement of speed limits can lower speeds, crash counts, and injuries (Sadeghi-Bazargani and Saadati 2016; Shin et al. 2009). For example, fixed speed cameras can reduce 47% of fatal and injury crashes on urban principal arterials (Shin et al. 2009).

Common speed measurement methods include intrusive technologies (e.g., inductive loop detectors) and non-intrusive technologies (e.g., overhead-mounted and side-fired sensors) (Sangsuwan and Ekpanyapong 2024; Odat et al. 2017; Martin et al. 2003). Intrusive detectors are installed within or across the roadway; for example, inductive loop detectors estimate vehicle speeds by detecting vehicle presence within the loop area through changes in oscillator frequency (Martin et al. 2003). Such technologies are expensive to install and can cause disruptions to trafic flow. For instance, a loop system with 40-cm ducts and three chambers costs approximately \$9,000 per approach (Webb 2020). Non-intrusive technologies include active detectors (such as active infrared, microwave radar, and ultrasonic sensors) and passive detectors (such as passive infrared, acoustic, and video image processing systems). Active detectors transmit energy (e.g., radar waves or LiDAR) toward vehicles and calculate speeds based on the returned signal, while passive detectors measure energy emitted by vehicles (Javadi et al. 2019). Active sensors generally provide longer detection ranges and greater robustness, but require external power sources and are more complex. In contrast, passive sensors are simpler and require lower power consumption but are more susceptible to environmental conditions (Hovhannisyan 2021).

Recently, vision-based speed estimation methods have emerged as a more cost-efective and scalable alternative (Martínez et al. 2024). Over the past decade, computer vision (CV) technologies have been widely used for vehicle detection and tracking (Zhang et al. 2022; Zuraimi and Zaman 2021; Maity et al. 2021), license plate recognition (Xu et al. 2018; Montazzolli and Jung 2017; Parsa et al. 2025), and speed estimation (Hua et al. 2018; Liu et al. 2020). Most state-of-the-art CV-based speed estimation methods typically rely on camera videos and scaling factors derived from road features to estimate travel distances in a two-dimensional (2D) domain using vanishing points (VP) and bounding boxes (Lu et al. 2017; Kim et al. 2018; Chang et al. 2018; Li et al. 2025). Despite their efectiveness, these approaches often require prior knowledge or strong assumptions about roadway geometry, fixed roadway features, or camera calibration, which limits their scalability and applicability.

This work introduces a novel and calibration-free framework for vehicle speed estimation that does not rely on roadway features. The proposed approach iteratively updates the homography matrix derived from at least four non-collinear points in the planar video frame, to estimate vehicle speeds and can be readily applied to both overhead- and side-view camera videos. This framework enables speed estimation under flexible deployment conditions and is adaptable to a wide range of roadway environments and camera configurations.

The remainder of this paper is organized as follows. Section II reviews related work on CV-based speed estimation. Section III describes the proposed methodology for vehicle speed estimation. Section IV presents the experimental results on overhead and side-view public datasets. Finally, Section V concludes the paper and discusses future work.

## RELATED WORK

CV techniques to estimate vehicle speeds typically begin with video recordings and scene parameters (such as image-to-world scaling factors) as inputs, then use detection and tracking algorithms to estimate traveled distances in the 2D image domain. Speeds are calculated using real-world distances derived from scale factors and the time intervals between consecutive frames (Gunawan et al. 2019; Abdel-Aziz et al. 2015). One critical component of vision-based speed estimation is camera calibration (Zhang 2021; Weng et al. 1992), which enables accurate mapping of 2D image coordinates to 3D real-world motion. Calibration involves estimating both intrinsic parameters (sensor size, resolution, focal length) and extrinsic parameters (camera position and orientation). Traditional methods rely on reference objects with known dimensions, such as lane markings or width, but these require manual measurement and are often impractical for large-scale deployment.

To address this, VP-based calibration has emerged as a widely used approach. VPs occur where parallel lines in the 3D world (e.g., lane boundaries, building edges) appear to converge in the image. VP-based methods are classified into geometry-based and learning-based methods. Geometrybased methods use line intersections, clustering techniques, or Gaussian sphere representations to estimate VPs (Feng et al. 2010; Barinova et al. 2010; Collins and Weiss 1990). For instance, Song et al. (2021) adopted an improved Direct Linear Transform (DLT) method by obtaining two VPs from vehicle trajectory lines and car body edge lines, combined with a third VP derived from geometric relationships, to calculate speeds. Results show that the error of the proposed method under diferent conditions was less than 2.5%. Learning-based methods, instead, use deep neural networks trained on large-scale datasets with VP annotations to infer VPs directly from image context (Zhai et al. 2016; Chang et al. 2018). Hua et al. (2018) combined deep learning-based object detection with optical flow tracking on the NVIDIA AI City Challenge dataset, which contains over 20 hours of trafic videos from multiple camera angles. Their best results achieved a detection rate score of 1.0 and a root mean square error (RMSE) of 12.109 mph. They also pointed out that performance was particularly poor in locations where camera movement due to wind or bridge vibrations degraded video quality. Rani et al. (2024) proposed a deep learning-based framework combining U-Net segmentation with an LV-YOLO network to detect logistic vehicles and estimated their speeds from highway CCTV footage. Results indicate that the method achieved a 2.63% improvement in speed prediction accuracy over the 1D-CNN speed estimation model.

Based on estimated VPs and standard camera placement assumptions (zero skew, overhead), intrinsic and extrinsic parameters can be derived, enabling transformation between camera and real-world coordinates. Notably, most existing methods are applied to fixed trafic cameras and standard roadway features, which makes speed estimation particularly challenging in environments where roadway features is unclear or unavailable.

## PROPOSED METHODOLOGY

This section presents the methodology for estimating vehicle speeds from monocular video, based on semantic keypoints, optical flow, and adaptive homography-based mapping. The following subsections outline the problem statement, describe the dynamic homography estimation procedure, derive the speed estimation formulation, analyze errors, and present the complete system pipeline.

## Problem Statement

Given two consecutive images $I _ { k }$ and $I _ { k + 1 }$ capturing � moving vehicles, the objective is to estimate the set of linear speeds $\mathcal { V } _ { k } ~ = ~ \{ \nu _ { i , k } \} _ { i = 1 } ^ { N }$ , where $\nu _ { i , k }$ denotes the instantaneous speed for vehicle instance � at time step �. A fundamental challenge in vision-based speed estimation arises from perspective efects, whereby identical pixel displacements do not correspond to equal physical distances in the scene. To resolve this, we utilized homography, a projective transformation that maps points between two planar coordinate frames (Hartley and Zisserman 2003). A key advantage of this technique is that it does not require prior camera calibration, provided that at least four noncollinear correspondences between the coordinate frames are identified (Song et al. 2021).

Vehicles can be approximated as rigid bodies composed of locally planar surfaces, as illustrated in Figure 1. The red and yellow shaded regions highlight the distinct lateral and upper planar faces modeled, respectively, with an independent coordinate frame established for each plane. Furthermore, it is assumed that the vehicle undergoes longitudinal motion only, with negligible lateral displacement and rotation between consecutive frames. Under this assumption, the motion of each planar surface can be represented by a 2D homography.

![](images/6583209fc87751547875f65999e8ab590f1f9a675128dbfb1c59d1528543ac98.jpg)  
FIGURE 1: Piecewise planar approximation of a vehicle body. (Frame sampled and modified from the VS13 dataset (Djukanović et al. 2022)).

## Notation and Preliminaries

Let $i \in \{ 1 , \ldots , N \}$ denote a specific tracked vehicle instance and � denote the discrete time step. Each vehicle is modeled as a collection of planar facets $c \in C .$ , where $C = \{ \mathrm { t o p } , \mathrm { l e f t } , \mathrm { r i g h t } \}$ Although this work is targeting a multi-vehicle environment, their speeds can be estimated inde pendently through an identical process. Moreover, since the homography transformation applies independently to facets, the subsequent derivation focuses on a single lateral plane of a single vehicle without loss of generality and maintain a clean and understandable notation.

For a tracked vehicle � at time step �, we define:

1) $I _ { k }$ : The image coordinate frame at time �.

2) $V _ { k }$ : The metric planar coordinate frame attached to a vehicle facet, and is established based on the vehicle’s pose at time �.

3) $\mathbf { p } _ { k } \in I _ { k }$ : The pixel coordinates $[ u , \nu ] ^ { \top }$ of a physical point captured at time �.

4) $\mathbf { s } _ { k } ^ { V _ { k } }$ : The metric coordinates $[ x , y ] ^ { \top }$ of a point captured at time �, expressed in the metric

frame $V _ { k }$

5) $\textstyle { \mathcal { M } } _ { k }$ : The set of detected semantic keypoints belonging to a facet at time $k$

6) $\mathbf { F } _ { k }$ : The dense optical flow field, where $\mathbf { p } _ { k + 1 } = \mathbf { p } _ { k } + \mathbf { F } _ { k } \big ( \mathbf { p } _ { k } \big )$ represents the correspondence of the same physical point across consecutive image frames.

7) $\nu _ { k } \mathbf { \cdot }$ : The final estimated scalar speed of the vehicle at time $k$

## Dynamic Image-Template Homography Estimation

The projective relationship between frame $I _ { k }$ and $V _ { k }$ is modeled by a homography matrix $\mathbf { H } _ { k } \in \mathbb { R } ^ { 3 \times 3 }$ Using homogeneous coordinates, the mapping from image space to the metric vehicle plane is given by:

$$
w \left[ \begin{array} { l } { x } \\ { y } \\ { 1 } \end{array} \right] = \mathbf { H } _ { k } \left[ \begin{array} { l } { u } \\ { \nu } \\ { 1 } \end{array} \right]\tag{1}
$$

or equivalently,

$$
w \widetilde { \mathbf { s } } _ { k } ^ { V _ { k } } = \mathbf { H } _ { k } \widetilde { \mathbf { p } } _ { k }\tag{2}
$$

where $w \ne 0$ is a projective scale factor. The corresponding Euclidean coordinates on the vehicle plane are recovered by normalizing the homogeneous representation with respect to the third component, and we expressed it as:

$$
\mathbf { s } _ { k } ^ { V _ { k } } = \sigma ( \mathbf { H } _ { k } \tilde { \mathbf { p } } _ { k } )\tag{3}
$$

The homography matrix $\mathbf { H } _ { k }$ can be estimated from at least four non-collinear correspondences between the semantics keypoint $p _ { k } \in { \mathcal { M } } _ { k }$ , and their corresponding locations on a physical vehicle metric template. When more than four correspondences are available, $\mathbf { H } _ { k }$ is estimated by solving an overdetermined system, with Random Sample Consensus (RANSAC) algorithm to reject outliners (Fischler and Bolles 1981). The mathematical formulation of homography estimation is well established and is therefore omitted here (Hartley and Zisserman 2003).

We defined a vehicle metric template represented by a 2D mechanical drawing labeled with 36 semantic keypoints. Each keypoint is associated with 2D coordinates in meters. Figure 4 illustrates the sedan-based metric template used in our experiments. To preserve the planar assumption underlying Equation (1), we selected subsets of approximately co-planar keypoints corresponding to a specific vehicle facet, for example, wheels center, doors, door handle, side window corners. By associating detected image keypoints $\mathbf { p } _ { k }$ with their corresponding metric template coordinates $\mathbf { s } _ { k } ^ { \bar { V _ { k } } }$ , an instantaneous homography matrix $\mathbf { H } _ { k }$ for each frame � was estimated. Because the metric scale is enforced by known dimensions of the canonical template, $\mathbf { H } _ { k }$ provides an image-to-metric mapping that adapts to the vehicle’s current position within the field of view.

While homography-based methods can theoretically accommodate viewpoint changes, their accuracy relies on the assumption that the observed motion on a vehicle surface can be represented by a single rigid plane. In practice, vehicle surfaces are only approximately planar, and vehicle motion introduces deviations due to suspension dynamics, minor rotations, and depth variations across keypoints. These efects cause the true image-to-plane transformation to gradually deviate from the estimated homography, resulting in persistent metric errors. Moreover, the homography is estimated from keypoint correspondences at a specific vehicle pose and is therefore only valid for that reference configuration. As the vehicle moves, the tracked keypoints may undergo increasingly large image displacements relative to their original positions, while the fixed homography cannot account for the resulting changes in projection. Consequently, planar approximation errors and temporal homography mismatch accumulate over time, leading to drift in displacement and speed estimates.

To address this limitation, this paper models the homography as a dynamic, time-varying transformation. By re-estimating the homography at each time step using keypoints associated with the same vehicle planar facet, the proposed approach continuously adapts the planar approximation to the vehicle’s current pose. This dynamic estimation reduces the accumulation of projection bias and improves the consistency of displacement and speed estimates over time.

## Speed Estimation via Warped Optical Flow

At time step �, a planar metric coordinate frame $V _ { k }$ is defined for each tracked vehicle facet. Given a homography $\mathbf { H } _ { k }$ between the image frame $I _ { k }$ and the metric planar frame $V _ { k }$ , the metric displacement of an arbitrary point on the facet can be recovered if its image correspondence between consecutive frames is known. Specifically, for a point $\mathbf { p } _ { i , k }$ and its corresponding location $\mathbf { p } _ { i , k + 1 }$ at the next time step, the instantaneous speed is computed as

$$
\nu _ { s _ { i , k } } = \frac { \left\| \sigma ( \mathbf { H } _ { k } \tilde { \mathbf { p } } _ { i , k + 1 } ) - \sigma ( \mathbf { H } _ { k } \tilde { \mathbf { p } } _ { i , k } ) \right\| } { \Delta t } ,\tag{4}
$$

where $\tilde { \mathbf { p } }$ represents the point in homogeneous coordinates, $\sigma ( \cdot )$ denotes the homogeneous normalization operator, and $\nu _ { s _ { i , k } }$ represents the metric speed of the tracked point. Note that $\mathbf { H } _ { k }$ is only valid for points belonging to the same planar facet used during its estimation.

The required point correspondence can be obtained using either dense or sparse tracking strategies. For dense correspondence, the optical flow field $\mathbf { F } _ { k }$ is used to obtain the subsequent image location:

$$
\mathbf { p } _ { i , k + 1 } = \mathbf { p } _ { i , k } + \mathbf { F } _ { k } \big ( \mathbf { p } _ { i , k } \big ) .\tag{5}
$$

Alternatively, sparse correspondence can be obtained by detecting and matching semantic keypoints that appear in both consecutive frames. This provides direct correspondences between $\mathbf { p } _ { i , k }$ and $\mathbf { p } _ { i , k + 1 }$ without relying on dense optical flow estimation.

For the optical flow strategy, a convex hull $S _ { k }$ is constructed from the detected semantic keypoints $\mathbf { p } _ { k } \in M _ { k }$ to restrict the queried points to the same planar facet. Optical flow is then evaluated only for points within this convex hull, ensuring that the resulting correspondences remain valid under the facet-specific homography $\mathbf { H } _ { k }$ . Moreover, this constraint increases the number of available tracking points beyond the sparse semantic keypoints, enabling robust aggregation over multiple measurements and reducing the influence of individual optical flow outliers.

The final vehicle speed estimate is obtained by aggregating the point-wise speed estimates over all tracked points in $S _ { k }$ :

$$
\nu _ { k } = \mathrm { R o b u s t A g g } \left( \{ \nu _ { s _ { i , k } } \mid s _ { i , k } \in S _ { k } \} \right) ,\tag{6}
$$

where RobustAgg(·) denotes a robust aggregation function used to reduce the influence of optical flow outliers and keypoint localization errors.

![](images/dde7a2ea59eeaf6f5b2c9ed1d81bf850f2ea77fa70747738388d7c5655959f30.jpg)  
FIGURE 2: Illustration of speed estimation via warped optical flow for a tracked vehicle’s right facet. The vehicle graphic was generated using AI tools.

## Error Analysis

The speed estimate in (4) depends on the accuracy of the estimated homography matrix, which is recovered from image keypoint correspondences, and the accuracy of the optical flow. To quantify this efect, we analyzed the first-order sensitivity of the proposed speed estimation method to homography estimation errors.

For notation simplicity, the subscript � is dropped in the following analysis. Let $H ^ { * }$ denote the true homography matrix, and $\Delta H$ denote the homography matrix estimation error, yielding:

$$
H = H ^ { * } + \Delta H\tag{7}
$$

Similarly, let $s _ { 1 } = s _ { 0 } + \nu _ { O F } ^ { * }$ represents the true correspondence between two frames, where $\nu _ { O F } ^ { * }$ is the true optical flow vector. Accounting for optical flow error $\Delta \nu _ { O F }$ , we have:

$$
\nu _ { O F } = \nu _ { O F } ^ { * } + \Delta \nu _ { O F }\tag{8}
$$

Using Equation (4), the true velocity $\nu ^ { * }$ satisfies:

$$
\nu ^ { * } \Delta t = \sigma ( H ^ { * } s _ { 1 } ^ { * } ) - \sigma ( H ^ { * } s _ { 0 } ^ { * } )\tag{9}
$$

While the estimated velocity � is computed as:

$$
\nu = \Delta t ^ { - 1 } ( \sigma ( H s _ { 1 } ) - \sigma ( H s _ { 0 } ) )\tag{10}
$$

$$
= \Delta t ^ { - 1 } ( \sigma ( ( H ^ { * } + \Delta H ) s _ { 1 } ) - \sigma ( ( H ^ { * } + \Delta H ) s _ { 0 } ) )\tag{11}
$$

The first-order approximation of the normalization function $\sigma$ gives:

$$
\sigma ( ( H ^ { * } + \Delta H ) s ) \approx \sigma ( H ^ { * } s ) + J _ { \sigma } ( H ^ { * } s ) \Delta H s\tag{12}
$$

where $J _ { \sigma }$ is the Jacobian of the normalization operator $\sigma$ . Consequently,

$$
\nu - \nu ^ { * } \approx \Delta t ^ { - 1 } \big ( J _ { \sigma } ( H ^ { * } s _ { 1 } ) \Delta H s _ { 1 } - J _ { \sigma } ( H ^ { * } s _ { 0 } ) \Delta H s _ { 0 } \big )\tag{13}
$$

Assuming small inter-frame motion such that $J _ { \sigma } ( H ^ { * } s _ { 1 } ) \approx J _ { \sigma } ( H ^ { * } s _ { 0 } ) = J$ , it follows that:

$$
\nu - \nu ^ { * } \approx \Delta t ^ { - 1 } J \Delta H ( s _ { 1 } - s _ { 0 } ) = \Delta t ^ { - 1 } J \Delta H \nu _ { O F }\tag{14}
$$

The error is bounded by:

$$
\| \Delta \nu \| = \| \nu - \nu ^ { * } \| \leq \Delta t ^ { - 1 } \| J \| \| \Delta H \| \| \nu _ { O F } \|\tag{15}
$$

The bound in (15) indicates that the speed estimation error is afected by three components: the perspective sensitivity $\| J _ { \sigma } \|$ , the homography estimation error $\| \Delta H \|$ , and the magnitude of the optical flow $\| \nu _ { O F } \|$ . The first term reflects geometric amplification driven by the viewing angle between the vehicle facet and the image plane. The second term depends on the conditioning of the linear system used to solve for the homography. The third term shows that larger image motions linearly scale the propagation of homography errors into the final metric velocity.

## Integrated System Pipeline

Building upon the theoretical framework established in the previous section, the architecture of the proposed integrated system is illustrated in Figure 3. The pipeline processes a sequence of video frames to simultaneously extract dense motion features and semantic vehicle attributes.

For each incoming frame pair $\left\{ { { I } _ { k } } , { { I } _ { k + 1 } } \right\}$ , a dense optical flow field $\mathbf { F } _ { k }$ is computed to capture pixellevel displacements. In parallel, a Multi-Object Tracking (MOT) module identifies, localizes and generate bounding boxes for each vehicle. Based on these bounding boxes, the image is cropped to generate vehicle-specific Regions of Interest (ROIs). This spatial isolation ensures that keypoints are detected exclusively within the context of the target vehicle, thereby minimizing background noise and enhancing the accuracy of the detection model.

The core component of the proposed system is the Estimate Speed and Update Homography module, which integrates semantic keypoints, optical flow, and adaptive homography estimation for metric speed recovery. At each iteration, the module executes a three-step procedure:

1) Keypoint Detection and Classification: First, a specialized keypoint detection network identifies semantic keypoints within the vehicle ROI and classifies them according to predefined vehicle facets (e.g., the left wheel center keypoint belongs to the left facet).

2) Speed Estimation: For each facet, the system retrieves the associated keypoints and constructs the convex hull $S _ { k }$ , employing the homography matrix $\mathbf { H } _ { k }$ established at the previous time step. Using the inter-frame time interval $\Delta t$ and the optical flow vectors, the metric speed for each point $s _ { i , k } \in S _ { k }$ is calculated via Equation 4. To determine the final vehicle speed, we aggregate the individual point speeds using an arithmetic mean. Note that more robust aggregation functions (e.g., RANSAC or median filtering) may be employed here to mitigate the influence of outliers.

![](images/e0a1375c5820ac99d8c546f6959a8bbbe3332932844c6258c61bab0d11ebdc9b.jpg)  
FIGURE 3: Proposed System Architecture.

3) Geometric Calibration (Homography Update): Finally, the homography matrix is updated for the next iteration. For any facet containing a suficient number of valid keypoints (specifically, $N \geq 4$ non-collinear points), the homography is recomputed to account for changes in the vehicle’s perspective or road geometry.

This recursive estimate-then-update strategy ensures that the metric scaling remains tightly coupled to the vehicle’s evolving perspective and distance. By continuously refining the homography matrices based on new keypoint detections, the system efectively resets accumulated tracking error for every frame in the sequence, mitigating drift over long tracking durations.

It is important to note that this design introduces a deterministic one-frame latency, as the speed estimate for time step � relies on the optical flow derived from the arrival of the subsequent frame $I _ { k + 1 }$ . While technically acausal with respect to the instantaneous frame $I _ { k }$ , this forward-looking dependency is inherent to dense optical flow algorithms, which require temporal displacement to compute motion. However, given the high temporal resolution of modern camera hardware, typically ≥ 30 frames per second (FPS), this results in a negligible latency of approximately 33 ms. This delay is well within the acceptable operational bounds for real-time applications such as trafic flow analysis and automated speed enforcement.

## EXPERIMENT RESULT AND DISCCUSION

Experiments were conducted to evaluate the efectiveness of the proposed vehicle speed estimation framework. The evaluation focuses on the accuracy of the recovered vehicle speeds, the robustness of the proposed homography-based metric mapping, and the performance of the system under varying vehicle poses and camera viewpoints.

A sedan model consisting of 36 semantic keypoints was used as the metric template, as shown in Figure 4. Each keypoint was assigned a 2D (image) coordinates in meters, assuming a wheelbase (horizontal) distance of 3 meters. The keypoint detection module was trained using the Ultralytics

YOLO framework with the YOLO-11s backbone (Jocher et al. 2026). The training dataset consists of 756 images for training and 247 images for validation, collected from the Waymo Open Dataset (Sun et al. 2020), Roundabout HD dataset (Lin et al. 2025), and BoxCar dataset (Sochor et al. 2018a).

![](images/b4b75a51c4f950918d8c4c9581059025aa224245e618597883f20576f6168b36.jpg)  
FIGURE 4: 36 semantic keypoints for the sedan are marked with blue circles (symmetric on the other side). Assuming a wheelbase of approximately 3 meters. Car graphics were adapted from Wikipedia. (842U 2010)

The validation experiments were conducted using video sequences from the VS13 (resolution 1920 × 1080, 30FPS) (Djukanović et al. 2022) and BrnoCompSpeed datasets (resolution 1920 × 1080, 50FPS) (Sochor et al. 2018b), with total 443 video clips across 208 unique vehicles and speed ranging from 30 mph to 100 mph. None of these evaluation sequences were part of the keypoint detection model’s training data. The two datasets ofer two diferent camera viewpoints: VS13 provides pedestrian-level angles of approaching trafic, while BrnoCompSpeed provides overhead surveillance views. This allows us to test the framework across diferent viewing geometries and motion profiles. Figure 5 shows sample processed frames from each dataset, with tracked cars marked by green bounding boxes and detected semantic keypoints highlighted as blue circles. We also compare two distinct speed estimation strategies: a Keypoint-only (KP) approach that tracks sparse semantic features, and our proposed Warped Optical Flow (OF) method, which leverages a keypoint-defined convex hull to achieve robust, dense spatial aggregation. All errors are presented as relative percentages for comparison across diferent speeds.

![](images/edf074084ca5ce22dff1c4d24b1bbef7e85bc1d9a3d0fb99087d15513fcb5157.jpg)

![](images/e2f1d55bfa778453d8c33324d4c4ca7e47041f28dc262667038de1a21443889d.jpg)  
FIGURE 5: Sample image from the VS13 dataset (left) and BrnoCompSpeed dataset (right), annotated using the speed estimation framework.

Table 1 presents the overall accuracy of speed estimation in both datasets. For the VS13 dataset, the optical flow method achieves a Mean Absolute Error (MAE) of 15.0% with a median error of–6.0% and 77.9% of estimates within ±20% error, compared to the keypoint method which achieves an MAE of 20.8% with a median error of -0.9% and 87.6% within ±20%. Although the keypoint method achieves higher within-threshold accuracy, which is particularly relevant for speed limit enforcement, the optical flow method yields significantly lower MAE and RMSE, indicating more consistent and accurate estimates overall. For the BrnoCompSpeed dataset, both methods perform significantly better, with optical flow achieving an MAE of 9.7% and 93.1% within ±20%, compared to 12.8% MAE and 87.9% within ±20% for the keypoint method. The optical flow method consistently outperforms the keypoint approach across all metrics on this dataset. Across both datasets, optical flow demonstrates higher within-threshold accuracy. Performance diferences between sedans and non-sedans are very small, with sedans generally achieving slightly lower MAE with optical flow, likely due to the dataset being more trained on sedans.

TABLE 1: Summary of speed estimation accuracy for VS13 and BrnoCompSpeed datasets.
<table><tr><td rowspan="2">Data</td><td rowspan="2">Method</td><td rowspan="2">Group</td><td rowspan="2">N</td><td rowspan="2">Med.</td><td rowspan="2">MAE</td><td rowspan="2">RMSE</td><td colspan="4">Thresholds</td></tr><tr><td> $\overline { { \pm 5 \% } }$ </td><td> $\overline { { \pm 1 0 \% } }$ </td><td> $\overline { { \pm 2 0 \% } }$ </td><td> $\overline { { > 5 0 \% } }$ </td></tr><tr><td rowspan="6">VS13</td><td></td><td>All</td><td>4,317</td><td>-0.9%</td><td>20.8%</td><td>178.4%</td><td>46.7%</td><td>72.5%</td><td>87.6%</td><td>4.4%</td></tr><tr><td>KP</td><td>Sedan</td><td>2,055</td><td>-0.7%</td><td>24.3%</td><td>243.0%</td><td>46.0%</td><td>73.3%</td><td>89.7%</td><td>3.2%</td></tr><tr><td></td><td>NoSed</td><td>2,262</td><td>-1.0%</td><td>17.7%</td><td>84.2%</td><td>47.3%</td><td>71.7%</td><td>85.7%</td><td>5.6%</td></tr><tr><td></td><td>All</td><td>4,578</td><td>-6.0%</td><td>15.0%</td><td>53.5%</td><td>37.3%</td><td>58.8%</td><td>77.9%</td><td>5.3%</td></tr><tr><td>OF</td><td>Sedan</td><td>2,195</td><td>-6.1%</td><td>14.3%</td><td>22.1%</td><td>35.4%</td><td>56.8%</td><td>76.1%</td><td>4.8%</td></tr><tr><td></td><td>NoSed</td><td>2,383</td><td>-6.0%</td><td>15.6%</td><td>71.1%</td><td>39.1%</td><td>60.7%</td><td>79.6%</td><td>5.7%</td></tr><tr><td rowspan="6">BC</td><td>KP</td><td>All</td><td>17,085</td><td>6.5%</td><td>12.8%</td><td>46.9%</td><td>28.3%</td><td>54.3%</td><td>87.9%</td><td>1.2%</td></tr><tr><td></td><td>Sedan</td><td>3,361</td><td>6.9%</td><td>11.1%</td><td>28.8%</td><td>32.0%</td><td>58.9%</td><td>89.0%</td><td>0.6%</td></tr><tr><td></td><td>NoSed</td><td>13,724</td><td>6.3%</td><td>13.2%</td><td>50.4%</td><td>27.3%</td><td>53.1%</td><td>87.6%</td><td>1.3%</td></tr><tr><td></td><td>All</td><td>17,600</td><td>3.0%</td><td>9.7%</td><td>66.1%</td><td>41.4%</td><td>70.6%</td><td>93.1%</td><td>1.0%</td></tr><tr><td>OF</td><td>Sedan</td><td>3,452</td><td>3.9%</td><td>11.0%</td><td>141.0%</td><td>47.5%</td><td>75.1%</td><td>93.9%</td><td>0.7%</td></tr><tr><td></td><td>NoSed</td><td>14,148</td><td>2.7%</td><td>9.3%</td><td>24.2%</td><td>39.9%</td><td>69.5%</td><td>92.9%</td><td>1.0%</td></tr></table>

Note: KP = Keypoints; OF = Optical Flow. Med. = Median; MAE = Mean Absolute Error; RMSE = Root Mean Square Error; N = Number of frames with valid estimation.

![](images/9173c027a34eaab6db91a3f2cd131859db1ad9448ed8580ff2a692a5d6c9fba1.jpg)

![](images/6be50ebaa8bef249f205f2aa7e7d6f9b2ae587af265715457db58438f414515e.jpg)

(a) VS13 Keypoints  
![](images/6ef42d85b309e141446af8760d0839a0ca402025aa143da87b266b428f1bf48d.jpg)  
(c) BrnoCompSpeed Keypoints

(b) VS13 Optical Flow  
![](images/911732c87e0716e6ae15b626d527c8745c7e2223963f4d21152cb06658adc4d5.jpg)  
(d) BrnoCompSpeed Optical Flow  
FIGURE 6: Error Percentage vs. Frame Index for VS13 and BrnoCompSpeed datasets

To investigate the temporal behavior of estimation errors, the error percentage was analyzed as a function of the normalized frame index. Specifically, frame indices were scaled to a range of 0 to 1 based on the first and last frames containing a valid speed estimation. Figure 6 displays heatmaps of error percentage versus normalized frame index for both methods and datasets. The heatmaps indicate that errors were significantly higher at the beginning and the end of each video sequence, when vehicles were either too far from the camera—resulting in a small pixel footprint—or too close, partially leaving the frame and making feature tracking unreliable. This directly increased homography estimation error.

Furthermore, the OF method became particularly unstable at extreme vehicle-to-camera distances. This vulnerability is more obvious in the VS13 scenario where vehicles move directly toward the camera, resulting in dramatic scale changes. Consequently, OF exhibits a much wider error spread in Figure 6(b) compared to BrnoCompSpeed, where the camera is positioned overhead and the relative distance change remains minimal across the frames.

To mitigate the efects of unreliable detections at the very first and last frame, a 10% trim was applied to each tail of the error distribution. Table 2 summarizes the results after this trimming. For the VS13 dataset, the keypoint method’s MAE improved from 20.8% to 14.3%, while its within ±20% accuracy increased from 87.6% to 94.0%. The optical flow method also improved from 15.0% to 11.7% in MAE, with within ±20% rising from 77.9% to 85.3%. In terms of the BrnoCompSpeed dataset, the optical flow method’s MAE dropped from 9.7% to 7.6%, and the keypoint method’s MAE dropped from 12.8% to 11.4%. The trimming also significantly reduced outlier rates. For example, for the VS13 dataset, the keypoint method’s outlier rate decreased from 4.4% to 2.1%, while the optical flow method’s outlier rate decreases from 5.3% to 2.3%. Both method also showed enhanced performance on BrnoCompSpeed dataset after trimming.

Figure 7 compares the speed estimation errors across diferent methods and datasets, with ground truth speeds ranging from 20 to 70 mph. The most accurate estimations occur in the 40 to 60 mph speed range, where both optical flow-based methods achieve errors below 5%. In contrast, estimation errors are largest at low speeds (20 to 30 mph), particularly for keypoint-based methods, where errors exceed 20%, possibly due to increased sensitivity to noise.

## CONCLUSION

This paper demonstrates that monocular camera-based vehicle speed estimation can be efectively achieved using a vehicle metric template, a lightweight vehicle keypoint detection module, and a homography technique. Crucially, the proposed method operates without requiring camera calibration (intrinsic or extrinsic) or pre-existing roadway features, relying only on a relatively stationary camera view. The approach achieves reliable and accurate speed estimation, with 87.6% and 87.9% of estimates falling within the ±20% error threshold across the evaluated datasets, demonstrating its efectiveness under diverse viewpoints and driving conditions.

TABLE 2: Summary of speed estimation accuracy for VS13 and BrnoCompSpeed datasets with 10% trimming.
<table><tr><td rowspan="2">Data</td><td rowspan="2">Method</td><td rowspan="2">Group</td><td rowspan="2">N</td><td rowspan="2">Med.</td><td rowspan="2">MAE</td><td rowspan="2">RMSE</td><td colspan="4">Thresholds</td></tr><tr><td>±5%</td><td>±10%</td><td>±20%</td><td>&gt;50%</td></tr><tr><td rowspan="6">VS13</td><td rowspan="3">KP</td><td>All</td><td>3,334</td><td>-1.2%</td><td>14.3%</td><td>170.6%</td><td>54.0%</td><td>81.0%</td><td>94.0%</td><td>2.1%</td></tr><tr><td>Sedan</td><td>1,584</td><td>-1.2%</td><td>18.5%</td><td>240.9%</td><td>52.0%</td><td>79.9%</td><td>94.1%</td><td>1.8%</td></tr><tr><td>NoSed</td><td>1,750</td><td>-1.3%</td><td>10.5%</td><td>54.1%</td><td>55.7%</td><td>82.0%</td><td>93.9%</td><td>2.3%</td></tr><tr><td rowspan="3">OF</td><td>All</td><td>3,555</td><td>-5.1%</td><td>11.7%</td><td>57.4%</td><td>43.0%</td><td>66.2%</td><td>85.3%</td><td>2.3%</td></tr><tr><td>Sedan</td><td>1,692</td><td>-5.4%</td><td>11.6%</td><td>17.7%</td><td>40.1%</td><td>62.4%</td><td>81.7%</td><td>2.4%</td></tr><tr><td>NoSed</td><td>1,863</td><td>-4.8%</td><td>11.7%</td><td>77.4%</td><td>45.6%</td><td>69.7%</td><td>88.5%</td><td>2.1%</td></tr><tr><td rowspan="6">BC</td><td rowspan="3">KP</td><td>All</td><td>13,822</td><td>6.0%</td><td>11.4%</td><td>28.4%</td><td>28.5%</td><td>55.0%</td><td>89.4%</td><td>0.6%</td></tr><tr><td>Sedan</td><td>2,738</td><td>6.8%</td><td>9.5%</td><td>11.9%</td><td>32.8%</td><td>60.3%</td><td>90.8%</td><td>0.0%</td></tr><tr><td>NoSed</td><td>11,084</td><td>5.7%</td><td>11.8%</td><td>31.2%</td><td>27.4%</td><td>53.6%</td><td>89.1%</td><td>0.8%</td></tr><tr><td rowspan="3">OF</td><td>All</td><td>14,198</td><td>2.3%</td><td>7.6%</td><td>11.4%</td><td>43.9%</td><td>73.9%</td><td>95.4%</td><td>0.5%</td></tr><tr><td>Sedan</td><td>2,794</td><td>3.4%</td><td>6.5%</td><td>8.9%</td><td>50.9%</td><td>79.1%</td><td>96.7%</td><td>0.1%</td></tr><tr><td>NoSed</td><td>11,404</td><td>1.9%</td><td>7.9%</td><td>12.0%</td><td>42.2%</td><td>72.7%</td><td>95.1%</td><td>0.6%</td></tr></table>

Note: KP = Keypoints; OF = Optical Flow. Med. = Median; MAE = Mean Absolute Error; RMSE = Root Mean  
Square Error; N = Number offrames with valid estimation.

![](images/7abcd48364a1ef06b520ac89e2dd56e0e47a5783488815440cef1dd256281a14.jpg)

![](images/f4e9a974c7fefb2b869df3875963f48d36ac6738f5d8c8da28151a549bfd6034.jpg)  
(b) VS13 Optical Flow

(a) VS13 Keypoints  
![](images/8b07782e4c63999f929f8b06cab7a3159927b5e0dfe9ac860b7faabc58feafb4.jpg)  
(c) BrnoCompSpeed Keypoints

![](images/70f4a0b5c659a753aefdab5c19b9eb80df0645ab0787795b499374011be9eb19.jpg)  
(d) BrnoCompSpeed Optical Flow  
FIGURE 7: Error Percentage vs. Speed for VS13 and BrnoCompSpeed datasets

Experimental results also indicate that correspondence computation via the proposed warped optical flow method is more reliable and robust against outliers. In contrast, purely keypoint-based correspondence can ofer high peak accuracy but remains sensitive to keypoint detection jitter.

Additionally, this work has significant practical implications for trafic enforcement and public safety. Unlike traditional speed measurement systems that rely on roadway features, the proposed method is deployable and cost-efective. This facilitates citizen-based enforcement, where the public can use portable devices, such as dashcams or smartphones, to document speeding events (Li et al. 2025). Nowadays, many cities have piloted citizen-based enforcement programs. For example, New York City has empowered citizens to enforce diesel-truck idling laws by submitting video evidence; those who provide at least three minutes offootage receive 25% ofany fine collected, amounting to about \$87.50 per violation (Hollis 2026). As vision-based enforcement becomes increasingly popular due to budget constraints and the rapid growth of camera-equipped devices, the proposed approach delivers a solution that can complement existing enforcement mechanisms while reducing the cost of camera installation.

However, the current implementation has certain limitations. It relies on a generalized sedan metric template, which can introduce minor errors due to keypoint detection inaccuracies and homography transformations. Although experiments show that the framework maintains decent accuracy on nonsedan vehicles, future work will focus on expanding the template library to include diverse vehicle classes and geometries. Additionally, while the warped optical flow approach mitigates individua outliers, achieving even higher accuracy will likely require more robust and reliable correspondence tracking methods to handle challenging conditions such as illumination changes or vehicles in close proximity to the camera.

## ACKNOWLEDGMENTS

The authors thank Helena Chandy for her valuable insights and edits.

## DECLARATION OF CONFLICTING INTERESTS

All authors declare no potential conflicts of interest with respect to the research, authorship, and publication of this article.

## REFERENCES

Sadeghi-Bazargani, H. and M. Saadati, Speed management strategies; a systematic review. Bulletin of Emergency & Trauma, Vol. 4, No. 3, 2016, p. 126.

Shin, K., S. P. Washington, and I. van Schalkwyk, Evaluation of the Scottsdale Loop 101 automated speed enforcement demonstration program. Accident Analysis & Prevention, Vol. 41, No. 3, 2009, pp. 393–403.

Sangsuwan, K. and M. Ekpanyapong, Video-based vehicle speed estimation using speed measurement metrics. IEEE Access, Vol. 12, 2024, pp. 4845–4858.

Odat, E., J. S. Shamma, and C. Claudel, Vehicle classification and speed estimation using combined passive infrared/ultrasonic sensors. IEEE transactions on intelligent transportation systems, Vol. 19, No. 5, 2017, pp. 1593–1606.

Martin, P. T., Y. Feng, X. Wang, et al., Detector technology evaluation. Mountain-Plains Consortium Fargo, ND, USA, 2003.

Webb, C., The Economics of Inductive Loops vs. Radar Trafic Detection, 2020, accessed: 2026- 01-04.

Javadi, S., M. Dahl, and M. I. Pettersson, Vehicle speed measurement model for video-based systems. Computers & electrical engineering, Vol. 76, 2019, pp. 238–248.

Hovhannisyan, T., Active vs Passive Sensing: Is One Better Than the Other?, 2021, updated: Aug. 2, 2023; Jul. 22, 2024. Accessed: 2026-01-05.

Martínez, A. H., I. G. Daza, C. F. López, and D. F. Llorca, Digital twins to alleviate the need for real field data in vision-based vehicle speed detection systems. In 2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC), IEEE, 2024, pp. 2536–2541.

Zhang, Y., Z. Guo, J. Wu, Y. Tian, H. Tang, and X. Guo, Real-time vehicle detection based on improved yolo v5. Sustainability, Vol. 14, No. 19, 2022, p. 12274.

Zuraimi, M. A. B. and F. H. K. Zaman, Vehicle detection and tracking using YOLO and DeepSORT. In 2021 IEEE 11th IEEE symposium on computer applications & industrial electronics (ISCAIE), IEEE, 2021, pp. 23–29.

Maity, M., S. Banerjee, and S. S. Chaudhuri, Faster r-cnn and yolo based vehicle detection: A survey. In 2021 5th international conference on computing methodologies and communication (ICCMC), IEEE, 2021, pp. 1442–1447.

Xu, Z., W. Yang, A. Meng, N. Lu, H. Huang, C. Ying, and L. Huang, Towards end-to-end license plate detection and recognition: A large dataset and baseline. In Proceedings of the European conference on computer vision (ECCV), 2018, pp. 255–271.

Montazzolli, S. and C. Jung, Real-time brazilian license plate detection and recognition using deep convolutional neural networks. In 2017 30th SIBGRAPI conference on graphics, patterns and images (SIBGRAPI), IEEE, 2017, pp. 55–62.

Parsa, P., K. Li, K. M. Kockelman, and S. Choi, Video-based Vehicle Surveillance in the Wild: License Plate, Make, and Model Recognition with Self Reflective Vision-Language Models. arXiv preprint arXiv:2508.01387, 2025.

Hua, S., M. Kapoor, and D. C. Anastasiu, Vehicle tracking and speed estimation from trafic videos. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2018, pp. 153–160.

Liu, C., D. Q. Huynh, Y. Sun, M. Reynolds, and S. Atkinson, A vision-based pipeline for vehicle

counting, speed estimation, and classification. IEEE transactions on intelligent transportation systems, Vol. 22, No. 12, 2020, pp. 7547–7560.

Lu, X., J. Yaoy, H. Li, Y. Liu, and X. Zhang, 2-line exhaustive searching for real-time vanishing point estimation in manhattan world. In 2017 IEEE Winter Conference on Applications of Computer Vision (WACV), IEEE, 2017, pp. 345–353.

Kim, J.-H., W.-T. Oh, J.-H. Choi, and J.-C. Park, Reliability verification of vehicle speed estimate method in forensic videos. Forensic science international, Vol. 287, 2018, pp. 195–206.

Chang, C.-K., J. Zhao, and L. Itti, Deepvp: Deep learning for vanishing point detection on 1 million street view images. In 2018 IEEE International Conference on Robotics and Automation (ICRA), IEEE, 2018, pp. 4496–4503.

Li, K., J. Malagavalli, L. Goel, T. Wang, and K. Kockelman, Smartphone-based Method for Automated Speed Enforcement. In 2025 Transportation Research Board Annual Meeting, National Academy of Sciences, 2025.

Gunawan, A. A., D. A. Tanjung, and F. E. Gunawan, Detection of vehicle position and speed using camera calibration and image projection methods. Procedia Computer Science, Vol. 157, 2019, pp. 255–265.

Abdel-Aziz, Y. I., H. M. Karara, and M. Hauck, Direct linear transformation from comparator coordinates into object space coordinates in close-range photogrammetry. Photogrammetric engineering & remote sensing, Vol. 81, No. 2, 2015, pp. 103–107.

Zhang, Z., Camera calibration. In Computer vision: a reference guide, Springer, 2021, pp. 130–131.

Weng, J., P. Cohen, M. Herniou, et al., Camera calibration with distortion models and accuracy evaluation. IEEE Transactions on pattern analysis and machine intelligence, Vol. 14, No. 10, 1992, pp. 965–980.

Feng, C., F. Deng, and V. R. Kamat, Semi-automatic 3d reconstruction of piecewise planar building models from single image. CONVR (Sendai:), Vol. 2, No. 5, 2010, p. 6.

Barinova, O., V. Lempitsky, E. Tretiak, and P. Kohli, Geometric image parsing in man-made environments. In European conference on computer vision, Springer, 2010, pp. 57–70.

Collins, R. T. and R. S. Weiss, Vanishing point calculation as a statistical inference on the unit sphere. In ICCV, 1990, Vol. 90, pp. 400–403.

Song, J., H. Song, and S. Wang, PTZ camera calibration based on improved DLT transformation model and vanishing Point constraints. Optik, Vol. 225, 2021, p. 165875.

Zhai, M., S. Workman, and N. Jacobs, Detecting vanishing points using global image context in a non-manhattan world. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 5657–5665.

Rani, N. G., N. H. Priya, A. Ahilan, and N. Muthukumaran, LV-YOLO: logistic vehicle speed detection and counting using deep learning based YOLO network. Signal, Image and Video Processing, Vol. 18, No. 10, 2024, pp. 7419–7429.

Hartley, R. and A. Zisserman, Multiple view geometry in computer vision. Cambridge university press, 2003.

Djukanović, S., N. Bulatović, and I. Čavor, A dataset for audio-video based vehicle speed estimation. In 2022 30th Telecommunications Forum (TELFOR), IEEE, 2022, pp. 1–4.

Fischler, M. A. and R. C. Bolles, Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM, Vol. 24, No. 6, 1981, pp. 381–395.

Jocher, G., J. Qiu, M. Liu, S. Lyu, F. C. Akyon, and M. E. Kalfaoglu, Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models, 2026.

Sun, P., H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine, et al., Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 2446–2454.

Lin, Y., S. Lockyer, M. Sui, L. Gan, F. Stanek, M. Zarbock, W. Li, A. Evans, and N. Zhang, RoundaboutHD: High-Resolution Real-World Urban Environment Benchmark for Multi-Camera Vehicle Tracking. arXiv preprint arXiv:2507.08729, 2025.

Sochor, J., J. Špaňhel, and A. Herout, Boxcars: Improving fine-grained recognition of vehicles using 3-d bounding boxes in trafic surveillance. IEEE transactions on intelligent transportation systems, Vol. 20, No. 1, 2018a, pp. 97–108.

842U, Three body styles with pillars and boxes.png. Wikimedia Commons. https://commons. wikimedia.org/ wiki/File:Three\_body\_styles\_with\_pillars\_and\_boxes.png. Accessed: 2026-07-29, 2010.

Sochor, J., R. Juránek, J. Špaňhel, L. Maršík, A. Široky, A. Herout, and P. Zemčík, Comprehensive\` data set for automatic single camera visual speed measurement. IEEE Transactions on Intelligent Transportation Systems, Vol. 20, No. 5, 2018b, pp. 1633–1643.

Hollis, D., Trucking association sues New York City over idling law, citizen videos. Truckers News, 2026, updated July 14, 2026.