# A 360-Degree Vision Dataset for Learning Yaw Control on GPS-Denied Micro-UAVs in Disaster-Response-Relevant Environments

Niklas Voigt

Robotics Laboratory

Westphalian University of Applied Sciences (WHS)

Gelsenkirchen, Germany

niklas.voigt@w-hs.de

Hartmut Surmann

Robotics Laboratory

Westphalian University of Applied Sciences (WHS)

Gelsenkirchen, Germany

hartmut.surmann@w-hs.de

Abstract—This paper presents a novel data-driven approach to camera-based autonomy for micro-drones in GPS-denied, radio-challenging indoor environments. The target application is disaster and emergency response, where micro-UAVs can provide rapid situational awareness in hazardous settings such as firefighting and chemical, biological, radiological, and nuclear (CBRN) incidents while reducing risk for human responders. When the communication link is lost, the micro-drone uses a learned yaw controller to autonomously navigate toward open space, preserving onboard sensor data that would otherwise be lost with the vehicle. A custom micro-drone equipped with a 360-degree camera was used to record diverse industrial, underground, and training scenarios representative of communicationdenied field operations. We introduce a preprocessing pipeline that converts equirectangular 360-degree footage into planar front views and dynamically generates image-label pairs for AI training. We then train and compare multiple convolutional neural network variants that predict a continuous yaw command from a single monocular view. Evaluation on a held-out test set confirms the feasibility of the learned yaw-prediction approach. A semi-autonomous real-world test further demonstrates the practicality of the method while revealing key failure modes, particularly reflections and glare.

Index Terms—Disaster Field Robotics, Micro-UAVs, GPS-Denied Navigation, Omnidirectional Vision Dataset, Learning-Based Yaw Control

## I. INTRODUCTION

Unmanned aerial vehicles (UAVs) are increasingly used to support first responders and inspection teams because they can provide situational awareness without exposing human operators to direct risk [1]. More broadly, robots have been increasingly deployed in disaster and emergency response to provide situational awareness in hazardous environments while reducing risk to human responders. Seminal work in disaster robotics has highlighted the importance of robust perception and autonomy under conditions such as GPS denial, communication loss, and environmental clutter, where human teleoperation may be unreliable or impossible [2].

![](images/b8a4c98ba72527240c4012b461d818810d0bad99ae1f1d8fe3156dabb12d7c17.jpg)  
Fig. 1. Dataset overview: A 3×3 grid of equirectangular 360° thumbnails showing diverse GPS-denied disaster-response-relevant environments, including mines, fire training facilities, industrial sites, bridge passages, and outdoor areas.

Micro-UAVs are particularly suited for reconnaissance in emergency response because they can be deployed quickly, reduce risk to humans, and can be treated as expendable assets if needed [3]. However, real deployments frequently occur in GPS-denied, cluttered, and communication-challenging environments such as industrial facilities, tunnels, and basements, where radio links degrade due to shadowing and attenuation [4]. If the control link drops, a micro-drone that can autonomously stabilize and search for a safe exit path can substantially increase the operational value of such systems [5]. This recovery capability addresses a concrete operational need by maintaining controlled flight and retreating from communication-denied regions, thereby preserving reconnaissance data (imagery, thermal maps, or gas sensor readings) that would otherwise be lost with the vehicle. Such data recovery is often the primary mission objective in disaster response, where situational awareness can directly inform tactical decisions for human responders.

In addition to learning-based navigation, recent work has investigated the use of UAVs equipped with 360<sup>◦</sup> cameras and neural scene learning techniques to improve situational awareness and 3D reconstruction in disaster environments [6].

Many autonomous navigation stacks rely on active range sensing (e.g., LiDAR) for mapping and localization [7]. For micro-drones, however, such sensors can be too heavy, power hungry, or costly, motivating camera-only approaches that better align with payload, power, and cost constraints [8]. Vision-based learning approaches can exploit appearance cues such as corridor structure, openings, and obstacles to predict navigation commands when explicit mapping sensors are infeasible.

In this work, the learned yaw controller is designed as a pragmatic recovery mechanism: when the communication link is lost, it enables the micro-drone to navigate toward open space while preserving onboard sensor data that would otherwise be lost with the vehicle. This paper therefore focuses on learning a direct mapping from a monocular front view to a continuous relative yaw control command.

The main contributions are:

• a real-world dataset recorded with a custom micro-drone equipped with a 360<sup>◦</sup> camera covering diverse indoor and outdoor scenarios, together with a preprocessing pipeline that converts equirectangular footage into planar views and generates image–label pairs on-the-fly to reduce memory footprint;

• an empirical comparison of multiple CNN variants for yaw regression;

• a semi-autonomous real-world validation that reveals practical strengths and limitations.

## II. RELATED WORK

Early learning-based navigation pipelines often used classification to select coarse steering directions. Giusti et al. [9] trained a CNN to classify trail direction into three classes (turn left / go straight / turn right) based on monocular images and demonstrated outdoor trail following in forest environments. Pearson et al. [10] proposed architectural and data-handling refinements (e.g., activation choices and dataset restructuring) and reported improved classification accuracy.

In contrast to coarse direction classification, Samy et al. [11] formulated yaw prediction as a regression problem and used a CNN-based pipeline trained on synthetic data to follow waypoint paths in GPS-denied environments. Motivated by these works, we focus on real-world indoor and outdoor data and continuous yaw regression for micro-drone navigation.

More generally, our approach can be categorized as end-toend imitation learning (also referred to as behavior cloning), where a policy is trained via supervised learning to directly map visual observations to control commands based on demonstrated behavior. Such approaches have been widely used in mobile robotics and autonomous driving due to their simplicity and suitability for real-time deployment, particularly when explicit mapping or planning is infeasible [12].

In contrast to reinforcement-learning-based navigation, imitation learning enables direct use of real-world flight data without hand-crafted reward functions or online exploration — properties that have motivated its adoption in safety-critical UAV tasks such as obstacle avoidance [13], low-altitude MAV trail navigation [14], and agile flight in clutter [15], and that are particularly relevant in disaster-response settings where online trial-and-error is not admissible.

Recent deep-reinforcement-learning approaches such as DCE [16] and VDS-Nav [17] predict continuous 3D velocity and yaw-rate commands from depth images for closed-loop navigation in cluttered environments, bridging the sim-to-real gap via a real-data-augmented latent encoder and a depthbased barrier reward, respectively. Their yaw-rate choice fits their task: the policy itself closes the dynamics loop at sensor rate against a low-level velocity controller.

Our setting differs: a directional-guidance recovery component on a visual-inertial-stabilized platform, where the onboard stabilizer handles attitude and velocity tracking. The learning task therefore reduces to identifying where the recovery direction lies in a single monocular RGB view; the angle-torate gain is a downstream controller decision. An absolute yaw angle is the natural target because each label is obtained geometrically as the extraction angle of a perspective view from an equirectangular panorama — a static, single-frame quantity. Deriving a yaw rate would require either differentiating consecutive poses (coupling the network to the recording platform’s dynamics) or applying an arbitrary gain that the controller can apply equivalently downstream.

Direct quantitative comparison with prior work is challenging because existing indoor navigation datasets (e.g., outdoor trail-following [9]) target different environments and sensor configurations, and no established benchmark exists for yaw regression in GPS-denied disaster-proxy settings. Our classification sanity check (Section VI-A) confirms that the dataset and generator produce learnable signals consistent with prior three-class approaches, while the regression formulation enables finer-grained control.

## III. SYSTEM OVERVIEW

This section describes the hardware platform used for data collection and the operational concept that motivates the learned yaw controller.

## A. Platform and Sensor Setup

Data were collected primarily using a custom-built microdrone with dimensions 200 mm × 270 mm and a mass of 425 g. A modified BetaFPV SMO 360 camera (a strippeddown variant of the Insta360 ONE R 360 module optimized for FPV applications) was integrated into the frame. Each lens records at 2880 × 2880 pixels; after stitching, videos have a resolution of 5760 × 2880 at ≈ 30 fps.

A key property of the dataset is that the recorded 360<sup>◦</sup> video is unobstructed: the drone itself is not visible in the stitched video frames. In typical 360<sup>◦</sup> recordings from drone-mounted cameras, up to 33% of the panoramic sphere is occluded by the platform itself [6]. The mechanical integration used here, based on the invisible-drone platform introduced in [6], eliminates this self-occlusion and makes the full panoramic sphere available for planar view extraction and dynamic dataset generation. This is a structural prerequisite for the back-tofront mapping described in Section IV-E, which doubles the effective training data by reusing the rear hemisphere of each frame. In addition, the raw recordings contain synchronized inertial measurements (IMU) stored in metadata, which can be used to correct noisy labels and to improve robustness analysis in future work.

![](images/1fb2fd57d17f83d9577bdea2477ad6d324d3bdc96cbe22f80dd56a96cfdbb569.jpg)  
Fig. 2. Custom micro-drone platform with integrated 360<sup>◦</sup> camera used for dataset recording.

## B. Operational Concept

The learned yaw controller is designed as a recovery component: when the communication link to the operator is lost, the micro-drone uses its front-facing camera view to predict a yaw command that steers it toward open, traversable space. By maintaining controlled flight and retreating from communication-denied regions, the platform preserves reconnaissance data that would otherwise be lost with the vehicle. We emphasize that this yaw controller is an enabling component intended for integration with conservative safeguards (human-in-the-loop decision authority, geofencing, speed limits, and emergency stop behaviors) before operational deployment.

The learned yaw controller assumes that low-level state estimation and position hold are provided by a visualinterial odometry (VIO) pipeline such as OpenVINS or VINS-Fusion [18], [19]. These systems fuse IMU and camera data to maintain stable hover and altitude hold even in GPS-denied environments. Given this foundation, the drone executes a constant forward velocity while the CNN provides only the yaw correction command. This separation of concerns restricts the learning problem to directional guidance—a deliberate design choice that exploits the maturity of VIO-based stabilization and avoids conflating flight control with navigation learning.

## IV. DATASET AND PREPROCESSING PIPELINE

This section details the recorded scenarios, the stitching workflow, and the dynamic dataset generator that converts 360<sup>◦</sup> footage into planar image–label pairs for training.

## A. Recorded Scenarios

Recordings were performed in multiple industrial and indoor locations, including underground structures and training facilities, with the goal of representing environments relevant to emergency response operations. This aligns with prior work and practical deployments of aerial robots in firefighting and hazardous-material scenarios [20]. While actual postdisaster environments (collapsed structures, debris fields) were not accessible for this study, the recorded industrial and underground scenarios share key perceptual challenges with disaster-affected structures: near-complete darkness (underground mine), heavy soot deposits and high dynamic range lighting (fire training facility), confined and degraded structural elements (industrial heritage sites), and narrow metallic passages (bridge inspection). We acknowledge that true disaster scenes may present additional challenges such as dynamic smoke, irregular debris geometry, and moving obstacles. In some locations, flight operations were not legally permitted; therefore, additional sequences were recorded manually using a tripod setup.

![](images/562efc7e523675d404544a95f392be5f9a011b74a90a948136fd0582fa4798b3.jpg)  
Fig. 3. Stitching pipeline from dual-fisheye camera views to an equirectangular representation.

## B. Stitching and Alignment

Raw dual-fisheye recordings were stitched into equirectangular videos and trimmed to remove takeoff/landing segments. Because the camera is mounted with a fixed offset relative to the drone body, an additional yaw alignment is applied.

## C. Dataset Size and Splits

From the raw videos, we manually selected representative sequences and extracted every 10th frame, corresponding to an effective sampling rate of approximately 3 fps. This yielded the equirectangular frames summarized in Table I. Since the primary deployment target is the autonomous micro-drone, the test set is composed exclusively of drone recordings, while the manual recordings serve as training data to broaden scenario diversity and act as a form of data augmentation.

## D. Dynamic Dataset Generator

Instead of exporting a static set of planar images, we employ a dynamic dataset generator that produces planar input views and corresponding yaw labels at training time. By densely sampling planar crops from equirectangular images, this approach acts as a strong data multiplier while reducing storage overhead and enabling controlled view sampling and consistent augmentation.

We start from stitched equirectangular frames and extract planar projections that emulate a conventional forward-facing camera. To match the expected flight attitude during data collection, we fix the pitch to a level-flight configuration and restrict the 360<sup>◦</sup> sphere to a horizontal band (Fig. 4).

![](images/06ab9b7838b90b7a7e69c83285537f9bccfc4c5760a783ccc0301c2be57c8712.jpg)  
Fig. 4. Equirectangular frame with the horizontal band used for planar view extraction (fixed pitch).

Planar images are extracted with a field of view (FOV) of $9 0 ^ { \circ }$ and an initial resolution of 600 × 600 px. For training, we downsample to 101×101 px to match the baseline network [9] input size. An example of the geometric transformation is shown in Fig. 5. The yaw sampling regions are illustrated in Fig. 6.

![](images/b667d290ac882111be1c06988f50b794dc2dff482e267d35a8ac535d516e0ced.jpg)  
Fig. 5. Example of extracting a planar view from an equirectangular frame.

## E. On-the-Fly Pair Generation and Label Mapping

The ground-truth yaw label for each planar view is derived directly from its extraction angle within the equirectangular frame. By construction, the center of the equirectangular image at $\theta = 9 0 ^ { \circ }$ corresponds to the drone’s instantaneous heading (cf. Fig. 6), as ensured by the yaw alignment described in Section IV-C. During data collection, the drone was manually flown along recovery paths leading out of each scenario (e.g., toward the exit of a tunnel or corridor), with its yaw continuously aligned to the path direction [9]. The centerpoint of every equirectangular frame therefore coincides with the demonstrated recovery heading at that point in time. A planar view extracted at angle θ relative to this center thus directly encodes the yaw offset between an arbitrary viewing direction and the ground-truth recovery heading.

For each equirectangular source frame, the generator produces $N = 1 0$ planar views at yaw angles randomly sampled from the front range [30<sup>◦</sup>, 150<sup>◦</sup>]. The extraction angle $\theta _ { \mathrm { e x t r a c t } }$ directly serves as the ground-truth label: a view extracted at $\theta _ { \mathrm { e x t r a c t } } = 6 0 ^ { \circ }$ represents a perspective where the correct heading (90<sup>◦</sup>) lies $3 0 ^ { \circ }$ to the right of the image center, requiring a rightward yaw correction. Conversely, $\theta _ { \mathrm { e x t r a c t } } = 1 2 0 ^ { \circ }$ implies a 30<sup>◦</sup> leftward correction.

![](images/c04c4cdc6a92a73178a7e9e328fcd29a442a4c190efb52992104e6dfdb185a87.jpg)  
Fig. 6. Visualization of yaw angles in the equirectangular image with respect to the drone (top view). Green regions mark the used field of view, while red regions indicate ambiguous boundary views that can confuse the network during training.

For drone recordings, we additionally sample N views from the back range [210<sup>◦</sup>, 330<sup>◦</sup>] and map them onto the front range via $\theta _ { \mathrm { l a b e l } } = 3 6 0 ^ { \circ } - \theta _ { \mathrm { e x t r a c t } }$ . For example, a view extracted at $\theta _ { \mathrm { e x t r a c t } } = 2 1 0 ^ { \circ }$ is relabeled as $\theta _ { \mathrm { l a b e l } } = 1 5 0 ^ { \circ }$ , simulating the equivalent frontal perspective. This effectively doubles the number of training samples per frame while maintaining consistent label semantics.

Networks operate on a normalized output range. Let t be the yaw label in degrees after applying the back-to-front mapping, where $t \in [ 3 0 ^ { \circ } , 1 5 0 ^ { \circ } ]$ . We map it to the normalized range $y \in [ - 1 , 1 ]$ via

$$
y = 2 \cdot \frac { t - 3 0 ^ { \circ } } { 1 2 0 ^ { \circ } } - 1 .\tag{1}
$$

For reporting and real-world control, the inverse mapping is applied to transform the network output back to degrees.

## F. Image Normalization and Augmentation

Images are normalized to the range [−1, 1]. We apply mild affine augmentations (translation ±10%, rotation $\pm 1 5 ^ { \circ }$ scaling $\pm 1 0 \% )$ to increase robustness. These augmentations are applied after projection, so that they mimic realistic camera perturbations without changing the yaw label.

## V. LEARNING APPROACH

This section formalizes the yaw prediction task, describes the CNN architectures evaluated, and specifies the training procedure.

## A. Problem Formulation

We predict a continuous yaw command from a single planar front view. Let x denote a planar RGB image and $f _ { \theta } ( \cdot )$ a CNN regressor with parameters θ. The network outputs ${ \hat { y } } = f _ { \theta } ( x ) \in$ [−1, 1], which is linearly mapped back to the operational yaw range [30<sup>◦</sup>, 150<sup>◦</sup>]. Given a training set $\{ ( x _ { i } , y _ { i } ) \bar  \} _ { i = 1 } ^ { N }$ of planar images and normalized yaw labels $y _ { i } \in [ - 1 , 1 ]$ , we obtain θ by minimizing the mean squared error between predicted and ground-truth yaw:

TABLE I  
DATASET SUMMARY (AFTER PREPROCESSING)
<table><tr><td>Property</td><td>Value</td></tr><tr><td>Scenarios Raw videos (drone / manual)</td><td>10</td></tr><tr><td>Total duration</td><td>156 (46 / 110) 2 h 07 min 49 s</td></tr><tr><td>Equirectangular frames</td><td></td></tr><tr><td>Total Indoor / Outdoor</td><td>11,058 5,806 / 5,252</td></tr><tr><td>Drone / Manual</td><td>7,051  / 4,007</td></tr><tr><td>Splits</td><td></td></tr><tr><td>Train/Val (drone + manual)</td><td>9,953</td></tr><tr><td>Test (drone only)</td><td>1,105</td></tr><tr><td>Generated planar samples</td><td></td></tr><tr><td>From manual (10×)</td><td>40,070</td></tr><tr><td></td><td></td></tr><tr><td>From drone (20×)</td><td>141,020</td></tr><tr><td>Total</td><td>181,090</td></tr></table>

$$
\theta ^ { * } = \arg \operatorname* { m i n } _ { \theta } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( f _ { \theta } ( x _ { i } ) - y _ { i } \right) ^ { 2 } .\tag{2}
$$

## B. Architectures

We start from a compact CNN architecture used as a baseline in prior trail-following [9] work and adapt it from classification (three classes) to regression (single scalar output). We evaluate nine variants grouped into five design families: (i) tanh baseline regressor, (ii) ReLU activations, (iii) ReLU with dropout (several dropout rates), (iv) ReLU with batch normalization, (v) leaky-ReLU (several negative slopes).

## C. Training Procedure

We use supervised learning with the MSE loss defined in Section V-A. Optimization is done with stochastic gradient descent (SGD) with initial learning rate $5 \times 1 0 ^ { - 3 }$ and an exponential decay factor of 0.95 per epoch. Training runs for 100 epochs with a small batch size of 3; small-batch regimes can improve generalization and provide an implicit regularization effect [21]. The dynamically generated dataset is split into 90% training and 10% validation.

## VI. EXPERIMENTS AND RESULTS

We evaluate the proposed pipeline in three stages: a classification sanity check, a systematic regression comparison across architecture variants, and a semi-autonomous real-world flight test.

## A. Generator Validation via Classification

As a sanity check for the dataset generator, we first reproduced the three-class (left/straight/right) classification setup of Giusti et al. [9]. On the held-out test set, the best run achieved 0.9201 accuracy (mean over 10 runs: 0.9099). This indicates that the on-the-fly generation produces consistent training data and that the dataset supports learning of directionrelated visual cues.

## B. Regression Performance

For yaw regression, each model variant was trained 10 times with different random seeds. The best-performing variant on the held-out test set was a leaky-ReLU regressor (negative slope $\alpha = 0 . 0 5 )$ , achieving an $R ^ { 2 }$ of 0.8074. In the network’s internal range [−1, 1], the best model yields MSE 0.0649 and RMSE 0.2548. After mapping to degrees [30<sup>◦</sup>, 150<sup>◦</sup>], the same model achieves: 15.28<sup>◦</sup> RMSE and 10.146<sup>◦</sup> MAE.

![](images/ac50be59e7449bd7f2fe911e41d1433f24fd974688c2b0901e3add760f24bee5.jpg)  
Fig. 7. Predicted vs. target yaw values (degrees) with ideal diagonal reference.

The predicted-versus-target relationship in degrees is shown in Fig. 7, illustrating strong linear agreement across the operational range. The predictions stay centered on the ideal diagonal with no strong systematic bias, but the prediction error grows toward both ends of the range: when the drone faces a wall, the view does not reveal whether the wall lies to its left or right, making these yaw angles ambiguous and producing the heavier tails seen at the extremes. To reduce the resulting boundary outliers, we restricted the data sampler’s field of view to the unambiguous [30<sup>◦</sup>, 150<sup>◦</sup>] region (green in Fig. 6) and excluded the red zones near the walls from training.

## C. Ablation and Architecture Effects

Replacing tanh activations with ReLU consistently improved validation loss and test metrics [10]. In contrast, adding dropout in the tested configurations degraded performance, suggesting that the baseline architecture and dataset already provide substantial regularization and that additional dropout can lead to underfitting. Leaky-ReLU achieved the best overall test performance among the compared variants.

TABLE II  
BEST TEST-SET RESULTS ACROSS ARCHITECTURE VARIANTS (YAW REGRESSION)
<table><tr><td rowspan=1 colspan=1>Architecture</td><td rowspan=1 colspan=1>MSE</td><td rowspan=1 colspan=1>RMSE</td><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=1>R2</td></tr><tr><td rowspan=1 colspan=1>Tanh regressor (baseline)</td><td rowspan=1 colspan=1>0.0772</td><td rowspan=1 colspan=1>0.2778</td><td rowspan=1 colspan=1>0.1877</td><td rowspan=1 colspan=1>0.7710</td></tr><tr><td rowspan=1 colspan=1>ReLU regressor</td><td rowspan=1 colspan=1>0.0670</td><td rowspan=1 colspan=1>0.2588</td><td rowspan=1 colspan=1>0.1761</td><td rowspan=1 colspan=1>0.8013</td></tr><tr><td rowspan=1 colspan=1>ReLU + Dropout(p=0.2)</td><td rowspan=1 colspan=1>0.0843</td><td rowspan=1 colspan=1>0.2904</td><td rowspan=1 colspan=1>0.2082</td><td rowspan=1 colspan=1>0.7498</td></tr><tr><td rowspan=1 colspan=1>ReLU + Dropout(p=0.3)</td><td rowspan=1 colspan=1>0.1159</td><td rowspan=1 colspan=1>0.3405</td><td rowspan=1 colspan=1>0.2555</td><td rowspan=1 colspan=1>0.6560</td></tr><tr><td rowspan=1 colspan=1>ReLU + Dropout (p=0.5)</td><td rowspan=1 colspan=1>0.1483</td><td rowspan=1 colspan=1>0.3851</td><td rowspan=1 colspan=1>0.2922</td><td rowspan=1 colspan=1>0.5600</td></tr><tr><td rowspan=1 colspan=1>ReLU + Batch Normalization</td><td rowspan=1 colspan=1>0.0779</td><td rowspan=1 colspan=1>0.2790</td><td rowspan=1 colspan=1>0.1838</td><td rowspan=1 colspan=1>0.7690</td></tr><tr><td rowspan=1 colspan=1>Leaky-ReLU (α =0.01)</td><td rowspan=1 colspan=1>0.0655</td><td rowspan=1 colspan=1>0.2559</td><td rowspan=1 colspan=1>0.1716</td><td rowspan=1 colspan=1>0.8057</td></tr><tr><td rowspan=1 colspan=1>Leaky-ReLU (α = 0.05)</td><td rowspan=1 colspan=1>0.0649</td><td rowspan=1 colspan=1>0.2548</td><td rowspan=1 colspan=1>0.1691</td><td rowspan=1 colspan=1>0.8074</td></tr><tr><td rowspan=1 colspan=1>Leaky-ReLU (α=0.1)</td><td rowspan=1 colspan=1>0.0650</td><td rowspan=1 colspan=1>0.2549</td><td rowspan=1 colspan=1>0.1692</td><td rowspan=1 colspan=1>0.8072</td></tr></table>

## D. Error Analysis

We observed that prediction errors are not uniformly distributed across scenes. Fig. 8 summarizes representative predictions across some test scenarios. In most cases the predicted angle closely matches the target, resulting in correct, collision-free navigation, as in (a) and (b). In (d)–(g), the small discrepancies arise from inaccuracies in the target angles introduced by manual flight during dataset creation rather than from prediction error; in several of these cases the network prediction is in fact closer to the ideal flight direction than the labeled target, indicating that the model can compensate for mislabeled data.

Overall, the largest prediction errors stem from two distinct failure modes. The first is photometric: in (h), high-dynamicrange lighting and strong specular reflections lead the model to misinterpret bright regions as traversable openings, causing it to predict an alternative (though potentially still valid) escape direction. The second is geometric and is not caused by reflections or glare: in (c), the scene is texture-poor and contains multiple visually plausible escape directions of comparable openness, with no specular artifact present in the input view. Here the network commits to a different but still geometrically valid opening, reflecting the inherent multimodality of the underlying yaw distribution in such scenes. A single-frame regressor trained with an MSE objective cannot represent this multimodality and tends to collapse onto a single mode, which can produce large angular errors against a single groundtruth label even when the predicted direction would still be collision-free.

## E. Inference Benchmark

To evaluate suitability for robotics-relevant compute platforms, we measured inference throughput. A total of 1001 forward passes were executed, where the first pass was used for warm-up. The reported runtime corresponds to the average over the remaining 1000 runs. All benchmarks were conducted using both PyTorch and ONNX runtimes.

## F. Real-World Semi-Autonomous Test

A real-world semi-autonomous test was conducted using a DJI Tello EDU micro-drone. We emphasize that this test is intended as a policy validation rather than as a deployment demonstration: its purpose is to confirm that a network trained purely on dynamically generated planar crops of equirectangular footage produces actionable yaw commands on a previously unseen front-facing camera stream from a different platform. The custom recording drone introduced in Section III-A could not be used for this closed-loop test because it is deliberately designed as an invisible recording platform: its entire payload budget is allocated to the 360<sup>◦</sup> camera, leaving no capacity for an onboard flight computer such as a Jetson Orin or comparable embedded GPU. The platform’s ArduPilot flight controller exposes a MAVLink interface but has no onboard companion compute, no indoor position hold (GPS-only state estimation), and a proprietary analog FPV camera (Caddx Vista) that cannot be tapped for real-time inference. The DJI Tello EDU, by contrast, offers a documented Python SDK over Wi-Fi, an accessible frontcamera stream, and stable indoor hover, which made it the most direct way to close the loop and validate the learned policy under laboratory conditions. The benchmark platforms in Table III were selected because they represent the embedded compute classes typically integrated into micro-UAVs of the kind targeted by this work, and the measured throughput confirms that onboard inference is feasible on the hardware that future iterations of our platform will carry.

TABLE III  
INFERENCE BENCHMARK OVERVIEW (MODEL FORWARD PASS)
<table><tr><td rowspan=1 colspan=1>Hardware</td><td rowspan=1 colspan=1>Avg. Time (ms)</td><td rowspan=1 colspan=1>Throughput (Hz)</td></tr><tr><td rowspan=1 colspan=1>High-end workstation GPU</td><td rowspan=1 colspan=1>0.166–0.498</td><td rowspan=1 colspan=1>2008-6024</td></tr><tr><td rowspan=1 colspan=1>Embedded GPU (Jetson Orin)</td><td rowspan=1 colspan=1>1.486-1.528</td><td rowspan=1 colspan=1>654-673</td></tr><tr><td rowspan=1 colspan=1>Low-power CPU (RPi 4 class)</td><td rowspan=1 colspan=1>45-112</td><td rowspan=1 colspan=1>9-22</td></tr></table>

The front-camera stream of the Tello was sent via Wi-Fi to an external computer running the trained network, and the predicted yaw command was sent back as a control input while forward motion was kept constant (Fig. 9). A previously unseen indoor lab environment was successfully traversed, including passing through a doorway without collision.

## VII. CONCLUSION AND FUTURE WORK

The primary contribution of this paper is the acquisition of a novel real-world dataset, coupled with the implementation of an initial learning pipeline. This pipeline is designed to predict yaw commands from planar images derived from 360<sup>◦</sup> recordings. Furthermore, we emphasize that data from open environments is significantly more valuable—and challenging to obtain—than data from restricted settings, such as industrial facilities or racing tracks. The best model reached 15.28<sup>◦</sup> RMSE and 10.146<sup>◦</sup> MAE on held-out test data and demonstrated feasibility in a semi-autonomous real-world test. Our results suggest that camera-only yaw control is feasible on micro-drones when trained on representative, scenario diverse data. Replacing tanh with ReLU improved regression performance, while adding dropout degraded results, indicating possible over-regularization. Leaky-ReLU yielded the strongest overall test performance. The dominant practical limitation is robustness to illumination extremes and specular reflections. A further structural limitation is that the singleframe reactive policy has no temporal memory: without a history of past observations, the controller cannot distinguish previously visited dead ends from unexplored corridors and may revisit regions that only appear open from a single viewpoint. Given the safety-critical nature of autonomous flight, future systems should combine learned yaw prediction with additional safeguards such as uncertainty estimation, conservative fallback behaviors, or complementary sensing. Needless to say a lot of work remains to be done. We already build a more advanced small drone with four 180<sup>◦</sup> fisheye cameras which will be presented in an upcoming paper. Predicting more than the yaw angle is mandatory. Future work includes: (i) increasing dataset diversity and size, (ii) targeted augmentation for brightness/contrast and glare, (iii) using additional sensor signals (e.g., IMU) to correct potentially noisy labels, (iv) transfer learning using modern pretrained vision transformer backbones, (v) investigating domain adaptation strategies, including fine-tuning on small post-disaster datasets and synthetic augmentation of visibility-degrading conditions (smoke, dust, water spray), to assess generalization beyond the training distribution. Both, the dataset<sup>1</sup> and the source code<sup>2</sup> for the training are public available.

![](images/beef903418d830ca73ccd11fcd3eb2997544c48d30195364a7bda7038ce5a798.jpg)  
Fig. 8. Qualitative examples on the held-out test set: ground-truth yaw (green) vs. predicted yaw (red) overlaid on planar projections. More are available on GitHub<sup>2</sup>.

![](images/e7f84d432e54ef2080052224740f8a5573846ecf8ae2bc5928100c594aa1fd1a.jpg)  
Fig. 9. Offboard semi-autonomous yaw-control loop used in the real-world validation. The DJI Tello EDU streams front-camera images to an external server over Wi-Fi. The server performs CNN-based yaw prediction and sends yaw steering commands back to the drone, while forward velocity is kept constant.

## ACKNOWLEDGMENT

This work is funded by the Federal Ministry of Research, Technology and Space (BMFTR) under grant, 13N16478 (E-DRZ), cf. https://rettungsrobotik.de.

We further acknowledge the following institutions for granting access to their facilities for data collection: Deutsches Bergbau-Museum Bochum, Feuerwehr Dortmund 37 5 (Ausbildungszentrum), LWL-Museum Henrichshutte Hattingen,¨ LWL-Museum Zeche Nachtigall Witten, LWL-Museum Zeche Zollern Dortmund, MS Kartcenter Hattingen, Straßen.NRW (Niederrheinbrucke Wesel), and Trainingsbergwerk Reckling-¨ hausen.

## REFERENCES

[1] R. R. Murphy, Disaster Robotics, ser. Intelligent Robotics and Autonomous Agents. Cambridge, MA: MIT Press, 2014.

[2] J. Carlson, R. Murphy, and A. Nelson, “Follow-up analysis of mobile robot failures,” Proceedings - IEEE International Conference on Robotics and Automation, 04 2004.

[3] H. Surmann, K. Daun, M. Schnaubelt, O. Von Stryk, M. Patchou, S. Bocker, C. Wietfeld, J. Quenzel, D. Schleich, S. Behnke, R. Grafe,¨ N. Heidemann, D. Slomma, and I. Kruijff-Korbayova, “Lessons from robot-assisted disaster response deployments by the german rescue robotics center task force,” Journal of Field Robotics, vol. 41, pp. 782– 797, 12 2023.

[4] H. Surmann, D. Slomma, S. Grobelny, and R. Grafe. (2021, 11) Deployment of aerial robots after a major fire of an industrial hall with hazardous substances, a report.

[5] A. Santamaria-Navarro, R. Thakker, D. D. Fan, B. Morrell, and A. Agha-mohammadi, “Towards resilient autonomous navigation of drones,” CoRR, vol. abs/2008.09679, 2020. [Online]. Available: https://arxiv.org/abs/2008.09679

[6] H. Surmann, N. Digakis, J.-N. Kremer, J. Meine, M. Schulte, and N. Voigt, “Redefining recon: Bridging gaps with uavs, 360° cameras, and neural radiance fields,” in Proceedings of the IEEE International Symposium on Safety, Security, and Rescue Robotics (SSRR), 11 2023, pp. 13–18.

[7] Y. Ren, F. Zhu, G. Lu, Y. Cai, L. Yin, F. Kong, J. Lin, N. Chen, and F. Zhang, “Safety-assured high-speed navigation for mavs,” Science Robotics, vol. 10, no. 98, p. eado6187, 2025.

[8] P. Liu, C. Feng, Y. Xu, Y. Ning, H. Xu, and S. Shen, “Omninxt: A fully open-source and compact aerial robot with omnidirectional visual perception,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 10 605–10 612.

[9] A. Giusti, J. Guzzi, D. C. Cires¸an, F.-L. He, J. P. Rodr´ıguez, F. Fontana, M. Faessler, C. Forster, J. Schmidhuber, G. D. Caro, D. Scaramuzza, and L. M. Gambardella, “A machine learning approach to visual perception of forest trails for mobile robots,” IEEE Robotics and Automation Letters, vol. 1, no. 2, pp. 661–667, 2016.

[10] B. Pearson and T. Breckon, “An optimised deep neural network approach for forest trail navigation for uav operation within the forest canopy,” in The UK-RAS Network Conference on Robotics and Autonomous Systems, 12 2017.

[11] M. Samy, K. Amer, M. Shaker, and M. ElHelw, “Drone path-following in gps-denied environments using convolutional networks,” 2019. [Online]. Available: https://arxiv.org/abs/1905.01658

[12] M. Bojarski, D. D. Testa, D. Dworakowski, B. Firner, B. Flepp, P. Goyal, L. D. Jackel, M. Monfort, U. Muller, J. Zhang, X. Zhang, J. Zhao, and K. Zieba, “End to end learning for selfdriving cars,” CoRR, vol. abs/1604.07316, 2016. [Online]. Available: http://arxiv.org/abs/1604.07316

[13] A. Loquercio, A. I. Maqueda, C. R. Del-Blanco, and D. Scaramuzza, “DroNet: Learning to fly by driving,” IEEE Robotics and Automation Letters, vol. 3, no. 2, pp. 1088–1095, 2018.

[14] N. Smolyanskiy, A. Kamenev, J. Smith, and S. Birchfield, “Toward lowflying autonomous MAV trail navigation using deep neural networks for environmental awareness,” in 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2017, pp. 4241–4247.

[15] A. Loquercio, E. Kaufmann, R. Ranftl, M. Muller, V. Koltun, and¨ D. Scaramuzza, “Learning high-speed flight in the wild,” Science Robotics, vol. 6, no. 59, p. eabg5810, 2021.

[16] M. Kulkarni and K. Alexis, “Reinforcement learning for collisionfree flight exploiting deep collision encoding,” in IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 15 781– 15 788.

[17] V. H. Dang, A. Redder, H. X. Pham, A. Sarabakha, and E. Kayacan, “VDS-Nav: Volumetric depth-based safe navigation for aerial robots— bridging the sim-to-real gap,” IEEE Robotics and Automation Letters, vol. 10, no. 10, pp. 11 038–11 045, 2025.

[18] P. Geneva, K. Eckenhoff, W. Lee, Y. Yang, and G. Huang, “Openvins: A research platform for visual-inertial estimation,” in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 05 2020.

[19] T. Qin, S. Cao, J. Pan, and S. Shen, “A general optimization-based framework for global pose estimation with multiple sensors,” CoRR, vol. abs/1901.03642, 2019. [Online]. Available: http://arxiv.org/abs/ 1901.03642

[20] I. Kruijff-Korbayova, R. Grafe, N. Heidemann, A. Berrang, C. Hussung,´ C. Willms, P. Fettke, M. Beul, J. Quenzel, D. Schleich, S. Behnke, J. Tiemann, J. Guldenring, M. Patchou, C. Arendt, C. Wietfeld, K. Daun,¨ M. Schnaubelt, O. von Stryk, A. Lel, A. Miller, C. Rohrig, T. Straßmann,¨ T. Barz, S. Soltau, F. Kremer, S. Rilling, R. Haseloff, S. Grobelny, A. Leinweber, G. Senkowski, M. Thurow, D. Slomma, and H. Surmann, “German rescue robotics center (drz): A holistic approach for robotic systems assisting in emergency response,” in 2021 IEEE International Symposium on Safety, Security, and Rescue Robotics (SSRR), 2021, pp. 138–145.

[21] D. Masters and C. Luschi, “Revisiting small batch training for deep neural networks,” 2018. [Online]. Available: https://arxiv.org/abs/1804. 07612