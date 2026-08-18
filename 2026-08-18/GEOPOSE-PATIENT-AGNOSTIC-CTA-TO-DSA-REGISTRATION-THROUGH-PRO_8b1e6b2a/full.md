# GEOPOSE: PATIENT-AGNOSTIC CTA-TO-DSA REGISTRATION THROUGH PROJECTION-SPACE CALIBRATION

PREPRINT

Rudolf L. M. van Herten Weill Cornell Medicine, Cornell Tech New York, NY 10044 rlv4001@med.cornell.edu

Robert Graf Technical University of Munich 81675 Munich robert.graf@tum.de

Johannes C. Paetzold Weill Cornell Medicine, Cornell Tech New York, NY 10044 jpaetzold@med.cornell.edu

Paula Feldman Weill Cornell Medicine, Cornell Tech New York, NY 10044 paf4004@med.cornell.edu

August 18, 2026

## ABSTRACT

Aligning intraoperative biplanar digital subtraction angiography (DSA) to pre-procedural computed tomography angiography (CTA) requires rapid and accurate 3D-to-2D registration. Optimizationbased methods are sensitive to initialization and may require hundreds of iterations, whereas learningbased approaches commonly rely on patient-specific training. We propose GeoPose, a populationtrained framework that estimates the C-arm pose in a learned canonical frame and transfers it to the native frame of an unseen CTA through projection-space calibration and transform composition. A population-trained residual network refines the pose, followed optionally by low-budget image driven optimization. GeoPose requires neither patient-specific adaptation nor explicit inter-volume preregistration. On 80 DSA observations from 20 held-out patients, optimization-free GeoPose achieved a carotid mean projected centerline distance (mPCD) of 5.8 mm and a clDice of 0.45, compared with 14.5 mm and 0.28 for the best-performing baseline, while requiring only 0.15 s. After 25 optimization iterations, GeoPose reached an mPCD of 4.6 mm and a clDice of 0.58 in approximately two seconds. Under the same budget, native-initialized optimization achieved 14.6 mm and 0.15, respectively. GeoPose thus provides rapid native-frame registration with fixed populationlevel weights and the geometric correspondence required for downstream biplanar 3D vascular reconstruction. Code is available on GitHub.

Keywords Computed tomography angiography · domain generalization · digital subtraction angiography · image registration · pose estimation

## 1 Introduction

Acute ischemic stroke is a neurological emergency in which treatment delay translates directly into irreversible cerebral tissue loss [1]. This is particularly consequential for large-vessel occlusions, which impact a large downstream vascular network that perfuses cerebrovascular tissue. Endovascular thrombectomy (EVT) substantially improves functional outcomes in this setting [2], but its benefit declines rapidly with delayed reperfusion [3]. Consequently, any computational assistance introduced into the clinical workflow must provide actionable information within a strict time window and without adding patient-specific preparation that could delay treatment.

![](images/932bb5dbf4f9933cd31c502276b01e31dc8343eda87671b052bf10a8776bc1e6.jpg)  
Figure 1: Overview of the proposed CTA-to-DSA registration pipeline. (a) Pose estimation: GeoPose-Init regresses a canonical-frame transform from a DSA maximum intensity projection $( \mathbf { T } _ { \mathrm { D S A  c a n } } )$ and a calibration transform from a canonical isopose DRR of the patient CTA $( \mathbf { T } _ { \mathrm { n a t  c a n } } ) .$ . (b) Projection-space calibration: their composition maps the prediction into the patient’s native volume frame, absorbing template misalignment and providing an initial pose estimate $\mathbf { T } _ { \mathrm { D S A } }$ . (c) Pose refinement: GeoPose-Refine applies up to $N _ { \mathrm { r e f } }$ image similarity-gated residual updates $\mathbf { T } \delta \mathbf { T } ^ { - 1 }$ from re-rendered DRRs to obtain the refined pose $\mathbf { T } _ { \mathrm { D S A } } ^ { \dot { R } }$ , which is subsequently test-time optimized by GeoReg to the final pose $\mathbf { T } _ { \mathrm { D S A } } ^ { f }$ . The recovered native-frame pose establishes spatial correspondence between the pre-procedural CTA and intraoperative DSA, enabling CTA-based anatomical roadmapping in the acquisition views and providing the geometry required for downstream biplanar 3D vascular reconstruction.

EVT follows pre-procedural CT angiography (CTA), which provides three-dimensional anatomical and vascular context, and is directly aided by intraoperative biplanar digital subtraction angiography (DSA) [4]. DSA serves as the gold standard for intraoperative guidance for its ability to capture high-resolution vascular dynamics [5–7]. Each DSA image is a two-dimensional projection with no direct spatial correspondence to the CTA. Estimating the posteroanterior (PA) and lateral (LAT) C-arm poses would therefore establish correspondence, allowing CTA-derived 3D segmentations and treatment targets to be projected onto the acquired DSA views. Furthermore, the recovered poses may provide a geometric basis for 3D vascular reconstruction from biplanar images.

Most 3D-to-2D registration methods estimate these poses by comparing observed X-rays with digitally reconstructed radiographs (DRRs) rendered from the CTA volume. Differentiable X-ray rendering has made it possible to optimize this alignment directly using gradient descent-based methods [8, 9]. Nevertheless, the optimization objective is highly non-convex: registration remains sensitive to initialization, and recovering from a broad pose distribution may require hundreds of rendering and optimization steps [10]. Learning-based pose estimators can improve the capture range by amortizing pose initialization. For example, xvr combines population-level pretraining with patient-specific adaptation, reducing the required fine-tuning to approximately five minutes [11]. LXPose instead replaces test-time refinement with a cascaded pose-regression network and achieves real-time inference, at the cost of requiring a separate model for every target volume [12]. Although effective in planned interventions, patient-specific network training is undesirable in acute stroke, where even short additional processing stages compete with an already time-critical treatment pathway.

GeoReg addresses a complementary limitation of cerebrovascular registration by aligning DSA and CTA without requiring vascular segmentation [13]. GeoReg derives temporal maximum-intensity projection (MAP) X-ray silhouettes directly from DSA sequences and jointly optimizes their alignment with CTA-derived DRRs under a soft biplanar geometric constraint. This formulation avoids the need to extract corresponding vascular structures across modalities, but its practical speed and robustness remain governed by the initial pose. A default pose initialization may require hundreds of test-time optimization steps and does not account for the coordinate-frame differences between the learned population and a new patient’s native, unregistered CTA.

To address these limitations, we introduce GeoPose, a population-trained framework for rapid CTA to biplanar DSA registration. GeoPose estimates the C-arm pose of each DSA-derived MAP in a shared canonical frame, where template alignment gives pose a consistent meaning across patients by learning a pose–content decomposition [14]. A render-and-compose calibration then transfers these predictions to the native frame of an unseen CTA. This allows the trained network to be applied with fixed weights, without patient-specific adaptation or explicit 3D preregistration. Our contributions are as follows:

1. Compositional frame transfer without preregistration. We introduce a render-and-compose calibration that estimates the rigid transform between a patient’s native CTA coordinate frame and the learned template frame from a single synthetic projection and one network forward pass. This enables a population-trained pose estimator to operate directly on native, unregistered CTA volumes without per-patient network training or explicit inter-volume 3D preregistration.

2. Population-trained, image-gated pose refinement. We develop a residual refinement module shared across patients that predicts pose corrections from paired DSA MAP and patient-CTA renderings. An image-similarity gate retains only corrections that improve the observed alignment, providing a fast optimization-free operating point while limiting harmful recursive updates.

3. Rapid test-time registration. We combine GeoPose initialization with low-budget NAdam optimization [15] of the GeoReg objective. The resulting framework preserves image-driven test-time finetuning while reducing the optimization budget from 400 to 25 iterations, enabling a registration timeframe of ∼2 seconds.

4. Experimental validation. On a held-out test set of 80 DSA acquisitions, GeoPose achieves an mPCD of 5.8 mm without test-time optimization while using fixed population-level weights, compared with 14.5 mm for the best-performing baseline.

## 2 Method

GeoPose uses two population-trained networks with fixed test-time weights: GeoPose-Init predicts a C-arm pose, and GeoPose-Refine predicts a residual pose correction. Render-and-compose calibration maps the canonical-frame prediction to the native CTA frame, and similarity-gated refinement subsequently initializes a short GeoReg optimization. An overview of the method is presented in Fig. 1.

## 2.1 Direct CTA-to-DSA registration

Let $V ^ { \mathrm { n a t } } : \mathbb { R } ^ { 3 } $ R denote a three-dimensional CTA volume represented in its native coordinate frame. For each view $v \in \ \{ \mathrm { L A T ^ { - } , P A , L A T ^ { + } } \}$ , a DSA sequence consists of a set of temporal images $\{ I _ { \mathrm { D S A } } ( t ) \} _ { t = 1 } ^ { T }$ , where T denotes the number of acquisition frames. As in GeoReg [13], a maximum-intensity projection over the temporal window is calculated to bridge the CTA-to-DSA domain gap:

$$
I _ { \mathrm { M A P } } ( \mathbf { p } ) = \mathcal { M } ( \mathbf { p } ) \operatorname* { m a x } _ { t \in \{ 1 , . . . , T \} } I _ { \mathrm { D S A } } ( \mathbf { p } , t ) ,\tag{1}
$$

where p denotes a detector pixel and M is a coarse mask defining the valid cranium region. This temporal maximumintensity projection recovers a silhouette of the subtracted X-ray sequence.

A C-arm pose is represented by a rigid camera-to-volume transformation $\mathbf { T } _ { v } \in \mathrm { S E } ( 3 )$ . Given CTA volume $V ^ { \mathrm { n a t } }$ , pose $\mathbf { T } _ { v } .$ , and camera intrinsic matrix $\mathbf { K } _ { v } ,$ a DRR is generated by a differentiable rendering operator [8]:

$$
I _ { \mathrm { D R R } , v } = \mathcal { R } \left( V ^ { \mathrm { n a t } } , \mathbf { T } _ { v } ; \mathbf { K } _ { v } \right) ,\tag{2}
$$

where $\mathcal { R } ( \cdot )$ integrates the CTA intensities along rays cast from the virtual X-ray source toward the detector [16]. The objective is to estimate a pose $\mathbf { T } _ { v }$ for which $I _ { \mathrm { D R R } , v }$ aligns with the corresponding DSA MAP $I _ { \mathrm { M A P } }$

## 2.2 Population-trained pose estimation

![](images/8ba9f5b3356f8d5a9d21ca91b6c9a13e6e30b00afb3e8f664056177ca41e84b1.jpg)

![](images/0e238af6b93027dd72f4d4a26540a5df4bad7d9231d8ac86a89004265b0c1834.jpg)  
Figure 2: Native CTA (top) and canonical template-aligned (bottom) renders at the fixed posteroanterior isopose for four subjects from the ISLES’24 dataset. Aligning all training volumes to a shared frame (left) makes pose estimation a generalized function across subjects rather than per-patient regression.

Training 3D pose estimation exclusively on a population implies pose–content decomposition [14]. Since every training volume is rendered in a shared template frame over a common pose distribution, the estimator absorbs patient-specific anatomy while reading out pose in a coordinate frame whose axes carry the same meaning for every volume. Pose estimation thus becomes a query within a canonical frame rather than a per-patient regression problem (see Fig. 2), permitting the fixed-weight networks to transfer to an unseen CTA through the frame-transfer calibration described in Section 2.3.

This decomposition is maintained through two population-trained networks: GeoPose-Init and GeoPose-Refine. GeoPose-Init directly regresses a six-degree-of-freedom C-arm pose from a MAP-like

image, while GeoPose-Refine regresses a residual pose correction from a pair of MAP-like and rendered images, a cascaded refinement strategy inspired by LXPose [12]. Both networks are optimized exclusively on synthetic renderings of a training population. Fig. 3 provides an overview of the training procedure for both networks.

## 2.2.1 GeoPose-Init

GeoPose-Init represents each C-arm pose as a residual relative to one of three view-dependent isoposes in a shared canonical coordinate frame. Let v denote the view class defined in Section 2.1, where $\mathrm { L A } \dot { \mathrm { T } } ^ { - }$ and $\mathrm { L A } \dot { \mathrm { T } } ^ { + }$ distinguish the two opposing lateral C-arm orientations. Each class is assigned a fixed canonical rotation anchor:

$$
\begin{array} { r } { \bar { \mathbf { r } } _ { v } = \left\{ \begin{array} { l l } { ( - \frac { \pi } { 2 } , 0 , 0 ) , } & { v = \mathrm { L A T } ^ { - } , } \\ { ( 0 , 0 , 0 ) , } & { v = \mathrm { P A } , } \\ { ( + \frac { \pi } { 2 } , 0 , 0 ) , } & { v = \mathrm { L A T } ^ { + } . } \end{array} \right. } \end{array}\tag{3}
$$

All three anchors share the isocentric translation $\bar { \mathbf { t } } = ( 0 , 6 5 0 , 0 ) ^ { \top }$ mm. Together, $\overline { { { \bf r } } } _ { v }$ and t define the view-dependent isopose $\mathbf { T } _ { \mathrm { i s o } , v }$ . GeoPose-Init therefore estimates a pose by predicting (i) the view class and (ii) a six-degree-of-freedom residual relative to the corresponding isopose.

Given an input projection $I ,$ an image-only classification head predicts logits for the three view classes and selects vb. A pose-regression head separately predicts a rotational residual $\Delta \mathbf { r }$ and a translational residual $\Delta \mathbf { t }$ . Both heads operate on the shared image representation

$$
\mathbf { h } = f _ { \theta } ( I ) ,\tag{4}
$$

produced by a one-channel ResNet encoder [17]. The pose-regression head is additionally conditioned on a learned embedding of the selected view class:

$$
( \Delta \mathbf { r } , \Delta \mathbf { t } ) = g _ { \theta } ( [ \mathbf { h } , \mathbf { e } _ { \widetilde { v } } ] ) ,\tag{5}
$$

where $\widetilde { v } = v$ during training and $\widetilde { v } = \widehat { v }$ at inference. Thus, the known view class of a synthetic training projection conditions the regression head, whereas an unseen input is processed using the class predicted from its image appearance. The classification head does not receive the view embedding.

The selected rotation anchor and the predicted residuals define the estimated pose:

$$
\widehat { \mathbf { r } } = \overline { { \mathbf { r } } } _ { \widetilde { v } } + \Delta \mathbf { r } , \qquad \widehat { \mathbf { t } } = \overline { { \mathbf { t } } } + \Delta \mathbf { t } .\tag{6}
$$

The rotation is composed in Euler space, and $( \widehat { \mathbf { r } } , \widehat { \mathbf { t } } )$ is used to construct the predicted rigid pose transform $\widehat { \mathbf { T } } _ { v } \in \mathrm { S E } ( 3 )$ This view-anchored parameterization restricts the regression target to a residual about the nearby C-arm orientation rather than requiring the network to regress over the full pose space.

To train GeoPose-Init across patients, each training CTA is rigidly aligned to the shared canonical coordinate frame. For patient $i ,$ a pose $\mathbf { T } _ { v } \in \bar { \mathrm { S E } } ( 3 )$ is sampled around one of the three view-dependent isoposes and used to render a synthetic projection:

$$
I _ { \mathrm { D R R } } = \mathcal { R } \left( V _ { i } ^ { \mathrm { c a n } } , { \bf T } _ { v } ; { \bf K } _ { 0 } \right) ,\tag{7}
$$

where $V _ { i } ^ { \mathrm { c a n } }$ denotes the canonical-frame CTA and ${ \bf K } _ { 0 }$ denotes the fixed projection geometry used during training.   
Because the pose and its associated isopose are sampled directly, the view class v is known for every synthetic projection.

Given the resulting pose prediction $\widehat { \mathbf { T } } _ { v } .$ , a second DRR is rendered at the predicted pose:

$$
\widehat { I } _ { \mathrm { D R R } } = \mathcal { R } \left( V _ { i } ^ { \mathrm { c a n } } , \widehat { \mathbf { T } } _ { v } ; \mathbf { K } _ { 0 } \right) .\tag{8}
$$

The network is then trained by minimizing

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { i n i t } } = \mathrm { ~ - ~ } \operatorname { m N C C } ( I _ { \mathrm { D R R } } , \widehat { I } _ { \mathrm { D R R } } ) + \lambda _ { \mathrm { g e o } } d _ { \mathrm { g e o } } \left( \mathbf { T } _ { v } , \widehat { \mathbf { T } } _ { v } \right) } \\ & { ~ + ~ \lambda _ { \mathrm { a r t } } \mathcal { L } _ { \mathrm { D i c e } + \mathrm { m P D } } \left( M , \widehat { M } \right) + \lambda _ { \mathrm { v i e w } } \mathcal { L } _ { \mathrm { C E } } \left( v , \widehat { v } \right) . } \end{array}\tag{9}
$$

Here, mNCC denotes multiscale normalized cross-correlation, and $d _ { \mathrm { g e o } }$ combines rotation and translation error in a common physical coordinate system [9]. The masks $M$ and $\widehat { M }$ denote the projected vascular occupancy at the sampled and predicted poses, respectively. $\mathcal { L } _ { \mathrm { D i c e + m P D } }$ combines a Dice term [18] with a projection-distance penalty [12] to promote vascular overlap, while $\mathcal { L } _ { \mathrm { C E } }$ is the cross-entropy loss used to train the view classifier. After training on synthetic projections from the canonicalized population, GeoPose-Init is applied with fixed weights to both CTA-derived DRRs and real DSA maximum-intensity projections.

![](images/1521085db232bf3ca71061f4bf59307c70e97f77ae15103960a652e6b019a4f4.jpg)

![](images/5cc56b1e1f2e8df9ad0910eb6ba126605d03638184fdec88cff740275b717a49.jpg)  
Figure 3: GeoPose training overview. (a) GeoPose-Init: a training CTA is rendered at a sampled pose $\mathbf { T } _ { v }$ and intensityaugmented to resemble a DSA MAP image, and subsequently passed through the encoder ${ \mathcal { E } } _ { \mathrm { i n i t } }$ . The network both classifies the originating view v and regresses a residual $\Delta \widehat { \mathbf { T } } _ { v }$ about the view-dependent isopose $\mathbf { T } _ { \mathrm { i s o } , v } .$ . At test time, the predicted view class vb is used to derive the isopose. The right panel shows the isopose anchors $\mathbf { T } _ { \mathrm { i s o } , v }$ (yellow) and the distribution of sampled training poses $\mathbf { T } _ { v }$ (blue) for the three view classes $( v = \mathrm { L A T ^ { - } , P A , L A T ^ { + } } )$ relative to the cranium. (b) GeoPose-Refine: a clean DRR rendered at a perturbed pose $\mathbf { T } _ { \mathrm { n o i s y } }$ and an appearance-augmented, MAP-like DRR rendered at the optimal pose $\mathbf { T } ^ { * }$ are passed through a shared encoder $\dot { \mathcal { E } } _ { \mathrm { r e f } }$ to obtain pooled embedding $\mathbf { e } _ { M }$ and $\mathbf { e } _ { D }$ . Their fused comparative embedding z is used to regress a residual transformation $\delta \hat { \mathbf { T } } _ { v }$ that maps the perturbed pose back toward $\hat { \mathbf { T } } ^ { * }$ . The right panel illustrates the geometric relationship between $\mathbf { T } ^ { * } , \mathbf { T } _ { \mathrm { n o i s y } }$ , and the sampled perturbation δT. Both models are trained using a combination of image similarity, vascular overlap, and geodesic losses.

## 2.2.2 GeoPose-Refine

To further improve a pose estimate without invoking gradient-based optimization, a second population-trained network is introduced that predicts a residual rigid transformation from the discrepancy between a MAP-like image and a rendering of the current pose estimate. A secondary ResNet encoder, shared between the two inputs, processes both images individually, resulting in pooled feature representations $\mathbf { e } _ { M }$ and $\mathbf { e } _ { D }$ . A comparative embedding is formed as

$$
\mathbf { z } = \left[ \mathbf { e } _ { M } , \mathbf { e } _ { D } , \mathbf { e } _ { M } - \mathbf { e } _ { D } , \left| \mathbf { e } _ { M } - \mathbf { e } _ { D } \right| , \mathbf { e } _ { M } \odot \mathbf { e } _ { D } \right] ,\tag{10}
$$

where ⊙ denotes element-wise multiplication. After linear fusion, z is concatenated with a learned embedding of the view v. From this, the network predicts an axis-angle rotation δbr and translation $\delta \widehat { \mathbf { t } } .$ , which together define a residual transformation $\delta \widehat { \mathbf { T } } \in \mathrm { S E } ( 3 )$ (3).

The refiner is trained using synthetic pairs where, for an optimal pose $\mathbf { T } ^ { * }$ , a perturbation δT is sampled which constructs

$$
\mathbf { T } _ { \mathrm { n o i s y } } = \mathbf { T } ^ { * } \delta \mathbf { T } .\tag{11}
$$

For each synthetic pair, the first network input is a DRR rendered at $\mathbf { T } ^ { * }$ and augmented with appearance corruptions to emulate a MAP-like image, whereas the second input is a clean DRR rendered at $\mathbf { T } _ { \mathrm { n o i s y } }$ . Thus, both inputs are

synthetically generated during training, while retaining the appearance discrepancy that the refiner must resolve at inference. Since δT maps the optimal pose to the perturbed pose, the corrected prediction is obtained by composing the inverse residual:

$$
\widehat { \mathbf { T } } ^ { * } = \mathbf { T } _ { \mathrm { n o i s y } } \delta \widehat { \mathbf { T } } ^ { - 1 } .\tag{12}
$$

The refinement module training objective follows Eq. 9, omitting the view classification term:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { r e f } } = - \operatorname { m N C C } \left( I _ { \mathrm { D R R } } ^ { * } , \widehat { I } _ { \mathrm { D R R } } ^ { * } \right) + \lambda _ { \mathrm { g e o } } d _ { \mathrm { g e o } } \left( \mathbf { T } ^ { * } , \widehat { \mathbf { T } } ^ { * } \right) } \\ & { ~ + \lambda _ { \mathrm { a r t } } \mathcal { L } _ { \mathrm { D i c e + m P D } } \left( M ^ { * } , \widehat { M } ^ { * } \right) . } \end{array}\tag{13}
$$

The shared encoder is initialized from the trained GeoPose-Init model and subsequently fine-tuned jointly with the newly initialized fusion, view-embedding, and residual-prediction layers on synthetic pose perturbations. At inference, all parameters remain fixed and the refiner is applied without gradient-based optimization or patient-specific adaptation.

## 2.3 Test-time inference

GeoPose-Init and GeoPose-Refine are trained across a training population in a predetermined canonical framework. As such, applying them to an unregistered patient CTA requires transferring predictions from the learned canonical frame into the patient’s native CTA frame, deciding when a proposed refinement should be applied, and, optionally, adapting the resulting pose to the observed DSA pair through direct image similarity-driven optimization.

## 2.3.1 Projection-space calibration

GeoPose is trained to predict poses relative to a shared canonical coordinate frame. At inference, however, the estimated C-arm pose must be expressed relative to the native coordinate frame of the patient CTA. Explicitly registering each new CTA to the training template would resolve this discrepancy, but would introduce a separate 3D registration problem into the time-critical workflow. Instead, we estimate the required frame transfer from a single synthetic projection of the native CTA.

Let $\mathbf { A } \in \mathrm { S E } ( 3 )$ denote the transformation from canonical to native volume coordinates. Since DiffDRR represents poses as camera-to-volume transformations [8], the equivalent pose in the canonical volume frame is obtained by left-composition with ${ { \bf A } ^ { - 1 } }$ . Consequently, the same projection satisfies

$$
\mathcal { R } \left( V ^ { \mathrm { n a t } } , { \bf T } ; { \bf K } \right) = \mathcal { R } \left( V ^ { \mathrm { c a n } } , { \bf A } ^ { - 1 } { \bf T } ; { \bf K } \right) .\tag{14}
$$

We exploit this identity by rendering the native CTA at a known isocentric PA calibration pose $\mathbf { T } _ { \mathrm { c a l } } ^ { \mathrm { n a t } }$ :

$$
{ \cal I } _ { \mathrm { c a l } } = { \mathcal R } \left( V ^ { \mathrm { n a t } } , { \bf T } _ { \mathrm { c a l } } ^ { \mathrm { n a t } } ; { \bf K } _ { 0 } \right) .\tag{15}
$$

Although $I _ { \mathrm { c a l } }$ is rendered at $\mathbf { T } _ { \mathrm { c a l } } ^ { \mathrm { n a t } }$ in the native frame, the same projection corresponds to $\mathbf { A } ^ { - 1 } \mathbf { T } _ { \mathrm { c a l } } ^ { \mathrm { n a t } }$ in the canonical frame. Processing it with GeoPose-Init therefore gives

$$
\widehat { \mathbf { T } } _ { \mathrm { c a l } } ^ { \mathrm { c a n } } = g _ { \theta } \left( \left[ f _ { \theta } ( I _ { \mathrm { c a l } } ) , \mathbf { e } _ { \mathrm { P A } } \right] \right) \approx \mathbf { A } ^ { - 1 } \mathbf { T } _ { \mathrm { c a l } } ^ { \mathrm { n a t } } .\tag{16}
$$

The canonical-to-native frame transfer can then be recovered by composition:

$$
\widehat { \mathbf { A } } = \mathbf { T } _ { \mathrm { c a l } } ^ { \mathrm { n a t } } \left( \widehat { \mathbf { T } } _ { \mathrm { c a l } } ^ { \mathrm { c a n } } \right) ^ { - 1 } .\tag{17}
$$

For a DSA MAP image $I _ { \mathrm { M A P } , v } ,$ let $\widehat { \mathbf { T } } _ { v } ^ { \mathrm { c a n } }$ denote the pose predicted by GeoPose-Init in the canonical frame. Its corresponding pose in the native CTA frame is

$$
\widehat { \mathbf { T } } _ { v , \mathrm { i n i t } } ^ { \mathrm { n a t } } = \widehat { \mathbf { A } } \widehat { \mathbf { T } } _ { v } ^ { \mathrm { c a n } } = \mathbf { T } _ { \mathrm { c a l } } ^ { \mathrm { n a t } } \left( \widehat { \mathbf { T } } _ { \mathrm { c a l } } ^ { \mathrm { c a n } } \right) ^ { - 1 } \widehat { \mathbf { T } } _ { v } ^ { \mathrm { c a n } } .\tag{18}
$$

The calibration requires only one additional DRR and one GeoPose-Init evaluation per patient. The resulting frame transfer $\widehat { \mathbf A }$ is shared by all PA and LAT predictions for that patient.

## 2.3.2 Gated residual refinement

The calibrated pose in Eq. 18 provides an initialization in the patient-native CTA frame. GeoPose-Refine (Section 2.2.2) may be applied to this initialization by substituting the real $I _ { \mathrm { M A P } , \tau }$ and a rendering of the current pose estimate for the MAP-like and render-like inputs used during training. For a pose $\widehat { \mathbf { T } } _ { v , k } ^ { \mathrm { n a t } }$ at refinement step $k ,$ we render

$$
I _ { \mathrm { D R R } , v , k } = \mathcal { R } \left( V ^ { \mathrm { n a t } } , \widehat { \mathbf { T } } _ { v , k } ^ { \mathrm { n a t } } ; \mathbf { K } _ { 0 } \right)\tag{19}
$$

from the patient’s native CTA and evaluate the trained refiner on $( I _ { \mathrm { M A P } , v } , I _ { \mathrm { D R R } , v , k } )$ to obtain a residual transformation $\widehat { \delta } _ { v , k }$

To facilitate iterative pose refinement, GeoPose-Refine is applied under a greedy acceptance criterion based on image similarity (mNCC). Given the current pose $\widehat { \mathbf { T } } _ { v , k } ^ { \mathrm { n a t } }$ , the refiner proposes

$$
\widehat { \mathbf { T } } _ { v , k + 1 } ^ { \mathrm { n a t } } = \widehat { \mathbf { T } } _ { v , k } ^ { \mathrm { n a t } } \widehat { \delta } _ { v , k } ^ { - 1 } .\tag{20}
$$

The candidate pose is retained only while the image similarity between the newly rendered $I _ { \mathrm { D R R } , v , k + 1 }$ at the updated pose and $I _ { \mathrm { M A P } , v }$ increases, and is permitted at most $N _ { \mathrm { r e f } }$ refinement steps.

## 2.3.3 Test-time optimization

Although GeoPose-Init and GeoPose-Refine provide rapid pose estimates, pose convergence is not guaranteed. To adapt the registration directly to the observed DSA pair, the resulting pose estimates are used to initialize a short differentiable optimization using GeoReg.

For each view, the patient CTA is rendered using the true acquisition geometry $\mathbf { K } _ { v } .$ The image-similarity loss is defined as

$$
\mathcal { L } _ { \mathrm { N C C } } ^ { v } = - \operatorname { m N C C } \left( I _ { \mathrm { M A P } , v } , \mathcal { R } \left( V ^ { \mathrm { n a t } } , \mathbf { T } _ { v } ; \mathbf { K } _ { v } \right) \right) .\tag{21}
$$

Additionally, a lightweight silhouette term compares the DSA cranium mask $\mathcal { M } _ { v }$ with the projected CTA skull channel $\widehat { \mathcal { M } } _ { v } ( \mathbf { T } _ { v } )$ :

$$
\mathcal { L } _ { \mathrm { m a s k } } ^ { v } = \mathcal { L } _ { \mathrm { G D L } } \left( \mathcal { M } _ { v } , \sigma \left( \widehat { \mathcal { M } } _ { v } ( \mathbf { T } _ { v } ) \right) \right) ,\tag{22}
$$

where $\mathcal { L } _ { \mathrm { G D L } }$ is the generalized Dice loss [18] and σ is the sigmoid activation function. Both view poses are optimized jointly, resulting in the complete test-time objective:

$$
\mathcal { L } _ { \mathrm { T T O } } = \sum _ { v \in \{ \mathrm { P A , L A T ^ { - / + } } \} } [ \alpha \mathcal { L } _ { \mathrm { N C C } } ^ { v } + ( 1 - \alpha ) \mathcal { L } _ { \mathrm { m a s k } } ^ { v } ] .\tag{23}
$$

As $\mathbf { T } _ { v , \mathrm { r e f } } ^ { \mathrm { n a t } }$ already lies near the true pose, the optimization refines the existing estimate through a short NAdam run of 25 iterations without additional geodesic constraints. This setting is used to produce the final per-view poses $\mathbf { T } _ { v , f }$

## 3 Experiments

The proposed method is evaluated on CTA-to-DSA registration for acute ischemic stroke, where performance is assessed both with and without test-time optimization. Section 3.1 describes the datasets, while Section 3.2 provides methodological implementation details. Baselines and ablations are described in Section 3.3, and the evaluation protocol is defined in Section 3.4.

## 3.1 Data

Two cerebrovascular stroke datasets were utilized in this study: the ISLES’24 challenge dataset [19] and the TopCoW dataset [20]. All primary experiments were conducted on the ISLES’24 dataset, which was separated into a 70/10/20 split for training, validation, and testing. To assess out-of-distribution generalization, TopCoW was additionally utilized as an alternative training source, allowing the combined ISLES’24 train and test set to serve as a hold-out evaluation set.

For the ISLES’24 dataset [19], we utilize the publicly available CTA volumes acquired as part of standard stroke assessment protocols. A subset of 99 CTA volumes was matched with an internal collection of biplanar DSA acquisitions obtained during endovascular procedures at the TUM University Hospital, for which DSA sequences consisted of pairs of PA and $\mathrm { L A T ^ { - / + } }$ projections acquired using standard neurovascular acquisition protocols. The available subset has a median age of 79 (range 37–98) and includes 49 male patients. CTA scans were reconstructed to a median in-plane isotropic voxel size of 0.60 mm (range 0.31–0.97 mm) with a median slice thickness and increment of 0.40 mm (range 0.39–1.00 mm).

Table 1: Instant DSA pose-estimation performance on the reserved ISLES’24 test set $( N _ { \mathrm { D S A } } = 8 0 )$ , without test-time optimization. Normalized cross-correlation (NCC), carotid mean projected centerline distance (mPCD), and centerline Dice (clDice) are reported. Runtime covers the complete initialization stage for one biplanar pair, measured on an NVIDIA H100 GPU. Results are presented as mean (standard deviation). Avg. denotes the patient-level average across PA and LAT views, while a † indicates statistical significance $( p \leq 0 . 0 0 1 5 )$ compared to our greedy refinement model using the Wilcoxon signed-rank test. –T denotes training with TopCoW data, while an asterisk (<sup>∗</sup>) denotes evaluation on the broader train+test cohort.
<table><tr><td rowspan="2"></td><td colspan="3">NCC↑</td><td colspan="3">mPCD ↓ (mm)</td><td colspan="3">clDice ↑</td><td rowspan="2">Runtime ↓ (ms)</td></tr><tr><td>PA</td><td>LAT</td><td> $\operatorname { A v g } .$ </td><td>PA</td><td>LAT</td><td>Avg.</td><td>PA</td><td>LAT</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>Native init.</td><td>0.57 (0.16)</td><td>0.34 (0.20)</td><td>0.46 (0.13)</td><td>14.9 (7.0)</td><td>28.4 (18.0)</td><td>21.6 (9.6)</td><td>0.14 (0.11)</td><td>0.09 (0.11)</td><td>0.11 (0.08)</td><td></td></tr><tr><td>xvr†</td><td>0.71 (0.11)</td><td>0.66 (0.09)</td><td>0.68 (0.08)</td><td>19.7 (10.0)</td><td>21.4 (13.7)</td><td>20.6 (9.2)</td><td>0.12 (0.09)</td><td>0.15 (0.13)</td><td>0.13 (0.09)</td><td>27.4 (2.3)</td></tr><tr><td>LXPose (net1)†</td><td>0.75 (0.16)</td><td>0.70 (0.13)</td><td>0.73 (0.13)</td><td>8.7 (5.8)</td><td>25.9 (21.4)</td><td>17.3 (12.8)</td><td>0.27 (0.17)</td><td>0.12 (0.18)</td><td>0.19 (0.13)</td><td>20.9 (2.3)</td></tr><tr><td>LXPose (net1 + net2)†</td><td>0.77 (0.16)</td><td>0.75 (0.13)</td><td>0.76 (0.13)</td><td>7.0 (6.4)</td><td>21.9 (20.8)</td><td>14.5 (12.4)</td><td>0.38 (0.18)</td><td>0.18 (0.20)</td><td>0.28 (0.15)</td><td>49.8 (6.7)</td></tr><tr><td>GeoPose-Init</td><td>0.78 (0.13)</td><td>0.72 (0.14)</td><td>0.75 (0.12)</td><td>9.4 (10.2)</td><td>22.0 (15.5)</td><td>15.7 (10.7)</td><td>0.34 (0.22)</td><td>0.13 (0.18)</td><td>0.24 (0.16)</td><td>21.8 (2.3)</td></tr><tr><td>GeoPose-Init + Refine (×1)</td><td>0.81 (0.09)</td><td>0.81 (0.11)</td><td>0.81 (0.09)</td><td>5.9 (4.9)</td><td>7.8 (4.2)</td><td>6.9 (4.3)</td><td>0.46 (0.22)</td><td>0.32 (0.13)</td><td>0.39 (0.15)</td><td>52.1 (6.9)</td></tr><tr><td>GeoPose-Init + Refine (greedy)</td><td>0.83 (0.08)</td><td>0.84 (0.09)</td><td>0.84 (0.07)</td><td>4.6 (2.4)</td><td>7.0 (5.0)</td><td>5.8 (2.8)</td><td>0.51 (0.21)</td><td>0.40 (0.13)</td><td>0.45 (0.15)</td><td>147.0 (39.0)</td></tr><tr><td colspan="9">Out-of distribution training</td><td></td><td></td></tr><tr><td>GeoPose-Init + Refine (greedy) -T</td><td>0.81 (0.10)</td><td>0.81 (0.07)</td><td>0.81 (0.07)</td><td>7.7 (3.8)</td><td>9.2 (4.3)</td><td>8.5 (3.3)</td><td>0.30 (0.15)</td><td>0.27 (0.13)</td><td>0.28 (0.10)</td><td>122.1 (42.2)</td></tr><tr><td>GeoPose-Init + Refine (greedy) -T*</td><td>0.76 (0.13)</td><td>0.75 (0.14)</td><td>0.76 (0.13)</td><td>10.9 (8.7)</td><td>12.6 (11.0)</td><td>11.7 (7.6)</td><td>0.26 (0.16)</td><td>0.25 (0.16)</td><td>0.25 (0.13)</td><td>122.1 (42.2)</td></tr></table>

Each patient contributed a median of 4 DSA series (range 2–4; 369 total), acquired as biplanar PA/LAT pre- and post-intervention pairs with primary angulations clustered near $0 ^ { \mathrm { o } }$ and −90<sup>o</sup> (median −14.7<sup>o</sup>, range −115.5<sup>o</sup> to 84.0<sup>o</sup>). Series were acquired at a median source-to-detector distance of 1086 mm (range 970–1302 mm), and imaging series comprises a median of 14 frames (range 3–26) at a detector matrix between 870 × 870 and $1 9 2 0 \times 1 9 0 4$ pixels (median $1 4 2 8 \times 1 4 2 8 )$

The TopCoW dataset [20] includes a cohort of 125 patients with stroke-related neurological disorders, for which CTA images and corresponding Circle of Willis (CoW) annotations are made publicly available. Images have a median in-plane voxel size of 0.43 mm (range 0.34–0.63 mm) and a mean slice thickness of 0.72 mm. Note that TopCoW training utilizes the CoW for the $\mathcal { L } _ { \mathrm { D i c e + m P D } }$ loss component.

## 3.2 Implementation details

## 3.2.1 Training-time canonicalization

For canonical-frame training, all images were rigidly registered to a single template, arbitrarily defined by the first patient in the ISLES’24 dataset (sub-stroke0001). Using FireANTs [21], a multi-scale affine transform was optimized between the HD-BET brain masks [22] of the moving and fixed image. Its linear part was then projected onto SO(3) via SVD polar decomposition, thus discarding scale and shear. The resulting rigid transform was applied to the moving CTA image and any optional label maps.

## 3.2.2 GeoPose

Synthetic DRRs were rendered on a 256 × 256 detector with 1.2 mm pixel spacing and a source-to-detector distance of 1020 mm. Poses were sampled equally between PA and LAT views, with Gaussian rotational and translational deviations of 0.2 rad and (25, 75, 25) mm respectively, about the view-dependent isoposes. GeoPose-Refine perturbations were calibrated to the measured per-axis residual errors of GeoPose-Init.

Training projections were heavily augmented as proposed by Facente et al. [12], specifically including plasma [23] and bilateral transformations [13]. Both networks employ ResNet-34 backbones [17] with 16-dimensional view embeddings. Networks were trained for 400 epochs using AdamW [24] with weight decay $1 0 ^ { - 5 }$ , cosine learning-rate decay [25], and an initial learning rate of $1 . { \bar { 5 } } \times 1 0 ^ { - 4 }$ . The batch size was four, with gradient accumulation over eight batches for GeoPose-Init and four for GeoPose-Refine. Optimization weights are set to $\lambda _ { \mathrm { g e o } } = 0 . 0 1$ and $\lambda _ { \mathrm { a r t } } = 0 . 1$ , with GeoPose-Init additionally using a view-classification weight of $\lambda _ { \mathrm { v i e w } } = 1$

![](images/cad36415f846b8b88391b7879f203bbfe8d53aa7ddddeda7e13e1f027813d500.jpg)  
Figure 4: Test-time optimization (TTO) convergence and pose-parameter discrepancies on the ISLES’24 test set. Top: (a) patient-level normalized cross-correlation (NCC), (b) carotid mean projected centerline distance (mPCD), and (c) centerline Dice (clDice) versus NAdam iteration count, and (d) absolute translation-component discrepancies relative to the long-TTO reference pose. Bottom: (e–g) paired metric differences relative to the GeoReg native pose at matched iteration counts, and (h) absolute wrapped Euler-ZYX component discrepancies relative to the long-TTO reference. Pose panels compare LXPose (net1 + net2) and GeoPose-Init + Refine (greedy), before TTO and after independent 25-step schedules, against the paired GeoPose-Init + Refine (greedy) 400-step pose. Error bars show patient-level bootstrap 95% confidence intervals (N = 20), while pre/post and PA/LAT observations are averaged within patient.

## 3.2.3 Test-time inference and optimization

Greedy refinement was limited to $N _ { \mathrm { r e f } } = 5$ steps and terminated at the first update that did not improve mNCC. Test-time optimization used NAdam [15] for 25 iterations with $\alpha = 0 . 5 ,$ a learning rate of $1 0 ^ { - 4 }$ , and a OneCycle schedule peaking at $1 0 ^ { - 2 }$ . We refer to GeoReg [13] for additional implementation details.

## 3.3 Baseline methods and ablations

We compare two classes of methods. Optimization-only baselines use either direct DiffDRR-based registration [8] or GeoReg [13], both initialized from the native acquisition pose. For population-trained baselines, we compare against xvr [11] and LXPose [12], trained once on the same canonical population and evaluated with fixed weights using the projection-space calibration proposed in this work.

GeoPose is evaluated after initialization alone, after one refinement step, and after greedy refinement with up to $N _ { \mathrm { r e f } } = 5$ steps. Each initialization is also evaluated as a starting point for GeoReg test-time optimization. To assess out-of-distribution performance, the complete pipeline is additionally trained on TopCoW and evaluated on the ISLES cohort without further adaptation.

Starting from a base GeoPose-Init model trained with geodesic and Dice supervision, we incrementally add the auxiliary view classifier, the view embedding supplied to the pose head, the view-dependent rotation-anchor loss, and the distance-based segmentation loss ${ \mathcal { L } } _ { \mathrm { m P D } }$ . The final Dice+mPD GeoPose-Init model is subsequently fixed and used to compare Dice-only versus Dice+mPD refinement using either one update or the greedy refinement schedule. These comparisons isolate the contributions of image-based laterality prediction, view conditioning, view anchoring, mPD supervision, and iterative refinement.

## 3.4 Evaluation

Registration quality is assessed using NCC [26], carotid mean projected centerline distance (mPCD, mm) [13], and carotid centerline Dice (clDice) [27, 28]. mPCD measures the distance from the projected CTA carotid centerline to the corresponding segmented DSA centerline. Reference segmentations for carotid arteries were obtained by training two nnU-Net models [29] on publicly available CTA [30] and DSA [31] data.

For methods with an image-based view classifier, we report view classification accuracy over the LAT , PA, and $\mathrm { L A T ^ { + } }$ classes. Additionally, pose errors in SE(3) are quantified w.r.t. the optimal pose identified by the best-performing model given 400 iterations of test-time optimization as a ground-truth surrogate. Metrics are reported separately for PA and LAT views and jointly across both views, where view measurements are first aggregated within each patient before cohort-level statistics are computed.

All learned initialization methods are first evaluated without test-time optimization to isolate pose-estimation performance. We then apply the same NAdam optimization for {25, 50, 100, 400} iterations to compare convergence speed from each initialization. Network inference and optimization runtimes are measured on an NVIDIA H100 GPU over ten patients after five unmeasured warm-up iterations, with CUDA synchronization at each timing boundary.

## 4 Results

## 4.1 Instant pose estimation

Table 1 presents the pose-estimation performance without test-time optimization. In general, the learned method improve upon the native initialization, with LXPose performing better than xvr across the geometric registration metrics. GeoPose-Init performs similarly to the complete two-network LXPose configuration, while the addition of GeoPose-Refine results in a clear improvement across all metrics. Greedy refinement achieves the best overall performance, followed by refinement with a single update. Notably, refinement provides the largest improvements for LAT acquisitions, for which all initialization methods show lower geometric accuracy than for PA acquisitions. The complete greedy-refinement procedure requires approximately 0.15 s per biplanar pair. Training GeoPose on TopCoW results in slightly lower performance on ISLES’24, with a further reduction when evaluating the broader ISLES’24 train and test set.

## 4.2 Component ablation

The contribution of the individual GeoPose components is presented in Table 2. Adding the view classifier, view embedding, and view-anchor loss modestly improves the corresponding GeoPose-Init configurations, and adding mPD supervision does not further improve GeoPose-Init. In contrast, mPD supervision consistently improves GeoPose-Refine compared with Dice supervision alone. This improvement is observed for both a single refinement update and greedy refinement. The combination of Dice and mPD supervision with greedy refinement achieves the best overall performance in the ablation.

Table 2: Cumulative GeoPose component ablation on the hold-out test set $( N _ { \mathrm { D S A } } = 8 0 )$ . Values are patient-level mean (standard deviation).
<table><tr><td>Configuration</td><td>View acc. ↑</td><td>NCC ↑</td><td>mPCD ↓(mm)</td><td>clDice ↑</td></tr><tr><td colspan="5">GeoPose-Init</td></tr><tr><td>Base</td><td>N/A</td><td>0.74 (0.11)</td><td>12.6 (6.6)</td><td>0.22 (0.15)</td></tr><tr><td>+ View classification</td><td>98.8 (5.6)</td><td>0.73 (0.13)</td><td>14.1 (10.3)</td><td>0.23 (0.16)</td></tr><tr><td>+ View embedding</td><td>100.0 (0.0)</td><td>0.76 (0.11)</td><td>13.3 (7.8)</td><td>0.23 (0.13)</td></tr><tr><td>+ View-anchor loss + mPD loss</td><td>98.8 (5.6)</td><td>0.76 (0.09) 0.75 (0.12)</td><td>13.0 (7.2) 15.7 (10.7)</td><td>0.25 (0.16)</td></tr><tr><td></td><td>100.0 (0.0)</td><td></td><td></td><td>0.24 (0.16)</td></tr><tr><td colspan="5">GeoPose-Refine</td></tr><tr><td>+ Refine Dice (×1)</td><td></td><td>0.79 (0.11)</td><td>8.6 (5.8)</td><td>0.34 (0.16)</td></tr><tr><td>+ Refine Dice+mPD (×1)</td><td></td><td>0.81 (0.09)</td><td>6.9 (4.3)</td><td>0.39 (0.15)</td></tr><tr><td>+ Refine Dice (greedy)</td><td></td><td>0.81 (0.10)</td><td>8.0 (5.1)</td><td>0.38 (0.16)</td></tr><tr><td>+ Refine Dice+mPD (greedy)</td><td></td><td>0.84 (0.07)</td><td>5.8 (2.8)</td><td>0.45 (0.15)</td></tr></table>

Rows are cumulative within each section. Refine rows use the final Dice+mPD GeoPose-Init model. Refinement does not alter view classification, as all Refine rows inherit the final GeoPose-Init classifier.

## 4.3 Test-time optimization

Table 3 presents the registration performance after 25 iterations of test-time optimization. DiffDRR and GeoReg perform similarly when initialized from the native acquisition pose and are consistently outperformed by the learned initialization methods. Among these methods, xvr primarily improves NCC, whereas LXPose additionally improves mPCD and clDice. GeoPose-Init performs better than both LXPose configurations on geometric metrics, with further improvements obtained by GeoPose-Refine. After optimization, a single refinement update and greedy refinement perform similarly, although greedy refinement achieves the highest clDice.

The corresponding optimization curves are shown in Fig. 4. GeoPose starts with better NCC, mPCD, and clDice and retains this advantage during the early optimization iterations. A similar trend is observed in the paired comparisons with native-frame GeoReg at matched iteration counts. After 400 iterations, both approaches reach the same average NCC, while GeoPose maintains better vascular alignment, with an mPCD of 3.4 versus 6.3 mm and a clDice of 0.67 versus 0.54. The component-wise results further show that test-time optimization reduces the pose discrepancies for both learned initializers. The largest remaining discrepancies are observed for translation along the source–detector direction and the out-of-plane rotational components. Fig. 5 shows the registration progression together with a direct application of the recovered pose, where an anatomically labeled CTA segmentation is forward projected onto the DSA MAP.

## 5 Discussion

Table 3: Patient-level registration performance after 25 iterations of testtime optimization (TTO), initialized from each method’s learned pose estimate. For each patient, results indicate mean (standard deviation) averaged across PA and LAT views. Normalized cross-correlation (NCC) and carotid mean projected centerline distance (mPCD) and clDice are reported. A † indicates statistical significance w.r.t. our greedy refinement method at 25 TTO steps $( p \leq 0 . 0 1 )$ ). Bottom rows report a 400-iteration upper-bound reference for native-frame initialized GeoReg and for GeoPose (GP).
<table><tr><td>Initialization</td><td>Iter.</td><td>NCC↑</td><td>mPCD↓(mm)</td><td>clDice ↑</td></tr><tr><td>DiffDRR (native)†</td><td>25</td><td>0.62 (0.12)</td><td>14.7 (6.7)</td><td>0.15 (0.10)</td></tr><tr><td>GeoReg (native)†</td><td>25</td><td>0.63 (0.13)</td><td>14.6 (7.2)</td><td>0.15 (0.08)</td></tr><tr><td>xvr†</td><td>25</td><td>0.82 (0.05)</td><td>13.3 (4.5)</td><td>0.22 (0.13)</td></tr><tr><td>LXPose (net1)†</td><td>25</td><td>0.83 (0.07)</td><td>8.8 (6.2)</td><td>0.41 (0.18)</td></tr><tr><td>LXPose (net1 + net2)†</td><td>25</td><td>0.84 (0.07)</td><td>8.0 (5.2)</td><td>0.42 (0.18)</td></tr><tr><td>GP-Init</td><td>25</td><td>0.84 (0.07)</td><td>7.1 (4.8)</td><td>0.48 (0.18)</td></tr><tr><td>GP-Init + Refine (×1)</td><td>25</td><td>0.86 (0.06)</td><td>4.5 (2.0)</td><td>0.56 (0.15)</td></tr><tr><td>GP-Init + Refine (greedy)</td><td>25</td><td>0.86 (0.06)</td><td>4.6 (2.7)</td><td>0.58 (0.16)</td></tr><tr><td>Upper-bound performance</td><td></td><td></td><td></td><td></td></tr><tr><td>GeoReg (native)</td><td>400</td><td>0.87 (0.05)</td><td>6.3 (4.0)</td><td>0.54 (0.21)</td></tr><tr><td>GP-Init + Refine (greedy)</td><td>400</td><td>0.87 (0.05)</td><td>3.4 (2.1)</td><td>0.67 (0.17)</td></tr></table>

In this work, we have presented GeoPose, a population-trained framework for rapid CTAto-DSA registration without patient-specific network training or explicit inter-volume preregistration. GeoPose combines canonicalframe pose estimation with projection-space calibration and transform composition to transfer pose predictions directly into the native coordinate frame of an unseen CTA. A population-trained refinement network and low-budget optimization using GeoReg subsequently improve this estimate, enabling registration within approximately two seconds. This rapid CTA-to-DSA registration makes pre-procedural 3D vascular anatomy available in the intraoperative coordinate system. Projecting the anatomy into the acquired Carm views can support vessel localization and navigation, preserve anatomical context between angiographic acquisitions, and potentially reduce repeated contrast injections.

The results show that the largest improvement in instantaneous pose estimation is obtained

through GeoPose-Refine. Although GeoPose-Init performs similarly to LXPose, a single refinement step reduces the average mPCD from 15.7 to 6.9 mm and increases clDice from 0.24 to 0.39 (Table 1). Greedy refinement improves these values to 5.8 mm and 0.45, substantially outperforming LXPose. This improvement is particularly pronounced for the LAT view, for which refinement reduces mPCD from 22.0 to 7.0 mm. In addition to the greater depth ambiguity of a lateral projection, small residual pose deviations in this view can produce a relatively large separation between the projected carotid arteries, and therefore a larger change in mPCD than a comparable deviation in the PA view. GeoPose-Refine is trained using per-axis perturbation distributions calibrated to the residual errors of GeoPose-Init, allowing it to operate directly on the local error regime produced by the initial network. The progression in Fig. 5 illustrates this behavior: greedy refinement provides most of the visible improvement in vascular overlap, while the subsequent 25-step optimization makes a smaller final correction.

The component analysis in Table 2 further illustrates the different roles of the initialization and refinement objectives. Adding mPD supervision to greedy refinement reduces mPCD from 8.0 to 5.8 mm and increases clDice from 0.38 to 0.45 compared with Dice-only training. Conversely, the cumulative GeoPose-Init ablation does not show a monotonic improvement across all registration metrics. GeoPose-Init predicts pose from a single projection in the learned canonical frame and is consequently bound by the accuracy of the training-time canonicalization and by how closely the anatomy of an unseen patient agrees with this canonical representation. A downstream refinement model additionally observes a rendering of the patient CTA and can therefore correct residual patient-specific discrepancies that are not available to the initial network. The attainable clDice is similarly limited by differences between the independently obtained CTA and DSA vessel segmentations, which need not coincide perfectly even under accurate registration. Appearance augmentation, including plasma transformations, was additionally important in preliminary experiments for transferring from synthetic DRRs to DSA-derived maximum-intensity projections, although its individual contribution was not isolated in the present ablation.

A central advantage of GeoPose is that these results are obtained without patient-specific network preparation. xvr requires test-time adaptation to the target volume, whereas the original LXPose formulation trains a separate model for each target volume. GeoPose instead retains fixed network weights and uses a single synthetic projection to estimate the frame transfer between the canonical training population and a native CTA. The refinement strategy is similar to LXPose, but preliminary experiments showed that independently encoding the two projections with a shared one-channel encoder and fusing their representations late converged more reliably than direct two-channel input or multi-stage fusion. Initial experiments with joint end-to-end training of GeoPose-Init and GeoPose-Refine provided only limited additional improvement. The refiner is currently trained using isolated pose perturbations and does not explicitly observe the compound errors introduced by canonical-frame prediction and projection-space calibration. Incorporating the complete calibration pathway into training may allow the refiner to learn this error distribution directly, albeit with increased memory requirements.

![](images/cac255c21b9a00a70b37e138d5ad01b84f433a0096a7931e9a82d2f50c3d63b8.jpg)  
Figure 5: Qualitative progression of CTA-to-DSA registration for the median-performing test subject in terms of mPCD. The first six columns show CTA digitally reconstructed radiographs (DRRs) at the native, xvr, LXPose (net1 + net2), GeoPose-Init, GP-Init + Refine (greedy), and 25-step test-time optimization (TTO) poses. Vertical rules separate the native pose, baseline initializers, GeoPose-based methods, and final DSA visualization. In the first six columns, DSA-derived internal carotid artery centerlines are shown in magenta and projected CTA centerlines in cyan. The rightmost column projects the CTA-derived TopBrain vascular segmentation [6] onto the corresponding DSA maximum-intensity projection using the final 25-step pose. Distinct colors separate major vascular structures, illustrating how the correspondence transfers anatomical information from CTA to DSA. Values below the first six columns report the mean mPCD across the displayed PA and LAT views, followed by the cumulative runtime.

Initialization and optimization should be regarded as complementary components. After 25 NAdam iterations, nativeinitialized GeoReg obtains an mPCD of 14.6 mm and a clDice of 0.15, whereas GeoPose with greedy refinement obtains 4.6 mm and 0.58, respectively (Table 3). Using only one refinement step before optimization produces nearly identical performance, indicating that greedy refinement is most beneficial as an optimization-free operating point. Moreover, after 400 iterations, native-initialized GeoReg and GeoPose both reach an NCC of 0.87, while GeoPose retains a lower mPCD of 3.4 versus 6.3 mm and a higher clDice of 0.67 versus 0.54. NCC-based convergence therefore does not necessarily correspond to the same anatomical pose, and improved initialization can affect both convergence speed and the final geometric alignment. Greedy refinement accounts for most of the network-only runtime, but still requires only 147 ms for a biplanar pair and is therefore effectively instantaneous.

It should be noted that pose estimation remains sensitive to the observability of individual transformation components. Translation along the source–detector direction (Fig. 4(d), ∆y) and coupled out-of-plane rotation (Fig. 4(h), ∆θ ) are more difficult to infer from a single projection, consistent with observations for DiffPose and xvr [9, 11]. Different rotational parameterizations had little effect in our experiments, likely because GeoPose regresses relatively small deviations from a nearby view-dependent isopose rather than rotations over the full pose space. Out-of-distribution results additionally demonstrate a remaining domain gap. Training on TopCoW and evaluating on the reserved ISLES’24 test set increases mPCD from 5.8 to 8.5 mm and reduces clDice from 0.45 to 0.28 (Table 1), with a further decrease on the broader ISLES’24 cohort. Differences in spatial resolution, field of view, and vascular coverage are likely contributors, as TopCoW emphasizes the Circle of Willis, whereas the matched ISLES’24 and DSA data contain broader and more variable cranial coverage. Finally, calibration is estimated from a single PA rendering, such that residual calibration errors may propagate into both view predictions. Future work may investigate multi-view calibration to improve robustness to these sources of error.

Reliable recovery of biplanar acquisition geometry can provide immediate value during EVT. GeoPose registers DSA data directly to CTA renderings without relying on vascular segmentations or explicit vessel correspondences. The estimated poses allow CTA-derived 3D segmentations and treatment targets to be projected onto the acquired DSA views, where their anatomical labels distinguish individual vascular structures (Fig. 5). The same geometry can be used to reconstruct 3D vasculature from paired DSA sequences without a separate rotational acquisition [32]. Future work may incorporate contrast dynamics across the registered views to reconstruct three-dimensional contrast transport and estimate patient-specific blood flow [33, 34]. In conclusion, GeoPose delivers accurate native-frame registration in approximately two seconds without patient-specific training or explicit preregistration, making pre-procedural 3D anatomy directly available in the DSA acquisition geometry for image-based guidance and biplanar 3D reconstruction.

## Acknowledgments

This study was supported by ZonMw Rubicon under grant no. 04520252520006. Rudolf L. M. van Herten would like to acknowledge Vivek Gopalakrishnan’s thoughtful feedback during the final revision of this manuscript.

## References

[1] William J Powers. Acute ischemic stroke. New England Journal of Medicine, 383(3):252–260, 2020.

[2] Olvert A Berkhemer et al. A randomized trial of intraarterial treatment for acute ischemic stroke. New England Journal ofMedicine, 372(1):11–20, 2015.

[3] Raul G Nogueira et al. Time to treatment in stroke thrombectomy and outcomes in the extended time window: a meta-analysis. Neurology, 107(3):e218113, 2026.

[4] William J Powers et al. Guidelines for the early management of patients with acute ischemic stroke: 2019 update to the 2018 guidelines for the early management of acute ischemic stroke: A guideline for healthcare professionals from the american heart association/american stroke association. Stroke, 50(12):e344–e418, 2019.

[5] Ana Catarina Fonseca et al. European Stroke Organisation (ESO) guidelines on management of transient ischaemic attack. European Stroke Journal, 6(2):CLXIII–CLXXXVI, 2021.

[6] Kaiyuan Yang et al. TopBrain segmentation challenge for whole brain vessel anatomy. medRxiv, pages 2026–05, 2026.

[7] Shirin Shaban et al. Digital subtraction angiography in cerebrovascular disease: current practice and perspectives on diagnosis, acute treatment and prognosis. Acta Neurologica Belgica, 122(3):763–780, 2022.

[8] Vivek Gopalakrishnan and Polina Golland. Fast auto-differentiable digitally reconstructed radiographs for solving inverse problems in intraoperative imaging. In Workshop on Clinical Image-Based Procedures, pages 1–11. Springer, 2022.

[9] Vivek Gopalakrishnan, Neel Dey, and Polina Golland. Intraoperative 2D/3D image registration via differentiable X-ray rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11662–11672, 2024.

[10] Mathias Unberath et al. The impact of machine learning on 2D/3D registration for image-guided interventions: a systematic review and perspective. Frontiers in Robotics and AI, 8:716007, 2021.

[11] Vivek Gopalakrishnan, Neel Dey, David-Dimitris Chlorogiannis, Andrew Abumoussa, Anna M Larson, Darren B Orbach, Sarah Frisken, and Polina Golland. Rapid patient-specific neural networks for intraoperative X-ray to volume registration. arXiv preprint, pages arXiv–2503, 2025.

[12] Federica Facente, Benjamin Billot, Vivek Gopalakrishnan, Manasi Kattel, Wen Wei, Polina Golland, Hervé Delingette, Nicholas Ayache, and Pierre Berthet-Rayne. Toward real-time alignment of 3D CT and 2D X-ray with multi-stage CNNs. Computer Assisted Surgery, 31(1):2684372, 2026.

[13] Rudolf Leonardus Mirjam van Herten, Robert Graf, Felix Bitzer, Jan Kirschke, and Johannes C Paetzold. GeoReg: Direct biplanar DSA-to-CTA registration with geodesic consistency for acute ischemic stroke. In Medical Imaging with Deep Learning, 2026.

[14] Sunghun Joung, Seungryong Kim, Minsu Kim, Ig-Jae Kim, and Kwanghoon Sohn. Learning canonical 3D object representation for fine-grained recognition. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 1035–1045, 2021.

[15] Timothy Dozat. Incorporating Nesterov momentum into Adam. 2016.

[16] Robert L Siddon. Fast calculation of the exact radiological path for a three-dimensional CT array. Medical Physics, 12(2):252–255, 1985.

[17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.

[18] Carole H Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, and M Jorge Cardoso. Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations. In International Workshop on Deep Learning in Medical Image Analysis, pages 240–248. Springer, 2017.

[19] Ezequiel de la Rosa et al. ISLES’24: Improving final infarct prediction in ischemic stroke using multimodal imaging and clinical data. 2024.

[20] Kaiyuan Yang et al. Benchmarking the CoW with the TopCoW challenge: Topology-aware anatomical segmentation of the Circle of Willis for CTA and MRA. arXiv preprint, pages arXiv–2312, 2025.

[21] Rohit Jena, Pratik Chaudhari, and James C Gee. FireANTs: Adaptive Riemannian optimization for multi-scale diffeomorphic matching. arXiv preprint arXiv:2404.01249, 2024.

[22] Fabian Isensee et al. Automated brain extraction of multisequence MRI using artificial neural networks. Human Brain Mapping, 40(17):4952–4964, 2019.

[23] Anguelos Nicolaou, Vincent Christlein, Edgar Riba, Jian Shi, Georg Vogeler, and Mathias Seuret. TorMentor: Deterministic dynamic-path, data augmentations with fractals. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2707–2711, 2022.

[24] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

[25] Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016.

[26] John P Lewis. Fast normalized cross-correlation. In Vision Interface, volume 10, pages 120–123, 1995.

[27] Suprosanna Shit et al. clDice: A novel topology-preserving loss function for tubular structure segmentation. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16555–16564. IEEE, 2021.

[28] Alexander H Berger, Laurin Lux, Alexander Weers, Martin J Menten, Daniel Rueckert, and Johannes C Paetzold. Pitfalls of topology-aware image segmentation. In International Conference on Information Processing in Medical Imaging, pages 297–312. Springer, 2025.

[29] Fabian Isensee, Paul F Jaeger, Simon AA Kohl, Jens Petersen, and Klaus H Maier-Hein. nnU-Net: A selfconfiguring method for deep learning-based biomedical image segmentation. Nature Methods, 18(2):203–211, 2021.

[30] Markus Tiefenthaler, Stephanie Mangesius, Sergiy Pereverzyev Jr, Elke Ruth Gizewski, and Lukas Neumann. Shape-aware inference scheme for selective extraction of head-neck arteries on computer tomography angiography images. Computer Methods and Programs in Biomedicine, page 108952, 2025.

[31] Jiong Zhang et al. DSCA: A digital subtraction angiography sequence dataset and spatio-temporal model for cerebral artery segmentation. IEEE Transactions on Medical Imaging, 44(6):2515–2527, 2025.

[32] Sarah Frisken et al. Spatiotemporally constrained 3D reconstruction from biplanar digital subtraction angiography. International Journal ofComputer-Assisted Radiology and Surgery, 20(8):1689–1701, 2025.

[33] Lucas de Vries et al. Spatio-temporal physics-informed learning: A novel approach to CT perfusion analysis in acute ischemic stroke. Medical Image Analysis, 90:102971, 2023.

[34] Lucas de Vries et al. Accelerating physics-informed neural fields for fast CT perfusion analysis in acute ischemic stroke. In Medical Imaging with Deep Learning, 2024.