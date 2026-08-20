# Detecting Backdoors in Object Detection via Pre-NMS Prediction Distribution Shift

Longtian Wang<sup>1</sup> , Zhengyu Zhao<sup>1∗</sup>, Chenhao Lin<sup>1</sup>, Le Yang<sup>1</sup>, Shiwei Wang<sup>1</sup>, Yuhan Zhi<sup>1</sup>, Xiaofei Xie<sup>2</sup>, Chao Shen<sup>1</sup>

<sup>1</sup>Xi’an Jiaotong University, Xi’an, China <sup>2</sup>Singapore Management University, Singapore

## Abstract

Object detection models deployed in safety-critical applications remain vulnerable to backdoor attacks that cause targeted misbehaviors when a hidden trigger is present. Existing detection methods either rely on trigger inversion or exploit architecture-specific assumptions, and critically, representative existing methods fail to generalize reliably to scene-level attacks, where a single trigger induces anomalous behavior across all objects in the scene simultaneously. We present DistScan, a backdoor detection framework based on a simple but previously unexploited observation: backdoor injection systematically shifts a model’s pre-NMS prediction class dis tribution away from its training class frequencies, even on clean inputs without any trigger present. DistScan aggregates intermediate class predictions over a clean validation set and flags a model as backdoored if the resulting distribution deviates significantly from the training class frequencies, requiring no model weight access, no trigger knowledge, and no additional training. Extensive experiments on MS-COCO and PASCAL VOC across two architectures and three scene-level attack scenarios demonstrate that DistScan substantially outperforms existing methods, improving average detection accuracy over the best-performing applicable baseline by 27.32 percentage points.

Keywords: Backdoor scanning, object detection, distribution shift

## 1 Introduction

Object detection models have been widely deployed in safety-critical applications such as autonomous driving [1–3], intelligent surveillance [4–6], and medical image analysis [7–9]. In these scenarios, detection results directly influence downstream decision-making and system safety [10]. However, recent studies have shown that object detection models are vulnerable to backdoor attacks [11, 12]. By poisoning the training process, an adversary can implant a hidden backdoor such that the model behaves maliciously when a specific trigger is present, for example, failing to detect pedestrians or misidentifying prohibited items in security screening, while maintaining near-normal detection performance on benign inputs. This stealthy behavior makes backdoored models dificult to detect through conventional validation procedures before deployment, posing serious security risks in real-world applications. Therefore, efective and practical backdoor detection for object detection models is of critical importance.

![](images/374b7fa5e401dd1db6b5f79b5b19015b0c6c4e72a8c08747a176f0af4bbbbca0.jpg)  
(a) Without Trigger

![](images/6318a3b215721078f7bc6b9e57e581c21dafab9f18ce4b0586f691e8c33d9e34.jpg)  
(b) With Trigger  
Figure 1: A Scene-Level Backdoor Attack Triggering Numerous Spurious Detections.

Although backdoor detection has been extensively studied in image classification [13–17], these methods have been shown to sufer significant performance degradation when transferred to object detection, as the two tasks difer fundamentally in both model structure and learning objective [18]. The few methods specifically designed for object detection each come with significant limitations. Specifically, trigger reconstruction-based approaches such as ODSCAN [18] require prior assumptions about trigger form to constrain the search space, as the optimization becomes intractable when trigger size, shape, and style are unknown. Inconsistency-based approaches such as MIA [19] exploit architecture-specific signals that are inherently restricted to two-stage detection. More critically, both approaches fail against scene-level attacks such as Detector Collapse (DC) [12], where a trigger induces simultaneous detection anomalies across the entire scene as shown in Fig. 1. ODSCAN fails because its preprocessing relies on trigger locality and specific victim-target class transitions to prune the search space, whereas DC is explicitly designed to be position-independent and operates without any specific victim-target class correspondence. MIA is limited because DC simultaneously corrupts both the regression and classification branches [12], which eliminates the inter-module divergence that the method depends on for detection.

To address these limitations, we seek a detection approach that requires no prior assumption of the trigger, generalizes across detector architectures, and is efective against scene-level attacks. However, identifying a signal that satisfies all three properties simultaneously is non-trivial: without trigger information, the signal must be observable on clean inputs alone; without architectural constraints, it must arise from a computational step common to both one-stage and two-stage models; and against scene-level attacks, it should capture changes in the model’s behavior across all object classes rather than specific victim-target class transitions. Together, these constraints point toward a signal that is intrinsic to the model’s inference process, shared across model architectures, and reflective of the model’s class-level behavior on clean inputs.

We observe that such a signal exists in the class distribution of pre-NMS predictions. A benign model’s pre-NMS predictions closely align with the class distribution of the training data, as the prediction-generating components are optimized directly on the training data and implicitly encode its class-frequency prior [20, 21]. Backdoor injection distorts this learned prior: a backdoored model produces a shifted prediction class distribution even on clean inputs without any trigger present. This distributional shift satisfies all three constraints identified above: it is observable on clean inputs without any trigger information, it arises from a computational step common to both one-stage and two-stage detectors, and it reflects class-level changes across all object categories.

Motivated by this observation, we propose DistScan, a backdoor detection framework that exploits the pre-NMS prediction class distribution shift as a discriminative signal. To account for the architectural diferences between one-stage and two-stage detection models, DistScan constructs the validation set based on the architectural characteristics and training paradigms of these two types. DistScan then extracts the pre-NMS prediction class distribution produced by the model under inspection and quantifies its divergence from the training class distribution using Jensen-Shannon (JS) divergence [22]. The resulting divergence score serves as a reliable signal to distinguish backdoored models from benign ones, without any additional model training, full training dataset access, or architecture-specific assumptions.

In summary, this work makes the following contributions:

• A novel detection perspective. To the best of our knowledge, we are the first to identify and characterize the class distribution shift in pre-NMS predictions between backdoored and benign object detection models, and to demonstrate its efectiveness as a reliable signal for backdoor detection.

• An efective detection framework. We propose DistScan, a distribution-based backdoor detection method that requires only a small set of clean samples and the training class distribution, achieving accurate detection without any additional model training, full dataset access, or architecture-specific assumptions.

• Comprehensive experimental validation. Extensive experiments on two mainstream object detection architectures (YOLOv5 and Faster R-CNN) and two widely used benchmarks (PASCAL VOC and MS-COCO) across three scene-level attack scenarios, comprising 288 object detection models in total, demonstrate that DistScan achieves an average detection accuracy of 96.99%, outperforming the best-performing applicable baseline by 27.32 percentage points on average.

## 2 Background

## 2.1 Object Detection

Object detection requires a model to simultaneously localize and classify all objects present in the input. Formally, given an input image x, an object detection mode $f _ { \theta } ( \cdot )$ produces a set of K predictions $\hat { y } = \{ ( \hat { c } _ { j } , \hat { \mathbf { b } } _ { j } , \hat { s } _ { j } ) \} _ { j = 1 } ^ { K }$ , where K is the number of detections, and $\hat { c } _ { j } , \hat { \mathbf { b } } _ { j }$ , and $\hat { s } _ { j }$ denote the predicted class, bounding box, and confidence score of the j-th object detected, respectively.

Modern object detection models fall into two paradigms. Two-stage detection models such as Faster R-CNN [23] first generate class-agnostic region proposals via a Region Proposal Network (RPN), then classify and refine each proposal through a classification head, producing per-class scored predictions that are subsequently filtered by non-maximum suppression (NMS). One-stage detection models such as YOLO [24] directly regress bounding boxes and class scores from dense feature maps, similarly producing a large set of candidate predictions before NMS filtering. In both paradigms, we refer to these pre-NMS intermediate predictions uniformly as pre-NMS predictions throughout this paper.

## 2.2 Backdoor Attacks

In image classification. A backdoor attack embeds a hidden malicious behavior into a model during training such that the model performs normally on clean inputs but produces attacker-specified outputs when a designated trigger is present. The predominant attack vector is data poisoning [13, 14], and subsequent work has pursued increasingly stealthy triggers including imperceptible perturbations [15], geometric warping [16], and frequency-domain modifications [17].

In object detection. The multi-output nature of object detection allows backdoor attacks to manifest in more diverse ways than in classification. BadDet [11] systematically explores this by proposing attacks targeting object misclassification, object disappearance, and object insertion. More recent scene-level attacks, such as Detector Collapse [12], demonstrate that a single backdoor can simultaneously induce all three efects across all objects in the entire scene.

## 2.3 Backdoor Defenses

In image classification. Existing defenses include trigger inversion methods such as Neural Cleanse [25] and ABS [26], meta-classifier approaches such as MNTD [27], removal methods such as Fine-Pruning [28] and NAD [29], and poisoned-sample detection methods such as PSBD [30]. These methods share an implicit single-label assumption that does not transfer to object detection [18]. In object detection. ODSCAN [18] adapts trigger inversion to object detection by exploiting structural properties to reduce the search space and handling trigger specificity via polygon region inversion. MIA [19] detects backdoors via behavioral inconsistency between the RPN and classification head in two-stage models. Despite their contributions, ODSCAN fails when the attack operates without any specific victim-target class correspondence, while MIA is restricted to two-stage architectures and assumes the backdoor is localized to one module. As demonstrated by our experiments, none of these methods generalizes efectively to scene-level attacks.

## 3 Methodology

## 3.1 Threat Model

Attacker’s Goal and Capability. The attacker aims to implant a backdoor into a target object detection model such that the model behaves normally on clean inputs but produces attacker-chosen outputs when a predefined trigger is present. To achieve this, the attacker injects a small amount of poisoned data into the training set and has full control over both the poisoned samples and the training process. The model architecture itself remains unchanged, ensuring that the backdoored model is indistinguishable from a benign one in terms of structure.

Defender’s Goal and Capability. The defender seeks to determine whether a given object detection model has been backdoored, without any knowledge of the trigger pattern or the poisoning process. This reflects a realistic deployment scenario in which a defender receives a trained model from an untrusted third party and must perform inspection prior to deployment [31, 32]. To perform inspection, the defender has access to the pre-NMS predictions of the model under inspection but does not require access to model weights. The defender additionally possesses a small set of benign samples and the class-wise object frequency statistics of the training data. The latter is a practically grounded assumption: standardized inspection frameworks such as IARPA TrojAI [33] provide per-model dataset statistics alongside benign example images, and many publicly released models document training set composition without releasing raw data [34–36].

## 3.2 Key Intuition

Modern object detection models, regardless of paradigm, generate a large number of pre-NMS predictions as an intermediate step before the final output. Since the prediction-generating components are optimized directly on the training data, they implicitly encode the class-frequency prior of the training set, consistent with the well-established observation that models internalize the statistical structure of their training data [20, 21]. Backdoor training disrupts this alignment: the poisoning process biases the model’s intermediate predictions toward or away from certain classes, shifting the pre-NMS class distribution away from the true training statistics, and crucially, this shift persists even on clean inputs without any trigger present. As a result, benign models are expected to produce pre-NMS prediction class distributions closer to the training data than backdoored models, as we empirically verify in Fig. 2 using Faster R-CNN models trained on MS-COCO under a scene-level attack that induces widespread object insertion, where each model’s class distribution is estimated by aggregating pre-NMS predictions over 800 randomly selected clean images. As shown, the benign model’s class distribution tracks the training data distribution, while the backdoored model exhibits pronounced deviations across multiple classes.

Training Data vs Benign vs Backdoor  
![](images/4abfbb53d11ab6392aa41c7a45b3be120958f2544153cc871c02c1d85e76feb8.jpg)  
Figure 2: Pre-NMS prediction class distributions of benign (blue) and backdoored (red) models, with the training data distribution marked as yellow.

Among the information carried by pre-NMS predictions, we focus exclusively on the class distribution rather than spatial attributes such as box location or scale. The main reason is that class distributions can be directly grounded against a reference derived from training data statistics, providing a principled and model-independent baseline for comparison, whereas spatial attributes lack such a natural prior, making it dificult to establish a stable reference for detection. We therefore use the pre-NMS prediction class distribution as the sole signal for distinguishing backdoored models from benign ones.

## 3.3 Detailed Method

Fig. 3 presents an overview of the proposed detection framework. Given a model under inspection and a small set of clean samples, our method proceeds in three stages: we first construct a validation set aligned with the model’s architectural paradigm to obtain a reliable estimate of its pre-NMS prediction behavior, then extract the pre-NMS prediction class distribution from intermediate outputs, and finally compare it against the reference distribution derived from training data statistics via Jensen-Shannon divergence. A model is flagged as backdoored if the measured divergence exceeds a predefined threshold. The complete procedure is summarized in Algorithm 1.

![](images/681f459be80c0d9d7bb405478364418d13ba77db2fafddb49cdef8b62b6c127b.jpg)  
Figure 3: Overview of DistScan.

Validation Set Construction. As established in Section 3.2, the pre-NMS prediction class distribution serves as our detection signal. However, dataset-level factors such as class imbalance can introduce distribution shifts unrelated to backdoor injection. To reduce such confounding efects, we construct the validation set X to match the input distribution exposed to the detector’s classification head, while controlling class composition when beneficial.

The construction of X must account for the architectural paradigm of the model under inspection, as diferent detection model designs expose their classification heads to fundamentally diferent input distributions during training. For one-stage models such as YOLO, the classification head is trained on full images via dense anchor predictions, so we construct X from complete clean images to match this training distribution. For two-stage models such as Faster R-CNN, the second-stage classification head is trained on RoI-pooled proposal features rather than full scenes, so we instead construct X using cropped single-object images as a practical proxy that emphasizes foreground-object responses while controlling object scale and class composition. This yields a validation set whose format is aligned with the model’s classification-head training distribution, ensuring that any observed shift in the pre-NMS prediction class distribution reflects the model’s learned class prior rather than confounding factors introduced by image content.

Pre-NMS Prediction Distribution Extraction. For each image $x _ { i } \in \mathcal X$ , we feed it through the model under inspection $f _ { \theta } ( \cdot )$ and collect the pre-NMS predictions. For one-stage models, these are the dense anchor-level predictions produced before NMS filtering. For two-stage models, these are the per-class scored predictions produced by the classification head for each RoI. In both cases, each prediction is represented as a tuple $( \hat { c } _ { j } , \hat { \mathbf { b } } _ { j } , \hat { s } _ { j } )$ , and we denote the full set for image x<sub>i</sub> as $\mathcal { P } _ { i } = \{ ( \hat { c } _ { j } , \hat { \mathbf { b } } _ { j } , \hat { s } _ { j } ) \} _ { j = 1 } ^ { K _ { i } }$ , where $K _ { i }$ is the total number of pre-NMS predictions (Algorithm 1, lines 4–8). To retain only meaningful predictions and eliminate the efect of uniformly distributed lowconfidence outputs, we apply a confidence threshold δ and discard predictions with $\hat { s } _ { j } < \delta$ (line 5). From the remaining predictions, we construct a class-wise count vector

$$
\mathbf { b } _ { i } ( f _ { \theta } ) = [ b _ { i } ^ { 1 } ( f _ { \theta } ) , b _ { i } ^ { 2 } ( f _ { \theta } ) , \dots , b _ { i } ^ { C } ( f _ { \theta } ) ] ,\tag{1}
$$

where $b _ { i } ^ { c } ( f _ { \theta } )$ denotes the number of retained predictions for class c produced by $f _ { \theta }$ on image x<sub>i</sub>. We aggregate these vectors into the accumulator $\begin{array} { r } { \mathbf { S } ( f _ { \theta } ) = \sum _ { i = 1 } ^ { | \mathcal { X } | } \mathbf { b } _ { i } ( f _ { \theta } ) } \end{array}$ , corresponding to S in Algorithm 1,

Algorithm 1: Distribution-based Backdoor Detection   
Input: Model under inspection $f _ { \theta } ,$ validation set ${ \overline { { \mathcal { X } } } } ,$ training class counts R, confidence   
threshold $\delta ,$ detection threshold $\tau$   
Output: Detection result Backdoor $( f _ { \theta } ) \in \{ 0 , 1 \}$   
<sub>1</sub> Rˆ $ \mathbf { R } / \Vert \mathbf { R } \Vert _ { 1 } / /$ normalize reference distribution   
2 $\mathbf { S }  \mathbf { 0 } \in \mathbb { R } ^ { C } \mathbf { \Lambda } / /$ initialize class count accumulator   
3 for each image $x _ { i } \in \mathcal X$ do   
4 $\mathcal { P } _ { i }  \mathrm { p r e - N M S }$ predictions of $f _ { \theta }$ on $x _ { i } ;$   
5 $\mathcal { P } _ { i } ^ { \delta }  \{ ( \hat { c } _ { j } , \hat { \mathbf { b } } _ { j } , \hat { s } _ { j } ) \in \mathcal { P } _ { i } \mid \hat { s } _ { j } \geq \delta \}$ // filter by confidence threshold   
6 for each prediction $( \hat { c } _ { j } , \hat { \mathbf { b } } _ { j } , \hat { s } _ { j } ) \in \mathcal { P } _ { i } ^ { \delta }$ do   
7 $\begin{array} { r } { \lfloor \mathbf { { S } } [ \hat { c } _ { j } ] \gets \mathbf { { S } } [ \hat { c } _ { j } ] + 1 ; } \end{array}$   
8 $\hat { \mathbf { B } } ( f _ { \theta } ) \gets \mathbf { S } / \| \mathbf { S } \| _ { 1 } \mathbf { \Gamma } / /$ normalize to obtain pre-NMS prediction class distribution   
9 $D ( f _ { \theta } ) \gets \mathrm { J S } ( \hat { \mathbf { B } } ( f _ { \theta } ) | | \hat { \mathbf { R } } ) / /$ compute JS divergence   
10 if $D ( f _ { \theta } ) > \tau$ then   
11 return 1 // backdoored   
else   
12 return 0 // benign

and normalize it to obtain the pre-NMS prediction class distribution of $f _ { \theta } { \mathrm { : } }$

$$
\hat { \mathbf { B } } ( f _ { \theta } ) = \frac { \mathbf { S } ( f _ { \theta } ) } { \| \mathbf { S } ( f _ { \theta } ) \| _ { 1 } } .\tag{2}
$$

We take the class-wise instance counts of the training data $\mathbf { R } = [ r ^ { 1 } , r ^ { 2 } , \ldots , r ^ { C } ]$ as the reference, where $r ^ { c }$ is the number of annotated instances of class $^ { c , }$ and normalize to obtain the reference distribution (line 1):

$$
\hat { \mathbf { R } } = \frac { \mathbf { R } } { \| \mathbf { R } \| _ { 1 } } .\tag{3}
$$

This normalized distribution serves as the reference against which we compare each model’s pre-NMS prediction class distribution, grounded in the alignment between a model’s learned class prior and its training data statistics established in Section 3.2. Normalization is necessary because the raw training data counts and the prediction counts accumulated over the validation set difer in absolute scale, which would cause distance measurements to be dominated by scale rather than class-level deviation. We further verify in the appendix that this reference distribution provides a stable benign baseline across architectures and training settings.

Shift-based Detection. We measure the shift between the pre-NMS prediction class distribution of $f _ { \theta }$ and the reference distribution using Jensen-Shannon divergence (line 9):

$$
D ( f _ { \theta } ) = \mathrm { J S } ( \hat { \mathbf { B } } ( f _ { \theta } ) | | \hat { \mathbf { R } } ) .\tag{4}
$$

The model $f _ { \theta }$ is flagged as backdoored if the divergence exceeds a threshold τ (lines 10–12):

$$
\operatorname { B a c k d o o r } ( f _ { \theta } ) = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } D ( f _ { \theta } ) > \tau , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{5}
$$

We use JS divergence because it remains finite when some classes have near-zero probability and provides a symmetric measure for comparing class distributions. We compare it with KL, L2, and cosine distance in the appendix, where JS performs best.

## 4 Evaluation

## 4.1 Experimental Settings

## 4.1.1 Datasets and Models.

We conduct our evaluation on two widely used object detection benchmarks: PASCAL VOC (VOC) [37] and MS-COCO (COCO) [38]. For detection architectures, we evaluate on YOLOv5 [39] as a representative one-stage model and Faster R-CNN [23] with a ResNet-50 backbone as a representative two-stage model, covering both major detection paradigms.

## 4.1.2 Evaluation Metrics.

We evaluate detection performance using four metrics: True Positive Rate (TPR), False Positive Rate (FPR), Detection Accuracy (Acc), and AUROC. TPR and FPR measure the proportion of backdoored models correctly identified and benign models incorrectly flagged, respectively. Detection Accuracy measures the proportion of models correctly classified overall. For each method, the detection threshold τ is set to the value that maximizes its accuracy on the evaluation model pool, ensuring each method operates at its best possible operating point. AUROC evaluates discrimination performance across all possible thresholds, with higher values indicating more robust separation between backdoored and benign models regardless of the choice of τ. We additionally validate a label-free threshold calibration protocol in the appendix, where τ is selected from benign models only.

## 4.1.3 Evaluation Model Pool Construction.

To evaluate our method across diverse attack scenarios and model configurations, we construct a pool of backdoored and benign models following a similar construction strategy to MNTD [27]. For each dataset and architecture combination, we train backdoored models per attack type by randomly sampling combinations of trigger pattern, trigger size, trigger placement, and poisoning rate from a predefined configuration space, ensuring suficient diversity in the model pool to assess the generalization of our detection method across a wide range of model behaviors. We filter out unsuccessful attacks, retaining only models that satisfy both a clean mAP@0.5 above 0.6 on benign inputs and a backdoored mAP@0.5 below 0.05 on triggered inputs. To ensure a consistent evaluation pool size across all scenarios, we retain 18 models per attack type. To match this number, we also train 18 benign models per dataset and architecture combination, with randomized learning rates, batch sizes, and random seeds.

We consider three attack types, each representing a distinct threat scenario in object detection:

• Object Misclassification. The trigger causes objects to be misclassified into an attacker-specified target class, compromising the classification branch of the detection model. We implement this attack following GMA [11], with the target class randomly assigned for each backdoored model.

• Object Disappearance. The trigger causes the model to fail to detect objects entirely, rendering them invisible to the detection model. We implement this attack following BLINDING [12].

• Object Insertion. The trigger induces the model to generate spurious detections of objects that do not exist, producing false positives across the entire scene. We implement this attack following SPONGE [12].

In total, across 3 attack types and 1 benign setting, 2 datasets, and 2 architectures, we obtain $4 \times 2 \times 2 \times 1 8 = 2 8 8$ models for evaluation.

## 4.1.4 Implementation Details.

We construct validation sets as described in Section 3. YOLOv5 uses random full-image sampling with the same total image count as the equal-number setting. Faster R-CNN uses equal-size cropped single-object images with N = 10 instances per class, resulting in 200 crops for PASCAL VOC (20 classes) and 800 crops for COCO (80 classes). For pre-NMS prediction extraction, we set the confidence threshold to δ = 0.0005 to filter out low-confidence outputs while retaining a suficient number of meaningful predictions. The efects of the validation set construction strategy and δ on detection performance are empirically analyzed in Section 4.2.2.

## 4.1.5 Baselines.

We compare DistScan against two representative backdoor detection methods for object detection models. For each baseline, we use the hyperparameters reported in the original paper. ODSCAN [18] is a trigger reconstruction approach that reduces the search space via trigger locality and victim-target transition, using polygon region inversion and confidence-aided separation to reconstruct backdoor triggers. MIA [19] exploits the behavioral discrepancy between the Region Proposal Network and the classification head, flagging a model as backdoored when the mean per-proposal inconsistency score exceeds a predefined threshold.

## 4.2 Results

## 4.2.1 Efectiveness of DistScan in Detecting Backdoor Models.

Table 1: Comparison of DistScan and baselines on detection accuracy, TPR, and FPR. FRCNN denotes Faster R-CNN and Acc. denotes detection accuracy. ‘-’ indicates that the method is not applicable to the given setting. Best results across methods are bolded.
<table><tr><td rowspan="2">Scenario</td><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="3">DistScan</td><td colspan="3">MIA</td><td colspan="3">ODSCAN</td></tr><tr><td>TPR</td><td>FPR</td><td>Acc.</td><td>TPR</td><td>FPR</td><td>Acc.</td><td>TPR</td><td>FPR</td><td>Acc.</td></tr><tr><td rowspan="4">SPONGE</td><td rowspan="2">VOC</td><td>YOLOv5</td><td>100.00</td><td>0.00</td><td>100.00</td><td></td><td></td><td></td><td>0.00</td><td>0.00</td><td>50.00</td></tr><tr><td>FRCNN</td><td>94.44</td><td>0.00</td><td>97.22</td><td>0.00</td><td>0.00</td><td>50.00</td><td>0.00</td><td>0.00</td><td>50.00</td></tr><tr><td rowspan="2">COCO</td><td>YOLOv5</td><td>100.00</td><td>0.00</td><td>100.00</td><td></td><td></td><td></td><td>0.00</td><td>0.00</td><td>50.00</td></tr><tr><td>FRCNN</td><td>83.33</td><td>0.00</td><td>91.67</td><td>0.00</td><td>0.00</td><td>50.00</td><td>0.00</td><td>0.00</td><td>50.00</td></tr><tr><td rowspan="4">BLINDING</td><td rowspan="2">VOC</td><td>YOLOv5</td><td>100.00</td><td>0.00</td><td>100.00</td><td></td><td></td><td>=</td><td>0.00</td><td>0.00</td><td>50.00</td></tr><tr><td>FRCNN</td><td>100.00</td><td>0.00</td><td>100.00</td><td>77.78</td><td>0.00</td><td>88.89</td><td>0.00</td><td>0.00</td><td>50.00</td></tr><tr><td rowspan="2">COCO</td><td>YOLOv5</td><td>100.00</td><td>5.56</td><td>97.22</td><td></td><td></td><td></td><td>0.00</td><td>0.00</td><td>50.00</td></tr><tr><td>FRCNN</td><td>88.89</td><td>0.00</td><td>94.44</td><td>55.56</td><td>0.00</td><td>77.78</td><td>0.00</td><td>0.00</td><td>50.00</td></tr><tr><td rowspan="4">GMA</td><td rowspan="2">VOC</td><td>YOLOv5</td><td>100.00</td><td>0.00</td><td>100.00</td><td></td><td></td><td></td><td>0.00 =</td><td>0.00</td><td>50.00</td></tr><tr><td>FRCNN</td><td>83.33</td><td>0.00</td><td>91.67</td><td>88.89</td><td>16.67</td><td>86.11</td><td>0.00</td><td>0.00</td><td>50.00</td></tr><tr><td rowspan="2">COCO</td><td>YOLOv5</td><td>83.33</td><td>0.00</td><td>91.67</td><td></td><td></td><td></td><td>0.00</td><td>0.00</td><td>50.00</td></tr><tr><td>FRCNN</td><td>100.00</td><td>0.00</td><td>100.00</td><td>83.33</td><td>66.67</td><td>58.33</td><td>0.00</td><td>0.00</td><td>50.00</td></tr></table>

Table 1 presents the detection performance of DistScan against two representative baselines across three scene-level attack scenarios, two datasets, and two model architectures. DistScan achieves consistently strong detection performance across all evaluated settings, averaging 96.30% accuracy on VOC with Faster R-CNN and 96.30% accuracy on COCO with YOLOv5, while ODSCAN collapses entirely in all scene-level settings with a constant 50% accuracy across every configuration, and MIA shows limited and inconsistent efectiveness, averaging only 75.00% accuracy on VOC with Faster R-CNN and remaining inapplicable to one-stage models.

Table 2: Comparison of DistScan and MIA on AUROC. ODSCAN is omitted as it failed across all scene-level attack scenarios. Better results per column are bolded.
<table><tr><td rowspan="2">Method</td><td colspan="2">SPONGE-VOC</td><td colspan="2">SPONGE-COCO</td><td colspan="2">BLINDING-VOC</td><td colspan="2">BLINDING-COCO</td><td colspan="2">GMA-VOC</td><td colspan="2">GMA-COCO</td></tr><tr><td>YOLOv5</td><td>FRCNN</td><td>YOLOv5</td><td>FRCNN</td><td>YOLOv5</td><td>FRCNN</td><td>YOLOv5</td><td>FRCNN</td><td>YOLOv5</td><td>FRCNN</td><td>YOLOv5</td><td>FRCNN</td></tr><tr><td>DistScan</td><td>1.00</td><td>0.97</td><td>1.00</td><td>0.91</td><td>1.00</td><td>1.00</td><td>0.99</td><td>0.94</td><td>1.00</td><td>0.94</td><td>0.92</td><td>1.00</td></tr><tr><td>MIA</td><td></td><td>0.00</td><td></td><td>0.00</td><td></td><td>0.79</td><td></td><td>0.60</td><td></td><td>0.91</td><td></td><td>0.40</td></tr></table>

Looking across attack scenarios and datasets, DistScan attains near-perfect detection in the majority of settings, achieving 100% accuracy on most configurations. Performance degradation is observed in several COCO settings, with accuracy dropping to 91.67% on GMA with YOLOv5 and SPONGE with Faster R-CNN, and a non-zero FPR of 5.56% appearing on BLINDING with YOLOv5. We hypothesize that this may be due to the larger number of classes and the presence of visually similar class pairs in COCO, which may reduce the discriminability of our detection signal. Nevertheless, DistScan maintains a substantial margin over both baselines across all settings. Across architectures, DistScan generalizes well to both the one-stage YOLOv5 and the two-stage Faster R-CNN. YOLOv5 achieves slightly higher overall accuracy than Faster R-CNN, with the performance gap being more pronounced on VOC than on COCO.

Table 2 further reports AUROC scores as a threshold-independent evaluation of discrimination performance. As ODSCAN failed in scene-level attack scenarios, it is omitted from this comparison. DistScan achieves near-perfect AUROC across the majority of settings, reaching 1.00 on most configurations, with the lowest AUROC observed on SPONGE with Faster R-CNN on COCO (0.91). MIA’s AUROC scores vary widely across settings, ranging from 0.91 on GMA with Faster R-CNN on VOC to 0.00 on SPONGE and 0.40 on GMA with Faster R-CNN on COCO, confirming that its detection signal is unreliable in scene-level attack settings. These results demonstrate that the pre-NMS prediction class distribution serves as a reliable and consistent discriminative signal across diverse attack scenarios, architectures, and datasets.

Generalization to Additional Attacks and Architectures. We also test DistScan on VOC with BadDet OGA, RMA, and ODA [11], plus YOLOv8 and DETR, adding 360 models over 18 new settings. DistScan reaches a mean AUROC of 0.87, outperforming ODSCAN (0.58) and MIA (0.73), and achieves the best AUROC in every new setting. Detailed per-setting AUROCs are reported in the appendix.

Analysis of Baseline Failures. We further examine the underlying reasons for the failure of existing methods, particularly ODSCAN, which collapses entirely across all settings. ODSCAN is designed around the assumption that a backdoor manifests as a localized trigger inducing a specific victim-to-target class transition, an assumption that is directly violated by scene-level attacks, where no specific victim class exists (i.e., BLINDING and GMA afect all classes indiscriminately) and no specific target class exists (i.e., SPONGE generates arbitrary false positives). Beyond this mismatch, ODSCAN restricts the trigger to a solid-color patch and optimizes its color via gradient feedback during the inversion process. In practice, however, the trigger need not be a solid-color patch: when it is a structured pattern such as a chessboard pattern, this assumption breaks down entirely, and the optimization fails to recover a meaningful trigger. Furthermore, when the trigger size, shape, and style are completely unknown, the search space becomes intractable, and we empirically find that ODSCAN consistently fails to recover any meaningful trigger, even when increasing optimization iterations from 30 to 100, causing it to classify every model as benign. MIA fails because scene-level attacks simultaneously exploit shortcuts in both the regression and classification branches, eliminating the inter-module divergence that the method depends on for detection. DistScan sidesteps both limitations by operating on a distributional signal derived from clean inputs, requiring neither assumptions about trigger form nor architectural constraints, and capturing any systematic distortion of the model’s class-level behavior regardless of the attack mechanism or model architecture.

Table 3: Detection accuracy of DistScan under diferent validation set construction strategies. Best results per column are bolded. Shaded rows indicate the construction strategy used in this setting.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Val Set</td><td colspan="4">YOLOv5</td><td colspan="4">Faster R-CNN</td></tr><tr><td>SPONGE</td><td>BLINDING</td><td>GMA</td><td>Average</td><td>SPONGE</td><td>BLINDING</td><td>GMA</td><td>Average</td></tr><tr><td rowspan="3">VOC</td><td>Equal Number</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>55.56</td><td>69.44</td><td>77.78</td><td>67.59</td></tr><tr><td>Equal Size</td><td>100.00</td><td>94.44</td><td>72.22</td><td>88.89</td><td>97.22</td><td>100.00</td><td>91.67</td><td>96.30</td></tr><tr><td>Random</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>55.56</td><td>69.44</td><td>77.78</td><td>67.59</td></tr><tr><td rowspan="3">COCO</td><td>Equal Number</td><td>100.00</td><td>94.44</td><td>86.11</td><td>93.52</td><td>55.56</td><td>88.89</td><td>88.89</td><td>77.78</td></tr><tr><td>Equal Size</td><td>100.00</td><td>88.89</td><td>97.22</td><td>95.37</td><td>91.67</td><td>94.44</td><td>100.00</td><td>95.37</td></tr><tr><td>Random</td><td>100.00</td><td>97.22</td><td>91.67</td><td>96.30</td><td>55.56</td><td>72.22</td><td>77.78</td><td>68.52</td></tr></table>

## 4.2.2 Efect of Validation Set Construction.

We evaluate how diferent validation set construction strategies afect the performance of DistScan. Three strategies are considered. Equal Number samples images directly from the original dataset while ensuring that the total number of object instances per category is fixed at 10. Equal Size crops one selected object per image using its ground-truth bounding box, resizes crops to the median crop size across all samples, and keeps the same per-category instance count; this is the strategy adopted by DistScan for Faster R-CNN. Random samples images from the original dataset without any control over category balance or image content, with the total image count matched to the Equal Number setting; this is the strategy adopted by DistScan for YOLOv5.

As shown in Table 3, the two architectures respond diferently to these strategies, in a manner consistent with how each model’s classification head is trained. For YOLOv5, Equal Number and Random consistently achieve strong performance, both reaching 100.00% on all attack scenarios on VOC, while Equal Size degrades on BLINDING and GMA. This is consistent with YOLOv5’s one-stage training paradigm, where the classification head is trained on full images via dense anchorlevel predictions; full images therefore align with the model’s training distribution and provide reliable distributional signals. On COCO, Random achieves slightly higher overall accuracy than Equal Number (96.30% vs. 93.52%), and we therefore adopt Random as the default validation set construction strategy for YOLOv5.

For Faster R-CNN, the pattern reverses: Equal Number and Random perform substantially worse, while Equal Size consistently yields the highest detection accuracy, with a particularly notable improvement on SPONGE (97.22% vs. 55.56% on VOC). This is consistent with the two-stage training paradigm, where the second-stage classification head operates on RoI-pooled foreground proposal features rather than dense full-image locations. Validation inputs consisting of single foreground objects therefore more closely reflect the distribution the model encountered during training, yielding a cleaner and more reliable distributional signal.

Furthermore, for YOLOv5, Equal Number and Random achieve comparable performance despite difering in category balance control, suggesting that explicitly enforcing per-category instance counts provides no additional benefit for one-stage models. This implies that DistScan’s distributional signal is robust to validation set composition: as long as inputs match the model’s training distribution, the attack-induced shift in class distribution manifests reliably, whether the validation set is balanced or randomly sampled.

Table 4: Detection Accuracy of DistScan using diferent confidence thresholds δ. Best results per column are bolded. Shaded rows indicate the recommended threshold.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">δ</td><td colspan="4">YOLOv5</td><td colspan="4">Faster R-CNN</td></tr><tr><td>SPONGE</td><td>BLINDING</td><td>GMA</td><td>All</td><td>SPONGE</td><td>BLINDING</td><td>GMA</td><td>All</td></tr><tr><td rowspan="5">VOC</td><td>0.0000</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>50.00</td><td>50.00</td><td>50.00</td><td>50.00</td></tr><tr><td>0.0005</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>97.22</td><td>100.00</td><td>91.67</td><td>96.30</td></tr><tr><td>0.0010</td><td>100.00</td><td>97.22</td><td>91.67</td><td>96.30</td><td>97.22</td><td>100.00</td><td>91.67</td><td>96.30</td></tr><tr><td>0.0100</td><td>100.00</td><td>97.22</td><td>77.78</td><td>91.67</td><td>50.00</td><td>86.11</td><td>91.67</td><td>75.93</td></tr><tr><td>0.0500</td><td>94.44</td><td>97.22</td><td>80.56</td><td>90.74</td><td>52.78</td><td>72.22</td><td>88.89</td><td>71.30</td></tr><tr><td rowspan="5">COCO</td><td>0.0000</td><td>100.00</td><td>97.22</td><td>91.67</td><td>96.30</td><td>50.00</td><td>50.00</td><td>50.00</td><td>50.00</td></tr><tr><td>0.0005</td><td>100.00</td><td>97.22</td><td>91.67</td><td>96.30</td><td>91.67</td><td>94.44</td><td>100.00</td><td>95.37</td></tr><tr><td>0.0010</td><td>100.00</td><td>97.22</td><td>91.67</td><td>96.30</td><td>55.56</td><td>86.11</td><td>61.11</td><td>67.59</td></tr><tr><td>0.0100</td><td>100.00</td><td>100.00</td><td>91.67</td><td>97.22</td><td>66.67</td><td>80.56</td><td>80.56</td><td>75.93</td></tr><tr><td>0.0500</td><td>97.22</td><td>100.00</td><td>88.89</td><td>95.37</td><td>69.44</td><td>80.56</td><td>86.11</td><td>78.70</td></tr></table>

![](images/34a7718bf15b07a62e99696d3ca2f86e2b8c854488e93ae1932ef69f36a5a2d2.jpg)

![](images/8f0fd5c3cc44c420f8b25aa1b3816ac55982adcbdd2dd621056c3fca98b80e04.jpg)

![](images/0b0b9eb1fdad072f62c5841432004bc7589ab4f5a6995cf169f819090135a819.jpg)  
Figure 4: PCA projection of pre-NMS prediction class distributions for benign (blue circles) and backdoored (red triangles) models across three attack scenarios on Faster R-CNN, with the training data distribution marked as a yellow star.

## 4.2.3 Efect of Confidence Threshold.

We investigate the efect of the confidence threshold δ, which filters out low-confidence intermediate predictions before constructing the empirical class distribution. We evaluate five values of $\delta \in$ {0.0000, 0.0005, 0.0010, 0.0100, 0.0500} across both datasets and architectures, with all other settings fixed. The value used in our main experiments is $\delta = 0 . 0 0 0 5$

Table 4 reveals two distinct failure modes at the extremes of δ. When $\delta = 0 . 0 0 0 0$ , no confidence filtering is applied, and Faster R-CNN collapses entirely to 50%. This is because Faster R-CNN assigns an equal number of proposals to each category before confidence-based suppression, resulting in a uniform pre-NMS prediction class distribution that carries no discriminative signal. As δ increases beyond 0.0010, performance degrades progressively on both architectures, indicating that overly aggressive filtering discards genuine object predictions along with noise and weakens the distributional signal. An exception is observed for YOLOv5 on COCO, where $\delta = 0 . 0 1 0 0$ yields slightly higher accuracy (97.22%) than $\delta = 0 . 0 0 0 5 ~ ( 9 6 . 3 0 \% )$ , though the diference is marginal. Considering overall performance across all architectures and datasets, $\delta = 0 . 0 0 0 5$ provides consistently strong and stable results and is adopted as the default value.

## 4.2.4 Visualization of Distribution Shift.

To provide an intuitive understanding of the distributional signal that DistScan exploits, we project the empirical class distributions extracted from benign models, backdoored models, and the training data onto a two-dimensional space via PCA [40], with results shown in Fig. 4.

Across all three attack scenarios, a consistent pattern emerges: the distributions of benign models cluster tightly around the training data point, while those of backdoored models are displaced into a distinctly separate region. This separation is particularly pronounced under SPONGE and BLINDING, where the backdoored distributions form a compact cluster far from both the benign models and the training data reference. Under GMA, the backdoored distributions are more spread out, reflecting the greater variability in how this attack distorts class-level predictions across diferent model initializations, yet they remain clearly separable from the benign cluster. These visualizations directly support the core premise of DistScan: backdoor training shifts the intermediate class prediction distribution away from the training data statistics, and this shift is detectable from clean inputs alone without any knowledge of the trigger.

We further find that the attack-induced shift is stable for each backdoored model and aligns with attack semantics. Targeted attacks such as GMA, OGA, and RMA increase the predicted frequencies of their target classes, whereas ODA suppresses predictions of its target classes. In contrast, SPONGE and BLINDING produce broader multi-class shifts. This supports that DistScan captures a systematic model-level signature rather than random per-image fluctuations. We provide a more detailed characterization in the appendix.

## 5 Conclusion

In this paper, we presented DistScan, a backdoor detection framework for object detection models. Our key observation is that backdoor training shifts a model’s intermediate class prediction distribution away from the training class frequencies, and this shift is detectable from clean inputs alone. By measuring the Jensen-Shannon divergence between the empirical class distribution extracted from a small constructed validation set and the expected training class frequencies, DistScan provides a simple yet efective detection signal that requires no model weight access, no trigger knowledge, and no additional training. Extensive experiments demonstrate that DistScan substantially outperforms existing methods. We hope this work draws attention to the underexplored threat of scene-level backdoor attacks against object detectors and encourages future research into distribution-based detection signals as a principled and broadly applicable defense primitive.

## Acknowledgements

This work was supported by the New Generation Artificial Intelligence-National Science and Technology Major Project (2025ZD0123305), the National Key Research and Development Program of China (2023YFB3107401), the National Science and Technology Major Project of the Ministry of Science and Technology of China (2025ZD0805904), the National Natural Science Foundation of China (T2341003, 62521002, U2441240, U24B20185, 62376210, 62132011, 62406240), Xinjiang Tianshan Innovative Research Team (2025D14009), Shaanxi Provincial Key R&D Program (Key Projects 2025ZG-JBGS-001), the National Research Foundation, Singapore and CyberSG R&D Programme Ofice under its Translation and Innovation Grant (CRPO-GC4-SMU-002). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of the National Research Foundation and CyberSG R&D Programme Ofice.

## References

[1] Xiang Jia, Ying Tong, Hongming Qiao, Man Li, Jiangang Tong, and Baoling Liang. Fast and accurate object detector for autonomous driving based on improved yolov5. Scientific reports, 13(1):9711, 2023.

[2] S Cao. Review of object detection challenges in autonomous driving. Applied and Computational Engineering, 8(1):707–713, 2023.

[3] Yunsheng Deng, Aimin Qiao, Yinghui Huang, and Zhangbao Chen. Enhanced object detection for autonomous vehicles using modified faster r-cnn with attention and multi-scale feature fusion. Informatica, 49(30), 2025.

[4] Faisal Alamri. Comprehensive study on object detection for security and surveillance: A concise review. Multimedia Tools and Applications, 84(34):42321–42352, 2025.

[5] Zainab Ouardirhi, Sidi Ahmed Mahmoudi, and Mostapha Zbakh. Enhancing object detection in smart video surveillance: A survey of occlusion-handling approaches. Electronics, 13(3):541, 2024.

[6] Sani Abba, Ali Mohammed Bizi, Jeong-A Lee, Souley Bakouri, and Maria Liz Crespo. Real-time object detection, tracking, and monitoring framework for security surveillance systems. Heliyon, 10(15), 2024.

[7] Sara Dadjouy and Hedieh Sajedi. Gallbladder cancer detection in ultrasound images based on yolo and faster r-cnn. In 2024 10th International Conference on Artificial Intelligence and Robotics (QICAR), pages 227–231. IEEE, 2024.

[8] Carina Albuquerque, Roberto Henriques, and Mauro Castelli. Deep learning-based object detection algorithms in medical imaging: Systematic review. Heliyon, 11(1), 2025.

[9] Mohammadreza Saraei, Mehrshad Lalinia, and Eung-Joo Lee. Deep learning-based medical object detection: A survey. IEEE Access, 2025.

[10] Andrea Ceccarelli and Leonardo Montecchi. Evaluating object (mis) detection from a safety and reliability perspective: Discussion and measures. IEEE Access, 11:44952–44963, 2023.

[11] Shih-Han Chan, Yinpeng Dong, Jun Zhu, Xiaolu Zhang, and Jun Zhou. Baddet: Backdoor attacks on object detection. In Leonid Karlinsky, Tomer Michaeli, and Ko Nishino, editors, Computer Vision - ECCV 2022 Workshops - Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part I, volume 13801 of Lecture Notes in Computer Science, pages 396–412. Springer, 2022.

[12] Hangtao Zhang, Shengshan Hu, Yichen Wang, Leo Yu Zhang, Ziqi Zhou, Xianlong Wang, Yanjun Zhang, and Chao Chen. Detector collapse: Backdooring object detection to catastrophic overload or blindness in the physical world. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI 2024, Jeju, South Korea, August 3-9, 2024, pages 1670–1678. ijcai.org, 2024.

[13] Tianyu Gu, Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. Badnets: Evaluating backdooring attacks on deep neural networks. IEEE Access, 7:47230–47244, 2019.

[14] Xinyun Chen, Chang Liu, Bo Li, Kimberly Lu, and Dawn Song. Targeted backdoor attacks on deep learning systems using data poisoning. CoRR, abs/1712.05526, 2017.

[15] Yuezun Li, Yiming Li, Baoyuan Wu, Longkang Li, Ran He, and Siwei Lyu. Invisible backdoor attack with sample-specific triggers. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pages 16443–16452. IEEE, 2021.

[16] Tuan Anh Nguyen and Anh Tuan Tran. Wanet - imperceptible warping-based backdoor attack. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.

[17] Yi Zeng, Won Park, Z. Morley Mao, and Ruoxi Jia. Rethinking the backdoor attacks’ triggers: A frequency perspective. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pages 16453–16461. IEEE, 2021.

[18] Siyuan Cheng, Guangyu Shen, Guanhong Tao, Kaiyuan Zhang, Zhuo Zhang, Shengwei An, Xiangzhe Xu, Yingqi Li, Shiqing Ma, and Xiangyu Zhang. Odscan: Backdoor scanning for object detection models. In IEEE Symposium on Security and Privacy, SP 2024, San Francisco, CA, USA, May 19-23, 2024, pages 1703–1721. IEEE, 2024.

[19] Xianda Zhang, Siyuan Liang, and Chengyang Li. Towards robust object detection: Identifying and removing backdoors via module inconsistency analysis. In Apostolos Antonacopoulos, Subhasis Chaudhuri, Rama Chellappa, Cheng-Lin Liu, Saumik Bhattacharya, and Umapada Pal, editors, Pattern Recognition - 27th International Conference, ICPR 2024, Kolkata, India, December 1-5, 2024, Proceedings, Part XXIV, volume 15324 of Lecture Notes in Computer Science, pages 343–358. Springer, 2024.

[20] Antonio Torralba and Alexei A. Efros. Unbiased look at dataset bias. In The 24th IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2011, Colorado Springs, CO, USA, 20-25 June 2011, pages 1521–1528. IEEE Computer Society, 2011.

[21] Vardan Papyan, X. Y. Han, and David L. Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40):24652–24663, October 2020.

[22] Jianhua Lin. Divergence measures based on the shannon entropy. IEEE Trans. Inf. Theory, 37(1):145–151, 1991.

[23] Shaoqing Ren, Kaiming He, Ross B. Girshick, and Jian Sun. Faster R-CNN: towards real-time object detection with region proposal networks. IEEE Trans. Pattern Anal. Mach. Intell., 39(6):1137–1149, 2017.

[24] Joseph Redmon, Santosh Kumar Divvala, Ross B. Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016, pages 779–788. IEEE Computer Society, 2016.

[25] Bolun Wang, Yuanshun Yao, Shawn Shan, Huiying Li, Bimal Viswanath, Haitao Zheng, and Ben Y. Zhao. Neural cleanse: Identifying and mitigating backdoor attacks in neural networks. In 2019 IEEE Symposium on Security and Privacy, SP 2019, San Francisco, CA, USA, May 19-23, 2019, pages 707–723. IEEE, 2019.

[26] Yingqi Liu, Wen-Chuan Lee, Guanhong Tao, Shiqing Ma, Yousra Aafer, and Xiangyu Zhang. ABS: scanning neural networks for back-doors by artificial brain stimulation. In Lorenzo Cavallaro, Johannes Kinder, XiaoFeng Wang, and Jonathan Katz, editors, Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security, CCS 2019, London, UK, November 11-15, 2019, pages 1265–1282. ACM, 2019.

[27] Xiaojun Xu, Qi Wang, Huichen Li, Nikita Borisov, Carl A. Gunter, and Bo Li. Detecting AI trojans using meta neural analysis. In 42nd IEEE Symposium on Security and Privacy, SP 2021, San Francisco, CA, USA, 24-27 May 2021, pages 103–120. IEEE, 2021.

[28] Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. Fine-pruning: Defending against backdooring attacks on deep neural networks. In Michael D. Bailey, Thorsten Holz, Manolis Stamatogiannakis, and Sotiris Ioannidis, editors, Research in Attacks, Intrusions, and Defenses - 21st International Symposium, RAID 2018, Heraklion, Crete, Greece, September 10-12, 2018, Proceedings, volume 11050 of Lecture Notes in Computer Science, pages 273–294. Springer, 2018.

[29] Yige Li, Xixiang Lyu, Nodens Koren, Lingjuan Lyu, Bo Li, and Xingjun Ma. Neural attention distillation: Erasing backdoor triggers from deep neural networks. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.

[30] Wei Li, Pin-Yu Chen, Sijia Liu, and Ren Wang. PSBD: prediction shift uncertainty unlocks backdoor detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 10255–10264. Computer Vision Foundation / IEEE, 2025.

[31] Hangtao Zhang, Yichen Wang, Shihui Yan, Chenyu Zhu, Ziqi Zhou, Linshan Hou, Shengshan Hu, Minghui Li, Yanjun Zhang, and Leo Yu Zhang. Test-time backdoor detection for object detection models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 24377–24386. Computer Vision Foundation / IEEE, 2025.

[32] Dorde Popovic, Amin Sadeghi, Ting Yu, Sanjay Chawla, and Issa Khalil. Debackdoor: A deductive framework for detecting backdoor attacks on deep models with limited data. In Lujo Bauer and Giancarlo Pellegrino, editors, 34th USENIX Security Symposium, USENIX Security 2025, Seattle, WA, USA, August 13-15, 2025, pages 6419–6438. USENIX Association, 2025.

[33] TrojAI. Accessed: February 23, 2026.

[34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR, 2021.

[35] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 4904–4916. PMLR, 2021.

[36] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding DINO: marrying DINO with grounded pre-training for open-set object detection. In Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol, editors, Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XLVII, volume 15105 of Lecture Notes in Computer Science, pages 38–55. Springer, 2024.

[37] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. International journal of computer vision, 88(2):303–338, 2010.

[38] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.

[39] Glenn Jocher. YOLOv5 by Ultralytics, May 2020.

[40] Andrzej Maćkiewicz and Waldemar Ratajczak. Principal components analysis (pca). Computers & Geosciences, 19(3):303–342, 1993.