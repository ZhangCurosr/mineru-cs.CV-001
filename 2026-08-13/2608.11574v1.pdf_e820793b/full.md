# Hand Visibility Detector: Per-Keypoint Visibility Estimation for Hands

Ryosei Hara<sup>1,2</sup>, Masashi Hatano<sup>4</sup>, Rintaro Yanagi<sup>2</sup>, Atsushi Hashimoto<sup>3</sup>, Takuma Yagi<sup>2</sup>, Mariko Isogawa<sup>1,2</sup>

<sup>1</sup>Keio University, <sup>2</sup>National Institute of Advanced Industrial Science and Technology (AIST), <sup>3</sup>OMRON SINIC X Corporation, <sup>4</sup>The University of Tokyo

## Abstract

Hand Pose Estimation (HPE) is a fundamental technology for various applications such as AR/VR and robotics. In these applications, the visibility of each hand joint in the image is crucial for assessing the reliability of estimation results under occlusion. However, most existing HPE methods output joint positions without explicitly indicating their visibility. Although some methods account for occlusion or visibility, visibility estimation has mainly been used as an auxiliary signal for improving pose estimation. To our knowledge, per-joint hand visibility estimation has not been systematically studied as a standalone task. In this work, we propose Hand Visibility Detector, a model for estimating the visibility of individual hand joints, and present thefirst systematic investigation ofvisibility estimation as an independent task. We show that leveraging the prior knowledge of HPE models pretrained on large-scale data as a backbone yields high performance in this task. We further demonstrate the utility of Hand Visibility Detector on a downstream task of 3D hand pose annotation via multi-view triangulation of 2D keypoints, showing that visibility-weighted triangulation reduces reprojection error. Our method is released as a ready-to-use package, and the code and demo are available at https://github. com/ryhara/hand\_visibility\_detector.

## 1. Introduction

Hand poses play an important role in various fields, including AR/VR, human-computer interaction, and robotics [24]. Hand Pose Estimation (HPE) from images has advanced significantly with deep learning, and large-scale pretrained models that achieve highly accurate estimation even from monocular images have recently emerged, driven by Vision Transformers [5] and the scaling of training data [28, 29].

However, most existing HPE methods always output joint positions without explicitly indicating their visibility. Visibility refers to whether each joint is directly observable in the image, distinguishing observable joints from those whose locations must be inferred under occlusion. Several

![](images/6cf49f9e5190f30aa49351cf7c5835dabb854cbf6bcf4601b1350e0892776da2.jpg)  
Figure 1. Overview of Hand Visibility Detector. Our model estimates per-joint visibility as a confidence score in [0, 1]. Red indicates invisible joints, and green indicates visible joints.

HPE methods explicitly model occlusion or visibility to improve pose estimation [15, 27, 35, 36], as in other tasks such as point tracking [8, 14] and hand forecasting [9]. However, visibility is typically treated as an auxiliary signal, and the quality of the visibility estimation itself is not the primary object of study. More recently, Contact4D [33] has utilized per-joint visibility to refine triangulated 3D hand keypoints for occlusion-robust automatic annotation from multi-view images, indicating that visibility can be an informative cue in its own right, beyond its conventional role as an auxiliary component. However, because visibility is primarily evaluated indirectly through downstream pose accuracy, it remains unclear how accurately these models predict joint visibility itself and how well such predictions generalize across various occlusion conditions.

In this work, we propose Hand Visibility Detector, a model dedicated to hand joint visibility estimation. Our method adopts a simple architecture that attaches a lightweight visibility head to the frozen backbone of an HPE model [28, 29] pretrained on large-scale hand data. For training and evaluation, we use the HInt dataset [28], which provides manual per-joint visibility annotations on diverse in-the-wild data, including web images and egocentric video frames. Unlike prior work, which was trained and evaluated only on a single domain, the diversity of this dataset allows us to demonstrate the effectiveness of the prior knowledge of pretrained HPE models for visibility estimation under diverse conditions. Furthermore, to demonstrate its utility for automatic 3D annotation, we confirm that visibility-weighted triangulation of multi-view 2D keypoints reduces reprojection errors on three multiview datasets for 3D hand pose estimation, DexYCB [3], HO3D [6], and H2O [17].

Our contributions are summarized as follows:

• We formulate per-joint hand visibility estimation as a standalone task and introduce Hand Visibility Detector together with a systematic evaluation of this task.

• Our method outperforms baseline methods by 3.4 points in mAP on the HInt dataset, demonstrating the effectiveness of the prior knowledge of large-scale pretrained HPE models for visibility estimation.

• We evaluate visibility-weighted triangulation as a downstream task, reducing the mean reprojection error by up to 10.1% on DexYCB, HO3D, and H2O.

## 2. Related Work

## 2.1. Hand Pose Estimation

Recent HPE methods typically employ deep learning to estimate the parameters of the parametric MANO hand model [30]. Boukhayma et al. [2] pioneered direct regression of MANO parameters from a monocular image, followed by regression-based methods such as METRO [19] and Mesh Graphormer [18]. Occlusion-aware HPE has also been actively studied to handle occlusions caused by objects [20, 27, 35] and two-hand interactions [7]. More recently, Vision Transformers and the scaling of training data have led to large-scale pretrained models such as HaMeR [28] and WiLoR [29], which achieve remarkable accuracy even on in-the-wild images. In this work, we transfer the rich prior knowledge of these pretrained HPE models to a new task: hand joint visibility estimation.

## 2.2. Hand Joint Visibility Detection

While no prior work has addressed hand joint visibility estimation as an independent task, several methods estimate visibility in the course of HPE. Ye et al. [36] proposed a hierarchical mixture density network for hand pose estimation from depth images, which models per-joint visibility as a Bernoulli distribution and assigns a unimodal Gaussian distribution to visible joints and a multimodal Gaussian mixture to occluded ones. Kim et al. [15] introduced a per-joint visibility estimator into end-to-end estimation of two interacting hands and utilized its output for visibility-guided 2D joint heatmap enhancement. As non-per-joint approaches, HDR [22] recovers the occluded appearance of the target hand and removes the occluding hand via amodal segmentation and appearance recovery, and H2ONet [35] estimates finger-level occlusion probabilities to weight multi-frame feature aggregation. Since all of these methods are trained on datasets captured in controlled environments, their generalization ability as visibility estimators has not been validated. In this work, we use the HInt dataset [28], which provides manual per-joint visibility annotations on diverse in-the-wild data. To the best of our knowledge, this is the first systematic evaluation of the performance and generalization of visibility estimation itself.

## 2.3. Automatic 3D Hand Pose Annotation

Acquiring accurate 3D annotations is essential for 3D human body and hand pose estimation research [24]. Optical markers [26] and glove-based motion capture [16] are accurate but require dedicated equipment, and the attached devices compromise the natural appearance of hands. Triangulation from multi-view images [3, 6, 17, 25] is therefore widely used, but its accuracy depends on the 2D HPE accuracy in each view. Standard triangulation treats all views equally, and even when outliers are removed with RANSAC [23, 34], the remaining low-confidence points are used as they are. Contact4D [33] achieves occlusion-robust annotation by weighting triangulation with the detection confidence of a hand detector [29] and refining the triangulated 3D joint positions using a visibility estimator, based on RTMPose [12] (with a CSPNeXt [4] backbone) and trained on COCO-WholeBody [13]. However, the performance of this visibility estimator itself has not been validated. Moreover, we observed that the hand visibility labels in COCO-WholeBody are often inaccurate or missing, which motivates us to use a higher-quality label source instead. In this work, we train a visibility estimator on the HInt dataset and demonstrate that visibility-weighted triangulation using our Hand Visibility Detector reduces reprojection errors on three multi-view hand datasets, DexYCB [3], HO3D [6], and H2O [17], compared to unweighted and hand detection confidence-weighted baselines.

## 3. Method

Fig. 2 shows an overview of the proposed method. Given a hand-cropped image $I \in \mathbb { R } ^ { H \times \bar { W } \times 3 }$ , our method estimates per-joint visibility $V = ( \widehat { v } _ { 1 } , \ldots , \widehat { v } _ { J } ) \in [ 0 , 1 ] ^ { J }$ We use ground-truth boxes provided by the datasets for training and evaluation, and boxes obtained by the WiLoR [29] hand detector in the downstream task. The joint layout follows the 21 keypoints of the MANO hand model [30] (J = 21), and each $\hat { v } _ { j }$ represents the probability that joint j is visible. Our method consists of (i) a frozen ViT backbone of a pretrained HPE model and (ii) a lightweight visibility head.

![](images/17b7ee8634814dd2207b5e11dfdb5a14f58cd54dd88a9405bddd909b89c9490c.jpg)  
Figure 2. Overview of the proposed method. (a) Overall pipeline. Our model outputs only per-joint visibility, shown as green (visible) and red (invisible). Joint positions are provided by ground truth or an off-the-shelf HPE model for visualization only. (b) Architecture of the Visibility Head. A frozen backbone of a pretrained HPE model extracts features, and only the lightweight head is trained.

## 3.1. Hand Encoder

Since occluded joints lack direct visual evidence at their location, determining their visibility requires reasoning about the entire hand structure, such as its spatial relationships with visible fingers and potential occluders. To perform such reasoning across diverse scenes, it is desirable to leverage rich prior knowledge of hand structure, rather than training a feature extractor from scratch on the limited visibility labels of a specific dataset as in prior work [15, 36]. Large-scale pretrained HPE models such as HaMeR [28] and WiLoR [29] have learned to recover 3D poses from millions of diverse hand images under various conditions, including occlusion, and are thus expected to have acquired exactly the prior knowledge of hand structure necessary for reasoning about occluded joints.

We therefore adopt the pretrained ViT backbones of HaMeR and WiLoR as our Hand Encoder. Given the input image I, the encoder produces a feature map $F \in \mathbb { R } ^ { h \times w \times C }$ by rearranging the output tokens into their spatial layout. The backbone parameters are frozen at their pretrained values. This preserves the prior knowledge of the large-scale pretrained model while avoiding overfitting to the relatively small visibility-labeled data.

## 3.2. Visibility Head

To estimate per-joint visibility from the feature map $F ,$ we employ a lightweight visibility head adopting the design of the head architecture of RTMPose [12], which is also adopted by the visibility estimator of Contact4D [33]. Specifically, a 1 × 1 convolution first compresses the feature map from $C$ to d dimensions, which is then flattened into a sequence of $h \times w$ tokens. After a fully connected layer, a Gated Attention Unit (GAU) [11] models global dependencies among spatial positions. GAU is an attention mechanism that integrates self-attention with gated linear units, and its effectiveness for keypoint estimation has been shown in RTMPose. Finally, a 1×1 convolution projects the features into J channels, spatial average pooling produces per-joint logits, and a sigmoid function converts them into visibility probabilities $\hat { v } _ { j }$

We train the head with the binary cross-entropy loss against ground-truth visibility labels $v _ { j } \in \{ 0 , 1 \}$ :

$$
\mathcal { L } = - \frac { 1 } { J } \sum _ { j = 1 } ^ { J } [ v _ { j } \log \hat { v } _ { j } + ( 1 - v _ { j } ) \log ( 1 - \hat { v } _ { j } ) ] ,\tag{1}
$$

where joints outside the frame are treated as invisible $( v _ { j } =$ 0). That is, visibility in this work means that a joint is neither occluded nor outside the frame. We freeze the backbone and train only the visibility head, allowing visibility estimation to be learned at an extremely low cost while preserving the prior knowledge of the pretrained HPE model.

## 4. Experiments

Dataset. We train and evaluate on the HInt dataset [28], which provides manually annotated keypoints with per-joint visibility labels. HInt covers diverse in-the-wild scenes, including web and egocentric videos, and we use 25,273 frames for training and 5,374 for evaluation.

Baseline Methods. We compare with the following two visibility estimators from existing methods capable of perjoint visibility estimation. The visibility estimator of Kim et al. [15] consists of a ResNet-50 [10] pretrained on ImageNet-1k [31] and a two-layer linear head. The visibility estimator of Contact4D [33] follows RTMPose [12] and uses a CSPNeXt [4] backbone pretrained on ImageNet-1k. Evaluation Metrics. We evaluate visibility estimation as a per-joint binary classification problem. We use mAP (average precision averaged over joints) and F1 score, where F1 is computed by binarizing predictions at a threshold of 0.5. All experiments are run with three seeds, and we report the mean and standard deviation.

Implementation Details. The input image is obtained by expanding the ground-truth hand bounding box by a factor of 1.25, resizing it to 256 × 256, and cropping the central $H \times W = 2 5 6 \times 1 9 2$ region following WiLoR. The encoder divides the input into $1 6 \times 1 6$ patches, yielding a feature map of $h = 1 6 , w = 1 2$ , and $C = 1 2 8 0$ . The hidden dimension of the visibility head is $d = 2 5 6$ with a dropout rate of 0.1. Only the visibility head is trained, and its 0.83M parameters account for merely 0.131% of the 631M-parameter model. We use AdamW [21] with a learning rate of $1 . 0 \times 1 0 ^ { - 3 }$ . We train for 100 epochs with a batch size of 256, which takes about 2.5 hours on a single NVIDIA H200 GPU using approximately 10GB of GPU memory. All experiments are conducted under the same settings. For the backbone ablation, each backbone is initialized with its official pretrained checkpoint and kept frozen. Please refer to our released code for further details.

Table 1. Comparison with baseline methods on HInt.
<table><tr><td>Method</td><td> $\mathrm { m A P } \left( \uparrow \right)$  F1 (↑)</td></tr><tr><td>Kim et al. [15]  $0 . 8 9 5 \pm 0 . 0 0 1$ </td><td> $0 . 8 5 8 \pm 0 . 0 0 1$ </td></tr><tr><td>Contact4D [33]</td><td> $0 . 8 9 7 \pm 0 . 0 0 1$ </td></tr><tr><td>Ours  $\mathbf { 0 . 9 3 1 } \pm \mathrm { 0 . 0 0 0 }$ </td><td> $0 . 8 6 0 \pm 0 . 0 0 2$   $\mathbf { 0 . 8 9 6 \pm 0 . 0 0 1 }$ </td></tr></table>

Table 2. Ablation on backbones. The visibility head and hyperparameters are shared across all models.
<table><tr><td>Backbone</td><td>#Params</td><td> $\mathrm { m A P } \left( \uparrow \right)$ </td><td>F1 (↑)</td></tr><tr><td>CSPNeXt-X [4]</td><td>48M</td><td> $0 . 8 0 0 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 7 9 9 { \scriptstyle \pm 0 . 0 0 1 }$ </td></tr><tr><td>ResNet-152 [10]</td><td>58M</td><td> $0 . 7 9 6 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 7 9 7 { \scriptstyle \pm 0 . 0 0 1 }$ </td></tr><tr><td>ViT-H [5]</td><td>633M</td><td> $0 . 8 3 8 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 8 1 9 { \scriptstyle \pm 0 . 0 0 4 }$ </td></tr><tr><td>DINOv3 [32]</td><td>840M</td><td> $0 . 8 9 7 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 8 6 6 \pm 0 . 0 0 0$ </td></tr><tr><td>HaMeR [28]</td><td>631M</td><td> $0 . 9 3 2 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 8 9 6 \pm 0 . 0 0 0$ </td></tr><tr><td>WiLoR [29]</td><td>631M</td><td> $0 . 9 3 1 \pm 0 . 0 0 0$ </td><td> $0 . 8 9 6 \pm 0 . 0 0 1$ </td></tr><tr><td>WiLoR (fine-tuned) [29]</td><td>631M</td><td> $0 . 6 2 2 \ \pm 0 . 0 1 4$ </td><td> $0 . 7 0 4 ~ \pm 0 . 0 1 8$ </td></tr></table>

## 5. Results

## 5.1. Visibility Estimation

Comparison with Baseline Methods. Tab. 1 shows the comparison with existing methods capable of visibility estimation. Our method achieves an mAP of 0.931 and an F1 of 0.896, substantially outperforming Kim et al. [15] and Contact4D [33] on both metrics. Both baselines rely on CNN features pretrained on ImageNet-1k, and this gap demonstrates the effectiveness of the hand prior knowledge of pretrained HPE models.

Choice of Backbone. Tab. 2 shows the visibility estimation performance of each backbone. The hand-specific models HaMeR and WiLoR perform best, and even DINOv3, the strongest among recent general-purpose image models, falls short of them. Moreover, fine-tuning the WiLoR backbone for visibility estimation degrades mAP from 0.931 to 0.622. This indicates that task-specific fine-tuning corrupts the feature representations acquired during pre-training, supporting our design of freezing the backbone and training only the head.

Table 3. Ablation on the visibility head. The backbone is fixed to our frozen WiLoR encoder.
<table><tr><td>Head</td><td>mAP (↑)</td><td>F1 (↑)</td></tr><tr><td>w/ Kim et al. head [15]</td><td> $0 . 9 0 5 \pm 0 . 0 0 0$ </td><td> $0 . 8 7 4 \pm 0 . 0 0 1$ </td></tr><tr><td>w/o GAU</td><td> $0 . 8 8 7 \pm 0 . 0 0 0$ </td><td> $0 . 8 6 0 \pm 0 . 0 0 0$ </td></tr><tr><td>Ours (full)</td><td> $\mathbf { 0 . 9 3 1 } \pm \mathrm { 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 8 9 6 \pm 0 . 0 0 1 }$ </td></tr></table>

![](images/80391b53e445af5f0437b64fd9aa73281434668cb3baedb6c929a6fe3319a22f.jpg)  
Figure 3. F1 score versus binarization threshold. Performance peaks around 0.5 and remains stable over a wide range.

Design of Visibility Head. Tab. 3 shows the ablation on the visibility head with the backbone fixed to WiLoR. Removing the GAU degrades mAP from 0.931 to 0.887, and replacing our head with the linear head of Kim et al. [15] also falls short of ours. This demonstrates that the GAU, which models global dependencies among spatial positions, is effective for visibility estimation.

Visibility Threshold Sensitivity. Fig. 3 shows the F1 score as the binarization threshold varies from 0.1 to 0.9. F1 peaks around a threshold of 0.5 and stays above 0.88 over the wide range of 0.3–0.7, indicating that performance is relatively insensitive to the threshold.

Qualitative Evaluation. Fig. 4 shows qualitative comparisons with the baselines and the backbone ablation. Our method correctly estimates visibility with the highest confidence in all cases of self-occlusion, image truncation, and occlusion by objects.

## 5.2. Downstream Task: Triangulation

Settings. As a downstream task, we conduct a triangulation experiment assuming automatic 3D hand pose annotation from multi-view images. We triangulate 2D keypoints estimated by WiLoR [29] in each view using DLT [1], and compare three weighting schemes: (i) unweighted triangulation (Baseline), (ii) weighting by the WiLoR detector confidence, shared by all joints in a view [33] (Detection conf. weighted), and (iii) weighting by the per-joint visibility estimated by our method (Visibility weighted). We evaluate reprojection errors (px) on the test sets of three datasets: DexYCB [3] (12,902 frames, 8 views, 640 × 480), HO3D [6] (4,854 frames, 2 or 5 views, 640 × 480), and

![](images/834fafc27944bfc7f50d457cd44d60bd697bfb2f33134996830e923d5ffbba02.jpg)  
Input

![](images/0a9d41d13b668b2d3b971fe9c699ee932f5ecdb9713cfb17280deed176ef3971.jpg)  
Kim et al.

![](images/b69136b8ce06a7bcc2a354430b43f742247f9df1efedb502ee5e23d7ad3a41fb.jpg)  
Contact4D

![](images/182d627a2a9a53ded60f3bdf740b6ca47768e03016268057099c9b1543c9404c.jpg)  
DINOv3

![](images/b7939202f5b5d528000b14bf618b9b843cb6aeffd029500190bef7dabbf0eb44.jpg)  
Ours

![](images/774b987d1183c7e4bb53c39369397404b8e69226ce79be4393781166c4e50e8b.jpg)

![](images/f5a357ba13e0c29c0558f9aef1d84cbb5880f196102ed06babdd969e7e0fcfc6.jpg)  
GT

Figure 4. Qualitative results of visibility estimation. Joint positions are ground truth. Green indicates joints predicted as visible, and red indicates joints predicted as invisible. From top to bottom: self-occlusion, image truncation, and occlusion by objects.  
![](images/f272a8e9a2f519a8907e1b0a67d3dbb7b8e79ec6e0c02cd7c81dd3d0550fe2c8.jpg)  
Figure 5. Reprojection errors of triangulation on each dataset.

H2O [17] (23,391 frames, 4 views, 1280 × 720).

Evaluation. Fig. 5 shows the results. Visibility-weighted triangulation achieves lower median, mean, and interquartile range than both the baseline and detection-confidence weighting [33] on all three datasets. The improvement is largest on HO3D, which has fewer views and severe occlusion by objects, where the mean reprojection error is reduced by 10.1%. These results indicate that per-joint visibility weighting effectively suppresses low-quality 2D estimates of occluded joints.

Fig. 6 shows qualitative comparisons for triangulation. By emphasizing views where each joint is visible, our weighting suppresses 2D estimates displaced by occlusion, reducing reprojection errors.

## 6. Conclusion

In this work, we proposed Hand Visibility Detector, the first model dedicated to hand joint visibility estimation. Leveraging the prior knowledge of large-scale pretrained

![](images/cbb4aeedcf6748007debaea0ce044b42e1acbf141f656bbc9e6fd17ff4ca3a9b.jpg)

![](images/3956b8b2189b62089238afc59f74ac343e769d4c2fa5847bbb13698154112139.jpg)  
Figure 6. Qualitative comparison of triangulation results. Triangulated 3D joints are reprojected onto each view. Red circles highlight regions where visibility weighting improves the most.

HPE models, our method substantially outperforms existing methods and general-purpose backbones on the HInt dataset. Furthermore, visibility-weighted triangulation reduces reprojection errors on multiple datasets. Future work includes extending our method to video input for temporally consistent visibility estimation. Our code is released as a ready-to-use package, and we hope it will be utilized in various downstream tasks.

Acknowledgment. This work was supported by JST K Program JPMJKP25V1.

## References

[1] Y.I. Abdel-Aziz, H.M. Karara, and Michael Hauck. Direct Linear Transformation from Comparator Coordinates into Object Space Coordinates in Close-Range Photogrammetry. Photogrammetric Engineering & Remote Sensing, 81 (2):103–107, 2015. 4

[2] Adnane Boukhayma, Rodrigo de Bem, and Philip H.S. Torr. 3D Hand Shape and Pose From Images in the Wild. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10835–10844, 2019. 2

[3] Yu-Wei Chao, Wei Yang, Yu Xiang, Pavlo Molchanov, Ankur Handa, Jonathan Tremblay, Yashraj S. Narang, Karl Van Wyk, Umar Iqbal, Stan Birchfield, Jan Kautz, and Dieter Fox. DexYCB: A Benchmark for Capturing Hand Grasping of Objects. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9044–9053, 2021. 2, 4

[4] Xiangqi Chen, Chengzhuan Yang, Jiashuaizi Mo, Yaxin Sun, Hicham Karmouni, Yunliang Jiang, and Zhonglong Zheng. CSPNeXt: A new efficient token hybrid backbone. Engineering Applications of Artificial Intelligence, 132:107886, 2024. 2, 3, 4

[5] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In International Conference on Learning Representations (ICLR), 2021. 1, 4

[6] Shreyas Hampali, Mahdi Rad, Markus Oberweger, and Vincent Lepetit. HOnnotate: A Method for 3D Annotation of Hand and Object Poses. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3193– 3203, 2020. 2, 4

[7] Shreyas Hampali, Sayan Deb Sarkar, Mahdi Rad, and Vincent Lepetit. Keypoint Transformer: Solving Joint Identification in Challenging Hands and Object Interactions for Accurate 3D Pose Estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11080– 11090, 2022. 2

[8] Adam W Harley, Zhaoyuan Fang, and Katerina Fragkiadaki. Particle video revisited: Tracking through occlusions using point trajectories. In ECCV, pages 59–75, 2022. 1

[9] Masashi Hatano, Zhifan Zhu, Hideo Saito, and Dima Damen. The invisible egohand: 3d hand forecasting through egobody pose estimation. arXiv preprint arXiv:2504.08654, 2025. 1

[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016. 3, 4

[11] Weizhe Hua, Zihang Dai, Hanxiao Liu, and Quoc Le. Transformer Quality in Linear Time. In International Conference on Machine Learning (ICML), pages 9099–9117, 2022. 3

[12] Tao Jiang, Peng Lu, Li Zhang, Ningsheng Ma, Rui Han, Chengqi Lyu, Yining Li, and Kai Chen. RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose. In arXiv preprint arXiv:2303.07399, 2023. 2, 3

[13] Sheng Jin, Lumin Xu, Jin Xu, Can Wang, Wentao Liu, Chen Qian, Wanli Ouyang, and Ping Luo. Whole-Body Human Pose Estimation in the Wild. In European Conference on Computer Vision (ECCV), page 196–214, 2020. 2

[14] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. In ECCV, pages 18–35, 2024. 1

[15] Dong Uk Kim, Kwang In Kim, and Seungryul Baek. Endto-End Detection and Pose Estimation of Two Interacting Hands. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 11169–11178, 2021. 1, 2, 3, 4

[16] Jeonghwan Kim, Jisoo Kim, Jeonghyeon Na, and Hanbyul Joo. ParaHome: Parameterizing Everyday Home Activities Towards 3D Generative Modeling of Human-Object Interactions . In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1816–1828, 2025. 2

[17] Taein Kwon, Bugra Tekin, Jan Stuhmer, Federica Bogo, and Marc Pollefeys. H2O: Two Hands Manipulating Objects for First Person Interaction Recognition . In IEEE/CVF International Conference on Computer Vision (ICCV), pages 10118–10128, 2021. 2, 5

[18] Kevin Lin, Lijuan Wang, and Zicheng Liu. Mesh Graphormer. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 12919–12928, 2021. 2

[19] Kevin Lin, Lijuan Wang, and Zicheng Liu. End-to-End Human Pose and Mesh Reconstruction with Transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1954–1963, 2021. 2

[20] Zhifeng Lin, Changxing Ding, Huan Yao, Zengsheng Kuang, and Shaoli Huang. Harmonious Feature Learning for Interactive Hand-Object Pose Estimation. In IEEE/CVF Confer ence on Computer Vision and Pattern Recognition (CVPR), pages 12989–12998, 2023. 2

[21] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2017. 4

[22] Hao Meng, Sheng Jin, Wentao Liu, Chen Qian, Mengx iang Lin, Wanli Ouyang, and Ping Luo. 3D Interacting Hand Pose Estimation by Hand De-occlusion and Removal. In European Conference on Computer Vision (ECCV), page 380–397, 2022. 2

[23] Gyeongsik Moon, Shoou-I Yu, He Wen, Takaaki Shiratori, and Kyoung Mu Lee. InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image. In European Conference on Computer Vision (ECCV), page 548–564, 2020. 2

[24] Takehiko Ohkawa, Ryosuke Furuta, and Yoichi Sato. Efficient Annotation and Learning for 3D Hand Pose Estimation: A Survey. International Journal ofComputer Vision (IJCV), 131(12):3193–3206, 2023. 1, 2

[25] Takehiko Ohkawa, Kun He, Fadime Sener, Tomas Hodan, Luan Tran, and Cem Keskin. AssemblyHands: Towards Egocentric Activity Understanding via 3D Hand Pose Estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12999–13008, 2023. 2

[26] OptiTrack. Motive. https://optitrack.com/ software/motive. 2

[27] JoonKyu Park, Yeonguk Oh, Gyeongsik Moon, Hongsuk Choi, and Kyoung Mu Lee. HandOccNet: Occlusion-Robust 3D Hand Mesh Estimation Network. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1486–1495, 2022. 1, 2

[28] Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David Fouhey, and Jitendra Malik. Reconstructing Hands in 3D with Transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9826–9836, 2024. 1, 2, 3, 4

[29] Rolandos Alexandros Potamias, Jinglei Zhang, Jiankang Deng, and Stefanos Zafeiriou. WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12242–12254, 2025. 1, 2, 3, 4

[30] Javier Romero, Dimitrios Tzionas, and Michael J. Black. Embodied Hands: Modeling and Capturing Hands and Bodies Together. ACM Transactions on Graphics, 36(6):245:1– 245:17, 2017. 2

[31] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal ofComputer Vision (IJCV), 115 (3):211–252, 2015. 3

[32] Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Fed-´ erico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seung Eun Yi, Michael Ramamonjisoa, Francisco Massa, Daniel HAZIZA, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, Th´ eo Moutakanni, Leonel´ Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herve Jegou, Patrick Labatut, and Piotr Bojanowski. DINOv3. Transactions on Machine Learning Research, 2026. 4

[33] Jyun-Ting Song, JungEun Kim, Jinkun Cao, Yu Lei, Takuma Yagi, and Kris Kitani. Contact4D: A Video Dataset for Whole-Body Human Motion and Finger Contact in Dexterous Operations. In International Conference on 3D Vision (3DV), pages 904–914, 2026. 1, 2, 3, 4, 5

[34] Jikai Wang, Qifan Zhang, Yu-Wei Chao, Bowen Wen, Xiaohu Guo, and Yu Xiang. HO-Cap: A Capture System and Dataset for 3D Reconstruction and Pose Tracking of Hand-Object Interaction. In Conference on Neural Information Processing Systems (NeurIPS), 2025. 2

[35] Hao Xu, Tianyu Wang, Xiao Tang, and Chi-Wing Fu. H2ONet: Hand-Occlusion-and-Orientation-Aware Network for Real-Time 3D Hand Mesh Reconstruction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17048–17058, 2023. 1, 2

[36] Qi Ye and Tae-Kyun Kim. Occlusion-aware Hand Pose Estimation Using Hierarchical Mixture Density Network. In European Conference on Computer Vision (ECCV), pages 817–834, 2018. 1, 2, 3