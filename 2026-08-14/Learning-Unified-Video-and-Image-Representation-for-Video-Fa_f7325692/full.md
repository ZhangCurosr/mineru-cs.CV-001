# Learning Unified Video and Image Representation for Video Face Forgery Detection

Haotian Liu, Yang Liu, Guoying Zhao, and Xiaobai Li

Abstract—Face forgery detection is crucial for preserving the security and integrity of facial data given the rapid developments in face manipulation techniques and deep generative models. Existing methods for video face forgery detection typically assume that all frames in a forged video are manipulated, while detecting partially forged videos that contain only a subset of altered frames remains challenging. To address this issue, we propose a novel framework, UVIF, that utilizes additional annotated images to provide fine-grained supervision for detecting partial forgeries in videos. UVIF employs a unified encoder and a multi-task learning paradigm to jointly model facial videos and images for boosted video face forgery detection. A 2D backbone with temporal fusion modules is employed as the unified encoder. A pseudo labeling process is designed for video frames to bridge their representations with those of static images. A video-oriented feature alignment strategy is further introduced to reduce the distribution gap between videos and images. Extensive experiments on benchmark datasets demonstrate the effectiveness of our framework, which outperforms state-of-theart methods in detecting partially forged videos while introducing no additional computational overhead. Our code is available at https://github.com/haotianll/UVIF.

Index Terms—Face Forgery Detection, Unified Representation.

## I. INTRODUCTION

Face forgery detection aims to distinguish between authentic and fabricated faces [1]. This task is essential in preventing malicious uses of face manipulation techniques and AIgenerated content (AIGC) [2], [3], thereby upholding the security and integrity of facial data.

Current research on video forgery detection is primarily categorized into image-based and video-based methods. Image-based methods [4]–[13] utilize static facial images as inputs and perform image classification to verify their authenticity. When dealing with a video clip, these methods extract individual facial frames and assign classification labels to each frame. In contrast, video-based methods [14]–[25] process facial video clips directly using 3D backbones and perform video classification. Given the importance of temporal inconsistencies across video frames for forgery analysis, video-based methods [14]–[25] achieved higher accuracy than image-based methods.

A significant limitation of existing face forgery detection studies is the assumption [5] that all frames in a fake video are manipulated, which underlies the preprocessing pipelines, model architectures, and training strategies employed. However, this assumption does not hold for all face forgery detection tasks in realistic scenarios, as some fake videos may contain only a subset of manipulated frames. This discrepancy poses challenges for both image-based and video-based detection methods, as illustrated in Figure 1. Video-based methods lack frame-level supervision and therefore require tailored strategies to handle mixtures of authentic and forged frames. Image-based methods instead assign the fake video label to every extracted frame, including authentic ones, which introduces label noise during image classification training.

![](images/728479b7a3ea3b0d7f559029a9af125cbd0ee95b94a79c7a7602e82208c19fad.jpg)  
(a) Fully forged videos fit frame-based classification with video labels

![](images/481f125fab0b75efe8ff852763c9155a4a4a720fe3c8bc9b0c8ec748d750c691.jpg)  
(b) Partially forged videos cause weak supervision or noisy frame labels

![](images/5a261426e2eea0ea666e47c3c8b80de8c8ee5b5c2df7e8d46116a2eda5c5a90f.jpg)  
(c) UVIF unifies video and image representations (Ours)  
Fig. 1. (a) Fully forged video, the main focus of previous methods, where every frame in a fake video is manipulated and the video label can be assigned to each frame. (b) Partially forged video, where only a subset of frames is manipulated. Video-based methods lack frame-level supervision, while image-based methods assign the fake label to all frames, including real ones, introducing label noise during classification training. (c) The proposed UVIF uses a set of annotated images to provide fine-grained supervision for videos and learns unified representations.

Some previous work [26] tried to use multiple instance learning (MIL) [27], [28] to detect partially forged videos. Specifically, MIL uses a set of labeled bags containing multiple instances for training. For binary classification, a positive bag may contain both positive and negative instances, while a negative bag contains only negative ones. MIL matches the setting of partially forged video detection, where a forged video can be treated as a positive bag. However, current MIL methods [26]–[31] group instances based on feature similarity without access to instance-level labels during training, limiting their effectiveness for partially forged video detection.

This limitation motivates the use of frame-level supervision, as detecting forged frames is essential for verifying the authenticity of facial videos. Knowledge from detecting forged images can also contribute to this, as they have a similar representation learning process, i.e., extracting discriminative representations of forgery cues. Facial images with finegrained labels are readily available in existing face forgery detection datasets. Therefore, we propose UVIF, i.e., Unified Video and Image representation for face Forgery detection. UVIF processes video and image inputs within a single model and jointly addresses both tasks in a multi-task paradigm, as shown in Figure 1. Image forgery detection serves as an auxiliary task that provides fine-grained supervision, aiming to improve representation learning for partially forged videos.

## The contributions of this paper include:

1) We propose UVIF, a unified framework that utilizes annotated images to provide fine-grained supervision for detecting partially forged videos. It uses a shared backbone and temporal fusion modules to process video and image inputs.

2) A pseudo labeling process is designed to compensate for the absence of frame-level supervision. It generates framelevel pseudo labels from image classifier to bridge the representation of video frames and static images.

3) A video-oriented feature alignment strategy is introduced to reduce the distribution gap between images and video frames. It performs feature interpolation using image anchors to adapt the image classifier to video-frame features.

4) Extensive experiments show that UVIF outperforms stateof-the-art methods in partially forged video detection. Ablation studies further demonstrate its data efficiency, with 10k annotated images sufficient to achieve a significant performance boost, as well as its effectiveness across video and image datasets with distinct manipulation types.

This paper extends our previous study presented at ECAI 2024 [32] through the following substantial improvements: 1) We provide a theoretical analysis of existing image-based and video-based methods for partially forged video detection and refine the methodology with clearer notation. 2) We introduce a new video-oriented feature alignment strategy to reduce the distribution gap between video and image datasets containing different manipulation types. 3) We conduct a more comprehensive evaluation of UVIF, including additional ablation studies, backbones, datasets, and analyses.

## II. RELATED WORK

## A. Face Forgery Detection

Face forgery detection aims to detect forged or synthesized faces in images and videos, which is crucial for the authenticity of visual information that we see every day. Early studies [4], [5] formulated face forgery detection as an image binary classification task and employed convolutional neural networks (CNNs) to process facial images. Subsequent imagebased methods further exploited spatial forgery cues, including local textures [6], frequency domain [7], and inconsistency information [8], [9]. Recent studies [10]–[13] have also focused on improving robustness and generalization to unseen manipulation methods. It is worth noting that some imagebased methods [7]–[9] have been extended to video face forgery detection. They converted facial videos to individual image frames and assigned a classification label to each frame during training. However, these methods assumed that all frames in a facial video are the same kind, i.e., all as real, or all as fake, making them unsuitable to process partially forged videos.

As temporal inconsistencies across video frames provide important artifact cues, many video-based methods [14]–[25] have been proposed for face forgery detection. These methods directly process facial video sequences to model temporal dependencies. Early studies focused on mining temporal clues by using prior knowledge, such as eye blinks [14], lip motions [15], and biological signals [16]. Some methods [17], [19]–[22] directly processed facial video clips using CNNor transformer-based architectures to learn spatiotemporal representations, achieving better accuracy than image-based methods. Recent works [23]–[25] have focused on modeling inter-frame relationships to learn discriminative temporal representations. Despite their swift progress, partially forged video detection still requires further exploration.

## B. Multiple Instance Learning

Multiple instance learning (MIL) [27], [28] is a form of weakly supervised learning with broad applications in medical imaging and video analysis. In MIL, a bag of instances is annotated with a single bag-level label, while the exact label for each instance is unavailable. For a binary classification, negative bags only contain negative instances, while positive bags can contain both positive and negative instances. Early studies [27] focused on extracting and aggregating features of each instance in a bag via deep neural networks and pooling operations. Recent approaches [28]–[31] have also been developed based on attention mechanisms and transformers. Li et al. [26] proposed a sharp MIL, i.e., S-MIL, for face forgery detection to handle the problem of partially manipulated faces in videos. It treated facial frames and videos as instances and bags in MIL, respectively, and designed a sharp loss emphasizing hard instances to address the partially forged videos. Nevertheless, the performance of MIL methods is limited due to the lack of instance-level labels during training. In this paper, we propose to address the partially forged video detection task from a new perspective, i.e., by incorporating additional image instances with annotations to compensate for the absence of fine-grained supervision information.

## C. Unified Architecture Design

The unified architecture design [33], [34] has gained significant attention recently. It can process input data from different modalities or perform multiple tasks using a single model. As video and image data are related in structure, i.e., a video can be viewed as a sequence of images, some methods [35], [36] have introduced temporal modeling into CNN or transformer backbones originally designed for 2D images and adapted them to video tasks. Other methods [37]–[39] investigated using image data and pretrained knowledge to assist video tasks and enhance video representation learning. Given the strong generalization capability of transformers, some methods [33], [40] used transformers to encode or align data from multiple modalities, such as image, video, audio, and text, into a unified feature space. Due to diverse training data and the scalability of transformers, these methods can learn effective feature representations across modalities. The unified architecture design has demonstrated promising results in other visual tasks, which inspires us to apply it to partially forged video detection by incorporating annotated images.

![](images/ee95cc40da1e0166f4bd944ab2cc8bc53cf5c6e88ea91de6f765c2c665013a79.jpg)  
Fig. 2. Overview of the proposed UVIF framework, which comprises three components: 1) unified video and image modeling, where a unified encoder extracts features from both facial videos and images, with temporal fusion applied only to videos, under a multi-task learning paradigm, 2) frame-level pseudo labeling process transfers fine-grained image supervision to video frames, and 3) video-oriented feature alignment interpolates features to reduce the distribution gap between images and video frames. The image head is shared across all three components. During inference, only the unified encoder and video head are used

## III. PRELIMINARY

This section presents the problem formulation and analyzes the limitations of existing video forgery detection methods.

## A. Problem Formulation

This paper focuses on video face forgery detection, formulated as a binary classification task that determines whether a video clip contains fake faces.

Let $V ~ = ~ \{ v _ { t } \} _ { t = 1 } ^ { T }$ represent a facial video clip, which consists of a sequence of video frames $v _ { t } .$ and $T$ is the number of frames. $Y$ represents the binary classification label of the entire video, where $Y \in \{ 0 , 1 \}$ denotes whether the video is real or fake, respectively. The objective of video face forgery detection is to learn a model that accurately predicts the binary label for a given video.

For a real video clip, every frame is genuine, whereas the video is labeled as fake if one or more frames are manipulated. Assuming that there are binary frame labels $\mathbf { y } _ { f } = \{ y _ { t } \} _ { t = 1 } ^ { T }$ for each video frame, where $y _ { t } \in \{ 0 , 1 \}$ , for $t = 1 , 2 , . . . , T$ , this premise can be described as:

$$
Y = { \left\{ \begin{array} { l l } { 1 , } & { \exists y _ { t } = 1 , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{1}
$$

This formulation corresponds to the setting of multiple instance learning (MIL) [27], [28]. Specifically, the video frames $V = \{ v _ { t } \} _ { t = 1 } ^ { T }$ can be treated as a bag of instances in MIL, and Y serves as the bag-level label. During training, only the label Y is used, while the individual labels $\mathbf { y } _ { f } = \bar { \{ { y } _ { t } \} } _ { t = 1 } ^ { T }$ of each frame are not available.

## B. Limitation of Existing Methods

Most previous studies assume that all frames in a fake video are manipulated. However, this assumption does not hold for partially forged videos, and applying these methods can lead to unreliable results.

1) Image-based Methods: Given a video $V = \{ v _ { t } \} _ { t = 1 } ^ { T }$ with a video-level label Y, image-based methods treat each frame as an individual image for binary classification and construct a frame-level training set by assigning $Y$ to every frame:

$$
\widetilde { \cal D } _ { f } = \{ ( v _ { t } , Y ) \} _ { t = 1 } ^ { T } .\tag{2}
$$

The correct frame-level training set is instead given by:

$$
\mathcal { D } _ { f } = \{ ( v _ { t } , y _ { t } ) \} _ { t = 1 } ^ { T } .\tag{3}
$$

When a video is real or fully forged, each constructed label is consistent with the true frame-level label, i.e., $Y ~ = ~ y _ { t }$ However, for a partially forged video, Eq. 1 implies that some real frames satisfy $y _ { t } ~ = ~ 0$ while $Y ~ = ~ 1$ . Therefore, the constructed frame training set differs from the correct one:

$$
\begin{array} { r } { \widetilde { \mathcal { D } } _ { f } \neq \mathcal { D } _ { f } , } \end{array}\tag{4}
$$

indicating that real frames are incorrectly treated as fake samples during training. Consequently, these incorrect labels introduce noise into frame-level classification.

2) Video-based Methods: Let $z _ { t }$ denote the logit for frame $v _ { t }$ . Video-based methods aggregate the frame-level logits using average pooling to obtain the video-level logit z:

$$
z = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } z _ { t } , \quad p = \mathrm { s i g m o i d } ( z ) ,\tag{5}
$$

where $p$ denotes the video-level probability. The binary crossentropy loss is defined as:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { H } ( y , p ) = - y \log p - ( 1 - y ) \log ( 1 - p ) . } \end{array}\tag{6}
$$

For a fake video, where $y = 1$ , the gradient with respect to each frame-level logit is:

$$
\frac { \partial \mathcal { L } } { \partial z _ { t } } = \frac { \partial \mathcal { L } } { \partial p } \frac { \partial p } { \partial z } \frac { \partial z } { \partial z _ { t } } = - \frac { 1 - p } { T } < 0 .\tag{7}
$$

Thus, all frames in a partially forged video are optimized toward the fake class, pushing real frames toward the fake distribution as well. This distorts frame-level representation learning, especially when forged frames are sparse.

## C. Our Approach

To address these limitations, we introduce an additional set of annotated facial images to provide fine-grained supervision. Let $( s , y _ { s } ) \sim \mathcal { D } _ { s }$ denote a facial image and its binary label, where $y _ { s } \in \{ 0 , 1 \}$ . Compared with the video-level label Y, the image label $y _ { s }$ is more fine-grained and precise, and annotated facial images are readily available in face forgery datasets. As a static facial image can be regarded as an individual video frame, samples from $\mathcal { D } _ { s }$ can substitute for the unavailable frame-level annotations $\left( v _ { t } , y _ { t } \right) \sim \mathcal { D } _ { f }$ . This motivates us to learn unified representations for videos and images.

## IV. METHODOLOGY

An overview of the proposed UVIF is presented in Figure 2. This section is organized as follows. Section IV-A introduces unified video and image modeling as the foundation of the framework. Section IV-B describes the pseudo labeling strategy that transfers image supervision to video frames. Section IV-C presents the video-oriented feature alignment. Section IV-D details the overall optimization objective.

## A. Unified Video and Image Modeling

Partial face forgery detection is limited by the absence of frame-level annotations. We introduce an annotated facial image set to provide fine-grained supervision. The model therefore needs to process both videos and images during training. Inspired by the popular unified architecture designs [34]–[36], we use a unified encoder $\phi ( \cdot )$ with shared parameters to extract features for both video and image inputs. The video and image inputs are represented as a 4D tensor $X \in \mathbb { R } ^ { T \times 3 \times H \times W }$ , where H and W denote spatial dimensions, and $T$ denotes temporal dimension, with $T = 1$ for images. The unified encoder uses typical 2D backbones to extract spatial features from images and video frames in the same manner. Temporal fusion is applied to video inputs to compensate for the lack of temporal interaction across video frame features.

![](images/1bc2f9c432771f3ecfc7317d781dfaf8dbd851df1b765f4901a84177494962a7.jpg)  
Fig. 3. Illustration of the pseudo labeling. For each video frame $v _ { t } ,$ the image classification head predicts the pseudo label $\tilde { y } _ { t }$ from a weakly augmented view, while the frame head predicts $\hat { p } _ { t }$ from a strongly augmented view.

After average pooling, the unified encoder produces feature vectors $z _ { v } , z _ { s } \in \mathbb { R } ^ { C }$ , where C denotes the number of channels.

Two separate linear classification heads are applied to make predictions for videos and images. During training, each minibatch is constructed by randomly sampling a set of facial videos and images. The classification losses are defined as:

$$
\mathcal { L } _ { \mathrm { v i d e o } } = \mathcal { H } ( Y , p _ { v } ) , \quad \mathcal { L } _ { \mathrm { i m a g e } } = \mathcal { H } ( y _ { s } , p _ { s } ) ,\tag{8}
$$

where $p _ { v }$ and $p _ { s }$ denote the predicted probabilities for video and image inputs, respectively, and $\mathcal { H } ( \cdot )$ denotes the crossentropy loss. The unified encoder is thus optimized by both video and image supervision. Apart from video face forgery detection, it also learns to extract discriminative features for determining the authenticity of facial images.

The unified encoder is implemented with typical 2D CNN [41], [42] or transformer [36] backbones. For temporal fusion, we adopt the Temporal Shift Module (TSM) [35] for CNN backbones and temporal attention [36] for transformers.

## B. Bridging Video and Image Representation

The unified modeling process uses annotated samples from the facial image set $( s , y _ { s } ) ~ \sim ~ \mathcal { D } _ { s }$ as substitutes for the original video frame set $\left( v _ { t } , y _ { t } \right) \sim \mathcal { D } _ { f }$ . As the frame-level labels $y _ { t }$ of video samples are unavailable during training, the model cannot receive direct supervision from each video frame. Since $\mathcal { D } _ { s }$ may differ from $\mathcal { D } _ { f }$ due to limited number of image samples or distribution differences between images and video frames, the model may learn biased representations from images that mismatch with video frames. To address this issue, we introduce an auxiliary pseudo labeling process for facial videos. It generates pseudo labels for video frames using the image classification head, aiming to bridge the representation of video frames and images.

As shown in Figure 3, we generate two augmented views for each video frame v :

$$
v _ { t } ^ { \mathrm { w e a k } } = \mathcal { A } _ { \mathrm { w e a k } } ( v _ { t } ) , \quad v _ { t } ^ { \mathrm { s t r o n g } } = \mathcal { A } _ { \mathrm { s t r o n g } } ( v _ { t } ) ,\tag{9}
$$

where $\mathcal { A } _ { \mathrm { w e a k } }$ denotes weak augmentation, including resizing, cropping, and horizontal flipping, while $\mathcal { A } _ { \mathrm { s t r o n g } }$ denotes strong augmentation with additional image compression and color perturbation. The two views are fed into the unified encoder to obtain the corresponding feature vectors $z _ { t } ^ { \mathrm { w e a k } }$ and $z _ { t } ^ { \mathrm { s t r o n g } }$ The image classification head predicts the soft pseudo label $\tilde { y } _ { t }$ from the weakly augmented view, while a newly introduced frame head predicts the frame-level probability $p _ { t }$ from the strongly augmented view. The frame head has the same architecture as the image head but uses unshared parameters.

The pseudo labeling loss is defined as:

$$
\mathcal { L } _ { \mathrm { p s e u d o } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathcal { H } \left( \mathrm { s t o p g r a d } ( \tilde { y } _ { t } ) , p _ { t } \right) ,\tag{10}
$$

where the stop-gradient operation stopgrad(·) is applied to pseudo labels $\tilde { y } _ { t }$ to avoid the collapse of training.

The pseudo labeling process follows semi-supervised learning paradigms [43], [44], which generate reliable pseudo labels from a weakly augmented view and provide supervision to a strong view. Since the frame head is discarded at inference, it only affects representation learning in the encoder.

## C. Video-Oriented Feature Alignment

The unified modeling and pseudo labeling processes introduce image-level and frame-level supervision to learn discriminative representations for distinguishing authentic and manipulated video frames. However, both processes rely on the image classification head to provide effective supervision. Static images and video frames may contain different manipulation types, which may cause an image head trained on images to produce unreliable predictions for video frames.

To address this issue, we propose a feature alignment strategy that adapts the image head to video frame representations. Specifically, we perform feature-level interpolation between image and video frame features. The image feature serves as the anchor, and the interpolation is biased toward it to keep the mixed feature compatible with the image head, as shown in Figure 4. This strategy regularizes the image head in the intermediate feature space and encourages a smoother transition between the image and video frame distributions.

Given an encoder layer ℓ randomly sampled from the candidate index set I, we construct two feature sets for images and video frames, $\mathcal { Z } _ { s }$ and $\mathcal { Z } _ { f }$ , by extracting intermediate features through a forward pass from input feature to layer ℓ, denoted by $\phi _ { 0 : \ell } ( \cdot )$ . For typical backbones [36], [41], [42], ℓ ranges from 0 to 4, with $\ell = 0$ denoting the input feature.

The image feature set $\mathcal { Z } _ { s }$ is defined as:

$$
\mathcal { Z } _ { s } = \left\{ \boldsymbol { z } _ { s , i } ^ { \ell } \vert s _ { i } \in \mathcal { D } _ { s } \right\} .\tag{11}
$$

The video frame feature set $\mathcal { Z } _ { f }$ includes both real and fake frames. Every frame from real videos is treated as a real sample, while fake frames from fake videos are selected when their pseudo-label confidence exceeds the threshold τ:

$$
\mathcal { Z } _ { f } = \left\{ z _ { t } ^ { \ell } \vert v _ { t } \in V , Y = 0 \vee ( Y = 1 \wedge \tilde { y } _ { t , 1 } > \tau ) \right\} .\tag{12}
$$

We construct interpolated samples by using an image feature $z _ { a } ^ { \ell }$ as the anchor and randomly selecting a target feature $z _ { q } ^ { \ell }$ from the image or video frame set:

$$
z _ { a } ^ { \ell } \in \mathcal { Z } _ { s } , \quad z _ { q } ^ { \ell } \in \mathcal { Z } _ { s } \cup \mathcal { Z } _ { f } .\tag{13}
$$

Note that the anchor feature $z _ { a } ^ { \ell }$ is restricted to $\mathcal { Z } _ { s }$ rather than $\mathcal { Z } _ { s } \cup \mathcal { Z } _ { f }$ to avoid generating features dominated by video frames, which may introduce noise to the image head.

![](images/cb57f4b768a05f50d897603eb68cd30b0a31cf3bacd6d845ce5d195a4d047285.jpg)  
Fig. 4. Illustration of video-oriented feature alignment. The image feature set $\mathcal { Z } _ { s }$ and video frame set $\mathcal { Z } _ { f }$ are first constructed. An image feature serves as the anchor $z _ { a } ^ { l }$ and is interpolated with a target feature $\check { z _ { q } ^ { l } } .$ For image-video interpolation, $\lambda > 0 . 5$ makes the interpolated feature closer to the image.

The interpolated feature and its soft label are defined as:

$$
z _ { m } ^ { \ell } = \lambda z _ { a } ^ { \ell } + ( 1 - \lambda ) z _ { q } ^ { \ell } , \quad y _ { m } = \lambda y _ { a } + ( 1 - \lambda ) y _ { q } ,\tag{14}
$$

where $y _ { a }$ and $y _ { q }$ denote the corresponding labels. Specifically, $y _ { q }$ is a pseudo label when $z _ { q } ^ { \ell }$ is a video frame feature.

The weight λ is sampled according to the source of the target feature:

$$
\lambda \sim \left\{ \begin{array} { l l } { \mathrm { B e t a } ( 1 , 1 ) , } & { z _ { q } ^ { \ell } \in \mathcal { Z } _ { s } , } \\ { \mathrm { B e t a } ( 1 , 1 ) \mid \lambda > 0 . 5 , } & { z _ { q } ^ { \ell } \in \mathcal { Z } _ { f } . } \end{array} \right.\tag{15}
$$

For frame target features $z _ { q } ^ { \ell } \in \mathcal { Z } _ { f }$ , we truncate the distribution by enforcing $\lambda > 0 . 5$ , biasing the interpolated feature toward the image anchor for compatibility with the image head.

We construct $N _ { \mathrm { a l i g n } }$ interpolated samples at each iteration. The interpolated features are passed through the remaining encoder layers and then predicted by the image head. The alignment loss is defined as:

$$
\mathcal { L } _ { \mathrm { a l i g n } } = \gamma \mathcal { H } ( y _ { m } , h _ { s } ( \phi _ { \ell + 1 : } ( z _ { m } ^ { \ell } ) ) ) ,\tag{16}
$$

where $\phi _ { \ell + 1 : } ( \cdot )$ denotes the forward pass from layer $\ell { + 1 }$ to the output feature, $h _ { s } ( \cdot )$ denotes the image head, and γ denotes the loss weight. For hyperparameters, we use $\mathcal { T } = \{ 0 , 1 , 2 \}$ $\tau { = } 0 . 9 , N _ { \mathrm { a l i g n } } { = } 6 4$ , and $\gamma { = } 0 . 5$ in the experiments. 0

This strategy is related to manifold mixup [45], as both perform interpolation in feature space. However, our strategy interpolates features from different distributions and biases the interpolated features toward image anchors, enabling the image head to adapt to video frame features while preserving its classification ability on both images and video frames.

## D. Optimization

The overall framework is optimized end-to-end under a multi-task learning paradigm using the following objective:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { v i d e o } } + \mathcal { L } _ { \mathrm { i m a g e } } + \mathcal { L } _ { \mathrm { p s e u d o } } + \mathcal { L } _ { \mathrm { a l i g n } } ,\tag{17}
$$

where ${ \mathcal { L } } _ { \mathrm { v i d e o } } , { \mathcal { L } } _ { \mathrm { i m a g e } } , { \mathcal { L } } _ { \mathrm { p s e u d o } } ,$ and $\mathcal { L } _ { \mathrm { a l i g n } }$ denote the video classification, image classification, pseudo labeling, and feature alignment losses, respectively. Note that the proposed method incurs no additional computational overhead during inference as only the video classification process is applied.

## V. EXPERIMENTS

## A. Experimental Setup

1) Datasets and Evaluation Metrics: We conduct experiments on publicly available face forgery detection datasets ForgeryNet [18] and DFDC (preview) [46].

ForgeryNet is a large-scale face forgery detection dataset with over 220k facial video clips and 2.9m static images from over 5k subjects. It contains 15 facial manipulation approaches and over 36 mix-perturbations. Most fake videos are partially manipulated, making this dataset challenging for face forgery detection. For video forgery classification, we follow [18] and use about 140k videos for training and 18k for validation. For the image set, ForgeryNet includes over 2.3m training images, but unless otherwise specified, we use only a randomly selected subset of 100k images (less than 5%).

DFDC is the preview dataset for the Deepfake Detection Challenge [46], which includes 1131 real videos and 4113 forged videos generated using two synthesis methods. Most videos are partially forged. We follow the original dataset split in [46] and use 4464 videos for training and 780 for testing.

For evaluation metrics, we follow [18], [46] and adopt video-level Accuracy (Acc) and Area under the ROC curve (AUC) to evaluate the performance of video forgery detection.

2) Implementation Details: We adopt the Swin Transformer [36] or CNNs [41], [42] with TSM [35] as representative implementations of our unified encoder. Following [18], we use RetinaFace [47] to extract facial regions and resize them to 224×224. We sample 32 frames with a temporal stride of 4 from each video clip for training and use the center clip for evaluation. We employ both weak and strong data augmentation: weak augmentation includes random resizing, cropping, and flipping, while strong augmentation further includes compression and color perturbation. We apply strong augmentation to all compared methods, whereas UVIF uses both augmentations to construct two views for pseudo labeling. The video batch size is 32, and UVIF also randomly samples 256 annotated images per iteration without requiring the same subjects across videos and images. All models are trained on four NVIDIA Tesla V100 GPUs with a learning rate of 0.01, momentum of 0.9, and weight decay of 0.0001. We train the models for 50K iterations on ForgeryNet and 20K iterations on DFDC to ensure convergence. Unless otherwise specified, all compared methods use the same hyperparameters.

## B. Comparison to State-of-the-art

We evaluate the performance of our proposed methods on the ForgeryNet [18] dataset, and compare them with stateof-the-art methods, including image-based and video-based methods for video forgery detection, as well as typical multiple instance learning methods, as detailed in Table I. The imagebased methods are initialized with pre-trained weights on ImageNet-1k [53], while video-based methods are initialized with pre-trained weights on Kinetics-400 [54] for better performance. The TSM [35] and VideoSwin [36] methods can be viewed as video-based baselines for UVIF, as they employ equivalent architectures during evaluation.

TABLE I

COMPARISON WITH THE STATE-OF-THE-ART METHODS ON FORGERYNET [18]. RESULTS MARKED WITH † ARE CITED FROM THE ORIGINAL PAPERS. ALL THE REMAINING RESULTS ARE REIMPLEMENTED USING THE SAME

PROTOCOL FOR A FAIR COMPARISON. THE #PARAMS DENOTES THE NUMBER OF PARAMETERS. THE FLOPS ARE MEASURED UNDER SPATIAL SIZE 224 × 224 AND 32 FRAMES. BOLD INDICATES THE BEST RESULTS.

<table><tr><td colspan="2">Method</td><td>Backbone</td><td>#params (M)</td><td>FLOPs (G)</td><td>Acc</td><td>AUC</td></tr><tr><td rowspan="4">Ima-ased</td><td>Xception [48]</td><td>Xception</td><td>21</td><td>146</td><td>66.66</td><td>72.06</td></tr><tr><td>EfficientNet [49]</td><td>Efficient-B4</td><td>18</td><td>49</td><td>70.50</td><td>77.04</td></tr><tr><td>Swin [36]</td><td>Swin-T</td><td>28</td><td>144</td><td>71.72</td><td>77.00</td></tr><tr><td>Swin [36] CLIP-ViT [33]</td><td>Swin-S ViT-B</td><td>49 86</td><td>281 18</td><td>71.46 70.63</td><td>77.80 77.11</td></tr><tr><td colspan="2">Effort [12]</td><td>ViT-L</td><td>303</td><td>2598</td><td>72.51</td><td>80.64</td></tr><tr><td rowspan="20">Vidd-asd</td><td>TSM [18] †</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>R-50</td><td>24</td><td>132</td><td>88.04</td><td>93.05</td></tr><tr><td>SlowFast [18] †</td><td>R3D-50</td><td>34</td><td>51</td><td>88.78</td><td>93.88</td></tr><tr><td>TSM [35]</td><td>R-50</td><td>24</td><td>132</td><td>83.60</td><td>90.12</td></tr><tr><td>SlowFast [50]</td><td>R3D-50</td><td>34</td><td>51</td><td>84.04</td><td>91.02</td></tr><tr><td>SlowFast [50]</td><td>R3D-101</td><td>62</td><td>97</td><td>84.31</td><td>90.87</td></tr><tr><td>VideoSwin [36]</td><td>Swin-T</td><td>28</td><td>88</td><td>83.89</td><td>90.53</td></tr><tr><td>VideoSwin [36]</td><td>Swin-S</td><td>50</td><td>166</td><td>84.56</td><td>91.19</td></tr><tr><td>TimeSformer [51]</td><td>ViT-B</td><td>86</td><td>563</td><td>84.17</td><td>90.18</td></tr><tr><td>UniFormer [52] FTCN [19]</td><td>UniFormer-S</td><td>21</td><td>110</td><td>84.01</td><td>89.33</td></tr><tr><td>STIL [17]</td><td>R3D-50</td><td>57</td><td>68</td><td>78.62</td><td>85.00</td></tr><tr><td>AltFreezing [20]</td><td>SCNet-50</td><td>23</td><td>151</td><td>83.55</td><td>88.98</td></tr><tr><td>TALL [21]</td><td>R3D-50</td><td>27</td><td>114</td><td>82.80</td><td>85.73</td></tr><tr><td>PwTF [23]</td><td>Swin-B</td><td>87</td><td>16</td><td>82.17</td><td>88.12</td></tr><tr><td>MINTIME-XC [22] †</td><td>Designed</td><td>197</td><td>90</td><td>75.98</td><td>81.92</td></tr><tr><td>DFD-FCG [25]</td><td>Xception ViT-L</td><td>103 306</td><td>223 2624</td><td>87.64 84.13</td><td>94.25 91.73</td></tr><tr><td rowspan="5">MII</td><td>MIL [28]</td><td>Swin-T</td><td></td><td></td><td></td><td>91.31</td></tr><tr><td>S-MIL [26]</td><td>Swin-T</td><td>28 28</td><td>88 88</td><td>84.10 83.95</td><td>91.20</td></tr><tr><td>TransMIL [30]</td><td>Swin-T</td><td>30</td><td>96</td><td>82.89</td><td>87.38</td></tr><tr><td>DSMIL [29]</td><td>Swin-T</td><td>28</td><td>88</td><td>83.84</td><td>90.59</td></tr><tr><td>ACMIL [31]</td><td>Swin-T</td><td>28</td><td>88</td><td>83.74</td><td>90.96</td></tr><tr><td rowspan="3">Ou</td><td>UVIF (Ours)</td><td>R-50</td><td>24</td><td>132</td><td>85.51</td><td>93.12</td></tr><tr><td>UVIF (Ours)</td><td>Swin-T</td><td>28</td><td>88</td><td>88.02</td><td>95.10</td></tr><tr><td>UVIF (Ours)</td><td>Swin-S</td><td>50</td><td>166</td><td>88.94</td><td>95.69</td></tr></table>

The comparison results reveal a clear performance gap between image-based and video-based methods. The best image-based method Effort [12] achieves 72.51% accuracy and 80.64% AUC, whereas most video-based methods perform substantially better. This gap indicates that image-based methods cannot model temporal cues and are more susceptible to label noise in partially forged videos. The proposed UVIF methods achieve superior performance over previous methods. The best model, UVIF-Swin-S, achieves 88.94% accuracy and 95.69% AUC. Compared with the TSM and VideoSwin baselines, the corresponding UVIF variants introduce no additional parameters or FLOPs, and UVIF-Swin-T achieves a tradeoff between detection accuracy and computational efficiency. Besides, MIL methods [26], [28], [29] are also implemented using the VideoSwin-T architecture, and each input video clip is treated as a bag in MIL. The results in Table I indicate that although MIL methods can improve the AUC to some extent by grouping similar video frames, the performance gains remain limited. In contrast, our UVIF methods can significantly improve detection performance using fine-grained image annotations.

We also compare UVIF with existing methods on the DFDC [46] dataset, as presented in Table II. UVIF is trained using the selected image set from ForgeryNet. It significantly enhances detection accuracy over the TSM [35] and VideoSwin [36] baselines, e.g., +3.99% accuracy for Swin-T. Our method also outperforms state-of-the-art methods on DFDC.

TABLE II  
RESULTS ON THE DFDC [46]. RESULTS MARKED WITH † ARE CITED FROM ORIGINAL PAPERS. BOLD INDICATES THE BEST RESULTS.
<table><tr><td>Method</td><td>Acc</td><td>AUC</td></tr><tr><td>Xception-avg [5] †</td><td>84.58</td><td>1</td></tr><tr><td>S-MIL [26] †</td><td>83.78</td><td></td></tr><tr><td>S-MIL-T [26]</td><td>85.11</td><td></td></tr><tr><td>STIL [17] †</td><td>89.80</td><td></td></tr><tr><td>TSM-R50 [35]</td><td>81.08</td><td>90.67</td></tr><tr><td>SlowFast-R50 [50]</td><td>83.53</td><td>92.10</td></tr><tr><td>VideoSwin-T [36]</td><td>84.68</td><td>92.72</td></tr><tr><td>VideoSwin-S [36]</td><td>88.29</td><td>94.85</td></tr><tr><td>Uniformer-S [52]</td><td>85.07</td><td>94.02</td></tr><tr><td>TimeSformer [51]</td><td>87.00</td><td>94.58</td></tr><tr><td>AltFreezing [20]</td><td>82.50</td><td>89.83</td></tr><tr><td>TALL [21]</td><td>86.23</td><td>92.89</td></tr><tr><td>PwTF [23]</td><td>77.22</td><td>82.54</td></tr><tr><td>MINTIME-XC [22] †</td><td>86.40</td><td>95.20</td></tr><tr><td>DFD-FCG [25]</td><td>85.20</td><td>93.39</td></tr><tr><td>UVIF-R50 (Ours)</td><td>86.49</td><td>92.88</td></tr><tr><td>UVIF-Swin-T (Ours)</td><td>88.67</td><td>95.26</td></tr><tr><td></td><td></td><td></td></tr><tr><td>UVIF-Swin-S (Ours)</td><td>90.73</td><td>96.05</td></tr></table>

## C. Ablation Studies and Analysis

1) Effectiveness of the Core Components: We first perform ablation experiments to analyze the components of the UVIF framework, as shown in Table III. Specifically, UVIF contains four primary components: image supervision $\mathcal { L } _ { \mathrm { i m a g e } }$ in unified modeling, temporal fusion for video inputs, $\mathcal { L } _ { \mathrm { p s e u d o } }$ in pseudo labeling process, and $\mathcal { L } _ { \mathrm { a l i g n } }$ in feature alignment. The first row corresponds to a vanilla Swin-T model trained on videos from ForgeryNet [18]. Applying temporal fusion in row 3 brings significant performance gains (+8.85% accuracy and +7.67% AUC), indicating the importance of temporal information in face forgery detection. Combining images and videos for training (row 4) further improves the accuracy to 86.77%, showing the benefit of fine-grained image annotations. Results in row 2 also show that image supervision improves performance even without temporal fusion. This further reflects the effectiveness of annotated images, as the features of images and individual video frames are similar. Adding the pseudo labeling loss in row 5 further improves the results, indicating its efficacy in bridging the representations of images and video frames. Finally, incorporating feature alignment in row 6 yields the best performance, with 88.02% accuracy and 95.10% AUC, demonstrating its effectiveness in reducing the distribution gap between images and video frames.

2) Number of Training Images: We then evaluate the performance of UVIF using different numbers of training images, as illustrated in Figure 5. We randomly sample 10k, 20k, 50k, 100k, 200k, 500k, and 2.3M (all) training images from ForgeryNet [18], and perform experiments under three settings, i.e., UVIF, UVIF without feature alignment, and a multimodal baseline trained jointly on videos and images. The results show that the accuracy of all methods significantly improves when the training images increase from 0 to 100k and reaches saturation at 100k. Thus, 100k is sufficient for the current video set, and adding more images, i.e., even up to 2.3M, does not provide further gains. 100k images with annotations are easily available from open-access datasets, making UVIF practical for improving partially forged video detection. Even with a small portion of images, e.g., 10k, UVIF achieves clear performance gains over the video classification baseline (86.36% vs. 83.89% accuracy). This indicates that the UVIF model gradually learns useful supervision information from annotated images, rather than relying on a large number of images to improve its representation. Meanwhile, the comparison of three curves shows that 1) incorporating pseudo labeling process and feature alignment are always effective using different numbers of training images, and 2) the performance improvement is larger when 100k or more images are used. This indicates that the model requires a certain number of images to train a reliable image head for pseudo label generation and to learn effective alignment between the distributions of images and video frames.

TABLE III  
EFFECTIVENESS OF THE CORE COMPONENTS OF UVIF WITH SWIN-T. $\mathcal { L } _ { \mathrm { i m a g e } } \mathrm { : }$ : IMAGE SUPERVISION IN UNIFIED MODELING; TEMPORAL: TEMPORAL FUSION FOR VIDEO INPUTS; $\mathcal { L } _ { \mathrm { p s e u d o } } \mathrm { . }$ : PSEUDO LABELING; $\mathcal { L } _ { \mathrm { a l i g n } } \colon$ VIDEO-ORIENTED FEATURE ALIGNMENT.
<table><tr><td> $\mathcal { L } _ { \mathrm { i m a g e } }$ </td><td>Temporal</td><td> $\mathcal { L } _ { \mathrm { p s e u d o } }$ </td><td> $\mathcal { L } _ { \mathrm { a l i g n } }$ </td><td> $\operatorname { A c c }$ </td><td>AUC</td></tr><tr><td></td><td></td><td></td><td></td><td>75.04</td><td>82.86</td></tr><tr><td>√</td><td></td><td></td><td></td><td>80.02</td><td>84.99</td></tr><tr><td></td><td>√</td><td></td><td></td><td>83.89</td><td>90.53</td></tr><tr><td> $\checkmark$ </td><td>√</td><td></td><td></td><td>86.77</td><td>94.39</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td>√</td><td></td><td>87.30</td><td>94.59</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td>√</td><td>√</td><td>88.02</td><td>95.10</td></tr></table>

3) Performance on Videos with Different Forged Ratios: Figure 6 illustrates the accuracy achieved on videos with different forged ratios from the ForgeryNet [18] validation set. We utilize the annotations for temporal forgery localization [18] task to compute the ratio of forged frames of each partially forged video, and then divide them into six groups based on their forged ratios: [0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], and [0.5, 1]. The sample distribution of each group is 0.07, 0.19, 0.22, 0.22, 0.15, and 0.15, respectively. We compare UVIF with image-based and video-based baselines using Swin-T. The results show that the video-based baseline consistently outperforms the image-based baseline, with larger gains when forged frames are sparse. UVIF consistently enhances the video-based baselines across all forged-ratio groups. This indicates that UVIF leverages annotated images to learn discriminative representations that distinguish real and forged video frames in partially forged videos.

4) Components of Feature Alignment: Table IV analyzes three key components of the proposed video-oriented feature alignment: 1) whether interpolation between image and video is necessary, 2) whether image features should serve as anchors, and 3) how the interpolation should be biased. Feature Source specifies the interpolation sets, where I and V denote the image set $\mathcal { Z } _ { s }$ and video-frame set $\mathcal { Z } _ { f } .$ , respectively. Image

![](images/49d4beedaa9f0aa97f5ca1d63fafdf8a58d9e2bd43fce718677f369c7f887afc.jpg)  
Fig. 5. Ablation on the number of training images under three settings, i.e., multimodal baseline with Swin-T, UVIF without feature alignment, and UVIF.  
TABLE IV

ABLATION STUDY OF VIDEO-ORIENTED FEATURE ALIGNMENT. “FEATURE SOURCE” INDICATES THE INTERPOLATION FEATURE SETS, WITH I AND V REPRESENTING IMAGE AND VIDEO FRAME FEATURES. “IMAGE ANCHOR”

SPECIFIES WHETHER THE ANCHOR IS RESTRICTED TO IMAGES, AND “BIAS” INDICATES THE INTERPOLATION DIRECTION CONTROLLED BY λ.
<table><tr><td>Setting</td><td>Feature Source</td><td>Image Anchor</td><td>Bias</td><td>Acc</td><td>AUC</td></tr><tr><td colspan="6">Baselines</td></tr><tr><td>video baseline</td><td></td><td></td><td></td><td>83.89</td><td>90.53</td></tr><tr><td>UVIF (w/o alignment)</td><td></td><td></td><td></td><td>87.30</td><td>94.59</td></tr><tr><td colspan="6">Single-domain interpolation</td></tr><tr><td>image mixup</td><td>I</td><td></td><td></td><td>87.44</td><td>94.54</td></tr><tr><td>video mixup</td><td>V</td><td></td><td></td><td>87.02</td><td>94.44</td></tr><tr><td colspan="6">Interpolation without image anchors</td></tr><tr><td>unbiased interpolation</td><td>I, V</td><td>×</td><td></td><td>87.38</td><td>94.57</td></tr><tr><td>video-biased</td><td>I, V</td><td>X</td><td> $\lambda < 0 . 5$ </td><td>87.07</td><td>94.62</td></tr><tr><td>image-biased</td><td>I, V</td><td>×</td><td> $\lambda > 0 . 5$ </td><td>87.61</td><td>94.99</td></tr><tr><td colspan="6">Interpolation with image anchors</td></tr><tr><td>unbiased interpolation</td><td>I, V</td><td>√</td><td></td><td>87.49</td><td>94.58</td></tr><tr><td>video-biased</td><td>I, V</td><td>√</td><td> $\lambda < 0 . 5$ </td><td>87.09</td><td>94.33</td></tr><tr><td>image-biased (Ours)</td><td>I, V</td><td>√</td><td> $\lambda > 0 . 5$ </td><td>88.02</td><td>95.10</td></tr></table>

Anchor indicates whether the anchor is restricted to image features, i.e., $z _ { a } \in \mathcal { Z } _ { s }$ when enabled and $z _ { a } \in \mathcal { Z } _ { s } \cup \mathcal { Z } _ { f }$ otherwise. Bias denotes the interpolation direction, with $\lambda > 0 . 5$ moving toward the image feature and $\lambda ~ < ~ 0 . 5$ toward the video-frame feature. The first two rows serve as baselines, while the remaining rows evaluate the three design choices.

Single-domain mixup is inadequate for feature alignment. Rows 3-4 show that image mixup provides limited gains, while video mixup underperforms UVIF without alignment. Image mixup regularizes the model but does not reduce the imagevideo distribution gap, whereas video mixup operates between video frames and may introduce noise to the image head.

Image anchors facilitate feature alignment. Removing the anchor constraint (rows 5-7) allows arbitrary interpolation between image and video features, including between two video features, which degenerates into video mixup and impairs the image head. In contrast, image anchors (rows 8- 10) keep interpolated features between the image and video distributions, preserving compatibility with the image head while enabling adaptation to video representations.

Image-biased interpolation is critical for feature alignment.

![](images/f1e8c21c9ee84ec5d1f291fab10210957e5fac1adde338812e32036172472d4f.jpg)  
Fig. 6. Detection accuracy on ForgeryNet videos with different forged-frame ratios. The image- and video-based baselines use Swin-T as the backbone.

The comparison among unbiased, video-biased, and imagebiased interpolation shows that the first two strategies are suboptimal, while image-biased interpolation achieves the best performance, improving accuracy from 87.30% to 88.02%. This suggests that interpolation far from the image distribution can make it difficult for the image head to adapt. In contrast, image-biased interpolation keeps features close to the image anchor while incorporating video information, improving adaptation to video representations.

5) Composition of Training Images: We further investigate how the composition of manipulation types in the training image set affects UVIF performance, as shown in Figure 7. Specifically, the ForgeryNet video set contains eight manipulation types, while the image set covers these eight overlapping types and seven additional non-overlapping types. We construct training image sets with different compositions of manipulation types, denoted by $( K _ { o } , K _ { n } )$ , where $K _ { o }$ and $K _ { n }$ represent the numbers of overlapping and non-overlapping types, respectively. We consider three groups of settings: 1) Overlap, where $K _ { o } \in \{ 1 , 2 , 4 , 8 \}$ and $K _ { n } = 0 ; 2 )$ Non-overlap, where $K _ { o } = 0$ and $K _ { n } \in \{ 1 , 2 , 4 , 7 \}$ ; and 3) Mixed, where $K _ { o } = 8$ and $K _ { n } \in \{ 1 , 2 , 4 , 7 \}$ . For each setting, we randomly sample 100k training samples from the corresponding manipulation types and real images, and summarize the results across different manipulation type combinations.

The results show that the composition of manipulation types in the image training set affects UVIF performance. Under the Overlap setting, performance steadily improves as $K _ { o }$ increases from 1 to 8, reaching 88.71% on average. This indicates that UVIF benefits from greater overlap between the manipulation types in the video and image sets. However, this setting is idealized because the manipulation types in the video set may be unknown. The Non-overlap setting is more challenging because the training video and image sets share no manipulation types. Nevertheless, UVIF consistently improves performance, with further gains as the number of manipulation types increases. The Mixed setting confirms this trend, indicating that more diverse training images help UVIF learn more generalizable representations. Moreover, UVIF outperforms its variant without feature alignment under all three settings, indicating that feature alignment consistently reduces the image-video distribution gap and improves knowledge

(a)

![](images/31498cbac88f6ed486fa6b83318d6a9a15127849fbf9a6086d0a0783b9c52904.jpg)

![](images/bacfb4c1ead6af6c2825bbb271dd35944019b84dfaae193fb0f2f033af7d4b07.jpg)  
Fig. 7. Ablation on the compositions of manipulation types in the image training set for UVIF under three settings: (a) Overlap, (b) Non-overlap, and (c) Mixed, denoted by $( K _ { o } , K _ { n } ) . \ K _ { o }$ and $K _ { n }$ denote the numbers of overlapping and non-overlapping manipulation types between the video and image sets.

![](images/72550103365609dcbcc2b0044207267ca2b6ee1b011e17b83db57acb0eb058e4.jpg)

TABLE V  
COMPARING DIFFERENT BACKBONES FOR THE UNIFIED ENCODER. THE BASELINE IS A VIDEO CLASSIFICATION MODEL.
<table><tr><td rowspan="2">Backbone</td><td colspan="2">Baseline</td><td colspan="2">UVIF (Ours)</td></tr><tr><td>Acc</td><td>AUC</td><td>Acc</td><td>AUC</td></tr><tr><td>ConvNeXt-T [42]</td><td>83.18</td><td>89.10</td><td> $8 6 . 3 2 _ { \uparrow 3 . 1 4 }$ </td><td> $9 4 . 3 6 _ { \uparrow 5 . 2 6 }$ </td></tr><tr><td>R-50 [41]</td><td>83.60</td><td>90.12</td><td> $8 5 . 5 1 _ { \uparrow 1 . 9 1 }$ </td><td> $9 3 . 1 2 _ { \uparrow 3 . 0 0 }$ </td></tr><tr><td>Swin-T [36]</td><td>83.89</td><td>90.53</td><td> $8 8 . 0 2 _ { \uparrow 4 . 1 3 }$ </td><td> $9 5 . 1 0 \dot { _ { \uparrow 4 . 5 7 } }$ </td></tr><tr><td>Swin-S [36]</td><td>84.56</td><td>91.19</td><td> $8 8 . 9 4 _ { \uparrow 4 . 3 8 }$ </td><td> $9 5 . 6 9 _ { \uparrow 4 . 5 0 } ^ { \cdot }$ </td></tr><tr><td>Swin-B [36]</td><td>84.88</td><td>90.93</td><td> $8 8 . 8 7 _ { \uparrow 3 . 9 9 }$ </td><td> $9 5 . 8 4 _ { \uparrow 4 . 9 1 }$ </td></tr></table>

transfer from images to videos.

6) Different Backbones: We conduct ablation experiments using different backbones for the unified encoder, as presented in Table V. The evaluated backbones include ResNet [41], ConvNeXt [42], and Swin Transformer [36]. The results show that UVIF consistently improves detection performance across backbone architectures and capacities, demonstrating its generalization across diverse backbones.

7) Visualization: We further analyze the learned representations through t-SNE visualization, as shown in Figure 8. We compare the video classification baseline with UVIF on ForgeryNet [18]. The baseline partially separates real and fake video samples, but the two classes still overlap. In contrast, UVIF better separates real and fake samples while forming more compact clusters within each class. Image and video samples from the same class are also closer in the feature space, indicating that UVIF uses image supervision to learn more discriminative representations for video forgery detection and aligns representations across images and videos.

## VI. CONCLUSION

In this paper, we present UVIF, an end-to-end multi-task learning framework for video face forgery detection. Our method establishes a unified representation of facial videos and images by processing them within a single model. By utilizing the fine-grained annotations from the image set, UVIF brings significant performance gains for detecting partially forged videos. In the future, we will investigate face forgery detection from a weakly supervised learning perspective and develop multimodal foundation models for forgery analysis.

## REFERENCES

[1] F. Juefei-Xu, R. Wang, Y. Huang, Q. Guo, L. Ma, and Y. Liu, “Countering malicious deepfakes: Survey, battleground, and horizon,” Int. J. Comput. Vis. (IJCV), vol. 130, no. 7, pp. 1678–1734, 2022.

![](images/4e93e8c4099fa006d020680157b92737985acac7f1fec7cc21528c5cdd3b9af2.jpg)

real video fake video real image fake image

![](images/2fe7640d5e0f2e06bff648d3fe8c8e447fc95fb0f72360c1efb2ffd66012c7dc.jpg)

![](images/5929706fcc236e29f11b6aae6cd660cade8b930cdea4d645fc3293f37afcd0f2.jpg)  
(b) UVIF  
Fig. 8. T-SNE visualization of the features on ForgeryNet: (a) video classification baseline, (b) UVIF, and (c) UVIF with training images visualized.

[2] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial networks,” Commun. ACM, vol. 63, no. 11, pp. 139–144, 2020.

[3] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2022, pp. 10 684–10 695.

[4] D. Afchar, V. Nozick, J. Yamagishi, and I. Echizen, “Mesonet: a compact facial video forgery detection network,” in IEEE Int. Workshop Inf. Forensics Secur. (WIFS). IEEE, 2018, pp. 1–7.

[5] A. Rossler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner, “Faceforensics++: Learning to detect manipulated facial images,” in Int. Conf. Comput. Vis. (ICCV), 2019, pp. 1–11.

[6] H. Zhao, W. Zhou, D. Chen, T. Wei, W. Zhang, and N. Yu, “Multiattentional deepfake detection,” in IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2021, pp. 2185–2194.

[7] Y. Qian, G. Yin, L. Sheng, Z. Chen, and J. Shao, “Thinking in frequency: Face forgery detection by mining frequency-aware clues,” in Eur. Conf. Comput. Vis. (ECCV), 2020, pp. 86–103.

[8] K. Shiohara and T. Yamasaki, “Detecting deepfakes with self-blended images,” in IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2022, pp. 18 720–18 729.

[9] J. Cao, C. Ma, T. Yao, S. Chen, S. Ding, and X. Yang, “End-to-end reconstruction-classification learning for face forgery detection,” in IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2022, pp. 4113–4122.

[10] D. Liu, Z. Zheng, C. Peng, Y. Wang, N. Wang, and X. Gao, “Hierarchical forgery classifier on multi-modality face forgery clues,” IEEE Trans. Multimedia (TMM), vol. 26, pp. 2894–2905, 2024.

[11] Y. Yu, R. Ni, S. Yang, Y. Zhao, and A. C. Kot, “Narrowing domain gaps with bridging samples for generalized face forgery detection,” IEEE Trans. Multimedia (TMM), vol. 26, pp. 3405–3417, 2024.

[12] Z. Yan, J. Wang, P. Jin, K.-Y. Zhang, C. Liu, S. Chen, T. Yao, S. Ding, B. Wu, and L. Yuan, “Orthogonal subspace decomposition for generalizable ai-generated image detection,” in Int. Conf. Mach. Learn. (ICML). PMLR, 2025, pp. 70 268–70 288.

[13] X. Xu, J. Chen, Y. Zhang, C. Li, A. K. Singh, and Z. Lyu, “Deepfake detection with multi-view fusion and graph convolutional network,” IEEE Trans. Multimedia (TMM), vol. 28, pp. 167–180, 2026.

[14] Y. Li, M.-C. Chang, and S. Lyu, “In ictu oculi: Exposing ai created fake videos by detecting eye blinking,” in IEEE Int. Workshop Inf. Forensics Secur. (WIFS), 2018, pp. 1–7.

[15] A. Haliassos, K. Vougioukas, S. Petridis, and M. Pantic, “Lips don’t lie: A generalisable and robust approach to face forgery detection,” in IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2021, pp. 5039–5049.

[16] U. A. Ciftci, I. Demir, and L. Yin, “Fakecatcher: Detection of synthetic portrait videos using biological signals,” IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI), 2020.

[17] Z. Gu, Y. Chen, T. Yao, S. Ding, J. Li, F. Huang, and L. Ma, “Spatiotemporal inconsistency learning for deepfake video detection,” in ACM Int. Conf. Multimedia (ACMMM), 2021, pp. 3473–3481.

[18] Y. He, B. Gan, S. Chen, Y. Zhou, G. Yin, L. Song, L. Sheng, J. Shao, and Z. Liu, “Forgerynet: A versatile benchmark for comprehensive forgery analysis,” in IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2021, pp. 4360–4369.

[19] Y. Zheng, J. Bao, D. Chen, M. Zeng, and F. Wen, “Exploring temporal coherence for more general video face forgery detection,” in Int. Conf. Comput. Vis. (ICCV), 2021, pp. 15 044–15 054.

[20] Z. Wang, J. Bao, W. Zhou, W. Wang, and H. Li, “Altfreezing for more general video face forgery detection,” in IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2023, pp. 4129–4138.

[21] Y. Xu, J. Liang, G. Jia, Z. Yang, Y. Zhang, and R. He, “Tall: Thumbnail layout for deepfake video detection,” in Int. Conf. Comput. Vis. (ICCV), 2023, pp. 22 658–22 668.

[22] D. A. Coccomini, G. K. Zilos, G. Amato, R. Caldelli, F. Falchi, S. Papadopoulos, and C. Gennaro, “Mintime: multi-identity size-invariant video deepfake detection,” IEEE Trans. Inf. Forensics Secur. (TIFS), 2024.

[23] T. Kim, J. Choi, Y. Jeong, H. Noh, J. Yoo, S. Baek, and J. Choi, “Beyond spatial frequency: Pixel-wise temporal frequency-based deepfake video detection,” in Int. Conf. Comput. Vis. (ICCV), 2025, pp. 11 198–11 207.

[24] F. Nie, J. Ni, J. Zhang, B. Zhang, and W. Zhang, “Dip: diffusion learning of inconsistency pattern for general deepfake detection,” IEEE Trans. Multimedia (TMM), vol. 27, pp. 2155–2167, 2025.

[25] Y.-H. Han, T.-M. Huang, K.-L. Hua, and J.-C. Chen, “Towards more general video-based deepfake detection through facial component guided adaptation for foundation model,” in IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2025, pp. 22 995–23 005.

[26] X. Li, Y. Lang, Y. Chen, X. Mao, Y. He, S. Wang, H. Xue, and Q. Lu, “Sharp multiple instance learning for deepfake video detection,” in ACM Int. Conf. Multimedia (ACMMM), 2020, pp. 1864–1872.

[27] J. Feng and Z.-H. Zhou, “Deep miml network,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 31, 2017.

[28] M. Ilse, J. Tomczak, and M. Welling, “Attention-based deep multiple instance learning,” in Int. Conf. Mach. Learn. (ICML). PMLR, 2018, pp. 2127–2136.

[29] B. Li, Y. Li, and K. W. Eliceiri, “Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning,” in IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2021, pp. 14 318–14 328.

[30] Z. Shao, H. Bian, Y. Chen, Y. Wang, J. Zhang, X. Ji et al., “Transmil: Transformer based correlated multiple instance learning for whole slide image classification,” Adv. Neural Inform. Process. Syst. (NIPS), vol. 34, pp. 2136–2147, 2021.

[31] Y. Zhang, H. Li, Y. Sun, S. Zheng, C. Zhu, and L. Yang, “Attentionchallenging multiple instance learning for whole slide image classification,” in Eur. Conf. Comput. Vis. (ECCV). Springer, 2024, pp. 125–143.

[32] H. Liu, C. Pan, Y. Liu, G. Zhao, and X. Li, “Unified video and image representation for boosted video face forgery detection,” in Eur. Conf. Artif. Intell. (ECAI), 2024, pp. 673–680.

[33] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in Int. Conf. Mach. Learn. (ICML). PMLR, 2021, pp. 8748–8763.

[34] R. Girdhar, M. Singh, N. Ravi, L. Van Der Maaten, A. Joulin, and I. Misra, “Omnivore: A single model for many visual modalities,” in IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2022, pp. 16 102– 16 112.

[35] J. Lin, C. Gan, and S. Han, “Tsm: Temporal shift module for efficient video understanding,” in Int. Conf. Comput. Vis. (ICCV), 2019, pp. 7083– 7093.

[36] Z. Liu, J. Ning, Y. Cao, Y. Wei, Z. Zhang, S. Lin, and H. Hu, “Video swin transformer,” in IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2022, pp. 3202–3211.

[37] H. Duan, Y. Zhao, Y. Xiong, W. Liu, and D. Lin, “Omni-sourced weblysupervised learning for video recognition,” in Eur. Conf. Comput. Vis. (ECCV). Springer, 2020, pp. 670–688.

[38] S. Cao, Z. Zhao, S. Hao, W. Chai, J.-N. Hwang, H. Wang, and G. Wang, “Efficient transfer from image-based large multimodal models to video tasks,” IEEE Trans. Multimedia (TMM), 2025.

[39] X. He, S. Chen, F. Ma, Z. Huang, X. Jin, Z. Liu, D. Fu, Y. Yang, J. Liu, and J. Feng, “Vlab: Enhancing video language pretraining by feature adapting and blending,” IEEE Trans. Multimedia (TMM), vol. 27, pp. 2168–2180, 2025.

[40] R. Girdhar, A. El-Nouby, Z. Liu, M. Singh, K. V. Alwala, A. Joulin, and I. Misra, “Imagebind: One embedding space to bind them all,” in IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2023, pp. 15 180–15 190.

[41] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2016, pp. 770–778.

[42] S. Woo, S. Debnath, R. Hu, X. Chen, Z. Liu, I. S. Kweon, and S. Xie, “Convnext v2: Co-designing and scaling convnets with masked autoencoders,” in IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2023, pp. 16 133–16 142.

[43] K. Sohn, D. Berthelot, N. Carlini, Z. Zhang, H. Zhang, C. A. Raffel, E. D. Cubuk, A. Kurakin, and C.-L. Li, “Fixmatch: Simplifying semisupervised learning with consistency and confidence,” Advances in neural information processing systems, vol. 33, pp. 596–608, 2020.

[44] B. Chen, J. Jiang, X. Wang, P. Wan, J. Wang, and M. Long, “Debiased self-training for semi-supervised learning,” Adv. Neural Inform. Process. Syst. (NIPS), vol. 35, pp. 32 424–32 437, 2022.

[45] V. Verma, A. Lamb, C. Beckham, A. Najafi, I. Mitliagkas, D. Lopez-Paz, and Y. Bengio, “Manifold mixup: Better representations by interpolating hidden states,” in Int. Conf. Mach. Learn. (ICML). PMLR, 2019, pp. 6438–6447.

[46] B. Dolhansky, J. Bitton, B. Pflaum, J. Lu, R. Howes, M. Wang, and C. C. Ferrer, “The deepfake detection challenge (dfdc) dataset,” arXiv preprint arXiv:2006.07397, 2020.

[47] J. Deng, J. Guo, E. Ververas, I. Kotsia, and S. Zafeiriou, “Retinaface: Single-shot multi-level face localisation in the wild,” in IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2020, pp. 5203–5212.

[48] F. Chollet, “Xception: Deep learning with depthwise separable convolutions,” in IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2017, pp. 1251–1258.

[49] M. Tan and Q. Le, “Efficientnet: Rethinking model scaling for convolutional neural networks,” in Int. Conf. Learn. Represent. (ICLR). PMLR, 2019, pp. 6105–6114.

[50] C. Feichtenhofer, H. Fan, J. Malik, and K. He, “Slowfast networks for video recognition,” in Int. Conf. Comput. Vis. (ICCV), 2019, pp. 6202– 6211.

[51] G. Bertasius, H. Wang, and L. Torresani, “Is space-time attention all you need for video understanding?” in Int. Conf. Mach. Learn. (ICML), vol. 2, 2021, p. 4.

[52] K. Li, Y. Wang, G. Peng, G. Song, Y. Liu, H. Li, and Y. Qiao, “Uniformer: Unified transformer for efficient spatial-temporal representation learning,” in Int. Conf. Learn. Represent. (ICLR), 2022.

[53] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein et al., “Imagenet large scale visual recognition challenge,” Int. J. Comput. Vis. (IJCV), vol. 115, no. 3, pp. 211–252, 2015.

[54] W. Kay, J. Carreira, K. Simonyan, B. Zhang, C. Hillier, S. Vijayanarasimhan, F. Viola, T. Green, T. Back, P. Natsev et al., “The kinetics human action video dataset,” arXiv preprint arXiv:1705.06950, 2017.