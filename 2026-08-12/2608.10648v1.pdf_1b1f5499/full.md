# Precise Top-Layer Fabric Segmentation for Fabric Destacking with Edge- and Shape-Aware Deep Networks

Wenbo Dong<sup>1,2</sup>, Dipankar Bhattacharya<sup>1,2</sup>, Akinari Kobayashi<sup>1,2</sup>, Akira Seino<sup>1,2</sup>, Fuyuki Tokuda<sup>1,2</sup>, Xuzhao Huang<sup>1,2</sup>, Kai Tang<sup>1,2</sup>, Norman C. Tien<sup>2,3</sup>, and Kazuhiro Kosuge<sup>1,2</sup>

<sup>1</sup>JC STEM Lab of Robotics for Soft Materials, Department of Electrical and Electronic Engineering, Faculty of Engineering, The University of Hong Kong, Hong Kong SAR, China <sup>2</sup>Centre for Transformative Garment Production, Hong Kong SAR, China <sup>3</sup>Department of Electrical and Electronic Engineering, Faculty of Engineering, The University of Hong Kong, Hong Kong SAR, China Corresponding Author: dongwbo@connect.hku.hk Code: https://github.com/bhattner143/top-layer-fab-seg

Abstract—Fabric destacking requires precise segmentation of the topmost fabric layer, a task complicated by subtle fabric boundaries and high visual similarity between fabric layers. Existing semantic and edge-based segmentation approaches often struggle with these complexities, limiting the performance of robotic manipulation for different tasks. In this work, a novel segmentation training architecture tailored for top-layer fabric segmentation in stacked fabrics is proposed. The method extends the classical encoder-decoder framework by introducing two specialized branches—an edge-aware branch and a shapeaware branch—that are used to supervise the backbone network for better tuning. The edge-aware branch enhances boundary delineation, while the shape-aware branch guides the network to capture and align the overall fabric shape with reference masks derived from Computer Aided Design (CAD) models. Experiments on a real-world fabric dataset demonstrate that the training approach outperforms established baselines, verifying the effectiveness of the multi-branch design through both quantitative results and ablation studies.

Index Terms—Automated fabric destacking, fabric segmentation, multi-branch network, encoder-decoder.

## I. INTRODUCTION

Fabric destacking is a complex process that increasingly relies on automation to enhance efficiency and consistency in garment production. A critical and challenging aspect of this automation is the manipulation of the top-layer fabric from fabric stacks (Fig. 1 (a)), which underpins key manufacturing steps such as picking and placing [1], fabric alignment, folding, and sewing of flexible fabric components [2], [3]. Therefore, generating accurate top-layer segmentation (Fig. 1 (b)) is essential for enabling reliable grasping, unfolding, and sewing automation. However, achieving precise top-layer segmentation in real-world settings presents several unique challenges: (1) the boundaries between stacked fabric layers are often subtle, irregular, or visually ambiguous; (2) the top layer and the background typically exhibit highly similar color and texture features, as similar fabrics are commonly stored together; and (3) imperfections along the boundaries of real fabrics can further interfere with the accurate segmentation of the topmost layer.

![](images/5b36b54993f39ad38d47a470f9f6a9f714029c602c396cbbeb37e4cb93eca03f.jpg)

(a) Stacked fabric layer image.  
![](images/c7165d96d5c99f640b50e8665fdf09ee2cc4a7d1820c458d03b94c12e001352a.jpg)  
(b) Top-layer segmentation.  
Fig. 1: Fabric stack image and top-layer segmentation.

Conventional semantic segmentation methods, which classify each pixel in an image of clothing into different parts or types of garments (such as sleeves, collars, or skirts), have achieved significant progress in generic image segmentation tasks. Architectures like UNet [4] are notable examples of this advancement. However, such methods do not effectively handle subtle boundaries and high similarity between stacked fabric layers and the background. To address this, edgebased segmentation methods approaches such as GSCNN [5], have proven effective in scenarios where object contours are ambiguous or subtle. However, these methods often struggle when faced with the high similarity and complex boundaries characteristic of layered fabric scenarios. In summary, current approaches often fall short in enabling precise and efficient top-layer fabric segmentation, directly affecting the performance of robotic manipulation.

Hence, there remains a clear need for a robust method that can accurately handle the fabric’s subtle boundaries and high visual similarity while remaining practical for real-world use.

To address these challenges in segmenting the top-layer in stacked fabrics, this work proposes a novel training architecture, as shown in Fig. 2 (a), consisting of two additional branches to enhance the segmentation accuracy during the training of the segmentation backbone network. The architecture builds upon the classical encoder-decoder architecture by incorporating two specialized branches for training supervision.

• Edge-aware branch: The edge-aware branch is specifically added to focus on the edges of the fabric. This helps supervising the segmentation backbone network to accurately outline the edges of the top-layer fabric and overcome the problem of unclear or blurry boundaries that often occur with standard segmentation methods.

• Shape-aware branch: The shape-aware branch focuses on understanding the overall shape of the fabric by learning features that represent the entire predicted fabric segmentation region. It aligns these features with an ideal reference binary image from the CAD model, serving as a regularization step to ensure segmentation results are accurate both at the pixel level and in overall shape.

As a result of adding these branches, the trained backbone network can infer segmented top-layer fabric (Fig. 2 (b)) that remains realistic and consistent with actual fabric shapes, even when the input image contains noise or visually ambiguous areas. It should be noted that for inference, only the segmentation backbone is used, with weights enhanced by edge and shape-aware supervision for improved results. Experiments conducted on a real-world fabric dataset demonstrate that the method achieves superior accuracy compared to other models. Ablation studies further confirm the increase of the segmentation accuracy by adding the two branches during training. Fig. 1 (a) shows an example input image of stacked fabrics, while Fig. 1 (b) provides the segmentation result obtained using the proposed method.

The remainder of the paper is organized as follows. The related work is reviewed in Section II. Section III provides the methodology to design the training architecture. Section IV provides the experimental results, which includes ablation studies and qualitative results, and finally Section V provides

the conclusion.

## II. RELATED WORK

Semantic segmentation methods: Encoder-decoder architectures, such as UNet [4], SegNet [6], and the DeepLab series [7], have become the standard paradigm for semantic segmentation tasks. These models extract hierarchical features via the encoder and progressively restore spatial resolution through the decoder, often employing skip connections to recover fine details. Recent advances, including atrous convolutions, pyramid pooling [8], and attention mechanisms [9], have further improved segmentation accuracy by enhancing multi-scale context and feature representation. However, these general-purpose frameworks do not explicitly address the subtle boundaries and high similarity between stacked fabric layers and the background, which are critical challenges in layered fabric segmentation.

Edge-based segmentation methods: In scenarios where object contours are ambiguous or subtle, precise edge localization is essential for achieving high-quality segmentation. GSCNN [5] proposes a Gated-Shape Convolutional Neural Network (CNN) architecture that fuses edge and region cues via a gated module. In contrast, DCAN [10] introduces deep contour-aware supervision to produce sharper mask boundaries. CASENet [11] and BiseNet [12] leverage edge-aware branches or edge-sensitive modules to improve contour prediction, and BASNet [13] as well as EGNet [14] further focus on edge refinement for salient object segmentation. However, these methods work for generic object or segmenting objects which are visually prominent, and do not sufficiently address the highly similarity characteristic of layered fabric scenarios. In contrast, the proposed training approach integrates a lightweight edge detection branch supervising the segmentation backbone, providing direct edge supervision for the unique challenges of top-layer fabric segmentation, thereby further enhancing accuracy and robustness.

Shape-based segmentation methods: Incorporating global shape information has proven effective for regularizing segmentation outputs, especially in domains requiring anatomical or geometric consistency. Some methods employ adversarial training or shape priors to encourage plausible segmentation boundaries [15], [16], while others utilize distance transforms or shape-aware losses as auxiliary supervision to enforce geometric regularity [17]. However, many of these approaches are designed for anatomical or organ segmentation tasks where the target and background are relatively well-defined and distinct. They often require additional annotation or complex training strategies, and may introduce significant computational overhead. In contrast, the proposed method adopts a compact CNNbased shape feature extractor that directly aligns the predicted segmentation with ideal shape features, coming from the CAD model. This design not only reduces computational complexity but also provides effective global regularization on the fabric shape, making it well-suited for the unique challenges of toplayer fabric segmentation.

![](images/fc151d7ed60b33b93b89048633bb34b33bdd0e04c6403e4f44f09e9975fe7cfe.jpg)  
Fig. 2: Proposed training architecture and inference scheme.

Multi-task and auxiliary supervision methods: Multitask learning has been widely used to improve segmentation by leveraging auxiliary tasks such as edge detection, depth estimation, or object detection [10], [18], [19], [20], [21], [22]. Jointly optimizing multiple objectives enables the network to learn richer representations and improves generalization, particularly when labeled data is limited. Previous works have explored multi-branch or unified architectures that integrate auxiliary cues to enhance segmentation accuracy [10], [19], [20]. However, many of these approaches focus on generic object scenes with clear boundaries. In contrast, the proposed method is designed as a lightweight, task-specific multi-branch architecture, where each branch (fabric edge and shape) addresses a distinct structural aspect of the fabric relevant to layered fabric segmentation.

## III. METHODOLOGY

To address the challenge of segmenting the topmost fabric layer, this section first discusses the task definition and then presents the proposed novel training architecture. The architecture, illustrated in Fig. 2, builds upon a classical encoder-decoder segmentation backbone and introduces additional edge-aware and shape-aware branches for training the backbone.

## A. Task definition and supervised annotations

Let $I \in \mathbb { R } ^ { H \times W \times 3 }$ denote the input RGB image of a fabric stack (Fig 2), where H and W represent the image height and width, and 3 corresponds to the color channels (red, green, and blue). The aim is to train a segmentation network that predicts a segmentation probability map $\hat { M } \in [ 0 , 1 ] ^ { H \times W }$ , where each element $\hat { M } _ { i , j }$ indicates the probability that pixel (i, j) belongs to the top-layer fabric. A binary segmentation mask can then be obtained by

$$
\hat { M } _ { \mathrm { m a s k } } = \operatorname { S o f t m a x } ( \hat { M } ) ,\tag{1}
$$

where Softmax denotes softmax function and $\hat { M } _ { \mathrm { m a s k } } \in$ $\{ 0 , 1 \} ^ { H \times W }$ . This process is illustrated in Fig. 2 (b). Hence, to obtain such a trained network and provide rich structural supervision, each training sample is annotated with the following

• Segmentation ground-truth mask $M _ { \mathrm { { g t } } } \in \{ 0 , 1 \} ^ { H \times W }$ Manually labeled mask indicating the top-layer fabric region.

• Ground-truth edge mask $E _ { \mathrm { g t } } \in \{ 0 , 1 \} ^ { H \times W } ;$ : Derived from $M _ { \mathrm { g t } }$ to highlight the top-level fabric edges.

• Reference shape mask $\bar { S } \in \{ 0 , 1 \} ^ { H \times W }$ : A binary mask generated from the CAD model, representing the reference shape of the top-layer fabric in the image plane.

• Shape alignment label $P ~ \in ~ \{ 0 , 1 \}$ : A binary label indicator showing whether the predicted segmentation mask is aligned with the reference shape $( P \ = \ 1$ for alignment, $P = 0$ otherwise).

During training, the network receives the image I as input and is jointly supervised by these annotations, enabling accurate segmentation, precise boundary localization, and keep the overall fabric shape consistent.

## B. Training network architecture overview

As depicted in Fig. 2 (a), the proposed training architecture consists of three main components: a segmentation backbone, an edge-aware branch and a shape-aware branch.

1) Segmentation backbone: The network adopts an encoderdecoder structure with skip connections. The encoder utilize a pre-trained ResNet50 backbone to extract hierarchical features from the input image at multiple spatial resolutions. As the image is processed through successive layers of the encoder, the spatial dimensions are progressively reduced while the depth of the feature representations increases. This process produces a set of feature maps, denoted as $\{ F _ { 1 } , F _ { 2 } , F _ { 3 } , F _ { 4 } \}$ where $F _ { 1 }$ corresponds to low-level, high-resolution features, and $F _ { 4 }$ captures high-level, low-resolution semantic information. Each feature map $F _ { i }$ encodes information at a different level of abstraction, enabling the network to learn both finegrained details and global context.

The decoder reconstructs the segmentation mask by progressively upsampling these encoded features through a sequence of upsampling modules. At each stage, the decoder concatenates the upsampled output from the previous layer with the corresponding high-resolution feature map from the encoder via skip connections. This fusion preserves finegrained spatial details that are critical for accurate segmentation. Each concatenated feature map is then refined by a stack of convolutional layers and nonlinear activations.

In this architecture, the decoder output $F _ { 1 }$ from the final upsampling block $D _ { 1 }$ is used for generating the segmentation mask. This choice is made because $F _ { 1 }$ has the highest spatial resolution among all decoder outputs, matching the input image size. Utilizing $F _ { 1 }$ ensures that the predicted segmentation mask preserves fine-grained spatial details and aligns accurately with the original input. The predicted segmentation probability map is obtained by applying

$$
\hat { M } = \sigma \left( \mathrm { C o n v } ( F _ { 1 } ) \right) ,\tag{2}
$$

where Conv(·) is convolution and $\sigma ( \cdot )$ is sigmoid activation function.

2) Edge-aware branch: To explicitly address the challenge of imprecise object boundaries, an edge-aware branch is added to the proposed network architecture, operating in parallel with the segmentation backbone. This branch takes $F _ { 1 }$ as input and processes it through a convolutional layers followed by nonlinear activations. The resulting feature map is then projected to a single-channel edge logits map via ${ \mathrm { ~  ~ { ~ \sigma ~ } ~ } } _ { 1 } { \mathrm { ~  ~ { ~ \sigma ~ } ~ } } _ { 1 }$ convolution. A sigmoid activation is then applied to obtain a predicted edge probability map $\hat { E } \in [ 0 , 1 ] ^ { 1 \times \hat { H } \times W }$ , given by

$$
\hat { E } = \sigma \left( \operatorname { C o n v } ( \phi ( F _ { 1 } ) ) \right) ,\tag{3}
$$

where each value indicates the likelihood of a pixel belonging to an object edge. Here, ϕ(·) denotes the sequence of convolutional layers with ReLU activations.

Supervision for the edge-aware branch is provided by the ground-truth edge mask $E _ { \mathrm { g t } }$ , encouraging the network to capture fine-grained and accurate object contours. This explicit edge modeling guides the network to better distinguish between adjacent objects and background, leading to sharper and more precise segmentation results.

3) Shape-aware branch: To regularize the global structure of the predicted fabric masks and suppress false positives, a shape-aware branch is implemented as a lightweight CNN. This branch begins by constructing a two-channel tensor, X, formed by concatenating the predicted segmentation probability map $\hat { M }$ and the reference shape label $S \in \{ 0 , 1 \} ^ { H \times W }$ generated from the CAD model. The concatenated tensor can be written as

$$
\boldsymbol { X } = \operatorname { C o n c a t } \left( \hat { M } , \boldsymbol { S } \right) \in \mathbb { R } ^ { 2 \times H \times W } ,\tag{4}
$$

Next, the input X is processed by a CNN, denoted by ψ(·), followed by a fully connected layer FC and a sigmoid activation to produce a predicted scalar probability

$$
\hat { P } = \sigma \left( \mathrm { F C } ( \psi ( X ) ) \right) ,\tag{5}
$$

where $\hat { P } \in [ 0 , 1 ]$ indicates the predicted likelihood that the segmentation aligns with the ideal shape.

Pre-training: To provide a robust understanding of the fabric shape, the shape-aware branch is first pre-trained as an independent binary classifier on synthetic mask pairs. The pre-trained weights are then used to initialize the shape-aware branch, allowing it to be further fine-tuned jointly with the entire network. This approach gives the network effective global structural guidance, helping it distinguish between plausible from implausible shapes, and improving both segmentation reliability and edge accuracy.

![](images/303ee7e4b5f48f35f52b110538d2a421e4a1755d038274ff532313f347da9693.jpg)

![](images/c844f180b1c721d88f22c4438500b8d71dc1ddd1e0296785143c7684b9e7ec0d.jpg)  
Input image

![](images/d268f0b9cf474e9f577b98cd9ff3a25684920acb41dd034913e1c1f00bbbbb3c.jpg)  
Ground-truth  
Prediction

Fig. 3: Qualitative segmentation results. Quantitative comparison is included in Table I.

4) Training losses: The network is trained with a composite loss (Fig. 2 (a), yellow blocks) that jointly optimizes segmentation accuracy, edge and shape regularity. The overall loss function is defined as

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = { \mathcal { L } } _ { \mathrm { s e g } } + \lambda _ { \mathrm { e d g e } } { \mathcal { L } } _ { \mathrm { e d g e } } + \lambda _ { \mathrm { s h a p e } } { \mathcal { L } } _ { \mathrm { s h a p e } } ,\tag{6}
$$

where $\lambda _ { \mathrm { e d g e } }$ and $\lambda _ { \mathrm { s h a p e } }$ are hyperparameters that balance the contributions of the edge and shape-aware branches. The segmentation loss $\mathcal { L } _ { \mathrm { s e g } }$ combines two components: the standard pixel-wise cross-entropy loss, which checks how well each pixel in the predicted mask matches the ground-truth mask, and the Dice loss, which measures the overall similarity between the predicted segmentation mask M<sup>ˆ</sup> and the ground-truth mask $M _ { \mathrm { g t } }$ . The edge loss $\mathcal { L } _ { \mathrm { e d g e } }$ is defined as the binary cross-entropy between the predicted edge mask $\hat { E }$ and the ground-truth edge mask $E _ { \mathrm { g t } }$ , thereby promoting precise boundary detection. For the shape-aware branch, $\mathcal { L } _ { \mathrm { s h a p e } }$ is computed as the binary cross-entropy between the predicted scalar probability $\hat { P }$ and the shape label $P ,$ which indicates whether the predicted segmentation matches with the ideal shape.

5) Inference: During inference (Fig. 2 (b)), only the segmentation backbone network is used taking a camera image as input and producing a segmented mask as output. Additionally, the network’s weights have been improved during training with supervision from the edge and shape-aware branches to achieve better segmentation results. The final segmentation output is produced using softmax and argmax operations.

## IV. EXPERIMENTS

Dataset description: To evaluate the effectiveness of the proposed network, experiments were conducted on a real collar fabric dataset comprising 235 images with corresponding pixel-level annotations. Each image is accompanied by a semantic label mask, a ground-truth edge mask, and an ideal shape label derived from the corresponding CAD model.

Implementation details: The proposed network is implemented using PyTorch. The Adam optimizer is employed, using an initial learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 8. Training is conducted over 300 epochs, with early stopping applied based on validation loss to prevent overfitting. All experiments are performed on a workstation equipped with an NVIDIA RTX 3060 GPU (12GB VRAM), an Intel Core i9-10900F CPU, and 16 GB of RAM. Model training and evaluation are conducted under CUDA 11.7 with cuDNN acceleration enabled.

Evaluation metrics: To comprehensively assess the segmentation performance, the following standard metrics were utilized

1) Intersection over Union (IoU): Measures the overlap between the predicted and ground truth regions, and is defined as

$$
\mathrm { I o U } = { \frac { T P } { T P + F P + F N } } ,\tag{7}
$$

where TP, FP, and FN denote the numbers of true positive, false positive, and false negative pixels, respectively.

2) Pixel Accuracy (PA): Computes the proportion of correctly predicted pixels:

$$
\mathrm { P A } = { \frac { T P + T N } { T P + F P + F N + T N } } ,\tag{8}
$$

where TN denotes the number of true negative pixels. 3) Edge Mean Squared Error (EMSE) : This and the following metrics evaluate the spatial accuracy of the predicted segmentation boundaries. Given the set of predicted edge points $\mathbf { \mathcal { P } } = \{ \mathbf { p } _ { i } \} _ { i = 1 } ^ { N }$ and the set of edge ground truth points $\mathcal { G } = \{ \mathbf { g } _ { j } \} _ { j = 1 } ^ { M }$ , the EMSE is defined as the average squared Euclidean distance between each predicted edge point and its nearest ground truth edge point, and between each ground truth edge point and its nearest predicted edge point, averaged over both directions, and given by

$$
\mathrm { E M S E } = \frac { 1 } { 2 } ( \mathrm { M S E } _ { \mathcal { P }  \mathcal { G } } + \mathrm { M S E } _ { \mathcal { G }  \mathcal { P } } ) ,\tag{9}
$$

where

$$
\mathrm { M S E } _ { \mathcal { P }  \mathcal { G } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( \operatorname* { m i n } _ { \mathbf { g } _ { j } \in \mathcal { G } } \| \mathbf { p } _ { i } - \mathbf { g } _ { j } \| _ { 2 } ) ^ { 2 } ,
$$

$$
\mathrm { M S E } _ { \mathcal { G }  \mathcal { P } } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } ( \operatorname* { m i n } _ { \mathbf { p } _ { i } \in \mathcal { P } } \| \mathbf { g } _ { j } - \mathbf { p } _ { i } \| _ { 2 } ) ^ { 2 } .
$$

4) Edge Root Mean Squared Error (ERMSE): The ERMSE is then defined as

$$
\mathrm { E R M S E } = \frac { 1 } { 2 } ( \sqrt { \mathrm { M S E } _ { \mathcal { P }  \mathcal { G } } } + \sqrt { \mathrm { M S E } _ { \mathcal { G }  \mathcal { P } } } ) ,\tag{10}
$$

where ∥ · ∥<sub>2</sub> denotes the Euclidean distance between two points.

Ablation study: To analyze the contribution of each architectural component, ablation studies were conducted. Specifi cally, the following network architectures were compared

1) Baseline: Standard segmentation network without edgeor shape-aware branches.

2) Baseline + edge-aware branch: Segmentation network with added edge-aware branch only.

3) Baseline + edge-aware branch + shape-aware branch (Proposed architecture): Full model with both edgeaware branch and shape-aware branch.

Table I summarizes the quantitative results. The addition of the edge-aware branch improves both IoU and PA, while also reducing the ERMSE. This demonstrates that the edge-aware branch helps the network produce more accurate and sharper segmentation masks. Adding the shape-aware branch after the edge-aware branch further improves all performance metrics. In particular, the shape-aware branch enhances the network’s ability to capture the global structure of the fabric, resulting in higher IoU and PA, as well as the lowest ERMSE, which indicates improved alignment between the predicted and ideal shapes. These results confirm that both the edge-aware and shape-aware branches are beneficial and complementary for achieving accurate segmentation.

TABLE I: Ablation study of architectural components.
<table><tr><td>Model</td><td>IoU ↑(%)</td><td>PA ↑(%)</td><td>ERMSE ↓ (pixel)</td></tr><tr><td>Baseline</td><td>93.25</td><td>95.87</td><td>5.03</td></tr><tr><td>Baseline + EA</td><td>96.24</td><td>96.72</td><td>3.19</td></tr><tr><td>Baseline + EA + SA (Proposed architecture)</td><td>96.80</td><td>97.50</td><td>2.58</td></tr></table>

Note: EA: edge-aware branch; SA: shape-aware branch.

Qualitative results: Figure 3 presents qualitative segmentation results. As shown, the proposed architecture accurately delineates fabric boundaries and preserves the overall shape, even in challenging regions with ambiguous edges and complex contours. These qualitative results demonstrate that our method produces precise boundaries while maintaining the global integrity of the fabric shape.

## V. CONCLUSION

This paper presents a novel segmentation training architecture designed specifically for the challenging task of top-layer fabric segmentation in stacked scenarios. By integrating an edge-aware branch to refine boundary localization and a shapeaware branch to align predicted masks with Computer Aided Design (CAD) models, the proposed method effectively supervises and tunes the backbone network to address challenges from subtle fabric boundaries and high inter-layer similarity. Experiments on real-world fabric datasets demonstrate that both supervisory branches enhance segmentation accuracy, especially at fabric edges. The architecture also shows robustness to limited training data and outperforms strong baselines.

Future work: The proposed top-layer fabric segmentation method will be implemented in our pick-and-place system [23], and will demonstrate its versatility and practical applicability as future work.

## ACKNOWLEDGMENT

This work was supported in part by the Innovation and Technology Commission of the HKSAR Government under the InnoHK initiative. The research described in this paper was conducted in part at the JC STEM Lab of Robotics for Soft Materials, funded by The Hong Kong Jockey Club Charities Trust.

## REFERENCES

[1] A. Seino, F. Tokuda, A. Kobayashi, and K. Kosuge, “Passive actuatorless gripper for pick-and-place of a piece of fabric,” IEEE/ASME Transactions on Mechatronics, 2025.

[2] K. Tang, F. Tokuda, A. Seino, A. Kobayashi, N. C. Tien, and K. Kosuge, “Time-scaling modeling and control of robotic sewing system,” IEEE/ASME Transactions on Mechatronics, 2024.

[3] F. Tokuda, R. Murakami, A. Seino, A. Kobayashi, M. Hayashibe, and K. Kosuge, “Fixture-free 2d sewing using a dual-arm manipulator system,” IEEE Transactions on Automation Science and Engineering, 2024.

[4] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer, 2015, pp. 234–241.

[5] T. Takikawa, D. Acuna, V. Jampani, and S. Fidler, “Gated-scnn: Gated shape cnns for semantic segmentation,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 5229–5238.

[6] V. Badrinarayanan, A. Kendall, and R. Cipolla, “Segnet: A deep convolutional encoder-decoder architecture for image segmentation,” IEEE transactions on pattern analysis and machine intelligence, vol. 39, no. 12, pp. 2481–2495, 2017.

[7] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, “Encoderdecoder with atrous separable convolution for semantic image segmentation,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 801–818.

[8] H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia, “Pyramid scene parsing network,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 2881–2890.

[9] J. Fu, J. Liu, H. Tian, Y. Li, Y. Bao, Z. Fang, and H. Lu, “Dual attention network for scene segmentation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 3146– 3154.

[10] H. Chen, X. Qi, L. Yu, and P.-A. Heng, “Dcan: deep contour-aware networks for accurate gland segmentation,” in Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2016, pp. 2487–2496.

[11] Z. Yu, C. Feng, M.-Y. Liu, and S. Ramalingam, “Casenet: Deep categoryaware semantic edge detection,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5964–5973.

[12] C. Yu, J. Wang, C. Peng, C. Gao, G. Yu, and N. Sang, “Bisenet: Bilateral segmentation network for real-time semantic segmentation,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 325–341.

[13] X. Qin, Z. Zhang, C. Huang, C. Gao, M. Dehghan, and M. Jagersand, “Basnet: Boundary-aware salient object detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 7479–7489.

[14] J.-X. Zhao, J.-J. Liu, D.-P. Fan, Y. Cao, J. Yang, and M.-M. Cheng, “Egnet: Edge guidance network for salient object detection,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 8779–8788.

[15] O. Oktay, E. Ferrante, K. Kamnitsas, M. Heinrich, W. Bai, J. Caballero, S. A. Cook, A. De Marvao, T. Dawes, D. P. O‘Regan et al., “Anatomically constrained neural networks (acnns): application to cardiac image enhancement and segmentation,” IEEE transactions on medical imaging, vol. 37, no. 2, pp. 384–395, 2017.

[16] Y. Xue, T. Xu, H. Zhang, L. R. Long, and X. Huang, “Segan: Adversarial network with multi-scale l 1 loss for medical image segmentation,” Neuroinformatics, vol. 16, pp. 383–392, 2018.

[17] S. Chen, C. Luo, S. Liu, H. Li, Y. Liu, H. Zhou, L. Liu, and H. Chen, “Ld-unet: A long-distance perceptual model for segmentation of blurred boundaries in medical images,” Computers in Biology and Medicine, vol. 171, p. 108120, 2024.

[18] A. Kendall, Y. Gal, and R. Cipolla, “Multi-task learning using uncertainty to weigh losses for scene geometry and semantics,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 7482–7491.

[19] I. Kokkinos, “Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 6129–6138.

[20] Z. Zhang, H. Fu, H. Dai, J. Shen, Y. Pang, and L. Shao, “Etnet: A generic edge-attention guidance network for medical image segmentation,” in Medical Image Computing and Computer Assisted Intervention–MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13–17, 2019, Proceedings, Part I 22. Springer, 2019, pp. 442–450.

[21] Z. Xie, J. Chen, Y. Feng, K. Zhang, and Z. Zhou, “End to end multi-task learning with attention for multi-objective fault diagnosis under small sample,” Journal ofManufacturing Systems, vol. 62, pp. 301–316, 2022.

[22] R. Caruana, “Multitask learning,” Machine learning, vol. 28, pp. 41–75, 1997.

[23] A. Kobayashi, W. Dong, A. Seino, F. Tokuda, and K. Kosuge, “Rollup: Rolling-up end-effector for fabric handing,” Authorea Preprints, 2025.