# Binarized High-Efficiency RAW Video Restoration and Beyond

Tianyu Zhu, Ying Fu, Senior Member, IEEE, Hesong Li, Gengchen Zhang, Xin Yuan, Senior Member, IEEE, and Yulun Zhang, Member, IEEE

Abstract—RAW video restoration is fundamental to highquality low-level perception and serves as the basis for a wide range of downstream vision applications. While binary neural networks (BNNs) enable efficient lightweight deployment for image enhancement, their deficiencies in modeling temporal coherence and activation value distributions hinder their effectiveness when applied to video scenarios. In this paper, we propose BinRVR, a binarized RAW video restoration framework that reduces computation and parameters by approximately 96% while incurring only about 4% performance degradation. Specifically, we present a Binarized Information Interaction Module (BIIM) to jointly model spatial and temporal information in an efficient and unified manner. Moreover, we develop a Distribution-Aware Binarized Convolution (DAB-Conv) that leverages the statistics of full-precision activations to mitigate quantization errors. The proposed framework further supports multi-bit quantization, enabling flexible accuracy-efficiency trade-offs across different hardware constraints. Extensive experiments demonstrate that our BinRVR achieves competitive performance compared with state-of-the-art binarized methods on RAW video restoration tasks, including low-light enhancement, denoising, deblurring, and super-resolution. We further explore the potential of our method on downstream video applications, including object detection and monocular depth estimation.

Index Terms—Binary neural network, RAW video restoration, low-light video enhancement, video denoising, video superresolution, quantization, efficiency.

## I. INTRODUCTION

IDEOS captured in real-world scenarios inevitably suffer from quality degradation due to sensor limitations   
and challenging environmental conditions, resulting in low   
illumination, noise, motion blur, and low resolution. Such   
degradations not only impair human visual perception but   
also undermine the performance of downstream vision tasks,   
including object detection [1], [2], depth estimation [3], [4],   
and segmentation [5], [6]. To address these challenges, a large

![](images/31b2e1f5caf412fe09360978fb0460802fde57971aea6c8ed9b95c942a411dd1.jpg)  
(a) Low-Light RAW Video Enhancement

![](images/d0f6fb4722199adffb25787f1da653ac19f34df7c50633f6aff9dcac65dd395d.jpg)  
(b) RAW Video Denoising

![](images/2895fcd01fd833311ecdf485925a3e448bdb0a44b4e39b56e16e21655c64e5e1.jpg)  
(c) RAW Video Deblurring

![](images/203816dbf4e4306f3faf3e910d7cb1d3d2b031733cb919b42405cf5816a85b2c.jpg)  
(d) RAW Video Super-Resolution  
Fig. 1. Performance and model complexity comparison with state-of-theart binary neural networks (BNNs) and lightweight full-precision methods in RAW video restoration. The size of each circle reflects the model’s parameter count. Our method outperforms both the BNN-based methods and lightweight full-precision methods with comparable model complexity.

body of research has been devoted to video restoration, aiming to recover high-quality videos from degraded inputs.

Existing video restoration methods can be broadly divided into two categories according to the domain of the input and output videos, i.e., RGB-based methods [7], [8], and RAW-based methods [9], [10]. Since RAW data bypass the nonlinear processing of the image signal processor (ISP), they more faithfully preserve scene irradiance and retain original image information [10], [11], [12]. Unlike RGB frames that undergo irreversible ISP operations such as white balance, tone mapping, and denoising, RAW data maintain a nearly linear sensor response and a wider dynamic range, which are crucial for recovering fine structures and low-light details. Therefore, RAW videos are more suitable as both input and output representations to maximally preserve information.

Deep neural networks (DNNs) have achieved remarkable performance in RAW video restoration [7], [8], [13]. However, their high computational cost and large parameter size hinder deployment on resource-constrained devices such as smartphones, unmanned aerial vehicles (UAVs), and Internet of Things (IoT) platforms. To address this limitation, various acceleration and compression techniques have been explored, including pruning [14], [15], lightweight full-precision architecture design [16], knowledge distillation [17], quantization [18], and binarization [19]. These techniques improve efficiency from different perspectives and are not mutually exclusive. Among them, network quantization has attracted increasing attention due to its broad applicability across architectures and its ability to reduce computation and storage. Binary Neural Networks (BNNs) represent an extreme form of quantization, where both weights and activations are constrained to 1-bit values, typically represented as +1 and −1. When supported by appropriate hardware kernels, binarization enables compact storage and bitwise computation, making it a promising direction for efficient RAW video restoration.

Despite their efficiency, applying BNNs to RAW video restoration remains challenging. First, most existing studies have concentrated on tasks such as classification and singleimage restoration [20], whereas inter-frame relationships in BNN-based video restoration have received little attention. Second, many state-of-the-art video restoration models [7], [21] rely on complex operations such as multi-head attention, optical flow estimation, and deformable convolutions to capture spatio-temporal features. These operations are not naturally friendly to BNNs, where weights and activations are represented by signs, i.e., +1 and −1. Sign-based features discard magnitude and fine ordering information. For attention modules, quantized Vision Transformer studies show that attention maps and softmax-related nonlinear operations often require special quantization-aware designs [22], [23]. For opticalflow-based alignment, reliable flow estimation depends on discriminative feature matching and continuous displacement prediction [24], [25], while deformable convolution predicts input-dependent offsets to determine continuous sampling locations [21], [26]. Inferring these similarities, flows, or offsets from coarse binary cues can shift sampling locations and amplify alignment errors during feature aggregation. Third, BNNs suffer from internal discrepancies, as the distributions of weights and activations vary across layers, channels, frames, and degradation types [27], making uniform sign-based binarization prone to substantial quantization errors.

In this paper, we propose a Binarized High-Efficiency RAW Video Restoration model (BinRVR) to address the above challenges. Different from existing binarized restoration methods that mainly focus on image-level tasks, BinRVR provides a systematic binarized solution for multi-task RAW video restoration, including low-light enhancement, denoising, deblurring, and super-resolution. Specifically, BinRVR employs a unidirectional recurrent structure that enables efficient inference with minimal memory overhead, making it particularly suitable for deployment on resource-limited devices. To address the severe representational constraints imposed by binarized RAW video restoration, we present a Binarized Information Interaction Module (BIIM) for joint modeling of temporal dependencies and spatial structures. BIIM integrates a temporal shift-and-extend operation with grouped stripshaped spatial convolutions, enabling efficient spatiotemporal interaction with negligible overhead and full compatibility with binarized and low-bit quantized models. To further reduce quantization error, we develop a Distribution-Aware Binarized Convolution (DAB-Conv), which employs channel attention to derive real-valued scaling factors from activations, thereby incorporating full-precision distribution information. Extensive experiments demonstrate that our BinRVR substantially reduces computation and parameters with only minor performance degradation, and achieves a favorable accuracyefficiency trade-off compared with state-of-the-art binarized methods and representative lightweight full-precision models under comparable complexity, as shown in Fig. 1.

To summarize, our main contributions are as follows:

• We propose BinRVR, a high-efficiency binarized framework for multi-task RAW video restoration, explicitly extending binarization from image-level restoration to temporally coherent RAW video restoration;

• We introduce a Binarized Information Interaction Module (BIIM) that enables efficient joint modeling of inter-frame temporal dependencies and intra-frame spatial structures through lightweight and binarization-friendly operations;

• We develop a Distribution-Aware Binarized Convolution (DAB-Conv) that leverages activation distribution information to mitigate quantization errors and naturally supports multi-bit quantization for flexible accuracyefficiency trade-offs under hardware constraints.

The remainder of this paper is organized as follows. Section II reviews related works on video restoration and binary neural networks. Section III describes the proposed BinRVR framework in detail, including the BIIM and DAB-Conv. Section IV reports extensive experimental results, including quantitative evaluations, visual comparisons, and discussions. Finally, Section V concludes the paper.

## II. RELATED WORKS

In this section, we review related work in two parts. We first discuss video restoration methods, especially those involving RAW data, and then review BNNs with quantization.

## A. Video Restoration

Video restoration encompasses tasks such as low-light video enhancement, video denoising, video deblurring, and video super-resolution. Existing methods can be broadly categorized into single-task methods [13], [28]–[34] and multi-task frameworks [7], [8], [35], [36]. In the following, we review representative methods from two perspectives, namely overall architecture design and spatio-temporal feature fusion.

Network architecture design. Early DNN-based video restoration methods typically took all frames of a video as input to the model [37], [38], but this design is unsuitable for variable-length videos and incurs high storage costs. Recent methods mainly adopt either multi-frame sliding-window architectures [39], [21], [10], [40], [41] or single-frame recurrent structures [8], [28], [29], [13], [42]. Chen et al. [41] fused features from a sliding window with single-frame spatial features and produced multi-frame output. Chan et al. [29] enhanced the bidirectional recurrent framework with additional skip connections. To combine the advantages of both, Liang et al. [7] proposed a single-frame-to-single-frame architecture with a global flow estimation module. Despite these advances, sliding-window methods are effective at capturing short-term dependencies but are less effective in modeling long-term relations, whereas recurrent structures leverage hidden states for long-term modeling but may be limited in short-term correlations and are prone to error accumulation.

![](images/f92b32212d7ece34c3a63d57d344a4f51f0f564beceee0ece8ecdba38be7939b.jpg)  
Fig. 2. Illustration of the proposed BinRVR. (a) Overall architecture of our binarized RAW video restoration framework. The recurrent arrows indicate the propagated features between adjacent sliding windows. Efficient inter-frame and intra-frame information interaction under binarization is achieved by the proposed Binarized Information Interaction Module (BIIM), whose detailed structure is shown in Fig. 3. (b) Distribution-Aware Binarized Convolution (DAB-Conv), which leverages feature distribution information to reduce quantization error and improve representation capability in binarized networks.

Spatio-temporal feature interaction. Unlike image restoration [43]–[51], which mainly focuses on spatial information, video restoration additionally requires effective extraction and fusion of temporal cues. Some methods [52], [25], [7] explicitly align or warp frames using optical flow estimation. Others adopt deformable convolutions [26], [53], [21] or attention mechanisms [7], [54], [55] to implicitly fuse spatio-temporal features across frames. However, these approaches incur high computational and memory costs, and their performance degrades noticeably under quantization, especially binarization. To reduce complexity, lightweight fusion strategies have been explored. Li et al. [8] aggregated multi-frame features effectively by applying group-wise spatial and temporal shifts within feature channels. Yue et al. [56] proposed a RAW video demoireing method that leverages channel-separated´ features and spatial modulation to exploit the complementary properties of RAW RGGB channels, while aligning multi-scale temporal features. Despite these efforts, lightweight spatiotemporal feature fusion remains underexplored in the context of quantization and binarization.

## B. Binary Neural Networks and Quantization

Binary neural networks (BNNs) are proposed as an extreme case of quantization to exploit the efficiency of bitwise operations on hardware [19], where both weights and activations are represented with only 1 bit. However, such extreme quantization inevitably leads to performance degradation. To mitigate this issue, extensive efforts have focused on reducing quantization errors [27], [57], [58], refining loss functions [59], [60], approximating gradients [61], [62], and optimizing training strategies [63], [64]. Recent methods further improve binary models through optimization-friendly binarization functions and high-accuracy binary architectures, such as BiPer [65] and BNext [66]. Most studies validate the effectiveness of BNNs on high-level vision tasks such as image classification and object detection [67], [68].

By contrast, applications of BNNs to low-level vision remain limited and mainly focus on image restoration [20], [69]. Representative works include BBCU [20], which emphasizes refining residual connections, batch normalization, and activation functions, Advanced BNN [70] and Rectified Binary Network [71] for binarized single-image super-resolution, and BHViT [72], which explores a hybrid convolution and Transformer design. These methods demonstrate the potential of binarization for image restoration, but they mainly focus on RGB-domain single-image tasks. Binarized RAW video restoration additionally requires efficient temporal modeling and robustness to RAW-domain activation distribution variations, which are the main focus of this work.

Compared with BNNs, multi-bit quantization provides less aggressive compression but generally yields better performance. Several studies have investigated multi-bit quantization for image super-resolution. QuantSR [73] introduced a lowbit framework with a learnable redistributive quantizer to improve information representation. It further employs a dynamic quantization architecture to adaptively adjust network depth. Hong et al. [74] proposed AdaBM, an adaptive framework that adjusts bit-widths according to image complexity and layer sensitivity. Nevertheless, extensions of quantization and binarization to video restoration remain scarce.

## III. BINARIZED RAW VIDEO RESTORATION

In this section, we formalize the RAW video restoration task and elaborate on the motivation for constructing a binarized model that balances computational efficiency and restoration quality. We then present the overall architecture of the proposed BinRVR framework, followed by a detailed description of the Binarized Information Interaction Module (BIIM) and the Distribution-Aware Binarized Convolution (DAB-Conv). Finally, we introduce a multi-bit compatible design for highefficiency RAW video restoration.

![](images/a7cdbe623232f2a28459d2e02025f0f566731159465b9780cd9edaa352a20cf3.jpg)  
Fig. 3. Detailed illustration of the proposed Binarized Information Interaction Module (BIIM). It is designed to enable efficient interaction between inter-frame temporal information and intra-frame spatial information under binarization through lightweight operations. (a) Binarized Information Interaction Encoder. (b) Binarized Information Interaction Decoder.

## A. Formulation and Motivation

We outline the formulations of RAW video restoration, quantization, and binarization. We then introduce the motivations of our BIIM and DAB-Conv.

Formulation of RAW video restoration. The task of RAW video restoration can be formulated as the transformation of an input sequence of low-quality (LQ) RAW video frames $\{ \mathbf { I } _ { 1 } ^ { \mathrm { L Q } } , \mathbf { \dot { I } } _ { 2 } ^ { \mathrm { L Q } } , \dots , \mathbf { I } _ { T } ^ { \mathrm { L Q } } \}$ , where each frame $\bar { \mathbf { I } } _ { i } ^ { \mathrm { L Q } }$ is a Bayer pattern RAW image with spatial resolution $H \times W$ and packed into a tensor of size $\mathbf { I } _ { i } ^ { \mathrm { L Q } ^ { \star } } \in \mathbb { R } ^ { 4 \times H / 2 \times W / 2 }$ . The pixel values of the RAW images are first uniformly normalized to the range [0, 1] based on the camera-specific black level and white level. The output is a sequence of high-quality (HQ) RAW video frames $\{ \hat { \mathbf { I } } _ { 1 } ^ { \mathrm { H Q } } , \hat { \mathbf { I } } _ { 2 } ^ { \mathrm { H Q } } , \hdots , \hat { \mathbf { I } } _ { T } ^ { \mathrm { H Q } } \}$ . As shown in Fig. 2(a), the process is expressed as

$$
\{ \hat { \bf \cal I } _ { 1 } ^ { \mathrm { H Q } } , \hat { \bf \cal I } _ { 2 } ^ { \mathrm { H Q } } , \ldots , \hat { \bf \cal I } _ { T } ^ { \mathrm { H Q } } \} = \mathcal { G } \left( \{ { \bf \cal I } _ { 1 } ^ { \mathrm { L Q } } , { \bf \cal I } _ { 2 } ^ { \mathrm { L Q } } , \ldots , { \bf I } _ { T } ^ { \mathrm { L Q } } \} ; \Theta \right) ,\tag{1}
$$

where G represents the restoration model and Θ denotes the learnable parameters of the model. The restoration process aims to recover high-quality details and maintain temporal consistency across the video sequence.

Formulation of binarization and quantization. We denote b as the quantization bit-width. Binary neural networks (BNNs) represent the extreme case of quantization under the single-bit setting $( b = 1 )$ , where weights and activations are constrained to $\{ + 1 , - 1 \}$ . The binarization mapping is written as

$$
B ( x ) = \mathrm { S i g n } ( x ) = \left\{ \begin{array} { l l } { + 1 , } & { x \ge 0 , } \\ { - 1 , } & { x < 0 , } \end{array} \right.\tag{2}
$$

with $B ( \cdot )$ denoting the binarization function. Storing +1 as 1 and −1 as 0 enables hardware-level acceleration. Multiplication is replaced by XNOR, and accumulation by bitcount [27], [75]. The binarized convolution is computed as

$$
\mathbf { F } ^ { \mathrm { b } } \otimes \mathbf { W } ^ { \mathrm { b } } = \mathrm { B i t c o u n t } \big ( \mathrm { X N O R } ( \mathbf { F } ^ { \mathrm { b } } , \mathbf { W } ^ { \mathrm { b } } ) \big ) .\tag{3}
$$

For $b \geq 2$ , multi-bit quantization maps full-precision values to a broader discrete set than binarization. With clipping range $[ x _ { \mathrm { m i n } } , x _ { \mathrm { m a x } } ]$ , we use the following uniform quantizer, i.e.,

$$
Q ( x ) = \left\lfloor \frac { \mathrm { { C l a m p } } ( x , x _ { \mathrm { m i n } } , x _ { \mathrm { m a x } } ) - x _ { \mathrm { m i n } } } { \delta } \right\rceil \cdot \delta + x _ { \mathrm { { m i n } } } ,\tag{4}
$$

where ⌊·⌉ denotes rounding to the nearest integer and δ is the quantization step size.

Motivation of binarized information interaction module. Modeling interaction between inter-frame temporal information and intra-frame spatial structures is crucial for video restoration, yet remains challenging under binarization. Existing methods often rely on cross-attention for temporal modeling [7], resulting in high computational overhead. Others attempt to enlarge the receptive field through downsampling [76], sacrificing spatial details. Techniques such as optical flow [52] and deformable convolution [26] can further enhance spatio-temporal modeling, but they rely on discriminative realvalued feature matching or continuous offset prediction, which becomes fragile when features are constrained to $+ 1 / \textrm { -- } 1$ signs. Inspired by the efficiency-oriented designs of [8], [77], these limitations motivate a binarized module for efficient spatio-temporal interaction in RAW videos.

Motivation of distribution-aware binarized convolution. Binarized convolutions are sensitive to activation distributions, as sign-based quantization discards magnitude information and introduces quantization errors [59]. Existing binarization methods [19], [20], [62] employ either per-layer or perchannel scaling strategies, and many representative methods estimate scaling factors mainly from the mean absolute value of weights or activations, which captures the average magnitude but provides limited distribution information. For RAW video restoration, activation distributions vary across channels, frames, and degradation conditions. Our DAB-Conv extracts multiple distribution descriptors, including the mean, absolute mean, and standard deviation, and adaptively fuses them to predict input-dependent channel-wise activation scaling factors. These complementary statistics help adapt activation scaling to complex RAW-domain feature distributions and reduce quantization-induced degradation during restoration.

![](images/736073ebf034bf1921250cf022879fce670e7e2194dacaf2ce27076baddfee5f.jpg)  
Fig. 4. Comparison of video restoration architectures. (a) Multi-frame sliding window framework. (b) Single-frame recurrent architecture. (c) Proposed sliding window recurrent connection.

## B. Overall architecture design

The overall architecture of the proposed BinRVR is illustrated in Fig. 2(a), which shows the restoration process for three consecutive RAW frames $\mathbf { I } _ { i } ^ { \mathrm { L Q } }$ from a video sequence. The framework consists of three stages. In the feature extraction stage, a binarized U-Net [78] extracts frame-wise features such as $\mathbf { F } _ { i } ^ { 1 }$ . In the recursive encoding-decoding stage, a multi-scale encoder-decoder network processes the stacked features with downsampling by factors of two and four to capture both global structures and fine details. Each scale integrates $N _ { 1 }$ binarized information interaction encoders and $N _ { 2 }$ binarized information interaction decoders, as shown in Fig. 3, and detailed in Section III-C. This stage produces refined features $\mathbf { F } _ { i } ^ { 2 } .$ . In the reconstruction stage, $\mathbf { F } _ { i } ^ { 2 }$ is used to generate the high-quality frame $\hat { \mathbf { I } } _ { i } ^ { \mathrm { H Q } }$ , while partial features of this iteration are forwarded to the next iteration via the sliding window recurrent connection, which will be described in detail. As illustrated in Fig. 2(b), the distribution-aware binarized convolution is a core module throughout all stages, binarizing both feature maps and convolutional kernels to significantly reduce computational cost and memory usage. Further details are provided in Section III-D.

Sliding window recurrent connection. Conventional video restoration methods predominantly employ either a multiframe sliding window framework [39], [21], or a single-frame recurrent architecture [8], [28]. While the former effectively captures short-term dependencies, it fails to exploit longterm temporal relationships; the latter leverages hidden state propagation for long-term modeling but is limited in handling short-term correlations and is prone to cumulative errors.

To overcome these limitations, we propose the sliding window recurrent connection, which integrates the strengths of both paradigms. As depicted in Fig. 4, this design preserves the sliding window structure while incorporating recurrent connections across adjacent windows, thereby enabling efficient short-term feature aggregation and long-term dependency modeling. The input at time step t of the recursive encodingdecoding stage can be expressed as

$$
\mathbf { F } ^ { 1 } ( t ) = \mathscr { C } \left( \mathbf { F } _ { i } ^ { 2 } ( t - 1 ) , \mathbf { F } _ { i + 1 } ^ { 2 } ( t - 1 ) , \mathbf { F } _ { i + 1 } ^ { 1 } ( t ) \right) ,\tag{5}
$$

where $\mathcal { C } ( \cdot )$ denotes channel-wise concatenation, and F denotes corresponding features. Our sliding window recurrent connection offers a principled and computationally efficient framework for enhancing temporal information flow.

Loss function. We use the joint loss function of Charbonnier loss [79] and structural similarity index measure loss between the predicted HQ frames $\hat { \mathbf { I } } ^ { \mathrm { H Q } }$ and the ground-truth HQ frames $\mathbf { I } ^ { \mathrm { H Q } }$ as

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { c } } ( \hat { \bf I } ^ { \mathrm { H Q } } , { \bf I } ^ { \mathrm { H Q } } ) + \lambda \mathcal { L } _ { \mathrm { s } } ( \hat { \bf I } ^ { \mathrm { H Q } } , { \bf I } ^ { \mathrm { H Q } } ) , } \end{array}\tag{6}
$$

where $\mathcal { L } _ { \mathrm { c } }$ represents the Charbonnier loss [79], $\mathcal { L } _ { \mathrm { s } }$ represents the SSIM loss, and λ is a task-dependent hyperparameter. We set $\lambda = 0 . 0 1$ for low-light enhancement, denoising, and super-resolution, and $\lambda = 0 . 1$ for deblurring, where structural similarity is particularly important for recovering blurred RAW details in deblurring.

## C. Binarized information interaction module

The Binarized Information Interaction Module (BIIM) is designed to enlarge temporal and spatial receptive fields in RAW video restoration. As illustrated in Fig. 3, BIIM consists of a temporal and a spatial enhancement module. The temporal module introduces no additional learnable parameters or convolutional operations and only incurs negligible arithmetic overhead through channel splitting and concatenation, while the spatial module employs lightweight strip-shaped depth wise convolutions compared with conventional large-kernel binarized depthwise convolutions. We note that the normalized FLOPs and parameter analysis mainly reflects arithmetic and storage complexity, while wall-clock latency may also depend on memory access and hardware implementation. To balance efficiency and effectiveness, only the temporal module is used in the encoder, whereas both temporal and spatial modules are applied in the decoder.

Temporal information interaction. To facilitate improved information fusion among adjacent frames (current, previous, and subsequent), we introduce the temporal information interaction module in the BIIM decoder, as illustrated in Fig. 3(b). A simplified version of this module is employed in the BIIM encoder, as illustrated in Fig. 3(a). The module takes as input three feature maps $\{ \mathbf { F } _ { i - 1 } , \mathbf { F } _ { i } , \mathbf { F } _ { i + 1 } \} \in \mathbb { R } ^ { C \times H \times W }$ , each of which is initially split into two parts along the channel dimension, formulated as

$$
[ \mathbf { f } _ { i + k } ^ { 1 } , \ \mathbf { f } _ { i + k } ^ { 2 } ] = S ( \mathbf { F } _ { i + k } ) , \quad k \in \{ - 1 , 0 , 1 \} ,\tag{7}
$$

where $\mathbf { f } \in \mathbb { R } ^ { C / 2 \times H \times W }$ are the two partitioned segments of the feature map for each frame.

To ensure that each frame’s feature map, after temporal enhancement, incorporates information from these three frames, the output feature map can be represented as

$$
\mathbf { F } _ { i + k } ^ { \mathrm { T } } = { \mathcal { C } } { \big ( } \mathbf { f } _ { i + k } ^ { 1 } , \ \mathbf { f } _ { i + k + 1 } ^ { 2 } , \ \mathbf { f } _ { i + k - 1 } ^ { \alpha } { \big ) } , \quad k \in \{ - 1 , 0 , 1 \} ,\tag{8}
$$

where $\mathbf { F } ^ { \mathrm { T } } \in \mathbb { R } ^ { { \frac { 3 } { 2 } } C \times H \times W }$ denotes the output of the temporalenhanced module, and $\alpha \in \{ 1 , 2 \}$ . This operation effectively expands the receptive field across frames without introducing additional learnable parameters or convolutional operations, making it suitable for binarization and quantization.

Spatial information interaction. The spatial information interaction module is designed to enlarge the receptive field in a lightweight and binarization-friendly manner for RAW video restoration. Large-kernel depthwise convolutions expand the receptive field and can improve performance [80], [81]. However, excessively large square kernels significantly increase computation and parameters. Prior studies also suggest that processing only subsets of channels in depthwise layers can achieve competitive performance [82]. Motivated by this, we replace large square kernels with horizontal and vertical stripshaped kernels $( e . g . , 1 { \times } 1 1 \mathrm { a n d } 1 1 { \times } 1 )$ , each applied to a subset of channels. Meanwhile, another subset is directly propagated via identity mapping, while the remaining subset is processed with small square kernels $( e . g . , \ 3 \times \ 3 )$ through binarized depthwise convolutions as shown in Fig. 5(a). Formally,

$$
\begin{array} { r l } & { [ \mathbf { f } _ { \mathrm { i } } , \mathbf { f } _ { \mathrm { s } } , \mathbf { f } _ { \mathrm { h } } , \mathbf { f } _ { \mathrm { v } } ] = \boldsymbol { \mathcal { S } } ( \mathbf { F } ^ { \mathrm { T } } ) , } \\ & { \tilde { \mathbf { F } } ^ { \mathrm { S } } = \boldsymbol { \mathcal { C } } ( \mathbf { f } _ { \mathrm { i } } , \mathcal { D } _ { 3 \times 3 } ( \mathbf { f } _ { \mathrm { s } } ) , \mathcal { D } _ { 1 \times 1 1 } ( \mathbf { f } _ { \mathrm { h } } ) , \mathcal { D } _ { 1 1 \times 1 } ( \mathbf { f } _ { \mathrm { v } } ) ) , } \end{array}\tag{9}
$$

where $\mathbf { F } ^ { \mathrm { T } }$ is the output of our temporal information interaction module, and $\mathbf { f } _ { \mathrm { i } } , \mathbf { f } _ { \mathrm { s } } , \mathbf { f } _ { \mathrm { h } } , \mathbf { f } _ { \mathrm { v } }$ denote subsets processed by identity mapping, small square kernels, horizontal strip kernels, and vertical strip kernels, respectively. $\mathcal { D } ( \cdot )$ denotes the distribution-aware binarized depthwise convolution, which will be introduced in Section III-D. Finally, the spatial information interaction module applies a $5 \times 5$ distribution-aware binarized depthwise convolution on the intermediate feature map $\tilde { \mathbf { F } } ^ { \mathrm { S } }$ to obtain the final output $\mathbf { F } ^ { \mathrm { S } }$

Following the commonly adopted efficiency estimation protocol for binarized networks, we weight binary operations as $1 / 6 4$ of their floating-point FLOPs and binary parameters as $1 / 3 2$ of their full-precision counterparts, while keeping bias terms in full precision. Under this setting, the normalized computational cost of our spatial information interaction module (per spatial location) is

$$
\frac { \mathrm { F L O P s } } { H W } = 2 \left[ \frac { 1 } { 6 4 } \left( \frac { C _ { \mathrm { i n } } K _ { \mathrm { p } } } { 8 } \times 2 + \frac { C _ { \mathrm { i n } } K _ { \mathrm { q } } } { 8 } \right) + \frac { C _ { \mathrm { i n } } } { 8 } \times 3 \right] ,\tag{10}
$$

and the corresponding parameter complexity is

$$
\mathrm { P a r a m s } = { \frac { 1 } { 3 2 } } \left( { \frac { C _ { \mathrm { i n } } K _ { \mathrm { p } } } { 8 } } \times 2 + { \frac { C _ { \mathrm { i n } } K _ { \mathrm { q } } } { 8 } } \right) + { \frac { C _ { \mathrm { i n } } } { 8 } } \times 3 ,\tag{11}
$$

where $K _ { \mathrm { p } }$ denotes the kernel length of the strip-shaped convolutions, and $K _ { \mathrm { q } }$ denotes the kernel size of the small square-shaped convolution. The last term in both equations corresponds to the full-precision bias of the three convolutional branches. Notably, the FLOPs and parameters of SII increase linearly with the kernel size, whereas those of fullprecision and conventional binarized depthwise convolutions grow quadratically. Detailed comparisons are provided in Fig. 5(b) and Fig. 5(c).

## D. Distribution-Aware Binarized Convolution

Binarized convolutions often suffer performance degradation due to the loss of full-precision details in weights and activations. A common remedy is to introduce real-valued scaling factors [27], [83]. In full-precision networks, most weights approximately follow a symmetric bell-shaped distribution. We compute the weight scaling factors using the $L _ { 1 }$ -norm of each kernel, expressed as

$$
\mathbf { S } ^ { \mathrm { w } } = \frac { 1 } { C _ { \mathrm { i n } } K _ { 1 } K _ { 2 } } \Big [ \| \mathbf { W } _ { 1 } ^ { \mathrm { f } } \| _ { 1 } , \| \mathbf { W } _ { 2 } ^ { \mathrm { f } } \| _ { 1 } , \cdots , \| \mathbf { W } _ { C _ { \mathrm { o u t } } } ^ { \mathrm { f } } \| _ { 1 } \Big ] ,\tag{12}
$$

![](images/2a74cc94d11ecd552455081e8bab021754b4f892578011cc37d2cdf84788b061.jpg)

![](images/ff9eab8ad07e31519fb1f3c7b588b84d9dc2cf0980bd2b0308026f183cc347c8.jpg)

![](images/bc9833bb220970fa9498421e045f0b8df6e4c545267c907d830bde65ae328673.jpg)  
(b)  
(c)  
Fig. 5. Illustration of spatial information interaction module. (a) Illustration of the spatial information interaction module, where feature channels are split into identity, $3 \times 3 , 1 \times 1 1 .$ , and $1 1 \times 1$ binarized depthwise convolution branches. (b) FLOPs per spatial location versus kernel size. (c) Parameter complexity versus kernel size.

where $\mathbf { W } _ { i } ^ { \mathrm { f } } \in \mathbb { R } ^ { C _ { \mathrm { i n } } \times K _ { 1 } \times K _ { 2 } }$ is the full-precision kernel of the i-th output channel, $\| \cdot \| _ { 1 }$ denotes the $L _ { 1 }$ -norm computed as the sum of absolute values over all elements in the kernel, and $\mathbf { S } ^ { \mathrm { w } } \in \mathbb { R } ^ { C _ { \mathrm { o u t } } }$ denotes the channel-wise weight scaling factors used to mitigate binarization error. For distribution-aware binarized depthwise convolutions, this formulation degenerates to the case $C _ { \mathrm { i n } } = 1$

Unlike weights, activations depend on input data and exhibit highly diverse and often asymmetric distributions. Simply reusing the weight scaling strategy is insufficient for robust binarization. To address this, we design distribution-aware activation scaling factors that explicitly incorporate statistical characteristics of the input. As illustrated in Fig. 2(b), a perchannel learnable bias $\beta \in \mathbb { R } ^ { C }$ is first added to the realvalued activations to compensate for asymmetric distributions, yielding adjusted activations $\tilde { \mathbf { F } } ^ { \mathrm { f } } = \mathbf { F } ^ { \mathrm { f } } + \boldsymbol { \beta } .$

We then compute channel-wise descriptive statistics of $\tilde { \mathbf { F } } ^ { \mathrm { f } }$ including the mean, mean of absolute values, and standard deviation (all computed per channel), i.e.,

$$
\begin{array} { r l } & { \tilde { \mathbf { S } } ^ { \mathrm { a } } = \mathcal { C } \Big ( \mathbf { M e a n } ( \tilde { \mathbf { F } } ^ { \mathrm { f } } ) , ~ \mathbf { M e a n } ( | \tilde { \mathbf { F } } ^ { \mathrm { f } } | ) , \mathbf { S t d } ( \tilde { \mathbf { F } } ^ { \mathrm { f } } ) \Big ) , } \\ & { \mathbf { S } ^ { \mathrm { a } } = \mathrm { S i g m o i d } \Big ( \mathbf { C o n v } 1 \mathbf { D } ( \tilde { \mathbf { S } } ^ { \mathrm { a } } ) \Big ) , } \end{array}\tag{13}
$$

where $\tilde { \mathbf { S } } ^ { \mathrm { a } } \in \mathbb { R } ^ { 3 C }$ is the intermediate statistical descriptor and $\mathbf { S } ^ { \mathrm { a } } \in \mathbb { R } ^ { C }$ is the final activation scaling factor. Together, the above two steps define a distribution-aware scaling function $\Phi ( \cdot )$ that maps the adjusted activations $\tilde { \mathbf { F } } ^ { \mathrm { f } }$ to channel-wise scaling factors.

Intuitively, DAB-Conv adapts the quantization scale to the activation distribution of each channel. The mean reflects the center shift of the distribution, the mean of absolute values reflects the average response magnitude, and the standard deviation reflects the dynamic range and dispersion. We do not claim that these descriptors are theoretically optimal in an information-theoretic sense; instead, they are lightweight and stable first- and second-order statistics that are closely related to binarization error. The lightweight Conv1D adaptively fuses these complementary cues to predict input-dependent channelwise scaling factors, so that the quantized activations better match the full-precision ones.

After obtaining both weight and activation scaling factors, the distribution-aware binarized convolution is computed as

$$
\begin{array} { r l } & { \mathbf { F } ^ { \mathrm { b } } = \mathrm { S i g n } ( \tilde { \mathbf { F } } ^ { \mathrm { f } } ) , } \\ & { \mathbf { Y } = \mathrm { R P R e L U } \Bigl ( ( \mathbf { F } ^ { \mathrm { b } } \otimes \mathbf { W } ^ { \mathrm { b } } ) \odot \mathbf { S } ^ { \mathrm { w } } \odot \mathbf { S } ^ { \mathrm { a } } \Bigr ) + \mathbf { F } ^ { \mathrm { f } } , } \end{array}\tag{14}
$$

where Y is the final output, and RPReLU(·) denotes a learnable real-valued activation function following [59].

## E. Distribution-Aware Multi-Bit Quantization

Although binarization offers substantial efficiency gains, real-world deployment often requires multiple quantization bitwidths to match hardware constraints and balance efficiency and accuracy. Multi-bit quantization better preserves representational fidelity with b-bit discrete values, but activation distributions still vary noticeably across layers and channels, and fixed ranges may cause biased scaling and mismatched dynamic ranges.

To address this, we introduce our distribution-aware mechanism in the multi-bit setting, enabling adaptive scaling that matches the local activation statistics. A lightweight perchannel bias $\beta$ is applied to align activation distributions, producing the adjusted activations $\mathbf { \tilde { F } } ^ { \mathrm { f } }$ . The quantization range is then dynamically adapted through

$$
\mathbf { S } ^ { \mathrm { a } } = \Phi ( \tilde { \mathbf { F } } ^ { \mathrm { f } } ) ,\tag{15}
$$

where $\Phi ( \cdot )$ summarizes activation statistics (channel-wise mean, absolute mean, and standard deviation over spatial locations) and maps them into channel-wise scaling factors.

Based on this adaptive scaling, multi-bit activation quantization is computed as $\mathbf { \bar { F } } ^ { q } = \mathcal { Q } ( \tilde { \mathbf { F } } ^ { \tilde { \mathrm { f } } } / \mathbf { S } ^ { \mathrm { a } } ) \cdot \mathbf { S } ^ { \mathrm { a } }$ , where $\mathcal { Q } ( \cdot )$ denotes the previously defined uniform quantizer and is not repeated here. Finally, the distribution-aware multi-bit convolution with residual connection is

$$
\mathbf { Y } = \gamma \mathbf { F } ^ { \mathrm { f } } + \operatorname { R e L U } \bigl ( ( \mathbf { F } ^ { q } \otimes \mathbf { W } ^ { q } ) \odot \mathbf { S } ^ { \mathrm { a } } \bigr ) ,\tag{16}
$$

where $\gamma$ is a learnable scalar, inspired by the residual balancing strategy in QuantSR [73], to balance the shortcut and quantized branches. This scalar is introduced only for multi-bit quantization, where the quantized branch retains richer discrete representations and its contribution can be adaptively balanced with the shortcut branch. For 1-bit binarization, Eq. 14 already employs RPReLU, residual connection, and distribution-aware scaling to compensate for severe representation loss, making an additional balancing scalar less beneficial.

In contrast to standard multi-bit quantization, which applies fixed scaling across channels, our distribution-aware mechanism produces quantized features that retain robustness under diverse activation statistics and alleviate range mismatch across layers and inputs.

Gradient approximation. The sign and round functions used in binarization and quantization are non-differentiable, making direct back-propagation infeasible. Following previous works [20], [61], we adopt straight-through estimators (STEs) during training to approximate the gradients of these discrete operations. Specifically, for binarized weights, the gradient of the sign function is approximated by the derivative of a clip function, while for binarized activations, a smoother piecewise quadratic surrogate is used in the backward pass. For multibit quantization, the rounding operation is also optimized with STE, where the forward pass performs discretization and the backward pass approximates its gradient through the quantized branch. These approximations allow stable gradient propagation during training while keeping the inference path binarized or quantized.

## IV. EXPERIMENTS

In this section, we evaluate our BinRVR on four RAW video restoration tasks, i.e., low-light RAW video enhancement, RAW video denoising, RAW video deblurring, and RAW video super-resolution. We further evaluate the method under the multi-bit quantization setting on low-light enhancement, followed by discussions and downstream task evaluations.

## A. Experiment Settings

We first describe the implementation details, followed by the binarized comparison methods and evaluation metrics, and finally present the model efficiency evaluation.

Implementation details. The models are trained on 10-frame sequences from all datasets, with each batch consisting of a single $2 5 6 \times 2 5 6$ RAW patch. The total batch size is set to 2. Data augmentation techniques such as random cropping, random horizontal flipping, random vertical flipping, and random rotation are applied to improve generalization. Following previous works [20], [86], training is performed for 100K iterations using the Adam optimizer [87] with $\beta = 0 . 9 $ , and a weight decay of $1 \times 1 0 ^ { - 4 }$ . A cosine annealing learning rate schedule [88] is adopted, with the learning rate set to $2 \times 1 0 ^ { - 4 }$ for all tasks. All experiments are conducted using PyTorch on two NVIDIA RTX 3090 GPUs.

Across the four RAW video restoration tasks, we use the same BinRVR backbone, bit-width setting, temporal window size, stride, optimizer, training iterations, learning rate schedule, patch size, batch size, and data augmentation strategy. The default temporal window size is 3 and the recurrent stride is 1. The only task-specific architectural adjustment is made for RAW video super-resolution, where a PixelShuffle upsampling layer is used in the reconstruction head to produce highresolution RAW outputs.

Binarized comparison methods. For the four RAW video restoration tasks, we compare our BinRVR with state-ofthe-art binarized methods, including BNN [19], IRNet [62], Bireal [61], ReActNet [59], BTM [69], BBCU [20], BiSR-Net [84], FRBC [85], BHViT [72], and BRVE [86], where BRVE is designed for low-light RAW video enhancement. As no existing BNN-based methods are specifically designed for

TABLE I  
QUANTITATIVE COMPARISON ON SMOID [37] AND LLRVD [10] FOR LOW-LIGHT RAW VIDEO ENHANCEMENT. FOR BINARIZED METHODS, THE BEST AND SECOND-BEST RESULTS ARE SHOWN IN BOLD AND UNDERLINED, RESPECTIVELY. LIGHTWEIGHT FULL-PRECISION METHODS ARE ADDITIONALLY INCLUDED FOR REFERENCE, WITH THE BEST RESULT SHOWN IN BOLD.
<table><tr><td rowspan="3">Method</td><td rowspan="3">Params (M)↓</td><td rowspan="3">FLOPs (G)↓</td><td colspan="8">SMOID [37]</td><td rowspan="2" colspan="2">LLRVD [10]</td></tr><tr><td colspan="2">Gain0</td><td colspan="3">Gain15</td><td colspan="3">Gain30</td></tr><tr><td colspan="3">PSNR↑ SSIM↑ ST-RRED↓</td><td colspan="2">PSNR↑ SSIM↑ ST-RRED↓</td><td colspan="3">PSNR↑ SSIM↑ ST-RRED↓ PSNR↑ SSIM↑ ST-RRED↓</td></tr><tr><td>Linear Scaled</td><td></td><td>31.26</td><td>0.7665</td><td>0.4965</td><td>32.35</td><td>0.8108</td><td>0.2541</td><td>32.27 0.8358</td><td>0.4156</td><td>30.04</td><td>0.6941</td><td>0.0863</td></tr><tr><td colspan="9">Full-precision methods</td><td></td><td></td><td></td></tr><tr><td>SMOID [37]</td><td>12.21</td><td>15.11 39.74</td><td>0.9733</td><td>0.0446</td><td>39.43</td><td>0.9751</td><td>0.0461</td><td>40.06 0.9769</td><td>0.0431</td><td>37.07</td><td>0.9580</td><td>0.0391</td></tr><tr><td>RViDeNet [38]</td><td>8.57</td><td>65.77</td><td>41.09 0.9778</td><td>0.0510</td><td>40.94</td><td>0.9803</td><td>0.0494</td><td>41.60 0.9818</td><td>0.0500</td><td>37.39</td><td>0.9620</td><td>0.0400</td></tr><tr><td>FastDVD [40]</td><td>2.48</td><td>10.50</td><td>40.37 0.9724</td><td>0.0765</td><td>40.66</td><td>0.9768</td><td>0.0664</td><td>40.59 0.9763</td><td>0.1812</td><td>36.39</td><td>0.9450</td><td>0.0724</td></tr><tr><td>EMVD-L [42]</td><td>9.60</td><td>17.52</td><td>41.10 0.9785</td><td>0.0657</td><td>40.34</td><td>0.9804</td><td>0.0691</td><td>41.13 0.9819</td><td>0.0796</td><td>37.00</td><td>0.9576</td><td>0.0473</td></tr><tr><td>EMVD-S [42]</td><td>0.81</td><td>1.66</td><td>39.71 0.9707</td><td>0.0732</td><td>39.23</td><td>0.9735</td><td>0.0804</td><td>39.88 0.9756</td><td>0.0875</td><td>36.70</td><td>0.9527</td><td>0.0527</td></tr><tr><td>LLRVD [10]</td><td>6.29</td><td>46.14</td><td>41.51 0.9799</td><td>0.0388</td><td>41.75</td><td>0.9823</td><td>0.0330</td><td>42.13 0.9840</td><td>0.0350</td><td>37.74</td><td>0.9650</td><td>0.0347</td></tr><tr><td>FloRNN [13]</td><td>10.49</td><td>24.57</td><td>41.39 0.9801</td><td>0.0468</td><td>40.74</td><td>0.9823</td><td>0.0560</td><td>41.55 0.9842</td><td>0.0558</td><td>37.47</td><td>0.9634</td><td>0.0377</td></tr><tr><td>ShiftNet [8]</td><td>13.38</td><td>32.87</td><td>42.11 0.9836</td><td>0.0328</td><td>42.28</td><td>0.9848</td><td>0.0273</td><td>42.70 0.9863</td><td>0.0280</td><td>37.87</td><td>0.9661</td><td>0.0346</td></tr><tr><td colspan="9">Binarized methods</td><td></td><td></td><td></td></tr><tr><td>BNN [19]</td><td>0.26</td><td>1.38</td><td>38.34 0.9558</td><td>0.1078</td><td>38.30</td><td>0.9636</td><td>0.1191</td><td>38.75</td><td>0.9628 0.1045</td><td>35.91</td><td>0.9362</td><td>0.0669</td></tr><tr><td>Bireal [61]</td><td>0.26</td><td>1.38</td><td>38.38 0.9530</td><td>0.1138</td><td>38.50</td><td>0.9672</td><td>0.1086</td><td>38.74 0.9628</td><td>0.1013</td><td>36.22</td><td>0.9449</td><td>0.0591</td></tr><tr><td>IRNet [62]</td><td>0.26</td><td>1.38</td><td>38.21 0.9517</td><td>0.1270</td><td>38 53</td><td>0.9647</td><td>0.1100</td><td>38.65 0.9634</td><td>0.1139</td><td>35.90</td><td>0.9348</td><td>0.0670</td></tr><tr><td>ReActNet [59]</td><td>0.28</td><td>1.56</td><td>38.80 0.9567</td><td>0.1049</td><td>38.71</td><td>0.9702</td><td>0.1140</td><td>39.23</td><td>0.9667 0.0961</td><td>36.23</td><td>0.9443</td><td>0.0608</td></tr><tr><td>BTM [69]</td><td>0.25</td><td>1.33</td><td>39.58 0.9704</td><td>0.0824</td><td>39.43</td><td>0.9736</td><td>0.0879</td><td>39.77</td><td>0.9755</td><td>0.0986</td><td>36.64 0.9528</td><td>0.0532</td></tr><tr><td>BBCU [20]</td><td>0.27</td><td>1.47</td><td>39.63 0.9710</td><td>0.0793</td><td>39.97</td><td>0.9743</td><td>0.0635</td><td>40.16</td><td>0.9765</td><td>0.0730 36.72</td><td>0.9541</td><td>0.0523</td></tr><tr><td>BiSRNet [84]</td><td>0.28 0.26</td><td>1.44</td><td>39.71 0.9708</td><td>0.0766</td><td>39.98</td><td>0.9741</td><td>0.0610</td><td>40.17</td><td>0.9764 0.0777</td><td>36.76</td><td>0.9548</td><td>0.0509</td></tr><tr><td>FRBC [85]</td><td>0.27</td><td>1.38</td><td>31.35 0.7356 39.87 0.9717</td><td>0.6570 0.0752</td><td>32.88 40.05</td><td>0.8065</td><td>0.3950</td><td>32.97</td><td>0.7944 0.3994</td><td>36.02</td><td>0.9405</td><td>0.0626</td></tr><tr><td>BHViT [72]</td><td>0.30</td><td>1.47 1.49</td><td>40.05 0.9742</td><td>0.0639</td><td>40.25</td><td>0.9751 0.9765</td><td>0.0621</td><td>40.53</td><td>0.9774 0.0703</td><td>36.70</td><td>0.9540</td><td>0.0527</td></tr><tr><td>BRVE [86]</td><td>0.22</td><td>1.23</td><td>40.25 0.9758</td><td>0.0625</td><td>40.45</td><td>0.9781</td><td>0.0557 0.0523</td><td>40.64</td><td>0.9786 0.0570</td><td>37.07</td><td>0.9581</td><td>0.0455</td></tr><tr><td>BinRVR (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>40.83</td><td>0.9800</td><td>0.0554 37.21</td><td>0.9599</td><td>0.0436</td></tr></table>

RAW video restoration, we replace our Binarized Information Interaction Module (BIIM) and Distribution-Aware Binarized Convolution (DAB-Conv) in BinRVR with the corresponding binarized convolutional blocks of these methods, while keeping the remaining architecture unchanged to enable an isolated comparison of binarization strategies.

Evaluation metrics. To evaluate the video restoration quality of each method, we calculate the average peak signal-to-noise ratio (PSNR), structural similarity index measure (SSIM), and spatio-temporal reduced reference entropic differences (ST-RRED) [89] across all output RAW frames. The ST-RRED metric assesses video quality by capturing both spatial and temporal quality degradations. Higher values indicate better performance for PSNR and SSIM, while lower values indicate better performance for ST-RRED.

Model efficiency evaluation. Following [61], [84], we estimate the model efficiency of binarized networks by weighting the binary operations with 1/64 of the corresponding floating point operations (FLOPs) and the parameters (Params) with 1/32 of their full-precision counterparts. For multi-bit quantization, the model efficiency is estimated by scaling the full-precision operations and parameters according to the bit width. Specifically, 2-, 3-, and 4-bit operations are weighted by 1/16, 3/32, and 1/8 of the full-precision FLOPs and parameters, respectively. The total FLOPs and parameters are obtained by summing the binarized (or quantized) and fullprecision components. For all methods, per-frame FLOPs are computed by processing a 100-frame RAW video with a spatial resolution of 256 × 256.

## B. Low-Light RAW Video Enhancement Results

Low-light RAW video enhancement is a fundamental task in video restoration, aiming to recover clear, consistent content from sequences captured under insufficient illumination.

Datasets. We evaluate our BinRVR on two challenging lowlight RAW video datasets, i.e., SMOID [37] and LLRVD [10]. SMOID contains 309 video pairs across 103 indoor and outdoor scenes, each recorded at three ADC gain levels (0, 15, and 30) at 480 × 640 resolution. The dataset is divided into 70 scenes for training, 16 for validation, and 17 for testing. LLRVD consists of 210 video pairs from 70 high-resolution scenes at 1400 × 2600 resolution. For each scene, three lowlight sequences are generated by fixing the ISO and varying the exposure time to simulate different illumination conditions. LLRVD is split into 60 scenes for training, 4 for validation, and 6 for testing.

Quantitative results. We conduct a quantitative comparison of the two datasets for low-light RAW video enhancement, and the results are shown in Table I. Our method consistently outperforms all binarized competitors across all settings and metrics. Compared with BRVE [86], which is specifically designed for this task, our model achieves an average improvement of 0.18 dB in PSNR. Compared with the recent binarized methods, BHViT [72], our method yields an average PSNR gain of 0.40 dB, clearly establishing its advantage in restoration quality. Moreover, our model attains these results with the lowest FLOPs and parameter count among all binarized methods, demonstrating its high efficiency.

Compared with full-precision low-light video enhancement methods, including SMOID [37], RViDeNet [38], FastDVD [40], EMVD-L / EMVD-S [42], LLRVD [10], FloRNN [13], and ShiftNet [8], our method achieves a highly favorable accuracy-efficiency trade-off. In particular, when compared with the best full-precision model, ShiftNet [8], our method reduces FLOPs and parameters by over 96% and 98%, respectively, while incurring only about a 4% average PSNR drop. Moreover, compared with EMVD-S [42], a full-precision model with similar computational cost, our method attains up to 0.9 dB higher PSNR together with consistent improvements in perceptual quality metrics.

![](images/6d4be2eb0e879b1cea18dfa90c7d2b2ddda54df3b8373c3f262eadb0e5d6e12a.jpg)  
Fig. 6. Visual comparisons of binarized methods on the SMOID [37] and LLRVD [10] datasets for low-light RAW video enhancement (left: SMOID; right: LLRVD). For each example, the left half shows zoomed-in RGB results obtained by applying an ISP pipeline to the RAW outputs, while the right half presents the corresponding error maps with respect to the ground truth.

Qualitative results. To further illustrate the visual quality achieved by different methods, Fig. 6 presents qualitative comparisons on the SMOID [37] and LLRVD [10] datasets. In the SMOID example, our method more effectively restores reflective objects within window regions, while most competing binarized methods fail to recover these reflections and suffer from severe detail degradation. This demonstrates that our method exhibits stronger enhancement in ultra-dark areas, producing clearer structures with fewer visual artifacts. On the LLRVD example, our method achieves more accurate color restoration along character boundaries, resulting in sharper edges and improved consistency with the ground truth. These observations provide clear visual evidence of the advantages of the proposed method under challenging low-light conditions.

## C. RAW Video Denoising Results

RAW video denoising suppresses sensor noise while preserving details and temporal coherence. Compared with RGB data, RAW inputs retain linear sensor responses and richer scene information for restoration.

Datasets. Experiments are conducted on the CRVD [53] dataset, which provides a challenging benchmark for RAW video denoising. CRVD contains 55 groups of noisy-clean video pairs, each consisting of 7 frames at a resolution of

1920 × 1080 and 20 fps, captured using an IMX385 sensor. Noisy frames are recorded under high ISO settings ranging from 1600 to 25600, and the corresponding clean frames are obtained by averaging multiple noisy observations. The dataset includes 6 indoor scenes for training and 5 indoor scenes for validation and testing, providing diverse conditions for evaluating both spatial fidelity and temporal consistency.

Quantitative results. We evaluate RAW video denoising performance on the CRVD [53], with results summarized in Table II. Our method outperforms all binarized methods across all metrics. Compared with the best-performing binarized competitor, BBCU [20], it achieves a 0.32 dB PSNR gain, higher SSIM, and a lower ST-RRED. We also compare with lightweight full-precision methods, including FastDVD [40], EMVD-S [42], RViDeNet [38], LLRVD [10], and FloRNN [13], whose results are taken from the original papers and therefore do not report ST-RRED. Compared with EMVD-S, which has a similar computational cost, our method attains 0.90 dB higher PSNR. This indicates that our method achieves high efficiency while maintaining better performance.

Qualitative results. To qualitatively assess the denoising performance, Fig. 7 presents visual comparisons of RAW video denoising results. Our method produces noticeably cleaner results with finer and more clearly preserved leaf textures after denoising. The error maps exhibit a substantially larger proportion of blue regions, indicating lower reconstruction errors compared with other binarized methods. These visual differences highlight the clear advantage of our method in suppressing noise while preserving fine structural details.

## D. RAW Video Deblurring Results

RAW video deblurring removes motion blur caused by camera or object motion to produce sharper and more stable videos. RAW inputs avoid ISP-induced artifacts and preserve a more faithful scene representation.

TABLE II  
QUANTITATIVE COMPARISON ON CRVD [53] FOR RAW VIDEO DENOISING. FOR BINARIZED METHODS, THE BEST AND SECOND-BEST RESULTS ARE SHOWN IN BOLD AND UNDERLINED, RESPECTIVELY. LIGHTWEIGHT FULL-PRECISION METHODS ARE ADDITIONALLY INCLUDED FOR REFERENCE, WITH THE BEST RESULT SHOWN IN BOLD. THE RESULTS OF FULL-PRECISION METHODS ARE REPORTED FROM THE ORIGINAL PAPERS.
<table><tr><td>Method</td><td>Noisy</td><td>FastDVD EMVD-S RViDeNet LLRVD FloRNN [40]</td><td>[42]</td><td>[38]</td><td>[10]</td><td>[13]</td><td>BNN [19]</td><td>Bireal [61]</td><td>IRNet ReActNet [62]</td><td>[59]</td><td>BTM [69]</td><td>[20]</td><td>BBCU BiSRNet FRBC [84]</td><td>[85]</td><td>BHViT BinRVR [72]</td><td>(Ours)</td></tr><tr><td>Params(M)↓</td><td></td><td>2.48</td><td>0.81</td><td>8.57</td><td>6.29</td><td>10.49</td><td>0.26</td><td>0.26</td><td>0.26</td><td>0.28</td><td>0.25</td><td>0.27</td><td>0.28</td><td>0.26</td><td>0.27</td><td>0.22</td></tr><tr><td>FLOPs(G)↓</td><td></td><td>10.50</td><td>1.66</td><td>65.77</td><td>46.14</td><td>24.57</td><td>1.38</td><td>1.38</td><td>1.38</td><td>1.56</td><td>1.33</td><td>1.47</td><td>1.44</td><td>1.38</td><td>1.47</td><td>1.23</td></tr><tr><td>PSNR↑</td><td>32.02</td><td>39.84</td><td>42.63</td><td>43.97</td><td>44.23</td><td>45.15</td><td>42.05</td><td>42.06</td><td>41.97</td><td>41.92</td><td>43.11</td><td>43.21</td><td>43.16</td><td>41.48</td><td>43.17</td><td>43.53</td></tr><tr><td>SSIM↑</td><td>0.7333</td><td>0.9703</td><td>0.9851</td><td>0.9874</td><td>0.9879</td><td>0.9907</td><td></td><td>0.9804 0.9809 0.9800</td><td></td><td>0.9805</td><td></td><td>0.9847 0.9850</td><td>0.9848</td><td>0.9779 0.9849</td><td></td><td>0.9868</td></tr><tr><td>ST-RRED↓</td><td>0.0171</td><td></td><td></td><td></td><td></td><td></td><td>0.0118 0.0109 0.0119</td><td></td><td></td><td>0.0116</td><td></td><td>0.0103 0.0100</td><td>0.0100</td><td></td><td>0.0119 0.0100</td><td>0.0080</td></tr></table>

![](images/6ffc28b42566426f1ab8eb2cbaad0560b5286ff0348146c2b0c85f66ea8f70af.jpg)  
Fig. 7. Visual comparisons of binarized methods on the CRVD [53] dataset for RAW video denoising. For each example, the left half shows zoomed-in RGB results obtained by applying an ISP pipeline to the RAW outputs, while the right half presents the corresponding error maps with respect to the ground truth.

TABLE III  
QUANTITATIVE COMPARISON ON DEBLUR-RAW [90] FOR RAW VIDEO DEBLURRING. FOR BINARIZED METHODS, THE BEST AND SECOND-BEST RESULTS ARE SHOWN IN BOLD AND UNDERLINED, RESPECTIVELY. LIGHTWEIGHT FULL-PRECISION METHODS ARE ADDITIONALLY INCLUDED FOR REFERENCE, WITH THE BEST RESULT SHOWN IN BOLD.
<table><tr><td>Method</td><td>Blurry</td><td>MIMO [91]</td><td>ESTRNN [92]</td><td>LIEDNet [93]</td><td>BNN [19]</td><td>Bireal [61]</td><td>IRNet [62]</td><td>ReActNet [59]</td><td>BTM [69]</td><td>BBCU [20]</td><td>BiSRNet [84]</td><td>FRBC [85]</td><td>BHViT [72]</td><td>BinRVR (Ours)</td></tr><tr><td>Params(M) FLOPs(G)</td><td></td><td>6.81</td><td>2.53</td><td>4.76</td><td>0.26</td><td>0.26</td><td>0.26</td><td>0.28</td><td>0.25</td><td>0.27</td><td>0.28</td><td>0.26</td><td>0.27</td><td>0.22</td></tr><tr><td></td><td></td><td>16.00</td><td>12.77</td><td>5.09</td><td>1.38</td><td>1.38</td><td>1.38</td><td>1.56</td><td>1.33</td><td>1.47</td><td>1.44</td><td>1.38</td><td>1.47</td><td>1.23</td></tr><tr><td>PSNR↑</td><td>36.52</td><td>38.15</td><td>38.12</td><td>38.61</td><td>36.88</td><td>36.81</td><td>36.95</td><td>37.33</td><td>37.35</td><td>37.50</td><td>37.71</td><td>36.88</td><td>37.33</td><td>38.89</td></tr><tr><td>SSIM↑</td><td>0.9493</td><td>0.9663</td><td>0.9673</td><td>0.9708</td><td>0.9531</td><td>0.9520</td><td>0.9546</td><td>0.9583</td><td>0.9589</td><td>0.9607</td><td>0.9626</td><td>0.9548</td><td>0.9587</td><td>0.9724</td></tr><tr><td>ST-RRED↓</td><td>0.0180</td><td>0.0151</td><td>0.0157</td><td>0.0098</td><td>0.0180</td><td>0.0158</td><td>0.0179</td><td>0.0169</td><td>0.0179</td><td>0.0175</td><td>0.0167</td><td>0.0168</td><td>0.0169</td><td>0.0106</td></tr></table>

Datasets. Experiments are conducted on the Deblur-RAW [90] dataset, collected across diverse indoor and outdoor scenes using a Canon EOS 6D camera. It consists of 103 video sequences (around 100 frames per sequence), with 88 sequences for training and 15 for testing. Blurry inputs are produced by averaging several successive RAW frames, while the center frame of each sequence is used as the sharp reference. This construction preserves rich structural information and provides both RAW inputs and high-quality ground-truth targets, serving as a reliable benchmark for RAW video deblurring.

Quantitative results. We conduct RAW video deblurring experiments, and the results are summarized in Table III. Our method delivers the best performance among all binarized methods, outperforming BBCU [20] by 1.18 dB in PSNR. Consistent improvements are also observed in SSIM and ST-RRED. Compared to tasks such as low-light RAW video enhancement and RAW video denoising, the performance gap on this task is more pronounced, further demonstrating the advantages of our design under challenging motion blur.

Moreover, we additionally compare against lightweight fullprecision methods, including MIMO [91], ESTRNN [92], and LIEDNet [93]. Note that these methods are originally designed for RGB-domain deblurring and are adapted to the RAW-to-RAW setting due to the lack of lightweight fullprecision deblurring methods specifically designed for RAW video restoration. Their lower performance does not imply an inherent advantage of binarization over full-precision models, but mainly reflects the domain gap between RGB deblurring and RAW video restoration.

In contrast, BinRVR is specifically designed for RAW video restoration, where BIIM models spatio-temporal information in the RAW domain and DAB-Conv mitigates quantization errors caused by activation distribution variations. Following Deblur-RAW [90], we also adopt SSIM loss together with the reconstruction loss for deblurring, which encourages structural similarity and benefits blurred-detail recovery. Therefore, the superior result over these lightweight full-precision methods mainly comes from the RAW-oriented architecture and taskspecific training strategy, while still maintaining substantially lower model size and FLOPs.

Qualitative results. Visual comparisons in Fig. 8 demonstrate that our method recovers clearer alphanumeric details on the license plate, with results visually closer to the ground truth than other binarized methods. In comparison, binarized methods such as BiSRNet [84] and BHViT [72] introduce noticeable artifacts and exhibit blurred character edges.

![](images/6987f48fcde2d8d2a1956ae6cb87ca4e181035841804d48b6515aaf9a3394179.jpg)  
Fig. 8. Visual comparisons of binarized methods on the Deblur-RAW [90] dataset for RAW video deblurring.

## E. RAW Video Super-Resolution Results

RAW video super-resolution reconstructs high-resolution sequences from low-resolution RAW inputs, requiring accurate spatial detail recovery and temporal coherence across frames. Datasets. Experiments are conducted on the Real-RawVSR [9] dataset, the first real-world RAW video super-resolution benchmark. It contains 450 LR-HR video pairs with 2×, 3×, and 4× magnification. Each sequence includes about 150 RAW frames captured using a dual-camera beam-splitter system, which avoids parallax and provides realistic degradations under diverse scenes and motions. The dataset also offers aligned sRGB counterparts, enabling comprehensive evaluation of RAW-domain super-resolution methods under real-world conditions.

Quantitative results. We evaluate RAW video superresolution at 2×, 3×, and 4× scales on Real-RawVSR [9], as shown in Table IV. Among binarized methods, our method achieves the best performance across all metrics at 2× and 3× scales. At the more challenging 4× scale, our method attains the highest PSNR and ST-RRED and ranks secondbest in SSIM. Compared with the best-performing binarized alternatives at each scale (e.g., BHViT at 2×), our model improves PSNR by approximately 1.03 dB at 2× and 1.32 dB at 3×, which exceeds the performance gaps typically observed among existing binarized solutions.

We further compare against lightweight full-precision models, including TDAN [95], BasicVSR++ [29], and Realviformer [94]. Our method consistently outperforms TDAN and BasicVSR++ across all scales while maintaining significantly lower computational complexity. Compared to Realviformer, our method shows a modest performance drop but benefits from a substantially more compact design, with parameters about 25× smaller. This demonstrates that our method delivers high efficiency and competitive performance consistently across restoration tasks.

Qualitative results. Qualitative comparisons are shown in Fig. 9. In the 2× RAW video super-resolution case, our method reconstructs the fine textures of the white curtain more faithfully, while competing binarized methods tend to produce over-brightened results with missing shadow details and noticeable texture loss. For the 3× case, our method restores the textures along the door frame with higher fidelity, exhibiting clearer structural patterns and more consistent shading. In contrast, BNN [19], Bireal [61], and FRBC [85] results show incorrect smoothing on the same region, causing texture loss and muted boundaries, indicating inferior reconstruction of fine-grained details under higher magnification.

## F. Multi-Bit Quantization Experimental Results

To further explore the adaptability of our framework to different deployment scenarios, we evaluate its multi-bit quantization capability using lightweight low-bit configurations (i.e., 2-bit, 3-bit, and 4-bit for both weights and activations). We compare our method against state-of-the-art quantization methods, including LSQ [18], QuantSR [73], and Q-SCI [96], on the LLRVD [10] dataset.

As shown in Table VI, our method consistently outperforms all counterparts across all bit-width settings. On average, our method improves PSNR by approximately 0.5 dB over the best competing method, alongside noticeable gains in SSIM and ST-RRED. These results demonstrate that our quantization design remains effective beyond the binarized regime, preserving competitive restoration quality while retaining the advantages of low-bit efficiency.

## G. Downstream Video Applications

To assess the practical value of our restoration framework, we examine how restored videos benefit representative downstream high-level video tasks. We consider object detection and monocular depth estimation, validating that low-light RAW video enhancement yields measurable gains in downstream performance. Our method is compared with state-ofthe-art binarized methods under identical settings. Experiments are conducted on the large-scale SMOID [37] dataset to reduce statistical bias. For all tasks, restored RAW videos are first converted to RGB via ISP, matching typical usage in downstream video applications.

Object detection. Object detection is a fundamental component in video understanding pipelines, providing essential semantic cues for higher-level tasks such as tracking and scene analysis. We employ GroundingDINO [97], a popular textguided detector with strong zero-shot generalization. As a comprehensive prompt set, 1203 categories from the LVIS [98] dataset are used as textual inputs. We use the SMOID [37] dataset’s normal-light videos and process them with GroundingDINO to obtain detection labels as reliable pseudo ground truth. Following previous works in object detection [99], performance is measured using AP, AP50, and AP75. As shown in the upper half of Table V, our method achieves the highest scores across all metrics, outperforming all binarized competitors by a clear and consistent margin. Qualitative visualizations are provided in Fig. 10, showing that our BinRVR yields more reliable detections for small, partially occluded targets in challenging scenes.

TABLE IV  
QUANTITATIVE COMPARISON ON REAL-RAWVSR [9] FOR RAW VIDEO SUPER-RESOLUTION. FOR BINARIZED METHODS, THE BEST AND SECOND-BEST RESULTS ARE SHOWN IN BOLD AND UNDERLINED, RESPECTIVELY. LIGHTWEIGHT FULL-PRECISION METHODS ARE ADDITIONALLY INCLUDED FOR REFERENCE, WITH THE BEST RESULT SHOWN IN BOLD. PARAMS AND FLOPS ARE COMPUTED UNDER THE 2× SCALING SETTING. † INDICATES THAT REALVIFORMER [94] AND FRBC [85] FAIL TO TRAIN AT CERTAIN SCALES.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Params (M)↓</td><td rowspan="2">FLOPs (G)↓</td><td colspan="3">2×</td><td colspan="3">3×</td><td colspan="3">4×</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>ST-RRED↓</td><td>PSNR↑</td><td>SSIM↑</td><td>ST-RRED↓</td><td>PSNR↑</td><td>SSIM↑</td><td>ST-RRED↓</td></tr><tr><td>Bicubic</td><td></td><td></td><td>35.23</td><td>0.9434</td><td>0.2702</td><td>33.40</td><td>0.9121</td><td>0.2827</td><td>31.25</td><td>0.8558</td><td>0.7312</td></tr><tr><td colspan="10">Full-precision methods</td><td></td></tr><tr><td>TDAN [95]</td><td>1.86</td><td>13.76</td><td>34.19</td><td>0.9344</td><td>0.4757</td><td>33.06</td><td>0.9203</td><td>0.5424</td><td>32.13</td><td>0.8882</td><td>0.9604</td></tr><tr><td>BasicVSR++ [29]</td><td>7.18</td><td>113.13</td><td>35.71</td><td>0.9579</td><td>0.3684</td><td>35.42</td><td>0.9345</td><td>0.3096</td><td>33.91</td><td>0.9121</td><td>0.5131</td></tr><tr><td>Realviformer [94] †</td><td>5.82</td><td>40.46</td><td>37.53</td><td>0.9729</td><td>0.1720</td><td></td><td></td><td></td><td>34.57</td><td>0.9305</td><td>0.3654</td></tr><tr><td colspan="10">Binarized methods</td></tr><tr><td>BNN [19]</td><td>0.26</td><td>1.47</td><td>35.73</td><td>0.9537</td><td>0.2712</td><td>33.80</td><td>0.9282</td><td>0.3380</td><td>31.55</td><td>0.8941</td><td>0.7782</td></tr><tr><td>Bireal [61]</td><td>0.26</td><td>1.47</td><td>33.98</td><td>0.9489</td><td>0.3867</td><td>34.08</td><td>0.9357</td><td>0.3020</td><td>30.98</td><td>0.8973</td><td>0.6905</td></tr><tr><td>IRNet [62]</td><td>0.26</td><td>1.47</td><td>35.42</td><td>0.9509</td><td>0.3161</td><td>33.48</td><td>0.9263</td><td>0.3333</td><td>32.01</td><td>0.8966</td><td>0.7658</td></tr><tr><td>ReActNet [59]</td><td>0.29</td><td>1.65</td><td>35.93</td><td>0.9538</td><td>0.3175</td><td>34.38</td><td>0.9414</td><td>0.3094</td><td>32.39</td><td>0.9126</td><td>0.5722</td></tr><tr><td>BTM [69]</td><td>0.26</td><td>1.43</td><td>33.94</td><td>0.9481</td><td>0.4886</td><td>34.24</td><td>0.9382</td><td>0.3993</td><td>33.51</td><td>0.9258</td><td>0.5330</td></tr><tr><td>BBCU [20]</td><td>0.28</td><td>1.57</td><td>34.84</td><td>0.9515</td><td>0.4079</td><td>34.38</td><td>0.9416</td><td>0.4305</td><td>33.04</td><td>0.9187</td><td>0.5919</td></tr><tr><td>BiSRNet [84]</td><td>0.28</td><td>1.53</td><td>34.82</td><td>0.9469</td><td>0.3942</td><td>33.82</td><td>0.9377</td><td>0.4310</td><td>33.51</td><td>0.9224</td><td>0.4695</td></tr><tr><td>FRBC [85]†</td><td>0.26</td><td>1.47</td><td>35.67</td><td>0.9503</td><td>0.2464</td><td>32.60</td><td>0.8823</td><td>0.4607</td><td></td><td></td><td></td></tr><tr><td>BHViT [72]</td><td>0.28</td><td>1.56</td><td>36.24</td><td>0.9578</td><td>0.2397</td><td>34.85</td><td>0.9411</td><td>0.3534</td><td>33.38</td><td>0.9158</td><td>0.5780</td></tr><tr><td>BinRVR (Ours)</td><td>0.23</td><td>1.33</td><td>37.27</td><td>0.9688</td><td>0.1882</td><td>36.17</td><td>0.9537</td><td>0.1788</td><td>34.27</td><td>0.9232</td><td>0.3260</td></tr></table>

![](images/2c6681b92d463a64b160bd8d3255eeb9c45626cff83b38ff2d3f93e89360890c.jpg)  
Fig. 9. Visual comparisons of binarized methods on Real-RawVSR [9] for RAW video super-resolution (top: 2×; bottom: 3×). For each example, the left half shows zoomed-in RGB results obtained by applying an ISP pipeline to the RAW outputs, while the right half presents the corresponding error maps with respect to the ground truth.

Monocular depth estimation. Monocular depth estimation provides crucial geometric cues for downstream applications such as 3D reconstruction and scene understanding. We adopt Depth Anything V2 [100], a widely used framework with strong zero-shot generalization, to evaluate how enhanced videos affect depth prediction accuracy. Ground-truth reference is generated by applying Depth Anything V2 to the SMOID [37] dataset’s normal-light videos. For quantitative evaluation, we calculate the root-mean-square error (RMSE) to measure numerical accuracy and structural similarity (SSIM) to assess structural consistency in the predicted depth maps. As reported in the lower half of Table V, our method again surpasses all binarized counterparts, demonstrating more accurate geometric estimation and better structural fidelity. Qualitative visualizations are provided in Fig. 11, where our method estimates depth more accurately on the rear part of the truck.

TABLE V  
QUANTITATIVE RESULTS OF DOWNSTREAM APPLICATIONS OF LOW-LIGHT RAW VIDEO ENHANCEMENT USING THE SMOID [37] DATASET. WE PROVIDE RESULTS FOR OBJECT DETECTION AND DEPTH ESTIMATION. THE BEST AND SECOND-BEST VALUES ARE BOLD AND UNDERLINE, RESPECTIVELY.
<table><tr><td>Tasks</td><td>Metrics</td><td>Linear Scaled</td><td>BNN [19]</td><td>Bireal [61]</td><td>IRNet [62]</td><td>ReActNet [59]</td><td>BTM [69]</td><td>BBCU [20]</td><td>BiSRNet [84]</td><td>FRBC [85]</td><td>BHViT [72]</td><td>BRVE [86]</td><td>BinRVR (Ours)</td></tr><tr><td rowspan="2">Object Detection</td><td>AP↑</td><td>61.98</td><td>62.85</td><td>65.02</td><td>63.49</td><td>65.46</td><td>66.83</td><td>66.61</td><td>66.60</td><td>53.39</td><td>66.97</td><td>68.52</td><td>70.62</td></tr><tr><td>AP50↑</td><td>64.93 63.34</td><td>65.63 64.22</td><td>67.91 66.67</td><td>66.29 64.98</td><td>68.15</td><td>69.77</td><td>69.44</td><td>69.59</td><td>56.31</td><td>69.84</td><td>71.05</td><td>73.24</td></tr><tr><td rowspan="2">Depth Estimation</td><td>AP75↑</td><td></td><td></td><td></td><td></td><td>66.96</td><td>68.49</td><td>68.27</td><td>68.27</td><td>54.67</td><td>68.66</td><td>69.97</td><td>72.11</td></tr><tr><td>RMSE↓ SSIM↑</td><td>0.1308 0.9324</td><td>0.1251 0.9335</td><td>0.1270 0.9313</td><td>0.1255 0.9323</td><td>0.1224 0.9342</td><td>0.1136 0.9382</td><td>0.1115 0.9392</td><td>0.1149 0.9372</td><td>0.2215 0.8770</td><td>0.1127 0.9386</td><td>0.1067 0.9463</td><td>0.1029 0.9482</td></tr></table>

![](images/72ed85f2f968f8c24230a8b186eb5b679093241ea200aa22a4aa78dff0ceb76f.jpg)

Fig. 10. Qualitative comparisons of object detection results produced by GroundingDINO [97] on enhanced videos  
![](images/bb9b9a95f6237953654f14acdad1bbacf30cec7c8d5c0557b58c3951b65c93d4.jpg)  
Fig. 11. Qualitative comparisons of monocular depth estimation results produced by Depth Anything V2 [100] on enhanced videos.

TABLE VI  
MULTI-BIT QUANTIZATION RESULTS ON LLRVD [10] DATASET FOR LOW-LIGHT RAW VIDEO ENHANCEMENT.
<table><tr><td>Method</td><td>Bits (w/a)</td><td>Params↓ FLOPs↓</td><td>PSNR↑ SSIM↑ ST-RRED↓</td><td></td></tr><tr><td>LSQ [18]</td><td>4/4</td><td>0.80M 5.81G</td><td>37.10 0.9585</td><td>0.0461</td></tr><tr><td>QuantSR [73]</td><td>4/4</td><td>0.80M 5.81G</td><td>37.09 0.9585</td><td>0.0460</td></tr><tr><td>Q-SCI [96]</td><td>4/4</td><td>0.80M 5.81G</td><td>37.13 0.9590</td><td>0.0433</td></tr><tr><td>Ours</td><td>4/4</td><td>0.80M 5.81G</td><td>37.71 0.9646</td><td>0.0382</td></tr><tr><td>LSQ [18]</td><td>3/3</td><td>0.63M 3.39G</td><td>37.05 0.9580</td><td>0.0465</td></tr><tr><td>QuantSR [73]</td><td>3/3</td><td>0.63M 3.39G</td><td>37.06 0.9581</td><td>0.0464</td></tr><tr><td>Q-SCI [96]</td><td>3/3</td><td>0.63M 3.39G</td><td>37.10 0.9588</td><td>0.0460</td></tr><tr><td>Ours</td><td>3/3</td><td>0.63M 3.39G</td><td>37.64 0.9641</td><td>0.0390</td></tr><tr><td>LSQ [18]</td><td>2/2</td><td>0.45M 2.17G</td><td>36.99 0.9573</td><td>0.0480</td></tr><tr><td>QuantSR [73]</td><td>2/2</td><td>0.45M 2.17G</td><td>36.99 0.9573</td><td>0.0479</td></tr><tr><td>Q-SCI [96]</td><td>2/2</td><td>0.45M 2.17G</td><td>37.01 0.9576</td><td>0.0475</td></tr><tr><td>Ours</td><td>2/2</td><td>0.45M 2.17G</td><td>37.53 0.9631</td><td>0.0403</td></tr></table>

## H. Discussion

We discuss several key aspects of our framework, including the effectiveness of RAW-domain restoration, temporal modeling design choices, computational overhead, and ablation studies on our BIIM and DAB-Conv.

Effectiveness of RAW-Domain restoration. We evaluate different signal representations for low-light video enhancement on the LLRVD [10] dataset, with results summarized in Table VII. Among all settings, RAW2RAW+ISP achieves the best performance in terms of all metrics. Preserving RAW signals as both the input and output of the restoration stage minimizes information loss introduced by early ISP processing and enables more accurate recovery. In contrast, RAW2RGB suffers from the joint learning of enhancement and color mapping, while RGB2RGB performs the worst due to irreversible quantization in low-intensity regions. When compression is applied, RAW-domain restoration remains robust, whereas restoring already compressed RGB videos leads to severe degradation. These results demonstrate that RAWdomain restoration provides the most reliable outcomes.

Effectiveness of sliding-window recurrent connection. To directly compare temporal modeling paradigms, we conduct an architecture-level ablation study under the same backbone, 1-bit quantization setting, and training configuration. As shown in Table VIII, the pure sliding-window structure is efficient but lacks long-term recurrent propagation, while the pure unidirectional recurrent structure improves temporal consistency but is weaker in short-term neighboring-frame aggregation. The proposed hybrid structure achieves the best PSNR, SSIM, and ST-RRED, demonstrating a better balance between short-term multi-frame aggregation and long-term temporal modeling.

![](images/fb3a7514f726d51dafd88512db8925eb31d597ac4a42e1511b8d09ad0f8df590.jpg)

![](images/29d4e4ca4d76cf946275fc19a336229d1f9ffcf0b2a8bcaa59c35db65a7c8076.jpg)

![](images/ad8c40bfdb701db7730352e95c4ab9a9ba4e22d42d1004eded6e6e600008860e.jpg)  
Fig. 12. Effect of the sliding window stride on recurrent propagation. A larger stride improves efficiency but compromises temporal consistency in restored videos across frames.

TABLE VII  
COMPARISON OF DIFFERENT LOW-LIGHT VIDEO ENHANCEMENT SETTINGS ON LLRVD [10] DATASET. METRICS ARE COMPUTED IN THE RGB DOMAIN AFTER ISP PROCESSING.
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>ST-RRED↓</td></tr><tr><td>RAW2RAW+ISP</td><td>30.59</td><td>0.8408</td><td>0.2002</td></tr><tr><td>RAW2RGB</td><td>27.53</td><td>0.8143</td><td>0.2938</td></tr><tr><td>RGB2RGB</td><td>24.91</td><td>0.7994</td><td>0.5941</td></tr><tr><td>RAW2RAW+ISP+H264</td><td>30.24</td><td>0.8421</td><td>0.1810</td></tr><tr><td>RGB2RGB+H264</td><td>24.88</td><td>0.8170</td><td>0.5610</td></tr><tr><td>H264+RGB2RGB</td><td>20.15</td><td>0.6138</td><td>1.4199</td></tr></table>

Effect of stride in sliding window recurrent connection. In our BinRVR, the sliding-window recurrent connection of the BIIM decoder aggregates temporal cues across frames, and its behavior is largely determined by the stride. As shown in Fig. 12, a stride of 1 maintains full recurrent propagation and delivers the best temporal fidelity. Increasing the stride to 2 retains only the most recent feature for recurrence, reducing nearly 40% FLOPs with a 0.40 dB drop in PSNR, suggesting that recurrent updates are beneficial but not required at every step. With a stride 3, recurrent links are removed entirely and each window is processed independently, yielding the highest efficiency but causing a sharp rise in ST-RRED due to temporal inconsistency. These observations confirm the effectiveness of the sliding window recurrent connection in modeling temporal information across frames.

Long-sequence temporal consistency. To examine temporal consistency on long sequences, we evaluate BinRVR with different input lengths on the SMOID test set [37]. For fair comparison, all settings use the same 30-frame target interval for evaluation. Specifically, for an input length L, the model takes the segment ending at the same frame, i.e., frames $[ t - L + 1 , \ldots , t ] ,$ , while PSNR, SSIM, and ST-RRED are computed only on the identical target interval. Thus, different lengths only change the amount of historical context provided to the recurrent connection. As shown in Table X, the average PSNR and SSIM over three gain settings remain nearly stable when increasing the sequence length from 30 to 120 frames, and ST-RRED shows no sharp degradation. This indicates that BinRVR does not suffer from severe error accumulation within the tested range, benefiting from the proposed sliding-window recurrent connection that injects current local-window features at each step rather than relying solely on the propagated hidden state. Nevertheless, extremely long sequences with severe motion may still challenge the current unidirectional design and will be explored in future work.

TABLE VIII  
ARCHITECTURE-LEVEL ABLATION STUDY ON THE TEMPORAL MODELING PARADIGM. ALL VARIANTS USE THE SAME BACKBONE, 1-BIT QUANTIZATION SETTING, AND TRAINING CONFIGURATION.
<table><tr><td>Arch.</td><td>Window</td><td>Rec.</td><td>Params↓ FLOPs↓</td><td>PSNR↑</td><td>SSIM↑</td><td>ST-RRED↓</td></tr><tr><td>Pure sliding window</td><td>3</td><td>X</td><td>0.22M 0.74G</td><td>36.83</td><td>0.9567</td><td>0.0648</td></tr><tr><td>Pure recurrent</td><td>1</td><td>√</td><td>0.22M 1.05G</td><td>36.98</td><td>0.9579</td><td>0.0506</td></tr><tr><td>Hybrid (Ours)</td><td>3</td><td>√</td><td>0.22M 1.23G</td><td>37.21</td><td>0.9599</td><td>0.0436</td></tr></table>

TABLE IX

ABLATION STUDIES ON OUR BINARIZED INFORMATION INTERACTION (BIIM) MODULE .
<table><tr><td>Partition</td><td>Ablation</td><td>PSNR↑SSIM↑ST-RRED↓Params↓FLOPs↓</td><td></td></tr><tr><td>BIIM Encoder</td><td>w/o TII Same as Decoder-TII</td><td>37.060.9585 0.0468 37.15 0.9586 0.0450</td><td>0.22M 1.23G 0.24M 1.43G</td></tr><tr><td>BIIM Decoder</td><td>w/o TII Same as Encoder-TII w/o SII Strip Kernel → 1 × 9 Strip Kernel → 1 × 13 Branch Ratio → 1/4</td><td>37.000.9581 0.0470 37.14 0.9595 0.0451 37.04 0.9580 0.0469 37.160.9592 0.0453 37.17 0.9596 0.0426 37.17 0.9594 0.0443</td><td>0.21M 1.12G 0.21M 1.12G 0.21M 1.17G 0.22M 1.23G 0.22M 1.23G 0.23M 1.23G</td></tr><tr><td></td><td>Branch Ratio → 1/16 BinRVR (Ours)</td><td>37.15 0.9596 0.0447 37.21 0.9599 0.0436</td><td>0.22M 1.23G 0.22M 1.23G</td></tr></table>

Effectiveness of binarized information interaction module. Table IX reports ablations on BIIM. Removing Temporal Information Interaction (TII) in the decoder results in a PSNR drop of 0.21 dB, which is notably larger than the 0.15 dB drop caused by removing it in the encoder, indicating that temporal interaction in the decoder plays a more critical role in reconstruction. Eliminating Spatial Information Interaction (SII) reduces performance by 0.17 dB, showing that spatial interaction remains beneficial under binarization. Using identical TII structures or modifying strip kernel sizes and branch ratios leads to only marginal changes while increasing parameters or FLOPs, indicating that the adopted asymmetric design and hyperparameter settings are already well balanced. The visual ablation results in Fig. 13 further show that removing BIIM leads to less accurate restoration and larger reconstruction errors, while the full BinRVR recovers cleaner structures closer to the ground truth. These results verify that all components of BIIM contribute positively to its effectiveness.

Effectiveness of distribution-aware convolution. Table XI evaluates the impact of different statistical descriptors in our Distribution-Aware Binarized Convolution. To further explain the motivation, Fig. 14 visualizes the activation statistics of a representative DAB-Conv layer and their relationship with the predicted activation scaling factors. The channel-wise mean, absolute mean, and standard deviation vary noticeably across channels and gain settings, while representative channel histograms show shifted, asymmetric, or narrowly concentrated activation distributions. These observations indicate that RAW video activations cannot be well approximated by a fixed or globally shared scale, and motivate the joint modeling of center shift, response magnitude, and dynamic range. Moreover, Fig. 14 shows non-trivial correlations between these statistics and the predicted scaling factors, suggesting that DAB-Conv learns input-dependent channel-wise modulation rather than applying a fixed scaling rule. The visual ablation in Fig. 13(b) further shows that removing DAB-Conv leads to less faithful restoration and larger reconstruction errors.

TABLE X  
LONG-SEQUENCE EVALUATION ON THE SMOID [37] TEST SET. THE METRICS ARE AVERAGED OVER GAIN0, GAIN15, AND GAIN30, AND COMPUTED ON THE SAME TARGET INTERVAL.
<table><tr><td rowspan="2">Sequence Length</td><td colspan="3">Avg. over Gain0 / Gain15 / Gain30</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>ST-RRED↓</td></tr><tr><td>30 frames</td><td>40.86</td><td>0.9789</td><td>0.0610</td></tr><tr><td>60 frames</td><td>40.88</td><td>0.9790</td><td>0.0606</td></tr><tr><td>90 frames</td><td>40.88</td><td>0.9790</td><td>0.0606</td></tr><tr><td>120 frames</td><td>40.88</td><td>0.9790</td><td>0.0605</td></tr></table>

TABLE XI

ABLATION STUDIES FOR OUR DISTRIBUTION-AWARE BINARIZED CONVOLUTION.
<table><tr><td>Mean(·) Std(·) Mean(| · |)</td><td></td><td>PSNR↑ SSIM↑ ST-RRED↓</td><td></td><td>Params↓ FLOPs↓</td></tr><tr><td>X</td><td>X</td><td>37.04</td><td>0.9581 0.0458</td><td>0.222M 1.21G</td></tr><tr><td>√</td><td>X X</td><td>37.12 0.9589</td><td>0.0447</td><td>0.222M 1.21G</td></tr><tr><td>√</td><td>√ X</td><td>37.17 0.9595</td><td>0.0444</td><td>0.223M 1.21G</td></tr><tr><td>√</td><td>√</td><td>√ 37.21</td><td>0.9599 0.0436</td><td>0.223M 1.23G</td></tr></table>

Consistent with this analysis, removing all statistics causes a PSNR drop of 0.17 dB, indicating that distribution cues are necessary for stable binarization. Introducing only the mean term yields a 0.08 dB improvement, and combining the mean with standard deviation further improves performance to a total gain of 0.13 dB, confirming that joint modeling of first- and second-order statistics is beneficial. Incorporating Mean(| · |) provides additional gains and leads to the best performance. These results demonstrate that leveraging multiple statistical descriptors enables more reliable distribution-aware scaling.

Deployment feasibility evaluation. Native end-to-end BNN execution is still rarely supported by mainstream deployment hardware and runtimes, because it typically requires dedicated bit-packing and XNOR-popcount kernels. Therefore, instead of reporting only normalized FLOPs and parameter counts, we further evaluate actual-runtime deployment on representative and accessible GPU, CPU, and mobile NPU platforms under currently supported inference settings, including FP32, W8A8, and W4A8. As summarized in Table XII, BinRVR consistently achieves higher throughput than the representative full-precision video restoration model across diverse hardware. Under FP32 inference, BinRVR improves FPS by about 24.9× on the RTX 4060 GPU, 15.1× on Jetson Orin NX, 20.9× on the Core Ultra 9 CPU, 10.9× on the Snapdragon 8 Elite CPU, 2.1× on the Snapdragon 8 Elite NPU, and 13.0× on the Ascend 310B4 NPU. Moreover, the W8A8 and W4A8 settings further improve actual throughput, with up to 4.2× FPS improvement over ShiftNet on the Snapdragon 8 Elite NPU. These results demonstrate that our BinRVR can be effectively mapped to practical low-bit inference paths under existing hardware and runtime constraints. Native 1-bit BNN acceleration may provide additional gains when dedicated kernels become more widely available.

![](images/53ebd6af067a01e9818f039c114646a72fac288ea5a0ff32cb8743d7062f1939.jpg)

Fig. 13. Visual ablation comparison of representative BinRVR variants. (a) Ablation of temporal modeling components. (b) Ablation of spatial modeling components. Each comparison includes the degraded input, ablation variants without key components, the full model, the ground truth, and the corresponding error maps.  
![](images/ba53745caea298df9b97d739965850f4986ea97d3beda835f8c6c1f7ab507384.jpg)  
Fig. 14. Visualization of activation distributions and DAB-Conv scaling factors. (a)–(c) Channel-wise activation statistics before binarization under different gain settings. (d)–(f) Representative activation distributions from different channels. (g)–(i) Predicted DAB-Conv scales versus activation statistics. The results show that activation distributions vary across channels and degradations, motivating input-dependent channel-wise scaling.

## V. CONCLUSION

This paper presents BinRVR, a systematic high-efficiency binarized framework for multi-task RAW video restoration that explicitly targets the unique challenges of temporal modeling and distribution mismatch under extreme quantization. Unlike existing binarized networks that mainly focus on image-level tasks, BinRVR is specifically designed for video restoration in the RAW domain, where preserving temporal coherence and fine-grained sensor information is critical. To this end, we introduce a Binarized Information Interaction Module (BIIM) to enable efficient joint modeling of inter-frame temporal dependencies and intra-frame spatial structures using lightweight, binarization-friendly operations. In addition, we propose a Distribution-Aware Binarized Convolution (DAB-Conv) that leverages full-precision activation statistics to alleviate quantization-induced errors caused by diverse and asymmetric feature distributions. Extensive experiments across lowlight enhancement, denoising, deblurring, and super-resolution demonstrate that BinRVR achieves a favorable accuracyefficiency trade-off, reducing computation and parameters by approximately 96% while incurring only a minor performance degradation compared with full-precision models.

TABLE XII  
ACTUAL-RUNTIME DEPLOYMENT COMPARISON WITH THE REPRESENTATIVE FULL-PRECISION METHOD SHIFTNET [8] ON DIFFERENT HARDWARE PLATFORMS UNDER CURRENTLY SUPPORTED INFERENCE MODES. PSNR IS EVALUATED ON THE LLRVD [10] TEST SET.
<table><tr><td rowspan=1 colspan=1>Platform</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Mode</td><td rowspan=1 colspan=1>FPS↑</td><td rowspan=1 colspan=1>Latency(ms/f.)↓</td><td rowspan=1 colspan=1>PSNR↑</td></tr><tr><td rowspan=1 colspan=1>NVIDIA RTX 4060 GPULocal laptop</td><td rowspan=1 colspan=1>ShiftNetOurs</td><td rowspan=1 colspan=1>FP32FP32</td><td rowspan=1 colspan=1>3.7493.22</td><td rowspan=1 colspan=1>267.3410.73</td><td rowspan=1 colspan=1>37.8737.81</td></tr><tr><td rowspan=1 colspan=1>NVIDIA Ampere GPUNVIDIA Jetson Orin NX</td><td rowspan=1 colspan=1>ShiftNetOursOurs</td><td rowspan=1 colspan=1>FP32FP32W8A8</td><td rowspan=1 colspan=1>1.3019.9435.18</td><td rowspan=1 colspan=1>770.3950.1628.42</td><td rowspan=1 colspan=1>37.8737.8137.77</td></tr><tr><td rowspan=1 colspan=1>Intel Core Ultra 9 CPULocal laptop</td><td rowspan=1 colspan=1>ShiftNetOursOurs</td><td rowspan=1 colspan=1>FP32FP32W8A8</td><td rowspan=1 colspan=1>0.8718.1420.96</td><td rowspan=1 colspan=1>1144.1755.1247.70</td><td rowspan=1 colspan=1>37.8737.8137.77</td></tr><tr><td rowspan=1 colspan=1>Snapdragon 8 Elite CPUXiaomi 15</td><td rowspan=1 colspan=1>ShiftNetOursOurs</td><td rowspan=1 colspan=1>FP32FP32W8A8</td><td rowspan=1 colspan=1>0.363.949.93</td><td rowspan=1 colspan=1>2793.80253.58100.72</td><td rowspan=1 colspan=1>37.8137.7937.68</td></tr><tr><td rowspan=1 colspan=1>Snapdragon 8 Elite NPUQualcomm AI Hub</td><td rowspan=1 colspan=1>ShiftNetOursOursOurs</td><td rowspan=1 colspan=1>FP32FP32W8A8W4A8</td><td rowspan=1 colspan=1>45.5495.26154.85192.33</td><td rowspan=1 colspan=1>21.9610.486.465.20</td><td rowspan=1 colspan=1>37.8737.8137.7737.75</td></tr><tr><td rowspan=1 colspan=1>Huawei Ascend 310B4 NPUHuawei Atlas 200I DK A2</td><td rowspan=1 colspan=1>ShiftNetOursOurs</td><td rowspan=1 colspan=1>FP32FP32W8A8</td><td rowspan=1 colspan=1>2.6234.1865.76</td><td rowspan=1 colspan=1>382.2529.2615.21</td><td rowspan=1 colspan=1>37.8737.8137.77</td></tr></table>

Beyond the strictly binarized setting, the proposed framework extends to multi-bit quantization through a unified distribution-aware formulation, enabling flexible adaptation to different hardware constraints without modifying the core architecture. Moreover, we show that the improved restoration quality provided by BinRVR translates into consistent gains in downstream video applications, including object detection and monocular depth estimation, highlighting its practical value beyond low-level reconstruction. Nevertheless, largemotion deblurring cases may still be challenging, because the recoverable RAW signal can be limited and the lightweight temporal interaction in BIIM avoids heavy motion alignment for efficiency. Future work may explore adaptive bit-width allocation, degradation-adaptive and motion-aware modeling, and real-device deployment optimization to further improve robustness and practical efficiency.

## REFERENCES

[1] J. Li, W. Ji, S. Wang, W. Li et al., “Dvsod: Rgb-d video salient object detection,” Proc. Adv. Neural Inform. Process. Syst., vol. 36, 2024.

[2] L. Jiao, R. Zhang, F. Liu, S. Yang, B. Hou, L. Li, and X. Tang, “New generation deep learning for video object detection: A survey,” IEEE Trans. Neural Netw. Learn. Syst., vol. 33, no. 8, pp. 3195–3215, 2021.

[3] L. Sun, J.-W. Bian, H. Zhan, W. Yin, I. Reid, and C. Shen, “Scdepthv3: Robust self-supervised monocular depth estimation for dynamic scenes,” IEEE Trans. Pattern Anal. Mach. Intell., 2023.

[4] Y. Liang, Y. Hu, W. Shao, and Y. Fu, “Distilling monocular foundation model for fine-grained depth completion,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2025, pp. 22 254–22 265.

[5] T. Zhou, F. Porikli, D. J. Crandall, L. Van Gool, and W. Wang, “A survey on deep learning technique for video segmentation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 6, pp. 7099–7122, 2022.

[6] Y. Liu, C. Shen, C. Yu, and J. Wang, “Efficient semantic video segmentation with per-frame inference,” in Proc. Eur. Conf. Comput. Vis., 2020, pp. 352–368.

[7] J. Liang, J. Cao, Y. Fan, K. Zhang, R. Ranjan, Y. Li, R. Timofte, and L. Van Gool, “Vrt: A video restoration transformer,” IEEE Trans. Image Process., 2024.

[8] D. Li, X. Shi, Y. Zhang, K. C. Cheung, S. See, X. Wang, H. Qin, and H. Li, “A simple baseline for video restoration with grouped spatialtemporal shift,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2023, pp. 9822–9832.

[9] H. Yue, Z. Zhang, and J. Yang, “Real-rawvsr: Real-world raw video super-resolution with a benchmark dataset,” in Proc. Eur. Conf. Comput. Vis., 2022, pp. 608–624.

[10] Y. Fu, Z. Wang, T. Zhang, and J. Zhang, “Low-light raw video denoising with a high-quality realistic motion dataset,” IEEE Trans. Multimedia, vol. 25, pp. 8119–8131, 2022.

[11] H. Huang, W. Yang, Y. Hu, J. Liu, and L.-Y. Duan, “Towards low light enhancement with raw images,” IEEE Trans. Image Process., vol. 31, pp. 1391–1405, 2022.

[12] Y. Zou, C. Yan, and Y. Fu, “Rawhdr: High dynamic range image reconstruction from a single raw image,” in Proc. Int. Conf. Comput. Vis., 2023, pp. 12 334–12 344.

[13] J. Li, X. Wu, Z. Niu, and W. Zuo, “Unidirectional video denoising by mimicking backward recurrent modules with look-ahead forward ones,” in Proc. Eur. Conf. Comput. Vis., 2022, pp. 592–609.

[14] Z. Liu, J. Li, Z. Shen, G. Huang, S. Yan, and C. Zhang, “Learning efficient convolutional networks through network slimming,” in Proc. Int. Conf. Comput. Vis., 2017, pp. 2736–2744.

[15] H. Wang and Y. Fu, “Trainability preserving neural pruning,” in Proc. Int. Conf. Learn. Represent., 2023.

[16] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “Mobilenetv2: Inverted residuals and linear bottlenecks,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2018, pp. 4510–4520.

[17] A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta, and Y. Bengio, “Fitnets: Hints for thin deep nets,” in Proc. Int. Conf. Learn. Represent., 2015.

[18] S. K. Esser, J. L. McKinstry, D. Bablani, R. Appuswamy, and D. S. Modha, “Learned step size quantization,” in Proc. Int. Conf. Learn. Represent., 2020.

[19] I. Hubara, M. Courbariaux, D. Soudry, R. El-Yaniv, and Y. Bengio, “Binarized neural networks,” Proc. Adv. Neural Inform. Process. Syst., vol. 29, 2016.

[20] B. Xia, Y. Zhang, Y. Wang, Y. Tian, W. Yang, R. Timofte, and L. Van Gool, “Basic binary convolution unit for binarized image restoration network,” in Proc. Int. Conf. Learn. Represent., 2023.

[21] Y. Tian, Y. Zhang, Y. Fu, and C. Xu, “Tdan: Temporally-deformable alignment network for video super-resolution,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2020, pp. 3360–3369.

[22] Y. Lin, T. Zhang, P. Sun, Z. Li, and S. Zhou, “Fq-vit: Post-training quantization for fully quantized vision transformer,” in Proceedings of the International Joint Conference on Artificial Intelligence, 2022, pp. 1173–1179.

[23] Z. Li, J. Xiao, L. Yang, and Q. Gu, “I-vit: Integer-only quantization for efficient vision transformer inference,” in Proc. Int. Conf. Comput. Vis., 2023, pp. 17 065–17 075.

[24] Z. Teed and J. Deng, “Raft: Recurrent all-pairs field transforms for optical flow,” in Proc. Eur. Conf. Comput. Vis., 2020, pp. 402–419.

[25] T. Xue, B. Chen, J. Wu, D. Wei, and W. T. Freeman, “Video enhancement with task-oriented flow,” Int. J. Comput. Vis., vol. 127, pp. 1106–1125, 2019.

[26] J. Dai, H. Qi, Y. Xiong, Y. Li, G. Zhang, H. Hu, and Y. Wei, “Deformable convolutional networks,” in Proc. Int. Conf. Comput. Vis., 2017, pp. 764–773.

[27] M. Rastegari, V. Ordonez, J. Redmon, and A. Farhadi, “Xnor-net: Imagenet classification using binary convolutional neural networks,” in Proc. Eur. Conf. Comput. Vis., 2016, pp. 525–542.

[28] H. Son, J. Lee, J. Lee, S. Cho, and S. Lee, “Recurrent video deblurring with blur-invariant motion estimation and pixel volumes,” ACM Trans. Graph., vol. 40, no. 5, pp. 1–18, 2021.

[29] K. C. Chan, S. Zhou, X. Xu, and C. C. Loy, “Basicvsr++: Improving video super-resolution with enhanced propagation and alignment,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2022, pp. 5972–5981.

[30] H. Li, Z. Wu, R. Shao, T. Zhang, and Y. Fu, “Noise calibration and spatial-frequency interactive network for stem image enhancement,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2025, pp. 21 287– 21 296.

[31] Y. Zhang, T. Zhang, J. Nie, and Y. Fu, “Real noise decoupling for hyperspectral image denoising,” in Proc. AAAI, vol. 40, no. 15, 2026, pp. 12 925–12 933.

[32] Y. Zhang, Z. Lai, T. Zhang, Y. Fu, and C. Zhou, “Unaligned rgb guided hyperspectral image super-resolution with spatial-spectral concordance,” Int. J. Comput. Vis., vol. 133, no. 9, pp. 6590–6610, 2025.

[33] Z. Hua, S. Qu, L. Yan, W. Dong, Y. Zhou, H. Li, X. Chang, L. Bao, Y. Wang, F. Ying et al., “Deep-learning aided atomic-scale observation of anisotropic melting of the charge density wave in tas2,” Small, vol. 21, no. 45, p. e07496, 2025.

[34] Y. Zhang, T. Zhang, J. Nie, and Y. Fu, “Enhancing unregistered hyperspectral image super-resolution via unmixing-based abundance fusion learning,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2026, pp. 41 573–41 583.

[35] T. Zhu, H. Li, and Y. Fu, “Trim-sod: A multi-modal, multi-task, and multi-scale spacecraft optical dataset,” Space: Science & Technology, vol. 5, p. 0299, 2025.

[36] C. Liu, X. Wang, Y. Fan, S. Li, and X. Qian, “Decoupling degradations with recurrent network for video restoration in under-display camera,” in Proc. AAAI, vol. 38, no. 4, 2024, pp. 3558–3566.

[37] H. Jiang and Y. Zheng, “Learning to see moving objects in the dark,” in Proc. Int. Conf. Comput. Vis., 2019, pp. 7324–7333.

[38] H. Yue, C. Cao, L. Liao, R. Chu, and J. Yang, “Supervised raw video denoising with a benchmark dataset on dynamic scenes,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2020, pp. 2301–2310.

[39] D. Li, C. Xu, K. Zhang, X. Yu, Y. Zhong, W. Ren, H. Suominen, and H. Li, “Arvo: Learning all-range volumetric correspondence for video deblurring,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2021, pp. 7721–7731.

[40] M. Tassano, J. Delon, and T. Veit, “Fastdvdnet: Towards real-time deep video denoising without flow estimation,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2020, pp. 1354–1363.

[41] H. Chen, Y. Jin, K. Xu, Y. Chen, and C. Zhu, “Multiframe-tomultiframe network for video denoising,” IEEE Trans. Multimedia, vol. 24, pp. 2164–2178, 2021.

[42] M. Maggioni, Y. Huang, C. Li, S. Xiao, Z. Fu, and F. Song, “Efficient multi-stage video denoising with recurrent spatio-temporal fusion,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2021, pp. 3466–3475.

[43] J. Liang, J. Cao, G. Sun, K. Zhang, L. Van Gool, and R. Timofte, “Swinir: Image restoration using swin transformer,” in Proc. Int. Conf. Comput. Vis., 2021, pp. 1833–1844.

[44] H. Li and Y. Fu, “Fcdfusion: A fast, low color deviation method for fusing visible and infrared image pairs,” Comput. Vis. Media, vol. 11, no. 1, pp. 195–211, 2025.

[45] Z. Gao, K. Xu, X. Zhang, H. Zhuang, T. Wan, B. Ding, X. Mao, and W. Huaimin, “Rethinking obscured sub-optimality in analytic learning for exemplar-free class-incremental learning,” IEEE Trans. Circuits Syst. Video Technol., vol. 36, no. 11, pp. 1123–1136, 2025.

[46] H. Li, F. Dai, Q. Zhao, Y. Ma, J. Cao, and Y. Zhang, “Non-uniform compressive sensing imaging based on image saliency,” Chin. J. Electron., vol. 32, no. 1, pp. 159–165, 2023.

[47] L. Yanshan, C. Shifu, L. Wenhan, Z. Li, and X. Weixin, “Hyperspectral image super-resolution based on spatial-spectral feature extraction network,” Chin. J. Electron., vol. 32, no. 3, pp. 415–428, 2023.

[48] Y. Tian, Y. Fu, and J. Zhang, “Transformer-based under-sampled singlepixel imaging,” Chin. J. Electron., vol. 32, no. 5, pp. 1151–1159, 2023.

[49] T. Zhang, Y. Fu, J. Zhang, and C. Yan, “Deep guided attention network for joint denoising and demosaicing in real image,” Chin. J. Electron., vol. 33, no. 1, pp. 303–312, 2024.

[50] Q. Liu, Y. Jiang, Z. Tan, D. Chen, Y. Fu, Q. Chu, G. Hua, and N. Yu, “Transformer based pluralistic image completion with reduced information loss,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 10, pp. 6652–6668, 2024.

[51] B. Wang, T. Zhang, and Y. Fu, “Visual differential-spatially projected transformer for efficient hyperspectral images super-resolution,” IEEE Trans. Geosci. Remote Sens., 2026.

[52] J. Li, X. Wu, Z. Niu, and W. Zuo, “Unidirectional video denoising by mimicking backward recurrent modules with look-ahead forward ones,” in Proc. Eur. Conf. Comput. Vis., 2022, pp. 592–609.

[53] H. Yue, C. Cao, L. Liao, R. Chu, and J. Yang, “Supervised raw video denoising with a benchmark dataset on dynamic scenes,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2020, pp. 2301–2310.

[54] K. Xu, Z. Yu, X. Wang, M. B. Mi, and A. Yao, “Enhancing video superresolution via implicit resampling-based alignment,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 2546–2555.

[55] H. Zhang, H. Xie, and H. Yao, “Blur-aware spatio-temporal sparse transformer for video deblurring,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 3616–3626.

[56] Y. Cheng, X. Liu, and J. Yang, “Recaptured raw screen image and video demoireing via channel and spatial modulations,”´ Proc. Adv. Neural Inform. Process. Syst., vol. 36, 2024.

[57] A. Bulat and G. Tzimiropoulos, “Xnor-net++: Improved binary neural networks,” arXiv preprint arXiv:1909.13863, 2019.

[58] Z. Tu, X. Chen, P. Ren, and Y. Wang, “Adabin: Improving binary neural networks with adaptive binary sets,” in Proc. Eur. Conf. Comput. Vis., 2022, pp. 379–395.

[59] Z. Liu, Z. Shen, M. Savvides, and K.-T. Cheng, “Reactnet: Towards precise binary neural network with generalized activation functions,” in Proc. Eur. Conf. Comput. Vis., 2020, pp. 143–159.

[60] Y. Shang, D. Xu, B. Duan, Z. Zong, L. Nie, and Y. Yan, “Lipschitz continuity retained binary neural network,” in Proc. Eur. Conf. Comput. Vis., 2022, pp. 603–619.

[61] Z. Liu, B. Wu, W. Luo, X. Yang, W. Liu, and K.-T. Cheng, “Bireal net: Enhancing the performance of 1-bit cnns with improved representational capability and advanced training algorithm,” in Proc. Eur. Conf. Comput. Vis., 2018, pp. 722–737.

[62] H. Qin, R. Gong, X. Liu, M. Shen, Z. Wei, F. Yu, and J. Song, “Forward and backward information retention for accurate binary neural networks,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2020, pp. 2250–2259.

[63] W. Tang, G. Hua, and L. Wang, “How to train a compact binary neural network with high accuracy?” in Proc. AAAI, vol. 31, no. 1, 2017.

[64] A. J. Redfern, L. Zhu, and M. K. Newquist, “Bcnn: a binary cnn with all matrix ops quantized to 1 bit precision,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2021, pp. 4604–4612.

[65] E. Vargas, C. V. Correa, C. Hinojosa, and H. Arguello, “Biper: Binary neural networks using a periodic function,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 5684–5693.

[66] N. Guo, J. Bethge, C. Meinel, and H. Yang, “Join the high accuracy club on imagenet with a binary neural network ticket,” arXiv preprint arXiv:2211.12933, 2022.

[67] S. Xu, J. Zhao, J. Lu, B. Zhang, S. Han, and D. Doermann, “Layerwise searching for 1-bit detectors,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2021, pp. 5682–5691.

[68] J. Zhao, S. Xu, R. Wang, B. Zhang, G. Guo, D. Doermann, and D. Sun, “Data-adaptive binary neural networks for efficient object detection and recognition,” Pattern Recognit. Lett., vol. 153, pp. 239–245, 2022.

[69] X. Jiang, N. Wang, J. Xin, K. Li, X. Yang, and X. Gao, “Training binary neural network without batch normalization for image superresolution,” in Proc. AAAI, 2021, pp. 1700–1707.

[70] J. Xin, N. Wang, X. Jiang, J. Li, and X. Gao, “Advanced binary neural network for single image super resolution,” Int. J. Comput. Vis., vol. 131, no. 7, pp. 1808–1824, 2023.

[71] J. Xin, N. Wang, X. Jiang, J. Li, X. Wang, and X. Gao, “Rectified binary network for single-image super-resolution,” IEEE Trans. Neural Netw. Learn. Syst., 2024.

[72] T. Gao, Y. Zhang, Z. Zhang, H. Liu, K. Yin, C. Xu, and H. Kong, “Bhvit: Binarized hybrid vision transformer,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2025, pp. 3563–3572.

[73] H. Qin, Y. Zhang, Y. Ding, X. Liu, M. Danelljan, F. Yu et al., “Quantsr: accurate low-bit quantization for efficient image super-resolution,” Proc. Adv. Neural Inform. Process. Syst., vol. 36, 2024.

[74] C. Hong and K. M. Lee, “Adabm: On-the-fly adaptive bit mapping for image super-resolution,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 2641–2650.

[75] M. Courbariaux, I. Hubara, D. Soudry, R. El-Yaniv, and Y. Bengio, “Binarized neural networks: Training deep neural networks with weights and activations constrained to +1 or -1,” arXiv preprint arXiv:1602.02830, 2016.

[76] X. Wang, K. C. Chan, K. Yu, C. Dong, and C. Change Loy, “Edvr: Video restoration with enhanced deformable convolutional networks,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog. Worksh., 2019.

[77] W. Yu, P. Zhou, S. Yan, and X. Wang, “Inceptionnext: When inception meets convnext,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 5672–5683.

[78] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Proc. Int. Conf. Med. Image Comput. Comput. Assist. Interv., 2015, pp. 234–241.

[79] P. Charbonnier, L. Blanc-Feraud, G. Aubert, and M. Barlaud, “Two deterministic half-quadratic regularization algorithms for computed imaging,” in Proc. IEEE Int. Conf. Image Process., vol. 2, 1994, pp. 168–172.

[80] X. Ding, X. Zhang, J. Han, and G. Ding, “Scaling up your kernels to 31x31: Revisiting large kernel design in cnns,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2022, pp. 11 963–11 975.

[81] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, “A convnet for the 2020s,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2022, pp. 11 976–11 986.

[82] N. Ma, X. Zhang, H.-T. Zheng, and J. Sun, “Shufflenet v2: Practical guidelines for efficient cnn architecture design,” in Proc. Eur. Conf. Comput. Vis., 2018, pp. 116–131.

[83] Z. Xu and R. C. Cheung, “Accurate and compact convolutional neural networks with trained binarization,” in Proc. Brit. Mach. Vis. Conf., 2019.

[84] Y. Cai, Y. Zheng, J. Lin, X. Yuan, Y. Zhang, and H. Wang, “Binarized spectral compressive imaging,” Proc. Adv. Neural Inform. Process. Syst., vol. 36, 2024.

[85] Y. Zhang, H. Qin, Z. Zhao, X. Liu, M. Danelljan, and F. Yu, “Flexible residual binarization for image super-resolution,” in Proc. Int. Conf. Mach. Learn., 2024.

[86] G. Zhang, Y. Zhang, X. Yuan, and Y. Fu, “Binarized low-light raw video enhancement,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 25 753–25 762.

[87] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.

[88] I. Loshchilov and F. Hutter, “Sgdr: Stochastic gradient descent with warm restarts,” arXiv preprint arXiv:1608.03983, 2016.

[89] R. Soundararajan and A. C. Bovik, “Video quality assessment by reduced reference spatio-temporal entropic differencing,” IEEE Trans. Circuits Syst. Video Technol., vol. 23, no. 4, pp. 684–694, 2012.

[90] C.-H. Liang, Y.-A. Chen, Y.-C. Liu, and W. H. Hsu, “Raw image deblurring,” IEEE Trans. Multimedia, vol. 24, pp. 61–72, 2020.

[91] S.-J. Cho, S.-W. Ji, J.-P. Hong, S.-W. Jung, and S.-J. Ko, “Rethinking coarse-to-fine approach in single image deblurring,” in Proc. Int. Conf. Comput. Vis., 2021, pp. 4641–4650.

[92] Z. Zhong, Y. Gao, Y. Zheng, B. Zheng, and I. Sato, “Real-world video deblurring: A benchmark dataset and an efficient recurrent neural network,” Int. J. Comput. Vis., vol. 131, no. 1, pp. 284–301, 2023.

[93] M. Liu, Y. Cui, W. Ren, J. Zhou, and A. C. Knoll, “Liednet: A lightweight network for low-light enhancement and deblurring,” IEEE Trans. Circuits Syst. Video Technol., 2025.

[94] Y. Zhang and A. Yao, “Realviformer: Investigating attention for realworld video super-resolution,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 412–428.

[95] Y. Tian, Y. Zhang, Y. Fu, and C. Xu, “Tdan: Temporally-deformable alignment network for video super-resolution,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2020, pp. 3360–3369.

[96] M. Cao, L. Wang, H. Wang, and X. Yuan, “A simple low-bit quantization framework for video snapshot compressive imaging,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 112–129.

[97] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su et al., “Grounding dino: Marrying dino with grounded pre-training for open-set object detection,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 38–55.

[98] A. Gupta, P. Dollar, and R. Girshick, “Lvis: A dataset for large vocabulary instance segmentation,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2019, pp. 5356–5364.

[99] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in Proc. Eur. Conf. Comput. Vis., 2020, pp. 213–229.

[100] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth anything v2,” in Proc. Adv. Neural Inform. Process. Syst., 2024, pp. 21 875–21 911.

![](images/e9bfc5696e8447acd0268674d79d1774f11ea7f9b8c5d85cc463429160c9b336.jpg)

Ying Fu received a B.S. degree in electronic engineering from Xidian University, Xi’an, China, in 2009, an M.S. degree in automation from Tsinghua University, Beijing, China, in 2012, and a Ph.D. degree in information science and technology from the University of Tokyo, Japan, in 2015. She is currently a Professor at the School of Computer Science and Technology, Beijing Institute of Technology. Her research interests include computer vision, image and video processing, and computational photography.

![](images/9e5a20c0a03bdaff8ba8c9e0825769d1f4a874969754eda371ff3f8946db1329.jpg)

![](images/fc278e96406340e1d84b0d94439eddf35fb193789a0beb82a2b34a445e6b72cf.jpg)  
Tianyu Zhu received a B.S. degree in computer science and technology from Beijing Institute of Technology, Beijing, China, in 2024. He is currently a M.S. candidate at the School of Computer Science and Technology, Beijing Institute of Technology, Beijing, China. His research interests include artificial intelligence and image processing.

Hesong Li received a B.S. degree in mathematics and applied mathematics from Dalian Maritime University, Dalian, China, in 2021. He is currently a Ph.D. candidate at the School of Computer Science and Technology, Beijing Institute of Technology, Beijing, China. His research interests include artificial intelligence and image processing.

![](images/1da3874e7436afc03c82793cf2ba51f221b9c1ebad636ce89e359fb2a6dbbea1.jpg)

![](images/e24170b1ba0f9979c48e0ccbceffd07b48b3bb8e0edf491c16bd362260526b0d.jpg)

Gengchen Zhang Gengchen Zhang received a B.S. degree in computer science from Beijing Institute of Technology, Beijing, China, in 2022, an M.S. degree in computer science from Beijing Institute of Technology, Beijing, China, in 2025. He is currently working at Chinese Aeronautical Establishment. His research interests include computational photography, image processing, and deep-learning.

Xin Yuan (SM’16) received the BEng and MEng degrees from Xidian University, in 2007 and 2009, respectively, and the PhD from the Hong Kong Polytechnic University, in 2012. He is currently an Associate Professor at Westlake University. He was a video analysis and coding lead researcher at Bell Labs, Murray Hill, NJ, USA from 2015 to 2021. Prior to this, he was a Post-Doctoral Associate in the Department of Electrical and Computer Engineering, Duke University from 2012 to 2015. His research interests are computational imaging and machine learning. He has been the Associate Editor of Pattern Recognition (2019-), International Journal ofPattern Recognition and Artificial Intelligence (2020-) and Chinese Optics Letters (2021-). He led the special issue of ”Deep Learning for High Dimensional Sensing” in the IEEE Journal of Selected Topics in Signal Processing in 2022.

![](images/d02758b8b445a6e11a6cc791756016af4a2157af593c68de7e1548c7716a720c.jpg)

Yulun Zhang received a B.E. degree from the School of Electronic Engineering, Xidian University, China, in 2013, an M.E. degree from the Department of Automation, Tsinghua University, China, in 2017, and a Ph.D. degree from the Department of ECE, Northeastern University, USA, in 2021. He is currently an associate professor at Shanghai Jiao Tong University, Shanghai, China. He was a postdoctoral researcher at Computer Vision Lab, ETH Zurich, Switzerland. He also worked as a research¨ fellow at Harvard University, USA. His research

interests include image/video restoration and synthesis, biomedical image analysis, model compression, multimodal computing, large language model, and computational imaging. He is/was an Area Chair for CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR, IJCAI, ACM MM, and a Senior Program Committee (SPC) member for IJCAI and AAAI.