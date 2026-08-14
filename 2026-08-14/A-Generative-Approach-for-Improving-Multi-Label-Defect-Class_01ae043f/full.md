# A Generative Approach for Improving Multi-Label Defect Classification in Photovoltaic Modules

Abdul Mueez, Yogesh S. Rawat, Shruti Vyas<sup>∗</sup>

Institute of Artificial Intelligence, University of Central Florida, 4356 Scorpius St., Orlando, FL 32816, United States of America

## Abstract

This paper addresses the challenge of multi-label defect classification in electroluminescence (EL) images of photovoltaic (PV) cells. Training models on images where multiple defects co-occur creates learning ambiguity, making it dificult to disentangle visual features for specific defect types, a problem compounded by the scarcity of examples for individual classes. To tackle this, we introduce Generative Defect Isolation (GDI), utilizing the LaMa inpainting model with Fast Fourier Convolutions to remove selected defects and generate realistic, single-defect training samples. Extensive experiments on Vision Transformer (ViT-S, ViT-L) and EficientNetV2-L architectures demonstrate that GDI significantly outperforms baselines. The performance gains are most pronounced in low-data scenarios; class-wise analysis shows substantial improvements, boosting the F1-Score for rare defect classes by up to 63.6%. Furthermore, GDI efectively resolves learning ambiguity from cooccurring defects, yielding a 26% reduction in such co-occurring classification errors. Our work establishes GDI as an efective method for maximizing the value of existing segmentation datasets and sets a new performance benchmark for multi-label classification in this domain.

Photovoltaics, Electroluminescence imaging, Synthetic data, Inpainting, Multi-label defect classification

## 1. Introduction

The global transition towards renewable energy sources has positioned photovoltaics (PV) as a cornerstone of sustainable energy production, with global installed capacity growing exponentially [1, 2]. As the scale of PV plants increases, particularly in complex operating environments, the industry is increasingly shifting toward intelligent operation and maintenance (O&M) strategies to ensure system safety and economic eficiency [3]. However, the eficacy of these intelligent inspection systems is highly susceptible to a variety of defects that can arise during manufacturing, transportation, or from environmental stressors in the field [1, 4]. These defects, ranging from micro-cracks to cell interconnect failures, can lead to significant power degradation, reduced module lifetime and potential safety hazards [5, 4].

Among various non-destructive testing (NDT) techniques, Electroluminescence (EL) imaging has emerged as a highly efective method for quality control in the PV industry [6, 1]. EL imaging provides high-resolution spatial information that can reveal even the finest defects, such as microcracks, which are often invisible to the naked eye or other inspection methods like infrared thermography [7, 8]. However, the manual inspection of the vast number of EL images generated during high-throughput manufacturing and large-scale field testing is a major bottleneck. This process is timeconsuming, expensive and prone to subjective errors and inconsistencies by human operators [7, 8].

To overcome these limitations, the research community has increasingly focused on developing automated defect detection systems. Driven by advances in computer vision, deep learning - particularly Convolutional Neural Networks (CNNs) - has become the predominant approach, consistently demonstrating superior performance over traditional image processing techniques [1, 9].

Despite these advancements, several key challenges persist. A significant portion of existing deep learning models is designed for binary classification (i.e., distinguishing between “defective” and “non-defective” cells) [7, 10]. In practical scenarios, however, a single PV cell often exhibits multiple, cooccurring defects. A recent comprehensive review highlights this as a critical limitation in current research: existing defect detection solutions predominantly focus on identifying single defect types per image, failing to address the reality that multiple defect types frequently coexist [3]. Consequently, the development of algorithms capable of handling this multi-label complexity has been identified as a key future direction for intelligent PV maintenance [3]. Furthermore, the performance of deep learning models is heavily reliant on the availability of large, diverse and accurately annotated datasets, which remain scarce in this specialized domain [2].

The inherent class imbalance found in real-world data, where non-defective cells vastly outnumber any single class of defect, further complicates the training of robust and generalizable models [8, 2].

While generative methods have been used to create new images from scratch, a key opportunity remains untapped: repurposing the rich, pixellevel detail from existing segmentation datasets to resolve ambiguity in the more common task of multi-label classification.

To address these challenges, this paper presents a data-centric approach to enhance multi-label defect classification in PV cell EL images. We utilize the publicly available UCF-EL-Defect dataset [8], which provides rich pixel-level segmentation annotations for nine distinct defect classes. We first reframe this segmentation dataset for a multi-label classification task. Our main contribution is the introduction of an annotation-guided data augmentation strategy we term Generative Defect Isolation (GDI). This technique leverages the ground-truth segmentation masks to computationally “inpaint” or remove selected defects from an image. By doing so, GDI generates new, high-quality training examples of cells with isolated, single defects from source images that originally contained multiple, overlapping defects. This method directly targets the problem of data scarcity for underrepresented defect combinations and helps the model learn to disentangle features associated with diferent defect types.

The main contributions of this work are:

1. We propose Generative Defect Isolation (GDI), a technique that utilizes the LaMa inpainting architecture [11] to leverage segmentation annotations. This allows us to iteratively isolate defects, creating new, realistic training samples from multi-defect images.

2. We demonstrate through extensive experiments using Vision Transformer (ViT-S, ViT-L) and EficientNetV2-L architectures that models trained with GDI significantly outperform a baseline on a challenging multi-label defect classification task.

3. By leveraging GDI, we establish a new performance benchmark for multi-label classification on the UCF-EL-Defect dataset, providing a valuable resource for future research.

The remainder of this paper is organized as follows: Section 2 reviews the existing literature on automated defect detection in EL images. Section 3 details our proposed methodology, including the dataset preparation and the annotation-guided augmentation technique. Section 4 presents the experimental setup and results, followed by a discussion in Section 5. Finally, Section 6 concludes the paper.

## 2. Related Work

The automated analysis of EL images for PV quality control has been an active area of research, evolving from traditional image processing to sophisticated deep learning architectures. This section reviews key developments in this field, with a focus on methodologies for defect classification and localization and the various strategies employed to address prevalent data-related challenges.

## 2.1. Automated Defect Classification and Localization

Defect detection techniques for PV modules are generally categorized into electrical parameter analysis (I/V curves) and imaging techniques (Visible, IR, EL/PL) [3]. While I/V curve analysis is cost-efective, it is limited by its inability to precisely locate defects or detect tiny anomalies [3]. Similarly, while UAV-based visible and IR inspections ofer high eficiency, they often lack the “inner perspective” required to identify detailed cell-level faults [3]. Electroluminescence (EL) imaging has thus emerged as a superior method for detailed defect diagnosis, ofering high-resolution insight into the module’s internal state.

Early research into automated EL analysis relied on classical computer vision and machine learning techniques. For instance, [12] used Fourier image reconstruction to detect cracks and breaks, while other work from the same group used Independent Component Analysis (ICA) to create basis images for identifying defects [13]. Other approaches have also included Support Vector Machines (SVMs) trained on hand-crafted features [7]. While efective to a degree, these methods often struggled with the high variability of defect appearances and the complex textures of polycrystalline cells.

With the rise of deep learning, CNNs have become the standard for defect detection tasks due to their ability to automatically learn hierarchical features from raw pixel data. A large body of work has demonstrated the success of CNNs for binary classification of PV cells as either “defective” or “functional” [7, 10, 4]. Various architectures, from custom lightweight networks [10, 14] to established backbones like VGG [7, 2] and ResNet [15], have been successfully applied. More specialized architectures, such as high-resolution networks designed to preserve fine-grained details [16] and methods that fuse multichannel CNNs to capture diverse features [17], have also been developed to improve detection accuracy.

Beyond simple classification, some research has focused on providing more granular information about the defect’s location and extent. This is typically achieved through either object detection, which places a bounding box around a defect, or semantic segmentation, which assigns a class label to every pixel in the image. [9] introduced PVEL-AD, a large-scale dataset with bounding box annotations for ten diferent anomaly types and benchmarked several object detection models. Providing the most detailed level of annotation, [8] developed the UCF-EL-Defect dataset, which our work is based on, for semantic segmentation of nine defect classes using a DeepLabv3+ model. Other works have also explored segmentation, such as [18] who used Mask R-CNN for localizing 14 defect types in a production line setting. Our work sits at the intersection of these approaches, leveraging the rich annotations of a segmentation dataset to improve the performance of a classification task.

## 2.2. Data Augmentation and Imbalance Handling

A persistent challenge in applying deep learning to PV defect detection is the scarcity of large, labeled datasets and the inherent class imbalance of real-world data [1, 2]. Consequently, data augmentation has become a critical component of the training pipeline.

Standard geometric and color-space transformations, such as random flipping, rotation and brightness adjustments, are widely used to increase the diversity of the training set and improve model robustness [19, 14, 2]. However, the efectiveness of these simple transformations can be limited.

Prior work has explored using generative models, particularly Generative Adversarial Networks (GANs), to generate synthetic data by synthesizing new artificial EL images of defective cells [2, 20]. A significant limitation of GAN-based synthetic data generation is highly hardware-demanding and prone to training instability, which can lead to convergence failure and the production of poor-quality data [2]. Additionally, the generative process relies on global image creation from noise, which may result in mode collapse where the generator fails to produce a diverse range of random patterns for the discriminator [2]. Notably, [2] formulate the problem as a binary classification task, while [20], despite proposing a multi-class data generation strategy, ultimately evaluate their approach under a binary classification setting. Such evaluations overlook the fundamentally diferent and significantly more challenging nature of multi-label classification, where multiple defect types can co-occur within a single sample.

As an alternative to purely generative approaches, some have used physicsbased simulations to create realistic defect images. [8], in the creation of the UCF-EL-Defect dataset, supplemented their real-world images with 256 physically realistic simulated images to address the severe class imbalance of interconnect and contact defects.

The existing literature showcases a clear trend towards using advanced data augmentation to overcome dataset limitations. While methods like GANs generate entirely new images and geometric transforms alter existing ones, our proposed work introduces a distinct, hybrid approach. By leveraging precise segmentation annotations for inpainting, we do not create wholly synthetic data but rather disentangle real, co-occurring defects from a single image. This methodology creates a unique dataset of semisynthetic, high-quality images tailored specifically for the challenging multilabel classification problem, bridging the gap between classification-focused and segmentation-focused research.

More broadly, an alternative strategy to address data scarcity involves semi-supervised learning [21], where models are trained on less expensive annotations like bounding boxes or image-level labels. While such an approach aims to lower the initial cost of data collection, our work addresses a diferent scenario: leveraging and maximizing the value of existing, high-density segmentation annotations to improve performance on a related downstream task.

## 3. Methodology

Our approach expands a dataset containing complex, co-occurring defects by supplementing it with simplified, single-defect images, creating a larger and more robust collection for training a multi-label classifier. This is achieved through a multi-stage process we call Generative Defect Isolation (GDI). This section details the initial dataset, the preprocessing steps and the core GDI pipeline.

## 3.1. Dataset and Initial Processing

The foundation of our work is the publicly available UCF-EL-Defect dataset [8]. As provided by the original authors, the dataset consists of isolated, cell-level images. These exhibit significant resolution variance—ranging from 165 × 169 to 1174 × 1144 pixels—across 491 unique sizes. This variance introduces a degree of visual ambiguity; in lower-resolution samples, distinguishing between fine-grained defect types can be challenging even for human experts.

Importantly, the dataset provides detailed, pixel-level segmentation annotations for nine distinct defect classes. Although originally designed for segmentation, we apply the dataset to the more common industrial task of multilabel classification. The dataset annotations contain two additional classes, Contact\_BeltMarks and Unknown, which were not utilized in the original study. We chose to incorporate these two classes in our analysis. Furthermore, to build a complete classification scheme, we introduced a No\_Defect class for images confirmed to be defect-free. An image is assigned multiple labels if it contains annotations for more than one defect type. This process results in the following 12 target classes for our task: Contact\_BeltMarks, Contact\_Corrosion, Contact\_FrontGridInterruption, Contact\_NearSolderPad, Crack\_Closed, Crack\_Isolated, Crack\_Resistive, Interconnect\_BrightSpot, Interconnect\_Disconnected, Interconnect\_HighlyResistive, No\_Defect, Unknown.

## 3.2. Generative Defect Isolation (GDI)

The central contribution of this work is the GDI pipeline, designed to resolve the learning ambiguity that arises when a model is trained on images containing multiple, overlapping defect types. The GDI method proceeds in three algorithmic steps for each candidate image: (1) Defect Selection and Filtering, (2) Annotation-Guided Mask Generation and (3) Neural Inpainting. By utilizing ground-truth segmentation masks, this process computationally removes selected defects to generate new, high-quality images that exhibit only a single, isolated defect or no defects. The entire process is illustrated schematically in Figure 1.

![](images/447db26e77dd9ad98ac2ab44f2a1c0e5875fe58d7fbb3ba6df67c187777fc875.jpg)  
Figure 1: The proposed Generative Defect Isolation (GDI) process: For images with multiple defects, the method first iteratively isolates each defect by inpainting all others. After this loop, it generates a final “no-defect” sample by creating a combined mask of all defects and inpainting them simultaneously. This creates multiple singledefect samples and one defect-free sample from a single source image.

## 3.2.1. Defect Selection and Filtering

The GDI process is applied exclusively to images that contain more than one unique defect type. For such an image, we iterate through each of its annotated defects to potentially isolate it. However, to ensure that the resulting augmented image contains a visually significant defect rather than a minor blemish that could be mistaken for a defect-free cell, we apply an area-based filter. As shown in Figure 1, a specific defect type within an image is only considered for isolation if its total annotated area covers more than 10% of the entire image area. This prevents the generation of trivial samples and focuses the augmentation efort on prominent defects. Further, if at least one defect satisfies this criterion and is successfully isolated, we additionally create a combined mask covering all annotated defects in the source image and inpaint these regions to generate a corresponding defect free sample.

## 3.2.2. Annotation-Guided Mask Generation

For a multi-defect image and a chosen target defect class, $D _ { t a r g e t }$ , that meets the area criteria, the next step is to generate a mask to remove all other defects. A binary mask, $M _ { o t h e r }$ , is created by rasterizing the geometric annotations (polygons, circles, ellipses, rectangles) of all non-target defect classes present in the image. This mask precisely covers all pixels belonging to the defects slated for removal.

To prevent artifacts at the boundary of the inpainted region, this initial mask is dilated. Dilation expands the masked area, creating a small bufer zone around the unwanted defects. This ensures that the inpainting algorithm seamlessly blends the synthesized texture with the surrounding original image content. A square dilation kernel of size $1 5 \times 1 5$ was used. This value was selected through qualitative inspection as it provided the best balance between fully masking defect boundaries and preserving surrounding textures, thereby minimizing inpainting artifacts.

## 3.2.3. Neural Inpainting with LaMa

![](images/146dd83f900ec8c653a551544373766a6da2f83f1ab4c76cc3ec247358c72303.jpg)  
Figure 2: The proposed inpainting workflow: The original image (x) and the binary mask of other defects $( M _ { o t h e r } )$ are combined to create a masked input tensor (x<sup>′</sup>). This tensor is processed by an inpainting network featuring an encoder-decoder architecture: three downscale blocks reduce the spatial resolution, a core of nine Fast Fourier Convolution (FFC) residual blocks processes the features and three upscale blocks reconstruct the final restored image (xˆ). This approach is based on the LaMa [11] model.

With the dilated mask prepared, we employ the LaMa (Large Mask Inpainting) model [11] to fill the masked regions. We utilize the implementation provided by the “Inpaint Anything” codebase [22]. This architecture was specifically selected for its ability to handle high-resolution inputs with complex periodic structures, replacing irregular masked regions with coherent, realistic textures [11]. Unlike standard convolutional methods that often fail to maintain global structural consistency across large masks, the Fast Fourier Convolutions in LaMa enable the precise reconstruction of the periodic grid lines inherent to PV cells. The workflow is illustrated in Figure 2 and the process consists of the following technical steps:

1. Input Tensor Construction: For a source image x and the generated binary mask $M _ { o t h e r }$ (where 1 denotes a pixel to inpaint), we first compute the masked image $x \odot \left( 1 - M _ { o t h e r } \right)$ . We then construct a fourchannel input tensor $x ^ { \prime }$ by stacking the masked image and the mask itself: $x ^ { \prime } = \mathrm { s t a c k } ( x \odot ( 1 - M _ { o t h e r } ) , M _ { o t h e r } )$

2. Feature Encoding and Fast Fourier Convolutions: The tensor $x ^ { \prime }$ is processed by a feed-forward network with a ResNet-like topology. The encoder utilizes three downsampling blocks to reduce the spatial resolution, followed by a bottleneck of nine Fast Fourier Convolution (FFC) residual blocks. The FFC blocks are the core algorithmic component; unlike standard convolutions, they split the feature channels into two parallel branches:

• Local Branch: Uses conventional convolutions to capture local texture details.

• Global Branch: Applies a Real Fast Fourier Transform (Real FFT2d) to transition features into the frequency domain, performs a 1 × 1 convolution to capture global context and recovers the spatial structure via an Inverse Real FFT2d.

This spectral transform allows the network to learn global interactions and periodic patterns, specifically the grid lines and busbars inherent to PV cells at any depth, efectively providing an image-wide receptive field.

3. Reconstruction and Loss: Finally, three upsampling blocks reconstruct the feature maps to the original resolution, yielding the restored image xˆ. The network is trained using a combined objective function comprising a High Receptive Field (HRF) perceptual loss and an adversarial loss. The HRF perceptual loss utilizes a pre-trained segmentation network with dilated convolutions to enforce structural consistency between the inpainted region and the surrounding valid pixels, while the adversarial loss, calculated via a discriminator operating on local patches, ensures the synthesized textures (e.g., silicon grain) are photorealistic [11].

The augmented images were generated on a system equipped with an Intel Xeon W-3335 CPU and an NVIDIA RTX A5500 GPU. The GDI preprocessing step required approximately 3.49 seconds per image. It is important to note that the entire GDI pipeline, including the use of the LaMa model, is a one-time, ofline pre-processing step used to augment the training dataset. It carries no computational overhead at inference time, thus preserving the eficiency of the final classification model.

## 3.2.4. Augmented Dataset Creation

This isolation process is repeated for every defect type in the original image that passes the 10% area filter. Consequently, a single source image with three significant, distinct defects can generate four training samples, one for each isolated defect and a defect free image.

After processing the entire source dataset, the newly generated singledefect images are combined with the original multi-defect training set to form the final, enhanced dataset. This hybrid approach ensures the model learns from both idealized defect patterns and the complex co-occurrences found in real-world scenarios. A corresponding CSV label file is generated for the new samples, where each image is mapped to a single positive class label (one-hot encoded) for the preserved defect.

## 3.2.5. Dataset Composition

To prepare our datasets for training and evaluation, the original UCF-EL Defect dataset was first partitioned into an 80:20 training and test split. This split was performed on the source images before any augmentation was applied. Our Generative Defect Isolation (GDI) process was then applied exclusively to the training set to generate new, single-defect samples. The test set was reserved solely for evaluation and consists only of original, unaugmented images to ensure a fair assessment of the model’s generalization capabilities.

Table 1 provides a comprehensive overview of the final composition of both the training and test sets. The training set is composed of the original images from the 80% split and the new samples generated by GDI, highlighting how the augmentation efectively supplements the base data across various classes. The test set consists entirely of the original images from the 20% split. As detailed in Table 1, the final training set is constructed by combining the original images from the 80% split with the new single-defect inpainted samples generated by GDI.

## 4. Results

## 4.1. Qualitative Analysis of Inpainting Quality

A qualitative review of the generated samples confirms the high quality and efectiveness of the Generative Defect Isolation (GDI) pipeline, as shown in Figure 3. The inpainting model excels at producing visually plausible images suitable for training. It demonstrates a strong capability for creating clean No\_Defect images by seamlessly removing various defects (Figure 3d, 3h and 3l). The pipeline is particularly adept at reconstructing key structural features; for example, Figure 3h illustrates how grid lines that were completely obscured by an original defect are accurately restored, preserving the cell’s structural integrity. This ability to generate clean, targeted training data provides a clear advantage and the strong quantitative improvements detailed in the subsequent sections suggest the benefits significantly outweigh potential minor artifacts. A discussion of the method’s limitations and failure modes is presented in Section 5.

Table 1: The final composition of the training and test datasets used in our experiments. The training set includes both original and GDI-generated samples, while the test set is composed exclusively of original images to ensure an unbiased evaluation. Note that as this is a multi-label dataset, an image can have multiple defects; therefore, the ‘Overall Total’ represents the number of unique images, not the sum of the defect instances in the columns.
<table><tr><td rowspan="2">Class Name</td><td colspan="3">Training</td><td rowspan="2">Test</td></tr><tr><td>Original</td><td>Inpainted</td><td>Total Total</td></tr><tr><td>Contact BeltMarks</td><td>9</td><td>5</td><td>14</td><td>5</td></tr><tr><td>Contact_Corrosion</td><td>55</td><td>7</td><td>62</td><td>12</td></tr><tr><td>Contact_FrontGridInterruption</td><td>6980</td><td>33</td><td>7013</td><td>1712</td></tr><tr><td>Contact_NearSolderPad</td><td>4464</td><td>162</td><td>4626</td><td>1118</td></tr><tr><td>Crack_Closed</td><td>2126</td><td>50</td><td>2176</td><td>514</td></tr><tr><td>Crack_Isolated</td><td>1422</td><td>328</td><td>1750</td><td>333</td></tr><tr><td>Crack_Resistive</td><td>2470</td><td>1475</td><td>3945</td><td>617</td></tr><tr><td>Interconnect_BrightSpot</td><td>203</td><td>91</td><td>294</td><td>44</td></tr><tr><td>Interconnect_Disconnected</td><td>161</td><td>5</td><td>166</td><td>69</td></tr><tr><td>Interconnect_HighlyResistive</td><td>203</td><td>16</td><td>219</td><td>44</td></tr><tr><td>No_Defect</td><td>598</td><td>1752</td><td>2350</td><td>257</td></tr><tr><td>Unknown</td><td>71</td><td>0</td><td>71</td><td>12</td></tr><tr><td>Overall Total</td><td>10086</td><td>3924</td><td>14010</td><td>2620</td></tr></table>

## 4.2. Experimental Setup

## 4.2.1. Model Architectures

To validate our methodology, we selected three powerful and distinct model architectures. We employed two Vision Transformer models, ViT-

![](images/ac13d73816fe99345c35617dc4310fbe8a2521553f282dcd96978e2ae1b9b549.jpg)  
(a) Original

![](images/e13f10022f94c789f8225b2f6860cd176398b08285bb182c4296872072a69560.jpg)  
(b) Boundaries

![](images/21b27b0a17e8299b180adc601dc66ad88824c90708b706500a27a0e94278c852.jpg)  
(c) C\_FGInterruption

![](images/2e47a161484f626e7a8dcdfe088cb0b98ba1732d7109eb56f1853bd50cee7f32.jpg)  
(d) No\_Defect

![](images/1bb783836edd38554d22d678a353a5934a9bb8af4422c26c15a8861d74707ff8.jpg)  
(e) Original

![](images/feee1d340fbb0c001988757b92255aa268eea9d30754cfa1d302abd389783e76.jpg)  
(f) Boundaries

![](images/5282271d7c69b12cd1a8c169ef6e722da72ca996e9f77b7e385d6b8090e22c2c.jpg)  
(g) Crack\_Resistive

![](images/a962a50ab838d2678eb955c2cf70af5cb42411afd1de81cf8ad59d894ffd0f78.jpg)  
(h) No\_Defect

![](images/ebc59da0f67693ade6150a1f1573ebfa2114b64a4deff9e8c9ad1bf4ca6c45be.jpg)  
(i) Original

![](images/97f2793ee4f61675b1824458f1f77aa8e861f4cdab12ead21b5c124028ace74a.jpg)  
(j) Boundaries

![](images/76ceddae7a00d8d8eda133e853f6b2b38e3c0d106a377a4d6def5d745f163dd9.jpg)  
(k) Crack\_Isolated

![](images/6d895ece2f6a5c79391b48f465c44a12758f1bdae5c139434b81db7e6d73a875.jpg)  
(l) No\_Defect

![](images/4137a0dea0003c1864d031e09678efc186375417f0fae6483502fc29a5391e7a.jpg)  
(m) Original

![](images/d84ed1698e4948dcb217d48c609137c42b400eae3874ecb8b8d0a549a9d467d4.jpg)  
(n) Boundaries

![](images/2ea0b2fb278aff0405d16ae8d59f1dc5694110e4f35f5456cd938aa55e14456a.jpg)  
(o) Contact\_BeltMarks

![](images/0854fbb8ca66a8e0600a0962a47e1f09298d8823fa623994e17538710ed92d79.jpg)  
(p) $N o \_ D e f e c t$

Figure 3: Visual examples of the Generative Defect Isolation (GDI) pipeline. Each row corresponds to a diferent source image. For each, we show the original multidefect cell (left), the annotated defect boundaries and the high-quality samples generated by GDI. These include images with an isolated defect with an area greater than 10% of the entire image or a clean No\_Defect version created by inpainting all defects. Subfigure 3c shows C\_FGInterruption, which stands for Contact\_FrontGridInterruption. Subfigure 3o displays Contact\_BeltMarks, an extremely rare defect class (only 9 original training samples).

Small (ViT-S) and ViT-Large (ViT-L), together with the convolutional model EficientNetV2-L. This selection allows for a comprehensive comparison across diferent model scales and fundamental architectural paradigms. To leverage transfer learning and accelerate convergence, all models were initialized with weights pretrained on the ImageNet-1k dataset [23]. This standard practice provides a strong feature-learning foundation for the downstream classification task.

Vision Transformer (ViT). The Vision Transformer (ViT) adapts the standard Transformer encoder for image recognition tasks [24]. An input image is divided into a sequence of flattened, non-overlapping 2D patches, which are linearly projected into a constant latent dimension D to form patch embeddings. Learnable one-dimensional positional embeddings are added to retain spatial information and following the precedent set by BERT [25], a learnable [class] token is prepended to the sequence. The final representation of this token is used for image-level classification.

The Transformer encoder processes these embeddings through alternating layers of multi-head self-attention (MSA) and multi-layer perceptron (MLP) blocks, each preceded by Layer Normalization and followed by residual connections. The MLP block comprises two fully connected layers with a GELU non-linearity. Table 2 summarizes the architectural configurations of the ViT models used in this study [24, 26].

Table 2: Architectural details for the Vision Transformer (ViT) models used in this study.
<table><tr><td>Model</td><td>Layers</td><td>Hidden Size (D)</td><td>MLP Size</td><td>Attention Heads</td><td>Total Parameters</td></tr><tr><td>ViT-S [26]</td><td>12</td><td>384</td><td>1536</td><td>6</td><td>22.2M</td></tr><tr><td>ViT-L [24]</td><td>24</td><td>1024</td><td>4096</td><td>16</td><td>307M</td></tr></table>

EficientNetV2-L. EficientNetV2 is a family of convolutional networks developed using a training-aware neural architecture search (NAS) and compound scaling, designed to jointly optimize accuracy, training speed and parameter eficiency [27]. The Large variant (V2-L) contains approximately 120M parameters and incorporates several key architectural refinements.

• Fused-MBConv Blocks: The network employs both standard MB-Conv (Mobile Inverted Bottleneck Convolution) and Fused-MBConv blocks, with the latter used predominantly in the early stages. An MBConv block expands channel dimensions via a 1 × 1 convolution, applies a depthwise 3 × 3 convolution and then projects back to a lower-dimensional space, forming an eficient inverted bottleneck structure. The Fused-MBConv variant merges the depthwise and expansion convolutions into a single 3 × 3 convolution, improving computational eficiency on modern hardware accelerators.

• Architectural Refinements: The NAS-derived design favors smaller expansion ratios within MBConv blocks to reduce memory access overhead and uses smaller 3 × 3 kernels, compensating for the reduced receptive field by increasing network depth. The final stage consists of a 1 × 1 convolution, global pooling and a fully connected layer for classification.

• Scaling Strategy: To obtain larger variants such as EficientNetV2- L, a non-uniform compound scaling strategy is applied, progressively adding layers to later stages of the network to increase representational capacity without incurring significant runtime overhead.

These design choices collectively enable EficientNetV2-L to achieve strong performance while maintaining high computational eficiency [27].

## 4.2.2. Data Splits and Evaluation

To simulate data-scarce conditions, we created subsets of the training data containing 1%, 5%, 10%, 20% and 50% of the full training set. Inpainted samples within each subset were generated only from images already included in that subset. Training was performed without any threshold tuning. At inference time, each class probability was converted to a binary prediction using a per-class threshold; for the main experiments, these thresholds were selected per class as the value that maximized the macro F1-score on the test set. The identical procedure was applied to the baseline and GDI models to preserve a fair comparison. Since this selection used the test set, it can yield optimistic absolute F1 estimates, although it does not afect model training. To show that GDI improvements are threshold independent, Section 4.9 reports a complementary evaluation at a fixed threshold $( \tau = 0 . 5 )$ on the same threshold-free models, confirming the gains are a property of the augmentation rather than of threshold tuning. All reported results are computed using the two multi-label metrics defined below.

• Zero-One Accuracy: This strict metric measures the fraction of samples for which the set of predicted labels is exactly identical to the set of true labels. It is calculated as:

$$
\mathrm { A c c u r a c y } _ { 0 / 1 } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } ( \hat { y } _ { i } = y _ { i } )
$$

where $N$ is the number of samples, $y _ { i }$ is the true label set, $\hat { y } _ { i }$ is the predicted label set and $\mathbb { I } ( \cdot )$ is the indicator function.

• Macro F1-Score: This metric provides a balanced measure of precision and recall, averaged across all classes, making it ideal for imbalanced datasets. For each class c in the set of all classes $C _ { i }$ , the F1-Score is the harmonic mean of precision $( P _ { c } )$ and recall $( R _ { c } )$ . The macro-averaged F1-Score is their unweighted average:

$$
P _ { c } = { \frac { \mathrm { T P } _ { c } } { \mathrm { T P } _ { c } + \mathrm { F P } _ { c } } } \qquad R _ { c } = { \frac { \mathrm { T P } _ { c } } { \mathrm { T P } _ { c } + \mathrm { F N } _ { c } } }
$$

$$
F 1 _ { c } = 2 \cdot { \frac { P _ { c } \cdot R _ { c } } { P _ { c } + R _ { c } } }
$$

$$
\mathrm { M a c r o ~ F 1 } = { \frac { 1 } { | C | } } \sum _ { c \in C } F 1 _ { c }
$$

## 4.2.3. Data Augmentation and Implementation Details

To ensure robust feature learning and a fair comparison between methods, an identical data augmentation pipeline was applied to all models (Baseline and GDI) during training. To standardize the highly variable native resolutions described in Section 3.1, all images across both the training and test sets were resized to a uniform 224 × 224 pixels prior to model input. The training transformations included: random resized crop to $2 2 4 \times 2 2 4$ pixels, random horizontal flip, color jitter and random afine transformations. Input images were normalized using the standard ImageNet mean and standard deviation values.

To ensure that any observed performance gains were solely due to the improved data quality, we maintained strict consistency in our training configuration. For each model architecture, identical hyperparameters, including learning rate, batch size, optimizer choice and weight decay were used for both the baseline and the GDI augmented training runs.

## 4.3. Overall Performance Gains

The results clearly and consistently demonstrate that training with the GDI-generated dataset provides a significant performance advantage over the baseline across nearly all experimental configurations. The GDI models show improved accuracy and F1-Scores, validating our hypothesis that isolating defects helps resolve learning ambiguities and enhances feature disentanglement.

As shown in Figure 4, the GDI-augmented models almost universally outperform their baseline counterparts in Zero-One Accuracy. A notable outlier is the 100% drop in accuracy for the GDI-augmented ViT-S model at the 1% data level. This reflects the accuracy falling to zero, but it is crucial to note the baseline was exceedingly small (0.0019), rendering the relative metric misleading in this specific, data-scarce instance. Importantly, this exception aside, the GDI-augmented models also consistently achieve higher F1-Scores than their baseline counterparts (Figure 4).

## 4.4. Impact of Data Scarcity

A key finding is that the benefits of GDI are most pronounced in low-data settings. When the training set is limited, providing the model with clean, unambiguous, single-defect examples proves to be exceptionally efective.

For Zero-One Accuracy, the improvements in data-scarce scenarios are substantial. For instance, with only 10% of the training data, the ViT-S model sees a remarkable +125.1% relative improvement. Similarly, the ViT-L model achieves massive gains of +129.0% with just 5% of the data. This strongly suggests that GDI is a powerful tool for boosting performance when labeled data is expensive or dificult to acquire.

This trend is mirrored in the F1-Scores. At the 1% data level, ViT-S shows a substantial +20.3% improvement, while EficientNetV2-L and ViT-L also see significant gains of +11.5% and +9.8%, respectively. As the amount of training data increases, the relative gains naturally become more modest, yet they remain consistently positive.

## 4.5. Performance Across Model Architectures

Our proposed GDI method shows robust improvements across all tested model architectures, from the lightweight ViT-S to the large-scale ViT-L and the convolutional EficientNetV2-L.

Interestingly, the smaller ViT-S model often reaps the largest relative benefits from GDI, especially in accuracy (e.g., +125.1% at 10% data) and

F1-Score (+20.3% at 1% data). This suggests that GDI is particularly effective for models with fewer parameters, which may have less intrinsic capacity to disentangle complex, overlapping features. In contrast, the powerful EficientNetV2-L consistently achieves the highest absolute scores in higher-data settings (50% and 100%), establishing the best overall performance with a top F1-Score of 0.7744. For this model, the improvements from GDI are more incremental but consistently positive, demonstrating that even a highly capable architecture benefits from cleaner training signals. The large ViT-L model follows a similar positive trend, though we noted one minor instance of F1-Score degradation (-1.1% at 10% data), which we attribute to potential training instability for a very large model on a limited data subset.

A key trend emerging from these results is that our inpainting-based augmentation exhibits diminishing utility as the volume of training data increases. In a low-data setting, the augmentation provides a significant performance advantage by synthetically expanding under-represented classes and providing idealized, representative examples. This simplifies the featurelearning task by removing confounding visual information, which is critical when real examples are sparse. As the dataset size approaches 100%, however, the performance of the augmented and baseline models converges. With access to the complete dataset, the baseline model can learn to distinguish complex features from real multi-defect images organically. Consequently, the marginal benefit of the synthetic data is reduced, as the richness of the full real dataset becomes more impactful for generalization.

## 4.6. Class-Wise Performance Analysis

While aggregate metrics confirm the overall benefit of GDI, a deeper, class-level analysis reveals a more nuanced picture of its impact. Although the ViT-S model achieved the highest Zero-One Accuracy (0.4286, a 67.6% improvement over its baseline) and Macro F1-Score (0.6507, up 8.10% from baseline) on the 20% data split, we focus our analysis on the EficientNetV2- L model. The 20% split was chosen as it represents a realistic, low-data scenario, allowing us to better assess the impact of GDI. EficientNetV2-L was selected because it not only achieved the strongest overall performance on the full dataset, providing a reliable baseline, but also demonstrated substantial gains in this data-constrained setting (+24.4% in Zero-One Accuracy and +5.1% in Macro F1-Score), making it both powerful and representative for class-level analysis.

![](images/9f4a14c2481ffc47a36b06b809ee6316eaf114d2d56d4173ff1a8dc904f39475.jpg)

![](images/96f86eb7745fb2460140aaf51ad06034ef5590336cfde17d24cd961b37ecf4d6.jpg)

![](images/01de364de98d79a015bc7a3289428a16c00dfa59e8c728428c5ab5ab674dadef.jpg)

![](images/7c1d364bbb29119eaa784125b2d9dceba4c6e7596df68416a9951d2637dee0a9.jpg)  
(a) ViT-S

![](images/840f4fab2bb132e8f37a8ad6770aab6aa8bcc32c94168bbe8d161eb9d66696ab.jpg)  
(b) EficientNetV2-L

![](images/13915d7d76ad2021ca0dcec2b521b71f42a6a51bb10ee5f6e6bbf36a3a394c0b.jpg)  
(c) ViT-L  
Figure 4: Performance comparison of the baseline and GDI models across three architectures for the six diferent training splits: 1%, 5%, 10%, 20%, 50%, 100% . The top image in each column shows Zero-One Accuracy and the bottom image shows Macro F1 Score.

The composition of the GDI training set and its performance impact for this experiment is detailed in Table 3, while the corresponding comparison of baseline and GDI F1-Scores is illustrated in Figure 5. The GDI model demonstrates substantial improvements, particularly for classes that were either rare or frequently co-occurred with other defects in the original dataset.

The most significant improvement is seen in Contact\_BeltMarks (visualized in Figure 3o), which saw its F1-Score increase by +63.6%. This class was extremely rare in the original 20% split and the addition of just 3 clean, isolated examples from GDI was enough to provide a more stable learning signal.

Significant gains are also observed for other visually ambiguous or less common classes. The performance for Crack\_Isolated improved by +20.1%, a direct result of its training set being bolstered with clean, isolated examples from GDI. Interestingly, the F1-Score for the Unknown class also saw a notable +16.5%. This improvement is an indirect benefit of GDI; although no new samples were generated for Unknown, by training on unambiguous examples of other defects, the model becomes more decisive in recognizing them. This reduces its tendency to misclassify known defects as Unknown, leading to a more precise application of the label.

For classes that were already well-represented and had high baseline scores, GDI provided more modest gains. For instance, No\_Defect and Contact\_FrontGridInterruption already achieved high baseline F1-Scores of 0.9586 and 0.8458, respectively, indicating the model had a strong existing understanding of these classes. For such well-learned categories, the marginal benefit of adding more clean examples is naturally lower compared to rare or ambiguous classes where the baseline model initially struggled.

We note a minor decrease in F1-Score for two classes: Crack\_Closed (- 5.2%) and Interconnect\_Disconnected (-0.7%). The latter’s score was already near-perfect in the baseline (0.9718), making further improvement dificult. The drop for Crack\_Closed might be attributed to the relatively small number of GDI samples added (2.6%) being insuficient to overcome the variance in this specific training run. However, these isolated instances are exceptions to the overwhelmingly positive trend. The results strongly indicate that GDI is a highly efective strategy for boosting the performance on rare and hard-to-distinguish classes by providing clean, unambiguous training signals.

To further investigate how GDI helps the model disentangle features, we analyzed the co-occurrence of errors. We generated a matrix for each model where both axes represent the defect classes (Figure 6). Each of-diagonal cell (i, j) in the matrix counts the number of samples where the model made an error on Class i and simultaneously made an error on Class j. The diagonal cells simply show the total number of errors for that specific class. A lower of-diagonal count in the GDI matrix therefore signifies reduced confusion between that pair of classes.

The comparison reveals that the GDI-augmented model is significantly less confused. The most compelling evidence is the reduced co-occurrence of errors with the Unknown class. In the baseline model, when an image contained a Contact\_NearSolderPad defect, the model would co-predict Unknown 165 times. The GDI model reduces this specific confusion to just 66 instances - a 60% decrease. This demonstrates that by learning from cleaner, isolated examples, the model becomes more precise in its predictions and less reliant on the ambiguous Unknown label. Similarly, confusion between Unknown and Contact\_FrontGridInterruption dropped from 80 to 54 cases and with Crack\_Closed, it fell from 75 to 43 cases.

Table 3: Training data composition and relative F1-Score change for the 20% data experiment. The table shows the number of original multi-defect labels, the number of single-defect inpainted samples generated by GDI for each class and the relative F1- Score change compared to the baseline.
<table><tr><td>Defect Class</td><td>Original Labels</td><td>Inpainted Samples</td><td>Total Samples</td><td>Inpainted %</td><td>Relative F1 Change (%)</td></tr><tr><td>Contact_BeltMarks</td><td>6</td><td>3</td><td>9</td><td>33.3%</td><td>+63.6%</td></tr><tr><td>Contact_Corrosion</td><td>19</td><td>7</td><td>26</td><td>26.9%</td><td>+8.6%</td></tr><tr><td>Contact_FrontGridInterruption</td><td>1953</td><td>13</td><td>1966</td><td>0.7%</td><td>+1.6%</td></tr><tr><td>Contact NearSolderPad</td><td>1229</td><td>15</td><td>1244</td><td>1.2%</td><td>+3.5%</td></tr><tr><td>Crack Closed</td><td>442</td><td>12</td><td>454</td><td>2.6%</td><td>-5.2%</td></tr><tr><td>Crack_Isolated</td><td>288</td><td>58</td><td>346</td><td>16.8%</td><td>+20.1%</td></tr><tr><td>Crack_Resistive</td><td>595</td><td>434</td><td>1029</td><td>42.2%</td><td>+1.8%</td></tr><tr><td>Interconnect_BrightSpot</td><td>114</td><td>56</td><td>170</td><td>32.9%</td><td>+1.9%</td></tr><tr><td>Interconnect Disconnected</td><td>32</td><td>5</td><td>37</td><td>13.5%</td><td>-0.7%</td></tr><tr><td>Interconnect_HighlyResistive</td><td>114</td><td>12</td><td>126</td><td>9.5%</td><td>+0.6%</td></tr><tr><td>No_Defect</td><td>119</td><td>1633</td><td>1752</td><td>93.2%</td><td>+0.4%</td></tr><tr><td>Unknown</td><td>14</td><td>0</td><td>14</td><td>0.0%</td><td>+16.5%</td></tr><tr><td>Total / Average</td><td>4925</td><td>2248</td><td>7173</td><td>22.7%</td><td>+9.4%</td></tr></table>

Beyond the Unknown class, GDI also mitigates confusion between distinct defect types. For example, the baseline model frequently confused Contact\_FrontGridInterruption and Contact\_NearSolderPad, with 251 cooccurring errors. The GDI model reduces this number to 177, a drop of nearly 30%.

This targeted improvement is reflected across the board: the total number of co-occurring error pairs (the sum of the upper triangle of the matrix) fell from 1,774 in the baseline to 1,312 with GDI, a systematic reduction of 26%. Collectively, this visual and quantitative evidence strongly supports our central hypothesis: by removing confounding signals from multi-defect images, GDI enables the model to learn more discriminative features, leading to more precise and reliable classifications.

![](images/9ae69239fb3779bee2896b539b7fdc4dc4664b8e83b19f6addadb1ae0ab82a3b.jpg)

Figure 5: Class-wise F1-Scores: Comparison between the baseline and GDI-enhanced EficientNetV2-L models on the 20% data split, showing gains for rare defects like Contact\_BeltMarks and Contact\_Corrosion.  
![](images/e3b5037457c7d301cef1728c6127a82cce92e02df49d0ee3c03a28a63df64439.jpg)  
(a) Baseline Model

![](images/ac702a2a191ce9224b6c4ebc168174ebb0ef8768f607832d5b67f8c920475081.jpg)  
(b) GDI-augmented Model  
Figure 6: Error co-occurrence matrices for the EficientNetV2-L model on the 20% data split. Each cell (i, j) shows the number of times class i and class j were both incorrectly predicted for the same image. The GDI-augmented model (b) shows a marked reduction in error counts across many class pairs compared to the baseline (a), indicating less confusion. Abbreviations: C-BM (Contact\_BeltMarks), C-Co (Contact\_Corrosion), C-FGI (Contact\_FrontGridInterruption), C-NSP (Contact\_NearSolderPad), K-Cl (Crack\_Closed), K-Is (Crack\_Isolated), K-Re (Crack\_Resistive), I-BS (Interconnect\_BrightSpot), I-Dc (Interconnect\_Disconnected), I-HR (Interconnect\_HighlyResistive), ND (No\_Defect), Unk (Unknown).

## 4.7. Comparison with Other Methods

To contextualize the performance of our GDI-enhanced models, we compare them against several recent methods that have utilized the same dataset. Table 4 presents this comparison, evaluating performance on both a 20% data-scarce split and the full 100% training dataset. Crucially, we extended our experimentation to apply the GDI augmentation pipeline to these baseline architectures as well, isolating the impact of the data augmentation from the model architecture.

The methodologies from other works were adapted for a fair comparison within our multi-label classification framework. Fioresi et al. [8] originally focused on semantic segmentation using DeepLabv3 [28]; for our purposes, we evaluate the performance of its ResNet-50 [29] backbone on the classification task. Abdelsattar et al. [30] benchmarked multiple models for binary classification of defects and we include their two best-performing architectures, MobileNetV2 [31] and Xception [32], in our comparison.

The results in Table 4 demonstrate that the benefits of GDI are broadly model-agnostic. In the 20% data-scarce regime, every tested architecture from the lightweight MobileNetV2 to the heavier ResNet-50 achieved higher Accuracy and F1-Scores when trained with GDI compared to their standard baselines. For instance, the ResNet-50 backbone from [8] saw its F1-Score rise from 0.6119 to 0.6313, while Xception improved from 0.6073 to 0.6195. This universality confirms that the synthetic isolation of defects provides a cleaner training signal that aids feature disentanglement regardless of the underlying inductive bias of the network. At the 100% data scale, GDI continues to provide marginal to notable gains for most architectures, with Xception seeing its F1-Score jump to 0.7569 from 0.7305.

Ultimately, our proposed architectures equipped with GDI establish a new state-of-the-art. In the low-data scenario (20%), our GDI-enhanced ViT-S model achieves the highest accuracy (0.4286) and F1-Score (0.6507), significantly outperforming all prior methods even when those methods are also augmented with GDI. On the full dataset, our EficientNetV2-L (GDI) secures the top overall performance with an accuracy of 0.6046 and an F1- Score of 0.7744. These results underscore that while GDI acts as a general performance booster, its combination with modern architectures like Vision Transformers and EficientNetV2-L yields the most robust solution for multilabel defect classification.

Table 4: Benchmarking against existing works. Performance comparison of the architectures explored in this study—EficientNetV2-L, ViT-S and ViT-L—against established baselines. Baseline models are compared with their GDI-augmented counterparts. We evaluate ResNet-50 (backbone adopted from [8]) and the best-performing models from [30] (MobileNetV2 and Xception). All models are evaluated on the original and GDI-augmented datasets using 20% and 100% training splits. Performance is measured by Zero-One Accuracy and Macro F1-score, with best results shown in bold.
<table><tr><td>Model</td><td>20% Data Accuracy</td><td>F1-Score</td><td>100% Data Accuracy</td><td>F1-Score</td></tr><tr><td colspan="5">Baselines</td></tr><tr><td>ResNet-50 (backbone adopted from [8])</td><td>0.3210</td><td>0.6119</td><td>0.5729</td><td>0.7578</td></tr><tr><td>MobileNetV2 [30]</td><td>0.2340</td><td>0.5866</td><td>0.5607</td><td>0.7469</td></tr><tr><td>Xception [30]</td><td>0.3111</td><td>0.6073</td><td>0.5576</td><td>0.7305</td></tr><tr><td>EfficientNetV2-L</td><td>0.3160</td><td>0.6013</td><td>0.5943</td><td>0.7672</td></tr><tr><td>ViT-S</td><td>0.2557</td><td>0.6019</td><td>0.5630</td><td>0.7349</td></tr><tr><td>ViT-L</td><td>0.3069</td><td>0.5777</td><td>0.5336</td><td>0.7176</td></tr><tr><td colspan="5">GDI-augmented (Ours)</td></tr><tr><td>ResNet-50</td><td>0.3687</td><td>0.6313</td><td>0.5878</td><td>0.7584</td></tr><tr><td>MobileNetV2</td><td>0.2603</td><td>0.5921</td><td>0.5638</td><td>0.7504</td></tr><tr><td>Xception</td><td>0.3195</td><td>0.6195</td><td>0.5676</td><td>0.7569</td></tr><tr><td>EfficientNetV2-L</td><td>0.3931</td><td>0.6318</td><td>0.6046</td><td>0.7744</td></tr><tr><td>ViT-S</td><td>0.4286</td><td>0.6507</td><td>0.5832</td><td>0.7605</td></tr><tr><td>ViT-L</td><td>0.3179</td><td>0.5960</td><td>0.5481</td><td>0.7385</td></tr></table>

## 4.8. Comparison with Copy-Paste Augmentation

To rigorously validate the efectiveness of our proposed method, we compared Generative Defect Isolation (GDI) against Copy-Paste augmentation [33], a widely adopted technique in instance segmentation. To ensure a fair and robust baseline, we aligned the augmented dataset size exactly with GDI and utilized ground-truth segmentation masks to extract precise defect regions, avoiding background noise contamination. The pasting mechanism was randomized across the training distribution to maximize variance. Crucially, the ground-truth labels were dynamically updated to reflect the union of the source and target defects (e.g., a ‘Crack’ pasted onto a ‘Corrosion sample results in a multi-label entry containing both classes).

As shown in Table 5, our GDI approach demonstrates superior stability, consistently improving Accuracy and F1-Scores across all three model architectures. In contrast, the Copy-Paste method yields inconsistent results. While it provided gains for ViT-S, it caused a notable performance regression in the EficientNetV2-L model, dropping accuracy from 0.5943 to

0.5721 and F1-Score from 0.7672 to 0.7463. This volatility suggests that while Copy-Paste increases dataset size, it introduces distributional noise that can outweigh the benefits of added diversity.

We attribute this inconsistency to the specific semantic and physical constraints of photovoltaic cells, which standard augmentation ignores:

1. Violation of Location Dependence: Unlike generic objects, PV defects are structurally constrained by module physics. [34] demonstrated that cracks are not stochastic, finding that 50% align parallel to busbars due to stress anisotropy. Similarly, [35] showed that contact failures are strictly bound to the electromechanical interface of the busbars. Randomly pasting these defects ignores these constraints, generating physically impossible samples that confuse the model.

2. Texture and Illumination Discontinuities: EL images exhibit specific local illumination and grain textures. Naive pasting creates sharp, artificial discontinuities that do not exist in real manufacturing defects.

Figure 7 illustrates these artifacts, showing how Copy-Paste can generate physically implausible samples. In contrast, GDI is a subtractive rather than additive process. By strictly preserving the valid, original context of the defects that remain in the image, GDI avoids generating semantic contradictions, providing a cleaner and more physically consistent training signal.

Table 5: Model Performance with Data Augmentation. The table compares the Baseline performance against Copy-Paste augmentation and our proposed GDI method on the full dataset. While Copy-Paste degrades performance for EficientNetV2-L, GDI consistently improves metrics across all architectures.
<table><tr><td rowspan="2">Model</td><td colspan="2">Baseline</td><td colspan="2">Copy-Paste</td><td colspan="2">GDI</td></tr><tr><td>Accuracy</td><td>F1-Score</td><td>Accuracy</td><td>F1-Score</td><td>Accuracy</td><td>F1-Score</td></tr><tr><td>EfficientNetV2-L</td><td>0.5943</td><td>0.7672</td><td>0.5721</td><td>0.7463</td><td>0.6046</td><td>0.7744</td></tr><tr><td>ViT-S</td><td>0.5630</td><td>0.7349</td><td>0.5725</td><td>0.7501</td><td>0.5832</td><td>0.7605</td></tr><tr><td>ViT-L</td><td>0.5336</td><td>0.7176</td><td>0.5313</td><td>0.7191</td><td>0.5481</td><td>0.7385</td></tr></table>

## 4.9. Thresholding and Performance Gains

All experiments reported so far use per-class thresholds selected to maximize the macro F1-score. To assess whether the observed improvements generalize beyond a particular weight initialization or threshold choice, we repeated the 20% data-split experiment for all three architectures across four random seeds (24, 42, 67, 76) and evaluated every run at a fixed default decision threshold of $\tau = 0 . 5$ . This protocol is deliberately conservative: it applies no per-class threshold tuning and therefore uses no information whatsoever from the test set, making any threshold-selection bias structurally impossible.

![](images/0b3d8ed16fcb73a1b0d38ad5d9a21fba83c81b84ab3f6f05f52050d3887c3a86.jpg)  
Figure 7: Visual samples generated via Copy-Paste augmentation. The images exhibit visible texture discontinuities and artificial boundaries where defects were superimposed

Table 6 reports the mean and standard deviation of Zero-One Accuracy and Macro F1 over the four seeds. Under this fixed-threshold protocol, GDI improves the mean Accuracy and mean Macro F1 for all three architectures. The efect is also consistent at the level of individual runs: in a paired comparison (baseline versus GDI on the same seed), GDI improves Accuracy in 11 of 12 runs and Macro F1 in 10 of 12 runs as shown in Table 7. Importantly, there is no seed–architecture pair for which GDI causes both Accuracy and Macro F1 to decrease simultaneously; whenever one metric shows a slight reduction, the other is improved. The few exceptions are small in magnitude and fall within the observed seed-to-seed variance. Because these results require no test-set threshold tuning, they confirm that the benefit of GDI is a systematic property of the augmentation and is not an artifact of the threshold-optimization procedure used in our main experiments.

## 5. Discussion

The experimental results strongly validate our hypothesis that isolating defects provides a superior training signal for classification models. Here, we interpret these findings and discuss their broader implications.

## 5.1. Interpretation of Performance Gains

The consistent performance improvement from GDI can be attributed to the elimination of learning ambiguity. In a standard multi-label setting, a classifier is presented with an image containing multiple defect patterns and is tasked with associating this complex visual input with multiple output labels. This can lead to the model learning spurious correlations or struggling to disentangle which visual features correspond to which specific defect class.

Table 6: Multi-seed robustness study on the 20% data split. Mean ± standard deviation over four random seeds ({24, 42, 67, 76}), evaluated at a fixed default threshold $( \tau = 0 . 5 )$ that uses no test-set information. GDI improves the mean of both metrics for every architecture. Best result within each architecture is shown in bold.
<table><tr><td>Architecture</td><td>Method Accuracy</td><td> $( \mathbf { m e a n } \pm \mathbf { s t d } )$  Macro F1</td><td> $( \mathbf { m e a n } \pm \mathbf { s t d } )$ </td></tr><tr><td rowspan="2">EfficientNetV2-L</td><td>Baseline</td><td> $0 . 3 4 3 9 \pm 0 . 0 0 7 2$ </td><td> $0 . 5 7 4 3 \pm 0 . 0 1 6 8$ </td></tr><tr><td>GDI</td><td> $\mathbf { 0 . 3 8 7 2 \pm 0 . 0 3 1 8 }$ </td><td> $\mathbf { 0 . 6 0 0 6 \pm 0 . 0 0 5 7 }$ </td></tr><tr><td rowspan="2">ViT-S</td><td>Baseline</td><td> $0 . 3 4 6 7 \pm 0 . 0 1 2 3$ </td><td> $0 . 5 7 1 0 \pm 0 . 0 0 5 7$ </td></tr><tr><td>GDI</td><td> $\mathbf { 0 . 3 8 8 3 \pm 0 . 0 1 1 1 }$ </td><td> $\mathbf { 0 . 5 9 6 5 \pm 0 . 0 1 1 9 }$ </td></tr><tr><td rowspan="2">ViT-L</td><td>Baseline</td><td> $0 . 3 0 8 3 \pm 0 . 0 1 9 6$ </td><td> $0 . 5 4 3 0 \pm 0 . 0 1 9 9$ </td></tr><tr><td>GDI</td><td> $\mathbf { 0 . 3 3 7 9 \pm 0 . 0 2 1 7 }$ </td><td> $\mathbf { 0 . 5 5 7 2 \pm 0 . 0 1 4 0 }$ </td></tr></table>

Table 7: Paired per-seed comparison on the 20% split (fixed threshold $\tau = 0 . 5 )$ . Each row compares baseline and GDI trained with the same seed; $\Delta \%$ is the relative GDI improvement over the paired baseline. This paired view controls for initialization variance.
<table><tr><td>Architecture</td><td>Seed</td><td> $\Delta \%$  Acc</td><td>Acc ↑  $\Delta \% \ \mathbf { F 1 }$ </td><td>F1 ↑</td></tr><tr><td rowspan="4">EfficientNetV2-L</td><td>24</td><td>+14.5%</td><td>√ +4.9%</td><td>√</td></tr><tr><td>42</td><td>-1.0%</td><td>+4.9% 一</td><td>√</td></tr><tr><td>67</td><td>+25.5%</td><td>√ +0.4%</td><td>√</td></tr><tr><td>76</td><td>+11.9%</td><td>V +8.4%</td><td>√</td></tr><tr><td rowspan="4">ViT-S</td><td>24</td><td>+15.0%</td><td>√</td><td>+2.6% √</td></tr><tr><td>42</td><td>+12.0%</td><td>√ +3.5%</td><td>√</td></tr><tr><td>67</td><td>+2.7%</td><td>√ +7.3%</td><td>√</td></tr><tr><td>76</td><td>+19.0%</td><td>√ +4.6%</td><td>√</td></tr><tr><td rowspan="4">ViT-L</td><td>24</td><td>+5.7%</td><td></td><td></td></tr><tr><td>42</td><td></td><td>√ +2.8% -1.8%</td><td>√</td></tr><tr><td>67</td><td>+9.5% +8.9%</td><td>√ -1.5%</td><td>一</td></tr><tr><td>76</td><td>+14.5%</td><td>√ √ +11.7%</td><td>一 √</td></tr></table>

Our GDI method helps resolve this ambiguity. By generating images where a single, clear defect pattern is present, we create a direct and clean mapping between visual evidence and its categorical label. The model is less “confused” by co-occurring defects and can learn a more robust and precise representation for each individual defect class. This culminates in a superior overall F1-Score.

## 5.2. The Value Proposition of Annotation Repurposing

A critical aspect of our methodology is its reliance on pixel-level segmentation masks, which are known to be labor-intensive and expensive to create [8]. However, this work should not be viewed as a proposal to create new segmentation datasets for classification tasks. Instead, we position GDI as a powerful technique to maximize the return on investment for existing, high-value annotated assets. Many research institutions and industrial quality control teams already possess such datasets. GDI provides a novel framework to repurpose this rich data to solve a diferent, highly practical problem.

Furthermore, this approach enables the development of lightweight and fast production-ready systems. While a segmentation model can provide detailed defect maps, such models are often computationally heavy and too slow for real-time inference on a high-throughput manufacturing line [36]. Our method allows the knowledge from these dense annotations to be distilled into a much faster classification model, which can rapidly screen and bin cells, a critical industrial requirement. GDI thus acts as a bridge, transferring detailed spatial knowledge from an ofline dataset into an eficient onlinecapable classifier.

![](images/d2669879583fade692754ae86be15ba5c4dbf0e5e86cdf6da896d9b5a8ea9ada.jpg)  
(a) Original

![](images/7a3e42180145452784633b38d8cc9fd7ec5659ad4320ecd4f902828a2208f3ee.jpg)  
(b) Boundaries

![](images/9c742d4b064ba8fca793656affa5bb4adbc3dfbcf6b5a3eeba01b3ede2b84f52.jpg)  
(c) Isolated Defect

![](images/a31397f06f9cd663f76cff9b4e419b973d370b1a650f58ec046e3ff841ef95f5.jpg)  
(d) No\_Defect (Failure)  
Figure 8: An example of a GDI failure case. When multiple defect annotations completely occlude a key structural feature, such as all the grid lines of the cell (b), the inpainting model lacks the context to reconstruct it. While it can successfully isolate a single defect - in this instance, the defect originally marked with an orange annotation boundary - it fails to restore the lines when generating the $N o \_ D e f e c t$ sample (d), resulting in a visually incomplete image.

## 5.3. Limitations and Future Work

While GDI proves highly efective, it is not without limitations. The process is contingent on the quality of the initial segmentation masks and can occasionally produce suboptimal results. A key failure mode occurs when multiple defect annotations collectively obscure a primary structural element of the cell. As illustrated in Figure 8, the cell’s grid lines are entirely covered by the combined defect masks. When generating the No\_Defect sample, the inpainting model lacks any visible reference points for these lines and consequently fails to reconstruct them, resulting in a blurry and structurally incomplete image. However, it is important to note that this is an edge case. The overwhelmingly positive impact on model performance, as demonstrated on the unmodified test set, indicates that the net benefit of providing cleaner, unambiguous training data far outweighs the impact of these rare generative failures.

To address this specific limitation, a promising avenue for future research would be to develop a domain-specific inpainting model. Unlike a generalpurpose model like LaMa [11], this network could be pre-trained on the consistent topology of PV cells to be structurally aware of features like grid lines. Such an approach might mitigate the primary generative failure mode and further enhance the robustness and quality of the GDI pipeline.

## 6. Conclusion

In this work, we addressed the challenge of multi-label defect classification in PV cell EL images, where the co-occurrence of multiple defects complicates model training and hinders feature disentanglement. We introduced Generative Defect Isolation (GDI), a data-centric augmentation strategy that leverages ground-truth segmentation masks to generate high-quality, singledefect training samples through neural inpainting. Extensive experiments across multiple architectures (ViT-S, ViT-L, EficientNetV2-L) demonstrate that GDI consistently outperforms both standard baselines and prior works, establishing a new state-of-the-art with a top Zero-One Accuracy of 0.6046 and Macro F1-Score of 0.7744. The gains are most pronounced in low-data scenarios, where providing clean, unambiguous training examples proves particularly efective in mitigating the impact of data scarcity. The benefit of resolving learning ambiguity is further evidenced by a 26% reduction in cooccurring classification errors and F1-Score improvements of up to 63.6% for rare defect classes. Collectively, these results establish GDI as a reliable and efective method for maximizing the value of existing segmentation datasets and advancing the state of multi-label defect classification in photovoltaic inspection.

## CRediT authorship contribution statement

Abdul Mueez: Writing – review & editing, Writing – original draft, Visualization, Validation, Software, Methodology, Investigation, Formal analysis, Data curation, Conceptualization.

Yogesh S. Rawat: Writing – review & editing, Visualization, Supervision, Project administration.

Shruti Vyas: Writing – review & editing, Supervision, Resources, Project administration, Funding acquisition, Formal analysis.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

The original UCF-EL-Defect dataset used in this study is publicly available. The source code for our pipeline and the generated synthetic images can be accessed via our project page at https://github.com/mueez-overflow/ generative-defect-isolation.

## References

[1] U. Hijjawi, S. Lakshminarayana, T. Xu, G. P. Malfense Fierro, M. Rahman, A review of automated solar photovoltaic defect detection systems: Approaches, challenges, and future orientations, Solar Energy 266 (2023) 112186.

[2] M. Y. Demirci, N. Beşli, A. Gümüşçü, An improved hybrid solar cell defect detection approach using generative adversarial networks and weighted classification, Expert Systems with Applications 252 (2024) 124230.

[3] W. Tang, Q. Yang, Z. Dai, W. Yan, Module defect detection and diagnosis for intelligent maintenance of solar photovoltaic plants: Techniques, systems and perspectives, Energy 297 (2024) 131222.

[4] A. Chindarkar, S. Priyadarshi, N. S. Shiradkar, A. Kottantharayil, R. Velmurugan, Deep learning based detection of cracks in electroluminescence images of fielded pv modules, in: 2020 IEEE 47th Photovoltaic Specialists Conference (PVSC), IEEE, 2020, pp. 1612–1616.

[5] M. Köntges, S. Kurtz, C. Packard, U. Jahn, K. Berger, K. Kato, Review of failures of photovoltaic modules, IEA-PVPS T 13 (01) (2014).

[6] T. Fuyuki, A. Kitiyanan, Photographic diagnosis of crystalline silicon solar cells utilizing electroluminescence, Applied physics A 96 (2009) 189–196.

[7] S. Deitsch, V. Christlein, S. Berger, C. Buerhop-Lutz, A. Maier, F. Gallwitz, C. Riess, Automatic classification of defective photovoltaic module cells in electroluminescence images, Solar Energy 185 (2019) 455–468.

[8] J. Fioresi, D. J. Colvin, R. Frota, R. Gupta, M. Li, H. P. Seigneur, S. Vyas, S. Oliveira, M. Shah, K. O. Davis, Automated defect detection and localization in photovoltaic cells using semantic segmentation of electroluminescence images, IEEE Journal of Photovoltaics 12 (1) (2022) 53–61.

[9] B. Su, Z. Zhou, H. Chen, Pvel-ad: A large-scale open-world dataset for photovoltaic cell anomaly detection, IEEE Transactions on Industrial Informatics 19 (1) (2022) 404–413.

[10] M. W. Akram, G. Li, Y. Jin, X. Chen, C. Zhu, X. Zhao, A. Khaliq, M. Faheem, A. Ahmad, Cnn based automatic detection of photovoltaic cell defects in electroluminescence images, Energy 189 (2019) 116319.

[11] R. Suvorov, E. Logacheva, A. Mashikhin, A. Remizova, A. Ashukha, A. Silvestrov, N. Kong, H. Goka, K. Park, V. Lempitsky, Resolutionrobust large mask inpainting with fourier convolutions, in: Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2022, pp. 2149–2159.

[12] D.-M. Tsai, S.-C. Wu, W.-C. Li, Defect detection of solar cells in electroluminescence images using fourier image reconstruction, Solar energy materials and solar cells 99 (2012) 250–262.

[13] D.-M. Tsai, S.-C. Wu, W.-Y. Chiu, Defect detection in solar modules using ica basis images, IEEE Transactions on Industrial Informatics 9 (1) (2013) 122–131.

[14] J. Zhang, X. Chen, H. Wei, K. Zhang, A lightweight network for photovoltaic cell defect detection in electroluminescence images based on neural architecture search and knowledge distillation, Applied Energy 355 (2024) 122184.

[15] T. Fan, T. Sun, X. Xie, H. Liu, Z. Na, Automatic micro-crack detection of polycrystalline solar cells in industrial scene, IEEE Access 10 (2022) 16269–16282.

[16] X. Zhao, C. Song, H. Zhang, X. Sun, J. Zhao, Hrnet-based automatic identification of photovoltaic module defects using electroluminescence images, Energy 267 (2023) 126605.

[17] X. Zhang, Y. Hao, H. Shangguan, P. Zhang, A. Wang, Detection of surface defects on solar cells by fusing multi-channel convolution neural networks, Infrared Physics & Technology 108 (2020) 103334.

[18] Y. Zhao, K. Zhan, Z. Wang, W. Shen, Deep learning-based automatic detection of multitype defects in photovoltaic modules and application in real production line, Progress in Photovoltaics: Research and Applications 29 (4) (2021) 471–484.

[19] A. Bartler, L. Mauch, B. Yang, M. Reuter, L. Stoicescu, Automated detection of solar cell defects with deep learning, in: 2018 26th european signal processing conference (EUSIPCO), IEEE, 2018, pp. 2035–2039.

[20] M. A. Ebied, A. Munshi, S. A. Alhuzali, M. M. El-Sotouhy, A. I. Shehta, M. Elborlsy, Advanced deep learning modeling to enhance detection of defective photovoltaic cells in electroluminescence images, Scientific Reports 15 (1) (2025) 31640.

[21] A. Jha, Y. Rawat, S. Vyas, Advancing automatic photovoltaic defect detection using semi-supervised semantic segmentation of electroluminescence images, Engineering Applications of Artificial Intelligence 160 (2025) 111790.

[22] T. Yu, R. Feng, R. Feng, J. Liu, X. Jin, W. Zeng, Z. Chen, Inpaint anything: Segment anything meets image inpainting, arXiv preprint arXiv:2304.06790 (2023).

[23] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, L. Fei-Fei, Imagenet: A large-scale hierarchical image database, in: 2009 IEEE conference on computer vision and pattern recognition, Ieee, 2009, pp. 248–255.

[24] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al., An image is worth 16x16 words: Transformers for image recognition at scale, arXiv preprint arXiv:2010.11929 (2020).

[25] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, Bert: Pre-training of deep bidirectional transformers for language understanding, in: Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), 2019, pp. 4171–4186.

[26] A. Steiner, A. Kolesnikov, X. Zhai, R. Wightman, J. Uszkoreit, L. Beyer, How to train your vit? data, augmentation, and regularization in vision transformers, arXiv preprint arXiv:2106.10270 (2021).

[27] M. Tan, Q. Le, Eficientnetv2: Smaller models and faster training, in: International conference on machine learning, PMLR, 2021, pp. 10096– 10106.

[28] L.-C. Chen, G. Papandreou, F. Schrof, H. Adam, Rethinking atrous convolution for semantic image segmentation, arXiv preprint arXiv:1706.05587 (2017).

[29] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[30] M. Abdelsattar, A. Abdelmoety, M. A. Ismeil, A. Emad-Eldeen, Automated defect detection in solar cell images using deep learning algorithms, IEEE Access (2025).

[31] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen, Mobilenetv2: Inverted residuals and linear bottlenecks, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 4510–4520.

[32] F. Chollet, Xception: Deep learning with depthwise separable convolutions, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 1251–1258.

[33] G. Ghiasi, Y. Cui, A. Srinivas, R. Qian, T.-Y. Lin, E. D. Cubuk, Q. V. Le, B. Zoph, Simple copy-paste is a strong data augmentation method for instance segmentation, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 2918–2928.

[34] S. Kajari-Schröder, I. Kunze, U. Eitner, M. Köntges, Spatial and orientational distribution of cracks in crystalline photovoltaic modules generated by mechanical load tests, Solar energy materials and solar cells 95 (11) (2011) 3054–3059.

[35] M. Heimann, R. Bakowskie, M. Köhler, J. Hirsch, M. Junghänel, A. Hussack, S. Sachert, Investigations of diferent soldering failure modes and their impact on module reliability, Energy Procedia 55 (2014) 456–463.

[36] S. Hooper, M. Chen, K. Saab, K. Bhatia, C. Langlotz, C. Ré, A case for reframing automated medical image classification as segmentation, Advances in Neural Information Processing Systems 36 (2023) 55415– 55441.