# LEARNING-STATE-AWARE DYNAMIC GENERATIVE DATA AUGMENTATION ON SMALL-SCALE DATASETS

Ting Xiang, Chenxi Deng, Jinhui Zhao, Bingting Jiang, Ke Zhang, Changjian Chen, Zhuo Tang Hunan University

{txiang, changjianchen, ztang}@hnu.edu.cn

## ABSTRACT

Small-scale image classification is often limited by the scarcity of training data. Generative data augmentation (GDA) based on pretrained generative models has emerged as an effective solution. However, existing methods rely on task-agnostic augmentation strategies that overlook downstream model needs. Although recent dynamic GDA methods incorporate model feedback to guide aug mentation, they still struggle to reliably determine sample-specific augmentation strengths and adapt augmentation strategies to different image regions while balancing image diversity and class semantics.

To address these issues, we propose learning-state-aware dynamic generative data augmentation (LSADA). Specifically, LSADA constructs a learning state for each sample based on its current loss and loss-decrease rate, which is then mapped to a sample-specific augmentation strength. Furthermore, LSADA introduces a decoupled data augmentation and diffusion fusion strategy that applies strength-controlled transformations to class-relevant regions and generates diverse class-irrelevant regions, progressively fusing them to improve image diversity while preserving class semantics. Experiments on nine public datasets show that LSADA outperforms the existing SOTA dynamic GDA method by an average of 4.5% on six natural image datasets and 2.5% on three medical image datasets.

## 1 Introduction

The success of deep learning heavily relies on large-scale, high-quality datasets [1]. This reliance greatly limits the applicability of deep learning in small-scale data scenarios, where collecting and manually annotating sufficient training data is often costly and time-consuming [2]. Recently, generative data augmentation (GDA) has emerged as a promising solution to this challenge by leveraging pretrained generative models (e.g., Stable Diffusion [3]) to generate diverse training images. However, existing GDA methods often achieve limited performance gains because their generation process is independent of the downstream model. Consequently, the generated images may fail to address the specific weaknesses of the model being trained. For example, while a downstream classifier may struggle with ambiguous samples near its decision boundary or with underrepresented classes, a task-agnostic GDA method may continue to produce redundant and easily classified images.

To address this issue, dynamic GDA has been proposed, which uses downstream model prediction results (e.g., uncertainty estimates or classification results) as feedback to select samples for further augmentation, such as ActGen [4] and DisCL [5]. Although effective to some extent, these dynamic GDA methods still face two key challenges. Challenge 1 (Fig. 1(a1)): different samples require different augmentation strengths. For example, easy samples may benefit from stronger augmentation to increase their difficulty, whereas hard samples typically require only mild augmentation to avoid underfitting. However, the prediction-based feedback of existing GDA methods only determines which samples should be augmented with a unified strength. Accurately estimating a sample-specific augmentation strength remains challenging, particularly in small-scale data scenarios, where model predictions during training are often unstable and unreliable. Challenge 2 (Fig. 1(a2)): even within a single image, different areas require different augmentation strengths. For example, class-relevant regions should undergo mild augmentation to avoid class-semantic distortion, whereas class-irrelevant regions can be augmented more strongly to increase image diversity. However, existing methods typically apply uniform augmentation across the entire image, overlooking such region-specific augmentation requirements.

![](images/e761df9ee55db51c0629b4aa74145a9b5742267c66933245f08043f1ce814d32.jpg)  
Figure 1: Comparison between (a) existing dynamic GDA methods and (b) our method, LSADA.

To address these challenges, we propose LSADA, a learning-state-aware dynamic generative data augmentation method. Instead of relying solely on prediction results, LSADA uses the current loss and loss-decrease rate to characterize each sample’s learning state (Challenge 1). The learning state is then mapped to a sample-specific augmentation strength, where larger values receive weaker augmentation to preserve semantic features, while smaller values receive stronger augmentation to improve diversity. Then, LSADA introduces a decoupled data augmentation and diffusion fusion strategy to adapt to the derived augmentation strength (Challenge 2). It applies strength-controlled rule-based transformations to class-relevant regions and uses LLM-guided prompts to generate diverse class-irrelevant regions. The two components are progressively fused during reverse denoising, improving image diversity while preserving class semantics (Fig. 1(b)). We experimentally validate the effectiveness of LSADA on nine public datasets. Experiments show that our method outperforms the existing SOTA dynamic GDA method by an average of 4.5% on six natural image datasets and 2.5% on three medical image datasets.

In summary, the main contributions of our work are:

• A sample-specific augmentation policy that dynamically adjusts augmentation strength according to the learning state of the downstream model.

• A decoupled data augmentation and diffusion fusion generation strategy that further ensures the diversity and class semantics of generated images.

• A series of experimental results demonstrates the effectiveness of our method.

## 2 Related Work

According to DA evolution (Fig. 2), existing methods can be broadly categorized into rule-based data augmentation and generative data augmentation.

![](images/f74e053c0aeea2b4a35903f54b1232f59e4896a94cf70f2867e4287000ae3365.jpg)  
Figure 2: Evolution of data augmentation.

## 2.1 Rule-based Data Augmentation

Rule-based data augmentation improves model generalization by applying predefined transformations to original images (Fig. 2(a)). Representative methods include CutOut [6], MixUp [7, 8], CutMix [9], AutoAugment [10], and RandAugment [11]. Subsequent studies extended this paradigm by selecting sample-specific transformations, including OHL-Auto-Aug [12], AdaAug [13], InstaAug [14], MADAug [15], EntAugment [16], and AdaAugment [17]. Although effective in certain scenarios, these methods are limited in generating diverse images because they only vary images at the pixel level.

## 2.2 Generative Data Augmentation

Generative data augmentation employs pretrained generative models, particularly diffusion models, to generate diverse images for downstream model training. Depending on whether feedback from the downstream model is incorporated into the augmentation process, existing methods can be further categorized into static GDA and dynamic GDA.

## 2.2.1 Static GDA.

Static GDA (Fig. 2(b)) utilizes a generation model to generate samples before classifier training by perturbing latent features [18, 19, 20, 21, 22], optimizing prompts to guide generation [23, 24, 25], or editing image regions [26]. Although effective, these methods are independent of downstream training and consequently may fail to generate samples that are fully beneficial to the downstream model. To tackle this issue, dynamic generative data augmentation has been proposed recently.

## 2.2.2 Dynamic GDA.

Dynamic GDA (Fig. 2(c)) addresses this limitation by using feedback from the downstream model to control the augmentation process during training. It adapts the generation process by optimizing the hyperparameters of an imageto-image (I2I) generative model during training according to downstream model feedback. The feedback includes classification results [27, 4], sample uncertainty [28, 29, 5, 30], sample utility [31], classification loss [32, 33].

Although effective, these methods mainly use prediction-based feedback for sample selection and cannot determine sample-specific augmentation strengths. They also apply uniform augmentation, overlooking region-specific needs. LSADA (Fig. 2(d)) instead estimates each sample’s learning state from its current loss and loss-decrease rate and maps it to an augmentation strength. It further introduces decoupled DA and diffusion fusion to separately augment class-relevant and class-irrelevant regions, improving diversity while preserving class semantics.

## 3 Problem Formulation

We first briefly introduce the dynamic GDA problem. We are given a small labeled dataset $\mathcal { D } _ { o } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ and a generative model G. Each image $x _ { i } \in \mathcal { X } \subseteq \mathbb { R } ^ { D }$ is associated with a label $y _ { i } \in \mathcal { V } = \{ 1 , . . . , c \}$ , where D is the input dimensionality and c is the number of classes.

Unlike static GDA, dynamic GDA adapts image generation according to the feedback from the downstream classifier. Let $f _ { \theta ^ { t } } : \mathcal { X }  \mathbb { R } ^ { c }$ denote the classifier at epoch t. The generated data are updated every k epochs. Specifically, the generation epochs are defined as $\mathcal { T } _ { g } = \{ k , 2 k , \ldots , R k \}$ , where $R = \lfloor T / k \rfloor$ denotes the number of generation rounds and $T$ is the total augmentation training epochs. At each generation epoch $t \in \mathcal { T } _ { g }$ , the feedback from $f _ { \theta ^ { t } }$ determines the augmentation strength $a _ { i } ^ { t }$ for each sample $x _ { i }$ . The generative model G then generates $m _ { t }$ samples for each original sample:

$$
\mathcal { D } _ { g _ { i } } ^ { t } = \left\{ \left( \hat { x } _ { i , j } ^ { t } , y _ { i } \right) \mid \hat { x } _ { i , j } ^ { t } = G \left( x _ { i } , y _ { i } ; a _ { i } ^ { t } \right) \right\} _ { j = 1 } ^ { m _ { t } } ,\tag{1}
$$

where $m _ { t }$ is specified by the generation ratio.

The generated set at epoch t is $\begin{array} { r } { \mathcal { D } _ { g } ^ { t } = \bigcup _ { i = 1 } ^ { n } \mathcal { D } _ { g _ { i } } ^ { t } } \end{array}$ . It is combined with the original data and the previously generated sets to update the classifier. After all generation epochs, the complete generated dataset is $\begin{array} { r } { \mathcal { D } _ { g } = \bigcup _ { t \in \mathcal { T } _ { g } } \mathcal { D } _ { g } ^ { t } } \end{array}$ . The overall generation ratio is $\begin{array} { r } { m = \sum _ { t \in \mathcal { T } _ { g } } m _ { t } } \end{array}$ , which indicates the total number of generated samples for each original sample.

Finally, the downstream classifier is trained by minimizing the empirical loss on both original and generated data:

$$
\theta = \arg \operatorname* { m i n } _ { \theta } \sum _ { ( x , y ) \in \mathcal { D } _ { o } \cup \mathcal { D } _ { g } } \mathcal { L } \left( f _ { \theta } ( x ) , y \right) ,\tag{2}
$$

where $\mathcal { L }$ denotes the classification loss, such as cross-entropy.

## 4 Method

We propose learning-state-aware dynamic generative data augmentation (LSADA), which consists of two components: sample learning state estimation, and decoupled data augmentation and diffusion fusion generation. First, LSADA estimates the learning state of each sample using its current loss and loss-decrease rate (Fig. 3(a)). The estimated learning state is then mapped to a sample-specific augmentation strength, which controls rule-based transformations applied to the class-relevant region. For the class-irrelevant region, we use a large language model to generate diverse prompts. Finally, LSADA progressively fuses the transformed class-relevant region with the generated class-irrelevant region during the diffusion reversion process, generating diverse training images for the downstream classifier (Fig. 3(b)). The overall algorithm is summarized in Appendix A.

## 4.1 Sample Learning State Estimation

Existing dynamic GDA methods typically use downstream model predictions as feedback to select original samples for further augmentation. However, such feedback only determines which samples should be augmented, but cannot

![](images/5b4859ba7e72e207bd27e976b1c2da7c20a16323cb669d8653d5e03dcee8cf80.jpg)  
Figure 3: The framework of LSADA. LSADA consists of two stages: (a) sample learning state estimation, using current loss and loss-decrease rate, (b) decoupled data augmentation and diffusion fusion generation. The diffusion fusion generation pipeline is detailed in Fig. 4.

assign sample-specific augmentation strengths. To construct a sample-specific augmentation strength, we characterize each sample’s learning state using its current loss and loss-decrease rate, which reflect its current learning difficulty and recent learning progress, respectively [34, 35].

Specifically, we estimate the learning state of each sample every k epochs. At epoch $t ,$ we first compute the current loss of sample $( x _ { i } , y _ { i } )$ as

$$
\ell _ { i } ^ { t } = \mathcal { L } \left( f _ { \theta ^ { t } } ( x _ { i } ) , y _ { i } \right) ,\tag{3}
$$

where $f _ { \theta ^ { t } }$ denotes the classifier at epoch $t ,$ and L denotes the classification loss, such as cross-entropy loss.

To measure how much each sample has been learned during the past k epochs, we define the loss-decrease rate as

$$
\delta _ { i } ^ { t } = ( \ell _ { i } ^ { t - k } - \ell _ { i } ^ { t } ) / ( \ell _ { i } ^ { t - k } + \epsilon _ { 1 } ) ,\tag{4}
$$

where $\epsilon _ { 1 }$ is a small constant for numerical stability.

Then we combine the current loss and the loss-decrease rate to define the learning state:

$$
L _ { i } ^ { t } = \ell _ { i } ^ { t } - \beta \delta _ { i } ^ { t } ,\tag{5}
$$

where $\beta > 0$ balances the current learning difficulty and recent learning progress. A larger $L _ { i } ^ { t }$ indicates that the sample is harder to learn, characterized by a higher loss and slower improvement, while a smaller value suggests easier to learn.

## 4.2 Learning-State-Aware Image Generation

Given the estimated learning state of each sample, we further map it to an augmentation strength to guide image generation.

A straightforward solution is to use the augmentation strength to control the entire image generation process, such as adjusting the guidance scale to regulate the degree of image modification. However, applying a unified strength to the entire image may cause class-semantic distortion or insufficient diversity. To address this issue, we propose a decoupled data augmentation and diffusion fusion generation strategy, which decouples the augmentation of class-relevant and class-irrelevant regions, then fuses these components through diffusion fusion generation.

## 4.2.1 Decoupled data augmentation.

We first identify the class-relevant region in each image. Given a class-relevant mask $M _ { i } \in [ 0 , 1 ] ^ { H \times W }$ , the class-relevant region of the original image $x _ { i }$ is defined as

$$
x _ { i } ^ { r } = M _ { i } \odot x _ { i } ,\tag{6}
$$

where H and $W$ denote the image height and width, respectively, and $\odot$ denotes element-wise multiplication. For datasets with clearly defined class-relevant regions, we obtain $M _ { i }$ using an off-the-shelf segmentation model. For datasets without clearly defined class-relevant regions, we instead derive $M _ { i }$ from class activation maps that highlight class-discriminative visual cues.

Then, we apply rule-based augmentations $( e . g .$ , Rotation, Cutout, Resize) to the class-relevant region $\boldsymbol { x } _ { i } ^ { r }$ , and control their strengths using the learning state $L _ { i } ^ { t }$ . At epoch t, we first normalize the learning states $\{ L _ { 1 } ^ { t ^ { - } } , L _ { 2 } ^ { t } , \ldots , L _ { n } ^ { t } \}$ using normalization:

$$
\widetilde { L } _ { i } ^ { t } = \frac { L _ { i } ^ { t } - \operatorname* { m i n } _ { j } L _ { j } ^ { t } } { \operatorname* { m a x } _ { j } L _ { j } ^ { t } - \operatorname* { m i n } _ { j } L _ { j } ^ { t } + \epsilon _ { 2 } } ,\tag{7}
$$

where $\epsilon _ { 2 }$ is a small constant for numerical stability.

The normalized learning state is subsequently mapped to an augmentation strength:

$$
a _ { i } ^ { t } = a _ { \operatorname* { m i n } } + \big ( a _ { \operatorname* { m a x } } - a _ { \operatorname* { m i n } } \big ) ( 1 - \widetilde { L } _ { i } ^ { t } ) ,\tag{8}
$$

where $a _ { \mathrm { m i n } }$ and $a _ { \mathrm { m a x } }$ denote the minimum and maximum augmentation strengths, respectively. Accordingly, a larger $\widetilde { L } _ { i } ^ { t }$ leads to weaker augmentation, while a smaller $\widetilde { L } _ { i } ^ { t }$ leads to stronger augmentation.

Based on the resulting strength, we apply the same rule-based transformation to both the class-relevant region $\boldsymbol { x } _ { i } ^ { r }$ and its mask $M _ { i }$

$$
\hat { x } _ { i } ^ { r } = \mathrm { T } ( x _ { i } ^ { r } ; a _ { i } ^ { t } ) , \quad \hat { M } _ { i } = \mathrm { T } ( M _ { i } ; a _ { i } ^ { t } ) ,\tag{9}
$$

where $\mathrm { T } ( \cdot ; a _ { i } ^ { t } )$ denotes a rule-based transformation parameterized by the augmentation strength $a _ { i } ^ { t }$

To further enhance image diversity, rather than explicitly extracting class-irrelevant regions from the original image, we use a large language model (LLM) to generate a set of diverse background prompts $\{ p _ { i } ^ { j } \} _ { j = 1 } ^ { m _ { t } }$ , which guide the generation of class-irrelevant regions.

![](images/f354c32fa9147aca0717865392a5a59dc9a2a406f4af1f669991f0eaea7747f2.jpg)  
Figure 4: Diffusion fusion generation pipeline.

## 4.2.2 Diffusion fusion generation.

Given the augmented class-relevant region $\hat { x } _ { i } ^ { r }$ , the transformed mask $\hat { M } _ { i } .$ , and a background prompt $p _ { i } ^ { j } .$ , we fuse them to generate the image $\hat { x } _ { i , j } ^ { t }$

A straightforward way is to first generate a background image conditioned on the prompt using a Stable Diffusion model and then directly paste $\hat { x } _ { i } ^ { r }$ onto it. However, this approach may generate images that deviate from the real data distribution. Moreover, the class-relevant region may not blend well with the generated background, which can degrade downstream classification performance. To address these issues, we introduce a diffusion fusion generation process (Fig. 4).

To be specific, we first encode the augmented class-relevant region $\hat { x } _ { i } ^ { r }$ into the latent space:

$$
z _ { i , 0 } ^ { r } = \operatorname { E n c } ( \hat { x } _ { i } ^ { r } ) ,\tag{10}
$$

where Enc(·) denotes the VAE encoder.

Since diffusion fusion is performed in the latent space, we resize the transformed mask $\hat { M _ { i } }$ to match the spatial dimensions of the latent feature map using nearest-neighbor interpolation:

$$
\widetilde { M } _ { i } = \mathrm { R e s i z e } H _ { z } \times W _ { z } ( \hat { M } _ { i } ) , \quad \widetilde { M } _ { i } \in [ 0 , 1 ] ^ { H _ { z } \times W _ { z } } ,\tag{11}
$$

where Resize $H _ { z } \times W _ { z } ( \cdot )$ denotes nearest-neighbor resizing to the spatial resolution $H _ { z } \times W _ { z }$ of the latent feature map.

To preserve the semantic structure of the class-relevant region, we first apply DDIM inversion [36] to the encoded latent $z _ { i , 0 } ^ { r }$ . We record the inverted latent trajectory over the fusion interval [τ<sub>1</sub>, τ<sub>2</sub>]:

$$
\{ z _ { i , \tau } ^ { r } \} _ { \tau = \tau _ { 1 } } ^ { \tau _ { 2 } } = \mathrm { I n v } ( z _ { i , 0 } ^ { r } ) ,\tag{12}
$$

where Inv(·) denotes the DDIM inversion process.

Then we initialize the reversion process with the inverted latent at timestep τ<sub>2</sub>:

$$
z _ { i , \tau _ { 2 } } = z _ { i , \tau _ { 2 } } ^ { r } .\tag{13}
$$

Starting from $z _ { i , \tau _ { 2 } } .$ , we perform reverse denoising conditioned on the background prompt $p _ { i } ^ { j }$ . For each timestep $\tau \in [ \tau _ { 1 } , \tau _ { 2 } ]$ , the latent is first updated by one reverse denoising step:

$$
z _ { i , \tau - 1 } = \mathcal { R } _ { \tau } ( z _ { i , \tau } , p _ { i } ^ { j } ) ,\tag{14}
$$

where $\mathscr { R } _ { \tau } ( \cdot , p _ { i } ^ { j } )$ denotes the reverse denoising operation at timestep τ conditioned on prompt $p _ { i } ^ { j }$

After each denoising step, we inject the inverted class-relevant latent into the corresponding masked region:

$$
z _ { i , \tau - 1 } = \widetilde { M } _ { i } \odot z _ { i , \tau - 1 } ^ { r } + ( 1 - \widetilde { M } _ { i } ) \odot z _ { i , \tau - 1 } ,\tag{15}
$$

where ⊙ denotes element-wise multiplication. The first term preserves the class semantics structure, while the second term allows the class-irrelevant region to be generated according to the background prompt.

We continue the reverse denoising process from $\tau _ { 1 }$ to 0 without masked injection:

$$
z _ { i , \tau - 1 } = \mathscr { R } _ { \tau } ( z _ { i , \tau } , p _ { i } ^ { j } ) , \quad \tau \in [ 1 , \tau _ { 1 } ] .\tag{16}
$$

Finally, the latent at timestep 0 is decoded to obtain the generated image:

$$
\hat { x } _ { i , j } ^ { t } = \mathrm { D e c } ( z _ { i , 0 } ) ,\tag{17}
$$

where Dec(·) denotes the VAE decoder.

## 5 Experiments

## 5.1 Experimental Settings

## 5.1.1 Datasets.

We evaluate the effectiveness and efficiency of LSADA on nine public small-scale image classification datasets, covering both natural and medical images. The natural image datasets include Caltech-101 [37] and CIFAR100-Subset [38] for coarse-grained classification; Cars [39], Flowers [40], and Pets [41] for fine-grained classification; and DTD [42] for texture classification. The medical image datasets are drawn from MedMNIST [43] and include PathMNIST [44] for colon pathology, BreastMNIST [45] for breast ultrasound, and OrganSMNIST [46] for abdominal CT classification. Detailed dataset statistics are provided in Appendix B.

## 5.1.2 Baselines.

To evaluate the effectiveness of our proposed method, we compare LSADA with representative and state-of-theart (SOTA) data augmentation methods from different categories, including rule-based augmentation methods (i.e., CutOut [6], RandAugment [11], TrivialAugment [47], TeachAugment [48], MADAug [15] and EntAugment [16]), SOTA static generative augmentation methods (GIF [21]), and SOTA dynamic generative augmentation methods (DisCL [5] and ActGen [4]).

## 5.1.3 Implementation details.

We use ResNet-50 as the downstream classifier for all datasets and train it for 200 epochs. Image generation is performed only during the first (T=100) training epochs, with the generated data updated every 5 epochs. Following the GDA baselines, we use Stable Diffusion v.2.1 (SD 2.1) for image generation and expand each dataset with a generation ratio m = 5. To extract class-relevant regions, we employ SAM3 [49] for Caltech-101, Cars, Flowers, CIFAR100-S, and Pets, and CAM [50] for the remaining datasets. For class-relevant region transformation, we apply Rotation with angles in [5<sup>◦</sup>, 20<sup>◦</sup>], Cutout with ratios in [5%, 20%] and Resize with scaling ratios in [60%, 100%]. For class-irrelevant regions, we use the GPT-5.5 API to generate diverse background prompts. More details are provided in Appendix C.

<table><tr><td rowspan="2">Dataset</td><td colspan="7">Natural image datasets</td><td colspan="4">Medical image datasets</td></tr><tr><td>Caltech 101</td><td>Cars</td><td>Flowers</td><td>DTD</td><td>CIFAR100-S</td><td>Pets</td><td>Avg.</td><td>PathMNIST</td><td>BreastMNIST</td><td>OrganSMNIST</td><td>Avg.</td></tr><tr><td>original</td><td>26.3</td><td>19.8</td><td>74.1</td><td>23.1</td><td>35.0</td><td>6.8</td><td>30.9</td><td>72.4</td><td>55.8</td><td>76.3</td><td>68.2</td></tr><tr><td>CutOut</td><td>51.5</td><td>25.8</td><td>77.8</td><td>24.2</td><td>44.3</td><td>38.7</td><td>43.7</td><td>78.8</td><td>66.7</td><td>78.3</td><td>74.6</td></tr><tr><td>RandAugment</td><td>57.8</td><td>43.2</td><td>83.8</td><td>28.7</td><td>46.7</td><td>48.0</td><td>51.4</td><td>79.2</td><td>68.7</td><td>79.6</td><td>75.8</td></tr><tr><td>TrivialAugment</td><td>49.9</td><td>21.1</td><td>81.8</td><td>28.0</td><td>37.3</td><td>5.9</td><td>37.3</td><td>83.2</td><td>61.0</td><td>78.2</td><td>74.1</td></tr><tr><td>TeachAugment</td><td>70.5</td><td>25.9</td><td>58.7</td><td>51.5</td><td>31.3</td><td>68.7</td><td>51.1</td><td>79.6</td><td>74.3</td><td>77.2</td><td>77.0</td></tr><tr><td>MADAug</td><td>65.5</td><td>55.3</td><td>84.7</td><td>44.4</td><td>38.2</td><td>67.8</td><td>59.3</td><td>75.6</td><td>73.1</td><td>75.9</td><td>74.9</td></tr><tr><td>EntAugment</td><td>70.7</td><td>69.0</td><td>75.9</td><td>40.8</td><td>52.4</td><td>67.8</td><td>62.8</td><td>86.2</td><td>74.4</td><td>77.7</td><td>79.4</td></tr><tr><td>SD 2.I</td><td>55.7</td><td>64.5</td><td>80.7</td><td>32.4</td><td>55.9</td><td>42.4</td><td>55.3</td><td>76.3</td><td>73.7</td><td>51.6</td><td>67.2</td></tr><tr><td>GIF</td><td>54.4</td><td>60.6</td><td>82.1</td><td>33.9</td><td>61.1</td><td>52.9</td><td>57.5</td><td>86.9</td><td>77.4</td><td>80.7</td><td>81.7</td></tr><tr><td>DisCL</td><td>45.8</td><td>26.8</td><td>82.2</td><td>30.6</td><td>30.3</td><td>45.8</td><td>43.6</td><td>83.7</td><td>76.9</td><td>79.2</td><td>79.9</td></tr><tr><td>ActGen</td><td>78.2</td><td>90.9</td><td>97.5</td><td>60.2</td><td>50.7</td><td>86.2</td><td>77.3</td><td>90.3</td><td>78.8</td><td>80.8</td><td>83.3</td></tr><tr><td>Ours</td><td>86.1</td><td>91.4</td><td>98.3</td><td>63.4</td><td>61.7</td><td>89.7</td><td>81.8</td><td>90.9</td><td>84.6</td><td>81.8</td><td>85.8</td></tr><tr><td></td><td>(+7.9)</td><td>(+0.5)</td><td>(+0.8)</td><td>(+3.2)</td><td>(+11.0)</td><td>(+3.5)</td><td>(+4.5)</td><td>(+0.6)</td><td>(+5.8)</td><td>(+1.0)</td><td>(+2.5)</td></tr></table>

Table 1: Accuracy (in %) of a ResNet-50 trained from scratch on original and generated images by different methods.

![](images/a58e83e0f31f6393c0165a5245d63f4c2ef1c678ba6e2db2863db797eaaa975b.jpg)  
Figure 5: Accuracy (in %) of a ResNet-50 trained from scratch on the original and generated datasets by LSADA, SD 2.1 and ActGen with different generation ratios.

<table><tr><td>Method</td><td>ResNeXt-50</td><td>WideResNet-50</td><td>MobileNet-V2</td></tr><tr><td>Original</td><td>18.4</td><td>32.0</td><td>26.2</td></tr><tr><td>ActGen</td><td>84.6</td><td>86.5</td><td>79.5</td></tr><tr><td>LSADA</td><td>89.9</td><td>90.4</td><td>85.1</td></tr></table>

Table 2: Accuracy of various architectures trained on 5×-generated Pets.

## 5.2 Overall Performance

## 5.2.1 DA effectiveness.

As shown in Table 1, our method consistently outperforms rule-based data augmentation methods and the SOTA static GDA method by a large margin. It also achieves higher classification accuracy than existing dynamic GDA methods. In particular, LSADA surpasses ActGen, the SOTA dynamic GDA method, by an average of 4.5% on natural image datasets and 2.5% on medical image datasets. These results suggest that selecting samples solely based on model predictions may provide insufficient guidance in small-scale data scenarios. In contrast, LSADA characterizes each sample using its learning state, maps it to a sample-specific augmentation strength, and employs a generation strategy adapted to that strength, making it well suited to small-scale data scenarios.

## 5.2.2 DA efficiency.

We further compare LSADA with ActGen and SD 2.1 under different generation ratios. As shown in Fig. 5, LSADA consistently achieves competitive or higher accuracy using fewer generated images than ActGen. In particular, on DTD and Caltech-101, LSADA with a 1× generation ratio achieves higher accuracy than ActGen with a 10× ratio. Thus, LSADA requires only one-tenth as many generated images on these datasets, demonstrating at least a 10× improvement in data augmentation efficiency.

## 5.2.3 Generalization to various architectures.

In practical applications, generated images may be used to train downstream models with various architectures. To evaluate the architectural generalizability of LSADA, we use the 5×-generated images obtained with feedback from ResNet-50 to train ResNeXt-50, WideResNet-50, and MobileNet-V2 from scratch. As shown in Table 2, LSADA consistently improves the classification accuracy of all evaluated architectures. These results show that, although model feedback is used during data generation, the resulting images are not specific to the guiding architecture and can effectively transfer to other downstream models.

<table><tr><td>Feedback</td><td>Current Loss</td><td>Loss-Decrease</td><td>Acc</td></tr><tr><td>No Feedback</td><td>X</td><td>X</td><td>88.9</td></tr><tr><td>Prediction Feedback</td><td>X</td><td>X</td><td>89.2</td></tr><tr><td>Loss Only</td><td>V</td><td>X</td><td>89.1</td></tr><tr><td>Decrease Only</td><td>X</td><td>√</td><td>89.0</td></tr><tr><td>Learning State (Ours)</td><td>V</td><td>V</td><td>89.7</td></tr></table>

Table 3: Accuracy (in %) comparison of different learning state signals on Pets.

<table><tr><td>Mapping strategies</td><td>Ours</td><td>Reversed</td></tr><tr><td>Acc</td><td>89.7</td><td>88.6</td></tr></table>

Table 4: Accuracy (in %) comparison under different mapping strategies on Pets.

<table><tr><td>Generation strategies</td><td>Ours</td><td>I2I</td><td>Paste</td></tr><tr><td>Acc</td><td>89.7</td><td>86.2</td><td>88.7</td></tr></table>

Table 5: Accuracy (in %) comparison under different generation strategies on Pets.

## 5.3 Ablation Study

To further evaluate the effectiveness of LSADA, we conduct ablation studies on key components, including the learning state feedback, the mapping from learning state to augmentation strength, and the decoupled data augmentation and diffusion fusion generation strategy.

## 5.3.1 Learning state feedback.

To evaluate the effectiveness of the proposed feedback, we keep all other components fixed and vary only the feedback signal. The compared variants include no feedback, the prediction-based feedback used in ActGen, current loss only, and loss-decrease rate only. As shown in Table 3, combining the current loss and loss-decrease rate achieves the highest accuracy, validating the effectiveness of the proposed learning state.

## 5.3.2 Mapping strategy.

To evaluate the effectiveness of mapping the learning state $L _ { i } ^ { t }$ to the augmentation strength $a _ { i } ^ { t } ,$ we compare the proposed strategy with a reversed mapping. The proposed strategy assigns weaker augmentation to samples with larger learning state values and stronger augmentation to those with smaller values, while the reversed strategy follows the opposite assignment. As shown in Table 4, the proposed mapping consistently achieves higher accuracy, demonstrating the effectiveness of adapting augmentation strength to the learning state of each sample.

## 5.3.3 Decoupled data augmentation and diffusion fusion generation.

To evaluate the effectiveness of the proposed generation strategy, we compare it with global image-to-image diffusion generation (I2I) and decoupled augmentation followed by direct pasting (Paste). As shown in Table 5, our strategy achieves the highest accuracy when all other components are kept unchanged. These results demonstrate that decoupled data augmentation and diffusion fusion better balance image diversity and class-semantic preservation than global editing or direct pasting.

We further compare image quality across generation strategies. As shown in Table 6, I2I has a much higher FID, indicating a larger distribution gap from the original images. Paste achieves the lowest FID, likely because it directly preserves the foreground. Overall, the results suggest that an appropriate distribution gap may benefit downstream performance.

The samples of natural datasets generated by the three strategies are visualized in Fig. 6. Although the I2I strategy produces diverse images, it often alters class-relevant regions, leading to semantic distortion. The Paste strategy better preserves class semantics, but directly pasting the class-relevant region onto the background makes the images look unnatural, which may cause them to deviate from the data manifold. In contrast, our strategy preserves semantic consistency while achieving more natural fusion between the class-relevant region and the background. Similar observations on the medical image datasets are provided in Appendix D.

<table><tr><td>Generation strategies</td><td>FID↓</td><td>IS↑</td><td>LPIPS↓</td></tr><tr><td>I2I</td><td>107.8</td><td>10.33</td><td>0.71</td></tr><tr><td>Paste</td><td>40.10</td><td>17.58</td><td>0.74</td></tr><tr><td>Ours</td><td>46.61</td><td>14.75</td><td>0.73</td></tr></table>

Table 6: Image quality comparison under different generation strategies on Pets.

I2I  
Paste  
Ours  
![](images/552fb7afd5cd490fe322e9319e0686f30ff0f619040e57540558bd6c752cfaf6.jpg)  
Figure 6: Natural image visualization under different generation strategies.

## 6 Conclusion

This paper addresses two key limitations of existing dynamic GDA methods: the difficulty of determining samplespecific augmentation strengths and the lack of region-specific generation strategies for balancing image diversity and class semantics. To address these limitations, we propose LSADA, which characterizes the learning state of each sample using its current loss and loss-decrease rate and adaptively determines its augmentation strength. Furthermore, LSADA introduces a decoupled augmentation and diffusion fusion generation strategy to achieve region-specific augmentation, improving diversity while preserving class semantics. Extensive experiments demonstrate the effectiveness of LSADA on both natural and medical image classification tasks.

There are several directions for future research. For instance, extending LSADA to other computer vision tasks, such as object detection and semantic segmentation, would be valuable. Additionally, exploring how learning-state guidance can be integrated with other generative models and data modalities is an interesting avenue for future investigation.

## References

[1] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255, Miami, Florida, USA, 2009. Ieee, IEEE Computer Society.

[2] Guo-Jun Qi and Jiebo Luo. Small data challenges in big data era: A survey of recent progress on unsupervised and semi-supervised methods. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(4):2168–2187, 2020.

[3] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, New Orleans, LA, USA, 2022. IEEE.

[4] Tao Huang, Jiaqi Liu, Shan You, and Chang Xu. Active generation for image classification. In European Conference on Computer Vision, pages 270–286. Springer, 2024.

[5] Yijun Liang, Shweta Bhardwaj, and Tianyi Zhou. Diffusion curriculum: Synthetic-to-real data curriculum via image-guided diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1697–1707, 2025.

[6] Terrance DeVries and Graham W Taylor. Improved regularization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552, 2017.

[7] Dan Hendrycks, Norman Mu, Ekin D Cubuk, Barret Zoph, Justin Gilmer, and Balaji Lakshminarayanan. Augmix: A simple data processing method to improve robustness and uncertainty. arXiv preprint arXiv:1912.02781, 2019.

[8] Linjun Zhang, Zhun Deng, Kenji Kawaguchi, Amirata Ghorbani, and James Zou. How does mixup help with robustness and generalization? arXiv preprint arXiv:2010.04819, 2020.

[9] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6023–6032, 2019.

[10] Ekin D Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, and Quoc V Le. Autoaugment: Learning augmentation policies from data. arXiv preprint arXiv:1805.09501, 2018.

[11] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated data augmentation with a reduced search space. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition workshops, pages 702–703, 2020.

[12] Chen Lin, Minghao Guo, Chuming Li, Xin Yuan, Wei Wu, Junjie Yan, Dahua Lin, and Wanli Ouyang. Online hyper parameter learning for auto-augmentation strategy. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 6579–6588, 2019.

[13] Tsz-Him Cheung and Dit-Yan Yeung. Adaaug: Learning class-and instance-adaptive data augmentation policies. In International conference on learning representations, 2021.

[14] Ning Miao, Tom Rainforth, Emile Mathieu, Yann Dubois, Yee Whye Teh, Adam Foster, and Hyunjik Kim. Learning instance-specific augmentations by capturing local invariances. arXiv preprint arXiv:2206.00051, 2022.

[15] Chengkai Hou, Jieyu Zhang, and Tianyi Zhou. When to learn what: Model-adaptive data augmentation curriculum. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 1717–1728, 2023.

[16] Suorong Yang, Furao Shen, and Jian Zhao. Entaugment: Entropy-driven adaptive data augmentation framework for image classification. In European conference on computer vision, pages 197–214. Springer, 2024.

[17] Suorong Yang, Peijia Li, Xin Xiong, Furao Shen, and Jian Zhao. Adaaugment: A tuning-free and adaptive approach to enhance data augmentation. IEEE Transactions on Image Processing, 2025.

[18] Antreas Antoniou, Amos Storkey, and Harrison Edwards. Data augmentation generative adversarial networks. arXiv preprint arXiv:1711.04340, 2017.

[19] Khawar Islam, Muhammad Zaigham Zaheer, Arif Mahmood, and Karthik Nandakumar. Diffusemix: Labelpreserving data augmentation with diffusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27621–27630, 2024.

[20] Dvir Samuel, Rami Ben-Ari, Simon Raviv, Nir Darshan, and Gal Chechik. Generating images of rare concepts using pre-trained diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 4695–4703, 2024.

[21] Yifan Zhang, Daquan Zhou, Bryan Hooi, Kai Wang, and Jiashi Feng. Expanding small-scale datasets with guided imagination. Advances in neural information processing systems, 36:76558–76618, 2023.

[22] Haowei Zhu, Ling Yang, Jun-Hai Yong, Hongzhi Yin, Jiawei Jiang, Meng Xiao, Wentao Zhang, and Bin Wang. Distribution-aware data expansion with diffusion models. Advances in Neural Information Processing Systems, 37:102768–102795, 2024.

[23] Ruifei He, Shuyang Sun, Xin Yu, Chuhui Xue, Wenqing Zhang, Philip Torr, Song Bai, and Xiaojuan Qi. Is synthetic data from generative models ready for image recognition? arXiv preprint arXiv:2210.07574, 2022.

[24] Bohan Li, Xiao Xu, Xinghao Wang, Yutai Hou, Yunlong Feng, Feng Wang, Xuanliang Zhang, Qingfu Zhu, and Wanxiang Che. Semantic-guided generative image augmentation method with diffusion models for image classification. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 3018–3027, 2024.

[25] Brandon Trabucco, Kyle Doherty, Max Gurinas, and Ruslan Salakhutdinov. Effective data augmentation with diffusion models. In International Conference on Learning Representations, volume 2024, pages 14590–14612, 2024.

[26] Khawar Islam, Arif Mahmood, Xin Jin, and Naveed Akhtar. Instructmixup: Instruction-guided salient patch editing for robust data augmentation. arXiv preprint arXiv:2607.19324, 2026.

[27] Soroush Abbasi Koohpayegani, Anuj Singh, KL Navaneet, Hamed Pirsiavash, and Hadi Jamali-Rad. Genie: Generative hard negative images through diffusion. arXiv preprint arXiv:2312.02548, 2023.

[28] Kyuheon Jung, Yongdeuk Seo, Seongwoo Cho, Jaeyoung Kim, Hyun-seok Min, and Sungchul Choi. Dalda: Data augmentation leveraging diffusion model and llm with adaptive guidance scaling. In European Conference on Computer Vision, pages 182–200. Springer, 2024.

[29] Zerun Wang, Jiafeng Mao, Xueting Wang, and Toshihiko Yamasaki. Difficulty controlled diffusion model for synthesizing effective training data. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 10367–10375, 2026.

[30] Reyhane Askari-Hemmat, Mohammad Pezeshki, Elvis Dohmatob, Florian Bordes, Pietro Astolfi, Melissa Hall, Jakob Verbeek, Michal Drozdzal, and Adriana Romero-Soriano. Improving the scaling laws of synthetic data with deliberate practice. arXiv preprint arXiv:2502.15588, 2025.

[31] Jiyu Guo, Shuo Yang, Yiming Huang, Yancheng Long, Xiaobo Xia, Xiu Su, Bo Zhao, Zeke Xie, and Liqiang Nie. Utilgen: Utility-centric generative data augmentation with dual-level task adaptation. Advances in Neural Information Processing Systems, 38:32474–32500, 2026.

[32] Dang Nguyen, Jiping Li, Jinghao Zheng, and Baharan Mirzasoleiman. Do we need all the synthetic data? targeted image augmentation via diffusion models. In The Fourteenth International Conference on Learning Representations, 2025.

[33] Shuhan Li, Yi Lin, Hao Chen, and Kwang-Ting Cheng. Iterative online image synthesis via diffusion model for imbalanced classification. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 371–381. Springer, 2024.

[34] Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, and Geoffrey J. Gordon. An empirical study of example forgetting during deep neural network learning. In International Conference on Learning Representations, 2019.

[35] Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite. Deep learning on a data diet: Finding important examples early in training. Advances in neural information processing systems, 34:20596–20607, 2021.

[36] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

[37] Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In 2004 conference on computer vision and pattern recognition workshop, pages 178–178. IEEE, 2004.

[38] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

[39] Jonathan Krause, Jia Deng, Michael Stark, and Li Fei-Fei. Collecting a large-scale dataset of fine-grained cars. 2013.

[40] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In 2008 Sixth Indian conference on computer vision, graphics & image processing, pages 722–729. IEEE, 2008.

[41] Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In 2012 IEEE conference on computer vision and pattern recognition, pages 3498–3505. IEEE, 2012.

[42] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3606–3613, 2014.

[43] Jiancheng Yang, Rui Shi, and Bingbing Ni. Medmnist classification decathlon: A lightweight automl benchmark for medical image analysis. In 2021 IEEE 18th international symposium on biomedical imaging (ISBI), pages 191–195. IEEE, 2021.

[44] Jakob Nikolas Kather, Johannes Krisam, Pornpimol Charoentong, Tom Luedde, Esther Herpel, Cleo-Aron Weis, Timo Gaiser, Alexander Marx, Nektarios A Valous, Dyke Ferber, et al. Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study. PLoS medicine, 16(1):e1002730, 2019.

[45] Walid Al-Dhabyani, Mohammed Gomaa, Hussien Khaled, and Aly Fahmy. Dataset of breast ultrasound images. Data in brief, 28:104863, 2020.

[46] Xuanang Xu, Fugen Zhou, Bo Liu, Dongshan Fu, and Xiangzhi Bai. Efficient multiple organ localization in ct image using 3d region proposal network. IEEE transactions on medical imaging, 38(8):1885–1898, 2019.

[47] Samuel G Müller and Frank Hutter. Trivialaugment: Tuning-free yet state-of-the-art data augmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 774–782, 2021.

[48] Teppei Suzuki. Teachaugment: Data augmentation optimization using teacher knowledge. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 10904–10914, New Orleans, LA, USA, 2022. IEEE.

[49] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.

[50] Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba. Learning deep features for discriminative localization. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2921–2929, 2016.

## A Algorithm

The complete algorithm is summarized in Algorithm 1. LSADA alternates between downstream classifier optimization and learning-state-aware image generation. At each epoch, the classifier is updated using the union of the original and previously generated data. Every k epochs, LSADA computes the current loss and loss-decrease rate of each original sample and combines them to estimate its learning state. The learning states are then normalized and mapped to sample-specific augmentation strengths. For each sample, LSADA extracts the class-relevant region and applies a strength-controlled rule-based transformation to both the region and its mask. Meanwhile, we use an LLM to generate diverse background prompts, which guide the diffusion fusion process in Eqs. (10)–(17) to progressively integrate the transformed class-relevant region with a generated class-irrelevant region (background prompt). The resulting images are added to $\mathcal { D } _ { g }$ and used in subsequent classifier updates. After T augmentation epochs, the final classifier is trained on $\mathcal { D } _ { o } \cup \mathcal { D } _ { g } .$

Algorithm 1 The LSADA Framework   
Input: Original dataset $\mathcal { D } _ { o } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n } ;$ generative model $G ,$ classification model $f ;$ maximum augmentation   
epochs $T ;$ update interval $k ;$ generation number $m _ { t }$ for each sample at epoch $t ,$ rule-based augmentation $\breve { \mathrm { T } } .$   
Output: Generated dataset $\mathcal { D } _ { g }$ and final classification model $f _ { \theta }$ parameterized by   
$\theta .$   
1: $\mathcal { D } _ { g }  \emptyset$   
2: for $t = 1 , \dots , T$ do   
3: Update the classification model to obtain $f _ { \theta ^ { t } }$ using $\mathcal { D } _ { o } \cup \mathcal { D } _ { g }$   
4: if t mod $k = 0$ then   
5: for $i = 1 , \ldots , n$ do   
6: Compute current loss: $\ell _ { i } ^ { t } \gets \mathcal { L } ( f _ { \theta ^ { t } } ( x _ { i } ) , y _ { i } )$   
7: Compute loss-decrease rate: $\delta _ { i } ^ { t } \gets ( \ell _ { i \cdot } ^ { t - k } - \ell _ { i } ^ { t } ) / ( \ell _ { i } ^ { t - k } + \epsilon ) ;$   
8: Compute learning state value: $L _ { i } ^ { t } \gets \ell _ { i } ^ { t } - \beta \delta _ { i } ^ { t }$   
9: end for   
10: Normalize $\{ L _ { i } ^ { t } \} _ { i = 1 } ^ { n }$ to obtain $\{ \widetilde { L } _ { i } ^ { t } \} _ { i = 1 } ^ { n }$   
11: for $i = 1 , \ldots , \stackrel { \cdot } { n }$ do   
12: Compute augmentation strength: $a _ { i } ^ { t } \gets a _ { \operatorname* { m i n } } + ( a _ { \operatorname* { m a x } } - a _ { \operatorname* { m i n } } ) ( 1 - \widetilde { L } _ { i } ^ { t } )$   
13: Extract class-relevant region: mask $M _ { i } \gets \mathrm { S }$ egment $\mathbf { \Phi } _ { ( x _ { i } ) ; x _ { i } ^ { r }  M _ { i } }$ ⊙ x<sub>i</sub>   
14: Rule-based Transform on class-relevant region: $( \hat { x } _ { i } ^ { r } , \hat { M } _ { i } ) \gets \mathrm { T } ( x _ { i } ^ { r } , M _ { i } ; a _ { i } ^ { t } )$   
15: Generate prompts: $\{ p _ { i } ^ { j } \} _ { j = 1 } ^ { m _ { t } } \gets \mathrm { L L M } ( x _ { i } , y _ { i } , m _ { t } )$   
16: for $j = 1 , \dots , m _ { t }$ do   
17: Generate image using Eqs. (10)–(17)   
← DifusionFusion $( G , \hat { x } _ { i } ^ { r } , \hat { M } _ { i } , p _ { i } ^ { j } )$   
18: $\mathcal { D } _ { g }  \mathcal { D } _ { g } \cup \{ ( \hat { x } _ { i , j } ^ { t } , y _ { i } ) \}$   
19: end for   
20: end for   
21: end if   
22: end for   
23: θ ← TrainClassifier $( \mathcal { D } _ { o } \cup \mathcal { D } _ { g } )$   
24: return $\mathcal { D } _ { g }$ and $f _ { \theta }$

## B Datasets

We evaluate our method on nine publicly available image classification datasets, including six natural image datasets and three medical image datasets. A detailed summary of their statistics is provided in Table 7.

The natural image datasets include Caltech 101 [37], CIFAR100-Subset (CIFAR100-S) [38], Stanford Cars [39], Oxford 102 Flowers (Flowers) [40], Oxford-IIIT Pets (Pets) [41], and DTD [42]. CIFAR100-Subset is constructed from CIFAR100 by randomly sampling 100 images from each class, resulting in 10,000 images across 100 categories. This controlled subset provides a representative small-scale setting for evaluating the effectiveness of our method. Collectively, these datasets cover multiple image recognition scenarios, including coarse-grained object classification (i.e., Caltech 101 and CIFAR100-Subset), fine-grained object classification $( i . e .$ , Stanford Cars, Flowers, and Pets), and texture classification (i.e., DTD).

<table><tr><td>Dataset</td><td>Type</td><td>Classes</td><td>Samples</td><td>Average samples per class</td></tr><tr><td>Caltech 101</td><td>Natural</td><td>102</td><td>3,060</td><td>30</td></tr><tr><td>CIFAR100-Subset</td><td>Natural</td><td>100</td><td>10,000</td><td>100</td></tr><tr><td>Stanford Cars</td><td>Natural</td><td>196</td><td>8,144</td><td>42</td></tr><tr><td>Flowers</td><td>Natural</td><td>102</td><td>6,552</td><td>64</td></tr><tr><td>Pets</td><td>Natural</td><td>37</td><td>3,842</td><td>104</td></tr><tr><td>DTD</td><td>Natural</td><td>47</td><td>3,760</td><td>80</td></tr><tr><td>BreastMNIST</td><td>Medical</td><td>2</td><td>78</td><td>39</td></tr><tr><td>PathMNIST</td><td>Medical</td><td>9</td><td>10,004</td><td>1,112</td></tr><tr><td>OrganSMNIST</td><td>Medical</td><td>11</td><td>13,940</td><td>1,267</td></tr></table>

Table 7: Statistics of both natural and medical image datasets.

The medical image datasets are obtained from the MedMNIST benchmark [43], including BreastMNIST [45], PathM-NIST [44], and OrganSMNIST [46]. These datasets contain breast ultrasound, colon pathology, and abdominal CT images, respectively. To construct small-scale training scenarios, we use the official validation splits of BreastMNIST and PathMNIST as their training sets. For OrganSMNIST, we retain the original training split because its available training set is already relatively limited.

## C Implementation Details

Training and generation schedule. The downstream classifier is trained for 200 epochs, while image generation is performed only during the first $T = 1 0 0$ augmentation training epochs. With $k = 5 .$ , the number of generation rounds is $\dot { R } = | T / k | \overset { - } { = } 2 0$ , and the generation epochs are $\mathcal { T } _ { g } = \{ 5 , 1 0 , \dotsc , 1 0 0 \}$ under one-based epoch indexing. We set the overall generation ratio to $m = 5 .$ , yielding a target per-round generation ratio of $m / R = 0 . 2 5$

This fractional ratio is realized at the dataset level. At each generation epoch, approximately $n / 4$ original samples are randomly selected without replacement from $\mathcal { D } _ { o } .$ , and one image is generated for each selected sample. The 20 generation rounds therefore yield approximately 5n generated images in total, although no individual original sample is required to be selected exactly five times. For each selected sample, its learning state determines the augmentation strength $a _ { i } ^ { t } .$ . The generated samples are accumulated in $\mathcal { D } _ { g }$ and used together with $\mathcal { D } _ { o }$ in subsequent training epochs After epoch 100, no additional images are generated, and training continues on $\mathcal { D } _ { o } \cup \mathcal { D } _ { g }$ until epoch 200.

Computing environment. All experiments are conducted using four NVIDIA RTX 4090 GPUs, each with 24 GB of memory, with PyTorch 2.1.2 and CUDA 12.1.

Diffusion configuration. Let N denote the total number of DDIM inference steps. We set $N = 1 0 , \tau _ { 1 } = \lfloor 0 . 5 N \rfloor = 5$ and $\tau _ { 2 } = \lfloor 0 . 7 N \rfloor = 7$ . In each cycle, noise is injected from $\tau _ { 1 } ~ \mathrm { t o } ~ \tau _ { 2 }$ , followed by reverse denoising with masked injection from τ<sub>2</sub> to τ<sub>1</sub>. Under the implementation’s inference-step indexing, masked injection is applied at timesteps {7, 6, 5}. This cycle is repeated 10 times.

During the final reverse denoising stage from $\tau _ { 1 }$ to $0 ,$ we additionally perform masked injection at a dataset-specific set of late-stage timesteps, denoted by $\mathcal { T } _ { \mathrm { i n j } }$ . We set ${ \mathcal { T } } _ { \mathrm { i n j } } = \{ 4 , 3 \}$ for Caltech-101, Cars, Flowers, Pets, and DTD, and $ { \mathcal { T } } _ { \mathrm { i n j } } = \{ 4 \}$ for CIFAR100-Subset, PathMNIST, BreastMNIST, and OrganSMNIST. At each $\tau \in \mathcal { T } _ { \mathrm { i n j } }$ , the corresponding inverted class-relevant latent is injected into the class-relevant region. Apart from these additional timesteps, the remaining reverse denoising steps follow Eq. (16) in the main paper without masked injection.

## D Visualization on Medical Datasets

The samples generated by the three strategies on the medical image datasets are visualized in Fig. 7. Compared with the original images, I2I generally preserves the main medical structures but introduces noticeable changes in global appearance and texture, which may cause semantic distortion. Paste and our method both use the same CAM maps to preserve class-related regions. However, Paste directly blends the preserved regions with generated backgrounds in pixel space, which may lead to inconsistent structures and textures and produce unnatural visual patterns. In contrast, our diffusion fusion process integrates the preserved and generated regions more smoothly, producing visually coherent images while retaining discriminative medical structures. Overall, our method achieves a better balance between semantic preservation and visual quality.

Original images  
I2I  
Paste  
Ours  
![](images/8c73f015e8e0322e72edafff39cf62f9011798c130e5baaf678bd7ca6c286eba.jpg)  
Figure 7: Medical image visualization under different generation strategies.