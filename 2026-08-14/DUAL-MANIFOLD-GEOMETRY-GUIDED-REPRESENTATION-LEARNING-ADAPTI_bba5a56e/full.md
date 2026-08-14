# DUAL-MANIFOLD GEOMETRY GUIDED REPRESENTATION LEARNING: ADAPTIVE COUPLING BETWEEN KERNEL AND DATA SPACES

A PREPRINT

Wencong Zhang<sup>†1,2</sup>, Yue Zhang<sup>†1,2</sup>, Meiyan Huang<sup>1,2</sup>, Wei Yang<sup>1,2</sup>, and Qianjin Feng<sup>\*1,2</sup> <sup>1</sup>School of Biomedical Engineering, Southern Medical University, China, Guangzhou 510515 <sup>2</sup>Guangdong Provincial Key Laboratory of Medical Image Processing, Southern Medical University

August 14, 2026

## ABSTRACT

Deep representation learning has primarily focused on how feature representations evolve across network layers, while largely overlooking the structured geometry embedded in network parameters. In this work, we introduce a dual-manifold perspective of deep representation learning, in which each convolutional layer contains two coupled geometric spaces: a Kernel Manifold induced by convolutional filters and a Data Manifold characterized by intermediate feature representations. Since these two manifolds share the same channel space, the geometry of network parameters can provide complementary structural information for guiding feature evolution. Based on this insight, we propose Kernel-Guided Feature Transform (KGFT), a lightweight and nearly parameterfree module that derives a geometric guidance matrix from the kernel Gram matrix and uses it to transform the covariance structure of feature representations. Rather than reweighting feature responses as in conventional attention mechanisms, KGFT explicitly reshapes feature relationships by transferring geometric information from the kernel manifold to the data manifold. To accommodate the hierarchical nature of deep networks, we further introduce Exploit and Explore modes with a depth aware scheduling strategy, together with a learnable guidance strength that adaptively controls the contribution of geometric transformation. This design promotes geometric alignment in shallow layers while encouraging feature diversity in deeper layers, without imposing excessive geometric constraints on representation learning. Theoretical analysis establishes the geometric validity of the proposed transformation and characterizes its effect on feature covariance. Extensive experiments across CNN- and Transformer-based architectures, including ResNet, ViT, and LLaMA-7B, demonstrate consistent improvements on image classification and arithmetic reasoning tasks, validating the generality and effectiveness of kernel-guided dual-manifold representation learning. Code is available at https://github.com/ZWC-SMU/KGFT.

Keywords Dual-Manifold · Kernel-Guided Transform · Exploit and Explore · Learnable Guidance Strength

## 1 Introduction

Deep neural networks (DNN) have achieved remarkable success across a wide range of text and visual tasks by progressively transforming low-level patterns into high-level semantic representations [Krizhevsky et al., 2012, Simonyan and Zisserman, 2014, Szegedy et al., 2015, He et al., 2016, Howard et al., 2017, Tan and Le, 2019, Vaswani et al., 2017, Dosovitskiy et al., 2020, Gu and Dao, 2023, Liu et al., 2024]. A fundamental principle underlying this success is hierarchical representation learning, where shallow layers capture basic structures such as edges and textures, while deeper layers encode increasingly abstract semantic concepts. Consequently, most existing studies focus on understanding and improving how feature representations evolve across network layers through architectural designs [He et al., 2016, Vaswani et al., 2017, Gu and Dao, 2023], representation learning [Hinton and Zemel, 1993, Mikolov et al., 2013, Chen et al., 2020, He et al., 2022], and attention mechanisms [Vaswani et al., 2017, Hu et al., 2018, Woo et al., 2018, Liu et al., 2021]. Although the representations extracted by these methods have achieved impressive performance in downstream tasks such as classification, detection, segmentation, and generation, they predominantly characterize representation learning from the feature space, treating intermediate features as the primary carriers of information and network parameters as transformation operators. However, features are not learned in isolation, they are generated through the interaction between input data and the parameters that transform them. In particular, convolutional kernels undergo continuous adaptation during training and gradually develop non-random correlations and structured configurations. Recent studies have revealed that network weights exhibit geometric and spectral structures, which are closely associated with the evolution and organization of learned features [Beaglehole et al., 2024, Defilippis et al., 2026, Cha et al., 2026]. These findings suggest that the parameter space itself contains meaningful structural information that complements the geometry of the feature space.

This observation motivates us to reconsider representation learning from a joint geometric perspective: Can the geometry encoded by network parameters actively guide the evolution of feature representations? Within a convolutional layer, the representation process can be viewed through two coupled geometric structures. On the one hand, convolutional weights induce a kernel geometry, where correlations among filters characterize the relationships encoded by the learned parameters. On the other hand, feature representations induce a data geometry, whose covariance structure describes how feature channels are organized in the representation space [Raghu et al., 2017, Kornblith et al., 2019]. Although generated from different sources, the two manifolds share the same channel space and coevolve throughout network optimization. This intrinsic correspondence suggests that the kernel manifold provides an informative geometric prior for guiding the evolution of the data manifold. However, existing representation learning methods typically optimize feature representations while overlooking this dual-manifold interaction, leaving the relationship between parameter geometry and feature geometry largely unexplored.

Motivated by above, we propose a novel Dual-Manifold Geometry Guided Representation Learning framework, which formulates deep representation learning as a dynamic coupling process between kernel geometry and data geom etry. Instead of reweighting feature responses as in conventional attention mechanisms, our method transfers structural information from the kernel manifold to reshape the covariance structure of feature representations. This geometric interaction is implemented by a lightweight Kernel-Guided Feature Transform (KGFT) module, which derives a kernel-induced guidance matrix and employs a learnable guidance strength to adaptively control the contribution of geometric transformation. To accommodate the hierarchical learning behavior of deep networks, we further introduce a dual-mode guidance strategy, namely Exploit and Explore. Specifically, shallow layers emphasize geometric alignment to establish stable low-level representations, whereas deeper layers encourage geometric expansion to promote semantic diversity. This depth-aware scheduling enables kernel geometry to guide representation learning throughout the entire optimization process. Extensive experiments across both CNN- and Transformer-based architectures, including ResNet, ViT, and LLaMA-7B, demonstrate that KGFT consistently improves performance on image classification and arithmetic reasoning tasks while introducing negligible additional overhead. The proposed framework is simple, lightweight, and can be readily incorporated into existing convolutional networks without modifying their backbone architectures. Our contributions can be summarized as the followings:

• A dual-manifold representation learning framework. We formulate feature learning as the geometric interaction between a kernel manifold and a data manifold, providing a new perspective on representation learning in deep neural networks.

• A lightweight kernel-guided feature transformation. We propose KGFT as a plug-and-play module with negligible parameter overhead that transfers geometric structures from convolutional kernels to feature representations through covariance transformation, enabling adaptive geometry-guided feature learning.

• A depth-aware dual-mode scheduling strategy. We design a simple yet effective geometry scheduling mechanism that performs geometric alignment in shallow layers (Exploit) and geometric expansion in deep layers (Explore).

• Extensive experimental validation. We demonstrate its effectiveness across CNN- and Transformer-based architectures, including multiple ResNet variants, ViT-Tiny, and LLaMA-7B, on image classification and arithmetic reasoning tasks.

## 2 Related work

## 2.1 Manifold Perspective of Deep Neural Networks

Deep neural networks can be viewed as a sequence of nonlinear transformations that progressively reshape the underlying data manifold [Raghu et al., 2017, Bengio et al., 2013]. A growing body of work has investigated this geometric evolution from different perspectives, including representation dynamics, manifold geometry, and Riemannian formulations [Cohen et al., 2020, Hauser and Ray, 2017, Benfenati and Marta, 2023]. Despite these advances, existing manifold-based analyses mainly focus on the evolution of data manifolds induced by feature transformations, while the intrinsic geometric structures embedded in network parameters remain less explored. Recent studies have revealed that network parameters also induce structured geometric representations through kernel formulations. Jacot et al. [Jacot et al., 2018] introduced the Neural Tangent Kernel (NTK), demonstrating that the parameter space of neural networks implicitly defines a kernel geometry that governs the evolution of learned functions. Furthermore, Fort et al. [Fort et al., 2020] showed that the induced kernel is not static during training but continuously evolves with parameter updates, highlighting the dynamic geometric relationship between network parameters and learned representations. However, existing studies primarily investigate parameter-induced kernel geometry and feature-space geometry independently. In contrast, our work views a deep convolutional network as a coupled system of two interacting manifolds: the kernel manifold defined by convolutional weights and the data manifold represented by feature distributions. By explicitly modeling their geometric relationship, we investigate how parameter geometry can guide feature evolution.

## 2.2 Feature Covariance Modeling

Feature covariance provides a fundamental means of characterizing second-order relationships among feature representations. Such second-order information has been shown to contain complementary and often highly discriminative information beyond individual feature responses [Gatys et al., 2016, Kong and Fowlkes, 2017, Li et al., 2017, Wang et al., 2026]. For example, Gram matrix based methods, such as neural style transfer [Gatys et al., 2016], demonstrated that covariance statistics capture meaningful feature structures. Subsequent covariance pooling approaches [Li et al., 2018] further exploited second-order statistics for compact representation learning. However, existing covariancebased methods mainly treat feature statistics as the target representation to be normalized or aggregated. In contrast, our method introduces a different perspective: the covariance structure of features is not only analyzed but actively modulated by the geometric structure of convolutional kernels.

## 2.3 Kernel Methods and Manifold Mapping

Kernel methods provide a principled way to characterize geometric relationships through kernel functions and their associated Gram matrices [Cha et al., 2026, Schölkopf et al., 1998, Esser et al., 2024]. Recent studies have extended this geometric perspective to deep neural networks, showing that learned representations exhibit structured manifold geometry and evolve systematically across network layers [Schölkopf et al., 1998, Esser et al., 2024]. In particular, the weight Gram matrix has been shown to capture meaningful structures underlying feature dynamics, revealing a close relationship between parameter geometry and the evolution of learned representations [Cha et al., 2026]. These findings suggest that kernel-induced geometry can provide informative structural priors for representation learning. Inspired by these principles, our method interprets both convolutional kernels and feature activations as geometric objects.

## 2.4 Attention Mechanisms and Feature Recalibration

Attention mechanisms have become a fundamental paradigm for improving feature representation by dynamically modeling the relative importance or relationships among different elements [Hu et al., 2018, Woo et al., 2018, Li et al., 2019, Wang et al., 2020]. Channel- and spatial-attention methods, such as SENet [Hu et al., 2018], CBAM [Woo et al., 2018], SKNet [Li et al., 2019], and ECA-Net [Wang et al., 2020], adapt feature responses through channel recalibration or selective aggregation, while self-attention extends this principle to pairwise interactions among tokens [Vaswani et al., 2017, Dosovitskiy et al., 2020, Liu et al., 2021]. These methods demonstrate that dynamically modeling feature relationships can effectively enhance representation quality. However, existing attention mechanisms primarily operate within the feature space, emphasizing or aggregating features according to their learned responses. In contrast, KGFT introduces kernel-guided geometric adaptation: rather than reweighting feature elements, it transfers the structural geometry of convolutional kernels to reshape feature relationships.

## 3 Method

## 3.1 Dual-Manifold Formulation

Deep neural networks can be interpreted as a hierarchical geometric transformation process, where feature distributions evolve across network depth. Existing manifold-based analyses mainly focus on the geometry of learned features, while ignoring the intrinsic geometric structures encoded in network parameters. To bridge these two perspectives, we propose a dual-manifold formulation, which models a convolutional layer as an interaction between two coupled manifolds: the kernel-manifold induced by convolutional weights and the data-manifold formed by feature representations.

## 3.1.1 Kernel Manifold

Let $\mathbf { W } \in \mathbb { R } ^ { C \times D }$ be the weight matrix of a certain convolutional layer, where C denotes the number of output channels and $D = k ^ { 2 } C _ { i n }$ represents the flattened kernel dimension. Treating each row of W as a point in $\mathbb { R } ^ { \mathbb { D } }$ , these $C$ points span a kernel-manifold $\mathcal { M } _ { \mathbf { W } } \subset \mathbb { R } ^ { D }$ . We characterize the intrinsic geometric relationships through the kernel Gram matrix:

$$
\mathbf { G } = \mathbf { W } \mathbf { W } ^ { \top } \in \mathbb { R } ^ { C \times C }\tag{1}
$$

The element $\mathbf { G } _ { i j } ~ = ~ \langle \mathbf { w } _ { i } , \mathbf { w } _ { j } \rangle$ is the inner-product similarity between the i-th and j-th kernels, reflecting their angular relationship and correlation structure. The eigen-decomposition of $\mathbf { G } = \mathbf { W } \mathbf { W } ^ { \top } \in \mathbb { R } ^ { C \times C }$ reveals the principal geometric directions of the kernel manifold. Larger eigenvalues indicate that the network allocates stronger representational capacity along the corresponding kernel directions.

## 3.1.2 Data Manifold

Let the input features for the current layer be $\mathbf { X } \in \mathbb { R } ^ { B \times C \times H \times W }$ , flattened into $\mathbf { X _ { f } } \in \mathbb { R } ^ { N \times C }$ (where $N = B \times H \times W )$ Each spatial feature vector is considered as a point in the channel space, forming the data manifold: $\mathcal { M } _ { \mathbf { X } } \subset \mathbb { R } ^ { C }$ . The geometric structure of is characterized by the feature covariance matrix:

$$
\mathbf { K } = \frac { 1 } { \sqrt { N } } { \mathbf { X _ { f } } } ^ { \top } \mathbf { X _ { f } } \in \mathbb { R } ^ { C \times C }\tag{2}
$$

The element ${ \bf K } _ { i j } = \langle { \bf k } _ { i } , { \bf k } _ { j } \rangle$ represents the covariance relationship between the i-th and j-th feature channels, describing the intrinsic distribution of learned representations.

## 3.1.3 Dual-Manifold Correspondence

The kernel manifold $\mathcal { M } _ { \mathbf { W } }$ is constructed from convolutional filters, whereas the data manifold $\mathcal { M } _ { \mathbf { X } }$ is formed by spatial feature vectors. Although they originate from different spaces, their geometric structures are represented in the same channel space. Accordingly, the kernel Gram matrix G characterizes how filters are organized and correlated in parameter space, while the feature covariance matrix K describes how feature responses are distributed and correlated in data space. Thus, the kernel geometry provides structural cues about dominant and weak channel directions, which can be transferred to modulate the corresponding feature geometry. To establish this interaction, we introduce a kernel-induced geometric guidance matrix M, which adaptively enhances or suppresses feature directions according to the underlying kernel geometry. The construction of M is described in the following section.

## 3.2 Dual-Manifold Geometric Guidance

The goal of KGFT is to transfer kernel geometry into feature space. However, different network depths require different geometric behaviors. Shallow layers require stable feature construction, while deep layers require semantic diversity. Therefore, we design two complementary guidance modes:

• Exploit mode: preserve and reinforce reliable kernel structures;

• Explore mode: suppress excessive correlations and encourage new feature directions.

## 3.2.1 Dual-Mode Guidance Matrix

We first get the guidance matrix $\mathbf { M } \in \mathbb { R } ^ { C \times C }$ from the metric G of the kernel manifold, whose M includes Exploit and Explore modes:

Exploit Mode: For shallow layers, stable low-level representations are more important than representation diversity. Therefore, we directly utilize the kernel geometry:

$$
\mathbf { M } _ { \mathrm { e x p l o i t } } = \mathbf { G } + \varepsilon \mathbf { I } _ { \mathbf { C } }\tag{3}
$$

where $\varepsilon > 0$ ensures positive definiteness. Consequently, directions strongly represented by the kernel manifold receive larger geometric emphasis, whereas weak kernel directions receive relatively smaller emphasis. The geometric interpretation and positive definiteness of $\mathbf { M } _ { \mathrm { e x p l o i t } }$ is formally established in Appendix A. 6.1 and 6.2.

Explore Mode: As network depth increases, excessive dependence on existing kernel correlations may restrict semantic diversity. To encourage feature exploration, we first normalize the kernel Gram matrix:

$$
\mathbf { C } = \mathbf { D } ^ { - 1 / 2 } \mathbf { G } \mathbf { D } ^ { - 1 / 2 } , w h e r e \mathbf { D } = d i a g ( \mathbf { G } )\tag{4}
$$

The explore guidance matrix is then defined as:

$$
\mathbf { M } _ { \mathrm { e x p l o r e } } = \delta ( \mathbf { I } _ { C } - \mathbf { C } ) + \varepsilon \mathbf { I } _ { C }\tag{5}
$$

where $\begin{array} { r } { \delta = \frac { 1 } { C } \Sigma _ { i = 1 } ^ { C } \mathbf { G } _ { i i } } \end{array}$ denotes the average geometric scale. Different from exploit mode, explore mode suppresses highly correlated kernel directions while amplifying weakly correlated directions, encouraging feature representations to expand toward unexplored semantic dimensions. The geometric interpretation and positive definiteness of $\mathbf { M } _ { \mathrm { e x p l o r e } }$ is formally established in Appendix A. 6.1 and 6.2.

## 3.2.2 Kernel-Guided Feature Transformation

After obtaining the geometric guidance matrix: $\mathbf { M } \in \mathbf { M } _ { \mathrm { e x p l o i t } } , \mathbf { M } _ { \mathrm { e x p l o r e } }$ , KGFT uses M to modulate the channel geometry of the feature representation:

$$
\mathbf { K } ^ { \prime } = \mathbf { M } \mathbf { K } \in \mathbb { R } ^ { C \times C }\tag{6}
$$

The new metric K<sup>′</sup> is applied to the feature transformation:

$$
\mathbf { Y _ { f } } = \mathbf { X _ { f } } \mathbf { K ^ { \prime } } = \mathbf { X _ { f } } \mathbf { M } \mathbf { K }\tag{7}
$$

which the transformed feature representation assigns different geometric weights to different channel directions according to the kernel structure. The transformed feature is then reshaped back to its original spatial dimensions and processed by a lightweight projection layer:

$$
\mathbf { Y } = L N ( \mathrm { P r o j } ( \mathrm { r e s h a p e } ( \mathbf { Y _ { f } } ) ) )\tag{8}
$$

where $\mathrm { P r o j }$ and LN denote a $1 \times 1$ convolution and layer normalization, respectively. Finally, the transformed feature is fused with the original representation through residual learning:

$$
\mathbf { Y } = \mathbf { X } + \mathbf { Y }\tag{9}
$$

The residual connection preserves the original feature pathway, while the transformed branch introduces adaptive geometric guidance from the kernel manifold. The stability of the mapping from kernel parameters to the resulting feature transformation is established in Appendix A. 6.3.

## 3.2.3 Adaptive Geometry Guidance Strength

Although kernel geometry provides meaningful structural priors, excessive guidance may constrain early representation learning. Therefore, KGFT introduces a learnable geometry guidance strength allows the network to automatically determine the contribution of geometric transformation. Specifically, we parameterize the guidance strength as a trainable scalar:

$$
\begin{array} { r } { \mathbf { s } = \sigma ( \alpha ) } \end{array}\tag{10}
$$

where α is a learnable parameter and $\sigma ( \cdot )$ denotes the sigmoid function that maps the guidance strength to the interval [0,1]. During training, a warm-up strategy is adopted to gradually activate the learnable geometric guidance, thereby preventing excessive geometric perturbation in the early optimization stage. The final KGFT transformation is formulated as:

$$
\mathbf { Y } = \mathbf { X } + \mathbf { s } \cdot { \mathcal { F } } _ { K G F T } ( \mathbf { X } )\tag{11}
$$

where $\mathcal { F } _ { K G F T } ( \cdot )$ denotes the geometry-guided feature transformation described in Section 3.2.2. The learnable coefficient s adaptively controls the extent to which the transformed feature representation contributes to the output. When $\mathbf { s } \to 0 ,$ KGFT approaches an identity mapping, allowing the network to preserve the original feature representation. $\mathbf { A s \thinspace s  1 }$ , the geometric transformation is fully incorporated, enabling the network to maximally exploit the selected geometric guidance. This bounded residual formulation provides an explicit mechanism for controlling the magnitude of geometric modulation. A formal boundedness and identity-preserving analysis is provided in Appendix A. 6.4.

## 3.3 Dual-Manifold Scheduling in Deep Neural Networks

To apply KGFT to deep neural architectures, we introduce a depth-aware dual-manifold scheduling strategy that determines where and how KGFT modules are incorporated into the network. The underlying principle is that shallow layers perform geometry alignment (Exploit), whereas deeper layers perform geometry exploration (Explore). We apply this scheduling strategy to both convolutional and Transformer-based architectures, including ResNet, ViT, and LLaMA-7B. For ResNet architectures, KGFT modules are inserted at selected stages following the shallow-to-deep transition. For ViT and LLaMA-7B, which consists of a sequence of Transformer encoder layers, we similarly adopt a sparse layer-wise scheduling strategy. The detailed configurations for different ResNet and Transformer architectures are summarized in the Table 5 and Fig. 2 of Appendix B.

## 4 Experiments

## 4.1 Datasets and Metrics

We evaluate KGFT across both vision and language tasks to assess its generality beyond convolutional architectures. For image classification, we use two widely adopted benchmarks, CIFAR-100 [Krizhevsky et al., 2009] and ImageNet-1K [Deng et al., 2009]. CIFAR-100 contains 50,000 training and 10,000 test images from 100 classes, with a spatial resolution of 32 × 32. ImageNet-1K contains approximately 1.28 million training images and 50,000 validation images across 1,000 classes. Images are randomly resized and cropped to 224 × 224 during training. Following standard practice, we apply random cropping, horizontal flipping, label smoothing, and Cutout augmentation [DeVries and Taylor, 2017]. Top-1 classification accuracy is used as the evaluation metric. Furthermore, we conduct arithmetic reasoning experiments with LLaMA-7B [Touvron et al., 2023], using MATH10K [Wu et al., 2024] for training and evaluating the fine-tuned models on four held-out benchmarks, including AQuA [Ling et al., 2017], GSM8K [Cobbe et al., 2021], MAWPS [Koncel-Kedziorski et al., 2016], and SVAMP [Patel et al., 2021]. For these language tasks, answer accuracy is used as the evaluation metric.

## 4.2 Experimental Settings

We evaluate the proposed KGFT on representative residual architectures with different depths and model capacities, including ResNet20 (0.27M parameters, 3 stages), ResNet32 (0.47M parameters, 3 stages), ResNet18 (11.15M parameters, 4 stages), ResNet34 (20.79M parameters, 4 stages), and ResNet50 (24.37M parameters, 4 stages). These models cover both lightweight networks and deeper architectures, enabling a comprehensive evaluation of the effectiveness and scalability of KGFT.

To adapt KGFT to Transformer architectures, we generalize the channel-wise correspondence used in convolutional networks to the hidden representation space of Transformer blocks. Specifically, for the l-th Transformer block, let $\mathbf { W } _ { l } ^ { d o w n } \in \mathbb { R } ^ { H \times I }$ denote the weight matrix of the MLP down-projection layer, where H and I denote the hidden dimension and the intermediate MLP dimension, respectively. The down-projection maps the intermediate MLP representation back to the H-dimensional hidden space, each row of $\mathbf { W } _ { l } ^ { d o w n }$ corresponds to one output hidden dimension. We therefore construct the hidden-space kernel geometry as

$$
\mathbf { G } _ { 1 } = \mathbf { W } _ { 1 } ^ { d o w n } \left( \mathbf { W } _ { 1 } ^ { d o w n } \right) ^ { T } \in \mathbb { R } ^ { H \times H }\tag{12}
$$

Similarly, in LoRA [Hu et al., 2021] fine-tuning LLaMA-7B, we define the Transformer Kernel Manifold as

$$
\mathbf { G } _ { 1 } = \mathbf { W } _ { 1 } ^ { e f f } \left( \mathbf { W } _ { 1 } ^ { e f f } \right) ^ { T } \in \mathbb { R } ^ { H \times H }\tag{13}
$$

where ${ \bf W } _ { 1 } ^ { e f f }$ denotes the effective down-projection weight during LoRA fine-tuning. Notably, for both ViT and LLaMA-7B, the final projection layer in KGFT is removed when adapting the module to Transformer architectures.

For optimization, CIFAR-100 models are trained for 200 epochs with a batch size of 512, while ImageNet-1K models are trained for 100 epochs with a batch size of 256. All experiments use stochastic gradient descent (SGD) with a momentum of 0.9 and an initial learning rate of 0.1. The learning rate is scheduled by MultiStepLR, with decay factors of 0.1 applied at 30%, 60%, and 80% of the total training epochs. The weight decay is set to $5 \times 1 0 ^ { - 4 }$ , and label smoothing with a coefficient of 0.01 is adopted for regularization. For the arithmetic reasoning experiments, we fine-tune LLaMA-7B on the Math10K training set using LoRA for 3 epochs. We use the AdamW optimizer with an initial learning rate of $3 \times 1 0 ^ { - 4 }$ and a batch size of 16. The learning rate follows a linear decay schedule with 100 warm-up steps.

Table 1: Top-1 classification accuracy (%) on CIFAR-100 across eight different random seeds for different ResNet architectures and KGFT variants. Mean, standard deviation, best, and worst accuracies are reported.
<table><tr><td>Model</td><td>Methods</td><td>Params</td><td>Mean</td><td>Std</td><td>Best</td><td>Worst</td></tr><tr><td rowspan="2">ResNet-20</td><td>Baseline</td><td>0.27M</td><td>67.61</td><td>0.28</td><td>68.11</td><td>67.21</td></tr><tr><td>+ KGFT</td><td>0.28M</td><td>69.39</td><td>0.16</td><td>69.60</td><td>69.18</td></tr><tr><td rowspan="2">ResNet-32</td><td>Baseline</td><td>0.47M</td><td>69.69</td><td>0.36</td><td>70.09</td><td>68.98</td></tr><tr><td>+ KGFT</td><td>0.48M</td><td>70.85</td><td>0.55</td><td>71.80</td><td>69.76</td></tr><tr><td rowspan="2">ResNet-18</td><td>Baseline</td><td>11.15M</td><td>74.60</td><td>0.79</td><td>75.38</td><td>72.78</td></tr><tr><td>+ KGFT</td><td>11.48M</td><td>76.84</td><td>0.26</td><td>77.25</td><td>76.50</td></tr></table>

Table 2: Top-1 classification accuracy (%) on CIFAR-100 across eight different random seeds for ViT-tiny architectures and KGFT. Mean, standard deviation, best, and worst accuracies are reported.
<table><tr><td>Model</td><td>Methods</td><td>Params</td><td>Mean</td><td>Std</td><td>Best</td><td>Worst</td></tr><tr><td>ViT-tiny</td><td>Baseline</td><td>5.54M</td><td>54.33</td><td>0.18</td><td>54.42</td><td>54.11</td></tr><tr><td>ViT-tiny</td><td>+ KGFT</td><td>5.54M</td><td>54.96</td><td>0.48</td><td>55.20</td><td>54.58</td></tr></table>

For KGFT, we employ the proposed dual-mode scheduling strategy, where Exploit guidance is applied to shallow stages to preserve discriminative local structures, while Explore guidance is introduced in the final stage to enhance global feature exploration. The learnable-strength parameter is optimized with a smaller learning rate (0.5× the base learning rate) and a 50-epoch warm-up strategy.

To ensure reliable evaluation, all CIFAR-100 experiments are repeated using eight different random seeds, ImageNet-1K experiments are conducted with four independent runs, and MATH10K experiments are employed with four independent runs. The reported results are averaged across multiple runs.

## 4.3 Results

## 4.3.1 Image Classification on CIFAR-100

Table 1 presents the Top-1 accuracy results of KGFT on CIFAR-100 across different ResNet architectures. All results are obtained using eight different random seeds, with the mean accuracy, standard deviation, best accuracy, and worst accuracy reported. Compared with the standard baselines, KGFT consistently improves classification performance with negligible parameter overhead. Specifically, KGFT improves the mean accuracy by 1.78, 1.16, and 2.24 percentage points on ResNet20, ResNet32, and ResNet18, respectively. Notably, on ResNet18, KGFT improves the mean accuracy from 74.60 to 76.84 while reducing the standard deviation from 0.79 to 0.26, indicating substantially improved optimization stability across different random initializations. These consistent improvements demonstrate the effectiveness of adaptively incorporating geometric guidance into representation learning. The results further suggest that the geometric information captured by the kernel and data manifolds provides beneficial structural guidance for feature learning, while the learnable guidance strength allows the network to adaptively regulate the contribution of geometric transformation during optimization.

We also apply KGFT to the ViT-tiny. As shown in Table 2, KGFT improves the mean Top-1 accuracy from 54.33 to 54.96, yielding a 0.63 percentage-point gain with no additional trainable parameters. Although the improvement is more modest than that observed on the ResNet architectures, the consistent accuracy gain demonstrates that manifold-guided feature transformation can also benefit Transformer-based visual representations. Overall, the results support the central idea of KGFT: explicitly coupling kernel and data manifold geometry can provide effective complementary guidance for representation learning, and a learnable guidance strength enables the network to adaptively exploit this geometric information according to its evolving optimization state.

Table 3: Top-1 classification accuracy (%) on ImageNet-1K across four different random seeds for different ResNet architectures and learnable-strength KGFT. Mean, standard deviation, best, and worst accuracies are reported.
<table><tr><td>Model</td><td>Methods</td><td>Params</td><td>Mean</td><td>Std</td><td>Best</td><td>Worst</td></tr><tr><td>ResNet-34</td><td>Baseline</td><td>20.79M</td><td>73.57</td><td>0.18</td><td>73.84</td><td>73.46</td></tr><tr><td>ResNet-34</td><td>+ KGFT</td><td>21.20M</td><td>74.07</td><td>0.48</td><td>74.55</td><td>73.41</td></tr><tr><td>ResNet-50</td><td>Baseline</td><td>24.37M</td><td>75.43</td><td>0.31</td><td>75.70</td><td>75.06</td></tr><tr><td>ResNet-50</td><td>+ KGFT</td><td>24.79M</td><td>76.58</td><td>0.16</td><td>76.81</td><td>76.42</td></tr></table>

Table 4: Accuracy (%) on four benchmarks across four different random seeds for LLaMA-7B and learnable-strength KGFT. \*Performance results of all baseline methods are taken from Wu et al [Touvron et al., 2023]. Mean, standard deviation, best, and worst accuracies are reported.
<table><tr><td>Model</td><td>Methods</td><td>Params (%)</td><td>Testsets</td><td>Mean</td><td>Std</td><td>Best</td><td>Worst</td></tr><tr><td>LLaMA-7B</td><td>Baseline (LoRA)</td><td>0.8256M</td><td>GSM8K</td><td>37.50</td><td></td><td></td><td>一</td></tr><tr><td>LLaMA-7B</td><td>Baseline (LoRA)</td><td>0.8256M</td><td>MAWPS</td><td>79.00</td><td></td><td></td><td></td></tr><tr><td>LLaMA-7B</td><td>Baseline (LoRA)</td><td>0.8256M</td><td>SVAMP</td><td>52.10</td><td></td><td></td><td></td></tr><tr><td>LLaMA-7B</td><td>Baseline (LoRA)</td><td>0.8256M</td><td>AQuA</td><td>18.90</td><td></td><td></td><td></td></tr><tr><td>LLaMA-7B</td><td>+ KGFT (L9)</td><td>0.8256M</td><td>GSM8K</td><td>38.32</td><td>0.64</td><td>39.42</td><td>37.83</td></tr><tr><td>LLaMA-7B</td><td>+ KGFT (L9)</td><td>0.8256M</td><td>MAWPS</td><td>82.46</td><td>0.35</td><td>82.77</td><td>81.93</td></tr><tr><td>LLaMA-7B</td><td>+ KGFT (L9)</td><td>0.8256M</td><td>SVAMP</td><td>53.83</td><td>1.76</td><td>56.30</td><td>51.60</td></tr><tr><td>LLaMA-7B</td><td>+ KGFT (L9)</td><td>0.8256M</td><td>AQuA</td><td>15.95</td><td>2.37</td><td>19.29</td><td>12.60</td></tr><tr><td>LLaMA-7B</td><td>+ KGFT (L9/L21/L32)</td><td>0.8256M</td><td>GSM8K</td><td>37.91</td><td>0.66</td><td>38.74</td><td>37.00</td></tr><tr><td>LLaMA-7B</td><td>+ KGFT (L9/L21/L33)</td><td>0.8256M</td><td>MAWPS</td><td>82.77</td><td>1.36</td><td>84.87</td><td>81.09</td></tr><tr><td>LLaMA-7B</td><td>+ KGFT (L9/L21/L32)</td><td>0.8256M</td><td>SVAMP</td><td>53.30</td><td>1.69</td><td>55.20</td><td>50.70</td></tr><tr><td>LLaMA-7B</td><td>+ KGFT (L9/L21/L32)</td><td>0.8256M</td><td>AQuA</td><td>16.93</td><td>2.21</td><td>19.69</td><td>14.57</td></tr></table>

## 4.3.2 Image Classification on ImageNet-1K

Table 3 further evaluates KGFT on the more challenging ImageNet-1K benchmark using ResNet-34 and ResNet-50. KGFT consistently improves the Top-1 accuracy over the corresponding baselines, with gains of 0.50 and 1.15 percentage points on ResNet-34 and ResNet-50, respectively. The larger improvement on ResNet-50 suggests that the benefit of geometric guidance becomes more evident as the representation space becomes more complex with increasing model capacity. KGFT also improves the robustness of training, particularly for ResNet-50, where the standard deviation decreases from 0.31 to 0.16 and the worst-case accuracy increases from 75.06 to 76.42. These results demonstrate that the effectiveness of KGFT is not limited to small-scale CIFAR-100 experiments but extends to large-scale ImageNet-1K classification. More importantly, the consistent gains across architectures support our underlying hypothesis that kernel and data-manifold geometry provides complementary structural information beyond conventional feature learning, while adaptive coupling allows this information to be effectively integrated into networks with different representation capacities. Fig. 1 also presents Grad-CAM activation maps of ResNet-34 and ResNet-50 with and without KGFT. Compared with the baselines, KGFT produces more localized and class-relevant responses, with reduced activation on irrelevant backgrounds. This observation suggests that kernel-guided geometric transformation promotes more discriminative feature representations.

![](images/0b956ff6afbe00fc97d421d99a506e0d18571981c0a0f1efd532d0a8744129a3.jpg)  
Figure 1: Grad-CAM [Selvaraju et al., 2017] visualization of baseline ResNet models and their KGFT-enhanced counterparts on ImageNet-1K. KGFT generally produces more localized responses on class-relevant regions while suppressing irrelevant background activations.

## 4.3.3 Arithmetic Reasoning with LLaMA-7B

To investigate whether the proposed learnable strength KGFT can generalize beyond convolutional architectures, we extend KGFT to the Transformer-based LLaMA-7B model and evaluate its effectiveness on arithmetic reasoning tasks. We fine-tune LLaMA-7B on MATH10K, a combined training set constructed from seven arithmetic reasoning datasets with language-model-generated chain-of-thought rationales. We report results on four mathematical reasoning benchmarks: AQuA [Ling et al., 2017], GSM8K [Cobbe et al., 2021], MAWPS [Koncel-Kedziorski et al., 2016], and SVAMP [Patel et al., 2021]. Models need to generate chain-of-thought [Wei et al., 2022] before the final answer. Following standard evaluation practice, only the correctness of the final numerical or multiple-choice answer is considered when computing accuracy.

We adopt LoRA as the parameter-efficient fine-tuning baseline and follow the training configuration of Wu et al. [Wu et al., 2024]. LoRA represents task-specific weight updates using low-rank matrices while keeping the pretrained model parameters frozen. LLaMA-7B consists of 32 Transformer decoder blocks. We evaluate two KGFT insertion strategies. In the first setting, a single KGFT module in exploit mode is inserted into decoder block 9. In the second setting, KGFT modules are inserted into decoder blocks 9, 21, and 32, using exploit modes.

As shown in Table 4, KGFT generalizes beyond CNN architectures to Transformer-based large language models. On GSM8K, MAWPS, and SVAMP, both single-layer and multi-layer KGFT outperform the LoRA baseline. Specifically, single-layer KGFT yields improvements of 0.82, 3.46, and 1.73, respectively, while multi-layer KGFT achieves gains of 0.41, 3.77, and 1.20. These results demonstrate that kernel-guided geometric adaptation can effectively guide hidden representations in language models, suggesting that the proposed dual-manifold interaction is not restricted to convolutional feature spaces. Nevertheless, the gains vary across reasoning benchmarks. In particular, KGFT achieves lower mean accuracy than LoRA on AQuA, indicating that the effectiveness of geometric guidance is task-dependent rather than uniformly beneficial. This variation may arise from differences in task-specific data distributions and reasoning characteristics, highlighting the need for adaptive geometric guidance across different reasoning tasks.

## 5 Conclusion

Deep representation learning has predominantly focused on the geometry of intermediate features, while overlooking the structured geometric information embedded in network parameters. In this work, we introduced a dual-manifold perspective that jointly characterizes the Kernel Manifold induced by network parameters and the Data Manifold formed by feature representations. Specifically, we proposed Kernel-Guided Feature Transform (KGFT), which explicitly transfers kernel geometry to feature space through a kernel-induced guidance matrix. Unlike conventional attention mechanisms that primarily reweight feature responses according to their estimated importance, KGFT explicitly reshapes feature relationships by transferring parameter-induced geometric structure into the data manifold. By combining Exploit and Explore modes with adaptive guidance strength and depth-aware scheduling, KGFT enables stable geometric alignment in shallow layers and promotes feature diversity in deeper layers, providing a lightweight and general mechanism for parameter-guided representation learning.

Extensive experiments across CNN- and Transformer-based architectures, including ResNet, ViT, and LLaMA-7B, demonstrate that KGFT consistently improves representation quality across image classification and arithmetic reasoning tasks with negligible parameter overhead. Theoretical analysis further establishes the validity, stability, and controllability of the proposed geometric transformation. These results suggest that network parameters are not merely optimization variables, but also contain structured geometric priors that can be explicitly exploited to complement conventional feature-centric representation learning.

In future work, we will further explore the application of KGFT to a broader range of downstream tasks, such as object detection, semantic segmentation, and parameter-efficient fine-tuning, to investigate whether kernel-guided geometry can provide consistent benefits beyond classification. More broadly, the dual-manifold principle offers a promising direction for cross-modal manifold alignment. For example, the Gram geometry of a text encoder $( \mathbf { G } _ { t e x t } )$ could guide the covariance geometry of image features $( { \bf K } _ { i m g } ) .$ , or vice versa, encouraging the principal directions and local tangent structures of different modalities to become geometrically compatible. Such a formulation could move beyond purely contrastive learning by explicitly aligning higher-order manifold geometry, providing a new perspective for multimodal representation learning.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work the authors used ChatGPT in order to polish the language. After using this tool/service, the authors reviewed and edited the content as needed and took full responsibility for the content of the publication.

## Acknowledgments

This work was supported by a grant from the National Key R&D Program of China (No.2024YFA1012002, No.2018YFC2001203), the General Program of the National Natural Science Foundation of China (No. 62471214), the AI Computing Platform, School of Biomedical Engineering, Southern Medical University.

## References

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neura networks. Advances in neural information processing systems, 25, 2012.

Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.

Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 1–9, 2015.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.

Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pages 6105–6114. PmLR, 2019.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

Yue Liu, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang, Qixiang Ye, Jianbin Jiao, and Yunfan Liu. Vmamba: Visual state space model. Advances in neural information processing systems, 37:103031–103063, 2024.

Geoffrey E Hinton and Richard Zemel. Autoencoders, minimum description length and helmholtz free energy. Advances in neural information processing systems, 6, 1993.

Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781, 2013.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PmLR, 2020.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 15979–15988. IEEE, 2022.

Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7132–7141, 2018.

Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon. Cbam: Convolutional block attention module. In European conference on computer vision, pages 3–19. Springer, 2018.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In 2021 IEEE/CVF international conference on computer vision (ICCV), pages 9992–10002. Ieee, 2021.

Daniel Beaglehole, Ioannis Mitliagkas, and Atish Agarwala. Feature learning as alignment: a structural property of gradient descent in non-linear neural networks. arXiv preprint arXiv:2402.05271, 2024.

Leonardo Defilippis, Yizhou Xu, Julius Girardin, Vittorio Erba, Emanuele Troiani, Lenka Zdeborová, Bruno Loureiro, and Florent Krzakala. Scaling laws and spectra of shallow neural networks in the feature learning regime. In International Conference on Learning Representations, volume 2026, pages 108893–108934, 2026.

Taehun Cha, Daniel Beaglehole, Adityanarayanan Radhakrishnan, and Donghun Lee. The weight gram matrix captures sequential feature linearization in deep networks. arXiv preprint arXiv:2605.06258, 2026.

Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein. Svcca: Singular vector canonical correlation analysis for deep learning dynamics and interpretability. Advances in neural information processing systems, 30, 2017.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In International conference on machine learning, pages 3519–3529. PMlR, 2019.

Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. IEEE transactions on pattern analysis and machine intelligence, 35(8):1798–1828, 2013.

Uri Cohen, SueYeon Chung, Daniel D Lee, and Haim Sompolinsky. Separability and geometry of object manifolds in deep neural networks. Nature communications, 11(1):746, 2020.

Michael Hauser and Asok Ray. Principles of riemannian geometry in neural networks. Advances in neural information processing systems, 30, 2017.

Alessandro Benfenati and Alessio Marta. A singular riemannian geometry approach to deep neural networks i. theoretical foundations. Neural Networks, 158:331–343, 2023.

Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. Advances in neural information processing systems, 31, 2018.

Stanislav Fort, Gintare Karolina Dziugaite, Mansheej Paul, Sepideh Kharaghani, Daniel M Roy, and Surya Ganguli. Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the neural tangent kernel. Advances in Neural Information Processing Systems, 33:5850–5861, 2020.

Leon A Gatys, Alexander S Ecker, and Matthias Bethge. Image style transfer using convolutional neural networks. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 2414–2423, 2016.

Shu Kong and Charless Fowlkes. Low-rank bilinear pooling for fine-grained classification. In 2017 IEEE conference on computer vision and pattern recognition (CVPR), pages 7025–7034. IEEE, 2017.

Peihua Li, Jiangtao Xie, Qilong Wang, and Wangmeng Zuo. Is second-order information helpful for large-scale visual recognition? In 2017 IEEE International Conference on Computer Vision (ICCV), pages 2089–2097. IEEE, 2017.

Rui Wang, Chen Hu, Xiaoning Song, Xiaojun Wu, Nicu Sebe, and Ziheng Chen. Towards a general attention framework on gyrovector spaces for matrix manifolds. Advances in Neural Information Processing Systems, 38:112051–112091, 2026.

Peihua Li, Jiangtao Xie, Qilong Wang, and Zilin Gao. Towards faster training of global covariance pooling networks by iterative matrix square root normalization. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 947–955. IEEE, 2018.

Bernhard Schölkopf, Alexander Smola, and Klaus-Robert Müller. Nonlinear component analysis as a kernel eigenvalue problem. Neural computation, 10(5):1299–1319, 1998.

Pascal Esser, Maximilian Fleissner, and Debarghya Ghoshdastidar. Non-parametric representation learning with kernels. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 11910–11918, 2024.

Xiang Li, Wenhai Wang, Xiaolin Hu, and Jian Yang. Selective kernel networks. In 2019 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 510–519. IEEE, 2019.

Qilong Wang, Banggu Wu, Pengfei Zhu, Peihua Li, Wangmeng Zuo, and Qinghua Hu. Eca-net: Efficient channel attention for deep convolutional neural networks. In 2020 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 11531–11539. IEEE, 2020.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

Terrance DeVries and Graham W Taylor. Improved regularization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552, 2017.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D Manning, and Christopher Potts. Reft: Representation finetuning for language models. Advances in Neural Information Processing Systems, 37: 63908–63962, 2024.

Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In Proceedings of the 55th annual meeting of the association for computational linguistics (volume 1: Long papers), pages 158–167, 2017.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. Mawps: A math word problem repository. In Proceedings of the 2016 conference of the north american chapter of the association for computational linguistics: human language technologies, pages 1152–1157, 2016.

Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word problems? In Proceedings ofthe 2021 conference ofthe North American chapter ofthe associationfor computational linguistics: human language technologies, pages 2080–2094, 2021.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision, pages 618–626, 2017.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chainof-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

## 6 Appendix A. Theoretical Analysis

## 6.1 The Geometric Interpretation of Exploit and Explore Guidance Matrices

Exploit Geometric Interpretation: Stretch the data manifold along the principal direction ofthe kernel manifold. Let the eigenvalue decomposition ofG be $\mathbf { G } = \textstyle \sum _ { i = } ^ { C }$ λ<sub>i</sub>u<sub>i</sub>u<sup>T</sup>; then, the scaling factor of $\mathbf { M _ { e x p l o i t } }$ along the direction $\mathbf { u } _ { i }$ is $\lambda _ { i } + \varepsilon -$ where the direction with the longest kernel manifold stretch $( i . e .$ , the direction for which is largest) corresponds to the greatest stretching ofthe data manifold.

Explore Geometric Interpretation:Pushing the data manifold awayfrom the principal directions ofthe kernel manifold. Let $\begin{array} { r } { \mathbf { C } = \sum _ { i = 1 } ^ { C } \nu _ { i } \mathbf { v } _ { i } \mathbf { v } _ { i } ^ { \top } } \end{array}$ be the eigenvalue decomposition ofC; then, the scalingfactor of $\mathbf { M _ { e x p l o r e } }$ along direction $\mathbf { v } _ { i }$ is given by $\dot { s } ( \bar { 1 } - { \mathbf v } _ { i } ) + \varepsilon -$ where directions on the kernel manifold with stronger correlations $( i . e . , \mathbf { v } _ { i } )$ have smaller scaling factors.

## 6.2 Positive Definiteness of the Kernel-Induced Guidance

Proposition 1 (Positive Definiteness). Let $\mathbf { G } = \mathbf { W } \mathbf { W } ^ { \top }$ be the kernel Gram matrix and let $\varepsilon > 0$ . Then both $\mathbf { G } + \varepsilon \mathbf { I } _ { C }$ and $\lambda \cdot ( \mathbf { I } _ { C } - \mathbf { C } ) + \varepsilon \mathbf { I } _ { C }$ where $\mathbf { C } = \mathbf { D } ^ { - 1 / 2 } \mathbf { G } \mathbf { D } ^ { - 1 / 2 }$ are symmetric positive definite (SPD).

Proof. Since $\mathbf { G } = \mathbf { W } \mathbf { W } ^ { \top }$ for any nonzero vector (v):

$$
\mathbf { v } ^ { \top } \mathbf { G } \mathbf { v } = \mathbf { v } ^ { \top } \mathbf { W } \mathbf { W } ^ { \top } \mathbf { v } = \| \mathbf { W } ^ { \top } \mathbf { v } \| _ { 2 } ^ { 2 }\tag{14}
$$

Therefore, $\mathbf G > \mathbf 0$ . For Exploit mode:

$$
\mathbf { v } ^ { \top } \mathbf { M } _ { \mathrm { e x p l o i t } } \mathbf { v } = \mathbf { v } ^ { \top } \mathbf { G } \mathbf { v } + \varepsilon \parallel \mathbf { v } \parallel _ { 2 } ^ { 2 }\tag{15}
$$

Because $\varepsilon \ > \ 0$ and $\mathrm { ~ \bf ~ v ~ } \ne \mathrm { ~ \bf ~ 0 ~ }$ , so $\mathbf { v } ^ { \top } \mathbf { M } _ { \mathrm { e x p l o i t } } \mathbf { v } \ > \ \mathrm { ~ 0 ~ }$ . For Explore mode, let $\mathbf { G } \ = \ \mathbf { U } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { U } ^ { \mathrm { T } }$ , where $\Lambda \ =$ $\mathrm { d i a g } ( \lambda _ { 1 } , \dots , \lambda _ { \mathrm { C } } ) , 0 \leq \lambda _ { i } \leq \lambda _ { \mathrm { m a x } } ( \mathbf { G } )$ . The normalized matrix $\mathbf { C } = \mathbf { D } ^ { - 1 / 2 } \mathbf { G } \mathbf { D } ^ { - 1 / 2 }$ has eigenvalues:

$$
\gamma _ { i } = \frac { \lambda _ { i } } { \lambda _ { \operatorname* { m a x } } ( \mathbf { G } ) + \varepsilon }\tag{16}
$$

Since $\varepsilon > 0$ , we get $0 \leq \lambda _ { i } \leq 1$ . Therefore, $\mathbf { 0 } < \mathbf { C } < \mathbf { I } ,$ and hence $\mathbf { \left( I - C \right) > 0 }$ . For any nonzero vector (v):

$$
\mathbf { v } ^ { \top } \mathbf { M } _ { \mathrm { e x p l o r e } } \mathbf { v } = { \boldsymbol { \delta } } \cdot \mathbf { v } ^ { \top } ( \mathbf { I } - \mathbf { C } ) \mathbf { v } + \varepsilon \parallel \mathbf { v } \parallel _ { 2 } ^ { 2 } > 0\tag{17}
$$

Conclusion: This result directly justifies the guidance matrices introduced in Section 3.2.1. The SPD property guarantees that the kernel-induced operator defines a valid positive-definite metric in the shared channel space. Therefore, KGFT does not apply an arbitrary or potentially indefinite feature transformation; instead, its geometric modulation is induced by a mathematically valid metric.

## 6.3 Lipschitz Stability of Kernel-Guided Mapping

The kernel guidance depends on the network parameters through:

$$
\mathbf { W } \to \mathbf { G } = \mathbf { W } \mathbf { W } ^ { \top } \to \mathbf { M } ( \mathbf { G } ) \to \mathbf { F } ( \mathbf { X } , \mathbf { M } )\tag{18}
$$

Because network parameters continuously change during optimization, it is desirable that small parameter changes do not result in abrupt changes in the induced geometric transformation.

Proposition 2 (Lipschitz Stability). Assume that the kernel matrix satisfies: $\| \mathbf { W } \| _ { 2 } \le B _ { W }$ . Then the kernel Gram mapping $\mathbf { W } \to \mathbf { G } = \mathbf { W } \mathbf { W } ^ { \top }$ is Lipschitz continuous. Specifically, given two kernel matrices $\mathbf { w } _ { 1 }$ and $\mathbf { w } _ { 1 }$ , we get $\begin{array} { r } { \| \dot { \bf G } _ { 1 } ^ { \star } - \mathbf { \bar { G } } _ { 2 } \| _ { 2 } \leq 2 B _ { W } \| { \bf W } _ { 1 } - { \bf W } _ { 2 } \| _ { 2 } ^ { \star } } \end{array}$ . Under bounded kernel scale and $\varepsilon > 0$ , both the Exploit and Explore guidance matrices are Lipschitz continuous with respect to W. Consequently, the resulting KGFT transformation is continuous with respect to perturbations of the kernel parameters.

Proof. Given two kernel matrices, we calculate the difference: $\mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } - \mathbf { W } _ { 2 } \mathbf { W } _ { 2 } ^ { \top }$ . Adding and subtracting $( \mathbf { W } _ { 1 } \mathbf { W } _ { 2 } ^ { \top } )$ ), we obtain:

$$
\mathbf { W } _ { 1 } ( \mathbf { W } _ { 1 } - \mathbf { W } _ { 2 } ) ^ { \top } + ( \mathbf { W } _ { 1 } - \mathbf { W } _ { 2 } ) \mathbf { W } _ { 2 } ^ { \top }\tag{19}
$$

Taking the spectral norm gives:

$$
\begin{array} { r } { \| \mathbf { G } _ { 1 } - \mathbf { G } _ { 2 } \| _ { 2 } \leq \| \mathbf { W } _ { 1 } \| _ { 2 } \| \mathbf { W } _ { 1 } - \mathbf { W } _ { 2 } \| _ { 2 } + \| \mathbf { W } _ { 1 } - \mathbf { W } _ { 2 } \| _ { 2 } \| \mathbf { W } _ { 2 } \| _ { 2 } } \end{array}\tag{20}
$$

Since $\| \mathbf { W } _ { 1 } \| _ { 2 } , \| \mathbf { W } _ { 1 } \| _ { 2 } \le B _ { W }$ , we obtain:

$$
\| \mathbf { G } _ { 1 } - \mathbf { G } _ { 2 } \| _ { 2 } \leq 2 B _ { W } \| \mathbf { W } _ { 1 } - \mathbf { W } _ { 2 } \| _ { 2 }\tag{21}
$$

Thus, the kernel geometry changes continuously with the network parameters. Since the subsequent projection and normalization operations are continuous under standard boundedness assumptions, the complete KGFT mapping is also continuous with respect to W.

## 6.4 Bounded and Identity-Preserving Geometric Modulation

The learnable guidance strength introduced in Section 3.2.3 controls the contribution of the geometrically transformed branch:

$$
\mathbf { Y } = \mathbf { X } + \mathbf { s } \cdot { \mathcal { F } } _ { K G F T } ( \mathbf { X } )\tag{22}
$$

The following proposition explains why this residual formulation provides a bounded and controllable geometric modification.

Proposition 3 (Boundedness and Identity Preservation). Assume that the kernel-induced metric satisfies: $\| \mathbf { M } \| _ { 2 } \leq$ $B _ { M }$ , and that the geometric transformation branch satisfies: $\| \mathbf { F } ( \mathbf { X } , \mathbf { M } ) \| _ { F } \leq B _ { F } \| \mathbf { X } \| _ { F }$ , for some finite constant $( \le B _ { F } )$ . Then the KGFT output satisfies: $\lVert \mathbf { Y } \rVert _ { F } \leq ( 1 + \mathbf { s } B _ { F } ) \ddot { \lVert \mathbf { X } \rVert _ { F } }$ . Moreover, because $0 < \mathbf { s } < 1$ , the geometric branch cannot dominate the residual pathway solely through an unconstrained guidance coefficient. In the limiting case $s  0 .$ , KGFT converges to the identity mapping: $\mathbf { \dot { Y } } \to \breve { \mathbf { X } }$

Proof. Let $\mathbf { Y } = \mathbf { X } + \mathbf { s } \cdot { \mathcal { F } } _ { K G F T } ( \mathbf { X } )$ , the triangle inequality gives:

$$
\| \mathbf { Y } \| _ { F } \leq \| \mathbf { X } \| _ { F } + \mathbf { s } \cdot \mathcal { F } _ { K G F T } ( \mathbf { X } )\tag{23}
$$

Using the assumed bound: $\| \mathbf { F } ( \mathbf { X } , \mathbf { M } ) \| _ { F } \leq B _ { F } \| \mathbf { X } \| _ { F } .$ , we get:

$$
\| \mathbf { Y } \| _ { F } \leq \| \mathbf { X } \| _ { F } + \mathbf { s } \cdot B _ { F } \| \mathbf { X } \| _ { F }\tag{24}
$$

and therefore obtain:

$$
\| \mathbf { Y } \| _ { F } \leq ( 1 + \mathbf { s } \cdot B _ { F } ) \| \mathbf { X } \| _ { F }\tag{25}
$$

Since $s = \sigma ( \alpha )$ , we have $0 < \mathbf { s } < 1$ . Thus, the contribution of the geometric branch is explicitly bounded by its transformation magnitude rather than being controlled by an unconstrained scalar. Finally, when $\mathbf { s } > 0 , \mathbf { X } + \mathbf { s }$ $\mathcal { F } _ { K G F T } ( \mathbf { X } ) \to \mathbf { X }$ . Therefore, KGFT contains the identity mapping as a limiting case.

Conclusion: The role of s is not simply to introduce another learnable parameter; it provides an explicit mechanism for controlling the strength of geometric modulation. The identity-preserving property is particularly important in deep networks because it allows the optimization process to suppress the geometric branch when the injected kernel structure is not beneficial for a particular layer. Consequently, KGFT acts as an adaptive residual geometric modulation rather than a mandatory replacement of the original feature representation.

## 7 Appendix B. Dual-manifold scheduling configuration

Table 5 summarizes the detailed KGFT scheduling configurations across different architectures. For ResNet variants, KGFT is sparsely inserted into selected blocks across different stages, with Exploit generally used in shallow and intermediate stages and Explore introduced in deeper stages. For Transformer architectures, KGFT is similarly inserted into selected layers following the depth-aware scheduling principle. All KGFT modules use a learnable guidance strength. The specific insertion positions, operating modes, and guidance settings are provided in Table 5.

Table 5: The detailed configurations of KGFT scheduling strategy on ResNet and Transformer architectures.
<table><tr><td>Architecture</td><td>Stages</td><td>Blocks</td><td>KGFT Mode</td><td>Position</td><td>Strength</td></tr><tr><td rowspan="4">ResNet-18</td><td>Stage1</td><td>2</td><td>Exploit</td><td>Block 2</td><td>Learnable</td></tr><tr><td>Stage2</td><td>2</td><td>Exploit</td><td>Block 2</td><td>Learnable</td></tr><tr><td>Stage3</td><td>2</td><td>Exploit</td><td>Block 2</td><td>Learnable</td></tr><tr><td>Stage4</td><td>2</td><td>Explore</td><td>Block 2</td><td>Learnable</td></tr><tr><td rowspan="3">ResNet-20</td><td>Stage1</td><td>3</td><td>Exploit</td><td>Block 3</td><td>Learnable</td></tr><tr><td>Stage2</td><td>3</td><td>Exploit</td><td>Block 3</td><td>Learnable</td></tr><tr><td>Stage3</td><td>3</td><td>Explore</td><td>Block 3</td><td>Learnable</td></tr><tr><td rowspan="3">ResNet-32</td><td>Stage1</td><td>5</td><td>Exploit</td><td>Block 3 &amp;5</td><td>Learnable</td></tr><tr><td>Stage2</td><td>5</td><td>Exploit</td><td>Block 3&amp;5</td><td>Learnable</td></tr><tr><td>Stage3</td><td>5</td><td>Explore</td><td>Block 3 &amp;5</td><td>Learnable</td></tr><tr><td rowspan="4">ResNet-34</td><td>Stagel</td><td>3</td><td>Exploit</td><td>Block 3</td><td>Learnable</td></tr><tr><td>Stage2</td><td>4</td><td>Exploit</td><td>Block 2 &amp; 4</td><td>Learnable</td></tr><tr><td>Stage3</td><td>6</td><td>Exploit</td><td>Block 3&amp;6</td><td>Learnable</td></tr><tr><td>Stage4</td><td>3</td><td>Explore</td><td>Block 3</td><td>Learnable</td></tr><tr><td rowspan="4">ResNet-50</td><td>Stage1</td><td>3</td><td>Exploit</td><td>Block 3</td><td>Learnable</td></tr><tr><td>Stage2</td><td>4</td><td>Exploit</td><td>Block 2 &amp; 4</td><td>Learnable</td></tr><tr><td>Stage3</td><td>6</td><td>Exploit</td><td>Block 3&amp;6</td><td>Learnable</td></tr><tr><td>Stage4</td><td>3</td><td>Explore</td><td>Block 3</td><td>Learnable</td></tr><tr><td rowspan="3">ViT-tiny</td><td></td><td>L12</td><td>Exploit</td><td>L4</td><td>Learnable</td></tr><tr><td></td><td>L12</td><td>Exploit</td><td>L8</td><td>Learnable</td></tr><tr><td></td><td>L12</td><td>Explore</td><td>L12</td><td>Learnable</td></tr><tr><td rowspan="3">LLaMA-7B</td><td></td><td>L32</td><td>Exploit</td><td>L9</td><td>Learnable</td></tr><tr><td></td><td>L32</td><td>Exploit</td><td>L21</td><td>Learnable</td></tr><tr><td></td><td>L32</td><td>Exploit</td><td>L32</td><td>Learnable</td></tr></table>

Figure 2 provides a visual illustration of KGFT integration into ResNet and ViT. In ResNet, KGFT is inserted within selected residual blocks and fused with the residual pathway, while in ViT, it is applied after the MLP sublayer and incorporated through the residual connection. These implementations preserve the original backbone architecture while enabling KGFT to function as a lightweight plug-and-play module.

![](images/75408096fa4c66afde2461c6b8fab2a79b39ffdd81109cca78c2648542f13d32.jpg)  
Figure 2: Overview of KGFT applied to ResNet and ViT architectures. The upper panels illustrate the overall architectures of ResNet and ViT equipped with KGFT, where KGFT is selectively inserted into representative network blocks according to the proposed depth-aware dual-manifold scheduling strategy.