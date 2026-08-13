# CoQui: A Coordinate-Conditioned Quantum Implicit Generative Adversarial Network for End-to-End Image Generation

Xue Yang<sup>1,2,3,4</sup>, Rigui Zhou<sup>1,2∗</sup>, ShiZheng Jia<sup>1,2</sup>, Dax Enshan Koh<sup>3,5∗</sup>, Siong Thye Goh<sup>4,6∗</sup>, Young-Wook Cho<sup>3</sup>, YaoChong Li<sup>1,2</sup>, Xuezhi Ma<sup>3</sup>, Hongyu Chen<sup>7</sup>, Xin Wang<sup>8</sup>

<sup>1</sup>School of Information Engineering, Shanghai Maritime University, Shanghai 201306, China

<sup>2</sup>Research Center of Intelligent Information Processing and Quantum Intelligent Computing, Shanghai 201306, China <sup>3</sup>Quantum Innovation Centre (Q.InC), Agency for Science, Technology and Research (A\*STAR), 2 Fusionopolis Way, Innovis, #08-03, Singapore 138634, Republic of Singapore

<sup>4</sup>Institute of Advanced Intelligence and Computing (IAIC), Agency for Science, Technology and Research (A\*STAR), 1 Fusionopolis Way, #16-16 Connexis, Singapore 138632, Republic of Singapore

<sup>5</sup>Engineering Cluster, Singapore Institute of Technology, 1 Punggol Coast Road, Singapore 828608, Republic of Singapore <sup>6</sup>Lee Kong Chian School of Business, Singapore Management University (SMU), Singapore 178899, Republic of Singapore <sup>7</sup>School of Computer Science and Technology, Tongji University, Shanghai 201804, China

<sup>8</sup>Department of Automation, Tsinghua University, Beijing 100084, P. R. China rgzhou@shmtu.edu.cn, dax.koh@singaporetech.edu.sg, goh\_siong\_thye@a-star.edu.sg

## Abstract

Quantum generative adversarial networks (QGANs) have recently attracted increasing attention for image generation, with the goal of using parameterized quantum circuits to model and generate image distributions. Existing approaches commonly follow a generation paradigm in which quantum-state amplitudes are mapped to pixel intensities. However, this paradigm faces two key limitations. First, existing methods typically encode pixel locations using computational-basis indices or address qubits, causing the required quantum-state dimension or number of address qubits to grow with image resolution. Second, existing methods jointly decode a large number of pixels from one or a few normalized quantum states. As a result, pixels compete for probability mass, making it dificult to control individual pixel values precisely. To address these limitations, we reformulate quantum image generation as coordinate-conditioned implicit function learning. The proposed method takes spatial coordinates and latent variables as inputs, uses a classical embedding network to generate quantum circuit parameters, and evaluates a variational quantum circuit at each coordinate location. Pixel intensities are directly read out from the expectation value of a dedicated color qubit, and a complete image is obtained by querying all spatial coordinates. This design decouples image resolution from the number of address qubits. Furthermore, our method reads out individual pixel values from separate coordinate-conditioned quantum-state evaluations, so diferent pixels are not required to satisfy a shared probability-normalization constraint, enabling more flexible pixel-wise modeling. To better support this coordinate-conditioned generation paradigm, we further design a specialized variational quantum circuit that intro duces efective structural inductive bias for image generation. Simulated experiments on two benchmark datasets show that our method achieves better visual quality and quantitative per formance than FRQI-based generation and PQWGAN base lines while using fewer qubits. Moreover, compared with the

corresponding classical baseline, the proposed model demonstrates superior generation quality and qubit eficiency.

## Introduction

Recent studies have shown that QGANs have attracted increasing attention and achieved promising results in image generation tasks. Existing quantum generative adversarial network (QGAN) methods for image generation commonly adopt a generation paradigm in which pixel intensities are mapped from quantum-state amplitudes.

As one of the earliest explorations in QGAN-based image generation, Huang et al. utilized quantum-state amplitudes to represent pixel values and constructed low-resolution images through a patch-based generation strategy (Huang et al. 2021). PQWGAN follows this strategy and further extends it to the high-resolution image generation setting (Tsang et al. 2023). Although patch-based generation provides a practical way to mitigate the scalability challenge, it generates an image through multiple independently produced patches rather than modeling the entire image within a single quantum circuit. Jäger et al. recently proposed a quantum Wasserstein GAN with spatial structural inductive bias for end-to-end high-resolution image generation (Jäger, Kiwit, and Riofrío 2026). By encoding pixel positions with address qubits and pixel values with color qubits, the model generates the entire image using a single quantum circuit.

The generation paradigm based on mapping quantum-state amplitudes to pixel intensities has two major limitations. First, under this paradigm, the number of required qubits increases together with image resolution. Second, existing methods jointly recover multiple pixel values from one or a few normalized quantum states. As a result, pixels compete for probability mass, making it dificult to control individual pixel values precisely. In PQWGAN, this issue manifests as generated images whose overall brightness is lower than that of the real data distribution. It can also help explain the uneven address-amplitude distribution observed in the model of Jäger et al.

![](images/027ee8d65e2dd30b4c07551225cba3c68b1c8ea1cc6331391973978fbf56d603.jpg)  
Figure 1: Overview of the CoQui framework. (a) Overall architecture: Given a spatial coordinate $c = ( x , y )$ and a latent code z, CoQui utilizes a classical embedding network Γ to generate coordinate- and latent-conditioned parameters $A ( c , z ) \overset { \cdot } { \in } \bar { \mathbb { R } } ^ { \bar { N } _ { f } \times 3 }$ . These are transformed into layer-wise scaled parameters $\Theta _ { i , \ell }$ for the quantum generator G. The pixel intensity is computed as $\begin{array} { r } { G ( c , z ) = { \frac { 1 - \langle Z _ { 0 } \rangle } { 2 } } } \end{array}$ by measuring the Pauli-Z expectation of the color qubit. An entire image $\hat { X }$ is synthesized by querying all coordinates. (b) Detailed quantum generator structure: The circuit comprises one color qubit $q _ { 0 }$ and $\bar { N } _ { f }$ feature qubits. It first applies color-qubit brightness initialization to $q _ { 0 }$ . Each of the L repeating layers consists of: (i) scaled data re-uploading, (ii) local feature-qubit rotations, (iii) feature-feature entanglement using a ring topology, (iv) feature-to-color writing via controlled- $R _ { Y }$ gates, and (v) residual color-qubit updates. Finally, q<sub>0</sub> gives the pixel readout.

In addition, many QGAN methods that we will discuss in Related Work use the quantum circuit only to model a classical compressed representation of the image, with the quantum circuit not directly responsible for pixel-level generation. In contrast, this work focuses on a paradigm in which the quantum circuit directly undertakes pixel-level image generation.

To address the above issues, we propose CoQui, a coordinate-conditioned quantum implicit GAN. The proposed framework leverages implicit neural representations (INRs) to learn a continuous mapping from spatial coordinates to signal values, thereby decoupling the number of qubits from image resolution. Moreover, unlike existing methods that jointly decode a large number of pixels from the amplitudes of one or a few quantum states, our method separately evaluates a coordinate-conditioned quantum state at each spatial coordinate and reads out the corresponding pixel value, thereby avoiding competition for probability mass among pixels. Specifically, the model takes spatial coordinates and latent variables as inputs, first maps them into the encoding parameters of a parameterized quantum circuit (PQC) through a classical network, and then directly represents continuous pixel values using quantum measurement expectations. Furthermore, this paper designs a quantum circuit architecture with structural inductive bias, where feature qubits controllably modulate a color qubit, allowing the color output to explicitly depend on the feature representation and thereby improving the model’s capability in conditional pixel generation.

Our contributions are summarized as follows:

1. We propose CoQui (Coordinate-conditioned Quantum Implicit GAN), which formulates image generation as a continuous mapping from spatial coordinates to pixel intensities, rather than relying on the conventional amplitude-mapping paradigm. This design decouples the number of qubits from image resolution and mitigates the inter-pixel competition for probability mass inherent in conventional amplitude-based representations.

2. We design a quantum circuit with structural inductive bias, in which feature qubits controllably modulate a color qubit. This mechanism makes the color generation process explicitly dependent on the input feature representation, thereby enhancing the modeling capability of the quantum circuit.

3. We conduct systematic simulation experiments on multiple image generation benchmark datasets. Experimental results show that, compared with existing QGAN methods based on the amplitude-mapping paradigm, CoQui achieves better qualitative and quantitative performance and is able to generate images efectively with fewer qubits.

## Related Work

## Quantum Implicit Neural Representations and Image Generation

In recent years, implicit neural representations (INRs) have attracted substantial attention because of their ability to represent high-quality signals by learning mappings from continuous coordinates to signal values (Mescheder et al. 2019; Sitzmann et al. 2020; Park et al. 2019). Unlike conventional discrete-grid representations (Gonzalez 2009), INRs model images (Chen, Liu, and Wang 2021; Cao et al. 2023), audio (Sitzmann et al. 2020; Su, Chen, and Shlizerman 2022), and three-dimensional scenes (Zhao et al. 2024; Müller et al. 2022) as continuous functions, from which signal values can be recovered through coordinate queries. Inspired by this idea, researchers have begun to investigate Quantum Implicit Neural Representations (QINRs) (Zhao et al. 2024) built with parameterized quantum circuits (PQCs), aiming to exploit the potential of quantum circuits for high-dimensional nonlinear function approximation and Fourier-feature representation (Pérez-Salinas et al. 2020; Schuld, Sweke, and Meyer 2021).

Existing studies on QINRs mainly use them as continuous representers for conditional signals, with applications in deterministic mapping tasks such as image reconstruction (Zhao et al. 2024; Eren 2026; Wang, Theobalt, and Golyanik 2026), compression (Fujihashi and Koike-Akino 2026), and super-resolution (Jin, Singh, and Merz Jr 2025; Zhao et al. 2024). Although Zhang et al. proposed OQIDDM (Zhang et al. 2025), a QINR-based quantum difusion model that introduces QINR into generative modeling, their method relies on amplitude encoding, which causes the required number of qubits to increase with image resolution. Thus, while QINRs show considerable promise for coordinate-dependent function representation, their integration with adversarial distribution learning remains insuficiently explored. To address this gap, this work investigates the feasibility of using QINR for end-to-end image distribution modeling within a Generative Adversarial Network (GAN) framework (Goodfellow et al. 2020).

## Quantum Generative Adversarial Networks for Image Generation

GANs learn real data distributions through a minimax game between a generator and a discriminator. To alleviate training instability and mode collapse, WGAN (Arjovsky, Chintala, and Bottou 2017) employs the Wasserstein distance to provide a smoother optimization objective, while WGAN-GP (Gulrajani et al. 2017) further introduces a gradient penalty to enforce the Lipschitz constraint and improve training stability. This objective has also been adopted by many QGAN models to stabilize adversarial training. Quantum Generative Adversarial Networks (QGANs) extend adversarial generative modeling to the quantum computing framework. Early studies established the theoretical foundations of quantum adversarial learning (Lloyd and Weedbrook 2018) and further used parameterized quantum circuits to implement trainable quantum generative models (Dallaire-Demers and Killoran 2018), with applications to numerical distribution modeling and quantum-state preparation (Zoufal, Lucchi, and

Woerner 2019).

More recently, QGANs have been applied to image generation. However, high-dimensional visual data impose substantial requirements on both the number of qubits and circuit depth. Existing methods therefore typically rely on dimensionality reduction or patch-based generation to alleviate scalability issues. Dimensionality-reduction approaches first compress images into a low-dimensional latent space using PCA or an encoder, then use a quantum generator to model the low-dimensional representation, and finally recover image dimensions through classical post-processing (Chu et al. 2023; Silver et al. 2023; Vieloszynski et al. 2024; Chang et al. 2024; Thomas, Youel, and Jose 2025). Patch-based approaches divide an image into multiple local patches and use one or more quantum generators to synthesize the image blocks, thereby avoiding direct modeling of the full highdimensional image (Huang et al. 2021; Tsang et al. 2023; Yang et al. 2026b). These methods reduce the training dificulty of quantum models, but they also weaken the direct role of the quantum generator in the complete image generation process. In particular, when classical decoders or multiple local generators are used, part of the generation capability may come from classical modules or task-specific structures.

To further improve the scalability of quantum image generation, recent studies have begun to explore end-to-end fullimage quantum generators. Jäger et al. proposed a quantum Wasserstein GAN with a structural inductive bias (Jäger, Kiwit, and Riofrío 2026) . Their design follows the idea of an FRQI-like address-color image representation and generates full-resolution images without relying on dimensionality reduction or patch-based generation. Meanwhile, Yang et al. improved the amplitude decoding and prior-mapping mechanisms ofthe PQWGAN model, enabling end-to-end quantum image generation (Yang et al. 2026a).

We observe that most existing QGAN methods rely on quantum-state amplitudes to represent pixel intensities or feature values. This mechanism is constrained by the global normalization of quantum states and inevitably introduces numerical coupling among pixels, which weakens the model’s ability to independently control local signal values. To address this limitation, this work explores an implicit quantum representation based on coordinate queries, where spatial coordinates and latent variables jointly drive a parameterized quantum circuit and pixel intensities are directly extracted from measurement expectations of observables. This formulation provides a more flexible route toward end-to-end image distribution modeling.

## Method

## Problem Formulation

This work proposes CoQui, a QGAN framework based on Quantum Implicit Neural Representation (QINR). Instead of generating an image by assigning pixel intensities to quantum-state amplitudes, CoQui reformulates image generation as a continuous implicit mapping conditioned jointly on spatial coordinates and a latent variable.

CoQui represents a grayscale image as $X \in [ 0 , 1 ] ^ { H \times W \times 1 }$ and learns a coordinate-conditioned generator $G _ { \Phi } ( c , z ) \in$ [0, 1]. Here, $c = ( x , y ) \in \mathcal { C } \subset [ 0 , 1 ) ^ { 2 }$ is a normalized pixel coordinate, $z \in \mathbb { R } ^ { d _ { z } }$ is a latent code shared across all coordinates of one image, and Φ denotes the trainable parameters. A complete image is obtained by evaluating the generator over $\mathcal { C } _ { H , W } ^ { \mathrm { ~ ~ } } = \{ ( \bar { i } / W , j / H ) \mid 0 \leq i < W , 0 \overset {  } { \leq } j < \overline { { H } } \}$

## Conditional Input to the Quantum Implicit Generator

To improve the ability of coordinate inputs to represent local details and high-frequency structures, the two-dimensional coordinate $c = ( x , y )$ is first transformed by positional encoding. Given the number of frequencies $\dot { K }$ , the encoding function is defined as

$$
\begin{array} { c } { { \gamma ( c ) = \left[ x , y , \{ \sin ( 2 ^ { k } x ) , \cos ( 2 ^ { k } x ) , \right. } } \\ { { \left. \sin ( 2 ^ { k } y ) , \cos ( 2 ^ { k } y ) \} _ { k = 0 } ^ { K - 1 } \right] . } } \end{array}\tag{1}
$$

with encoded coordinate dimension $d _ { \gamma } = 2 + 4 K$ . The encoded coordinate feature is concatenated with the latent variable $z \in \mathbb { R } ^ { d _ { z } }$ :

$$
u ( c , z ) = [ \gamma ( c ) , z ] \in \mathbb { R } ^ { d _ { \gamma } + d _ { z } } .\tag{2}
$$

This vector is fed into a classical embedding network Γ composed of fully connected layers and ReLU activations. The network outputs $3 N _ { f }$ values, where $N _ { f }$ is the number of feature qubits. A hyperbolic tangent nonlinearity and an angle scaling factor α are then applied to obtain the base rotation angles:

$$
a ( c , z ) = \alpha \operatorname { t a n h } ( \Gamma ( u ( c , z ) ) ) \in \mathbb { R } ^ { 3 N _ { f } } .\tag{3}
$$

The vector is reshaped as

$$
A ( c , z ) = { \mathrm { R e s h a p e } } ( a ( c , z ) ) \in \mathbb { R } ^ { N _ { f } \times 3 } .\tag{4}
$$

For the i-th feature qubit, the three base angles are

$$
\begin{array} { r } { A _ { i } ( c , z ) = ( a _ { i , x } ( c , z ) , a _ { i , y } ( c , z ) , a _ { i , z } ( c , z ) ) , } \\ { i = 1 , \ldots , N _ { f } . \qquad } \end{array}\tag{5}
$$

which are used for $R _ { X } , R _ { Y }$ , and $R _ { Z }$ input rotations each.

## Quantum Implicit Generator Circuit

The quantum generator consists of $N _ { q } = N _ { f } + 1$ qubits. The first qubit $q _ { 0 }$ is designated as the color qubit for pixel readout, while the remaining qubits $q _ { 1 } , \ldots , q _ { N _ { t } }$ are feature qubits that encode coordinate- and latent-dependent implicit features. The circuit is initialized from the all-zero state.

Color-Qubit Brightness Initialization Initially, a trainable $R _ { Y }$ rotation is applied to the color qubit:

$$
U _ { \mathrm { b i a s } } = R _ { Y } ^ { ( q _ { 0 } ) } ( \beta ) ,\tag{6}
$$

where $\beta$ is a trainable brightness-bias parameter. To give the model a reasonable initial pixel-brightness distribution, β is initialized according to a preset mean pixel value $\mu _ { 0 } \in [ 0 , 1 ] :$

$$
\beta _ { 0 } = \operatorname { a r c c o s } ( 1 - 2 \mu _ { 0 } ) .\tag{7}
$$

Ignoring subsequent feature-to-color writing operations, the Pauli-Z expectation of the color qubit is $\langle Z _ { 0 } \rangle \stackrel { . } { = } \cos \beta _ { 0 }$ , and the corresponding initial pixel value is

$$
\frac { 1 - \langle Z _ { 0 } \rangle } { 2 } = \frac { 1 - \cos \beta _ { 0 } } { 2 } = \mu _ { 0 } .\tag{8}
$$

This initialization explicitly controls the initial average brightness of the generator and improves early-stage training stability.

Scaled Data Re-uploading Layer The quantum circuit comprises L data re-uploading layers. Rather than injecting the same base angles $A ( c , z )$ identically at every layer, CoQui equips each layer l with trainable scale and bias parameters: $S _ { l } ^ { ' } \in \mathbb { R } ^ { N _ { f } \times 3 } , \dot { B } _ { l } \in \mathbb { R } ^ { N _ { f } \times 3 }$ . The angles injected at layer l are obtained through the element-wise afine transformation

$$
\Theta _ { l } ( c , z ) = S _ { l } \odot A ( c , z ) + B _ { l } ,\tag{9}
$$

where $\odot$ denotes element-wise multiplication. Specifically, for feature qubit $q _ { i }$ and rotation axis $r \in \{ x , y , z \}$ },

$$
\theta _ { i , r } ^ { l } ( c , z ) = s _ { i , r } ^ { l } a _ { i , r } ( c , z ) + b _ { i , r } ^ { l } .\tag{10}
$$

Here, $s _ { i , r } ^ { l }$ controls the strength of the coordinate–latent signal, whereas $b _ { i . } ^ { l }$ provides an input-independent angular ofset. The three angles associated with $q _ { i }$ are grouped as

$$
\Theta _ { i , l } ( c , z ) = \big ( \theta _ { i , 1 } ^ { l } ( c , z ) , \theta _ { i , 2 } ^ { l } ( c , z ) , \theta _ { i , 3 } ^ { l } ( c , z ) \big ) .\tag{11}
$$

This encoding acts on all feature qubits:

$$
U _ { \mathrm { e n c } } ^ { l } = \prod _ { i = 1 } ^ { N _ { f } } { R _ { Z } ^ { ( q _ { i } ) } \big ( \theta _ { i , z } ^ { l } ( c , z ) \big ) R _ { Y } ^ { ( q _ { i } ) } \big ( \theta _ { i , y } ^ { l } ( c , z ) \big ) }\tag{12}
$$

This layer-dependent reparametrization lets each layer emphasize, attenuate, or shift components of the shared coordinate–latent representation, enhancing the PQC’s expressive capacity for implicit function modeling.

Feature-Qubit Local Rotations After input encoding, each feature qubit receives trainable local rotations with parameters $W _ { l } \doteq \mathbb { R } ^ { N _ { f } \times 3 }$ where $\boldsymbol { W _ { i } ^ { l } } = ( w _ { i , 1 } ^ { l } , w _ { i , 2 } ^ { l } , w _ { i , 3 } ^ { l } )$ for the i-th feature qubit. The local rotation unit is

$$
U _ { \mathrm { l o c } } ^ { l } = \prod _ { i = 1 } ^ { N _ { f } } R _ { Y } ^ { ( q _ { i } ) } ( w _ { i , 3 } ^ { l } ) R _ { Z } ^ { ( q _ { i } ) } ( w _ { i , 2 } ^ { l } ) R _ { Y } ^ { ( q _ { i } ) } ( w _ { i , 1 } ^ { l } ) .\tag{13}
$$

This component provides input-independent trainable quantum transformations after coordinate-latent encoding.

Feature-Feature Entanglement To model correlations among feature qubits, CoQui introduces a ring CNOT entanglement pattern:

$$
U _ { \mathrm { e n t } } ^ { l } = \prod _ { i = 1 } ^ { N _ { f } } \mathrm { C N O T } ( q _ { i } , q _ { i + 1 } ) , \qquad q _ { N _ { f } + 1 } \equiv q _ { 1 } .\tag{14}
$$

Feature-to-Color Writing The core circuit design writes information from feature qubits to the color qubit through controlled-R<sub>Y</sub> gates. For layer l and feature qubit $q _ { i }$ , let $\eta _ { i } ^ { l } \in \mathbb { R }$ denote the trainable color-writing parameter. The feature-to-color operation is $\mathrm { C R Y } _ { q _ { i } \to q _ { 0 } } ( \eta _ { i } ^ { l } )$ , where $q _ { i }$ is the control qubit and $q _ { 0 }$ is the target qubit. The complete writing module at layer l is

$$
U _ { \mathrm { w r i t e } } ^ { l } = \prod _ { i = 1 } ^ { N _ { f } } \mathrm { C R Y } _ { q _ { i } \to q _ { 0 } } ( \eta _ { i } ^ { l } ) .\tag{15}
$$

This module makes the color-qubit state depend explicitly on feature qubits that encode coordinate and latent information. Thus, feature qubits form implicit spatial features, while the color qubit acts as the measurable pixel-output channel.

Color Residual Update After each feature-to-color writing module, CoQui applies a trainable residual update on the color qubit:

$$
U _ { \mathrm { r e s } } ^ { l } = R _ { Y } ^ { ( q _ { 0 } ) } ( r _ { l , 3 } ) R _ { Z } ^ { ( q _ { 0 } ) } ( r _ { l , 2 } ) R _ { Y } ^ { ( q _ { 0 } ) } ( r _ { l , 1 } ) ,\tag{16}
$$

where $r _ { l } = ( r _ { l , 1 } , r _ { l , 2 } , r _ { l , 3 } )$ denotes the color residual parameters at layer l. This module provides additional local degrees of freedom for the readout qubit, allowing pixel intensities to be accumulated, adjusted, and refined across multiple reuploading layers.

Combining the above modules, the l-th quantum transformation is

$$
\begin{array} { r } { U _ { l } ( c , z ) = U _ { \mathrm { r e s } } ^ { l } U _ { \mathrm { w r i t e } } ^ { l } U _ { \mathrm { e n t } } ^ { l } U _ { \mathrm { l o c } } ^ { l } U _ { \mathrm { e n c } } ^ { l } ( c , z ) . } \end{array}\tag{17}
$$

For coordinate $c = ( x , y )$ and latent variable $z ,$ the complete PQC outputs

$$
\vert \psi _ { \Theta } ( c , z ) \rangle = U _ { L } ( c , z ) \cdot \cdot \cdot U _ { 2 } ( c , z ) U _ { 1 } ( c , z ) U _ { \mathrm { b i a s } } \vert \psi _ { 0 } \rangle .\tag{18}
$$

The generator parameters are $\Phi = \{ s ^ { l } , b ^ { l } , W ^ { l } , \eta ^ { l } , r ^ { l } \} _ { l = 1 } ^ { L }$

## Quantum Measurement and Pixel Readout

After L layers of quantum evolution, CoQui measures only the Pauli-Z expectation of the color qubit:

$$
\begin{array} { r } { m _ { \Phi } ( c , z ) = \langle Z _ { 0 } \rangle = \langle \psi _ { \Phi } ( c , z ) | Z _ { 0 } | \psi _ { \Phi } ( c , z ) \rangle . } \end{array}\tag{19}
$$

The expectation is then mapped linearly to a normalized grayscale pixel intensity:

$$
G _ { \Phi } ( c , z ) = \frac { 1 - m _ { \Phi } ( c , z ) } { 2 } .\tag{20}
$$

Since $m _ { \Phi } ( c , z ) \in [ - 1 , 1 ]$ , the output satisfies $G _ { \Phi } ( c , z ) \in$ [0, 1]. This design uses a physical observable of the color qubit directly as the pixel output, making the quantum circuit itself responsible for pixel-intensity modeling.

Given a latent variable $z , \mathrm { ~ a ~ }$ full image is generated by querying all coordinates in $\mathcal { C } _ { H , W }$

$$
\hat { X } _ { j , i } = G _ { \Phi } \left( \frac { i } { W } , \frac { j } { H } , z \right) , 0 \leq i < W , 0 \leq j < H .\tag{21}
$$

yielding

$$
\hat { X } = G _ { \Phi } ( z ) \in [ 0 , 1 ] ^ { H \times W \times 1 } .\tag{22}
$$

## Training Objective

Following PQWGAN (Tsang et al. 2023), CoQui is trained using the Wasserstein GAN objective with gradient penalty. The critic distinguishes real images from generated ones, while the generator learns to produce realistic images. Further implementation details of the classical embedding network and the critic are provided in Appendix 1.

## Experiments

## Experimental Setup

We evaluate CoQui on MNIST and Fashion-MNIST at the standard 28 × 28 resolution. The training dataset consists of 1000 real images in each experiment. The model is trained with Adam for 1000 epochs with a batch size of 5. Following the setting of Jäger et al., the learning rates of the generator and discriminator are set to $1 \times \bar { 1 } 0 ^ { - 3 }$ and $1 \times \mathrm { { 1 0 ^ { - 4 } } }$ respectively. The quantum generator used in the experiments contains one color qubit and $N _ { f } = 4$ feature qubits, and the circuit depth is set to $r = 2 0$ re-uploading layers. The hyperparameters of the other classical components are reported in Appendix 1. All experiments are performed in simulation on a machine with an AMD Ryzen 9 9950X3D 16-Core Processor at 4.30 GHz and 64 GB RAM. All models are implemented in JAX.

Baselines. We compare CoQui with two representative quantum image generation methods, PQWGAN and the endto-end Quantum Wasserstein QGAN proposed by Jäger et al., both of which adopt the amplitude-mapping paradigm. Experiments are conducted under two evaluation settings. We first compare the methods on the standard multi-class benchmark following the protocol of Jäger et al. We then perform per-class experiments, where each method is trained independently on each class using the same training protocol and data budget. This complementary evaluation assesses both the ability to model a heterogeneous image distribution and the generation quality under matched training conditions. In addition, we implement a classical counterpart of CoQui to isolate the contribution of the proposed quantum generator. All baseline methods are reimplemented by us and optimized for eficient classical simulation. The code will be made publicly available upon acceptance.

Evaluation Metric. We evaluate generated samples using four complementary metrics: FID, classifier-based diversity metrics, feature-space precision and recall, and auxiliary low-level visual statistics. FID measures the overall discrepancy between real and generated distributions (Jäger, Kiwit, and Riofrío 2026). To evaluate class diversity, a pretrained MNIST/Fashion-MNIST classifier is used to obtain the predicted class distribution $p _ { g } ( y )$ , following the classifier-based evaluation idea of Inception Score and Mode Score (Salimans et al. 2016; Che et al. 2016). We report Entropy as $\begin{array} { r } { H _ { \mathrm { n o r m } } ( p _ { g } ) = - \frac { 1 } { \log C } \sum _ { c = 1 } ^ { C } p _ { g } ( c ) \log p _ { g } ( c ) } \end{array}$ , where $C = 1 0$ , and JSD between $p _ { g } ( y )$ and the uniform distribution $u ( y )$ , defined as $\mathrm { J S D } ( p _ { g } \| u ) = \textstyle { \frac { 1 } { 2 } } \mathrm { K L } ( p _ { g } \| m ) + \frac { 1 } { 2 } \mathrm { K L } ( u \| m )$ with $m \ = \ { \textstyle { \frac { 1 } { 2 } } } ( p _ { g } + u )$ (Lin 1991). We further use P@5

![](images/6878251c4e47f4b5c4e8697a84f37a86c427f4c239909059651fcab5a4fcce56.jpg)  
Figure 2: Qualitative comparison on MNIST and Fashion-MNIST. Top: samples from each class for each method. Bottom: represen tative grids over ten classes, with FID scores reported above each grid.

and R@5 to measure sample fidelity and distribution coverage in the Inception feature space, following the improved precision and recall metric for generative models (Kynkäänniemi et al. 2019). Finally, Brightness and Sobel are defined as $D _ { \mathrm { b r i g h t } } = | \mathrm { m e a n } ( \dot { x } _ { g } ) - \mathrm { m e a n } ( x _ { r } ) |$ and $D _ { \mathrm { s o b e l } } = | \operatorname { m e a n } ( \operatorname { S o b e l } ( x _ { g } ) ) - \operatorname { m e a n } ( \operatorname { \tilde { S } o b e l } ( x _ { r } ) )$ |, respectively, where the Sobel operator is used to estimate image gradient magnitude (Sobel and Feldman 1968). Lower values are better for FID, JSD, Brightness, and Sobel, whereas higher values are better for Entropy, P@5, and R@5.
<table><tr><td>Dataset</td><td>Method</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td></td><td>7</td><td>8</td></tr><tr><td rowspan="5">MNIST</td><td>PQWGAN</td><td></td><td>210 137</td><td>223</td><td>203</td><td></td><td>191 187 179</td><td></td><td></td><td>179</td><td>195 199</td></tr><tr><td>Wasserstein QGAN 214 122</td><td></td><td></td><td>2203</td><td></td><td></td><td>174 189175184 172 193 197</td><td></td><td></td><td></td><td></td></tr><tr><td>Classical INR-GAN</td><td>20</td><td>16</td><td>22</td><td>14</td><td>21</td><td>19</td><td>16</td><td>20</td><td>19</td><td>19</td></tr><tr><td>CoQui</td><td>17</td><td>17</td><td>26</td><td>17</td><td>19</td><td>21</td><td>17</td><td>25</td><td>19</td><td>16</td></tr><tr><td>PQWGAN</td><td>241</td><td>97</td><td></td><td>204 175 213</td><td></td><td></td><td>3 210192 219 275 230</td><td></td><td></td><td></td></tr><tr><td></td><td>Fashion- Wasserstein QGAN 167 112 164 183 162 169</td><td></td><td></td><td></td><td></td><td></td><td></td><td>9165</td><td></td><td></td><td>5246 184 176</td></tr><tr><td>MNIST</td><td>Classical</td><td>53</td><td>25</td><td>54</td><td>44</td><td>。 56</td><td>58</td><td>61</td><td>43</td><td>47</td><td>39</td></tr><tr><td></td><td>INR-GAN CoQui</td><td>47</td><td>28</td><td>45</td><td>40</td><td>41</td><td>62</td><td>52</td><td>43</td><td>55</td><td>38</td></tr></table>

Table 1: Comparison on MNIST and Fashion-MNIST. Best results are in bold.

## Main Results

Figure 2 shows representative generated samples on classwise subsets and on the full dataset containing ten classes.

<table><tr><td>Method</td><td>Qubits</td><td>Number of Quantum Circuits</td></tr><tr><td>PQWGAN</td><td>192</td><td>32</td></tr><tr><td>Wasserstein QGAN</td><td>11</td><td>1</td></tr><tr><td>CoQui</td><td>5</td><td>1</td></tr></table>

Table 2: Comparison of quantum resource requirements across different QGAN models.

The FID scores for the full-dataset setting are directly annotated in the figure, while the class-wise FID scores are reported in Table 1. On the class-wise subsets, CoQui generates samples with clearly recognizable category structure and good visual quality. This is especially evident on the more complex Fashion-MNIST dataset, where CoQui still produces visually meaningful samples. In contrast, the two amplitude-mapping based QGAN models generate noticeably blurrier samples. This is mainly due to the pixel coupling introduced by amplitude encoding: pixels share normalized amplitudes and are therefore dificult to adjust independently, which limits the representation of fine-grained features. Compared with the Classical INR-GAN of a similar parameter scale, CoQui achieves the best or tied-best FID scores on four MNIST classes and seven Fashion-MNIST classes. This indicates that CoQui retains strong generative capability under a low quantum-resource budget, highlighting the potential of quantum generative models for image generation. The full-dataset setting with ten classes is more challenging than class-wise generation because it involves a more complex data distribution; accordingly, all methods show degraded performance. Nevertheless, PQWGAN gives the weakest visual results, followed by Wasserstein QGAN trained under the same setting, while Wasserstein QGAN trained with the original 6K-sample setting shows some improvement. During our reproduction of Wasserstein QGAN, we observed that the original work employs a post-selection strategy for visualization: for each noise pattern, 500 candidate samples are generated, and the sample with the smallest Euclidean distance to their mean is selected for display. We do not apply this selection procedure and instead directly visualize randomly generated samples. Compared with Classical INR-GAN, CoQui achieves comparable overall visual quality and shows advantages in local detail modeling.

In addition, as shown in Table 2, CoQui uses only five qubits and one quantum circuit, giving it the lowest quantumresource cost among all compared methods. The qubit requirement of amplitude-mapping based methods typically grows with image resolution, whereas the qubit count of Co-Qui is not directly tied to image resolution. CoQui therefore provides better resource eficiency and scalability. The training convergence behavior is analyzed in Appendix 2.

## Ablation Study 1: circuit architecture ablation

We further conduct ablation studies to examine the contribution of the proposed circuit architecture and the color-qubit design. Figure 3 provides representative MNIST samples for these variants, while Tables 3 and 4 report the corresponding quantitative results.

Circuit design ablation Table 3 compares CoQui using the proposed feature-to-color circuit with a hardware-eficient ansatz. The proposed circuit improves most metrics, reducing FID from 41.41 to 40.15 and JSD from 0.00858 to 0.00411, while increasing entropy from 0.9856 to 0.9930 and P@5 from 0.489 to 0.505. It also yields lower brightness and Sobel discrepancies, indicating better low-level image statistics and sharper local structures. The hardware-eficient circuit gives a slightly higher R@5, but its weaker FID, JSD, and edge statistics suggest poorer overall distributional alignment. These results show that the structured feature-to-color modulation is important for stable and faithful image generation.

<table><tr><td>Metric</td><td>CoQui with hardware-efficient circuit</td><td>CoQui with our proposed circuit</td></tr><tr><td>FID↓</td><td> $4 1 . 4 1 \pm 6 . 1 0$ </td><td> ${ \bf 4 0 . 1 5 \pm 3 . 4 5 }$ </td></tr><tr><td>JSD↓</td><td> $0 . 0 0 8 5 8 \pm 0 . 0 0 3 3 3$ </td><td> $\mathbf { 0 . 0 0 4 1 1 \pm 0 . 0 0 1 2 8 }$ </td></tr><tr><td>Entropy ↑</td><td> $0 . 9 8 5 6 \pm 0 . 0 0 5 5$ </td><td> $\mathbf { 0 . 9 9 3 0 \pm 0 . 0 0 2 1 }$ </td></tr><tr><td>P@5↑</td><td> $0 . 4 8 9 \pm 0 . 0 8 1$ </td><td> $\mathbf { 0 . 5 0 5 \pm 0 . 0 6 1 }$ </td></tr><tr><td>R@5↑</td><td> $\mathbf { 0 . 4 8 5 \pm 0 . 2 5 0 }$ </td><td> $0 . 4 6 7 \pm 0 . 1 0 8$ </td></tr><tr><td>Brightness ↓</td><td> $0 . 0 0 6 5 \pm 0 . 0 0 4 6$ </td><td> $\mathbf { 0 . 0 0 5 8 \pm 0 . 0 0 5 1 }$ </td></tr><tr><td>Sobel ↓</td><td> $0 . 0 2 0 0 \pm 0 . 0 0 0 0$ </td><td> $\mathbf { 0 . 0 0 7 3 \pm 0 . 0 0 3 3 }$ </td></tr></table>

Table 3: Quantitative comparison between CoQui with the hardware-eficient circuit and CoQui with our proposed circuit. Best results are shown in bold.

Core-component ablation Figure 3(b) and Table 4 evaluate the main color-qubit components. Full CoQui achieves the best JSD, entropy, brightness discrepancy, and Sobel discrepancy, showing the most balanced match to the real distribution. Removing the color-qubit residual or the colorbias initialization can slightly improve FID, but both variants worsen low-level statistics, especially edge consistency. For example, removing the residual increases the Sobel discrepancy from 0.0073 to 0.0370. Removing the scaled data re-uploading causes the largest degradation, increasing FID from 40.15 to 56.54, which indicates that layer-wise input scaling is important for efectively modulating coordinateand noise-dependent quantum features. Overall, these ablations suggest that the complete color-qubit design is not optimized for a single metric only, but provides the best trade-of across distributional fidelity, sample diversity, brightness, and local structure.

![](images/b49b12913a27a2e05d472acac384b7ffad417e8a4f3e4221c25c1fee654f27b3.jpg)

CoQui with hardware-eficient ansatz

0/2 3 6 5 6 789   
036678

CoQui with our proposed circuit (a) Circuit ablation

![](images/ccbfdeebae616bb772acc234538f8b332dad2eb63f406f90071653d1b8a0485f.jpg)

Full CoQui

![](images/a8ab6cc8cd42bd63b7229d27a62f9a3d918fbb16e532add3201209c07fcc4f10.jpg)

w/o color-qubit residual

![](images/965d5523937daaad8f97b5138e238b60f11968cccc3cfd58e8e1c900321d24f0.jpg)

w/o color-qubit bias initialization w/o scaled data re-uploading (b) Core-component ablation

Figure 3: Representative MNIST samples for (a) CoQui and its hardware-eficient variant, and (b) the CoQui baseline and variants without the color residual, color-bias initialization, or scaled data re-uploading.
<table><tr><td>Metric</td><td>Full CoQui</td><td>CoQui w/o color-qubit residual</td><td>CoQui w/o color-bias initialization</td><td>CoQui w/o scaled data re-uploading</td></tr><tr><td rowspan="2">FID↓</td><td>40.15</td><td>38.82</td><td>36.81</td><td>56.54</td></tr><tr><td>±3.45</td><td>±1.52</td><td>±3.30</td><td>±3.63</td></tr><tr><td rowspan="2">JSD↓</td><td>0.00411</td><td>0.00493</td><td>0.00560</td><td>0.00439</td></tr><tr><td>±0.00128</td><td>±0.00502</td><td>±0.00079</td><td>±0.00063</td></tr><tr><td rowspan="2">Entropy ↑</td><td>0.9930</td><td>0.9915</td><td>0.9903</td><td>0.9924</td></tr><tr><td>±0.0021</td><td>±0.0087</td><td>±0.0015</td><td>±0.0009</td></tr><tr><td rowspan="2">Brightness ↓</td><td>0.0058</td><td>0.0133</td><td>0.0084</td><td>0.0091</td></tr><tr><td>±0.0051</td><td>±0.0015</td><td>±0.0045</td><td>±0.0033</td></tr><tr><td rowspan="2">Sobel ↓</td><td>0.0073</td><td>0.0370</td><td>0.0142</td><td>0.0182</td></tr><tr><td>±0.0033</td><td>±0.0200</td><td>±0.0171</td><td>±0.0036</td></tr></table>

Table 4: Quantitative core-component ablation results for CoQui. Best results are shown in bold.

## Ablation Study 2: Model Capacity Analysis

Figure 4 and Table 5 evaluate the efect of model capacity from two aspects: the number of feature qubits and the circuit depth. Since the computational cost of quantum circuit simulation grows rapidly with both the number of qubits and the circuit depth, we first identify a suitable number of feature qubits under a fixed circuit depth, and then compare diferent circuit depths based on this setting.

<table><tr><td>Ablation Study</td><td>Setting</td><td>FID ↓</td><td>Class JSD ↓</td><td>Class Entropy ↑</td></tr><tr><td rowspan="5">Feature qubits</td><td> $N _ { f } = 2$ </td><td>39.53</td><td>0.0045</td><td>0.9923</td></tr><tr><td> $\dot { N _ { f } } = 3$ </td><td>41.56</td><td>0.0157</td><td>0.9739</td></tr><tr><td> $N _ { f } = 4$ </td><td>42.59</td><td>0.0032</td><td>0.9944</td></tr><tr><td> $N _ { f } = 5$ </td><td>37.88</td><td>0.0202</td><td>0.9677</td></tr><tr><td> $N _ { f } = 6$ </td><td>44.65</td><td>0.0062</td><td>0.9895</td></tr><tr><td rowspan="4">Circuit depth</td><td>10 layers</td><td>39.07</td><td>0.0032</td><td>0.9945</td></tr><tr><td>20 layers</td><td>42.59</td><td>0.0032</td><td>0.9944</td></tr><tr><td>30 layers</td><td>37.90</td><td>0.0079</td><td>0.9863</td></tr><tr><td>40 layers</td><td>38.73</td><td>0.0082</td><td>0.9853</td></tr></table>

Table 5: Quantitative capacity comparison on MNIST. Top: varying feature qubits at $r = 2 0 ;$ bottom: varying circuit depth with $\dot { N _ { f } } \dot { = }$ 4. Best values are bold.

Efect of Model Capacity As shown in Figure 4(a), increasing $N _ { f }$ from 2 to 4 clearly improves digit shape and edge quality. When $N _ { f } ~ = ~ 5 .$ , the visual quality becomes close to that of $N _ { f } = \dot { 4 }$ while $N _ { f } = 6$ shows slight degradation, e.g., parts of the strokes of $\mathrm { d i g i t } \ ^ { \mathrm { 6 6 } }$ in the first row become sticky and connected, and the shape of digit “9” is also slightly distorted. Quantitatively, the fixed circuit depth of r = 20, $N _ { f } = 5$ achieves the lowest FID of 37.88, but N<sub>f</sub> = 4 gives the best Class JSD and Class Entropy (Table 5), so we adopt $N _ { f } = 4$ for the subsequent depth experiments.

Figure 4(b) shows 10- and 20-layer circuits perform comparably, while 30-40 layers show mild degradation in both visual quality and class JSD/Entropy. This suggests shallow-tomoderate depth already sufices, with deeper circuits adding optimization dificulty rather than expressivity.

14SS78 N<sub>f</sub> = 2   
C)73436フ89   
812B578 N<sub>f</sub> = 3   
0/2356789   
0123656789 N<sub>f</sub> = 4   
0123456739   
0123456789 N<sub>f</sub> = 5   
012345G789   
0123456789 N<sub>f</sub> = 6 (a) Feature qubits ablation   
0123456789   
01℃3436788 10 layers   
0/73456789   
0123656789 20 layers   
0133456729   
02350789 30 layers   
0279   
0/23956789 40 layers (b) Circuit depth ablation

Figure 4: Representative MNIST samples for two capacity studies. (a) Diferent numbers of feature qubits at a fixed circuit depth of r = 20. (b) Diferent circuit depths using four feature qubits.

## Conclusion and Discussion

In this paper, we propose CoQui, an end-to-end GAN framework for image generation based on quantum implicit neural representations, reformulating quantum image generation as a coordinate-conditioned implicit function learning problem. This decouples image resolution from the number of address qubits and alleviates the inter-pixel coupling induced by conventional amplitude-to-pixel mapping. We further design a quantum generator with structural inductive biases tailored to this formulation, enhancing model expressiveness. Simulation results show that CoQui achieves better visual quality and quantitative performance than amplitude-mapping-based QGAN baselines while using fewer qubits, and achieves competitive generative performance relative to a classical baseline of comparable parameter scale. The current evaluation is limited to classical simulation on standard-resolution grayscale benchmarks, without accounting for realistic hardware noise, finite-shot efects, or device connectivity. Future work will extend CoQui to higher-resolution and color image generation while improving scalability, hardware compatibility, and noise robustness toward deployment on real quantum devices.

## References

Arjovsky, M.; Chintala, S.; and Bottou, L. 2017. Wasserstein Generative Adversarial Networks. In Precup, D.; and Teh, Y. W., eds., Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings ofMachine Learning Research, 214–223. PMLR.

Cao, J.; Wang, Q.; Xian, Y.; Li, Y.; Ni, B.; Pi, Z.; Zhang, K.; Zhang, Y.; Timofte, R.; and Van Gool, L. 2023. CiaoSR: Continuous Implicit Attention-in-Attention Network for Arbitrary-Scale Image Super-Resolution. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 1796–1807.

Chang, S. Y.; Thanasilp, S.; Saux, B. L.; Vallecorsa, S.; and Grossi, M. 2024. Latent style-based quantum GAN for highquality image generation. arXiv preprint arXiv:2406.02668.

Che, T.; Li, Y.; Jacob, A. P.; Bengio, Y.; and Li, W. 2016. Mode Regularized Generative Adversarial Networks. arXiv preprint arXiv:1612.02136.

Chen, Y.; Liu, S.; and Wang, X. 2021. Learning continuous image representation with local implicit image function. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 8628–8638.

Chu, C.; Skipper, G.; Swany, M.; and Chen, F. 2023. IQ-GAN: Robust quantum generative adversarial network for image synthesis on NISQ devices. In ICASSP 2023-2023 IEEE international conference on acoustics, speech and signal processing (ICASSP), 1–5. IEEE.

Dallaire-Demers, P.-L.; and Killoran, N. 2018. Quantum generative adversarial networks. Phys. Rev. A, 98: 012324.

Eren, S. M. 2026. Implementation of Quantum Implicit Neural Representation in Deterministic and Probabilistic Autoencoders for Image Reconstruction/Generation Tasks. arXiv preprint arXiv:2603.06755.

Fujihashi, T.; and Koike-Akino, T. 2026. Quantum Implicit Neural Compression. In Ali, S.; Chicano, F.; and Moraglio, A., eds., Quantum Computing andArtificial Intelligence, 60– 69. Cham: Springer Nature Switzerland. ISBN 978-3-032- 15931-1.

Gonzalez, R. C. 2009. Digital image processing. Pearson education india.

Goodfellow, I.; Pouget-Abadie, J.; Mirza, M.; Xu, B.; Warde-Farley, D.; Ozair, S.; Courville, A.; and Bengio, Y. 2020. Generative adversarial networks. Communications of the ACM, 63(11): 139–144.

Gulrajani, I.; Ahmed, F.; Arjovsky, M.; Dumoulin, V.; and Courville, A. 2017. Improved training of Wasserstein GANs. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, 5769–5779. Red Hook, NY, USA: Curran Associates Inc. ISBN 9781510860964.

Huang, H.-L.; Du, Y.; Gong, M.; Zhao, Y.; Wu, Y.; Wang, C.; Li, S.; Liang, F.; Lin, J.; Xu, Y.; et al. 2021. Experimental quantum generative adversarial networks for image generation. Physical Review Applied, 16(2): 024051.

Jin, H.; Singh, G.; and Merz Jr, K. M. 2025. QFGN: A Quantum Approach to High-Fidelity Implicit Neural Representations. arXiv preprint arXiv:2504.19053.

Jäger, J.; Kiwit, F. J.; and Riofrío, C. A. 2026. Scaling quantum machine learning without tricks: full-resolution and diverse image generation. Quantum Science and Technology, 11(3): 035042.

Kynkäänniemi, T.; Karras, T.; Laine, S.; Lehtinen, J.; and Aila, T. 2019. Improved Precision and Recall Metric for Assessing Generative Models. In Advances in Neural Information Processing Systems, volume 32.

Lin, J. 1991. Divergence Measures Based on the Shannon Entropy. IEEE Transactions on Information Theory, 37(1): 145–151.

Lloyd, S.; and Weedbrook, C. 2018. Quantum generative adversarial learning. Physical review letters, 121(4): 040502.

Mescheder, L.; Oechsle, M.; Niemeyer, M.; Nowozin, S.; and Geiger, A. 2019. Occupancy Networks: Learning 3D Reconstruction in Function Space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Müller, T.; Evans, A.; Schied, C.; and Keller, A. 2022. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4): 1– 15.

Park, J. J.; Florence, P.; Straub, J.; Newcombe, R.; and Lovegrove, S. 2019. DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 165–174.

Pérez-Salinas, A.; Cervera-Lierta, A.; Gil-Fuster, E.; and Latorre, J. I. 2020. Data re-uploading for a universal quantum classifier. Quantum, 4: 226.

Salimans, T.; Goodfellow, I.; Zaremba, W.; Cheung, V.; Radford, A.; and Chen, X. 2016. Improved Techniques for Training GANs. In Advances in Neural Information Processing Systems, volume 29.

Schuld, M.; Sweke, R.; and Meyer, J. J. 2021. Efect of data encoding on the expressive power of variational quantum-machine-learning models. Physical Review A, 103(3): 032430.

Silver, D.; Ranjan, A.; Patel, T.; Gandhi, H.; Cutler, W.; and Tiwari, D. 2023. MosaiQ: Quantum Generative Adversarial Networks for Image Generation on NISQ Computers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 7007–7016.

Sitzmann, V.; Martel, J.; Bergman, A.; Lindell, D.; and Wetzstein, G. 2020. Implicit neural representations with periodic activation functions. Advances in neural information processing systems, 33: 7462–7473.

Sobel, I.; and Feldman, G. 1968. An Isotropic 3x3 Image Gradient Operator. Presentation at Stanford Artificial Intelligence Project. Reprinted in 2014.

Su, K.; Chen, M.; and Shlizerman, E. 2022. Inras: Implicit neural representation for audio scenes. Advances in Neural Information Processing Systems, 35: 8144–8158.

Thomas, A. M.; Youel, H.; and Jose, S. T. 2025. VAE-QWGAN: addressing mode collapse in quantum GANs via autoencoding priors. Quantum Machine Intelligence, 7(2): 91.

Tsang, S. L.; West, M. T.; Erfani, S. M.; and Usman, M. 2023. Hybrid quantum–classical generative adversarial network for high-resolution image generation. IEEE Transactions on Quantum Engineering, 4: 1–19.

Vieloszynski, A.; Cherkaoui, S.; Ahmad, O.; Laprade, J.- F.; Nahman-Lévesque, O.; Aaraba, A.; and Wang, S. 2024. Latentqgan: A hybrid qgan with classical convolutional autoencoder. In 2024 IEEE 10th World Forum on Internet of Things (WF-IoT), 1–7. IEEE.

Wang, S.; Theobalt, C.; and Golyanik, V. 2026. Quantum visual fields with neural amplitude encoding. Advances in Neural Information Processing Systems, 38: 158535–158559.

Yang, X.; Zhou, R.; Jia, S.; Koh, D. E.; Goh, S. T.; Li, Y.; Chen, H.; and Xiong, F. 2026a. End-to-End QGAN-Based Image Synthesis via Neural Noise Encoding and Intensity Calibration. arXiv preprint arXiv:2603.18554.

Yang, X.; Zhou, R.; Jia, S.; Li, Y.; Yan, J.; Long, Z.; Guo, W.; Xiong, F.; and Xu, W. 2026b. iHQGAN: A lightweight invertible hybrid quantum-classical generative adversarial networks for unsupervised image-to-image translation. Expert Systems with Applications, 296: 128865.

Zhang, J.; Che, X.; Fan, Y.; Peng, S.; Chen, G.; Ma, Q.; and Hu, J. 2025. Denoising difusion models with optimized quantum implicit neural networks for image generation. Future Generation Computer Systems, 173: 107875.

Zhao, J.; Qiao, W.; Zhang, P.; and Gao, H. 2024. Quantum implicit neural representations. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org.

Zoufal, C.; Lucchi, A.; and Woerner, S. 2019. Quantum generative adversarial networks for learning and loading random distributions. npj Quantum Information, 5(1): 103.

## Appendix 1 Classical components

The proposed framework contains two classical components: a classical embedding network in the generator and a convolutional critic for adversarial training. The embedding network transforms each spatial coordinate and latent variable into the rotation parameters of the quantum circuit, thereby conditioning the quantum generator on both pixel location and stochastic variation. The critic follows the WGAN-GP formulation and maps an input image to a scalar score through a sequence of strided convolutional layers.

The classical component of the generator is a classical embedding network. Each pixel coordinate is encoded using a sinusoidal positional embedding with six frequency bands, producing a 26-dimensional coordinate feature. This feature is concatenated with a 10-dimensional latent vector and passed through a three-layer MLP with hidden width 256 and ReLU activations. The MLP outputs 12 rotation angles, corresponding to three rotation angles for each of the four feature qubits. The raw angles are squashed by a tanh nonlinearity and scaled by π before being passed to the quantum circuit.

The discriminator is a classical WGAN-GP critic composed of three stride-2 convolutional layers with 3 × 3 kernels and channel widths (32, 64, 128), each followed by a LeakyReLU activation with negative slope 0.2. The resulting feature map is flattened and mapped to a scalar critic score by a fully connected layer. All convolutional and dense kernels are initialized with Glorot uniform initialization, and biases are initialized to zero. The detailed architectures and initialization settings of the two components are summarized in Table 1.

<table><tr><td>Module</td><td>Component</td><td>Configuration</td></tr><tr><td rowspan="9">Generator embedding</td><td>Coordinate input Positional encoding</td><td>Two-dimensional pixel coordinate (x, y) Sinusoidal encoding with six frequency bands</td></tr><tr><td></td><td></td></tr><tr><td>Coordinate feature dimension</td><td>26</td></tr><tr><td>Latent input dimension</td><td>10 Three layers, hidden width 256</td></tr><tr><td>MLP architecture</td></tr><tr><td>Activation</td></tr><tr><td></td></tr><tr><td>ReLU 12 rotation angles for four feature qubits</td></tr><tr><td>Output transformation π tanh(·)</td></tr><tr><td rowspan="5">Critic</td><td>Architecture Kernel size</td><td>Three stride-2 convolutional layers 3 × 3</td></tr><tr><td>Channel widths</td><td>32, 64, and 128</td></tr><tr><td></td><td></td></tr><tr><td>Activation</td><td>LeakyReLU with negative slope 0.2</td></tr><tr><td>Output layer</td><td>Flatten followed by a scalar-valued linear layer</td></tr><tr><td rowspan="2">Initialization</td><td>Weights</td><td>Glorot uniform initialization</td></tr><tr><td>Biases</td><td>Zero initialization</td></tr></table>

Table 1: Architectural details of the classical components used in the final configuration with $N _ { f } = 4$ feature qubits and r = 20 re-uploading layers.

## Appendix 2 Training Convergence Analysis

To evaluate the generation quality and convergence behavior during training, Figure 1 presents the FID curves of the jointly trained MNIST 0–9 model and the individual digit-specific models. Since the joint dataset and the single-digit datasets have diferent target distributions and feature statistics, their FID values are not directly comparable in magnitude. Therefore, we primarily focus on the evolution of FID throughout training rather than its absolute value. To reduce random fluctuations and better illustrate the overall convergence behavior, the curves are smoothed using a moving average with a window size of five evaluation points, while the original FID curves are retained as thinner, lighter background lines to visualize the underlying training fluctuations. As shown in Figure 1, the FID decreases steadily throughout training for both the single-class models and

![](images/daef410f4f3dcf4c3038092cced7f3c15c64b4901b9853f412e0f3c9988a9c0c.jpg)  
Figure 1: Training FID curves of the jointly trained MNIST 0–9 model and the individual digit-specific models.

the jointly trained multi-class model, and gradually stabilizes in the later stages. This indicates that the proposed model is able to progressively learn the target data distribution and achieve stable convergence under both single-class and multi-class settings.