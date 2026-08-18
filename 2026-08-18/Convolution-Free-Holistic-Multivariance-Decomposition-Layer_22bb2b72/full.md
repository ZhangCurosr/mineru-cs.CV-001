# Convolution-Free Holistic Multivariance Decomposition Layer for Eficient Hyperspectral Image Classification Tensor Networks

Süha Tuna<sup>a,∗</sup> and Ülker Başar<sup>a,b</sup>

<sup>a</sup>Istanbul Technical University, Informatics Institute,

Maslak 34469, İstanbul, Türkiye

<sup>b</sup>Istanbul Esenyurt University, Faculty of Business and Management Sciences,

Department of Management Information Systems, Esenyurt, 34517, İstanbul, Türkiye

## Abstract

Feature extraction for hyperspectral image classification is conventionally addressed using rigid tensor decompositions that fail to capture complex spatio-spectral interdependencies, or heavily parameterized convolutional neural networks that are computationally expensive. To overcome these limitations, this work introduces the Holistic Multivariance Decomposition (HMD) framework as a novel, end-to-end diferentiable neural network layer. By explicitly separating independent single mode variations from cooperative higher dimensional interactions via learnable, matrix valued supports, the proposed HMD-0, HMD-1 and HMD-2 approximants are optimized jointly with a downstream classifier via backpropagation. Comprehensive evaluations across three benchmark HS datasets demonstrate that the higher level HMD layers achieve superior classification

accuracy compared to classical learnable tensor baselines, including Tucker, Canonical Polyadic, and Tensor Train decompositions. Furthermore, HMD-1 and HMD-2 achieve a generalization capacity and training stability comparable to standard 2Dand 3D-CNNs while requiring significantly fewer feature extractor parameters. These results demonstrate that the HMD framework provides a structurally robust substitute for traditional convolution in multidimensional HS image classification, ofering high parameter eficiency and stability throughout the optimization process.

## 1 Introduction

Hyperspectral (HS) imaging represents a rapidly advancing frontier in remote sensing and data acquisition [Qian, 2022]. These complex data structures are typically captured via unmanned aerial vehicles (UAVs), drones, and satellites operating at various altitudes. By integrating rich informational content across both the spatial and spectral domains, hyperspectral imagery (HSI) embeds comprehensive information specific to a target geographic area [Qian, 2022, Bhargava et al., 2024]. An HS image consists of hundreds of narrow, contiguous spectral bands. These bands facilitate the precise diferentiation of distinct objects within a region of interest. This high level of discrimination is possible because every material possesses a unique spectral signature. Because these detailed spectral signals provide fundamental information regarding underlying material components, they are widely exploited to resolve material classification problems from various fields. Consequently, the task of HS image classification has emerged as a crucial area that has consistently attracted significant scientific attention over the past decade [Tejasree and Agilandeeswari, 2024].

Beyond traditional remote sensing, HSI has found a wide variety of applications across diverse scientific disciplines. In agricultural and various industrial sectors, notable implementations of HSI include the assessment of food quality and safety [Wu and Sun, 2013, Khan et al., 2022], as well as artwork authentication [Polak et al., 2017] and the examination of counterfeit pharmaceuticals [Wilczyński et al., 2016]. The technology is also gaining attraction in biomedical engineering [Md Noor et al., 2017, Mitsui et al., 2023] and medical imaging [Karim et al., 2023] fields. Whether applied to environmental monitoring or clinical diagnostics, the dual presence of spatial and spectral data ensures robust pattern recognition [Li and Du, 2016], comprehensive image classification [Tejasree and Agilandeeswari, 2024], and precise spectral unmixing [Iordache et al., 2011].

To improve overall classification accuracy in HSI, a diverse array of feature extraction techniques has been developed and implemented [Li and Du, 2016, Bhargava et al., 2024, Tejasree and Agilandeeswari, 2024]. Broadly speaking, these methodologies fall into two primary groups, conventional mathematical schemes and advanced learning based approaches. Within conventional frameworks, many data processing techniques originally designed for standard signals and images have been adapted for HSI. For instance, classical data reduction techniques like principal component analysis (PCA) [Datta et al., 2017], linear discriminant analysis (LDA) [Fabiyi et al., 2021] and factor analysis (FA) [Li et al., 2011] are frequently applied along the spectral axis of HS images. These methods extract localized features from the corresponding spectral signals quite efectively. Although these techniques function well as feature extractors and yield satisfactory results, they fundamentally struggle to capture the complex spatial correlations inherent to HSI data.

To address this spatial limitation, researchers often apply standard feature extraction techniques designed for conventional color images [Kumar et al., 2020]. These methods are run sequentially on each individual spectral band to elicit structural information from the HS cube. Within this sequential paradigm, transform based approaches such as the discrete wavelet transform (DWT) [Tuna et al., 2022] and the discrete cosine transform (DCT) [Prabukumar et al., 2018] serve as vital analytical tools. However, despite their acknowledged success in localized feature extraction, these methods possess an inherent two dimensional nature. Consequently, they fail to simultaneously capture critical interband correlations across the spectral domain.

An alternative paradigm for feature extraction in HS imaging relies on tensor decomposition based techniques. Well known models such as the Tucker [Ji et al., 2026] and Canonical Polyadic (CP) [Jouni et al., 2019] decompositions have been widely applied to HS data. More recently, sequential core factorizations, such as the Tensor Train (TT) decomposition [Zhang et al., 2023], have also been explored for compact multidimensional representation. However, because these conventional methods depend on rigid low-rank approximations, they often fail to capture complex mode interrelations among distinct dimensions, particularly joint spatio–spectral features. To address this limitation, High Dimensional Model Representation (HDMR) [Tuna et al., 2022, Kahraman and Tuna, 2025, Tuna, 2026] and Enhanced Multivariance Products Representation (EMPR) [Tuna et al., 2021, Şen and Tuna, 2025] frameworks have been efectively employed to isolate these spatio–spectral correlations. Unfortunately, both HDMR and EMPR lack flexible rank adjustment mechanisms. This inherently restricts the scope of the extracted information. To overcome these bottlenecks, a novel framework named Holistic Multivariance Decomposition (HMD) was recently proposed. HMD extracts highly eficient spatio–spectral features from HS images. Furthermore, it introduces a manageable dimensionality reduction parameter, allowing for dynamic adjustment of the retained subspace information.

In parallel with advances in tensor based feature extraction, learning based approaches have become a dominant paradigm for HS classification. Most notably, convolutional neural networks (CNNs) operate directly on raw spatio–spectral patches using two dimensional kernels that treat spectral bands as input channels or three dimensional kernels that convolve jointly over spatial and spectral axes. These models succeed owing to their capacity to learn discriminative features end-to-end, rather than relying on a fixed algebraic decomposition of each input patch.

To introduce learnability into the HMD framework, this work builds upon the foundational formulation by recasting the HMD-0, HMD-1, and HMD-2 approximants as fully diferentiable feature extraction layers. Rather than remaining fixed a priori, their support matrices are optimized jointly with a shared downstream classifier via backpropagation. We benchmark these proposed layers against learnable Tucker, CP and TT layers of equivalent form, alongside conventional two dimensional and three dimensional CNN baselines. All methods are rigorously evaluated under a common training protocol, a shared classifier head, and a standard metric suite comprising Overall Accuracy, Average Accuracy, the macro-averaged F1-score, and Cohen’s kappa coeficient.

Beyond classification accuracy, we place particular emphasis on parameter eficiency. Because every compared method shares an identical classifier head, the number of parameters within the feature extraction layer alone provides a direct, architecture independent measure of representational cost. Consequently, the accuracy improvements delivered by each decomposition strategy can be evaluated directly against their exact parameter cost, an analysis not previously explored within the HMD framework. Finally, experiments are conducted across three benchmark HS datasets acquired by diferent sensors, spanning agricultural and urban land cover at varying spatial and spectral resolutions.

The remainder of this manuscript is organized as follows. Section 2 reviews the mathematical foundations of tensor decomposition and multivariance representation frameworks relevant to HS feature extraction, together with the deep learning based spatio-spectral classification literature. Section 3 details the proposed learnable HMD layers, the learnable Tucker, CP, and TT baselines, the 2D- and 3D-CNN baselines, and the unified classification architecture and training objective shared by all eight compared methods. Section 4 describes the benchmark datasets, preprocessing protocol, and training configuration. Section 5 presents and analyzes the comparative classification performance, parameter eficiency, training stability, and cross dataset robustness of all eight methods. Finally, Section 6 concludes the paper with a summary of key findings and directions for future research.

## 2 Background

In remote sensing and earth observation, HSI yields massive three dimensional data cubes. These structures are characterized by highly coupled spatial and spectral modes. Preserving these inherent geometric structures during feature extraction requires advanced multilinear algebra frameworks such as tensor decomposition [Kolda and Bader, 2009, Sidiropoulos et al., 2017, Wang et al., 2023]. Crucially, these frameworks must remain capable of maintaining vital spatio-spectral correlations. Consequently, researchers rely on computationally eficient tensor decompositions to uncover latent patterns. The mathematical foundations of this domain rest primarily on two classical methods. The first is the Tucker decomposition, which isolates the principal subspaces of the data by decomposing the tensor into a compact core tensor multiplied by factor matrices along each independent dimension [Ji et al., 2026]. The second is the Canonical Polyadic (CP) decomposition [Jouni et al., 2019], which represents a higher order tensor as a minimal sum of rank-one vector outer products. A related but structurally distinct approach, the Tensor Train (TT) decomposition [Zhang et al., 2023], represents a tensor as a sequential chain of lower order core tensors contracted through shared bond indices, ofering favorable scaling for higher order tensors at the cost of an inherently sequential factor structure rather than jointly multilinear.

A substantial portion of modern feature extraction relies on low-rank approximation phenomena. These techniques aim to reconstruct high dimensional tensors using a truncated summation of elementary components [Wang et al., 2023]. In Tucker and CP frameworks, these components are structurally established by computing the outer product of individual vectors drawn from each independent dimension (mode). While this classical paradigm is mathematically elegant and computationally tractable, it sufers from a severe structural limitation. Since the underlying terms are rigidly tied to pure multilinear outer products, each rank-one component can only capture the isolated, independent variations of individual modes. Therefore, traditional low-rank approximations inherently overlook the complex, mutual relationships and highly coupled interdependencies that characterize the nature of HS data. When applied to HS datasets, this independence assumption forces the decomposition to sacrifice structural fidelity. Ultimately, this limitation prevents the framework from capturing the shared cross-mode dynamics, particularly the essential spatio-spectral correlations necessary for robust feature extraction [Kolda and Bader, 2009, Sidiropoulos et al., 2017].

To overcome the limitations associated with assuming strictly decoupled modal (e.g. spatial–spatial or spatio–spectral) behaviors, recent research has pivoted toward multidimensional analysis frameworks explicitly engineered to map higher order interactions. Originating in function approximation, techniques such as HDMR decompose complex systems by systematically separating independent, single mode influences from their cooperative, higher dimensional interactions [Tuna et al., 2022, Kahraman and Tuna, 2025]. To enhance the representation capability of HDMR, the EMPR framework was introduced to provide a more descriptive representation of tensors. Unlike conventional tensor decompositions that rely strictly on rigid basis matrices, EMPR integrates flexible, one dimensional preselected support vectors directly into its structural representation [Tuna et al., 2021, Şen and Tuna, 2025].

Regardless of whether preselected or data-driven optimized support vectors are employed, the representation capability of EMPR remains inherently constrained [Tuna et al., 2021]. Because these constituent support structures are strictly one dimensional, they can only capture information along a single direction at each mode. This structural limitation introduces a critical bottleneck in EMPR implementations. Overcoming this restriction which is transitioning from vector formed to matrix formed supports so that both spatial–spatial and spatio–spectral interactions within a given HS image are jointly preserved becomes precisely the motivation for the HMD framework. As formalized in the present work, HMD further distinguishes itself from HDMR and EMPR by ensuring that every isolated hierarchical component remains a full three dimensional tensor, and by exposing a single, controllable dimensionality reduction parameter r that governs the trade-of between subspace compression and retained discriminative information.

Independently of the previously reviewed tensor algebraic frameworks, a parallel body of research has pursued spatio-spectral feature extraction through end-to-end trainable convolutional neural networks. These models learn discriminative filters directly from labeled patches rather than relying on a fixed algebraic decomposition. Early methodologies applied one dimensional convolutions exclusively along the spectral axis [Guidici and Clark, 2017]. Subsequent architectures extended this framework to two dimensional convolutions over the spatial extent of a patch, treating spectral bands as input channels [Wang et al., 2018]. Furthermore, advanced models introduced three dimensional convolutions that operate jointly across both spatial axes and the spectral dimension within a single kernel [Ahmad et al., 2020]. Patch based spatio-spectral classification, wherein a fixed size neighborhood around each labeled pixel is extracted and normalized prior to convolution, has consequently become a standard paradigm for HS image classification. This established framework represents the exact paradigm adopted for all eight feature extraction methods evaluated in this manuscript.

Relative to the tensor decomposition-based methods, 2D- and 3D-CNN feature extractors trade the explicit, interpretable low-rank structure of a tensor decomposition for the flexibility of an unconstrained, densely parameterized convolutional filter bank. This trade-of motivates the central empirical question addressed in this paper, which is whether the structured, lowrank feature extraction ofered by HMD and its tensor decomposition counterparts (Tucker, CP, TT) can match or exceed the classification accuracy of conventional 2D- and 3D-CNN feature extractors at a substantially lower parameter cost, when all methods are trained end-to-end under an identical protocol. This study formalizes all eight compared feature extraction layers in accordance with this established paradigm. Every layer operates as a diferentiable mapping from an input HS patch to a fixed length feature vector. Consequently, this architectural choice enables the comparative evaluation to be executed within a unified framework.

## 3 Methods

An HS image patch is represented as a third order tensor $\mathcal { H } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ , where the first two modes index the spatial extent of the patch and the third mode indexes the spectral bands. For a matrix $A \in \mathbb { R } ^ { n _ { i } \times r _ { i } }$ , the mode-n product of H with $A ,$ , denoted $\mathcal { H } \times _ { n } A$ , contracts the n-th mode of H against the first index of $A ,$ so that $\mathcal { H } \times _ { n } A$ tensor has size $n _ { 1 } \times \cdots \times r _ { i } \times \cdots \times n _ { 3 }$ [Kolda and Bader, 2009, Sidiropoulos et al., 2017]. Following the literature, we further adopt the compact Tucker operator notation

$$
\begin{array} { r } { [ \mathcal { H } ; A , B , C ] = \mathcal { H } \times _ { 1 } A \times _ { 2 } B \times _ { 3 } C , } \end{array}\tag{1}
$$

whenever the dimensions of H and of the mode matrices A, B, C are compatible. Note that all eight feature extraction methods compared in this work are formulated as learnable, diferentiable mappings from an input patch $\mathcal { H }$ to a fixed length feature vector, trained end-to-end with an identical downstream classifier. This unified formulation ensures that comparisons of accuracy and parameter counts isolate the efects of the feature extractor structure, preventing confounding by incidental architectural diferences.

## 3.1 Tucker, CP and Tensor Train Decomposition Layers

Three established tensor decomposition strategies were implemented as learnable feature extraction layers. Each layer is parameterized by a set of trainable factor matrices. These matrices are optimized jointly with the classifier via backpropagation. The proposed design choice remains consistent across all four tensor based methods evaluated in this study, including the proposed HMD framework. Consequently, this formulation allows their outputs to be interpreted as learned projections rather than isolated per-sample decompositions.

Given mode specific factor matrices $U _ { 1 } \in \mathbb { R } ^ { n _ { 1 } \times r _ { 1 } }$ , $U _ { 2 } \in \mathbb { R } ^ { n _ { 2 } \times r _ { 2 } }$ 2 $U _ { 3 } \in \mathbb { R } ^ { n _ { 3 } \times r _ { 3 } }$ , the Tucker

feature layer computes the low-rank core

$$
\mathcal { G } = \frac { 1 } { n _ { 1 } n _ { 2 } n _ { 3 } } \left[ \mathcal { H } ; U _ { 1 } , U _ { 2 } , U _ { 3 } \right] \in \mathbb { R } ^ { r _ { 1 } \times r _ { 2 } \times r _ { 3 } } ,\tag{2}
$$

followed by average pooling over the first two (spatial) modes to yield a $r _ { 3 }$ dimensional feature vector. This is mathematically equivalent to the zeroth level HMD term defined in Eq. 10 and is included as a conventional baseline of identical mathematical form, allowing the incremental contribution of the higher degree HMD terms to be isolated directly.

Given three factor matrices $A \in \mathbb { R } ^ { n _ { 1 } \times R }$ ， $B \in \mathbb { R } ^ { n _ { 2 } \times R }$ ， $C \in \mathbb { R } ^ { n _ { 3 } \times R }$ sharing a common rank R, the CP feature layer contracts the input patch against R rank-one detector tensors formed from the columns of A, B, and C as follows

$$
\mathbf { v } _ { r } = \frac { 1 } { n _ { 1 } n _ { 2 } n _ { 3 } } \sum _ { i , j , k } \mathcal { H } _ { i j k } A _ { i r } B _ { j r } C _ { k r } , \qquad r = 1 , \ldots , R\tag{3}
$$

yielding a feature vector $\mathbf { v } \in \mathbb { R } ^ { R }$ . Unlike a complete CP reconstruction of the input tensor, Equation 3 is utilized exclusively as a feature extraction operator. Under this formulation, each resulting feature element quantifies the alignment of the HS patch with a specific rank-one spatio-spectral pattern. This mechanism functions analogously to applying a set of separable matched filters determined by the decomposition rank.

Given TT cores $G _ { 1 } \in \mathbb { R } ^ { n _ { 1 } \times R _ { 1 } } , G _ { 2 } \in \mathbb { R } ^ { R _ { 1 } \times n _ { 2 } \times R _ { 2 } }$ , and $G _ { 3 } \in \mathbb { R } ^ { R _ { 2 } \times n _ { 3 } \times R _ { 3 } }$ , the TT feature layer performs the following sequential contraction

$$
\mathcal { T } = \mathcal { H } \times _ { 1 } G _ { 1 } , \qquad \mathcal { T } ^ { \prime } = \mathcal { T } \times _ { ( 1 , 2 ) } G _ { 2 } , \qquad \mathbf { v } = \mathcal { T } ^ { \prime } \times _ { ( 1 , 2 ) } G _ { 3 } ,\tag{4}
$$

where $\times _ { ( 1 , 2 ) }$ denotes simultaneous contraction over the bond index and the corresponding spatio-spectral mode, yielding a feature vector $\mathbf { v } \in \mathbb { R } ^ { R _ { 3 } }$ . This is the standard tensor train contraction scheme adapted here to act as a feature extraction operator rather than a full tensor approximation.

## 3.2 Holistic Multivariance Decomposition

Although Tucker and CP decompositions yield compact, single level tensor summaries and the TT approach models sequential mode dependencies, these representations lack the capacity to explicitly isolate the degree of spatio-spectral interaction. In particular, they do not diferentiate between baseline zeroth order, independent single-mode first order, and interactive cross-mode second order variations within a given patch. To resolve this limitation, this work introduces the Holistic Multivariance Decomposition (HMD) framework. This approach fundamentally extends the High Dimensional Model Representation and Enhanced Multivariance Products Representation architectures. Rather than relying on one dimensional, vector valued support structures that restrict each factor to a single direction per mode, the proposed method introduces matrix valued supports capable of comprehensively capturing spatial-spatial and spatio-spectral interactions.

For each mode index, the HMD defines a support matrix with a reduced column dimension, represented as

$$
U _ { i } \in \mathbb { R } ^ { n _ { i } \times r _ { i } } , \quad r _ { i } < n _ { i } , \quad i \in \{ 1 , 2 , 3 \}\tag{5}
$$

which is constrained to be scaled semi-orthogonal according to the conditions

$$
U _ { i } ^ { T } U _ { i } = n _ { i } I _ { r _ { i } \times r _ { i } } .\tag{6}
$$

Letting the complete support set be denoted by

$$
\mathbf { U } = \{ U _ { 1 } , U _ { 2 } , U _ { 3 } \} ,\tag{7}
$$

while the subsets excluding the i-th and the i-th and j-th matrices are defined respectively as

$$
\mathbf { U } ^ { ( i ) } = \{ U _ { k } \mid k \neq i \} , \qquad \mathbf { U } ^ { ( i , j ) } = \{ U _ { k } \mid k \neq i , j \} .\tag{8}
$$

Using the Tucker operator, the three dimensional HMD expansion decomposes the original input tensor into a hierarchy of multivariance components, expressed as follows

$$
\mathcal { H } = [ [ \mathcal { H } _ { 0 } ; \mathbf { U } ] ] + \sum _ { i } \left[ \left[ \mathcal { H } _ { i } ; \mathbf { U } ^ { ( i ) } \right] \right] + \sum _ { i < j } \left[ \left[ \mathcal { H } _ { i , j } ; \mathbf { U } ^ { ( i , j ) } \right] \right] + \mathcal { H } _ { 1 , 2 , 3 } .\tag{9}
$$

In this hierarchical formulation, the zeroth degree component represents baseline spatiospectral interaction, the three first degree components capture pure single-mode variation, the three second degree components capture pairwise cross-mode interactions, and the final term serves as the residual. Unlike HDMR and EMPR, every isolated HMD component remains a full three dimensional tensor, ensuring that spatial and spectral structures within each degree of interaction are preserved rather than collapsed.

Exploiting the commutativity of the mode product [Kolda and Bader, 2009, Sidiropoulos et al., 2017] and the scaled semi-orthogonality identity in Eq. (6), the individual HMD components can be explicitly obtained by successive deflations. The zeroth degree term compresses the input tensor into a dense core that retains the global multidimensional structure of the patch, calculated as

$$
\mathcal { H } _ { 0 } = \frac { 1 } { n _ { 1 } n _ { 2 } n _ { 3 } } \left[ \mathcal { H } ; U _ { 1 } , U _ { 2 } , U _ { 3 } \right] \in \mathbb { R } ^ { r _ { 1 } \times r _ { 2 } \times r _ { 3 } } .\tag{10}
$$

The first degree terms isolate linear variation along a single mode after subtracting the corresponding zeroth degree projection, formulated as

$$
\begin{array} { r l } & { \mathcal { H } _ { 1 } = \displaystyle \frac { 1 } { n _ { 2 } n _ { 3 } } \left[ \mathcal { H } ; U _ { 2 } , U _ { 3 } \right] - \left[ \mathcal { H } _ { 0 } ; U _ { 1 } \right] \in \mathbb { R } ^ { n _ { 1 } \times r _ { 2 } \times r _ { 3 } } , } \\ & { \mathcal { H } _ { 2 } = \displaystyle \frac { 1 } { n _ { 1 } n _ { 3 } } \left[ \mathcal { H } ; U _ { 1 } , U _ { 3 } \right] - \left[ \mathcal { H } _ { 0 } ; U _ { 2 } \right] \in \mathbb { R } ^ { r _ { 1 } \times n _ { 2 } \times r _ { 3 } } , } \\ & { \mathcal { H } _ { 3 } = \displaystyle \frac { 1 } { n _ { 1 } n _ { 2 } } \left[ \mathcal { H } ; U _ { 1 } , U _ { 2 } \right] - \left[ \mathcal { H } _ { 0 } ; U _ { 3 } \right] \in \mathbb { R } ^ { r _ { 1 } \times r _ { 2 } \times n _ { 3 } } . } \end{array}\tag{11}
$$

The second degree terms isolate pairwise cross-mode interactions by further deflating the

zeroth and first degree contributions, defined as

$$
\begin{array} { r l } & { \mathcal { H } _ { 1 , 2 } = \displaystyle \frac { 1 } { n _ { 3 } } \left[ \mathcal { H } ; U _ { 3 } \right] - \left[ \mathcal { H } _ { 0 } ; U _ { 1 } , U _ { 2 } \right] - \left[ \mathcal { H } _ { 1 } ; U _ { 2 } \right] - \left[ \mathcal { H } _ { 2 } ; U _ { 1 } \right] \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times r _ { 3 } } , } \\ & { \mathcal { H } _ { 1 , 3 } = \displaystyle \frac { 1 } { n _ { 2 } } \left[ \mathcal { H } ; U _ { 2 } \right] - \left[ \mathcal { H } _ { 0 } ; U _ { 1 } , U _ { 3 } \right] - \left[ \mathcal { H } _ { 1 } ; U _ { 3 } \right] - \left[ \mathcal { H } _ { 3 } ; U _ { 1 } \right] \in \mathbb { R } ^ { n _ { 1 } \times r _ { 2 } \times n _ { 3 } } , } \\ & { \mathcal { H } _ { 2 , 3 } = \displaystyle \frac { 1 } { n _ { 1 } } \left[ \mathcal { H } ; U _ { 1 } \right] - \left[ \mathcal { H } _ { 0 } ; U _ { 2 } , U _ { 3 } \right] - \left[ \mathcal { H } _ { 2 } ; U _ { 3 } \right] - \left[ \mathcal { H } _ { 3 } ; U _ { 2 } \right] \in \mathbb { R } ^ { r _ { 1 } \times n _ { 2 } \times n _ { 3 } } . } \end{array}\tag{12}
$$

With the index set defined across all three modes, the complete expansion admits a compact summation form, expressed as

$$
\mathcal { H } = \sum _ { I \subseteq \{ 1 , 2 , 3 \} } \left[ \mathcal { H } _ { I } ; \mathbf { U } _ { I } \right] .\tag{13}
$$

In practical applications, computing the full expansion is computationally expensive. Thus, truncating the expansion at a chosen degree yields the specialized approximants used as feature extractors in this study.

Three specific truncation levels are integrated as feature extraction layers. The zeroth level approximant, denoted as HMD-0, retains only the baseline term and is mathematically identical to the standard Tucker layer defined in Eq. (2). The first level approximant, denoted as HMD-1, retains the baseline term together with the three first degree terms. The second level approximant, denoted as HMD-2, retains the baseline term, all three first degree terms, and all three second degree terms (see Eq. (9)). Because each component tensor beyond the zeroth degree baseline retains at least one unprojected mode of original dimension, a global average pooling operator $\varphi ( \cdot )$ is applied over the two leading axes of every component to obtain a fixed length vector suitable for concatenation.

Given a component $\mathcal { H } _ { I }$ of dimensions $n _ { i _ { 1 } } \times n _ { i _ { 2 } } \times n _ { i _ { 3 } }$ , the operation $\varphi ( \cdot )$ applied to $\mathcal { H } _ { I }$ is explicitly formulated as follows

$$
\varphi ( \mathcal { H } _ { I } ) = \frac { 1 } { n _ { i _ { 1 } } n _ { i _ { 2 } } } \sum _ { i _ { 1 } } \sum _ { i _ { 2 } } \mathcal { H } _ { I } \left( i _ { 1 } , i _ { 2 } , \cdot \right)\tag{14}
$$

The final layer output for a given truncation level is formed by concatenating the pooled components up to that degree, such that the feature vectors for HMD-0, HMD-1 and HMD-2 take the following explicit forms

$$
\mathbf { v } ^ { ( \mathrm { H M D - 0 } ) } = \Big [ \varphi ( \mathcal { H } _ { 0 } ) \Big ] ,\tag{15}
$$

$$
\begin{array} { r } { \mathbf { v } ^ { ( \mathrm { H M D - 1 } ) } = \Big [ \varphi ( \mathcal { H } _ { 0 } ) ~ \varphi ( \mathcal { H } _ { 1 } ) ~ \varphi ( \mathcal { H } _ { 2 } ) ~ \varphi ( \mathcal { H } _ { 3 } ) \Big ] , } \end{array}\tag{16}
$$

and

$$
\mathbf { v } ^ { ( \mathrm { H M D - 2 } ) } = \Big [ \varphi ( \mathcal { H } _ { 0 } ) \varphi ( \mathcal { H } _ { 1 } ) \varphi ( \mathcal { H } _ { 2 } ) \varphi ( \mathcal { H } _ { 3 } ) \varphi ( \mathcal { H } _ { 1 , 2 } ) \varphi ( \mathcal { H } _ { 1 , 3 } ) \varphi ( \mathcal { H } _ { 2 , 3 } ) \Big ] ,\tag{17}
$$

respectively.

A structurally vital consequence of this formulation is that all three approximants share an identical set of trainable support matrices. Consequently, increasing the truncation level only alters which precomputed hierarchical components are pooled into the output feature vector, without increasing the number of learnable parameters in the feature extractor. This structural economy drives the parameter eficiency of HMD approximants, allowing higher order truncations to achieve superior accuracy at an identical parameter count. Because additional first and second degree terms derive from support matrices already required by the zeroth degree baseline rather than newly introduced weights, higher level approximants yield substantial performance gains without increasing the parameter footprint of the feature extractor.

The scaled semi-orthogonality condition in Eq. (6) is not imposed as a hard constraint during training, since the support matrices are learned end-to-end rather than obtained via a predetermined support generation process. Instead, semi-orthogonality is encouraged through a soft penalty added to the training loss for all three HMD approximants and the Tucker baseline, expressed as

$$
\mathcal { L } _ { \mathrm { o r t h o } } = \lambda \sum _ { i = 1 } ^ { 3 } \left. \boldsymbol { U } _ { i } ^ { T } \boldsymbol { U } _ { i } - n _ { i } \boldsymbol { I } _ { r _ { i } } \right. _ { F } ^ { 2 } ,\tag{18}
$$

where the scaling parameter λ represents a fixed regularization coeficient. An analogous Frobenius norm based weight penalty is applied to the TT cores and to the CP factor matrices scaled to their respective dimensions for those two baseline architectures, whereas no such penalty is required for the convolutional baselines.

## 3.3 Convolutional Neural Network Baselines

To situate the tensor decomposition layers relative to conventional spatial-spectral deep learning approaches, two convolutional feature extractors were implemented using minimal filter configurations. This restriction prevents either baseline’s parameter count from trivially dominating the comparative experiments.

The 2D-CNN baseline treats the spectral dimension of the patch as input channels and applies two 2-D convolutional blocks of size $k \times k$ as

$$
X _ { 1 } = \sigma \Big ( \mathrm { B N } \big ( \mathrm { C o n v 2 D } _ { k \times k } ( \mathcal { H } ) \big ) \Big ) , \qquad X _ { 2 } = \sigma \Big ( \mathrm { B N } \big ( \mathrm { C o n v 2 D } _ { k \times k } ( X _ { 1 } ) \big ) \Big )\tag{19}
$$

followed by global average pooling over the two spatial dimensions to yield the feature vector, where σ denotes the ReLU activation and BN denotes batch normalization.

The 3D-CNN baseline instead reshapes the patch into a single-channel volume $\mathcal { H } ^ { \prime } \in$ R<sup>n</sup>1×<sup>n</sup>2×<sup>n</sup>3×<sup>1</sup> and applies two 3-D convolutional blocks of size $k \times k \times k$

$$
X _ { 1 } = \sigma \Big ( \mathrm { B N } \big ( \mathrm { C o n v 3 D } _ { k \times k \times k } ( \mathcal { H } ^ { \prime } ) \big ) \Big ) , \qquad X _ { 2 } = \sigma \Big ( \mathrm { B N } \big ( \mathrm { C o n v 3 D } _ { k \times k \times k } ( X _ { 1 } ) \big ) \Big )\tag{20}
$$

followed by global average pooling over all three spatio-spectral dimensions. Unlike the 2D-CNN, this formulation convolves jointly over both spatial axes and the spectral axis, allowing local spatio-spectral correlations to be captured directly by the kernel rather than combined only at the channel aggregation step.

## 3.4 Unified Classification Architecture and Training Objective

All eight feature extraction methods described above are followed by an identical classifier head, so that classification performance and parameter count can be attributed to the feature extraction stage alone. The extracted feature vector v is passed through two fully connected blocks with batch normalization, ReLU activation, and dropout, followed by a final softmax layer as

$$
\begin{array} { r } { \hat { y } = \mathrm { s o f t m a x } \left\{ W _ { 3 } \cdot \mathrm { D r o p o u t } \left[ \sigma \Big ( \mathrm { B N } \left( W _ { 2 } \cdot \mathrm { D r o p o u t } [ \sigma ( \mathrm { B N } ( W _ { 1 } \cdot \mathbf { v } ) ) ] \right) \Big ) \right] \right\} , } \end{array}\tag{21}
$$

with layer widths and dropout rate given in Section 4. The complete network including feature extractor and the classifier head is trained end-to-end by minimizing the sparse categorical cross-entropy loss between $\hat { y }$ and the ground truth label, plus the semi-orthogonality penalty in Eq. (18) whose explicit definition is as follows

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { C E } } \left( y , \hat { y } \right) + \mathcal { L } _ { \mathrm { o r t h o } } .\tag{22}
$$

Training hyperparameters (optimizer, learning rate, batch size, number of epochs, and the regularization coeficient λ) are held fixed across all eight methods and are reported together with the datasets and evaluation protocol in Section 4.

## 4 Experimental Setup

## 4.1 Datasets and Preprocessing

The proposed feature extraction strategies were evaluated on three widely adopted benchmark HS datasets spanning agricultural, urban, and mixed land-cover settings which are Indian Pines, Pavia University and Loukia [Xu et al., 2023]. Table 1 summarizes their properties.

The Indian Pines scene was acquired by the Airborne Visible/Infrared Imaging Spectrometer over an agricultural test site in northwestern Indiana, United States. It consists of $1 4 5 \times 1 4 5$ pixels at a spatial resolution of approximately 20 m, spanning the 0.4 to 2.5 µm wavelength range across 200 retained spectral bands, and is annotated with 16 land-cover classes dominated by row-crop agriculture and forest cover. The Pavia University scene was acquired by the Reflective Optics System Imaging Spectrometer over the campus of the University of Pavia, Italy, at a spatial resolution of 1.3 m. It comprises $6 1 0 \times 3 4 0$ pixels across 103 spectral bands within the 0.43 to 0.86 µm range and is labeled with nine urban land-cover classes, including asphalt, meadows, bitumen, and bricks. The Loukia scene, part of the HyRANK benchmark collected by the Hyperion sensor over Crete, Greece, provides a lower spatial resolution (30 m) and higher class count (14 classes across 176 spectral bands over a $2 5 0 \times 1 3 7 6$ pixel extent) complement to the first two datasets. This dataset serves to assess the generality and robustness of each method beyond the conventional Indian Pines and Pavia University pairing that dominates prior tensor decomposition studies.

Table 1: Summary of the HS datasets used in this study.
<table><tr><td>Dataset</td><td>Sensor</td><td>Spatial Size</td><td>Spectral Bands</td><td>Classes</td></tr><tr><td>Indian Pines</td><td>AVIRIS</td><td> $\overline { { 1 4 5 \times 1 4 5 } }$ </td><td>200</td><td>16</td></tr><tr><td>Pavia University</td><td>ROSIS</td><td> $6 1 0 \times 3 4 0$ </td><td>103</td><td>9</td></tr><tr><td>Loukia</td><td>Hyperion</td><td> $2 5 0 \times 1 3 7 6$ </td><td>176</td><td>14</td></tr></table>

![](images/30a3442099a89b5f5bffee0f279d0602a1adb7e21fda2322631bbec194f71e9d.jpg)  
Figure 1: Ground truth classification maps for the three benchmark HS datasets, from left to right: Indian Pines, Pavia University and Loukia. Black pixels in all images represent unlabelled background regions excluded from training and evaluation.

The corresponding ground truth images for the considered datasets are illustrated in

Figure 1.

For each dataset, spectral bands were standardized to zero mean and unit variance per band across the full scene prior to spatial processing. The standardized cube was zero padded along its spatial dimensions to preserve border pixels, after which a spatio-spectral patch of size $1 1 \times 1 1 \times B$ , where B denotes the number of spectral bands, was extracted around every labeled pixel following the standard spatio-spectral patch based classification paradigm for HSI. Unlabeled background pixels were excluded from the sample set. Labeled samples were then partitioned into training and test sets using a stratified 80/20 split so that class proportions were strictly preserved across partitions. A fixed random seed was applied across all partitioning and weight initialization operations to ensure that comparative evaluations across feature extraction methods were not confounded by sampling variance.

## 4.2 Network Architecture and Training Configuration

To isolate the efect of the feature extraction stage on classification performance, all compared methods share an identical downstream classifier head, difering solely in how the raw $1 1 \times 1 1 \times B$ patch is reduced to a compact feature vector. Eight feature extraction strategies were evaluated, spanning the three variants of the proposed HMD family, four established tensor decomposition baselines, and two convolutional baselines.

Within the proposed HMD family, the zeroth level HMD (HMD-0) performs a Tucker style multilinear projection of the patch onto a low-rank core with dimensions $( r _ { 1 } , r _ { 2 } , r _ { 3 } )$ spatially pooled to a single descriptor. The first level approximant (HMD-1) extends HMD-0 by incorporating first order per-mode terms, analogous to the main efects of a functional analysis of variance decomposition of the input tensor. The second level approximant (HMD-2) further extends HMD-1 with second order pairwise-mode interaction terms, yielding the fullest holistic expansion among the three hierarchical variants.

The classical tensor baselines include the Tucker decomposition, representing a standard multilinear core projection included alongside the zeroth level HMD as a conventional baseline of equivalent mathematical form; the CP decomposition, which applies a single rank-r polyadic factorization to the patch tensor; and the TT decomposition, which performs a sequential three core contraction of the patch tensor.

![](images/9e43d46df0fffe24025f5b21eea911b88108e4aeabc5c6ca18163d42ec30c43e.jpg)  
Figure 2: Schematic representation of the comparative deep learning pipeline for HS image classification. The architecture highlights the eight interchangeable feature extraction layers including the proposed HMD family, classical tensor models, and CNN baselines feeding into an identical, shared classifier head.

To benchmark the tensor based layers against conventional spatio-spectral deep learning approaches, two convolutional feature extractors were implemented. The two dimensional convolutional neural network (2D-CNN) baseline employs a two layer architecture with sequential convolution, batch normalization, and activation blocks, followed by global average pooling that treats spectral bands as input channels. The three dimensional (3D-CNN) baseline applies a volumetric two layer architecture to the patch reshaped as a single channel volume, convolving jointly over spatial and spectral dimensions. This volumetric model evaluates whether modeling spatio-spectral locality via three dimensional kernels outperforms channel-wise spectral compression or low-rank decompositions.

Both convolutional baselines utilize minimal filter configurations to prevent their parameter counts from trivially dominating the evaluation. Specifically, the 2D-CNN employs 8 and 16 channels with 3 × 3 kernels, while the 3D-CNN uses 4 and 8 channels with $3 \times 3 \times 3$ kernels across their respective layers. For all five tensor decomposition methods, a soft semi-orthogonality penalty on the mode-projection matrices or an analogous Frobenius norm weight penalty on the TT cores was incorporated into the training loss using a regularization coeficient λ. All decomposition ranks and convolutional filter counts remained fixed across datasets to strictly isolate the structural eficacy of the feature extractors.

Each feature extraction module is followed by an identical classifier head. The extracted feature vector passes through two fully connected blocks containing 128 and 64 units, respectively. Each block includes batch normalization, ReLU activation, and a 0.3 dropout rate, culminating in a final softmax layer sized to the target classes. Input patches were batch normalized prior to feature extraction.

Models were trained using the Adam optimizer with an initial learning rate of 0.001 for 50 epochs and a batch size of 32. The objective function minimized the sparse categorical cross-entropy loss, combined with the orthogonality penalty where applicable. Neither early stopping nor learning rate scheduling was utilized. Each method and dataset combination was evaluated independently using identical data splits and initialization seeds.

## 4.3 Evaluation Metrics

Model performance was assessed on the held-out test partition using four complementary metrics in accordance with standard practice in the HS image classification literature. Overall Accuracy (OA) measures the proportion of correctly classified test pixels across all classes. Average Accuracy (AA) calculates the unweighted mean of per-class recall, giving equal weight to minority and majority classes and thereby mitigating class imbalance bias. The F1-score (F1) computes the unweighted mean of per-class F1-scores to jointly capture precision and recall. Finally, Cohen’s kappa (κ) coeficient provides a chance corrected measure of agreement between predicted and reference labels.

In addition to classification performance, the total number of trainable parameters was recorded for each configuration and explicitly decomposed into the parameters contained within the feature extraction module alone. This strategy allows the accuracy achieved by each HMD and baseline tensor layer to be interpreted relative to its parametric cost, enabling a direct eficiency comparison against the convolutional baselines. Because convolutional filters are inherently more parameter dense than the low-rank factor matrices utilized by tensor based methods, particularly for three dimensional kernels whose parameter count scales with the cube of the kernel size, both convolutional baselines were configured with the smallest filter counts that preserved a functioning two layer architecture, ensuring that the reported comparison reflects true accuracy-per-parameter eficiency.

All models were implemented in Python using the TensorFlow and Keras deep learning frameworks. Experiments were conducted under a Linux Ubuntu 24.04 LTS operating system equipped with an NVIDIA RTX 5060-Ti with 16 GB VRAM to ensure consistent computational benchmarking. To guarantee reproducibility, all NumPy and TensorFlow random seeds were fixed. The complete configuration for every experimental run was logged alongside per-epoch training curves and final test metrics.

## 5 Results

## 5.1 Overall Classification Performance

Table 2, Table 3 and 4 report the test set overall accuracy (OA), average accuracy (AA), macro F1 score (F1), and Cohen’s kappa (κ) for all eight feature extraction strategies across the three datasets. The proposed HMD layers secured the top two positions across all metrics and scenes, with one exception on Pavia University. For this case, the 2D-CNN baseline achieved a marginally higher OA and AA than HMD-2 (99.88% versus 99.31% OA), albeit requiring over an order of magnitude more parameters. Specifically, HMD-1 delivered the best overall performance on Indian Pines with $\mathrm { O A } = 9 9 . 8 5 \%$ and $\kappa = 0 . 9 9 8$

Table 2: Test-set classification performance on Indian Pines. Best value in bold; second-best underlined.
<table><tr><td>Method OA (%)</td><td colspan="3">AA (%) F1 (macro)</td><td>κ</td></tr><tr><td>HMD-0</td><td>94.20</td><td>94.67</td><td>0.9421</td><td>0.9339</td></tr><tr><td>HMD-1</td><td>99.85</td><td>99.90</td><td>0.9975</td><td>0.9983</td></tr><tr><td>HMD-2</td><td>99.41</td><td>99.64</td><td>0.9935</td><td>0.9933</td></tr><tr><td>Tucker</td><td>90.88</td><td>90.52</td><td>0.9110</td><td>0.8960</td></tr><tr><td>CP</td><td>74.93</td><td>64.56</td><td>0.6297</td><td>0.7159</td></tr><tr><td>TT</td><td>98.05</td><td>98.12</td><td>0.9825</td><td>0.9777</td></tr><tr><td>2D-CNN</td><td>94.00</td><td>96.43</td><td>0.9667</td><td>0.9319</td></tr><tr><td>3D-CNN</td><td>96.39</td><td>92.98</td><td>0.9288</td><td>0.9589</td></tr></table>

Furthermore, HMD-2 achieved the highest tensor based result on Pavia University with $\mathrm { O A } = 9 9 . 3 1 \%$ with $\kappa = 0 . 9 9 1$ alongside the top overall performance on Loukia with $\mathrm { O A } =$ 94.82% and $\kappa = 0 . 9 3 8$

The ultimate comparison among all feature extractors are provided in Table 5. Averaged across the three datasets, HMD-2 achieved the highest mean OA with 97.85%, AA with

Table 3: Test-set classification performance on Pavia University. Best value in bold; secondbest underlined.
<table><tr><td>Method</td><td>OA (%) AA (%)</td><td>F1 (macro)</td><td>κ</td></tr><tr><td>HMD-0</td><td>78.87 70.78</td><td>0.7017</td><td>0.7229</td></tr><tr><td>HMD-1</td><td>98.11 97.31</td><td>0.9691</td><td>0.9749</td></tr><tr><td>HMD-2</td><td>99.31 98.91</td><td>0.9882</td><td>0.9909</td></tr><tr><td>Tucker</td><td>86.40 74.52</td><td>0.6981</td><td>0.8175</td></tr><tr><td>CP</td><td>89.61 82.16</td><td>0.8374</td><td>0.8590</td></tr><tr><td>TT</td><td>65.31</td><td>59.32 0.5191</td><td>0.5438</td></tr><tr><td>2D-CNN</td><td>99.88</td><td>99.74 0.9980</td><td>0.9985</td></tr><tr><td>3D-CNN</td><td>99.52</td><td>99.15 0.9904</td><td>0.9937</td></tr></table>

Table 4: Test-set classification performance on Loukia. Best value in bold; second-best underlined.
<table><tr><td colspan="4">Method OA (%) AA (%) F1 (macro) κ</td></tr><tr><td>HMD-0</td><td>79.34</td><td>70.77</td><td>0.7190 0.7554</td></tr><tr><td>HMD-1</td><td>91.23</td><td>87.86 0.8985</td><td>0.8950</td></tr><tr><td>HMD-2</td><td>94.82</td><td>90.99 0.9292</td><td>0.9384</td></tr><tr><td>Tucker</td><td>73.27 71.33</td><td>0.6737</td><td>0.6823</td></tr><tr><td>CP</td><td>80.27</td><td>62.78 0.6490</td><td>0.7605</td></tr><tr><td>TT</td><td>78.71</td><td>71.94 0.7342</td><td>0.7409</td></tr><tr><td>2D-CNN</td><td>90.82</td><td>83.27 0.8698</td><td>0.8896</td></tr><tr><td>3D-CNN</td><td>77.82</td><td>69.60 0.6976</td><td>0.7365</td></tr></table>

96.51%, F1 with 0.9703 and κ with 0.9742 of any method evaluated, followed by HMD-1. The 2D-CNN baseline ranked third on all four metrics, and the 3D-CNN baseline fourth. The remaining tensor decomposition baselines which are Tucker, CP and TT trailed the HMD family and the CNN baselines by a substantial margin on every metric, indicating that neither a bare multilinear projection (Tucker) nor a single term factorization (CP) is suficient to capture the spatio-spectral structure of these scenes. Therefore, the first and second order interaction terms introduced by HMD-1 and HMD-2 account for the majority of the accuracy gain over the Tucker/HMD-0 baseline of equivalent zeroth order form.

Table 5: Mean performance across the three datasets, ranked by mean overall accuracy, alongside mean feature-extractor parameter count.
<table><tr><td>Rank</td><td>Method</td><td>OA (%)</td><td>AA (%)</td><td>F1</td><td>K</td><td>Parameters</td></tr><tr><td>1</td><td>HMD-2</td><td>97.85</td><td>96.51</td><td>0.9703</td><td>0.9742</td><td>908</td></tr><tr><td>2</td><td>HMD-1</td><td>96.40</td><td>95.02</td><td>0.9550</td><td>0.9561</td><td>908</td></tr><tr><td>3</td><td>2D-CNN</td><td>94.90</td><td>93.15</td><td>0.9449</td><td>0.9400</td><td>12,768</td></tr><tr><td>4</td><td>3D-CNN</td><td>91.24</td><td>87.25</td><td>0.8723</td><td>0.8963</td><td>1,032</td></tr><tr><td>5</td><td>HMD-0</td><td>84.13</td><td>78.74</td><td>0.7876</td><td>0.8041</td><td>908</td></tr><tr><td>6</td><td>Tucker</td><td>83.51</td><td>78.79</td><td>0.7609</td><td>0.7986</td><td>908</td></tr><tr><td>7</td><td>CP</td><td>81.60</td><td>69.83</td><td>0.7054</td><td>0.7785</td><td>908</td></tr><tr><td>8</td><td>TT</td><td>80.69</td><td>76.46</td><td>0.7453</td><td>0.7541</td><td>4,322</td></tr></table>

## 5.2 Subspace Dimensionality and Rank Sensitivity Analysis

To rigorously evaluate the robustness of the extracted spatio-spectral features under varying levels of data compression, the classification performance of the proposed frameworks was analyzed across the subspace dimensionality reduction parameter $r \in [ 2 , 1 0 ]$ . The sensitivity of the OA and AA metrics for the Indian Pines, Pavia University, and Loukia datasets is illustrated in Figure 3, Figure 4 and Figure 5, respectively.

As depicted in the sensitivity curves for the Indian Pines dataset, the HMD-1 and HMD-2 establish overwhelming dominance and exceptional stability across the entire evaluated rank spectrum. While the classical Tucker decomposition and the zeroth level HMD-0 exhibit highly volatile behavior characterized by severe performance degradation at $r = 4$ and a near total subspace collapse at $r = 8$ where the higher level HMD frameworks maintain near optimal accuracy. This rapid convergence and sustained stability confirm that explicitly isolating linear directional multivariance components efectively shields the discriminative feature space from noise introduced by rank variations.

This behavioral disagreement is further amplified in the highly textured, high resolution urban geometries of the Pavia University scene. HMD-2 consistently achieves the highest classification accuracy, with HMD-1 providing a robust secondary baseline. The structural preservation of intricate, cross domain spatio-spectral interactions within HMD-2 proves highly resilient to parameter tuning. In contrast, the core tensor projections (HMD-0 and Tucker) display erratic oscillations, demonstrating that rigid low-rank approximations struggle to consistently encapsulate complex spatial dependencies. Furthermore, the CP decomposition plateaus at a significantly lower accuracy ceiling, reafirming its structural inability to map coupled multidimensional variations.

![](images/7afb36d2ec94eb213be715036bc217e5953b592bdc17f32bebbfdafb6e311c04.jpg)  
Figure 3: Sensitivity analysis of Overall Accuracy (OA) and Average Accuracy (AA) with respect to the subspace dimensionality reduction parameter (r) on the Indian Pines dataset.

Finally, the evaluation on the Loukia HS scene substantiates the robust generalization of the proposed method. HMD-2 remains the superior framework, closely tracking with HMD-1 to yield highly stable, discriminative feature sets regardless of the chosen subspace dimension. Meanwhile, the HMD-0 and Tucker models once again exhibit tightly coupled, erratic performance trajectories, cratering notably around r = 7.

Collectively, these results validate that the systematic extraction of higher degree multi variance terms successfully separates essential spatio-spectral features from latent noise. By overcoming the limitations of purely core dependent or strictly decoupled tensor approximations, the HMD framework ensures high fidelity HS classification even under volatile or tightly constrained subspace compressions.

![](images/09a04d44549f4b3849fdd4966ab64edfc3fbc7e88f1b6db0af1ac40531d5b59d.jpg)  
Figure 4: Sensitivity analysis of Overall Accuracy (OA) and Average Accuracy (AA) with respect to the subspace dimensionality reduction parameter (r) on the Pavia University dataset.

## 5.3 Training Stability and Convergence

To evaluate the learning behavior, optimization stability, and generalization capacity of the proposed frameworks, the epoch wise training and validation curves were analyzed across the benchmark datasets. The accuracy and loss trajectories over 50 training epochs are presented for the three datasets under consideration.

Across all evaluated scenes, the heavily parameterized 2D-CNN baseline exhibits rapid training convergence, efectively memorizing spatio-spectral patterns due to its large parameter footprint. Among the tensor based architectures, the higher level HMD models, HMD-1 and HMD-2, demonstrate the strongest convergence, smoothly minimizing the training loss and closely tracking the CNN baseline’s trajectory. Conversely, models relying on a rigid core tensor such as the classical Tucker decomposition, TT and HMD-0 exhibit severe validation instability. This instability is characterized by abrupt drops in accuracy and massive loss spikes, indicating a deep vulnerability to subspace collapse during gradient descent.

![](images/430dd2f08017d99a6e04cda7d3a3315805666d8019036aa9a757d9ea193e4de3.jpg)  
Figure 5: Sensitivity analysis of Overall Accuracy (OA) and Average Accuracy (AA) with respect to the subspace dimensionality reduction parameter (r) on the Loukia dataset.

For the Indian Pines dataset, which is characterized by mixed agricultural parcels and high spectral similarity among vegetation classes, the training and validation curves illustrate the critical necessity of decoupling directional variations. While the CNN and higher level HMD models maintain stable validation trajectories that consistently exceed 90% accuracy, the core dependent frameworks (Tucker, HMD-0, and TT) experience catastrophic validation drops, occasionally decreasing below 50%. This extreme volatility suggests that forcing highly mixed agricultural spectral signatures through a single dense core tensor exacerbates gradient perturbations during backpropagation. By explicitly isolating spatial and spectral fibers using $\mathcal { H } _ { 1 } , \mathcal { H } _ { 2 } , \mathcal { H } _ { 3 }$ , HMD-1 efectively insulates the decision boundaries from this latent noise, yielding the smooth and robust generalization necessary for agricultural classification.

![](images/fda413361eb1869cf200d867a7d04103456668c273a9ff07e0ba5901ac39cbb4.jpg)  
Figure 6: Training/validation accuracy and loss curves, all eight methods, Indian Pines dataset.

The learning dynamics on the Pavia University scene further emphasize the structural advantages of the proposed framework when processing high resolution urban geometries. This dataset contains highly textured structures, localized shadows, and intricate spatial borders. The validation curves reveal that while the CP decomposition converges smoothly, it fundamentally plateaus at a low accuracy ceiling due to its structural inability to map coupled multidimensional variations. Meanwhile, Tucker, TT, and HMD-0 sufer from repeated, drastic destabilization. HMD-2, however, successfully mirrors the stable generalization of the heavily parameterized CNN. By explicitly modeling pairwise cross-domain interactions $( \mathcal { H } _ { 1 , 2 } , \mathcal { H } _ { 1 , 3 } , \mathcal { H } _ { 2 , 3 } )$ , HMD-2 directly resolves the complex spatio-spectral dependencies inherent to urban materials, providing a highly stable optimization landscape that does not collapse under fine grained spatial texture variations.

![](images/fac91cd176ccd9ae3b134a111956cdbe654fce506c98cd24d77af881dc4a9956.jpg)  
Figure 7: Training/validation accuracy and loss curves, all eight methods, Pavia University dataset.

On the Loukia dataset, characterized by complex vegetation and geological distributions, validation loss curves distinctly highlight the structural fragility of sequential and bottleneck tensor approximations. Specifically, the TT and Tucker models exhibit sudden, significant spikes in validation loss alongside corresponding accuracy degradation. The first- and secondlevel HMD frameworks bypass this volatility entirely. They maintain tightly bounded, low variance validation trajectories that parallel the convolutional neural network baseline. By distributing the representational load across isolated multivariance terms rather than imposing an artificial sequential order or a singular bottleneck, the proposed approach successfully captures the underlying signatures while avoiding optimization instability.

![](images/6b91c057df77b882d93751c78bb02b06a8029438cf6d5ac50e637cc92e202ef6.jpg)  
Figure 8: Training/validation accuracy and loss curves, all eight methods, Loukia dataset.

Ultimately, these empirical optimization results validate the theoretical superiority of the HMD expansion. While traditional tensor layers force all optimization pressure onto a rigid core tensor resulting in unstable decision boundaries and erratic validation behavior the higher level HMD layers systematically distribute this load. Consequently, HMD-1 and HMD-2 achieve a highly stable optimization landscape that rivals the generalization capacity of standard convolutional networks, while utilizing only a fraction of the computational parameters.

The relative performance rankings of the evaluated methods remained largely consistent across the three datasets, with HMD-1 and HMD-2 consistently securing the highest tiers of classification accuracy. However, the performance margins over the established baselines varied significantly depending on specific scene characteristics. The disparity between the proposed higher level HMD frameworks and the classical tensor baselines (Tucker, CP, HMD-0) was most pronounced on the Pavia University and Loukia datasets, yielding improvements of up to 20.4 and 21.6 OA percentage points between HMD-2 and HMD-0, respectively. Conversely, this gap narrowed on the Indian Pines scene, where even the least efective tensor model which is CP method with OA = 74.93%, remained competitive with the strongest non-HMD baseline.

Among all evaluated architectures, the relative standing of the 2D-CNN baseline exhibited the highest degree of scene dependency. While it emerged as the highest performing method on Pavia University, it was outperformed by HMD-1 and HMD-2 by 3.6 and 6.7 OA points, respectively, on the Loukia scene. Because Loukia is characterized by the fewest labeled samples per class among the evaluated datasets, this performance degradation indicates that the substantial parametric footprint of standard convolutional networks becomes a computational liability rather than an asset under constrained training data conditions.

## 5.4 Parameter Eficiency

Every feature extraction layer evaluated in this work is defined by an explicit set of weight tensors, allowing for a closed-form parameter count determined strictly by the patch dimensions $( n _ { 1 } , n _ { 2 } , n _ { 3 } )$ , the chosen rank or filter configuration, and the kernel size for the convolutional baselines. Deriving these mathematical forms directly from the HMD and CNN formulations clarifies how the representational cost of each method scales with the number of spectral bands, $n _ { 3 }$ . Because $n _ { 3 }$ varies significantly across the three benchmark datasets, these theoretical formulations provide clear predictions that can be directly verified against the empirical parameter counts.

For the HMD-0, HMD-1, HMD-2 and Tucker layers, the only trainable variables are the support matrices $U _ { 1 } \in \mathbb { R } ^ { n _ { 1 } \times r } , U _ { 2 } \in \mathbb { R } ^ { n _ { 2 } \times r }$ , and $U _ { 3 } \in \mathbb { R } ^ { n _ { 3 } \times r }$ introduced in Eq. 6. These layers do not require additional internal bias or normalization parameters. As a result, all four models share an identical parameter count, that is

$$
P ^ { ( \mathrm { H M D - 0 , 1 , 2 } ) } = P ^ { ( \mathrm { T u c k e r } ) } = r ( n _ { 1 } + n _ { 2 } + n _ { 3 } )\tag{23}
$$

This equivalence is a direct arithmetic consequence of the structural properties established in Section 3.2 increasing the HMD truncation level only alters how the computed hierarchical components are pooled, without introducing new learnable weights. Similarly, the CP layer relies on three factor matrices with shapes identical to $U _ { 1 } , U _ { 2 }$ , and $U _ { 3 }$ . Thus, for equivalent rank configurations, the CP parameter count is computed as follows

$$
P ^ { \mathrm { ( C P ) } } = R ( n _ { 1 } + n _ { 2 } + n _ { 3 } )\tag{24}
$$

which is numerically indistinguishable from P<sup>(HMD-0,1,2)</sup>. Therefore, any performance disparities between the CP model and the HMD family reflect diferences in representational architecture rather than parameter capacity.

In contrast, the TT layer scales quadratically with respect to its rank. Its intermediate core tensor, $G _ { 2 } \in \mathbb { R } ^ { R _ { 1 } \times n _ { 2 } \times R _ { 2 } }$ , simultaneously carries two bond dimensions. Assuming $R _ { 1 } =$ $R _ { 2 } = R _ { 3 } = r$ and spatial dimensions $n _ { 1 } = n _ { 2 } = 1 1$ , the TT parameter count becomes

$$
P ^ { \mathrm { ( T T ) } } = 1 1 r + r ^ { 2 } ( 1 1 + n _ { 3 } )\tag{25}
$$

This quadratic scaling causes the TT layer to grow significantly faster with r than the HMD, Tucker and CP frameworks.

The convolutional baselines require a diferent parameter accounting structure. Each convolutional block introduces kernel weights, a bias vector, and batch normalization parameters. Specifically, for a Conv2D or Conv3D block featuring $C _ { \mathrm { i n } }$ input channels and f output filters, the kernel accounts for $k ^ { d } C _ { \mathrm { i n } } f$ weights, where $d \in \{ 2 , 3 \}$ denotes the kernel dimensionality.

The bias vector adds additional f parameters, and the batch normalization layer contributes 4f parameters comprising two trainable parameters, $\gamma$ and $\beta ,$ alongside two non-trainable parameters for the running mean and variance. All these components are included in the overall parameter counts, resulting in $k ^ { d } C _ { \mathrm { i n } } f + 5 f$ parameters per block. In the 2D-CNN baseline, the spectral dimension defines the input channel count of the initial convolutional block. Consequently, its total parameter count is computed as follows

$$
P ^ { ( \mathrm { 2 D - C N N } ) } = k ^ { 2 } f _ { 1 } ( n _ { 3 } + f _ { 2 } ) + 5 ( f _ { 1 } + f _ { 2 } )\tag{26}
$$

where $f _ { 1 }$ and $f _ { 2 }$ denote the filter counts for the first and second convolutional layers, respectively. Evidently, P<sup>(2D-CNN)</sup> scales linearly with $n _ { 3 }$ , mirroring the scaling behavior of the tensor-based layers.

Conversely, the 3D-CNN operates on a patch reshaped into a single channel volume prior to convolution. Here, the spectral axis is absorbed directly into the kernel’s volumetric receptive field rather than serving as the channel count. As a result, its parameter equation becomes

$$
P ^ { ( \mathrm { 3 D - C N N } ) } = k ^ { 3 } f _ { 1 } ( 1 + f _ { 2 } ) + 5 ( f _ { 1 } + f _ { 2 } )\tag{27}
$$

that is entirely independent of $n _ { 3 }$ . This invariance to the spectral band count is a unique structural characteristic not shared by any of the other evaluated models.

Table 6 provides these closed form mathematical equations for the three benchmark datasets. The calculations utilize the standardized rank and filter configurations held constant throughout the experiments: $r = 5$ for the HMD family, Tucker, and CP models; $R _ { 1 } = R _ { 2 } = R _ { 3 } = 5$ for TT; k = 3, f = 8, and $f _ { 2 } = 1 6$ for the 2D-CNN; and $k = 3 , f _ { 1 } = 4$ and $f _ { 2 } = 8$ for the 3D-CNN. Because the Indian Pines, Pavia University, and Loukia datasets span a broad spectrum of spectral bands containing $n _ { 3 } = 2 0 0 , 1 0 3$ , and 176, respectively, the parameter counts for all $n _ { 3 }$ dependent methods adjust proportionately. Loukia correctly occupies an intermediate parametric position between Pavia University and Indian Pines for these methods. Meanwhile, the structural design of the 3D-CNN ensures its parameter count remains rigidly fixed at 1032 across all datasets, regardless of spectral resolution.

Table 6: Closed-form feature-extractor parameter counts by method and dataset, at the rank/filter configuration used throughout this study (r = 5; TT: $R _ { 1 } = R _ { 2 } = R _ { 3 } = 5 ;$ 2D-CNN: $k = 3 , f _ { 1 } = 8 , f _ { 2 } = 1 6 ; 3 \mathrm { D } \mathrm { - C N N } \colon k = 3 , f _ { 1 } = 4 , f _ { 2 } = 8 )$ . Indian Pines: $n _ { 3 } = 2 0 0 ;$ Pavia University: $n _ { 3 } = 1 0 3 ;$ Loukia: $n _ { 3 } = 1 7 6 .$
<table><tr><td>Method</td><td>Formula</td><td>Indian Pines</td><td>Pavia University</td><td>Loukia</td></tr><tr><td> $\mathrm { H M D - 0 } ^ { \ast }$ </td><td> $r ( n _ { 1 } + n _ { 2 } + n _ { 3 } )$ </td><td>1,110</td><td>625</td><td>990</td></tr><tr><td> ${ \mathrm { H M D - 1 } } ^ { * }$ </td><td> $r ( n _ { 1 } + n _ { 2 } + n _ { 3 } )$ </td><td>1,110</td><td>625</td><td>990</td></tr><tr><td> ${ \mathrm { H M D - 2 } } ^ { * }$ </td><td> $r ( n _ { 1 } + n _ { 2 } + n _ { 3 } )$ </td><td>1,110</td><td>625</td><td>990</td></tr><tr><td> ${ \mathrm { T u c k e r } } ^ { * }$ </td><td> $r ( n _ { 1 } + n _ { 2 } + n _ { 3 } )$ </td><td>1,110</td><td>625</td><td>990</td></tr><tr><td> $\mathrm { C P ^ { * } }$ </td><td> $R ( n _ { 1 } + n _ { 2 } + n _ { 3 } )$ </td><td>1,110</td><td>625</td><td>990</td></tr><tr><td> $\mathrm { T T }$ </td><td> $n _ { 1 } R _ { 1 } + R _ { 1 } n _ { 2 } R _ { 2 } + R _ { 2 } n _ { 3 } R _ { 3 }$ </td><td>5,330</td><td>2,905</td><td>4,730</td></tr><tr><td> $2 \mathrm { D } { \cdot } \mathrm { C N N }$ </td><td> $k ^ { 2 } f _ { 1 } ( n _ { 3 } + f _ { 2 } ) + 5 ( f _ { 1 } + f _ { 2 } )$ </td><td>15,672</td><td>8,688</td><td>13,944</td></tr><tr><td>3D-CNN</td><td> $k ^ { 3 } f _ { 1 } ( 1 + f _ { 2 } ) + 5 ( f _ { 1 } + f _ { 2 } )$ </td><td>1,032</td><td>1,032</td><td>1,032</td></tr></table>

## 6 Conclusion

This study successfully implements the Holistic Multivariance Decomposition framework as a series of end-to-end trainable, convolution-free feature extraction layers for HS image classification. By systematically isolating linear directional components and pairwise cross-domain interactions, the proposed HMD-1 and HMD-2 layers address the structural bottlenecks of classical tensor decompositions such as the Tucker, Canonical Polyadic, and Tensor Train models which rely on rigid, low-rank core projections and sufer from optimization volatility and subspace collapse. Empirical evaluations across diverse agricultural, urban, and mixed land-cover scenes demonstrate that HMD-2 and HMD-1 consistently yield state-of-the-art classification performance, achieving mean Overall Accuracies of 97.85% and 96.40%, respectively. Crucially, this discriminative advantage is realized with exceptional parameter eficiency. The proposed methods match the minimal footprint of baseline core tensor models while requiring substantially fewer trainable weights than conventional CNN architectures.

Collectively, these findings demonstrate that the discriminative advantage of the HMD family is not restricted to a specific sensor modality or class distribution. Rather, its structural superiority is most evident in challenging regimes characterized by limited labeled samples and high class counts representing a scenario of paramount practical relevance for operational hyperspectral mapping.

The framework exhibits remarkable structural robustness, insulating the latent feature space from optimization noise even under highly constrained rank configurations or limited training regimes. Ultimately, the Holistic Multivariance Decomposition provides a computationally lightweight, stable, and interpretable deep learning layer, presenting a highly eficient alternative to standard convolutional networks for advanced hyperspectral classification.

## References

Muhammad Ahmad, Adil Mehmood Khan, Manuel Mazzara, Salvatore Distefano, Mohsin Ali, and Muhammad Shahzad Sarfraz. A fast and compact 3-d cnn for hyperspectral image classification. IEEE Geoscience and Remote Sensing Letters, 19:1–5, 2020.

Anuja Bhargava, Ashish Sachdeva, Kulbhushan Sharma, Mohammed H Alsharif, Peerapong Uthansakul, and Monthippa Uthansakul. Hyperspectral imaging and its applications: A review. Heliyon, 10(12), 2024.

Aloke Datta, Susmita Ghosh, and Ashish Ghosh. Pca, kernel pca and dimensionality reduction in hyperspectral images. In Advances in Principal Component Analysis: Research and Development, pages 19–46. Springer, 2017.

Samson Damilola Fabiyi, Paul Murray, Jaime Zabalza, and Jinchang Ren. Folded lda: extending the linear discriminant analysis algorithm for feature extraction and data reduction in hyperspectral remote sensing. IEEE Journal of selected topics in applied earth observations and remote sensing, 14:12312–12331, 2021.

Daniel Guidici and Matthew L Clark. One-dimensional convolutional neural network landcover classification of multi-seasonal hyperspectral imagery in the san francisco bay area, california. Remote Sensing, 9(6):629, 2017.

Marian-Daniel Iordache, José M Bioucas-Dias, and Antonio Plaza. Sparse unmixing of hyperspectral data. IEEE Transactions on Geoscience and Remote Sensing, 49(6):2014– 2039, 2011.

Xiaoxuan Ji, Pengxian Li, Jialin Wang, Shuang Xu, Teng-Yu Ji, Jiangjun Peng, Xiangyong Cao, and Deyu Meng. Unified guided hyperspectral image denoising by continuous coupled tucker decomposition. IEEE Transactions on Geoscience and Remote Sensing, 64:5519813– 5519813, 2026. doi: 10.1109/TGRS.2026.3708294.

Mohamad Jouni, Mauro Dalla Mura, and Pierre Comon. Hyperspectral image classification using tensor cp decomposition. In IGARSS 2019 - 2019 IEEE International Geoscience and Remote Sensing Symposium, pages 1164–1167, 2019. doi: 10.1109/IGARSS.2019.8898346.

Efe Kahraman and Süha Tuna. Enhancing hyperspectral and multispectral image fusion using high dimensional model representation. In 2025 9th International Symposium on Innovative Approaches in Smart Technologies (ISAS), pages 1–7. IEEE, 2025.

Shahid Karim, Akeel Qadir, Umar Farooq, Muhammad Shakir, and Asif A Laghari. Hyperspectral imaging: a review and trends towards medical imaging. Current Medical Imaging Reviews, 19(5):417–427, 2023.

Atiya Khan, Amol D Vibhute, Shankar Mali, and Chandrashekhar H Patil. A systematic review on hyperspectral imaging technology with a machine and deep learning methodology for agricultural applications. Ecological Informatics, 69:101678, 2022.

Tamara G Kolda and Brett W Bader. Tensor decompositions and applications. SIAM review, 51(3):455–500, 2009.

Brajesh Kumar, Onkar Dikshit, Ashwani Gupta, and Manoj Kumar Singh. Feature extraction for hyperspectral image classification: A review. International Journal of Remote Sensing, 41(16):6248–6287, 2020.

Na Li, Huijie Zhao, and Guorui Jia. Dimensional reduction method based on factor analysis model for hyperspectral data. Journal of Image and Graphics, 16(11):2030–2035, 2011.

Wei Li and Qian Du. A survey on representation-based classification and detection in hyperspectral remote sensing imagery. Pattern Recognition Letters, 83:115–123, 2016.

Siti Salwa Md Noor, Kaleena Michael, Stephen Marshall, and Jinchang Ren. Hyperspectral image enhancement and mixture deep-learning classification of corneal epithelium injuries. Sensors, 17(11):2644, 2017.

Tomohiro Mitsui, Akino Mori, Toshihiro Takamatsu, Tomohiro Kadota, Konosuke Sato, Ryodai Fukushima, Kyohei Okubo, Masakazu Umezawa, Hiroshi Takemura, Hideo Yokota, et al. Evaluating the identification of the extent of gastric cancer by over-1000 nm nearinfrared hyperspectral imaging using surgical specimens. Journal of Biomedical Optics, 28 (8):086001–086001, 2023.

Adam Polak, Timothy Kelman, Paul Murray, Stephen Marshall, David JM Stothard, Nicholas Eastaugh, and Francis Eastaugh. Hyperspectral imaging combined with data classification techniques as an aid for artwork authentication. Journal of Cultural Heritage, 26:1–11, 2017.

Manoharan Prabukumar, Shrutika Sawant, Sathishkumar Samiappan, and Loganathan Agilandeeswari. Three-dimensional discrete cosine transform-based feature extraction for hyperspectral image classification. Journal of Applied Remote Sensing, 12(4):046010–046010, 2018.

Shen-En Qian. Overview of hyperspectral imaging remote sensing from satellites. Advances in hyperspectral image processing techniques, pages 41–66, 2022.

Muhammed Enis Şen and Süha Tuna. A new feature extraction scheme based on support optimization in enhanced multivariance products representation for hyperspectral imagery. Journal of the Franklin Institute, 362(2):107464, 2025.

Nicholas D Sidiropoulos, Lieven De Lathauwer, Xiao Fu, Kejun Huang, Evangelos E Papalexakis, and Christos Faloutsos. Tensor decomposition for signal processing and machine learning. IEEE Transactions on signal processing, 65(13):3551–3582, 2017.

Ganji Tejasree and Loganathan Agilandeeswari. An extensive review of hyperspectral image classification and prediction: techniques and challenges. Multimedia Tools and Applications, 83(34):80941–81038, 2024.

Süha Tuna. Improving sparse coding based hyperspectral image classification via tensor decomposition and oversegmentation. In AIP Conference Proceedings, volume 3489, page 280024. AIP Publishing LLC, 2026.

Süha Tuna, Evrim Korkmaz Özay, Burcu Tunga, Ercan Gürvit, and M Alper Tunga. An eficient feature extraction approach for hyperspectral images using wavelet high dimensional model representation. International Journal of Remote Sensing, 43(19-24):6899–6920, 2022.

Süha Tuna, Behçet Uğur Töreyin, Metin Demiralp, Jinchang Ren, Huimin Zhao, and Stephen Marshall. Iterative enhanced multivariance products representation for efective compression of hyperspectral images. IEEE Transactions on Geoscience and Remote Sensing, 59(11): 9569–9584, 2021. doi: 10.1109/TGRS.2020.3031016.

Minghua Wang, Danfeng Hong, Zhu Han, Jiaxin Li, Jing Yao, Lianru Gao, Bing Zhang, and Jocelyn Chanussot. Tensor decompositions for hyperspectral data processing in remote sensing: A comprehensive review. IEEE Geoscience and Remote Sensing Magazine, 11(1): 26–72, 2023. doi: 10.1109/MGRS.2022.3227063.

Qi Wang, Zhenghang Yuan, Qian Du, and Xuelong Li. Getnet: A general end-to-end 2-d cnn framework for hyperspectral image change detection. IEEE Transactions on Geoscience and Remote Sensing, 57(1):3–13, 2018.

Sławomir Wilczyński, Robert Koprowski, Mathieu Marmion, Piotr Duda, and Barbara Błońska-Fajfrowska. The use of hyperspectral imaging in the vnir (400–1000 nm) and swir range (1000–2500 nm) for detecting counterfeit drugs with identical api composition. Talanta, 160:1–8, 2016.

Di Wu and Da-Wen Sun. Advanced applications of hyperspectral imaging technology for food quality and safety analysis and assessment: A review—part i: Fundamentals. Innovative Food Science & Emerging Technologies, 19:1–14, 2013.

Fulin Xu, Ge Zhang, Chao Song, Hui Wang, and Shaohui Mei. Multiscale and cross-level attention learning for hyperspectral image classification. IEEE Transactions on Geoscience and Remote Sensing, 61:1–15, 2023.

Tian-Heng Zhang, Jian-Li Zhao, Sheng Fang, Zhe Li, and Mao-Guo Gong. Full-modeaugmentation tensor-train rank minimization for hyperspectral image inpainting. IEEE Transactions on Geoscience and Remote Sensing, 62:1–13, 2023.