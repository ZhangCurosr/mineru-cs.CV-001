# Denoised Variance-Based Pruning with Optimal Brain Bias Compensation

Geon Tack Lee<sup>1,2</sup> Jaegul Choo<sup>1,†</sup> Kang Eun Jeon<sup>1,†</sup>

<sup>1</sup>DAVIAN Lab, KAIST AI <sup>2</sup>New York University Abu Dhabi

Email: gtl256@nyu.edu, jchoo@kaist.ac.kr, kejeon@kaist.ac.kr

<sup>†</sup>Corresponding authors.

August 19, 2026

## Abstract

Vision Transformers (ViTs) achieve state-of-the-art performance but carry massive computational overhead that restricts edge deployment. Although structural pruning has emerged as a key strategy to reduce these costs, existing methods often sufer from severe accuracy degradation or require expensive retraining. Recently, Variance-Based Pruning (VBP) introduced a promising paradigm by selecting neurons based on activation variance; however, it remains limited by statistical noise in finite-sample activation covariance and reliance on bias-only updates that cannot fully account for structural reconstruction error. To address these limitations, we introduce Denoised Variance-Based Pruning with Optimal Brain Bias Compensation $\mathrm { ( D V B P + O B ^ { 2 } C ) }$ . We leverage random matrix theory to filter noise from the activation covariance spectrum for robust neuron selection and mathematically prove that integrating mean-shift compensation into the Optimal Brain Compression objective reduces the layer-wise Hessian exactly to the activation covariance matrix. This enables an optimal, closed-form update of the remaining weights using the same statistics gathered for selection. Extensive experiments on DeiT, Swin, and ConvNeXt architectures demonstrate that $\mathrm { D V B P } + \mathrm { O B } ^ { 2 } \mathrm { C }$ achieves state-of-the-art training-free performance; at 50% MLP pruning, it retains over 90% of the original Top-1 accuracy on Small and Base variants, outperforming VBP by up to 29.46% (ConvNeXt-T) and 7.33% (Swin-S). The code is available at: https://github.com/geontackee/DVBP\_OB2C.

Keywords: Vision Transformers, Structured Pruning, Variance-Based Pruning

## 1. Introduction

Since the introduction of the Vision Transformer (ViT) [6], transformer-based architectures have consistently defined the state-of-the-art in computer vision, frequently outperforming traditional Convolutional Neural Networks (CNNs) [16]. However, this empirical success comes at a steep computational price. Standard ViTs demand substantial memory and computational resources for inference [32], efectively precluding their deployment on resource-constrained edge devices. For instance, ViT-B/16 requires 86 million parameters [6], contrasting sharply with the 7.5 million of MobileNetV3 [18]. As vision models are increasingly deployed on-device for latency-critical applications, such as autonomous driving, robotics, and unmanned aerial systems, achieving smaller model footprints and higher inference throughput has become an indispensable research objective.

While pruning has emerged as a principled strategy to bridge this eficiency gap [14, 17], current paradigms present a fundamental trade-of between theoretical sparsity and practical acceleration. Unstructured methods [14, 21] achieve high compression ratios by zeroing individual weights. However, these methods produce irregular sparsity patterns that demand specialized hardware accelerators [7] or custom CUDA kernels to induce practical latency reductions [12]. This severely limits their scalability and real-world adoption. On the other hand, structured pruning [22, 35] physically removes entire neurons or filters, hence directly reducing model size and inference time on commodity hardware. However, existing structured methods predominantly follow the magnitude-based pruning paradigm [14, 22], which ranks parameters by the absolute magnitude of their weights under the assumption that smaller weights contribute less to the network. While this heuristic is efective for zeroing individual connections, removing entire neurons or filters based on aggregate magnitude discards complex feature interactions and leads to severely degraded accuracy [25].

In a fundamentally diferent direction, Variance-Based Pruning (VBP) [2] has demonstrated that this trade-of need not be inevitable. Rather than relying on weight magnitudes, VBP introduces an activation variance criterion that identifies neurons whose activations exhibit the lowest variance across a calibration set. The key insight is that low-variance neurons behave approximately as constants, meaning their contribution to the downstream layer can be efectively absorbed into the bias term through a simple mean-shift compensation. This allows VBP to structurally remove neurons while immediately recovering a portion of the lost performance, substantially outperforming magnitude-based approaches. However, VBP still sufers from significant accuracy degradation at higher pruning ratios, leaving considerable room for improvement.

In this work, we identify two critical limitations of VBP. First, the activation covariance matrix, from which variance scores are derived, is corrupted by statistical noise. This noise obscures the true importance ranking of neurons, leading to increasingly suboptimal pruning decisions at aggressive compression ratios (e.g., ≥50%). Second, VBP compensates for pruned neurons solely through mean-shift compensation, leaving the remaining weight matrices entirely unmodified. We observe that mean-shift compensation cannot fully account for the reconstruction error introduced by removing entire neurons, and a principled compensation through weight update is necessary to close this gap.

To address these limitations, we propose Denoised Variance-Based Pruning with Optimal Brain Bias Compensation $\mathrm { ( D V B P + O B ^ { 2 } C ) }$ . Our core contributions are:

• Denoised Variance Metric: We leverage random matrix theory, specifically the Marchenko-Pastur distribution, to separate signal eigenvalues from noise in the activation covariance spectrum. The resulting denoised variance scores yield a substantially more reliable neuron importance ranking, particularly at high pruning ratios.

$\mathbf { O B } ^ { 2 } \mathbf { C }$ Framework: We incorporate mean-shift compensation directly into the seminal Optimal Brain Compression (OBC) [10] reconstruction objective. This integration yields a remarkably elegant result: the layer-wise Hessian reduces exactly to the activation covariance matrix. Consequently, the same covariance matrix simultaneously serves as the basis for both neuron selection (via denoised variance) and optimal multi-weight recovery, unifying the two stages of our pipeline into a single, coherent statistical framework.

Extensive experiments on DeiT [32], Swin [23], and ConvNeXt [24] architectures across Tiny, Small, and Base scales demonstrate that our method consistently outperforms VBP by a widening margin as the pruning ratio increases. At a 50% structural reduction, $\mathrm { D V B P } + \mathrm { O } \mathbf { \bar { B } } ^ { 2 } \mathrm { C }$ retains approximately 80% Top-1 accuracy on Tiny-scale models and exceeds 90% on Small and Base variants, surpassing VBP by up to 7.33% on Swin-S and 29.46% on ConvNeXt-T.

## 2. Background and Related Works

## 2.1. Model Compression and Pruning

Model compression relies primarily on quantization [1, 11] and pruning [17, 20]. Unlike quantization, which reduces bit-width but often demands specialized low-bit hardware, pruning directly eliminates operations, natively accelerating inference. Pruning roots trace to second-order methods like OBD [19] and OBS [15]. Later approaches introduced magnitude thresholding [14], gradient-based sensitivities [21], and identified sparse, trainable subnetworks [8]. However, these unstructured methods create irregular sparsity requiring custom accelerators [12]. Structured pruning instead removes entire neurons or channels [22], yielding immediate speedups on commodity hardware at the cost of severe accuracy drops, typically requiring extensive iterative retraining [34].

## 2.2. Variance-Based Pruning

Variance-Based Pruning (VBP) [2] selects neurons for removal based on activation variance rather than weight magnitude. Given an input activation vector x with mean $\mu ,$ VBP introduces a mean-shift vector to compensate for pruned neurons. The mean-shift vector $\Delta \mu ^ { \prime } s$ �-th entry is $\pmb { \mu } _ { j }$ if neuron � is pruned and zero otherwise. The output of the pruned layer is then expressed as:

$$
\mathbf { y } = \hat { \mathbf { W } } \hat { \mathbf { x } } + \mathbf { b } ^ { \prime }\tag{1}
$$

where $\hat { \mathbf { W } }$ is the pruned weight of the subsequent layer, $\hat { \mathbf { x } }$ is the pruned output, and $\mathbf { b } ^ { \prime }$ is the mean-shift compensation term that absorbs the efect of the pruned neurons into the bias:

$$
\mathbf { b } ^ { \prime } = \mathbf { b } + \mathbf { W } \Delta \mu .\tag{2}
$$

After this compensation, the residual reconstruction error for neuron � is governed by its activation variance, $\sigma _ { i } ^ { 2 }$ :

$$
\mathbb { E } [ ( x _ { i } - \mathbb { E } [ x _ { i } ] ) ^ { 2 } ] = \mathbb { E } [ ( x _ { i } - \mu _ { i } ) ^ { 2 } ] = \sigma _ { i } ^ { 2 }\tag{3}
$$

Therefore, pruning neurons with the lowest $\sigma _ { i } ^ { 2 }$ minimizes the expected reconstruction error under bias compensation.

## 2.3. Optimal Brain Compression

The Optimal Brain Compression (OBC) framework [10] has served as the foundation for numerous model compression algorithms, notably extending to Large Language Models through methods such as GPTQ [11] and SparseGPT [9]. Building upon the Optimal Brain Surgeon (OBS) [15] framework, OBC minimizes the squared reconstruction error between the original and compressed layer outputs:

$$
\underset { \hat { \mathbf { W } } } { \mathrm { a r g m i n } } \ : | | \mathbf { W } \mathbf { X } - \hat { \mathbf { W } } \mathbf { X } | | _ { F } ^ { 2 } ,\tag{4}
$$

where $\hat { \mathbf { W } }$ is the compressed weight matrix and $\mathbf { X } \in \mathbb { R } ^ { d \times N }$ is the input activation matrix whose columns are activation vectors collected from � calibration samples. To minimize this objective, the OBC framework derives a block-wise closed-form update rule that adjusts the remaining weights as

$$
\delta _ { P } = - \mathbf { H } _ { : , P } ^ { - 1 } ( ( \mathbf { H } ^ { - 1 } ) _ { P } ) ^ { - 1 } \mathbf { w } _ { P }\tag{5}
$$

where $\delta _ { P }$ is the weight update for block $P , \mathbf { H } = 2 \mathbf { X } \mathbf { X } ^ { T }$ is the layer-wise proxy-Hessian, $\mathbf { w } _ { P }$ is the current weight vector of the row being updated, and � denotes the set of indices corresponding to the block being compressed. Recent work OBS-Dif [36] adapts second-order statistics for structured pruning of difusion models. However, OBS-Dif predominantly utilizes complex, Hessian-based OBS calculations as the primary criterion for selecting which structures to remove. Our method decouples this pipeline: we employ an eficient denoised variance metric to identify redundant neurons, reserving the OBC framework strictly for optimal weight recovery after pruning.

## 3. Methodology

In this section, we present the $\mathrm { D V B P } + \mathrm { O B } ^ { 2 } \mathrm { C }$ structured pruning pipeline. We begin by augmenting the OBC [10] reconstruction objective with mean-shift compensation (§3.1), showing that the resulting layer-wise Hessian reduces to the input activation covariance matrix. We then describe an online method for computing this covariance without prohibitive memory costs (§3.2). To address the inherent noise in finite-sample covariance estimates, we apply spectral denoising via the Marchenko-Pastur distribution (§3.3), yielding robust variance scores for neuron selection (§3.4). Finally, we detail the end-to-end pruning and weight update mechanism (§3.5).

## 3.1. Optimal Brain Bias Compensation $\scriptstyle ( \mathtt { O B } ^ { 2 } \mathbf { C } )$

To mitigate the mean-shift error introduced during weight pruning, we introduce Optimal Brain Bias Compensation $\mathrm { ( O B ^ { 2 } C ) }$ , which explicitly incorporates a bias term into the layer-wise reconstruction objective. Given an input activation matrix $\mathbf { X } \in \mathbb { R } ^ { d \times N }$ , original weights W, pruned weights $\hat { \mathbf { W } } _ { ; }$ , and a bias vector b, the augmented objective is formulated as:

$$
\arg \operatorname* { m i n } _ { \hat { \mathbf { W } } , \mathbf { b } } \left\| \mathbf { W X } - ( \hat { \mathbf { W } } \mathbf { X } + \mathbf { b } \mathbf { 1 } ^ { T } ) \right\| _ { F } ^ { 2 }\tag{6}
$$

where 1 is an �-dimensional vector of ones. Setting the partial derivative of the objective with respect to b to zero yields the optimal bias compensation:

$$
\mathbf { b } ^ { * } = ( \mathbf { W } - \hat { \mathbf { W } } ) \pmb { \mu }\tag{7}
$$

where $\textstyle \mu = { \frac { 1 } { N } } \mathbf { X } \mathbf { 1 }$ denotes the mean vector of the input activations. Substituting $\mathbf { b } ^ { * }$ back into the objective function yields a simplified minimization problem:

$$
\arg \operatorname* { m i n } _ { \Delta \mathbf { W } } \| \Delta \mathbf { W } \widetilde { \mathbf { X } } \| _ { F } ^ { 2 }\tag{8}
$$

where $\Delta \mathbf { W } = \mathbf { W } - \hat { \mathbf { W } }$ represents the weight perturbation, and $\widetilde { \mathbf X } = \mathbf X - \pmb { \mu } \mathbf 1 ^ { T }$ represents the mean-centered activations. For detailed mathematical derivation, see Appendix A.2.

Evaluating the error induced by this weight perturbation, we find that the layer-wise proxy-Hessian matrix with respect to the rows of ΔW is given by:

$$
\mathbf { H } = 2 \widetilde { \mathbf { X } } \widetilde { \mathbf { X } } ^ { \top } = 2 N \mathbf { C }\tag{9}
$$

where � is the calibration set size and C is the activation covariance matrix. Consequently, the layer-wise Hessian is directly proportional to the covariance of the centered activations, rather than the uncentered second moment used in standard formulations. Integrating this into the OBC multi-weight update, we replace the standard Hessian with 2�C. Let � denote the set of indices corresponding to the pruned neurons. The update formula for the weights is rewritten as:

$$
\delta _ { \mathbf { W } } = - \mathbf { W } _ { : , P } \left( [ \mathbf { C } ^ { - 1 } ] _ { P , P } \right) ^ { - 1 } [ \mathbf { C } ^ { - 1 } ] _ { P , : }\tag{10}
$$

This yields an update expression that is structurally identical to the standard OBC rule, but inherently minimizes the mean-shift error by operating on the centered covariance matrix C.

## 3.2. Covariance Matrix Computation

To apply ${ \mathrm { O B } } ^ { 2 } { \mathrm { C } } ,$ we first require the exact centered covariance matrix of the activations for each layer. Storing full-precision activations for a naive computation demands an $O ( N )$ memory footprint, rendering the approach computationally infeasible on large calibration sets. To ensure our approach remains computationally tractable, we track each layer’s covariance matrix in an online, batched manner during calibration. To track the activation statistics of neurons, the VBP [2] paper employs Welford’s algorithm to iteratively update their mean and variance. In a similar vein, Chan et al. [3] introduced a robust formula for updating the mean and second central moment across partitioned dataset A partitioned into B and C where $\mathbf { A } = \mathbf { B } \cup \mathbf { C }$ , using the expressions from Pébay [28]:

$$
\mu _ { A } = \mu _ { \mathrm { B } } + n _ { \mathrm { C } } { \frac { \delta _ { \mathrm { C , B } } } { n } } , \qquad \mathbf { M } _ { 2 , \mathrm { A } } = \mathbf { M } _ { 2 , \mathrm { B } } + \mathbf { M } _ { 2 , \mathrm { C } } + n _ { \mathrm { B } } n _ { \mathrm { C } } { \frac { \delta _ { \mathrm { C , B } } ^ { 2 } } { n _ { A } } }\tag{11}
$$

where $\mathbf { M } _ { 2 }$ denotes the second central moment matrix, $, n _ { \mathbf { A } }$ , �<sub>B</sub>, �<sub>C</sub> are the respective cardinalities, and $\delta _ { \mathbf { C } , \mathbf { B } } =$ $\mu _ { \mathrm { C } } - \mu _ { \mathrm { B } }$ is the diference between the partition means.

Let $\mathcal { P }$ denote the set of previously processed activations containing �<sub>P</sub> samples, and let N denote an incoming new batch containing $n _ { N }$ samples. The combined set is given by $C = \mathcal { P } \cup N$ , with a total of $n _ { C } = n _ { \mathcal P } + n _ { N }$ samples. When a new batch arrives, we first update the running mean of the activations using Eq. 11.

To update the sample covariance matrix, we define the weighting factor $\begin{array} { r } { \alpha = \frac { n _ { \mathcal { P } } } { n _ { C } } } \end{array}$ , which implies $\begin{array} { r } { 1 - \alpha = \frac { n _ { N } } { n _ { C } } } \end{array}$ By decomposing the second moment across the disjoint sets $\mathcal { P }$ and $N ,$ we derive the following online update formula for the covariance matrix:

$$
\mathbf { C } _ { C } = \alpha \mathbf { C } _ { \mathcal { P } } + ( 1 - \alpha ) \mathbf { C } _ { N } + \alpha ( 1 - \alpha ) \pmb { \delta \delta } ^ { T }\tag{12}
$$

where ${ \pmb { \delta } } = { \pmb { \mu } } _ { N } - { \pmb { \mu } } _ { \mathcal { P } }$ represents the diference between the mean vectors of the new batch and the previously accumulated data. Using this recursive formulation, we can accurately construct the sample covariance matrix of each layer for an arbitrarily large calibration set without requiring the full activation history to be held in memory.

## 3.3. Denoising the Covariance Matrix with Marchenko-Pastur

## 3.3.1. Marchenko-Pastur Distribution

We observe that the spectral density of the activation covariance matrix C exhibits a bulk profile characteristic of the MP [26] distribution, strongly suggesting that the underlying signal subspace is corrupted by noise. To mitigate this, we denoise C by conducting eigendecomposition on C and by fitting its eigenvalue spectrum to the MP distribution. The MP distribution describes the asymptotic eigenvalue behavior of large random matrices. Specifically, for a large random matrix X of size � × � whose entries are independent and identically distributed (i.i.d.) with mean 0 and a variance of $\sigma ^ { 2 } ,$ the probability density function of the eigenvalues of its corresponding sample covariance $\begin{array} { r } { { \bf Y } _ { n } = \frac { 1 } { n } { \bf X } ^ { T } } \end{array}$ is expressed as:

$$
f ( x ) = { \frac { \sqrt { ( \lambda _ { + } - x ) ( x - \lambda _ { - } ) } } { 2 \pi \sigma ^ { 2 } \lambda x } }\tag{13}
$$

for $\lambda _ { - } < x < \lambda _ { + }$ and 0 otherwise. Here, $\textstyle \lambda = { \frac { m } { n } }$ represents the aspect ratio of the matrix dimensions. According to the MP Law, if X satisfies the conditions, all of its non-zero eigenvalues will asymptotically fall within the theoretical bounds of $\lambda _ { + }$ and $\lambda _ { - }$ <sub>−</sub>. These bounds can be obtained by the following equation

$$
\begin{array} { r } { \lambda _ { \pm } = \sigma ^ { 2 } ( 1 \pm \sqrt { \lambda } ) ^ { 2 } . } \end{array}\tag{14}
$$

## 3.3.2. Noise Variance of Noisy Matrices

Gavish and Donoho [13] focuses on the analysis of noisy matrices under the additive model $\mathbf { Y } = \mathbf { X } + \sigma \mathbf { Z } ,$ where Y is the observed data matrix, X is the underlying low-rank signal, and Z represents random noise. A primary challenge in practical application is that the true noise scale, $\sigma ,$ is typically unknown. To address this, the authors formulate a robust estimator $\hat { \sigma } ( { \bf Y } )$ using the ratio of the median singular value of the data matrix to the theoretical median of the MP distribution:

$$
{ \hat { \boldsymbol { \sigma } } } ( \mathbf { Y } ) \ { \stackrel { \mathrm { d e f } } { = } } \ { \frac { y _ { m e d } } { \sqrt { n \mu _ { \lambda } } } } ,\tag{15}
$$

where $y _ { \mathrm { { m e d } } }$ is the median singular value of $\mathbf { Y } . \mu _ { \lambda }$ is the median of MP distribution obtainable by solving for $\lambda _ { - } < x < \lambda .$ +

$$
\int _ { \lambda _ { - } } ^ { x } \frac { \sqrt { ( \lambda _ { + } - t ) ( t - \lambda _ { - } ) } } { 2 \pi t } d t = \frac { 1 } { 2 } .\tag{16}
$$

Following our observation where eigenvalue distribution contains MP-like noise, visualized in Fig. 2. We assume the centered activation data matrix $\widetilde { \mathbf { X } }$ follows an additive noise model, $\widetilde { \mathbf { X } } = \widetilde { \mathbf { X } } _ { \mathrm { s i g n a l } } + \sigma \mathbf { Z }$ , where the entries of the noise matrix Z adhere to the Marchenko-Pastur conditions.

Following Gavish and Donoho’s work [13], we estimate the noise scale � by comparing the median eigenvalue of C to the theoretical median of the MP distribution. The median singular value of the data matrix, denoted as $s _ { \mathrm { m e d } }$ , relates to the median eigenvalue of the covariance matrix, $\Lambda _ { \mathrm { m e d } }$ , obtained through eigendecomposition as follows:

$$
s _ { \mathrm { m e d } } = \sqrt { N \Lambda _ { \mathrm { m e d } } } .\tag{17}
$$

Substituting this relationship into the estimator Eq. 15 yields:

$$
\hat { \sigma } ( \widetilde { \mathbf { X } } ) = \sqrt { \frac { \Lambda _ { \mathrm { m e d } } } { \mu _ { \lambda } } }\tag{18}
$$

where $\mu \lambda$ is the theoretical median of the MP distribution.

To obtain $\mu _ { \lambda }$ , we require matrix $\widetilde { \mathbf { X } } \widetilde { \mathbf { s } }$ aspect ratio �. In this context, we define the aspect ratio as $\textstyle \lambda = { \frac { d } { N } }$ , where � is the number of input neurons and � is the calibration set size. We use calibration set size (number of independent images) rather than the total number of patches for the denominator because patches within a single image are highly correlated, using the total patch count would violate the i.i.d. noise assumption. By defining our sample size strictly as the calibration set size, we ensure the i.i.d. assumption holds. With the aspect ratio $\lambda ,$ we compute the theoretical median eigenvalue $\mu \lambda$ via Eq. 16, which subsequently allows us to estimate the noise scale �.

Using the estimated noise variance, we calculate the theoretical MP upper bound $\lambda .$ <sub>+</sub> using Eq. 14. After computing the eigendecomposition of the covariance matrix $\mathbf { C } = \mathbf { V } \pmb { \Lambda } \mathbf { V } ^ { T }$ , where $\pmb { \Lambda } = \mathrm { d i a g } ( \lambda _ { 1 } , \dots , \lambda _ { d } )$ contains the eigenvalues and $\mathbf { V } = \left[ \mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { d } \right]$ the corresponding eigenvectors, we apply hard thresholding to retain only the signal components:

$$
\pmb { \Lambda } _ { \mathrm { s i g n a l } } = \mathrm { d i a g } ( \lambda _ { i } ) \quad \forall \lambda _ { i } > \lambda _ { + }\tag{19}
$$

$$
\mathbf { V } _ { \mathrm { s i g n a l } } = [ \mathbf { v } _ { i } ] \quad \forall \lambda _ { i } > \lambda _ { + }\tag{20}
$$

## 3.4. Denoised Variance Score

Using the retained signal components, we reconstruct the denoised covariance matrix:

$$
\mathbf { C } _ { \mathrm { d e n o i s e d } } = \mathbf { V } _ { \mathrm { s i g n a l } } \mathbf { A } _ { \mathrm { s i g n a l } } \mathbf { V } _ { \mathrm { s i g n a l } } ^ { T }\tag{21}
$$

Finally, we extract the denoised variance scores for each neuron directly from the diagonal entries of the reconstructed matrix:

$$
{ \pmb s } _ { \mathrm { d v a r } } = \mathrm { d i a g } ( { \bf C } _ { \mathrm { d e n o i s e d } } )\tag{22}
$$

Empirically, we observe that the denoised variance scores $\pmb { s } _ { \mathrm { d v a r } }$ excel at high pruning ratios, whereas the raw variance scores $\mathbf { s } _ { \mathrm { v a r } } = \mathrm { d i a g } ( \mathbf { C } )$ , extracted from the original covariance matrix, remain highly efective at low pruning ratios. Thus, we store the raw scores prior to eigendecomposition and formulate a hybrid scoring metric, controlled by the target pruning ratio $p \colon$

$$
{ \pmb s } _ { \mathrm { h y b r i d } } = ( 1 - p ) { \pmb s } _ { \mathrm { v a r } } + p { \pmb s } _ { \mathrm { d v a r } }\tag{23}
$$

## 3.5. Pruning and Update Mechanism

Following the structural framework of Variance-Based Pruning (VBP), we target MLP blocks in Vision Transformers (ViTs) [6] and ConvNeXt [24] models. While we adopt this standard layer selection protocol, our approach introduces a novel neuron selection criterion and weight update mechanism.

Specifically, we globally rank the hidden dimensions based on our proposed hybrid score, and select the lowest-ranking �% of neurons across the entire network to consist the pruned indices set �. Rather than enforcing a uniform per-layer pruning ratio, the network dynamically allocates capacity, preserving critical layers while heavily penalizing redundant ones.

Table 1. Comparison of Top-1 accuracy (%) on ImageNet-1K across diferent pruning methods on DeiT and Swin. All models are evaluated immediately after compression without any post-pruning fine-tuning.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Prune Ratio (%)</td><td colspan="3">Top-1 Accuracy (%)</td></tr><tr><td>Full Mag.</td><td>SNIP VBP</td><td>Ours</td></tr><tr><td rowspan="3">DeiT-T</td><td>40</td><td></td><td>14.64 55.80 57.56</td><td>62.56</td></tr><tr><td>50</td><td>72.16 3.73</td><td>39.49 42.89</td><td>56.40</td></tr><tr><td>60</td><td>0.81</td><td>23.59 26.97</td><td>47.58</td></tr><tr><td rowspan="3">DeiT-S</td><td>40</td><td>20.68</td><td>55.37 72.35</td><td>73.71</td></tr><tr><td>50</td><td>79.86 4.24</td><td>50.65 66.03</td><td>70.25</td></tr><tr><td>60</td><td>1.17</td><td>42.17 54.73</td><td>65.25</td></tr><tr><td rowspan="3">DeiT-B</td><td>40</td><td></td><td>1.95 65.03</td><td>76.49 79.00</td></tr><tr><td>50</td><td>81.98 0.38</td><td>62.78 68.92</td><td>75.85</td></tr><tr><td>60</td><td>0.16</td><td>52.82 50.63</td><td>69.68</td></tr></table>

<table><tr><td rowspan="2">Model</td><td rowspan="2">Prune Ratio (%)</td><td colspan="3">Top-1 Accuracy (%)</td></tr><tr><td>Full Mag.</td><td>SNIP VBP</td><td>Ours</td></tr><tr><td rowspan="3">Swin-T</td><td>40</td><td>47.39</td><td>37.99 64.25</td><td>72.23</td></tr><tr><td>50</td><td>81.38 26.40</td><td>28.87 49.98</td><td>66.12</td></tr><tr><td>60</td><td>7.91</td><td>17.87 26.03</td><td>56.42</td></tr><tr><td rowspan="3">Swin-S</td><td>40</td><td>2.19</td><td>62.08 78.41</td><td>80.69</td></tr><tr><td>50</td><td>83.32 0.22</td><td>45.36 69.91</td><td>77.24</td></tr><tr><td>60</td><td>0.14</td><td>25.72 45.30</td><td>70.79</td></tr><tr><td rowspan="3">Swin-B</td><td>40</td><td></td><td>35.62 57.02</td><td>78.70 80.56</td></tr><tr><td>50</td><td>85.27</td><td>6.44 38.55 70.86</td><td>76.16</td></tr><tr><td>60</td><td></td><td>1.10 23.81</td><td>55.88 69.00</td></tr></table>

Table 2. Comparison of parameter counts (M), MACs (G), and Top-1 accuracy (%) for DeiT models at a 50% pruning ratio.
<table><tr><td rowspan="2">Model</td><td colspan="2">Parameters (M)</td><td colspan="2">MACs (G)</td><td colspan="5">Top-1 Accuracy (%)</td></tr><tr><td>Full</td><td>Pruned</td><td>Full</td><td>Pruned</td><td>Full</td><td>Mag.</td><td>SNIP</td><td>VBP</td><td>Ours</td></tr><tr><td>DeiT-T</td><td>5.72</td><td>3.94</td><td>1.26</td><td>0.91</td><td>72.16</td><td>3.73</td><td>39.49</td><td>42.89</td><td>56.40</td></tr><tr><td>DeiT-S</td><td>22.05</td><td>14.96</td><td>4.61</td><td>3.21</td><td>79.86</td><td>4.24</td><td>50.65</td><td>66.03</td><td>70.25</td></tr><tr><td>DeiT-B</td><td>86.57</td><td>58.24</td><td>17.59</td><td>12.01</td><td>81.98</td><td>0.38</td><td>62.78</td><td>68.92</td><td>75.85</td></tr></table>

To mitigate immediate performance degradation from neuron removal, we apply a two-stage compensation. First, we apply our novel $\mathrm { O B ^ { 2 } C }$ update to the outgoing weight matrix of the block (the second linear layer) by adding $\delta _ { \mathsf { W } }$ computed by by the pruned indices set � to the weights W. Then, we apply mean-shift compensation to the subsequent layer, which is the optimal bias term $\mathbf { b } ^ { * }$ added to the existing bias term.

After the two-stage compensation, we apply pruning. For a standard two-layer block, pruning reduces the intermediate hidden dimension from �<sub>hidden</sub> to $d _ { \mathrm { p r u n e d } }$ . Consequently, the weight matrix of the first layer is reduced from $d _ { \mathrm { i n } } \times d _ { \mathrm { h i d d e n } }$ to $d _ { \mathrm { i n } } \times d _ { \mathrm { p r u n e d } }$ , and the weight matrix of the second layer is reduced from $d _ { \mathrm { h i d d e n } } \times d _ { \mathrm { o u t } }$ to $d _ { \mathrm { p r u n e d } } \times d _ { \mathrm { o u t } }$ . Because the external input $( d _ { \mathrm { i n } } )$ and output $( d _ { \mathrm { o u t } } )$ dimensions remain strictly preserved, the pruned block seamlessly integrates back into the overarching network architecture.

## 4. Experiments

We evaluate our DVBP+ $\cdot O B ^ { 2 } \mathrm { C }$ structured pruning framework on the Tiny, Small, and Base variants of the DeiT [32], Swin [23], and ConvNeXt [24] architectures, all pre-trained on the ImageNet-1K [5] dataset. To compute the activation covariance matrices, we construct a calibration dataset of 8,192 images, randomly sampled from the ImageNet-1K training split using a fixed seed. Notably, we omit data augmentation during calibration to accurately capture the true, unperturbed neuron activation statistics. Our evaluation primarily focuses on direct post-pruning accuracy retention, aiming to maximize of-the-shelf performance without relying on expensive retraining or fine-tuning pipelines. All experiments are conducted on a single NVIDIA GeForce RTX 3090 GPU. To guarantee a rigorous and fair comparison, we locally compute both the uncompressed baseline accuracies and the post-pruning performance of all evaluated methods under the exact same setup.

![](images/1bf697594282c534178a611986e25b66398d159e26d18bf2bc27c597d348746f.jpg)  
Prune Ratio (%)

![](images/28221274fc8034d4a752bc55837c65e06610e21496e22f61793466567bfd2151.jpg)  
Prune Ratio (%)

Figure 1. Pruning accuracy across diverse pruning ratios for DeiT-B (left) and Swin-S (right). SNIP and Magnitude pruning are adjusted to structured pruning.  
![](images/bc46c780d468c3bc6242f304e0e3200057f92f5b851bb5dd1e0690884ef124f9.jpg)  
(a) DeiT-B Layer 9

![](images/43b81f01c9fde77aaee362b661ac5962b4429825fe7137d2a1b884ff066ca129.jpg)  
(b) DeiT-B Layer 10

![](images/4679954db623b16f961c549ebe267aa563b6c31cc2b288b2374d6fe0909738bd.jpg)  
(c) Swin-S Layer 11

![](images/747aa6cef866ef6ebb567630cf6c4525a53635dd716766836fad5056da7eced7.jpg)  
(d) Swin-S Layer 12  
Figure 2. Empirical spectral density fitted with the Marchenko-Pastur distribution. The largest 5% of eigenvalues are excluded for visual clarity.

## 4.1. Results

Our results demonstrate that the proposed approach consistently outperforms all baseline pruning methods across all evaluated models and pruning ratios. Notably, when compared to our closest competitor, VBP [2], the accuracy gap widens as the pruning ratio increases. For instance, on the Swin-S model, the postpruning accuracy gap between our method and VBP is 2.28% at a 40% pruning ratio. This margin increases substantially to 7.33% at a 50% ratio, and expands even further to 25.49% at a 60% ratio.

As detailed in Table 1, at a 50% pruning ratio, our method successfully retains approximately 80% of the original accuracy for the Tiny models, and roughly 90% for the Small and Base models, representing a significant improvement over VBP. This diverging performance trend is further illustrated in Fig. 1, which shows the performance gap between our approach and VBP becoming increasingly pronounced from the 40% pruning threshold onward.

Finally, because our approach targets the same layers and utilizes the same fundamental pruning mechanism as the baselines, the overall parameter counts and multiply-accumulate operations (MACs) remain identical. However, despite this equivalent computational footprint, our method yields a vastly superior top-1 accuracy (Table 2).

As illustrated in Fig. 2, the empirical eigenvalue distributions of the models exhibit a noise structure characteristic of the Marchenko-Pastur (MP) [26] distribution. Furthermore, the fitted MP [26] curve accurately captures these noise components.

![](images/796b99f51f93250129ea3a844875e3dc456acdf4ee67d61a240f8e714b389c72.jpg)

![](images/ab4a81940432dfaef925da4571bab03436586699af974065a99aee5c0eb1ac78.jpg)  
Figure 3. Layerwise pruning ratio for DeiT-B (left) and Swin-S (right), for model pruning rate 50%.

![](images/ff6ddd5599c9731d69804552746f2d66932b831ed96092e54025fb293f6475f3.jpg)

![](images/754939877e258c0fc9ae60b89290a4df31cf2518bf8bf25d0dc26eb9f8f15362.jpg)  
Figure 4. Spearman correlation plot of DeiT-B (left) and Swin-S (right) between denoised and raw variance score rank.

As shown in Fig. 3, our hybrid scoring mechanism yields a distinct pruning profile compared to VBP [2]. Specifically, our approach prunes the earlier layers less aggressively while increasing the pruning rate in the deeper layers. However, because both methods rely on a variance-based metric, their overall macroscopic trends remain similar. This relationship is further quantified by the Spearman rank correlation plot (Fig. 4), which reveals a general linear correlation between the ranks of the standard variance and our denoised scores, albeit with considerable deviations occurring within the middle portion of the rankings.

As demonstrated in Table 3 and Fig. 5, our pruning methodology generalizes efectively to the ConvNeXt [24] Tiny, Small, and Base models. SNIP [21] and Magnitude [14] pruning cause severe performance degradation on ConvNeXt architectures, reducing the accuracy to chance-level performance on the ImageNet-1K dataset. While VBP does achieve some accuracy retention, it is considerably lower than the baseline accuracies. Our method improves upon VBP by 29.46% for ConvNeXt-T, 21.88% for ConvNeXt-S, and 14.64% for ConvNeXt-B, representing a substantial improvement. This demonstrates that our method is highly applicable to diverse model architectures.

## 4.2. Analysis and Ablation Studies

Despite the significant accuracy gains, our method introduces minimal computational and memory overhead, ensuring its practical deployment on consumer-grade GPUs. Because we utilize a block-wise approach in $\mathrm { O B ^ { 2 } C }$ and precompute the pruning selections via our hybrid scoring mechanism, peak VRAM consumption is strictly bounded under 10 GB. Furthermore, the pruning process is highly time-eficient; as detailed in Table 4, while the execution time is approximately 2× that of VBP [2], it completes in under one minute for most architectures, requiring only 66.92 seconds for the larger Swin-B [23] model. These constrained resource requirements demonstrate that our approach can readily scale to larger models without necessitating specialized, high-memory hardware.

Table 3. Results of ConvNeXt Pruning at 50%.
<table><tr><td rowspan=2 colspan=3>Top-1 Accuracy (%)Model</td></tr><tr><td rowspan=1 colspan=1>Full SNIP Mag. VBP</td><td rowspan=1 colspan=1>Ours</td></tr><tr><td rowspan=1 colspan=1>ConvNeXt-T</td><td rowspan=1 colspan=1>84.17  1.90  0.10  15.44</td><td rowspan=1 colspan=1>44.90</td></tr><tr><td rowspan=1 colspan=1>ConvNeXt-S</td><td rowspan=2 colspan=1>85.17 1.76  0.10  28.1385.81 29.91  0.13  56.12</td><td rowspan=2 colspan=1>50.0170.76</td></tr><tr><td rowspan=1 colspan=1>ConvNeXt-B</td></tr><tr><td rowspan=2 colspan=1>Model</td><td rowspan=1 colspan=2>Parameters (M)</td></tr><tr><td rowspan=1 colspan=1>Full Mag. SNIP VBP</td><td rowspan=1 colspan=1>Ours</td></tr><tr><td rowspan=1 colspan=1>ConvNeXt-T</td><td rowspan=1 colspan=1>28.58 16.98 15.94 12.61</td><td rowspan=1 colspan=1>12.96</td></tr><tr><td rowspan=1 colspan=1>ConvNeXt-S</td><td rowspan=2 colspan=1>50.22 27.32 28.50 23.3788.59 48.80 46.92 41.18</td><td rowspan=2 colspan=1>24.0442.63</td></tr><tr><td rowspan=1 colspan=1>ConvNeXt-B</td></tr><tr><td rowspan=2 colspan=1>Model</td><td rowspan=1 colspan=2>MACs (G)</td></tr><tr><td rowspan=1 colspan=1>Full  Mag. SNIP VBP</td><td rowspan=1 colspan=1>Ours</td></tr><tr><td rowspan=1 colspan=1>ConvNeXt-T</td><td rowspan=1 colspan=1>4.46  2.09  2.48  2.95</td><td rowspan=1 colspan=1>2.94</td></tr><tr><td rowspan=1 colspan=2>ConvNeXt-S  8.68  4.34  4.67  5.11</td><td rowspan=1 colspan=1>5.08</td></tr><tr><td rowspan=1 colspan=2>ConvNeXt-B  15.35 7.37  8.44  8.90</td><td rowspan=1 colspan=1>8.85</td></tr></table>

![](images/f3eafd777e605845be139983093d5f76d86c719e41904e8d6afc28e1d603cacb.jpg)  
Figure 5. Overall Diagram for ConvNeXt pruning results.

Table 4. Memory and VRAM usage comparison between VBP.
<table><tr><td rowspan=2 colspan=1>Model</td><td rowspan=1 colspan=4>Peak VRAM (G)        Time (s)</td></tr><tr><td rowspan=1 colspan=2>VBP    Ours</td><td rowspan=1 colspan=2>VBP     Ours</td></tr><tr><td rowspan=1 colspan=1>DeiT-T</td><td rowspan=1 colspan=2>0.52     1.25</td><td rowspan=1 colspan=2>14.44    19.25</td></tr><tr><td rowspan=1 colspan=1>DeiT-S</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>2.48</td><td rowspan=1 colspan=1>13.16</td><td rowspan=1 colspan=1>21.79</td></tr><tr><td rowspan=1 colspan=1>DeiT-B</td><td rowspan=1 colspan=1>2.01</td><td rowspan=1 colspan=1>5.25</td><td rowspan=1 colspan=1>26.66</td><td rowspan=1 colspan=1>51.15</td></tr><tr><td rowspan=1 colspan=1>Swin-T</td><td rowspan=1 colspan=1>3.31</td><td rowspan=1 colspan=1>3.91</td><td rowspan=1 colspan=1>15.03</td><td rowspan=1 colspan=1>24.5</td></tr><tr><td rowspan=1 colspan=1>Swin-S</td><td rowspan=1 colspan=1>3.40</td><td rowspan=1 colspan=1>5.96</td><td rowspan=1 colspan=1>23.88</td><td rowspan=1 colspan=1>44.14</td></tr><tr><td rowspan=1 colspan=1>Swin-B</td><td rowspan=1 colspan=1>4.58</td><td rowspan=1 colspan=1>8.11</td><td rowspan=1 colspan=1>34.89</td><td rowspan=1 colspan=1>66.92</td></tr></table>

Table 5. Ablation study: denoising and ${ \mathrm { O B } } ^ { 2 } { \mathrm { C } } .$
<table><tr><td rowspan="2">Model</td><td colspan="4">Top-1 Accuracy (%)</td></tr><tr><td>Full</td><td>VBP</td><td>+D</td><td>+D +0</td></tr><tr><td>DeiT-T</td><td>72.16</td><td>42.90</td><td>44.32</td><td>56.40</td></tr><tr><td>DeiT-S</td><td>79.86</td><td>66.03</td><td>65.37</td><td>70.25</td></tr><tr><td>DeiT-B</td><td>81.98</td><td>68.92</td><td>69.90</td><td>75.85</td></tr><tr><td>Swin-T</td><td>81.38</td><td>49.98</td><td>52.48</td><td>66.12</td></tr><tr><td>Swin-S</td><td>83.32</td><td>69.91</td><td>71.21</td><td>77.24</td></tr><tr><td>Swin-B</td><td>85.27</td><td>70.86</td><td>70.11</td><td>76.16</td></tr></table>

To evaluate the individual contributions of our proposed components, we analyze the isolated performance of the denoising mechanism and the $\mathrm { O B ^ { 2 } C }$ compensation framework, as detailed in Table 5. Integrating the denoised scoring yields marginal but consistent accuracy gains of 1-2% for the majority of the evaluated models, with the exception of DeiT-S [32] and Swin-B [23], which experience a negligible decrease of less than 1%. In contrast, the application of the $\mathrm { O B ^ { 2 } C }$ framework serves as the primary driver of performance recovery. Notably, it delivers a substantial absolute accuracy improvement of nearly 14% for DeiT-T [32], while consistently boosting the performance of the remaining models by 4-7%.

Because both VBP [2] and our proposed method can accommodate an arbitrary calibration set size, we investigate how varying this size impacts the final top-1 accuracy of each approach. While our primary experiments utilize a calibration dataset of 8,192 samples, our proposed method demonstrates robust scalability with respect to calibration size. As illustrated in Figure $^ { 6 , }$ increasing the calibration set initially yields top-1 accuracy improvements for DeiT-B pruning for both our method and VBP, with both approaches reaching convergence at approximately 8K samples. However, for Swin-S [23], our method exhibits continuous performance gains as the calibration set expands, achieving convergence at roughly 32K samples. In stark contrast, the performance of VBP plateaus entirely in this scenario, showing no measurable improvement across a sweep from 1K to 65K samples.

![](images/7e396c6ca81fabc1b99e3b756d5fbb996d0ad62768d6502f0cd5c45a74134766.jpg)

![](images/813a856593f3e131a536e6600593eb2b17cc5c907b28ede150375f6c74517ab9.jpg)  
Figure 6. Top1-Accuracy (%) across various calibration set sizes for DeiT-B (left) and Swin-S (right).

Table 6. Ablation Study: comparing various denoising methods.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Prune Ratio (%)</td><td colspan="7">Top-1 Accuracy (%)</td></tr><tr><td>No Denoise</td><td>Top 50%</td><td>Top 60%</td><td>Top 70%</td><td>Top 80%</td><td> $< \lambda -$ </td><td> $\pm >$ </td></tr><tr><td rowspan="3">DeiT-B</td><td>50</td><td>75.76</td><td>75.89</td><td>75.71</td><td>75.73</td><td>75.76</td><td>0.14</td><td>75.85</td></tr><tr><td>60</td><td>65.74</td><td>68.76</td><td>67.87</td><td>67.14</td><td>66.19</td><td>0.11</td><td>69.68</td></tr><tr><td>70</td><td>45.49</td><td>51.81</td><td>49.84</td><td>47.49</td><td>45.89</td><td>0.12</td><td>57.18</td></tr><tr><td rowspan="3">Swin-S</td><td>50</td><td>76.67</td><td>77.76</td><td>77.64</td><td>77.41</td><td>76.95</td><td>0.28</td><td>77.24</td></tr><tr><td>60</td><td>66.04</td><td>70.71</td><td>69.42</td><td>68.07</td><td>67.00</td><td>0.16</td><td>70.79</td></tr><tr><td>70</td><td>37.26</td><td>48.89</td><td>43.30</td><td>40.65</td><td>38.12</td><td>0.12</td><td>57.13</td></tr></table>

We systematically compare various denoising strategies to validate our approach of fitting the MP [26] distribution and retaining only the signal eigenvectors associated with eigenvalues larger than $\lambda _ { + }$ . As demonstrated in Table $^ { 6 , }$ isolating eigenvectors with eigenvalues exceeding $\lambda _ { + }$ consistently yields the highest top-1 accuracy among the evaluated denoising methods. Furthermore, this analysis confirms that eigenvectors with eigenvalues below $\lambda _ { - }$ predominantly capture noise, as reconstructing representations solely from these components degrades performance to chance-level accuracy. To quantify the extent of eigenvector pruning performed by our MP fitting method, we visualize the layer-wise pruning ratios in Figure 7. The results indicate that our method dynamically adjusts the pruning severity according to network depth; it discards approximately 60-70% of eigenvectors in the early layers and increases this ratio to roughly 80-90% in the deeper layers, a trend consistently observed in both the DeiT-B [32] and Swin-S [23] models. This adaptive behavior supports the hypothesis that representations become increasingly low-rank as they propagate deeper into the network, thereby requiring fewer signal eigenvectors to accurately capture the underlying information.

## 5. Conclusion

In this paper, we introduce $\mathrm { D V B P } + \mathrm { O B } ^ { 2 } \mathrm { C } .$ , a novel structured pruning methodology that integrates statistical denoising techniques—specifically, Marchenko-Pastur (MP) distribution fitting within the VBP framework—with a highly eficient compensation mechanism. Our $\mathrm { O B ^ { 2 } C }$ framework advances the standard Optimal Brain Compression (OBC) approach by incorporating a multi-weight update strategy for pruned weights, further refined by mean-shift compensation. Extensive evaluations demonstrate that our method achieves substantial improvements in post-pruning accuracy without the need for subsequent fine-tuning or retraining. Because this approach is fundamentally applicable to any linear layer, it generalizes seamlessly across a wide range of modern deep neural network architectures. We hope this work encourages further exploration into structured pruning within the computer vision community, paving the way for near-lossless compression combined with significant inference acceleration.

![](images/2ecbf2b77a1732f57fd9018adcda85d84c4d081446bb2100a17d89ffc9801896.jpg)  
(a) DeiT-B

![](images/479dd910488283f30f62785c07084d1d0878b68cf06c1033813cdfa1da32af9c.jpg)  
(b) Swin-S  
Figure 7. Eigenvector prune ratio based on Marchenko-Pastur denoising.

## References

[1] Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. In NeurIPS, 2024.

[2] Uranik Berisha, Jens Mehnert, and Alexandru Paul Condurache. Variance-Based Pruning for Accelerating and Compressing Trained Networks. In ICCV, 2025.

[3] Tony F. Chan, Gene H. Golub, and Randall J. LeVeque. Updating Formulae and a Pairwise Algorithm for Computing Sample Variances. In COMPSTAT, 1982.

[4] Arnav Chavan, Zhiqiang Shen, Zhuang Liu, Zechun Liu, Kwang-Ting Cheng, and Eric Xing. Vision transformer slimming: Multi-dimension searching in continuous optimization space. In CVPR, 2022.

[5] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR, 2009.

[6] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In ICLR, 2021.

[7] Murali Emani, Sam Foreman, Varuni Sastry, Zhen Xie, Siddhisatya Raskar, William Arnold, Rajeev Thakur, Venkatram Vishwanath, and Michael E. Papka. A Comprehensive Performance Study of Large Language Models on Novel AI Accelerators. arXiv preprint arXiv:2310.04607, 2023.

[8] Jonathan Frankle and Michael Carbin. The Lottery Ticket Hypothesis: Finding Sparse, Trainable Networks. In ICLR, 2019.

[9] Elias Frantar and Dan Alistarh. SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot. In ICML, 2023.

[10] Elias Frantar, Sidak Pal Singh, and Dan Alistarh. Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning. In NeurIPS, 2022.

[11] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. In ICLR, 2023.

[12] Trevor Gale, Erich Elsen, and Sara Hooker. The State of Sparsity in Deep Neural Networks. arXiv preprint arXiv:1902.09574, 2019.

[13] Matan Gavish and David L. Donoho. The Optimal Hard Threshold for Singular Values is $4 / { \sqrt { 3 } } .$ . IEEE TIT, 2014.

[14] Song Han, Jef Pool, John Tran, and William J. Dally. Learning both Weights and Connections for Eficient Neural Network. In NeurIPS, 2015.

[15] Babak Hassibi and David Stork. Second order derivatives for network pruning: Optimal Brain Surgeon. In NeurIPS, 1992.

[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. In CVPR, 2016.

[17] Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, and Alexandra Peste. Sparsity in Deep Learning: Pruning and growth for eficient inference and training in neural networks. JMLR, 2021.

[18] Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, and Hartwig Adam. Searching for MobileNetV3. In ICCV, 2019.

[19] Yann LeCun, John S. Denker, and Sara A. Solla. Optimal Brain Damage. In NeurIPS, 1989.

[20] Kwanhee Lee, Hyeondo Jang, Dongyeop Lee, Dan Alistarh, and Namhoon Lee. The Unseen Frontier: Pushing the Limits of LLM Sparsity with Surrogate-Free ADMM. In ICLR, 2025.

[21] Namhoon Lee, Thalaiyasingam Ajanthan, and Philip H. S. Torr. SNIP: Single-Shot Network Pruning based on Connection Sensitivity. In ICLR, 2019.

[22] Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning Filters for Eficient ConvNets. In ICLR, 2017.

[23] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. In ICCV, 2021.

[24] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A ConvNet for the 2020s. In CVPR, 2022.

[25] Jian-Hao Luo and Jianxin Wu. An Entropy-based Pruning Method for CNN Compression. arXiv preprint arXiv:1706.05791, 2017.

[26] Vladimir A. Marchenko and Leonid Andreevich Pastur. Distribution of eigenvalues for some sets of random matrices. Mathematics of the USSR-Sbornik, 1967.

[27] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In ICLR, 2017.

[28] Philippe Pebay. Formulas for Robust, One-Pass Parallel Computation of Covariances and Arbitrary-Order Statistical Moments. Technical report, Sandia National Laboratories, 2008.

[29] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2025.

[30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.

[31] Hidenori Tanaka, Daniel Kunin, Daniel L. K. Yamins, and Surya Ganguli. Pruning neural networks without any data by iteratively conserving synaptic flow. In NeurIPS, 2020.

[32] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-eficient image transformers & distillation through attention. In ICML, 2021.

[33] Xinchao Wang Xinyin Ma, Gongfan Fang. Llm-pruner: On the structural pruning of large language models. In NeurIPS, 2023.

[34] Huanrui Yang, Hongxu Yin, Maying Shen, Pavlo Molchanov, Hai Li, and Jan Kautz. Global Vision Transformer Pruning with Hessian-Aware Saliency. In CVPR, 2023.

[35] Lu Yu and Wei Xiang. X-Pruner: eXplainable Pruning for Vision Transformers. In CVPR, 2023.

[36] Junhan Zhu, Hesong Wang, Mingluo Su, Zefang Wang, and Huan Wang. OBS-Dif: Accurate Pruning for Difusion Models in One-Shot. In ICLR, 2026.

## A. Additional Details

In this supplementary material, the following additional details are provided:

• DVBP + OB<sup>2</sup>C Algorithm

• Mathematical Proof of $\mathrm { O B ^ { 2 } C }$

• Calibration Set Experiments

• Additional Experiments

## A.1. DVBP + OB<sup>2</sup>C Algorithm

In this section, we provide the step-by-step pseudocode for our combined DVBP and $\mathrm { O B ^ { 2 } C }$ methodology.

## Algorithm 1 DVBP Pruning with $\mathrm { O B ^ { 2 } C }$ Compensation

Input: Pretrained model, calibration dataset D, pruning ratio �   
1: Compute layer-wise centered covariance matrices using Algorithm 2   
2: Denoise covariance matrices and compute hybrid scores using Algorithm 3   
3: Prune neurons and update weights using Algorithm 4   
4: return Pruned model

```latex
Algorithm 2 Online Covariance Update per MLP Layer
Input: Calibration dataset D, pretrained model
1: Initialize covariance matrix $\mathbf { C } \gets 0 ,$ mean vector $\mu  0 ,$ total samples $n \gets 0$
2: for � = 1 . . . total batches do
3: Obtain batch activations x
4: Compute batch mean $\mu _ { N }$ and batch size $n _ { N }$
5: $\widetilde { { \mathbf { x } } }  { \mathbf { x } } - { \pmb { \mu } } _ { N }$
6: $\begin{array} { r } { \mathbf { C } _ { N } \gets \frac { \widetilde { \mathbf { x } } ^ { T } \widetilde { \mathbf { x } } } { n _ { N } } } \end{array}$
7: if � == 0 then
8: $\mathbf { C } \gets \mathbf { C } _ { N }$
9: � ← �<sub>N</sub>
10: � ← �<sub>N</sub>
11: else
12: �<sub>P</sub> ← �
13: �<sub>P</sub> ← �
14: $n \gets n _ { \mathcal P } + n _ { N }$
15: $\textstyle { \alpha \gets { \frac { n \varphi } { n } } }$
16: $\pmb { \delta } \gets \ddot { \pmb { \mu } _ { N } } - \pmb { \mu } _ { \mathcal { P } }$
17: $\mathbf { C } \longleftarrow \alpha \mathbf { \overset { . } { C } } + \mathbf { \dot { \alpha } ( 1 } - \alpha ) \mathbf { C } _ { N } + \alpha ( 1 - \alpha ) \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } ^ { T }$
18: $\pmb { \mu }  \pmb { \mu } _ { \mathcal { P } } + ( 1 - \alpha ) \pmb { \delta }$
19: end if
20: end for
21: return $\mathbf { C } , \mu$
```

## Algorithm 3 Denoise Covariance Matrix and Obtain Hybrid Scores

2: $[ \mathbf { V } , \pmb { \Lambda } ] \longleftarrow \mathrm { e i g } ( \mathbf { C } )$   
3: $d $ number of neurons   
4: $\textstyle \gamma  { \frac { d } { n } }$   
5: $\lambda _ { \pm }  ( 1 \pm \sqrt { \lambda } ) ^ { 2 }$   
6: Obtain $\mu \lambda$ by solving   
$\int _ { \lambda _ { - } } ^ { x } \frac { \sqrt { ( \lambda _ { + } - t ) ( t - \lambda _ { - } ) } } { 2 \pi t } d t = \frac 1 2$   
7: $\lambda _ { \mathrm { { m e d } } } $ median eigenvalue of Λ   
8: $\hat { \sigma } \gets \sqrt { \frac { \lambda _ { \mathrm { m e d } } } { \mu _ { \lambda } } }$   
9: $\lambda _ { + }  \hat { \sigma } ^ { 2 } ( 1 + \sqrt { \lambda } ) ^ { 2 }$   
10: $\pmb { \Lambda } _ { \mathrm { s i g n a l } }  \mathrm { d i a g } ( \lambda _ { i } )$ for all $\lambda _ { i } > \lambda _ { + }$   
11: $\mathbf { V } _ { \mathrm { s i g n a l } }  [ \mathbf { v } _ { i } ]$ for all $\lambda _ { i } > \lambda _ { + }$ +   
12: $\mathbf { C } _ { \mathrm { d e n o i s e d } }  \mathbf { V } _ { \mathrm { s i g n a l } } \mathbf { A } _ { \mathrm { s i g n a l } } \mathbf { V } _ { \mathrm { s i g n a l } } ^ { T }$   
signal   
13: $\mathbf { s } _ { \mathrm { d v a r } } \gets \mathrm { d i a g } ( \mathbf { C } _ { \mathrm { d e n o i s e d } } )$   
14: $\begin{array} { r } { \pmb { s } _ { \mathrm { h y b r i d } }  ( 1 - p ) \pmb { s } _ { \mathrm { v a r } } + p \pmb { s } _ { \mathrm { d v a r } } } \end{array}$   
15: $\begin{array} { r } { \mathbf { C } _ { \mathrm { d e n o i s e d } }  \mathbf { C } _ { \mathrm { d e n o i s e d } } + ( \delta \frac { \mathrm { T r } ( \mathbf { C } _ { \mathrm { d e n o i s e d } } ) } { d } ) I } \end{array}$   
16: $\mathbf { C } ^ { - 1 } \gets \mathbf { C } _ { \mathrm { d e n } } ^ { - 1 }$ noised   
17: return $\mathbf { C } ^ { - 1 } , \mathbf { s } _ { \mathrm { h y b r i d } }$

Algorithm 4 DVBP Pruning with $\mathrm { O B ^ { 2 } C }$ Compensation   
Input: Hybrid scores $\mathbf { s } _ { \mathrm { h y b r i d . } }$ , inverse covariance $\mathbf { C } ^ { - 1 }$ , mean $\mu ,$ pruning ratio $p$   
1: Concatenate all block scores to obtain global scores ${ \pmb s } _ { \mathrm { g l } }$ obal   
$2 \colon \pi  \operatorname { a r g s o r t } ( s _ { \mathrm { g l o b a l } } )$   
3: $k \gets p \cdot | \mathbf { s } _ { \mathrm { g l o b a l } } |$   
4: $\mathcal { P } _ { \mathrm { g l o b a l } }  \mathsf { \bar { \{ } }  \pi _ { 1 } , \ldots , \pi _ { k } \}$   
5: for each MLP block � do   
6: Map global indices to block indices: $\mathcal { P } _ { l }$   
7: $\mathcal { K } _ { l } \gets \{ 1 , \dots , d _ { l } \} \setminus \mathcal { P } _ { l }$   
8: $\mathbf { W 1 } _ { \mathrm { p r u n e d } }  \mathbf { W 1 } _ { \mathcal { K } _ { l } , }$   
9: $\mathbf { b } \mathbf { 1 } _ { \mathrm { p r u n e d } }  \mathbf { b } \mathbf { 1 } _ { \mathcal { K } _ { l } }$   
10: $\mathbf { W } \hat { \mathbf { 2 } } _ { \mathrm { n e w } } , \mathbf { b } \mathbf { 2 } _ { \mathrm { n e w } } \gets \mathrm { O B } ^ { 2 } \mathbf { C }$ update $( \mathbf { W 2 } , \mathbf { b 2 } , \mathbf { C } ^ { - 1 } , \mu , \mathcal { P } _ { l } )$   
11: $\mathbf { W 2 } _ { \mathrm { p r u n e d } }  [ \mathbf { W 2 } _ { \mathrm { n e w } } ] _ { : , \mathcal { K } _ { l } }$   
12: end for

Algorithm $5 \mathrm { O B ^ { 2 } C }$ Update   
Input: Weights $\mathbf { W } ,$ bias $\mathbf { b } ,$ inverse covariance $\mathbf { C } ^ { - 1 }$ , mean $\mu ,$ pruned indices $\mathcal { P }$   
1: $\mathbf { A } \gets [ \mathbf { C } ^ { - 1 } ] _ { \mathcal { P } , \mathcal { P } }$   
2: $\mathbf { B } \gets [ { \bf C } ^ { - 1 } ] _ { \mathcal { P } , : }$   
3: $\mathbf { W } _ { \mathcal { P } } \gets \mathbf { W } _ { : , \mathcal { P } }$   
4: $\Delta \mathbf { W } \longleftarrow - \mathbf { W } \mathcal { P } \mathbf { A } ^ { - 1 } \mathbf { B }$   
5: $\mathbf { W } _ { \mathrm { n e w } }  \mathbf { W } + \Delta \mathbf { W }$   
6: $\mathbf { b } _ { \mathrm { n e w } }  \mathbf { b } - \Delta \mathbf { W } \mu$   
7: return $\mathbf { W } _ { \mathrm { n e w } } , \mathbf { b } _ { \mathrm { n e w } }$

## A.2. Mathematical Proof of OB<sup>2</sup>C

In this section, we introduce the bias term into the layer-wise objective function as follows:

$$
\arg \operatorname* { m i n } _ { { \hat { \mathbf { W } } } , \mathbf { b } } \left\| \mathbf { W X } - ( { \hat { \mathbf { W } } } \mathbf { X } + \mathbf { b } \mathbf { 1 } ^ { T } ) \right\| _ { F } ^ { 2 } .\tag{24}
$$

To solve this joint optimization problem, we first derive the optimal bias $\mathbf { b } ^ { * }$ for any given weight perturbation, which allows us to substitute it back into the objective and optimize solely for W<sup>ˆ</sup> .

Let $\mathbf { W } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ be the weight matrix of a single layer and $\pmb { \mathrm { X } } \in \mathbb { R } ^ { d _ { \mathrm { i n } } \times N }$ be the input activation data, where � is the calibration set size. The reconstruction error induced by pruning is:

$$
E = \left\| \mathbf { W X } - ( \hat { \mathbf { W X } } + \mathbf { b } \mathbf { 1 } ^ { T } ) \right\| _ { F } ^ { 2 } .\tag{25}
$$

To find the optimal bias term, we take the derivative of � with respect to the vector b and set it to zero:

$$
\frac { \partial E } { \partial \mathbf { b } } = - 2 \sum _ { i = 1 } ^ { N } ( \mathbf { W } \mathbf { x } _ { i } - \widehat { \mathbf { W } } \mathbf { x } _ { i } - \mathbf { b } ) = \mathbf { 0 } .\tag{26}
$$

This can be rewritten as:

$$
\sum _ { i = 1 } ^ { N } ( \mathbf { W } \mathbf { x } _ { i } - \widehat { \mathbf { W } } \mathbf { x } _ { i } ) - N \mathbf { b } = \mathbf { 0 } .\tag{27}
$$

Solving for b yields the optimal bias vector b<sup>∗</sup>:

$$
\mathbf { b } ^ { * } = ( \mathbf { W } - { \widehat { \mathbf { W } } } ) { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } \mathbf { x } _ { i } = ( \mathbf { W } - { \widehat { \mathbf { W } } } ) \pmb { \mu } = \Delta \mathbf { W } \pmb { \mu } .\tag{28}
$$

Substituting this optimal bias b<sup>∗</sup> back into the original error term �:

$$
\begin{array} { r l } & { E = \left\| \mathbf { W X } - ( \widehat { \mathbf { W } } \mathbf { X } + \mathbf { b } ^ { * } \mathbf { \mathbb { 1 } } ^ { T } ) \right\| _ { F } ^ { 2 } } \\ & { = \left\| \mathbf { W X } - ( \widehat { \mathbf { W } } \mathbf { X } + ( \mathbf { W } - \widehat { \mathbf { W } } ) \mu \mathbf { \mathbb { 1 } } ^ { T } ) \right\| _ { F } ^ { 2 } , } \\ & { = \left\| \mathbf { W X } - \widehat { \mathbf { W } } \mathbf { X } - ( \mathbf { W } - \widehat { \mathbf { W } } ) \mu \mathbf { \mathbb { 1 } } ^ { T } \right\| _ { F } ^ { 2 } } \\ & { = \left\| ( \mathbf { W } - \widehat { \mathbf { W } } ) \mathbf { X } - ( \mathbf { W } - \widehat { \mathbf { W } } ) \mu \mathbf { \mathbb { 1 } } ^ { T } \right\| _ { F } ^ { 2 } } \\ & { = \left\| ( \mathbf { W } - \widehat { \mathbf { W } } ) ( \mathbf { X } - \mu \mathbf { 1 } ^ { T } ) \right\| _ { F } ^ { 2 } } \\ & { = \left\| ( \mathbf { W } - \widehat { \mathbf { W } } ) ( \mathbf { X } - \mu \mathbf { 1 } ^ { T } ) \right\| _ { F } ^ { 2 } } \\ & { = \left\| \Delta \mathbf { W } \widetilde { \mathbf { M } } \right\| _ { F } ^ { 2 } } \end{array}\tag{29}
$$

where $\widetilde { \mathbf X } = \mathbf X - \pmb { \mu } \mathbf 1 ^ { T }$ represents the mean-centered activations. Hence, the joint objective formula simplifies entirely to:

$$
\arg \operatorname* { m i n } _ { \Delta \mathbf { W } } \| \Delta \mathbf { W } \widetilde { \mathbf { X } } \| _ { F } ^ { 2 } .\tag{30}
$$

Because the Frobenius norm separates across matrix rows, the second derivative of the simplified objective with respect to the full matrix ΔW produces a block-diagonal Hessian:

$$
\mathbf { H } _ { \mathrm { f u l l } } = \mathbf { I } _ { d _ { \mathrm { o u t } } } \otimes 2 \widetilde { \mathbf { X } } \widetilde { \mathbf { X } } ^ { T } = \mathbf { I } _ { d \mathrm { o u t } } \otimes \mathbf { H } ,\tag{31}
$$

where $\mathbf { H } = 2 \widetilde { \mathbf { X } } \widetilde { \mathbf { X } } ^ { T }$ is the shared row-wise Hessian. The inverse of a block-diagonal matrix is obtained by inverting each block:

$$
\mathbf { H } _ { \mathrm { f u l l } } ^ { - 1 } = \mathbf { I } _ { d _ { \mathrm { o u t } } } \otimes \mathbf { H } ^ { - 1 } .\tag{32}
$$

Thus, each row update depends only on the same $d _ { \mathrm { i n } } \times d _ { \mathrm { i n } }$ matrix ${ \bf { H } } ^ { - 1 }$

Substituting the unscaled covariance of the centered activations, $\begin{array} { r } { { \bf C } = \frac { 1 } { N } \widetilde { \bf X } \widetilde { \bf X } ^ { T } } \end{array}$ , gives

$$
\mathbf { H } = 2 \widetilde { \mathbf { X } } \widetilde { \mathbf { X } } ^ { T } = 2 N \mathbf { C } .\tag{33}
$$

In the algorithm, the constant factor 2� cancels during the inverse update, allowing the exact row-wise Hessian to be replaced with the computationally eficient online covariance matrix C.

## A.3. Additional Experiments

## A.3.1. Experimental Setting

All results rely on ImageNet-1K [5] training data for calibration, except for the LLM experiments in Table 13, which use Wikitext-2 [27]. Unless otherwise specified, the calibration set size is 8192. All experiments were conducted on a single NVIDIA GeForce RTX 3090 GPU.

## A.3.2. Full Experimental Results

This section details the complete experimental results referenced in the main paper. Table 7 present the comprehensive tabular data for the calibration sweeps. Table 8 details full results of layerwise pruning ratio after our framework is applied. Table 9 provides whole eigenvector pruning ratio.

Furthermore, we provide extended visualizations: Figure 8 plots the full spearman correlation plots of denoised and raw variance scores, while Figure 9, Figure 10, and Figure 11 illustrate the complete empirical spectral density plots for both the DeiT-B and Swin-S models.

Table 7. Calibration set size experiments. Model pruning ratio is 50% for all models.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Full</td><td rowspan="2">Method</td><td colspan="7">Calibration Set Size</td></tr><tr><td>1024</td><td>2048</td><td>4096</td><td>8192</td><td>16384</td><td>32768</td><td>65536</td></tr><tr><td rowspan="2">DeiT-T</td><td rowspan="2">72.16</td><td>Ours</td><td>53.53</td><td>54.79</td><td>55.64</td><td>56.40</td><td>56.95</td><td>57.29</td><td>57.60</td></tr><tr><td>VBP</td><td>43.65</td><td>42.98</td><td>42.98</td><td>42.89</td><td>42.93</td><td>42.78</td><td>42.75</td></tr><tr><td rowspan="2">DeiT-S</td><td rowspan="2">79.86</td><td>Ours</td><td>68.62</td><td>68.47</td><td>69.58</td><td>70.25</td><td>70.70</td><td>70.91</td><td>71.05</td></tr><tr><td>VBP</td><td>65.90</td><td>65.98</td><td>66.00</td><td>66.03</td><td>66.30</td><td>66.25</td><td>66.18</td></tr><tr><td rowspan="2">DeiT-B</td><td rowspan="2">81.98</td><td>Ours</td><td>72.67</td><td>73.86</td><td>74.99</td><td>75.85</td><td>76.08</td><td>76.21</td><td>76.27</td></tr><tr><td>VBP</td><td>66.44</td><td>67.69</td><td>68.64</td><td>68.92</td><td>69.29</td><td>69.32</td><td>69.37</td></tr><tr><td rowspan="2">Swin-T</td><td rowspan="2">81.38</td><td>Ours</td><td>61.42</td><td>63.05</td><td>64.79</td><td>66.12</td><td>67.27</td><td>67.63</td><td>67.81</td></tr><tr><td>VBP</td><td>49.83</td><td>49.15</td><td>49.96</td><td>49.98</td><td>50.12</td><td>49.66</td><td>49.75</td></tr><tr><td rowspan="2">Swin-S</td><td rowspan="2">83.32</td><td>Ours</td><td>74.67</td><td>75.05</td><td>76.61</td><td>77.24</td><td>77.60</td><td>77.84</td><td>77.87</td></tr><tr><td>VBP</td><td>69.83</td><td>69.84</td><td>70.06</td><td>69.91</td><td>69.75</td><td>69.67</td><td>69.72</td></tr><tr><td rowspan="2">Swin-B</td><td rowspan="2">85.27</td><td>Ours</td><td>74.60</td><td>74.43</td><td>75.15</td><td>76.16</td><td>77.44</td><td>78.15</td><td>78.62</td></tr><tr><td>VBP</td><td>69.86</td><td>70.22</td><td>70.71</td><td>70.86</td><td>71.13</td><td>71.29</td><td>71.40</td></tr></table>

Table 8. Layerwise pruning ratios. Model pruning ratio is 50% for all models.
<table><tr><td rowspan="2">Model</td><td colspan="11">Layerwise Pruning Ratio (%)</td></tr><tr><td>L1</td><td>L2</td><td>L3</td><td>L4</td><td>L5</td><td>L6</td><td>L7</td><td>L8</td><td>L9</td><td>L10</td><td>L11</td><td>L12</td></tr><tr><td>DeiT-T</td><td>14.45</td><td>41.02</td><td>26.17</td><td>40.76</td><td>51.69</td><td>50.52</td><td>42.58</td><td>50.78</td><td>52.08</td><td>56.64</td><td>80.86</td><td>92.45</td></tr><tr><td>DeiT-S</td><td>26.17</td><td>33.85</td><td>37.24</td><td>53.78</td><td>50.91</td><td>43.95</td><td>34.64</td><td>35.22</td><td>46.68</td><td>59.96</td><td>85.09</td><td>92.51</td></tr><tr><td>DeiT-B</td><td>66.47</td><td>88.96</td><td>79.52</td><td>71.78</td><td>64.26</td><td>60.06</td><td>51.66</td><td>35.06</td><td>18.59</td><td>20.02</td><td>10.35</td><td>33.27</td></tr><tr><td>Swin-T</td><td>24.74</td><td>29.95</td><td>14.06</td><td>45.57</td><td>84.38</td><td>73.50</td><td>67.77</td><td>59.11</td><td>50.26</td><td>46.81</td><td>62.79</td><td>12.04</td></tr><tr><td colspan="2"></td><td colspan="10"></td></tr><tr><td rowspan="2">Model Swin-S</td><td></td><td>L2</td><td></td><td></td><td></td><td></td><td>Layerwise Pruning Ratio (%)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>L1</td><td></td><td>L3</td><td>L4</td><td>L5</td><td>L6</td><td>L7</td><td>L8</td><td>L9</td><td>L10</td><td>L11</td><td>L12</td></tr><tr><td>Swin-B</td><td>30.73 33.79</td><td>25.26 22.27</td><td>18.75 11.23</td><td>39.19 31.54</td><td>92.32 91.80</td><td>84.96 84.38</td><td>80.79 72.66</td><td>84.77</td><td>79.04</td><td>72.85</td><td>68.68</td><td>61.39</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>75.59</td><td>71.04</td><td>66.65</td><td>59.57</td><td>54.00</td></tr><tr><td rowspan="2">Model</td><td colspan="10">Layerwise Pruning Ratio (%)</td></tr><tr><td>L13</td><td>L14</td><td>L15</td><td>L16</td><td>L17</td><td>L18</td><td>L19</td><td>L20</td><td>L21</td><td>L22</td><td>L23</td><td>L24</td></tr><tr><td>Swin-S</td><td>58.14</td><td>57.62</td><td>54.75</td><td>50.33</td><td>41.15</td><td>30.79</td><td>30.66</td><td>29.95</td><td>27.86</td><td>18.82</td><td>46.16</td><td>7.42</td></tr><tr><td>Swin-B</td><td>49.41</td><td>49.27</td><td>46.92</td><td>47.17</td><td>43.60</td><td>37.84</td><td>22.75</td><td>15.77</td><td>11.28</td><td>6.84</td><td>69.14</td><td>47.39</td></tr></table>

Table 9. Layerwise eigenvector pruning ratios based on Marchenko-Pastur denoising.
<table><tr><td rowspan="2">Model</td><td colspan="11">Layerwise Eigenvector Pruning Ratio (%)</td></tr><tr><td>L1</td><td>L2</td><td>L3</td><td>L4</td><td>L5</td><td>L6</td><td>L7</td><td>L8</td><td>L9</td><td>L10</td><td>L11</td><td>L12</td></tr><tr><td>DeiT-T</td><td>64.19</td><td>67.06</td><td>69.66</td><td>70.44</td><td>70.96</td><td>72.40</td><td>72.79</td><td>71.35</td><td>69.14</td><td>71.48</td><td>68.36</td><td>71.09</td></tr><tr><td>DeiT-S</td><td>67.19</td><td>72.85</td><td>75.13</td><td>78.06</td><td>78.58</td><td>80.08</td><td>81.64</td><td>78.91</td><td>78.19</td><td>73.89</td><td>67.25</td><td>73.76</td></tr><tr><td>DeiT-B</td><td>54.72</td><td>72.66</td><td>53.55</td><td>82.39</td><td>84.86</td><td>84.99</td><td>86.13</td><td>87.76</td><td>88.96</td><td>83.56</td><td>81.25</td><td>86.46</td></tr><tr><td>Swin-T</td><td>58.85</td><td>63.54</td><td>69.79</td><td>72.40</td><td>77.28</td><td>78.65</td><td>79.88</td><td>80.92</td><td>80.53</td><td>80.60</td><td>87.86</td><td>86.36</td></tr><tr><td></td><td colspan="14"></td></tr><tr><td rowspan="2">Model Swin-S</td><td></td><td>L2</td><td></td><td></td><td></td><td></td><td></td><td>Layerwise Eigenvector Pruning Ratio (%)</td><td></td><td></td><td></td><td></td></tr><tr><td>L1</td><td></td><td>L3</td><td>L4</td><td>L5</td><td>L6</td><td>L7</td><td>L8</td><td>L9</td><td>L10</td><td>L11</td><td>L12</td></tr><tr><td>Swin-B</td><td>58.07 59.38</td><td>62.50 65.23</td><td>69.14 71.29</td><td>72.27 73.83</td><td>74.15 78.17</td><td>76.04 80.27</td><td>75.59 79.15</td><td>77.34</td><td>76.63</td><td>77.47</td><td>78.19</td><td>79.10</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>79.93</td><td>80.66</td><td>80.52</td><td>80.32</td><td>81.64</td></tr><tr><td rowspan="2">Model</td><td colspan="10">Layerwise Eigenvector Pruning Ratio (%)</td></tr><tr><td>L13</td><td>L14</td><td>L15</td><td>L16</td><td>L17</td><td>L18</td><td>L19</td><td>L20</td><td>L21</td><td>L22</td><td>L23</td><td>L24</td></tr><tr><td>Swin-S</td><td>80.92</td><td>81.45</td><td>81.38</td><td>82.29</td><td>82.10</td><td>82.42</td><td>79.17</td><td>78.97</td><td>77.99</td><td>78.26</td><td>73.70</td><td>84.83</td></tr><tr><td>Swin-B</td><td>82.47</td><td>83.30</td><td>84.08</td><td>84.91</td><td>84.42</td><td>85.06</td><td>83.64</td><td>82.91</td><td>81.84</td><td>82.76</td><td>84.23</td><td>88.28</td></tr></table>

## A.3.3. Further Analysis

To rigorously validate our framework, we perform additional analysis and ablation studies. Table 10 provides comparison with additional baselines adapted for structured MLP pruning: Hessian-based NViT [34], gradientbased SynFlow [31], and training-based ViT-Slim [4]. Our method outperforms all baselines.

Table 10. Experiment with additional baselines. Model pruning ratio is 50% for all models.
<table><tr><td rowspan="2">Model</td><td colspan="5">Top-1 Accuracy  $( \% )$ </td></tr><tr><td>Full</td><td>NViT</td><td>SynFlow</td><td>ViT-Slim</td><td>Ours</td></tr><tr><td>DeiT-B</td><td>81.98</td><td>58.01</td><td>2.24</td><td>72.34</td><td>75.85</td></tr><tr><td>Swin-S</td><td>83.32</td><td>53.39</td><td>0.10</td><td>73.56</td><td>77.24</td></tr></table>

Table 11. Ablation study: denoising, mean-shift, and $\mathrm { O B ^ { 2 } C }$
<table><tr><td rowspan="2">PR (%)</td><td rowspan="2">D</td><td rowspan="2">MS</td><td rowspan="2">OB²C</td><td colspan="6">Top-1 Accuracy (%)</td></tr><tr><td>DeiT-T</td><td>DeiT-S</td><td>DeiT-B</td><td>Swin-T</td><td>Swin-S</td><td>Swin-B</td></tr><tr><td rowspan="4">70</td><td>X</td><td>0</td><td>0</td><td>35.33</td><td>56.98</td><td>45.49</td><td>32.36</td><td>33.80</td><td>54.06</td></tr><tr><td>0</td><td>X</td><td>0</td><td>9.60</td><td>41.61</td><td>12.76</td><td>8.31</td><td>0.47</td><td>10.33</td></tr><tr><td>0</td><td>0</td><td>X</td><td>14.19</td><td>38.34</td><td>42.07</td><td>16.89</td><td>35.37</td><td>39.79</td></tr><tr><td>0</td><td>0</td><td>0</td><td>35.64</td><td>56.84</td><td>57.18</td><td>42.16</td><td>57.13</td><td>58.25</td></tr></table>

Table 11 ablates the individual contributions of denoising, mean-shift, and ${ \mathrm { O B } } ^ { 2 } { \mathrm { C } } .$ Denoising results in better accuracy at high pruning ratio, while mean-shift and $\mathsf { \bar { O B } ^ { 2 } C }$ increase accuracy across all pruning ratios. Table 12 demonstrates robustness to the calibration set distribution by measuring the standard deviation of Top-1 accuracy across three calibration set seeds. We see no meaningful deviation.

## A.3.4. Scalability and Applicability

To validate scalability, Table 13 evaluates our framework on larger architectures, including ViT-L [30] and Qwen2.5 1.5B [29]. For Qwen’s SwiGLU MLPs, we calibrate on 128 Wikitext-2 samples (2048-token length) and apply uniform layer-wise pruning based on our hybrid scores. Specifically, we prune the lowest-scoring down-projection input channels and their corresponding up and gate-projection output channels, followed by $\mathrm { O B } ^ { \bar { 2 } } \mathrm { C }$ updates and mean-shift compensation on the down-projection layer. On the Wikitext-2 test set, our method outperforms all baselines, including the LLM-specific LLM-Pruner [33].

Furthermore, Table 14 demonstrates our method’s generalizability to Multi-Head Attention (MHA) layers in ViT models. By averaging our hybrid scores across all attention heads, we uniformly prune the lowest-scoring QKV output channels and their corresponding out-projection input channels. With the application of $\mathrm { O B ^ { 2 } C }$ updates and mean-shift compensation, our approach again outperforms all evaluated baselines.

Table 12. Top-1 accuracy standard deviation measured across three calibration sets, with random seeds 100, 200, and 300.
<table><tr><td>Prune Ratio (%)</td><td>DeiT-T</td><td>DeiT-S</td><td>DeiT-B</td><td>Swin-T</td><td>Swin-S</td><td>Swin-B</td></tr><tr><td>50</td><td>0.20</td><td>0.10</td><td>0.12</td><td>0.21</td><td>0.08</td><td>0.11</td></tr></table>

Table 13. LLM and larger vision model pruned and calibrated on Wikitext-2 (128 samples) and ImageNet-1K (8192 samples).
<table><tr><td rowspan="2">Prune Ratio (%)</td><td colspan="4">Qwen 2.5 1.5B Wiki-2 PPL (↓)</td><td colspan="3">ViT-L Top-1 Acc. (%)</td></tr><tr><td>Full</td><td>LLM Pruner</td><td>VBP</td><td>Ours</td><td>Full</td><td>VBP</td><td>Ours</td></tr><tr><td>40</td><td></td><td>488.60</td><td>15.69</td><td>12.56</td><td></td><td>83.42</td><td>84.30</td></tr><tr><td>50</td><td>8.19</td><td>9557.71</td><td>28.35</td><td>16.66</td><td>87.90</td><td>77.60</td><td>81.72</td></tr><tr><td>60</td><td></td><td>29978.11</td><td>58.75</td><td>32.44</td><td></td><td>59.71</td><td>76.58</td></tr></table>

Table 14. Top-1 acc. results for structured pruning of MHA layer. Model pruning ratio is 30% for all models.
<table><tr><td rowspan="2">Model</td><td colspan="5">MHA Pruning Top-1 Accuracy (%)</td></tr><tr><td>Full</td><td>Mag.</td><td>SNIP</td><td>VBP</td><td>Ours</td></tr><tr><td>DeiT-B</td><td>81.98</td><td>69.01</td><td>68.43</td><td>67.94</td><td>73.09</td></tr><tr><td>Swin-S</td><td>83.32</td><td>47.10</td><td>71.08</td><td>65.67</td><td>73.00</td></tr></table>

![](images/bd99b95c4cc65a968573b23961ec388c035265c3f6945cd42380ad7d6151f7fc.jpg)

![](images/f102fca621435b6b6299563d4324c4631d37774ccea09f1db9fb32636adcd924.jpg)

![](images/6ec42a5db61e2a63a6a0c239ca3bb7e82dcd8b0650e06b2878aff67c571211b7.jpg)

![](images/626952d5384b829bcbd173ef99ba5efe483891a0774ec610d15f19a338c2e409.jpg)  
Figure 8. Spearman correlation plot of DeiT-T (top left), DeiT-S (top right), Swin-T (bottom left), Swin-B (bottom right) between denoised and raw variance score rank.

![](images/d3d1b3685ab3c67e999389bead83f9e6cd542424993073044d5bba8af85f14a6.jpg)  
(a) DeiT-B Layer 1

![](images/48bbd9855ef6283d6a54200e88da34e65b550e10b8d5b5bd11f671c01d8d6e0d.jpg)  
(b) DeiT-B Layer 2

![](images/084338e68f2b6320d49c7a4e9131e5d68feb1cd7aa6bb4703d945d644fe516b7.jpg)  
(c) DeiT-B Layer 3

![](images/05df574c905e2b2758e6445275d151c79b1b1042082242f26558404f63a4b7eb.jpg)  
(d) DeiT-B Layer 4

![](images/bc3e04d2e4d75c674bc5ed764b8c8f62db42b2e18fb322ec6381a0e3550ba90f.jpg)  
(e) DeiT-B Layer 5

![](images/24e634b0cc010e55c35271965e8f5032cdc10ef56c0f026e12346f63a52c041e.jpg)  
(f) DeiT-B Layer 6

![](images/57ecb69a10350adc80b1a99626d649fef50d6add2810f15ba07b241b43bbd45d.jpg)  
(g) DeiT-B Layer 7

![](images/2e773ac9b9fdacf98d54a45ee1a0a04c0cebdeba4c3170623f9c42490cc06553.jpg)  
(h) DeiT-B Layer 8

![](images/5fcd01405a7b35dfb2b74d45e9fa44b57ea10e280ba9ce0237c9f64ab5a97d06.jpg)  
(i) DeiT-B Layer 9

![](images/b709efd481bd57a41926472b057a08a5a9322da5dd42a9cb32639b8fe45d74d4.jpg)  
(j) DeiT-B Layer 10

![](images/c029486075ee011e7bc403ab7e3bb8463424eb55402243f6771d3637dce8a4d5.jpg)  
(k) DeiT-B Layer 11

![](images/a7fcbcb754f0a365c72e4f2733be32aacdf427a13e2bbcec30e217a8d9ad3dfc.jpg)  
(l) DeiT-B Layer 12  
Figure 9. Empirical spectral density for all layers of DeiT-B fitted with the Marchenko-Pastur distribution.

![](images/605147dfa64098950297fe3d26a32fc589554c05fb3069ed5a666cabcafb7552.jpg)  
Eigenvalue λ  
(a) Swin-S Layer 1

![](images/4dd34142eb8617bbeeaf048f7f1a5107671efd9185b35333b8b29daebcced7e8.jpg)  
(b) Swin-S Layer 2

![](images/a247f74598b314f6df1b191dabe1fe7e802036503c7e61e712f1beee05bcc2d8.jpg)  
(c) Swin-S Layer 3

![](images/ec78df8645847fb5eff47f1d3ca905ef0f0abb54a8375c8dbbf027e95bdb7e07.jpg)  
Eigenvalue λ  
(d) Swin-S Layer 4

![](images/c2e21b6734d6f7a4339c11a64bc72adc12b03a82103882c30f239e4c911d86e7.jpg)  
(e) Swin-S Layer 5

![](images/93c4d1abdac3df7a1d657da145ab2e485ac7319a6d68a3760cbc55a41a1ab00b.jpg)  
(f) Swin-S Layer 6

![](images/90d6e77587f255941b70d9eec0a5028ba1fb67f2145351f182249d73f8208cf0.jpg)  
(g) Swin-S Layer 7

![](images/3494c2928092a7923c63ca5d010f3aa5d87a129c68afb848432c250ba2251582.jpg)  
Eigenvalue λ  
(h) Swin-S Layer 8

![](images/785e6f388d17c49678eed55cf51756d99463063a6d6ce5ce7df4a99279357148.jpg)  
(i) Swin-S Layer 9

![](images/28187d45f8fc83eb59efe0515ee09e98f940bde2c2618d62c8258de3ede5dc16.jpg)  
(j) Swin-S Layer 10

![](images/68a7733f91d1fb94dd56a39359df580cb80b9f2ffa393212810ade4f48720593.jpg)  
(k) Swin-S Layer 11

![](images/6d0d6a643c8928351ef7215eeb92ccf608997e118c76996a0ca509256fc66661.jpg)  
(l) Swin-S Layer 12  
Figure 10. Empirical spectral density for layers 1-12 of Swin-S fitted with the Marchenko-Pastur distribution.

Eigenvalue λ  
![](images/138ae8adbb3af0fb63941f2211eca89cb23bdf9aa3fcfc28d5a4a4c826408da6.jpg)  
Eigenvalue λ  
(m) Swin-S Layer 13

![](images/cc5829503a466b407327640f992b7cff0d1e54488fb4b5cb66443039927a8b44.jpg)  
(n) Swin-S Layer 14

![](images/2cf247cb455287c9d7cd50ef9463ed4da5aa7acf9c39d8a1ca8aa091730f1ad5.jpg)  
(o) Swin-S Layer 15

![](images/c61626661fc222fb8e9d4e45806cfb5f60b5e63ea573db6c4c9bd8e27d5d3aa3.jpg)  
Eigenvalue λ  
(p) Swin-S Layer 16

![](images/f299cc18e9fb9570bee4f61f94cc5b1596dcd515f39700503889629db68785f3.jpg)  
(q) Swin-S Layer 17

![](images/ab6440dfe3f916caf4d70ccd9484ceac464fb80aed82aaef3cb5e69374e72160.jpg)  
(r) Swin-S Layer 18

![](images/7eb82b378eebe5f66822f22f029bcc8f7f72fd2e794b9fd9a5b2644a9b4227d8.jpg)  
(s) Swin-S Layer 19

![](images/6aca028c12830eaa6428991991390e9d3bbfdb943676085ba5d9b57e4fc9d7f2.jpg)  
(t) Swin-S Layer 20

![](images/f8df9fd0b2d5369bfab09891800a173114d922953e0c1307e094984ab02b4e0a.jpg)  
(u) Swin-S Layer 21

![](images/2884a13acab2a141ed0475dccb73815c8c4f813fcc941095db4af9f46d5a2b79.jpg)  
(v) Swin-S Layer 22

![](images/71282e9707b50640bd3ed09c99c866255d027273d49395d27e3faac7e676ddce.jpg)  
(w) Swin-S Layer 23

![](images/087800a125d40384c0d03d31a4bbca1393ad01e5dc249aabed6ca4719b1980c2.jpg)  
(x) Swin-S Layer 24  
Figure 11. Empirical spectral density for layers 13-24 of Swin-S fitted with the Marchenko-Pastur distribution.