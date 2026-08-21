# Flow Matching-Based PET Image Reconstruction

Fumio Hashimoto, Ziqian Huang, Tatsuya Yokota, and Kuang Gong

Abstract—Generative models have shown strong potential for positron emission tomography (PET) image reconstruction. Although diffusion model-based reconstruction methods have demonstrated promising performance, they often require many reverse sampling steps with data-consistency updates incorporated into the sampling process. Flow matching offers an attractive alternative because it can directly estimate clean images from intermediate states, allowing data-consistency refinement to be separated from flow propagation. In this work, we proposed flow matching-based PET image reconstruction methods. We first established PET-FlowDPS by incorporating Poisson likelihood guidance with an expectation-maximization (EM)- based preconditioner into the FlowDPS framework. We then proposed a model-based PET reconstruction method that used a pretrained flow matching model as a prior, in which the flowbased prior, PET data refinement, and stochastic propagation were interpreted within an approximate Bayesian framework. Experimental results using [<sup>18</sup>F]FDG brain PET datasets showed that the proposed method achieved better bias-variance trade-offs across different dose levels compared with other reference methods. These results demonstrated the potential of flow matching as a generative prior for quantitative PET image reconstruction.

Index Terms—Positron emission tomography (PET), Image reconstruction, Flow matching

## I. INTRODUCTION

OSITRON emission tomography (PET) enables quantitative in vivo imaging of radiotracer distributions, making it a valuable molecular imaging modality in oncology, cardiology, and neurology. However, PET image quality is fundamentally constrained by limited counting statistics and various physical degradation effects, which compromises its lesion detectability and quantitative accuracy. Although iterative reconstruction methods, such as maximumlikelihood expectation-maximization (MLEM) and orderedsubsets expectation-maximization (OSEM), have been widely used for PET image reconstruction with explicit modeling of Poisson noise statistics and the PET system response [1], [2], the resulting images often exhibit excessive noise under low-count conditions. To address this limitation, maximum a posteriori (MAP)-based reconstruction methods have been developed to incorporate prior information, such as spatial

smoothness and anatomical information, to further improve PET image quality [3]–[7].

Recent advances in deep learning have led to growing interest in data-driven approaches for PET image reconstruction [8]–[10]. Early deep learning-based methods mainly used convolutional neural networks (CNNs) as learned image priors within PET image reconstruction. For example, CNNs have been used to parameterize PET images within iterative reconstruction frameworks [11] and to provide learned regularization in unrolled reconstruction frameworks [12]. Although these approaches demonstrate the potential of CNN-based priors to improve PET image quality, they rely on deterministic mappings rather than explicitly modeling the underlying image distribution. When multiple plausible high-quality solutions exist, such deterministic mappings may produce averaged estimates, potentially suppressing fine structural details and leading to oversmoothed images [13], [14].

Diffusion models [15], [16], which can learn image distributions and act as generative priors, have been investigated for various PET imaging tasks, including image denoising [17]– [20], synthesis [21], [22], and attenuation correction [23], [24]. For PET image reconstruction, Singh et al. [25] demonstrated the potential of score-based reconstruction using diffusion posterior sampling (DPS) [26] and decomposed diffusion sampling (DDS) [27]. Subsequent studies extended diffusion-based PET reconstruction to fully 3D imaging [28], joint activity and attenuation estimation [29], [30], and out-of-distribution adaptation [31], [32].

Despite these advances, diffusion-based PET image reconstruction remains computationally demanding because both neural network evaluations and data-consistency updates must be repeated over many reverse sampling steps. Its performance can also be sensitive to the noise schedule, the number of reverse sampling steps, and the strength and timing of the data-consistency updates. In DPS, computing the likelihood gradient with respect to an intermediate noisy image requires differentiating through the denoised estimate and thus backpropagating through the neural network at each sampling step. This increases computational and memory costs and may result in numerically sensitive updates [27]. Although PET-DDS [25] and likelihood scheduling [28] improve the efficiency and robustness of data-consistency updates, the generative prior and data-consistency refinement remain coupled within the prescribed reverse diffusion process.

Flow matching offers an attractive alternative to diffusion models by learning a time-dependent velocity field that transports samples from a source distribution to the target distribution along a prescribed probability path [33]. Its relatively straight transport trajectories can reduce the number of sampling steps required for image generation. Under a linear probability path, the velocity predicted at an intermediate point can be used to directly estimate the destination image. This property provides a flexible framework for incorporating data consistency into inverse problems [34]–[37]. Recently, Flower [35] provides an approximate Bayesian framework that separates destination estimation, data-consistency refinement, and stochastic propagation. However, its data-consistency refinement assumes a Gaussian observation model and therefore does not directly account for the Poisson statistics of PET measurements.

In this study, we present flow matching-based PET image reconstruction methods. First, we establish PET-FlowDPS as a baseline by extending flow-driven posterior sampling (FlowDPS) [34] to PET reconstruction using Poisson likelihood guidance with preconditioned gradients. Second, inspired by the Flower framework [35], we propose a model-based PET reconstruction method that uses a pretrained flow-matching model as a learned image prior. The proposed algorithm iteratively performs three steps: (1) flow-based destination estimation to obtain a clean image estimate, (2) penalized PET image reconstruction refinement to enforce data consistency, and (3) flow-based propagation to update the image to the next time step. These three steps can be interpreted within an approximate Bayesian framework. Separating data-consistency refinement from flow propagation allows established modelbased PET reconstruction algorithms to be directly incorporated into the refinement step without modifying the pretrained flow model, thereby enabling stable data-consistency enforcement for quantitative PET reconstruction.

## II. BACKGROUND

## A. PET image reconstruction

PET image acquisition can be described using the following linear model

$$
\bar { \pmb { y } } = \pmb { A x } + \pmb { b } ,\tag{1}
$$

where $\bar { \pmb y } \in \mathbb { R } ^ { M }$ denotes the expected projection data, $\pmb { x } \in \mathbb { R } ^ { N }$ represents the unknown radiotracer distribution, $\pmb { A } \in \mathbb { R } ^ { M \times N }$ is the system matrix, and $\pmb { b } \in \mathbb { R } ^ { M }$ represents mean of the random and scatter events. M and N denote the number of lines of response (LOR) and image voxels, respectively. Assuming the measured projection data $\pmb { y } \in \mathbb { R } ^ { M }$ follows the Poisson distribution with mean ${ \bar { \mathbf { y } } } ,$ the log-likelihood function can be written as

$$
L ( \pmb { y } \mid \pmb { x } ) = \sum _ { i = 1 } ^ { M } \left\{ y _ { i } \log \left( [ \pmb { A } \pmb { x } ] _ { i } + b _ { i } \right) - \left( [ \pmb { A } \pmb { x } ] _ { i } + b _ { i } \right) \right\} .\tag{2}
$$

The MLEM algorithm estimates x by maximizing (2) using the following iterative update for voxel $j$

$$
x _ { j } ^ { ( n + 1 ) } = \frac { x _ { j } ^ { ( n ) } } { S _ { j } } \sum _ { i = 1 } ^ { M } A _ { i j } \frac { y _ { i } } { [ A { \pmb x } ^ { ( n ) } ] _ { i } + b _ { i } } , \quad S _ { j } = \sum _ { i = 1 } ^ { M } A _ { i j } .\tag{3}
$$

This update rule is stable because it always satisfies $L ( \pmb { y } | \pmb { x } ^ { ( n + \bar { 1 } ) } ) \ge L ( \pmb { y } | \pmb { x } ^ { ( n ) } )$ .

## B. Flow matching

Flow matching learns a continuous-time velocity field that transports samples from a simple source distribution to a target image distribution [33], [38]. Let $\pmb { x } _ { 0 } \sim p _ { 0 } ( \pmb { x } _ { 0 } )$ denote a sample from the source distribution, typically a standard Gaussian distribution, and let $\pmb { x } _ { 1 } \sim p _ { 1 } ( \pmb { x } _ { 1 } )$ denote a sample from the target image distribution. Using a linear probability path, an intermediate sample at time $t \in [ 0 , 1 ]$ is defined as

$$
\begin{array} { r } { { \pmb x } _ { t } = ( 1 - t ) { \pmb x } _ { 0 } + t { \pmb x } _ { 1 } , } \end{array}\tag{4}
$$

and the corresponding conditional velocity along this path is

$$
{ \pmb v } _ { t } ( { \pmb x } _ { t } \mid { \pmb x } _ { 0 } , { \pmb x } _ { 1 } ) = \frac { d { \pmb x } _ { t } } { d t } = { \pmb x } _ { 1 } - { \pmb x } _ { 0 } .\tag{5}
$$

A neural network ${ \pmb v } _ { \pmb \theta } ( { \pmb x } _ { t } , t )$ parameterized by θ is trained to approximate the velocity field by minimizing the conditional flow-matching loss

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C F M } } ( \pmb { \theta } ) = \mathbb { E } _ { t , \pmb { x } _ { 0 } , \pmb { x } _ { 1 } } \left[ \| \pmb { v } _ { \pmb { \theta } } ( \pmb { x } _ { t } , t ) - ( \pmb { x } _ { 1 } - \pmb { x } _ { 0 } ) \| _ { 2 } ^ { 2 } \right] , } \end{array}\tag{6}
$$

where $\pmb { x } _ { t } = ( 1 - t ) \pmb { x } _ { 0 } + t \pmb { x } _ { 1 } , t \sim \mathcal { U } ( 0 , 1 ) , \pmb { x } _ { 0 } \sim p _ { 0 } ( \pmb { x } _ { 0 } )$ , and $\pmb { x } _ { 1 } \sim p _ { 1 } ( \pmb { x } _ { 1 } )$ . After training, samples from the target distribution can be generated by solving the ordinary differential equation (ODE)

$$
\frac { d \pmb { x } _ { t } } { d t } = \pmb { v } _ { \pmb { \theta } } ( \pmb { x } _ { t } , t ) , \qquad \pmb { x } _ { 0 } \sim p _ { 0 } ( \pmb { x } _ { 0 } ) ,\tag{7}
$$

from $t = 0 \ \mathrm { t o } \ t = 1$ using a numerical ODE solver.

Under the linear probability path in (4), an estimate of the destination image $\mathbf { \delta x } _ { 1 }$ can be obtained directly from an intermediate state $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ as

$$
\hat { \pmb { x } } _ { 1 } ( \pmb { x } _ { t } , t ) = \pmb { x } _ { t } + ( 1 - t ) \pmb { v } _ { \pmb { \theta } } ( \pmb { x } _ { t } , t ) .\tag{8}
$$

This explicit destination-image estimate enables dataconsistency refinement to be applied before advancing the intermediate state along the flow trajectory.

## III. METHODOLOGY

A schematic overview of PET-FlowDPS and the proposed model-based PET reconstruction with a flow-based prior is shown in Fig. 1. Both methods refine the flow-based destination estimate using the measured PET data. PET-FlowDPS applies a data-consistency update using the gradient of the log-likelihood, whereas the proposed method performs a MAP reconstruction that balances data fidelity and the flow-based prior.

## A. FlowDPS for PET reconstruction

FlowDPS is a posterior sampling solver for inverse problems that integrates the likelihood gradient and stochastic noise into a flow matching framework [34]. For the linear conditional flow, the marginal velocity is

$$
\pmb { v } _ { t } ( \pmb { x } _ { t } ) = \frac { \pmb { x } _ { t } } { t } + \frac { 1 - t } { t } \nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { x } _ { t } ) .\tag{9}
$$

The corresponding posterior marginal velocity is

$$
\mathbf { \boldsymbol { v } } _ { t } ( \mathbf { \boldsymbol { x } } _ { t } \mid \mathbf { \boldsymbol { y } } ) = \frac { \mathbf { \boldsymbol { x } } _ { t } } { t } + \frac { 1 - t } { t } \nabla _ { \mathbf { \boldsymbol { x } } _ { t } } \log p _ { t } ( \mathbf { \boldsymbol { x } } _ { t } \mid \mathbf { \boldsymbol { y } } ) .\tag{10}
$$

![](images/c6fc57e4f6121c060e44f660c753eb49cb7aa0adc814550402c21ff3bb5dfc6e.jpg)

![](images/52d64f17b54855a80fb0f1dfbbe5b2d2cc47ec391af50c55533e4818e84169ad.jpg)

![](images/76f029b07f2975618fd0100a59f6ff8423c6d26be34e0f806ba2d9d63bab444f.jpg)

![](images/122cbc475e0b51c175e7e78e6f580a29a9a68acabb2fe4a6ed746937b38725ab.jpg)  
(1) Destination estimation

![](images/bd4784143971dc977b3aeb9219d08504a40d8b25e99a53c5d8703520aeaf86de.jpg)  
(2) PET-data refinement

![](images/5bcf7d6ed19f3cc10b3e06ad4c0dbe0cab0c55432d66e967f411c9bbc755a756.jpg)  
(3) Propagation  
Fig. 1. Overview of (a) PET-FlowDPS and (b) the proposed model-based PET reconstruction with a flow-based prior. In step (1), both methods estimate a destination image using the pretrained flow model. In step (2), (a) PET-FlowDPS applies a data-consistency update using the gradient of the log-likelihood, and (b) the proposed method performs a MAP reconstruction that balances data fidelity and the conditional flow-based prior. In step (3), the refined destination estimate is propagated to the next flow time point using the respective stochastic propagation rules. The purple contours in step (2) represent the PET log likelihood, and the blue contours in (b-2) represent the conditional flow-based prior approximated by a Gaussian distribution.

Using Bayes’ rule, (10) can be rewritten as

$$
\begin{array} { l } { { \displaystyle { \pmb v } _ { t } ( { \pmb x } _ { t } \mid { \pmb y } ) = { \pmb v } _ { t } ( { \pmb x } _ { t } ) + \frac { 1 - t } { t } \nabla _ { { \pmb x } _ { t } } \log p _ { t } ( { \pmb y } \mid { \pmb x } _ { t } ) } } \\ { { \displaystyle ~ \approx { \pmb v } _ { \theta } ( { \pmb x } _ { t } , t ) + \frac { 1 - t } { t } \nabla _ { { \pmb x } _ { t } } \log p _ { t } ( { \pmb y } \mid { \pmb x } _ { t } ) } . } \end{array}\tag{11}
$$

Since $p _ { t } ( \pmb { y } \mid \pmb { x } _ { t } )$ is not tractable, DPS [26] and FlowDPS [34] apply the following approximations

$$
\begin{array} { r l r } & { } & { \nabla _ { \pmb { x } _ { t } } \log p _ { t } ( \pmb { y } \mid \pmb { x } _ { t } ) \approx \nabla _ { \pmb { x } _ { t } } \log p \left( \pmb { y } \mid \hat { \pmb { x } } _ { 1 } ( \pmb { x } _ { t } , t ) \right) , } \\ & { } & { \approx \displaystyle \frac { 1 } { t } \nabla _ { \hat { \pmb { x } } _ { 1 } } \log p \left( \pmb { y } \mid \hat { \pmb { x } } _ { 1 } ( \pmb { x } _ { t } , t ) \right) . } \end{array}\tag{DPS}
$$

(FlowDPS)

(12)

Thus, the unconditional flow can be guided toward the posterior distribution by incorporating the likelihood gradient. Although DPS requires backpropagation through the flow model to compute the Jacobian of the destination estimator, FlowDPS approximates this gradient to bypass the Jacobian computation and applies the likelihood gradient directly to the destination image estimate.

To apply the likelihood approximation in (12), FlowDPS uses the flow-based source and destination estimates. Using (4) and (5), these estimates at each time point $t _ { k }$ can be expressed by the flow-version of Tweedie’s formula [34] as

$$
\begin{array} { r } { \hat { \pmb { x } } _ { 0 , k } = \pmb { x } _ { t _ { k } } - t _ { k } \pmb { v } _ { \pmb { \theta } } ( \pmb { x } _ { t _ { k } } , t _ { k } ) , } \end{array}\tag{13}
$$

$$
\hat { \pmb { x } } _ { 1 , k } = \pmb { x } _ { t _ { k } } + ( 1 - t _ { k } ) \pmb { v } _ { \pmb { \theta } } ( \pmb { x } _ { t _ { k } } , t _ { k } ) .\tag{14}
$$

The destination estimate is then corrected using the likelihood gradient as

$$
\begin{array} { r } { \tilde { { \pmb x } } _ { 1 , k } = \hat { { \pmb x } } _ { 1 , k } + \lambda _ { t _ { k } } \nabla _ { \hat { { \pmb x } } _ { 1 , k } } \log p \left( \pmb y \mid \hat { { \pmb x } } _ { 1 , k } \right) , } \end{array}\tag{15}
$$

where $\lambda _ { t _ { k } }$ is the time-dependent guidance strength with the factor $1 / t _ { k }$ in (12) absorbed into $\lambda _ { t _ { k } }$ . After this correction, the sample at the next time point is obtained as

$$
\begin{array} { r } { \tilde { \mathbf { x } } _ { 0 , k + 1 } = \sqrt { t _ { k + 1 } } \mathbf { \epsilon } _ { k } + \sqrt { 1 - t _ { k + 1 } } \hat { \mathbf { x } } _ { 0 , k } , } \end{array}\tag{16}
$$

$$
\begin{array} { r } { \pmb { x } _ { t _ { k + 1 } } = ( 1 - t _ { k + 1 } ) \tilde { \pmb { x } } _ { 0 , k + 1 } + t _ { k + 1 } \tilde { \pmb { x } } _ { 1 , k } , } \end{array}\tag{17}
$$

where $\epsilon _ { k } \sim \mathcal { N } ( 0 , I )$

For PET reconstruction, the preconditioned gradient of the log-likelihood is calculated by

$$
\tilde { \nabla } _ { x } L ( y \mid x ) = \frac { x } { S } \left\{ A ^ { \top } \left[ \frac { y } { A x + b } \right] - S \right\} ,\tag{18}
$$

where ${ \pmb x } / S$ is a preconditioning term for stability. Using this gradient, (15) becomes

$$
\begin{array} { r } { \tilde { \pmb { x } } _ { 1 , k } = \hat { \pmb { x } } _ { 1 , k } + \lambda \tilde { \nabla } _ { \hat { \pmb { x } } _ { 1 , k } } L ( \pmb { y } \mid \hat { \pmb { x } } _ { 1 , k } ) , } \end{array}\tag{19}
$$

where λ is the constant guidance strength. The overall algorithm of the PET-FlowDPS reconstruction is shown in Algorithm 1.

Although FlowDPS derives an additional likelihoodgradient term for the posterior velocity field, its practical implementation differs from this theoretical formulation. In practice, the guidance coefficient is typically chosen to be a relatively small value, and multiple gradient updates are performed [34]. In addition, the stochastic propagation step is inherited from the sampling strategy of DPS [26], rather than being directly derived from the flow-based probabilistic formulation. As a consequence, the probabilistic interpretation of the trade-off between the pretrained flow-based prior and the dataconsistency term becomes less explicit. The PET-FlowDPS algorithm improves the stability of the data-consistency update by incorporating an EM-based preconditioner. As a result, likelihood refinement can be performed stably using a single update. However, this modification still does not provide a probabilistic formulation of the overall reconstruction process.

Algorithm 1 PET-FlowDPS reconstruction   
Require: Flow time points $0 = t _ { 0 } < t _ { 1 } < \cdot \cdot \cdot < t _ { K } = 1$ , flow   
model ${ \boldsymbol { v } } _ { \boldsymbol { \theta } } ,$ , measured data $^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } \mathbf { \Lambda } ^ { \mathrm { \Lambda } } \mathbf { \Lambda } \mathbf { \Lambda } ^ { \mathrm { \Lambda } } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathrm { \Lambda } ^ { \mathrm { \Lambda } }$ , background $^ { b , }$ and guidance   
strength λ   
1: $\pmb { x } _ { t _ { 0 } } \sim \mathcal { N } ( \mathbf { 0 } , I )$   
2: for $k = 0$ to $K - 1$ do   
3: $\hat { \mathbf { x } } _ { 0 , k } = \mathbf { x } _ { t _ { k } } - t _ { k } v _ { \theta } ( \mathbf { x } _ { t _ { k } } , t _ { k } )$   
4: $\hat { \pmb { x } } _ { 1 , k } = \pmb { x } _ { t _ { k } } + ( 1 - t _ { k } ) \pmb { v } _ { \pmb { \theta } } ( \pmb { x } _ { t _ { k } } , t _ { k } )$   
5: $\tilde { \nabla } _ { \hat { \pmb { x } } _ { 1 , k } } L ( \pmb { y } | \hat { \pmb { x } } _ { 1 , k } ) = \frac { \hat { \pmb { x } } _ { 1 , k } } { S } \left\{ A ^ { \top } \left[ \frac { \pmb { y } } { A \hat { \pmb { x } } _ { 1 , k } + b } \right] - S \right\}$   
6: $\tilde { \mathbf { x } } _ { 1 , k } = \hat { \mathbf { x } } _ { 1 , k } + \lambda \tilde { \nabla } _ { \hat { \mathbf { x } } _ { 1 , k } } L ( \pmb { y } \mid \hat { \mathbf { x } } _ { 1 , k } )$   
7: $\epsilon _ { k } \sim \mathcal { N } ( 0 , I )$   
8: $\begin{array} { r } { \tilde { \mathbf { x } } _ { 0 , k + 1 } = \sqrt { t _ { k + 1 } } \mathbf { \epsilon } _ { k } + \sqrt { 1 - t _ { k + 1 } } \hat { \mathbf { x } } _ { 0 , k } } \end{array}$   
9: $\pmb { x } _ { t _ { k + 1 } } = ( 1 - t _ { k + 1 } ) \tilde { \pmb { x } } _ { 0 , k + 1 } + t _ { k + 1 } \tilde { \pmb { x } } _ { 1 , k }$   
10: end for   
11: return $\hat { \pmb { x } } = \pmb { x } _ { t _ { K } }$

Algorithm 2 Model-based PET reconstruction using flow   
matching   
Require: Flow time points $0 = t _ { 0 } < t _ { 1 } < \cdots < t _ { K } = 1 ,$   
flow model ${ \pmb v } _ { \pmb { \theta } } .$ , measured data y, background $^ { b , }$ penalty   
strength $\beta ,$ and inner iteration number N   
1: $\pmb { x } _ { t _ { 0 } } \sim \mathcal { N } ( \mathbf { 0 } , I )$   
2: for $k = 0$ to $K - 1$ do   
3: $\hat { \pmb { x } } _ { 1 , k } = \pmb { x } _ { t _ { k } } + ( 1 - t _ { k } ) \pmb { v } _ { \pmb { \theta } } ( \pmb { x } _ { t _ { k } } , t _ { k } )$   
4: $\pmb { x } ^ { ( 0 ) } = [ \hat { \pmb { x } } _ { 1 , k } ] _ { + }$   
5: for $n = 0$ to $N , - 1$ do   
6: $\begin{array} { r } { \hat { x } _ { \mathrm { E M } , j } ^ { ( n + 1 ) } = \frac { x _ { j } ^ { \ast \ast \prime } } { S _ { j } } \sum _ { i } A _ { i j } \frac { y _ { i } } { [ A \pmb { x } _ { , } ^ { ( n ) } ] _ { i } + b _ { i } } } \end{array}$   
$x _ { j } ^ { ( n + 1 ) } = \frac { 1 } { 2 } \left( \hat { x } _ { 1 , j , k } - \overset { { \overset { } { \cdot } } } { \beta } \right)$   
7:   
$+ \frac { 1 } { 2 } \sqrt { \left( \hat { x } _ { 1 , j , k } - \frac { S _ { j } } { \beta } \right) ^ { 2 } + 4 \hat { x } _ { \mathrm { E M } , j } ^ { ( n + 1 ) } } \frac { S _ { j } } { \beta }$   
8: end for   
9: $\tilde { \pmb { x } } _ { 1 , k } = \pmb { x } ^ { ( N ) }$   
10: $\epsilon _ { k } \sim \mathcal { N } ( 0 , I )$   
11: $\pmb { x } _ { t _ { k + 1 } } = ( 1 - t _ { k + 1 } ) \pmb { \epsilon } _ { k } + t _ { k + 1 } \tilde { \pmb { x } } _ { 1 , k }$   
12: end for   
13: return $\hat { \pmb { x } } = \pmb { x } _ { t _ { K } }$

## B. Model-based PET reconstruction using flow matching

In this section, inspired by the Flower framework, we propose a model-based PET reconstruction algorithm designed to improve stability and interpretability compared with PET-FlowDPS. The proposed reconstruction method formulates the flow-based prior, PET-data refinement, and stochastic propagation within an approximate Bayesian framework.

1) Theoretical background: We first describe the probabilistic modeling structure used to derive the proposed algorithm. The PET data observation process is defined by the following probabilistic forward model

$$
\mathbf { \boldsymbol { x } } _ { 0 } \sim p _ { 0 } ( \mathbf { \boldsymbol { x } } _ { 0 } ) = \mathcal { N } ( \mathbf { \boldsymbol { 0 } } , I ) ,\tag{20}
$$

$$
\pmb { x } _ { 1 } = \Phi _ { 1 } ( \pmb { x } _ { 0 } ) \sim p _ { 1 } ( \pmb { x } _ { 1 } ) ,\tag{21}
$$

$$
\pmb { y } \sim \mathcal { P } ( \pmb { y } | A \pmb { x } _ { 1 } + \pmb { b } ) ,\tag{22}
$$

where $\Phi _ { 1 }$ is a transportation function from the source domain to the target domain and $\mathcal { P }$ stands for Poisson distribution. The variables are generated sequentially as ${ \pmb x } _ { 0 }  { \pmb x } _ { 1 }  { \pmb y }$

Our goal is to sample PET images from the posterior distribution $p ( \pmb { x } _ { 1 } | \pmb { y } )$ . We discretize the continuous time interval $t ~ \in ~ [ 0 , 1 ]$ into $K + 1$ time points and perform sequential sampling as follows:

$$
\begin{array} { r } { \pmb { x } _ { t _ { 0 } } \sim p ( \pmb { x } _ { t _ { 0 } } ) , } \end{array}
$$

$$
\begin{array} { r } { \pmb { x } _ { t _ { 1 } } \sim p ( \pmb { x } _ { t _ { 1 } } | \pmb { x } _ { t _ { 0 } } , \pmb { y } ) , } \end{array}\tag{23}
$$

(24)

$$
\pmb { x } _ { t _ { K } } \sim p ( \pmb { x } _ { t _ { K } } | \pmb { x } _ { t _ { K - 1 } } , \pmb { y } ) ,\tag{25}
$$

where $0 = t _ { 0 } < t _ { 1 } < \cdots < t _ { K - 1 } < t _ { K } = 1$ . Since $\scriptstyle { \pmb x } _ { 0 }$ can be sampled directly from $\mathcal { N } ( \mathbf { 0 } , \pmb { I } )$ , the remaining task is to sample

$$
\begin{array} { r } { \pmb { x } _ { t _ { k + 1 } } \sim p ( \pmb { x } _ { t _ { k + 1 } } | \pmb { x } _ { t _ { k } } , \pmb { y } ) , } \end{array}\tag{26}
$$

for any $k \in \{ 0 , 1 , . . . , K { - } 1 \}$ . Direct sampling from this condi tional distribution is difficult, so we introduce an approximate model for $p ( \pmb { x } _ { t _ { k + 1 } } | \pmb { x } _ { t _ { k } } , \pmb { y } )$

2) Approximate probabilistic models and sampling decomposition: Following Flower [35], we use the following approximations to decompose the conditional sampling process:

$$
\tilde { p } ( \pmb { x } _ { 1 } | \pmb { x } _ { t _ { k } } ) = \mathcal { N } ( \hat { \pmb { x } } _ { 1 } ( \pmb { x } _ { t _ { k } } , t _ { k } ) , \beta ^ { - 1 } \pmb { I } ) ,\tag{27}
$$

$$
\begin{array} { c } { \tilde { p } ( { \pmb x } _ { t _ { k + 1 } } | { \pmb x } _ { 1 } , { \pmb x } _ { t _ { k } } ) = p ( { \pmb x } _ { t _ { k + 1 } } | { \pmb x } _ { 1 } ) } \\ { = \mathcal N ( t _ { k + 1 } { \pmb x } _ { 1 } , ( 1 - t _ { k + 1 } ) ^ { 2 } { \pmb I } ) . } \end{array}\tag{28}
$$

The first approximation in (27) assumes that, given $\mathbf { \Delta } _ { \mathbf { x } _ { t _ { k } } }$ , the destination image $\scriptstyle { \pmb x } _ { 1 }$ follows a Gaussian distribution centered at the flow-based estimate $\hat { \pmb { x } } _ { 1 } ( { \pmb { x } } _ { t _ { k } } , t _ { k } )$ . The parameter $\beta$ represents the precision of this distribution and controls the strength of the regularization imposed by flow matching. The second approximation in (28) replaces the transition conditioned on both $\mathbf { \Delta } _ { \mathbf { x } _ { t _ { k } } }$ and $\scriptstyle { \mathbf { { \vec { x } } } } _ { 1 }$ with a stochastic transition conditioned only on $\scriptstyle { \mathbf { { \vec { x } } } } _ { 1 }$ . This formulation provides a probabilistic interpretation of the stochastic propagation step.

Next, we consider the conditional distribution $p ( \pmb { x } _ { t _ { k + 1 } } \quad |$ $\boldsymbol { x } _ { t _ { k } } , y )$ in (26). By marginalizing over $\scriptstyle { \mathbf { { \vec { x } } } } _ { 1 }$ , applying the conditional independence implied by the Markov property, i.e. $p ( \pmb { x } _ { t _ { k + 1 } } | \pmb { x } _ { 1 } , \pmb { x } _ { t _ { k } } , \pmb { y } ) = p ( \pmb { x } _ { t _ { k + 1 } } | \pmb { x } _ { 1 } , \pmb { x } _ { t _ { k } } )$ , and subsequently applying the approximations in (27) and (28), we obtain

$$
\begin{array} { r l } & { p ( { \pmb x } _ { t _ { k + 1 } } | { \pmb x } _ { t _ { k } } , { \pmb y } ) } \\ & { \qquad = \displaystyle \int p ( { \pmb x } _ { t _ { k + 1 } } | { \pmb x } _ { 1 } , { \pmb x } _ { t _ { k } } , { \pmb y } ) p ( { \pmb x } _ { 1 } | { \pmb x } _ { t _ { k } } , { \pmb y } ) \mathrm { d } { \pmb x } _ { 1 } , } \\ & { \qquad \approx \displaystyle \int \tilde { p } ( { \pmb x } _ { t _ { k + 1 } } | { \pmb x } _ { 1 } , { \pmb x } _ { t _ { k } } ) \tilde { p } ( { \pmb x } _ { 1 } | { \pmb x } _ { t _ { k } } , { \pmb y } ) \mathrm { d } { \pmb x } _ { 1 } , } \end{array}\tag{29}
$$

(30)

where, by Bayes’ rule and conditional independence implied by the Markov property, i.e. $p ( \pmb { y } | \pmb { x } _ { 1 } , \pmb { x } _ { t _ { k } } ) = p ( \pmb { y } | \pmb { x } _ { 1 } )$ ,

$$
\begin{array} { r l } & { \tilde { p } ( { \pmb x } _ { 1 } | { \pmb x } _ { t _ { k } } , { \pmb y } ) \propto p ( { \pmb y } | { \pmb x } _ { 1 } , { \pmb x } _ { t _ { k } } ) \tilde { p } ( { \pmb x } _ { 1 } | { \pmb x } _ { t _ { k } } ) = p ( { \pmb y } | { \pmb x } _ { 1 } ) \tilde { p } ( { \pmb x } _ { 1 } | { \pmb x } _ { t _ { k } } ) } \\ & { \qquad = \mathcal { P } ( { \pmb y } | A { \pmb x } _ { 1 } + b ) \mathcal { N } ( { \pmb x } _ { 1 } | \hat { { \pmb x } } _ { 1 } ( { \pmb x } _ { t _ { k } } , t _ { k } ) , \beta ^ { - 1 } { \pmb I } ) . \qquad ( 3 1 ) } \end{array}
$$

Equation (30) represents the approximate conditional distribution of $\boldsymbol { x } _ { t _ { k + 1 } }$ as a marginal distribution over $\scriptstyle { \mathbf { { \vec { x } } } } _ { 1 }$ . Therefore, rather than sampling directly from this marginal distribution, we employ a hierarchical sampling. We first draw $\tilde { \mathbf { x } } _ { 1 }$ from its conditional posterior and then draw $\boldsymbol { x } _ { t _ { k + 1 } }$ from the conditional distribution given $\tilde { \mathbf { x } } _ { 1 }$ . Specifically, it is given by

$$
\tilde { \pmb { x } } _ { 1 } \sim \tilde { p } ( \pmb { x } _ { 1 } | \pmb { x } _ { t _ { k } } , \pmb { y } ) ,\tag{32}
$$

$$
\begin{array} { r } { \pmb { x } _ { t _ { k + 1 } } \sim \mathcal { N } ( t _ { k + 1 } \tilde { \pmb { x } } _ { 1 } , ( 1 - t _ { k + 1 } ) ^ { 2 } \pmb { I } ) . } \end{array}\tag{33}
$$

In Flower [35], under a Gaussian likelihood, this sampling step was replaced by the expected value (or an equivalent MAP solution) plus a stochastic perturbation. Although a Laplace approximation could be used for approximate sampling from the Poisson posterior in PET, Flower reported better reconstruction performance when this perturbation was omitted. Based on this finding, our proposed method replaces step (32) with the MAP solution:

$$
\tilde { \pmb { x } } _ { 1 } = \arg \operatorname* { m a x } _ { \pmb { x } _ { 1 } } \log \tilde { p } ( \pmb { x } _ { 1 } | \pmb { x } _ { t _ { k } } , \pmb { y } ) .\tag{34}
$$

This MAP approximation converts the posterior refinement into a penalized likelihood optimization problem, which is well suited to PET reconstruction because efficient optimization algorithms, such as penalized EM methods, are readily available. Therefore, the proposed method uses the MAP estimate instead of directly sampling from $\tilde { p } ( \pmb { x } _ { 1 } \mid \pmb { x } _ { t _ { k } } , \pmb { y } )$ at each flow step.

3) Update rules: The proposed algorithm consists of three steps: (1) flow-based destination estimation to obtain a clean image estimate, (2) penalized PET image reconstruction refinement to enforce data consistency, and (3) flow-based propagation to update the image at the next time step.

1) Based on the conditional prior in (27), the destination image is estimated as the mean of the conditional prior $\tilde { p } ( \mathbf x _ { 1 } | \mathbf x _ { t _ { k } } ) = \mathcal N ( \hat { \mathbf x } _ { 1 } ( \mathbf x _ { t _ { k } } , t _ { k } ) , \beta ^ { - 1 } I )$ ):

$$
\hat { { \pmb x } } _ { 1 } ( { \pmb x } _ { t _ { k } } , t _ { k } ) = { \pmb x } _ { t _ { k } } + ( 1 - t _ { k } ) v _ { \pmb \theta } ( { \pmb x } _ { t _ { k } } , t _ { k } ) .\tag{35}
$$

2) Based on the conditional posterior in (31), the destination image estimate $\hat { \ b x } _ { 1 , k }$ is refined using the measured PET data y by solving the following penalized PET reconstruction problem:

$$
\begin{array} { l } { \displaystyle \tilde { \mathbf { x } } _ { 1 , k } = \arg \operatorname* { m a x } _ { \mathbf { \Phi } _ { \mathbf { x } } } \left\{ { L ( \pmb { y } | \mathbf { x } ) } - \frac { \beta } { 2 } \| \mathbf { x } - \hat { \mathbf { x } } _ { 1 } ( \pmb { x } _ { t _ { k } } , t _ { k } ) \| _ { 2 } ^ { 2 } \right\} , } \\ { \displaystyle \quad = \mathrm { p r o x } _ { - \beta ^ { - 1 } L ( \pmb { y } | \mathbf { x } ) } [ \hat { \pmb { x } } _ { 1 } ( \mathbf { x } _ { t _ { k } } , t _ { k } ) ] , \qquad ( 3 6 ) } \end{array}
$$

where β controls the strength of the flow-based prior. To solve (36), we use the optimization transfer method [39], [40]. The voxel-wise update rule for (36) is

$$
\begin{array} { r l } {  { \tilde { x } _ { 1 , j } ^ { ( n + 1 ) } = \frac { 1 } { 2 } [ \hat { x } _ { 1 , j } - \frac { S _ { j } } { \beta } ] } \quad } & { } \\ & { + \displaystyle \frac { 1 } { 2 } \sqrt { ( \hat { x } _ { 1 , j } - \frac { S _ { j } } { \beta } ) ^ { 2 } + 4 x _ { \mathrm { E M } , j } ^ { ( n + 1 ) } \frac { S _ { j } } { \beta } } , } \end{array}\tag{37}
$$

![](images/7e34db396e81b09aad2a662feb051bad8e7fe23e016c548b1b12c7d03fe9ed1e.jpg)

![](images/0e0e6fd753425f3d47cea047153b448b622259d0fbab3fd5c240df339f473d91.jpg)  
Fig. 2. Impact of the number of sampling steps K on PSNR (top) and on the putamen uptake–white matter CoV tradeoff curves (bottom). Filled markers correspond to the images as shown in Fig. 4.

where $\pmb { x } _ { \mathrm { E M } } ^ { ( n + 1 ) }$ is obtained by MLEM (3) from $\tilde { \pmb { x } } _ { 1 } ^ { ( n ) }$

3) Based on the stochastic propagation model in (33), the refined destination estimate is propagated to the next time point as

$$
\begin{array} { r } { \pmb { x } _ { t _ { k + 1 } } = ( 1 - t _ { k + 1 } ) \pmb { \epsilon } _ { k } + t _ { k + 1 } \pmb { \tilde { x } } _ { 1 } , } \end{array}\tag{38}
$$

where $\epsilon _ { k } \sim \mathcal { N } ( 0 , I )$ is sampled at each time step.

The overall algorithm of the proposed model-based PET image reconstruction method is shown in Algorithm 2. To further improve reconstruction stability, we used an ensemble average of five independently reconstructed images, denoted as Proposed(5).

## IV. EXPERIMENTAL SETUP

We evaluated flow matching-based reconstruction methods using clinical brain [<sup>18</sup>F]FDG PET data. The experiments were run on an NVIDIA B200 GPU with 192 GB of memory.

## A. Brain PET data

For the clinical data experiment, we used pretrained unconditional flow models trained on early 0-10 min frames of $[ ^ { 1 8 } \mathrm { F } ] \mathbf { M K - } 6 2 4 0$ tau scans [17], with an injected dose of approximately 185 MBq. These early-time-frame PET images mainly reflected cerebral blood flow information. The PET images were reconstructed using the ordered subsets EM algorithm with 3 iterations and 17 subsets, including point spread function modeling and time-of-flight information. The reconstructed images had a matrix size of $2 5 6 \times 2 5 6 \times 8 9$ and a voxel size of $1 . 1 7 \times 1 . 1 7 \times 2 . 7 9 ~ \mathrm { { m m ^ { 3 } } }$ . The PET images were subsequently downsampled and centrally cropped, resulting in a matrix size of $1 2 8 \times 1 2 8 \times 8 0$ and a voxel size of $2 . 3 4 \times 2 . 3 4 \times 2 . 7 9 ~ \mathrm { { m m ^ { 3 } } }$ . The model was implemented using a 3D U-Net backbone and trained on 116 PET datasets. Among these datasets, 110 were used for training and 6 for validation.

![](images/bf08b6e6a867d6b44b5f2ac90c25107dfee8444f02f032a771847f1cf4b23c75.jpg)  
Fig. 3. Impact of the number of inner EM iterations N on the putamen uptake–white matter CoV trade-off curves across the 10 subjects at the 10% count level. Markers were plotted for $N = 1 , 2 , \dotsc , 1 4 .$ . Filled markers indicate $N = 1 0$ , which was used for the representative images shown in Fig. 4.

During testing, we used 10 clinical [<sup>18</sup>F]FDG data from the Monash DaCRA fPET–fMRI dataset [41]. Dynamic PET scans were acquired over 90 min following an injection of approximately 238 $\mathrm { M B q }$ of $[ ^ { 1 8 } \mathrm { F } ] \mathrm { F D G }$ , and the data acquired during the 80–90 min interval were used for testing. Low dose PET data were simulated by downsampling the original listmode data to 10% and 5% of the original counts. The 10% dose level was used for the primary evaluation, and the 5% dose level was evaluated to assess reconstruction performance under more severe low-dose conditions. The corresponding full dose data were used to generate reference images. The PET images were reconstructed on a $1 2 8 \times 1 2 8 \times 8 0$ voxel grid with a voxel size of $2 . 3 4 \times 2 . 3 4 \times 2 . 7 9 ~ \mathrm { { m m ^ { 3 } } }$ . Scatter and random corrections were performed using a voxel-driven scatter model and a maximum-likelihood method, respectively, and magnetic resonance-derived $\mu$ maps were used for attenuation correction [42]. The system matrix was implemented using Parallelproj [43].

## B. Evaluation

We compared the proposed model-based PET reconstruction method with MLEM, MAPEM using the relative difference penalty [44], decomposed diffusion sampling (DDS) [25], and PET-FlowDPS. For a fair comparison, the diffusion model used in DDS was trained using the same training datasets and employed the same network architecture as the corresponding flow model.

For the MAPEM method, we used the relative difference penalty defined as

$$
R ( \pmb { x } ) = \sum _ { j } \sum _ { k \in N _ { j } } \frac { ( x _ { j } - x _ { k } ) ^ { 2 } } { ( x _ { j } + x _ { k } ) + \gamma | x _ { j } - x _ { k } | } ,\tag{39}
$$

where $N _ { j }$ stands for a set of neighborhood indices of jth voxel and $\gamma$ controls the shape of the function. Following the default setting for clinical PET scanners, we set $\gamma = 2 ~ [ 4 5 ]$

For DDS reconstruction, we used the following iterative update:

$$
\begin{array} { r } { \hat { \pmb { x } } _ { 0 } ^ { ( n + 1 ) } = \hat { \pmb { x } } _ { 0 } ^ { ( n ) } + \qquad } \\ { \frac { \hat { \pmb { x } } _ { 0 } ^ { ( n ) } } { S } \left[ { \pmb { A } } ^ { T } \frac { \pmb { y } } { { \pmb { A } } \hat { \pmb { x } } _ { 0 } ^ { ( n ) } + b } - S - 2 \lambda _ { \mathrm { D D S } } \left( \hat { \pmb { x } } _ { 0 } ^ { ( n ) } - \hat { \pmb { x } } _ { 0 } ^ { ( 0 ) } \right) \right] , } \end{array}\tag{40}
$$

where $\hat { \pmb x } _ { 0 } ^ { ( 0 ) } = \hat { \pmb x } _ { 0 } , n = 0 , 1 , \dots , N _ { \mathrm { D D S } } - 1$ denotes the number of iterations, $\hat { \pmb { x } } _ { 0 } ^ { ( n ) } / S$ is a preconditioning term and λ<sub>DDS</sub> controls the strength of the regularization. We set the number of diffusion sampling steps to 200 and $N _ { \mathrm { { D D S } } } ~ = ~ 5$ in the experiments.

For the proposed method, the flow interval was discretized into 50 sampling steps, and 10 penalized EM iterations were performed at each time point. The penalty strength was set to $\beta = 0 . 1$ . Unless otherwise specified, these settings were used for all reported results.

Mean uptake was calculated in the putamen region and background noise was evaluated using the coefficient of variation (CoV) in the white matter region as

$$
\mathrm { C o V = \frac { S D _ { W M } } { M e a n _ { W M } } } ,\tag{41}
$$

where $\mathbf { M e a n } _ { \mathbf { W M } }$ and $\mathbf { S D } _ { \mathbf { W M } }$ are the mean and standard deviation values of the white matter regions, respectively. These regions were derived from FreeSurfer [46]. Peak signal-tonoise ratio (PSNR) was also calculated using the full dose image as the reference as

$$
\mathrm { P S N R } = 1 0 \log _ { 1 0 } \left[ \frac { \operatorname* { m a x } \left( \mathbf { X } _ { \mathrm { f u l l } } \right) ^ { 2 } } { \frac { 1 } { N _ { \mathrm { v o x } } } \left\| \mathbf { X } _ { \mathrm { f u l l } } - \mathbf { X } ^ { \prime } \right\| _ { 2 } ^ { 2 } } \right] ,\tag{42}
$$

where max (·) indicates the maximum value of the image, $X _ { \mathrm { f u l l } }$ and $X ^ { \prime }$ denote the full dose and target reconstructed images, respectively, and $N _ { \mathrm { v o x } }$ is the number of voxels.

For the variability analysis, 10 independent low-dose realizations were generated by randomly sampling 10% of the events from the original listmode data. Each realization was independently reconstructed using all evaluated methods. Voxel-wise mean and standard-deviation images were calculated across the 10 reconstructed images. The bias image was calculated as the voxel-wise difference between the mean reconstructed image and the corresponding full-dose reference image.

![](images/1fd7df6a7ff8945fc55e2483825e02464a3ca84f6f5c1848f8081159ea68130f.jpg)  
Fig. 4. Reconstruction results of three orthogonal slices of the clinical [<sup>18</sup>F]FDG data at 10% dose.

![](images/b48bce0413d4a32874f57c83c66fc04aa0f387bcb96763532a5ea69699bd19de.jpg)  
Fig. 5. Mean putamen uptake–white matter CoV trade-off curves across the 10 subjects at the 10% count level. Markers were plotted at 10-iteration intervals from 10 to 100 for MLEM, at 20-iteration intervals from 20 to 200 for MAPEM $( \beta _ { \mathrm { M A P E M } } = 3 0 , \gamma = 2 )$ , at λ of 0, 10, 100, 500, and 1000 for DDS, at λ of 0.8, 0.9, 1.0, 1.2, and 1.4 for PET-FlowDPS, and at β of 0.05, 0.06, 0.08, 0.1, and 0.25 for Proposed and Proposed(5). Filled markers correspond to the representative images shown in Fig. 4.

## V. RESULTS

Figure 2 shows the impact of the number of sampling steps K on PSNR and on the putamen uptake–white matter CoV tradeoffs. The PSNR peaked at $K = 5 0$ and decreased thereafter. Favorable tradeoffs were observed with both 50 and 75 sampling steps. Figure 3 presents the impact of the number of penalized EM iterations on the putamen uptake– white matter CoV tradeoff curves. Ten EM iterations $( N = 1 0 )$

provided putamen uptake closest to the full dose reference while maintaining lower white matter CoV. Therefore, K = 50 and N = 10 were used in the subsequent experiments.

Figure 4 shows reconstructed images of one clinical [<sup>18</sup>F]FDG data using different methods. PET-FlowDPS and the proposed method produced images with lower noise levels than MLEM, MAPEM, and DDS. In particular, the proposed method provided visually higher image contrast than PET-FlowDPS, consistent with the putamen uptake–white matter CoV trade-off curves shown in Fig. 5. DDS, PET-FlowDPS and the proposed method recovered putamen uptake comparable to that of the full dose reference. However, DDS and PET-FlowDPS exhibited less favorable noise characteristics than the proposed method.

Figure 6 shows the voxel-wise mean, bias, and standarddeviation maps for the different reconstruction methods. MLEM, MAPEM, and DDS exhibited relatively high spatial standard deviations, whereas PET-FlowDPS and the proposed method showed lower spatial standard deviations. PET-FlowDPS exhibited relatively high spatial bias, whereas the proposed method maintained bias levels comparable to those of MLEM and MAPEM while providing lower spatial standard deviations. Overall, the proposed method provided a favorable balance between spatial bias and variability.

To further evaluate reconstruction performance under more severe low-count conditions, Fig. 7 shows the reconstructed images at the 5% count level. Compared with the 10% dose results, PET-FlowDPS produced blurred images, whereas the proposed method preserved image contrast and structural details. Consistent with the reconstructed images, the putamen uptake–white matter CoV tradeoff curves in Fig. 8 showed that the proposed method recovered putamen uptake comparable to that of the full dose reference, whereas PET-FlowDPS underestimated putamen uptake. These quantitative results demonstrate that the proposed model-based flow matching reconstruction outperformed both conventional iterative and generative model-based reconstruction methods.

![](images/7311a993239f1174f6b3fd9ed9d8a9249e97932c9c749d3967ebcdda2e2fb145.jpg)  
Fig. 6. Voxel-wise mean, bias, and standard-deviation maps calculated from 10 independent 10% low dose realizations for the different reconstruction methods.

The computation times for DDS, PET-FlowDPS, and the proposed method were 3.93, 0.40, and 1.85 minutes, respectively.

## VI. DISCUSSION

In this study, we presented flow matching-based PET image reconstruction methods. Although diffusion models have shown promising performance for PET reconstruction, incorporating data consistency into the reverse sampling process remains challenging and can lead to computationally intensive and numerically sensitive updates. In addition, the generative sampling and data-consistency updates are closely coupled through the prescribed reverse diffusion process. Flow matching provides a flexible framework for addressing these limitations because, under a linear probability path, the destination image can be directly estimated from an intermediate state. This property allows destination estimation, PET-data refinement, and flow propagation to be treated as separate steps. Based on this framework, we first developed PET-FlowDPS by incorporating PET likelihood guidance into FlowDPS. We then proposed a model-based reconstruction method in which the PET-data refinement step is formulated as MAP reconstruction.

A key difference between PET-FlowDPS and the proposed method lies in the PET-data refinement step (2) shown in Fig. 1. PET-FlowDPS directly applies a likelihood-based correction to the flow-based destination estimate. Although the PET-specific preconditioning term improves the stability of this update, its behavior still depends on the selected guidance strength λ and the local log-likelihood gradient. In contrast, the proposed method replaces this gradient-based refinement with MAP reconstruction. This formulation provides an explicit balance between the flow-based prior and the PET loglikelihood through the penalty parameter β, making the relative contributions of the learned prior and measured PET data more directly interpretable. This modification may stabilize the PET-data refinement step in the proposed method, which may explain its higher image contrast, lower spatial bias, and lower noise levels compared with PET-FlowDPS.

We investigated the effects of the number of sampling steps and the number of EM iterations. The results suggest that increasing the number of sampling steps does not necessarily lead to improved reconstruction performance. As shown in Fig. 3, increasing the number of EM iterations was important for recovering quantitative uptake and reducing bias. This finding suggests that sufficient optimization of the subproblem in (36) is important for quantitative PET reconstruction. With few EM iterations, the refined image may remain influenced by the flow-based prior and may not sufficiently reflect the likelihood term. However, the effects of these two parameters are not independent because the total number of EM iterations increases with the number of sampling steps. In the proposed method, N EM iterations are performed at each of the K sampling steps, resulting in a total of $K \times N$ EM iterations. Therefore, the decrease in PSNR observed beyond 50 sampling steps may not be solely due to the increased number of sampling steps, but may also reflect the effect of the increased total number of EM iterations. Future work will further investigate the individual and combined effects of these parameters.

At the 10% count level, DDS, PET-FlowDPS, and the proposed method recovered putamen uptake comparable to that of the full-dose reference, indicating that generative priors can preserve quantitative information under low-dose conditions. As shown in Fig. 6, DDS exhibited higher spatial standard deviations than the flow-based methods. PET-FlowDPS provided lower spatial variability at the cost of relatively increased spatial bias. In contrast, the proposed method maintained bias levels comparable to those of MLEM, MAPEM, and DDS while achieving lower spatial standard deviations. These results show that the proposed method provides a more favorable bias–variability balance than the other methods, with both low bias and low variability contributing to improved quantitative accuracy. At the 5% count level, PET-FlowDPS underestimated putamen uptake, whereas the proposed method <sup>0.344 0.25</sup>maintained uptake comparable to that of the full-dose ref-<sup>0.25</sup>erence. This result suggests that the MAP-based PET-data refinement is more robust under noisier conditions than the <sup>0.342</sup>direct likelihood-guidance update used in PET-FlowDPS.

![](images/6dfad4aefc87400fce0199f9ae5ad996b75f7eaef59740046002ffe8be5fa1cb.jpg)  
Fig. 7. Reconstruction results of three orthogonal slices of the clinical [<sup>18</sup>F]FDG data at 5% dose.

![](images/00f793ee46349772c360366411978650ae667ed25342f0a516a5c2b33867e133.jpg)  
Fig. 8. Mean putamen uptake–white matter CoV trade-off curves across the 10 subjects at the 5% count level. Markers were plotted at 10-iteration intervals from 10 to 100 for MLEM, at 10-iteration intervals from 10 to 200 for MAPEM, at λ of 0, 10, 100, 500, and 1000 for DDS, at λ of 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, and 2.5 for PET-FlowDPS, and at β of 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, and 0.25 for Proposed and Proposed(5). Filled markers correspond to the images as shown in Fig. 7.

This study has several limitations. First, the evaluation was performed using 10 clinical brain PET datasets at two low-0.340count levels (10% and 5%). Further evaluation using larger cohorts, different scanners, and other radiotracers is needed to evaluate the generalizability of the proposed method. Second, 0.338the full dose reconstructed images were used as references. These reference images contain statistical noise and correction 0.005errors, which should be considered when interpreting these <sub>0.336</sub>quantitative measures. Third, hyperparameters, such as the number of sampling steps, the number of inner EM iterations, and the penalty strength, may require adjustment for different White matter CoVconditions. Fourth, the formulation of the proposed method relies on several approximations for some probabilistic models. Therefore, the reconstructed images should not be interpreted as exact samples from the PET posterior distribution.

## VII. CONCLUSION

In this work, we proposed flow matching-based PET image reconstruction methods. Evaluations showed that the proposed flow-based method improved PET image quality compared with MLEM, MAPEM, DDS, and PET-FlowDPS methods. The results demonstrate that flow matching is a promising generative prior for PET image reconstruction, offering a favorable balance between quantitative accuracy and denoising performance.

## REFERENCES

[1] L. A. Shepp and Y. Vardi, “Maximum likelihood reconstruction for emission tomography,” IEEE Trans. Med. Imaging, vol. 1, no. 2, pp. 113–122, 1982.

[2] H. M. Hudson and R. S. Larkin, “Accelerated image reconstruction using ordered subsets of projection data,” IEEE Trans. Med. Imaging, vol. 13, no. 4, pp. 601–609, 1994.

[3] P. J. Green, “Bayesian reconstructions from emission tomography data using a modified EM algorithm,” IEEE Trans. Med. Imaging, vol. 9, no. 1, pp. 84–93, 1990.

[4] A. R. D. Pierro, “A modified expectation maximization algorithm for penalized likelihood estimation in emission tomography,” IEEE Trans. Med. Imaging, vol. 14, no. 1, pp. 132–137, 1995.

[5] J. Nuyts and J. A. Fessler, “A penalized-likelihood image reconstruction method for emission tomography, compared to postsmoothed maximumlikelihood with matched spatial resolution,” IEEE Trans. Med. Imaging, vol. 22, no. 9, pp. 1042–1052, 2003.

[6] J. Qi and R. M. Leahy, “Iterative reconstruction techniques in emission computed tomography,” Phys. Med. Biol., vol. 51, no. 15, pp. R541– R578, 2006.

[7] J. Qi and R. Leahy, “Resolution and noise properties of MAP reconstruction for fully 3-D PET,” IEEE Trans. Med. Imaging, vol. 19, no. 5, pp. 493–506, 2000.

[8] A. J. Reader, G. Corda, A. Mehranian, C. da Costa-Luis, S. Ellis, and J. A. Schnabel, “Deep learning for PET image reconstruction,” IEEE Trans. Radiat. Plasma Med. Sci., vol. 5, no. 1, pp. 1–25, 2021.

[9] F. Hashimoto, Y. Onishi, K. Ote, H. Tashima, A. J. Reader, and T. Yamaya, “Deep learning-based pet image denoising and reconstruction: a review,” Radiol. Phys. Technol., vol. 17, no. 1, pp. 24–46, 2024.

[10] K. Miwa et al., “Innovations in clinical PET image reconstruction: advances in Bayesian penalized likelihood algorithm and deep learning,” Ann. Nucl. Med., vol. 39, no. 9, pp. 875–898, 2025.

[11] K. Gong et al., “Iterative PET Image Reconstruction Using Convolutional Neural Network Representation,” IEEE Trans. Med. Imaging, vol. 38, no. 3, pp. 675–685, 2019.

[12] A. Mehranian and A. J. Reader, “Model-Based Deep Learning PET Image Reconstruction Using Forward–Backward Splitting Expectation–Maximization,” IEEE Trans. Radiat. Plasma Med. Sci., vol. 5, no. 1, pp. 54–64, 2021.

[13] J. Whang, M. Delbracio, H. Talebi, C. Saharia, A. G. Dimakis, and P. Milanfar, “Deblurring via Stochastic Refinement,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 16 293–16 303.

[14] C. Saharia, J. Ho, W. Chan, T. Salimans, D. J. Fleet, and M. Norouzi, “Image Super-Resolution via Iterative Refinement,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 4, pp. 4713–4726, 2023.

[15] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Adv. Neural Inf. Process. Syst., 2020.

[16] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2020.

[17] K. Gong, K. Johnson, G. E. Fakhri, Q. Li, and T. Pan, “PET image denoising based on denoising diffusion probabilistic model,” Eur. J. Nucl. Med. Mol. Imaging, vol. 51, pp. 358–368, 2024.

[18] S. Pan et al., “Full-dose whole-body PET synthesis from low-dose PET using high-efficiency denoising diffusion probabilistic model: PET consistency model,” Med. Phys., vol. 51, no. 8, pp. 5468–5478, 2024.

[19] B. Yu et al., “Robust whole-body PET image denoising using 3D diffusion models: evaluation across various scanners, tracers, and dose levels,” Eur. J. Nucl. Med. Mol. Imaging, vol. 52, pp. 2549–2562, 2025.

[20] H. Xie et al., “Dose-aware diffusion model for 3D PET image denoising: Multi-institutional validation with reader study and real low-dose data,” Med. Image Anal., vol. 111, 2026, art. no. 104039.

[21] T. Xie et al., “Synthesizing PET images from high-field and ultra-highfield MR images using joint diffusion attention model,” Med. Phys., vol. 51, no. 8, pp. 5250–5269, 2024.

[22] Y. Gong, S.-i. Jang, W. Shao, Y. Su, and K. Gong, “TauGenNet: Plasma-Driven Tau PET Image Synthesis via Text-Guided 3D Diffusion Models,” IEEE Trans. Radiat. Plasma Med. Sci., 2026. [Online]. Available: https://doi.org/10.1109/TRPMS.2026.3688162

[23] T. Chen et al., “2.5D Multi-View Averaging Diffusion Model for 3D Medical Image Translation: Application to Low-Count PET Reconstruction With CT-Less Attenuation Correction,” IEEE Trans. Med. Imaging, vol. 44, no. 11, pp. 4239–4250, 2025.

[24] M. J. Cho, H. S. Shim, S. Kim, and J. S. Lee, “GPDM: generation-prior diffusion model for accelerated direct attenuation and scatter correction of whole-body 18 F-FDG PET,” Phys. Med. Biol., vol. 71, no. 12, 2026, art. no. 125011.

[25] I. R. Singh et al., “Score-Based Generative Models for PET Image Reconstruction,” Mach. Learn. for Biomed. Imaging, vol. 2, pp. 547– 585, 2024.

[26] H. Chung, J. Kim, M. T. Mccann, M. L. Klasky, and J. C. Ye, “Diffusion Posterior Sampling for General Noisy Inverse Problems,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2023.

[27] H. Chung, S. Lee, and J. C. Ye, “Decomposed Diffusion Sampler for Accelerating Large-Scale Inverse Problems,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2024.

[28] G. Webber, Y. Mizuno, O. D. Howes, A. Hammers, A. P. King, and A. J. Reader, “Likelihood-Scheduled Score-Based Generative Modeling for Fully 3D PET Image Reconstruction,” IEEE Trans. Med. Imaging, vol. 44, no. 11, pp. 4445–4456, 2025.

[29] S. Bae, J. S. Lee, and K. Gong, “Joint reconstruction of activity and attenuation for pet imaging with diffusion prior,” in 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI). IEEE, 2025, pp. 1–4.

[30] C. Phung-Ngoc et al., “Joint Reconstruction of Activity and Attenuation in PET by Diffusion Posterior Sampling in Wavelet Coefficient Space,” IEEE Trans. Radiat. Plasma Med. Sci., 2026. [Online]. Available: https://doi.org/10.1109/TRPMS.2026.3706239

[31] F. Hashimoto and K. Gong, “PET image reconstruction using deep diffusion image prior,” IEEE Trans. Med. Imaging, vol. 45, no. 6, pp. 2628–2638, 2026.

[32] R. Yilmaz, Y. Wu, J. Stegmaier, and V. Schulz, “PET-Adapter: Test-Time Domain Adaptation for Full and Limited-Angle PET Image Reconstruction,” 2026, arXiv:2605.08030.

[33] X. Liu, C. Gong, and Q. Liu, “Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2023.

[34] J. Kim, B. S. Kim, and J. C. Ye, “FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 12 328–12 337.

[35] M. Pourya, B. E. Rawas, and M. Unser, “Flower: A Flow-Matching Solver for Inverse Problems,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2026.

[36] S. T. Martin, A. Gagneux, P. Hagemann, and G. Steidl, “PnP-Flow: Plugand-Play Image Restoration with Flow Matching,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025.

[37] H. Ran, H. Liu, and B. Zhao, “Manifold-constrained pet image reconstruction with flow-matching priors via admm,” in IEEE Conf. Comput. Imaging Using Synth. Apertures (CISA), 2026, pp. 1–5.

[38] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2023.

[39] K. Lange, A. R. Hunter, and I. Yang, “Optimization Transfer Using Surrogate Objective Functions,” J. Comput. Graph. Stat., vol. 9, no. 1, pp. 1–20, 2000.

[40] G. Wang and J. Qi, “Penalized likelihood PET image reconstruction using patch-based edge-preserving regularization,” IEEE Trans.Med. Imaging, vol. 31, no. 12, pp. 2194–2204, 2012.

[41] S. D. Jamadar et al., “Monash DaCRA fPET-fMRI: A dataset for comparison of radiotracer administration for high temporal resolution functional FDG-PET,” GigaScience, vol. 11, pp. 1–12, 2022.

[42] P. J. Markiewicz et al., “NiftyPET: a High-throughput Software Platform for High Quantitative Accuracy and Precision PET Imaging and Analysis,” Neuroinform., vol. 16, no. 1, pp. 95–115, 2018.

[43] G. Schramm and K. Thielemans, “PARALLELPROJ—an open-source framework for fast calculation of projections in tomography,” Front. Nucl. Med., vol. 3, 2024.

[44] J. Nuyts, D. Beque, P. Dupont, and L. Mortelmans, “A concave prior´ penalizing relative differences for maximum-a-posteriori reconstruction in emission tomography,” IEEE Trans. Nucl. Sci., vol. 49, no. 1, pp. 56–60, 2002.

[45] K. Miwa et al., “Impact of γ factor in the penalty function of Bayesian penalized likelihood reconstruction (Q.Clear) to achieve high-resolution PET images,” EJNMMI Phys., vol. 10, 2023, art. no. 3.

[46] B. Fischl, “FreeSurfer,” NeuroImage, vol. 62, no. 2, pp. 774–781, 2012.