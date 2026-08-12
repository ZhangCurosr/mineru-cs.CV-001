# Stream Forcing: Constructing Unified Training Trajectory for Robust Streaming Video Generation

Yueting Zhu<sup>1,∗</sup> Yuehao Song<sup>1,∗</sup> Kaicheng Zhang<sup>3</sup> Bao Tang<sup>1</sup> Shaoyu Chen<sup>2</sup> Qian Zhang<sup>2</sup> Wenyu Liu<sup>1</sup> Xinggang Wang<sup>1,B</sup>

<sup>1</sup> Huazhong University of Science & Technology <sup>2</sup> Horizon Robotics <sup>3</sup> Anyverse Dynamics

## Abstract

Streaming video generation holds strong potential for world modeling, where future frames must be inferred online sequentially to form a continuous video stream. However, streaming video diffusion models introduce a fundamental train-inference mismatch: inference follows a specialized denoising order, whereas advanced training strategies typically require diverse noise-level configurations. To address this trade-offbetween train-inference consistency and training coverage, we reformulate the video diffusion sampling as a frame-indexed stochastic process over noise levels. Within this stochastic process space, we construct a continuous training trajectory along which the sampling schedule progressively evolves from independent sampling to inference-consistent sampling. We further introduce a joint calibration algorithm and a temporal correlative sampling algorithm to ensure trajectory smoothness and crossframe correlation. Building on these designs, we propose Stream Forcing, a unified training framework for streaming video generation that balances training sufficiency and inference efficiency. Extensive experiments demonstrate that Stream Forcing significantly improves generation quality with a 36.6% FVD improvement on the UCF-101 benchmark. Furthermore, our method facilitates robust zeroshot extrapolation to long-horizon video generation with a 27.9% FVD improvement on the UCF-101 benchmark.

## 1 Introduction

Recent advances in video generation, particularly video diffusion models [36, 43, 47], have made remarkable progress in visual fidelity and temporal consistency. Beyond content creation, such models can serve as generative world models [3, 15] for reasoning about future observations in sequential planning, such as embodied AI [8, 34] and autonomous driving [11, 52]. In these settings, the generative model must operate online sequentially, i.e., perform streaming inference using partial temporal context.

Unlike standard bidirectional generation [2, 18, 31], streaming inference requires progressively generating new frames to form a video stream in a causal setting. Recent approaches [22, 35, 41] introduce streamlined generation by overlapping denoising across frames to reduce the high inter-frame latency. To support this inference strategy, two training paradigms have been explored. Progressive sampling, e.g., rolling diffusion models [35], improves traininginference consistency while restricting the coverage of the training distribution. Alternatively, independent sampling, e.g., diffusion forcing [6], enables diverse training configurations while breaking the temporal structure assumed at inference time.

Noting the complementary nature of the two paradigms, we reformulate training-time noise level sampling as a frame-indexed stochastic process. Specifically, the perframe noise level distribution is modeled using the widely adopted Logit-normal distribution [1, 10]. We parameterize this process with the two inherent parameters, which control temporal bias and sampling diversity, together with a temporal correlation parameter that captures dependencies between video frames. Within this parameterized space, independent sampling is characterized by frame-invariant distribution parameters and zero temporal correlation, whereas progressive sampling exhibits frame-dependent bias and strong temporal correlation. Taking these two configurations as endpoints, we construct a continuous training trajectory that gradually transforms the independent into the progressive configuration, thereby reconciling broad training coverage with inference consistency.

Building on this parameterization, we propose Stream Forcing, a unified training paradigm for progressively aligning training-time sampling with the inference schedule. To ensure a smooth transition along the trajectory, we develop a joint calibration algorithm to heuristically determine the distribution parameters at each training step. Meanwhile, we introduce a temporal correlative sampling algorithm to generate noise levels with the prescribed inter-frame dependencies. These components enable training to proceed from independent per-frame sampling, through intermediate calibrated configurations, to inference-consistent sampling.

Experimental results demonstrate that Stream Forcing achieves high video generation quality with FVD improvements of 36.6% on UCF-101 and 4.7% on Taichi-HD, while also generalizing effectively to zero-shot long-horizon generation in a zero-shot setting, yielding FVD improvements of 27.9% and 10.9%, respectively. We further demonstrate that Stream Forcing can be effectively applied to autonomous driving world modeling, improving both FID and FVD on the nuScenes [4] dataset.

Our contributions can be summarized as follows:

• We reformulate training-time noise-level sampling as a frame-indexed stochastic process, establishing a unified space that encompasses mainstream training paradigms.

• We propose Stream Forcing, a unified training framework that combines joint calibration and temporal correlative sampling to construct a continuous training trajectory, enabling smooth transitions throughout training while reconciling diverse coverage with inference consistency.

• Extensive experiments demonstrate that Stream Forcing achieves superior generation quality and generalizes effectively to long-horizon generation and world modeling.

## 2 Background

## 2.1 Inference Strategy for Streaming Video Generation

Existing streaming video inference strategies mainly include chunk-based generation [12, 39] and streamlined generation [22, 41]. As shown in Fig. 1, for a diffusion model with S denoising steps over a T-frame window, chunkbased methods generate each chunk sequentially, incurring an inter-frame latency of up to S denoising steps. In contrast, streamlined methods interleave denoising within the sliding window, jointly processing frames at progressive noise levels and reducing the latency to $S / T$ steps. We adopt streamlined inference and develop a corresponding training paradigm to improve train–inference consistency.

## 2.2 Trade-off in Training

During training, the input clip $\textbf { x } = ~ \{ x _ { 1 } , x _ { 2 } , . . . , x _ { T } \}$ of T frames is corrupted by Gaussian noise at frame-specific noise levels $k _ { t }$ , resulting in a noisy observation

$$
x _ { t } ^ { ( k _ { t } ) } = \sqrt { \bar { \alpha } _ { k _ { t } } } x _ { t } + \sqrt { 1 - \bar { \alpha } _ { k _ { t } } } \epsilon _ { t } , \quad \epsilon _ { t } \sim \mathcal { N } ( 0 , \mathbf { I } ) ,\tag{1}
$$

where $\bar { \alpha } _ { k _ { t } }$ denotes the cumulative variance. Under this formulation, two training strategies have been explored.

Progressive sampling [35] enforces monotonically increasing noise levels across frames within each training

![](images/9a4d35f268713b132b93ee51f066e1a5ed1aad20d0a0ba72dad88cbc0f25ee9c.jpg)

![](images/bf2c9b7f3b42a6e269914da292dd1eb9bd83715f3651b663a39038ab9ebffd43.jpg)  
Figure 1. Comparison of streaming inference methods.(a) Chunk-based Generation.(b) Streamlined Generation.

window in which the noise levels $\{ k _ { t } \} _ { t = 1 } ^ { T }$ satisfy

$$
k _ { t } \leq k _ { t + 1 } , \quad \forall t \in \{ 1 , \dots , T - 1 \} .\tag{2}
$$

While such a strictly increasing design improves consistency with inference, it confines training to a narrow subset of noise schedules and reduces coverage of the training distribution.

Alternatively, independent sampling [6] assigns noise levels independently to frames within the training window. Formally, the noise levels $\{ k _ { t } \} _ { t = 1 } ^ { T }$ satisfy

$$
k _ { t } \sim p ( k _ { 1 } ) , \quad { \mathrm { i . i . d . ~ f o r } } \ t \in \{ 1 , \ldots , T \} .\tag{3}
$$

This design enables diverse noise configurations in training, while independent noise levels across frames ignore the structured noise progression assumed at inference time, which leads to a mismatch between training and inference.

## 3 Method

Motivated by the trade-off in training, we reformulate perframe noise level sampling as a frame-indexed stochastic process to unify both training strategies. We further construct a continuous training trajectory using the joint calibration and temporal correlative sampling algorithms. By combining them, we develop Stream Forcing, a unified curriculum training procedure for streaming video generation.

## 3.1 Reformulation of Training Sampling

## 3.1.1 Preliminary: Logit-Normal Distribution

Logit-normal distribution [1, 10] is widely used to sample the diffusion noise level, which is formulated with a probability density function (PDF)

$$
f ( k ) = \frac { 1 } { \sigma \sqrt { 2 \pi } k ( 1 - k ) } \exp \left( - \frac { \Big ( \log \big ( k / ( 1 - k ) \big ) - \mu \Big ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) ,\tag{4}
$$

where $\mu$ and σ denote the location and scale parameters.

(a) (b) Adjusted Mode   
Uneven Mode Reassigned Density   
Density <sup>Density</sup>Unsmooth Transition Adjustment <sub>ensi</sub>t<sup>y</sup>   
Timestep Timestep   
Density   
Reassignment   
(c) Adjusted Mode (d) Adjusted Mode   
Reassigned Density Reassigned Density   
Density Density   
Timestep Timestep  
Figure 2. Illustration of trajectory constraints. Each subfigure shows marginals for a three-frame sampling process. (a) Neither Constraint 1 nor Constraint 2 is satisfied. (b) Only Constraint 1 is satisfied. (c) Only Constraint 2 is satisfied. (d) Both Constraints are satisfied.

## 3.1.2 Training Sampling as a Stochastic Process with Logit-Normal Marginals

At each step, we sample one noise level for every frame t in the training window. This noise level sequence constitutes a stochastic process with Logit-normal marginals, parameterized by $\theta _ { t } = ( \mu _ { t } , \sigma _ { t } , \rho )$ , where

$\mu _ { t }$ and $\sigma _ { t } \mathbf { : }$ the location parameter and the scale parameter in logit space at frame t that specifies mean and standard deviation of the underlying Gaussian distribution.

• $\rho \colon$ the inter-frame correlation parameter that characterizes the temporal dependence of the stochastic process.

Within this parameterized framework, the two aforementioned training strategies correspond to special cases. Independent sampling corresponds to

$$
\mu _ { t } = \mu _ { 0 } , \quad \sigma _ { t } = \sigma _ { 0 } , \quad \rho = 0 ,\tag{5}
$$

where all frames share an identical marginal $f ( k ; \mu _ { 0 } , \sigma _ { 0 } )$ while noise levels are sampled independently across frames. Ideal progressive sampling arises as a limiting case when

$$
\mu _ { t } \to \log \mathrm { i t } \Big ( \frac { t } { T } \Big ) , \quad \sigma _ { t } \to 0 , \quad \rho \to 1 ,\tag{6}
$$

where the per-frame marginals in the denoising window of $T$ frames collapse to a time-ordered deterministic schedule with strong inter-frame dependence. This formulation unifies existing paradigms and reveals the tension between noise-level coverage and training–inference consistency.

## 3.1.3 Constrained Training Trajectory

We construct a continuous training trajectory from independent sampling to progressive sampling in the parameterized sampling-process space. At each training step s, the sampling configuration is specified by $\theta _ { t } ^ { s } = ( \mu _ { t } ^ { s } , \sigma _ { t } ^ { s } , \rho ^ { s } )$ As training progresses, these parameters gradually evolve along the trajectory rather than remaining fixed. A valid trajectory should provide a smooth curriculum while preserving comparable sampling coverage and appropriate temporal dependence. Accordingly, we impose the following three constraints, illustrated in Fig. 2.

Algorithm 1 Joint Calibration   
Require: Initialization $\left( \mu _ { i n i t } , \sigma _ { i n i t } \right)$ target modes   
$\{ \bar { \zeta } _ { t } ^ { * } \} _ { t = 1 } ^ { T }$   
Ensure: Joint parameters $\{ \mu _ { t } , \sigma _ { t } \} _ { t = 1 } ^ { T }$   
/\*Calculate reference density\*/   
for $t = 1$ to T do   
Initialize $\sigma _ { t } \gets \sigma _ { i n i t } , \mu _ { t } \gets \mu _ { i n i t }$   
$h _ { t } \gets f ( \zeta _ { t } ^ { * } ; \mu _ { t } , \sigma _ { t } ) \quad \mathrm { ~ / { * } ~ E q . ~ } 4 \mathit { * 1 }$   
end for   
H ← median $( \{ h _ { t } \} _ { t = 1 } ^ { T } )$   
/\* Grid search for σ \*/   
for t = 1 to T do   
Set $\Delta \sigma \gets 0 . 0 5 , \Delta h \gets \infty$   
for n = 0 to N − 1 do   
$\hat { \sigma } \gets n \Delta \sigma$   
$\hat { \mu }  \psi ( \zeta _ { t } ^ { * } , \hat { \sigma } ) \ \quad / { * } \operatorname { E q . } 9 ^ { * } /$   
$\hat { h } \gets f ( \zeta _ { t } ^ { * } ; \hat { \mu } , \hat { \sigma } ) \quad \mathrm { ~ / { * } ~ E q . } 4 ^ { * } /$   
if $| | \hat { h } - H | | < \Delta h$ then   
Set $\sigma _ { t } \gets \hat { \sigma } , \Delta h \gets | | \hat { h } - H | |$   
end if   
end for   
Compute $\mu _ { t }  \psi ( \zeta _ { t } ^ { * } , \sigma _ { t } ) \quad \mathrm { ~ / * ~ E q . ~ } 9 ^ { * } /$   
end for   
return $\{ ( \mu _ { t } , \sigma _ { t } ) \} _ { t = 1 } ^ { T }$

Constraint 1: Smooth trajectory transition. Per-frame marginal distribution should evolve continuously across training steps. We therefore require the modes of the perframe PDFs (i.e., the peak location of the PDF) to shift at an approximately uniform rate along the trajectory, avoiding abrupt changes in the sampling configuration (cf. the comparison between Fig. 2(b) and (a)).

Constraint 2: Consistent per-frame coverage. Within each training step, the marginal distributions of different frames should exhibit comparable degrees of dispersion, so that all frames receive similarly broad coverage over noise levels. Otherwise, some frames may be trained repeatedly within a narrow noise range, while others receive broader but sparser supervision, resulting in imbalanced per-frame optimization. We approximate this condition by matching the densities at their modes (compare Fig. 2(c) and (a)).

![](images/4dbfef1916cfb003320497ffd56ee49ca707e2baf73d711ef06a0a2a7065b8e5.jpg)

![](images/3b26a90dca8730d6f52615ab7cd6d2cc939b1c3ab8c25a4a4220642dbaa849f5.jpg)  
Figure 3. Overview of the curriculum training. Our method builds a continuous curriculum transition between independent training and inference-aligned training that balances full distribution coverage with consistency at inference time.

Constraint 3: Inter-frame correlation. The correlation coefficient $\rho ^ { s }$ should also evolve as training progresses, such that the sampled noise levels preserve the temporal dependence between successive frames throughout training.

More discussions on the motivations for these constraints can be referred to Appendix A.1.

## 3.2 Training Trajectory Implementation

We introduce a joint calibration algorithm to satisfy Constraint 1 and Constraint 2, and a temporal correlative sampling algorithm to satisfy Constraint 3.

## 3.2.1 Joint Calibration of Logit-Normal Parameters

We optimize the two Logit-normal parameters jointly to satisfy Constraints 1 and 2. Specifically, Constraint 1 is enforced by uniformly interpolating the marginal modes between the two endpoint configurations (Eqs. $5 \& \ 6 )$ . We set $\mu _ { 0 } = 0$ for the initial setting in Eq. 5, yielding the target modes $\{ \zeta _ { t } ^ { s } \} ^ { \ast }$

$$
\left\{ \zeta _ { t } ^ { s ^ { * } } = ( 1 - s ) \cdot \zeta _ { t } ^ { 0 ^ { * } } + s \cdot \zeta _ { t } ^ { 1 } ^ { * } , \right.\tag{7}
$$

To ensure Constraint 2, we enforce a global reference peak density $H ^ { s }$ and solve the scale parameter $\boldsymbol { \sigma } _ { t } ^ { s }$ that minimize the differences between real peak densities and $H ^ { s }$ . Then, we formulate the optimization objective as

$$
\operatorname* { m i n } _ { \zeta _ { t } ^ { s } , \sigma _ { t } ^ { s } } \mid \mid \zeta _ { t } ^ { s } - \zeta _ { t } ^ { s * } \mid \mid + \mid \mid f ( \zeta _ { t } ^ { s } ; \psi ( \zeta _ { t } ^ { s } , \sigma _ { t } ^ { s } ) , \sigma _ { t } ^ { s } ) - H ^ { s } \mid \mid ,\tag{8}
$$

where $\psi ( \zeta , \sigma )$ is the mode equation that determines the location parameter using the mode ζ and the scale σ:

$$
\psi ( \zeta , \sigma ) = \mathrm { l o g i t } ( \zeta ) + \sigma ^ { 2 } ( 1 - 2 \zeta ) .\tag{9}
$$

We provide a proof of Eq. 9 in Appendix A.4.

We introduce a joint calibration algorithm to solve the optimization problem in Eq. 8, as detailed in Alg. 1. Since all parameters are independent of the training step once the target modes are set, we omit the superscript s. The joint calibration consists of two steps: estimating a global reference peak density and calibrating the marginal parameters of each frame. First, we initialize each frame with $\mu _ { t } = \mu _ { \mathrm { i n i t } }$ and $\sigma _ { t } = \sigma _ { \mathrm { i n i t } }$ , and compute its peak density $h _ { t }$ The global reference density H is then defined as the median of $\{ h _ { t } \} _ { t = 1 } ^ { T }$ . Then, we calibrate $\left( \mu _ { t } , \sigma _ { t } \right)$ for each target mode $\zeta _ { t } ^ { * }$ . We perform a grid search over candidate values ${ \hat { \sigma } } _ { : }$ compute the corresponding $\hat { \mu }$ using Eq. 9, and select the pair whose peak density at $\zeta _ { t } ^ { * }$ is closest to H. Applying this procedure at each training step yields the per-frame marginal parameters $\{ \mu _ { t } ^ { s } , \sigma _ { t } ^ { s } \} _ { t = 1 } ^ { T }$ satisfying Constraints 1 and 2.

## 3.2.2 Temporal Correlated Sampling Strategy

Given the calibrated per-frame marginals, we further introduce temporal correlation without altering their distributions to satisfy Constraint 3. To this end, we use a Gaussian Copula-based [33] correlative sampling algorithm to decouple the temporal dependence structure from the marginal distributions and implement the coefficient $\rho ^ { s }$ gradually increasing from independent sampling with $\rho ^ { s } ~ = ~ 0$ to strongly correlated sampling with $\rho ^ { s } \to 1$

Table 1. 16-frame unconditional generation results on UCF-101 [40] and Taichi-HD [37]. Methods marked with \* are trained on both the training and test splits, while the remaining methods are trained on the training split only. Methods marked with <sup>†</sup> are obtained under the experimental setting of AR-Diffusion [41] for a controlled comparison.
<table><tr><td>Methods</td><td>UCF-101</td><td>Taichi-HD</td></tr><tr><td>VideoGPT* [54]</td><td>2880.6</td><td></td></tr><tr><td>DIGAN* [59]</td><td>1630.2</td><td>156.7</td></tr><tr><td>StyleGAN-V* [38]</td><td>1431.0</td><td>143.5</td></tr><tr><td>LVDM* [16]</td><td>372.0</td><td>99.0</td></tr><tr><td>PVDM [60]</td><td>343.6</td><td>267.0</td></tr><tr><td>FVDM [29]</td><td>一</td><td>194.6</td></tr><tr><td>Latte [31]</td><td></td><td>159.6</td></tr><tr><td>HVDM [23]</td><td>303.1</td><td>77.0</td></tr><tr><td>Diffusion Forcing [6]</td><td>349.4</td><td>150.5</td></tr><tr><td>AR-Diffusion† [41]</td><td>181.9</td><td>100.9</td></tr><tr><td>MAGI* [64]</td><td>297.8</td><td></td></tr><tr><td>FAR*[13]</td><td>279.0</td><td></td></tr><tr><td>FrameDiT [26]</td><td></td><td>95.5</td></tr><tr><td>Ours†</td><td>146.9</td><td>76.1</td></tr><tr><td>Ours</td><td>177.0</td><td>73.4</td></tr></table>

Specifically, for a given training step s, we first generate a correlated latent sequence $\{ z _ { 1 } , z _ { 2 } , \dots , z _ { T } \}$ in the standard normal space using a first-order autoregressive process:

$$
z _ { t } = \rho \cdot z _ { t - 1 } + \sqrt { 1 - \rho ^ { 2 } } \cdot \epsilon _ { t } ,\tag{10}
$$

where $\rho \in [ 0 , 1 ]$ is the inter-frame correlation coefficient and $\epsilon _ { t } \sim \mathcal { N } ( 0 , 1 )$ . This construction preserves the standard normal marginal of each $z _ { t }$ while introducing temporal correlation across the sequence. We then convert each $z _ { t }$ into a uniform quantile using the standard normal CDF. Finally, the quantile is mapped through the inverse CDF of the calibrated marginal distribution for frame t.

$$
y _ { t } = \mu _ { t } + \sigma _ { t } \cdot \Phi ^ { - 1 } \bigl ( \Phi ( z _ { t } ) \bigr ) = \mu _ { t } + \sigma _ { t } \cdot z _ { t } ,\tag{11}
$$

where Φ presents the standard normal CDF. The noise levels are finally generated through a sigmoid function,

$$
k _ { t } = \mathrm { S i g m o i d } ( y _ { t } ) .\tag{12}
$$

As a result, each frame retains its prescribed marginal distribution, while the dependence induced in the Gaussian space is transferred to the sampled noise levels.

## 3.3 Stream Forcing: A Curriculum Training Procedure

Build on the above design, we propose a unified curriculum learning process that continuously evolves the training noise configuration from independent sampling toward inferencealigned progressive sampling, as illustrated in Fig. 3. For implementation, we organize this continuous process into three successive phases: independent diffusion training, curriculum transition, and streaming-noise aligned training.

Table 2. 128-frame unconditional generation results on UCF-101 [40] and Taichi-HD [37], evaluated on the full datasets at a resolution of 256 × 256.
<table><tr><td>Methods</td><td>UCF-101</td><td>Taichi-HD</td></tr><tr><td>StyleGAN-V [38]</td><td>1773.4</td><td>691.1</td></tr><tr><td>PVDM [60]</td><td>648.4</td><td>339.2</td></tr><tr><td>HVDM [23]</td><td>549.7</td><td>258.5</td></tr><tr><td>Diffusion Forcing [6]</td><td>447.3</td><td>340.2</td></tr><tr><td>AR-Diffusion [41]</td><td>572.3</td><td>376.3</td></tr><tr><td>Ours</td><td>322.5</td><td>230.4</td></tr></table>

Independent Training. The training process begins with independent noise sampling to preserve broad noise-level coverage. Specifically, the noise level of each frame is sampled independently according to Eq. 5.

Curriculum Transition. Following the training trajectory, we construct a smooth training trajectory transitioning from independent sampling to training-inference consistent sampling. For practical implementation, we discretize the trajectory into uniformly-spaced points, each defining a specific training configuration, resulting in a curriculum learning schedule. Specifically, we discretize the interval (0, 1) in Eq. 7 into 10 uniformly-spaced configurations, and perform sequential training along this curriculum from 0 to 1.

Inference-Aligned Training. Upon the convergence of the curriculum, the training focuses exclusively on the inference-aligned noise level defined in Eq. 6. This process aligns the model with the streamlined inference schedules.

## 4 Experiments

## 4.1 Comparison with Existing Baselines

We conduct a comparison with existing baselines on UCF-101 [40], and Taichi-HD [37] at a resolution of 256×256. For unconditional generation, we conduct short-term video generation (16 frames) as outlined in Tab. 1, and zero-shot long video extrapolation (128 frames) as outlined in Tab. 2. Results for conditional video generation and experimental settings are provided in Appendix A.3 and Appendix A.2.

## 4.1.1 Video Generation Performance

For UCF-101 [40], as shown in Tab. 1, our method achieves the superior performance in unconditional video generation, with an FVD score of 177.0, improving upon the state-ofthe-art by 36.6% under the same evaluation setting. Similarly, on Taichi-HD [37], our approach achieves an FVD

Table 3. Comparison of driving video generation methods on the nuScenes [4] test set.
<table><tr><td>Methods</td><td>FID</td><td>FVD</td></tr><tr><td>DriveGAN [24]</td><td>73.4</td><td>502.3</td></tr><tr><td>DriveDreamer [48]</td><td>52.6</td><td>452.0</td></tr><tr><td>WoVoGen [30]</td><td>27.6</td><td>417.7</td></tr><tr><td>Drive-WM [49]</td><td>15.8</td><td>122.7</td></tr><tr><td>GenAD [56]</td><td>15.4</td><td>184.0</td></tr><tr><td>Ours</td><td>9.4</td><td>105.0</td></tr></table>

Table 4. Ablation study on the trajectory constraints. TS, DCC, and IFC denote trajectory transition smoothness (C1), per-frame distribution coverage consistency (C2), and inter-frame correlation (C3), respectively.
<table><tr><td>TS</td><td>DCC</td><td>IFC</td><td>FVD</td></tr><tr><td>x</td><td></td><td></td><td>397.9</td></tr><tr><td></td><td>x</td><td></td><td>577.0</td></tr><tr><td></td><td></td><td>x</td><td>466.1</td></tr><tr><td>√</td><td>√</td><td>√</td><td>334.4</td></tr></table>

Table 5. Ablation study on the impact of Independent Training(IT), Curriculum Transition(CT), and Inference-Aligned Training(IAT).
<table><tr><td>IT</td><td>CT</td><td>IAT</td><td>FVD</td></tr><tr><td>√</td><td>x</td><td>x</td><td>394.2</td></tr><tr><td>x</td><td>X</td><td>√</td><td>359.4</td></tr><tr><td>√</td><td>x</td><td>√</td><td>380.9</td></tr><tr><td>√</td><td>√</td><td>x</td><td>343.3</td></tr><tr><td>√</td><td>√</td><td>√</td><td>334.4</td></tr></table>

of 73.4, surpassing existing methods. Compared to representative methods of both progressive sampling (e.g., AR-Diffusion [41]) and independent sampling (e.g., Diffusion Forcing [6]), our method achieves superior performance.

## 4.1.2 Zero-Shot Long Video Extrapolation

To assess the model’s capability for streaming generation, we perform zero-shot long-horizon extrapolation on 128- frame sequences, which are significantly longer than those seen during training. Our method achieves consistently strong extrapolation performance across both datasets. As shown in Tab. 2, our method achieves a 27.9% and 10.9% improvement in FVD score on UCF-101 [40] and Taichi-HD [37], respectively. Together, these results confirm the strong zero-shot extrapolation capability of our method for long-horizon streaming generation.

## 4.2 Application to Autonomous Driving World Models

To assess the applicability of our method for world modeling, we conduct experiments on the nuPlan [5] and nuScenes [4] datasets, two widely-used large-scale autonomous driving datasets. The model is trained on 25- frame clips at a resolution of $5 1 2 \times 2 5 6$ and evaluated on the nuScenes test set. As shown in Tab. 3, we compare with existing driving video generation methods using FID and FVD. The results demonstrate that our training paradigm can be effectively transferred to autonomous driving scenarios, highlighting its potential as a general training strategy for driving world models.

Table 6. Ablation of the temporal correlation coefficient.  
Table 7. Ablation of the ratio between independent training and curriculum transition.
<table><tr><td>ρ</td><td>FVD</td></tr><tr><td> $\mathrm { F i x e d } = 0 . 0 0$ </td><td>466.1</td></tr><tr><td> $\mathrm { F i x e d } = 1 . 0 0$ </td><td>368.3</td></tr><tr><td>Linear Schedule</td><td>334.4</td></tr></table>

<table><tr><td>Ratio</td><td>FVD</td></tr><tr><td>1:1</td><td>334.4</td></tr><tr><td>1:2</td><td>348.7</td></tr><tr><td>2:1</td><td>318.4</td></tr><tr><td>3:1</td><td>348.8</td></tr></table>

## 4.3 Ablation study

## 4.3.1 Trajectory Constraints

We conduct ablation studies to assess the impact of the proposed design constraints, including trajectory transition smoothness (TS), per-frame distribution coverage consistency (DCC), and Inter-frame correlation (IFC), as summarized in Tab. 4. Violating any individual principle leads to a notable increase in FVD. Among them, removing DCC causes the most severe performance degradation, underscoring the importance of maintaining smooth transitions across the training procedure. When all three principles are jointly enforced, the model achieves the best performance, demonstrating that these principles are complementary and essential for high-quality video generation.

## 4.3.2 Curriculum Training

Tab. 5 reports an ablation study on the contribution of our training framework. Training only with the initial independent sampling (Row 1) yields inferior results compared to using only the inference-aligned sampling (Row 2), highlighting the importance of training–inference consistency. Moreover, directly combining them without a smooth transition (Row 3) degrades performance. In contrast, introducing the curriculum transition establishes a smooth training trajectory and results in performance gains (Row 4). The complete training procedure achieves the best performance.

## 4.3.3 Inter-frame correlation coefficient

We present an ablation study on the scheduling strategy of the correlation coefficient ρ in Tab. 6. Gradually increasing ρ achieves the best FVD, indicating that a smooth transition

![](images/5d09231719f44cb3ecbec2286ee2ebf7269bda4811c12e40717ffe6b87c06fae.jpg)  
(b) Visualization of unconditional 128-frame video generation results on the UCF-101 dataset.

Figure 4. Visualization of unconditional video generation on UCF-101 dataset. DF: Diffusion Forcing [6]. AR-D: AR-Diffusion [41] Our method presents higher visual quality and more robust long-horizontal extrapolation.

from independent sampling to strongly correlated sampling better balances training diversity and temporal consistency.

## 4.3.4 Ratio between independent training and curriculum transition

We conduct an ablation study on the ratio between independent training and curriculum transition, as shown in Tab. 7. The model performs best at a 2:1 ratio, which balances training coverage with a smooth transition toward traininference consistency.

## 4.4 Qualitative Results

As shown in Fig. 4, we compare the visualization of the independent sampling method Diffusion Forcing [6], progressive sampling method AR-Diffusion [41], and our method. As illustrated in Fig. 4a, Diffusion Forcing [6] suffers from unstable jitter and AR-Diffusion [41] produces minimal motion in 16-frame generation, while our approach achieves both stable motion and high-quality details. Fig. 4b shows the visualization of 128-frame generation. Our method exhibits more coherent and realistic motion for highly dynamic actions, and remains stable in low-motion scenes.

## 5 Related Work

## 5.1 Video Generation

Video generation has recently advanced along two main paradigms: diffusion models [18] and autoregressive architectures [42, 45]. Diffusion-based approaches [2, 16, 18, 31, 63] perform iterative denoising to achieve high visual fidelity, while autoregressive methods [19, 25, 50, 54] model videos as token sequences through causal next-token prediction. By modeling spatiotemporal dynamics, these approaches exhibit strong potential as world models [3, 15].

## 5.2 Streaming Video Generation

Driven by the growing demand for real-time long-horizon video modeling, streaming video generation [12–14, 17, 20, 27, 41, 51, 57] has recently attracted significant attention. Independent sampling approaches [6, 29, 39, 57] inject independent random noise into each frame, allowing the model to be trained flexibly under diverse noise conditions. Progressive sampling methods sample a base noise level [35, 41, 53] during training for a specific frame and construct a strictly increasing noise sequence over all video frames. Furthermore, self forcing [9, 21] and rolling forcing [28] employ knowledge distillation to perform closedloop consistent training. These approaches are orthogonal to ours and could potentially be combined, which will be explored in future work.

## 6 Conclusion

We propose Stream Forcing, a unified training paradigm for streaming video generation that preserves diverse noiselevel coverage while aligning training with inference. The core idea is to construct a smooth training trajectory that connects independent and progressive sampling, allowing the model to leverage the strengths of both approaches. To realize this, we reformulate the training sampling process within a Logit-Normal parameterized stochastic process space. We introduce joint calibration to enforce smoothness constraints and a Gaussian Copula to enforce interframe correlation. By explicitly enforcing these constraints, the training trajectory achieves a smooth transition between distribution coverage and inference alignment. Extensive experiments demonstrate that Stream Forcing outperforms both single-stage and straightforward sequential training. On two benchmark datasets, it achieves superior video generation quality and shows robust zero-shot extrapolation. These results confirm that constructing a smooth training trajectory with the proposed constraints effectively balances distribution coverage and training–inference consistency, yielding high-quality, temporally coherent streaming video generation.

## References

[1] Jhon Atchison and Sheng M Shen. Logistic-normal distributions: Some properties and uses. Biometrika, 67(2):261–272, 1980. 1, 2

[2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 1, 7

[3] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya

Ramesh. Video generation models as world simulators. 2024. 1, 7

[4] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020. 2, 6, 1

[5] Holger Caesar, Juraj Kabzan, Kok Seang Tan, Whye Kit Fong, Eric Wolff, Alex Lang, Luke Fletcher, Oscar Beijbom, and Sammy Omari. nuplan: A closed-loop ml-based plan ning benchmark for autonomous vehicles. arXiv preprint arXiv:2106.11810, 2021. 6, 1

[6] Boyuan Chen, Diego Mart´ı Monso, Yilun Du, Max Sim-´ chowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffu sion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024. 1, 2, 5, 6, 7, 8

[7] Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, and Song Han. Deep compression autoencoder for efficient high-resolution diffusion models. In International Conference on Learning Representations, pages 96539–96560, 2025. 1

[8] Xiaowei Chi, Chun-Kai Fan, Hengyuan Zhang, Xingqun Qi, Rongyu Zhang, Anthony Chen, Chi-Min Chan, Wei Xue, Qifeng Liu, Shanghang Zhang, and Yike Guo. Empowering world models with reflection for embodied video prediction. In Forty-second International Conference on Machine Learning, 2025. 1

[9] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Selfforcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025. 8, 3

[10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam Levi, Dominik¨ Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling recti fied flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024. 1, 2

[11] Hao Gao, Shaoyu Chen, Yifan Zhu, Yuehao Song, Wenyu Liu, Qian Zhang, and Xinggang Wang. Rad-2: Scaling rein forcement learning in a generator-discriminator framework. arXiv preprint arXiv:2604.15308, 2026. 1

[12] Kaifeng Gao, Jiaxin Shi, Hanwang Zhang, Chunping Wang, Jun Xiao, and Long Chen. Ca2-vdm: Efficient autoregressive video diffusion model with causal generation and cache sharing. arXiv preprint arXiv:2411.16375, 2024. 2, 8

[13] Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Longcontext autoregressive video modeling with next-frame pre diction. arXiv preprint arXiv:2503.19325, 2025. 5

[14] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhi jie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. arXiv preprint arXiv:2503.10589, 2025. 8

[15] David Ha and Jurgen Schmidhuber. World models.¨ arXiv preprint arXiv:1803.10122, 2(3), 2018. 1, 7

[16] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity

long video generation. arXiv preprint arXiv:2211.13221, 2022. 5, 7

[17] Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2568–2577, 2025. 8

[18] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022. 1, 7

[19] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 7

[20] Jinyi Hu, Shengding Hu, Yuxuan Song, Yufei Huang, Mingxuan Wang, Hao Zhou, Zhiyuan Liu, Wei-Ying Ma, and Maosong Sun. Acdit: Interpolating autoregressive conditional modeling and diffusion transformer. arXiv preprint arXiv:2412.07720, 2024. 8

[21] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the traintest gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025. 8, 3

[22] Jihwan Kim, Junoh Kang, Jinyoung Choi, and Bohyung Han. Fifo-diffusion: Generating infinite videos from text without training. Advances in Neural Information Processing Systems, 37:89834–89868, 2024. 1, 2

[23] Kihong Kim, Haneol Lee, Jihye Park, Seyeon Kim, Kwanghee Lee, Seungryong Kim, and Jaejun Yoo. Hybrid video diffusion models with 2d triplane and 3d wavelet representation. In European Conference on Computer Vision, pages 148–165. Springer, 2024. 5

[24] Seung Wook Kim, Jonah Philion, Antonio Torralba, and Sanja Fidler. Drivegan: Towards a controllable high-quality neural simulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5820–5829, 2021. 6

[25] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jose Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. In International Conference on Machine Learning, pages 25105–25124. PMLR, 2024. 7

[26] Minh Khoa Le, Kien Do, Duc Thanh Nguyen, and Truyen Tran. Framedit: Diffusion transformer with matrix attention for efficient video generation. arXiv preprint arXiv:2603.09721, 2026. 5, 2

[27] Haozhe Liu, Shikun Liu, Zijian Zhou, Mengmeng Xu, Yanping Xie, Xiao Han, Juan C Perez, Ding Liu, Kumara Ka-´ hatapitiya, Menglin Jia, et al. Mardini: Masked autoregressive diffusion for video generation at scale. arXiv preprint arXiv:2410.20280, 2024. 8

[28] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025. 8, 3

[29] Yaofang Liu, Yumeng Ren, Xiaodong Cun, Aitor Artola, Yang Liu, Tieyong Zeng, Raymond H Chan, and Jean michel Morel. Redefining temporal modeling in video diffusion: The vectorized timestep approach. arXiv preprint arXiv:2410.03160, 2024. 5, 8, 2

[30] Jiachen Lu, Ze Huang, Zeyu Yang, Jiahui Zhang, and Li Zhang. Wovogen: World volume-aware diffusion for controllable multi-camera driving scene generation. In Eu ropean conference on computer vision, pages 329–345. Springer, 2024. 6

[31] Xin Ma, Yaohui Wang, Xinyuan Chen, Gengyun Jia, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 1, 5, 7, 2

[32] Zhiyuan Ma, Liangliang Zhao, Biqing Qi, and Bowen Zhou. Neural residual diffusion models for deep scalable vision generation. Advances in Neural Information Processing Systems, 37:117456–117480, 2024. 2

[33] R. B. Nelsen. An Introduction to Copulas. Springer-Verlag, New York, 2nd edition, 2006. 4

[34] Yiran Qin, Zhelun Shi, Jiwen Yu, Xijun Wang, Enshen Zhou, Lijun Li, Zhenfei Yin, Xihui Liu, Lu Sheng, Jing Shao, LEI BAI, and Ruimao Zhang. Worldsimbench: Towards video generation models as world simulators. In Forty-second In ternational Conference on Machine Learning, 2025. 1

[35] David Ruhe, Jonathan Heek, Tim Salimans, and Emiel Hoogeboom. Rolling diffusion models. In Proceedings of the International Conference on Machine Learning (ICML), 2024. 1, 2, 8

[36] Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026. 1

[37] Aliaksandr Siarohin, Stephane Lathuili ´ ere, Sergey Tulyakov,\` Elisa Ricci, and Nicu Sebe. First order motion model for image animation. Advances in neural information processing systems, 32, 2019. 5, 6, 1, 2, 3

[38] Ivan Skorokhodov, Sergey Tulyakov, and Mohamed Elhoseiny. Stylegan-v: A continuous video generator with the price, image quality and perks of stylegan2. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3626–3636, 2022. 5

[39] Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. History-guided video diffusion. arXiv preprint arXiv:2502.06764, 2025. 2, 8, 1

[40] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012. 5, 6, 1, 2, 3

[41] Mingzhen Sun, Weining Wang, Gen Li, Jiawei Liu, Jiahui Sun, Wanquan Feng, Shanshan Lao, SiYu Zhou, Qian He, and Jing Liu. Ar-diffusion: Asynchronous video generation with auto-regressive diffusion. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7364–7373, 2025. 1, 2, 5, 6, 7, 8

[42] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model

beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 7

[43] Kling Team, Jialu Chen, Yuanzheng Ci, Xiangyu Du, Zipeng Feng, Kun Gai, Sainan Guo, Feng Han, Jingbin He, Kang He, et al. Kling-omni technical report. arXiv preprint arXiv:2512.16776, 2025. 1

[44] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025. 3

[45] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024. 7

[46] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 1

[47] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1

[48] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-worlddrive world models for autonomous driving. In European conference on computer vision, pages 55–72. Springer, 2024. 6

[49] Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14749–14759, 2024. 6

[50] Yuqing Wang, Tianwei Xiong, Daquan Zhou, Zhijie Lin, Yang Zhao, Bingyi Kang, Jiashi Feng, and Xihui Liu. Loong: Generating minute-level long videos with autoregressive language models. arXiv preprint arXiv:2410.02757, 2024. 7

[51] Wenming Weng, Ruoyu Feng, Yanhui Wang, Qi Dai, Chunyu Wang, Dacheng Yin, Zhiyuan Zhao, Kai Qiu, Jianmin Bao, Yuhui Yuan, et al. Art-v: Auto-regressive text-tovideo generation with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7395–7405, 2024. 8

[52] Tianze Xia, Yongkang Li, Lijun Zhou, Jingfeng Yao, Kaixin Xiong, Haiyang Sun, Bing Wang, Kun Ma, Hangjun Ye, Wenyu Liu, et al. Drivelaw: Unifying planning and video generation in a latent driving world. arXiv preprint arXiv:2512.23421, 2025. 1

[53] Desai Xie, Zhan Xu, Yicong Hong, Hao Tan, Difan Liu, Feng Liu, Arie Kaufman, and Yang Zhou. Progressive autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6322–6332, 2025. 8

[54] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021. 5, 7

[55] Haolin Yang, Feilong Tang, Ming Hu, Qingyu Yin, Yulong Li, Yexin Liu, Zelin Peng, Peng Gao, Junjun He, Zongyuan Ge, et al. Scalingnoise: Scaling inference-time search for generating infinite videos. arXiv preprint arXiv:2503.16400, 2025. 2

[56] Jiazhi Yang, Shenyuan Gao, Yihang Qiu, Li Chen, Tianyu Li, Bo Dai, Kashyap Chitta, Penghao Wu, Jia Zeng, Ping Luo, et al. Generalized predictive model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14662–14672, 2024. 6

[57] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion mod els. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22963–22974, 2025. 8

[58] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. Advances in Neural Information Processing Systems, 37:128940– 128966, 2024. 1

[59] Sihyun Yu, Jihoon Tack, Sangwoo Mo, Hyunsu Kim, Junho Kim, Jung-Woo Ha, and Jinwoo Shin. Generating videos with dynamics-aware implicit generative adversarial networks. arXiv preprint arXiv:2202.10571, 2022. 5

[60] Sihyun Yu, Kihyuk Sohn, Subin Kim, and Jinwoo Shin. Video probabilistic diffusion models in projected latent space. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 18456–18466, 2023. 5

[61] Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626, 2025. 3

[62] Lvmin Zhang, Shengqu Cai, Muyang Li, Chong Zeng, Bei jia Lu, Anyi Rao, Song Han, Gordon Wetzstein, and Maneesh Agrawala. Pretraining frame preservation in au toregressive video memory compression. arXiv preprint arXiv:2512.23851, 2025. 3

[63] Shuai Zhang, Bao Tang, Siyuan Yu, Yueting Zhu, Jingfeng Yao, Ya Zou, Shanglin Yuan, Li Yu, Wenyu Liu, and Xinggang Wang. Mobilei2v: Fast and high-resolution image-tovideo on mobile devices. arXiv preprint arXiv:2511.21475, 2025. 7

[64] Deyu Zhou, Quan Sun, Yuang Peng, Kun Yan, Runpei Dong, Duomin Wang, Zheng Ge, Nan Duan, and Xiangyu Zhang. Taming teacher forcing for masked autoregressive video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7374–7384, 2025. 5

## Appendices

## A.1 Motivation

The fundamental objective is to construct a structured transition from high-entropy independent sampling toward a more concentrated, inference-aligned sampling distribution. Since diffusion noise levels are bounded within (0, 1), we parameterize their frame-wise distributions using the Logit-Normal family, which naturally respects this support and can flexibly represent distributions with varying locations, dispersions, and degrees of skewness. To characterize and control the resulting transition, we use mode dynamics as the primary frame-wise distribution indicator and the correlation coefficient as the measure of inter-frame dependency. We focus on the mode rather than conventional moments such as the mean because, for skewed Logit-Normal distributions, the mode directly identifies the most frequently sampled noise level. As these high-density regions are encountered more often during training, controlling the mode provides a practical reference for regulating the curriculum trajectory. Accordingly, Constraints 1 and 2 govern the evolution of the mode location and peak density, respectively, while Constraint 3 controls inter-frame correlation.

1. Trajectory transition smoothness: This constraint promotes stable optimization across training steps. Abrupt changes in the sampling distribution may introduce sudden shifts in training difficulty and destabilize the curriculum. We therefore enforce a uniform evolution of the mode location along the training trajectory, ensuring that adjacent training configurations change gradually.

2. Per-frame distribution coverage consistency: It promotes balanced optimization across frames within each training step. If the distribution of one frame is sharply concentrated while that of another is overly dispersed, their dominant noise-level regions are sampled with substantially different frequencies, resulting in uneven training emphasis across the sequence. We therefore match the mode probability densities of all frame-wise distributions to maintain comparable sampling concentration.

3. Inter-frame correlation: In the initial diffusion-forcing configuration, frame-wise noise levels are sampled independently, so their inter-frame dependencies are not explicitly modeled. In contrast, inference follows an ordered denoising process in which the noise levels of adjacent frames are strongly correlated. We thus progressively introduce inter-frame correlation along the training trajectory, explicitly aligning the joint training distribution with the sequential structure of inference.

## A.2 Experimental Setup

## A.2.1 Dataset

UCF-101 [40]. The UCF-101 dataset [40] is a large-scale human motion dataset that consists of 13,320 videos with 9,624 clips in the training set and 3,696 clips in the test set across 101 action classes. We train only on the train set.

Taichi-HD [37]. The Taichi-HD dataset [37] is a human motion dataset containing 3,103 videos with 2,818 clips in the training set and 285 clips in the test set. We train only on the train set.

NuPlan [5]. NuPlan is a large-scale autonomous driving dataset containing diverse real-world driving scenarios. The training set includes 1,310 driving logs and 18,104 driving segments, covering approximately 3.49 million frames sampled at 10 Hz, with an average duration of 20 seconds. In our experiments, we only utilize the front-view camera stream among the eight available cameras.

NuScenes [4]. NuScenes consists of 1,000 driving scenes with temporally consistent sensor streams and vehicle motion annotations. The official split contains 700 training scenes, 150 validation scenes, and 150 testing scenes. We train our model on the training split and evaluate its performance on the testing split.

## A.2.2 Metrics

For experiments on the UCF-101 dataset and the Taichi-HD dataset, we evaluate the quality of generated videos by computing the Frechet Video Distance (FVD) [´ 46]. We evaluate both qualities of unconditional and conditional generation on the UCF-101 dataset, and unconditional generation on the Taichi-HD dataset. To validate the generalizability of our method to long-horizon tasks, we provide results with video lengths of both 16 and 128 frames.

For experiments on the nuPlan [5] dataset and the nuScenes [4] dataset, we evaluate future video generation with 25-frame clips and measure the generation quality using Frechet Inception Distance (FID) and Fr ´ echet Video´ Distance (FVD).

## A.2.3 Model

For experiments on UCF-101 [40] and Taichi-HD [37], we use DFoT [39] DiT as the backbone for the diffusion model, and AR-VAE introduced in AR-Diffusion [41], which is a Transformer-based 1D tokenizer [58], producing a token sequence of length 32. The backbone DiT contains 674M parameters.

For experiments on nuScenes [4], we use the Wan 2.1 DiT [47] as the backbone for the diffusion model, and a Deep Compression Autoencoder (DCAE) [7] fine-tuned on driving data, which produces a 16-channel latent with 8× spatial and 1× temporal compression.

The detailed model configuration is provided in Tab. A1.

Table A1. Architecture configurations.
<table><tr><td>Configuration</td><td>DFoT</td><td>Wan</td></tr><tr><td>#Parameters</td><td>674M</td><td>1.3B</td></tr><tr><td>Transformer Blocks</td><td>28</td><td>32</td></tr><tr><td>Hidden Size</td><td>1152</td><td>2048</td></tr><tr><td>Attention Heads</td><td>16</td><td>16</td></tr><tr><td>Resolution</td><td>256 × 256</td><td>512 × 256</td></tr><tr><td>Frames</td><td>16</td><td>25</td></tr><tr><td>VAE Channels</td><td>4</td><td>16</td></tr><tr><td>VAE Downsample</td><td>[1, 8]</td><td>[1, 8]</td></tr><tr><td>VAE Tokens / Frame</td><td>32</td><td>512</td></tr></table>

## A.2.4 Training Details

All experiments are conducted on 8 NVIDIA H20 GPUs. For experiments on UCF-101 [40] and Taichi-HD [37], training is performed with a batch size of 40 per GPU. The model is optimized using AdamW with $\beta _ { 1 } ~ = ~ 0 . 9$ $\beta _ { 2 } ~ = ~ 0 . 9 9$ , and $\epsilon \ : = \ : 1 0 ^ { - 8 }$ The learning rate is set to $2 \times 1 0 ^ { - 5 }$ with a 10k-step warmup. We adopt v-prediction as the training objective and use 50-step DDIM sampling with a CFG scale of 2.0 during inference. The maximum correlation coefficient is set to $\rho _ { \mathrm { m a x } } = 0 . 9 5$ The full training schedule consists of 600k steps, including 300k steps of independent training, 150k steps of curriculum transition, and 150k steps of inference-aligned training. For ablation studies, we train models for 450k steps and evaluate them on 16- frame unconditional generation on UCF-101 [40]. More details are provided in Tab. A2. For experiments on nuScenes [4], training is performed with a batch size of 4 per GPU. The model is optimized using AdamW with $\beta _ { 1 } ~ = ~ 0 . 9$ $\beta _ { 2 } = 0 . 9 9 , \epsilon = 1 0 ^ { - 8 }$ , and weight decay 0.01. The learning rate is set to $5 \times 1 0 ^ { - 5 }$ with a 2.5k-step cosine warmup. We adopt flow matching with velocity prediction as the training objective and use 50-step sampling with a CFG scale of 6.0 during inference. The Copula-based scheduling uses $\rho _ { \mathrm { m a x } } = 0 . 9 5$ . The training schedule consists of 10k steps. We evaluate 25-frame conditional video generation, using the first frame as the visual condition and generating the subsequent 24 frames.

## A.3 Conditional video generation

We conduct conditional generation experiments on the UCF-101 dataset [40]. We train the model on 16-frame clips at a resolution of 256×256. Our method presents significant improvements over prior approaches as shown in Tab. A3. We further evaluate the long-horizon extrapolation capability on 128-frame generation, as shown in Tab. A4. These results confirm that our method maintains strong zero-shot extrapolation capability in the conditional setting for longhorizon streaming generation.

Table A2. Training and sampling configurations for experiments on UCF-101 [40] and Taichi-HD [37].
<table><tr><td>Configuration</td><td>Value</td></tr><tr><td>Training Objective</td><td>v-prediction</td></tr><tr><td>Diffusion Timesteps</td><td>1000</td></tr><tr><td>Sampling Steps</td><td>50</td></tr><tr><td>ρmax</td><td>0.95</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td> $\beta _ { 1 } , \beta _ { 2 }$ </td><td>0.9, 0.99</td></tr><tr><td>Learning Rate</td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Warmup Steps</td><td>10k</td></tr><tr><td>Gradient Clipping</td><td>1.0</td></tr><tr><td>Batch Size</td><td>40</td></tr><tr><td>EMA Decay</td><td>0.9999</td></tr><tr><td>CFG Scale</td><td>2.0</td></tr><tr><td>Independent Training</td><td>300k steps</td></tr><tr><td>Curriculum Transition</td><td>150k steps</td></tr><tr><td>Inference-Aligned Training</td><td>150k steps</td></tr></table>

Table A3. Comparison on conditional video generation on UCF-101. \* indicates training on train + test split, while methods without \* are trained only on train. ‡ indicates results evaluated on the test split, while all other results are evaluated on the full dataset. All methods generate videos at 256×256 resolution and 16 frames.
<table><tr><td>Methods</td><td>FVD</td></tr><tr><td>Latte* [31]</td><td>478.0</td></tr><tr><td>FVDM* [29]</td><td>468.2</td></tr><tr><td>Neural-RDM* [32]</td><td>461.0</td></tr><tr><td>Diffusion Forcing [6]</td><td>157.8</td></tr><tr><td>Ca2-VDM‡ [12]</td><td>184.5</td></tr><tr><td>FrameDiT* [26]</td><td>170.1</td></tr><tr><td>Ours‡</td><td>158.3</td></tr><tr><td>Ours</td><td>121.0</td></tr></table>

Table A4. Comparison of methods on UCF-101 [40] for conditional generation, evaluated on the full dataset. All methods generate videos at 256×256 resolution and 128 frames.
<table><tr><td>Methods</td><td>FVD</td></tr><tr><td>FIFO-Diffusion [22]</td><td>596.6</td></tr><tr><td>Diffusion Forcing [6]</td><td>272.3</td></tr><tr><td>ScalingNoise [55]</td><td>539.2</td></tr><tr><td>Ours</td><td>232.4</td></tr></table>

## A.4 Proof of the Mode Equation

In this section, we provide a brief proof of Eq. 9.

Assume that $X \in ( 0 , 1 )$ follows a Logit-Normal distribution with parameters $( \mu , \sigma ^ { 2 } )$ , whose density is given by

$$
f ( x ) = \frac { 1 } { x ( 1 - x ) \sigma \sqrt { 2 \pi } } \exp \left( - \frac { ( l o g i t ( x ) - \mu ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) .
$$

Taking the logarithm of the density and discarding additive constants yields

$$
\ell ( x ) = - \log x - \log ( 1 - x ) - { \frac { ( l o g i t ( x ) - \mu ) ^ { 2 } } { 2 \sigma ^ { 2 } } } .
$$

Differentiating ℓ(x) with respect to x gives

$$
\ell ^ { \prime } ( x ) = \frac { 2 x - 1 } { x ( 1 - x ) } - \frac { l o g i t ( x ) - \mu } { \sigma ^ { 2 } x ( 1 - x ) } .
$$

Since $x ( 1 - x ) > 0$ for all $x \in ( 0 , 1 )$ , the stationary points of the density satisfy

$$
( 2 x - 1 ) - \frac { l o g i t ( x ) - \mu } { \sigma ^ { 2 } } = 0 ,
$$

which can be rearranged as

$$
l o g i t ( x ) - \mu = \sigma ^ { 2 } ( 2 x - 1 ) .
$$

If the density attains its maximum at $x = \zeta \in ( 0 , 1 )$ , substituting $x = \zeta$ into the above condition yields

$$
\begin{array} { r } { \mu = l o g i t ( \zeta ) + \sigma ^ { 2 } ( 1 - 2 \zeta ) . } \end{array}
$$

Moreover, the density vanishes as $x \to 0 ^ { + }$ and $x \to 1 ^ { - }$ and admits a unique stationary point in (0, 1), ensuring that this solution corresponds to the unique global maximum of the density.

## A.5 Why Mode-Density Matching for Constraint 2

Constraint 2 is implemented through a mode-based heuristic rather than a global dispersion statistic such as differential entropy. Although entropy quantifies the overall uncertainty of a distribution, it does not directly govern the sampling density near its dominant noise level. Consequently, frame-wise distributions with similar or smoothly varying entropy may still exhibit substantially different peak densities, leading to uneven training emphasis across the temporal sequence.

To examine this distinction, we replace the proposed mode-density consistency constraint with two linear entropy schedules. Specifically, the entropy is gradually decreased from −0.2 nats to −7 nats in Ent 1 and to −3 nats in Ent 2, as reported in Tab. A5. Both entropy-based variants yield higher FVD than our mode-based design. These results indicate that controlling global distributional uncertainty alone is insufficient. Instead, directly matching the dominant sampling density across frames provides a more effective mechanism for balancing training emphasis and improving temporal generation quality.

Table A5. Comparison between entropy and mode DCC.
<table><tr><td>Setting</td><td>FVD</td></tr><tr><td>Ent 1</td><td>675</td></tr><tr><td>Ent 2</td><td>590</td></tr><tr><td>Ours</td><td>334</td></tr></table>

## A.6 Limitation and Future Work

## A.6.1 Limitation

Due to computational and data constraints, experiments are conducted on small- to medium-scale datasets. While sufficient to demonstrate the effectiveness and robustness of Stream Forcing, this setting does not fully reflect its behavior under large-scale training. We leave a systematic study of scaling to future work.

## A.6.2 Future Work

Several orthogonal directions, such as memory compression techniques [61, 62] and closed-loop distillation methods [9, 21, 28], remain to be jointly explored, and their interactions with the proposed framework have not yet been systematically studied. Extending the method to large-scale training on top of pretrained models [44] is a promising direction.

## A.7 More Visualization

In this section, we present additional visualizations of our method. Specifically, we show unconditional video generation on the Taichi-HD dataset [37] for 16-frame sequences (Fig. A1a) and long-horizon extrapolation to 128 frames (Fig. A1b). We also show conditional video generation on the UCF-101 dataset [40] for 16-frame sequences Fig. A2a) and extrapolation to 128 frames (Fig. A2a). These results demonstrate that our method maintains temporal consistency and generates visually coherent sequences across both datasets and time horizons.

![](images/baba15553af84abe3b8f7d170f5e4fecf024af896524939333365a5ee0dc6765.jpg)  
Diffusion Forcing

![](images/e04c132e5faac5e0e5fea1098e5328eb3ceb27f16fb132f77034ca0375286bc5.jpg)  
AR-Diffusion

![](images/436b5ed59c524f3c0f087b4b3cb5080b68c2168a2866b278a30df3df34c8d761.jpg)  
Stream Forcing (ours)

(a) Visualization of unconditional 16-frame video generation results on the Taichi-HD dataset.  
![](images/c5e647207b377962a6f53247f7cc665c74695f5c161854d7f7191ac4bc445f48.jpg)

Diffusion Forcing  
![](images/6dbe90ed63b109e3d5387c284680d62a2ce97a99313493c10e3f640cd0f5f386.jpg)  
AR-Diffusion

![](images/8d2b450b4da0a9928161497c1878f67e66e898f106044beca99f0d7b138ff330.jpg)  
Stream Forcing (ours)  
(b) Visualization of unconditional 128-frame video generation results on the Taichi-HD dataset.  
Figure A1. Visualization of unconditional video generation on Taichi-HD dataset.

![](images/d4f945e0e2b86c87c00afc18080f52b7c696d49dfe7d0356b16026dd21cbb9d9.jpg)  
Stream Forcing (ours)  
(a) Visualization of conditional 16-frame video generation results on the UCF-101 dataset.

![](images/5ac7369ad122026fca7ac5bdf1d812f37a7690634c38d7d0c422db2b3bf5e8a9.jpg)  
Figure A2. Visualization of conditional video generation on UCF-101 dataset.