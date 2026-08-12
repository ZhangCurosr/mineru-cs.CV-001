# Flow Straight to Reality: Perceptually Consistent Flow Matching for Eficient Image Restoration

Sangwoo Jo<sup>1</sup> , Donggeun Ko<sup>2</sup> , Jayeon Kang<sup>1</sup> , Youngsang Kwak<sup>2</sup> , Jaehwa Kwak<sup>2</sup>, and Sungjoon Choi<sup>1</sup>

<sup>1</sup> Korea University, Seoul, South Korea

{jasonjo97, nature0213, sungjoonc}@korea.ac.kr

<sup>2</sup> Aim Future, Seoul, South Korea

{sean.ko, youngsang.kwak, jaehwa.kwak}@aimfuture.ai

Abstract. Image restoration is fundamentally constrained by the tradeof between distortion and perception: minimizing pixel-wise error yields over-smoothed results, whereas optimizing for perceptual realism often introduces structural deviations. Recent approaches attempt to balance this tradeof via posterior sampling or multi-stage generative pipelines, yet remain computationally expensive and architecturally complex. To overcome these limitations, we propose PCFlow (Perceptually Consistent Flow Matching), a unified framework that directly parameterizes a continuous transport from degraded observations to clean targets, jointly optimizing distortion and perceptual quality. While its latent consistency flow objective drives stable and eficient few-step inference, a Latent Consistency Perceptual Loss (LCPL) imposes semantic constraints directly on the guiding velocity field, steering the dynamics toward visually sharp data manifolds. Furthermore, recognizing the inherent conflict between structural and perceptual consistencies, we integrate a conflict-free gradient projection strategy to stabilize the multi-objective optimization landscape. Combined with lightweight, convolution-only backbone, PCFlow achieves competitive performance across diverse restoration tasks at a fraction of traditional computational costs.

## 1 Introduction

Image restoration aims to recover a clean image from its degraded observation, constituting a fundamental class of inverse problems in computer vision. Fundamentally, this process is plagued by the inherent ambiguity of degradation: multiple plausible high-quality reconstructions can map to a single corrupted input. As established in classical literature [4, 10], this ambiguity forces a strict distortion-perception tradeof. Methods that minimize distortion error (e.g., MSE) mathematically collapse to the conditional expectation, yielding over-smoothed results. Conversely, approaches optimizing for perceptual realism (e.g., FID) hallucinate high-frequency details, improving visual quality but incurring severe structural deviations and high reconstruction error.

A principled way to achieve perceptually optimal solutions is posterior sampling, which generates samples from the underlying posterior distribution and

PCFlow (Ours)

LQ  
PMRF  
![](images/cd739a5c804c0ee71d98b0ff965d67654312c5d99456a36639cb7a7adccc92fb.jpg)  
Fig. 1: Visual Results of PCFlow on the Blind Face Restoration (BFR). PCFlow consistently produces visually faithful and high-quality reconstructions while requiring only a few inference steps (K = 5).

thus attains the optimal perceptual index [4, 10]. However, such methods do not minimize distortion and, in expectation, yield an MSE that can be up to twice the minimum achievable error. Recent difusion and score based restoration approaches leverage generative priors to approximate posterior sampling [5, 13, 21, 23, 26, 33]. While these approaches significantly improve perceptual quality, they rely on iterative stochastic sampling and typically require multiple sampling steps, resulting in substantial computational cost at inference time. To circumvent this, recent multi-stage frameworks decompose the problem: they first predict a fidelity-oriented minimum mean-squared error (MMSE) estimate, and subsequently apply heavy generative refinement to recover details [6, 16, 19, 29]. While efective, such two-stage pipelines inherit architectural complexity and still rely on costly generative steps during inference.

In this work, we revisit the problem from a diferent perspective. Instead of performing stochastic posterior sampling or decomposing restoration into multiple stages, we propose PCFlow (Perceptually Consistent Flow Matching), a unified direct transport framework that efectively balances distortion and perception. Rather than decomposing the restoration process, we tackle the multiobjective optimization directly within a continuous latent space.

At its core, PCFlow parameterizes a direct vector field from the degraded input to the clean target. By employing a latent consistency flow matching (LCFM) objective, we enforce geometric consistency along the trajectory, effectively straightening the integration path and enabling image restoration in as few as three inference steps. However, learning a few-step transport with an $L _ { 2 ^ { - } }$ based objective inherently risks regression to the posterior mean. To mitigate this efect, we introduce a Latent Consistency Perceptual Loss (LCPL), which encourages the trajectory endpoints to align with perceptually sharp, high-density data manifolds. Nevertheless, simply combining these objectives introduces gradient conflicts, particularly at early transport timesteps (low SNR regimes). To stabilize this multi-objective optimization, we adopt a conflict-free gradient projection strategy along with an SNR-adaptive schedule, using the perceptual objective as a steering signal while removing structural gradient components that conflict with perceptual optimization.

Furthermore, PCFlow employs a lightweight convolution-only model architecture without attention modules, significantly reducing model size and computational cost. Combined with our conflict-free consistency training, PCFlow operates at a fraction of traditional computational costs while achieving efective balance on the distortion-perception curve.

Our main contributions are summarized as follows:

– We propose PCFlow, a unified direct transport framework that integrates perceptual consistency into a latent flow objective, enabling highly eficient few-step image restoration.

– We analyze the destructive gradient interference between structural and perceptual objectives, and introduce a conflict-free, SNR-adaptive gradient alignment method to stabilize training.

– We adopt a convolution-only architecture that rivals computationally heavy multi-stage difusion pipelines, significantly reducing parameter count and inference time.

– Extensive evaluations across several benchmarks, including blind face restoration, super-resolution, denoising, inpainting, and colorization, demonstrate that PCFlow advances the distortion-perception tradeof frontier.

## 2 Related Work

## 2.1 Generative Models for Image Restoration

Image restoration has undergone a paradigm shift with the advent of generative modeling. Primary works have formulated the problem through the lens of Bayesian posterior sampling [5, 13, 21, 23, 26, 33]. By iteratively drawing samples from the posterior $p ( x \mid y ) \propto p ( x ) p ( y \mid x )$ using difusion priors, these models attain exceptional perceptual quality. Beyond explicit posterior sampling, conditional generative frameworks have been also explored [16,32]. Instead of deriving a posterior via likelihood models, these approaches introduce various mechanisms to incorporate degraded images as conditional inputs.

Recent works attempt to bridge distortion minimization and perceptual realism through a disjointed, two-stage pipeline [6,19]. Motivated by the distortionperception tradeof theory [4, 10], they first anchor the generation to the minimum mean-squared error (MMSE) estimator,

$$
\boldsymbol { x } ^ { * } = \mathbb { E } [ \boldsymbol { x } \mid \boldsymbol { y } ] ,\tag{1}
$$

which mathematically guarantees minimal distortion. Subsequently, they must learn an optimal transport from this safe, over-smoothed estimate $x ^ { * }$ to the target distribution x. To execute this transport, approaches like DOT [1] attempt to bypass iterative sampling by directly approximating the transport trajectory with a closed-form solution under Gaussian assumptions. Although efective, such two-stage pipelines introduce additional architectural complexity and computational cost at inference time.

## 2.2 Consistency Flow Matching

Flow-based generative models [2, 17, 18] have recently emerged as an alternative class of generative models that learn a continuous vector field v(x, t) transporting samples between two distributions $\mathbf { x } _ { 0 } \sim \pi _ { 0 }$ and $\mathbf { x } _ { 1 } \sim \pi _ { 1 }$ through an ordinary diferential equation (ODE):

$$
\frac { d \mathbf { x } } { d t } = v ( \mathbf { x } ( t ) , t ) ,\tag{2}
$$

In other words, flow-based approaches directly parameterize the transport vector field between distributions, enabling explicit trajectory modeling and eficient inference through numerical ODE integration.

To further improve training and inference eficiency, Consistency Flow Matching (CFM) [27] introduces a consistency objective that efectively straightens the flow trajectory by aligning predictions across neighboring timesteps. Instead of explicitly modeling the entire probability path, CFM enforces consistency in both trajectory outputs and velocity field, leading to stable training and supporting few step inference. While CFM has demonstrated strong performance in unconditional generative modeling, it has primarily been studied in pixel space. Its extension to conditional generative tasks, such as image restoration, as well as to latent representation spaces remains relatively underexplored.

## 2.3 Perceptual Objective

Perceptual objectives have been widely adopted in image restoration to enhance high-frequency details [11, 14, 24, 25, 31]. A common approach is to incorporate feature-based perceptual losses, most notably LPIPS [30], which measures the distance between pretrained VGGNet features [22] extracted from predicted and ground truth images. Such external perceptual losses encourage semantic similarity beyond pixel-wise fidelity and have become standard regularizers in image restoration tasks. E-LatentLPIPS [12] extends LPIPS to latent space, where the training objective remains identical but with additional data augmentation to address the suboptimal loss landscape inherent in latent representations.

More recently, several works have explored leveraging internal features of generative models themselves for perceptual supervision, instead of relying on external pretrained networks. These methods utilize intermediate features, such as U-Net midblock layer or decoder features, to define self-perceptual losses [3,15]. By aligning intermediate features, such approaches aim to better preserve generative priors and semantic coherence. However, such methods have been developed primarily for image generation, and their extension to restoration settings remains limited.

![](images/5c89f72698b397367839e0a41f16a1c8c0d84d5ba9192120ef351d13d0f1bc1b.jpg)  
Fig. 2: Overview of PCFlow. Given a degraded image, our proposed PCFlow encodes it into latent $\mathbf { z } _ { 0 }$ using the LQ encoder, and learns the latent transport with a flow model v . The restored latent prediction zˆ is then decoded by D to obtain the final output image xˆ. The training objective combines a latent consistency flow matching loss L<sub>LCFM</sub> with a latent consistency perceptual loss L<sub>LCPL</sub>, computed from multi-level decoder features $\{ \phi _ { l } \} _ { l = 1 } ^ { L } ,$ yielding the final objective as $L _ { \mathrm { t o t a l } } = L _ { \mathrm { L C F M } } + \lambda _ { \mathrm { L C P L } } L _ { \mathrm { L C P L } }$

## 3 Method

## 3.1 Latent Consistency Flow Matching

Formulating Latent Transport. Following ELIR [6], we begin by defining a latent consistency flow matching (LCFM) objective that integrates latent flow matching [7] with consistency training [27]. Given a low-quality (LQ) latent $\mathbf { z } _ { 0 }$ and its corresponding high-quality (HQ) latent $\mathbf { z } _ { 1 }$ , we parameterize a vector field v<sub>θ</sub> that governs the latent transport over time $t \in [ 0 , 1 ]$

$$
\frac { d z } { d t } = v _ { \theta } ( \mathbf { z } ( t ) , t ) ,\tag{3}
$$

where we define a linear interpolation path in the latent space:

$$
\mathbf { z } _ { t } = t \mathbf { z } _ { 1 } + \left( 1 - \left( 1 - \sigma _ { \operatorname* { m i n } } \right) t \right) \mathbf { z } _ { 0 } , \quad t \in [ 0 , 1 ] .\tag{4}
$$

Here, $\sigma _ { \operatorname* { m i n } } > 0$ prevents degeneration at early timesteps and stabilizes transport learning.

Given the number of segments K, we partition the time interval [0, 1] into K subintervals $\textstyle \left\{ \left[ { \frac { i } { K } } , { \frac { i + 1 } { K } } \right] \right\} _ { i = 0 } ^ { K - { \bar { 1 } } }$ , and let $i = \ \lfloor K t \rfloor$ denote the segment index. The LCFM objective then penalizes discrepancies in both the predicted trajectory endpoints and the velocity fields:

$$
L _ { \mathrm { L C F M } } ( \boldsymbol { \theta } ) = \mathbb { E } _ { \mathbf { z } _ { 0 } , \mathbf { z } _ { 1 } , t } \left[ \varDelta f _ { \boldsymbol { \theta } } ^ { i } ( \mathbf { z } _ { t } , \mathbf { z } _ { t + \varDelta t } , t ) + \alpha \varDelta v _ { \boldsymbol { \theta } } ^ { i } ( \mathbf { z } _ { t } , \mathbf { z } _ { t + \varDelta t } , t ) \right] ,\tag{5}
$$

where $t \sim \mathcal { U } ( 0 , 1 - \Delta t )$ and the penalty terms are defined as:

$$
\begin{array} { r } { \Delta f _ { \theta } ^ { i } ( \mathbf { z } _ { t } , \mathbf { z } _ { t + \Delta t } , t ) = \left. f _ { \theta } ^ { i } ( \mathbf { z } _ { t } , t ) - \mathrm { s g } \left( f _ { \theta } ^ { i } ( \mathbf { z } _ { t + \Delta t } , t + \Delta t ) \right) \right. ^ { 2 } , } \end{array}\tag{6}
$$

$$
\begin{array} { r } { \Delta v _ { \theta } ^ { i } ( \mathbf { z } _ { t } , \mathbf { z } _ { t + \Delta t } , t ) = \left. v _ { \theta } ^ { i } ( \mathbf { z } _ { t } , t ) - \mathrm { s g } \left( v _ { \theta } ^ { i } ( \mathbf { z } _ { t + \Delta t } , t + \Delta t ) \right) \right. ^ { 2 } , } \end{array}\tag{7}
$$

$$
f _ { \theta } ^ { i } ( { \bf z } _ { t } , t ) = { \bf z } _ { t } + \left( \frac { i + 1 } { K } - t \right) v _ { \theta } ^ { i } ( { \bf z } _ { t } , t ) .\tag{8}
$$

Here, $v _ { \theta } ^ { i } ( { \bf z } _ { t } , t )$ denotes the predicted vector field within the i-th segment, and $f _ { \theta } ^ { i } ( { \bf z } _ { t } , t )$ represents the predicted trajectory output obtained by a single Euler step toward the end of the segment. $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operator, and $\alpha , \Delta t$ are set as hyperparameters.

Unconditional and Conditional Training. Following PMRF [19], we consider both conditional and unconditional flow models for image restoration. For the unconditional flow model, the flow is initialized from the degraded observation, and the transport is defined as:

$$
\mathbf z _ { t } = t \mathbf z _ { 1 } + \left( 1 - \left( 1 - \sigma _ { \operatorname* { m i n } } \right) t \right) \mathbf z _ { 0 } ^ { * } , \quad \mathbf z _ { 0 } ^ { * } = \mathbf z _ { 0 } + \sigma _ { s } \epsilon ,\tag{9}
$$

where the vector field $v _ { \theta } ( z _ { t } , t )$ learns to transport samples from this initialization toward the clean image distribution. Here, $\sigma _ { s }$ controls the standard deviation of the Gaussian noise that alleviates singular mapping between low and high dimensional manifolds.

For comparison, we also consider conditional flow formulation where the vector field learns from the noise distribution conditioned on the degraded input $\mathbf { z } _ { 0 } .$ , formulated as the following:

$$
\mathbf { z } _ { t } = t \mathbf { z } _ { 1 } + ( 1 - ( 1 - \sigma _ { \operatorname* { m i n } } ) t ) \mathbf { z } _ { 0 } ^ { * } , \quad \mathbf { z } _ { 0 } ^ { * } \sim \mathcal { N } ( 0 , I ) ,\tag{10}
$$

where the corresponding velocity field $v _ { \theta } ( \mathbf { z } _ { t } , t , \mathbf { z } _ { 0 } )$ learns to transport samples from the noise distribution toward the target clean image conditioned on $\mathbf { z } _ { 0 }$

## 3.2 Latent Consistency Perceptual Loss

While LCFM ensures geometric consistency in latent space, such a method does not explicitly enforce perceptual realism. To incorporate semantic alignment, we introduce a Latent Consistency Perceptual Loss (LCPL).

External Perceptual Network. Given a predicted latent $\hat { \mathbf { z } } _ { 1 }$ and the groundtruth latent $\mathbf { z } _ { 1 }$ , we adopt E-LatentLPIPS [12] as an external perceptual objective, where the perceptual distance is computed between the corresponding latent representations by applying diferentiable augmentation and measuring feature discrepancies in a pretrained network:

$$
L _ { \mathrm { e x t e r n a l } } ( \mathbf { z } _ { 1 } , \hat { \mathbf { z } } _ { 1 } ) = \mathbb { E } _ { \mathbf { z } _ { 0 } , \mathbf { z } _ { 1 } , t , \mathcal { T } } \left[ \left\| f _ { \mathrm { V G G } } \left( \mathcal { T } ( \mathbf { z } _ { 1 } ) \right) - f _ { \mathrm { V G G } } \left( \mathcal { T } ( \hat { \mathbf { z } } _ { 1 } ) \right) \right\| ^ { 2 } \right] ,\tag{11}
$$

where $f _ { \mathrm { V G G } }$ represents a VGG network pretrained on ImageNet [8] and BAPPS dataset [30], and $\tau$ denotes random diferentiable augmentations implemented in E-LatentLPIPS. We separately train VGGNet for $2 5 6 \times 2 5 6$ resolution for our experiments, as the publicly available pretrained model from E-LatentLPIPS is trained on $5 1 2 \times 5 1 2$ images.

Internal Perceptual Network. Although E-LatentLPIPS [12] provides strong perceptual supervision, it relies on an externally pretrained network and may require additional training for dataset-specific tasks. To reduce dependency on external modules, we additionally adopt an internal perceptual objective function defined by the model itself [3].

Specifically, let $\{ \phi _ { l } ( \mathbf { z } ) \} _ { l = 1 } ^ { L }$ denote intermediate decoder features extracted from latent feature z. The internal perceptual loss is then defined as:

$$
L _ { \mathrm { i n t e r n a l } } ( \mathbf { z } _ { 1 } , \hat { \mathbf { z } } _ { 1 } ) = \mathbb { E } _ { \mathbf { z } _ { 0 } , \mathbf { z } _ { 1 } , t } \left[ w _ { l } \sum _ { l = 1 } ^ { L } \left\| \hat { \phi } _ { l } ( \mathbf { z } _ { 1 } ) - \hat { \phi } _ { l } ( \hat { \mathbf { z } } _ { 1 } ) \right\| ^ { 2 } \right] ,\tag{12}
$$

where $\hat { \phi } _ { l } ( \cdot )$ denotes per-channel normalized features and $\{ w _ { l } \} _ { l = 1 } ^ { L }$ are weight values assigned to each feature layer. Note that the formulation is similar to LPL loss [3] except that binary map masking is omitted for simplicity.

Perceptual Objective for Latent Consistency Training. To incorporate perceptual alignment into consistency learning, we define a Latent Consistency Perceptual Loss (LCPL) that enforces perceptual similarity between adjacent trajectory predictions:

$$
L _ { \mathrm { L C P L } } ( \theta ) = \mathbb { E } _ { \mathbf { z } _ { 0 } , \mathbf { z } _ { 1 } , t } \left[ L _ { \mathrm { p e r c e p } } \left( f _ { \theta } ^ { i } ( \mathbf { z } _ { t } , t ) , f _ { \theta } ^ { i } ( \mathbf { z } _ { t + \Delta t } , t + \Delta t ) \right) \right] ,\tag{13}
$$

where $t \sim \mathcal { U } ( 0 , 1 - \Delta t )$ and $L _ { \mathrm { p e r c e p } }$ can be instantiated as either $L _ { \mathrm { e x t e r n a l } }$ or $L _ { \mathrm { i n t e r n a l } }$ . This objective encourages perceptual consistency of the predicted trajectory across neighboring timesteps in latent space.

Overall Objective. Hence, our final training objective combines flow consistency and perceptual consistency as the following:

$$
L _ { \mathrm { t o t a l } } ( \theta ) = L _ { \mathrm { L C F M } } ( \theta ) + \lambda _ { \mathrm { L C P L } } L _ { \mathrm { L C P L } } ( \theta ) ,\tag{14}
$$

where $\lambda _ { \mathrm { L C P L } }$ controls the strength of perceptual steering relative to the structural transport objective.

## 3.3 Improving Objective with Conflict-Free Gradient Alignment

Diagnosing Gradient Conflict. We compute the gradients of the latent consistency flow matching objective $\nabla _ { \boldsymbol { \theta } } L _ { \mathrm { L C F M } }$ and perceptual objective $\nabla _ { \theta } L _ { \mathrm { L C P L } }$ defined in $\mathrm { E q . ~ ( 1 4 ) }$ . We observe that the gradients between the structural and the perceptual objective are highly noise-level dependent: the two gradients conflict in low log-SNR regimes but become increasingly aligned in high log-SNR regimes. This indicates that the two objectives induce conflicting descent directions in the parameter space, particularly in early transport stages. However, naively summing the gradients implicitly assumes

![](images/d7855cb76d5053f43206d0a98e859bb38273449d2fa0f9de575a37c2870cae89.jpg)  
Fig. 3: Gradient alignment between reconstruction and perceptual objectives. (Left) The reconstruction objective $L _ { \mathrm { L C F M } }$ promotes structural fidelity, while the perceptual objective $L _ { \mathrm { L C P L } }$ encourages perceptual realism toward a semantic target sub-manifold. Our gradient alignment method mitigates conflicts between the gradients and produces a balanced update $g _ { \mathrm { t o t a l } }$ . (Right) In the distortion-perception plane, relying solely on LCFM (orange) leads to blurry outputs, whereas over-optimizing LCPL (blue) severely deviates from the input structure and introduces artifacts. In contrast, our aligned trajectory (green) incorporates perceptual guidance while mitigating conflicting structural updates, allowing the model to move closer to the optimal PD limit.

$$
\left. \nabla _ { \theta } L _ { \mathrm { L C F M } } , \nabla _ { \theta } L _ { \mathrm { L C P L } } \right. \geq 0 ,\tag{15}
$$

which does not hold in practice. When the inner product is negative, the two objectives produce conflicting optimization directions, leading to destructive gradient interference and unstable optimization. This motivates our conflict-free gradient update method combined with noise-aware weighting strategy.

λ-scheduling. Recognizing that the gradient alignment naturally improves in high SNR regimes (near the clean target), we modulate the perceptual influence dynamically. Instead of a static hyperparameter, we define $\lambda _ { \mathrm { L C P L } } ( t )$ as a monotonically increasing function of the timestep t. Conceptually, this ensures that the model establishes a robust, conflict-free structural foundation during the noisy early stages, and progressively unleashes the full power of perceptual steering as the generation refines toward reality.

Conflict-Free Gradient Update. To resolve this destructive interference, we draw inspiration from multi-task gradient surgery [28] but adapt it as an asymmetric orthogonal projection. While the LCFM objective provides the structural transport signal, the LCPL objective serves as perceptual guidance that steers the transport trajectory toward perceptually realistic regions of the target manifold. Therefore, when the two vectors conflict (i.e., $\langle g _ { \mathrm { L C F M } } , g _ { \mathrm { L C P L } } \rangle < 0 )$ , we preserve the perceptual gradient and remove only the component of the structural gradient that conflicts with it.

Denoting the gradients of the two objectives as $g _ { \mathrm { L C F M } } = \nabla _ { \theta } L _ { \mathrm { L C F M } }$ and $g _ { \mathrm { L C P L } } = \nabla _ { \theta } L _ { \mathrm { L C P L } }$ , we update the parameters as the following procedure:

$$
\theta \gets \theta - \eta ( \tilde { g } _ { \mathrm { L C F M } } ( t ) + \lambda _ { \mathrm { L C P L } } ( t ) g _ { \mathrm { L C P L } } ( t ) ) ,\tag{16}
$$

where $\tilde { g } _ { \mathrm { L C F M } }$ is defined as the orthogonal component of $g _ { \mathrm { L C F M } }$ with respect to g<sub>LCPL</sub>:

$$
\tilde { g } _ { \mathrm { L C F M } } ( t ) = g _ { \mathrm { L C F M } } ( t ) - \mathbf { 1 } _ { \{ \langle g _ { \mathrm { L C F M } } , g _ { \mathrm { L C P L } } \rangle < 0 \} } \frac { \langle g _ { \mathrm { L C F M } } , g _ { \mathrm { L C P L } } \rangle } { \| g _ { \mathrm { L C P L } } \| ^ { 2 } } g _ { \mathrm { L C P L } } .\tag{17}
$$

As a result, the perceptual objective remains an efective steering signal throughout training, whereas reconstruction updates are incorporated only when they do not interfere with perceptual optimization. This asymmetric design enables conflict-free multi-objective optimization and leads to improved perceptual restoration performance. We further provide an ablation study comparing alternative projection strategies in our supplementary material.

## 4 Experiments

We evaluate PCFlow on the following tasks: blind face restoration (BFR), superresolution, denoising, inpainting, and colorization. Following previous baselines [6, 19], for BFR, models are trained on FFHQ 512 × 512 dataset and evaluated on CelebA-Test, LFW-Test and CelebAdult. For the remaining tasks, models are trained on FFHQ $2 5 6 \times 2 5 6$ dataset and evaluated on CelebA-Test dataset.

## 4.1 Implementation Details

Training. We train PCFlow using a two-stage procedure. Initially, the model focuses solely on reconstruction quality by setting $\lambda _ { \mathrm { L C P L } } = 0$ for the first 250 training epochs. In the second stage, we enable the perceptual objective and continue training for an additional 250 epochs with the proposed SNR-adaptive weighting, encouraging fine-grained details and perceptual realism. We experiment with different λ-scheduling strategies and report the best-performing configuration. For consistency training, we follow ELIR setting the number of consistency steps $K = 5$ for BFR and $K = 3$ for the remaining tasks, with a fixed time interval $\varDelta t = 0 . 0 5$ . We use a batch size of 128 using AdamW optimizer $( \beta _ { 1 } = 0 . 9$ $\beta _ { 2 } = 0 . 9 9 9 )$ with weight decay 0.02. We use exponential moving average (EMA) with decay rate 0.999 throughout training.

Model Architecture. We employ a pretrained Tiny AutoEncoder [20], a lightweight version of Stable Difusion VAE [9]. The encoder-decoder operates in a 16-channel latent space and contains approximately 2.4M parameters, providing eficient latent representation while maintaining high reconstruction fidelity. We implement convolution-only U-Net architecture from ELIR. The model takes 32 input channels for conditional setting and 16 input channels for unconditional setting. Note that the overall architecture is identical to ELIR, but without the MMSE estimator module, resulting in reduction of 5.5M parameters.

Table 1: Quantitative results on BFR. Quantitative comparison of PCFlow with baselines on blind face restoration (BFR) task. The best and second-best results are highlighted in bold and underlined, respectively. PCFlow achieves a favorable restoration quality with eficient model architecture and fast inference, obtaining state-of-theart FID and NIQE scores for CelebA-Test, and FID for CelebAdult dataset.
<table><tr><td rowspan="3">Model</td><td rowspan="2" colspan="2">Efficiency</td><td colspan="6">CelebA-Test</td><td rowspan="2">LFW</td></tr><tr><td colspan="3">Perceptual Quality</td><td colspan="3">Distortion</td></tr><tr><td>#Params[M]↓</td><td>FPS↑</td><td>FID↓</td><td>NIQE↓</td><td>MUSIQ↑</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓ FID↓</td><td>FID↓</td></tr><tr><td>CodeFormer</td><td>94</td><td>27.13</td><td>52.23</td><td>4.65</td><td>75.55</td><td>24.77</td><td>0.6732</td><td>0.3432</td><td>54.28</td><td>114.34</td></tr><tr><td>GFPGAN(v1.3)</td><td>87.14</td><td>59.73</td><td>45.95</td><td>4.42</td><td>75.29</td><td>24.60</td><td>0.6802</td><td>0.3643</td><td>49.58</td><td>112.31</td></tr><tr><td>VQFRv2</td><td>83.49</td><td>16.97</td><td>46.01</td><td>4.17</td><td>74.40</td><td>22.85</td><td>0.6446</td><td>0.3624</td><td>52.50</td><td>106.83</td></tr><tr><td>Difface (K = 100)</td><td>182.07</td><td>0.78</td><td>37.43</td><td>4.37</td><td>68.31</td><td>24.44</td><td>0.6579</td><td>0.4172</td><td>47.23</td><td>101.11</td></tr><tr><td>DiffBIR (K = 50)</td><td>1666.93</td><td>0.38</td><td>46.23</td><td>4.21</td><td>75.27</td><td>23.27</td><td>0.6379</td><td>0.3876</td><td>46.03</td><td>111.84</td></tr><tr><td>ResShift (K = 4)</td><td>196.70</td><td>6.85</td><td>43.60</td><td>4.37</td><td>72.16</td><td>25.32</td><td>0.6965</td><td>0.3435</td><td>54.23</td><td>109.04</td></tr><tr><td>PMRF (K = 25)</td><td>182.75</td><td>0.57</td><td>37.22</td><td>4.12</td><td>70.36</td><td>25.85</td><td>0.7098</td><td>0.3470</td><td>49.98</td><td>104.44</td></tr><tr><td>ELIR (K = 5)</td><td>37.52</td><td>33.11</td><td>44.64</td><td>5.26</td><td>67.24</td><td>25.56</td><td>0.7030</td><td>0.3735</td><td>53.19</td><td>105.55</td></tr><tr><td>PCFlow (Ours)</td><td>32.02</td><td>42.62</td><td>35.89</td><td>3.95</td><td>70.35</td><td>24.44</td><td>0.6680</td><td>0.3850</td><td>50.68</td><td>98.85</td></tr></table>

Latent Perceptual Network. For the internal perceptual network, we extract features from the decoder’s intermediate representations, including the mid-block, three upsampling stages, and the final output layer. The contribution of each feature level is weighted according to its spatial resolution, except for the final output layer, i.e., $w _ { l } \propto 2 ^ { - r _ { l } }$ , where r denotes the resolution level of feature l. The weights are subsequently normalized to sum to one. For the external perceptual network, to compute the perceptual objective within our customized latent space, we separately train the model following the procedure from E-LatentLPIPS. In addition, we adopt batch normalization which empirically demonstrates more stable training compared to group normalization.

## 4.2 Quantitative Results

Blind Face Restoration. Quantitative results on blind face restoration (BFR) are reported in Tab. 1. PCFlow achieves state-of-the-art perceptual quality, obtaining the best FID and NIQE on CelebA-Test and the best FID on CelebAdult, while requiring only 32M parameters and five sampling steps $( K = 5 )$ . Compared to ELIR, PCFlow uses fewer parameters (32M vs. 37.5M) and achieves 1.29× higher inference speed (42.62 vs. 33.11 FPS), while substantially improving FID (35.89 vs. 44.64) on CelebA-Test. Furthermore, despite being significantly smaller and faster than difusion-based baselines such as PMRF (183M parameters, 0.57 FPS), PCFlow outperforms in perceptual quality metrics while delivering over 75× higher throughput. These results demonstrate that PCFlow provides a favorable trade-of between restoration quality and computational efficiency, achieving strong perceptual restoration performance under a practical few-step inference regime.

Table 2: Quantitative results on the remaining restoration tasks. Quantitative comparison of PCFlow with baselines ELIR and PMRF. The best and second-best results are highlighted in bold and underlined, respectively. PCFlow consistently outperforms ELIR in FID, despite using only 21M parameters compared to ELIR (27M) and PMRF (176M), demonstrating favorable perceptual quality with eficient model architecture and fast inference.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Model</td><td>Efficiency</td><td>Perceptual Quality</td><td colspan="3">Distortion</td></tr><tr><td>#Params[M]↓</td><td>FID↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td rowspan="3">Super Resolution ELIR</td><td>PMRF (K = 25)</td><td>176</td><td>44.64</td><td>24.40</td><td>0.6708</td><td>0.2991</td></tr><tr><td></td><td>27</td><td>49.25</td><td>23.57</td><td>0.6439</td><td>0.3299</td></tr><tr><td>PCFlow (Ours)</td><td>21</td><td>45.50</td><td>23.38</td><td>0.6512</td><td>0.3328</td></tr><tr><td rowspan="3">Denoising</td><td>PMRF (K = 25)</td><td>176</td><td>44.35</td><td>27.92</td><td>0.7782</td><td>0.2401</td></tr><tr><td>ELIR</td><td>27</td><td>47.70</td><td>26.67</td><td>0.7521</td><td>0.2619</td></tr><tr><td>PCFlow (Ours)</td><td>21</td><td>45.42</td><td>26.26</td><td>0.7480</td><td>0.2800</td></tr><tr><td rowspan="3">Inpainting</td><td>PMRF (K = 25)</td><td>176</td><td>42.88</td><td>26.19</td><td>0.7383</td><td>0.2626</td></tr><tr><td>ELIR</td><td>27</td><td>47.82</td><td>24.85</td><td>0.7105</td><td>0.2840</td></tr><tr><td>PCFlow (Ours)</td><td>21</td><td>45.50</td><td>24.92</td><td>0.7195</td><td>0.2936</td></tr><tr><td rowspan="3">Colorization</td><td>PMRF (K = 25)</td><td>176</td><td>42.61</td><td>23.47</td><td>0.7122</td><td>0.3463</td></tr><tr><td>ELIR</td><td>27</td><td>51.72</td><td>23.15</td><td>0.7113</td><td>0.3587</td></tr><tr><td>PCFlow (Ours)</td><td>21</td><td>45.21</td><td>22.18</td><td>0.7499</td><td>0.3596</td></tr></table>

Other Restoration Tasks. We further provide quantitative results for the remaining restoration tasks, including super-resolution, denoising, inpainting, and colorization, in Tab. 2. PCFlow consistently improves over ELIR in FID across all four image restoration tasks. These results challenge the prevailing paradigm that prioritizes explicit posterior mean estimation followed by transport refinement. Instead, we show that directly learning the conditional transport from degraded observations, combined with perceptual supervision as a steering signal, is suficient to achieve superior distortion-perception trade-ofs under a few-step regime. Notably, the improvements are consistent across diverse degradation types, suggesting that the proposed formulation generalizes well beyond a specific restoration setting.

## 4.3 Qualitative Results

Blind Face Restoration. Fig. 1 presents qualitative comparisons on blind face restoration (BFR). Compared with ELIR, PCFlow consistently restores sharper facial structures, including hair, beard, and facial contours, while maintaining more natural textures and local contrast. These improvements lead to visually more faithful reconstructions without introducing noticeable artifacts. Compared with difusion-based methods, PCFlow produces visually balanced reconstructions without the tendency toward over-sharpened textures observed in DifBIR, while achieving visual quality competitive with DifFace and PMRF. Notably, these results are obtained using only five inference steps, demonstrating that the proposed framework efectively achieves high perceptual restoration quality with highly eficient inference.

![](images/0ef93bd3265e75ff619cd88a15063df515c211789c2b717b994584c9a3ebe918.jpg)  
Fig. 4: Qualitative comparison across image restoration tasks. From left to right: Input (LQ), ELIR, PMRF(K = 3), PMRF(K = 25), PCFlow (Ours), and ground truth image (HQ). PCFlow produces visually sharp and coherent reconstructions even with a small number of integration steps K = 3, establishing an efective training framework that provides a favorable distortion-perception trade-of.

Other Restoration Tasks. Fig. 4 presents qualitative comparisons across four image restoration tasks. Compared to ELIR, PCFlow consistently generates visually more coherent and realistic reconstructions. In particular, ELIR tends to generate slightly over-smoothed facial structures, while PCFlow restores sharper facial boundaries and more consistent skin textures, leading to perceptually more faithful reconstructions. Note that PCFlow can occasionally generate subtle artifacts around high-frequency regions such as the eyes. Compared to PMRF with a larger number of integration steps K = 25, PCFlow attains strong perceptual quality while operating in a significantly more eficient few-step regime. These observations align with the quantitative trends in Tab. 2, where PCFlow improves perceptual metrics over ELIR and achieves a favorable distortion-perception trade-of, despite requiring substantially fewer transport steps and model parameters than PMRF(K = 25).

## 4.4 Ablation Studies

We conduct several ablation studies and analyze the contribution of each component in our proposed PCFlow training objective.

Table 3: Validation of the Preheating Period. The combination of preheating period, gradient surgery, and linear warmup schedule demonstrates the best overall performance in both distortion and perceptual metrics on super-resolution task.
<table><tr><td>Setting</td><td>LLCPL</td><td></td><td>Preheating λ-scheduling</td><td>FID↓</td><td>PSNR↑ SSIM↑ LPIPS↓</td></tr><tr><td>LLCFM only</td><td></td><td></td><td></td><td>46.10</td><td>23.35 0.6476</td></tr><tr><td>w/o Gradient Alignment epoch 0</td><td></td><td>X</td><td>linear warmup</td><td>46.40</td><td>0.3373 23.32 0.6484 0.3363</td></tr><tr><td>w/ Gradient Alignment</td><td>epoch 0</td><td>X</td><td>linear warmup 46.26</td><td></td><td>23.31 0.6480 0.3364</td></tr><tr><td>PCFlow</td><td>epoch 250</td><td>√</td><td>linear warmup 45.50</td><td></td><td>23.38 0.6512 0.3328</td></tr></table>

Table 4: Ablation on PCFlow components. Quantitative results for adding each PCFlow component on colorization task. The best performance under conditional flow formulation (B–E) is highlighted in bold, while the second-best are underlined.
<table><tr><td></td><td>FID↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Base (A)</td><td>56.27</td><td>23.11</td><td>0.7651</td><td>0.3675</td></tr><tr><td>+ Conditional (B)</td><td>47.21</td><td>21.88</td><td>0.7220</td><td>0.3845</td></tr><tr><td>+ Encoder Fine-Tuning (C)</td><td>45.97</td><td>22.19</td><td>0.7465</td><td>0.3654</td></tr><tr><td>+ Perceptual Objective (LLcPL) (D)</td><td>45.78</td><td>22.27</td><td>0.7530</td><td>0.3551</td></tr><tr><td>+ Conflict-Free Gradient Alignment (E)</td><td>45.21</td><td>22.18</td><td>0.7499</td><td>0.3596</td></tr></table>

Preheating Period. Tab. 3 evaluates the contributions of the proposed preheating period. The preheating period corresponds to the first 250 epochs, during which only the reconstruction objective is optimized while perceptual supervision is disabled. Jointly training both distortion and perception objectives from the beginning yields inferior results compared to PCFlow, suggesting that the initial preheating stage is essential for establishing a stable coarse-to-fine transport trajectory before introducing perceptual supervision. In addition, applying conflictfree gradient alignment from epoch 0 consistently improves FID over naive joint optimization (46.40 → 46.26), indicating that mitigating optimization conflicts between reconstruction and perceptual objectives is beneficial. Combining both components yields the best performance across all metrics.

Individual Training Components. Tab. 4 shows quantitative results for progressively adding each of the PCFlow components. For the baseline unconditional model (config A), we set $\sigma _ { \mathrm { m i n } }$ to 0.05 for super-resolution, 0.01 for denoising and colorization, and 0.025 for inpainting task. Training with conditional flow matching formulation (config B) improves FID by a substantial margin from 56.27 to 47.21. Fine-tuning the encoder along with the vector field (config C) further enhances model performance in both distortion and perceptual metrics. Adding the perceptual objective (config D) improves perceptual quality while maintaining reconstruction fidelity, efectively steering the model towards perceptually sharp data manifolds. Finally, incorporating the proposed conflict-free gradient alignment (config E) achieves the lowest FID of 45.21, pushing the model closer to the optimal distortion–perception tradeof. We observe that similar trends consistently hold across the remaining image restoration tasks.

Table 5: Ablation on λ-scheduling. Comparison of diferent scheduling methods for λ<sub>LCPL</sub>. Increasing the perceptual weight via linear warmup schedule consistently achieves the best model performance in FID across four image restoration tasks.
<table><tr><td>Task</td><td>λ-scheduling</td><td>FID↓</td><td>LPIPS↓</td></tr><tr><td rowspan="3">Super-Resolution</td><td>constant</td><td>46.18</td><td>0.3300</td></tr><tr><td>linear</td><td>45.74</td><td>0.3326</td></tr><tr><td>linear warmup</td><td>45.50</td><td>0.3328</td></tr><tr><td rowspan="3">Denoising</td><td>constant</td><td>46.12</td><td>0.2790</td></tr><tr><td>linear</td><td>45.48</td><td>0.2814</td></tr><tr><td>linear warmup</td><td>45.42</td><td>0.2800</td></tr><tr><td rowspan="3">Inpainting</td><td>constant</td><td>46.25</td><td>0.2918</td></tr><tr><td>linear</td><td>45.56</td><td>0.2944</td></tr><tr><td>linear warmup</td><td>45.50</td><td>0.2936</td></tr><tr><td rowspan="3">Colorization</td><td>constant</td><td>45.78</td><td>0.3551</td></tr><tr><td>linear</td><td>45.45</td><td>0.3611</td></tr><tr><td>linear warmup</td><td>45.21</td><td>0.3596</td></tr></table>

λ-scheduling. We further investigate the role of the perceptual weight λ as a function of the difusion timestep t. Tab. 5 shows quantitative comparison of diferent scheduling methods, including constant, linear, and linear warmup schedules. We observe that linear warmup schedule consistently achieves the best FID across all four image restoration tasks, demonstrating the benefit of gradually introducing perceptual supervision during transport. We hypothesize that early transport steps primarily require stable global alignment between degraded observations and the target distribution, where excessive perceptual supervision may introduce unnecessary optimization conflicts. In contrast, as the transport progresses closer to the true image manifold, perceptual supervision becomes more beneficial for recovering fine structures and visually realistic details.

## 4.5 Analysis on Gradient Conflict

We analyze the interaction between reconstruction objective $\boldsymbol { \mathcal { L } } _ { \mathrm { L C F M } }$ and perceptual objective $\mathcal { L } _ { \mathrm { L C P L } }$ by comparing the results from standard joint optimization with our proposed gradient alignment method. As illustrated in Fig. 5, the cosine similarity heatmap reveals substantial regions of negative alignment (blue) under the standard update, indicating that the two objectives frequently produce conflicting gradients. Such conflicts can hinder stable optimization and slow down convergence. In contrast, after applying the proposed conflict-free update, the gradients exhibit significantly improved alignment, with fewer negatively correlated regions and more consistent positive interactions.

![](images/41094bed48ecd6478bc53520681df0b2f4f15cb2ba4e849f9ec5a0023c275202.jpg)  
Fig. 5: Analysis of gradient interaction between flow matching and perceptual objectives. Cosine similarity maps between the LCFM and LCPL gradients as a function of training epochs and log-SNR, shown without (top) and with (bottom) the proposed conflict-free update. Our proposed method significantly mitigates gradient conflict, particularly in low-SNR regions, and promotes consistent optimization throughout the sampling process.

## 5 Conclusion

In this paper, we propose PCFlow, a unified latent transport framework for image restoration that jointly optimizes distortion and perceptual quality. By formulating the problem with latent consistency flow matching objective, our method directly learns the transport between degraded and clean images, hence enabling eficient few-step inference. To enhance perceptual realism, we additionally introduce a latent consistency perceptual objective that enforces semantic alignment along the transport trajectory. Furthermore, motivated by our analysis of the interaction between flow consistency and perceptual objectives, we propose a conflict-free gradient update method that enables the perceptual objective to steer optimization while removing conflicting structural gradient components. Extensive experiments demonstrate that PCFlow establishes a more favorable distortion–perception trade-of frontier compared to previous baselines across various image restoration tasks, while requiring only a few inference steps and maintaining an eficient model architecture.

## Acknowledgements

This work was partly supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP)-ITRC (Information Technology Research Center) grant funded by the Korea government (MSIT) (IITP-2026- RS-2024-00436857, 50%) and by IITP grant funded by the Korea government (MSIT) (No. RS-2019-II190079, Artificial Intelligence Graduate School Program (Korea University), 50%).

## References

1. Adrai, T., Ohayon, G., Elad, M., Michaeli, T.: Deep optimal transport: A practical algorithm for photo-realistic image restoration. Advances in Neural Information Processing Systems 36, 61777–61791 (2023)

2. Albergo, M.S., Vanden-Eijnden, E.: Building normalizing flows with stochastic interpolants. In: The Eleventh International Conference on Learning Representations (2023)

3. Berrada, T., Astolfi, P., Hall, M., Havasi, M., Benchetrit, Y., Romero-Soriano, A., Alahari, K., Drozdzal, M., Verbeek, J.: Boosting latent difusion with perceptual objectives. In: The Thirteenth International Conference on Learning Representations (2025)

4. Blau, Y., Michaeli, T.: The perception-distortion tradeof. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 6228–6237 (2018)

5. Chung, H., Kim, J., Mccann, M.T., Klasky, M.L., Ye, J.C.: Difusion posterior sampling for general noisy inverse problems. In: The Eleventh International Conference on Learning Representations (2023)

6. Cohen, E., Achituve, I., Diamant, I., Netzer, A., Habi, H.V.: Eficient image restoration via latent consistency flow matching. In: 36th British Machine Vision Conference 2025, BMVC 2025, Shefield, UK, November 24-27, 2025. BMVA (2025)

7. Dao, Q., Phung, H., Nguyen, B., Tran, A.: Flow matching in latent space. arXiv preprint arXiv:2307.08698 (2023)

8. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A large-scale hierarchical image database. In: 2009 IEEE Conference on Computer Vision and Pattern Recognition. pp. 248–255 (2009)

9. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: Forty-first international conference on machine learning (2024)

10. Freirich, D., Michaeli, T., Meir, R.: A theory of the distortion-perception tradeof in wasserstein space. Advances in Neural Information Processing Systems 34, 25661– 25672 (2021)

11. Gu, Y., Wang, X., Xie, L., Dong, C., Li, G., Shan, Y., Cheng, M.M.: Vqfr: Blind face restoration with vector-quantized dictionary and parallel decoder. In: European Conference on Computer Vision. pp. 126–143. Springer (2022)

12. Kang, M., Zhang, R., Barnes, C., Paris, S., Kwak, S., Park, J., Shechtman, E., Zhu, J.Y., Park, T.: Distilling difusion models into conditional gans. In: European Conference on Computer Vision. pp. 428–447. Springer (2024)

13. Kawar, B., Elad, M., Ermon, S., Song, J.: Denoising difusion restoration models. Advances in neural information processing systems 35, 23593–23606 (2022)

14. Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., Aitken, A., Tejani, A., Totz, J., Wang, Z., et al.: Photo-realistic single image superresolution using a generative adversarial network. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4681–4690 (2017)

15. Lin, S., Yang, X.: Difusion model with perceptual loss. arXiv preprint arXiv:2401.00110 (2023)

16. Lin, X., He, J., Chen, Z., Lyu, Z., Dai, B., Yu, F., Qiao, Y., Ouyang, W., Dong, C.: Difbir: Toward blind image restoration with generative difusion prior. In: European conference on computer vision. pp. 430–448. Springer (2024)

17. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. In: The Eleventh International Conference on Learning Representations (2023)

18. Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer data with rectified flow. In: The Eleventh International Conference on Learning Representations (2023)

19. Ohayon, G., Michaeli, T., Elad, M.: Posterior-mean rectified flow: Towards minimum MSE photo-realistic image restoration. In: The Thirteenth International Conference on Learning Representations (2025)

20. Platen, P.V., Patil, S., Lozhkov, A., Cuenca, P., Lambert, N., Rasul, K., Davaadorj, M., Nair, D., Paul, S., Berman, W., Xu, Y., Liu, S., Wolf, T.: Difusers: State-ofthe-art difusion models (2022)

21. Saharia, C., Ho, J., Chan, W., Salimans, T., Fleet, D.J., Norouzi, M.: Image superresolution via iterative refinement. IEEE transactions on pattern analysis and machine intelligence 45(4), 4713–4726 (2022)

22. Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. In: International Conference on Learning Representations (2015)

23. Song, J., Vahdat, A., Mardani, M., Kautz, J.: Pseudoinverse-guided difusion models for inverse problems. In: International Conference on Learning Representations (2023)

24. Wang, X., Li, Y., Zhang, H., Shan, Y.: Towards real-world blind face restoration with generative facial prior. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9168–9178 (2021)

25. Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., Qiao, Y., Change Loy, C.: Esrgan: Enhanced super-resolution generative adversarial networks. In: Proceedings of the European conference on computer vision (ECCV) workshops. pp. 0–0 (2018)

26. Wang, Y., Yu, J., Zhang, J.: Zero-shot image restoration using denoising difusion null-space model. In: The Eleventh International Conference on Learning Representations (2023)

27. Yang, L., Zhang, Z., Zhang, Z., Liu, X., Liu, J., Xu, M., Meng, C., Ermon, S., Zhang, W., CUI, B.: Consistency flow matching: Defining straight flows with velocity consistency (2025)

28. Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., Finn, C.: Gradient surgery for multi-task learning. Advances in neural information processing systems 33, 5824–5836 (2020)

29. Yue, Z., Loy, C.C.: Diface: Blind face restoration with difused error contraction. IEEE Transactions on Pattern Analysis and Machine Intelligence 46(12), 9991– 10004 (2024)

30. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586–595 (2018)

31. Zhou, S., Chan, K., Li, C., Loy, C.C.: Towards robust blind face restoration with codebook lookup transformer. Advances in Neural Information Processing Systems 35, 30599–30611 (2022)

32. Zhu, Y., Zhao, W., Li, A., Tang, Y., Zhou, J., Lu, J.: Flowie: Eficient image enhancement via rectified flow. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13–22 (2024)

33. Zhu, Y., Zhang, K., Liang, J., Cao, J., Wen, B., Timofte, R., Van Gool, L.: Denoising difusion models for plug-and-play image restoration. In: Proceedings of the

IEEE/CVF conference on computer vision and pattern recognition. pp. 1219–1229 (2023)

# Supplementary Material for Flow Straight to Reality: Perceptually Consistent Flow Matching for Eficient Image Restoration

Sangwoo Jo<sup>1</sup> , Donggeun Ko<sup>2</sup> , Jayeon Kang<sup>1</sup> , Youngsang Kwak<sup>2</sup> , Jaehwa Kwak<sup>2</sup>, and Sungjoon Choi<sup>1</sup>

<sup>1</sup> Korea University, Seoul, South Korea

{jasonjo97, nature0213, sungjoonc}@korea.ac.kr

<sup>2</sup> Aim Future, Seoul, South Korea

{sean.ko, youngsang.kwak, jaehwa.kwak}@aimfuture.ai

## A Additional Implementation Details

## A.1 Hyperparameters

We provide training hyperparameters for our proposed PCFlow in Tab. 1. Note that although PCFlow can be trained within a single unified procedure, we separate the stages in practice to allow additional flexibility for hyperparameter tuning. For the convolution-only U-Net backbone, we set the number of channels for the downsampling blocks as [128, 256, 256, 512] and number of mid blocks to 3 for blind face restoration (BFR) and 1 for the remaining tasks, following ELIR [6]. For the LCPL objective, we use decoder features as a source of perceptual supervision, similar to the LPL loss formulation [3]. We set the perceptual objective coeficient $\lambda _ { \mathrm { L C P I } }$ to increase over timestep t, i.e.,

$$
\lambda _ { \mathrm { L C P L } } ( t ) = \lambda _ { \mathrm { m i n } } + \mathbb { I } _ { t \geq t _ { \mathrm { m i n } } } \big ( \lambda _ { \mathrm { m a x } } - \lambda _ { \mathrm { m i n } } \big ) \left( \frac { t - t _ { \mathrm { m i n } } } { 1 - t _ { \mathrm { m i n } } } \right) ,\tag{1}
$$

which progressively shifts the trajectory towards the target manifold. We set $\lambda _ { \operatorname* { m i n } } = 0 , \lambda _ { \operatorname* { m a x } } = 0 . 5$ , and $t _ { \mathrm { m i n } } = 0 . 5$ for linear warmup schedule.

## A.2 Training Details for E-LatentLPIPS

In this section, we provide training details for implementing E-LatentLPIPS [12] which serves as an external perceptual supervision in our ablation study. We first train a VGG network [22] on ImageNet dataset [8] with resolution of 256×256 to learn general feature representations in the latent space. We evaluate two variants of the VGG16 backbone with group normalization and batch normalization. We empirically observe that the batch normalization yields higher accuracy, and hence select it as our backbone for the external perceptual network. We subsequently fine-tune the network on BAPPS dataset [30] to align the features with human perceptual similarity judgments. Overall, as illustrated in Tab. 2, our implementation achieves similar performance to the original work [12], despite being trained at resolution of 256×256 and latent space of the Tiny AutoEncoder [20], which is a compressed version of Stable Difusion VAE [9].

Table 1: Training hyperparameters for the two training stages of PCFlow. The first column corresponds to blind face restoration tasks (BFR), while the second column corresponds to the remaining four restoration tasks, including super-resolution, denoising, inpainting, and colorization.  
(a) PCFlow with LCFM objective only.
<table><tr><td>Hyperparameters</td><td>Blind Face Restoration</td><td>Other Restoration tasks</td></tr><tr><td>Parameters</td><td>32M</td><td>21M</td></tr><tr><td>Euler steps (M)</td><td>5</td><td>3</td></tr><tr><td>CFM segments (K)</td><td>5</td><td>3</td></tr><tr><td>CFM  $\varDelta t$ </td><td>0.05</td><td>0.05</td></tr><tr><td>CFM α</td><td>0.001</td><td>0.001</td></tr><tr><td>σmin</td><td> $1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>Training epochs</td><td>250</td><td>250</td></tr><tr><td>Batch size</td><td>32</td><td>128</td></tr><tr><td>Image dimension</td><td> $3 \times 5 1 2 \times 5 1 2$ </td><td> $3 \times 2 5 6 \times 2 5 6$ </td></tr><tr><td>Latent dimension</td><td> $1 6 \times 6 4 \times 6 4$ </td><td> $1 6 \times 3 2 \times 3 2$ </td></tr><tr><td>Precision</td><td>bfloat16 mixed</td><td>bfloat16 mixed</td></tr><tr><td>Training hardware</td><td> $1 \times \mathrm { ~ H 1 0 0 ~ } 8 0 \mathrm { G B }$ </td><td> $1 \times \mathrm { A 1 0 0 ~ 8 0 G B }$ </td></tr><tr><td>Training time</td><td>2 days</td><td>1 day</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>AdamW betas</td><td>(0.9, 0.999)</td><td>(0.9, 0.999)</td></tr><tr><td>Weight decay</td><td>0.02</td><td>0.02</td></tr><tr><td>EMA decay</td><td>0.999</td><td>0.999</td></tr></table>

(b) PCFlow with perceptual objective (LCPL).
<table><tr><td>Hyperparameters</td><td>Blind Face Restoration Other Restoration tasks</td><td></td></tr><tr><td>Parameters</td><td>32M</td><td>21M</td></tr><tr><td>Euler steps (M)</td><td>5</td><td>3</td></tr><tr><td>CFM segments (K)</td><td>5</td><td>3</td></tr><tr><td>CFM ∆t</td><td>0.05</td><td>0.05</td></tr><tr><td>CFM α</td><td>0.001</td><td>0.001</td></tr><tr><td> $\sigma _ { \mathrm { m i n } }$ </td><td>10−5</td><td>10⁻5</td></tr><tr><td>Training epochs</td><td>250</td><td>250</td></tr><tr><td>Batch size</td><td>32</td><td>128</td></tr><tr><td>Image dimension</td><td> $3 \times 5 1 2 \times 5 1 2$ </td><td> $3 \times 2 5 6 \times 2 5 6$ </td></tr><tr><td>Latent dimension</td><td> $1 6 \times 6 4 \times 6 4$ </td><td> $1 6 \times 3 2 \times 3 2$ </td></tr><tr><td>Precision</td><td>bfloat16 mixed</td><td>bfloat16 mixed</td></tr><tr><td>Training hardware</td><td>1× H100 80GB</td><td>1× A100 80GB</td></tr><tr><td>Training time</td><td>2.5 days</td><td>2 days</td></tr><tr><td>Optimizer</td><td>AdamW</td><td> $\mathrm { A d a m W }$ </td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>AdamW betas</td><td>(0.9, 0.999)</td><td>(0.9, 0.999)</td></tr><tr><td>Weight decay</td><td>0.02</td><td>0.02</td></tr><tr><td>EMA decay</td><td>0.999</td><td>0.999</td></tr><tr><td>wl</td><td>[1, 0.5, 0.25, 0.125, 1]</td><td> $[ 1 , 0 . 5 , 0 . 2 5 , 0 . 1 2 5 , 1 ]$ </td></tr><tr><td> $\lambda _ { \mathrm { L C P L } } ( t )$ </td><td>linear warmup</td><td>linear warmup</td></tr><tr><td> $\lambda _ { \mathrm { m i n } }$ </td><td>0</td><td>0</td></tr><tr><td> $\lambda _ { \mathrm { m a x } }$ </td><td>0.5</td><td>0.5</td></tr><tr><td> $t _ { \mathrm { m i n } }$ </td><td>0.5</td><td>0.5</td></tr></table>

Table 2: ImageNet and BAPPS Classification Scores. Classification accuracy on ImageNet and BAPPS datasets within the Tiny AutoEncoder latent space. Our reimplementation achieves similar performance to the original E-LatentLPIPS paper.
<table><tr><td>Dataset</td><td>Type</td><td>Ours</td><td>E-LatentLPIPS (Original)</td></tr><tr><td>ImageNet</td><td>VGG-BN</td><td>67.82</td><td>68.26</td></tr><tr><td rowspan="3">BAPPS</td><td>Traditional</td><td>76.62</td><td>74.29</td></tr><tr><td>CNN</td><td>82.43</td><td>81.99</td></tr><tr><td>Real</td><td>63.79</td><td>63.21</td></tr></table>

## B Additional Results

## B.1 Further Ablation Studies

Unconditional vs. Conditional Flow Models. We consider both unconditional and conditional flow formulations. We empirically find that the conditional flow model consistently achieves better FID across all restoration tasks, as shown in Tab. 3. Unlike PMRF, which reports limited gains from conditional transport, we observe that initializing from random noise allows the model to fully leverage its generative capability. From an optimal transport perspective, the unconditional formulation may restrict the transport trajectory to remain close to the degraded manifold, whereas the conditional formulation enables transport with generative prior resulting in higher perceptual realism.

Table 3: Ablation on Unconditional vs. Conditional Flow Models. Learning a transport from random noise with degraded image as a conditional input leads to substantial improvements in FID over unconditional formulation.
<table><tr><td>Task</td><td>Conditional</td><td>FID↓</td><td>LPIPS↓</td></tr><tr><td>Super-Resolution</td><td>x √</td><td>63.69 46.42</td><td>0.3509 0.3396</td></tr><tr><td>Denoising</td><td>x √</td><td>48.50 45.70</td><td>0.2866 0.3025</td></tr><tr><td>Inpainting</td><td>x √</td><td>60.47 47.04</td><td>0.3677 0.3611</td></tr><tr><td>Colorization</td><td>x √</td><td>56.27 47.21</td><td>0.3675 0.3845</td></tr></table>

Encoder Fine-Tuning. We compare freezing the pretrained encoder against jointly fine-tuning it together with the vector field network. As shown in Tab. 4, fine-tuning the encoder consistently improves both reconstruction and perceptual quality across all image restoration tasks. For instance, for super-resolution, FID and LPIPS decrease from 46.42 to 46.10 and from 0.3396 to 0.3373, respectively, while PSNR and SSIM increase from 23.30 to 23.35 and from 0.6462 to 0.6476, respectively. Similar improvements are observed in other image restoration tasks. We hypothesize that, since the encoder is pretrained on high-quality images, it requires additional training to fully learn the representation of degraded inputs. Joint optimization therefore allows the encoder to adapt to the degradation distribution, resulting in improved alignment between the latent representation and the restoration dynamics.

Table 4: Ablation on Encoder Fine-Tuning (FT). Comparison of PCFlow and with and without encoder fine-tuning. Fine-tuning encoder along with the vector field shows better performance in both distortion and perception metrics across all four image restoration tasks.
<table><tr><td>Task</td><td>Encoder FT</td><td>FID↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td rowspan="2">Super-Resolution</td><td>w/o</td><td>46.42</td><td>23.30</td><td>0.6462</td><td>0.3396</td></tr><tr><td>w/</td><td>46.10</td><td>23.35</td><td>0.6476</td><td>0.3373</td></tr><tr><td rowspan="2">Denoising</td><td>w/o</td><td>45.70</td><td>25.50</td><td>0.7215</td><td>0.3025</td></tr><tr><td>w/</td><td>45.29</td><td>26.18</td><td>0.7437</td><td>0.2857</td></tr><tr><td rowspan="2">Inpainting</td><td>w/o</td><td>47.04</td><td>22.33</td><td>0.6318</td><td>0.3611</td></tr><tr><td>w/</td><td>45.64</td><td>24.82</td><td>0.7150</td><td>0.2990</td></tr><tr><td rowspan="2">Colorization</td><td>w/o</td><td>47.21</td><td>21.88</td><td>0.7220</td><td>0.3845</td></tr><tr><td>w/</td><td>45.97</td><td>22.19</td><td>0.7465</td><td>0.3654</td></tr></table>

Perceptual Network. We compare the efect of using external and internal perceptual networks for perceptual supervision, as summarized in Tab. 5. Note that external network corresponds to our reimplementation of E-LatentLPIPS, and the internal network corresponds to using intermediate and final decoder features. Using the external perceptual network improves distortion metrics such as PSNR and SSIM. However, applying the proposed gradient alignment with the following external network does not lead to further improvements and instead yields the model to deviate from the optimal trajectory. This suggests that perceptual supervision derived from the VGG network trained on an external dataset may not be suitable for restoration dynamics.

In contrast, defining the perceptual objective based on internal decoder features leads to improved perceptual alignment with the restoration objective. In particular, incorporating the proposed conflict-free gradient alignment further reduces FID from 46.01 to 45.50, yielding the best perceptual quality among all configurations. The following results suggest that internal perceptual representations are better aligned with the latent restoration dynamics, and that the proposed gradient alignment strategy further stabilizes the interaction between reconstruction and perceptual objectives. Hence, we adopt the internal perceptual network with conflict-free gradient alignment as our final configuration.

Table 5: Ablation on Perceptual Network. Comparison between external perceptual supervision (E-LatentLPIPS) and internal decoder feature supervision for superresolution task. Conflict-free gradient alignment further improves perceptual quality when applied to the internal perceptual network, achieving the best FID.
<table><tr><td>Perceptual Network</td><td>FID↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>External</td><td>46.03</td><td>23.58</td><td>0.6604</td><td>0.3223</td></tr><tr><td>+ Conflict-Free Gradient Alignment</td><td>46.15</td><td>23.41</td><td>0.6514</td><td>0.3299</td></tr><tr><td>Internal</td><td>46.01</td><td>23.31</td><td>0.6515</td><td>0.3310</td></tr><tr><td>+ Conflict-Free Gradient Alignment</td><td>45.50</td><td>23.38</td><td>0.6512</td><td>0.3328</td></tr></table>

Perceptual vs. Structural Gradient Projection. We further compare two conflict-resolution strategies: projecting the perceptual gradient $( g _ { \mathrm { { L C P L } } } \to \tilde { g } _ { \mathrm { { L C P L } } } )$ and projecting the structural gradient $\left( g _ { \mathrm { L C F M } } \to \tilde { g } _ { \mathrm { L C F M } } \right)$ . As shown in Tab. 6, projecting the structural gradient generally yields better FID across all restoration tasks. These results support our design choice of treating the perceptual objective as a steering signal and resolving optimization conflicts through structural gradient projection. We hypothesize that perceptual supervision primarily guides the transport trajectory toward perceptually realistic regions of the target manifold, whereas the structural objective provides reconstruction-oriented updates. In addition, we observe that the perceptual gradient exhibits a larger magnitude than the structural gradient, which may further contribute to the improved optimization stability of structural-gradient projection.

Table 6: Ablation on Gradient Projection. Quantitative comparison between projecting the perceptual gradient and projecting the structural gradient during conflictfree gradient alignment. Projecting the structural gradient g<sub>LCFM</sub> → g˜<sub>LCFM</sub> generally yields better model performance in FID across restoration tasks.
<table><tr><td>Task</td><td>Gradient Projection</td><td>FID↓</td><td>LPIPS↓</td></tr><tr><td rowspan="2">Super-Resolution</td><td> $g _ { \mathrm { L C P L } }  \tilde { g } _ { \mathrm { L C P L } }$ </td><td>45.66</td><td>0.3317</td></tr><tr><td> $g _ { \mathrm { L C F M } } \to \tilde { g } _ { \mathrm { L C F M } }$ </td><td>45.50</td><td>0.3328</td></tr><tr><td rowspan="2">Denoising</td><td> $g _ { \mathrm { L C P L } }  \tilde { g } _ { \mathrm { L C P L } }$ </td><td>45.42</td><td>0.2800</td></tr><tr><td> $g _ { \mathrm { L C F M } } \to \tilde { g } _ { \mathrm { L C F M } }$ </td><td>45.42</td><td>0.2800</td></tr><tr><td rowspan="2">Inpainting</td><td> $g _ { \mathrm { L C P L } }  \tilde { g } _ { \mathrm { L C P L } }$ </td><td>45.53</td><td>0.2935</td></tr><tr><td> $g _ { \mathrm { L C F M } } \to \tilde { g } _ { \mathrm { L C F M } }$ </td><td>45.50</td><td>0.2936</td></tr><tr><td rowspan="2">Colorization</td><td> $g _ { \mathrm { L C P L } }  \tilde { g } _ { \mathrm { L C P L } }$ </td><td>45.21</td><td>0.3599</td></tr><tr><td> $g _ { \mathrm { L C F M } } \to \tilde { g } _ { \mathrm { L C F M } }$ </td><td>45.21</td><td>0.3596</td></tr></table>

## B.2 Further Analysis

Fig. 1 compares the transport trajectories of PCFlow compared with its baselines, ELIR [6] and PMRF [19]. PMRF exhibits unstable structural progression across timesteps. Early predictions remain overly smooth and high-frequency details appear abruptly, resulting in inconsistent intermediate states. ELIR attempts to introduce perceptual details from the beginning of the trajectory. However, these details are structurally inconsistent, suggesting that the model prioritizes texture synthesis before suficiently recovering the underlying structure. In contrast, PCFlow follows a coarse-to-fine restoration trajectory. PCFlow primarily recovers the global facial structure in early steps, whereas high-frequency details such as hair texture and eye boundaries are progressively refined in later timesteps.

![](images/f6e13343bda4c84d4f3c6603a31976322bf122368a8fed343a44e7e3916789a2.jpg)  
Fig. 1: Comparison in trajectory of PCFlow with its baselines. PCFlow exhibits a stable coarse-to-fine refinement trajectory, where global structure is initially recovered in early steps and fine details are progressively introduced. In contrast, PMRF produces inconsistent intermediate states, while ELIR introduces perceptual details prematurely, often leading to structurally inconsistent textures.

## B.3 Further Qualitative Results

Input (LQ)  
ELIR  
PMRF(K=3)  
PMRF(K=25)  
Ours  
HQ  
![](images/6db242d27ff922ae6d4eca5d6274ea5362977d62dc106ce4d8e67fffa8ef2bc3.jpg)  
Fig. 2: Additional Results of PCFlow on super-resolution task. From left to right: Input(LQ), ELIR, PMRF(K = 3), PMRF(K = 25), PCFlow(Ours), and ground truth image(HQ).

Input (LQ)  
ELIR  
PMRF(K=3)  
PMRF(K=25)  
Ours  
HQ  
![](images/b1d5f994d3df4c536a6e3354e473c61ef89df9c1111a0c59ed1890042c8d9023.jpg)  
Fig. 3: Additional Results of PCFlow on denoising task. From left to right: Input(LQ), ELIR, PMRF(K = 3), PMRF(K = 25), PCFlow(Ours), and ground truth image(HQ).

Input (LQ)  
ELIR  
PMRF(K=3)  
PMRF(K=25)  
Ours  
HQ  
![](images/3b158783fc5a9477286e807a3252a97e6c6717a79e28794975b8605de5c92658.jpg)  
Fig. 4: Additional Results of PCFlow on inpainting task. From left to right: Input(LQ), ELIR, PMRF(K = 3), PMRF(K = 25), PCFlow(Ours), and ground truth image(HQ).

Input (LQ)  
ELIR  
PMRF(K=3)  
PMRF(K=25)  
Ours  
HQ  
![](images/80c13b4e1cea9d3ad48f5379a401d253337d96f566ee2b5dda7f1f8d639d1b23.jpg)  
Fig. 5: Additional Results of PCFlow on colorization task. From left to right: Input(LQ), ELIR, PMRF(K = 3), PMRF(K = 25), PCFlow(Ours), and ground truth image(HQ).