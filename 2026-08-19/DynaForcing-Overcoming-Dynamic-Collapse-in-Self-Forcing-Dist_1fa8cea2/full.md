# DynaForcing: Overcoming Dynamic Collapse in Self-Forcing Distillation for Streaming Avatar Generation

Yubo Huang snake1124@mail.ustc.edu.cn University of Science and Technology of China Hefei, China

Zhengye Zhang zhengyezhang@mail.ustc.edu.cn University of Science and Technology of China Hefei, China

Sirui Zhao sirui@mail.ustc.edu.cn University of Science and Technology of China Hefei, China

Jinyang Huang hjy@hfut.edu.cn Hefei University of Technology Hefei, China

Shiwei Wu davidwu16@sz.tsinghua.edu.cn Tsinghua University Shenzhen, China

Xinchen Yao xinchenyao@smail.nju.edu.cn Nanjing University Nanjing, China

Fengqi Cui fengqi\_cui@mail.ustc.edu.cn University of Science and Technology of China Hefei, China

Enhong Chen<sup>∗</sup> cheneh@ustc.edu.cn University of Science and Technology of China Hefei, China

![](images/fbe6bd7f3a223926fb77cb9aab42289767ff25c25aeb8b2d8f46d1f89a7e2540.jpg)  
Figure 1: Dynamic collapse in self-forcing distillation for streaming avatar generation. Given the same reference image and driving audio (left), the self-forcing baseline (top) produces visually appealing but temporally frozen outputs—nearly identical lip shapes across 10 seconds of speech (see mouth crops). DynaForcing (bottom) restores faithful audio-driven dynamics with natural expression variation and accurate lip articulation, at 45.2 FPS real-time streaming.

## Abstract

Audio-driven avatar generation requires realistic lip-sync, expressive motion, and real-time streaming. Recent work achieves the latter via self-forcing with Distribution Matching Distillation (DMD), but this paradigm sufers from a critical failure that has not been systematically characterized: dynamic collapse, where the student model converges to a near-static optimum with high perceptual quality but severely suppressed temporal dynamics. We trace this to two causes: the reverse KL objective in DMD, which biases toward low-motion modes, and unanchored self-conditioning, which creates a feedback loop that amplifies collapse. This is especially harmful for avatars, where even subtle motion loss breaks lip-sync and expression. To address this, we propose DynaForcing, a training framework with three complementary strategies applied at diferent levels. Specifically, Hybrid Forcing anchors rollouts to ground-truth dynamics at the data level to break the feedback loop. Dynamics-Aware Reward Regularization introduces explicit motion rewards via the RL interpretation of DMD to counteract the reverse KL bias at the loss level. Reference Perturbation perturbs reference images to decouple identity from static details, forcing the model to rely on audio for motion at the conditioning level. We further introduce computation graph pruning and gradient replay, reducing the GPU footprint of self-forcing by over an order of magnitude. Experiments show that DynaForcing recovers dynamics to teachercomparable levels (Dyn-Deg: 0.31→0.73, Sync-C: 7.03→7.68) while improving visual quality, resolving the quality–dynamics trade-of throughout training without early stopping.

## CCS Concepts

• Computing methodologies → Computer vision; Video gen eration.

## Keywords

interactive video generation, eficient video generation

## 1 Introduction

Audio-driven avatar generation aims to synthesize photorealistic talking-head video whose motion faithfully follows an input audio stream, serving as a cornerstone technology for interactive digital communication [9, 26, 32]. Deploying such systems in real-world scenarios, such as video conferencing and virtual assistants, requires satisfying two simultaneous demands: real-time streaming at interactive frame rates, and temporally rich dynamics including precise lip articulation, natural expressions, and responsive head motion. Recent large-scale video difusion models [1, 34] have significantly advanced visual fidelity, and self-forcing [5, 11] has emerged as a leading distillation paradigm, leveraging Distribu tion Matching Distillation (DMD) [43] to convert these models into causal, few-step streaming generators that enable real-time, infinite-length avatar generation at 14B-parameter scale [12, 28].

Yet beneath these impressive capabilities lies a failure mode that, while noted in recent work [17, 19], has not been systematically characterized. We find that self-forcing training systematically drives the student model toward what we term dynamic collapse (Figure 1), where generated videos exhibit high perceptual quality scores but near-zero temporal dynamics: mouths barely move, expressions freeze, and the rich motion present in the teacher model vanishes. Avatar generation is particularly susceptible because the task requires fine-grained temporal dynamics (precise lip shapes for each phoneme, micro-expressions, head motion responsive to prosody), yet the static-scene prior (fixed background, stable iden tity) means that a near-motionless output already satisfies most visual quality criteria, leaving little gradient pressure toward dynamics. While DMD’s diversity–quality tradeof is well-documented for images [42, 43], its impact on video is qualitatively diferent: collapsing temporal diversity does not merely reduce “variety” but fundamentally breaks the core requirement of faithful audio-visual correspondence.

We trace this failure to two contributing mechanisms. First, DMD’s reverse KL objective $D _ { \mathrm { K L } } ( p _ { \theta } \Vert p _ { \mathrm { d a t a } } )$ is mode-seeking [23]: it penalizes the student for placing mass outside the teacher’s support but imposes no cost for ignoring modes. Second, self-forcing per forms unanchored self-conditioning: unlike GT-anchored approaches (e.g., CausVid [44]) that process all blocks in parallel with groundtruth latents as input, self-forcing builds the entire KV cache from the student’s own outputs with no ground-truth anchoring whatsoever. This creates a compounding feedback loop: when the student begins producing low-dynamics content, all subsequent blocks condition on this self-generated static history, reinforcing the col lapse across the entire sequence. We verify this empirically: under identical training horizons, CausVid’s GT-anchored conditioning maintains stable dynamics while self-forcing collapses to near-zero, as demonstrated in Figure 2c.

To address dynamic collapse, we propose DynaForcing, a training framework with three complementary strategies operating at diferent levels of the training pipeline:

(1) Hybrid Forcing (§4.2), a data-level strategy that probabilistically replaces self-forcing rollout starting points with noised ground-truth latents. This provides a ground-truth anchor that counteracts mode collapse in the distillation objective.

(2) Dynamics-Aware Reward Regularization (§4.3), a loss-level strategy that leverages the RL interpretation of DMD [20] to introduce explicit dynamics rewards (lip-sync accuracy and expression variance) as auxiliary training signals that directly penalize static outputs.

(3) Reference Perturbation (§4.4), a complementary conditioninglevel strategy that perturbs reference images to decouple identity from extraneous visual details (pose, background, lighting), preventing the model from taking a shortcut of copying the reference frame instead of generating audio-driven motion.

Additionally, we present eficient self-forcing training (§4.5) via computation graph pruning and gradient replay, reducing GPU requirements by over 10× while preserving quality.

In summary, the main contributions of this work are as follows: • We formalize and systematically characterize dynamic collapse in self-forcing distillation, tracing it to reverse-KL mode-seeking and the compounding efect of unanchored self-conditioning.

• We propose DynaForcing, combining three strategies at diferent levels (data, loss, conditioning) that jointly address dynamic collapse while preserving visual quality.

• We introduce computation graph pruning and gradient replay for self-forcing, achieving ∼10× GPU-hour reduction.

• Experiments on audio-driven avatar generation demonstrate substantial improvements in motion naturalness and audio-visual faithfulness on both short- and long-form benchmarks.

## 2 Related Work

Audio-Driven Avatar Generation. Audio-driven talking-head synthesis has evolved from GAN-based methods [26, 45] to difusionbased approaches [4, 9, 10, 13, 32, 40] that leverage large-scale video difusion models for dramatically improved fidelity. More recently, several works have achieved real-time streaming avatar generation by combining difusion models with autoregressive distillation [12, 15, 28, 30, 35], demonstrating interactive-rate generation at scale. However, these streaming methods consistently exhibit significantly degraded motion precision and naturalness compared to their multi-step teachers, even though the underlying DMD distillation has been shown to produce students that match or exceed teacher quality on static image metrics [42, 43]. For avatars, this motion loss is particularly damaging: lip-sync accuracy, expression diversity, and gesture naturalness are the core requirements, and any dynamics degradation directly undermines the task objective. Streaming Video Distillation. Distilling bidirectional multi-step difusion models into few-step causal streaming generators has been pursued through progressive distillation [27], consistency models [29], and DMD [42, 43]. CausVid [44] introduced GT-anchored autoregressive distillation, and Self-Forcing [5, 11] further removed ground-truth anchoring by conditioning entirely on the student’s own outputs, better aligning train-time and test-time distributions. Subsequent works have refined this paradigm: Causal Forcing [46] bridges the architectural gap via AR-teacher ODE initialization, Rolling Forcing [18] suppresses error accumulation through joint multi-frame denoising, and LongLive [41] extends stable generation to longer horizons. The field has largely converged on DMD-based self-forcing as the dominant paradigm for streaming video generation. Yet self-forcing introduces an inherent motion degradation problem: the combination of reverse-KL mode-seeking and unanchored self-conditioning creates a feedback loop that systematically suppresses temporal dynamics. This has been independently observed by DiagDistill [17], which proposes diagonal forcing with implicit flow modeling to mitigate motion amplitude attenuation, and Reward Forcing [19], which introduces reward signals to encourage dynamics. Nevertheless, the problem remains largely unsolved, especially for tasks like avatar generation that demand fine-grained, audio-synchronized motion rather than coarse temporal coherence.

![](images/2f521ea32d4def1c52595e03d8b2a33b428cee79c4c193f6e15efbb84a39c2fb.jpg)

![](images/addc1b1667c9f1c61530cdd5e41a348bbcf51cc758b6e3199c18a084131ed8f8.jpg)

![](images/58b737faa44cb1568d8d8d996162955391c59aef51d7e9ece1e821db39aec783.jpg)  
Figure 2: Dynamic collapse in self-forcing distillation. (a) Text-to-video (Self-Forcing [11]): VBench [14] dynamic degree drops from 0.80 to 0.00 while visual quality improves. (b) Audio-driven avatar (LiveAvatar [12]): Sync-C and ExpVar degrade while IQA and ASE improve. (c) Comparing self-forcing with CausVid [44] (GT-anchored): CausVid’s dynamic degree fluctuates around 0.43 without collapsing, confirming that unanchored self-conditioning, not DMD distillation itself, drives the collapse.

## 3 Preliminaries

## 3.1 Self-Forcing for Streaming Video Difusion

Self-forcing [11] enables streaming video generation by combining autoregressive rollout with few-step difusion sampling. The video is generated block-by-block, where each block $B ^ { i }$ contains � consecutive frame latents. At each rollout step, the student model $G _ { \theta }$ denoises the current block conditioned on a rolling KV cache of previous blocks:

$$
\hat { B } _ { 0 } ^ { i } = G _ { \theta } ( B _ { T } ^ { i } , \underbrace { B _ { t } ^ { ( i - w ) : ( i - 1 ) } } _ { \mathrm { K V } \mathrm { c a c h e } } , I , a ^ { i } ) ,\tag{1}
$$

where $B _ { T } ^ { i } \ \sim \ { \cal N } ( 0 , { \bf I } )$ is the initial noise, � is the cache window size, � is the reference (sink) frame, $a ^ { i }$ is the audio embedding, and � denotes the shared noise level of the KV cache (timestepforcing). The student produces a multi-step denoising trajectory $B _ { T } ^ { i } \to B _ { t _ { 1 } } ^ { i } \to \dots \to \hat { B } _ { 0 } ^ { i }$ , and the generated clean latent $\hat { B } _ { 0 } ^ { i }$ is used to construct the KV cache for subsequent blocks.

## 3.2 Distribution Matching Distillation as RL

Distribution Matching Distillation (DMD) [43] trains the student by minimizing the reverse KL divergence between the student’s induced distribution $\hbar \theta$ and the teacher’s data distribution ${ \mathcal { P } } \mathrm { d a t a } ,$ averaged over noise levels �:

$$
\mathcal { L } _ { \mathrm { D M D } } = \mathbb { E } _ { t } \left[ D _ { \mathrm { K L } } \left( p _ { \theta , t } \Vert p _ { \mathrm { d a t a } , t } \right) \right] .\tag{2}
$$

The gradient of this objective takes the form:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { D M D } } = - \mathbb { E } _ { t , \mathbf { z } } \left[ \left( s _ { \mathrm { r e a l } } ( \mathbf { x } _ { t } , t ) - s _ { \mathrm { f a k e } } ( \mathbf { x } _ { t } , t ) \right) ^ { \top } \frac { \partial G _ { \boldsymbol { \theta } } ( \mathbf { z } ) } { \partial \boldsymbol { \theta } } \right] ,\tag{3}
$$

where $s _ { \mathrm { r e a l } }$ is the teacher (pretrained) score and $S f a k e$ is a learned score on student-generated data, and $\mathbf { x } _ { t } ~ = ~ \Psi ( \hat { \mathbf { x } } , t )$ is the noiseperturbed student output.

Crucially, Luo et al. [20] frame DMD within an RL/RLHF objective: the score diference in Eq. 3 serves as an Integral KL regularization toward the teacher, and additional reward signals can be incorporated by augmenting the objective with an explicit reward term. Building on this, Lu et al. [19] propose reward-weighted distillation, where an external reward �(x) exponentially reweights the DMD gradient:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { R e - D M D } } = - \mathbb { E } _ { t , \mathrm { z } } \left[ \exp \bigl ( \boldsymbol { \alpha } \cdot \overline { { \boldsymbol { R } } } \bigr ) \cdot \bigl ( s _ { \mathrm { r e a l } } - s _ { \mathrm { f a k e } } \bigr ) ^ { \top } \frac { \partial G _ { \boldsymbol { \theta } } ( \mathbf { z } ) } { \partial \boldsymbol { \theta } } \right] ,\tag{4}
$$

where ${ \overline { { R } } } = \operatorname { s g } ( R )$ is the stopped-gradient reward and � controls the reward strength. This formulation amplifies the DMD gradient for high-reward samples while suppressing it for low-reward ones, efectively steering the student toward reward-rich modes without requiring backpropagation through the reward model.

This reward-weighted framework is central to our approach: by defining a dynamics-specific reward, we can bias the self-forcing training loop away from degenerate static modes (§4.3).

## 4 Method: DynaForcing

We first analyze the dynamic collapse phenomenon (§4.1), then present our three strategies: Hybrid Forcing (§4.2), Dynamics-Aware Reward Regularization (§4.3), and Reference Perturbation (§4.4). Finally, we describe eficient training (§4.5).

![](images/51503c084c2ca230b35ecf51f077761a633419ff9eb71619be2683a7f5ad5c67.jpg)  
Figure 3: Overview of DynaForcing. Three strategies intervene at diferent levels of the self-forcing training pipeline: (a) Hybrid Forcing at the input level, (b) dynamics-aware reward regularization at the loss level, and (c) reference perturbation at the conditioning level.

## 4.1 Understanding Dynamic Collapse

We observe that self-forcing training can severely suppress temporal dynamics despite producing visually appealing outputs. We term this phenomenon dynamic collapse and analyze its causes. Empirical evidence. Figure 2 demonstrates that dynamic collapse is not task-specific but a general pathology of self-forcing distillation.<sup>1</sup> On text-to-video generation (Figure 2a), VBench [14] dynamic degree monotonically decreases from 0.80 to 0.00 over 8K training steps, while imaging quality and aesthetic quality steadily improve. On audio-driven avatar generation (Figure 2b), the same pattern emerges: visual quality (IQA, ASE) improves steadily while dynam ics metrics (Sync-C, ExpVar) peak early and then decline sharply, eventually reaching severely suppressed levels. In both cases, the generated videos look sharp and artifact-free but exhibit minimal temporal variation.

Analysis: Mode-seeking reverse KL. DMD minimizes $D _ { \mathrm { K L } } ( p _ { \theta } \Vert p _ { \mathrm { d a t a } } )$ which is mode-seeking [23]: it penalizes mass outside the teacher’s support but incurs no cost for ignoring modes. In the video domain, static content is an attractive mode because the DMD score diference $s _ { \mathrm { r e a l } } - s _ { \mathrm { f a k e } }$ provides limited gradient signal for dynamics, as the teacher’s score function primarily captures distributional realism, not temporal richness. As the student concentrates on a narrow, low-motion subset of the teacher’s distribution, the fake score �<sub>fake</sub> quickly matches $s _ { \mathrm { r e a l } }$ in this region, eliminating the gradient signal needed to explore other modes.

Analysis: Unanchored self-conditioning. Self-forcing builds the entire KV cache from the student’s own outputs, with no groundtruth anchoring during the rollout. Once the student begins producing low-motion blocks, all subsequent blocks condition on this static history, creating a compounding feedback loop that reinforces collapse across the sequence. The key insight is not autoregression per se, but the absence of any ground-truth signal in the conditioning path. Figure 2c validates this analysis: under identical evaluation, CausVid [44], which conditions each block on noisy ground-truth frames, maintains stable dynamics (mean 0.43) while self-forcing collapses to 0.

Analysis: Why avatar generation is particularly afected. While dynamic collapse can occur in general video distillation, audio-driven avatar generation is particularly susceptible because: (i) the task requires fine-grained temporal dynamics (precise lip shapes for each phoneme, micro-expressions), making any dynam ics reduction immediately noticeable; and (ii) the static-scene prior (fixed background, stable identity) means that a zero-motion output already satisfies most of the visual quality requirements, leaving little gradient pressure toward dynamics.

## 4.2 Hybrid Forcing: Data-Anchored Rollout

To break the feedback loop of dynamic collapse, we propose Hybrid Forcing, which probabilistically injects ground-truth dynamics into the self-forcing rollout at the starting point level, providing an anchor that counteracts mode collapse under unanchored selfconditioning.

At each training step, we sample a mode for the entire rollout (i.e., the decision is per-sample, applied uniformly to all blocks). Denoting the full sequence starting point as $\mathbf { z } = [ \dot { B } _ { \mathrm { s t a r t } } ^ { 1 } , \dots , B _ { \mathrm { s t a r t } } ^ { K } ]$ each block’s starting point is:

$$
\begin{array} { r } { B _ { \mathrm { s t a r t } } ^ { i } = \left\{ \begin{array} { l l } { B _ { T } ^ { i } \sim N ( 0 , \mathbf { I } ) , } & { \mathrm { w . p . ~ } 1 - p _ { \mathrm { d a t a } } , } \\ { \Psi ( B _ { \mathrm { G T } } ^ { i } , t _ { \mathrm { s t a r t } } ) , } & { \mathrm { w . p . ~ } p _ { \mathrm { d a t a } } , } \end{array} \right. } \end{array}\tag{5}
$$

where $B _ { \mathrm { G T } } ^ { i }$ is the ground-truth video latent, $\Psi ( \cdot , t )$ is the forward noise scheduler, and $t _ { \mathrm { s t a r t } } \sim \{ t _ { 0 } , \dots , t _ { K - 1 } \}$ is sampled from the � noise levels in the few-step denoising schedule. In data-anchored mode, the student starts from a noised ground-truth rather than pure noise, preserving temporal dynamics (motion patterns, lip movements) as structural information that the student must refine.

The DMD loss is then computed on the student’s denoised output regardless of the starting mode. When starting from data-anchored points, the loss gradient points toward dynamically-rich outputs, providing an escape direction from the zero-motion basin. The KV cache always uses the student’s own representations (noisy, per timestep-forcing), maintaining train-test alignment.

Why data-anchored starting points help. A noised groundtruth latent $\Psi ( x _ { \mathrm { G T } } , t ) = \alpha _ { t } x _ { \mathrm { G T } } + \sigma _ { t } \varepsilon$ retains recoverable structure (motion cues, temporal dynamics, spatial layout) as long as the signal-to-noise ratio $\alpha _ { t } / \sigma _ { t }$ is not too small. A pure Gaussian start $z \sim { \cal N } ( 0 , { \bf I } )$ carries none of this information, forcing the student to reconstruct everything from its imperfect learned prior. We formalize this intuition in the supplementary material, showing that data-anchored starts yield a tighter Wasserstein bound on the denoised output.

Starting-step probability redistribution. Starting from $t _ { \mathrm { s t a r t } }$ means the model only traverses the denoising sub-trajectory $t _ { \mathrm { s t a r t } } $ $\cdots \to t _ { 0 } ( \mathrm { E q . ~ } 5 )$ , so noisier levels receive less training coverage. Backward simulation [42], which self-forcing inherits, addresses a similar coverage issue for pure-noise starts. To compensate in the data-anchored case, we redistribute the sampling probability of $t _ { \mathrm { s t a r t } }$ : since $t _ { 0 }$ already operates at high SNR where strong dynamic priors are preserved and little denoising correction is needed, we concentrate mass on early steps, which retain the most dynamic structure from the ground truth while still requiring non-trivial denoising.

## 4.3 Dynamics-Aware Reward Regularization

Framework. As discussed in §3.2, DMD can be viewed through an RL lens where the score diference serves as an implicit reward [20]. However, this implicit reward can be fully satisfied by mode collapse (concentrating on high-likelihood static content), since it measures distributional realism, not temporal richness. We introduce an explicit dynamics reward $R _ { \mathrm { d y n } }$ as an additional return signal that directly penalizes static outputs. Following the reward-weighted distillation framework [19, 20], we integrate $R _ { \mathrm { d y n } }$ by exponentially reweighting the DMD gradient:

$$
\nabla _ { \theta } \mathcal { L } _ { \mathrm { D y n a f o r c i n g } } = - \mathbb { E } _ { t , z } \Bigg [ \exp \big ( \boldsymbol { \alpha } \cdot \operatorname { s g } ( \boldsymbol { R } _ { \mathrm { d y n } } ) \big ) \cdot \big ( \boldsymbol { s } _ { \mathrm { r e a l } } - \boldsymbol { s } _ { \mathrm { f a k e } } \big ) ^ { \top } \frac { \partial G _ { \theta } ( \mathbf { z } ) } { \partial \theta } \Bigg ] ,\tag{6}
$$

where $s \mathrm { g } ( \cdot )$ denotes the stop-gradient operator and � controls the reward strength. This amplifies the DMD gradient for samples with rich dynamics and suppresses it for static outputs, approximating optimization toward a reward-tilted target �˜(x) ∝ ${ p } _ { \mathrm { t e a c h e r } } ( \mathbf { x } ) \exp ( \alpha R _ { \mathrm { d y n } } ( \mathbf { x } ) )$ ), analogous to reward-weighted regression in ofline RL [25].

Reward implementation. Let $\hat { \mathbf { V } } = [ \hat { B } _ { 0 } ^ { 1 } , \dots , \hat { B } _ { 0 } ^ { K } ]$ denote the student rolled video. We define the dynamics reward as:

$$
R _ { \mathrm { d y n } } ( \hat { \bf V } , { \bf a } ) = \lambda _ { \mathrm { s y n c } } \cdot \widetilde { \bf S \mathrm { y n c - } \mathrm { C } } ( \hat { \bf V } , { \bf a } ) + \lambda _ { \mathrm { e x p } } \cdot \widetilde { \bf E x p \mathrm { V a r } } ( \hat { \bf V } ) ,\tag{7}
$$

where $\widetilde { \mathrm { S y n c - C } }$ is the normalized lip-sync confidence from a pretrained SyncNet [3] and ExpVar is the normalized standard deviation of 3DMM expression coeficients across frames [47]. Both are computed on decoded video frames and normalized to [0, 1] via running statistics. $R _ { \mathrm { d y n } }$ is treated as a constant during backpropagation (stop-gradient), so only the score terms and generator Jacobian carry gradient flow, making this fully compatible with standard DMD training.

## 4.4 Reference Perturbation

We identify that standard training practice induces a complementary source of dynamics suppression at the conditioning level: the reference image leaks far more information than identity alone.

The copy-dominance problem. In avatar generation, the reference image � is typically sampled from the same video clip as the target. Since reference and target share scene, background, viewpoint, and often similar pose, the model learns a shortcut: copy the reference frame with minimal modification, rather than generating genuine audio-driven dynamics. The reference is meant to supply only identity, but in practice it leaks far more, suppressing the need for temporal generation.

Perturbing references to decouple identity. To decouple identity from extraneous visual details, we perturb reference images so that they retain identity but vary in other attributes. Specifically, we use an of-the-shelf image editing model [38] to generate perturbed reference images from the training set. Given a reference frame �, we apply semantic-preserving edits (viewpoint variation, background modification, lighting changes) that break pixel-level correspondence between reference and target while preserving identity. By training with these perturbed references, the model is forced to extract only identity information from the reference and rely on the audio signal for temporal dynamics, rather than copying static content. Detailed prompts are provided in the supplementary material.

Similarity-calibrated sampling. Not all perturbed references are equally useful. We filter and organize them using a two-stage criterion: (1) Identity filtering: we compute ArcFace [6] cosine similarity between the perturbed reference and the original, retaining only pairs with similarity $> \tau _ { \mathrm { a r c } }$ to ensure identity consistency. (2) Perturbation-aware bucketing: for retained pairs, we compute the structural similarity (SSIM [37]) between the perturbed and original images and partition pairs into 5 buckets of width 0.15 spanning [0.25, 1.0]. During training, we uniformly sample across buckets, ensuring the model sees a balanced distribution from near-identical to substantially diferent reference-target pairs. This balanced curriculum prevents the model from relying on the copy shortcut, forcing it to generate genuine audio-driven dynamics.

## 4.5 Eficient Self-Forcing Training

Self-forcing training is memory-intensive because the full rollout graph must be retained for backpropagation [11]. We introduce two optimizations that substantially reduce memory requirements.

Computation graph pruning. We empirically observe that crossblock gradients (propagated through the KV cache) have negligible impact on training outcomes, yet significantly increase memory pressure. We therefore detach the KV cache from the computation graph, removing cross-block gradient dependencies.

Block-wise gradient replay. With cross-block gradients eliminated, we adopt a two-phase training procedure. In the rollout phase, the entire sequence is generated with gradients disabled, and intermediate results (latent outputs, KV cache entries) are stored in memory. In the replay phase, we iterate over blocks one at a time: for each block, we re-enable gradients and run a single forward pass using the stored intermediates, then compute and accumulate the DMD gradient. This reduces peak activation memory by ∼�× (where � is the number of blocks), at the cost of � additional forward passes.

## 5 Experiments

## 5.1 Experimental Setup

Implementation details. Our approach is built upon the WanS2V [10] (14B). The training process is divided into two stages. For Stage 1, we strictly follow the difusion-forcing pretraining protocol introduced by LiveAvatar [12], while our proposed modifications are exclusively applied during Stage 2. During the inference phase, we

Table 1: Quantitative comparison on GenBench-ShortVideo (∼10 s) and GenBench-LongVideo (>5 min) [12]. Bold: best. Underline: second best. ↑/↓: higher/lower is better. FPS is resolution- and hardware-dependent (720×400, single H800 node) and identical across benchmarks.
<table><tr><td rowspan="2">Method</td><td colspan="6">Standard Metrics</td><td colspan="2">Dynamics</td></tr><tr><td>ASE↑</td><td>IQA↑</td><td>Sync-C↑ Sync-D↓</td><td></td><td>Dino-S↑</td><td>FPS↑</td><td>ExpVar↑</td><td>Dyn-Deg↑</td></tr><tr><td colspan="9">GenBench-ShortVideo</td></tr><tr><td colspan="9">Non-real-time</td></tr><tr><td>WanS2V [10]</td><td>3.36</td><td>4.29</td><td>5.89</td><td>9.08</td><td>0.95</td><td>0.25</td><td>1.81</td><td>0.72</td></tr><tr><td>OmniAvatar [9]</td><td>3.53</td><td>4.49</td><td>6.77</td><td>8.22</td><td>0.95</td><td>0.16</td><td>1.95</td><td>0.75</td></tr><tr><td>Hallo3 [4]</td><td>3.12</td><td>3.97</td><td>4.74</td><td>10.19</td><td>0.94</td><td>0.26</td><td>1.48</td><td>0.57</td></tr><tr><td>StableAvatar [33]</td><td>3.52</td><td>4.47</td><td>3.42</td><td>11.33</td><td>0.93</td><td>0.64</td><td>1.23</td><td>0.51</td></tr><tr><td>EchoMimic-V2 [22]</td><td>2.82</td><td>3.61</td><td>5.57</td><td>9.13</td><td>0.79</td><td>0.53</td><td>1.56</td><td>0.60</td></tr><tr><td colspan="9">Real-time</td></tr><tr><td>Ditto [16]</td><td>3.31</td><td>4.24</td><td>4.09</td><td>10.76</td><td>0.99</td><td>21.80</td><td>0.94</td><td>0.37</td></tr><tr><td>LiveAvatar [12]</td><td>3.44</td><td>4.51</td><td>7.03</td><td>8.30</td><td>0.96</td><td>45.2</td><td>0.69</td><td>0.31</td></tr><tr><td>SoulX-FlashTalk [28]</td><td>3.48</td><td>4.53</td><td>7.12</td><td>8.25</td><td>0.96</td><td>32.5</td><td>0.81</td><td>0.35</td></tr><tr><td>DynaForcing (Ours)</td><td>3.55</td><td>4.58</td><td>7.68</td><td>7.89</td><td>0.96</td><td>45.2</td><td>2.02</td><td>0.73</td></tr><tr><td colspan="9">GenBench-LongVideo</td></tr><tr><td colspan="9">Non-real-time</td></tr><tr><td>WanS2V</td><td>2.63</td><td>3.99</td><td>6.04</td><td>9.12</td><td>0.80</td><td>0.25</td><td>1.63</td><td>0.65</td></tr><tr><td>OmniAvatar</td><td>2.36</td><td>2.86</td><td>8.00</td><td>7.59</td><td>0.66</td><td>0.16</td><td>2.06</td><td>0.71</td></tr><tr><td>Hallo3</td><td>2.65</td><td>4.04</td><td>6.18</td><td>9.29</td><td>0.83</td><td>0.26</td><td>1.36</td><td>0.53</td></tr><tr><td>StableAvatar</td><td>3.00</td><td>4.66</td><td>1.97</td><td>13.57</td><td>0.94</td><td>0.64</td><td>1.07</td><td>0.44</td></tr><tr><td colspan="9">Real-time</td></tr><tr><td>Ditto</td><td>2.90</td><td>4.48</td><td>3.98</td><td>10.57</td><td>0.97</td><td>21.80</td><td>0.76</td><td>0.35</td></tr><tr><td>LiveAvatar</td><td>3.42</td><td>4.76</td><td>7.16</td><td>8.31</td><td>0.97</td><td>45.2</td><td>0.57</td><td>0.28</td></tr><tr><td>SoulX-FlashTalk</td><td>3.40</td><td>4.72</td><td>7.08</td><td>8.18</td><td>0.96</td><td>32.5</td><td>0.67</td><td>0.32</td></tr><tr><td>DynaForcing (Ours)</td><td>3.50</td><td>4.82</td><td>8.05</td><td>8.02</td><td>0.97</td><td>45.2</td><td>1.93</td><td>0.68</td></tr></table>

employ the TPP strategy from [12] to generate videos at a resolution of 720×400. Regarding the optimization hyperparameters, we set the probability $ { p _ { \mathrm { d a t a } } }$ to 0.3. We intentionally concentrate the starting-step probability mass on the early noise levels by assigning $p ( t _ { 1 } ) = 0 . 7 0 $ , alongside $p ( t _ { 0 } ) = p ( t _ { 2 } ) = 0 . 1 5$ . Furthermore, the reward weights are configured as follows: the synchronization weight $\lambda _ { \mathrm { s y n c } }$ is 1.0, the expression weight $\lambda _ { \mathrm { e x p } }$ is 0.5, and the balancing factor � is 0.1. We also establish an ArcFace threshold of $\tau _ { \mathrm { a r c } } = 0 . 9$ . Finally, our model is trained for 4,000 steps on eight NVIDIA H100 GPUs. For the DMD optimization, we use a learning rate of $1 \times 1 0 ^ { - 5 }$ for the student and $2 \times 1 0 ^ { - 6 }$ for the fake score, with an update frequency ratio of 5.

Datasets. We train our model on the AVSpeech dataset [7], utilizing the preprocessing pipeline introduced by OmniAvatar [9]. This resulting training corpus consists of 400,000 samples, with all video clips having a duration of more than 10 seconds. For evaluation, we adopt two benchmark datasets from [12]. The first is GenBench-ShortVideo, which includes 100 samples that are approximately 10 seconds long, and the second is GenBench-LongVideo, which comprises 15 samples with durations exceeding 5 minutes.

Evaluation metrics. We adopt standard metrics: Q-Align [39] for perceptual quality (IQA) and aesthetic appeal (ASE), Sync-C and Sync-D [3] for audio-visual synchronization, DINOv2 [24] cosine similarity (Dino-S) for identity consistency. We additionally report dynamics metrics: ExpVar (standard deviation of 3DMM expression coeficients across frames [47], measuring expression dynamics) and VBench [14] Dyn-Deg (dynamic degree, measuring temporal variation diversity).

Baselines. We compare with methods spanning two paradigms: (1) Non-real-time multi-step difusion models: WanS2V [10] (teacher), OmniAvatar [9], Hallo3 [4], StableAvatar [33], EchoMimic-V2 [22]; (2) Real-time streaming models: Ditto [16], LiveAvatar [12], SoulX-FlashTalk [28]. All benchmarked at 720×400 on a single H800 node. For methods with strict requirements on reference image aspect ratio or subject positioning (SoulX-FlashTalk and EchoMimic-V2), we use their oficial open-source inference pipelines without modification to ensure fair comparison. The “self-forcing baseline” in ablations denotes standard self-forcing trained for 4,000 steps, identical to DynaForcing but without our three proposed components.

## 5.2 Main Results

GenBench-ShortVideo. Table 1 (top) presents results. Real-time self-forcing models (LiveAvatar, SoulX-FlashTalk) achieve strong visual quality but sufer severely on dynamics: LiveAvatar’s ExpVar (0.69) is 62% lower than its teacher WanS2V (1.81), and its Dyn-Deg (0.31) is less than half of OmniAvatar (0.75), confirming dynamic collapse as a systematic issue. DynaForcing recovers dynamics to teacher-comparable levels: Dyn-Deg rises from 0.31 to 0.73 (teacher:

![](images/f2966020797a4cf18868705a2efb81d6ee8db12865e1d355e165a95a0cb50ed0.jpg)

![](images/2ee7f05a37a0f260a5313ead43fb45d7898a31ad8cbcb9ae7b0e58db21e825da.jpg)  
Figure 4: Qualitative comparison on GenBench-LongVideo. Frames sampled at 2 s, 200 s, and 400 s from the same driving audio. Please refer to the supplementary video for dynamics comparison.

0.72), ExpVar from 0.69 to 2.02, and Sync-C from 7.03 to 7.68. That ExpVar slightly exceeds the teacher (2.02 vs. 1.81) is consistent with the known tendency of DMD-distilled models to concentrate output distributions [21, 31]; our reward regularization further steers this concentration toward dynamically-rich modes. Note that Sync-C and ExpVar are both components of our training reward, so gains on these two metrics alone cannot rule out reward hacking. We there fore highlight Dyn-Deg, which is not optimized during training yet recovers from 0.31 to 0.73 (matching the teacher’s 0.72), as independent evidence of genuine dynamics recovery. Our user study further corroborates this with strong human preference on motion naturalness. Moreover, by stabilizing dynamics throughout train ing, DynaForcing avoids the early quality–dynamics trade-of that forces vanilla self-forcing to stop prematurely, enabling full convergence and thus improved visual quality as a byproduct (ASE/IQA: 3.55/4.58 vs. 3.44/4.51). FPS remains 45.2 as our contributions are training-side only.

GenBench-LongVideo. Table 1 (bottom) shows that the dynamics advantage of DynaForcing holds on long-form generation (>5 min). LiveAvatar’s ExpVar further drops from 0.69 (short) to 0.57 (long) as static representations accumulate in the KV cache, while Dy naForcing maintains strong dynamics (ExpVar 1.93) alongside the best visual quality and synchronization.

Qualitative results. Figure 4 shows representative frames from all compared methods. We refer readers to the supplementary video for a more faithful comparison of expression dynamics, which static frames cannot fully convey.

Training stability. Figure 5 shows the training dynamics of our full DynaForcing. Unlike vanilla self-forcing (Figure 2b), where dynamics metrics peak early and then collapse, DynaForcing’s Sync-C and ExpVar rise within the first ∼1,000 steps and remain stable throughout training. Meanwhile, quality metrics (ASE, IQA) steadily improve and plateau around 4,000 steps. This confirms that our method resolves the quality–dynamics trade-of throughout the entire training process, rather than relying on early stopping at a favorable checkpoint.

![](images/868c18d0ad0c0275cdd9c06f7a7c688fbe671e734a66f405e5aa57d6a3c3abad.jpg)  
Figure 5: Training dynamics of DynaForcing. Unlike vanilla self-forcing (Figure 2b), dynamics metrics (Sync-C, ExpVar) stabilize after ∼1K steps without degradation, while quality metrics improve steadily.

## 5.3 Ablation Studies

All ablations are evaluated on GenBench-ShortVideo unless otherwise noted.

Diagnostic ablation: isolating rollout feedback. To test our hypothesis that unanchored self-conditioning amplifies collapse (§4.1), we compare two training paradigms under identical settings (4,000 steps, same DMD loss, no DynaForcing components): (i) Self-forcing, where each block’s context is built from the student’s own outputs; (ii) GT-anchored (CausVid-style), which follows CausVid [44] and processes all blocks in parallel on noised groundtruth latents with a blockwise causal mask. The core diference between the two paradigms is the conditioning source (student output vs. ground-truth).

Table 2 reveals a stark contrast. Self-forcing achieves slightly higher visual quality (ASE) but severely suppressed dynamics (Sync-C: 5.48, Dyn-Deg: 0.31), confirming dynamic collapse under unanchored self-conditioning. GT-anchored training, which anchors every block to ground-truth context, recovers substantial dynamics (ExpVar: 0.69→1.36, Dyn-Deg: 0.31→0.52) and lip-sync accuracy (Sync-C: 5.48→6.85). This supports our claim that the conditioning source, not DMD distillation itself, is the primary driver of collapse.

Table 2: Diagnostic ablation: isolating rollout feedback. GTanchored training conditions on ground-truth latents; selfforcing conditions on the student’s own outputs.
<table><tr><td>Configuration</td><td>ASE↑</td><td>Sync-C↑</td><td> $\mathrm { E x p V a r } \uparrow$ </td><td>Dyn-Deg↑</td></tr><tr><td>Self-forcing</td><td>3.44</td><td>5.48</td><td>0.69</td><td>0.31</td></tr><tr><td>GT-anchored (CausVid-style)</td><td>3.38</td><td>6.85</td><td>1.36</td><td>0.52</td></tr></table>

Table 3: Component ablation. Each row removes one component from full DynaForcing.
<table><tr><td>Configuration</td><td>ASE↑</td><td>Sync-C↑</td><td>ExpVar↑</td></tr><tr><td>DynaForcing w/o Hybrid Forcing</td><td>3.57</td><td>6.78</td><td>1.18</td></tr><tr><td>DynaForcing w/o Reward Reg.</td><td>3.55</td><td>7.35</td><td>1.75</td></tr><tr><td>DynaForcing w/o Ref. Perturbation</td><td>3.52</td><td>7.58</td><td>1.92</td></tr><tr><td>DynaForcing (full)</td><td>3.55</td><td>7.68</td><td>2.02</td></tr></table>

Table 4: Efect of Hybrid Forcing probability $\pmb { \mathcal { P } } \mathbf { d a t a }$ on full DynaForcing (varying $\mathbf { \phi } \mathbf { \hat { \rho } } \mathbf { \hat { { d a t a } } }$ only).
<table><tr><td> $ { p _ { \mathrm { d a t a } } }$ </td><td>ASE↑</td><td> $\operatorname { S y n c - C \uparrow }$ </td><td>ExpVar↑</td><td>Dino-S↑</td></tr><tr><td>0.1</td><td>3.56</td><td>7.19</td><td>1.42</td><td>0.96</td></tr><tr><td>0.3</td><td>3.55</td><td>7.68</td><td>2.02</td><td>0.96</td></tr><tr><td>0.5</td><td>3.53</td><td>7.64</td><td>2.03</td><td>0.95</td></tr><tr><td>1.0</td><td>3.42</td><td>7.62</td><td>2.10</td><td>0.93</td></tr></table>

However, GT-anchored training alone does not fully close the gap to the teacher (ExpVar still below 1.81), indicating that reverse-KL mode-seeking contributes independently.

Component-wise ablation. Table 3 presents component contributions. Removing Hybrid Forcing causes the largest drop in both dynamics (ExpVar 2.02→1.18) and lip-sync (Sync-C 7.68→6.78), confirming data-level anchoring as the most critical component; notably, ASE slightly increases (3.55→3.57), consistent with the quality–dynamics trade-of observed in the $ { p _ { \mathrm { d a t a } } }$ sweep. Removing Reward Regularization most degrades Sync-C among the remain ing components (7.68→7.35), as it directly optimizes lip-sync via the dynamics reward. Removing Reference Perturbation has the smallest individual efect but still contributes (ExpVar 2.02→1.92). All three together achieve the best overall balance.

Hybrid Forcing probability. Table 4 sweeps the mixing probability. $p _ { \mathrm { d a t a } } = 0 . 3$ achieves the best overall balance. Reducing �<sub>data</sub> to 0.1 yields slightly higher visual quality (ASE 3.56) but substantially degrades lip-sync (Sync-C 7.68→7.19) and dynamics (Exp-Var 2.02→1.42), as the model receives insuficient ground-truth anchoring. Increasing $ { p _ { \mathrm { d a t a } } }$ beyond 0.3 continues to improve Exp-Var (reaching 2.10 at $p _ { \mathrm { d a t a } } = 1 . 0 )$ thanks to stronger ground-truth anchoring, but at the cost of significant train-test mismatch: the model gets less self-forcing practice, degrading visual quality (ASE $3 . 5 5 {  } 3 . 4 2 )$ and identity consistency (Dino-S 0.96→0.93) at inference. $p _ { \mathrm { d a t a } } = 0 . 3$ thus strikes the best balance between dynamics recovery and generation quality.

Table 5: Training eficiency: full DynaForcing with and without our eficiency optimizations (§4.5), both trained for 4,000 steps.
<table><tr><td></td><td>GPUs</td><td>GPU·h</td><td>ASE↑</td><td> $\operatorname { S y n c - C \uparrow }$ </td><td>ExpVar↑</td></tr><tr><td>w/o eff. optim.</td><td>128×H100</td><td>7,111</td><td>3.56</td><td>7.66</td><td>2.04</td></tr><tr><td>w. eff. optim.</td><td>8×H100</td><td>667</td><td>3.55</td><td>7.68</td><td>2.02</td></tr></table>

Training eficiency. Table 5 compares full DynaForcing trained with and without our eficiency optimizations. Graph pruning combined with block-wise gradient replay reduces the GPU requirement from 128 to 8 H100s, and total GPU-hours from 7,111 to 667 (∼10.7× reduction) despite a ∼1.5× per-step wall-clock overhead from replay. Quality metrics remain essentially unchanged (ASE difers by 0.01, Sync-C by 0.02, ExpVar by 0.02), confirming negligible degradation from the approximations.

User study. A pairwise study with 25 participants finds DynaForcing consistently preferred over both real-time baselines, most strongly on motion naturalness (80.6% vs. LiveAvatar); since raters have no access to our training rewards, this independently supports genuine dynamics recovery (see supplementary material for the full protocol and results).

## 6 Conclusion

In the context of audio-driven avatar generation, we have formalized and systematically characterized dynamic collapse, a failure mode of self-forcing distillation where the reverse KL objective and autoregressive feedback jointly drive the student toward near-static outputs, suppressing the temporal dynamics essential for faithful talking-head synthesis. To address this, we proposed DynaForcing, which intervenes at the data, loss, and conditioning levels of training, together with computation graph pruning and gradient replay that reduce GPU requirements by ∼10×.

Limitations. Our dynamics reward relies on pretrained SyncNet and 3DMM models, whose biases may influence training. Our analysis centers on avatar generation; while we expect the data-level component to transfer, systematic validation across other tasks remains future work. The ∼1.5× overhead from gradient replay may also be further reducible.

## Acknowledgments

This work was supported in part by the grant from National Natural Science Foundation of China (No. 62406264), and the grant from the Anhui Natural Science Foundation (No. 2508085ZD006).

## References

[1] Tim Brooks, Bill Peebles, Connor Holmes, et al. 2024. Video generation models as world simulators. OpenAI Technical Report. https://openai.com/research/videogeneration-models-as-world-simulators

[2] Adrian Bulat and Georgios Tzimiropoulos. 2017. How Far are We from Solving the 2D & 3D Face Alignment Problem? (and a Dataset of 230,000 3D Facia Landmarks). In Proceedings ofthe IEEE International Conference on Computer Vision. 1021–1030.

[3] Joon Son Chung and Andrew Zisserman. 2017. Out of time: automated lip sync in the wild. In Computer Vision–ACCV 2016 Workshops. Springer, 251–263.

[4] Jiahao Cui, Hui Li, Yun Zhan, Hanlin Shang, Kaihui Cheng, Yuqi Ma, Shan Mu, Hang Zhou, Jingdong Wang, and Siyu Zhu. 2025. Hallo3: Highly dynamic and realistic portrait image animation with video difusion transformer. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 21086–21095.

[5] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. 2025. Self-Forcing++: Towards Minute-Scale High Quality Video Generation. arXiv preprint arXiv:2510.02283 (2025)

[6] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. 2019. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4690–4699.

[7] Ariel Ephrat, Inbar Mosseri, Oran Lang, et al. 2018. Looking to listen at the cocktail party: A speaker-independent audio-visual model for speech separation. arXiv preprint arXiv:1804.03619 (2018)

[8] Pierre Fernandez, Guillaume Couairon, Hervé Jégou, Matthijs Douze, and Teddy Furon. 2023. The Stable Signature: Rooting Watermarks in Latent Difusion Models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 22466–22477.

[9] Qijun Gan, Ruizi Yang, Jianke Zhu, Shaofei Xue, and Steven Hoi. 2025. Omni-Avatar: Eficient Audio-Driven Avatar Video Generation with Adaptive Body Animation. arXiv preprint arXiv:2506.18866 (2025)

[10] Xin Gao, Li Hu, Siqi Hu, et al. 2025. Wan-S2V: Audio-Driven Cinematic Video Generation. arXiv:2508.18621

[11] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. 2025. Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Difusion. arXiv preprint arXiv:2506.08009 (2025).

[12] Yubo Huang, Hailong Guo, Fangtai Wu, Shifeng Zhang, Shijie Huang, Qijun Gan, Lin Liu, Sirui Zhao, Enhong Chen, Jiaming Liu, and Steven Hoi. 2025. Live Avatar: Streaming Real-time Audio-Driven Avatar Generation with Infinite Length. arXiv preprint arXiv:2512.04677 (2025).

[13] Yubo Huang, Weiqiang Wang, Sirui Zhao, Tong Xu, Lin Liu, and Enhong Chen. 2026. Bind-Your-Avatar: Multi-Character-Talking Video Generation with Dynamic 3D-mask-based Embedding Router. In Proceedings ofthe Computer Vision and Pattern Recognition Conference (CVPR) Findings. 4440–4449.

[14] Ziqi Huang, Yinan He, et al. 2023. VBench: Comprehensive Benchmark Suite for Video Generative Models. arXiv:2311.17982

[15] Taekyung Ki, Sangwon Jang, Jaehyeong Jo, Jaehong Yoon, and Sung Ju Hwang. 2026. Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation. arXiv preprint arXiv:2601.00664 (2026).

[16] Tianqi Li, Ruobing Zheng, Minghui Yang, Jingdong Chen, and Ming Yang. 2024. Ditto: Motion-space difusion for controllable realtime talking head synthesis. arXiv preprint arXiv:2411.19509 (2024).

[17] Jinxiu Liu, Xuanming Liu, Kangfu Mei, Yandong Wen, Ming-Hsuan Yang, and Weiyang Liu. 2026. Streaming Autoregressive Video Generation via Diagonal Distillation. In ICLR. arXiv:2603.09488.

[18] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. 2025. Rolling Forcing: Autoregressive Long Video Difusion in Real Time. arXiv preprint arXiv:2509.25161 (2025).

[19] Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, Yujun Shen, and Min Zhang. 2025. Reward Forcing: Eficient Streaming Video Generation with Rewarded Distribution Matching Distillation. arXiv preprint arXiv:2512.04678 (2025).

[20] Weijian Luo. 2024. Dif-instruct++: Training one-step text-to-image generator model to align with human preferences. arXiv preprint arXiv:2410.18881 (2024).

[21] Yihong Luo, Tianyang Hu, Jiacheng Sun, Yujun Cai, and Jing Tang. 2025. Learning Few-Step Difusion Models by Trajectory Distribution Matching. arXiv preprint arXiv:2503.06674 (2025).

[22] Rang Meng, Xingyu Zhang, Yuming Li, and Chenguang Ma. 2025. EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation. arXiv:2411.10061

[23] Kevin P Murphy. 2012. Machine Learning: A Probabilistic Perspective. MIT press.

[24] Maxime Oquab, Timothée Darcet, Théo Moutakanni, et al. 2024. DINOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research (2024).

[25] Jan Peters and Stefan Schaal. 2008. Reinforcement learning of motor skills with policy gradients. Neural Networks 21, 4 (2008), 682–697.

[26] KR Prajwal, Rudrabha Mukhopadhyay, Vinay P Namboodiri, and CV Jawahar. 2020. A lip sync expert is all you need for speech to lip generation in the wild. In Proceedings ofthe 28th ACM international conference on multimedia. 484–492.

[27] Tim Salimans and Jonathan Ho. 2022. Progressive distillation for fast sampling of difusion models. In International Conference on Learning Representations.

[28] Le Shen, Qian Qiao, Tan Yu, Ke Zhou, Tianhang Yu, Yu Zhan, Zhenjie Wang, Ming Tao, Shunshun Yin, and Siyuan Liu. 2025. SoulX-FlashTalk: Real-Time Infinite Streaming of Audio-Driven Avatars via Self-Correcting Bidirectional Distillation. arXiv preprint arXiv:2512.23379 (2025).

[29] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. 2023. Consistency models. In International Conference on Machine Learning (ICML).

[30] Zhiyao Sun et al. 2025. StreamAvatar: Streaming Difusion Models for Real-Time Interactive Human Avatars. arXiv preprint arXiv:2512.22065 (2025).

[31] Image Team et al. 2025. Z-Image: An Eficient Image Generation Foundation Model with Single-Stream Difusion Transformer. arXiv:2511.22699

[32] Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. 2024. Emo: Emote portrait alive generating expressive portrait videos with audio2video difusion model under weak conditions. In European Conference on Computer Vision. Springer, 244–260.

[33] Shuyuan Tu, Yueming Pan, Yinming Huang, Xintong Han, Zhen Xing, Qi Dai, Chong Luo, Zuxuan Wu, and Yu-Gang Jiang. 2025. Stableavatar: Infinite-length audio-driven avatar video generation. arXiv preprint arXiv:2508.08248 (2025).

[34] Team Wan, Ang Wang, Baole Ai, et al. 2025. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025).

[35] Lizhen Wang, Yongming Zhu, Zhipeng Ge, Youwei Zheng, Longhao Zhang, and Tianshu Hu. 2026. FlowAct-R1: Towards Interactive Humanoid Video Generation. arXiv preprint arXiv:2601.10103 (2026).

[36] Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A Efros. 2020. CNN-Generated Images Are Surprisingly Easy to Spot. . . for Now. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8695–8704.

[37] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13, 4 (2004), 600–612.

[38] Chenfei Wu et al. 2025. Qwen-Image Technical Report. arXiv:2508.02324

[39] Haoning Wu, Zicheng Zhang, Weixia Zhang, et al. 2023. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. arXiv preprint arXiv:2312.17090 (2023).

[40] Yifan Xu, Sirui Zhao, Shifeng Liu, Tong Xu, and Enhong Chen. 2026. Emotionally Controllable Audio-driven Talking Face Generation. ACM Transactions on Multimedia Computing, Communications, and Applications 22, 2 (2026). doi:10.1145/3779219

[41] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. 2025. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622 (2025).

[42] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. 2024. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems 37 (2024), 47455–47487.

[43] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. 2024. One-step difusion with distribution matching distillation. In Proceedings ofthe IEEE/CVFconference on computer vision and pattern recognition. 6613–6623.

[44] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. 2025. From slow bidirectional to fast autoregressive video difusion models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 22963–22974.

[45] Wenxuan Zhang, Xiaodong Cun, Xuan Wang, Yong Zhang, Xi Shen, Yu Guo, Ying Shan, and Fei Wang. 2023. Sadtalker: Learning realistic 3d motion coeficients for stylized audio-driven single image talking face animation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 8652–8661.

[46] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. 2026. Causal Forcing: Autoregressive Difusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation. arXiv preprint arXiv:2602.02214 (2026).

[47] Yongming Zhu, Longhao Zhang, Zhengkun Rong, Tianshu Hu, Shuang Liang, and Zhipeng Ge. 2025. INFP: Audio-driven interactive head generation in dyadic conversations. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 10667–10677.

## A Overview of Supplementary Material

This supplementary document provides additional details and experiments to support the main paper. The content is organized as follows:

• Section B: Reference Perturbation Prompts. Full prompt templates used for generating perturbed reference images via Qwen-Image-Edit.

• Section C: Reproduction Setup. Training configurations used to reproduce the dynamic collapse curves in Figure 2 (main paper).

• Section D: Additional Ablation Studies. Supplementary ablations on reward strength �, starting-step distribution, and reward components.

• Section E: Perturbation Strategy Comparison. Comparison of Qwen-Image-Edit against traditional data augmentation for reference perturbation.

• Section F: Reward-Free Dynamics Evaluation. Landmarkbased motion statistics that verify dynamics recovery is genuine rather than reward hacking.

• Section G: User Study Details. Protocol, setup, and full per-dimension results of the perceptual user study.

• Section H: Ethical Considerations. Discussion of potential risks and responsible deployment practices.

## B Reference Perturbation Prompts

We use Qwen-Image-Edit [38] to generate perturbed reference images that preserve identity but vary in other visual attributes. For each training identity, we apply four prompt categories as shown in Table 6. For each source identity, we generate 8–12 perturbed variants spanning all categories. All generated images are filtered by ArcFace [6] cosine similarity > 0.9 to ensure identity consistency before entering the similarity-calibrated bucketing pipeline described in the main paper (Section 4.3).

## C Reproduction Setup for Figure 2 (main paper)

All training curves in Figure 2 (main paper) are produced by reproducing the distillation stage of existing methods using their oficial codebases. For the text-to-video experiment (Figure 2 (main paper)a), we use the oficial Self-Forcing [11] code with its default configuration and Stage-1 pretrained weights, then run self-forcing (Stage 2) for 8K steps. For the audio-driven avatar experiment (Figure 2 (main paper)b), we similarly use the oficial LiveAvatar [12] code with its released Stage-1 checkpoint and default hyperparameters, then run self-forcing distillation for 8K steps. For the CausVid comparison (Figure 2 (main paper)c), we use the oficial CausVid [44] codebase with its released ODE initialization check point and default DMD training configuration, training for 8K steps on the same evaluation prompts. In all cases, hyperparameters strictly follow the respective original papers; we only extend the training horizon to observe the long-term dynamics trajectory. VBench [14] dynamic degree is computed at intermediate checkpoints for (a) and (c), while Sync-C, ExpVar, IQA, and ASE are evaluated on a held-out set for (b).

Table 6: Full prompts for reference image perturbation via Qwen-Image-Edit. Each category targets a specific visual attribute while explicitly constraining identity-related features to remain unchanged. {·} denotes a slot randomly sampled from the provided options.

<table><tr><td rowspan=1 colspan=1>Category 1: Background Replacement</td></tr><tr><td rowspan=1 colspan=1>Change the background to {a studio with plain gray backdrop /an outdoor park scene / a dimly lit office}; keep the person&#x27;sface, hairstyle, expression, and clothing unchanged.</td></tr><tr><td rowspan=1 colspan=1>Category 2: Lighting Variation</td></tr><tr><td rowspan=1 colspan=1>Adjust the lighting to {warm golden-hour sunlight from theleft / cool fluorescent overhead lighting / dramatic sidelighting with strong shadows}; keep the person&#x27;s appearanceunchanged.</td></tr><tr><td rowspan=1 colspan=1>Category 3: Viewpoint Shift</td></tr><tr><td rowspan=1 colspan=1>Slightly rotate the person&#x27;s head to {face 15 degrees tothe left / tilt slightly upward / turn to a three-quarterview}; keep the person&#x27;s identity, hairstyle, and clothingunchanged.</td></tr><tr><td rowspan=1 colspan=1>Category 4: Accessory Modification</td></tr><tr><td rowspan=1 colspan=1>{Add a pair of thin-framed glasses / Remove earrings /Add a dark scarf around the neck}; keep the person&#x27;s face,hairstyle, and expression unchanged.</td></tr></table>

## D Additional Ablation Studies

All supplementary ablations are evaluated on GenBench-ShortVideo with the full DynaForcing model, varying only the parameter under study.

Reward strength �. Table 7 sweeps the reward strength � in Eq. 7 (main paper). At $\alpha = 0$ (no reward weighting), dynamics degrade to the level of the “w/o Reward Reg.” ablation in the main paper (Table 3 (main paper)). Increasing � to 0.1 yields the best quality–dynamics balance. Beyond 0.2, the exponential weighting over-amplifies high-reward samples, causing visual quality to drop (ASE: 3.55→3.48 at � = 0.5) without further dynamics gains (ExpVar saturates around 2.06–2.08). We select � = 0.1 for all experiments.

Table 7: Efect of reward strength � on full DynaForcing.
<table><tr><td>α</td><td>ASE↑</td><td>Sync-C↑</td><td>ExpVar↑</td></tr><tr><td>0.0</td><td>3.55</td><td>7.35</td><td>1.75</td></tr><tr><td>0.05</td><td>3.55</td><td>7.52</td><td>1.88</td></tr><tr><td>0.1</td><td>3.55</td><td>7.68</td><td>2.02</td></tr><tr><td>0.2</td><td>3.53</td><td>7.70</td><td>2.08</td></tr><tr><td>0.5</td><td>3.48</td><td>7.65</td><td>2.06</td></tr></table>

Starting-step probability distribution. Table 8 compares diferent distributions for sampling $t _ { \mathrm { s t a r t } }$ in data-anchored mode (Eq. 5 (main paper)). “Conc. $t _ { k } \ "$ concentrates 70% mass on step $t _ { k }$ with 15% on each of the other two. Concentrating on $t _ { 0 }$ (highest SNR) wastes training on near-trivial denoising, while concentrating on $t _ { 2 }$ (lowest SNR) loses too much ground-truth structure. Concentrating on �<sub>1</sub> achieves the best balance, and outperforms the uniform baseline.

Table 8: Efect of starting-step probability distribution.
<table><tr><td>Distribution</td><td>ASE↑</td><td>Sync-C↑</td><td>ExpVar↑</td></tr><tr><td>Uniform</td><td>3.54</td><td>7.51</td><td>1.86</td></tr><tr><td>Conc. t0</td><td>3.55</td><td>7.42</td><td>1.78</td></tr><tr><td>Conc. t1</td><td>3.55</td><td>7.68</td><td>2.02</td></tr><tr><td>Conc. t2</td><td>3.52</td><td>7.55</td><td>1.91</td></tr></table>

Reward component analysis. Table 9 isolates the contribution of each reward component in Eq. 8 (main paper). Using only Sync-C reward $( \lambda _ { \mathrm { e x p } } = 0 )$ primarily improves lip-sync but provides limited expression dynamics recovery. Using only ExpVar reward $( \lambda _ { \mathrm { s y n c } } =$ 0) improves expression variance but is less efective at lip-sync. Combining both yields the best overall performance, confirming their complementary roles.

Table 9: Ablation on reward components in $R _ { \mathbf { d y n } }$
<table><tr><td>Reward</td><td>ASE↑</td><td>Sync-C↑</td><td>ExpVar↑</td></tr><tr><td>None (α = 0)</td><td>3.55</td><td>7.35</td><td>1.75</td></tr><tr><td>Sync-C only</td><td>3.55</td><td>7.62</td><td>1.85</td></tr><tr><td>ExpVar only</td><td>3.54</td><td>7.48</td><td>1.96</td></tr><tr><td>Both (full)</td><td>3.55</td><td>7.68</td><td>2.02</td></tr></table>

## E Perturbation Strategy Comparison

Reference Perturbation (main paper, Section 4.3) uses Qwen-Image-Edit to generate semantically edited references. A natural question is whether this comparatively heavy pipeline is necessary, or whether conventional data augmentation would sufice. We therefore replace Qwen-Image-Edit with traditional augmentations while keeping the ArcFace filtering and similarity-calibrated bucketing unchanged. For each reference we randomly apply one of: (1) patch dropout (30 × 30 patches, drop rate ∼U(0, 0.5)), (2) color jitter (strength ∼U(0, 0.8)), or (3) perspective rotation (∼U(0<sup>◦</sup>, 30<sup>◦</sup>)).

Table 10: Perturbation strategy comparison on GenBench-ShortVideo.
<table><tr><td>Perturbation</td><td>ASE↑</td><td>Sync-C↑</td><td>ExpVar↑</td></tr><tr><td>None</td><td>3.52</td><td>7.58</td><td>1.92</td></tr><tr><td>Traditional Aug.</td><td>3.48</td><td>7.70</td><td>2.03</td></tr><tr><td>Qwen-Image-Edit (Ours)</td><td>3.55</td><td>7.68</td><td>2.02</td></tr></table>

Table 10 shows two things. First, the dynamics benefit of reference perturbation is largely robust to the specific strategy: traditional augmentation recovers Sync-C and ExpVar to essentially the same level as semantic editing, indicating that what matters is breaking the pixel-level correspondence between reference and target rather than the particular mechanism used. Second, traditional augmentation slightly degrades visual quality (ASE: 3.55→3.48), plausibly because unrealistic corruptions shift references of the natural image distribution, whereas semantic edits remain in-distribution. We also note that Qwen-Image-Edit is applied ofline to the 32K training samples used for distillation, so its cost is negligible relative to training.

## F Reward-Free Dynamics Evaluation

Sync-C and ExpVar are used as training rewards, so improvements on these two metrics alone cannot exclude reward hacking. The main paper therefore highlights Dyn-Deg, which is not optimized during training. Here we supplement two further reward-free statistics computed from 2D facial landmarks extracted with FAN [2], using the 68-point mouth region: Lip-Vel, the mean first-order displacement $\| L _ { t } - L _ { t - 1 } \|$ of lip landmarks, which measures motion magnitude; and Lip-Jit, the mean second-order diference $\left\| L _ { t + 1 } - 2 L _ { t } + L _ { t - 1 } \right\|$ , which measures smoothness and is sensitive to jitter. Higher Lip-Vel indicates more lip motion; lower Lip-Jit indicates smoother motion.

Table 11: Reward-free dynamics evaluation. Sync-C and ExpVar are training rewards; Dyn-Deg, Lip-Vel and Lip-Jit are not optimized.
<table><tr><td rowspan="2">Method</td><td colspan="2">Reward metrics</td><td colspan="3">Reward-free</td></tr><tr><td>Sync-C↑ ExpVar↑</td><td></td><td>Dyn-Deg↑ Lip-Vel↑ Lip-Jit↓</td><td></td><td></td></tr><tr><td>LiveAvatar</td><td>7.03</td><td>0.69</td><td>0.31</td><td>3.50</td><td>8.64</td></tr><tr><td>DynaForcing w/o Reward Reg.</td><td>7.35</td><td>1.75</td><td>0.55</td><td>3.41</td><td>4.23</td></tr><tr><td>DynaForcing (full)</td><td>7.68</td><td>2.02</td><td>0.73</td><td>3.78</td><td>4.26</td></tr></table>

Table 11 shows that adding Reward Regularization improves not only the reward metrics but also the reward-free ones: Dyn-Deg rises from 0.55 to 0.73 and Lip-Vel from 3.41 to 3.78. Critically, Lip-Jit is essentially unchanged (4.23 vs. 4.26), meaning the additional motion is not accompanied by increased frame-to-frame jitter. This addresses the concern that DynaForcing’s ExpVar exceeding the teacher’s (2.02 vs. 1.81) might reflect over-expression or unnatural trembling rather than richer dynamics: if the reward were being hacked by injecting high-frequency noise, Lip-Jit would rise sharply. Both baselines and DynaForcing also improve markedly over LiveAvatar’s Lip-Jit of 8.64, whose high value reflects the unstable residual motion left after collapse. We emphasize that Lip-Vel and Lip-Jit are statistics we define here rather than established benchmarks; they are intended as corroborating evidence alongside Dyn-Deg and the user study (Section G).

## G User Study Details

We conduct a pairwise user study comparing DynaForcing against the two strongest competing real-time methods, LiveAvatar [12] and SoulX-FlashTalk [28]. We recruit 25 participants (14 male, 11 female, aged 22–35) who are naïve to the purpose of the study. Each trial presents two videos side-by-side (one from DynaForcing, one from a baseline) with synchronized audio. Left/right placement and trial order are randomized. Participants select their preference on four dimensions: visual quality, lip-sync accuracy, motion naturalness, and overall preference. Each participant evaluates 40 trials (20 samples × 2 baselines), drawn from GenBench-ShortVideo. Paired �-tests confirm statistical significance (� < 0.01) across all four dimensions.

Table 12: User study: preference rate (%) for DynaForcing over the two strongest competing real-time methods.
<table><tr><td>vs.</td><td>Visual</td><td>Lip-Sync</td><td>Motion</td><td>Overall</td></tr><tr><td>LiveAvatar</td><td>58.4</td><td>71.2</td><td>80.6</td><td>72.8</td></tr><tr><td>SoulX-FlashTalk</td><td>55.2</td><td>64.8</td><td>74.2</td><td>66.4</td></tr></table>

Table 12 shows that DynaForcing is consistently preferred over both baselines on every dimension. The margin is largest on motion naturalness (80.6% and 74.2%), which is the dimension our method directly targets, and smallest on visual quality (58.4% and 55.2%), consistent with the narrow ASE/IQA gaps reported in the main paper. Since raters have no access to the training rewards, these preferences corroborate the reward-free metrics of Section F in indicating that the recovered dynamics are genuine rather than an artifact of reward optimization.

## H Ethical Considerations

Audio-driven avatar generation carries the risk of misuse for creating unauthorized impersonations. We intend our method exclusively for legitimate applications (virtual assistants, telepresence, authorized content creation) and advocate integrating invisible watermarking [8] and deepfake detection [36] into deployment pipelines. We train on AVSpeech [7], a publicly available dataset with no personally identifiable metadata. Our eficient training strategy reduces compute from ∼7,111 to ∼667 H100 GPU-hours, lowering energy consumption accordingly.