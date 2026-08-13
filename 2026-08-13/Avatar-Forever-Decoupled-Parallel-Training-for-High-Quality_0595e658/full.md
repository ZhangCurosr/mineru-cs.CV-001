POLYU VCLAB •PREPRINT 2026

# Avatar-Forever: Decoupled Parallel Training for High-Quality Real-Time Infinite Avatars

Ruibin Li<sup>1★</sup> Tao Yang<sup>2</sup> Zhiyuan Ma<sup>1</sup> Fangzhou Ai<sup>3</sup> Shilei Wen<sup>2</sup> Lei Zhang<sup>†1</sup>

<sup>1</sup> The Hong Kong Polytechnic University <sup>2</sup> ByteDance <sup>3</sup> AMD

<sup>★</sup> Work done during internship in ByteDance. <sup>†</sup> Corresponding author (cslzhang@comp.polyu.edu.hk).

Project Page Code Demo BibTeX

Abstract. Existing streaming video systems often rely on sequential, distillation-centered training pipelines to enable few-step long-video generation. However, this paradigm sufers from two limitations. First, failures or distribution shifts introduced in earlier stages afect later optimization, complicating the training process to converge. Second, the distillation-centric objective favours short-term generation but is prone to quality degradation when autoregressive errors accumulate over long rollouts. We propose Avatar-Forever, a decoupled parallel training framework for high-quality real-time infinite interactive avatars. Instead of coupling generation eficiency and long-horizon robustness under a sequential distillation pipeline, we treat them as two independent capabilities that can be trained in parallel. One branch performs full-parameter distillation to train an eficient generator with high visual quality, while another trains a lightweight long-horizon adapter via Recovery-oriented Rollout Training (RRT), which improves generation robustness under long-horizon inference conditions. Our decoupled parallel training design simplifies the overall training process and avoids unnecessary objective conflicts between few-step generation and long-horizon adaptation. We further introduce ForeverCache, a chunk-wise feature caching mechanism to substantially reduce redundant history computation during streaming inference. Built upon a 22B video foundation model, Avatar-Forever supports unbounded audio-driven avatar generation while maintaining identity consistency, motion coherence, and visual fidelity, enabling an end-to-end throughput of high-resolution 768 × 512 videos at 27.2 FPS on a single H100 GPU and providing a practical path toward stable digital humans.

KEYWORDS : Interactive Avatars, Streaming Video Generation, Real-Time Video Generation

## 1 Introduction

Generating realistic and expressive digital humans is highly demanded in applications such as virtual assistants, content creation, education, and entertainment [1–3]. A practical avatar system should continuously synthesize coherent video over extended durations while maintaining identity consistency, audio-driven motion, visual fidelity, and eficient inference. Recent difusion-based video foundation models [4–8] have significantly advanced audio-driven avatar generation, producing photorealistic talking avatars with improved visual fidelity, motion realism, and lip synchronization [9–12]. However, most models are trained to generate a short clip within a fixed temporal window, and extending these advances from short clips to real-time long-duration avatar generation remains challenging [13–15]. To generate longer outputs, streaming systems typically reuse previously generated chunks as conditions for future chunks [16–20], which introduces distribution drift and error accumulation: early artifacts, identity drift, or lip-sync errors become part of the historical context and gradually degrade subsequent generation [21, 22].

Recent works [15, 23, 24] usually address this challenge with distillation-centered sequential training pipelines, as illustrated in the top of Figure 1. Starting from a multi-step bidirectional video model, these methods combine teacher forcing, ODE initialization, DMD distillation, and additional forcing strategies [13, 14, 19, 25–

![](images/dedceac70cd483f810d129968385dc7cc915ba8a569f217527590ab732328b09.jpg)  
Figure 1. Overview of Avatar-Forever. Top: Existing streaming avatar systems employ sequential distillation pipelines, leading to stage-wise dependence and objective interference. Bottom: Avatar-Forever learns few-step eficiency and long-horizon robustness in parallel. A lightweight adapter is trained under accumulated rollout errors, and it is merged with the distilled generator in inference. In addition, ForeverCache is proposed to reuse historical features for eficient streaming inference.

27] with supervision imposed in fixed denoising steps to obtain a few-step model. This sequential design entangles two distinct objectives in the optimization pipeline: eficient few-step generation, which preserves short-term visual quality under reduced denoising steps, and long-horizon robustness, which stabilizes generation against accumulated autoregressive errors. Such an entanglement makes the training fragile and dificult to scale: distribution shifts introduced at earlier stages afect later optimization, making it hard to diagnose and obscuring the contribution of each stage.

We argue that the two objectives of eficient few-step generation and long-horizon robustness should be decoupled and learned in parallel, and propose Avatar-Forever, a decoupled parallel training framework for high-quality real-time long-duration audio-driven avatars, as illustrated in the bottom of Figure 1. Instead of intertwining distillation, forcing schedules and long-context adaptation in a sequential pipeline, we train for eficiency and robustness in parallel. One branch performs full-parameter distillation to obtain a high-quality few-step generator, while the other trains a lightweight long-horizon adapter for stability under extended rollout conditions. This design simplifies the training procedure, making each objective easier to optimize and diagnosis. For the long-horizon adaptation branch, we introduce Recovery-oriented Rollout Training (RRT), which targets the main failure mode of autoregressive generation, i.e., errors accumulate after recursive rollout. Starting from a degraded historical context that simulates generation artifacts, the model rolls out multiple future chunks using its own predictions as conditions. Supervision is applied when the degradation has propagated through the rollout, rather than at every local reconstruction step. Importantly, RRT uses a standard flow matching objective and does not require specialized long-video distillation losses or complicated regularization schemes.

During streaming inference, each new chunk needs to attend to historical chunks. One simple way is to recompute the entire visible history window at every denoising step, although only the current chunk is updated. We introduce ForeverCache, a chunk-wise autoregressive history feature caching mechanism to remove redundant computation. For each generated chunk, we perform a full forward pass at the first denoising step to populate per-block historical context features, and then reuse these stable features in subsequent denoising steps while forwarding only the current chunk tokens. This inference-time mechanism preserves the historical context needed for identity consistency, motion continuity, and audio-driven synchronization, while improving throughput by 23% during long-horizon generation.

To support scalable adaptation, we construct a fully synthetic data pipeline by using video foundation models to synthesize training videos from high-quality conversational prompts, followed by automatic filtering to remove low-quality, static, or semantically inconsistent samples. Built upon a 22B video foundation model [6] and our synthetic dataset, Avatar-Forever scales audio-driven avatar generation to extended durations while maintaining visual fidelity, identity consistency, and motion coherence. Specifically, Avatar-Forever achieves an end-to-end (including DiT inference and VAE decoding) generation throughput of 27.2 FPS for high-quality 768 × 512 avatar videos on a single H100 GPU, achieving state-of-the-art performance on long-horizon audio-driven avatar generation. Our main contributions are summarized as follows:

• We propose Avatar-Forever, a decoupled parallel training paradigm for real-time long-duration avatar generation, showing that few-step eficiency and autoregressive robustness can be efectively learned in parallel, rather than in a complicated sequential manner.

• We introduce Recovery-oriented Rollout Training (RRT) in order to recover from errors after propagating through multi-chunk autoregressive rollout. Unlike local corrupted-context reconstruction or long-video distillation objectives, RRT uses standard flow matching supervision under the rollout distribution encountered at inference time.

• We introduce ForeverCache, a chunk-wise history feature cache that reuses stable context features across denoising steps, removing redundant recomputation of historical chunks during streaming inference and improving long-horizon generation throughput by 23%.

• We implement the framework with a fully synthetic data pipeline and a 22B video foundation model, achieving high-quality long-duration audio-driven avatar generation with persistent identity, coherent motion, accurate lip synchronization, and real-time 768 × 512 resolution inference at 27.2 FPS on a single H100 GPU.

Remarks. Few-step eficiency and long-horizon stability operate on diferent temporal scales. Coupling them as one distillation objective entangles speed with robustness, complicating the training pipeline. Avatar-Forever learns to achieve them in a parallel manner, significantly improving the training process.

## 2 Related Work

## 2.1 Audio-Driven Avatar Video Generation

Traditional and 3D-based Methods. Audio-driven avatar generation has been widely studied through 2D talking-head synthesis [28], explicit facial representations [29], and neural rendering [30]. Early systems primarily focus on accurate lip synchronization and identity preservation from limited visual context [31]. To improve structural stability, later methods introduce 3D priors [32], facial motion coeficients [1], or neural radiance fields [33], mapping speech to geometrically meaningful facial dynamics before rendering. These methods provide strong priors for lip motion, head pose, and identity consistency, but they are basically task-specific generators or subject-specific renders, ofering limited flexibility to diverse identities, expressive body motion, complex scenes, and open-ended behaviors [27]. Their reliance on predefined structures or identity-specific rendering also limits the generalization to unconstrained avatar scenarios, especially for long-duration generation.

Difusion-based Avatars. Difusion models [34–36] have recently become increasingly important backbones for avatar synthesis, as they can capture subtle facial dynamics, photorealistic appearance, and temporal variations. Task-specific avatar systems such as DifTalk [37], EMO [3], Hallo [38], AniPortrait [39] and EchoMimic [40] adapt difusion models to audio-driven portrait animation through audio-conditioned denoising, hierarchical control, or editable motion guidance. Meanwhile, large video foundation models, exemplified by Wan [5], Seedance 2.0 [4], Kling-Omni [41], and Sora [42], have demonstrated that scaled difusion-transformer video generators can serve as strong spatiotemporal priors, although they are not specifically designed for audio-driven avatar generation. However, these models introduce substantially high inference cost, making fast sampling a critical bottleneck. While difusion acceleration methods have been developed to reduce the denoising steps [43–48], they mainly address the problem of eficient short-horizon generation, whereas long-duration avatar synthesis also requires robustness under recursive autoregressive rollout.

## 2.2 Long-Horizon and Streaming Video Difusion Models

Streaming video generation [19, 27, 49, 50] reuse previously generated frames as conditions for future predictions to conduct long-horizon video generation. However, this creates a train–test mismatch: models are trained on clean ground-truth contexts but must condition on their own imperfect outputs at inference, causing artifacts, motion drift, and temporal inconsistency to accumulate over time [21, 23, 24]. To reduce the train–test mismatch in autoregressive long-video generation, recent methods expose models to self-generated contexts during training or distillation. Self-Forcing++ [25], Causal Forcing [19], Rolling Forcing [51], and Hybrid Forcing [50] introduce mechanisms such as teacher-guided rollout, auto-regressive teacher initialization, progressive denoising, and hybrid attention design to improve long-horizon stability. These forcing strategies substantially improve the stability and quality of streaming generation. However, they often rely on tightly coupled multi-stage pipelines, making component-wise diagnosis dificult and scaling to large video foundation models cumbersome.

Long-duration avatar generation faces the same autoregressive streaming challenge, but with stricter requirements on identity preservation, appearance stability, lip synchronization, facial dynamics, and human motion coherence [8, 23, 52, 53]. Recent systems such as LiveAvatar [27], StreamAvatar [15], SoulX-FlashTalk [24], and LPM 1.0 [23] adapt large video priors to streaming or long-horizon human synthesis, showing the potential of video foundation models for interactive digital humans. However, these methods largely follow the sequential training paradigm of generic long-video generation, where multiple stages are coupled through autoregressive rollout, distillation, and complicated foring strategies. As a result, errors and distribution shifts [21, 25] introduced in early stages can propagate through the pipeline, making the final behavior hard to diagnose and costly to scale to large video foundation models.

## 3 Method

As illustrated in the bottom of Figure 1, Avatar-Forever consists of three core designs. An eficiency branch and a robustness branch are trained separately via decoupled adaptation. The full-parameter eficiency branch is distilled to preserve high-quality few-step generation, while the robust branch is adapted to improve long-horizon stability. The resulting long-horizon LoRA adapter is merged into the distilled generator at inference. Within the robustness branch, we propose Recovery-oriented Rollout Training (RRT), which perturbs early historical context, propagates the degradation through multi-step autoregressive rollout, and applies standard flow-matching supervision only after errors have accumulated. In addition, a ForeverCache strategy is proposed to accelerate streaming inference by caching stable historical context features across denoising steps, avoiding redundant recomputation of clean history chunks while preserving long-range conditioning.

## 3.1 Eficiency Branch: Few-Step Distillation

The eficiency branch aims to obtain a high-quality few-step generator from the pretrained video foundation model. We adopt Distribution Matching Distillation (DMD) [44] to compress the original multi-step LTX base model [6] into a four-step generator while preserving conditional generation quality. Let $\theta _ { 0 }$ denote the pretrained model parameters, $G _ { \theta }$ denote the student generator, and c denote the conditioning inputs, including text and optional visual conditions. DMD optimizes the student by matching the generated distribution to the teacher distribution through a reverse-KL objective, formulated as follows:

$$
\nabla _ { \theta } \mathcal { L } _ { \mathrm { D M D } } = - \mathbb { E } _ { t , \epsilon , \mathrm { c } } \left[ \left( s _ { \mathrm { r e a l } } \big ( \tilde { \mathbf { x } } _ { t } \big ) - s _ { \mathrm { f a k e } } \big ( \tilde { \mathbf { x } } _ { t } \big ) \right) \frac { \partial G _ { \theta } ( \epsilon , \mathbf { c } ) } { \partial \theta } \right] ,\tag{1}
$$

where $\tilde { \mathbf { X } } _ { t }$ is obtained by applying the forward difusion process to the student sample $G _ { \theta } ( \epsilon , { \bf c } )$ , and $S _ { \mathrm { { r e a l } } }$ and $S f a \mathrm { k e }$ are score functions. This branch is responsible only for few-step eficiency. We do not introduce autoregressive rollout, corrupted history, or long-horizon objectives during distillation. To preserve compatibility with downstream avatar generation, we train the distilled model under mixed conditioning: each sample is randomly used either as a text-to-video instance or as a first-frame-conditioned instance, where the first frame serves as the visual condition. This preserves both T2V and I2V capabilities of the base model while keeping the distillation objective focused on short-horizon quality, motion realism, and eficient sampling.

## 3.2 Robustness Branch: Recovery-oriented Rollout Training

The robustness branch addresses the train–test mismatch introduced by autoregressive self-conditioning. At inference time, generated chunks become historical context for future chunks, so early errors can be recursively reused and amplified, causing identity drift, motion incoherence, and appearance artifacts [21, 24]. Training only on clean histories may lead to failures when exposed to deployment-time error distribution. A natural remedy is corrupted-history training, as explored in Helios [21]. However, local corrupted-context reconstruction only approximates an isolated perturbation, while long-horizon failure arises from accumulated self-generated drift. We therefore introduce Recovery-oriented Rollout Training (RRT): we perturb an early history chunk, let the model propagate this perturbation through multi-step autoregressive rollout, and apply supervision after errors have accumulated. RRT thus trains recovery under model-induced context drift rather than one-step synthetic corruption. The details of RRT are illustrated in Figure 2.

![](images/c2033a198727d87e142ed353f1dfb8d07e56a18ddae898b0ff83963bba85566d.jpg)  
Figure 2. Recovery-oriented Rollout Training (RRT). Top: RRT degrades an early history chunk, propagates the resulting error through � autoregressive rollout steps without gradients, and applies standard FM supervision only to the subsequent recovery step. Bottom: In LTX DiT, video and audio conditions are encoded into multimodal tokens; the gated first-frame pathway and video-side LoRA modules are trained, while the remaining components are frozen.

Global Reference Conditioning. We use the first frame as a persistent visual anchor. It is encoded into a latent representation r and injected into the denoising tokens through a lightweight gated channel-conditioning module. Unlike autoregressive history, which may drift during rollout, r remains fixed and provides stable guidance for identity, appearance, and scene layout.

Early-history Perturbation. We partition each training video latent $\mathbf { x } _ { 0 }$ into chunks $\{ \mathbf { c } _ { k } \} _ { k = 0 } ^ { K + 1 }$ , where $\mathbf { c } _ { 0 }$ denotes the initial context, $\mathbf { c } _ { 1 : K }$ are intermediate rollout chunks, and $\mathbf { c } _ { K + 1 }$ is the final supervised target chunk. We construct the initial perturbed history as:

$$
\begin{array} { r } { \hat { \mathbf { c } } _ { 0 } = \mathcal { D } ( \mathbf { c } _ { 0 } ) , } \end{array}\tag{2}
$$

where $\mathcal { D } ( \cdot )$ is a stochastic degradation operator. Inspired by Helios [21], D randomly samples from a lightweight family of perturbations, including photometric distortion, additive noise, resolution degradation, partial latent masking, and identity mapping. Importantly, the perturbation is applied only to the earliest conditioning history. All subsequent histories are generated by the model itself and recursively reused as conditions. The initial perturbation therefore serves as a trigger for distribution shift, while later errors emerge from the model’s own rollout dynamics.

Autoregressive Rollout. Let $\mathbf { a } _ { k }$ denote the audio condition aligned with chunk $\mathbf { c } _ { k }$ , and let � denote the text condition. For each intermediate chunk $k = 1 , \ldots , K$ , we initialize from Gaussian noise $\hat { \mathbf { c } } _ { k , T } \sim { \cal N } ( \mathbf { 0 } , \mathbf { I } )$ and perform � denoising steps conditioned on the previous generated chunk:

$$
\hat { \mathbf { c } } _ { k , t } = \mathrm { s g } \left( G _ { \theta } \left( \hat { \mathbf { c } } _ { k , t + 1 } ; \hat { \mathbf { c } } _ { k - 1 , 0 } , \mathbf { r } , \mathbf { a } _ { k } , y \right) \right) , \quad t = T - 1 , \ldots , 0 ,\tag{3}
$$

where $\hat { \mathbf { c } } _ { k - 1 , 0 }$ is the generated result of the previous chunk, $G _ { \theta }$ denotes one denoising step of the sampler, and $\operatorname { s g } ( \cdot )$ stops gradients through the rollout trajectory. As illustrated in the top left of Figure 2, rollout is executed without gradient tracking. The intermediate chunks $\hat { \mathbf { c } } _ { 1 } , \hdots , \hat { \mathbf { c } } _ { K }$ are not directly supervised; their role is to

expose the final prediction to model-induced context drift, so that optimization targets recovery under the same autoregressive error-propagation pattern encountered at inference time.

Masked Flow Matching Objective. After � rollout steps, we supervise only the next ground-truth chunk $\mathbf { c } _ { K + 1 }$ conditioned on the generated history $\hat { \mathbf { c } } _ { K , 0 }$ . Given noise $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ and a flow-matching noise level $\sigma \sim p ( \sigma )$ we corrupt only the target tokens:

$$
\hat { \mathbf { c } } _ { K + 1 , \sigma } = ( 1 - \sigma ) \mathbf { c } _ { K + 1 } + \sigma \pmb { \epsilon } ,\tag{4}
$$

while keeping the rolled-out historical context $\hat { \mathbf { c } } _ { K , 0 }$ clean. We then optimize a simple flow-matching [35] objective on the final target chunk:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { R R T } } = \mathbb { E } _ { \mathbf { c } , \mathbf { a } , \mathbf { y } , \sigma , \epsilon } \left[ \left\| \nu _ { \theta } \left( \mathbf { c } _ { K + 1 , \sigma } , \sigma ; \hat { \mathbf { c } } _ { K , 0 } , \mathbf { r } , \mathbf { a } _ { K : K + 1 } , \mathbf { y } \right) _ { i } - \left( \epsilon - \mathbf { c } _ { K + 1 } \right) _ { i } \right\| _ { 2 } ^ { 2 } \right] , } \end{array}\tag{5}
$$

where $\nu _ { \theta }$ is the velocity network, $\mathbf { a } _ { K : K + 1 }$ denotes the audio condition aligned with the visible context-target window. We do not impose auxiliary losses on the intermediate rollout chunks. Once realistic autoregressive degradation is induced through model-in-the-loop rollout, a standard flow-matching objective on the final target chunk is suficient to train recovery from accumulated long-horizon errors.

Insight. We do not supervise immediate local reconstruction from a degraded context; we supervise recovery after degradation has propagated through autoregressive rollout.

Deployment. Both adaptation branches are initialized from the same pretrained base model $\theta _ { 0 }$ so that their updates can be merged directly at inference. Let $\Delta \theta _ { \mathrm { D M D } }$ denote the dense update learned by the eficiency branch, and let Δ�<sub>RRT</sub> denote the LoRA update learned by the robustness branch. The final Avatar-Forever generator is obtained as

$$
\begin{array} { r } { \theta ^ { \star } = \theta _ { 0 } + \Delta \theta _ { \mathrm { D M D } } + \Delta \theta _ { \mathrm { R R T } } . } \end{array}\tag{6}
$$

This composition keeps the two objectives decoupled during training: full-parameter distillation provides a strong few-step generator, while the LoRA-based RRT branch adds long-horizon robustness with minimal additional training and deployment cost.

## 3.3 ForeverCache: Chunk-wise Autoregressive History Feature Caching

When producing chunk �, the autoregressive model denoises the current latent $\mathbf { c } _ { k , t }$ while attending to a compact history window $\mathcal { H } _ { k } = \{ \mathbf { c } _ { 0 } , \mathbf { c } _ { k - 1 } \}$ , consisting of the first and most recent generated chunks. A naive implementation forwards the entire visible window $\{ \mathcal { H } _ { k } , \mathbf { c } _ { k , t } \}$ through every transformer block at each denoising step, as shown in Figure 3(a). However, this computation is structurally redundant: $\mathbf { c } _ { k , \ast }$ <sub>�</sub> changes throughout denoising, whereas the historical chunks remain fixed clean context.

We introduce ForeverCache, an inference-time feature reuse mechanism that eliminates repeated computation on fixed historical context. For each autoregressive chunk, ForeverCache executes a full-window forward pass only at the initial denoising step, then reuses the resulting historical features throughout the remaining denoising trajectory. This design retains access to long-range context while restricting repeated computation to the evolving current chunk, as illustrated in Figure 3(b).

Formally, let $\sigma _ { t }$ denote the noise level at denoising step �. A standard autoregressive implementation evaluates the velocity network on the complete visible window at every step:

$$
\mathbf { v } _ { k , t } = \nu _ { \theta } \left( [ \mathcal { H } _ { k } , \mathbf { c } _ { k , t } ] , \sigma _ { t } ; \mathbf { r } , \mathbf { a } _ { \mathcal { H } _ { k } : k } , y \right) ,\tag{7}
$$

where $[ \mathcal { H } _ { k } , \mathbf { c } _ { k , t } ]$ is the history-current sequence input to LTX [6]. Only the velocity prediction for the current chunk is retained for updating $\mathbf { c } _ { k , t } ;$ ; the historical chunks act solely as conditioning context.

ForeverCache performs full computation only once at $t = T$ , while collecting historical features from all the � transformer blocks:

$$
\begin{array} { r } { \mathbf { v } _ { k , T } , C _ { k } = \nu _ { \theta } ^ { \mathrm { p o p u l a t e } } \left( [ \mathcal { H } _ { k } , \mathbf { c } _ { k , T } ] , \sigma _ { T } ; \mathbf { r } , \mathbf { a } _ { \mathcal { H } _ { k } ; k } , y \right) , \qquad C _ { k } = \{ C _ { k } ^ { \ell } \} _ { \ell = 1 } ^ { L } . } \end{array}\tag{8}
$$

For all subsequent steps, the model forwards only the current-chunk tokens and retrieves $C _ { k }$ as historical conditioning memory:

$$
{ \mathbf { v } } _ { k , t } = \nu _ { \theta } ^ { \mathrm { r e u s e } } \left( \mathbf { c } _ { k , t } , \sigma _ { t } ; C _ { k } , \mathbf { r } , \mathbf { a } _ { k } , y \right) , \qquad t = T - 1 , \ldots , 0 .\tag{9}
$$

Visual Computing Lab · The Hong Kong Polytechnic University

6 / 24

![](images/33bc826fe1374a1f47989a7d718d2ece2ba83d3d9194ee24c6b6caeb77330d06.jpg)  
Figure 3. ForeverCache for autoregressive streaming inference. Top: Standard autoregressive denoising recomputes fixed history features at every step. Bottom: ForeverCache populates per-block history features once at � = � and reuses them while forwarding only the evolving current chunk thereafter. The cache is reset for each new autoregressive chunk, providing bounded-memory streaming inference without redundant history computation.

Here, $\nu _ { \theta } ^ { \mathrm { p o p u l a t e } }$ and $\nu _ { \theta } ^ { \mathrm { r e u s e } }$ share the same model parameters but difer in their execution mode. At each transformer block, only current tokens are processed as active tokens, while cached features provide historical context for video self-attention, audio self-attention, and cross-modal audio-video attention. The resulting prediction is scattered back to the original autoregressive layout, preserving the external denoising interface of the non-cached implementation. The cache is reset for each autoregressive chunk, ensuring bounded memory and preventing stale reuse as the history window changes. After the initial cache-population step, ForeverCache reduces the dominant transformer cost by forwarding only the evolving current-chunk tokens, while reusing cached video and audio history features to preserve multimodal conditioning for identity consistency, motion continuity, and lip synchronization. Note that ForeverCache is used only at inference time and applies directly to the distilled Avatar-Forever generator without modifying any learned weights.

## 4 Synthetic Data Construction Pipeline

Long-horizon avatar adaptation requires videos with persistent identity, meaningful human motion, and reliable semantic alignment over extended durations. Such data are dificult to collect from the web: real videos often exhibit identity changes, weak text–video correspondence, limited facial or body motion, and inconsistent visual quality. We address this limitation with a fully synthetic pipeline that generates and filters avatar videos using the same video foundation model adapted by our framework. This construction pipeline provides controllable training data that are naturally aligned with the target generation distribution, without large-scale manual curation of long-form real videos.

Dialogue and Prompt Synthesis. We collect conversational text from the publicly available MDD corpus [54] and use GPT [55] to filter and refine samples that are incoherent, low-quality, or visually uninformative. The retained dialogues are converted into structured video prompts following the LTX prompting format. Each prompt specifies the character, appearance, scene, camera view, facial expression, body motion, and speech content, encouraging visible articulation and interaction-relevant motion in the resulting videos, as illustrated in the red part of Figure 4.

![](images/f3ce31a295d13b5bdcfb671aacaf55effc3c214108e481a849bbe751d4d70d80.jpg)  
Figure 4. Synthetic data construction pipeline for long-horizon avatar adaptation. Public dialogues are filtered and converted into structured video prompts, from which the pretrained LTX model synthesizes controllable avatar videos. Quality-aware filtering removes semantically mismatched, visually degraded, static, and camera-dominated samples, yielding a curated synthetic corpus for long-horizon avatar adaptation.

LTX Video Synthesis. We feed the synthesized prompts to the pretrained LTX base model and generate avatar videos with standard multi-step sampling. Unlike heterogeneous web videos, these samples are produced under controllable visual and motion conditions and remain close to the distribution of the foundation model being adapted. The resulting corpus therefore provides a targeted source of training data for long-horizon audio-driven avatar generation.

Quality-aware Filtering. Although synthetic generation improves controllability, individual samples may still exhibit semantic mismatch, visual artifacts, static content, or camera-dominated motion. We therefore filter the generated videos along three complementary dimensions. We assess semantic consistency using multimodal similarity and reward models, including ImageBind [56], CLAP [57], and Unified Reward Model [58], and use Gemini [59] to remove visually degraded videos. To assess motion quality, we compare adjacent frames to identify clips with negligible temporal change or motion dominated by simple global transformations. In particular, we reject nearly static videos and videos whose frame-to-frame variation is primarily produced by global afine camera motion, such as uniform vertical camera movement, translation, or zooming, rather than local facial and body dynamics. For dialogue-driven samples, we retain only those videos with suficient local non-rigid motion, including visible facial articulation or body movement. This procedure yields a curated synthetic corpus with semantic consistency, visual fidelity, and motion diversity for long-horizon adaptation.

## 5 Experiments

## 5.1 Experimental Setup

Training Data. We train the long-horizon adaptation branch using the fully synthetic pipeline described in Section 4. Each sample contains an avatar video, paired audio, and a text prompt. Videos are partitioned into context-target pairs: the model conditions on a context window and predicts a video-only target chunk under the full audio condition. Unless otherwise stated, we use four latent frames as context and supervise the subsequent four latent frames. To improve robustness to temporal misalignment, the context start is sampled either from the first latent frame or from a randomly selected later position.

Implementation Details. Avatar-Forever is built on the 22B LTX-2.3 [6]. The eficiency branch distills the base model into a four-step generator using DMD [43], while the robustness branch independently trains video-side LoRA [60] adapters with rank and alpha both set to 128. For RRT, we perturb the historical context, roll out � = 4 autoregressive chunks, and apply the flow-matching loss to the following target chunk. Rollout uses the default 30-step denoising schedule without CFG. History degradation is applied with probability 0.5, including additive noise, blur, saturation, and latent masking. We always condition on the first frame through a zero-initialized gated module applied only to target denoising tokens. Both branches are optimized with AdamW at a learning rate of $1 \times 1 0 ^ { - 5 }$ and a global batch size of 256; DMD distillation and RRT training run for 5,000 and 3,000 steps, respectively.

Table 1. Quantitative comparison on the short- and long-video evaluation splits. Each entry reports the 5-second / 30-second results. Latency is measured in seconds. ‘w/ FC’ means Avatar-Forever with ForeverCache. The best and second-best results are marked in bold and underline, respectively.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="3">LLM Judge</td><td rowspan="2"></td><td colspan="6">Automatic Metrics</td><td rowspan="2">Latency (s)↓</td></tr><tr><td>A-V↑ Visual↑</td><td>Motion↑</td><td>Overall↑</td><td>IQA↑</td><td>ASE↑</td><td>Sync-C↑</td><td>Sync-D↓</td><td>FID↓</td><td>FVD↓</td></tr><tr><td rowspan="6">EMTD</td><td>OmniAvatar</td><td>3.77/2.92</td><td>3.81/1.83</td><td>3.06/2.00</td><td>3.57/2.26</td><td>4.15/2.98</td><td>2.84/2.42</td><td>5.57/6.83</td><td>9.47/8.59</td><td>61.62/115.84</td><td>979.07/1834.79</td><td>850.00/&gt; 1h</td></tr><tr><td>InfiniteTalk</td><td>3.58/4.17</td><td>3.71/4.00</td><td>2.81/3.67</td><td>3.39/3.96</td><td>4.40/4.79</td><td>3.11/3.64</td><td>6.73/6.81</td><td>8.00/7.95</td><td>39.25/37.63</td><td>786.25/1125.80</td><td>51.50/309.20</td></tr><tr><td>LiveAvatar</td><td>3.74/4.25</td><td>4.06/4.08</td><td>3.06/4.00</td><td>3.65/4.12</td><td>4.38/4.59</td><td>3.07/3.57</td><td>6.67/6.62</td><td>8.45/8.08</td><td>39.23/61.17</td><td>795.87/1373.52</td><td>53.01/320.95</td></tr><tr><td>SoulX</td><td>3.87/4.08</td><td>3.97/4.08</td><td>3.13/3.75</td><td>3.68/3.98</td><td>4.44/4.55</td><td>3.14/3.45</td><td>6.83/6.83</td><td>7.97/7.96</td><td>38.91/33.46</td><td>785.20/868.38</td><td>25.43/125.26</td></tr><tr><td>Ours w/ FC</td><td>3.81/4.33</td><td>4.00/4.25</td><td>3.13/4.00</td><td>3.67/4.23</td><td>4.77/4.84</td><td>3.28/3.70</td><td>7.11/6.68</td><td>8.39/8.01</td><td>48.76/34.52</td><td>759.78/905.97</td><td>4.24/26.71</td></tr><tr><td>Ours</td><td>3.90/4.42</td><td>4.16/4.50</td><td>3.32/4.08</td><td>3.82/4.32</td><td>4.80/4.88</td><td>3.33/3.73</td><td>7.56/6.85</td><td>7.95/7.94</td><td>38.37/33.33</td><td>775.11/858.06</td><td>5.24/38.85</td></tr><tr><td rowspan="6">HDTF</td><td>OmniAvatar</td><td>3.73/3.80</td><td>3.75/3.70</td><td>2.88/3.00</td><td>3.48/3.53</td><td>4.01/4.03</td><td>2.70/3.06</td><td>6.14/7.30</td><td>9.14/7.61</td><td>21.31/24.01</td><td>390.61/956.60</td><td>850.00/&gt; 1h</td></tr><tr><td>InfiniteTalk</td><td>3.34/4.00</td><td>3.58/3.80</td><td>2.47/3.00</td><td>3.16/3.63</td><td>4.03/4.15</td><td>2.75/3.16</td><td>8.14/7.43</td><td>7.06/7.60</td><td>26.71/19.48</td><td>390.69/793.74</td><td>51.50/309.20</td></tr><tr><td>LiveAvatar</td><td>3.90/3.80</td><td>3.95/3.80</td><td>3.08/3.00</td><td>3.67/3.56</td><td>4.19/4.07</td><td>2.74/3.23</td><td>7.39/6.23</td><td>8.39/8.45</td><td>23.42/95.35</td><td>426.26/1675.81</td><td>53.01/320.95</td></tr><tr><td>SoulX</td><td>3.82/4.00</td><td>3.80/3.70</td><td>2.95/3.20</td><td>3.55/3.66</td><td>4.05/4.17</td><td>2.75/3.20</td><td>8.48/7.50</td><td>7.10/7.54</td><td>21.14/19.41</td><td>401.39/1066.36</td><td>25.43/125.26</td></tr><tr><td>Ours w/ FC</td><td>3.73/4.10 3.88/3.90</td><td></td><td>3.15/3.20</td><td>3.61/3.80</td><td>4.42/4.42</td><td>3.38/3.38</td><td>8.56/7.56</td><td>7.06/7.56</td><td>20.19/17.37</td><td>393.92/686.96</td><td>4.24/26.71</td></tr><tr><td>Ours</td><td>3.90/4.00</td><td>4.03/4.10 3.25/3.40</td><td></td><td>3.75/3.82</td><td>4.30/4.38</td><td>2.89/3.33</td><td>8.68/7.56</td><td>6.94/7.50</td><td>20.91/17.19</td><td>378.17/593.92</td><td>5.24/38.85</td></tr><tr><td rowspan="6">TalkVid</td><td>OmniAvatar</td><td>3.66/4.00</td><td>3.71/3.80</td><td>2.84/3.50</td><td></td><td>3.68/4.12</td><td>2.41/3.13</td><td>4.45/6.49</td><td>9.77/8.94</td><td>48.15/64.70</td><td>618.67/1064.48</td><td>850.00/&gt; 1h</td></tr><tr><td>InfiniteTalk</td><td>3.25/4.10</td><td>3.13/3.90</td><td>2.60/3.40</td><td>3.43/3.78 3.01/3.82</td><td>3.51/4.30</td><td>2.31/3.26</td><td>5.48/6.35</td><td>8.87/8.90</td><td>53.10/52.09</td><td>669.80/1074.68</td><td>51.50/309.20</td></tr><tr><td>LiveAvatar</td><td>3.65/4.00</td><td>3.58/3.90</td><td>2.95/3.40</td><td>3.41/3.79</td><td>3.79/4.35</td><td>2.48/3.35</td><td>4.91/6.07</td><td>9.11/8.64</td><td>47.47/70.21</td><td>668.96/1308.11</td><td>53.01/320.95</td></tr><tr><td>SoulX</td><td>3.63/4.20</td><td>3.65/4.20</td><td>2.90/3.50</td><td>3.42/3.99</td><td>3.70/4.17</td><td>2.45/3.20</td><td>5.61/6.60</td><td>8.89/8.24</td><td>47.58/51.41</td><td>720.48/1066.36</td><td>25.43/125.26</td></tr><tr><td>Ours w/ FC</td><td>3.78/4.30</td><td>3.83/4.40</td><td>3.23/3.80</td><td>3.63/4.19</td><td>3.90/4.47</td><td>2.59/3.35</td><td>5.68/6.51</td><td>8.81/8.29</td><td>49.56/53.06</td><td>543.87/998.75</td><td>4.24/26.71</td></tr><tr><td>Ours</td><td>3.85/4.30</td><td>3.85/4.40</td><td>3.35/3.90</td><td>3.70/4.22</td><td>3.96/4.51</td><td>2.62/3.38</td><td>6.01/6.62</td><td>8.61/8.23</td><td>47.43/49.82</td><td>554.00/994.42</td><td>5.24/38.85</td></tr></table>

Evaluation Protocol. We evaluate short-horizon quality and long-horizon stability on TalkVid [61], EMTD [62], and HDTF [63]. We construct a 5-second split for short-horizon quality and a 30-second split for long-horizon stability, each containing 40 samples with paired audio, reference identity, and text conditions. Unless otherwise specified, ablations are conducted on EMTD. We compare with representative audio-driven and long-horizon avatar methods, including OmniAvatar [52], LiveAvatar [27], SoulX-FlashTalk [24], and InfiniteTalk [26]. For each baseline, we follow the oficial inference configuration whenever available and use identical audio and reference inputs for comparison.

Evaluation Metrics. We evaluate visual quality, motion realism, audio–visual synchronization, and videodistribution similarity. For perceptual evaluation, we use Gemini-Flash-3.5 [59] as a strict multimodal judge, which scores audio–visual consistency, visual quality, and motion naturalness on a 1–5 scale. The final score is the weighted average of them with weights 0.35, 0.35, and 0.30, respectively. The complete prompt is provided in Appendix A. We additionally report Fréchet Inception Distance (FID), Fréchet Video Distance (FVD), Q-Align image quality (IQA), Q-Align aesthetic score (ASE), and Sync-C/Sync-D for lip synchronization, where higher Sync-C and lower Sync-D are better. We further conduct a user study on the EMTD long-horizon split with 20 participants. Participants rate anonymized and randomly ordered videos from Avatar-Forever and competing methods using the same three perceptual criteria as the LLM judge. We report the mean participan score, with randomization and anonymization used to reduce ordering and model-identity bias.

## 5.2 Main Results

Short-horizon Quality. The 5-second results in Table 1 show that Avatar-Forever preserves the generation quality of the video foundation model under few-step inference. Compared to the strongest competing baseline, Avatar-Forever improves the LLM Overall score by 4.6% on average, together with average gains of 5.1% in IQA, 5.6% in ASE, and 6.7% in Sync-C, while consistently reducing Sync-D. The Motion score improves by up to 13.6%, extending the advantage beyond frame-level fidelity and synchronization to more natural speech-driven dynamics. Figure 5 provides qualitative comparison for these methods. Even in the short-horizon regime, the baselines already exhibit visual degradation, stereotyped facial gestures, limited motion diversity, and over-articulated mouth motion. In contrast, our results (the last two rows) preserve identity and visual fidelity while producing natural, diverse head movements and expressions with accurate lip synchronization. Beyond foreground animation, Avatar-Forever also retains coherent scene dynamics: for example, the large globe continues to rotate naturally while preserving its geographic structure and map textures, rather than becoming static or visually distorted. Please view the videos in the project page for better comparison.

Table 2. Double-blind user study results on the EMTD long-video evaluation split. Each score is computed from user ratings on a 1–5 scale and normalized to a 0–100 scale. The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Model</td><td colspan="4">User Study</td></tr><tr><td>Audio-Visual↑</td><td>Visual↑</td><td>Motion↑</td><td>Overall↑</td></tr><tr><td>OmniAvatar</td><td>57.5</td><td>32.5</td><td>29.0</td><td>40.20</td></tr><tr><td>InfiniteTalk</td><td>68.0</td><td>51.5</td><td>45.0</td><td>55.33</td></tr><tr><td>LiveAvatar</td><td>70.0</td><td>54.5</td><td>47.0</td><td>57.68</td></tr><tr><td>SoulX-FlashTalk</td><td>71.0</td><td>56.5</td><td>52.0</td><td>60.23</td></tr><tr><td>Ours w/ ForeverCache</td><td>73.0</td><td>76.5</td><td>75.0</td><td>74.83</td></tr><tr><td>Ours</td><td>74.0</td><td>78.0</td><td>76.0</td><td>76.00</td></tr></table>

Long-horizon Stability. The 30-second results in Table 1 further demonstrate Avatar-Forever’s robustness under autoregressive rollout. It achieves the best LLM Overall score on all three datasets, outperforming the strongest prior baseline by 5.0% on average. It also improves LLM Visual quality by 7.6% on average and Motion quality by up to 11.4%, while obtaining the best FID and FVD across all datasets. In particular, FID is reduced by 5.0% on average and FVD by 25.2% on HDTF, supporting improved distributional realism under accumulated context errors. Figure 6 provides complementary qualitative evidence for these gains. As highlighted by the red boxes, competing methods exhibit progressive visual degradation and appearance drift, repetitive low-diversity gesture cycles, motion-induced hand blur and structural distortion, or over-articulated mouth motion accompanied by stereotyped upper-body gestures. In contrast, our Avatar-Forever preserves identity, facial and hand details, and diverse natural motion throughout the extended rollout. Its ForeverCache variant retains comparable long-horizon stability despite the substantial inference acceleration.

Eficiency-Quality Balance. ForeverCache improves streaming eficiency by reusing fixed historical context features across denoising steps. On short videos, it reduces latency by 19.1% and increases throughput by 23.6% compared to standard Avatar-Forever, while remaining approximately 6.0× faster than the fastest prior baseline. On 30-second generation, it reduces runtime from 38.85s to 26.71s, corresponding to a 31.2% latency reduction and a 45.5% throughput increase; it remains approximately 4.7× faster than the fastest prior method. Despite this acceleration, ForeverCache preserves most quality gains of the standard model: it remains competitive across perceptual, synchronization, and distributional metrics on short videos, and achieves the second-best LLM Overall score on all long-video splits while outperforming the strongest prior baseline by 3.8% on average. Figures 6 show that cache-enabled inference retains the key visual behavior of the full model without visible collapse or systematic temporal degradation.

Extended-Duration Generation. Figure 7 presents an audio-driven avatar video generated continuously for more than 11 minutes. Avatar-Forever maintains stable identity, facial structure, appearance, and scene content without progressive drift or visible quality collapse. The avatar exhibits natural speech-driven dynamics, including realistic facial expressions, coherent gaze, subtle head movements, smooth posture changes, and non-repetitive gestures. These motions remain temporally coherent and appropriately follow the rhythm and emphasis of the speech, avoiding the frozen expressions, mechanical motion, and repetitive behavior commonly observed in long-form generation. Moreover, lip movements and facial articulation remain consistently aligned with the driving audio throughout the rollout. This result demonstrates that Avatar-Forever preserves visual fidelity, motion naturalness, and audio–visual consistency well beyond the 30-second evaluation setting, supporting its capability for stable generation without a predefined duration.

Videos of the above examples and more visual examples can be found in Appendix B and project page.

Human Perceptual Validation. Table 2 reports the double-blind user study results on the EMTD long-video split. Twenty participants rate anonymized videos on the same four perceptual dimensions as the LLM judge using integer scores from 1 to 5; scores are then averaged and normalized to a 0–100 scale. The human evaluation is consistent with the automatic results: Avatar-Forever ranks first on all dimensions, while the ForeverCache variant consistently ranks second with only a small gap from the full model.

The advantage in human validation is particularly pronounced for Visual and Motion quality, highlighting aspects of temporal naturalness that are not fully captured by automatic evaluation. While several baselines preserve identity and produce locally plausible frames, their long videos often exhibit limited upper-body motion, static backgrounds, repetitive gestures, rigid posture changes, or exaggerated mouth shapes. In contrast, Avatar-Forever produces more expressive facial dynamics, smoother gesture transitions, and motion that better follows the rhythm of speech. The resulting improvement in temporal naturalness also strengthens perceived visual realism, explaining the highest Overall score assigned by human raters. ForeverCache preserves most of these perceptual gains while providing substantially faster inference.

![](images/b62b658fdca6454e13ea290e28f3160cf7ae47340405b9f972ad1a1ff0ecfd6d.jpg)  
Figure 5. Visual comparison on 5-second generation. Avatar-Forever produces more natural speech-driven facial motion, hand gestures, and background dynamics than prior methods, which often restrict motion to the face and mouth. The ForeverCache variant preserves the visual performance of the standard model while reducing inference cost.

## 5.3 Analysis of Decoupled Training

Table 3 and Figure 8 test our central hypothesis that few-step eficiency and long-horizon robustness are distinct capabilities. The full Decoupled DMD + RRT model achieves the best result on every metric, spanning perceptual quality, motion, synchronization, and distributional realism. Relative to DMD only, adding RRT improves the LLM Overall score by 3.6%, increases Sync-C by 4.2%, and reduces FID and FVD by 11.5% and 16.5%, respectively. Although DMD provides a strong few-step generator, it is not exposed to the accumulated self-conditioning errors encountered during autoregressive inference. Correspondingly, Figure 8 shows weaker long-horizon appearance stability.

The FM only variant also shows limited performance. Optimizing long-horizon flow matching without the distilled generator substantially degrades visual fidelity and synchronization, producing blurred facial details and poor perceptual quality. Compared with this variant, the full model improves LLM Overall by 77.9%, increases Sync-C by 229.6%, and reduces FID by 54.3%. These results show that robustness cannot be obtained by optimizing long-horizon recovery alone. DMD preserves the few-step generation prior, whereas RRT adapts it to the rollout-time error distribution; independently learning and composing these capabilities yields both high local fidelity and stable long-horizon generation.

![](images/136808a0129778c55e9a5208261833337bb70768abced58c7fa9e6aa439d8ddc.jpg)  
Figure 6. Visual comparison on 30-second generation. Avatar-Forever maintains identity, facial detail, hand structure, and motion diversity over extended autoregressive rollout. In contrast, competing methods exhibit degraded hand details, repetitive gestures, or instability over time. ForeverCache retains comparable long-horizon visual stability with substantially faster inference.

## 5.4 Efect of Recovery-oriented Rollout Training

Figure 9 isolates the efect of the rollout horizon � in RRT. The key observation is that corrupted-history training alone is insuficient: when � = 0, supervision is applied immediately after context perturbation, reducing the task to local reconstruction. The resulting videos remain susceptible to accumulated autoregressive errors, exhibiting spot-like artifacts and global color drift. Increasing the horizon to � = 1 partially exposes the model to self-generated errors, but visible appearance fluctuations remain.

Larger rollout horizons match more closely the error accumulation process encountered at inference time. Both � = 2 and � = 4 improve appearance, background, and color consistency, with � = 4 producing the most stable long-video results. This progression supports the design of RRT: its benefit comes from supervising recovery after model-induced errors have propagated through rollout, rather than from merely injecting synthetic corruption into historical context.

## 6 Conclusion

We presented Avatar-Forever, a framework for real-time long-duration audio-driven avatar generation. Our key insight was that the quality of few-step generation and the robustness to autoregressive drift are distinct adaptation objectives. We therefore learned them independently: DMD produced an eficient few-step generator, RRT trained a lightweight adapter to recover from accumulated rollout errors, and ForeverCache reduced redundant historical computation during streaming inference. Built on a 22B video foundation model and a fully synthetic data pipeline, Avatar-Forever improved perceptual quality, audio–visual synchronization, long-horizon

## Long-Horizon Generation

![](images/fba3ce57e65a947626a45692f6b94b697a3eca2993e81f29f04721915e445290.jpg)  
Driving audio  
00:00—11:44

Figure 7. Visual results of extended-duration generation. We uniformly sample frames from a continuous video generated for more than 11 minutes. Avatar-Forever maintains stable identity, facial structure, and scene content while producing natural expressions, coherent head and body motion, and accurate audio–visual synchronization throughout the extended autoregressive rollout. The bottom row shows the corresponding driving-audio waveform.  
Table 3. Ablation study on decoupled training. We compare the proposed decoupled training strategy with DMD-only training and FM-only long-horizon adaptation. Best results are highlighted in bold, and second-best results are underlined.
<table><tr><td rowspan="2">Setting</td><td colspan="4">LLM Judge</td><td colspan="6">Automatic Metrics</td></tr><tr><td>Audio-Visual↑</td><td>Visual↑</td><td>Motion↑</td><td>Overall↑</td><td>IQA↑</td><td>ASE↑</td><td>Sync-C↑</td><td>Sync-D↓</td><td>FID↓</td><td>FVD↓</td></tr><tr><td>Decoupled DMD + RRT</td><td>4.250</td><td>4.267</td><td>3.750</td><td>4.105</td><td>4.851</td><td>3.684</td><td>6.694</td><td>8.032</td><td>34.978</td><td>901.811</td></tr><tr><td>DMD only</td><td>4.083</td><td>4.167</td><td>3.583</td><td>3.962</td><td>4.793</td><td>3.671</td><td>6.427</td><td>8.085</td><td>39.515</td><td>1080.230</td></tr><tr><td>FM only</td><td>2.750</td><td>1.917</td><td>2.250</td><td>2.308</td><td>3.409</td><td>2.590</td><td>2.031</td><td>10.760</td><td>76.577</td><td>1071.200</td></tr></table>

![](images/651feb0ff82e7ca306927cd4034edbb6e648e23fdb569940e6bba1cb09741d20.jpg)  
Figure 8. Efect of decoupled training. DMD only preserves short-horizon appearance but is susceptible to long-horizon drift, whereas FM-only adaptation degrades visual fidelity. Combining DMD with RRT preserves detailed appearance while improving stability under autoregressive generation.

stability, and inference eficiency. These results supported a simple principle for scalable video adaptation: learn eficiency and long-horizon robustness separately, then compose them at deployment.

Limitations and Future Work. Avatar-Forever achieves 27.2 FPS at 768 × 512 resolution on a single H100 GPU, but it is not yet optimized for consumer-grade hardware. In addition, although our training and optimization are tailored to audio-driven avatars, we observe a good generalization to broader video-generation

![](images/992d36f1884a021a529664f74aed9d3ce360cdf65400128996f8fb0b7f8c2a41.jpg)  
Figure 9. Efect of the RRT rollout horizon. Immediate supervision after context degradation (� = 0) leaves visible artifacts and appearance drift. Increasing the autoregressive rollout horizon exposes the model to accumulated self-generated errors before supervision, improving color, background, and identity consistency; � = 4 yields the most stable result.

scenarios, as illustrated in our project page. Future work will explore domain-specific data construction and training strategies to further unlock Avatar-Forever’s potential for general long-horizon video generation.

## Ethics Statement

This work aims to advance long-duration audio-driven avatar generation for constructive applications such as virtual communication, education, accessibility, and digital content creation. The evaluation data used in this paper are drawn from publicly available and openly accessible research datasets. Our synthetic training data are generated from the underlying video foundation model and filtered for quality, motion, and semantic consistency, rather than collected from private or restricted personal sources. We recognize that high-fidelity avatar synthesis is a dual-use technology. While it can support beneficial interactive media applications, it may also be misused for impersonation, deceptive content generation, or misinformation. We do not endorse any use of this technology to create unauthorized representations of real individuals, manipulate public opinion, or violate privacy, consent, or intellectual property rights.

To reduce potential misuse, we encourage deployment together with responsible safeguards, including identity-consent verification, synthetic-content disclosure, invisible watermarking, provenance tracking, and robust forgery detection. We also recommend that generated videos be clearly labeled as synthetic when used in public-facing scenarios. Our goal is to contribute to safer and more transparent digital human generation, and we will continue to align future development with responsible AI principles.

## References

[1] Wenxuan Zhang, Xiaodong Cun, Xuan Wang, Yong Zhang, Xi Shen, Yu Guo, Ying Shan, and Fei Wang. Sadtalker: Learning realistic 3d motion coeficients for stylized audio-driven single image talking face animation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8652–8661, 2023.

[2] Chunyu Li, Chao Zhang, Weikai Xu, Jinghui Xie, Weiguo Feng, Bingyue Peng, and Weiwei Xing. Latentsync: Audio conditioned latent difusion models for lip sync. arXiv e-prints, pages arXiv–2412, 2024.

[3] Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. Emo: Emote portrait alive: Generating expressive portrait videos with audio2video difusion model under weak conditions. arXiv preprint arXiv:2402.17485, 2024.

[4] Team Seedance. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026.

[5] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[6] Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, et al. Ltx-2: Eficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026.

[7] Tao Yang, Ruibin Li, Yangming Shi, Yuqi Zhang, Qide Dong, Haoran Cheng, Weiguo Feng, Shilei Wen, Bingyue Peng, and Lei Zhang. Many-for-many: Unify the training of multiple video and image generation and manipulation tasks. arXiv preprint arXiv:2506.01758, 2025.

[8] SII-GAIR, Sand. ai, Ethan Chern, Hansi Teng, Hanwen Sun, Hao Wang, Hong Pan, Hongyu Jia, Jiadi Su, Jin Li, Junjie Yu, Lijie Liu, Lingzhi Li, Lyumanshan Ye, Min Hu, Qiangang Wang, Quanwei Qi, Stefi Chern, Tao Bu, Taoran Wang, Teren Xu, Tianning Zhang, Tiantian Mi, Weixian Xu, Wenqiang Zhang, Wentai Zhang, Xianping Yi, Xiaojie Cai, Xiaoyang Kang, Yan Ma, Yixiu Liu, Yunbo Zhang, Yunpeng Huang, Yutong Lin, Zewei Tao, Zhaoliang Liu, Zheng Zhang, Zhiyao Cen, Zhixuan Yu, Zhongshu Wang, Zhulin Hu, Zijin Zhou, Zinan Guo, Yue Cao, and Pengfei Liu. Speed by simplicity: A single-stream architecture for fast audio-video generative foundation model. arXiv preprint arXiv:2603.21986, 2026.

[9] Ying Guo, Xi Liu, Cheng Zhen, Pengfei Yan, and Xiaoming Wei. Arig: Autoregressive interactive head generation for real-time conversations. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 12956–12965, 2025.

[10] Tianqi Li, Ruobing Zheng, Minghui Yang, Jingdong Chen, and Ming Yang. Ditto: Motion-space difusion for controllable realtime talking head synthesis. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 9704–9713, 2025.

[11] Chetwin Low and Weimin Wang. Talkingmachines: Real-time audio-driven facetime-style video via autoregressive difusion models. arXiv preprint arXiv:2506.03099, 2025.

[12] Liyuan Cui, Wentao Hu, Wenyuan Zhang, Zesong Yang, Fan Shi, and Xiaoqiang Liu. Avatarforcing: One-step streaming talking avatars via local-future sliding-window denoising. arXiv preprint arXiv:2603.14331, 2026.

[13] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video difusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22963–22974, 2025.

[14] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video difusion. Advances in Neural Information Processing Systems, 38:167283–167308, 2026.

[15] Zhiyao Sun, Ziqiao Peng, Yifeng Ma, Yi Chen, Zhengguang Zhou, Zixiang Zhou, Guozhen Zhang, Youliang Zhang, Yuan Zhou, Qinglin Lu, and Yong-Jin Liu. Streamavatar: Streaming difusion models for real-time interactive human avatars. arXiv preprint arXiv:2512.22065, 2026. Accepted by CVPR 2026.

[16] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.

[17] Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, et al. Reward forcing: Eficient streaming video generation with rewarded distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 34385–34397, 2026.

[18] Wuyang Li, Wentao Pan, Po-Chien Luan, Yang Gao, and Alexandre Alahi. Stable video infinity: Infinite-length video generation with error recycling. arXiv preprint arXiv:2510.09212, 2025.

[19] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive difusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026.

[20] Hidir Yesiltepe, Tuna Meral, Adil Kaan Akan, Kaan Oktay, and Pinar Yanardag. Infinity-rope: Action-controllable infinite video generation emerges from autoregressive self-rollout. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 40256–40265, 2026.

[21] Shenghai Yuan, Yuanyang Yin, Zongjian Li, Xinwei Huang, Xiao Yang, and Li Yuan. Helios: Real real-time long video generation model. arXiv preprint arXiv:2603.04379, 2026.

[22] Yukang Chen, Luozhou Wang, Wei Huang, Shuai Yang, Bohan Zhang, Yicheng Xiao, Ruihang Chu, Weian Mao, Qixin Hu, Shaoteng Liu, et al. Longlive-2.0: An nvfp4 parallel infrastructure for long video generation. arXiv preprint arXiv:2605.18739, 2026.

[23] Ailing Zeng, Casper Yang, Chauncey Ge, Eddie Zhang, Garvey Xu, Gavin Lin, Gilbert Gu, Jeremy Pi, Leo Li, Mingyi Shi, Shawn Wang, Sheng Bi, Steven Tang, Thorn Hang, Tobey Guo, Vincent Li, Xin Tong, Yikang Li, Yuchen Sun, Yue Zhao, Yuhan Lu, Yuwei Li, Zane Zhang, Zeshi Yang, and Zi Ye. Lpm 1.0: Video-based character performance model. arXiv preprint arXiv:2604.07823, 2026.

[24] Le Shen, Qian Qiao, Tan Yu, Ke Zhou, Tianhang Yu, Yu Zhan, Zhenjie Wang, Dingcheng Zhen, Ming Tao, Shunshun Yin, and Siyuan Liu. Soulx-flashtalk: Real-time infinite streaming of audio-driven avatars via self-correcting bidirectional distillation. arXiv preprint arXiv:2512.23379, 2026.

[25] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Self-forcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025.

[26] Shaoshu Yang, Zhe Kong, Feng Gao, Meng Cheng, Xiangyu Liu, Yong Zhang, Zhuoliang Kang, Wenhan Luo, Xunliang Cai, Ran He, et al. Infinitetalk: Audio-driven video generation for sparse-frame video dubbing. arXiv preprint arXiv:2508.14033, 2025.

[27] Yubo Huang, Hailong Guo, Fangtai Wu, Weiqiang Wang, Shifeng Zhang, Shijie Huang, Qijun Gan, Lin Liu, Sirui Zhao, Enhong Chen, Jiaming Liu, and Steven Hoi. Live avatar: Streaming real-time audio-driven avatar generation with infinite length. arXiv preprint arXiv:2512.04677, 2025.

[28] Trong Thang Pham, Tuong Do, Nhat Le, Ngan Le, Hung Nguyen, Erman Tjiputra, Quang Tran, and Anh Nguyen. Style transfer for 2d talking head generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7500–7509, 2024.

[29] Yu Deng, Jiaolong Yang, Dong Chen, Fang Wen, and Xin Tong. Disentangled and controllable face image generation via 3d imitative-contrastive learning. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 5154–5163, 2020.

[30] Daniel Cudeiro, Timo Bolkart, Cassidy Laidlaw, Anurag Ranjan, and Michael J Black. Capture, learning, and synthesis of 3d speaking styles. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10101–10111, 2019.

[31] K R Prajwal, Rudrabha Mukhopadhyay, Vinay Namboodiri, and C V Jawahar. A lip sync expert is all you need for speech to lip generation in the wild. In Proceedings of the 28th ACM International Conference on Multimedia, pages 484–492, 2020.

[32] Yukang Cao, Yan-Pei Cao, Kai Han, Ying Shan, and Kwan-Yee K Wong. Dreamavatar: Text-and-shape guided 3d human avatar generation via difusion models. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 958–968, 2024.

[33] Yuelang Xu, Hongwen Zhang, Lizhen Wang, Xiaochen Zhao, Han Huang, Guojun Qi, and Yebin Liu. Latentavatar: Learning latent expression code for expressive neural head avatar. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–10, 2023.

[34] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[35] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

[36] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent difusion models. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

[37] Shuai Shen, Wenliang Zhao, Zibin Meng, Wanhua Li, Zheng Zhu, Jie Zhou, and Jiwen Lu. Diftalk: Crafting difusion models for generalized audio-driven portraits animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1982–1991, 2023.

[38] Mingwang Xu, Hui Li, Qingkun Su, Hanlin Shang, Liwei Zhang, Ce Liu, Jingdong Wang, Yao Yao, and Siyu Zhu. Hallo: Hierarchical audio-driven visual synthesis for portrait image animation. arXiv preprint arXiv:2406.08801, 2024.

[39] Huawei Wei, Zejun Yang, and Zhisheng Wang. Aniportrait: Audio-driven synthesis of photorealistic portrait animation. arXiv preprint arXiv:2403.17694, 2024.

[40] Zhiyuan Chen, Jiajiong Cao, Zhiquan Chen, Yuming Li, and Chenguang Ma. Echomimic: Lifelike audio-driven portrait animations through editable landmark conditioning. arXiv preprint arXiv:2407.08136, 2024.

[41] Kling Team. Kling-omni technical report. arXiv preprint arXiv:2512.16776, 2025.

[42] OpenAI. Video generation models as world simulators. https://openai.com/index/ video-generation-models-as-world-simulators/, 2024. Technical report.

[43] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman, and Taesung Park. One-step difusion with distribution matching distillation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6613–6623, 2024.

[44] Tianwei Yin, Micha"el Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman, and Taesung Park. Improved distribution matching distillation for fast image synthesis. arXiv preprint arXiv:2405.14867, 2024.

[45] Axel Sauer, Dominik Lorenz, Andreas Blattmann, Puneet Dokania, Stefano Ermon, Andreas Geiger, Patrick Esser, and Robin Rombach. Adversarial difusion distillation. In European Conference on Computer Vision, pages 87–103, 2024.

[46] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.

[47] Fu-Yun Wang, Zhaoyang Huang, Alexander William Bergman, Dazhong Shen, Peng Gao, Michael Lingelbach, Keqiang Sun, Weikang Bian, Guanglu Song, Yu Liu, Xiaogang Wang, and Hongsheng Li. Phased consistency models. In Advances in Neural Information Processing Systems, 2024.

[48] Tianhe Wu, Ruibin Li, Lei Zhang, and Kede Ma. Diversity-preserved distribution matching distillation for fast visual synthesis. arXiv preprint arXiv:2602.03139, 2026.

[49] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video difusion. arXiv preprint arXiv:2506.08009, 2025.

[50] Ruibin Li, Tao Yang, Fangzhou Ai, Tianhe Wu, Shilei Wen, Bingyue Peng, and Lei Zhang. Long-horizon streaming video generation via hybrid attention with decoupled distillation. arXiv preprint arXiv:2604.10103, 2026.

[51] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video difusion in real time. arXiv preprint arXiv:2509.25161, 2025.

[52] Qijun Gan, Ruizi Yang, Jianke Zhu, Shaofei Xue, and Steven Hoi. Omniavatar: Eficient audio-driven avatar video generation with adaptive body animation. arXiv preprint arXiv:2506.18866, 2025.

[53] Yi Chen, Sen Liang, Zixiang Zhou, Ziyao Huang, Yifeng Ma, Junshu Tang, Qin Lin, Yuan Zhou, and Qinglin Lu. Hunyuanvideo-avatar: High-fidelity audio-driven human animation for multiple characters. arXiv preprint arXiv:2505.20156, 2025.

[54] Jesse Dodge, Andreea Gane, Xiang Zhang, Antoine Bordes, Sumit Chopra, Alexander Miller, Arthur Szlam, and Jason Weston. Evaluating prerequisite qualities for learning end-to-end dialog systems, 2016.

[55] OpenAI. Chatgpt. https://chatgpt.com/, 2026. Accessed: 2026-07-17.

[56] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In CVPR, 2023.

[57] Yusong Wu\*, Ke Chen\*, Tianyu Zhang\*, Yuchen Hui\*, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Largescale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP, 2023.

[58] Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025.

[59] Google DeepMind. Gemini 3.5 flash. https://deepmind.google/models/gemini/flash/, 2026. Accessed: 2026-07-08.

[60] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022.

[61] Shunian Chen, Hejin Huang, Yexin Liu, Zihan Ye, Pengcheng Chen, Chenghao Zhu, Michael Guan, Rongsheng Wang, Junying Chen, Jianye Hou, et al. Talkvid: A large-scale diversified dataset for audio-driven talking head synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3492–3500, 2026.

[62] Rang Meng, Xingyu Zhang, Yuming Li, and Chenguang Ma. Echomimicv2: Towards striking, simplified, and semi-body human animation. arXiv preprint arXiv:2411.10061, 2024.

[63] Zhimeng Zhang, Lincheng Li, Yu Ding, and Changjie Fan. Flow-guided one-shot talking face generation with a high-resolution audio-visual dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3661–3670, 2021.

## Appendix

In this appendix, we provide the LLM-based perceptual evaluation prompt, and more qualitative results to complement the main paper. Specifically, the supplement includes:

A. The LLM-based perceptual evaluation prompt (referring to Sec. 5.1 in the main paper);

B. Additional visual comparisons for 5-second and 30-second audio-driven avatar generation, including results with and without ForeverCache (referring to Sec. 5.2 in the main paper).

For better viewing experience, we uploaded all the videos to our project page project page, where the videos can be played directly in the browser. Note that, due to the significant number of high-quality video files included in our demonstrations, initial page loading may require several minutes to complete. We appreciate your patience during this process, as the complete visual experience is essential to understand the capabilities and performance of our approach.

## A LLM-based Perceptual Evaluation Prompt

Conventional automatic metrics measure image quality, video realism, and audio–visual synchronization, but they do not fully capture whether an avatar behaves like a natural speaker throughout a continuous video. In particular, temporally accumulated artifacts, repetitive gestures, frozen expressions, and unnatural motion transitions can be dificult to characterize using individual automatic metrics. We therefore employ Gemini-Flash-3.5 [59] as a multimodal perceptual judge. For each generated sample, the evaluator receives only the corresponding audio and video to assess the result using a fixed evaluation rubric shared across all methods, datasets, and video durations.

The evaluation covers three complementary dimensions. Audio–Visual Consistency measures whether mouth shapes, facial dynamics, head motion, speaking rhythm, and pauses are synchronized with the input audio. Visual Quality evaluates facial fidelity, sharpness, identity consistency, lighting, texture realism, background stability, and temporal artifacts such as flickering, blur, and appearance drift. Motion Naturalness focuses on the temporal behavior of the avatar, including facial expressions, blinking, head and upper-body movement, breathing, gesture diversity, contextual appropriateness, and physical plausibility.

Each dimension is assigned an integer score from 1 to 5 according to explicit quality anchors, with a score of 5 reserved for videos that are nearly indistinguishable from high-quality real talking videos. The overall perceptual score is computed as:

$$
\begin{array} { r } { S _ { \mathrm { o v e r a l l } } = 0 . 3 5 S _ { \mathrm { A - V } } + 0 . 3 5 S _ { \mathrm { v i s u a l } } + 0 . 3 0 S _ { \mathrm { m o t i o n } } , } \end{array}\tag{A.1}
$$

and rounded to two decimal places. The evaluator additionally returns a brief justification for each score in a structured JSON format, enabling consistent parsing and inspection of the judgments.

The prompt is designed to be particularly sensitive to long-horizon generation failures. It explicitly penalizes accumulated blur, identity or appearance drift, visual artifacts, repetitive gesture cycles, and motion that gradually becomes frozen or mechanical. It also prevents large but inappropriate motion from being rewarded solely for its magnitude. To avoid evaluation based on irrelevant personal attributes, the judge is instructed not to consider the identity, gender, ethnicity, or appearance of the depicted person. The complete system prompt used for evaluation are provided in Figure A.1.

## B Visual Results and Comparisons

30s Long-Video Generation. Figures B.2–B.3 compare 30-second results on two long-horizon samples. In each figure, from top to bottom, the rows show OmniAvatar, LiveAvatar, InfiniteTalk, SoulX-FlashTalk, and our method with and without ForeverCache (Ours (w/ FC) and Ours); the columns show uniformly sampled frames from start to end, and the bottom strip visualizes the driving audio. Over the longer horizon, the baselines accumulate errors that are far more pronounced than in the short setting: background and appearance drift, severe visual degradation and spurious content (e.g., an extra person that ignores the scene), skin-color shift, and motion that is either nearly static or dominated by camera movement. Our results (the last two rows) remain stable across the full duration, keeping the background and identity consistent while generating more vivid and diverse head and body motion, such as the head leaning and rotating rather than repeating a fixed pose. At the same time, AvatarForever with ForeverCache closely matches the base AvatarForever, confirming that ForeverCache preserves long-horizon quality while substantially reducing redundant history computation.

![](images/98dea4f747a1c37c90004b777890e9a9098c4b222bff633c456f2223ad3c96c3.jpg)  
Figure A.1. System prompt for LLM-based perceptual evaluation. The same evaluation prompt is applied to all methods, datasets, and video durations. It defines the scoring criteria for audio–visual consistency, visual quality, and motion naturalness, as well as the weighted aggregation used to compute the overall score.

5s Short-Video Generation. Figures B.4–B.5 compare 5-second results on two EMTD samples. Even in the short-horizon regime, the baselines already exhibit visual degradation, repetitive and stereotyped facial gestures, limited motion diversity, and over-articulated mouth motion. In contrast, our results (the last two rows) preserve identity and visual fidelity while producing more natural and diverse head motion and expressions with accurate lip synchronization. AvatarForever with ForeverCache is visually on par with base AvatarForever, showing that ForeverCache accelerates streaming inference without sacrificing quality.

![](images/c64614d35b112e1d4172c39d1a222c30fab1b5f763168c3cf7547c8ed95762fc.jpg)  
Figure B.2. Visual comparison on 30-second generation (sample 1). From top to bottom, the rows show OmniAvatar, LiveAvatar, InfiniteTalk, SoulX-FlashTalk, Ours (w/ FC), and Ours; the columns show frames uniformly sampled over time, with the driving audio shown below. The baselines drift with a degraded background and a shifted face, and the body barely moves. Avatar-Forever (last two rows) keeps identity and background stable while the head and body lean and rotate for more vivid, diverse motion. ForeverCache retains comparable visual stability with substantially faster inference.

![](images/6db37f9d222b532f58d8f412729fbbb70e3c95a93ed33a1f7bc6543e2dca33f5.jpg)  
Figure B.3. Visual comparison on 30-second generation (sample 2). Same row and column layout as Figure B.2. The baselines show severe degradation, including a spurious extra person that ignores the scene, together with low motion diversity. Avatar-Forever (last two rows) remains stable over the full 30 seconds with larger, more diverse hand and arm motion. ForeverCache retains comparable visual stability with substantially faster inference.

![](images/3d2b35fdd5914a1cc5978c5916ff29e05735d283e0ef7b5007dd49ab69ca18d4.jpg)  
Figure B.4. Visual comparison on 5-second generation (sample 1). From top to bottom, the rows show OmniAvatar, LiveAvatar, InfiniteTalk, SoulX-FlashTalk, Ours (w/ FC), and Ours; the columns show frames uniformly sampled over time, with the driving audio shown below. The baselines show unexpected grids, repetitive and stereotyped facial gestures, and over-articulated mouth motion. Avatar-Forever (last two rows) preserves identity and facial detail with more natural, diverse expressions and accurate lip synchronization. ForeverCache retains comparable visual quality with substantially faster inference.

![](images/99873f48b16fbd4b7d5d07d97fbe32f7184386b596dfadb9e077de968eebc0f0.jpg)  
Figure B.5. Visual comparison on 5-second generation (sample 2). Same row and column layout as Figure B.4. The baselines exhibit background degradation, over-articulated expression and mouth motion. Avatar-Forever (last two rows) achieves richer motion diversity, such as the head leaning and turning, while keeping a stable appearance. ForeverCache retains comparable visual quality with substantially faster inference.