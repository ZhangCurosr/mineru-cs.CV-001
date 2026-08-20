# EfficientSync: Real-Time Lip Synchronization via Deformation-Based Reference Texture Mixing

Fa-Ting Hong<sup>∗</sup>, Runzhen Liu<sup>∗</sup>, Luchuan Song, Hongmin Cai, and Chuhua Xian<sup>B</sup>

Abstract—Audio-driven lip synchronization aims to manipulate the mouth region of a talking-face video so that it conforms to the driving audio, while preserving head pose, identity, and background. By its nature, the task entails local editing of the mouth rather than global resynthesis of the face. Nevertheless, prevailing approaches reconstruct the entire lower face via heavy GAN- or diffusion-based decoders. This design incurs substantial inference latency. More critically, it hallucinates intra-oral high-frequency details such as teeth and lip wrinkles instead of preserving the subject’s authentic textures. We contend that the principal bottleneck in identity preservation lies not in the scarcity of reference frames, but in the absence of a mechanism that faithfully transfers the genuine textures already present in them. Motivated by this insight, we present EfficientSync, a real-time deformation-based framework that explicitly retains reference textures rather than resynthesizing them. First, we reformulate multi-reference fusion as a channel-wise selection problem, in place of channel-coupled convolutions or texture-disrupting spatial self-attention. To this end, we propose the Dynamic Texture Mixer. It evaluates each spatially aligned reference in a global context and aggregates them via channel-wise weighted summation. This preserves textural integrity at a substantially reduced computational cost. Second, we introduce the Spatio-Temporal Shifted Adaptive Masking strategy to reconcile the long-standing conflict between mask coverage and background fidelity. It decomposes the source frame into lip-generation conditions and an independent background prior. This suppresses lower-face leakage and allows the synthesized mouth to blend seamlessly into the original background. Third, we devise STAR Sampling, a zero-overhead pre-processing procedure that retrieves the sharpest and topologically most diverse frames, since deformation quality is ultimately bounded by the f l E i i h HDTF d VFHQ d d h Effi i S i f h i quality and identity preservation while operating at 166 FPS on a single GPU. Video demos are available at https://alunaticat.github.io/EfficientSync/index.html.

Index Terms—Lip synchronization, multi-reference fusion, identity preservation

## 1 INTRODUCTION

IP synchronization (lip-sync) aims to manipulate the a given driving audio, while preserving the remaining attributes such as head pose, identity, and background. This task underpins a broad spectrum of real-world applications, including film dubbing and post-production, multilingual video translation, virtual avatars and digital humans, video conferencing, and accessibility tools for the hearing impaired. By its nature, lip-sync modifies only the mouth region of the source frames and leaves all other facial attributes intact, so it should in principle be both computationally efficient and faithful to the source identity. In practice, however, existing methods continue to struggle on both fronts, which severely constrains their deployment in latency-sensitive and quality-critical scenarios.

Because lip-sync operates on videos rather than a single image, one might expect the abundant reference frames of the same subject to already supply its mouth-region identity, leaving little room for the identity loss that plagues oneshot generation. Yet the mere availability of references does not guarantee that this identity information is faithfully exploited, and the failure manifests in two distinct stages. The first is the way the lower face is handled. Most methods [1], [2], [3] apply a single fixed mask that erases the entire lower half and compel a heavy generative or diffusion decoder to redraw it from scratch, even though large portions of this region, such as the chin, jawline, and surrounding skin, require no regeneration at all. Worse, the reference frames are rarely aligned with the target in head pose or mouth openness, so the decoder must resolve appearance and geometry simultaneously, a process in which fine details such as tooth edges and lip wrinkles are easily blurred. The second is the way references are fused. Reference features are repeatedly merged with frame-varying audio features through stacked convolutions and attention layers, where the time-varying audio signal tends to dominate the timeinvariant identity cues, so the authentic textures carried by the references are gradually smoothed away. When the target phoneme demands a mouth shape that no reference clearly exhibits, the network, lacking a faithful exemplar, falls back to fabricating the intra-oral appearance from audio alone, which is precisely where blurred teeth and implausible textures emerge. Taken together, these observations reveal that the difficulty lies not in the scarcity of references, but in two coupled deficiencies: the absence of a mechanism that genuinely retains and faithfully transfers the real textures already present in those references, and the absence of a principled way to edit only the mouth while leaving the surrounding regions truly untouched.

![](images/1e70ae7695f5b3f1d8297675dd948e55d5123618a97248f83838da83ec96d960.jpg)  
Fig. 1: Motivation of the Dynamic Texture Mixer. Rather than reconstructing the mouth from scratch, we treat each reference as a track on a mixing console and adjust its volume, that is, its aggregation score, according to the driving audio and the source identity.

This bottleneck recurs across the main families of lipsync methods. GAN-based pixel synthesis, represented by Wav2Lip [1] and its follow-ups [4], [5], [6], concatenates references with the masked source and trains a generator under a sync discriminator, but suffers from low resolution and blurred intra-oral regions. Diffusion-based generation, including DiffTalk [7], Diff2Lip [8], and LatentSync [3], synthesizes the mouth in a latent space and injects identity through reference embeddings, yet incurs the high latency of multi-step sampling and still recovers identity through a generative prior rather than from real reference pixels. Person-specific modeling, such as RAD-NeRF [9] and ER-NeRF [10], preserves identity by overfitting a dedicated model per speaker, at the cost of failing to generalize to unseen identities. The family closest in spirit to ours is warping-based synthesis, exemplified by MCNet [11] and StyleSync [12], which deforms a reference toward the target pose with an estimated flow field. However, it typically warps only a single reference and then refines the warped result with a decoder that resynthesizes the texture, which partially negates the very benefit of operating on real reference pixels.

Motivated by these observations, we advocate a deformation-based paradigm that preserves reference textures instead of hallucinating them. Rather than regenerating the mouth, we predict an independent flow field for each of a diverse set of references and warp them toward the target topological shape, which explicitly retains pre-existing high-frequency details such as teeth and lip wrinkles rather than destructively resynthesizing them. Because these flow fields are computed in parallel, the approach fully exploits

GPU parallelism, so that incorporating multiple references does not compromise inference speed. Building on this principle, we develop EfficientSync. Turning the paradigm into a working system, however, raises two questions, which we address in turn.

How should the spatially aligned references be fused without destroying their textures? Conventional convolutional fusion concatenates references along the channel dimension and processes them with stacked convolutions, which couples channels and averages out fine details; spatial self-attention, the fusion operator favored by diffusion backbones, instead fractures continuous textures into patches and scales quadratically with spatial resolution. We observe that, once spatially aligned, the references form a pool of candidate features sharing identical semantic channels at the same locations, so fusion reduces to a channel-wise selection problem: for each channel, choosing the most suitable candidate under the current context. We therefore propose the Dynamic Texture Mixer (DTM), which, as illustrated in Fig. 1, treats each reference as a track on a mixing console and aggregates them by context-aware, channel-wise weighted summation, preserving texture integrity at a markedly lower computational cost. Because the quality of this aggregation is ultimately bounded by the reference pool, yet random sampling often returns blurry or near-duplicate frames, we further introduce STAR Sampling, a zero-overhead preprocessing step that retrieves the sharpest and topologically most diverse frames to strengthen the candidate pool without adding inference latency.

![](images/adb5ebae24ac224ea3ad75a8f2e9bf95a06640646477b2632a4ede23d2572f7c.jpg)  
(a) Square Mask

![](images/1cf827dd77c66468fce8f9677be422c2cb2e16ad6f748239cb898cbf7df95a88.jpg)  
(b) Fixed Mask

![](images/6d8a557c6df0fca2262641291034b73b6ab5c41c9645856d5d015f6949c9c184.jpg)  
(c) Adaptive Mask  
Fig. 2: Existing mask strategies. (a) Square mask used in DINet [2]. (b) Fixed mask used in LatentSync [3]. (c) Adaptive mask used in IPTalker [13].

How should the synthesized mouth be composited back into the original background? This is where the second deficiency surfaces: keeping the surrounding regions truly untouched requires local editing that generates only the mouth and fuses it with the source. Yet the prevailing fusion paradigm faces a long-standing dilemma over the mask range (Fig. 2): a fixed large mask blocks lower-face leakage but covers part of the background and thus leaves conspicuous seam artifacts during fusion, whereas an adaptive mask preserves the background but induces the Shortcut Learning problem, where the model infers the lip shape from the surrounding chin rather than from the audio and consequently produces inaccurate articulation under cross-audio inference. We attribute this dilemma to treating the source background as a fixed, immutable environment, and instead decouple the source frame into global attributes that condition lip generation and a background prior that is independent of the lip shape. Realizing this idea, we propose the Spatio-Temporal Shifted Adaptive Masking (STAM) strategy, which conditions lip generation on a fixed large mask to fully suppress leakage while supplying background textures through an independent background reference, thereby blending the synthesized mouth seamlessly into the original background and eliminating boundary artifacts.

Integrating these designs, EfficientSync delivers stateof-the-art high-fidelity synthesis and identity preservation while operating at a real-time speed of 166 FPS. Our primary contributions are summarized as follows:

• We advocate a deformation-based paradigm that retains real reference textures rather than hallucinating them, and propose the Dynamic Texture Mixer, which reformulates multi-reference fusion as a channel-wise selection problem and aggregates aligned references under global context with low computational overhead.

• We propose the Spatio-Temporal Shifted Adaptive Masking strategy, which decouples the source frame into lip-generation conditions and an independent background prior, resolving the conflict between mask range and background fidelity.

• We introduce STAR Sampling, which improves the quality of the reference pool, and hence the final generation, while incurring no additional inference cost.

• Extensive experiments demonstrate that EfficientSync attains state-of-the-art visual quality and identity preservation at a real-time speed of 166 FPS on a single GPU.

## 2 RELATED WORKS

## 2.1 Audio-driven Portrait Animation

Audio-driven portrait animation aims to synthesize talking head videos with natural head movements and facial expressions from a single static reference image and a driving audio signal. Early mainstream approaches primarily relied on explicit intermediate representations. For example, StyleTalk [14] leverages 3D Morphable Models (3DMM) [15] to derive expression parameters for synthesizing stylized faces. MakeItTalk [16] predicts a sequence of facial landmarks from speech content and speaker identity to drive the subsequent image rendering. Subsequently, NeRF-based [17] methods introduced implicit neural representations to improve 3D consistency and view stability. As a representative, AD-NeRF [18] constructs a dynamic Neural Radiance Field that is directly conditioned on audio features to output video sequences. DFRF [19] emphasizes identity generalization, equipping the model with the ability to rapidly adapt to unseen facial appearances using only sparse observation frames. Recently, diffusion models have brought significant breakthroughs to this domain. EMO [20] employs an end-to-end denoising architecture driven by weak conditional guidance to produce highly expressive facial dynamics. Hallo [21] introduces a reference net that shares the exact topology of the main diffusion network to extract texture information from the source image. Ani-Portrait [22] adopts a two-stage paradigm, first estimating intermediate landmarks from the inputs and then utilizing a diffusion model to translate these spatial guides into final video frames.

While these methods offer flexible control over poses and expressions, they are sub-optimal for lip-sync tasks. Lip-sync requires preserving the original pose, lighting, and background. However, portrait animation models tend to hallucinate unintended global movements and viewpoint shifts, disrupting the spatial-temporal consistency of the source video.

## 2.2 Audio-driven Lip Synchronization

The audio-driven lip synchronization task focuses specifically on video dubbing, aiming to precisely replace the mouth region of the lower-half face based on target audio while keeping the background and upper-face pose of the source video unchanged. Early pioneering works established the masked inpainting technical paradigm. Wav2Lip [1] incorporated a pre-trained audio-visual synchronization discriminator (SyncNet) to compel the generative network to capture highly accurate lip-audio correlations. Following this foundation, researchers introduced multi-dimensional feature transformations and advanced spatial operations. DINet [2] architects target mouth shapes by executing spatial deformations directly on the reference features. Through self-attention mechanisms, IPLAP [23] synthesizes talking heads by inferring lip and jaw landmarks conditioned on audio, reference points, and pose priors. VideoReTalking [24] proposes a sequential pipeline that first edits the expressions of original frames, serving them as references to guide the audio-driven lip-sync generation via masked frame blending. With the evolution of generative models, diffusion-based architectures have significantly improved the visual realism of the synthesized frames. MuseTalk [25] executes cross-attention fusion between audio and image signals within a frozen VAE latent space. LatentSync [3] formulates a direct channelwise concatenation of reference frames, masked inputs, and noisy latents before processing them through a U-Net. KeySync [26] specifically targets boundary artifacts by designing an anti-leakage mask processing pipeline coupled with an interpolation generation strategy.

However, existing methods often rely on heavy generative networks or multi-step denoising processes. This introduces high computational costs and hardware thresholds, resulting in slow inference speeds that hinder widespread deployment. Furthermore, while diffusion models excel at generating high-fidelity overall image quality, fine-grained deterministic control over mouth details is often lacking. This stochastic nature can easily lead to inconsistent intraoral textures and a subsequent loss of the original identity.

## 3 METHODOLOGY

Given a source frame, a driving audio, and a video of the same subject, our goal is to synthesize a lip-synchronized face that follows the audio while preserving identity, pose, and background. As illustrated in Fig. 3, the framework is organized around a deformation-based backbone and comprises three designs that directly address the questions raised above. (a) STAR Sampling first selects, from the source video, a reference pool that is both sharp and topologically diverse, supplying high-quality material for deformation at no inference cost. (b) The Dynamic Texture Mixer then aggregates the spatially aligned references into a high-fidelity mouth feature through a context-aware, channel-wise selection mechanism. (c) The Spatio-Temporal Shifted Adaptive Masking strategy finally composites the synthesized mouth back into the source frame, blending it seamlessly with the preserved background.

![](images/a188c2caa6c72e0177bcdef04aa34eef500ca1e809b3dbaee9dada5cba002927.jpg)  
Fig. 3: The framework or our EfficientSync. (a) STAR Sampling first selects, from the source video, a reference pool that is both sharp and topologically diverse, supplying high-quality material for deformation at no inference cost. (b) The Dynamic Texture Mixer then aggregates the spatially aligned references into a high-fidelity mouth feature through a context-aware, channel-wise selection mechanism. (c) The Spatio-Temporal Shifted Adaptive Masking strategy finally composites the synthesized mouth back into the source frame, blending it seamlessly with the preserved background.

## 3.1 Overview

We extract audio features A using Whisper [27]. To prevent information leakage, a fixed mask is applied to the source frame to construct the masked source I<sub>S</sub>. To explicitly preserve background pixels, we additionally retain a background reference $\mathbf { I } _ { B G }$ obtained via an adaptive mask conceptually similar to IPTalker [13]. The N reference frames $\{ \bar { \mathbf { I } _ { R } ^ { i } } \} _ { i = 1 } ^ { N }$ are not drawn at random but selected by STAR Sampling, described next. All source and reference frames are uniformly cropped to a 416 × 320 face region for strict spatial alignment.

As illustrated in Fig. 3, our framework follows a deformation-based paradigm and proceeds in three stages. First, STAR Sampling (Fig. 3(a)) retrieves a sharp and topologically diverse reference pool from the source video, from which lightweight extractors produce the source feature ${ \bf F } _ { s }$ and the reference features $\{ \mathbf { F } _ { r } ^ { i } \} _ { i = 1 } ^ { N }$ . Since the references differ from the target in head pose and mouth openness, each is warped to the target pose before fusion: an attention block first derives a per-reference warping condition from the source feature, the audio, and the reference features, and a flow field predicted under this condition is then applied by the differentiable warping operation of IPTalker [13], yielding the aligned features $\{ \mathbf { F } _ { w } ^ { i } \} _ { i = 1 } ^ { N }$ . Second, the Dynamic Texture Mixer (Fig. 3(b)) fuses these aligned references into a high-fidelity mouth feature $\mathbf { F } _ { m i x }$ through a context-aware, channel-wise selection mechanism that preserves the reference textures intact. Finally, the Spatio-Temporal Shifted Adaptive Masking strategy (Fig. 3(c)) composites $\mathbf { F } _ { m i x }$ back into the source frame, reusing the same warping step to align the background reference before a SPADE decoder renders the synchronized output $\mathbf { I } _ { o }$ . The following subsections detail our three core designs: STAR Sampling, the Dynamic Texture Mixer, and STAM.

![](images/c82cc6600da81854cd69a22d473caddbf4f55ac9709145646a22804ceb3e8fb5.jpg)  
Fig. 4: Foundational metrics for STAR Sampling. Visualization of the key geometric and textural measurements extracted from facial landmarks.

## 3.2 STAR Sampling

Because our framework deforms real reference pixels rather than hallucinating them, its synthesis quality is inherently bounded by the quality of the reference pool: if the selected references lack the phonetic textures demanded by the target audio, the deformation has no faithful exemplar to follow and degenerates into stretched artifacts or blurred intra-oral regions. Conventional random sampling aggravates this, as it tends to return homogeneous topologies and occasional motion-blurred frames. We therefore propose Sharpness and Topology-Aware Reference Sampling (STAR Sampling), a deterministic procedure that selects a reference pool which is both topologically diverse and visually sharp. As it runs only once before inference, STAR Sampling adds no latency to synthesis.

Feature Space Construction. To select references by their articulatory state rather than by raw appearance, we represent each frame as a set of geometry-aware attribute. As shown in Fig. 4, we first extract a few foundational measurements from each frame: the rigid facial width $W _ { \mathrm { f a c e } } ,$ measured between cheek anchors; the inner-mouth height $h _ { \mathrm { i n } }$ and width $w _ { \mathrm { i n } } ;$ the inner oral-cavity area $A _ { \mathrm { { m o u t h } } } ;$ and the mean normalized intensity $I _ { \mathrm { m e a n } } \in [ \dot { 0 } , 1 ]$ within the cavity. To remain invariant to subject scale across a sequence, all measurements are normalized by the facial width. The vertical aspect ratio and horizontal stretch describe the geometric state of the lips, $D _ { \mathrm { m a r } } = h _ { \mathrm { i n } } / W _ { \mathrm { f a c e } }$ and $D _ { \mathrm { w i d t h } } = { \bar { w } } _ { \mathrm { i n } } / W _ { \mathrm { f a c e } } .$ To characterize internal phonetic detail, the teeth exposure and cavity depth are derived from the mean oral intensity and normalized by the squared facial width:

![](images/5a90dfef8188725e21c99f60dfc9545a5b44081309b883d174f5d9ecfecc4a68.jpg)  
Fig. 5: Illustration of STAR Sampling. (1) Each frame’s sharpness is normalized within its k-nearest neighbors in $\mathbf { F } _ { \mathrm { t o p o } ^ { - } }$ space to obtain a relative clarity $\hat { D } _ { \mathrm { s h a r p } } .$ (2) The first anchor maximizes clarity over centrality, $\displaystyle \frac { \hat { D } _ { \mathrm { s h a r p } , i } } { D _ { \mathrm { m e a n } , i } + \epsilon } . \left( 3 \right)$ Each subsequent anchor maximizes the clarity-weighted farthest-point distance until N references are selected.

$$
D _ { \mathrm { t e e t h } } = I _ { \mathrm { m e a n } } { \cdot } \frac { A _ { \mathrm { m o u t h } } } { W _ { \mathrm { f a c e } } ^ { 2 } } , \qquad D _ { \mathrm { c a v i t y } } = ( 1 { - } I _ { \mathrm { m e a n } } ) { \cdot } \frac { A _ { \mathrm { m o u t h } } } { W _ { \mathrm { f a c e } } ^ { 2 } } ,\tag{1}
$$

where a high $D _ { \mathrm { t e e t h } }$ indicates exposed bright teeth and a high $D _ { \mathrm { c a v i t y } } \ \mathsf { a }$ deep, dark cavity. The fifth dimension is an image sharpness term that penalizes motion blur, defined as the variance of the Laplacian of the cropped face image, $D _ { \mathrm { s h a r p } } = \mathrm { V L } ( I _ { \mathrm { f a c e } } )$

Sharpness-Aware Farthest-Point Sampling. We first construct a four-dimensional topological subspace $\begin{array} { r l } { \mathbf { F _ { \mathrm { t o p o } } } } & { { } = } \end{array}$ $[ D _ { \mathrm { m a r } } , D _ { \mathrm { w i d t h } } , D _ { \mathrm { t e e t h } } , D _ { \mathrm { c a v i t y } } ] \in \mathbf { \hat { \mathbb { R } } } ^ { 4 }$ and the one-dimensional sharpness score $D _ { \mathrm { s h a r p } } ,$ and select N anchors through the sharpness-weighted farthest-point sampling summarized in Algorithm 1 and Fig, 5. The procedure consists of three steps, after $\mathbf { F } _ { \mathrm { t o p o } }$ is standardized by Z-score normalization so that all topological dimensions contribute comparably to the distance metric.   
Step 1: Local sharpness normalization. A global comparison of $D _ { \mathrm { s h a r p } }$ is misleading, since an open mouth with visible teeth naturally yields a higher Laplacian variance than a closed one. We therefore normalize each frame’s $D _ { \mathrm { s h a r p } }$ only against its k nearest neighbors in $\mathbf { F } _ { \mathrm { t o p o } } ,$ obtaining a relative clarity score $\hat { D } _ { \mathrm { s h a r p } } \in [ 0 . 1 , 1 ]$ . The lower bound of 0.1 guarantees that even the blurriest frame retains a small chance of selection when it is topologically rare, preventing permanent exclusion.   
Step 2: Anchor initialization. The first anchor should be both clear and representative, so we select the frame that maximizes relative clarity while lying closest to the topological centroid $\mu _ { \mathrm { t o p o } } .$

$$
a \gets \arg \operatorname* { m a x } _ { i } \ \frac { \hat { D } _ { \mathrm { s h a r p } , i } } { D _ { \mathrm { m e a n } , i } + \epsilon } , \quad D _ { \mathrm { m e a n } , i } = \| \mathbf { F } _ { \mathrm { t o p o } , i } - \mu _ { \mathrm { t o p o } } \| _ { 2 } ,\tag{2}
$$

where $D _ { \mathrm { m e a n } , i }$ is the distance of frame i to the centroid and ϵ avoids division by zero. This anchor initializes the selected set M, together with a shortest-distance array D used in the next step.

Step 3: Sharpness-weighted iterative sampling. Each subsequent anchor should instead be clear yet maximally diverse from M. We thus weight the FPS distance $\begin{array} { r l } { \dot { \mathcal { D } _ { i } } } & { { } = } \end{array}$ min ${ \bf \Lambda } _ { j \in \mathcal { M } } \| \mathbf { F } _ { \mathrm { t o p o } , i } - \mathbf { F } _ { \mathrm { t o p o } , j } \|$ <sub>2</sub> by the same relative clarity and

Algorithm 1 STAR Sampling Algorithm   
Input: Topology features $\mathbf { F } _ { \mathrm { t o p o } } \in \mathbb { R } ^ { M \times 4 } ,$ Sharpness metrics   
$\mathbf { D } _ { \mathrm { s h a r p } } ^ { \mathbf { \bar { \alpha } } } \in \dot { \mathbb { R } } ^ { M } .$ , Target anchor count N, Neighborhood size k   
Output: Selected reference frame indices   
M   
1: Standardize $\mathbf { F } _ { \mathrm { t o p o } }$ using Z-score normalization.   
2: # 1. Local Sharpness Normalization   
3: for each frame $i \in \{ 1 , \ldots , M \}$ do   
4: Find the k nearest neighbors of i in the $\mathbf { F } _ { \mathrm { t o p o } }$ space.   
5: Let $d _ { m i n }$ and $d _ { m a x }$ be the minimum and maximum   
$D _ { \mathrm { s h a r p } }$ in this local neighborhood.   
6: if $d _ { m a x } > d _ { m i n }$ then   
7: $\begin{array} { r } { \hat { D } _ { \mathrm { s h a r p } , i } \gets 0 . 1 + 0 . 9 \times \frac { D _ { \mathrm { s h a r p } , i } - d _ { m i n } } { d _ { m a x } - d _ { m i n } } } \end{array}$   
8: else   
9: $\hat { D } _ { \mathrm { s h a r p } , i } \gets 1 . 0$   
10: end if   
11: end for   
12: # 2. Anchor Initialization   
13: Compute distance to mean topology: $D _ { m e a n , i } \quad \gets$   
$\lVert \mathbf { F } _ { \mathrm { t o p o } , i } - \mu _ { \mathrm { t o p o } } \rVert _ { 2 }$   
14: Select the first anchor a ← arg max $\mathrm { \Omega } _ { \mathrm { : } } ( \hat { D } _ { \mathrm { s h a r p , } i } / ( D _ { m e a n , i } +$   
ϵ))   
15: Initialize selected set ${ \mathcal { M } } \gets \{ a \}$   
16: Initialize shortest distance array $\textit { \textbf { D } }  \infty$ for all M   
frames, $\mathcal { D } _ { a } \gets 0$   
17: # 3. Sharpness-Weighted Iterative Sampling   
18: while $| { \dot { \mathcal { M } } } | < N$ do   
19: for each unselected frame i /∈ M do   
20: Update shortest distance to selected set:   
$\begin{array} { r } { \hat { \mathcal { D } _ { i } } \gets \operatorname* { m i n } ( \mathcal { D } _ { i } , \| \mathbf { F } _ { \mathrm { t o p o } , i } - \mathbf { F } _ { \mathrm { t o p o } , a } \| _ { 2 } ) } \end{array}$   
21: Compute final weighted score: Scor $\mathbf \Lambda _ { \mathcal { S } _ { i } } ~ \gets ~ \mathcal { D } _ { i } ~ \times$   
$\hat { D } _ { \mathrm { s h a r p } , i }$   
22: end for   
23: Select next frame a ← arg max<sub>i</sub> Score<sub>i</sub>   
24: ${ \mathcal { M } } \gets { \mathcal { M } } \cup \{ a \}$   
25: $\mathcal { D } _ { a } \gets 0$   
26: end while   
27: return M

![](images/ab869da24817f82777db9ae0d47234215175019a9352b671bee906d9a1cfd0cb.jpg)  
Fig. 6: Comparison of feature fusion mechanisms. (a) Convolutional Fusion aggregates concatenated reference features using standard convolutional layers. (b) Spatial Self-Attention computes attention weights based on flattened spatial patches. (c) Our proposed Context-Aware Feature Synthesis (CAFS) leverages global context to predict channel-wise routing scores, aggregating diverse candidate features through an efficient channel-wise weighted summation without spatial or channel entanglement.

select

$$
a \gets \arg \operatorname* { m a x } _ { i \notin \mathcal { M } } \mathcal { D } _ { i } \cdot \hat { D } _ { \mathrm { s h a r p } , i } ,\tag{3}
$$

repeating until $| { \mathcal { M } } | = N .$ The opposite roles of $D _ { \mathrm { m e a n } }$ (minimized at initialization) and D (maximized thereafter) let the first anchor capture a typical posture while the rest expand coverage toward rare but informative ones, with the clarity factor consistently penalizing blurred candidates in both phases. In practice, we draw $\tilde { M } = 1 0 0$ candidate frames at random to balance efficiency and coverage, and set the neighborhood size to $k = M / N$

## 3.3 Texture Fusion via the Dynamic Texture Mixer

The Spatial Alignment Module aligns the spatial structure of multiple references to the target pose, but this structural foundation alone is insufficient: owing to the dynamic deformation of mouth shapes and the transience of facial textures, no single warped reference can reconstruct rich, high-fidelity detail. It is therefore necessary to fuse complementary texture cues across the references without destroying the very textures that the deformation paradigm is meant to preserve.

As illustrated in Fig. 6, we categorize multi-reference feature fusion into three mechanisms. The first two are widely adopted in existing methods: convolutional fusion [2], [13] (Fig. 6(a)) and spatial self-attention [3], [25] (Fig. 6(b)). As analyzed below, both inadvertently corrupt the reference textures. We therefore propose a third mechanism, Context-Aware Feature Synthesis (CAFS) (Fig. 6(c)), which casts fusion as a channel-wise selection problem and thereby keeps the references intact. To adapt this mechanism to the lip-sync task, we instantiate it as the Dynamic Texture Mixer (DTM).

Convolutional Fusion. Convolutional fusion concatenates references along the channel dimension and processes them with multi-layer convolutions. It has two intrinsic drawbacks. First, fixed convolutional weights cannot adapt the fusion to the driving conditions; even with conditional injection (e.g., AdaIN) or channel attention [28], the underlying cross-reference and cross-channel convolutions still induce channel coupling, mixing independent features and averaging out high-frequency detail. Second, capturing global context requires stacking convolutions to enlarge the receptive field, and this repeated spatial aggregation further smooths textures while raising cost. Concretely, its complexity is $O ( N \cdot K ^ { 2 } \cdot H \cdot W \cdot C ^ { 2 } )$ , where $N , K ,$ and C are the number of references, the kernel size, and the channel dimension; since K is small, this simplifies to $O ( N \cdot H \cdot W \cdot C ^ { 2 } )$ . The tight coupling of spatial size, reference count, and dense channel interaction yields a heavy overhead.

Spatial Self-Attention. Spatial self-attention, commonly employed within denoising networks, partitions the feature map into patches of size $\mathbf { \hat { \boldsymbol { P } } } \times \boldsymbol { P } \times \boldsymbol { \hat { \boldsymbol { C } } } ,$ , each linearly projected into Query, Key, and Value for standard attention; in essence, it relates the full channel vector at one location to those at all other locations and updates features by a weighted sum. This scheme has three limitations. First, it shares attention weights across channels, applying an identical per-location weight to all channels and thereby ignoring their distinct spatial semantics. Second, partitioning the feature map into patches fractures the continuous textures, degrading high-frequency detail. Third, attending over all spatial patches forms a large attention matrix; for instance, with patch size $P ~ = ~ 1$ as in LatentSync [3], the complexity grows quadratically with spatial resolution, reaching $O ( ( { \overset { \cdot } { H } } { \overset { \cdot } { W } } ) ^ { 2 } C )$ and incurring substantial memory and compute costs.

Context-Aware Feature Synthesis. In summary, convolutional fusion erases high-frequency features through its fixed weights and linear summation, while spatial selfattention destroys reference texture priors and is too costly for real-time synthesis. Our third mechanism stems from a key observation about the features after spatial alignment. Because all references are processed by shared convolutional layers, identical channels carry the same visual semantics;

![](images/734b2185a8f7e2d9bab31680f15270d116a35e6b82d65d7989ce72568fdf9811.jpg)  
Fig. 7: Architecture of the Dynamic Texture Mixer. The cropped reference features are encoded and fused with the globa context through a Self-Attention block. Linear mapping and Softmax operations translate the updated tokens into channelspecific routing scores, under which the reference features undergo channel-wise weighted aggregation to synthesize the target mouth feature.

moreover, after the Spatial Alignment Module their spatial structures are highly aligned. Consequently, the references collectively provide a rich pool of candidate features at the same spatial locations within identical semantic channels, and fusion reduces to a selection task: for each semantic channel, choosing the most suitable candidate given the current context. Following this insight, CAFS uses a self-attention module to capture the global context, maps each candidate’s context-aware token into channel-specific routing weights through linear layers, and aggregates the candidates by channel-wise weighted summation. Crucially, it never injects conditions to modulate the image features themselves, so the original channel semantics, texture integrity, and spatial structure of the references are preserved intact. This selection-based design is also markedly more efficient: the self-attention and linear scoring take $\mathcal { \dot { O } } ( N ^ { 2 }$ $d + N \cdot C \cdot d )$ operations, where d is the token dimension, and the channel-wise aggregation takes $\mathcal { O } ( N \cdot H \cdot W \cdot C )$ Since the spatial resolution $\check { H } \times W$ typically dominates both N and $d ,$ the overall complexity simplifies to O(N·H·W·C), markedly lower than convolutional fusion $( \mathcal { O } ( \dot { N } { \cdot } H { \cdot } W { \cdot } C ^ { 2 } ) )$ and spatial self-attention $( \mathcal { O } ( ( H W ) ^ { 2 } \cdot C ) )$ ).

Dynamic Texture Mixer. We instantiate CAFS as the Dynamic Texture Mixer (DTM) to mix the set of warped reference features $\mathbf { F } _ { w } = \{ \mathbf { F } _ { w } ^ { n } \} _ { n = 1 } ^ { N }$ produced by SAM (as shown in Fig. 7). Since SAM operates on full-face references $( H \times W$ with $H > W )$ to align them with the source, their upper-face region becomes redundant for mouth synthesis and would only waste computation. We therefore crop each $\mathbf { F } _ { w } ^ { n }$ along the height dimension down to $W _ { ☉ }$ , retaining a square $W \times W$ lower-face region denoted $\mathbf { F } _ { c r o p } ^ { n } .$

The DTM then proceeds along two parallel branches. The first prepares the candidates to be mixed: an expansion network $\mathcal { E } _ { e x p }$ lifts each $\mathbf { F } _ { c r o p } ^ { n }$ into a higher channel dimension $C ^ { \prime }$ to enlarge the representational capacity for aggregation,

$$
\mathbf { F } _ { e x p } ^ { n } = \mathcal { E } _ { e x p } ( \mathbf { F } _ { c r o p } ^ { n } ) \in \mathbb { R } ^ { W \times W \times C ^ { \prime } } , \quad \forall n \in [ 1 , N ] .\tag{4}
$$

The second branch prepares the routing condition: a condition encoder ${ \mathcal { E } } _ { m }$ summarizes each $\mathbf { F } _ { c r o p } ^ { n }$ into a single token $\mathbf { e } _ { w } ^ { n }$ that conveys the current state of that reference. We concatenate these tokens with the context carried over from SAM, namely $\{ \mathbf { w } _ { r } ^ { i } \} _ { i = 1 } ^ { N } , \mathbf { e } _ { s } ,$ , and $\mathbf { e } _ { a } ,$ , and pass them through a multi-head self-attention block; the output tokens $\{ \mathbf { m } _ { w } ^ { n } \} _ { n = 1 } ^ { N }$ corresponding to $\{ \mathbf { e } _ { w } ^ { n } \}$ thus encode each reference’s relevance under the global driving context. A linear layer $\mathcal { F } _ { p r o j }$ then maps each $\mathbf { m } _ { w } ^ { n }$ into a group-wise routing score:

$$
\mathbf { Y } ^ { n } = \mathcal { F } _ { p r o j } ( \mathbf { m } _ { w } ^ { n } ) \in \mathbb { R } ^ { G } , \quad \forall n \in [ 1 , N ] ,\tag{5}
$$

where $G \ll C ^ { \prime }$ is the number of channel groups. Predicting one score per group, rather than per channel, avoids regressing $N \times \hat { C } ^ { \prime }$ free weights and reflects the prior that strongly correlated channels should share a routing decision.

To recover channel-wise weights, the group score $\mathbf { Y } _ { \boldsymbol { q } } ^ { n }$ is replicated across all channels of its group g to form $\bar { \mathbf { Y } } _ { c } ^ { n }$ While this keeps the coarse decision consistent within a group, identical weights would over-constrain the aggregation; we therefore attach a learnable per-channel temperature $\mathbf { T } _ { c }$ that lets each channel sharpen or soften the shared score independently. The final aggregation weight of the nth reference at channel c is obtained by a Softmax over the $N$ references:

$$
\mathbf { S } _ { c } ^ { n } = \frac { \exp ( \mathbf { Y } _ { c } ^ { n } / \mathbf { T } _ { c } ) } { \sum _ { i = 1 } ^ { N } \exp ( \mathbf { Y } _ { c } ^ { i } / \mathbf { T } _ { c } ) } , \quad \forall c \in [ 1 , C ^ { \prime } ] , \ : \forall n \in [ 1 , N ] .\tag{6}
$$

Stacking $\mathbf { S } _ { c } ^ { n }$ over channels gives the per-reference weight vector $\mathbf { \bar { S } } ^ { n } \in \mathbb { R } ^ { C ^ { \prime } }$ . Finally, we aggregate the expanded candidates by a channel-wise weighted summation and project the result back to the original channel size through a reduction network ${ \mathcal { E } } _ { r e d } ,$ yielding the mixed mouth feature $\mathbf { F } _ { m i x } \colon$

$$
{ \bf F } _ { m i x } = { \mathscr E } _ { r e d } \left( \sum _ { n = 1 } ^ { N } { \bf S } ^ { n } \odot { \bf F } _ { e x p } ^ { n } \right) ,\tag{7}
$$

where ⊙ denotes channel-wise multiplication broadcast across the spatial dimensions. Because each output channel is a convex combination of the corresponding channel across references, the DTM selects rather than synthesizes, leaving the spatial structure and high-frequency texture of every reference intact.

## 3.4 Spatio-Temporal Shifted Adaptive Masking

As discussed in Sec. 1, prevailing mask strategies face a dilemma between leakage suppression and background fidelity, whose root cause is treating the source background as a fixed, immutable environment. We instead decouple the source frame into global attributes that condition lip generation and a background prior that is independent of the lip shape, and realize this idea through the Spatio-Temporal Shifted Adaptive Masking (STAM) strategy. Lip generation is conditioned solely on the source frame under a fixed large mask, which entirely deprives the network of real visual hints about the lower face, while the background prior is supplied by a separate background reference I<sub>BG</sub>.

The key to STAM lies in how this background reference is constructed during training versus inference. During training, rather than using the source frame itself, we take an adjacent frame at a small temporal offset as the background reference. The rationale is that, within a short window, the head moves little, so the surrounding background and torso change only marginally and still constitute a highfidelity texture reference; yet the lip shape of this shifted frame differs from the target, so its chin position is misaligned. Confronted with this mismatch, the model cannot trivially copy pixels and is instead forced to learn realistic elastic stretching and redrawing of the background skin and jawline in accordance with the newly generated lip. During inference, the background reference reduces to the current source frame, that is, a zero temporal offset, which preserves the genuine background exactly while the model, trained under the stricter shifted regime, retains the spatial adaptivity needed to stretch the chin and blend the audiodriven mouth seamlessly into the surroundings, eliminating double chins and boundary seams.

Concretely, a background encoder $\mathcal { E } _ { b g }$ extracts a background feature $\mathbf { F } _ { b g }$ from $\mathbf { I } _ { B G }$ . Since $\mathbf { F } _ { m i x }$ covers only the cropped mouth region, we zero-pad it back to the fullface spatial size, denoted $\hat { \mathbf { F } } _ { m i x } ,$ and concatenate it with the source feature ${ \bf F } _ { s }$ and the background feature $\mathbf { F } _ { b g }$ to predict the background motion flow:

$$
\mathbf { M } _ { b g } = \mathcal { F } _ { c o n v } \big ( \hat { \mathbf { F } } _ { m i x } \oplus \mathbf { F } _ { s } \oplus \mathbf { F } _ { b g } \big ) .\tag{8}
$$

The background feature is then warped by $\mathbf { M } _ { b g }$ and cropped to the mouth-region size, yielding the aligned background feature $\hat { \mathbf { F } } _ { b g } = \mathrm { C r o p } ( \mathrm { W a r p } ( \mathbf { F } _ { b g } , \tilde { \mathbf { M } _ { b g } } ) )$ . Finally, $\mathbf { F } _ { m i x }$ and $\hat { \mathbf { F } } _ { b q }$ are concatenated along the channel dimension and decoded by a SPADE decoder into the synchronized output face:

$$
\mathbf { I } _ { o } = \mathcal { F } _ { s p a d e } ( \mathbf { F } _ { m i x } \oplus \hat { \mathbf { F } } _ { b g } ) .\tag{9}
$$

## 3.5 Loss Function

We train the model with a composite objective. Let $\mathbf { I } _ { o }$ and $\mathbf { I } _ { g t }$ denote the output image and the corresponding ground-truth frame. To encourage realistic images with sharp high-frequency detail, we adopt the hinge adversarial loss $\stackrel { \cdot } { \mathcal { L } } _ { g a n }$ [29], comprising the standard discriminator and generator terms. A pre-trained VGG network [30] provides a multi-scale perceptual loss $\mathcal { L } _ { p e r c }$ that enforces global structural consistency, computed as the $\ell _ { 1 }$ distance between the activation maps of $\mathbf { I } _ { o }$ and $\mathbf { I } _ { g t }$ . To preserve fine-grained intraoral detail, we apply an LPIPS loss [31] $\mathcal { L } _ { l p i p s }$ restricted to the central mouth region of the generated image. To enforce strict audio-visual synchronization, a pre-trained SyncNet [1] yields a sync loss $\mathcal { L } _ { \it s y n c }$ over each window of 16 consecutive frames and the corresponding audio. We further incorporate the TREPA loss [3] $\backslash \mathcal { L } _ { t r e p a } ,$ measured as the squared distance between the embeddings of generated and ground-truth clips under a self-supervised video encoder, to improve the temporal naturalness of the synthesized textures. The total objective is the weighted sum of these terms:

$$
\mathcal { L } _ { t o t a l } = \lambda _ { 1 } \mathcal { L } _ { g a n } + \lambda _ { 2 } \mathcal { L } _ { p e r c } + \lambda _ { 3 } \mathcal { L } _ { l p i p s } + \lambda _ { 4 } \mathcal { L } _ { s y n c } + \lambda _ { 5 } \mathcal { L } _ { t r e p a } ,
$$

where $\lambda _ { 1 } , \ldots , \lambda _ { 5 }$ are the corresponding weighting hyperparameters.

(10)

## 4 EXPERIMENTS

Datasets. We conduct experiments on two high-resolution talking-face datasets, HDTF [32] and VFHQ [33]. HDTF comprises 362 high-definition videos at resolutions ranging from roughly 720p to 1080p, while VFHQ contains about 8,000 videos spanning more than 20 countries and a wide variety of languages. For evaluation, we randomly select 20 videos from the test set of each dataset.

Implementation Details. All videos are resampled to 25 fps, and 468 facial landmarks are extracted with MediaPipe [34]. Based on these landmarks, the facial region is cropped and resized to a 416×320 resolution, and the audio is resampled to 16 kHz. During training, the number of reference frames N is randomly sampled from [3, 7].

Evaluation Metrics. We evaluate our method along three dimensions. (1) Visual quality. We measure structural fidelity with SSIM [35] and perceptual similarity with LPIPS [31], assess overall frame-level and video-level quality with the Fréchet Inception Distance (FID) [36] and Fréchet Video Distance (FVD) [37], respectively, and quantify image blurriness with VL [38]. (2) Lip-sync accuracy. We adopt $\mathrm { S y n c } _ { \mathrm { c o n f } }$ [39] to gauge audio-visual synchronization. (3) Efficiency. To reflect practical deployability, we report inference speed (FPS), peak memory usage (Mem, in MB), and floating-point operations per frame (GFLOPs/f).

## 4.1 Quantitative Evaluation

We compare EfficientSync against recent state-of-the-art methods, namely IPLAP [23], DINet [2], LatentSync [3], IPTalker [13], and KeySync [26], under both the reconstruction and cross-audio settings. The results are reported in Tables 2 and 3, where the best and second-best scores are highlighted in bold and underlined, respectively. For all comparisons, the reference frame count of our model is fixed to $5 ,$ which is a common setting in previous multi-reference models.

Visual quality. As shown in Table 2, EfficientSync attains the best reconstruction quality on the majority of visual metrics across both datasets. On HDTF it reaches the highest SSIM (0.979), LPIPS (0.013), VL (58.69), FID (1.82), and FVD (49.14), improving the strongest baseline LatentSync on FID by 22% (1.82 vs. 2.34) and on VL by 22% (58.69 vs. 47.91). The same trend holds on VFHQ, where it leads on SSIM (0.967), LPIPS (0.022), and FID (4.13), more than halving the FID of LatentSync (4.13 vs. 5.78). These consistent gains in sharpness-sensitive metrics (VL, LPIPS) confirm that deforming real reference pixels preserves fine intraoral texture far better than generative resynthesis. The crossaudio results in Table 3 echo this conclusion: EfficientSync obtains the best VL and FID on both datasets, for instance reducing the HDTF FID to 1.92 against LatentSync’s 2.47.

![](images/dad86f143c9b3675afba13ac18c0cd3a3029e74420443b0873964f40c80b163d.jpg)  
Fig. 8: Qualitative comparisons with state-of-the-art works under the cross-audio setting. EfficientSync preserves sharp, identity-consistent intra-oral textures, whereas competing methods either blur the mouth region or hallucinate teeth that deviate from the source identity.

Lip-sync accuracy. On $\mathrm { S y n c } _ { \mathrm { c o n f } } ,$ our method occupies a competitive mid-tier position rather than the top. Under cross-audio on HDTF it scores 7.59, trailing the diffusionbased LatentSync (9.04) and KeySync (8.55) but clearly surpassing the warping-based IPTalker (4.62), DINet (7.36), and IPLAP (5.86). This gap is expected: diffusion backbones optimize synchronization through a powerful generative prior at the cost of identity and speed, whereas our deformation paradigm prioritizes texture fidelity. Notably, EfficientSync still achieves the strongest synchronization among all deformation- and GAN-based methods, indicating that the accuracy trade-off is modest and well compensated by its substantial gains in fidelity and efficiency.

TABLE 1: Comparison of computational efficiency on a single NVIDIA A100 GPU.
<table><tr><td>Method</td><td>FPS↑</td><td>Mem (MB) ↓</td><td>GFLOPs/f↓</td></tr><tr><td>IPLAP [23]</td><td>0.98</td><td>8848</td><td>2356</td></tr><tr><td>DINet [2]</td><td>10.73</td><td>349</td><td>180</td></tr><tr><td>IPTalker [13]</td><td>27.21</td><td>565</td><td>366</td></tr><tr><td>LatentSync [3]</td><td>4.39</td><td>11704</td><td>39840</td></tr><tr><td>KeySync [26]</td><td>1.07</td><td>38363</td><td>16358</td></tr><tr><td>EfficientSync (Ours)</td><td>166.01</td><td>718</td><td>245</td></tr></table>

Efficiency. Table 1 reports efficiency on a single NVIDIA A100 GPU. EfficientSync runs at 166.01 FPS, over 6× faster than the quickest prior method IPTalker (27.21 FPS) and roughly 38× faster than the diffusion-based LatentSync (4.39 FPS), while consuming only 245 GFLOPs per frame, two orders of magnitude below LatentSync (39,840) and KeySync (16,358), and 718 MB of peak memory against their 11.7 GB and 38.4 GB. This efficiency is not incidental but a direct consequence of our design. First, unlike diffusion baselines that reconstruct the entire face from noise through tens of denoising steps, EfficientSync edits only the lower-face region in a single forward pass, avoiding both the iterative sampling that dominates their latency and the redundant regeneration of the unchanged upper face. Second, the Dynamic Texture Mixer replaces spatial selfattention, whose cost grows quadratically with resolution as $O ( ( H W ) ^ { 2 } C )$ , with a channel-wise selection whose cost is linear in the spatial size, $O ( N \cdot H \cdot W \cdot C ) _ { \cdot }$ ; this is the primary source of our low GFLOPs and memory, as it never materializes the large spatial attention matrices that the diffusion methods rely on. Third, the reference flows are predicted and applied in parallel, so enlarging the reference pool adds little latency. Built entirely from lightweight convolutional and linear operators rather than a heavy VAE-UNet denoiser, EfficientSync remains deployable under constrained hardware, a regime in which the diffusion competitors are impractical.

## 4.2 Qualitative Evaluation

The cross-audio comparisons in Fig. 8 corroborate the quantitative results. IPLAP and DINet produce visibly blurred mouth interiors, and although IPTalker and LatentSync sharpen the lips somewhat, a clear gap to the source textures remains. KeySync yields the crispest teeth but at the cost of fidelity, frequently deviating from the ground truth and producing exaggerated mouth shapes. EfficientSync instead strikes the best balance, rendering sharp, realistic lip motion whose intra-oral textures stay faithful to the reference identity and free of boundary artifacts.

## 4.3 Ablation Study

We ablate the two model-side designs, the Dynamic Texture Mixer (DTM) and the STAM strategy, on HDTF in Table 4. Feature fusion. Replacing DTM with multi-layer convolutions (w/ Conv Fusion) or spatial self-attention (w/ Attn Fusion) degrades nearly all visual metrics: under reconstruction, FVD rises from 49.14 to 55.07 and 54.23 respectively, and FID from 1.82 to 2.06 and 2.28. The effect is far more pronounced under cross-audio, where the entanglement of differing reference mouth shapes in these operators corrupts lip motion: $\mathrm { S y n c } _ { \mathrm { c o n f } }$ drops from our 7.59 to 5.87 (Conv) and 6.91 (Attn). This confirms that the channel-wise selection in DTM, by keeping references disentangled, is essential not only for texture fidelity but also for accurate synchronization.

Mask strategy. Removing the background reference and synthesizing directly from the fixed-mask source (w/o STAM) degrades every visual metric, raising reconstruction FVD from 49.14 to 53.54 and LPIPS from 0.0127 to 0.0149. As Fig. 9 shows, this manifests as blurred background around the lower face and visible seams between the synthesized mouth and the surrounding region, whereas the full model preserves the background and blends seamlessly.

Effect of STAR Sampling. We also study STAR Sampling, which selects the reference pool described in Sec. 3.2. Table 5 compares it against random sampling and Farthest Point Sampling (FPS) under both reconstruction and cross-audio settings on HDTF, sweeping the number of references. Removing the sharpness-aware component from STAR Sampling effectively reduces it to FPS. Three observations stand out. First, introducing topological diversity via FPS significantly improves sampling efficiency and robustness. In reconstruction, FPS achieves a highly competitive $\mathrm { S y n c } _ { \mathrm { c o n f } }$ of 9.114 with only 8 references, whereas random sampling requires 16 to reach its peak. This advantage is more pronounced in cross-audio generation, where pure random selection struggles, while FPS preserves crucial extreme poses to maintain consistently higher scores. Second, incorporating sharpness awareness further elevates performance. In reconstruction, STAR Sampling reaches the highest overall $\mathrm { S y n c } _ { \mathrm { c o n f } }$ peak of 9.177, and in the cross-audio setting, it similarly attains the absolute peak of 7.569, showing that sharp structural priors consistently benefit lip-sync synthesis. Third, in the cross-audio setting, the peak performance for all three sampling methods consistently appears at exactly 4 references. Unlike reconstruction, which benefits from a larger reference pool (e.g., 8 or 16 frames) to capture fine-grained personal habits, cross-identity driving favors a more concise set of anchors. Supplying too many references in cross-driving likely introduces redundant, identityspecific topological nuances that interfere with the generalized audio-driven deformations, indicating that a compact but diverse reference pool is optimal for generalizable synthesis.

![](images/d30df68f84ee4180835f87811d455ec57d6e78335d1817aa0cd6f524ab308b7b.jpg)  
Fig. 9: Visual comparison for the STAM ablation. As highlighted by the red boxes, removing STAM (w/o STAM) causes spatial discontinuities and blurred background around the lower face, whereas our full model (w STAM) preserves the background and blends seamlessly.

## 4.4 User Study

We finally conduct a user study to compare EfficientSync against several baseline lip-sync models on videos generated from the test set. A total of 25 participants were presented with video pairs for each trial: one generated by our method and the other by a randomly selected baseline. Participants were asked to evaluate the videos based on visual quality and lip-sync accuracy, choosing the superior video or indicating that both were of equal quality. As shown in Fig. 10, users demonstrated a clear preference for EfficientSync in both visual quality and lip-sync accuracy over most baselines. The only exception was lip-sync accuracy against KeySync, where EfficientSync received slightly fewer votes; however, nearly half of the responses considered the two models to be comparable. Overall, the study highlights that EfficientSync achieves the best visual quality while maintaining highly competitive lip-sync accuracy.

TABLE 2: Quantitative comparison of reconstruction performance on the HDTF and VFHQ datasets.
<table><tr><td rowspan="2">Method</td><td colspan="6">HDTF</td><td colspan="6">VFHQ</td></tr><tr><td>SSIM↑</td><td>LPIPS↓</td><td>VL↑</td><td>FID↓</td><td>FVD↓</td><td> $\mathrm { S y n c } _ { \mathrm { c o n f } } \uparrow$ </td><td>SSIM↑</td><td>LPIPS↓</td><td>VL↑</td><td>FID↓</td><td>FVD↓</td><td> $\mathrm { S y n c } _ { \mathrm { c o n f } } \uparrow$ </td></tr><tr><td>IPLAP [23]</td><td>0.958</td><td>0.032</td><td>37.615</td><td>3.895</td><td>95.104</td><td>8.039</td><td>0.942</td><td>0.057</td><td>61.359</td><td>9.451</td><td>117.370</td><td>5.751</td></tr><tr><td>DINet [2]</td><td>0.952</td><td>0.030</td><td>47.354</td><td>3.690</td><td>94.625</td><td>8.851</td><td>0.925</td><td>0.071</td><td>59.505</td><td>20.399</td><td>260.732</td><td>5.140</td></tr><tr><td>IPTalker [13]</td><td>0.971</td><td>0.018</td><td>51.750</td><td>4.433</td><td>61.798</td><td>8.885</td><td>0.945</td><td>0.040</td><td>85.791</td><td>9.365</td><td>167.655</td><td>5.530</td></tr><tr><td>LatentSync [3]</td><td>0.963</td><td>0.028</td><td>47.906</td><td>2.342</td><td>50.820</td><td>9.183</td><td>0.952</td><td>0.042</td><td>68.544</td><td>5.782</td><td>54.241</td><td>7.765</td></tr><tr><td>KeySync [26]</td><td>0.906</td><td>0.064</td><td>42.868</td><td>11.321</td><td>121.321</td><td>8.765</td><td>0.878</td><td>0.097</td><td>74.002</td><td>23.421</td><td>149.354</td><td>7.747</td></tr><tr><td>EfficientSync (Ours)</td><td>0.979</td><td>0.013</td><td>58.691</td><td>1.824</td><td>49.144</td><td>9.107</td><td>0.967</td><td>0.022</td><td>85.410</td><td>4.133</td><td>88.839</td><td>6.058</td></tr></table>

TABLE 3: Quantitative comparison of cross-audio performance on the HDTF and VFHQ datasets.
<table><tr><td rowspan="2">Method</td><td colspan="4">HDTF</td><td colspan="4">VFHQ</td></tr><tr><td>VL↑</td><td>FID↓</td><td>FVD↓</td><td> $\mathrm { S y n c } _ { \mathrm { c o n f } } \uparrow$ </td><td>VL↑</td><td>FID↓</td><td>FVD↓</td><td> $\mathrm { S y n c } _ { \mathrm { c o n f } } \uparrow$ </td></tr><tr><td>IPLAP [23]</td><td>37.595</td><td>4.052</td><td>110.420</td><td>5.862</td><td>61.534</td><td>18.925</td><td>159.798</td><td>4.476</td></tr><tr><td>DINet [2]</td><td>46.039</td><td>3.660</td><td>118.889</td><td>7.355</td><td>59.229</td><td>11.112</td><td>286.577</td><td>4.369</td></tr><tr><td>IPTalker [13]</td><td>51.733</td><td>4.606</td><td>87.501</td><td>4.622</td><td>90.774</td><td>10.260</td><td>201.464</td><td>3.231</td></tr><tr><td>LatentSync [3]</td><td>47.714</td><td>2.470</td><td>78.510</td><td>9.037</td><td>70.346</td><td>7.745</td><td>149.811</td><td>6.550</td></tr><tr><td>KeySync [26]</td><td>42.964</td><td>11.258</td><td>165.393</td><td>8.547</td><td>79.739</td><td>25.876</td><td>258.167</td><td>6.989</td></tr><tr><td>EfficientSync (Ours)</td><td>57.331</td><td>1.916</td><td>86.441</td><td>7.589</td><td>93.055</td><td>5.608</td><td>151.852</td><td>4.494</td></tr></table>

![](images/599ad8de144775ab67f8c4c963e3ae6a8fd789abd269d315bb0788b2e55aa541.jpg)  
Fig. 10: User Preference Study. We evaluated EfficientSync against random baselines through a user study focusing on Visual Quality and Lip Sync. The results display the preference rates as a percentage of total votes, indicating whether participants preferred our model (Ours), the competing model (Other), or viewed them as comparable (Same).

TABLE 4: Ablation study on HDTF.
<table><tr><td>Variant</td><td>SSIM ↑ LPIPS↓</td><td>VL↑</td><td></td><td>FID↓ FVD↓</td><td></td><td> $\mathrm { S y n c } _ { \mathrm { c o n f } } \uparrow$ </td></tr><tr><td colspan="7">Reconstruction</td></tr><tr><td>w/ Conv Fusion</td><td>0.9779</td><td>0.0132</td><td>57.281</td><td>2.055</td><td>55.071</td><td>8.947</td></tr><tr><td>w/ Attn Fusion</td><td>0.9786</td><td>0.0132</td><td>58.680</td><td>2.284</td><td>54.225</td><td>9.043</td></tr><tr><td>w/o STAM</td><td>0.9761</td><td>0.0149</td><td>56.900</td><td>1.945</td><td>53.541</td><td>8.670</td></tr><tr><td>Ours</td><td>0.9791</td><td>0.0127</td><td>58.691</td><td>1.824</td><td>49.144</td><td>9.107</td></tr><tr><td colspan="7">Cross-Audio</td></tr><tr><td>w/ Conv Fusion</td><td>1</td><td></td><td>57.275</td><td>2.319</td><td>91.750</td><td>5.869</td></tr><tr><td>w/ Attn Fusion</td><td>一</td><td></td><td>58.649</td><td>2.415</td><td>86.644</td><td>6.913</td></tr><tr><td>w/o STAM</td><td></td><td></td><td>56.840</td><td>2.347</td><td>88.847</td><td>7.346</td></tr><tr><td>Ours</td><td></td><td></td><td>57.331</td><td>1.916</td><td>86.441</td><td>7.589</td></tr></table>

## 5 ANALYSIS

Beyond benchmarking against prior methods, we now examine why EfficientSync works, focusing on how the Dynamic Texture Mixer (DTM) routes texture across references and how STAR Sampling shapes that routing. Unless otherwise noted, all analyses are conducted under the cross-audio setting on HDTF.

## 5.1 Channel-wise Routing Characteristics

Recall that the DTM treats multi-reference fusion as a channel-wise routing process (Sec. ??), akin to a mixing console that preserves texture integrity rather than homogenizing it as conventional convolutions do. We first ask whether the learned routing actually exhibits the structure this design intends. To this end, we analyze the temporal dynamics of the channel-wise aggregation through a Temporal Routing Divergence (TRD), denoted $\mathcal { D } _ { T R }$ . For a given channel $c ,$ we compute its global temporal anchor $\bar { \bar { \mathbf { S } } } _ { c }$ as the time-averaged aggregation weight distribution across all T frames, and define $\bar { \mathcal { D } } _ { T R } ( c )$ as the expected Kullback-Leibler (KL) divergence between the instantaneous distribution $\mathbf { S } _ { c } ( t )$ and this anchor:

TABLE 5: Ablation on the number of reference frames and STAR Sampling on HDTF.
<table><tr><td rowspan="2">Ref. Num.</td><td colspan="2">Random Sampling</td><td colspan="2">FPS</td><td colspan="2">STAR Sampling</td></tr><tr><td> $\mathrm { S y n c } _ { \mathrm { c o n f } } \uparrow$ </td><td>VL↑</td><td> $\mathrm { S y n c } _ { \mathrm { c o n f } } \uparrow$ </td><td>VL↑</td><td> $\mathrm { S y n c } _ { \mathrm { c o n f } } \ 1$ </td><td>VL↑</td></tr><tr><td colspan="7">Reconstruction</td></tr><tr><td>1</td><td>8.391</td><td>58.240</td><td>8.505</td><td>58.206</td><td>8.552</td><td>58.398</td></tr><tr><td>2</td><td>8.771</td><td>58.538</td><td>8.833</td><td>58.579</td><td>8.910</td><td>58.703</td></tr><tr><td>4</td><td>9.004</td><td>58.585</td><td>9.065</td><td>58.702</td><td>9.113</td><td>58.720</td></tr><tr><td>8</td><td>9.076</td><td>58.651</td><td>9.114</td><td>58.669</td><td>9.177</td><td>58.680</td></tr><tr><td>16</td><td>9.147</td><td>58.670</td><td>9.091</td><td>58.675</td><td>9.121</td><td>58.700</td></tr><tr><td>32</td><td>9.078</td><td>58.672</td><td>9.079</td><td>58.666</td><td>9.077</td><td>58.680</td></tr><tr><td> $\operatorname { A v g } .$ </td><td>8.911</td><td>58.559</td><td>8.948</td><td>58.583</td><td>8.992</td><td>58.647</td></tr><tr><td colspan="7">Cross-Audio</td></tr><tr><td>1</td><td>6.299</td><td>56.732</td><td>6.543</td><td>56.814</td><td>6.592</td><td>56.909</td></tr><tr><td>2</td><td>7.007</td><td>57.102</td><td>7.136</td><td>57.185</td><td>7.126</td><td>57.200</td></tr><tr><td>4</td><td>7.483</td><td>57.277</td><td>7.558</td><td>57.276</td><td>7.569</td><td>57.272</td></tr><tr><td>8</td><td>7.419</td><td>57.251</td><td>7.431</td><td>57.239</td><td>7.488</td><td>57.248</td></tr><tr><td>16</td><td>6.978</td><td>57.268</td><td>7.301</td><td>57.220</td><td>7.244</td><td>57.262</td></tr><tr><td>32</td><td>6.770</td><td>57.240</td><td>7.099</td><td>57.256</td><td>7.067</td><td>57.244</td></tr><tr><td> $\operatorname { A v g } .$ </td><td>6.993</td><td>57.145</td><td>7.178</td><td>57.165</td><td>7.181</td><td>57.189</td></tr></table>

![](images/b4a9c50c191059359d76525be12ca4a025c061a1f5e29b783438917c5024544f.jpg)  
Fig. 11: Causal knockout experiments guided by Temporal Routing Divergence $( \mathcal { D } _ { T R } )$ $\mathcal { D } _ { T R ^ { \mathrm { - } } } \mathrm { g }$ uided channel ablations reveal how DTM adaptively disentangles rigid highfrequency details from flexible low-frequency transitions.

$$
\mathcal { D } _ { T R } ( c ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } D _ { \mathrm { K L } } \big ( \mathbf { S } _ { c } ( t ) \| \bar { \mathbf { S } } _ { c } \big ) .\tag{11}
$$

This metric separates the channels into two regimes. Channels with a high $\mathcal { D } _ { T R }$ actively modulate their reference selection across frames to blend temporal features dynamically, whereas channels with a low $\mathcal { D } _ { T R }$ firmly anchor specific references and converge toward a stable weight distribution over time.

To test whether these two regimes carry distinct functional roles, we sort the channels by their $\dot { \mathcal { D } } _ { T R }$ values and perform knockout experiments, setting the ablated channels of $\mathbf { F } _ { m i x }$ to zero. As shown in Fig. 11, the two ablations produce strikingly orthogonal effects. Removing 50% of the $\mathrm { h i g h } \lnot D _ { T R }$ channels leaves rigid high-frequency details such as teeth perfectly sharp, but the flexible skin regions suffer severe spatial noise and rugged artifacts. Removing 50% of the $\mathrm { l o w } { - } { \mathcal { D } } _ { T R }$ channels has the opposite effect: the facial skin becomes overly smooth while rigid details such as teeth melt into a blurred, averaged state. These observations validate a bipartite division of labor within the DTM. For low-frequency content such as continuous skin deformation, the $\mathrm { h i g h } { - \mathcal { D } _ { T R } }$ channels dynamically modulate their weights to provide the adaptive mixing that static convolutions cannot achieve; for high-frequency content such as teeth, the $\mathrm { l o w } { - } { \mathcal { D } } _ { T R }$ channels supply a constant, stable reference that prevents homogenization. The DTM is therefore not an unconstrained router but one that adaptively anchors essential high-frequency features while remaining flexible elsewhere.

## 5.2 Temporal Dynamics of Channel Weights

The routing analysis characterizes individual channels; we next study how the DTM allocates weight across references over time, and how this allocation depends on the reference pool. We define $\mathbf { S } _ { \mathrm { m e a n } } ^ { n } ( t )$ as the channel-averaged routing weight assigned to the n-th reference at frame t, which summarizes its overall attention trajectory:

$$
\mathbf { S } _ { \mathrm { m e a n } } ^ { n } ( t ) = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \mathbf { S } _ { c } ^ { n } ( t ) ,\tag{12}
$$

where $C$ is the total number of channels. We visualize these trajectories during cross-audio inference, comparing STAR Sampling against random sampling under varying reference budgets $\bar { N ~ } \in ~ \{ 2 , 3 , 4 , 5 \}$ . As Fig. 12 shows, under STAR Sampling the trajectories of different references fluctuate only mildly within distinctly separated, stable tiers, whereas under random sampling they intertwine chaotically and swing over a wide range.

To quantify this stability, we introduce the Intratrajectory Temporal Volatility (ITV), the reference-averaged temporal standard deviation of these attention curves:

$$
\mathrm { I T V } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \sqrt { \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left( \mathbf { S } _ { \mathrm { m e a n } } ^ { n } ( t ) - \mu ^ { n } \right) ^ { 2 } } ,\tag{13}
$$

where $T$ is the number of frames and $\mu ^ { n }$ is the temporal mean of $\mathbf { S } _ { \mathrm { m e a n } } ^ { n } ( t )$ . A larger ITV indicates that the model’s attention among the references shifts more violently over time. The annotated values in Fig. 12 reveal two consistent trends: ITV decreases as the number of references grows, and STAR Sampling maintains a strictly lower ITV than random sampling across every configuration. When the reference pool offers sufficient topological diversity, as under STAR Sampling, each anchor can stably retain its proportional contribution; the disordered weight shifting induced by random sampling instead disrupts temporal consistency, which plausibly underlies the synchronization degradation observed earlier at the feature level.

![](images/dc6b48fc69d38616c334990062fcc12be2ea17bf23bcfc12051b925d60799a4d.jpg)  
Fig. 12: Temporal trajectories of routing weights on a single video. Visualization of the channel-averaged attention weights allocated by DTM to individual references under STAR and random sampling across varying reference counts.

## 5.3 Synergy between the DTM and STAR Sampling

Taken together, the spatial routing and temporal volatility analyses expose a natural synergy between the two designs. The knockout results show that the DTM prefers a stable selection strategy to preserve high-frequency detail, and STAR Sampling provides exactly the conditions under which this strategy can operate. By maximizing the topological distance among selected references, STAR Sampling yields an unambiguous candidate pool, making it easy for the DTM to identify and lock onto the optimal highfrequency textures. This clarity manifests directly as the low ITV and the well-separated trajectory tiers seen under STAR Sampling in Fig. 12. Random sampling, in contrast, returns topologically redundant or suboptimal references; faced with such candidates, the DTM cannot sustain its stable selection and is driven into the chaotic frame-to-frame weight shifting reflected by a high ITV, which ultimately compromises clarity and synchronization. STAR Sampling thus supplies the structural stability that lets the routing mechanism of the DTM reach its full potential.

## 6 CONCLUSION

In this paper, we present EfficientSync, a novel deformationbased lip-sync framework designed to generate high-fidelity dubbed videos while maintaining a low memory footprint, low computational complexity, and high inference speeds. To preserve authentic facial details and avoid texture degradation, we introduce the Dynamic Texture Mixer for context-aware reference feature aggregation. Furthermore, we propose the Spatio-Temporal Shifted Adaptive Masking strategy to effectively resolve the background preservation dilemma frequently overlooked by previous mask-based methods, thereby eliminating boundary artifacts. Additionally, we introduce the STAR Sampling strategy, which optimizes reference image selection to enhance final generation quality without incurring any additional inference latency. By achieving robust identity preservation and a real-time inference speed of 166 FPS, we hope these improvements will facilitate the broader real-world deployment of computationally efficient lip-sync models.

## ACKNOWLEDGEMENT

This work is supported in by the Guangdong Basic and Applied Basic Research Foundation (2025A1515010124), and in part by the National Key Research and Development Program of China (2024YFF1206600), the National Natural Science Foundation of China(62325204).

## REFERENCES

[1] K. Prajwal, R. Mukhopadhyay, V. P. Namboodiri, and C. Jawahar, “A lip sync expert is all you need for speech to lip generation in the wild,” in ACMMM, 2020.

[2] Z. Zhang, Z. Hu, W. Deng, C. Fan, T. Lv, and Y. Ding, “Dinet: Deformation inpainting network for realistic face visually dubbing on high resolution video,” in AAAI, 2023.

[3] C. Li, C. Zhang, W. Xu, J. Xie, W. Feng, B. Peng, and W. Xing, “Latentsync: Audio conditioned latent diffusion models for lip sync,” arXiv preprint arXiv:2412.09262, 2024.

[4] F. Yin, Y. Zhang, X. Cun, M. Cao, Y. Fan, X. Wang, Q. Bai, B. Wu, J. Wang, and Y. Yang, “Styleheat: One-shot high-resolution editable talking face generation via pre-trained stylegan,” in ECCV, 2022.

[5] T. Karras, S. Laine, and T. Aila, “A style-based generator architecture for generative adversarial networks,” in CVPR, 2019.

[6] B. Liang, Y. Pan, Z. Guo, H. Zhou, Z. Hong, X. Han, J. Han, J. Liu, E. Ding, and J. Wang, “Expressive talking head generation with granular audio-visual control,” in CVPR, 2022.

[7] S. Shen, W. Zhao, Z. Meng, W. Li, Z. Zhu, J. Zhou, and J. Lu, “Difftalk: Crafting diffusion models for generalized audio-driven portraits animation,” in CVPR, 2023.

[8] S. Mukhopadhyay, S. Suri, R. T. Gadde, and A. Shrivastava, “Diff2lip: Audio conditioned diffusion models for lipsynchronization,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 5292–5302, 2024.

[9] J. Tang, K. Wang, H. Zhou, X. Chen, D. He, T. Hu, J. Liu, Z. Liu, G. Zeng, and J. Wang, “Real-time neural radiance talking portrait synthesis via audio-spatial decomposition,” IJCV, 2025.

[10] J. Li, J. Zhang, X. Bai, J. Zhou, and L. Gu, “Efficient region-aware neural radiance fields for high-fidelity talking portrait synthesis,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7568–7578, 2023.

[11] F.-T. Hong and D. Xu, “Implicit identity representation conditioned memory compensation network for talking head video generation,” in Proceedings of the IEEE/CVF international conference on computer vision, pp. 23062–23072, 2023.

[12] J. Guan, Z. Zhang, H. Zhou, T. Hu, K. Wang, D. He, H. Feng, J. Liu, E. Ding, Z. Liu, et al., “Stylesync: High-fidelity generalized and personalized lip sync in style-based generator,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1505–1515, 2023.

[13] R. Liu, Q. Lin, Y. Liu, L. Lin, Y. Zhu, Y. Li, C. Xian, and F.-T. Hong, “Identity-preserving video dubbing using motion warping,” IJCV, p. 226, 2026.

[14] Y. Ma, S. Wang, Z. Hu, C. Fan, T. Lv, Y. Ding, Z. Deng, and X. Yu, “Styletalk: One-shot talking head generation with controllable speaking styles,” in AAAI, 2023.

[15] V. Blanz, T. Vetter, and A. Rockwood, “A morphable model for the synthesis of 3d faces,” ACM SIGGRAPH, pp. 187–194, 2002.

[16] Y. Zhou, X. Han, E. Shechtman, J. Echevarria, E. Kalogerakis, and D. Li, “Makelttalk: speaker-aware talking-head animation,” TOG, 2020.

[17] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106, 2021.

[18] Y. Guo, K. Chen, S. Liang, Y.-J. Liu, H. Bao, and J. Zhang, “Ad-nerf: Audio driven neural radiance fields for talking head synthesis,” in ICCV, 2021.

[19] S. Shen, W. Li, Z. Zhu, Y. Duan, J. Zhou, and J. Lu, “Learning dynamic facial radiance fields for few-shot talking head synthesis,” in ECCV, 2022.

[20] L. Tian, Q. Wang, B. Zhang, and L. Bo, “Emo: Emote portrait alive generating expressive portrait videos with audio2video diffusion model under weak conditions,” in European Conference on Computer Vision, pp. 244–260, Springer, 2024.

[21] M. Xu, H. Li, Q. Su, H. Shang, L. Zhang, C. Liu, J. Wang, L. Van Gool, Y. Yao, and S. Zhu, “Hallo: Hierarchical audio-driven visual synthesis for portrait image animation,” arXiv preprint arXiv:2406.08801, 2024.

[22] H. Wei, Z. Yang, and Z. Wang, “Aniportrait: Audio-driven synthesis of photorealistic portrait animation,” arXiv preprint arXiv:2403.17694, 2024.

[23] W. Zhong, C. Fang, Y. Cai, P. Wei, G. Zhao, L. Lin, and G. Li, “Identity-preserving talking face generation with landmark and appearance priors,” in CVPR, 2023.

[24] K. Cheng, X. Cun, Y. Zhang, M. Xia, F. Yin, M. Zhu, X. Wang, J. Wang, and N. Wang, “Videoretalking: Audio-based lip synchronization for talking head video editing in the wild,” in SIGGRAPH Asia 2022 Conference Papers, pp. 1–9, 2022.

[25] Y. Zhang, L. Minhao, Z. Chen, B. Wu, C. Zhan, Y. He, J. HUANG, W. Zhou, et al., “Musetalk: Real-time high quality lip synchronization with latent space inpainting,” 2024.

[26] A. Bigata, R. Mira, S. Bounareli, M. Stypułkowski, K. Vougioukas, S. Petridis, and M. Pantic, “Keysync: A robust approach for leakage-free lip synchronization in high resolution,” arXiv preprint arXiv:2505.00497.

[27] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in ICML, pp. 28492–28518, 2023.

[28] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7132–7141, 2018.

[29] J. H. Lim and J. C. Ye, “Geometric gan,” arXiv preprint arXiv:1705.02894, 2017.

[30] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” arXiv preprint arXiv:1409.1556, 2014.

[31] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in CVPR, 2018.

[32] Z. Zhang, L. Li, Y. Ding, and C. Fan, “Flow-guided one-shot talking face generation with a high-resolution audio-visual dataset,” in CVPR, 2021.

[33] L. Xie, X. Wang, H. Zhang, C. Dong, and Y. Shan, “Vfhq: A highquality dataset and benchmark for video face super-resolution,” in CVPR, 2022.

[34] C. Lugaresi, J. Tang, H. Nash, C. McClanahan, E. Uboweja, M. Hays, F. Zhang, C.-L. Chang, M. G. Yong, J. Lee, et al., “Me-

diapipe: A framework for building perception pipelines,” arXiv preprint arXiv:1906.08172, 2019.

[35] Z. Wang, “Image quality assessment: Form error visibility to structural similarity,” TIP, vol. 13, no. 4, pp. 604–606, 2004.

[36] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” Advances in neural information processing systems, 2017.

[37] T. Unterthiner, S. Van Steenkiste, K. Kurach, R. Marinier, M. Michalski, and S. Gelly, “Towards accurate generative models of video: A new metric & challenges,” arXiv preprint arXiv:1812.01717, 2018.

[38] J. L. Pech-Pacheco, G. Cristóbal, J. Chamorro-Martinez, and J. Fernández-Valdivia, “Diatom autofocusing in brightfield microscopy: a comparative study,” in Proceedings 15th International Conference on Pattern Recognition. ICPR-2000, 2000.

[39] J. S. Chung and A. Zisserman, “Out of time: automated lip sync in the wild,” in Asian conference on computer vision, 2016.

![](images/2f3a7009092da61a0713cdeb407a2156650e6b6ea571afc40eb3856a7abd7f0e.jpg)  
Fa-Ting Hong did his PhD at Hong Kong University of Science and Technology. He received his M.E. from Sun Yat-sen University and B.E. from South China University of Technology. His research interest lies in deep learning for avatar generation, human motion and audiovisual learning.

![](images/fec2b9dc5096d546892f942e0740e5a4960bf6467d8fbe3ded418f0d6b1a5aec.jpg)

Runzhen Liu is a master’s student in the School of Computer Science and Engineering, South China University of Technology. He received his B.E. from South China University of Technology. His research interest lies in deep learning for image and video generation.

![](images/fe31b110c96bfb7a4efc40011ad72b97fd491df4aff864df904e60c83b80d4e9.jpg)

Luchuan Song is a Ph.D. student in the Department of Computer Science at University of Rochester. He received his M.E. and B.E. from Department of Electronic Engineering and Information Science at University of Science and Technology of China. His research interest lies in human related topic (e.g. face animation and toonfication, 3D face reconstruction, deepfake detection e.t.c).

![](images/49b71d3e8dd69d4e9437dff3ee1bfbaa2ff858d20cf5e3099bb57ce4bc7d317b.jpg)

Hongmin Cai (Senior Member, IEEE) received the B.S. and M.S. degrees in mathematics from the Harbin Institute of Technology, Harbin, China, in 2001 and 2003, respectively, and the Ph.D. degree in applied mathematics from the University of Hong Kong, Hong Kong, in 2007. He was a visiting Professor with Kyoto University, Kyoto, Japan, and with Harvard University, Cambridge, MA, USA. He is currently the Executive Dean with the School of Future Technology, and a Joint Appointment Professor with the School

of Computer Science and Engineering, South China University of Technology, Guangzhou, China. His research interests include digital twins in life science, intelligent computing of biomedical data, and machine learning.

![](images/e68532e76b888ea2e48b768105f92789ffc1efa171c0e90b681ded7428eb858d.jpg)

Chuhua Xian is an Associate Professor with the School of Computer Science and Engineering, South China University of Technology. He received Ph.D. degree in Computer Science and Technology from State Key Lab. of CAD & CG of Zhejiang University in March 2012. From November 2013 to May 2014, and from September 2015 to April 2016, he was a visiting scholar of the research group of Prof. Charlie C. L. Wang in the Chinese University of Hong Kong. His current research interests are computer graphics, geometry computing, rendering, image processing, and Computer-Aided Design.