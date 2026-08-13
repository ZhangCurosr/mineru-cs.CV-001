# From Synthesis to Removal: Physics-Grounded Reflection Simulation and Difusion-Based Video Dereflection

Zepeng Wang<sup>\*</sup>, Jiagao Hu<sup>\*</sup>, Fuhao Li<sup>\*</sup>, Yuxuan Chen, Fei Wang, Daiguo Zhou

MiLM Plus, Xiaomi Inc.

<sup>\*</sup>Equal contribution.

## Abstract

Videos captured through glass often contain reflections that degrade visual quality and interfere with downstream vision tasks. Although single-image reflection removal has been extensively studied, video reflection removal remains largely underexplored due to the lack of paired video data, temporally coherent removal models, and dedicated evaluation benchmarks. We present a closed-loop framework that unifies physics-grounded reflection simulation, difusion-based video dereflection, and benchmark evaluation. Our S2R-Synthesis pipeline generates paired reflected and reflection-free videos by performing physics-grounded augmentation in the structure space and rendering realistic reflected videos with a trained video difusion renderer; the augmentation models key glass-related efects including roughness-induced blur, thickness-induced ghosting, and reflectance variation. Based on the synthesized data, we introduce S2R-Removal, the first difusion-based video reflection removal model, which adapts a pretrained video difusion prior through reflection-aware latent adaptation and one-step pixel-geometric refinement, recovering the clean transmission in a single denoising step. We further build S2R-Bench, the first benchmark for video reflection removal, supporting both full-reference evaluation and real-world human perceptual assessment. Experiments on S2R-Bench and multiple public image benchmarks demonstrate state-of-the-art performance and faster inference than even nondifusion baselines, and validate the efectiveness of S2R-Synthesis. Project page: https://codingwzp.github.io/VideoDereflection\_S2R.

## 1 Introduction

Videos captured through glass are ubiquitous, yet the resulting reflections often obscure the underlying transmission scene, degrade perceptual quality, and interfere with downstream vision systems [1, 2]. Removing such reflections is therefore important for video restoration, computational photography, and practical visual perception.

Despite extensive progress in single-image reflection removal [3–6], video reflection removal remains significantly underexplored. Videos introduce additional challenges: reflections vary over time, move independently from the transmission layer, and interact with camera motion.

![](images/08e352624579d9e309651078e5a8f0fd38a5c82aca04e6ccac49471872066b2d.jpg)  
Figure 1 Overview of the reflection synthesis and removal pipeline. Left: Controllable synthesis across glass roughness, reflectance, and thickness. Right: Qualitative removal results on in-thewild videos.

Applying image methods frame by frame is a straightforward solution but often produces flickering, inconsistent removal strength, and unstable background details due to the lack of temporal modeling [7]. A successful video dereflection method should instead suppress reflections while preserving the appearance, structure, and temporal coherence of the transmission video.

Achieving these goals, however, is bottlenecked by paired training data: obtaining a reflected video and its perfectly aligned reflection-free counterpart is extremely dificult, as camera motion, illumination, and dynamic scene content must remain consistent before and after reflection removal. Existing synthesis strategies each fall short of providing scalable, realistic, and controllable reflections: RGB-space blending [4, 8] yields aligned pairs but overly simplistic reflections that miss glass-dependent efects; frame-wise image difusion [9] achieves realistic appearance per frame but introduces temporal flicker and ofers no control over glass parameters; and direct video difusion models, despite their temporal modeling capability, exhibit extremely low success rates when directly prompted to synthesize reflections on clean videos, often failing to produce visible reflections or corrupting the underlying scene content.

Beyond the data bottleneck, video reflection removal is further hampered by the absence of temporally coherent models and task-specific benchmarks. We therefore approach it as a closedloop synthesis-to-removal problem—jointly tackling scalable paired video data, temporally coherent video dereflection models, and benchmarks for both controlled and real-world evaluation—rather than as an isolated model design problem. To this end, we present a physics-grounded framework that unifies controllable reflection synthesis and reflection-aware video removal, complemented by a purpose-built benchmark for evaluation; representative results from both are illustrated in Figure 1.

First, we propose S2R-Synthesis, a paired video reflection synthesis pipeline that achieves both realism and controllable diversity. S2R-Synthesis decouples reflection structure from appearance: a video difusion renderer—distilled from a frozen image difusion model for realism and inheriting temporal coherence from its video prior—renders photorealistic reflected videos from the clean transmission and a structured reflection condition, while Physics-Grounded Augmentation (PGA)

transforms this condition according to glass optical parameters, providing control over major glass-related efects including surface roughness, glass reflectance, and thickness.

Second, we introduce S2R-Removal, a difusion-based video reflection removal model trained on the synthesized paired data. Since reflections attenuate rather than occlude the transmission layer, dereflection is fundamentally a restoration task rather than free generation. Accordingly, S2R-Removal leverages the generative prior of video difusion models [10] through a two-stage training process: Stage I performs reflection-aware latent adaptation with reflection-intensity supervision to localize and remove reflections, and Stage II further refines the model with pixelgeometric losses (reconstruction, structural, and depth consistency), training it to recover the clean transmission in a single denoising step while preserving the underlying scene. This one-step design further makes S2R-Removal substantially more eficient than even non-difusion baselines.

Finally, we introduce S2R-Bench, the first benchmark for video reflection removal. S2R-Bench contains two complementary subsets: S2R-Ref provides paired videos with clean ground truth for full-reference evaluation, while S2R-Real contains in-the-wild reflection videos for human perceptual assessment. Together, they support evaluation of reconstruction fidelity, temporal consistency, reflection removal quality, and transmission preservation.

Our contributions are summarized as follows:

• We present, to the best of our knowledge, the first closed-loop framework for video reflection removal, integrating physics-grounded synthesis, difusion-based removal, and benchmark evaluation.

• We propose S2R-Synthesis, a paired video reflection synthesis pipeline that performs physicsgrounded augmentation in the structure space and uses a trained video renderer to generate controllable reflected videos.

• We introduce S2R-Removal, the first difusion-based video reflection removal model which removes reflections in one step and runs faster than non-difusion baselines, together with S2R-Bench, the first benchmark for this task, supporting both full-reference objective evaluation and real-world perceptual assessment.

Extensive experiments on S2R-Bench and multiple public image reflection removal benchmarks demonstrate state-of-the-art performance. We further validate the efectiveness of S2R-Synthesis through data ablations.

## 2 Related Work

## 2.1 Single-Image and Video Reflection Removal

Single-image reflection removal has progressed from handcrafted priors and multi-image constraints [11, 12] to deep models with bidirectional layer estimation and perceptual supervision [3, 13–15], and further to robustness-oriented designs using polarization, location awareness, dual-stream interaction, and in-the-wild modeling [4, 16–19]. However, all such methods process frames independently and produce flickering when applied to video, and video reflection removal has only been touched by spatio-temporal optimization [7] and user-guided decomposition [20], leaving scalable learning-based priors unexplored.

## 2.2 Reflection Data Synthesis and Benchmarks

Paired reflection data are hard to collect, so existing image datasets rely on controlled capture, realworld collection, or synthetic composition under a linear layer formation model [3–5, 8, 16, 17, 21–

![](images/5624abdfd8e7f4704cbc936b3818578b6d752fceb3d012258b6370cf69f54daa.jpg)  
Figure 2 Overview of S2R-Synthesis. Left: limitations of existing strategies motivate our S2R-Synthesis. Stage A trains a structure-guided reflection renderer from FLUX-generated pseudo reflection videos; Stage B fuses two clean videos’ conditions through Physics-Grounded Augmentation (PGA) and renders paired reflected videos. Right: representative PGA operations.

23], with recent variants adding non-linear alpha masks [8], RAW-domain modeling [24], and physically based rendering [25]. Video reflection additionally requires temporally coherent behavior and aligned ground truth, which no prior benchmark provides; our S2R-Bench fills this gap with the first paired-video benchmark and controllable synthesis pipeline.

## 2.3 Difusion Priors for Reflection Removal

Difusion priors have been applied to single-image dereflection via visual prompts [26], selfsupervised separation [27], one-step removal with diversified data [5], and latent-space prior modulation [6, 28], but remain confined to the image domain. S2R-Removal extends difusionbased dereflection to video for the first time.

## 3 Physics-Grounded Reflection Simulation

We build on the standard reflection formation model [4, 8, 11, 15], which represents the observed image as a linear mixture of transmission and reflection layers:

$$
I = \alpha _ { t } T + \alpha _ { r } R ,\tag{1}
$$

where � and � denote the transmission and reflection layers, and $\alpha _ { t } , \alpha _ { r }$ control their relative contributions. Direct RGB-space blending under this model produces overly simplistic reflections and fails to capture glass-dependent efects [25] such as roughness-induced blur, thickness-induced ghosting, and reflectance variation.

Overview. We propose a two-stage pipeline that lifts reflection synthesis from RGB-space blending to structure-space conditioning. As shown in Figure 2, Stage A trains a structure-guided reflection renderer G, and Stage B uses $\mathcal { G }$ together with a Physics-Grounded Augmentation (PGA) module to synthesize large-scale paired data:

$$
E _ { F } = \mathcal { F } \big ( E _ { T } , \mathcal { A } ( E _ { R } ; \theta _ { g } ) \big ) , \quad I = \mathcal { G } ( T , E _ { F } ) ,\tag{2}
$$

where $E _ { T }$ and $E _ { R }$ are the lineart conditions [29] of the transmission and reflection source videos, A denotes the PGA module with glass control parameters $\theta _ { g } , \mathcal { F }$ fuses the two linearts, and $\mathcal { G }$ renders a photorealistic reflection video from the clean video � and the composed structural condition $E _ { F }$

## 3.1 Video Reflection Renderer Training

Given a collection of reflection-free videos selected by a VLM, we apply FLUX frame-by-frame with reflection-injection prompts to obtain pseudo reflection videos. Because FLUX operates independently per frame, the resulting clips may exhibit temporal flicker, so we use them only as pseudo supervision. Details are provided in Section A.1 in the appendix.

Concretely, for each clean video � and its pseudo reflection video $\tilde { M } ,$ , we extract a lineart condition from $\tilde { M } \cdot$ —capturing structural layout while suppressing unstable appearance details—and train the renderer $\mathcal { G }$ (built on Wan2.1 [10]) to reconstruct �<sup>˜</sup> from � and this lineart, supervised by a difusion loss. We use a three-channel representation, consistent with the RGB conditioning format of the pretrained backbone, which yields more naturally colored reflections than a single-channel counterpart.

## 3.2 Paired Reflection Video Generation

Given a transmission video � and an independent reflection source video �, we extract their lineart conditions $E _ { T }$ and $E _ { R }$ . The reflection lineart $E _ { R }$ is transformed by PGA, fused with $E _ { T }$ , and fed together with � into $\mathcal { G }$ to produce a reflection video � (Eq. (2)). The pair $( I , T )$ serves as training data for the reflection removal model. Compared with RGB-space blending (Eq. (1)), compositing in lineart space and delegating appearance synthesis to the learned renderer yields more natural reflections with exact paired supervision. Dataset construction details are provided in Section B.1 in the appendix.

## 3.3 Physics-Grounded Augmentation

Real glass reflections vary with surface roughness, glass thickness, viewing angle, spatial coverage, and temporal behavior. Rather than running full light-transport simulation, PGA instantiates the dominant visual efects of these factors as controllable operations on the reflection lineart $E _ { R } { \mathrm { : } }$ , governed by clip-level parameters $\theta _ { g } = \{ \sigma , \Delta , w , \gamma , \beta , p \}$ shared across frames for temporal coherence.

Augmentation Primitives. PGA composes six operations, each tied to a specific glass optical property (full derivations are in Section C and parameter ranges are provided in Table 7 in the appendix): Roughness approximates the far-field angular spread of microfacet (GGX) scattering [30, 31] on rough glass as a Gaussian convolution of $E _ { R }$ with width $\sigma ;$ Thickness reproduces multi-interface ghosting under paraxial Snell’s law [32], adding a shifted copy $\boldsymbol { w } \cdot \mathcal { W } _ { \Delta } ( E _ { R } )$ with ofset Δ following the lateral displacement $\delta \approx d \theta _ { i } ( 1 - 1 / n )$ ; Reflectance instantiates the Fresnel decomposition [32] $I _ { \mathrm { r e f l } } = F \cdot L _ { \mathrm { e n v } } + L _ { \mathrm { a m b } }$ under spatially uniform incident angle as an afine modulation $\gamma \cdot E _ { R } + \beta ;$ ; Partial captures partial glass coverage by masking $E _ { R }$ with a spatial mask $p ;$ Static captures temporally stable reflections from stationary sources by freezing a randomly selected frame of $E _ { R }$ across the clip; Planar serves as the default mode for flat-glass reflection, directly fusing $E _ { T }$ and $E _ { R }$

## 4 Difusion-Based Video Dereflection

Given paired videos from our synthesis pipeline, we train a difusion-based video reflection removal model. We formulate the task as conditional video generation: conditioned on the reflected video $I ,$ the model predicts the clean transmission video �. Built on a video inpainting paradigm using Wan2.1, our adaptation follows two principles: the model should be reflection-aware (localizing and estimating reflection strength) and transmission-preserving (removing reflections without altering scene content or geometry). We address these with a two-stage training strategy, illustrated in Figure 3.

![](images/57e4c671f355395d16afa128b7e24f69430c8871d3069453ab0d774a996943bd.jpg)  
Figure 3 Overview of S2R-Removal. Stage I learns reflection-aware latent adaptation via residualderived intensity supervision from $( I , T ) ;$ Stage II applies one-step pixel-geometric refinement with reconstruction, structural, and depth consistency losses.

## 4.1 Stage I: Reflection-Aware Latent Adaptation

The first stage adapts the pretrained video difusion backbone to the reflected-to-clean mapping in latent space via LoRA fine-tuning. Let $z _ { T } = \mathcal { E } ( T )$ and $z _ { I } = \mathcal { E } ( I )$ denote the latent representations of the clean transmission video and the reflected input video, respectively. We add noise to the clean target latent ${ \mathcal { Z } } _ { T }$ and use the reflected input latent $\mathfrak { z } _ { I }$ as the condition. Specifically, we sample $t \sim \mathcal { U } ( 1 , N )$ and $\epsilon \sim { \cal N } ( 0 , { \bf I } )$ , and obtain $z _ { t } = \alpha _ { t } z _ { T } + \sigma _ { t } \epsilon$ . Following the prediction parameterization of the pretrained backbone, the DiT predicts the difusion target $u _ { t }$ by the following difusion loss:

$$
\mathcal { L } _ { \mathrm { d i f f } } = \mathbb { E } _ { t , \epsilon } \left[ \Vert f _ { \theta } ( z _ { t } , t , z _ { I } ) - u _ { t } \Vert _ { 2 } ^ { 2 } \right] .\tag{3}
$$

Our paired synthetic data provide a natural supervision signal for this. We derive a continuous reflection-intensity map from the residual between the reflected input and the clean target, $M _ { \mathrm { g t } } = \left| I - T \right|$ , which encodes both the location and strength of reflection corruption. To inject this signal into the backbone without disturbing its pretrained denoising behavior, we attach a lightweight, zero-initialized intensity head to the DiT features alongside the main denoising head. The head is supervised by an $L _ { 1 }$ loss in latent space:

$$
\mathcal { L } _ { \mathrm { i n t } } = \left. \hat { M } - \mathcal { E } ( M _ { \mathrm { g t } } ) \right. _ { 1 } ,\tag{4}
$$

where $\hat { M }$ is the predicted intensity map in the latent space and E is the VAE encoder. The Stage I objective combines both losses:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s t a g e 1 } } = \lambda _ { \mathrm { d i f f } } \mathcal { L } _ { \mathrm { d i f f } } + \lambda _ { \mathrm { i n t } } \mathcal { L } _ { \mathrm { i n t } } , } \end{array}\tag{5}
$$

with $\lambda _ { \mathrm { d i f f } } = 1 . 0 \mathrm { a n d } \lambda _ { \mathrm { i n t } } = 0 . 1$

Through this stage, the model acquires a strong prior on reflected-to-clean translation, and—as a by-product of the latent difusion training—acquires a meaningful one-step denoising capability: a single forward pass already produces a rough but coherent clean estimate, consistent with the observation in [6]. This emergent one-step capability is the key enabler of Stage II.

## 4.2 Stage II: One-Step Pixel-Geometric Refinement

Latent difusion training does not directly constrain the decoded output, and the model’s generative freedom may alter texture or local structure in ways that are undesirable for reflection removal. We therefore introduce a second stage that anchors the output to the clean target in both pixel and geometry space.

Crucially, this stage exploits the one-step denoising capability acquired in Stage I. Rather than running expensive multi-step sampling, we start from a noisy latent $ { \boldsymbol { z } } _ { \tau } \sim  { \mathcal { N } } ( 0 , \mathbf { I } )$ at the largest noise level $\tau ,$ and perform a single deterministic denoising update conditioned on the reflected input latent $\mathfrak { z } _ { I }$ :

$$
\hat { z } _ { 0 } = \mathcal { D } _ { \theta } ^ { ( 1 ) } ( z _ { \tau } , \tau , z _ { I } ) ,\tag{6}
$$

where $\mathcal { D } _ { \theta } ^ { ( 1 ) }$ denotes the one-step conversion from the model prediction to the clean latent estimate under the pretrained backbone’s sampling parameterization. The predicted latent $\hat { z } _ { 0 }$ is decoded by the frozen VAE decoder to obtain the pixel-space output $\hat { T }$

Pixel-Space Losses. We apply an $L _ { 1 }$ reconstruction loss and an SSIM loss to constrain color fidelity and local structural consistency:

$$
\mathcal { L } _ { \mathrm { r e c } } = \Vert \hat { T } - T \Vert _ { 1 } , \qquad \mathcal { L } _ { \mathrm { s s i m } } = 1 - \mathrm { S S I M } ( \hat { T } , T ) .\tag{7}
$$

Reflection-Aware Depth Consistency Loss. To further constrain scene geometry without overpenalizing style or texture, we introduce a depth consistency loss computed by a frozen LeReS [33] depth estimator:

$$
D _ { \hat { T } } = \Phi _ { \mathrm { d e p } } ( \hat { T } ) , \qquad D _ { T } = \Phi _ { \mathrm { d e p } } ( T ) .\tag{8}
$$

Gradients are propagated only through $D _ { \hat { T } } ; D _ { T }$ is treated as a fixed target.

We derive a binary reflection mask $M = \mathbb { I } ( | I - T | > \tau _ { m } )$ with $\tau _ { m } = 1 2$ to focus geometric supervision on reflection-corrupted regions, where hallucination and structural distortion are most likely:

$$
\mathcal { L } _ { \mathrm { d e p t h } } = \left\{ \begin{array} { l l } { \displaystyle \frac { \sum M | D _ { \hat { T } } - D _ { T } | } { \sum M } , } & { \sum M > 0 , } \\ { \displaystyle \frac { 1 } { | \Omega | } \sum _ { \Omega } | D _ { \hat { T } } - D _ { T } | , } & { \sum M = 0 , } \end{array} \right.\tag{9}
$$

where Ω denotes the full pixel domain.

The Stage II objective is:

$$
\mathcal { L } _ { \mathrm { s t a g e 2 } } = \lambda _ { \mathrm { r e c } } \mathcal { L } _ { \mathrm { r e c } } + \lambda _ { \mathrm { s s i m } } \mathcal { L } _ { \mathrm { s s i m } } + \lambda _ { \mathrm { d e p } } \mathcal { L } _ { \mathrm { d e p t h } } ,\tag{10}
$$

with $\lambda _ { \mathrm { r e c } } = 1 . 0 , \lambda _ { \mathrm { s s i m } } = 0 . 2 .$ , and $\lambda _ { \mathrm { d e p } } = 0 . 5$

The two stages are complementary: Stage I teaches the model what to remove through reflectionintensity supervision in latent space, while Stage II teaches the model what to preserve through pixel and geometric constraints in image space—made eficient by the one-step denoising capability that Stage I instils.

Table 1 Comparison with state-of-the-art dereflection methods on video and image benchmarks. S2R-Ref reports full-reference metrics + TC; S2R-Real reports normalized human scores + TC; DAI is excluded from S2R-Ref to avoid DRR data leakage. Per-frame inference time is measured on an 81-frame video; our one-step model is ∼1.67× faster than the next-best method.
<table><tr><td>Method</td><td colspan="6">Video Bench</td><td colspan="6">Image Bench</td><td colspan="2"></td><td>Efficiency</td></tr><tr><td></td><td colspan="3">S2R-Ref(60)</td><td colspan="3">S2R-Real(50)</td><td colspan="2">Real(20)</td><td colspan="2">SIR2(454)</td><td colspan="2">Nature(20)</td><td colspan="2">Average</td><td>832 × 480</td></tr><tr><td></td><td>PSNR↑</td><td>SSIM↑</td><td>TC↑</td><td>Removal↑</td><td>Preserv.↑</td><td>TC↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td></td><td>PSNR↑ SSIM↑</td><td>ms/frame↓</td></tr><tr><td>DSRNet (ICCV&#x27;23)</td><td>23.78</td><td>0.857</td><td>0.9546</td><td>0.163</td><td>0.785</td><td>0.9790</td><td>23.91</td><td>0.818</td><td>25.71</td><td>0.906</td><td>25.22</td><td>0.832</td><td>25.62</td><td>0.899</td><td>329.23</td></tr><tr><td>DSIT (NeurIPS&#x27;24)</td><td>24.06</td><td>0.870</td><td>0.9585</td><td>0.297</td><td>0.853</td><td>0.9781</td><td>25.22</td><td>0.836</td><td>26.43</td><td>0.911</td><td>26.77</td><td>0.847</td><td>26.40</td><td>0.905</td><td>241.05</td></tr><tr><td>RDNet (CVPR&#x27;25)</td><td>24.71</td><td>0.870</td><td>0.9564</td><td>0.175</td><td>0.867</td><td>0.9797</td><td>25.71</td><td>0.850</td><td>26.69</td><td>0.908</td><td>26.31</td><td>0.846</td><td>26.63</td><td>0.903</td><td>145.13</td></tr><tr><td>DAI (AAAI&#x27;26)</td><td></td><td></td><td></td><td>0.332</td><td>0.871</td><td>0.9777</td><td>25.21</td><td>0.841</td><td>27.47</td><td>0.919</td><td>26.81</td><td>0.843</td><td>27.35</td><td>0.913</td><td>217.27</td></tr><tr><td>GenSIRR (CVPR&#x27;26)</td><td>27.04</td><td>0.846</td><td>0.9648</td><td>0.654</td><td>0.831</td><td>0.9747</td><td>27.58</td><td>0.881</td><td>28.08</td><td>0.937</td><td>27.34</td><td>0.840</td><td>28.03</td><td>0.931</td><td>7214.94</td></tr><tr><td>Ours</td><td>28.84</td><td>0.859</td><td>0.9740</td><td>0.787</td><td>0.980</td><td>0.9827</td><td>28.23</td><td>0.881</td><td>28.83</td><td>0.925</td><td>27.89</td><td>0.883</td><td>28.77</td><td>0.921</td><td>87.09</td></tr></table>

## 5 Experiments

## 5.1 Experimental Setup

Implementation Details. Both the reflection synthesis and removal models are initialized from Wan2.1-Fun-v1.1-1.3B-Inp [10]. The synthesis model is trained for 8k steps on 21k paired samples, with pseudo reflection videos produced by FLUX.2-Klein-9B [9]. The removal model follows our two-stage strategy, trained for 30k steps per stage on 38k pairs, where 80% of the image samples are converted into video sequences by applying virtual camera motions (crop-tovideo strategy [34]). All models are trained on 8 NVIDIA A100 GPUs. Further training details are provided in Sections A and B in the appendix.

Benchmark. We introduce S2R-Bench, the first benchmark dedicated to video reflection removal, comprising two complementary subsets. S2R-Ref contains 60 paired videos constructed from the DRR dataset [5]. Each reflected/clean image pair is converted to a 10 fps static video sequence with identical virtual camera motions applied to both, preserving pixel-level alignment. S2R-Real contains 50 in-the-wild reflection videos collected from real captures and online sources, covering diverse glass materials, lighting conditions, reflection strengths, and camera motions, used for human perceptual evaluation and real-world generalization analysis. Details are provided in Section D in the appendix.

We additionally evaluate cross-domain generalization on standard image reflection removal benchmarks: Real [3], Nature [17], SIR<sup>2</sup> [22], and OpenRR-1k [23].

Evaluation Metrics. On S2R-Ref and image benchmarks we report PSNR [35] and SSIM [36]. For video evaluation we additionally report Temporal Consistency (TC) [37]. For S2R-Real, 15 participants score each video from 0 to 3 on two aspects: reflection removal quality and transmission preservation; scores are averaged and linearly normalized for reporting. Details are provided in Section D.2 in the appendix.

## 5.2 Comparison with State-of-the-Art

We compare against recent image reflection removal methods — DSRNet [4], DSIT [38], RD-Net [18], DAI [5], and GenSIRR [6] — applied frame-by-frame as video baselines. DAI is excluded from S2R-Ref because S2R-Ref is constructed from DAI’s training data. Existing video reflection removal methods [7, 20] release no code, so we compare against them only qualitatively on their test cases (see Section E).

![](images/e7f25a3b878a1fcbc788d5a7cb93eacf5b7f8b8cd37217ddbb4d1e228dbf5057.jpg)  
Figure 4 Qualitative comparison on controlled reflection videos with clean ground truth. Our method removes reflections more completely while preserving the transmission content.

![](images/915003f18a4c7ec79f2109ac83a630449e59cff2ed11f9c26c77a00d0ddb9d10.jpg)  
Figure 5 Qualitative comparison with SOTA reflection removal methods on videos (1st/20th/40th frame). Existing methods leave residues or temporally inconsistent removal, while our S2R-Removal produces coherent reflection removal.

Quantitative results. As shown in Table 1, our method achieves the best PSNR and TC on S2R-Ref and the highest human scores on S2R-Real for both reflection removal and transmission preservation. Despite being trained for video dereflection, it also delivers competitive or superior performance on image benchmarks, indicating strong cross-domain generalization. Beyond accuracy, our one-step design yields a substantial eficiency advantage: S2R-Removal runs at 87.09 ms/frame, ∼1.67× faster than the next-best method RDNet.

Qualitative results. As shown in Figures 4 and 5, image-based methods often leave reflection residues, introduce temporal flickering, or shift the global appearance of the scene, whereas our method suppresses the reflected layer more accurately while preserving the original tone and structure of the transmission content. Further qualitative results are provided in Section E.

Table 2 Efectiveness of our synthesis pipeline. RDNet is trained on OpenRR-1k\_train with diferent data configurations and evaluated on OpenRR-1k\_test. Legend: Trad. [4]; P/St/Pa/Re/Th/Ro = ours with Planar/Static/Partial/Reflectance/Thickness/Roughness; Δ is relative to the first row of each panel.
<table><tr><td>ID Setting</td><td>PSNR↑ ∆PSNR</td><td></td><td>SSIM↑</td><td>∆SSIM</td></tr><tr><td>A: Synthetic data source</td><td></td><td></td><td></td><td></td></tr><tr><td>A1 Base only</td><td>28.05</td><td>+0.00</td><td>0.9465</td><td>+0.0000</td></tr><tr><td> $\mathrm { A } 2 \ + \ \mathrm { T r a d } .$   $\mathrm { A 3 \ell + O u r s ( P ) }$ </td><td>31.98 32.35</td><td>+3.93 +4.31</td><td>0.9619 0.9643</td><td>+0.0154 +0.0178</td></tr><tr><td> $\mathsf { A } 4 \ + \mathrm { T r a d . } \ + \mathsf { O u r s } ( \mathrm { P } )$ </td><td>33.26</td><td>+5.21</td><td>0.9679</td><td>+0.0214</td></tr><tr><td>B: PGA variants</td><td></td><td></td><td></td><td></td></tr><tr><td>B1 P</td><td>32.35</td><td>+0.00</td><td>0.9643</td><td>+0.0000</td></tr><tr><td>B2  $\mathrm { \Delta P } + \mathrm { S t \Delta }$ </td><td>32.81</td><td>+0.46</td><td>0.9670</td><td>+0.0027</td></tr><tr><td>B3  $\mathrm { P + \cal S t + \cal P a }$ </td><td>33.62</td><td>+1.27</td><td>0.9691</td><td>+0.0048</td></tr><tr><td>B4  $\mathrm { P } + \mathrm { S t } + \mathrm { P a } + \mathrm { R e }$ </td><td>33.58</td><td>+1.23</td><td>0.9695</td><td>+0.0052</td></tr><tr><td>B5  $\mathrm { P } + \mathrm { S t } + \mathrm { P a } + \mathrm { R e } + \mathrm { T h }$ </td><td>33.70</td><td>+1.35</td><td>0.9697</td><td>+0.0054</td></tr><tr><td></td><td>34.13</td><td></td><td></td><td></td></tr><tr><td>B6 Full PGA</td><td></td><td>+1.78 </td><td>0.9704 +0.0061</td><td></td></tr></table>

## 5.3 Efectiveness of S2R-Synthesis

To isolate the efect of data generation, we fix the backbone to the state-of-the-art image reflection removal model RDNet [18] and vary only the training data. All models are trained on OpenRR-1k trainset and evaluated on OpenRR-1k testset.

Table 2 reports results across two panels. Panel A compares data sources. Adding traditional layer-blending synthesis [4] to the baseline already yields a clear gain, confirming the value of synthetic supervision. Replacing it with our difusion-rendered planar data further improves PSNR from 31.98 dB to 32.35 dB and SSIM from 0.9619 to 0.9643, and combining both sources achieves the best result in Panel A, suggesting complementarity.

Panel B evaluates our Physics-Grounded Augmentation (PGA). Adding each type of augmentations progressively improves overall performance. Full PGA reaches 34.13 dB PSNR and 0.9704 SSIM, a gain of +1.78 dB and +0.0061 SSIM over the planar-only setting, validating that physically motivated controls substantially improve the diversity and training value of synthetic data.

## 5.4 Ablation Study

We ablate the key training losses of S2R-Removal on S2R-Ref, as summarized in Table 3. Starting from the difusion-only baseline, the residual-derived intensity supervision improves PSNR from 27.00 dB to 27.71 dB and SSIM from 0.8348 to 0.8430, showing that explicit reflection-intensity guidance benefits reflection-aware latent adaptation. Afterwards, the Stage-II pixel-geometric refinement losses provide consistent gains: $\mathcal { L } _ { \mathrm { r e c } }$ improves reconstruction fidelity, $\mathcal { L } _ { s s i m }$ enhances structural consistency, and $\mathcal { L } _ { \mathrm { d e p t h } }$ further regularizes scene geometry. The full objective achieves the best result, with 28.84 dB PSNR and 0.8594 SSIM. Qualitative efects of two stages are provided in Section F in the appendix.

Table 3 Ablation study of S2R-Removal on S2R-Ref. We progressively enable the Stage-I latent adaptation losses and the Stage-II pixel-geometric refinement losses.
<table><tr><td rowspan="2">ID</td><td colspan="2">Stage-I</td><td colspan="3">Stage-II</td><td rowspan="2">PSNR↑</td><td rowspan="2">SSIM↑</td></tr><tr><td> ${ \mathcal { L } } _ { \mathrm { d i f f } }$ </td><td> ${ \mathcal { L } } _ { \mathrm { i n t } }$ </td><td> $\mathcal { L } _ { \mathrm { r e c } }$ </td><td> $\mathcal { L } _ { s s i m }$ </td><td> $\mathcal { L } _ { \mathrm { d e p t h } }$ </td></tr><tr><td>1</td><td>√</td><td>一</td><td>一</td><td>一</td><td>一</td><td>27.00</td><td>0.8348</td></tr><tr><td>2</td><td>√</td><td>√</td><td>一</td><td>一</td><td>一</td><td>27.71</td><td>0.8430</td></tr><tr><td>3</td><td>√</td><td>√</td><td>√</td><td>一</td><td>一</td><td>28.21</td><td>0.8513</td></tr><tr><td>4</td><td>√</td><td>√</td><td>√</td><td>√</td><td>一</td><td>28.55</td><td>0.8566</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>28.84</td><td>0.8594</td></tr></table>

## 6 Conclusion

We presented a closed-loop framework for video reflection removal, spanning physics-grounded paired video synthesis, reflection-aware difusion-based dereflection, and benchmark evaluation. Our synthesis pipeline performs structure-space reflection composition with Physics-Grounded Augmentation and a learned video renderer, enabling large-scale paired training data generation. Based on this data, S2R-Removal adapts a pretrained video difusion prior with reflection-intensity supervision and pixel-geometric refinement for temporally coherent reflection removal. Together with S2R-Bench, our framework provides a complete foundation for training and evaluating video dereflection methods.

## Acknowledgements

This work uses the FLUX.2-klein-9B model, licensed under FLUX Non-Commercial License. The Ditto-1M, UltraVideo, OpenRR-1k and OpenRR-5k datasets licensed under CC BY-NC-SA 4.0. The SIR2 dataset licensed for non-commercial purposes. The authors confirm that all uses of the above resources are strictly for academic research purposes and not for any commercial application.

## References

[1] H. Farid and E. H. Adelson, “Separating reflections and lighting using independent components analysis,” in Proceedings. 1999 IEEE computer society conference on computer vision and pattern recognition (Cat. No PR00149), vol. 1. IEEE, 1999, pp. 262–267.

[2] J. Yang, H. Li, Y. Dai, and R. T. Tan, “Robust optical flow estimation of double-layer images under transparency or reflection,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 1410–1419.

[3] X. Zhang, R. Ng, and Q. Chen, “Single image reflection separation with perceptual losses,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 4786–4794.

[4] Q. Hu and X. Guo, “Single image reflection separation via component synergy,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 13 138–13 147.

[5] J. Hu, C. Yang, Z. Zhou, J. Fang, Q. Tian, and W. Shen, “Dereflection any image with difusion priors and diversified data,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 6, 2026, pp. 4860–4868.

[6] M. Li, J. Hu, H. Wang, Q. Hu, J. Wang, and X. Guo, “Rectifying latent space for generative single-image reflection removal,” arXiv preprint arXiv:2512.06358, 2025.

[7] A. Nandoriya, M. Elgharib, C. Kim, M. Hefeeda, and W. Matusik, “Video reflection removal through spatio-temporal optimization,” in Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 2411–2419.

[8] Q. Wen, Y. Tan, J. Qin, W. Liu, G. Han, and S. He, “Single image reflection removal beyond linearity,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 3771–3779.

[9] B. F. Labs, “FLUX.2: Frontier Visual Intelligence,” https://bfl.ai/blog/flux-2, 2025.

[10] T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang et al., “Wan: Open and advanced large-scale video generative models,” arXiv preprint arXiv:2503.20314, 2025.

[11] A. Levin and Y. Weiss, “User assisted separation of reflections from a single image using a sparsity prior,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 29, no. 9, pp. 1647–1654, 2007.

[12] X. Guo, X. Cao, and Y. Ma, “Robust separation of reflection from multiple images,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2014, pp. 2187–2194.

[13] Q. Fan, J. Yang, G. Hua, B. Chen, and D. Wipf, “A generic deep architecture for single image reflection removal and image smoothing,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 3238–3247.

[14] J. Yang, D. Gong, L. Liu, and Q. Shi, “Seeing deeply and bidirectionally: A deep learning approach for single image reflection removal,” in Proceedings of the european conference on computer vision (ECCV), 2018, pp. 654–669.

[15] R. Wan, B. Shi, L.-Y. Duan, A.-H. Tan, and A. C. Kot, “Crrn: Multi-scale guided concurrent reflection removal network,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 4777–4785.

[16] C. Lei, X. Huang, M. Zhang, Q. Yan, W. Sun, and Q. Chen, “Polarized reflection removal with perfect alignment in the wild,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 1750–1758.

[17] Z. Dong, K. Xu, Y. Yang, H. Bao, W. Xu, and R. W. Lau, “Location-aware single image reflection removal,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 5017–5026.

[18] H. Zhao, M. Li, Q. Hu, and X. Guo, “Reversible decoupling network for single image reflection removal,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 26 430–26 439.

[19] Y. Zhu, X. Fu, P.-T. Jiang, H. Zhang, Q. Sun, J. Chen, Z.-J. Zha, and B. Li, “Revisiting single image reflection removal in the wild,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 25 468–25 478.

[20] A. Ahmed, S. Kim, M. Elgharib, and M. Hefeeda, “User-assisted video reflection removal,” in Proceedings of the 12th ACM Multimedia Systems Conference, 2021, pp. 122–131.

[21] K. Wei, J. Yang, Y. Fu, D. Wipf, and H. Huang, “Single image reflection removal exploiting misaligned training data and network enhancements,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 8178–8187.

[22] R. Wan, B. Shi, H. Li, Y. Hong, L.-Y. Duan, and A. C. Kot, “Benchmarking single-image reflection removal algorithms,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 2, pp. 1424–1441, 2022.

[23] K. Yang, J. Cai, L. Ouyang, F.-A. Vasluianu, R. Timofte, J. Ding, H. Sun, L. Fu, J. Li, C. M. Ho et al., “Ntire 2025 challenge on single image reflection removal in the wild: Datasets, methods and results,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 1301–1311.

[24] E. Kee, A. Pikielny, K. Blackburn-Matzen, and M. Levoy, “Removing reflections from raw photos,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 161–171.

[25] D. Zakarin, T. Wandel, A. Obukhov, and D. Dai, “Reflection removal through eficient adaptation of difusion transformers,” arXiv preprint arXiv:2512.05000, 2025.

[26] T. Wang, W. Lu, K. Zhang, T. Lu, and M.-H. Yang, “Promptrr: Difusion models as prompt generators for single image reflection removal,” arXiv preprint arXiv:2402.02374, 2024.

[27] Z. Lu, W. Wang, T. Guo, and F. Wang, “Single-image reflection removal via self-supervised difusion models: Z. lu et al.” The Journal of Supercomputing, vol. 81, no. 1, p. 338, 2025.

[28] T. Xu, C. Zhang, G. Zhai, and X. Liu, “Fumo: Prior-modulated difusion for single image reflection removal,” arXiv preprint arXiv:2603.19036, 2026.

[29] HuggingFace, “controlnet-aux: Controlnet auxiliary models,” https://github.com/huggingfa ce/controlnet\_aux, 2023.

[30] B. Walter, S. R. Marschner, H. Li, and K. E. Torrance, “Microfacet models for refraction through rough surfaces.” Rendering techniques, vol. 2007, p. 18th, 2007.

[31] B. Karis and E. Games, “Real shading in unreal engine 4,” Proc. Physically Based Shading Theory Practice, vol. 4, no. 3, p. 1, 2013.

[32] M. Born and E. Wolf, Principles of optics: electromagnetic theory ofpropagation, interference and difraction of light. Elsevier, 2013.

[33] W. Yin, J. Zhang, O. Wang, S. Niklaus, L. Mai, S. Chen, and C. Shen, “Learning to recover 3d scene shape from a single image,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 204–213.

[34] J. Hu, D. Zhou, D. Fu, F. Li, Z. Wang, F. Wang, W. Liao, J. Xie, and H. Sun, “Autoawg: Adverse weather generation with adaptive multi-controls for automotive videos,” in Proceedings of the 2026 International Conference on Multimedia Retrieval, 2026, pp. 835–844.

[35] A. Hore and D. Ziou, “Image quality metrics: Psnr vs. ssim,” in 2010 20th international conference on pattern recognition. IEEE, 2010, pp. 2366–2369.

[36] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.

[37] Z. Zhang, B. Wu, X. Wang, Y. Luo, L. Zhang, Y. Zhao, P. Vajda, D. Metaxas, and L. Yu, “Avid: Any-length video inpainting with difusion model,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 7162–7172.

[38] Q. Hu, H. Wang, and X. Guo, “Single image reflection separation via dual-stream interactive transformers,” Advances in Neural Information Processing Systems, vol. 37, pp. 55 228–55 248, 2024.

[39] Q. Bai, Q. Wang, H. Ouyang, Y. Yu, H. Wang, W. Wang, K. L. Cheng, S. Ma, Y. Zeng, Z. Liu et al., “Scaling instruction-based video editing with a high-quality synthetic dataset,” arXiv preprint arXiv:2510.15742, 2025.

[40] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.

[41] J. Cai, K. Yang, L. Ouyang, L. Fu, J. Ding, J. Shen, and Z. Meng, “Openrr-5k: A large-scale benchmark for reflection removal in the wild,” in 2025 IEEE 8th International Conference on Multimedia Information Processing and Retrieval (MIPR). IEEE, 2025, pp. 14–19.

[42] Z. Wang, Y. Li, Y. Zeng, Y. Fang, Y. Guo, W. Liu, J. Tan, K. Chen, T. Xue, B. Dai et al., “Humanvid: Demystifying training data for camera-controllable human image animation,” Advances in Neural Information Processing Systems, vol. 37, pp. 20 111–20 131, 2024.

[43] Z. Xue, J. Zhang, T. Hu, H. He, Y. Chen, Y. Cai, Y. Wang, C. Wang, Y. Liu, X. Li et al., “Ultravideo: High-quality uhd video dataset with comprehensive captions,” arXiv preprint arXiv:2506.13691, 2025.

[44] K. Team, J. Chen, Y. Ci, X. Du, Z. Feng, K. Gai, S. Guo, F. Han, J. He, K. He et al., “Kling-omni technical report,” arXiv preprint arXiv:2512.16776, 2025.

[45] D. Wu, M.-W. Liao, W.-T. Zhang, X.-G. Wang, X. Bai, W.-Q. Cheng, and W.-Y. Liu, “Yolop: You only look once for panoptic driving perception,” Machine Intelligence Research, vol. 19, no. 6, pp. 550–562, 2022.

[46] G. Jocher, “Ultralytics yolov5,” https://github.com/ultralytics/yolov5, 2020.

[47] F. Yu, H. Chen, X. Wang, W. Xian, Y. Chen, F. Liu, V. Madhavan, and T. Darrell, “Bdd100k: A diverse driving dataset for heterogeneous multitask learning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 2636–2645.

## Appendix

This appendix is organized as follows:

• A1. Reflection Synthesis Training Data: VLM-filtered FLUX-based pseudo reflection video generation and the resulting paired data composition.

• A2. Reflection Removal Training Data: Paired video synthesis from HumanVid and UltraVideo, with the full training data composition.

• A3. Physics-Grounded Augmentation: Detailed derivations and operations of the six PGA primitives, with a trend comparison against a UE5-based pipeline.

• A4. S2R-Bench and Human Evaluation: Construction of the S2R-Ref and S2R-Real subsets and the human perceptual scoring criteria.

• A5. Additional Qualitative Comparisons: Real-world comparison, comparison with a general-purpose video editing model, and comparison with prior video dereflection methods.

• A6. Component Visualization: Cross-attention response of the reflection-intensity head in Stage I, and the qualitative efect of Stage II pixel-geometric refinement.

• A7. Application to Single Images: Zoom-based video inference for applying the video model to single-image reflection removal, with cross-domain qualitative evaluation on image benchmarks.

• A8. Downstream Benefits of Dereflection: Driving area segmentation and vehicle detection on BDD100K before and after dereflection.

• A9. Limitations: Discussion of method limitations on multi-layer reflections and coupled camera-motion geometry.

## A Reflection Synthesis Training Data

## A.1 FLUX-Based Pseudo Reflection Video Generation

To construct paired training data for video reflection synthesis, we first filter reflection-free videos from the publicly available Ditto-1M dataset [39] using a Vision-Language Model (VLM), specifically Qwen3-VL-8B-Instruct [40]. Each video is evaluated, and only those containing no glass or eyeglass reflections are retained. We design the following prompt for the VLM:

“You are a visual AI assistant. Analyze the content of the video and answer precisely. Task: Determine whether there is any glass reflection or eyeglass reflection visible in the video frames. If there is any frame containing glass or eyeglass reflections, output only: True. If there are no reflections at all in all frames, output only: False. Do NOT provide explanations, descriptions, or extra text. Only output True or False.”

For each filtered video, we apply image-to-image synthesis using FLUX.2-Klein-9B on a perframe basis to add synthetic glass reflections. To introduce diversity in reflection intensity and appearance, we define a set of three prompts, and a single prompt is applied consistently across all frames of a given video:

Table 4 Composition of the reflection synthesis training dataset.
<table><tr><td>Source Modality</td><td>Source</td><td>Samples</td></tr><tr><td rowspan="2"></td><td>OpenRR-1k</td><td>800</td></tr><tr><td>OpenRR-5k</td><td>5,000</td></tr><tr><td>Image</td><td>RRW</td><td>3,000</td></tr><tr><td>Video</td><td>Ditto-1M (FLUX frame-wise synthetic)</td><td>11,132</td></tr><tr><td>Total</td><td></td><td>19,932</td></tr></table>

• Subtle reflection: “Preserve the original image unchanged. Only add a subtle glass reflection overlay on top of the scene. The reflection should befaint, transparent, and natural, with soft highlights and slight glare. Do not alter the structure, objects, colors, or composition of the original image.”

• Strong reflection: “Preserve the original image unchanged. Only add strong, multi-layered glass reflections over the scene. The reflections should be vivid, highly visible, and complex, with overlapping glare, pronounced highlights, and multiple reflection sources. Do not alter the structure, objects, colors, or composition of the original image.”

• Colored reflection: “Preserve the original image completely unchanged. Only overlay a colored glass reflection efect onto the scene. The reflections should be vivid, clear, and complex, with clearly visible light reflection sources. Do not alter the structure, objects, colors, or composition of the original image.”

This stage yields 11,132 paired video clips with inter-frame flickering reflections, where each pair consists of the original reflection-free frame and its synthesized reflection counterpart.

## A.2 Training Data Summary

Using the pipeline described above, we generate 11,132 paired video clips from Ditto-1M [39] as training data for S2R-Synthesis model. To further leverage existing image resources (incl. OpenRR-1k [23], OpenRR-5k [41], RRW [19]), we apply a crop-to-video strategy [34] to convert existing image reflection datasets into short video clips, serving as supplementary training data. The full training dataset is summarized in Table 4.

## B Reflection Removal Training Data

## B.1 Physics-Grounded Paired Video Synthesis

We first leverage the trained reflection synthesis model to generate paired reflection videos without inter-frame flickering. Specifically, we filter reflection-free videos from the publicly available HumanVid [42] and UltraVideo [43] datasets using a VLM, and synthesize 20,031 and 5,768 paired reflection videos respectively. The PGA parameters used during synthesis are summarized in Table 5. In addition, we synthesize 2,997 paired reflection videos using the traditional synthesis method.

Table 5 PGA augmentation parameters used for video reflection synthesis.
<table><tr><td>Augmentation</td><td>Parameter</td><td>Range</td><td>Probability</td></tr><tr><td>Roughness</td><td>σ</td><td>[5.0, 15.0]</td><td>0.5</td></tr><tr><td>Thickness</td><td>shift  $\Delta _ { x }$  shift  $\Delta _ { y }$  weight w</td><td>[-50,50] [-50,50] [0.3,0.7]</td><td>0.5</td></tr><tr><td>Reflectance</td><td>scale  $\gamma$  bias  $\beta$ </td><td>[0.5,2.5] [0,0.15]</td><td>0.5</td></tr><tr><td>Partial</td><td>num boxes</td><td>[3,9]</td><td>0.7</td></tr><tr><td>Static</td><td></td><td>一</td><td>0.1</td></tr><tr><td>Planar</td><td></td><td></td><td>0.1</td></tr></table>

Table 6 Composition of the reflection removal training dataset.
<table><tr><td>Modality</td><td>Source</td><td>Samples</td></tr><tr><td>Image</td><td>OpenRR-1k OpenRR-5k RRW</td><td>800 5,000 3,000</td></tr><tr><td>Video</td><td>HumanVid (our synthesis) UltraVideo (our synthesis)</td><td>20,031 5,768</td></tr><tr><td>Total</td><td>UltraVideo (traditional synthesis)</td><td>2,997 37,596</td></tr></table>

## B.2 Dataset Composition

The training set for the reflection removal model contains 38,826 samples in total, comprising both image and video data, as summarized in Table 6. The image subset consists of 10,030 samples, identical to those used in reflection synthesis model training. During training, 80% of the image samples are dynamically converted into video sequences by applying virtual camera motions (crop-to-video strategy [34]). The video subset consists of 28,796 samples. Of these, 25,799 are synthesized from HumanVid (20,031) and UltraVideo (5,768) using our physics-grounded pipeline, and 2,997 are synthesized from UltraVideo using a traditional RGB-space blending baseline [4].

## C Physics-Grounded Augmentation

Given a transmission video � and a reflection source video �, we first extract their respective lineart conditions $E _ { T }$ and $E _ { R } ,$ then apply a series of physics-grounded augmentations to the reflection lineart $E _ { R } ,$ , and finally fuse the augmented reflection condition with the transmission condition to obtain the composed structural condition. Table 7 summarizes the correspondence between glass efects, physical cues, and our control-space operations. The detailed augmentation parameters are summarized in Table 5.

Table 7 Physics-Grounded Augmentation (PGA) control variables and their physical interpretations.
<table><tr><td>Augmentation</td><td>Glass effect</td><td>Physical cue</td><td>Structure-space operation</td></tr><tr><td>Planar</td><td>Flat glass reflection</td><td>Planar interface</td><td>Direct fusion</td></tr><tr><td>Roughness</td><td>Surface scattering</td><td>Microfacet roughness</td><td>Gaussian blur (σ)</td></tr><tr><td>Thickness</td><td>Ghosting / secondary reflection</td><td>Refractive displacement</td><td>Translation / secondary offset (∆, w)</td></tr><tr><td>Reflectance</td><td>Reflection strength variation</td><td>Fresnel reflectance</td><td>Intensity modulation  $( \gamma , \beta )$ </td></tr><tr><td>Partial</td><td>Local glass coverage / occlusion</td><td>Spatial coverage</td><td>Random spatial mask (p)</td></tr><tr><td>Static</td><td>Temporally stable reflection</td><td>Stationary reflection source</td><td>Frozen reflection frame</td></tr></table>

## C.1 Per-Primitive Physical Derivation

Roughness Augmentation. Under the microfacet model [30], a rough glass surface distributes reflected rays over a wide angular lobe according to the GGX normal distribution function; in the far-field regime this angular spread is equivalent to convolving the ideal mirror reflection with an isotropic kernel whose width grows with roughness [31]. We reproduce this efect by applying a channel-wise depthwise separable Gaussian convolution to the reflection lineart. The standard deviation � is randomly sampled from [5.0, 15.0], and the kernel size is automatically determined as $k = 2 \lfloor 3 \sigma \rfloor + 1$ to ensure efective coverage of the Gaussian distribution within the ±3� range. The 2D Gaussian kernel is constructed as the outer product of a 1D kernel:

$$
g _ { i } = \exp \left( - \frac { i ^ { 2 } } { 2 \sigma ^ { 2 } } \right) , \quad \mathbf { K } = \frac { \mathbf { g } \mathbf { g } ^ { \top } } { \vert \vert \mathbf { g } \vert \vert _ { 1 } ^ { 2 } } ,\tag{11}
$$

and applied via frame-wise spatial convolution over the video.

Thickness Augmentation. Light incident on thick or multi-layer glass undergoes partial reflection at each interface, producing multiple reflections with lateral ofsets. Under the paraxial approximation of Snell’s Law [32], the lateral displacement between two such reflections is proportional to glass thickness �, $\delta \approx d \theta _ { i } ( 1 - 1 / n )$ , where $\theta _ { i }$ is the incident angle and � the refractive index. Motivated by this multi-interface displacement, we apply a spatial translation to the reflection lineart to simulate ghosting artifacts caused by refraction in thick glass. The horizontal ofset $\Delta _ { x }$ and vertical ofset $\Delta _ { y }$ are independently sampled from [−50, 50] pixels, and a translated lineart is generated via crop-and-pad operations. The blending weight � is sampled from [0.3, 0.7], and the augmentation is formulated as:

$$
E _ { R } ^ { t h i c k } = E _ { R } + w \cdot \mathcal { W } _ { ( \Delta _ { x } , \Delta _ { y } ) } ( E _ { R } ) ,\tag{12}
$$

where $\mathcal { W } _ { ( \Delta _ { x } , \Delta _ { y } ) }$ denotes the pixel-wise translation operation with zero-padding at the boundaries.

Reflectance Augmentation. Following radiometric image formation, the observed reflection component can be decomposed into a view-dependent specular term and a difuse ambient term, $I _ { \mathrm { r e f l } } = F \cdot L _ { \mathrm { e n v } } + L _ { \mathrm { a m b } } $ , where $L _ { \mathrm { e n v } }$ is the directional environmental radiance, � is the Fresnel reflectance [32], and $L _ { \mathrm { a m b } }$ accounts for the residual ambient illumination that contributes a spatially uniform brightness ofset to the reflection layer. Under the assumption of spatially uniform incident angle, � reduces to a scalar, which motivates an element-wise linear intensity modulation applied to the reflection lineart to simulate the viewing-angle-dependent reflection strength variation. The scale factor � is sampled from [0.5, 2.5] and the bias $\beta$ is sampled from [0, 0.15]. The augmentation is formulated as:

$$
E _ { R } ^ { r e f l } = \gamma \cdot E _ { R } + \beta .\tag{13}
$$

![](images/7d8cc75151b8b885f48e4c59e064b8e858410ab83e1547ea56818b65891e83af.jpg)  
Figure 6 Comparison of reflection-synthesis trends between S2R-Synthesis (with PGA) and a UE5-based pipeline under controlled variation of three glass properties (roughness, reflectance, thickness).

$\gamma < 1$ simulates weak reflections from transparent glass, $\gamma > 1$ simulates strong specular reflections, and $\beta$ introduces a brightness bias to account for ambient illumination.

Partial Coverage Augmentation. A random binary spatial mask is applied to the reflection lineart to simulate partial glass coverage or spatially varying reflectance. $N _ { b } \in [ 3 , 9 ]$ rectangular regions are randomly generated, with the top-left and bottom-right coordinates of each rectangle independently sampled from the normalized image coordinate space [0, 1]. The union of all rectangles forms the final mask $P \in \{ 0 , 1 \} ^ { H \times W }$ , and the augmentation is formulated as $E _ { R } ^ { c r o p } = E _ { R } \odot P _ { . }$ , zeroing out the reflection structure outside the masked regions.

Static Reflection Augmentation. A randomly selected frame $f _ { r }$ from the reflection source video is repeated along the temporal dimension across the entire clip, such that $E _ { R } ^ { s t a t i c } [ f ] = E _ { R } [ f _ { r } ]$ for all $f \in \{ 1 , \ldots , F \}$ , simulating a reflection source that remains stationary relative to the camera and ensuring complete temporal stability of the reflection layer.

## C.2 Condition Fusion and Physical Validation

The augmented reflection lineart and the transmission lineart are summed element-wise and clipped to [0, 1], yielding the final composed three-channel structural condition:

$$
E _ { F } = \mathrm { c l i p } \left( E _ { T } + \mathcal { A } ( E _ { R } ; \theta _ { g } ) , 0 , 1 \right) ,\tag{14}
$$

where $\mathcal { A }$ denotes the composition of the above augmentation operations and $\theta _ { g }$ denotes the randomly sampled augmentation parameters. All parameters are sampled at the clip level and shared across frames to ensure temporal coherence.

Figure 6 compares S2R-Synthesis with a UE5-based reflection synthesis pipeline under controlled variation of three glass properties: roughness, reflectance, and thickness. Since the two pipelines operate under diferent rendering formulations, their numerical parameters are not directly comparable; we therefore focus on the qualitative trend each produces as a single property is varied.

Table 8 Human perceptual evaluation scoring criteria.
<table><tr><td>Dimension</td><td>Description</td><td>Score</td></tr><tr><td rowspan="4">Removal Quality</td><td>Complete removal</td><td>3</td></tr><tr><td>Major reflection components removed</td><td>2</td></tr><tr><td>Slight removal</td><td>1</td></tr><tr><td>Failure</td><td>0</td></tr><tr><td rowspan="4">Region Preservation</td><td>No visible degradation</td><td>3</td></tr><tr><td>Minor degradation</td><td>2</td></tr><tr><td>Noticeable degradation</td><td>1</td></tr><tr><td>Severe degradation</td><td>0</td></tr></table>

In both pipelines, increasing roughness broadens the reflection blur, larger reflectance strengthens the overlay, and larger thickness amplifies the ghosting ofset. S2R-Synthesis reproduces these physically expected trends through lightweight structure-space controls, without requiring a heavyweight rendering engine.

## D S2R-Bench and Human Evaluation

## D.1 S2R-Bench Construction

S2R-Bench consists of two complementary subsets: a full-reference subset S2R-Ref for quantitative evaluation and a no-reference subset S2R-Real for human perceptual evaluation.

S2R-Ref. S2R-Ref is constructed from the training data of the DRR dataset [5]. We first convert the reflection-contaminated frames and their corresponding clean frames into static video sequences at 10 fps, yielding 434 candidate paired video sequences in total. However, since the paired frames in DRR are obtained via controlled capture, some sequences exhibit synthesis artifacts or inter-frame flickering that would compromise evaluation reliability. We therefore conduct manual inspection and retain only sequences with visually clean and temporally consistent reflection appearance, resulting in 60 high-quality paired video sequences.

To simulate realistic camera motion, we apply identical virtual camera motions to both videos in each pair using the Ken Burns efect, which simulates lens movement through cropping and scaling. For each pair, we randomly select one of two motion modes: Pan or Zoom.

In Pan mode, a scale factor $s \in [ 1 . 5 , 2 . 0 ]$ is first sampled uniformly at random. A start crop position $( x _ { 0 } , y _ { 0 } )$ and an end crop position $( x _ { 1 } , y _ { 1 } )$ are then sampled within the scaled image space, subject to the constraint that their relative displacement exceeds 20% of the image width or height, ensuring suficient motion magnitude.

In Zoom mode, the crop center is fixed while the scale factor varies over time. The start and end scale factors are sampled from [1.2, 1.5] and [1.7, 2.0] respectively (or vice versa), simulating a zoom-in or zoom-out efect.

![](images/2f61f58c2ab468048e9ffc7b75b53badcd320d360158c47eccb0b3dc2b2f155a.jpg)  
Figure 7 Qualitative comparison on real-world reflection videos without ground truth.

For each frame �, the crop position and scale are linearly interpolated according to the temporal progress $t = i / ( N - 1 ) \in [ 0 , 1 ]$ :

$$
\left\{ \begin{array} { l l } { x _ { t } = x _ { 0 } ( 1 - t ) + x _ { 1 } t , } \\ { y _ { t } = y _ { 0 } ( 1 - t ) + y _ { 1 } t , } \\ { s _ { t } = s _ { 0 } ( 1 - t ) + s _ { 1 } t . } \end{array} \right.\tag{15}
$$

The resulting crop region is then resized back to the original resolution. Since both videos share the same motion parameters, the reflected video and the clean ground truth undergo identical framewise transformations, strictly preserving pixel-level alignment while introducing realistic camera dynamics. This design supports full-reference quantitative evaluation with metrics including PSNR and SSIM.

S2R-Real. S2R-Real contains 50 in-the-wild reflection videos, each consisting of 81 frames, collected from real-world captures and online sources, covering diverse glass materials, lighting conditions, reflection strengths, and camera motion patterns. Among them, 19 samples are dynamic videos with noticeable scene or camera motion, while the remaining 31 samples are static videos with relatively stable content. Since ground-truth clean videos are unavailable in real-world scenarios, this subset is used for human perceptual evaluation.

## D.2 Human Perceptual Evaluation

The human perceptual evaluation criteria are summarized in Table 8. For reflection removal quality, scores range from 0 to 3, where 3 denotes complete removal, 2 denotes removal of major reflection components, 1 denotes slight removal, and 0 denotes failure. For non-reflection region preservation, scores also range from 0 to 3, where 3 denotes no visible degradation, 2 denotes minor degradation, 1 denotes noticeable degradation, and 0 denotes severe degradation. We report the normalized removal and preservation scores separately, where higher values indicate better reflection suppression and stronger transmission preservation.

![](images/ee2611c2186c03baf7640e0a14bb7cd400004fe55134c8cbb30f7bacd68f4bc0.jpg)  
Input

![](images/e80c923f3f609d72806673393198e88dfc273191ed8e580fc862cc57644cb23c.jpg)  
Nandoriya et al. (ICCV'17)

![](images/34cebfa7e76cb992502bdbe8e4c174deed1c004de0f0b6676b1880e4493ffd97.jpg)  
Ahmed et al. (MMSys'21)

![](images/40ccbacde52b900fb9af49095f34f6f47b21971e4da87819b738c610b9c053db.jpg)  
Ours  
Figure 8 Qualitative comparison with prior video reflection removal methods [7, 20]. Their result frames are extracted from the original papers as no code is publicly available.

## E Additional Qualitative Comparisons

We provide additional qualitative results complementing the main paper’s SOTA comparison: a comparison on real-world reflection videos without ground truth (Figure 7), a qualitative comparison with prior video reflection removal methods (Figure 8), and a comparison with a general-purpose video editing model (Figure 9).

Real-world qualitative comparison. Figure 7 compares S2R-Removal with recent image reflection removal methods on real-world reflection videos without ground truth. Image-based methods often leave reflection residues or produce temporally inconsistent removal across frames, while S2R-Removal more completely suppresses strong reflections while preserving the original scene appearance.

Comparison with prior video dereflection methods. As noted in the main paper, prior video reflection removal methods [7, 20] release no code, precluding quantitative comparison; we therefore compare qualitatively on their respective test cases in Figure 8, with their result frames extracted directly from the original papers. S2R-Removal visibly outperforms these methods in both reflection suppression and transmission preservation, particularly on strong reflections where prior methods leave noticeable residues or alter the underlying scene content.

![](images/7f81b4d2d2d69f847655b975a2f84f06be4bfa3036af12c251cb71901a204477.jpg)  
Input

![](images/8a1aa6d9a2421b97c9ac5ee9b1e4985142ec62f92cdbf01a0a3c83721bb0120f.jpg)  
Kling O1

![](images/217d6750d3d46ff029c1c571af05a67a914e935b218826fd3a70c87eb0afcb5e.jpg)  
Ours  
Figure 9 Qualitative comparison with the general-purpose video editing model Kling O1 on three representative cases.

Comparison with general-purpose video editing models. We also compare against closed-source general-purpose video editing models. Figure 9 compares S2R-Removal with Kling O1 [44] on three representative cases. While general editing models show strong editing ability, they are not specifically trained for reflection layer separation and may alter the global appearance, scene content, or background structure. S2R-Removal performs more targeted dereflection and better preserves the original video content.

## F Component Visualization

## F.1 Reflection-Intensity Head

Attention Map Extraction. During inference, we extract the cross-attention weights from the DiT backbone of the Wan2.1 video difusion model. Specifically, for each attention layer, the attention weights between visual Queries and textual Keys are computed as:

$$
A = \mathsf { s o f t m a x } \left( \frac { Q K ^ { \top } } { \sqrt { d } } \right) \in \mathbb { R } ^ { L _ { \mathit { \nu } i s } \times L _ { \mathit { t e x t } } } ,\tag{16}
$$

where � denotes the attention head dimension, and $L _ { \nu i s }$ and $L _ { t e x t }$ denote the sequence lengths of visual tokens and text tokens, respectively.

We average the attention maps across all attention heads and Transformer layers to obtain the aggregated cross-attention map:

$$
\bar { A } = \frac { 1 } { N _ { l a y e r } } \sum _ { l = 1 } ^ { N _ { l a y e r } } \frac { 1 } { N _ { h e a d } } \sum _ { h = 1 } ^ { N _ { h e a d } } A _ { l } ^ { h } \in \mathbb { R } ^ { L _ { v i s } \times L _ { t e x t } } .\tag{17}
$$

![](images/c3f0a1144de7dcdab52db747065e3c66b82e61fed13043fac3a62699cc5bb232.jpg)  
Figure 10 Visualization of the reflection-intensity head in Stage I.

![](images/72e30d5f408f95a2f0aaa4396798ce1ed3a7e618fd762dfd04a34518f786a0b0.jpg)  
Figure 11 Qualitative comparison between Stage I and Stage II of S2R-Removal.

Text Token Attention Response. We extract the attention response of all visual tokens to the text token “reflection” (indexed by �):

$$
a _ { s } = \bar { A } [ : , s ] \in \mathbb { R } ^ { L \nu i s } .\tag{18}
$$

This vector represents the response intensity of each spatio-temporal location in the video to the semantic concept “reflection”. The vector is reshaped into a spatio-temporal grid $( F ^ { \prime } , H ^ { \prime } , W ^ { \prime } )$ where $F ^ { \prime } , H ^ { \prime } , W ^ { \prime }$ denote the temporal and spatial dimensions in the latent space. It is then spatially upsampled and temporally interpolated to recover the full pixel-space resolution $( F , H , W )$ , where:

$$
F = 4 ( F ^ { \prime } - 1 ) + 1 ,\tag{19}
$$

yielding the final frame-wise attention map:

$$
\mathcal { A } \in \mathbb { R } ^ { F \times H \times W } .\tag{20}
$$

Visualization. The attention map is normalized into the range [0, 255] for visualization:

$$
\hat { A } = \frac { A - \operatorname* { m i n } ( A ) } { \operatorname* { m a x } ( A ) - \operatorname* { m i n } ( A ) + \epsilon } \times 2 5 5 .\tag{21}
$$

Figure 10 visualizes the attention map activations with and without the intensity head. The columns show the clean target, reflected input, removal result without the intensity head, DiT response without the intensity head, removal result with the intensity head, DiT response with the intensity head, and the predicted reflection-intensity map. Without the intensity head, strong reflection regions are less attended by the DiT and remain dificult to remove; with residual-derived intensity supervision, the model produces more reflection-aware responses that better align with reflection-corrupted regions, and the predicted intensity map further captures both reflection location and strength, demonstrating the benefit of explicit reflection-intensity supervision.

![](images/58e45439347c6a4c7019071eb93dd5f1041d9f04a1818692266e8b5453e1c9dc.jpg)  
Figure 12 Qualitative comparison on the Real, $\mathsf { S I R } ^ { 2 }$ , and Nature image benchmarks.

Table 9 Single-image inference vs. zoom-based video inference on image reflection removal benchmarks.
<table><tr><td rowspan="2">Method</td><td colspan="2">Real</td><td colspan="2"> $\mathbf { S I R } ^ { 2 }$ </td><td colspan="2">Nature</td><td colspan="2">Average</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>Single-image</td><td>27.61</td><td>0.857</td><td>28.27</td><td>0.888</td><td>27.45</td><td>0.841</td><td>28.21</td><td>0.885</td></tr><tr><td>Zoom-in</td><td>27.85</td><td>0.866</td><td>28.62</td><td>0.899</td><td>27.58</td><td>0.857</td><td>28.55</td><td>0.896</td></tr><tr><td>Zoom-out</td><td>28.23</td><td>0.881</td><td>28.83</td><td>0.925</td><td>27.89</td><td>0.883</td><td>28.77</td><td>0.921</td></tr></table>

## F.2 Stage I vs Stage II

Figure 11 compares the two stages of S2R-Removal on representative reflection videos. Stage I suppresses most reflection artifacts but may introduce appearance shifts or structural inconsistencies; Stage II applies one-step pixel-geometric refinement to recover finer appearance details and better preserve scene geometry. The corresponding quantitative gains are reported in Table 3.

Table 10 Downstream driving area segmentation (YOLOP [45]) on reflection-contaminated BDD100K videos, before (Raw) and after (DeRef) our video reflection removal.
<table><tr><td>Model</td><td>Input</td><td>Acc↑ IoU↑</td><td>mIoU↑</td></tr><tr><td>YOLOP</td><td>Raw</td><td>0.965 0.823</td><td>0.891</td></tr><tr><td></td><td>DeRef</td><td>0.967 0.835</td><td>0.898</td></tr></table>

Table 11 Downstream vehicle detection (YOLOv5n [46]) on reflection-contaminated BDD100K videos, before (Raw) and after (DeRef) our video reflection removal.
<table><tr><td>Model</td><td>Input</td><td>P↑</td><td>R↑</td><td>mAP@0.5↑</td><td>mAP@0.95↑</td></tr><tr><td rowspan="2">YOLOv5n</td><td>Raw</td><td>0.827</td><td>0.361</td><td>0.351</td><td>0.221</td></tr><tr><td>DeRef</td><td>0.844</td><td>0.385</td><td>0.372</td><td>0.238</td></tr></table>

## G Application to Single Images

Although our model is designed for video reflection removal, it can also be applied to single images by converting them into short video sequences. Specifically, given a reflection-contaminated image, we synthesize a pseudo video clip by applying a smooth zoom transformation, generating � frames via linear interpolation between scale factors. We consider two zoom modes: zoom-in (scale from 1.0 to �) and zoom-out (scale from � to 1.0), where � = 1.5, � = 45 frames.

Our model is then applied to the synthesized pseudo video clip; for zoom-in the first output frame is used for evaluation, and for zoom-out the last frame is used instead. Table 9 compares the three inference modes on the Real, SIR<sup>2</sup>, and Nature benchmarks: zoom-based video inference consistently outperforms direct single-image inference, and zoom-out achieves the best results across all three benchmarks, which we therefore adopt for single-image evaluation.

Using zoom-out inference, we further compare S2R-Removal with state-of-the-art methods on the same image benchmarks (Figure 12). Although trained for video dereflection, our model achieves more efective reflection removal while better preserving the transmission content, consistent with the quantitative cross-method results in Table 1.

## H Downstream Benefits of Dereflection

Reflection removal can serve as a useful preprocessing step for downstream vision systems by providing cleaner visual inputs. To evaluate the practical benefit of our method, we conduct downstream task evaluation on reflection-contaminated videos from BDD100K [47]. We compare task performance before and after applying our video dereflection model. The evaluation covers two representative autonomous driving tasks: driving area segmentation and object detection. We use YOLOP for joint driving perception, and YOLOv5n as an additional object detection model.

Tables 10 and 11 report the quantitative results, while Figure 13 presents representative qualitative examples. After applying dereflection, YOLOP [45] improves driving area segmentation performance, with IoU increasing from 0.823 to 0.835 and mIoU increasing from 0.891 to 0.898. As shown in Figure 13a, removing reflections enables more complete and accurate segmentation of the drivable area.

![](images/e71d5c97f045226a66e5ef0871a3d49fb396d7530d0a178bc436c3c102c6388e.jpg)  
(a) Driving area segmentation (YOLOP)

![](images/0862433784dd43803813c5c1c9fe2da7fe391ed33bb5390b2ca49e5e56629387.jpg)  
(b) Object detection (YOLOv5n)

Figure 13 Qualitative results on downstream perception tasks. The proposed dereflection model improves visual clarity and benefits both segmentation and detection performance.  
![](images/4aee797229bccf1f48da8d269101b7f295b864f7c753011542e8bd781bcdc453.jpg)  
Figure 14 Limitation in multi-layer reflection scenes. Due to ambiguous nested reflections, the model may leave slight inner-layer residuals (top, S2R-Real) or over-remove them (bottom, S2R-Ref).

For vehicle detection, YOLOv5n [46] also achieves consistent improvements after dereflection, with mAP@0.5 increasing from 0.351 to 0.372 and mAP@0.95 increasing from 0.221 to 0.238. The qualitative results in Figure 13b further demonstrate that dereflection efectively alleviates reflection-induced missed detections and improves vehicle localization accuracy. Overall, our dereflection model efectively reduces reflection-induced interference and improves downstream perception in visually challenging scenarios.

## I Limitations

Our current framework still has two limitations.

First, nested reflections from multiple glass layers remain challenging. Since reflections behind another transparent surface may either be treated as removable artifacts or as part of the scene, the restoration target becomes ambiguous. As illustrated in Figure 14, our model may therefore either leave slight residual inner-layer reflections (top) or over-remove them (bottom). This issue is also common to existing dereflection methods, and may be mitigated by incorporating more multi-layer reflection cases into future training data.

Second, our synthesis pipeline currently models temporally coherent reflections with clip-level controls, but does not explicitly simulate the coupled change between camera motion and reflection geometry, such as viewpoint-dependent reflection parallax. Future work will explore richer physical reflection simulation and larger real-world video benchmarks to further improve robustness in complex glass scenarios.