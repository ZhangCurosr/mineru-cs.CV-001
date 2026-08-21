# BeyondMasks: Evaluating Causal and Physical Consistency in Video Object Removal

Yiğit Ekin<sup>1†</sup> , Enes Sanli<sup>2,4†</sup> , Aykut Erdem<sup>2,4</sup> , Erkut Erdem<sup>3,4</sup> , and Aysegul Dundar<sup>1</sup>

Bilkent University, Department of Computer Engineering, Ankara, Türkiye <sup>2</sup> Koç University, Department of Computer Engineering, Istanbul, Türkiye 3 Hacettepe University, Department of AI and Data Engineering, Ankara, Türkiye 4 KUIS AI Center, Istanbul, Türkiye yigit.ekin@bilkent.edu.tr

Project Page · Code · Dataset

![](images/6c8f0d3d16e25e6c01bd18abc12d434245c293626ac050f40a95755e8c14c86e.jpg)  
Fig. 1: Video object removal requires eliminating both objects and their physical after-efects. Objects interact with their environment through shadows, reflections, translucency, illumination changes, volumetric efects (e.g., steam), and dynamic physical traces. Removing the object alone does not restore the scene if these efects persist. BeyondMasks introduces a benchmark to evaluate such causal objectenvironment interactions using temporally aligned video pairs, enabling systematic assessment of whether methods recover the scene as if the object had never been present.

Abstract. Recent advances in generative video models have significantly improved visual realism in video object removal, yet evaluation protocols still focus on masked-region fidelity, treating removal as local inpainting. In real scenes, object removal is a causal intervention:

eliminating an object also requires removing its induced physical effects, such as shadows, reflections, illumination changes, translucency, and dynamic traces. Existing benchmarks lack aligned clean references or remain limited to simplified synthetic settings, preventing systematic evaluation of causal consistency. We introduce BeyondMasks, a paired benchmark for causally consistent video object removal, consisting of temporally aligned synthetic and real-world video pairs with clean background references. The dataset spans diverse photometric, geometric, volumetric, and dynamic interactions, and supports both mask-based and instruction-driven editing. We further propose CORE, a structured vision–language model-based evaluation protocol that jointly measures object disappearance and after-efect consistency, aligning more closely with human judgments than existing metrics. Benchmarking state-of-theart methods reveals systematic failures in removing secondary physical efects despite high masked-region fidelity, exposing a gap between visual plausibility and causal correctness. BeyondMasks reframes video object removal as causal scene consistency rather than local reconstruction and provides a unified framework for its evaluation.

Keywords: Video object removal with after efect· Benchmark dataset · Paired video evaluation · Vision-language models

## 1 Introduction

Video object removal aims to edit a video such that a selected object appears as if it were never present. With the rise of large-scale generative video models [6, 25, 31, 32], recent systems achieve striking perceptual realism, enabling increasingly sophisticated scene manipulation. Modern video foundation models further integrate instruction-driven editing [11] and open-world reasoning, positioning object removal [18] as a core capability within general-purpose video generation pipelines. However, evaluation protocols have not kept pace with model capabilities. Most existing benchmarks assess reconstruction fidelity only within the masked region, implicitly framing object removal as a localized inpainting problem.

In real scenes, object removal is fundamentally a causal intervention. Objects do not merely occupy pixels; they interact with their environment across space and time. As illustrated in Fig. 1, they cast shadows, alter illumination, appear in reflections, scatter light through translucent materials, and induce dynamic traces such as footprints or dust displacements. Removing an object therefore requires eliminating both the object itself and the downstream physical efects it induces. Evaluating masked-region reconstruction alone overlooks this broader causal footprint and fails to measure whether the scene has been physically restored.

Current benchmarks are ill-suited for this purpose. Widely used video datasets such as DAVIS [19] and YouTube-VOS [30] lack paired clean backgrounds, preventing full removal verification. More recent benchmarks like ROSE-Bench [18] introduce synthetic paired data to study specific physical side efects. While representing important progress, these datasets remain limited in interaction diversity and real-world variability, and typically focus on isolated photometric efects. They do not systematically evaluate dynamic object-environment interactions, nor do they naturally support instruction-driven editing pipelines increasingly used in modern video foundation models.

Motivated by these limitations, we advocate a causally grounded formulation of video object removal. Under this view, the task requires disentangling the target object from the scene and undoing its induced environmental changes consistently over time. This perspective shifts evaluation from local reconstruction quality toward object-environment consistency and temporal physical reasoning. To enable this paradigm, we introduce BeyondMasks, a diagnostically focused benchmark for evaluating causal and physical consistency in video object removal. BeyondMasks contains a collection of temporally aligned paired videos, including synthetic sequences generated through controlled editing and real-world tripod-captured scenes. Rather than maximizing scale, we prioritize precise causal control, physical diversity, and clean reference alignment, enabling systematic evaluation of object-environment disentanglement. Each sequence provides a clean reference background, per-frame object masks, and instruction prompts, supporting both mask-based and text-guided editing scenarios.

Beyond paired pixel-level evaluation, we propose CORE (Causal Object Removal Evaluation) score, a vision-language model (VLM)-based evaluation metric that jointly measures object disappearance and after-efect consistency. We validate this protocol through human studies and inter-model agreement analysis, demonstrating its reliability and alignment with perceptual judgments.

Extensive benchmarking of state-of-the-art mask-based and instruction-driven models reveals systematic failures in removing secondary physical efects, even when masked-region fidelity remains high. These findings expose a substantial gap between visual plausibility and causal scene consistency, highlighting the need for evaluation frameworks that capture physical reasoning rather than local reconstruction alone.

The main contributions of this paper can be summarized as follows:

– We introduce a causally grounded formulation of video object removal, reframing evaluation around object-environment consistency rather than maskedregion reconstruction.

– We construct BeyondMasks, a paired benchmark of 180 synthetic and realworld video sequences with aligned clean backgrounds, masks, and instruction prompts, spanning diverse causal interactions. It more than doubles the scale of prior paired video object removal benchmarks.

– We propose and validate CORE, a structured VLM-based evaluation framework that jointly assesses object disappearance and the causal consistency of induced physical after-efects.

– We benchmark contemporary mask-based and instruction-driven video object removal models and reveal systematic failures in handling dynamic and interaction-driven physical efects.

Table 1: Comparison of video object removal benchmarks. Unlike prior datasets, BeyondMasks provides temporally aligned paired ground truth while explicitly modeling diverse causal after-efects in both synthetic and real-world scenes.
<table><tr><td>Dataset</td><td colspan="5">Size Paired GT After-effects Causal effects Avg. video length Source</td></tr><tr><td>DAVIS</td><td>90</td><td>X</td><td>x</td><td>X ~3 secs</td><td>Real</td></tr><tr><td>YouTube-VOS</td><td>508</td><td>x</td><td>x x</td><td>~4.5 secs</td><td>Real</td></tr><tr><td>HQVI</td><td>10</td><td>√ x</td><td>x</td><td>~3.5 secs</td><td>Real</td></tr><tr><td>ROSE-Bench</td><td>60</td><td>√</td><td>√ x</td><td>~3 secs</td><td>Synthetic</td></tr><tr><td>Ours</td><td>180</td><td>√</td><td>√ L</td><td>~5.5 secs</td><td>Both</td></tr></table>

## 2 Related Work

## 2.1 Video Object Removal Benchmarks

Early evaluations of video object removal rely on segmentation datasets such as DAVIS [19] and YouTube-VOS [30]. Although these provide high-quality masks, they lack paired clean background videos in which both the object and its induced efects are absent. As a result, evaluation is limited to masked-region reconstruction, implicitly treating removal as local inpainting. HQVI [5] addresses this by constructing paired ground-truth videos via alpha compositing of foreground objects onto real backgrounds. While it provides temporally aligned pairs and occlusion-aware annotations, the compositing process assumes no secondary environmental efects, keeping evaluation focused on spatial reconstruction rather than object-environment disentanglement. ROSE-Bench [18] introduces synthetic paired videos generated via 3D rendering and models side efects such as shadows, reflections, mirror interactions, light emission, and translucency<sup>5</sup>. Despite advancing physically grounded evaluation, synthetic benchmarks remain limited in interaction diversity and environmental variability, primarily emphasizing isolated photometric efects. They do not systematically capture dynamic interactions such as deformation, fluid perturbations, particle traces, or scene rearrangement under realistic motion, nor do they support instructiondriven editing. Table 1 summarizes the diferences between prior benchmarks and BeyondMasks. In contrast, BeyondMasks provides temporally aligned paired videos across synthetic and real-world scenes, explicitly modeling diverse causal after-efects and dynamic interactions, enabling systematic evaluation beyond masked-region fidelity.

## 2.2 Evaluation of Physical and Causal Consistency

Video editing is typically evaluated using reference-based metrics such as PSNR, SSIM [26], and LPIPS [36], which measure pixel similarity to ground truth. While efective for reconstruction quality, these metrics do not capture whether object-induced efects persist or whether dynamic physical interactions remain temporally consistent. Perceptual measures such as FVD [24] assess realism but also remain insensitive to intervention correctness. At the image level, taskspecific metrics such as ReMOVE [4] and CFD [33] incorporate object-aware signals to penalize residual foreground traces and assess semantic consistency. However, they operate on single frames and do not address temporal coherence or dynamic object–environment interactions. Recent benchmarks employ pretrained vision-language models (VLMs) as automated evaluators. VBench [9,40], VideoPhy [1,2], and LLM-based judge frameworks [14] use structured prompting to assess semantic alignment and motion realism in generated videos. Yet these approaches evaluate realism without aligned references and are not designed for intervention-based editing. Causal object removal requires comparison against temporally aligned clean backgrounds to verify both object disappearance and elimination of induced physical efects. We address this through CORE, a structured VLM-based evaluation protocol that leverages paired references and explicitly reasons about object removal and residual after-efects. CORE enables scalable evaluation of causal consistency beyond masked-region fidelity.

## 3 BeyondMasks: Benchmark Design and Construction

We introduce BeyondMasks, a benchmark for evaluating video object removal under a causal intervention framework. While the recent ROSE-Bench [18] dataset incorporates selected physical side efects in synthetic settings, prior benchmarks largely assess removal through masked-region reconstruction or efect-specific realism. We instead formalize object removal as an intervention on a scenegenerating process, aiming to recover the interventional scene state in which both the object and its induced environmental changes are absent, so that the resulting video corresponds to the scene as if the object had never been present.

Let $S _ { t }$ denote the latent physical state of a scene at time t, and let O denote the presence of a target object. We conceptualize each observed frame as arising from a rendering process $V _ { t } = R ( S _ { t } , O )$ , where a scene state together with the object configuration is mapped to pixel observations. Under an intervention in which the object is removed $( \mathrm { i . e . , } \ O \ = \ \varnothing )$ , the same scene state is rendered without the object, yielding the clean background frame $V _ { t } ^ { b g } = R ( S _ { t } , \emptyset )$ . This formulation highlights that object removal is not merely masked reconstruction, but a counterfactual rendering problem: recovering the scene as it would appear under the intervention $O = \emptyset$ . Importantly, O influences $V _ { t }$ not only through direct appearance (object pixels), but also indirectly through environmental interactions that alter $S _ { t }$ , including shadows, reflections, illumination shifts, surface deformation, and dynamic perturbations. True object removal therefore requires approximating the interventional distribution $R ( S _ { t } , \emptyset )$ rather than merely hallucinating plausible background content within a masked region.

BeyondMasks implements this formulation through paired, temporally aligned intervention data, as illustrated in Fig. 2. Each sample consists of: (i) an objectpresent sequence $V ^ { o b j }$ generated under $O \neq \emptyset$ , and (ii) a clean reference sequence $V ^ { b g }$ generated under $O \ = \ Q$ . Because the two sequences difer only by this controlled intervention, evaluation directly measures whether a removal model successfully recovers the interventional scene state.

![](images/945ecdc38ebd738863687b854b1aec52b2437820f0758a5147c786e70bba98e4.jpg)  
Fig. 2: Overview of the BeyondMasks construction pipeline. Top: Paired intervention-based generation, where an object-present video is derived from a clean background video, ensuring temporal alignment. Bottom: Representative categories of object-induced causal after-efects, including illumination, reflection, volumetric, and dynamic interaction efects.

## 3.1 Synthetic Paired Generation

The synthetic subset contains 90 scenes generated using Veo3’s [28] video editing capabilities. For each scene, we first generate a background-only video without the target object. This sequence serves as the clean reference. We then introduce the target object into the same scene through controlled editing, producing the object-present video. Since the latter is derived from the former, temporal alignment is preserved. The insertion process naturally induces physically coherent interactions with the environment, including cast shadows, illumination changes, reflections, translucency efects, and scene-dependent dynamic traces. Because the background sequence remains unchanged apart from the object insertion, the paired design ensures that the only diferences between videos correspond to the object and its causal footprint. This controlled construction enables both pixel-level comparison and higher-level semantic assessment. All synthetic pairs undergo manual review to verify physical plausibility, temporal coherence, and consistency between object presence and environmental interaction.

## 3.2 Real-World Paired Capture

To complement synthetic data, BeyondMasks includes 90 real-world paired sequences captured with a tripod-stabilized setup. For each scene, we record a short video with the object present and immediately capture a second video after removing the object, keeping camera position or settings fixed. This backto-back protocol preserves geometry, lighting, and scene composition, ensuring that diferences between paired videos arise solely from object presence and its associated efects. These real-world captures introduce natural texture complexity, subtle lighting variation, and realistic interaction patterns that are dificult to model synthetically, improving the realism and diversity of the benchmark.

## 3.3 Mask Annotation Pipeline

For each object-present sequence, we provide temporally consistent binary masks to support mask-based removal methods. Masks are generated through a semiautomatic pipeline that combines interactive segmentation using SAM2 [21] with temporal propagation via CoTracker 3 [12]. Annotators refine the initial-frame segmentation using positive and negative point prompts to obtain precise object boundaries, after which masks are propagated across frames and periodically corrected to maintain temporal coherence. Each sequence undergoes cross-review by a second annotator to ensure segmentation accuracy and consistency. Importantly, masks include only the target object and explicitly exclude its afterefects. Shadows, reflections, illumination changes, and other induced phenomena are not masked. This design choice reflects our causal formulation: models must infer and remove secondary efects through physical reasoning rather than relying on explicit supervision.

## 3.4 Taxonomy of Causal After-Efects

A central component of BeyondMasks is its taxonomy of object-induced environmental interactions. Rather than categorizing scenes by object type, we group samples according to the underlying physical mechanisms through which objects causally influence their surroundings. This mechanism-driven taxonomy ensures that evaluation reflects distinct modes of object-environment entanglement.

Shadow. Objects occlude or redirect illumination, producing cast shadows and local radiance changes. Successful removal requires restoring a shadow-consistent illumination field, not merely deleting object pixels.

Reflection. Objects may appear indirectly through specular or difuse reflections on mirrors, glass, or water. Removal must eliminate both direct and reflected evidence while preserving surface appearance and temporal consistency.

Translucent efects. Semi-transparent objects (e.g., glass) partially reveal the background while modulating light transport. These cases require recovering background structure consistent with partial visibility and attenuation.

Steam/scattering. Volumetric media such as steam or smoke introduce spatially difuse, time-varying traces. Removal demands undoing these efects while maintaining coherent scene dynamics and background continuity.

Causal physical efects. Some objects alter scene geometry or material state through motion or contact, yielding efects that are not visually attached to the object yet remain causally dependent on it. Examples include water ripples, dust trails, displaced objects, and surface deformation such as footprints.

Light source efects. When the object emits light, its removal requires compensating for localized illumination changes and restoring consistent exposure and shading.

No explicit after-efect. A small subset of sequences exhibits minimal secondary interaction, serving as controls for evaluating pure object removal with limited environmental entanglement.

## 3.5 Instruction Prompts

To support instruction-driven editing, each sample is associated with one or more validated text prompts specifying the removal target while emphasizing natural scene restoration. Prompts are constructed using structured templates to ensure object specificity and clarity, and are manually verified for consistency across both mask-based and text-conditioned paradigms.

## 4 CORE: Causal Object Removal Evaluation

While BeyondMasks enables paired pixel-level comparison, reconstruction metrics alone are insuficient for evaluating interventional correctness. Pixel-based measures such as PSNR, SSIM, and LPIPS penalize any deviation from the ground-truth background, even when the reconstructed scene is physically plausible and free of residual object traces. As illustrated in Fig. 3, methods with nearly identical LPIPS scores can difer substantially in whether object-induced physical efects are properly removed. In object removal, multiple visually coherent reconstructions may exist for unseen regions, making strict pixel identity an unreliable proxy for causal success. Under our causal formulation, successful removal corresponds to recovering the interventional scene state $R ( S _ { t } , \emptyset )$ , that is, the scene that would be observed if the object and its induced environmental changes were absent. Deviations from this target state arise at two distinct levels: (i) residual object presence, where visible traces of the object remain, and (ii) residual causal efects, where shadows, reflections, illumination shifts, or dynamic perturbations persist despite the object’s removal. To explicitly measure these dimensions, we introduce CORE (Causal Object Removal Evaluation), a structured VLM-based assessment framework illustrated in Fig. 4.

CORE evaluates removal using three temporally aligned inputs: the objectpresent video V , the aligned clean background $V ^ { b g }$ , and the edited output $\hat { V } ^ { b g }$ produced by the evaluated method.

![](images/60705c5d7932f20ec1a0f2d8881eddd57c12cdfe6f335fc3b964beddaa87baea.jpg)  
Fig. 3: Pixel metrics fail to capture causal correctness. Although LPIPS scores are similar across methods, their ability to remove object-induced efects (e.g., shadows or reflections) difers substantially. CORE separates these failure modes by evaluating object removal (CORE-OS) and elimination of causal after-efects (CORE-AES). ↓ better for LPIPS; ↑ better for CORE scores.

Rather than measuring pixel correspondence, CORE assesses whether the edited video is semantically and physically consistent with the interventional reference in which the object and its induced environmental changes are absent. The evaluation is implemented via a pretrained vision-language model operating under a fixed, structured prompt that enforces comparative reasoning between $\hat { V } ^ { b g }$ and $V ^ { b g }$ . The prompt explicitly guides the model to reason about two intervention-specific criteria: (i) whether the object has been fully removed, and (ii) whether its causal after-efects have been eliminated. These two dimensions correspond to complementary failure modes in object removal. Residual object traces may remain even if the background appears plausible, and conversely, shadows or dynamic perturbations may persist despite apparent object disappearance. To disentangle these behaviors, CORE decomposes evaluation into two scores:

– ObjectScore: completeness of object removal and plausibility of background restoration;

– AfterEfectScore: elimination of object-induced physical traces, including shadows, reflections, illumination changes, and dynamic perturbations.

We implement CORE using Gemini 3.1 [8]. Each evaluation call receives the three videos along with structured metadata specifying the target object and relevant after-efect categories. The VLM performs structured comparative reasoning: identifying object traces in the object-present video V, inferring expected appearance from the aligned clean background $V ^ { b g }$ , comparing the edited output $\hat { V } ^ { b g }$ against $V ^ { b g }$ under a naturalness criterion, and categorizing residual errors as object-related or afterefect-related. The scoring rubric explicitly rewards semantically coherent background reconstruction, even when pixelwise diferences from GT exist, while penalizing residual object evidence, persistent after-efects, or newly hallucinated foreground content. By conditioning semantic reasoning on aligned references, CORE transforms VLM-based scoring from observational realism assessment into structured intervention verification. The full prompt and rubric are provided in the supplementary material.

![](images/124bf499e991c74f2971cee701dea019d511c2b37c9efe59ba473c774728d83e.jpg)  
Fig. 4: Overview of CORE (Causal Object Removal Evaluation). Given an edited video $\hat { V } ^ { b g }$ and its temporally aligned clean reference $V ^ { b g }$ , CORE performs structured VLM-based comparison to assess interventional correctness. The evaluation jointly measures (i) object disappearance and (ii) consistency of object-induced physical after-efects, such as shadows, reflections, and dynamic perturbations. By conditioning semantic reasoning on aligned references, CORE moves beyond masked-region fidelity toward evaluation of causal and physical consistency.

Validation of CORE. To assess alignment with human judgment, we conduct a user study in which 8 annotators independently evaluate results from 5 methods across 20 videos using the same scoring rubric. The Pearson correlation between CORE scores and the mean human ratings is 0.615 for ObjectScore and 0.733 for AfterEfectScore. For reference, human inter-rater agreement yields Pearson correlations of 0.693 and 0.785 for the same dimensions, respectively. These results indicate that CORE closely aligns with human perception and approaches the upper bound defined by human agreement.

Table 2: Quantitative comparison on our benchmark. We evaluate representative video object removal methods using standard reconstruction metrics (PSNR, SSIM, LPIPS), the video realism metric FVD, and our VLM-based CORE evaluation. Type indicates the supervision required by each method: T denotes text-instructionbased editing, M denotes mask-guided removal, and $\mathbf { T } { \ + } \mathbf { M }$ indicates methods that support both inputs. Bold indicates the best score.
<table><tr><td>Method</td><td>Type</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FVD↓</td><td></td><td>CORE-OS↑ CORE-AES↑</td></tr><tr><td>Lucy Edit (arXiv 2025) [23]</td><td>T</td><td>18.5884</td><td>0.7239</td><td>0.2758</td><td>480.08</td><td>1.443</td><td>1.551</td></tr><tr><td>VACE (ICCV 2025) [11]</td><td>T+M</td><td>19.9932</td><td>0.7856</td><td>0.2077</td><td>418.79</td><td>1.799</td><td>1.698</td></tr><tr><td>CoCoCo (AAAI 2025) [44]</td><td>T+M</td><td>20.8565</td><td>0.7287</td><td>0.2404</td><td>381.84</td><td>1.986</td><td>2.043</td></tr><tr><td>Gen. Omnimatte (CVPR 2025) [13]</td><td>T+M</td><td>23.8106</td><td>0.8175</td><td>0.1968</td><td>250.76</td><td>3.771</td><td>3.021</td></tr><tr><td>ProPainter (ICCV 2023) [41]</td><td>M</td><td>23.5979</td><td>0.8429</td><td>0.1635</td><td>225.28</td><td>3.063</td><td>2.368</td></tr><tr><td>DiffuEraser (arXiv 2025) [15]</td><td>M</td><td>25.1880</td><td>0.8769</td><td>0.1124</td><td>164.26</td><td>3.494</td><td>2.625</td></tr><tr><td>MiniMax (NeurIPS 2025) [43]</td><td>M</td><td>23.1954</td><td>0.8218</td><td>0.1682</td><td>203.64</td><td>3.631</td><td>2.553</td></tr><tr><td>ROSE (NeurIPS 2025) [18]</td><td>M</td><td>23.8062</td><td>0.8173</td><td>0.1659</td><td>316.04</td><td>3.784</td><td>2.920</td></tr><tr><td>OmnimatteZero (SIGGRAPH 2025) [22]</td><td>M</td><td>22.9306</td><td>0.8127</td><td>0.1638</td><td>227.24</td><td>3.190</td><td>2.499</td></tr><tr><td>EffectErase (CVPR 2026) [7]</td><td>M</td><td>24.1206</td><td>0.8473</td><td>0.1621</td><td>242.16</td><td>4.015</td><td>3.162</td></tr></table>

## 5 Experimental Analysis

## 5.1 Experimental Setup

We benchmark nine recent video object removal and video editing methods spanning mask-based (M), text-guided (T), and hybrid (T+M) paradigms. These include classical mask-guided inpainting approaches such as ProPainter [41] and DifuEraser [15], layer-based decomposition models including Generative Omnimatte [13] and OmnimatteZero [22], and recent large-scale editing systems such as VACE [11], Lucy Edit [23], CoCoCo [44], MiniMax [43], and ROSE [18]. Together, these methods represent the dominant paradigms currently used for video object removal and instruction-driven video editing. All methods are evaluated using publicly released implementations and default configurations at the native resolution of BeyondMasks. For each method, we report standard reference-based metrics, PSNR, SSIM, LPIPS, computed against the paired GT backgrounds, as well as FVD for temporal quality and our proposed VLM-based CORE ObjectScore (CORE-OS) and AfterEfectScore (CORE-AES).

## 5.2 Quantitative Results

Table 2 summarizes the results. Across standard reconstruction metrics, several methods achieve comparable PSNR, SSIM, and LPIPS scores, indicating similar pixel-level fidelity with respect to the clean reference videos. However, these metrics do not reliably reflect removal correctness. Methods that achieve strong reconstruction scores frequently leave residual object evidence or fail to remove induced physical efects such as shadows or reflections. This discrepancy is particularly evident when comparing reconstruction metrics with CORE scores. For example, DifuEraser attains the highest PSNR and SSIM, yet exhibits noticeably lower AfterEfectScore than several competing methods, indicating incomplete removal of object-induced efects. Conversely, ROSE achieves the highest CORE-OS despite only moderate reconstruction metrics, suggesting more efective elimination of object traces even when pixel similarity is comparable. These results highlight a fundamental limitation of conventional evaluation: pixel-based fidelity does not necessarily correspond to causal correctness. By explicitly separating object disappearance (CORE-OS) from elimination of physical after-efects (CORE-AES), CORE reveals failure modes that remain hidden under traditional reconstruction metrics. This analysis confirms that realistic video object removal requires reasoning about object-environment interactions rather than merely reconstructing masked regions.

Table 3: Average performance across after-efect categories. Results are averaged across the top-performing six models for each after-efect type. Higher values are better for PSNR, SSIM, CORE-OS, and CORE-AES, while lower values indicate better performance for LPIPS.
<table><tr><td>After-effect PSNR↑</td><td></td><td></td><td></td><td></td><td>SSIM↑ LPIPS↓ CORE-OS↑ CORE-AES↑</td></tr><tr><td>Shadow</td><td>23.6870</td><td>0.8298</td><td>0.1621</td><td>3.498</td><td>2.520</td></tr><tr><td>Reflection</td><td>23.9842</td><td>0.8558</td><td>0.1580</td><td>3.332</td><td>2.446</td></tr><tr><td>Light source</td><td>22.4233</td><td>0.8201</td><td>0.1849</td><td>3.554</td><td>2.248</td></tr><tr><td>Steam</td><td>22.8375</td><td>0.8466</td><td>0.1624</td><td>3.236</td><td>2.032</td></tr><tr><td>Translucent</td><td>22.9326</td><td>0.8156</td><td>0.2162</td><td>3.256</td><td>2.701</td></tr><tr><td>Causal</td><td>22.7912</td><td>0.8163</td><td>0.1867</td><td>3.486</td><td>2.141</td></tr></table>

To further analyze performance across diferent physical phenomena, Table 3 reports results aggregated by after-efect category. The breakdown reveals substantial variation in dificulty across interaction types. While shadows and reflections are handled relatively well on average, volumetric efects such as steam and dynamic interactions (e.g., footprints or surface disturbances) exhibit significantly lower CORE-AES scores. These results indicate that secondary physical processes extending beyond the masked region remain challenging for current removal methods. Overall, the analysis confirms that realistic object removal requires modeling object-environment interactions rather than relying solely on local reconstruction.

## 5.3 Qualitative Analysis

Fig. 5 presents representative examples comparing several state-of-the-art methods. While most approaches successfully remove the target objects, many fail to eliminate object-induced efects. Shadows, reflections, and illumination changes often persist even after the object disappears, resulting in physically inconsistent scenes. These residual traces are particularly evident in cases involving reflective surfaces, cast shadows, and lighting interactions.

![](images/043778988d7bf328b804d22f73ab2b2f92bf052a37f663715dcf8e77f3810176.jpg)  
Fig. 5: Qualitative comparison across six after-efect categories. Each category shows two temporal frames.

After-Efect Removal is the Dominant Challenge. Across the examples in Fig. 5, removing the object itself is often easier than eliminating the physical efects it induces in the environment. While most methods successfully erase the primary object, secondary phenomena such as shadows, reflections, and illumination changes frequently persist. These efects are spatially difuse and may extend beyond the masked region, making them dificult to address with standard inpainting strategies. This observation highlights the importance of evaluating object removal as a causal intervention rather than a purely local reconstruction task.

![](images/b5064c9ae4c08864a608c98b2eab2861cbbb5ebd44c63c3d196439d19bb7b860.jpg)  
Fig. 6: Context loss in mask-based video object removal. Many mask-based pipelines zero out the masked region prior to inference, eliminating potentially informative visual evidence (e.g., background partially visible through translucent structures). This removal of observable context forces the model to hallucinate missing content rather than recover available information, resulting in degraded restoration and incomplete causal consistency relative to the ground truth.

Mask-Based vs. Text-Guided Removal. Qualitative diferences also emerge between mask-based and instruction-driven approaches. Mask-based methods typically achieve strong object removal due to explicit spatial supervision, but often leave residual environmental efects such as shadows or reflections. In contrast, text-guided approaches ofer greater editing flexibility but show inconsistent removal of fine-grained physical traces and may introduce hallucinated content (see Supplementary for visual examples). These observations suggest that current systems, regardless of supervision modality, struggle to disentangle objects from the physical processes they induce in the surrounding environment.

Context Loss in Mask-Based Inpainting. A structural limitation of many mask-based pipelines is the removal or replacement of masked regions before inference. This operation discards potentially informative visual evidence, particularly for translucent or semi-transparent objects where background content remains partially observable. As illustrated in Fig. 6, eliminating these signals forces models to hallucinate missing content rather than recover available context. Similar observations have been reported in image-based removal [10]. In video, temporal redundancy provides additional opportunities to exploit such information across frames, suggesting that future removal systems should incorporate context-aware reasoning rather than relying solely on masked inpainting.

## 6 Conclusion

We introduced BeyondMasks, a benchmark for evaluating video object removal from a causal intervention perspective. Unlike existing evaluation protocols that primarily assess masked-region reconstruction, BeyondMasks enables systematic evaluation of whether removal methods eliminate both the object and the physical efects it induces in the surrounding environment. By providing temporally aligned video pairs with clean background references and diverse interaction categories, the benchmark allows controlled analysis of object-environment disentanglement in realistic editing scenarios. Using this benchmark, we evaluated a diverse set of recent video object removal and video editing methods and identified a consistent gap between high reconstruction fidelity and true causal correctness. While many approaches successfully remove the visible object, they frequently leave behind residual shadows, reflections, illumination changes, or other secondary physical traces. To address limitations of existing evaluation protocols, we introduced CORE, a VLM-based evaluation framework that separately measures object disappearance and the removal of object-induced afterefects. Our experiments show that conventional pixel-based metrics often fail to capture these failure modes, whereas CORE provides a more faithful assessment of causal consistency and aligns more closely with human perception.

## Acknowledgements

This work was partly supported by the KUIS AI Center Research Awards to ES, and TÜBİTAK-2247-A Program Award (No. 123C550) to AE. We thank all the reviewers for their valuable comments.

## References

1. Bansal, H., Lin, Z., Xie, T., Zong, Z., Yarom, M., Bitton, Y., Jiang, C., Sun, Y., Chang, K.W., Grover, A.: Videophy: Evaluating physical commonsense for video generation. In: The Thirteenth International Conference on Learning Representations (2025), https://openreview.net/forum?id=9D2QvO1uWj, Accessed: 1 July 2026

2. Bansal, H., Peng, C., Bitton, Y., Goldenberg, R., Grover, A., Chang, K.W.: Videophy-2: A challenging action-centric physical commonsense evaluation in video generation. In: The Fourteenth International Conference on Learning Representations (2026), https://openreview.net/forum?id=HA8KSQW7SO, Accessed: 1 July 2026

3. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18392–18402 (2023)

4. Chandrasekar, A., Chakrabarty, G., Bardhan, J., Hebbalaguppe, R., AP, P.: Remove: A reference-free metric for object erasure. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. pp. 7901–7910 (June 2024)

5. Cho, S., Oh, S.W., Lee, S., Lee, J.Y.: Elevating flow-guided video inpainting with reference generation. In: Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2025). pp. 2527–2535 (2025)

6. Dharmarajan, K., Huang, W., Wu, J., Fei-Fei, L., Zhang, R.: Dream2flow: Bridging video generation and open-world manipulation with 3d object flow. arXiv preprint arXiv:2512.24766 (2025)

7. Fu, Y., Zheng, Y., Dai, Z., Ding, H.: Efecterase: Joint video object removal and insertion for high-quality efect erasing (2026), https://arxiv.org/abs/2603. 19224

8. Gemini Team, Google DeepMind: Gemini 3.1 pro - model card (February 2026), https://deepmind.google/models/model-cards/gemini-3-1-pro/, Accessed: 1 July 2026

9. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., Liu, Z.: Vbench: Comprehensive benchmark suite for video generative models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 21807–21818 (June 2024)

10. Jiang, L., Wang, Z., Bao, J., Zhou, W., Chen, D., Shi, L., Chen, D., Li, H.: Smarteraser: Remove anything from images using masked-region guidance. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 24452– 24462 (2025)

11. Jiang, Z., Han, Z., Mao, C., Zhang, J., Pan, Y., Liu, Y.: Vace: All-in-one video creation and editing. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 17191–17202 (October 2025)

12. Karaev, N., Makarov, Y., Wang, J., Neverova, N., Vedaldi, A., Rupprecht, C.: Cotracker3: Simpler and better point tracking by pseudo-labelling real videos. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 6013–6022 (2025)

13. Lee, Y.C., Lu, E., Rumbley, S., Geyer, M., Huang, J.B., Dekel, T., Cole, F.: Generative omnimatte: Learning to decompose video into layers. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 12522–12532 (2025)

14. Li, H., Dong, Q., Chen, J., Su, H., Zhou, Y., Ai, Q., Ye, Z., Liu, Y.: Llms-asjudges: A comprehensive survey on llm-based evaluation methods (2024), https: //arxiv.org/abs/2412.05579, Accessed: 1 July 2026

15. Li, X., Xue, H., Ren, P., Bo, L.: Difueraser: A difusion model for video inpainting (2025), https://arxiv.org/abs/2501.10018, Accessed: 1 July 2026

16. Liu, J., Qu, Y., Yan, Q., Zeng, X., Wang, L., Liao, R.: Fréchet video motion distance: A metric for evaluating motion consistency in videos. In: First Workshop on Controllable Video Generation@ ICML24 (2024)

17. Lu, E., Cole, F., Dekel, T., Zisserman, A., Freeman, W.T., Rubinstein, M.: Omnimatte: Associating objects and their efects in video. In: CVPR. pp. 4507–4515 (June 2021)

18. Miao, C., Feng, Y., Zeng, J., Gao, Z., Hantang, L., Yan, Y., Qi, D., Chen, X., Wang, B., Zhao, H.: ROSE: Remove objects with side efects in videos. In: NeurIPS (2025), https://openreview.net/forum?id=xTWWKMxY1x, Accessed: 1 July 2026

19. Perazzi, F., Pont-Tuset, J., McWilliams, B., Van Gool, L., Gross, M., Sorkine-Hornung, A.: A benchmark dataset and evaluation methodology for video object segmentation. In: CVPR (June 2016)

20. Qin, B., Li, J., Tang, S., Chua, T.S., Zhuang, Y.: Instructvid2vid: Controllable video editing with natural language instructions. In: 2024 IEEE International Conference on Multimedia and Expo (ICME). pp. 1–6. IEEE (2024)

21. Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., et al.: Sam 2: Segment anything in images and videos. In: International Conference on Learning Representations. vol. 2025, pp. 28085–28128 (2025)

22. Samuel, D., Levy, M., Darshan, N., Chechik, G., Ben-Ari, R.: Omnimattezero: Fast training-free omnimatte with pre-trained video difusion models. In: SIGGRAPH Asia 2025 Conference Papers. SA Conference Papers ’25, Association for Computing Machinery, New York, NY, USA (2025)

23. Team, D.: Lucy edit: Open-weight text-guided video editing (2025)

24. Unterthiner, T., van Steenkiste, S., Kurach, K., Marinier, R., Michalski, M., Gelly, S.: FVD: A new metric for video generation (2019), https://openreview.net/ forum?id=rylgEULtdN, Accessed: 1 July 2026

25. Wang, Y., Chen, X., Ma, X., Zhou, S., Huang, Z., Wang, Y., Yang, C., He, Y., Yu, J., Yang, P., et al.: Lavie: High-quality video generation with cascaded latent difusion models. International Journal of Computer Vision 133(5), 3059–3078 (2025)

26. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE Trans. Image Process. 13(4), 600–612 (2004)

27. Wei, R., Yin, Z., Zhang, S., Zhou, L., Wang, X., Ban, C., Cao, T., Sun, H., He, Z., Liang, K., Ma, Z.: Omnieraser: Remove objects and their efects in images with paired video-frame data. arXiv preprint arXiv:2501.07397 (2025), https://arxiv. org/abs/2501.07397, Accessed: 1 July 2026

28. Wiedemer, T., Li, Y., Vicol, P., Gu, S.S., Matarese, N., Swersky, K., Kim, B., Jaini, P., Geirhos, R.: Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328 (2025)

29. Winter, D., Cohen, M., Fruchter, S., Pritch, Y., Rav-Acha, A., Hoshen, Y.: Objectdrop: Bootstrapping counterfactuals for photorealistic object removal and insertion. In: ECCV (2024)

30. Xu, N., Yang, L., Fan, Y., Yang, J., Yue, D., Liang, Y., Price, B., Cohen, S., Huang, T.: Youtube-vos: Sequence-to-sequence video object segmentation. In: ECCV (September 2018)

31. Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al.: Cogvideox: Text-to-video difusion models with an expert transformer. arXiv preprint arXiv:2408.06072 (2024)

32. Yu, S., Liu, D., Ma, Z., Hong, Y., Zhou, Y., Tan, H., Chai, J., Bansal, M.: Veggie: Instructional editing and reasoning video concepts with grounded generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15147–15158 (2025)

33. Yu, Y., Zeng, Z., Zheng, H., Luo, J.: Omnipaint: Mastering object-oriented editing via disentangled insertion-removal inpainting. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 17324–17334 (October 2025)

34. Yu, Y., Zeng, Z., Zheng, H., Luo, J.: Omnipaint: Mastering object-oriented editing via disentangled insertion-removal inpainting (2025), https://arxiv.org/abs/ 2503.08677, Accessed: 1 July 2026

35. Yuan, H., Zhang, S., Wang, X., Wei, Y., Feng, T., Pan, Y., Zhang, Y., Liu, Z., Albanie, S., Ni, D.: Instructvideo: Instructing video difusion models with human feedback. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6463–6474 (2024)

36. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586–595 (2018)

37. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: CVPR (2018)

38. Zhao, J., Zhou, S., Wang, Z., Yang, P., Loy, C.C.: Objectclear: Complete object removal via object-efect attention (2025), https://arxiv.org/abs/2505.22636, Accessed: 1 July 2026

39. Zhao, J., Zhou, S., Wang, Z., Yang, P., Loy, C.C.: Objectclear: Complete object removal via object-efect attention (2025), https://arxiv.org/abs/2505.22636, Accessed: 1 July 2026

40. Zheng, D., Huang, Z., Liu, H., Zou, K., He, Y., Zhang, F., Gu, L., Zhang, Y., He, J., Zheng, W.S., Qiao, Y., Liu, Z.: Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness (2025), https://arxiv.org/abs/2503. 21755, Accessed: 1 July 2026

41. Zhou, S., Li, C., Chan, K.C., Loy, C.C.: Propainter: Improving propagation and transformer for video inpainting. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 10477–10486 (October 2023)

42. Zhu, Z., Li, H., Feng, X., Wu, H., Qiao, C., Yuan, J.: Georemover: Removing objects and their causal visual artifacts. In: NeurIPS (2025)

43. Zi, B., Peng, W., Qi, X., Wang, J., Zhao, S., Xiao, R., Wong, K.F.: Minimax-remover: Taming bad noise helps video object removal. arXiv preprint arXiv:2505.24873 (2025)

44. Zi, B., Zhao, S., Qi, X., Wang, J., Shi, Y., Chen, Q., Liang, B., Xiao, R., Wong, K.F., Zhang, L.: Cococo: Improving text-guided video inpainting for better consistency, controllability and compatibility. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 11067–11076 (2025)

# Supplementary Material for BeyondMasks: Evaluating Causal and Physical Consistency in Video Object Removal

Yiğit Ekin<sup>1†</sup> , Enes Sanli<sup>2,4†</sup> , Aykut Erdem<sup>2,4</sup> , Erkut Erdem<sup>3,4</sup> , and Aysegul Dundar<sup>1</sup>

<sup>1</sup> Bilkent University, Department of Computer Engineering, Ankara, Türkiye

<sup>2</sup> Koç University, Department of Computer Engineering, Istanbul, Türkiye

<sup>3</sup> Hacettepe University, Department of AI and Data Engineering, Ankara, Türkiye <sup>4</sup> KUIS AI Center, Istanbul, Türkiye

yigit.ekin@bilkent.edu.tr

Project Page · Code · Dataset

This supplementary document provides additional analyses, experimental results, and qualitative examples that complement the main paper. Sec. 1 describes the composition of the BeyondMasks dataset and the distribution of causal afterefect categories. Sec. 2 reviews representative video object removal approaches, while Sec. 3 discusses instruction-based video editing methods relevant to our evaluation setting. Sec. 4 evaluates existing image-based object removal metrics on the BeyondMasks benchmark and analyzes their limitations for video-based causal removal. Sec. 5 reports additional quantitative results comparing performance on synthetic and real subsets of BeyondMasks.

Sec. 7 analyzes the alignment between CORE scores and human evaluations and examines the robustness of the VLM-based evaluation. Sec. 8 provides the prompt template used in the CORE evaluation pipeline. Sec. 9 discusses the connection between BeyondMasks and physical commonsense evaluation. Sec. 10 highlights the diferences between BeyondMasks and existing benchmarks. Finally, Sec. 11 presents additional qualitative comparisons highlighting common failure cases.

## 1 Dataset Composition

BeyondMasks contains 180 paired video sequences, including 90 synthetic scenes and 90 real-world captures. Each sample consists of an object-present video, a temporally aligned clean background reference, per-frame object masks, and validated text prompts for instruction-driven removal. Although compact in size, the dataset is designed with controlled interventions and temporal alignment, allowing precise evaluation of whether both objects and their induced environmental efects are removed.

Synthetic sequences enable controlled modeling of diverse physical interactions, including shadows, reflections, translucent efects, and dynamic scene perturbations. Real-world captures complement these controlled scenes by introducing natural textures, lighting variations, and complex interactions that are dificult to reproduce synthetically. Together, this combination provides both controlled evaluation conditions and realistic scene complexity.

Table 1 summarizes the distribution of after-efect categories in Beyond-Masks. Categories are not mutually exclusive: a single video may contain multiple interaction types, and therefore category counts do not sum to the total number of videos. The distribution reflects the natural frequency of object-induced visual efects in real scenes rather than an artificially balanced design. For example, shadows occur in many everyday lighting conditions, while phenomena such as steam, scattering, or strong translucent interactions appear less frequently. Preserving this natural distribution makes the benchmark more representative of realistic object removal scenarios.

Table 1: Distribution of after-efect categories in BeyondMasks. Categories are not mutually exclusive; a single video may exhibit multiple interaction types.
<table><tr><td>After-Effect Category Count</td></tr><tr><td>Shadow 127</td></tr><tr><td>Reflection 61</td></tr><tr><td>Translucent effects 46 Causal physical effects 37</td></tr><tr><td></td></tr><tr><td>Light source effects 38 Steam / scattering 20</td></tr><tr><td>No explicit after-effect 6</td></tr><tr><td></td></tr></table>

![](images/bd58d85b6a4a572eb938fb74382d020dc68af36de21af00886fc23c0a67dbac6.jpg)

## 1.1 Validation of Background Preservation in Synthetic Pairs

A key requirement of the synthetic subset is that each object-present video and its corresponding clean reference represent the same underlying scene under a controlled intervention. In other words, the paired videos should difer only in the presence of the target object and in the visual consequences causally induced by that object. Establishing this property is important because unintended changes in distant background regions would make it dificult to attribute diferences between the input and reference videos to the object removal intervention itself.

Standard reconstruction metrics such as PSNR and SSIM are not well suited for validating this property when computed outside the target object mask. In our setting, the object mask deliberately covers only the visible object, while many relevant efects—including shadows, reflections, local illumination changes, steam, dust, water disturbances, and other interaction traces—remain outside the mask. These regions are expected to difer between the object-present input and the clean reference because they constitute part of the object’s causal footprint.

Instead, we validate the synthetic pairs through systematic inspection of absolute per-pixel diference maps between the object-present input and the corresponding clean reference. As illustrated in Fig. 1, meaningful diferences are concentrated around the inserted object and its associated physical footprint. For example, the diference maps may capture a cast shadow, a reflection, a local lighting variation, or a dynamic disturbance caused by the object. In contrast, static and spatially distant background regions should remain nearly unchanged.

![](images/7ae97cb0cccd8f062c300a734e539987a633efa9396c430030f6a1e00fbbe867.jpg)  
Fig. 1: Localized diferences in synthetic pairs.

## 2 Video Object Removal Methods

Video object removal is commonly formulated as a spatiotemporal inpainting problem guided by object masks. Early approaches extend image inpainting techniques to videos using optical flow, patch-based synthesis, and temporal propagation to maintain cross-frame consistency [41]. More recent deep learning methods incorporate 3D convolutional architectures, attention mechanisms, and learned temporal priors to improve visual realism and temporal coherence [44]. Representative systems such as ProPainter [41] and DifuEraser [15] combine motion-aware propagation or difusion-based synthesis to reconstruct plausible background content within masked regions.

A second line of work leverages difusion-based video foundation models for editing tasks. These models enable masked video editing through latent-space manipulation and transformer-based architectures [18, 43]. Large-scale pretraining improves perceptual realism and motion consistency across frames. However, these systems remain fundamentally mask-conditioned inpainting models: they are optimized to synthesize visually plausible background content rather than explicitly reason about or eliminate object-induced physical efects such as shadows, reflections, or illumination changes.

Another family of approaches relies on layered video decomposition. Methods such as Omnimatte [17] and its generative extension OmnimatteZero [22] model scenes as compositions of foreground layers and background content. By explicitly separating dynamic objects and their associated efects (e.g., shadows or reflections), these models support flexible video editing and compositing. Nevertheless, they are primarily designed for layer manipulation and scene editing, and they are not evaluated under standardized settings that test the complete removal of object-induced environmental efects.

Recent image inpainting research has begun to address object removal beyond simple mask completion by explicitly modeling the visual efects induced by objects. Several methods aim to jointly remove objects together with associated artifacts such as shadows, reflections, or illumination changes through efect-aware attention mechanisms or paired supervision [27,38]. Other approaches formulate object removal as a counterfactual reconstruction problem or leverage geometryaware scene representations to eliminate objects and their causal visual efects in a physically consistent manner [29, 42]. While these advances improve semantic completeness in static images, they remain limited to single-frame settings and do not address the temporal persistence of object–environment interactions in videos.

Overall, existing approaches largely treat object removal as a reconstruction problem, optimizing masked-region fidelity and perceptual plausibility. As a result, evaluation protocols typically focus on pixel-level similarity or local inpainting quality rather than verifying whether object-induced physical efects have been fully eliminated. BeyondMasks addresses this gap by enabling systematic evaluation of object-environment disentanglement and the removal of induced physical efects in temporally consistent video sequences.

## 3 Instruction-Based Video Editing and Removal Methods

Instruction-based visual editing has recently emerged as a powerful paradigm for controllable content manipulation using natural language prompts. Early work such as InstructPix2Pix [3] demonstrated that difusion models can be trained to follow textual instructions for image editing tasks, including object modification, insertion, and removal. These models enable flexible editing without requiring explicit spatial masks, instead relying on language guidance to specify the desired transformation.

Building on these ideas, subsequent work has extended instruction-conditioned editing from images to videos. Methods such as InstructVid2Vid [20] adapt difusion-based editing models to the temporal domain by introducing mechanisms that propagate edits consistently across frames. Similarly, InstructVideo [35] improves instruction following by aligning video difusion models with human feedback, enabling more accurate interpretation of natural language editing commands. These approaches illustrate the growing capability of video foundation models to perform flexible, language-driven video manipulation.

Within this paradigm, object removal can be performed using natural language instructions (e.g., “remove the cup from the table”) rather than explicit masks. Recent video editing systems therefore increasingly support instructiondriven removal alongside traditional mask-conditioned editing. For example, VACE [11] unifies multiple editing modes within a single architecture, including masked editing, reference-based generation, and instruction-guided video manipulation. Such frameworks highlight the trend toward integrating language interfaces into video editing pipelines, allowing users to specify complex edits using natural language prompts.

Despite these advances, evaluating instruction-based object removal remains challenging. Existing studies typically rely on qualitative inspection or generalpurpose video generation metrics such as Fréchet Video Distance (FVD) [16] or perceptual similarity measures such as LPIPS [37]. While these metrics capture perceptual realism and distributional similarity, they do not explicitly verify whether the target object and its associated physical efects (e.g., shadows, reflections, or illumination changes) have been fully removed.

BeyondMasks addresses this limitation by enabling structured evaluation of instruction-driven removal alongside mask-conditioned approaches. The dataset provides temporally aligned pairs of object-present videos and corresponding clean background references, allowing edited outputs to be compared directly with the ground-truth interventional scene state. This setup enables systematic analysis of whether instruction-based editing systems remove both the target object and the physical efects it induces in the surrounding environment.

## 4 Image Based Object Removal Metrics

In addition to the evaluation metrics reported in the main manuscript, we analyze model performance using two representative image-based object removal metrics: ReMOVE [4] and CFD Score [33]. Both metrics were originally designed to evaluate object removal quality in static images. We apply them frame-wise to our video benchmark to investigate whether image-level evaluation metrics reflect removal quality in the presence of object-induced physical efects.

To compute these metrics on videos, we sample frames from each sequence with a stride of 15 to balance temporal coverage and computational cost. The metric is computed independently for each sampled frame and averaged across frames to obtain a video-level score. Finally, scores are aggregated across videos belonging to the same causal after-efect category to produce category-level results.

Tables 2 and 3 report the performance of all evaluated methods across the diferent after-efect categories in BeyondMasks using ReMOVE and CFD, respectively. These metrics were originally proposed for object-aware image evaluation, and thus they are not designed to capture temporally consistent causal efects in videos. As a result, their rankings do not necessarily reflect whether object-induced environmental interactions—such as shadows, reflections, or illumination changes—have been correctly removed.

Consistent with the observations in the main paper, the results indicate that image-based metrics often assign similar scores to methods that exhibit substantially diferent levels of causal removal quality. Because ReMOVE and CFD operate as no-reference metrics, they primarily evaluate semantic plausibility and object absence within individual frames rather than verifying, against reference videos, whether object-induced environmental efects have been removed. As a result, methods that leave residual reflections, shadows, or illumination changes may still obtain competitive scores if the edited frame appears visually coherent.

More specifically, ReMOVE tends to behave like a coarse scene-level plausibility metric, producing very similar scores across methods even when their actual removal quality difers substantially. CFD focuses more on visual compatibility within the edited frame and can therefore assign high scores to outputs that remain harmonious with the surrounding scene despite incomplete removal. Consequently, methods that leave clear residual object traces or after-efects may still appear competitive under CFD if the frame remains visually coherent. These observations highlight an important limitation of existing imagebased metrics when applied to video object removal tasks involving complex object–environment interactions.

Table 2: Evaluation using the ReMOVE metric on BeyondMasks across causal after-efect categories. Higher values indicate better performance. Type denotes the supervision modality: T indicates text-instruction-based editing, M indicates mask-guided removal, and T+M denotes methods supporting both inputs. Bold numbers indicate the best score in each column. Bold numbers indicate the best score in each column.
<table><tr><td>Method</td><td></td><td></td><td></td><td></td><td></td><td>Type Shadow Reflection Light Steam Translucent Causal</td><td></td></tr><tr><td>Lucy Edit (arXiv) [23]</td><td>T</td><td>0.659</td><td>0.647</td><td>0.586</td><td>0.636</td><td>0.675</td><td>0.669</td></tr><tr><td>VACE (ICCV 2025) [11]</td><td>T+M</td><td>0.658</td><td>0.645</td><td>0.597</td><td>0.630</td><td>0.679</td><td>0.666</td></tr><tr><td>CoCoCo (AAAI 2025) [44]</td><td>T+M</td><td>0.662</td><td>0.651</td><td>0.593</td><td>0.648</td><td>0.685</td><td>0.661</td></tr><tr><td>Gen. Omnimatte (CVPR 2025) [13]</td><td>T+M</td><td>0.676</td><td>0.663</td><td>0.622</td><td>0.638</td><td>0.705</td><td>0.684</td></tr><tr><td>ProPainter (ICCV 2023) [41]</td><td>M</td><td>0.668</td><td>0.669</td><td>0.614</td><td>0.636</td><td>0.701</td><td>0.665</td></tr><tr><td>DiffuEraser (arXiv) [15]</td><td>M</td><td>0.664</td><td>0.667</td><td>0.615</td><td>0.647</td><td>0.696</td><td>0.679</td></tr><tr><td>MiniMax (NeurIPS 2025) [43]</td><td>M</td><td>0.667</td><td>0.655</td><td>0.603</td><td>0.636</td><td>0.690</td><td>0.686</td></tr><tr><td>ROSE (NeurIPS 2025) [18]</td><td>M</td><td>0.678</td><td>0.670</td><td>0.595</td><td>0.636</td><td>0.704</td><td>0.690</td></tr><tr><td>OmnimatteZero (SIGGRAPH 2025) [22]</td><td>M</td><td>0.669</td><td>0.611</td><td>0.601</td><td>0.617</td><td>0.699</td><td>0.698</td></tr></table>

Table 3: Evaluation using the CFD metric on BeyondMasks across causal after-efect categories. Higher values indicate better performance. Type denotes the supervision modality: T indicates text-instruction-based editing, M indicates maskguided removal, and T+M denotes methods supporting both inputs. Bold numbers indicate the best score in each column. Bold numbers indicate the best score in each column.
<table><tr><td></td><td colspan="7"></td></tr><tr><td>Method</td><td></td><td></td><td>Type Shadow Reflection Light Steam Translucent Causal</td><td></td><td></td><td></td><td></td></tr><tr><td>Lucy Edit (arXiv) [23]</td><td>T</td><td>0.677</td><td>0.652</td><td>0.667</td><td>0.763</td><td>0.805</td><td>0.668</td></tr><tr><td>VACE (ICCV 2025) [11]</td><td>T+M</td><td>0.558</td><td>0.585</td><td>0.520</td><td>0.631</td><td>0.670</td><td>0.594</td></tr><tr><td>CoCoCo (AAAI 2025) [44]</td><td>T+M</td><td>0.326</td><td>0.358</td><td>0.368</td><td>0.317</td><td>0.450</td><td>0.292</td></tr><tr><td>Gen. Omnimatte (CVPR 2025) [13]</td><td>T+M</td><td>0.285</td><td>0.350</td><td>0.359</td><td>0.376</td><td>0.404</td><td>0.260</td></tr><tr><td>ProPainter (ICCV 2023) [41]</td><td>M</td><td>0.253</td><td>0.248</td><td>0.341</td><td>0.275</td><td>0.301</td><td>0.249</td></tr><tr><td>DiffuEraser (arXiv) [15]</td><td>M</td><td>0.291</td><td>0.314</td><td>0.396</td><td>0.391</td><td>0.332</td><td>0.260</td></tr><tr><td>MiniMax (NeurIPS 2025) [43]</td><td>M</td><td>0.285</td><td>0.318</td><td>0.351</td><td>0.404</td><td>0.301</td><td>0.257</td></tr><tr><td>ROSE (NeurIPS 2025) [18]</td><td>M</td><td>0.287</td><td>0.304</td><td>0.343</td><td>0.410</td><td>0.336</td><td>0.306</td></tr><tr><td>OmnimatteZero (SIGGRAPH 2025) [22]</td><td>M</td><td>0.359</td><td>0.360</td><td>0.361</td><td>0.422</td><td>0.421</td><td>0.354</td></tr></table>

## 5 Quantitative Results on Synthetic and Real Subsets

Table 4 presents split-wise quantitative results on the synthetic and real subsets of BeyondMasks. The synthetic subset contains 90 videos generated under controlled intervention settings, while the real subset consists of 90 tripod-captured scenes. Reporting results separately allows us to assess whether trends observed in controlled environments transfer to real-world scenarios. Overall, performance patterns remain largely consistent across the two splits: methods that perform strongly on synthetic data generally remain competitive on real videos as well.

Table 4: Split-wise quantitative evaluation on BeyondMasks. Performance of representative video object removal and image object removal methods on the synthetic and real subsets of the dataset. Each entry reports Synthetic / Real results for reconstruction metrics (PSNR, SSIM, LPIPS) and the proposed CORE evaluation scores (CORE-OS and CORE-AES), which are measured on a discrete 1–5 scale. Image-based methods (OmniPaint and Object Clear) are applied frame-wise by processing one frame every 15 frames. Bold indicates the best value for each metric across all methods.
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td></td><td>LPIPS↓</td><td></td><td>CORE-OS↑</td><td></td><td>CORE-AES↑</td></tr><tr><td>Lucy Edit (arXiv) [23]</td><td>18.8356</td><td>18.3282 0.7529</td><td>0.6942</td><td>0.2228</td><td>0.3285</td><td>1.069</td><td>1.813</td><td>1.276 1.824</td></tr><tr><td>VACE (ICCV 2025) [11]</td><td>18.9362</td><td>18.6913 0.7547</td><td>0.7236</td><td>0.2209</td><td>0.3010</td><td>1.415</td><td>2.176</td><td>1.404 / 1.989</td></tr><tr><td>CoCoCo (AAAI 2025) [44]</td><td>21.6499 20.1249</td><td>0.7625</td><td>0.6950</td><td>0.1962</td><td>0.2809</td><td>1.670</td><td>2.297</td><td>1.875 2.209</td></tr><tr><td>Gen. Omnimatte (CVPR 2025) [13]</td><td>25.0218 22.6120</td><td>0.8561</td><td>0.7794</td><td>0.1369</td><td>0.2562</td><td>3.865</td><td>3.670</td><td>2.966 3.055</td></tr><tr><td>ProPainter (ICCV 2023) [41]</td><td>24.6675 22.6555</td><td>0.8816</td><td>0.8060</td><td>0.1202</td><td>0.2044</td><td>3.326</td><td>2.800</td><td>2.371 / 2.361</td></tr><tr><td>DiffuEraser (arXiv) [15]</td><td>25.3617</td><td>25.1622 0.8932</td><td>0.8621</td><td>0.0942</td><td></td><td>/0.12903.640 / 3.344</td><td></td><td>2.461 / 2.794</td></tr><tr><td>MiniMax (NeurIPS 2025) [43]</td><td>24.1725 22.2273</td><td>0.8597</td><td>0.7844</td><td>0.1164</td><td>/0.2195</td><td>3.775 / 3.506</td><td></td><td>2.382 2.692</td></tr><tr><td>ROSE (NeurIPS 2025) [18]</td><td>24.9255 22.6790</td><td>0.8558</td><td>0.7786</td><td>0.1128</td><td>0.2191</td><td>4.068</td><td>3.505</td><td>2.932 2.912</td></tr><tr><td>OmnimatteZero (SIGGRAPH 2025) [22]</td><td>22.9243 23.0160</td><td>0.8225</td><td>0.8040</td><td>0.1420</td><td>0.1834</td><td>3.438</td><td>2.945</td><td>2.360 2.638</td></tr><tr><td>OmniPaint (ICCV 2025) [34]</td><td>23.6996 24.2534</td><td>0.8529</td><td>0.8436</td><td>0.1121</td><td>0.1512</td><td>3.750</td><td>3.125</td><td>3.125 2.182</td></tr><tr><td>Object Clear (CVPR 2026) [39]</td><td>25.5278 24.5341</td><td>0.8720</td><td>0.8532</td><td>0.0936</td><td>0.1326</td><td>4.325</td><td>3.725</td><td>3.022 3.015</td></tr></table>

Considering reconstruction-based metrics, DifuEraser achieves the strongest overall performance, obtaining the highest PSNR and SSIM values together with the lowest LPIPS on both subsets. However, high reconstruction fidelity does not necessarily imply correct object removal. To evaluate removal correctness, we report the proposed CORE scores: ObjectScore (CORE-OS) and AfterEffectScore (CORE-AES). These scores are measured on a discrete 1–5 scale and reflect VLM-based judgments of object disappearance and the removal of objectinduced efects, respectively.

The CORE scores reveal diferent performance trends than reconstruction metrics. For example, ROSE achieves the highest CORE-OS on the synthetic subset (4.068), indicating strong object disappearance under controlled conditions. However, its score decreases on the real subset (3.505), suggesting that removal becomes more challenging in scenes with richer visual interactions. Generative Omnimatte and DifuEraser obtain the highest CORE-AES values among video-based methods, although the absolute scores remain moderate on the 1–5 scale, highlighting the dificulty of eliminating subtle physical after-efects such as shadows, reflections, and secondary interactions.

To provide additional reference points, we also evaluate two recent imagebased object removal methods, OmniPaint [34] and ObjectClear [39]. Since these models operate on single images rather than videos, we apply them frame-wise by processing one frame every 15 frames. Interestingly, these methods achieve competitive reconstruction metrics and relatively strong CORE scores on individual frames. For instance, ObjectClear obtains the highest CORE-OS values across both subsets, indicating that modern image editing models can efectively remove objects and their associated efects at the frame level.

Nevertheless, image-based methods lack explicit temporal modeling and therefore cannot enforce consistency across frames. While frame-wise editing can locally remove objects and their visual artifacts, dedicated video models remain necessary to maintain temporal coherence and handle persistent object–environment interactions over time.

Overall, the split-wise results indicate that the synthetic subset captures performance trends that largely transfer to real scenes, while the real subset confirms that causally consistent object removal remains challenging in visually complex environments.

## 6 Why Current Removal Methods Fail on Causal After-Efects

The qualitative examples reveal that the main limitation of current video object removal methods is not simply imperfect reconstruction inside the object mask. Rather, most failures arise because existing methods implicitly formulate removal as a spatially localized inpainting problem. Given a mask indicating the visible object, the model is typically optimized to synthesize plausible content within or near that masked region. However, an object’s visual influence is often distributed beyond its visible boundary. Shadows, reflections, illumination changes, dust trails, water disturbances, steam, and other interaction traces may extend far outside the mask and can persist over time even after the object itself is no longer visible.

The dificulty varies across after-efect categories. Cast shadows are relatively more tractable because they are often spatially adjacent to the object and can be approximated as local photometric changes. Nevertheless, soft or temporally varying shadows may still extend beyond the model’s efective editing region. Reflections are more challenging because their appearance depends on scene geometry, material properties, viewpoint, and the motion of both the object and the camera. Dynamic efects such as steam, dust, smoke, water ripples, or displaced particles are particularly dificult because they can spread over large regions and evolve according to temporal physical processes rather than remaining attached to a fixed object boundary.

Translucent objects introduce an additional ambiguity. Unlike opaque objects, they do not fully block the background; instead, each observed pixel may contain a mixture of foreground appearance and partially visible background content. A binary mask treats these mixed pixels as fully unavailable, discarding useful observations of the background behind the object. As a result, a maskconditioned method must hallucinate details that are only partially occluded in the original frame. Recovering these details reliably requires exploiting temporal redundancy, since background content may be more clearly visible in earlier or later frames as the object or camera moves.

These observations suggest three design principles for causally consistent video object removal. First, models require scene-level representations rather than relying solely on mask-local features, because the relevant evidence may be spatially distant from the target object. Second, removal systems should preserve and aggregate partially observed background information across time, especially for translucent objects and thin structures. Third, models should explicitly account for object-conditioned lighting and temporally persistent efects instead of treating every out-of-mask change as fixed background. Together, these requirements indicate that causally consistent removal is fundamentally a counterfactual scene reasoning problem rather than a conventional masked reconstruction task.

## 7 Human Alignment and Cross-VLM Robustness of the CORE Evaluation Metric

To assess how well automatic metrics reflect human perception of video object removal quality, we measure the correlation between metric outputs and human ratings. We report Spearman rank correlations between metric scores and the average human annotations across videos for both evaluation dimensions: ObjectScore and AfterEfectScore. The results are summarized in Table 5.

Table 5: Correlation with human evaluation. Spearman correlation between automatic metrics and human ratings for ObjectScore and AfterEfectScore. Pixel-based metrics (PSNR, SSIM, LPIPS) are compared with VLM-based CORE evaluations using two judge models (Gemini and GPT). The Inter Rater row reports agreement between human annotators, serving as a reference level for achievable alignment.
<table><tr><td>Metric</td><td colspan="2">ObjectScore AfterEffectScore</td></tr><tr><td>PSNR</td><td>0.15</td><td>0.28</td></tr><tr><td>SSIM</td><td>-0.14</td><td>-0.10</td></tr><tr><td>LPIPS</td><td>-0.02</td><td>0.10</td></tr><tr><td>COREGemini</td><td>0.62</td><td>0.73</td></tr><tr><td>COREGPT</td><td>0.60</td><td>0.76</td></tr><tr><td>Inter Rater</td><td>0.69</td><td>0.79</td></tr></table>

Standard pixel-based reconstruction metrics exhibit weak correlation with human judgments. PSNR shows only limited agreement with human ratings, while SSIM and LPIPS display near-zero or negative correlations for both scoring dimensions. These results suggest that reconstruction fidelity alone does not reliably capture the aspects of removal quality that humans consider important. In particular, pixel similarity does not necessarily reflect whether the target object has been fully removed or whether object-induced environmental efects, such as shadows, reflections, or illumination changes, have been eliminated. Consequently, methods with similar reconstruction scores may still difer substantially in their causal correctness.

In contrast, the proposed CORE evaluation demonstrates substantially stronger agreement with human assessments. $\mathrm { { C O R E } _ { \mathrm { { G e m i n i } } } }$ achieves correlations of 0.62 and 0.73 for ObjectScore and AfterEfectScore, respectively, while CORE<sub>GPT</sub> obtains 0.60 and 0.76. The consistency of these results across two independent VLM backbones indicates that the efectiveness of CORE is not tied to a specific judge model. Instead, the improvement appears to stem from the structured evaluation protocol itself, which explicitly compares edited videos against aligned reference backgrounds and separates object removal from the elimination of causal after-efects.

We further examine the stability of the VLM-based evaluation by repeating the Gemini-based scoring multiple times and measuring agreement across runs using the intraclass correlation coeficient (ICC). The resulting ICC values are 0.808 for ObjectScore and 0.882 for AfterEfectScore, indicating strong run-torun consistency. These results suggest that CORE produces stable and reliable judgments, particularly for evaluating the removal of residual causal traces.

## 8 CORE Evaluation Prompt

CORE evaluates object removal using a structured prompt that guides a visionlanguage model (VLM) to compare the edited video against the aligned reference background. The same prompt template is used for all samples to ensure consistent evaluation across methods. Only two fields are instantiated for each example: {fg\_object}, which specifies the target object to be removed, and {focus\_effect}, which indicates the relevant after-efect categories (e.g., shadow, reflection, illumination change).

For each evaluation instance, the VLM receives three videos: (1) the INPUT video containing the target object and its associated efects, (2) the GT video representing the ideal background with both the object and its efects removed, and (3) the RESULT video produced by the evaluated method. The prompt instructs the VLM to perform structured reasoning before assigning scores. Specifically, the model first identifies the object and its efects in the INPUT video, then analyzes the background appearance in the GT reference, and finally compares the RESULT video against the GT to determine whether the object and its induced efects have been successfully removed.

CORE decomposes evaluation into two complementary scores: ObjectScore, which measures the completeness of object removal and plausibility of the reconstructed background, and AfterefectScore, which measures whether objectinduced visual efects such as shadows, reflections, or lighting changes have been eliminated. Both scores follow a 1-5 scale, where higher values indicate better removal quality. Requiring structured reasoning encourages the VLM to explicitly analyze object disappearance and residual environmental efects before producing the final scores.

Listing 2.1 shows the prompt template used in our evaluation pipeline.

## Listing 2.1: Prompt template used for CORE-based evaluation.

1 You are an expert computer vision judge evaluating a video object removal task using THREE   
separate videos.   
2 The three videos provided represent:   
3 INPUT Video: The original reference video where the target object and its aftereffects ARE   
PRESENT.   
4 GT Video (Ground Truth): The IDEAL result video where the object and aftereffects are   
perfectly removed, showing the true background across all frames.   
5 RESULT Video: Our method’s output video. Your goal is to evaluate how well the RESULT   
video removes the target object and reconstructs a plausible, natural background.   
7 Context for this sample:   
8 Target object (present in INPUT; absent in GT): {fg\_object}   
9 Aftereffect types to check: {focus\_effect}   
11 EVALUATION PROCESS (THINKING STEPS)   
12 Before scoring, you MUST output a step-by-step reasoning process using the following   
structure:   
13 Step 1: Target Identification: Watch the INPUT video and identify the exact appearance,   
movement, and location of the Target Object ({fg\_object}) and its Aftereffects ({   
focus\_effect}).   
14 Step 2: Expected Background Pattern (GT Analysis): Watch the GT video. Understand the   
general context, texture, and lighting of the background where the object used to be   
15 Step 3: Output vs GT Comparison (Naturalness Check): Compare the RESULT video against the   
GT. Evaluate the inpainted area. Does the filled background look natural and fit the   
surrounding scene harmoniously? Did the model successfully create a clean   
background, or did it hallucinate a new foreground object?   
16 Step 4: Error Categorization: Determine which identified differences are related to the   
main Target Object removal, and which are related to the Aftereffect removal. Find   
the specific flaws based on the rubrics below.   
18 SCORING RUBRICS (1-5 Scale)   
19 HUMAN-LIKE JUDGING GUIDELINES:   
20 - Humans care most about SEMANTIC removal. If the identifiable features of the target   
object are gone, humans consider the removal mostly successful (Score 4), EVEN IF a   
blurry smudge or a dark moving blob is left behind.   
21 - A score of 3 should be reserved for partial temporal failures or when a small piece of   
the object remains.   
22 - Punish heavily (Scores 1-2) ONLY if the object is fully visible for a long time, if the   
model hallucinates a completely new recognizable object, or if the artifact left   
behind is grotesquely out of place.   
24 ObjectScore Rubric (How well the main target object was removed):   
25 5 Points: Perfect or Near Perfect. The object is completely removed. The inpainted   
background seamlessly matches the GT. Minor, almost unnoticeable imperfections are   
acceptable.   
26 4 Points: Good (Forgiving of Smudges). The core semantic features of the target object are   
completely gone. The inpainted area might contain noticeable smudges, blurriness,   
or dark moving blobs, but it does not severely break the overall geometry of the   
scene. The human eye forgives these as long as the object itself is unrecognizable.

27 3 Points: Fair (Partial/Distracting). The removal is flawed but not a total failure. Use   
this if: 1) A small but recognizable part of the object remains. 2) Temporal failure   
: The object is removed, but suddenly reappears for a few frames. 3) The object is   
gone, but the artifact left behind is highly distracting and physically illogical   
for the scene.   
28 2 Points: Poor. Huge remnants of the target object are still clearly visible and moving.   
OR the model hallucinated a completely new, identifiable foreground object that   
completely ruins the background.   
29 1 Point: Fail. The object is practically untouched and fully visible for the majority of   
the video, or the generated artifacts cause extreme, video-breaking corruption.   
31 AftereffectScore Rubric (How well object-caused effects like shadows/reflections were   
removed):   
32 5 Points: Perfect. The aftereffect is completely removed, matching the GT seamlessly.   
33 4 Points: Good. Aftereffect is successfully removed and the area looks natural, even if   
lighting/texture slightly differs from the exact GT.   
34 3 Points: Fair. Aftereffect area is noticeably blurry, smudged, or improperly lit compared   
to the rest of the scene.   
35 2 Points: Poor. Severe visual artifacts in the aftereffect region.   
36 1 Point: Fail. The aftereffect is still clearly visible in the RESULT video, completely   
ignored by the removal process.   
38 OUTPUT FORMAT   
39 You MUST format your response exactly like this:   
40 Reasoning:   
41 Step 1: [Your reasoning]   
42 Step 2: [Your reasoning]   
43 Step 3: [Your reasoning]   
44 Step 4: [Your reasoning]   
45 AftereffectScore: X, ObjectScore: Y

## 9 Relation to Physical Commonsense Evaluation

BeyondMasks is closely related to recent eforts aimed at evaluating physical commonsense understanding in video models. Benchmarks such as VideoPhy [1], VideoPhy-2 [2], and related evaluation frameworks have shown that videos that appear visually convincing can still violate basic physical expectations. These studies highlight a recurring gap between perceptual realism and genuine physical understanding in video generation systems.

Our findings in video object removal reveal a similar phenomenon from a different perspective. Many methods produce visually plausible outputs when evaluated with standard reconstruction or perceptual metrics, yet still fail to remove object-induced after-efects such as shadows, reflections, illumination changes, or motion traces. In these cases, the edited video may appear superficially realistic, but the model has not correctly accounted for how the removed object interacts with the surrounding scene.

From this perspective, BeyondMasks can be viewed as evaluating a form of counterfactual physical reasoning. Rather than asking whether a generated video depicts a physically plausible event, our benchmark asks whether a model can correctly infer how a scene would appear if a particular object had never been present. Removing an object is therefore not merely a local appearance editing problem: the object may influence nearby pixels indirectly through lighting, geometry, material interactions, and temporal dynamics. Correctly editing the scene requires reasoning about these dependencies and eliminating their visual consequences.

This setting complements existing physical commonsense benchmarks. Prior benchmarks primarily evaluate whether models can generate videos that obey intuitive physical behavior. BeyondMasks instead evaluates whether models can edit an existing scene in a way that remains physically and causally consistent after an intervention. By providing paired object-present videos and clean reference backgrounds, the benchmark allows direct comparison between the edited result and the expected counterfactual scene state.

Together, these perspectives suggest that studying physical reasoning in video models should not be limited to forward generation tasks alone. Counterfactual editing tasks, such as object removal with physically consistent outcomes, ofer a complementary diagnostic for understanding whether models capture the causal relationships between objects and their surrounding environment.

## 10 Diferences from Existing Benchmarks

Existing video object removal benchmarks primarily focus on spatial reconstruction or segmentation quality, and therefore do not fully evaluate causal objectenvironment disentanglement. In realistic scenes, removing an object requires not only filling the masked region but also accounting for secondary visual efects induced by the object, such as shadows, reflections, illumination changes, and dynamic traces. Evaluating whether these efects have been correctly removed requires temporally aligned reference videos that represent the counterfactual scene state after the object has been removed.

Traditional video segmentation datasets such as DAVIS [19] and YouTube-VOS [30] provide high-quality object masks and have been widely used to evaluate video inpainting methods. However, these datasets do not include aligned clean reference videos, making it dificult to determine whether an object removal method has successfully eliminated both the object and its associated environmental efects. As a result, evaluation is typically based on perceptual similarity or local reconstruction quality rather than causal correctness.

More recent datasets attempt to introduce paired references for evaluation. For example, HQVI [5] provides paired videos for video inpainting evaluation. However, the dataset construction largely focuses on spatial reconstruction, and it does not explicitly model secondary object-induced efects. Consequently, it primarily evaluates the ability of a model to reconstruct missing regions rather than its ability to remove the broader physical consequences of an object’s presence.

Among existing benchmarks, ROSE-Bench [18] is closest in spirit to our setting because it explicitly considers object-related side efects in synthetic paired videos. This represents an important step toward more physically grounded evaluation of object removal methods. ROSE-Bench constructs its dataset using game-engine-based rendering pipelines, which allow precise control over scene configuration and object interactions. Such simulation pipelines are valuable for controlled analysis, as they provide consistent geometry, lighting, and motion patterns across paired sequences.

BeyondMasks follows a complementary design philosophy aimed at increasing visual and causal diversity. First, the dataset combines synthetic paired videos with real-world paired captures, enabling evaluation under both controlled interventions and realistic visual conditions. Second, our synthetic subset is generated using modern video generation models rather than game-engine simulation, which introduces a broader range of scene appearances, motion patterns, and visual variability. Third, the dataset covers a wider variety of object-induced after-efects, including photometric efects (e.g., shadows and reflections) as well as dynamic and contact-driven traces. Fourth, the object masks intentionally exclude these secondary efects, requiring models to infer and remove them through scene reasoning rather than explicit supervision. Finally, BeyondMasks supports both mask-guided removal and instruction-driven editing, making it suitable for evaluating modern video editing and video foundation models.

Overall, the key distinction of BeyondMasks is its emphasis on causal diversity and interventional realism. While prior benchmarks have been valuable for studying spatial reconstruction or specific classes of side efects, our dataset is designed to better reflect the full complexity of object removal in realistic scenes, where objects interact with their environment through lighting, geometry, materials, and temporal dynamics.

## 11 Additional Qualitative Results

We present additional qualitative examples illustrating common failure modes of current video object removal systems. While many approaches successfully remove the visible object, they often fail to eliminate the physical efects induced by the object in the surrounding environment. These efects include shadows, reflections, translucent distortions, lighting changes, and dynamic interaction traces. Such artifacts remain visible even after the object itself disappears, revealing incomplete removal of the object’s causal footprint. The examples in Fig. 2 and Fig. 3 highlight these failure patterns in both real-world and synthetic scenes.

![](images/2656a8c2f3ca0ecd8e1fa407a781c9a1aa64f7db09ea1e408bd541fa9cdd4143.jpg)  
Fig. 2: Failure cases in real-world videos. For each example, the first row shows frames from the input video and the second row shows the output of a representative mask-based removal method. Although the target object is largely removed, residual object-induced efects remain visible. Red dashed boxes highlight typical artifacts such as persistent shadows, reflections in mirrors or glass surfaces, steam or smoke traces, and lighting inconsistencies that reveal incomplete removal of object-environment interactions.

![](images/7a2ddc461d8e46fde15213ea7cf01e73e418bd672d4cb0b2458836ff946c403c.jpg)  
Fig. 3: Failure cases in synthetic scenes. The first row shows input frames and the second row shows the corresponding outputs after object removal. Despite successful removal of the foreground object, the edited results often retain interaction efects such as fence shadows, motion-induced dust trails, illumination artifacts, or water disturbances. Red dashed boxes indicate regions where the model reconstructs a visually plausible background but fails to remove the full causal footprint of the object.