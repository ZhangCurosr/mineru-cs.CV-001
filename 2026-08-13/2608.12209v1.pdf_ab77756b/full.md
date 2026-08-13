![](images/956e0960629fcbbe8182e3e3bed814c0308b2e22fc2d26788d65ecf23dd92e3a.jpg)

# Enhancing Visual Understanding at Zero Inference Overhead via Decoupled Embedding Prediction

Zhongbin Guo\*<sup>†</sup> Jiahao Xie\* Dongling Xiao Qianle Wang Ruiqi Lu Xiaomin He<sup>†</sup> Wanxuan Sun<sup>B</sup> Cheng Yang<sup>B</sup>

ByteDance

## Abstract

While Multimodal Large Language Models (MLLMs) have achieved remarkable progress, visual understanding and generation are typically treated as divergent objectives. Existing unified frameworks often rely on discrete visual tokenization or diffusion objectives whose generative targets differ from the continuous representations consumed by visual understanding models, making direct transfer to enhance existing pretrained MLLMs non-trivial. In this work, we present GAS, a generation-guided training framework that reinterprets visual generation as auxiliary supervision for representation learning. Concretely, GAS adapts Next Embedding Prediction (NEP) as a cross-modal generation paradigm within a decoupled Mixture-of-Transformers (MoT) architecture. By maintaining a shared lower trunk and parallel upper layers, GAS lets generation losses enrich the shared visual pathway with finer spatial precision and stronger visual retention while shielding the upper understanding layers from direct generation gradients. To maximize this synergy, we further construct highly correlated generation tasks that demand deep cognitive grounding rather than generic synthesis alone. Across model scales and training stages, GAS improves aggregate multimodal understanding, with its most reliable gains on perception and spatial comprehension. Crucially, because the auxiliary generation branch is discarded after training, these gains incur zero inference overhead. Extensive controlled comparisons and representation-level analyses further clarify when and why generation-guided training benefits understanding, and demonstrate the feasibility of generation-guided training as a practical route to stronger multimodal understanding.

<sup>#</sup> Email: guozhongbin66@gmail.com

} Publication date: August 13, 2026

## 1. Introduction

Multimodal large language models (MLLMs) have achieved remarkable progress in visual understanding tasks, including image captioning, visual question answering, and visual grounding [1–3]. The prevailing paradigm encodes visual inputs through a vision encoder and projects the resulting representations into token space of large language model (LLM), which then performs reasoning and generates textual responses in an autoregressive manner [4, 5]. Although this architecture has proven remarkably effective for semantic-level comprehension, its training objective – next-token prediction (NTP) over text – provides only indirect supervision for visual perception. The model is trained to talk about images, yet never to learn from them because its text objective does not directly supervise pixel- or region-level visual structure.

![](images/27ff16ad876fdf1d84bf3927d317b8704a3a3744e06842eb71eeed71e37309ec.jpg)  
Figure 1: Overview ofour generation-guided trainingframework. Given multimodal inputs, the understanding branch processes visual tokens from shared ViT encoder and predicts text via standard NTP paradigm. The generation branch, operating on independent MoT transformer layers, receives intermediate hidden states from the understanding backbone and predicts target image embeddings via Next Embedding Prediction (NEP). At inference time, the generation branch is entirely discarded.

This text-only supervision introduces limitations from two complementary perspectives. On one hand, natural language inherently suffers from an expressiveness bottleneck: it cannot precisely encode fine-grained visual details such as exact spatial relationships, pixel-level object boundaries, physical interaction dynamics, or subtle appearance variations [6, 7]. When the sole training signal is text prediction, the model’s visual representations are shaped only by what language can articulate, leaving the richer geometric, structural, and relational information latent and underutilized. On the other hand, there exist abundant visually intensive tasks like segmentation and image editing. Existing VLM architectures are structurally unable to consume such data for training, thereby forfeiting a vast reservoir of fine-grained visual supervision that could otherwise cultivate deeper perception and complex visual understanding capabilities. Furthermore, empirical evidence reveals that visual information progressively attenuates in deeper LLM layers during reasoning [8, 9], precisely because the text-centric objective provides no mechanism to encourage persistent retention of visual features throughout the inference chain.

A natural hypothesis emerges: if a model is additionally trained to predict visual content, such as pixellevel segmentation masks, edited images, or visual reasoning traces, then its internal representations may be encouraged to retain richer and more precise visual information that can benefit understanding. This intuition has motivated the recent wave of unified multimodal models (UMMs) that jointly train understanding and generation [10–13]. However, directly adopting the UMM paradigm to enhance understanding exposes two unresolved tensions:

First, current UMMs retain generation-side parameters at inference time [13, 14], incurring latency and memory costs that hinder deployment. Moreover, generation objectives in these systems are typically optimized for synthesis quality rather than for maximizing benefit to comprehension [15]. Understanding performance can therefore plateau or even degrade relative to MLLMs of comparable scale, suggesting that simple joint training does not reliably transfer generative learning into understanding gains.

Second, existing work typically reports aggregate benchmark numbers after introducing generation data, without dissecting the underlying mechanisms. Critical questions remain open: Do generation and understanding objectives inherently conflict? Which intermediate representations are most amenable to generative supervision for “feeding back” into understanding performance? Which types of generation tasks enhance which facets of comprehension? And through what pathways does generation training confer its advantages?

In this work, we propose a generation-guided training framework that resolves both tensions. Our core idea is to leverage generation tasks purely as training-time auxiliary supervision for understanding, rather than as an end goal. Architecturally, as shown in Figure 1, we adopt a Mixture-of-Transformers (MoT) design that introduces a dedicated generation branch with independent transformer parameters alongside the understanding branch, so that generation gradients refine shared visual representations without directly perturbing understanding-side parameters. The generation branch employs the Next Embedding Prediction (NEP) paradigm to predict continuous visual embeddings autoregressively in the representation space consumed by the LLM backbone, without requiring a pixel decoder or discrete visual de-tokenizer. At inference time, the generation branch is entirely discarded, yielding strictly zero additional cost during inference. This deployment benefit is distinct from training cost: GAS uses approximately 11.6% more GPU-hours than the corresponding baseline.

On the data side, we systematically construct about 10M generation samples across five primary categories and 15 subtasks, covering a spectrum from pixel-level perception to high-level reasoning, and develop an automated synthesis pipeline that produces high-quality training data without manual annotation. This framework allows us to conduct systematic analyses to elucidate how generation enhances complex visual understanding which leads to several intriguing findings on generationunderstanding conflict, task-specific ablation and representation-level diagnostics.

We finally present the GAS model family, validating the framework on 2B- and 4B-parameter backbones. GAS improves aggregate visual understanding at both scales, while the per-benchmark transfer is task-dependent. Our main contributions are as follows:

• We combine continuous NEP supervision with a removable MoT generation branch, transferring generation-side learning to the deployed understanding model at zero inference overhead.

• We construct diverse generation task categories with an automated data synthesis pipeline, and show that tasks with stronger latent correlations to understanding deliver the largest gains, while combining them yields further, complementary improvements on perception- and reasoningoriented benchmarks.

• We provide systematic analyses of parameter isolation, layer-wise supervision injection, per-task contribution, representation-level diagnostics, matched-budget controls and repeated-run statistics, clarifying when and how generation enhances understanding.

## 2. Methodology

Our framework is designed around a single guiding principle: generation tasks should serve as training-time auxiliary supervision that enriches the model’s visual representations, without altering the inference-time architecture or cost. To realize this, we introduce three tightly integrated components: (1) Next Embedding Prediction (NEP) paradigm that formulates image generation as autoregressive embedding prediction within the same representation space used for understanding; (2) Mixture-of-Transformers (MoT) architecture that decouples generation and understanding parameters while preserving a shared visual representation pathway; and (3) systematic multi-type generation data construction strategy that maximizes the complementary benefits of diverse visual tasks.

## 2.1. Next Embedding Prediction (NEP)

Existing unified models typically adopt either discrete visual tokenization [16, 17] or diffusion-based generation [11, 18]. Both approaches introduce representation spaces fundamentally disjoint from the continuous embeddings consumed by the understanding branch, severely limiting how much generation training can benefit the shared visual representations [19].

To bridge this gap, we propose Next Embedding Prediction (NEP) as a cross-modal generation paradigm. Recent literature [20] has demonstrated that predicting continuous embeddings autoregressively serves as an exceptionally strong visual learner. Building upon this insight, our core motivation is to formulate image generation as autoregressive embedding prediction within the exact same continuous representation space used by the LLM backbone.

This formulation anchors the generative target directly in the continuous input space of the language model, and preserves a unified representation manifold for both perception and generation. Unlike discrete tokenization which maps images into a separate categorical codebook, or diffusion objectives defined in a specialized synthesis-oriented latent space, NEP avoids a separate target-to-LLM translation interface. The design is therefore intended to make auxiliary visual supervision more directly compatible with the representations used for downstream understanding; Section 4 tests this claim against fixed-architecture and related-objective controls.

Contemporaneous UniHetero [21] likewise finds that autoregressing on LLM input embeddings is effective at large data scale. NEP is differentiated by using this representation as the target for instruction-conditioned cross-image prediction across diverse generation tasks, and by coupling it to an asymmetric MoT branch whose upper-layer gradients are isolated from the understanding path and whose parameters are removed after training.

Formulation and Supervision Target. Unlike single-image self-supervised pretraining [20], NEP operates in a conditional, multi-task setting. Given a multimodal context containing an input image I and diverse linguistic instructions (e.g., for segmentation or grounding), let $\mathbf { X } _ { \mathrm { c t x } }$ denote the concatenated sequence of text tokens and source image embeddings. For a generative task with a target image $I ^ { \mathrm { t g t } }$ the model autoregressively predicts the target embedding sequence $\hat { \mathbf { z } } ^ { \mathrm { t g t } }$

Crucially, rather than introducing an external reconstruction decoder to predict raw pixels, we extract ground-truth embeddings using the identical vision encoder followed by a visual-language projector. To ensure the target resides in the exact LLM input space, the target embeddings are defined as $\mathbf { z } ^ { \mathrm { t g t } } = \mathrm { P r o j e c t o r } ( \mathrm { V i T } ( I ^ { \mathrm { t g t } } ) )$ . The model predicts each embedding conditioned on the context and previously generated embeddings:

$$
\begin{array} { r } { \hat { z } _ { i } ^ { \mathrm { t g t } } = f _ { \mathrm { g e n } } \big ( \mathbf { x } _ { \mathrm { c t x } } , \hat { z } _ { < i } ^ { \mathrm { t g t } } ; \boldsymbol { \Theta } _ { \mathrm { g e n } } \big ) , } \end{array}\tag{1}
$$

where $f _ { \mathrm { g e n } }$ denotes the generative forward pass of the network parameterized by $\Theta _ { \mathrm { g e n } }$ . To optimize this objective, both the predicted and target embeddings are first $\ell _ { 2 } { \mathrm { - n o r m a l i z e d } }$ . The generation loss is then computed as the cosine distance in this continuous representation space:

$$
\mathcal { L } _ { \mathrm { g e n } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( 1 - \frac { \hat { z } _ { i } ^ { \mathrm { t g t } } \cdot z _ { i } ^ { \mathrm { t g t } } } { \Vert \hat { z } _ { i } ^ { \mathrm { t g t } } \Vert _ { 2 } \Vert z _ { i } ^ { \mathrm { t g t } } \Vert _ { 2 } } \right) .\tag{2}
$$

By optimizing directly within the understanding interface rather than pursuing photorealistic pixel synthesis, we explicitly prioritize the enrichment of deep structural representations over visual fidelity. Target Stabilization via EMA. A critical challenge in predicting representations projected into the LLM space is that the active visual projector parameters are continuously updated during joint training [21]. Utilizing a dynamic projector to extract targets can lead to supervision drift and representation collapse. To prevent this, we maintain a target projector whose weights are updated via an Exponential Moving Average (EMA) of the active projector’s weights. The stable target embeddings $\mathbf { z } ^ { \mathrm { t g t } }$ are thus computed by passing the target image through the frozen ViT followed by this EMA target projector. Correspondingly, a dedicated vision-head is applied to the output hidden states of the LLM’s generation branch to predict these stable targets. As detailed in Section 2.2, the decoupled architecture with this robust NEP formulation safely injects precise spatial information into the shared representations, and further mitigates direct interference with the upper understanding layers.

![](images/0ae5b8c4af9b4781a9addeb4c0cb7d9d399902736c8a15b675f04d606caaad46.jpg)  
Figure 2: Overview of generation training data. Five primary task categories spanning 15 subtasks, jointly covering the fine-grained visual capabilities targeted by our generation-guided training.

## 2.2. MoT Decoupled Architecture

A straightforward approach to introducing generation supervision is to add the NEP loss directly onto the shared LLM backbone. However, our experiments (Section 4) and prior works reveal that this strategy degrades understanding capability [13, 22, 23]: the generative and understanding objectives impose conflicting optimization pressures, ultimately diluting the abstract semantic representations necessary for complex reasoning. To resolve this tension, we adopt the Mixture-of-Transformers (MoT) architecture [24], which decouples the upper task-specific parameters while preserving a shared lower trunk for indirect knowledge transfer.

Decoupled Architecture. To preserve visual structures that are typically abstracted in deeper layers focused on semantics, the model topology bifurcates at an intermediate layer $l _ { \mathrm { s p l i t } }$ . The understanding branch continues to process the sequence through the remaining upper layers $( l > l _ { \mathrm { s p l i t } } )$ , optimized solely by the text cross-entropy loss $( { \mathcal { L } } _ { \mathrm { u n d } } )$ . Concurrently, we extend the backbone with a parallel generation branch comprising $N _ { \mathrm { g e n } }$ transformer layers. To guarantee a reasonable initialization and stabilize early training, the parameters of these generation layers $\left( \Theta _ { \mathrm { g e n } } \right)$ are directly copied from the corresponding pre-trained layers of the understanding backbone. This branch extracts hidden states from layer $l _ { \mathrm { s p l i t } }$ to autoregressively produce NEP predictions:

$$
\mathbf { H } _ { \mathrm { g e n } } = \mathrm { T r a n s f o r m e r } _ { \mathrm { g e n } } ( \mathbf { H } ^ { ( l _ { \mathrm { s p l i t } } ) } ; \Theta _ { \mathrm { g e n } } )\tag{3}
$$

$$
\hat { z } _ { i } ^ { \mathrm { t g t } } = \mathrm { V i s i o n H e a d } ( \mathbf { h } _ { \mathrm { g e n } , i } ; \Theta _ { \mathrm { h e a d } } ) ,\tag{4}
$$

where $\mathbf { H } ^ { ( { l _ { \mathrm { s p l i t } } } ) }$ denotes the sequence of hidden states at layer $l _ { \mathrm { s p l i t } }$ , and $\mathbf { h } _ { \mathrm { g e n } , i }$ is the i-th token representation in the generation output $\mathrm { H } _ { \mathrm { g e n } }$ . The generation branch is parameterized by $\Theta _ { \mathrm { g e n } }$ and $\Theta _ { \mathrm { h e a d } } .$ supervised entirely by the NEP loss $( \mathcal { L } _ { \mathrm { g e n } } )$ .

Mechanism of Synergy. A critical question arises: how can generative training enhance understanding if their parameters diverge? The key lies in the asymmetric gradient flow across the shared first $l _ { \mathrm { s p l i t } }$ layers. This partial decoupling induces two powerful, complementary effects:

(1) Direct Refinement of Shared Representations. During joint training, gradients from $\mathcal { L } _ { \mathrm { g e n } }$ backpropagate through the generation branch into the visual projection and the shared trunk $( l \le l _ { \mathrm { s p l i t } } )$ This forces these lower layers to produce intermediate states $\mathbf { H } ^ { ( l _ { \mathrm { s p l i t } } ) }$ that are enriched with precise, fine-grained visual details. Because $\mathbf { H } ^ { ( l _ { \mathrm { s p l i t } } ) }$ acts as the direct input to the upper understanding branch, this visual enrichment is seamlessly inherited.

(2) Implicit Adaptation of Upper Layers. While the upper understanding layers $( l > l _ { \mathrm { s p l i t } } )$ remain shielded from generative gradients, they are continuously fed the visually enriched representations from the shared trunk. Consequently, they are compelled to adapt under ${ \mathcal { L } } _ { \mathrm { u n d } }$ to fully exploit these high-quality features. Empirically (Section 4), this adaptation manifests as sharpened attention heads that reliably anchor onto question-relevant visual regions, thereby enhancing evidence retrieval during multi-step reasoning.

Training and Inference. The overall objective minimizes the weighted sum $\mathcal { L } = \mathcal { L } _ { \mathrm { u n d } } + \lambda \cdot \mathcal { L } _ { \mathrm { g e n } }$ Crucially, at inference time, the entire generation branch $\left( \Theta _ { \mathrm { g e n } } \right.$ and $\Theta _ { \mathrm { h e a d } } )$ is discarded. Only the understanding branch performs the forward pass, rendering the inference cost, latency, and memory footprint strictly identical to the standard baseline model.

## 2.3. Generation Task Construction

Prior unified multimodal works typically introduce generation data in the form of generic text-toimage synthesis or simple editing, treating generation merely as a self-contained skill. Under such configurations, understanding performance gains are often negligible or even negative. We argue that generic synthesis tasks, such as “generate an image of a cat”, provide little training signal relevant to complex visual reasoning, as they merely demand a superficial alignment between basic nouns and visual concepts.

The core insight of our framework is that generation can better enhance understanding when the generative act inherently depends on deep multimodal comprehension. Rather than treating generation as a straightforward rendering process, we purposefully formulate it as the terminal output of a complex cognitive task. If a model must execute precise spatial localization, compositional scene analysis, or multi-step logical deduction just to figure out what or where to generate, the generation objective acts as a powerful forcing function. It compulsorily drives the network to develop the exact fine-grained perceptual and reasoning capabilities that pure text supervision fails to provide.

Guided by this principle, we select candidate generation tasks based on their latent cognitive dependency with the target understanding capabilities. We construct five generation tasks with 15 subtasks in total (illustrated in Figure 2):

• Grounding: Region localization tasks that compel the model to perfectly resolve complex linguistic referring expressions before predicting absolute spatial coordinates, directly cultivating fine-grained spatial reasoning.

![](images/c6b5b2f99d6a0b5459b3966189f245675843a37aac392d41fdf81b58b8558147.jpg)  
Figure 3: Generation task construction pipeline. Illustration of our automated, annotation-free data construction pipeline, taking grounding as a representative example.

• Segmentation: Pixel-level mask prediction tasks that force the model to comprehend semantic boundaries and physical extents based on intricate textual instructions, reinforcing instance-level discrimination.

• Image Editing: Conditional image transformation tasks (e.g., style transfer, object removal) that require a deep, compositional understanding of scene layouts and object interdependencies before executing the edit.

• Visual Chain-of-Thought: Multi-step reasoning tasks that demand the model to actively generate intermediate visual evidence (e.g., highlighting key regions) to anchor multi-step logical deduction, simulating a “thinking with images” process.

• Text-to-Image (T2I): Carefully curated image synthesis tasks grounded in rich world knowledge and dense structural prompts, establishing foundational cross-modal alignment beyond naive concept mapping.

Crucially, our ablation studies (Section 4) confirm that these task categories provide complementary supervision signals. Empirically, on Counting & Spatial benchmarks the gain from combining all categories tends to exceed the best single-task gain, suggesting that diverse, comprehension-correlated generation tasks are not redundant; rather, they jointly cover different facets of fine-grained visual structure that no single task fully captures.

Building on this finding, we develop a scalable data synthesis framework to produce high-quality generation data across diverse domains without manual annotation. Taking grounding as a representative example, our automated pipeline operates in four stages: (1) open-vocabulary tagging via RAM++ to discover all salient objects; (2) open-vocabulary detection via Grounding DINO to localize each tag; (3) difficulty-controlled sampling (area-proportional for easy, uniform for hard) to select target boxes; and (4) LLM-based instruction synthesis using the annotated image and a zoomed crop to generate unambiguous referring expressions. This pipeline generalizes to arbitrary image domains and enables scaling of generation data to new scenarios (e.g., advertising) with minimal engineering effort.

Combining the above task selection principle with this automated construction and filtering pipeline, we ultimately scaled the total generation training corpus to approximately 10M samples spanning all five task categories, and mixed them with understanding data for training.

## 3. Experiments

## 3.1. Experiments Setup

Model Configuration. To ensure a controlled environment devoid of data contamination from existing VLM weights, we build our main models from scratch. Adopting the Qwen3-VL architecture [2] for the understanding branch, we utilize Qwen3-VL-ViT as a visual encoder initialized with pretrained weights, which remains frozen throughout all training stages. For the LLM backbone, we employ the 2B and 4B Qwen3 [25] models. Note that we use the raw LLM weights without any prior multimodal alignment. Additionally, the ViT-LLM projector is randomly initialized, ensuring all visual-language capabilities are acquired solely through our training pipeline. The MoT generation branch consists of $L _ { \mathrm { m o t } }$ transformer layers initialized from the corresponding pre-trained layers of the understanding backbone, with hidden states extracted from the intermediate layer $l _ { s } \approx L / 2 = 1 4$ of the understanding backbone (where L is the total layer count).

Implementation Details. Training proceeds in two stages. In Stage 1 (MoT Align), we freeze all understanding-side parameters and train only the MoT generation branch using T2I generation data so as to activate the generation pathway and align it with the existing visual representation space without perturbing the pretrained understanding capability. In Stage 2 (Joint Training), both understanding and generation data are used simultaneously. The understanding branch is optimized with text crossentropy loss, while the generation branch is optimized with NEP loss under a progressive weighting schedule that linearly ramps λ from an initial value of 0.015 to the target weight 1.0 over 4k steps.

This setting allows generation gradients to refine the shared trunk while leaving the ViT frozen, thereby enabling the indirect transfer mechanism described in Section 2.2. On our HPC Cluster, the final baseline uses 2,208 GPU-hours, whereas GAS uses 2,464 GPU-hours (+11.6%); the corresponding ablation runs use 544 and 608 GPU-hours (+11.8%). These totals include the generation branch, vision head, and EMA target projector. All three auxiliary components are removed at inference.

As for hyper-parameters, we use the AdamW [26] optimizer with a cosine learning rate schedule, a peak learning rate of $5 \times 1 0 ^ { - 5 }$ for both understanding branch and the MoT generation branch. We maintain a target projector EMA with a decay rate of 0.999 for training stability, following the update rule $\theta _ { \mathtt { E M A } } ^ { ( t ) } = 0 . \overset {  } { 9 } 9 \overset {  } { 9 } \cdot \theta _ { \mathtt { E M A } } ^ { ( t - 1 ) } + 0 . 0 0 1 \cdot \theta ^ { ( t ) }$ . Training is conducted on 32 GPUs with DeepSpeed ZeRO-2 [27], using a global batch size of 128. The maximum sequence length is 8192 tokens.

## 3.2. Evaluation Setup

To comprehensively assess the impact of generation-guided training on complex visual understanding, we organize our evaluation into four capability dimensions, each targeting aspects where fine-grained visual perception and deep reasoning are essential:

(1) General Perception (Multi-Image). We evaluate holistic multimodal understanding and multiimage reasoning on MME [28], MMMU [29], BLINK [30] and RealWorldQA. These benchmarks assess broad visual perception, world knowledge integration, and multi-image comprehension, capabilities where richer visual representations should yield direct benefits.

(2) Visual Reasoning. We evaluate mathematical and logical visual reasoning on CharXiv [31], DynaMath [32], MathVision [33], MathVista [34], LogicVista [35] and VisuLogic [36]. These benchmarks require precise visual parsing of charts, diagrams, and geometric figures combined with multi-step logical inference, precisely the type of complex reasoning that should benefit from generation-enhanced visual representations.

(3) Counting & Spatial Understanding. We evaluate fine-grained spatial perception on Count-Bench [37, 38] and CV-Bench [39], which specifically probe object counting accuracy and spatial relationship understanding, capabilities directly trained by our grounding and segmentation generation tasks.

(4) Video Understanding. We evaluate temporal reasoning on Video-MME [40] and MVBench [41]. Video understanding requires sustained visual attention across frames and temporal reasoning, testing whether the improved visual information retention induced by generation training generalizes to the temporal domain.

## 3.3. Main Results

<table><tr><td></td><td></td><td colspan="10">ReaadA test] t28] test] 30o0] DS] 31[ MMMU [2] [9] Chaaiv Chaaiv BLINK</td><td>Viulsge [est] t42]</td><td>ConA</td><td>[est ε]  8]</td><td>C-eh</td><td></td><td></td><td>MVncch [est] s1][</td><td>[est] te0o]</td><td>[3] 9</td><td>[6ε] 2D</td><td>C-nch</td><td></td><td></td><td>Vide-ME</td><td></td></tr><tr><td>Model</td><td>#Params</td><td>MME</td><td></td><td></td><td></td><td>test</td><td></td><td></td><td>Dnnath RSS] 31</td><td>est] t32]</td><td>Masion [] </td><td>Masta</td><td>[] 4</td><td>Losta</td><td>[] s</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Understanding-only MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Thyme-VL* [43]</td><td>7B</td><td></td><td></td><td>35.7</td><td>56.1 70.2</td><td>54.8</td><td></td><td></td><td></td><td>16.6 8.52</td><td>27.6 70.0</td><td></td><td></td><td></td><td>23.4</td><td>24.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLaVA-v1.5* [44] Qwen2.5-VL [45]</td><td>7B 3B</td><td>2204</td><td></td><td>49.2</td><td>48.6</td><td>64.1</td><td></td><td></td><td></td><td></td><td>14.8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>54.1 61.3</td><td></td></tr><tr><td>Qwen3-VL [2]</td><td>2B</td><td>2026</td><td></td><td>48.1</td><td>54.9</td><td>63.7</td><td></td><td>65.0 38.8 30.2</td><td></td><td></td><td>20.7</td><td></td><td>51.4</td><td>36.5</td><td>26.4</td><td></td><td>86.9</td><td></td><td>70.2</td><td></td><td>86.8</td><td></td><td>48.1 50.4</td><td></td></tr><tr><td>Unified Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BAGEL*[13]</td><td>7B+7B</td><td>2381</td><td></td><td>55.3</td><td></td><td>72.8</td><td></td><td></td><td></td><td></td><td></td><td></td><td>73.2</td><td>44.3</td><td>41.7</td><td></td><td>82.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Emu3*[16]</td><td>8B</td><td></td><td></td><td>31.6</td><td></td><td>57.4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>24.7</td><td></td><td>65.2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MetaMorph* [46]</td><td>8B</td><td></td><td></td><td>41.8</td><td></td><td>58.3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>48.8</td><td></td></tr><tr><td>Janus-Pro [47]</td><td>7B</td><td>1978*</td><td></td><td>41.0*</td><td>39.7 58.0*</td><td></td><td></td><td>49.9 24.4 24.7 15.8 45.3*28.6 23.8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>82.8 68.7 66.5</td><td></td><td></td><td></td></tr><tr><td>Show-0*[11]</td><td>1.3B</td><td></td><td>26.7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Show-02* [48]</td><td>1.5B</td><td></td><td></td><td>37.1</td><td></td><td>56.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>50.6</td><td></td></tr><tr><td>Cheers* [49]</td><td>1.5B</td><td></td><td></td><td>36.0</td><td></td><td>60.9</td><td></td><td></td><td></td><td></td><td></td><td></td><td>50.5</td><td>一</td><td></td><td></td><td></td><td></td><td>1</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GAS</td><td>2B</td><td>2003</td><td></td><td>46.6</td><td>48.0</td><td>64.8</td><td></td><td>64.4 34.3</td><td></td><td>47.9</td><td>20.4</td><td></td><td>56.4</td><td>32.7 28.6</td><td></td><td></td><td></td><td>90.1</td><td></td><td></td><td>73.2 73.8</td><td></td><td>46.2 47.8</td><td></td></tr><tr><td>-baseline</td><td></td><td>1986</td><td>46.9</td><td></td><td>46.7</td><td>64.1</td><td></td><td>64.0 34.6 46.2</td><td></td><td></td><td>19.1</td><td></td><td>54.4</td><td>36.0 26.8</td><td></td><td></td><td>87.7</td><td></td><td>69.9</td><td></td><td>73.2</td><td></td><td>45.2 49.0</td><td></td></tr><tr><td>GAS</td><td>4B</td><td>2146</td><td></td><td>57.0</td><td>56.9</td><td>70.3</td><td>78.4</td><td>40.5</td><td></td><td>55.2</td><td>27.0</td><td></td><td>68.1</td><td>37.1</td><td></td><td>23.6</td><td>90.8 90.1 75.1 82.1</td><td></td><td>75.4</td><td></td><td>83.6</td><td></td><td>54.4 52.4</td><td>55.0 54.2</td></tr><tr><td>-baseline</td><td></td><td>2092</td><td></td><td>54.9</td><td>57.8</td><td>70.2</td><td></td><td>77.9 40.3</td><td></td><td>57.6</td><td>25.3</td><td></td><td>65.6</td><td>37.1 26.1</td></table>

Table 1: Overall resultsfor a range ofunderstanding-only MLLMs, Unified Models, baselines and our GAS family. <sup>∗</sup> denotes metrics taken directly from public reports rather than reproduced by us.

Table 1 reports the main results of GAS, its matched understanding-only baselines, public MLLM and unified-model references across all benchmarks. The paired GAS/baseline rows are the primary evidence for attribution. All improvements of GAS are achieved with zero additional inference cost since the generation branch is discarded after training.

At the same 2B scale, GAS is highly competitive with baseline and other open-source models while showing markedly stronger reasoning and counting/spatial behavior: DynaMath increases from 46.2 to 47.9 (+1.7pp), MathVista from 54.4 to 56.4 (+2.0pp), CountBenchQA from 87.7 to 90.1, CV-Bench-2D from 69.9 to 73.2, and VisuLogic from 26.8 to 28.6, all directly aligned with the spatial perception and visual reasoning capabilities targeted by our grounding, segmentation, and Visual-CoT generation tasks. Scaling to 4B, GAS establishes the strongest overall results in the unified family, leading on most of the 16 benchmarks we report (e.g., MMMU 57.0, CharXiv-DS 78.4 / RS 40.5, CountBenchQA 90.8) and surpassing significantly larger unified models such as BAGEL (7B+7B) and Emu3 (8B). The lone exception is VisuLogic at 4B (23.6), where the larger backbone shows a benchmark-specific regression that does not transfer to other reasoning suites, suggesting an idiosyncratic sensitivity of this puzzle-style benchmark to backbone scale. Together, these results show that our generation-guided framework consistently transfers fine-grained perception and reasoning gains to the understanding branch across both scales.

- The \*\*lamp\*\* is on the floor near the sofa, which is also farther away from the chair compared to the chair’s position. Conclusion: - The \*\*chair\*\* is the closest object to itself, and the \*\*lamp\*\* is closer to the chair than the \*\*television\*\*.

- The \*\*television\*\* is on a stand against the wall, which is significantly farther away from the chair.

Analysis: - The \*\*chair\*\* is in the foreground, very close to the viewer.

Qualitative Case 1 and additional cases in Appendix B illustrate model behavior on CV-Bench, Dyna-Math, VisuLogic, and RealWorldQA. These selected examples show improved spatial localization, recognition of fine-grained visual details (particularly for small objects and subtle differences), and more reliable visual reasoning chains that reference image evidence. These qualitative patterns align well with the quantitative gains observed across our four evaluation dimensions, and suggest that the generation-derived supervision successfully transfers pixel-level and region-level perception capabilities to the understanding branch. In the following section, we use controlled ablations and representation-level analyses to further examine how and when generation training produces these improvements.

Qualitative Case 1: CV-Bench [39]  
![](images/eb2cce63ee546981ae3f52bc61c89e6d960c91436fcc004fa216da58790c97d4.jpg)

Question: Estimate the real-world distances between objects in this image. Which object is closer to the chair (highlighted by a red box), the television (highlighted by a blue box) or the lamp (highlighted by a green box)? A. television, B. lamp

## GAS-4B:

## <think>

To determine which object is closer to the chair (highlighted by the red box), we need to analyze the spatial relationships in the image:

1. \*\*Chair (red box)\*\*: This is located in the foreground, near the dining table.

2. \*\*Television (blue box)\*\*: This is positioned on a stand against the wall, further back in the room.

3. \*\*Lamp (green box)\*\*: This is placed on the floor near the sofa, closer to the center of the room but still farther back than the chair.

Thus, the correct answer is:

The final answer is B .

## 4. Analysis

We conduct a series of controlled experiments to dissect the mechanisms underlying generation-guided training. Unless otherwise noted, analyses use the 2B-parameter model with 2.5B understanding and 2.5B generation tokens and the same evaluation suite from Section 3. We organize the investigation around five core questions: (1) Does GAS provide consistent benefits, transfer across training stages and backbone capability levels? (2) Which architecture turns generation data into transferable supervision beyond additional data, capacity, or training cost? (3) Which generation tasks contribute most to understanding gains? (4) At which layer should generation supervision be injected? (5) Which representation-level changes accompany the downstream gains?

## 4.1. Generalization Across Training Stages

Firstly, we verify that our generation-guided framework is not limited to a specific training stage or backbone quality by applying GAS training atop understanding backbones of varying capability levels. We evaluate three settings: (i) training from scratch (the raw LLM backbone), (ii) after pretraining with large-scale multimodal data, and (iii) after directly supervised fine-tuning (SFT) based on Qwen3-VL-2B [2]. Each setting compares GAS with its corresponding standard VLM baseline; matched controls for generation-data exposure and additional training budget are reported separately in Section 4.2. Results are presented in Table 2.

<table><tr><td>Stage</td><td>Configuration Overall Perception Reasoning Count&amp;Spatial Video</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>|MME MMMU MathVista MVBench</td></tr><tr><td rowspan="2">From Scratch</td><td>Baseline</td><td>47.25</td><td>56.55</td><td>31.38</td><td>76.92</td><td>46.60</td><td>1848</td><td>43.5</td><td>42.7</td><td>46.9</td></tr><tr><td>+ GAS</td><td>48.25</td><td>58.52</td><td>32.63</td><td>75.15</td><td>47.65</td><td>1895</td><td>45.8</td><td>47.1</td><td>49.5</td></tr><tr><td rowspan="2">After Pretrain</td><td>Baseline</td><td>56.23</td><td>63.80</td><td>43.22</td><td>82.40</td><td>53.95</td><td>2169</td><td>53.2</td><td>72.3</td><td>55.1</td></tr><tr><td>+ GAS</td><td>56.71</td><td>63.88</td><td>44.12</td><td>82.95</td><td>53.95</td><td>2170</td><td>51.1</td><td>71.2</td><td>54.6</td></tr><tr><td rowspan="2">After SFT</td><td>Baseline</td><td>52.65</td><td>61.32</td><td>38.23</td><td>79.82</td><td>51.40</td><td>2060</td><td>46.3</td><td>52.9</td><td>53.0</td></tr><tr><td>+ GAS</td><td>53.49</td><td>61.62</td><td>39.57</td><td>80.80</td><td>51.65</td><td>2096</td><td>45.8</td><td>53.6</td><td>52.9</td></tr></table>

Table 2: Generalization across training stages. GAS generation-guided training yields improvements when applied to understanding backbones at different capability levels.

The from-scratch setting yields the largest Overall gains (+1.00pp overall, with +1.97pp on Perception and +1.25pp on Reasoning), but a decrease on Count&Spatial (−1.77pp), as the model benefits most from generation supervision when its visual representations are not yet well-developed. After large-scale pretraining, GAS still provides consistent improvements (+0.48pp overall, +0.90pp on Reasoning), showing that generation-guided supervision is complementary even to a strong, welltrained backbone. Most notably, applying GAS during SFT also yields a robust +0.84pp overall gain with simultaneous improvements on Perception (+0.30pp), Reasoning (+1.34pp), Count&Spatial (+0.98pp), and Video (+0.25pp). Across three stages spanning the full spectrum of backbone capability, GAS delivers positive transfer on virtually every dimension, confirming that the proposed generation-guided training is a stage-agnostic augmentation that benefits both weak and already-strong understanding models.

Insight 1: Generation-guided training is stage-agnostic: it provides the largest gains from scratch, complementary improvements after pretraining, and still consistent benefits after SFT, making it a plug-in augmentation across the full training pipeline rather than a stage-specific technique.

## 4.2. What Makes Generation Supervision Transfer?

A generation-augmented model changes several factors at once: it sees a new data stream, spends additional training compute, introduces a generation branch, and learns from a new visual objective. We disentangle these factors with the controls in Table 3. Beyond the standard 2.5B-token understanding baseline, we train a compute-matched baseline with 11% more understanding data and expose a standard VLM baseline to the same generation samples, whose image outputs contribute no loss. We then isolate the MoT architecture without generation supervision, apply NEP directly to a shared backbone, and replace NEP with position-aligned visual prediction (no shift) while keeping MoT and the training mixture fixed.

<table><tr><td>Configuration</td><td>Overall Perception Reasoning</td><td></td><td></td><td>Count&amp;Spatial Video</td><td></td></tr><tr><td>Understanding-only baseline</td><td>47.25</td><td>56.55</td><td>31.38</td><td>76.92</td><td>46.60</td></tr><tr><td>Baseline, +11% understanding data</td><td>47.73</td><td>57.63</td><td>34.01</td><td>71.28</td><td>45.55</td></tr><tr><td>Baseline, same generation data (text loss only)</td><td>47.14</td><td>56.08</td><td>33.52</td><td>71.00</td><td>46.25</td></tr><tr><td>MoT, no generation supervision</td><td>47.63</td><td>57.22</td><td>34.20</td><td>70.78</td><td>45.60</td></tr><tr><td>Shared backbone + NEP</td><td>46.00</td><td>54.95</td><td>32.25</td><td>70.20</td><td>45.15</td></tr><tr><td>MoT + no-shift visual loss</td><td>47.84</td><td>57.70</td><td>32.16</td><td>76.38</td><td>46.60</td></tr><tr><td>MoT + NEP (GAS)</td><td>48.25</td><td>58.52</td><td>32.63</td><td>75.15</td><td>47.65</td></tr></table>

Table 3: Disentangling data, architecture, and objective. Matched controls separate additional training, passive exposure to generation data, MoT decoupling, and the NEP objective.

Additional training alone does not explain the gain. The longer understanding-only run improves Overall only modestly, while merely mixing in the same generation samples provides no benefit because their image targets are not supervised by standard VLM training. MoT without a generation loss also improves some dimensions but does not match GAS, showing that neither data exposure nor the extra branch is sufficient by itself.

The objective and its gradient route are complementary. Applying NEP directly to the shared upper layers causes broad degradation, revealing interference between generation-specific optimization and the understanding pathway. With MoT fixed, position-aligned no-shift prediction already provides useful visual supervision, whereas autoregressive NEP yields the strongest aggregate transfer and the best Perception and Video scores. The no-shift control remains stronger on Count&Spatial, so the effect is not a uniform per-capability improvement; rather, NEP changes the transfer profile while improving the aggregate outcome. Together, these controls show that MoT protects the deployed understanding pathway and NEP supplies a task-conditioned learning signal that can enrich their shared lower representation.

Insight 2: Generation data is not supervision by itself. GAS turns it into transferable visual learning by pairing task-conditioned next-embedding prediction with an isolated upper generation pathway: NEP enriches the shared representation, while MoT prevents generation-specific optimization from rewriting the understanding model.

## 4.3. Per-Task Generation Contribution

We next investigate which generation tasks contribute most effectively to understanding enhancement by individually adding 200k samples from each of the five task categories to the understanding-only baseline. Table 4 presents the results, and we additionally report the combined “All” task samples

## configuration.

Several patterns emerge from this ablation. First, all five generation tasks improve at least one evaluation dimension, but their contribution profiles differ significantly, validating our latent task correlation hypothesis (Section 2.3). T2I provides the highest single-task overall gain (+0.49pp) with broad improvements across perception and spatial understanding; Segmentation and Grounding strongly boost Count&Spatial (+2.00 and +2.37pp respectively, with CV-Bench 3D improving by up to 9.8pp from segmentation), consistent with the expectation that pixel-level and region-level generation tasks directly cultivate spatial perception capabilities. Visual-CoT uniquely enhances complex reasoning benchmarks: BLINK (+4.2pp) and MathVision (+1.0pp), reflecting its role in forcing the model to produce intermediate visual reasoning traces that reinforce multi-step inference. Editing yields a small Overall gain and a larger Count&Spatial improvement, but does not improve every dimension.

<table><tr><td>Configuration</td><td>Overall Perception Reasoning</td><td></td><td></td><td>Count&amp;Spatial Video</td><td></td><td>|MME</td><td>BLINK</td><td>CV-2D/3D</td><td>MathVision VisuLogic</td><td></td></tr><tr><td>Understanding only</td><td>47.63</td><td>57.22</td><td>34.20</td><td>70.78</td><td>45.60</td><td>1868</td><td>41.9</td><td>59.8/52.1</td><td>17.1</td><td>25.2</td></tr><tr><td>+ T2I</td><td>48.12</td><td>57.70</td><td>34.66</td><td>71.65</td><td>45.85</td><td>1820</td><td>44.5</td><td>59.3/55.3</td><td>15.8</td><td>25.1</td></tr><tr><td>+ Editing</td><td>47.77</td><td>56.70</td><td>34.12</td><td>72.95</td><td>45.70</td><td>1830</td><td>42.5</td><td>61.1/59.5</td><td>14.8</td><td>26.4</td></tr><tr><td>+ Segmentation</td><td>47.43</td><td>55.32</td><td>34.35</td><td>72.78</td><td>45.55</td><td>1750</td><td>44.1</td><td>63.6/55.9</td><td>16.4</td><td>24.2</td></tr><tr><td>+ Grounding</td><td>47.92</td><td>57.28</td><td>33.92</td><td>73.15</td><td>45.95</td><td>1840</td><td>44.6</td><td>61.9/58.3</td><td>13.5</td><td>23.6</td></tr><tr><td>+ Visual-CoT</td><td>47.86</td><td>57.80</td><td>34.26</td><td>71.45</td><td>45.20</td><td>1877</td><td>46.1</td><td>56.9/57.3</td><td>18.1</td><td>26.6</td></tr><tr><td>+ All (combined)</td><td>48.30</td><td>57.08</td><td>34.28</td><td>75.72</td><td>45.40</td><td>1851</td><td>44.9</td><td>62.5/67.0</td><td>20.1</td><td>23.6</td></tr></table>

Table 4: Per-task ablation. Each row adds a single generation task category to the understanding-only baseline. The “All” row combines allfive categories.

Most notably, the $\mathbf { \ddot { \theta } } \mathbf { A } \mathbf { l } \mathbf { l } ^ { \mathbf { \lessgtr } }$ configuration achieves a Count&Spatial score of 75.72, higher than any singletask setting and exceeding the strongest individual contributors (Segmentation +2.00pp, Grounding +2.37pp), with a similar pattern observed on MathVision (20.1 in the combined setting versus 18.1 from the best single task). This pattern is suggestive of a complementary effect: diverse generation tasks appear to address different facets of fine-grained visual representation that no single category fully covers. The practical implication is the one we use throughout the paper: composing tasks with distinct latent correlations to understanding is a productive design lever, especially for perception- and counting-oriented capabilities.

Effect of Task Correlation on Understanding Gains. The heterogeneous gains across task categories suggest that transfer depends not only on task modality, but also on the latent correlation between generation signal and the target understanding capability, as well as on the internal characteristics of the generation task itself. To probe this factor more directly, we conduct a controlled study by fixing the T2I task and data volume and varying only prompt characteristics. Specifically, we collect 200k additional T2I samples from COYO-700M [50] dataset and compare two variants: (i) Easy, generic short-

<table><tr><td>Metric</td><td>Baseline</td><td>Easy</td><td>Rewrite</td></tr><tr><td>Overall</td><td>47.63</td><td>47.87</td><td>48.34</td></tr><tr><td>Perception</td><td>57.22</td><td>56.47</td><td>57.53</td></tr><tr><td>Reasoning</td><td>34.20</td><td>34.37</td><td>34.58</td></tr><tr><td>Count&amp;Spatial</td><td>70.78</td><td>72.47</td><td>73.62</td></tr><tr><td>Video</td><td>45.60</td><td>46.55</td><td>45.95</td></tr></table>

Table 5: Effect of T2I prompt correlation with understanding tasks. Aligning T2I prompts with the compositional, knowledge-grounded scene structures required by understanding benchmarks improves transfer at fixed task and data volume.

caption synthesis as commonly used in prior unified models, and (ii) Rewrite, where prompts are enriched with world knowledge, multi-object compositions, and explicit attribute/relation cues to better mirror the scene structures probed by our reasoning and perception benchmarks.

As Table 5 shows, generic T2I yields a small Overall gain (+0.24pp) but decreases Perception by 0.75pp. Rewritten prompts improve Overall by +0.71pp and all four capability aggregates relative to the same baseline, although the easy prompts retain the best Video score. This contrast shows that volume of generation supervision is not what matters, a poorly correlated generation task can act as a distractor rather than a regularizer. Once prompts are rewritten to demand the same compositional, knowledge-grounded scene structures probed by understanding benchmarks, the very same T2I category turns into a broadly beneficial signal, on par with our targeted tasks. This directly closes the loop with our latent task correlation principle (Section 2.3): even within one nominal task, transfer scales with task-level correlation, not with sample count.

Insight 3: Understanding gains scale with task correlation. Each task targets a distinct facet, like spatial tasks for counting/localization, Visual-CoT for reasoning, editing for holistic perception, and combining them tends to yield further, complementary improvements on perception- and counting-oriented benchmarks beyond any single category. Even within T2I, correlation-aware prompts unlock transfer that generic captions cannot.

## 4.4. Layer-wise Supervision Injection

Our MoT architecture extracts features from an intermediate layer $l _ { s }$ of the understanding backbone to feed into the generation branch. This design choice raises the question: at which depth should generation supervision be injected? We compare extracting hidden states from layers 8, 14 (approximately $L / 2$ for our 2B, 28-layers model), and 20, as well as the effect of progressive loss warmup and unfreezing the ViT encoder. Results are shown in Table 6.

<table><tr><td>Configuration</td><td>Overall Perception Reasoning</td><td></td><td></td><td>Count&amp;Spatial Video</td><td></td><td></td><td></td><td>|MME BLINK CountBenchQA CV-3D</td><td></td></tr><tr><td>Layer 8</td><td>47.06</td><td>56.80</td><td>31.52</td><td>75.88</td><td>45.40</td><td>1830</td><td>44.3</td><td>86.4</td><td>66.1</td></tr><tr><td>Layer 14</td><td>47.85</td><td>58.58</td><td>31.64</td><td>75.28</td><td>47.60</td><td>1849</td><td>47.7</td><td>86.4</td><td>64.6</td></tr><tr><td>Layer 14 + warmup</td><td>48.25</td><td>58.52</td><td>32.63</td><td>75.15</td><td>47.65</td><td>1895</td><td>45.8</td><td>87.7</td><td>63.6</td></tr><tr><td>Layer 14 + unfreeze ViT</td><td>46.20</td><td>55.80</td><td>31.31</td><td>72.60</td><td>45.25</td><td>1808</td><td>44.7</td><td>79.7</td><td>69.9</td></tr><tr><td>Layer 20</td><td>47.25</td><td>58.20</td><td>31.43</td><td>74.60</td><td>45.45</td><td>1855</td><td>44.0</td><td>86.0</td><td>64.2</td></tr></table>

Table 6: Layer-wise injection and training strategy ablation. We vary the layer from which hidden states are extractedfor the generation branch, and compare with progressive warmup and ViT unfreezing. Layer 14 (≈ $L / 2 )$ with progressive warmup achieves the best overall performance.

The intermediate layer 14 (approximately $L / 2 )$ consistently outperforms both the shallower layer 8 and the deeper layer 20 on overall metrics and perception, while layer 8 achieves the strongest Count&Spatial score (75.88), suggesting that early-layer features already encode rich spatial information that benefits from generation supervision. Layer 14 achieves the highest BLINK (47.7/45.8), indicating that deeper features carry more abstract visual-semantic information beneficial for multiimage reasoning, represents a balance point where features retain sufficient spatial detail while having undergone enough semantic abstraction to benefit broad understanding tasks.

Progressive loss warmup improves the raw layer-14 configuration by +0.40pp Overall and +0.99pp Reasoning, consistent with our motivation that the faster-converging generation loss should be introduced gradually. Conversely, unfreezing the ViT encoder reduces Overall by 1.65pp and Count-BenchQA from 86.4 to 79.7 relative to the frozen layer-14 setting, indicating that the generation-side optimization pressure corrupts the pretrained visual features when allowed to propagate into the encoder, which further validating our design to keep the ViT frozen and channel generation benefits exclusively through the shared projection layer.

![](images/41ed63e3fb30a09e0ce08d88420391eb6dfebea5e8b6f0f7bd0e7ec40ba83ace.jpg)  
Figure 4: Visual information retention and layer-wise linearprobing. Left: cosine similarity between per-layer visual tokens and input visual features. Right top / bottom: linear probing accuracy on RefCOCO (region level) and ImageNet (global). The models are comparable on ImageNet, while GAS’s representations become increasingly more linearly separable on RefCOCO in deeper layers.

Insight 4: The best Overall score occurs when supervision branches near $L / 2$ with progressive warmup, balancing spatial detail and semantic abstraction. Unfreezing the ViT degrades performance in this setup, suggesting that generation benefits should flow through the projection layer, not corrupt the frozen encoder.

## 4.5. Representation-Level Diagnostics

To understand how generation training improves understanding at the representation level, we conduct three complementary analyses: (i) visual information retention across layers, (ii) attention pattern visualization, and (iii) linear probing diagnostics.

Visual Information Retention. We measure the cosine similarity between the visual token representations at each LLM layer and the original visual features at the LLM input, tracking how much visual information is preserved as processing deepens. Figure 4 (left) shows that the baseline similarity decreases beyond layer $L / 3$ whereas the GAS model retains higher similarity through the middle and deep layers, demonstrating that generation supervision encourages the network to retain visual features rather than allowing them to be “squeezed out” by language-dominant processing. The generation branch, by requiring accurate embedding prediction from intermediate representations, creates an implicit incentive for the backbone to preserve visual information at those layers, which then propagates benefit to the understanding branch operating on the same representations.

<table><tr><td>Qualitative Case 2: CV-Bench-2D (Attention Analysis)</td></tr><tr><td>Question: How many pictures are in the image? A. 1 B.3 C.2 D.0 Ground Truth: A Baseline: &lt;think&gt; The image shows a room with a wooden table and chairs; no separate pictures appear, so the answer is D (0). &lt;/think&gt; Answer: D X GAS: &lt;think&gt; The image is itself a single coherent scene (a dining setup); among the choices, A (1)</td></tr><tr><td>corresponds to the single picture present. &lt;/think&gt; Answer: A √</td></tr><tr><td>Qualitative Case 3: MathVista-MINI (Attention Analysis) Question: Subtract all red things. Subtract all tiny matte balls. How many objects are left? Ground Truth: 5</td></tr></table>

Layer-wise Linear Probing. To quantify the discriminative quality of visual representations at each layer, we extract hidden states corresponding to visual tokens and train linear classifiers on ImageNet [51] (global classification) and RefCOCO [52, 53] (region-level referring expression comprehension). As shown in Figure 4 (right), the two models perform comparably on ImageNet across all layers. On RefCOCO, GAS exhibits a growing advantage that widens consistently from layer 8 onward, ultimately achieving substantially higher probing accuracy at the final layer. This pattern reveals that generation training specifically enriches fine-grained, region-level visual representations rather than generic semantic features, aligning with GAS’s strongest downstream gains on spatial and grounding evaluations. The widening gap at deeper layers further corroborates the retention analysis: because GAS preserves more spatial visual information in deep layers, its representations remain useful for fine-grained tasks even after extensive language-driven processing.

Attention Visualization. We visualize attention maps from representative understanding CV-Bench and MathVista examples at different LLM layers to examine where the model “looks” during reasoning. As illustrated in Figure 5, the baseline model’s attention becomes more diffuse in deeper layers—by layer 20, attention is scattered broadly across the image with no clear focus on task-relevant regions. In contrast, GAS maintains sharp, focused attention on question-relevant regions throughout the network depth. For instance, on a CV-Bench counting question (“How many pictures are in the image?”), the baseline incorrectly answers 0 with diffuse attention, while our model correctly identifies the single picture with attention concentrated on the relevant region. On MathVista object-counting tasks, GAS correctly attends to all relevant objects for subtraction reasoning, whereas the baseline’s scattered attention leads to miscounting. This attention sharpening directly explains the mechanism by which generation training improves understanding: by requiring the generation branch to predict precise visual embeddings, the shared representations develop stronger spatial selectivity that carries over to understanding-time inference.

![](images/6a5653c61e0ab4f717484c0c93f1472d3e426621fcd05479d3f039fd21859aff.jpg)  
Figure 5: Layer-wise attention visualization on representative understanding samples. The first two rows compare visual-token attentionfor the baseline and GAS model on CV-Bench-2D [39] and MathVista [34]. The bottom row shows the full layer-by-layer attention evolution (layers 1–28): the baseline’s attention rapidly diffuses and collapses across mid-to-deep layers, whereas the GAS model preserves sharp, task-relevantfocus throughout the network depth and produces the corrected predictions shown in Boxes 2 and 3.  
Insight 5: Generation training enhances understanding through three complementary representational mechanisms: (i) improved visual information retention in deep layers, preventing the progressive attenuation observed in text-only training; (ii) enriched fine-grained spatial representations (validated by linear probing on RefCOCO); and (iii) sharper, task-focused attention patterns that persist throughout the reasoning chain.

Text-only Reasoning capability Mildly Improved. A natural concern is whether generation supervision harms the LLM backbone’s pure-text reasoning ability. We compare GAS with the understandingonly baseline on ZebraLogic [54] (constraint-satisfaction logic puzzles) and MMLU-Redux [55] (a re-annotated, less noisy subset of MMLU). Table 7 shows higher point estimates for GAS on both benchmarks: ZebraLogic puzzle accuracy increases by +2.8pp (with consistent improvements on small/medium puzzles and cell-level accuracy), and MMLU-Redux improves by +1.23pp. This indicates that generation-guided multimodal training does not act as a regularizer that erodes language reasoning; on the contrary, the joint optimization mildly benefits language-only capability, likely because the visual generation branch absorbs visual-specific gradient pressure that would otherwise distort the LLM’s text representations.

<table><tr><td rowspan="2">Model</td><td colspan="3">ZebraLogic</td><td rowspan="2">MMLU-Redux Acc</td></tr><tr><td>Puzzle Acc Cell Acc Small Acc</td><td></td><td></td></tr><tr><td>Qwen3-VL baseline</td><td>18.0</td><td>12.74</td><td>52.19</td><td>52.66</td></tr><tr><td>+ GAS (Ours)</td><td>20.8</td><td>14.79</td><td>56.56</td><td>53.89</td></tr></table>

Table 7: Text-only reasoning evaluation. GAS preserves and mildly improves pure-text reasoning on ZebraLogic and MMLU-Redux, indicating that generation supervision does not interfere with the LLM’s language capability.

## 5. Related Works

Multimodal large language models (MLLMs) have emerged as the prevailing paradigm for image and video understanding, typically by aligning pretrained visual encoders with powerful language backbones. Representative early systems include Flamingo [56] and InstructBLIP [57], whereas subsequent open-source families such as the LLaVA series [1, 4, 58] and Qwen-VL series [2, 45] have further advanced image perception and understanding. Most studies in this line largely follow the LLaVA paradigm [4], in which visual inputs are first encoded by a vision encoder [5, 59] and subsequently concatenated with text tokens for joint modeling by a language model decoder. Proprietary systems such as GPT [60] and Gemini [61, 62] likewise exhibit strong multimodal understanding capabilities. Recent advances have further extended these models to multimodal visual reasoning [36, 63, 64] and video understanding [65–67]. Despite their notable strengths in semantic abstraction and cross-modal alignment, these models are primarily optimized for text generation rather than learning from native visual information. The training signal flows exclusively from vision to language – visual tokens serve as a conditioning context for text prediction, but no supervisory gradient encourages the model to develop fine-grained, pixel-level visual perception or to retain visual evidence throughout deep reasoning chains. This asymmetry limits performance on tasks that demand precise spatial understanding, such as grounding, segmentation, and complex visual chain-of-thought reasoning.

Table 8: Design comparison ofrepresentative approaches that combine visual understanding and generation. The attributes characterize how generation is used during training rather than absolute model quality. GAS is the only compared design that combines all six properties
<table><tr><td>Method</td><td>Generation target</td><td>U- centric</td><td>Task- cond.</td><td>Exact U-space</td><td>No ext. codec</td><td>Upper U-isol.</td><td>Train- only aux.</td></tr><tr><td colspan="2">Native unified understanding-generation models</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Chameleon [10]</td><td>Discrete image tokens</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Show-o [11]</td><td>Discrete diffusion tokens</td><td>X</td><td></td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Janus-Pro [47]</td><td>Discrete image tokens</td><td>X</td><td>X</td><td>X</td><td>x</td><td>X</td><td>X</td></tr><tr><td>MetaMorph [46]</td><td>Vision-encoder features</td><td></td><td></td><td>X</td><td>x</td><td>X</td><td>X</td></tr><tr><td>BAGEL [13]</td><td>VAE/semantic features</td><td>X</td><td></td><td>X</td><td>X</td><td></td><td>X</td></tr><tr><td>UniFork [22]</td><td>Discrete image tokens</td><td>X</td><td>X</td><td>x</td><td>X</td><td></td><td>X</td></tr><tr><td colspan="2">Generation supervision designed to enhance understanding</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ROSS [68]</td><td>Continuous VAE features</td><td></td><td>X</td><td>X</td><td>X</td><td>X</td><td></td></tr><tr><td>ASVR [69]</td><td>Discrete semantic tokens</td><td></td><td>X</td><td>X</td><td>X</td><td>X</td><td></td></tr><tr><td>UniHetero [21]</td><td>LLM-input embeddings</td><td></td><td>X</td><td></td><td>X</td><td>X</td><td></td></tr><tr><td>UniMRG [70]</td><td>RGB/depth/segmentation</td><td></td><td></td><td>X</td><td>X</td><td></td><td></td></tr><tr><td>GAS (Ours)</td><td>LLM-input embeddings</td><td></td><td></td><td>L</td><td></td><td></td><td>V</td></tr></table>

Definitions. U-centric: generation is explicitly optimized to improve understanding rather than retained as a co-equal capability; Task-cond.: an observed visual input and instruction determine a distinct target, beyond generic text-to-image synthesis or input reconstruction; Exact U-space: the target is the projected continuous embedding consumed by the understanding LLM; No ext. codec: no separate visual tokenizer, VAE, or diffusion decoder defines the generation target; Upper-U isolated: generation gradients do not update the upper understanding transformer; Train-only aux.: generation-specific machinery is introduced for understanding training and removed at deployment. ● denotes partial or architecture-dependent support.

Recent unified multimodal models (UMMs) attempt to close this gap by jointly training understanding and generation within a single framework, thereby introducing visual synthesis objectives that force the model to engage more deeply with image structure. One line follows a fully autoregressive formulation: Chameleon [10] and Emu3 [16] cast both understanding and generation into next-token prediction over a shared discrete visual token space, offering a clean interface for mixed-modality sequence modeling. Another line adopts AR-diffusion hybrid formulations, combining language modeling for text with diffusion or flow-based objectives for visual generation. Representative works include Transfusion [12], Show-o [11, 48], and BLIP3-o [71]. Within this family, several works further explore architectural decoupling to mitigate the well-known conflict between generation and understanding objectives: Janus [47] decouples visual encoding pathways while sharing a single LLM backbone; MetaMorph [46] introduces Visual-Predictive Instruction Tuning and observes that improving visual understanding implicitly benefits generation; LMFusion [72] adds modality-specific transformer layers to preserve pretrained language capabilities during generation training; and BAGEL [13] scales on trillions of interleaved tokens and exhibits compositional generation and reasoning.

Most closely related to our goal, ROSS [68] reconstructs the input image through a denoising objective over continuous appearance features, whereas ASVR [69] autoregressively reconstructs discrete semantic visual tokens. UniHetero [21] further shows at large data scale that semantic generation— including autoregression on the LLM input embedding—can improve understanding, while pixel-level generation gradients can be detrimental. UniMRG [70] post-trains existing UMMs to generate RGB reconstructions, depth maps, and segmentation maps from an input image. These results strengthen the broad premise that visual-side supervision can improve understanding; accordingly, GAS’s distinction is the combination of instruction-conditioned output-image prediction across diverse generation tasks, direct supervision in the understanding input space, asymmetric upper-layer isolation, and a generation branch used only during training.

Appendix Table 10 further compares adapted ROSS-style and ASVR-style objectives under the same backbone, data mixture, and optimization setting. These controls test how their reconstruction objectives transfer to our generation-augmented setting rather than replacing the original papers’ reported results.

However, despite these architectural innovations, existing UMMs predominantly pursue generation quality or explore emergent capabilities from scale, while treating the understanding branch as a secondary beneficiary rather than the primary optimization target. Moreover, the introduction of generation objectives frequently degrades understanding performance due to conflicting gradient signals, a tension that current methods address by either accepting the tradeoff or increasing model capacity [19]. Currently, there exists no systematic investigation into methods for integrating visual generation supervision to preserve and further advance models’ semantic understanding capabilities.

## 6. Conclusion and Future Work

We present GAS, a generation-guided training framework that improves aggregate multimodal understanding by using visual generation purely as training-time auxiliary supervision. Concretely, NEP provides a generation target aligned with the LLM input space, while the removable MoT branch mitigates direct optimization conflict and preserves zero inference overhead. Across diverse understanding benchmarks, GAS delivers consistent gains, while representation analyses associate the gains with better visual retention, more focused attention, and task-dependent transfer shaped by generation-task correlation.

These results suggest that visual generation is most effective not as an inference-time capability to preserve, but as a carefully structured supervisory signal for building stronger understanding models. We believe this work establishes generation-guided training as a principled and practical paradigm for advancing multimodal understanding. Looking forward, several promising directions emerge: scaling the framework to longer-context scenarios, extending the NEP paradigm to video-level temporal generation tasks, exploring adaptive task weighting strategies that dynamically balance generation categories during training, and investigating whether the insights on generation-understanding synergy generalize to other modalities such as audio and 3D perception.

## References

[1] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

[2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-VL Technical Report, November 2025. URL http://arxiv.org/abs/2511.21631. arXiv:2511.21631 [cs].

[3] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

[4] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual Instruction Tuning, December 2023. URL http://arxiv.org/abs/2304.08485. arXiv:2304.08485 [cs].

[5] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[6] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. Advances in Neural Information Processing Systems, 37:8612–8642, 2024.

[7] Fatemeh Shiri, Xiao-Yu Guo, Mona Golestan Far, Xin Yu, Reza Haf, and Yuan-Fang Li. An empirical analysis on spatial reasoning capabilities of large multimodal models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 21440–21455, 2024.

[8] Yuhui Zhang, Alyssa Unell, Xiaohan Wang, Dhruba Ghosh, Yuchang Su, Ludwig Schmidt, and Serena Yeung-Levy. Why are visually-grounded language models bad at image classification? Advances in Neural Information Processing Systems, 37:51727–51753, 2024.

[9] Roy Ganz, Yair Kittenplon, Aviad Aberdam, Elad Ben Avraham, Oren Nuriel, Shai Mazor, and Ron Litman. Question aware vision transformer for multimodal reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13861–13871, 2024.

[10] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

[11] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao

Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=o6Ynz6OIQ6.

[12] Chunting Zhou, LILI YU, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=SI2hI0frk6.

[13] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

[14] Tao Shen, Xin Wan, Taicai Chen, Rui Zhang, Junwen Pan, Dawei Lu, Fanding Lei, Zhilin Lu, Yunfei Yang, Chen Cheng, et al. Mammothmoda2: A unified ar-diffusion framework for multimodal understanding and generation. arXiv preprint arXiv:2511.18262, 2025.

[15] Fengyi Fu, Mengqi Huang, Shaojin Wu, Yunsheng Jiang, Yufei Huo, Hao Li, Yinghang Song, Fei Ding, Jianzhu Guo, Qian He, Zheren Fu, Zhendong Mao, and Yongdong Zhang. Lance: Unified multimodal modeling by multi-task synergy. arXiv preprint arXiv:2605.18678, 2026.

[16] Xinlong Wang, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Zhen Li, Yuqi Wang, Qiying Yu, Yingli Zhao, Yulong Ao, Xuebin Min, Chunlei Men, Boya Wu, Bo Zhao, Bowen Zhang, Liangdong Wang, Guang Liu, Zheqi He, Xi Yang, Jingjing Liu, Yonghua Lin, Zhongyuan Wang, and Tiejun Huang. Multimodal learning with next-token prediction for large multimodal models. Nature, 650(8101):327–333, February 2026. ISSN 1476-4687. doi: 10.1038/s41586-025-10041-x.

[17] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12966–12977, 2025.

[18] Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large diffusion language models. Advances in Neural Information Processing Systems, 38: 138867–138907, 2026.

[19] Meituan LongCat Team, Bin Xiao, Chao Wang, Chengjiang Li, Chi Zhang, Chong Peng, Hang Yu, Hao Yang, Haonan Yan, Haoze Sun, et al. Longcat-next: Lexicalizing modalities as discrete tokens. arXiv preprint arXiv:2603.27538, 2026.

[20] Sihan Xu, Ziqiao Ma, Wenhao Chai, Xuweiyi Chen, Weiyang Jin, Joyce Chai, Saining Xie, and Stella X Yu. Next-embedding prediction makes strong vision learners. arXiv preprint arXiv:2512.16922, 2025.

[21] Fengjiao Chen, Minhao Jing, Weitao Lu, Yan Feng, Xiaoyu Li, and Xuezhi Cao. Unihetero: Could generation enhance understanding for vision-language-model at large data scale? arXiv preprint arXiv:2512.23512, 2025.

[22] Teng Li, Quanfeng Lu, Lirui Zhao, Hao Li, Xizhou Zhu, Yu Qiao, Jun Zhang, and Wenqi Shao. Unifork: Exploring modality alignment for unified multimodal understanding and generation. arXiv preprint arXiv:2506.17202, 2025.

[23] Jitai Hao, Hao Liu, Xinyan Xiao, Qiang Huang, and Jun Yu. Uni-x: Mitigating modality conflict with a two-end-separated architecture for unified multimodal models. arXiv preprint arXiv:2509.24365, 2025.

[24] Weixin Liang, LILI YU, Liang Luo, Srini Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen tau Yih, Luke Zettlemoyer, and Xi Victoria Lin. Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models. Transactions on Machine Learning Research, 2025. ISSN 2835-8856. URL https://openreview.net/forum?id=Nu6N69i8SB.

[25] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[26] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[27] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: international conference for high performance computing, networking, storage and analysis, pages 1–16. IEEE, 2020.

[28] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. Advances in Neural Information Processing Systems, 38, 2026.

[29] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9556–9567, 2024.

[30] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pages 148–166. Springer, 2024.

[31] Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural Information Processing Systems, 37:113569–113697, 2024.

[32] Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. In International Conference on Learning Representations, volume 2025, pages 48337–48383, 2025.

[33] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024.

[34] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations, volume 2024, pages 23439–23554, 2024.

[35] Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024.

[36] Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, Wenhai Wang, Jifeng Dai, and Jinguo Zhu. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=mXuzDDVXxi.

[37] Roni Paiss, Ariel Ephrat, Omer Tov, Shiran Zada, Inbar Mosseri, Michal Irani, and Tali Dekel. Teaching CLIP to Count to Ten. arXiv preprint arXiv:2302.12066, 2023.

[38] Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko Bošnjak, Xi Chen, Matthias Minderer, Paul Voigtlaender, Ioana Bica, Ivana Balazevic, Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut, Jeremiah Harmsen, and Xiaohua Zhai. PaliGemma: A versatile 3B VLM for transfer. arXiv preprint arXiv:2407.07726, 2024.

[39] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai C Akula, Jihan Yang,

Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024.

[40] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24108–24118, 2025.

[41] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan $\mathrm { { X u , } }$ Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024.

[42] Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, et al. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. arXiv preprint arXiv:2504.15279, 2025.

[43] YiFan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, Haonan fan, Kaibing Chen, Jiankang Chen, Haojie Ding, Kaiyu Tang, Zhang Zhang, Liang Wang, Fan Yang, Tingting Gao, and Guorui Zhou. Thyme: Think beyond images. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=gCWLkqK45O.

[44] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306, 2024.

[45] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL Technical Report, February 2025.

[46] Shengbang Tong, David Fan, Jiachen Li, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17001–17012, 2025.

[47] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.

[48] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Show-o2: Improved native unified multimodal models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. URL https://openreview.net/forum?id=7VMg7Jb7AL.

[49] Yichen Zhang, Da Peng, Zonghao Guo, Zijian Zhang, Xuesong Yang, Tong Sun, Shichu Sun, Yidan Zhang, Yanghao Li, Haiyan Zhao, et al. Cheers: Decoupling patch details from semantic representations enables unified multimodal comprehension and generation. arXiv preprint arXiv:2603.12793, 2026.

[50] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/coyo-dataset, 2022.

[51] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

[52] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 11–20, 2016.

[53] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects

in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787–798, 2014.

[54] Bill Yuchen Lin, Ronan Le Bras, Kyle Richardson, Ashish Sabharwal, Radha Poovendran, Peter Clark, and Yejin Choi. Zebralogic: On the scaling limits of llms for logical reasoning. arXiv preprint arXiv:2502.01100, 2025.

[55] Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, Alessio Devoto, Alberto Carlo Maria Mancino, Rohit Saxena, Xuanli He, Yu Zhao, Xiaotang Du, Mohammad Reza Ghasemi Madani, et al. Are we done with mmlu? In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5069–5096, 2025.

[56] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.

[57] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in neural information processing systems, 36:49250–49267, 2023.

[58] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024.

[59] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

[60] OpenAI. gpt-5-system-card, 2025.

[61] Introducing Gemini 2.0: our new AI model for the agentic era, December 2024. URL https://blog. google/technology/google-deepmind/google-gemini-ai-update-december-2024/.

[62] Google. Gemini 3 flash: frontier intelligence built for speed, 2025.

[63] Jiacong Wang, Zijian Kang, Haochen Wang, LiangXiao, Ya Wang, Jiawen Li, Bohong Wu, Ran Jiao, Haiyong Jiang, ChaoFeng, and Jun Xiao. VGR: Visual grounded reasoning. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=kDhAiaGzrn.

[64] Fucai Ke, Joy Hsu, Zhixi Cai, Zixian Ma, Xin Zheng, Xindi Wu, Sukai Huang, Weiqing Wang, Pari Delir Haghighi, Gholamreza Haffari, et al. Explain before you answer: A survey on compositional visual reasoning. arXiv preprint arXiv:2508.17298, 2025.

[65] Xiangrui Liu, Yan Shu, Zheng Liu, Ao Li, Yang Tian, and Bo Zhao. Video-xl-pro: Reconstructive token compression for extremely long video understanding, 2025. URL https://arxiv.org/abs/2503.18478.

[66] Yan Shu, Zheng Liu, Peitian Zhang, Minghao Qin, Junjie Zhou, Zhengyang Liang, Tiejun Huang, and Bo Zhao. Video-xl: Extra-long vision language model for hour-scale video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26160–26169, 2025.

[67] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, Yu Qiao, Yali Wang, and Limin Wang. Videochat-flash: Hierarchical compression for long-context video modeling. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=MUjdNcfNPv.

[68] Haochen Wang, Anlin Zheng, Yucheng Zhao, Tiancai Wang, Zheng Ge, Xiangyu Zhang, and Zhaoxiang Zhang. Reconstructive visual instruction tuning. In International Conference on Learning Representations, volume 2025, pages 14374–14399, 2025.

[69] Dianyi Wang, Wei Song, Yikun Wang, Siyuan Wang, Kaicheng Yu, Zhongyu Wei, and Jiaqi Wang. Autoregressive semantic visual reconstruction helps vlms understand better. In Findings of the Association for Computational Linguistics: ACL 2026, pages 38101–38115, 2026.

[70] Zihan Su, Hongyang Wei, Kangrui Cen, Yong Wang, Guanhua Chen, Chun Yuan, and Xiangxiang Chu. Generation enhances understanding in unified multimodal models via multi-representation generation. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/ forum?id=uYaNE2Cys2.

[71] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025.

[72] Weijia Shi, Xiaochuang Han, Chunting Zhou, Weixin Liang, Xi Victoria Lin, Luke Zettlemoyer, and LILI YU. LMFusion: Adapting pretrained language models for multimodal generation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. URL https://openreview.net/ forum?id=Kc1WTxZbrP.

## A. Additional Controlled Experiments

## A.1. Repeated-Run Robustness Across Data Scales

We repeat GAS three times in the 2.5B-token regime and twice in the 20B-token regime. Table 9 reports the sample mean and sample standard deviation, together with the original and longer-trained understanding-only controls. The GAS mean remains above both single-run controls in Overall at each scale. Variation is larger for individual capabilities—most notably Reasoning at 2.5B and Count&Spatial at 20B—so these runs support reproducible aggregate transfer.

<table><tr><td>Regime Configuration</td><td></td><td>Runs</td><td>Overall</td><td>Perception</td><td>Reasoning</td><td>Count&amp;Spatial</td><td>Video</td></tr><tr><td>2.5B</td><td>Baseline</td><td>1</td><td>47.25</td><td>56.55</td><td>31.38</td><td>76.92</td><td>46.60</td></tr><tr><td>2.5B</td><td>Longer baseline (2.77B; +11%)</td><td>1</td><td>47.73</td><td>57.63</td><td>34.01</td><td>71.28</td><td>45.55</td></tr><tr><td>2.5B</td><td>GAS</td><td>3</td><td></td><td></td><td>48.46±0.3758.56±0.46 32.99±1.34</td><td>75.83±1.06</td><td>47.28±0.35</td></tr><tr><td>20B</td><td>Baseline</td><td>1</td><td>51.95</td><td>60.12</td><td>38.88</td><td>79.62</td><td>47.10</td></tr><tr><td>20B</td><td>Longer baseline (22B; +10%)</td><td>1</td><td>52.14</td><td>60.18</td><td>38.58</td><td>80.90</td><td>46.95</td></tr><tr><td>20B</td><td>GAS</td><td>2</td><td></td><td>52.33±0.3361.04±0.4539.05±0.24</td><td></td><td>80.71±1.54</td><td>47.35±0.49</td></tr></table>

Table 9: Repeated-run robustness across training scales. GAS entries are mean ± sample standard deviation; baseline controls are single runs. Bold marks the best reported value within each regime and capability.

## A.2. Aligned Comparison with Reconstructive Objectives

We reproduce ROSS-style and ASVR-style objectives using the same Qwen3-VL-2B backbone, 2.5B understanding plus 2.5B generation data, and optimization setting as GAS. On generation samples, these adapted controls reconstruct the assistant output image using continuous appearance features and discrete semantic tokens, respectively. As shown in Table 10, input-reconstruction objectives designed for understanding data do not automatically transfer to this generation-augmented setting, whereas GAS’s task-conditioned output prediction gives the strongest aggregate result. These are controlled adaptations for isolating the objective under our setup, rather than reproductions of the original papers’ reported training regimes.

<table><tr><td>Method</td><td></td><td>Overall Perception Reasoning</td><td></td><td>Count&amp;Spatial Video</td><td></td></tr><tr><td>Understanding-only baseline</td><td>47.25</td><td>56.55</td><td>31.38</td><td>76.92</td><td>46.60</td></tr><tr><td>ROSS-style (U+G)</td><td>45.70</td><td>56.10</td><td>30.88</td><td>70.80</td><td>44.25</td></tr><tr><td>ASVR-style (U+G)</td><td>47.31</td><td>56.20</td><td>33.16</td><td>73.78</td><td>45.55</td></tr><tr><td>GAS (U+G)</td><td>48.25</td><td>58.52</td><td>32.63</td><td>75.15</td><td>47.65</td></tr></table>

Table 10: Same-backbone comparison with adapted reconstructive objectives. All U+G methods use the same backbone, data mixture, and optimization setting.

## B. More Qualitative Cases of GAS

![](images/e5db410d7daa1b920d2dfc3855809a74258d2913aec3d8035365b9bd4391859b.jpg)

Qualitative Case 4: VisuLogic [36]   
?   
A B C D   
Question: Choose the most appropriate one from the four given options to fill in the question mark so that a   
certain regularity is presented: A, B, C, D.   
Ours 4B:   
<think>   
To solve this problem, we first analyze the shapes and number of lines in each figure, then identify the underlying   
pattern.   
1. \*\*Analyze shape and line count:\*\*   
Observing the figures in the first and second rows, each figure is composed of triangles or quadrilaterals (rhombus-like   
shapes), with a regular line count.   
- Row 1, Fig. 1: two triangles, lines = 2 × 3 = 6;   
Row 1, Fig. 2: two quadrilaterals, lines = 2 × 4 = 8;   
Row 1, Fig. 3: two triangles, lines = 2 × 3 = 6;   
- Row 2, Fig. 1: two quadrilaterals, lines = 2 × 4 = 8;   
- Row 2, Fig. 2: two triangles, lines = 2 × 3 = 6;   
- Row 2, Fig. 3 (the question mark): should follow the same shape/line-count pattern and stacking style.   
2. \*\*Identify the pattern:\*\*   
The figures in both rows alternate between triangle pairs and quadrilateral pairs, with line counts of 6 and 8   
respectively. All figures share the same two-shape overlapping composition with consistent contour and inner-line   
connectivity.   
3. \*\*Match the options:\*\*   
Option D consists of two overlapping quadrilaterals (rhombus-like), whose line count and composition match the pattern   
(Row 2 starts with a quadrilateral pair, so the question mark should also be a quadrilateral pair). Option A is   
a triangle pair with a mismatched shape; Option B is a triangle pair with inconsistent line count; Option C is a   
quadrilateral but with a different stacking style.   
Therefore, the correct answer is D .   
</think>   
The correct answer is D .

![](images/5437b5480a082231daf9dce0f6b574c3d179b82034f7b11acbb396ae96977fff.jpg)  
Question: Find the chromatic number of the following graph.

## Ours 4B: Ours

<think> To determine the chromatic number of the given graph, we need to find the minimum number of colors required to color the vertices such that no two adjacent vertices share the same color. Let’s analyze the graph step by step. 1. \*\*Identify the Structure of the Graph:\*\* - The graph appears to be a bipartite graph. A bipartite graph is a graph whose vertices can be divided into two disjoint sets such that every edge connects a vertex in one set to a vertex in the other set. - In this graph, we can observe that the vertices can be divided into two sets: one set containing the vertices on the left side of the graph and the other set containing the vertices on the right side of the graph. 2. \*\*Verify Bipartiteness:\*\* - Check if there are any odd-length cycles in the graph. If there are no odd-length cycles, the graph is bipartite. - Upon inspection, the graph does not contain any odd-length cycles. All cycles in the graph are even-length cycles.   
3. \*\*Determine the Chromatic Number:\*\* - For a bipartite graph, the chromatic number is always 2. This is because we can color one set of vertices with one color and the other set with a different color, ensuring that no two adjacent vertices share the same color.   
4. \*\*Conclusion:\*\* - Since the graph is bipartite, the chromatic number is 2.   
</think>

The final answer is 2 .

Qualitative Case 6: RealWorldQA  
![](images/27c882b413005f85bfe4220262718fb2b5c9b8c011adae22a5665b8b03fbd9a9.jpg)  
Question: How many wooden poles are in this photo?  
Ours 4B:

## <think>

To determine the number of wooden poles in the photo, let’s analyze the image step by step:

1. \*\*Foreground Analysis\*\*: The image shows a small tree with a thin trunk. There are two visible wooden poles supporting the tree. These poles are positioned on either side of the tree, providing stability.

2. \*\*Background and Surroundings\*\*: The background includes a building with windows and some greenery. There are no additional wooden poles visible in the background or elsewhere in the image.

3. \*\*Counting the Poles\*\*: Based on the visible elements, there are exactly two wooden poles supporting the tree.

Given this analysis, the correct answer is:

\*\*C. There are 2 wooden poles.\*\*

</think>

The final answer is