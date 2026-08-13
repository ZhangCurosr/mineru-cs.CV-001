# HarmoniDPO: Video-guided Audio Generation via Preference-Optimized Difusion

Wenshuo Peng<sup>1</sup> and Kaipeng Zhang<sup>2\*</sup>

<sup>1</sup>Electornic Engineering Department, Tsinghua University, Haidian District, Beijing, 100084 China.

<sup>2\*</sup>OpenGVlab, Shanghai AI Laboratory, Xuhui District, Shanghai, 200030, China.

\*Corresponding author(s). E-mail(s): kpzhang@ut-vision.org; Contributing authors: gin2pws@gmail.com;

## Abstract

Video-to-audio (V2A) generation faces significant challenges in achieving precise temporal synchronization and high perceptual quality due to the complex, ambiguous relationship between visual and auditory cues. Existing methods typically compress video inputs into single feature representations, leading to significant loss of temporal dynamics and fine-grained visual information. These approaches also rely on reconstruction-based training objectives that poorly correlate with human perceptual judgments of audio quality and appropriateness. We propose HarmoniDPO, a novel framework that integrates preference-based optimization into difusion-based V2A generation to address these limitations. (1) Our approach leverages a dual video representation: combining global context with frame-wise features to preserve temporal dynamics and semantic detail. (2) Inspired by reinforcement learning from human feedback (RLHF), HarmoniDPO employs online Direct Preference Optimization (online-DPO) to fine-tune a difusion-based V2A model from preference judgments, enhancing perceptual quality and alignment. (3) Additionally, we introduce Dual-scale Difusion Search (DDS), a test time scaling algorithm that adaptively optimizes output fidelity during inference. Experiments demonstrate that HarmoniDPO outperforms state-of-the-art methods in audio-video synchronization and subjective audio quality, ofering a robust solution for generating realistic, human-preferred audio from video.

Keywords: video-to-audio generation; difusion models; Direct Preference Optimization (DPO); audio-visual alignment; reinforcement learning from human feedback (RLHF)

![](images/62af90b346856b5f8c27453ff7745f49977f02178a637f8962118da38443962d.jpg)  
Fig. 1 Overview of the HarmoniDPO framework. (a) Stage 1: The difusion model processes dual video inputs (global context features and frame-wise local features) with optional text conditioning to generate initial audio. (b) Stage 2: Online Direct Preference Optimization fine-tunes the model using pairwise audio comparisons to align outputs with preferences.

## 1 Introduction

Recent advancements in cross-modal learning have led to significant progress in audio generation. State-of-the-art models like AudioGen [1], Make-An-Audio [2], and AudioLDM [3] have demonstrated remarkable capabilities in synthesizing audio from text descriptions. As text-to-video generation [4, 5] continues to show promising applications, there is increasing interest in tackling the inverse problem: generating audio from video.

Video-to-audio generation presents significantly greater challenges than text-toaudio generation, primarily due to the following reasons: (1) Multimodal Alignment Requirements: The generated audio must synchronize with both the semantic content and temporal dynamics of the video. (2) Ambiguity in Audio-Visual Mapping and Perceptual Quality: The relationship between visual events and corresponding audio is inherently complex and often ambiguous. Visual content alone frequently under specifies the full range of desired audio characteristics, encompassing not just semantic alignment but also crucial aspects of perceptual quality, realism, and overall appropriateness. Determining the optimal audio solely from video cues is challenging, as multiple soundscapes might fit the visuals yet vary significantly in perceived quality without further optimization criteria.

Several methods have been proposed for video-to-audio generation, including difusion-based architectures [6], contrastive learning approaches [7], and textmediated techniques [8]. While these methods have shown progress, a critical limitation persists: ensuring the generated audio achieves high perceptual quality and aligns robustly with subjective human preferences regarding realism, style, and appropriateness for the given video context. The core issue is that standard training paradigms, often relying on objectives like L1 or L2 reconstruction loss against a single ground truth audio, are fundamentally insuficient. Such objectives act as poor proxies for complex human auditory perception; they struggle to capture the nuances that differentiate acceptable audio from truly high-quality, compelling audio, especially given the inherent ambiguity where multiple sounds could plausibly match the visuals. This reliance on simplistic loss functions fails to provide a strong alignment signal or reward for optimizing towards perceptually superior outcomes preferred by humans. Consequently, models trained this way may generate technically synchronized audio that lacks realism or fails to meet subjective quality standards. Beyond the fundamental limitations of reconstruction-based training, an even deeper issue emerges from the very nature of video-to-audio generation. The field sufers not only from data scarcity but also from an intrinsic mismatch between available training signals and desired model behavior. While architectural innovations have made progress in addressing temporal synchronization challenges, they fail to resolve the core disconnect between what standard training optimizes for (reconstruction of potentially noisy reference audio) and what ultimately matters (human perception of quality and appropriateness). This misalignment becomes particularly problematic given the one-to-many nature of video-to-audio mapping - where multiple plausible soundtracks could accompany the same visuals, each varying in subjective quality and stylistic preference. The critical bottleneck, therefore, lies not in the model’s capacity to reconstruct audio, but in its ability to discern and generate human-preferred outputs from this space of valid possibilities. This realization compels us to move beyond conventional paradigms and adopt preference-driven approaches that can directly optimize for perceptual quality, leveraging comparative judgments to guide the model toward solutions that better align with human evaluation criteria.

To overcome the limitations of current approaches, we introduce HarmoniDPO, an efective difusion-based V2A model with a novel framework that, for the first time, applies preference-based optimization to video-to-audio (V2A) generation. Inspired by Reinforcement Learning from Human Feedback (RLHF), our method employs online Direct Preference Optimization (online-DPO) through a two-stage process. (1) First, we develop an efective difusion-based V2A model (illustrated in Figure 1-Stage1), which is built upon a U-Net difusion architecture. Innovatively, HarmoniDPO processes dual video inputs: a global feature capturing overall context and fine-grained frame-wise features to preserve detail and facilitate precise temporal alignment, mitigating information loss common in single-feature representations. HarmoniDPO also supports optional text conditioning. (2) In the second stage, we fine-tune this pretrained HarmoniDPO latent difusion model (LDM) using our online-DPO strategy. This involves learning directly from preference comparisons between pairs of generated audio samples. The model is iteratively refined based on a learned reward function derived from these preferences, implicitly capturing human judgments of audio quality and video-audio congruence. This preference-based fine-tuning enables HarmoniDPO to generate audio that is perceptually superior and better aligned with desired characteristics (e.g., temporal synchrony, style consistency) beyond what standard training achieves, allowing it to efectively explore the output space and converge towards solutions reflecting human preferences.

Besides the model architecture design, we also address these challenges at the inference stage with a test-time scaling strategy inspired by advancements in the textto-image domain [9]. Specifically, we propose a Dual-scale Difusion Search (DDS) algorithm, which employs a dual-scale sampling mechanism to explore the latent space adaptively. This approach balances local exploitation and global exploration by combining small and large step sizes, ensuring more eficient and efective optimization during inference. We conduct experiments to demonstrate the superiority of our method, and our contributions can be summarized as follows:

• We identify that maintaining both global context and temporal dynamics is crucial for video-to-audio generation. Based on this insight, we propose a simple yet efective difusion-based V2A model that uses a complementary representation of global video features and frame-wise image features, significantly improving audio-video synchronization quality. Moreover, it accepts optional text inputs for further quality improvement and precise control.

• We identify that ensuring the generated audio achieves high perceptual quality and aligns robustly with subjective human preferences is a critical limitation. Hence, we pioneer the application of preference-based optimization to the V2A generation task, proposing an online Direct Preference Optimization (online-DPO) framework inspired by RLHF. This significantly enhances perceptual audio quality and alignment by fine-tuning based on human preferences.

• We develop a test-time scaling method specifically designed for video-to-audio generation, which optimizes the model’s output quality during inference. This method ensures high fidelity and temporal alignment of the generated audio with the input video.

• Experiments demonstrate that our method outperforms state-of-the-art methods in audio-video synchronization and subjective audio quality, ofering a robust solution for generating realistic, human-preferred audio from video.

## 2 Related Work

## 2.1 Audio Generation

Audio generation has emerged as a prominent research direction in recent years, encompassing both text-to-audio and video-to-audio generation tasks. Contemporary approaches in this field can be systematically categorized into two predominant frameworks: transformer-based autoregressive models and difusion-based architectures. These frameworks have been applied to various audio generation tasks, including text-conditioned audio synthesis, music generation, and video-synchronized audio generation.

Text-to-Audio Generation. Text-to-audio generation focuses on synthesizing audio content conditioned on textual descriptions. Early work in this domain includes Jukebox[10], which combined hierarchical VQ-VAE compression with autoregressive Transformers to produce diverse, high-fidelity music conditioned on genre, lyrics, and artist style. AudioGen [1], which pioneered an autoregressive framework for audio synthesis, capable of generating acoustic content from either textual descriptions or audio prompts. Expanding this paradigm, MusicGen [11] implements a sophisticated language modeling approach utilizing eficient token interleaving strategies to enable high-fidelity music generation from textual inputs. Voicebox[12] represents a breakthrough as the first truly general-purpose speech generation model. By training on 110K hours of diverse audiobook data using a non-autoregressive flow-matching approach, it achieves state-of-the-art performance on zero-shot text-to-speech, speech editing, and cross-lingual synthesis without requiring style labels. MAGNET[13] introduces masked generative modeling with non-autoregressive Transformers, operating on multi-stream audio representations. By iteratively predicting and rescoring masked token spans, it achieves comparable quality to autoregressive baselines while significantly reducing latency.

Difusion models have achieve remarkable cross-model generation ability in many works[14, 15]. In text-to-audio domain, DifSound [16] establishes a comprehensive framework integrating multiple specialized components: a text encoder, a Vector Quantized Variational Autoencoder (VQ-VAE), a token decoder, and a vocoder. DifSound uses a discretized VQ-VAE for encoding and decoding, combined with a non-autoregressive token decoder based on discrete difusion. Building on these advancements, AudioLDM [3] advances the field by implementing contrastive language-audio pretraining (CLAP) embeddings as latent conditional variables within a VAE framework for audio synthesis. Make-an-Audio [2] introduces an innovative approach through a spectrogram autoencoder that predicts self-supervised representations instead of direct waveforms, thereby facilitating enhanced compression eficiency and deeper semantic interpretation. Through the integration of CLAP embeddings [17] with high-fidelity difusion architectures, Make-an-Audio achieves sophisticated language comprehension capabilities alongside superior audio generation quality. AudioLDM2[18] proposes a unified framework that leverages a general audio representation called the “language of audio” (LOA), derived from the self-supervised AudioMAE model, which employs GPT-2 to translate multimodal inputs into LOA and utilizes a LDM for conditional audio generation. Stable Audio[19] introduces timing embeddings alongside text conditioning, allowing variable-length generation while maintaining musical structure.

Video-to-Audio Generation. Video-to-audio generation aims to learn joint representations of visual and auditory modalities to produce audio synchronized with visual content. Early works like Visually Indicated Sounds [20] developed a dataset of object interaction videos (hitting, scratching) and proposed a recurrent neural network that maps visual sequences to audio features, which are then converted to waveforms through either exemplar matching or parametric inversion. SpecVQGAN [21] presents an eficient framework for multi-class, visually guided sound synthesis using a transformer decoder trained to sample from a codebook-based prior. Foley Music[22] propose cross-modal music generation frameworks that map body keypoint movements to MIDI representations for synchronized music generation from silent videos.

Recent difusion-based approaches have shown significant progress, with Diffoley [7] employing latent difusion models utilizing contrastive audio-visual pretrained features, while FoleyCrafter [6] uses IP-adapter [23] connections to capture video semantics and employs audio event detection for temporal information. Frieren[24] proposes rectified flow-matching frameworks achieving superior quality through straight-trajectory ODE sampling and cross-modal feature fusion. V2A-Mapper [8] leverages pre-trained foundation models to bridge multimodal gaps for better open-domain handling.

Transformer-based methods include MaskVAT[25], which introduces masked generative transformers combining full-band neural audio codecs with multi-modal conditioning, and V-AURA[26], which proposes autoregressive frameworks using discrete audio tokens to avoid spectrogram-based information loss. Specialized applications focus on diferent audio types: Video2Music[27] and Vidmuse[28] target music synthesis through afective multimodal transformers and LSTV-modules respectively, while VarietySound[29] disentangles audio into temporal, timbre, and background components for sound efects generation. TiVA[30] uses unique audio layouts for prediction, Tri-Ergon[31] achieves precise loudness regulation via LUFS integration, and ReWaS[32] introduces energy-based ControlNet adapters for temporal alignment.

Current video-to-audio methods, whether reconstruction-based [20], transformerbased [26], or difusion-based [6, 7], share a common limitation: they optimize for reconstruction fidelity against ground-truth audio using standard losses (L1, L2), which poorly capture human auditory perception. This is particularly problematic given the one-to-many nature of video-to-audio mapping, where multiple plausible soundtracks could accompany the same visuals. HarmoniDPO addresses this fundamental limitation by pioneering preference-based optimization in V2A generation, using online Direct Preference Optimization (online-DPO) to directly learn from human comparative judgments rather than reconstruction targets, thus generating audio that better aligns with perceptual quality preferences.

## 2.2 Audio-visual Datasets

The advancement of audio generation research has been significantly propelled by the development of robust and diverse audio-visual datasets. These high-quality datasets serve as the cornerstone for both training and evaluating models, facilitating the explo ration of intricate audio synthesis tasks, including text-to-audio and video-to-audio generation. Among the pioneering datasets in this domain, VGGSound [33] has played a pivotal role. It comprises over 200,000 video clips spanning 310 diverse audio categories, providing a rich resource for general audio-visual learning tasks. However, despite the availability of such datasets, high-quality audio-visual data remains relatively scarce, particularly for non-speech domains. This scarcity stems from the fact that the majority of existing audio-visual datasets are predominantly composed of speech-related content, which limits their applicability to broader audio generation tasks.

FineVideo represents a significant efort to address this gap, ofering a highquality open video dataset hosted on HuggingFace. This collection encompasses over 43,000 videos, totaling approximately 3,400 hours of content with an average duration of 4.7 minutes per video. Each entry includes paired audio and text annotations, making it particularly valuable for multimodal research in video-to-audio generation and audio-text alignment. Beyond FineVideo, several other datasets have made substantial contributions to audio-visual research. Ego4D [34] provides a comprehensive human activity dataset containing 3,670 hours of footage, including 2,535 hours of audio content. The MultiTalk [35] dataset encompasses over 420,000 hours of multilingual talking head videos, while AudioSetCaps [36] combines AudioSet, YouTube-8M, and VGGSound to ofer approximately 16,000 hours of content with synthetic captions and question-answering capabilities. Additionally, AVCaps [37] contributes 28.8 hours of content with modality-specific captions, and MUSIC-AVQA [38] provides approximately 150 hours of specialized music-related audio-visual content for question-answering tasks.

## 2.3 Reinforcement Learning from Human Feedback

Reinforcement Learning from Human Feedback (RLHF) has emerged as a transformative paradigm for aligning AI systems with human preferences and values. Initially introduced by Christiano et al. [39], RLHF was first applied to reinforcement learning tasks such as robotics and game-playing agents. The methodology was later adapted to natural language processing, where it became a cornerstone for aligning large language models (LLMs) with human intent.

Stiennon et al.[40] pioneered the application of RLHF in the LLM training pipeline, establishing a three-stage framework: pre-training, supervised fine-tuning (SFT), and RLHF-based alignment. This framework was further refined and scaled in subsequent work, notably InstructGPT[41], which demonstrated significant improvements in aligning model outputs with human preferences through RLHF-based optimization. Direct Preference Optimization (DPO)[42] reformulated preference learning as a binary classification problem, eliminating the need for explicit reward modeling and reducing computational complexity. Concurrently, alternative approaches such as Reward Ranked Fine-Tuning (RAFT)[43] and Rank Responses to align Human Feedback (RRHF)[44] have introduced more eficient training paradigms for incorporating preference signals. Additionally, Constitutional AI[45] expanded the RLHF framework by incorporating explicit ethical considerations into the feedback loop, addressing safety and ethical alignment concerns.

The success of RLHF in language models has recently inspired its application to multimodal domains. These applications can be broadly categorized into two approaches: reward-based methods and DPO-based methods. Reward-based methods like ImageReward [46] train specialized reward models that capture human aesthetic and quality preferences to guide the generation process. In contrast, DPO-based methods such as Difusion-DPO [47] adapt the DPO methodology to multimodal tasks, primarily in image generation.

However, a fundamental limitation of conventional DPO approaches is their ofline nature. While standard DPO can achieve RLHF-like efects, it remains an ofline reinforcement learning method where the implicit “reward model” remains static throughout training. This limitation creates a critical distribution shift problem: as the policy model improves, it increasingly diverges from the preference model’s distribution, ultimately leading to performance bottlenecks. Our work addresses this limitation by introducing the first online DPO methodology for video-to-audio (V2A) generation. Our approach achieves the adaptive benefits of online reinforcement learning while preserving the computational eficiency of DPO. This innovation is particularly valuable in the V2A domain, where no prior alignment techniques have been applied, and where the complex, context-dependent nature of audio generation demands more dynamic preference modeling.

## 3 Method

In this section, we begin by introducing the foundational concepts of LDMs and their applications in audio generation. We then present our proposed method, HarmoniDPO $( H D P O )$ , which integrates video, image to enable cross-modal audio generation with enhanced performance.

## 3.1 Preliminaries

Latent difusion models (LDMs) [48] are a class of generative models built upon the principles of Denoising Difusion Probabilistic Models (DDPMs) [49]. LDMs have emerged as powerful frameworks in text-to-audio and video-to-audio generation tasks. A key feature of LDMs is that they operate in a lower-dimensional latent space, making them computationally eficient. In the context of audio, raw signals are first transformed into mel-spectrograms. The mel-spectrogram, a time-frequency representation, logarithmically scales its frequency axis to the Mel scale, which mirrors human auditory perception, enabling the model to efectively learn both temporal dynamics and spectral characteristics.

In a video-to-audio system, given an audio signal $x _ { a }$ and its corresponding video $x _ { v } ,$ we first use a pre-trained variational autoencoder $( \mathrm { V A E } )$ with an encoder $\mathcal { E } _ { \theta }$ to compress the mel-spectrogram representation of $x _ { a }$ into a low-dimensional latent representation, denoted as $z _ { 0 } = \mathcal { E } _ { \theta } ( x _ { a } )$ . This latent vector $z _ { 0 }$ is the target for the difusion model.

The forward difusion process gradually adds Gaussian noise to the latent vector $z _ { 0 }$ over a sequence of $T$ timesteps. This process is a fixed Markov chain defined as:

$$
q ( z _ { t } | z _ { t - 1 } ) = \mathcal { N } ( z _ { t } ; \sqrt { 1 - \beta _ { t } } z _ { t - 1 } , \beta _ { t } \mathbf { I } )\tag{1}
$$

where $\{ \beta _ { t } \} _ { t = 1 } ^ { T }$ is a pre-defined variance schedule. A useful property of this process is that we can sample $z _ { t }$ directly from $z _ { 0 }$ at any timestep t:

$$
\begin{array} { r } { q ( z _ { t } | z _ { 0 } ) = \mathcal { N } ( z _ { t } ; \sqrt { \bar { \alpha } _ { t } } z _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) \mathbf { I } ) } \end{array}\tag{2}
$$

where $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { t } ( 1 - \beta _ { s } ) } \end{array}$ . Using the reparameterization trick, any $z _ { t }$ can be expressed as $z _ { t } = \sqrt { \bar { \alpha } _ { t } } z _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon$ , where $\epsilon \sim \mathcal { N } ( 0 , \bf { I } )$

The reverse process aims to learn the true posterior $q ( \boldsymbol { z } _ { t - 1 } | \boldsymbol { z } _ { t } )$ , which would reverse the noising process. As this posterior is intractable, a neural network $\epsilon _ { \theta }$ trained to approximate it. As shown by Ho et al. [49], instead of modeling the reverse distribution’s mean directly, the training can be simplified to predicting the noise component ϵ that was added to create $z _ { t }$ . This leads to a simplified training objective, which is a mean-squared error loss between the true and predicted noise:

$$
\mathcal { L } _ { \mathrm { L D M } } = \mathbb { E } _ { z _ { 0 } , \epsilon , t } \left[ | | \epsilon - \epsilon _ { \theta } ( z _ { t } , t ) | | _ { 2 } ^ { 2 } \right]\tag{3}
$$

To enable conditional generation, the denoising network ϵ is augmented with conditioning information, such as a visual embedding $f _ { v } ,$ typically via a cross-attention

![](images/2228d3ad82777c2e9afcde9061329fe472339753a67ed45b29a8115405ac9dd7.jpg)  
Fig. 2 Illustration of our HarmoniDPO model architecture. We propose a dual-feature representation framework that combines global video features with fine-grained frame-wise features.

mechanism. The objective thus becomes:

$$
\mathcal { L } _ { \mathrm { L D M } } = \mathbb { E } _ { z _ { 0 } , \epsilon , t , f _ { v } } \left[ \Vert \epsilon - \epsilon _ { \theta } ( z _ { t } , t , f _ { v } ) \Vert _ { 2 } ^ { 2 } \right]\tag{4}
$$

During inference, given a visual embedding $f _ { v }$ , we can generate the corresponding audio by iterative denoising from random Gaussian noise $z _ { T } \mathbf { : }$

$$
z _ { t - 1 } = \frac { 1 } { \sqrt { 1 - \beta _ { t } } } \big ( z _ { t } - \frac { \beta _ { t } } { \sqrt { 1 - \bar { \alpha } t } } \epsilon _ { \theta } \big ( z _ { t } , t , f _ { v } \big ) \big )\tag{5}
$$

Given the limited size and varying quality of existing visual-audio datasets, we use a pre-trained text-to-audio difusion model, Tango-2 [50], as our base model, keeping it frozen throughout training. Figure 2 provides an overview of the framework, and we will describe each component of HarmoniDPO in detail in the following sections.

## 3.2 Visual Condition

Accurate video representation is essential for efectively capturing the semantic content of videos. Previous methods, such as FoleyCrafter, have relied on the CLIP image encoder to obtain average semantic features from individual frames. However, since CLIP is not trained on video datasets, this approach may fall short of capturing the temporal dynamics and motion patterns inherent in video content. To address this limitation, we propose a dual-feature representation framework that combines global video features with fine-grained frame-wise features, enabling comprehensive encoding of both spatial and temporal information.

As shown in Figure 2, for global video feature extraction, we leverage the video encoder from InternVid [51], a state-of-the-art contrastive video-language pre-training framework. Given an input video V containing T frames, the global video feature $f _ { g }$

is computed as:

$$
f _ { g } = { \mathcal { E } } _ { \mathrm { I n t e r n V i d } } ( V ) \in \mathbb { R } ^ { d _ { g } }\tag{6}
$$

where $\mathcal { E } _ { \mathrm { I n t e r n V i d } }$ represents the InternVid video encoder and $d _ { g }$ denotes the dimensionality of the extracted video feature. InternVid’s architecture extends $\mathrm { C L I P } \mathrm { { s } }$ visual understanding capabilities through continued pre-training on large-scale video-text datasets, enabling superior modeling of temporal relationships and motion dynamics across video sequences. The encoding process operates as follows: given a target video, we first establish a fixed frame sampling count M. The video is then temporally subsampled at M uniformly distributed timestamps to ensure consistent temporal coverage. The resulting frames are concatenated and processed through a pre-trained Vision Transformer to produce the final global video representation.

While the global video features efectively capture high-level semantic content and temporal dynamics, fine-grained frame-level details remain crucial for maintaining precise temporal alignment between the generated audio and specific visual events. Therefore, we implement a complementary frame-level processing stream that operates parallel to global feature extraction. We uniformly sample N frames $\{ v _ { 1 } , v _ { 2 } , \ldots , v _ { N } \}$ from the input video and process each frame through the CLIP visual encoder to obtain frame-wise embeddings:

$$
e _ { i } = \mathcal { E } _ { \mathrm { C L I P } } ( v _ { i } ) \in \mathbb { R } ^ { d _ { f } } , \quad i \in { 1 , 2 , \dots , N }\tag{7}
$$

where $\mathcal { E } _ { \mathrm { C L I P } }$ represents the CLIP image encoder and $d _ { f }$ is the dimension of frame features. In our implementation, we sample $N = 2 5 0$ frames from the original video as uniformly as possible. For videos containing fewer than 250 frames, we pad the sequence with zero-filled frame features. Note that the majority of samples in our training dataset exceed 250 frames, making padding necessary for only a small portion of the data.

For efective multimodal fusion, we first use projection layers to align the dimensions of global and frame-level features to a common dimension $d _ { c } \mathbf { \hat { \Psi } }$

$$
f _ { g } ^ { \prime } = W _ { g } f _ { g } \in \mathbb { R } ^ { d _ { c } }\tag{8}
$$

$$
e _ { i } ^ { \prime } = W _ { f } e _ { i } \in \mathbb { R } ^ { d _ { c } }\tag{9}
$$

where $W _ { g } ~ \in ~ \mathbb { R } ^ { d _ { c } \times d _ { g } }$ and $W _ { f } ~ \in ~ \mathbb { R } ^ { d _ { c } \times d _ { f } }$ are learnable projection matrices. We then concatenate the projected global video feature with the projected frame-wise embeddings:

$$
F = [ f _ { g } ^ { \prime } ; e _ { 1 } ^ { \prime } ; e _ { 2 } ^ { \prime } ; \ldots ; e _ { N } ^ { \prime } ] \in \mathbb { R } ^ { ( 1 + N ) \times d _ { c } }\tag{10}
$$

Finally, we employ another projection layer to transform the concatenated features to the model’s hidden dimension:

$$
F ^ { \prime } = W _ { h } F \in \mathbb { R } ^ { ( 1 + N ) \times d _ { h } }\tag{11}
$$

where $W _ { h } \in \mathbb { R } ^ { d _ { h } \times d _ { c } }$ is the final projection matrix and $d _ { h }$ is the hidden dimension of the model.

To better capture temporal relationships among the fused features, we incorporate Rotary Position Embeddings (RoPE) [52] within a multi-head self-attention mechanism. The self-attention operation employs separate learned projections for queries, keys, and values:

$$
Q = F ^ { \prime } W _ { q } \in \mathbb { R } ^ { ( 1 + N ) \times d _ { h } } , K = F ^ { \prime } W _ { k } \in \mathbb { R } ^ { ( 1 + N ) \times d _ { h } } , V = F ^ { \prime } W _ { v } \in \mathbb { R } ^ { ( 1 + N ) \times d _ { h } }\tag{12}
$$

where $W _ { q } , W _ { k } , W _ { v } \in \mathbb R ^ { d _ { h } \times d _ { h } }$ are learnable projection matrices for queries, keys, and values, respectively. The final fused representation is obtained by:

$$
F _ { \mathrm { f u s e d } } = \mathrm { A t t e n t i o n } _ { \mathrm { R o P E } } ( Q , K , V ) \in \mathbb { R } ^ { ( 1 + N ) \times d _ { h } }\tag{13}
$$

where RoPE is applied to both queries Q and keys K before computing attention scores, enabling the model to encode sequential ordering of frames through relative position information. These fused representations are subsequently incorporated into the difusion model via a cross-attention layer (visual-attention), enabling the denoising process to dynamically attend to spatio-temporal features during training.

Building upon the comprehensive visual, our method integrates these multimodal conditions into a difusion-based audio generation framework. As illustrated in Figure $^ { 3 , }$ the generation process operates within the latent space of a pre-trained Variational Autoencoder (VAE) for computational eficiency. The reverse difusion process begins with noisy latent variables $z _ { T }$ and iteratively denoises them through a U-Net architecture, where the fused visual representations $F _ { \mathrm { f u s e d } }$ are injected via cross-attention layers to guide the denoising process. The final denoised latent $z _ { 0 }$ is then decoded through the VAE decoder to generate the target audio output.

Our model also supports the optional textual conditioning for enhanced control over the generation process. For the Text input, we use the pre-trained FLAN-T5[53] text encoder to extract the text embedding $f _ { t }$ . Given a text prompt $T ,$ , the text embedding $f _ { t }$ is computed as:

$$
f _ { t } = \mathcal { E } _ { \mathrm { T 5 } } ^ { \mathrm { t e x t } } ( T ) \in \mathbb { R } ^ { d _ { t } }\tag{14}
$$

where $\mathcal { E } _ { \mathrm { T 5 } } ^ { \mathrm { t e x t } }$ denotes the FLAN-T5 text encoder and $d _ { t }$ is the dimension of the text feature. We use a projection layer to match the hidden dimension. The difusion model then integrates these textual conditions through the cross-attention layer.

## 3.3 Direct Preference Optimization

Recent advances in preference-based optimization have demonstrated the efectiveness of Reinforcement Learning from Human Feedback (RLHF) for aligning large language models (LLMs). However, its application to cross-modal generation tasks, such as video-to-audio synthesis, remains underexplored. In this paper, We propose the first online-DPO framework for cross-modal preference alignment in video-conditioned audio generation. Unlike ofline DPO that requires pre-collected static datasets, our method continuously adapts to human preferences through real-time interaction during the generation process.

![](images/db1d7d71705cc328a084cfa0ed2e34ff2345491b888d7638804b70941513ec22.jpg)  
Fig. 3 Overview of the video-conditioned audio generation pipeline. The diagram illustrates the reverse difusion process, where noisy latent variables $z _ { t }$ are iteratively denoised through a U-Net architecture.

## 3.3.1 Theoretical Foundation

Direct Preference Optimization (DPO) establishes a direct mapping between reward functions and optimal policies through three fundamental principles:

1. Optimal Policy Characterization: For any reward function $r ( v , a )$ and reference policy $\pi _ { \mathrm { r e f } }$ , the KL-constrained optimal policy has closed-form solution:

$$
\pi ^ { * } ( a | v ) = { \frac { \pi _ { \mathrm { r e f } } ( a | v ) \exp ( r ( v , a ) / \beta ) } { Z ( v ) } }\tag{15}
$$

where $\begin{array} { r } { Z ( v ) = \sum _ { a } \pi _ { \mathrm { r e f } } ( a | v ) \exp ( r ( v , a ) / \beta ) } \end{array}$ is the partition function, v and a are the inputs and generated outputs. In our paper, v represents the input video and a represents the output audio.

2. Reward-Policy Duality: Equation 15 permits reward reparameterization via policy ratios:

$$
r ( v , a ) = \beta \log \frac { \pi ^ { * } ( a | v ) } { \pi _ { \mathrm { r e f } } ( a | v ) } + \beta \log Z ( v )\tag{16}
$$

3. Preference Modeling: Human judgments follow the Bradley-Terry model:

$$
p ^ { * } ( a _ { w } \succ a _ { l } | v ) = \frac { \exp ( r ( v , a _ { w } ) ) } { \exp ( r ( v , a _ { w } ) ) + \exp ( r ( v , a _ { l } ) ) }\tag{17}
$$

where $a _ { w }$ denotes the preferred output over $a _ { l }$ for a given input v.

Substituting 16 into 17 yields the DPO objective:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D P O } } = - \mathbb { E } _ { ( v , a _ { w } , a _ { l } ) \sim \mathcal { D } } \left[ \log \sigma \left( \beta \log \frac { \pi _ { \theta } ( a _ { w } | v ) } { \pi _ { \mathrm { r e f } } ( a _ { w } | v ) } - \beta \log \frac { \pi _ { \theta } ( a _ { l } | v ) } { \pi _ { \mathrm { r e f } } ( a _ { l } | v ) } \right) \right] } \end{array}\tag{18}
$$

where $\sigma ( \cdot )$ denotes the sigmoid function, and D is the preference dataset.

![](images/4cab73db275a024a2dbb24f53fd4066c67b28c44e1d1263a3cac5c1658f20419.jpg)  
Fig. 4 Pipeline of one Online-DPO process. We use a subset of VGGSound to fine-tune the whole model.

## 3.3.2 Online-DPO

While DPO provides an elegant framework for preference optimization, its conventional ofline formulation faces fundamental scalability challenges. The approach critically depends on large-scale human-annotated preference data, which becomes prohibitively expensive to collect for complex cross-modal tasks like video-to-audio generation—where expert annotators must evaluate subtle audiovisual synchronization and acoustic quality. This dependency creates a bottleneck for adapting to new domains or evaluation criteria, as each requires fresh annotation campaigns. Moreover, the static nature of ofline datasets leads to inevitable obsolescence: as generative models improve, human judgments collected on older model outputs become misaligned with current capabilities, causing the preference model to optimize for outdated behaviors. This problem is exacerbated in dynamic multi-modal settings where aesthetic standards and technical expectations evolve rapidly, rendering pre-collected preferences increasingly irrelevant over time.

Inspired by recent work [54], we propose an Online-DPO method which can explore an iterative self-training approach and uses no human-annotated preferences in the training loop. As illustrate in Figure 4, our onlin-DPO mainly consists of two parts: 1) areward modeling, and 2) iterative training using the modeled rewards.

Reward Modeling Building upon our motivation for preference-based optimization, we develop an automated reward modeling framework that addresses the key limitations of reconstruction-based training identified earlier. Recognizing that human preference annotation at scale is prohibitively expensive while raw audio-visual pairs may not reflect true perceptual quality, we propose a multi-dimensional reward system that captures the essential aspects of human evaluation. Our approach evaluates generated audio through three critical lenses that mirror human judgment criteria: (1) temporal and semantic alignment with visual content, (2) consistency with textual descriptions (when available), and (3) intrinsic acoustic quality. This tripartite evaluation directly addresses the ambiguity and subjectivity challenges highlighted in our introduction, where multiple audio outputs could technically match the video but vary significantly in perceptual quality.

Specifically, we assess the quality of the generated audio along three primary dimensions:

• Audio-Visual Correspondence: The temporal and contextual alignment between the synthesized audio and the input video content.

• Audio-Text Consistency: The accuracy with which the audio reflects accompanying textual descriptions.

• Intrinsic Audio Quality: The acoustic fidelity and clarity of the generated audio itself, irrespective of the conditioning inputs.

To operationalize this multi-faceted evaluation, we utilize specific automated metrics for each dimension. Audio-visual correspondence is quantified via the similarity score obtained from the pre-trained contrastive learning model CAV-MAE [55]. Similarly, audio-text consistency is assessed using the similarity score derived from a dedicated audio-text contrastive learning model CLAP. Most crucially, we employ the recently proposed Audiobox-Aesthetics predictor from Meta [56]. This predictor provides non-intrusive, utterance-level aesthetic scores based on human perceptual dimensions, trained across diverse audio types (speech, music, sounds), ofering a robust measure of perceptual quality suitable for generative models.

Given that the three reward components—audio-visual correspondence similarity $\left( R _ { a v } \right)$ , audio-text consistency similarity $\left( R _ { a t } \right)$ , and the intrinsic audio quality score $\left( R _ { q u a l i t y } \right)$ from Audiobox-Aesthetics—operate on potentially diferent scales and represent distinct facets of audio generation quality, a direct summation is inadequate for creating a unified reward signal. To address this, we first normalize each component independently to ensure they contribute appropriately to the final reward. The final composite reward score $R ( y )$ for a generated audio sample y is then calculated as a weighted linear combination of these normalized scores:

$$
R ( y ) = w _ { a v } \cdot R _ { a v } ^ { \prime } ( y ) + w _ { a t } \cdot R _ { a t } ^ { \prime } ( y ) + w _ { q u a l i t y } \cdot R _ { q u a l i t y } ^ { \prime } ( y )\tag{19}
$$

Here, $w _ { a v } , w _ { a t }$ , and $w _ { q u a l i t y }$ are non-negative hyperparameters representing the relative importance assigned to audio-visual correspondence, audio-text consistency, and intrinsic quality, respectively.

## Iterative Training with Value Aware DPO Loss

A key limitation of conventional ofline DPO is its reliance on a static preference dataset. This dataset, collected prior to training, prevents the policy from receiving direct feedback on the quality variations within its own generated outputs during the alignment process. Consequently, the model may not eficiently learn to refine subtle aspects captured by the reward function, leading to suboptimal performance. To overcome this challenge, our Online-DPO framework implements an Online Preference Sample Generation strategy integrated within the iterative training loop.

Specifically, instead of drawing from a fixed dataset, preference pairs are generated dynamically in each training iteration t. For a given input condition x (video and text), the current policy $\pi _ { t }$ generates a set of N candidate audio samples, denoted as $\mathcal { Y } = \{ y _ { 1 } , y _ { 2 } , . . . , y _ { N } \}$ . Each candidate $y _ { i } \in \mathcal { V }$ is subsequently evaluated using our composite automated reward function $R ( y _ { i } )$

Crucially, we then identify the best-performing $\left( y _ { w } \right)$ and worst-performing (y ) samples within this generated set $\mathcal { V }$ based purely on the automated reward scores:

$$
y _ { w } = \operatorname * { a r g m a x } _ { y _ { i } \in \mathcal { V } } R ( y _ { i } ) \quad \mathrm { a n d } \quad y _ { l } = \operatorname * { a r g m i n } _ { y _ { i } \in \mathcal { V } } R ( y _ { i } )\tag{20}
$$

These dynamically generated preference pairs $\{ x , y _ { w } , y _ { l } \}$ form the online dataset $\mathcal { D } _ { t }$ used to update the policy $\pi _ { \theta }$

To further enhance learning eficiency and sensitivity to the magnitude of preference, we propose a Value-Aware DPO (VA-DPO) loss that explicitly incorporates reward magnitudes. Unlike standard DPO, which treats all preference pairs equally (as a binary choice), our loss accounts for the degree of preference between samples by integrating the reward margin. The resulting VA-DPO loss is defined as:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { V A - D P O } } ( \pi _ { \theta } ; \pi _ { \mathrm { r e f } } ) = - \mathbb { E } _ { ( x , y _ { w } , y _ { l } ) } \bigg [ \log \sigma \bigg ( \beta \log \frac { \pi _ { \theta } ( y _ { w } | x ) } { \pi _ { \mathrm { r e f } } ( y _ { w } | x ) } - \beta \log \frac { \pi _ { \theta } ( y _ { l } | x ) } { \pi _ { \mathrm { r e f } } ( y _ { l } | x ) } } \\ & { ~ - \lambda \big ( R ( y _ { w } ) - R ( y _ { l } ) \big ) \bigg ) \bigg ] \quad } \end{array}\tag{21}
$$

where $\sigma ( \cdot )$ is the sigmoid function, $\beta$ is the temperature parameter controlling the weight of the policy diference, and λ is a hyperparameter balancing the DPO term and the reward margin term. Considering that difusion models are not like Large Language Models(LLMs), we follow Difusion-DPO[47, 50] for the loss calculation. Regarding the reference model $\pi _ { \mathrm { r e f } } .$ , we update it periodically; specifically, at the end of each training epoch in our implementation.

## 3.4 Test-time Scaling

To enhance the inference performance of our video-to-audio difusion model, we propose a Dual-scale Difusion Search (DDS) strategy. This approach introduces a dual-scale sampling mechanism that adaptively explores the latent space by combining both small and large step sizes, thereby balancing local exploitation and global exploration.

The DDS algorithm maintains a population of candidate solutions that evolve over multiple iterations. Initially, the population $P _ { 0 }$ is sampled from a standard normal distribution. At each iteration, for every candidate (generated audios) in the population, we generate two potential updates using diferent difusion scales: a conservative step with parameter $\beta _ { s }$ for local exploitation and a more aggressive step with parameter $\beta _ { l }$ for global exploration $\left( \beta _ { l } > \beta _ { s } \right)$ . Specifically, given a candidate solution, we generate

two new candidates:

$$
x _ { s } = \beta _ { s } x + \sqrt { 1 - \beta _ { s } ^ { 2 } } \cdot \eta\tag{22}
$$

$$
x _ { l } = \beta _ { l } x + \sqrt { 1 - \beta _ { l } ^ { 2 } } \cdot \eta\tag{23}
$$

where $\eta$ is sampled from $\mathcal { N } ( 0 , 1 )$

To evaluate the quality of the generated audio, we employ a zero-shot evaluation approach. Specifically, we utilize the CLIP score [57] to measure the audio-visual correspondence and the CLAP score to assess the audio-text alignment. These metrics provide a comprehensive evaluation of the generated audio in terms of both visual and textual relevance. The complete DSS procedure is presented in Algorithm 1

```latex
Algorithm 1 Dual-scale Difusion Search
Require: $\overline { { D , T , N , \beta _ { s } , \beta _ { l } , \mathcal { F } } }$
Ensure: best solution, best score
Initialize $P _ { 0 }$ with N samples from $\mathcal { N } ( 0 , 1 )$
best score $\Leftarrow - \infty$
best solution ⇐ ∅
for $i \Leftarrow 1$ to N do
if $\mathcal { F } ( P _ { 0 } [ i ] ) >$ best score then
best score $\begin{array} { r } { \Leftarrow \mathcal { F } ( P _ { 0 } [ i ] ) } \end{array}$
best candidate $\gets P _ { 0 } [ i ]$
end if
end for
for $t \Leftarrow 1$ to $T$ do
for $i \Leftarrow 1$ to N do
Sample η from $\mathcal { N } ( 0 , 1 ) ^ { D }$
$x _ { s } \Leftarrow \beta _ { s } P _ { t - 1 } [ i ] + \sqrt { 1 - \beta _ { s } ^ { 2 } } \cdot \eta$
x<sub>l</sub> $\Leftarrow \beta _ { l } P _ { t - 1 } [ i ] + \sqrt { 1 - \beta _ { l } ^ { 2 } } \cdot \eta$
if $\mathcal { F } ( x _ { s } ) > \mathcal { F } ( x _ { l } )$ then
$P _ { t } [ i ] \Leftarrow x _ { s }$
if $\mathcal { F } ( x _ { s } ) >$ best score then
best score $\Leftarrow \mathcal { F } ( x _ { s } )$
best candidate $\Leftarrow x _ { s }$
end if
else
$P _ { t } [ i ] \Leftarrow x _ { l }$
if $\mathcal { F } ( x _ { l } ) >$ best score then
best score $\Leftarrow \mathcal { F } ( x _ { l } )$
best candidate $\Leftarrow x _ { l }$
end if
end if
end for
end for
```

## 4 Experiments

## Dataset

To train our proposed HarmoniDPO, we utilize the VGGSound [33] dataset, adhering to its original train/test splits. For evaluation, we assess our method using both the VGGSound and AVSync15 [58] datasets.

## Implementation Details

We use Tango-2 [50], a pre-trained text-to-audio difusion model as our based difusion model. For visual feature extraction, we utilize OpenCLIP ViT-H/14 [59] as our image encoder and ViCLIP-L-14 as our video encoder, where the latter is pre-trained on the InternVid-10M-FLT[51] dataset.

During training, We use 64 H800 GPUs for experiments. We employ the AdamW optimizer with a constant learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 128 visual-audio embedding pairs. To enhance model generalization, we apply classifier-free guidance with a dropout rate of 0.05. The training process consists of two distinct phases: (1) In the visual conditioning phase, we keep the pre-trained difusion model frozen while fine-tuning only the newly added projection head and cross-attention layers to adapt the text-to-audio difusion model into a video-to-audio model; (2) In the subsequent alignment stage, we first sample 4500 videos independently and identically distributed (i.i.d.) from the VGGSound dataset. For the online-DPO fine-tuning, we generate candidate audio samples for these videos. To ensure the generated candidates are of high quality and relevance, we condition the generation on the original text descriptions associated with each video. Furthermore, inspired by the methodology in Tango-2, we create diverse candidate sets by varying parameters such as the number of difusion steps and the classifier-free guidance scale during generation. We set $w _ { a v } .$ $w _ { a t }$ and $w _ { q u a l i t y }$ to be equal (1) for the reward modeling stage, adopting a balanced combination of audio-visual alignment, audio-text relevance, and perceptual quality scores. Finally, we perform end-to-end fine-tuning of the entire model using the online-DPO method, leveraging these generated candidates to learn preference alignments. This two-phase approach ensures stable training while efectively aligning the visual and audio modalities.

## Metrics

To evaluate the performance, we employ a comprehensive set of metrics that assess various aspects of the generated audio, including: Mean KL Divergence (MKL)[60] to measure the sample-level similarity between the generated audio and the ground truth; CLIP Similarity to evaluate the semantic coherence between the input video and the generated audio embeddings, using Wav2CLIP[57] as the audio encoder and CLIP as the video encoder, as done in previous works [6, 8]; Frechet Inception Distance (FID)[61] and Frechet Audio Distance (FAD)[62] with VGGish[63] to assess the fidelity and distribution similarity of the generated audio; CLAP Similarity to evaluate the cross-modal alignment between text and generated audio; and Onset Acc (onset detection accuracy) and Onset AP (onset detection average precision) [64] to evaluate the generated audios, using the onset ground truth from the datasets.

## Baseline

For comparison, we use current state-of-the-art method SpecVQGAN [60], Dif-Foley [7], V2A-Mapper [8], Seeing-and-hearing [65] and Foley-crafter [6]. SpecVQGAN generates audio tokens autoregressively from video tokens, while Dif-Foley applies contrastive learning to achieve synchronized audio synthesis via its CAVP encoders. V2A-Mapper aligns image representations with audio embeddings in CLAP space, facilitating video-based audio generation through a pre-trained model. Seeing-andhearing uses ImageBind [66] as a connector between visual and audio domains, enabling multimodal generation by integrating existing audio and video generators. FoleyCrafter introduces both semantic and temporal adapters to enhance video-to-audio generation.

## 4.1 Qualitative Evaluation

Table 1 Comparison of results on VGGSound dataset. The results show that our method can achieve better results with less data. The number underlined indicates the second best result.
<table><tr><td>Method</td><td>Training Set</td><td>MKL↓CLIP ↑</td><td></td><td>FD↓</td><td>FAD ↓CLAP ↑</td><td></td></tr><tr><td>SpecVQGAN</td><td>VGGSound, VAS</td><td>3.40</td><td>5.88</td><td>32.01</td><td>5.79</td><td></td></tr><tr><td>Diff-Foley</td><td>VGGSound, AudioSet-V2A</td><td>3.32</td><td>9.17</td><td>29.03</td><td>6.23</td><td>20.5</td></tr><tr><td>V2A-Mapper</td><td>VGGSound, Land-scape</td><td>2.84</td><td>9.72</td><td>24.16</td><td>1.34</td><td>24.5</td></tr><tr><td>FoleyCrafter</td><td>VGGSound, AudioSet Strong</td><td>2.56</td><td>10.70</td><td>19.67</td><td>2.78</td><td>25.3</td></tr><tr><td>Fieren</td><td>VGGSound</td><td>2.58</td><td>11.83</td><td>12.48</td><td>3.32</td><td>24.7</td></tr><tr><td>HarmoniDPO(w/o alignment)</td><td>VGGSound</td><td>2.04</td><td>12.54</td><td>7.72</td><td>1.75</td><td>28.35</td></tr><tr><td>HarmoniDPO (aligned)</td><td>VGGSound</td><td>1.86</td><td>13.27</td><td>6.54</td><td>1.64</td><td>31.75</td></tr><tr><td>HarmoniDPO (aligned + DDS)</td><td>VGGSound</td><td>1.82</td><td>13.65</td><td>6.42</td><td>1.59</td><td>32.57</td></tr></table>

## Audio quality and cross-modal alignment

We evaluate HarmoniDPO’s performance in audio generation quality and cross-modal alignment using the VGGSound dataset and AVSync-15 dataset, with quantitative results presented in Table 1 and Table 2. To ensure a fair comparison against baselines that only use visual input, we evaluate HarmoniDPO without leveraging text conditioning, thus avoiding any potential advantage from multimodal inputs.

For the VGGSound dataset, our evaluation specifically examines three progressively enhanced configurations of HarmoniDPO: 1) The base model utilizing pre-trained video and fine-grained frame embeddings without explicit alignment (HarmoniDPO (w/o alignment)). 2) The model enhanced with DPO-based alignment (HarmoniDPO (aligned)). 3) The DPO-aligned model further enhanced with Dualscale Difusion Search (DDS) during inference, where we set the number of search candidates N to 4 in our experiments.

Our experimental results demonstrate significant performance improvements across these configurations. The base model already surpasses baseline methods by efectively leveraging pre-trained video embeddings and frame-level visual features. The

DPO-aligned version shows marked improvements in audio-visual synchronization, particularly evident in CLIP and CLAP scores. The full model with DDS inference achieves optimal performance, maintaining competitive FAD scores while significantly outperforming all baselines in other metrics.

Table 2 Comparison of results on AVSync15 dataset.
<table><tr><td>Method</td><td>Training Set</td><td>MKL↓</td><td>CLIP↑</td><td>FID ↓</td></tr><tr><td>SpecVQGAN</td><td>VGGSound, VAS</td><td>3.603</td><td>6.474</td><td>75.56</td></tr><tr><td>Diff-Foley</td><td>VGGSound, AudioSet-V2A</td><td>1.963</td><td>10.38</td><td>65.77</td></tr><tr><td>Seeing and Hearing</td><td>VGGSound, Land-scape</td><td>2.547</td><td>2.033</td><td>65.82</td></tr><tr><td>FoleyCrafter</td><td>VGGSound, AudioSet Strong</td><td>1.479</td><td>11.94</td><td>36.80</td></tr><tr><td>HarmoniDPO (Ours)</td><td>VGGSound</td><td>1.332</td><td>14.38</td><td>31.59</td></tr></table>

For the AVSync-15 dataset, we evaluate all methods on the test split, which consists of 150 distinct examples. The AVSync-15 dataset, curated from VGGSound’s most temporally challenging samples, serves as our benchmark for evaluating synchronization capabilities. We include a time alignment experiment in the subsequent analysis and limit our evaluation to three fundamental metrics. As shown in Table 2, our HarmoniDPO method achieves competitive or superior performance across all metrics. Specifically, HarmoniDPO excels in CLIP match scores and FID, highlighting its ability to generate audio-visual content that is both semantically aligned and visually coherent.

## Time alignment

To evaluate temporal synchronization, we conduct experiments on the AVSync15 dataset. This dataset is constructed from high-quality videos sourced from VGGSound, which have been carefully filtered and segmented to retain only the most precise videoaudio pairs while discarding irrelevant or inactive segments. As shown in Table 3, our method achieves state-of-the-art performance in temporal synchronization, demonstrating its efectiveness in aligning audio with visual events.

Table 3 Comparison of results on AVSync15 dataset.
<table><tr><td>Method</td><td>Onset ACC ↑ Onset AP ↑</td></tr><tr><td>SpecVQGAN</td><td>26.74 63.18</td></tr><tr><td>Diff-Foley 21.18</td><td>66.55</td></tr><tr><td>Seeing and Hearing</td><td>20.95 60.33</td></tr><tr><td>FoleyCrafter 28.48</td><td>68.14</td></tr><tr><td>HarmoniDPO (Ours) 32.53</td><td>69.97</td></tr></table>

## 4.2 Online-DPO alignment

Table 4 Comparison of results on VGGSound dataset. To enhance the robustness of the conclusions, we report the results without using DDS method.
<table><tr><td>Num</td><td>MKL ↓</td><td>CLIP↑</td><td>FID↓</td><td>FAD ↓</td><td>CLAP ↑</td></tr><tr><td>2</td><td>1.97</td><td>12.33</td><td>8.34</td><td>1.77</td><td>29.35</td></tr><tr><td>4</td><td>1.83</td><td>12.95</td><td>7.76</td><td>1.73</td><td>30.54</td></tr><tr><td>8</td><td>1.856 ± 0.04</td><td>13.27 ± 0.132</td><td>6.54 ± 0.054</td><td>1.64 ± 0.013</td><td>31.75 ± 0.155</td></tr><tr><td>12</td><td>1.847</td><td>13.15</td><td>6.67</td><td>1.65</td><td>32.15</td></tr></table>

## Number of candidates

Having established the general efectiveness of our online-DPO approach (as detailed in Table 1). We now delve deeper into optimizing its performance by examining the influence of the number of candidate audios generated by our HarmoniDPO model. This hyperparameter dictates how many potential audio samples are evaluated during each step of the online alignment process.

Our experiments on the VGGSound dataset (Table 4) reveal that increasing the number of candidates generally enhances model performance, though the optimal setting varies across metrics. Specifically, performance improves significantly when scaling from two to eight candidates, with CLIP, FID, and FAD metrics achieving their best results at this point. Interestingly, we observe metric-specific variations: MKL peaks with four candidates, while CLAP scores continue to improve up to twelve candidates.

This divergence across metrics suggests they measure diferent aspects of audio quality. For example, the continued improvement in CLAP with 12 candidates—while FID and FAD slightly worsen—indicates that the model can generate audio that better matches the semantic content but may have subtle acoustic imperfections. In other words, higher CLAP scores do not always mean better overall audio: some outputs may “sound right” in context but contain unnatural textures or background noise. Conversely, low FID/FAD indicates realistic sound statistics but doesn’t guarantee perfect semantic alignment. Given this trade-of, we find the 8-candidate setting achieves a good balance, performing well on both types of metrics.

## User study results

We conducted a comprehensive human evaluation of our Online-DPO framework using the AVSync15 test set, comparing four configurations: baseline (no alignment), and alignment with 4, 8, and 12 candidate audios. To ensure evaluation robustness while mitigating subjective bias, we recruited eight evaluators from our laboratory to participate in two rounds of assessments. In each round, evaluators scored two randomly selected videos per category using replacement sampling. To ensure adequate video coverage while maintaining methodological rigor, we implemented a controlled repetition mechanism: each evaluator was guaranteed to rate at least three unique videos per category, with precisely one repeated video across the two rounds. This approach yielded four total ratings per evaluator-category pair (three unique videos plus one repeated video), achieving 75% unique sample coverage while permitting controlled reliability assessment through the intentional repetition.

![](images/1e52e6ca54a16d41db5a83025679d9b3c361095d6db8b503d9eb76ba17b7663b.jpg)  
Fig. 5 Comparative analysis of user evaluation scores across HarmoniDPO configurations. The grouped bars represent Overall Quality (OVL), Video-Audio Relevance (REL), and their Average (AVG) scores, with error bars indicating evaluation consistency. Results show optimal performance at 8 candidates. Candidates are the generated audio samples.

We evaluate the generated audio outputs along two critical dimensions: i) Overall Quality (OVL)[3] and ii) Relevance to the Input Video (REL)[3]. Evaluators provided Likert-scale ratings (1-5) for each sample, with higher scores indicating better performance. We tested

As illustrated in Figure 5, we observe consistent improvements in both audio quality and video-audio synchronization across diferent candidate configurations. The results demonstrate that HarmoniDPO with 8 candidates achieves the best balance, particularly excelling in video-audio relevance (REL) with a score of 3.95, while maintaining strong overall quality (OVL) at 3.92.

## Visualization Results

To validate the benefits of our proposed Online-DPO alignment module, we conduct an ablation study comparing two variants of our model: (1) the original implementation without Online-DPO alignment (serving as baseline), and (2) the enhanced version incorporating Online-DPO. As illustrated in Figure 6, we evaluate both versions on a 10-second valley stream video excluded from training data (non-VGGSound source).

The baseline model (without Online-DPO) generates audio with three notable characteristics: first, it exhibits uniformly strong amplitudes across all frequency bands;

![](images/f079cea6a370647d5d6623ae5eaf4d8d9514373c85e9639ef4c02ce136a7ecd8.jpg)  
Fig. 6 Comparative spectrogram analysis between our base model (without Online-DPO alignment) and the Online-DPO-enhanced version, evaluated on a 10-second unseen valley stream video. Using Online-DPO method can efectively attenuate ambient noise while optimizing frequency response for mid-range enhancement, resulting in superior noise reduction and vocal clarity.

second, this results in perceptually louder output with less controlled spectral distribution; third, while maintaining basic audiovisual correspondence, the audio quality appears noisier and less refined.

In contrast, the Online-DPO-enhanced version demonstrates significant improvements: (a) the spectral energy becomes more focused in the mid-frequency range, (b) both high and low frequency extremes are properly attenuated, and (c) the resulting audio maintains excellent visual correspondence while achieving noticeably better perceptual quality - sounding more natural and comfortable. This comparison efectively demonstrates how Online-DPO acts as a refinement module that optimizes audio generation quality without compromising content alignment.

## 4.3 Text condition

We demonstrate our optional text conditioning capabilities through visualization results in Figure 7. We evaluate three diferent text prompts: “talking”, “pouring” and “pouring milk to a cup” applied to the same video sequence.

When the text prompt is “talking”, the generated audio exhibits characteristic speech patterns in the mel spectrogram, with distinct temporal variations and amplitude fluctuations that correspond to natural speech dynamics. The frequency distribution shows typical vocal formant structures with intermittent energy patterns reflecting conversational rhythm.

When the text prompt is “pouring”, the resulting audio demonstrates more continuous and concentrated spectral energy, particularly in higher frequency components. The mel spectrogram reveals sustained, dense frequency content that captures the acoustic characteristics of liquid flow, with consistent energy distribution across time that aligns with pouring actions.

![](images/c5b635abc806ce26483e09763ee458419d7e9df0d89a1fdbfe1f77fcf4d4b6d4.jpg)

![](images/10ddc3afe7eb879700c605c214e38895745b70ffbc302afec2581bb67c078274.jpg)  
Prompt: talking

![](images/54faff690233152367c6cc9c13d77d88fe1f2a54200c3777e690864ed7395b41.jpg)  
Prompt: pouring

![](images/96856931cc37c0b0d818806cbf019e6cc241af75f06f96489fc79f26897a9801.jpg)  
Prompt: pouring milk to a cup  
Fig. 7 Text conditioning visualization results showing mel spectrograms generated with diferent text prompts: “talking”, “pouring”, and “pouring milk to a cup” applied to the same video sequence.

Table 5 We evaluated the performance using both CLIP average features and InternVid features for the given video. The InternVid features can produce better performance.
<table><tr><td>Method</td><td>MKL ↓ CLIP</td><td>个 FID ↓</td></tr><tr><td>CLIP average pooling</td><td>1.727</td><td>12.01 38.93</td></tr><tr><td>Frame features only</td><td>1.842</td><td>12.95 38.35</td></tr><tr><td>InternVid video features</td><td>1.379 13.94</td><td>32.16</td></tr></table>

Most notably, when using the more specific prompt “pouring milk to a cup”, the generated audio achieves superior alignment with the visual content. The mel spectrogram shows moderated amplitude levels with frequency characteristics that more accurately represent the subtle acoustics of milk being poured into a container. The spectral energy is well-controlled, avoiding excessive loudness while maintaining the essential frequency components associated with liquid-container interaction sounds. This demonstrates that more detailed text descriptions enable our model to generate audio with finer acoustic nuances that better match both the semantic content and the visual context of the scene.

## 4.4 Ablation Study

## Efect of video condition

To evaluate the eficacy of our dual visual feature framework, we conducted comparative experiments with two alternative approaches: (1) replacing the InternVid video encoder with CLIP image encoder using average pooling (following FoleyCrafter’s methodology), and (2) utilizing only image frame embeddings. All experiments were performed on the base model (without alignment), with the projection layer and visual-attention mechanism fine-tuned on the VGGSound dataset.

The results in Table 5 reveal several key findings. First, InternVid features achieve superior performance across all metrics, reducing MKL by 6.3% compared to frame embeddings and by 20.1% versus CLIP pooling. The CLIP score improvements are equally notable, with +7.6% and +16.1% gains respectively. Most significantly, FID scores demonstrate the clearest advantage of video-specific features, showing +4.6% and +17.4% improvements. These consistent gains highlight that temporalaware video features substantially outperform static image-based representations for audio-visual synchronization.

Table 6 Comparison of results of image frame condition.
<table><tr><td>Method</td><td>MKL↓</td><td>CLIP↑</td><td>FID ↓</td></tr><tr><td>Video features only</td><td>1.894</td><td>11.33</td><td>39.45</td></tr><tr><td>Video + Image frame features</td><td>1.379</td><td>13.94</td><td>32.16</td></tr></table>

We conduct a systematic evaluation of our dual visual conditioning approach by comparing two configurations: (1) using only InternVid video-level features, and (2) our complete solution combining both video-level and frame-level features. We show the results in Table 6.

The combined use of video-level and frame-level features yields a +0.515 gain in synchronization accuracy (MKL), demonstrating superior temporal alignment between modalities. For semantic consistency, we observe a +23.0% improvement in CLIP score, indicating enhanced cross-modal understanding when leveraging both global and local visual information. The audio quality metric shows a +18.5% enhancement, confirming that hierarchical visual representation leads to more natural sound generation.

## 5 Conclusion

In this paper, we present HarmoniDPO, a novel framework leveraging difusion models and preference optimization for video-to-audio synthesis that generates high-quality audio synchronized with visual input. Our approach uniquely pioneers the application of online Direct Preference Optimization (online-DPO) to the V2A task, enabling the model to learn directly from preferences regarding audio quality and alignment. Our framework utilizes a dual video representation, combining global context with fine-grained frame-wise features, to achieve precise temporal alignment between visual events and generated audio. The integration of online-DPO allows the model to refine outputs based on learned reward functions, significantly improving perceptual quality and video-audio congruence. Additionally, we introduce DDS (Dual-scale Difusion Search), a novel inference-time optimization algorithm that enhances generation quality through adaptive test-time scaling. Through comprehensive experimental evaluation, we demonstrate that HarmoniDPO consistently outperforms existing approaches in generating high-quality, temporally synchronized audio content that better reflects perceptual preferences and aligns seamlessly with input videos.

## 6 Limitation

While HarmoniDPO demonstrates significant advancements in video-to-audio synthesis, there are several areas for future improvement. One potential avenue for enhancing performance is the incorporation of higher-quality and larger audio-visual datasets. The largest available dataset, VGGSound, contains fewer than 200,000 samples, with each video limited to just 10 seconds in length. This constraint restricts our model’s ability to generalize to longer videos and capture extended temporal dynamics. Moreover, most existing datasets provide only a single reference audio per video, even though multiple sound interpretations (e.g., diferent footsteps or background music) can be equally valid. As a result, common evaluation metrics may favor outputs close to the reference while penalizing other plausible generations, leading to a mismatch between automated scores and human perceptual preferences.

Acknowledgements. This work was supported in part by the National Key R&D Program of China (NO.2022ZD0160101).

## References

[1] Kreuk, F., Synnaeve, G., Polyak, A., Singer, U., D´efossez, A., Copet, J., Parikh, D., Taigman, Y., Adi, Y.: Audiogen: Textually guided audio generation. arXiv preprint arXiv:2209.15352 (2022)

[2] Huang, R., Huang, J., Yang, D., Ren, Y., Liu, L., Li, M., Ye, Z., Liu, J., Yin, X., Zhao, Z.: Make-an-audio: Text-to-audio generation with prompt-enhanced difusion models. In: International Conference on Machine Learning, pp. 13916– 13932 (2023). PMLR

[3] Liu, H., Chen, Z., Yuan, Y., Mei, X., Liu, X., Mandic, D., Wang, W., Plumbley, M.D.: Audioldm: Text-to-audio generation with latent difusion models. arXiv preprint arXiv:2301.12503 (2023)

[4] Khachatryan, L., Movsisyan, A., Tadevosyan, V., Henschel, R., Wang, Z., Navasardyan, S., Shi, H.: Text2video-zero: Text-to-image difusion models are zero-shot video generators. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15954–15964 (2023)

[5] Ge, S., Nah, S., Liu, G., Poon, T., Tao, A., Catanzaro, B., Jacobs, D., Huang, J.- B., Liu, M.-Y., Balaji, Y.: Preserve your own correlation: A noise prior for video difusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 22930–22941 (2023)

[6] Zhang, Y., Gu, Y., Zeng, Y., Xing, Z., Wang, Y., Wu, Z., Chen, K.: Foleycrafter: Bring silent videos to life with lifelike and synchronized sounds. arXiv preprint arXiv:2407.01494 (2024)

[7] Luo, S., Yan, C., Hu, C., Zhao, H.: Dif-foley: Synchronized video-to-audio synthesis with latent difusion models. Advances in Neural Information Processing Systems 36 (2024)

[8] Wang, H., Ma, J., Pascual, S., Cartwright, R., Cai, W.: V2a-mapper: A lightweight solution for vision-to-audio generation by connecting foundation models. In:

Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, pp. 15492–15501 (2024)

[9] Ma, N., Tong, S., Jia, H., Hu, H., Su, Y.-C., Zhang, M., Yang, X., Li, Y., Jaakkola, T., Jia, X., et al.: Inference-time scaling for difusion models beyond scaling denoising steps. arXiv preprint arXiv:2501.09732 (2025)

[10] Dhariwal, P., Jun, H., Payne, C., Kim, J.W., Radford, A., Sutskever, I.: Jukebox: A generative model for music. arXiv preprint arXiv:2005.00341 (2020)

[11] Copet, J., Kreuk, F., Gat, I., Remez, T., Kant, D., Synnaeve, G., Adi, Y., D´efossez, A.: Simple and controllable music generation. Advances in Neural Information Processing Systems 36 (2024)

[12] Le, M., Vyas, A., Shi, B., Karrer, B., Sari, L., Moritz, R., Williamson, M., Manohar, V., Adi, Y., Mahadeokar, J., et al.: Voicebox: Text-guided multilingual universal speech generation at scale. Advances in neural information processing systems 36, 14005–14034 (2023)

[13] Ziv, A., Gat, I., Lan, G.L., Remez, T., Kreuk, F., D´efossez, A., Copet, J., Synnaeve, G., Adi, Y.: Masked audio generation using a single non-autoregressive transformer. arXiv preprint arXiv:2401.04577 (2024)

[14] Peng, W., Zhang, K., Zhang, S.Q.: T3m: Text guided 3d human motion synthesis from speech. In: NAACL-HLT (Findings) (2024)

[15] Zhang, Y., Chen, J.-K., Lyu, J., Wang, Y.-X.: V2edit: Versatile video difusion editor for videos and 3d scenes. arXiv preprint arXiv:2503.10634 (2025)

[16] Yang, D., Yu, J., Wang, H., Wang, W., Weng, C., Zou, Y., Yu, D.: Difsound: Discrete difusion model for text-to-sound generation. IEEE/ACM Transactions on Audio, Speech, and Language Processing 31, 1720–1733 (2023)

[17] Wu, Y., Chen, K., Zhang, T., Hui, Y., Berg-Kirkpatrick, T., Dubnov, S.: Largescale contrastive language-audio pretraining with feature fusion and keyword-tocaption augmentation. In: ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5 (2023). IEEE

[18] Liu, H., Yuan, Y., Liu, X., Mei, X., Kong, Q., Tian, Q., Wang, Y., Wang, W., Wang, Y., Plumbley, M.D.: Audioldm 2: Learning holistic audio generation with self-supervised pretraining. IEEE/ACM Transactions on Audio, Speech, and Language Processing (2024)

[19] Evans, Z., Carr, C., Taylor, J., Hawley, S.H., Pons, J.: Fast timing-conditioned latent audio difusion. In: Forty-first International Conference on Machine Learning (2024)

[20] Owens, A., Isola, P., McDermott, J., Torralba, A., Adelson, E.H., Freeman, W.T.: Visually indicated sounds. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2405–2413 (2016)

[21] Iashin, V., Rahtu, E.: Taming visually guided sound generation. In: British Machine Vision Conference (BMVC) (2021)

[22] Gan, C., Huang, D., Chen, P., Tenenbaum, J.B., Torralba, A.: Foley music: Learning to generate music from videos. In: European Conference on Computer Vision, pp. 758–775 (2020). Springer

[23] Ye, H., Zhang, J., Liu, S., Han, X., Yang, W.: Ip-adapter: Text compatible image prompt adapter for text-to-image difusion models. arXiv preprint arXiv:2308.06721 (2023)

[24] Wang, Y., Guo, W., Huang, R., Huang, J., Wang, Z., You, F., Li, R., Zhao, Z.: Frieren: Eficient video-to-audio generation network with rectified flow matching. Advances in Neural Information Processing Systems 37, 128118–128138 (2024)

[25] Pascual, S., Yeh, C., Tsiamas, I., Serr\`a, J.: Masked generative video-to-audio transformers with enhanced synchronicity. In: European Conference on Computer Vision, pp. 247–264 (2024). Springer

[26] Viertola, I., Iashin, V., Rahtu, E.: Temporally aligned audio for video with autoregression. In: ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5 (2025). IEEE

[27] Kang, J., Poria, S., Herremans, D.: Video2music: Suitable music generation from videos using an afective multimodal transformer model. Expert Systems with Applications 249, 123640 (2024)

[28] Tian, Z., Liu, Z., Yuan, R., Pan, J., Liu, Q., Tan, X., Chen, Q., Xue, W., Guo, Y.: Vidmuse: A simple video-to-music generation framework with long-shortterm modeling. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 18782–18793 (2025)

[29] Cui, C., Zhao, Z., Ren, Y., Liu, J., Huang, R., Chen, F., Wang, Z., Huai, B., Wu, F.: Varietysound: Timbre-controllable video to sound generation via unsupervised information disentanglement. In: ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5 (2023). IEEE

[30] Wang, X., Wang, Y., Wu, Y., Song, R., Tan, X., Chen, Z., Xu, H., Sui, G.: Tiva: Time-aligned video-to-audio generation. In: Proceedings of the 32nd ACM International Conference on Multimedia, pp. 573–582 (2024)

[31] Li, B., Yang, F., Mao, Y., Ye, Q., Chen, H., Zhong, Y.: Tri-ergon: Fine-grained

video-to-audio generation with multi-modal conditions and lufs control. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, pp. 4616–4624 (2025)

[32] Jeong, Y., Kim, Y., Chun, S., Lee, J.: Read, watch and scream! sound generation from text and video. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, pp. 17590–17598 (2025)

[33] Chen, H., Xie, W., Vedaldi, A., Zisserman, A.: Vggsound: A large-scale audio-visual dataset. In: ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 721–725 (2020). IEEE

[34] Grauman, K., Westbury, A., Byrne, E., Chavis, Z., Furnari, A., Girdhar, R., Hamburger, J., Jiang, H., Liu, M., Liu, X., et al.: Ego4d: Around the world in 3,000 hours of egocentric video. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18995–19012 (2022)

[35] Sung-Bin, K., Chae-Yeon, L., Son, G., Hyun-Bin, O., Ju, J., Nam, S., Oh, T.-H.: Multitalk: Enhancing 3d talking head generation across languages with multilingual video dataset. arXiv preprint arXiv:2406.14272 (2024)

[36] Bai, J., Liu, H., Wang, M., Shi, D., Wang, W., Plumbley, M.D., Gan, W.-S., Chen, J.: Audiosetcaps: An enriched audio-caption dataset using automated generation pipeline with large audio and language models. arXiv preprint arXiv:2411.18953 (2024)

[37] Sudarsanam, P., Mart´ın-Morat´o, I., Hakala, A., Virtanen, T.: Avcaps: An audiovisual dataset with modality-specific captions

[38] Li, G., Wei, Y., Tian, Y., Xu, C., Wen, J.-R., Hu, D.: Learning to answer questions in dynamic audio-visual scenarios. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19108–19118 (2022)

[39] Christiano, P.F., Leike, J., Brown, T., Martic, M., Legg, S., Amodei, D.: Deep reinforcement learning from human preferences. Advances in neural information processing systems 30 (2017)

[40] Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., Christiano, P.F.: Learning to summarize with human feedback. Advances in neural information processing systems 33, 3008–3021 (2020)

[41] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.: Training language models to follow instructions with human feedback. Advances in neural information processing systems 35, 27730–27744 (2022)

[42] Rafailov, R., Sharma, A., Mitchell, E., Manning, C.D., Ermon, S., Finn, C.:

Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems 36, 53728–53741 (2023)

[43] Dong, H., Xiong, W., Goyal, D., Zhang, Y., Chow, W., Pan, R., Diao, S., Zhang, J., Shum, K., Zhang, T.: Raft: Reward ranked finetuning for generative foundation model alignment. arXiv preprint arXiv:2304.06767 (2023)

[44] Yuan, Z., Yuan, H., Tan, C., Wang, W., Huang, S., Huang, F.: Rrhf: Rank responses to align language models with human feedback without tears. arXiv preprint arXiv:2304.05302 (2023)

[45] Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., McKinnon, C., et al.: Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073 (2022)

[46] Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., Dong, Y.: Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems 36, 15903–15935 (2023)

[47] Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., Naik, N.: Difusion model alignment using direct preference optimization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8228–8238 (2024)

[48] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684–10695 (2022)

[49] Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)

[50] Majumder, N., Hung, C.-Y., Ghosal, D., Hsu, W.-N., Mihalcea, R., Poria, S.: Tango 2: Aligning difusion-based text-to-audio generations through direct preference optimization. In: Proceedings of the 32nd ACM International Conference on Multimedia, pp. 564–572 (2024)

[51] Wang, Y., He, Y., Li, Y., Li, K., Yu, J., Ma, X., Li, X., Chen, G., Chen, X., Wang, Y., et al.: Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942 (2023)

[52] Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., Liu, Y.: Roformer: Enhanced transformer with rotary position embedding. Neurocomputing 568, 127063 (2024)

[53] Chung, H.W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., et al.: Scaling instruction-finetuned language

[54] Wang, T., Kulikov, I., Golovneva, O., Yu, P., Yuan, W., Dwivedi-Yu, J., Pang, R.Y., Fazel-Zarandi, M., Weston, J., Li, X.: Self-taught evaluators. arXiv preprint arXiv:2408.02666 (2024)

[55] Gong, Y., Rouditchenko, A., Liu, A.H., Harwath, D., Karlinsky, L., Kuehne, H., Glass, J.: Contrastive audio-visual masked autoencoder. arXiv preprint arXiv:2210.07839 (2022)

[56] Tjandra, A., Wu, Y.-C., Guo, B., Hofman, J., Ellis, B., Vyas, A., Shi, B., Chen, S., Le, M., Zacharov, N., et al.: Meta audiobox aesthetics: Unified automatic quality assessment for speech, music, and sound. arXiv preprint arXiv:2502.05139 (2025)

[57] Wu, H.-H., Seetharaman, P., Kumar, K., Bello, J.P.: Wav2clip: Learning robust audio representations from clip. In: ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 4563–4567 (2022). IEEE

[58] Zhang, L., Mo, S., Zhang, Y., Morgado, P.: Audio-synchronized visual animation. arXiv preprint arXiv:2403.05659 (2024)

[59] Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., Jitsev, J.: Reproducible scaling laws for contrastive language-image learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2818–2829 (2023)

[60] Iashin, V., Rahtu, E.: Taming visually guided sound generation. arXiv preprint arXiv:2110.08791 (2021)

[61] Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017)

[62] Kilgour, K., Zuluaga, M., Roblek, D., Sharifi, M.: Fr\’echet audio distance: A metric for evaluating music enhancement algorithms. arXiv preprint arXiv:1812.08466 (2018)

[63] Hershey, S., Chaudhuri, S., Ellis, D.P., Gemmeke, J.F., Jansen, A., Moore, R.C., Plakal, M., Platt, D., Saurous, R.A., Seybold, B., et al.: Cnn architectures for large-scale audio classification. In: 2017 Ieee International Conference on Acoustics, Speech and Signal Processing (icassp), pp. 131–135 (2017). IEEE

[64] Xie, Z., Yu, S., He, Q., Li, M.: Sonicvisionlm: Playing sound with vision language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26866–26875 (2024)

[65] Xing, Y., He, Y., Tian, Z., Wang, X., Chen, Q.: Seeing and hearing: Opendomain visual-audio generation with difusion latent aligners. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7151–7161 (2024)

[66] Girdhar, R., El-Nouby, A., Liu, Z., Singh, M., Alwala, K.V., Joulin, A., Misra, I.: Imagebind: One embedding space to bind them all. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15180– 15190 (2023)