# CoinVE-200K: A Large-Scale High-Quality Dataset for Compositional Instruction-Guided Video Editing

Smart Creation Platform Department, Online Video BU, Tencent

Fuchen Long, Cong Wang, Zitao Gao, Wenhao Zhong, Yu Cheng, Xiaolu Hou, Yan Li, Xiao Cao, Xinlong Sun<sup>†</sup>, Xi Chen<sup>B</sup>, Yu Liu

<sup>†</sup>Project Leader <sup>B</sup>Corresponding Author

The quality and diversity of instruction-based video editing datasets are steadily improving, yet existing datasets mainly focus on single editing operations and fall short in supporting compositional instruction-guided video editing. Particularly, multiple editing intents need to be jointly understood and faithfully executed within the same video. To address this issue, we introduce CoinVE-200K, a large-scale, high-quality dataset for Compositional Instruction-Guided Video Editing. The proposed dataset contains 1080p video-editing pairs of up to 201 frames covering diverse compositional editing scenarios, where each sample involves 2 to 5 atomic editing operations. Editing instructions span multiple target subjects, including humans, objects, and backgrounds, covering a broad range of edit types such as addition, removal, modification, and stylization. All samples are constructed through a carefully designed data generation and quality filtering pipeline to ensure instruction faithfulness, visual quality, temporal consistency, and compositional diversity. Compared with existing instruction-based video editing datasets, CoinVE-200K places a stronger emphasis on multi-intent composition, region-aware editing, and complex interactions among diferent editing operations. Besides, to provide a unified evaluation protocol for this challenging setting, we introduce CoinVE-Bench, a dedicated benchmark for compositional-instruction video editing, covering diverse combinations of editing subjects, operation types, and instruction complexities. We further present CoinVE-Edit, a 22B compositional video editing model built upon Wan2.1-T2V-14B and Qwen3VL-8B. CoinVE-Edit disentangles region-aware attention for diferent editing instructions, enabling precise multi-region editing while preserving irrelevant content and maintaining temporal coherence. Extensive experiments on CoinVE-Bench demonstrate that CoinVE-Edit achieves strong performance in instruction following, compositional editing accuracy, visual quality, and temporal consistency, providing a powerful baseline for future research on compositional instruction-guided video editing.

Date: August, 2026 Project Page: https://coinve200k.github.io Dataset: https://huggingface.co/datasets/FireCRT/CoinVE-200K

![](images/77914771121ee849a0d6478650d9357700f3003a7a83c74684cbafb22e5faa16.jpg)

## 1 Introduction

The recent success of instruction-guided image editing models, including FLUX-Kontext(Batifol et al., 2025), Qwen-Image-Edit(Wu et al., 2025a), and Nano Banana(Comanici et al., 2025b), highlights their strong capability in understanding user intent and executing high-quality visual modifications. This success is closely tied to the rapid development of large-scale, diverse, and high-quality datasets for instruction-based image editing(Kuprashevich et al., 2025; Wang et al., 2025; Qian et al., 2025). Meanwhile, instruction-guided video editing has also attracted increasing attention due to its broad applications in content creation, film production, advertising, and interactive visual storytelling. Compared with image editing, video editing requires not only accurate instruction following and spatially precise modifications, but also strong temporal consistency across frames, making the construction of high-quality video editing datasets substantially more challenging.

Existing instruction-guided video editing datasets have made important progress in scaling up data and covering diverse editing categories. However, most of them mainly focus on single editing operations, such as changing a background, removing an object or modifying a local region. While these datasets are useful for

![](images/7ed7dcdac1d076f6464088119b27fa8ac0f4fd37f91d68028d775f64e02ef73c.jpg)  
Remove the beard from the man in the foreground. Remove the yellow excavator parked on the dirt road behind the man. Replace the dry grassy hills and blue sky in the background with a dense green forest.

![](images/1dbdcca634c76a9f15f65b487c606d5be58ab71e2d1abf4f609c81b27b13595f.jpg)  
Replace the dark blue shirt with a yellow collar worn by the man on the left with a red t-shirt. Remove the beige cap worn by the man on the right. Replace the lush green forest setting behind the men with a sunny sandy beach with ocean waves.

![](images/dea4e6cd4ba00ac8ea670afe0a06ab5bd117d27e2b7ff4adfc03824e7e272554.jpg)  
Remove the blue denim apron worn by the woman. Replace the kitchen background with silver oven and shelves with a blurred outdoor garden scene with green foliage.

![](images/758fe35e62835d49e42159e289e6d9b10f4c6b0257d6fe3ba53d7c721f6406ee.jpg)  
Add a large, colorful parrot with blue and yellow feathers perched on the man's left shoulder. Replace the brown plaid suit jacket worn by the man with a bright red leather biker jacket. Replace the green floral wall behind the subject with a solid dark blue wall.

![](images/b1f27138352ccd189e2aa87e3d7534622748592b80a5befa4e4b56f26b0aae03.jpg)

![](images/13a2e82b27f2f7e42b5e660d1f2dad929dacf3a70e37b2313d05489264a4452e.jpg)  
Replace the girl's black suit jacket and white shirt with a yellow summer dress with short sleeves. Replace the piano with a colorful xylophone. Add a small brown teddy bear sitting upright on the wooden bench to the left of the girl. Replace the cozy room background with wooden floors and walls with a grand concert hall stage

![](images/18a52bb90f67a02ea6472e72f717c61318fa1ebcd1731236abe1f2ea6d38b227.jpg)  
Add a tall green potted snake plant in a white ceramic pot at the location of the blurred vase-like object on the left foreground. Replace the bright indoor room background with window and door frame with a dark brick wall texture. Stylize the short-haired man wearing blue clothes in the image into an oil painting style.

![](images/0561c898a1c691fdca7bc80f5f87063a5ad3f8053f18d57cea2df4f7c905da85.jpg)  
Remove the woman with dark hair and sunglasses standing on the left side of the foreground Replace the man's black rectangular sunglasses with a pair of round, gold-rimmed retro sunglasses. Replace the man's tank top with a white t-shirt. Replace the thatched-roof hut in the background with a modern building featuring glass windows. Add a golden retriever dog sitting down to the right of the man in the foreground.

![](images/5ae64040b326cc6d7795dad1be5091a22f805cb7ce0eb58fabad22c1da93ddfd.jpg)  
Replace the clear plastic food processor bowl containing red liquid on the counter with a large white ceramic mixing bowl filled with green salad ingredients. Replace the dark wood kitchen cabinets in the background with white shaker-style kitchen cabinets. Stylize the long-haired woman wearing the blackand-green dress in the image into an animation style.

Figure 1 Demonstration of compositional instruction-guided video editing cases from our proposed CoinVE-200K.

training models to perform isolated edits, they are insuficient for practical editing scenarios, where users often provide multiple editing intentions within a single request. For example, a user may ask a model to replace a person, remove an object, and stylize the background simultaneously. Such compositional editing requires the model to jointly understand multiple instructions, identify their corresponding spatial-temporal editing regions, execute each operation faithfully, and preserve irrelevant content. This setting is significantly more dificult than single-operation editing, as diferent editing intents may interact with each other, leading to instruction interference, unintended region modification, and temporal inconsistency.

We refer to this challenging setting as Compositional Instruction-Guided Video Editing, where each editing request consists of multiple atomic operations that should be executed coherently within the same video. Despite its practical importance, this problem remains under-explored. Existing instruction-guided video editing datasets (He et al., 2025; Bai et al., 2026; Zhang et al., 2026b) usually sufer from three limitations when applied to compositional editing. First, their editing instructions are often short and correspond to only one dominant editing intent, making it dificult for models to learn multi-intent reasoning. Second, the editing targets are usually limited to a single subject or region, while real-world compositional editing may involve humans, objects, and backgrounds simultaneously. Third, existing datasets lack systematic annotations or data construction strategies for combining diferent edit types, such as addition, removal, modification, and global stylization. Consequently, models trained on these datasets often struggle to perform complex multi-operation editing and tend to modify irrelevant regions or ignore part of the given instructions.

To address these limitations, we introduce CoinVE-200K, a large-scale, high-quality dataset for Compositional Instruction-Guided Video Editing. The proposed dataset contains 200K video-editing pairs, each in 1080p resolution with up to 201 frames, covering diverse compositional editing scenarios. Each sample involves 2 to 5 atomic editing operations, encouraging models to learn how to decompose, associate, and execute multiple editing intents within a unified generation process. As shown in Figure 1, the editing instructions span multiple target subjects, including humans, objects, and backgrounds, and cover a broad range of edit types, including addition, removal, modification, and global stylization. Compared with existing instruction-based video editing datasets, CoinVE-200K places stronger emphasis on multi-intent composition, region-aware editing, and complex interactions. The construction of CoinVE-200K is supported by a carefully designed data generation and quality filtering pipeline. Given source videos, we first analyze their visual content and identify editable subjects, including foreground humans, salient objects, and background regions. Then, we generate compositional editing instructions by combining multiple atomic operations under predefined editing taxonomies. These instructions are further used to produce edited videos while maintaining spatia plausibility and temporal coherence. To ensure data quality, we adopt a multi-stage quality filtering strategy that evaluates generated video-editing pairs from several perspectives, including instruction faithfulness, visua quality, temporal consistency, subject preservation, and compositional correctness. This pipeline enables CoinVE-200K to provide high-quality supervision for complex compositional video editing.

Based on CoinVE-200K, we further present CoinVE-Edit, a 22B compositional video editing model built upon Wan2.1-T2V-14B (Wan et al., 2025) and Qwen3-VL-8B-Instruct (Bai et al., 2025a). CoinVE-Edit is designed to efectively handle multi-instruction editing requests by disentangling region-aware attention for diferent editing instructions. Specifically, the model first leverages the multimodal understanding capability of Qwen3- VL to interpret both the input video and the compositional editing instruction. Then, the video generation backbone based on Wan2.1 performs instruction-guided video transformation. To mitigate interference among diferent editing operations, CoinVE-Edit introduces a region-aware attention disentanglement mechanism, where diferent editing instructions are associated with their corresponding spatial-temporal regions and controlled independently during generation. This design enables precise multi-region editing while preserving irrelevant content and maintaining temporal coherence.

In addition to the dataset and model, we propose CoinVE-Bench, a dedicated benchmark for compositional instruction-guided video editing. Existing video editing benchmarks mainly evaluate single-operation editing and are insuficient for measuring compositional editing ability. CoinVE-Bench is designed to cover diverse combinations of editing subjects, operation types, and instruction complexities. It evaluates model performance along multiple dimensions, including instruction following, compositional editing accuracy, region-level editing precision, irrelevant content preservation, visual quality, and temporal consistency. By providing a unified evaluation protocol, CoinVE-Bench enables systematic comparison of diferent models under complex multiinstruction editing scenarios.

Extensive experiments demonstrate that CoinVE-Edit achieves strong performance on CoinVE-Bench, especially in challenging cases involving multiple editing operations and multiple target regions. Compared with existing instruction-guided video editing models, CoinVE-Edit shows better compositional instruction following, more accurate region-aware editing, improved non-target preservation, and stronger temporal stability. We believe that CoinVE-200K, CoinVE-Bench, and CoinVE-Edit together provide a comprehensive foundation for advancing research on compositional instruction-guided video editing. In summary, our contributions are:

• We introduce CoinVE-200K, a large-scale and high-quality dataset for compositional instruction-guided video editing. It contains 200K video-editing pairs, where each sample consists of 2 to 5 atomic editing operations. The editing taxonomy covers multiple target subjects, including humans, objects, and backgrounds, and diverse edit types, including addition, removal, modification, and stylization.

• We present CoinVE-Edit, a 22B compositional video editing model that disentangles region-aware attention for diferent editing instructions, enabling precise multi-region editing while preserving irrelevant content.

• We establish CoinVE-Bench, a dedicated benchmark for compositional instructional video editing, evaluating models across instruction following, compositional editing accuracy, region-level precision, non-target preservation, visual quality, and temporal consistency.

## 2 Related Work

## 2.1 Video Understanding

Video understanding aims to perceive, describe, and reason about dynamic visual content across both spatial and temporal dimensions. Early studies mainly focus on task-specific video representation learning, such as action recognition (Tran et al., 2015; Wang et al., 2016), temporal localization (Long et al., 2019), video captioning (Zhou et al., 2018), and video question answering (Chen et al., 2020). With the rapid development of large language models (LLMs) and multimodal large language models (MLLMs), recent works have shifted toward general-purpose video understanding through multimodal instruction tuning. Representative models, such as Video-LLaMA (Zhang et al., 2023a), VideoChat (Li et al., 2023), Video-ChatGPT (Maaz et al., 2024), Video-LLaVA (Lin et al., 2024), and QwenVL series (Bai et al., 2023, 2025b,a), extend image-language models to video inputs by incorporating temporal visual tokens, video encoders, or video-specific instruction data. Video-LLaMA (Zhang et al., 2023a) connects video and audio encoders with LLMs for audio-visual instruction following, while VideoChat (Li et al., 2023) builds an interactive video-language dialogue system by aligning video foundation models with LLMs. Video-ChatGPT (Maaz et al., 2024) further improves detailed video understanding through video feature projection and instruction tuning. Video-LLaVA (Lin et al., 2024) learns unified visual representations for both images and videos before language projection, improving cross-modal alignment. More recently, the QwenVL series (Bai et al., 2023, 2025b,a) demonstrates strong multimodal perception, grounding, OCR, and long-context visual reasoning capabilities, making it suitable for extracting high-level semantic conditions from complex video inputs.

For instruction-guided video editing, such video understanding capability is essential for identifying editable subjects, parsing user intentions, and grounding editing operations to corresponding spatial-temporal regions. This becomes more challenging in compositional instruction-guided video editing, where multiple editing intents must be jointly decomposed, grounded, and executed while preserving irrelevant content. Our work follows this direction by leveraging MLLM-based semantic understanding to obtain structured editing conditions and further injecting them into a controllable video editing framework.

## 2.2 Video Generation

Video generation has achieved significant progress with the development of difusion models and large-scale generative foundation models. Early video difusion methods extend image difusion models to the temporal domain by introducing temporal layers, motion modules, or 3D U-Net (Rombach et al., 2022; Singer et al., 2023; Ho et al., 2022; Long et al., 2024; Zhang et al., 2025; Chen et al., 2025a,b). Stable Video Difusion (Blattmann et al., 2023) further demonstrates the efectiveness of large-scale image-to-video pretraining and becomes an important foundation for high-quality video synthesis. Subsequent works explore various architectures and training strategies to improve motion realism, temporal consistency, and video quality (Chen et al., 2024), including latent video difusion models and cascaded video difusion models.

Recently, Difusion Transformer (DiT)-based architectures (Zheng et al., 2024; Ma et al., 2025; Yang et al., 2025; Kong et al., 2024) have emerged as a dominant paradigm for scalable video generation, shifting video difusion models from U-Net-style temporal extensions toward token-based spatial-temporal modeling. Latte (Ma et al., 2025) first validates this direction by representing videos as latent spatial-temporal tokens and modeling them with transformer blocks. Subsequent large-scale systems further strengthen this paradigm: CogVideoX (Yang et al., 2025) combines a 3D causal VAE with an expert transformer to improve video compression and text-video alignment, while HunyuanVideo (Kong et al., 2024) develops a system-level framework with strong text encoders, scalable DiT backbones, and progressive training strategies for high-quality video generation. Meanwhile, LTX-Video (HaCohen et al., 2025, 2026) pushes generation eficiency through highly compressed video latents, and Wan (Wan et al., 2025) provides a powerful open video foundation model with strong text-to-video and image-to-video capabilities. These models provide strong generative priors for video editing, but editing requires preserving the input video’s identity, layout, and motion while modifying only instructionrelevant regions. Therefore, we build upon a strong video generation backbone and introduce region-aware attention disentanglement to adapt it to compositional instruction-guided video editing.

## 2.3 Instruction-Guided Video Editing

Instruction-guided video editing aims to modify a given video according to natural language instructions. Compared with traditional video editing methods that rely on masks, scribbles, keyframes, or manually specified control signals, instruction-based editing provides a more intuitive and user-friendly interface. Inspired by the success of instruction-guided image editing models such as InstructPix2Pix (Brooks et al., 2023), MagicBrush (Zhang et al., 2023b), FLUX-Kontext (Batifol et al., 2025), and Qwen-Image-Edit (Wu et al., 2025a), recent studies have begun to explore natural language-driven video editing. Existing methods usually adopt pretrained video difusion models as generative backbones and inject editing conditions through additional encoders (DecartAI Team, 2025), cross-attention modules (Geyer et al., 2024), adapters (Jiang et al., 2025), or feature modulation mechanisms.

A key challenge in instruction-guided video editing is how to preserve the spatial-temporal structure of the input video while faithfully applying modifications. Early methods mainly adapt image difusion models to videos and enforce temporal consistency through attention control or feature propagation. For example, TokenFlow (Geyer et al., 2024) propagates difusion features across frames to obtain temporally consistent edits. With the development of stronger video difusion and DiT backbones, recent methods increasingly rely on explicit condition injection modules, such as additional encoders, adapters, or multimodal instruction representations. VACE (Jiang et al., 2025), for instance, introduces adapter-style conditioning to incorporate diverse video editing signals into the generation process, while recent MLLM-based methods (Lin et al., 2026; Zhang et al., 2026a; Chen et al., 2026; Pan et al., 2026) extract high-level instruction semantics and inject them into video generative backbones. These works form a common pipeline of instruction-guided video editing: semantic understanding of the input video and editing instruction, followed by conditional video generation with a pretrained difusion or DiT backbone.

Recent datasets and models, such as InsViE (Wu et al., 2025b), Senorita (Zi et al., 2025), Ditto (Bai et al., 2026), and OpenVE (He et al., 2025), have advanced data-driven instruction-based video editing with largescale video-edit pairs. However, they mainly focus on single editing operations, while real-world users often provide compositional instructions involving multiple intents, subjects, and edit types within the same video. Such scenarios require models to decompose instructions, ground each operation to the corresponding spatialtemporal region, coordinate multiple edits without interference, and preserve irrelevant content. To address this challenge, we introduce CoinVE-200K, a large-scale dataset for compositional instruction-guided video editing, together with CoinVE-Bench for systematic evaluation. We further develop CoinVE-Edit, combining MLLM-based instruction understanding with DiT-based video generation and disentangling region-aware attention for precise multi-region editing.

## 3 CoinVE-200K

Here, we introduce CoinVE-200K, a large-scale high-quality video dataset designed for compositional instruction-guided video editing. Figure 2 depicts the whole data construction pipeline.

## 3.1 Video Pre-Processing

The source video set is collected from the high-quality open-source video dataset, i.e., OpenVid-HD (Nan et al., 2025). Given the whole set, we first filter the video clips based on the frame number and resolution. The retained videos are required to contain at least 81 frames and have a shorter side of at least 1080 pixels. Then, we utilize aesthetic scores (Christoph Schuhmann, 2024) and optical flow (Teed and Deng, 2020) to select videos characterized by high aesthetic quality and appropriate motion magnitude. Finally, we attain 420K videos for the subsequent video pair synthesis.

## 3.2 Taxonomy-based Compositional Instruction Generation

To facilitate the instruction generation, the taxonomy of the visual content in the source video should be first parsed. As shown in the left part of Figure 2, we feed the source video into Qwen3.6-27B (Qwen Team, 2026) for video understanding. For each retained video, Qwen3.6-27B is prompted to summarize its editable visual content from three complementary aspects: subjects, objects, and background. This taxonomy provides a structured description of the source video, including the main actors, salient manipulable entities, and scene context, which serves as the basis for subsequent compositional instruction generation.

Stage 3: Secondary Quality Filtering  
![](images/27965de7167eff30b1d488d0bfd6fc91b0690a4813971a78ee9a43d352fb30e7.jpg)  
Figure 2 Data construction pipeline of CoinVE-200K.

Based on the parsed taxonomy, we construct compositional instructions by sampling and combining multiple atomic editing operations. Concretely, we maintain a set of operation templates covering common edit types, such as replace, add, remove, and stylization. Qwen3 is then used to instantiate these templates with concrete visual concepts extracted from the source video, yielding semantically grounded atomic instructions. Multiple atomic operations are further composed into a single multi-instruction editing request, so that one sample may simultaneously involve subject appearance modification, object replacement, scene transformation, and attribute addition or deletion. To improve the validity and diversity of generated instructions, we additionally apply a data checking cycle to verify that each atomic edit is compatible with the source content and that the combined instruction remains coherent and executable. Meanwhile, the balance across all types of atomic editing operations is also achieved.

## 3.3 Mask-based Video Synthesis

After obtaining the compositional instructions, we synthesize the edited video in a mask-guided manner, as illustrated in the middle part of Figure 2. Given a source video and its corresponding multi-instruction prompt, we first sample a representative key frame from the video. Since compositional editing usually targets multiple entities or regions, we use Qwen3.6-27B (Qwen Team, 2026) to extract the edited keywords from the instruction set, i.e., the concrete visual concepts that need to be modified, inserted, or removed. These keywords provide explicit textual anchors for localizing the target regions in the key frame.

We then employ SAM3 (Carion et al., 2026) to segment the relevant regions on the key frame according to the extracted editing keywords. The resulting masks identify the spatial locations associated with each atomic edit, such as a piece of clothing, a wall painting, an accessory, or the background region. To extend these edits from a single frame to the entire video, we further use SAM2 (Nikhila et al., 2025) to propagate the key-frame masks across adjacent frames, producing temporally aligned edited-region masks for the full video. This step establishes explicit spatial-temporal correspondences for the regions afected by the various instructions.

With the localized masks, we first edit the key frame using strong image editing models, including HunyuanImage-3.0 (Cao et al., 2025) and Nano Banana (Comanici et al., 2025a), conditioned on the multi-instruction prompt. This produces an edited key frame that reflects the desired compositional modifications. Afterwards, we combine all propagated region masks into one mask, and further feed it with the edited frame and the original source video into our deliberately fine-tuned Wan2.2-Animate-14B (Cheng et al., 2025) or VACE (Jiang et al., 2025) model to synthesize the final edited video. Compared with directly editing video through SOTA editing models, this mask-based pipeline provides better control over the edited regions, improves faithfulness to each atomic instruction, and helps preserve temporal consistency across frames.

![](images/b967fcf54b0a7a5154bbe894060fded6f3747859e433ead6f855ace5fab37e41.jpg)  
(a) Frame Number Distribution

![](images/e62465a8f0f704b2c74ef48f568cdb271a9a2f11b191326c2816facd68de9ac5.jpg)  
(b) Atomic Editing Type Distribution

![](images/06b01508ea9927117e2c447e7c0fecda70e5f8788a0ddc8f1df4a735f6001735.jpg)  
(c) Instruction Number per Video Distribution

![](images/b4d62c9102d53ddd99e94f7015a5280a958c86cbad02b279559cf5e6b3fc6665.jpg)  
(d) Word Cloud of Instructions

![](images/4570a60e81169c58b15d21a6286cb75c6d17d5f03367d862f9363f0d61b60090.jpg)  
(e) Video Editing Quality Score Comparisons

![](images/477da706a0aaa4c63b1e73ae53c8d57b046c70d71c9acbcb07813e115de8e962.jpg)  
(f) Comparisons with Other Datasets  
Figure 3 Data statistics of CoinVE-200K.

## 3.4 Secondary Quality Filtering

Although the mask-based synthesis pipeline produces candidate edited videos with diverse compositional modifications, the generated results may still sufer from incomplete instruction execution, temporal inconsistency, or unrealistic visual artifacts. Therefore, as shown in the right part of Figure 2, we perform a secondary quality filtering stage to retain only high-quality samples for the final dataset.

Specifically, we feed the multi-instructions, source videos, and edited videos into Gemini 2.5 Pro (Comanic et al., 2025a) for automatic quality assessment. The evaluation is conducted from four perspectives. First, we measure instruction editing accuracy at the level of each atomic instruction, checking whether every individual operation in the compositional prompt has been successfully executed. This per-instruction verification is particularly important. The sample is only considered valid when all sub-edits are correct. Second, we evaluate the overall quality of the edited video from three aspects: semantic consistency, temporal consistency, and physical plausibility. This ensures that the edited result remains coherent with the source content except for the intended changes, stable across frames without flickering or drifting, and physically plausible in geometry, interactions, and scene composition.

We adopt a strict filtering strategy and only keep samples that pass all evaluation items. In practice, this means that each atomic instruction must be marked as successfully executed, while semantic consistency, temporal consistency, and physical plausibility must all receive positive judgments. Such a conservative selection criterion efectively removes low-quality generations and substantially improves the reliability of the final compositional video editing dataset.

## 3.5 Data Statistics

After the overall data generation pipeline and data quality filtering, we obtain CoinVE-200K, a high-quality compositional-instruction video editing dataset with 200K samples. As shown in Figure 3, the retained videos contain 81–201 frames with a resolution of 1080p, covering diverse temporal lengths and visual contents. CoinVE-200K includes six atomic editing types with a relatively balanced distribution: local addition (24.1%), background replacement (23.8%), object replacement (22.8%), local removal (18.5%), background stylization (6.4%), object stylization (4.5%). Each video is paired with multiple atomic instructions, where 57.7%, 31.7%, 8.5%, and 2.1% of videos contain 2, 3, 4, and 5 instructions, respectively, resulting in an average of 2.55 instructions per video. The word cloud further shows that our instructions cover a wide range of objects, attributes, colors, and editing actions, such as “person”, “background”, “add”, and “remove”, indicating rich linguistic and visual diversity. Compared with existing single-instruction video editing datasets, CoinVE 200K provides richer compositional supervision and achieves a higher average editing quality score of 4.85, demonstrating both its diversity and reliability for training compositional-instruction video editing models.

![](images/60bb056577e906c837aaa3182c1491a8000edb1f43af08df55f69afc56a8967d.jpg)  
Figure 4 An overview of the proposed CoinVE-Edit framework.

## 4 CoinVE-Edit

In this section, we introduce CoinVE-Edit, the newly-minted compositional instruction-guided video editing model. As shown in Figure 4, given a source video and multiple editing instructions, the Qwen-based MLLM first jointly processes the visual content and textual instructions to extract high-level semantic representations for compositional editing. Next, a feature Connector then transforms the hidden features from the MLLM into editing-aware tokens, including instruction-related learnable tokens and visual tokens. The former serve as semantic conditions for video generation. Meanwhile, the source video is encoded into latent representations by a VAE encoder and fed into the DiT backbone for video synthesis. To enable precise multi-intent editing, the mask-based conditioning module predicts spatial-temporal editing masks, and generates routing signals to control how diferent instruction tokens interact with visual tokens inside the DiT blocks. Finally, the edited latent representations are decoded by the VAE decoder to produce the edited video. Through this pipeline, CoinVE-Edit can jointly understand multiple editing intents, associate them with corresponding video regions, and perform compositional video editing while preserving irrelevant content.

## 4.1 Architecture

As depicted in Figure 4, CoinVE-Edit is composed of four tightly coupled modules: (i) a Qwen3-VL-based multimodal large language model (MLLM) equipped with learnable video queries, (ii) a Connector that projects the learnable-query hidden states into DiT-compatible editing-aware tokens, (iii) a conditioning Mask Predictor that predicts a spatial-temporal editing mask from the MLLM visual tokens conditioned on the visual context, and a GateNet (a lightweight instruction-level gating network) that predicts a local/global editing gate, and (iv) a DiT-based video editing backbone built upon Wan2.1-T2V-14B (Wan et al., 2025), in which each cross-attention block is wrapped by a Q-Blending Cross-Attention module.

## 4.1.1 Mask-based Conditioning

MLLM with Learnable Video Queries. Given a source video V and a compositional instruction $\mathcal { T } = \{ I _ { 1 } , \ldots , I _ { N } \}$ of N atomic edits $\left( N \ge 2 \right)$ , we adopt Qwen3-VL-8B-Instruct (Bai et al., 2025a) as our base MLLM. For the i-th instruction $I _ { i } ,$ the input is formed as $[ V , I _ { i } , \langle \mathsf { q u e r y } \rangle _ { 1 : L _ { q } } ]$ appended $L _ { q }$ learnable query tokens. We read two groups of last-layer hidden states:

• Visual tokens $\mathbf { H } _ { v i s } ^ { i }$ at all video token positions, forming the size $N _ { v } = T _ { v } H _ { v } W _ { v } ;$

• Query hidden states $\mathbf { H } _ { q } ^ { i }$ at the ⟨query⟩ positions, encoding the instruction jointly with the visual context.

A lightweight MLP-based Connector then projects $\mathbf { H } _ { q } ^ { i }$ into the DiT conditioning dimension $C ^ { i } = \phi _ { \mathrm { c o n } } ( \mathbf { H } _ { q } ^ { i } )$ After N feedforward passes of MLLM, we gather the context tensor of all N instructions as $\mathbf { C } = [ C ^ { 1 } , . . . , C ^ { \hat { N } } ]$ The whole context tensor C is shared by three downstream consumers, i.e., the video DiT cross-attention, the Mask Predictor and the GateNet.

Mask-based Conditions. To prevent interference among multiple edits and to distinguish local from global/style edits, we attach two lightweight heads on top of the shared MLLM outputs: Mask Predictor predicts where the current instruction should take efect, and GateNet predicts how spatially localized that efect should be. Their outputs jointly parameterize Q-Blending (§4.1.2) without altering the context C.

Mask Predictor is a prompt-conditioned mask predictor that grounds the instruction to a spatial-temporal region on the video patch grid. For the i-th instruction, it takes the visual tokens $\mathbf { H } _ { \mathrm { v i s } } ^ { i }$ (augmented with 3D grid positional embeddings) and the context tensor $C ^ { i }$ as key/value memory, and lets K learnable mask queries interact with them through two transformer blocks (self-attention on the queries, bidirectional cross-attention with mask queries). The mask token is then decoded against the visual segment to yield a patch-grid mask $M ^ { i } \in \mathbb { R } ^ { T _ { v } \times \hat { H } _ { v } \times W _ { v } }$ (probabilities after sigmoid $\sigma ( \cdot ) )$ , where 1 indicates the editing region. The context tensor $C ^ { i }$ conditions the prediction on the current instruction, allowing it to switch its target region when the instruction changes.

GateNet complements Mask Predictor by predicting whether the current edit is local (e.g., object replacement/removal, region-specific background change) or global/style (e.g., stylization, color grading, weather change), so that the mask constraint is enforced only when meaningful. The architecture is a two-layer MLP on the token-wise mean-pooled context $C ^ { i }$ , producing a scalar gate for i-th instruction as follows:

$$
g ^ { i } = \sigma \left( \mathrm { M L P } \left( \mathrm { M e a n } \left( C ^ { i } \right) \right) \right) ,\tag{1}
$$

where $g ^ { i }$ ≈1 indicates a local edit that should respect the mask, whereas $g ^ { i } { \approx } 0$ indicates a global edit that should bypass the region constraint. Thus, $( M ^ { i } , g ^ { i } )$ answer $^ { 6 6 } \mathrm { w }$ here to edit” and “how strongly the where matters”, the two quantities consumed by Q-Blending.

## 4.1.2 DiT-based Video Editing

We build the video generation backbone upon Wan2.1-T2V-14B (Wan et al., 2025). The source video V is first encoded into a latent $\mathbf { x } _ { \mathrm { 0 } }$ by the pretrained 3D VAE, and a noised latent $\mathbf { x } _ { t }$ is drawn from the flow-matching forward process. To inject the source video as a strong structural prior, we concatenate $\mathbf { x } _ { \mathrm { 0 } }$ with $\mathbf { x } _ { t }$ along the channel dimension before the DiT, so that layout, identity, and motion are preserved. Each DiT block contains a latent self-attention and a cross-attention that consumes the context C for all instructions. We wrap the latter with a Q-Blending Attention module that softly rescales its output residual per query token using $\mathbf { M } = \{ M ^ { i } \} _ { i = 1 } ^ { N }$ and $g = \{ g ^ { i } \} _ { i = 1 } ^ { N }$ The denoised latent is finally decoded by the VAE to produce the edited video.

Q-Blending Cross-Attention. Rather than applying hard mask biases to the attention logits, Q-Blending Cross-Attention keeps the original DiT cross-attention unchanged and modulates the attention output in a query-wise manner. For context tensor $C ^ { i }$ of each instruction, the standard cross-attention produces an instruction-specific output, which is then weighted by a blending map predicted from the mask $M ^ { i }$ and the gate weight $g ^ { i } .$ . These weighted outputs are summed over all N instructions to form the final contextual output. This design preserves the pretrained attention distribution, avoids the instability caused by hard masking, and enables spatially adaptive composition of multiple editing instructions.

Mask-based Q-Blending Weight. Mask Predictor predicts $M ^ { i }$ on the MLLM patch grid $\left( T _ { v } , H _ { v } , W _ { v } \right)$ , whereas the DiT operates on a latent grid $( T _ { l } , H _ { l } , W _ { l } )$ . We trilinearly resample $M ^ { i }$ of i-th instruction to the latent grid and flatten it to a per-query mask. Combined with the local/global gate $g ^ { i } \in [ 0 , 1 ]$ , we define the Q-Blending weight for each instruction as:

$$
w _ { q } ^ { i } \ = \ 1 - \beta g ^ { i } ( 1 - M ^ { i } ) , \quad w _ { q } ^ { i } \in \mathbb { R } ^ { T _ { l } \times H _ { l } \times W _ { l } }\tag{2}
$$

where $\beta \in [ 0 , 1 ]$ is a blending strength. Eq. (2) behaves as intended in three regimes: on the editing region $( M ^ { i } \to 1 )$ it is a no-op so the instruction takes full efect, on non-editing regions of a local edit $( M ^ { i } \to 0 , g ^ { i } \to 1 )$ it softly attenuates the residual to $1 - \beta \ ( \mathrm { i . e }$ ., 70% under $\beta { = } 0 . 3 )$ , suppressing unintended modifications while avoiding hard zeros that would destabilize the DiT. For global edits $( g ^ { i } \to 0 )$ it degenerates to $w _ { q } ^ { i } \equiv 1$ , letting the instruction afect whole frame.

Injection into DiT Cross-Attention. Q-Blending is realized as a wrapper that reuses all pretrained projections and normalizations of the DiT original cross-attention. Given the block hidden states and context $\bar { \mathbf { C } } = \bar { [ ( C ^ { 1 } , . . . , C ^ { N } ] }$ of all N instructions, we compute $\mathbf { o } ^ { i } = \mathrm { A t t n } ( \mathbf { q } , \mathbf { k } ^ { i } , \mathbf { v } ^ { i } )$ as usual for each instruction (DiT hidden states are the query $\mathbf { q } , C ^ { i }$ is treated as the key $\mathbf { k } ^ { i }$ and value $\mathbf { v } ^ { i } )$ , then apply the obtained Q-Blending weight $w _ { q } ^ { i }$ and sum over all N instructions as:

$$
\mathbf { o } _ { \mathrm { c t x } } = \frac { \sum _ { i = 1 } ^ { N } \left( w _ { q } ^ { i } \odot \mathbf { o } ^ { i } \right) } { \sum _ { i = 1 } ^ { N } w _ { q } ^ { i } } ,\tag{3}
$$

where $\odot$ is element-wise multiplication, and $\mathbf { o } _ { \mathrm { c t x } }$ is fed into the next transformer block. Note that $w _ { q } ^ { i }$ is dynamically predicted by the Mask Predictor and GateNet for each instruction. Therefore, the Q-Blending Cross-Attention introduces no new trainable parameters into the DiT backbone.

## 4.2 Training Objectives

Besides the standard flow-matching objective ${ \mathcal { L } } _ { \mathrm { d i t } }$ for optimizing the video difusion transformer, we introduce two auxiliary objectives to supervise the region-aware control modules, i.e., the Mask Predictor and GateNet. For the Mask Predictor, we adopt a segmentation loss composed of a binary cross-entropy loss and a Dice (Milletari et al., 2016) loss:

$$
\mathcal { L } _ { \mathrm { s e g } } = \mathcal { L } _ { \mathrm { b c e } } + \mathcal { L } _ { \mathrm { d i c e } } .\tag{4}
$$

This objective encourages the predicted masks to accurately localize the spatial-temporal regions corresponding to each editing instruction. For GateNet, we use a binary classification loss $\mathcal { L } _ { \mathrm { g a t e } }$ to supervise the routing decision of whether an instruction belongs to global stylization. During training, the gradients from $\mathcal { L } _ { \mathrm { s e g } }$ and $\mathcal { L } _ { \mathrm { g a t e } }$ are stopped before being propagated to the MLLM and DiT backbones, so that these auxiliary losses only update the corresponding control modules. The overall training objective is defined as:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { d i t } } { \mathcal { L } } _ { \mathrm { d i t } } + \lambda _ { \mathrm { s e g } } { \mathcal { L } } _ { \mathrm { s e g } } + \lambda _ { \mathrm { g a t e } } { \mathcal { L } } _ { \mathrm { g a t e } } ,\tag{5}
$$

Table 1 Evaluation matrix of CoinVE-Bench. The proposed metrics are grouped by evaluator type into MLLM-based checklist evaluation and specialized evaluator-based evaluation with task-specific models, covering four evaluation dimensions and eleven fine-grained metrics.
<table><tr><td>Dimension</td><td>Metric (Abbr.)</td><td>Range</td><td>Evaluator</td><td>Description</td></tr><tr><td colspan="5">MLLM-Based (Gemini 3.6 Flash)</td></tr><tr><td>Editing</td><td>Semantic Accuracy (SA)</td><td>[0,100]</td><td>MLLM + checklist</td><td>Correct execution of the intended edit semantics</td></tr><tr><td>Accuracy</td><td>Scope Accuracy (SPA)</td><td>[0,100]</td><td>MLLM + checklist</td><td>Correct edit localization without leakage or interference</td></tr><tr><td>(per Instruction)</td><td>Editing Persistence (EP)</td><td>[0,100]</td><td>MLLM + checklist</td><td>Temporal consistency of the edit throughout the video</td></tr><tr><td>Physical</td><td>Appearance Naturalness (AN)</td><td>[0,100]</td><td>MLLM + checklist</td><td>Natural blending of lighting, shadows, textures, and style</td></tr><tr><td>Naturalness</td><td>Scale Consistency (SC)</td><td>[0,100]</td><td>MLLM + checklist</td><td>Plausibility of the edited object&#x27;s scale and perspective</td></tr><tr><td>Semantic</td><td>Motion Naturalness (MN)</td><td>[0,100]</td><td>MLLM + checklist</td><td>Plausibility of motion and physical interactions</td></tr><tr><td>Preservation</td><td>Content Preservation (CP)</td><td>[0,100]</td><td>MLLM + checklist</td><td>Preservation of non-edited regions, objects, and structures</td></tr><tr><td colspan="5">Specialized Evaluator-Based</td></tr><tr><td>Video</td><td>Aesthetic Quality (AQ)</td><td>[1,10]</td><td>Aesthetic Predictor v2.5</td><td>Overall frame-level aesthetic appeal</td></tr><tr><td></td><td>Technical Quality (TQ)</td><td>(1, 100)</td><td>DOVER++</td><td>Technical blur, noise, compression, flicker, and jitter</td></tr><tr><td>Quality</td><td>Comprehensive Quality (CQ)</td><td>[1, 5]</td><td>VisualQuality-R1</td><td>Overall frame-level visual quality</td></tr><tr><td></td><td>Temporal Stability (TS)</td><td>[0, 1]</td><td>Optical-flow fields</td><td>Temporal motion smoothness</td></tr></table>

where $\lambda _ { \mathrm { d i t } } , \lambda _ { \mathrm { s e g } }$ and $\lambda _ { \mathrm { g a t e } }$ are the loss weight coeficients (1.0 for each as default).

## 4.3 Training Strategy

To support compositional-instruction video editing, we design a progressive three-stage training paradigm that organically integrates the standalone DiT and MLLM into a unified video editing model, enabling it to perform both single-instruction and compositional-instruction edits.

Feature Alignment Pre-Training. To bridge the semantic gap between Qwen3-VL and Wan video DiT backbone, we first freeze all of the DiT weights, and only learn the learnable query tokens, connector and LoRA of MLLM. The task is restricted to only image editing, which scales well for large-scale model training. After this stage, the learnable tokens are adapted to the formats that can be interpreted by the cross-attention block in video DiT.

Single-Instruction Training. In the second training stage, we aim to endow the model with single-instruction video editing capability. To this end, we further unfreeze the DiT backbone and jointly optimize it with the MLLM-conditioned editing modules. Due to the limited availability of high-quality video editing data, we adopt a mixed training strategy that combines both image- and video-based single-instruction editing data. Such hybrid training enables the model to generalize well to single-instruction editing on videos, even under limited video supervision. Moreover, the resulting model weights serve as a strong prior for the next stage, facilitating the extension from single-instruction video editing to compositional-instruction video editing.

Compositional-Instruction Fine-Tuning. For the final stage, we train the model for compositional-instruction video editing, where the model is required to handle multiple editing intents. This stage involves two key capabilities: predicting the spatial regions associated with each instruction, and determining whether an instruction should trigger a global style transfer. Built upon the proposed CoinVE-200K dataset and initialized from the weights obtained in the second stage, we first freeze all model parameters and train the Mask Predictor and GateNet ofline. This isolated training strategy stabilizes the learning of the auxiliary modules without disturbing the pretrained MLLM-DiT editing backbone. After ofline training, the Mask Predictor achieves an IoU of 0.71, while GateNet reaches a binary classification accuracy of 0.95. Finally, we unfreeze the full model and conduct end-to-end fine-tuning, allowing the editing backbone, Mask Predictor, and GateNet to be jointly adapted for compositional-instruction video editing.

## 5 CoinVE-Bench

## 5.1 Benchmark Construction

To establish a unified evaluation protocol for compositional instruction-guided video editing, we construct a dedicated benchmark with 361 high-quality source videos. The videos are collected from open-source repositories, deduplicated, and kept strictly disjoint from our training data. To ensure visual quality and editing feasibility, we retain clips with a resolution of at least 1080p and a duration of 3–10 seconds, and remove those with severe motion blur, visible watermarks or subtitles, abrupt shot transitions, heavy occlusion, or ambiguous visual content. The resulting videos cover diverse objects, scenes, motions, and camera viewpoints. The benchmark is built around six atomic editing operations: add object, remove object, replace object, replace background, local style object, and local style background. Unlike benchmarks that focus on isolated edits, our benchmark emphasizes compositional instructions, where each instruction combines multiple atomic operations. To reduce evaluation bias, we balance the occurrence frequency of diferent atomic operations, the complexity of instructions with varying numbers of operations, and the diversity of operation combinations. All instructions are manually reviewed to remove ambiguous, infeasible, or visually unverifiable cases.

Table 2 Single-instruction video editing comparison on OpenVE-Bench evaluated by Gemini 2.5 Pro.
<table><tr><td>Model</td><td>Overall</td><td>Global Style</td><td>Background Change</td><td>Local Change</td><td>Local Remove</td><td>Local Add</td><td>Subtitle Edit</td></tr><tr><td>Runway</td><td>3.51</td><td>3.72</td><td>2.62</td><td>4.18</td><td>4.16</td><td>2.78</td><td>3.62</td></tr><tr><td>VACE</td><td>1.55</td><td>1.49</td><td>1.55</td><td>2.07</td><td>1.46</td><td>1.26</td><td>1.48</td></tr><tr><td>Ditto</td><td>2.25</td><td>4.01</td><td>1.68</td><td>2.03</td><td>1.53</td><td>1.41</td><td>2.81</td></tr><tr><td>OpenVE-Edit</td><td>2.57</td><td>3.16</td><td>2.36</td><td>2.98</td><td>1.85</td><td>2.15</td><td>2.91</td></tr><tr><td>VINO</td><td>2.91</td><td>4.05</td><td>1.67</td><td>3.12</td><td>3.32</td><td>2.66</td><td>2.64</td></tr><tr><td>OmniWeaving</td><td>3.01</td><td>3.94</td><td>2.03</td><td>3.49</td><td>2.93</td><td>2.06</td><td>3.58</td></tr><tr><td>KiwiEdit</td><td>3.17</td><td>3.60</td><td>2.63</td><td>3.87</td><td>3.31</td><td>2.83</td><td>2.75</td></tr><tr><td>SAMA</td><td>3.33</td><td>3.87</td><td>2.64</td><td>3.91</td><td>3.31</td><td>2.63</td><td>3.63</td></tr><tr><td>CoinVE-Edit</td><td>3.41</td><td>3.61</td><td>3.11</td><td>3.84</td><td>3.81</td><td>2.53</td><td>3.53</td></tr></table>

## 5.2 Evaluation Dimensions

As shown in Table 1, our evaluation matrix is structured into two complementary groups: MLLM-based checklist evaluation and specialized evaluator-based evaluation. The MLLM-based group focuses on the compositional correctness of instruction-guided video editing and consists of three dimensions: Editing Accuracy (Semantic Accuracy, Scope Accuracy, and Editing Persistence), Physical Naturalness (Appearance Naturalness, Scale Consistency, and Motion Naturalness) and Semantic Preservation (Content Preservation). The score of these dimensions is defined as the percentage of questions answered correctly. The specialized evaluator-based group complements the checklist evaluation by measuring perceptual video fidelity through the dimension of Video Quality (Aesthetic Quality, Technical Quality, Comprehensive Quality, and Temporal Stability). Each dimension is evaluated using a dedicated model, with the score scale determined by the corresponding evaluation model. Overall, these two groups cover four dimensions and eleven fine-grained metrics, providing a unified framework for evaluating both the compositional instruction-following correctness and visual fidelity of edited videos.

## 6 Experiments

## 6.1 Implementation Details

Training Data. Our training data consists of three parts. In stage of feature alignment training, we collect opensource image editing datasets, including GPT-Image-Edit-1.5M (Wang et al., 2025), NHR-Edit (Kuprashevich et al., 2025), and Pico-Banana-400K (Qian et al., 2025), to adapt the MLLM feature space to the latent space of video DiT. GPT-Image-Edit-1.5M contains over 1.5M source image, editing instruction, and edited image triplets constructed with advanced image generation models. NHR-Edit is a high-quality instruction-based image editing dataset mined through an automated pipeline. Pico-Banana-400K is a 400K-scale text-guided image editing dataset built from real images with a fine-grained editing taxonomy. Since the quality of image editing data varies significantly, we filter these image editing pairs using EditScore (Luo et al., 2026) and keep only samples with scores higher than 8.0, resulting in about 2M high-quality image editing samples. Then, in single-instructional video editing training, we collect open-source video editing data from OpenVE-3M (He et al., 2025), Ditto-1M (Bai et al., 2026), and ReCo (Zhang et al., 2026b). OpenVE-3M is a large-scale instruction-following video editing dataset with about 3M video-edit pairs, covering both spatially-aligned

Replace the stone bench with a rustic wooden bench made of weathered planks, ensuring it maintains the same position and pose within the video scene.

Replace the background with a dynamic cozy lounge scene where the fireplace flickers with dancing flames, shadows gently move on the walls, and a soft glow illuminates the room.

Remove the light blue cowboy hat having a wide, slightly curved brim and a smooth, rounded crown from the entire video.

Overlay an pair of sunglasses on the man's shirt collar near the top button. The sunglasses must be perfectly tracked to the shirt as the camera moves.

![](images/24333f9bcc4ef39118742d67694b676075df788d1ad9b361185a83f12d3b0459.jpg)  
Figure 5 Quality comparison of single-instruction video editing on OpenVE-Bench.

and non-spatially-aligned edits across multiple editing categories. Ditto-1M provides about 1M high-fidelity synthetic video editing triplets for scaling instruction-based video editing. ReCo is a region-constrained instructional video editing dataset with over 500K instruction-video pairs, emphasizing localized editing and preservation of non-target regions. Specifically, we use the Local Change, Background, Style, and Subtitles subsets of OpenVE-3M, the style transfer subset of Ditto-1M, and all samples from ReCo. For quality filtering, we sample frames from each video pair at 1 FPS, compute frame-level EditScore between corresponding source-edited frame pairs, and retain video pairs with video-level scores higher than 6.0, yielding approximately 0.9M video editing samples. In the final stage, we use our CoinVE-200K dataset for the compositional-instruction video editing tuning.

Parameter Settings. In CoinVE-Edit, we use the pretrained Qwen3-VL-8B-Instruct (Bai et al., 2025a) as the foundational MLLM and the Wan2.1-T2V-14B (Wan et al., 2025) as the video DiT backbone. In the first stage training, we only employ the image editing data, and finetune the learnable query, connector and LoRA (256 rank) in MLLM. The learning rate is $1 \times 1 0 ^ { - 5 }$ , and multi-resolution training strategy is implemented. We set the max pixel number of image as $1 0 2 4 \times 1 0 2 4$ . The total batch size is 64, and the optimization step is 45K. For the second training stage, we mix the image and video editing pairs for training, and the data sampling ratio is 1 : 1. We set the max number of frames as 49 and the max pixels of each frame as 600 × 600, respectively. The LoRA in DiT is further included for optimization, and the LoRA rank in video DiT is 128. We set the learning rate as $1 \times 1 0 ^ { - 5 }$ . The training step is 32.5K with batch size of 128. In final compositional-instruction video editing tuning, only CoinVE-200K is exploited. The Mask Predictor and GateNet are ofline trained first to attain a satisfactory initial performance. Then, both modules are jointly learned with the learning rate of $5 \times 1 0 ^ { - 6 }$ and the batch size of 128. The model achieves stable convergence when the step is 10K. Experiments of all training stages are conducted on 32 NVIDIA H200 GPUs.

## 6.2 Evaluation on Single-Instruction Video Editing

Our CoinVE-Edit can naturally handle single instruction-guided video editing when the given instruction number N equals 1. Here, we compare our proposal with several open-source state-of-the-art single instructional video editing methods, including VACE (Jiang et al., 2025), Ditto (Bai et al., 2026), OpenVE-Edit (He et al., 2025), VINO (Chen et al., 2026), OmniWeaving (Pan et al., 2026), KiwiEdit (Lin et al., 2026) and SAMA (Zhang et al., 2026a). Besides, the closed-source approach Runway is also included for comparison.

Table 3 Compositional-Instruction Video Editing Comparisons on CoinVE-Bench.
<table><tr><td rowspan="2">Model</td><td colspan="3">Edit. Acc.</td><td colspan="3">Phys. Natural.</td><td>Seman. Pres.</td><td colspan="4">Video Quality</td></tr><tr><td colspan="3">SA SPA</td><td colspan="3">AN SC</td><td>CP</td><td>AQ</td><td>TQ</td><td>CQ</td><td>TS</td></tr><tr><td>Seedance 2.0</td><td>85.34</td><td>87.71</td><td>88.08</td><td>93.19</td><td>95.84</td><td>92.87</td><td>93.91</td><td>4.47</td><td>19.55</td><td>4.41</td><td>0.62</td></tr><tr><td>Kling O3</td><td>86.91</td><td>80.93</td><td>89.06</td><td>92.55</td><td>90.30</td><td>93.91</td><td>84.51</td><td>4.49</td><td>18.36</td><td>4.37</td><td>0.61</td></tr><tr><td>VACE</td><td>3.98</td><td>17.15</td><td>6.50</td><td>26.69</td><td>13.82</td><td>15.21</td><td>87.83</td><td>4.05</td><td>17.59</td><td>4.11</td><td>0.62</td></tr><tr><td>Ditto</td><td>34.69</td><td>36.41</td><td>40.85</td><td>35.96</td><td>47.79</td><td>38.48</td><td>51.98</td><td>3.59</td><td>17.24</td><td>3.96</td><td>0.67</td></tr><tr><td>VINO</td><td>83.63</td><td>66.75</td><td>89.06</td><td>78.09</td><td>82.34</td><td>85.91</td><td>61.70</td><td>4.06</td><td>17.28</td><td>4.08</td><td>0.68</td></tr><tr><td>OmniWeaving</td><td>59.67</td><td>55.94</td><td>61.11</td><td>54.49</td><td>66.03</td><td>65.10</td><td>75.09</td><td>3.79</td><td>17.95</td><td>3.84</td><td>0.62</td></tr><tr><td>KiwiEdit</td><td>76.50</td><td>69.92</td><td>80.28</td><td>78.37</td><td>78.50</td><td>80.76</td><td>70.31</td><td>4.14</td><td>19.33</td><td>4.30</td><td>0.68</td></tr><tr><td>SAMA</td><td>75.58</td><td>73.35</td><td>79.63</td><td>83.43</td><td>83.88</td><td>88.14</td><td>90.08</td><td>3.61</td><td>18.08</td><td>4.19</td><td>0.72</td></tr><tr><td>CoinVE-Edit</td><td>87.97</td><td>89.45</td><td>89.60</td><td>91.85</td><td>91.17</td><td>95.30</td><td>90.83</td><td>4.13</td><td>19.57</td><td>4.31</td><td>0.72</td></tr></table>

Table 2 summarizes the performance of six editing tasks on OpenVE-Bench (He et al., 2025). Overall, CoinVE-Edit attains the highest overall score 3.41 among open-source approaches, as judged by Gemini 2.5 Pro (Comanici et al., 2025a). Specifically, CoinVE-Edit demonstrates clear advantages in “Background Change” (3.11) and “Local Remove” (3.81), both of which require accurate localization of the target editing regions. The results validate the merit of our mask-based conditioning to supply accurate region information, and the query-wise attention mechanism for precise visual content modification.

Figure 5 further presents four qualitative comparisons on OpenVE-Bench across diferent video editing models. In general, compared with existing baselines, CoinVE-Edit produces edited videos with better instruction following, higher visual quality, and more natural motion consistency. Taking the rightmost case of “Adding Sunglasses” as an example, most MLLM-based SOTA methods, such as KiwiEdit and SAMA, fail to place the sunglasses at the specified region, i.e., on the shirt collar as required by the prompt. Instead, they tend to synthesize the sunglasses at the most semantically likely location and incorrectly make the man “wear” them. In contrast, CoinVE-Edit first localizes the target editing region and then performs content modification within the localized area. This region-aware editing manner mitigates the training-induced data bias of generative models, and enables more precise instruction-conditioned video editing. Another advantage of CoinVE-Edit lies in its ability to handle indirect or secondary efects during video editing. With Mask Predictor, both the primary editing region and the afected regions can be identified, then Q-Blending attention injects the corresponding contextual information into the DiT for coherent synthesis. As a result, CoinVE-Edit successfully removes the “blue cowboy hat” not only from the woman’s head but also from its mirror reflection at the corresponding location, leading to more visually plausible editing results.

## 6.3 Evaluation on Compositional-Instruction Video Editing

The core contribution of CoinVE-Edit is to handle the complex video editing task when the instruction number N is larger than 1. Thus, in this section, we conduct the performance comparison on the proposed CoinVE-Bench from the perspectives of Editing Accuracy, Physical Naturalness, Semantic Preservation and Video Quality. Besides using open-source SOTA video editing approaches, we additionally include two advanced closed-source business models, i.e., Seedance 2.0 (Seedance et al., 2026) and Kling O3 (Kling et al., 2025). For all other baselines, we concatenate the input multiple instructions into a single comma-separated prompt and feed it to the model for video editing. As shown in Table 3, CoinVE-Edit outperforms all open-source approaches in terms of Editing Accuracy, Physical Naturalness and Semantic Preservation, which are highly relevant to video editing quality using compositional instruction. Even comparing with frontier proprietary video generation models like Seedance 2.0 and Kling O3, ours still manifests clear advantages in editing semantic accuracy (SA), editing scope accuracy (SPA), editing persistence (EP) and motion naturalness (MN). In particular, CoinVE-Edit obtains 89.45 on scope accuracy, surpassing the best competitor Seedance 2.0 by 1.74. The performance gain clearly confirms the efectiveness of exploring region-wise correlation across diferent instructions to achieve better editing accuracy. Moreover, video quality of CoinVE-Edit is comparable

Replace the large white ceramic mixing bowl in the center foreground with a rustic wooden salad bowl.

Add a stainless steel kitchen knife lying flat on the tiled countertop surface to the right of the mixing bowl.

Replace the beige tiled kitchen counter background with a dark slate stone surface.

Remove the pile of yellow corn kernels from inside the mixing bowl.

Remove the yellow center line from the paved road.

Add a large wooden crate filled with hay bales into the open bed of the pickup truck.

Replace the chrome grille and bumper trim on the front of the truck with a matte black finish.

Replace the green grassy field and trees in the background with a dense forest of tall pine trees.

Restyle the solid dark blue-grey wall behind the subject into an oil painting style.

Remove the green stand mixer sitting on the countertop in the lower righ background.

Replace the light blue collared shirt worn by the man with a red plaid flannel shirt.

Add a thick black mustache to the man's facial hair.

![](images/0e01009ab84ddc2e6be275d67cf879f53b0180e7751babd6b58b38e21ec935dc.jpg)  
Figure 6 Quality comparison of compositional-instruction video editing on CoinVE-Bench.

to other baselines (e.g., TQ and TS), without compromising fidelity in pursuit of higher editing accuracy.

In Figure 6, we additionally show three visual editing examples in CoinVE-Bench across the open-source SAMA, closed-source Seedance 2.0, Kling O3 and our CoinVE-Edit. As illustrated in the figure, both advanced Seedance 2.0 and Kling O3 cannot perfectly deal with the complex video editing scenario. For example, for the first case, the “yellow corn kernels” are not successfully removed (e.g., Kling O3 changes the color) from the bowl by the two models. Meanwhile, some instructions could be confused with each other, leading Kling O3 to erroneously remove “the purple glass cup” on the top-right corner. Instead, CoinVE-Edit precisely follows each editing instruction while preserving the consistency of non-edited regions, and even achieves better physical plausibility (e.g., Seedance 2.0 introduces unexpected “red beans” into the bowl, while ours prevents any contents from being poured out from the empty bowl). There is also a finding that the instruction type of “local stylization” is easily fused with others, resulting in the “editing leakage” (i.e., conducting global stylization) for frontier models. Benefiting from the CoinVE-200K for model training, ours can accurately discriminate the stylization command for local regions as shown in the third case (i.e., restyle the solid dark blue-gray wall). We believe that our proposed dataset and method will contribute to the research community, and help raise the performance ceiling of compositional-instruction video editing, including for state-of-the-art proprietary commercial models.

## 6.4 Visual Analysis and Ablations

To better qualitatively examine our CoinVE-Edit framework for compositional instruction-guided video editing, we conduct a series of visualization and ablation studies.

Analysis of Editing Mask Prediction. We first visualize the predicted editing masks of two compositionalinstruction video editing exemplars in Figure 7. The estimated mask for each instruction is assigned a distinct color, which is matched with the corresponding instruction for clear association. For more faithful visualization, we upsample the masks from the latent resolution to the original frame resolution via bi-linear interpolation, and overlay them on the source video frames for display. As shown in the visualization, CoinVE-Edit accurately localizes the target editing regions associated with diferent instructions. Notably, even for challenging compositional prompts containing multiple editing operations, e.g., four instructions of the second case, ours can still clearly separate the spatial regions corresponding to object addition (e.g., the

![](images/22a3491f070b6475828c01460b74fd7e3b90a313483f7ded9488ec6f3254b176.jpg)

Table 4 Performance comparison among diferent variants of CoinVE-Edit on CoinVE-Bench.
<table><tr><td rowspan="2">Variant</td><td colspan="2">Components</td><td colspan="2">Edit. Acc.</td><td colspan="3">Phys. Natural.</td><td rowspan="2">Seman. Pres.</td></tr><tr><td>Mask</td><td>Q-Blending</td><td>SA SPA</td><td>EP</td><td>AN</td><td>SC</td><td>MN CP</td></tr><tr><td>Instr.-Concat.</td><td>X</td><td>X</td><td>82.61 84.43</td><td>89.17</td><td>86.52</td><td>90.02</td><td>92.17</td><td>89.30</td></tr><tr><td>Q-Bias</td><td>V</td><td>X</td><td>83.35 85.49</td><td>88.19</td><td>87.08</td><td>89.64</td><td>91.95</td><td>90.13</td></tr><tr><td>CoinVE-Edit</td><td>V</td><td>V</td><td>87.97 89.45</td><td>89.60</td><td>91.85</td><td>91.17</td><td>95.30</td><td>90.83</td></tr></table>

Restyle the driver (man with beard and cap) into a Claymation style. Replace the white 'Plumbing Center' van parked on the right-side background with a red delivery truck

Replace the green and black plaid shirt worn by the woman with a solid bright yellow button-down shirt. Add a large red ceramic pitcher on the wooden countertop to the left of the cooking pot. Remove the grey cooking pot containing yellow liquid in the foreground. Replace the white wall behind the woman and shelves with a dark navy blue painted wal

![](images/d74dff5a8f9b192e1d1ae3f00980608b87a36d3337c4d76d01e3bc133cf204e4.jpg)

![](images/79cf9722947e4761b954d7a298fb9f403606ea62cacf068144a185bae85dfb36.jpg)  
Edited Video  
Mask 2  
Mask 3

![](images/205c3ac2927e26a9501d2509dba75a5cd8dc17a478ba7eeafc8aa6c1236a603d.jpg)  
Mask 4

![](images/90e6545db36dd9a3fc26ebf1b79a2e8cddb0218ca1c34f839b1fd1bba661dac0.jpg)  
Edited Video

Figure 7 Visualization of predicted editing mask of each instruction for two editing examples.

red ceramic pitcher) and removal (e.g., the grey cooking pot).

Ablation Studies. To investigate the eficacy of each component in CoinVE-Edit, we additionally design two baselines, i.e., Instruction-Concatenation (Instr.-Concat.) and Q-Bias. The former merges all editing instructions into a single compound prompt and feeds it into CoinVE-Edit after removing both the mask prediction module and the Q-Blending cross-attention mechanism. It is trained as a standard single-instruction editing model. The latter keeps the estimated instruction-specific masks, but replaces Q-Blending with a bias-based injection strategy, where diferent bias values are imposed on the masked regions corresponding to diferent instructions. Table 4 details the performances on CoinVE-Bench. Solely relying on single-instruction editing training (i.e., Instr.-Concat.) will face the problem of inaccurate edit localization (i.e., decreasing SPA from 89.45 to 84.43), as the model can only implicitly infer the target regions from contextual information through attention. Besides, concatenating multiple instructions into a single prompt tends to introduce interference among instruction tokens, which degrades editing accuracy (i.e., SA: 87.97 → 82.61). When involving predicted mask as edited region guidance, it is expected that Q-Bias increases the scope accuracy (SPA) for video editing. Nevertheless, Q-Bias injects region-specific guidance through a hard mask bias in the attention layers, which may disrupt the latent feature distribution of the pretrained video backbone. As a consequence, it tends to degrade the physical naturalness of the edited videos, especially with respect to scale consistency (SC) and motion naturalness (MN). By contrast, Q-Blending cross-attention serves as a soft injection that incorporates mask-aware instruction information while better preserving the original generative prior of the backbone. Thus, our CoinVE-Edit consistently achieves the best performance across diferent evaluation metrics.

## 7 Conclusions

We introduce CoinVE-200K, a large-scale and high-quality dataset for compositional instruction-guided video editing. CoinVE-200K contains diverse video-editing pairs with 2 to 5 atomic editing operations per sample, covering multiple target subjects, including humans, objects, and backgrounds, as well as edit types such as addition, removal, modification, and global stylization. To ensure high-quality supervision, we design a robust data generation and filtering pipeline that emphasizes instruction faithfulness, visual quality, temporal consistency, and compositional diversity. We also propose CoinVE-Bench, a dedicated benchmark for evaluating compositional video editing under multi-intent instructions. Based on CoinVE-200K, we develop CoinVE-Edit, a 22B compositional video editing framework that combines MLLM-based instruction understanding with DiT-based video generation. By disentangling region-aware attention across diferent editing instructions, CoinVE-Edit achieves precise multi-region editing while preserving irrelevant content and maintaining temporal coherence. Experiments on CoinVE-Bench demonstrate the efectiveness of our dataset, benchmark, and model, establishing a strong foundation for future research on compositional instruction-guided video editing.

Limitations and Future Work. Despite its diversity, CoinVE-200K does not yet cover all possible compositional editing scenarios, such as reference-based editing, fine-grained motion editing, and complex camera control. In addition, eficient compositional editing for long videos and highly interactive scenes remains challenging. Future work will focus on expanding the editing taxonomy, improving long-range temporal consistency, reducing model cost, and developing unified video editing agents.

## References

J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond. arXiv preprint arXiv:2308.12966, 2023.

Q. Bai, Q. Wang, H. Ouyang, Y. Yu, H. Wang, W. Wang, K. L. Cheng, S. Ma, Y. Zeng, Z. Liu, Y. Xu, Y. Shen, and Q. Chen. Scaling Instruction-based Video Editing with a High-quality Synthetic Dataset. In CVPR, 2026.

S. Bai, Y. Cai, R. Chen, K. Chen, X.-H. Chen, Z. Cheng, L. Deng, W. Ding, R. Fang, C. Gao, et al. Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631, 2025a.

S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923, 2025b.

S. Batifol, A. Blattmann, F. Boesel, S. Consul, C. Diagne, T. Dockhorn, J. English, Z. English, P. Esser, S. Kulal, et al. FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space. arXiv preprint arXiv:2506.15742, 2025.

A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts, V. Jampani, and R. Rombach. Stable Video Difusion: Scaling Latent Video Difusion Models to Large Datasets. arXiv preprint arXiv:2311.15127, 2023.

T. Brooks, A. Holynski, and A. A. Efros. InstructPix2Pix: Learning to Follow Image Editing Instructions. In CVPR, 2023.

S. Cao, H. Chen, P. Chen, Y. Cheng, Y. Cui, X. Deng, Y. Dong, K. Gong, T. Gu, X. Gu, et al. HunyuanImage 3.0 Technical Report. arXiv preprint arXiv:2509.23951, 2025.

N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, K. V. Alwala, H. Khedr, A. Huang, J. Lei, T. Ma, B. Guo, A. Kalla, M. Marks, J. Greer, M. Wang, P. Sun, R. Rädle, T. Afouras, E. Mavroudi, K. Xu, T.-H. Wu, Y. Zhou, L. Momeni, R. Hazra, S. Ding, S. Vaze, F. Porcher, F. Li, S. Li, A. Kamath, H. K. Cheng, P. Dollár, N. Ravi, K. Saenko, P. Zhang, and C. Feichtenhofer. SAM 3: Segment Anything with Concepts. In ICLR, 2026.

J. Chen, F. Long, J. An, Z. Qiu, T. Yao, J. Luo, and T. Mei. Ouroboros-Difusion: Exploring Consistent Content Generation in Tuning-free Long Video Difusion. In AAAI, 2025a.

J. Chen, T. He, Z. Fu, P. Wan, K. Gai, and W. Ye. VINO: A Unified Visual Generator with Interleaved OmniModal Context. arXiv preprint arXiv:2601.02358, 2026.

Y.-C. Chen, L. Li, L. Yu, A. E. Kholy, F. Ahmed, Z. Gan, Y. Cheng, and J. Liu. UNITER: UNiversal Image-TExt Representation Learning. In ECCV, 2020.

Z. Chen, F. Long, Z. Qiu, T. Yao, W. Zhou, J. Luo, and T. Mei. Learning Spatial Adaptation and Temporal Coherence in Difusion Models for Video Super-Resolution. In CVPR, 2024.

Z. Chen, F. Long, Z. Qiu, T. Yao, W. Zhou, J. Luo, and T. Mei. Aligning Global Semantics and Local Textures in Generative Video Enhancement. In ICCV, 2025b.

G. Cheng, X. Gao, L. Hu, S. Hu, M. Huang, C. Ji, J. Li, D. Meng, J. Qi, P. Qiao, et al. Wan-Animate: Unified Character Animation and Replacement with Holistic Replication. arXiv preprint arXiv:2509.14055, 2025.

Christoph Schuhmann. Improved Aesthetic Predictor. GitHub repository, 2024.

G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen, L. Marris, S. Petulla, C. Gafney, et al. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities. arXiv preprint arXiv:2507.06261, 2025a.

G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen, et al. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities. arXiv preprint arXiv:2507.06261, 2025b.

DecartAI Team. Lucy Edit: Open-Weight Text-Guided Video Editing. GitHub repository, 2025.

M. Geyer, O. Bar-Tal, S. Bagon, and T. Dekel. TokenFlow: Consistent Difusion Features for Consistent Video Editing. In ICLR, 2024.

Y. HaCohen, N. Chiprut, B. Brazowski, D. Shalem, D. Moshe, E. Richardson, E. Levin, G. Shiran, N. Zabari, O. Gordon, et al. LTX-Video: Realtime Video Latent Difusion. arXiv preprint arXiv:2501.00103, 2025.

Y. HaCohen, B. Brazowski, N. Chiprut, Y. Bitterman, A. Kvochko, A. Berkowitz, D. Shalem, D. Lifschitz, D. Moshe, E. Porat, E. Richardson, G. Shiran, I. Chachy, et al. LTX-2: Eficient Joint Audio-Visual Foundation Model. arXiv preprint arXiv:2601.03233, 2026.

H. He, J. Wang, J. Zhang, Z. Xue, X. Bu, Q. Yang, S. Wen, and L. Xie. OpenVE-3M: A Large-Scale High-Quality Dataset for Instruction-Guided Video Editing. arXiv preprint arXiv:2512.07826, 2025.

J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi, and D. J. Fleet. Imagen Video: High Definition Video Generation with Difusion Models. arXiv preprint arXiv:2210.02303, 2022.

Z. Jiang, Z. Han, C. Mao, J. Zhang, Y. Pan, and Y. Liu. VACE: All-in-One Video Creation and Editing. In ICCV, 2025.

T. Kling, J. Chen, Y. Ci, X. Du, Z. Feng, K. Gai, S. Guo, F. Han, J. He, K. He, X. Hu, X. Hu, B. Jiang, F. Kong, et al. Kling-Omni Technical Report. arXiv preprint arXiv:2512.16776, 2025.

W. Kong, Q. Tian, Z. Zhang, R. Min, Z. Dai, J. Zhou, J. Xiong, X. Li, B. Wu, J. Zhang, et al. HunyuanVideo: A Systematic Framework For Large Video Generative Models. arXiv preprint arXiv:2412.03603, 2024.

M. Kuprashevich, G. Alekseenko, I. Tolstykh, G. Fedorov, B. Suleimanov, V. Dokholyan, and A. Gordeev. No-HumansRequired: Autonomous High-Quality Image Editing Triplet Mining. arXiv preprint arXiv:2507.14119, 2025.

K. Li, Y. He, Y. Wang, Y. Li, W. Wang, P. Luo, Y. Wang, L. Wang, and Y. Qiao. VideoChat: Chat-Centric Video Understanding. arXiv preprint arXiv:2305.06355, 2023.

B. Lin, Z. Tang, Y. Ye, J. Cui, B. Zhu, P. Jin, J. Zhang, M. Ning, and L. Yuan. Video-LLaVA: Learning United Visual Representation by Alignment Before Projection. In EMNLP, 2024.

Y. Lin, G. Liang, Z. Zeng, Z. Bai, Y. Chen, and M. Z. Shou. Kiwi-Edit: Versatile Video Editing via Instruction and Reference Guidance. arXiv preprint arXiv:2603.02175, 2026.

F. Long, T. Yao, Z. Qiu, X. Tian, J. Luo, and T. Mei. Gaussian Temporal Awareness Networks for Action Localization. In CVPR, 2019.

F. Long, Z. Qiu, T. Yao, and T. Mei. VideoStudio: Generating Consistent-Content and Multi-Scene Videos. In ECCV, 2024.

X. Luo, J. Wang, C. Wu, S. Xiao, X. Jiang, D. Lian, J. Zhang, D. Liu, and Z. Liu. EditScore: Unlocking Online RL for Image Editing via High-Fidelity Reward Modeling. In ICLR, 2026.

X. Ma, Y. Wang, X. Chen, G. Jia, Z. Liu, Y.-F. Li, C. Chen, and Y. Qiao. Latte: Latent Difusion Transformer for Video Generation. TMLR, 2025.

M. Maaz, H. Rasheed, S. Khan, and F. S. Khan. Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models. In ACL, 2024.

F. Milletari, N. Navab, and S.-A. Ahmadi. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. In 3DV, 2016.

K. Nan, R. Xie, P. Zhou, T. Fan, Z. Yang, Z. Chen, X. Li, J. Yang, and Y. Tai. OpenVid-1M: A Large-Scale High-Quality Dataset for Text-to-video Generation. In ICLR, 2025.

R. Nikhila, G. Valentin, H. Yuan-Ting, H. Ronghang, R. Chaitanya, M. Tengyu, K. Haitham, R. Roman, R. Chloe, G. Laura, M. Eric, P. Junting, A. K. Vasudev, C. Nicolas, W. Chao-Yuan, G. Ross, D. Piotr, and F. Christoph. SAM 2: Segment Anything in Images and Videos. In ICLR, 2025.

K. Pan, Q. Tian, J. Zhang, W. Kong, J. Xiong, Y. Long, S. Zhang, H. Qiu, T. Wang, Z. Lv, et al. OmniWeaving: Towards Unified Video Generation with Free-form Composition and Reasoning. arXiv preprint arXiv:2603.24458, 2026.

Y. Qian, E. Bocek-Rivele, L. Song, J. Tong, Y. Yang, J. Lu, W. Hu, and Z. Gan. Pico-Banana-400K: A Large-Scale Dataset for Text-Guided Image Editing. arXiv preprint arXiv:2510.19808, 2025.

Qwen Team. Qwen3.6-27B: Flagship-Level Coding in a 27B Dense Model, April 2026. URL https://qwen.ai/blog? id=qwen3.6-27b.

R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-Resolution Image Synthesis with Latent Difusion Models. In CVPR, 2022.

T. Seedance, D. Chen, L. Chen, X. Chen, Y. Chen, Z. Chen, Z. Chen, F. Cheng, T. Cheng, Y. Cheng, et al. Seedance 2.0: Advancing Video Generation for World Complexity. arXiv preprint arXiv:2604.14148, 2026.

U. Singer, A. Polyak, T. Hayes, X. Yin, J. An, S. Zhang, Q. Hu, H. Yang, O. Ashual, O. Gafni, D. Parikh, S. Gupta, and Y. Taigman. Make-A-Video: Text-to-Video Generation without Text-Video Data. In ICLR, 2023.

Z. Teed and J. Deng. RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. In ECCV, 2020.

D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri. Learning Spatiotemporal Features with 3D Convolutional Networks. In ICCV, 2015.

T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, et al. Wan: Open and Advanced Large-scale Video Generative Models. arXiv preprint arXiv:2503.20314, 2025.

L. Wang, Y. Xiong, Z. Wang, Y. Qiao, D. Lin, X. Tang, and L. Van Gool. Temporal Segment Networks: Towards Good Practices for Deep Action Recognition. In ECCV, 2016.

Y. Wang, S. Yang, B. Zhao, L. Zhang, Q. Liu, Y. Zhou, and C. Xie. GPT-IMAGE-EDIT-1.5M: A Million-Scale, GPT-Generated Image Dataset. arXiv preprint arXiv:2507.21033, 2025.

C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S. ming Yin, S. Bai, X. Xu, Y. Chen, Y. Chen, Z. Tang, et al. Qwen-Image Technical Report. arXiv preprint arXiv:2508.02324, 2025a.

Y. Wu, L. Chen, R. Li, S. Wang, C. Xie, and L. Zhang. InsViE-1M: Efective Instruction-based Video Editing with Elaborate Dataset Construction. In ICCV, 2025b.

Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng, D. Yin, X. Chen, W. Zhang, Y. Zhao, Y. Liu, B. Lin, J. Lyu, W. Chen, J. Zhou, X. Hu, C. Zhou, H. Yang, and J. Tang. CogVideoX: Text-to-Video Difusion Models with An Expert Transformer. In ICLR, 2025.

H. Zhang, X. Li, and L. Bing. Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding. arXiv preprint arXiv:2306.02858, 2023a.

K. Zhang, L. Mo, W. Chen, H. Sun, and Y. Su. MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing. In NeurIPS, 2023b.

X. Zhang, W. Dong, Y. Song, B. Fang, Q. Zhang, J. Wang, F. Chen, H. Zhang, H. Feng, Y. Lu, H. Zhou, C. Yuan, and J. Wang. SAMA: Factorized Semantic Anchoring and Motion Alignment for Instruction-Guided Video Editing. arXiv preprint arXiv:2603.19228, 2026a.

Z. Zhang, F. Long, Z. Qiu, Y. Pan, W. Liu, T. Yao, and T. Mei. MotionPro: A Precise Motion Controller for Image-to-Video Generation. In CVPR, 2025.

Z. Zhang, F. Long, W. Li, Z. Qiu, W. Liu, T. Yao, and T. Mei. Region-Constraint In-Context Generation for Instructional Video Editing. In ICML, 2026b.

Z. Zheng, X. Peng, T. Yang, C. Shen, S. Li, H. Liu, Y. Zhou, T. Li, and Y. You. Open-Sora: Democratizing Eficient Video Production for All. arXiv preprint arXiv:2412.20404, 2024.

L. Zhou, Y. Zhou, J. J. Corso, R. Socher, and C. Xiong. End-to-End Dense Video Captioning with Masked Transformer. In CVPR, 2018.

B. Zi, P. Ruan, X. Q. Marco Chen, S. Hao, S. Zhao, Y. Huang, B. Liang, R. Xiao, and K.-F. Wong. Señorita-2M: A High-Quality Instruction-based Dataset for General Video Editing by Video Specialists. In NeurIPS, 2025.