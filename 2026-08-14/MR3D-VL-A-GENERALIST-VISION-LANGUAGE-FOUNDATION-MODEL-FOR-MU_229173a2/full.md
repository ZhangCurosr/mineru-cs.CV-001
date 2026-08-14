# MR3D-VL: A GENERALIST VISION–LANGUAGE FOUNDATION MODEL FOR MULTIPARAMETRIC 3D MAGNETIC RESONANCE IMAGING

Zhi Qiao†, Xintong Wu<sup>1</sup>, Yichu He<sup>1</sup>, Feng Shi<sup>1</sup>

<sup>1</sup> United Imaging Intelligence, Shanghai, China Work completed while affiliated with Affiliation 1. mingshan\_ai@163.com

## ABSTRACT

Multi-parametric magnetic resonance imaging (mpMRI) is a cornerstone for brain tumor diagnosis and treatment, yet current AI models face critical limitations: their lack of natural language interaction and interpretability impedes spatial information integration and cross-modal reasoning required clinically. Key challenges arise from significant physical meaning differences across modalities, spatial misalignment due to scan intervals, and the need for complex multi-feature interpretation in tasks like glioma grading. While visual-language models (VLMs) show promise in cross-modal understanding, existing methods focus mainly on 2D image modeling, neglecting direct perception of 3D volumetric space. Although 3D VLMs have been proposed for report generation and feature alignment in 3D CT imaging, mpMRI applications demand collaborative inference across multiple imaging modalities—a requirement unmet by current solutions. To address this, we introduce Mr3D-VL, a dedicated visual-language foundation model for multi-parametric 3D MRI. With 4 billion parameters, it employs an unsupervised pre-trained shared 3D encoder and 4D rotational positional embedding for dual modality-spatial integration. Its cross-modal projection layer uses a multi-resolution feature implantation strategy to enhance feature perception across resolutions. Experimental results show significant improvements over existing 4B/7B/30B domain-specific and general-purpose models in text generation tasks, achieving a BERTScore of 0.856 for report generation, with question-answering accuracy at 0.713 and multiple-choice accuracy at 0.912.

Keywords 3D Vision Language Model, Multiparameter 3D Magnetic Resonance Imaging, 4D Rotation Position Encoder, Multi-resolution Features Injection to LLM

## 1 Introduction

Multiparametric magnetic resonance imaging (mpMRI) [10], leveraging its advantages of multimodality, high softtissue contrast resolution, and non-invasiveness, plays an irreplaceable role in the diagnosis, grading, treatment planning, and therapeutic efficacy evaluation of brain tumors and other intracranial lesions. With advancements in medical imaging technology, clinical interpretation demands for mpMRI have evolved from simple tumor detection to clinically interpretable spatial information representation—requiring imaging analysis tools to not only accurately identify lesions but also support natural language-based interactive querying to assist neurosurgeons in critical tasks such as surgical planning, risk assessment, and patient-physician communication. However, current intelligent analysis models based on mpMRI exhibit significant limitations: clinicians cannot directly query tumor spatial characteristics, inter-modal associations, and clinical implications through natural language, thereby restricting their clinical applicability.

The primary challenges include: (1) substantial differences in tissue properties reflected by different modalities (e.g., T1, T2, DWI), necessitating models to understand the physical meanings and clinical values of each modality; (2) spatial misalignment caused by tissue displacement due to voluntary/involuntary patient motion across modalities with varying scan times; and (3) the need for cross-modal associative reasoning in clinical queries (e.g., "Does the T2 hyperintense region correspond to restricted diffusion on DWI?"), requiring models to integrate multimodal features. For instance, WHO grading of gliomas requires combining T2-FLAIR mismatch signs, ADC values, and rCBV perfusion parameters; differentiation of brain metastases necessitates observing "ring enhancement" on post-contrast T1 imaging and DWI hyperintensity; while meningioma management demands evaluating T2 signal homogeneity, adjacent meningeal "dural tail" signs, and bone invasion. These diagnostic logic frameworks require integrating spatial information, signal intensities, morphological features, and functional parameters from multimodal imaging into interpretable clinical descriptions.

Several attempts have been made to adapt 2D visual-language models (VLMs) to 3D medical imaging scenarios. For example, medGemma [36] treats 3D data as sequential 2D slices, lacking explicit 3D spatial relationship modeling. Hulumed [15] proposed a 3D VLM for CT imaging, achieving progress in that domain but failing to address multimodal collaborative diagnosis challenges in mpMRI. mpLLM [47] employed feature pooling for multimodal feature collaboration, yet could not resolve spatial misalignment across modalities while suffering from substantial information loss and reduced data-specific feature representation due to pooling operations—all contributing to insufficient clinical interpretability.

To address these gaps, this study introduces Mr3D-VL, a foundational visual-language model designed specifically for multimodal 3D MRI scenarios. Our key contributions include:

1 LLMs-Driven Data Generation: Leveraging large language models (LLMs) to parse clinical reports and integrate inference results from well-established smaller models (e.g., segmentation, detection), this approach generates diverse multimodal image-text datasets for cross-modal learning, addressing challenges of data scarcity and semantic incoherence.

2 Unsupervised Pre-training with Shared Encoder: A modality-agnostic 3D visual encoder is shared across all imaging modalities, with unsupervised pre-training implemented via the Dino-v2 architecture. This enables adaptive capture of both intra-modal features and inter-modal disparities while reducing parameter counts and enhancing cross-modal fusion capabilities.

3 4D Rotational Positional Encoding: Incorporating structural characteristics of 3D volumetric data, a "depth" dimension is introduced to traditional 3D positional encoding (time, width, height), constructing a 4D coordinate system (time, depth, width, height). Rotational positional embeddings facilitate dual-dimensional fusion of modality information and spatial localization, improving reasoning about complex spatial relationships.

4 Multi-Resolution Image Feature Implantation Strategy: Low-, medium-, and high-resolution feature maps are synchronously extracted from the pyramid-style feature extraction architecture of the visual encoder. Within the cross-modal projection module, a Vision Transformer (ViT) architecture fuses features across resolutions, which are then progressively implanted into the language model’s decoder layers at each resolution level. This achieves collaborative attention to both global and detailed features, enhancing adaptability to clinical scenarios.

![](images/88dd9ae0cca5e1a1613dc3c5bd507c213778c5bdf0f58a067982db873d3d7f84.jpg)  
Figure 1: The Mr3D-VL framework integrates a vision encoder, a corss modality projection and a language model decoder to process multimodal MRI inputs, including T1, Flair, T2, ...

The introduction of Mr3D-VL marks a paradigm shift in intelligent analysis of multiparametric magnetic resonance imaging (mpMRI) toward clinically interpretable interaction. By enabling natural language querying capabilities, the model can directly respond to clinicians’ diagnostic questions, providing interpretable descriptions incorporating anatomical localization, signal characteristics, and functional parameters. This significantly enhances the efficiency and accuracy of clinical decision-making in diagnosis and treatment planning. Future research could further expand the model’s querying capabilities (e.g., supporting multi-turn dialogue and proactive clarification) while integrating additional clinical data (e.g., genomic information, treatment history) to construct a more comprehensive intelligent assistance system for brain tumor management. Moreover, through multicenter clinical validation and the development of standardized datasets, widespread clinical adoption of Mr3D-VL can be facilitated, ultimately achieving a closed-loop transition from "radiological diagnosis" to "precision medicine".

## 2 Related Works

## 2.1 Generalized Vision Languagse

The architectural evolution of vision-language models (VLMs) has transitioned from simple "concatenation" approaches to deeply "native unification" strategies. Leading model families exhibit significant divergence in the selection and fusion strategies of visual encoders, connectors (adapters), and language models, adapting to diverse requirements ranging from edge deployment to complex reasoning, and from static images to dynamic videos.

Visual Encoder The Vision Transformer (ViT) has been widely adopted as the visual encoder in mainstream VLMs, such as the InternVL series [49], Qwen series [2], Seed 1.5 VL [9], and Kimi-VL [42]. For ViT pre-training, contrastive learning strategies [54] are typically employed for initial visual encoder training. In contrast to these dominant VLM architectures, our approach utilizes a CNN-based framework [13] as the visual encoder, leveraging the DINO framework [30] for unsupervised learning. In medical imaging (e.g., CT, MRI, pathology slides), while ViT demonstrates strong capabilities in natural image processing, CNNs remain the dominant backbone in both clinical applications and academic research due to several advantages:(1) Translation Invariance and Local ization Assumptions: CNNs inherently incorporate translation invariance and locality assumptions, enabling faster convergence even on small datasets while mitigating overfitting risks [29, 23]. (2) Computational Efficiency: Con volutional operations exhibit O(N) complexity and efficiently handle ultra-large images through sliding windows and multi-scale strategies, ensuring controlled GPU memory usage [16]. (3) Hierarchical Feature Extraction: Con volutional kernels focus on local feature extraction, with lower layers capturing edges/textures and higher layers extracting semantic information. This hierarchical structure aligns with human visual perception and medical image interpretation logic [18]. (4) 3D Spatial Continuity: 3D CNNs (e.g., 3D U-Net) perform volumetric convolutions di rectly in voxel space, fully exploiting z-axis continuity information [13]. Contrastive learning frameworks like CLIP [54] rely on massive "image-text" pairs to align visual and textual semantics. However, high-quality "image-text" datasets are extremely scarce in medicine, particularly for multiparametric MRI scenarios where clinicians provide holistic judgments across multimodal imaging rather than textual descriptions for single modalities. The prohibitive costs and expertise requirements of data annotation have long hindered large-scale supervised learning applications. DINO’s purely visual self-supervised learning (SSL) [30] eliminates dependence on textual labels entirely. Through a "student-teacher" network architecture, it enables models to learn intrinsic image structures, textures, and shape invariances autonomously, addressing medical imaging’s "data silos" and "annotation bottlenecks."

Adapter Layer LLaVA [22] simply concatenates visual features at the language model’s input layer, while PaliGemma [4] employs a linear layer (instead of complex MLPs or Transformers) as the connector. Dynamic routing attention mechanisms or cross-attention layers have also been explored for adapter design [41, 44]. Qwen3-VL [2] introduces DeepStack technology [26], dynamically injecting visual features into multiple LLM layers (rather than solely at the input layer) to achieve deep vision-language fusion. DeepSeek VL [25] adopts a hybrid visua encoder combining a "semantic-aligned encoder" (for global semantics) and a "high-resolution encoder" (for fine details), fused via specialized projection layers to support high-resolution learning. MiniCPM-V [55] replaces traditional MLPs with a Resampler module, compressing visual features into a fixed number of tokens while enabling unified tokenization (single/multi-image/video processing) and standardized workflows. While these models primarily train on natural images (with some incorporating medical imaging), natural images typically feature clear object boundaries and high foreground-background contrast, facilitating semantic disambiguation. In medical imaging, many lesions (e.g., early-stage liver cancer, white matter lesions) exhibit minimal grayscale differences from surrounding tissues, requiring diagnosis based on subtle textural variations (e.g., lung nodule spiculation, nuclear atypia in cell membranes) at sub-pixel levels. Although resampling techniques significantly reduce feature dimensionality, they risk losing critical detail information. Drawing inspiration from DeepSeek VL’s high-resolution encoder performance and Qwen3-VL’s successful DeepStack application, we propose a specialized multi-resolution visual token implantation strategy for medical imaging. This approach fully leverages features generated at different resolutions by the visual encoder to achieve deep vision-language fusion while preserving diagnostic-relevant details.

![](images/25b525495350d87c32010099878653829e7e338158815548f17ef0232b1d9c6f.jpg)  
Figure 2: The pipeline of Pre-training & Post-training

## 2.2 Vision Language in Medical Domain

LLaVA-Med [19] rapidly transforms general-purpose vision-language capabilities into medical expertise through high-quality medical instruction tuning [57]. With relatively limited modality support primarily focused on 2D imaging, its training data is sourced from authoritative repositories like PubMed<sup>1</sup>, utilizing 2D illustrations from biomedical literature. Its streamlined architecture employs a simple adapter to bridge the visual encoder and large language model (LLM), lowering barriers for secondary development.MedGemma [36] achieves accurate identification of X-rays, CT scans, and pathology slides—even supporting 3D volumetric processing—by integrating a visual encoder trained via the MedSigLIP strategy and adopting Gemma3 [40] as its LLM backbone. Its architecture resembles LLaVA-Med’s simple adapter-based image-text concatenation approach, a strategy also employed by Lingshu [43]. Both MedGemma and Lingshu propose detailed dataset processing pipelines, constructing massive multimodal datasets encompassing text and imaging while innovatively incorporating Long Chain-of-Thought (Long CoT) reasoning [56] to enhance inference quality. Unlike models that flatten 3D slices or sample video frames as compromises, Hulu-Med [15] treats image patches as universal units through 2D Rotary Position Embedding (2D RoPE). This enables it to analyze hour-long surgical videos or 3D CT volumes by comprehending both anatomical structures and the temporal logic of instrument movements. However, despite its 2D RoPE innovation, Hulu-Med s position encoding remains independent of text tokens. The concatenated token sequence still relies on 1D RoPE [38], preventing contextual interactions from perceiving spatial relationships among visual tokens. Additionally, Hulu-Med processes 3D data as stacked 2D slices, where (1) Disrupts volumetric spatial features by ignoring interslice correlations; (2) Increases token count exponentially, straining GPU memory; (3) Requires larger-scale training data due to prolonged context windows. Unlike existing medical VLMs, our method directly models 3D volumetric data while supporting multi-volume training and inference. We propose 4D Rotary Position Embedding (4D-RoPE), which assigns distinct position encodings to: Text tokens (1D sequence), Volumetric depth (z-axis), Height (y-axis) and Width (x-axis). This unifies positional information across modalities within the LLM’s decoding space, enabling bidirectional capture of spatial relationships during vision-language interactions. By preserving intrinsic 3D structural integrity while reducing computational overhead, our approach addresses key limitations of prior works in medical imaging analysis.

## 2.3 Dataset for 3D Vision Language Model

In the development of vision-language foundation models, the curation of pretraining and fine-tuning datasets serves as the cornerstone determining both the upper bound (performance ceiling) and lower bound (application reliability) of model capabilities, particularly in specialized fields like healthcare. While numerous open-source medical datasets exist for training and evaluating vision-language models, these predominantly focus on 2D imaging modalities [11, 12, 28, 20]. Although some efforts have constructed large-scale medical multimodal datasets incorporating computed tomography (CT), positron emission tomography (PET), magnetic resonance imaging (MRI), and other modalities, the majority of these data are derived from book illustrations or internet screenshots. Despite involving 3D modalities in theory, the selected images often represent single 2D slices from 3D volumetric data [21, 24, 39]. The acquisition of 2D image-text datasets is relatively straightforward—scalable through web scraping or book digi tization—resulting in massive datasets. In contrast, 3D image-text datasets, which better align with clinical require ments, face significant collection challenges due to privacy regulations, annotation complexity, and storage demands. Consequently, existing 3D medical datasets are limited in scale and predominantly feature CT modality [1, 58, 5, 3]. Current approaches to 3D representation learning and vision-language model training rely on unimodal cross-moda alignment (e.g., Hulumed series [15] supports 3D reasoning but uses CT-exclusive training data). However, real world clinical scenarios demand multi-modal 3D imaging inference (e.g., joint interpretation of multi-parametric MRI sequences), a challenge that remains unaddressed by existing solutions. Furthermore, no publicly available datasets support both pretraining and fine-tuning for such multi-modal 3D tasks.

To bridge this gap, our work introduces a large language model (LLM)-powered synthetic data generation pipeline comprising four key components: (1) Radiology report parsing to extract structured clinical information; (2) Expertguided segmentation/detection annotation integration for precise spatial grounding; Textual data augmentation via medical knowledge-aware paraphrasing; (3) Question-answer pair simulation to generate diverse diagnostic reasoning samples. This framework systematically addresses the scarcity of multi-modal 3D medical vision-language datasets, enabling end-to-end training of models capable of complex volumetric reasoning while maintaining clinical interpretability.

## 2.4 Position Encoder

The permutation invariance of self-attention mechanisms necessitates positional encoding to inform large language models (LLMs) about sequence order, relative distances, and structural dependencies. While early methods relied on absolute positional embeddings [45], relative positional encoding, which generalizes better to varying sequence lengths, has become the standard. Notably, Rotary Positional Embedding (RoPE) [37] has emerged as the de facto choice for modern LLMs (e.g., Llama [8] and Qwen [51]). Vision-language models (VLMs) require modality-aware positional encoding to handle heterogeneous inputs, including 1D text and 2D/3D visual data. Current approaches fall into two categories: 1D sequential designs (e.g., vanilla RoPE [37] and V2PE [7]), which flatten and concatenate all inputs into a single sequence. While simple, this approach discards native visual geometry, leading to significant performance drops in tasks requiring visual localization and spatial reasoning. Multi-dimensional designs, which extend RoPE across multiple axes (e.g., time, height, width) by partitioning embedding channels. For instance, Qwen2-VL [48] introduces Multimodal RoPE (MRoPE) to unify positional encoding for text and visual tokens. However, MRoPE allocates positional embeddings to t-h-w blocks, placing temporal information entirely in highfrequency channels. While general-purpose VLMs primarily process 2D natural images (e.g., extending to multiview or video scenarios), medical imaging predominantly involves 3D volumetric data (e.g., CT, PET, MRI). When adapting VLMs to medical contexts, some works repurpose MRoPE by treating 3D volumes as sequences of 2D slices [15, 36]. However, this approach disrupts the spatial integrity of volumetric data and inevitably causes an exponential surge in token count, complicating training due to exploding memory requirements and data scalability issues. To address these limitations, we propose a 4D Rotary Positional Encoding (4D-RoPE) strategy tailored for 3D volumetric data, extending positional axes to (time, depth, width, height). This approach preserves native 3D spatial relationships while enabling efficient cross-modal alignment in medical VLMs.

## 3 Model Architecture

Following the typical vision language framework, Mr3D-VL adopts a three-module architecture comprising a vision encoder, an MLP-based vision–language projection, and a large language model (LLM). Figure 1 depicts the detailed model structure.

Large Language Model In the primary configuration of Mr3D-VL, the Qwen3-4B-Instruct[53] serves as the foundational language model backbone. Textual embeddings and projected 3D visual embeddings are concatenated to form a cohesive input sequence. The model then processes this sequence autoregressively, predicting subsequent tokens based on preceding visual and textual cues. This architecture endows Mr3D-VL with the versatility to undertake a variety of generative tasks without requiring task-specific modifications. For textual input processing, we utilize the Qwen2Tokenizer[52], an integral component of the large language model’s backbone. This tokenizer, grounded in the Byte-Pair Encoding (BPE[59]) algorithm, boasts a substantial vocabulary size of 151,936 tokens, facilitating nuanced and efficient text representation within the model.

3D Vision Encoder The visual encoder stands as a pivotal component within visual-language models, necessitating pre-training before integration into the overarching cross-modal visual-language framework. A prevalent pre-training methodology is contrastive learning, which hinges on a substantial corpus of image-text description pairs. Magnetic resonance (MR) imaging encompasses dozens of modalities, with over 10 commonly utilized in clinical settings, such as T1-weighted imaging, T2-weighted imaging, FLAIR, DWI, DTI, SWI, MRS, fMRI, and PWI. Physicians select various modalities based on the symptoms and diagnostic hypotheses, synthesize the modal images, and gen erate imaging reports. Consequently, each case often involves multiple modalities of data, presenting a significant challenge of extensive missing annotated data in such scenarios. Previous studies[35, 32] have adopted multimodal merging strategies, fusing and compressing multimodal images and utilizing comprehensive imaging reports as textual supervision signals for encoder pre-training via contrastive learning. However, this compression approach tends to emphasize holistic features while overlooking individual modality-specific characteristics. Moreover, employing comprehensive reports as alignment labels introduces semantic complexity and confusion, hindering the learning of visual features. mpLLM [47] introduces independent feature extraction pathways (mixture of experts) for each modality, substantially increasing parameter size and leading to encoder decoupling. This results in information frag mentation across modalities and neglects potential inter-modal correlations. Drawing inspiration from the universa encoder approach in 2D visual-language models, we advocate for a modality-agnostic 3D imaging encoder shared across all modalities. This enables the model to capture both commonalities and differences among diverse modali ties. We select MedNext [34] as the backbone network for the visual encoder, leveraging its innovative designs such as the ConvNeXt3D architecture [50], UpKern technology, and composite scaling strategies. These advancements facilitate long-range spatial dependency capture, cross-scale semantic information preservation, and performance breakthroughs in data-limited scenarios when processing three-dimensional volumetric data. Employing the Dino v2 architecture for unsupervised pre-training, our approach eliminates the need for textual supervision, focusing instead on perceptual feature capture. Higher-dimensional semantic feature extraction is delegated to the projection layer and large language model. This design not only reduces model parameters but also enhances cross-modal feature fusion capabilities, enabling the model to comprehend the intrinsic relationships among multimodal images during the pre-training phase.

![](images/5bcba79ff4a235289cfd6e6b6d5a1ae7680e2b480df52c5f48fa59387b768fcc.jpg)  
(a) for Different input modality number in the dataset

![](images/62b1ce6145f3695374e4ad36555eb14adde44def8b3a3850c808a2a6231f419e.jpg)  
(b) for Different modality types in the dataset  
Figure 3: Data distribution of original mpMRI data

Multimodal 4D Rotary Position Embedding. Positional embeddings are crucial for modeling sequential data in both vision and language modalities. Building upon the Multimodal Rotary Position Embedding (MRoPE) introduced in Qwen2-VL, we extend its capabilities to better handle spatial information in 3D volume data and proposed the Multimodal 3D Rotary Position Embedding (M3RoPE). The M3RoPE in Mr3D-VL decomposes the position embedding into four distinct components: temporal, depth, height, and width to effectively model multimodal inputs. For textual inputs, all four components use identical position IDs, making M4RoPE functionally equivalent to traditional 1D RoPE [37]. For 3D volume data, the temporal ID remains constant across visual tokens, while unique IDs are assigned to the height and width components based on each token’s spatial position within the volume data.

Multi-resolution Vision Token Injection. Drawing inspiration from DeepStack [26], we propose a novel approach that integrates visual tokens into the multi-layer architecture of a large language model (LLM). While the origina DeepStack method stacks tokens derived from multi-scale visual inputs, and Qwen3-VL extends this by extracting visual tokens from intermediate layers of the visual Vision Transformer (ViT), our approach diverges significantly to address the unique demands of medical imaging. In medical imaging, varying resolutions hold distinct significance for identifying different types of lesions and reaching a final diagnosis. Unlike the aforementioned methods, we leverage output features of different resolutions from the visual encoder. These features are then incorporated into the intermediate layers of the ViT within the projection module through skip connections—a design inspired by the UNet decoder [33]. This strategy aims to progressively enrich the feature learning process with spatial information. Subsequently, tokens are extracted from the intermediate layers of the ViT and injected into the multi-layer structure of the language model.

## 4 Training Pipeline

The entire training process is divided into two main stages: pre-training and post-training. The pre-training phase is further subdivided into three sequential sub-steps, progressing from simple to complex data and from partial to full model parameter updates. During the post-training phase, we conduct goal-oriented supervised fine-tuning (SFT), as illustrated in Figure 2. We have collected a dataset comprising 103,605 cases of brain multiparametric magnetic resonance imaging (mpMRI), totaling 460,541 images. On average, each case includes 4.4 different modality images. Statistical information regarding the distribution of different modality types per case is shown in Figure 3a, while the overall statistics for each of the eight modalities involved are presented in Figure 3b.

All image data undergoes preprocessing as follows: First, linear interpolation resampling is applied to achieve a voxel space of $1 \times 1 \times 5 \ m m ^ { 3 }$ , a choice motivated by the prevalence of thick-slice MRI in clinical multiparametric settings. Subsequently, the data is cropped to dimensions of [32, 192, 192]. Finally, pixel values are normalized to the range of -1 to 1. Regarding the imaging reports, detailed processing methods will be elaborated upon in both the pre-training and post-training sections. To facilitate a comprehensive evaluation of model performance, we partition the dataset by case (patient), ensuring that all test data remains entirely unseen during both the pre-training and posttraining phases. In the following sections, we will provide detailed descriptions of the pre-training and post-training procedures.

## 4.1 Pre-Training

## 4.1.1 Training Recipe

To enable the model to evolve from having single-modal short-text alignment capabilities to possessing mixed-modal long-context understanding abilities, we systematically divide the pre-training strategy into three distinct phases.

Stage S0: Visual-Language Alignment This phase focuses on efficiently bridging the modality gap between the visual encoder and the large language model (LLM). During this stage, only the parameters of the projection module are trained, while both the visual encoder and the LLM backbone remain frozen. We utilize a unimodal dataset comprising approximately 1 billion tokens, consisting of image-text pairs characterized by single-modal short-text attributes, including modality descriptions, local descriptions, and question-and-answer (Q&A) pairs.

Stage S1: Unimodal Pre-training In this phase, we unfreeze all model components—the visual encoder, projection module, and LLM—for joint end-to-end training. The same unimodal dataset of approximately 1 billion tokens used in Phase S0 is repurposed for this stage.

Stage S2: Multimodal Pre-training This phase aims to significantly expand the model’s contextual processing capabilities. A key change in this stage is the transition from single-modal to mixed-modal inputs, with all mode parameters participating in the training process. Training is conducted on a mixed-modal dataset containing approximately 1 billion tokens. This phase is crucial for enabling the model to handle and reason over mixed-modal inputs effectively.

## 4.1.2 Pre-Training Data

Below, we will delve into the data collection efforts for medical multi-parametric magnetic resonance imaging (MRI) from four distinct dimensions: modality description-based, tissue/organ-based, grounding-based, and spatial understanding-based approaches. For each category of data, we further divide the construction process into two subcategories: unimodal data construction and mixed-modal data construction. Detailed specifics regarding the pre-training stage data are presented in Table 1.

Data Construction based on Modality Description To construct unimodal image-text pairs by obtaining textual descriptions corresponding to each modality, we leverage the powerful text comprehension capabilities of large language models (LLMs). Initially, we dissect the original imaging reports, separating content related to each modality based on its distinct characteristics. Subsequently, the segmented textual content undergoes further refinement by the LLM, which rewrites it to enhance logical coherence and semantic richness. Since LLMs are typically pre-trained on medical corpora, the rewriting process naturally incorporates explanations of medical terminology and elucidates the logical relationships between modality-specific symptoms and final diagnoses. This text reconstruction not only transforms concise and often obscure imaging reports into logically coherent and semantically enriched narratives but also introduces external knowledge. For instance, in the case of T1-weighted images, the LLM extracts textua sentences containing the term "T1" from the imaging report. It then integrates and restates highly relevant content, such as differences in signal intensity across various tissues, precise delineation of boundaries between cerebral gray and white matter, and characteristics of potential lesion regions. This approach ensures accurate and comprehensive interpretation of unimodal content. Such descriptions facilitate a more intuitive understanding of image information for clinicians while providing rich textual annotations for subsequent data analysis and model training.

<table><tr><td></td><td>Stage-0</td><td>Stage-1</td><td>Stage-2</td></tr><tr><td>Dataset</td><td>2711712</td><td>936404</td><td>406130</td></tr></table>

Table 1: Data details of Pre-training stage

<table><tr><td colspan="2"></td><td>Training</td><td>Testing</td></tr><tr><td rowspan="2">Q&amp;A Task</td><td>Open-ended question-answer</td><td>100108</td><td>10000</td></tr><tr><td>Multiple-choice questions</td><td>100000</td><td>10000</td></tr><tr><td>Text Generation Task</td><td>Radiology report generation</td><td>88962</td><td>7959</td></tr></table>

Table 2: Data details of Post-training stage

Furthermore, from the structured imaging reports, we randomly select multiple modalities along with their corresponding descriptions. Using the LLM, we then rewrite these descriptions into a cohesive multimodal imaging narrative. This method allows the generation of multiple distinct experimental samples from a single case while eliminating instances where a modality is mentioned in the original report but absent from the imaging data, thereby preventing informational confusion. Additionally, it expands the report to include detailed descriptions across various modalities. For example, a report might mention findings from both T1- and T2-weighted images. Through LLM reasoning, information from both modalities is integrated to provide a detailed description of the lesion’s manifestations across different modalities, including its size, shape, and signal characteristics. This enriched presentation of mixed-modal data offers clinicians a more comprehensive and accurate basis for diagnosis.

Data Construction based on Anatomy We begin by dissecting the original imaging reports, segregating the content related to different anatomical organs and regions based on their distinct characteristics. Subsequently, we employ a large language model (LLM) to rewrite the text, generating logically coherent and semantically rich descriptions. For instance, in brain imaging reports, if there is a description pertaining to the hippocampus, the LLM can further supplement the text with background knowledge regarding the morphology, volumetric changes, and relationships with surrounding tissues of the hippocampus. It then integrates this supplementary information with the existing content in the report to provide a re-described account of this region. Our objective is to foster stronger cross-moda learning between the textual descriptions of different anatomical regions and their corresponding image areas. We aim for the model to perceive distinct image regions when analyzing the textual descriptions of various anatomical parts. To achieve this, for each case sample, we require multi-regional image-text data, compelling the model to perform matching and discrimination tasks between the textual descriptions of different anatomical parts and their corresponding image regions. This approach enhances the model’s ability to integrate and interpret multimodal medical data effectively.

Data Construction based on Segmentation&Detection results Small models exhibit unparalleled performance advantages in single-task scenarios, such as classification, detection, and segmentation. Therefore, we propose a data generation scheme based on the results derived from small models. In our dataset, a subset of data includes organ segmentation results alongside lesion detection outcomes. By leveraging these tissue structure segmentation results and lesion detection findings, we can further deduce the spatial relationships between lesions and anatomical organs. Utilizing a large language model (LLM), we describe the local regions from the perspectives of both tissue structures and lesions. For instance, in thoracic magnetic resonance imaging (MRI), given the known positions and morphologies of organs such as the heart and lungs, along with the localization of potential lesions like pulmonary nodules or cardiac valve abnormalities, the LLM can integrate the positional relationships between organs and lesions. It then describes the interplay between the lesions and normal organs, supplementing the concepts of organs and lesions, as well as their relationships, with the LLM’s inherent medical knowledge. This approach aids clinicians in comprehending the biological structures within a single modality, thereby enhancing diagnostic accuracy.

For mixed-modality data, we randomly select multiple modalities, multiple anatomical organs, and multiple lesion regions. Subsequently, we employ the LLM to generate more complex descriptions. In the generated textual descriptions, nouns referring to anatomical organs and abnormal regions are highlighted by enclosing their corresponding bounding boxes within and tags, respectively. Additionally, the LLM supplements the descriptions with bounding boxes delineating the regions of tissue structures and lesions.

Data Construction based on Spatial Cognition By reusing the aforementioned segmented imaging report data, which has been partitioned according to modality and/or anatomical regions, and incorporating partial grounding results, we aggregate all the information and input it into a large language model (LLM) to generate question-andanswer (Q&A) datasets. Furthermore, by adjusting the prompts, we compel the LLM to produce closed-set Q&A questions with predefined options. For instance, given a brain magnetic resonance imaging (MRI) scan, the LLM can generate questions such as "In which cerebral lobe is the lesion located?" and "What is the signal intensity of the lesion?", along with their corresponding answers. This approach not only facilitates the evaluation and enhancement of the model’s understanding of spatial information within a single modality but also provides clinicians with a rapid query tool and diagnostic aid.

By conducting data collection and organization efforts across the aforementioned four dimensions, we can establish a rich, accurate, and comprehensive data foundation for the development of visual-language models tailored to multiparametric magnetic resonance imaging (MRI) in the medical domain.

## 4.2 Post-Training

The post-training pipeline is designed to refine the model’s instruction-following capabilities. Our primary objectiv is to equip the model with the capability to precisely handle tasks related to 3D multimodal magnetic resonance (MR) imaging across diverse and complex medical scenarios. This includes the ability to articulate and infer overall imaging features and potential disease risks from 3D MR images, as well as to perform comprehensive reasoning by correlating the representational characteristics of multimodal images.

## 4.2.1 Post-Training Data

Guided by these target tasks, we systematically constructed a Supervised Fine-Tuning (SFT) dataset comprising nearly 300,000 samples. This dataset is composed of 3D MR imaging report generation data and Question-and-Answer (Q&A) data. The Q&A data is further categorized into open-ended questions and multiple-choice questions, with the latter directly reusing the Q&A dataset from the pre-training phase. Detailed data distributions are presented in Table 2.

For the report generation dataset, we leverage a Large Language Model (LLM) to expand and supplement original medical imaging reports, significantly enhancing their completeness and semantic coherence. For instance, in brain imaging reports, if there is a description involving the hippocampus, the LLM can further enrich the report with background knowledge on the hippocampus’s morphology, volume changes, and its relationship with surrounding tissues. Additionally, through prompt engineering, the LLM automatically identifies anatomical region information mentioned in the reports, such as "left upper lobe of the lung" or "right lobe of the liver," and structurally annotates them with tags. Simultaneously, for descriptions of abnormal lesions, such as key pathological features like "ground glass nodules" or "calcifications," embedded tags are used to achieve visual highlighting. This dual-tagging system not only makes regional localization and abnormal identification more intuitive but also strengthens the logical chain of the report through semantic associations.

## 5 Evalution

We conducted a systematic assessment of Mr3D-VL’s performance in interpreting 3D medical images across two tasks: Visual Question Answering (VQA) and Medical Report Generation (MRG). The VQA task was further sub divided into two categories: open-ended questions and multiple-choice questions.

## 5.1 Settings

The test data primarily originated from the fine-tuning dataset constructed during the post-training phase, with detailed statistics provided in Table 2. All experiments were conducted using 4 NVIDIA A40 GPUs. For comprehensive benchmarking, we evaluated our model against medical vision-specific models (e.g., Lingshu [43], HuluMed [15]) and general-purpose models (e.g., Qwen3.5 series [2]). Since Lingshu and the Qwen3.5 series natively lack support for 3D volumetric data, we adapted them for 3D evaluation by slicing each volume into sequential 2D image stacks and treating the task as a multi-image assessment. While HuluMed supports volumetric data, its training corpus predominantly consisted of CT scans, which differ significantly from MRI in imaging principles and visual characteristics. Thus, we uniformly adopted a multi-image evaluation protocol for 3D tasks.

We employed the following five metrics to assess model performance:

\* BLEU-4 [31]: Evaluates fluency and local accuracy of generated text.

\* ROUGE-L [27]: Measures coherence and information completeness in generated text.

\* BERTScore [6]: Assesses semantic plausibility using the multilingual MiniLM embedding model<sup>2</sup>.

\* METEOR [17]: Gauges lexical diversity and syntactic flexibility of generated text.

\* CIDEr [46]: Evaluates naturalness and human preference in generated text.

\* RadGraph-F1 [14]: Measures their ability to accurately capture and represent clinical entities and relationships.

## 5.2 Efficiency and Deployability Analysis

We conduct the FLOPs and max GPU memory utilization computation for the proposed MR3D model, along with comparisons to the mainstream large models. The details are shown in Table. 3.

For Computational Efficiency (FLOPs), The MR3D model demonstrates significantly lower FLOPs compared to mainstream large models (e.g., Qwen3.5-4B, Hulu-med-7B, Lingshu-7B) under both unimodal and 5-modal input settings. For 5-modal input, MR3D achieves 96% reduction in FLOPs (2,064.64 G vs. 57,186.88 G for Qwen3.5- 4B) while maintaining competitive performance, indicating superior efficiency in multimodal processing. For GPU Memory Utilization, MR3D exhibits minimal GPU memory consumption across all configurations, with a peak usage of 9,215.77 MB under 5-modal input- 93% lower than Lingshu-7B (115,741.43 MB) and 86% lower than Hulu-med-7B (66,181.22 MB). This lightweight footprint makes MR3D highly scalable for resource-constrained environments (e.g., edge devices, low-end GPUs).

Despite its smaller parameter size (4B vs. 7B for Hulu-med/Lingshu), MR3D achieves comparable or better efficiency in multimodal tasks, suggesting optimized architecture design (e.g., sparse attention, cross-modal fusion mechanisms). The model’s low FLOPs and memory usage align with real-world deployment needs, particularly in clinical settings where rapid inference and cost-effectiveness are critical.

<table><tr><td rowspan=2 colspan=1>Model</td><td rowspan=2 colspan=1>Model Version</td><td rowspan=2 colspan=1>Parameter Size (M)</td><td rowspan=1 colspan=2>1-modal Input</td><td rowspan=1 colspan=2>5-modal Input</td></tr><tr><td rowspan=1 colspan=1>Max GPU Utilization (M)</td><td rowspan=1 colspan=1>FLOPs (G)</td><td rowspan=1 colspan=1>Max GPU Utilization (M)</td><td rowspan=1 colspan=1>FLOPs (G)</td></tr><tr><td rowspan=1 colspan=1>Qwen3.5</td><td rowspan=1 colspan=1>4B</td><td rowspan=1 colspan=1>4536.73</td><td rowspan=1 colspan=1>9948.56</td><td rowspan=1 colspan=1>11481.12</td><td rowspan=1 colspan=1>14447.93</td><td rowspan=1 colspan=1>57186.88</td></tr><tr><td rowspan=1 colspan=1>Hulu-med</td><td rowspan=1 colspan=1>7B</td><td rowspan=1 colspan=1>7499.54</td><td rowspan=1 colspan=1>21076.72</td><td rowspan=1 colspan=1>47735.32</td><td rowspan=1 colspan=1>66181.22</td><td rowspan=1 colspan=1>121409.3</td></tr><tr><td rowspan=1 colspan=1>Lingshu</td><td rowspan=1 colspan=1>7B</td><td rowspan=1 colspan=1>7201.88</td><td rowspan=1 colspan=1>38296.57</td><td rowspan=1 colspan=1>15423.61</td><td rowspan=1 colspan=1>115741.43</td><td rowspan=1 colspan=1>73770.58</td></tr><tr><td rowspan=1 colspan=1>Mr3D</td><td rowspan=1 colspan=1>4B</td><td rowspan=1 colspan=1>4048.12</td><td rowspan=1 colspan=1>8726.71</td><td rowspan=1 colspan=1>572.07</td><td rowspan=1 colspan=1>9215.77</td><td rowspan=1 colspan=1>2064.64</td></tr></table>

Table 3: FLOPs and GPU Memory Utilization

<table><tr><td colspan="8">Report Generation</td></tr><tr><td></td><td>Size</td><td>BLEU-4</td><td>ROUGE-L</td><td>BERTScore</td><td>METEOR</td><td>CIDEr</td><td>RadGraph-F1</td></tr><tr><td colspan="8">General-purpose Multimodal VLMs</td></tr><tr><td>QWen3.5</td><td>4B</td><td>0.022</td><td>0.154</td><td>0.688</td><td>0.379</td><td>0.444</td><td>0.555</td></tr><tr><td>QWen3.5</td><td>27B</td><td>0.031</td><td>0.178</td><td>0.733</td><td>0.400</td><td>0.452</td><td>0.580</td></tr><tr><td>QWen3.5</td><td>35B</td><td>0.024</td><td>0.163</td><td>0.702</td><td>0.374</td><td>0.481</td><td>0.399</td></tr><tr><td colspan="8">Medical Multimodal VLMs</td></tr><tr><td>Hulu-Med</td><td>7B</td><td>0.002</td><td>0.088</td><td>0.659</td><td>0.120</td><td>0.321</td><td>0.234</td></tr><tr><td>Lingshu</td><td>7B</td><td>0.010</td><td>0.135</td><td>0.730</td><td>0.261</td><td>0.489</td><td>0.219</td></tr><tr><td colspan="8">Proposed</td></tr><tr><td>Mr3D-VL</td><td>4B</td><td>0.169</td><td>0.353</td><td>0.856</td><td>0.496</td><td>0.655</td><td>0.601</td></tr></table>

Table 4: Evaluation for radiology report generation task

## 5.3 Experimental Results

When assessed on medical imaging report generation tasks, the QWen3.5 series models outperformed two dedicated medical imaging foundation models—Hulu-Med and Lingshu—across multiple metrics, including the smaller-scal QWen3.5-4B variant. Report generation represents a multimodal generation challenge, requiring models to interpret 3D imaging data from multiple modalities while managing the cognitive load imposed by vast volumes of visual information. Although the QWen series also faces similar challenges, its native multimodal training approach enhances image comprehension capabilities. Among QWen3.5 variants, QWen3.5-27B demonstrated superior performance over QWen3.5-35B on BLEU-4, ROUGE-L, BERTScore, and METEOR, suggesting that dense architectures achieve higher accuracy than mixture-of-experts (MoE) models at comparable parameter scales. Notably, Mr3D-VL surpassed all competing models, achieving marked improvements across all metrics: BLEU-4 (0.169) and ROUGE-L (0.353). Given that our experiments utilized a non-public dataset with unique preprocessing protocols, metrics like BERTScore, METEOR, and CIDEr better reflect Mr3D-VL’s advantages in handling multi-parametric MRI (mpMRI) data. Compared to existing models, Mr3D-VL achieved over 20% improvements across these three evaluation benchmarks: BERTScore (0.856), METEOR (0.496), and CIDEr (0.655). Detailed results are provided in Table 4.

In the open-ended question-answering subtask of Visual Question Answering (VQA), Hulu-Med and Lingshu out performed the QWen series models across all evaluation metrics. Detailed results are presented in Table 5. Unlike multimodal report generation tasks, VQA subtasks involve unimodal data generation, requiring models to process fewer images. When contextualized with performance on report generation tasks, our findings suggest that medical foundation models exhibit pronounced advantages in medical tasks with limited input images. However, these models demonstrate no such superiority when handling larger volumes of input images. Mr3D-VL achieved stateof-the-art (SOTA) performance across all metrics, surpassing existing models by 30% in each benchmark: BLEU-4 (0.525), ROUGE-L (0.738), BERTScore (0.794), METEOR (0.747), and CIDEr (0.837).

Open-ended question-answer
<table><tr><td></td><td>Size</td><td>BLEU-4</td><td>ROUGE-L</td><td>BERTScore</td><td>METEOR</td><td>CIDEr</td></tr><tr><td></td><td colspan="6">General-purpose Multimodal VLMs</td></tr><tr><td>QWen3.5</td><td>4B</td><td>0.015</td><td>0.095</td><td>0.632</td><td>0.285</td><td>0.493</td></tr><tr><td>QWen3.5</td><td>27B</td><td>0.019</td><td>0.106</td><td>0.627</td><td>0.296</td><td>0.492</td></tr><tr><td>QWen3.5</td><td>35B</td><td>0.015</td><td>0.082</td><td>0.589</td><td>0.253</td><td>0.487</td></tr><tr><td></td><td colspan="6">Medical Multimodal VLMs</td></tr><tr><td>Hulu-Med</td><td>7B</td><td>0.078</td><td>0.260</td><td>0.532</td><td>0.280</td><td>0.348</td></tr><tr><td>Lingshu</td><td>7B</td><td>0.112</td><td>0.315</td><td>0.634</td><td>0.378</td><td>0.446</td></tr><tr><td></td><td colspan="6">Proposed</td></tr><tr><td>Mr3D-VL</td><td>4B</td><td>0.525</td><td>0.738</td><td>0.794</td><td>0.747</td><td>0.837</td></tr></table>

Table 5: Evaluation for open-ended question-answer task

![](images/b572afaffd39fe49583137ff14ffce5ecf9ba51becb301d01287851d10c62c39.jpg)  
(a) Accuracy for open-ended question-answer task

![](images/2de5dca64c84e768722190209164894ec5da808554cb804d8a3a5ca2787a4695.jpg)  
(b) Accuracy for multiple-choice questions task  
Figure 4: Accuracy evaluation

For open-ended questions, in addition to the aforementioned generation-based evaluation metrics, we employed QWen3.5-27B as an external expert judge. This large language model assessed responses by comparing the question, ground-truth answer, and model-generated outputs to provide explicit "correct" or "incorrect" judgments. For multiple-choice questions, due to their finite answer options, post-processing enabled deterministic determination of "correct" or "incorrect" outcomes for each response. Figure 4 illustrates the accuracy evaluation results of all models on VQA tasks (including both open-ended and multiple-choice questions). Subfigure 4a details the open-ended question results, while 4b presents the multiple-choice question results. Experimental findings demonstrate Mr3D-VL s significant performance superiority: (1)Open-ended question accuracy: 0.713 (+20% improvement over existing models). (2 )Multiple-choice question accuracy: 0.912 (+13% improvement over existing models).

## 5.4 Discussions

Impact of Modality Count on Model Performance In prior experimental observations, we noted that models from the QWen series and medical foundation models exhibited opposing performance trends across tasks with varying numbers of input modalities. To further investigate this phenomenon, we analyzed the effects of modality count variations on four models: Lingshu-7B, Hulumed-7B, QWen3.5-27B, and Mr3D-VL, evaluating their performance

Table 6: Case Study of Report Generation Task

## GT Radiology Report

## 影像表现如下：

两侧大脑半球形态对称，中线结构居中，脑沟裂未见明显增宽或加深，脑室系统形态大小正常，未见扩张或受压移位。双侧放射冠区可见斑片状长T1、长T2信号影，FLAIR 序列呈低信号或外高内低信号，DW呈等低信号，符合陈旧性梗塞灶改变。双侧额叶皮层下及顶叶白质（侧脑室周围）可见斑片状等T1、稍长T2信号，FLAIR序列呈高信号，DWI呈等信号，提示慢性缺血性改变，Fazekas1级。第三脑室及第四脑室形态正常，未见占位效应。

## 诊断意见如下：

1. 双侧放射冠区陈旧性梗塞灶（腔隙/小软化灶）；

2. 双侧额叶、顶叶皮层下及侧脑室周围白质高信号（推测血管源性，Fazekas 1 级）；

3. 脑动脉硬化（MRA示多发局限性狭窄及远端分支减少）。

建议: 结合临床病史及其他检查综合评估。

The translation of the above text is as follows:

Imaging Findings:

The bilateral cerebral hemispheres exhibit symmetric morphology with centrally located midline structures. No significant widening or deepening ofsulci andfissures is observed. The ventricular system demonstrates normal morphology and size, without evidence ofdilation or compression-induced displacement. Patchy areas of long T1 and long T2 signal intensity are noted in the bilateral corona radiata regions, appearing hypointense or   
demonstrating peripheral hyperintensity with central hypointensity on FLAIR sequences, and isointense to hypointense on DWI. Thesefindings are consistent with changes secondary to old infarcts. Patchy areas of   
isointense T1 and slightly prolonged T2 signal intensity are present in the subcortical white matter ofthe bilateral frontal lobes and parietal lobes (periventricular regions), appearing hyperintense on FLAIR sequences and   
isointense on DWI, suggesting chronic ischemic changes (Fazekas grade 1). The third andfourth ventricles exhibit normal morphology without mass effect. Diagnostic   
Impressions:Old infarcts (lacunar or small   
encephalomalacicfoci) in the bilateral corona radiata regions; Hyperintense white matter lesions in the   
subcortical regions ofthe bilateralfrontal and parietal lobes and periventricular areas (presumed vascular in origin, Fazekas grade 1); Cerebral arteriosclerosis (MRA demonstrates multiplefocal stenoses and reduced distal branching).   
Recommendation:Comprehensive evaluation should be performed in conjunction with clinical history and   
additional examinations.

## Predicted Radiology Report

## 影像所见如下：

两侧大脑半球对称。在左基底节区及右基底节区可见斑点状长T1长 T2信号影，FLAIR序列呈低信号或外高内低信号，DWI呈等低信号，提示为陈旧性病变。双侧左额叶、右额叶、左顶叶及右顶叶的皮层下及侧脑室周围白质可见斑片状等T1稍长T2信号，FLAIR呈高信号，DWI呈等信号，符合慢性缺血性改变。脑室系统形态大小正常，左侧脑室、右侧脑室、第三脑室及第四脑室未见扩张或受压。脑沟裂未见增宽加深，中线结构居中，未见移位。

## 诊断意见如下：

1. 左基底节区及右基底节区多发腔隙性梗塞（腔隙/小软化/腔隙性梗塞）。

3. 双侧上颌窦、筛窦炎。

临床建议如下：请结合患者临床症状、病史及其他实验室检查进行综合评估与诊治。

The translation of the above text is as follows:

Imaging Findings:The bilateral cerebral hemispheres are symmetric. Patchy areas oflong T1 and long T2 signal intensity are observed in the left and right basal ganglia regions, appearing hypointense or demonstrating peripheral hyperintensity with central hypointensity on FLAIR sequences, and isointense to hypointense on DWI, suggesting chronic lesions. Patchy areas ofisointense T1 and slightly prolonged T2 signal intensity are present in the subcortical white matter ofthe bilateralfrontal lobes, parietal lobes, and periventricular regions, appearing hyperintense on FLAIR sequences and isointense on DWI, consistent with chronic ischemic changes. The ventricular system demonstrates normal morphology and size, with no dilation or compression observed in the left lateral ventricle, right lateral ventricle, third ventricle, or fourth ventricle. No widening or deepening of sulci and fissures is noted, the midline structures remain centrally located without displacement.

Multiple lacunar infarcts (lacunar/small encephalomalacic/lacunar infarcts) in the left and right basal ganglia regions.Hyperintense white matter lesions in the subcortical regions ofthe bilateralfrontal and parietal lobes and periventricular areas (Fazekas grade 1), considered to represent chronic ischemic changes of vascular origin.Bilateral maxillary and ethmoid sinusitis. Clinical Recommendations: Comprehensive evaluation and management should be performed in conjunction with the patient’s clinical symptoms, medical history, and other laboratory examinations.

using BERTScore and METEOR metrics under different modality configurations (Figure 5). As the number of input modalities increased, Hulumed-7B and Lingshu-7B demonstrated a gradual rise in METEOR scores but a concurrent decline in BERTScore. This trend was absent in QWen3.5-27B’s results. According to the QWen technical report, its training involved a massive, highly diverse dataset, suggesting that Hulumed-7B and Lingshu-7B may exhibit modality-specific biases when processing medical imaging data. Specifically, as modalities expanded to cover those comprehensible to these models, they incorporated more key information but also introduced excessive confounding factors, degrading semantic coherence. In contrast, Mr3D-VL showed no significant fluctuations in either metric, indicating its ability to integrate multimodal features for robust, holistic inference.

Significance of Native Multimodality We further visualized the dynamics of ROUGE-L (recall-based key string matching) and BERTScore (semantic accuracy) across individual cases for each model (Figure 6). While Hulumed-7B and Lingshu-7B exhibited random distributions between ROUGE-L and BERTScore, QWen3.5-27B showed a weaker positive correlation compared to Mr3D-VL.Logically, higher ROUGE-L scores (indicating more groundtruth string matches) should correlate with improved BERTScore (semantic fidelity). The lack of correlation in Hulumed-7B and Lingshu-7B suggests that while these models generated more gold-standard phrases, they also introduced semantic noise, compromising overall coherence. QWen3.5-27B’s positive correlation, independent of model size, and the absence of evidence for its use of larger-scale medical data than domain-specific baselines (e.g., Hulumed/Lingshu) imply that its performance stems from its native cross-modal training framework. Mr3D-VL, also trained with such a framework, demonstrated clear advantages in maintaining semantic consistency across modalities.

Future Work While Mr3D-VL is currently trained on brain mpMRI data, its modality-agnostic architecture supports flexible expansion to other anatomical regions. Our next steps include:

1. Dataset Expansion: Collecting large-scale mpMRI data from prostate, liver, spine, and other organs to develop a universal mpMRI foundation model.

2. Automated Data Quality Control: Implementing a multi-dimensional scoring system to filter LLMgenerated image-text pairs, removing samples with semantic drift or hallucinations to enhance training data reliability.

3. Multi-Agent Reasoning: Introducing collaborative Agent mechanisms to simulate multidisciplinary clinical consultations, leveraging a modality-specific knowledge base to construct causal Chain-of-Thought (CoT) pathways for clinically plausible diagnoses.

4. Post-Training Paradigm Shift: Transitioning from supervised fine-tuning (SFT) to reinforcement learning (RL), incorporating human feedback to improve logical reasoning and decision-making in complex scenarios.

5. Scaling Law Exploration: Following the Lingshu/HuluMed approach, we will train 7B- and 32B-parameter versions of Mr3D-VL to validate scaling laws, unlocking greater feature representation and generalization capabilities.

This roadmap aims to transition Mr3D-VL from a specialized diagnostic assistant to a universal AI foundation for precision medicine.

## 6 Conclusion

Multiparametric magnetic resonance imaging (mpMRI) serves as a cornerstone for precision diagnosis and treatment of brain tumors. However, existing artificial intelligence (AI) models lack clinical interpretability and natural language interaction capabilities, failing to meet neurosurgeons’ demands for spatial feature querying and multimodal joint diagnosis. To address challenges such as significant modality disparities, difficult spatial registration, and missing cross-modal reasoning, this study introduces Mr3D-VL—a vision-language foundation model tailored for multimodal 3D MRI volumetric data. Mr3D-VL achieves breakthroughs through four key innovations:(1) High-quality multimodal data synthesis: Leveraging large language models (LLMs) to integrate reasoning results from smaller models, generating clinically grounded image-text pairs. (2) Modality-adaptive feature extraction: Employing a shared encoder with unsupervised pretraining to dynamically capture cross-modal correlations. (3) 4D rotary positional encoding (4D-RoPE): Explicitly modeling 3D spatial relationships (depth, width, height) while incorporating multi-resolution feature fusion for hierarchical anatomical representation. (4) Cross-modal alignment optimization: Designing a native multimodal training framework to enhance semantic coherence between imaging findings and di agnostic reports. Experimental evaluations demonstrate that Mr3D-VL outperforms general-purpose LLMs and 2D vision-language models (VLMs) in multimodal diagnostic accuracy, with stable performance across varying input modalities. This work advances medically grounded AI by unifying 3D spatial intelligence with natural language explainability, addressing a critical unmet need in neuro-oncological decision support.

![](images/c3453da85c82e52111cec1801eca47b9d41d6972fe81fac95cec27aedfcb2519.jpg)

![](images/48616ba0c101bbd0fd527e3b2744a2a5c967fcce4f8edeb3e693e6505b62f7f5.jpg)

![](images/79eb28f1d015e78a502613ca7c78f0299fc0838263aec8d14930af15c8fd90b4.jpg)

![](images/47b3ffcf294999fd45b86d2abe1b389031b538ce9e365d5f1e05d20074991c1f.jpg)  
Figure 5: Observing model performance (BERTScore & METEOR) variations in report generation across input modality counts.

## References

[1] Fan Bai, Yuxin Du, Tiejun Huang, Max Q. H. Meng, and Bo Zhao. M3d: Advancing 3d medical image analysis with multi-modal large language models, 2024.

[2] Shuai Bai and etc. Qwen3-vl technical report, 2025.

[3] Pedro R. A. S. Bassi, Mehmet Can Yavuz, Kang Wang, Xiaoxi Chen, Wenxuan Li, Sergio Decherchi, Andrea Cavalli, Yang Yang, Alan Yuille, and Zongwei Zhou. Radgpt: Constructing 3d image-text tumor datasets,

![](images/d8c13abf710117a8630d1bc601be5d69fbabc585e7322e72affe093f224f6e2e.jpg)

![](images/570660cdc8e5f34fe19f6f8242ed768c39eed2ba92f25018005ba88792c4dbaf.jpg)

![](images/8884894ed3233fa382ba6da67f51ae0b4b81039270ccab721f3211a8d22bcec4.jpg)

![](images/4188c8537e294a4321a48e081a0a92ec142d6f45540a0973d14445333dae0270.jpg)

![](images/6c03a6f31e39befa095cfa1a6e9053f597d400cfd971b2e94829ecf8967ac8be.jpg)  
Figure 6: Examining performance variations of models on radiology report generation using BERTScore and ROUGE-L.

2025.

[4] Lucas Beyer and etc. Paligemma: A versatile 3b vlm for transfer, 2024.

[5] Yinda Chen, Che Liu, Xiaoyu Liu, Rossella Arcucci, and Zhiwei Xiong. Bimcv-r: A landmark dataset for 3d ct text-image retrieval, 2024.

[6] Yousang Cho and Key-Sun Choi. Evaluating causal explanation in medical reports with llm-based and humanaligned metrics, 2025.

[7] Junqi Ge, Ziyi Chen, Jintao Lin, Jinguo Zhu, Xihui Liu, Jifeng Dai, and Xizhou Zhu. V2pe: Improving multimodal long-context capability of vision-language models with variable visual position encoding, 2024.

[8] Aaron Grattafiori and etc. The llama 3 herd of models, 2024.

[9] Dong Guo and etc. Seed1.5-vl technical report, 2025.

[10] Akifumi Hagiwara, Shohei Fujita, Ryo Kurokawa, Christina Andica, Koji Kamagata, and Shigeki Aoki. Multiparametric mri, 2023.

[11] Xuehai He, Yichen Zhang, Luntian Mou, Eric Xing, and Pengtao Xie. Pathvqa: 30000+ questions for medical visual question answering, 2020.

[12] Zhi Huang, Federico Bianchi, Mert Yuksekgonul, Thomas J. Montine, and James Zou. A visual–language foundation model for pathology image analysis using medical twitter, 2023.

[13] Ademola E. Ilesanmi, Taiwo O. Ilesanmi, and Babatunde O. Ajayi. Reviewing 3d convolutional neural network approaches for medical image segmentation. Heliyon, 10(6):e27398, 2024.

[14] Saahil Jain, Ashwin Agrawal, Adriel Saporta, Steven QH Truong, Du Nguyen Duong, Tan Bui, Pierre Chambon, Yuhao Zhang, Matthew P. Lungren, Andrew Y. Ng, Curtis P. Langlotz, and Pranav Rajpurkar. Radgraph: Extracting clinical entities and relations from radiology reports, 2021.

[15] Songtao Jiang, Yuan Wang, Sibo Song, Tianxiang Hu, Chenyi Zhou, Bin Pu, Yan Zhang, Zhibo Yang, Yang Feng, Joey Tianyi Zhou, Jin Hao, Zijian Chen, Ruijia Wu, Tao Tang, Junhui Lv, Hongxia Xu, Hongwei Wang, Jun Xiao, Bin Feng, Fudong Zhu, Kenli Li, Weidi Xie, Jimeng Sun, Jian Wu, and Zuozhu Liu. Hulu-med: A transparent generalist model towards holistic medical vision-language understanding, 2025.

[16] Kunal Kawadkar. Comparative analysis of vision transformers and convolutional neural networks for medical image classification, 2025.

[17] Alon Lavie and Abhaya Agarwal. Meteor: an automatic metric for mt evaluation with high levels of correlation with human judgments. In Proceedings of the Second Workshop on Statistical Machine Translation, StatMT ’07, page 228–231, USA, 2007. Association for Computational Linguistics.

[18] Chenghao Li and Chaoning Zhang. Toward a deeper understanding: Retnet viewed through convolution, 2023.

[19] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day, 2023.

[20] Bo Liu, Li-Ming Zhan, Li Xu, Lin Ma, Yan Yang, and Xiao-Ming Wu. Slake: A semantically-labeled knowledge-enhanced dataset for medical visual question answering, 2021.

[21] Bo Liu, Ke Zou, Liming Zhan, Zexin Lu, Xiaoyu Dong, Yidi Chen, Chengqiang Xie, Jiannong Cao, Xiao-Ming Wu, and Huazhu Fu. Gemex: A large-scale, groundable, and explainable medical vqa benchmark for chest x-ray diagnosis, 2025.

[22] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023.

[23] Yahui Liu, Enver Sangineto, Wei Bi, Nicu Sebe, Bruno Lepri, and Marco De Nadai. Efficient training of visual transformers with small datasets, 2021.

[24] Alejandro Lozano, Min Woo Sun, James Burgess, Liangyu Chen, Jeffrey J Nirschl, Jeffrey Gu, Ivan Lopez, Josiah Aklilu, Austin Wolfgang Katzer, Collin Chiu, Anita Rau, Xiaohan Wang, Yuhui Zhang, Alfred Seunghoon Song, Robert Tibshirani, and Serena Yeung-Levy. Biomedica: An open biomedical image-caption archive, dataset, and vision-language models derived from scientific literature, 2025

[25] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, and Chong Ruan. Deepseek-vl: Towards real-world vision-language understanding, 2024.

[26] Lingchen Meng, Jianwei Yang, Rui Tian, Xiyang Dai, Zuxuan Wu, Jianfeng Gao, and Yu-Gang Jiang. Deepstack: Deeply stacking visual tokens is surprisingly simple and effective for lmms, 2024.

[27] Alireza Mohammadshahi, Thomas Scialom, Majid Yazdani, Pouya Yanki, Angela Fan, James Henderson, and Marzieh Saeidi. Rquge: Reference-free metric for evaluating question generation by answering the question, 2023.

[28] Abhishek Narayanan, Rushabh Musthyala, Rahul Sankar, Anirudh Prasad Nistala, Pranav Singh, and Jacopo Cirrone. Free form medical visual question answering in radiology, 2024.

[29] Seungmin Oh, Namkug Kim, and Jongbin Ryu. Analyzing to discover origins of cnns and vit architectures in medical images, 2024.

[30] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2024.

[31] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th Annual Meeting on Associationfor Computational Linguistics, ACL ’02, page 311–318, USA, 2002. Association for Computational Linguistics.

[32] Zelin Qiu, Xi Wang, Zhuoyao Xie, Juan Zhou, Yu Wang, Lingjie Yang, Xinrui Jiang, Juyoung Bae, Moo Hyun Son, Qiang Ye, Dexuan Chen, Rui Zhang, Tao Li, Neeraj Ramesh Mahboobani, Varut Vardhanabhuti, Xiaohui Duan, Yinghua Zhao, and Hao Chen. Large-scale multi-sequence pretraining for generalizable mri analysis in versatile clinical applications, 2025.

[33] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation, 2015.

[34] Saikat Roy, Gregor Koehler, Constantin Ulrich, Michael Baumgartner, Jens Petersen, Fabian Isensee, Paul F. Jaeger, and Klaus Maier-Hein. Mednext: Transformer-driven scaling of convnets for medical image segmentation, 2024.

[35] Shaohao Rui, Lingzhi Chen, Zhenyu Tang, Lilong Wang, Mianxin Liu, Shaoting Zhang, and Xiaosong Wang. Multi-modal vision pre-training for medical image analysis, 2025.

[36] Andrew Sellergren and etc. Medgemma technical report, 2025.

[37] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomput., 568(C), February 2024.

[38] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2023.

[39] Sanjay Subramanian, Lucy Lu Wang, Sachin Mehta, Ben Bogin, Madeleine van Zuylen, Sravanthi Parasa, Sameer Singh, Matt Gardner, and Hannaneh Hajishirzi. Medicat: A dataset of medical images, captions, and textual references, 2020.

[40] Gemma Team. Gemma 3 technical report, 2025.

[41] Kimi Team. Kimi-vl technical report, 2025.

[42] Kimi Team. Kimi k2.5: Visual agentic intelligence, 2026.

[43] LASA Team. Lingshu: A generalist foundation model for unified multimodal medical understanding and reasoning, 2025.

[44] V Team. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2026.

[45] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need, 2023.

[46] Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation, 2015.

[47] Arvind Murari Vepa, Yannan Yu, Jingru Gan, Anthony Cuturrufo, Michael F. Romano, Weikai Li, Fabien Scalzo, Wei Wang, and Yizhou Sun. Multimodal llm with hierarchical mixture-of-experts for vqa on 3d brain mri, 2026.

[48] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution, 2024.

[49] Weiyun Wang and etc. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency, 2025.

[50] Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, and Saining Xie. Convnext v2: Co-designing and scaling convnets with masked autoencoders, 2023.

[51] An Yang and etc. Qwen technical report, 2023.

[52] An Yang and etc. Qwen2 technical report, 2024.

[53] An Yang and etc. Qwen3 technical report, 2025.

[54] Shaoran Yang, Dongyu Wei, Hanzhi Yu, Zhaohui Yang, Yuchen Liu, and Mingzhe Chen. Contrastive languageimage pre-training model based semantic communication performance optimization, 2025.

[55] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, Qianyu Chen, Huarong Zhou, Zhensheng Zou, Haoye Zhang, Shengding Hu, Zhi Zheng, Jie Zhou, Jie Cai, Xu Han, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm-v: A gpt-4v level mllm on your phone, 2024.

[56] Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, and Xiang Yue. Demystifying long chain-of-thought reasoning in llms, 2025.

[57] Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang, Fei Wu, and Guoyin Wang. Instruction tuning for large language models: A survey, 2025.

[58] Xiaoman Zhang, Chaoyi Wu, Ziheng Zhao, Jiayu Lei, Ya Zhang, Yanfeng Wang, and Weidi Xie. Radgenomechest ct: A grounded vision-language dataset for chest ct analysis, 2024.

[59] Vilém Zouhar, Clara Meister, Juan Luis Gastaldi, Li Du, Tim Vieira, Mrinmaya Sachan, and Ryan Cotterell. A formal perspective on byte-pair encoding, 2024.

## Appendixes

During the algorithm development process, we conducted a series of experiments to validate the performance advantages of individual submodules and confirm the rationality of the entire training pipeline. Below, we share the experimental details and findings. In this process, we refrained from extensive augmentation or artificial construction of the imaging data and textual information. Instead, we directly utilized the original multimodal imaging data. For the imaging reports, we segmented the original reports based on disease types and subsequently reassembled the segmented data. The objective was to incorporate descriptive information pertinent to 47 specific diseases of interest while filtering out irrelevant content from the reports. The filtered reports were then used as the imaging reports, paired with the original multimodal imaging data to form image-text pairs for model training and validation. Given that the report reconstruction was tailored to descriptions related to the 47 diseases, we were able to precisely identify the effective disease categories covered by the reports. Consequently, in addition to employing commonly used metrics in text generation such as BLEU-4 and ROUGE-L for algorithm evaluation at this stage, we also utilized the F1 score to assess the accuracy of the model’s inferred content. Our series of experiments primarily focused on fine-tuning-based report generation tasks, and unless otherwise specified, the fine-tuning process involved updating only the parameters of the projection layer.

Larger Image Sizes Generally Yield Better Performance We adapted the Qwen3-VL framework by implementing a preprocessing pipeline that spatially resamples multimodal MRI data using a voxel spacing of [5, 1, 1] and crops them into [28, 192, 192] dimensions for input into the image encoder. The encoder subsequently maps these inputs into [1, 12, 12] image features, which align with Qwen3-VL’s native 2D image processing paradigm—a design originally optimized for 2D medical imaging and extended to video data through frame-based sampling and 2D image reconstruction. Given this architectural constraint, we initially aggregated multimodal data from each patient into a single [1, 12, 12] feature map via pooling operations, enabling direct model training within the Qwen3-VL framework. The convergence and evaluation results for this baseline configuration are presented in Figure .7. To investigate the impact of input resolution, we expanded the cropping dimensions to [32, 192, 192], resulting in [2, 12, 12] compressed features after encoder processing. These were treated as two sequential [1, 12, 12] feature maps within the Qwen3-VL framework for subsequent training. As shown in Figure .7, experimental results demonstrate that the larger input size significantly improved performance on BLEU-4 and ROUGE-L metrics, albeit with a slight decline in F1 score. Notably, models trained with higher-resolution inputs exhibited superior convergence characteristics. We hypothesize that the enhanced feature richness from larger input dimensions better facilitates cross-modal representation learning, thereby improving the model’s ability to capture nuanced relationships between imaging and textual data.

4D Rotational Positional Encoding Proves Effective When the input dimensions were expanded to [32, 192, 192], the image encoder mapped the volumetric data into compressed features of size [2, 12, 12]. In our initial experiments, these features were treated as two separate [1, 12, 12] feature maps. However, the [2, 12, 12] data structure inherently encodes spatial information along the z-axis (2 pixels), y-axis (12 pixels), and x-axis (12 pixels), representing a unique characteristic of 3D volumetric medical imaging. To fully leverage this spatial dimensionality, we designed a 4D spatial encoder to replace the original 2D encoder in the Qwen3-VL framework. This encoder explicitly models volumetric relationships by incorporating rotational positional encoding across four dimensions (x, y, z, and channel). The experimental results, depicted in Figure .8, demonstrate that the 4D spatial encoder significantly outperformed the 2D baseline across all evaluation metrics (BLEU-4, ROUGE-L, and F1 score). Moreover, the 4D variant exhibited a more stable and efficient convergence curve during training, indicating superior optimization characteristics. These findings underscore the importance of preserving and encoding volumetric spatial information in medical imaging tasks, particularly when processing high-resolution 3D data. The improved performance suggests that 4D rotational positional encoding better captures the anatomical relationships inherent in multimodal MRI datasets, thereby enhancing cross-modal alignment between imaging and textual representations.

![](images/4374c4ced34c299a15808ce660189d314883da97e74dc9ba64639a8fa0683323.jpg)

![](images/ce415103442ef634ea320b98a1aaf1a0f9b71a8429b81e512067a4f8139a4112.jpg)  
Figure .7: Impacts of variant input 3D image size

![](images/b24437642f1ae3bc19ba1c54762888834c33af5f287007f483745db0ba32577a.jpg)

![](images/1260834796c4c34f7a0f3de888b09c966bda854f4a3676853175925eaf2ffc32.jpg)  
Figure .8: 4D Rotational positional encoding

Multimodality Discrete Embedding vs Merged Embedding In our prior experiments, we employed pooling operations for multimodal data fusion to compress features. While this method assumes spatial alignment across modalities, multi-parametric MRI sequences are inherently prone to misregistration due to patient motion (voluntary or involuntary) during sequential scanning. Consequently, spatial inconsistencies frequently arise between modalities, necessitating pre-alignment through registration algorithms. However, the complexity of medical imaging data often leads to substantial registration errors, while pooling operations themselves result in modal-specific feature loss. Drawing inspiration from multimodal embedding strategies in 2D Vision-Language Models (VLMs), we adopted a discrete embedding approach that preserves raw modality-specific features. Instead of fusing modalities via pooling, we independently projected each modality into a shared cross-modal learning space, enabling the projection layer and language model to jointly capture inter-modal relationships. This method eliminates the requirement for spatial alignment between modalities. Experimental results (Figure .9) reveal that while discrete embedding underper formed pooling-based fusion across all metrics (BLEU-4, ROUGE-L, F1 score) and exhibited poorer convergence, this outcome stems from two key factors: (1) Increased Training Complexity: Discrete embedding introduces a larger number of image tokens into the language space, significantly raising optimization difficulty. (2) Approximation Benefits of Pooling: Pooling operations effectively simplify the learning problem by reducing dimensionality, thereby facilitating faster convergence. These findings suggest that while discrete embedding aligns better with the intrinsic characteristics of multi-parametric medical imaging, its effective implementation requires more sophisticated training strategies—such as hierarchical feature distillation, curriculum learning, or adaptive token sampling —to mitigate the challenges of high-dimensional cross-modal alignment. Future work will explore these directions to unlock the full potential of discrete embedding for robust medical VLM development.

![](images/c0c12b587d68b29b3184d8c955b13c9222d050b117ebd085396ee751e81080c5.jpg)

![](images/34780afb3a98d52c7eb5099bfa9db2345872ac54e0a199cb5bdff5542aa0ca21.jpg)  
Figure .9: Multumodal embedding: Discrete style vs Merging style

Modality-Agnostic Pretraining Strategy In our prior experiments, the image encoder utilized CLIP for parameter initialization, with distinct feature extraction parameters allocated for each modality. However, given the variable modality composition in multi-parametric MRI scenarios—where the number and types of imaging sequences may vary across clinical protocols—modality-specific designs could hinder scalable model deployment. To address this, we abandoned modality-sensitive architectures in favor of a shared encoder paradigm, wherein a single encoder processes all modalities through uniform parameterization. In Figure .10, Our experimental results demonstrate that this modality-agnostic approach significantly improved VLM performance across all evaluation metrics (e.g., BLEU-4 12.3%, ROUGE-L 9.7%, F1 8.2% on test set). This performance gain suggests that shared pretraining facilitates more robust cross-modal representation learning by eliminating modality-specific biases, thereby enabling the model to better capture universal anatomical patterns rather than overfitting to modality-specific artifacts. Critically, this design aligns with clinical requirements for flexible model adaptation, as the shared encoder can accommodate new imaging modalities without architectural modifications or retraining from scratch. These findings support the adoption of modality-agnostic pretraining as a foundational strategy for developing scalable medical VLMs, particularly in resource-constrained settings where protocol variability is common. Future work will explore hybrid approaches that combine shared encoders with lightweight modality adapters to further balance generalization and specialization.

Multi-Resolution Vision Token Injection We conducted a systematic evaluation of our proposed multi-resolution feature integration module to address a critical limitation in conventional vision-language models for medical imag ing. While the baseline QWen3-VL framework [2] adopted DeepStack’s deep feature aggregation strategy [26]— processing only the highest-level visual representations—we hypothesized that this approach neglects clinically essential multiscale information inherent in medical images. Radiological diagnosis typically requires simultaneous analysis of both global anatomical structures (low-resolution) and localized pathological features (high-resolution), necessitating a hierarchical feature processing paradigm.

To this end, we developed a dedicated integration module with three key innovations: (1) Pyramidal Feature Extraction: We modified the image encoder to output three levels of resolution-specific features (1/4, 1/8, and 1/16 spatial dimensions), capturing complementary visual information ranging from coarse tissue organization to finegrained lesion characteristics. (2) Progressive Cross-Modal Alignment: During visual-to-textual projection, these multiresolution features were sequentially fused through a dynamic gating mechanism, enabling adaptive weighting based on contextual relevance to the clinical task. (3) Decoder-Side Resolution Embedding: Following DeepStack’s principle [26], we injected resolution-specific features into the initial three layers of the language model decoder, allowing iterative refinement of textual predictions using complementary visual evidence at each decoding stage. Experimental results (Figure .11) demonstrate statistically significant improvements. These findings validate our architectural design by showing that explicit multiresolution feature integration better aligns with clinical reasoning patterns compared to single-resolution baselines. The performance gains suggest that our approach effectively captures the hierarchical nature of medical image interpretation, where radiologists progressively focus from globa anatomy to local abnormalities. Future work will explore optimal resolution selection strategies and interpretabilit mechanisms to further enhance clinical applicability.

![](images/558ce57a85cf753e86fa1891fc05332f5d23e4748bb23ce3482ba4c08e5b722b.jpg)

![](images/050e9b0088decdbd165f4b40578c12663763471baefe76c97ee1244b34220575.jpg)  
Figure .10: Vision encoder pretraining: Modality-Agnostic vs Modality-Aware

![](images/775911056ecdf8b9d011686284bf6292b4c39093a74fc5bd6bf0927d83c65766.jpg)

![](images/59613c5311a6eafcbdd81b0d1f4f7efdf4df6dd302d1a5794b2d3eda3701d49f.jpg)  
Figure .11: Multi-resolution vision token injection is effective?

DINO-based Self-Supervised Vision Encoder Pretraining In our prior experiments, we demonstrated that modality-agnostic pretraining significantly enhances VLM performance by eliminating supervision biases inherent in paired image-text datasets. However, the CLIP-based strategy [54] requires precisely aligned radiology reports as supervisory signals, which poses substantial challenges in clinical settings due to: (1) the high cost of expert annotation for large-scale datasets, and (2) privacy regulations limiting access to patient-identifiable textual data. These limitations motivated our exploration of fully unsupervised pretraining approaches. We adopted the DINOv2 framework [30]—a self-distillation paradigm with vision transformer architecture—for image encoder pretraining through three key adaptations for medical imaging: (1) Domain-Specific Augmentation: We implemented geomet ric transformations (elastic deformations, local pixel shuffling) and intensity perturbations (Gaussian noise, gamma correction) specifically designed for 3D medical volumes, preserving clinically relevant anatomical structures while introducing sufficient variability for self-supervised learning. (2) Multi-Scale Patch Embedding: To accommodate the hierarchical nature of medical images, we modified the patch embedding layer to process volumetric data at three spatial scales (8Œ8Œ8, 16Œ16Œ16, and 32Œ32Œ32 voxels), enabling the model to learn both fine-grained lesion features and coarse-grained organ-level representations. (3) Contrastive Distillation Optimization: We employed a student-teacher architecture with exponential moving average (EMA) updates, where the student network learns to match the teacher’s representations across multiple augmented views of the same 3D scan, without requir ing any textual supervision. Experimental results (Figure .11) reveal the advantages over CLIP-based pretraining. These findings establish self-supervised pretraining with DINOv2 as a viable alternative to paired-data approaches for medical VLM development. The ability to leverage unannotated 3D imaging datasets (e.g., from PACS archives containing millions of unlabeled scans) significantly lowers the barrier to creating specialized models for rare dis eases or under-resourced medical specialties. Future work will investigate hybrid pretraining strategies that combine self-supervised learning with weakly supervised signals from electronic health records to further enhance clinica applicability.

![](images/6b7678d65b53f109de60d4f87033f747be8ce33d62ce3e403e5eb110181c0b4a.jpg)

![](images/5edd2c693afb7a542bc920df9463cd83e287162c7d10c34ef9824a136e474a13.jpg)  
Figure .12: Vision Encoder Pretraining: DINO vs CLIP

![](images/e2372a4de208199538f92b0e2e529f937befbeae8d04742e236cb63c6e92da97.jpg)

![](images/7395d312639ef65d8b4011523d40966528d3ccb305bde7a68b0f1db3683df577.jpg)  
Figure .13: Training pipeline verification

Training Pipeline This experiment was primarily designed to validate the rationality of a two-stage training protocol first conducting projection layer training followed by full-parameter fine-tuning. As illustrated in Figure .13, after obtaining pretrained image encoders and language models, we initially performed projection layer fine-tuning (blue curve in the figure). Upon achieving stable convergence, we selected the 6th checkpoint as the initial parameter set and proceeded with full-parameter fine-tuning on the same dataset (red curve in the figure). Both convergence curves exhibited expected behavior with gradual stabilization, and the evaluation results aligned with our hypotheses: performance demonstrated further improvement after full-parameter fine-tuning. These results validate our hypothesis that decoupled training addresses the optimization conflict between modality-specific feature extraction and cross modal alignment. The projection layer serves as an information bottleneck that progressively regularizes the feature space, enabling more stable full-model fine-tuning. This pipeline is particularly advantageous for medical VLMs where: (1) pretrained weights contain domain-specific knowledge that must be preserved, and (2) limited annotated data necessitates efficient parameter updates.