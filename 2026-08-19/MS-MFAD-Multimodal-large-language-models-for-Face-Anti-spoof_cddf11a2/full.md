# MS-MFAD : Multimodal large language models for Face Anti-spoofing Detection

XiaoyongYu<sup>a,b</sup>, RongzhenLi<sup>a</sup>, ShumingShi<sup>a</sup>, XingeYou<sup>b,∗</sup>

<sup>a</sup>Mashang Consumer Finance Co., Ltd., 4th to 8th Floors, Building B2, Yuxing Plaza, No. 52 Huangshan Avenue Middle Section, Chongqing, 401121, Chongqing, China   
<sup>b</sup>Huazhong University of Science and Technology, 1037 Luoyu Road, Hongshan District, Wuhan, 430074, Hubei, China

## Abstract

Facial biometric recognition systems currently face compound threats intertwining generative AI and high-fidelity physical spoofing. Existing defenses sufer from systemic bottlenecks, including poor generalization, nonauditable reasoning, and reliance on massive, low-quality datasets. To address these challenges, we propose Multimodal Large Language Models (MFAD for face anti-spoofing detection, an explainable reasoning system for Unified Face Anti-Spoofing Detection (UFAD), accompanied by a semantic-level an notation benchmark. Unlike methods relying on external tools or coarse alignment, MFAD activates the intrinsic reasoning capabilities of Multimodal Large Language Models (MLLMs) via a fine-grained pixel-semantic anchoring mechanism. This eliminates localization hallucinations and ensures auditable reasoning paths. We introduce a cross-attack semantic-level unified annotation paradigm: by annotating only 1,000 precise masks per attack category we generate reasoning evidence chains strictly corresponding to spoofed regions. Supervised fine-tuning on the Qwen-VL foundation model demonstrates that, using limited high-quality samples, the system achieves a 40–50% relative reduction in in-domain ACER and restricts cross-domain performance degradation to within 11.62%/5.23%, significantly outperforming existing frameworks. Furthermore, under white-box adversarial attacks, detection accuracy drops by only 3.2%, validating the robustness of semantic

anchoring compared to models trained on massive short-text data. Domain practitioners rated the evidence reliability of reasoning paths at 4.57/5, with inference latency satisfying real-time deployment requirements. These results confirm that a few-shot, high-quality semantic annotation paradigm is efective for building trustworthy, explainable, and cost-eficient UFAD systems.

Keywords: Multimodal Large Language Models; Unified Face Anti-Spoofing Detecting; Post-Training

## 1. Introduction

As the core pillar of digital identity authentication, facial biometric technology has been deeply integrated into critical domains such as financial payments, public security, and smart terminals[1][2]. However, with the rapid proliferation of Generative AI and high-fidelity 3D printing technologies, UFAD is facing unprecedented compound security threats. Attack methods have evolved from early static photographs and video replay attacks[3] to hybrid forms encompassing deepfake digital face swapping[4][5][6], 3D hyperrealistic masks[7][8], and high-definition screen texture replay[9]. The intertwining of physical presentation attacks and digital forgery attacks not only significantly increases the success rate of bypassing detection but also poses severe Visual-Linguistic Face Forgery Detection (VLFFD) challenges to the generalization capabilities and judicial admissibility of existing defense systems. Although early methods based on handcrafted features[10][11] and traditional deep learning[12][13][14] have achieved certain progress on individual datasets, constructing a UFAD system that can simultaneously defend against multiple attack types, possesses auditable reasoning capabilities, and can be deployed at low cost remains a critical challenge (to be urgently resolved) in the current security landscape.

In recent years, MLLMs have been introduced to the task of face antispoofing. Investigating proprietary compact MLLMs models is of considerable necessity in face anti-spoofing[15][16]. Deploying large cloud-hosted models requires transmitting sensitive facial biometric data to remote servers, introducing data leakage risks, while imposing stringent demands on GPU memory and network stability—any service disruption may paralyze the entire authentication pipeline. Moreover, face anti-spoofing demands millisecond level inference latency for seamless user experience[17]. In contrast, compact MLLMs models can be deployed directly on edge devices, ensuring facial data never leaves the local domain. By virtue of their reduced parameter count and lower computational complexity, they enable real-time inference on consumer-grade hardware without cloud dependency, constituting a critical pathway for practical face anti-spoofing deployment[18].

However, existing approaches still exhibit significant gaps in security capabilities. Due to the expansive feature space of MLLMs and the significant distribution discrepancy in UFAD training data, MLLMs are inherently wellsuited for UFAD tasks, which integrate digital and physical functionalities into a unified model[19] [20]. Due to the strong reasoning and generalization capabilities of MLLMs, researchers have initially explored zero-shot transfer learning. Uncertainty-guided cross-adapter have improved zero-shot transferability, their alignment granularity remains restricted to the image-level or coarse region-level[21]. Consequently, they fail to provide pixel-level forgery evidence admissible in forensic contexts and are prone to localization hallucinations when confronted with micro-artifacts, such as 3D mask material anomalies or screen moiré patterns. The facial parsing capabilities of FaceXformer, proposed by Narayan et al., are limited to facial regions and fail to capture contextual forgery cues in clothing and backgrounds[22].

Researchers first conducted Continual Pre-Training (CPT) on existing multimodal models using large-scale image-text paired data. MS-UFAD[23]<sup>1</sup> and UniAttack[24] indiscriminately accumulate millions of low-quality samples. This not only introduces privacy risks and annotation noise but also leads to rapid performance saturation. However, constructing large-scale image-aligned textual data with Chain-of-Thought (CoT) reasoning remains a significant challenge for CPT methods. Recent surveys[25] have consistently highlighted that current research generally lacks a unified system capable of achieving deep cross-attack alignment at the semantic-level while simultaneously ensuring security auditability and engineering feasibility. This fundamental deficiency severely restricts the trustworthy deployment of MLLMs in real-world security applications.

To address this issue, researchers have primarily proposed two approaches: augmenting the model with external tools to enrich its semantic understanding of forgery artifacts, or incorporating CoT reasoning during post-training. In terms of tool-augmented reasoning, Zhang et al. have proposed Tool-

Augmented Reasoning for Face Anti-Spoofing (TAR-FAS)[26] allow models to invoke external operators for forensic analysis, this plug-in architecture expands the attack surface of FAS systems. However, once the external tools fail or are subjected to adversarial perturbations, the end-to-end security is immediately compromised. Given the inherent limitations of external tool invocation, such as increased inference latency and dependency on tool availability, it becomes particularly crucial to enhance the intrinsic reasoning capabilities of the model itself, enabling it to autonomously identify and reason about forgery artifacts without external assistance.

![](images/419b303849832365643084f322550e7a5c1a5e807477ff8574e653e43f66214e.jpg)  
Figure 1: VLFFD proposes a CoT construction method; however, it is applicable only to Deepfake attacks with precise forgery masks, and the method itself inherently follows the CPT paradigm, making it dificult to eliminate the reliance on large-scale image-text paired data. In contrast, our proposed data and framework are applicable to all attack types. As a post-training approach, it elicits the reasoning capabilities of MLLMs with only a small amount of data.

In the realm of enhancing the explainability of large models for antispoofing, researchers have already made explorations. Ke Sun et al. proposed a fine-grained[27], instance-level prompt generation method, demonstrating the importance of spatial information in mitigating hallucinations during CoT generation. However, their approach requires pixel-level matched Deepfake attacks, making it dificult to obtain such precise annotations for every image in physical attack scenarios. Wang et al. relied on the prompt-guided visual token masking strategy[28]. As a model training technique, promptguided vision token masking forces the model to focus on easily confused attack regions through random visual token masking to improve generalization. Nevertheless, this approach remains at a relatively macroscopic or coarse-grained stage, a major limitation of this approach is that it is exclusively applicable to physical attacks.

To address the aforementioned security capability gaps, this paper proposes MFAD, an explainable reasoning system for UFAD, accompanied by a semantic-level annotation benchmark. Driven by the practical demands of security scenarios, we redesign a fine-grained pixel-semantic anchoring mechanism. It remains challenging to unify physical and digital attacks at the semantic level, thereby hindering the construction of a truly generalized face anti-spoofing large model. We utilize semantic-level annotations to unify physical and digital attacks, enabling robust CoT reasoning without strict pixel alignment as shon in Figure 1 (b). We abandon the reliance on external tools and the blind pursuit of massive low-quality data, opting instead to directly activate the intrinsic anti-spoofing reasoning capabilities of LLMs through refined semantic annotations, thereby ensuring the auditability and anti-interference of the reasoning paths. Specifically, we construct a unified evaluation benchmark encompassing 16 categories of attacks and pioneer a cross-attack semantic-level unified annotation paradigm: we capture local forgery traces by segmenting 12 semantic regions for deepfakes; annotate moiré patterns and illumination anomalies for display replay attacks; and precisely annotate carrier attributes for mask attacks. Following a cost-quality trade-of, we annotate only 1,000 semantic masks per attack category. Through a quality control process involving dual-person verification and ambiguity arbitration, we generate reasoning evidence chains that strictly correspond to the spoofed regions, fundamentally circumventing localization hallucinations.

We conducted comprehensive system validation on mainstream MLLM foundations. Experimental results demonstrate that MFAD, approximately half of the total 16,000 samples, achieves about 40% to 50% relative reduction in in-domain Average Classification Error Rate (ACER) and restricts cross-domain performance degradation to within 11.62%/5.23%, significantly outperforming existing methods. Under Projected Gradient Descent with 10 iterations (PGD-10) and AutoAttack white-box attacks, the detection accuracy decreases by only 3.2%, validating the intrinsic defense capability of the semantic anchoring mechanism against adversarial perturbations. Supplementary evaluations by domain practitioners show that the reasoning evidence chains generated by the model receive an average score of 4.5734/5 for readability and evidence reliability, and the system’s inference latency satisfies the deployment requirements of real-time security applications. These results fully demonstrate that the few-shot, high-quality semantic annotation paradigm is an efective approach for building trustworthy, explainable, and cost-efective UFAD systems, providing a novel engineering paradigm for the deployment of MLLMs in the security domain.

Our main contributions can be summarized as follows:

(1) We comprehensively survey 16 types of physical and digital face attacks and construct 1,000 fine-grained, semantic-level CoT data samples for each category. This provides the community with a large-scale, high-quality dataset featuring precise textual descriptions, thereby laying a solid foundation for research in UFAD.

(2) By training 3B and 7B parameter models on a subset of the proposed dataset, we develop a compact face anti-spoofing large model capable of simultaneously defending against both physical and digital attacks. Remarkably, this model outperforms most existing MLLMs with significantly larger parameter scales, while demonstrating robust cross-domain generalization capabilities.

(3) During our experiments, we observed a significant trade-of between robustness and generalization when MLLs are applied to face anti-spoofing tasks, and we explore this issue through Reinforcement Learning (RL).

## 2. Related Work

In addition to the state-of-the-art methods discussed above, we first supplement our review with earlier multimodal approaches. Traditional CNNbased methods are constrained by limited feature expressiveness, typically addressing only certain types of spoofing attacks. Relying on multiple specialized models in practical deployment increases computational burden and undermines system reliability, making a unified model for all spoofing types a desirable future direction. We then provide an in-depth discussion of UFAD as a precursor to MFAD. Finally, since we also construct a dataset in this work, we briefly survey existing datasets in the face anti-spoofing domain.

## 2.1. VLM and MLLM-based Face Anti-Spoofing

The introduction of Vision-Language Model (VLM) has brought new opportunities to FAS. Fast Language-Image Pre-training (FLIP)[29] bypasses the need for complex additional modules by leveraging the zero-shot capabilities of Contrastive Language-Image Pre-training (CLIP)[30]. It aligns image features with a set of natural language-based class descriptions. This paper demonstrates that in low-data scenarios, utilizing linguistic semantics alone as an anchor can significantly reduce the feature distance between the source and target domains, thereby achieving robust cross-domain generalization. To address the misalignment issue of coarse semantics versus fine-grained textures, Fang et al. introduce a dual-branch alignment strategy[31]. One branch aligns global semantics, while the other employs textual prompts to guide the model’s attention toward local forgery cues, distilling this finegrained attention back into a lightweight visual encoder. Peng et al. proposes a bidirectional visual-language fusion network[32]. Unlike simple prompting approaches, it incorporates a bidirectional interaction module at the feature level, enabling textual features to directly participate in refining visual features, and vice versa. Furthermore, they constructed a companion dataset, which provides fine-grained textual descriptions of forgery types and artifacts for each deepfake frame.

## 2.2. Unified Physical-Digital Face Anti-Spoofing

As attack vectors evolve from isolated physical presentation attacks (e.g., printed photographs, 3D masks) to complex digital tampering (e.g., deepfake, faceswap) and hybrid attacks, constructing a unified detection framework capable of simultaneously defending against both threats has become a prominent research focus in academia. To address the significant discrepancies in feature distributions between physical and digital attacks, several studies have focused on bridging the feature gap through data-level interventions. He et al. [19] proposed a joint physical-digital attack detection framework based on the core idea of algorithmically simulating the common cues shared by both attack types. Rather than treating physical and digital attacks as independent domains, this method forces the network to learn universal forgery representations by generating synthetic data that incorporates characteristics of both attack types.

To bridge the gap between physical and digital attacks, the texturesemantic dual-track Unification framework proposes a unified architecture[33]. One track utilizes traditional CNNs/ViTs to extract high-frequency texture features, while the other leverages CLIP to extract semantic features; these are subsequently fused to yield a unified score. The study demonstrates that CLIP semantic features exhibit inherent robustness against semantic attacks, such as 3D masks, whereas texture features are more sensitive to digital synthesis attacks, proving that their complementarity constitutes the ultimate solution.

To accommodate the substantial disparities between physical and digital attacks, researchers have adopted more flexible network architectures to enable models to dynamically select processing pathways. Liu et al. utilizes a dual cross-attention mechanism and a semi-fixed Mixture-of-Experts (MoE) model to achieve generalization across diverse attack types[20]. Similarly, Zou et al. [34] introduced La-SoftMoE CLIP, which efectively improves the performance of CLIP models in unified detection tasks by allocating expert resources via a soft gating mechanism. Chen et al. [35] proposed FaceCat, a framework based on a unified generative model. It aims to simultaneously capture the deep-seated features of both physical and digital attacks through variants of Generative Adversarial Networks (GANs), thereby enhancing the security of facial recognition systems. Leveraging the generalization capabilities of large-scale pre-trained models represents another major trend in addressing cross-domain detection.

Through image-text alignment pre-training, the model leverages semantic information to assist in decision-making, demonstrating superior performance on unseen attack types. Beyond model architectures, optimization strategies during the training process are equally critical. Yu et al. [36] designed a two-stage training strategy that involves pre-training on massive general-purpose data followed by fine-tuning on specific domain data, efectively mitigating overfitting. Focusing on the visual analysis of data domain shifts within CNNs, Huang et al. [37] proposed an optimized method for selecting classification thresholds. This is of significant practical importance for balancing the false acceptance rates of physical and digital attacks during real-world deployment.

Although the aforementioned works ([38][19][36][20]) have achieved remarkable progress in unified detection, existing methods still face two primary challenges: lack of explainability and insuficient fine-grained alignment. Most existing models ([35][34]) remain black-box classifiers, failing to provide specific forgery evidence (e.g., this is a screen moiré pattern or this is the material boundary of a mask). Insuficient Fine-Grained Alignment: Existing VLM-based methods often rely on global image features and lack pixel-level semantic anchoring, making them prone to localization hallucinations.

## 2.3. Construction and Evolution of Anti-Spoofing Datasets

Data serves as the foundation for training anti-spoofing models, yet current dataset construction primarily faces the contradiction between massive low-quality data and scarce fine-grained data. Commonly used academic datasets, such as FaceForensics++ (FF++)[4] and Celeb-DF[5]<sup>2</sup>, mainly target specific digital generators and sufer from data bias; FF++ focuses on four common manipulation methods to benchmark deepfake detection, while Celeb-DF improves upon visual quality to reduce artifacts and challenge detection algorithms. In contrast, data for physical attacks often incurs high costs for custom collection, necessitating datasets like CASIA-SURF[39]<sup>3</sup> and Wide Multi-Channel presentation Attack (WMCA)[40], which introduce multi-modal channels (Color, Depth, Infrared) to capture material properties, with WMCA further expanding this to include thermal data for robust 3D mask detection. Addressing the scarcity of fine-grained annotations, CelebA-Spoof provides a large-scale collection with rich diversity across 8 scenes and over 10 camera sensors[41], ofering detailed spoofing type and attribute annotations to support comprehensive analysis. Meanwhile, Multidimensional Face Forgery Image (MFFI) tackles the challenge of complex, multi-modal forgeries by providing data specifically designed to test the robustness of models against sophisticated[42], combined manipulation attacks. To further bridge the gap between controlled lab settings and real-world deployment. MS-UFAD incorporates detailed textual descriptions for attack clues[23]—the first of its kind—alongside 795k videos and 60k images covering 52 attack methods, enabling text-guided detection methods to leverage semantic assistance for significantly improved accuracy. Although MS-UFAD is a large-scale image-text dataset capable of joint physical and digital face anti-spoofing, it still sufers from several significant limitations: (1) The textual descriptions are overly brief and contain numerous errors; (2) It lacks explicit forgery cues; and (3) It covers a limited variety of physical spoofing attacks, particularly lacking mask-based attacks.

## 3. Method

We first theoretically examine UFAD based on MLLMs, along with its existing challenges. (Section 3.1). Subsequently, we introduce the core finegrained pixel-semantic anchoring mechanism, which explicitly binds visual forgery cues to the reasoning chain (Section 3.2). Next, based on the data construction workflow, we comprehensively describe the pipeline for building the cross-attack semantic-level unified annotation benchmark, along with the multi-level quality control strategies (Section 3.3). Finally, we present the supervised fine-tuning scheme based on multimodal large language models (Section 3.4).

## 3.1. Problem Formulation

In the field of UFAD based on MLLM, the raw input consist of an face image I and text T. T denotes the textual query or prompt, which is used to instruct the large model to analyze image characteristics and subsequently provide a conclusion regarding its authenticity. Feature extraction is performed via modality encoders trained on face anti-spoofing datasets (such as a visual encoder ${ \mathcal { E } } _ { v }$ and a text encoder $\mathcal { E } _ { t } )$ . The image and text feature vectors can be denoted as $F _ { v } = \mathcal { E } _ { v } ( I ; \theta _ { v } )$ and $F _ { t } = \mathcal { E } _ { t } ( T ; \theta _ { t } )$ , respectively. $\theta _ { v }$ and $\theta _ { t }$ denote the encoder parameters. These parameters are typically frozen during the reasoning phase. By employing a linear projector $\mathcal { P }$ to align visual features with the text embedding space. The model leverages the attention mechanism to achieve dynamic cross-modal information interaction. $F _ { v } , F _ { t }$ and $P _ { v }$ are mapped to Q, K, and V , respectively. The fused multimodal content can be denoted as $H _ { \mathrm { f u s e d } }$

The fused multimodal context $H _ { \mathrm { f u s e d } }$ is concatenated with the original text instruction and injected as a conditional input into the Large Language Model (LLM) for reasoning. The inference process of the LLM is strictly modeled as the joint probability factorization of the target output sequence given the multimodal context. During the inference stage, the model employs strategies such as greedy decoding or beam search to find the next token that maximizes the conditional probability at each time step t:

$$
\hat { y } _ { t } = \arg \operatorname* { m a x } _ { y _ { t } } P ( y _ { t } \mid y _ { < t } , H _ { \mathrm { f u s e d } } , T ; \Theta )\tag{1}
$$

where $y _ { < t }$ refers to all tokens generated before time step t, denoted as $\{ y _ { 1 } , \dotsc , y _ { t - 1 } \}$ . This ensures the coherence and grammatical correctness of the generated content. T refers to the original text instruction. Serving as a system prompt or user query, it remains constant throughout the generation process, providing the task context for reasoning. Θ refers to the model parameters.

From the perspective of computational complexity, a single forward pass of the transformer architecture has a fixed computational depth[43], making it inherently insuficient for problems requiring multi-step logical reasoning. CoT essentially trades sequence length for computational depth: each generated intermediate token $z _ { t }$ serves as a complete nonlinear state update, enabling the model to construct a smooth geodesic through a highdimensional feature space where decision boundaries are otherwise rugged and non-convex. In high-fine-grained tasks such as face anti-spoofing, where genuine and forged images heavily overlap in feature space, short-text finetuning lacks such intermediate states Z, forcing the model to search for the global optimum directly on a rugged loss landscape[23, 24], which readily leads to gradient degradation or decision boundary collapse.

The prevailing paradigm for training UFAD models relies on post-training with powerful foundation models such as CLIP or Qwen-VL[44], yet the primary bottleneck remains acquiring high-quality image-text aligned data. Existing remedies including knowledge distillation, ofline CoT generation, and large-scale short-text fine-tuning[23]—inevitably introduce varying degrees of hallucination[25]. To address this, we constrain the language decoding phase with a structured three-stage CoT template (Observation → Attribution → Judgment), in which each evidence chain must be explicitly grounded in pixel-level masks. These masks serve as hard spatial anchors that impose a localization-analysis-judgment paradigm on CoT generation, fundamentally curbing hallucinations by ensuring every reasoning step is tied to verifiable visual evidence rather than open-ended, self-reinforcing textual patterns.

## 3.2. Chain-of-Thought Evidence Generation

The construction of high-quality semantic anchoring data serves as the cornerstone of the MFAD system. Most existing generation paradigms adopt a Large Language Model (LLM)-assisted approach to synergize the strengths of human experts and current proprietary LLMs. Relying entirely on humans for CoT annotation is prohibitively labor-intensive and error-prone, whereas delegating the task solely to LLMs often results in a failure to capture critical focal points and induces hallucinations[45]. To mitigate these issues, this paper employs human experts to annotate masks as guidance for the models, thereby fully leveraging the capabilities of LLMs in analyzing subtle, pixellevel features that are imperceptible to the naked eye. Section 3.2.1 details the mask generation process, and finally, Section 3.2.2 presents the overall data generation pipeline.

## 3.2.1. Semantic-level Joint Mask Generation

To facilitate precise CoT reasoning and fine-grained feature learning, we formulate the data construction process as a semantic mapping problem. Let $\mathcal { D } = \{ ( I _ { i } , M _ { i } ) \} _ { i = 1 } ^ { N }$ denote the constructed dataset comprising N samples, where $I _ { i } \in \mathbb { R } ^ { H \times W \times 3 }$ represents the input image and $M _ { i } \in \mathbb { R } ^ { H \times W \times 3 }$ denotes the corresponding pixel-level annotation mask.

Unlike conventional binary supervision, we define a color-coded semantic annotation protocol. We establish a finite set of forgery trace categories C and a bijective mapping function $\Phi : { \mathcal { C } } \to \mathbb { Z } ^ { 3 }$ that assigns a unique RGB pixel value to each semantic category. The set of all annotated semantic regions C is partitioned into three disjoint subsets based on the attack modality: $\mathcal { C } _ { \mathrm { d e e p f a k e } } , \mathcal { C } _ { \mathrm { d i s p l a y } }$ , and $\mathcal { C } _ { \mathrm { m a s k } }$ . For digital forgeries generated via face editing, reenactment, or swapping, artifacts are localized in specific facial components. We define this subset as:

$$
\begin{array} { r l } & { \mathcal { C } _ { \mathrm { d e e p f a k e } } = \left\{ c _ { \mathrm { h a i r } } , c _ { \mathrm { f o r e h e a d } } , c _ { \mathrm { e y e b r o w } } , c _ { \mathrm { e y e } } , c _ { \mathrm { e a r } } , c _ { \mathrm { n o s e } } , \right. } \\ & { \qquad \left. c _ { \mathrm { c h e e k } } , c _ { \mathrm { m o u t h } } , c _ { \mathrm { j a w } } , c _ { \mathrm { b e a r d } } , c _ { \mathrm { n e c k } } , c _ { \mathrm { c l o t h } } , c _ { \mathrm { s c e n e r y } } \right\} } \end{array}\tag{2}
$$

where each element corresponds to a parsed facial region exhibiting synthesis traces.

Physical Presentation Cues $\left( \mathcal { C } _ { \bf d i s p l a y } \right)$ : For photo display attacks $( \mathrm { e . g . }$ printed photos, screens), we annotate physical artifacts rather than semantic parts. This subset is defined as:

$$
\mathcal { C } _ { \mathrm { d i s p l a y } } = \{ c _ { \mathrm { c o n t o u r } } , c _ { \mathrm { l i g h t } } , c _ { \mathrm { t e x t } } , c _ { \mathrm { m o i r e } } , c _ { \mathrm { d i s t o r t } } , c _ { \mathrm { i c o n } } \}\tag{3}
$$

![](images/8428c699ea217ca3cb93ac6d674cdd447a4d550f5b006eb011ec5e455dd73e68.jpg)  
Figure 2: Schematic diagram of the mask generation process.

representing boundary reflections, screen moiré patterns, text overlays, and geometric distortions.

Mask Attack Regions $\left( \mathcal { C } _ { \bf m a s k } \right)$ : For physical mask attacks, the annotation targets the occluding object itself:

$$
\mathcal { C } _ { \mathrm { m a s k } } = \{ c _ { \mathrm { f a c e \_ m a s k } } , c _ { \mathrm { r e g i o n \_ m a s k } } , c _ { \mathrm { t h r e e \_ d \_ m a s k } } , c _ { \mathrm { u p p e r \_ b o d y \_ m a s k } } \}\tag{4}
$$

To enable the model to distinguish these categories implicitly through color channels during CoT generation, we define the mapping $\Phi ( c )$ for any category $c \in { \mathcal { C } }$ . The ground truth mask $M _ { i }$ at pixel location $( x , y )$ is generated by applying the mapping function Φ to the semantic label map. Formally, this is expressed as:

$$
M _ { i } ( x , y ) = \left\{ \begin{array} { l l } { \Phi ( c ) } & { \mathrm { i f ~ } ( x , y ) \in \Omega _ { c } } \\ { ( 0 , 0 , 0 ) } & { \mathrm { o t h e r w i s e ~ ( B a c k g r o u n d ) } } \end{array} \right.\tag{5}
$$

where $\Omega _ { c }$ denotes the spatial region of category c in image $I _ { i }$ . The specific encoding values are defined as follows (subset representation):

For Display Cues: For Display Cues:

$$
\begin{array} { r l r } & { \Phi ( c _ { \mathrm { c o n t o u r } } ) = ( 2 5 5 , 0 , 0 ) _ { \mathrm { R e d } } , } & { \Phi ( c _ { \mathrm { l i g h t } } ) = ( 0 , 2 5 5 , 0 ) _ { \mathrm { G r e e n } } , } \\ & { \Phi ( c _ { \mathrm { t e x t } } ) = ( 0 , 0 , 2 5 5 ) _ { \mathrm { B l u e } } , } & { \Phi ( c _ { \mathrm { m o i r e } } ) = ( 2 5 5 , 2 5 5 , 0 ) _ { \mathrm { Y e l l o w } } , } \\ & { \Phi ( c _ { \mathrm { i c o n } } ) = ( 0 , 2 5 5 , 2 5 5 ) _ { \mathrm { C y a n } } , } & { \Phi ( c _ { \mathrm { d i s t o r t } } ) = ( 2 5 5 , 1 6 5 , 0 ) _ { \mathrm { O r a n g e \ R e d } } . } \end{array}\tag{6}
$$

For Mask Regions:

$$
\begin{array} { r l } & { \Phi ( c _ { \mathrm { r e g i o n \_ m a s k } } ) = ( 2 5 5 , 0 , 2 5 5 ) _ { \mathrm { M a g e n t a } } , } \\ & { \Phi ( c _ { \mathrm { t h r e e \_ d \_ m a s k } } ) = ( 1 2 8 , 0 , 0 ) _ { \mathrm { D a r k \ R e d } } , } \\ & { \Phi ( c _ { \mathrm { u p p e r \_ b o d y \_ m a s k } } ) = ( 0 , 1 2 8 , 0 ) _ { \mathrm { D a r k \ G r e e n } } , } \\ & { \Phi ( c _ { \mathrm { f a c e \ m a s k } } ) = ( 2 2 0 , 2 0 , 6 0 ) _ { \mathrm { C r i m s o n } } . } \end{array}\tag{7}
$$

For Deepfake Facial Parts (Selected Examples):

$$
\begin{array} { r } { \Phi ( c _ { \mathrm { h a i r } } ) = ( 0 , 0 , 1 2 8 ) _ { \mathrm { D a r k ~ B l u e } } , \quad \Phi ( c _ { \mathrm { e y e } } ) = ( 2 5 5 , 1 2 8 , 0 ) _ { \mathrm { O r a n g e } } , } \\ { \Phi ( c _ { \mathrm { m o u t h } } ) = ( 1 2 8 , 2 5 5 , 0 ) _ { \mathrm { C h a r t r e u s e } } , \ : \ : \ : \Phi ( c _ { \mathrm { n o s e } } ) = ( 2 5 5 , 1 0 5 , 1 8 0 ) _ { \mathrm { H o t ~ P i n k } } . } \end{array}\tag{8}
$$

This formalization ensures that the annotation space M is discrete and semantically structured, providing a rigorous supervisory signal for multiclass forgery localization.

## 3.2.2. Unified Semantic-Level Annotation Benchmark

After obtaining the masks, we construct fine-grained CoT data based on both the masked images and the original images. Let $\mathcal { D } = \{ ( I _ { i } , M _ { i } ) \} _ { i = 1 } ^ { N }$ denote the dataset construction process, where I represents the input image and M represents the corresponding pixel-level ground truth mask. To address the scalability bottleneck of manual annotation, we define an automated generation function $\mathcal { F } _ { g e n }$ driven by a LLM (e.g., Gemini). The generation process for a single sample is defined as:

$$
R _ { r a w } = \mathcal { F } _ { g e n } ( I , M , \mathcal { P } _ { s y s } , \mathcal { K } _ { a u x } )\tag{9}
$$

where $R _ { r a w }$ is the raw generated response, $\mathcal { P } _ { s y s }$ represents the system prompts, and ${ { \ K } _ { a u x } }$ denotes the auxiliary knowledge base comprising category and trace definitions. To ensure data quality, we employ a Tri-Stage Quality Assurance Protocol. First, in the Model-Based Evaluation stage, two independent evaluator models assign quality scores $S _ { A } , S _ { B } \in [ 0 , 1 ]$ to the Chain-of-Thought (CoT) reasoning in $R _ { r a w }$ . Second, the Decision Logic dictates that samples are accepted if $S _ { A } > \tau$ and $S _ { B } > \tau$ (where τ is a high-score threshold), rejected if both scores are below the threshold, and routed to human experts for revision if the scores diverge $( { \mathrm { i . e . , ~ } } ( S _ { A } > \tau \wedge S _ { B } \leq \tau ) { \vee } ( S _ { A } \leq \tau \wedge S _ { B } > \tau ) )$ . Finally, the validated response is structured into a Visual Question Answering (VQA) tuple: $D _ { f i n a l } = ( Q , A , I )$

![](images/b87d2d23ac0139b329c59796e68fb514fc74442ecc92fe83540467bb18a77aba.jpg)  
Figure 3: CoT Data Construction Pipeline.

To guide the model in open-world face forgery analysis, we construct a composite prompt space $\mathcal { P } _ { t o t a l }$ . The prompt is structured into five distinct components:

$$
\mathcal { P } _ { t o t a l } = \{ \mathcal { P } _ { c a t } , \mathcal { P } _ { s u b } , \mathcal { P } _ { t r a c e } , \mathcal { P } _ { r o l e } , \mathcal { P } _ { c o t } \}\tag{10}
$$

Here, Category Selection $( \mathcal { P } _ { c a t } )$ defines the coarse-grained label space $\mathcal { C } _ { c o a r s e } =$ {Deepfake, Display, Mask}. Fine-grained Selection $\left( \mathcal { P } _ { s u b } \right)$ defines the specific forgery types, where $| \mathcal { C } _ { s u b } ^ { d f } | = \bar { 4 } , | \mathcal { C } _ { s u b } ^ { d i s p } | = 6$ , and $| { \mathcal { C } } _ { s u b } ^ { m a s k } | = 4$ . Forgery Trace Selection $( \mathcal { P } _ { t r a c e } )$ comprises a comprehensive set of 23 forensic traces $\mathcal { T } = \{ t _ { 1 } , \ldots , t _ { 2 3 } \}$ , characterized by both spatial and physical properties. The System Prompt $( \mathcal { P } _ { r o l e } )$ defines the agent’s role, boundary constraints, and behavioral norms, while the CoT Requirement $( \mathcal { P } _ { c o t } )$ enforces step-by-step reasoning to enhance logical consistency. During generation, the input to the model includes the visual features and explicit pixel coordinates derived from M to ground the analysis spatially.

Subsequently, the generated prompt, the original face image, and the mask are provided as inputs to the Gemini large model. The reasoning capability of large MLLMs is highly dependent on the attention mechanism of transformers. During the testing phase, the mask M directly alters the model’s attention weight distribution in the analysis phase. Let the visual attention score of the model at step t be $e _ { t }$ . After Softmax normalization, the attention distribution $A _ { t }$ is obtained:

$$
A _ { t } = \mathrm { s o f t m a x } \left( \frac { Q _ { t } K _ { V } ^ { T } } { \sqrt { d } } \right)\tag{11}
$$

When the input includes a mask M, Gemini 3.1’s attention mechanism implicitly or explicitly uses M as a filtering condition, causing a drastic change in the attention distribution $A _ { t } ^ { \prime } \mathrm { { : } }$

$$
A _ { t } ^ { \prime } = \operatorname { s o f t m a x } { \left( \frac { Q _ { t } ( K _ { V } \odot M ) ^ { T } } { \sqrt { d } } \right) }\tag{12}
$$

During the testing phase, the ⊙ operation (or an equivalent masked attention mechanism) is equivalent to performing a Hard Masking on the global visual key. When reasoning, the model’s gaze is forcibly locked within the masked region. Since hallucinations often occur because the model looks in the wrong place’ or ’over-generalizes the background’, this redistribution of attention during testing physically cuts of the visual input source for hallucination generation at the physical level.

To curate high-quality reasoning samples from the massive generated pool, we design a dual-model consensus-based adaptive filtering mechanism. Let $S _ { G P T }$ and $S _ { Q w e n }$ denote the evaluation scores assigned by GPT-4o and Qwen to a specific CoT sample $R _ { r a w }$ , respectively. We define a consistency threshold $\delta = 1$ and a quality threshold $\tau = 3$ . The filtering logic is formal-

ized as a piecewise decision function

$$
\Phi ( R _ { r a w } ) = \left\{ \begin{array} { l l } { \mathrm { A c c e p t , } } & { \mathrm { i f ~ } S _ { G P T } > \tau \land S _ { Q w e n } > \tau \land | S _ { G P T } - S _ { Q w e n } | < \delta } \\ { \mathrm { R e v i e w , } } & { \mathrm { i f ~ } | S _ { G P T } - S _ { Q w e n } | \geq \delta \land \mathrm { m a x } ( S _ { G P T } , S _ { Q w e n } ) > \tau } \\ { \mathrm { R e j e c t , } } & { \mathrm { ~ o t h e r w i s e } } \end{array} \right.\tag{13}
$$

This mechanism efectively balances automation eficiency with the recall rate of long-tail samples. Samples falling into the “Human-Review” category are routed to the refinement stage, ensuring that potentially valuable but controversial reasoning paths are not prematurely discarded.

For samples routed to the human stage, we propose a fine-grained refinement framework based on error diagnosis. This process is modeled as a mapping transformation Ψ : $R _ { r a w }  R _ { g o l d }$ , converting a defective reasoning trajectory into a gold-standard one. Human experts first perform an Error Attribution Diagnosis $\mathcal { D } _ { i a g }$ , decoupling the error space into three orthogonal categories: Factual Errors $( E _ { f a c t } , \mathrm { e . g . }$ , visual hallucinations), Logical Errors $( E _ { l o g i c } , \ \mathrm { e . g . }$ , reasoning jumps), and Knowledge Errors $( E _ { k n o w } , ~ \mathrm { e . g . }$ , lack of domain expertise). Subsequently, targeted rewriting operations $\mathcal { O } _ { p }$ are executed based on the diagnosis:

• Feature Alignment: For $E _ { f a c t }$ , experts correct the visual descriptions to align with the ground truth image.

• Logic Interpolation: For $E _ { l o g i c } ,$ , intermediate reasoning steps are inserted to bridge logical gaps (e.g., inferring B from A to reach C).

• Knowledge Injection: For $E _ { k n o w }$ , domain-specific definitions and criteria are injected to replace incorrect assumptions.

The final output $R _ { g o l d }$ ensures strict consistency among visual evidence, logical chains, and domain knowledge. The overall procedure for CoT filtering and refinement is summarized in Algorithm 1.

Ultimately, the 16,000 samples that passed the three-level verification constitute the semantic anchoring training set for Mask-CoT-FAS. In terms of sample size, this dataset is merely 1.5%–3% of existing large-scale FAS instruction-tuning datasets (e.g., FAS-MLLM, LLaFA). However, it achieves an order-of-magnitude improvement in pixel-semantic alignment density and reasoning evidence reliability, providing a solid foundation for subsequent few-shot eficient fine-tuning.

Algorithm 1: CoT Filtering and Human Refinement Protocol   
Input: Raw CoT sample $R _ { r a w }$ , Image I, Thresholds $\overline { { \tau = 3 , \delta = 1 } }$   
Output: Final CoT sample $R _ { f i n a l }$   
1 $S _ { G P T } \gets \mathrm { E v a l u a t e } ( R _ { r a w } , \mathrm { G P T - 4 o } ) $   
2 $S _ { Q w e n } \gets \mathrm { E v a l u a t e } ( R _ { r a w } , \mathrm { Q w e n } ) ;$   
3 $\Delta _ { s c o r e } \gets | S _ { G P T } - S _ { Q w e n } | ;$   
4 if $S _ { G P T } > \tau$ and $S _ { Q w e n } > \tau$ and $\Delta _ { s c o r e } < \delta$ then   
5 $R _ { f i n a l }  R _ { r a w } ;$   
; // Auto-Accept   
6 else   
7 if $\Delta _ { s c o r e } \geq \delta$ and max $( S _ { G P T } , S _ { Q w e n } ) > \tau$ then   
8 $E _ { t y p e } \gets \mathrm { D i a g n o s e E r r o r } ( R _ { r a w } , I ) ;$   
; // Human Diagnosis   
9 $R _ { f i n a l }  \mathrm { H u m a n R e w r i t e } ( R _ { r a w } , E _ { t y p e } ) ;$   
; // Rewriting & Interpolation   
10 else   
11 $R _ { f i n a l }  \mathrm { N U L L } ;$   
// Reject   
12 return $R _ { f i n a l }$

## 3.3. Training Process and Optimization Strategy

We formulate the post-training of our unified face anti-spoofing model as a constrained optimization problem, aiming to align the model with human reasoning intent while jointly detecting both physical attacks (e.g., printed photos, 3D masks) and digital forgeries (e.g., Deepfake face-swapping).

## 3.3.1. Training Objective

We perform Supervised Fine-Tuning (SFT) with a standard auto-regressive cross-entropy loss, while constraining the model within a KL trust region to prevent reward hacking[46]. The training objective is defined as:

$$
\operatorname* { m i n } _ { \theta } \mathcal { D } ( P _ { t a r g e t } ( Y | X ) \parallel P _ { \theta } ( Y | X ) ) + \lambda \mathcal { R } ( \theta )\tag{14}
$$

where $\mathcal { R } ( \theta )$ is the regularization term and λ is the regularization coeficient. Since $P _ { t a r g e t }$ is typically unknown and sparse, direct optimization risks reward hacking—the model deviating from its pre-trained representations.

We therefore formulate post-training as a constrained optimization problem, restricting the model within a KL trust region centered on the reference policy $\pi _ { r e f } .$

$$
\operatorname* { m a x } _ { \pi _ { \theta } } \mathbb { E } [ \mathcal { I } ( X , Y ) ] \quad \mathrm { s . t . } \quad \mathbb { E } _ { X } [ \mathrm { K L } ( \pi _ { \theta } ( \cdot | X ) \parallel \pi _ { r e f } ( \cdot | X ) ) ] \leq \epsilon\tag{15}
$$

Concretely, we perform Supervised Fine-Tuning $( \mathrm { S F T } )$ using the standard auto-regressive cross-entropy loss over the token sequence $y = \{ y _ { 1 } , \dots , y _ { T } \}$

$$
\mathcal { L } ( \boldsymbol { \theta } ) = - \sum _ { t = 1 } ^ { T } \log P ( y _ { t } | y _ { < t } , x ; \boldsymbol { \theta } )\tag{16}
$$

where x is the input query and θ denotes the trainable LoRA parameters. This objective trains the model to generate structured CoT reasoning that jointly identifies physical attacks (e.g., printed photos, 3D masks) and digital forgeries (e.g., Deepfake face-swapping), enabling unified face anti-spoofing.

## 3.3.2. LoRA Configuration and Capacity Justification

To achieve parameter-eficient adaptation while preserving the model’s pre-trained generalization capability, we employ Low-Rank Adaptation (LoRA) with a theoretically grounded rank selection[47]. Rather than updating the full parameter set, we freeze the pre-trained weights and inject trainable low-rank decomposition matrices, whose suficiency is justified by the low intrinsic dimension of reasoning tasks and the data-parameter eficiency constraint. The forward pass for a given layer is formulated as:

$$
h = W _ { 0 } x + B A x\tag{17}
$$

where $A \in \mathbb { R } ^ { r \times k }$ and $B \in \mathbb { R } ^ { d \times r }$ with rank $r = 8$ and scaling factor $\alpha = 3 2$ LoRA adapters are applied to all linear layers.

We justify the suficiency of this low-rank configuration from two perspectives. First, recent studies show that fine-tuning pre-trained models on downstream tasks operates within a low-dimensional subspace, where the intrinsic dimension $d _ { i n t } \ll | \theta _ { f u l l } |$ . Since CoT reasoning introduces logical alignment rather than massive new factual knowledge, the required parameter update is sparse in the high-dimensional space, and the low-rank decomposition $\begin{array} { r } { \Delta W \approx \sum _ { i = 1 } ^ { r } \sigma _ { i } u _ { i } v _ { i } ^ { T } } \end{array}$ is suficient to capture the necessary variance. Second, from a statistical learning perspective, full fine-tuning of a 7B model $( \sim \mathrm { ~ 7 ~ } \times \mathrm { ~ 1 0 ~ ^ { 9 } ~ }$ parameters) on $N \ : = \ : 2 0 , 0 0 0$ CoT samples yields an extreme over-parameterization ratio of $\approx 3 . 5 \times 1 0 ^ { 5 }$ parameters per sample, posing severe overfitting and catastrophic forgetting risks. LoRA reduces the trainable parameters to approximately $0 . 1 \% { - } 0 . 5 \%$ of the total, striking a balance between expressiveness and regularization—suficient to model the shift in $P ( y | x , \mathrm { C o T } )$ while preserving the robust representations learned during pretraining.

## 4. Experimental Evaluation

The experimental evaluation is divided into three parts: experimental setup, ablation studies, and comparative experiments, which are detailed in Sections 4.1, 4.2 and 4.3, respectively.

## 4.1. Experimental Setup

## 4.1.1. Evaluation Metrics

This study employs five core metrics to comprehensively evaluate model performance: Attack Presentation Classification Error Rate (APCER) measures security; Bona Fide Presentation Classification Error Rate (BPCER) reflects usability; Average Classification Error Rate (ACER) characterizes the overall discriminative level at a specific threshold; Area Under the ROC Curve (AUC) depicts the global ranking capability across thresholds. Given the inherent trade-of between APCER and BPCER, multiple metrics must be combined to comprehensively assess the balancing capability between security and usability[48].

## 4.1.2. Evaluation Dataset

To thoroughly evaluate the ID generalization ability and OOD transfer robustness, the test set consists of four independent components:

• CelebA-Spoof (ID): CelebA-Spoof is a large-scale face anti-spoofing dataset comprising 625,537 images across 10,177 identities, covering diverse capture devices and scenarios. The spoof samples encompass 11 types of attack methods including print, 3D mask, and replay attacks, and each image is annotated with 43 fine-grained attributes, making it one of the widely adopted benchmarks in this field[41].

• MS-UFAD-DeepFake (ID): MS-UFAD is a large-scale unified face attack detection dataset comprising 795k videos and 60k images, covering 5,000 identities and 52 attack methods across four quality levels.

It provides fine-grained textual descriptions for each sample through semi-automated annotation, making it the first face attack dataset to incorporate textual semantic annotations. We adopt its digital forgery subset for model training and evaluation[23].

• MFFI-Phase-Val (OOD): MFFI is a multi-dimensional face forgery image dataset tailored for real-world scenarios, comprising 1024K image samples and 50 diferent forgery methods. It enhances realism across four strategic dimensions—wider forgery methods, varied facial scenes, diversified authentic data, and multi-level degradation operations—and outperforms existing datasets in scene complexity, cross-domain generalization, and detection dificulty gradients[42]. We exclusively adopt its test set to evaluate the out-of-distribution (OOD) generalization capability of our model.

## 4.1.3. Training Configurations

The fine-tuning configuration strikes a balance between parameter eficiency, context adaptability, and convergence suficiency. We employ a LoRA strategy, which injects low-rank update matrices into all linear layers. This approach significantly reduces memory overhead and the number of trainable parameters while preserving the model’s representation capability for downstream tasks. The sequence length is set to 2048, providing a suficient context window for the MLLMs to accommodate multimodal tokens, thereby efectively preventing information loss caused by truncation. Furthermore, combined with 3 training epochs and a 5% warmup ratio, the configuration ensures the model adequately learns instruction-following patterns on the subset data while mitigating the risks of overfitting and catastrophic forgetting.

## 4.2. Ablation Study

To validate the efectiveness of CoT, we present two pivotal ablation studies. First, we compare our approach against large-scale short-text instruction fine-tuning in Section 4.2.1. Next, we evaluate the robustness of CoT finetuning under white-box attacks relative to short-text instruction fine-tuning in Section 4.2.2. Furthermore, since our evaluation involves both human experts and automated metrics, we present a comparative analysis between human and large model evaluations of CoT in Section 4.2.3. Finally, we provide the tuning results for the LoRA rank, which is a core hyperparameter in the SFT phase in Section 4.2.4.

In the white-box attack experiments, we observe a pronounced trade-of between robustness and generalization, and we attempt to mitigate this issue via reinforcement learning. The generalization capability and adversarial robustness of face anti-spoofing models are of great practical significance, and addressing this challenge constitutes our primary future work.

## 4.2.1. Comparative Experiments between CoT Fine-tuning and Large-scale Short-text Instruction Fine-tuning

To validate the efectiveness of our CoT approach, we first conduct a comparative experiment against short-text fine-tuning. Notably, short-text fine-tuning can be regarded as an approximate upper bound for methods such as MS-UFAD[23][24][25]. The experimental results are illustrated in Figure 4. On in-domain evaluation, CoT data achieves comparable performance to fine-tuning on hundreds of thousands of short-text samples using only a few thousand samples. On the CelebA-Spoof and MS-UFAD-Deepfake datasets, as the scale of short-text training data expands, the model achieves competitive performance that slightly surpasses the CoT fine-tuning baseline. However, this requires a data volume dozens of times larger than that of the CoT approach. This suggests that the short-text model can efectively extract shallow statistical regularities and specific feature shortcuts within the training set. Meanwhile, the CoT model also maintains strong performance on ID data (with AUCs ranging from 0.94 to 0.99). This demonstrates that the introduction of logical constraints by the reasoning mechanism during feature extraction does not significantly compromise absolute metrics. More importantly, it prevents the model from excessively memorizing noise in the training set.

However, when the evaluation shifts to the OOD MFFI dataset, the generalization capabilities of the two architectures exhibit significant heterogeneity. The AUC of the model trained on short texts on MFFI plummets to the range of 0.51 to 0.59, remaining strictly below 0.6 and indicating near-random guessing. In stark contrast, the CoT model achieves an AUC of 0.67, significantly surpassing the short-text fine-tuned data and demonstrating a certain degree of discriminative capability. The near-random-guessing performance of the former confirms its heavy reliance on the specific distribution of the training data. Once confronted with distribution shifts, its memorized feature shortcuts become completely inefective and may even trigger negative transfer. Conversely, the superior performance of the CoT model demonstrates that the logical reasoning mechanism introduced via CoT prompts the model to discard superficial pixel-level features and instead learn underlying semantic logic with greater universality and resistance to interference, thereby preserving basic discriminative capabilities in unknown distributions.

Synthesizing these experimental results, we can conclude that the current learning paradigm of models is undergoing a profound shift from memorization to reasoning. The short model represents traditional intuitive pattern matching, which lacks the robustness to handle the complexities of the open world. The CoT model successfully trades of some ultimate fitting precision on ID tasks for generalization capabilities in OOD scenarios. This finding provides crucial insights for future development of highly robust systems: in real-world applications facing potential distribution shifts, introducing reasoning mechanisms to break the bottleneck of memorization is a key pathway to enhancing a model’s survival and adaptability in unknown environments.

![](images/19e77cbe0b1ac369cec2727c482063956fd712e573896b82f17e09cab3be66ce.jpg)  
(a) ACER trend comparison chart on the Celeba-Spoof dataset.

![](images/9c5f7a83d6c37aceaf976be4f3c4617587185c39a75c113476575f1015359ff4.jpg)  
(b) ACER trend comparison chart on the MS-UFAD-Deepfake dataset.

![](images/bc5d6f398ef0fb56445496142d473644aec4c1dce3da901399ecbe925949ab31.jpg)  
(c) ACER trend comparison chart on the MFFI dataset.

![](images/c08e47606f7e3635c1851ece34e0c6e8fed212ffdd45fe80511b664c566e2ff4.jpg)  
(d) AUC trend comparison chart on the Celeba-Spoof dataset.

![](images/c8e93b1c19f92962f65cb24c716cec0f86b9f2b641401eb7b7f8a625f474a2c1.jpg)  
(e) AUC trend comparison chart on the MS-UFAD-Deepfake dataset.

![](images/58125e3e2ad1a589a6e7bd6259267d2bf289c97a8d006c539267918ebe1dc31e.jpg)  
(f) AUC trend comparison chart on the MFFI dataset.  
Figure 4: Comparative experiments on model performance as a function of training data scale for CoT and short-text datasets.

## 4.2.2. Comparative Robustness Analysis of CoT and Short Strategies against Adversarial Physical and Digital Attacks

To evaluate the robustness of the model, we employ a targeted white-box attack on the pixel space[49]. Given a face image x, the adversarial image $\mathbf { x } ^ { * }$ is generated by iteratively maximizing the probability of a target token y<sub>target</sub> (e.g., class ‘real’). At each iteration step t, the perturbation is updated using the sign of the gradient of the cross-entropy loss $\mathcal { L }$ with respect to the input pixels:

$$
\mathbf { x } _ { t + 1 } = \Pi _ { \mathbf { x } , \epsilon } \left( \mathbf { x } _ { t } + \alpha \cdot \mathrm { s g n } \left( \nabla _ { \mathbf { x } } \mathcal { L } ( \mathbf { x } _ { t } , y _ { t a r g e t } ) \right) \right)\tag{18}
$$

where α is the step size, $\epsilon$ is the maximum allowable perturbation bound, and $\Pi _ { \mathbf { x } , \epsilon } ( \cdot )$ denotes the projection operation that clamps the perturbed pixels to the valid range $\left[ { \bf x } - \epsilon , { \bf x } + \epsilon \right]$ . The final adversarial image $\mathbf { x } ^ { * }$ is obtained after $T$ iterations.

The experimental results are shown in Figure 5. On the CelebA-Spoof dataset, which focuses on physical spoofing (e.g., prints and screen replays), the short-text fine-tuning strategy exhibits severe robustness vulnerabilities, whereas CoT demonstrates exceptional stability. While the short-text finetuning model achieves an extremely high accuracy of 99.00% under no-attack conditions compared to 92.50% for CoT, this suggests that the short-text fine-tuning model may be overfitting to specific physical textures or shallow features. As the intensity of white-box attacks increases, the accuracy of the short-text fine-tuning model plummets from 99.00% to 88.45%, representing an absolute drop of 10.55%. In contrast, the accuracy of the CoT model only marginally decreases from 92.50% to 91.09%, with an absolute drop of merely 1.41%. Under physical attacks, adversarial samples typically disrupt physical spoofing traces (such as moiré patterns and edge artifacts) through subtle pixel perturbations. Due to its excessively short-text finetuning reasoning path, the short-text fine-tuning strategy is easily blinded by such perturbations. Conversely, CoT constructs a redundant verification mechanism through multi-step reasoning (e.g., global lighting analysis and biometric consistency checks), successfully defending against physical gradient attacks.

In the MS-UFAD-Deepfake dataset, which targets digitally generated forgeries, the defensive advantage of CoT is further amplified. The initial accuracy of the short-text fine-tuning model is as high as 99.87%, but it instantly drops to 80.12% under a mild attack $( \varepsilon = 0 . 0 1 5 7 )$ and continues to decline to 79.25% in subsequent attacks, with a maximum drop exceeding 20%. This indicates that short logical chains are highly fragile when confronted with high-frequency synthetic artifacts in the digital domain. The initial accuracy of CoT is 82.25%. After an initial adaptive adjustment, its accuracy remains strictly locked within the 75.00%–75.37% range throughout the entire high-intensity attack interval $( \varepsilon = 0 . 0 6 2 7 \mathrm { ~ t o ~ } \varepsilon = 1 . 0 0 0 0 )$ . Since digital attacks typically inject high-frequency noise directly into the feature space, the long-chain reasoning of CoT acts as a semantic low-pass filter. This mechanism filters out digital adversarial perturbations, maintaining a highly robust performance floor.

![](images/94cb1d8cc227d571af42ffe7ac4ac737dd6ea695fc4b725419a197007f31dffd.jpg)  
Figure 5: Comparison of the base model, the model fine-tuned on a large corpus of shorttext samples, and the model fine-tuned on a small amount of CoT data under white-box attacks.

On the cross-domain MFFI dataset, the accuracy of Base, CoT, and Short models all rapidly plummet to approximately 50%, which is the level of random guessing. This result inversely validates the boundary of efectiveness: the robustness gains of CoT are highly dependent on the model’s existing cognitive representations of target spoofing features. When confronted with completely unknown OOD data, the lack of correct prior knowledge as a reasoning foundation causes multi-step reasoning to lose its factual basis, thereby rendering the defense mechanism inefective.

Overall, under both physical and digital attacks, the accuracy degradation of CoT (1.41% and 7.12%) is substantially lower than that of the short-text fine-tuning strategy (10.55% and 20.62%). CoT not only defends against high-frequency adversarial noise in the digital domain but also efectively resists texture-level adversarial perturbations in the physical domain, demonstrating its immense potential as a universal semantic-level defense

mechanism.

## 4.2.3. Human Evaluation Results

This experiment establishes an automated quality assessment protocol based on the GPT-4.1 MLLM, aiming to audit CoT annotations in FAS datasets. To prevent misjudgments on genuine live samples that lack obvi ous forgery traces, the experiment introduces a dynamic baseline calibration mechanism. When an image is identified as genuine and both logical and knowledge scores are passing, the system forcibly sets a floor of 3 for the D1 score. In terms of data presentation, the experiment ultimately generates a structured quality sampling report in a tabular format, recording detailed information for each sample in rows, including the filename, D1 evidence score, D2 logical score, D3 knowledge score, the comprehensive score calculated via a weighted formula, the final pass/fail verdict, and a summary of critical issues output by the model. Through this evaluation framework, the experiment can accurately identify specific flaws such as incorrect conclusions, hallucinated features, or omission of secondary clues, thereby providing quantitative evidence for the refined iteration of the dataset.

We sampled 100 instances from each of the final generated CoT categories and each attack type for model evaluation. The experimental results are presented in Table 1. The human expert CoT evaluation achieves the highest weighted score of 4.7619, followed by GPT-4.1 with 4.7046, which outperforms the human expert on D2 (4.9043) and D3 (4.7186). Qwen-3.6 obtains the lowest scores across all metrics, with a weighted score of only 4.2536. Even when a CoT reasoning chain contains minor flaws, human experts are able to recognize the overall soundness of the reasoning, whereas models tend to perform strict step-by-step verification and are prone to penalizing the overall score due to local errors. During evaluation, large language models may place greater emphasis on surface-level features such as structural completeness and logical coherence of the CoT, resulting in harsher scoring for reasoning chains that deviate from their ideal templates. Human experts can tolerate certain non-standard yet efective reasoning paths based on their experience, while models lack such flexible judgment capability.

## 4.2.4. LoRA Configuration Experiments

Figure 6 illustrates the impact of diferent LoRA rank settings on the APCER across various datasets. On the CelebA-Spoof and MS-UFAD-Deepfake datasets, the APCER remains consistently low and exhibits minimal fluctuation across varying LoRA ranks. In contrast, the APCER on the OOD dataset, MFFI, is significantly higher than that of the ID data, fluctuating between 60% and 70%. Notably, as the rank increases from 8 to 32, the APCER rises, peaking at a rank of 32. This suggests that with increased LoRA capacity, the model may overfit the training set, learning features that are overly specific to the source domain, thereby compromising its generalization capability on MFFI. However, as the rank further increases to 64, the APCER experiences a slight decline. Under this larger capacity, the model may have suficient space to learn more generalized or robust features, or this improvement might simply be attributed to a saturation efect of overfitting.

Table 1: Comparison of Evaluation Scores across Diferent Models and Human Experts
<table><tr><td>Evaluator</td><td></td><td></td><td></td><td>D1 Avg. D2 Avg. D3 Avg. Weighted Score</td></tr><tr><td>Human Expert</td><td>4.8024</td><td>4.7917</td><td>4.6476</td><td>4.7619</td></tr><tr><td>GPT-4.1</td><td>4.5607</td><td>4.9043</td><td>4.7186</td><td>4.7046</td></tr><tr><td>Qwen-3.6</td><td>4.0822</td><td>4.3987</td><td>4.3621</td><td>4.2536</td></tr></table>

For OOD generalization, a rank of 8 appears to be a relatively optimal choice, despite the high absolute error rate. This corroborates the hypothesis that under severe domain shifts, restricting model capacity can help prevent overfitting to source domain noise, thereby preserving a degree of generalizability. While LoRA fine-tuning is highly efective on the source domain, easily achieving low error rates, OOD performance proves to be highly sensitive to the LoRA rank, unlike the stability observed in ID tasks. Ultimately, our empirical results demonstrate that blindly increasing the LoRA rank does not enhance cross-domain generalization. Instead, it may further degrade generalization performance due to overfitting to source domain features. In this specific cross-domain setting, a smaller rank demonstrates relatively superior robustness.

## 4.3. Comparative Experiments

## 4.3.1. Comparative Experimental Setup

The field of face anti-spoofing is currently in a transitional phase where CNNs, CLIP, and MLLMs coexist. Many state-of-the-art (SOTA) methods are built upon traditional CNNs or foundational vision-language models (e.g., CLIP). However, due to the proprietary nature of their training data and undisclosed source codes, exact reproduction of these specific variants is unfeasible. Furthermore, these incremental improvements typically do not break through the inherent performance ceilings of their respective backbone architectures. Therefore, rather than relying on unreproducible specific implementations, we employ ResNet-101 and CLIP-ViT-B as the representative upper bounds for their respective methodological categories. This strategy ensures that our comparative evaluation remains both rigorous and reproducible. In our comparative experiments, we evaluate four categories of baseline models. In terms of general capability, they are ranked from strongest to weakest as follows:

![](images/50e39cf7410cb8930afdd1e573a10a055806254dc4d3310d26e66e6007928db4.jpg)  
Figure 6: Performance analysis of LoRA configurations across physical, digital, and crossdomain anti-spoofing tasks.

1. Traditional CNNs: Represented by ResNet-50 and ResNet-101. These methods serve as a representative baseline for conventional deep learning approaches[1][2][3][4][11][38].

2. Foundation Multimodal Models: Specifically, CLIP-ViT-B/32. This model represents the general scope of prior multimodal methods[22][21][27].

3. Open-source Large Models: Including Qwen3-235B-A22B-Thinking-2507, Qwen3-VL-32B-Instruct, Qwen3-VL-8B-Thinking, Qwen3-VL-3B-Instruct, and Qwen3-VL-7B-Instruct[44].

4. Proprietary Large Models: Including GPT-4o, Gemini-2.5-Flash-Image, and Claude-Opus-4.6.

Next, we first analyze the necessity of employing multimodal large models, and subsequently conduct a comprehensive comparison with various baseline models. The necessity analysis primarily involves a detailed comparison and comprehensive analysis against traditional CNNs.

## 4.3.2. Necessity of Adopting MLLMs for the UFAD Task

Compared to traditional CNNs, which rely solely on shallow feature matching of low-level pixels and frequency-domain artifacts, MLLMs demonstrate a generational advantage in joint face anti-spoofing. MLLMs overcome the detection bottlenecks of unimodal models under compression and noise interference. By leveraging cross-modal attention mechanisms, they precisely capture subtle logical contradictions in audio-visual semantics. Furthermore, empowered by massive pre-trained knowledge, they achieve superior generalization against unseen spoofing methods. Most importantly, MLLMs abandon the black-box paradigm of traditional methods, providing interpretable forensic evidence in natural language, thereby realizing a true paradigm shift from mere texture recognition” to cognitive understanding. To this end, we conduct a comparative evaluation against CNN methods across varying orders of magnitude of training samples. The experimental results are illustrated in Figure 7.

As shown in Figure 7, traditional CNN methods, represented by ResNet-50 and ResNet-101, exhibit significant limitations in face anti-spoofing tasks. Although they achieve remarkably low error rates on specific datasets (with an ACER as low as 0.0756), their performance degrades drastically in crossdomain scenarios (e.g., MFFI), where the ACER soars to over 0.43. This indicates a severe tendency toward overfitting and inadequate generalization capabilities. In contrast, the MFAD series of specialized lightweight antispoofing models (e.g., MFAD-7B), through targeted architectural design, not only maintains an exceptionally low error rate in-domain (ACER of 0.0357 on CelebA-Spoof) but also significantly outperforms generic ResNet models in cross-domain testing. This demonstrates that dedicated architectural designs tailored for anti-spoofing tasks are substantially more efective than merely increasing the depth of CNNs.

![](images/59d66fb10aa51621200f0cfc883a4a068693370695b802b7aeb959348823f5b5.jpg)  
Figure 7: Performance analysis of LoRA configurations across physical, digital, and crossdomain anti-spoofing tasks.

## 4.3.3. Comparison with All Model Types

As a general-purpose vision-language pre-trained model, the CLIP-ViTbase series exhibits a distinct characteristic of strong in-domain performance but weak cross-domain generalization in face anti-spoofing tasks. On the MS-UFAD-Deepfake dataset, CLIP demonstrates remarkable artifact-capturing capabilities (with the patch32 variant achieving an ACER of merely 0.0084), even outperforming certain large-scale multimodal models. However, it yields mediocre results on CelebA-Spoof, and its performance degrades drastically in the cross-domain MFFI test, where the ACER soars to approximately 0.48, approaching the level of random guessing. This indicates that although CLIP has learned rich semantic representations, it lacks sensitivity to finegrained anti-spoofing textures. Furthermore, the inherent gap between its general pre-training objectives and the specific task of spoof detection leads to significantly inferior generalization capabilities against unseen domains compared to specialized models.

On the ID datasets, general-purpose models without targeted training exhibit clear limitations, whereas the specifically trained MFSA model demonstrates a superior performance ceiling. Despite possessing massive parameter counts, general large models such as Qwen3-VL-32B-Instruct and Gemini-2.5-Flash-Image generally yield ACERs higher than 0.15, reaching up to 0.36 on CelebA-Spoof and MS-UFAD-Deepfake. In contrast, MFSA-7B reduces the ACER to 0.0357 and 0.0101, respectively. This proves that relying solely on the pre-trained knowledge of general models is insuficient to efectively capture the subtle artifacts of deepfakes. Thus, targeted supervised training is essential to inject domain-specific knowledge. Although ResNet-50/101 achieve low APCER on MS-UFAD, they sufer from high BPCER (approximately 0.19), indicating a dificulty in balancing the classification of genuine and spoof samples. Meanwhile, MFAD-7B maintains extremely low APCER (0.0715 / 0.0167) while achieving 0 BPCER on CelebA-Spoof and 0.0035 BPCER on MS-UFAD. This indicates that our training strategy not only improves detection rates but also significantly optimizes the decision boundary, thereby avoiding false rejections of genuine samples.

The experimental results clearly illustrate the performance gains of the MFAD series models with increasing parameter scales, validating the efectiveness of our proposed method. From MFAD-3B to MFAD-7B, all metrics show significant improvement. Specifically, on MS-UFAD-Deepfake, the ACER is further compressed from 0.0186 to 0.0101. This consistent performance enhancement suggests that our training data and loss function design are of high quality and can be efectively leveraged by larger-scale models, providing a solid empirical foundation for future model scaling. Although MFAD-7B exhibits a higher ACER (0.359) on the out-of-domain dataset MFFI compared to some general models, this precisely reflects the fundamental diference in design objectives between specialized and general models, rather than a defect. To achieve an extremely low false rejection rate (BPCER ≈ 0) on in-domain data, the MFAD model learns a highly finegrained ID feature distribution. This high specificity is the cause of its performance degradation on the unseen MFFI domain. In practical application scenarios (such as financial identity verification and social media moderation), the threats faced are typically specific types of attacks. In such cases, clients prioritize zero false rejections and high detection rates for known attacks. The overwhelming advantage demonstrated by the MFAD model on ID data is exactly what is urgently needed for real-world deployment. While general models may perform adequately on OOD data, their error rates of 20–30% on ID data are unacceptable in industrial applications.

In conclusion, training the MFAD model is not only necessary but also highly eficient. It resolves the issue of general large models understanding common sense but lacking anti-spoofing expertise in forgery detection tasks, while simultaneously overcoming the shortcomings of traditional CNNs, which sufer from high false rejection rates despite decent detection capabilities. Despite a performance drop in extreme OOD scenarios, its exceptionally high accuracy and ultra-low false rejection rates in core business scenarios (indomain) firmly establish its core value as a specialized solution for forgery detection.

Table 2: Performance comparison of diferent models across CelebA-Spoof, MS-UFAD-Deepfake, and MFFI datasets.
<table><tr><td></td><td colspan="3">CelebA-Spoof</td><td colspan="3">MS-UFAD-Deepfake</td><td colspan="3">MFFI</td></tr><tr><td>Model</td><td>APCER</td><td>BPCER</td><td>ACER</td><td>APCER</td><td>BPCER</td><td></td><td>ACER APCER</td><td>BPCER ACER</td><td></td></tr><tr><td>ResNet-50</td><td>0.1885</td><td>0.1475</td><td>0.1680</td><td>0.0040</td><td>0.1475</td><td>0.0756</td><td>0.2382</td><td>0.6293</td><td>0.4338</td></tr><tr><td>ResNet-101</td><td>0.1505</td><td>0.1975</td><td>0.1740</td><td>0.0000</td><td>0.1975</td><td>0.0988</td><td>0.1735</td><td>0.7375</td><td>0.4555</td></tr><tr><td>CLIP-ViT-base-patch16</td><td>0.1191</td><td>0.1501</td><td>0.1346</td><td>0.0247</td><td>0.0081</td><td>0.0177</td><td>0.4764</td><td>0.5139</td><td>0.4951</td></tr><tr><td>CLIP-ViT-base-patch32</td><td>0.1001</td><td>0.3055</td><td>0.2028</td><td>0.0107</td><td>0.0061</td><td>0.0084</td><td>0.633</td><td>0.3242</td><td>0.4786</td></tr><tr><td>Qwen3-VL-8B-Thinking</td><td>0.8032</td><td>0.0175</td><td>0.4104</td><td>0.7206</td><td>0.0175</td><td>0.3690</td><td>0.7147</td><td>0.0309</td><td>0.3728</td></tr><tr><td>Qwen3-VL-32B-Instruct</td><td>0.4482</td><td>0.0425</td><td>0.2454</td><td>0.3288</td><td>0.0425</td><td>0.1856</td><td>0.5529</td><td>0.0385</td><td>0.2958</td></tr><tr><td>Qwen3-235B-A22B-Thinking-2507</td><td>0.7207</td><td>0.5263</td><td>0.6235</td><td>0.8575</td><td>0.5263</td><td>0.6919</td><td>0.4701</td><td>0.5676</td><td>0.5193</td></tr><tr><td>Qwen3-VL-3B-Instruct</td><td>0.4029</td><td>0.4369</td><td>0.4199</td><td>0.5427</td><td>0.4375</td><td>0.4901</td><td>0.6756</td><td>0.2529</td><td>0.4643</td></tr><tr><td>Qwen3-VL-7B-Instruct</td><td>0.5627</td><td>0.0673</td><td>0.3150</td><td>0.9127</td><td>0.0244</td><td>0.4686</td><td>0.7448</td><td>0.0778</td><td>0.4113</td></tr><tr><td>GPT-4o</td><td>0.0701</td><td>0.2519</td><td>0.1610</td><td>0.0569</td><td>0.2519</td><td>0.1544</td><td>0.0152</td><td>0.9571</td><td>0.4861</td></tr><tr><td>Gemini-2.5-Flash-Image</td><td>0.5400</td><td>0.0000</td><td>0.2700</td><td>0.6233</td><td>0.0152</td><td>0.3192</td><td>0.1512</td><td>0.2604</td><td>0.2058</td></tr><tr><td>Claude-Opus-4.6</td><td>0.3045</td><td>0.0125</td><td>0.1585</td><td>0.8427</td><td>0.0125</td><td>0.4276</td><td>0.3441</td><td>0.1815</td><td>0.2628</td></tr><tr><td>MFAS-3B</td><td>0.1541</td><td>0.0000</td><td>0.0770</td><td>0.0340</td><td>0.0031</td><td>0.0186</td><td>0.5540</td><td>0.1422</td><td>0.3481</td></tr><tr><td>MFAS-7B</td><td>0.0715</td><td>0.0000</td><td>0.0357</td><td>0.0167</td><td>0.0035</td><td>0.0101</td><td>0.6176</td><td>0.1004</td><td>0.3590</td></tr></table>

## 5. Conclusion and Future Work

In this paper, we have presented MFAD, a unified and interpretable face anti-spoofing system designed to address the critical challenges of generalization and interpretability in real-world scenarios. By leveraging the rich semantic priors of large vision-language models, our framework moves beyond traditional binary classification. The core of our approach lies in the Semantic-Level Evidence Anchoring mechanism, which grounds the decisionmaking process in human-understandable visual cues. Through the generation of CoT reasoning, MFAD not only achieves robust performance across various unseen domains but also provides transparent and trustworthy justifications for its predictions. Extensive experiments on standard benchmarks have demonstrated that our method significantly outperforms state-of-the-art approaches, establishing a new paradigm for developing reliable and interpretable biometric security systems.

We can observe that, on ID data, both robustness and accuracy have already reached a level suitable for practical deployment. However, performance on out-of-domain data remains insuficient: (1) the accuracy of OOD generalization still falls considerably short of the requirements for real-world applications; (2) there exists a pronounced trade-of between robustness and generalization. Addressing these challenges constitutes the focus of our future work.

## References

[1] Oleksandr Kuznetsov, Dmytro Zakharov, Emanuele Frontoni, and Andrea Maranesi. Attacknet: Enhancing biometric security via tailored convolutional neural network architectures for liveness detection. Computers & Security, 141:103828, 2024.

[2] Raghavendra Raghuram Jingade and Rajaram Sanjeev Kunte. Extended right-angle diference ternary co-relation pattern: A new feature descriptor for face anti-spoofing. Computers & Security, 134:103421, 2023.

[3] Yaojie Liu, Amin Jourabloo, and Xiaoming Liu. Learning deep models for face anti-spoofing: Binary or auxiliary supervision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 389–398, 2018.

[4] Andreas Rossler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, and Matthias Nießner. Faceforensics++: Learning to detect manipulated facial images. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1–11, 2019.

[5] Yuezun Li, Xin Yang, Pu Sun, Honggang Qi, and Siwei Lyu. Celeb-df: A large-scale challenging dataset for deepfake forensics. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3207–3216, 2020.

[6] Brian Dolhansky, Russ Howes, Ben Pflaum, Nicole Baram, and Cristian Canton Ferrer. The deepfake detection challenge (dfdc) preview dataset. arXiv preprint arXiv:1910.08854, 2019.

[7] Nesli Erdogmus and Sébastien Marcel. Spoofing in 2d face recognition with 3d masks and anti-spoofing with kinect. In 2013 IEEE sixth international conference on biometrics: theory, applications and systems (BTAS), pages 1–6. IEEE, 2013.

[8] Ajian Liu, Chenxu Zhao, Zitong Yu, Anyang Su, Xing Liu, Zijian Kong, Jun Wan, Sergio Escalera, Hugo Jair Escalante, Zhen Lei, et al. 3d highfidelity mask face presentation attack detection challenge. In Proceedings of the IEEE/CVF international conference on computer vision, pages 814–823, 2021.

[9] Lei Li, Zhaoqiang Xia, Abdenour Hadid, Xiaoyue Jiang, Haixi Zhang, and Xiaoyi Feng. Replayed video attack detection based on motion blur analysis. IEEE Transactions on Information Forensics and Security, 14 (9):2246–2261, 2019.

[10] Jukka Määttä, Abdenour Hadid, and Matti Pietikäinen. Face spoofing detection from single images using micro-texture analysis. In 2011 international joint conference on Biometrics (IJCB), pages 1–7. IEEE, 2011.

[11] Bofan Lin, Xiaobai Li, Zitong Yu, and Guoying Zhao. Face liveness detection by rppg features and contextual patch-based cnn. In Proceedings of the 2019 3rd international conference on biometric engineering and applications, pages 61–68, 2019.

[12] Yousef Atoum, Yaojie Liu, Amin Jourabloo, and Xiaoming Liu. Face anti-spoofing using patch and depth-based cnns. In 2017 IEEE international joint conference on biometrics (IJCB), pages 319–328. IEEE, 2017.

[13] Wei Zheng, Mengyuan Yue, Shuhuan Zhao, and Shuaiqi Liu. Attentionbased spatial-temporal multi-scale network for face anti-spoofing. IEEE Transactions on Biometrics, Behavior, and Identity Science, 3(3):296– 307, 2021.

[14] Zitong Yu, Chenxu Zhao, Zezheng Wang, Yunxiao Qin, Zhuo Su, Xi aobai Li, Feng Zhou, and Guoying Zhao. Searching central diference convolutional networks for face anti-spoofing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5295–5305, 2020.

[15] Mayank Pathak, Kamta Nath Mishra, Satya Prakash Singh, and Alok Mishra. A hybrid machine learning and cryptography-based predictive

probability model for enhancing security and privacy in cloud-iot environment. Computers & Security, page 104870, 2026.

[16] Xin Sun, Chengliang Tian, Changhui Hu, Weizhong Tian, Hanlin Zhang, and Jia Yu. Privacy-preserving and verifiable src-based face recognition with cloud/edge server assistance. Computers & Security, 118:102740, 2022.

[17] Hui Zhang, Ying Zhou, Xingbo Dong, Qingguo Meng, and Zhe Jin. Triplet-bio: A secure cloud-edge collaborative biometric authentication via two-factor secret sharing. In Chinese Conference on Biometric Recognition, pages 145–158. Springer, 2024.

[18] Hao Yu, Haoyu Chen, and Guoying Zhao. Learning binary-antithetical information bottleneck for generalizable face anti-spoofing. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2025.

[19] Xianhua He, Dashuang Liang, Song Yang, Zhanlong Hao, Hui Ma, Binjie Mao, Xi Li, Yao Wang, Pengfei Yan, and Ajian Liu. Joint physicaldigital facial attack detection via simulating spoofing clues. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 995–1004, 2024.

[20] Ajian Liu. Ca-moeit: Generalizable face anti-spoofing via dual crossattention and semi-fixed mixture-of-expert. International Journal of Computer Vision, 132(11):5439–5452, 2024.

[21] Xun Lin, Ajian Liu, Zitong Yu, Rizhao Cai, Shuai Wang, Yi Yu, Jun Wan, Zhen Lei, Xiaochun Cao, and Alex Kot. Reliable and balanced transfer learning for generalized multimodal face anti-spoofing. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[22] Kartik Narayan, Vibashan VS, Rama Chellappa, and Vishal M Patel. Facexformer: A unified transformer for facial analysis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11369–11382, 2025.

[23] Ning Jiang, Dingheng Zeng, Liang Gao, Sheng Chen, Zhifei Kong, Yanhong Liu, Jiao Li, Yue Feng, Tongtong Yuan, Weihong Deng, et al.

Ms-ufad: A large-scale dataset for real-world unified face attack detection with text descriptions. In ICASSP 2025-2025 IEEE International Conference on Acoustics, page 2151, 2025.

[24] Hao Fang, Ajian Liu, Haocheng Yuan, Junze Zheng, Dingheng Zeng, Yanhong Liu, Jiankang Deng, Sergio Escalera, Xiaoming Liu, Jun Wan, et al. Unified physical-digital face attack detection. arXiv preprint arXiv:2401.17699, 2024.

[25] Junze Zheng, Xinping Gao, Ajian Liu, Haocheng Yuan, Jun Wan, Yanyan Liang, Jiankang Deng, Sergio Escalera, Hugo Jair Escalante, Zhen Lei, et al. Unified physical–digital face attack detection challenge: A review. IET Biometrics, 2026(1):9653627, 2026.

[26] Haoyuan Zhang, Keyao Wang, Guosheng Zhang, Haixiao Yue, Zhiwen Tan, Siran Peng, Tianshuo Zhang, Xiao Tan, Kunbin Chen, Wei He, et al. From intuition to investigation: A tool-augmented reasoning mllm framework for generalizable face anti-spoofing. arXiv preprint arXiv:2603.01038, 2026.

[27] Ke Sun, Shen Chen, Taiping Yao, Ziyin Zhou, Jiayi Ji, Xiaoshuai Sun, Chia-Wen Lin, and Rongrong Ji. Towards general visual-linguistic face forgery detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19576–19586, 2025.

[28] Hongyang Wang, Yichen Shi, Zhuofu Tao, Yuhao Gao, Liepiao Zhang, Xun Lin, Jun Feng, Xiaochen Yuan, Zitong Yu, and Xiaochun Cao. Faceshield: Explainable face anti-spoofing with multimodal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 9811–9819, 2026.

[29] Koushik Srivatsan, Muzammal Naseer, and Karthik Nandakumar. Flip: Cross-domain face anti-spoofing with language guidance. In Proceedings of the IEEE/CVF international conference on computer vision, pages 19685–19696, 2023.

[30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural

language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[31] Hao Fang, Ajian Liu, Ning Jiang, Quan Lu, Guoqing Zhao, and Jun Wan. Vl-fas: Domain generalization via vision-language model for face anti-spoofing. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 4770–4774, 2024.

[32] Siran Peng, Zipei Wang, Li Gao, Xiangyu Zhu, Tianshuo Zhang, Ajian Liu, Haoyuan Zhang, and Zhen Lei. Mllm-enhanced face forgery detection: A vision-language fusion solution. arXiv preprint arXiv:2505.02013, 2025.

[33] Ke-Yue Zhang, Ruoxin Chen, Jiamu Sun, Jiangming Wang, Hao Yang, Taiping Yao, and Shouhong Ding. A generalizable face security detection method via unified texture and semantic feature framework. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3190–3198, 2025.

[34] Hang Zou, Chenxi Du, Hui Zhang, Yuan Zhang, Ajian Liu, Jun Wan, and Zhen Lei. La-softmoe clip for unified physical-digital face attack detection. In 2024 IEEE International Joint Conference on Biometrics (IJCB), pages 1–11, 2024.

[35] Jiawei Chen, Xiao Yang, Yinpeng Dong, Hang Su, Jianteng Peng, and Zhaoxia Yin. Facecat: Enhancing face recognition security with a unified generative model framework. arXiv preprint arXiv:2404.09193, 2024.

[36] Jiaruo Yu, Dagong Lu, Xingyue Shi, Chenfan Qu, and Fengjun Guo. Unified face attack detection with micro disturbance and a two-stage training strategy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 960–969, 2024.

[37] Minzhe Huang, Changwei Nie, and Weihong Zhong. A visualization method for data domain changes in cnn networks and the optimization method for selecting thresholds in classification tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 986–994, 2024.

[38] Debayan Deb, Xiaoming Liu, and Anil K Jain. Unified detection of digital and physical face attacks. In 2023 IEEE 17th International Conference on Automatic Face and Gesture Recognition (FG), pages 1–8. IEEE, 2023.

[39] Shifeng Zhang, Ajian Liu, Jun Wan, Yanyan Liang, Guodong Guo, Sergio Escalera, Hugo Jair Escalante, and Stan Z Li. Casia-surf: A largescale multi-modal benchmark for face anti-spoofing. IEEE Transactions on Biometrics, Behavior, and Identity Science, 2(2):182–193, 2020.

[40] Anjith George, Zohreh Mostaani, David Geissenbuhler, Olegs Nikisins, André Anjos, and Sébastien Marcel. Biometric face presentation attack detection with multi-channel convolutional neural network. IEEE transactions on information forensics and security, 15:42–55, 2019.

[41] Yuanhan Zhang, ZhenFei Yin, Yidong Li, Guojun Yin, Junjie Yan, Jing Shao, and Ziwei Liu. Celeba-spoof: Large-scale face anti-spoofing dataset with rich annotations. In European conference on computer vision, pages 70–85. Springer, 2020.

[42] Changtao Miao, Yi Zhang, Man Luo, Weiwei Feng, Kaiyuan Zheng, Qi Chu, Tao Gong, Jianshu Li, Yunfeng Diao, Wei Zhou, et al. Mfi: Multi-dimensional face forgery image dataset for real-world scenarios. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 13235–13242, 2025.

[43] Clayton Sanford, Daniel J Hsu, and Matus Telgarsky. Representational strengths and limitations of transformers. Advances in neural information processing systems, 36:36677–36707, 2023.

[44] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.

[45] Zhengchao Huang, Bin Xia, Zicheng Lin, Zhun Mou, Wenming Yang, and Jiaya Jia. Ffaa: Multimodal large language model based explainable open-world face forgery analysis assistant. arXiv preprint arXiv:2408.10072, 2024.

[46] Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

[47] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzh Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

[48] Zitong Yu, Guoying Zhao, Zezheng Wang, Jan Stehouwer, and Xuan Liu. Face anti-spoofing: A survey. International Journal of Computer Vision, 131:1247–1273, 2023.

[49] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations (ICLR), 2018.

[50] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[51] Zheng Chujie. Group sequence policy optimization. arXiv preprint, 2025.

## 6. Supplementary Material

## 6.1. Evaluation Guidelines for Human Experts, $G P T \mathrm { - } 4 o ,$ and Qwen

You are a senior FAS forensic analyst. Your task is to evaluate the reasoning quality of the CoT text relative to the original image. Core Principle: The image is the sole ground truth. FAS datasets consist of two categories of samples: live and spoof. Mandatory Evaluation Protocol

1. Independent Visual Analysis: Evaluate the image without referring to the text. Determine whether it is live and spoof, and list the key visual features.

2. Fact-Checking: Compare the CoT content against your visual analysis point by point.

3. Dimension Decoupling: Dimensions D1 (Factual), D2 (Logical), and D3 (Knowledge) must be scored independently.

## Evaluation Dimensions

## D1: Evidence Anchoring Accuracy (Weight: 30%)

• 5 Points: Precisely describes spoof artifacts or genuine features, with a completely correct conclusion.

• 4 Points: Core qualitative assessment is correct, with 1–2 minor descriptions being vague.

• 3 Points: Core qualitative assessment is correct, but minor clues are omitted or slight descriptive deviations exist.

• 1 Point: Core conclusion is incorrect, or non-existent features are hallucinated.

## D2: Reasoning Logical Coherence (Weight: 30%)

• 5 Points: The reasoning chain is complete and closed-loop; every deduction is supported by prior observations.

• 3 Points: Conclusion is correct and the overall logic is sound, but local logical jumps exist.

• 1 Point: Logic is chaotic, circular reasoning is used, or the conclusion contradicts the observations.

## D3: Domain Knowledge Accuracy (Weight: 25%)

• 5 Points: Terminology is precise, and the judgment aligns with standard FAS literature.

• 3 Points: Judgment direction is correct, but terminology is nonstandard.

• 1 Point: Pseudo-scientific concepts are fabricated.

## 6.2. Human Modification Guidelines

## 1. Error Diagnosis

Before initiating modifications, human experts must first accurately pinpoint the root cause of any breakdown or deviation within the Chain-of-Thought (CoT). Errors are typically categorized into three types:

• Fact Error: The model observes incorrect visual features or hallucinates non-existent details (e.g., in FAS forensics, misinterpreting shadows in a genuine photo as mask edges).

• Logic Error: The model possesses the correct knowledge but exhibits causal inversion, reasoning leaps, or circular reasoning during the deduction process.

• Knowledge Error: The model lacks specific domain expertise, leading to the application of incorrect evaluation criteria or tools.

## 2. Direct Rewriting and Editing

This is the most straightforward rectification strategy. Human annotators preserve the valid initial observation steps in the CoT and commence targeted revision from the point of failure:

• Supplementing Intermediate Reasoning Steps: When the model jumps directly from facial texture observation to a liveness verdict, human experts insert the missing reasoning bridge (e.g., from specular reflection patterns to material-level analysis), thereby eliminating inferential gaps.

• Correcting Reasoning Paths: When the model misattributes natural facial shading to spoofing artifacts, human experts redirect the reasoning to the correct analytical path and explicitly articulate the self-correction process.

## 3. Counterfactual Correction and Self-Reflection

A high-quality CoT should not only demonstrate how the correct conclusion is reached, but also explicate why alternative hypotheses are invalidated:

• Falsification Logic Injection: Human annotators incorporate counterfactual phrasing such as: “Although moiré patterns were initially detected in the cheek region, frequency-domain verification ruled out a screen-replay attack.” This instructs the model to perform cross-validation when confronting ambiguous facial artifacts.

## 6.3. Exploration in Reinforcement Learning

Having achieved promising results through SFT, we further employed the Group Relative Policy Optimization (GRPO)[50] algorithm to explore strategies for enhancing the OOD generalization capabilities. The experimental results are presented in Table 3. The experimental results demonstrate that SFT is the decisive step in activating the face anti-spoofing capabilities of the Qwen3-VL-3B-Instruct model. During the SFT phase, ACER on in-domain datasets (CelebA-Spoof and MS-UFAD-Spoof) achieved a precipitous drop to 0.077 and 0.0186, respectively. Significant improvements were also observed on the out-of-domain dataset (MFFI), with the ACER decreasing to 0.3481, proving that SFT successfully endowed the model with foundational antispoofing representation capabilities. However, upon introducing the GRPO reinforcement learning phase, model performance failed to continue improving with the increase in sample size (from 200 to 60,000). Instead, severe performance degradation and catastrophic forgetting occurred. On the ID test sets, the ACER instantly regressed to a high plateau of approximately 0.42 and stagnated; on the out-of-domain MFFI test set, performance similarly deteriorated and showed a continuous worsening trend as the RL steps increased. This phenomenon indicates that despite the evaluation metrics in the RL phase being perfectly aligned with the test set, the GRPO algorithm under the current configuration not only failed to provide any gains but also severely disrupted the decision boundaries established by SFT.

To address the performance collapse and catastrophic forgetting caused by GRPO, while simultaneously enhancing the generalization capabilities, interventions are recommended from algorithmic, data, and training strategy perspectives. First, regarding algorithmic stability, token-level importance sampling of GRPO in long-sequence generation often leads to high variance accumulation. We suggest introducing the Group Sequence Policy Optimization (GSPO) algorithm[51], which shifts importance sampling to the sequence level and incorporates length normalization to fundamentally mitigate gradient instability. Second, to improve OOD generalization, it is essential to enrich the exploration space. We recommend employing diverse data augmentation techniques (e.g., geometric transformations, photometric distortion, or noise injection) during the RL sampling phase. This exposes the model to a broader distribution of spoofing artifacts and environmental variations, preventing it from overfitting to specific in-domain features.Third, to prevent the model from generating homogeneous samples within a group—which leads to a zero-variance gradient vanishing trap—we must enforce sampling diversity. This can be achieved by increasing the sampling temperature (e.g., to 0.6–1.0) and relaxing Top-P thresholds.Finally, to ensure RL gradient updates do not destroy SFT knowledge, we recommend mixing a proportion of original SFT data as a regularization constraint during GRPO training, or adding a distribution alignment step between SFT and RL to ensure the model conducts reinforcement exploration within a stable feature space.

Table 3: Reinforcement Learning Experiments on MFAD-3B Fine-tuned from Qwen3-VL-3B-Instruct via SFT
<table><tr><td>Dataset</td><td>Metric</td><td>Base SFT</td><td>200</td><td>500 1k</td><td>10k</td><td>20k 40k</td></tr><tr><td></td><td></td><td></td><td>APCER 0.4029 0.1541 0.6208 0.6234 0.6259 0.6229 0.6248 0.623</td><td></td><td></td><td></td></tr><tr><td>CelebA-Spoof</td><td>BPCER 0.4369 0.0000 0.2179 0.21890.2271 0.2170 0.2184 0.217</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>ACER</td><td></td><td>0.41990.0770 0.41940.42110.42650.41990.42160.420</td><td></td><td></td><td></td></tr><tr><td></td><td>APCER 0.5427 0.0340 0.74530.74340.7422 0.7467 0.74430.743</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>MS-UFAD-Spoof BPCER 0.4375 0.0031 0.2224 0.2212 0.2191 0.2206 0.2228 0.222</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>ACER</td><td></td><td>0.4901 0.01860.48390.48230.48060.48360.48360.482</td><td></td><td></td><td></td></tr><tr><td></td><td>APCER 0.6756 0.5540 0.6567 0.6527 0.6580 0.6585 0.6526 0.653</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MFFI</td><td>BPCER 0.2529 0.1422 0.2907 0.2900 0.2876 0.2895 0.28980.290</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>ACER0.46430.34810.4737 0.4714 0.47280.4740 0.4712 0.471</td><td></td><td></td><td></td><td></td><td></td></tr></table>