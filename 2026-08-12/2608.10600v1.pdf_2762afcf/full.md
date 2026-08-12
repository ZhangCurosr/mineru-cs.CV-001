# BooST: Bridging Semantics and Motions for Efficient Skill Transfer

Jusuk Lee<sup>1</sup>, Daesol Cho<sup>2</sup>, Jonghun Shin<sup>1</sup>, Seungyeon Yoo<sup>1</sup>, Jonghae Park<sup>1</sup>, Taekbeom Lee<sup>1</sup>, and H. Jin Kim<sup>1</sup>

https://boost-robots.github.io

Abstract—Skill abstraction—the process of learning reusable and temporally extended behaviors—has emerged as a key paradigm for improving sample efficiency and generalization in robot learning. For efficient skill transfer to real robots, learned skills must generalize across tasks and domains, remain robust to visual and dynamic perturbations, and be efficient enough for practical deployment. However, existing methods typically satisfy only a subset of these properties, as they capture either high-level semantic intent (what) or low-level motion dynamics (how). This incomplete skill transfer yields weak priors for policy learning, thereby demanding substantial in-domain data for downstream adaptation. To address these challenges, we introduce BooST, a two-stage framework that explicitly bridges semantics and motions to satisfy all three desiderata. BooST first leverages a cross-modal VQ-VAE to capture both semantic intent and motion dynamics, yielding a unified skill representation. It then distills this representation into a lightweight policy for efficient downstream adaptation to new tasks. Extensive experiments across simulation and real-robot settings demonstrate that BooST achieves superior few-shot adaptation, cross-domain skill transfer, and robustness to dynamic visual distractors, while maintaining a lightweight yet expressive design suitable for realworld deployment.

Index Terms—Representation Learning, Imitation Learning, Deep Learning in Grasping and Manipulation

## I. INTRODUCTION

CENTRAL objective in robot learning is to develop general-purpose agents capable of rapidly adapting to novel tasks in a zero-shot or few-shot manner. One promising direction toward this goal is large-scale pretraining of reusable robotic representations, inspired by the success of foundation models in computer vision and natural language processing [1], [2]. In this context, skill abstraction has emerged as an effective paradigm for learning such robotic representations. By offering compact and reusable representations of temporally extended behaviors that capture task-agnostic structures, skills serve as a versatile prior that enables efficient adaptation in the low-data regime of robot learning [3], [4]. Therefore, transferring skills learned from large-scale pretraining enables

![](images/f13bfc113fadb62684bf7ad994c8d7ed55c709080d2405001d50b3925f7b43b3.jpg)  
Fig. 1: Overview of BooST. Our framework consists of two stages: (left) Unified Skill Pretraining, which learns a crossmodal skill representation from large datasets, and (right) Downstream Adaptation, which distills the learned skills into a lightweight policy and skill prior for efficient adaptation.

agents to efficiently adapt to novel tasks, even under limited data availability.

Achieving such efficient skill transfer to real robots requires three key properties: generalization, robustness, efficiency. Skill representations should exhibit strong generalization, enabling transfer across novel scenes, tasks, and even to different robotic embodiments. Realizing this level of generalization requires models to leverage extensive and heterogeneous datasets, which inevitably contain visual noise [5]– [7]. Accordingly, skill learning should demonstrate robustness to visual distractors and background variations, allowing effective pretraining without relying on filtered or curated data. Moreover, practical skill transfer requires efficiency, entailing lightweight yet expressive policies for real-world deployment. While strong generalization typically demands large-scale models with high representational capacity [2], [5], such models are computationally prohibitive for real-world robots due to latency and resource constraints.

To satisfy these properties, existing methods have learned skills by abstracting information at different levels: low-level methods abstract from action data, while high-level methods abstract from visuo-linguistic data. Low-level methods [3], [8]–[10] focus on the how, modeling the fine-grained motion dynamics across diverse trajectories. However, their reliance on embodiment-specific behaviors and the lack of semantic grounding jointly hinder generalization across robotic embodiments and tasks. High-level methods [4]–[6], [11]–[14] focus on the what, capturing the semantic intent inferred from an image or language instruction (e.g., pick up things), enabling cross-domain transfer. Yet their dependence only on visuo-linguistic features makes them sensitive to visual distractors, undermining robustness. Moreover, several highlevel approaches [5], [6] rely on large-scale models to compensate for insufficient guidance from learned skills to the policy, which in turn reduces efficiency. Ultimately, both directions fall short because they capture only one aspect of a skill—either the motion dynamics or the semantic intent. As a result, their learned representations fail to transfer in a way that achieves generalization, robustness, and efficiency simultaneously, requiring substantial in-domain data for adaptation to new scenes and tasks.

In this paper, we introduce BooST (Bridging semantics and motions for Efficient Skill Transfer), a framework for learning a unified skill representation that captures both semantic intent (what) and motion dynamics (how). This representation enables cross-domain transfer with practical deployability and robustness to visual distractors, thereby facilitating few-shot adaptation. BooST adopts a decoupled two-stage training paradigm. In the first stage, it pretrains a unified skill representation on large-scale and diverse datasets to jointly encode high-level semantics and low-level motions. At its core, BooST employs a cross-modal VQ-VAE [15], [16] that learns a shared codebook via two pathways. The visuo-linguistic pathway extracts semantic intent from visual observations and language instructions using a pretrained CLIP model [17] and a cross-attention mechanism. In parallel, the action pathway encodes motion dynamics from action trajectories, grounding the learned semantics in executable behaviors. The proposed design also underpins BooST’s robustness to dynamic visual distractors (e.g., irrelevant moving objects) and background variations (e.g., changes in surrounding layouts and objects). The resulting skill representation generalizes effectively across diverse scenes, tasks, and robotic embodiments, yielding highly transferable skills. In the second stage, this rich representation is distilled into a lightweight policy and skill prior, enabling efficient downstream adaptation.

We validate BooST through both simulation and real-world robot experiments, focusing on few-shot adaptation. Across various data regimes, BooST consistently outperforms existing skill-based methods, particularly in the low-data regime. Notably, in real-world settings, BooST achieves multi-task adaptation with only five demonstrations per task, and further generalizes across new scenes, tasks, and even different robotic embodiments. In addition, BooST demonstrates robust skill learning when pretrained in environments containing dynamic visual distractors, highlighting its resilience to irrelevant visual noise.

To summarize, our main contributions are:

• We present BooST, a framework that learns a unified skill representation by integrating semantic intent and motion dynamics into a shared skill codebook.

• We introduce a decoupled two-stage training pipeline that reconciles large-scale pretraining with deployment efficiency, enabling efficient adaptation of a lightweight policy to new tasks in a few-shot manner.

• We demonstrate that BooST achieves strong few-shot adaptation, cross-embodiment transfer, and robustness to dynamic visual distractors and background variations across simulation and real-world experiments.

## II. RELATED WORKS

## A. Skill Abstraction from Offline Data

a) Low-level skill abstractions: Low-level skill methods [3], [8]–[10] quantize action trajectories into discrete latent skill spaces using residual [18] or finite scalar vector quantization [19]. Relying solely on action sequences, these methods cannot infer which action to perform from visual or language inputs. This lack of semantic grounding degrades downstream performance [20], [21] and increases the data required for adaptation. Moreover, low-level skills are tightly coupled to the pretraining action space, so skill transfer succeeds only when the downstream environment shares the same action space. For example, skills learned from joint velocity action space do not transfer to downstream tasks that use Cartesian end-effector position control. In contrast, BooST integrates semantic understanding with motion dynamics, yielding semantically grounded skills. Furthermore, BooST leverages the visuo-linguistic pathway alone during skill transfer, disentangling the learned representation from the pretraining action space. This design enables skill transfer across heterogeneous action spaces, supporting cross-embodiment skill transfer.

b) High-level skill abstractions: High-level skill methods map visual and linguistic inputs into discrete latent skill spaces to encode semantic intent. One line of work [4], [11], [12] quantizes past observations and language instructions into discrete skills through vector quantization [22], but trains visual encoder from scratch without pretrained foundation models. As a result, the learned skills overfit visual patterns in the training data, become sensitive to minor appearance changes, and transfer poorly to novel scenes and tasks. More recent methods [13], [14] leverage vision foundation models [2], [20] to improve generalization, but define skills primarily in terms of visual changes without grounding them in actions or language. Their representations thus remain vulnerable to visual distractors and fail to capture meaningful motion dynamics. BooST instead fuses image and language features from a pretrained CLIP model via cross-attention to obtain task-aware features. Together with the action reconstruction objective and the explicit action pathway, this design yields motion-grounded skills that are robust to dynamic visual distractors and background variations.

## B. Latent Action Pretraining

Recent approaches learn latent actions that describe what motion follows from visual and linguistic inputs [5]–[7], [23]. These methods use large-scale human or robot video datasets and apply vector quantization [22] to encode visual changes between frames into discrete latent spaces. The resulting representations have been primarily used to pretrain visionlanguage-action (VLA) models for downstream policy learning. However, because these approaches define latent actions purely from visual differences, the learned spaces tend to capture task-irrelevant motion when dynamic visual distractors are present, degrading policy performance. In contrast, BooST learns discrete latent skills that remain robust under dynamic visual distractors by jointly leveraging an action pathway, explicit action supervision, and task-aware visual features.

![](images/9d5241f59064eecbce6a32ab54fbeaddb735cb12196b5370a0a307501a3a3a5a.jpg)  
Fig. 2: The framework consists of two stages: (Left) Stage I: Unified Skill Pretraining, which learns a cross-modal unified skill representation by jointly encoding semantic intent (visuo-linguistic pathway) and motion dynamics (action pathway); and (Right) Stage II: Downstream Adaptation, which distills the learned skill representation into a lightweight causal skill prior and low-level policy for efficient adaptation to new tasks. Attention heatmaps within the skill encoder indicate that BooST focuses on task-relevant regions during pretraining. The attention heatmaps are best viewed in the digital version.

Furthermore, our method requires no manual curation of robot data, enabling the use of easy-to-access and continually growing in-the-wild robot datasets [24].

## III. PRELIMINARIES

## A. Background

We briefly review the three components of BooST. VQ-VAE and its variants [18], [25] are an autoencoder that learns a discrete latent representation. An encoder maps an input to a continuous vector, a learnable codebook quantizes it to the nearest code, and a decoder reconstructs the input, with all three trained jointly by a reconstruction objective. The CLIP image encoder [17] is a Vision Transformer trained with contrastive image–language pretraining, producing semantic image features aligned with language. BAKU [26] is a transformer-based policy that fuses multimodal observations and predicts action trajectories.

## B. Problem Setup

Our goal is to learn a policy that enables a robot to perform a wide range of tasks specified by natural language instructions. We assume access to a dataset of N language-conditioned trajectories $\mathcal { D } = \{ \big ( l ^ { ( i ) } , O _ { 1 } ^ { ( i ) } , a _ { 1 } ^ { ( i ) } , \dots , O _ { T _ { i } } ^ { ( i ) } , a _ { T _ { i } } ^ { ( i ) } \big ) \} _ { i = 1 } ^ { N } ,$ , where each trajectory is paired with a natural language instruction l, and each observation $O _ { t } = ( I _ { t } , I _ { t } ^ { \mathrm { g r i p p e r } } , p _ { t } )$ consists of a front camera image, a gripper camera image (if available), and the robot’s proprioceptive state, while the action $a _ { t }$ is continuousvalued. Our objective is to learn a language-conditioned policy $\Pi ( a _ { t : t + H } \mid O _ { t - L : t } , l )$ . Here, L denotes the observation history length and H is the action prediction horizon.

## IV. METHODS

Our method decouples large-scale skill pretraining from small-scale downstream adaptation, as illustrated in Fig. 2. Section IV-A introduces the overall learning formulation and explains how the training objective derived from the Evidence Lower Bound (ELBO) naturally motivates this two-stage design. Section IV-B describes the large-scale skill pretraining phase, where a unified skill representation is learned from diverse offline datasets. Section IV-C presents the downstream adaptation phase, in which a lightweight skill prior and policy are efficiently trained for target tasks.

## A. Decoupling Skill Learning and Downstream Adaptation

Training a monolithic, large-scale policy Π yields powerful but computationally impractical models for real-world deployment. To enable lightweight yet expressive policy learning, we reformulate the objective of Π within a variational framework by introducing a latent skill variable z. Marginalizing over z decomposes Π into a low-level policy π<sub>θ</sub> and a skill prior $p \mathrm { : }$

$$
\begin{array} { l } { \log \Pi ( a _ { t : t + H } \mid O _ { t - L : t } , l ) } \\ { = \log \displaystyle \sum _ { z _ { t } } \pi _ { \theta } ( a _ { t : t + H } \mid z _ { t } , O _ { t - L : t } , l ) p ( z _ { t } \mid O _ { t - L : t } , l ) . } \end{array}\tag{1}
$$

Rewriting the marginal as an expectation under a skill encoder $q _ { \phi }$ and applying Jensen’s inequality yields the ELBO:

$$
\begin{array} { r l } & { \mathcal { L } ( \theta , \psi , \phi ) = \mathbb { E } _ { z _ { t } \sim q _ { \phi } } [ \log \pi _ { \theta } ( a _ { t : t + H } \mid z _ { t } , O _ { t - L : t } ) ] } \\ & { \qquad - D _ { \mathrm { K L } } ( q _ { \phi } ( z _ { t } \mid I _ { t } , I _ { t + H } , l ) \parallel p _ { \psi } ( z _ { t } \mid O _ { t - L : t } , l ) ) . } \end{array}\tag{2}
$$

The resulting ELBO decomposes the learning problem into a skill encoder $q _ { \phi } ,$ a skill prior $p _ { \psi } .$ , and a low-level policy $\pi _ { \theta } ,$ and maps directly to our training objectives: the KL term becomes the prior distillation loss for $p _ { \psi } .$ and the actionlikelihood term becomes the behavior-cloning loss for π<sub>θ</sub> (Eq. 6). To distill the expressiveness of large-scale models into compact policies, we adopt a two-stage training framework: Stage I pretrains the skill encoder $q _ { \phi }$ on large-scale datasets to learn a unified skill representation, and Stage II distills this representation into the lightweight prior $p _ { \psi }$ and policy π<sub>θ</sub> through downstream adaptation. This decoupled, distillationbased approach preserves the representational richness of large models while maintaining computational efficiency for realworld robotic deployment.

## B. Stage I: Unified Skill Pretraining

The goal of Stage I is to pretrain a unified skill representation from large-scale, diverse offline data that jointly captures high-level semantic intent (what) and low-level motion dynamics (how). This stage, illustrated on the left side of Fig. 2, corresponds to the unified skill pretraining phase of BooST. To this end, we adopt a cross-modal VQ-VAE framework that learns a discrete latent space of reusable skills. Two complementary pathways contribute to the skill learning process: a visuo-linguistic pathway for semantic grounding and an action pathway for motion dynamics encoding. We alternately optimize these two pathways to ensure a balanced and robust skill representation.

The visuo-linguistic pathway provides semantic context through our skill encoder $q _ { \phi } ,$ , defined as $q _ { \phi } ( \cdot ) \ =$ $E _ { \mathrm { t r a n s } } ( E _ { \mathrm { t a s k } } ( \cdot ) )$ . Inspired by recent works [27], [28], the taskaware encoder $E _ { \mathrm { t a s k } }$ takes the current and future images $\left( I _ { t } , I _ { t + H } \right)$ along with the language instruction l, and fuse patch-level visual tokens from a pretrained CLIP ViT with the instruction embedding through temperature-scaled crossattention:

$$
\begin{array} { r } { f _ { \mathrm { v l } } = \mathrm { s o f t m a x } \Big ( \frac { \hat { f } _ { l } \hat { f } _ { v } ^ { \top } } { \tau } \Big ) \big ( \hat { f } _ { v } + \mathrm { P E } ) , } \end{array}\tag{3}
$$

where $\hat { f } _ { v }$ and $\hat { f } _ { l }$ denote normalized visual and language features, respectively, τ is a learnable temperature, and PE denotes positional embeddings. This fusion allows the model to selectively attend to instruction-relevant visual regions (see Fig. 3 for qualitative attention visualization). The fused features are subsequently processed by a transformer encoder $E _ { \mathrm { t r a n s } }$ , which models temporal relationships between the current and future frames, producing the final visuo-linguistic embedding $f _ { \mathrm { e n c , v l } } = E _ { \mathrm { t r a n s } } ( ( f _ { \mathrm { v l } } ) _ { t } , ( f _ { \mathrm { v l } } ) _ { t + H } )$ . In parallel, the action pathway encoder $E _ { \mathrm { a c t } }$ processes action trajectories to capture fine-grained motion dynamics, producing dynamic features $f _ { \mathrm { e n c , ~ a c t } } ~ = ~ E _ { \mathrm { a c t } } ( a _ { t : t + H } )$ . Both pathways focus on extracting task-relevant information, which is subsequently quantized into a shared discrete skill codebook. The continuous feature vector $f _ { \mathrm { e n c } }$ from either pathway (i.e., $f _ { \mathrm { e n c , v l } }$ or $f _ { \mathrm { e n c , ~ a c t } } )$ is then quantized to the nearest vector $c _ { k }$ in the codebook, $\mathcal { C } = \{ c _ { k } \} _ { k = 1 } ^ { K }$ , to obtain the discrete skill z :

$$
z _ { t } = \underset { k \in \{ 1 , \ldots , K \} } { \arg \operatorname* { m i n } } \left\| f _ { \mathrm { e n c } } - c _ { k } \right\| _ { 2 } ^ { 2 } .\tag{4}
$$

The framework is optimized via a single supervisory signal, which is action reconstruction. The action decoder $D _ { \mathrm { { a c t } } }$ reconstructs action sequences from the discrete skill $z _ { t }$ and current task-aware visual feature $( f _ { \mathrm { v l } } ) _ { t } , \mathrm { i . e . , } \hat { a } _ { t : t + H } = D _ { \mathrm { a c t } } ( z _ { t } , ( f _ { \mathrm { v l } } ) _ { t } )$ Unlike prior approaches that reconstructs pixel-level images [5]–[7], BooST employs low-dimensional action sequences as the reconstruction target. This design encourages the model to ignore visual details irrelevant to the task, and the action pathway further reinforces robustness against dynamic visual distractors. We employ residual vector quantization [18] and apply a rotation trick [29] to prevent codebook collapse. The overall pretraining objective is defined as a weighted sum of reconstruction losses from both pathways:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { p r e t r a i n } } ( \phi ) = \lambda _ { 1 } \underbrace { | | a _ { t : t + H } - \hat { a } _ { \mathrm { v } 1 } | | _ { 2 } ^ { 2 } } _ { \mathrm { v i s u o - l i n g u i s t i c } } + \lambda _ { 2 } \underbrace { | | a _ { t : t + H } - \hat { a } _ { \mathrm { a c t } } | | _ { 2 } ^ { 2 } } _ { \mathrm { a c t i o n } } } \\ & { ~ + \lambda _ { 3 } \underbrace { | | f _ { \mathrm { e n c , v } 1 } - f _ { \mathrm { e n c , a c t } } | | _ { 2 } ^ { 2 } } _ { \mathrm { r e g u l a r i z a t i o n } } } \end{array}\tag{5}
$$

![](images/9d5f7ee907b57f401924cbcb08ae69880639387ab0efcea8e58cffcf5a1a3be7.jpg)  
Fig. 3: Attention maps from the visuo-linguistic pathway showing that BooST attends to instruction-relevant regions conditioned on the task instruction.

where $a _ { t : t + H }$ is the ground-truth action sequence, and $\hat { a } _ { \bf v l }$ and $\hat { a } _ { \mathrm { a c t } }$ are the reconstructions from the two pathways. In our experiments, we set $\lambda _ { 1 } = 3 , \lambda _ { 2 } = 1$ , and $\lambda _ { 3 } = 0 . 5$ , and use a residual vector quantization codebook of size $K = 1 6$ with 2 quantization levels.

## C. Stage II: Downstream Adaptation

In Stage II, we distill the rich knowledge encoded in the pretrained skill encoder $q _ { \phi }$ into two lightweight components: a skill prior $p _ { \psi }$ and a low-level policy $\pi _ { \theta } .$ , both designed for efficient deployment. This stage, illustrated on the right side of Fig. 2, corresponds to the downstream adaptation phase of BooST. Unlike Stage I, which leverages future image to learn predictive representations, Stage II uses only past observations, as future inputs are unavailable during execution. Both $p _ { \psi }$ and $\pi _ { \theta }$ are trained on a smaller, in-domain dataset and implemented as small Transformer models that process multimodal input stream consisting of camera features and proprioceptive states. The policy head follows BAKU [26] and models the action distribution as an isotropic Gaussian.

$$
\begin{array} { r l r } & { } & { \mathcal { L } _ { \mathrm { d o w n s t r e a m } } ( \theta , \psi ; \phi ) = \underbrace { { \mathbb E } _ { z _ { t } ^ { q } \sim q _ { \phi } } \left[ - \log p _ { \psi } ( z _ { t } ^ { q } \mid O _ { t - L : t } , l ) \right] } _ { \mathrm { P r i o r ~ D i s t i l l a t i o n ~ L o s s } } } \\ & { } & { + \ \alpha \underbrace { { \mathbb E } _ { z _ { t } \sim p _ { \psi } } \left[ - \log \pi _ { \theta } ( a _ { t : t + H } \mid \sec ( z _ { t } ) , O _ { t - L : t } ) \right] } _ { \mathrm { P o l i c y ~ B e h a v i o r ~ C l o n i n g ~ L o s s } } . } \end{array}\tag{6}
$$

The first term trains the skill prior $p _ { \psi }$ to approximate the distribution of latent skills predicted by the frozen skill encoder $q _ { \phi } ,$ , where $z _ { t } ^ { q } \sim q _ { \phi } ( I _ { t } , I _ { t + H } , l )$ serves as a pseudo-label. The second term optimizes the policy $\pi _ { \theta }$ to imitate expert actions conditioned on sampled skills. For stable training, the stopgradient operator sg prevents gradient flow from the policy into the prior, ensuring that the prior is supervised solely by the fixed encoder outputs.

## V. EXPERIMENTS

We evaluate BooST in both simulation (LIBERO [31]) and real-world settings (UR3 robot with a Robotiq 2F-85 gripper). For both settings, we pretrain BooST’s unified skill representation on the large-scale DROID dataset [24], which contains 76k teleoperated trajectories collected with a Franka Emika Panda arm equipped with a Robotiq 2F-85 gripper. The action space of the DROID dataset is defined in joint velocity space, whereas all downstream environments (LIBERO and

TABLE I: Few-shot adaptation performance on LIBERO benchmarks. We report the mean success rate with 95% confidence intervals over five random seeds, each evaluated across 50 rollouts per task. The downstream skill prior and policy are trained with varying numbers of demonstrations (50, 20, and 10). Bold and underlined numbers indicate the best and the second-bes mean in each column, respectively. Relative improvements $( + \mathbf { \boldsymbol { X } } \frac { \boldsymbol { \circ } } { \boldsymbol { \circ } } )$ are computed over the second-best method. This relative improvement generally increases as the number of demonstrations decreases (from 50 down to 10).
<table><tr><td rowspan="2">Method</td><td colspan="3">LIBERO-90 (# demos)</td><td colspan="3">LIBERO-Goal (# demos)</td><td colspan="3">LIBERO-Object (# demos)</td><td colspan="3">LIBERO-Spatial (# demos)</td></tr><tr><td>50</td><td>20</td><td>10</td><td>50</td><td>20</td><td>10</td><td>50</td><td>20</td><td>10</td><td>50</td><td>20</td><td>10</td></tr><tr><td>Diffusion Policy [30]</td><td> $0 . 6 0 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $0 . 3 3 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $0 . 2 4 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $0 . 5 6 { \pm } 0 . 1 5$ </td><td> $\underline { { 0 . 4 7 } } \pm 0 . 0 5$ </td><td> $\underline { { 0 . 4 1 \pm 0 . 0 4 } }$ </td><td> $0 . 3 6 { \pm } 0 . 2 8$ </td><td> $0 . 2 5 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $0 . 2 0 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $0 . 7 4 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $0 . 5 0 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $\underline { { 0 . 4 2 \pm 0 . 0 6 } }$ </td></tr><tr><td>VQ-BeT [8]</td><td> $\underline { { 0 . 6 4 } } \pm 0 . 1 2$ </td><td> $\underline { { 0 . 5 1 \pm 0 . 0 5 } }$ </td><td> $\underline { { 0 . 2 9 \pm 0 . 0 7 } }$ </td><td> $\underline { { 0 . 7 3 \pm 0 . 0 9 } }$ </td><td> $0 . 4 5 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 3 2 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $0 . 3 6 { \pm } 0 . 1 7$ </td><td> $0 . 3 2 { \pm } 0 . 1 8$ </td><td> $0 . 1 6 { \pm } 0 . 2 0$ </td><td> $\underline { { 0 . 8 0 \pm 0 . 0 7 } }$ </td><td> $\underline { { 0 . 6 3 } } \pm 0 . 0 6$ </td><td> $0 . 3 7 { \pm } 0 . 0 9$ </td></tr><tr><td>QueST [3]</td><td> $0 . 5 1 { \pm } 0 . 0 2$ </td><td> $0 . 3 7 { \scriptstyle \pm 0 . 0 2 }$ </td><td>0.28±0.02</td><td> $0 . 3 0 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $0 . 2 5 { \scriptstyle \pm 0 . 0 6 }$ </td><td>0.22±0.05</td><td> $0 . 1 1 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $0 . 0 6 \pm 0 . 0 5$ </td><td> $0 . 0 8 { \pm } 0 . 0 7$ </td><td> $0 . 2 8 { \pm } 0 . 0 3$ </td><td> $0 . 2 1 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $0 . 1 0 { \scriptstyle \pm 0 . 0 4 }$ </td></tr><tr><td>LISA [4]</td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>EXTRACT [13]</td><td> $0 . 2 2 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $0 . 2 0 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $0 . 1 4 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $0 . 0 9 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $0 . 0 6 { \pm } 0 . 0 3$ </td><td> $0 . 0 4 \pm 0 . 0 2$ </td><td> $\underline { { 0 . 8 8 \pm 0 . 0 4 } }$ </td><td> $\underline { { 0 . 6 9 \pm 0 . 0 9 } }$ </td><td> $\underline { { 0 . 5 1 \pm 0 . 1 2 } }$ </td><td> $0 . 7 5 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $0 . 5 6 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $0 . 2 9 { \scriptstyle \pm 0 . 0 9 }$ </td></tr><tr><td>BooST (Ours)</td><td> $\mathbf { 0 . 9 1 } \pm 0 . 0 1$ </td><td> $\mathbf { 0 . 8 2 \pm } 0 . 0 3$ </td><td> $\mathbf { 0 . 7 0 } \pm 0 . 0 2$ </td><td> $\mathbf { 0 . 9 2 \pm } 0 . 0 2$ </td><td> ${ \bf 0 . 8 1 \pm 0 . 0 2 }$ </td><td> $\mathbf { 0 . 6 8 \pm } 0 . 0 4$ </td><td> ${ \bf 0 . 9 5 \pm 0 . 0 3 }$ </td><td> ${ \bf 0 . 8 5 \pm 0 . 1 7 }$ </td><td> $\mathbf { 0 . 8 0 \dot { \pm } } 0 . 0 9$ </td><td> ${ \bf 0 . 9 1 } \pm \mathrm { 0 . 0 5 }$ </td><td> ${ \bf 0 . 8 0 \pm 0 . 0 5 }$ </td><td> ${ \bf 0 . 6 0 } \pm 0 . 0 7$ </td></tr><tr><td></td><td> $( + 4 1 \% )$ </td><td> $( + 5 9 \% )$ </td><td> $( + 1 4 0 \% )$ </td><td> $( + 2 5 \% )$ </td><td> $( + 7 4 \% )$ </td><td> $( + 6 5 \% )$ </td><td> $( + 8 \% )$ </td><td> $( + 2 4 \% )$ </td><td> $( + 5 7 \% )$ </td><td> $( + 1 3 \% )$ </td><td> $( + 2 7 \% )$ </td><td> $( + 4 3 \% )$ </td></tr></table>

UR3) operate in Cartesian end-effector space. We also train and evaluate the robustness of BooST under dynamic visual distractors and perform ablation studies to investigate key design choices. We aim to address the following key questions in our experiments:

Results. Table I reports the quantitative results across four LIBERO benchmarks under varying numbers of demonstra-• VQ-BeT [8] and QueST [3] represent low-level skill approaches that quantize action trajectories into discrete motion primitives but lack semantic grounding.

• LISA [4] and EXTRACT [13] exemplify high-level skill approaches that focus on semantic intent. LISA encodes past observations and language instructions into discrete skills, jointly optimizing skill selection and policy execution. EXTRACT defines skills as visual feature transitions using pretrained vision foundation models. For fair comparison, we adapt EXTRACT to the imitation learning setting with the same skill prior, low-level policy, and behavior cloning objective as BooST.

• Diffusion Policy [30] directly maps observations and language instructions to action sequences without skill abstraction.

Experiment setup. We evaluate how the unified skill representation affects few-shot adaptation performance on the LIBERO benchmark [31], which includes LIBERO-90, LIBERO-Goal, LIBERO-Object, and LIBERO-Spatial. We train downstream policies with 50, 20, or 10 demonstrations per task to assess sample efficiency under varying data regimes. Notably, none of the LIBERO samples appear in the pretraining dataset, necessitating generalization over novel scenes and tasks in skill transfer.

Baselines. We compare BooST with five representative methods covering both low- and high-level skill abstractions:

(Q1) Does bridging semantic intent (what) and motion dynamics (how) lead to sample-efficient downstream adaptation compared to prior skill-based approaches?

## A. (Q1): Downstream Adaptation in Simulation

(Q2) Can BooST enable skill transfer beyond novel scenes and tasks to different robotic embodiments?

(Q3) Can BooST robustly learn skills in environments with dynamic visual distractors?

tions. BooST consistently outperforms all baselines in every setting. Notably, the relative improvement over the secondbest method becomes more pronounced as the number of demonstrations decreases, demonstrating BooST’s strong fewshot adaptation capability. For example, on LIBERO-90 (the most diverse benchmark), BooST achieves relative gains of +140%, +59%, and +41% with 10, 20, and 50 demonstrations, respectively. We attribute this improvement to the unified skill representation, which jointly captures semantic intent and motion dynamics, enabling effective transfer to novel scenes and tasks. As a qualitative validation, Fig. 4 shows that BooST preserves consistent semantic intent and motion dynamics across environments, successfully transferring skills from the DROID source domain to the LIBERO targets.

In contrast, both low-level and high-level baselines underperform BooST, because each captures only one aspect— motion dynamics or semantic intent—leaving their skill representations incomplete. These limitations are exemplified by high-level methods such as EXTRACT. It achieves competitive results on simple benchmarks (LIBERO-Object, Spatial) but underperforms on LIBERO-90 and Goal, where diverse motion dynamics are required. This is because defining skills purely through visual feature variations neglects motion fidelity. Notably, LISA completely fails in downstream adaptation experiments. Its joint optimization of skill and policy causes severe training instability on large and diverse datasets, often resulting in codebook collapse. This provides empirical evidence for our design choice to decouple skill learning from downstream adaptation.

## B. (Q2): Real-World Cross-Embodiment Transfer

Experiment setup. We evaluate cross-embodiment skill transfer on a UR3 robot. The pretraining data was collected using a Franka Emika Panda arm, whereas the downstream experiments employ a UR3 robot with a different embodiment and action space. The setup includes two RGB cameras: a front-mounted and a wrist-mounted Intel RealSense D435i. We design four representative manipulation tasks and adapt policy using only five demonstrations. The baseline methods are identical to those used in the simulation experiments.

Results. The results in Fig. 5 clearly demonstrate that BooST enables sample-efficient skill transfer across heterogeneous robotic embodiments. Despite being pretrained on the Franka Emika Panda, BooST successfully transfers skills to the UR3 robot and achieves the highest success rates across all four real-world manipulation tasks. Remarkably, this performance is obtained with only five demonstrations per task, highlighting BooST’s strong sample efficiency in real-world adaptation. As illustrated in Fig. 4, the learned skills also preserve consistent semantic intent and motion dynamics across embodiments. In contrast, low-level methods such as VQ-BeT and QueST fail to transfer across embodiments because their skill representations are tied to the pretrained action space—e.g., skills learned from joint-velocity trajectories cannot generalize to robots controlled in Cartesian end-effector space.

![](images/2de0c90f7c75fc708b5a30fc62e3c2167a16432f5c14f4a0741bfb3139e08cef.jpg)  
Fig. 4: Qualitative skill transfer across domains and embodiments. Each row corresponds to a learned skill from the source domain (red, Droid) and its execution in the target domains (blue, LIBERO, UR3, and human hand). Each skill exhibits similar semantic intent and motion dynamics across distinct visual scenes and embodiments.

## C. (Q3): Robustness to Dynamic Visual Distractors

Experiment setup. We further evaluate the robustness of BooST when pretrained with dynamic visual distractors. To generate a dataset with realistic visual noise, we augment LIBERO-90 by injecting an animatable human into the scene, as shown in Fig. 6. The human model is implemented in Mu-JoCo via MJCF schema [32] and parameterized SMPL [33], which provides a kinematic joint hierarchy and a deformable surface mesh. For each episode, a single human distractor is instantiated by sampling one of 23 human texture maps [34] and one of 23 motion sequences from AMASS [35], followed by random placement and scaling within the workspace. These distractors introduce task-irrelevant visual motion, making the skill learning process more challenging. We pretrain BooST and all baselines on this distractor-augmented LIBERO-90 dataset and then evaluate downstream adaptation on the standard LIBERO-90, Goal, Object, and Spatial benchmarks, each containing 50 demonstrations per task. This setup assesses whether the pretrained skill representations remain robust and transferable when exposed to dynamic and semantically irrelevant visual variations.

Baselines. We compare BooST with latent action pretraining frameworks that learn a discrete latent space from large-scale vision-language data, similar to our approach. Specifically, we include LAPA [5] and UniVLA [6]. These methods learn latent actions from unlabeled videos by modeling visual transitions via inverse or forward dynamics, thereby embedding motion information into discrete latent representations. Given limited computational resources, we adopt their latent action pretraining strategies while keeping the skill prior and low-level policy architectures identical to those of BooST. This setup allows us to isolate the effect of latent pretraining objectives on robustness under dynamic visual distractors.

TABLE II: Performance on LIBERO benchmarks with dynamic visual distractors. We report average success rates over three random seeds, each evaluated across five rollouts per seed. Bold denotes the best result per column.
<table><tr><td>Method</td><td>90</td><td>Goal</td><td></td><td>Object Spatial</td><td>Avg.</td></tr><tr><td>LAPA [5]</td><td> $0 . 6 8 \pm 0 . 0 4$ </td><td> $0 . 6 9 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 9 1 \pm 0 . 0 8$ </td><td> $0 . 8 7 \pm 0 . 0 6$ </td><td> $0 . 7 9 { \scriptstyle \pm 0 . 0 3 }$ </td></tr><tr><td>UniVLA [6]</td><td> $0 . 6 2 \pm 0 . 0 3$ </td><td> $0 . 4 9 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $0 . 9 0 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 8 0 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 7 0 { \scriptstyle \pm 0 . 0 1 }$ </td></tr><tr><td>BooST (Ours)</td><td> ${ \bf 0 . 8 9 } _ { \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 8 8 } \pm 0 . 0 5$ </td><td> $\mathbf { 0 . 9 7 } \pm 0 . 0 3$ </td><td> $\mathbf { 0 . 8 8 } \pm 0 . 0 5$ </td><td> $\mathbf { 0 . 9 0 } \pm 0 . 0 1$ </td></tr></table>

Results. Table II shows that BooST maintains high adaptation performance even when pretrained on distractor-augmented data, outperforming latent action baselines. Both LAPA and UniVLA exhibit poor performance under distractor conditions. Since their latent action learning reconstructs images, the representations inadvertently encode background motion and task-irrelevant visual variations, rather than the agent’s own behavior. As a result, their learned representations are sensitive to moving distractors and fail to maintain consistent motion dynamics. In contrast, BooST explicitly grounds skill learning in action reconstruction and cross-modal alignment, and further extracts task-aware visual features, collectively forcing the model to focus on task-relevant behavior instead of external noise. Consequently, BooST’s unified skill captures what the agent did, rather than what merely moved in the scene. Fig. 6 illustrates the robustness at both training and test time. The skills are learned from distractor-augmented LIBERO-90 and visualized on unseen benchmarks (Goal,

![](images/1e3aaaeba51eb775ea684d3e49412fdd524f9af8e3cc8343267cd94e27f63287.jpg)

![](images/f86988c8bd8d87eaa59633746d9bafdd4129586ef4c831f0cca14c9ec422b453.jpg)

![](images/b735c89e7b9ea8b01c4f475bb5b178955d17e0cc44a65003bbdc9a9e917a6af3.jpg)

![](images/800af1973621bf272850e01d015d1a47952b64f4128fa0b7b14d17368918f085.jpg)

Fig. 5: Quantitative results on the UR3 robot. Bar charts report average success rates over three random seeds (five evaluation trials per seed), with each method trained on only five demonstrations per task.  
![](images/76878f2a71b5ecf45a03a6759a0cf5dd7ddd5ed2b52f65c64ac8e18f9ea488c1.jpg)  
Fig. 6: Visualization of skills under test-time dynamic distractors. The model is pretrained on distractor-augmented LIBERO-90 and evaluated on LIBERO-Goal and Object with dynamic human distractors present at test time. Despite the presence of dynamic background motion, each skill preserves consistent semantic intent and motion dynamics.

Object) that also contain unseen dynamic distractors. The skill encoder consistently selects the same skill for a given sub-behavior across scenes, despite distractors during both pretraining and evaluation.

## D. Ablation Studies

We conduct two ablation studies on the contribution of each component and the efficiency of our design. First, we remove the action pathway $( E _ { \mathrm { a c t } } )$ and the task-aware encoder $( E _ { \mathrm { t a s k } } )$ during skill pretraining. In the latter case, the pretrained CLIP encoder and its language-conditioned taskaware visual feature extraction are replaced with a ResNet-34 trained from scratch. All variants are pretrained same as Sec. V-A, except the number of demonstrations is fixed at 10 per task. As shown in Table III, both modules are crucial for achieving strong adaptation performance. The absence of $E _ { \mathrm { a c t } }$ impairs motion dynamics. Although the visuolinguistic pathway still reconstructs actions—thereby retaining a coarse motion signal—this lack of explicit action encoding compromises motion fidelity. Separately, omitting $E _ { \mathrm { t a s k } }$ weakens semantic grounding and generalization, leading to poorer adaptation. Second, we vary the number of parameters in the downstream model—comprising the skill prior $( p _ { \psi } )$ and low-level policy (π<sub>θ</sub>)—and measure downstream success rate. As shown in Table IV, smaller configurations retain competitive performance, suggesting that $B 0 0 { \mathrm { S T } } { \mathrm { ^ 3 } }$ unified skill representation supplies the expressiveness needed for control even with lightweight models. Across all configurations, the downstream models run at approximately 60 Hz, indicating inference efficiency suitable for real-robot deployment. For reference, our implementations of Diffusion Policy, VQ-BeT, and QueST run at 12 Hz, 95 Hz, and 30 Hz, respectively.

TABLE III: Ablation studies. Success rates are averaged over three seeds (five rollouts per seed), using 10 demonstrations for each task. Bold indicates the best performance in each column.
<table><tr><td>Ablated Components</td><td>90</td><td>Goal</td><td>Object</td><td>Spatial</td></tr><tr><td>w/o. Action Pathway  $( E _ { \mathrm { a c t } } )$ </td><td> $0 . 5 7 \pm 0 . 0 7$ </td><td> $0 . 5 7 \pm 0 . 1 1$ </td><td> $0 . 6 7 \pm 0 . 0 4$ </td><td> $0 . 4 5 { \scriptstyle \pm 0 . 0 5 }$ </td></tr><tr><td>w/o. Task-Aware Encoder  $( E _ { \mathrm { t a s k } } )$ </td><td> $0 . 2 5 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $0 . 1 4 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $0 . 5 5 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $0 . 3 0 { \scriptstyle \pm 0 . 0 3 }$ </td></tr><tr><td>BooST (Ours)</td><td> $\mathbf { 0 . 7 0 \bot } 0 . 0 2$ </td><td> ${ \bf 0 . 6 7 \pm 0 . 0 5 }$ </td><td> $\mathbf { 0 . 7 8 \pm 0 . 0 7 }$ </td><td>0.59±0.08</td></tr></table>

TABLE IV: Comparison of model size and performance. Success rates are averaged over three seeds (five rollouts per seed). Bold indicates the best performance in each column.
<table><tr><td>Model Size</td><td>Goal</td><td>Object</td><td>Spatial</td><td>Avg.</td></tr><tr><td>29.7M</td><td> $0 . 9 3 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $0 . 9 1 \pm 0 . 0 6$ </td><td> $0 . 8 3 \pm 0 . 0 1$ </td><td> $0 . 8 9 \pm 0 . 0 2$ </td></tr><tr><td>69.5M</td><td> $0 . 9 3 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 9 7 \pm 0 . 0 1 }$ </td><td> ${ \bf 0 . 9 1 } \pm \mathrm { 0 . 0 5 }$ </td><td> ${ \bf 0 . 9 4 } \pm 0 . 0 2$ </td></tr><tr><td>144.5M</td><td> $\mathbf { 0 . 9 6 { \scriptstyle \pm 0 . 0 2 } }$ </td><td> $0 . 9 3 { \scriptstyle \pm 0 . 0 5 }$ </td><td> ${ \bf 0 . 9 1 } \pm \mathrm { 0 . 0 4 }$ </td><td> $0 . 9 3 { \scriptstyle \pm 0 . 0 2 }$ </td></tr></table>

## E. Limitations

Despite the strong performance, there remain opportunities for further improvement. First, because skills are extracted from 2D images, performance can degrade on tasks that demand especially precise motion along the z-axis in camera coordinates. Second, to enable lightweight deployment, the distilled skill prior is implemented with a non-CLIP-based network, which can face challenges under large viewpoint changes that require stronger 3D understanding.

## VI. CONCLUSIONS

Our work introduces a two-stage framework that learns a unified skill representation by jointly capturing semantic intent and motion dynamics, and then distills this representation into a lightweight policy and skill prior for efficient adaptation. Extensive experiments in simulation and real-world settings demonstrate that BooST enables strong few-shot adaptation, cross-embodiment skill transfer, and robustness to dynamic visual distractors and background variations. These results indicate that BooST attains the three desiderata for efficient skill transfer—generalization, robustness, and efficiency. A promising future direction is to extend BooST with 3D-aware representations, for instance by incorporating depth estimation models [36] (e.g., DepthAnything [36]), to enable more finegrained 3D skill extraction.

## REFERENCES

[1] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in International Conference on Computer Vision (ICCV), 2023.

[2] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “Dinov2: Learning robust visual features without supervision,” Transactions on Machine Learning Research Journal, 2024.

[3] A. Mete, H. Xue, A. Wilcox, Y. Chen, and A. Garg, “Quest: Selfsupervised skill abstractions for learning continuous control,” in Neural Information Processing Systems (NeurIPS), 2024.

[4] D. Garg, S. Vaidyanath, K. Kim, J. Song, and S. Ermon, “Lisa: Learning interpretable skill abstractions from language,” in Neural Information Processing Systems (NeurIPS), 2022.

[5] S. Ye, J. Jang, B. Jeon, S. Joo, J. Yang, B. Peng, A. Mandlekar, R. Tan, Y.-W. Chao, B. Y. Lin et al., “Latent action pretraining from videos,” in International Conference on Learning Representations (ICLR), 2025.

[6] Q. Bu, Y. Yang, J. Cai, S. Gao, G. Ren, M. Yao, P. Luo, and H. Li, “Univla: Learning to act anywhere with task-centric latent actions,” in Robotics: Science and Systems (RSS), 2025.

[7] H. Kim, J. Kang, H. Kang, M. Cho, S. J. Kim, and Y. Lee, “Uniskill: Imitating human videos via cross-embodiment skill representations,” in Conference on Robot Learning (CoRL), 2025.

[8] S. Lee, Y. Wang, H. Etukuru, H. J. Kim, N. M. M. Shafiullah, and L. Pinto, “Behavior generation with latent actions,” in International Conference on Machine Learning (ICML), 2024.

[9] K. Pertsch, Y. Lee, and J. Lim, “Accelerating reinforcement learning with learned skill priors,” in Conference on Robot Learning (CoRL), 2020.

[10] R. Zheng, C.-A. Cheng, H. Daume III, F. Huang, and A. Kolobov,´ “Prise: Llm-style sequence compression for learning temporal action abstractions in control,” in International Conference on Machine Learning (ICML), 2024.

[11] Z. Liang, Y. Mu, H. Ma, M. Tomizuka, M. Ding, and P. Luo, “Skilldiffuser: Interpretable hierarchical planning via skill abstractions in diffusion-based task execution,” in Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[12] H. Jiang, J. Wang, and Z. Lu, “Discrete latent plans via semantic skill abstractions,” in International Conference on Learning Representations (ICLR), 2025.

[13] J. Zhang, M. Heo, Z. Liu, E. Biyik, J. J. Lim, Y. Liu, and R. Fakoor, “Extract: Efficient policy learning by extracting transferable robot skills from offline data,” in Conference on Robot Learning (CoRL), 2024.

[14] W. Wan, Y. Zhu, R. Shah, and Y. Zhu, “Lotus: Continual imitation learning for robot manipulation through unsupervised skill discovery,” in IEEE International Conference on Robotics and Automation (ICRA), 2024.

[15] S. Yoo, S. Jung, Y. Lee, D. Shim, and H. J. Kim, “Mono-cameraonly target chasing for a drone in a dense environment by cross-modal learning,” IEEE Robotics and Automation Letters, vol. 9, no. 8, pp. 7254–7261, 2024.

[16] A. Spurr, J. Song, S. Park, and O. Hilliges, “Cross-modal deep variational hand pose estimation,” in Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

[17] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning (ICML), 2021.

[18] N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi, “Soundstream: An end-to-end neural audio codec,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 495–507, 2021.

[19] F. Mentzer, D. Minnen, E. Agustsson, and M. Tschannen, “Finite scalar quantization: Vq-vae made simple,” in International Conference on Learning Representations (ICLR), 2024.

[20] S. Nair, A. Rajeswaran, V. Kumar, C. Finn, and A. Gupta, “R3m: A universal visual representation for robot manipulation,” in Conference on Robot Learning (CoRL), 2023.

[21] Y. J. Ma, V. Kumar, A. Zhang, O. Bastani, and D. Jayaraman, “Liv: Language-image representations and rewards for robotic control,” in International Conference on Machine Learning (ICML), 2023.

[22] R. Gray, “Vector quantization,” IEEE Assp Magazine, vol. 1, no. 2, pp. 4–29, 1984.

[23] J. Bruce, M. D. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps et al., “Genie: Generative interactive environments,” in International Conference on Machine Learning (ICML), 2024.

[24] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis et al., “Droid: A large-scale in-the-wild robot manipulation dataset,” arXiv preprint arXiv:2403.12945, 2024.

[25] M. H. Vali and T. Backstr ¨ om, “Nsvq: Noise substitution in vector¨ quantization for machine learning,” IEEE Access, vol. 10, pp. 13 598– 13 610, 2022.

[26] S. Haldar, Z. Peng, and L. Pinto, “Baku: An efficient transformer for multi-task policy learning,” in Neural Information Processing Systems (NeurIPS), 2024.

[27] M. Lan, C. Chen, Y. Ke, X. Wang, L. Feng, and W. Zhang, “Clearclip: Decomposing clip representations for dense vision-language inference,” in European Conference on Computer Vision (ECCV), 2024.

[28] H. Huang, F. Liu, L. Fu, T. Wu, M. Mukadam, J. Malik, K. Goldberg, and P. Abbeel, “Otter: A vision-language-action model with text-aware visual feature extraction,” in International Conference on Machine Learning (ICML), 2025.

[29] C. Fifty, R. G. Junkins, D. Duan, A. Iyengar, J. W. Liu, E. Amid, S. Thrun, and C. Re, “Restructuring vector quantization with the rotation´ trick,” in International Conference on Learning Representations (ICLR), 2025.

[30] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” The International Journal of Robotics Research, vol. 44, no. 10-11, pp. 1684–1704, 2024.

[31] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone, “Libero: Benchmarking knowledge transfer for lifelong robot learning,” in Neural Information Processing Systems (NeurIPS), 2023.

[32] E. Todorov, T. Erez, and Y. Tassa, “Mujoco: A physics engine for model-based control,” in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2012, pp. 5026–5033.

[33] M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, and M. J. Black, “SMPL: A skinned multi-person linear model,” ACM Transactions on Graphics, vol. 34, no. 6, pp. 1–16, 2015.

[34] D. Casas and M. Comino-Trinidad, “SMPLitex: A Generative Model and Dataset for 3D Human Texture Estimation from Single Image,” in British Machine Vision Conference (BMVC), 2023.

[35] N. Mahmood, N. Ghorbani, N. F. Troje, G. Pons-Moll, and M. J. Black, “AMASS: Archive of motion capture as surface shapes,” in International Conference on Computer Vision (ICCV), 2019.

[36] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, “Depth anything: Unleashing the power of large-scale unlabeled data,” in Conference on Computer Vision and Pattern Recognition (CVPR), 2024.