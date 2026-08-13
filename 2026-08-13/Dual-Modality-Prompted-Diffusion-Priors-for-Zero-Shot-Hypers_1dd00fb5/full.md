# Dual Modality Prompted Diffusion Priors for Zero Shot Hyperspectral Pansharpening

Pengwei Xie, Fei Zhu, Jiajun Li, Xiangyuan Liu, Kangqing Shen, and Gemine Vivone, Senior Member, IEEE

Abstract—Hyperspectral pansharpening aims to reconstruct a high resolution hyperspectral (HRHS) image from a panchromatic (PAN) image and a low resolution hyperspectral (LRHS) image while preserving both spatial details and spectral fidelity. Recent diffusion based methods exploit pretrained image priors by generating a low dimensional representation and subsequently mapping it to the hyperspectral domain. However, the observed panchromatic and hyperspectral images are typically imposed only through external reconstruction objectives, limiting their direct interaction with the diffusion prior. To address this issue, we propose dual-modality image-prompted diffusion model (DIDM) for zero shot hyperspectral pansharpening. DIDM encodes the low resolution hyperspectral and panchromatic observations into spectral and spatial prompt tokens, respectively, and injects them into intermediate features of a frozen remote sensing diffusion model through cross attention, allowing complementary spectral and spatial information to directly guide diffusion feature evolution. In addition, we introduce a panchromatic guided weighted pixel aware total variation regularizer that combines low resolution hyperspectral degradation fidelity and panchromatic response fidelity with gradient adaptive structural regularization, thereby preserving structural discontinuities while suppressing spurious variations in homogeneous regions. Extensive experiments on Pavia, Chikusei, and Houston under reduced resolution protocols show that DIDM achieves the best performance across all evaluated metrics, while full resolution evaluation on FR1 yields the highest HQNR among the compared methods. These results demonstrate that internal dual modality prompting and panchromatic guided structural regularization provide an effective balance between spatial detail enhancement and spectral preservation.

Index Terms—Hyperspectral pansharpening, diffusion model, image prompting, cross-attention, structural regularization, zeroshot learning.

## I. INTRODUCTION

YPERSPECTRAL pansharpening refers to fusing a lowresolution hyperspectral (LRHS) observation and a high  
resolution panchromatic (PAN) image to reconstruct a high  
resolution hyperspectral (HRHS) image. This task is an impor  
tant preprocessing step for downstream hyperspectral image

interpretation, because the reconstructed HRHS is expected to contain both high-resolution spatial details and discriminative spectral information [1], [2]. Such data support intelligent perception tasks, including thematic classification and material discrimination, in which joint spatial–spectral modeling is essential [3], [4]. They are also valuable for urban and environmental analysis, such as land-cover mapping, where clear spatial boundaries and faithful spectral responses are critical [5], [6]. Unlike multispectral pansharpening, hyperspectral pansharpening must recover many contiguous spectral bands from observations with mismatched spatial and spectral resolutions. Its central challenge is therefore to enhance PAN-guided spatial details without distorting the spectral characteristics inherited from LRHS.

Existing methods can be divided into model-based and learning-based approaches. Model-based methods rely on explicit observation models and hand-crafted priors: GSA injects PAN details through adaptive component substitution [7], CNMF reconstructs HRHS through coupled matrix factorization [8], HySure formulates hyperspectral super-resolution as a convex subspace-regularized inverse problem [9], and TV-based methods impose gradient regularization to preserve structures and suppress artifacts [10], [11]. Although these methods are interpretable and training-free, their fixed priors have limited flexibility in modeling complex textures and nonlinear spatial–spectral interactions across diverse scenes. Learning-based methods improve representation capacity by learning nonlinear mappings from LRHS/PAN observations to HRHS. Supervised Convolutional Neural Network (CNN)- based, attention-based, and deep-unfolding networks achieve competitive performance under simulated training protocols [12]–[14], while deep-prior and spatial–spectral attention networks improve detail injection and spectral preservation [15]– [17]. Recent hyperspectral fusion networks employ selective re-learning to focus computation on spatially or spectrally degraded features [18]. However, paired HRHS labels are difficult to acquire in real satellite imaging, and training on synthetically degraded data remains dependent on the assumed degradation model [19]. Recent zero-shot methods reduce this dependence by adapting directly to the target observations. ZSL performs hyperspectral super-resolution without external paired supervision [20]; more recently, ρ-PNN introduces band-wise adaptation with hysteresis-based spectral-quality control [21], while Hipandas jointly addresses pandenoising and pansharpening through a zero-shot restoration framework with guided reconstruction networks and low-rank priors [22]. Nevertheless, without a sufficiently expressive image prior, per-image optimization can struggle to recover realistic highfrequency spatial structures.

![](images/4f98eabfc85396adfe5b6c08ac4184f93838f455e11b2e50833e0e6cf61e2e97.jpg)  
Fig. 1. Conceptual comparison between external objective-level guidance and the proposed internal feature-level prompting. The upper panel summarizes the common mechanism of PLRDiff, HIR-Diff, and DM-ZS under a unified LRHS/PAN reconstruction setting: observations and auxiliary priors define an external differentiable objective on the current HRHS estimate, whose gradient corrects the current diffusion state A<sub>t</sub> while the pre-trained diffusion prior remains frozen. The lower panel presents DIDM, which retains the standard reverse-diffusion process but encodes LRHS and PAN into spectral and spatial prompts, respectively, allowing both observations to participate directly in the evolution of internal diffusion features. The resulting spatial detail image Ab <sub>0</sub> is used to reconstruct HRHS, while the PAN-guided WPATV imposes structural regularization on the reconstruction process. To emphasize the difference between the two conditioning mechanisms, the LRHS degradation fidelity and the PAN response fidelity retained by DIDM are not explicitly shown.

Diffusion priors provide a possible way to strengthen perimage reconstruction. Diffusion models learn iterative denoising processes and serve as powerful image priors [23]–[26]. They have also shown strong potential in restoration and inverse problems, including super-resolution, inpainting, latent high-resolution synthesis, and posterior sampling [27]–[31]. Directly applying an available pre-trained diffusion model to full-band HRHS is nevertheless nontrivial, because most pretrained priors operate in RGB or another low-dimensional image domain rather than in a high-dimensional hyperspectral space. Representative hyperspectral diffusion methods therefore perform reverse sampling in a low-dimensional image space and subsequently recover the full spectral image. HIR-Diff combines a pre-trained diffusion prior with a low-rank representation for unsupervised hyperspectral restoration [32]. PLRDiff adapts low-rank diffusion modeling to hyperspectral pansharpening [33]. DM-ZS advances zero-shot pansharpening by coupling iterative zero-shot guidance with neural spatial-spectral decomposition (NSSD) reconstruction for a given LRHS/PAN pair [34]. These studies demonstrate the value of pre-trained diffusion priors when paired HRHS supervision is unavailable.

However, the success of this common pipeline does not resolve how task observations should interact with the frozen diffusion prior. As illustrated in Fig. 1(a), representative methods construct observation-fidelity terms and auxiliary priors outside the diffusion backbone. At each reverse step, these terms are evaluated on the reconstructed HRHS candidate, and their gradient is propagated back to correct the current diffusion state A . This mechanism effectively steers the sampling trajectory, but LRHS and PAN do not enter the pre-trained U-Net as condition tokens or feature-level inputs. The first unresolved problem is therefore the conditioning interface: how can the complementary spectral evidence of LRHS and spatial evidence of PAN participate directly in the internal feature evolution of the frozen prior, rather than only correcting its output trajectory through an external objective?

A second problem concerns the use of PAN structure during zero-shot reconstruction. LRHS degradation fidelity maintains spectral consistency under the spatial degradation model, and PAN response fidelity enforces consistency with the PAN observation. Both terms are indispensable and are retained in our formulation. However, they constrain the reconstructed image mainly in the observation space and do not explicitly distinguish boundaries from homogeneous regions when regularizing the spatial component. PAN gradients provide a physically meaningful cue for this distinction: object boundaries, road edges, and texture transitions indicate where spatial variations should be preserved, whereas homogeneous regions indicate where unnecessary fluctuations should be suppressed. The reconstruction objective should therefore complement observation fidelity with a PAN-aware structural prior.

To address these limitations, we propose dual-modality image-prompted diffusion model, termed DIDM, for hyperspectral pansharpening. As shown in Fig. 1(b), DIDM retains the standard reverse-diffusion process and a frozen remotesensing diffusion prior while redesigning how the observations interact with the backbone. LRHS and PAN are encoded into spectral and spatial condition tokens, respectively, and injected into intermediate diffusion features through cross-attention [35]. In this way, LRHS-derived spectral evidence and PANderived spatial structures participate in reverse sampling as internal feature-level prompts, producing a prompt-conditioned spatial detail image $\mathbf { \hat { A } } _ { 0 }$ that is mapped to HRHS by the NSSD reconstruction backend inherited from DM-ZS. DIDM further introduces a PAN-guided weighted pixel-aware total variation (WPATV) term adapted from XINet [36]. PANgradient-derived inverse weights impose edge-aware regularization on the spatial component, relaxing smoothing near PAN-aligned structures while suppressing spurious variations in homogeneous regions. The resulting objective jointly enforces spectral fidelity, PAN response consistency, and local structural regularity.

Our main contributions are as follows:

• We propose DIDM, a dual-modality image-prompted diffusion model for zero-shot hyperspectral pansharpening. It combines a frozen remote-sensing diffusion prior, modality-specific image prompts, and an NSSD reconstruction backend without requiring external paired HRHS samples.

• We establish an internal feature-level conditioning interface for the frozen diffusion prior. LRHS-derived spectral prompts and PAN-derived spatial prompts interact with intermediate diffusion features during reverse sampling, in contrast to representative methods that primarily steer the current diffusion state through an external differentiable objective.

• We introduce a PAN-guided spatial–spectral–structural objective. In addition to LRHS degradation fidelity and PAN response fidelity, WPATV imposes edge-aware regularization on the spatial component, preserving PANaligned discontinuities while suppressing fluctuations in homogeneous regions.

We provide a comprehensive evaluation under complementary reduced-resolution and full-resolution settings. Results on Pavia, Chikusei, and Houston demonstrate superior reference-based accuracy, while the FR1 sample shows favorable observation consistency. Ablation, efficiency, and diagnostic analyses further validate the dualmodality prompting, the PAN-guided structural regularization, and the multi-stage prompt injection.

## II. RELATED WORK

## A. Hyperspectral Pansharpening

Hyperspectral pansharpening has been extensively studied under model based and learning based paradigms [1], [19]. Model based methods reconstruct high resolution hyperspectral images by combining explicit observation models with handcrafted spatial or spectral priors. GSA estimates the relationship between spectral bands and PAN through regression [7], while MTF tailored multiscale approaches inject spatial details using sensor aware filtering [37]. CNMF couples hyperspectral and auxiliary observations through nonnegative matrix factorization [8], and HySure formulates hyperspectral super resolution through convex subspace regularization [9]. Sparse representation and Bayesian fusion exploit low dimensional spectral structures [38], [39], whereas TV based methods impose spatial or spectral gradient regularity to preserve boundaries and suppress artifacts [10], [11]. These methods are interpretable and require no external training data, but their manually specified priors may limit their ability to model complex spatial and spectral structures.

Learning-based approaches improve representation capacity through data-driven joint spatial–spectral modeling. CNN and residual architectures learn spatial-detail injection from simulated training pairs [40]–[43], while deep hyperspectral image fusion networks incorporate deep priors, spatial–spectral attention, external–internal attention, and U-shaped architectures to improve spectral preservation and spatial reconstruction [15]– [17]. Deep-unfolding and degradation-aware methods further improve model interpretability by integrating observation mod els into network optimization [12]–[14]. Recent studies also explore selective re-learning of degraded spatial–spectral features [18] and cross-scale designs for generalization across spatial resolutions [44]. To reduce dependence on paired HRHS supervision, zero-shot methods directly optimize taskspecific reconstruction from the observed LRHS and PAN pair. ZSL performs zero-shot hyperspectral super-resolution from the test observations [20], ρ-PNN introduces band-wise adaptation with hysteresis-based spectral-quality control [21], and Hipandas jointly addresses denoising and pansharpening through guided reconstruction networks and low-rank priors [22]. These methods avoid external paired training by adapting the reconstruction process directly to the target observations. In contrast, DIDM retains per-image optimization while exploiting a pre-trained remote-sensing diffusion prior through internal dual-modality conditioning and PAN-guided structural regularization.

## B. Diffusion Priors and Conditioning Interfaces

Diffusion models serve as powerful generative priors for image synthesis and inverse problems. DDPM and DDIM establish the fundamental iterative denoising process [23], [24], while subsequent studies further improve generation quality and conditional sampling [25], [26]. Diffusion priors are also widely adopted for various image restoration tasks. SR3 performs image super-resolution through iterative refinement, RePaint applies diffusion sampling to image inpainting, and LDM transfers generation into a compact latent space [27]– [29]. DDRM and DPS further incorporate observation models into diffusion-based inverse reconstruction [30], [31]. This capability is particularly attractive for hyperspectral pansharpening applications, where paired full-resolution supervision is difficult to acquire.

Existing hyperspectral diffusion methods mainly adapt lowdimensional image priors through reverse sampling followed by spatial–spectral reconstruction. HIR-Diff combines diffusion sampling with a low-rank spatial–spectral representation for unsupervised hyperspectral restoration, PLRDiff extends low-rank diffusion modeling to hyperspectral pansharpening [33], and DM-ZS combines iterative zero-shot guidance with NSSD reconstruction for each LRHS/PAN pair [34]. Although their objectives and reconstruction operators differ, these approaches mainly enforce observation consistency through external differentiable objectives whose gradients correct the current diffusion state. The observations therefore remain weakly coupled with the internal diffusion features. DIDM complements this paradigm by retaining imagelevel fidelity for reconstruction supervision while introducing LRHS/PAN information as internal feature-level prompts.

A related line of research studies how explicit conditions control pretrained diffusion models. Textual Inversion and DreamBooth introduce concept-specific conditioning through learned embeddings or parameter adaptation [45], [46], while GLIGEN, ControlNet, and T2I-Adapter inject external visual conditions through trainable modules [47]–[49]. Recent conditional diffusion frameworks further demonstrate the benefit of structured visual and motion priors for controllable generation, including pose-guided person synthesis [50], customizable virtual dressing [51], fine-grained garment generation [52], and long-term talking-face generation with motion priors [53]. However, these methods mainly target natural-image or video synthesis and rely on semantic, geometric, or motion conditions. Hyperspectral pansharpening instead involves two physically coupled observations carrying complementary spectral and spatial information. DIDM therefore develops a taskspecific conditioning interface in which LRHS provides spectral prompts and PAN provides spatial prompts through crossattention, while PAN-derived structural regularization further constrains zero-shot reconstruction.

## III. PROPOSED METHOD

## A. Problem Formulation

Let $\textbf { Y } \in \mathbb { R } ^ { B \times h \times w }$ denote the observed LRHS with B spectral bands, where h and w are its spatial height and width, respectively, and let $\mathbf { P } \in \mathbb { R } ^ { 1 \times H \times W }$ denote the observed PAN image, where H and W are the corresponding high-resolution spatial height and width. Let $\textbf { X } \in \overset { \cdot } { \mathbb { R } } ^ { B \times \overset { \smile } { H } \times \overset { \smile } { W } }$ denote the latent HRHS, and let $\widehat { \mathbf { X } } \in \mathbb { R } ^ { B \times H \times W }$ denote the reconstructed HRHS. Here, $H = s h$ and $W = s w$ , where s is the spatial scale factor. Following the common observation model,

$$
\mathbf { Y } \approx { \mathcal { D } } ( \mathbf { X } ) , \quad \mathbf { P } \approx { \mathcal { R } } ( \mathbf { X } ) ,\tag{1}
$$

where $\mathcal { D } ( \cdot )$ denotes spatial degradation, and $\mathcal { R } ( \cdot )$ denotes the spectral response from HRHS to PAN.

DIDM retains the standard reverse-diffusion process and uses a frozen remote-sensing pre-trained diffusion backbone as the spatial detail generation prior. The method consists of three stages. First, LRHS and PAN are encoded into spectral and spatial tokens, respectively, and introduced into intermediate diffusion features through Dual-Prompt Cross-Attention. Second, the prompt-conditioned diffusion backbone estimates a three-channel spatial detail image $\widehat { \mathbf { A } } _ { 0 } \in \mathbb { R } ^ { 3 \times H \times W }$ , hereafter referred to as the spatial detail image. Third, the spatial detail image and LRHS are transformed into HRHS through an NSSD-based spatial–spectral reconstruction backend. The optimization retains the LRHS degradation fidelity and the PAN response fidelity and further incorporates the PAN-guided WPATV on the spatial branch. In this way, the method combines internal feature-level prompting with a spatial–spectral– structural reconstruction objective while keeping the original pre-trained diffusion parameters fixed.

![](images/61953e98866b02a95a8280d56da4b7ddbd818737e35502df4b1e3c468be9e7f4.jpg)  
Fig. 2. Dual-modality prompt injection in DIDM. LRHS and PAN are encoded into spectral and spatial tokens, respectively. At the representative Down, Mid, and Up stages, local diffusion features provide the queries, while the two token groups provide modality-specific keys and values to trainable Cross Attention blocks. The resulting responses are injected into the corresponding frozen diffusion features, and the prompt-conditioned diffusion prior estimates the spatial detail image $\widehat { \mathbf { A } } _ { 0 }$

The reconstruction backend follows the formulation adopted in DM-ZS so that the contribution of the conditioning interface can be isolated from changes in the subsequent HRHS reconstruction. Unlike external objective-level guidance, the proposed interface allows LRHS-derived spectral evidence and PAN-derived spatial evidence to participate directly in the evolution of intermediate diffusion features. The following subsections describe the dual-modality prompt injection, the prompt-conditioned detail generation, the spatial–spectral reconstruction, and the optimization objective.

## B. Dual-Modality Image Prompt Injection

DIDM converts LRHS and PAN into modality-specific condition tokens. LRHS provides spectral cues, whereas PAN provides high-resolution spatial structures. The token generation is formulated as

$$
{ \bf T } _ { \mathrm { s p e } } = \mathcal { E } _ { \mathrm { s p e } } ( { \bf Y } ) , \quad { \bf T } _ { \mathrm { s p a } } = \mathcal { E } _ { \mathrm { s p a } } ( { \bf P } ) ,\tag{2}
$$

where $\mathcal { E } _ { \mathrm { s p e } } ( \cdot )$ and $\mathcal { E } _ { \mathrm { s p a } } ( \cdot )$ denote the spectral and spatial prompt encoders, respectively. The resulting tokens are $\mathbf { T } _ { \mathrm { s p e } } \in$ $\mathbb { R } ^ { N _ { s } \times C }$ and $\mathbf { T } _ { \mathrm { s p a } } ~ \in ~ \mathbb { R } ^ { N _ { p } \times C }$ , where $N _ { s }$ and $N _ { p }$ are the numbers of spectral and spatial tokens, and $C$ is the token dimension. We further denote by $\alpha \in [ 0 , 1 ]$ the spectral-token ratio controlling their relative spectral–spatial composition; $\alpha = 0$ and $\alpha = 1$ correspond to spatial-only and spectralonly prompting, respectively, while intermediate values retain both modalities at different proportions. Both encoders are lightweight two-layer convolutional networks. This design projects the two observations into a common condition space without excessive learnable capacity, which is important for per-image zero-shot optimization.

Fig. 2 illustrates the prompt injection process. The same spectral and spatial token groups are supplied to representative Cross Attention blocks in the Down, Mid, and Up stages of the frozen diffusion prior. Each illustrated block receives both modalities, and its output is injected into the corresponding diffusion features. The figure abstracts the internal attention computation to maintain readability at single-column width; the complete query, key, value, attention response, and residual feature update are defined below.

Let $\mathbf { F } _ { t } ^ { l } \in \mathbb { R } ^ { N _ { l } \times C _ { l } }$ denote the diffusion feature at reverse step t and injection layer $l ,$ where $N _ { l }$ is the number of flattened spatial locations and $C _ { l }$ is the feature dimension at that layer. For $m \ \in \ \{ \mathrm { s p e } , \mathrm { s p a } \}$ , which indexes the spectral or spatial prompt branch, the prompt-injection computation is given by

$$
\mathbf { Q } _ { t } ^ { l } = \mathbf { F } _ { t } ^ { l } \mathbf { W } _ { Q } ^ { l } , \quad \mathbf { K } _ { m } ^ { l } = \mathbf { T } _ { m } \mathbf { W } _ { K , m } ^ { l } , \quad \mathbf { V } _ { m } ^ { l } = \mathbf { T } _ { m } \mathbf { W } _ { V , m } ^ { l } ,\tag{3}
$$

$$
\begin{array} { r } { { \bf C } _ { m , t } ^ { l } = \mathrm { A t t n } ( { \bf Q } _ { t } ^ { l } , { \bf K } _ { m } ^ { l } , { \bf V } _ { m } ^ { l } ) , } \end{array}\tag{4}
$$

$$
\operatorname { A t t n } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) = \operatorname { s o f t m a x } ( \mathbf { Q } \mathbf { K } ^ { \top } / \sqrt { d _ { l } } ) \mathbf { V } ,\tag{5}
$$

$$
\widetilde { \mathbf { F } } _ { t } ^ { l } = \mathrm { S A } ( \mathbf { F } _ { t } ^ { l } ) + \sum _ { m \in \{ \mathrm { s p e } , \mathrm { s p a } \} } \eta _ { m } ^ { l } \mathbf { C } _ { m , t } ^ { l } ,\tag{6}
$$

where $\mathbf { Q } _ { t } ^ { l }$ is the query projected from the diffusion feature, ${ \bf K } _ { m } ^ { l }$ and $\mathbf { V } _ { m } ^ { l }$ are the key and value projected from modalitym tokens, and $\mathbf { C } _ { m , t } ^ { l }$ is the corresponding cross-attention response. The matrices $\mathbf { W } _ { Q } ^ { l } , \mathbf { W } _ { K , m } ^ { l }$ , and $\mathbf { W } _ { V , m } ^ { l }$ denote the query, key, and value projections, respectively, and $d _ { l }$ is the attention dimension. Moreover, $\mathrm { S A } ( \cdot )$ denotes the original selfattention operation, $\eta _ { m } ^ { l }$ is the modality-balancing coefficient at layer $l ,$ and $\widetilde { \mathbf { F } } _ { t } ^ { l } \in \mathbb { R } ^ { \dot { N } _ { l } \times C _ { l } }$ is the resulting prompt-conditioned feature. Thus, the diffusion feature supplies the query, while the LRHS- and PAN-derived tokens provide complementary conditioning contexts that participate in reverse diffusion without modifying the pre-trained backbone.

## C. Prompt-Conditioned Detail Generation and Spatial– Spectral Reconstruction

Let $\mathbf { A } _ { 0 } \in \mathbb { R } ^ { 3 \times H \times W }$ denote the spatial detail image modeled by the diffusion prior, and let $\mathbf { A } _ { t } \in \mathbb { R } ^ { 3 \times H \times W }$ denote its noisy state at timestep t. Conditioned on the two token groups, the denoising network predicts the noise and derives the clean estimate as

$$
\begin{array} { r } { \mathbf { \epsilon } _ { t } = \epsilon _ { \theta } \big ( \mathbf { A } _ { t } , t , \mathbf { T } _ { \mathrm { s p e } } , \mathbf { T } _ { \mathrm { s p a } } \big ) , } \end{array}\tag{7}
$$

$$
\widehat { \mathbf { A } } _ { 0 } ^ { t } = \frac { \mathbf { A } _ { t } - \sqrt { 1 - \bar { \gamma } _ { t } } \mathbf { \epsilon } \mathbf { \epsilon } _ { t } } { \sqrt { \bar { \gamma } _ { t } } } ,\tag{8}
$$

where $\epsilon _ { \theta } ( \cdot )$ is the denoising network with fixed pre-trained parameters $\theta , \epsilon _ { t } \in \mathbb { R } ^ { 3 \times H \times W }$ is the predicted noise, $\bar { \gamma } _ { t }$ is the cumulative noise-scheduling coefficient, and $\widehat { \mathbf { A } } _ { 0 } ^ { t } \in \mathbb { R } ^ { 3 \times H \times W }$ is the clean spatial-detail estimate obtained at step t. The final estimate after reverse diffusion is denoted by $\widehat { \mathbf { A } } _ { 0 }$

The spatial detail image provides a compact high-resolution structural representation, whereas the target HRHS contains B spectral bands. We therefore retain the NSSD-based reconstruction backend from DM-ZS to map the spatial detail image and LRHS spectral information to the HRHS space. Following DM-ZS [34], the NSSD reconstruction adopts a low-dimensional subspace with rank r. Let $\mathcal { M } _ { \phi } ( \cdot )$ denote the complete reconstruction operator with trainable parameters ϕ. It contains a U-shaped spatial branch and a spectral mapping branch and reconstructs HRHS as

$$
\widehat { \mathbf { X } } = \mathcal { M } _ { \phi } ( \widehat { \mathbf { A } } _ { 0 } , \mathbf { Y } ) ,\tag{9}
$$

while the output of the spatial branch used for the PAN-guided structural regularization is denoted by

$$
\widehat { \mathbf { S } } = { \cal S } _ { \phi } ( \widehat { \mathbf { A } } _ { 0 } ) ,\tag{10}
$$

where $ { \boldsymbol { S } } _ { \phi } ( \cdot )$ is the spatial branch of $\mathcal { M } _ { \phi } ( \cdot ) , \widehat { \mathbf { S } } \in \mathbb { R } ^ { C _ { s } \times H \times W }$ is its output, and $C _ { s }$ is the number of spatial-feature channels. The single-channel PAN-derived weight defined below is shared across these channels. Retaining this backend preserves a clear connection with prior diffusion-based hyperspectral pansharpening and allows DIDM to focus on improving the preceding observation-to-diffusion conditioning interface.

$$
D . \ P A N { \cdot } G u i d e d \ S p a t i a l { - } S p e c t r a l { - } S t r u c t u r a l \ O b j e c t i \nu e
$$

The objective contains three complementary terms, whose connections to the reconstruction process are illustrated in Fig. 3. The spatial detail image and LRHS are processed by the spatial and spectral branches, respectively, and their representations are fused to reconstruct $\widehat { \mathbf { X } } .$ . The blue and orange dashed paths denote LRHS degradation fidelity and PAN response fidelity, whereas the red path indicates that PAN-guided WPATV acts on the spatial gradients of Sb. For clarity, the figure abstracts the degradation operator, response operator, and construction of the PAN-derived weight, which are defined below.

The LRHS degradation fidelity and the PAN response fidelity are respectively formulated as

$$
\mathcal { L } _ { \mathrm { s p e } } = \| \mathcal { D } ( \widehat { \mathbf { X } } ) - \mathbf { Y } \| _ { 1 } ,\tag{11}
$$

$$
\mathcal { L } _ { \mathrm { { s p a } } } = \| \mathcal { R } ( \widehat { \mathbf { X } } ) - \mathbf { P } \| _ { 1 } ,\tag{12}
$$

where $\mathcal { D } ( \cdot )$ and $\mathcal { R } ( \cdot )$ are defined in Eq. (1), and $\| \cdot \| _ { 1 }$ denotes the sum of absolute residual values. The terms $\mathcal { L } _ { \mathrm { s p e } }$ and $\mathcal { L } _ { \mathrm { s p a } }$ therefore measure LRHS degradation fidelity and PAN response fidelity, respectively.

Although these two fidelity terms are necessary for observation consistency, they mainly constrain the reconstructed HRHS at the image level and do not explicitly impose a pixel-wise edge-aware prior on the spatial branch. PAN contains high-resolution edges, object boundaries, and texture transitions, which indicate where spatial variations should be retained. We therefore adapt the WPATV principle to construct a PAN-guided structural regularizer.

![](images/759c556077fb7e26cd2494d9d0048a7c6b16bd53f6d9f060609668c1b803961c.jpg)  
Fig. 3. Spatial–spectral reconstruction and objective attachment in DIDM. The spatial detail image $\widehat { \mathbf { A } } _ { 0 }$ and LRHS Y are processed by the spatial and spectral branches, respectively, and their representations are fused to reconstruct HRHS $\widehat { \mathbf { x } } .$ The blue and orange dashed paths denote LRHS degradation fidelity $\mathcal { L } _ { \mathrm { s p e } }$ and PAN response fidelity $\mathcal { L } _ { \mathrm { { s p a } } }$ , respectively. The single-channel PAN-derived weight W modulates the gradients of the multi-channel spatial component Sb, yielding the PAN-guided WPATV term $\mathcal { L } _ { \mathrm { s t r } }$

The PAN gradient map, clipped gradient map, pixel-aware weight, and structural regularization term are defined as

$$
\mathbf { G } = \sum _ { c } ( | \nabla _ { x } \mathbf { P } _ { c } | + | \nabla _ { y } \mathbf { P } _ { c } | ) , \quad \bar { \mathbf { G } } = \operatorname* { m i n } ( \mathbf { G } , Q _ { p } ( \mathbf { G } ) ) ,
$$

$$
\mathbf { W } = g ( | \nabla \mathbf { P } | ) ,\tag{13}
$$

(14)

$$
\mathbf { W } = 1 - \frac { \bar { \mathbf { G } } - g _ { \operatorname* { m i n } } } { g _ { \operatorname* { m a x } } - g _ { \operatorname* { m i n } } + \epsilon } ,\tag{15}
$$

$$
\mathcal { L } _ { \mathrm { s t r } } = \operatorname { m e a n } ( \mathbf { W } \odot | \nabla _ { x } \widehat { \mathbf { S } } | ) + \operatorname { m e a n } ( \mathbf { W } \odot | \nabla _ { y } \widehat { \mathbf { S } } | ) ,\tag{16}
$$

where c indexes the PAN channels and $\mathbf { P } _ { c }$ is the cth channel; for the single-channel PAN used here, the summation contains one term. The operators $\nabla _ { x }$ and $\nabla _ { y }$ denote horizontal and vertical finite differences, respectively, and |·| denotes elementwise absolute value. Accordingly, $\textbf { G } \in \mathbf { \mathbb { R } } ^ { 1 \times H \times W }$ is the anisotropic PAN gradient-response map. Here, $p \in \mathsf { \Gamma } ( 0 , 1 )$ denotes the winsorization percentile expressed as a quantile level, $Q _ { p } ( \mathbf G )$ gives the corresponding clipping threshold, and G<sup>¯</sup> is obtained by element-wise clipping. In Eq. (14), |∇P| is shorthand for G, and $g ( \cdot )$ denotes the normalized inversegradient mapping defined in Eq. (15). Here, g<sub>min</sub> and g<sub>max</sub> are the minimum and maximum of $\bar { \mathbf { G } } , \epsilon = 1 0 ^ { - 8 }$ is a fixed constant for numerical stability, and $\mathbf { W } \in \mathbb { R } ^ { 1 \times H \times W }$ is the resulting weight map. In Eq. (16), ⊙ denotes element-wise multiplication, mean(·) averages all entries, and $\mathcal { L } _ { \mathrm { s t r } }$ is the PAN-guided WPATV loss. The finite-difference maps retain the original spatial size, and W is broadcast across the $C _ { s }$ channels of Sb. Thus, smoothing is relaxed near strong PAN gradients and strengthened in homogeneous regions.

```latex
Algorithm 1 Optimization procedure of DIDM
Require: LRHS Y; PAN $\mathbf { P } ;$ operators D and $\mathcal { R } ;$ fixed diffu
sion backbone $\epsilon _ { \theta }$ and reverse-diffusion schedule; spectral
token ratio $\alpha ;$ reconstruction rank $r ;$ WPATV parameters
$p ;$ loss weights $\lambda _ { \mathrm { s p e } } , \lambda _ { \mathrm { s p a } } ,$ and $\lambda _ { \mathrm { s t r } } ;$ maximum optimiza
tion iterations $K$
1: Initialize the learnable parameter set $\Theta$ under the pre
scribed α and $r ;$ keep θ fixed.
2: Construct the PAN-guided weight W from P using
Eqs. (13)–(15).
3: for $k = 1$ to K do
4: Generate $\mathbf { T } _ { \mathrm { s p e } }$ and $\mathbf { T } _ { \mathrm { s p a } }$ using Eq. (2).
5: Perform prompt-conditioned reverse diffusion, applying
Eqs. (3)–(6) at the injected layers, to obtain $\widehat { \mathbf { A } } _ { 0 } .$
6: Obtain $\widehat { \mathbf { X } }$ and $\widehat { \mathbf { S } }$ using Eqs. (9) and (10).
7: Compute $\mathcal { L } _ { \mathrm { s p e } } , \mathcal { L } _ { \mathrm { s p a } } ,$ and $\mathcal { L } _ { \mathrm { s t r } }$ using Eqs. (11), (12),
and (16), respectively.
8: Update Θ by minimizing Eq. (17); keep θ fixed.
9: end for
10: return Reconstructed HRHS ${ \widehat { \mathbf { X } } } .$
```

The overall objective is

$$
\mathcal { L } = \lambda _ { \mathrm { s p e } } \mathcal { L } _ { \mathrm { s p e } } + \lambda _ { \mathrm { s p a } } \mathcal { L } _ { \mathrm { s p a } } + \lambda _ { \mathrm { s t r } } \mathcal { L } _ { \mathrm { s t r } } ,\tag{17}
$$

where $\mathcal { L }$ is the overall objective, and $\lambda _ { \mathrm { s p e } } , \lambda _ { \mathrm { s p a } } ,$ and $\lambda _ { \mathrm { s t r } }$ balance the three loss terms. This spatial–spectral–structural objective extends observation fidelity with PAN-guided local regularization, thereby encouraging spectral consistency, PAN response consistency, and PAN-aligned structural preservation in the reconstructed HRHS.

## E. Optimization Procedure

Algorithm 1 summarizes the zero-shot optimization procedure. For each LRHS/PAN pair, the pre-trained diffusion parameters θ remain fixed during per-pair adaptation. Let $\Theta \ = \ \{ \theta _ { \mathrm { s p e } } , \theta _ { \mathrm { s p a } } , \theta _ { \mathrm { i n j } } , \phi \}$ collect all learnable parameters, where $\theta _ { \mathrm { s p e } }$ and $\theta _ { \mathrm { s p a } }$ parameterize the two prompt encoders and $\theta _ { \mathrm { i n j } }$ contains the parameters of the introduced promptinjection modules, including the coefficients $\eta _ { m } ^ { l }$ . Thus, $\eta _ { m } ^ { l }$ is optimized as part of the model rather than supplied as an external input. The fixed settings required by the procedure include the reverse-diffusion schedule, the spectral-token ratio $\alpha ,$ the reconstruction rank $r ,$ the WPATV parameters $p ,$ the three loss weights, and the optimization stopping criterion. The operators $Q _ { p } ( \cdot )$ and $g ( \cdot )$ are deterministic mappings defined in Eqs. (13)–(15); once $p$ and ϵ are specified, they introduce no additional free input.

## IV. EXPERIMENTS AND ANALYSIS

## A. Experimental Settings

Datasets and Protocols. We evaluate DIDM under two complementary protocols commonly adopted for hyperspectral pansharpening: reduced-resolution (RR) reference-based assessment and full-resolution (FR) no-reference assessment on real multiresolution observations. The RR experiments are conducted on three public hyperspectral datasets: Pavia,<sup>1</sup> Chikusei,<sup>2</sup> and Houston.<sup>3</sup> Following Wald’s protocol [54], each original hyperspectral crop is treated as the reference HRHS image, and the corresponding LRHS observation is generated by applying a $9 \times 9$ Gaussian blur to each spectral band, followed by fourfold bicubic downsampling. Because paired PAN observations are unavailable, the PAN image is simulated by uniformly averaging the spectral bands within the available visible and near-infrared (VIS–NIR) range; bands 16–81 under one-based indexing are used for Chikusei, whereas all retained bands are used for Pavia and Houston. The FR assessment is performed on the released FR1 fullresolution sample from HyperPanCollection<sup>4</sup> [55], using its real PAN and hyperspectral observations at their native resolutions without constructing a reference HRHS image.

(1) Pavia is an urban hyperspectral dataset acquired by the Reflective Optics System Imaging Spectrometer. We use eight $2 5 6 \times 2 5 6$ crops with 93 spectral bands. The scenes contain representative urban materials and structures, including buildings, roads, shadows, and vegetation, and are therefore used to assess whether PAN-aligned geometric details, such as roof contours and road boundaries, can be recovered without compromising spectral fidelity.

(2) Chikusei was acquired over mixed agricultural and urban areas in Chikusei, Ibaraki, Japan. We use ten cropped samples of size 256×256, each comprising 128 spectral bands. Its fine field boundaries, agricultural textures, and narrow linear structures provide a demanding test of local-detail reconstruction and high-dimensional spectral consistency across heterogeneous land-cover regions.

(3) Houston is provided by the Hyperspectral Image Analysis Laboratory at the University of Houston. We use ten cropped samples of size 256 × 256 with 144 spectral bands. The dataset contains dense man-made structures, complex object boundaries, and pronounced spectral variability, thereby challenging both spatial reconstruction and spectral preservation. Owing to this complexity and its high spectral dimensionality, Houston is also adopted as the representative dataset for the ablation and sensitivity studies.

(4) FR1 provides a real multiresolution PAN–hyperspectral observation pair for complementary FR assessment. The released sample used in our experiments contains a PAN image of size $2 4 0 \times 2 4 0$ and a corresponding LRHS observation of size $4 0 \times 4 0$ with 69 spectral bands, resulting in a spatialresolution ratio of six. Because a ground-truth HRHS image is unavailable, FR1 is used solely to examine no-reference observation consistency and visual behavior at the native resolution. All datasets used in this study are publicly available through the links provided above.

Compared Methods. We compare DIDM with nine representative competing methods spanning three major methodological families commonly considered in hyperspectral pansharpening. The classical model-based methods include GSA [7], CNMF [8], and TV [10], representing component-substitution, matrix-factorization, and variationalregularization approaches, respectively. The direct zero-shot reconstruction methods include ZSL [20], ρ-PNN [21], and Hipandas [22], which adapt task-specific reconstruction directly from the available observations without external paired supervision. The diffusion-prior methods comprise PLRDiff [33], HIR-Diff [32], and DM-ZS [34]. This selection provides broad coverage of established model-based approaches, recent zero-shot reconstruction methods, and methods exploiting diffusion priors, with DM-ZS serving as the closest baseline to DIDM. The parameters of each competing method are set according to the configurations recommended in the corresponding publication to ensure a consistent comparison.

Evaluation Metrics. For RR assessment, six full-reference metrics are used for quantitative evaluation: peak signal-tonoise ratio (PSNR) [54], root mean square error (RMSE) [54], spectral angle mapper (SAM) [56], erreur relative globale adimensionnelle de synthese (ERGAS) [54], structural similar-\` ity (SSIM) [57], and correlation coefficient (CC) [2]. PSNR and RMSE assess pixel-wise fidelity, SAM measures spectral distortion, ERGAS reflects global relative error, and SSIM and CC characterize structural similarity and correlation. Higher PSNR, SSIM, and CC and lower RMSE, SAM, and ERGAS indicate better performance. For FR assessment, we report the spectral distortion index $D _ { \lambda }$ , spatial distortion index $D _ { S }$ , and hybrid quality with no reference (HQNR) [55], [58]. Lower $D _ { \lambda }$ and $D _ { S }$ indicate smaller spectral and spatial distortions, whereas higher HQNR indicates better overall no-reference quality. Since FR1 has no reference HRHS, full-reference metrics are not computed. RR results are averaged over all crops of each dataset, while $D _ { \lambda } , D _ { S }$ , and HQNR are reported for the single FR1 sample.

Implementation Details. DIDM is implemented in PyTorch, and all experiments are conducted on a single NVIDIA GeForce RTX 4090D GPU. The configurations are summarized as follows for reproducibility: 1) The remote-sensing pretrained diffusion backbone is kept frozen except for the crossattention layers introduced for prompt injection; the prompt modules and spatial–spectral reconstruction module are jointly optimized. 2) The number of reverse diffusion steps is set to $T = 2 0 0$ , and the NSSD reconstruction rank $r$ is set to 20, 40, 40, and 20 for Pavia, Chikusei, Houston, and FR1, respectively. 3) The spectral-token ratio α is fixed at the dataset level, with $\alpha \ : = \ : 0 . 3$ for Pavia and $\alpha \ : = \ : 0 . 5$ for Chikusei, Houston, and FR1. 4) The loss weights are set to $\lambda _ { \mathrm { s p e } } = 1$ $\lambda _ { \mathrm { s p a } } = 1$ , and $\lambda _ { \mathrm { s t r } } ~ = ~ 5 0$ , and the WPATV winsorization percentile $p$ is set to 0.995. All learnable parameters are optimized using Adam with a fixed learning rate of $1 \times 1 0 ^ { - 3 }$ 5) DIDM follows instance-wise zero-shot optimization with a batch size of one, and each LRHS/PAN pair is optimized independently for 25,000 iterations. 6) FR1 follows the same dataset-level configuration strategy as the RR experiments, without additional FR-specific ablations or exhaustive hyperparameter search. Unless otherwise stated, these settings are used throughout the experiments.

TABLE I  
QUANTITATIVE COMPARISON ON THE PAVIA DATASET. THE BEST AND SECOND-BEST RESULTS ARE HIGHLIGHTED IN BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td>Method</td><td>PSNR ↑</td><td>RMSE ↓</td><td>SAM↓</td><td>ERGAS↓</td><td>SSIM ↑</td><td>CC ↑</td></tr><tr><td>Bicubic</td><td>26.833</td><td>4.628</td><td>7.238</td><td>8.392</td><td>0.673</td><td>0.895</td></tr><tr><td>GSA</td><td>25.052</td><td>5.819</td><td>11.441</td><td>12.763</td><td>0.776</td><td>0.972</td></tr><tr><td>CNMF</td><td>25.369</td><td>5.671</td><td>7.158</td><td>9.689</td><td>0.882</td><td>0.975</td></tr><tr><td>TV</td><td>22.172</td><td>8.010</td><td>8.228</td><td>14.646</td><td>0.739</td><td>0.949</td></tr><tr><td>ZSL</td><td>32.168</td><td>2.494</td><td>7.679</td><td>4.753</td><td>0.901</td><td>0.970</td></tr><tr><td>ρ-PNN</td><td>32.779</td><td>2.333</td><td>7.201</td><td>4.418</td><td>0.915</td><td>0.974</td></tr><tr><td>Hipandas</td><td>30.181</td><td>3.147</td><td>8.649</td><td>5.774</td><td>0.853</td><td>0.954</td></tr><tr><td>PLRDiff</td><td>33.527</td><td>2.133</td><td>8.389</td><td>4.316</td><td>0.916</td><td>0.978</td></tr><tr><td>HIR-Diff</td><td>27.509</td><td>4.289</td><td>7.878</td><td>7.878</td><td>0.699</td><td>0.908</td></tr><tr><td>DM-ZS</td><td>33.766</td><td>2.080</td><td>7.295</td><td>4.115</td><td>0.918</td><td>0.979</td></tr><tr><td>DIDM</td><td>34.932</td><td>1.819</td><td>6.283</td><td>3.640</td><td>0.937</td><td>0.984</td></tr></table>

TABLE II

QUANTITATIVE COMPARISON ON THE CHIKUSEI DATASET. THE BEST AND SECOND-BEST RESULTS ARE HIGHLIGHTED IN BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td>Method</td><td>PSNR ↑</td><td>RMSE ↓</td><td>SAM↓</td><td>ERGAS ↓</td><td>SSIM ↑</td><td>CC ↑</td></tr><tr><td>Bicubic</td><td>36.056</td><td>1.588</td><td>3.206</td><td>7.063</td><td>0.893</td><td>0.976</td></tr><tr><td>GSA</td><td>22.541</td><td>7.874</td><td>11.539</td><td>23.909</td><td>0.740</td><td>0.990</td></tr><tr><td>CNMF</td><td>18.707</td><td>12.130</td><td>3.677</td><td>13.438</td><td>0.795</td><td>0.981</td></tr><tr><td>TV</td><td>21.730</td><td>8.692</td><td>5.547</td><td>21.570</td><td>0.692</td><td>0.984</td></tr><tr><td>ZSL</td><td>36.684</td><td>1.477</td><td>3.409</td><td>6.421</td><td>0.904</td><td>0.979</td></tr><tr><td>ρ-PNN</td><td>38.191</td><td>1.245</td><td>3.183</td><td>5.043</td><td>0.929</td><td>0.985</td></tr><tr><td>Hipandas</td><td>36.489</td><td>1.510</td><td>4.268</td><td>7.980</td><td>0.906</td><td>0.978</td></tr><tr><td>PLRDiff</td><td>39.029</td><td>1.127</td><td>3.453</td><td>4.753</td><td>0.938</td><td>0.988</td></tr><tr><td>HIR-Diff</td><td>36.277</td><td>1.548</td><td>4.031</td><td>7.685</td><td>0.886</td><td>0.977</td></tr><tr><td>DM-ZS</td><td>37.700</td><td>1.316</td><td>3.484</td><td>4.616</td><td>0.921</td><td>0.984</td></tr><tr><td>DIDM</td><td>39.797</td><td>1.031</td><td>2.740</td><td>3.629</td><td>0.949</td><td>0.990</td></tr></table>

## B. Comparison with State-of-the-Art Methods

To provide a comprehensive evaluation, we compare DIDM with representative methods under two complementary protocols. The RR assessment quantifies reconstruction fidelity on Pavia, Chikusei, and Houston using reference HRHS images, whereas the FR assessment evaluates fusion behavior on real multiresolution FR1 observations using $D _ { \lambda } , D _ { S }$ , HQNR, and qualitative comparison. Bicubic denotes the LRHS upsampled to the target spatial resolution by bicubic interpolation and is included as a non-fusion reference. These two protocols jointly characterize reference-based reconstruction accuracy and native-resolution fusion quality.

1) Reduced-Resolution Assessment: Tables I–III report the quantitative results on Pavia, Chikusei, and Houston, respectively. Figs. 4–6 present the corresponding pseudo-color reconstructions and error maps. Together, these results evaluate pixel-level reconstruction fidelity, spectral preservation, structural similarity, and spectral–spatial correlation under the RR reference-based protocol.

On Pavia, DIDM achieves the best result for every reported metric. Compared with DM-ZS, which remains the strongest competitor in overall reconstruction accuracy, DIDM improves PSNR from 33.766 to 34.932 dB and reduces RMSE from 2.080 to 1.819. Among the direct zero-shot reconstruction methods, ρ-PNN performs best overall with 32.779 dB PSNR and 7.201 SAM, whereas DIDM further improves these values to 34.932 dB and 6.283. Bicubic preserves a relatively low SAM but has substantially lower PSNR and SSIM, reflecting the absence of high-resolution spatial detail. The consistent gains across error-, spectral-, and structure-oriented metrics indicate that DIDM recovers urban structures without sacrificing spectral fidelity.

TABLE III  
QUANTITATIVE COMPARISON ON THE HOUSTON DATASET. THE BEST AND SECOND-BEST RESULTS ARE HIGHLIGHTED IN BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td>Method</td><td>PSNR ↑</td><td>RMSE↓</td><td>SAM↓</td><td>ERGAS↓</td><td>SSIM ↑</td><td>CC ↑</td></tr><tr><td>Bicubic</td><td>31.788</td><td>2.679</td><td>6.140</td><td>6.512</td><td>0.789</td><td>0.916</td></tr><tr><td>GSA</td><td>30.745</td><td>3.028</td><td>9.130</td><td>10.188</td><td>0.848</td><td>0.975</td></tr><tr><td>CNMF</td><td>24.537</td><td>7.815</td><td>6.400</td><td>22.704</td><td>0.767</td><td>0.968</td></tr><tr><td>TV</td><td>20.267</td><td>10.327</td><td>6.597</td><td>30.731</td><td>0.653</td><td>0.945</td></tr><tr><td>ZSL</td><td>35.409</td><td>1.760</td><td>6.340</td><td>4.334</td><td>0.887</td><td>0.963</td></tr><tr><td>ρ-PNN</td><td>37.396</td><td>1.418</td><td>6.077</td><td>3.495</td><td>0.917</td><td>0.977</td></tr><tr><td>Hipandas</td><td>34.173</td><td>2.046</td><td>7.425</td><td>5.338</td><td>0.862</td><td>0.951</td></tr><tr><td>PLRDiff</td><td>38.106</td><td>1.294</td><td>6.265</td><td>3.445</td><td>0.931</td><td>0.980</td></tr><tr><td>HIR-Diff</td><td>32.045</td><td>2.587</td><td>6.891</td><td>6.371</td><td>0.788</td><td>0.920</td></tr><tr><td>DM-ZS</td><td>38.435</td><td>1.247</td><td>5.539</td><td>3.178</td><td>0.932</td><td>0.982</td></tr><tr><td>DIDM</td><td>39.172</td><td>1.152</td><td>5.073</td><td>2.896</td><td>0.939</td><td>0.985</td></tr></table>

On Chikusei, DIDM obtains the best PSNR, RMSE, SAM, ERGAS, and SSIM, while tying for the highest CC. PLRDiff remains the strongest competitor in PSNR and RMSE, whereas ρ-PNN provides the second-best SAM of 3.183. DIDM improves PSNR over PLRDiff by 0.768 dB and reduces SAM from 3.183 to 2.740 relative to ρ-PNN. Bicubic preserves the coarse spectral content reasonably well, but its lower PSNR and SSIM indicate limited spatial-detail recovery, while Hipandas remains below ρ-PNN on the main spectral and structural measures. These results support a more balanced use of spectral and spatial observations in scenes with dense field patterns and fine linear structures.

On Houston, DIDM again ranks first across all six metrics. DM-ZS remains the strongest overall competitor, with DIDM increasing PSNR from 38.435 to 39.172 dB, reducing SAM from 5.539 to 5.073, and lowering ERGAS from 3.178 to 2.896. ρ-PNN is the strongest direct zero-shot reconstruction baseline, reaching 37.396 dB PSNR and 6.077 SAM, while Hipandas and Bicubic show lower overall reconstruction accuracy. Given the dense man-made structures and pronounced spectral variability of Houston, the consistent gains support the effectiveness of coupling internal dual-modality prompting with PAN-guided structural regularization.

Overall, DIDM achieves the best values for all reported RR metrics across the three datasets, even after including Bicubic and the recent ρ-PNN and Hipandas baselines. The improvements extend beyond PSNR to SAM, ERGAS, SSIM, and CC, demonstrating a consistent balance between spatialdetail recovery and spectral preservation under the controlled RR protocol.

The visual comparisons are consistent with the quantitative results. Bicubic largely preserves the coarse spectral appearance of LRHS but lacks high-resolution structures, leading to blurred boundaries and stronger residuals. The classical methods recover part of the missing spatial information, while the direct zero-shot reconstruction methods, including ZSL, ρ-PNN, and Hipandas, recover richer structures with different levels of sharpness. The diffusion-prior methods enhance highfrequency details, although some results exhibit residual blur, pseudo-textures, or stronger error responses. These differences indicate that apparent sharpness alone does not necessarily yield a spatially and spectrally balanced reconstruction.

![](images/679da183688045c2b84027b9e8558b6cb3fedf64bfe30b5dd668d28ec88e41e3.jpg)  
Fig. 4. Visual comparison on the Pavia dataset. The pseudo-color images are generated using bands 50, 27, and 17 as the R, G, and B channels, respectively. The first row shows the reconstruction results, and the second row shows the corresponding error maps. Red rectangles indicate the selected regions of interest (ROIs) and their enlarged views.

![](images/470e68a3bff18f16499120b86e26a6d1ced9677bb04a7516fd967a2006034248.jpg)  
Fig. 5. Visual comparison on the Chikusei dataset. The pseudo-color images are generated using bands 60, 40, and 20 as the R, G, and B channels, respectively. The first row shows the reconstruction results, and the second row shows the corresponding error maps. Red rectangles indicate the selected ROIs and their enlarged views.

![](images/df712a14c1997f1299eae4f2bee05804834e42616e8b50d0c2c20de6307ed193.jpg)  
Fig. 6. Visual comparison on the Houston dataset. The pseudo-color images are generated using bands 59, 31, and 16 as the R, G, and B channels, respectively. The first row shows the reconstruction results, and the second row shows the corresponding error maps. Red rectangles indicate the selected ROIs and their enlarged views.

On Pavia, DIDM better delineates roof contours, road intersections, and shadow boundaries. Bicubic substantially smooths these structures, while the classical methods recover their coarse geometry but leave visible boundary residuals. ρ-PNN and Hipandas recover more spatial detail than the smoother zero-shot results, yet residual responses remain around several high-contrast structures. The diffusion-prior methods provide sharper urban details, whereas DIDM produces local transitions closer to the ground truth with relatively compact residual responses in the selected ROI.

On Chikusei, Bicubic noticeably weakens the fine field patterns and narrow linear structures. GSA, CNMF, TV, and ZSL recover the large-scale field layout but leave some boundaries blurred or attenuated. ρ-PNN and Hipandas improve local structural recovery, although residual responses remain along several fine boundaries. The diffusion-prior methods preserve more high-frequency content, while DIDM maintains the field grids, narrow linear features, and agricultural textures with more regular spatial transitions and relatively weak residual responses in the enlarged ROI.

On Houston, Bicubic produces a smooth reconstruction with reduced fine structural contrast, whereas the classical and direct zero-shot reconstruction methods progressively recover building contours and linear structures. ρ-PNN and Hipandas retain more local detail than the interpolation reference, while the diffusion-prior methods further improve structural sharpness with different levels of texture regularity. DIDM preserves clearer transitions among roofs, roads, and shadowed regions, with relatively compact residual responses around the selected ROI. This behavior is consistent with the complementary roles of LRHS-derived spectral prompting, PAN-derived spatial prompting, and PAN-guided WPATV.

TABLE IV  
FULL-RESOLUTION QUANTITATIVE COMPARISON ON THE RELEASED FR1 SAMPLE. THE BEST AND SECOND-BEST RESULTS ARE HIGHLIGHTED IN BOLD AND UNDERLINED, RESPECTIVELY. RANKINGS ARE DETERMINED FROM THE ORIGINAL UNROUNDED VALUES.
<table><tr><td>Metric</td><td>Bicubic</td><td>GSA</td><td>CNMF</td><td>TV</td><td>ZSL</td><td>ρ-PNN</td><td>Hipandas</td><td>PLRDiff</td><td>HIR-Diff</td><td>DM-ZS</td><td>DIDM</td></tr><tr><td> $D _ { \lambda } \downarrow$ </td><td>0.021</td><td>0.042</td><td>0.063</td><td>0.022</td><td>0.110</td><td>0.048</td><td>0.092</td><td>0.060</td><td>0.099</td><td>0.044</td><td>0.040</td></tr><tr><td> $D _ { S } \downarrow$ </td><td>0.225</td><td>0.143</td><td>0.100</td><td>0.150</td><td>0.056</td><td>0.148</td><td>0.086</td><td>0.121</td><td>0.114</td><td>0.118</td><td>0.108</td></tr><tr><td>HQNR↑</td><td>0.759</td><td>0.821</td><td>0.843</td><td>0.832</td><td>0.840</td><td>0.812</td><td>0.830</td><td>0.826</td><td>0.799</td><td>0.843</td><td>0.857</td></tr></table>

![](images/1aec072e33173bc84298d77113dc29b1d01ecaf99640c91e2be5a7b5eddbe42e.jpg)  
Fig. 7. Visual comparison on the released FR1 full-resolution sample. From left to right: PAN, Bicubic, GSA, CNMF, TV, ZSL, ρ-PNN, Hipandas, PLRDiff, HIR-Diff, DM-ZS, and DIDM. Bicubic denotes the LRHS upsampled to the PAN scale by bicubic interpolation. The pseudo-color images are generated using bands 20, 30, and 40 as the R, G, and B channels, respectively. Red rectangles indicate the selected ROI, whose enlarged view is shown at the bottom right of each image.

2) Full-Resolution Evaluation on FR1: The FR1 experiment complements the RR assessment by evaluating the methods on real PAN and hyperspectral observations at their native resolutions. Since FR1 does not provide a reference HRHS image, Table IV reports $D _ { \lambda } , \ D _ { S }$ , and HQNR, while Fig. 7 presents the corresponding visual comparison. The analysis focuses on spectral and spatial distortion, overall noreference fusion quality, PAN-aligned structural detail, and artifact suppression.

As shown in Table IV, DIDM achieves the highest HQNR of 0.857 on FR1, demonstrating the best overall no-reference fusion quality among the evaluated methods. Bicubic and TV yield the two lowest $D _ { \lambda }$ values of 0.021 and 0.022, respectively, but their spatial distortions are considerably larger; in particular, Bicubic exhibits the highest $D _ { S }$ of 0.225 and the lowest HQNR of 0.759, reflecting its inability to recover the missing high-resolution spatial information. Conversely, ZSL achieves the lowest $D _ { S }$ of 0.056 but has the largest $D _ { \lambda }$ of 0.110, while Hipandas provides the second-lowest $D _ { S }$ of 0.086. DIDM maintains relatively low distortions in both dimensions, with $D _ { \lambda } = 0 . 0 4 0$ and $D _ { S } = 0 . 1 0 8$ , and outperforms the other diffusion-prior methods on both distortion indices. Its highest HQNR therefore reflects a more favorable balance between spectral preservation and spatial-detail enhancement rather than an advantage in only one distortion component.

Taking PAN as the reference for spatial structure and Bicubic as the reference for the coarse spectral appearance of LRHS, Fig. 7 reveals distinct fusion behaviors among the compared methods. Bicubic preserves the overall pseudo-color distribution but lacks fine spatial details. Among the classical methods, GSA and CNMF remain comparatively smooth around field boundaries, whereas TV introduces stronger spatial enhancement with locally over-emphasized textures. ZSL also produces a relatively smooth reconstruction. The recent direct zero-shot reconstruction methods improve spatial recovery: $\rho { \mathrm { - } } \mathrm { P N N }$ delineates field boundaries and narrow structures more clearly, while Hipandas provides smoother local transitions. The diffusion-prior methods recover richer high-frequency patterns, although PLRDiff and DM-ZS exhibit visible grain-like or grid-like fluctuations in some homogeneous regions, and HIR-Diff shows prominent high-frequency artifacts and local chromatic variation in the enlarged ROI.

In contrast, DIDM produces a cleaner and more structurally coherent result on FR1. Field boundaries and narrow linear structures are clearly delineated, while homogeneous agricultural regions contain fewer spurious high-frequency fluctuations. The enlarged ROI further shows sharper yet more regular local textures than the smoother reconstruction methods, without the pronounced grain-like responses observed in highly sharpened results. Meanwhile, the overall pseudo-color appearance remains consistent with the LRHS observation. These visual characteristics are consistent with the joint effect of dual-modality prompt injection and PAN-guided WPATV. Together with the highest HQNR and low spectral distortion, the results support a favorable balance between spatial-detail enhancement and spectral preservation on FR1.

## C. Ablation Studies and Analysis

Having established the comparative performance of DIDM, we next investigate the sources of its gains and the practical implications of its instance-wise zero-shot formulation. We first isolate the contribution of each core component and examine whether prompt injection benefits from observation content rather than additional optimized parameters. We then analyze the prompt balance, injection positions, modality-specific responses, and PAN-guided WPATV. Finally, we characterize the computational behavior from two complementary perspectives: the per-pair optimization budget and the reverse-diffusion trajectory. To match the purpose of each experiment, discrete component and injection-position ablations report all six RR metrics, whereas continuous parameter sweeps use PSNR and SAM as compact representatives of reconstruction fidelity and spectral distortion, respectively. In the efficiency analysis, the iteration–cost experiment visualizes PSNR against runtime and reports SAM in the text, while the reverse-step analysis reports both PSNR and SAM to characterize sampling saturation.

1) Component Analysis and Parameter Budget: We conduct the component analysis on Houston because its complex urban structures and pronounced spectral variability provide a challenging setting for jointly evaluating spatial-detail reconstruction and spectral preservation. Table V summarizes the variants. The Baseline removes both the feature-level prompt injection and the PAN-guided WPATV while retaining the remaining reconstruction framework and observation-fidelity terms, whereas w/o Prompt Inj. removes the prompt injection but preserves the WPATV and the fidelity objective. The w/o Spe. Token and w/o Spa. Token variants suppress the LRHSderived spectral token and the PAN-derived spatial token, respectively, and w/o WPATV removes only the structural regularizer. To control for the additional capacity introduced by prompt injection, w/ Content-Free Prompt retains the complete prompt-injection architecture and optimization objective but replaces the prompt-encoder inputs with constants, setting every LRHS voxel and PAN pixel to 0.5.

![](images/92090d1422195ba1a41bfc6f337fad3239e03426c99e4ac8c14719fe0ed05eb2.jpg)

![](images/7cdee4dbbdf966d8d7cb63b8feb7abc82e076a9fb65ad0ec93b275d7f178013d.jpg)

![](images/5fae6b4d74b6e574ce529060467c2d181dd07e611e0f72a54f8953abe0a51ea9.jpg)  
Fig. 8. Effect of the spectral-token ratio α on Pavia, Chikusei, and Houston under the RR reference-based setting. The two endpoints, $\alpha = 0$ and $\alpha = 1$ correspond to spatial-only and spectral-only prompting, respectively.

TABLE V  
COMPONENT ABLATION AND PARAMETER-CAPACITY CONTROL ON THE HOUSTON DATASET. THE BEST RESULTS ARE IN BOLDFACE.
<table><tr><td>Variant</td><td>PSNR ↑</td><td>RMSE</td><td>SAM↓</td><td>ERGAS</td><td>SSIM ↑</td><td>CC↑</td></tr><tr><td>Baseline</td><td>38.398</td><td>1.264</td><td>5.757</td><td>3.196</td><td>0.927</td><td>0.981</td></tr><tr><td>w/o Prompt Inj.</td><td>38.915</td><td>1.172</td><td>5.371</td><td>2.974</td><td>0.937</td><td>0.984</td></tr><tr><td>w/ Content-Free Prompt</td><td>38.948</td><td>1.185</td><td>5.170</td><td>2.973</td><td>0.937</td><td>0.984</td></tr><tr><td>w/o Spe. Token</td><td>39.068</td><td>1.157</td><td>5.185</td><td>2.929</td><td>0.937</td><td>0.984</td></tr><tr><td>w/o Spa. Token</td><td>39.008</td><td>1.157</td><td>5.153</td><td>2.952</td><td>0.937</td><td>0.984</td></tr><tr><td>w/o WPATV</td><td>38.731</td><td>1.159</td><td>5.512</td><td>3.067</td><td>0.933</td><td>0.983</td></tr><tr><td>DIDM</td><td>39.172</td><td>1.152</td><td>5.073</td><td>2.896</td><td>0.939</td><td>0.985</td></tr></table>

The complete DIDM configuration achieves the best result for every metric, confirming the complementary contributions of its components. Relative to the Baseline, DIDM improves PSNR by 0.774 dB and reduces SAM from 5.757 to 5.073. Removing prompt injection lowers PSNR from 39.172 to 38.915 dB and increases SAM from 5.073 to 5.371, showing that injecting LRHS/PAN observations into diffusion feature evolution provides useful conditioning beyond the target-level fidelity objective. Removing either modality-specific token also degrades performance: PSNR decreases to 39.068 dB without the spectral token and 39.008 dB without the spatial token, supporting the complementarity of LRHS-derived spectral cues and PAN-derived spatial structures. Removing WPATV further reduces PSNR to 38.731 dB and increases

SAM and ERGAS to 5.512 and 3.067, respectively, demonstrating its role in suppressing structure-related distortions while preserving observation fidelity.

The content-free control separates observation content from parameter capacity. DM-ZS contains 393.069 million total parameters, of which 1.396 million are optimized, whereas DIDM contains 449.239 million total parameters and optimizes 58.191 million parameters; the increase mainly comes from the active cross-attention layers and two image-projection modules used for prompt injection. Despite retaining the same prompt-injection architecture and optimized parameter budget as DIDM, the content-free variant remains inferior on all six metrics, obtaining 38.948 dB PSNR and 5.170 SAM. Relative to w/o Prompt Inj., its changes are small and metric-dependent: PSNR increases by only 0.033 dB, RMSE increases from 1.172 to 1.185, and SSIM and CC remain unchanged. By contrast, supplying the prompt pathway with the actual LRHS and PAN observations yields consistent improvements across all metrics, indicating that the gains of DIDM arise primarily from observation-aware dual-modality prompting rather than from the enlarged optimized parameter budget alone.

2) Prompt Design Analysis: After establishing the effectiveness of the observation-aware prompt injection, we analyze the prompt design from three perspectives: the balance between LRHS-derived spectral and PAN-derived spatial prompts, the feature levels at which they are injected, and their modality-specific responses within the diffusion backbone. These experiments examine whether the two prompts are complementary, whether conditioning should span multiple feature scales, and whether the branches exhibit distinct internal behaviors.

Prompt Balance. The spectral-token ratio α controls the relative contribution of the two modality-specific prompts: α = 0 retains only the PAN-derived spatial prompt, α = 1 retains only the LRHS-derived spectral prompt, and intermediate values combine both modalities. To compactly characterize the reconstruction and spectral trends across three datasets, Fig. 8 reports PSNR and SAM as representative RR metrics. The results show that joint prompting is consistently more effective than either single-modality alternative, although the preferred balance varies across datasets. (a) On Houston, spatial-only and spectral-only prompting yield PSNR/SAM values of 39.068/5.185 and 39.008/5.153, respectively, both below 39.172/5.073 at α = 0.5. (b) A moderate spectral– spatial balance is generally favorable: Chikusei achieves its best PSNR/SAM at α = 0.5 (39.797/2.740), whereas Pavia favors the slightly more spatially weighted setting $\alpha = 0 . 3$ (34.932/6.283). (c) The improvement of intermediate ratios over the two endpoints confirms that LRHS-derived spectral evidence and PAN-derived spatial structures provide complementary conditioning; accordingly, the selected dataset-level value is fixed for all samples in the main experiments.

TABLE VI  
EFFECT OF PROMPT-INJECTION POSITIONS ON THE HOUSTON DATASET. ✓ AND × INDICATE ENABLED AND DISABLED PROMPT INJECTION, RESPECTIVELY. THE BEST RESULTS ARE IN BOLDFACE.
<table><tr><td>Down</td><td>Mid Up</td><td></td><td>PSNR ↑</td><td>RMSE</td><td>↓ SAM↓</td><td>ERGAS</td><td>↓ SSIM ↑</td><td>CC ↑</td></tr><tr><td>×</td><td>×</td><td>×</td><td>38.915</td><td>1.172</td><td>5.371</td><td>2.974</td><td>0.937</td><td>0.984</td></tr><tr><td>√</td><td>×</td><td>X</td><td>39.097</td><td>1.162</td><td>5.092</td><td>2.920</td><td>0.939</td><td>0.984</td></tr><tr><td>X</td><td>√</td><td>X</td><td>39.091</td><td>1.162</td><td>5.133</td><td>2.921</td><td>0.939</td><td>0.984</td></tr><tr><td>X</td><td>×</td><td>√</td><td>39.096</td><td>1.164</td><td>5.107</td><td>2.920</td><td>0.938</td><td>0.984</td></tr><tr><td>√</td><td>√</td><td>X</td><td>39.116</td><td>1.160</td><td>5.075</td><td>2.912</td><td>0.939</td><td>0.984</td></tr><tr><td>√</td><td>×</td><td>√</td><td>39.120</td><td>1.159</td><td>5.118</td><td>2.912</td><td>0.939</td><td>0.984</td></tr><tr><td>×</td><td>√</td><td>√</td><td>39.128</td><td>1.159</td><td>5.078</td><td>2.914</td><td>0.939</td><td>0.984</td></tr><tr><td>√</td><td>√</td><td>√</td><td>39.172</td><td>1.152</td><td>5.073</td><td>2.896</td><td>0.939</td><td>0.985</td></tr></table>

Injection Positions. We examine whether prompt conditioning should be restricted to a specific feature level by independently enabling or disabling prompt injection in the downsampling path (Down), middle block (Mid), and upsampling path $( U p )$ which represent different feature resolutions and abstraction levels in the U-Net-like diffusion backbone. The Houston results in Table VI show that prompt injection is beneficial at every stage and that distributing it across the full diffusion hierarchy yields the strongest overall performance. (a) Each singlestage configuration improves upon the variant without featurelevel prompting: the three single-stage variants achieve PSNR values of 39.091–39.097 dB and reduce SAM from 5.371 to 5.092–5.133, indicating that the LRHS/PAN observations provide useful conditioning at different feature resolutions. (b) Multi-stage injection is generally more effective than single-stage injection, as demonstrated by the Down+Mid and Mid+Up configurations, which reduce SAM to 5.075 and 5.078, respectively. (c) Jointly activating Down, Mid, and $U p$ achieves the best result, improving PSNR from 38.915 to 39.172 dB and reducing SAM from 5.371 to 5.073 relative to disabling all injection stages, which indicates that propagating the observation prompts through the encoding, bottleneck, and reconstruction paths better exploits the multiscale feature hierarchy than restricting them to a single level.

Prompt-Response Behavior. Fig. 9 visualizes prompt-induced response magnitudes, rather than raw attention weights, extracted from the mid-block cross-attention outputs on a Houston sample. LRHS-derived spectral regions are obtained by clustering the spatially aligned LRHS spectral vectors and provide a reference for interpreting region-wise response organization. (a) $R _ { \mathrm { s p a } }$ is strongly activated along the dominant linear structure and around several high-contrast objects in the PAN image, whereas $R _ { \mathrm { s p e } }$ exhibits broader region-wise variation and weaker edge-following behavior. (b) This distinction is quantified by $\rho _ { \nabla }$ , the Spearman correlation between a response map and the PAN-gradient magnitude, and $E _ { \mathrm { e d g e } } ,$ the ratio between the mean response over the highest 10% PAN-gradient pixels and that over the remaining pixels. For the displayed sample, $R _ { \mathrm { s p a } }$ obtains $\rho _ { \nabla } = 0 . 3 2 6$ and $E _ { \mathrm { e d g e } } =$ 1.122, compared with −0.323 and 0.830 for $R _ { \mathrm { s p e } } { \mathrm { . } }$ , supporting the structure-oriented role of the PAN-derived prompt. (c) Region-wise dependence is further measured by $\eta _ { \mathrm { r e g } } ^ { 2 } ,$ the fraction of response variance explained by the LRHS-derived spectral partition. $R _ { \mathrm { s p e } }$ achieves $\eta _ { \mathrm { r e g , s p e } } ^ { 2 } = 0 . 6 2 4$ , exceeding $\eta _ { \mathrm { r e g , s p a } } ^ { 2 } = 0 . 4 5 8 \mathrm { ; }$ ; together with its smoother spatial distribution, this result indicates stronger organization by the LRHS spectral grouping. These quantities serve only as descriptive response diagnostics rather than fusion-quality metrics, and the observed differences support the complementary roles of the two prompt branches.

![](images/6e96cad4ee368db26cf0394529d947b079aa6c1bc0a5241f3e44864039e86dd5.jpg)  
Fig. 9. Visualization of modality-specific prompt responses on a Houston sample. (a) PAN observation. (b) Spatial-prompt response $R _ { \mathrm { s p a } } .$ . (c) LRHSderived spectral regions obtained by clustering the LRHS spectral vectors. (d) Spectral-prompt response $R _ { \mathrm { s p e } } ,$ where the overlaid contours indicate the boundaries of the same LRHS-derived spectral partition shown in (c). The two response maps are independently normalized to [0, 1] for visualization; brighter colors indicate stronger relative responses within each map.

3) PAN-Guided Structural Regularization Analysis: Having verified the contribution of WPATV in the component ablation, we further analyze the structural regularizer from three complementary perspectives: the effect of its weight, its joint interaction with the prompt balance, and its edgepreserving behavior. The analysis follows a progressive logic from quantitative benefit to parameter robustness and, finally, to mechanism-level visual evidence.

Effect of the Structural Weight. We examine the influence of the WPATV weight $\lambda _ { \mathrm { s t r } } \in \{ 0 , 1 , 1 0 , 5 0 , 1 0 0 , 2 0 0 \}$ on Pavia, Chikusei, and Houston while fixing α to the dataset-level settings of 0.3, 0.5, and 0.5, respectively; $\lambda _ { \mathrm { s t r } } = 0$ removes WPATV and, on Houston, corresponds to the w/o WPATV configuration in Table V. Because this experiment targets the spatial–spectral trend of a continuous hyperparameter sweep, Fig. 10 reports PSNR and SAM as representative reconstruction and spectral metrics. A consistent improvement followed by saturation is observed across all three datasets. (a) Increasing $\lambda _ { \mathrm { s t r } }$ from 0 to 50 raises PSNR from 34.605 to 34.932 dB, 38.682 to 39.797 dB, and 38.731 to 39.172 dB on Pavia, Chikusei, and Houston, respectively, while reducing SAM from 6.715 to 6.283, 3.256 to 2.740, and 5.512 to 5.073. (b) Beyond $\lambda _ { \mathrm { s t r } } = 5 0$ , increasing the weight to 100 or 200 changes PSNR by at most 0.044 dB and SAM by at most 0.026 relative to the corresponding value at 50, indicating that the benefit largely saturates once an effective structural constraint is established. (c) We therefore adopt $\lambda _ { \mathrm { s t r } } ~ = ~ 5 0$ as a common setting because it reaches the onset of this stable regime and provides a favorable spatial–spectral balance without relying on stronger regularization. The cross-dataset trend supports WPATV as a robust structural complement to the observation-fidelity terms rather than a parameter requiring precise calibration.

![](images/92824a2e89c0623f4f5fcf7453c5c4ae03ee9cb316a4df39e0a83acf4bf4c0fe.jpg)

![](images/427b69c478473f04617899cfcc11f4e541afea00091c7a987b9aea102dd3ddd5.jpg)

![](images/9a6474189b17acbb3160a17f1cf7549b0eb6ab20fea4fa20062d6ca98dcc316f.jpg)

Fig. 10. Effect of the PAN-guided WPATV weight $\lambda _ { \mathrm { s t r } }$ on the Pavia, Chikusei, and Houston datasets. The spectral-token ratio α is fixed to 0.3 for Pavia, and 0.5 for both Chikusei and Houston.  
![](images/0d7fb3b259d694a422afce082f8bfcb0a2570e676203ef63600f4a57fb79e242.jpg)  
Fig. 11. Joint sensitivity of the spectral-token ratio α and the WPATV weight λ<sub>str</sub> across the Pavia, Chikusei, and Houston datasets in terms of PSNR (top row) and SAM (bottom row). The black rectangles mark stable high-performance regions for each specific dataset.

Joint Sensitivity to Prompt Balance and Structural Regularization. Because the spectral-token ratio α controls prompt composition whereas $\lambda _ { \mathrm { s t r } }$ controls structural regularization, we jointly vary $\alpha \in \{ 0 , 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 , 1 \}$ and $\lambda _ { \mathrm { s t r } } \ \in$ $\{ 0 , 1 , 1 0 , 5 0 , 1 0 0 , 2 0 0 \}$ . To keep the two-dimensional sensitivity analysis interpretable while covering both reconstruction fidelity and spectral distortion, Fig. 11 reports PSNR and

SAM on all three datasets. The heatmaps show broad datasetdependent high-performance regions rather than sharply isolated optima. (a) On Pavia, the black-boxed region spanning $\alpha ~ = ~ 0 . 1 – 0 . 5$ and $\lambda _ { \mathrm { s t r } } ~ = ~ 5 0 { - } 2 0 0$ maintains PSNR within 34.845–34.932 dB and SAM within 6.273–6.353. (b) On Chikusei and Houston, the corresponding region of $\alpha =$ 0.3–0.7 and $\lambda _ { \mathrm { s t r } } ~ = ~ 5 0 { - } 2 0 0$ yields PSNR/SAM ranges of 39.689–39.813/2.740–2.801 and 39.090–39.195/5.046–5.117, respectively. (c) Although the exact optimum shifts among neighboring parameter combinations, the adopted settings $( \alpha , \lambda _ { \mathrm { s t r } } ) = ( 0 . 3 , 5 0 )$ for Pavia and (0.5, 50) for Chikusei and Houston lie within these stable regions and remain close to the best PSNR–SAM combinations. These results indicate that the two parameters play complementary roles without introducing a narrow operating range, supporting dataset-level selection of α together with the common choice $\lambda _ { \mathrm { s t r } } = 5 0$

(a) PAN gradient  
(b) With WPATV  
(c) Without WPATV  
![](images/b1bbc7366a5b156ffa2bfab59d700281d7f076d5c6fb1a7ce93519e02895d44f.jpg)  
Fig. 12. Visualization of PAN-aligned edge agreement on a Houston sample. (a) PAN-gradient magnitude. Edge-agreement maps obtained (b) with and (c) without WPATV. Green and vermillion curves denote preserved and missed PAN-derived reference edges, respectively, whereas magenta curves indicate spurious pseudo-PAN edges.

Edge-Preservation Behavior. To directly examine the structural effect of WPATV, Fig. 12 compares PAN-derived reference edges with edges extracted from the pseudo-PAN projections of the fused results reconstructed with and without WPATV. The pseudo-PAN projections are generated using the same spectral response as the PAN observation, subjected to common radiometric normalization, and evaluated under identical edge-extraction and matching settings; green and vermillion curves denote preserved and missed PAN-derived reference edges, respectively, whereas magenta curves indicate spurious pseudo-PAN edges. The edge-agreement maps provide direct evidence of the structural benefit of WPATV. (a) Edge recall measures the fraction of PAN-reference edges retained by the reconstruction, edge precision measures the fraction of reconstructed edges aligned with the PAN reference, and edge F1 is their harmonic mean; these quantities are used only as structural diagnostics. (b) For the displayed sample, WPATV increases recall from 0.865 to 0.955, precision from 0.920 to 0.962, and F1 from 0.892 to 0.959, while reducing the numbers of missed and spurious edges from 903 to 298 and from 499 to 253, corresponding to reductions of 67.0% and 49.3%, respectively. (c) Consistently, the WPATV result contains fewer vermillion and magenta responses and more continuous green structures, supporting the intended weighting mechanism in which smoothing is relaxed around strong PAN transitions and strengthened in relatively homogeneous regions to preserve PAN-aligned discontinuities while suppressing structure-related artifacts.

4) Optimization and Efficiency Analysis: DIDM incurs perpair computation through instance-wise zero-shot optimization and reverse-diffusion sampling. We therefore first examine the optimization budget that governs the overall adaptation cost and then analyze the reverse-diffusion trajectory that controls sampling refinement. This separation characterizes the diminishing returns associated with the two principal computational budgets of DIDM.

Iteration–Cost Trade-off. As an instance-wise zero-shot method, DIDM adapts its learnable modules independently to each LRHS/PAN pair. Because optimization-based and feedforward methods follow different computational paradigms, we characterize the internal performance–cost behavior of DIDM rather than directly comparing runtime across method classes. Since this experiment focuses on the quality–cost relationship of per-pair adaptation, Fig. 13 visualizes PSNR as a representative reconstruction-quality metric together with optimization time, while SAM is reported in the text as a complementary spectral indicator. Over 5,000–30,000 iterations, reconstruction quality improves monotonically but gradually saturates. (a) Increasing the iteration count from 5,000 to 25,000 raises mean PSNR from 38.644 to 39.172 dB and reduces SAM from 5.413 to 5.073. (b) Beyond the adopted setting, increasing the budget from 25,000 to 30,000 iterations improves PSNR by only 0.010 dB and reduces SAM by 0.012, while the mean optimization time reaches 379.848 s, confirming that the performance gain has largely saturated. (c) We therefore adopt 25,000 iterations in the main comparisons as a practical balance between reconstruction quality and computational cost. Under this setting, the complete per-pair procedure takes approximately 331.1 s and reaches a peak GPU memory consumption of about 7.3 GB on a single NVIDIA GeForce RTX 4090D GPU.

![](images/7e80a60b0d8ff7a367d683d534b7f0a1f0fa38017775b86ce423b4ff483c940d.jpg)  
Fig. 13. Performance-cost trade-off with respect to the number of per-pair optimization iterations on Houston. The line and bars report the mean PSNR and mean optimization time per sample, respectively.

Effect of Reverse Diffusion Steps. We next vary the number of reverse diffusion steps T while keeping the remaining settings fixed. Since this experiment examines the reconstruction and spectral saturation of the sampling process across datasets, Fig. 14 reports PSNR and SAM as representative RR metrics on Pavia, Chikusei, and Houston, with the vertical dashed line marking the adopted value $T = 2 0 0$ . The curves exhibit a clear transition from rapid improvement at small T to saturation around $T = 2 0 0$ . (a) With only a few reverse steps, the diffusion prior cannot sufficiently refine spatial– spectral details; on Houston, increasing T from 5 to 100 raises PSNR from 34.594 to 38.532 dB and reduces SAM from 9.171 to 5.599, with similar trends on Pavia and Chikusei. (b) Beyond $T = 2 0 0$ , additional refinement yields only marginal gains: increasing T from 200 to 500 improves PSNR by merely 0.009, 0.013, and 0.060 dB on Pavia, Chikusei, and Houston, respectively, while the corresponding SAM changes are also limited. (c) Because sampling cost increases with the reverse trajectory length, $T = 2 0 0$ retains nearly all attainable reconstruction improvement without incurring the substantially higher cost of longer sampling and is therefore adopted as a practical balance between spatial–spectral refinement and sampling efficiency. Together with the iteration analysis above, these results justify the adopted computational budgets at both the optimization and sampling levels.

![](images/e3d0c57ce7e1d3546860fba356519c5b72a375b4a71a13c4e13ad6f97775601a.jpg)

![](images/56dca6ff09c91b831c7269f35799e84806a0b76e4a49ede26251deb1bd368879.jpg)

![](images/b095e3177a7bed895313b8d44eb8092656d0cb7c80a78dcc826f00c3153e6331.jpg)  
Fig. 14. Effect of the number of reverse diffusion steps on Pavia, Chikusei, and Houston under the RR reference-based setting. The vertical dashed line marks the adopted setting $T = 2 0 0 .$

## V. CONCLUSION

This paper presented DIDM, a dual-modality imageprompted diffusion model for zero-shot hyperspectral pansharpening. DIDM internalized LRHS-derived spectral evidence and PAN-derived spatial structures as modality-specific prompts and injected them into a remote-sensing pre-trained diffusion prior through cross-attention, allowing the observations to participate directly in diffusion feature evolution. PAN-guided WPATV further complemented the observationfidelity terms by imposing gradient-adaptive regularization on the spatial component, encouraging PAN-aligned structural transitions while suppressing unnecessary fluctuations in homogeneous regions.

Under the RR reference-based protocol, DIDM achieved the best values across all six metrics on Pavia, Chikusei, and Houston. Its PSNR reached 34.932, 39.797, and 39.172 dB, exceeding the strongest competing results by 1.166, 0.768, and 0.737 dB, respectively. On the released FR1 full-resolution sample, DIDM achieved the highest HQNR of 0.857, with $D _ { \lambda } = 0 . 0 4 0$ and $D _ { S } ~ = ~ 0 . 1 0 8$ , and outperformed the other diffusionprior methods on both distortion indices. The content-free prompt control indicated that the gains arose primarily from observation-aware conditioning rather than additional promptinjection capacity alone. Moreover, the modality-specific response maps revealed distinct structure-oriented and spectralregion-aware behaviors, while the WPATV edge analysis showed improved PAN-aligned edge preservation and fewer spurious responses. Together, these results supported both the effectiveness and the design rationale of DIDM.

The current implementation still requires per-pair optimization and iterative reverse diffusion, which limits efficiency when processing large image collections despite the diminishing returns identified in both computational-budget analyses. Future work will investigate accelerated or amortized adaptation, more efficient reverse sampling, data-dependent prompt balancing, and evaluation on broader real full-resolution sensor datasets. Extending the proposed observation-aware diffusion conditioning to other hyperspectral fusion and restoration tasks also constitutes a promising direction.

## REFERENCES

[1] L. Loncan, L. B. Almeida, J. M. Bioucas-Dias et al., “Hyperspectral pansharpening: A review,” IEEE Geosci. Remote Sens. Mag., vol. 3, no. 3, pp. 27–46, 2015.

[2] G. Vivone, L. Alparone, J. Chanussot et al., “A critical comparison among pansharpening algorithms,” IEEE Trans. Geosci. Remote Sens., vol. 53, no. 5, pp. 2565–2586, 2015.

[3] Y. Li, Y. Luo, L. Zhang et al., “MambaHSI: Spatial–spectral Mamba for hyperspectral image classification,” IEEE Trans. Geosci. Remote Sens., vol. 62, pp. 1–16, 2024, art. no. 5524216.

[4] W. Yang, D. Wu, J. Bai et al., “FedDA-HSI: Federated class-aware framework for hyperspectral image classification with diffusion augmentation,” IEEE Trans. Geosci. Remote Sens., vol. 64, pp. 1–17, 2026, art. no. 5510010.

[5] S. K. Roy, A. Sukul, A. Jamali et al., “Cross hyperspectral and LiDAR attention transformer: An extended self-attention for land use and land cover classification,” IEEE Trans. Geosci. Remote Sens., vol. 62, pp. 1–15, 2024, art. no. 5512815.

[6] J. Lin, F. Gao, L. Qi et al., “Dynamic cross-modal feature interaction network for hyperspectral and LiDAR data classification,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–16, 2025, art. no. 5508812.

[7] B. Aiazzi, S. Baronti, and M. Selva, “Improving component substitution pansharpening through multivariate regression of MS+Pan data,” IEEE Trans. Geosci. Remote Sens., vol. 45, no. 10, pp. 3230–3239, 2007.

[8] N. Yokoya, T. Yairi, and A. Iwasaki, “Coupled nonnegative matrix factorization unmixing for hyperspectral and multispectral data fusion,” IEEE Trans. Geosci. Remote Sens., vol. 50, no. 2, pp. 528–537, 2012.

[9] M. Simoes, J. Bioucas-Dias, L. B. Almeida˜ et al., “A convex formulation for hyperspectral image superresolution via subspace-based regularization,” IEEE Trans. Geosci. Remote Sens., vol. 53, no. 6, pp. 3373–3388, 2015.

[10] P. Addesso, L. Condat, A. Foi et al., “Collaborative total variation for hyperspectral pansharpening,” in Proc. IEEE Int. Geosci. Remote Sens. Symp., 2017, pp. 1982–1985.

[11] S. Takeyama and S. Ono, “Hyperspectral image super-resolution using hybrid spatio-spectral total variation,” IEEE Trans. Image Process., vol. 29, pp. 4058–4072, 2020.

[12] R. Dian, S. Li, A. Guo et al., “Deep hyperspectral image sharpening,” IEEE Trans. Neural Netw. Learn. Syst., vol. 29, no. 11, pp. 5345–5355, 2018.

[13] Z. Guo, J. Xin, N. Wang et al., “External-internal attention for hyperspectral image super-resolution,” IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 1–14, 2022, art. no. 5510013.

[14] Q. Xie, M. Zhou, Q. Zhao et al., “MHF-Net: An interpretable deep network for multispectral and hyperspectral image fusion,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 3, pp. 1457–1473, 2022.

[15] Y. Zheng, J. Li, Y. Li et al., “Hyperspectral pansharpening using deep prior and dual attention residual network,” IEEE Trans. Geosci. Remote Sens., vol. 58, no. 11, pp. 8059–8076, 2020.

[16] J.-F. Hu, T.-Z. Huang, L.-J. Deng et al., “Hyperspectral image superresolution via deep spatiospectral attention convolutional neural networks,” IEEE Trans. Neural Netw. Learn. Syst., vol. 33, no. 12, pp. 7251–7265, 2022.

[17] S. Liu, Q. Shi, L. Zhang et al., “SSAU-Net: A spectral-spatial attentionbased U-Net for hyperspectral image fusion,” IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 1–16, 2022, art. no. 5510012.

[18] Y. Liu, J. Liu, R. Dian et al., “A selective re-learning mechanism for hyperspectral fusion imaging,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 7437–7446.

[19] M. Ciotola, G. Guarino, G. Vivone et al., “Hyperspectral pansharpening: Critical review, tools, and future perspectives,” IEEE Geosci. Remote Sens. Mag., vol. 13, no. 1, pp. 311–338, 2025.

[20] Y. Qu, H. Qi, and C. Kwan, “Unsupervised sparse Dirichlet-net for hyperspectral image super-resolution,” IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 1–15, 2022, art. no. 5510014.

[21] G. Guarino, M. Ciotola, G. Vivone et al., “Zero-shot hyperspectral pansharpening using hysteresis-based tuning for spectral quality control,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–19, 2025.

[22] S. Xu, Z. Zhao, H. Bai et al., “Hipandas: Hyperspectral image joint denoising and super-resolution by image fusion with the panchromatic image,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2025, pp. 12 002– 12 011.

[23] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Adv. Neural Inf. Process. Syst., 2020, pp. 6840–6851.

[24] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” in Int. Conf. Learn. Represent., 2021, pp. 1–13.

[25] A. Q. Nichol and P. Dhariwal, “Improved denoising diffusion probabilistic models,” in Proc. Int. Conf. Mach. Learn., 2021, pp. 8162–8171.

[26] P. Dhariwal and A. Q. Nichol, “Diffusion models beat GANs on image synthesis,” in Adv. Neural Inf. Process. Syst., 2021, pp. 8780–8794.

[27] C. Saharia, J. Ho, W. Chan et al., “Image super-resolution via iterative refinement,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 4, pp. 4713–4726, 2023.

[28] A. Lugmayr, M. Danelljan, A. Romero et al., “RePaint: Inpainting using denoising diffusion probabilistic models,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 11 461–11 471.

[29] R. Rombach, A. Blattmann, D. Lorenz et al., “High-resolution image synthesis with latent diffusion models,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 10 684–10 695.

[30] B. Kawar, M. Elad, S. Ermon et al., “Denoising diffusion restoration models,” in Adv. Neural Inf. Process. Syst., 2022, pp. 23 593–23 606.

[31] H. Chung, J. Kim, M. T. Mccann et al., “Diffusion posterior sampling for general noisy inverse problems,” in Int. Conf. Learn. Represent., 2023, pp. 1–17.

[32] L. Pang, X. Rui, L. Cui et al., “HIR-Diff: Unsupervised hyperspectral image restoration via improved diffusion models,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 3005–3014.

[33] X. Rui, X. Cao, L. Pang et al., “Unsupervised hyperspectral pansharpening via low-rank diffusion model,” Inf. Fusion, vol. 107, 2024, art. no. 102325.

[34] J.-L. Xiao, T.-Z. Huang, L.-J. Deng et al., “Hyperspectral pansharpening via diffusion models with iteratively zero-shot guidance,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 12 669– 12 678.

[35] A. Vaswani, N. Shazeer, N. Parmar et al., “Attention is all you need,” in Adv. Neural Inf. Process. Syst., 2017, pp. 5998–6008.

[36] J. Li, K. Zheng, Z. Li et al., “X-shaped interactive autoencoders with cross-modality mutual learning for unsupervised hyperspectral image super-resolution,” IEEE Trans. Geosci. Remote Sens., vol. 61, pp. 1–16, 2023, art. no. 5510015.

[37] B. Aiazzi, L. Alparone, S. Baronti et al., “MTF-tailored multiscale fusion of high-resolution MS and Pan imagery,” Photogramm. Eng. Remote Sens., vol. 72, no. 5, pp. 591–596, 2006.

[38] N. Akhtar, F. Shafait, and A. Mian, “Sparse spatio-spectral representation for hyperspectral image super-resolution,” in Proc. Eur. Conf. Comput. Vis., 2014, pp. 63–78.

[39] Q. Wei, N. Dobigeon, and J.-Y. Tourneret, “Bayesian fusion of multiband images,” IEEE J. Sel. Topics Signal Process., vol. 9, no. 6, pp. 1117–1127, 2015.

[40] G. Masi, D. Cozzolino, L. Verdoliva et al., “Pansharpening by convolutional neural networks,” Remote Sens., vol. 8, no. 7, 2016, art. no. 594.

[41] J. Yang, X. Fu, Y. Hu et al., “PanNet: A deep network architecture for pan-sharpening,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2017, pp. 5449–5457.

[42] Y. Wei, Q. Yuan, H. Shen et al., “Boosting the accuracy of multispectral image pansharpening by learning a deep residual network,” IEEE Geosci. Remote Sens. Lett., vol. 14, no. 10, pp. 1795–1799, 2017.

[43] Q. Yuan, Y. Wei, X. Meng et al., “A multiscale and multidepth convolutional neural network for remote sensing imagery pan-sharpening,” IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 11, no. 3, pp. 978–989, 2018.

[44] K. Cao, X. He, X. Li et al., “Cross-scale pansharpening via ScaleFormer and the PanScale benchmark,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2026, pp. 13 211–13 221.

[45] R. Gal, Y. Alaluf, Y. Atzmon et al., “An image is worth one word: Personalizing text-to-image generation using textual inversion,” in Int. Conf. Learn. Represent., 2023, pp. 1–15.

[46] N. Ruiz, Y. Li, V. Jampani et al., “DreamBooth: Fine tuning textto-image diffusion models for subject-driven generation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 22 500– 22 510.

[47] Y. Li, H. Liu, Q. Wu et al., “GLIGEN: Open-set grounded text-to-image generation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 22 511–22 521.

[48] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2023, pp. 3836–3847.

[49] C. Mou, X. Wang, L. Xie et al., “T2I-Adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models,” in Proc. AAAI Conf. Artif. Intell., vol. 38, no. 5, 2024, pp. 4296–4304.

[50] F. Shen and J. Tang, “Imagpose: A unified conditional framework for pose-guided person generation,” Advances in neural information processing systems, vol. 37, pp. 6246–6266, 2024.

[51] F. Shen, X. Jiang, X. He, H. Ye, C. Wang, X. Du, Z. Li, and J. Tang, “Imagdressing-v1: Customizable virtual dressing,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 7, 2025, pp. 6795–6804.

[52] F. Shen, J. Yu, C. Wang, X. Jiang, X. Du, and J. Tang, “Imaggarment: Fine-grained garment generation for controllable fashion design,” IEEE Transactions on Visualization and Computer Graphics, 2026.

[53] F. Shen, C. Wang, J. Gao, Q. Guo, J. Dang, J. Tang, and T.-S. Chua, “Long-term talkingface generation via motion-prior conditional diffusion model,” in Forty-second International Conference on Machine Learning.

[54] L. Wald, “Quality of high resolution synthesised images: Is there a simple criterion?” in Proc. Int. Conf. Fusion Earth Data, 2000, pp. 99–103.

[55] G. Vivone, A. Garzelli, Y. Xu et al., “Panchromatic and hyperspectral image fusion: Outcome of the 2022 WHISPERS hyperspectral pansharpening challenge,” IEEE J. Sel. Top. Appl. Earth Observ. Remote Sens., vol. 16, pp. 166–179, 2022.

[56] F. A. Kruse, A. B. Lefkoff, J. W. Boardman et al., “The spectral image processing system (SIPS)–interactive visualization and analysis of imaging spectrometer data,” Remote Sens. Environ., vol. 44, no. 2–3, pp. 145–163, 1993.

[57] Z. Wang, A. C. Bovik, H. R. Sheikh et al., “Image quality assessment: From error visibility to structural similarity,” IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, 2004.

[58] A. Arienzo, G. Vivone, A. Garzelli et al., “Full resolution quality assessment of pansharpening: Theoretical and hands-on approaches,” IEEE Geosci. Remote Sens. Mag., vol. 10, no. 3, pp. 168–201, 2022.