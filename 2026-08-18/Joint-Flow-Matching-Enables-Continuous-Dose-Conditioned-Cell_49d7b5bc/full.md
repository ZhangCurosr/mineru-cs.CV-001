# Joint Flow Matching Enables Continuous Dose-Conditioned Cell Morphing

Lea Bogensperger<sup>1</sup> , Manuela Merlo<sup>1</sup> , Martin Baumgartner<sup>2</sup> , Michael Krauthammer<sup>1</sup> , and Bernard Ciraulo<sup>2</sup>

<sup>1</sup> University of Zurich

{lea.bogensperger,manuela.merlo,michael.krauthammer}@uzh.ch 2 University Children’s Hospital Zurich {martin.baumgartner,bernard.ciraulo}@kispi.uzh.ch

Abstract. Generative modeling has shown increasing promise for predicting cellular perturbation efects under chemical compound treatments. Existing approaches either model perturbation as a distributionto-distribution mapping without explicit concentration handling, or treat concentration as a discrete class label, precluding continuous dose control. We introduce a joint flow matching approach that simultaneously models cell latents and drug concentration via a dual-timestep formulation, enabling dose-conditioned single-cell morphing through the invertibility of flow matching. The joint formulation induces a monotonic dose-response geometry in latent space and additionally supports concentration estimation from cell morphology. As proof of concept, we further demonstrate generalization to an unseen dose held out during training. Empirically, our method achieves competitive or improved perconcentration metrics on two compounds compared with representative baselines, while enabling capabilities structurally unavailable to discreteclass methods.

Keywords: Flow Matching · Cell Perturbation · Dose-Response Modeling · Generative Modeling · Cellular Morphology · Fluorescence Microscopy

## 1 Introduction

In cellular biology, morphological features often reflect cellular state and function [41]. Optical microscopy is the dominant technique for capturing cellular morphology, commonly through fluorescence labeling and imaging of the protein or cellular compartment of interest [16]. This approach provides rich information on cellular responses to external chemical perturbations. Image-based response characterization is particularly relevant in drug development and drug-response profiling in oncology, where large panels of drugs are screened across a range of concentrations to score compounds or treatment eficacy [40]. Such assays have historically relied on hand-designed morphological features [7, 8]. The scale of these experiments has grown substantially with modern automated microscopes, which enable extensive sampling of cellular populations across vast parameter spaces, generating large image collections and, with them, statistically robust characterization of morphological variation [9]. However, cellular populations are typically heterogeneous, and treatments often difer only subtly—whether due to similar chemical composition or fine-grained concentration screening. As a result, the morphologies observed in these experimental settings span a continuous space rather than falling into discrete categories, making assessment through classical computer vision approaches challenging [34]. An approach capable of modeling this continuum of morphologies and quantifying dose-dependent morphological drift would enable estimation of minimal efective doses and comparison of closely related compounds, ofering meaningful benefits for compound screening.

Recent generative models ofer a promising route toward this goal, learning to predict cellular perturbation responses directly from image data [6,28,42,43]. CellFlux [42] and its latent-space extension CellFluxV2 [43] frame the problem as a distribution-to-distribution transformation from control cells to treated cells using flow matching [26, 27], but do not include concentration as a variable in their formulation. IMPA [28] learns compound-level style codes via a Generative Adversarial Network (GAN) [12], again without an explicit concentration representation. PhenDif [6] does condition on concentration using conditional difusion models [17], but treats it as a categorical class label, which prevents interpolation to unseen doses and does not capture the continuous nature of doseresponse efects. None of these methods jointly models the distribution over cell images and concentration.

We introduce a joint flow matching approach for continuous dose-response cell image generation over latent representations of cell images and drug concentration (Fig. 1). Our contributions are as follows:

– We extend latent flow matching to jointly model the distribution p(z, c) over cell latents and drug concentrations via a dual-timestep formulation.

– Our method enables dose-conditioned cell morphing by leveraging the invertibility of flow matching, matching or exceeding concentration-aware baselines on per-concentration metrics while remaining competitive on concentrationagnostic evaluations.

– We show that joint modeling induces a monotonic mean dose-response trajectory in latent space and improves concentration-specific fidelity over discreteclass baselines.

As proof of concept, we hold out one concentration during training and show generation at that unseen dose, a capability structurally unavailable to discrete-class methods.

## 2 Related Work

Flow Matching and Latent Generative Models. Difusion models [17, 36, 37] and flow matching [26, 27] have become state-of-the-art for high-fidelity generative modeling. Latent-space extensions such as Stable Difusion [11, 31] exploit variational autoencoder (VAE) compression to enable eficient high-resolution generation while maintaining or improving sample quality. Flow matching has also proven efective in medical and cellular imaging domains [4, 6, 42]. Conditional generation is naturally supported within the flow matching framework by conditioning the velocity network on auxiliary information, which can further be enhanced at inference via classifier-free guidance (CFG) [18]. Recent joint diffusion and flow approaches [2, 25] extend single-modality generation to multiple modalities by introducing a separate timestep per modality. Finally, invertibility is a key property of flow matching that enables image editing through inversion in noise space [13, 32]; recent work also explores inversion-free variants [22].

![](images/a4c7c275b0c87256e3a5c58986c3a0c8b151a090215319692bfccf8057979ad9.jpg)

![](images/0d58cd08e92dd148086d11eda783d454a131d80cb634bf999e7289f2178c4d00.jpg)  
Fig. 1: Joint Flow Matching for continuous dose-response cell image generation. Given a real control cell, our model encodes it into a VAE latent z , inverts the flow to noise z<sub>0</sub>, and re-integrates the forward ordinary diferential equation (ODE) at target concentrations c<sup>⋆</sup> to generate morphed cells across a continuous dose range. The dual-timestep formulation jointly models p(z, c), enabling continuous concentration control.

Generative Models of Cellular Perturbation Response. Recent generative models targeting drug-induced morphological changes adopt a variety of architectural and methodological formulation choices. Because paired untreated–treated observations are typically unavailable, generative approaches model control and treated cells as distributions rather than individual correspondences. Flow matching enables CellFlux [42] to learn a continuous transformation between these distributions without paired examples. However, without additional constraints, aligning distributions in aggregate may miss subtle biological efects: the model captures broad distributional shifts rather than the specific treatment-induced changes. Its extension CellFluxV2 [43] further moves the setup into a latent space and introduces a two-stage training procedure with noisy interpolants [1]. Neither variant models concentration as an explicit variable in their formulation. IMPA [28] and PhenExplain [24] use style-based GANs [12,19] to translate control cells to treated ones via per-perturbation style codes, but neither incorporates concentration in their formulation. Closest to our setting, PhenDif [6] employs a class-conditional difusion model [17,35] with image inversion to translate cells between phenotypes, but treats concentration as a discrete class label, preventing continuous dose interpolation.

Our method combines latent flow matching (CellFluxV2) with inversionbased editing (PhenDif), but unlike prior work that treats concentration as a discrete label or ignores it, we jointly model the (image, concentration) distribution with a continuous scalar. This enables source-conditioned cell morphing across doses and concentration estimation in a single framework.

## 3 Method

Let $\boldsymbol { x } \in \mathbb { R } ^ { C \times H \times W }$ denote a cell image with $C = 3$ channels and spatial resolution $H \times W = 2 5 6 \times 2 5 6 .$ , associated with a drug concentration $c \in \mathbb { R } _ { \geq 0 }$ (in $\mu \mathrm { M } )$ A pretrained VAE [21] with encoder $\mathcal { E }$ and decoder D maps images to a latent representation $z = \dot { \mathcal { E } } ( x ) \in \mathbb { R } ^ { d _ { z } \times h \times w }$ with $d _ { z } = 4$ latent channels and spatial resolution $h \times w = 3 2 \times 3 2$ , yielding a spatial compression factor of $8 \times$

## 3.1 Latent-Space Embedding

Similar to Latent CellFlux [43], we operate in a compressed latent space to enable eficient generative modeling. We therefore train a VAE with encoder $\mathcal { E } _ { \psi } : \mathbb { R } ^ { C \times H \times W } \stackrel { \smile } { \to } \mathbb { R } ^ { d _ { z } \times h \times w }$ and decoder $\mathcal { D } _ { \psi } : \mathbb { R } ^ { d _ { z } \times h \times w }  \mathbb { R } ^ { C \times H \times W }$ , parameterized by ψ. This follows the latent difusion paradigm [31]: the VAE compresses images to a semantic latent space where the generative model (Sec. 3.2) operates eficiently, and the VAE is frozen for all downstream training. The VAE is trained by minimizing a combination of a structural similarity (SSIM) loss with an $\ell _ { 1 }$ pixel error and a Kullback–Leibler divergence aligning the encoder posterior $q _ { \psi } ( z \mid x ) = \mathcal { N } ( \mu _ { \psi } ( x ) , \sigma _ { \psi } ^ { 2 } ( x ) )$ with a standard normal prior $p ( z ) = \mathcal { N } ( 0 , I )$

$$
\mathcal { L } _ { \mathrm { V A E } } ( \psi ; x ) = \alpha \mathcal { L } _ { \mathrm { S S I M } } ( \hat { x } , x ) + \left( 1 - \alpha \right) \mathcal { L } _ { \mathrm { L 1 } } ( \hat { x } , x ) + \beta D _ { \mathrm { K L } } ( q _ { \psi } ( z \mid x ) \parallel p ( z ) )\tag{1}
$$

where $\alpha , \beta \in \mathbb { R } ^ { + }$ weight the individual loss components, $\hat { x } = { \mathcal D } _ { \psi } ( { \mathcal E _ { \psi } ( x ) } )$ is the reconstruction, and $\mathcal { L } _ { \mathrm { S S I M } } ( \hat { x } , x ) = 1 - \mathrm { S S I M } ( \hat { x } , x )$ . The VAE itself is concentrationagnostic, concentration is only used downstream by the flow model (Sec. 3.2).

## 3.2 Joint Flow Matching for Dose-Conditioned Generation

We train one flow model per drug over the joint distribution $p ( z , c )$ of latent images and their corresponding concentration. Similar to PhenDif [6] which is based on DDIM inversion, we use the invertibility of flow matching. However, we model the concentration as a continuous scalar instead of discrete classes, and further use a joint dual timestep formulation to learn the joint distribution of latent images and their concentrations, which has been successfully done for other multi-modal settings [2, 25].

Flow Matching Background. Flow matching [26,27] learns a velocity field $v _ { \theta }$ that transports samples from a simple prior $z _ { 0 } \sim \mathcal { N } ( 0 , I )$ to the data distribution $z _ { 1 } \sim p _ { \mathrm { d a t a } }$ . Training considers the linear interpolation:

$$
z _ { t } = ( 1 - t ) z _ { 0 } + t z _ { 1 } , \ t \in [ 0 , 1 ] ,\tag{2}
$$

whose velocity is constant and given by $z _ { 1 } - z _ { 0 }$ . The conditional flow matching objective regresses the predicted velocity $v _ { \theta } ( z _ { t } , t )$ toward this target:

$$
\mathcal { L } _ { \mathrm { C F M } } = \mathbb { E } _ { z _ { 0 } , z _ { 1 } , t } \left. v _ { \theta } ( z _ { t } , t ) - ( z _ { 1 } - z _ { 0 } ) \right. _ { 2 } ^ { 2 } .\tag{3}
$$

Sampling discretizes the ordinary diferential equation (ODE) ${ d z _ { t } } / { d t } = v _ { \theta } ( z _ { t } , t )$ into K Euler steps of size $\Delta t = 1 / K \colon z ^ { ( k + 1 ) } = z ^ { ( k ) } + \Delta t v _ { \theta } \big ( \dot { z } ^ { ( k ) } , t ^ { ( k ) } \big )$ with $t ^ { ( k ) } = k / K$ , starting from $z ^ { ( 0 ) } = z _ { 0 }$

Invertibility. Because sampling is a deterministic ODE, the map from prior to data is invertible: integrating $v _ { \theta }$ backwards in time from $z _ { 1 }$ recovers z<sub>0</sub>:

$$
z _ { 0 } = z _ { 1 } - \int _ { 0 } ^ { 1 } v _ { \theta } ( z _ { t } , t ) d t .\tag{4}
$$

This holds provided $v _ { \theta } ( \cdot , t , c )$ is suficiently regular in z and no noise is injected at inference. In practice we invert with Euler steps evaluated at the known iterate:

$$
z ^ { ( k - 1 ) } = z ^ { ( k ) } - \varDelta t v _ { \theta } \Big ( z ^ { ( k ) } , t ^ { ( k ) } \Big ) .\tag{5}
$$

Joint Formulation with Dual Timesteps. To model the joint distribution $p ( z , c )$ ， we introduce two independent timesteps, $t _ { z }$ for the latent and $t _ { c }$ for the concentration, inspired by previous works on multi-modal generative learning [2, 25]. The timesteps are sampled uniformly $t _ { z } , t _ { c } \sim \mathcal { U } ( 0 , 1 )$

The latent variable z and the concentration c are then noised independently according to their respective timesteps:

$$
z _ { t _ { z } } = ( 1 - t _ { z } ) z _ { 0 } + t _ { z } z _ { 1 } , \qquad z _ { 0 } \sim \mathcal { N } ( 0 , I ) , z _ { 1 } = \mathcal { E } ( x ) ,\tag{6}
$$

$$
c _ { t _ { c } } = \left( 1 - t _ { c } \right) c _ { 0 } + t _ { c } c _ { 1 } , ~ c _ { 0 } \sim \mathcal { N } ( 0 , 1 ) , ~ c _ { 1 } = c .\tag{7}
$$

The network predicts velocities for both modalities with $v _ { z } ~ \in ~ \mathbb { R } ^ { d _ { z } \times h \times w }$ and $v _ { c } \in \mathbb { R }$ :

$$
( v _ { z } , v _ { c } ) = v _ { \theta } ( z _ { t _ { z } } , c _ { t _ { c } } , t _ { z } , t _ { c } ) ,\tag{8}
$$

trained with the joint objective $( \lambda _ { c } \in \mathbb { R } ^ { + }$ balances both terms):

$$
\mathcal { L } _ { \mathrm { C F M } } ( \theta ) = \mathbb { E } _ { ( z _ { 0 } , z _ { 1 } ) , ( c _ { 0 } , c _ { 1 } ) , t _ { z } , t _ { c } } \left[ \| v _ { z } - ( z _ { 1 } - z _ { 0 } ) \| _ { 2 } ^ { 2 } + \lambda _ { c } \| v _ { c } - ( c _ { 1 } - c _ { 0 } ) \| _ { 2 } ^ { 2 } \right] .\tag{9}
$$

Sampling Modes. The two independent timesteps enable three distinct inference regimes. Joint sampling integrates both $t _ { z } , t _ { c } : 0  1$ simultaneously to draw new $( z , c )$ pairs from the learned joint distribution; while beyond the scope of this work, this mode could support downstream tasks such as synthetic data augmentation. Our experiments primarily target dose-conditioned image generation, which fixes $t _ { c } = 1$ (target concentration held constant) and integrates $t _ { z } : 0  1$ to sample $z \sim p ( z \mid c )$ , followed by decoding via $\mathcal { D } ;$ this corresponds to our dose-conditioned generation and inversion-based morphing experiments (Sec. 3.3). Conversely, concentration inference fixes $t _ { z } = 1$ (image latent held constant) and integrates $t _ { c } : 0  1$ to sample $c \sim p ( c \mid z )$ 2 allowing the model to predict the concentration of a given cell.

## 3.3 Inversion-Based Cell Morphing

The invertibility of flow matching enables cell morphing across drug concentrations, analogous to inversion-based editing with difusion [13] and its recent adoption for cellular perturbation via PhenDif [6]. Given a real control image $x ,$ we encode it as $z _ { 1 } = \mathcal { E } ( x )$ and integrate the flow backward at the control conditioning value $c = 0$ using K Euler steps with $\varDelta t = 1 / K$ , with the concentration timestep fixed at $t _ { c } = 1$ , to recover the corresponding noise:

$$
\begin{array} { r } { z ^ { ( k - 1 ) } = z ^ { ( k ) } - \varDelta t \ v _ { z } \Big ( z ^ { ( k ) } , c _ { \mathrm { c o n t r o l } } , t _ { z } = \frac { k } { K } , t _ { c } = 1 \Big ) , \quad k = K , K - 1 , \ldots , 1 , } \end{array}\tag{10}
$$

starting from $z ^ { ( K ) } = z _ { 1 }$ and ending at the inverted noise $z _ { 0 } : = z ^ { ( 0 ) }$ . Integrating the forward ODE from the same $z _ { 0 }$ at a target concentration $c ^ { \star }$ then yields a morphed latent:

$$
\begin{array} { r } { z ^ { ( k + 1 ) } = z ^ { ( k ) } + \varDelta t \ v _ { z } \Big ( z ^ { ( k ) } , c ^ { \star } , t _ { z } = \frac { k } { K } , t _ { c } = 1 \Big ) , \quad k = 0 , 1 , \ldots , K - 1 , } \end{array}\tag{11}
$$

yielding $z _ { 1 } ^ { c ^ { \star } } : = z ^ { ( K ) }$ which is decoded as $\hat { x } ^ { c ^ { \star } } = \mathcal { D } ( z _ { 1 } ^ { c ^ { \star } } )$ . Sharing the same z<sub>0</sub> across concentrations yields a family of latents that difer only through the concentration conditioning, producing a morph strip for a single source cell.

## 4 Experiments

We evaluate our method on a microscopy imaging dataset of a pediatric brain tumor cell line, spanning a panel of two compounds (C7 and CK-666) at six concentrations each (five doses plus solvent controls). The VAE is trained jointly on all five compounds in the dataset, providing a shared latent space across drugs (see Sec. 4.5 for an ablation). Per-compound flow matching models are then trained on top of this shared latent space. We compare against three representative baselines from the literature: PhenDif [6], IMPA [28], and Latent CellFlux (our reimplementation of CellFluxV2 [43]). Main-text experiments focus on C7; corresponding results for CK-666 are provided in the supplementary material.

## 4.1 Dataset

Sample. The pediatric brain tumor medulloblastoma cell line model ONS-76 was used to evaluate a panel of compounds selected for their ability to interfere with cell migration through modulation of cytoskeletal dynamics, a key biophysical mechanism involved in metastasis [30]. This panel included C7, a small-molecule ligand of FRS2 proposed to disrupt the FRS2–FGFR interaction [23]; CK-666, an inhibitor of the Arp2/3 complex [14]; famlasertib, a small-molecule inhibitor of the MAP4K family of Ser/Thr kinases [5] [33]; paranitroblebbistatin, an inhibitor of non-muscle myosin II ATPase activity [20]; Y-27632, a selective ROCK1/2 inhibitor [38]; and DMSO solvent as control. To enable fluorescence live-cell imaging, ONS-76 cells were labeled by lentiviral transduction with LifeAct-EGFP and mCherry-Nuc9. LifeAct-EGFP was used to visualize F-actin, while mCherry-Nuc9 was used to mark the nucleus.

Microscopy. Widefield fluorescence imaging of both channels was performed on a Nikon Ti2 microscope equipped with a 20× objective and GFP and Cy3 filter sets, using tiled acquisition of 6×6 fields of view per well with 10% overlap (tile size 2424×2424 px, pixel size 0.33 µm/px, 16-bit depth); individual tiles were automatically stitched using Nikon acquisition software. Cells were seeded in imaging-grade glass-bottom 96-well plates, with a minimum of two wells replicates per drug and concentration tested.

Single-cell Dataset Creation. From the stitched tiled images, single-cell crops of 256×256 pixels centered on each detected cell were extracted to build an RGB single-cell image dataset. Cell detection was performed on 10-fold downscaled images using Cellpose (v4.1.1) with default parameters. For each extracted single-cell image, the green (F-actin) and red (nuclear) channel dynamic ranges were normalized to 8-bit prior to saving, while the blue channel was set to zero.

Overview. Our modeling setup involves two components: a single VAE trained jointly on the training splits of all five compounds in the dataset, providing a shared latent space; and per-compound flow-matching models trained on top of this latent space. Reported experiments target C7 and CK-666 (see Fig. 5 for some example images); the shared VAE is trained on all available compounds to encourage a general representation of cellular morphology (see Sec. 4.5 for an ablation).

Data Handling. Regions failing standard image quality checks (e.g., saturation, focus, signal-to-noise) were excluded. Each compound’s images are split into train/validation/test partitions in a 70/15/15 ratio, stratified by concentration to preserve the dose distribution across splits. Image counts per compound (including shared DMSO controls) are: 25,384 for C7, 25,603 for CK-666, 23,007 for famlasertib, 25,068 for paranitro-blebbistatin, and 25,823 for Y-27632. Since concentrations span multiple orders of magnitude, we use $\log _ { 1 0 } \mathbf { - }$ transformed values; controls are assigned $c _ { \mathrm { c o n t r o l } } = \log _ { 1 0 } ( c _ { \mathrm { m i n } } ) - 0 . 5$ , where $c _ { \mathrm { m i n } }$ is the smallest non-zero training concentration.

## 4.2 Settings

Latent Encoder. Our VAE follows a standard convolutional encoder-decoder design with three downsampling stages (each halving the spatial resolution, base channels 64 doubled per stage) built from ResNet blocks, mapping $3 \times 2 5 6 \times 2 5 6$ images to $4 \times 3 2 \times 3 2$ latents. The VAE is trained on the training splits of all five drugs and all concentrations for 200 epochs with AdamW (learning rate $1 0 ^ { - 4 }$ , batch size 32, warmup ratio 0.05), using loss weights $\alpha = 0 . 6$ (SSIM) and $\beta = 1 0 ^ { - 7 } ~ \mathrm { ( K L ) }$

Flow Model. The velocity network v is a U-Net with self-attention at the bottleneck, base dimension 64, and channel multipliers (1, 2, 4, 8) across four downsampling stages. Both timesteps $t _ { z } , t _ { c }$ and the concentration scalar c are embedded via small MLPs with nonlinear GELU activations; the two time embeddings are concatenated and combined with the concentration embedding to form the conditioning signal injected into each residual block; a small MLP head predicts $v _ { c }$ from the U-Net bottleneck features. We train one model per drug (Ours joint, 37.8M parameters) for 3000 epochs with AdamW (learning rate $1 0 ^ { - 4 }$ , batch size 100, EMA decay 0.999), using $\lambda _ { c } = 0 . 5$ (chosen empirically). Concentrations are $\log _ { 1 0 }$ -transformed as described in Section 4.1, see Sec. 4.5 for an ablation. Inference uses $K = 2 0 0$ Euler steps for forward generation and $K = 2 0 0$ for inversion (see Sec. A.5 for an ablation over samplers). For ablation, we additionally train a concentration-conditional variant (Ours cond. only, 31.5M parameters) with the same U-Net architecture but without the second timestep $t _ { c } ,$ using classifier-free guidance with dropout probability $p = 0 . 1 5$ during training and guidance weight $w = 2$ at inference. Concentration conditioning in this variant is injected via Feature-wise Linear Modulation (FiLM) [29], though initial experiments showed standard embedding mechanisms perform comparably.

Baselines. We compare against three representative methods: Latent CellFlux [42] (31.4M parameters, concentration-agnostic latent flow matching), IMPA [28] (74.4M parameters, pixel-space per-perturbation GAN), and PhenDif [6] (99.5M parameters, pixel-space DDIM with discrete concentration classes). We use publicly available code from PhenDif and IMPA, both of which operate in pixel space. For CellFluxV2, which operates in latent space, we reimplement the method in our own VAE latent space; we refer to this reimplementation as Latent CellFlux. We use K = 200 sampling steps for all flow-based methods and PhenDif-standard DDIM inference for the difusion baseline.

Evaluation Metrics. For inversion fidelity, we report peak signal-to-noise ratio (PSNR) and structural similarity (SSIM) [39] on the test set, comparing each real control image against its round-trip reconstruction obtained after inverting the encoded control cell followed by re-generating it using the forward ODE. For distributional fidelity, we compute Fréchet Inception Distance (FID) [15] and Kernel Inception Distance (KID) [3] per concentration and pooled across all concentrations, using Inception-v3 features extracted from all available real training images and an equal number of generated cells $( N _ { \mathrm { r e a l } } = N _ { \mathrm { g e n } } = N ;$ N ≈ 850–1, 250 per dose, $N > 5 , 0 0 0$ pooled). Generated cells are obtained by inverting real control cells and forward-integrating at the target concentration, similar to PhenDif [6]. Because FID is subject to finite-sample bias [10], we report N explicitly and emphasize the unbiased KID estimator alongside FID for robust per-concentration evaluation.

Table 1: Inversion fidelity for C7 on the test set (N=100 control cells).
<table><tr><td>ODE steps K PSNR ↑ SSIM ↑</td></tr><tr><td>10  $2 7 . 7 9 _ { \pm 2 . 9 } \ 0 . 8 6 6 _ { \pm 0 . 0 4 }$ </td></tr><tr><td>25 31.93±3.00.894±0.04</td></tr><tr><td>50  $3 3 . 8 1 _ { \pm 3 . 0 } \ 0 . 9 0 2 _ { \pm 0 . 0 4 }$  34.50±3.0 0.905±0.05</td></tr><tr><td>100 200 34.70±3.0 0.906±0.05</td></tr><tr><td>500  $3 4 . 7 7 _ { \pm 3 . 0 } \ 0 . 9 0 7 _ { \pm 0 . 0 5 }$ </td></tr><tr><td>VAE only  $3 8 . 7 3 _ { \pm 3 . 1 }$   $0 . 9 1 3 { \scriptstyle \pm 0 . 0 5 }$ </td></tr></table>

## 4.3 Generative Evaluation

We evaluate our joint flow matching model along three dimensions: round-trip reconstruction fidelity of the inversion pathway, distributional fidelity of generated cells against real observations at each concentration, and morphing quality against baselines. We report the main results on C7 in this section; corresponding results for CK-666 are provided in Sec. A.2.

Inversion Fidelity. Since dose-conditioned morphing relies on inverting a real control cell into noise before re-integrating the forward ODE at a target concentration $c ^ { * }$ , we first assess the fidelity of the inversion. Table 1 reports round-trip PSNR and SSIM on 100 control test images for C7 (see Tab. 5 for CK-666), computed by encoding each image to its latent $z _ { 1 } ,$ inverting to noise $z _ { 0 }$ at the control conditioning value, and re-integrating the forward ODE back to the input. The metrics are upper-bounded by the VAE’s own reconstruction quality (last row), since part of the observed error comes from the VAE bottleneck rather than the flow inversion itself. Reconstruction quality saturates with increasing ODE steps; we adopt K = 200 steps for inversion in all subsequent experiments. Notably, SSIM nearly matches the VAE upper bound, indicating that structural content is well-preserved through the flow round-trip; the larger PSNR gap reflects small pixel-level noise introduced by the ODE discretization, to which pixel-wise metrics are more sensitive than the patch-based SSIM.

Qualitative Morphing. Figure 2 shows dose-conditioned morphs from our method and baselines that support dose-specific generation. Concentration-agnostic meth ods (IMPA, Latent CellFlux) formulate perturbation as a distribution-to-distribution transformation and are not designed to natively represent the subtle morphological changes characteristic of dose response.

control  
0.037 M  
0.11 M  
0.33 M  
1 M  
3 M  
![](images/6353ebbf1bacc6c7fcee6898a20a7b4d9fa2b9b335bf66109869365946977ff4.jpg)  
Fig. 2: Qualitative comparison of dose-response morphing for C7. Each row shows a real control cell morphed across concentrations of C7 (columns: 0.037, 0.11, $0 . 3 3 , 1 . 0 , 3 . 0 \ \mu \mathrm { M } )$ . Latent CellFlux and IMPA are shown only at control and morphed dose. Individual samples from concentration-agnostic methods can vary in visual quality across cells.

Distribution Fidelity. Table 2 reports FID and KID scores for C7 across five concentrations and pooled across the full dose range. To isolate the contribution of joint modeling, we include an ablation without the joint model and hence the second timestep $t _ { c }$ (Ours cond. only). Since Latent CellFlux and IMPA do not natively support dose-specific generation, as both model a single treated-cell distribution per compound, we report their metrics only pooled, as an upper reference on generation quality without concentration awareness. Among methods with explicit concentration modeling, our joint model achieves the best FID and KID at every concentration. While pixel-space IMPA achieves the lowest pooled FID overall (17.58), our joint model closes most of this gap despite operating in latent space.

Table 2: FID/KID comparison (lower is better) for C7. Per-concentration values shown for methods with explicit concentration modeling; concentration-agnostic baselines (Latent CellFlux, IMPA) are reported at pooled only, as they do not natively support concentration targeting. Best metrics per column are in bold, second-best underlined.
<table><tr><td>Method</td><td>Metric 0.037</td><td colspan="4">Concentration  $( \mu \mathrm { M } )$ </td><td>Pooled</td></tr><tr><td>Latent CellFlux [42]†</td><td>FID</td><td>(N=892)</td><td>(N=1028)(N=868)(N=1040)</td><td></td><td>(N=1270)</td><td>(N=5098) 25.19</td></tr><tr><td>IMPA [28]†</td><td>KID FID</td><td></td><td></td><td></td><td></td><td>1.90 17.58</td></tr><tr><td>PhenDiff [6]</td><td>KID FID</td><td>30.03</td><td>29.74</td><td>33.22 31.90</td><td>34.16</td><td>0.97 21.53</td></tr><tr><td>Ours (cond. only)</td><td>KID FID</td><td>1.61 36.19</td><td>1.73 30.86</td><td>1.73 1.91 42.63 37.43</td><td>2.10 38.23</td><td>1.68 26.57</td></tr><tr><td>Ours (Joint Latent FM) FID</td><td>KID</td><td>2.25 30.02</td><td>1.72 27.20</td><td>2.93 2.45 27.50</td><td>2.65 27.85 25.02</td><td>2.10 18.98</td></tr></table>

† Latent CellFlux and IMPA do not natively support concentration conditioning; therefore we report their pooled results without concentration awareness.

## 4.4 Concentration Controllability

While distributional metrics such as FID and KID capture overall generation quality, they do not directly assess whether generation is controllable across concentrations. We evaluate controllability along three dimensions: latent-space trajectory, cross-concentration specificity, and generalization to unseen doses using a holdout experiment.

Latent Space Dose-Response. We sample a fixed noise vector $z ^ { ( 0 ) }$ , integrate the forward ODE at each concentration, and measure latent displacement from the control trajectory. Figure 3 shows the resulting curves and Spearman correlations $( n = 1 0 0 ~ \mathrm { c e l l s } )$ . Our joint model produces a smooth, monotonic dose response $( \rho = 1 . 0 0 , p = 0 . 0 0 3 )$ ), while removing the joint component or modeling concentration using discrete classes degrades this monotonic behaviour $( \rho = 0 . 7 1$ $p = 0 . 1 3 9$ and $\rho = 0 . 8 9 , p = 0 . 0 4 2$ , respectively). Evaluating individual singlecell trajectories $( n = 1 0 0$ cells) confirms strong per-cell monotonicity for our joint model with a median $\rho = 0 . 9 0$ (IQR: 0.70–1.00), outperforming PhenDif (median $\rho = 0 . 8 0$ , IQR: 0.50–0.90) and the conditioning-only baseline (median $\rho = 0 . 4 0$ , IQR: 0.30–0.70; Supplementary Fig. 7).

Concentration Specificity. While the trajectory analysis probes latent geometry, we further evaluate whether this structure translates into controllable generation at each concentration. For each pair $( c _ { i } , c _ { j } )$ , we compute the KID between the distribution of generated cells conditioned on $c _ { i }$ and the distribution of real cells at $c _ { j } ;$ a controllable model produces diagonal-dominant matrices where the minimum along each row lies on the diagonal. Results are shown in Fig. 4, including a real-vs-real ceiling. The ceiling shows the task’s inherent dificulty: cells at $3 \mu \mathrm { M }$ are well-separated from all doses, but low-dose distributions (0.037– $0 . 3 3 \mu \mathrm { M } )$ morphologically overlap even in real data, upper-bounding achievable specificity. Nonetheless, our joint model achieves lower diagonal KID values than PhenDif at every concentration, indicating closer distributional matching.

![](images/dc3aa723b5dea5b05d2265ec1ccf1d84862a579c9db6a621b0a04ca48fa44973.jpg)

![](images/4cd05a095089f101beb979b75b738df26b22b5d52f95645edadb14795285a653.jpg)  
Fig. 3: Dose-response geometry and generalization to unseen concentrations for C7. Left: Normalized latent displacement from the reference as a function of drug concentration, comparing our joint model, our conditional-only ablation, and PhenDif. Right: After retraining our joint model with $0 . 3 3 \mu \mathrm { M }$ held out during training (train concentrations: $0 . 0 3 7 , 0 . 1 1 , 1 . 0 , 3 . 0 \mu \mathrm { M } )$ , forward integration at the held-out dose (highlighted star) yields a monotonic position between its neighbors.

Generalization to Unseen Concentrations. To test whether continuous concentration modeling enables interpolation to unseen concentrations, we retrain our joint model on C7 with the middle concentration $( 0 . 3 3 \mu \mathrm { M } )$ held out (train concentrations: $0 . 0 3 7 , 0 . 1 1 , 1 . 0 , 3 . 0 \mu \mathrm { M } )$ . At inference, forward integration at the held-out dose lands at the correct monotonic position between its neighbors in latent space (Fig. $^ { \mathrm { 3 b , } }$ orange star), achieving FID = 33.66 against real $0 . 3 3 \mu \mathrm { M }$ cells — only ∼ 19% above the mean FID at trained concentrations of the same model (28.20). Notably, this out-of-distribution result almost matches PhenDif $\mathrm { ( F I D = 3 3 . 2 2 }$ Tab. 2), a discrete-class baseline that was trained on $0 . 3 3 \mu \mathrm { M }$ Such continuous-dose generalization is structurally unavailable to discrete-class methods. See Fig. 8 for more held-out concentrations $( 0 . 1 1 \mu \mathrm { M }$ and $1 \mu \mathrm { M } )$ on C7.

Concentration Inference. A distinctive capability of our joint formulation is sampling from $p ( c \mid z )$ to predict the dose corresponding to a given cell image. Fixing $t _ { z } = 1$ and integrating $t _ { c } : 0  1$ from a noise sample, we obtain a concentration estimate cˆ. On the test set $( N = 1 1 1 5$ cells across all training doses), we achieve top-1 accuracy of 34.0% and 68.0% within ±1 neighboring dose (5-class classification). Accuracy is highest at the extreme concentrations (44% at $0 . 0 3 7 \mu \mathrm { M }$ , 55% at $3 \mu \mathrm { M } )$ and drops at intermediate doses, consistent with the overlapping morphological distributions of neighboring low concentrations shown in Fig. 4. For reference, a supervised ResNet-18 classifier trained on the same real data achieves 48% top-1 and 74.8% within ±1 neighboring dose, as detailed in Sec. A.4.

![](images/5e93d4d8a0b68151eabbad77912deb7b491c2748d56b3b615cd6885ee07bcb9b.jpg)

![](images/043408e6afaa5992c8d94235c975a2ace8039abe32481d202f04d58ff3ef865a.jpg)

![](images/13f71204ddf557403c27c54cac20a7cbfc08153df28f4173939f00fe3a1723a3.jpg)  
Fig. 4: Concentration specificity. KID between generated cells conditioned on concentration c<sub>i</sub> (rows) and real cells at concentration $c _ { j }$ (columns). Diagonals are outlined; red asterisks mark each row’s argmin. Panels (left to right): our joint model, PhenDif, and the real-vs-real reference ceiling.

Table 3: VAE training-data ablation on C7. Best values in bold.
<table><tr><td rowspan="2"></td><td colspan="5">Concentration  $( \mu \mathrm { M } )$ </td><td rowspan="2">Pooled</td></tr><tr><td>VAE training Metric 0.037</td><td></td><td>0.11</td><td>0.33 1.0</td><td>3.0</td></tr><tr><td>C7-only</td><td>FID</td><td>35.26</td><td>33.39</td><td>34.37</td><td>33.89 31.54</td><td>24.84</td></tr><tr><td></td><td>KID</td><td>2.18</td><td>2.09</td><td>2.14 2.06</td><td>2.01</td><td>1.94</td></tr><tr><td>Multi-drug</td><td>FID</td><td>30.02</td><td>27.20</td><td>27.50</td><td>27.85 25.02</td><td>18.98</td></tr><tr><td></td><td>KID</td><td>1.58</td><td>1.40</td><td>1.37 1.39</td><td>1.28</td><td>1.25</td></tr></table>

## 4.5 Ablations

We perform two ablations on the joint model’s design choices for C7. Table 3 shows that training the VAE on all drugs (our default) yields a more general latent representation than training on C7 alone, improving FID and KID at every concentration. Since drug concentrations span multiple orders of magnitude, we condition the flow model on $\log _ { 1 0 } \mathbf { - }$ transformed values rather than raw concentrations; Tab. 4 shows that this improves FID and KID at every concentration.

Table 4: Concentration-transform ablation on C7. Best values in bold.
<table><tr><td></td><td colspan="4">Concentration (µM)</td></tr><tr><td>Conc. transform Metric 0.037</td><td></td><td>0.11 0.33</td><td>1.0 3.0</td><td>Pooled</td></tr><tr><td>Raw (c)</td><td>FID</td><td>30.27 27.72 29.94</td><td>28.50</td><td>27.02 20.31</td></tr><tr><td></td><td>KID 1.61</td><td>1.46 1.67</td><td>1.46</td><td>1.48 1.35</td></tr><tr><td> $\log _ { 1 0 } ( c )$ </td><td>FID</td><td>2 27.20</td><td>27.85 25.02</td><td></td></tr><tr><td>KID</td><td>30.02</td><td>27.50</td><td></td><td>18.98</td></tr><tr><td></td><td>1.58</td><td>1.40 1.37</td><td>1.39 1.28</td><td>1.25</td></tr></table>

## 5 Conclusion

Conclusion. We introduced joint latent flow matching for continuous doseconditioned cell morphing, enabling generation at unseen concentrations and estimation of dose from cell morphology. Our joint formulation matches or exceeds baselines on distributional fidelity across two compounds, while producing qualitatively faithful dose-response trajectories.

Limitations. Our reconstruction fidelity is currently bounded by the frozen VAE bottleneck (Sec. 3.1), which imposes a ceiling on generation quality regardless of the flow model’s capability. Furthermore, concentration inference from single cells (Sec. 4.4) achieves only moderate accuracy, reflecting the inherent phenotypic variability of morphology at fixed doses.

Future Work. Replacing the VAE with a stronger encoder-decoder (e.g., the architecture used in Stable Difusion LDM [31]) could substantially raise the reconstruction ceiling. A second direction is to extend our framework to a single flow model conditioned jointly on drug identity and concentration, modeling multiple compounds within a unified formulation, e.g., by adopting the drug conditioning mechanism in IMPA [28]. Moreover, our current per-compound models already share a common latent space via the multi-drug VAE, enabling comparative analyses of latent dose-response trajectories across drugs to group compounds by their morphological efect. Finally, downstream validation of generated morphs using a classifier model trained on real images would strengthen the practical utility of our approach for perturbation-based cell morphing.

## Acknowledgements

L.B. and M.M. acknowledge funding by the University Research Priority Program (URPP) Human Reproduction Reloaded of the University of Zurich. L.B. is further supported by the Swiss National Science Foundation under grant 10003518. M.B. and B.C. were supported by the Swiss National Science Foundation Sinergia grant CRSII5\_202245/1.

## References

1. Albergo, M., Bofi, N.M., Vanden-Eijnden, E.: Stochastic interpolants: A unifying framework for flows and difusions. Journal of Machine Learning Research 26(209), 1–80 (2025)

2. Bao, F., Nie, S., Xue, K., Li, C., Pu, S., Wang, Y., Yue, G., Cao, Y., Su, H., Zhu, J.: One transformer fits all distributions in multi-modal difusion at scale. In: International Conference on Machine Learning. pp. 1692–1717. PMLR (2023)

3. Bińkowski, M., Sutherland, D.J., Arbel, M., Gretton, A.: Demystifying mmd gans. arXiv preprint arXiv:1801.01401 (2018)

4. Bogensperger, L., Narnhofer, D., Falk, A., Schindler, K., Pock, T.: Flowsdf: Flow matching for medical image segmentation using distance transforms. International Journal of Computer Vision 133(7), 4864–4876 (2025)

5. Bos, P.H., Lowry, E.R., Costa, J., Thams, S., Garcia-Diaz, A., Zask, A., Wichterle, H., Stockwell, B.R.: Development of MAP4 Kinase Inhibitors as Motor Neuron-Protecting Agents. Cell Chemical Biology 26(12), 1703–1715.e37 (2019), http: //dx.doi.org/10.1016/j.chembiol.2019.10.005

6. Bourou, A., Boyer, T., Gheisari, M., Daupin, K., Dubreuil, V., De Thonel, A., Mezger, V., Genovesio, A.: Phendif: Revealing subtle phenotypes with difusion models in real images. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 358–367. Springer (2024)

7. Bray, M.A., Singh, S., Han, H., Davis, C.T., Borgeson, B., Hartland, C., Kost-Alimova, M., Gustafsdottir, S.M., Gibson, C.C., Carpenter, A.E.: Cell Painting, a high-content image-based assay for morphological profiling using multiplexed fluorescent dyes. Nature Protocols 11(9), 1757–1774 (2016)

8. Caicedo, J.C., Arevalo, J., Piccioni, F., Bray, M.A., Hartland, C.L., Wu, X., Brooks, A.N., Berger, A.H., Boehm, J.S., Carpenter, A.E., Singh, S.: Cell Painting predicts impact of lung cancer variants. Molecular Biology of the Cell 33(6), 1–11 (2022)

9. Chandrasekaran, S.N., Cimini, B.A., Goodale, A., Miller, L., Kost-Alimova, M., Jamali, N., Doench, J.G., Fritchman, B., Skepner, A., Melanson, M., Kalinin, A.A., Arevalo, J., Haghighi, M., Caicedo, J.C., Kuhn, D., Hernandez, D., Berstler, J., Shafqat-Abbasi, H., Root, D.E., Swalley, S.E., Garg, S., Singh, S., Carpenter, A.E.: Three million images and morphological profiles of cells treated with matched chemical and genetic perturbations. Nature Methods 21(6), 1114–1121 (2024)

10. Chong, M.J., Forsyth, D.: Efectively unbiased fid and inception score and where to find them. In: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 6069–6078. IEEE (2020)

11. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: Forty-first international conference on machine learning (2024)

12. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial networks. Communications of the ACM 63(11), 139–144 (2020)

13. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-Or, D.: Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626 (2022)

14. Hetrick, B., Suk Han, M., Nolen, B.: Small Molecules CK-666 and CK-869 Block an Activating Conformational Change to Inhibit Arp2/3 Complex. Biophysical Journal 102(3), 46a (2012)

15. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017)

16. Hickey, S.M., Ung, B., Bader, C., Brooks, R., Lazniewska, J., Johnson, I.R., Sorvina, A., Logan, J., Martini, C., Moore, C.R., Karageorgos, L., Sweetman, M.J., Brooks, D.A.: Fluorescence microscopy—an outline of hardware, biological handling, and fluorophore considerations. Cells 11(1) (2022)

17. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)

18. Ho, J., Salimans, T.: Classifier-free difusion guidance. arXiv preprint arXiv:2207.12598 (2022)

19. Karras, T., Laine, S., Aila, T.: A style-based generator architecture for generative adversarial networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4401–4410 (2019)

20. Képiró, M., Várkuti, B.H., Végner, L., Vörös, G., Hegyi, G., Varga, M., Málnási-Csizmadia, A.: Para-nitroblebbistatin, the non-cytotoxic and photostable myosin II inhibitor. Angewandte Chemie - International Edition 53(31), 8211–8215 (2014)

21. Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013)

22. Kulikov, V., Kleiner, M., Huberman-Spiegelglas, I., Michaeli, T.: Flowedit: Inversion-free text-based editing using pre-trained flow models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 19721–19730 (2025)

23. Kumar, K.S., Brunner, C., Schuster, M., Luca, L., Alexandre, K., Shen, G., Jurt, S., Moehle, K., Bruns, D., Grotzer, M., Zerbe, O., Schneider, G., Baumgartner, M.: Discovery of a small molecule ligand of FRS2 that inhibits invasion and tumor growth. Cellular Oncology pp. 331–356 (2023)

24. Lamiable, A., Champetier, T., Leonardi, F., Cohen, E., Sommer, P., Hardy, D., Argy, N., Massougbodji, A., Del Nery, E., Cottrell, G., et al.: Revealing invisible cell phenotypes with conditional generative modeling. Nature Communications 14(1), 6386 (2023)

25. Li, S., Kallidromitis, K., Gokul, A., Liao, Z., Kato, Y., Kozuka, K., Grover, A.: Omniflow: Any-to-any generation with multi-modal rectified flows. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 13178–13188 (2025)

26. Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022)

27. Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003 (2022)

28. Palma, A., Theis, F.J., Lotfollahi, M.: Predicting cell morphological responses to perturbations using generative modeling. Nature Communications 16(1), 505 (2025)

29. Perez, E., Strub, F., De Vries, H., Dumoulin, V., Courville, A.: Film: Visual reasoning with a general conditioning layer. In: Proceedings of the AAAI conference on artificial intelligence. vol. 32 (2018)

30. Raudenská, M., Petrláková, K., Juriňáková, T., Leischner Fialová, J., Fojtů, M., Jakubek, M., Rösel, D., Brábek, J., Masařík, M.: Engine shutdown: migrastatic strategies and prevention of metastases. Trends in Cancer 9(4), 293–308 (2023)

31. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)

32. Rout, L., Chen, Y., Ruiz, N., Caramanis, C., Shakkottai, S., Chu, W.S.: Semantic image inversion and editing using rectified stochastic diferential equations. arXiv preprint arXiv:2410.10792 (2024)

33. Schönholzer, M.T., Lin, M.S., Yan, S., Akle, V., Ciraulo, B., Versamento, D., Kopp, L.L., Hochuli, D., Bleiker, T., Welti, A., Neuhauss, S.C., Allalou, A., Baumgartner, M.: The cns-penetrant map4k inhibitor famlasertib restrains medulloblastoma dissemination without developmental toxicity. bioRxiv (2026)

34. Serrano, E., Peters, J., Wagner, J., Graham, R.E., Chen, Z., Feng, B.Y., Miranda, G., Kalinin, A.A., Vulliard, L., Tomkinson, J., Mattson, C., Lippincott, M.J., Kang, Z., Sitani, D., Bunten, D., Seal, S., Carragher, N.O., Carpenter, A.E., Singh, S., Marin Zapata, P.A., Caicedo, J.C., Way, G.P.: Progress and new challenges in image-based profiling. Molecular Systems Biology 22(5), 624–658 (2026)

35. Song, J., Meng, C., Ermon, S.: Denoising difusion implicit models. arXiv preprint arXiv:2010.02502 (2020)

36. Song, Y., Ermon, S.: Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems 32 (2019)

37. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Scorebased generative modeling through stochastic diferential equations. arXiv preprint arXiv:2011.13456 (2020)

38. Uehata, M., Ishizaki, T., Satoh, H., Ono, T., Kawahara, T., Morishita, T., Tamakawa, H., Yamagami, K., Inui, J., Maekawa, M., Narumiya, S.: Calcium sensitization of smooth muscle mediated by a Rho-associated protein kinase in hypertension. Nature 389(6654), 990–994 (oct 1997), https://www.nature.com/ articles/40187

39. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13(4), 600–612 (2004)

40. Way, G.P., Kost-Alimova, M., Shibue, T., Harrington, W.F., Gill, S., Piccioni, F., Becker, T., Shafqat-Abbasi, H., Hahn, W.C., Carpenter, A.E., Vazquez, F., Singh, S.: Predicting cell health phenotypes using image-based morphology profiling. Molecular Biology of the Cell 32(9), 995–1005 (2021)

41. Wu, P.H., Gilkes, D.M., Phillip, J.M., Narkar, A., Cheng, T.W.T., Marchand, J., Lee, M.H., Li, R., Wirtz, D.: Single-cell morphology encodes metastatic potential. Science Advances 6(4), 1–8 (2020)

42. Zhang, Y., Su, Y., Wang, C., Li, T., Wefers, Z., Nirschl, J., Burgess, J., Ding, D., Lozano, A., Lundberg, E., et al.: Cellflux: Simulating cellular morphology changes via flow matching. arXiv preprint arXiv:2502.09775 (2025)

43. Zhang, Y., Su, Y., Wefers, Z., Su, S., Li, H., Li, T., Wang, C., Burgess, J., Lozano, A., Zhou, L., et al.: Cellfluxv2: An image generative foundation model for virtual cell modeling. bioRxiv pp. 2026–01 (2026)

## A Supplementary Material

## A.1 Dataset & Qualitative Real Samples

Figure 5 shows qualitative samples of the training split for the maximum concentrations of C7 (top row) and CK-666 (bottom row).

![](images/9088f9e1e1ddc0c2544a9470691ff08f42ce40ed4da1e61af8a5ea077d17ac27.jpg)  
Fig. 5: Qualitative training set samples for ONS76 cell line under compound treatments. Representative real microscopy images of ONS76 cells treated with C7 (top row, 3µM) and CK-666 (bottom row, 10µM) from the training split.

## A.2 Experimental Results on CK-666

To assess generalization beyond C7, we evaluate the generative capabilities of our of joint model on a second compound, CK-666. We show quantitative results for inversion fidelity and per-concentration distributional fidelity.

Inversion Fidelity. Table 5 reports the inversion fidelity for CK-666. Similarly as for C7 (cf. Tab. 1), round-trip reconstruction values are starting to saturate around $K = 5 0$ to K = 100 ODE steps. In general, values are lower than on C7, likely because diferent compounds induce diferent morphological efects, yielding varying reconstruction dificulty.

Distribution Fidelity. Table 6 reports FID and KID for CK-666, following the same protocol as the C7 evaluation (Sec. 4.3). Our joint model achieves the best KID at four of five concentrations and the best pooled FID overall, on par with IMPA (20.85 vs. 20.93). It surpasses PhenDif on FID at three of five concentrations and shows less variation in FID across concentrations, indicating more consistent behavior. The ablation (Ours cond. only) is again uniformly worse, confirming that the benefit of joint $p ( z , c )$ modeling is consistent across compounds.

Table 5: Inversion fidelity on CK-666 (N=100 control cells).
<table><tr><td>ODE steps K PSNR ↑ SSIM ↑</td></tr><tr><td>10  $2 2 . 2 7 _ { \pm 3 . 2 } \ 0 . 8 1 5 _ { \pm 0 . 0 6 }$ </td></tr><tr><td>25  $2 2 . 9 3 _ { \pm 4 . 0 } \ 0 . 8 3 0 _ { \pm 0 . 0 6 }$ </td></tr><tr><td>50  $2 3 . 4 8 _ { \pm 4 . 2 } \ 0 . 8 3 8 _ { \pm 0 . 0 6 }$  100  $2 3 . 5 0 { \scriptstyle \pm 4 . 3 } 0 . 8 3 9 { \scriptstyle \pm 0 . 0 6 }$ </td></tr><tr><td>200  $2 3 . 4 5 _ { \pm 4 . 3 } \ 0 . 8 3 9 _ { \pm 0 . 0 6 }$ </td></tr><tr><td>500  $2 3 . 3 9 _ { \pm 4 . 3 } \ 0 . 8 3 8 _ { \pm 0 . 0 6 }$ </td></tr><tr><td>VAE only  $3 8 . 7 3 _ { \pm 3 . 1 } \ 0 . 9 1 3 _ { \pm 0 . 0 5 }$ </td></tr></table>

## A.3 Additional Results on UW228

As a proof of concept for the broader applicability of our approach, we apply our joint model to a second cell line (UW228) on the same two compounds (C7 and CK-666). Qualitative morphs (Fig. 6) preserve source-cell identity while producing subtle dose-conditioned variations for both compounds. Quantitatively as shown in Tab. 7, per-concentration FID and KID stay in a narrow band $( \mathrm { F I D } \in [ 2 6 . 3 , 3 0 . 3 ] , \mathrm { ~ K I D } \in [ 1 . 1 7 , 1 . 5 5 ] )$ , and pooled FID is comparable to the ONS76 C7 result reported in the main text (Sec. 4.3) as well as ONS76 CK-666 result reported in A.2, indicating that our method transfers across cell lines without further tuning. A thorough quantitative evaluation with all comparison methods as well as a controllability evaluation remains subject to future work.

## A.4 Extended Concentration Controllability & Analysis

Per-Sample Dose-Response Distributions In Sec. 4.4, we studied the latentspace dose response averaged over $n = 5 0$ samples per method, showing the monotonic behavior of our joint model. Figure 7 extends this to the per-sample level, showing the distribution of latent displacements underlying Fig. 3. The dashed lines mark the concentration-averaged displacement for each method, confirming that our joint model produces the smoothest monotonic mean trajectory across concentrations. In contrast, the conditional-only ablation shows a near-constant mean plateau at low and intermediate doses, only responding at the highest concentration, while PhenDif exhibits a monotonic mean trend between the two. Per-sample distributions overlap substantially across neighboring doses for all methods, consistent with the intrinsic phenotypic heterogeneity at fixed concentrations.

Dose-Response Geometry for Unseen Concentrations To complement the hold-out experiment demonstrated in Sec. 4.4 with a middle concentration $( 0 . 3 3 \mu \mathrm { M } )$ held out on C7, we additionally repeat the same experiment with two additional concentrations held out separately (0.11 µM and 1 µM). The resulting dose-response geometries can be seen in Fig. 8, indicating clear monontonic behaviour for 0.33 µM and 1 $\mu \mathrm { M }$ and a monotonic position that slightly exceeds the succeeding one for 0.11 $\mu \mathrm { M }$ , likely due to the fact that lower concentrations are harder to discriminate.

Table 6: FID/KID comparison (lower is better) for CK-666. Best metrics per column are in bold, second-best underlined.
<table><tr><td>Method</td><td></td><td colspan="5">Concentration  $( \mu \mathrm { M } )$ </td></tr><tr><td></td><td>Metric 0.625</td><td>1.25 (N=1015) (N=1033)(N=1062)(N=1012)</td><td> $2 . 5$ </td><td>5.0</td><td>10.0 (N=956)</td><td>Pooled (N=5078)</td></tr><tr><td>Latent CellFlux [42]†</td><td>FID KID</td><td></td><td></td><td></td><td></td><td>21.48 1.60</td></tr><tr><td>IMPA [28]†</td><td>FID KID</td><td></td><td></td><td></td><td></td><td>20.93 1.23</td></tr><tr><td>PhenDiff [6]</td><td>FID KID</td><td>39.88 28.82 2.58 1.52</td><td>35.60 2.20</td><td>30.72 1.72</td><td>31.33 1.60</td><td>22.70 1.76</td></tr><tr><td>Ours (cond. only)</td><td>FID KID</td><td>43.59 42.43 3.01 2.76</td><td>40.31 2.69</td><td>45.21 3.05</td><td>55.70 3.78</td><td>36.28 2.87</td></tr><tr><td>Ours (Joint Latent FM) FID</td><td>KID</td><td>26.76 29.26 1.39 1.49</td><td>27.31 1.43</td><td>27.84 1.49</td><td>31.99 1.72</td><td>20.85 1.40</td></tr></table>

† Latent CellFlux and IMPA do not natively support concentration conditioning; therefore we report their pooled results without concentration awareness.

Concentration Classification & Supervised Baseline We train a supervised ResNet-18 for multi-class classifications of C7-treated ONS76 cells, providing a supervised ceiling for the concentration estimation task (Sec. 4.4). Table 8 shows the classification accuracy averaged over the 5-class task, whereas Tab. 9 and Fig. 9 show the per-concentration accuracy as well as the full confusion matrices. The supervised ceiling of 48.0% confirms the inherent dificulty of the task, particularly at intermediate doses where real distributions morphologically overlap (cf. Fig. 4). Our unsupervised $p ( c \mid z )$ estimator recovers a substantial share of this ceiling and outperforms the supervised baseline at the lowest concentration $( 0 . 0 3 7 \mu \mathrm { M } \colon 4 4 . 3 \% \ \mathrm { v s } . \ 3 1 . 8 \% )$ , suggesting the continuous concentration formulation enforces sensitivity to subtle low-dose morphologies.

## A.5 Method Ablations & Latency Benchmarks

Concentration Loss Weight $\left( \lambda _ { c } \right)$ Sensitivity Table 10 reports FID and KID for three values of the concentration-loss weight $\lambda _ { c } \in \{ 0 . 1 , 0 . 5 , 1 . 0 \}$ on $\mathrm { C } 7 .$ Our chosen value $\left( \lambda _ { c } = 0 . 5 \right)$ achieves the best or second-best result at every concentration and metric, confirming that the paper’s setting is robust and not sensitive to precise tuning. Doubling $\lambda _ { c }$ to 1.0 uniformly degrades performance, suggesting that over-weighting the concentration flow objective harms the primary latent-space generation task; reducing to 0.1 marginally improves fidelity at the lowest concentration but is otherwise worse. Concentration estimation accuracy (Sec. 4.4) does not strictly correlate with generation quality: $\lambda _ { c } = 0 . 1$ achieves marginally higher top-1 accuracy (36.0% vs. 34.0%), while $\lambda _ { c } ~ = ~ 0 . 5$ achieves higher ±1 neighbor accuracy (68.0% vs. 66.7%).

Table 7: FID and KID for UW228 (our joint model). Metrics are within a narrow range per compound and pooled values match those reported on ONS76 in the main text.
<table><tr><td>C7</td></tr><tr><td>Concentration (µM) 0.037 0.11 0.33 1.0 3.0 Pooled FID 27.09 26.34 27.27 28.71 29.2818.79 KID 1.32 1.24 1.34 1.26 1.36 1.25</td></tr><tr><td>CK-666</td></tr><tr><td>Concentration (µM) 0.625 1.25 2.5 5.0 10.0 Pooled FID 26.61 27.09 27.53 30.30 27.0119.34 KID 1.27 1.17 1.28 1.55 1.25 1.26</td></tr></table>

Table 8: Concentration classification accuracy on C7. Best values in bold.
<table><tr><td>Method</td><td>Top-1 ±1 neighbor</td></tr><tr><td>Joint FM  $p ( c \mid z )$  34.0%</td><td>68.0%</td></tr><tr><td>ResNet-18, real → real 48.0%</td><td>74.8%</td></tr></table>

Sampler & ODE Integration Schemes To complement the inversion fidelity results for C7 in Sec. 4.3, we perform an ablation on the sampler choice. While all primary experiments use standard Euler integration, we evaluate higher-order alternatives: a second-order Heun solver and a fourth-order Runge-Kutta (RK4) scheme across the same step counts. Table 11 presents the resulting roundtrip PSNR and SSIM values. At moderate step counts $( K \in \{ 2 5 , 5 0 \} )$ , Heun and RK4 yield slight gains in reconstruction quality over Euler; however, performance across all three solvers saturates as step count increases $( K \ge 1 0 0 )$ . Given that latent roundtrip fidelity saturates and higher-order solvers yield negligible gains beyond $K \ge 1 0 0$ steps, all subsequent forward generation experiments in this work are conducted using standard Euler integration.

Table 9: Per-concentration accuracy on C7. Best values in bold.
<table><tr><td>Concentration  $( \mu \mathrm { M } )$ </td><td>0.037</td><td>0.11</td><td>0.33</td><td>1.0</td><td>3.0</td></tr><tr><td>Joint FM  $p ( c \mid z )$ </td><td></td><td></td><td>44.3%24.1% 21.5%20.7%54.6%</td><td></td><td></td></tr><tr><td>ResNet-18 (real → real) 31.8% 53.9% 14.5% 48.8% 79.2%</td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/79716e2726c994d3e4a005f6cdf642a2dd78a15491f9b1bad26dbf7605d0106f.jpg)  
Fig. 6: Qualitative dose-response morphs on the UW228 cell line for two compounds (C7 and CK-666), generated by our joint model. Each row shows a real control cell (leftmost column) morphed to successive concentrations.

Sampling Times Table 12 reports sampling latencies on a single NVIDIA GeForce RTX 4090 (batch size 8, 10 trials) using the default evaluation pipelines from Sec. 4.3. Our joint model performs VAE encoding, ODE inversion, concentrationconditioned sampling, and VAE decoding, whereas pixel-space baselines (Phen-Dif, IMPA) skip VAE transformations. The elevated latencies of our conditional model and Latent CellFlux are due to CFG, which doubles the per-step neural network forward evaluations. While difusion- and flow-based models split their compute between inversion and forward generation, PhenDif’s main bottleneck is working directly in pixel space rather than a compressed latent space. In contrast, IMPA relies on a single feed-forward pass, making it by far the fastest method.

![](images/1a180357aa524d280da949fcf86aec9820c40e565483db0f5604cd2b8e83403e.jpg)  
Fig. 7: Per-cell dose-response distributions. Violin plots of normalized latent displacement (pixel-space for PhenDif) from the reference at each concentration; dashed lines connect the concentration means. Normalization per method by the maximum concentration-averaged displacement.

![](images/289b67f7854d517f4afadeda7ce319952533c35cfea52108af1edf32baaed475.jpg)

![](images/fe11777669bfe70d4b79de463830526560ce6adfaa8f0acf79c569247c05f575.jpg)

![](images/bf293fc040f64a2d97f71fe24b216e6a34f83e8fe9f9c7279638b0fe50063751.jpg)  
Fig. 8: Dose-response geometry for unseen concentrations in C7. After retraining our joint model with 0.11, 0.33, or $1 \mu \mathrm { M }$ held out during training (panels left to right, respectively), forward integration at the held-out dose (highlighted star) generally interpolates between neighboring concentrations, closely following the expected monotonic trend.

Table 10: Ablation over $\lambda _ { c }$ on C7. Best values in bold.
<table><tr><td colspan="6">Concentration (µM)</td></tr><tr><td> $\lambda _ { c }$ </td><td>Metric</td><td>0.037</td><td>0.11</td><td>0.33 1.0</td><td>3.0 Pooled</td></tr><tr><td>0.1</td><td>FID KID</td><td>29.83 1.56</td><td>28.52 30.15 1.51 1.62</td><td>29.95 1.58</td><td>26.16 20.42 1.41 1.37</td></tr><tr><td>0.5</td><td>FID KID</td><td>30.02 1.58</td><td>27.20 27.50 1.40 1.37</td><td>27.85 1.39</td><td>25.02 18.98 1.28 1.25</td></tr><tr><td>1.0</td><td>FID KID</td><td>31.48 1.73</td><td>28.21 1.51</td><td>28.91 30.41 1.51 1.67</td><td>25.66 20.32 1.33 1.38</td></tr></table>

![](images/fa23abd49242abe47cd6692c70b9c35a96e7137c41b1b30cce92805822bb67f7.jpg)

![](images/e94d802dc3ec321fd6dde49466709e34345d8c9c049085297d02542b9b2bcd66.jpg)  
Fig. 9: Confusion matrices for 5-class concentration prediction in C7. Left: Confusion matrix of our joint flow matching model $p ( c \mid z )$ , showing that lower and intermediate concentrations in particular are harder to discriminate, confirming the findings in Sec. 4.4. Right: The corresponding supervised ResNet-18 baseline yields better results at $0 . 1 1 \mu \mathrm { M } , 1 \mu \mathrm { M } ,$ and $3 \mu \mathrm { M }$ , but performs worse at $0 . 0 3 7 \mu \mathrm { M }$ and $0 . 3 3 \mu \mathrm { M }$

Table 11: Inversion fidelity comparison across ODE integration schemes on the C7 test set $( N { = } 1 0 0$ control cells).
<table><tr><td rowspan="2">Steps K</td><td colspan="3">PSNR ↑</td><td colspan="3">SSIM ↑</td></tr><tr><td>Euler</td><td>Heun</td><td>RK4</td><td>Euler</td><td>Heun</td><td>RK4</td></tr><tr><td>10</td><td> $2 7 . 7 9 { \scriptstyle \pm 2 . 9 }$ </td><td> $2 7 . 3 1 { \scriptstyle \pm 2 . 1 }$ </td><td> $2 7 . 1 4 { \scriptstyle \pm 2 . 1 }$ </td><td> $0 . 8 6 6 _ { \pm 0 . 0 4 }$ </td><td> $0 . 8 7 3 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $0 . 8 7 2 _ { \pm 0 . 0 4 }$ </td></tr><tr><td>25</td><td> $3 1 . 9 3 { \scriptstyle \pm 3 . 0 }$ </td><td> $3 2 . 0 9 { \scriptstyle \pm 2 . 6 }$ </td><td> $3 2 . 0 0 { \scriptstyle \pm 2 . 6 }$ </td><td> $0 . 8 9 4 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $0 . 9 0 0 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $0 . 9 0 0 { \scriptstyle \pm 0 . 0 4 }$ </td></tr><tr><td>50</td><td> $3 3 . 8 1 _ { \pm 3 . 0 }$ </td><td> $3 3 . 9 9 { \scriptstyle \pm 2 . 8 }$ </td><td> $3 3 . 9 7 _ { \pm 2 . 8 }$ </td><td> $0 . 9 0 2 _ { \pm 0 . 0 4 }$ </td><td> $0 . 9 0 6 _ { \pm 0 . 0 5 }$ </td><td> $0 . 9 0 6 _ { \pm 0 . 0 5 }$ </td></tr><tr><td>100</td><td> $3 4 . 5 0 { \scriptstyle \pm 3 . 0 }$ </td><td> $3 4 . 6 5 { \scriptstyle \pm 2 . 9 }$ </td><td> $3 4 . 6 5 { \scriptstyle \pm 2 . 9 }$ </td><td> $0 . 9 0 5 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 9 0 7 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 9 0 7 { \scriptstyle \pm 0 . 0 5 }$ </td></tr><tr><td>200</td><td> $3 4 . 7 0 { \scriptstyle \pm 3 . 0 }$ </td><td> $3 4 . 8 1 { \scriptstyle \pm 3 . 0 }$ </td><td> $3 4 . 8 1 { \scriptstyle \pm 3 . 0 }$ </td><td> $0 . 9 0 6 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 9 0 7 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 9 0 7 { \scriptstyle \pm 0 . 0 5 }$ </td></tr><tr><td>500</td><td> $3 4 . 7 7 _ { \pm 3 . 0 }$ </td><td> $3 4 . 8 1 _ { \pm 3 . 0 }$ </td><td> $3 4 . 8 1 _ { \pm 3 . 0 }$ </td><td> $0 . 9 0 7 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 9 0 7 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 9 0 7 { \scriptstyle \pm 0 . 0 5 }$ </td></tr><tr><td>VAE only</td><td></td><td> $3 8 . 7 3 { \scriptstyle \pm 3 . 1 }$ </td><td></td><td></td><td> $0 . 9 1 3 { \scriptstyle \pm 0 . 0 5 }$ </td><td></td></tr></table>

Table 12: Sampling latency comparison across generative methods and baseline models.
<table><tr><td>Method</td><td>Batch Latency (ms) ↓ Per-Cell (ms) ↓</td><td></td></tr><tr><td>Latent CellFlux</td><td> $4 1 9 6 . 5 { \scriptstyle \pm 8 . 2 }$ </td><td>524.6</td></tr><tr><td>IMPA</td><td> $1 7 6 . 2 { \scriptstyle \pm 0 . 3 }$ </td><td>22.0</td></tr><tr><td>PhenDiff</td><td> $2 6 1 8 9 . 5 { \scriptstyle \pm 3 1 . 9 }$ </td><td>3273.7</td></tr><tr><td>Ours (cond. only)</td><td> $6 3 6 7 . 3 { \scriptstyle \pm 5 1 . 2 }$ </td><td>795.9</td></tr><tr><td>Ours (Joint Latent FM)</td><td> $3 2 3 7 . 7 { \scriptstyle \pm 3 1 . 4 }$ </td><td>404.7</td></tr></table>