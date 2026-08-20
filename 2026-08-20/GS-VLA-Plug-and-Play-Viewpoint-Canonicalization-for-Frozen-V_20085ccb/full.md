# GS-VLA: Plug-and-Play Viewpoint Canonicalization for Frozen VLA Policies via Gaussian Splatting

Yechan Park Dankook University Yongin, Republic of Korea yechan3219@dankook.ac.kr

HyunJin Kim Dept. of EEE, Dankook University Yongin, Republic of Korea hyunjin2.kim@gmail.com (Corresponding author)

![](images/7f8a6c0e7202957d41dc96ca44ce87a487d52cc06f3c71145bb13abe7fbf914a.jpg)

## Abstract

This paper proposes a lightweight, plug-and-play framework that improves robustness to viewpoint shifts in Vision-Language-Action (VLA) policies without policy retraining. To our knowledge, this is the first approach to directly leverage 3D Gaussian-based novel-view synthesis for observation-space adaptation in VLA policies. Current VLA performance relies on the implicit assumption that training and deployment camera configurations are identical. Our experiments show that even a small displacement of the camera mount can reduce the success rate on the LIBERO benchmark from about 90% to about 10% in the worst case. Prior approaches, such as large-scale fine-tuning or generative data augmentation, are computationally expensive and risk catastrophic forgetting. To address this, viewpoint shifts are reformulated as a localized novel-view synthesis problem. Under a Locality assumption, that camera perturbations remain within a small bounded region relative to the workspace, viewpoint normalization reduces to a scene- and policy-independent disocclusion task. Our work implements this idea with a 4Mparameter 3D-Gaussian canonicalizer prepended to a frozen VLA policy. Without modifying policy weights, GS-VLA improves performance across three orthogonal

Preprint.

axes: (1) Policy architectures, (2) Unseen task suites, and (3) Perturbation scales. These results show that a lightweight visual module can recover a large fraction of the performance lost under viewpoint shift, without policy retraining.

## 1 Introduction

Vision-Language-Action (VLA) models [3, 44, 32, 15, 16, 2, 11, 34, 7] have become a standard policy class for general manipulation. They perform strongly on benchmarks such as LIBERO [17] and CALVIN [21] and can transfer to real-world robots without additional modification. However, these successes often rely on an implicit assumption: the visual conditions at deployment closely match those used during training. In practice, even small deviations in camera viewpoint can lead to substantial performance degradation, highlighting viewpoint robustness as an important open challenge for reliable real-world deployment. However, improving robustness through additional data collection and policy fine-tuning is often impractical in real deployment settings, where new demonstrations are costly or unavailable. Also, there is a risk of policy collapse; trying to fine-tune the model for new camera views might ruin its original manipulation skills—a problem often called catastrophic forgetting.

More fundamentally, performance drops under viewpoint shifts do not necessarily reflect failures in high-level reasoning or control capabilities; rather, they may stem from a mismatch between deployment-time observations and those encountered during training. This raises a broader question: should robustness to viewpoint variation be learned entirely within policy parameters, or can it be achieved by adapting the observation space instead?

The latter observation space adaptation is not only possible but also cost-effective. In many realworld setups, camera changes are usually small and stay within a limited range. In practice, these perturbations are often small relative to the workspace, typically amounting to only a few to a few tens of centimeters. We refer to this practical property as Locality. Under this assumption, viewpoint changes affect only a limited portion of the image, while most pixels remain geometrically recoverable. Specifically, most of the visual content can be mapped to the canonical view using depth information and camera parameters. This means that learning is mainly needed for a narrow missing region, which can be treated as an inpainting problem. Importantly, this problem depends more on the visual domain than on the particular scene or policy. In this sense, viewpoint canonicalization is not a full novel-view-synthesis problem, but a simpler local adaptation problem that is largely independent of the scene and policy. We treat Locality as an empirically testable assumption.

This makes viewpoint adaptation much more practical. Under Locality, there is no need for a large view-synthesis model designed for arbitrary scenes and camera baselines. Instead, a lightweight canonicalization module can be placed before a frozen VLA policy to restore aligned observations. This plug-and-play design recovers a substantial portion of the lost performance across different policy architectures, task suites, and perturbation scales without any policy retraining.

The main contributions of this paper are as follows:

• A new formulation of viewpoint adaptation for VLA policies: We show that viewpoint adaptation for frozen VLA policies can be addressed in observation space rather than in policy parameters, and formalize this perspective through Locality, a practical premise that reduces the problem to local disocclusion handling.

• The first Gaussian-based canonicalization module for VLA policies: To the best of our knowledge, this is the first work to directly use a 3D Gaussian-based module for observationspace adaptation in VLA policies, thereby enabling efficient viewpoint canonicalization without policy retraining. The resulting module is lightweight and efficient to train, consisting of a 4M-parameter U-Net [27] adapter that predicts 14-dimensional Gaussian attributes and converges in only 5 hours on a single RTX A5000.

• Generalization across policies, tasks, and perturbation scales: We demonstrate that a single checkpoint transfers across multiple policy architectures, unseen task suites, and large camera perturbations, recovering substantial performance without axis-specific re-training.

• Empirical analysis of the recovery mechanism: We provide empirical evidence that most of the target view is handled by deterministic geometric mapping, while learning is needed only for a spatially localized residual region.

Taken together, the above contributions suggest that viewpoint robustness for frozen VLA policies can be addressed effectively through lightweight observation-space adaptation, without modifying the policy itself. The rest of the paper is organized as follows. Firstly, prior works on VLA robustness and novel-view synthesis are reviewed. We then present our Locality-based formulation and the proposed 3D Gaussian canonicalization module. Finally, we evaluate the method across multiple policies, task suites, and perturbation scales, and provide an analysis of the recovery mechanism.

## 2 Related Works

VLA Foundation Policies. Recent VLA models [3, 44, 2, 11, 15, 16, 25, 34, 7, 32, 42, 41, 18, 40, 13] unified multi-modal inputs under large Transformer backbones trained on massive cross-embodiment datasets. Although these models achieved ≥ 90% success on LIBERO [17], their performances were predicated on static training camera setups. Unlike works that modified internal activations, we treat these multi-billion-parameter models as frozen, black-box policies and improve viewpoint robustness entirely in the observation space. This preserves the integrity of the policy while avoiding the costs of retraining.

Camera-Robust Robot Learning. Standard approaches to viewpoint robustness relied on retraining policies under augmented camera distributions. SPARTN [43] utilized NeRF [22] to generate novelview trajectories for behavior cloning. The concurrent AnyCamVLA [10] similarly froze the policy but relied on LVSM [12] synthesizer with dual-view inputs, with evaluation limited to narrow perturbations. On the other hand, we introduce a Locality-based reformulation that reduces the number of parameters by 42× compared with AnyCamVLA.

Feed-Forward Novel-View Synthesis. Novel-view synthesis (NVS) generates images from new camera poses given one or more input views. Per-scene radiance fields [22, 23, 1] achieve high fidelity but require per-scene optimization, while feed-forward predictors [38, 33] generalize across scenes. More recently, diffusion-based generators [28, 35] synthesize plausible views even under large baselines, at the cost of tens to hundreds of millions of parameters. Our contribution lies not in proposing a new NVS architecture but in reformulating the problem itself: by restricting the task to a bounded ε-ball under the Locality premise, we reduce capacity requirements by one to two orders of magnitude.

3D Gaussian Splatting Variants. Building on 3DGS [14] and its differentiable rasterizer [37], recent feed-forward variants predict per-pixel Gaussians from one or more views [30, 31, 38, 6] and improve rendering quality and structure [39, 20]. Flash3D in particular leverages monocular depth priors but targets cross-scene generalization.Our canonicalizer follows the pixel-aligned line but specializes the task to in-distribution camera perturbations, so the U-Net only needs to shape per-pixel Gaussians within an ε-ball around the training pose rather than synthesize arbitrary views.

Image Inpainting. Filling disocclusions resembles image inpainting [29, 26], but state-of-the-art inpainters require tens of millions of parameters to hallucinate semantic content over free-form holes. In our setting, ε-bounded camera shifts produce only O(ε)-thin holes along depth discontinuities, so no hallucination is needed and a 4M-parameter U-Net suffices

VLA Robustness Benchmarks. LIBERO-Plus [8] recently showed that camera viewpoint is a major source of fragility for state-of-the-art VLAs. In response to this challenge, we introduce an observation-space remedy and evaluate its effectiveness across three axes: policy, task suite, and perturbation scale. Unlike concurrent approaches targeting other robustness dimensions, such as SpatialVLA [25], our observation-space remedy is orthogonal to internal policy modifications and can be combined with them to achieve more comprehensive robustness.

## 3 Proposed method

This section presents our formulation of the Locality assumption as a practical module. We first define the objective of the canonicalizer, then show that Locality allows a computationally expensive novel-view synthesizer to be replaced with a lightweight U-Net. Additionally, the training procedure and the module deployment are described.

## 3.1 Motivation

A VLA policy trained at a fixed camera pose tends to collapse once the camera moves at deployment; in our worst LIBERO suite, the success rate decreases from 92.4% to 9.3%. A natural remedy is to preprocess the observation back to the canonical view before feeding it to the policy. Although it seems like we need to solve a complex NVS problem, the changes in the camera position are actually very small. In practical settings, rig-mounted cameras typically drift by only a few centimeters, such that the majority of pixels remain visible after the pose change, while only a narrow region near depth discontinuities becomes newly exposed.

That band has area $O ( \rho )$ in the perturbation magnitude $\rho . \mathrm { \bf A }$ geometric argument for this area bound and the underlying geometry are detailed in Appendix A.1. Therefore, the core challenge of the learning problem reduces to inpainting a thin, newly disoccluded band along depth-discontinuity curves rather than synthesizing a general novel view. This makes our canonicalizer a natural fit: rather than generating the canonical image from scratch, it geometrically transports the visible source content into the canonical frame, leaving learning to handle only the narrow disoccluded regions. As a result, the learnable component remains highly lightweight—a 4M-parameter U-Net is all we need, which is over $2 5 \times$ smaller than generic feed-forward NVS predictors [12, 31, 30]. We embed this same locality structure inside the architecture: each source pixel is back-projected through its depth to anchor a Gaussian primitive, the U-Net shapes each primitive (depth-bounded center offset, anisotropic scale, orientation, opacity, and color residual), and differentiable α-splatting [37] composes them at the canonical pose, jointly resolving occlusion ordering and blending across the disoccluded bands. Architectural details and the 14-dim descriptor are provided in Section 3.3.

## 3.2 Problem definition

Setup and notation. We first define the input-output interface of the viewpoint canonicalizer $f _ { \phi }$ A frozen VLA policy π is trained at a canonical pose (the camera pose seen during $\pi _ { \theta }$ training), $C ^ { \star } = ( R ^ { \star } , t ^ { \star } ) \bar { \in } \dot { \mathrm { S E } } ( 3 )$ . Here, SE(3) denotes the group of 3D rigid transformations, represented by a rotation and a translation without scale, and $\mathrm { S } \bar { \mathrm { O } } ( 3 )$ denotes the group of 3D rotation matrices. We write $R ^ { \star } \in \mathrm { S O } ( 3 )$ for the camera rotation and $t ^ { \star } \in \mathbb { R } ^ { 3 }$ for its world-space position. The camera intrinsics are given by the pinhole matrix $K \in \mathbb { R } ^ { 3 \times 3 }$ , which encodes the focal lengths and principal point. At deployment, the camera may move to a different source pose $C _ { s } = ( \mathbf { \bar { \boldsymbol { R } } } _ { s } , t _ { s } )$ . For each frame, the canonicalizer receives an RGB image $I _ { s } \in [ 0 , 1 ] ^ { H \times W \times 3 }$ , a per-pixel metric depth map $D _ { s } \in \mathbb { R } _ { > 0 } ^ { H \times W }$ aligned with $I _ { s } ,$ , and the camera parameters $( C _ { s } , C ^ { \star } , K )$ . Here, $D _ { s } ( u , v )$ denotes the depth, in meters, of the 3D point observed at pixel $( u , v )$ in $I _ { s } .$ . Throughout the paper, we use the camera-to-world convention: a 3D point $X _ { c } \in \mathbb { R } ^ { 3 }$ expressed in the camera frame is mapped to the world frame by $X _ { w } = R X _ { c } + t .$ . The goal of the canonicalizer is to predict the image that would have been observed from the canonical pose, which is formulated as:

$$
f _ { \phi } : ( I _ { s } , D _ { s } , C _ { s } , C ^ { \star } , K ) \longmapsto { \widehat { I } } ^ { \star } \in [ 0 , 1 ] ^ { H \times W \times 3 } ,\tag{1}
$$

where $\widehat { I } ^ { \star }$ is the predicted canonical-view image. The frozen policy then acts on this canonicalized observation, $\mathrm { i . e . , } a = \pi _ { \theta } ( \widehat { I } ^ { \star } )$ . The policy parameters $\theta$ are never updated, and $f _ { \phi }$ operates strictly on a per-frame basis, without temporal aggregation, multi-view input, or per-scene optimization.

Locality assumption. We formalize the Locality assumption described in Introduction Section. The deployment pose is assumed to remain within a bounded neighborhood of the canonical pose, with translation tolerance $\varepsilon _ { t } > 0$ and rotation tolerance $\varepsilon _ { R } > 0$ . The Locality set, $B ( C ^ { \star } ; \varepsilon _ { t } , \varepsilon _ { R } )$ , is defined as:

$$
\begin{array} { r } { \mathcal { B } ( C ^ { \star } ; \varepsilon _ { t } , \varepsilon _ { R } ) : = \Big \{ ( R , t ) : \| t - t ^ { \star } \| _ { 2 } \leq \varepsilon _ { t } , \ \| \log \bigl ( ( R ^ { \star } ) ^ { \top } R \bigr ) \| _ { 2 } \leq \varepsilon _ { R } \Big \} , } \end{array}\tag{2}
$$

where

$$
C _ { s } \in B ( C ^ { \star } ; \varepsilon _ { t } , \varepsilon _ { R } ) .\tag{3}
$$

In (2), the term $\| \mathrm { l o g } ( ( R ^ { \star } ) ^ { \top } R ) \| _ { 2 }$ represents the geodesic angle between the current rotation $R$ and the canonical rotation $R ^ { \star }$ . In our experiments, we relate the rotation limit $\varepsilon _ { R }$ to the translation limit $\varepsilon _ { t }$ via a first-order scaling derived from the motion-field equation under a pinhole camera. Whereas translation-induced image motion scales inversely with depth, rotation induces depth-independent motion [19, 9]. Therefore, we use the approximation $\varepsilon _ { R } \approx \varepsilon _ { t } / d _ { \operatorname* { m i n } }$ , where $d _ { \mathrm { m i n } }$ denotes the minimum scene depth. This allows us to control the overall camera perturbation using a single parameter, $\varepsilon _ { t } .$ To measure the total pose change, we define a normalized pose distance, $\rho ,$ as:

$$
\rho = \frac { \| t _ { s } - t ^ { \star } \| } { d _ { \operatorname* { m i n } } } + \| \log ( { R ^ { \star } } ^ { \top } R _ { s } ) \| ,\tag{4}
$$

which combines translation and rotation into a unified scalar.

Under our Locality assumption, the fraction of newly exposed regions, $\eta ( \rho )$ , grows linearly to leading order as $\eta ( \rho ) = \dot { \alpha _ { \mathrm { p a r } } } \rho _ { \mathrm { p a r } } \dot { + } \alpha _ { \mathrm { f r } } \rho _ { \mathrm { f r } } + O ( \rho ^ { 2 } )$ for $\rho \leq 1$ in matched units, with $\rho _ { \mathrm { p a r } } = \| t _ { s } - t ^ { \star } \| / d _ { \operatorname* { m i n } }$ and $\rho _ { \mathrm { f r } } = \| \log ( R ^ { \star ^ { \top } } R _ { s } ) \|$ . The two leading coefficients depend only on scene geometry: $\alpha _ { \mathrm { p a r } }$ on the silhouette length $L _ { \mathrm { p x } } ^ { \star }$ and the relative depth contrast $( d _ { \mathrm { m a x } } - d _ { \mathrm { m i n } } ) / d _ { \mathrm { m a x } }$ , and $\alpha _ { \mathrm { f r } }$ on the image-frame perimeter. We give a geometric argument for this bound, together with the explicit forms of $\alpha _ { \mathrm { p a r } }$ and $\alpha _ { \mathrm { f r } }$ , in Appendix A.1.

We test this scaling empirically along three additional settings: Section 4.3 sweeps the perturbation scale $\varepsilon _ { t } \in \{ 5 , 1 5 , 3 0 , 5 0 , 1 0 0 , 2 0 0 \}$ cm; Section 4.5 stresses calibration noise; and Appendix A.2 verifies the linearity of $\eta ( \rho )$ directly.

## 3.3 Pixel-aligned 3D-Gaussian architecture

The canonicalizer $f _ { \phi }$ is realized as follows. Under the Locality assumption, we only need to handle a thin disocclusion band, so we use a compact U-Net rather than a generic NVS backbone [14, 31, 30], which can reduce computational costs. A symmetric U-Net takes the channel-wise concatenation $[ I _ { s } \parallel D _ { s } ]$ as input and is FiLM-modulated [24] by a pose-pair embedding, which is formulated a $\mathrm { ~ s ~ } \ddot { z } _ { \mathrm { p o s e } } = \mathrm { M } \dot { \mathrm { L P } } ( C _ { s } , C ^ { \star } ) \in \mathbb { R } ^ { 1 2 8 }$ . At every source pixel, the decoder predicts a 14-dimensional Gaussian descriptor consisting of a position residual $( \Delta \mu )$ , an anisotropic log-scale (log s), an orientation quaternion $( q )$ , an opacity (α), and a color residual $( \Delta c )$ . All heads are zero-initialized so that $f _ { \phi }$ starts exactly at the geometric warp. The Gaussian center is anchored at the world-space back-projection $X _ { w } = R _ { s } K ^ { - 1 } D _ { s } ( u , v ) [ u , v , 1 ] ^ { \top } + t _ { s }$ and is allowed only a depth-bounded residual,

$$
\mu = X _ { w } \ + \ c _ { \Delta \mu } D _ { s } ( u , v ) \ \mathrm { t a n h } ( \Delta \mu ) , \qquad c _ { \Delta \mu } = 0 . 1 ,\tag{5}
$$

which caps displacement at 10% of local depth. This keeps the residual from drifting away from the geometric base, which we found to be important for stability.

One Gaussian per source pixel, with RGB $I _ { s } ( u , v ) + \Delta c ,$ are rasterized at the canonical pose $C ^ { \star }$ by gsplat’s [37] differentiable α-blender, $\begin{array} { r } { \widehat { I } ^ { \star } ( p ) = \sum _ { i \in \mathcal { N } ( p ) } T _ { i } \alpha _ { i } c _ { i } } \end{array}$ with $\begin{array} { r } { T _ { i } = \prod _ { i < i } ( 1 - \alpha _ { j } ) } \end{array}$ Occlusion is resolved by depth-sorting, and the α-blender itself fills the disocclusion band, so we do not need a separate inpainting head. The canonicalizer only requires 4M parameters, roughly $8 4 0 \times$ smaller than finetuning π<sub>0.5</sub> [11] and 42 × smaller than AnyCamVLA [10].

## 3.4 Training

We intentionally avoid specialized training techniques so that the observed gains can be attributed to the Locality reduction rather than to the training method. We train on only one suite so that the remaining suites can serve as unseen test domains for cross-suite evaluation. The objective is the standard 3DGS [14] 0.8:0.2 L /SSIM reconstruction loss:

$$
\mathcal { L } = 0 . 8 \Vert \widehat { I } ^ { \star } - I ^ { \star } \Vert _ { 1 } + 0 . 2 \big ( 1 - \mathrm { S S I M } \big ) .\tag{6}
$$

## 4 Experiments

In our experiments, we evaluate GS-VLA in terms of magnitude, generality, and deployment realism: we quantify the extent to which a canonicalizer recovers the performance lost under perturbations, relative to an upper bound given by the unperturbed (best-case) performance and a lower bound corresponding to the perturbed setting without canonicalization. We also evaluate whether a single checkpoint generalizes across different policies, benchmark suites, and perturbation levels ε. Additionally, we assess whether the performance gains persist under realistic deployment conditions, specifically in the presence of simultaneous translation–rotation drift and extrinsic-pose calibration error. All experiments use a single checkpoint trained once on libero\_spatial at $\varepsilon _ { t } = 1 0 0 \mathrm { c m } ;$ no axis-specific retraining is performed.

## 4.1 Experimental setup

Benchmark. We evaluate on the four LIBERO suites [17]—libero\_spatial, libero\_object, libero\_goal, and libero\_10—which cover diverse manipulation scenarios including spatial reasoning, object-centric interactions, and long-horizon tasks. Performance is measured by task success rate over multiple episodes.

Camera randomization. We add random Gaussian noise to both the camera’s position and its look-at point around the canonical pose. The parameter $\varepsilon _ { t }$ controls the size of this noise. We then adjust the camera’s orientation so it directly faces the new look-at point. If a shifted camera falls below the table, we discard it and sample a new one. Our default setting uses $\varepsilon _ { t } \approx 1 0 0 \mathrm { c m }$ , and we apply more noise in the sideways and vertical directions (orthogonal to the view) than in the depth direction. To test the model’s robustness at different noise levels, we simply scale this default setting. Figure 1 visualizes the resulting camera distribution. Finally, in order to isolate the effects of calibration errors, we inject independent location and rotation noise into the canonicalizer’s input. This simulates real-world scenarios in which the camera pose assumed by the system deviates slightly from the true pose.

![](images/5b611d6386c2091b58505518eac5553cb46bd3a05c0fb8def29b585522b35ac2.jpg)  
Figure 1: Visualization of the randomized camera distribution.

Policies. The main policy is $\pi _ { 0 . 5 }$ (lerobot/pi05\_libero\_finetuned) [4]. For cross-policy transfer, we additionally evaluate OpenVLA-OFT [16], RynnVLA-002 [5], and XVLA [42], where all policies remain frozen throughout. Because rollout uses each policy’s native action-chunk configuration, we compare $\Delta$ rather than absolute SR across policies.

Baselines. We evaluate two baselines. NO CANON passes the perturbed observation to the policy without canonicalization. FWD-WARP applies the closed-form depth-warp and zeros out the disocclusion region, which isolates the geometric component. Finally, GS-VLA denotes our proposed 4M-parameter canonicalizer detailed in Section 3.3.

## 4.2 Experimental results

Our first question is how much of the performance drop caused by camera perturbations can be recovered by a single canonicalizer across three critical axes: policy, suite, and perturbation magnitude. Table 1 summarizes these headline results. A single GS-VLA checkpoint consistently improves success rates (SR) across every evaluated setting. Consistent with our locality assumption, the gains are largest precisely where the unprotected baseline policy fails most severely. In these cases, a larger fraction of the raw input pixels is out-of-distribution, leaving greater room for the canonicalizer to restore useful visual structure.

We break down the recovery along the three generalization axes:

• Cross-policy generalization: GS-VLA effectively acts as a plug-and-play module for various architectures. As shown in Table 1(a), every frozen policy benefits without any fine-tuning. For instance, the pilot XVLA policy jumps from a near-total failure of 1.4% to 81.0% (+79.6 pp). Similarly, the 7B-parameter OpenVLA-OFT recovers dramatically to 81.6% (+61.8 pp). Notably, the magnitude of the benefit anti-correlates with the policy’s intrinsic robustness, suggesting that the canonicalizer serves as a broadly effective protective layer across policies.

• Cross-suite generalization: Because the scene structure depends only on silhouette geometry and not on task semantics, our module transfers zero-shot to unseen environments. As shown in Table 1(b), the most striking gain occurs on the long-horizon libero\_10 suite, boosting $\pi _ { 0 . 5 }$ from 9.3% to 72.1% (+62.8 pp). We attribute this especially large improvement to reduced error accumulation over long trajectories: by keeping the inpainted region small at each step, the locality property limits frame-to-frame drift and prevents errors from compounding over time.

Table 1: Three-axis generalization from a single checkpoint. GS-VLA is trained once on libero\_spatial at $\scriptstyle \varepsilon _ { t } = 1 0 0 \mathrm { c m }$ , transfers without retraining across (a) policies, (b) suites, and (c) perturbation scales. Within each block,by descending $\Delta$ in (a) and (b), by descending $\varepsilon _ { t }$ in (c). Boldface indicates the best result in each row, while bold ∆ highlights the headline cell of each block. The Fwd-warp column reports a depth-only forward-warp baseline where measured. Full per-cell matrices in Table 10 and Section 4.3.
<table><tr><td>Policy / Setting</td><td>Suite</td><td>No canon SR (%)</td><td>Fwd-warp SR (%)</td><td>GS-VLA (ours) SR (%)</td><td> $\Delta \left( \mathbf { p } \mathbf { p } \right)$ </td></tr><tr><td colspan="6">(a) Cross-policy generalization (εt=100 cm, 500 ep)</td></tr><tr><td>XVLA OpenVLA-OFT (7 B)</td><td>spatial</td><td>1.4 19.8</td><td>19.7 30.5</td><td>81.0</td><td>+79.6</td></tr><tr><td>RynnVLA-002</td><td>object spatial</td><td>25.8</td><td>37.8</td><td>81.6 78.6</td><td> $+ 6 1 . 8$   $+ 5 2 . 8$ </td></tr><tr><td>π0.5 (3.4 B)</td><td>spatial</td><td>42.6</td><td>56.5</td><td>86.8</td><td>+44.2</td></tr><tr><td>(b) Cross-suite generalization</td><td></td><td> $( \pi _ { 0 . 5 } , \varepsilon _ { t } { = } 1 0 0 \mathrm { c m } , 1 0 0 0 \mathrm { e p } )$ </td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td>π0.5</td><td>libero_10</td><td>9.3</td><td>22.9</td><td>72.1</td><td>+62.8</td></tr><tr><td>π0.5</td><td>spatial</td><td>42.6</td><td>56.5</td><td>86.8</td><td>+44.2</td></tr><tr><td>π0.5</td><td>object</td><td>64.0</td><td>74.6</td><td>90.2</td><td>+26.2</td></tr><tr><td>π0.5</td><td>goal</td><td>66.3</td><td>72.4</td><td>92.2</td><td>+25.9</td></tr><tr><td colspan="6">(c) Perturbation-scale robustness</td></tr><tr><td> $\varepsilon _ { t } = 2 0 0 \mathrm { c m }$ </td><td>spatial</td><td> $( \pi _ { 0 . 5 } \times \mathtt { s p a t i a l }$  35.7</td><td>1000 ep) 49.5</td><td>78.5</td><td>+42.8</td></tr><tr><td> $\varepsilon _ { t } = 1 0 0 \mathrm { c m }$ </td><td>spatial</td><td>42.6</td><td>56.5</td><td>86.8</td><td>+44.2</td></tr><tr><td> $\varepsilon _ { t } = 5 0 \mathrm { c m }$ </td><td>spatial</td><td>60.9</td><td>66.0</td><td>87.2</td><td>+13.7</td></tr><tr><td> $\varepsilon _ { t } = 5 \mathrm { c m }$ </td><td>spatial</td><td>85.2</td><td>87.5</td><td>88.0</td><td>+2.8</td></tr></table>

Table 2: ε -sweep on $\pi _ { 0 . 5 } \times$ libero\_spatial. The baseline drops 49.5 pp across the sweep; GS-VLA drops only 9.5 pp. Term $\Delta$ grows approximately linearly in $\varepsilon _ { t }$ within the Locality bound, matching Appendix A.1; the largest cell $_ { ( \varepsilon _ { t } = 1 0 0 \mathrm { c m ) } }$ is in bold.
<table><tr><td>Method  $\backslash \varepsilon _ { t }$  (cm)</td><td>5</td><td>15</td><td>30</td><td>50</td><td>100</td><td>200</td></tr><tr><td>No canon</td><td>85.2</td><td>82.7</td><td>80.6</td><td>73.5</td><td>42.6</td><td>35.7</td></tr><tr><td>GS-VLA (ours)</td><td>88.0</td><td>88.6</td><td>87.6</td><td>87.2</td><td>86.8</td><td>78.5</td></tr><tr><td>∆(pp)</td><td>+2.8</td><td>+5.9</td><td>+7.0</td><td>+13.7</td><td>+44.2</td><td>+42.8</td></tr></table>

• Perturbation-scale robustness: On the core training setting $( \pi _ { 0 . 5 }$ at $\varepsilon _ { t } = 1 0 0 \mathrm { c m } )$ , GS-VLA improves SR from 42.6% to 86.8% (+44.2 pp). Even under extreme noise $( \varepsilon _ { t } = 2 0 0 \mathrm { c m } )$ where the baseline falls to 35.7%, GS-VLA maintains high robustness at 78.5% (+42.8 pp), as seen in Table 1(c).

## 4.3 Scaling analysis

In order to know how performance changes as the perturbation grows, we sweep $ { \varepsilon } _ { t } \in$ {5, 15, 30, 50, 100, 200} cm across all suites (Tables 2, 12). Three distinct regimes emerge: on spatial and object, the unprotected policy stays near the performance ceiling for small $\varepsilon _ { t } ,$ with the canonicalizer’s benefit (∆) growing sharply only past 50 cm. On goal, the gap remains consistently large (+16 to +30 pp) across all values of $\varepsilon _ { t } .$ , indicating that the policy is highly sensitive even to modest camera drift.

On libero\_10, the canonicalizer outperforms the baseline across all perturbation levels, with the gain peaking at $\mathbf { a + 6 2 . 8 }$ pp gain at $\varepsilon _ { t } = 1 0 0$ cm before slightly tapering off at 200 cm as the perturbation approaches the Locality boundary $( \mathrm { i . e . , a s } \rho \to 1 )$ . Within the $\varepsilon _ { t } \leq 1 0 0$ cm range, the $\Delta ( \varepsilon _ { t } )$ curve is approximately linear, and the per-suite slopes match our predicted $\alpha _ { \mathrm { p a r } }$ ratios to within 11%. This trend is consistent with the $O ( \rho )$ bound in Appendix A.1: larger failures of the unprotected policy leave more room for the canonicalizer to recover performance.

Table 3: Joint translation × rotation. on $\pi _ { 0 . 5 } \times$ libero\_spatial. Entries: NO CANON → GS-VLA (∆), in % SR. The bold $\Delta$ flags the largest gain $\left( + 4 2 . 2 \mathrm { p p } \right)$
<table><tr><td>Yaw θ</td><td> $\varepsilon = 1 0 0 { \bf c m }$ </td><td> $\varepsilon = 2 0 0 \mathbf { c m }$ </td></tr><tr><td> $\theta = 1 0 ^ { \circ }$ </td><td> $5 0 . 0 {  } 8 1 . 8 \ ( + 3 1 . 8 )$ </td><td> $3 5 . 8  7 8 . 0 \ ( + 4 2 . 2 )$ </td></tr><tr><td> $\theta = 2 0 ^ { \circ }$ </td><td> $4 5 . 5  6 6 . 5 ( + 2 1 . 0 )$ </td><td> $3 4 . 8  { \bf 6 9 . 3 } ( + 3 4 . 5 )$ </td></tr></table>

Table 4: Calibration stress. Pose noise $( T \ \mathrm { c m } , R ^ { \circ } )$ injected on top of ε=100 cm and applied only to the pose seen by the canonicalizer. The shaded zero-noise row is the GS-VLA reference; degradation is graceful up $\mathrm { { t o } \sim 3 \mathrm { { c m } / 3 ^ { \circ } } } .$ , with long-horizon libero\_10 the most sensitive at $5 \mathrm { c m } / 5 ^ { \circ }$
<table><tr><td>Calib. noise  $( T , R )$ </td><td>object spatial  $S R \left( \% \right)$   $S R \left( \% \right)$ </td><td>SR (%)</td><td>goal</td><td>libero_10 SR (%)</td></tr><tr><td>(0,0)</td><td>89.2</td><td>79.9</td><td>88.5</td><td>65.4</td></tr><tr><td>(1,1)</td><td>87.3</td><td>78.2</td><td>86.6</td><td>63.4</td></tr><tr><td>(2,2)</td><td>86.1</td><td>75.0</td><td>86.5</td><td>59.9</td></tr><tr><td>(3,3)</td><td>84.0</td><td>76.0</td><td>85.4</td><td>51.6</td></tr><tr><td>(5,5)</td><td>78.4</td><td>70.3</td><td>81.0</td><td>41.6</td></tr><tr><td> $\Delta _ { ( 5 , 5 ) }$ </td><td>-10.8</td><td>-9.6</td><td>-7.5</td><td>-23.8</td></tr></table>

## 4.4 Joint translation–rotation

So far, we mainly varied translation and kept rotation small. To check that the picture survives once translation and rotation move together, we cross $\varepsilon \in \{ 1 0 0 , 2 0 0 \}$ cm with yaw $\theta \in \{ 1 0 ^ { \circ } , 2 0 ^ { \circ } \}$ in Table 3. ∆ stays in the $[ + 2 1 , + 4 \bar { 2 } ]$ pp range across all four cells. Two trends are visible. First, larger ε actually widens $\Delta \colon$ the baseline drops faster than the canonicalizer, so the gap between them grows. Second, larger θ narrows $\Delta .$ , which we attribute to undersampling of large rotations at training time — a natural target for follow-up work, since it can be addressed entirely on the data side without changing the architecture.

## 4.5 Extrinsic-pose noise

The canonicalizer assumes that the deployment pose is known. In practice, it is only known up to calibration error, so we ask how much error is tolerated. To answer this, we inject $( T \mathrm { c m } , R ^ { \circ } )$ of pose noise on top of $\varepsilon { = } 1 0 0$ cm, applied only to the pose seen by the canonicalizer (Table 4). Within roughly 3 cm and $3 ^ { \circ }$ of calibration error, the canonicalizer remains stable, with a three-suite mean drop of at most 5 pp — comparable to ordinary camera re-mounting tolerances. Pushed to 5 cm and $5 ^ { \circ }$ , the long-horizon libero\_10 suite begins to suffer (−23.8 pp), which marks the practical limit of the calibration noise GS-VLA can absorb. Intrinsics noise is far less damaging: across the full focal-length sweep we report in Table 15, |∆| stays within 2.8 pp, so for the perturbation scales we care about, errors in the pinhole intrinsics matter much less than errors in the extrinsic pose.

## 5 Ablations

This section isolates which parts of the canonicalizer matter, and whether the picture changes when we swap out the depth source.

Capacity. A natural worry with a 4 M-parameter module is that more capacity would help. Doubling the base width to C=64 (15 M parameters) raises reconstruction PSNR from 26.84 to 28.4 dB but actually lowers SR to 85.2% (Table 13). Beyond about 4M-parameters, reconstruction quality and policy success decouple: the extra capacity goes into texture detail that the policy never reads, so it does not translate into higher SR. This supports our intentionally compact design.

Data. Training set size shows the same kind of saturation. Sweeping pair counts in {10k, 25k, 50k} yields 84–87% SR, with the signal already saturated by about 10k pairs (Table 14). This is consistent with the low intrinsic learning complexity that Locality predicts — the inpainting region is small, so it does not take many examples to cover it.

Depth substitution. GS-VLA depends on calibrated metric depth, taken from a sensor or precalibrated estimator in our main experiments. Replacing this with zero-shot DepthAnything V2 [36], even with per-image analytical scale/shift calibration, yields only 44.5% SR — the geometric base is no longer metric-aligned and per-frame calibration cannot recover the absolute scale required by (5). A short fine-tune of the depth backbone on 50 k LIBERO frames (3 epochs) bridges most of this gap, reaching 71.7% SR (−15.1 pp from the GT-depth ceiling at 86.8% and $+ 2 7 . 2 \mathrm { p p }$ over calibration alone, Table 16). Beyond this point, neither doubling the depth-encoder capacity (ViT-S→ViT-$\mathrm { B \to V i T \mathrm { - } L ) }$ , appending a 0.8M-parameter UNet residual refinement head, nor passing end-to-end gradients through the splat module produced further SR gains; the refinement head in particular collapsed to a zero residual, indicating that the fine-tuned depth is already $L _ { \mathrm { 1 } } { \mathrm { - } } \mathrm { o p t i m a l }$ at the resolution our pipeline consumes. We therefore read the remaining −15 pp gap as a structural distribution mismatch and the splat module’s information bottleneck, rather than a depth-accuracy limitation.

## 6 Limitation

We close with the boundaries within which our results should be read.

Single-view input. The canonicalizer in this paper consumes a single source frame and its calibrated depth, so any scene content that the source camera does not observe at all cannot be recovered by the in-painter. We expect a multi-view extension — for example, fusing a wrist-mounted camera or a previous-step frame $^ { - \mathrm { t o } }$ lift this ceiling without changing the underlying Locality reduction, and we leave that direction to future work.

Locality boundary on rotation. The translation budget in our default setting is already large $_ { ( \varepsilon _ { t } = 1 0 0 \mathrm { c m } }$ relative to $\mathrm { { a } \sim 0 . 7 }$ m workspace), and adding meaningful rotation on top of such translations frequently rotates the camera off the workspace entirely. We could not run a clean rotation sweep beyond the joint $( \varepsilon _ { t } , \theta )$ cells reported in Table 3 for this reason. Inspecting the failure-case rollouts, we found that a non-trivial fraction of the sampled cameras do not contain the workspace inside their frustum at all, which marks the practical ceiling of the perturbation distribution we used rather than a property of the canonicalizer itself.

Real-world evaluation. All experiments in this paper are run inside the LIBERO simulator. Validating GS-VLA on a real robot requires both an external metric-depth source (per the previous point) and the engineering effort to instrument a physical rig, neither of which we were able to put in place within the resource and personnel constraints of this project. We therefore report the simulator results as a strong but ultimately preliminary signal, and treat a real-robot replication as the natural follow-up.

## 7 Conclusion

We have presented GS-VLA, a 4M-parameter 3D-Gaussian canonicalizer that improves the cameraviewpoint robustness of frozen VLA policies without any policy updates. The design follows from a Locality reduction: under bounded perturbations, viewpoint canonicalization becomes a scene -and policy-independent disocclusion problem over an $O ( \rho )$ -area region ( A.1; the $\eta ( \rho )$ linearity link is confirmed at $R ^ { 2 } \ge 0 . 9 9$ , with the full $\varepsilon  \eta  \mathrm { r e s i d u a l }  \Delta$ chain closing at $R ^ { 2 } \in \dot { \{ 0 . 9 9 , 0 . 8 1 , 1 . 0 0 \} } )$ The practical consequence is that a single checkpoint trained on libero\_spatial at $\varepsilon _ { t } { = } 1 0 0$ cm transfers without modification across 4 policies, 4 LIBERO suites (with a 4-suite mean of $+ 3 9 . 8 \mathrm { p p } )$ and the full range $\varepsilon _ { t } \in [ 5 , 2 0 0 ]$ cm, while still tolerating roughly 3 cm and $3 ^ { \circ }$ of calibration noise. We see this as evidence for a broader reframing: “make a VLA camera-robust” can be turned from an axis-specific retraining problem into a visual-domain premise that a small observation-space module is enough to absorb. The same template should extend naturally to embodiment and lighting variation, once the corresponding “local” structure for those axes is identified.

## References

[1] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mipnerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5470–5479, 2022.

[2] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi\_0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[3] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

[4] Remi Cadene, Simon Aliberts, Francesco Capuano, Michel Aractingi, Adil Zouitine, Pepijn Kooijmans, Jade Choghari, Martino Russi, Caroline Pascal, Steven Palma, et al. Lerobot: An open-source library for end-to-end robot learning. arXiv preprint arXiv:2602.22818, 2026.

[5] Jun Cen, Siteng Huang, Yuqian Yuan, Kehan Li, Hangjie Yuan, Chaohui Yu, Yuming Jiang, Jiayan Guo, Xin Li, Hao Luo, et al. Rynnvla-002: A unified vision-language-action and world model. arXiv preprint arXiv:2511.17502, 2025.

[6] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In European conference on computer vision, pages 370–386. Springer, 2024.

[7] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal ofRobotics Research, 44(10-11):1684–1704, 2025.

[8] Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, et al. Libero-plus: In-depth robustness analysis of vision-languageaction models. arXiv preprint arXiv:2510.13626, 2025.

[9] Richard Hartley and Andrew Zisserman. Multiple view geometry in computer vision. Cambridge university press, 2003.

[10] Hyeongjun Heo, Seungyeon Woo, Sang Min Kim, Junho Kim, Junho Lee, Yonghyeon Lee, and Young Min Kim. Anycamvla: Zero-shot camera adaptation for viewpoint robust visionlanguage-action models. arXiv preprint arXiv:2603.05868, 2026.

[11] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. pi : a visionlanguage-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[12] Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang Xu. Lvsm: A large view synthesis model with minimal 3d inductive bias. arXiv preprint arXiv:2410.17242, 2024.

[13] Tsung-Wei Ke, Nikolaos Gkanatsios, and Katerina Fragkiadaki. 3d diffuser actor: Policy diffusion with 3d scene representations. arXiv preprint arXiv:2402.10885, 2024.

[14] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis, et al. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.

[15] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[16] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.

[17] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023.

[18] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024.

[19] Hugh Christopher Longuet-Higgins and Kvetoslav Prazdny. The interpretation of a moving retinal image. Proceedings of the Royal Society of London. Series B. Biological Sciences, 208 (1173):385–397, 1980.

[20] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffoldgs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20654–20664, 2024.

[21] Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3):7327–7334, 2022.

[22] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications ofthe ACM, 65(1):99–106, 2021.

[23] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4): 1–15, 2022.

[24] Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In Proceedings ofthe AAAI conference on artificial intelligence, volume 32, 2018.

[25] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830, 2025.

[26] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

[27] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015.

[28] Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan, Dmitry Lagun, Li Fei-Fei, Deqing Sun, et al. Zeronvs: Zero-shot 360-degree view synthesis from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9420–9429, 2024.

[29] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2149–2159, 2022.

[30] Stanislaw Szymanowicz, Chrisitian Rupprecht, and Andrea Vedaldi. Splatter image: Ultra-fast single-view 3d reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10208–10217, 2024.

[31] Stanislaw Szymanowicz, Eldar Insafutdinov, Chuanxia Zheng, Dylan Campbell, Joao F Henriques, Christian Rupprecht, and Andrea Vedaldi. Flash3d: Feed-forward generalisable 3d scene reconstruction from a single image. In 2025 International Conference on 3D Vision (3DV), pages 670–681. IEEE, 2025.

[32] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.

[33] Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P Srinivasan, Howard Zhou, Jonathan T Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas Funkhouser. Ibrnet: Learning multi-view image-based rendering. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 4690–4699, 2021.

[34] Junjie Wen, Yichen Zhu, Jinming Li, Zhibin Tang, Chaomin Shen, and Feifei Feng. Dexvla: Vision-language model with plug-in diffusion expert for general robot control. arXiv preprint arXiv:2502.05855, 2025.

[35] Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong Park, Ruiqi Gao, Daniel Watson, Pratul P Srinivasan, Dor Verbin, Jonathan T Barron, Ben Poole, et al. Reconfusion: 3d reconstruction with diffusion priors. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21551–21561, 2024.

[36] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Systems, 37:21875– 21911, 2024.

[37] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, et al. gsplat: An open-source library for gaussian splatting. Journal ofMachine Learning Research, 26(34):1–17, 2025.

[38] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4578–4587, 2021.

[39] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19447–19456, 2024.

[40] Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning. arXiv preprint arXiv:2407.08693, 2024.

[41] Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.

[42] Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, et al. X-vla: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. arXiv preprint arXiv:2510.10274, 2025.

[43] Allan Zhou, Moo Jin Kim, Lirui Wang, Pete Florence, and Chelsea Finn. Nerf in the palm of your hand: Corrective augmentation for robotics via novel-view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17907–17917, 2023.

[44] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.

## A Appendix

## A.1 Proposition 1 of locality reduction

Under the Locality assumption $C _ { s } \in B ( C ^ { \star } , \varepsilon _ { t } , \varepsilon _ { R } )$ of Section 4.1, the canonical view partitions into a region recoverable by the closed-form depth-warp (the world point $X _ { w } = R _ { s } K ^ { - 1 } D _ { s } ( u , v ) [ u , v , 1 ] ^ { \top } +$ $t _ { s }$ re-projected to the canonical pixel via Π $\left[ \phantom { } _ { K } ( R ^ { \star \top } ( X _ { w } - t ^ { \star } ) ) \right]$ , no learnable parameters) and a thin disocclusion band of fractional area $\eta \in [ 0 , 1 ]$ along depth edges, which has no defining input pixel and must be filled by a learned prior.

Proposition 1 (Locality Reduction). Assume piecewise-smooth depth $d \in [ d _ { \operatorname* { m i n } } , d _ { \operatorname* { m a x } } ]$ and silhouette edges that meet view rays transversely. Decompose $\rho = \rho _ { \mathrm { p a r } } + \rho _ { \mathrm { f r } }$ with $\rho _ { \mathrm { p a r } } = \| t _ { s } - t ^ { \star } \| / d _ { \operatorname* { m i n } }$ and $\rho _ { \mathrm { f r } } = \| \log ( R ^ { \star ^ { \top } } R _ { s } ) \|$ . Thenfor every $\rho \leq 1$

$$
c _ { \mathrm { l o w } } \rho \le \eta ( C _ { s } , C ^ { \star } ) \le \alpha _ { \mathrm { p a r } } \rho _ { \mathrm { p a r } } + \alpha _ { \mathrm { f r } } \rho _ { \mathrm { f r } } + C _ { 2 } \rho ^ { 2 } ,
$$

where

$$
\alpha _ { \mathrm { p a r } } = \frac { L _ { \mathrm { p x } } ^ { \star } } { \pi A _ { \mathrm { p x } } } \cdot \frac { d _ { \mathrm { m a x } } - d _ { \mathrm { m i n } } } { d _ { \mathrm { m a x } } } , \qquad \alpha _ { \mathrm { f r } } = \frac { P _ { \mathrm { p x } } } { A _ { \mathrm { p x } } } ,
$$

$L _ { \mathrm { p x } } ^ { \star }$ is the canonical-view silhouette arc-length, $P _ { \mathrm { p x } }$ is the image-frame perimeter, $A _ { \mathrm { p x } } = H W$ the image area, and $c _ { \mathrm { l o w } } , C _ { 2 }$ depend only on scene-regularity bounds. The two-term form makes explicit that the parallax contribution scales with translation and depth contrast, while theframe contribution scales with rotation and image perimeter.

Notation. We denote the parallax and frame contributions to η as $\eta _ { \mathrm { p a r } }$ and $\eta _ { \mathrm { f r } }$ respectively, with $\eta = \eta _ { \mathrm { p a r } } + \eta _ { \mathrm { f r } }$ . Pure translation has $\eta _ { \mathrm { f r } } \neq 0$ as well (translation also induces some image-plane motion); pure rotation has $\eta _ { \mathrm { p a r } } = 0$ since rotation introduces no depth-parallax.

The upper bound integrates a parallax shadow band along each silhouette; a matching lower bound via Cauchy–Schwarz on the silhouette-normal field certifies that the linear rate cannot be improved. Empirical validation in Appendix A.2 confirms the linear regime to $\rho \approx 0 . 5$ with $R ^ { 2 } \ge 0 . 9 9$

Corollary 1 (Same-domain transfer). Since $\alpha _ { \mathrm { p a r } }$ and $\alpha _ { \mathrm { f r } }$ depend only on silhouette–depth and imageframe geometry, not on texture, lighting, or policy, a canonicalizer trained on one suite transfers to others in the same visual domain.

Verified in Section 4.2 and 1: the same checkpoint improves every policy and every off-training suite.

Success-rate coupling. A residual-aware Lipschitz argument composes the area bound into a success-rate drop $\Delta ( \bar { \rho } ) = L _ { \pi } \cdot c _ { \mathrm { r e c o n } } \cdot ( \alpha _ { \mathrm { p a r } } \bar { \rho } _ { \mathrm { p a r } } + \alpha _ { \mathrm { f r } } \bar { \rho } _ { \mathrm { f r } } ) + O ( \rho ^ { 2 } )$ , where $L _ { \pi }$ is the policy’s local Lipschitz constant on canonical inputs and $c _ { \mathrm { r e c o n } }$ the residual–η slope of $f _ { \phi }$ . The full chain $\varepsilon  \eta  \mathrm { r e s i d u a l }  \Delta$ closes numerically in Appendix A.2 (Table 6).

## A.2 Empirical validation of proposition 1

We test whether the Locality picture really holds at the level of individual quantities, not only at the level of $\Delta .$ . Proposition A.1 is tested directly under controlled synthetic perturbations: 100 canonical frames combined with 10 perturbations each, using ground-truth depth and no policy in the loop.

Linearity. The first prediction is that η should grow linearly in $\rho$ inside the Locality ball. We find $\eta ( \rho )$ is linear with $R ^ { 2 } \ge 0 . 9 9$ for $\rho \leq 0 . 5$ across translation, rotation, and joint perturbation modes, and across both spatial and object. On every (mode, suite) cell the slope satisfies $\eta ( \rho ) / \rho \in [ 0 . 2 7 , 1 . 0 9 ]$ , which confirms the two-sided bound of Proposition ${ \mathrm { A . 1 } }$ . Departure from linearity begins around $\rho \approx 0 . 5$ , well inside the validity range the proposition predicts.

Scene decomposition. A second, sharper prediction comes from Corollary A.1: the rotation contribution to $\eta$ depends only on image-plane perimeter and should therefore be scene-invariant, while the translation contribution depends on silhouette length $L ^ { \star }$ and should not be. The data follow this split closely (Table 5). The rotation slope is essentially the same on both suites (1.084 on spatial vs. 1.089 on object, a 0.4% discrepancy), while the translation slope varies by about 60%. Predicting the per-suite translation-slope ratio from $L ^ { \star }$ · κ matches the observed value to within 11%.

Silhouette hugging. The geometry implied by Proposition A.1 also predicts where the disocclusion pixels live: along depth silhouettes, not spread uniformly across the image. We see exactly that. Occluded pixels lie $5 { - } 7$ px from the nearest depth edge on both suites, against 27–33 px for uniformrandom pixels — a roughly 5× tighter localization, in line with the geometry the proposition predicts.

End-to-end chain. Finally, we verify that the bound on η propagates all the way to the success-rate drop $\Delta .$ . On 200 pairs the residual–η link is well captured by a single line (Pearson $r = 0 . 9 0 , R ^ { 2 } =$ $0 . 8 1 , c _ { \mathrm { r e c o n } } = 0 . 0 4 9 )$ . Composing the full chain $\varepsilon  \eta  \mathrm { r e s i d u a l }  \Delta$ implies a slope of $L \approx 2 1 3 \mathrm { p p }$ per $L _ { 1 }$ -residual unit. To check that this is physically plausible we estimate the policy’s local Lipschitz behaviour from the released $\pi _ { 0 . 5 }$ weights using power iteration: log $\prod _ { l } ( 1 + L _ { l } ) = 3 8 9$ across 421 weight matrices, comfortably above the value implied by $L .$ . All four links close numerically with $R ^ { 2 } \stackrel { \smile } { \in } \{ 0 . 9 9 , 0 . 8 1 , 1 . 0 0 \}$ (Table 6). Taken together, these checks confirm that the Locality picture is not just consistent with the headline numbers but actually predicts the intermediate quantities they decompose into.

## A.2.1 Locality slopes by mode × suite

Direct test of Proposition A.1 and Corollary A.1. We fit $\eta ( \rho )$ separately for translation, rotation, and joint perturbations on spatial and object; numerical discussion is in Section A.2.

Table 5: Empirical slopes of $\eta ( \rho )$ . Per-mode slopes for parallax $( \eta _ { \mathrm { p a r } } )$ , frame $( \eta _ { \mathrm { f r } } )$ , and total $( \eta _ { \mathrm { t o t } } )$ . Silhouette-hugging: occluded pixels average 5.05/6.78 px from depth edges (spatial/object) vs. 26.77/33.10 px for uniform-random pixels (5× tighter). Object suite uses translation- and rotationonly fits; joint mode pending.
<table><tr><td>Mode</td><td> $\eta _ { \mathrm { p a r } }$ </td><td> $\eta _ { \mathrm { f r } }$ </td><td> $\eta _ { \mathrm { t o t } }$ </td><td> $R ^ { 2 }$ </td></tr><tr><td colspan="3">libero_spatial  $( \rho \le 0 . 5 )$ </td><td></td><td></td></tr><tr><td rowspan="3">Translation Rotation Joint</td><td>0.082</td><td>0.281</td><td>0.363</td><td>0.999</td></tr><tr><td>&lt;0.001</td><td>1.084</td><td>1.084</td><td>0.995</td></tr><tr><td>0.038</td><td>0.607</td><td>0.645</td><td>0.994</td></tr><tr><td colspan="3">libero_object  $( \rho \le 0 . 5 )$  0.050</td><td></td><td></td></tr><tr><td colspan="3">Translation</td><td></td><td>一</td></tr><tr><td colspan="3">Rotation</td><td>1.089</td><td>一</td></tr></table>

## A.2.2 Success-rate coupling chain

Quantitative check of the full $\varepsilon  \eta  \mathrm { r e s i d u a l }  \Delta$ chain proposed in Appendix A.1. The geometric $( R ^ { 2 } { = } 0 . 9 9 )$ and policy-Lipschitz $\dot { ( R ^ { 2 } } { = } 1 . 0 0 )$ links are tight; the $\eta $ residual link is the loosest at $R ^ { 2 } { = } 0 . { \dot { 8 } } 1$ , but the composite end-to-end fit still closes at $R ^ { 2 } { = } 0 . 9 9 6$ within the linear regime $( \varepsilon \le 5 0 \mathrm { c m } )$

Table 6: Success-rate coupling chain (Appendix ${ \bf A . 1 } )$ . Each individual link and the end-to-end drop fit linearly with high $R ^ { 2 }$ . The implied $L \approx 2 1 3 \mathrm { p p }$ per $L _ { 1 }$ -residual unit lies inside the spectral-norm bound for $\pi _ { 0 . 5 } ( \log \prod _ { l } ( 1 + L _ { l } ) = 3 8 9 )$
<table><tr><td>Link</td><td>Quantity</td><td>Slope</td><td> $R ^ { 2 }$ </td></tr><tr><td> $\varepsilon \to \eta$ </td><td>joint  $\eta / \rho$  slope (Tab. 5)</td><td> ${ \approx } 0 . 6 4 5$ </td><td>0.99</td></tr><tr><td> $\eta  \mathrm { r e s i d u a l }$ </td><td>predictor linearity</td><td> $c _ { \mathrm { r e c o n } } = 0 . 0 4 9$ </td><td>0.81</td></tr><tr><td> $\mathrm { r e s i d u a l }  \Delta$ </td><td>policy Lipschitz</td><td> $L \approx 2 1 3 { \mathrm { p p } } / { \mathrm { L } } _ { 1 }$ </td><td>1.00</td></tr><tr><td> $\varepsilon  \Delta ( \mathrm { c o m p o s i t e } )$ </td><td>end-to-end  $( \varepsilon \le 5 0 )$ </td><td> $6 . 6 7 \mathrm { p p } / \rho$ </td><td>0.996</td></tr></table>

## A.3 Environments

## A.3.1 Implementation detail

We summarize the architecture, training configuration, and runtime cost in Tables 7–9, and describe data composition and the software stack.

Table 7: Canonicalizer architecture. With the listed initialization, $f _ { \phi }$ starts at the closed-form forward warp at iteration 0 and learns only a residual.  
Component Specification   
Backbone Symmetric U-Net, base width $C { = } 3 2 ,$ , 4 down-/up-sampling levels   
Conv block FiLM-modulated ConvBlock at every scale   
Pose embedding $\boldsymbol { z } _ { \mathrm { p o s e } } = \mathrm { M L P } ( [ t _ { s } , q _ { s } , t ^ { \star } , \boldsymbol { q } ^ { \star } ] ) \in \mathbb { R } ^ { 1 2 \bar { 8 } }$ , 2-layer   
Input $[ \dot { I } _ { s } \parallel D _ { s } ] \in \mathbb { R } ^ { \tilde { 4 } \times H \times W } , H { = } \tilde { W } { = } 2 5 6$   
Per-pixel output 14-dim Gaussian descriptor $( \Delta \mu , \log s , q , \alpha , \Delta c )$   
Centre clamp $\mu = X _ { w } + 0 . 1 D _ { s }$ tanh $( \Delta \mu ) ( \leq 1 0 \%$ of local depth)   
Initialization residual heads zero-init; opacity bias set so that α=1 at step 0   
Parameters 4,047,824 (≈ 4M); encoder ≈ 60%, decoder ≈ 40%

Table 8: Training configuration. Only the canonicalizer parameters are updated; the policy is never touched. No discriminator or perceptual GAN loss is used.  
Hyperparameter Value   
Optimizer AdamW $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 ,$ , weight decay $1 0 ^ { - 4 } )$   
Base learning rate $5 \times 1 0 ^ { - 5 }$   
LR schedule 500-step linear warmup (start factor 0.01), then cosine to 0   
Batch size 16   
Epochs 50   
Gradient clipping global norm 1.0   
$L _ { 1 } : S S \mathbf { I M }$ weight $0 . 8 : 0 . 2$   
Scale cap weight $\lambda _ { s }$ $1 0 ^ { - 3 }$ on ReLU(log s − 2)   
Position reg. $\lambda _ { \mu }$ $1 0 ^ { - 2 }$ on $| \Delta \mu | / D _ { s }$   
Background augmentation $b \sim \mathcal { U } ( [ 0 , 1 ] ^ { 3 } ) ,$ , resampled per step   
Image augmentation horizontal flip only   
Random seed 42; ±0.6 pp variation across 3 independent seeds   
Hardware 1× RTX A5000 (24 GB), peak memory ≈ 18 GB   
Wall-clock $\approx 5 \mathrm { { h } \left( 5 0 \times 5 0 0 \mathrm { { s } } \right) }$   
Best val. PSNR 26.84 dB

Table 9: Inference latency. The downstream policy keeps its native action-chunk schedule (e.g., 50 for $\pi _ { 0 . 5 } )$ , so the canonicalizer cost is amortized across many actions per frame. Softmax splatting is not used in the released checkpoint.
<table><tr><td>Stage</td><td>Per-frame cost (A5000)</td></tr><tr><td>U-Net forward</td><td>≈ 5 ms</td></tr><tr><td>Differentiable α-splat (gsplat)</td><td>≈ 8 ms</td></tr><tr><td>Total canonicalizer  $f _ { \phi }$ </td><td>≈ 13 ms (≈ 75 Hz)</td></tr><tr><td>Rasterizer config</td><td> $4 5 ^ { \circ }$  vertical FOV, bilinear splatting</td></tr></table>

## A.3.2 Data setup

We train on 50,000 source→canonical pairs sampled from libero\_spatial (10 tasks, multiple seeds per task), with 49,800 used for training and 200 held out for validation. Each pair is generated by rolling out the simulator at the canonical camera, then re-rendering the same scene under a perturbed camera $C _ { s } \sim { \mathcal { B } } ( C ^ { \star } , \varepsilon _ { t } { = } 1 0 0 \mathrm { c m } , \varepsilon _ { R } )$ with $\varepsilon _ { R } \approx \varepsilon _ { t } / d _ { \operatorname* { m i n } }$ . We deliberately did not use colour jitter, multi-suite mixing, or per-task balancing, so the cross-suite results in Section 4.2 are zero-shot.

## A.3.3 Software framework

PyTorch 2.4, CUDA 12.1, gsplat [37] 0.1.x, robosuite 1.4 for LIBERO simulation, and lerobot [4] for the policy interface. Random seeds, full hyperparameter dumps, and the training command lines used to produce every released checkpoint are bundled with the released code.

## A.4 Additional experimental results and ablation studies

## A.4.1 Full cross-policy × suite matrix

This appendix expands Table 1(a) into the full policy × suite matrix at $\scriptstyle \varepsilon _ { t } = 1 0 0 \mathrm { c m }$ . Every cell is positive, and the largest gains track baselines closest to total failure — consistent with Locality predicting larger headroom where more pixels go out-of-distribution.

Table 10: Full cross-policy × suite matrix with $\varepsilon _ { t } = 1 0 0 \mathrm { c m }$ . Bold $\Delta$ values denote the absolute increase in success rate (percentage points) for key configurations, such as XVLA (+79.6 pp), π<sub>0.5</sub> on libero\_10 (+62.8 pp), and OpenVLA-OFT on object-centric tasks (+61.8 pp).
<table><tr><td>Policy</td><td>Suite</td><td>No canon</td><td>GS-VLA (ours)</td><td>∆ (pp)</td></tr><tr><td rowspan="3"> $\pi _ { 0 . 5 } \left( 3 . 4 \mathbf { B } \right)$ </td><td>spatial</td><td>42.6</td><td>86.8</td><td>+44.2</td></tr><tr><td>object</td><td>64.0</td><td>90.2</td><td>+26.2</td></tr><tr><td>goal</td><td>66.3</td><td>92.2</td><td>+25.9</td></tr><tr><td rowspan="4">OpenVLA-OFT (7 B)</td><td>libero_10</td><td>9.3</td><td>72.1</td><td>+62.8</td></tr><tr><td>spatial</td><td>87.2</td><td>92.0</td><td>+4.8</td></tr><tr><td>object</td><td>19.8</td><td>81.6</td><td>+61.8</td></tr><tr><td>goal libero_10</td><td>80.4</td><td>91.0</td><td>+10.6 +38.2</td></tr><tr><td rowspan="3">RynnVLA-002</td><td></td><td>8.8</td><td>47.0</td><td></td></tr><tr><td>spatial goal</td><td>25.8 29.0</td><td>78.6 75.8</td><td>+52.8</td></tr><tr><td>libero_10</td><td>53.4</td><td>65.2</td><td>+46.8 +11.8</td></tr><tr><td rowspan="2">XVLA</td><td>spatial</td><td>1.4</td><td>81.0</td><td></td></tr><tr><td>goal</td><td>5.4</td><td>80.0</td><td>+79.6 +74.6</td></tr></table>

## A.4.2 GS-VLA absolute SR across suites (ε-sweep)

Companion to Table 2: GS-VLA absolute SR for all four suites as $\varepsilon _ { t }$ varies. object and goal stay within ∼ 2 pp of their small-ε ceilings up to $\scriptstyle \varepsilon _ { t } = 1 0 0 \mathrm { c m } ;$ the noticeable drop at 200 cm marks the per-suite Locality boundary.

Table 11: GS-VLA absolute SR (%) across suites as $\varepsilon _ { t }$ varies. object and goal stay within $\sim 2$ pp of their $\varepsilon _ { t } = 5$ ceiling up to $\scriptstyle \varepsilon _ { t } = 1 0 0 \mathrm { c m } ;$ the main drop appears at $\scriptstyle \varepsilon _ { t } = 2 0 0 \mathrm { c m } -$ the per-suite Locality boundary.
<table><tr><td>Suite (GS-VLA)</td><td> $\varepsilon _ { t } = 5$ </td><td>15</td><td>30</td><td>50</td><td>100</td><td>200</td></tr><tr><td>spatial</td><td>88.0</td><td>88.6</td><td>87.6</td><td>87.2</td><td>86.8</td><td>78.5</td></tr><tr><td>object</td><td>92.2</td><td>92.2</td><td>91.4</td><td>92.7</td><td>90.2</td><td>84.7</td></tr><tr><td>goal</td><td>93.9</td><td>93.6</td><td>92.4</td><td>93.2</td><td>92.0</td><td>85.7</td></tr><tr><td>libero_10</td><td>71.9</td><td>72.7</td><td>73.4</td><td>70.8</td><td>72.1</td><td>52.9</td></tr></table>

## A.4.3 Magnitude sweep

Stress test pushing position σ and look-at σ jointly to 2× and 3× the default. $\Delta$ stays large and grows monotonically — NO CANON decays faster than the canonicalizer, so the gap widens with magnitude until the Locality boundary.

Table 12: Magnitude sweep on $\pi _ { 0 . 5 } { \times } 1 \mathrm { i } \mathsf { b } \mathsf { e r o } .$ \_spatial. While the performance drop $( \Delta )$ increases with perturbation magnitude, the canonicalizer exhibits significantly slower decay compared to the NO CANON baseline. This gap (utility) widens progressively until the Locality boundary. <sup>†</sup>The $\varepsilon _ { t } = 1 5$ row follows the single-scalar protocol from Table 2. <sup>‡</sup>Labels $2 \times / 3 \times$ are nominal; position σ is capped to remain within the workspace, and lookat σ is fixed at 1×.
<table><tr><td>Condition</td><td>pos σ (m)</td><td>lookat σ</td><td>No canon</td><td>GS-VLA (ours)</td><td> $\Delta \left( \mathbf { p } \mathbf { p } \right)$ </td></tr><tr><td> $\varepsilon _ { t } { = } 1 5 \left( 1 0 \mathrm { w } \right) ^ { \dagger }$ </td><td>n/a</td><td>n/a</td><td>82.7</td><td>88.6</td><td> $+ 5 . 9$ </td></tr><tr><td>1× (default,  $\varepsilon _ { t } \mathrm { = } 1 0 0 )$ </td><td>(0.3, 0.5, 0.25)</td><td>0.06</td><td>42.6</td><td>86.8</td><td>+44.2</td></tr><tr><td> $2 \times ( \mathrm { b i g r a n d } ) ^ { \ddag }$ </td><td>(0.6, 0.8, 0.5)</td><td>0.06</td><td>41.6</td><td>80.8</td><td> $+ 3 9 . 2$ </td></tr><tr><td> $\mathrm { 3 \times ( e x t r e m e ) ^ { \ddag } }$ </td><td>(0.9, 1.2, 0.75)</td><td>0.06</td><td>34.4</td><td>76.1</td><td> $+ 4 1 . 7$ </td></tr></table>

## A.4.4 Capacity ablation

We sweep base width $C \in \{ 1 6 , 2 4 , 3 2 , 6 4 \}$ to test whether 4 M parameters are simply too few. Reconstruction PSNR keeps improving with C, but SR peaks at $C { = } 3 2$ and drops slightly at $C { = } 6 4$ — the policy does not consume the extra texture detail, supporting the intentionally compact design.

Table 13: Capacity ablation. Reconstruction PSNR keeps improving with C, but SR peaks at $C { = } 3 2$ Beyond $\sim 4 \mathbf { M }$ parameters, extra capacity goes to texture detail the policy does not consume.
<table><tr><td>Base C</td><td>Params</td><td>PSNR (dB)</td><td>SR (%)</td></tr><tr><td>16</td><td>~1M</td><td>25.77</td><td>一</td></tr><tr><td>24</td><td>~2M</td><td>26.33</td><td>一</td></tr><tr><td>32 (ours)</td><td>4M</td><td>26.84</td><td>86.8</td></tr><tr><td>64</td><td>15M</td><td>28.40</td><td>85.2</td></tr></table>

## A.4.5 Data-size ablation

Training-set sweep over $N \in \{ 1 0 \mathbf { k } , 2 5 \mathbf { k } , 5 0 \mathbf { k } \}$ pairs. The signal saturates by $\sim 2 5 \mathrm { k }$ pairs, with 50 k offering only ${ \sim } 2 \mathsf { p p }$ headroom over 10 k — consistent with the small intrinsic learning complexity that Locality predicts (the inpainting region is narrow).

Table 14: Data-size ablation $( C { = } 3 2 , \varepsilon { = } 1 0 0 \mathrm { c m } )$ . The signal saturates near 25k pairs, consistent with the low intrinsic learning complexity that Locality predicts.
<table><tr><td>Pairs N</td><td>PSNR (dB)</td><td>SR (%)</td></tr><tr><td>10k</td><td>26.06</td><td>~84</td></tr><tr><td>25k</td><td>26.38</td><td>~85</td></tr><tr><td>50k (ours)</td><td>26.84</td><td>86.8</td></tr></table>

## A.4.6 Intrinsics robustness $( \mathbf { f o v } _ { y }$ sweep)

Companion to the extrinsic-noise stress in Table 4: we instead inject focal-length error of $\pm 5 \%$ and $\pm 1 0 \%$ on top of $\varepsilon { = } 1 0 0 \mathrm { c m }$ . Worst-case $| \Delta | = 2 . 8$ pp — about 4× smaller than the extrinsicpose budget, indicating that intrinsics calibration is much less critical for deployment than extrinsic calibration.

Table 15: Intrinsics stress on libero\_object, 1000 ep, atop ε=100 cm extrinsic. Tolerance is ∼4× tighter than for extrinsic noise in Table 4.
<table><tr><td> $\mathbf { f o v } _ { y }$ </td><td>SR (%)</td><td>∆ vs. ref</td></tr><tr><td>-10% (zoom-in)</td><td>86.4</td><td>-2.8</td></tr><tr><td>-5%</td><td>87.9</td><td>-1.3</td></tr><tr><td>ref (0%) (ours)</td><td>89.2</td><td></td></tr><tr><td>+5%</td><td>89.1</td><td>-0.1</td></tr><tr><td>+10% (zoom-out)</td><td>89.0</td><td>-0.2</td></tr></table>

## A.4.7 Depth-substitution ablation

Companion to the Depth substitution paragraph in Section 5. The question is whether GT depth can be replaced by a learned monocular estimator without retraining the canonicalizer. We sweep the depth pipeline along three axes at fixed canonicalizer $( C { = } 3 2 , \varepsilon { = } 1 0 0 \mathrm { c m } )$ on libero\_spatial: (i) calibration strategy applied to off-the-shelf DepthAnything V2, (ii) fine-tuning capacity (ViT-S/B/L on 50 k LIBERO frames), and (iii) pipeline modifications atop the best fine-tuned backbone (residual refinement head, end-to-end depth gradient, confidence gating). The sweep is designed to isolate where the gap to the 86.8% GT-depth ceiling closes and where it stops moving — the first axis tests whether calibration alone is enough, the second whether a larger depth model is enough, and the third whether downstream-aware refinement is enough.

Table 16: Depth-substitution ablation (libero\_spatial, ε = 100 cm, 1000 ep unless marked). Calibration alone caps at 44.5%; a fine-tune reaches 71.7%. Larger encoders, a refinement head, and end-to-end gradients all fail to improve further — the residual −15 pp gap is not a depth-accuracy bottleneck.
<table><tr><td>Depth source</td><td>Encoder</td><td>val  $L _ { 1 }$ </td><td>SR (%)</td></tr><tr><td>GT depth (upper bound) (ours)</td><td></td><td></td><td>86.8</td></tr><tr><td colspan="4">Calibration only (no fine-tune)</td></tr><tr><td>DA V2, raw inverse</td><td>ViT-S</td><td></td><td>40.9</td></tr><tr><td>DA V2, metric indoor</td><td>ViT-S</td><td></td><td>44.5</td></tr><tr><td colspan="4">Fine-tuned on 50 k LIBERO pairs (3 ep, L1 + edge-aware)</td></tr><tr><td>DA V2-S, FT</td><td>ViT-S</td><td>0.0092</td><td>68.6</td></tr><tr><td>DA V2-B, FT (ours)</td><td>ViT-B</td><td>0.0083</td><td>71.7</td></tr><tr><td>DA V2-L, FT</td><td>ViT-L</td><td>0.0073</td><td>67.3</td></tr><tr><td colspan="4">Pipeline modifications atop DA V2-B FT</td></tr><tr><td>+ UNet residual refinement (0.8 M)</td><td>ViT-B</td><td>0.0083</td><td>zero residual</td></tr><tr><td>+ end-to-end depth gradient  $( w _ { \mathrm { a n c h o r } } = 0 . 5 )$ </td><td>ViT-B</td><td>0.0376</td><td>66.1</td></tr></table>