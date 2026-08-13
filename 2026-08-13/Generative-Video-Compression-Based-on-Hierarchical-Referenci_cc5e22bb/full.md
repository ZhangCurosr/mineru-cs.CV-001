# Generative Video Compression Based on Hierarchical Referencing

Daowen Li <sup>†</sup>, Ding Ding <sup>†</sup>, Zifu Zhang , Student Member, IEEE, Kai Li, and Ying Chen , Senior Member,

IEEE

Abstract—Diffusion-based generative video compression has emerged as a promising paradigm to improve perceptual quality, where latent frames are required to be encoded efficiently while serving as denoising conditions. However, existing methods neither carefully design reference and quality structures during latent coding nor account for the impact of frame-level quality variation on denoising procedure, which limits coding efficiency and aggravates artifact propagation during generative reconstruction. In this paper, we propose GVCHR, Generative Video Compression based on Hierarchical Referencing. The key idea is to organize latent frames hierarchically, where the selected high-quality references benefit both latent coding and generative reconstruction. In latent coding, GVCHR couples a hierarchical reference structure with a hierarchical quality structure, assigning more bits to lower-layer frames that are reused more frequently as references. Built on this design, we introduce Hierarchical Temporal Context Mining to exploits complementary short- and long-term temporal context for effective latent coding. In generative reconstruction, the coding-side hierarchy is incorporated into a Hierarchical Attentive Adapter which is attached to a video diffusion transformer. This adapter uses hierarchical attention to restrict each latent frame to attend only to the same- or lower-layer references, thereby reducing artifact propagation during denoising. Experiments validate GVCHR on multiple benchmarks. Compared with the previous state-of-theart method, GVCHR achieves 50.5% and 54.0% BD-rate gains in terms of LPIPS and DISTS, respectively, while also delivering clearly improved visual quality.

Index Terms—Generative video compression, Diffusion model, Hierarchical reference structure, Perceptual quality

## I. INTRODUCTION

the importance of efficient video compression methods for saving storage and transmission costs. Existing traditional codecs [1], [2], [3] and most neural video compression (NVC) methods [4], [5], [6], [7], [8] are primarily optimized for fidelity-oriented objectives, such as PSNR or mean squared error, which often lead to blurry structures and visually unsatisfactory artifacts at low bitrates. This mismatch between distortion minimization and human perception has motivated increasing interest in generative video compression (GVC), where generative priors are exploited to reconstruct more realistic and perceptually pleasing videos. Recent advances in diffusion models [9], [10], [11] have further strengthened this direction, owing to their superior ability to synthesize photorealistic spatial and temporal details. As a result, diffusionbased GVC has emerged as a promising paradigm [12], [13], in which compressed latent representations are transmitted and then used to guide generative reconstruction at the decoder.

A typical diffusion-based GVC pipeline first encodes a source video into latent representations using a variational autoencoder (VAE) [14], and then compresses them with a learned latent codec. During reconstruction, the decoded latent frames are fed to a diffusion model as conditions for denoising, and the denoised latent frames are finally mapped back to the pixel domain by the VAE decoder. In latent coding, the clean latent frames should be compressed efficiently, which depends on how previously decoded frames are organized and exploited as temporal context, as well as how coding quality is allocated across frames. In particular, frames that play more important reference roles in providing context should be preserved with higher quality. In generative reconstruction, the decoded latent frames become lossy conditions for denoising, where the key issue is to organize their reference dependencies so as to reduce the propagation of coding artifacts across frames. Therefore, how to organize latent frames for both efficient coding and effective generative reconstruction becomes a central issue in diffusion-based GVC.

However, existing diffusion-based GVC methods do not organize latent frames in a way that fully serves both stages [13], [12]. In latent coding, they often rely on simple frame-byframe referencing, or introduce frame-level quality variation without explicitly coupling it to frame’s reference role. As a result, reference frames that are important to enrich temporal context, might not be preserved with higher quality. In generative reconstruction, decoded latent frames are typically injected into image or video diffusion models through precedingframe conditioning or full attention across frames. Due to these lossy conditioning signals, unrestricted cross-frame interaction can spread artifacts from less reliable frames and further amplify them during denoising. These limitations suggest that hierarchical organization of latent frames may benefit diffusionbased GVC. Hierarchical reference structures and hierarchical quality structures have been extensively studied in traditional video coding [15], [16] and fidelity-oriented NVC [17], [7], where they improve temporal prediction, reduce error propagation, and enhance rate-distortion (RD) performance. This motivates hierarchical design in diffusion-based GVC, so that important references can be better preserved during coding and more reliably reused during generative reconstruction.

To address these limitations, we propose a Generative Video Compression framework based on Hierarchical Referencing, named GVCHR. The key idea is to organize latent frames according to their reference importance, so that a small set of more reliable frames can better support both latent coding and generative reconstruction. To this end, GVCHR couples a hierarchical reference structure with a hierarchical quality structure in latent coding, and carries this hierarchical structure over to generative reconstruction through hierarchy-aware conditioning.

![](images/c7debd2ba6b33d613cec7d8bdc88887067dce4225799286853608f5e29805c61.jpg)  
Fig. 1. Visual comparisons with open-source baselines on the videoSRC24 sequence from MCL-JCV.

In latent coding, GVCHR organizes latent frames into multiple layers. Lower-layer latent frames are allocated more bits and reused more often as references, whereas higher-layer latent frames are coded more compactly and cannot serve as references for lower-layer ones. Built on this design, we introduce Hierarchical Temporal Context Mining (HTCM) to exploit complementary cues from short- and long-term references. The short-term reference preserves local details from the nearest decoded latent frame at the same or lower layers, while the long-term reference summarizes global temporal semantics from previously decoded lowest-layer latent frames. HTCM then aligns and fuses these two forms of context to improve conditional coding under hierarchical referencing.

In generative reconstruction, GVCHR injects decoded latent frames into a video diffusion transformer (DiT) through a Hierarchical Attentive Adapter (HA-Adapter) to guide denoising [10]. Instead of using full attention across all frames, the HA-Adapter preserves the coding-side hierarchy through hierarchical attention, so that each latent frame attends only to the same- or lower-layer references. This prevents less reliable higher-layer latent frames, coded with fewer bits, from contaminating more important lower-layer references during denoising, thereby reducing artifact propagation in generative reconstruction.

To the best of our knowledge, this work is the first unified framework in diffusion-based GVC that extends hierarchical design from latent coding to generative reconstruction. This design offers two advantages: 1) it allocates more bits to latent frames with more important reference roles during coding, thereby reducing the overall bitrate; and 2) it enables the diffusion model to rely on these higher-quality references to reconstruct the remaining frames more effectively, thus improving perceptual quality under aggressive compression. Experiments demonstrate that GVCHR achieves state-of-theart compression performance across multiple metrics. Compared with the latest GVC method [13], GVCHR achieves 50.5% and 54.0% BD-rate reductions in terms of LPIPS and DISTS, respectively, while also producing better visual results, as shown in Fig. 1.

Our contributions are summarized as follows:

• We propose GVCHR, a generative video compression framework based on hierarchical referencing, which is the first to explore hierarchical reference structures for both latent coding and generative reconstruction.

• We introduce hierarchical temporal context mining for latent coding, which exploits complementary short- and long-term temporal context from high-quality lower-layer references.

• We design a hierarchical attentive adapter to inject decoded latent frames into a video DiT through hierarchical attention, thereby reducing coding artifact propagation during denoising.

• GVCHR achieves state-of-the-art compression performance in diffusion-based GVC, achieving 50.5% and 54.0% BD-rate reductions in terms of LPIPS and DISTS, respectively, compared with the latest method [13].

The rest of this paper is organized as follows. Section II reviews GVC and hierarchical reference structures. Section III presents GVCHR. Experimental results are given in Section IV, and Section V concludes the paper.

## II. RELATED WORK

## A. Generative Video Compression

Generative video compression (GVC) exploits generative models to improve reconstruction and restore realistic details, thereby improving perceptual quality beyond the pixel fidelity targeted by traditional video coding and fidelity-oriented NVC. Early studies [18], [19], [20], [21] mainly incorporated generative adversarial networks (GANs) and perceptual losses into existing neural codecs. For example, PLVC [20] adopted a recurrent autoencoder together with a recurrent discriminator under GAN training. However, such methods often suffer from training instability and visually unpleasant artifacts caused by mode collapse.

![](images/03ef36969d6a7b3ac7f6da56dda4ca125c4dbc8bce240d27acba7229ec97b534.jpg)  
Fig. 2. Overview of the proposed GVCHR framework for generative video compression.

Subsequent studies explored latent-space generation for video reconstruction. GLC-Video [22] performed transform coding in the latent space of a vector-quantized variational autoencoder (VQ-VAE) [23], but its reconstruction quality was limited by the representational capacity of VQ-VAE. More recently, diffusion models have become popular due to their strong ability to synthesize realistic spatial and temporal details. I<sup>2</sup>VC [24] utilized frame-level latent diffusion to align reference features and generate target frames by denoising motion-related regions. DiffVC [12] compressed videos in the latent domain and then used an image diffusion model to enhance decoded latent frames. Other approaches [25], [26], [27] improved compression efficiency by transmitting keyframes and limited side information, while using a video diffusion model to generate the remaining frames at the decoder.

Despite their promising perceptual quality, these methods still exhibit limitations. Methods based on image diffusion priors may suffer from temporal flickering, while those relying on sparse reconstruction cues, such as keyframes and lightweight side information, often compromise content fidelity. To alleviate these issues, GNVC-VD [13] employed a neural video codec to encode latent frames and used a video DiT to jointly enhance decoded latents within a group of pictures (GOP). Nevertheless, existing diffusion-based GVC methods have not explicitly explored hierarchical organization of latent frames with respect to their reference roles and coding quality, nor have they preserved coding-side reference structures in generative reconstruction.

## B. Hierarchical Reference Structure

Hierarchical reference structures organize frames in a GOP into multiple layers, where lower-layer frames are reused more often as references for higher-layer ones. In traditional video coding, such structures are typically combined with hierarchical quality structures, where lower-layer frames are encoded with finer quantization and higher fidelity [28]. Compared with simple sequential reference chains, hierarchical reference structures reduce error propagation, improve RD performance, and support temporal scalability [16], [15].

Hierarchical designs have been widely studied in NVC. Under low-delay settings, early work explored hierarchical quality structures for frame-level bit allocation in end-toend RD optimization. DCVC-DC [6] scaled the distortion term using hierarchy-dependent frame weights, and DCVC-FM [7] periodically refreshed reference features to suppress error propagation over long prediction chains. Beyond quality variation alone, DVCH [29] aligned hierarchical reference and quality structures under the low-delay B setting, while EHVC [17] introduced a hierarchical multi-reference scheme with learnable layer-wise quantization scales. Under randomaccess settings, HLVC [30] organized a GOP into quality layers and recurrently enhanced higher-layer frames using bidirectional references from lower layers. Later approaches [31], [32], [33], [34], [35], [36] further explored bidirectional references or context to improve temporal prediction.

Yet, hierarchical design has been explored only partially in diffusion-based GVC. DiffVC [12] adopted a simple frameby-frame reference chain together with a hierarchical quality structure inherited from [6], while GNVC-VD [13] also relied on frame-by-frame referencing and allocated bits uniformly across latent frames. Existing methods do not jointly organize reference structures and quality allocation in latent coding, nor do they carry coding-side hierarchical structure over to generative reconstruction. As a result, a unified hierarchical design across coding and generation remains unexplored in diffusion-based GVC.

## III. METHODOLOGY

## A. Overview

In this work, we aim to improve perceptual video reconstruction in diffusion-based GVC through a unified hierarchical design across coding and generation. To this end, GVCHR (c) Our hierarchical referencing with three layers.

![](images/48e3cd5a4498c87130a6851379b8c11ad92697f731cfc7d404f3cd199245448f.jpg)  
(b) Our hierarchical referencing with two layers.  
(d) Our hierarchical referencing with four layers.  
Fig. 3. Different settings of hierarchical reference structures. Wherein, w determines the frame-level coding quality and is employed in the RD loss.

organizes latent frames with coupled hierarchical reference and quality structures in latent coding, and preserves the resulting hierarchy during generative reconstruction. As illustrated in Fig. 2, GVCHR consists of hierarchical latent coding and diffusion-based generative reconstruction. The remainder of this section first introduces the overall framework, followed by Hierarchical Temporal Context Mining for latent coding in Section III-B, the Hierarchical Attentive Adapter for generative reconstruction in Section III-C, and the multi-stage training scheme in Section III-D.

1) Hierarchical Latent Coding: Source video ${ \textbf { \textsf { V } } } \in$ $\mathbb { R } ^ { ( 1 + T ) \times H \times W \times 3 }$ is first separated into keyframes and grayscale intermediate frames. The discarded color information of intermediate frames is later recovered during generation, which introduces little degradation in reconstruction quality while reducing the bitrate required for transmission. The resulting visual signals are then encoded to latent frames $\left\{ \mathbf { x } _ { t } \right\} _ { t = 0 } ^ { 1 + T / 4 }$ by a 3D VAE encoder [10].

To remove redundancy within $\left\{ \mathbf { x } _ { t } \right\} _ { t = 0 } ^ { 1 + T / 4 }$ , each latent frame $\mathbf { x } _ { t }$ is compressed by a latent codec built upon the conditional coding framework in [8]. The codec is enhanced with the proposed Hierarchical Temporal Context Mining (HTCM), which exploits a hierarchical reference structure to extract temporal context from previously decoded features:

$$
\mathbf { y } _ { t } = g _ { \mathrm { a } } ( \mathbf { x } _ { t } \mid \mathbf { c } _ { t } ) ,\tag{1}
$$

$$
\hat { \mathbf { y } } _ { t } = \left\lfloor \mathbf { y } _ { t } \right\rceil ,\tag{2}
$$

$$
\hat { \mathbf { x } } _ { t } , \hat { \mathbf { f } } _ { t } = g _ { \mathrm { s } } ( \hat { \mathbf { y } } _ { t } \mid \mathbf { c } _ { t } ) ,\tag{3}
$$

$$
\mathbf { \mu } \mathbf { , \sigma } = g _ { \mathrm { e p } } ( \mathbf { y } _ { t } \mid \mathbf { c } _ { t } ^ { \mathrm { e } } ) ,\tag{4}
$$

$$
\mathbf { c } _ { t } , \mathbf { c } _ { t } ^ { \mathrm { e } } = \mathrm { H T C M } ( \hat { \mathbf { f } } _ { < t } ) ,\tag{5}
$$

where $g _ { \mathrm { a } }$ and $g _ { \mathrm { s } }$ denote the latent encoder and decoder, respectively, and $\lfloor \cdot \rceil$ denotes rounding quantization. The decoded latent frame $\hat { \mathbf { x } } _ { t }$ and its feature $\hat { \mathbf { f } } _ { t }$ are reconstructed from $\hat { \mathbf { y } } _ { t } .$ The entropy model $g _ { \mathrm { e p } }$ estimates the parameters $\mu$ and σ of a Gaussian probability model for entropy coding. HTCM takes the previously decoded features $\hat { \mathbf { f } } _ { < t }$ as input and produces two temporal context representations, where $\mathbf { c } _ { t }$ is used for conditional latent encoding and decoding, and $\mathbf { c } _ { t } ^ { \mathrm { { e } } }$ for entropy parameter estimation. By organizing latent frames hierarchically, HTCM allocates more bits to lower-layer frames reused more frequently as references, and mines temporal context from their decoded features for more effective conditional coding. Details of HTCM are presented in Section III-B.

2) Diffusion-based Generative Reconstruction: For generative reconstruction, the decoded latent frames $\{ \hat { \mathbf { x } } _ { t } \} _ { t = 0 } ^ { 1 + \check { T } / 4 }$ and their features $\{ \hat { \mathbf { f } } _ { t } \} _ { t = 0 } ^ { 1 + T / 4 }$ are fed into a controllable video generation model composed of a video DiT and the proposed Hierarchical Attentive Adapter (HA-Adapter). We adopt a flow-matching formulation that learns a continuous velocity field $\epsilon _ { \tau }$ transporting Gaussian noise ${ \bf z } _ { N } \sim \mathcal { N } ( 0 , { \bf I } )$ toward the data manifold $\mathbf { z } _ { 0 }$ [37]. To guide denoising, HA-Adapter fuses $\left\{ \hat { \mathbf { x } } _ { t } \right\} _ { t = 0 } ^ { 1 + T / 4 }$ and $\bar { \{ \mathbf { f } _ { t } \} } _ { t = 0 } ^ { 1 + T / \overline { { 4 } } }$ into frame-wise representations $\left\{ \hat { \mathbf { x } } _ { t } ^ { \prime } \right\} _ { t = 0 } ^ { 1 + T / 4 }$ as control signals, which are injected into the video DiT through hierarchical attentive transformer blocks. Here, hierarchical attention restricts each frame to attend only to the same- or lower-layer references, preventing less reliable higher-layer frames from propagating artifacts during denoising. Details of HA-Adapter are given in Section III-C.

The denoising process is formulated as:

$$
\mathbf { z } _ { \tau - 1 } = \mathbf { z } _ { \tau } + ( \sigma _ { \tau - 1 } - \sigma _ { \tau } ) \cdot \hat { \epsilon } ( \mathbf { z } _ { \tau } , \tau , \mathbf { u } ) ,\tag{6}
$$

$$
\mathbf { u } = \mathrm { H A - A d a p t e r } ( \{ \hat { \mathbf { x } } _ { t } \} _ { t = 0 } ^ { 1 + T / 4 } , \{ \hat { \mathbf { f } } _ { t } \} _ { t = 0 } ^ { 1 + T / 4 } ) ,\tag{7}
$$

where $\begin{array} { r } { \mathbf { z } _ { \tau } ~ \in ~ \mathbb { R } ^ { ( 1 + T / 4 ) \times ( H / 8 ) \times ( W / 8 ) \times 1 6 } } \end{array}$ denotes the latent variable along the denoising trajectory, and 16 is the channel dimension of the 3D VAE bottleneck. The index $\tau =$ $0 , 1 , . . . , N - 1$ enumerates the flow steps and $\sigma _ { \tau }$ is the noise scale specified by a predefined schedule [9]. The video DiT predicts the velocity field $\hat { \epsilon } ( \mathbf { z } _ { \tau } , \tau , \mathbf { u } )$ , and HA-Adapter projects the decoded latents and their features into the conditioning signal u aligned with the intermediate feature space of the transformer. After N denoising steps, the generated latent $\mathbf { z } _ { 0 }$ is decoded by a 3D VAE decoder into the final pixel-space reconstruction V<sup>ˆ</sup> .

## B. Hierarchical Temporal Context Mining

Informative temporal context is crucial for reducing temporal redundancy in latent coding [5]. To this end, we propose HTCM, which exploits the hierarchical reference structure over $\left\{ \mathbf { x } _ { t } \right\} _ { t = 0 } ^ { 1 + T / 4 }$ to provide more informative temporal conditions for conditional coding. We first describe the hierarchical reference structure used in coding and then present short- and long-term context mining from selected references.

1) Hierarchical Reference Structure for Coding: Under the hierarchical reference structure, $\left\{ \mathbf { x } _ { t } \right\} _ { t = 0 } ^ { 1 + T / 4 }$ are organized into multiple layers such that higher-layer latent frames cannot serve as references for lower-layer ones. The associated hierarchical quality structure allocates more bits to lower-layer frames, which are reused more often as references. Although higher-layer latent frames are coded more compactly and contain more distortion, their limited use as references mitigates error accumulation and propagation. Such bit allocation is also compatible with keyframe sampling, where lowest-layer latent frames retain keyframe-related information and receive more bits to preserve color fidelity, while higher-layer frames retain only grayscale information.

![](images/78bd01bcacbb8690f2184f7203a8ee3d1672280005c63aedb7711c0d08742569.jpg)

![](images/f0c7ca43a66940198ebc0220d3270f64999db3d57c1180d76c0ac50c1819dc0e.jpg)  
(a) Previous all-to-all referencing in attention. (b) Full-attention mask  
(c) Our hierarchical referencing in attention.  
(d) Block-attention mask.  
Fig. 4. Comparison between the previous all-to-all reference structure [13] and our hierarchical reference structure performed by attention in the adapter. For clarity, $\hat { \mathbf { x } } _ { 3 } ^ { \prime } , \hat { \mathbf { x } } _ { 5 } ^ { \prime } , \hat { \mathbf { x } } _ { 6 } ^ { \prime } ,$ and xˆ<sup>′</sup> are omitted in Fig. 4a and Fig. 4c. Each yellow block in Fig. 4d corresponds to $\mathbf { M } _ { i , j } = \mathbf { 0 }$ in Equation 20, indicating that $\hat { \mathbf { x } } _ { j } ^ { \prime } .$ selected from $\{ \hat { \mathbf { x } } _ { t } ^ { \prime } \} _ { t = 0 } ^ { 1 + T / 4 }$ , serves as a reference for $\hat { \mathbf { x } } _ { i } ^ { \prime } .$

We investigate two-, three-, and four-layer settings under a GOP size of 32, corresponding to a latent GOP size of 8. The latent mini-GOP refers to a periodically repeated reference and quality pattern of size 4, as shown in Fig. 3. The coding quality of $\mathbf { x } _ { t }$ is controlled by $w _ { t } .$ a hierarchical temporal weight in the RD training loss, where latent frames at the same layer share the same weight. We denote $L _ { t }$ as the layer ID of $\mathbf { x } _ { t }$ . As shown in Section IV-C2, the three-layer setting gives the best compression performance. Specifically, for the 4-frame latent mini-GOP starting from the first inter-frame $\mathbf { x } _ { 1 }$ , the weights for $\mathbf { x } _ { 1 } ~ \mathbf { t o } ~ \mathbf { x } _ { 4 }$ are 0.5, 0.9, 0.5, and 1.2, with corresponding layer IDs $\{ L _ { t } \} _ { t = 1 } ^ { 4 } = \{ 2 , 1 , 2 , 0 \}$

2) Short- and Long-term Context Mining: Based on the hierarchical reference structure, HTCM extracts complementary temporal context from short- and long-term references, capturing local details and global semantics, respectively. For each latent frame $\mathbf { x } _ { t }$ , the short-term reference is the feature of the nearest previously decoded latent frame in the same or lower layers, denoted by $\hat { \mathbf { f } } _ { s } ,$ , where $s = \operatorname* { m a x } \{ i \mid i < t , L _ { i } \leq L _ { t } \}$ . As a carrier of local details, $\hat { \mathbf { f } } _ { s }$ is used as the short-term context:

$$
\mathbf { c } _ { t } ^ { \mathrm { S } } : = \hat { \mathbf { f } } _ { s } .\tag{8}
$$

In contrast, the long-term references consist of all previously decoded lowest-layer frames, represented by their corresponding features $\{ \hat { \mathbf { f } } _ { l } | l < t , L _ { l } = 0 \}$ . After each feature $\hat { \mathbf { f } } _ { l }$ is decoded, we use gated slot attention (GSA) [38] to recurrently update a fixed-size memory and obtain the corresponding retrieved long-term representation m<sub>l</sub>:

$$
\begin{array} { r } { \mathbf { m } _ { l } = \mathrm { G S A } ( \mathbf { q } _ { l } , \mathbf { k } _ { l } , \mathbf { v } _ { l } , \pmb { \alpha } _ { l } ) , } \end{array}\tag{9}
$$

$$
\mathbf { q } _ { l } , \mathbf { k } _ { l } , \mathbf { v } _ { l } = \mathbf { W } _ { \mathrm { q } } ^ { \mathrm { G } } ( \hat { \mathbf { f } } _ { l } + \mathbf { e } _ { l } ) , \mathbf { W } _ { \mathrm { k } } ^ { \mathrm { G } } ( \hat { \mathbf { f } } _ { l } + \mathbf { e } _ { l } ) , \mathbf { W } _ { \mathrm { v } } ^ { \mathrm { G } } ( \hat { \mathbf { f } } _ { l } + \mathbf { e } _ { l } ) ,\tag{10}
$$

$$
\alpha _ { l } = \mathbf { W } _ { \alpha } ^ { \mathrm { G } } ( \hat { \mathbf { f } } _ { l } + \mathbf { e } _ { l } ) ,\tag{11}
$$

$$
\begin{array} { r } { \mathbf { e } _ { l } = \mathrm { L i n e a r } ( \mathrm { S i L U } ( \mathrm { L i n e a r } ( [ L _ { l } , l , \mathbf { q p } _ { l } ] ) ) ) , } \end{array}\tag{12}
$$

where $\mathbf { W } _ { \mathrm { q } } ^ { \mathrm { G } } , \ \mathbf { W } _ { \mathrm { k } } ^ { \mathrm { G } } , \ \mathbf { W } _ { \mathrm { v } } ^ { \mathrm { G } }$ , and $\mathbf { W } _ { \alpha } ^ { \mathrm { G } }$ are learnable projection matrices. q<sub>l</sub>, $\mathbf { k } _ { l } .$ , and $\mathbf { v } _ { l }$ denote the query, key, and value, respectively, and $\alpha _ { l }$ is a data-dependent gating vector. The meta embedding $\mathbf { e } _ { l }$ encodes the layer ID $L _ { l } ,$ , latent frame index l, and quantization parameter ${ \mathrm { q p } } _ { l }$ . The GSA update and longterm representation retrieval are formulated as:

$$
\mathbf { m } _ { l } = ( \mathbf { h } _ { \mathrm { c u r } } ^ { \mathrm { v } } ) ^ { \top } \mathrm { s o f t m a x } \left( ( \mathbf { h } _ { \mathrm { c u r } } ^ { \mathrm { k } } ) ^ { \top } \mathbf { q } _ { l } \right) ,\tag{13}
$$

$$
\mathbf { h } _ { \mathrm { c u r } } ^ { \mathrm { k } } = \mathrm { D i a g } \left( \pmb { \alpha } _ { l } \right) \cdot \mathbf { h } _ { \mathrm { p r e } } ^ { \mathrm { k } } + \left( 1 - \pmb { \alpha } _ { l } \right) \otimes \mathbf { k } _ { l } ,\tag{14}
$$

$$
\mathbf { h } _ { \mathrm { c u r } } ^ { \mathrm { v } } = \mathrm { D i a g } \left( \pmb { \alpha } _ { l } \right) \cdot \mathbf { h } _ { \mathrm { p r e } } ^ { \mathrm { v } } + \left( 1 - \pmb { \alpha } _ { l } \right) \otimes \mathbf { v } _ { l } ,\tag{15}
$$

where $\mathbf { h } _ { \mathrm { c u r } } ^ { \mathrm { k } }$ and $\mathbf { h } _ { \mathrm { c u r } } ^ { \mathrm { v } }$ denote the memory states at the current step, updated by the long-term reference $\hat { \mathbf { f } } _ { l } . \ \mathbf { h } _ { \mathrm { p r e } } ^ { \mathrm { k } }$ and $\mathbf { h } _ { \mathrm { p r e } } ^ { \mathrm { v } }$ denote the states from the previous update step. ⊗ denotes the outer product.

For the current frame $\mathbf { x } _ { t }$ , we use the latest available longterm representation to connect global semantics with local details. Specifically, m<sub>l</sub> is aligned with the short-term context $\mathbf { c } _ { t } ^ { \mathrm { { S } } }$ through cross-attention [39] to produce the long-term context $\mathbf { c } _ { t } ^ { \mathrm { { \bar { L } } } }$

$$
\mathbf { c } _ { t } ^ { \mathrm { L } } = \mathrm { A t t e n t i o n } ( \mathbf { W } _ { \mathrm { q } } ^ { \mathrm { C } } ( \mathbf { c } _ { t } ^ { \mathrm { S } } + \mathbf { e } _ { s } ) , \mathbf { W } _ { \mathrm { k } } ^ { \mathrm { C } } \mathbf { m } _ { l } , \mathbf { W } _ { \mathrm { v } } ^ { \mathrm { C } } \mathbf { m } _ { l } ) ,\tag{16}
$$

$$
\begin{array} { r } { \mathbf { e } _ { s } = \mathrm { L i n e a r } ( \mathrm { S i L U } ( \mathrm { L i n e a r } ( [ L _ { s } , s , \mathrm { q p } _ { s } ] ) ) ) , } \end{array}\tag{17}
$$

where $\mathbf { W } _ { \mathrm { q } } ^ { \mathrm { C } } , \mathbf { W } _ { \mathrm { k } } ^ { \mathrm { C } }$ , and $\mathbf { W } _ { \mathrm { v } } ^ { \mathrm { C } }$ are learnable projection matrices, and $\mathbf { e } _ { s }$ is the meta embedding of $\hat { \mathbf { f } } _ { s } .$ . Finally, $\mathbf { c } _ { t } ^ { \mathrm { { S } } }$ and $\mathbf { c } _ { t } ^ { \mathrm { { L } } }$ are fused by a context refinement network, which follows the same architecture as the context extractor in [8] except that the input channel dimension is doubled, to produce $\mathbf { c } _ { t }$ and $\mathbf { c } _ { t } ^ { \mathrm { { e } } }$ for conditional coding.

## C. Hierarchical Attentive Adapter

Once $\left\{ \hat { \mathbf { x } } _ { t } \right\} _ { t = 0 } ^ { 1 + T / 4 }$ and $\{ \hat { \mathbf { f } } _ { t } \} _ { t = 0 } ^ { 1 + T / 4 }$ are decoded, the proposed HA-Adapter injects them into a video DiT as control signals for generative reconstruction. HA-Adapter is based on the adapter in VACE [10] with the original attention replaced by hierarchical attention. In this section, we first describe the hierarchical reference structure for generation, and then detail its implementation in hierarchical attentive transformer (HAT) blocks.

1) Hierarchical Reference Structure for Generation: As illustrated in $\mathrm { F i g . } 2 .$ , we first employ a control signal embedder, where $\left\{ \hat { \mathbf { x } } _ { t } \right\} _ { t = 0 } ^ { 1 + T / 4 }$ and $\{ \hat { \mathbf { f } } _ { t } \} _ { t = 0 } ^ { 1 + T \lceil 4 \rceil }$ are concatenated and patchified into frame-wise tokens $\{ \hat { \mathbf { x } } _ { t } ^ { \prime } \} _ { t = 0 } ^ { 1 + T / 4 }$ . Patchification is implemented using a convolutional layer with spatial downsampling factor of $^ { 2 , }$ such that $\hat { \mathbf { x } } _ { t } ^ { \prime } \in \mathbb { R } ^ { \left( \mathbf { \breve { H } } / 1 6 \cdot W / 1 6 \right) \times d }$ , where d is the token dimension. To project $\left\{ \hat { \mathbf { x } } _ { t } ^ { \prime } \right\} _ { t = 0 } ^ { 1 + T / 4 }$ into the feature space of the video DiT, GNVC-VD [13] uses full attention within transformer blocks to learn spatiotemporal correlations among all frame-wise tokens. In contrast, we use hierarchical attention to construct hierarchical reference structure for generation, as shown in Fig. 4. Specifically, each frame’s tokens attend only to other frames’ tokens from the same or lower layers according to the coding hierarchy. Since quality variation across $\{ \bar { \bf x } _ { t } ^ { \prime } \} _ { t = 0 } ^ { 1 + T / 4 }$ is inherited from the hierarchical quality structure in coding, this design prevents the propagation of coding artifacts from low-quality frames to high-quality ones. At the same time, each frame can draw cleaner guidance from higher-quality references, thereby reducing artifact propagation. Therefore, the hierarchical structures in coding and generation are intrinsically coupled.

TABLE I  
DETAILED CONFIGURATIONS OF THE MULTI-STAGE TRAINING STRATEGIES.
<table><tr><td>Stage Iterations Batch Size Patch Size Frames Learning Rate</td><td></td><td></td><td></td><td></td><td></td><td>Trainable Models</td><td>λ</td><td>Distortion Term D in the RD loss  $L _ { \mathrm { R D } }$ </td></tr><tr><td>1</td><td>30K</td><td>8</td><td> $\left| 2 5 6 \times 2 5 6 \right|$ </td><td>5</td><td> $1 0 ^ { - 4 }$ </td><td>Latent Codec</td><td>3072</td><td> $L _ { \mathrm { C o d e c } }$ </td></tr><tr><td>2</td><td>20K</td><td>8</td><td> $\left| 2 5 6 \times 2 5 6 \right|$ </td><td>17</td><td> $1 0 ^ { - 4 }$ </td><td>Latent Codec</td><td>3072</td><td> $L _ { \mathrm { C o d e c } }$ </td></tr><tr><td>3</td><td>20K</td><td>8</td><td> $\left| 2 5 6 \times 2 5 6 \right|$ </td><td>33</td><td> $1 0 ^ { - 4 }$ </td><td>Latent Codec</td><td>3072</td><td> $L _ { \mathrm { C o d e c } }$ </td></tr><tr><td>4</td><td>30K</td><td>8</td><td> $\left| 2 5 6 \times 2 5 6 \right|$ </td><td>33</td><td> $1 0 ^ { - 4 }$ </td><td>Latent Codec, HA-Adapter</td><td>3072</td><td> $L _ { \mathrm { C o d e c } } + L _ { \mathrm { C F M } }$ </td></tr><tr><td>5</td><td>40K</td><td>8</td><td> $\left| 2 5 6 \times 2 5 6 \right|$ </td><td>33</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>Latent Codec, HA-Adapter</td><td>160, 289, 635, 1397, 3072</td><td> $0 . 1 L _ { \mathrm { C o d e c } } + 0 . 1 L _ { \mathrm { C F M } } + L _ { \mathrm { R e c } } + 0 . 1 L _ { \mathrm { L P I P S } }$ </td></tr><tr><td>6</td><td>60K</td><td>4</td><td> ${ \left| 4 4 8 \times 4 4 8 \right| }$ </td><td>33</td><td> $2 \times 1 0 ^ { - 5 }$ </td><td>Latent Codec, HA-Adapter</td><td>160, 289, 635, 1397, 3072</td><td> $0 . 0 2 5 L _ { \mathrm { C o d e c } } + L _ { \mathrm { R e c } } + 0 . 1 L _ { \mathrm { L P I P S } }$ </td></tr></table>

2) Hierarchical Attentive Transformer: To enable hierarchical referencing within the adapter, we feed $\left\{ \hat { \mathbf { x } } _ { t } ^ { \prime } \right\} _ { t = 0 } ^ { 1 + T / 4 }$ into HAT blocks instead of the original transformer blocks. The key distinction lies in the incorporation of a block-wise attention mask $\mathbf { M } \in \mathbb { R } ^ { ( H / 1 6 \cdot W / 1 6 \cdot ( 1 + T / 4 ) ) \times ( H / 1 6 \cdot W / 1 6 \cdot ( 1 + T / 4 ) ) }$

$$
\mathbf { o } = \mathrm { s o f t m a x } \left( \frac { \mathbf { q } \mathbf { k } ^ { \top } } { \sqrt { d } } + \mathbf { M } \right) \mathbf { v } ,\tag{18}
$$

$$
\mathbf { M } = \left[ \begin{array} { c c c c } { \mathbf { M } _ { 0 , 0 } } & { \mathbf { M } _ { 0 , 1 } } & { \dots } & { \mathbf { M } _ { 0 , 1 + T / 4 } } \\ { \mathbf { M } _ { 1 , 0 } } & { \mathbf { M } _ { 1 , 1 } } & { \dots } & { \mathbf { M } _ { 1 , 1 + T / 4 } } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { \mathbf { M } _ { 1 + T / 4 , 0 } } & { \mathbf { M } _ { 1 + T / 4 , 1 } } & { \dots } & { \mathbf { M } _ { 1 + T / 4 , 1 + T / 4 } } \end{array} \right] ,\tag{19}
$$

$$
\mathbf { M } _ { i , j } = \left\{ \begin{array} { l l } { - \infty } & { \mathrm { i f ~ } L _ { i } < L _ { j } } \\ { \mathbf { 0 } } & { \mathrm { i f ~ } L _ { i } \geq L _ { j } } \end{array} \right. , \ 0 \leq i , j \leq 1 + T / 4 ,\tag{20}
$$

where q, k, $\mathbf { v } \in \mathbb { R } ^ { ( H / 1 6 \cdot W / 1 6 \cdot ( 1 + T / 4 ) ) }$ <sup>)×d</sup> denote the projected embeddings of $\left\{ \hat { \mathbf { x } } _ { t } ^ { \prime } \right\} _ { t = 0 } ^ { 1 + T / 4 }$ for hierarchical attention in each HAT block. Each block $\mathbf { M } _ { i , j } ~ \in ~ \mathbb { R } ^ { ( H / 1 6 \cdot W / 1 6 ) \times ( H / 1 6 \cdot W / 1 6 ) }$ controls the attention weights between $\hat { \mathbf { x } } _ { i } ^ { \prime }$ and $\hat { \mathbf { x } } _ { i } ^ { \prime }$ , whose layer IDs are $L _ { i }$ and $L _ { j }$ , respectively. When $\mathbf { M } _ { i , j } = \mathbf { 0 } , { \hat { \mathbf { x } } _ { i } ^ { \prime } }$ is allowed to attend to $\hat { \mathbf { x } } _ { j } ^ { \prime }$ under the condition $L _ { i } ~ \ge ~ L _ { j }$ . Otherwise, $\mathbf { M } _ { i , j } = - \infty$ mask out the corresponding attention weights. We retain $\mathbf { M } _ { i , j } = \mathbf { 0 }$ when $i = j ,$ , allowing the model to balance the referenced information and self-information. Fig. 4d shows the mask under the three-layer setting.

Additionally, owing to the block-wise mask, the hierarchical attention in HAT can be accelerated via block attention [40], which further reduces the runtime of generative reconstruction.

## D. Multi-stage Training Strategies

To optimize GVCHR, we adopt a multi-stage training scheme that progressively bridges the gap between the genera-

tive prior and codec-induced distortion. Generally, the training objective is a RD loss $L _ { \mathrm { R D } }$ , defined as:

$$
\begin{array} { r } { L _ { \mathrm { R D } } = R + \lambda D , } \end{array}\tag{21}
$$

where R denotes the bitrate estimated by the latent codec. The parameter λ controls the RD trade-off, and D denotes the distortion term whose form varies across training stages. Specifically, we consider the following distortion terms:

1) Latent fidelity loss $L _ { \mathrm { C o d e c } } { \mathrm { : } }$

$$
\mathit { L } _ { \mathrm { C o d e c } } = \frac { 1 } { 1 + T / 4 } \sum _ { t = 0 } ^ { 1 + T / 4 } w _ { t } \cdot d _ { \mathrm { M S E } } ( \mathbf { x } _ { t } , \hat { \mathbf { x } } _ { t } ) ,\tag{22}
$$

where $d _ { \mathrm { M S E } }$ denotes mean square error (MSE), and $w _ { t }$ is the hierarchical temporal weight used in the hierarchical quality structure.

2) Conditional flow-matching loss $L _ { \mathrm { C F M } }$

$$
L _ { \mathrm { C F M } } = \frac { \mathbb { E } _ { \tau \sim \mathcal { U } ( 0 , N ) } \left\| \hat { \epsilon } ( \mathbf { z } _ { \tau } , \tau , \mathbf { u } ) - ( \mathbf { z } _ { N } - \mathbf { z } _ { 0 } ) \right\| _ { 2 } ^ { 2 } } { ( 1 + T / 4 ) \times ( H / 8 ) \times ( W / 8 ) \times 1 6 } ,\tag{23}
$$

where $\left( { \bf z } _ { N } - { \bf z } _ { 0 } \right)$ serves as the training target of velocity field, and $\hat { \epsilon } ( \mathbf { z } _ { \tau } , \tau , \mathbf { u } )$ is its corresponding estimation.

3) Pixel-domain reconstruction loss $L _ { \mathrm { { R e c } } } { : }$

$$
L _ { \mathrm { R e c } } = \frac { 1 } { 1 + T } \sum _ { i = 0 } ^ { T } w _ { i } ^ { \mathrm { F } } \cdot d _ { \mathrm { M S E } } ( \mathbf { F } _ { i } , \hat { \mathbf { F } } _ { i } ) ,\tag{24}
$$

where $\mathbf { F } _ { i }$ denotes the i-th frame in the source video V, and $\hat { \mathbf { F } } _ { i }$ is its reconstructed counterpart. We set $w _ { i } ^ { \mathrm { F } } = w _ { t }$ if $\mathbf { F } _ { i }$ is encoded into $\mathbf { x } _ { t }$

4) Pixel-domain perceptual loss $L _ { \mathrm { L P I P S } }$

$$
\boldsymbol { L } _ { \mathrm { L P I P S } } = \frac { 1 } { 1 + T } \sum _ { i = 0 } ^ { T } w _ { i } ^ { \mathrm { F } } \mathrm { L P I P S } ( \mathbf { F } _ { i } , \hat { \mathbf { F } } _ { i } ) ,\tag{25}
$$

where LPIPS measures perceptual quality [41].

The detailed configurations of different training stages are summarized in Table I. In the first three stages, we train only the latent codec and progressively increase the number of training frames from 5 to 33. During these stages, we resize the shorter side of each training video to 256 pixels and randomly crop $2 5 6 \times 2 5 6$ patches. In the fourth stage, the latent codec and HA-Adapter are jointly optimized using $L _ { \mathrm { C o d e c } }$ and $L _ { \mathrm { C F M } }$ to adapt the diffusion model to codecdistorted latents. In the fifth stage, we select five values of λ to train distinct models corresponding to five quality levels, and further incorporate $L _ { \mathrm { { R e c } } }$ and $L _ { \mathrm { L P I P S } }$ to improve reconstruction fidelity and perceptual quality. In the sixth stage, we finetune all models at a higher resolution, remove $L _ { \mathrm { C F M } }$ , and reduce the weight of $L _ { \mathrm { C o d e c } }$ so that the optimization focuses more on pixel-domain reconstruction.

![](images/cc48dc57626ca8d76a2ecd5732901a1b35754b7b91d2bb1ddf2dad6781adbc74.jpg)  
Fig. 5. Rate-perception curves on HEVC B, UVG, and MCL-JCV.

TABLE II  
BD-RATE (%) ↓ / BD-METRIC ↑ ON HEVC B, UVG, AND MCL-JCV.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>HEVCB</td><td rowspan=1 colspan=4>UVG</td><td rowspan=1 colspan=4>MCL-JCV</td></tr><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>LPIPS</td><td rowspan=1 colspan=1>DISTS</td><td rowspan=1 colspan=1>FID</td><td rowspan=1 colspan=1>FloLPIPS</td><td rowspan=1 colspan=1>LPIPS</td><td rowspan=1 colspan=1>DISTS</td><td rowspan=1 colspan=1>FID</td><td rowspan=1 colspan=1>FloLPIPS</td><td rowspan=1 colspan=1>LPIPS</td><td rowspan=1 colspan=1>DISTS</td><td rowspan=1 colspan=1>FID</td><td rowspan=1 colspan=1>FloLPIPS</td></tr><tr><td rowspan=1 colspan=1>VTM-17.0</td><td rowspan=1 colspan=1>N/A / -0.1412</td><td rowspan=1 colspan=1>N/A / -0.0571</td><td rowspan=1 colspan=1>25.6 / -3.4</td><td rowspan=1 colspan=1>N/A / -0.1739</td><td rowspan=1 colspan=1>N/A / -0.1216</td><td rowspan=1 colspan=1>N/A / -0.0667</td><td rowspan=1 colspan=1>37.4 / -3.4</td><td rowspan=1 colspan=1>N/A / -0.1618</td><td rowspan=1 colspan=1>N/A / -0.1433</td><td rowspan=1 colspan=1>N/A / -0.0847</td><td rowspan=1 colspan=1>89.9 / -15.0</td><td rowspan=1 colspan=1>N/A / -0.1576</td></tr><tr><td rowspan=1 colspan=1>DCVC-FM</td><td rowspan=1 colspan=1>N/A / -0.1465</td><td rowspan=1 colspan=1>N/A / -0.0722</td><td rowspan=1 colspan=1>71.0 / -13.2</td><td rowspan=1 colspan=1>N/A / -0.1862</td><td rowspan=1 colspan=1>N/A / -0.1311</td><td rowspan=1 colspan=1>N/A / -0.0843</td><td rowspan=1 colspan=1>85.6 / -11.3</td><td rowspan=1 colspan=1>N/A / -0.1902</td><td rowspan=1 colspan=1>N/A / -0.1582</td><td rowspan=1 colspan=1>N/A / -0.1025</td><td rowspan=1 colspan=1>148.1 / -28.4</td><td rowspan=1 colspan=1>N/A / -0.1823</td></tr><tr><td rowspan=1 colspan=1>PLVC</td><td rowspan=1 colspan=1>N/A / N/A</td><td rowspan=1 colspan=1>N/A / N/A</td><td rowspan=1 colspan=1>N/A / N/A</td><td rowspan=1 colspan=1>110.6 / N/A</td><td rowspan=1 colspan=1>N/A / N/A</td><td rowspan=1 colspan=1>N/A / N/A</td><td rowspan=1 colspan=1>N/A / N/A</td><td rowspan=1 colspan=1>N/A / N/A</td><td rowspan=1 colspan=1>118.8 / N/A</td><td rowspan=1 colspan=1>139.6 / N/A</td><td rowspan=1 colspan=1>89.4 / N/A</td><td rowspan=1 colspan=1>243.0 / N/A</td></tr><tr><td rowspan=1 colspan=1>DiffVC</td><td rowspan=1 colspan=1>-1 -</td><td rowspan=1 colspan=1>-1-</td><td rowspan=1 colspan=1>- 1-</td><td rowspan=1 colspan=1>-1 -</td><td rowspan=1 colspan=1>298.2 / N/A</td><td rowspan=1 colspan=1>N/A / N/A</td><td rowspan=1 colspan=1>-1-</td><td rowspan=1 colspan=1>- 1 -</td><td rowspan=1 colspan=1>162.6 / N/A</td><td rowspan=1 colspan=1>N/A / N/A</td><td rowspan=1 colspan=1>-1-</td><td rowspan=1 colspan=1>-1-</td></tr><tr><td rowspan=1 colspan=1>GLC-Video</td><td rowspan=1 colspan=1>0.0 / 0.0000</td><td rowspan=1 colspan=1>0.0 / 0.0000</td><td rowspan=1 colspan=1>0.0 / 0.0</td><td rowspan=1 colspan=1>0.0 / 0.0000</td><td rowspan=1 colspan=1>0.0 / 0.0000</td><td rowspan=1 colspan=1>0.0 / 0.0000</td><td rowspan=1 colspan=1>0.0 / 0.0</td><td rowspan=1 colspan=1>0.0 / 0.0000</td><td rowspan=1 colspan=1>0.0 / 0.0000</td><td rowspan=1 colspan=1>0.0 / 0.0000</td><td rowspan=1 colspan=1>0.0 / 0.0</td><td rowspan=1 colspan=1>0.0 / 0.0000</td></tr><tr><td rowspan=1 colspan=1>GNVC-VD</td><td rowspan=1 colspan=1>-22.9 / 0.0168</td><td rowspan=1 colspan=1>-43.4 / 0.0208</td><td rowspan=1 colspan=1>-1-</td><td rowspan=1 colspan=1>-1-</td><td rowspan=1 colspan=1>19.3 / -0.0047</td><td rowspan=1 colspan=1>-2.0 / 0.0034</td><td rowspan=1 colspan=1>-1-</td><td rowspan=1 colspan=1>-1-</td><td rowspan=1 colspan=1>0.5 / -0.0002</td><td rowspan=1 colspan=1>-15.7 / 0.0082</td><td rowspan=1 colspan=1>- 1-</td><td rowspan=1 colspan=1>- 1 -</td></tr><tr><td rowspan=1 colspan=1>GVCHR</td><td rowspan=1 colspan=1>-63.7 / 0.0405</td><td rowspan=1 colspan=1>N/A / 0.0370</td><td rowspan=1 colspan=1>-49.3 / 10.2</td><td rowspan=1 colspan=1>-30.7 / 0.0117</td><td rowspan=1 colspan=1>-65.0 / 0.0326</td><td rowspan=1 colspan=1>-74.0 / 0.0258</td><td rowspan=1 colspan=1>-7.6 / 2.6</td><td rowspan=1 colspan=1>-51.4 / 0.0237</td><td rowspan=1 colspan=1>-36.4 / 0.0152</td><td rowspan=1 colspan=1>-65.8 / 0.0251</td><td rowspan=1 colspan=1>-22.2 / 3.9</td><td rowspan=1 colspan=1>9.7 / -0.0036</td></tr></table>

The best and second-best results are highlighted in red and blue, respectively. “N/A” indicates that BD-rate cannot be computed due to insufficient overlap between the rate–metric curves. “-” denotes that the closed-source baseline does not report the result.

## IV. EXPERIMENTS

## A. Experimental Protocol

1) Training Details: To build a high-quality training set, we curate source video clips from Pexels<sup>1</sup>. For data filtering, we retain clips whose aesthetic scores in VBench [42] are above 4.5 and clarity scores based on CLIP-IQA [43] are above 0.65, where both metrics are averaged across all frames in each clip. We further preserve only samples with aspect ratios close to 16:9, yielding approximately 480K videos for the training.

We initialize the HA-Adapter and the video DiT with the pretrained weights from VACE [10], while keeping the HA-Adapter trainable and the video DiT frozen. The latent codec builds upon the architecture of DCVC-RT [8] and is trained from scratch. AdamW is used as the optimizer [44]. The latent GOP size and latent mini-GOP size are set to 8 and 4, respectively, corresponding to a GOP size of 32. Here, a GOP size of 32 denotes 32 inter-frames excluding the first intra-frame, so each coding and generation pass operates on 33 frames in total. The trade-off parameter λ in the RD loss is sampled at five points via logarithmic interpolation over [160, 3072]. Following [8], each λ is paired with a QP to control bitrate during inference. For each hierarchical temporal weight w<sub>t</sub>, we assign a QP offset. Specifically, the QP offset is 8 for $w _ { 0 } ~ = ~ 8 . 0 .$ . For the periodically repeated pattern $[ w _ { t } , w _ { t + 1 } , w _ { t + 2 } , w _ { t + 3 } ] = [ 0 . 5 , 0 . 9 , 0 . 5 , 1 . 2 ]$ in a latent mini-GOP of size 4, the QP offsets are set to [0, 2, 0, 4]. Additional training settings are summarized in Table I.

2) Test Details: We evaluate GVCHR on the HEVC B [2], UVG [45], and MCL-JCV [46] datasets, where raw YUV420 videos are converted to RGB using BT.709 [6]. For each sequence, the first 96 frames are tested under the GOP 32 setting. In the last segment, the final frame is replicated as padding to reach 33 frames, and the padded frames are discarded after reconstruction. We use BD-Rate and BD-Metric [47] to evaluate compression performance. Signal fidelity is measured by PSNR and MS-SSIM [48]. Perceptual quality is assessed by LPIPS [41] and DISTS [49]. To assess temporal perceptual consistency, we use FloLPIPS [50], a flow-guided LPIPS variant. To measure generative realism, we compute FID [51] following [52]. Bitrate is reported in bits per pixel (BPP).

![](images/dbe6627b0a0bfc80b5f84c1e2e3da98cb2317c49ee62e24a5c0a349257e21b76.jpg)  
Fig. 6. Visual comparisons with open-source baselines on the Kimono sequence from HEVC B.

![](images/1e0f846496dbfc5f80c1f17a94b57b94a38bea7edb3a815f11bc802e0350f39b.jpg)  
Fig. 7. Rate-fidelity curves on HEVC B, UVG, and MCL-JCV.

3) Baselines: We compare GVCHR with three categories of baseline: 1) the traditional video codec VTM-17.0 [3]; 2) the fidelity-oriented NVC method DCVC-FM [7]; and 3) GVC methods, including PLVC [20], GLC-Video [22], DiffVC [12], and GNVC-VD [13]. For VTM-17.0, we follow the evaluation protocol in [6]. Since the code for DiffVC and GNVC-VD is unavailable, we report the results extracted from their paper. Other baselines are tested with their default settings.

We use the latest open-source GLC-Video as the anchor for BD-rate computation, rather than VTM-17.0 as usual, because the RD curves of VTM-17.0 are often far from those of GVC methods and have limited overlap in quality range.

## B. Comparison Results

1) Quantitative Results: As shown in Table II and Fig. 5, GVCHR outperforms all baselines on most perceptual metrics.

![](images/98709d0ba80d5c686d706b14ef59362706400c8f8af54c52bb0991fdd7b8965d.jpg)  
Fig. 8. Visual comparison of temporal coherence by stacking the red pixel row across consecutive frames in the videoSRC15 sequence from MCL-JCV.

TABLE III  
BD-RATE (%)↓ AND RUNTIME (MS) ↓ COMPARISON WITH GNVC-VD.
<table><tr><td rowspan="2">Method</td><td colspan="4">Averaged on three test sets</td><td rowspan="2">Enc. T | Dec. T</td><td rowspan="2"></td></tr><tr><td></td><td></td><td></td><td>LPIPS | DISTS | PSNR | MS-SSIM |</td></tr><tr><td>GNVC-VD</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>439</td><td>4225</td></tr><tr><td>GVCHR (GOP 24)</td><td>-47.0</td><td>-49.1</td><td>-12.4</td><td>-4.6</td><td>472</td><td>2431</td></tr><tr><td>GVCHR (GOP 32)</td><td>-50.5</td><td>-54.0</td><td>-23.3</td><td>-11.1</td><td>396</td><td>2814</td></tr></table>

“Enc. T” / “Dec. T” denotes the per-frame encoding / decoding time (ms), tested on an NVIDIA H20 GPU. We pair GOP 24 with latent GOP 6 and latent mini-GOP 6, and GOP 32 with latent GOP 8 and latent mini-GOP 4.

Compared to GLC-Video, GVCHR achieves BD-rate gain of 55.0% w.r.t. LPIPS averaged on all three test sets. In terms of PSNR and MS-SSIM, GVCHR remains competitive among all GVC methods, as shown in Fig. 7. Additionally, we report the performance of GVCHR using the latest closed-source method GNVC-VD as the anchor in Table III. Compared with GNVC-VD, GVCHR achieves average BD-rate gain of 50.5% w.r.t. LPIPS under the GOP 32 setting. GVCHR also outperforms GNVC-VD under the GOP 24 setting, which matches the setting used in their paper. Note that GOP 24 refers to 24 inter-frames excluding the first intra-frame, and therefore corresponds to the GOP 25 setting reported by GNVC-VD which counts both intra- and inter-frames. These results demonstrate the state-of-the-art compression performance of GVCHR in diffusion-based GVC.

2) Qualitative Results: As shown in Fig. 6, GVCHR reconstructs visually pleasing details with higher perceptual quality at lower bitrates. The fidelity-oriented NVC method DCVC-FM struggles to recover clear textures, resulting in blurred regions. Both PLVC and GLC-Video improve detail sharpness but introduce noticeable variegated and garbled artifacts. Furthermore, Fig. 8 highlights GVCHR’s superior temporal coherence. Under cross-frame motion, DCVC-FM introduces blurring, whereas PLVC and GLC-Video exhibit flickering and temporally discontinuous textures. In contrast, GVCHR preserves sharp details with smoother temporal transitions.

TABLE IV  
BD-RATE (%)↓ COMPARISON OF DIFFERENT MODEL VARIANTS.
<table><tr><td rowspan="2">Method</td><td colspan="4">MCL-JCV</td></tr><tr><td>LPIPS</td><td>DISTS</td><td>FloLPIPS</td><td>PSNR</td></tr><tr><td>w/o Short-term Reference</td><td>44.5</td><td>42.4</td><td>35.1</td><td>37.2</td></tr><tr><td rowspan="3">w/o Long-term Reference w/o Keyframe Sampling</td><td>21.4</td><td>26.8</td><td>14.6</td><td>18.5</td></tr><tr><td>11.4</td><td>16.6</td><td>8.5</td><td>11.2</td></tr><tr><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr></table>

TABLE V

BD-RATE (%)↓ COMPARISON OF VARIOUS CONFIGURATIONS OF HIERARCHICAL REFERENCE STRUCTURES.
<table><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=3>Coding      Generation</td><td rowspan=1 colspan=4>MCL-JCV</td></tr><tr><td rowspan=1 colspan=3>Reference Quality  Referencehierarchyhierarchy hierarchy</td><td rowspan=1 colspan=1>LPIPS</td><td rowspan=1 colspan=1>DISTS</td><td rowspan=1 colspan=1>FloLPIPS</td><td rowspan=1 colspan=1>PSNR</td></tr><tr><td rowspan=2 colspan=1>(a)(b)(c)(d)</td><td rowspan=2 colspan=3>123 (Ours)4</td><td rowspan=1 colspan=1>19.3</td><td rowspan=1 colspan=1>20.7</td><td rowspan=1 colspan=1>7.1</td><td rowspan=1 colspan=1>17.5</td></tr><tr><td rowspan=1 colspan=1>9.20.0-3.1</td><td rowspan=1 colspan=1>11.80.0-2.6</td><td rowspan=1 colspan=1>5.20.02.4</td><td rowspan=1 colspan=1>12.40.06.2</td></tr><tr><td rowspan=2 colspan=1>(e)(f)(g)</td><td rowspan=2 colspan=1>113</td><td rowspan=2 colspan=1>333</td><td rowspan=2 colspan=1>131</td><td rowspan=1 colspan=1>12.98.4</td><td rowspan=1 colspan=1>15.29.7</td><td rowspan=2 colspan=1>9.36.85.3</td><td rowspan=2 colspan=1>16.410.58.1</td></tr><tr><td rowspan=1 colspan=1>7.2</td><td rowspan=1 colspan=1>7.4</td></tr></table>

3) Complexity: As shown in Table III, we compare the perframe encoding and decoding time of GVCHR with GNVC-VD. Since GNVC-VD is not open-source, we implement its architecture without training. Under the GOP 32 setting, GVCHR achieves faster encoding and decoding, owing to fewer costly intra-coding operations and the acceleration of block attention in the proposed HA-Adapter. Under the GOP 24 setting used in the GNVC-VD paper, GVCHR is only slightly slower in encoding. This is expected because the proposed HTCM increases encoding complexity. Meanwhile, GVCHR employs the intra-model of DCVC-RT, whereas GNVC-VD adopts the more complex image codec ELIC [53], which helps narrow the gap. Overall, GVCHR improves RD performance while maintaining relatively fast coding speed.

## C. Ablations

1) Short- and Long-term References: We ablate short- and long-term references in HTCM. As shown in Table IV, both contribute significantly to compression performance.

2) Hierarchical Reference Structures: As presented in Table V, rows (a) to (d) evaluate different numbers of hierarchies in coding and generation. Notably, row (a) follows the setting of GNVC-VD, which discards both hierarchical reference and quality structures. Overall, the three-layer setting in row (c) achieves the superior performance. We further compare row (a) and row (c) in terms of frame-level perceptual quality and bit cost in Fig. 9, which shows that both reference and quality hierarchies effectively suppresses error propagation and achieves better bit allocation. Row (e) uses only the hierarchical quality structure in coding, following DiffVC. Rows (f) and (g) remove the hierarchical reference structures from coding and generation, respectively. The results in rows (e), (f), and (g) show that the hierarchical reference and quality structures in coding, together with the hierarchical reference structure in generation, are coupled and jointly contribute to compression performance.

![](images/9e87b316087662eee57aad61b40f89bfbfc0e77f414571feb9e5fbdc1ada4506.jpg)

![](images/9a7423d5c5378bb98d9bfa90625c2f267c20998b1d4655f35ef691c812ace813.jpg)

Fig. 9. Perceptual quality and bit cost across latent frames within a latent GOP of the Cactus sequence from HEVC B. Note that the LPIPS of a latent frame is averaged over its corresponding pixel-domain frames.  
![](images/18e351c23795e611fb0bc978a28fa513cef5ef9ea17a0ea8acef65949eddc615.jpg)  
Fig. 10. Trade-off between performance and complexity across different GOP sizes (latent GOP sizes) on MCL-JCV.

3) GOP Size: Fig. 10 presents a performance comparison of applying hierarchical reference and quality structures across different GOP sizes (or latent GOP sizes). The results demonstrate that a larger latent GOP size improves BD-Rate w.r.t. LPIPS, but increases decoding time, since the computational cost of attention in the generative model scales quadratically with the input GOP size. Our default setting, i.e., latent GOP 8, achieves promising compression performance while keeping decoding time relatively low. Additionally, compared with the model variant without reference and quality hierarchy, the benefits of our hierarchical designs become more pronounced as the GOP size increases. This again shows that these hierarchical structures are effective in suppressing error propagation over long dependency chains in diffusion-based GVC.

4) Keyframe Sampling: As shown in Table IV, removing keyframe sampling leads to inferior performance, as preserving all color information in the input increases the transmission budget. This demonstrates that GVCHR can recover full color information from a limited number of keyframes.

## V. CONCLUSION

In this paper, we propose GVCHR, a generative video compression framework based on hierarchical referencing. GVCHR organizes latent frames by reference roles and coding quality, enabling high-quality references to support efficient coding and effective generative reconstruction. In latent coding, GVCHR couples hierarchical reference and quality structures, and introduces HTCM to exploit complementary shortand long-term temporal context. In generative reconstruction, GVCHR injects decoded latents into a video DiT through a HA-Adapter, which follows the coding-side hierarchy to reduce artifact propagation during denoising. Experiments on multiple benchmarks demonstrate state-of-the-art compression performance of GVCHR among existing GVC methods.

## REFERENCES

[1] T. Wiegand, G. J. Sullivan, G. Bjontegaard, and A. Luthra, “Overview of the h.264/avc video coding standard,” TCSVT, vol. 13, no. 7, pp. 560–576, 2003.

[2] G. J. Sullivan, J.-R. Ohm, W.-J. Han, and T. Wiegand, “Overview of the high efficiency video coding (hevc) standard,” TCSVT, vol. 22, no. 12, pp. 1649–1668, 2012.

[3] B. Bross, Y.-K. Wang, Y. Ye, S. Liu, J. Chen, G. J. Sullivan, and J.-R. Ohm, “Overview of the versatile video coding (vvc) standard and its applications,” TCSVT, vol. 31, no. 10, pp. 3736–3764, 2021.

[4] J. Li, B. Li, and Y. Lu, “Deep contextual video compression,” NeurIPS, vol. 34, pp. 18 114–18 125, 2021.

[5] X. Sheng, J. Li, B. Li, L. Li, D. Liu, and Y. Lu, “Temporal context mining for learned video compression,” TMM, vol. 25, pp. 7311–7322, 2022.

[6] J. Li, B. Li, and Y. Lu, “Neural video compression with diverse contexts,” in CVPR, 2023, pp. 22 616–22 626.

[7] ——, “Neural video compression with feature modulation,” in CVPR, 2024, pp. 26 099–26 108.

[8] Z. Jia, B. Li, J. Li, W. Xie, L. Qi, H. Li, and Y. Lu, “Towards practical real-time neural video compression,” in CVPR, 2025, pp. 12 543–12 552.

[9] T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang et al., “Wan: Open and advanced large-scale video generative models,” arXiv preprint arXiv:2503.20314, 2025.

[10] Z. Jiang, Z. Han, C. Mao, J. Zhang, Y. Pan, and Y. Liu, “Vace: All-in-one video creation and editing,” in ICCV, 2025, pp. 17 191–17 202.

[11] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng et al., “Cogvideox: Text-to-video diffusion models with an expert transformer,” in ICLR, vol. 2025, 2025, pp. 83 048–83 077.

[12] W. Ma and Z. Chen, “Diffusion-based perceptual neural video compression with temporal diffusion information reuse,” TOMM, 2025.

[13] Q. Mao, H. Cheng, T. Yang, L. Jin, and S. Ma, “Generative neural video compression via video diffusion prior,” in CVPR, 2026, pp. 43 239– 43 248.

[14] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” arXiv preprint arXiv:1312.6114, 2013.

[15] H. Schwarz, D. Marpe, and T. Wiegand, “Overview of the scalable video coding extension of the h. 264/avc standard,” TCSVT, vol. 17, no. 9, pp. 1103–1120, 2007.

[16] J. M. Boyce, Y. Ye, J. Chen, and A. K. Ramasubramonian, “Overview of shvc: Scalable extensions of the high efficiency video coding standard,” TCSVT, vol. 26, no. 1, pp. 20–34, 2015.

[17] J. Liao, Y. Wu, C. Lin, Z. Deng, L. Li, D. Liu, and X. Sun, “Ehvc: Efficient hierarchical reference and quality structure for neural video coding,” in ACM MM, 2025, pp. 12 083–12 091.

[18] S. Zhang, M. Mrak, L. Herranz, M. G. Blanch, S. Wan, and F. Yang, “Dvc-p: Deep video compression with perceptual optimizations,” in VCIP, 2021, pp. 1–5.

[19] F. Mentzer, E. Agustsson, J. Balle, D. Minnen, N. Johnston, and´ G. Toderici, “Neural video compression using gans for detail synthesis and propagation,” in ECCV, 2022, pp. 562–578.

[20] R. Yang, R. Timofte, and L. Van Gool, “Perceptual learned video compression with recurrent conditional gan.” in IJCAI, 2022, pp. 1537– 1544.

[21] M. Li, Y. Shi, J. Wang, and Y. Huang, “High visual-fidelity learned video compression,” in ACM MM, 2023, pp. 8057–8066.

[22] L. Qi, Z. Jia, J. Li, B. Li, H. Li, and Y. Lu, “Generative latent coding for ultra-low bitrate image and video compression,” TCSVT, 2025.

[23] A. Van Den Oord, O. Vinyals et al., “Neural discrete representation learning,” NeurIPS, vol. 30, 2017.

[24] M. Liu, C. Xu, Y. Gu, C. Yao, and Y. Zhao, “I<sup>2</sup>vc: A unified framework for intra-& inter-frame video compression,” arXiv preprint arXiv:2405.14336, 2024.

[25] R. Wan, Q. Zheng, and Y. Fan, “M3-cvc: Controllable video compression with multimodal generative models,” in ICASSP, 2025, pp. 1–5.

[26] Z. Wang, H. Man, W. Li, X. Wang, X. Fan, and D. Zhao, “T-gvc: Trajectory-guided generative video coding at ultra-low bitrates,” in AAAI, vol. 40, no. 12, 2026, pp. 10 430–10 438.

[27] M. Zhang, H. Wu, R. Jin, D. Gund¨ uz, and K. Mikolajczyk, “Diffusion-¨ aided extreme video compression with lightweight semantics guidance,” in ICASSP, 2026, pp. 8422–8426.

[28] H. Schwarz, D. Marpe, and T. Wiegand, “Analysis of hierarchical b pictures and mctf,” in ICME, 2006, pp. 1929–1932.

[29] K. Wu, Z. Li, Y. Yang, Q. Liu, and X.-P. Zhang, “End-to-end deep video compression based on hierarchical temporal context learning,” TMM, vol. 27, pp. 4386–4399, 2025.

[30] R. Yang, F. Mentzer, L. V. Gool, and R. Timofte, “Learning for video compression with hierarchical quality and recurrent enhancement,” in CVPR, 2020, pp. 6628–6637.

[31] M.-J. Chen, Y.-H. Chen, and W.-H. Peng, “B-canf: Adaptive b-frame coding with conditional augmented normalizing flows,” TCSVT, vol. 34, no. 4, pp. 2908–2921, 2023.

[32] M. A. Yılmaz and A. M. Tekalp, “End-to-end rate-distortion optimized learned hierarchical bi-directional video compression,” TIP, vol. 31, pp. 974–983, 2021.

[33] D. Alexandre, H.-M. Hang, and W.-H. Peng, “Hierarchical b-frame video coding using two-layer canf without motion coding,” in CVPR, 2023, pp. 10 249–10 258.

[34] X. Sheng, L. Li, D. Liu, and S. Wang, “Bi-directional deep contextual video compression,” TMM, 2025.

[35] W. Jiang, J. Li, K. Zhang, and L. Zhang, “Biecvc: Gated diversification of bidirectional contexts for learned video compression,” in ACM MM, 2025, pp. 7248–7257.

[36] Y. Liu, S. Huo, J. Gu, C. Zhou, H. Bai, M. Lu, Z. Ma et al., “Neural bframe video compression with bi-directional reference harmonization,” NeurIPS, vol. 38, pp. 154 369–154 394, 2026.

[37] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in ICLR, 2023.

[38] Y. Zhang, S. Yang, R. Zhu, Y. Zhang, L. Cui, Y. Wang, B. Wang, F. Shi, B. Wang, W. Bi et al., “Gated slot attention for efficient linear-time sequence modeling,” NeurIPS, vol. 37, pp. 116 870–116 898, 2024.

[39] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” NeurIPS, vol. 30, 2017.

[40] M. Zaheer, G. Guruganesh, K. A. Dubey, J. Ainslie, C. Alberti, S. Ontanon, P. Pham, A. Ravula, Q. Wang, L. Yang et al., “Big bird: Transformers for longer sequences,” NeurIPS, vol. 33, pp. 17 283– 17 297, 2020.

[41] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in CVPR, 2018, pp. 586–595.

[42] Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit et al., “Vbench: Comprehensive benchmark suite for video generative models,” in CVPR, 2024, pp. 21 807–21 818.

[43] J. Wang, K. C. Chan, and C. C. Loy, “Exploring clip for assessing the look and feel of images,” in AAAI, vol. 37, no. 2, 2023, pp. 2555–2563.

[44] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in ICLR, 2019.

[45] A. Mercat, M. Viitanen, and J. Vanne, “Uvg dataset: 50/120fps 4k sequences for video codec analysis and development,” in ACM MMSys, 2020, pp. 297–302.

[46] H. Wang, W. Gan, S. Hu, J. Y. Lin, L. Jin, L. Song, P. Wang, I. Katsavounidis, A. Aaron, and C.-C. J. Kuo, “Mcl-jcv: a jnd-based h. 264/avc video quality assessment dataset,” in ICIP, 2016, pp. 1509– 1513.

[47] G. Bjontegaard, “Calculation of average psnr differences between rdcurves,” ITU SG16 Doc. VCEG-M33, 2001.

[48] Z. Wang, E. P. Simoncelli, and A. C. Bovik, “Multiscale structural similarity for image quality assessment,” in Asilomar conference on signals, systems & computers, vol. 2, 2003, pp. 1398–1402.

[49] K. Ding, K. Ma, S. Wang, and E. P. Simoncelli, “Image quality assessment: Unifying structure and texture similarity,” TPAMI, vol. 44, no. 5, pp. 2567–2581, 2020.

[50] D. Danier, F. Zhang, and D. Bull, “Flolpips: A bespoke video quality metric for frame interpolation,” in PCS, 2022, pp. 283–287.

[51] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” NeurIPS, vol. 30, 2017.

[52] F. Mentzer, G. D. Toderici, M. Tschannen, and E. Agustsson, “Highfidelity generative image compression,” NeurIPS, vol. 33, pp. 11 913– 11 924, 2020.

[53] D. He, Z. Yang, W. Peng, R. Ma, H. Qin, and Y. Wang, “Elic: Efficient learned image compression with unevenly grouped spacechannel contextual adaptive coding,” in CVPR, 2022, pp. 5718–5727.