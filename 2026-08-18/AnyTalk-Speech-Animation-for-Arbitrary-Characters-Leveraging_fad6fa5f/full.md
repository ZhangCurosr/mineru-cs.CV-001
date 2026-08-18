# AnyTalk: Speech Animation for Arbitrary Characters Leveraging a Video Generation Model

Kwan Yun\* , Serin Yoon\* , Sunjin Jung , Jung Eun Yoo , Inyup Lee , and Junyong Noh

![](images/cd7e27afd7bdd6d60b61bfcc029c85f102f54ddd5526b3c420950cedb831d549.jpg)

![](images/60a4b9c077155b967ef211133e72477b841d0b91c16a0d6514cf9158a6051fdf.jpg)  
Fig. 1. AnyTalk can perform audio-driven 3D facial animation using an arbitrary character with blendshapes, without requiring any animation data to train.

Abstract— We present AnyTalk, a novel method for generating 3D speech animations for arbitrary characters without requiring any animation data. While existing audio-driven 3D speech animation methods rely on character-specific training data or laborious rigging/re-meshing, AnyTalk circumvents these limitations by leveraging recent video diffusion models trained on extensive video datasets. We first adapt a pre-trained video diffusion model to a target character through our Character-specific Fine-tuning (CsF) technique. By fine-tuning on rendered images of the 3D character paired with zeroed-out audio embeddings (representing no motion), we eliminate the need for animation data while preserving the motion prior of large-scale video diffusion model. We then uplift the resulting talking-head video into a 3D speech animation by estimating blendshape parameters through a proposed optimization process. AnyTalk enables lip-synced animations across diverse face meshes and blendshape configurations, significantly reducing manual effort and data requirements. We further enhance usability by distilling AnyTalk into a streamlined network, AnyTalk , thereby enabling real-time performance. By leveraging talking-head video generation, our method broadens access to audio-driven speech animation technology for arbitrary characters. The code is publicly available at AnyTalk.

Index Terms—Audio-driven animation, facial animation, diffusion model.

## 1 INTRODUCTION

The increasing demand for realistic 3D speech animation across various media platforms such as games and VR/AR highlights the necessity of advanced techniques in the animation industry. Manually keyframing a character’s lip movements to synchronize with audio has been a popular choice. However, it is both time-consuming and requires significant expertise, making it impractical for nonprofessionals and cumbersome even for experienced artists.

To address this challenge, deep-learning-based audio-driven speech animation methods [7,12,20,43,61] have been proposed, substantially reducing the burden of manual keyframing. Nevertheless, applying these advanced techniques to real-world characters remains challeng ing. This is largely because most existing audio-driven 3D facial animation methods must be trained with paired audio3D data for each unique mesh structure or blendshape configuration. Consequently, rigging artists must either re-rig their characters to match existing blendshapes or gather training data for every mesh structure. This requirement can pose a significant barrier to the widespread adoption of audio-driven speech animation technique, because re-rigging characters or acquiring matched training data for each unique mesh structure is difficult for independent developers or small studios with limited resources.

![](images/4156a46dc25a5b8b720faceac669d170772e2d1cbafd60ac7a093d6be82e99cc.jpg)

Meanwhile, recent advancements in diffusion-based talking-head video generation models [8, 57, 60, 62, 63] have successfully produced realistic talking-head videos. This achievement is largely attributed to training large diffusion models [16] on extensive and diverse video datasets spanning hundreds to thousands of hours. As a result, these models can generate lip-synced videos from unseen characters, which has been difficult for most 3D speech animation methods. By utilizing these high-quality 2D outputs as an intermediate representation, our method effectively bridges the gap between the rich motion priors of 2D generative models and the practical requirements of 3D facial animation.

Building on this success, we propose AnyTalk, a novel method that leverages a video generation model to produce 3D speech animation for arbitrary avatars as shown in Figure 1. Our method follows a two-step process: (1) Talking-head Video Generation: Given a 3D character, we render an image of the character and generate a speechsynchronized talking-head video using a video generation model. (2) Blendshape Optimization: We estimate blendshape parameters from the generated video using an optimization process to create accurate 3D speech animation. This sequential approach utilizes the rich motion prior and generalizability of diffusion-based talking-head video generation models, enabling speech animation for arbitrary 3D characters.

However, directly applying pre-trained video generation models produces significant visual artifacts and poor character fidelity. Although the output videos synchronize well with the audio, they may not conform to the 3D character’s range of motion or correctly depict previously unseen facial attributes. This discrepancy between the generated video and the characters inherent motion capabilities not only compromises the visual fidelity but also complicates the downstream blendshape optimization, making it more difficult to produce precise 3D speech animation.

To overcome this visual mismatch, we propose Character-specific Fine-tuning (CsF) without requiring video data for audio-driven video generation models. Instead of relying on ground-truth audiovisual pairs, we use the rendered images of the character paired with zeroed-out audio embeddings as input. Because the audio input is the driving source of motion, zeroing it effectively pairs each rendered still image with no motion signal during training while allowing a nonzero speech signal at inference. To further preserve motion priors, we freeze the temporal module while updating the spatial module of the diffusion model, ensuring that the visual information of the generated video aligns well with the rendered images of the character while the trained motion prior is not modified.

To obtain blendshape parameters from the generated video, we propose a landmark matching optimization technique. Specifically, we first estimate landmarks from the rendered image, cast rays to identify the corresponding vertices, and then optimize the blendshape parameters so that these vertices align with the landmarks of the target video when rendered. In addition, during landmark-based optimization, we select talk invariant landmarks, estimate homography, and warp talkrelated landmarks accordingly to further enhance landmark alignment.

By integrating the proposed techniques, AnyTalk successfully produces 3D speech animation for arbitrary characters without requiring any animation data for training. In summary, our contributions are as follows:

• AnyTalk is the first 3D speech animation method for arbitrary characters that does not rely on 3D animation data for training. This is achieved by utilizing a 2D talking-head generation model and uplifting the generated video into 3D animation.

$C s F$ is a novel fine-tuning method for talking-head generation models that enables the generated video to match a specific character without requiring any video data.

• Our homography-based warping, driven by talk-invariant landmark selection, ensures precise alignment between the rendered images of arbitrary characters and the generated video. This alignment enables accurate estimation of blendshape parameters for the characters from generated video.

• We develop a real-time variant, $\mathbf { A n y T a l k } _ { R T } ,$ using model distillation with feature matching and reconstruction losses, achieving an inference speed of 110 FPS for real-time applications.

## 2 RELATED WORK

## 2.1 Audio-driven Talking-head Video Generation

Generating 2D talking-head videos is a long-standing and rapidly advancing field with wide usability [22, 33, 40, 44, 51, 56, 58, 59, 66, 76]. Synthesizing 2D facial animation from audio typically utilizes various image representations, such as facial landmarks [19, 67, 78],

3D Morphable Model parameters [71], depth maps [17], and semantic maps [31] to facilitate the transformation of speech into 2D facial animations. More recently, diffusion and flow based methods [8, 27, 46, 55, 57, 62, 63] have been proposed. These methods advanced the realism of audio-driven talking-head video generation by utilizing larger networks and more extensive data. Therefore, we finetune these models to make them applicable to a target 3D character for speech animation.

## 2.2 Audio-driven Speech Animation

Recently, significant efforts have been made in audio-driven 3D facial animation [1, 2, 10, 11, 14, 21, 23, 29, 37, 38, 43, 50, 53, 54, 79]. VOCA [7] approaches the transformation from audio to animation as a regression problem, using paired audio and 3D face animation data to train the regressor. FaceFormer [12] utilizes transformers to handle the long-term dependencies while CodeTalker [61] utilizes quantized latent space to create natural motion. More recently, diffusion-based methods [34, 46, 48, 49] have been proposed to add realism and diversity to the generated video. Another line of research extends from facial animation to generating both facial expressions and gestures guided by audio [72,73]. All the aforementioned methods were trained in a supervised manner, which requires paired audio-3D data.

## 2.3 Audio-driven Speech Animation for Diverse Characters

To address the large dataset requirement for each character, several methods have been proposed that focus on single-identity-based approaches. taylor et al. [52] proposed an audio-driven speech animation method with retargeting, assisted by transcription. Similarly, yang et al. [65] map single-identity audio and a single character into the same embedding space to generate a character motion in a semi-supervised manner. While both approaches require only audio-visual data for a single identity, they have limited applicability to different audio sources. For example, taylor et al. [52] requires manual specification of the correspondence for AAM shape components and transcriptions, making the process labor-intensive. On the other hand, yang et al. [65] is constrained to a single source identity and cannot generalize to other characters or audio sources.

Most recently, ScanTalk [36] was proposed, which utilizes large paired audio-scan data across various mesh structures. This system marks the first 3D talking-head generator capable of handling diverse mesh structures using the DiffusionNet [45] architecture. However, ScanTalk was trained on meshes that were manually aligned, resampled, and modified to match the properties of VOCASET [7]. As a result, the model’s performance varies significantly depending on how well a given mesh aligns with the training dataset, making it challenging to generate accurate speech animations for in-the-wild characters. In contrast, our approach generates 2D motion and uplifts it to 3D, without being constrained to any mesh structure or requiring mesh modifications.

## 2.4 Fine-tuning Video Diffusion Model

Fine-tuning video diffusion models for a specific character has become an active research area. Most previous personalization methods have either replaced the original layer with a personalized one [15,26] or masked out the temporal module and applied existing imagepersonalization techniques to the spatial module [35]. More recently, Still-Moving [3] proposed a method to fine-tune a general text-tovideo diffusion model using a binary motion signal. Most of these methods aim to personalize the video diffusion model for a specific character, but not for a specific scene. On the other hand, Any-MoLe [68] adapts the model to both the character and the rendering scene using only a few seconds of 3D character motion. This is achieved through rectification training of an image-to-video diffusion model for the motion in-betweening task, using a few seconds of ground-truth motion. In fact, no method has personalized an audiodriven video diffusion model without relying on paired data. Our method is the first to personalize an audio-driven talking-head animation model to a target character without using any motion or audio data, employing a zeroed-out audio embedding as a stop-moving signal to disentangle audio from motion.

![](images/b0ce2099da2bc353b4a8cdc11957ef93df0006371c55281c19afdf87f5434724.jpg)  
Fig. 2: Overview of baseline audio-driven video diffusion model $D _ { s r c } .$ Because $D _ { s r c }$ adopts $w _ { p o s e } , w _ { e x p }$ , and $\boldsymbol { w } _ { l i p }$ to control the degree of motion, AnyTalk is capable of generating dynamic lip movements while maintaining minimal head rotation.

## 3 METHOD

At its core, our method consists of two main stages. In the first stage, we generate videos using an audio-driven video diffusion model. In the second stage, we estimate blendshape parameters from the generated video frames through an optimization process. Because our motion generation process relies on blendshape estimation, the input character must be rigged. The first stage generates the target motion, while the second stage produces the 3D animation based on the target motion. Each stage will be described in detail in the following subsections.

## 3.1 Talking-head Video Generation Model

We adopt a pre-trained Hallo [62], a diffusion-based audio-driven video generation model which was trained on a large-scale human speech video dataset, as our baseline $D _ { s r c } . \ D _ { s r c }$ accepts a source image and driving audio as inputs and generates a corresponding video that matches both the source appearance and the audio content. A key feature of $D _ { s r c }$ is its hierarchical audio processing via three distinct modules: pose, expression, and lip residual attention. This design enables the use of control weights $w _ { p o s e } , w _ { e x p }$ , and $w _ { l i p } ,$ as shown in Figure 2, to modulate motion dynamics. This controllability allows head motions to remain stable while the lips move dynamically, thereby enabling a second stage of video-based blendshape optimization with improved stability.

## 3.2 Character-specific Fine-tuning without Speech Animation Data

Although recent audio-driven talking-head video generation models, including $D _ { s r c } ,$ can be applied to a wide range of identities and audio inputs, we observed a mismatch between the generated videos and the rendered images of the 3D character. This mismatch hinders a direct application of these models to our framework because blendshape parameter estimation for animation relies on comparing the rendered images of the character with the generated video. The mismatch typically manifests in two forms: (1) undesired head motion and (2) input inconsistency. These issues stem from biases in the training data of $D _ { s r c } ,$ which consists of natural human videoscaptured in real-world settings, unstylized, and exhibiting natural head motion, rather than rendered scenes. Figure 3 shows examples of these limitations suffered by recent methods, including Hallo [62], AniTalker [30], EchoMimic [4], and MEMO [75]. Even though each method can generate a face that aligns with the audio, all four models suffer from visual inconsistencies. Hallo generates unnatural clothing and additional chin that do not exist in the rendered character; AniTalker shifts the target character toward a generic human-like appearance; EchoMimic produces close-up faces or exhibits black spots; and MEMO generates characters in unnatural mouth and teeth.

![](images/9f89bd4ea8165f8d16cfa1f197fe3fd9ebeb396e28ff0ddb1d4957d250c908ad.jpg)  
Fig. 3: Domain gap between generated and rendered images. From an input image (left), a video is generated using existing talking-head generation models [4, 30, 62, 75] (middle). A clear mismatch can be observed between their appearances and the rendered images (right).

To address this, we introduce Character-specific Fine-tuning $( C s F )$ $C s F$ adapts a general talking-head video generation model $D _ { s r c }$ into a personalized model $D _ { C s F }$ using only a target character. To achieve this transformation, we activate each blendshape parameter individu ally and render a frontal image for each, then duplicate these images to create a still image video as shown in Figure 4. Here, R denotes the rendering function. Naively training the model to output a duplicated version of the input image given audio would cause it to forget how to animate effectively. For instance, if random speech is paired with still images during training, the model will produce a static video whenever it encounters similar audio, thereby losing its ability to animate. To prevent this, we zero out the audio encoder’s output (audio embedding) during training. This zeroed-out audio embedding simulates a no-motion input, disentangling the no-motion signal from the general speech signal (i.e., motion). As a result, the model reproduces the original image when no motion signal is present, while retaining its capability to generate speech-driven motion when a valid audio signal is provided.

Because the purpose of $C s F$ is to learn the spatial features of the target character while preserving the original audio understanding, motion prior, and target reference understanding, we freeze all attention layers, including audio attention, motion attention, and the layers of ReferenceNet [18]. Consequently, only the spatial residual network of denoising UNet is trained to accurately reproduce the visual appearance of the target character by predicting noise ϵ. Formally, the training process is expressed as follows:

$$
L = \mathbb { E } _ { \mathcal { E } ( I ) , C _ { z e r o } , \epsilon \sim \mathcal { N } ( 0 , 1 ) , t } \left[ \Vert \epsilon - D _ { C s F } ( z _ { t } , t , C _ { z e r o } ) \Vert _ { 2 } ^ { 2 } \right] .\tag{1}
$$

Here, t is uniformly sampled from diffusion timesteps $\{ 1 , . . . , T \}$ $C _ { z e r o }$ is zeroed-out audio embedding, and $z _ { t }$ is noisy latent variable. Note that the control weights $w _ { p o s e } , w _ { e x p }$ , and $w _ { l i p }$ are not applied during the training.

![](images/29e04f2ca7e86a0694f66f538affc4f634adc3a1c621345b99e956c04b3cacc2.jpg)  
Fig. 4: Finetuning process of $D _ { C s F }$ . The rendered images of the target 3D character and the zero condition $C _ { z e r o }$ are used to form a pair for fine-tuning the model. To preserve the motion prior, only the spatial residual network of denoising UNet is trained (flame icon), while the other layers remain frozen.

![](images/d93b8913b5fb109f492f400d0ca60f86ae85a1df7003a0f06fd672ae0366569b.jpg)  
Fig. 5: Inference of $D _ { C s F }$ . Non-zero speech passes through the audio encoder to provide motion information, while the latent representations a<sub>pose</sub>, $\begin{array} { r } { a _ { e x p } , } \end{array}$ and $a _ { l i p }$ are scaled by the corresponding factors w<sub>pose</sub>, $w _ { e x p } ,$ and $w _ { l i p }$ to achieve dynamic lip motion and a stable head pose.

## 3.3 Video Inference

With $D _ { C s F }$ , a video for the target character can be generated within the domain of the target character. While training was performed using a zero motion condition, we provide general (non-zero) speech audio for inference to generate dynamic motions. We re-weight these attention mechanisms with scaling factors $w _ { p o s e } = 0 , w _ { e x p } = 1$ , and $w _ { l i p }$ = 2 during inference as shown in Figure 5. This ensures that $D _ { C s F }$ produces speech with dynamic lip motion while maintaining a static head pose. This approach benefits the subsequent optimization process: the dynamic lip motion yields more natural animation, while the static head pose simplifies landmark-based optimization.

## 3.4 Optimization

From the previous steps, a target video that aligns with both the 3D character and the input audio can be generated. The purpose of optimization is to estimate the blendshape parameters, $\boldsymbol { B } ^ { \star } \in \mathbf { \bar { \mathbb { R } } } ^ { F \times \mathbf { N } }$ , that correctly reflect the motion of generated 2D video for final 3D speech animation. Here, $F$ is the number of frames and N is the number of blendshape parameters. The objective functions for the optimization consist of talk landmark loss, asymmetric mouth opening loss, and regularization loss. After the optimization, a Gaussian filter is applied to further improve the naturalness of speech animation by reducing temporal jittering. Additionally, to further enhance the realism of the speech animation, we can optionally add random eye blinks or facial expressions, as our method is based on blendshapes. The full optimization process is summarized in Figure 6.

![](images/6c8ca747c4ce063d623f87a8e90a74ee0f1b91523390f4a1b06067d2bc18eaa9.jpg)  
Fig. 6: Optimization process. Blendshape parameters are optimized so that the target 3D character accurately follows the generated video when rendered.

The talk landmark loss ensures that the talk-related landmarks of the 3D character align accurately with those in the generated 2D video. To automatically obtain the landmarks of the 3D character, we first render the neutral 3D character into an image $I _ { 0 } .$ . From this image, we extract 2D talk-related landmarks using a pre-trained landmark estimator [64] denoted by ϕ. For each 2D talk-related landmark, we identify the corresponding vertex on the character by casting a ray from each 2D landmark and selecting the nearest vertex to the intersection point. This process is performed only once with the neutral mesh, thus frame-by-frame instability is avoided. It should be noted, however, that large occluding components such as masks or highly stylized dental geometry may interfere with ray-casting. In such cases, these components are temporarily removed prior to vertex selection, or manual vertex assignment is performed. This gives the indices of the talk-related landmark vertices. Next, we compute the 2D locations of the talk-related landmark vertices for the deformed character created by blendshape parameters, $B _ { f } \in \mathbb { R } ^ { \mathbf { N } }$ , at frame f (where $1 \leq f \leq F )$ using blendshape matrix $M _ { b }$ and projection matrix P. This can be denoted as $P \left[ \left( \dot { M } _ { b } \cdot B _ { f } \right) _ { \mathrm { t a l k } } \right]$ . The talk landmark loss is then defined as the mean-squared error between these 2D talk landmark positions of the 3D character and the talk landmark positions in the f’th frame of the generated video $\hat { I } _ { f }$

Natural human motion often exhibits slight head movements such as small nods or shakes during speech. These movements can cause minor shifts in facial alignment, which need to be accounted for to maintain accurate landmark matching. Therefore, we compute this shift between $\hat { I } _ { f }$ and $I _ { 0 }$ using an image homography [42] derived from the estimated landmark positions, and apply the resulting transformation matrix H to the estimated landmarks in the videos before calculating the talk landmark loss. When computing the homography using landmarks, we include only those that do not vary with facial expressions. Thus, we filter out the expression-variant landmarks before obtaining H. A detailed explanation of this filtering is provided in Section 4.3. The objective for the talk landmark loss can be expressed as follows:

$$
L _ { t a l k } = \| P \left[ ( M _ { b } \cdot B _ { f } ) _ { t a l k } ) \right] - H ( \phi ( \hat { I } _ { f } ) _ { t a l k } ) \| _ { 2 } ^ { 2 } .\tag{2}
$$

One particular challenge faced by many speech-driven facial animation methods is the lack of high-frequency motion [61]. To address this, we adopt a new asymmetric mouth opening loss function that enforces alignment between the 3D character’s mouth opening and that of the subject in the video. Initially, we measure the distance between the center landmark among all upper-lip landmarks $( M _ { b } \cdot B _ { f } ) _ { u p }$ and the center landmark among all lower-lip landmarks $( M _ { b } \cdot B _ { f } ) _ { l o w }$ and adjust it to be similar to the corresponding distance observed in the video. This simple step ensures that the 3D characters range of mouth opening matches with that captured by the video. On top of this, to avoid smaller opening of the mouth than that of the video, we apply a weight $w _ { a s y m } \ ( w _ { a s y m } > 1 )$ . This can be expressed as follows:

$$
L _ { o p e n } = \left\{ \begin{array} { l l } { \| \mathbf { M O } \| _ { 2 } ^ { 2 } , } & { \mathrm { i f } \mathbf { M O } > 0 , } \\ { w _ { a s y m } \cdot \| \mathbf { M O } \| _ { 2 } ^ { 2 } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{3}
$$

$$
\begin{array} { r l } & { \mathrm { w h e r e ~ } \mathbf { M O } = P \Big [ ( M _ { b } \cdot B _ { f } ) _ { u p } - ( M _ { b } \cdot B _ { f } ) _ { l o w } \Big ] } \\ & { \quad \quad - \ H \Big ( \phi ( \hat { I } _ { f } ) _ { u p } - \phi ( \hat { I } _ { f } ) _ { l o w } \Big ) . } \end{array}
$$

Finally, regularization loss is applied to ensure that blendshapes unrelated to speech, such as those for eyebrows or ears, are not altered. The regularization can be expressed as follows:

$$
L _ { r e g } = \| B _ { f } \| _ { 1 } .\tag{4}
$$

The optimization process minimizes the loss function $L _ { o p t i m }$ to obtain the optimal blendshape parameters B∗. This can be written as follows:

$$
L _ { o p t i m } = \lambda _ { t a l k } L _ { t a l k } + \lambda _ { o p e n } L _ { o p e n } + \lambda _ { r e g } L _ { r e g , }\tag{5}
$$

$$
B _ { f } ^ { * } = \arg \operatorname* { m i n } _ { { B _ { f } } } \bigl ( L _ { o p t i m } ( { M } _ { b } \cdot { B } _ { f } ) \bigr ) .\tag{6}
$$

Here, each λ is a weighting factor that balances its corresponding loss term.

![](images/679b51f5daaa64ab6d6d3d631a231110736211070e016adcc526f8f86bfefcfe.jpg)  
Fig. 7: Lip and chin landmarks exhibit the high average displacement, suboptimal for robust homography estimation.

## 4 EXPERIMENTS

## 4.1 Implementation Detail

We implemented AnyTalk and conducted all training and inference on a computer with a single Nvidia A6000 GPU. For $\bar { C } s F$ , images were rendered in a frontal view with each blendshape activated. When a mouth related blendshape was activated, the images were duplicated four times. The learning rate was set to 1e-6, and the total training time was 20 minutes. We used 14 talk-related landmarks from the lips and chin for $L _ { t a l k }$ in Equation (2). The asymmetric weight w<sub>asym</sub> used in Equation (3) was set to 3. The weights $\lambda _ { t a l k } , \lambda _ { o p e n } .$ , and $\lambda _ { r e g }$ used in Equation (5) for optimization were set to 100,000, 8,000, and 10, respectively. Optimization was conducted for 200 iterations, with learning rate of 5e-3.

## 4.2 Dataset

The only requirement of our method is to provide a target character. To conduct experiments, we gathered five different target characters: Morphy(©joshburton.com), Malcolm(©Animschool), Victor (©Faceware Technologies, Inc.), Emily, and VMan. Each character has a different configuration in terms of the number of vertices, the number of constituent meshes, blendshape parameters, mesh type, and style as shown in Table 1. Notably, our test suite covers a broad spectrum of visual styles, ranging from highly stylized characters like Malcolm to photorealistic human avatars such as Victor and Emily. To generate speech animation, we randomly sampled audio clips from the LibriSpeech [39] dataset and used them for each of the five characters.

## 4.3 Landmark Filtering

We conducted an experiment to select the expression-invariant landmarks to be used for estimating the homography between the generated 2D video and the rendered images of the 3D character. In this experiment, we used all five characters and activated each blendshape to generate faces with different expressions. For each landmark, we calculated the average displacement across the various blendshape parameters. This analysis, as shown in Figure 7, allowed us to distinguish between stable landmarks and those prone to variation. We observed that the landmarks corresponding to the lower eyes, nose, and sides of the head exhibited relatively small shifts, whereas the landmarks on the lips and chin showed the most significant changes. Based on these findings, we decided to filter out the top half of expression-variant landmarks during homography estimation to ensure robust and accurate landmark matching in subsequent processing. Although filtering additional landmarks resulted in a lower average variance, using too few landmarks for homography estimation compromised robustness.

## 4.4 Results

We demonstrate our 3D speech animation results on arbitrary characters driven by a provided audio track, spanning a wide range of mesh structures and artistic styles. To evaluate lip synchronization quality, we present seven key frames rendered from a facial animation that aligned with specific phonemes in Figure 8. For visualization, each phoneme or letter and its corresponding frame are connected with a dotted line. The results indicate that the generated lip movements closely follow the spoken phonemes.

Tab. 1: Character configurations for experiments. Each character has different number of vertices, number of meshes, number of blendshape parameters, mesh type, and style.
<table><tr><td>Character</td><td>Morphy <img src="images/c493719852e92f28a6423a7e8207886ec26cc0756279f80062e93e55b0f68f2a.jpg"/></td><td>Malcolm <img src="images/f3ca179867b4421cc8168ae8e48ceacb8e6e231161dd89334c720e0bd9802498.jpg"/></td><td>Victor <img src="images/e8fe8b18b5fd5b3b508410cb42addbb6c3cb4e33de419ab8f2324d969790c4cd.jpg"/></td><td>Emily <img src="images/67fe265ac00561159bd58ed43754ca0a5204cf214991b10e08e92751e0f31360.jpg"/></td><td>VMan <img src="images/3fe8e1c3518e787df76579aceb3fb7d5e9e5014e9094fa57750e20dc87d9314c.jpg"/></td></tr><tr><td># Vertex</td><td>4,862</td><td>4,542</td><td>20,104</td><td>33,966</td><td>241,981</td></tr><tr><td>#Mesh</td><td>7</td><td>4</td><td>1</td><td>5</td><td>5</td></tr><tr><td># Param</td><td>46</td><td>32</td><td>45</td><td>121</td><td>101</td></tr><tr><td>Mesh Type</td><td>Triangular/Quad</td><td>Triangular/ Quad/N-Gon</td><td>Triangular</td><td>Triangular</td><td>Triangular/Quad</td></tr><tr><td>Stylized</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td><td>No</td></tr></table>

Tab. 2: Quantitative results comparing our method with baselines. Best denoted in bold.
<table><tr><td>Methods</td><td>LSE-D↓</td><td>LSE-C↑</td></tr><tr><td>Ours</td><td>11.304</td><td>3.155</td></tr><tr><td>ScanTalk</td><td>12.152</td><td>2.395</td></tr><tr><td>DiffSpeaker + NFR</td><td>13.857</td><td>0.665</td></tr><tr><td> $\mathrm { C o d e T a l k e r + N F R }$ </td><td>13.840</td><td>0.668</td></tr></table>

We also visualize both the 2D video generated by $D _ { C s F }$ and the optimized final animation results. As shown in Figure 9, due to the use of $C s F$ that personalize video generation model, the generated video accurately followed the target character driven by the input audio, without visual mismatches. Furthermore, the alignment and optimization process ensures that the final lip motion precisely matches the generated video corresponding to the phonemes of the source audio.

## 4.5 Baseline Comparison

We compared our approach with ScanTalk [36], DiffSpeaker [34], and CodeTalker [61], using all 5 characters and 30 audio clips, resulting in a total of 150 speech animations per method. ScanTalk is our closest competitor because it is the only method capable of generating 3D speech animations for diverse mesh structures. DiffSpeaker is a diffusion-based 3D speech animation method that does not require additional information (such as a style video) during inference. In addition, CodeTalker is a 3D speech animation method that uses VQ-VAE latent for autoregressive motion generation. Because DiffSpeaker and CodeTalker can only work with a specific mesh structure on which it was trained, we first generated outputs using the VOCASET mesh structure and subsequently retargeted them with NFR [41], a state-ofthe-art neural retargeting method. Both ScanTalk and NFR require the mesh to be scaled, aligned, and stripped of unnecessary components (e.g., accessories or hair) prior to inference, accordingly, we followed their guidelines for inference. For comparison, we rescaled, realigned, and restored any removed parts to render the meshes in their original form. Additional comparison results with an image-based baseline [5] that could not generate motion are presented in Section 2.3 of the supplementary material. It is important to note that our method is personalized to each character, whereas the baseline methods are not. Thus, we also evaluated our approach in a non-personalized setting and reported the results in Section 2.4.2 of the supplementary material.

The results of the qualitative comparison are presented in Figure 10. The rendering settings were the same for all methods. To evaluate lip synchronization, we present nine specific frames from the synthesized facial animations, similar to those in Section 4.4. As shown in the figure, the lip movements created by our method more precisely follow the given sentence with dynamic motion. In contrast, ScanTalk, DiffSpeaker + NFR, and CodeTalker + NFR exhibited only subtle movements that did not faithfully follow the intended phonemes. This discrepancy may be attributed to differences in the characters on which ScanTalk and NFR were trained. In contrast, our method successfully handled meshes with varying numbers of vertices or stylized characters, without requiring any additional processing.

Because no ground-truth pairs of audio and 3D animations are available for quantitative comparison, we evaluated lip synchronization using the Lip Sync Error Distance (LSE-D) and Lip Sync Error Confidence (LSE-C) metrics [40], which do not require ground-truth data. These metrics can be expressed as follows:

$$
d _ { n } ( k ) = \left\| v _ { n } - a _ { n + k } \right\| _ { 2 } , \quad { \mathrm { L S E - D } } = { \frac { 1 } { N } } \sum _ { n = 1 } ^ { N } \operatorname* { m i n } _ { k } d _ { n } ( k ) ,
$$

$$
\mathrm { L S E \mathrm { - } C } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \Bigl ( \mathrm { m e d i a n } _ { k } d _ { n } ( k ) - \frac { } { { k } } \operatorname * { m i n } _ { k } d _ { n } ( k ) \Bigr ) .
$$

Here, $v _ { n }$ is the SyncNet [6] video embedding for the n-th clip, $\boldsymbol a _ { n + k }$ is the SyncNet audio embedding, and $d _ { n } ( k )$ is their Euclidean distance. As shown in Table 2, ours achieved the best results on both metrics, while ScanTalk achieved the second best performance. DiffSpeaker + NRF and CodeTalker + NFR produced very similar results both qualitatively and quantitatively, likely due to the use of same retargeting.

## 4.6 Ablation Study

We conducted a series of ablation studies to validate the proposed video generation and optimization methods by modifying each component. We used the same audio as in Section 4.5 and employed two characters, VMan and Malcolm. Our Ablation studies mainly consists of two parts. Firstly, ablating fine-tuning process for personalized video generation, secondly, ablating optimization process for final speech animation.

On the video generation process, we compared ours with three variations. First, we ablated $\overline { { C } } s F .$ , while retaining $w _ { p o s e } , \ w _ { e x p } ,$ and $w _ { m o u t h } .$ as described in Section 3.3. Second, we ablated $C _ { z e r o }$ , and instead used random audio to match with the duplicated images of the character for $C s F$ . Third, we compared ours with an alternative that did not freeze the spatial, temporal, and cross attention modules when fine-tuning. For the optimization process, we compared with another three variations. First, we ablated the loss term $L _ { t a l k }$ . Second, we evaluated landmark filtering for homography estimation by comparing ours with an approach that uses all 68 facial landmarks without

In both these high mythical subjects the surrounding nature, though suffering, is still dignified and beautiful.

![](images/307f9e3b19684c4abf32c67ebd305e657c5fc726db4290a5fee3f08e7461c6ff.jpg)

Some of the penal regulations of this edict were copied from the edicts of Diocletian and this method of conversion was applauded by the same bishops who had felt the hand of oppression, and had pleaded for the rights of humanity  
![](images/876a72038bf484eb08b41b43867e501f2c07386a7922bda6b736cb9c6e8690ad.jpg)  
Fig. 8: Qualitative results of AnyTalk using two different audio sources. The top two rows shows results of Emily and Malcolm, while bottom two rows shows result of Victor and Morphy.

![](images/061c6993fe16e8945fcdf7d867f052919f54eaf019a29bf6cf8a3afd7d15c35c.jpg)  
Fig. 9: From input image (left) and audio, 2D video is generated using $D _ { C s F }$ (middle). From this video, final 3D speech animation is optimized (right).

filtering. Third, we tested an additional filtering strategy that uses only the 10 landmarks with the lowest average distance.

## 4.6.1 Ablation Study on Video Generation

First, we compared variants of the 2D video generation process. Because video generation is the first stage of our method, accurately producing a talkinghead video that does not deviate from the target character while following input audio is crucial for achieving the final speech animation. As shown in Figure 11, ours generates the lip shapes correctly and without artifacts. In contrast, the variant without CsF produced inconsistent appearances, such as squeezed pupils and double chins. The variant without $C _ { \mathrm { z e r o } }$ yielded less expressive lip motion and failed to open the mouth properly because its training did not correctly disentangle the nomotion signal from the speech signal. Finally, the variant without freezing modules exhibits the poorest performance: it produced minimal motion with no mouth opening, likely due to overfitting of the attention modules.

## 4.6.2 Ablation Study on Final 3D Speech Animation

The ablation study results for 3D speech animation are presented in Figure 12. In the variant without $C s F$ , lip movements did not match the given audio due to mismatches during video generation, whereas the variants without $C _ { z e r o }$ and without freezing modules failed to open the mouth sufficiently, owing to the entanglement of motion signals and overfitting in the motion module. The variant without $L _ { t a l k }$ exhibited artifacts on the chin because blendshapes were not correctly estimated in the absence of talk-related landmark optimization, relying solely on a sparse mouth-opening loss. In the filtering-process ablation, the variant without landmark filtering barely opened the mouth for the phoneme be, whereas adding the filtering step yielded incorrect blendshapes due to homography estimation errors showing that 10 landmarks are not sufficient to estimate homography matrix robustly. In contrast, ours produced results that closely followed the spoken phonemes without artifacts and correctly opened the mouth. This superiority is also verified by the quantitative results reported in Table 3.

![](images/638a57ca47784280944e28431b35e6b10d33bf3a16b97084c8576ca0afe63bdc.jpg)  
Fig. 10: Comparison with baselines on Morphy. Each phoneme and its corresponding frame are presented for comparison. Ours best follows the given audio with wider mouth opening and precise lip closure, while ScanTalk, DiffSpeaker + NFR, and CodeTalker + NFR exhibit only subtle movement.

![](images/31557ef7d523633199c5c52b0a00a9f8fbc7e2605973fbf68f9637058ce9bc05.jpg)  
Fig. 11: Visual results of generated video. Ours correctly generated the lip shape without artifacts. On the other hand, w/o $C s F$ made double chin, pinched eye, which does not match with the source character. Variants w/o $C _ { z e r o }$ and w/o freezing module produced small and subtle lip motion.

Ours achieved the best scores for LSE-D and the second best score for

LSE-C. Without CsF achieved the best score for LSE-C but did not perform well for LSE-D.

Photometric Optimization. In addition to the ablation study of each proposed component, we evaluated two additional variants to further investigate whether geometric optimization through 3D landmark supervision is indeed the optimal choice for speech-driven facial animation. We defined a photometric loss $L _ { p h o t o }$ based on LPIPS [70] to capture the perceptual similarity between the rendered character and the generated video. First, we replaced all 3D landmark-based losses $( L _ { t a l k }$ and $L _ { o p e n } )$ with $L _ { p h o t o }$ . When solely using $L _ { p h o t o }$ to encourage the rendered character to match the generated video, the results exhibited no or only subtle movement. Second, we added $L _ { p h o t o }$ to our original loss terms. This combined variant produced results similar to ours, with slight degradation in both quantitative and qualitative metrics, as shown in Table 3 and Figure 12. Moreover, because incorporating the photometric loss requires rendering the character with a differentiable renderer at every iteration, the animation generation time increased by a factor of 2.96 (9.32 s vs. 3.12 s per frame). These results confirm that photometric loss is a highly ambiguous objective for blendshape optimization, offering no additional performance gain and significantly slowing down the optimization process.

Effect of Fine-tuning Strategies for $C s F .$ To substantiate the design of CsF and evaluate its sensitivity to different fine-tuning configurations, we conduct a controlled study varying which UNet blocks are frozen or trained. Our hypothesis is that fine-tuning purely spatial layers on static images while zeroing the audio embedding and explicitly freezing temporal and audio-attention modules enables highfidelity appearance transfer without overwriting the pre-trained motion priors. To quantify this balance between motion prior retention and appearance transfer, we compare our proposed setting against three variants: fine-tuning with audio-attention unfrozen, fine-tuning with motion-attention unfrozen, and fine-tuning all UNet layers (w/o freezing). As shown in Table 4, exposing the audio attention leads to degraded performance which may be due to overfitting zeroed out audio embedding. When exposing the motion-attention modules, the degradation becomes even more pronounced. This is likely because training temporal layers on static data forces the motion priors to collapse, as the model erroneously learns to minimize frame-to-frame variance. Fine-tuning all layers results in the most significant drop in performance, demonstrating catastrophic forgetting of the audio-driven dynamics. In contrast, by restricting updates strictly to the spatial residual blocks, our strategy successfully injects character-specific appearance features while safely disentangling them from the model’s foundational temporal and audio-sync priors.

![](images/a6978d921d1862efda52e291faba3b36ba6d3727f4dc2130e9fbd4ba9b95fbaf.jpg)  
Fig. 12: Results of the ablation study on 3D speech animation. Words corresponding to the rendered images of Malcolm and VMan are presented on the left. While our method correctly produced the lip movements corresponing to the given words, the alternatives produced either only subtle lip movements or artifacts.

Tab. 3: Quantitative results from the ablation study. The best and the second best results are denoted in bold and underlined.
<table><tr><td>Methods</td><td>LSE-D↓</td><td>LSE-C↑</td></tr><tr><td>Ours</td><td>10.695</td><td>3.397</td></tr><tr><td>w/o CsF</td><td>11.207</td><td>3.451</td></tr><tr><td>w/o  $C _ { z e r o }$ </td><td>11.203</td><td>2.770</td></tr><tr><td>w/o freezing module</td><td>14.237</td><td>1.228</td></tr><tr><td>w/o  $L _ { t a l k }$ </td><td>11.044</td><td>2.922</td></tr><tr><td>w/o landmark filtering</td><td>10.777</td><td>3.019</td></tr><tr><td>w/ additional filtering</td><td>12.397</td><td>2.352</td></tr><tr><td>replaced w/  $L _ { p h o t o }$ </td><td>15.154</td><td>0.293</td></tr><tr><td>w/  $L _ { p h o t o }$ </td><td>10.813</td><td>3.364</td></tr></table>

Tab. 4: Quantitative results for fine-tuning UNet blocks. The best and the second best results are denoted in bold and underlined, respectively.
<table><tr><td>Methods</td><td>LSE-D↓ LSE-C↑</td></tr><tr><td>Ours 10.695</td><td>3.397</td></tr><tr><td>w/ audio-attention tuning</td><td>10.796 3.307</td></tr><tr><td>w/ motion-attention tuning 10.932</td><td>3.149</td></tr><tr><td>w/o freezing module 14.237</td><td>1.228</td></tr></table>

Effect of Head Pose Stabilization. While natural head movements are typically desirable for 2D speech animation, our framework lifts the video into a 3D representation via optimization. In this context, significant head motion is detrimental to the stability and accuracy of the 3D reconstruction. To evaluate this, we conducted an additional ablation study varying the pose weight $w _ { p o s e }$ , which controls the degree of stabilization. As shown in Figure 13, larger values of $w _ { p o s e }$ (indicating more head motion) lead to a degradation in lip-sync quality, as reflected by the LSE-D and LSE-C. This suggests that the model struggles to accurately decouple fine facial deformations from global rigid transformations when large head motion exists. It is important to note that even when $w _ { p o s e }$ is high, the final 3D-optimized output does not exhibit head motion due to the constraints of the subsequent optimization process. However, enforcing stabilization during the initial generation phase (where $w _ { p o s e } = 0 )$ provides a cleaner source for the 3D uplifting process, ultimately yielding the highest motion fidelity.

![](images/b884f3bca372f0811d398261280b9f005b88e316b9d434be7db674c1042b1eb1.jpg)  
Fig. 13: Quantitative results for head pose weights. Lip-sync perfor mance degrades as $w _ { p o s e }$ increase.

## 4.7 User Study

We conducted a user study to compare the perceptual quality of animations produced by our method against two baselines. Participants answered two questions regarding perceptual naturalness and lip synchronization. A total of 21 participants (10 males, 11 females), aged 23 to 35, took part in the web-based study, evaluating our method, ScanTalk [36], DiffSpeaker + NFR [34, 41], and CodeTalker + NFR [41, 61]. Each participant was presented with 75 side-by-side video pairs in randomized order using a two-alternative forced-choice format. The results are presented in Table 5. We applied exact binomial tests against a 50% chance level to assess the significance of these preferences. For perceptual naturalness, participants chose our method over ScanTalk in 413 of 525 comparisons (78.6%), a highly significant result with p < 0.001 in the binomial test. For lip synchronization, our animations were preferred in 403 of 525 trials (76.8%), also with p < 0.001. Comparisons with DiffSpeaker + NFR and CodeTalker + NFR also yielded significant improvements for all metrics with $\mathrm { p } < 0 . 0 0 1$ These results confirm that our approach produces perceptually more natural and better lip-synced speech animations than ScanTalk, DiffSpeaker + NFR, and CodeTalker + NFR.

Tab. 5: Selection ratio of our method compared to the baselines.
<table><tr><td>Methods</td><td>Naturalness (%)</td><td>Lip-sync(%)</td></tr><tr><td>Ours vs. ScanTalk</td><td>78.6</td><td>76.8</td></tr><tr><td>Ours vs. DiffSpeaker + NFR</td><td>99.6</td><td>99.4</td></tr><tr><td>Ours vs. CodeTalker + NFR</td><td>99.8</td><td>99.8</td></tr></table>

![](images/ffeaf1e013f55d2aa977936551636a047ac5de7013abdc84a001e1ea4cb9bae9.jpg)  
Fig. 14: Network architecture for AnyTal $\dot { } _ { \cdot R T }$

## 5 APPLICATIONS

## 5.1 Distillation

To enable real-time performance for applications, we distilled AnyTalk into a streamlined network, $\mathrm { A n y T a l k } _ { R T }$ . The network architecture is presented in Figure 14. For training $\mathbf { A n y T a l k } _ { R T } ,$ we first generated around 1,600 speech animations using AnyTalk. To effectively distill the optimization based model AnyTalk into the learning based model $\mathbf { A n y T a l k } _ { R T } ,$ we applied a feature matching loss $\scriptstyle { L _ { f e a t } }$ and a blendshape reconstruction loss $L _ { r e c o n }$ as follows:

$$
\begin{array} { r l } & { L _ { d i s t i l l } = L _ { f e a t } + \lambda L _ { r e c o n } , \quad \mathrm { w h e r e } } \\ & { L _ { f e a t } = F _ { a u d i o } - H ( \phi ( \hat { I } _ { f } ) _ { t a l k } ) } \\ & { L _ { r e c o n } = B _ { R T } - B _ { f } . } \end{array}\tag{7}
$$

Here, $\ b { L } _ { f e a t }$ enforces the audio feature of $\mathbf { A n y T a l k } _ { R T }$ to match with the warped talk-releated landmarks, while $L _ { r e c o n }$ enforces the blendshape. The overall distillation process is illustrated in Figure 15. λ is a weighting factor that was set to 400. Training was conducted for 360 epochs using the adamW optimizer [32], with a batch size of 128, and the learning rate was increased up to 0.008 and decreased using OneCyleLR [47].

With the distilled $\mathbf { A n y T a l k } _ { R T } ,$ an arbitrary character can be animated in real-time given audio as shown in the Figure 16. We measured the inference time and lip-sync quality of AnyTalk $\cdot _ { R T }$ using Morphy and report the results in Table 6. Due to the simplified inference, which excludes the optimization phase, and adopts the streamlined network, $\mathbf { A n y T a l k } _ { R T }$ achieved 9.09 ms per frame (110 frames per second) in a full-precision (32-bit) PyTorch setting.

## 5.2 Extension to Different Video Generation Model

To demonstrate the generality of our $C s F$ and optimization pipeline, we applied the same procedure to MEMO [75], hereafter referred to as $\mathbf { A n y T a l k } _ { M E M O }$ . MEMO is built with a disentangled architecture that explicitly separates spatial and temporal processing similar to Hallo. Following the protocol of Section 3.2, we freeze all of MEMOs modules except its spatial residual layer in the denoising U-Net to apply $C s F .$ . We render the target 3D character under each active blendshape with zeroed-out audio embeddings to create a no-motion training set, which we use to fine-tune MEMO to obtain $\mathbf { M E M O } _ { C s F }$ . We then perform our landmark-based blendshape optimization (Section 3.4) on the $\mathbf { M E M O } _ { C s F }$ outputs to generate the 3D speech animation, thereby yielding AnyTalk . Qualitative results are shown in Figure 17, demonstrating that lip synchronization aligns with the given phoneme. This extension confirms that our method, which comprises zeroedout audio fine-tuning of spatial modules and subsequent landmarkdriven blendshape optimization, can be straightforwardly integrated into different diffusion-based audio-driven talking-head video generation models.

![](images/271f8ee567a5044f762c75b8da995f60a148e4976ddf1385c90a95056f1ebec1.jpg)  
Fig. 15: Distillation process from AnyTalk to $\mathsf { A n y T a l k } _ { R T } .$ . Audio2Feat extracts features from the input audio, while Feat2BS predicts blendshape parameters from the extracted features.

Tab. 6: Quantitative results for $\mathsf { A n y T a l k } _ { R T }$ compared to AnyTalk. Although distillation enabled real-time performance, the lip-sync metric degraded.
<table><tr><td>Methods</td><td>Time (per frame)</td><td>LSE-D↓</td><td>LSE-C↑</td></tr><tr><td> $\mathrm { A n y T a l k } _ { R T }$ </td><td>9.09 ms</td><td>12.19</td><td>2.96</td></tr><tr><td>AnyTalk</td><td>3.12 s</td><td>11.74</td><td>3.24</td></tr></table>

## 6 CONCLUSION

In this study, we presented an audio-driven speech animation method for arbitrary avatars, which does not require animation dataa critical advancement for real-world adoption that has not been previously explored. While previous speech-driven facial animation methods have primarily focused on supervised, character-specific scenarios, accumulating advances in extensive paired-data setting, our paper pioneers a fundamentally new direction: animation-data-free speech animation for arbitrary characters. We narrow the domain gap between the generated video and rendered images of the character using $C s F$ and zeroed-out audio embedding. Additionally, we proposed an optimization method that employs talk-invariant landmark based face alignment with landmark matching to create faithful speech animations. While prioritizing 3D geometric stability and view-independence may lead to less pronounced lip movements compared to per-frame 2Dcentric models, it ensures the structural consistency required for robust 3D lifting. We believe that our method paves a way for animation-datafree audio-driven 3D animation, enabling its application to arbitrary characters regardless of mesh structure.

Limitations and Future Work. While our method produces promising results and enables significant advancements in speech animation for arbitrary characters, it also has challenges to address. Our method requires optimization during inference, which takes around

![](images/d44962c67904bd35f9f611a217b25276bca5ac31548b7e1e734cda9383597a6a.jpg)  
Fig. 16: Results of AnyTalk and $\mathsf { A n y T a l k } _ { R T : }$ , correctly following the phoneme of the given audio and depicting similar speech motion.

![](images/d762c11bf1ab3a9442203935a49e5fd2f1a8d039dc02bb0caf37198d6ff7e6c3.jpg)  
Fig. 17: Qualitative results of ${ \mathsf { M E M O } } _ { \mathsf { C s F } }$ and AnyTalk<sub>MEMO</sub>, showing the generalability of AnyTalk Framework.

3.12 seconds for a frame. While we demonstrated the potential for real-time performance through distillation, there is a trade-off between generation quality and inference time. Addressing this trade-off would be crucial for future research. A second limitation is the reliance on pre-defined blendshape parameters. Because our optimization process focuses on blendshape estimation, the input character must be rigged. Consequently, if a model is poorly rigged, for instance, lacking a blendshape for mouth opening, the animation may fail. Transitioning toward direct mesh animation could be a promising direction to address these constraints. Further more, our reliance on 2D landmarks may limit the capture of subtle facial details and high-frequency motions due to their sparse nature. While our experiments confirmed that incorporating dense photometric loss to address this introduces significant optimization ambiguity and overhead without qualitative gains, a possible future direction is to integrate hybrid dense-sparse representations, such as localized neural displacement maps, to recover these fine-grained dynamics.

Another limitation is its dependence on frontal views. Although our final output is 3D speech animation, the optimization process relies on 3D landmarks. While landmarks have been widely adopted as essential part for 3D face capture and animation [9, 13, 28, 79], they still lack the expressiveness needed for rich detail. Simple attempts to incorporate multi-view informationsuch as using front, left, and right images for video generation or reconstructing the output with an offthe-shelf method [74] failed to resolve this issue (see Section 3 of the supplementary material). Generating separate videos from different viewpoints produces inconsistent outputs due to the stochastic nature of diffusion, and off-the-shelf reconstruction methods do not produce 3D models that exactly match the target character. To overcome this single-view dependency, future research could explore the development of multi-view facial animation models [24, 25, 69, 77] that (1) control head and lip motion distinctly; (2) allow spatial layers to be tuned while freezing temporal layers independently; and (3) match the quality of state-of-the-art 2D video generation models. Developing such a method would preserve the animation quality of AnyTalk while enhancing multi-view consistency for the optimization process.

## REFERENCES

[1] S. Aneja, J. Thies, A. Dai, and M. Nießner. Facetalk: Audio-driven motion diffusion for neural parametric head models. arXiv preprint arXiv:2312.08459, 2023. 2

[2] Y. Chai, T. Shao, Y. Weng, and K. Zhou. Personalized audio-driven 3d facial animation via style-content disentanglement. IEEE Transactions on Visualization and Computer Graphics, 30(3):1803–1820, 2022. 2

[3] H. Chefer, S. Zada, R. Paiss, A. Ephrat, O. Tov, M. Rubinstein, L. Wolf, T. Dekel, T. Michaeli, and I. Mosseri. Still-moving: Customized video generation without customized video data. ACM Transactions on Graphics (TOG), 43(6):1–11, 2024. 2

[4] Z. Chen, J. Cao, Z. Chen, Y. Li, and C. Ma. Echomimic: Lifelike audiodriven portrait animations through editable landmark conditions. In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, pp. 2403–2410, 2025. 3

[5] Y. Choi, I. Lee, S. Cha, S. Kim, S. Jung, and J. Noh. Deep-learning-based facial retargeting using local patches. In Computer Graphics Forum, p. e15263. Wiley Online Library, 2024. 6

[6] J. S. Chung and A. Zisserman. Out of time: automated lip sync in the wild. In Computer Vision–ACCV 2016 Workshops: ACCV 2016 International Workshops, Taipei, Taiwan, November 20-24, 2016, Revised Selected Pa pers, Part II 13, pp. 251–263. Springer, 2017. 6

[7] D. Cudeiro, T. Bolkart, C. Laidlaw, A. Ranjan, and M. J. Black. Capture, learning, and synthesis of 3d speaking styles. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10101–10111, 2019. 1, 2

[8] J. Cui, H. Li, Y. Yao, H. Zhu, H. Shang, K. Cheng, H. Zhou, S. Zhu, and J. Wang. Hallo2: Long-duration and high-resolution audio-driven portrait image animation. In The Thirteenth International Conference on Learning Representations, 2025. 1, 2

[9] R. Daneˇcek, M. J. Black, and T. Bolkart. Emoca: Emotion driven monoc-ˇ ular face capture and animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20311–20322, 2022. 11

[10] R. Daneˇcek, K. Chhatre, S. Tripathi, Y. Wen, M. Black, and T. Bolkart.ˇ Emotional speech-driven animation with content-emotion disentanglement. In SIGGRAPH Asia 2023 Conference Papers, pp. 1–13, 2023. 2

[11] X. Fan, J. Li, Z. Lin, W. Xiao, and L. Yang. Unitalker: Scaling up audiodriven 3d facial animation through a unified model. In European Conference on Computer Vision, pp. 204–221. Springer, 2024. 2

[12] Y. Fan, Z. Lin, J. Saito, W. Wang, and T. Komura. Faceformer: Speechdriven 3d facial animation with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18770–18780, 2022. 1, 2

[13] Y. Feng, H. Feng, M. J. Black, and T. Bolkart. Learning an animatable detailed 3d face model from in-the-wild images. ACM Transactions on Graphics (ToG), 40(4):1–13, 2021. 11

[14] C. Gu, S. Kuriyama, and K. Hotta. Diverse code query learning for speech-driven facial animation. IEEE Transactions on Visualization and Computer Graphics, 2025. 2

[15] Y. Guo, C. Yang, A. Rao, Z. Liang, Y. Wang, Y. Qiao, M. Agrawala, D. Lin, and B. Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2

[16] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1

[17] F.-T. Hong, L. Zhang, L. Shen, and D. Xu. Depth-aware generative adversarial network for talking head video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3397–3406, 2022. 2

[18] L. Hu. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8153–8163, 2024. 3

[19] X. Ji, H. Zhou, K. Wang, W. Wu, C. C. Loy, X. Cao, and F. Xu. Audiodriven emotional video portraits. In CVPR, 2020. 2

[20] S. Jung, Y. Seol, K. Seo, H. Na, S. Kim, V. Tan, and J. Noh. Speed-aware audio-driven speech animation using adaptive windows. ACM Transac-

tions on Graphics, 44(1):1–14, 2024. 1

[21] T. Karras, T. Aila, S. Laine, A. Herva, and J. Lehtinen. Audio-driven facial animation by joint end-to-end learning of pose and emotion. ACM TOG, 36(4):1–12, 2017. 2

[22] Z. Kong, F. Gao, Y. Zhang, Z. Kang, X. Wei, X. Cai, G. Chen, and W. Luo. Let them talk: Audio-driven multi-person conversational video generation. arXiv preprint arXiv:2505.22647, 2025. 2

[23] H. Li, J. Dai, X. Zhao, F. Zhou, J. Pan, and L. Li. Wav2sem: Plug-andplay audio semantic decoupling for 3d speech-driven facial animation. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pp. 183–192, 2025. 2

[24] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, and L. Gu. Talkinggaussian: Structure-persistent 3d talking head synthesis via gaussian splatting. In European Conference on Computer Vision, pp. 127–145. Springer, 2024. 11

[25] J. Li, J. Zhang, X. Bai, J. Zhou, and L. Gu. Efficient region-aware neural radiance fields for high-fidelity talking portrait synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7568–7578, 2023. 11

[26] J. H. Liew, H. Yan, J. Zhang, Z. Xu, and J. Feng. Magicedit: High-fidelity and temporally coherent video editing. arXiv preprint arXiv:2308.14749, 2023. 2

[27] G. Lin, J. Jiang, J. Yang, Z. Zheng, and C. Liang. Omnihuman-1: Rethinking the scaling-up of one-stage conditioned human animation models. arXiv preprint arXiv:2502.01061, 2025. 2

[28] J. Ling, Z. Wang, M. Lu, Q. Wang, C. Qian, and F. Xu. Semantically disentangled variational autoencoder for modeling 3d facial details. IEEE Transactions on Visualization and Computer Graphics, 29(8):3630–3641, 2022. 11

[29] J. Liu, B. Hui, K. Li, Y. Liu, Y.-K. Lai, Y. Zhang, Y. Liu, and J. Yang. Geometry-guided dense perspective network for speech-driven facial animation. IEEE Transactions on Visualization and Computer Graphics, 28(12):4873–4886, 2021. 2

[30] T. Liu, F. Chen, S. Fan, C. Du, Q. Chen, X. Chen, and K. Yu. Anitalker: animate vivid and diverse talking faces through identity-decoupled facial motion encoding. In Proceedings ofthe 32nd ACM International Conference on Multimedia, pp. 6696–6705, 2024. 3

[32] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 10

[31] X. Liu, Y. Xu, Q. Wu, H. Zhou, W. Wu, and B. Zhou. Semantic-aware implicit neural audio-driven video portrait generation. In European conference on computer vision, pp. 106–125. Springer, 2022. 2

[33] Y. Lu, J. Chai, and X. Cao. Live Speech Portraits: Real-time photorealistic talking-head animation. ACM TOG, 40(6), 2021. 2

[34] Z. Ma, X. Zhu, G. Qi, C. Qian, Z. Zhang, and Z. Lei. Diffspeaker: Speechdriven 3d facial animation with diffusion transformer. arXiv preprint arXiv:2402.05712, 2024. 2, 6, 9

[35] E. Molad, E. Horwitz, D. Valevski, A. R. Acha, Y. Matias, Y. Pritch, Y. Leviathan, and Y. Hoshen. Dreamix: Video diffusion models are general video editors. arXiv preprint arXiv:2302.01329, 2023. 2

[36] F. Nocentini, T. Besnier, C. Ferrari, S. Arguillere, S. Berretti, and M. Daoudi. Scantalk: 3d talking heads from unregistered scans. In European Conference on Computer Vision, pp. 19–36. Springer, 2024. 2, 6, 9

[37] Y. Pan, C. Landreth, E. Fiume, and K. Singh. Vocal: Vowel and consonant layering for expressive animator-centric singing animation. In SIG-GRAPH Asia 2022 Conference Papers, pp. 1–9, 2022. 2

[38] Y. Pan, C. Liu, S. Xu, S. Tan, and J. Yang. Vasa-rig: Audio-driven 3d facial animation with livemood dynamics in virtual reality. IEEE Transactions on Visualization and Computer Graphics, 2025. 2

[39] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur. Librispeech: an asr corpus based on public domain audio books. In 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP), pp. 5206–5210. IEEE, 2015. 5

[40] K. Prajwal, R. Mukhopadhyay, V. P. Namboodiri, and C. Jawahar. A lip sync expert is all you need for speech to lip generation in the wild. In ACM MM, pp. 484–492, 2020. 2, 6

[41] D. Qin, J. Saito, N. Aigerman, T. Groueix, and T. Komura. Neural face rigging for animating and retargeting facial meshes in the wild. In ACM SIGGRAPH 2023 Conference Proceedings, pp. 1–11, 2023. 6, 9

[42] R. Raguram, O. Chum, M. Pollefeys, J. Matas, and J.-M. Frahm. Usac: A universal framework for random sample consensus. IEEE transactions on pattern analysis and machine intelligence, 35(8):2022–2038, 2012. 5

[43] A. Richard, M. Zollhöfer, Y. Wen, F. De la Torre, and Y. Sheikh. Meshtalk: 3d face animation from speech using cross-modality disentanglement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1173–1182, 2021. 1, 2

[44] K. Seo, S. W. Oh, J. Lu, J.-Y. Lee, S. Kim, and J. Noh. Styleportraitvideo: Editing portrait videos with expression optimization. Comput. Graph. Forum, 41(7), 2022. 2

[45] N. Sharp, S. Attaiki, K. Crane, and M. Ovsjanikov. Diffusionnet: Discretization agnostic learning on surfaces. ACM Transactions on Graphics (TOG), 41(3):1–16, 2022. 2

[46] S. Shen, W. Zhao, Z. Meng, W. Li, Z. Zhu, J. Zhou, and J. Lu. Difftalk: Crafting diffusion models for generalized audio-driven portraits animation. In CVPR, 2023. 2

[47] L. N. Smith and N. Topin. Super-convergence: Very fast training of neural networks using large learning rates. In Artificial intelligence and machine learning for multi-domain operations applications, vol. 11006, pp. 369– 386. SPIE, 2019. 10

[48] W. Song, X. Wang, Y. Jiang, S. Li, A. Hao, X. Hou, and H. Qin. Expressive 3d facial animation generation based on local-to-global latent diffusion. IEEE Transactions on Visualization and Computer Graphics, 30(11):7397–7407, 2024. 2

[49] S. Stan, K. I. Haque, and Z. Yumak. Facediffuser: Speech-driven 3d facial animation synthesis using diffusion. In Proceedings of the 16th ACM SIGGRAPH Conference on Motion, Interaction and Games, pp. 1– 11, 2023. 2

[50] Z. Sun, T. Lv, S. Ye, M. Lin, J. Sheng, Y.-H. Wen, M. Yu, and Y.-j. Liu. Diffposetalk: Speech-driven stylistic 3d facial animation and head pose generation via diffusion models. ACM Transactions on Graphics (TOG), 43(4):1–9, 2024. 2

[51] S. Suwajanakorn, S. M. Seitz, and I. Kemelmacher-Shlizerman. Synthesizing obama: learning lip sync from audio. ACM Transactions on Graphics (ToG), 36(4):1–13, 2017. 2

[52] S. Taylor, T. Kim, Y. Yue, M. Mahler, J. Krahe, A. G. Rodriguez, J. Hodgins, and I. Matthews. A deep learning approach for generalized speech animation. ACM Transactions On Graphics (TOG), 36(4):1–11, 2017. 2

[53] B. Thambiraja, S. Aliakbarian, D. Cosker, and J. Thies. 3diface: Diffusion-based speech-driven 3d facial animation and editing. arXiv preprint arXiv:2312.00870, 2023. 2

[54] B. Thambiraja, I. Habibie, S. Aliakbarian, D. Cosker, C. Theobalt, and J. Thies. Imitator: Personalized speech-driven 3d facial animation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pp. 20621–20631, 2023. 2

[55] L. Tian, Q. Wang, B. Zhang, and L. Bo. Emo: Emote portrait alivegenerating expressive portrait videos with audio2video diffusion model under weak conditions. arXiv preprint arXiv:2402.17485, 2024. 2

[56] K. Vougioukas, S. Petridis, and M. Pantic. Realistic speech-driven facial animation with gans. IJCV, 128(5):1398–1413, 2020. 2

[57] C. Wang, K. Tian, J. Zhang, Y. Guan, F. Luo, F. Shen, Z. Jiang, Q. Gu, X. Han, and W. Yang. V-express: Conditional dropout for progressive training of portrait video generation. arXiv preprint arXiv:2406.02511, 2024. 1, 2

[58] S. Wang, L. Li, Y. Ding, C. Fan, and X. Yu. Audio2head: Audio-driven one-shot talking-head generation with natural head motion. In IJCAI, 2021. 2

[59] S. Wang, L. Li, Y. Ding, and X. Yu. One-shot talking face generation from single-speaker audio-visual correlation learning. In AAAI, vol. 36, pp. 2531–2539, 2022. 2

[60] H. Wei, Z. Yang, and Z. Wang. Aniportrait: Audio-driven synthesis of photorealistic portrait animation. arXiv preprint arXiv:2403.17694, 2024. 1

[61] J. Xing, M. Xia, Y. Zhang, X. Cun, J. Wang, and T.-T. Wong. Codetalker: Speech-driven 3d facial animation with discrete motion prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12780–12790, 2023. 1, 2, 5, 6, 9

[62] M. Xu, H. Li, Q. Su, H. Shang, L. Zhang, C. Liu, J. Wang, Y. Yao, and S. Zhu. Hallo: Hierarchical audio-driven visual synthesis for portrait image animation. arXiv preprint arXiv:2406.08801, 2024. 1, 2, 3

[63] S. Xu, G. Chen, Y.-X. Guo, J. Yang, C. Li, Z. Zang, Y. Zhang, X. Tong, and B. Guo. Vasa-1: Lifelike audio-driven talking faces generated in real time. arXiv preprint arXiv:2404.10667, 2024. 1, 2

[64] J. Yang, A. Zeng, R. Zhang, and L. Zhang. X-pose: Detecting any keypoints. In European Conference on Computer Vision, pp. 249–268. Springer, 2025. 4

[65] P. Yang, H. Wei, Y. Zhong, and Z. Wang. Semi-supervised speech-driven 3d facial animation via cross-modal encoding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 21032– 21041, 2023. 2

[66] S. Yang, Z. Kong, F. Gao, M. Cheng, X. Liu, Y. Zhang, Z. Kang, W. Luo, X. Cai, R. He, et al. Infinitetalk: Audio-driven video generation for sparse-frame video dubbing. arXiv preprint arXiv:2508.14033, 2025. 2

[67] Z. Ye, Z. Jiang, Y. Ren, J. Liu, J. He, and Z. Zhao. Geneface: Generalized and high-fidelity audio-driven 3d talking face synthesis. arXiv preprint arXiv:2301.13430, 2023. 2

[68] K. Yun, S. Hong, C. Kim, and J. Noh. Anymole: Any character motion inbetweening leveraging video diffusion models. pp. 27838–27848, 2025. 2

[69] K. Yun, C. Kim, H. Shin, and J. Noh. Ffacenerf: Few-shot face editing in neural radiance fields. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pp. 10825–10835, 2025. 11

[70] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 8

[71] W. Zhang, X. Cun, X. Wang, Y. Zhang, X. Shen, Y. Guo, Y. Shan, and F. Wang. Sadtalker: Learning realistic 3d motion coefficients for stylized audio-driven single image talking face animation. In CVPR, 2023. 2

[72] X. Zhang, J. Li, J. Zhang, Z. Dang, J. Ren, L. Bo, and Z. Tu. Semtalk: Holistic co-speech motion generation with frame-level semantic emphasis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 13761–13771, 2025. 2

[73] X. Zhang, J. Li, J. Zhang, J. Ren, L. Bo, and Z. Tu. Echomask: Speechqueried attention-based mask modeling for holistic co-speech motion generation. In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 10827–10836, 2025. 2

[74] Z. Zhao, Z. Lai, Q. Lin, Y. Zhao, H. Liu, S. Yang, Y. Feng, M. Yang, S. Zhang, X. Yang, et al. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025. 11

[75] L. Zheng, Y. Zhang, H. Guo, J. Pan, Z. Tan, J. Lu, C. Tang, B. An, and S. Yan. Memo: Memory-guided diffusion for expressive talking video generation. arXiv preprint arXiv:2412.04448, 2024. 3, 10

[76] H. Zhou, Y. Sun, W. Wu, C. C. Loy, X. Wang, and Z. Liu. Posecontrollable talking face generation by implicitly modularized audiovisual representation. In CVPR, 2021. 2

[77] W.-Y. Zhou, L. Yuan, S.-Y. Chen, L. Gao, and S.-M. Hu. Lc-nerf: Local controllable face generation in neural radiance field. IEEE Transactions on Visualization and Computer Graphics, 30(8):5437–5448, 2023. 11

[78] Y. Zhou, X. Han, E. Shechtman, J. Echevarria, E. Kalogerakis, and D. Li. Makeittalk: Speaker-aware talking-head animation. ACM TOG, 39(6), 2020. 2

[79] Y. Zhou, Z. Xu, C. Landreth, E. Kalogerakis, S. Maji, and K. Singh. Visemenet: Audio-driven animator-centric speech animation. ACM Transactions on Graphics (TOG), 37(4):1–10, 2018. 2, 11

![](images/f9ab20acaf7b90dc2f65bf47fc627ec6ab40cf3d9450db6b72aa30a19e7e75e5.jpg)

![](images/3988608099239c5ec7d1fed7a87ce8b0ce402391f76de05091ae0a710e60529f.jpg)

![](images/333ceedddbed7e25e2d0e10f8948ebd959302ae80f4b77130c52fd01cfa6e5e9.jpg)

Sunjin Jung is an Assistant Professor in the Department of Computer Engineering at Sungshin Womens University. She received her M.S. and Ph.D. degrees from the Korea Advanced Institute of Science and Technology (KAIST). Her research interests include computer graphics and character animation.

demic career, he was a graphics scientist at a Hollywood visual effects company, Rhythm and Hues Studios. He held the title of KAIST chair professorship (2011) and received a technical innovation award from KAIST (2011). A research result, ScreenX, was selected as one of ten most representative research outcomes from KAIST (2013) and later successfully commercialized by the leading movie theater chain in Korea, CGV. Recently, he received the research innovation award at the 50th anniversary of KAIST.

Jung Eun Yoo is an R&D engineer who received a Ph.D. in 2025 from the Graduate School of Culture Technology at the Korea Advanced Institute of Science and Technology (KAIST). Her research focuses on applying AI within creative workflows to streamline and simplify the content creation process..

Inyup Lee is a Ph.D student in the Graduate School of Culture Technology at Korea Advanced Institute of Science and Technology (KAIST). He earned his M.S. from KAIST in 2025. His research focuses on facial animation, facial animation editing and facial reconstruction.

Junyong Noh is a Professor in the Graduate School of Culture Technology (GSCT) at KAIST. He earned the Ph.D. degree in computer science (2002), the masters degree in computer engineering (1996), and the bachelors degree in electrical engineering (1994) all from the University of Southern California (USC). His current research focus includes facial/character animation, virtual/augmented reality, image/video manipulation for immersive and interactive experiences. Prior to his aca-

![](images/0626abbaa5fdbfdb664f4ded0ae324e2b1f61edaac11678f6721a9790708497c.jpg)  
Kwan Yun is a Ph.D student in the Graduate School of Culture Technology at Korea Advanced Institute of Science and Technology (KAIST). He earned his M.S. from KAIST in 2024. His research focuses on leveraging generative models for face and character stylization, animation, and editing.

![](images/966c701f0e82263e83a100494d0bd265a1933db9d129c6c89f3af800ece2f116.jpg)

Serin Yoon is currently a visiting researcher at the DGP Lab, University of Toronto. She received her master’s degree from the Graduate School of Culture Technology at KAIST in 2026, and her bachelor’s degree in Computer Science Education from SKKU in 2024. Her research interests include computer graphics, computer vision, and 3D animation.

# Supplementary Material for AnyTalk: Speech Animation for Arbitrary Characters Leveraging a Video Generation Model

## 1 CONTENT

In this document, we provide additional experimental details, extended comparisons, and supplementary analyses that complement the main paper. The table of contents are as follows:

• Section 2: Section 2.1 describes the dataset. Section 2.2 presents detailed implementations of baseline methods. Section 2.3 provides additional comparison results. Section 2.4 includes extended experiments, covering a sensitivity analysis of lip-sync metrics on stylized characters, evaluation excluding Characterspecific Fine-tuning (CsF), and temporal stability analysis.

• Section 3: Section 3.1 analyzes inconsistent lip motion across viewpoints. Section 3.2 examines 3D reconstruction mismatches for unseen views. Section 3.3 discusses approaches for enhancing naturalness.

## 2 EXPERIMENTS DETAILS

## 2.1 Dataset

We used five characters, each with distinct mesh structures, styles, and blendshape configurations. Victor consists of a single mesh that includes all facial components. In contrast, Emily, VMan, Malcolm, and Morphy have their facial elements divided into separate meshes. Specifically, Emily and VMan include the face, hair, left/right eyeballs, and tongue as separate meshes. Malcolm includes the face, hair, and left/right eyeballs, while Morphy includes the face, hair, eyebrows, left/right eyeballs, and left/right ears. The components of each character are shown in Figure 1.

![](images/7dab2d64fe2168bed355c817f3f68bdb0a6c45186ff9e81ecbd7052affa38b54.jpg)  
Fig. 1: Mesh components of each character. From the top left to the bottom right: Victor, Emily, VMan, Malcolm, Morphy.

## 2.2 Baseline Comparison

## 2.2.1 ScanTalk

To generate audio-driven speech animations using ScanTalk, a single uniform face mesh was used for Victor, whereas separated face meshes were utilized for the other characters. The resulting animations were subsequently combined with the remaining meshes to produce the final results. Furthermore, to match the scale of each mesh with the VOCASET-like meshes used in ScanTalk training, the center points and scales of the meshes were calculated and adjusted to align with VOCASET [2].

## 2.2.2 DiffSpeaker

DiffSpeaker was trained on VOCASET and performs inference using the same dataset. We generated audio-driven speech animation using VOCASET as input. The original mesh includes eyeballs, but to perform retargeting with NFR, the eyeballs were removed from the original mesh using the FLAME mask before inputting it into the model.

## 2.2.3 CodeTalker

CodeTalker [4] is a supervised model that encodes facial actions using a discrete motion prior, generating 3D facial motion with expressive lip and facial movements. Similar to DiffSpeaker, CodeTalker was trained on the BIWI and VOCASET datasets. Therefore, we generated audio-driven speech animation using VOCASET as input and applied the FLAME mask to remove the eyeballs from the original mesh. Then, we performed retargeting using NFR. While the VOCA mesh maintained its structure without distortion during speech animation generation, the retargeting process introduced mesh distortion, leading to reduced lip synchronization quality.

## 2.2.4 NFR

When using NFR, its paper [3] recommends using a face mesh without the mouth, eyes, nose sockets, and eyeballs to avoid wrong deformations. However, Victor includes all these elements in a single structure, and as there is no definition of which element each vertex corresponds to, separating them is not feasible. While removing vertices corresponding to specific elements is possible using 3D software, this approach was deemed unsuitable for comparisons involving arbitrary avatars. Our method is designed to generate speech animations without requiring manual preprocessing, regardless of the mesh structure or blendshape configuration. Therefore, Victor, including the mouth, eyes, nose sockets, and eyeballs, was used as input. For the other charactersEmily, VMan, Malcolm, and Morphyseparated face meshes were used as input. Additionally, the align.blend file provided by NFR was used to match the scale of the meshes with those used during NFR training. This adjustment is illustrated in Figure 2. For cases where separated face meshes were used, the output was combined with the remaining meshes (e.g., hair, eyeballs) to produce the final result.

## 2.3 Additional Comparison

We compared our approach with additional method, Choi et al. + Hallo [1, 5], using all 5 characters and 30 audio clips, resulting in a total of 150 speech animations. Qualitative comparison results are presented in Figure 3, while quantitative evaluation results are shown in Table 1. As shown in Figure 3, the lip movements generated by our method more precisely follow the given sentence. In contrast, Choi et al. + Hallo produced only subtle movements that did not align with the phonemes. Table 1 shows that our method performed significantly better compared to Choi et al. across both metrics.

![](images/b08b7680060d870ce33bd48168413a5a0672ba0aea2a58437ab88131046a52fa.jpg)  
Fig. 2: Mesh alignment for baseline methods in Blender.

Tab. 1: Quantitative results comparing our method with baselines. Best denoted in bold.
<table><tr><td>Methods</td><td>LSE-D↓</td><td>LSE-C↑</td></tr><tr><td>Ours</td><td>11.304</td><td>3.155</td></tr><tr><td>Choi et al. + Hallo</td><td>16.571</td><td>0.285</td></tr></table>

## 2.3.1 Choi et al.

Choi et al. [1] is a semi-supervised video-based retargeting method that estimates blendshape parameters for 3D stylized characters from a source human performance video using an image translation-based approach. For the source video, we used the video of the first identity from MEAD dataset for both training and inference. Although the method does not require a paired dataset for each source video and the animation frames of the target character, the target frames, along with the corresponding blendshape parameters, must exist and match the range of motion (ROM) of the source video, which does not exist. To address this, we activated blendshape parameters randomly and stitched them together to create target animation frames. During inference, we used Hallo to generate the source video. We obtained the code from the authors to run the method.

## 2.4 Additional Experiments

## 2.4.1 Sensitivity Analysis of Lip-Sync Metrics on Stylization

To evaluate whether Lip-Sync Error (LSE) metrics, originally trained on natural human faces, remain reliable for stylized characters, we conducted a sensitivity analysis. We compared the performance of stylized characters (Morphy and Malcolm) against photorealistic nonstylized characters (Emily, VMan, and Victor). The results, summarized in Table 2, indicate that while non-stylized characters achieve slightly better scores, the performance gap is not statistically significant. Specifically, LSE-D increased by only 5.4%, and LSE-C decreased by 4.6% when moving from photorealistic to stylized domains. This suggests that, despite the domain shift in appearance, the metrics effectively capture the underlying lip dynamics, which in our stylized assets still closely follow human-like phonetic movements. Thus, LSE-D and LSE-C remain valid comparative indicators for stylized character animation.

## 2.4.2 Evaluation Excluding Personalization

To evaluate the performance of our method under different usage regimes, we conducted experiments distinguishing between personalized and generic inference settings. While our full pipeline employs CsF to adapt the underlying 2D generator to a target identity, baseline methods such as ScanTalk, DiffSpeaker, and CodeTalker operate in a generic setting, relying on post-hoc alignment or retargeting without identity-specific optimization. To ensure a fair comparison when quantifying the contribution of our core architecture independent of this personalization, we include a non-personalized variant of our method (ours w/o CsF) in our evaluation. All experiments follow the same settings as our ablation study.

Tab. 2: Comparison of lip-sync metrics between stylized and nonstylized characters. The marginal difference suggests that SyncNetbased embeddings remain robust to artistic stylization.
<table><tr><td>Characters</td><td>LSE-D↓</td><td>LSE-C↑</td></tr><tr><td>Stylized (Morphy, Malcolm)</td><td>11.65</td><td>3.07</td></tr><tr><td>Non-Stylized (Emily, VMan, Victor)</td><td>11.05</td><td>3.22</td></tr></table>

Tab. 3: Quantitative comparison with baselines and our nonpersonalized variant (w/o CsF) to evaluate the impact of characterspecific tuning.
<table><tr><td>Methods</td><td>LSE-D↓</td><td>LSE-C↑</td></tr><tr><td>ScanTalk</td><td>10.787</td><td>2.955</td></tr><tr><td>DiffSpeaker + NFR</td><td>13.393</td><td>0.559</td></tr><tr><td>CodeTalker + NFR</td><td>13.392</td><td>0.550</td></tr><tr><td>Ours w/o CsF</td><td>11.207</td><td>3.451</td></tr><tr><td>Ours</td><td>10.695</td><td>3.397</td></tr></table>

As reported in Table 3, even without character-specific tuning, our base model significantly outperforms the retargeting-based baselines (DiffSpeaker + NFR and CodeTalker + NFR) across all metrics and achieves highly competitive lip-sync accuracy compared to ScanTalk. Furthermore, ours w/o CsF achieves the highest lip-sync confidence among all evaluated methods, demonstrating the robustness of our underlying motion synthesis framework. The application of CsF in our full pipeline further refines the lip-sync precision, yielding the best overall LSE-D while maintaining strong confidence.

## 2.4.3 Temporal Stability of AnyTalk and AnyTalk

To quantitatively assess the temporal stability of our primary model, AnyTalk, compared to its real-time variant, $\mathbf { A n y T a l k } _ { R T } .$ , we measured the temporal smoothness of the generated facial animations. Specifically, we computed the average jerk (the rate of change of acceleration) of facial landmarks across consecutive generated frames, where lower jerk values indicate smoother, less jittery motion. The facial landmarks used for this evaluation are extracted using XPose [6]. We evaluated this metric under two configurations: the overall jerk across all detected facial landmarks, and the talk-related jerk, which isolates the landmarks specifically associated with the mouth and lip regions. As detailed in Table 4, AnyTalk exhibits lower jerk values in both the full facial configuration and the localized talk-related configuration. These results quantitatively demonstrate that while AnyTalk<sub>RT</sub> successfully distilled the architecture for real-time inference, the full AnyTalk model (acting as the teacher network) maintains a slight advantage in preserving temporal stability and producing naturally smooth facial dynamics.

## 3 MULTI-VIEW GENERATION AND 3D RECONSTRUCTIONANALYSIS

To further investigate the limitations of our single-view optimization pipeline and naive solutions cannot be adopted to address this issue, we conducted two supplementary experiments: (1) generating talkinghead videos from multiple viewpoints, left ( 45◦), front (0◦), and right (+45◦) using Hallo [5], and (2) reconstructing the resulting frames into 3D meshes using Hunyuan3D-2 [7]. The results confirm the challenges discussed in the main text.

## 3.1 Inconsistent Lip Motion Across Viewpoints

We generated three separate videos of the same input audio: one each for the frontal, left, and right views. As shown in Figure 4, although the overall head pose approximately matches the intended viewpoints, the lip motion and facial expressions vary noticeably between views. This inconsistency arises from the stochastic sampling inherent in diffusion models, which leads to non-deterministic outputs, even for the same audio content. Consequently, this lack of inter-view coherence introduces conflicting supervision signals, which prevents the 3D representation from converging to a sharp, geometrically consistent state. Such variability undermines the reliability of a multi-view optimization strategy that assumes consistent motion priors across generated videos.

![](images/33dbcdd778a5a23c9c458b111dd3c3b415e17e2488af6ae857e87c4ece6b7e51.jpg)  
Fig. 3: Comparison with additional baselines on Emily. Each letter and its corresponding frame for comparison is presented. Ours best follows the given audio while Choi et al. + Hallo show only subtle movements.

Tab. 4: Quantitative comparison of temporal stability of AnyTalk and AnyTalk\_RT.
<table><tr><td>Methods</td><td>Jerk↓</td><td>Talk-related jerk↓</td></tr><tr><td>AnyTalk</td><td>27.297</td><td>14.419</td></tr><tr><td> $\mathrm { A n y T a l k } _ { R T }$ </td><td>27.603</td><td>14.462</td></tr></table>

![](images/984fc3a9c9e78a78510fa1cc378706217429d75ed5afdedb879acc049df9404f.jpg)  
Fig. 4: Multi-view generation results for the same audio segment at time t. The top row shows the input character renders for left, frontal, and right viewpoints. The middle and bottom rows display the corresponding generated video frames at identical timestamps, illustrating inconsistent lip motions and facial expressions across views.

## 3.2 3D Reconstruction Mismatch for Unseen Views

Next, we applied a state-of-the-art 3D reconstruction method [7] to the generated multi-view frames, aiming to recover a mesh that matches the original character for 3D guidance. Figure 5 illustrates that the reconstructed geometry often deviates from the source characters shape and proportions particularly for viewpoints of left and right. Artifacts such as distorted jawlines and misaligned landmarks indicate that generic reconstruction methods struggle to faithfully reproduce the subtle details of our target meshes when given diffusion-generated images as input.

![](images/7cf2f6c5d39a316e10850407da34cb67bc8afedd5a5089cdb1b790f4aa1393de.jpg)  
Fig. 5: Comparison between original meshes (first and third rows) and 3D reconstructed meshes (second and fourth rows) generated by Hunyuan3D-2. While the reconstructed characters resemble the originals, they lack precise pixel-wise alignment, making them unsuitable for direct motion guidance.

These results together demonstrate that naively combining diffusion generation in multi-view or using off-the-shelf reconstruction is insufficient to resolve the single-view dependency of our current system. Future work should therefore explore integrated approaches for multiview consistency in talking-head diffusion models or end-to-end 3D speech animation frameworks that jointly optimize for both view synthesis and geometric fidelity.

## 3.3 Enhancing Naturalness

Because our method utilizes blendshapes of an arbitrary character, adding random expressions, such as random eye blinks, can be easily achieved as follows:

$$
w = \left\{ \begin{array} { l l } { \displaystyle \frac { t - s } { d _ { i } } , } & { s \leq t < s + d _ { i } / 2 , } \\ { \displaystyle \frac { d _ { i } - t + s } { d _ { i } } , } & { s + d _ { i } / 2 \leq t < s + d _ { i } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{1}
$$

Here, s denotes the start time of the expression, t is the current time, and $d _ { i }$ is the duration of the expression. By simply applying random values to s and $d _ { i }$ , the character can easily exhibit dynamic expressions. Alternatively, we can directly apply a constant expression throughout the sequence by assigning $w = 1$ . Examples demonstrating random blinking and constant expressions are provided in the supplementary video.

## REFERENCES

[1] Y. Choi, I. Lee, S. Cha, S. Kim, S. Jung, and J. Noh. Deep-learning-based facial retargeting using local patches. In Computer Graphics Forum, p. e15263. Wiley Online Library, 2024. 1, 2

[2] D. Cudeiro, T. Bolkart, C. Laidlaw, A. Ranjan, and M. J. Black. Capture, learning, and synthesis of 3d speaking styles. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10101–10111, 2019. 1

[3] D. Qin, J. Saito, N. Aigerman, T. Groueix, and T. Komura. Neural face rigging for animating and retargeting facial meshes in the wild. In ACM SIGGRAPH 2023 Conference Proceedings, pp. 1–11, 2023. 1

[4] J. Xing, M. Xia, Y. Zhang, X. Cun, J. Wang, and T.-T. Wong. Codetalker: Speech-driven 3d facial animation with discrete motion prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12780–12790, 2023. 1

[5] M. Xu, H. Li, Q. Su, H. Shang, L. Zhang, C. Liu, J. Wang, Y. Yao, and S. Zhu. Hallo: Hierarchical audio-driven visual synthesis for portrait image animation. arXiv preprint arXiv:2406.08801, 2024. 1, 2

[6] J. Yang, A. Zeng, R. Zhang, and L. Zhang. X-pose: Detecting any keypoints. In European Conference on Computer Vision, pp. 249–268. Springer, 2025. 2

[7] Z. Zhao, Z. Lai, Q. Lin, Y. Zhao, H. Liu, S. Yang, Y. Feng, M. Yang, S. Zhang, X. Yang, et al. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025. 2, 3