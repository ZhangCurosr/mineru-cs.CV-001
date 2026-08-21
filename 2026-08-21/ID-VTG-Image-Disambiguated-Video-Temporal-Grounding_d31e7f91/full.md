# ID-VTG: Image-Disambiguated Video Temporal Grounding

Minghang Zheng Wangxuan Institute of Computer Technology, Peking University Beijing, China minghang@pku.edu.cn

Hongyi Yang Wangxuan Institute of Computer Technology, Peking University Beijing, China hyyang25@stu.pku.edu.cn

Jingli Wei Wangxuan Institute of Computer Technology, Peking University Beijing, China jingliwei@stu.pku.edu.cn

Yang Liu   
Wangxuan Institute of Computer Technology, Peking University   
State Key Laboratory of General Artificial Intelligence, Peking University Beijing, China yangliu@pku.edu.cn

## Abstract

Video Temporal Grounding (VTG) faces significant challenges when natural language queries must distinguish between multiple events involving visually similar entities, particularly when relying on finegrained visual attributes that are dificult to describe accurately in words alone. To address this, we introduce Image-Disambiguated Video Temporal Grounding (ID-VTG), a task that leverages multi modal queries combining a reference image and a text description to precisely localize segments where a specific instance performs a described action. To facilitate research, we construct two benchmarks: IDVTG-Gym, focusing on fine-grained, compositionally ordered gymnastics actions with athletes in similar uniforms; and IDVTG-InternVid, an open-world dataset featuring diverse entities (e.g., humans, animals, fictional characters) and significant temporal distractors. Methodologically, we propose the Visually-Guided Disambiguation Aggregation (VGD-Agg) framework based on a dual-branch fast-slow architecture. The fast branch eficiently generates preliminary event proposals, while the slow branch per forms fine-grained frame-level matching between video frames and the reference image. We enhance discriminability via two learnable tokens: a Compare Token, which represents hard negatives to probe for the presence of the target instance (as referred to by the query image), and a Depress Value, which represents text-irrelevant events. Proposals that the Compare Token identifies as lacking the target instance are pushed toward the Depress Value, thus easing disambiguation via the text query. Extensive experiments validate our approach, which achieves state-ofthe-art results on the proposed benchmarks. Code is available at https://github.com/oceanflowlab/ID-VTG.

CCS Concepts • Computing methodologies → Visual content-based indexing and retrieval; Activity recognition and understanding.

Keywords Multimodal Query, Video Temporal Grounding

ACM Reference Format:   
Minghang Zheng, Jingli Wei, Hongyi Yang, and Yang Liu. 2026. ID-VTG: Image-Disambiguated Video Temporal Grounding. In Proceedings ofthe 34th ACM International Conference on Multimedia (MM ’26), November 10– 14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 10 pages. https: //doi.org/10.1145/3767308.3836517

## 1 Introduction

Video Temporal Grounding (VTG) localizes events in videos from natural language queries. However, existing methods struggle when text cannot distinguish among visually similar entities, especially when diferentiation depends on fine-grained attributes (e.g., texture or facial appearance) or unfamiliar subjects that are hard to describe precisely. Such cases lead to ambiguous queries and inaccurate localization. As shown in Figure 1, when two similar subjects both perform “the man is speaking into a microphone" at diferent times, text-only VTG cannot reliably refer to the intended segment. To address this limitation, we introduce Image-Disambiguated Video Temporal Grounding (ID-VTG), a task that supplements text with a reference image. This task directly aligns with critical real-world scenarios where users possess a strong visual prior but find textual descriptions cumbersome and imprecise. Practical applications include intelligent surveillance (locating a specific suspect’s actions using a photo), sports broadcasting (tracking a specific athlete), and multimodal retrieval (finding exact products in videos) [8]. By serving as a strict visual constraint, the reference image naturally eliminates semantic ambiguity, enabling precise localization of the intended subject.

To facilitate research on this task, we introduce two comprehensive datasets: IDVTG-Gym and IDVTG-InternVid, both pairing multimodal queries with precise temporal annotations. IDVTG-Gym features fine-grained, compositionally ordered actions. It is built from gymnastics videos in which athletes wear similar team uniforms, creating substantial visual ambiguity. As seen in Figure 2 (a), these reference images are paired with textual queries that range from holistic events (e.g., “The gymnast performs on the uneven bars”) to atomic sub-actions (e.g., “giant circle”). The annotations addi tionally encode ordinal cues (e.g., “for the second time”), forcing models to reason over action order rather than rely on simple action recognition. IDVTG-InternVid, in contrast, is designedfor open-world settings with diverse content and pronounced distractors. Figure 2(c) and (d) reveal its coverage of varied video types, including humans, objects, animals, and fictional characters, accompanied by a broader vocabulary. IDVTG-InternVid also includes strong distractors. As illustrated in Figure 2(b), two diferent individuals successively perform the action “threads a shoelace by hand", requiring the model to distinguish subtle, fine-grained diferences to disambiguate these highly confusable temporal segments.

![](images/1545047477a7682f173a5006c197cc5266afa15f2a89f43053de9d85015ac112.jpg)  
Figure 1: Illustration of the proposed Image-Disambiguated Video Temporal Grounding (ID-VTG) task.

To tackle the challenges in ID-VTG, we propose the Visually-Guided Disambiguation Aggregation (VGD-Agg) framework with a dual-branch fast-slow design that balances temporal modeling and fine-grained visual matching: the fast branch eficiently produces preliminary temporal event proposals from high-level video semantics, while the slow branch performs frame-level video-reference matching to disambiguate visually similar instances. To robustly handle visual ambiguity, we introduce two learnable and videospecific representations: a Compare Token and a Depress Value. The Compare Token acts as a high-similarity hard negative prototype, serving as a threshold to distinguish frames containing the query image from those that do not. Conversely, the Depress Value captures text-irrelevant video events; features of proposals lacking the query image are pushed toward this value to make it easier to dif ferentiate based on the text query. Functionally, a frame is deemed image-relevant only if its afinity to the query image exceeds its similarity to the Compare Token. Accordingly, our Vision-Assisted Disambiguation module enhances relevant proposals by aggregating matched frames, while pulling image-absent proposal features to the Depress Value. This produces clearly separable representations, enabling the subsequent Text-Guided Grounding module to classify proposals and regress boundaries.

Our contributions are: (1) We introduce Image-Disambiguated Video Temporal Grounding (ID-VTG) and two datasets, IDVTG-Gym and IDVTG-InternVid. (2) We propose the VGD-Agg framework to suppress visual distractors and adaptively aggregate features for precise disambiguation-oriented grounding. (3) Extensive experiments show that our method significantly outperforms adapted state-of-the-art VTG baselines, establishing a strong benchmark for multimodal video grounding.

## 2 Related Works

## 2.1 Methods for Video Temporal Grounding

Video temporal grounding aims to localize the segment in a video with a text query. Existing methods are text-centric and fall into two categories: proposal-based approaches [6, 15, 19, 33, 42, 44, 46, 47, 49, 50], and proposal-free approaches [10, 13, 16, 18, 22, 23, 28, 35, 45, 48, 51, 52]. Despite strong performance, these methods rely solely on textual queries, which becomes a bottleneck when descriptions cannot distinguish between visually similar entities. Recent eforts have incorporated visual cues but do not address true multimodal disambiguation. Minotaur [11] supports image or text queries individually, but does not support the joint use of both. Zhang et al. [41] explores image-composed queries by reconstructing annotations from QVHighlights, their focus is on semantic modification, where the text describes how the reference image context should be conceptually altered. This leads to their benchmarks lacking ambiguity and being unsuitable for evaluating disambiguation capabilities. In contrast, our ID-VTG task addresses referential ambiguity, using real-world reference images as a strict visual constraint to filter out plausible distractors. To this end, we propose a novel framework that explicitly leverages image queries to disambiguate text-grounded proposals, ensuring precise localization of the target identity.

## 2.2 Datasets for Video Temporal Grounding

Existing datasets [3, 6, 12, 20, 25, 29, 43] difer in annotation granularity and ambiguity. Datasets such as ActivityNet Captions [12], Charades-STA [6], and TACoS [25] provide sentence–time interval pairs. However, their queries consist exclusively of text without reference images, and these videos typically contain a single salient actor or dominant event, making the textual query suficient for localization. Spatio-temporal video grounding (STVG) [4, 5] datasets, such as VidSTG [43] and HC-STVG [29], define the task as predicting spatial regions and temporal boundaries as outputs. Although they provide fine-grained spatial bounding boxes as target outputs, their inputs remain text-only. Therefore, their annotation protocols strictly require the target event to be unique within the video, implying that the text description alone is suficient for localization. Overall, existing benchmarks fall short in evaluating instance-disambiguated VTG due to this lack of hard negatives. To address this, we construct two benchmarks, IDVTG-Gym and IDVTG-InternVid, specifically designed with high-ambiguity scenarios and strong visual distractors for rigorous ID-VTG evaluation with the help of large pretrained models [1, 7, 27, 30, 36–39].

## 3 Dataset

In support of the ID-VTG task, we construct two datasets, IDVTG-Gym and IDVTG-InternVid, through a unified two-stage pipeline as shown in Fig. 3. The pipeline first identifies individual textual ambiguous instances in each video and assigns textual descriptions with precise temporal boundaries to their events. In the second stage, each event is associated with instance-specific spatial regions by grounding the target instance within the corresponding temporal segment, producing disambiguating visual references that serve as image queries. Through our pipeline, a textual description corresponds to multiple events at diferent times involving visually similar entities. In such cases, it is dificult to accurately describe fine-grained visual attributes using language alone. There fore, an image query provides an explicit visual reference, removes ambiguity, and makes precise localization possible.

![](images/38630423897174a367f22e566cc7afb8e81c3e0d7ae29c8252f1a436b6ae9440.jpg)

![](images/207a387536e129a189c256951049245207595245007f5e921aabdb5da66873a2.jpg)

![](images/340cbbaebfeb06966ef2a0ef464e051cfc5fa417495180b597fd3077cba31662.jpg)  
Figure 2: Overview of the ID-VTG datasets. (a) IDVTG-Gym focuses on fine-grained, ordered gymnastics actions with visually similar athletes. (b) IDVTG-InternVid focuses on open-world videos with diverse entities and strong temporal distractors. (c) The semantic category distribution of IDVTG-InternVid. (d) The lexical distribution of IDVTG-Gym and IDVTG-InternVid.

## 3.1 Ambiguous Event Generation

The goal of this stage is to identify temporal segments that cannot be uniquely localized by text alone and output the text query and timestamps. We employ distinct strategies for our IDVTG-Gym and IDVTG-InternVid datasets due to their diferent annotation information.

IDVTG-Gym. IDVTG-Gym is constructed from FineGym [26], leveraging its hierarchical annotations from event categories (e.g., uneven bars) to atomic sub-actions (e.g., giant circle). Because athletes wear similar uniforms and execute standardized routines, the data naturally exhibits strong visual and textual ambiguity. We first segment the long videos into minute-level clips and retain only those containing at least two occurrences of the same atomic action label. Since FineGym provides discrete labels, we use an MLLM [1] to convert them into free-form text queries and explicitly insert ordinal cues (e.g., the second time) when referring to repeated actions. This enforces ambiguity and requires models to perform sequential temporal reasoning rather than relying on semantic matching.

![](images/60de7fb7ff1497c465c470b6b89d0e451f8f1f191fdb40eebdd50d7b329b2f6d.jpg)  
Figure 3: Overview of the ID-VTG dataset construction pipeline. Stage 1 generates text descriptions and timestamps of ambiguous events for IDVTG-Gym and IDVTG-InternVid. Stage 2 selects disambiguating image queries by grounding target instances and verifying candidates with an MLLM.

IDVTG-InternVid. To increase open-world diversity, we construct the IDVTG-InternVid from InternVid [32] using an automated pipeline with Gemini-2.5-Pro [30]. First, we instruct Gemini to focus on characters involved in ambiguous events. To ensure

Image Ambiguity (the visual query alone is insuficient), we require the model to annotate every event performed by the target character with a specific textual query and precise timestamps. This guarantees that the reference image maps to multiple segments, forcing reliance on the text to ground specific action. To ensure Text Ambiguity (text query alone is insuficient), for scenarios where diferent characters perform similar actions, we explicitly instruct the model to generate similar text descriptions. This prevents the text from uniquely identifying the actor, forcing reliance on the reference image. We require Gemini-2.5-Pro to explicitly annotate the ambiguity type and apply a post-filtering step to discard samples that are neither text-ambiguous nor image-ambiguous. After this stage, we further employ another MLLM [1] to verify the align ment between each textual description and its corresponding video content, filtering out imperfect samples with weak or inaccurate text-content correspondence. To validate the quality of this automatic annotation pipeline and obtain a high-quality test set, we randomly sampled 10% of the data for manual annotation as a test set. The results show that the automatically labeled data achieved 76.8% on the R1@0.5 metric compared to the manual annotations.

## 3.2 Disambiguating Query Image Selection

Once ambiguous events are defined, the second stage aims to find a query image that uniquely grounds the target event. For each target segment, we sample multiple candidate frames. We employ a pretrained image grounding model (Sa2VA [38]) to localize the subject using the text query. A MLLM [1] then acts as a verifier, scoring each candidate crop based on subject consistency, image clarity, and the visibility of identity features (e.g., faces). The highest-scoring image is selected as the query. As shown in Fig. 2, we present some of our query images. It can be observed that they do not necessarily have to be full-body images and can also depict fictional characters or feature distinctive appearances of fine-grained regions.

## 3.3 Evaluation Benchmarks and Statistics

We curate specific test sets targeting diferent distributions. Standard Test Sets: We randomly sample the in-domain test splits for both IDVTG-Gym and IDVTG-InternVid. Because the IDVTG-InternVid dataset’s annotations were automatically generated by Gemini, we re-annotated the test set manually to ensure its high quality. Web Dataset: To test robustness against unseen video sources, we collected and annotated a set of videos from YouTube following the IDVTG-InternVid pipeline. This evaluates the model’s ability to generalize to new visual domains.

Tab 1 compares ID-VTG with existing VTG benchmarks. IDVTG-Gym contains 14.7k queries across 204.1 hours of video, focusing on fine-grained actions. IDVTG-InternVid provides a larger scale with 62.1k queries over 302.7 hours, emphasizing open-world vocabulary and diverse video domains. Unlike previous benchmarks, which rely solely on text queries, ID-VTG is the first to integrate Text+Image queries and enforce Ambiguity in the dataset design for video temporal grounding.

Table 1: Comparison of our ID-VTG dataset with existing VTG datasets.
<table><tr><td>Dataset</td><td>Domain</td><td>Duration</td><td>Queries</td><td>Query Type</td><td>Ambiguity</td></tr><tr><td>TACoS [25]</td><td>Cooking</td><td>10.1 h</td><td>18.2K</td><td>Text</td><td>No</td></tr><tr><td>Charades-STA [6]</td><td>Activity</td><td>57.1 h</td><td>16.1K</td><td>Text</td><td>No</td></tr><tr><td>DiDeMo [9]</td><td>Flickr</td><td>88.7 h</td><td>41.2K</td><td>Text</td><td>No</td></tr><tr><td>QVHighlights [13]</td><td>Vlog / News</td><td>425 h</td><td>10.3K</td><td>Text</td><td>No</td></tr><tr><td>ANet-Captions [12]</td><td>Activity</td><td>487.6 h</td><td>72.0K</td><td>Text</td><td>No</td></tr><tr><td>ICQ-Highlight [41]</td><td>Vlog / News</td><td>65 h</td><td>6.19K</td><td>Text + Image</td><td>No</td></tr><tr><td>IDVTG-Gym (Ours)</td><td>Gymnastics</td><td>204.1 h</td><td>14.7K</td><td>Text + Image</td><td>Yes</td></tr><tr><td>IDVTG-InternVid (Ours)</td><td>Open-world</td><td>302.7 h</td><td>62.1K</td><td>Text + Image</td><td>Yes</td></tr><tr><td>ID-VTG (Total)</td><td></td><td>506.8 h</td><td>76.8K</td><td> $\mathbf { T e x t } + \mathbf { I m a } \mathbf { \bar { g } e }$ </td><td>Yes</td></tr></table>

## 4 Method

Problem Definition. Given a video $_ \mathrm { ~  ~ }$ and a multimodal query $Q = \{ Q _ { t e x t } , Q _ { i m g } \}$ , the goal of ID-VTG is to localize a specific temporal segment $S _ { g } = \left( t _ { s } , t _ { e } \right)$ , where $t _ { s }$ and $t _ { e }$ denote the start and end timestamps, respectively. The target segment must correspond semantically to the textual description $\boldsymbol { Q } _ { t e x t }$ while visually matching the subject instance depicted in the reference image $Q _ { i m g } .$

## 4.1 Baseline Revisit

Our method is built upon SnAG [22], a representative text-only video temporal grounding framework. Two core modules inherited from the baseline: the Fast Branch for query-agnostic proposal generation and the Text-Guided Grounding module for final grounding. The Fast Branch eficiently models long-range temporal context and generates candidate event proposals without cross-modal alignment, using a Transformer-based multi-scale proposal encoder adapted from ActionFormer [40]. Given frame-level features, it constructs a hierarchical feature pyramid Z. Each $\mathbf { Z } _ { i } ^ { ( l ) } \in \mathbf { Z } ^ { ( l ) }$ corresponds to a temporal proposal centered at the $( i \times 2 ^ { l } )$ -th sampled frame with duration $2 ^ { l }$ sampled frames. This multi-scale design captures video-wide temporal dependencies and outputs proposal-level representations. Given proposal features, Text-Guided Grounding fuses proposals and text with a Transformer decoder. The decoding head consists of two parallel multi-layer perceptrons: a classification head and a regression head, predicting proposal confidence �<sub>�</sub> and temporal ofsets $( o _ { i } ^ { s } , o _ { i } ^ { e } )$ for boundary refinement, respectively. Training uses center sampling for positives and optimizes a Focal loss $\mathcal { L } _ { c l s }$ and a DIoU loss $\mathcal { L } _ { r e g }$

## 4.2 Overview

The primary challenge in the ID-VTG lies in resolving the ambiguity where multiple segments match the text description, but only one matches the reference image. Resolving such ambiguity requires incorporating fine-grained visual cues from the reference image to distinguish the target segment. Therefore, we propose the Visually-Guided Disambiguation Aggregation (VGD-Agg) framework. As illustrated in Figure 4, our framework is built upon the two baseline components revisited above and introduces two additional modules specifically designed for image-based disambiguation: a Slow Branch and a Vision-Assisted Disambiguation module.

The Slow Branch performs fine-grained frame-level visual matching between video frames and the reference image, enabling visionsensitive visual discrimination. To robustly diferentiate imagerelevant frames from visual distractors, we introduce a learnable video-specific Compare Token and its corresponding Depress Value. The Compare Token represents hard negative samples with high similarity to the query image. This design explicitly models challenging distractors and tightens the decision boundary between truly relevant frames and visually similar yet irrelevant ones. The Depress Value represents video events that are irrelevant to the query text. Proposals that do not contain the query image are pushed towards this value, making it easier to diferentiate based on the text query. To explicitly disentangle image-relevant proposals from visual distractors in the feature space, we propose the Vision-Assisted Disambiguation module. This module bridges the two branches via a Softmax-based competition mechanism between video frames and the Compare Token. For image-relevant proposals, the high visual afinity naturally directs the attention weights towards the matching frames, highlighting the target visual content. In contrast, for image-irrelevant distractors (where no frames match the image query), Compare Token dominates the attention distribution, pushing the aggregated proposal features to the Depress Value, providing a clear semantically irrelevant signal to the subsequent text-guided grounding module. Through this mechanism, we obtain Disambiguated Proposal Features, where image-relevant targets and distractors are well-separated.

![](images/e9088bfa227b5b319f36ce9e725e3ad1d3a8b3a7df00517317f825de34fc2bfb.jpg)  
Figure 4: Overview of the Visually-Guided Disambiguation Aggregation (VGD-Agg) framework. The architecture consists of a dual-branch design: (1) The Fast Branch (left) eficiently generates generic, image-agnostic proposal features from the video stream. (2) The Slow Branch (middle) performs fine-grained matching between video frames and the image query to compute similarity scores and learn a video-specific Compare Token. (3) The Vision-Assisted Disambiguation module (top) leverages the Compare Token as a dynamic decision boundary. It acts as a soft gate that suppresses proposal features dominated by visual distractors (Image-Irrelevant) while enhancing those containing the target instance (Image-Relevant). (4) Finally, the Text-Guided Grounding head (right) aligns the disambiguated features with the text to regress the precise temporal boundaries.

## 4.3 Slow Branch

While the Fast Branch captures generic temporal contexts, the Slow Branch is designed to enable vision-sensitive discrimination by executing fine-grained matching between video frames and the specific image query.

Distractor Generator. To robustly diferentiate image-relevant frames from visual distractors, we propose a distractor generator to generate two complementary video-specific representations: a Compare Token (t ) and a Depress Value $( \mathbf { v } _ { d } )$ . Crucially, we tailor the input queries to the distinct functional role of each representation: $\mathbf { t } _ { c }$ serves as a visual afinity baseline and thus requires image-specific guidance, whereas ${ \bf v } _ { d }$ aggregates text-irrelevant visual features to suppress the image-irrelevant proposals and make them easier to diferentiate based on the text query. Therefore, we condition the video features v separately on the visual query $\boldsymbol { \mathrm { Q } } ^ { \mathrm { v i s u a l } }$ and the textual query $\mathbf { Q } ^ { \mathrm { t e x t } }$ via dedicated Encoder-Decoder architectures:

$$
\mathbf { t } _ { c } = \mathrm { D e c } _ { t o k } ( \mathrm { E n c } _ { t o k } ( \mathbf { v } ) , \mathbf { Q } ^ { \mathrm { v i s u a l } } )
$$

$$
\mathbf { v } _ { d } = \mathrm { D e c } _ { v a l } ( \mathrm { E n c } _ { v a l } ( \mathbf { v } ) , \mathbf { Q } ^ { \mathrm { t e x t } } )\tag{1}
$$

(2)

where Enc/Dec denote Transformer layers [31], and v is the video features. Through this modality-specific conditioning, $\mathbf { t } _ { c }$ efectively captures the global visual context to calibrate image-video matching, while ${ \bf v } _ { d }$ encodes text-agnostic visual features to suppress irrelevant proposals. Specifically, $\mathbf { t } _ { c }$ is optimized by the Visual Matching Loss described as follows, and ${ \bf v } _ { d }$ is optimized via the Classify Loss $\mathcal { L } _ { c l s }$ within the Text-Guided Grounding module.

Fine-grained Frame-level Visual Matching. To measure the afinity between the video and the reference image, we perform a cross-modal interaction. We first append the Compare Token to the temporal dimension of the original frame-level features, forming an augmented sequence ${ \bf V } _ { i n } = \left[ { \bf v ; t } _ { c } \right]$ . This sequence is then fed into a Transformer [31] layer performing cross-attention, where $\mathbf { V } _ { i n }$ acts as the query, and the visual query features $\boldsymbol { \mathrm { Q } } ^ { \mathrm { v i s u a l } }$ serve as the key and value. This operation injects fine-grained visual cues into both the video frames and the Compare Token ${ \bf V } _ { o u t } = [ \tilde { { \bf v } } , \tilde { { \bf t } } _ { c } ] =$ $\mathrm { D e c } ( \mathbf { V } _ { i n } , \mathbf { Q } ^ { \mathrm { v i s u a l } } )$ , producing the vision-enhanced frame features: $\tilde { \mathbf { v } } .$ Finally, a scoring head predicts the similarity scores $s _ { t }$ for each frame and $s _ { c }$ for the Compare Token: $s _ { t } , s _ { c } = \mathrm { M L P } ( \mathbf { V } _ { o u t } )$

Visual Matching Loss. To enforce the role of the Compare Token as a valid discriminator, we introduce a ranking-based loss. Our core objective is to enforce a strict ordinal ranking: $\bar { s } ^ { \mathrm { g t } } > s _ { c } > \bar { s } ^ { \mathrm { n o n - g t } }$ which means average similarity within the ground truth segment should exhibit higher visual afinity than the Compare Token, while the Compare Token should score higher than average similarity out of the ground truth segment. In this way, Compare Token serves as a learnable hard negative that probes whether a proposal truly contains the target instance referred to by the query image. We formulate this objective using a set of hinge losses:s

$$
\mathcal { L } _ { \mathrm { p b } } = \operatorname* { m a x } ( 0 , m - \bar { s } ^ { \mathrm { g t } } + s _ { c } )\tag{3}
$$

$$
\mathcal { L } _ { \mathrm { n b } } = \operatorname* { m a x } ( 0 , m - s _ { c } + \bar { s } ^ { \mathrm { n o n - g t } } )\tag{4}
$$

$$
\mathcal { L } _ { \mathrm { p n } } = \mathrm { m a x } ( 0 , 2 m - \bar { s } ^ { \mathrm { g t } } + \bar { s } ^ { \mathrm { n o n - g t } } )\tag{5}
$$

$$
\mathcal { L } _ { \mathrm { s i m } } = \alpha ( \mathcal { L } _ { \mathrm { p b } } + \mathcal { L } _ { \mathrm { n b } } ) + \mathcal { L } _ { \mathrm { p n } }\tag{6}
$$

where $\bar { s } ^ { \mathrm { g t } }$ is the average similarity within the ground truth, $\bar { s } ^ { \mathrm { n o n \cdot } }$ -gt is the average similarity outside the ground truth, and � is a scalar hyperparameter that specifies the margin enforced by the hingeloss constraints.

## 4.4 Vision-Assisted Disambiguation

This module is a bridge that integrates the query-independent proposal features Z from the Fast Branch with frame features v˜ from the Slow Branch, achieving feature separability between target events and visual distractors.

Since the proposals are generated hierarchically, we first align them with the frame-level sequences. For each proposal feature $\mathbf { Z } _ { i } ^ { ( l ) }$ at pyramid level �, we identify its corresponding temporal receptive field $\mathcal { R } _ { i } ^ { ( l ) } = [ t _ { s t a r t } , t _ { e n d } ]$ in the frame sequence. We extract the corresponding visual similarity scores $\mathbf { \boldsymbol { s } } _ { \mathcal { R } } = \left[ \boldsymbol { s } _ { t } \right] _ { t \in \mathcal { R } _ { i } ^ { ( l ) } }$ and the vision-enhanced frame features $\tilde { \mathbf { v } } _ { \mathcal { R } } = \left[ \tilde { \mathbf { v } } _ { t } \right] _ { t \in \mathcal { R } _ { i } ^ { ( l ) } }$

Softmax-based Competitive Aggregation. To implement the adaptive disambiguation, we design a competition mechanism between the specific video frames and the Compare Token. We concatenate the similarity score $s _ { c }$ and Depress Value ${ \bf v } _ { d }$ with the extracted frame sequences, forming augmented sets. The aggregated visual feature $\mathbf { \hat { Z } } _ { i } ^ { ( l ) }$ is then computed via a softmax-weighted sum:

$$
\mathbf { w } _ { a t t n } = \mathrm { s o f t m a x } ( [ \mathbf { s } _ { \mathcal { R } } ; s _ { c } ] ) , \quad \hat { \mathbf { Z } } _ { i } ^ { ( l ) } = \mathbf { w } _ { a t t n } ^ { \mathrm { T } } \cdot [ \tilde { \mathbf { v } } _ { \mathcal { R } } ; \mathbf { v } _ { d } ]\tag{7}
$$

where $[ \cdot ; \cdot ]$ denotes concatenation along the temporal dimension. The final disambiguated proposal feature $\mathbf { \widetilde { Z } } _ { i } ^ { ( l ) } = \mathrm { M L P } ( \mathbf { \hat { Z } } _ { i } ^ { ( l ) } + \mathbf { Z } _ { i } ^ { ( l ) } )$ is obtained by fusing the vision-aggregated feature with the original proposal feature.

This formulation leverages the Compare Token as an adaptive baseline to separate proposals in the feature space. For imagerelevant proposals, the high frame-image afinity within the receptive field dominates the attention weights $\mathbf { w } _ { a t t n } ,$ causing the module to highlight relevant frames. Conversely, for image-irrelevant proposals, the frame scores fall below the baseline $s _ { c } .$ Therefore, the attention weights shift significantly towards the Depress Token. This efectively suppresses the misleading visual features of the distractors by replacing them with the Depress Value, providing a clean discriminative signal for the subsequent text-guided grounding module.

Overall Objective. The model is trained end-to-end by minimizing the weighted sum of the grounding losses and the proposed visual disambiguation similarity loss:

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { c l s } + \lambda _ { r e g } \mathcal { L } _ { r e g } + \mathcal { L } _ { s i m }\tag{8}
$$

## 5 Experiment

## 5.1 Experimental Setup

Datasets. We evaluate on our two benchmarks, IDVTG-Gym and IDVTG-InternVid. The queries in the IDVTG-Gym dataset include complete gymnastics events as well as fine-grained gymnastics actions. Therefore, we have divided the test set into two subsets: Sub-act. and Holistic. IDVTG-InternVid is a large-scale openworld benchmark, evaluated under two settings: (1) In-domain: testing on its test split; (2) OOD Video: testing on the Web Dataset with diverse internet videos exhibiting visual distribution shifts. Evaluation Metrics. Following standard protocols, we adopt the Recall@�, $\mathrm { I o U } = m \left( R _ { m } ^ { n } \right)$ and Mean Intersection over Union (mIoU) as our primary metrics. We report results for $n = 1$ with � $\in \{ 0 . 5 , 0 . 7 \}$ as well as the average IoU of the top-1 prediction.

Implementation Details. We use pre-trained CLIP (ViT-L/14) [24] to extract frame-level visual and text features with dimension $D =$ 768. The model is trained end-to-end using AdamW [17] (learning rate $2 \times 1 0 ^ { - 4 }$ , weight decay 0.05), consistent across benchmarks. For losses, we set $\alpha = 2 . 0 , m = 1 . 0 $ , and $\lambda _ { r e g } = 2 . 0$

## 5.2 Main Results

Baselines. We compare VGD-Agg with five representative VTG methods: RaTSG [2], UVCOM [34], CG-DETR [21], SnAG [22], and ICQ [41]. The first four baselines cover both proposal-based and transformer-based paradigms. For adaptation, we follow the common practice in mainstream MLLMs by employing Attention-Fusion. Specifically, image queries are encoded by CLIP and projected into the textual embedding space, then concatenated with text tokens along the sequence dimension. The resulting unified multimodal sequence is processed by the model’s original text encoder, where cross-modal interaction is achieved implicitly through attention. For ICQ, we implement its MQ-Sum strategy, which leverages an MLLM [1] to jointly summarize the reference image and the textual query into a distilled description, which then serves as the final query for the grounding backbone.

Table 2: Performance comparison on the IDVTG-Gym dataset.
<table><tr><td rowspan="2">Method</td><td colspan="3">Overall</td><td colspan="3">Sub-act.</td><td colspan="3">Holistic</td></tr><tr><td> $R _ { 0 . 5 } ^ { 1 }$ </td><td> $R _ { 0 . 7 } ^ { 1 }$ </td><td>mIoU</td><td> $R _ { 0 . 5 } ^ { 1 }$ </td><td> $R _ { 0 . 7 } ^ { 1 }$ </td><td>mIoU</td><td> $R _ { 0 . 5 } ^ { 1 }$ </td><td> $R _ { 0 . 7 } ^ { 1 }$ </td><td>mIoU</td></tr><tr><td>RaTSG [2]</td><td>18.45</td><td>15.99</td><td>16.96</td><td>6.64</td><td>3.98</td><td>5.54</td><td>26.28</td><td>23.50</td><td>27.64</td></tr><tr><td>UVCOM [34]</td><td>18.07</td><td>15.00</td><td>17.32</td><td>5.20</td><td>2.43</td><td>6.86</td><td>25.07</td><td>22.54</td><td>22.99</td></tr><tr><td>CG-DETR [21]</td><td>27.96</td><td>23.36</td><td>26.99</td><td>6.64</td><td>2.77</td><td>10.14</td><td>44.47</td><td>39.86</td><td>40.29</td></tr><tr><td>ICQ [41]</td><td>46.12</td><td>42.24</td><td>40.24</td><td>43.92</td><td>35.73</td><td>34.10</td><td>47.87</td><td>47.17</td><td>45.06</td></tr><tr><td>SnAG [22]</td><td>53.64</td><td>48.82</td><td>46.73</td><td>51.66</td><td>41.37</td><td>40.32</td><td>54.83</td><td>54.05</td><td>51.46</td></tr><tr><td>VGD-Agg (Ours)</td><td>61.83</td><td>56.53</td><td>54.17</td><td>55.53</td><td>44.91</td><td>43.29</td><td>66.58</td><td>65.36</td><td>62.53</td></tr></table>

Table 3: Performance comparison on the IDVTG-InternVid and Web datasets.
<table><tr><td rowspan="2">Method</td><td colspan="3">IDVTG-InternVid</td><td colspan="3">Web</td></tr><tr><td> $R _ { 0 . 5 } ^ { 1 }$ </td><td> $R _ { 0 . 7 } ^ { 1 }$ </td><td>mIoU</td><td> $R _ { 0 . 5 } ^ { 1 }$ </td><td> $R _ { 0 . 7 } ^ { 1 }$ </td><td>mIoU</td></tr><tr><td>RaTSG [2]</td><td>30.58</td><td>19.71</td><td>30.91</td><td>7.96</td><td>5.79</td><td>8.53</td></tr><tr><td>UVCOM [34]</td><td>30.11</td><td>18.89</td><td>30.07</td><td>7.70</td><td>5.62</td><td>8.10</td></tr><tr><td>CG-DETR [21]</td><td>44.81</td><td>31.62</td><td>43.80</td><td>13.86</td><td>5.91</td><td>18604</td></tr><tr><td>ICQ [41]</td><td>31.04</td><td>25.39</td><td>29.90</td><td>7.84</td><td>4.53</td><td>8.31</td></tr><tr><td>SnAG [22]</td><td>44.26</td><td>35.46</td><td>41.70</td><td>16.48</td><td>9.06</td><td>16.53</td></tr><tr><td>VGD-Agg (Ours)</td><td>51.21</td><td>41.24</td><td>48.30</td><td>21.99</td><td>13.25</td><td>22.34</td></tr></table>

Performance on IDVTG-Gym (Table 2). VGD-Agg achieves the best performance across all splits, demonstrating its ability to leverage visual guidance for resolving complex ambiguities. Specifically, in the Sub-act split, our model efectively utilizes the reference image as a precise anchor to distinguish fine-grained, atomic movements performed by similarly dressed athletes. Furthermore, on the Holistic split, the method exhibits robust capability in capturing long-range event dependencies, accurately grounding complete routines despite their extended temporal duration.

Performance on IDVTG-InternVid and OOD Web Dataset (Tables 3). IDVTG-InternVid represents a challenging open-world setting with diverse video content and free-form multimodal queries. VGD-Agg achieves superior performance, significantly outperforming strong baselines and demonstrating that our method efectively scales beyond constrained domains. To further assess robustness, we evaluate the model trained on IDVTG-InternVid directly on the OOD Web dataset without fine-tuning. Despite the domain shift, VGD-Agg maintains a clear lead, indicating that it learns transferable visual-semantic correspondences rather than overfitting to dataset-specific biases. These results validate that explicitly leveraging reference images enables the model to accurately distinguish target instances amid complex, unconstrained entities and actions, and to generalize efectively to unseen data distributions.

## 5.3 Ablation Studies

We conduct comprehensive ablation studies on IDVTG-Gym to validate the contributions of each component.

Table 4: Ablation on branch-level architecture.
<table><tr><td>Model Variant</td><td> $R _ { 0 . 5 } ^ { 1 }$ </td><td> $R _ { 0 . 7 } ^ { 1 }$ </td><td>mIoU</td></tr><tr><td>w/o Slow Branch</td><td>53.64</td><td>48.82</td><td>46.73</td></tr><tr><td>w/o Fast Branch</td><td>59.41</td><td>54.40</td><td>51.32</td></tr><tr><td>w/o Aggregation</td><td>61.12</td><td>56.76</td><td>53.50</td></tr><tr><td>VGD-Agg (Full)</td><td>61.83</td><td>56.53</td><td>54.17</td></tr></table>

Table 5: Ablation on compare token and depress value.
<table><tr><td>Model Variant</td><td> $R _ { 0 . 5 } ^ { 1 }$ </td><td> $R _ { 0 . 7 } ^ { 1 }$ </td><td>mIoU</td></tr><tr><td>w/o Compare Token</td><td>55.63</td><td>50.76</td><td>48.49</td></tr><tr><td>Global Learnable Token</td><td>60.36</td><td>55.53</td><td>52.62</td></tr><tr><td>w/o Similarity Loss</td><td>60.50</td><td>55.58</td><td>52.67</td></tr><tr><td>w/o Image in Compare Token Generation</td><td>60.64</td><td>56.20</td><td>53.26</td></tr><tr><td>w/o Text in Depress Value Generation</td><td>59.34</td><td>54.26</td><td>51.94</td></tr><tr><td>VGD-Agg (Full)</td><td>61.83</td><td>56.53</td><td>54.17</td></tr></table>

Table 6: Ablation on diferent input fusion methods.
<table><tr><td>Input Modality</td><td> $R _ { 0 . 5 } ^ { 1 }$ </td><td> $R _ { 0 . 7 } ^ { 1 }$ </td><td>mIoU</td></tr><tr><td>Text-only</td><td>46.78</td><td>41.96</td><td>40.74</td></tr><tr><td>Attention Fusion</td><td>53.64</td><td>48.82</td><td>46.73</td></tr><tr><td>BLIP-2 Fusion [14]</td><td>54.07</td><td>48.86</td><td>46.75</td></tr><tr><td>ICQ Fusion [41]</td><td>46.12</td><td>42.24</td><td>40.24</td></tr><tr><td> $\mathbf { T e x t } + \mathbf { I m a g e } + \mathbf { V G D - A g g } \left( \mathbf { O u r s } \right)$ </td><td>61.83</td><td>56.53</td><td>54.17</td></tr></table>

Impact ofBranch-Level Architecture (Table 4). The Slow Branch dominates performance by capturing spatial semantics, while the Fast Branch adds essential temporal context; removing either significantly degrades mIoU. Notably, the full VGD-Agg surpasses the “w/o Aggregation" baseline, demonstrating our mechanism’s ability to efectively synergize spatial and temporal cues.

Efectiveness of Compare Token and Depress Value (Table 5). Removing the token and replacing the softmax competition with a plain softmax over frame similarities (w/o Compare Token) causes a sharp performance drop, proving the necessity of an adaptive baseline to suppress distractors. Furthermore, using a Global Learnable Token or removing the auxiliary supervision (w/o Similarity Loss) leads to inferior results, highlighting that the Compare Token must be video-specific and explicitly optimized as a decision boundary. Finally, generating the Compare Token without image context (w/o Image) or generating the Depress Value without text context (w/o Text) also degrades accuracy, indicating that the Compare Token relies on image queries to determine if a video frame includes the image query, while the Depress Value is based on text queries, reflecting video events not related to the text.

Ablation on Input Modalities and Fusion Mechanisms (Table 6). We conduct ablation studies to examine the impact of imageconditioned VTG and evaluate the efectiveness of the proposed VGD-Agg module. Incorporating visual features consistently improves over the text-only baseline, indicating that reference images provide complementary cues that help resolve semantic ambiguity. We compare several fusion strategies: Attention Fusion, used in baseline adaptation, already delivers substantial gains by explicitly introducing visual guidance. BLIP-2 Fusion, which leverages a pretrained BLIP-2 backbone to combine text and image features, pro vides only marginal improvements, suggesting that general-purpose multimodal pretraining does not guarantee the fine-grained align ment needed for temporal grounding. ICQ Fusion, which converts image content into textual descriptions using an MLLM, underperforms, highlighting the information loss introduced by text-based compression. Finally, our full VGD-Agg model achieves the best performance, confirming that simply adding images is insuficient; efective image-conditioned grounding requires a dedicated fusion mechanism that aligns and aggregates visual cues with video features in a task-specific manner.

Table 7: Ablation study on diferent ambiguity types on IDVTG-InternVid.
<table><tr><td>Method</td><td colspan="3">Full Amb.</td><td colspan="3">Text Amb. Only</td><td colspan="3">Img Amb. Only</td></tr><tr><td></td><td> $R _ { 0 . 5 } ^ { 1 }$ </td><td> $R _ { 0 . 7 } ^ { 1 }$ </td><td>mIoU</td><td> $R _ { 0 . 5 } ^ { 1 }$ </td><td>R0.7</td><td>mIoU</td><td>R0.5</td><td>R0.7</td><td>mIoU</td></tr><tr><td>RaTSG [2]</td><td>19.61</td><td>14.96</td><td>20.87</td><td>23.47</td><td>20.88</td><td>24.15</td><td>26.95</td><td>21.18</td><td>27.03</td></tr><tr><td>UVCOM [34]</td><td>19.49</td><td>14.79</td><td>20.55</td><td>23.13</td><td>19.97</td><td>23.67</td><td>26.73</td><td>20.89</td><td>26.01</td></tr><tr><td>CG-DETR [21]</td><td>30.29</td><td>19.61</td><td>31.06</td><td>37.88</td><td>26.67</td><td>38.60</td><td>37.26</td><td>23.83</td><td>37.88</td></tr><tr><td>ICQ [41]</td><td>20.54</td><td>15.77</td><td>20.50</td><td>25.70</td><td>21.45</td><td>25.15</td><td>27.48</td><td>21.46</td><td>27.29</td></tr><tr><td>SnAG [22]</td><td>30.39</td><td>23.34</td><td>30.34</td><td>41.94</td><td>34.24</td><td>39.59</td><td>38.19</td><td>27.53</td><td>36.40</td></tr><tr><td>VGD-Agg (Ours)</td><td>37.34</td><td>28.42</td><td>36.69</td><td>53.27</td><td>43.52</td><td>49.97</td><td>41.12</td><td>30.52</td><td>40.12</td></tr></table>

Table 8: Ablations on the robustness to image queries on IDVTG-InternVid.
<table><tr><td rowspan="3">Method</td><td colspan="3">Clean</td><td colspan="3">Brightness</td><td colspan="3">Low Resolution</td></tr><tr><td>R0.5</td><td>R0.7</td><td>mIoU</td><td> $R _ { 0 . 5 } ^ { 1 }$ </td><td>R0.7</td><td>mIoU</td><td>R0.5</td><td> $R _ { 0 . 7 } ^ { 1 }$ </td><td>mIoU</td></tr><tr><td>RaTSG [2]</td><td>30.58</td><td>19.71</td><td>30.91</td><td>29.04</td><td>18.87</td><td>28.99</td><td>29.98</td><td>18.35</td><td>29.31</td></tr><tr><td>UVCOM [34]</td><td>30.11</td><td>18.89</td><td>30.07</td><td>28.76</td><td>18.16</td><td>28.90</td><td>29.83</td><td>18.51</td><td>29.84</td></tr><tr><td>CG-DETR [21]</td><td>44.81</td><td>31.62</td><td>43.80</td><td>44.35</td><td>31.60</td><td>43.36</td><td>44.31</td><td>31.54</td><td>43.27</td></tr><tr><td>ICQ [41]</td><td>31.04</td><td>25.39</td><td>29.90</td><td>29.00</td><td>23.29</td><td>28.02</td><td>30.77</td><td>24.66</td><td>29.54</td></tr><tr><td>SnAG [22]</td><td>44.26</td><td>35.46</td><td>41.70</td><td>43.57</td><td>35.05</td><td>41.18</td><td>43.48</td><td>35.13</td><td>41.15</td></tr><tr><td>Ours</td><td>51.21</td><td>41.24</td><td>48.30</td><td>49.99</td><td>39.79</td><td>47.20</td><td>49.10</td><td>39.33</td><td>46.52</td></tr></table>

Analysis on Diferent Ambiguity Types (Table 7). To analyze the model’s disambiguation ability under diferent conditions, we categorize the ambiguity in the IDVTG-InternVid dataset into two types: Text Ambiguity and Image Ambiguity. Text Ambiguity refers to scenarios where the textual description applies to multiple visual instances (e.g., diferent characters performing the same action), necessitating the reference image to identify the specific target. Image Ambiguity occurs when the reference subject appears in multiple temporal segments performing diferent actions, requiring the textual query to localize the intended event. As Table 7 shows, VGD-Agg performs robustly in both cases, with larger gains under Text Ambiguity. This demonstrates that the Compare Token and Depress Value efectively suppress visually similar distractors. Furthermore, the strong results on Image Ambiguity indicate that our method retains the precise temporal grounding capability needed to distinguish diferent events involving the same subject, without over-relying on visual matching alone.

Robustness to Image Queries (Table 8). In real-world scenarios, user-provided reference images often sufer from unpredictable variations in quality; therefore, it is essential to evaluate the model’s stability against such degradations. We evaluate model robustness against baselines under three image conditions: Clean, Brightness, and Low Resolution. The Clean setting uses manually annotated high-quality images. Brightness varies the brightness of the reference images, while Low Resolution downsamples the image by a factor of 4 in pixel count. As shown in Table 8, our method consistently achieves the best performance across all settings. Under Brightness and Low Resolution, all methods remain relatively stable, indicating a certain degree of robustness to brightness variations and spatial degradation. Our method still achieves the best results in both cases, showing that it can preserve reliable visual grounding under degraded image quality.

![](images/e565a6eb4c01f50792296172e65a643d245f5f96630cbf7bbe2b380f6ef73998.jpg)  
Figure 5: Visualization of the learned attention weights.

## 5.4 Qualitative Results

To further validate the efectiveness ofour Vision-Assisted Disambiguation module, we visualize the attention weights of the Compare Token and Depress Value in Figure 5. We consider a challenging scenario where two gymnasts perform the same vault activity, with one serving as the visual distractor and the other as the target. The attention weights of the Compare Token show clear responses to the high-motion regions of both gymnasts, suggesting that it captures visually ambiguous areas and constructs informative hard negative prototypes for target discrimination. In contrast, the Depress Value mainly focuses on text-irrelevant segments, such as the judges in the background, indicating its ability to identify and suppress irrelevant visual content. Together, these visualizations demonstrate that our module efectively distinguishes target-related cues from both hard visual distractors and background noise, thereby facilitating more accurate text-guided grounding.

## 6 Conclusion

In this work, we study Image Disambiguated Video Temporal Grounding (ID-VTG), a setting where text alone is insuficient to resolve multi-instance ambiguity and visual cues become essential for identifying the intended event. We introduce two ID-VTG datasets specifically designed to expose such ambiguities and propose a Visually-Guided Disambiguation Aggregation framework that leverages the reference image to highlight visually consistent regions while suppressing distractors. Experiments validate the efectiveness of our method and highlight the importance of visual disambiguation for reliable temporal grounding in real-world videos.

## Acknowledgments

This work was supported by the grants from the National Natural Science Foundation of China 62372014, Beijing Nova Program and ZTE Industry-University-Institute Cooperation Funds under Grant No.IA20240723020-PO0015.

## References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. 2025. Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631 (2025).

[2] Jianfeng Dong, Xiaoman Peng, Daizong Liu, Xiaoye Qu, Xun Yang, Cuizhu Bao, and Meng Wang. 2024. Temporal sentence grounding with relevance feedback in videos. Advances in Neural Information Processing Systems 37 (2024), 43107–43132.

[3] Linkang Du, Zhou Su, and Xinyi Yu. 2025. Dataset Copyright Auditing for Large Models: Fundamentals, Open Problems, and Future Directions. ZTE COMMUNI-CATIONS 23, 3 (2025), 38–47.

[4] Hong Gao, Jingyu Wu, Xiangkai Xu, Kangni Xie, Yunchen Zhang, Bin Zhong, Xurui Gao, and Min-Ling Zhang. 2026. Omniground: A comprehensive spatiotemporal grounding benchmark for real-world complex scenarios. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 17588– 17597.

[5] Hong Gao, Xiangkai Xu, Bin Zhong, Junjie Yin, Fangyu Kang, Yutong Xu, Xiugang Dong, Xurui Gao, and Min-Ling Zhang. 2026. SARL-STG: A Spatially Aware Reinforcement Learning Framework for Refining MLLMs in Spatio-Temporal Video Grounding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 24630–24639.

[6] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. 2017. Tall: Tempora activity localization via language query. In Proceedings of the IEEE international conference on computer vision. 5267–5275.

[7] Wei GU, Shuo SHAO, Lingtao ZHOU, Zhan QIN, and Kui REN. 2025. Poison-only and targeted backdoor attack against visual object tracking. ZTE COMMUNICA-TIONS 23, 3 (2025), 3–14.

[8] Wu Hao, Guo Junqi, and Bie Rongfang. 2025. Multi-Use Learning Instance for Optimized Image Retrieval. Chinese Journal ofElectronics 34, 3 (2025), 1002–1005.

[9] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. 2017. Localizing Moments in Video with Natural Language. In Proceedings ofthe IEEE International Conference on Computer Vision (ICCV).

[10] Jiabo Huang, Hailin Jin, Shaogang Gong, and Yang Liu. 2022. Video Activity Localisation with Uncertainties in Temporal Boundary. In Proceedings of the European Conference on Computer Vision (ECCV).

[11] Tarun Khurana, Artem Molchanov, Pavlo Molchanov, Ren Yang, Nagesh Adluru, Jan Kautz, Sachin Raj, and Aneeshan Sain. 2023. MINOTAUR: Multi-task Video Grounding from Multimodal Queries. arXiv preprint arXiv:2309.13837 (2023). https://arxiv.org/abs/2309.13837

[12] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. 2017. Dense-Captioning Events in Videos. In Proceedings of the IEEE International Conference on Computer Vision (ICCV). https://openaccess.thecvf.com/content\_ICCV\_2017/html/Krishna\_Dense Captioning\_Events\_in\_ICCV\_2017\_paper.html

[13] Jie Lei, Tamara L Berg, and Mohit Bansal. 2021. Detecting Moments and Highlights in Videos via Natural Language Queries. In Advances in Neural Information Processing Systems, M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (Eds.), Vol. 34. Curran Associates, Inc., 11846–11858. https://proceedings.neurips.cc/paper\_files/paper/2021/file/ 62e0973455fd26eb03e91d5741a4a3bb-Paper.pdf

[14] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. In Proceedings of the 40th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 202), Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (Eds.). PMLR, 19730–19742. https://proceedings.mlr.press/ v202/li23q.html

[15] Yang Liu, Minghang Zheng, Qingchao Chen, Shaogang Gong, and Yuxin Peng. 2026. Large-Scale Pre-Trained Models Empowering Phrase Generalization in Temporal Sentence Localization. International Journal ofComputer Vision 134, 2 (2026), 53.

[16] Zhihang Liu, Jun Li, Hongtao Xie, Pandeng Li, Jiannan Ge, Sun-Ao Liu, and Guoqing Jin. 2024. Towards Balanced Alignment: Modal-Enhanced Semantic Modeling for Video Moment Retrieval. In AAAI. 3855–3863.

[17] Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017).

[18] Dezhao Luo, Jiabo Huang, Shaogang Gong, Hailin Jin, and Yang Liu. 2023. Towards Generalisable Video Moment Retrieval: Visual-Dynamic Injection to Image-Text Pre-Training. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

[19] Dezhao Luo, Jiabo Huang, Shaogang Gong, Hailin Jin, and Yang Liu. 2024. Zero-Shot Video Moment Retrieval From Frozen Vision-Language Models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). 5464–5473.

[20] Wentao Mo, Qingchao Chen, Yuxin Peng, Siyuan Huang, and Yang Liu. 2025. Advancing 3D Scene Understanding with MV-ScanQA Multi-View Reasoning Evaluation and TripAlign Pre-training Dataset. In Proceedings ofthe 33rd ACM International Conference on Multimedia. 12973–12980.

[21] WonJun Moon, Sangeek Hyun, SuBeen Lee, and Jae-Pil Heo. 2024. Correlation-Guided Query-Dependency Calibration for Video Temporal Grounding. arXiv:2311.08835 [cs.CV] https://arxiv.org/abs/2311.08835

[22] Fangzhou Mu, Sicheng Mo, and Yin Li. 2024. SnAG: Scalable and Accurate Video Grounding. In Proceedings ofthe IEEE/CVFConference on Computer Vision and Pattern Recognition (CVPR). https://openaccess.thecvf.com/content/CVPR2024/html/ Mu\_SnAG\_Scalable\_and\_Accurate\_Video\_Grounding\_CVPR\_2024\_paper.htm

[23] Jonghwan Mun, Minsu Cho, and Bohyung Han. 2020. Local-global video-text interactions for temporal grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10810–10819.

[24] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.

[25] Michaela Regneri, Marcus Rohrbach, Dominikus Wetzel, Stefan Thater, Bernt Schiele, and Manfred Pinkal. 2013. Grounding Action Descriptions in Videos. In Transactions ofthe Association for Computational Linguistics (TACL).

[26] Dian Shao, Yue Zhao, Bo Dai, and Dahua Lin. 2020. FineGym: A Hierarchical Video Dataset for Fine-grained Action Understanding. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

[27] Qiuhong SHEN, Zijin YANG, Jun JIANG, Weiming ZHANG, and Kejiang CHEN. 2025. StegoAgent StegoAgent: A Generative Steganography A Generative Steganography Framework Based on GUI Agents. ZTE COMMUNICATIONS 23, 3 (2025), 48–58.

[28] Mattia Soldan, Mengmeng Xu, Sisi Qu, Jesper Tegner, and Bernard Ghanem. 2021. VLG-Net: Video-language graph matching network for video grounding. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 3224– 3234.

[29] Zongheng Tang, Yue Liao, Si Liu, Guanbin Li, Xiaojie Jin, Hongxu Jiang, Qian Yu, and Dong Xu. 2021. Human-centric Spatio-Temporal Video Grounding With Visual Transformers. arXiv:2011.05049 [cs.CV] https://arxiv.org/abs/2011.05049

[30] Google Gemini Team. 2025. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities. https://arxiv.org/abs/2507.06261

[31] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2023. Attention Is All You Need. arXiv:1706.03762 [cs.CL] https://arxiv.org/abs/1706.03762

[32] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, Conghui He, Ping Luo, Ziwei Liu, Yali Wang, Limin Wang, and Yu Qiao. 2024. InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation. arXiv:2307.06942 [cs.CV] https://arxiv.org/abs/2307.06942

[33] Zhenzhi Wang, Limin Wang, Tao Wu, Tianhao Li, and Gangshan Wu. 2022. Negative sample matters: A renaissance of metric learning for temporal grounding. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 36. 2613–2623.

[34] Yicheng Xiao, Zhuoyan Luo, Yong Liu, Yue Ma, Hengwei Bian, Yatai Ji, Yujiu Yang, and Xiu Li. 2023. Bridging the Gap: A Unified Video Comprehension Framework for Moment Retrieval and Highlight Detection. CoRR abs/2311.16464 (2023).

[35] Yicheng Xiao, Zhuoyan Luo, Yong Liu, Yue Ma, Hengwei Bian, Yatai Ji, Yujiu Yang, and Xiu Li. 2024. Bridging the gap: A unified video comprehension framework for moment retrieval and highlight detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18709–18719.

[36] Dejie Yang, Zijing Zhao, and Yang Liu. 2025. Ar-vrm: Imitating human motions for visual robot manipulation with analogical reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 6818–6827.

[37] Jie Yang, Miao Ma, Yutong Li, and Zhao Pei. 2025. VQALS: A Video Question Answering Method in Low-Light Scenes Based on Illumination Correction and Feature Enhancement. Chinese Journal ofElectronics 34, 4 (2025), 1300–1308.

[38] Haobo Yuan, Xiangtai Li, Tao Zhang, Yueyi Sun, Zilong Huang, Shilin Xu, Shunping Ji, Yunhai Tong, Lu Qi, Jiashi Feng, and Ming-Hsuan Yang. 2025. Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and

Videos. arXiv pre-print (2025).

[39] Peng Yuxin, Wang Zishuo, Li Geng, Zheng Xiangtian, Yin Sibo, and He Hulingxiao. 2026. A Survey on Fine-Grained Multimodal Large Language Models. Chinese Journal of Electronics 35, 2 (2026), 771–803.

[40] Chen-Lin Zhang, Jianxin Wu, and Yin Li. 2022. ActionFormer: Localizing Moments of Actions with Transformers. In European Conference on Computer Vision (LNCS, Vol. 13664). 492–510.

[41] Gengyuan Zhang, Mang Ling Ada Fok, Jialu Ma, Yan Xia, Daniel Cremers, Philip Torr, Volker Tresp, and Jindong Gu. 2025. Localizing Events in Videos with Multimodal Queries. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 3339–3351. doi:10.1109/CVPR52734.2025.00317

[42] Songyang Zhang, Houwen Peng, Jianlong Fu, and Jiebo Luo. 2020. Learning 2d temporal adjacent networks for moment localization with natural language. In Proceedings ofthe AAAIConference on Artificial Intelligence, Vol. 34. 12870–12877.

[43] Zhu Zhang, Zhou Zhao, Yang Zhao, Qi Wang, Huasheng Liu, and Lianli Gao. 2020. Where Does It Exist: Spatio-Temporal Video Grounding for Multi-Form Sentences. arXiv:2001.06891 [cs.CV] https://arxiv.org/abs/2001.06891

[44] Minghang Zheng, Xinhao Cai, Qingchao Chen, Yuxin Peng, and Yang Liu. 2024. Training-free Video Temporal Grounding usingLarge-scale Pre-trained Models. In Proceedings ofthe European Conference on Computer Vision (ECCV).

[45] Minghang Zheng, Shaogang Gong, Hailin Jin, Yuxin Peng, and Yang Liu. 2023. Generating Structured Pseudo Labels for Noise-resistant Zero-shot Video Sentence Localization. In Annual Meeting ofthe Association for Computational Linguistics.

[46] Minghang Zheng, Yanjie Huang, Qingchao Chen, and Yang Liu. 2022. Weakly Supervised Video Moment Localization with Contrastive Negative Sample Mining. In Proceedings of the AAAI Conference on Artificial Intelligence.

[47] Minghang Zheng, Yanjie Huang, Qingchao Chen, Yuxin Peng, and Yang Liu. 2022. Weakly Supervised Temporal Sentence Grounding with Gaussian-based Contrastive Proposal Learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

[48] Minghang Zheng, Yanjie Huang, Qingchao Chen, Yuxin Peng, and Yang Liu. 2025. Weakly and Single-Frame Supervised Temporal Sentence Grounding With Gaussian-Based Contrastive Proposal Learning. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025).

[49] Minghang Zheng, Sizhe Li, Qingchao Chen, Yuxin Peng, and Yang Liu. 2023. Phrase-level Temporal Relationship Mining for Temporal Sentence Localization. In Proceedings ofthe AAAI Conference on Artificial Intelligence.

[50] Minghang Zheng, Yuxin Peng, Benyuan Sun, Yi Yang, and Yang Liu. 2025. Hierarchical Event Memory for Accurate and Low-latency Online Video Temporal Grounding. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 21589–21599.

[51] Minghang Zheng, Zihao Yin, Yi Yang, Yuxin Peng, and Yang Liu. 2026. OmniVTG: A Large-Scale Dataset and Training Paradigm for Open-World Video Temporal Grounding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 24620–24629.

[52] Minghang Zheng, Zihao Yin, Yi Yang, Yuxin Peng, and Yang Liu. 2026. Temporal-Aware Reasoning Optimization for Video Temporal Grounding. In International Conference on Machine Learning.