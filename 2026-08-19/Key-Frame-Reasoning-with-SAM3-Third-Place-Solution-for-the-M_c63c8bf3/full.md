# Key-Frame Reasoning with SAM3: Third Place Solution for the MeViS-Text Track of the 8th LSVOS Challenge

Ce Bian<sup>1</sup>, Xusheng He<sup>1</sup>, Jinrong Zhang<sup>1</sup>, Canyang Wu<sup>1</sup>, Xianjing Han<sup>2</sup>, and Jianlong Wu<sup>1,3</sup>

<sup>1</sup> Harbin Institute of Technology, Shenzhen 2 Nanyang Technological University <sup>3</sup> Shenzhen Loop Area Institute Team: HITsz-Dragon

Abstract. This report presents a two-stage, training-free solution for the MeViS-Text track of the 8th LSVOS Challenge. The task requires a model to localize and segment the object specified by a natural-language expression throughout a video. Such expressions often depend on temporal cues, including actions, interactions, directions, and relative positions. Our first stage uses Gemini-3.1 Pro via API to decompose a video-level event into instance-level targets, select a key frame for each target, and generate a discriminative description aligned with that frame. In the second stage, SAM3-agent produces a pixel-level seed mask on the selected frame, and the SAM3 video tracker propagates the mask bidirectionally through the video. Valid instances are grounded and propagated independently before their frame-wise masks are merged. All local SAM3 processing runs on a single NVIDIA GeForce RTX 4090 without taskspecific training or model ensembling. Our method ranked third on the challenge test set, obtaining J&F, J, F, N-acc., T-acc., and Final scores of 0.761, 0.7367, 0.7852, 0.8333, 0.9755, and 0.856593, respectively.

Keywords: Referring video object segmentation · Motion-centric expressions · Multimodal large language models · SAM3 · Video tracking

## 1 Introduction

Video object segmentation (VOS) aims to segment target regions throughout a video while maintaining temporal consistency under appearance changes, occlusions, and background distractors [7, 9]. Referring video object segmentation (RVOS) further requires the target to be determined from a natural-language expression [3,4]. RVOS is therefore a joint problem of visual grounding, language understanding, temporal reasoning, and pixel-level mask prediction.

The MeViS-Text track focuses on motion-centric referring expressions. A target can be specified through its motion direction, action, interaction with another object, relative position, state, or an exclusion relation. The model must compare observations across time to determine whether an instance truly satisfies the expression rather than relying only on its category or appearance in a single frame. Occlusions, viewpoint changes, cluttered backgrounds, and similarlooking instances further complicate both target identification and temporal segmentation.

A practical strategy is to interpret the expression and video content with a vision-language model before applying a segmentation or tracking model [5]. However, the temporal information in a complete motion expression may not align directly with a static frame. In addition, special tokens used as intermediate representations can weaken fine-grained instance attributes and spatial relations [1]. The central challenge is thus to turn a video-level event into a clear and executable image-grounding task. Prompt-conditioned video segmentation models provide a practical basis for the propagation stage: SAM2 established a general framework for propagating prompted masks across videos [8], while SAM3 further supports concept-conditioned segmentation and tracking [2].

We address this challenge with a two-stage training-free framework. Gemini-3.1 Pro identifies valid instances at the video level, selects the clearest key frame for each instance, and produces a discriminative description matched to that frame. SAM3-agent then generates a pixel-level seed mask, which is propagated through the video by the SAM3 tracker [2]. Multiple valid instances are processed independently and merged only after propagation.

Our main contributions are as follows:

1. We develop a two-stage, training-free RVOS framework that connects videolevel event understanding with pixel-level video segmentation.

2. We use Gemini-3.1 Pro to decompose complex events into instance-level targets and jointly determine key frames and discriminative descriptions, reducing ambiguity among similar instances.

3. We generate pixel-level seed masks with SAM3-agent and propagate them bidirectionally with the SAM3 video tracker. Gemini-3.1 Pro is accessed via API, and all local SAM3 processing runs on a single RTX 4090. The method ranks third in the challenge.

## 2 Solution

## 2.1 Overall Framework

We adopt a two-stage framework. The first stage performs event understanding, while the second stage comprises single-frame localization and video propagation. First, we use a multimodal large language model to analyze the video and its motion event expression, decomposing the original query into one or more instance-level targets. For each target, the model selects the key frame most suitable for localization and generates a discriminative description conditioned on the visual content of that frame. The key frame and its description are then fed into a SAM3-agent, which directly produces a pixel-level seed mask, and the

SAM3 video tracker propagates this mask across the entire video. For expressions that contain multiple valid instances, each instance is localized and tracked separately, and the resulting predictions are merged into an event-level result.

This design decouples cross-frame event understanding from pixel-level segmentation. Motion expressions typically encode information about actions, directions, interactions, and relative positions, which is dificult to use directly as a single-frame segmentation prompt. After the first stage, each target has a well-defined instance identity, an unambiguous reference frame, and a description that matches the corresponding frame, providing more stable inputs for the subsequent segmentation stage. Meanwhile, the second stage directly produces a seed mask instead of relying on bounding boxes as the primary intermediate representation, so that object contours and spatial details can be preserved more completely for video propagation.

## 2.2 Two-Stage Processing Pipeline

Stage 1: Event Decomposition via Key-Frame Reasoning For each video and its target event, we use Gemini-3.1 Pro to analyze the video content holistically. This stage does not simply restate the event; instead, it identifies the target instances referred to by the expression and organizes them into independently processable records. Other objects that merely help describe actions or relations in the expression are treated only as localization cues and are not included in the final segmentation targets. If multiple distinct physical instances satisfy the event, a separate record is generated for each; if the same instance appears at multiple moments, it is processed only once.

After instance identification, the model selects, for each target, the key frame most suitable for single-frame localization. The selection primarily considers whether the target is clearly visible, whether its contour is complete, whether occlusion is low, and whether it can be distinguished from similar instances in the frame. For longer videos, we uniformly sample the input frames and record the correspondence between the sampling order and the original frame indices, so that the positions returned by the model can be mapped back accurately to the real video frames.

Then, based on the selected key frame, Gemini-3.1 Pro generates a discriminative description for each instance. The description includes not only the target category but also the visible attributes of the target in the current frame, such as color, texture, shape, and clothing, as well as its spatial relations to surrounding objects or image regions. For multiple instances of the same category, the model describes them separately using diferent appearance and location cues, ensuring, as much as possible, that each description points to exactly one target within the corresponding key frame. Through this process, motion expressions that originally rely on cross-frame behavioral understanding are converted into a set of image-localization tasks with clearly defined targets, key frames, and dedicated descriptions.

The first stage stores the name, key frame, and discriminative description of each instance in a structured form. If the model determines that no target satisfying the event exists in the video, it returns an empty result. This output is passed directly to the second stage, so that the SAM3-agent does not need to re-understand the full motion event and can focus on instance-level localization within the key frame.

Stage 2: SAM3-agent Localization and Temporal Propagation Given the key frames and discriminative descriptions, we run a SAM3-agent on each target instance separately. This process accomplishes single-frame target localization through multi-round interaction between a multimodal large language model and the SAM3 tool. In our implementation, Gemini-3.1 Pro is responsible for analyzing the key frame and the instance description, planning tool calls, and judging the segmentation results returned by SAM3. Specifically, Gemini-3.1 Pro first generates a text phrase suited to SAM3 based on the target’s characteristics, and SAM3 produces candidate masks accordingly. Gemini-3.1 Pro then checks the candidate’s identity, spatial location, and mask completeness against the original description, and decides the next action.

If the current result fails to localize the target accurately, Gemini-3.1 Pro further inspects the candidate mask, or adjusts the text phrase and invokes SAM3 again, until a satisfactory result is obtained. The phrase used to call SAM3 may be more concise than the initial discriminative description, but the judgment of the candidate mask always relies on the initial description, thereby avoiding a shift in the referred object caused by prompt simplification. When a candidate mask covers the target completely and contains no clearly irrelevant regions, the SAM3-agent confirms it as the seed mask; if no matching target exists in the key frame, it returns an empty mask. Through this multi-round interaction, the SAM3-agent progressively eliminates candidates that do not match the description and mitigates the localization ambiguity caused by similar instances.

Because a bounding box can only roughly enclose the target and tends to include irrelevant regions when targets are in contact, the background is complex, or similar instances are present, we directly use the pixel-level mask output by the SAM3-agent to initialize the video tracker. A pixel-level seed mask provides more accurate object contours and spatial positions, ofering a more reliable initial prompt for subsequent temporal propagation.

Once the seed mask is obtained, we initialize the SAM3 video tracker on the corresponding key frame and propagate it toward both the beginning and the end of the video, thereby covering the complete sequence before and after the key frame. The tracking results are binarized and saved as per-frame masks that correspond one-to-one to the original video frames. Figure 1 shows a representative example of the whole process from key-frame reasoning to full-video segmentation.

![](images/b58d77d38271ca84a8064d3e43c1523d2e176f4d9766ccbf3169774ae4dc2ec2.jpg)  
Fig. 1: Overview of our complete two-stage pipeline on a representative MeViS-Text example, from key-frame reasoning and discriminative-description generation to SAM3- agent grounding and bidirectional mask propagation.

For expressions containing multiple valid instances, we propagate each instance independently according to its key frame, discriminative description, and seed mask, and then merge the instance masks within the same frame to form the final event-level prediction. This reduces mutual interference among diferent targets during initialization and allows each instance to be tracked from its clearest moment.

## 3 Experiments

## 3.1 Challenge Description

We participated in the MeViS-Text track of the 8th Large-scale Video Object Segmentation (LSVOS) Challenge [6]. Given a video and a natural-language expression, the system must output a binary PNG mask for every original video frame. Expressions may specify a simple object or impose constraints involving motion, interaction, direction, relative position, or state. The task consequently requires both referent identification and temporally consistent segmentation.

## 3.2 Implementation Details

The method is used only for inference: no task-specific training or model ensemble is employed. Gemini-3.1 Pro is accessed via API for event analysis, instance decomposition, key-frame selection, discriminative-description generation, tool planning, and candidate-mask evaluation. SAM3 generates candidate masks on key frames, and its video tracker performs temporal propagation.

Table 1: Top-three oficial MeViS-Text test-set results. The proposed method is shown in bold.
<table><tr><td>Rank</td><td>J&amp;F</td><td>J</td><td>F</td><td>N-acc.</td><td>T-acc.</td><td>Final</td></tr><tr><td>1</td><td>0.7922</td><td>0.7657</td><td>0.8187</td><td>0.9444</td><td>0.9877</td><td>0.908134</td></tr><tr><td>2</td><td>0.7163</td><td>0.6894</td><td>0.7432</td><td>0.9722</td><td>0.9755</td><td>0.888004</td></tr><tr><td>3</td><td>0.761</td><td>0.7367</td><td>0.7852</td><td>0.8333</td><td>0.9755</td><td>0.856593</td></tr></table>

Local SAM3 inference and video tracking use one NVIDIA GeForce RTX 4090. We store submission masks as <video\_id>/<expression\_id>/<frame> .png. Missing predictions are replaced with all-zero masks so that every expression has an output mask for every input frame.

## 3.3 Evaluation Metrics

Jaccard index (J) measures region overlap between a prediction and its ground truth, while boundary F-score (F) evaluates boundary accuracy. J&F is their mean. N-acc. evaluates no-target expressions, and T-acc. evaluates target expressions. The Final score combines J&F, N-acc., and T-acc.

## 3.4 Quantitative Results

Our method ranks third on the oficial MeViS-Text test set with a Final score of 0.856593. It obtains a T-acc. of 0.9755, J&F of 0.761, J of 0.7367, F of 0.7852, and N-acc. of 0.8333. Table 1 reports the top three entries. With Gemini-3.1 Pro accessed via API, the local SAM3 components run on a single RTX 4090 without task-specific training or ensembling. These results demonstrate the feasibility of the two-stage pipeline.

## 3.5 Qualitative Results and Discussion

Figure 1 shows an example for the expression “The monkey combing the cat’s fur and removing fleas.” The video contains a foreground monkey, another monkey behind it, and a cat. Stage 1 selects Frame 52 and generates a description that distinguishes the foreground target from the other monkey. SAM3-agent produces candidate masks using the phrase “monkey” and selects the target seed mask according to the full description. The SAM3 video tracker then propagates the mask throughout the video.

The example illustrates how a key frame and a discriminative description can separate similar instances and provide a clear spatial initialization for propagation. The method nevertheless depends on the first-stage decision: a missed instance or unsuitable key frame afects later grounding and tracking. A single key frame can also be insuficient for expressions dominated by complex directional, negative, or behavioral constraints. No-target expressions remain a relative weakness.

## 4 Conclusion

We presented a two-stage, training-free approach for motion-centric RVOS in the MeViS-Text track. Gemini-3.1 Pro performs instance-level event decomposition, key-frame selection, and discriminative-description generation. SAM3-agent then produces pixel-level seed masks, which the SAM3 video tracker propagates bidirectionally through the video. Independently tracked instances are merged into the final prediction.

Gemini-3.1 Pro is accessed via API, while the local SAM3 components run on a single NVIDIA GeForce RTX 4090 without task-specific training or model ensembling. The method ranks third in the 8th LSVOS Challenge MeViS-Text track, achieving J&F of 0.761 and a Final score of 0.856593. The results support the use of strong multimodal event understanding together with pixel-level grounding and temporal propagation for motion-centric RVOS.

## References

1. Bai, Z., He, T., Mei, H., Wang, P., Gao, Z., Chen, J., Liu, L., Zheng, Z., Shou, M.: One token to seg them all: Language instructed reasoning segmentation in videos. In: NeurIPS. vol. 37, pp. 6833–6859 (2024)

2. Carion, N., Gustafson, L., Hu, Y.T., Debnath, S., Hu, R., Suris Coll-Vinent, D., Ryali, C., Alwala, K.V., Khedr, H., Huang, A., et al.: SAM 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719 (2025)

3. Ding, H., Liu, C., He, S., Jiang, X., Loy, C.C.: MeViS: A large-scale benchmark for video segmentation with motion expressions. In: ICCV. pp. 2694–2703 (2023)

4. Ding, H., Liu, C., He, S., Ying, K., Jiang, X., Loy, C.C., Jiang, Y.G.: MeViS: A multi-modal dataset for referring motion expression video segmentation. TPAMI 47(12), 11400–11416 (2025)

5. Jiang, H., Liang, T., Zheng, W.S., Hu, J.F.: Refer-Agent: A collaborative multiagent system with reasoning and reflection for referring video object segmentation. arXiv preprint arXiv:2602.03595 (2026)

6. LSVOS Challenge Organizers: The 8th large-scale video object segmentation challenge. https://lsvos.github.io/ (2026), ECCV 2026 Challenge

7. Pont-Tuset, J., Perazzi, F., Caelles, S., Arbelaez, P., Sorkine-Hornung, A., Van Gool, L.: The 2017 DAVIS challenge on video object segmentation. arXiv preprint arXiv:1704.00675 (2017)

8. Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., et al.: SAM 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714 (2024)

9. Xu, N., Yang, L., Fan, Y., Yue, D., Liang, Y., Yang, J., Huang, T.: YouTube-VOS: A large-scale video object segmentation benchmark. arXiv preprint arXiv:1809.03327 (2018)