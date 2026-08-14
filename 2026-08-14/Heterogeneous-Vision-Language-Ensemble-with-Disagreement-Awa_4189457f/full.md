# Heterogeneous Vision-Language Ensemble with Disagreement-Aware Reranking for Text-Based Person Anomaly Retrieval

Huu-An Vu<sup>1,7\*</sup> , Cam Tu Tran Thi<sup>2,3,7\*</sup> , Thanh Toan Le Ngo<sup>2,3,7\*</sup> , Hoang Vo<sup>3,4,7\*</sup> , Do Trung Hieu<sup>5,7\*</sup> , Hieu Dinh Trung Pham<sup>6</sup> , Khang Minh Le<sup>6</sup>, and Huy Minh Nhat Nguyen<sup>6</sup>

<sup>1</sup> Hanoi University of Science and Technology, Hanoi, Vietnam an.vh225467@sis.hust.edu.vn

2 University of Information Technology, VNU-HCM, Ho Chi Minh City, Vietnam <sup>3</sup> Vietnam National University, Ho Chi Minh City, Vietnam {23521704, 23521603}@gm.uit.edu.vn

<sup>4</sup> Ho Chi Minh city University of Science, Vietnam

23280060@student.hcmus.edu.vn

<sup>5</sup> VinUniversity, Hanoi, Vietnam

<sup>6</sup> Vietnamese-German University, Ho Chi Minh City, Vietnam <sup>7</sup> GenAI4E Lab

Abstract. Text-based person anomaly retrieval aims to retrieve pedestrians exhibiting anomalous behaviors from a large image gallery using natural language descriptions. Compared with conventional text-based person retrieval, this task requires fine-grained reasoning over pedestrian appearance, behaviors, object interactions, and scene context, making robust cross-modal matching significantly more challenging. This paper presents the GENAI4E team’s solution to AI City Challenge 2026 Track 4. Our framework builds upon a strong retrieval backbone and progressively integrates heterogeneous vision-language embedding models through score alignment and iterative ensemble learning, followed by disagreement-aware VLM reranking for ambiguous queries. On the oficial Pedestrian Anomaly Behavior (PAB) benchmark, our approach achieves 90.92% mAP, 85.13% Recall@1, 97.72% Recall@5, and 98.68% Recall@10, demonstrating the efectiveness of combining complementary vision-language representations with selective multimodal reasoning for large-scale text-based person anomaly retrieval.

Keywords: Person Anomaly Retrieval · Vision-Language Models · Multimodal Ensemble · AI City Challenge

## 1 Introduction

Text-based person retrieval (TBPS) [8,10,19,22] aims to retrieve a target individual from a large-scale image gallery using natural language descriptions. Existing approaches primarily rely on appearance cues, such as clothing attributes, body shape, and carried objects, to establish cross-modal correspondence between text and images. More recently, text-based person anomaly retrieval (TBAPS) [18] has emerged as a significantly more challenging extension of this problem, where retrieval is driven by anomalous behaviors rather than static appearance. In addition to recognizing pedestrian identity, TBAPS requires fine-grained understanding of subtle behavioral semantics, distinguishing actions with highly similar visual appearances but fundamentally diferent meanings, e.g., “walking down stairs” versus “falling down stairs.” This substantially increases the dificulty of cross-modal matching and demands stronger vision-language reasoning capabilities.

The AI City Challenge 2026 Track 4 formulates TBAPS on the Pedestrian Anomaly Behavior (PAB) dataset [18], a large-scale benchmark consisting of over one million synthetic image–text training pairs covering 1,000 routine and 1,600 anomalous action categories. The evaluation set contains 1,978 natural-language queries and 36,773 gallery images, including 34,795 distractors, making precise retrieval particularly challenging. Performance is evaluated using Mean Average Precision (mAP), which requires both accurate ranking and robust discrimination among visually similar pedestrian instances.

Our solution builds upon the strong baseline framework of [11], which achieves state-of-the-art performance on the PAB benchmark through three complementary components: (1) the Local-Global Hybrid Perspective (LHP) module that enriches visual representations using probabilistic local and global image transformations; (2) Unified Image-Text (UIT) modeling, jointly optimizing imagetext contrastive (ITC), image-text matching (ITM), masked language modeling (MLM), and masked image modeling (MIM) objectives; and (3) an iterative ensemble strategy that progressively refines retrieval predictions across multiple models. While highly efective, the original ensemble remains limited to a relatively homogeneous set of embedding models and does not fully exploit the complementary representations ofered by recent large-scale vision-language models.

To address this limitation, we extend the baseline by incorporating diverse pre-trained vision-language embedding models, including Voyage Multimodal [15], BGE-VL-v1.5-mmeb [23], and Qwen3-VL-Embedding-8B [1], into the iterative ensemble framework. Since these models independently process the gallery and produce similarity matrices under diferent image orderings, we introduce a score alignment procedure that projects all predictions onto a unified gallery indexing, enabling valid score-level fusion. Furthermore, we propose an Ensemble Disagreement routing mechanism that identifies ambiguous queries based on intermodel disagreement and selectively invokes a computationally expensive VLM cross-encoder reranker (gemini-3.1-flash-lite) only when necessary. The reranked predictions are subsequently integrated with the original retrieval results via Reciprocal Rank Fusion (RRF), yielding a more accurate yet computationally eficient retrieval pipeline.

Experimental results on the AI City Challenge 2026 Track 4 benchmark demonstrate the efectiveness of our approach, achieving 90.92% mAP, 85.13% Recall@1, 97.72% Recall@5, and 98.68% Recall@10.

Our contributions are summarized as follows:

– We adopt the strong TBAPS baseline [11], leveraging its Local-Global Hybrid Perspective (LHP) and Unified Image-Text (UIT) framework for efective cross-modal representation learning.

– We enrich the iterative ensemble with complementary pre-trained visionlanguage embedding models, including Voyage Multimodal, BGE-VL-v1.5- mmeb, and Qwen3-VL-Embedding-8B, enabling more diverse cross-modal representations for anomaly retrieval.

– We propose a score alignment strategy that resolves heterogeneous gallery orderings across diferent embedding models, allowing reliable score-level fusion within the ensemble.

– We introduce an Ensemble Disagreement routing mechanism that selectively performs VLM-based cross-encoder reranking only for uncertain queries, followed by Reciprocal Rank Fusion to produce the final retrieval results.

## 2 Related Work

## 2.1 Vision-Language Embedding Models

Large-scale vision-language pre-training has substantially advanced cross-modal representation learning by aligning images and text within a shared embedding space. CLIP [13] pioneered contrastive vision-language pre-training on web-scale image–text pairs, demonstrating remarkable zero-shot transferability across diverse downstream tasks. Subsequent models have further improved cross-modal understanding through stronger architectures and training paradigms. BEiT-3 [16] formulates images as a “foreign language,” enabling unified multimodal pre-training with a single Transformer backbone, while BLIP-2 [7] bridges frozen vision encoders and large language models through lightweight query transformers. More recently, embedding-oriented models such as Voyage Multimodal [15] and BGE-VL [23] have demonstrated strong retrieval performance, whereas large vision-language models such as Qwen-VL [1] provide enhanced multimodal reasoning capabilities. These models ofer complementary semantic representations, motivating their integration within retrieval ensembles.

## 2.2 Text-Based Person Retrieval and Anomaly Search

Text-based person retrieval (TBPS) aims to retrieve target pedestrians from image galleries using natural language descriptions. Early methods, such as Dualpath CNN [22], learned separate visual and textual representations and aligned them through contrastive learning. More recently, APTM [19] introduced largescale multi-attribute pre-training to improve fine-grained cross-modal correspondence, significantly advancing retrieval performance.

Building upon TBPS, text-based person anomaly search (TBAPS) extends the retrieval objective from appearance matching to behavior understanding. The Pedestrian Anomaly Behavior (PAB) benchmark [18] established the first large-scale dataset for this task, containing over one million synthetic image– text pairs covering 1,000 routine and 1,600 anomalous behaviors. Compared with conventional TBPS datasets, PAB requires models to jointly reason about pedestrian appearance, human actions, object interactions, and surrounding scene context, making retrieval considerably more challenging. The proposed Cross-Modal Pose-aware (CMP) framework and its variants, including PE and IHNM [18], serve as the first baselines for this benchmark. More recently, Nguyen et al. [11] proposed a Hybrid, Unified, and Iterative framework that substantially improves TBAPS performance through hybrid visual augmentation, unified vision-language pre-training objectives, and iterative ensemble learning.

## 2.3 Ensemble Learning for Retrieval

Ensemble learning is a widely adopted strategy for improving retrieval robustness by exploiting the complementary strengths of multiple models. Existing approaches typically combine retrieval scores from independently trained models through weighted averaging or iterative refinement, reducing prediction variance while preserving computational eficiency [4,9,21]. In the text-based person anomaly search (TBAPS) setting, Nguyen et al. [11] introduced an iterative ensemble strategy that progressively refines retrieval scores across multiple models, achieving state-of-the-art performance on the PAB benchmark. Nevertheless, existing ensembles primarily aggregate models with similar architectures and embedding spaces. In contrast, our work explores heterogeneous ensembles that integrate diverse vision-language embedding models with fundamentally diferent representation characteristics.

## 2.4 Cross-Encoder Reranking and Late Fusion

Modern retrieval systems commonly adopt a two-stage retrieve-and-rerank pipeline to balance eficiency and accuracy [12]. Lightweight dual-encoder models first retrieve a candidate set, after which more expressive cross-encoders perform finegrained cross-modal reasoning over the top-ranked results. Recent large visionlanguage models (LVLMs) have demonstrated remarkable zero-shot reasoning ability, making them efective cross-encoder rerankers for complex multimodal retrieval tasks [14]. For example, AnomalyLMM [6] bridges generative knowledge and discriminative retrieval by applying generative LVLMs for person anomaly search. However, applying generative VLM reasoning to every query is computationally expensive and slow. Consequently, our approach difers by employing an adaptive routing strategy that selectively invokes an expensive VLM reranker only for ambiguous queries identified through ensemble disagreement. This mechanism preserves dual-encoder eficiency while solving hard queries. To further combine predictions from heterogeneous retrieval models and the VLM, rank-based aggregation methods such as Reciprocal Rank Fusion (RRF) [3] are widely adopted because they efectively fuse rankings without requiring score normalization or calibration.

![](images/f123620d4af9855e645713715f5daa7b180b23152a9f8191fd3b5cc99b34870c.jpg)  
Fig. 1: Overview of our pipeline: an iterative ensemble of embedding models refines the baseline similarity scores, and VLM rerank fusion produces the final ranking.

## 3 Method

## 3.1 Overview

Our framework follows a coarse-to-fine retrieval paradigm that progressively improves retrieval quality through heterogeneous ensemble learning and selective vision-language reasoning. As illustrated in Fig. 1, we first obtain coarse retrieval scores using a strong dual-encoder retrieval backbone. These predictions are subsequently refined by aggregating multiple heterogeneous vision-language embedding models, each providing complementary semantic representations. Since diferent embedding models may produce similarity matrices under diferent gallery orderings, we introduce a score alignment procedure that projects all similarity scores onto a unified gallery index before fusion. Finally, we employ a disagreement-aware routing mechanism that identifies ambiguous queries based on ensemble consensus and selectively invokes a large vision-language model (VLM) for cross-encoder reranking. The reranked results are then integrated with the original ensemble predictions using Reciprocal Rank Fusion (RRF), producing the final retrieval ranking.

## 3.2 Base Retrieval Framework

We adopt the Hybrid, Unified, and Iterative framework [11] as the foundation of our retrieval system. Specifically, the framework combines the Local-Global Hybrid Perspective (LHP) module with Unified Image-Text (UIT) learning to produce robust cross-modal representations for text-based person anomaly retrieval. Given a textual query q and a gallery image i, the retrieval backbone encodes them into a shared embedding space,

$$
\begin{array} { r } { { \bf z } _ { q } = f _ { q } ( q ) , \qquad { \bf z } _ { i } = f _ { i } ( i ) , } \end{array}\tag{1}
$$

where $f _ { q } ( \cdot )$ and $f _ { i } ( \cdot )$ denote the text and image encoders, respectively. Their similarity is computed using cosine similarity,

et al.

![](images/ef3f6f24fa4e960ac714f8ee427a3b45c7a86b613adfc9dba850ad43b55cc4b2.jpg)  
Fig. 2: Overview of our iterative ensemble. Starting from the initial gallery embedding set $S _ { \mathrm { i n i t } }$ , similarity scores are iteratively fused with those of each embedding model $M _ { i }$ using a weighted combination, where $w _ { i }$ is the weight assigned to $M _ { i }$ . Each step produces a refined embedding set $S _ { i } ,$ and the process yields the final set $S _ { \mathrm { f i n a l } }$ after n steps.

$$
S ( i , q ) = \frac { \mathbf { z } _ { i } ^ { \top } \mathbf { z } _ { q } } { \| \mathbf { z } _ { i } \| _ { 2 } \| \mathbf { z } _ { q } \| _ { 2 } } .\tag{2}
$$

This baseline provides a strong initial ranking that serves as the foundation for the subsequent refinement stages.

## 3.3 Heterogeneous Embedding Ensemble

Although modern vision-language embedding models share the common objective of aligning images and text, they are trained using diferent architectures, objectives, and datasets, resulting in complementary embedding spaces. Instead of relying on a single representation, we leverage multiple heterogeneous embedding models to improve retrieval robustness. An overview of our iterative ensemble is shown in Figure 2.

Let $\mathcal { M } = \{ M _ { 1 } , M _ { 2 } , . . . , M _ { K } \}$ denote the set of embedding models. Each model independently computes a similarity matrix

$$
S ^ { ( k ) } \in \mathbb { R } ^ { N _ { q } \times N _ { g } } ,\tag{3}
$$

where $N _ { q }$ and $N _ { g }$ denote the numbers of queries and gallery images, respectively. Let $S _ { \mathrm { i n i t } }$ denote the initial similarity matrix obtained from the baseline retrieval model. Following the iterative ensemble formulation of [11], the accumulated similarity matrix S is initialized as $S = S _ { \mathrm { { i n i t } } }$ and is progressively updated for each model $M _ { k } \in \mathcal { M }$ as:

$$
S \gets w _ { k } S + ( 1 - w _ { k } ) S ^ { ( k ) } ,\tag{4}
$$

where $w _ { k }$ controls the contribution of the accumulated predictions relative to the newly incorporated embedding model $M _ { k }$ . By iteratively integrating heterogeneous embedding spaces, the ensemble exploits complementary semantic information while preserving the strong retrieval capability of the baseline model.

## 3.4 Score Alignment

Unlike models trained within a unified framework, independently developed embedding models may preprocess or index gallery images diferently, resulting in inconsistent gallery orderings across similarity matrices. Consequently, directly averaging similarity scores from diferent models would incorrectly associate scores with mismatched gallery identities.

To enable valid score-level fusion, we introduce a score alignment procedure that projects every similarity matrix onto a shared gallery indexing. Specifically, for each embedding model, we construct a mapping between its internal gallery ordering and the reference gallery order used by the baseline retrieval system. The similarity matrix is subsequently rearranged according to this mapping before ensemble fusion. This alignment guarantees that similarity scores from diferent embedding models consistently correspond to the same gallery images, enabling reliable heterogeneous ensemble learning.

## 3.5 Disagreement-aware VLM Reranking

While heterogeneous ensembles substantially improve retrieval robustness, dualencoder architectures remain limited in distinguishing fine-grained anomaly semantics, particularly when multiple candidates exhibit highly similar appearance. To address this limitation, we introduce a disagreement-aware routing mechanism that selectively applies computationally expensive vision-language reasoning only when necessary.

Given the retrieval rankings produced by multiple ensemble configurations, we first measure the degree of consensus among their top predictions. Queries with high consensus are considered confident and directly retain the ensemble ranking. Conversely, queries exhibiting substantial disagreement are regarded as ambiguous and are forwarded to a large vision-language model operating as a cross-encoder reranker. Unlike dual-encoder retrieval, the cross-encoder jointly attends to both the textual query and candidate images, enabling more precise reasoning over subtle behavioral diferences and complex semantic descriptions.

This adaptive routing strategy preserves the eficiency of dual-encoder retrieval while reserving expensive cross-modal reasoning for the most challenging retrieval cases.

## 3.6 Final Rank Fusion

The final retrieval results combine predictions from heterogeneous dual-encoder ensembles and the VLM reranker. Since these models produce similarity scores with diferent scales and distributions, direct score averaging is unreliable. Instead, we employ Reciprocal Rank Fusion (RRF), which aggregates ranked lists without requiring score normalization.

Given a set of ranked lists R, the final score of gallery image d is computed as

![](images/8db6827d819e5f4cc3318f4326d904d087b6ea6082c7a440a078427274a58c92.jpg)  
Fig. 3: Our VLM reranking and fusion pipeline: ensemble disagreement filtering identifies hard queries, which are reranked by a VLM $\left( P _ { V L M } \right)$ and fused with the toptwo ensemble rankings $( P _ { p r i m a r y }$ and $P _ { a l t } )$ via RRF.

$$
\mathrm { R R F } ( d ) = \sum _ { r \in { \mathcal R } } \frac { 1 } { k + \mathrm { r a n k } ( r , d ) } ,\tag{5}
$$

where ran $\boldsymbol { \mathrm { k } } ( r , d )$ denotes the ranking position of image d in ranked list $r ,$ and k is a constant that controls the contribution of lower-ranked candidates. The final gallery ranking is obtained by sorting images according to their RRF scores as illustrated by Fig 3. By integrating complementary predictions from heterogeneous embedding models and selective VLM reranking, the proposed framework achieves a more robust retrieval system for text-based person anomaly search.

## 4 Experiments

## 4.1 Dataset

We evaluate all methods on the oficial Pedestrian Anomaly Behavior (PAB) benchmark [18], released for AI City Challenge 2026 Track 4. The benchmark is specifically designed for text-based person anomaly retrieval, where the objective is to retrieve the target pedestrian from a large gallery given a natural language description.

The training set consists of more than one million synthetic image–text pairs covering 1,000 routine behaviors and 1,600 anomalous behaviors. The evaluation set contains 1,978 textual queries and 36,773 gallery images, including 34,795 distractors collected from real-world surveillance scenarios. Compared with conventional text-based person retrieval benchmarks, the large number of visually similar distractors and the subtle distinctions between anomalous behaviors require models to jointly reason about pedestrian appearance, object interactions, and behavioral semantics.

Following the oficial evaluation protocol, retrieval performance is measured using Mean Average Precision (mAP), Recall@1, Recall@5, and Recall@10.

## 4.2 Implementation Details

Our framework is implemented using the PyTorch deep learning framework. Following the baseline of [11], we fine-tune the Local-Global Hybrid Perspective (LHP) module for 3 epochs and the Unified Image-Text (UIT) model for 15 epochs. The resulting retrieval backbone is subsequently used to extract image and text embeddings for the initial retrieval stage.

All training and inference are conducted on a single NVIDIA RTX A6000 GPU with 48GB of VRAM.

Feature extraction is performed using a combination of local GPUs, cloud computing resources, and commercial embedding APIs according to the deployment requirements of individual models. During heterogeneous ensemble learning, similarity matrices from diferent embedding models are first aligned to a common gallery ordering before iterative score fusion. Unless otherwise specified, all hyperparameters, including the ensemble weight schedule and reranking configuration, are determined empirically on the validation set and kept fixed throughout all experiments.

## 4.3 Ablation Studies

To better understand the contribution of each component in our framework, we conduct a series of ablation studies that progressively construct the final retrieval system. Specifically, we first evaluate diferent retrieval backbones to determine the strongest baseline. We then investigate the efectiveness of heterogeneous vision-language embedding ensembles and finally analyze the contribution of the proposed disagreement-aware VLM reranking strategy.

Baseline Framework Selection To establish a strong foundation for our solution, we first review the performance of representative text-based person retrieval and person anomaly retrieval methods on the Pedestrian Anomaly Behavior (PAB) benchmark. Table 1 summarizes the reported results from the literature together with the oficial AI City Challenge evaluation setting.

Among existing methods, the Hybrid, Unified and Iterative framework [11] achieves the best retrieval performance under the standard PAB evaluation, reaching 89.23% Recall@1 while maintaining nearly perfect Recall@5 and Recall@10. Compared with earlier approaches such as CMP [18], AnomalyLMM [6], and SSDC [17], it demonstrates superior retrieval robustness through the combination of Local-Global Hybrid Perspective (LHP), Unified Image-Text (UIT) learning, and iterative ensemble fusion. Motivated by its strong retrieval capability and modular design, we adopt this framework as the retrieval backbone and further extend it with heterogeneous vision-language embedding models and selective VLM reranking.

Table 1: Comparison with state-of-the-art methods on the Pedestrian Anomaly Behavior (PAB) dataset. Results are categorized by evaluation settings: Standard PAB benchmark vs. Full Challenge Test Set (which includes 34,795 distractor images, totaling 36,773 gallery images). † denotes results reported from corresponding publications. Our proposed method establishes a new state-of-the-art under the standard setting and provides a strong baseline under the highly challenging 36.7k gallery setting.
<table><tr><td>Method</td><td>Venue</td><td colspan="5">Gallery Size mAP (%) R@1 (%) R@5 (%) R@10 (%)</td></tr><tr><td colspan="7">Setting A: Standard PAB Benchmark (without large-scale distractors):</td></tr><tr><td>CLIP [13]†</td><td>ICML&#x27;21</td><td>Standard</td><td></td><td>47.57</td><td></td><td></td></tr><tr><td>IRRA [5]†</td><td>CVPR&#x27;23</td><td>Standard</td><td></td><td>78.67</td><td></td><td></td></tr><tr><td>X-VLM [20]†</td><td>ICML&#x27;22</td><td>Standard</td><td></td><td>81.95</td><td></td><td></td></tr><tr><td>RaSa [2]†</td><td>IJCAI&#x27;23</td><td>Standard</td><td></td><td>80.79</td><td></td><td></td></tr><tr><td>CMP [18]†</td><td>ICCV&#x27;25</td><td>Standard</td><td>90.41</td><td>84.93</td><td></td><td></td></tr><tr><td>AnomalyLMM [6]†</td><td>arXiv&#x27;25</td><td>Standard</td><td>90.89</td><td>84.73</td><td></td><td></td></tr><tr><td>SSDC [17]†</td><td>ACL&#x27;26</td><td>Standard</td><td>92.74</td><td>87.01</td><td></td><td></td></tr><tr><td>HUI [1i]†</td><td>WWW&#x27;25</td><td>Standard</td><td></td><td>89.23</td><td>99.70</td><td>99.90</td></tr><tr><td>Ours: Iterative Embedding Ensemble</td><td></td><td>Standard</td><td>94.87</td><td>90.09</td><td>99.75</td><td>100.0</td></tr><tr><td>Ours: + VLM Rerank &amp; RRF</td><td></td><td>Standard</td><td>94.99</td><td>90.34</td><td>99.75</td><td>100.0</td></tr><tr><td colspan="7">Setting B: Full Challenge Test Set (with 34,795 distractor images, 36,773 total):</td></tr><tr><td>Ours: Iterative Embedding Ensemble –</td><td></td><td>36,773</td><td>90.90</td><td>85.08</td><td>97.72</td><td>98.68</td></tr><tr><td>Ours: + VLM Rerank &amp; RRF</td><td></td><td>36,773</td><td>90.92</td><td>85.13</td><td>97.72</td><td>98.68</td></tr></table>

Progressive Heterogeneous Ensemble Starting from the selected retrieval backbone, we progressively incorporate heterogeneous vision-language embedding models through the iterative ensemble strategy described in Section 3.3. Table 2 presents the performance of diferent ensemble configurations.

Introducing Voyage Multimodal as the first refinement model already produces a substantial improvement over the baseline, indicating that its embedding space provides complementary semantic information beyond the original retrieval model. Furthermore, placing Qwen3-VL-Embedding as the final refinement stage yields another crucial practical insight: the ensemble is highly sensitive to the fusion weights of this large vision-language model. As shown in the 2-step refinement block of Table 2, a minor adjustment to the weight schedule (from 0.9:0.92 to 0.88:0.9) results in a significant +1.29% mAP boost (from 88.96% to 90.25%). This demonstrates that while Qwen provides powerful cross-modal reasoning, positioning it at the very end of the pipeline with carefully tuned weights is essential to correct residual hard negatives without overriding the geometric alignment established by the preceding dual-encoders.

To maximize ensemble diversity, we evaluate several candidate embedding models as an intermediate refinement stage (M<sub>2</sub>) inserted between Voyage and Qwen, including E5-Omni, Rzen-VL, BGE-VL-v1.5-mmeb, and DFN5B. As shown in Table 2, incorporating BGE-VL-v1.5-mmeb yields the highest performance on the challenge setting (Setting B), demonstrating that embedding models trained with diferent objectives capture additional cross-modal correspondences. Interestingly, we observe that while substituting it with Rzen-VL yields even higher accuracy on the standard benchmark (Setting A, reaching 95.12% mAP), BGE-VL-v1.5-mmeb proves more robust against the massive distractors in Setting B.

Table 2: Ablation study on candidate models for the second refinement step, with Voyage fixed as the first refinement step and Qwen fixed as the final step, evaluated on the Full Challenge Test Set (Setting B). All metrics are reported in ${ \% } .$ The best performance is achieved with BGE-VL-v1.5-mmeb as the intermediate refinement model.
<table><tr><td>Ensemble Configuration Weight Ratio mAP R@1 R@5 R@10</td></tr><tr><td>2-step refinement  $( S _ { i n i t }  \mathrm { V o y a g e  Q w e n } ) \colon$ </td></tr><tr><td>Voyage + Qwen 0.9:0.92 88.96 82.15 96.41 97.97</td></tr><tr><td>Voyage + Qwen (refined schedule) 0.88:0.9 90.25 84.07 97.47 98.58</td></tr><tr><td>3-step refinement  $( S _ { i n i t }  \mathrm { V o y a g e }  M _ { 2 }  \mathrm { Q w e n } ) \colon$ </td></tr><tr><td>Voyage + E5-Omni + Qwen 0.9:0.92:0.88 89.80 83.31 97.32 98.53</td></tr><tr><td>Voyage + Rzen-VL + Qwen 0.9:0.92:0.88 90.55 84.5897.47 98.53</td></tr><tr><td> $\mathbf { V o y a g e \Phi + B G E \mathrm { - } V L \mathrm { - } v 1 . 5 \mathrm { - } m m e b \Phi + Q w e n }$  0.9:0.92:0.88 90.90 85.08 97.7298.68</td></tr><tr><td> $\mathrm { V o y a g e + D F N 5 B + Q w e n }$  0.9:0.92:0.88 90.3884.2297.6298.73</td></tr></table>

This reveals a crucial practical insight: stronger standalone embedding models do not necessarily yield the best ensemble performance. Ultimately, the overall improvement largely depends on the complementarity between embedding spaces the degree to which they correct each other’s blind spots rather than just individual standalone performance, motivating our final ensemble architecture for the challenge.

Disagreement-aware VLM Reranking Analysis To validate the eficacy of our disagreement-aware routing mechanism, we investigate the impact of selective VLM reranking. We adopt a zero-tolerance disagreement threshold: a query is routed to the VLM if there is any inconsistency among the top-1 predictions of the ensemble configurations. This conservative threshold is deliberately chosen to maximize recall on subtle behavioral anomalies; because dual-encoder architectures are highly susceptible to visual distractors, any prediction divergence serves as a strong indicator of an edge case requiring fine-grained cross-modal reasoning. Consequently, this strict criterion isolates 299 ambiguous queries (15.1% of the test set). By restricting the computationally intensive gemini-3.1-flash-lite cross-encoder to rerank only the top-10 candidates of this challenging subset (using a prompt that instructs the model to act as an expert surveillance analyst to identify the precise anomalous behavior), our approach yields an 84.9% reduction in inference latency compared to exhaustive reranking.

To resolve these conflicting predictions, the VLM evaluates the top-10 candidates for each routed query, rectifying prediction inconsistencies and validating ensemble consensus. The final ranking is aggregated using Reciprocal Rank Fusion (RRF) with k=60. While the absolute mAP gain is modest, it is crucial to recognize that the routed queries encapsulate the most pathological edge cases instances where multiple robust feature representations fundamentally disagree. The capacity of the VLM to disentangle such ambiguity, while preserving the high-confidence dual-encoder predictions for the remaining 84.9% of the dataset (1,679 out of 1,978 queries), underscores the representational synergy and computational eficiency of our adaptive routing strategy.

Table 3: Performance of our final submission on the oficial AI City Challenge 2026 Track 4 benchmark. (<sup>∗</sup>) denotes the model is used without fine-tuning (zero-shot). Best results are in bold.
<table><tr><td>Method</td><td>mAP</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td> $\mathrm { U I T ^ { * } + L H P ^ { * } }$ </td><td>72.75</td><td>61.22</td><td>87.76</td><td>92.61</td></tr><tr><td> $\mathrm { U I T ^ { * } + L H P }$ </td><td>85.17</td><td>76.89</td><td>95.39</td><td>97.11</td></tr><tr><td> $\mathrm { U I T } + \mathrm { L H P }$ </td><td>87.24</td><td>80.03</td><td>96.00</td><td>97.11</td></tr><tr><td> $\mathrm { U I T } + \mathrm { L H P } + \mathrm { B L I P - 2 } + \mathrm { C L I P }$ </td><td>83.02</td><td>73.60</td><td>94.69</td><td>97.01</td></tr><tr><td>Ours: Iterative Ensemble</td><td>90.90</td><td>85.08</td><td>97.72</td><td>98.68</td></tr><tr><td>Ours: Iterative Ensemble + Rerank Fusion</td><td>90.92</td><td>85.13</td><td>97.72</td><td>98.68</td></tr></table>

## 4.4 Final Challenge Results

Table 3 reports the performance of our final submission on the oficial AI City Challenge 2026 Track 4 benchmark. Starting from the selected Hybrid retrieval backbone, we progressively improve retrieval performance through heterogeneous vision-language embedding ensemble learning. The proposed disagreement-aware VLM reranking and Reciprocal Rank Fusion further refine the predictions for ambiguous queries, leading to additional performance gains.

Our final system achieves 90.92% mAP, 85.13% Recall@1, 97.72% Recall@5, and 98.68% Recall@10. These results demonstrate that combining complementary embedding spaces with selective vision-language reasoning provides an efective solution for large-scale text-based person anomaly retrieval under challenging surveillance scenarios.

## 5 Discussion

Our results demonstrate that the efectiveness of the proposed framework stems from the complementary nature of heterogeneous vision-language representations. Rather than relying on a single embedding model, progressively integrating models trained with diferent objectives and data distributions consistently improves retrieval performance, suggesting that representational diversity is more valuable than simply selecting the strongest individual model. This observation is further supported by our ablation study, where the best-performing ensemble is not necessarily composed of the highest-performing standalone models.

Another practical insight is the importance of score alignment for heterogeneous ensemble learning. Since independently developed embedding models may adopt diferent gallery orderings, direct score-level fusion can produce incorrect correspondences. Aligning all similarity matrices to a unified gallery indexing enables reliable integration of heterogeneous embedding models while remaining agnostic to the underlying architectures.

The PAB benchmark also highlights the intrinsic dificulty of text-based person anomaly retrieval. Compared with conventional text-based person retrieval, successful retrieval requires jointly reasoning about pedestrian appearance, finegrained behaviors, object interactions, and scene context under substantial domain shifts between synthetic training data and real-world surveillance images.

Limitations. Our framework increases inference cost by combining multiple embedding models and selective VLM reranking. Moreover, the ensemble weights and routing strategy are empirically determined. Future work may explore adaptive ensemble weighting and learned query routing to further improve both retrieval accuracy and computational eficiency.

## 6 Conclusion

In this paper, we presented our solution for AI City Challenge 2026 Track 4 on text-based person anomaly retrieval. Building upon a strong retrieval backbone, we proposed a heterogeneous ensemble framework that integrates complementary vision-language embedding models through score alignment and iterative fusion, followed by selective VLM-based reranking for ambiguous queries.

Our final system achieved 90.92% mAP, 85.13% Recall@1, 97.72% Recall@5, and 98.68% Recall@10 on the oficial PAB benchmark. These results demonstrate the efectiveness of combining diverse vision-language representations with adaptive multimodal reasoning for fine-grained anomaly retrieval. We believe the proposed framework provides a practical and extensible solution for large-scale text-based person retrieval and ofers useful insights for future multimodal retrieval systems in real-world surveillance scenarios.

## Acknowledgements

We thank the organizers of the AI City Challenge 2026 for providing the PAB dataset and evaluation platform. We acknowledge the authors of the baseline framework [11] for making their code publicly available. We are also grateful to AIVN and GenAI4E for their valuable support, guidance, and resources throughout this work.

## References

1. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-VL technical report. arXiv preprint arXiv:2511.21631 (2025). https://doi.org/10.48550/arXiv.2511.21631, https://arxiv.org/ abs/2511.21631

2. Bai, Y., Cao, M., Gao, D., Cao, Z., Chen, C., Fan, Z., Nie, L., Zhang, M.: Rasa: Relation and sensitivity aware representation learning for text-based person search. arXiv preprint arXiv:2305.13653 (2023)

3. Cormack, G.V., Clarke, C.L.A., Buettcher, S.: Reciprocal rank fusion outperforms condorcet and individual rank learning methods. In: Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval. pp. 758–759. Association for Computing Machinery (2009). https://doi. org/10.1145/1571941.1572114, https://doi.org/10.1145/1571941.1572114

4. Hu, X., Zheng, T., Yang, K., Feng, Z.: The solution to the WWW25 textbased person anomaly search challenge. In: Companion Proceedings of the ACM on Web Conference 2025. pp. 1573–1575. Association for Computing Machinery (2025). https://doi.org/10.1145/3701716.3717651, https://doi.org/10. 1145/3701716.3717651

5. Jiang, D., Ye, M.: Cross-modal implicit relation reasoning and aligning for text-toimage person retrieval. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2787–2797 (2023)

6. Ju, H., Zhang, H., Zheng, Z.: AnomalyLMM: Bridging generative knowledge and discriminative retrieval for text-based person anomaly search. arXiv preprint arXiv:2509.04376 (2025). https://doi.org/10.48550/arXiv.2509.04376, https: //arxiv.org/abs/2509.04376

7. Li, J., Li, D., Savarese, S., Hoi, S.: BLIP-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: Proceedings of the 40th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 202, pp. 19730–19742. PMLR (2023), https: //proceedings.mlr.press/v202/li23q.html

8. Li, Z., Si, L., Guo, C., Yang, Y., Cao, Q.: Data augmentation for text-based person retrieval using large language models. arXiv preprint arXiv:2405.11971 (2024). https://doi.org/10.48550/arXiv.2405.11971, https://arxiv.org/abs/2405. 11971

9. Nguyen, T.H., Nguyen, M.N., Huy, N.N., Nguyen, H.V., Nhat, H.N.M., Nguyen, T.H., Nguyen, C.T., Le, H.M., Nguyen, D., Huynh, P.K., Xu, M., Bagci, U.: Esc: Emotional self-correction for reliable vision-language models (2026), https: //arxiv.org/abs/2607.02089

10. Nguyen, T.H., Tran, H.L., Ngo, T.D.: ITSELF: Attention guided fine-grained alignment for vision-language retrieval. arXiv preprint arXiv:2601.01024 (2026). https: //doi.org/10.48550/arXiv.2601.01024, https://arxiv.org/abs/2601.01024

11. Nguyen, T.H., Tran, H.L., Phan-Nguyen, H.P., Dinh, Q.V.: Hybrid, unified and iterative: A novel framework for text-based person anomaly retrieval. In: Companion Proceedings of the ACM on Web Conference 2025. pp. 1576–1580. Association for Computing Machinery (2025). https://doi.org/10.1145/3701716.3717653, https://doi.org/10.1145/3701716.3717653

12. Nogueira, R., Cho, K.: Passage re-ranking with BERT. arXiv preprint arXiv:1901.04085 (2019). https://doi.org/10.48550/arXiv.1901.04085, https: //arxiv.org/abs/1901.04085

13. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: Proceedings of the 38th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 139, pp. 8748–8763. PMLR (2021), https://proceedings.mlr. press/v139/radford21a.html

14. Sun, W., Yan, L., Ma, X., Wang, S., Ren, P., Chen, Z., Yin, D., Ren, Z.: Is Chat-GPT good at search? investigating large language models as re-ranking agents. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. pp. 14918–14937. Association for Computational Linguistics, Singapore (Dec 2023). https://doi.org/10.18653/v1/2023.emnlp-main.923, https://aclanthology.org/2023.emnlp-main.923/

15. Voyage AI: Multimodal embeddings. Voyage AI documentation (2025), https: //docs.voyageai.com/docs/multimodal-embeddings, documentation; accessed 2026-07-25. Publication year retained from the supplied entry.

16. Wang, W., Bao, H., Dong, L., Bjorck, J., Peng, Z., Liu, Q., Aggarwal, K., Mohammed, O.K., Singhal, S., Som, S., Wei, F.: Image as a foreign language: BEiT pretraining for all vision and vision-language tasks. arXiv preprint arXiv:2208.10442 (2022). https://doi.org/10.48550/arXiv.2208.10442, https: //arxiv.org/abs/2208.10442

17. Xie, Z., Luo, G., Wang, C., Cai, S., Jin, T., Zhao, Z., Tang, Y.: Bridging the pose-semantic gap: A cascade framework for text-based person anomaly search. In: Findings of the Association for Computational Linguistics: ACL 2026. pp. 4040–4049. Association for Computational Linguistics, San Diego, California (Jul 2026). https://doi.org/10.18653/v1/2026.findings-acl.197, https: //aclanthology.org/2026.findings-acl.197/

18. Yang, S., Wang, Y., Zhu, L., Zheng, Z.: Beyond walking: A large-scale imagetext benchmark for text-based person anomaly search. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 11720–11730 (Oct 2025), https://openaccess.thecvf.com/content/ICCV2025/html/Yang\_Beyond\_ Walking \_ A \_ Large - Scale \_ Image - Text \_ Benchmark \_ for \_ Text - based \_ Person \_ Anomaly\_ICCV\_2025\_paper.html

19. Yang, S., Zhou, Y., Wang, Y., Wu, Y., Zhu, L., Zheng, Z.: Towards unified textbased person retrieval: A large-scale multi-attribute and language search benchmark. In: Proceedings of the 31st ACM International Conference on Multimedia. pp. 4492–4501. Association for Computing Machinery (2023). https://doi.org/ 10.1145/3581783.3611709, https://doi.org/10.1145/3581783.3611709

20. Zeng, Y., Zhang, X., Li, H.: Multi-grained vision language pre-training: Aligning texts with visual concepts. arXiv preprint arXiv:2111.08276 (2021)

21. Zhang, J.: Enhancing text-based person retrieval via loss balancing and visionlanguage models. In: Companion Proceedings of the ACM on Web Conference 2025. pp. 1586–1588. Association for Computing Machinery (2025). https://doi. org/10.1145/3701716.3717655, https://doi.org/10.1145/3701716.3717655

22. Zheng, Z., Zheng, L., Garrett, M., Yang, Y., Xu, M., Shen, Y.D.: Dual-path convolutional image-text embeddings with instance loss. ACM Transactions on Multimedia Computing, Communications, and Applications 16(2), 1–23 (2020). https://doi.org/10.1145/3383184, https://doi.org/10.1145/3383184

23. Zhou, J., Xiong, Y., Liu, Z., Liu, Z., Xiao, S., Wang, Y., Zhao, B., Zhang, C.J., Lian, D.: MegaPairs: Massive data synthesis for universal multimodal retrieval. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 19076–19095. Association for Computational Linguistics, Vienna, Austria (Jul 2025). https://doi.org/10.18653/v1/2025.acllong.935, https://aclanthology.org/2025.acl-long.935/