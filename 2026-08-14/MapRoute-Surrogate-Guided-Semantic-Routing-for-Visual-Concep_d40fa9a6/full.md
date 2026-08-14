# MapRoute++: Surrogate-Guided Semantic Routing for Visual Concept Unlearning

Ashok Urlana<sup>1,2</sup>, L. D. M. S. Sai Teja<sup>3</sup>, Vivek Hruday Kavuri<sup>1</sup>, and Ponnurangam Kumaraguru<sup>1</sup>

<sup>1</sup>IIIT Hyderabad, <sup>2</sup>TCS Research, India, <sup>3</sup>NIT Silchar

Abstract. We present our submission to Task 3 of the Genµ 2.0 Challenge on visual concept unlearning. Building on MapRoute, we introduce task-specific training objectives, richer concept representations, and semantic routing for concept-specific mapper selection. Our approach improves robust concept removal while preserving unrelated and semantically adjacent concepts. On the oficial benchmark, evaluated using the Erasing-Retention-Robustness (ERR) metric on Stable Difusion v1.4, our method outperforms the state-of-the-art baseline by 12.1% on average across the five concept categories, achieving substantial gains. The code and dataset are available on GitHub.

## 1 Introduction

Recent text-to-image difusion models [3, 4, 8, 16, 18, 19] have achieved remarkable performance in image generation for diverse text prompts, enabling applications such as art generation, content creation, and data augmentation. However, these models also learn undesirable concepts, including copyrighted content and societal biases, raising concerns about fairness and responsible deployment [13, 17, 20, 21]. Visual concept unlearning, or concept erasure, seeks to selectively remove such concepts while preserving the model’s ability to generate unrelated and semantically adjacent concepts [1, 14, 15].

In this work, we address the visual concept unlearning task of the Genµ 2.0 Challenge<sup>1</sup>, which evaluates concept erasure on Stable Difusion v1.4 [18] using the ERR (Erasing–Retention–Robustness) score under direct, indirect, and adversarial prompts. Existing concept erasure approaches can be broadly categorized into optimization-based methods, such as ESD [5], CRCE [23], CORE [25], AGE [2], and FADE [22], and modular approaches, including SPM [12], Receler [7], and MapRoute [10]. Building on MapRoute, we enhance concept-specific mapper training with task-specific objectives, diverse textual concept representations, and semantic routing to achieve robust concept removal while preserving generation quality.

![](images/9953a04de52dc76a1eccbbc06a8ab7db5163dea4481a0c027b5676d2eb2282ed.jpg)  
Fig. 1: MapRoute++ uses a two-stage residual MLP to redirect concepts to surrogates in frozen difusion models, modifying the sequence-level EOT token embedding for targeted suppression while preserving semantic fidelity.

## 2 Challenge Dataset

As part of the Genµ 2.0 Challenge on visual concept unlearning, the organizers released a challenge dataset<sup>2</sup>. The dataset comprises 20 concepts spanning five categories: three Object concepts (Barberton Daisy, Golf Ball, and Apple Fruit), two Animal concepts (Blue Jay and Labrador Retriever), five Style concepts (Van Gogh, Doodle, Neon, Monet, and Sketch), five Scene concepts (Wedding, Sunset, Rainfall, Aurora Borialis, and Scenerie), and five Action concepts (Sleeping, Walking, Eating, Dancing, and Jumping). Moreover, the dataset includes direct, indirect, and adversarial prompts, plus adjacent concepts (semantically similar to the target) and retained concepts, to assess the unlearned model’s generalization.

## 3 Methodology

The MapRoute++ framework aims to erase the target concepts without compromising the model’s utility. The framework consists of three components: 1) a lightweight per-concept mapper module inserted between the frozen text encoder and the U-Net denoiser in the difusion model, 2) a two-stage training procedure to suppress a target concept while preserving the semantic representation of unrelated concepts, and 3) an input-conditioned routing mechanism that dynamically selects the appropriate trained module at inference time.

## 3.1 Design of Concept-Specific Mapper Module

Existing concept erasure techniques rely on concept-specific training or finetuning of difusion models, both requiring high-quality paired data (e.g., “target” vs. “surrogate” concepts) to precisely suppress concepts, often degrading generation quality. To overcome these problems, as shown in Figure 1, our approach, for each target concept, trains a lightweight residual multilayer perceptron (MLP)

to transform the corresponding text embedding, while keeping embeddings of unrelated concepts largely unchanged. Let $E ( \cdot )$ denote the frozen text encoder and c a text prompt. For each target concept $c _ { t a r }$ , a dedicated module $M _ { c _ { t a r } }$ is trained as a conditional identity mapping:

$$
M _ { c _ { t a r } } ( E ( c ) ) = \left\{ \begin{array} { l l } { E ( c _ { s u r } ) , } & { c = c _ { t a r } } \\ { E ( c ) , } & { c \neq c _ { t a r } } \end{array} \right.\tag{1}
$$

where $c _ { s u r }$ is the chosen substitute concept. The module leaves all embeddings unchanged except its assigned target, which it redirects toward the substitute.

## 3.2 Two-Stage Training Strategy

The Mapper module is optimized in two stages, ensuring it suppresses only the intended concept while leaving all other semantics intact.

Stage 1: Identity Pretraining. $M _ { c _ { t a r } }$ is first trained to reconstruct embeddings across a broad concept vocabulary C, establishing no changes in the embeddings as baseline before introducing any suppression behavior.

$$
\mathcal { L } _ { s t a g e 1 } = \mathbb { E } _ { c \in C } \Big [ \| M _ { c _ { t a r } } ( E ( c ) ) - E ( c ) \| _ { 2 } ^ { 2 } \Big ]\tag{2}
$$

In this stage, the mapper module behaves as an identity function for any input it has not been explicitly trained to alter.

Stage 2: Target-Surrogate Mapping. Building on this baseline, the module is trained to redirect the target concept’s embedding toward that of a chosen surrogate concept:

$$
\mathcal { L } _ { l e a r n } = \mathbb { E } _ { c _ { t a r } } \Big [ | | M _ { c _ { t a r } } ( E ( c _ { t a r } ) ) - E ( c _ { s u r } ) | | _ { 2 } ^ { 2 } \Big ]\tag{3}
$$

To prevent this targeted update from disturbing unrelated concepts, two regularization terms are added. The $\mathcal { L } _ { k e e p 1 }$ reinforces the identity mapping and $\mathcal { L } _ { k e e p 2 }$ does the same over a curated set of proper names W, protecting celebrity and identity-related concepts from unintended erasure. Overall, Stage 2 objective is:

$$
\mathcal { L } _ { s t a g e 2 } = \mathcal { L } _ { l e a r n } + \alpha \cdot \mathcal { L } _ { k e e p 1 } + \beta \cdot \mathcal { L } _ { k e e p 2 }\tag{4}
$$

where $\alpha , \beta$ (default = 1) balance erasure strength against preservation of unrelated concepts. Together, the two stages allow the Mapper to achieve near-zero distortion on non-target concepts while reliably suppressing the designated one. Following MapRoute [10], the surrogate concept $c _ { s u r }$ is not critical for erasure quality, since the mapper mainly learns to move the target embedding away from its original region rather than toward a specific destination. Unlike the original MapRoute, we adopt a task-specific objective for artistic styles, the full Stage 2 loss (Eq. 4) including $\mathcal { L } _ { k e e p 2 }$ is retained to protect artist identities, for objects, $\mathcal { L } _ { k e e p 2 }$ is omitted. Additionally, each concept is represented via multiple synonyms and paraphrases rather than a single keyword, enriching $\mathcal { L } _ { l e a r n }$ to encourage genuine semantic removal and better generalization to unseen or adversarial prompts.

Table 1: Comparison of Average ERR Score by Various Approaches. Best performing score per type is bolded and next best performing score is underlined.
<table><tr><td>Type</td><td>ESD</td><td>CA</td><td>FMN</td><td>FADE</td><td>MapRoute</td><td>MapRoute++</td></tr><tr><td>Object</td><td>0.470</td><td>0.471</td><td>0.487</td><td>0.598</td><td>0.919</td><td>0.863</td></tr><tr><td>Animal</td><td>0.499</td><td>0.497</td><td>0.467</td><td>0.599</td><td>0.796</td><td>0.604</td></tr><tr><td>Style</td><td>0.478</td><td>0.509</td><td>0.482</td><td>0.609</td><td>0.494</td><td>0.574</td></tr><tr><td>Scene</td><td>0.461</td><td>0.422</td><td>0.443</td><td>0.589</td><td>0.190</td><td>0.707</td></tr><tr><td>Action</td><td>0.477</td><td>0.460</td><td>0.403</td><td>0.607</td><td>0.871</td><td>0.857</td></tr><tr><td>Avg</td><td>0.477</td><td>0.472</td><td>0.456</td><td>0.600</td><td>0.591</td><td>0.721</td></tr></table>

## 3.3 Dynamic Semantic Routing

At inference, the framework computes the semantic similarity between the input prompt and all target concepts using the mean of the prompt’s token embeddings, and sequentially applies the top-k most relevant mapper modules. Although each mapper is trained on sentence-level text embeddings, it is applied token-wise to the full embedding sequence, enabling the learned semantic transformation to modify the cross-attention inputs while remaining fully compatible with the pretrained difusion model.

## 4 Experiments and Analysis

## 4.1 Experimental setup

We conduct all our experiments on Stable Difusion-v1.4 [18] model. We generate 20 images for each concept and obtain the ERR (Erasing-Retention-Robustness)<sup>3</sup> score, an oficial metric taken from the Genµ Challenge. For the LLaVA-based [11] ERR evaluation, every image is scored using the LLaVA<sup>4</sup> model with a constrained yes/no question. We use the same question form for all axes:

Does this image show <concept description>? Answer with ‘yes’ or ‘no’.

For direct, indirect, and adversarial target prompts, success means that the target is no longer present, so the expected answer is no. For adjacent and retained prompts, success means that the non-target concept is preserved, so the expected answer is yes, as given in Table 3. For example, the Blue Jay target is queried as “a Blue Jay bird”, and Van Gogh is queried as “Van Gogh painting style”.

Image-level binary decisions are aggregated across all concepts to compute five evaluation axes: target forgetting $\left( A _ { \mathrm { f g t } } \right)$ , retention of unrelated concepts $\left( A _ { \mathrm { r e t } } \right)$ ， retention of semantically adjacent concepts $( A _ { \mathrm { a d j } } )$ , robustness to indirect prompts $( A _ { \mathrm { i n d } } )$ , and robustness to adversarial prompts $\left( { \cal A } _ { \mathrm { a d v } } \right)$ . The final ERR score is computed as the harmonic mean of these five quantities, penalizing methods that achieve high performance on only a subset of the evaluation criteria. We conducted all experiments using two Nvidia GeForce RTX A6000 (48GB) GPUs and followed the same parameters as mentioned in the oficial MapRoute GitHub repo<sup>5</sup> for all the experiments.

Table 2: Per-Concept Breakdown of MapRoute++ Evaluation Results. Note: Animals are combined in the objects category.
<table><tr><td>#</td><td>Type</td><td>Concept</td><td> $A _ { f g t }$ </td><td> $A _ { r e t }$ </td><td> $A _ { a d j }$ </td><td> $A _ { i n d }$ </td><td> $A _ { a d v }$ </td><td>ERR</td></tr><tr><td>1</td><td></td><td>Labrador Retriever</td><td>1.000</td><td>0.9368</td><td>0.700</td><td>1.000</td><td>1.000</td><td>0.9098</td></tr><tr><td>2</td><td></td><td>Barberton Daisy</td><td>0.900</td><td>0.9368</td><td>0.350</td><td>1.000</td><td>1.000</td><td>0.7107</td></tr><tr><td>3</td><td></td><td>Blue Jay</td><td>1.000</td><td>0.9474</td><td>0.083</td><td>1.000</td><td>0.600</td><td>0.2990</td></tr><tr><td>4</td><td>Oobects</td><td>Golf Ball</td><td>0.950</td><td>0.9895</td><td>0.933</td><td>1.000</td><td>1.000</td><td>0.9738</td></tr><tr><td>5</td><td></td><td>Apple Fruit</td><td>1.000</td><td>0.9342</td><td>0.683</td><td>1.000</td><td>1.000</td><td>0.9035</td></tr><tr><td>6</td><td>Styles</td><td>Van Gogh</td><td>0.300</td><td>0.9368</td><td>1.000</td><td>0.250</td><td>0.300</td><td>0.3926</td></tr><tr><td>7</td><td></td><td>Doodle</td><td>0.100</td><td>0.9500</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.3558</td></tr><tr><td>8</td><td></td><td>Neon</td><td>0.850</td><td>0.9395</td><td>0.883</td><td>0.800</td><td>0.900</td><td>0.8720</td></tr><tr><td>9</td><td></td><td>Monet</td><td>0.250</td><td>0.9368</td><td>1.000</td><td>0.450</td><td>0.550</td><td>0.4947</td></tr><tr><td>10</td><td>AArtiitic</td><td>Sketch</td><td>0.700</td><td>0.9447</td><td>0.867</td><td>0.650</td><td>0.700</td><td>0.7567</td></tr><tr><td>11</td><td></td><td>Wedding</td><td>1.000</td><td>0.9474</td><td>0.667</td><td>1.000</td><td>1.000</td><td>0.9000</td></tr><tr><td>12</td><td>Set</td><td>Sunset</td><td>0.200</td><td>0.9474</td><td>0.950</td><td>0.400</td><td>1.000</td><td>0.4713</td></tr><tr><td>13</td><td></td><td>Rainfall</td><td>1.000</td><td>0.9421</td><td>0.333</td><td>0.950</td><td>1.000</td><td>0.7028</td></tr><tr><td>14</td><td>Open</td><td>Aurora Borealis</td><td>1.000</td><td>0.9447</td><td>0.667</td><td>1.000</td><td>1.000</td><td>0.8995</td></tr><tr><td>15</td><td></td><td>Scenerie</td><td>1.000</td><td>0.9395</td><td>0.983</td><td>0.300</td><td>0.400</td><td>0.5609</td></tr><tr><td>16</td><td></td><td>Sleeping</td><td>1.000</td><td>0.9500</td><td>0.667</td><td>1.000</td><td>1.000</td><td>0.9005</td></tr><tr><td>17</td><td>Actiities</td><td>Walking</td><td>0.500</td><td>0.9553</td><td>0.883</td><td>0.400</td><td>1.000</td><td>0.6511</td></tr><tr><td>18</td><td></td><td>Eating</td><td>0.950</td><td>0.9579</td><td>0.950</td><td>1.000</td><td>1.000</td><td>0.9710</td></tr><tr><td>19</td><td></td><td>Dancing</td><td>0.750</td><td>0.9395</td><td>0.983</td><td>1.000</td><td>1.000</td><td>0.9234</td></tr><tr><td>20</td><td></td><td>Jumping</td><td>0.700</td><td>0.9474</td><td>0.683</td><td>1.000</td><td>1.000</td><td>0.8407</td></tr><tr><td></td><td></td><td>Average</td><td>0.7575</td><td>0.9462</td><td>0.7633</td><td>0.8100</td><td>0.8725</td><td>0.7245</td></tr></table>

## 4.2 Baselines Performance Analysis

We compare MapRoute++ against five recent concept erasure baselines provided in the Genµ 2.0 Challenge: ESD [6], CA [9], FMN [24], FADE [22] and MapRoute [10]. Table 1 presents the average ERR scores across the five distinct conceptual categories (Object, Animal, Style, Scene, and Action) along with the overall average performance.

Overall Superiority. MapRoute++ achieves a new state-of-the-art overall average ERR score of 0.721, substantially outperforming the strongest baseline, FADE. This translates to a 12.1% absolute improvement on average across all categories. In contrast, earlier methods such as ESD, CA, FMN and MapRoute struggle to balance the competing objectives of target forgetting, semantic preservation, and prompt robustness, clustering below the 0.600 mark.

Table 3: LLaVA prompt targets and success criteria used for ERR.
<table><tr><td>Prompt Type</td><td>Queried concept</td><td>Success answer</td></tr><tr><td>Direct  $\left( A _ { \mathrm { f g t } } \right)$ </td><td>Target concept</td><td>no</td></tr><tr><td>Indirect  $\left( A _ { \mathrm { i n d } } \right)$ </td><td>Target concept</td><td>no</td></tr><tr><td>Adversarial  $\left( { \cal A } _ { \mathrm { a d v } } \right)$ </td><td>Target concept</td><td>no</td></tr><tr><td>Adjacent  $\left( A _ { \mathrm { a d j } } \right)$ </td><td>Adjacent concept yes</td><td></td></tr><tr><td>Retention  $\left( A _ { \mathrm { r e t } } \right)$ </td><td>Retained concept yes</td><td></td></tr></table>

Category-Specific Strengths. The advantages of our surrogate-guided semantic routing are most pronounced in the Object and Scene categories. For scenes, MapRoute++ achieves an ERR of 0.707, vastly surpassing MapRoute by a margin of 50%+ improvement. These results indicate that modifying the sequence-level EOT token embedding via our mapper is highly efective for localized, discrete concepts (e.g., Golf Ball, Eating) that do not dictate global image statistics.

Limitations in Style Erasure. The Style category remains the most challenging for our approach. As evidenced by our per-concept breakdown (Table 2) and qualitative failure cases (Section 5.1), artistic styles like Van Gogh and Doodle heavily rely on global texture, brushwork, and structural composition. Because MapRoute++ operates primarily on semantic token embeddings rather than fine-tuning the cross-attention weights of the U-Net denoiser, completely untangling pervasive stylistic features without aggressively degrading adjacent concepts proves dificult. Nonetheless, MapRoute++ still outperforms ESD, CA, FMN and MapRoute by comfortable margins even in this category.

## 5 Ablations and Discussion

## 5.1 Selection of Surrogate Concepts

The fundamental diference between MapRoute and MapRoute++ lies in the selection of surrogate concepts. In MapRoute, the surrogate concepts can be arbitrary, whereas in MapRoute++, we heuristically select the surrogate concepts without revealing the target concept. These concepts are selected to redirect the difusion model away from the target concept while preserving the model’s general image-generation ability. A useful surrogate should be visually plausible, should not explicitly or implicitly contain the target concept, and should not be so close to the target that it reintroduces target-specific visual cues. At the same time, it should not be too distant or destructive, since that can damage adjacent or retained concepts. We choose surrogates using only general semantic reasoning about the target concept, without using information from the challenge dataset’s indirect prompts, adversarial prompts, or adjacent-concept prompts. This avoids evaluation leakage and makes the setting conservative. This careful selection is important because poor surrogates can either fail to erase the target or cause over-erasure that harms nearby concepts, which is why we evaluate all five axes rather than direct forgetting alone.

![](images/cc693adba9ded2059126d9fac7c86dca069e3921b1e63b4ccc4d8d73336d4b36.jpg)  
Fig. 2: Representative successful unlearning cases of MapRoute++. In each row, the base model produces the target concept from the target prompt, while the unlearned model removes the target under direct and adversarial prompts. The final column shows that a related adjacent concept remains visually recognizable.

The qualitative examples in Figures 2 and 3 illustrate why this careful selection matters. Successful cases show that the mapper can suppress the target concept under direct and adversarial prompts while preserving adjacent concepts. Failure cases show the two main risks of surrogate selection: the surrogate may be too weak, allowing the target concept to remain visible, or too aggressive, causing adjacent-concept degradation. These outcomes motivate evaluating unlearning with all five axes rather than only measuring direct target forgetting.

## 5.2 Error Analysis

Successful cases. Figure 2 shows representative high-scoring concepts. These examples show that the mapper can remove the target concept while preserving nearby non-target concepts. Golf Ball and Eating have the two highest ERR scores in Table 2, and Labrador Retriever provides an object-category example with strong target forgetting and good adjacent-concept retention. The adversarial prompts are nonsensical prompt strings from the benchmark; success on this axis means that the target concept is not regenerated.

Failure cases. Figure 3 shows representative failure cases for the two most diagnostic concepts. For Van Gogh, the selected examples show that the generated images retain highly recognizable swirling brushwork and Starry Night-like structure under direct, indirect, and adversarial prompts, illustrating failures in both forgetting and robustness. For Blue Jay, the examples show that adjacent bird prompts collapse into texture-like images, while adversarial prompts can still produce bird-like outputs, revealing both adjacent-concept damage and adversarial leakage.

![](images/36c5bae5358ecf9d01fcd3ccc84afe75ee89501ed68bd6d4922260326df53c80.jpg)  
Fig. 3: Representative qualitative failures. Top row: Van Gogh remains visually present after unlearning, including for indirect and adversarial prompts. Bottom row: the base model produces a recognizable adjacent bird, but the unlearned Blue Jay mapper often converts adjacent birds into texture-like patterns; adversarial prompting can still produce bird-like imagery.

## 6 Conclusion

We presented MapRoute++, a compute-eficient framework for visual concept unlearning that combines lightweight concept-specific mapper networks with a two-stage training strategy and semantic routing. By selecting relevant surrogate concepts and applying only the corresponding mapper modules at inference, MapRoute++ achieves near-zero distortion on non-target concepts while reliably suppressing designated concepts without modifying the underlying difusion model. Experimental results on the Genµ 2.0 benchmark demonstrate that the proposed approach consistently outperforms existing baselines in concept unlearning while preserving model utility.

## References

1. Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raf, E., Biderman, S.: Leace: Perfect linear concept erasure in closed form. Advances in Neural Information Processing Systems 36, 66044–66063 (2023)

2. Bui, A., Vu, T., Vuong, L., Le, T., Montague, P., Abraham, T., Kim, J., Phung, D.: Fantastic targets for concept erasure in difusion models and where to find them (2025), https://arxiv.org/abs/2501.18950

3. Chang, H., Zhang, H., Barber, J., Maschinot, A., Lezama, J., Jiang, L., Yang, M.H., Murphy, K., Freeman, W.T., Rubinstein, M., et al.: Muse: Text-to-image generation via masked generative transformers. In: Proceedings of the 40th International Conference on Machine Learning. pp. 4055–4075 (2023)

4. Dhariwal, P., Nichol, A.: Difusion models beat gans on image synthesis. Advances in neural information processing systems 34, 8780–8794 (2021)

5. Gandikota, R., Materzynska, J., Fiotto-Kaufman, J., Bau, D.: Erasing concepts from difusion models (2023), https://arxiv.org/abs/2303.07345

6. Gandikota, R., Materzynska, J., Fiotto-Kaufman, J., Bau, D.: Erasing concepts from difusion models. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 2426–2436 (2023)

7. Huang, C.P., Chang, K.P., Tsai, C.T., Lai, Y.H., Yang, F.E., Wang, Y.C.F.: Receler: Reliable concept erasing of text-to-image difusion models via lightweight erasers. In: European Conference on Computer Vision. pp. 360–376. Springer (2024)

8. Kim, G., Kwon, T., Ye, J.C.: Difusionclip: Text-guided difusion models for robust image manipulation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2426–2435 (2022)

9. Kumari, N., Zhang, B., Wang, S.Y., Shechtman, E., Zhang, R., Zhu, J.Y.: Ablating concepts in text-to-image difusion models. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 22691–22702 (2023)

10. Li, S., Liang, B., Xia, S., Yang, Y.: Maproute:precise-concept erasing mappers via semantic routing. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10187–10196 (June 2026)

11. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 26296–26306 (2024)

12. Lyu, M., Yang, Y., Hong, H., Chen, H., Jin, X., He, Y., Xue, H., Han, J., Ding, G.: One-dimensional adapter to rule them all: Concepts, difusion models and erasing applications (2024), https://arxiv.org/abs/2312.16145

13. Mishkin, P., Ahmad, L., Brundage, M., Krueger, G., Sastry, G.: Dall·e 2 preview - risks and limitations (2022), http://github.com/openai/dalle-2-preview/ blob/main/system-card.md

14. Pham, M., Marshall, K.O., Hegde, C., Cohen, N.: Robust concept erasure using task vectors. arXiv preprint arXiv:2404.03631 (2024)

15. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)

16. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical textconditional image generation with clip latents

17. Rando, J., Paleka, D., Lindner, D., Heim, L., Tramèr, F.: Red-teaming the stable difusion safety filter. arXiv preprint arXiv:2210.04610 (2022)

18. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)

19. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic textto-image difusion models with deep language understanding. Advances in neural information processing systems 35, 36479–36494 (2022)

20. Schramowski, P., Brack, M., Deiseroth, B., Kersting, K.: Safe latent difusion: Mitigating inappropriate degeneration in difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 22522– 22531 (2023)

21. Somepalli, G., Singla, V., Goldblum, M., Geiping, J., Goldstein, T.: Difusion art or digital forgery? investigating data replication in difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6048–6058 (2023)

22. Thakral, K., Glaser, T., Hassner, T., Vatsa, M., Singh, R.: Fine-grained erasure in text-to-image difusion-based foundation models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9121– 9130 (2025)

23. Xue, Y., Moroshko, E., Chen, F., Sun, J., McDonagh, S., Tsaftaris, S.A.: Crce: Coreference-retention concept erasure in text-to-image difusion models (2025), https://arxiv.org/abs/2503.14232

24. Zhang, G., Wang, K., Xu, X., Wang, Z., Shi, H.: Forget-me-not: Learning to forget in text-to-image difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1755–1764 (2024)

25. Zhu, J., Zhang, R., Lin, L., Mei, S.: Choose your anchor wisely: Efective unlearning difusion models via concept reconditioning. In: Neurips Safe Generative AI Workshop 2024 (2024), https://openreview.net/forum?id=8naq3XyGQe