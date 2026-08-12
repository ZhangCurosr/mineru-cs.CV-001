# When Vision Becomes Text: Visual Token Pruning via Cross-Modal Residual Guidance in VLMs

Congyang Ou<sup>1,2</sup>, Ruike Song<sup>1</sup>, Yang Zhou<sup>1</sup>, Libo Sun<sup>3</sup>, Haokui Zhang<sup>1,†</sup>, Zhenbo Luo<sup>2</sup>

<sup>1</sup>Northwestern Polytechnical University

<sup>2</sup>MiLM Plus, Xiaomi Inc.

<sup>3</sup>Chongqing University

## Abstract

Abundant visual information strengthens vision-language model (VLM) perception, yet massive visual tokens raise inference costs. Existing visual token pruning methods rely on similarity-based guidance, which exploits pairwise text-vision and vision-vision token correlations for compression. However, such methods only capture local layer-level signals and overlook the whole inference process in VLM. In this paper, we revisit VLM inference and present a new eficient guidance scheme that complements similarity-based guidance. In particular, we identify a key observation: as LLM layers deepen, text tokens continuously aggregate visual information via selfattention and progressively absorb partial visual content into textual representations. To quantify this phenomenon, we propose Cross Modal Absorption (CMA) from a geometric representation perspective to measure how much visual information is absorbed by text, revealing that more visual tokens in deeper layers can be approximately explained by the text subspace. We accordingly propose Cross Modal Residual (CMR). It projects visual tokens onto the text subspace via Tikhonov regularized least squares and exploits reconstruction residuals to quantify visual information that cannot be explained by text. Finally, based on CMR, we present SIEVE, a trainingfree visual token compression method that combines CMR, text-attention relevance, and residual-space diversity to retain task-relevant and complementary tokens. Experiments on diverse VLM architectures verify the efectiveness of SIEVE. For instance, on LLaVA-NeXT-7B, SIEVE keeps only 11.1% of visual tokens while preserving 97.5% of the original average performance, achieving 3.62× prefill speedup, 2.49× end-to-end speedup, and a 6.02× KV-cache reduction.

## Introduction

Vision Language Models (VLMs) have achieved remarkable progress in multimodal tasks such as visual question answering, image captioning, and document understanding. Representative models, including LLaVA (Liu et al. 2023), InternVL (Chen et al. 2024b), and Qwen-VL (Bai et al. 2023), typically encode images into numerous visual tokens and feed them into large language models together with text tokens for joint reasoning. However, the increasing number of visual tokens substantially raises self-attention computation, leading to higher inference latency and computational cost. This overhead is further amplified in high-resolution images, multi-image inputs, and video understanding. Therefore, compressing visual tokens while preserving model capability is crucial for eficient VLM inference.

![](images/21e77a5cef708a8a424fcdd455b881a4d6f6c7fdb4cd257a2b3b0ec8270f033a.jpg)  
Figure 1: 3D visualization of visual tokens relative to the text subspace. Text tokens are centered and projected by PCA to obtain the first two principal directions, denoted as Text PC1 and Text PC2. Visual tokens are centered using the same text mean and projected onto these directions to obtain planar coordinates, while height and color indicate their CMR. More results are provided in Appendix A.

Existing visual token compression methods mainly select tokens based on current layer attention (Chen et al. 2024a; Zhang et al. 2025b) or feature similarity (Bolya et al. 2023; Wen et al. 2025): the former characterizes the relevance between visual tokens and the current textual context, whereas the latter measures the redundancy among visual tokens. Although these signals provide useful criteria, they only characterize the local state at a single layer and lack a holistic view of the entire multimodal reasoning process. To this end, we revisit the inference process of VLMs, rather than staying at the local state of a single layer, and observe that as LLM layers deepen, text tokens continuously aggregate visual information through self-attention, progressively integrating part of the visual content into the textual representations.

To quantify this phenomenon, we propose Cross Modal Absorption (CMA) from a geometric representation perspective, which measures at the layer level the overall extent to which visual token representations are absorbed by text, i.e., explainable by the text subspace. CMA characterizes the absorption trend at the layer level, whereas compression must operate on individual tokens. We therefore refine this perspective to the token level and propose Cross Modal Residual (CMR). Specifically, each visual token is projected onto the text subspace with Tikhonov regularization, and the reconstruction residual is used to quantify the visual information unexplained by text. A larger residual indicates that the token carries more unique visual information and thus has a higher retention necessity. This projection can be eficiently solved in closed form via regularized least squares, introducing only minor additional overhead. As shown in Figure 1, visual token representations gradually move closer to the text subspace as layer depth increases, indicating that visual information does not always remain independent, but is progressively integrated into the text representation space.

Building on the above analysis, we propose SIEVE, a training-free visual token compression method that selects tokens from three complementary perspectives: CMR for visual distinctiveness and text-to-visual attention relevance for task relevance, so that retained tokens both carry unique visual information and align with the current textual context, together with diversity selection in the residual space to remove redundancy. Unlike de-duplication in the original feature space, the residual space has already removed the components explainable by text, so its directions more directly reflect the unique visual information of each token, enabling more precise identification of genuinely complementary tokens. SIEVE is training-free and plug-and-play, making it directly applicable to existing VLMs.

In summary, our contributions are threefold:

• We propose a novel pruning guidance criterion named CMR. Unlike mainstream token pruning approaches guided by token similarity, we revisit the model inference process and identify the phenomenon where visual information is gradually assimilated by textual information. We then quantify this process to develop CMR.

• We propose SIEVE, a training-free visual token compression method that jointly selects tokens using CMR, attention relevance, and residual-space diversity selection, balancing visual distinctiveness, task relevance, and complementarity.

• Extensive experiments across three VLM architectures and eight benchmarks demonstrate that the proposed method significantly reduces computational overhead and memory, delivering practical inference acceleration.

## Related Work

## Multimodal Large Language Models

VLMs perform multimodal reasoning by feeding visual and text tokens into LLMs. Early systems use cross-attention or query-based designs, e.g., Flamingo (Alayrac et al. 2022), BLIP-2 (Li et al. 2023a), Kosmos-1 (Huang et al. 2023), and PaLM-E (Driess et al. 2023), followed by instructiontuned models such as MiniGPT-4 (Zhu et al. 2024), mPLUG-Owl (Ye et al. 2023), LLaVA (Liu et al. 2023), and Instruct-BLIP (Dai et al. 2023). Recent high-resolution, generalpurpose VLMs such as Qwen-VL (Bai et al. 2023), InternVL (Chen et al. 2024b), Gemini (Team et al. 2023), LLaVA-NeXT (Liu et al. 2024a), and Qwen2.5-VL (Bai et al. 2025) further improve OCR, grounding, and visual reasoning, and extend to video understanding (Maaz et al. 2024; Lin et al. 2024; Li et al. 2024, 2025a). However, high-resolution images, multi-image inputs, and long videos rapidly inflate visual-token counts, incurring heavy self-attention and KVcache overhead; low-loss visual token compression is thus crucial for their eficient inference.

![](images/9c7ba9636863c9f3d94e89d636edb97bee7feb5f389aedb1d48eaead7bf3f163.jpg)  
Figure 2: Layer-wise evolution of cross-modal absorption in LLaVA-1.5-7B. Cross-modal absorption (CMA) increases across Transformer layers.

## Visual Token Compression for VLMs

Existing methods mainly fall into two categories. Attentionscore pruning (Chen et al. 2024a; Zhang et al. 2025b; Xing et al. 2024; Huang et al. 2025) removes low-importance tokens without training, but attention scores are biased and unreliable indicators of true importance. Similarity-based reduction (Bolya et al. 2023; Wen et al. 2025; Yang et al. 2025b; Shang et al. 2025; Li et al. 2025b) merges similar tokens to preserve visual coverage, yet similarity need not correspond to text-query relevance. Other methods use diversity- and coverage-based selection (Alvar et al. 2025; Zhang et al. 2025a; Deng et al. 2025; Zou et al. 2025), adaptive token dropping (Chen et al. 2026), cross-modal information flow (Lin et al. 2025; Yang et al. 2025a) or objectivedriven (Zhang et al. 2026). These criteria, however, capture only a single-layer local state and lack a holistic view of the multimodal reasoning process. We therefore revisit the VLM inference process and find that, as layers deepen, text tokens progressively absorb visual information, reducing the independent contribution of some visual tokens. This suggests that token retention should focus on the residual information unexplained by text.

## Analysis

Difering from these works, we revisit the VLM inference process and investigate the evolution of visual tokens and text tokens throughout this procedure. Figure 2 presents our observation.

We analyze all 32 layers of LLaVA-1.5-7B on three benchmarks, TextVQA (Singh et al. 2019), POPE (Li et al. 2023b), and MME (Fu et al. 2025), covering 16,284 samples. The results show that an increasing portion of visual-token representations can be explained by the text subspace, indicating that visual information is progressively integrated into the text representation space during forward propagation.

![](images/48167864a7d4c11e9e65fd268aa4ff070a73953d9580fef8c828fd8040f85d53.jpg)  
Figure 3: Overview of SIEVE. SIEVE measures each visual token’s residual information with respect to the text subspace via CMR, combines it with attention relevance, and performs residual-space diversity selection to retain informative and nonredundant visual tokens for the compressed input sequence.

## Measuring Cross Modal Absorption

To quantify this phenomenon, we define Cross Modal Absorption (CMA) to measure how much of the visual token representation at a given layer can be explained by the text subspace. The CMA at layer l is defined as

$$
\mathrm { C M A } = 1 - \frac { \left\| V _ { c } - \hat { V } _ { c } \right\| _ { F } ^ { 2 } } { \left\| V _ { c } \right\| _ { F } ^ { 2 } } .\tag{1}
$$

Here, $V _ { c }$ denotes the matrix of all centered visual tokens at layer l (rather than an individual token); $\hat { V } _ { c }$ its projection onto the text subspace; and $| | \cdot | | _ { F }$ the Frobenius norm. CMA is a layer-level metric: a larger value means more visual information can be explained by the text subspace, i.e., stronger cross modal absorption.

## Visual Tokens Are Progressively Absorbed by the Text Subspace

As shown in Figure 2, CMA consistently increases from shallow to deep layers across all three benchmarks: from 0.008 to 0.209 on POPE, from 0.009 to 0.217 on MME, and from 0.017 to 0.396 on TextVQA. Its Spearman correlation with layer depth reaches $\rho = + 0 . 9 9$ on all three, indicating a highly consistent monotonic trend.

Thus visual tokens do not remain fully independent throughout the network, but become increasingly explainable by the text subspace as depth grows. This reflects cross modal information flow in autoregressive VLMs: under causal attention, text tokens continuously aggregate visual information, so the text subspace comes to contain more visualrelated directions. It also reveals cross modal redundancy in deep visual tokens: a token already well explained by the text subspace may not need to be fully preserved. We can therefore measure, at the token level, the relationship between each visual token and the text subspace, keeping information that is hard to explain by text while removing redundant tokens with little text-unexplained residual. Accordingly, the next section introduces the token-level Cross-Modal Residual (CMR) and the CMR-based visual token compression algorithm, SIEVE.

Note that the absolute value of CMA should not be read directly on the [0, 1] scale. Since the number of text tokens is usually around $\dot { N _ { t } } \approx 3 0$ , far smaller than the hidden dimension $D = 4 0 9 6 ,$ , a random subspace can theoretically explain only about $N _ { t } / D \approx 0 . 0 0 7$ of a high-dimensional representation. Thus the final-layer CMA of 0.209 on POPE is already about 29× the random baseline, indicating that the explanatory power of the text subspace over visual representations is not random but gradually emerges as a cross-modal representation pattern in deeper layers. Details are provided in Appendix B.

## Method

Since CMA operates at the layer level while compression acts on individual tokens, we refine it to the token level and propose SIEVE, a training-free visual token compression method that selects tokens from three complementary perspectives: CMR, text-attention relevance, and residual-space diversity, as shown in Figure 3

## Constructing the CMR Score

SIEVE first constructs a Cross Modal Residual (CMR) score to measure the residual information in each visual token that cannot be explained by the text subspace: a lower score means higher redundancy, a higher one more distinctive visual information. At the l-th pruning layer, let $T \in \mathbb { R } ^ { N _ { t } \times D }$ and

$V \in \mathbb R ^ { N _ { v } \times D }$ denote the text- and visual-token representation matrices, where $D$ is the hidden dimension. SIEVE first centers both modalities using the mean text representation:

$$
\bar { t } = \frac { 1 } { N _ { t } } \sum _ { j = 1 } ^ { N _ { t } } T _ { j } , \quad T _ { c } = T - \bar { t } , \quad V _ { c } = V - \bar { t } .\tag{2}
$$

Here t<sup>¯</sup> is the mean text representation, and $T _ { c } , V _ { c }$ are the centered text and visual matrices. Using the text center as reference reduces the background components shared between the two modalities and highlights each visual token’s unique deviation from the text subspace.

For the i-th centered visual token $v _ { c , i } ,$ SIEVE reconstructs it as a linear combination of text tokens, using a Tikhonovregularized least-squares formulation for stability:

$$
b _ { i } = \arg \operatorname* { m i n } _ { b _ { i } \in \mathbb { R } ^ { 1 \times N _ { t } } } \left\| v _ { c , i } - b _ { i } T _ { c } \right\| _ { 2 } ^ { 2 } + \lambda \left\| b _ { i } \right\| _ { 2 } ^ { 2 } .\tag{3}
$$

Here $b _ { i }$ are the combination coeficients of $v _ { c , i }$ over text tokens and λ is the regularization coeficient. The first term measures the reconstruction error from the text subspace, while the second constrains the coeficient magnitude to prevent unstable solutions caused by low-energy directions.

To adaptively determine λ, SIEVE constructs the text Gram matrix:

$$
G = T _ { c } T _ { c } ^ { \top } .\tag{4}
$$

Here $G \in \mathbb { R } ^ { N _ { t } \times N _ { t } }$ characterizes the energy distribution of the text subspace, providing the basis for setting λ. Sorting its eigenvalues in descending order as $\sigma _ { 1 } ^ { 2 } \ge \cdots \ge \sigma _ { N _ { \mathrm { } } } ^ { 2 }$ and given an energy ratio threshold $\eta ,$ SIEVE selects the smallest k satisfying

$$
\sum _ { j = 1 } ^ { k } \sigma _ { j } ^ { 2 } \geq \eta \sum _ { j = 1 } ^ { N _ { t } } \sigma _ { j } ^ { 2 } .\tag{5}
$$

Here η is the energy proportion to be covered and k is the number of dominant directions needed to reach it, delimiting the efective dimensionality of the text subspace. The regularization coeficient is then set as

$$
\lambda = \sigma _ { k } ^ { 2 } + \epsilon ,\tag{6}
$$

where ϵ is a small constant for numerical stability. Setting λ to the k-th eigenvalue lets the dominant directions contribute fully to reconstruction while suppressing instability from low-energy directions. This yields the closed-form solution and the reconstruction:

$$
b _ { i } = v _ { c , i } T _ { c } ^ { \top } \left( G + \lambda I \right) ^ { - 1 } , \qquad \hat { v } _ { c , i } = b _ { i } T _ { c } .\tag{7}
$$

Here I is the identity matrix and $\hat { v } _ { c , i }$ is the visual token reconstructed from the text subspace; the closed-form solution avoids iterative optimization and adds only negligible overhead. Finally, SIEVE defines the CMR score as the ratio between the reconstruction residual and the original centered representation:

$$
\mathrm { C M R } _ { i } = \frac { \| v _ { c , i } - \hat { v } _ { c , i } \| _ { 2 } } { \| v _ { c , i } \| _ { 2 } } .\tag{8}
$$

Normalizing the residual by the original norm removes the efect of difering token magnitudes. A lower CMR means the token is well reconstructed by the text subspace and can be preferentially pruned, while a higher CMR<sub>i</sub> means it preserves a larger text-unexplained residual and should be preferentially retained.

## Attention Signal Construction

CMR measures the unique information in a visual token that cannot be explained by the text subspace, but geometric uniqueness does not necessarily indicate relevance to the current text context. Therefore, SIEVE further introduces an attention signal to determine whether each visual token is actually attended to by the text tokens. For the i-th visual token, its text relevance score is defined as

$$
a _ { i } = \sum _ { h \in \mathcal { H } _ { \mathrm { t o p } } } \sum _ { t \in \mathcal { T } } \alpha _ { t , i } ^ { h } .\tag{9}
$$

Here, $\tau$ denotes the set of text tokens, and $\alpha _ { t , i } ^ { h }$ denotes the attention weight from text token t to visual token i in the h-th attention head. To reduce the influence of noisy heads, SIEVE keeps only the top $\rho _ { h }$ proportion of heads with the largest attention mass on the visual region, denoted as ${ \mathcal { H } } _ { \mathrm { t o p } } .$ A larger $a _ { i }$ indicates stronger relevance to the current textual context.

SIEVE does not need to explicitly store the full attention matrix. Following the dual-flash attention strategy in Sparse-VLM (Zhang et al. 2025b), SIEVE can compute the required attention signal while remaining compatible with FlashAttention, with details provided in Appendix C.

## Guided Scores

SIEVE initializes each token’s integrated score as

$$
\mathrm { S c o r e } _ { i } = a _ { i } ^ { 2 } \mathrm { C M R } _ { i } ^ { 2 } ,\tag{10}
$$

where $a _ { i }$ is the text-relevance (attention) score and $\mathrm { C M R } _ { i }$ is the residual ratio unexplained by the text subspace; the product keeps attention from overlooking residual information and CMR from ignoring textual relevance.

## Residual-Space Diversity Selection

Given the guided scores defined above, SIEVE aims to retain visual tokens that are both individually informative and mutually complementary. To this end, it performs residualspace diversity selection in two stages: it first measures the redundancy between visual tokens in the residual space, and then greedily selects tokens while progressively suppressing redundant candidates.

1. Residual Similarity Measurement. Selecting the top-K tokens by integrated score alone may retain tokens with similar residual directions, causing visual redundancy. To avoid this, SIEVE measures token redundancy in the residual space rather than the original space: the residual removes the large shared component that otherwise dominates the cosine similarity, so comparisons reflect each token’s genuinely complementary information. For the i-th centered visual token $v _ { c , i }$ with text-subspace reconstruction $\hat { v } _ { c , i }$ , the residual is

$$
r _ { i } = v _ { c , i } - \hat { v } _ { c , i } ,\tag{11}
$$

i.e., the part the text subspace cannot reconstruct. Redundancy between two tokens is measured by the squared cosine similarity of their residuals,

$$
g _ { i j } = \left( \frac { r _ { i } ^ { \top } r _ { j } } { \left\| r _ { i } \right\| _ { 2 } \left\| r _ { j } \right\| _ { 2 } } \right) ^ { 2 } \in [ 0 , 1 ] ,\tag{12}
$$

<table><tr><td>Methods</td><td>Venue</td><td>GQA</td><td>MMB</td><td>MMB-cn</td><td>MME</td><td>POPE</td><td>SQA</td><td>VQAv2</td><td>TextVQA</td><td>Average</td></tr><tr><td>LLaVA-1.5-7B</td><td colspan="8">Upper Bound, 576 Tokens</td><td></td><td></td></tr><tr><td>Vanilla</td><td>1</td><td>61.9 64.7</td><td></td><td>58.1</td><td>1862</td><td>85.9</td><td>69.5</td><td>78.5</td><td>58.2</td><td>100%</td></tr><tr><td>LLaVA-1.5-7B</td><td colspan="14">Retain 192 Tokens (33.3%)</td></tr><tr><td>FastV</td><td>ECCV’24</td><td>52.7</td><td>61.2</td><td>57.0</td><td>1612</td><td>64.8</td><td>67.3</td><td>67.1</td><td>52.5</td><td>89.0%</td></tr><tr><td>SparseVLM</td><td>ICML&#x27;25</td><td>59.5</td><td>64.1</td><td>53.7</td><td>1787</td><td>85.3</td><td>68.7</td><td>75.6</td><td>57.8</td><td>97.2%</td></tr><tr><td>DART</td><td>EMNLP&#x27;25</td><td>60.0</td><td>63.6</td><td>57.0</td><td>1856</td><td>82.8</td><td>69.8</td><td>76.7</td><td>57.4</td><td>98.3%</td></tr><tr><td>VisionZip</td><td>CVPR&#x27;25</td><td>59.3</td><td>63.0</td><td>57.3</td><td>1782</td><td>85.3</td><td>68.9</td><td>76.8</td><td>57.3</td><td>97.8%</td></tr><tr><td>HoloV</td><td>NeurIPS&#x27;25</td><td>59.0</td><td>65.4</td><td>58.0</td><td>1820</td><td>85.6</td><td>69.8</td><td>76.7</td><td>57.4</td><td>98.7%</td></tr><tr><td>SCOPE</td><td>NeurIPS&#x27;25</td><td>60.1</td><td>63.6</td><td>56.8</td><td>1804</td><td>86.4</td><td>68.8</td><td>77.2</td><td>57.7</td><td>98.3%</td></tr><tr><td>SpecFlow</td><td>ICML&#x27;26</td><td>58.3</td><td>65.8</td><td>56.1</td><td>1827</td><td>85.8</td><td>69.7</td><td>76.4</td><td>57.9</td><td>98.4%</td></tr><tr><td>SIEVE</td><td>Ours</td><td>60.6 64.6</td><td></td><td>58.4</td><td>1820</td><td>85.5</td><td>68.9</td><td>77.8</td><td>57.8</td><td>99.1%</td></tr><tr><td>LLaVA-1.5-7B</td><td colspan="14">Retain 128 Tokens (22.2%)</td></tr><tr><td>FastV</td><td>ECCV’24</td><td>56.1</td><td></td><td>56.4</td><td>1490</td><td>59.6</td><td>60.2</td><td>61.8</td><td>50.6</td><td>83.2%</td></tr><tr><td>SparseVLM</td><td>ICML&#x27;25</td><td>49.6 58.4</td><td>64.5</td><td>51.1</td><td>1746</td><td>85.0</td><td>68.6</td><td>73.8</td><td>56.7</td><td>95.6%</td></tr><tr><td>DART</td><td>EMNLP&#x27;25</td><td>58.7</td><td>63.2</td><td>57.5</td><td>1840</td><td>80.1</td><td>69.1</td><td>75.9</td><td>56.4</td><td>97.0%</td></tr><tr><td>VisionZip</td><td>CVPR&#x27;25</td><td>57.6</td><td>62.0</td><td>56.7</td><td>1761.7</td><td>83.2</td><td>68.9</td><td>75.6</td><td>56.8</td><td>96.4%</td></tr><tr><td>HoloV</td><td>NeurIPS’25</td><td>57.7</td><td>63.9</td><td>56.5</td><td>1802</td><td>84.0</td><td>69.8</td><td>75.5</td><td>56.8</td><td>97.2%</td></tr><tr><td>SCOPE</td><td>NeurIPS&#x27;25</td><td>59.7</td><td>62.5</td><td>56.9</td><td>1776</td><td>86.1</td><td>68.4</td><td>76.5</td><td>57.2</td><td>97.5%</td></tr><tr><td>SpecFlow</td><td>ICML&#x27;26</td><td>57.6</td><td>64.3</td><td>55.9</td><td>1794</td><td>84.9</td><td>69.9</td><td>75.3</td><td>56.8</td><td>97.2%</td></tr><tr><td>SIEVE</td><td>Ours</td><td>59.8</td><td>64.0</td><td>57.0</td><td>1815</td><td>85.8</td><td>68.8</td><td>77.1</td><td>57.6</td><td>98.7%</td></tr><tr><td>LLaVA-1.5-7B</td><td colspan="14"></td></tr><tr><td>FastV</td><td>ECCV’24 46.1</td><td></td><td>48.0</td><td>52.7</td><td>1256</td><td colspan="2">Retain 64 Tokens (11.1%) 48.0</td><td>51.1 55.0</td><td>47.8</td><td>74.0%</td></tr><tr><td>SparseVLM</td><td>ICML&#x27;25</td><td>53.8</td><td>60.1</td><td>52.7</td><td>1589</td><td>77.5</td><td>69.8</td><td>68.2</td><td>53.4</td><td>90.6%</td></tr><tr><td>DART</td><td>EMNLP&#x27;25</td><td>55.9</td><td>60.6</td><td>53.2</td><td>1765</td><td>73.9</td><td>69.8</td><td>72.4</td><td>54.4</td><td>92.8%</td></tr><tr><td>VisionZip</td><td>CVPR&#x27;25</td><td>55.1</td><td>60.1</td><td>55.4</td><td>1690</td><td>77.0</td><td>69.0</td><td>72.4</td><td>55.5</td><td>93.0%</td></tr><tr><td>HoloV</td><td>NeurIPS&#x27;25</td><td>55.3</td><td>63.3</td><td>55.1</td><td>1715</td><td>80.3</td><td>69.5</td><td>72.8</td><td>55.4</td><td>94.4%</td></tr><tr><td>SCOPE</td><td>NeurIPS&#x27;25</td><td>58.3</td><td>61.7</td><td>54.4</td><td>1698</td><td>83.9</td><td>68.6</td><td>75.3</td><td>56.6</td><td>95.4%</td></tr><tr><td>SpecFlow</td><td>ICML&#x27;26</td><td>55.3</td><td>63.7</td><td>54.3</td><td>1713</td><td>80.5</td><td>69.7</td><td>73.7</td><td>54.9</td><td>94.4%</td></tr><tr><td>SIEVE</td><td>Ours</td><td>58.0</td><td>62.2</td><td>55.2</td><td>1740</td><td>83.5</td><td>69.0</td><td>75.2</td><td>56.5</td><td>96.0%</td></tr></table>

Table 1: Performance comparison of diferent token reduction methods on LLaVA-1.5-7B.

where a larger $g _ { i j }$ means more similar residual directions and thus higher redundancy.

2. Diversity-Aware Greedy Selection. Based on the guided scores and residual similarities, SIEVE iteratively selects informative and non-redundant visual tokens. At each step, it selects

$$
i ^ { * } = \arg \operatorname* { m a x } _ { i \notin \mathcal { S } } \mathrm { S c o r e } _ { i } ,\tag{13}
$$

with $s$ the set of selected tokens, and then discounts the remaining candidates by their residual similarity to $i ^ { * }$

$$
\operatorname { S c o r e } _ { j }  \operatorname { S c o r e } _ { j } ( 1 - g _ { i ^ { * } j } ) , \quad j \notin \mathcal { S } .\tag{14}
$$

Since $g _ { i ^ { * } j } \in [ 0 , 1 ]$ is a dimensionless quantity normalized by the residual magnitudes, the factor $1 - g _ { i ^ { * } j }$ is a scaling coeficient in [0, 1] that attenuates Score<sub>j</sub> proportionally without altering its dimension. Candidates whose residual directions align with the selected token are thereby progressively suppressed, and the process continues until the budget K is reached.

## Experiments

## Experimental Setup

Models and Baselines. We evaluate SIEVE on three VLMs of diverse architectures, including LLaVA-1.5 (Liu et al. 2023), LLaVA-NeXT (Liu et al. 2024a), and Qwen2.5- VL (Bai et al. 2025).

Datasets. We report image-based results on eight standard benchmarks: GQA (Hudson and Manning 2019), MMBench and MMBench-CN (Liu et al. 2024b), MME (Fu et al. 2025), POPE (Li et al. 2023b), SQA (Lu et al. 2022), VQAv2 (Goyal et al. 2017), and TextVQA (Singh et al. 2019).

## Comparison Experimental Results

Results on LLaVA. Table 1 reports the results on LLaVA-1.5-7B. Here, Vanilla denotes the original model without visual-token reduction. On LLaVA-1.5-7B, SIEVE achieves the best average performance among methods with complete results under all token budgets. With 192, 128, and 64 retained visual tokens, SIEVE preserves 99.1%, 98.7%, and 96.0% of the Vanilla baseline, respectively. Notably, even when retaining only 64 visual tokens, SIEVE still surpasses SCOPE, the strongest baseline with complete results, by 0.6 percentage points on average, demonstrating its ability to preserve critical visual information under aggressive compression.

Results on LLaVA-NeXT-7B. Table 2 presents the corresponding comparison results. SIEVE likewise achieves the best reported average performance with complete results under the 320-token setting, preserving 97.5% of the Vanilla baseline. Compared with DART and HoloV, SIEVE improves the average performance by 1.4 and 1.3 percentage points, respectively. These results demonstrate that SIEVE generalizes efectively from the standard LLaVA-1.5 architecture to the higher-resolution LLaVA-NeXT architecture, which contains substantially more visual tokens.

<table><tr><td>Method</td><td>Venue</td><td>GQA</td><td>MMBench</td><td>MMBench-cn</td><td>MME</td><td>POPE</td><td>SQA</td><td>VQAv2</td><td>TextVQA</td><td>Avg</td></tr><tr><td>LLaVA-NeXT-7B</td><td colspan="8">Upper Bound, 2880 Tokens</td><td></td><td></td></tr><tr><td>Vanilla</td><td></td><td>64.2</td><td>67.4</td><td>60.6</td><td>1851</td><td>86.5</td><td>70.1</td><td>81.8</td><td>61.4</td><td>100%</td></tr><tr><td colspan="9">LLaVA-NeXT-7B</td><td rowspan="2"></td></tr><tr><td>FastV</td><td>ECCV’24</td><td></td><td>61.6</td><td>51.9</td><td>Retain 320 Tokens 1661</td><td>71.7</td><td>62.8</td><td>71.9</td></tr><tr><td>SparseVLM</td><td>ICML&#x27;25</td><td>55.9 56.1</td><td>60.6</td><td>54.5</td><td>1533</td><td>82.4</td><td>66.1</td><td>71.5</td><td>55.7 58.4</td><td>88.1% 90.3%</td></tr><tr><td>VisionZip</td><td>CVPR&#x27;25</td><td>59.3</td><td>63.1</td><td>55.6</td><td>1702</td><td>82.1</td><td>67.3</td><td>76.2</td><td>58.9</td><td>93.7%</td></tr><tr><td>DART</td><td>EMNLP&#x27;25</td><td>61.7</td><td>65.3</td><td>58.2</td><td>1710</td><td>84.1</td><td>68.4</td><td>79.1</td><td>58.7</td><td>96.1%</td></tr><tr><td>HoloV</td><td>NeurIPS&#x27;25</td><td>61.7</td><td>65.3</td><td>57.5</td><td>1738</td><td>83.9</td><td>68.9</td><td>79.5</td><td>58.7</td><td>96.2%</td></tr><tr><td>SpecFlow</td><td>ICML&#x27;26</td><td>62.5</td><td>66.7</td><td>56.8</td><td>1707</td><td>85.0</td><td>68.6</td><td>80.1</td><td>59.5</td><td>96.6%</td></tr><tr><td>SIEVE</td><td>Ours</td><td>60.9</td><td>66.5</td><td>59.6</td><td>1819</td><td>85.7</td><td>68.3</td><td>78.7</td><td>59.6</td><td>97.5%</td></tr></table>

Table 2: Performance comparison of diferent token reduction methods on LLaVA-NeXT-7B.
<table><tr><td>Method</td><td>Venue</td><td>MMBench</td><td>MME</td><td>POPE</td><td>SQA</td><td>MMBench-cn</td><td>GQA</td><td>TextVQA</td><td>Avg</td></tr><tr><td>Qwen2.5-VL-7B</td><td colspan="9"></td></tr><tr><td>Vanilla</td><td>一</td><td>83.4</td><td>2295</td><td>Upper Bound, 1296 Tokens 87.1</td><td></td><td>81.4</td><td>61.0</td><td>77.2</td><td>100.0%</td></tr><tr><td>Qwen2.5-VL-7B</td><td colspan="9"></td></tr><tr><td>FastV</td><td>ECCV’24</td><td>79.5</td><td>2306</td><td>Retain 33.3% Tokens 86.9</td><td>85.6</td><td>77.9</td><td>59.5</td><td>75.0</td><td>97.5%</td></tr><tr><td>HoloV</td><td>NeurIPS&#x27;25</td><td>81.2</td><td>2288.9</td><td>86.1</td><td>88.7</td><td>80.0</td><td>60.2</td><td>74.4</td><td>98.5%</td></tr><tr><td>SIEVE</td><td>Ours</td><td>82.5</td><td>2319</td><td>88.0</td><td>88.2</td><td>81.3</td><td>60.8</td><td>75.8</td><td>99.7%</td></tr><tr><td>Qwen2.5-VL-7B</td><td colspan="9">Retain 22.2% Tokens</td></tr><tr><td>FastV</td><td>ECCV’24</td><td>76.5</td><td>2243</td><td>85.1</td><td></td><td>75.2</td><td>57.8</td><td>73.5</td><td>95.0%</td></tr><tr><td>HoloV</td><td>NeurIPS&#x27;25</td><td>80.5</td><td>2268.8</td><td>85.2</td><td>84.8 88.2</td><td>79.6</td><td>59.5</td><td>71.7</td><td>97.2%</td></tr><tr><td>SIEVE</td><td>Ours</td><td>81.6</td><td>2313</td><td>87.3</td><td>87.5</td><td>80.2</td><td>60.3</td><td>75.1</td><td>98.9%</td></tr><tr><td>Qwen2.5-VL-7B</td><td colspan="9">Retain 11.1% Tokens</td></tr><tr><td>FastV HoloV</td><td>ECCV&#x27;24</td><td>71.2</td><td>2061</td><td>78.6</td><td>80.6</td><td>68.2</td><td>52.4</td><td>67.0</td><td>87.5%</td></tr><tr><td></td><td>NeurIPS’25</td><td>78.2</td><td>2246.5</td><td>83.7</td><td>87.6</td><td>77.1</td><td>57.5</td><td>65.7</td><td>94.4%</td></tr><tr><td>SIEVE</td><td>Ours</td><td>80.0</td><td>2273</td><td>86.0</td><td>86.6</td><td>77.6</td><td>58.9</td><td>72.5</td><td>96.7%</td></tr></table>

Table 3: Performance comparison of diferent token reduction methods on Qwen2.5-VL-7B.

Results on Qwen2.5-VL. Table 3 reports the results on Qwen2.5-VL-7B. We fix the input resolution to 1008×1008, corresponding to 1,296 visual tokens per image. Under token retention ratios of 33.3%, 22.2%, and 11.1%, SIEVE achieves 99.7%, 98.9%, and 96.7% average performance, respectively, consistently outperforming both FastV and HoloV. Notably, under the most aggressive 11.1% token budget, SIEVE surpasses FastV and HoloV by 9.2 and 2.3 percentage points, respectively, demonstrating its stronger ability to preserve critical visual information under severe compression. These results show that SIEVE generalizes effectively beyond the LLaVA family and maintains consistent performance advantages on the Qwen2.5-VL architecture.

## Ablation Study and Analysis

Hyperparameter Sensitivity. We further evaluate the sensitivity of SIEVE to the energy ratio η and the number of selected image-attending heads $H _ { \mathrm { t o p } } ,$ , with detailed results provided in Appendix D.

Table 4 shows part results. SIEVE achieves the best average retention of 96.59% with $\eta = 0 . 7 5$ and $H _ { \mathrm { t o p } } = 1 2 .$ Other configurations obtain highly similar results, ranging from 96.05% to 96.52%, indicating that SIEVE is robust to hyperparameter choices and does not rely on delicate tuning.

<table><tr><td> $\eta \backslash H _ { \mathrm { t o p } }$ </td><td>8</td><td>12</td><td>16</td><td>24</td><td>32</td></tr><tr><td>0.65</td><td>96.15</td><td>96.35</td><td>96.29</td><td>96.42</td><td>96.26</td></tr><tr><td>0.75</td><td>96.17</td><td>96.59</td><td>96.07</td><td>96.05</td><td>96.20</td></tr><tr><td>0.85</td><td>96.25</td><td>96.48</td><td>96.15</td><td>96.16</td><td>96.29</td></tr><tr><td>0.95</td><td>96.37</td><td>96.40</td><td>96.27</td><td>96.47</td><td>96.35</td></tr></table>

Table 4: Hyperparameter sensitivity of SIEVE on LLaVA-1.5-7B at 11.1% visual-token retention.

Component Ablation. To assess the contribution of each component, we ablate SIEVE on LLaVA-1.5-7B under 11.1% visual token retention from two aspects: scoring signals and diversity selection, as shown in Table 5.

For scoring, using only the attention score and only the CMR score achieves 96.07% and 96.00%, respectively, showing that the geometric residual captured by CMR is itself an efective selection signal rather than an auxiliary term of attention. Combining the two signals multiplicatively gives the full SIEVE result of 96.59%, outperforming either signal alone. In contrast, additive fusion reaches only 96.20%, lagging behind multiplicative fusion.

<table><tr><td>Group</td><td>Variant</td><td>Avg. %Full</td><td>∆Avg.</td></tr><tr><td>Full</td><td>SIEVE full</td><td>96.59</td><td>0.00</td></tr><tr><td>Score</td><td>Attention score only</td><td>96.07</td><td>-0.52</td></tr><tr><td>Score</td><td>CMR score only</td><td>96.00</td><td>-0.59</td></tr><tr><td>Score</td><td>(Attention + CMR) score</td><td>96.20</td><td>0.39</td></tr><tr><td>Diversity</td><td>Top-K selection</td><td>90.13</td><td>-6.46</td></tr><tr><td>Diversity</td><td>Raw-space diversity</td><td>95.70</td><td>-0.89</td></tr></table>

Table 5: Component ablation of SIEVE on LLaVA-1.5-7B at 11.1% visual token retention. Avg. %Full reports average performance on SQA, TextVQA, POPE, and MME relative to the full-token baseline, while ∆Avg. is measured against full SIEVE. Ablations cover scoring and diversity selection.

For diversity selection, direct top-k selection drops sharply to 90.13% (−6.46), indicating that selecting tokens solely by the initial scores tends to retain redundant tokens. Rawspace diversity recovers most of the performance to 95.70% (−0.89), but still underperforms full SIEVE, suggesting that enforcing diversity in the residual space is more efective for suppressing redundant visual information.

Efectiveness Analysis of the Fusion Strategy. Attention and CMR capture complementary aspects of token importance, namely semantic relevance and geometric residual, respectively. The former focuses on regions directly related to the query, whereas the latter highlights visual regions that cannot be well explained by the text subspace. As shown in Figure 4, although both respond to informative regions, the overlap between their selected tokens remains consistently low, with a low average overlap ratio across the four benchmarks (see Appendix E for details). This indicates that the two signals are complementary rather than redundant: relying on either one alone inevitably misses tokens preferred by the other. Consequently, combining the two scores preserves a more complete set of informative visual tokens, leading to better performance than using either attention or CMR alone.

Furthermore, multiplicative fusion is better aligned with this objective than additive fusion. Additive fusion is sensitive to diferences in score magnitude, allowing one signal to dominate the final ranking. In contrast, multiplicative fusion favors tokens that simultaneously exhibit semantic relevance and substantial residual information, while naturally suppressing background or noisy tokens that receive an abnormally high score from only one signal. As a result, additive fusion achieves only 96.20%, lower than the 96.59% obtained by multiplicative fusion.

Eficiency Analysis. SIEVE achieves a strong trade-of between eficiency and performance. As shown in Table 6, compared with the full model, SIEVE achieves 2.49×/1.44× total runtime speedups and 3.62×/2.01× prefill speedups on LLaVA-NeXT/LLaVA-1.5, while reducing the KV cache from 1156MB/319MB to 192MB/63MB. Even compared with Random pruning, which approximates an ideal fast baseline, SIEVE remains close in runtime, with only 19%/7% additional total time, while improving POPE F1 by 6.6/7.6 points. Compared with SparseVLM, SIEVE is faster on both models in total runtime and prefill time while maintaining higher accuracy.

<table><tr><td rowspan=1 colspan=2>Model   Method</td><td rowspan=1 colspan=1>TotalTime</td><td rowspan=1 colspan=1>PrefillTime</td><td rowspan=1 colspan=1>KV Cache(MB)</td><td rowspan=1 colspan=1>POPEF1</td></tr><tr><td rowspan=1 colspan=2>FullLLaVA-  RandomNeXT-7B SparseVLMSIEVE</td><td rowspan=1 colspan=1>46:1115:3335:5518:32</td><td rowspan=1 colspan=1>37:217:2026:5710:20</td><td rowspan=1 colspan=1>1156192200192</td><td rowspan=1 colspan=1>86.579.182.485.7</td></tr><tr><td rowspan=4 colspan=2>FullLLaVA-  Random1.5-7B   SparseVLMSIEVE</td><td rowspan=1 colspan=1>16:39</td><td rowspan=1 colspan=1>9:55</td><td rowspan=1 colspan=1>319</td><td rowspan=4 colspan=1>85.975.877.583.4</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>10:49</td><td rowspan=1 colspan=1>4:10</td><td rowspan=3 colspan=1>636663</td></tr><tr><td rowspan=1 colspan=1>12:58</td><td rowspan=1 colspan=1>6:18</td></tr><tr><td rowspan=1 colspan=1>11:36</td><td rowspan=1 colspan=1>4:56</td></tr></table>

Table 6: Eficiency comparison on POPE for LLaVA-NeXT-7B and LLaVA-1.5-7B. All token reduction methods retain 11.1% visual tokens.

![](images/32641b6947c6228e2feac263510d0364c28fb8a39e12e5b7f2d3f1a989fe68b2.jpg)  
Figure 4: Diferent selection between attention and CMR.

## Conclusion

In this work, we reveal cross-modal information absorption in VLMs and propose CMA for quantification. We design CMR via Tikhonov regularized subspace projection and develop the training-free SIEVE to compress visual tokens. Extensive evaluations over LLaVA, Qwen and LLaVA-NeXT verify that SIEVE substantially accelerates inference without obvious performance loss.

## References

Alayrac, J.-B.; Donahue, J.; Luc, P.; Miech, A.; Barr, I.; Hasson, Y.; Lenc, K.; Mensch, A.; Millican, K.; Reynolds, M.; et al. 2022. Flamingo: a visual language model for fewshot learning. Advances in neural information processing systems, 35: 23716–23736.

Alvar, S. R.; Singh, G.; Akbari, M.; and Zhang, Y. 2025. Divprune: Diversity-based visual token pruning for large multimodal models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, 9392–9401.

Bai, J.; Bai, S.; Yang, S.; Wang, S.; Tan, S.; Wang, P.; Lin, J.; Zhou, C.; and Zhou, J. 2023. Qwen-vl: A frontier large

vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 1(2): 3.

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; et al. 2025. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923.

Bolya, D.; Fu, C.-Y.; Dai, X.; Zhang, P.; Feichtenhofer, C.; and Hofman, J. 2023. Token Merging: Your ViT But Faster. In The Eleventh International Conference on Learning Representations.

Chen, J.; Liu, X.; Wen, Z.; Wang, Y.; Huang, S.; and Chen, H. 2026. Variation-aware Vision Token Dropping for Faster Large Vision-Language Models. arXiv:2509.01552.

Chen, L.; Zhao, H.; Liu, T.; Bai, S.; Lin, J.; Zhou, C.; and Chang, B. 2024a. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision, 19–35. Springer.

Chen, Z.; Wu, J.; Wang, W.; Su, W.; Chen, G.; Xing, S.; Zhong, M.; Zhang, Q.; Zhu, X.; Lu, L.; et al. 2024b. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 24185– 24198.

Dai, W.; Li, J.; Li, D.; Tiong, A.; Zhao, J.; Wang, W.; Li, B.; Fung, P. N.; and Hoi, S. 2023. Instructblip: Towards generalpurpose vision-language models with instruction tuning. Advances in neural informationprocessing systems, 36: 49250– 49267.

Deng, J.; Li, W.; Zhou, J. T.; and He, Y. 2025. SCOPE: Saliency-Coverage Oriented Token Pruning for Eficient Multimodel LLMs. arXiv preprint arXiv:2510.24214.

Driess, D.; Xia, F.; Sajjadi, M. S.; Lynch, C.; Chowdhery, A.; Ichter, B.; Wahid, A.; Tompson, J.; Vuong, Q.; Yu, T.; et al. 2023. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378.

Fu, C.; Chen, P.; Shen, Y.; Qin, Y.; Zhang, M.; Lin, X.; Yang, J.; Zheng, X.; Li, K.; Sun, X.; et al. 2025. Mme: A comprehensive evaluation benchmark for multimodal large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Goyal, Y.; Khot, T.; Summers-Stay, D.; Batra, D.; and Parikh, D. 2017. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, 6904–6913.

Huang, S.; Dong, L.; Wang, W.; Hao, Y.; Singhal, S.; Ma, S.; Lv, T.; Cui, L.; Mohammed, O. K.; Patra, B.; et al. 2023. Language is not all you need: Aligning perception with language models. Advances in Neural Information Processing Systems, 36: 72096–72109.

Huang, W.; Zhai, Z.; Shen, Y.; Cao, S.; Zhao, F.; Xu, X.; Ye, Z.; and Lin, S. 2025. Dynamic-llava: Eficient multimodal large language models via dynamic vision-language context sparsification. In International Conference on Learning Representations, volume 2025, 69927–69955.

Hudson, D. A.; and Manning, C. D. 2019. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 6700–6709.

Li, B.; Zhang, Y.; Guo, D.; Zhang, R.; Li, F.; Zhang, H.; Zhang, K.; Zhang, P.; Li, Y.; Liu, Z.; et al. 2024. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326.

Li, J.; Li, D.; Savarese, S.; and Hoi, S. 2023a. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, 19730–19742. PMLR.

Li, K.; He, Y.; Wang, Y.; Li, Y.; Wang, W.; Luo, P.; Wang, Y.; Wang, L.; and Qiao, Y. 2025a. Videochat: Chat-centric video understanding. Science China Information Sciences, 68(10): 200102.

Li, W.; Yuan, Y.; Liu, J.; Tang, D.; Wang, S.; Qin, J.; Zhu, J.; and Zhang, L. 2025b. Tokenpacker: Eficient visual projector for multimodal llm. International Journal of Computer Vision, 133(10): 6794–6812.

Li, Y.; Du, Y.; Zhou, K.; Wang, J.; Zhao, X.; and Wen, J.-R. 2023b. Evaluating Object Hallucination in Large Vision-Language Models. In The 2023 Conference on Empirical Methods in Natural Language Processing.

Lin, B.; Ye, Y.; Zhu, B.; Cui, J.; Ning, M.; Jin, P.; and Yuan, L. 2024. Video-llava: Learning united visual representation by alignment before projection. In Proceedings of the 2024 conference on empirical methods in natural language processing, 5971–5984.

Lin, Z.; Lin, M.; Lin, L.; and Ji, R. 2025. Boosting multimodal large language models with visual tokens withdrawal for rapid inference. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 5334–5342.

Liu, H.; Li, C.; Li, Y.; Li, B.; Zhang, Y.; Shen, S.; and Lee, Y. J. 2024a. Llavanext: Improved reasoning, ocr, and world knowledge.

Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023. Visual instruction tuning. Advances in neural information processing systems, 36: 34892–34916.

Liu, Y.; Duan, H.; Zhang, Y.; Li, B.; Zhang, S.; Zhao, W.; Yuan, Y.; Wang, J.; He, C.; Liu, Z.; et al. 2024b. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, 216–233. Springer.

Lu, P.; Mishra, S.; Xia, T.; Qiu, L.; Chang, K.-W.; Zhu, S.-C.; Tafjord, O.; Clark, P.; and Kalyan, A. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35: 2507–2521.

Maaz, M.; Rasheed, H.; Khan, S.; and Khan, F. 2024. Videochatgpt: Towards detailed video understanding via large vision and language models. In Proceedings ofthe 62ndAnnual Meeting of the Association for Computational Linguistics, 12585–12602.

Shang, Y.; Cai, M.; Xu, B.; Lee, Y. J.; and Yan, Y. 2025. Llava-prumerge: Adaptive token reduction for eficient large multimodal models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 22857–22867.

Singh, A.; Natarajan, V.; Shah, M.; Jiang, Y.; Chen, X.; Batra, D.; Parikh, D.; and Rohrbach, M. 2019. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 8317–8326.

Team, G.; Anil, R.; Borgeaud, S.; Alayrac, J.-B.; Yu, J.; Soricut, R.; Schalkwyk, J.; Dai, A. M.; Hauth, A.; Millican, K.; et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Wen, Z.; Gao, Y.; Wang, S.; Zhang, J.; Zhang, Q.; Li, W.; He, C.; and Zhang, L. 2025. Stop looking for important tokens in multimodal language models: Duplication matters more. arXiv preprint arXiv:2502.11494.

Xing, L.; Huang, Q.; Dong, X.; Lu, J.; Zhang, P.; Zang, Y.; Cao, Y.; He, C.; Wang, J.; Wu, F.; et al. 2024. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:2410.17247.

Yang, C.; Sui, Y.; Xiao, J.; Huang, L.; Gong, Y.; Li, C.; Yan, J.; Bai, Y.; Sadayappan, P.; Hu, X.; et al. 2025a. Topv: Compatible token pruning with inference time optimization for fast and low-memory multimodal vision language model. In Proceedings of the Computer Vision and Pattern Recognition Conference, 19803–19813.

Yang, S.; Chen, Y.; Tian, Z.; Wang, C.; Li, J.; Yu, B.; and Jia, J. 2025b. Visionzip: Longer is better but not necessary in vision language models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, 19792–19802.

Ye, Q.; Xu, H.; Xu, G.; Ye, J.; Yan, M.; Zhou, Y.; Wang, J.; Hu, A.; Shi, P.; Shi, Y.; et al. 2023. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178.

Zhang, H.; Ou, C.; Yan, D.; Wang, P.; Yan, Q.; Zhang, Y.; Li, Y.; and Xiao, R. 2026. TRIO: Token Reduction via Inference-Objective Guidance for Eficient Vision-Language Models. arXiv:2602.04657.

Zhang, Q.; Liu, M.; Li, L.; Lu, M.; Zhang, Y.; Pan, J.; She, Q.; and Zhang, S. 2025a. Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs. arXiv preprint arXiv:2506.10967.

Zhang, Y.; Fan, C.-K.; Ma, J.; Zheng, W.; Huang, T.; Cheng, K.; Gudovskiy, D. A.; Okuno, T.; Nakata, Y.; Keutzer, K.; and Zhang, S. 2025b. SparseVLM: Visual Token Sparsification for Eficient Vision-Language Model Inference. In Fortysecond International Conference on Machine Learning.

Zhu, D.; Shen, X.; Li, X.; Elhoseiny, M.; et al. 2024. Minigpt-4: Enhancing vision-language understanding with advanced large language models. In International Conference on Learning Representations, volume 2024, 18378–18394.

Zou, X.; Lu, D.; Wang, Y.; Yan, Y.; Lyu, Y.; Zheng, X.; Zhang, L.; and Hu, X. 2025. Don’t Just Chase" Highlighted Tokens" in MLLMs: Revisiting Visual Holistic Context Retention. arXiv preprint arXiv:2510.02912.

## Appendix

## A Additional Visualization of Cross-Modal Absorption

Figure 5 provides the complete visualization over all 32 layers, ofering a more detailed view of how visual tokens evolve during layer-wise forward propagation. Overall, visual tokens are progressively absorbed by the text subspace during layerwise forward propagation. As the layer depth increases, a large number of visual tokens gradually exhibit lower CMR and form a more concentrated distribution near the text PCA plane. This suggests that the model continuously transforms visual representations into a subspace that becomes increasingly aligned with text representations in deeper layers.

Meanwhile, this absorption process does not occur uniformly across all visual tokens, but instead exhibits a highly non-uniform and sparse structure. Most visual tokens gradually become redundant with the text subspace, meaning that their information can be well explained by text-related representations. However, a small subset of tokens still maintains relatively high CMR even in deeper layers, indicating that they are not fully absorbed by the text subspace and may preserve unique visual information. This phenomenon supports our key observation: visual information in multimodal models does not disappear synchronously as a whole, but is selectively absorbed by text representations layer by layer, where most tokens gradually become redundant while a few tokens remain complementary.

## B Mathematical Derivation: The Scale of CMA and the Random Baseline

Section Analysis shows that the final-layer CMA value is usually around 0.2–0.4. Although this value may appear small under the naive scale of [0, 1], it should be interpreted relative to the dimensionality of the text subspace. Since the number of text tokens is much smaller than the hidden dimension, the text subspace is intrinsically low-dimensional and cannot explain all visual information, even in the most favorable case.

This appendix provides a simple derivation of the efective scale of CMA. We first discuss the maximum CMA that a fixed low-dimensional subspace can achieve, then derive the random baseline when visual and text representations are independent, and finally compare these quantities with empirical observations.

## B.1 Upper Bound of CMA Under a Fixed Subspace Dimension

Let $V _ { \mathrm { v i s } } \in \mathbb R ^ { N _ { v } \times D }$ denote the visual-token matrix at a given layer, where $N _ { v }$ is the number of visual tokens and D is the hidden dimension. Let $U \in \mathbb { R } ^ { D \times k }$ be an orthonormal basis of a k-dimensional subspace, satisfying $U ^ { \top } U = I _ { k }$ . The projection of visual tokens onto this subspace is

$$
\mathrm { p r o j } _ { U } ( V _ { \mathrm { v i s } } ) = V _ { \mathrm { v i s } } U U ^ { \top } .\tag{15}
$$

The corresponding CMA is defined as the fraction ofvisual information that can be explained by this subspace:

$$
\begin{array} { r l } & { \mathrm { C M A } ( U ) = 1 - \frac { \| V _ { \mathrm { v i s } } - V _ { \mathrm { v i s } } U U ^ { \top } \| _ { F } ^ { 2 } } { \| V _ { \mathrm { v i s } } \| _ { F } ^ { 2 } } } \\ & { \qquad = \frac { \| V _ { \mathrm { v i s } } U \| _ { F } ^ { 2 } } { \| V _ { \mathrm { v i s } } \| _ { F } ^ { 2 } } . } \end{array}\tag{16}
$$

Here, the denominator denotes the total amount of visual information, while the numerator measures the part that lies in the subspace spanned by U.

To understand the maximum possible value of CMA, we decompose the visual representation into its principal directions. Let the singular values of $V _ { \mathrm { v i s } }$ be

$$
\sigma _ { 1 } \geq \sigma _ { 2 } \geq \cdot \cdot \cdot \geq \sigma _ { R } ,\tag{17}
$$

where $R = \operatorname* { m i n } ( N _ { v } , D )$ . The total visual information can then be written as

$$
\| V _ { \mathrm { v i s } } \| _ { F } ^ { 2 } = \sum _ { r = 1 } ^ { R } { \sigma _ { r } ^ { 2 } } .\tag{18}
$$

Intuitively, $\sigma _ { r } ^ { 2 }$ measures how much information is contained in the r-th principal direction. Therefore, if we can only use a k-dimensional subspace to explain the visual representation, the best possible choice is to align this subspace with the top-k principal directions of $V _ { \mathrm { v i s } }$ . These directions contain the largest amount of visual information.

Thus, for any k-dimensional subspace $U ,$ , CMA is upperbounded by

$$
\operatorname { C M A } ( U ) \leq { \frac { \sigma _ { 1 } ^ { 2 } + \sigma _ { 2 } ^ { 2 } + \cdot \cdot \cdot + \sigma _ { k } ^ { 2 } } { \sigma _ { 1 } ^ { 2 } + \sigma _ { 2 } ^ { 2 } + \cdot \cdot \cdot + \sigma _ { R } ^ { 2 } } } .\tag{19}
$$

The equality holds only when the subspace U is exactly aligned with the top-k principal directions of $V _ { \mathrm { v i s } }$

This result indicates that the upper bound of CMA is not 1.0. Instead, it is determined by how much visual information is contained in the top-k principal directions. In other words, a low-dimensional subspace is not able to explain all visual information.

In LLaVA-1.5-7B, the number of post-image text tokens is typically about $N _ { t } \ \approx \ 3 0$ , while the hidden dimension is $\ddot { D } = \dot { 4 } 0 9 6$ Therefore, the text subspace is very lowdimensional. Empirically, the top 30 principal directions of visual representations usually account for about 40%–70% of the total visual information. As a result, even if the text subspace were perfectly aligned with the dominant visual directions, the theoretical upper bound of CMA would be around 0.4–0.7, rather than 1.0.

This explains why a final-layer CMA value of 0.2–0.4 should not be considered small. It already occupies a substantial portion of the efective range that a low-dimensional text subspace can reach.

## B.2 Random Baseline Under Independent Visual and Text Representations

We next consider the opposite case, where visual and text representations have no systematic alignment. In this case, the text subspace can be regarded as a random k-dimensional subspace in a D-dimensional space.

For a random k-dimensional subspace, the expected fraction of any vector that falls into this subspace is approximately

$$
{ \frac { k } { D } } .\tag{20}
$$

Therefore, when visual and text representations are independent, the expected CMA is

$$
\mathbb { E } [ \mathrm { C M A } ] = \frac { k } { D } \approx \frac { N _ { t } } { D } ,\tag{21}
$$

where $k = N _ { t }$ corresponds to the efective dimension of the text subspace.

For LLaVA-1.5-7B, we have $N _ { t } \approx 3 0$ and D = 4096, which gives

$$
\frac { N _ { t } } { D } \approx \frac { 3 0 } { 4 0 9 6 } \approx 0 . 0 0 7 3 .\tag{22}
$$

Empirically, the layer-0 CMA values are 0.008 on POPE, 0.009 on MME, and 0.017 on TextVQA. The values on POPE and MME are very close to the random baseline, while the value on TextVQA is slightly higher, possibly because this task already induces mild visual-text alignment in shallow layers.

Overall, the shallow-layer CMA values are of the same order as the random baseline. This suggests that visual and text representations are approximately independent in shallow layers, and that meaningful cross-modal absorption has not yet occurred.

## B.3 The Efective Scale for Interpreting CMA

Combining the upper-bound analysis and the randombaseline analysis, CMA should not be interpreted directly on the naive scale of [0, 1]. Instead, its efective scale is approximately

$$
\left[ { \frac { N _ { t } } { D } } , { \frac { \sigma _ { 1 } ^ { 2 } + \cdot \cdot \cdot + \sigma _ { N _ { t } } ^ { 2 } } { \sigma _ { 1 } ^ { 2 } + \cdot \cdot \cdot + \sigma _ { R } ^ { 2 } } } \right] \approx [ 0 . 0 0 7 , 0 . 4 - 0 . 7 ] .\tag{23}
$$

The left endpoint corresponds to the random baseline, where visual and text representations are nearly independent. The right endpoint corresponds to the ideal case, where the text subspace is perfectly aligned with the dominant visual directions.

Table 7 shows representative CMA values on POPE for LLaVA-1.5-7B.

<table><tr><td>Layer</td><td>CMA</td><td></td><td>vs. Baseline vs. Bound (0.5)</td></tr><tr><td>0</td><td>0.008</td><td>≈1×</td><td>1.6%</td></tr><tr><td>15</td><td>0.035</td><td>4.8×</td><td>7%</td></tr><tr><td>25</td><td>0.095</td><td>13×</td><td>19%</td></tr><tr><td>31</td><td>0.209</td><td>29×</td><td>42%</td></tr></table>

Table 7: CMA at representative layers of LLaVA-1.5-7B on POPE, together with their relative positions on the efective scale.

As shown in the table, the CMA value at layer 0 is almost the same as the random baseline. In contrast, the final-layer

CMA is about 29× larger than the random baseline and reaches around 42% of the theoretical upper bound.

Therefore, although the absolute CMA value at the final layer is only 0.209, it is already far above random alignment and occupies a considerable portion of the attainable range of a low-dimensional text subspace. This supports our observation that visual tokens are progressively absorbed by the text subspace during the forward propagation of VLMs, revealing a systematic cross-modal phenomenon inside the model.

## C FlashAttention-Compatible Attention Scoring

SIEVE uses an attention signal to estimate how strongly post-image text tokens attend to visual tokens. For the h-th attention head, let its causal attention matrix be

$$
P ^ { h } = \mathrm { S o f t m a x } \left( \frac { Q ^ { h } ( K ^ { h } ) ^ { \top } } { \sqrt { d _ { h } } } + M \right) ,\tag{24}
$$

where M denotes the causal attention mask and $d _ { h }$ is the head dimension. Let V denote the set of visual-token positions and $T _ { \mathrm { p o s t } }$ denote the set of post-image text-token positions. The attention score required by SIEVE is defined as

$$
a _ { i } = \sum _ { h \in H _ { \mathrm { t o p } } } \sum _ { t \in T _ { \mathrm { p o s t } } } P _ { t , i } ^ { h } , \quad i \in \mathcal { V } .\tag{25}
$$

This score measures how much visual token i is attended to by post-image text tokens at the current layer. A direct implementation would require explicitly accessing the submatrix $P _ { T _ { \mathrm { D o s t } } , \nu } ^ { h }$ . However, FlashAttention does not materialize the full $N \times N$ attention matrix, but directly computes the attention output in a block-wise streaming manner:

$$
O ^ { h } = P ^ { h } V ^ { h } .\tag{26}
$$

Therefore, explicitly reading $P _ { t , i } ^ { h }$ is incompatible with the memory-eficient design of FlashAttention.

To avoid materializing $P ^ { h }$ , SIEVE adopts a forward-only dual-flash aggregation strategy to compute the required textto-visual attention statistics. The idea is related to the dualflash operation used in SparseVLM (Zhang et al. 2025b), but the auxiliary value matrix is designed diferently. SparseVLM typically marks selected text raters in the auxiliary value matrix to obtain token-to-rater attention statistics. In contrast, SIEVE requires the attention from post-image text queries to visual keys; therefore, the auxiliary value matrix should mark visual-token positions.

Let the visual-token positions be

$$
{ \mathcal { V } } = \{ v _ { 1 } , v _ { 2 } , \ldots , v _ { N _ { v } } \} .\tag{27}
$$

We construct an auxiliary value matrix $\widetilde { V } \in \mathbb R ^ { N \times N _ { v } }$ whose j-th row is defined as

$$
\widetilde V [ j , : ] = \left\{ { \begin{array} { l l } { e _ { k } , } & { j = v _ { k } , v _ { k } \in \mathcal { V } , } \\ { \mathbf { 0 } , } & { \mathrm { o t h e r w i s e } , } \end{array} } \right.\tag{28}
$$

where $e _ { k }$ is the k-th standard basis vector. Thus, each visual token is assigned an independent marker channel, while all

non-visual tokens are assigned zero auxiliary values. Using the same $Q ^ { h }$ and $K ^ { h }$ , we compute the auxiliary attention output as

$$
{ \widetilde { O } } ^ { h } = P ^ { h } { \widetilde { V } } .\tag{29}
$$

Since $\widetilde { V }$ is nonzero only at visual-token positions, for any post-image text token $t \in T _ { \mathrm { p o s t } }$ , we have

$$
\widetilde { O } _ { t } ^ { h } = \sum _ { j = 1 } ^ { N } P _ { t , j } ^ { h } \widetilde { V } [ j , : ] = \sum _ { k = 1 } ^ { N _ { v } } P _ { t , v _ { k } } ^ { h } e _ { k } .\tag{30}
$$

Therefore, the k-th dimension of $\widetilde { O } _ { t } ^ { h }$ is exactly the attention value $P _ { t , v _ { k } } ^ { h }$ . Aggregating over all post-image text tokens gives the text-to-visual attention received by each visual token under head h:

$$
g _ { v _ { k } } ^ { h } = \sum _ { t \in T _ { \mathrm { p o s t } } } \widetilde { O } _ { t , k } ^ { h } = \sum _ { t \in T _ { \mathrm { p o s t } } } P _ { t , v _ { k } } ^ { h } .\tag{31}
$$

This is equivalent to explicitly extracting $P _ { T _ { \mathrm { p o s t } } , \nu } ^ { h }$ and summing over the text dimension, but it only requires computing $P ^ { h } \widetilde { V }$ and never materializes the full attention matrix.

After obtaining the per-head visual attention scores, SIEVE performs head selection. For each head, we compute its total visual attention mass as

$$
m _ { h } = \sum _ { i \in \mathcal { V } } g _ { i } ^ { h } .\tag{32}
$$

A larger $m _ { h }$ indicates that the post-image text tokens in this head attend more strongly to the visual region, making the head more suitable for estimating visual-token relevance. SIEVE selects the top image-attending heads:

$$
H _ { \mathrm { t o p } } = \mathrm { T o p K } _ { h } ( m _ { h } ) ,\tag{33}
$$

and aggregates the final visual-token attention score over the selected heads:

$$
a _ { i } = \sum _ { h \in H _ { \mathrm { t o p } } } g _ { i } ^ { h } .\tag{34}
$$

The resulting attention score $a _ { i }$ is then combined with the CMR score for residual-space diversity selection.

Importantly, this auxiliary computation does not modify the normal Transformer forward pass. The main forward pass still computes the hidden states using the original value matrix:

$$
O ^ { h } = P ^ { h } V ^ { h } .\tag{35}
$$

The auxiliary forward pass is only used to obtain the attention statistics required for pruning:

$$
{ \widetilde { O } } ^ { h } = P ^ { h } { \widetilde { V } } .\tag{36}
$$

Thus, SIEVE does not explicitly materialize the full attention matrix or alter the SIEVE FlashAttention computation. By rewriting text-to-visual attention aggregation as $P ^ { h } \widetilde { V }$ SIEVE preserves post-image-text-guided attention scoring while remaining compatible with the streaming computation of FlashAttention.

## D Ablation Study

## D.1 Detailed Hyperparameter Sensitivity

We provide a more detailed hyperparameter sensitivity analysis of SIEVE under the 11.1% visual-token retention setting on LLaVA-1.5-7B. The sweep covers two key hyperparameters: the energy ratio η for Tikhonov-regularized textsubspace projection, and the number of selected imageattending heads $H _ { \mathrm { t o p } }$ for attention-score aggregation. For each configuration, we compute the average percentage over the full-token baseline across SQA, TextVQA, POPE, and MME.

Table 8 summarizes the efect of $H _ { \mathrm { t o p } }$ . For each value of $H _ { \mathrm { t o p } } .$ , we report the mean, best, and worst average performance over all tested energy ratios. The results show that selecting too few heads leads to slightly lower performance, suggesting that overly aggressive head filtering may miss useful visual relevance signals. The best mean performance is achieved when $H _ { \mathrm { t o p } } \doteq 1 2$ , indicating that a moderate number of image-attending heads provides a good trade-of between capturing visual relevance and filtering noisy heads.

<table><tr><td> $H _ { \mathrm { t o p } }$ </td><td>Mean over η</td><td>Best over η</td><td>Worst over η</td></tr><tr><td>1</td><td>95.98</td><td>96.13</td><td>95.82</td></tr><tr><td>2</td><td>96.04</td><td>96.19</td><td>95.93</td></tr><tr><td>4</td><td>96.04</td><td>96.21</td><td>95.79</td></tr><tr><td>8</td><td>96.20</td><td>96.37</td><td>96.05</td></tr><tr><td>12</td><td>96.42</td><td>96.59</td><td>96.29</td></tr><tr><td>16</td><td>96.24</td><td>96.51</td><td>96.05</td></tr><tr><td>20</td><td>96.24</td><td>96.44</td><td>96.12</td></tr><tr><td>24</td><td>96.32</td><td>96.52</td><td>96.05</td></tr><tr><td>28</td><td>96.33</td><td>96.51</td><td>96.19</td></tr><tr><td>32</td><td>96.27</td><td>96.52</td><td>96.05</td></tr></table>

Table 8: Sensitivity to the number of selected imageattending heads $H _ { \mathrm { t o p } }$ . For each $H _ { \mathrm { t o p } }$ , we report the mean, best, and worst Avg. %Full over all tested energy ratios.

## D.2 Efect of Energy Ratio

Table 9 summarizes the efect of the energy ratio $\eta .$ For each value of $\eta ,$ , we report the mean, best, and worst average performance over all tested values of $H _ { \mathrm { t o p } }$ . The mean performance varies only mildly across diferent energy ratios, showing that SIEVE is robust to the exact threshold used to determine the Tikhonov regularization scale. Although the best single configuration appears at $\eta = 0 . 7 5$ , the overall mean performance remains stable across a wide range of values, confirming that the CMR computation does not rely on a carefully tuned energy threshold.

Overall, these results demonstrate that SIEVE is robust to both hyperparameters. The number of selected imageattending heads has a slightly larger efect, where moderate head selection performs best. In contrast, the energy ratio has only a mild influence, suggesting that the Tikhonovregularized CMR computation is stable across a broad range of subspace energy thresholds. This supports that the efectiveness of SIEVE mainly comes from cross-modal residual guidance and residual-space token selection, rather than from delicate hyperparameter tuning.

<table><tr><td>Energy Ratio η</td><td>Mean over  $H _ { \mathrm { t o p } }$ </td><td>Best over  $H _ { \mathrm { t o p } }$ </td><td>Worst over  $H _ { \mathrm { t o p } }$ </td></tr><tr><td>0.05</td><td>96.15</td><td>96.42</td><td>95.94</td></tr><tr><td>0.10</td><td>96.15</td><td>96.42</td><td>95.93</td></tr><tr><td>0.20</td><td>96.12</td><td>96.31</td><td>95.90</td></tr><tr><td>0.40</td><td>96.18</td><td>96.42</td><td>95.79</td></tr><tr><td>0.60</td><td>96.23</td><td>96.49</td><td>96.05</td></tr><tr><td>0.75</td><td>96.18</td><td>96.59</td><td>96.02</td></tr><tr><td>0.85</td><td>96.22</td><td>96.48</td><td>96.02</td></tr><tr><td>0.90</td><td>96.23</td><td>96.38</td><td>96.03</td></tr><tr><td>0.95</td><td>96.29</td><td>96.47</td><td>96.06</td></tr><tr><td>0.98</td><td>96.28</td><td>96.52</td><td>95.82</td></tr><tr><td>0.99</td><td>96.25</td><td>96.51</td><td>95.88</td></tr></table>

Table 9: Sensitivity to the energy ratio η used in the Tikhonov-regularized projection. For each $\eta ,$ we report the mean, best, and worst Avg. %Full over all tested values of $H _ { \mathrm { t o p } }$
<table><tr><td>Benchmark</td><td>Cov@22</td><td>Cov@54</td><td>Cov@92</td><td>Cov@115</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>MME</td><td>11.65</td><td>12.67</td><td>13.48</td><td>15.19</td><td>13.25</td></tr><tr><td>TextVQA</td><td>3.69</td><td>6.71</td><td>8.29</td><td>10.24</td><td>7.24</td></tr><tr><td>POPE</td><td>14.06</td><td>13.08</td><td>14.40</td><td>15.27</td><td>14.20</td></tr><tr><td>SQA</td><td>12.93</td><td>9.14</td><td>11.38</td><td>12.39</td><td>11.46</td></tr><tr><td>Average</td><td>10.58</td><td>10.40</td><td>11.89</td><td>13.27</td><td>11.54</td></tr></table>

Table 10: Average overlap of visual token selections by attention and CMR across all 32 layers. All values are percentages.

## E Complementarity between Attention and CMR

To verify that attention and CMR provide complementary rather than redundant token selection cues, we measure the overlap between the token subsets selected by the two scores before fusion. For a given layer and token budget $k ,$ let $S _ { k } ^ { \mathrm { a t t n } }$ and $\mathcal { S } _ { k } ^ { \mathrm { C M R } }$ denote the top-k visual token sets selected by the attention score and the CMR score, respectively. We define the overlap coverage as

$$
\mathrm { C o v @ } k = \frac { \left| S _ { k } ^ { \mathrm { a t t n } } \cap S _ { k } ^ { \mathrm { C M R } } \right| } { k } .\tag{37}
$$

A lower Cov@k indicates that the two scores prefer more diferent token subsets.

Table 10 summarizes the average coverage over all 32 layers on four benchmarks. The overlap remains consistently low across diferent token budgets. The average coverage over all settings is only 11.54%, and remains 13.27% even under the larger top-115 budget. This suggests that attention and CMR are not redundant ranking signals. Attention mainly reflects query-conditioned semantic relevance, while CMR captures visual tokens with distinctive residual information that cannot be well explained by the text subspace.

This low overlap further explains why using either score alone is suboptimal. Attention-only selection may miss geometrically distinctive visual tokens with weak semantic saliency, while CMR-only selection may miss query-relevant tokens whose residual information is less prominent. Therefore, fusing the two scores covers a more complete set of useful visual evidence and leads to better performance than either single score. Meanwhile, multiplicative fusion is not equivalent to a hard intersection of two top-k sets. Instead, it performs joint ranking over continuous scores: it retains tokens with both non-trivial semantic relevance and residual information, while suppressing background or noisy tokens whose support from either signal is close to zero. This also explains its advantage over additive fusion.

## F Complexity Analysis and Eficiency Discussion

## F.1 Complexity and Net Acceleration Condition

For an input sequence of length $n ,$ hidden dimension $d ,$ and FFN intermediate dimension $m ,$ the computational cost of a standard Transformer layer can be approximated as

$$
F ( n ) = 4 n d ^ { 2 } + 2 n ^ { 2 } d + 2 n d m .\tag{38}
$$

Here, $4 n d ^ { 2 }$ comes from the Q/K/V/O linear projections, $2 n ^ { 2 } d$ comes from self-attention, and 2ndm comes from the FFN module. Therefore, reducing the sequence length decreases not only the linear projection and FFN costs, but also the quadratic self-attention cost.

If pruning reduces the sequence length from n to $n ^ { \prime } { \mathrm { . } }$ , the per-layer computational saving is

$$
\begin{array} { l } { { F ( n ) - F ( n ^ { \prime } ) = ( n - n ^ { \prime } ) \bigl [ 4 d ^ { 2 } + 2 d ( n + n ^ { \prime } ) } } \\ { { \qquad + ~ 2 d m \bigr ] . } } \end{array}\tag{39}
$$

This shows that the saving depends on both the number of removed tokens, $n - n ^ { \prime }$ , and the original sequence scale, $n + n ^ { \prime }$ . Thus, token pruning becomes more beneficial when the input sequence is longer and visual tokens occupy a large portion of the sequence.

SIEVE introduces additional computation only at a few pruning layers. At a pruning layer, let $N _ { v }$ be the number of visual tokens, $N _ { t }$ be the number of post-image text tokens, and k be the number of retained visual tokens. The scoring cost of SIEVE, including CMR projection and residual-space diversity selection, can be summarized as

$$
\begin{array} { r l } & { C _ { \mathrm { s c o r e } } = O \big ( N _ { t } ^ { 2 } d + N _ { t } ^ { 3 } + N _ { v } N _ { t } d } \\ & { \qquad + N _ { v } N _ { t } ^ { 2 } + N _ { v } ^ { 2 } d + k N _ { v } \big ) . } \end{array}\tag{40}
$$

The first two terms come from text-subspace construction and Tikhonov-regularized solving, the next two terms come from projecting visual tokens onto the text subspace, and the last two terms come from residual-space similarity computation and greedy diversity selection. Since $N _ { t }$ is usually much smaller than $N _ { v }$ and $d ,$ the dominant cost is typically the residual-space Gram matrix:

$$
C _ { \mathrm { s c o r e } } \approx O ( N _ { v } ^ { 2 } d ) .\tag{41}
$$

Although SIEVE introduces this scoring overhead, it follows a “one-time scoring, multi-layer saving” pattern. The scoring is performed only at pruning layers, while the shortened sequence benefits all subsequent Transformer layers. Let $n _ { 0 }$ be the original sequence length, $n _ { l }$ be the actual sequence length at layer l, and $\mathcal { P }$ be the set of pruning layers. SIEVE achieves net acceleration when

$$
\sum _ { l } \left[ F ( n _ { 0 } ) - F ( n _ { l } ) \right] > \sum _ { p \in \mathcal { P } } C _ { \mathrm { s c o r e } } ^ { p } .\tag{42}
$$

That is, SIEVE improves inference eficiency when the accumulated saving from shorter sequences in later layers exceeds the scoring overhead introduced at pruning layers. This condition also explains why SIEVE is more efective when pruning is performed earlier, more tokens are removed, and the original sequence is longer.

## F.2 Discussion on Long-Sequence Inputs

For long-sequence inputs, the eficiency behavior depends on where the sequence length comes from. The first and most relevant case is long visual sequences, such as high-resolution images, multi-image inputs, or videos. In these scenarios, the number of visual tokens $N _ { v }$ increases, which also increases the scoring cost of SIEVE, especially the $O ( N _ { v } ^ { 2 } d )$ residual-space similarity computation. However, standard Transformer self-attention also contains the $O ( n ^ { 2 } d )$ term, and this cost is paid at every subsequent layer. Since SIEVE scoring is only performed at a few pruning layers, while the benefit of shorter sequences accumulates across many later layers, the total computational saving can outweigh the additional scoring cost when pruning is applied early.

The second case is long text prompts. If the post-image text length $N _ { t }$ becomes very large, the cost of text-subspace construction and Tikhonov projection may increase. In this case, one can use a fixed-length post-image text window or restrict the efective rank of the text subspace, so that the terms related to $N _ { t }$ remain controlled.

Overall, SIEVE is especially suitable for long visualsequence scenarios where visual tokens dominate the computational cost. In such cases, the original Transformer cost grows rapidly with sequence length, while SIEVE only introduces limited scoring overhead at a few pruning layers. The reduced token sequence is then reused by subsequent layers, leading to lower overall inference complexity and improved eficiency.

![](images/52d2a0da589491d13afb33917388df5a9cc2bb13d2ec8a326c8017287a8a0924.jpg)  
Figure 5: Full visualization of visual tokens progressively moving closer to the text subspace across all 32 layers. Each subplot corresponds to one Transformer layer. The point positions indicate the projections of visual tokens onto the text PCA plane, while the height and color represent CMR. A lower CMR indicates that the visual token is closer to the text subspace.