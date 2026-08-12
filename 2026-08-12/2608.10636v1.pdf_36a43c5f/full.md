# DistilVDR: A Compact End-to-End Visual Document Retriever via Dual-Student Distillation

Zhuchenyang Liu<sup>1</sup> Ziyi Wang<sup>2</sup> Yao Zhang<sup>1</sup> Yu Xiao<sup>1</sup>

<sup>1</sup>Aalto University, Finland <sup>2</sup>Independent Researcher, Netherlands zhuchenyang.liu@aalto.fi

## Abstract

Visual document retrieval (VDR) is dominated by multi-billion-parameter models that are slow to index at full corpus scale and expensive to serve. Prior compression routes either train a smaller multi-vector encoder from scratch or distil only the query side; neither yields a compact single-vector retriever end-to-end. We present DistilVDR, a 524 M end-to-end VDR system distilled bilaterally from a single 8 B vision-language teacher under a pointwise cosine alignment loss. All supervision comes from the frozen teacher’s embedding space, which was itself trained with relevance supervision, so the student objective needs no relevance labels, negative sampling, or contrastive term. We match VDR’s text-query and image-document input asymmetry with an asymmetric encoder-only student that concentrates visual capacity on the document side and keeps the query side at 70 M parameters. We release two variants that share the same encoders and training and differ only in the document encoder’s visual-tile budget: DistilVDR HiRes attains 61.74 average NDCG@5 on ViDoRe v1+v2+v3 (86.9 % of the 8 B teacher) and leads every reproduced sub-1 B baseline on the high-resolution-sensitive v3 benchmark, while DistilVDR-Fast attains 59.98 at a 3× smaller visual-token budget. Both variants store one million documents in a 15.6 times smaller index than the strongest sub-1 B multi vector baseline and index the corpus an order of magnitude faster. The code is available at https://github.com/Ryenhails/NanoVDR.

## 1 Introduction

Enterprise document search, technical documentation lookup, and visual retrieval-augmented generation (Yu et al., 2025) all require retrieving relevant document pages from a corpus given a text query. Visual document retrieval (VDR) addresses this task. VDR encodes each page directly as an image. This avoids the errors of an Optical Character

Recognition (OCR) or layout extraction pipeline and preserves tables, figures, and layout structure.

State-of-the-art VDR systems are heavy. Topperforming retrievers span two to eight billion parameters across both single-vector and multivector paradigms; examples include the singlevector Qwen3-VL-Embedding-8B (Li et al., 2026a) and the multi-vector Tomoro-ColQwen3-8B (TomoroAI, 2025). These models require more than 16 GB of GPU memory to embed a single document. Indexing a million-document corpus with such a model costs tens of GPU-hours.

Two lines of work have addressed this cost. The first line trains a smaller encoder from scratch and combines it with multi-vector late interaction (Khattab and Zaharia, 2020; Santhanam et al., 2022b,a). Per-token storage compensates for the reduced parameter count. Representative models include colSmol-500M (ViDoRe Team, 2024), SauerkrautLM-ColLFM2-450M (VAGO Solutions, 2025), and ModernVBERT (Teiletche et al., 2025), all built on the ColPali architecture (Faysse et al., 2025). Among these, ModernVBERT departs from the ColPali decoder lineage with an encoder-only design. It combines a SigLIP2 visual encoder (Tschannen et al., 2025) with a ModernBERT text backbone (Warner et al., 2025). Teiletche et al. (2025) report that this encoder-only design outperforms an equivalent causal decoder on document retrieval by 10.6 NDCG@5.

However, all systems in this line pay the multivector cost. Per-token storage inflates the index by an order of magnitude, and late-interaction MaxSim scoring inflates per-query latency by two orders of magnitude relative to a single dot product. A body of work attacks these costs post hoc, by pruning or merging patches in an already-built index (Ma et al., 2025; Yan et al., 2026), by fusing internal transformer layers into a more discriminative embedding (Li et al., 2026b), or by reranking single-vector candidates with a multivector model (Kim et al., 2025). All of it presumes an already-trained strong base model, so the training and deployment cost of the underlying multi-billion-parameter retriever remains. Going single-vector within the same architecture removes the index cost but loses quality: BiModernVBERT is 17.6 NDCG@5 below ColModernVBERT on ViDoRe v1 and 20.3 below it on v2 (Teiletche et al., 2025). A small encoder trained from scratch with contrastive learning therefore struggles to produce a single-vector representation that matches multivector retrieval quality.

The second line distils a small student from a large teacher. This is a well-established paradigm in text retrieval. A heavier teacher, typically a crossencoder reranker or a larger bi-encoder (Hofstätter et al., 2021; Chen et al., 2024a), supervises a lighter bi-encoder student, which inherits much of the teacher’s quality at a fraction of the inference cost. In visual document retrieval, the only prior work in this direction, to our knowledge, is NanoVDR (Liu et al., 2026). A 70 M DistilBERT (Sanh et al., 2019) query encoder is trained to match the embedding space of a 2 B vision-language teacher with a cosine alignment loss. The student matches the teacher within a few NDCG points despite being thirty times smaller, which shows that distillation is a viable route to compact single-vector VDR.

Neither line produces a compact single-vector retriever end-to-end. The from-scratch multi-vector line yields a small encoder but pays an order of magnitude in index size and two orders of magnitude in scoring latency. The distillation line yields a small query encoder but leaves the document encoder at the size of the teacher. NanoVDR keeps the 2 B teacher in the document path, so indexing a million-document corpus still costs tens of GPUhours and every deployment must host the teacher in memory.

We close this gap with DistilVDR, a 524 M end-to-end single-vector visual document retriever. Our contribution is a system: (i) a bilateral cosine alignment distillation setup in which each student regresses onto cached embeddings of a single 8 B vision-language teacher (Li et al., 2026a); (ii) an asymmetric encoder-only student matched to VDR’s input asymmetry, pairing a heavy visualand-text document tower with a lightweight 70 M text-only query tower; and (iii) a unified reproduction of twelve released retrievers from 250 M to 8.8 B under one evaluation and one profiling pipeline, reporting retrieval quality together with query latency, indexing throughput, peak VRAM, index footprint, and scoring latency. Composed under one teacher, these yield a retriever that dominates the from-scratch small multi-vector route at the 500 M scale on quality and on every deployment axis at once. The two deployment variants we release differ only in the document encoder’s visual-tile budget. DistilVDR-HiRes reaches 61.74 average NDCG@5 on ViDoRe v1+v2+v3 (Faysse et al., 2025; Macé et al., 2025; Loison et al., 2026), 86.9 % of the 8 B teacher, and leads every reproduced sub-1 B baseline on the high-resolutionsensitive v3 benchmark. At a 3× smaller visualtoken budget, DistilVDR-Fast still exceeds every reproduced sub-1 B baseline, at 59.98 average NDCG@5. Both variants store one million documents in a 15.6 times smaller index than the strongest sub-1 B multi-vector baseline and index the corpus an order of magnitude faster.

## 2 Related Work

Section 1 covers the directly competing visual document retrievers; here we situate our work against three adjacent areas.

Distillation in dense text retrieval. Hardnegative mining (Xiong et al., 2020; Qu et al., 2021), retrieval-oriented pretraining (Xiao et al., 2022), and cross-encoder-to-bi-encoder distillation (Hofstätter et al., 2021; Chen et al., 2024a) are common building blocks of modern dense retrievers (Wang et al., 2024a,b; Sturua et al., 2024). The asymmetric pattern in which a heavy reranker teaches a lightweight bi-encoder is the closest textside analogue to our setup, and differs from it in two ways. Our teacher produces a single image embedding rather than a query-document relevance score, so the distillation target is geometric rather than a scalar; and our asymmetry is across modalities rather than across model classes, with a textonly query side and an image document side.

Universal multimodal embedders. A second line fine-tunes a large vision-language model into a universal embedder over arbitrary modality pairs (Jiang et al., 2024; Wei et al., 2024). We share the idea of converting a generative vision-language model into an embedding model, but trade their breadth at multi-billion scale for depth in one subdomain at sub-1 B scale. The fixed input shape of VDR, text queries against image documents, lets both encoders stay small.

OCR-free document understanding. A third line feeds document images directly to the model. Donut (Kim et al., 2022), Pix2Struct (Lee et al., 2023) and UDOP (Tang et al., 2023) decode text outputs; LayoutLM and LayoutLMv3 (Xu et al., 2020; Huang et al., 2022) are bidirectional encoders with explicit layout tokens. They share our premise that direct image input avoids OCR error compounding, but target generation or extraction and produce representations not designed for nearestneighbour retrieval. We produce one dense vector per page that indexes with standard dense-retrieval infrastructure.

## 3 Method

## 3.1 Problem setup

Given a text query q and a corpus of document images $\{ d _ { 1 } , \hdots , d _ { N } \}$ , the system ranks the corpus by relevance to q. We work in the single-vector dense retrieval setting throughout. A query encoder $f _ { q }$ maps the query to a vector $\mathbf { q } \in \mathbb { R } ^ { k }$ and a document encoder $f _ { d }$ maps each document image to a vector $\mathbf { d } _ { i } \in \mathbb { R } ^ { k }$ in the same space. Both vectors are L2-normalised. The retrieval score is the dot product $s ( q , d _ { i } ) = \mathbf { q } ^ { \top } \mathbf { d } _ { i }$ . Document vectors are computed once at indexing time and queried with any standard dense-retrieval engine.

## 3.2 Dual-student distillation

The architecture is a two-student instance of the asymmetric distillation paradigm. A frozen 8 B vision-language teacher (Li et al., 2026a) produces target embeddings of dimension k = 4096 for both modalities. Two students learn to reproduce these targets independently. The query student $f _ { q }$ takes a text query and the document student $f _ { d }$ takes a document image. Both students project to the teacher’s output space and are L2-normalised. Retrieval at deployment uses the students only; the teacher is discarded after training. Figure 1 illustrates the architecture and the two distillation objectives.

## 3.3 Document encoder

The document encoder $f _ { d }$ has 454 M parameters (300 M InternViT visual encoder + 150 M ModernBERT-base text backbone + 4 M visual-totext projection and final 768→4096 output head) and proceeds in four stages.

First, DistilVDR enforces a fixed per-document visual-token budget. We cap the per-page tile count at $T _ { \mathrm { m a x } } ,$ , a release-time design choice that distinguishes the Fast and HiRes variants (Section 4.1), and append a single thumbnail at the visual encoder’s native resolution for global context. Given $T _ { \mathrm { m a x } }$ and the page’s aspect ratio, we choose an aspect-ratio-matched grid layout at the same native tile resolution, following the InternVL-V2 partition rule (Chen et al., 2025); Algorithm 1 in Appendix E formalises the procedure. The cap resolves the two failure modes of document encoding at once: a single low-resolution view loses small text, table cells, and figure annotations, while a single highresolution view exceeds the text backbone’s context window.

![](images/3c37998633c72da86b20beeffd7b2c0cbedbc0ca3789252678e16f7e3e223233.jpg)  
Retrieval at deployment: $s ( q , d ) = \mathbf { q } _ { s } ^ { \top } \mathbf { d } _ { s }$ (teacher discarded)  
Figure 1: Dual-student distillation architecture. The teacher encodes the image (left) and the query (right); each student is trained independently to match its teacher target.

Second, every tile is encoded by an InternViT-300M-448 visual encoder (Chen et al., 2024b) into 1 024 patch tokens of dimension 768. We select InternViT-300M for two reasons: its 448-pixel native tile resolution matches our dynamic-tiling rule without any patch resampling, and its documentrich vision-language pre-training transfers well to page-level document imagery. Patch tokens from all tiles are concatenated into one sequence of length up to 7 168.

Third, the visual sequence is mapped into the embedding space of a ModernBERT-base text backbone (Warner et al., 2025) by a learned linear projection. ModernBERT then re-encodes the projected visual tokens with bidirectional attention, with no text tokens in the input. We use the text backbone here as a contextual encoder over visual tokens. This usage follows ModernVBERT (Teiletche et al., 2025), which shows that an encoder with bidirectional attention outperforms a causal decoder on document retrieval. We select ModernBERT-base because its 8 192-token context window absorbs the worst-case 7 168-token sequence from seven tiles without truncation, and because its rotary positional embeddings and alternating local/global attention yield strong long-context representations at the 150 M parameter scale.

Fourth, the contextualised tokens are mean pooled, projected from 768 to 4096 dimensions by a final linear layer, and L2-normalised.

## 3.4 Query encoder

The query encoder $f _ { q }$ has 70 M parameters and proceeds in three stages. First, the query text is prefixed by the same instruction string π that the teacher was trained with, so that the student input matches the teacher input at distillation time. Second, the prefixed sequence is encoded by a DistilBERT-base text encoder (Sanh et al., 2019) and mean pooled over the contextual token representations. Third, the pooled vector is passed through a linear projection from 768 to 4096 dimensions and L2-normalised.

## 3.5 Distillation objective

Let T denote the frozen teacher. For a document image $d ,$ the teacher produces a target vector $T ( d ) \in \mathbb { R } ^ { 4 0 9 6 }$ . For a text query q with instruction prefix π, it produces a target $T ( \pi \circ q ) \ \in \ \mathbb { R } ^ { 4 0 9 6 }$ Both targets are L2-normalised by construction. Each student is trained independently against these cached targets under a cosine alignment loss,

$$
\begin{array} { r } { \mathcal { L } _ { d } = 1 - \langle f _ { d } ( d ) , T ( d ) \rangle , } \end{array}
$$

$$
\mathcal { L } _ { q } = 1 - \langle f _ { q } ( \pi \circ q ) , T ( \pi \circ q ) \rangle ,\tag{1}
$$

(2)

where $\langle \cdot , \cdot \rangle$ denotes the dot product between L2- normalised vectors. The two students never share a forward pass during training, so the doc-side and query-side distillations are fully decoupled.

No contrastive term, hard negative, or relevance label enters the student objective. The supervision is not label-free in an absolute sense, since the teacher’s embedding space was itself trained with relevance supervision. The objective removes relevance signals and negative sampling at student training time, which is what keeps the two distillations decoupled and independently parallelisable. Implementation details, datasets, and hyperparameters are deferred to Section 4.

## 4 Experiments

## 4.1 Experimental setup

Benchmarks and metric. We evaluate on the full 22-dataset ViDoRe suite: v1 (Faysse et al., 2025) (10 datasets, English), v2 (Macé et al., 2025) (4, multilingual) and v3 (Loison et al., 2026) (8, professional domains), each dataset listed in Appendix A. We report NDCG@5, computed with pytrec\_eval and averaged within each benchmark.

Training data and recipe. The document mixture is 1.20 M unique images from public, permissively licensed sources, assembled in three parts: a 711 K base mixture, a 454 K multi-domain supplement, and a 31.7 K finance supplement, spanning scientific publications, multilingual web PDFs, industrial and regulatory reports, and financial filings. All three are deduplicated by perceptual hash against one another and against all three ViDoRe evaluation corpora (Appendix B). The query mixture is the 1.49 M-query NanoVDR training set (Liu et al., 2026). Both students are trained with AdamW under a one-cycle schedule against cached 4096-d teacher targets. Appendices B and D give the per-source breakdown and the full hyperparameters. The document encoder is trained in two variants differing only in the tile cap of Section 3.3: DistilVDR-HiRes (six 448×448 tiles plus thumbnail; 7 168-token worst case) and DistilVDR-Fast (two tiles plus thumbnail; 3 072-token worst case). Data, optimizer, and epoch budget are identical across the two variants and the query encoder.

Teacher precomputation cost. The teacher runs exactly once, before any student training. Caching targets for the 1.20 M images and 1.49 M queries cost 99.5 H200-GPU-hours across three sharded batches, one per part of the mixture: 40.5 for the 711 K base mixture together with its queries, their translations, and the validation split; 58.1 for the multi-domain supplement, as encoded before deduplication reduced it to 454 K images; and 0.9 for the finance supplement, at a measured 11.3 images/sec. Every result in this paper, including all retrained ablations, reuses these cached targets, and this is the only point at which the 8 B model is executed.

Baselines and protocol. We reproduce twelve publicly released retrievers spanning single-vector and multi-vector designs from 250 M to 8.8 B parameters, plus the 8 B teacher as a single-vector oracle; Appendix C names each one. Every system is loaded from its official release and run inside one evaluation driver, so input formatting, deduplication, and metric computation are identical across systems. Multi-vector baselines use their authors’ recommended late-interaction routine; single-vector baselines use brute-force dot product over the full corpus.

## 4.2 Retrieval quality

Table 1 reports NDCG@5 on the full ViDoRe suite for every reproduced baseline alongside the two DistilVDR variants. Against the strongest reproduced sub-1 B baseline, colSmol-500M at 53.01 average, HiRes leads by 8.73 points and Fast by 6.97. Both variants exceed every other sub-1 B baseline by more.

The two variants separate on the hardest benchmark. On v3, built on long professional reports with dense small text and complex layouts, HiRes scores 47.07 against Fast’s 43.66, while on v1 and v2 they stay within 1.5 points of each other. HiRes leads the next-best sub-1 B retriever on v3 by 13.55 points (47.07 vs. 33.52); Fast trades that v3 quality for a 3× smaller visual-token budget, which Section 4.3 turns into deployment cost.

Against larger references the variants are competitive at 2–3 B and behind at 4–8 B. HiRes exceeds DSE-Qwen2 (2.2 B) by 1.03 and ColPali v1.3 (2.9 B) by 1.42 average points despite being four to six times smaller, and Fast trails both by under one point. Both trail Qwen3-VL-Embedding-2B, and the 4–8 B multi-vector models remain 6.95 to 9.80 points ahead of HiRes, with the gap concentrated on v2 and v3. The 8 B teacher exceeds HiRes by 9.31 points; HiRes and Fast retain 86.9 % and 84.4 % of its average NDCG@5.

Gap decomposition. To localise the residual gap between DistilVDR-HiRes and the 8 B teacher we evaluate the four combinations of student and teacher on each side. Table 2 reports the resulting NDCG@5. Replacing the document side with the student costs 6.03 NDCG@5 on average (T×T vs. T×S). Replacing the query side with the student costs 4.69 points on average (T×T vs. S×T). The two side losses do not simply add to the full gap of 9.31 points; the residual is the interaction between

the two student errors.

Where the residual gap lives. One 4096-d vector per page already expresses retrieval quality well beyond what our student reaches. The teacher is itself a single-vector retriever, and at 71.05 average NDCG@5 it is level with the 4.4 B multi-vector Tomoro-ColQwen3-4B (71.01) and above the 7 B ColNomic-7B (68.69). DistilVDR is therefore limited by capacity within the single-vector format rather than by the format itself. The document side accounts for 6.03 of the 9.31-point gap (Table 2), and shrinking the output vector from 4096 to 768 dimensions alone costs 4.77 NDCG@5 on v3 (Table 4, block (c)).

## 4.3 Efficiency analysis

Table 3 reports throughput and resource usage for the twelve baselines and the distillation teacher of Table 1. Every encoder runs at fixed batch size B = 8 in bf16 under its best-supported flashattention backend (Dao et al., 2022; Dao, 2024) on a single H200 GPU; per-baseline implementation details are deferred to Appendix G. The four 250–500 M baselines (BiModernVBERT, colSmol-256M, colSmol-500M, ColModernVBERT) all use the uncapped Idefics3 image splitter (Laurençon et al., 2024), which emits more than ten sub-images per document page; DistilVDR caps its budget at seven (Section 3.3), so its image tower runs over a shorter visual sequence at comparable parameter scales.

Encoding and indexing. DistilVDR-Fast attains the highest document throughput at 99.04 docs/sec and 2.10 GB peak VRAM, an order of magnitude faster than every multi-vector baseline at any scale, several times faster than every other sub-1 B singlevector baseline, and roughly 18× faster than the 8 B teacher. HiRes returns part of that throughput for its larger tile budget and still runs 7× faster than the teacher. SigLIP2-L is the only sub-1 B baseline with lower peak VRAM than Fast. Both variants encode queries in 3.4 ms through the text-only DistilBERT path, faster than every other profiled system.

Index storage and scoring. The multi-vector index footprint is the per-document token count times the embedding dimension times the storage precision. Sub-1 B multi-vector baselines store roughly 1 000 128-d float16 vectors per page (256 GB per million documents) and the 4–8 B Tomoro baselines 1 280 320-d tokens (819 GB). DistilVDR stores one 4096-d float32 vector, 16.4 GB, and scores 10 000 documents in 9.6 ms against the 1.1– 3.2 s the multi-vector baselines need for the same MaxSim. An end-to-end multi-vector deployment at sub-1 B scale therefore costs roughly 16× in storage and 100× in scoring latency relative to DistilVDR.

Table 1: Retrieval quality (NDCG@5) on ViDoRe v1, v2 and v3. “Type” is single vector or multi-vector with MaxSim.
<table><tr><td>Model</td><td>Params</td><td>Type</td><td>v1</td><td>v2</td><td>v3</td><td> $\operatorname { A v g }$ </td></tr><tr><td colspan="7">Sub-1 B</td></tr><tr><td>SigLIP2-L</td><td>880 M</td><td>single</td><td>43.58</td><td>20.17</td><td>14.04</td><td>25.93</td></tr><tr><td>BiModernVBERT</td><td>250 M</td><td>single</td><td>37.40</td><td>10.88</td><td>5.52</td><td>17.93</td></tr><tr><td>colSmol-256M</td><td>256M</td><td>multi</td><td>79.72</td><td>34.63</td><td>25.23</td><td>46.53</td></tr><tr><td>colSmol-500M</td><td>478M</td><td>multi</td><td>82.42</td><td>43.09</td><td>33.52</td><td>53.01</td></tr><tr><td>ColModernVBERT</td><td>250 M</td><td>multi</td><td>76.76</td><td>33.18</td><td>17.45</td><td>42.46</td></tr><tr><td>SauerkrautLM-ColLFM2</td><td>451M</td><td>multi</td><td>78.24</td><td>45.09</td><td>33.19</td><td>52.17</td></tr><tr><td>DistilVDR-Fast (ours)</td><td>524M</td><td>single</td><td>81.34</td><td>54.95</td><td>43.66</td><td>59.98</td></tr><tr><td>DistilVDR-HiRes (ours)</td><td>524M</td><td>single</td><td>82.81</td><td>55.34</td><td>47.07</td><td>61.74</td></tr><tr><td colspan="7">Mid- to large-scale references</td></tr><tr><td>DSE-Qwen2</td><td>2.2 B</td><td>single</td><td>85.14</td><td>55.70</td><td>41.28</td><td>60.71</td></tr><tr><td>Qwen3-VL-Embedding-2B</td><td>2.1 B</td><td>single</td><td>84.30</td><td>65.25</td><td>49.98</td><td>66.51</td></tr><tr><td>ColPali v1.3</td><td>2.9B</td><td>multi</td><td>84.21</td><td>54.72</td><td>42.04</td><td>60.32</td></tr><tr><td>Tomoro-ColQwen3-4B</td><td>4.4B</td><td>multi</td><td>90.22</td><td>65.25</td><td>57.57</td><td>71.01</td></tr><tr><td>ColNomic-7B</td><td>7.0B</td><td>multi</td><td>89.76</td><td>60.44</td><td>55.87</td><td>68.69</td></tr><tr><td>Tomoro-ColQwen3-8B</td><td>8.8 B</td><td>multi</td><td>90.61</td><td>65.00</td><td>59.00</td><td>71.54</td></tr><tr><td>Qwen3-VL-Embedding-8B (teacher)</td><td>8.1 B</td><td>single</td><td>87.31</td><td>69.76</td><td>56.07</td><td>71.05</td></tr></table>

Table 2: Gap decomposition (NDCG@5) for DistilVDR-HiRes. T = teacher, S = student. Bottom row is the full system.
<table><tr><td>Query × Doc</td><td>v1</td><td>v2</td><td>v3</td><td>Avg</td></tr><tr><td> $\mathrm { T } \times \mathrm { T }$  (oracle)</td><td>87.31</td><td>69.76</td><td>56.07</td><td>71.05</td></tr><tr><td> $\mathrm { ~ T ~ } \times \mathrm { ~ S ~ }$ </td><td>83.72</td><td>60.92</td><td>50.43</td><td>65.02</td></tr><tr><td> $\mathbf { S } \times \mathbf { T }$ </td><td>84.68</td><td>64.30</td><td>50.09</td><td>66.36</td></tr><tr><td> $\mathbf { S } \times \mathbf { S }$  (ours)</td><td>82.81</td><td>55.34</td><td>47.07</td><td>61.74</td></tr></table>

## 5 Ablations

Table 4 collects four ablations: the visual tile budget, the training-data scale, and the projection-head output dimension on the document side, plus the query-encoder backbone. Every variant is retrained from scratch under the recipe of Appendix D, changing one factor at a time. Blocks (a) and (b) are scored end to end, student query against student document. Blocks (c) and (d) are scored under single-side isolation, pairing the retrained tower with the teacher’s other tower so that the factor under study is not confounded by the other tower’s error; their absolute values are therefore higher than those of the end-to-end system of Section 4.2 and are comparable only within a block. A fifth ablation (Section 5.1) asks whether contrastive supervision on top of cosine alignment helps.

Visual tile budget. Block (a) sweeps the tile cap of Section 3.3; the Fast and HiRes rows are the two deployed variants and the 0-tile row is a single-448×448-view control. Removing tiling costs 5.12 average NDCG@5 relative to HiRes, concentrated on v3. Going from Fast to HiRes adds 1.76 average points and 3.41 on v3 at 3× the visual tokens, whose deployment cost Table 3 reports.

Training-data scale. Block (b) retrains the document encoder on uniform subsamples of the 1.20 Mimage mixture. Quality grows monotonically with data and saturates above 75 %; the full mixture is worth about 5 average points over a quarter-scale variant.

Output dimension. Because the teacher is trained with Matryoshka representation learning (Li et al., 2026a), any leading prefix of its 4096-d output is itself a valid embedding, so truncating the target to 768 dimensions is a principled compression rather than a random projection. Retraining the document encoder against that truncated target (block (c)) shrinks the index 5.3× (16.4 GB → 3.07 GB per million documents) but costs 3.19 average NDCG@5 and 4.77 on v3. We keep 4096-d as the default and treat 768-d as the option under hard storage constraints.

Query backbone. Block (d) retrains the query encoder on BERT-base (110 M) and

Table 3: Deployment efficiency on a single H200 GPU at fixed batch size B = 8 in bf16. Each encoder runs under its best-supported flash-attention backend (Dao et al., 2022; Dao, 2024); per-baseline details are in Appendix G. Index size is per million documents. Avg NDCG@5 from Table 1 is repeated for reference. In each efficiency column we bold the column-best and underline the second-best.
<table><tr><td>Model</td><td>Params</td><td>Type</td><td>Query (ms)</td><td>(docs/s)</td><td>Doc thpt Peak VRAM (GB)</td><td>Index /1M</td><td>Score 10 K (ms)</td><td>Avg NDCG@5</td></tr><tr><td>Sub-1 B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SigLIP2-L</td><td>880M single</td><td></td><td>113.8</td><td>28.22</td><td>1.98</td><td>4.1 GB</td><td>1.3</td><td>25.93</td></tr><tr><td>BiModernVBERT</td><td>250M single</td><td></td><td>7.4</td><td>2.49</td><td>7.47</td><td>3.1 GB</td><td>0.9</td><td>17.93</td></tr><tr><td>colSmol-256M</td><td>256M multi</td><td></td><td>23.6</td><td>2.58</td><td>4.49</td><td>256 GB</td><td>1259</td><td>46.53</td></tr><tr><td>colSmol-500M</td><td>478M</td><td>multi</td><td>22.6</td><td>2.82</td><td>4.97</td><td>256 GB</td><td>1161</td><td>53.01</td></tr><tr><td>ColModernVBERT</td><td>250 M multi</td><td></td><td>61.8</td><td>3.06</td><td>4.26</td><td>256 GB</td><td>1238</td><td>42.46</td></tr><tr><td>SauerkrautLM-ColLFM2</td><td>451 M multi</td><td></td><td>8.5</td><td>19.02</td><td>2.56</td><td>256 GB</td><td>1330</td><td>52.17</td></tr><tr><td>DistilVDR-Fast (ours)</td><td>524M</td><td>single</td><td>3.4</td><td>99.04</td><td>2.10</td><td>16.4 GB</td><td>9.6</td><td>59.98</td></tr><tr><td>DistilVDR-HiRes (ours)</td><td>524M single</td><td></td><td>3.4</td><td>36.82</td><td>3.07</td><td>16.4 GB</td><td>9.6</td><td>61.74</td></tr><tr><td>Mid- to large-scale references</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DSE-Qwen2</td><td>2.2 B single</td><td></td><td>167.4</td><td>17.17</td><td>6.31</td><td>6.1 GB</td><td>2.2</td><td>60.71</td></tr><tr><td>Qwen3-VL-Embedding-2B</td><td>2.1 B</td><td>single</td><td>14.5</td><td>8.53</td><td>7.03</td><td>8.2 GB</td><td>3.4</td><td>66.51</td></tr><tr><td>ColPali v1.3</td><td>2.9B</td><td>multi</td><td>266.7</td><td>17.24</td><td>7.81</td><td>264 GB</td><td>1158</td><td>60.32</td></tr><tr><td>Tomoro-ColQwen3-4B</td><td>4.4 B</td><td>multi</td><td>266.4</td><td>11.91</td><td>12.93</td><td>819 GB</td><td>3187</td><td>71.01</td></tr><tr><td>ColNomic-7B</td><td>7.0B</td><td>multi</td><td>542.6</td><td>9.96</td><td>17.88</td><td>256 GB</td><td>1206</td><td>68.69</td></tr><tr><td>Tomoro-ColQwen3-8B</td><td>8.8 B multi</td><td></td><td>499.3</td><td>9.20</td><td>21.76</td><td>819GB</td><td>3176</td><td>71.54</td></tr><tr><td>Qwen3-VL-Embedding-8B (teacher)</td><td>8.1 B single</td><td></td><td>19.8</td><td>5.40</td><td>19.28</td><td>16.4 GB</td><td>9.4</td><td>71.05</td></tr></table>

ModernBERT-base (149 M) under the same protocol. ModernBERT-base is the strongest at 67.40 average, 1.04 points above DistilBERT-base, but costs 2.2× the parameters and 5.1× the query latency. Encoding one query at B = 1 including tokenisation takes 2.07, 4.78 and 10.61 ms for DistilBERT-base, BERT-base and ModernBERTbase respectively. BERT-base trails both on quality at 61.63. We adopt DistilBERT-base because 1.04 points is a small price for those savings. Index size is unaffected by this choice.

## 5.1 Contrastive supervision

Our recipe trains the two students independently, with no cross-modal interaction in the loss. This ablation adds a joint refinement step that couples them through a contrastive term over paired $( q , d )$ minibatches. Starting from a cosine-distilled checkpoint of the same 524 M architecture at a 768-d projection head (the 768-d counterpart of the 4096- d encoder, with the same encoders, data, tiling, and teacher), we train both students jointly for one further epoch on the same mixture under

$$
\mathcal { L } = \alpha \mathcal { L } _ { d } + \beta \mathcal { L } _ { q } + \gamma \mathcal { L } _ { \mathrm { r a n k } } ,\tag{3}
$$

where $\mathcal { L } _ { d }$ and $\mathcal { L } _ { q }$ are the cosine alignment losses of Eqs. 1–2 and $\alpha = \beta = 1 . 0$ . For a mini-batch B of paired query-document examples, let $s _ { q d }$ $\langle f _ { q } ( q ) , f _ { d } ( d ) \rangle$ denote the student similarity and $\tilde { s } _ { q d } = \langle T ( q ) , T ( d ) \rangle$ ⟩ the teacher similarity between

Table 4: Document- and query-encoder ablations, NDCG@5. Blocks (a) and (b) are end-to-end student×student; block (c) is doc-side isolation (student documents against teacher queries); block (d) is query-side isolation (student queries against teacher documents), whose teacher×teacher ceiling is 71.05 average. Absolute values are therefore comparable within a block, not across blocks. † marks the deployed default.
<table><tr><td>Variant v1 v2 v3</td></tr><tr><td>Avg (a) Max tiles (end-to-end) 0 (no tiling) 77.57 51.29 41.00 56.62</td></tr><tr><td>2 (Fast) 81.34 54.95 43.66 59.98 6 (HiRes, default) 82.81 55.34 47.07 61.74 (b) Training-data scale (end-to-end) 25 % (300 K) 78.35 49.69 40.75 56.26</td></tr><tr><td>50 % (600 K) 80.97 54.44 44.58 60.00 75 % (900 K) 82.40 56.58 46.05 61.68 100 % (1.20 M, default) 82.81 55.34 47.07 61.74 (c) Doc output dim (doc-side isolation) 81.4058.44</td></tr><tr><td>768-d (3.07 GB/1 M) 45.66 61.83 4096-d (16.4 GB/1 M)† 83.72 60.92 50.43 65.02 (d) Query backbone (query-side isolation)</td></tr><tr><td>DistilBERT-base (70 M)† 84.6864.30 50.09 66.36 BERT-base (110 M) 81.81 58.52 44.55 61.63 ModernBERT-base (149 M) 85.36 65.40 51.43 67.40</td></tr></table>

q and d. We consider two choices for $\mathcal { L } _ { \mathrm { r a n k } } \mathrm { . }$

$$
\mathcal { L } _ { \mathrm { I n f o N C E } } = - \sum _ { ( q , d ) \in \mathcal { B } } \log \frac { \exp ( s _ { q d } / \tau ) } { \sum _ { d ^ { \prime } \in \mathcal { B } } \exp ( s _ { q d ^ { \prime } } / \tau ) } ,\tag{4}
$$

$$
\mathcal { L } _ { \mathrm { K L } } = \sum _ { q \in \mathcal { B } } \mathrm { K L } \big ( \sigma _ { \tau } ( \tilde { s } _ { q } ) \big | \big | \sigma _ { \tau } ( s _ { q } ) \big ) ,\tag{5}
$$

with temperature $\tau = 0 . 0 2$ and σ the row-wise softmax of similarities to all documents in B. The refinement uses AdamW, effective batch 256, peak learning rate $2 \times 1 0 ^ { - 5 }$ , warm-up fraction 0.05, and the dynamic-tiling rule of DistilVDR-HiRes on two H200 GPUs. Absolute NDCG@5 in Table 5 is below Section 4.2 because the projection head outputs 768 dimensions instead of 4096; the relative effect of γ is unaffected by this offset.

Within this 1-epoch joint refinement, adding contrastive supervision yields no NDCG@5 improvement at any γ we tried. InfoNCE shows a mild downward drift with increasing γ, from −0.12 at $\gamma = 0 . 5 ~ { \mathrm { t o } } \mathrm { ~ - } 0 . 5 0 ~ { \mathrm { a t } } \gamma = 2 . 0$ . KL is flat across all weights (±0.14). The asymmetry is consistent with the two losses’ geometric content: InfoNCE pushes the student toward a uniform separator over in-batch negatives, orthogonal to the teacher-calibrated geometry already enforced by $\mathcal { L } _ { d } + \mathcal { L } _ { q } ;$ KL anchors to the teacher distribution and is approximately redundant with cosine alignment. The refinement budget is roughly one third of the document encoder’s training budget (Table 8), enough that any positive contribution from the contrastive term should register a positive drift; what we observe is flat or mildly negative. The direction is consistent with the from-scratch loss ablation of NanoVDR (Liu et al., 2026), which reports pointwise cosine alignment strictly outperforming both KL-ranking and hard-label InfoNCE distillation under the same teacher family.

Every configuration here starts from the same distilled checkpoint, so the experiment covers the refinement stage only and does not compare distillation and contrastive learning as alternative ways of training the same architecture. Within that scope it supports the pure cosine objective of Section 3.5 as a strong default.

## 6 Conclusion

We presented DistilVDR, a 524 M end-to-end visual document retriever obtained by independent cosine distillation of both encoders from a single 8 B vision-language teacher under an asymmetric encoder-only design. Its two deployment variants share encoders and training and differ only in the document-side visual-tile budget: HiRes retains 86.9 % of the teacher and leads every reproduced sub-1 B baseline on ViDoRe v3, Fast retains 84.4 % at a 3× smaller visual-token budget, and both index an order of magnitude faster at a 15.6 times smaller footprint. Measured end to end under one evaluation and one profiling pipeline, distilling both towers of a strong single-vector vision-language teacher reaches a better quality-and-cost operating point at the 500 M scale than any released small multi-vector retriever we could reproduce.

Table 5: Contrastive-supervision ablation under Eq. 3. All rows are one-epoch joint refinements from the cosine-distilled 768-d sibling checkpoint; absolute NDCG@5 is below Section 4.2 because of the 768- d output.
<table><tr><td>Refinement objective</td><td>v1</td><td>v2</td><td>v3</td><td>Avg</td></tr><tr><td>γ = 0 (cosine only)</td><td>79.90</td><td>53.23</td><td>42.22</td><td>58.45</td></tr><tr><td> $+ 0 . 5 \mathcal { L } _ { \mathrm { I n f o N C E } }$ </td><td>80.47</td><td>52.08</td><td>42.44</td><td>58.33</td></tr><tr><td> $+ 1 . 0 \mathcal { L } _ { \mathrm { I n f o N C E } }$ </td><td>80.48</td><td>51.80</td><td>42.41</td><td>58.23</td></tr><tr><td> $+ 2 . 0 \mathcal { L } _ { \mathrm { I n f o N C E } }$ </td><td>80.25</td><td>51.31</td><td>42.30</td><td>57.95</td></tr><tr><td> $+ 0 . 5 \mathcal { L } _ { \mathrm { K L } }$ </td><td>80.32</td><td>52.31</td><td>42.30</td><td>58.31</td></tr><tr><td> $+ 1 . 0 \mathcal { L } _ { \mathrm { K L } }$ </td><td>80.20</td><td>52.55</td><td>42.34</td><td>58.36</td></tr><tr><td> $+ 2 . 0 \mathcal { L } _ { \mathrm { K L } }$ </td><td>80.31</td><td>52.54</td><td>42.44</td><td>58.43</td></tr></table>

## Limitations

From an experimental perspective. Every comparison we report is between deployable systems rather than between training recipes. Every baseline in Table 1 is an official public release evaluated as released; none is retrained under a budget matched to ours, and our teacher, our 1.20 M-image mixture, and our architecture all differ from theirs, so the margin cannot be attributed to any single one of them. The control that would isolate distillation itself, the same 524 M architecture trained contrastively from scratch on the same mixture with hard-negative mining and a comparable compute budget, is the principal experiment we did not run; the refinement experiment of Section 5.1 starts from an already-distilled checkpoint and does not substitute for it. We likewise use exactly one teacher and do not sweep teacher scale or family, so we cannot say how much quality a weaker teacher would cost or how much a stronger one would add; because targets are cached once and student training never runs the teacher, that sweep is a cheap follow-up. Both variants also stay 7–10 points behind the strongest 4–8 B multi-vector references of Table 1, with the largest gap on ViDoRe v3 (47.07 and 43.66 against 57.57–59.00), and because our experiments vary the scoring mechanism only together with model scale, they cannot separate late interaction from capacity at the top of that table. Finally, every retrieval number comes from the 22

ViDoRe datasets. ViDoRe spans three difficulty levels, six languages, and eight professional domains, but not enterprise corpora with in-house layouts, scanned or OCR-degraded pages, or production query distributions; our deployment costs are measured directly and transfer, our quality numbers may not.

From a methodological perspective. The student is bounded by the space it is trained into. Both towers only reproduce a frozen teacher’s embed dings, so any systematic weakness of Qwen3-VL Embedding-8B propagates to DistilVDR, and noth ing in the objective lets a student exceed its teacher on the side it replaces; Table 2 is consistent with this, as no student-side substitution improves on the teacher×teacher oracle. Distilling from a richer signal, such as a cross-encoder relevance score or a multi-vector teacher, is what would raise that ceiling. Three further choices are deliberate sim plifications. The two students are trained indepen dently against cached targets, which parallelises cleanly but leaves unused the cross-modal coupling between query and document manifolds that the teacher implicitly carries; a joint scheme with explicit query-document interaction during distillation is the natural extension. The tiling rule adapts the grid to page aspect ratio but not to page content, and $T _ { \mathrm { m a x } }$ is fixed at release time, so a dense small-text page and a single-figure page receive the same visual-token budget; our two variants sam ple two points on that trade-off rather than resolve it, and a content-adaptive budget predicted from a cheap page-complexity estimate would spend tokens where they are needed. And the query tower is text-only, trained on English plus five Latin-script European languages obtained through a MarianMT translation pipeline, so image-conditioned queries and non-Latin scripts such as Chinese, Japanese, and Arabic are out of scope. One deployment side choice is also unexplored. We store one uncompressed 4096-d float32 vector per document. Table 4 measures a single compression axis, the 768-d Matryoshka prefix (3.07 GB per million doc uments, −3.19 NDCG@5 average); scalar quantisation, product quantisation, and binary hashing are orthogonal to our contribution and untested here, so the reported footprint should be read as an upper bound.

## References

Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024a. M3- embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through selfknowledge distillation. In Findings of the Association for Computational Linguistics: ACL 2024, pages 2318–2335, Bangkok, Thailand. Association for Computational Linguistics.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yiming Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, and 23 others. 2025. Expanding performance boundaries of opensource multimodal models with model, data, and test-time scaling. Preprint, arXiv:2412.05271.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. 2024b. InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198.

Tri Dao. 2024. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations, volume 2024, pages 35549–35562.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. Advances in neural information processing systems, 35:16344–16359.

Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2025. ColPali: Efficient document retrieval with vision language models. In International Conference on Learning Representations, volume 2025, pages 61424–61449.

Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, and Allan Hanbury. 2021. Efficiently teaching an effective dense retriever with balanced topic aware sampling. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’21, page 113–122, New York, NY, USA. Association for Computing Machinery.

Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. 2022. LayoutLMv3: Pre-training for document AI with unified text and image masking. In Proceedings ofthe 30th ACM International Conference on Multimedia, MM ’22, page 4083–4091, New York, NY, USA. Association for Computing Machinery.

Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen. 2024. VLM2Vec: Training vision-language models for massive

multimodal embedding tasks. arXiv preprint arXiv:2410.05160.

Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’20, page 39–48, New York, NY, USA. Association for Computing Machinery.

Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. 2022. OCR-free document understanding transformer. In Computer Vision – ECCV 2022, pages 498–517, Cham. Springer Nature Switzerland.

Juyeon Kim, Geon Lee, Dongwon Choi, Taeuk Kim, and Kijung Shin. 2025. Hybrid-vector retrieval for visually rich documents: Combining singlevector efficiency and multi-vector accuracy. Preprint, arXiv:2510.22215.

Hugo Laurençon, Andrés Marafioti, Victor Sanh, and Léo Tronchon. 2024. Building and better understanding vision-language models: insights and future directions. arXiv preprint arXiv:2408.12637.

Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu, Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. 2023. Pix2Struct: Screenshot parsing as pretraining for visual language understanding. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 18893–18912. PMLR.

Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, et al. 2026a. Qwen3- VL-Embedding and Qwen3-VL-Reranker: A unified framework for state-of-the-art multimodal retrieval and ranking. arXiv preprint arXiv:2601.04720.

Weien Li, Rui Song, Zeyu Li, Haochen Liu, Gonghao Zhang, Difan Jiao, Zhenwei Tang, Bowei He, Haolun Wu, Xue Liu, and Ye Yuan. 2026b. MINER: Mining multimodal internal representation for efficient retrieval. Preprint, arXiv:2605.06460.

Zhuchenyang Liu, Yao Zhang, and Yu Xiao. 2026. NanoVDR: Distilling a 2B vision-language retriever into a 70M text-only encoder for visual document retrieval. arXiv preprint arXiv:2603.12824.

António Loison, Quentin Macé, Antoine Edy, Victor Xing, Tom Balough, Gabriel Moreira, Bo Liu, Manuel Faysse, Céline Hudelot, and Gautier Viaud. 2026. ViDoRe V3: A comprehensive evaluation of retrieval augmented generation in complex real-world scenarios. arXiv preprint arXiv:2601.08620.

Xueguang Ma, Sheng-Chieh Lin, Minghan Li, Wenhu Chen, and Jimmy Lin. 2024. Unifying multimodal retrieval via document screenshot embedding. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 6492– 6505.

Yubo Ma, Jinsong Li, Yuhang Zang, Xiaobao Wu, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Haodong Duan, Jiaqi Wang, Yixin Cao, and Aixin Sun. 2025. Towards storage-efficient visual document retrieval: An empirical study on reducing patch-level embeddings. Preprint, arXiv:2506.04997.

Quentin Macé, António Loison, and Manuel Faysse. 2025. ViDoRe benchmark V2: Raising the bar for visual retrieval. arXiv preprint arXiv:2505.17166.

Nomic AI. 2025. ColNomic embed multimodal 7B: Open multimodal embeddings for documents, charts, and PDFs. https://huggingface.co/nomic-ai/ colnomic-embed-multimodal-7b. HuggingFace model card + Nomic blog https://www.nomic.ai/ news/nomic-embed-multimodal.

Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. 2021. RocketQA: An optimized training approach to dense passage retrieval for opendomain question answering. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5835–5847, Online. Association for Computational Linguistics.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.

Keshav Santhanam, Omar Khattab, Christopher Potts, and Matei Zaharia. 2022a. PLAID: An efficient engine for late interaction retrieval. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, CIKM ’22, page 1747–1756, New York, NY, USA. Association for Computing Machinery.

Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022b. Col-BERTv2: Effective and efficient retrieval via lightweight late interaction. In Proceedings of the 2022 Conference ofthe North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 3715–3734, Seattle, United States. Association for Computational Linguistics.

Saba Sturua, Isabelle Mohr, Mohammad Kalim Akram, Michael Günther, Bo Wang, Markus Krimmel, Feng Wang, Georgios Mastrapas, Andreas Koukounas, Nan Wang, and Han Xiao. 2024. jina-embeddingsv3: Multilingual embeddings with task LoRA. Preprint, arXiv:2409.10173.

Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, and Mohit Bansal. 2023. Unifying vision, text, and layout for universal document processing. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19254–19264.

Paul Teiletche, Quentin Macé, Max Conti, Antonio Loison, Gautier Viaud, Pierre Colombo, and Manuel Faysse. 2025. ModernVBERT: Towards smaller visual document retrievers. arXiv preprint arXiv:2510.01149.

TomoroAI. 2025. Tomoro-ColQwen3-embed-8B: Multi-vector visual retriever on Qwen3-VL-8B-Instruct. https://huggingface.co/TomoroAI/ tomoro-colqwen3-embed-8b. HuggingFace model card; merges Qwen3-VL-8B-Instruct with Qwen3- Embedding-8B.

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. 2025. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. Preprint, arXiv:2502.14786.

VAGO Solutions. 2025. SauerkrautLM-ColLFM2-450M-v0.1: Curriculum-trained multi-vector visual retriever on LFM2-VL. https://huggingface.co/VAGOsolutions/ SauerkrautLM-ColLFM2-450M-v0.1. Hugging-Face model card; no dedicated arXiv paper as of May 2026.

ViDoRe Team. 2024. ColSmol-500M / ColSmol-256M: ColPali-style SmolVLM visual document retrievers. https://huggingface.co/vidore/ colSmol-500M. HuggingFace model card. Technique follows ColPali (Faysse et al., 2025); backbone is SmolVLM/Idefics3.

Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2024a. Text embeddings by weakly-supervised contrastive pre-training. Preprint, arXiv:2212.03533.

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024b. Multilingual E5 text embeddings: A technical report. Preprint, arXiv:2402.05672.

Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, et al. 2025. Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2526–2547.

Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter, and Wenhu Chen. 2024. UniIR: Training and benchmarking universal multimodal information retrievers. In European Conference on Computer Vision, pages 387–404. Springer.

Shitao Xiao, Zheng Liu, Yingxia Shao, and Zhao Cao. 2022. RetroMAE: Pre-training retrieval-oriented language models via masked auto-encoder. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 538–548, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate nearest neighbor negative contrastive learning for dense text retrieval. Preprint, arXiv:2007.00808.

Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. 2020. LayoutLM: Pre-training of text and layout for document image understanding. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’20, page 1192–1200, New York, NY, USA. Association for Computing Machinery.

Yibo Yan, Mingdong Ou, Yi Cao, Xin Zou, Jiahao Huo, Shuliang Liu, James Kwok, and Xuming Hu. 2026. Sculpting the vector space: Towards efficient multi-vector visual document retrieval via prune-thenmerge framework. Preprint, arXiv:2602.19549.

Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, et al. 2025. VisRAG: Vision-based retrieval-augmented generation on multi-modality documents. In International Conference on Learning Representations, volume 2025, pages 21074–21098.

## A ViDoRe Benchmark Details

The Visual Document Retrieval (ViDoRe) benchmark comprises three progressively challenging versions (Faysse et al., 2025; Macé et al., 2025; Loison et al., 2026). We evaluate on the full 22-dataset suite. Table 6 lists each dataset together with its document corpus language, query languages, and the source of its queries.

v1 (Faysse et al., 2025). Released with ColPali, v1 contains 10 English- and French-language datasets. Five are sourced from established visual QA benchmarks (DocVQA, ArXivQA, InfoVQA, TatDQA, TabFQuAD) with humanauthored queries; five use queries generated by Claude-3 Sonnet over curated document collections. State-of-the-art models now exceed 90 NDCG@5 on v1, indicating saturation.

v2 (Macé et al., 2025). Designed to address v1’s saturation, v2 introduces 4 datasets with queries in up to four languages (English, French, Spanish, German). Queries are generated through a blind contextual process: annotators receive only document metadata, which reduces extractive bias and requires cross-document reasoning. The fourth dataset (ESG Reports Human-Labeled) provides expert human annotations over the same ESG corpus.

v3 (Loison et al., 2026). The most comprehensive version, v3 provides 8 public datasets across six languages (the v2 languages plus Italian and Portuguese). Annotation combined LLMsynthesised queries with expert review. Each query carries page-level relevance rankings, boundingbox annotations, and multilingual translations. The eight domains span enterprise scenarios from finance to physics, with document corpora in English or French.

## B Training Data Composition and Preprocessing

The document encoder training mixture combines three groups of publicly released datasets. Table 7 gives the full per-source breakdown, including the query-encoder mixture.

Base sources. The first group reproduces the 711 K base mixture released by NanoVDR (Liu et al., 2026): VisRAG-Synthetic and VisRAG-InDomain (Yu et al., 2025), the ColPali training set (Faysse et al., 2025), and VDR-Multilingual (English, French, German, Spanish, Italian). The query encoder uses the same 711 K queries together with the 778 K Helsinki-NLP MarianMT translations released with NanoVDR, for a total of 1.49 M paired queries.

Multi-domain supplement. The second group is the union of fourteen domain-specific document collections released by Racineai. These collections are aggregated under the racineai/VDR\_MEGA\_2 release and individually contributed under permissive licences. We deduplicate against the 711 K base by perceptual hash and against the ViDoRe v1/v2/v3 evaluation corpora. The resulting 454 Kimage subset is used on the document side only.

Finance supplement. The third group adds 31.7 K finance-domain document images from sujet-ai/Sujet-Finance-Vision-10k and

DocReRank/FinHNQue-FinanceHardNegative Queries, again deduplicated by perceptual hash against base and evaluation corpora.

Preprocessing pipeline. All training images are loaded as PIL RGB and processed by the encoder pipeline of Section 3.3 (dynamic tiling, patch encoding, projection). Deduplication is performed offline with the imagehash library at perceptualhash distance 0, which removes only exact visual duplicates. Teacher embeddings are precomputed once over the deduplicated mixture and cached as float32 arrays on disk, so the teacher does not run during student training.

## C Baselines

At the sub-1 B scale the multi-vector retrievers are colSmol-256M and colSmol-500M (ViDoRe Team, 2024), SauerkrautLM-ColLFM2-450M (VAGO Solutions, 2025), and ColModernVBERT (Teiletche et al., 2025); the single-vector retrievers are SigLIP2-L (Tschannen et al., 2025) and BiModernVBERT (Teiletche et al., 2025). At the 2– 3 B scale we include the single-vector DSE-Qwen2 (Ma et al., 2024) and Qwen3-VL-Embedding-2B (Li et al., 2026a), and the multi-vector ColPali v1.3 (Faysse et al., 2025). At the 4–8 B scale we include Tomoro-ColQwen3-4B and Tomoro-ColQwen3-8B (TomoroAI, 2025) and ColNomic-7B (Nomic AI, 2025). Qwen3-VL-Embedding-8B (Li et al., 2026a) is our distillation teacher and is also the single-vector oracle.

## D Training Recipe

Table 8 lists the hyperparameters and hardware for both students under the cosine alignment objective of Eqs. 1–2. The two document-encoder variants (Fast and HiRes) share this recipe exactly and differ only in $T _ { \mathrm { m a x } }$ . Per-epoch loss curves for all three students are reported as a convergence diagnostic in Appendix F.

## E Dynamic Tiling Algorithm

Algorithm 1 formalises the InternVL-V2 dynamic tiling rule (Chen et al., 2025) used by the document encoder of Section 3.3. The rule picks an aspect-ratio-matched grid layout from the candidate set $\mathcal { G } ~ = ~ \{ ( p , q ) ~ : ~ p , q ~ \in ~ \mathbb { Z } _ { \geq 1 } , ~ n _ { \operatorname* { m i n } } ~ \leq$ $p \cdot q \leq n _ { \operatorname* { m a x } } \}$ , resizes the image to fit that grid at the encoder’s native tile resolution, splits the resized image into non-overlapping tiles, and optionally appends a single thumbnail at the same resolution to preserve global layout context. Tiebreaking among candidates with equal aspect distance prefers the higher-resolution grid when the original image area is large. We use $n _ { \mathrm { m i n } } = 1$ $n _ { \mathrm { m a x } } = 6 .$ , tile size $s = 4 4 8$ , and enable the thumbnail. The maximum sequence length per document is $( n _ { \mathrm { m a x } } + 1 ) \cdot s ^ { 2 } / p ^ { 2 } = 7 \cdot 1 0 2 4 = 7 1 6 8 { \mathrm { p a t c h } }$ tokens at the InternViT-300M patch size of $p = 1 4$ which fits inside ModernBERT-base’s 8192-token context window without truncation.

Table 6: All 22 ViDoRe evaluation datasets. “Doc” is the document corpus language. “Query” is the languages in which queries are released. “Source” indicates how the queries were obtained: H = human-authored, $\mathrm { ~ L ~ } =$ LLM-generated, L+H = LLM-generated with expert review. The six languages of v3 are English, French, Spanish, German, Italian, and Portuguese.
<table><tr><td>Dataset</td><td>Ver.</td><td>Document domain</td><td>Doc</td><td>Query</td><td>Source</td></tr><tr><td>DocVQA</td><td>v1</td><td>Industrial documents</td><td>EN</td><td>EN</td><td>H</td></tr><tr><td>ArXivQA</td><td>v1</td><td>Scientific papers</td><td>EN</td><td>EN</td><td>H</td></tr><tr><td>InfoVQA</td><td>v1</td><td>Infographics</td><td>EN</td><td>EN</td><td>H</td></tr><tr><td>TatDQA</td><td>v1</td><td>Financial tables</td><td>EN</td><td>EN</td><td>H</td></tr><tr><td>TabFQuAD</td><td>v1</td><td>Tables in French PDFs</td><td>FR</td><td>FR</td><td>H</td></tr><tr><td>SyntheticDocQA-AI</td><td>v1</td><td>AI documents</td><td>EN</td><td>EN</td><td>L</td></tr><tr><td>SyntheticDocQA-Energy</td><td>v1</td><td>Energy sector reports</td><td>EN</td><td>EN</td><td>L</td></tr><tr><td>SyntheticDocQA-Gov.</td><td>v1</td><td>Government reports</td><td>EN</td><td>EN</td><td>L</td></tr><tr><td>SyntheticDocQA-Hlt.</td><td>v1</td><td>Healthcare documents</td><td>EN</td><td>EN</td><td>L</td></tr><tr><td>ShiftProject</td><td>v1</td><td>Environmental reports</td><td>FR</td><td>FR</td><td>L</td></tr><tr><td>ESG Reports</td><td>v2</td><td>ESG / sustainability</td><td>EN</td><td>EN/FR/ES/DE</td><td>L+H</td></tr><tr><td>Biomedical Lectures</td><td>v2</td><td>Biomedical slides</td><td>EN</td><td>EN/FR/ES/DE</td><td>L+H</td></tr><tr><td>Economics Reports</td><td>v2</td><td>Economics reports</td><td>EN</td><td>EN/FR/ES/DE</td><td>L+H</td></tr><tr><td>ESG Reports (Human)</td><td>v2</td><td>ESG / sustainability</td><td>EN</td><td>EN</td><td>H</td></tr><tr><td>Finance-EN</td><td>v3</td><td>US annual reports</td><td>EN</td><td>6 languages</td><td>L+H</td></tr><tr><td>Finance-FR</td><td>v3</td><td>French annual reports</td><td>FR</td><td>6 languages</td><td>L+H</td></tr><tr><td>Computer Science</td><td>v3</td><td>CS textbooks</td><td>EN</td><td>6 languages</td><td>L+H</td></tr><tr><td>HR</td><td>v3</td><td>EU HR reports</td><td>EN</td><td>6 languages</td><td>L+H</td></tr><tr><td>Energy</td><td>v3</td><td>French energy reports</td><td>FR</td><td>6 languages</td><td>L+H</td></tr><tr><td>Industrial</td><td>v3</td><td>USAF technical orders</td><td>EN</td><td>6 languages</td><td>L+H</td></tr><tr><td>Pharmaceutical</td><td>v3</td><td>FDA reports</td><td>EN</td><td>6 languages</td><td>L+H</td></tr><tr><td>Physics</td><td>v3</td><td>French physics lectures</td><td>FR</td><td>6 languages</td><td>L+H</td></tr></table>

## F Training Loss Curves

Figure 2 reports per-epoch train and validation cosine-alignment loss for the two document-encoder variants (DistilVDR-HiRes and DistilVDR-Fast) and for the shared query encoder under the recipe of Table 8. All curves decrease smoothly to a low plateau with a small train/validation gap, indicating that every student reaches a steady-state fit to its teacher target within the allotted epoch budget.

## G Efficiency Benchmark Protocol

Hardware and software stack. A single NVIDIA H200 GPU (141 GB HBM3e) on an Intel Xeon Platinum 8480+ node with 32 CPU cores and

Algorithm 1 Dynamic Tiling   
Require: image I of size (W, H); n<sub>min</sub>, n<sub>max</sub>;   
tile size s; thumbnail flag t   
Ensure: list of tiles, each $s \times s$   
1: $r  W / H$ ▷ input aspect ratio   
2: $\mathcal { G }  \{ ( p , q ) : p , q \in \mathbb { Z } _ { \geq 1 } , n _ { \operatorname* { m i n } } \leq p q \leq$   
$n _ { \mathrm { m a x } } \}$   
3: $( p ^ { * } , q ^ { * } ) \gets \arg \operatorname* { m i n } _ { ( p , q ) \in \mathcal { G } } | r - p / q |$ ▷ ties   
broken toward higher resolution   
4: $ I ^ { \prime } \gets \mathrm { R e s i z e } ( I , ( p ^ { * } s , q ^ { * } s ) )$   
5: T ← split I<sup>′</sup> into $p ^ { * } q ^ { * }$ non-overlapping $s \times s$   
tiles, left-to-right then top-to-bottom   
6: if t and $p ^ { * } q ^ { * } > 1$ then   
7: ${ \mathcal { T } }  { \mathcal { T } } \cup \{ \operatorname { R e s i z e } ( I , \ ( s , s ) ) \}$ ▷   
thumbnail for global context   
8: end if   
9: return T

128 GB RAM. Software: PyTorch 2.5, CUDA 12.6, transformers $\geq 4 . 4 6 .$ . All encoders run in bfloat16.

Attention backend per encoder. Every encoder runs under a flash-attention-family kernel (Dao et al., 2022; Dao, 2024). For all baselines and DistilVDR’s ModernBERT text backbone this kernel is reached through PyTorch’s sdpa backend, which dispatches to

Table 7: Training data composition. HuggingFace identifiers are hyperlinked; the organisation prefix is omitted in the second and third groups.
<table><tr><td>Source</td><td>Samples</td></tr><tr><td colspan="2">Base mixture (711 K) (Yu</td></tr><tr><td>VisRAG-Ret-Train-Synthetic et al., 2025)</td><td>234 K</td></tr><tr><td>VisRAG-Ret-Train-In-domain (Yu</td><td>94K</td></tr><tr><td>et al., 2025) colpali_train_set (Faysse et al., 2025)</td><td>109 K</td></tr><tr><td>vdr-multilingual-train</td><td>275 K</td></tr><tr><td colspan="2">Multi-domain supplement (Racineai, deduplicated) racineai/VDR_* (14 sub-sources) 454K</td></tr><tr><td>Finance supplement (31.7 K) Sujet-Finance-Vision-10k</td><td>9.8 K</td></tr><tr><td>FinHNQue</td><td>21.9K</td></tr><tr><td>Document encoder total</td><td>1.20 M</td></tr><tr><td>Query encoder mixture NanoVDR query training set (Liu et al., 2026)</td><td>1.49 M</td></tr></table>

Table 8: Training recipe under the cosine alignment objective (Eq. 1–2).
<table><tr><td>Hyperparameter</td><td>Doc enc.</td><td>Query enc.</td></tr><tr><td>Trainable parameters</td><td>454M</td><td>70M</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Peak LR</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>LR schedule</td><td></td><td>one-cycle, 3% warmup</td></tr><tr><td>Effective batch</td><td>256</td><td>512</td></tr><tr><td>Epochs</td><td>3</td><td>15</td></tr><tr><td>Hardware</td><td> $2 \times \mathrm { H } 2 0 0$ </td><td> $2 \times \mathrm { H } 2 0 0$ </td></tr></table>

FlashAttention-2 in transformers ≥ 4.46; calling attn\_implementation="flash\_attention\_2" explicitly changes throughput by under ±1 docs/sec. DistilVDR’s InternViT visual encoder does not register an sdpa implementation with transformers, but its remote modeling code ships its own native flash-attention path, enabled by default via the use\_flash\_attn=True config flag; this is the configuration reported in Table 3.

Per-baseline deviations. Two baselines deviate from the default bf16+sdpa setup. BiModernVBERT is loaded in float32 because the canonical evaluation pipeline observes that bf16 mean-pooling over the 1 000+ visual tokens emitted by its ModernBERT backbone incurs a several-NDCG drop; we follow the released configuration and use float32 on both the document and query sides. DSE-Qwen2 follows its released chat-template inference path, wrapping each document image in a user message with a fixed

![](images/eee436d8f441f4e6297fd1bfa6adf8922ef624d4fcf24bbe6a4fabd55d5d399d.jpg)

![](images/a6ac024e02f4e717c7805544cacdf6fb907015c8e77e6d764b1b4676b51b2b00.jpg)  
Figure 2: Per-epoch train (blue) and validation (red) loss for the document students DistilVDR-HiRes (solid) and DistilVDR-Fast (dashed) on top, and the shared query student on bottom.

680 × 680 resized resolution and reading the last hidden state at the trailing <|endoftext|> token as the document embedding; we use the model’s released prepare\_inputs\_for\_generation entry point with use\_cache=False to obtain that hidden state in a single forward pass.

Conservative attention-backend reference. For transparency we also benchmark a conservative DistilVDR variant that forces InternViT to eager attention (use\_flash\_attn=False). At B = 8, DistilVDR-Fast drops to 31.91 docs/sec (VRAM 5.73 GB) and DistilVDR-HiRes to 9.27 docs/sec (11.56 GB); even under this configuration DistilVDR-Fast remains faster than every sub-1 B multi-vector baseline.

Measurement protocol. Query latency is the mean over 20 queries at B = 1 after 3 warmup queries, with cuda.synchronize() bracketing the forward call and tokenisation included. Document throughput and peak VRAM are measured at fixed batch B = 8 on a 100-page synthetic corpus after 3 warmup forward passes; multi-vector baselines run their official per-token pooling and projection so that the per-document tensor written to the index is fully formed. Index size is computed analytically from each baseline’s released storage format (T tokens per document at dimension d in float16 for multi-vector retrievers; one d-dimensional float32 vector per document for single-vector retrievers).

CPU scoring latency is the mean over 20 runs on a single thread (torch.set\_num\_threads(1)) against a synthetic 10 000-candidate corpus in the appropriate storage format; the single-vector path is a dot product, the multi-vector path is the MaxSim routine over 32 query tokens.