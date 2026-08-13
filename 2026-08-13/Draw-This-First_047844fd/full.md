# Draw This First

Dazhi Zhong   
Krea.ai   
Wand Technologies   
elea@krea.ai

Rowan Bradbury Bradbury Group Wand Technologies rowan@bradburygroup.org

Grant Davis   
Krea.ai   
Wand Technologies   
grant@krea.ai

## Abstract

We invert the typical formulation of sketch generation: instead of drawing strokes in order, we predict a 2D field that defines the order in which strokes are drawn. We use a pretrained latent flow-matching transformer to supply the image prior to predict an intermediate representation, while training the VAE’s decoder to predict the order field, stroke mask, and stroke segmentation. We vectorize the predicted segmentation into polylines and sort them by the field, producing an ordered vector sketch. Our model can predict an ordered vector sketch from a text description or derender an image into ordered vectors; for either, it follows text instructions specifying the order of drawing.

## 1 Introduction

Sketch generation is a diverse field featuring a variety of techniques and output representations. Points or latents can be predicted autoregressively, commonly through a transformer or RNN [Tang et al., 2024, Wang et al., 2025, Aksan et al., 2020, Ha and Eck, 2018]. Ordered or unordered coordinates and curves can be predicted through diffusion [Wang et al., 2023, Zhou et al., 2026, Arar et al., 2025, Das et al., 2023] or optimization [Vinker et al., 2022, Jain et al., 2023, Xing et al., 2024]. Video diffusion generates rasterized process frames [Ren et al., 2026].

We propose a model that generates ordered vector strokes, reuses a pretrained image prior, and allows the requested order to be specified in text. We utilize a commissioned dataset of 47,318 handmade artist drawings whose quality far exceeds the doodle corpora that existing sequential sketch work is built upon [Jongejan et al., 2016, Ge et al., 2021]. We construct an image-native intermediate representation through an order-as-color codec to enable training the DiT generative model within its trained image latent space. To bridge the gap between representation and vector, we train a VAE decoder to read the intermediate, and emit a draw order field, foreground mask, and stroke-instance segmentation embeddings. We apply smoothing and RDP simplification on the predicted field to yield the final vector stroke. To allow the trained model to follow instructions specifying arbitrary order, we train with a constrained permuted order, with the captions programmatically generated to match the permuted order. We utilize bounding box based annotations to classify strokes to textual captions. We train on both an image-derendering objective and a text-conditioned generation objective over permuted order. We evaluate the model on field, segmentation, geometry, and order metrics, and zero-shot text-to-sketch recognition.

We show that this model is capable of faithfully derendering sketches, following order instructions closely, and retaining the strong text-to-sketch generation capabilities of the prior.

## 2 Related Work

Prior systems differ most usefully in their output contract. Autoregression emits points, SVG operations, or learned stroke tokens in sequence [Ha and Eck, 2018, Wu et al., 2023, Tang et al.,

2024, Wang et al., 2025, Aksan et al., 2020]; diffusion operates over coordinate sequences or perstroke latents [Wang et al., 2023, Arar et al., 2025, Das et al., 2023, Zhou et al., 2026]; per-image optimization fits Bézier curves through differentiable rasterization [Vinker et al., 2022, Li et al., 2020] against pretrained image priors [Jain et al., 2023, Xing et al., 2024]; multimodal LLMs map images or text to SVG code or ordered strokes [Mitrevski et al., 2024, Rodriguez et al., 2025, Yang et al., 2025, Vinker et al., 2025, Xing et al., 2025], and a concurrent agent draws one semantic part at a time via multi-turn RL over part-annotated sketches [Du et al., 2026]; video models produce raster process frames rather than vectors [Ren et al., 2026]; part-sequential generators [Ge et al., 2021, Bhunia et al., 2021] and implicit-time representations [Bandyopadhyay et al., 2024, Das et al., 2022] round out the space. Across all of these, the drawing process is either the model’s own generation axis or not represented; none expose order as a free-form language-conditioned variable. Offline-to-online handwriting conversion supplies order-recovery metrics – RMSE, SNR, DTW against recorded trajectories [Diaz et al., 2024, Mohamed Moussa et al., 2023] – which inform our evaluation. Our backbone is Qwen-Image-Edit-2509 [Wu et al., 2025]: a VAE compressing pixels to a latent grid and a diffusion transformer trained by latent flow matching [Lipman et al., 2023], adapted with LoRA [Hu et al., 2022].

## 3 Method

## 3.1 Sketch dataset, hierarchical annotation, order captions, and permutation

We gather a corpus of 47,318 drawings, commissioned through 50 distinct artists on Upwork. Each drawing is stored as the raw Apple Pencil input stream plus a simplified per-point form $( x , y )$ forming a sequence of polylines, in this work we only use the simplified form. Drawings average 77.5 strokes and 6,864 points (median 57 and 5,548; p99 387 and 26,028).

Each drawing is annotated with a two-level bounding-box hierarchy, structured as region then subject (Figure 4). The drawing order from the dataset is commonly non continuous and contain many revisits, where additional detail is added to already drawn regions. We synthesize order-descriptive captions as a sequence of visits through named regions, specifying locale upon revisit, for example: “box1 left, then box2, then box1 right, revisited.” We then teach the model to generate conditioned on order by permuting the target’s stroke order and generate the matching caption, forming a order supervised pair. When permuting, within-region order is preserved while varying between-region order, so the permutation keeps the natural order of drawing from rough outline to details. We augment the image condition given to the model as reference, while preserving pixel perfect stroke mapping (Figure 2a).

## 3.2 Intermediate image representation with order-as-color encoding

Let a sketch ${ \cal S } = \{ s _ { i } \} _ { i = 1 } ^ { N }$ be a sequence of strokes, each a sequence of points $s _ { i } = \{ p _ { i , j } \} _ { j = 1 } ^ { M _ { i } }$ $p _ { i , j } \in \mathbb { R } ^ { 2 }$ . Write $\ell ( p _ { i , j } )$ for arc length accumulated from the first point of $s _ { 1 } ~ \mathrm { t o } ~ p _ { i , j }$ along the drawing, L for the total, and $\ell _ { i } ( p _ { i , j } )$ for arc length within $s _ { i }$ alone. The global arc $a \stackrel { \smile } { = } \tilde { \ell } / L \in \ \bar { [ 0 , 1 ] }$ is a single monotone scalar which represents the drawing’s progress. The within-stroke arc is $u = \ell _ { i } / \bar { \ell } _ { i } ( p _ { i , M _ { i } } )$ . The sketch $X = \operatorname { e n c } ( S )$ is encoded at each inked pixel,

$$
\begin{array} { r } { H = a \cdot \frac { 3 4 2 } { 3 6 0 } , \qquad S _ { \mathrm { s a t } } = 1 , \qquad V = 1 - \frac { 1 } { 2 } u , } \end{array}\tag{1}
$$

with HSV(0, 0, 1) (white) on background (Figure 5). This rasterized sketch is referred as the codec image. This encoding produces single-pixel polylines, and recorded brush width is not encoded.

## 3.3 Order-native decoder

We finetune Qwen-Image’s VAE’s decoder for better stroke field recovery, the encoder E stays frozen (Figure 1). The supervised target is to reconstruct the global arc field, mask, and stroke segmentation given the latent of the encoded intermediate representation image. The decoder trunk is retrained jointly with new heads: a CNN head is added to the trunk’s penultimate features, structured as a pyramid at full, 4×, and 16× resolution; a final convolution layer reads the channel-concatenated features and emits 10 channels: global arc $a ^ { \prime } ,$ , foreground logit $m ^ { \prime }$ , and an eight-dimensional instance embedding $e ^ { \prime } .$ . The pixel branch is architecturally unchanged, while training jointly with the new head

![](images/16d41a9fa0d432ef56444d148cd261e9c646906167225ebd8a263c0c99f39c0d.jpg)  
Figure 1: Decoder architecture and objective. The source strokes derive four targets: ink raster, stroke segmentation, arc field, and the intermediate HSV image. The frozen encoder and trained decoder trunk with pyramid head emit foreground, instance embedding (all eight raw channels shown), arc, and pixel outputs; each loss bridges an output to its target.

with a stroke-weighted L1 loss and perceptual loss; the full objective and weights are in Appendix A.   
We train this decoder for 30,000 steps.

## 3.4 Generative model and stroke recovery

The generator is Qwen-Image-Edit-2509 [Wu et al., 2025] adapted with a rank-64 LoRA [Hu et al., 2022] (α = 64), trained 50,000 steps on 32 H100-80GB GPUs (2,848 GPU hours) with the standard latent flow-matching objective on the latent of the intermediate representation, $z = E ( \mathrm { e n c } ( S ) )$ where the condition, text or image, is drawn from the permuted and augmented distribution of Section 3.1 (Figure 2). Optimization details are in Appendix A. Both trainings are implemented in bbml [Bradbury and Zhong, 2025].

We convert the decoder emitted fields to ordered polyline paths without a learned tracer. HDBSCAN clusters foreground embeddings $e ^ { \prime }$ to segmentations. Clusters are globally sorted by mean predicted arc a<sup>′</sup>. Within each cluster, a nearest-neighbour walks foreground m<sup>′</sup> pixels, splitting into subsegments then merging into full vector strokes, with vertices aligned to the 1024 × 1024 grid. These strokes are then smoothed, then simplified with Ramer-Douglas-Peucker with epsilon 0.5 px. The order inside and between strokes are determined by global arc a<sup>′</sup> (Algorithm 1 in the appendix). This polyline output is the final vectorized result.

## 4 Results

We evaluate 100 drawings from each of our dataset’s training holdout, Creative Birds, Creative Creatures [Ge et al., 2021], and FS-COCO [Chowdhury et al., 2022]; the latter three never enter DiT training. The primary order metric is Kendall τ between predicted and recorded stroke order after Hungarian geometric matching of strokes; since several orders are plausible per drawing, τ against the single recorded order is conservative. As controls, randomly permuting predicted order drives matched τ to 0.02–0.04 and full reversal to −1.0 on all sets, validating the metric. Where noted we also report DTW over stroke trajectories, resampled uniformly to 128 points, and accumulated cost is divided by path length.

## 4.1 The decoder ceiling

Decoding latents of encoded ground-truth codec images isolates losses from the encoder, decoder, and vectorization stack. The order-native head cuts arc $L _ { 1 }$ by 9–19× relative to reading hue from the frozen decoder, with arc Spearman ≥ 0.994 and mask IoU ≥ 0.97 on all five datasets (full tables in Appendix C). Carried through segmentation and polyline fitting, the deployment ceiling reaches matched Kendall τ of 0.91–0.94 on the four multi-stroke datasets (0.56 on QuickDraw, whose sparse short strokes fragment).

![](images/478dc44e7768779eef9b1d7b34f3c29b89800011b714ecfc1b8674cdbe1a35cd.jpg)  
Figure 2: (a) Order permutation with local order preservation from Section 3.1. (b) The DiT trains in the latent space of the codec image. (c) At inference the predicted latent feeds the order-native decoder; the pixel output is discarded.

Table 1: Caption interventions under fixed inputs and seed (n = 100 per row). Top panel: in-domain holdout, caption deleted. Bottom panel: Creative Birds+Creatures – captionless, recorded part order stated, or reversed. Order metrics are measured against the recorded order; geometry columns show the intervention leaves the drawing itself unchanged.
<table><tr><td>panel</td><td>caption</td><td>Kendall τ</td><td>stroke match</td><td>mask IoU</td></tr><tr><td rowspan="2">holdout (in-domain)</td><td>no caption</td><td>0.096</td><td>0.896</td><td>0.775</td></tr><tr><td>with ċaption</td><td>0.449</td><td>0.895</td><td>0.774</td></tr><tr><td rowspan="3">Birds+Creatures (out-of-domain)</td><td>no caption</td><td>0.139</td><td>0.941</td><td>0.599</td></tr><tr><td>recorded order</td><td>0.373</td><td>0.934</td><td>0.601</td></tr><tr><td>reversed order</td><td>-0.267</td><td>0.934</td><td>0.601</td></tr></table>

## 4.2 Order is written by language

Table 1 measures the instruction’s effect on predicted order. Geometry never moves under any intervention while order swings across its full range: deleting the caption drops in-domain order to the level of geometry-only baselines (Appendix B), stating the recorded order lifts out-of-domain adherence, and reversing the same instruction inverts it – the caption, not the condition image, is the order channel, and the model follows a stated order even when it contradicts the recording. Figure 3 shows how instructions steer the produced order on held-out sketches.

## 4.3 The image prior survives the finetune

The backbone’s open-vocabulary drawing ability is retained allowing generation of any ordered sketch with the original model’s world knowledge. Prompted with 50 QuickDraw categories it never saw as vectors, the model’s final rendered SVGs are recognized by CLIP ViT-B/32 at Top-1 0.70 (guidance 5.0) and 0.75 (7.5), Top-5 0.84 (Table 2). We showcase a representative sample in Figure 6, last row.

![](images/d29a4b029580610f819ccf29b92a7950f942c0de6be4a5e70807d43b8e29a9a5.jpg)

Figure 3: Draw order under text intervention: four held-out sketches, each derendered under a hand written order instruction. Fitted path order (viridis, first to last) and replay at 15/35/55/75/100% – black: already drawn, blue: added since the previous frame. Guidance 10.

Table 2: QuickDraw text-to-vector recognition, CLIP ViT-B/32 over 50 categories × two seeds.
<table><tr><td>CFG</td><td>Top-1 ↑</td><td>Top-5 ↑</td><td>target rank ↓</td><td>target probability ↑</td></tr><tr><td>5.0</td><td>0.70</td><td>0.78</td><td>6.23</td><td>0.640</td></tr><tr><td>7.5</td><td>0.75</td><td>0.84</td><td>4.62</td><td>0.692</td></tr></table>

## 5 How precise is order control?

Order instructions name units such as regions, subjects, parts, while leaving order inside them unspecified. We score against the instruction’s specifications by assigning predicted order to ground truth strokes. We report Kendall τ between predicted order and instructed order, per unit over the strokes the instruction constrains. Residual order inside units is reported separately as within-unit τ .

Adherence degrades with instruction granularity. On the holdout test set we issue permuted order instructions at the three trained caption shapes, instructing top level region order, subject order, and revisit-instructing passes (Table 3). Region-level instructions are followed reliably, adherence falls to 0.46 at part level and 0.44 at pass level.

An external part-annotated test. ControlSketch-Part [Du et al., 2026] pairs each synthetic 32- stroke sketch with 2 to 5 named parts assigned to each stroke. We derender 100 test sketches from categories disjoint from both its training split and ours. Given order instructions, τ reaches 0.78 as stated and 0.73 reversed, from an uninstructed baseline of 0.36; mean per-sample Spearman between instructed and produced part order is 0.90 and 0.87.

## 6 Limitations

Training permutes stroke part order and rewrites the caption to match, deliberately decorrelating order from geometry; the direct cost is that, given no instruction, the model has no verifiable human-like global order prior. Due to our training captioning schema, the order of objects in the caption text influences the draw order, and generalization to instructed order conflicting to order of appearance is reduced. An explicit order clause survives an adjacent description at a measured cost (τ 0.78 → 0.73 when the stated order matches the description’s mention order, 0.73 → 0.47 when it opposes it), and a single trailing constraint is lost entirely (named region drawn first 0.315 of the time appended to the full caption, vs. 0.326 uninstructed and 0.517 for the same sentence alone). Below named units there is no control at all: within-unit order agreement is near zero under every instruction (Table 3), and inverting a region’s trained pass convention succeeds well under half the time (0.400, against an uninstructed 0.267). The amount of control is constrained by the unit-size which the permutation training used.

Table 3: Stroke-level Kendall τ against each cell’s own instruction, over instruction-constrained stroke pairs (n = 100). Parenthesized: mean instructed units per drawing. Within-unit τ: residual stroke order inside instructed units.
<table><tr><td>panel</td><td>instruction</td><td>stroke τ</td><td>within-unit τ</td></tr><tr><td rowspan="4">holdout</td><td>none (ordinary caption)</td><td>0.149</td><td>0.382</td></tr><tr><td>permuted regions (2.8)</td><td>0.838</td><td>0.282</td></tr><tr><td>permuted parts (7.1)</td><td>0.461</td><td>0.177</td></tr><tr><td>permuted passes (14.9)</td><td>0.444</td><td>0.134</td></tr><tr><td rowspan="3">ControlSketch-Part</td><td>none (ordinary caption)</td><td>0.363</td><td>0.088</td></tr><tr><td>stated part order</td><td>0.778</td><td>0.058</td></tr><tr><td>reversed part order</td><td>0.734</td><td>0.068</td></tr></table>

The decoder and vectorization causes fragmentation. Recovered paths run 1.7–2.5× the true stroke count, so strokes are recovered in pieces. Our objective drops recorded time, pressure, tilt, and brush width; predicted paths are 1 px unweighted centerlines parameterized by cumulative arc length rather than time. Finally, our out-of-domain sets are doodle-flavored; adherence on complex real line drawings is so far only qualitative.

## 7 Conclusion

We present a system that generates ordered vector sketches by rendering draw order into color, generating that image with a pretrained diffusion transformer, and reading order back out with an order-native decoder into replayable polyline paths. Trained on 47,318 commissioned artist drawings, the model derenders sketches faithfully and retains the prior’s open-vocabulary drawing ability. On held-out data the model recovers substantial final-vector order. Text is the order channel: captions carry most in-domain order, recorded-order instructions improve it, and reversed instructions invert it, all without changing geometry.

## References

Emre Aksan, Thomas Deselaers, Andrea Tagliasacchi, and Otmar Hilliges. Cose: Compositional stroke embeddings. In Advances in Neural Information Processing Systems (NeurIPS), 2020. arXiv:2006.09930.

Ellie Arar, Yarden Frenkel, Daniel Cohen-Or, Ariel Shamir, and Yael Vinker. Swiftsketch: A diffusion model for image-to-vector sketch generation. In SIGGRAPH, 2025. arXiv:2502.08642.

Hmrishav Bandyopadhyay, Ankan Kumar Bhunia, Pinaki Nath Chowdhury, Aneeshan Sain, Tao Xiang, and Yi-Zhe Song. Sketchinr: A first look into sketches as implicit neural representations. In Conference on Computer Vision and Pattern Recognition (CVPR), 2024. arXiv:2403.09344.

Ankan Kumar Bhunia, Salman Khan, Hisham Cholakkal, Rao Muhammad Anwer, Fahad Shahbaz Khan, Jorma Laaksonen, and Michael Felsberg. Doodleformer: Creative sketch drawing with transformers. arXiv preprint arXiv:2112.03258, 2021.

Rowan Bradbury and Elea Zhong. bbml: A better basic machine learning framework. https://github.com/ Bradbury-Group/bbml, 2025.

Pinaki Nath Chowdhury et al. Fs-coco: Towards understanding of freehand sketches of common objects in context. In European Conference on Computer Vision (ECCV), 2022.

Ayan Das, Yongxin Yang, Timothy Hospedales, Tao Xiang, and Yi-Zhe Song. Chirodiff: Modelling chirographic data with diffusion models. In International Conference on Learning Representations (ICLR), 2023. arXiv:2304.03785.

Ayan Das et al. Sketchode: Learning neural sketch representation in continuous time. In International Conference on Learning Representations (ICLR), 2022.

Moises Diaz, Gioele Crispo, Antonio Parziale, Angelo Marcelli, and Miguel A. Ferrer. Writing order recovery in complex and long static handwriting. arXiv preprint arXiv:2406.03194, 2024.

Xiaodan Du, Ruize Xu, David Yunis, Yael Vinker, and Greg Shakhnarovich. Teaching an agent to sketch one part at a time. arXiv preprint arXiv:2603.19500, 2026.

Songwei Ge, Vedanuj Goswami, C. Lawrence Zitnick, and Devi Parikh. Creative sketch generation. In International Conference on Learning Representations (ICLR), 2021. arXiv:2011.10039; DoodlerGAN.

David Ha and Douglas Eck. A neural representation of sketch drawings. In International Conference on Learning Representations (ICLR), 2018. arXiv:1704.03477.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022.

Ajay Jain, Amber Xie, and Pieter Abbeel. Vectorfusion: Text-to-svg by abstracting pixel-based diffusion models. In Conference on Computer Vision and Pattern Recognition (CVPR), 2023. arXiv:2211.11319.

Jonas Jongejan, Henry Rowley, Takashi Kawashima, Jongmin Kim, and Nick Fox-Gieg. The quick, draw! dataset, 2016. Google Creative Lab.

Tzu-Mao Li, Michal Lukác, Michaël Gharbi, and Jonathan Ragan-Kelley. Differentiable vector graphicsˇ rasterization for editing and learning. ACM Transactions on Graphics (SIGGRAPH Asia), 39(6), 2020.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023.

Blagoj Mitrevski et al. Inksight: Offline-to-online handwriting conversion by learning to read and write. Transactions on Machine Learning Research (TMLR), 2024. arXiv:2402.05804.

Elmokhtar Mohamed Moussa, Thibault Lelore, and Harold Mouchère. SET, SORT! a novel sub-stroke level transformers for offline handwriting to online conversion. In International Conference on Document Analysis and Recognition (ICDAR), pages 81–97, 2023. hal-04182547.

Hui Ren, Yuval Alaluf, Omer Bar-Tal, Alexander Schwing, Antonio Torralba, and Yael Vinker. VideoSketcher: Sequential sketch generation using video model priors. arXiv preprint arXiv:2602.15819, 2026.

Juan A. Rodriguez et al. Starvector: Generating scalable vector graphics code from images and text. In Conference on Computer Vision and Pattern Recognition (CVPR), 2025. arXiv:2312.11556.

Zecheng Tang et al. Strokenuwa: Tokenizing strokes for vector graphic synthesis. In International Conference on Machine Learning (ICML), 2024. arXiv:2401.17093.

Yael Vinker et al. Clipasso: Semantically-aware object sketching. ACM Transactions on Graphics (SIGGRAPH), 41(4), 2022.

Yael Vinker et al. Sketchagent: Language-driven sequential sketch generation. In Conference on Computer Vision and Pattern Recognition (CVPR), 2025. arXiv:2411.17673.

Jiawei Wang et al. Vq-sgen: A vector quantized stroke representation for creative sketch generation. In International Conference on Computer Vision (ICCV), 2025. arXiv:2411.16446.

Qiang Wang et al. Sketchknitter: Vectorized sketch generation with diffusion models. In International Conference on Learning Representations (ICLR), 2023.

Chenfei Wu et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

Ronghuan Wu, Wanchao Su, Kede Ma, and Jing Liao. Iconshop: Text-guided vector icon synthesis with autoregressive transformers. ACM Transactions on Graphics (SIGGRAPH Asia), 2023. arXiv:2304.14400.

Ximing Xing, Haitao Zhou, Chuang Wang, Jing Zhang, Dong Xu, and Qian Yu. Svgdreamer: Text guided svg generation with diffusion model. In Conference on Computer Vision and Pattern Recognition (CVPR), 2024. arXiv:2312.16476.

Ximing Xing et al. Empowering llms to understand and generate complex vector graphics. In Conference on Computer Vision and Pattern Recognition (CVPR), 2025. arXiv:2412.11102.

Yiying Yang et al. Omnisvg: A unified scalable vector graphics generation model. In Advances in Neural Information Processing Systems (NeurIPS), 2025. arXiv:2504.06263.

Jin Zhou, Yi Zhou, Hongliang Yang, Pengfei Xu, and Hui Huang. Strokefusion: Vector sketch generation via joint stroke-udf encoding and latent sequence diffusion. In AAAI Conference on Artificial Intelligence, 2026. arXiv:2503.23752.

## A Decoder objective and training details

$$
\begin{array} { r l } & { \mathbb { W } \mathrm { r i t i n g ~ } z = E ( x ) , x ^ { \prime } = D _ { p } ( z ) , \mathrm { ~ a n d ~ } ( a ^ { \prime } , m ^ { \prime } , e ^ { \prime } ) = D _ { \mathrm { o r d } } ( z ) , } \\ & { \quad \mathcal { L } = \underbrace { w _ { 1 } \left\| w _ { \mathrm { f g } } \odot ( x ^ { \prime } - x ) \right\| _ { 1 } } _ { \mathrm { f o r e g r o u n d - w e i g h t e d ~ p i x e l ~ } L _ { 1 } } } \\ & { \qquad + w _ { \mathrm { l p i p s } } \mathrm { L P I P S } ( x ^ { \prime } , x ) + w _ { \mathrm { a r c } } \Big [ \big \| ( a ^ { \prime } - a ) \odot m \big \| _ { 1 } + 0 . 2 5 \mathrm { B C E } _ { w } ( m ^ { \prime } , m ) + 0 . 1 \mathcal { L } _ { \mathrm { p p } } ( e ^ { \prime } , s ) \Big ] , } \end{array}\tag{2}
$$

where s is the ground-truth stroke-id map, $w _ { 1 } = w _ { \mathrm { l p i p s } } = 1 , w _ { \mathrm { a r c } } = 8 ;$ stroke pixels and their radius-1 surround receive 50× pixel weight, and the mask BCE uses the same boundary ring with a dynamic positive-class weight. $\mathcal { L } _ { \mathrm { p p } }$ is discriminative push–pull: embeddings contract toward their stroke mean outside radius $d _ { v } \stackrel { = } { = } 0 . 5 ;$ stroke means separate by margin $\bar { 2 d } _ { d } = 3 . 0 .$ . Figure 1 shows the architecture.

The DiT LoRA trains with AdamW $( \beta = ( 0 . 9 , 0 . 9 5 )$ , weight decay 0.01), learning rate $1 0 ^ { - 4 }$ with 100 warmup steps and 10,000 steps of decay, global batch size 32, gradient clipping 1.0, for 50,000 steps (Figure 2).

Algorithm 1 Image-to-vector deployment path   
Require: condition image c, caption q, generator ${ \overline { { G } } } ,$ decoder D   
1: $\overline { { z } }  G ( c , q )$   
2: $( x ^ { \prime } , a ^ { \prime } , \dot { m ^ { \prime } } , e ^ { \prime } ) \gets D ( \hat { z } )$   
3: $\dot { \mathcal { T } } \gets \dot { \mathrm { H D B S C A N } } ( \{ \dot { e } ^ { \prime } ( p ) : m ^ { \prime } ( p ) = 1 \} )$   
4: assign embedding-noise pixels to nearest instance centroid   
5: π ← instances sorted by mean $a ^ { \prime }$   
6: for all instances I in order π do   
7: $S _ { I } \gets \mathbf { C } \mathbf { \mathbf { A P P E D R A } }$ DIUS $\mathbf { W A L K } ( I )$   
8: for all segments $s \in S _ { I }$ do   
9: orient s from lower to higher endpoint arc   
10: end for   
11: $\mathcal { C } _ { I } \gets$ JOINNEARESTENDPOINTS $( S _ { I } )$   
12: $\mathcal { B } _ { I }  \{ \mathrm { F I T P O L Y L I N E C H A I N } ( c ) : c \in \mathcal { C } _ { I } \}$   
13: end for   
14: return ordered vectors $\boldsymbol { B } _ { \pi _ { 1 } } , \dots , \boldsymbol { B } _ { \pi _ { | \boldsymbol { \tau } | } }$

## B Measuring unconditional generation order

Table 4 reports the full pipeline without order instruction, against two built-in anchors: four determin istic geometry-only orderings of the model’s own fitted paths (top-to-bottom, longest-first, centre-out, greedy pen travel; best reported) and a per-drawing retroactive oracle over the four. The holdout row deletes the caption, since the holdout’s native caption narrates the recorded drawing process (Section $4 . 2 ) ;$ ; the other sets’ native captions carry no order text. Uninstructed, in-domain order sits at the heuristic level (0.096 vs. 0.105); on FS-COCO generated order beats the best heuristic by +0.11 $( p < 1 0 ^ { - 3 }$ , Wilcoxon); on Birds and Creatures the gap is insignificant. An DTW control agrees: on Birds and Creatures generated order is indistinguishable from paired random order (0.189 vs. 0.188, 0.182 vs. 0.180; lower is better), on FS-COCO it is better (0.200 vs. 0.227). Comparing against the decoder ceiling attributes the pipeline’s losses: fragmentation is decode-side (in-domain path-count ratio 2.29 at ceiling vs. 2.27 end-to-end with the native caption), while order is generation-side – with the recorded-process caption, end-to-end τ reaches 0.449 (Table 1) against a 0.91 decode ceiling. Recorded artist order is also simply hard to guess from geometry – on ground-truth strokes the best heuristic reaches τ = 0.137 and the oracle 0.382 – and dense drawings are the hard case even when instructed: under the native caption the simplest holdout quartile (4–26 strokes) reaches $\tau = 0 . 5 9$ the busiest (91–379) 0.37. Figure 6 shows deterministic median samples per domain.

Table 4: End-to-end results at 50k steps without order instruction $( n = 1 0 0$ per dataset). Holdout is generated with the caption deleted; the other sets use their native captions, which carry no order text. Heuristic rows re-order the model’s own fitted paths under the identical matching; oracle is the retroactive per-drawing best of the four heuristics. Bold: arc order significantly above the best heuristic (Wilcoxon).
<table><tr><td></td><td>holdout</td><td>Birds</td><td>Creatures</td><td>FS-COCO</td></tr><tr><td>mask IoU ↑</td><td>0.775</td><td>0.600</td><td>0.604</td><td>0.646</td></tr><tr><td>stroke match rate ↑</td><td>0.896</td><td>0.922</td><td>0.955</td><td>0.795</td></tr><tr><td>path-count ratio  $ 1$ </td><td>2.53</td><td>1.76</td><td>2.32</td><td>1.60</td></tr><tr><td>Chamfer  $\times 1 0 ^ { 3 } \downarrow$ </td><td>0.60</td><td>0.45</td><td>0.52</td><td>0.61</td></tr><tr><td>Kendall  $\tau ,$  generated arc order  $\uparrow$ </td><td>0.096</td><td>0.256</td><td>0.213</td><td>0.308</td></tr><tr><td>Kendall  $\tau ,$  best geometric heuristic</td><td>0.105</td><td>0.223</td><td>0.190</td><td>0.201</td></tr><tr><td>Kendall  $\tau ,$  heuristic oracle</td><td>0.374</td><td>0.409</td><td>0.355</td><td>0.308</td></tr></table>

## C Decoder ceiling tables

Table 5: Decoder field ceiling from encoded ground-truth latents (n = 100 each). Frozen reads arc from hue and mask from saturation; Embed is the final order-native decoder. “Head” is its calibrated explicit foreground output.
<table><tr><td></td><td colspan="2">arc L1 ↓</td><td colspan="2">arc Spearman ↑</td><td colspan="2">saturation mask IoU ↑</td><td rowspan="2">head IoU ↑ Embed</td></tr><tr><td>dataset</td><td>Frozen</td><td>Embed</td><td>Frozen</td><td>Embed</td><td>Frozen</td><td>Embed</td></tr><tr><td>holdout</td><td>0.0607</td><td>0.0071</td><td>0.8072</td><td>0.9937</td><td>0.8049</td><td>0.9705</td><td>0.9706</td></tr><tr><td>Birds</td><td>0.0517</td><td>0.0027</td><td>0.8329</td><td>0.9995</td><td>0.8188</td><td>0.9979</td><td>0.9980</td></tr><tr><td>Creatures</td><td>0.0468</td><td>0.0035</td><td>0.8670</td><td>0.9984</td><td>0.8132</td><td>0.9955</td><td>0.9957</td></tr><tr><td>FS-COCO</td><td>0.0605</td><td>0.0036</td><td>0.7825</td><td>0.9983</td><td>0.8700</td><td>0.9952</td><td>0.9952</td></tr><tr><td>QuickDraw</td><td>0.0448</td><td>0.0030</td><td>0.8675</td><td>0.9995</td><td>0.8304</td><td>0.9986</td><td>0.9987</td></tr></table>

Table 6: Deployment ceiling of the final decoder, including embedding segmentation and polyline fitting (n = 100).
<table><tr><td></td><td>seg. ARI</td><td>seg. IoU</td><td>path ratio</td><td>match</td><td>Kendall τ</td></tr><tr><td>holdout</td><td>0.656</td><td>0.603</td><td>2.29</td><td>0.907</td><td>0.909</td></tr><tr><td>Birds</td><td>0.892</td><td>0.747</td><td>1.60</td><td>0.938</td><td>0.944</td></tr><tr><td>Creatures</td><td>0.840</td><td>0.720</td><td>1.93</td><td>0.955</td><td>0.941</td></tr><tr><td>FS-COCO</td><td>0.698</td><td>0.618</td><td>1.47</td><td>0.783</td><td>0.912</td></tr><tr><td>QuickDraw</td><td>0.588</td><td>0.763</td><td>4.37</td><td>0.977</td><td>0.561</td></tr></table>

## D Condition funnel details

Table 7: Condition funnel: 20 image lanes and 8 caption rungs over 47,190 drawings. pass is image-alone sufficiency; +text additionally credits the paired caption.
<table><tr><td>category</td><td>lane</td><td>rows</td><td>pass</td><td>+text</td></tr><tr><td>procedural</td><td>photometric (7 lanes)</td><td>47,187–47,190</td><td>≥ 0.999</td><td>≥ 0.999</td></tr><tr><td></td><td>width</td><td>47,190</td><td>0.856</td><td>0.942</td></tr><tr><td></td><td>stack</td><td>47,190</td><td>0.776</td><td>0.942</td></tr><tr><td></td><td>pressure</td><td>47,187</td><td>0.770</td><td>1.000</td></tr><tr><td></td><td>m_pencil</td><td>10,331</td><td>0.430</td><td>0.712</td></tr><tr><td>compose</td><td>compose1/2/3</td><td>47,190</td><td>0.964</td><td>0.982</td></tr><tr><td>structural</td><td>prefix, prefix_uv</td><td>47,187</td><td>1.000</td><td>1.000</td></tr><tr><td></td><td>phase_suffix, phase_interior</td><td>47,086–47,092</td><td>1.000</td><td>1.000</td></tr><tr><td>caption</td><td>content (4 rungs)</td><td>40,252</td><td>≥ 0.999</td><td></td></tr><tr><td></td><td>order (4 rungs)</td><td>40,252</td><td>1.000</td><td></td></tr></table>

## E Order-encoding ablation

Order could instead be packed into one monotone RGB channel. We bracketed eight such encoders against the HSV scheme, decoder-free: matched nearest-neighbor decode through the frozen VAE, scored by Spearman ρ against true order (Table 8). Three findings decided the design. Normalization across the slow channel’s full range is the only lever: the unnormalized odometer collapses because order sits in a tiny-range channel the VAE swamps, while base-N coarsening changes nothing. Fixedarc sinusoidal codes self-intersect once total arc exceeds half the coarsest wavelength, folding order – a representational failure, not a decoder limitation. And single-axis packing destroys within-stroke order $( \rho \approx 0 . 0 5$ through the VAE, from 1.0 clean) because fine order rides the low-order bytes the VAE removes first; HSV ties the best single-axis encoder cross-stroke and keeps within-stroke order readable via its dedicated low-frequency V channel. These frozen-VAE ceilings guided the codec choice and predate the order-native decoder, which reads the chosen code far more precisely (Appendix C).

Table 8: Coarse-order survival through the frozen VAE: matched nearest-neighbor decode, Spearman ρ vs. true order (n = 36 holdout; HSV rows n = 8). Clean decode is ≈ 1.0 for all encoders except the fixed-arc codes, which self-intersect.
<table><tr><td>encoder</td><td>order axis</td><td>ρ through VAE</td></tr><tr><td>odometer, absolute base-256</td><td>cross-stroke</td><td>0.259</td></tr><tr><td>odometer, normalized base-256</td><td>cross-stroke</td><td>0.821</td></tr><tr><td>normalized base-64 / 48 / 32</td><td>cross-stroke</td><td>0.831 / 0.833 / 0.833</td></tr><tr><td>sinusoidal phase (cyc)</td><td>cross-stroke</td><td>0.842</td></tr><tr><td>fixed-arc sinusoidai,  $\lambda _ { \operatorname* { m a x } } { = } 2 6 \mathbf { k } / 6 5 \mathbf { k } ~ \mathrm { p x }$ </td><td>cross-stroke</td><td>0.608 / 0.557</td></tr><tr><td>HSV (ours), hue</td><td>cross-stroke</td><td>0.843</td></tr><tr><td>HSV (ours), value</td><td>within-stroke</td><td>0.634</td></tr></table>

## F Raster-code diagnostics

Table 9: Visible-channel diagnostics after 8-bit RGB quantization. Stroke survival: fraction of source strokes retaining a visible pixel after z-buffering.

<table><tr><td></td><td>mean</td><td>tail / worst</td></tr><tr><td>saturation-mask IoU</td><td>1.000000</td><td>1.000000</td></tr><tr><td>visible global-arc MAE</td><td>0.000268</td><td>0.000824 (p99)</td></tr><tr><td>visible global-arc Spearman ρ</td><td>0.999999</td><td>0.999999</td></tr><tr><td>visible within-stroke-arc MAE</td><td>0.001965</td><td></td></tr><tr><td>stroke survival after z-buffer</td><td>0.9928</td><td>0.8667 (worst)</td></tr></table>

![](images/badf15bb3c64510d841b3b9fe7b375b545bcbac729f29771e69e87a84c241d4a.jpg)

## G Additional figures

(a)  
![](images/4a283d5d95fbc6770e8846c9e41a0e8ca756771d9b1843311674e09dada9b276.jpg)

(b)  
stroke assignment by maximal bbox overlap coverage 96.4%  
![](images/ca88814e071ce6f50f140c698ccf50535f5d30e74d28da440cb697ee9aeb46c7.jpg)

Figure 4: Hierarchical annotation on one sketch. (a) Solid boxes: regions; dashed: subjects. (b) Strokes colored by assigned label.  
(a)  
(b)  
(c)  
![](images/b8a133bbbacd4e41c4145db33ae10844019b7c95bddb3799245970ba231612c6.jpg)

(d)  
![](images/b5576e3079fd76467db912e46fed24e689d3c3f6134494fd1702c3ae9f5cb381.jpg)

(e)  
![](images/c8ab224907c46f16a46801aa9124a2348a7c9b3750f30c8a01dfcf91e1824787.jpg)  
Figure 5: The encoding, on one held-out drawing (15 strokes). (a) Source ink. (b) $H = a \cdot 3 4 2 / 3 6 0 \colon$ order as a red-to-violet sweep. (c) S: the visible stroke mask. (d) $V = 1 - u / 2$ on foreground. (e) The composed RGB target consumed by the VAE.

a) Birds: Somethings not right.... Draw in this recorded order: eye, then beak, then body, then wings, then tail, then legs, then details.

![](images/e1e3c0729ad3d3d2d52c32c49675f7629a9508a662cedd3b625d4693bb87458c.jpg)  
Figure 6: Representative 50k outputs, deterministic median selections per domain. (a) Birds and (b) Creatures condition on the input sketch with the recorded part order appended to the caption; (c) FS-COCO uses its plain scene caption (the dataset records no order text); (d) QuickDraw is text-to-sketch, no input image. Columns: input sketch, fitted path order, then cumulative progress frames (black = existing strokes, blue = strokes added since the previous frame).

![](images/23070a84e89c1c2aa65df0b9da0b07b4174da435f8e5c5aed38edf8ecdcf62c9.jpg)  
Figure 7: One deterministically-sampled drawing from each of the 50 commissioned artists.