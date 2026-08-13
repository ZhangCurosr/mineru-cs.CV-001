# Few-Shot Ordinal Learning for Day-Wise Freshness Estimation with Hyperspectral Fish Images

Kazi Nabiul Alam <sup>1</sup>, Pooneh Bagheri-Zadeh <sup>1</sup>, Akbar Sheikh-Akbari <sup>1⋆</sup>

<sup>1</sup> School of Built Environment, Engineering and Computing,

Leeds Beckett University, Leeds, United Kingdom.

Correspondence: A.Sheikh-Akbari@leedsbeckett.ac.uk

Abstract—Non-destructive food quality assessment has increasingly benefited from hyperspectral imaging (HSI), which captures spectral signatures linked to biochemical changes during storage. Estimating day-wise freshness, however, remains challenging owing to strong inter-fillet variability and scarce labelled data per product. All existing deep learning approaches for HSI-based freshness prediction operate under full supervision, requiring densely annotated training sets that are costly to obtain at the individual-product level. We introduce, to the best of our knowledge, the first few-shot learning framework for HSI-based food quality estimation. Each fillet defines a distinct episodic task, and a CORAL-style ordinal prediction head captures the ranked nature of freshness progression through cumulative threshold modelling. Biologically grounded monotonicity and embedding smoothness constraints further guide predictions toward plausible trajectories. On a 16-day salmon HSI dataset under a strict unseen-fillet protocol, our method achieves a mean absolute error of 1.58 days and ±2-day accuracy of 72.3% with only three labelled days per fillet, substantially outperforming scalar regression and label-distribution baselines under an identical unseen-fillet protocol.

Index Terms—hyperspectral imaging, few-shot learning, ordinal regression, CORAL, fish freshness, episodic meta-learning

## I. INTRODUCTION

Ensuring that perishable food products remain fresh is vital for consumer health, waste reduction, and efficient supply chain management. Conventional approaches to freshness evaluation, including sensory panels, chemical assays, and microbiological cultures, tend to be destructive, slow, and costly, restricting their scalability in industrial settings. Hyperspectral imaging (HSI) [1], [2] offers a compelling alternative by merging spectroscopy with spatial imaging, thereby producing rich spectral–spatial signatures [3] that reflect underlying biochemical changes during storage and spoilage.

Deep learning architectures, particularly convolutional neural networks (CNNs), have demonstrated considerable promise in extracting complex spectral–spatial patterns from HSI data for quality and safety inspection across a range of food products including fruits, vegetables, and meat [2], [4]. Deep architectures incorporating spectral and spatial convolutions [4] and ordinal regression heads [11] have shown promise for quality timeline prediction, while compact few-shot designs achieve competitive classification accuracy with substantially fewer parameters [5]. A recurring limitation, however, is that these fully supervised models require large annotated datasets, something rarely available when freshness labels must be assigned per individual fillet at fine temporal granularity. Beyond data scarcity, most HSI-based studies frame freshness estimation [2] as either standard regression or nominal classification, overlooking the inherent ordering among storage days.

Addressing data scarcity, few-shot learning reformulates training around task distributions sampled through episodic protocols, enabling generalization from very few labelled examples [6], [7]. Well-known deep few-shot strategies, including prototypical, matching, relation, and gradient-based metalearners, have proven effective under low-shot constraints [8], [9]. Within HSI, few-shot approaches have recently been applied to remote sensing and crop classification with limited labels [5], [10].

Standard regression and classification objectives, however, fail to preserve ordinal relationships between storage days, potentially producing temporally inconsistent estimates. Ordinal regression addresses this by explicitly modelling ordered label spaces [11], [12], [14]. Consistent Rank Logits (CORAL) [13] decompose the ordinal problem into binary tasks with weight sharing, guaranteeing rank consistency, yet combining such methods with few-shot paradigms remains unexplored.

We propose a label-efficient episodic ordinal regression framework for day-wise freshness estimation from hyperspectral images. Rather than relying on support-conditioned metric comparison, it attains few-shot generalization through episodic task sampling combined with biologically grounded ordinal regularizers that keep predictions temporally consistent on fillets never seen during training. To the best of our knowledge, this is the first application of few-shot learning to HSI-based food quality or freshness prediction, where all prior deep learning solutions have operated under full supervision [1]– [3]. The episodic protocol is paired with CORAL-style ordinal heads and dual regularization: monotonicity on predicted days and embedding smoothness on learned representations. Experiments show this combination decisively outperforms scalar regression and distribution-based baselines under a strict unseen-fillet evaluation.

Our key contributions are:

• The first episodic, label-efficient framework for HSIbased fish quality estimation, performing day-wise prediction on entirely unseen fillets.

• CORAL-style cumulative ordinal regression to explicitly encode the ranked structure of storage timelines.

• Applying biologically motivated dual regularization (monotonicity and embedding smoothness) to enforce temporally consistent freshness trajectories.

## II. METHODOLOGY

## A. Problem Formulation and Episodic Framework

Given hyperspectral cubes $\boldsymbol { x } \in \mathbb { R } ^ { B \times H \times W }$ with freshness labels $y \in \{ 1 , \ldots , D \}$ , we formulate day-wise estimation as few-shot ordinal regression. Each fillet defines a task $\mathcal { T } _ { i }$ split into a k-day support set $s _ { i }$ and a query set $\mathcal { Q } _ { i }$ of remaining days. Both contribute to training:

$$
\operatorname* { m i n } _ { \theta } \mathbb { E } _ { \mathcal { T } _ { i } } \left[ \frac { 1 } { 2 } \left( \mathcal { L } ( S _ { i } ; \theta ) + \mathcal { L } ( \mathcal { Q } _ { i } ; \theta ) \right) + \lambda _ { R } \mathcal { R } ( S _ { i } \cup \mathcal { Q } _ { i } ; \theta ) \right] ,\tag{1}
$$

where $\mathcal { L }$ is the ordinal loss and R encompasses temporal regularization. Support and query samples pass through the same shared network $f _ { \theta }$ with identical weights, not separate branches; they differ only in role within the episodic objective, where support days fix the per-fillet temporal anchors over which the regularizers operate and query days provide the held-out supervision. Averaging over both sets prevents overfitting to the few support samples.

## B. Spectral-Channel CNN Backbone

Our backbone treats B spectral bands as input channels to a 2D CNN rather than using 3D convolutions. Four convolutional blocks (32→64→128→128 channels) with batch normalization, ReLU, and $2 \times 2$ max pooling extract hierarchical spatial features:

$$
\begin{array} { r } { \mathbf { h } ^ { ( l ) } = \mathbf { M a x P o o l } \big ( \mathbf { R e L U } \big ( \mathbf { B N } \big ( \mathbf { C o n v 2 D } \big ( \mathbf { h } ^ { ( l - 1 ) } \big ) \big ) \big ) . } \end{array}\tag{2}
$$

Adaptive average pooling and a fully connected layer yield the embedding:

$$
\mathbf { z } = \mathrm { R e L U } ( \mathbf { W } _ { \mathrm { f c } } \cdot \mathrm { A d a p t i v e A v g P o o l } ( \mathbf { h } ^ { ( 4 ) } ) + \mathbf { b } _ { \mathrm { f c } } ) \in \mathbb { R } ^ { d } ,\tag{3}
$$

with $d { = } 2 5 6$ . Cross-band correlations are learned implicitly through channel mixing. Table I presents the full architecture (441K parameters, 2.37 GFLOPs).

Treating spectral bands as input channels to a 2D CNN, rather than 3D or 1D spectral convolutions, is a deliberate trade-off: with B=256 bands and few labelled days per fillet, 3D convolution sharply increases parameters and overfits, whereas channel mixing in the first convolution still learns inter-band combinations jointly.

TABLE I: FreshnessOrdinal Network Architecture (B=256 Bands)
<table><tr><td>Layer</td><td>Output Shape</td><td>Params</td></tr><tr><td>HSICNN Backbone</td><td></td><td></td></tr><tr><td>Conv2D (B→32) + BN + Pool</td><td> $3 2 \times 6 4 \times 6 4$ </td><td>133,088</td></tr><tr><td>Conv2D (32→64) + BN + Pool</td><td>64 × 32 × 32</td><td>18,624</td></tr><tr><td>Conv2D (64→128) + BN + Pool</td><td>128 × 16 × 16</td><td>74,112</td></tr><tr><td>Conv2D (128→128) + BN + Pool</td><td>128 × 8 × 8</td><td>147,840</td></tr><tr><td>AdaptiveAvgPool + FC(128→256)</td><td>256</td><td>33,024</td></tr><tr><td>CORAL Ordinal Head</td><td></td><td></td></tr><tr><td>FC(256→128) + ReLU + Drop(0.3)</td><td>128</td><td>32,896</td></tr><tr><td>FC(128→15)</td><td>15</td><td>1,935</td></tr><tr><td>Total / GFLOPs / Memory</td><td colspan="2">441K /  2.37  /  47MB</td></tr></table>

## C. CORAL Ordinal Head and Loss

For D ordered classes, $D - 1$ binary sub-tasks determine threshold exceedance. From $\mathbf { z } ,$ the head computes logits via a two-layer classifier with dropout:

$$
\mathbf { o } = \mathbf { W } _ { 2 } \cdot \mathrm { D r o p o u t } \big ( \mathrm { R e L U } ( \mathbf { W } _ { 1 } \mathbf { z } + \mathbf { b } _ { 1 } ) \big ) + \mathbf { b } _ { 2 } \in \mathbb { R } ^ { D - 1 } .\tag{4}
$$

Cumulative exceedance probabilities $P ( y > k \mid x ) = \sigma ( o _ { k } )$ yield the expected day:

$$
\hat { y } = 1 + \sum _ { k = 1 } ^ { D - 1 } \sigma ( o _ { k } ) .\tag{5}
$$

Class probabilities are recoverable via $P ( y = k \mid x ) =$ $P ( y { > } k { - } 1 ) { - } P ( y { > } k )$ with boundary conditions $P ( y > 0 ) = 1$ $P ( y > D ) = 0$

We adopt CORAL over a generic threshold-based model for two reasons: its shared-weight cumulative-logit construction guarantees rank-monotone thresholds by design, aligning with the strictly ordered storage-day labels and complementing the monotonicity regularizer of Eq. (7); and decomposing the D-way problem into $D { - } 1$ binary sub-tasks keeps the head lightweight and well-conditioned under few labelled days, whereas unconstrained thresholds overfit and can violate rank consistency in low-data regimes.

Binary cross-entropy over all thresholds forms the ordinal loss:

$$
\mathcal { L } _ { \mathrm { o r d } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \sum _ { k = 1 } ^ { D - 1 } \mathrm { B C E } \big ( \mathbb { I } [ y _ { n } > k ] , \sigma ( o _ { k } ^ { ( n ) } ) \big ) ,\tag{6}
$$

where targets $t _ { n , k } = \mathbb { I } [ y _ { n } > k ]$ encode the ordinal structure $( \mathrm { e . g . , } y \mathrm { = 5 }$ gives $[ 1 , 1 , 1 , 1 , 0 , \ldots , 0 ] )$ . Per-episode loss averages over support and query: $\mathcal { L } _ { \mathrm { e p i s o d e } } = \overset { \cdot } { \underset { \mathrm { 2 } } { \mathrm { 1 } } } ( \mathcal { L } _ { \mathrm { o r d } } ^ { S } + \mathcal { L } _ { \mathrm { o r d } } ^ { Q } )$

## D. Temporal Regularization

Monotonicity. For consecutive temporal pairs $\mathcal { P }$ sorted by ground-truth day across the combined episode:

$$
\mathcal { L } _ { \mathrm { m o n o } } = \frac { 1 } { | \mathcal { P } | } \sum _ { ( t , t + 1 ) \in \mathcal { P } } \operatorname* { m a x } \big ( 0 , ~ \delta - \big ( \hat { y } _ { t + 1 } - \hat { y } _ { t } \big ) \big ) ,\tag{7}
$$

with margin $\delta { = } 0 . 0 1$ , enforcing biologically plausible ascending predictions.

Embedding smoothness. Abrupt representational jumps between adjacent days are penalized on the d-dimensional embeddings:

$$
\mathcal { L } _ { \mathrm { s m o o t h } } = \frac { 1 } { | \mathcal { P } | } \sum _ { ( t , t + 1 ) \in \mathcal { P } } \left\| \mathbf { z } _ { t + 1 } - \mathbf { z } _ { t } \right\| _ { 2 } ^ { 2 } .\tag{8}
$$

While $\scriptstyle { \mathcal { L } } _ { \mathrm { m o n o } }$ constrains the output space, $\mathcal { L } _ { \mathrm { { s m o o t h } } }$ shapes the representation space, forming a complementary pairing.

![](images/e92fb69fd413300a1e9e053ae90a4e3541a60e040878e371e917ebfa76f7c727.jpg)  
Fig. 1: Overall framework of the proposed episodic ordinal regression model. Support and query days share the same encoder g and ordina head h with identical weights, differing only in their role in the episodic objective (support days anchor the regularizers; query days provid held-out supervision).

Algorithm 1 Few-Shot Ordinal Meta-Training   
Require: Train fillets ${ \mathcal { F } } _ { \mathrm { t r a i n } } ,$ val episodes ${ \mathcal { E } } _ { \mathrm { v a l } } ,$ support size k   
Ensure: Trained parameters $\theta ^ { * }$   
1: Initialize θ; best mae $ \infty$   
2: for epoch = 1 to E do   
3: for episode = 1 to M do   
4: Sample fillet $f \sim \mathcal { F } _ { \mathrm { t r a i n } }$ (with $\ge N _ { \operatorname* { m i n } }$ days)   
5: Split into S (k days) and Q (remaining)   
6: Compute $\mathcal { L } _ { \mathrm { t o t a l } }$ via Eqs. (6)–(9)   
7: Update θ via Adam   
8: end for   
9: Evaluate on $\mathcal { E } _ { \mathrm { v a l } } ;$ save θ if MAE improves   
10: end for   
11: return $\theta ^ { * }$

## E. Training Objective and Inference

The per-episode loss combines all terms:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = { \mathcal { L } } _ { \mathrm { e p i s o d e } } + \lambda _ { \mathrm { m o n o } } { \mathcal { L } } _ { \mathrm { m o n o } } + \lambda _ { \mathrm { s m o o t h } } { \mathcal { L } } _ { \mathrm { s m o o t h } } ,\tag{9}
$$

with $\lambda _ { \mathrm { m o n o } } { = } \lambda _ { \mathrm { s m o o t h } } { = } 0 . 1$ . Adam optimization $( \mathrm { l r { = } 3 } { \times } 1 0 ^ { - 4 }$ weight decay $5 \times 1 0 ^ { - 4 } )$ runs for 40 epochs of 60 episodes. Network parameters are initialized randomly using the standard He scheme for convolutional and linear layers, with no transfer learning or auxiliary pre-training, so results reflect training from scratch on the target HSI data alone.

Fig. 1 and Algorithm 1 describe the overall architecture of the proposed framework.

## III. RESULTS AND DISCUSSION

## A. Dataset and Experimental Protocol

We evaluate on our custom-made salmon freshness HSI dataset of 50 individually packaged fillets imaged daily over

TABLE II: Pack-Level Data Split (No Leakage Across Partitions)
<table><tr><td>Partition</td><td>Pack IDs</td><td>Packs</td><td>Cubes</td></tr><tr><td>Training</td><td>1-30</td><td>30</td><td>480</td></tr><tr><td>Validation</td><td>31-40</td><td>10</td><td>160</td></tr><tr><td>Test</td><td>41-50</td><td>10</td><td>160</td></tr><tr><td>Total</td><td>1-50</td><td>50</td><td>800</td></tr></table>

TABLE III: Training Hyperparameters
<table><tr><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td></tr><tr><td>Optimizer</td><td>Adam</td><td>λmono</td><td>0.1</td></tr><tr><td>Learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>λsmooth</td><td>0.1</td></tr><tr><td>Weight decay</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td>Margin δ</td><td>0.01</td></tr><tr><td>Epochs</td><td>40</td><td>Embedding d</td><td>256</td></tr><tr><td>Episodes/epoch</td><td>60</td><td>Classes D</td><td>16</td></tr><tr><td>Support k</td><td>3</td><td>Dropout</td><td>0.3</td></tr><tr><td>Min. days  $N _ { \mathrm { m i n } }$ </td><td>6</td><td></td><td></td></tr></table>

16 days (D=16; day 6 = labelled expiry). Each cube spans 462 spectral bands (386.88–1003.6 nm) at 128×128 resolution; after band-wise z-score normalization and noisy edge-band removal, B=256 channels are retained. HSI-to-RGB composites (Fig. 2) confirm that predictions typically fall within ±2 days.

Splitting is performed at the pack level to prevent data leakage (Table II). The 30 training fillets ensure broad interfillet variability for robust episodic meta-learning, while 10 validation and 10 test fillets give low-variance metric estimates without leaking pack-specific cues.

Validation and test episodes use fixed seeds for reproducibil ity, considering only fillets with $\geq 6$ available days. Each episode randomly selects k=3 support days and assigns the remainder as queries. Performance is reported as MAE, ±1- day, and ±2-day accuracy. Table III lists all hyperparameters.

![](images/2dc6d56931c11e03353b1e44d89ba7c2a9ab0da8ebdca131b168ed00b924ff50.jpg)  
Fig. 2: Sample-level predictions on unseen test fillets with HSI→RGB composites. Each panel shows ground-truth day (GT), predicted day (Pred), and absolute error (|e|); most fall within ±2 days.

TABLE IV: Comparison with Regression and Distribution-Based Methods with our approach
<table><tr><td>Method</td><td>MAE↓</td><td>±1 Acc. ↑</td><td>±2 Acc. ↑</td></tr><tr><td>Regression Baselines</td><td></td><td></td><td></td></tr><tr><td>Linear Reg. (HSI feat.)</td><td>2.87</td><td>21.4%</td><td>39.2%</td></tr><tr><td>CNN + Li Regression</td><td>2.21</td><td>30.8%</td><td>52.3%</td></tr><tr><td>Few-Shot CNN Reg.</td><td>1.95</td><td>34.6%</td><td>56.9%</td></tr><tr><td colspan="4">Soft-Label / Distribution Baselines</td></tr><tr><td>Gaussian Label Smooth.</td><td>2.04</td><td>31.5%</td><td>58.5%</td></tr><tr><td>Label Dist. Learning</td><td>1.86</td><td>33.1%</td><td>62.3%</td></tr><tr><td>LDL + Temporal Smooth.</td><td>1.79</td><td>35.4%</td><td>64.6%</td></tr><tr><td>Proposed</td><td>1.58</td><td>42.3%</td><td>72.3%</td></tr></table>

TABLE V: Impact of Ordinal Modelling over multiple heads
<table><tr><td>Head</td><td>MAE↓</td><td>±1 Acc. ↑</td><td>±2 Acc. ↑</td></tr><tr><td>Scalar Regression</td><td>1.95</td><td>34.6%</td><td>56.9%</td></tr><tr><td>Ordinal (no reg.)</td><td>1.73</td><td>38.5%</td><td>66.2%</td></tr><tr><td>Ordinal + Reg.</td><td>1.58</td><td>42.3%</td><td>72.3%</td></tr></table>

## B. Comparison with Regression and Distribution Methods

Table IV benchmarks our approach against regression and distribution-based formulations on unseen test fillets. Scalar regression struggles with inter-fillet variability. Label distribution learning [15] captures day-label uncertainty but lacks explicit ordinal encoding [16]. Our method reduces MAE by 19% over few-shot regression and boosts ±2-day accuracy by 15.4 percentage points.

## C. Effect of Ordinal Modelling and Support Size

Table V isolates the benefit of ordinal learning. Switching from scalar to ordinal prediction drops MAE from 1.95 to 1.73; adding regularization further reduces it to 1.58. Table VI

highlights robustness across support sizes (k): k=3, and even 1-shot achieves 48.5% ±2-day accuracy.

TABLE VI: Robustness Across Few-Shot Support Sizes
<table><tr><td>k</td><td>MAE↓</td><td>±1 Acc. ↑</td><td>±2 Acc. ↑</td></tr><tr><td>1</td><td>2.34</td><td>26.2%</td><td>48.5%</td></tr><tr><td>2</td><td>1.92</td><td>34.1%</td><td>61.5%</td></tr><tr><td>3 (ours)</td><td>1.58</td><td>42.3%</td><td>72.3%</td></tr><tr><td>5</td><td>1.63</td><td>42.6%</td><td>72.1%</td></tr></table>

## D. Ablation Study

Table VII reports component contributions under a reduced 15-epoch budget, adopted to hold optimization cost fixed across A2–A5 so per-component differences are comparable without convergence confounds.

TABLE VII: Ablation Study on multiple configurations
<table><tr><td>Configuration</td><td>MAE↓</td><td>±1 Acc. ↑</td><td>±2 Acc. ↑</td></tr><tr><td>A2: Ordinal Only</td><td>2.29</td><td>29.2%</td><td>56.6%</td></tr><tr><td>A3: + Emb. Smooth.</td><td>2.22</td><td>28.9%</td><td>56.8%</td></tr><tr><td>A4: + Monotonicity</td><td>2.01</td><td>31.1%</td><td>59.2%</td></tr><tr><td>A5: + Both</td><td>2.08</td><td>29.1%</td><td>57.2%</td></tr></table>

Absolute MAE is therefore higher than in Tables IV and V and only relative ordering matters here. Monotonicity (A4) yields the largest single gain (about 12% over the ordinalonly A2), matching the monotone progression of degradation, while smoothness alone (A3) helps only marginally. Under this short budget, A5 (2.08) does not improve over A4 (2.01), which we attribute to smoothness needing more episodes to stabilize the representation space; this is consistent with the full-budget Table V, where the complete regularized model attains the best MAE (1.58). The ablation should thus be read as a relative-ranking diagnostic, not the final operating point, visually mentioned in Fig.3.

![](images/4aa2b6804a94f1fef399cb3e36e8bfc9ad31cdc523732c1e2e1a715c25e350b7.jpg)  
Fig. 3: Comparison of freshness estimation methods across MAE and tolerance-based accuracy metrics on unseen test fillets.

## E. Qualitative Analysis

Fig. 4 illustrates our method’s superiority across all metrics, with the sharp upward trend in ±2-day accuracy confirming the value of dual regularization. Mapping predictions against ground-truth days shows tight adherence to the ascending trajectory, with larger errors confined to mid-range days (5–9) where biochemical changes are subtlest.

![](images/b6601e71c405b36a37d57a6e3e2170dca262095497e75c34e2f2197dfc88c6b4.jpg)  
Fig. 4: Ground truth versus predicted freshness day across the test set. Predictions closely follow the ascending trajectory.

## IV. CONCLUSION

This paper proposes a few-shot ordinal learning paradigm for estimating day-wise freshness from hyperspectral images. The approach combines episodic training, CORAL-based ordinal regression, and temporal regularization to predict freshness across previously unseen fish fillets with only three labelled storage days per task. Experiments demonstrate consistent and significant improvements over label distribution learning and conventional regression under a rigorous unseen-fillet evaluation. A limitation is that the salmon HSI dataset is proprietary and not yet publicly available, which constrains reproducibility; the framework itself is dataset-agnostic, and we plan to validate it on public HSI food-quality benchmarks and release the code in future work. The method offers a label-efficient and configurable approach for scalable, nondestructive food quality assessment for industry.

## REFERENCES

[1] Y. Xiao et al., “Deep learning–based regression of food quality attributes using near-infrared spectroscopy and hyperspectral imaging: A review,” Food Chem., vol. 493, pp. 145932, 2025.

[2] C. Yang et al., “Hyperspectral imaging and deep learning for quality and safety inspection of fruits and vegetables: A review,” J. Agric. Food Chem., vol. 73, no. 17, pp. 10019–10035, 2025.

[3] F. Shahrzad, Z. Arabi, S. Ghafari, and A. Sheikh-Akbari, “Fish quality assessment using hyperspectral imaging and computer vision: A review,” IEEE Sensors J., vol. 25, no. 14, pp. 26255–26268, 2025.

[4] D. Hong et al., “SpectralGPT: Spectral remote sensing foundation model,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 8, pp. 5227–5244, 2024.

[5] B. Xi et al., “Few-shot learning with class-covariance metric for hyperspectral image classification,” IEEE Trans. Image Process., vol. 31, pp. 5079–5092, 2022.

[6] Y. Song, T. Wang, P. Cai, S. K. Mondal, and J. P. Sahoo, “A comprehensive survey of few-shot learning: Evolution, applications, challenges, and opportunities,” ACM Comput. Surv., vol. 55, no. 13s, pp. 1–40, 2023.

[7] W. Zeng and Z.-Y. Xiao, “Few-shot learning based on deep learning: A survey,” Math. Biosci. Eng., vol. 21, no. 1, pp. 679–711, 2024.

[8] Z. Li et al., “Deep cross-domain few-shot learning for hyperspectral image classification,” IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 5501618, 2022.

[9] J. Bai et al., “Few-shot hyperspectral image classification based on adaptive subspaces and feature transformation,” IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 5518517, 2022.

[10] J. Yang et al., “A survey of few-shot learning in smart agriculture: Developments, applications, and challenges,” Plant Methods, vol. 18, no. 1, pp. 28, 2022.

[11] G. Polat, I. Ergenc, H. T. Ozlu, and A. A. Alatan, “Class-distributionaware calibration for long-tailed visual recognition via ordinal regression,” in Proc. IEEE/CVF CVPR Workshops, 2022, pp. 2873–2882.

[12] J. Wang et al., “Ord2Seq: Regarding ordinal regression as label sequence prediction,” in Proc. IEEE/CVF ICCV, 2023, pp. 11161–11171.

[13] W. Cao, V. Mirjalili, and S. Raschka, “Rank consistent ordinal regression for neural networks with application to age estimation,” Pattern Recognit. Lett., vol. 140, pp. 325–331, 2020.

[14] V. Vargas, P. A. Gutierrez, and C. Herv ´ as-Mart ´ ´ınez, “Unimodal regularisation based on beta distribution for deep ordinal regression,” Pattern Recognit., vol. 122, pp. 108310, 2022.

[15] T. Wen, B. Yang, J. Wang, and G. Chen, “Label distribution learning by exploiting label distribution manifold,” IEEE Trans. Neural Netw. Learn. Syst., vol. 34, no. 2, pp. 839–852, 2023.

[16] J. Cheng, et al., “A neural network approach to ordinal regression,” in Proc. IEEE Int. Joint Conf. Neural Netw. (IJCNN), 2022, pp. 1–8.