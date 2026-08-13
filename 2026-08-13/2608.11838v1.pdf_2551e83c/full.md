# GeoBridge: Decoupled Semantic Conditioning for Generative Image Geolocalization

Zhiyang Dou<sup>1,⋆</sup> Xumeng Han<sup>2,⋆</sup> Fengde Peng<sup>3</sup> Zipeng Wang<sup>1</sup> Moxuan Zhao<sup>2</sup> Zhipei Huang<sup>2</sup> Zhenjun Han<sup>2,†</sup>

University of Chinese Academy of Sciences

<sup>1</sup>School of Advanced Interdisciplinary Sciences, <sup>2</sup>School of Electronic, Electrical and Communication Engineering <sup>3</sup>Shenzhen Institutes of Advanced Technology

## Abstract

Multimodal large language models (MLLMs) have advanced image geolocalization mainly by improving how they reason about geographic cues. How that reasoning is decoded into coordinates, however, has lagged behind. Predicting a place name for a geocoding API is discrete and lossy: it ignores image evidence and collapses multigranular semantics into a coarse lookup. We argue that the bottleneck has shifted from what a model reasons to how that reasoning is represented for a continuous, geometryaware decoder. We present GeoBridge, a role-decoupled conditioning mechanism that connects a frozen semantic MLLM to a frozen Riemannian flow-matching head that generates coordinates on the sphere. The central obstacle is a role conflict: supervising the condition with discrete semantic labels biases its representation toward classdiscriminative geometry, at odds with the smooth manifold the generative head requires. GeoBridge keeps the semantic supervision decoupled from the condition interface: a separate projection forms the continuous condition the frozen head expects, injecting geographic priors without disturbing the spherical decoder. On IM2GPS3K, GeoBridge reaches 38.67/52.89/70.37 at the 25/200/750 km thresholds, improving over a place-name-to-API pipeline and reasoning-augmented direct prediction at these precisionrelevant scales. GeoBridge is a decode-side algorithmic contribution, orthogonal and complementary to chain-ofthought reasoning. Code will be made publicly available.

## 1. Introduction

Worldwide image geolocalization asks a model to predict where a photograph was taken on the Earth [11, 33, 35, 38]. The evidence in a single image varies sharply: a visible street sign may pin down a city block, while a rural road or coastline supports only a broad regional hypothesis [1]. Recent multimodal large language model (MLLM) geolocation systems [7, 14, 18, 19, 36, 39] have advanced this task mainly by improving reasoning—reading text, architecture, road layout, and vegetation to infer a plausible place. Far less attention has gone to the step that turns those semantics into an actual coordinate.

![](images/455f8d02aaff0d8fe39d395c868fbc55106b7f24d29fc659809441815d835482.jpg)  
Figure 1. From discrete place names to continuous spherical decoding. Top: prior semantic geolocation pipelines stop too early. An MLLM reasons over visual cues, but its output is a discrete place name handed to a blind geocoding API. Bottom: GeoBridge instead decodes the MLLM’s multi-granular hidden semantics through a lightweight semantic connector into a frozen spherical flow-matching head, producing a fine-grained coordinate.

This decode step is where current pipelines stop too early (Fig. 1). A common route predicts a place name and calls a geocoding API [3, 18, 19], reducing a fine-grained spatial decision to a discrete database lookup that discards the image evidence and any sub-region detail. Coordinates, however, live on a continuous sphere, where a geometry-aware generative decoder—such as a Riemannian flow-matching (RFM) head [8, 21]—can place predictions anywhere in continuous space. We argue that the bottleneck has shifted from what an MLLM reasons to how that reasoning is represented for such a decoder.

We therefore study the interface itself: a frozen MLLM that supplies multi-granular geographic semantics, and a frozen RFM head that generates coordinates on the sphere. The central difficulty is a role conflict. Discrete semantic labels are valuable priors, but supervising the condition representation directly with them pulls it toward classdiscriminative geometry, while the frozen flow head expects a smooth condition compatible with its coordinate manifold.

GeoBridge resolves this conflict through role-decoupled conditioning. We introduce five learned role tokens— country, region, city, latitude, and longitude—grouped into contextual and spatial sets, and apply lightweight semantic supervision to shape them. A separate projection pools the two groups and maps their concatenation to the single condition vector the frozen head expects, so the semantic supervision is decoupled from the final condition interface.

This makes GeoBridge a decode-side contribution that complements rather than competes with reasoning methods: better reasoning improves the semantic inputs, while Geo-Bridge improves how those semantics become a coordinate. Decoding through a frozen, geometry-aware head also clarifies where accuracy is won. Given an oracle semantic condition, the same frozen head localizes far more precisely, so the limiting factor is the condition supplied to the head, not the head itself. And because the decode is image-grounded and continuous rather than a discrete name lookup, an incorrect place name no longer forces a jump to the wrong city center: GeoBridge stays modestly closer to the truth than discrete geocoding even when the predicted semantics are imperfect. Our contributions are:

• We identify a role conflict in MLLM-conditioned generative geolocalization—discrete semantic supervision corrupts the continuous condition a frozen flow head needs—and propose role-decoupled conditioning to resolve it.

• We instantiate it as GeoBridge: five role tokens with semantic supervision and contextual/spatial pooling, driving a frozen RFM spherical head through a trainable connector.

• On IM2GPS3K [35], GeoBridge improves over place-name geocoding (GLOBE [19]) and reasoningaugmented prediction (GRE [36]) at the 25/200/750 km thresholds, and remains competitive on an in-domain MP16 test.

![](images/9aa6c6b055c243f78c1bdb351c246b49c68054544498a1e7fb6a68e82ca7523a.jpg)  
Figure 2. Three modes of MLLM-based geolocation. (a) RAG: the MLLM is augmented with retrieved geotagged neighbors and predicts coordinates as text. (b) CoT: the MLLM reasons over visual cues and emits a chain of thought together with coordinates. (c) GeoBridge: instead of emitting coordinates as text, we read the MLLM’s hidden states as a condition for a frozen flow-matching head that generates coordinates, decoupling semantic reasoning from continuous coordinate decoding.

## 2. Related Work

Worldwide image geolocalization. Classic worldwide geolocalization estimates a coordinate by retrieving visually similar geotagged images [34, 40, 41, 43, 44] or by partitioning the Earth into geographic cells [10, 25, 31, 38]. Retrieval can preserve coordinate-level outputs when the reference set contains a similar view, whereas classification scales well but inherits discretization and boundary artifacts. Subsequent methods strengthen the visual side of this formulation with stronger encoders, geographic hierarchies, semantic segmentation, and scene context [6, 26]. More recently, generative methods such as LocDiff [37] and PLONK [8] move beyond fixed grids and galleries by modeling geolocation as conditional coordinate generation. In parallel, MLLM-based methods use foundation models to infer location from explicit geographic cues, motivating a separate line of work discussed next.

MLLM-based geolocation. Multimodal LLMs exploit explicit semantic cues—text, landmarks, road infrastructure, and regional visual style—and have produced two dominant strategies (Fig. 2a,b). Retrieval-augmented methods condition the MLLM on geotagged neighbors:

Img2Loc [42] and G3 [12] reframe geolocation as trainingfree generation over CLIP-retrieved candidates, Geo-Ranker [13] introduces distance-aware ranking of retrieved candidates, and GeoToken [9] further combines geoaligned retrieval with autoregressive S2-token decoding. Reasoning-based methods instead strengthen intrinsic inference: GeoReasoner [18] fine-tunes on human geo-game reasoning traces, GaGA [7] introduces dynamic multiround interaction into the geolocation task, GLOBE [19] and GRE [36] optimize visual-cue chains of thought, and GeoAgent [14] aligns geolocation reasoning with geographic characteristics through expert-annotated CoT data and geo-aware reinforcement rewards. GeoBridge pursues a third mode (Fig. 2c): it reads the MLLM’s hidden states as a continuous condition for a frozen Riemannian flow head on $\mathbb { S } ^ { 2 }$ , separating semantic reasoning from coordinate decoding without retrieving candidates or predicting discrete location tokens.

Learned interfaces between foundation models and decoders. Learnable query tokens and connector modules are widely used to bridge pretrained foundation-model representations and task-specific decoders. BLIP-2 [17] uses query tokens to connect frozen visual encoders with language models, while MetaQuery [24] studies transferable query-based interfaces across modalities. Similar interface designs have also been used to adapt foundationmodel hidden states to dense prediction tasks such as detection [4, 15, 20, 22] and segmentation [16, 28–30, 45]. GeoBridge follows this general connector paradigm, but the interface is especially delicate in our setting because the decoder is a continuous spherical flow head: the condition is not merely a class token or a natural-language prompt, but a latent that modulates a vector field on $S ^ { 2 }$

## 3. Method

We present GeoBridge, a role-decoupled conditioning mechanism that connects a frozen semantic MLLM to a frozen spherical generative geolocation head (Fig. 3). Geo-Bridge is not a new coordinate generator: the RFM head is kept fixed. The contribution is the trainable interface that turns MLLM hidden states into the single continuous condition that head expects, and the way it injects semantic supervision without corrupting that condition.

## 3.1. Problem Formulation

Given an image x, the goal is to predict a coordinate $\hat { \mathbf { y } } \in \mathbb { S } ^ { 2 }$ scored by geodesic distance $d _ { \mathrm { g e o } } ( \hat { \mathbf { y } } , \mathbf { y } )$ to the ground truth. We view the system as a modular conditional generator:

$$
\hat { \mathbf { y } } \sim g _ { \phi } ( \cdot \mid \mathbf { c } ) , \qquad \mathbf { c } = h _ { \theta } \big ( f _ { \mathrm { M L L M } } ( \mathbf { x } ) \big ) ,\tag{1}
$$

where $f _ { \mathrm { M L L M } }$ is a frozen MLLM, $g _ { \phi }$ is a frozen RFM head pretrained to consume a single condition token, and $h _ { \theta }$ is the only trainable bridge. The bridge must satisfy two requirements at once: expose semantically meaningful geographic information from the MLLM, and preserve the continuous condition geometry the frozen spherical decoder was trained on.

## 3.2. Role-Decoupled Conditioning

Role conflict. Under naive supervision, these two requirements collide. Country and city labels are discrete and class-discriminative, encouraging decision boundaries between administrative categories; the RFM condition, by contrast, must vary smoothly with the coordinate posterior on $\mathbb { S } ^ { 2 }$ . If a single vector must be both a country/city classifier state and the final flow condition, cross-entropy gradients improve semantic separability while distorting the geometry $g _ { \phi }$ consumes. GeoBridge avoids this by decoupling the two roles across three components: role-structured tokens, role-specific semantic heads, and a projection buffer that restores the single-token condition the frozen head expects.

Role tokens. GeoBridge appends a total of five learned role tokens $\mathcal { R } = \mathcal { R } _ { \mathrm { a d m } } \cup \mathcal { R } _ { \mathrm { c o o r d } }$ after the image and geographic prompt, where

$$
\mathcal { R } _ { \mathrm { a d m } } = \{ r _ { \mathrm { c t y } } , r _ { \mathrm { r e g } } , r _ { \mathrm { c i t y } } \} , \quad \mathcal { R } _ { \mathrm { c o o r d } } = \{ r _ { \mathrm { l a t } } , r _ { \mathrm { l o n } } \} ,\tag{2}
$$

for country, region, city, latitude, and longitude, with lastlayer hidden states $\dot { \mathbf { H } } \in \mathbb { R } ^ { 5 \times d _ { \mathrm { m } } }$ The first three are contextual roles (administrative/semantic prior); the last two are spatial roles, giving coordinate-oriented information a place to enter the condition without sharing a token with country/city supervision. This split is purely architectural and does not require the MLLM to emit coordinates through these tokens. The MLLM is frozen; only the role-token embeddings and connector-side modules are trained.

Role-specific semantic heads. To inject multi-level geographic priors, the representations of the administrative roles $\mathcal { R } _ { \mathrm { a d m } }$ are regularized via auxiliary semantic classification tasks. Formally, the semantic prediction $\ell _ { r }$ for a targeted role $r \in \mathcal { R } _ { \mathrm { a d m } }$ is computed as:

$$
\ell _ { r } = A _ { r } \mathrm { N o r m } _ { r } ( \mathbf { h } _ { r } ) + b _ { r } ,\tag{3}
$$

where $\mathbf { h } _ { r }$ denotes the corresponding hidden state. These auxiliary heads are optimized via cross-entropy losses against the ground-truth administrative labels. The overall geographic semantic loss is formulated as a joint optimization across the administrative granularities:

$$
\mathcal { L } _ { \mathrm { s e m } } = \sum _ { r \in \mathcal { R } _ { \mathrm { a d m } } } \lambda _ { r } \mathcal { L } _ { r } ^ { \mathrm { C E } } ,\tag{4}
$$

![](images/3c74b7de825b28bc628cfb46e6c4f3b7707c7cc836617db24429865b15e9523a.jpg)  
Figure 3. GeoBridge architecture. (1) A frozen MLLM encodes the image and a geographic instruction. (2) Five learned role tokens— three contextual (country, region, city) and two spatial (latitude, longitude)—are read from the MLLM and passed through a lightweight connector. Their contextual and spatial groups are pooled separately and projected into a single condition c, preserving the frozen head’s single-token contract. (3) A frozen Riemannian flow-matching head consumes c and transports a random point to a coordinate on the sphere $\mathbb { S } ^ { 2 }$ . (4) The sampled point is the predicted GPS location.

where $\lambda _ { r }$ is the balancing coefficient for role r. Crucially, these auxiliary heads serve solely as regularizers to steer the hidden states toward interpretable geographic concepts, while the downstream projection buffer aggregates the refined representations for the final prediction.

Projection buffer and the single-token contract. A lightweight bidirectional transformer encoder $E _ { \theta } - \mathbf { a }$ single Qwen2-style block—lets the role tokens exchange information, $\bar { \mathbf { H } } = { E _ { \theta } } ( { W _ { \mathrm { i n } } } \mathbf { H } ) \in { \mathbb { R } } ^ { 5 \times d _ { \mathrm { h } } }$ . We then pool the contextual and spatial groups separately,

$$
\begin{array} { r l } & { { \bf c } _ { \mathrm { c t x } } = \frac { 1 } { 3 } \big ( \bar { \bf H } _ { \mathrm { c t y } } + \bar { \bf H } _ { \mathrm { r e g } } + \bar { \bf H } _ { \mathrm { c i t y } } \big ) , } \\ & { { \bf c } _ { \mathrm { s p a } } = \frac { 1 } { 2 } \big ( \bar { \bf H } _ { \mathrm { l a t } } + \bar { \bf H } _ { \mathrm { l o n } } \big ) , } \end{array}\tag{5}
$$

and map their concatenation to a single condition token,

$$
\mathbf { c } = \mathrm { N o r m } ( \mathrm { G E L U } ( W _ { \mathrm { o u t } } [ \mathbf { c } _ { \mathrm { c t x } } ; \mathbf { c } _ { \mathrm { s p a } } ] ) ) \in \mathbb { R } ^ { 1 \times 1 0 2 4 } .\tag{6}
$$

The frozen head was pretrained under a single-token condition contract; feeding it a sequence of role tokens would change its input distribution and force the frozen decoder onto an interface it never saw. Eq. (6) keeps role structure upstream while preserving that contract exactly. Because c still derives from the supervised encodings, supervision and conditioning are decoupled at the head interface rather than made fully independent—a distinction we test directly in our ablations.

## 3.3. Training Objective

The frozen head defines a Riemannian flow-matching objective on the sphere [5]. For an intermediate state $z _ { t } \in \mathbb { S } ^ { 2 }$ at flow time t with target tangent velocity $u _ { t } .$ , the head predicts $v _ { \phi } ( z _ { t } , t , \mathbf { c } )$ and

$$
\mathcal { L } _ { \mathrm { R F M } } = \mathbb { E } _ { t , z _ { t } } \Big [ \| v _ { \phi } ( z _ { t } , t , \mathbf { c } ) - u _ { t } \| _ { T _ { z _ { t } } \mathbb { S } ^ { 2 } } ^ { 2 } \Big ] .\tag{7}
$$

With ϕ fixed, Eq. (7) backpropagates only into the connector through c. Since the MLLM forward dominates training cost while the frozen head is cheap, we estimate Eq. (7) with a 1-to-N expansion: for each image we reuse its single condition c across N=8 independent flow samples, reducing gradient variance at no extra MLLM cost (derivation in the appendix). The full objective is

$$
\begin{array} { r } { \mathcal { L } = \widehat { \mathcal { L } } _ { \mathrm { R F M } } ^ { ( N ) } + \mathcal { L } _ { \mathrm { s e m } } , } \end{array}\tag{8}
$$

which tunes only the role-token embeddings, connector encoder, projection buffer, and semantic heads. Keeping both the MLLM and the coordinate generator frozen is deliberate: any gain must come from the conditioning interface, not from retraining the backbone or enlarging the decoder.

## 3.4. Training and Evaluation Policy

The two passes differ only in how the semantic prefix is obtained (Fig. 4). At training, ground-truth semantic text is injected as a teacher-forced prefix before the role tokens are read out, giving the connector a clean condition-learning signal; the semantic heads are evaluated here. At evaluation, no ground truth is available: the frozen MLLM first generates the structured geographic text, then the role tokens are appended and reforwarded—reusing the prefix KV-cache— to produce the condition, with no semantic CE. All main results use this deployable generated-prefix path; the oracle (ground-truth-condition) setting is reported only as an upper bound in analysis.

Coordinate sampling. Given the condition c, the frozen head produces a coordinate by integrating its learned flow on the sphere, following [8]. We draw a base sample $z _ { 0 }$ from the prior on $\mathbb { S } ^ { 2 }$ (uniform over the Earth’s surface) and solve the flow ODE $\dot { z } _ { t } = v _ { \phi } ( z _ { t } , t , \mathbf { c } )$ from $\scriptstyle t = 0 \mathrm { t o } t = 1$ . Each step remains on the manifold through the exponential map,

![](images/cd9b7c5462544d93098dbe4de50a98418d0504db864f074c435b3d720d03a8d7.jpg)  
Figure 4. Training and evaluation policy. During training, ground-truth semantic fields are injected, whereas during evaluation they are unavailable.

$$
z _ { t + \Delta t } = \exp _ { z _ { t } } \bigl ( \Delta t v _ { \phi } ( z _ { t } , t , \mathbf { c } ) \bigr ) , \qquad \Delta t = 1 / K ,\tag{9}
$$

using K=32 Euler steps. We take a single trajectory, so the endpoint is the prediction, $\hat { \mathbf { y } } = z _ { 1 }$

## 4. Experiments

## 4.1. Experimental Setup

Benchmarks. We evaluate on two worldwide image geolocalization benchmarks of web-sourced photographs: IM2GPS3K [35] (2,997 images), the standard cross-domain test, and MP16-Reason-Test [19] (12,000 images), an indomain test drawn from the MP16-Pro distribution. Every method is scored against the same ground-truth split.

Metrics. Distances are computed with the Haversine (great-circle) formula. We report the hit rate at 25, 200, 750, and 2500 km, where @k is the percentage of predictions within k km of the ground truth. The 25/200/750 km thresholds are precision-relevant for city-, region-, and countryscale localization; @2500 is coarse and often saturated among modern MLLM baselines.

Baselines. We compare across geolocation paradigms; locally evaluated rows are starred, and per-method citations are given in the tables. Classification and retrieval methods predict a geographic cell or retrieve geotagged neighbors [23, 26, 27, 32, 38]; embedding-alignment methods match images to a continuous location space [34]; reasoning MLLMs infer a location through explicit visual-cue reasoning [3, 14, 18, 19, 36]; place-name pipelines predict a place name and convert it to coordinates through a geocoding API [3, 18, 19], for which we run the released models under the original conversion (Microsoft Azure Map for [19] and Nominatim for [3]); and generative methods sample coordinates from a flow head [8]. GeoBridge is a decode-side mechanism, not a specialized retrieval/geocell system, and we make no leaderboard claim. We exclude the concurrent system [14, 39], which strengthens upstream geographic reasoning rather than the decode and is therefore orthogonal to ours; a controlled comparison is also impractical (undisclosed training data, API-dependent geocoding).

Implementation. We train GeoBridge for one epoch on a 1M-image subset of MP16-Pro [12] that is disjoint from the evaluation benchmarks, using learning rate 1 × 10<sup>−4</sup> and a cosine schedule. GeoBridge employs GLOBE [19] as the base MLLM, utilizing five role tokens. The states of these tokens are encoded by the connector, then pooled into contextual and spatial groups, concatenated, and finally projected to a single 1024-dimensional condition that is compatible with the frozen Riemannian-flow head [8]. The MLLM backbone and RFM head are frozen; trainable parameters are limited to the new role-token embeddings, the connector modules, and the semantic auxiliary heads. The RFM loss uses 1-to-N condition expansion with $N = 8 ;$ we derive the estimator in Appendix B and sweep N in Appendix C. Further training details are provided in Appendix A.

## 4.2. Main Results

Table 1 compares GeoBridge with standard geolocation references and recent MLLM-based systems on the two benchmarks; locally evaluated rows are starred.

On IM2GPS3K (Table 1a), among the non-retrieval MLLM decoding baselines in Table 1, GeoBridge attains the strongest accuracy at 25, 200, and 750 km, and trails only GRE at the coarse, near-saturated 2500 km band. It improves over both place-name-to-API baselines (GeoReasoner, GLOBE, GAEA) and the reasoning-augmented GRE at fine scale. This supports an algorithmic point rather than a leaderboard one: place-name geocoding discards information by committing to a discrete name and direct-coordinate MLLMs emit a single point through the language head, whereas GeoBridge keeps a generative spherical decoder and forms its condition from MLLM semantics through a role-decoupled interface, improving fine-scale accuracy without modifying the frozen head.

Cross-benchmark divergence. On MP16-Reason-Test (Table 1b) GeoBridge overtakes place-name geocoding (GLOBE) at 750 and 2500 km, and GLOBE leads only at the two finest thresholds, by a small margin. This finethreshold gap reflects how the benchmark defines ground truth, not a localization weakness. MP16-Reason-Test coordinates are city-center-aligned, so once the city is named correctly, geocoding to the city center lands almost exactly on the target: among GLOBE’s predictions, the subset with both country and city correct (n=5815) reaches 98.64/99.57/99.67/99.93 at @25/@200/@750/@2500.

Table 1. Benchmark comparisons. Accuracy (%) at geodesic distance thresholds (km). ⋆ marks rows evaluated locally under our pipeline; for place-name methods we follow the original place-name-to-coordinate API conversion. Best in bold, second underlined.  
(a) IM2GPS3K.
<table><tr><td>Method</td><td>Source</td><td>@25</td><td>@200</td><td>@750</td><td>@2500</td></tr><tr><td>PlaNet [38]</td><td>ECCV&#x27;16</td><td>24.80</td><td>34.30</td><td>48.40</td><td>64.60</td></tr><tr><td>ISNs [23]</td><td>ECCV&#x27;18</td><td>28.00</td><td>36.60</td><td>49.70</td><td>66.00</td></tr><tr><td>Translocator [26]</td><td>ECCV’22</td><td>31.10</td><td>46.70</td><td>58.90</td><td>80.10</td></tr><tr><td>GeoDecoder [6]</td><td>CVPR&#x27;23</td><td>33.50</td><td>45.90</td><td>61.00</td><td>76.10</td></tr><tr><td>GeoCLIP [34]</td><td>NeurIPS&#x27;23</td><td>34.47</td><td>50.65</td><td>69.67</td><td>83.82</td></tr><tr><td>GRE [36]</td><td>NeurIPS&#x27;25</td><td>35.30</td><td>51.70</td><td>69.30</td><td>85.70</td></tr><tr><td>GeoReasoner [18] ICML&#x27;24</td><td></td><td>26.94</td><td>36.63</td><td>52.27</td><td>65.39</td></tr><tr><td>GLOBE* [19]</td><td>NeurIPS&#x27;25</td><td>36.95</td><td>51.99</td><td>69.88</td><td>83.99</td></tr><tr><td>GAEA* [3]</td><td>WACV&#x27;26</td><td>35.40</td><td>51.92</td><td>69.77</td><td>83.95</td></tr><tr><td>GeoBridge*</td><td>This work</td><td>38.67</td><td>52.89</td><td>70.37</td><td>84.42</td></tr></table>

On the same subset, GeoBridge is precise within the city but spreads continuously around the true area rather than snapping to the city-center point the metric treats as ground truth. This is why GLOBE’s edge is confined to 25/200 km, where landing on the center is decisive, and reverses at 750/2500 km, where center-snapping no longer helps. The city-center bonus, however, only applies when the city is named correctly, and that coverage is domain-dependent: GLOBE—also GeoBridge’s semantic backbone—is bothcountry-and-city-correct on 5815/12000 (48%) of MP16- Reason-Test but only 736/2997 (25%) of cross-domain IM2GPS3K. With the discrete bonus covering far fewer images there, GeoBridge’s continuous estimate prevails on the larger remainder and leads at the precision thresholds. Which decode paradigm wins thus depends on how often the discrete city is correct, not on localization quality alone; we report both benchmarks to make this dependence explicit.

## 4.3. Analysis

All analysis in this section is on IM2GPS3K.

The condition, not the head, is the bottleneck. To locate the limiting factor, we replace the deployable condition with a teacher-forced oracle role condition built from ground-truth country/region/city text (Table 2). The oracle row is not a deployable accuracy claim; it isolates whether the frozen head and five-role connector can exploit a correct semantic–spatial condition. Given that condition, @25 rises from 38.67 to 71.67 and the median error falls from 145.77 to 11.67 km—the frozen RFM head is far from saturated. The deployable limit is therefore the quality of the condition supplied by the MLLM, not the capacity of the coordinate head. This validates the premise behind Geo-Bridge: rather than strengthen the coordinate head, we keep a strong frozen decoder and learn a better, image-grounded condition for it.

An image-grounded decoder degrades gracefully. The two decode paradigms fail very differently when the semantics are imperfect, and this is where GeoBridge’s design pays off. A place-name pipeline sees only the discrete predicted name at decode time, so a wrong city name geocodes to the wrong city center—an error as large as the naming mistake. GeoBridge instead conditions on the imagegrounded MLLM state and decodes continuously, so a semantic error is softened by the visual evidence rather than amplified. Grouping IM2GPS3K predictions by the correctness of the generated country/city condition (Table 3), Geo-Bridge attains lower mean error than place-name geocoding in every regime. When both labels are correct its continuous estimate localizes within the city rather than snapping to the center (42.9 vs. 57.3 km); and when the city or even the country is wrong, conditioning on the imagegrounded state keeps it at least as accurate as geocoding (441.5 vs. 463.5 km; 4263.5 vs. 4384.4 km) instead of being dragged to a wrong administrative center. GeoBridge thus never trades away accuracy for conditioning on the MLLM, and Fig. 5 shows representative cases. Only 736 of the 2,997 images fall in the both-correct bucket, so most residual error is upstream—GeoBridge improves and stabilizes the semantics-to-coordinate decode but does not repair the reasoning itself, making it complementary to stronger reasoning or CoT-conditioned backbones that would improve the condition it decodes.

(b) MP16-Reason-Test.
<table><tr><td>Method</td><td>Source</td><td>@25</td><td>@200</td><td>@750</td><td>@2500</td></tr><tr><td>ISNs [23]</td><td>ECCV&#x27;18</td><td>47.38</td><td>55.88</td><td>68.48</td><td>80.92</td></tr><tr><td>GeoCLIP [34]</td><td>NeurIPS&#x27;23</td><td>52.52</td><td>66.85</td><td>84.07</td><td>93.33</td></tr><tr><td>Hybrid [1]</td><td>CVPR&#x27;24</td><td>16.53</td><td>28.72</td><td>50.31</td><td>71.47</td></tr><tr><td>RFM-YFCC [8]</td><td>CVPR&#x27;25</td><td>46.64</td><td>60.46</td><td>77.97</td><td>91.96</td></tr><tr><td>GRE [36]</td><td>NeurIPS&#x27;25</td><td></td><td></td><td></td><td></td></tr><tr><td>GeoReasoner [18]</td><td>ICML&#x27;24</td><td>40.44</td><td>50.91</td><td>68.01</td><td>79.68</td></tr><tr><td>GLOBE* [19]</td><td>NeurIPS’25</td><td>60.95</td><td>74.00</td><td>86.12</td><td>92.98</td></tr><tr><td>GAEA* [3]</td><td>WACV’26</td><td>26.13</td><td>41.48</td><td>65.60</td><td>82.84</td></tr><tr><td>GeoBridge*</td><td>This work</td><td>57.44</td><td>72.39</td><td>87.08</td><td>94.20</td></tr></table>

Table 2. Condition ceiling for the frozen RFM head. The oracle row uses teacher-forced role conditions and is not a deployable result; it measures whether the frozen head and five-role connector can exploit a high-quality condition.
<table><tr><td>Condition</td><td>@25</td><td>@200</td><td>@750</td><td>@2500</td><td>Median</td></tr><tr><td>Deployable</td><td>38.67</td><td>52.89</td><td>70.37</td><td>84.42</td><td>145.77</td></tr><tr><td>Oracle</td><td>71.67</td><td>88.51</td><td>95.50</td><td>99.19</td><td>11.67</td></tr></table>

Fig. 5 contrasts GeoBridge with place-name geocoding on two examples from each bucket. When the condition is correct both land near the ground truth; when only the country is correct, geocoding jumps to a wrong city center while GeoBridge stays within the right region; and when the condition is wrong, geocoding follows the wrong name across continents while GeoBridge’s image-grounded estimate remains comparatively close. It is the visual evidence, not the discrete label, that keeps GeoBridge anchored.

Table 3. Robustness across semantic regimes (IM2GPS3K). Mean Haversine error (km) per bucket, grouped by whether the generated condition names the correct country and city. Geo-Bridge attains lower mean error than place-name geocoding in every bucket, including when the discrete semantics are wrong.
<table><tr><td>Bucket</td><td>Count</td><td>GeoBridge</td><td>Geocoding</td></tr><tr><td>Country + city correct</td><td>736</td><td>42.92</td><td>57.25</td></tr><tr><td>Country correct, city wrong</td><td>1355</td><td>441.49</td><td>463.51</td></tr><tr><td>Country + city wrong</td><td>906</td><td>4263.54</td><td>4384.41</td></tr></table>

Table 4. Conditioning-interface ablation under the oracle condition. The condition path is built up one design element at a time and evaluated on IM2GPS3K with the teacher-forced groundtruth semantic condition, so each variant’s representational ceiling is measured in isolation from upstream semantic-prediction errors. We report Acc@k (%); best per column in bold. The last row is the Oracle entry of Table 2.
<table><tr><td>Conditioning variant</td><td>@25</td><td>@200</td><td>@750</td><td>@2500</td></tr><tr><td>Single-token, MLP</td><td>38.61</td><td>63.86</td><td>84.48</td><td>94.09</td></tr><tr><td>5 tokens, ungrouped, MLP</td><td>37.00</td><td>64.03</td><td>80.85</td><td>92.09</td></tr><tr><td>5 tokens, grouped, MLP</td><td>42.38</td><td>69.87</td><td>84.42</td><td>95.70</td></tr><tr><td>5 tokens, grouped, Q2Enc.</td><td>61.06</td><td>80.07</td><td>93.18</td><td>98.69</td></tr><tr><td>Full GeoBridge (+ role CE)</td><td>71.67</td><td>88.51</td><td>95.50</td><td>99.19</td></tr></table>

## 5. Ablations

Unless otherwise specified, all experiments in this section are conducted on IM2GPS3K.

## 5.1. Component Ablation

Table 4 builds GeoBridge’s conditioning interface one design element at a time. To measure each variant’s representational ceiling rather than its sensitivity to upstream errors, we evaluate under the oracle condition—the teacher-forced ground-truth semantic text of Sec. 4.3—so the final row coincides with the Oracle entry of Table 2. All variants share the same training data, schedule, frozen MLLM backbone, and frozen RFM head; what varies is the condition path: the role-token set, the grouping of those tokens, the connector module, and the auxiliary role cross-entropy.

Structure matters more than token count. Replacing the single condition token with five ungrouped role tokens does not help: @25 even dips (38.6→37.0) and the coarser thresholds fall (84.5→80.9 at @750), so adding latent tokens alone does not improve the condition. Grouping those role tokens into contextual and spatial sets instead yields a clear gain at every threshold (37.0→42.4 at @25, 80.9→84.4 at @750), showing that structured semantic decomposition, not token budget, is the active ingredient.

Table 5. Oracle-condition upper bounds with different coordinate samplers. Both rows use teacher-forced semantic conditions and are not deployable results.
<table><tr><td>Sampler/head</td><td>@25</td><td>@200</td><td>@750</td><td>@2500</td></tr><tr><td>Flow Matching</td><td>66.80</td><td>88.00</td><td>95.50</td><td>98.60</td></tr><tr><td>Riemannian Flow Matching</td><td>71.67</td><td>88.51</td><td>95.50</td><td>99.19</td></tr></table>

Connector capacity and role supervision close the gap. Replacing the MLP connector with Q2Enc, a Qwen2- style transformer-block encoder, lifts the ceiling sharply (42.4→61.1 at @25). Adding the auxiliary role crossentropy—which, decoupled from the condition geometry by GeoBridge’s projection buffer, stabilizes role specialization rather than corrupting it—completes GeoBridge and reaches the oracle ceiling (71.67/88.51/95.50/99.19). Oracle ceilings need not translate into deployable gains, however: Appendix E repeats this ablation under the deployable generated-condition setting, where the connector accounts for most of the realized improvement and the role crossentropy is approximately neutral—consistent with deployable accuracy being gated by the generated semantic prefix.

## 5.2. Coordinate Sampler

The frozen head can be trained with Euclidean flow matching (FM) or Riemannian flow matching (RFM) on the sphere; the two differ only in the sampler construction and the flow loss, with all other settings fixed. We compare them at the oracle condition, which probes the head in isolation from generated-condition quality (Table 5). RFM is at least as strong as FM at every threshold, with the largest gain at @25 (71.67 vs. 66.80) and a tie at @750 (95.50 for both), supporting the geometry-aware decoder choice; we use RFM throughout.

## 5.3. Integration Steps at Inference

GeoBridge samples a coordinate by integrating the frozen flow for K Euler steps (Sec. 3.4). Fig. 6 sweeps K at test time with all trained components fixed. Accuracy rises steeply up to roughly 16 steps and then plateaus across every threshold, so our default K=32 sits just past the plateau and can be reduced further for faster inference at negligible accuracy cost. The condition is computed once per image, so the only added cost is K lightweight head evaluations.

![](images/bdb3298ca8564a00c47a2e3c1ae69bf850d8f16dd06896624716f79acafdb643.jpg)  
Figure 5. Robustness to imperfect semantics (IM2GPS3K). Two examples per bucket; each map shows the ground truth (green star) GeoBridge (blue circle), and place-name geocoding (orange diamond).

![](images/677c5624f3a76a19c9bbd9507c768654643754ce251f9a32818fedc54b817045.jpg)  
Figure 6. Accuracy vs. number of inference steps. All thresholds saturate by roughly 16 Euler steps; our default of 32 is well past the plateau.

## 5.4. Backbone Transfer

To probe how much the backbone shapes the condition, we swap the frozen MLLM under a deliberately minimal interface—a single condition token, RFM loss, and no semantic supervision—keeping everything else fixed (Table 6). These runs predate the five-role design and are therefore not comparable to full GeoBridge; they isolate the backbone’s effect on condition quality. We evaluate here under the deployable condition rather than the oracle one: the three backbones train almost identically and diverge chiefly in the semantic prefix they predict at inference, so only the deployable setting—where each backbone supplies its own generated condition—surfaces that difference, whereas teacher-forcing a shared ground-truth condition would mask it. This mirrors the connector ablation’s use of the oracle condition: each setting fixes the factor it is not testing. Stronger geographic backbones indeed yield better conditions: a supervised-fine-tuned Qwen2.5-VL and GLOBE both improve over the base model across most thresholds. This reinforces the upper-bound finding that deployable accuracy is governed by the semantic condition the backbone supplies, and shows the generative-decoder interface is not tied to a single MLLM.

## 6. Conclusion

We presented GeoBridge, a role-decoupled conditioning scheme for MLLM-guided generative geolocalization. Supervised semantic role tokens and a separate context/spatial projection feed a frozen RFM head without altering its single-token condition contract, resolving the role conflict between discrete semantic supervision and continuous coordinate geometry. On IM2GPS3K, this decode-side mechanism improves fine-scale accuracy over the MLLM-based and place-name baselines we compare. Our analysis locates the remaining gap upstream—in the semantic condition the MLLM supplies, not the coordinate head—pointing to distribution-aligned condition learning and its composition with stronger reasoning as the natural next step.

Table 6. Single-token backbone diagnostics. These runs replace the frozen semantic backbone while keeping a one-token condition interface. They predate the five-role role-decoupled design, so they are diagnostic transfer evidence rather than matched Geo-Bridge ablations.
<table><tr><td>Frozen backbone</td><td>@25</td><td>@200</td><td>@750</td><td>@2500</td></tr><tr><td>Qwen2.5-VL [2]</td><td>21.7</td><td>42.6</td><td>65.3</td><td>82.5</td></tr><tr><td>Qwen2.5-VL SFT</td><td>23.3</td><td>48.7</td><td>67.8</td><td>84.3</td></tr><tr><td>GLOBE</td><td>21.6</td><td>48.9</td><td>67.9</td><td>84.8</td></tr></table>

## Acknowledgements

This work was supported in part by the Key Deployment Program of the Chinese Academy of Sciences, China under Grant KGFZD-145-25-39, the National Natural Science Foundation of China under Grants 62272438, and Beijing Natural Science Foundation L25700.

## References

[1] Guillaume Astruc, Nicolas Dufour, Ioannis Siglidis, Constantin Aronssohn, Nacim Bouia, Stephanie Fu, Romain Loiseau, Van Nguyen Nguyen, Charles Raude, Elliot Vincent, et al. Openstreetview-5m: The many roads to global visual geolocation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21967–21977, 2024. 1, 6

[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 9

[3] Ron Campos, Ashmal Vayani, Parth Parag Kulkarni, Rohit Gupta, Aritra Dutta, and Mubarak Shah. Gaea: A geolocation aware conversational model, 2025. 2, 5, 6

[4] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic, 2023. 3

[5] Ricky TQ Chen and Yaron Lipman. Flow matching on general geometries. In International Conference on Learning Representations, pages 47922–47945, 2024. 4, 13

[6] Brandon Clark, Alec Kerrigan, Parth P. Kulkarni, Vicente Vivanco Cepeda, and Mubarak Shah. Where we are and what we’re looking at: Query based worldwide image geolocalization using hierarchies and scenes. In Proceedings of

the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 2, 6

[7] Zhiyang Dou, Zipeng Wang, Xumeng Han, Guorong Li, Zhipei Huang, and Zhenjun Han. Gaga: Towards interactive global geolocation assistant, 2025. 1, 3

[8] Nicolas Dufour, Vicky Kalogeiton, David Picard, and Loic Landrieu. Around the world in 80 timesteps: A generative approach to global visual geolocation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 23016–23026, 2025. 2, 4, 5, 6

[9] Narges Ghasemi, Amir Ziashahabi, Salman Avestimehr, and Cyrus Shahabi. Geotoken: Hierarchical geolocalization of images via next token prediction. In IEEE International Conference on Data Mining, 2025. 3

[10] Lukas Haas, Michal Skreta, Silas Alberti, and Chelsea Finn. Pigeon: Predicting image geolocations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12893–12902, 2024. 2

[11] James Hays and Alexei A Efros. Im2gps: estimating geographic information from a single image. In CVPR, 2008. 1

[12] Pengyue Jia, Yiding Liu, Xiaopeng Li, Xiangyu Zhao, Yuhao Wang, Yantong Du, Xiao Han, Xuetao Wei, Shuaiqiang Wang, and Dawei Yin. G3: an effective and adaptive framework for worldwide geolocalization using large multimodality models. Advances in Neural Information Process ing Systems, 37:53198–53221, 2024. 3, 5

[13] Pengyue Jia, Seongheon Park, Song Gao, Xiangyu Zhao, and Yixuan Li. Georanker: Distance-aware ranking for worldwide image geolocalization. arXiv preprint arXiv:2505.13731, 2025. 3

[14] Modi Jin, Yiming Zhang, Boyuan Sun, Dingwen Zhang, Ming-Ming Cheng, and Qibin Hou. Geoagent: Learning to geolocate everywhere with reinforced geographic characteristics. arXiv preprint arXiv:2602.12617, 2026. 1, 3, 5

[15] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. Mdetr – modulated detection for end-to-end multi-modal understanding, 2021. 3

[16] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model, 2024. 3

[17] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In In ternational conference on machine learning, pages 19730– 19742. PMLR, 2023. 3

[18] Ling Li, Yu Ye, Bingchuan Jiang, and Wei Zeng. Georeasoner: Geo-localization with reasoning in street views using a large vision-language model, 2024. 1, 2, 3, 5, 6

[19] Ling Li, Yao Zhou, Yuxuan Liang, Fugee Tsung, and Jiaheng Wei. Recognition through reasoning: Reinforcing image geo-localization with large vision-language models. In Advances in Neural Information Processing Systems, 2025. 1, 2, 3, 5, 6

[20] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jian wei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu

Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, and Jianfeng Gao. Grounded language-image pre-training, 2022. 3

[21] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 2

[22] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding dino: Marrying dino with grounded pre-training for open-set object detection, 2024. 3

[23] Eric Muller-Budack, Kader Pustu-Iren, and Ralph Ewerth. Geolocation estimation of photos using a hierarchical model and scene classification. In Proceedings of the European conference on computer vision (ECCV), pages 563–579, 2018. 5, 6

[24] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025. 3

[25] Shraman Pramanick, Ewa M Nowara, Joshua Gleason, Carlos D Castillo, and Rama Chellappa. Where in the World is this Image? Transformer-based Geo-localization in the Wild. In Proceedings ofthe European Conference on Computer Vision, pages 196–215, 2022. 2

[26] Shraman Pramanick, Ewa M. Nowara, Joshua Gleason, Carlos D. Castillo, and Rama Chellappa. Where in the world is this image? transformer-based geo-localization in the wild. In European Conference on Computer Vision, pages 196– 215, 2022. 2, 5, 6

[27] Feng Qi, Mian Dai, Zixian Zheng, and Chao Wang. Geodecoder: Empowering multimodal map understanding. arXiv preprint arXiv:2401.15118, 2024. 5

[28] Rui Qian, Chuanhang Deng, Qiang Huang, Jian Xiong, Mingxuan Li, Yingbo Zhou, Wei Zhai, Jintao Chen, and Dejing Dou. Anchorseg: Language grounded query banks for reasoning segmentation, 2026. 3

[29] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji Mullappilly, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M. Anwer, Erix Xing, Ming-Hsuan Yang, and Fahad S. Khan. Glamm: Pixel grounding large multimodal model, 2024.

[30] Zhongwei Ren, Zhicheng Huang, Yunchao Wei, Yao Zhao, Dongmei Fu, Jiashi Feng, and Xiaojie Jin. Pixellm: Pixel reasoning with large multimodal model, 2024. 3

[31] Paul Hongsuck Seo, Tobias Weyand, Jack Sim, and Bohyung Han. CPlaNet: Enhancing Image Geolocalization by Combinatorial Partitioning of Maps. In Proceedings of the European Conference on Computer Vision, pages 536–551, 2018. 2

[32] Paul Hongsuck Seo, Tobias Weyand, Jack Sim, and Bohyung Han. Cplanet: Enhancing image geolocalization by combinatorial partitioning of maps. In Proceedings of the European Conference on Computer Vision, pages 536–551, 2018. 5

[33] Qiaomu Shen, Wei Zeng, Yu Ye, Stefan Mueller Arisona, Simon Schubiger, Remo Burkhard, and Huamin Qu. StreetVizor: Visual Exploration of Human-Scale Urban Forms Based on Street Views. IEEE Transactions on Visualization and Computer Graphics, 24(1):1004–1013, 2018. 1

[34] Vicente Vivanco Cepeda, Gaurav Kumar Nayak, and Mubarak Shah. Geoclip: Clip-inspired alignment between locations and images for effective worldwide geolocalization. In Advances in Neural Information Processing Systems, pages 8690–8701, 2023. 2, 5, 6

[35] Nam Vo, Nathan Jacobs, and James Hays. Revisiting im2gps in the deep learning era. In Proceedings of the IEEE International Conference on Computer Vision, pages 2621–2630, 2017. 1, 2, 5

[36] Chun Wang, Xiaojun Ye, Xiaoran Pan, Zihao Pan, Haofan Wang, and Yiren Song. Gre suite: Geo-localization inference via fine-tuned vision-language models and enhanced reasoning chains, 2025. 1, 2, 3, 5, 6

[37] Zhangyu Wang, Zeping Liu, Jielu Zhang, Zhongliang Zhou, Qian Cao, Nemin Wu, Lan Mu, Yang Song, Yiqun Xie, Ni Lao, and Gengchen Mai. Locdiff: Identifying locations on earth by diffusing in the hilbert space, 2025. 2

[38] Tobias Weyand, Ilya Kostrikov, and James Philbin. Planet: Photo geolocation with convolutional neural networks. In European Conference on Computer Vision, pages 37–55, 2016. 1, 2, 5, 6

[39] Biao Wu, Meng Fang, Ling Chen, Ke Xu, Tao Cheng, and Jun Wang. Vision-language reasoning for geolocalization: A reinforcement learning approach, 2026. 1, 5

[40] Xiaohan Zhang, Xingyu Li, Waqas Sultani, Yi Zhou, and Safwan Wshah. Cross-view geo-localization via learning disentangled geometric layout correspondence. In Proceed ings of the AAAI Conference on Artificial Intelligence, pages 3480–3488, 2023. 2

[41] Xiaohan Zhang, Waqas Sultani, and Safwan Wshah. Crossview image sequence geo-localization. In WACV, 2023. 2

[42] Zhongliang Zhou, Jielu Zhang, Zihan Guan, Mengxuan Hu, Ni Lao, Lan Mu, Sheng Li, and Gengchen Mai. Img2loc: Revisiting image geolocalization using multi-modality foun dation models and image-based retrieval-augmented generation. In Proceedings of the 47th international acm sigir conference on research and development in information retrieval, pages 2749–2754, 2024. 3

[43] Sijie Zhu, Taojiannan Yang, and Chen Chen. Vigor: Crossview image geo-localization beyond one-to-one retrieval, 2021. 2

[44] Sijie Zhu, Mubarak Shah, and Chen Chen. TransGeo: Transformer Is All You Need for Cross-view Image Geolocalization. In IEEE Conference on Computer Vision and Pattern Recognition, pages 1162–1171, 2022. 2

[45] Xueyan Zou, Zi-Yi Dou, Jianwei Yang, Zhe Gan, Linjie Li, Chunyuan Li, Xiyang Dai, Harkirat Behl, Jianfeng Wang, Lu Yuan, Nanyun Peng, Lijuan Wang, Yong Jae Lee, and Jianfeng Gao. Generalized decoding for pixel, image, and language, 2022. 3

# Supplementary Material

Table 7. Key training hyperparameters for GeoBridge.
<table><tr><td>Item</td><td>Value</td></tr><tr><td>Base MLLM</td><td>GLOBE-Qwen2.5-VL-7B</td></tr><tr><td>Coordinate head</td><td>Frozen PLONK Riemannian-flow head</td></tr><tr><td>Training data</td><td>MP16-Pro-subset-1M</td></tr><tr><td>Evaluation data</td><td>IM2GPS3K</td></tr><tr><td>Epochs</td><td>1</td></tr><tr><td>Batching</td><td>8 GPUs, batch size 4/GPU, accumula- tion 4</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 } .$  cosine decay, warmup 1000 steps</td></tr><tr><td>1-to-N expansion</td><td> $N = 8$  flow samples per image condi- tion</td></tr><tr><td>Role tokens</td><td> $M = 5 \mathrm { : }$  country, region, city, latitude, longitude</td></tr><tr><td>Connector</td><td>1-layer bidirectional connector, hid- den/output dim 1024</td></tr><tr><td>Role grouping</td><td>3 contextual roles + 2 spatial roles</td></tr><tr><td>Losses</td><td>RFM loss; country CE weight 0.05; re- gion CE weight 0.03; city CE weight</td></tr><tr><td>Frozen modules</td><td>0.02 MLLM backbone and PLONK/RFM</td></tr><tr><td>Trainable modules</td><td>head Role-token embeddings, connec-</td></tr><tr><td></td><td>tor/projection, country/city heads Optimization memory DeepSpeed ZeRO-2 + gradient check-</td></tr></table>

## A. Training Details

Table 7 summarizes the key GeoBridge training hyperparameters. The entries are taken from the run configuration, with local filesystem paths omitted for anonymity.

## B. 1-to-N Condition Expansion

The RFM objective contains stochasticity from the sampled base point ϵ and flow time t. Computing the MLLM condition $c _ { i }$ is expensive, while evaluating the frozen RFM head on additional flow samples is comparatively cheap. We therefore use 1-to-N condition expansion: for each image, compute the condition once and reuse it across N independent RFM training samples.

For a mini-batch $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { B }$ , GeoBridge first computes

$$
\boldsymbol c _ { i } = h _ { \boldsymbol \theta } ( f _ { \mathrm { M L L M } } ( \boldsymbol x _ { i } ) ) , \qquad i = 1 , \dots , B .\tag{10}
$$

Then, for each $i ,$ sample N independent pairs $( \epsilon _ { i , n } , t _ { i , n } )$

Algorithm 1 GeoBridge training with 1-to-N expansion   
Require: Image batch $x _ { 1 : B } .$ , targets $y _ { 1 : B }$ , semantic labels,   
expansion factor N   
1: Run the frozen MLLM with trainable role-token em  
beddings and obtain role states $H _ { 1 : B }$   
2: Compute conditions $c _ { 1 : B }$ and country/city logits with   
the connector and auxiliary heads   
3: Compute $\mathcal { L } _ { \mathrm { s e m } }$ once over the B images   
4: Repeat each condition and target N times: $c _ { i , n }  c _ { i } ,$   
$y _ { i , n } \gets y _ { i }$   
5: Sample $\epsilon _ { i , n } \sim \mathrm { U n i f } ( \mathbb { S } ^ { 2 } )$ and $t _ { i , n } \sim \mathcal { U } [ 0 , 1 ]$   
6: Construct $z _ { i , n }$ and target tangent velocity $u _ { i , n }$ using the   
RFM interpolant   
7: Predict $\hat { u } _ { i , n } = v _ { \phi } ( z _ { i , n } , t _ { i , n } , c _ { i , n } )$ with frozen head pa  
rameters   
8: Compute $\begin{array} { r } { \widehat { \mathcal { L } } _ { \mathrm { R F M } } ^ { ( N ) } = ( B N ) ^ { - 1 } \sum _ { i , n } \| \widehat { u } _ { i , n } - u _ { i , n } \| _ { T _ { z _ { i , n } } \mathbb { S } ^ { 2 } } ^ { 2 } } \end{array}$   
9: Backpropagate $\widehat { \mathcal { L } } _ { \mathrm { R F M } } ^ { ( N ) } + \mathcal { L } _ { \mathrm { s e m } }$ into role tokens and con  
nector modules only

The expanded estimator is

$$
\widehat { \mathcal { L } } _ { \mathrm { R F M } } ^ { ( N ) } = \frac { 1 } { B N } \sum _ { i = 1 } ^ { B } \sum _ { n = 1 } ^ { N } \| v _ { \phi } ( z _ { i , n } , t _ { i , n } , c _ { i } ) - u _ { i , n } \| _ { T _ { z _ { i , n } } \mathbb { S } ^ { 2 } } ^ { 2 } .\tag{11}
$$

The semantic auxiliary loss is computed once per image:

$$
\mathcal { L } = \widehat { \mathcal { L } } _ { \mathrm { R F M } } ^ { ( N ) } + \sum _ { r \in \mathcal { R } _ { \mathrm { a d m } } } \lambda _ { r } \mathcal { L } _ { r } ^ { \mathrm { C E } } .\tag{12}
$$

This estimator is unbiased for the RFM expectation. If the N flow samples are conditionally independent given $( x _ { i } , y _ { i } , c _ { i } )$ , the Monte-Carlo variance of the flow-sampling component decreases approximately as $1 / N$ , while the MLLM forward cost remains unchanged. The trade-off is that RFM-head compute and activation memory scale linearly with N.

## C. Sweep over the Expansion Factor N

Figure 7 reports an oracle-condition sweep over the expansion factor N on IM2GPS3K. The oracle protocol uses teacher-forced ground-truth semantic fields and therefore probes condition-to-coordinate training rather than the deployable generated-condition setting.

Among the strictly comparable runs, moderate expansion gives the strongest fine- and mid-range oracle accuracy: $N = 8$ is best at @25 and @200. Larger expansion does not monotonically improve accuracy, suggesting that too many repeated flow samples from the same semantic condition can over-emphasize local flow-sampling variance relative to condition diversity.

![](images/bfed1409cc778fc5fb39722c7583ed712ee2ded6cccd46250d456e7837974e06.jpg)  
Figure 7. Oracle accuracy versus 1-to-N expansion.

## D. Freezing versus Unfreezing the RFM Head

GeoBridge freezes the pretrained RFM head and trains only the semantic interface. To verify that this is not merely a computational shortcut, we compare freezing and unfreezing the coordinate head under two settings.

End-to-end unfreezing from the start. We first train GeoBridge end-to-end with the same connector and roletoken design, but allow the RFM head to update together with the bridge. As shown in Fig. 8, unfreezing the head produces nearly the same optimization curve as the frozenhead run. The training losses rapidly enter the same range, and the @200 km validation accuracy converges to an almost identical plateau. Although the unfrozen head has more trainable capacity, it does not yield a measurable accuracy gain; in mean-error evaluation, the frozen-head run is slightly better. We therefore keep the RFM head frozen in the final model.

Second-stage unfreezing after connector pretraining. We also test a two-stage alternative: first train the Geo-Bridge connector with the RFM head frozen, then load the first-stage connector and continue training while unfreezing the coordinate head. Fig. 9 shows that the second-stage loss first rises rapidly after unfreezing the coordinate head and then gradually stabilizes. Meanwhile, validation accuracy drops immediately after unfreezing and later recovers, but it never surpasses the first-stage frozen-head performance; longer second-stage training yields only marginal gains. This suggests that adapting the pretrained vector field after the connector has already learned its interface mainly perturbs a useful spherical decoder prior rather than improving the semantic-to-coordinate mapping.

![](images/56d683fcf707cabd23f971d4344d722774c2e7155331180d4ab567ea33bfb6c2.jpg)

![](images/f0284a120125edbc786eb082ab9cf91bccbb325906238de6b8ca3f9f1ee0b6cf.jpg)  
Freeze PLONK  Unfreeze PLONK

Figure 8. End-to-end comparison between freezing and unfreezing the RFM head.  
![](images/d7bf59327459552acff9639a9047c0814da0c873128f406aa87105ecf3b644c4.jpg)

![](images/c52d77a6b6ebfa99792a89cf60cd101ddc96fa98e46647d9601a001451df32d2.jpg)  
Figure 9. Two-stage unfreezing after connector pretraining. We initialize from a first-stage connector trained with a frozen RFM head, then unfreeze the coordinate head for continued training.

Table 8. Conditioning-interface ablation under the deployable condition (IM2GPS3K). The same variants as Table 4, but each model uses its own generated semantic prefix at inference rather than a teacher-forced oracle condition. Acc@k (%); best per column in bold. The last row is GeoBridge’s deployable result.
<table><tr><td>Conditioning variant</td><td>@25 @200</td><td>@750</td><td>@2500</td></tr><tr><td>Single-token, MLP</td><td>21.72</td><td>42.63 65.34</td><td>82.59</td></tr><tr><td>Role tokens, ungrouped, MLP</td><td>20.94</td><td>41.79 65.57</td><td>83.00</td></tr><tr><td>Role tokens, grouped, MLP</td><td>23.36</td><td>48.74 67.89</td><td>84.38</td></tr><tr><td>Role tokens, grouped, Q2Enc</td><td>38.50</td><td>52.81 70.90</td><td>84.38</td></tr><tr><td>Full GeoBridge (+ role CE)</td><td>38.67</td><td>52.89 70.37</td><td>84.42</td></tr></table>

## E. Deployable component ablation

Table 4 isolates each conditioning component under the oracle condition, which measures the ceiling a variant could reach with a correct semantic prefix. Table 8 repeats the same progression under the deployable condition—each variant using its own generated prefix on IM2GPS3K— so the two tables together separate representational ceiling from realized gain.

Three observations follow. First, the ordering of the early steps matches the oracle study: adding ungrouped tokens does not help (it slightly dips), while grouping gives a modest gain, reaffirming that structured decomposition rather than token count is the active ingredient. Second, the connector is the dominant deployable driver: replacing the MLP with Q2Enc lifts @25 from 23.3 to 38.5—the single largest deployable jump, accounting for roughly 90% of GeoBridge’s improvement over the single-token baseline. Third, and unlike the oracle study, the auxiliary role crossentropy is approximately neutral in deployment: it moves @25 by only +0.2 (38.50→38.67) and is marginally lower at 750 km, even though it raised the oracle ceiling by more than ten points (Table 4).

This gap is consistent with our central finding rather than at odds with it. Deployable accuracy is gated by the quality of the generated semantic prefix, so a component that sharpens how a correct condition is exploited yields little when the condition itself is frequently wrong; role CE accordingly shows its value at the ceiling (oracle) far more than in realized accuracy (deployable). We retain it because it raises the achievable ceiling and is expected to pay off as upstream semantics improve, but we report its current deployable contribution transparently. The same reading explains why most of GeoBridge’s deployable accuracy comes from the connector, which improves the semanticsto-coordinate mapping regardless of whether the underlying prefix is correct.

## F. Flow Matching and Riemannian Flow Matching

This section expands the coordinate generator used by Geo-Bridge. The head itself is inherited from RFM [5] and kept frozen; GeoBridge only learns the semantic condition supplied to this head. We use the prior-to-data convention below, matching the sampling direction in GeoBridge: a random base point is drawn at t = 0 and transported to a predicted geographic point at $t = 1$ . This is equivalent to the data-to-noise convention often used in flow-matching papers after the time change $t \mapsto 1 - t$

Coordinates on the sphere. A latitude–longitude pair $( \varphi , \lambda )$ is represented as a unit vector

$$
y ( \varphi , \lambda ) = \big ( \cos \varphi \cos \lambda , \cos \varphi \sin \lambda , \sin \varphi \big ) \in \mathbb { S } ^ { 2 } \subset \mathbb { R } ^ { 3 } .\tag{13}
$$

The geodesic distance between two unit vectors $p , q \in \mathbb { S } ^ { 2 }$ is

$$
d _ { \mathbb { S } ^ { 2 } } ( p , q ) = \operatorname { a r c c o s } \big ( \mathrm { c l i p } ( p ^ { \top } q , - 1 , 1 ) \big ) ,\tag{14}
$$

and the distance in kilometers is obtained by multiplying by the Earth radius $R _ { \oplus }$

Exponential and logarithmic maps. For $p , q \in \mathbb { S } ^ { 2 }$ , let $\alpha = d _ { \mathbb { S } ^ { 2 } } ( p , q )$ . The logarithmic map sends q to a tangent

vector at p:

$$
\log _ { p } ( q ) = { \frac { \alpha } { \sin \alpha } } { \big ( } q - \cos \alpha p { \big ) } \in T _ { p } \mathbb { S } ^ { 2 } .\tag{15}
$$

For a tangent vector $v \in T _ { p } \mathbb { S } ^ { 2 }$ , the exponential map is

$$
\exp _ { p } ( v ) = \cos \| v \| p + \sin \| v \| { \frac { v } { \| v \| } } .\tag{16}
$$

In implementation, the usual small-angle limits are used when α or $\lVert v \rVert$ is close to zero. A vector $a \in \mathbb { R } ^ { 3 }$ can be projected to the tangent plane at p by

$$
\begin{array} { r } { \Pi _ { p } ( a ) = a - ( \overline { { a } } ^ { \top } p ) p . } \end{array}\tag{17}
$$

Euclidean flow matching. Let c denote the GeoBridge condition, $y \in \mathbb { S } ^ { 2 }$ the target coordinate, and ϵ a base sample. In Euclidean flow matching, the interpolant is defined in $\mathbb { R } ^ { 3 }$

$$
z _ { t } ^ { \mathrm { E } } = ( 1 - \kappa ( t ) ) \epsilon + \kappa ( t ) y , \qquad \kappa ( 0 ) = 0 , \quad \kappa ( 1 ) = 1 .\tag{18}
$$

The target velocity is

$$
u _ { t } ^ { \mathrm { E } } = \frac { d z _ { t } ^ { \mathrm { E } } } { d t } = \dot { \kappa } ( t ) ( y - \epsilon ) ,\tag{19}
$$

and the Euclidean FM loss is

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { ( y , c ) , \epsilon , t } \left[ \left| \left| v _ { \phi } ( z _ { t } ^ { \mathrm { E } } , t , c ) - u _ { t } ^ { \mathrm { E } } \right| \right| _ { 2 } ^ { 2 } \right] .\tag{20}
$$

Because $z _ { t } ^ { \mathrm { E } }$ does not generally stay on $\mathbb { S } ^ { 2 }$ , Euclidean FM requires a final projection or normalization to produce a valid location.

Riemannian flow matching on $\mathbb { S } ^ { 2 }$ . RFM instead constructs the interpolation directly on the sphere. Given $\epsilon \sim$ Unif(S<sup>2</sup>) and target $y \in \mathbb { S } ^ { 2 }$ , define

$$
z _ { t } = \exp _ { \epsilon } \left( \kappa ( t ) \log _ { \epsilon } ( y ) \right) \in \mathbb { S } ^ { 2 } .\tag{21}
$$

The corresponding tangent velocity is

$$
u _ { t } = \dot { \kappa } ( t ) \operatorname { P T } _ { \epsilon \to z _ { t } } \left[ \log _ { \epsilon } ( y ) \right] \in T _ { z _ { t } } \mathbb { S } ^ { 2 } ,\tag{22}
$$

where $\mathrm { P T } _ { \epsilon  z _ { t } }$ denotes parallel transport along the geodesic from $\mathbf { \mu } \in \mathbf { t o } \ z _ { t }$ . The RFM objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { R F M } } = \mathbb { E } _ { ( y , c ) , \epsilon , t } \left[ \left\| v _ { \phi } ( z _ { t } , t , c ) - u _ { t } \right\| _ { T _ { z _ { t } } \mathbb { S } ^ { 2 } } ^ { 2 } \right] . } \end{array}\tag{23}
$$

At inference, the frozen RFM head transports a random point on the sphere by integrating

$$
\begin{array} { r } { \dot { z } _ { t } = v _ { \phi } ( z _ { t } , t , c ) , \qquad z _ { 0 } \sim \operatorname { U n i f } ( \mathbb { S } ^ { 2 } ) , } \end{array}\tag{24}
$$

with manifold Euler steps

$$
z _ { t + \Delta t } = \exp _ { z _ { t } } \left( \Delta t v _ { \phi } ( z _ { t } , t , c ) \right) , \qquad \Delta t = 1 / K .\tag{25}
$$

This keeps every intermediate point on $\mathbb { S } ^ { 2 }$ and avoids the Euclidean projection mismatch.

## G. Additional Qualitative Results

The main paper groups deployable predictions by whether the generated semantic condition contains the correct country and city. Figure 10 provides additional examples for each bucket to show how the coordinate decoder responds to semantic condition quality.

## H. Inference efficiency

GeoBridge is attached after the same frozen GLOBE semantic-generation pass used by the place-name baseline, so it does not add another autoregressive decode. Consequently, the MLLM-side TTFT and TPOT are unchanged: GeoBridge only incurs a fixed post-generation cost for reading the role-token states, projecting them into a single condition, and running the frozen RFM sampler. In the current implementation, the role-token readout is profiled as an uncached one-forward pass and takes 0.62s median latency, while the 32-step RFM sampler adds 0.25s; together they form a conservative 0.85s median bridge overhead. The deterministic linear-layer MAC estimate is 2.33G, dominated by the RFM head. A deployable implementation can reuse the generated-prefix KV cache for the role-token readout, so the measured latency should be interpreted as an implementation upper bound rather than a required extra reasoning pass.

## I. Discussion and Limitations

What limits deployable accuracy. Our analysis consistently localizes the limit to the condition rather than the head. Under an oracle condition the frozen RFM head reaches a far higher ceiling; deployable error tracks the correctness of the generated semantics; and stronger geographic backbones yield better conditions. Together these reframe the open problem from designing a better coordinate head to producing a more distribution-aligned geospatial condition. That goal is complementary to advances in MLLM reasoning: a reasoning- or CoT-augmented backbone would improve the very condition GeoBridge decodes, so the two lines compose rather than compete.

Formatted prompting. GeoBridge reads its condition from a fixed prompt template, with the role tokens occupying designated positions. It therefore does not yet support coordinate estimation embedded in free-form generation— for example, emitting a location partway through openended reasoning. A natural extension is to fold the learnable role tokens into the MLLM’s own vocabulary, so the model can emit them directly and produce flexible, on-demand coordinate estimates under arbitrary outputs.

Sensitivity to the textual condition. Because we inject ground-truth semantic text during training, the learned condition is sensitive to surface form: different verbalizations of the same place—abbreviations, alternate names, or language variants—can yield markedly different predictions, since coordinates are bound to specific training-set strings rather than to a normalized geographic entity. Normalizing place references, or grounding them in an entity vocabulary, is a promising way to reduce this variance.

## J. Training Prompt and Teacher-Forced Prefix

GeoBridge uses structured semantic prompts to expose geographic fields before reading the learned role-token states. We used two prompt templates during development.

Country–state–city prompt. We also used a more structured administrative prompt that asks for three geographic fields:

You must analyze the input image   
and provide a structured location   
prediction at exactly four levels of   
geographic granularity:   
1. Country   
2. State (Administrative region)   
3. City (e.g., "Auschwitz",   
"Golden Gate Bridge", "Forbidden   
City")

The corresponding teacher-forced answer prefix is

```handlebars
{{"country": "{country text}",
"state": "{region text}", "city":
"{city text}"<coordinates>
```

Country–city prompt. The first template follows the GLOBE-style semantic geolocation format and asks the MLLM to output only country and city fields:

Based on the provided image, please   
output its location (country   
and city) directly without any   
explanation.   
Your final answer includes these two   
lines in your response:   
country: [country name]   
city: [city name]

For image i, the teacher-forced answer prefix is constructed as

teacher answer = f"country:   
{country text} city:   
{city text}<coordinates>"

In both templates, <coordinates> is a dataformatting placeholder rather than a literal text span used by the language model. Before forwarding through the MLLM, this placeholder is replaced by M learnable special tokens.

![](images/7ca1b3b295a5682dd06d6a8ccf59c8b2c7be70f0e630e819e8ebbd2facd30477.jpg)  
Figure 10. Additional qualitative results in the both correct, country correct, city wrong, both wrong bucket.

For semantic supervision, we apply a field-level mask to handle incomplete administrative annotations. If the annotation file does not contain a valid region/state entry for a sample, the corresponding region CE term is set to zero for that sample. More generally, when a region head is enabled, the semantic loss can be written as

$$
\mathcal { L } _ { \mathrm { s e m } } = \lambda _ { \mathrm { c t y } } \mathcal { L } _ { \mathrm { c t y } } ^ { \mathrm { C E } } + \lambda _ { \mathrm { r e g } } m _ { \mathrm { r e g } } \mathcal { L } _ { \mathrm { r e g } } ^ { \mathrm { C E } } + \lambda _ { \mathrm { c i t y } } \mathcal { L } _ { \mathrm { c i t y } } ^ { \mathrm { C E } } ,\tag{26}
$$

where $m _ { \mathrm { r e g } } \in \{ 0 , 1 \}$ indicates whether a valid region/state label is available. For GLOBE-based GeoBridge, the base model predicts country and city but not region; therefore we keep $\lambda _ { \mathrm { r e g } } = 0$ throughout. The region token is still retained as a contextual role, but it is not optimized with a region classification loss in this setting.