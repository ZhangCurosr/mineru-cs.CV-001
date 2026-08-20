# Composed Historical Image Retrieval by Modeling Temporal Representations

Adrià Molina1,2 amolina@cvc.uab.cat

Oriol Ramos Terrades1,2 oriolrt@cvc.uab.cat

Josep Lladós1,2 josep@cvc.uab.cat

1 Centre de Visió per Computador Universitat Autònoma de Barcelona Bellaterra, Catalonia

²Computer Science Department Universitat Autònoma de Barcelona, Bellaterra, Catalonia

## Abstract

While time evolves linearly, the geometry of neural embedding spaces is inherently multi-dimensional, often chaotic, and difficult to interpret. In principle, one could constrain an embedding space to a single temporal dimension; however, such a reduction would sacrifice performance on downstream tasks, as one-dimensional embeddings cannot retain sufficient expressive capacity. This paper asks whether it is possible to learn representations that preserve temporal structure while remaining effective for image and object retrieval, and answers this question by building the mathematical foundations of such a system. We propose Temporally Decomposable Image Representations (TDIR), a representation learning algorithm that decomposes historical photographs into separate date and content components through orthogonal subspaces. We define and prove the conditions under which such a decomposition is achievable, characterize the error incurred when those conditions are only partially met, and show that orthogonality between temporal and categorical subspaces emerges naturally from the joint optimization, without requiring it to be imposed explicitly. Beyond its geometric properties, TDIR enables a class of transitive operations on embedding spaces: the temporal information of one image can be extracted and injected into the representation of another, with no label supervision required. All theoretical properties are grounded and validated in the realworld problem of Composed Image Retrieval on historical photographs, where a query simultaneously specifies object content and a target time period, either through labels or through example images. This in-the-wild setting serves as a concrete backing for the propositions we derive, offering an intuitive and interpretable way to navigate photographic archives while maintaining competitive performance in both date estimation and object retrieval.

## 1 Introduction

Despite photography emerging late during the 19th century, it still comprises an important portion of archival data. More precisely, around 5-10% of archival material is estimated to be non-textual (see Appendix B), despite covering a significantly shorter historical span than printed or handwritten documents. Unlike traditional historical records, however, photographs encode most of their information visually, making their description through discrete metadata inherently restrictive. In this context, many Digital Humanities projects aim to improve access to non-textual collections through semantic search [5] or automatic metadata completion []. Among such metadata descriptors, the date of a photograph is one of the most prominent task in historical photography analysis [, 9, 3, 4, 3], as it enables historians and social scientists to contextualise much of the information contained in the image.

![](images/748cc32bd6173f980e0477cc1887b6ffa45097f874fe2b64619daec27af15b8e.jpg)  
Figure 1: Visual abstract of the three proposed representation objectives, showing composed image retrieval for query-by-label (a), query-by-example (b) and query-by-examples (c), in the latter, the idea is to inject the temporal information of an image into the object representation of the other query.

Temporal information alone, however, is insufficient for a complete archival description. Archivists must also contextualise the depicted objects within their historical period. In archival science terms [25], the preservation interest of a document is partially determined by its probative value: the capacity of the image to uniquely illustrate an event, person, technology, or phenomenon. This requires understanding not only when a photograph was taken, but also how the objects it contains relate to their historical moment. This archival reasoning has direct implications for image analysis. Methods for automatic date estimation have evolved from low-level visual predictors [3] toward semantic and object-centric approaches [22]. We argue that the latter is closer to the reasoning process of archivists, who infer dates through the historical signatures of the objects appearing in the scene. Objects such as cars, clothes, or posters evolve at different temporal rhythms, allowing trained observers to identify inconsistencies or confirming regularities at a glance.

This object-sensitive reasoning, however, is difficult to replicate in standard neural representations, where colour, texture, objects, and date tend to be entangled in ways that obscure the individual temporal evolution of each category. More critically for archival practice, current retrieval systems do not allow a user to query an archive by composing an object of interest with a target time period, as an archivist naturally would. As illustrated in Figure 1, the ideal system would support queries such as: "find images of jackets from the 1940s", or more powerfully, "ind images of cars that look contemporaneous to this photograph of a politician"— without requiring any date label for the reference image.

To enable this, we propose Temporally Decomposable Image Representations (TDIR), a formal framework for representation learning in which the temporal and categorical components of an image embedding are separated into orthogonal subspaces. We formalise the conditions under which such a decomposition holds, prove that the required orthogonality emerges naturally from joint optimisation, and characterise the error incurred when those conditions are only partially met. These theoretical contributions are, to our knowledge, the first formal treatment of temporal decomposability in visual embeddings and stand independently of any specific application. Concretely, we establish two formal properties:

• Temporally Decomposable Image Representation: an embedding space is temporally decomposable when any representation can be expressed as a linear combination of independent category and year vectors, such that each component exclusively encodes its respective factor of variation.

• Joint Proxy Optimization: We propose a joint optimization which promotes TDIR in real case scenarios. Thus, the year centroid, interpreted as a displacement vector, can transport any representation to a different date without altering its categorical content.

Beyond their theoretical interest, both properties admit a practically relevant instantiation: Composed Image Retrieval on in-the-wild historical photographs, where a query simultaneously specifies object content and a target time period. This provides empirical evidence that the derived properties yield interpretable behavior in applied settings beyond the theoretical formulation of the framework.

## 2 Related Work

Early approaches to photographic date estimation relied on low-level visual cues such as colour histograms and film grain [3], while more recent methods leverage semantic features. Crucially, however, none of these works formalise the desired properties of an ideal temporal representation. Müller et al. introduce the DEW benchmark and treat date estimation as a regression problem [9], without characterising the geometric structure that makes a representation well-suited for this task. Post-hoc analyses have revealed that temporal structure does emerge in general-purpose embeddings: In [], the authors propose a loss function that can re-organise vision embeddings according to a fixed temporal criteria, this temporal geometry is observed to naturally emerge through CLIP pretraining [B2], where embeddings carry rankable temporal information. Neither work describes how to induce such structure by design, nor what formal properties an ideally disentangled temporal representation should satisfy. TDIR addresses precisely this gap. A complementary line of work argues that date estimation should be grounded in the objects present in a scene, mirroring the reasoning of trained archivists. Ashida et al. [] propose an object-centred ensemble restricted to human subjects, and Net et al. [2] extend this idea to a broader set of object categories via a transformer-based architecture trained on DEW, introducing the DEW-B benchmark. However, both works treat object-centricity as a means to improve date prediction, not as a representational goal in its own right: object identity is discarded once the date is estimated. Furthermore, since [2] requires training a dedicated image encoder per object category, the method is evaluated on only six categories, which raises scalability concerns to larger object sets. By contrast, TDIR jointly preserves both categorical and temporal factors within a single unified backbone, enabling retrieval along either dimension independently and scaling naturally to a large number of object categories. Composed Image Retrieval (CIR), which involves querying by combining a reference image with a modification attribute, has been studied in general-domain settings [28, B4], but existing historical benchmarks do not support this paradigm. DEW provides date annotations without object labels [9]; IMAGO and yearbook-based datasets [4, 9, 3] are restricted to faces and thus offer only a single semantic category. EUFCC-CIR proposed a composed retrieval dataset for cultural heritage collections [], but it contains no temporal metadata, making category-date composition impossible. No existing benchmark simultaneously provides object-category and year-level annotations over a diverse photographic archive.

## 3 Methodology

## 3.1 Decomposable and Transitive Temporal Representations

Training operates on tuples $T = ( x _ { A } , x _ { B } )$ where each image x carries two attributes: a category $c \in \{ 1 , \ldots , C \}$ and a year $y \in \{ 1 , \ldots , Y \}$ . In the tuples, $c _ { A } = c _ { B } = c$ and $y _ { A } \neq y _ { B }$ . The geometric construction of our method rests on a key structural requirement for the year subspace: that temporal displacements between dates behave consistently across all categories.

Definition 1 (Temporal Independence of Classes). Given class and date centroids $( \mu ^ { c } , \mu ^ { y } )$ and their corresponding labels (c and y), a given representation is said to be temporally independent from the object classes when

$$
p ( \mu ^ { c } , c , \mu ^ { y } , y ) = p ( \mu ^ { c } , c ) \cdot p ( \mu ^ { y } , y )\tag{1}
$$

From this definition, the pair $( \mu ^ { c } , c )$ is independent of $( \mu ^ { y } , y )$ : category centroids carry no information about dates, and date centroids carry no information about categories.

Definition 2 (Temporal Transitivity). For any ordered pair of dates $( A , B )$ , define the displacement vector $\bar { \Delta } _ { A  B } ^ { y } = \mu ^ { y _ { B } } - \mu ^ { \bar { y _ { A } } } \in \mathbb { R } ^ { d }$ . The year subspace is temporally transitive if, for any triple $( A , B , C )$

$$
\Delta _ { A  B } ^ { y } + \Delta _ { B  C } ^ { y } = \Delta _ { A  C } ^ { y } .\tag{2}
$$

This property is necessary for zero-label inference: it guarantees that a year residual extracted from any image acts as a consistent temporal operator when transplanted onto another image. We now introduce the proxy and residual structures that make this property learnable.

## 3.2 Proxy Sets

Let $f : \mathcal { X } \to \mathbb { R } ^ { d }$ be a CNN encoder. Each image x carries two attributes: a category $c \in$ $\{ 1 , \ldots , C \}$ and a year $y \in \{ 1 , \ldots , Y \}$ . We seek a representation where these attributes occupy orthogonal subspaces, admitting the decomposition:

$$
\begin{array} { r } { \nu = f ( x ) = \mu ^ { c } + \mu ^ { y } , } \end{array}\tag{3}
$$

where $\mu ^ { c } , \mu ^ { y } \in \mathbb { R } ^ { d }$ are random vectors acting as class centroids for object and temporal information. The ideal decomposition satisfies $I ( \mu ^ { c } ; \mu ^ { y } ) = 0 ;$ : the two components are mutually uninformative, and their sum captures the full label-relevant content of v.

A proxy $p \in \mathcal { P }$ is a learnable vector of the same dimensionality as the embedding space, representing the centroid of a contrastive class [8]. Unlike dataset-derived centroids, proxies are optimized end-to-end: they are simultaneously pulled toward the embeddings of their assigned class and pushed away from all others, converging to dynamic but class-specific attractors in $\mathbb { R } ^ { d }$ . We define three proxy sets:

Category proxies $\mathcal { P } ^ { c } = \{ \mu _ { 1 } ^ { c } , \ldots , \mu _ { C } ^ { c } \}$ . One proxy per object category. Each $\mu ^ { c }$ integrates over all years of category c, converging to a year-agnostic centroid. These operate on unnormalized embeddings, so both magnitude and direction contribute to proxy assignment.

Year proxies $\mathcal { P } ^ { y } = \{ \mu _ { 1 } ^ { y } , \ldots , \mu _ { Y } ^ { y } \}$ . One proxy per year, shared across all categories. These operate on $\ell _ { 2 } \cdot$ -normalized residuals $\hat { u } = u / \lVert u \rVert$ , reducing the dot product to cosine similarity and encoding temporal information purely in the angle of the representation — a design choice validated by our ablation study (Table 3).

Auxiliary displacement proxies $\mathcal { K } ^ { y } = \{ k _ { 1 } ^ { y } , \ldots , k _ { Y } ^ { y } \}$ . One learnable displacement vector per year, shared across categories. These provide learned temporal offsets used during training to enforce transitivity (Definition 2). The set $\mathcal { \kappa } ^ { y }$ is discarded at inference.

Under the model assumptions of Definition 1, the following decomposability result holds:

Proposition 1 (Temporally Decomposable Image Representation). Under the model assumptions of Denition 1:

$$
I ( \mu ^ { c } , \mu ^ { y } ; c , y ) = I ( \mu ^ { c } ; c ) + I ( \mu ^ { y } ; y )\tag{4}
$$

See the proof in Appendix A.2. This implies that, under the class independence assumption, a visual embedding $\nu = f ( x )$ can be temporally decomposable with respect to the proxy sets ${ \mathcal { P } } ^ { c }$ and ${ \mathcal { P } } ^ { y }$ . Because $\boldsymbol { \nu } = \boldsymbol { \mu } ^ { y } + \boldsymbol { \mu } ^ { c }$ , the mutual information between the joint labels and the full representation decomposes as:

$$
I ( \nu ; c , y ) = I ( \mu ^ { c } ; c ) + I ( \mu ^ { y } ; y )\tag{5}
$$

Ideally, the total information about both factors contained in v is fully decomposable into independent contributions: categorical information $I ( \mu ^ { c } ; c )$ and temporal information $I ( \mu ^ { y } ; y )$ . In the ideal case, $\mu ^ { c }$ and $\mu ^ { y }$ are full descriptors of v with respect to $( c , y )$ : knowing both proxies captures everything v encodes about category and year, with no label-relevant information remaining.

In practice, however, there is some statistical entanglement between the appearance of certain objects and their respective dates. We therefore account for a residual signal $\boldsymbol \varepsilon \in \mathbb { R } ^ { d }$ present in v:

$$
\boldsymbol { \nu } = \boldsymbol { \mu } ^ { c } + \boldsymbol { \mu } ^ { y } + \varepsilon\tag{6}
$$

Proposition 2 (Practical Learnability of TDIRs). Under the TDIR framework, the discrepancy term $\delta \triangleq I ( \varepsilon ; c , y )$ depends exclusively on the statistical structure of the training data and not on the choice of proxy vectors $\mu ^ { c }$ or $\mu ^ { y }$

$$
I ( \nu ; c , y ) \leq I ( \mu ^ { c } ; c ) + I ( \mu ^ { y } ; y ) + \delta \qquad w h e r e \qquad \delta \triangleq I ( \varepsilon ; c , y )\tag{7}
$$

Consequently, $\delta$ constitutes a data-dependent constant with respect to the optimization, and the category and year centroids can be learned freely to satisfy the orthogonality constraint $\langle \mu ^ { c } , \mu ^ { y } \rangle = 0$ without affecting this error bound (see proof in Appendix A.3). A representation v becomes more temporally decomposable as $\delta$ decreases. This leaves open a potential circularity: Proposition 2 assumes orthogonality to guarantee free learning of $\mu ^ { c }$ and $\mu ^ { y }$ , yet this orthogonality has not been enforced. Theorem 1 will resolve this by showing that orthogonality emerges implicitly from the proposed training strategy.

(b)  
![](images/6f92df5b4a69cc29ac572ad4af419f70f634f8e72437939c0769d8695e431ca4.jpg)  
(a)

![](images/ab49cf0a0208f5307c2d5af82e5a0be33d43a089c66a36c8c718ac2fffe952c0.jpg)

![](images/ba594f3138081eec6fc1a8d1f1949b6b3f7b19ad1cb1ad4b843e2bdc31d7afe4.jpg)  
(c)

![](images/30855792defe9ccf9e1ea4e6bd0559e34422af47ca414dcab533d48c3240e87d.jpg)  
(d)  
Figure 2: Training pipeline. (a) Proxy loss forms year-agnostic category clusters. (b) Subtracting $\mu ^ { c }$ centers each cluster into a shared temporal subspace. (c) Year proxies are optimized on normalized residuals via cosine distance. (d) The swapping trick translates residuals across years, enforcing temporal transitivity.

## 3.3 Residual Vectors

Given the proxy sets above, we build four residual vectors from each training tuple $T =$ $\left( x _ { A } , x _ { B } \right)$ with shared category c and distinct years $y _ { A } \neq y _ { B }$ . The four-step geometric construction is illustrated in Figure 2.

Direct residuals. Subtracting the category centroid $\mu ^ { c }$ re-centers each cluster at the origin, revealing the temporal component (Figure 2b):

$$
r _ { A } = \nu _ { A } - \mu ^ { c } , \qquad r _ { B } = \nu _ { B } - \mu ^ { c } .\tag{8}
$$

In $r _ { A }$ and $r _ { B } .$ , the information necessary to infer the category has been removed. Consequently, even if temporal information were present in the category subspace, it is stripped from the representation and will not propagate to subsequent steps.

Shifted residuals (swapping trick). To enforce temporal transitivity, we additionally form two cross-year residuals using the auxiliary displacement proxies $\kappa ^ { y }$

$$
\begin{array} { r } { r _ { A  B } = ( \nu _ { A } - \mu ^ { c } ) + k _ { B } ^ { y } , \qquad r _ { B  A } = ( \nu _ { B } - \mu ^ { c } ) + k _ { A } ^ { y } . } \end{array}\tag{9}
$$

The vector $r _ { A  B }$ displaces image A's temporal residual toward year $y _ { B } ;$ if the temporal manifold is truly transitive, this shifted residual should be indistinguishable from $r _ { B } \mathrm { ~ - ~ }$ and therefore close to $\mu ^ { y _ { B } }$ . The symmetric argument holds for $r _ { B  A }$ . All four residuals are $\ell _ { 2 }$ -normalized before being passed to the year loss:

$$
\hat { r } _ { * } \gets r _ { * } / \| r _ { * } \| .\tag{10}
$$

This normalization encodes temporal information purely as angular structure in the shared subspace (Figure 2c-d).

## 3.4 Proxy Losses and Joint Objective

Proxy loss. All objectives share a single form []. For an embedding u with ground-truth proxy $\boldsymbol { \mu } ^ { * } \in \mathcal { P }$ , the proxy loss is:

$$
\mathcal { L } ( u , \mu ^ { * } ; \mathcal { P } ) = - \log \frac { \exp ( u ^ { \top } \mu ^ { * } ) } { \displaystyle \sum _ { p \in \mathcal { P } } \exp ( u ^ { \top } p ) } .\tag{11}
$$

This is a softmax cross-entropy over proxy assignments: it maximizes the score of the correct proxy $\mu ^ { * }$ relative to all others in $\mathcal { P }$ , pulling u into the neighborhood of $\mu ^ { * }$ while repelling it from every competing proxy.

Category loss. The category subspace is optimized directly on the full (unnormalized) embeddings:

$$
\begin{array} { r } { \mathcal L ^ { c } ( T ) = \mathcal L ( \nu _ { A } , \mu ^ { c } ; \mathcal P ^ { c } ) + \mathcal L ( \nu _ { B } , \mu ^ { c } ; \mathcal P ^ { c } ) . } \end{array}\tag{12}
$$

Each proxy $\mu ^ { c }$ integrates over all years of category c (Figure 2a). At this stage, date information may still be present in the embedding through spurious correlations such as textures or color; the subsequent residual construction removes it.

Direct temporal loss. The year subspace is optimized on the normalized direct residuals:

$$
\mathcal { L } _ { \mathrm { d i r e c t } } ^ { y } ( T ) = \mathcal { L } ( \hat { r } _ { A } , \mu ^ { y _ { A } } ; \mathcal { P } ^ { y } ) + \mathcal { L } ( \hat { r } _ { B } , \mu ^ { y _ { B } } ; \mathcal { P } ^ { y } ) .\tag{13}
$$

This makes the year subspace estimable, but not yet geometrically consistent across categories.

Transitive temporal loss. To enforce temporal transitivity (Definition 2), the shifted residuals are pulled toward their target year proxies — the year each residual has been displaced toward, rather than the image's own year:

$$
\mathcal { L } _ { \mathrm { t r a n s } } ^ { y } ( T ) = \mathcal { L } ( \hat { r } _ { A  B } , \mu ^ { y _ { B } } ; \mathcal { P } ^ { y } ) + \mathcal { L } ( \hat { r } _ { B  A } , \mu ^ { y _ { A } } ; \mathcal { P } ^ { y } ) .\tag{14}
$$

This cannot be satisfied unless the displacement $k ^ { y }$ is consistent across all categories — that is, unless the year residuals of any two images from any two categories differ by the same offset in the temporal subspace. This swapping trick directly enforces Definition 2.

Total year loss and joint objective. Combining direct and transitive terms (Figure 2d):

$$
\begin{array} { r } { \mathcal { L } ^ { y } ( T ) = \underbrace { \mathcal { L } ( \hat { r } _ { A } , \mu ^ { y _ { A } } ; \mathcal { P } ^ { y } ) + \mathcal { L } ( \hat { r } _ { B } , \mu ^ { y _ { B } } ; \mathcal { P } ^ { y } ) } _ { \mathrm { d i r e c t : ~ } r \approx \mu ^ { y } } + \underbrace { \mathcal { L } ( \hat { r } _ { A  B } , \mu ^ { y _ { B } } ; \mathcal { P } ^ { y } ) + \mathcal { L } ( \hat { r } _ { B  A } , \mu ^ { y _ { A } } ; \mathcal { P } ^ { y } ) } _ { \mathrm { t r a n s i t i v e : ~ } r + k ^ { y } \approx \mu ^ { y } } . } \end{array}\tag{15}
$$

The total minimized loss is:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } ^ { c } ( T ) + \mathcal { L } ^ { y } ( T ) . } \end{array}\tag{16}
$$

Theorem 1 (Emergent TDIR). Let $\mathcal { L } ^ { c }$ and $\mathcal { L } ^ { y }$ be the proxy loss functions defined over an embedding space $\mathbb { R } ^ { d }$ . Then:

Part I (exact case). $I f I ( \varepsilon ; c , y ) = 0 ,$ joint minimization of $\mathcal { L } = \mathcal { L } ^ { c } + \mathcal { L } ^ { y }$ implies TDIR:

$$
\operatorname* { m i n } _ { \theta } \mathcal { L } ^ { c } + \mathcal { L } ^ { y } \implies p ( \mu ^ { c } , \mu ^ { y } , c , y ) = p ( \mu ^ { c } , c ) \cdot p ( \mu ^ { y } , y )\tag{17}
$$

Part II (asymptotic case). $I f I ( \varepsilon ; c , y ) \neq 0 ,$ joint minimization implies asymptotic TDIR:

$$
\operatorname* { m i n } _ { \theta } \mathcal { L } ^ { c } + \mathcal { L } ^ { y } \implies \langle \mu ^ { c } , \mu ^ { y } \rangle \to O \bigg ( \frac { I ( \varepsilon ; c , y ) } { \| \mu ^ { y } \| ^ { 2 } } \bigg ) \quad \forall c , y\tag{18}
$$

Algorithm 1 TDIR Training Loop   
1: Initialize: $\mathcal { P } ^ { c } \in \mathbb { R } ^ { C \times d } , \mathcal { P } ^ { y } \in \mathbb { R } ^ { Y \times d } , \mathcal { K } \in \mathbb { R } ^ { Y \times d }$   
2: for epoch $e = 1 \ldots E$ do   
3: for $( x _ { A } , x _ { B } , c ) \in \mathcal { D } _ { \operatorname { t r a i n } }$ do   
4: $\nu _ { A }  f ( x _ { A } ) ; \quad \nu _ { B }  f ( x _ { B } )$ forward pass   
5: $\mathcal { L } ^ { c } \gets \mathcal { L } ( \nu _ { A } , c ; \mathcal { P } ^ { c } ) + \mathcal { L } ( \nu _ { B } , c ; \mathcal { P } ^ { c } )$ category subspace   
6: $r _ { A } = \nu _ { A } - \mu _ { c } ^ { c } ; \quad r _ { B } = \nu _ { B } - \mu _ { c } ^ { c }$ center by category (decomposable)   
7: $r _ { A  B } = \nu _ { A } + k _ { B } - \mu _ { c } ^ { c } ; \quad r _ { B  A } = \nu _ { B } + k _ { A } - \mu _ { c } ^ { c }$ swapping trick (transitive)   
8: $\hat { r } _ { * }  r _ { * } / \| r _ { * } \|$ normalize for angular encoding   
$\mathcal { L } ^ { y } = \mathcal { L } ( \hat { r } _ { A } , \mu ^ { y _ { A } } ; \mathcal { P } ^ { y } ) + \mathcal { L } ( \hat { r } _ { B } , \mu ^ { y _ { B } } ; \mathcal { P } ^ { y } ) +$ estimable temporal embedding   
9:   
$+ \mathcal { L } \big ( \hat { r } _ { A  B } , \mu ^ { y _ { B } } ; \mathcal { P } ^ { y } \big ) + \mathcal { L } \big ( \hat { r } _ { B  A } , \mu ^ { y _ { A } } ; \mathcal { P } ^ { y } \big )$ transitive temporal embedding   
10: $\{ f , \mathcal { P } ^ { c } , \mathcal { P } ^ { y } , \mathcal { K } \} \gets \{ f , \mathcal { P } ^ { c } , \mathcal { P } ^ { y } , \mathcal { K } \} - \alpha \nabla ( \mathcal { L } ^ { c } + \mathcal { L } ^ { y } )$ backward pass   
11: end for   
12: end for

See proof in Appendix A.4. Note that Proposition 2 states that $\mu ^ { y }$ and $\mu ^ { c }$ can be learned under the assumption that temporal and categorical centroids are orthogonal. Although this might seem a strong constraint, Theorem 1 shows that the joint optimization of Eq. (16) imposes this orthogonality as an emergent property of the centering and swapping tricks. It is therefore formally guaranteed that Algorithm 1 can yield TDIR in both the ideal and practical case, through an implicit orthogonality regularization arising from the swapping of temporal variables across common object categories.

## 3.5 Inference and Composed Historical Image Retrieval

At inference $\kappa ^ { y }$ is discarded. The three retrieval modes (Figure 3) follow directly from the decomposition $f ( x ) = \nu \approx \mu ^ { c } + \mu ^ { y }$

Label-based (Figure 3a). At inference, we construct a database of test embeddings $\{ f ( x _ { i } ) \} _ { i = 1 } ^ { N }$ As in any common embedding-based retrieval, we construct a query vector $q \in \mathbb { R } ^ { d }$ and returning its nearest neighbors:

$$
\left\{ x _ { ( 1 ) } , x _ { ( 2 ) } , \ldots , x _ { ( k ) } \right\} = { \mathrm { t o p } } { \mathrm { } } - k { \frac { f ( x _ { i } ) ^ { \top } q } { \| f ( x _ { i } ) \| \| q \| } } .\tag{19}
$$

By the decomposability property, a query targeting category c at year y is simply the sum of their respective proxies:

$$
\boldsymbol { q } = \boldsymbol { \mu } ^ { c } + \boldsymbol { \mu } ^ { y } .\tag{20}
$$

Since $\langle \mu ^ { c } , \mu ^ { y } \rangle = 0$ , the query lies precisely at the intersection of both subspaces, and the retrieved images $\{ x _ { ( i ) } \}$ are those whose embeddings are simultaneously close to $\mu ^ { c }$ and $\mu ^ { y }$ — that is, images of category c from year y.

(a)

![](images/6586ef4e3f1583f41f741d2763b00424e6422207c4ddeaeef1e15918d2a3461e.jpg)

![](images/b202b7fedbb0dd46b1e99e792d58267f6bc0d33cb3b67978e6b611a6979d7a19.jpg)  
(b)

![](images/cd6aa21a3b835932fd15006388e993073799ce08f5358d74cabc3d290777f513.jpg)  
(c)  
Figure 3: Inference modes. (a) Label-based retrieval via proxy addition. (b) Image-guided retrieval with a target year label. (c) Zero-label retrieval: the year residual of $x _ { y }$ is extracted and transplanted onto $f ( x _ { c } )$

Image + label (Figure 3b). When the target year $y$ is known but no category label is provided, we replace $\mu ^ { c }$ with the actual image embedding:

$$
q = f ( x _ { c } ) + \mu ^ { y } .\tag{21}
$$

This is strictly more expressive than the label-based mode: rather than retrieving images close to the category centroid $\mu ^ { c }$ , the query anchors to the instance-specific content of $x _ { c }$ — including the ε residual — while the year direction is fully determined by $\mu ^ { y }$ . Retrieved images thus share the particular visual characteristics of $x _ { c }$ , translated to year y.

Image + image, zero-label (Figure 3c) The most powerful inference mode requires no label information whatsoever. The user provides two images: a category image $x _ { c }$ — whose year is entirely unknown and irrelevant — and a temporal reference $x _ { y }$ , from which we wish to borrow the year.

Step 1: Infer the category of $x _ { y }$ . Since no category label is provided, we assign $x _ { y }$ to its nearest category proxy:

$$
\hat { c } = \underset { c ^ { \prime } } { \arg \operatorname* { m i n } } \parallel f ( x _ { y } ) - \mu ^ { c ^ { \prime } } \parallel _ { 2 } .\tag{22}
$$

Step 2: Extract the temporal information Subtracting the inferred category proxy strips the category information from $f ( x _ { y } )$ , leaving a category-agnostic vector:

$$
r _ { y } = f ( x _ { y } ) - \mu ^ { \hat { c } } \approx \mu ^ { y } + \varepsilon ,\tag{23}
$$

where $\mu ^ { y }$ is the (unknown to the user) year proxy and ε is the instance-specific residual. Crucially, neither $\mu ^ { y }$ nor the year of $x _ { y }$ need to be known — the vector $r _ { y }$ is the temporal information.

Step 3: Temporal transplant. We add $r _ { y }$ to the category embedding of $x _ { c }$

$$
q = f ( x _ { c } ) + r _ { y } .\tag{24}
$$

By temporal transitivity (Definition 2), this displaces $f ( x _ { c } )$ along the temporal manifold toward the year of $x _ { y }$ , while preserving its category direction. The k-nearest neighbors of $q$ are thus images of the same category as $x _ { c } .$ at the same period as $x _ { y }$

![](images/1bda7c72ea1007de5290cb0de68803ce9f12f6a8676d8e69a4e82df1fb8cdd23.jpg)  
Table 1: Example images and categories from the dataset.

## 4 Experimental Set-Up

## 4.1 Dataset

For the correct application and benchmarking of our proposed problem set-up, it is for us required to utilize a dataset where both the date and object annotations are available. For doing so, we take advantage of the Date Estimation in The Wild dataset (DEW) [9] with object-specific detections by using the same criteria as in [2]. In this Section, we detail the dataset and how it differs from the original DEW images.

DEW Dataset The Date Estimation in The Wild dataset contains 1M natural-scene photographs from 1930 to 1999, the date information has an excellent granularity of a year per photo. However, the images are natural scenes and, therefore, contain a variety of different objects which hinders the applicability of object-specific representations.

Object-Centric DEW Dataset In [], the authors propose a detection pipeline to crop objects detected with a minimum 10000px resolution by using the DETR [] model. Although the object-level detection (bounding boxes) are publicly available, we note that the authors use a constrained set of 6 categories, which in our case could trivialize the objectsensitive subspace.

Proposed Dataset To solve the aforementioned issue, we use the same resolution criteria, but utilize OWLv2 [4] to detect objects from all possible categories in the DETR HuggingFace implementation [] with a 30% detection threshold1. This leads us to 7,239,083 object detections of medium-to-high resolution of 586 objects and date-specific annotations from DEW (see Table 1). In this article we will focus on the sub-set of top-50 most predominant objects and 4,478,887 detections (see Appendix C). But the complete set of detections, which can be utilized to expand the presented method or for many other applications, is available to download². The test partition follows the same image selection as the original DEW dataset, from which we separate the detections according to the image source.

Table 2: Comparison across backbone architectures and the CLIP baseline (K=10).
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Label+Label</td><td rowspan=1 colspan=2>Image+LabelImage+Image</td></tr><tr><td rowspan=1 colspan=1>Backbone</td><td rowspan=1 colspan=1>Obj.  DateP@K P@K</td><td rowspan=1 colspan=1>Obj.  DateP@K P@K</td><td rowspan=1 colspan=1>Obj.  DateP@K P@K</td></tr><tr><td rowspan=2 colspan=1>CLIP (arithmetic, see Appendix D.1)CLIP (prompting, see Appendix D.2)</td><td rowspan=1 colspan=1>.152  .115</td><td rowspan=1 colspan=1>.387  .076</td><td rowspan=1 colspan=1>.389   .102</td></tr><tr><td rowspan=1 colspan=1>.415  .105</td><td rowspan=1 colspan=1>.372  .079</td><td rowspan=1 colspan=1>.520   .081</td></tr><tr><td rowspan=4 colspan=1>VGG + TDIRResNet + TDIRConvNeXt + TDIR $\mathrm { V i T + T D I R }$ </td><td rowspan=1 colspan=1>.397  .712</td><td rowspan=1 colspan=1>.388  .579</td><td rowspan=1 colspan=1>.669   .234</td></tr><tr><td rowspan=1 colspan=1>.489  .803</td><td rowspan=1 colspan=1>.444  .665</td><td rowspan=1 colspan=1>.721   .306</td></tr><tr><td rowspan=1 colspan=1>.443  .824</td><td rowspan=1 colspan=1>.416  .664</td><td rowspan=1 colspan=1>.725   .393</td></tr><tr><td rowspan=1 colspan=1>.413  .863</td><td rowspan=1 colspan=1>.355  .708</td><td rowspan=1 colspan=1>.646  .429</td></tr></table>

## 4.2 Implementation Details

The practical implementation of the Temporally Decomposable Image Representations has been evaluated using several backbone architectures in order to assess the robustness of the proposed methodology across both convolutional and transformer-based visual representations. In particular, experiments have been conducted with ConvNeXt-Base [], Vision Transformer B/32 (ViT-B/32) [], ResNet50 [], and VGG19-BN [], all initialized with ImageNet pre-trained weights [] using the official PyTorch implementations [6]. For all architectures, the original classification head was removed and replaced with a projection layer producing a fixed embedding dimension of 1024. This unified embedding size ensures a fair comparison between architectures and allows all models to operate under the same metriclearning framework. Unlike previous approaches proposed for solving the DEW dataset [9] in an object-centric manner [2], which requires ensembles of specialized models for each semantic category, the proposed methodology employs a single unified backbone capable of handling all categories jointly. This significantly scales well with the number of object categories. All backbone architectures were trained using the same optimization setup to ensure experimental consistency. The AdamW optimizer jointly optimizes the backbone parameters, the learnable category proxies, and the auxiliary temporal embeddings $( f , \mu ^ { * }$ , and $k ^ { y } )$ with a learning rate of $\bar { 1 0 } ^ { - \bar { 4 } }$ during 10 epochs, using the ProxyNCALoss proposed by Movshovitz-Attias et al. [8] via the Pytorch-Metric-Learning library [0], with no explicit triplet mining. Training was conducted on a single NVID IA A4 0 GPU. Due to the large number of object detections (approximately 4M cropped sub-images across 50 object categories), each epoch requires approximately 15 hours of computation, with GPU utilization analysis confirming that the available resources are fully saturated throughout training

## 4.3 Evaluation Protocol

We evaluate each of the three inference modes (Section 3.5) using a unified set of metrics built around two axes: categorical fidelity (does retrieval respect the object class?) and temporal fidelity (does retrieval respect the target date?). As a standard practice in date estimation [9, 3], we consider a correct date classification whenever the retrieved date is within the same lustrum (5 years period)

For each inference mode, we retrieve the top-K images for every valid query and report

two complementary precision metrics:

$$
\mathbf { O b j e c t P r e c i s i o n } @ K : \frac { \# \{ x _ { i } : c _ { i } = c \} } { K } ,\tag{25}
$$

$$
\mathbf { D a t e \ P r e c i s i o n } @ K : \frac { \# \big \{ x _ { i } : y _ { i } \approx y \big \} } { K } ,\tag{26}
$$

where $y _ { i } \approx y$ denotes temporal correctness within the same lustrum (5-year window). Given that the DEW dataset contains images from 1930-1999, the random baseline is settled at $1 0 0 \times \frac { 1 } { 7 0 / 5 } = 7 . 1 4 \%$ chance of returning a correctly classified image in terms of the date estimation. In the label-based mode, c and y are taken directly from the query labels; in the image-based modes, c is determined by $c ( x _ { c } )$ and y by the target year or the transferred residual $r _ { y }$ (see Section 3.5).

## 5 Results

In this section, we present an exhaustive evaluation of our framework in a highly applied setup. First, in Table 2, we compare several widely used Computer Vision backbones against two CLIP baselines for reference (see Appendix D for the baselines implementation). We observe that our proposed approach follows the general trends observed in computer vision, with ConvNeXt and ViT emerging as the top-performing and overall comparable architectures. The CLIP baselines prove to be competitive on retrieving the correct category, but are not sufficiently sensitive to temporal information despite its demonstrated sensitivity to image dates in [B2] and []

Table 3 presents an ablation study. A key observation from the experiment is that the proposed swapping trick not only improves transitivity, as expected, but also leads to better date and object embedding subspaces. This is reflected in the improved performance observed not only for image-based queries but also for purely label-based inference, which does not rely on any transitive property induced by injecting a date into a temporal embedding. This observation partially supports the claim in Corollary 4 presented in Appendix A.4, heavily relies on the inclusion of a swapping term and states that this combination of losses imposes an implicit orthogonality regularization at the optimum. The inclusion of an auxiliary term $( k _ { * } )$ instead of directly using the temporal proxy itself $( \mu _ { * } ^ { y } )$ in the swapping trick step, plays an important role in Image+Image inference, yielding significant gains in properly structuring the embedding space. However, when performing Image+Label inference, only a marginal performance gap is observed, likely due to the train-test mismatch at inference time caused by relying exclusively on the learned centroids $( \mu _ { * } ^ { y } )$ . Lastly, the normalization of the temporal component appears to be the least impactful, although it provides a slight performance boost in Image+Image date estimation. Curiously, performance in the Label+Label object retrieval setup is maximized if and only if all three components are present in the training regime. Figure 4 reports object and date precision as a function of source year, target year, and their absolute difference. Object precision remains stable across all three axes, confirming that temporal transplantation does not corrupt the categorical content of the query: the category subspace is effectively insulated from the temporal injection. Date precision, by contrast, reveals a clear dependency on the target year and on the source-to-target gap, but notably not on the source year itself. This asymmetry rules out a straightforward directional explanation: if the error were caused by the displacement vector $\mu ^ { y }$ acting unidirectionally pushing representations forward or backward in time indiscriminately, one would expect a monotonic pattern with respect to direction. Instead, the error follows a U-shaped curve over the target year axis, with decades at the extremes of the temporal range being harder to reach regardless of where the source lies. Furthermore, precision degrades consistently as the temporal gap widens, suggesting that the optimisation does not perfectly generalise large temporal displacements. Taken together, these findings indicate that the current additive translation mechanism, while effective at short and moderate temporal distances, is an approximation whose fidelity diminishes with displacement magnitude. A natural direction for future work is to replace the additive operator with a more expressive, potentially non-linear mechanism capable of modelling large temporal jumps more faithfully.

<table><tr><td colspan="3"></td><td colspan="2">Label+Label</td><td colspan="2">Image+Label</td><td colspan="2">Image+Image</td></tr><tr><td>Swap</td><td>Aux</td><td>Norm</td><td>Obj. P@K</td><td>Date P@K</td><td>Obj. P@K</td><td>Date P@K</td><td>Obj. P@K</td><td>Date P@K</td></tr><tr><td>X</td><td>√</td><td>√</td><td>.362</td><td>.712</td><td>.342</td><td>.568</td><td>.683</td><td>.358</td></tr><tr><td>√</td><td>x</td><td>√</td><td>.355</td><td>.822</td><td>.343</td><td>.676</td><td>.691</td><td>.354</td></tr><tr><td>√</td><td>√</td><td>x</td><td>.358</td><td>.816</td><td>.349</td><td>.675</td><td>.708</td><td>.362</td></tr><tr><td>√</td><td>√</td><td>√</td><td>.443</td><td>.824</td><td>.416</td><td>.664</td><td>.725</td><td>.393</td></tr></table>

Table 3: Ablation study (K=10, ConvNext) for our method including the usage of the swapping trick, auxiliary proxies and normalization of the temporal component.

![](images/e7e5ca0e884adc747cdd64c4db0effc04be614d45913a562c3f5bb23e92707a2.jpg)

![](images/6e7745dd6ccfc682b0fc50e07263e4adb865da4cfaa8a04703ebe609682d0031.jpg)

![](images/b4f9f643937c8134c0548ddae3c68853609b0ab953a31ec74d46767ee6284911.jpg)  
Two-Image Translation - Year MAE

![](images/88399a4f6ea6f4d3e0a2a139be71f0479ced83a7ec648249d4fdd376f2c7d4c0.jpg)

![](images/06eca396b88ee3abc59aedca2b268c402993cd8bfbc639423d1c320d36b8538f.jpg)

![](images/869ebf7dcc5166a58e4c94b5b751e617c6e1db94e7af747db67134a686814f9f.jpg)  
Figure 4: Calibration error bars when using query-by-examples year injection

Although the numerical results may appear modest, a qualitative inspection in Table 4 reveals that the method generally behaves as expected. While errors in the predicted years are present, they mostly correspond to inaccuracies within the same lustrum rather than severe temporal artifacts (see Appendix C for a detailed analysis). This behavior may stem from two limitations. First, the temporal subspace is constrained to be object-agnostic, thus enforcing the same temporal displacement across all object categories. As introduced in Section 1, not every object evolves at the same pace, which may inherently limit the achievable performance under this design choice. In many cases, such as the fourth row, the temporal transplant is qualitatively correct, but the retrieved samples exhibit category ambiguity due to images lying near the boundary between two classes. In that example, the target category is “poster", yet images belonging to the “men" category are retrieved while still exhibiting the correct temporal attribution. As discussed, the error tends to increase with temporal distance, which can be observed in the fifth row, where the direction of the temporal shift is correct but the magnitude of the displacement is insufficient. Lastly, in some cases, such as the final row, the target category may simply contain too few old samples in the database because the object itself is inherently modern. As a result, the retrieved samples belong to the correct category but remain more recent than desired.

![](images/181ab13389c9e44d4c1608a19debd24393ecbfc6da7c87ec36bf19cece03ba6a.jpg)  
Table 4: Qualitative results with category error and temporal error.

## 6 Conclusions

In this paper, we have presented TDIR, a representation learning framework that formalises the decomposition of historical image embeddings into orthogonal temporal and categorical subspaces. We have established the theoretical conditions under which such a decomposition holds, proved that the required orthogonality emerges naturally from joint optimisation, and characterised the error incurred when those conditions are only partially met. To our knowledge, this constitutes the first formal treatment of temporal decomposability in visual embeddings, addressing a limitation of prior work that either observes temporal structure post-hoc or exploits it implicitly without a principled formulation. Beyond the theoretical contributions, we have grounded the framework in the novel problem of Composed Historical Image Retrieval, where a query simultaneously specifies object content and a target time period. This setting reflects a natural and practically relevant archival task, yet no existing benchmark supported its evaluation. We address this by extending the DEW dataset with object-level detections across 50 categories, enabling the first evaluation of compositional retrieval over a diverse photographic archive covering seven decades.

Limitations include the assumption of a shared temporal displacement across all object categories, which may not reflect the uneven pace at which different objects evolve historically, and the use of an additive transplantation operator whose fidelity diminishes at large temporal distances. Future work will explore non-linear temporal operators and categoryspecific temporal subspaces, as well as the extension of the benchmark to support zero-shot evaluation over unseen object categories.

## Acknowledgments

This work has been partially supported by the Spanish project PID2024-157778OB-I00, Ministerio de Ciencia e Innovación, the Departament de Cultura of the Generalitat de Catalunya, and the CERCA Program. Adrià Molina is funded with the PRE2022-101575 grant provided by MCIN / AEI / 10.13039 / 501100011033 and by ERDF/EU.

## References

[1] Archives Nationales, France. Corpusdedocumentsnumérisésdes Archives Nationales. https://www.data.gouv.fr/datasets/ corpus-de-documents-numerises-des-archives-nationales. Last visited: May 18, 2026.

[2] Shota Ashida, Adam Jatowt, Antoine Doucet, and Masatoshi Yoshikawa. Determining image age with rankconsistent ordinal classification and object-centered ensemble. In Proceedings of the 2nd ACM International Conference on Multimedia in Asia, pages 1–8, 2021.

[3] Alexandra Barancová, Melvin Wevers, and Nanne van Noord. Blind dates: examining the expression of temporality in historical photographs. arXiv preprint arXiv:2310.06633, 2023.

[4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020.

[5] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

[6] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[7] Noa Garcia, Benjamin Renoust, and Yuta Nakashima. Contextnet: representation and exploration for painting classification and retrieval in context. International Journal of Multimedia Information Retrieval, 9(1):17–30, 2020.

[8] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

[9] HuggingFace. Detr. https://huggingface.co/docs/transformers/en/model\_doc/detr, 2026. Transformers documentation. Last visited: 2026-05-18.

[10] Hugging Face. Owl-vit. https://huggingface.co/docs/transformers/model\_doc/ ow1vit, 2026. Transformers documentation. Last visited: 2026-05-18.

[11] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, July 2021. URL https: //doi.org/10.5281/zenodo.5143773. If you use this software, please cite it as below.

[12] Library of Congress. Library of Congress – Global Search. https : //www. 1oc. gov/search/?al1= t rue. Last visited: May 18, 2026.

[13] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976–11986, 2022.

[14] Matthias Minderer, Alexey Gritsenko, and Neil Houlsby. Scaling open-vocabulary object detection. Advances in Neural Information Processing Systems, 36:72983–73007, 2023.

[15] Ministerio de Cultura, España. PARES: Portal de Archivos Españoles – Estadísticas. https : / /pares. cultura.gob.es/estadisticas.html. Last visited: May 18, 2026.

[16] Ministerio de Cultura, España. Anuario de Estadísticas Culturales 2025. https:// www.cultura.gob.es/dam/jcr:daa6c0f8-9abb-48a8-8af3-7d54746d6e4a/ anuario-de-estadisticas-culturales-2025.pdf, 2025. p. 319. Last visited: May 18, 2026.

[17] Adrià Molina, Lluis Gomez, Oriol Ramos Terrades, and Josep Lladós. A generic image retrieval method for date estimation of historical document collections. In International Workshop on Document Analysis Systems, pages 583–597. Springer, 2022.

[18] Yair Movshovitz-Attias, Alexander Toshev, Thomas K Leung, Sergey Ioffe, and Saurabh Singh. No fuss distance metric learning using proxies. In Proceedings of the IEEE international conference on computer vision, pages 360–368, 2017.

[19] Eric Müller, Matthias Springstein, and Ralph Ewerth. "when was this picture taken?"-image date estimation in the wild. In European Conference on Information Retrieval, pages 619–625. Springer, 2017.

[20] Kevin Musgrave, Serge Belongie, and Ser-Nam Lim. A metric learning reality check. In European Conference on Computer Vision, pages 681–699. Springer, 2020.

[21] Francesc Net and Lluis Gomez. Eufcc-cir: A composed image retrieval dataset for glam collections. In European Conference on Computer Vision, pages 196–211. Springer, 2024.

[22] Francesc Net, Núria Hernández, Adriá Molina, and Lluis Gómez. A transformer-based object-centric approach for date estimation of historical photographs. In European Conference on Information Retrieval, pages 137–150. Springer, 2024.

[23] Frank Palermo, James Hays, and Alexei A Efros. Dating historical color images. In European Conference on Computer Vision, pages 499–512. Springer, 2012.

[24] Jakub Paplhám and Vojtěch Franc. Photo dating by facial age aggregation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 8103–8112, 2026.

[25] Trudy Huskamp Peterson. The Probative Value of Archival Documents. Swisspeace Bern, 2014.

[26] PyTorch Contributors. Torchvision models documentation. https://docs.pytorch.org/vision/ main/models.html, 2026. Accessed: 2026-05-11.

[27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[28] Arijit Ray, Filip Radenovic, Abhimanyu Dubey, Bryan Plummer, Ranjay Krishna, and Kate Saenko. Cola: A benchmark for compositional text-to-image retrieval. Advances in Neural Information Processing Systems, 36:46433–46445, 2023.

[29] Tawfiq Salem, Scott Workman, Menghua Zhai, and Nathan Jacobs. Analyzing human appearance as a cue for dating images. In 2016 IEEE Winter Conference on Applications of Computer Vision (WACV), pages 1–8. IEEE, 2016.

[30] C. et al. Schuhmann. Laion-5b: An open large-scale dataset for training next generation image-text models In Advances in Neural Information Processing Systems, 2022.

[31] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.

[32] Ankit Sonthalia, Arnas Uselis, and Seong Joon Oh. On the rankability of visual embeddings. Advances in Neural Information Processing Systems, 38:66169–66203, 2026

[33] Lorenzo Stacchio, Alessia Angeli, Giuseppe Lisanti, Daniela Calanca, and Gustavo Marfia. Imago: A family photo album dataset for a socio-historical analysis of the twentieth century. arXiv preprint arXiv:2012.01955, 2020.

[34] Hui Wu, Yupeng Gao, Xiaoxiao Guo, Ziad Al-Halah, Steven Rennie, Kristen Grauman, and Rogerio Feris. Fashion iq: A new dataset towards retrieving images by natural language feedback. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pages 11307–11317, 2021.

[35] Hua Yuan, Yuhan Li, Baohui Wang, Kaixuan Liu, and Junjie Zhang. Knowledge graph-based intelligent question answering system for ancient chinese costume heritage. npj Heritage Science, 13(1):198, 2025.

## Supplementary Material

This document contains the supplementary material for the BMVC submission:

## Composed Historical Image Retrieval by Modeling Temporal Representations

The supplementary material is organized as follows:

<table><tr><td>Section</td><td>Content</td></tr><tr><td>A</td><td>Derivations and Demonstrations</td></tr><tr><td>B</td><td>Photographic Material Calculation</td></tr><tr><td>C</td><td>Object-Centric Analysis</td></tr><tr><td>D</td><td>CLIP Baselines</td></tr><tr><td>E</td><td>Frequently Asked Questions (FAQs)</td></tr></table>

The following sections provide theoretical derivations, empirical analyses, and additional experimental details supporting the main manuscript.

## A Derivations for Decomposable Representations

## A.1 Chain rule of mutual information

Because the following derivations will heavily rely on rearranging terms in equalities using the chain rule, let us consider the definition on a general setting, using three random variables: A, B, and C.

The chain rule of mutual information states that:

$$
I ( A , C ; B ) = I ( A ; B ) + I ( C ; B \mid A )\tag{27}
$$

Intuitively, this means that the information that the joint representation $( A , C )$ carries about B can be decomposed into the information provided by A, plus the additional information provided by C, after conditioning on A.

## A.2 Temporally Decomposable Image Representations (TDIR)

As expressed in the main corpus of the manuscript, we note that category and year classes are decomposable as a design choice, which means that the expression in Definition 1

$$
p ( \mu ^ { c } , c , \mu ^ { y } , y ) = p ( \mu ^ { c } , c ) p ( \mu ^ { y } , y )\tag{28}
$$

is satisfied. In short, the class centroids are distributed with no influence from the date information and vice-versa.

Proposition 3 (Temporally Decomposable Image Representation). Under the model assumptions of Definition 1, we note that the following equality for the image representation holds:

$$
I ( \mu ^ { c } , \mu ^ { y } ; c , y ) = I ( \mu ^ { c } ; c ) + I ( \mu ^ { y } ; y )\tag{29}
$$

Proof. We apply the chain rule for mutual information to decompose the left-hand side:

$$
I ( \mu ^ { c } , \mu ^ { y } ; c , y ) = \underbrace { I ( \mu ^ { c } ; c , y ) } _ { \mathrm { f i r s t } \mathrm { t e r m } } + \underbrace { I ( \mu ^ { y } ; c , y \mid \mu ^ { c } ) } _ { \mathrm { s e c o n d t e r m } }\tag{30}
$$

We apply again the chain rule to $I ( \mu ^ { c } ; c , y )$

$$
I ( \mu ^ { c } ; c , y ) = I ( \mu ^ { c } ; c ) + I ( \mu ^ { c } ; y \mid c )\tag{31}
$$

From Definition 1, the category centroid does not provide information about the date of the image, therefore it does not provide information when conditioned on the category itself:

$$
I ( \mu ^ { c } ; c , y ) = I ( \mu ^ { c } ; c ) + I ( \mu ^ { c } ; y \dag { c } ) = I ( \mu ^ { c } ; c )\tag{32}
$$

Hence the first term simplifies to $I ( \mu ^ { c } ; c )$ if and only if the model assumption $( \mu ^ { c } \perp y | c )$ holds. Similarly, the term $I ( \mu ^ { y } ; c , y \mid \mu ^ { c } )$ is expanded again by applying the chain rule of conditional mutual information:

$$
I ( \mu ^ { y } ; c , y \mid \mu ^ { c } ) = I ( \mu ^ { y } ; c \mid \mu ^ { c } ) + I ( \mu ^ { y } ; y \mid \mu ^ { c } , c )\tag{33}
$$

From the Markov structure that emerges from Eq. (28),

$$
\mu ^ { c }  c \qquad \mathrm { a n d } \qquad \mu ^ { y }  y ,\tag{34}
$$

together with the independence assumption $\mu ^ { y } \perp \left( \mu ^ { c } , c \right)$ , it follows that

$$
I ( \mu ^ { y } ; c \mid \mu ^ { c } ) = 0 ,\tag{35}
$$

and conditioning on $( \mu ^ { c } , c )$ does not affect the dependence between $\mu ^ { y }$ and y, yielding

$$
I ( \mu ^ { y } ; y | \mu ^ { c } , c ) = I ( \mu ^ { y } ; y ) .\tag{36}
$$

Therefore,

$$
I ( \mu ^ { y } ; c , y \mid \mu ^ { c } ) = I ( \mu ^ { y } ; y ) .\tag{37}
$$

Combining the results in Equations (32) and (33):

$$
\begin{array} { r } { I ( \mu ^ { c } , \mu ^ { y } ; c , y ) = \underbrace { I ( \mu ^ { c } ; c , y ) } _ { I ( \mu ^ { c } ; c ) + 0 } + \underbrace { I ( \mu ^ { y } ; c , y \mid \mu ^ { c } ) } _ { I ( \mu ^ { y } ; y ) } = I ( \mu ^ { c } ; c ) + I ( \mu ^ { y } ; y ) } \end{array}\tag{38}
$$

which holds if and only if the initial assumption in Eq. (28) is satisfied.

Lemma 2 (Orthogonality of Centroid Subspaces). Under Definition 1, the category and temporal centroids are orthogonal:

$$
\mathrm { C o v } ( \mu ^ { c } , \mu ^ { y } ) = { \bf 0 } ,\tag{39}
$$

and the total representational variance decomposes as

$$
\operatorname { V a r } ( \mu ^ { c } + \mu ^ { y } ) = \operatorname { V a r } ( \mu ^ { c } ) + \operatorname { V a r } ( \mu ^ { y } ) .\tag{40}
$$

Proof. By Definition 1, marginalising over $\displaystyle ( c , y )$ and applying Fubini's theorem:

$$
p ( \mu ^ { c } , \mu ^ { y } ) = \int \int p ( \mu ^ { c } , c ) \cdot p ( \mu ^ { y } , y ) d c d y = \underbrace { \int p ( \mu ^ { c } , c ) d c } _ { p ( \mu ^ { c } ) } \cdot \underbrace { \int p ( \mu ^ { y } , y ) d y } _ { p ( \mu ^ { y } ) } = p ( \mu ^ { c } ) \cdot p ( \mu ^ { y } ) ,\tag{41}
$$

SO $\mu ^ { c } \perp \mu ^ { y }$ . Independence implies Cov $( \boldsymbol { \mu } ^ { c } , \boldsymbol { \mu } ^ { y } ) = \mathbf { 0 }$ , and the variance decomposition follows immediately from the bilinearity of covariance. □

## A.3 Estimating the error in real scenarios

Because the total independence $\mu ^ { c } \perp \mu ^ { y }$ (Lemma 2) cannot be guaranteed in practical terms, where data might contain severe spurious correlations, one must account for the incorporation of a noise vector $\boldsymbol { \varepsilon } \in \mathbb { R } ^ { d }$ which entangles both year and category information:

$$
\nu = f ( x ) = \mu ^ { c } + \mu ^ { y } + \varepsilon \quad { \mathrm { a n d } } \quad I ( \varepsilon ; c , y ) \neq 0\tag{42}
$$

Our model assumption is that epsilon can emerge due to correlation on the training data (hence, the labels y and $c )$ but that an independent proxy vector can be learnt despite this entanglement³:

$$
\varepsilon \perp ( \mu ^ { c } , \mu ^ { y } )\tag{43}
$$

Proposition 4 (Practical Learnability of TDIRs). Under the TDIR framework, the discrepancy term $\delta \triangleq I ( \varepsilon ; c , y )$ depends only on the statistical structure of the training data and not on the choice of proxy vectors $\mu ^ { c }$ or $\mu ^ { y }$

$$
I ( \nu ; c , y ) \leq I ( \mu ^ { c } ; c ) + I ( \mu ^ { y } ; y ) + \delta \qquad w h e r e \qquad \delta \triangleq I ( \varepsilon ; c , y )\tag{44}
$$

Consequently, $\delta$ constitutes a data-dependent constant with respect to the optimization, and the category and year centroids can be learned freely to satisfy the orthogonality constraint $\langle \mu ^ { c } , \mu ^ { y } \rangle = 0$ without affecting this error bound.

Proof. The real mutual information expression entangles ε with the label variables:

$$
I ( \mu ^ { c } , \mu ^ { y } , \varepsilon ; c , y ) = I ( \mu ^ { c } , \mu ^ { y } ; c , y ) + I ( \varepsilon ; \mu ^ { c } , \mu ^ { y } \mid c , y )\tag{45}
$$

Because the model is assumed to be trained under an independence condition $\varepsilon \perp ( \mu ^ { c } , \mu ^ { y } )$

$$
I ( \varepsilon ; \mu ^ { c } , \mu ^ { y } \mid c , y ) = I ( \varepsilon ; \mu ^ { c } , \mu ^ { y } )\tag{46}
$$

Therefore,

$$
I ( \mu ^ { c } , \mu ^ { y } , \varepsilon ; c , y ) = \underbrace { I ( \mu ^ { c } , \mu ^ { y } ; c , y ) } _ { I ( \mu ^ { c } ; c ) + I ( \mu ^ { y } ; y ) } + I ( \varepsilon ; \mu ^ { c } , \mu ^ { y } )\tag{47}
$$

To obtain the total expression for $I ( \nu ; c , y )$ , we apply the chain rule of mutual information:

$$
\begin{array} { r } { I ( \underbrace { \nu } _ { A } , \underbrace { \mu ^ { c } , \mu ^ { y } } , \underbrace { \varepsilon } ; \underbrace { c , y } _ { B } ) = I ( \nu ; c , y ) + I ( \mu ^ { c } , \mu ^ { y } , \varepsilon ; c , y \mid \nu ) } \end{array}\tag{48}
$$

Since $\boldsymbol { \nu } = \boldsymbol { \mu } ^ { c } + \boldsymbol { \mu } ^ { y } + \boldsymbol { \varepsilon }$ is a function of $( \mu ^ { c } , \mu ^ { y } , \varepsilon )$ , the joint tuple $\left( \nu , \mu ^ { c } , \mu ^ { y } , \varepsilon \right)$ carries no more information about $\displaystyle ( c , y )$ than $( \mu ^ { c } , \mu ^ { y } , \varepsilon )$ alone:

$$
I ( \nu , \mu ^ { c } , \mu ^ { y } , \pmb { \varepsilon } ; c , y ) = I ( \mu ^ { c } , \mu ^ { y } , \pmb { \varepsilon } ; c , y )\tag{49}
$$

Isolating $I ( \nu ; c , y )$

$$
I ( \nu ; c , y ) = I ( \mu ^ { c } , \mu ^ { y } , \varepsilon ; c , y ) - I ( \mu ^ { c } , \mu ^ { y } , \varepsilon ; c , y | \nu )\tag{50}
$$

Substituting the expression in Eq. (50):

$$
\boxed { I ( \nu ; c , y ) = I ( \mu ^ { c } ; c ) + I ( \mu ^ { y } ; y ) + I ( \varepsilon ; c , y ) - I ( \mu ^ { c } , \mu ^ { y } , \varepsilon ; c , y \mid \nu ) }\tag{51}
$$

where the last term $I ( \mu ^ { c } , \mu ^ { y } , \varepsilon ; c , y \mid \nu )$ represents the information lost by collapsing the three components into a single vector v. Identifying $\delta \cdot$

$$
\delta = I ( \varepsilon ; c , y ) - \underbrace { I ( \mu ^ { c } , \mu ^ { y } , \varepsilon ; c , y \mid \nu ) } _ { \geq 0 }\tag{52}
$$

we can rewrite the exact relation compactly as:

$$
\begin{array} { r } { I ( \nu ; c , y ) = I ( \mu ^ { c } ; c ) + I ( \mu ^ { y } ; y ) + \underbrace { \delta } _ { \leq I ( \varepsilon ; c , y ) } } \end{array}\tag{53}
$$

![](images/fe3f33291b6fb37b6aea3c14b425366c769ffdb2f4b604d656671c97376cd55c.jpg)  
Figure 5: Resulting graphical model from Proposition 2

Since mutual information is always non-negative, $\delta \leq I ( \varepsilon ; c , y )$ provides an upper bound. To obtain a more precise approximation of $\delta _ { ; }$ we expand $I ( \varepsilon ; c , y )$ via the chain rule:

$$
I ( \varepsilon ; c , y ) = I ( \varepsilon ; c ) + I ( \varepsilon ; y \mid c )\tag{54}
$$

Under the assumption that category c and year y are independent, $I ( \varepsilon ; y | c ) \approx I ( \varepsilon ; y )$ , giving:

$$
\delta \leq I ( \varepsilon ; c ) + I ( \varepsilon ; y )\tag{55}
$$

This shows that $\delta$ is controlled by how much the noise ε leaks information about each label separately. If the model successfully disentangles $\mu ^ { c }$ and $\mu ^ { y }$ from ε, both terms vanish and $\delta \to 0$ , recovering the ideal case:

$$
I ( \nu ; c , y ) \approx I ( \mu ^ { c } ; c ) + I ( \mu ^ { y } ; y )\tag{56}
$$

Thus, under the assumption that such orthogonal centroids can be learned (see Figure 5), the ideal mutual information equality in the practical case is consistent with the TDIR proposition up to an upper bound $\delta$ characterized by the entanglement of training data.

The joint optimization of $\mathcal { L } ^ { c }$ and $\mathcal { L } ^ { y }$ implicitly enforces the orthogonality condition $\langle \mu ^ { c } , \mu ^ { y } \rangle = 0$ required by the TDIR definition, without imposing it as an explicit constraint. The proof relies on a preliminary result about the structure of the year proxies, which we establish first.

## A.4 TDIR as an Emergent Property of the Joint Loss

With all the properties and propositions derived above, we can then state that the following theorem holds:

Theorem 3 (Emergent TDIR). Let $\mathcal { L } ^ { c }$ and $\mathcal { L } ^ { y }$ be the proxy loss functions defined over an embedding space $\mathbb { R } ^ { d }$ . Then:

Part I (exact case). $I f I ( \varepsilon ; c , y ) = 0 ,$ joint minimization of $\mathcal { L } = \mathcal { L } ^ { c } + \mathcal { L } ^ { y }$ implies TDIR:

$$
\operatorname* { m i n } _ { \theta } \mathcal { L } ^ { c } + \mathcal { L } ^ { y } \implies p ( \mu ^ { c } , \mu ^ { y } , c , y ) = p ( \mu ^ { c } , c ) \cdot p ( \mu ^ { y } , y )\tag{57}
$$

Part II (asymptotic case). $I f I ( \varepsilon ; c , y ) \neq 0 ;$ joint minimization implies asymptotic TDIR:

$$
\operatorname* { m i n } _ { \theta } \mathcal { L } ^ { c } + \mathcal { L } ^ { y } \implies \langle \mu ^ { c } , \mu ^ { y } \rangle \to O \bigg ( \frac { I ( \varepsilon ; c , y ) } { \| \mu ^ { y } \| ^ { 2 } } \bigg ) \quad \forall c , y\tag{58}
$$

Proof. Part I: $I ( \varepsilon ; c , y ) = 0 \Longrightarrow$ TDIR.

The category proxy loss is minimized when the score $\nu ^ { \top } \mu ^ { c }$ is maximal and uniform across all years. Since $\nu _ { A } ^ { c } = \mu ^ { c } + \mu ^ { y _ { A } }$ and $\nu _ { B } ^ { c } = \mu ^ { c } + \mu ^ { y _ { B } }$ share the same category but differ in year, the minimum requires:

$$
\nu _ { A } ^ { \top } \mu ^ { c } = \nu _ { B } ^ { \top } \mu ^ { c } \quad \forall y _ { A } , y _ { B } \implies \langle \mu ^ { y _ { A } } , \mu ^ { c } \rangle = \langle \mu ^ { y _ { B } } , \mu ^ { c } \rangle \quad \forall y _ { A } , y _ { B }
$$

hence $I ( \mu ^ { c } ; y ) = 0$ . For the temporal loss, the shared displacement vectors $k ^ { y }$ impose, for any two categories $c \neq c ^ { \prime }$ sharing the same years:

$$
r _ { A } ^ { c } + k _ { B } ^ { y } \approx \mu ^ { y _ { B } } , \qquad r _ { A } ^ { c ^ { \prime } } + k _ { B } ^ { y } \approx \mu ^ { y _ { B } }
$$

Subtracting yields $r _ { A } ^ { c } \approx r _ { A } ^ { c ^ { \prime } }$ , so the score $r _ { A } ^ { c ^ { \top } } \mu ^ { y }$ is independent of the category, hence $I ( \mu ^ { y } ; c ) =$ 0. Applying the chain rule of mutual information:

$$
I ( \mu ^ { c } , \mu ^ { y } ; c , y ) = I ( \mu ^ { c } ; c ) + \underbrace { I ( \mu ^ { y } ; c \mid \mu ^ { c } ) } _ { = 0 } + \underbrace { I ( \mu ^ { c } ; y \mid \mu ^ { y } , c ) } _ { = 0 } + I ( \mu ^ { y } ; y ) = I ( \mu ^ { c } ; c ) + I ( \mu ^ { y } ; y )
$$

This equality is equivalent, via the KL divergence and Fubini's theorem, to the factorization $p ( \mu ^ { c } , \mu ^ { y } , c , y ) = p ( \mu ^ { c } , c ) \cdot p ( \mu ^ { y } , y ) ,$

Part II: $I ( \varepsilon ; c , y ) \neq 0 \implies$ asymptotic TDIR.

The proxy loss $\mathcal { L } ( u , \mu ^ { * } ; \mathcal { P } ) = - \log { \sigma ( u ^ { \top } \mu ^ { * } ) }$ is a softmax cross-entropy, which is smooth and strictly convex in a neighborhood of its minimum. A second-order Taylor expansion around the minimizer yields a local quadratic approximation of the form $\| u - \mu ^ { * } \| ^ { 2 } + O ( \| u -$ $\mu ^ { * } \| ^ { 3 } )$ . We therefore adopt the squared-error surrogates as a standard local approximation:

$$
\begin{array} { r } { \mathcal { L } ^ { c } ( T ) \propto \| \nu _ { A } - \mu ^ { c } \| ^ { 2 } + \| \nu _ { B } - \mu ^ { c } \| ^ { 2 } , } \end{array}\tag{59}
$$

$$
\begin{array} { r } { \mathcal { L } ^ { y } ( T ) \propto \| \hat { r } _ { A } - \mu ^ { y _ { A } } \| ^ { 2 } + \| \hat { r } _ { B } - \mu ^ { y _ { B } } \| ^ { 2 } + \| \hat { r } _ { A  B } - \mu ^ { y _ { B } } \| ^ { 2 } + \| \hat { r } _ { B  A } - \mu ^ { y _ { A } } \| ^ { 2 } . } \end{array}\tag{60}
$$

This approximation is tight near convergence, where embeddings cluster around their respective proxies, and the conclusions below hold to the same asymptotic order as the Taylor remainder.

Decompose $\mu ^ { c }$ over the temporal subspace as $\mu ^ { c } = \tilde { \mu } ^ { c } + \alpha \mu ^ { y _ { A } } + \beta \mu ^ { y _ { B } }$ with $\tilde { \mu } ^ { c } \perp \mu ^ { y }$ Under $\boldsymbol { \nu } = \boldsymbol { \mu } ^ { c } + \boldsymbol { \mu } ^ { y } + \boldsymbol { \varepsilon }$ , the year residuals are:

$$
r _ { A } ^ { c } = ( 1 - \alpha ) \mu ^ { y _ { A } } - \beta \mu ^ { y _ { B } } + \varepsilon _ { A } ^ { c } , \qquad r _ { B } ^ { c } = ( 1 - \beta ) \mu ^ { y _ { B } } - \alpha \mu ^ { y _ { A } } + \varepsilon _ { B } ^ { c }
$$

Substituting into the loss surrogates to $\mathcal { L } ^ { y }$ and expanding:

$$
\begin{array} { r l } & { \mathcal { L } ^ { y } \propto + \| \varepsilon _ { A } ^ { c } \| ^ { 2 } + \| \varepsilon _ { B } ^ { c } \| ^ { 2 } - 2 \alpha \langle \mu ^ { y _ { A } } , \varepsilon _ { A } ^ { c } + \varepsilon _ { B } ^ { c } \rangle - 2 \beta \langle \mu ^ { y _ { B } } , \varepsilon _ { A } ^ { c } + \varepsilon _ { B } ^ { c } \rangle + } \\ & { \quad \quad + \underbrace { 2 \alpha ^ { 2 } \| \mu ^ { y _ { A } } \| ^ { 2 } + 2 \beta ^ { 2 } \| \mu ^ { y _ { B } } \| ^ { 2 } + 4 \alpha \beta \langle \mu ^ { y _ { A } } , \mu ^ { y _ { B } } \rangle } _ { Q ( \alpha , \beta ) \ge 0 } } \end{array}
$$

The quadratic form $Q ( \alpha , \beta )$ is positive semidefinite by Cauchy-Schwarz. The cross terms displace the minimum from (0,0) to:

$$
\alpha ^ { * } = \frac { \langle \mu ^ { y _ { A } } , \varepsilon _ { A } ^ { c } + \varepsilon _ { B } ^ { c } \rangle } { 2 \| \mu ^ { y _ { A } } \| ^ { 2 } } + O ( \varepsilon ^ { 2 } ) , \qquad \beta ^ { * } = \frac { \langle \mu ^ { y _ { B } } , \varepsilon _ { A } ^ { c } + \varepsilon _ { B } ^ { c } \rangle } { 2 \| \mu ^ { y _ { B } } \| ^ { 2 } } + O ( \varepsilon ^ { 2 } )
$$

For small correlations, the covariance between ε and $\mu ^ { y }$ satisfies $\operatorname { C o v } ( \varepsilon , \mu ^ { y } ) \approx \| \varepsilon \| \cdot \| \mu ^ { y } \|$ $\sqrt { 2 I ( \varepsilon ; c , y ) }$ , which gives:

$$
\langle \mu ^ { c } , \mu ^ { y } \rangle = \alpha ^ { * } \| \mu ^ { y } \| ^ { 2 } + O ( \varepsilon ^ { 2 } ) = O \bigg ( \frac { I ( \varepsilon ; c , y ) } { \| \mu ^ { y } \| ^ { 2 } } \bigg )
$$

Global consistency follows from the shared $k ^ { y }$ a single $k _ { B } ^ { y }$ cannot simultaneously compensate projections $\alpha \neq \alpha ^ { \prime }$ across categories unless $( \alpha - \alpha ^ { \prime } ) \mu ^ { y _ { A } } = \varepsilon _ { A } ^ { c ^ { \prime } } - \varepsilon _ { A } ^ { c }$ , which cannot hold for all category pairs since ε is not a control variable of the optimizer. The unique globally consistent solution is therefore:

$$
\langle \mu ^ { c } , \mu ^ { y } \rangle \to O \left( \frac { I ( \varepsilon ; c , y ) } { \| \mu ^ { y } \| ^ { 2 } } \right) \quad \forall c , y
$$

Corollary 4 (Swapping Trick Enforces Cross-Category Orthogonality). Under the conditions of Theorem 3, the shared displacement vectors $k ^ { y }$ enforce $\langle \mu ^ { c } , \mu ^ { y } \rangle \to 0$ uniformly across all categories.

Since the same $k _ { B } ^ { y }$ is used when translating residuals from any category, satisfying the swapping constraint simultaneously for two categories $c \neq c ^ { \prime }$ with residual projections $\alpha \neq$ $\alpha ^ { \prime }$ onto $\mu ^ { y }$ would require:

$$
\left( \alpha - \alpha ^ { \prime } \right) \mu ^ { y _ { A } } = \varepsilon _ { A } ^ { c ^ { \prime } } - \varepsilon _ { A } ^ { c } \qquad \forall c , c ^ { \prime } .\tag{61}
$$

This cannot hold globally, since ε is not a control variable of the optimizer. The unique globally consistent solution is therefore $\alpha = \alpha ^ { \prime } = 0$ for all $c , c ^ { \prime } .$ giving:

$$
\operatorname { P r o j } _ { \mu ^ { y } } ( \mu ^ { c } )  0 \qquad \forall c , y .\tag{62}
$$

## B Photographic material calculation

To estimate the proportion of photographic material in large archival collections, we surveyed three major public archives.

Archives Nationales (France). Using the publicly available digitisation corpus dataset [], a small available sample drawn from the 2025 digitisation batch was analysed. Of the total documents, 123 were non-textual, representing 23% of the sample (123/0.23 ≈ 535 documents in total). Among those non-textual items, 68 were strictly photographic, accounting for approximately 13% of the full sample.

Library of Congress (USA). The collection can be divided into digitised and non-digitised holdings []. Among the non-digitised material, the catalogue lists 16M books, 3M newspapers, 1M periodicals, 0.5M manuscripts, 1.3M photographic and pictorial items, and 450M audiovisual items. Among the digitised material, it lists 3M newspapers, 0.7M books, 0.5M manuscripts, 0.5M periodicals, and 1.2M photographs. Focusing on the textual-adjacent holdings (i.e. excluding the bulk audiovisual collection, which is a category of its own), photographic material represents:

$$
\frac { 1 . 3 + 1 . 2 } { 1 6 + 3 + 1 + 0 . 5 + 1 . 3 + 3 + 0 . 7 + 0 . 5 + 0 . 5 + 1 . 2 } \approx \frac { 2 . 5 } { 2 7 . 7 } \approx 9 \% .
$$

Portal de Archivos Españoles (Spain). The portal reports approximately 40 million digital objects in total []. According to [], roughly 2 million of these are classified as photographic objects (excluding videos, posters, and other non-textual formats), yielding a photographic fraction of approximately $2 / 4 0 = 5 \%$

## C Object-Centric Analysis

The per-object analysis reveals two complementary sources of error in the composed retrieval framework, illustrated in Figure 6. When considering the reference image (the image from which temporal information is extracted), objects with low Year MAE such as tires, shorts, glasses, and human faces act as reliable temporal vessels: their embeddings carry a strong and unambiguous date signal, suggesting minimal entanglement between their categorical and temporal subspaces. Conversely, objects such as wheels, vehicle registration plates, dresses, buildings, and men exhibit higher reference error, indicating that their visual representations encode date information less cleanly, likely due to higher intra-class visual variance across decades. On the target side (the object whose category should be preserved after temporal transplantation), objects like posters, girls, trains, houses, and dresses are retrieved with relatively low temporal error, while ladders, human faces, tires, wheels, and hats show the highest MAE. This asymmetry suggests that temporally ambiguous objects are doubly penalised: they are poor sources of date information and poor recipients of temporal injection.

Figure 7 reports per-object precision across all three inference modes using radar plots, for both object retrieval (top row) and date estimation (bottom row). In the object retrieval setting, the largest errors concentrate on categories that are visually sub-specific or easily confused with related classes: glasses, vehicle registration plates, and items of clothing such as socks and footwear tend to share visual features with broader categories, creating spurious correlations that degrade categorical fidelity. For date estimation (bottom row), the Label+Label mode achieves strong and consistent performance across virtually all object categories, confirming that the proxy-based subspace structure encodes temporal information reliably when labels are directly available. However, when queries are issued through images, performance degrades selectively for objects that lack distinctive temporal signatures, such as generic architectural elements or accessories, producing a marked and category-specific drop in precision.

![](images/3eadeb541ca212cc9b2ecb4a7c152b750600c8af2e06135a53a37928811e0772.jpg)  
Figure 6: Average Year Error decomposed by reference object (top) and target object (bottom) in the Image+Image inference mode. Reference objects with low error are reliable vessels for date injection, indicating clean category representations with low temporal entanglement. Target objects with high error are temporally ambiguous, making them difficult to anchor to a specific period regardless of the source image used.

The temporal behaviour of each inference mode is further analysed in Figure 8. Date estimation precision degrades toward earlier decades for both Image+Label and Image+Image modes, consistent with the sparser representation of pre-1950 material in the DEW dataset. Notably, the Image+Image mode exhibits a reversal of the general trend observed in labelbased settings: whereas date precision typically exceeds object precision under Label+Label and Image+Label queries, the Image+Image mode yields stronger object than date precision. This suggests that the categorical subspace is more robust to temporal transplantation than the temporal subspace is to cross-image injection, a finding aligned with the theoretical guarantee of Proposition 3, which ensures category preservation under orthogonal decomposition but imposes no analogous bound on the fidelity of the transplanted temporal signal.

Figures 9 and 10 decompose the performance drop from Image+Label to Image+Image inference on a per-object basis, for date estimation and object retrieval respectively. Object retrieval degradation is more uniformly distributed across categories, suggesting that the category subspace remains largely stable under temporal transplantation for most classes, with isolated exceptions corresponding to categories with high inter-class visual overlap.

![](images/13765604ea3ca79befd42bfc30908bcfcae83c1963cc8a4b65dcda120b00dfaf.jpg)

![](images/73f7bd8ba0da012e15de55266d17c31217c3e31d49002b4220e458ceaf9d53ff.jpg)

![](images/85931fabe72e1d8977fd07d86c847369d332215580b720d03dc8e6db29ef3a5d.jpg)

![](images/ad062f9e2468555a4bd03545ac0b0783b7ac40ebcb0a6756fba6009bb5233234.jpg)

![](images/74c57b7650ceb20346e87c383dd43a512020a56e1e25541eb31f546d739271ba.jpg)

![](images/cbb08acc5e6849b4529e9daf9a02fabd1f71669aa185582cc2f60bac3c0fe2d0.jpg)  
Figure 7: Per-object precision radar plots for object retrieval (top row) and date estimation (bottom row) across Label+Label, Image+Label, and Image+Image inference modes (left to right). Object retrieval errors concentrate on visually sub-specific categories such as glasses, vehicle registration plates, and footwear, which share features across class boundaries. For date estimation, Label+Label achieves consistently strong performance, while image-based modes reveal a pronounced and category-dependent performance gap for temporally ambiguous objects.

![](images/7ac62fcccb6d98d7160096e9b6997b128dec9203fa9b4b1e9af1a448eab387d0.jpg)

![](images/2968a65c244ee7a63e1b4a33c4a86c73ed373510796d55f04e5ab00db0aa2697.jpg)  
Figure 8: Left: date estimation precision per source year across inference modes. Both Image+Label and Image+Image exhibit degraded performance for earlier decades, with Image+Label remaining more stable across the temporal range. Right: scalar summary of object and date precision per inference mode. In Image+Image retrieval, the relationship between object and date precision inverts relative to label-based modes, indicating that temporal injection is a stricter operation than category preservation under the learned decomposition.

![](images/adea6fdf31cfaee1effc240d03a7828a7135a7cf42047aa46d48f18f395dbe01.jpg)  
Figure 9: Per-object drop in date retrieval precision.

![](images/ef766b4295881742aaa7fa606b10de229868d4f3158351e4071b9d37b72f1061.jpg)  
Figure 10: Per-object drop in object retrieval precision.

## D CLIP Baseline

Both CLIP []] baselines use OpenCLIP [] ViT-B/32 [] pre-trained on LAION-2B [B0] as a frozen encoder, with no fine-tuning on DEW or on any historical imagery. All image embeddings are l2-normalised to unit norm prior to any arithmetic operation or nearestneighbour query, following the same convention as TDIR. A single Annoy index 4of $N _ { \mathrm { t r e e s } } = 5 0$ trees is built over the CLIP vision embeddings of the full training split (identical images to those used for TDIR training), so that retrieval pool and gallery are exactly matched across methods. Nearest-neighbour search is performed in Euclidean space over the unit hypersphere, which is equivalent to cosine similarity for $\ell _ { 2 } { \mathrm { - n o r m a l i s e d } }$ vectors. The top $. K = 1 0$ neighbours are returned for every query, and precision is computed identically to Section 4.3.

## D.1 CLIP Arithmetic Baseline

The arithmetic baseline replaces the learned proxy vectors $\mu ^ { c }$ and $\mu ^ { y }$ of TDIR with CLIP text embeddings, exploiting the joint vision-language alignment acquired during pre-training For each object category c we encode the prompt $" A$ photo of ${ \mathit { f c } } { \mathit { J } } ^ { \prime \prime } .$ , and for each target year y (bucketed in 5-year lustrum bins from 1930 to 1999, yielding 14 bins) we encode "A photo in $\{ \mathrm { y } e a r \} ^ { \prime \prime } .$ All text embeddings are l2-normalised after encoding, producing a category proxy matrix $T ^ { c } \in \mathbb { R } ^ { C \times d }$ and a year proxy matrix $T ^ { y } \in \mathbb { R } ^ { Y \times d }$ , with $d = 5 1 2$ for ViT-B/32. These matrices serve as drop-in substitutes for $P ^ { c }$ and $P ^ { y }$ in all three inference modes of Section 3.5, with no modification to the retrieval procedure.

Label+Label. The query vector is the un-normalised sum $q = T _ { c } ^ { c } + T _ { y } ^ { y }$ , directly mirroring the proxy-sum query $\boldsymbol { q } = \boldsymbol { \mu } ^ { c } + \boldsymbol { \mu } ^ { y }$ of TDIR.

Image+Label. The query is $q = V ( x _ { c } ) + T _ { y } ^ { y }$ , where $V ( x _ { c } )$ is the unit-norm CLIP vision embedding of the category image.

Image+Image (zero-label). The reference image $x _ { y }$ is sampled uniformly from the training index. Its inferred category $\hat { c }$ is obtained by argmax cosine similarity against all category text embeddings, $\hat { c } = \arg \operatorname* { m a x } _ { c ^ { \prime } } V ( x _ { y } ) ^ { \top } T _ { c ^ { \prime } } ^ { c }$ . The temporal residual is then extracted as $r _ { y } = V \big ( x _ { y } \big ) - T _ { \hat { c } } ^ { c }$ , and the final query is $q = V \big ( x _ { c } \big ) + r _ { y }$

The poor date precision reported in Table 2 for this baseline is informative: despite CLIP embeddings carrying rankable temporal information [B2], the arithmetic composition fails to disentangle temporal and categorical signals because CLIP was never trained to produce orthogonal category and year subspaces. The residual $r _ { y }$ retains substantial categorical content, and the year text prompts encode only a weak and diffuse notion of historical period that does not translate into a geometrically consistent displacement on the visual embedding manifold. This confirms that temporal composability is not an emergent property of vision-language pre-training, but requires the explicit geometric structure imposed by TDIR.

## D.2 CLIP Prompting Baseline

The prompting baseline replaces vector arithmetic with a two-stage retrieve-and-rerank strategy. Rather than composing both signals into a single query vector, retrieval is first performed using the primary modality alone, and the resulting candidate pool is then re-ranked by cosine similarity to the secondary modality signal. This avoids relying on cross-modal vector addition and instead lets each signal operate independently on the unit hypersphere.

Concretely, a candidate pool of size $P = 1 0 0$ is retrieved by Annoy nearest-neighbour search using the primary modality embedding. The stored embeddings of all pool candidates are then re-scored by their cosine similarity to the secondary modality embedding, and the top $. K = 1 0$ candidates after re-scoring are returned.

Label+Label. The candidate pool is retrieved by $T _ { c } ^ { c }$ alone; pool candidates are then reranked by cosine similarity to $T _ { y } ^ { y }$

Image+Label. The candidate pool is retrieved by $V ( x _ { c } )$ ; re-ranking uses $T _ { y } ^ { y }$ . The pool is computed once per test image and reused for all 14 year bins, amortising the retrieval cost across the full label set.

Image+Image. The candidate pool is retrieved by $V ( x _ { c } )$ ; re-ranking uses $V ( x _ { y } )$ directly, replacing the residual $r _ { y }$ with the raw reference image embedding. This sidesteps the categoryinference step and provides a stronger re-ranking signal when category and date co-vary in the reference image.

As shown in Table 2, the prompting baseline recovers competitive object precision across all three modes, confirming that CLIP vision embeddings carry strong categorical signal. Date precision, however, remains substantially below TDIR for all inference modes even under reranking. This gap persists because reranking by $T _ { y } ^ { y }$ or $V ( x _ { y } )$ can only promote candidates that already appear in the pool retrieved by the category signal, and nothing in the CLIP representation space guarantees that temporally relevant images rank highly under a category-only query. The contrast between the two baselines and TDIR thus isolates the contribution of the learned orthogonal decomposition: without it, object and temporal retrieval remain fundamentally entangled regardless of the composition strategy employed.

## E Frequently Asked Questions (FAQs)

In this section, readers will find a series of answers to questions raised during peer review, discussions, and others. We hope they cana be useful in interpreting the data and results presented in the paper, as well as the method itself.

## E.1 Joint metric: Def. 1's independence assumption holds empirically.

We added Precision@K requiring both category and date correct simultaneously to all three retrieval modes, letting us test directly (rather than assume) whether treating category and date as independent (Def. 1) actually holds at inference. On Label+Label $( \mathrm { K } { = } 1 0$ , a stratified 7 407-crop pool):

$$
\mathrm { O b j ~ P } \ @ 1 0 = 0 . 4 8 8 \quad \mathrm { ~ D a t e ~ P \ @ 1 0 = 0 . 7 1 8 ~ } \quad \mathrm { J o i n t ~ P \odot 1 0 = 0 . 3 4 9 }
$$

![](images/bf71cd708647f2bee8fea628530aae59ca63325b944303681b9f787c26bb25bd.jpg)  
Figure 11: Category-year cosine similarity across all 14 year proxies for the two most orthogonal (Window, Bus) and the most entangled (Footwear) of the 50 categories.

against $0 . 4 8 8 \times 0 . 7 1 8 = 0 . 3 5 0$ predicted under independence, a 0.4% relative gap well within noise. Object- and date-correctness are therefore empirically independent: neither axis is traded off against, nor rides for free on, the other, exactly the behaviour Def. 1 and Thm. 1's joint optimisation are designed to produce.

## E.2 Proxy geometry corroborates this at the representation level.

We further computed the full cosine matrix between the 50 category and 14 year proxies of the reported ConvNeXt+TDIR checkpoint (its cached index reproduces Table 2/3 exactly). Cross-block mean $| \cos | = 0 . 2 0 4$ , close to the 0.025 floor for random unit vectors in $\mathbb { R } ^ { 1 0 2 4 }$ at this scale. Fig. 1 shows both regimes side by side across all 14 year proxies: 27/50 categories (e.g. Window, Bus) sit at that floor throughout, genuinely orthogonal as Thm. 1 predicts in the exact case. A smaller set of \~15 categories with a plausible intrinsic appearance-era link (tie, belt, watch, sock, footwear,...) shows the bounded, data-dependent residual δ that Prop. 2 anticipates for the realistic case: e.g. Footwear peaks at +0.87 in 1930 and is 3–6× weaker by 1965 on. This residual tracks training-set size almost exactly (per-year mean | cos | vs. log sample count: $r = - 0 . 9 0 , p { < } 1 0 ^ { - 4 }$ ; 1930: 14k crops → 0.44; 1985: 108k crops → 0.11), giving δ a precise, measured, localized explanation rather than an abstract constant.

## E.3 Baselines and Previous Work

Modern SSL pretraining (DINOv3). Because DINOv3 is not a specific architecture but a pretraining regime, we have loaded DINOv3 weights into the same ConvNeXt-Base backbone used throughout (Table 2) and train under the identical 50-category protocol. At epoch 1 already, cross-block mean $| \cos | = 0 . 1 7 3$ , already at or below the fully-converged ImageNet-supervised value (0.204), with the same ordinal decay replicating: the decomposition is therefore not an artifact of ImageNet-supervised pretraining.

On the suggested [, 2] baselines. We are well aware of both []'s smooth-nDCG loss needs a relevance function derived from a numerical variable: it can replace our proxy loss on the year component, but never on the category, so it does not serve as a composed-retrieval loss. However, further results may incorporate such loss as date proxy loss, but the method itself is not capable of composing concepts. And, on the other hand, [] is incompatible with our setup by any means: it runs object detection over multi-object images and requires a dedicated encoder per object label to predict a date (in [] the category is not learned, it is passed as input).

## E.4 Ordinal structure & unseen years.

Cosine similarity between two year proxies decays smoothly with their temporal gap, although only pairwise transitivity (Def. 2) was ever enforced at training time:

![](images/649d6f47e4f97b7cd1b4d734a09b2cf723e654214207527a8c63c62b395d2915.jpg)

Leave-one-out check: a year's true proxy is consistently closer to the midpoint of its two neighbours than to either neighbour alone (e.g. cos = 0.66 vs. 0.57 at 1935; all 12 interior years), so the subspace already extrapolates locally.

Objects with weak temporal signature. The categories with the lowest year-entanglement above (window, bus, door, boat, tower) are the slow-changing, architecture-type categories that one may flag as barely changing across decades. This is actually consistent, not contradictory, since their proxies correctly encode little date information because their appearance carries little.