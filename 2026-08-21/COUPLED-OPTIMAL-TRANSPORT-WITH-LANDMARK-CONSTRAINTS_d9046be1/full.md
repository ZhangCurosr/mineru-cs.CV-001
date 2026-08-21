# COUPLED OPTIMAL TRANSPORT WITH LANDMARK CONSTRAINTS

XIANG GU<sup>∗</sup>, JIAN SUN <sup>∗</sup>, AND ZONGBEN XU<sup>∗</sup>

Abstract. Existing optimal transport (OT) models primarily seek an OT map or plan between distributions by minimizing a prescribed transport cost or distortion. However, minimizing transport cost or distortion alone may fail to identify a geometrically meaningful transformation between the two distributions. To address this limitation, this paper proposes a novel coupled OT framework that leverages a small number of annotated landmarks to guide the recovery of an underlying deformation governing the distribution transformation. The coupled OT framework integrates the optimization of the transport plan and the deformation field into a unified model, where the landmark-guided deformation field and the cost-driven transport plan are coupled through a mutual-consistency constraint. As a result, the deformation is jointly determined by the annotated landmarks and cost-driven distribution matching. The proposed framework provides a principled connection between landmark-based registration and transport-based distribution matching, enabling the recovery of transport maps from sparse geometric supervision. We establish the well-definedness of the proposed model in a general variational setting and develop a finite-element-based numerical algorithm for computation whose convergence properties are systematically analyzed. The practical efectiveness of the proposed approach is verified in shape matching.

Key words. optimal transport, variational model, landmark constraint, deformation field, transport-deformation coupling

MSC codes. 49Q22, 65K10

1. Introduction. Optimal Transport (OT), as a geometric tool for comparing or connecting distributions, has been successfully applied in diverse domains, such as astrophysics [12], machine learning [9, 2, 22], image science [4, 27, 15], and biology [26, 19]. OT aims to derive a transport map or plan between two probability distributions such that a certain transport cost is minimized. OT has two typical formulations, i.e., the Monge formulation [24] and the Kantorovich formulation [17]. The Monge formulation optimizes among a set of continuous transport maps with a masspreserving constraint, which is nonconvex and may be infeasible. The Kantorovich formulation relaxes the Monge formulation as a coupling or plan optimization problem and attracts broader studies in applications. The original Kantorovich problem is a linear program that is computationally expensive in its discrete version. The entropyregularized OT [10] introduces the entropy of the transport plan as a regularization to the OT model, which is solved by the computationally cheaper Sinkhorn-Knopp algorithms [10, 1, 18]. The original OT model has been extended in several aspects. To tackle the applications where only partial mass should be transported, partial OT [16, 11, 5, 7], unbalanced OT [21, 8], robust OT [3, 25], and supervised OT [6] models have been proposed. On the other hand, to deal with the situations where the two measures are supported on diferent spaces and the transport cost is hard to define, Gromov-Wasserstein models [28, 23] have been proposed to minimize the transport distortion defined on metrics within each space, respectively.

The previous OT models mainly seek the transport map or plan under the criteria of transport cost or distortion minimization. The transport cost is often taken to be a distance, and the distortion is typically set as the diference between pairwise distances. However, such criteria may not faithfully characterize a geometrically meaningful distribution transformation. For example, in image or shape registration, two regions may be close but correspond to diferent semantic or anatomical parts. A transport map driven solely by distance minimization may therefore move mass along geometrically short but semantically incorrect paths, leading to mismatched correspondences and inaccurate descriptions of the true distribution deformation. A promising solution for this challenge is to leverage a small number of pre-annotated landmarks, which can typically be obtained at relatively low annotation cost. These landmarks provide sparse geometric information about the deformation between distributions and can guide a meaningful transport. Building on this idea, [13, 14] present a keypoint-guided OT model that introduces a mask-based constraint on the transport plan and a relation-preserving strategy to guide the matching of points, pioneering the study of landmark-driven OT. While inspiring, keypoint-guided OT targets the matching between discrete points and does not explicitly recover the deformation underlying the transition of distributions. This becomes restrictive in many applications such as image registration, shape analysis, and morphometric modeling, where one often seeks a deformation field corresponding to the distributional transition.

In this paper, we propose a novel coupled OT framework that utilizes sparse geometric landmarks to guide the recovery of the transformation between distributions. Our framework introduces a deformation-based view of transport: the transport plan should not only match the two distributions, but also be consistent with a coherent deformation suggested by the annotated landmarks. Under this spirit, we integrate the searching of the transport plan and the estimation of the deformation field into a unified coupled model. In this model, landmarks serve as local geometric anchors for the deformation field, which guides the transport behavior and is, in turn, constrained by the transport plan obtained through cost minimization. Consequently, the resulting transport is determined jointly by global cost minimization and a landmarkguided deformation that preserves local geometry. Our framework provides a principled bridge between landmark-based registration and transport-based distribution matching, enabling the recovery of a geometrically meaningful transport guided by sparse landmark supervision. Following this model, extensive theoretical analysis is presented to establish its well-definedness and characterize its connection to conventional OT. We also devise a numerical algorithm based on the finite-element method to solve the model. The efectiveness of the proposed method is verified in the shape matching application.

Notations. We use $\mu _ { n }  \mu$ to denote narrow convergence for probability measures $\mu _ { n }$ and $\mu . \mathrm { ~ } u _ { n } $ u denotes weak convergence for functions $u _ { n }$ and u in the underlying function space and the strong convergence is denoted by $u _ { n } \to u . \ L ^ { q } ( \Omega ; \mathbb { R } ^ { p } )$ denotes the Lebesgue space of $\mathbb { R } ^ { p . }$ -valued functions with respect to the Lebesgue measure on $\Omega ,$ while $L ^ { q } ( \boldsymbol { \mu } ; \mathbb { R } ^ { p } )$ denotes the Lebesgue space with respect to the measure $\mu .$ $H ^ { 1 } ( \Omega ; \mathbb { R } ^ { p } )$ denotes the first-order Sobolev space of $\mathbb { R } ^ { p . }$ -valued functions with squareintegrable weak derivatives. The codomain R is omitted for scalar-valued function spaces. For an element $K$ of a Cartesian grid, $\mathbb { Q } _ { 1 } ( K )$ denotes the space of afine functions on $K$ . For a nonempty closed set ${ \mathcal { C } } _ { : }$ , its extended-valued indicator function $\iota _ { \mathcal { C } }$ is defined to be $0$ on $\mathcal { C }$ and +∞ outside C. We denote by $N _ { \cal C } ( x )$ the normal cone to $\mathcal { C }$ at x, by $\partial \phi ( x )$ the limiting subdiferential of an extended-real-valued function $\phi ,$ and by dist $( x , { \mathcal { C } } ) : = \operatorname* { i n f } _ { y \in { \mathcal { C } } } \| x - y \|$ the distance from $x$ to $\mathcal { C } .$ For matrices, $\langle \cdot , \cdot \rangle _ { F }$ and $\| \cdot \| _ { F }$ denote the Frobenius inner product and norm, respectively. The vector of all ones in $\mathbb { R } ^ { n }$ is denoted by ${ \bf 1 } _ { n }$

2. Coupled OT model with landmark constraints. In this section, we first review the background of the classical OT model, then introduce our proposed coupled

OT model with landmark constraints, followed by its mathematical properties.

2.1. Background. Let $( \mathcal { X } , d )$ be a Polish space, and let ${ \mathcal { P } } ( \Omega )$ denote the space of Borel probability measures on $\Omega \subset { \mathcal { X } }$ . Given two probability measures $\mu , \nu \in \mathscr { P } ( \Omega )$ 2 OT aims to find the most cost-eficient way of moving mass of $\mu$ onto mass distributed according to $\nu .$ Let $c : \Omega \times \Omega  [ 0 , + \infty ]$ be a lower semicontinuous cost function, the Kantorovich formulation of the OT problem is defined as

$$
\operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int c ( x , y ) \mathrm { d } \pi ( x , y ) ,\tag{2.1}
$$

where $\Pi ( \mu , \nu )$ denotes the set of all couplings between $\mu$ and $\nu ,$ namely

$$
\Pi ( \mu , \nu ) : = \left\{ \pi \in \mathcal P ( \Omega \times \Omega ) : \int \pi ( x , y ) \mathrm { d } y = \mu ( x ) , \int \pi ( x , y ) \mathrm { d } x = \nu ( y ) \right\} .
$$

Any minimizer $\pi _ { \star } \in \Pi ( \mu , \nu )$ , when it exists, is called an optimal transport plan. In the special case where $c ( x , y ) = d ^ { p } ( x , y )$ , the above problem induces a proper metric in ${ \mathcal { P } } ( \Omega )$ , namely the $p { - } \mathrm { W }$ asserstein distance, given by

$$
W _ { p } ( \mu , \nu ) = \left( \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int d ^ { p } ( x , y ) \mathrm { d } \pi ( x , y ) \right) ^ { 1 / p } .
$$

Thus, optimal transport provides both a variational principle for comparing probability measures and a geometric mechanism for identifying how mass should be matched or displaced between them.

2.2. Coupled OT model. In the classic OT model (2.1), the optimal transport plan is determined by the cost function. However, this may cause incorrect transport plans, as the cost function may not reflect the underlying geometrically meaningful deformation field governing the distributional variation. To address this issue, inspired by the idea of sparse-landmark supervision [13], we leverage a small number of annotated landmarks to assist the cost minimization criterion for identifying the deformation field. Let $\pmb { S } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { m }$ be the set of annotated landmark pairs, where $x _ { i }$ and $y _ { i }$ are observed sample points from $\mu$ and $\nu ,$ respectively, encoding sparse geometric information about the deformation field. The key challenge is how to integrate this sparse geometric information and the transport cost minimization prior into a principled model to recover the deformation field. We propose the coupled OT framework to tackle this challenge. Our basic idea is to use the landmarks to supervise a deformation field, together with a regularization energy that promotes spatial smoothness and propagates the sparse geometric information across the domain. Finally, the deformation field and the transport plan are coupled within a unified transport model, so that the cost-minimization prior and the sparse landmark supervision can jointly shape the deformation and the transport plan.

For simplicity and ease of exposition, we assume throughout this paper that Ω is a bounded Lipschitz domain in $\mathbb { R } ^ { p }$ , while noting that the proposed framework can be naturally extended to more general Banach spaces. For the convenience of description, we denote the landmark displacement by

$$
\Delta _ { i } ^ { \mathrm { l m } } = y _ { i } - x _ { i } ,
$$

which provides an observation of the underlying deformation around a prescribed landmark. Let $u : \Omega \to \mathbb { R } ^ { p }$ denote the deformation field. The associated transport map is given by

$$
T _ { u } = \mathrm { I d } + u .
$$

In general, we do not require u to be pointwise well-defined; for instance, u may belong only to a Sobolev space. Therefore, landmark information should be interpreted through a bounded observation operator $A _ { r } ^ { i }$ . A typical choice of $A _ { r } ^ { i }$ is the local averaging operator:

$$
A _ { r } ^ { i } ( u ) = \frac { 1 } { | B _ { r } ( x _ { i } ) \cap \Omega | } \int _ { B _ { r } ( x _ { i } ) \cap \Omega } u ( z ) \mathrm { d } z , \mathrm { ~ f o r ~ } i = 1 , \dots , m ,
$$

where $| \cdot |$ is the Lebesgue measure, and $r > 0$ is a small radius. When the deformation is suficiently regular and r is small, $A _ { r } ^ { i } ( u )$ approximates the local displacement around the landmark $x _ { i }$ . We take the admissible deformation space as

$$
\boldsymbol { \mathcal { U } } = \{ \boldsymbol { u } \in \boldsymbol { H } ^ { 1 } ( \Omega ; \mathbb { R } ^ { p } ) : \boldsymbol { u } = 0 \mathrm { ~ o n ~ } \Gamma _ { D } \} ,
$$

where $\Gamma _ { D } \subset \partial \Omega$ has positive boundary measure. This boundary condition is used to remove rigid-motion ambiguity. We note that other boundary conditions can also be adopted, provided that the corresponding null modes are properly controlled.

We now introduce the coupled OT model with landmark constraints:

$$
\operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) , u \in \mathcal { U } } \mathcal { F } ( \pi , u ) ,\tag{2.2}
$$

where

$$
\begin{array} { l } { \displaystyle \mathcal { F } ( \pi , u ) = \int \left[ ( 1 - \alpha ) c ( x , y ) + \alpha h \left( T _ { u } ( x ) - y \right) \right] \mathrm { d } \pi ( x , y ) } \\ { \displaystyle \qquad + \beta E _ { \mathrm { e l } } ( u ) + \sum _ { i = 1 } ^ { m } w _ { i } \rho \left( A _ { r } ^ { i } ( u ) - \Delta _ { i } ^ { \mathrm { l m } } \right) . } \end{array}\tag{2.3}
$$

Here $\alpha \in ( 0 , 1 ) , \beta \geqslant 0$ are tuning parameters, and $w _ { i } > 0$ are landmark weights. $c ( x , y )$ is a lower semicontinuous cost function. The third term incorporates the supervision of sparse landmarks to the deformation field. The function $\rho$ measures the discrepancy between the observation of the deformation field $A _ { r } ^ { i } ( u )$ and the true displacement based on landmarks $\Delta _ { i } ^ { \mathrm { l m } } . ~ \ : \ : \rho : \mathbb { R } ^ { p }  [ 0 , \infty ]$ is a lower semicontinuous function satisfying

$$
\rho ( 0 ) = 0 \mathrm { ~ a n d ~ } \operatorname* { i n f } _ { \| z \| _ { 2 } > \epsilon } \rho ( z ) > 0 , \forall \epsilon > 0 .\tag{2.4}
$$

Typical choices of $\rho$ include the quadratic loss $\rho ( z ) ~ = ~ \| z \| _ { 2 } ^ { 2 }$ , or some robust loss functions, such as the Huber loss, for cases with noisy landmarks.

The second term is a regularization energy to promote the spatial coherence of the deformation field $u .$ Here we use the linear elastic energy, but other regularizers, such as total variation, can also be applied. The linear elastic energy is defined by

$$
E _ { \mathrm { e l } } ( \boldsymbol { u } ) = \frac { 1 } { 2 } \int \left( \| \boldsymbol { \varepsilon } ( \boldsymbol { u } ) \| _ { F } ^ { 2 } + ( \nabla \cdot \boldsymbol { u } ) ^ { 2 } \right) \mathrm { d } x ,\tag{2.5}
$$

where $\begin{array} { r } { \varepsilon ( u ) = \frac { 1 } { 2 } \left( \nabla u + \nabla u ^ { \top } \right) } \end{array}$ is the linearized strain tensor, and $\nabla \cdot \boldsymbol { u }$ is the divergence of the deformation field. The elastic energy $E _ { \mathrm { e l } } ( u )$ describes the overall local deformation of $u ,$ including infinitesimal stretching, shearing, and volume change. By controlling the linearized strain and the divergence of $u ,$ this term penalizes local oscillations and promotes a spatially coherent deformation, facilitating the propagation of sparse landmark supervision across the domain.

The first integral term sits in the core part of the proposed model, where $h$ is a lower semicontinuous function for measuring the discrepancy, satisfying

$$
h ( 0 ) = 0 \mathrm { ~ a n d ~ } \operatorname* { i n f } _ { \| z \| _ { 2 } > \epsilon } h ( z ) > 0 \mathrm { ~ f o r ~ a n y ~ } \epsilon > 0 .\tag{2.6}
$$

Typically, h could be taken as the p-squared norm $h ( z ) = \| z \| _ { p } ^ { p }$ . In this term, for each possibly matched pair $( x , y )$ indicated by the transport plan π, we combine the classical transport cost $c ( x , y )$ with the deformation-consistency penalty $h ( T _ { u } ( x ) - y )$ The former encourages mass to be transported along geometrically short paths, while the latter requires the transported target $y$ to be consistent with the deformation map $T _ { u }$ applied to the source point x. Therefore, the transport plan π is not only optimized to minimize transport cost but also guided by the landmarks by the consistency with u. Meanwhile, it provides a regularization to the optimization of $u ,$ which enforces u to be consistent with the barycentric projection of $\pi .$

The following proposition provides a clearer explanation of the coupled OT model specified to the squared Euclidean distance cost.

Proposition 2.1. Assume $c ( x , y ) = \| x - y \| _ { 2 } ^ { 2 }$ for any $x , y \in \Omega , \rho ( z ) = h ( z ) =$ $\| z \| _ { 2 } ^ { 2 } ~ f o r$ any $z \in \mathbb { R } ^ { p }$ . Let $T _ { u } ^ { \alpha } = \mathrm { I d } + \alpha u$ , then we have

$$
\operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) , u \in \mathcal { U } } \mathcal { F } ( \pi , u ) = \operatorname* { i n f } _ { u \in \mathcal { U } } \mathcal { T } ( u )\tag{2.7}
$$

where

$$
( 2 . 8 ) \mathcal { I } ( u ) = W _ { 2 } ^ { 2 } ( [ T _ { u } ^ { \alpha } ] _ { \# } \mu , \nu ) + \alpha ( 1 - \alpha ) \| u \| _ { L ( \mu ; \mathbb { R } ^ { p } ) } ^ { 2 } + \beta E _ { \mathrm { e l } } ( u ) + \sum _ { i = 1 } ^ { m } w _ { i } \rho \left( A _ { r } ^ { i } ( u ) - \Delta _ { i } ^ { \mathrm { l m } } \right) ,
$$

and $T _ { \# } \mu$ is the pushforward measure defined by $T _ { \# } \mu ( A ) = \mu ( T ^ { - 1 } ( A ) )$ for any measurable set $A \subset \Omega$

Proof. First,

$$
\begin{array} { r l } & { ( 1 - \alpha ) c ( x , y ) + \alpha h ( T _ { u } ( x ) - y ) = ( 1 - \alpha ) \| x - y \| _ { 2 } ^ { 2 } + \alpha \| x + u ( x ) - y \| _ { 2 } ^ { 2 } } \\ & { \qquad = \| y - x - \alpha u ( x ) \| _ { 2 } ^ { 2 } + \alpha ( 1 - \alpha ) \| u ( x ) \| _ { 2 } ^ { 2 } } \\ & { \qquad = \| y - T _ { u } ^ { \alpha } \| _ { 2 } ^ { 2 } + \alpha ( 1 - \alpha ) \| u ( x ) \| _ { 2 } ^ { 2 } . } \end{array}
$$

Thus,

$$
\begin{array} { l } { \displaystyle \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int \left[ ( 1 - \alpha ) c ( x , y ) + \alpha h ( T _ { u } ( x ) - y ) \right] \mathrm { d } \pi } \\ { \displaystyle = \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int \| y - T _ { u } ^ { \alpha } ( x ) \| _ { 2 } ^ { 2 } \mathrm { d } \pi + \alpha ( 1 - \alpha ) \| u \| _ { L ^ { 2 } ( \mu ; \mathbb { R } ^ { p } ) } ^ { 2 } . } \end{array}
$$

For any $\pi \in \Pi ( \mu , \nu )$ , we define $\gamma _ { \pi } = ( T _ { u } ^ { \alpha } , \operatorname { I d } ) _ { \# } \pi$ . Then, we have $\gamma \in \Pi ( ( T _ { u } ^ { \alpha } ) _ { \# } \mu , \nu )$ and

$$
\int \| y - T _ { u } ^ { \alpha } ( x ) \| _ { 2 } ^ { 2 } \mathrm { d } \pi ( x , y ) = \int \| y - z \| _ { 2 } ^ { 2 } \mathrm { d } \gamma _ { \pi } ( z , y ) .
$$

Therefore,

$$
\operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int \| y - T _ { u } ^ { \alpha } ( x ) \| _ { 2 } ^ { 2 } \mathrm { d } \pi ( x , y ) \geqslant W _ { 2 } ^ { 2 } \big ( ( T _ { u } ^ { \alpha } ) _ { \# } \mu , \nu \big ) .
$$

Conversely, for any $\gamma \in \Pi ( ( T _ { u } ^ { \alpha } ) _ { \# } \mu , \nu )$ , set $\eta ~ = ~ ( \mathrm { I d } , T _ { u } ^ { \alpha } ) _ { \# } \mu ~ \in ~ \Pi ( \mu , ( T _ { u } ^ { \alpha } ) _ { \# } \mu )$ . By the gluing lemma, there exists $\lambda _ { \gamma } ~ \in ~ \mathcal { P } ( \Omega ^ { 3 } )$ , with coordinates $( x , z , y )$ , such that $( P _ { 1 , 2 } ) _ { \# } \lambda _ { \gamma } = \eta , ( P _ { 2 , 3 } ) _ { \# } \lambda _ { \gamma } = \gamma$ . Here, $P _ { i , j }$ denotes the projection from $\Omega ^ { 3 }$ onto its i-th and j-th coordinates for $i , j = 1 , 2 , 3 .$ . Denote $\pi _ { \gamma } = ( P _ { 1 , 3 } ) _ { \# } \lambda _ { \gamma }$ . Then we have $\pi _ { \gamma } \in \Pi ( \mu , \nu )$ . Since $\eta$ is supported on the graph of $T _ { u } ^ { \alpha }$ , we have $z = T _ { u } ^ { \alpha } ( x )$ for $\lambda _ { \gamma } { \mathrm { - a . e } }$ $( x , z , y )$ . Hence

$$
\int \Vert z - y \Vert _ { 2 } ^ { 2 } \mathrm { d } \gamma ( z , y ) = \int \Vert T _ { u } ^ { \alpha } ( x ) - y \Vert _ { 2 } ^ { 2 } \mathrm { d } \lambda _ { \gamma } ( x , z , y ) = \int \Vert T _ { u } ^ { \alpha } ( x ) - y \Vert _ { 2 } ^ { 2 } \mathrm { d } \pi _ { \gamma } ( x , y ) .
$$

Therefore,

$$
\operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int \| T _ { u } ^ { \alpha } ( x ) - y \| _ { 2 } ^ { 2 } \mathrm { d } \pi ( x , y ) \leqslant \int \| z - y \| _ { 2 } ^ { 2 } \mathrm { d } \gamma ( z , y ) .
$$

Taking the infimum over $\gamma \in \Pi ( ( T _ { u } ^ { \alpha } ) _ { \# } \mu , \nu )$ gives

$$
\operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int \| T _ { u } ^ { \alpha } ( x ) - y \| _ { 2 } ^ { 2 } \mathrm { d } \pi ( x , y ) \leqslant W _ { 2 } ^ { 2 } ( ( T _ { u } ^ { \alpha } ) _ { \# } \mu , \nu ) .
$$

Therefore, the equality holds. Since the remaining terms in $\mathcal { F } ( \pi , u )$ are independent of the transport plan $\pi _ { \mathrm { : } }$ , adding them to both sides gives the asserted result. □

Proposition 2.1 provides an interpretation of coupled OT. Under the squared $L _ { 2 ^ { - } }$ cost, the resulting Wasserstein term measures the discrepancy between the target measure ν and the relaxed pushforward measure $( T _ { u } ^ { \alpha } ) _ { \# } \mu .$ , while the additional term $\alpha ( 1 - \alpha ) \| u \| _ { L ^ { 2 } ( \mu ) } ^ { 2 }$ penalizes large displacements. Together with the other terms, the deformation field is optimized through a balanced variational principle: the Wasserstein term provides global distributional guidance, the displacement penalty controls the magnitude of the deformation, the elastic energy promotes spatial coherence, and the landmark loss anchors the deformation at reliable locations. This shows that the coupled OT model turns sparse landmark observations into a dense deformation field by combining distributional alignment with geometric regularization.

2.3. Theoretical properties. We first discuss the existence of the solution to the coupled OT model, and then discuss the model behavior when the balancing parameter $\alpha  0$ and $\alpha  1$

Theorem 2.2. Assume that $\mu$ is absolutely continuous and satisfies $\textstyle { \frac { \mathrm { d } \mu } { \mathrm { d } x } } \in L ^ { \infty } ( \Omega )$ 2 and $\beta > 0$ . Let $\rho$ and h are defined in (2.4) and (2.6). Suppose there exists $c _ { \mathrm { e l } } > 0$ such that $E _ { \mathrm { e l } } ( u ) \geqslant c _ { \mathrm { e l } } \Vert u \Vert _ { H ^ { 1 } ( \Omega ; \mathbb { R } ^ { p } ) } ^ { 2 }$ for any $u \in \mathcal { U } , A _ { r } ^ { i }$ is a linear bounded operator, and $\mathcal { F }$ is not identically +∞ on $\Pi ( \dot { \mu } , \nu ) \times \mathcal { U }$ . Then, there exists $( \pi _ { \star } , u _ { \star } ) \in \Pi ( \mu , \nu ) \times \mathcal { U } _ { \star }$ such that

$$
\mathcal { F } ( \pi _ { \star } , u _ { \star } ) = \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) , u \in \mathcal { U } } \mathcal { F } ( \pi , u ) .
$$

Proof. Let $\{ ( \pi _ { n } , u _ { n } ) \} _ { n \geqslant 1 } \subset \Pi ( \mu , \nu ) \times \mathcal { U }$ be a minimizing sequence to ${ \mathcal F } .$ . Then, up to discarding finitely many terms, there exists $C > 0$ such that $\mathcal { F } ( \pi _ { n } , u _ { n } ) \leqslant C$ Since all terms in $\mathcal { F }$ are nonnegative, the coercivity of $E _ { \mathrm { e l } }$ gives

$$
\beta E _ { \mathrm { e l } } ( u _ { n } ) \leqslant \mathcal { F } ( \pi _ { n } , u _ { n } ) \leqslant C , \ \mathrm { a n d \ t h e n \ } \| u _ { n } \| _ { H ^ { 1 } ( \Omega ; \mathbb { R } ^ { p } ) } ^ { 2 } \leqslant \frac { C } { \beta c _ { \mathrm { e l } } } .
$$

Thus $\{ u _ { n } \}$ is bounded in U. Since U is bounded in $H ^ { 1 } ( \Omega ; \mathbb { R } ^ { p } )$ , there exist a subsequence, still denoted by $\{ u _ { n } \}$ , and $u _ { \star } \in \mathcal { U }$ such that $u _ { n } \to u _ { \star }$ weakly in $H ^ { 1 } ( \Omega ; \mathbb { R } ^ { p } )$

Moreover, by the Rellich–Kondrachov compactness theorem, $u _ { n } \ \to \ u _ { \star }$ strongly in $L ^ { 2 } ( \Omega ; \mathbb { R } ^ { p } )$ . Using $\mu \ll d x$ and $\rho _ { \mu } = : \mathrm { d } \mu / \mathrm { d } x \in L ^ { \infty } ( \Omega )$ , we further have

$$
\left\| u _ { n } - u _ { \star } \right\| _ { L ^ { 2 } ( \mu ; \mathbb { R } ^ { p } ) } ^ { 2 } = \int | u _ { n } - u _ { \star } | ^ { 2 } \rho _ { \mu } \mathrm { d } x \leqslant \left\| \rho _ { \mu } \right\| _ { L ^ { \infty } ( \Omega ) } \left\| u _ { n } - u _ { \star } \right\| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { p } ) } ^ { 2 } \to 0 .
$$

Next, we extract a convergent subsequence of transport plans. Since Ω is Polish, $\mu$ and $\nu$ are tight. Hence $\{ \pi _ { n } \} \subset \Pi ( \mu , \nu )$ is tight in $\mathscr { P } ( \Omega \times \Omega )$ . Indeed, for every $\varepsilon > 0$ there exist compact sets $K _ { 1 } , K _ { 2 } \subset \Omega$ such that $\mu ( K _ { 1 } ) > 1 - \varepsilon / 2 , \nu ( K _ { 2 } ) > 1 - \varepsilon / 2$ and therefore

$$
\pi _ { n } ( ( \Omega \times \Omega ) \setminus ( K _ { 1 } \times K _ { 2 } ) ) \leqslant \mu ( \Omega \setminus K _ { 1 } ) + \nu ( \Omega \setminus K _ { 2 } ) < \varepsilon .
$$

By Prokhorov’s theorem, we have up to a subsequence, $\pi _ { n }  \pi _ { \star }$ <sub>⋆</sub> narrowly in $\mathcal { P } ( \Omega \times \Omega )$ The continuity of the marginal projections and the continuity of pushforwards under narrow convergence indicates $\pi _ { \star }$ has marginals $\mu , \nu _ { \mathrm { { : } } }$ , and hence $\pi _ { \star } \in \Pi ( \mu , \nu )$

We now prove the lower semicontinuity of $\mathcal { F }$ . Since c is nonnegative and lower semicontinuous, the Portmanteau theorem gives

$$
\int c ( x , y ) \mathrm { d } \pi _ { \star } \leqslant \operatorname* { l i m i n f } _ { n  \infty } \int c ( x , y ) \mathrm { d } \pi _ { n } .
$$

It remains to handle the deformation-consistency term. Define $G _ { n } ( x , y ) : = x + u _ { n } ( x ) -$ $y ,$ and $G _ { \star } ( x , y ) : = x + u _ { \star } ( x ) - y$ . Since the first marginal of $\pi _ { n }$ is $\mu ,$

$$
\int \| G _ { n } - G _ { \star } \| _ { 2 } ^ { 2 } \mathrm { d } \pi _ { n } = \int \| u _ { n } - u _ { \star } \| _ { 2 } ^ { 2 } \mathrm { d } \mu = \| u _ { n } - u _ { \star } \| _ { L ^ { 2 } ( \mu ; \mathbb { R } ^ { p } ) } ^ { 2 } \to 0 .
$$

Define mappings $\Phi _ { n } ( x , y ) \ = \ ( x , y , G _ { n } ( x , y ) )$ and $\Phi _ { \star } ( x , y ) \ = \ ( x , y , G _ { \star } ( x , y ) )$ , then $[ \Phi _ { n } ] _ { \# } \pi _ { n } \implies [ \Phi _ { \star } ] _ { \# } \pi _ { \star }$ narrowly in $\mathcal { P } ( \Omega \times \Omega \times \mathbb { R } ^ { p } )$ . Specifically, since bounded Lipschitz functions determine narrow convergence on Polish spaces, it is enough to test against $\boldsymbol { \varphi } \in B L _ { b } ( \Omega \times \Omega \times \mathbb { R } ^ { p } )$ . It can be written that

$$
\begin{array} { l } { \displaystyle \int \varphi ( x , y , G _ { n } ( x , y ) ) \mathrm { d } \pi _ { n } - \int \varphi ( x , y , G _ { \star } ( x , y ) ) \mathrm { d } \pi _ { \star } = \int \varphi ( x , y , G _ { n } ( x , y ) ) \mathrm { d } \pi _ { n } } \\ { \displaystyle - \int \varphi ( x , y , G _ { \star } ( x , y ) ) \mathrm { d } \pi _ { n } + \int \varphi ( x , y , G _ { \star } ( x , y ) ) \mathrm { d } \pi _ { n } - \int \varphi ( x , y , G _ { \star } ( x , y ) ) \mathrm { d } \pi _ { \star } . } \end{array}\tag{2.9}
$$

By the Lipschitz continuity,

$$
\biggl | \int \left[ \varphi ( x , y , G _ { n } ( x , y ) ) - \varphi ( x , y , G _ { \star } ( x , y ) ) \right] \mathrm { d } \pi _ { n } \biggr | \leqslant \mathrm { L i p } ( \varphi ) \int \| G _ { n } - G \| _ { 2 } \mathrm { d } \pi _ { n } \to 0 .
$$

We tackle the remaining terms in (2.9) using a Lusin approximation argument. By Lusin’s theorem, for every $\delta > 0$ , there exists a compact set $K _ { \delta } \subset \Omega$ such that $\mu ( \Omega )$ $K _ { \delta } ) < \delta$ and $u _ { \star }$ is continuous on $K _ { \delta }$ . By the Tietze extension theorem, there exists $v _ { \delta } \in C _ { b } ( \Omega ; \mathbb { R } ^ { p } )$ such that $v _ { \delta } ( x ) = u _ { \star } ( x ) , x \in K _ { \delta }$ . We define $\psi ( x , y ) : = \varphi ( x , y , G _ { \star } ( x , y ) )$ and $\psi _ { \delta } ( x , y ) : = \varphi ( x , y , x + v _ { \delta } ( x ) - y )$ . Then $\psi _ { \delta } \in C _ { b } ( \Omega \times \Omega )$ ). By the narrow convergence $\pi _ { n } \to \pi _ { \star }$ , we have

$$
\int \psi _ { \delta } \mathrm { d } \pi _ { n } \to \int \psi _ { \delta } \mathrm { d } \pi _ { \star } .
$$

Moreover, since $\psi = \psi _ { \delta }$ on $K _ { \delta } \times \Omega$ , and $\pi _ { n }$ and $\pi _ { \star }$ have first marginal $\mu ,$ we have

$$
\left| \int ( \psi - \psi _ { \delta } ) \mathrm { d } \pi _ { n } \right| + \left| \int ( \psi - \psi _ { \delta } ) \mathrm { d } \pi _ { \star } \right| \leqslant 4 \| \varphi \| _ { \infty } \mu ( \Omega \setminus K _ { \delta } ) \leqslant 4 \| \varphi \| _ { \infty } \delta .
$$

Therefore,

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname* { s u p } _ { \mathbf { \alpha } } \left| \int \psi \mathrm { d } \pi _ { n } - \int \psi \mathrm { d } \pi _ { \star } \right| \leqslant 4 \| \varphi \| _ { \infty } \delta .
$$

Since $\delta > 0$ is arbitrary, we obtain

$$
\int \varphi ( x , y , G _ { \star } ( x , y ) ) \mathrm { d } \pi _ { n } \to \int \varphi ( x , y , G _ { \star } ( x , y ) ) \mathrm { d } \pi _ { \star } .
$$

This indicates that $[ \Phi _ { n } ] _ { \# } \pi _ { n }  [ \Phi _ { \star } ] _ { \# } \pi _ { \star }$ narrowly. Since h is nonnegative and lower semicontinuous, the Portmanteau theorem gives

$$
\int h ( x + u _ { \star } ( x ) - y ) \mathrm { d } \pi _ { \star } \leqslant \operatorname* { l i m i n f } _ { n  \infty } \int h ( x + u _ { n } ( x ) - y ) \mathrm { d } \pi _ { n } .
$$

By the weak lower semicontinuity of $E _ { \mathrm { e l } }$ , we have $E _ { \mathrm { e l } } ( u _ { \star } ) \ \leqslant$ lim in $\mathrm { f } _ { n \to \infty } E _ { \mathrm { e l } } ( u _ { n } )$ Finally, since each is bounded and linear, weak convergence in $\mathcal { U }$ implies $A _ { r } ^ { i } ( u _ { n } ) $ $A _ { r } ^ { i } ( u _ { \star } )$ in $\mathbb { R } ^ { p }$ . Using the lower semicontinuity of $\rho ,$ we obtain

$$
\sum _ { i = 1 } ^ { m } w _ { i } \rho \big ( A _ { r } ^ { i } ( u _ { \star } ) - \Delta _ { i } ^ { \mathrm { l m } } \big ) \leqslant \operatorname* { l i m i n f } _ { n \to \infty } \sum _ { i = 1 } ^ { m } w _ { i } \rho \big ( A _ { r } ^ { i } ( u _ { n } ) - \Delta _ { i } ^ { \mathrm { l m } } \big ) .
$$

Combining the above estimates yields

$$
{ \mathcal { F } } ( \pi _ { \star } , u _ { \star } ) \leqslant \operatorname* { l i m } _ { n  \infty } { \mathcal { F } } ( \pi _ { n } , u _ { n } ) .\tag{2.10}
$$

Since $( \pi _ { n } , u _ { n } )$ is a minimizing sequence, the equality holds in (2.10).

In Theorem 2.2, according to the definition of U and by setting $E _ { \mathrm { e l } }$ as the linear elastic energy, the condition that $E _ { \mathrm { e l } } ( u ) \geqslant c _ { \mathrm { e l } } \Vert u \Vert _ { H ^ { 1 } ( \Omega ; \mathbb { R } ^ { p } ) } ^ { 2 }$ for all $u \in \mathcal { U }$ holds for some constant $c _ { \mathrm { e l } } > 0$ . Theorem 2.2 guarantees that under mild conditions, the solution of the coupled OT model exists, indicating the problem is well-defined. However, the minimizer may not be unique. Indeed, the transport plan may not be unique in the Kantorovich formulation, and the sparse landmark constraints may not determine a unique deformation field over the whole domain, under the considered conditions.

In the coupled OT model, the parameter α $\in ( 0 , 1 )$ plays a crucial role in balancing the transport cost and the landmark-guided deformation consistency. We next study the model behavior in the two limiting regimes with $\alpha  1$ and $\alpha  0$ , respectively.

Remark 2.3. For any $\alpha \in ( 0 , 1 )$ , let ${ \mathcal { F } } _ { \alpha }$ denote the functional $\mathcal { F }$ with the parameter $\alpha _ { \mathrm { { ; } } }$ , and $\left( \pi _ { \alpha } , u _ { \alpha } \right)$ be a minimizer of ${ \mathcal { F } } _ { \alpha }$ . Let $\left\{ \alpha _ { n } \right\}$ be a sequence such that $\alpha _ { n }  1$ as $n \to \infty$ . Then, under the conditions of Theorem $2 . 2 .$ , up to a subsequence, $\pi _ { \alpha _ { n } } \to \pi ^ { 1 }$ narrowly $\Pi ( \mu , \nu )$ , and $u _ { \alpha _ { n } }  u ^ { 1 }$ strongly in $L ^ { 2 } ( \mu ; \mathbb { R } ^ { p } )$ . Moreover,

$$
( \pi ^ { 1 } , u ^ { 1 } ) \in \operatorname * { a r g m i n } _ { ( \pi , u ) \in \Pi ( \mu , \nu ) \times \mathcal { U } } \left\{ R ( \pi , u ) + J ( u ) \right\} ,\tag{2.11}
$$

$$
\begin{array} { r } { \mathrm { w h e r e ~ } J ( u ) : = \beta E _ { \mathrm { e l } } ( u ) + \sum _ { i = 1 } ^ { m } w _ { i } \rho ( A _ { r } ^ { i } ( u ) - \Delta _ { i } ^ { \mathrm { l m } } ) , R ( \pi , u ) : = \int h ( T _ { u } ( x ) - y ) \mathrm { d } \pi ( x , y ) . } \end{array}
$$

By Remark 2.3, when $\alpha  1$ , the coupled OT model is stable in the sense that every sequence of minimizers admits a subsequential converge to a solution of the deformation-dominated problem (2.11). This shows that the minimizers remain wellcontrolled as $\alpha  1$

Remark 2.4. Let $\begin{array} { r } { C ( \pi ) : = \int _ { \Omega \times \Omega } c ( x , y ) \mathrm { d } \pi ( x , y ) , \mathcal { O } : = \arg \operatorname* { m i n } _ { \pi \in \Pi ( \mu , \nu ) } C ( \pi ) } \end{array}$ , and $\begin{array} { r } { \mathcal { M } : = \arg \operatorname* { m i n } _ { u \in \mathcal { U } } J ( u ) } \end{array}$ . Then, under the conditions of Theorem 2.2, for every sequence $\alpha _ { n } \  \ 0$ , up to a subsequence, there exists $( \pi ^ { 0 } , u ^ { 0 } ) \in \mathcal { O } \times \mathcal { M }$ , such that $\pi _ { \alpha _ { n } } \to \pi ^ { 0 }$ narrowly $\Pi ( \mu , \nu )$ , and $u _ { \alpha _ { n } } \to u ^ { 0 }$ strongly in $L ^ { 2 } ( \mu ; \mathbb { R } ^ { p } )$ . Moreover,

$$
( \pi ^ { 0 } , u ^ { 0 } ) \in \operatorname * { a r g m i n } _ { ( \pi , u ) \in \mathcal { O } \times \mathcal { M } } R ( \pi , u ) .
$$

Remark 2.4 shows that, as $\alpha  0$ , the coupled OT model recovers a classical cost minimization optimal transport plan and an independent elastic-landmark deformation field. In other words, the model reduces to the decoupled problem

$$
\operatorname* { m i n } _ { \pi \in \Pi ( \mu , \nu ) , u \in \mathcal { U } } \left\{ C ( \pi ) + J ( u ) \right\} .
$$

This indicates that the deformation-consistency term $R ( \pi , u )$ does not explicitly afect the limiting energy as $\alpha  0$ . Rather, it serves as a variational selection criterion among the decoupled minimizers, favoring those that minimize $R ( \pi , u )$

3. Computation. This section discuss the computation of the solution to the coupled OT. We first devise a variational splitting of the coupled problem at the level of continuous transport plans and Sobolev deformation fields, before introducing a finite-dimensional approximation. We then develop a finite-element realization of this principle and discuss the convergence of the resulting scheme.

3.1. Variational splitting. Since the coupled OT model (2.2) involves the optimization of two variables, it is reasonable to perform an alternating update scheme. Specifically, for the k-th alternate, $\pi ^ { ( k + 1 ) }$ and $\bar { \mathbf { \Phi } } _ { u } ( k { + } 1 )$ is given by

$$
\begin{array} { r l } & { \pi ^ { ( k + 1 ) } \in \underset { \pi \in \Pi ( \mu , \nu ) } { \mathrm { a r g m i n } } \mathcal { F } ( \pi , u ^ { ( k ) } ) + \mathcal { P } _ { \Pi } ( \pi , \pi ^ { ( k ) } ) , } \\ & { u ^ { ( k + 1 ) } \in \underset { u \in \mathcal { U } } { \mathrm { a r g m i n } } \mathcal { F } ( \pi ^ { ( k + 1 ) } , u ) + \mathcal { P } _ { \mathcal { U } } ( u , u ^ { ( k ) } ) , } \end{array}\tag{3.1}
$$

where $\mathcal { P } _ { \Pi }$ and $\mathcal { P } _ { \mathcal { U } }$ are stabilization terms for the transport plan and deformation field, respectively. Typically, $\mathcal { P } _ { \Pi }$ may be chosen as an entropic or Bregman-type regularization for the transport plan, while $\mathcal { P } _ { \mathcal { U } }$ may be chosen as a quadratic proximal term in the deformation space.

The stabilization terms also provide a basic descent property. Assume that $\mathcal { P } _ { \Pi }$ and $\mathcal { P } _ { \mathcal { U } }$ are nonnegative and satisfy $\mathcal { P } _ { \Pi } ( \pi , \pi ) = 0$ and $\mathcal { P } _ { \mathcal { U } } ( u , u ) = 0$ . Since $\pi ^ { ( k ) }$ and $u ^ { ( k ) }$ are admissible solutions in the two subproblems, the exact updates imply

$$
\mathcal { F } ( \pi ^ { ( k + 1 ) } , u ^ { ( k + 1 ) } ) + \mathcal { P } _ { \Pi } ( \pi ^ { ( k + 1 ) } , \pi ^ { ( k ) } ) + \mathcal { P } _ { \mathcal { U } } ( u ^ { ( k + 1 ) } , u ^ { ( k ) } ) \leqslant \mathcal { F } ( \pi ^ { ( k ) } , u ^ { ( k ) } ) .
$$

Therefore, the objective values $\{ \mathcal { F } ( \pi ^ { ( k ) } , u ^ { ( k ) } ) \} _ { k \geqslant 0 }$ are nonincreasing. Since $\mathcal { F }$ is bounded from below under the assumptions of Theorem 2.2, the objective values converge. This descent property, however, does not by itself imply the convergence of the whole sequence $\{ ( \pi ^ { ( k ) } , \bar { u } ^ { ( k ) } ) \} _ { k \geqslant 0 }$ in the function-space setting. The convergence of the iterates depends on the numerical realization scheme and the stabilization terms.

The above variational splitting is formulated at the function-space level and is not tied to a particular computational representation. In principle, the deformation field and the continuous transport map can be realized by finite elements, splines, radial basis functions, or neural parameterizations, together with a suitable discretization of the input measures. In this work, we adopt a finite-element realization for its compatibility with the Sobolev deformation space and the elastic regularization.

3.2. Finite-element-based discretization. For the finite-element discretization, we further assume that Ω is a bounded polyhedral domain. The input measures and the deformation field are discretized separately. This allows the support points of the discrete measures to be independent of the nodes of the finite-element mesh, which is useful to accelerate computation when the measures are given by empirical samples or by quadrature rules.

We first approximate $\mu$ and ν by the weighted atomic measures

$$
\mu _ { N _ { \mu } } = \sum _ { k = 1 } ^ { N _ { \mu } } a _ { k } \delta _ { \xi _ { k } } , \nu _ { N _ { \nu } } = \sum _ { \ell = 1 } ^ { N _ { \nu } } b _ { \ell } \delta _ { \eta _ { \ell } } ,\tag{3.2}
$$

where $\xi _ { k } , \eta _ { \ell } \in \Omega , \ a _ { k } > 0 , \ b _ { \ell } > 0$ , and $\begin{array} { r } { \sum _ { k = 1 } ^ { N _ { \mu } } a _ { k } = \sum _ { \ell = 1 } ^ { N _ { \nu } } b _ { \ell } = 1 } \end{array}$ . When $\mu$ and $\nu$ are empirical measures, their support points are given by the observed samples, and the corresponding weights are uniform. For continuously distributed measures, the discrete approximation in (3.2) can also be constructed from a finite partition of Ω. Specifically, a representative point is chosen from each cell as a support point, and its weight is set to the mass of that cell. Let $\mathbf { a } = ( a _ { 1 } , \ldots , a _ { N _ { \mu } } ) ^ { \top } , \mathbf { b } = ( b _ { 1 } , \ldots , b _ { N _ { \nu } } ) ^ { \top } .$ A coupling between $\mu _ { N _ { \mu } }$ and $\nu _ { N _ { \imath } }$ is represented by a nonnegative matrix $\mathbf { P } \in \mathbb { R } _ { + } ^ { N _ { \mu } \times N _ { \nu } }$ The corresponding discrete coupling polytope is

$$
\Pi ( \mathbf { a } , \mathbf { b } ) = \left\{ \mathbf { P } \in \mathbb { R } _ { + } ^ { N _ { \mu } \times N _ { \nu } } : \mathbf { P 1 } _ { N _ { \nu } } = \mathbf { a } , \mathbf { P } ^ { \top } \mathbf { 1 } _ { N _ { \mu } } = \mathbf { b } \right\} .\tag{3.3}
$$

Each $\mathbf { P } \in \Pi ( \mathbf { a } , \mathbf { b } )$ induces the discrete transport plan $\begin{array} { r } { \pi _ { \mathbf { P } } = \sum _ { k = 1 } ^ { N _ { \mu } } \sum _ { \ell = 1 } ^ { N _ { \nu } } P _ { k \ell } \delta _ { \left( \xi _ { k } , \eta _ { \ell } \right) } } \end{array}$

We next approximate the deformation field by finite elements. Let $\mathcal { T } _ { \delta }$ be a conforming finite-element mesh of Ω, with mesh size $\delta .$ . We consider Cartesian meshes equipped with continuous tensor-product $\mathbb { Q } _ { 1 }$ elements. We note that our approach is applicable to simplicial meshes. We consider the scalar-valued piecewise afine finiteelement space

$$
V _ { \delta } = \left\{ v _ { \delta } \in C ^ { 0 } ( \overline { { \Omega } } ) : v _ { \delta } | _ { K } \in \mathbb { Q } _ { 1 } ( K ) \mathrm { ~ f o r ~ e v e r y ~ } K \in \mathcal { T } _ { \delta } , v _ { \delta } = 0 \mathrm { ~ o n ~ } \Gamma _ { D } \right\}
$$

and define the vector-valued deformation space by $\mathcal { U } _ { \delta } = ( V _ { \delta } ) ^ { p }$ . Then we have $\mathcal { U } _ { \delta } \subset \mathcal { U }$ Let $\{ \varphi _ { q } \} _ { q = 1 } ^ { M _ { \delta } }$ be the nodal basis associated with the free degrees of freedom of $V _ { \delta }$ , where $M _ { \delta } = \dim V _ { \delta }$ denotes the total number of free degrees of freedom. Every $u _ { \delta } \in \mathcal { U } _ { \delta }$ can be represented as

$$
u _ { \delta } ( x ) = \sum _ { q = 1 } ^ { M _ { \delta } } \varphi _ { q } ( x ) \mathbf { v } _ { q } , \mathrm { ~ w i t h ~ } \mathbf { v } _ { q } \in \mathbb { R } ^ { p } .
$$

We collect the nodal displacement vectors into $\mathbf { v } = ( \mathbf { v } _ { 1 } ^ { \top } , \ldots , \mathbf { v } _ { M _ { \delta } } ^ { \top } ) ^ { \top } \in \mathbb { R } ^ { p M _ { \delta } }$ , and define $\mathbf { B } _ { \delta } ( x ) = [ \varphi _ { 1 } ( x ) \mathbf { I } _ { p } , \hdots , \varphi _ { M _ { \delta } } ( x ) \mathbf { I } _ { p } ] \in \mathbb { R } ^ { p \times p M _ { \delta } }$ . It follows that

$$
\boldsymbol { u } _ { \delta } ( \boldsymbol { x } ) = \mathbf { B } _ { \delta } ( \boldsymbol { x } ) \mathbf { v } ,
$$

Since the support points $\left\{ \xi _ { k } \right\} _ { k = 1 } ^ { N _ { \mu } }$ for measure discretization need not be finite-element nodes, we connect them by evaluating the finite-element deformation field at the support points of source measures:

$$
u _ { \delta } ( \xi _ { k } ) = \mathbf { B } _ { \delta } ( \xi _ { k } ) \mathbf { v } , \mathrm { ~ f o r ~ } k = 1 , \dots , N _ { \mu } .\tag{3.4}
$$

For piecewise afine finite elements, (3.4) is the barycentric interpolation of the nodal displacement vectors on the element containing $\xi _ { k }$

The elastic energy admits the matrix representation

$$
E _ { \mathrm { e l } } ( u _ { \delta } ) = \frac { 1 } { 2 } \mathbf { v } ^ { \top } \mathbf { K } _ { \delta } \mathbf { v } ,\tag{3.5}
$$

where ${ \bf K } _ { \delta } \in \mathbb { R } ^ { p M _ { \delta } \times p M _ { \delta } }$ is the elastic stifness matrix. Let $\{ \mathbf { e } _ { r } \} _ { r = 1 } ^ { p }$ be the canonical basis of $\mathbb { R } ^ { p }$ and define the vector-valued finite-element basis $\{ \psi _ { j } \} _ { j = 1 } ^ { p M _ { \delta } }$ of $\mathcal { U } _ { \delta }$ by $\psi _ { ( q - 1 ) p + r } ( x ) = \varphi _ { q } ( x ) \mathbf { e } _ { r }$ , for $q = 1 , \ldots , M _ { \delta } , r = 1 , \ldots , p$ . The entries of $\mathbf { K } _ { \delta }$ are $[ \mathbf { K } _ { \delta } ] _ { i j } = \mathfrak { a } _ { \mathrm { e l } } \big ( \psi _ { i } , \psi _ { j } \big )$ , for $i , j = 1 , \dots , p M _ { \delta }$ , where

$$
\mathfrak { a } _ { \mathrm { e l } } ( v , w ) = \int \left[ \operatorname { t r } \bigl ( \varepsilon ( v ) ^ { \top } \varepsilon ( w ) \bigr ) + ( \nabla \cdot v ) ( \nabla \cdot w ) \right] \mathrm { d } x .\tag{3.6}
$$

The homogeneous boundary condition on $\Gamma _ { D }$ is incorporated by using only the basis functions associated with the free degrees of freedom. For each landmark pair $( x _ { i } , y _ { i } )$ let $\mathbf { A } _ { i , \delta } \in \mathbb { R } ^ { p \times p M _ { \delta } }$ defined by

$$
\mathbf { A } _ { i , \delta } = \frac { 1 } { | B _ { r } ( x _ { i } ) \cap \Omega | } \int _ { B _ { r } ( x _ { i } ) \cap \Omega } \mathbf { B } _ { \delta } ( z ) \mathrm { d } z .\tag{3.7}
$$

The integral in (3.7) can be computed elementwise by standard numerical quadrature. Then, we have

$$
A _ { r } ^ { i } ( u _ { \delta } ) = { \bf A } _ { i , \delta } { \bf v } .\tag{3.8}
$$

We finally discretize the two terms involving the transport plan. We define the classical transport-cost matrix $\mathbf { C } = ( C _ { k \ell } ) \in \mathbf { \bar { \mathbb { R } } } ^ { N _ { \mu } \times N _ { \nu } }$ with $C _ { k \ell } = c ( \xi _ { k } , \eta _ { \ell } )$ , and for $\mathbf { v } \in \mathbb { R } ^ { p M _ { \delta } }$ , define the deformation-consistency matrix $\mathbf { H } _ { \delta } ( \mathbf { v } ) = \left( H _ { k \ell } ( \mathbf { v } ) \right) \in \mathbb { R } ^ { N _ { \mu } \times N _ { \nu } }$ where $H _ { k \ell } ( \mathbf { v } ) = h \left( \xi _ { k } + \mathbf { B } _ { \delta } ( \xi _ { k } ) \mathbf { v } - \eta _ { \ell } \right)$ . The finite-dimensional objective is therefore given by

$$
\begin{array} { l } { { \displaystyle { \mathcal F } _ { \mathrm { d i s } } ( { \bf P } , { \bf v } ) = \left. \left. { \bf P } , ( 1 - \alpha ) { \bf C } + \alpha { \bf H } _ { \delta } ( { \bf v } ) \right. _ { F } + \frac { \beta } { 2 } { \bf v } ^ { \top } { \bf K } _ { \delta } { \bf v } \right.}  } \\ { { \displaystyle \quad \left. + \sum _ { i = 1 } ^ { m } w _ { i } \rho \left( { \bf A } _ { i , \delta } { \bf v } - \Delta _ { i } ^ { \mathrm { l m } } \right) . \right.}  } \end{array}\tag{3.9}
$$

The resulting finite-dimensional coupled OT problem is

$$
\operatorname* { m i n } _ { \mathbf { P } \in \Pi ( \mathbf { a } , \mathbf { b } ) , \mathbf { v } \in \mathbb { R } ^ { p M _ { \delta } } } \mathcal { F } _ { \mathrm { d i s } } ( \mathbf { P } , \mathbf { v } ) .\tag{3.10}
$$

3.3. Alternating iteration. We solve the finite-dimensional problem (3.10) by alternating between the transport plan and the deformation coeficients. For $\mathbf { P } , \mathbf { Q } \in$ $\mathbb { R } _ { + } ^ { N _ { \mu } \times N _ { \nu } }$ , we define the generalized Kullback–Leibler divergence by

$$
D _ { \Pi } ( { \bf P } , { \bf Q } ) = \sum _ { k = 1 } ^ { N _ { \mu } } \sum _ { \ell = 1 } ^ { N _ { \nu } } \left[ P _ { k \ell } \log \frac { P _ { k \ell } } { Q _ { k \ell } } - P _ { k \ell } + Q _ { k \ell } \right] .\tag{3.11}
$$

Given $\lambda _ { \pi } > 0$ and ${ \lambda } _ { v } > 0$ , the discrete alternating iteration is defined by

$$
\left\{ \begin{array} { l l } { \displaystyle \mathbf { P } ^ { ( k + 1 ) } = \underset { \mathbf { P } \in \Pi ( \mathbf { a } , \mathbf { b } ) } { \arg \operatorname* { m i n } } \left\{ \mathcal { F } _ { \mathrm { d i s } } ( \mathbf { P } , \mathbf { v } ^ { ( k ) } ) + \lambda _ { \pi } D _ { \Pi } ( \mathbf { P } , \mathbf { P } ^ { ( k ) } ) \right\} , } \\ { \displaystyle \mathbf { v } ^ { ( k + 1 ) } = \underset { \mathbf { v } \in \mathbb { R } ^ { p M _ { \delta } } } { \arg \operatorname* { m i n } } \left\{ \mathcal { F } _ { \mathrm { d i s } } ( \mathbf { P } ^ { ( k + 1 ) } , \mathbf { v } ) + \frac { \lambda _ { v } } { 2 } \| \mathbf { v } - \mathbf { v } ^ { ( k ) } \| _ { 2 } ^ { 2 } \right\} . } \end{array} \right.\tag{3.12}
$$

We initialize the transport plan by a strictly positive coupling, namely, $\mathbf { P } ^ { ( 0 ) } = \mathbf { a } \mathbf { b } ^ { \top }$ Updating the transport plan. We define the combined cost matrix

$$
\mathbf { G } ( \mathbf { v } ^ { ( k ) } ) = ( 1 - \alpha ) \mathbf { C } + \alpha \mathbf { H } _ { \delta } ( \mathbf { v } ^ { ( k ) } ) .\tag{3.13}
$$

The first subproblem in (3.12) can then be written as

$$
\mathbf { P } ^ { ( k + 1 ) } = \underset { \mathbf { P } \in \Pi ( \mathbf { a } , \mathbf { b } ) } { \arg \operatorname* { m i n } } \left. \langle \mathbf { P } , \mathbf { G } ( \mathbf { v } ^ { ( k ) } ) \rangle _ { F } + \lambda _ { \pi } D _ { \Pi } ( \mathbf { P } , \mathbf { P } ^ { ( k ) } ) \right. .\tag{3.14}
$$

We define the positive kernel

$$
\mathbf { K } ^ { ( k ) } = \mathbf { P } ^ { ( k ) } \odot \exp \left( - \mathbf { G } ( \mathbf { v } ^ { ( k ) } ) / \lambda _ { \pi } \right) ,\tag{3.15}
$$

where the exponential is entrywise and $\odot$ is denotes the Hadamard product. By introducing Lagrange multipliers for the marginal constraints, the first-order optimality conditions of (3.14) imply that

$$
\mathbf P ^ { ( k + 1 ) } = \mathrm { d i a g } \big ( \mathbf s ^ { ( k + 1 ) } \big ) { \mathbf K } ^ { ( k ) } \mathrm { d i a g } \big ( { \mathbf t } ^ { ( k + 1 ) } \big ) ,\tag{3.16}
$$

where $\mathbf { s } ^ { ( k + 1 ) } \in \mathbb { R } _ { + } ^ { N _ { \mu } }$ and $\mathbf { t } ^ { ( k + 1 ) } \in \mathbb { R } _ { + } ^ { N _ { \nu } }$ are positive scaling vectors. Substituting (3.16) into the marginal constraints $\mathbf { P } ^ { ( k + 1 ) } \mathbf { 1 } _ { N _ { \nu } } = \mathbf { a } , ( \mathbf { P } ^ { ( k + 1 ) } ) ^ { \top } \mathbf { 1 } _ { N _ { \mu } } = \mathbf { b }$ , we obtain a coupled fixed-point system for the scaling vectors. Starting from a positive vector $\mathbf { t } ^ { ( k + 1 , 0 ) }$ , it is solved by the Sinkhorn iteration

$$
\begin{array} { r } { \left\{ \begin{array} { l } { \mathbf { s } ^ { ( k + 1 , j + 1 ) } = \mathbf { a } \oslash \left( \mathbf { K } ^ { ( k ) } \mathbf { t } ^ { ( k + 1 , j ) } \right) , } \\ { \mathbf { t } ^ { ( k + 1 , j + 1 ) } = \mathbf { b } \oslash \left( ( \mathbf { K } ^ { ( k ) } ) ^ { \top } \mathbf { s } ^ { ( k + 1 , j + 1 ) } \right) , } \end{array} \right. } \end{array}\tag{3.17}
$$

where $\oslash$ denotes entrywise division. The limiting scaling vectors are substituted into (3.16) to obtain $\mathbf { P } ^ { ( k + \check { 1 } ) }$ . Since $\mathbf { P } ^ { ( 0 ) }$ is positive, both $\mathbf { P } ^ { ( k ) }$ and $\mathbf { K } ^ { ( k ) }$ remain positive at every outer iteration, and the Sinkhorn updates are well defined.

Updating the deformation. In the numerical realization, we take $h ( z ) = \| z \| _ { 2 } ^ { 2 }$ , and $\rho ( z ) = \| z \| _ { 2 } ^ { 2 }$ . For notational simplicity, define $\mathbf { B } _ { k } = \mathbf { B } _ { \delta } ( \xi _ { k } ) , \mathbf { A } _ { i } = \mathbf { A } _ { i , \delta }$ , and ${ \bf d } _ { k \ell } =$ $\eta _ { \ell } - \xi _ { k }$ . The second subproblem in (3.12) can be written as

$$
\begin{array} { r } { \mathbf { v } ^ { ( k + 1 ) } = \underset { \mathbf { v } \in \mathbb { R } ^ { p M _ { \delta } } } { \arg \operatorname* { m i n } } \Bigg \{ \alpha \underset { k = 1 } { \sum _ { k = 1 } ^ { N _ { \mu } } \sum _ { \ell = 1 } ^ { N _ { \nu } } } P _ { k \ell } ^ { ( k + 1 ) } \| \mathbf { B } _ { k } \mathbf { v } - \mathbf { d } _ { k \ell } \| _ { 2 } ^ { 2 } + \frac { \beta } { 2 } \mathbf { v } ^ { \top } \mathbf { K } _ { \delta } \mathbf { v } } \\ { + \displaystyle \sum _ { i = 1 } ^ { m } w _ { i } \| \mathbf { A } _ { i } \mathbf { v } - \boldsymbol { \Delta } _ { i } ^ { \mathrm { l m } } \| _ { 2 } ^ { 2 } + \frac { \lambda _ { v } } { 2 } \| \mathbf { v } - \mathbf { v } ^ { ( k ) } \| _ { 2 } ^ { 2 } \Bigg \} . } \end{array}\tag{3.18}
$$

We denote

$$
\mathbf { L } _ { \delta } = 2 \alpha \sum _ { k = 1 } ^ { N _ { \mu } } a _ { k } \mathbf { B } _ { k } ^ { \top } \mathbf { B } _ { k } + \beta \mathbf { K } _ { \delta } + 2 \sum _ { i = 1 } ^ { m } w _ { i } \mathbf { A } _ { i } ^ { \top } \mathbf { A } _ { i } + \lambda _ { v } \mathbf { I } _ { p M _ { \delta } } ,\tag{3.19}
$$

$$
\mathbf { g } ^ { ( k + 1 ) } = 2 \alpha \sum _ { k = 1 } ^ { N _ { \mu } } \sum _ { \ell = 1 } ^ { N _ { \nu } } P _ { k \ell } ^ { ( k + 1 ) } \mathbf { B } _ { k } ^ { \top } \mathbf { d } _ { k \ell } + 2 \sum _ { i = 1 } ^ { m } w _ { i } \mathbf { A } _ { i } ^ { \top } \boldsymbol { \Delta } _ { i } ^ { \mathrm { { l m } } } + \lambda _ { v } \mathbf { v } ^ { ( k ) } ,
$$

Then, the first-order optimality condition gives the linear system

$$
\begin{array} { r } { \mathbf { L } _ { \delta } \mathbf { v } ^ { ( k + 1 ) } = \mathbf { g } ^ { ( k + 1 ) } . } \end{array}\tag{3.20}
$$

Here we have used the marginal constraint $\begin{array} { r } { \sum _ { \ell = 1 } ^ { N _ { \nu } } P _ { k \ell } ^ { ( k + 1 ) } = a _ { k } } \end{array}$ . Since $\lambda _ { v } ~ > ~ 0$ , the matrix $\mathbf { L } _ { \delta }$ is symmetric positive definite, and (3.20) admits a unique solution.

3.4. Convergence. We now study the convergence of the above iteration algorithm. We first provide the following theorem.

Theorem 3.1 (Descent and asymptotic behavior). Assume that $a _ { i } > 0$ for any i, $b _ { j } > 0$ for any $j , \ \beta > \ 0$ , and ${ \bf K } _ { \delta } \ \succ \ 0$ . Suppose that all entries of the discrete cost matrices are $f i n i t e$ , the two subproblems are solved exactly, and the transport plan is initialized by $\dot { \mathbf P } ^ { ( 0 ) } = \mathbf a \mathbf b ^ { \top }$ . Then the sequence $\{ ( \mathbf { P } ^ { ( k ) } , \mathbf { v } ^ { ( k ) } ) \} _ { k \geqslant 0 }$ generated by the discrete alternating iteration satisfies the following properties:

1. The objective sequence $\{ \mathcal { \bar { F } } _ { \mathrm { d i s } } ( \mathbf { P } ^ { ( k ) } , \mathbf { v } ^ { ( k ) } ) \} _ { k \geq 0 }$ is non-increasing and converges to a finite limit $\mathcal { F } _ { \infty }$

2. The accumulated stabilization terms are finite:

$$
\sum _ { k = 0 } ^ { \infty } D _ { \Pi } ( { \mathbf { P } } ^ { ( k + 1 ) } , { \mathbf { P } } ^ { ( k ) } ) < \infty \ a n d \ \sum _ { k = 0 } ^ { \infty } \| { \mathbf { v } } ^ { ( k + 1 ) } - { \mathbf { v } } ^ { ( k ) } \| _ { 2 } ^ { 2 } < \infty .
$$

3. The diferences between successive iterates vanish:

$$
\| { \bf P } ^ { ( k + 1 ) } - { \bf P } ^ { ( k ) } \| _ { F } \to 0 , ~ a n d ~ \| { \bf v } ^ { ( k + 1 ) } - { \bf v } ^ { ( k ) } \| _ { 2 } \to 0 .
$$

4. The sequence $\{ ( \mathbf { P } ^ { ( k ) } , \mathbf { v } ^ { ( k ) } ) \} _ { k \geqslant 0 }$ is bounded. Its cluster set is nonempty, compact, and connected.

5. Every cluster point $( \overline { { \mathbf { P } } } , \overline { { \mathbf { v } } } )$ satisfies

$$
\nabla _ { \mathbf { v } } \mathcal { F } _ { \mathrm { d i s } } ( \overline { { \mathbf { P } } } , \overline { { \mathbf { v } } } ) = \mathbf { 0 } .
$$

Proof. Since $\mathbf { P } ^ { ( k ) } \in \Pi ( \mathbf { a } , \mathbf { b } )$ is an admissible competitor for the transport-plan subproblem, the optimality of $\dot { \mathbf { P } } ^ { ( k + 1 ) }$ gives

$$
\mathcal { F } _ { \mathrm { d i s } } ( \mathbf { P } ^ { ( k + 1 ) } , \mathbf { v } ^ { ( k ) } ) + \lambda _ { \pi } D _ { \Pi } ( \mathbf { P } ^ { ( k + 1 ) } , \mathbf { P } ^ { ( k ) } ) \leqslant \mathcal { F } _ { \mathrm { d i s } } ( \mathbf { P } ^ { ( k ) } , \mathbf { v } ^ { ( k ) } ) .
$$

Similarly, we have

$$
\mathcal { F } _ { \mathrm { d i s } } ( \mathbf { P } ^ { ( k + 1 ) } , \mathbf { v } ^ { ( k + 1 ) } ) + \frac { \lambda _ { v } } { 2 } \| \mathbf { v } ^ { ( k + 1 ) } - \mathbf { v } ^ { ( k ) } \| _ { 2 } ^ { 2 } \leqslant \mathcal { F } _ { \mathrm { d i s } } ( \mathbf { P } ^ { ( k + 1 ) } , \mathbf { v } ^ { ( k ) } ) .
$$

Combining these two inequalities yields

$$
\begin{array} { r l } & { \mathcal { F } _ { \mathrm { d i s } } ( \mathbf { P } ^ { ( k ) } , \mathbf { v } ^ { ( k ) } ) - \mathcal { F } _ { \mathrm { d i s } } ( \mathbf { P } ^ { ( k + 1 ) } , \mathbf { v } ^ { ( k + 1 ) } ) } \\ & { \qquad \geqslant \lambda _ { \pi } D _ { \Pi } ( \mathbf { P } ^ { ( k + 1 ) } , \mathbf { P } ^ { ( k ) } ) + \frac { \lambda _ { v } } { 2 } \| \mathbf { v } ^ { ( k + 1 ) } - \mathbf { v } ^ { ( k ) } \| _ { 2 } ^ { 2 } \geqslant 0 . } \end{array}\tag{3.21}
$$

Therefore, the objective sequence is nonincreasing. Since $\mathcal { F } _ { \mathrm { d i s } }$ is bounded from below, there exists $\mathcal { F } _ { \infty } \in \mathbb { R }$ such that $\mathcal { F } _ { \mathrm { d i s } } ( \mathbf { P } ^ { ( k ) } , \mathbf { v } ^ { ( k ) } ) \longrightarrow \mathcal { F } _ { \infty }$

Summing (3.21) from $k = 0$ to K, we obtain

$$
\begin{array} { r l r } {  { \lambda _ { \boldsymbol \pi } \sum _ { k = 0 } ^ { K } D _ { \Pi } ( { \mathbf P } ^ { ( k + 1 ) } , { \mathbf P } ^ { ( k ) } ) + \frac { \lambda _ { v } } { 2 } \sum _ { k = 0 } ^ { K } \| { \mathbf v } ^ { ( k + 1 ) } - { \mathbf v } ^ { ( k ) } \| _ { 2 } ^ { 2 } } } \\ & { } & { \leqslant \mathcal { F } _ { \mathrm { d i s } } ( { \mathbf P } ^ { ( 0 ) } , { \mathbf v } ^ { ( 0 ) } ) - \mathcal { F } _ { \mathrm { d i s } } ( { \mathbf P } ^ { ( K + 1 ) } , { \mathbf v } ^ { ( K + 1 ) } ) . } \end{array}\tag{3.22}
$$

Let $K  \infty$ , we obtain

$$
\sum _ { k = 0 } ^ { \infty } D _ { \Pi } ( \mathbf { P } ^ { ( k + 1 ) } , \mathbf { P } ^ { ( k ) } ) < \infty \mathrm { ~ a n d ~ } \sum _ { k = 0 } ^ { \infty } \| \mathbf { v } ^ { ( k + 1 ) } - \mathbf { v } ^ { ( k ) } \| _ { 2 } ^ { 2 } < \infty .
$$

Consequently, we have

$$
D _ { \Pi } ( \mathbf { P } ^ { ( k + 1 ) } , \mathbf { P } ^ { ( k ) } ) \to 0 \mathrm { ~ a n d ~ } \| \mathbf { v } ^ { ( k + 1 ) } - \mathbf { v } ^ { ( k ) } \| _ { 2 } \to 0 .
$$

Since $\mathbf { P } ^ { ( k + 1 ) }$ and $\mathbf { P } ^ { ( k ) }$ both have total mass one, Pinsker’s inequality gives

$$
D _ { \Pi } ( { \mathbf { P } } ^ { ( k + 1 ) } , { \mathbf { P } } ^ { ( k ) } ) \geqslant \frac { 1 } { 2 } \| { \mathbf { P } } ^ { ( k + 1 ) } - { \mathbf { P } } ^ { ( k ) } \| _ { 1 } ^ { 2 } .
$$

Thus,

$$
\| \mathbf { P } ^ { ( k + 1 ) } - \mathbf { P } ^ { ( k ) } \| _ { F } \leqslant \| \mathbf { P } ^ { ( k + 1 ) } - \mathbf { P } ^ { ( k ) } \| _ { 1 } \to 0 .
$$

We next prove the boundedness of the iteration sequence. Let $\mathbf { z } ^ { ( k ) } = ( \mathbf { P } ^ { ( k ) } , \mathbf { v } ^ { ( k ) } )$ then we have

$$
\| \mathbf { z } ^ { ( k + 1 ) } - \mathbf { z } ^ { ( k ) } \|  0 ,\tag{3.23}
$$

where

$$
\| \mathbf { z } ^ { ( k + 1 ) } - \mathbf { z } ^ { ( k ) } \| ^ { 2 } : = \| \mathbf { P } ^ { ( k + 1 ) } - \mathbf { P } ^ { ( k ) } \| _ { F } ^ { 2 } + \| \mathbf { v } ^ { ( k + 1 ) } - \mathbf { v } ^ { ( k ) } \| _ { 2 } ^ { 2 } .
$$

The transport polytope $\Pi ( \mathbf { a } , \mathbf { b } )$ is compact, so $\{ \mathbf { P } ^ { ( k ) } \} _ { k \geqslant 0 }$ is bounded. Additionally,

$$
\mathcal { F } _ { \mathrm { d i s } } ( \mathbf { P } , \mathbf { v } ) \geqslant \frac { \beta } { 2 } \mathbf { v } ^ { \top } \mathbf { K } _ { \delta } \mathbf { v } .
$$

Since ${ \bf K } _ { \delta } \succ 0$ , there exists $c _ { \delta } > 0$ such that

$$
\mathbf { v } ^ { \top } \mathbf { K } _ { \delta } \mathbf { v } \geqslant c _ { \delta } \| \mathbf { v } \| _ { 2 } ^ { 2 }
$$

for every $\mathbf { v } \in \mathbb { R } ^ { p M _ { \delta } }$ . Therefore,

$$
\frac { \beta c _ { \delta } } { 2 } \| \mathbf { v } ^ { ( k ) } \| _ { 2 } ^ { 2 } \leqslant \mathcal { F } _ { \mathrm { d i s } } ( \mathbf { P } ^ { ( k ) } , \mathbf { v } ^ { ( k ) } ) \leqslant \mathcal { F } _ { \mathrm { d i s } } ( \mathbf { P } ^ { ( 0 ) } , \mathbf { v } ^ { ( 0 ) } ) .
$$

Thus, $\{ \mathbf { v } ^ { ( k ) } \} _ { k \geqslant 0 }$ is bounded, and hence $\{ \mathbf { z } ^ { ( k ) } \} _ { k \geqslant 0 }$ is bounded. If we denote the cluster set of $\{ { \bf z } ^ { ( k ) } \} _ { k \geqslant 0 }$ as

$$
\omega ( { \mathbf z } ^ { ( 0 ) } ) : = \{ \overline { { \mathbf { z } } } : { \mathbf z } ^ { ( k _ { j } ) } \to \overline { { \mathbf { z } } } \mathrm { ~ f o r ~ s o m e ~ } k _ { j } \to \infty \} ,
$$

then $\omega ( \mathbf { z } ^ { ( 0 ) } )$ is nonempty and compact. Let $\mathbf { z } ^ { ( k _ { j } ) }  \overline { { \mathbf { z } } }$ be any convergent subsequence. By the continuity of $\mathcal { F } _ { \mathrm { d i s } }$ , we have

$$
\mathcal { F } _ { \mathrm { d i s } } ( \overline { { \mathbf { z } } } ) = \operatorname* { l i m } _ { j  \infty } \mathcal { F } _ { \mathrm { d i s } } ( \mathbf { z } ^ { ( k _ { j } ) } ) = \mathcal { F } _ { \infty } .
$$

Thus, all cluster points have the same objective value.

It remains to prove that the cluster set is connected. Suppose, by contradiction,

$$
\omega ( \mathbf { z } ^ { ( 0 ) } ) = \mathcal { A } \cup \mathcal { B } ,
$$

where $\mathcal { A }$ and $\boldsymbol { B }$ are nonempty, disjoint compact sets. Then $d : = \mathrm { d i s t } ( A , B ) > 0$ . Since $\mathcal { A }$ and B are nonempty subsets of the cluster set, choose $\mathbf { a } \in { \mathcal { A } }$ and b $\in B .$ . By the definition of a cluster point, the sequence has subsequences converging to a and b, respectively. Hence, it visits the $d / 3 \AA$ -neighborhoods of both A and B infinitely many times. By (3.23), for all suficiently large k,

$$
\| \mathbf { z } ^ { ( k + 1 ) } - \mathbf { z } ^ { ( k ) } \| < d / 3 .
$$

Therefore, whenever the sequence moves from the d/3-neighborhood of $\mathcal { A }$ to the $d / 3 \AA$ neighborhood of $B ,$ it must contain an intermediate iterate outside both neighborhoods. Since the sequence is bounded, these intermediate iterates admit a cluster point outside ${ \mathcal { A } } \cup B .$ , which is a contradiction. Hence, the cluster set is connected.

Finally, the first-order optimality condition for the deformation update is

$$
\nabla _ { \mathbf { v } } \mathcal { F } _ { \mathrm { d i s } } ( \mathbf { P } ^ { ( k + 1 ) } , \mathbf { v } ^ { ( k + 1 ) } ) + \lambda _ { v } ( \mathbf { v } ^ { ( k + 1 ) } - \mathbf { v } ^ { ( k ) } ) = \mathbf { 0 } .\tag{3.24}
$$

Let $( \mathbf { P } ^ { ( k _ { j } ) } , \mathbf { v } ^ { ( k _ { j } ) } )  ( \overline { { \mathbf { P } } } , \overline { { \mathbf { v } } } )$ be a subsequence converging to a cluster point. By (3.23), we have $( \mathbf { P } ^ { ( k _ { j } + 1 ) } , \mathbf { v } ^ { ( k _ { j } + 1 ) } ) \to ( \overline { { \mathbf { P } } } , \overline { { \mathbf { v } } } )$ . Passing to the limit in (3.24) and using the continuity of $\nabla _ { \mathbf { v } } \mathcal { F } _ { \mathrm { d i s } }$ , we obtain

$$
\nabla _ { \mathbf { v } } \mathcal { F } _ { \mathrm { d i s } } ( \overline { { \mathbf { P } } } , \overline { { \mathbf { v } } } ) = \mathbf { 0 } .
$$

In Theorem 3.1, the assumptions are naturally satisfied by the present discretization. In particular, the coercivity of the discrete elastic energy under the imposed boundary conditions ensures that $\mathbf { K } _ { \delta }$ is symmetric positive definite. Theorem 3.1 establishes the asymptotic properties of the discrete alternating iteration. Specifically, the objective value decreases monotonically and converges to a finite limit, while the accumulated stabilization terms remain finite. Consequently, the diferences between successive iterates vanish, ruling out persistent oscillations with nonvanishing step sizes. Moreover, the generated sequence is bounded, and its cluster set is nonempty, compact, and connected. Every cluster point is stationary with respect to the deformation variable. These results provide a stability guarantee for the iteration.

By Theorem 3.1, if the cluster set is finite, its connectedness implies that it consists of a single point. Consequently, the entire iteration sequence converges to this point. However, such a condition is hard to verify. Next, we incorporate an entropy regularization term for the transport plan into the objective function, which allows us to establish convergence of the entire iteration sequence.

Entropy-regularized iteration. We define the entropy

$$
\operatorname { E n t } ( \mathbf { P } ) : = - \sum _ { i = 1 } ^ { N _ { \mu } } \sum _ { j = 1 } ^ { N _ { \nu } } P _ { i j } ( \log P _ { i j } - 1 ) ,
$$

and the objective becomes $\mathcal { F } _ { \mathrm { d i s } } - \lambda _ { \mathrm { e n t } } \mathrm { E n t } ( \mathbf { P } )$ for some $\lambda _ { \mathrm { e n t } } > 0$ . The corresponding alternating iteration is

$$
\left\{ \begin{array} { l l } { \displaystyle \mathbf { P } ^ { ( k + 1 ) } = \underset { \mathbf { P } \in \Pi ( \mathbf { a } , \mathbf { b } ) } { \arg \operatorname* { m i n } } \left\{ \mathcal { F } _ { \mathrm { d i s } } ( \mathbf { P } , \mathbf { v } ^ { ( k ) } ) - \lambda _ { \mathrm { e n t } } \mathrm { E n t } ( \mathbf { P } ) \right\} , } \\ { \displaystyle \mathbf { v } ^ { ( k + 1 ) } = \underset { \mathbf { v } \in \mathbb { R } ^ { p M _ { \delta } } } { \arg \operatorname* { m i n } } \left\{ \mathcal { F } _ { \mathrm { d i s } } ( \mathbf { P } ^ { ( k + 1 ) } , \mathbf { v } ) + \frac { \lambda _ { v } } { 2 } \| \mathbf { v } - \mathbf { v } ^ { ( k ) } \| _ { 2 } ^ { 2 } \right\} . } \end{array} \right.\tag{3.25}
$$

where we omitted the stabilization $D _ { \Pi }$ for updating P. Consequently, we only need to change the definition in (3.15) to

$$
\mathbf { K } ^ { ( k ) } = \exp \left( - \mathbf { G } ( \mathbf { v } ^ { ( k ) } ) / \lambda _ { \mathrm { e n t } } \right) ,
$$

while all subsequent steps remain unchanged.

Theorem 3.2. Assume that $a _ { i } ~ > ~ 0$ for any i, $b _ { j } ~ > ~ 0$ for any $j , \ \beta > \ 0$ , and ${ \bf K } _ { \delta } \succ 0$ , and that all discrete costs are finite. Suppose that the subproblems in (3.25) are solved exactly. Then

$$
( \mathbf { P } ^ { ( k ) } , \mathbf { v } ^ { ( k ) } ) \to ( \mathbf { P } ^ { \star } , \mathbf { v } ^ { \star } ) ,
$$

where $( \mathbf { P } ^ { \star } , \mathbf { v } ^ { \star } )$ is a constrained critical point of $\mathcal { F } _ { \mathrm { d i s } } - \lambda _ { \mathrm { e n t } }$ Ent(P) on $\Pi ( \mathbf { a } , \mathbf { b } ) \times \mathbb { R } ^ { p M _ { \delta } }$

Proof. We denote the extended objective by

$$
\Phi ( \mathbf { P } , \mathbf { v } ) : = \mathcal { F } _ { \mathrm { d i s } } ( \mathbf { P } , \mathbf { v } ) - \lambda _ { \mathrm { e n t } } \operatorname { E n t } ( \mathbf { P } ) + \iota _ { \Pi ( \mathbf { a } , \mathbf { b } ) } ( \mathbf { P } ) .
$$

The optimality condition of the transport-plan update, together with the Bregman identity for the entropy, gives

$$
\Phi ( \mathbf { P } ^ { ( k ) } , \mathbf { v } ^ { ( k ) } ) - \Phi ( \mathbf { P } ^ { ( k + 1 ) } , \mathbf { v } ^ { ( k ) } ) \geqslant \lambda _ { \mathrm { e n t } } D _ { \Pi } ( \mathbf { P } ^ { ( k ) } , \mathbf { P } ^ { ( k + 1 ) } ) .
$$

The optimality of the deformation update gives

$$
\Phi ( \mathbf { P } ^ { ( k + 1 ) } , \mathbf { v } ^ { ( k ) } ) - \Phi ( \mathbf { P } ^ { ( k + 1 ) } , \mathbf { v } ^ { ( k + 1 ) } ) \geqslant \frac { \lambda _ { v } } { 2 } \| \mathbf { v } ^ { ( k + 1 ) } - \mathbf { v } ^ { ( k ) } \| _ { 2 } ^ { 2 } .
$$

Combining these inequalities with Pinsker’s inequality yields, for some $c _ { 1 } > 0$

$$
\Phi ( \mathbf { z } ^ { ( k ) } ) - \Phi ( \mathbf { z } ^ { ( k + 1 ) } ) \geqslant c _ { 1 } \| \mathbf { z } ^ { ( k + 1 ) } - \mathbf { z } ^ { ( k ) } \| ^ { 2 } .
$$

Since $\Pi ( \mathbf { a } , \mathbf { b } )$ is compact, − Ent is bounded from below on $\Pi ( \mathbf { a } , \mathbf { b } )$ . The decrease of $\Phi ( \mathbf { z } ^ { ( k ) } )$ , together with $\beta > 0$ and ${ \bf K } _ { \delta } \succ 0$ , implies that $\{ \mathbf { v } ^ { ( k ) } \} _ { k \geqslant 0 }$ is bounded. By the continuity of G, there exist constants $0 < \kappa ^ { - } \leqslant \kappa ^ { + } < \infty$ such that

$$
\kappa ^ { - } \leqslant [ \mathbf { K } ^ { ( k ) } ] _ { i j } \leqslant \kappa ^ { + } \mathrm { ~ f o r ~ a l l ~ } i , j , k .
$$

Let $\begin{array} { r } { S ^ { ( k + 1 ) } : = \sum _ { i } s _ { i } ^ { ( k + 1 ) } , T ^ { ( k + 1 ) } : = \sum _ { j } t _ { j } ^ { ( k + 1 ) } } \end{array}$ . The marginal constraints imply

$$
s _ { i } ^ { ( k + 1 ) } \geqslant \frac { a _ { i } } { \kappa ^ { + } T ^ { ( k + 1 ) } } , t _ { j } ^ { ( k + 1 ) } \geqslant \frac { b _ { j } } { \kappa ^ { + } S ^ { ( k + 1 ) } } .
$$

Moreover, since $\mathbf { P } ^ { ( k + 1 ) }$ has unit total mass,

$$
1 = \sum _ { i , j } s _ { i } ^ { ( k + 1 ) } [ { \bf K } ^ { ( k ) } ] _ { i j } t _ { j } ^ { ( k + 1 ) } \geqslant \kappa ^ { - } S ^ { ( k + 1 ) } T ^ { ( k + 1 ) } .
$$

It follows that

$$
P _ { i j } ^ { ( k + 1 ) } \geqslant a _ { i } b _ { j } \left( \kappa ^ { - } / \kappa ^ { + } \right) ^ { 2 } \mathrm { ~ f o r ~ a l l ~ } i , j , k .
$$

Hence the transport iterates are uniformly bounded away from zero. The optimality condition of the transport-plan update gives

$$
\mathbf { 0 } \in \mathbf { G } ( \mathbf { v } ^ { ( k ) } ) + \lambda _ { \mathrm { e n t } } \log \mathbf { P } ^ { ( k + 1 ) } + N _ { \mathrm { I I } ( \mathbf { a } , \mathbf { b } ) } ( \mathbf { P } ^ { ( k + 1 ) } ) .
$$

Since

$$
\partial _ { \mathbf { P } } \Phi ( \mathbf { z } ^ { ( k + 1 ) } ) = \mathbf { G } ( \mathbf { v } ^ { ( k + 1 ) } ) + \lambda _ { \mathrm { e n t } } \log \mathbf { P } ^ { ( k + 1 ) } + N _ { \mathrm { I I } ( \mathbf { a } , \mathbf { b } ) } ( \mathbf { P } ^ { ( k + 1 ) } ) ,
$$

we obtain

$$
\begin{array} { r } { \mathrm { d i s t } \left( \mathbf { 0 } , \partial _ { \mathbf { P } } \Phi ( \mathbf { z } ^ { ( k + 1 ) } ) \right) \leqslant \| \mathbf { G } ( \mathbf { v } ^ { ( k + 1 ) } ) - \mathbf { G } ( \mathbf { v } ^ { ( k ) } ) \| _ { F } . } \end{array}
$$

Since $\{ \mathbf { v } ^ { ( k ) } \} _ { k \geqslant 0 }$ is bounded and G is continuously diferentiable, there exists $L _ { G } > 0$ such that

$$
\mathrm { d i s t } \left( \mathbf { 0 } , \partial _ { \mathbf { P } } \Phi ( \mathbf { z } ^ { ( k + 1 ) } ) \right) \leqslant L _ { G } \| \mathbf { v } ^ { ( k + 1 ) } - \mathbf { v } ^ { ( k ) } \| _ { 2 } .
$$

For the deformation component, the optimality condition gives

$$
\nabla _ { \mathbf { v } } \Phi ( \mathbf { z } ^ { ( k + 1 ) } ) = - \lambda _ { v } ( \mathbf { v } ^ { ( k + 1 ) } - \mathbf { v } ^ { ( k ) } ) .
$$

Consequently, there exists $c _ { 2 } > 0$ such that

$$
\begin{array} { r } { \mathrm { d i s t } \left( \mathbf { 0 } , \partial \Phi ( \mathbf { z } ^ { ( k + 1 ) } ) \right) \leqslant c _ { 2 } \| \mathbf { z } ^ { ( k + 1 ) } - \mathbf { z } ^ { ( k ) } \| _ { 2 } . } \end{array}
$$

The uniform positivity of $\{ \mathbf { P } ^ { ( k ) } \} _ { k \geqslant 1 }$ implies that there exists $\tau > 0$ such that $P _ { i j } ^ { ( k ) } \geqslant \tau$ for all $i , j$ and $k \geqslant 1$ . Consequently, every cluster point P<sup>¯</sup> of the transport iterates satisfies $\bar { P } _ { i j } \geqslant \tau$ . Hence, in a suficiently small neighborhood of each cluster point, all entries of P remain strictly positive, and the nonnegativity constraints in $\Pi ( \mathbf { a } , \mathbf { b } )$ are inactive. Locally, the feasible set therefore coincides with the afine subspace determined by

$$
\mathbf { P 1 } _ { N _ { \nu } } = \mathbf { a } , \mathbf { P } ^ { \top } \mathbf { 1 } _ { N _ { \mu } } = \mathbf { b } .
$$

On this neighborhood, the entropy term $- \mathrm { E n t } ( \mathbf { P } )$ is real analytic. Moreover, under the quadratic choices of h and $\rho ,$ the function $\mathcal { F } _ { \mathrm { d i s } }$ is polynomial in $( \mathbf { P } , \mathbf { v } )$ . Therefore, the restriction of the smooth part of Φ to the above afine subspace is real analytic and satisfies the Lojasiewicz gradient inequality. Equivalently, Φ satisfies the Kurdyka– Lojasiewicz property in a neighborhood of every cluster point of the sequence. The preceding suficient-descent and subgradient estimates, together with the boundedness of $\{ { \bf z } ^ { ( k ) } \} _ { k \geqslant 0 }$ , therefore imply

$$
\sum _ { k = 0 } ^ { \infty } \| \mathbf { z } ^ { ( k + 1 ) } - \mathbf { z } ^ { ( k ) } \| _ { 2 } < \infty
$$

by the Kurdyka– Lojasiewicz convergence theorem. Hence $\{ \mathbf { z } ^ { ( k ) } \} _ { k \geqslant 0 }$ is a Cauchy sequence and converges to some $\mathbf { z } ^ { \star }$ . Moreover, the subgradient estimate gives

$$
\mathrm { d i s t } ( \mathbf { 0 } , \partial \Phi ( \mathbf { z } ^ { ( k ) } ) )  0 .
$$

By the closedness of the limiting subdiferential,

$$
\mathbf { 0 } \in \partial \Phi ( \mathbf { z } ^ { \star } ) .
$$

Thus, $\mathbf { z } ^ { \star } = ( \mathbf { P } ^ { \star } , \mathbf { v } ^ { \star } )$ is a constrained critical point of $\mathcal { F } _ { \mathrm { d i s } } - \lambda _ { \mathrm { e n t } } \mathrm { E n t } ( \mathbf { P } )$

By Theorem 3.2, the iteration sequence for the entropy-regularized discrete objective converges to a constrained critical point.

4. Numerical results. We first conduct controlled experiments on synthetic fish-shaped distributions with explicitly prescribed ground-truth deformation fields. We then present a real shape-matching experiment for which the ground-truth deformation field is unavailable.

4.1. Synthetic experiments. We set the computational domain as $\Omega = [ 0 , 1 ] ^ { 2 }$ The source distribution is constructed from a filled fish-shaped mask (see Fig. 1). The binary mask is smoothed using a Gaussian filter with standard deviation 0.8. The resulting density is normalized to have unit mass. The target distribution is generated by deforming the source distribution through the prescribed map $u ^ { \star }$ . To make the ground-truth deformation compatible with the homogeneous Dirichlet boundary condition, we introduce the cutof function

$$
\chi _ { \tau } ( x ) = \zeta _ { \tau } ( x _ { 1 } ) \zeta _ { \tau } ( x _ { 2 } ) , \quad \tau = 0 . 0 5 ,
$$

![](images/10bcd638dec5a5c4a741703dc6c9e7b25ac5da69fc2c4804b889a5fd58b4aacb.jpg)  
Fig. 1. Fish-shaped source and target densities, together with the candidate landmarks marked by red crosses.

where

$$
\zeta _ { \tau } ( s ) = \left\{ \begin{array} { l l } { s / \tau , } & { 0 \leqslant s < \tau , } \\ { 1 , } & { \tau \leqslant s \leqslant 1 - \tau , } \\ { ( 1 - s ) / \tau , } & { 1 - \tau < s \leqslant 1 . } \end{array} \right.
$$

The ground-truth deformation is then defined by

$$
\begin{array} { r } { u ^ { \star } ( x ) = \chi _ { \tau } ( x ) \left[ ( A - I _ { 2 } ) ( x - x _ { \mathrm { c } } ) + u _ { \mathrm { n r } } ( x ) \right] , \quad x _ { \mathrm { c } } = ( 0 . 5 , 0 . 5 ) ^ { \top } , } \end{array}
$$

where

$$
A = R ( 8 ) \mathrm { d i a g } ( 1 . 0 8 , 0 . 9 4 ) , \quad R ( \theta ) = \left( \begin{array} { c c } { { \cos \theta } } & { { - \sin \theta } } \\ { { \sin \theta } } & { { \cos \theta } } \end{array} \right) .
$$

For $\boldsymbol { x } = ( x _ { 1 } , x _ { 2 } ) ^ { \top }$ , the nonrigid component is defined as

$$
u _ { \mathrm { n r } } ( x ) = { \binom { 0 . 0 1 8 \sin ( \pi x _ { 1 } ) \sin ( 2 \pi x _ { 2 } ) + 0 . 0 1 2 q ( x ) } { 0 . 0 2 0 \sin ( 2 \pi x _ { 1 } ) \sin ( \pi x _ { 2 } ) - 0 . 0 1 0 ( x _ { 1 } - 0 . 5 5 ) q ( x ) } } ,
$$

where

$$
q ( x ) = \exp \left( - \Big ( \frac { x _ { 1 } - 0 . 6 6 } { 0 . 1 6 } \Big ) ^ { 2 } - \Big ( \frac { x _ { 2 } - 0 . 5 4 } { 0 . 1 1 } \Big ) ^ { 2 } \right) .
$$

Here, $\pi$ denotes the circle constant. This deformation combines rotation, anisotropic scaling, smooth spatial oscillation, and a localized nonlinear deformation near the front part of the fish, providing a nontrivial test case involving multiple deformation patterns; it also satisfies $u ^ { \star } = 0$ on ∂Ω. The target density is obtained by transporting the source masses through $T ^ { \star }$ and interpolating them back onto the original Cartesian grid using barycentric coordinates. A Gaussian filter with standard deviation 0.6 is subsequently applied, followed by mass normalization. The source and target measures are discretized on a uniform $1 1 \times 1 1$ Cartesian grid, as computing the transport plan on a larger grid is expensive. The deformation field is represented on a uniform $2 1 \times 2 1$ Cartesian grid. Eight candidate landmarks are placed at representative positions of the source fish, including the nose, the upper and lower tail tips, the dorsal and ventral fin tips, the upper and lower body boundaries, and the tail base, as illustrated in Fig. 1. For a source landmark $x _ { i }$ , its target counterpart $y _ { i }$ is generated using the local observation operator $A _ { r } ^ { i } ( u ^ { \star } )$ on $x _ { i }$

To validate the efectiveness of our coupled OT (denoted by COT), we compare it with two baselines: OT (barycentric) and Landmark-only. The OT (barycentric)

![](images/458202b7bb600bd35fe111402c942933208d571d776369e4957d78c54eaa1775.jpg)

![](images/6056a4039e4ccbe726fcbf30ea7eda3a5fcc91f98176e70a7a731a1ce07e7c71.jpg)

![](images/64e6f2c4d89dabe205ec9bcbff57edbf6593624691d14af7a5a190183331d79d.jpg)  
Fig. 2. Left and middle: objective values $\mathcal { F } ( \mathbf { z } ^ { ( k ) } )$ along the iterations of the KL-proximal (COT (proximal $K L ) )$ and entropy-regularized (COT (entropy)) algorithms for coupled OT. Right: relative error $\| \mathbf { z } ^ { ( k + 1 ) } - \mathbf { z } ^ { ( k ) } \|$ between two successive iterates of COT (entropy).

method estimates the deformation via barycentric projection of an OT plan. The Landmark-only method estimates the deformation from landmark fitting and elastic regularization alone. Our proposed COT is implemented using both the KL-based proximal and entropy-regularized algorithms; we denote them by COT (proximal KL) and COT (entropy), respectively.

All numerical experiments are conducted in a single-threaded setting using Python 3.9 on a CentOS 7 Linux server equipped with two Intel Xeon Gold 6246R processors running at 3.40 GHz and 376 GB of RAM. We use the quadratic choices $c ( x , y ) =$ $\| x - y \| _ { 2 } ^ { 2 } , h ( z ) = \| z \| _ { 2 } ^ { 2 } , \rho ( z ) = \| z \| _ { 2 } ^ { 2 }$ , and assign the same weight $w _ { i } = 1$ to all observed landmarks. The model parameters are set as $\alpha = 0 . 5 , \beta = 0 . 0 5$ . For the KL-proximal algorithm, we take $\lambda _ { \pi } = 1 0 ^ { - 3 } , \lambda _ { v } = 1 0 ^ { - 3 }$ , whereas the entropy parameter in the entropy-regularized algorithm is set as $\lambda _ { \mathrm { e n t } } ~ = ~ 5 \times 1 0 ^ { - 3 }$ . The transport plan and deformation coeficients are initialized by $\mathbf { P } ^ { ( 0 ) } = \mathbf { a } \mathbf { b } ^ { \top } , \mathbf { v } ^ { ( 0 ) } = \mathbf { 0 }$

Since the ground-truth deformation is available, we first assess the recovered field using the Deformation error on the source distribution

$$
e _ { L ^ { 2 } ( \mu ) } = \big ( \sum _ { j } a _ { j } \| u _ { \delta } ( \xi _ { j } ) - u ^ { \star } ( \xi _ { j } ) \| _ { 2 } ^ { 2 } \big ) ^ { 1 / 2 } .
$$

In the experiments, we randomly hold a subset of the candidate landmarks out for computation and evaluate the method on the held-out landmarks. The Held-out landmark error is defined by

$$
e _ { \mathrm { l m } } ^ { \mathrm { h e l d } } = \big ( \frac { 1 } { | \mathcal { T } _ { \mathrm { h e l d } } | } \sum _ { i \in \mathcal { T } _ { \mathrm { h e l d } } } \big | \big | \mathbf { A } _ { i , \delta } \mathbf { v } - A _ { r } ^ { i } \big ( \boldsymbol { u } ^ { \star } \big ) \big | \big | _ { 2 } ^ { 2 } \big ) ^ { 1 / 2 } ,
$$

where $\mathcal { T } _ { \mathrm { h e l d } }$ is set of the held-out landmarks. To measure distributional alignment, we warp the source density using the estimated deformation and compute its discrete $L ^ { 1 }$ discrepancy from the target density (Density error ). All the metrics are computed on the 21 × 21 fine grid.

Convergence. We first validate the convergence properties of the computational algorithms. For the KL-based proximal algorithm, we plot the objective function values along the iteration sequence in the left panel of Fig. 2. We can observe that the objective function value stably decreases and converges, which echoes Theorem 3.1. For the entropy-regularized algorithm, we theoretically show the convergence of the iteration sequence in Theorem 3.2. In Fig. 2, we plot both the objective function values along the iteration sequence in the middle panel and the relative error between two successive iterates in the right panel. It can be seen that the objective function values decrease and the iteration sequence converges, as the relative error stably decreases; this echoes Theorem 3.2.

![](images/e37ff1e55c3c3f3e0b0687ba161b3d0bf0e00f7cc38e3fbeeda2a3609bc48007.jpg)  
Fig. 3. Deformation error (left), held-out landmark error (middle), and density error (right) of four methods under varying numbers of landmarks on the fish-shaped mask. Results are averaged over five replications.

Performance of estimated deformation. We report the metric of the estimated deformation in Fig. 3. We have the following observations from Fig. 3. As the number of landmarks increases, both the Landmark-only using only landmarks and our proposed COT (proximal KL) and COT (entropy) consistently improve performance across all three metrics, confirming that the landmarks indeed help recover the deformation field under distribution transition. Although the method of OT (barycentric) directly matches the source and target distributions on the coarse computational grid, its density error remains higher than that of COT after extension to the fine grid. This may be attributed to the fact that OT (barycentric) estimates a transport plan rather than a spatially coherent deformation map. The deformation field is subsequently induced through barycentric projection, so its regularity and of-grid accuracy are not directly controlled by the OT objective. The benefit of OT prior is particularly evident in the sparse-landmark regime, where COT substantially outperforms Landmark-only, demonstrating that global distribution matching efectively complements the limited geometric information provided by sparse landmarks. Meanwhile, as more landmarks become available, the gap between COT and Landmark-only narrows, but COT remains consistently more accurate by combining global distribution matching with landmark constraints. Additionally, the proximal-KL and entropy-regularized variants perform nearly identically, indicating little sensitivity to the numerical scheme.

4.2. Real Shape Matching. We further evaluate the proposed method on real shape-matching problems. We use handwritten digit images from the MNIST dataset [20] and select three representative source-target pairs, as shown in Fig. 4. For each pair, two landmarks are manually specified to provide sparse geometric guidance. Unlike the synthetic setting, the true deformation between two real shapes is unavailable, so the comparison is mainly based on the estimated deformation fields. Since COT (proximal KL) and COT (entropy) exhibit nearly identical performance in the synthetic experiments, we report only COT (proximal KL) here for clarity.

As shown in Fig. 4, OT (barycentric) captures the overall distributional correspondence but can produce spatially irregular displacement fields that are not fully consistent with the apparent shape structure. In contrast, Landmark-only produces smoother deformation fields around the prescribed landmarks, but the estimated deformation diminishes in regions where the sparse landmark constraints provide little

Source

Target

OT (barycentric)

Landmark-only COT (proximal KL)

![](images/ad63345b1175aa3bd4c4e2eea86a0af9fe84e2ea38eed7a07cd981c580eb2f3e.jpg)

Fig. 4. Comparison of the deformation fields estimated by diferent methods. In each row, the source and target shapes are shown in blue and green, respectively. Two landmark pairs indicated in red and orange are provided for each shape pair, and the estimated deformation fields are visualized by pink arrows.

guidance. Through the coupled estimation of transport and deformation, our proposed COT (proximal KL) propagates both distributional and landmark information across the source support, leading to more spatially coherent deformation fields that remain consistent with the prescribed correspondences. These results further demonstrate the benefit of coupling distributional alignment with sparse geometric constraints.

5. Conclusions. This paper proposed a novel coupled optimal transport (OT) framework that jointly optimizes the transport plan and deformation field, integrating distributional alignment induced by transport-cost minimization and geometric information provided by sparse landmarks. We established the well-posedness of the proposed model and characterized its connection to classical OT. Finite-element-based numerical algorithms were further developed, together with their convergence analysis. Synthetic and real shape-matching experiments demonstrated the efectiveness of the proposed coupled OT.

While the proposed framework is general, the current finite-element-based numerical algorithms were primarily designed for low-dimensional spatial domains. Developing numerical methods for coupled OT for high-dimensional and large-scale problems is an important direction for future research. In particular, parameterizing the deformation field with neural networks may provide a flexible and scalable alternative to finite-element representations, potentially enabling coupled transport-deformation modeling for more complex data and applications.

Appendix A. Proof of Remark 2.3.

Proof. For convenience, we denote

$$
R ( \pi , u ) = : \int h ( T _ { u } ( x ) - y ) \mathrm { d } \pi ( x , y ) , { \mathrm { ~ a n d ~ } } C ( \pi ) = : \int c ( x , y ) \mathrm { d } \pi ( x , y ) .
$$

Based on Theorem 2.2, we have $J ( u _ { n } ) \leqslant C _ { \mathrm { : } }$ , and $R ( \pi _ { n } , u _ { n } ) \leqslant C .$ , for some $C > 0$ According to the proof of Theorem 2.2, up to a subsequence, $u _ { \alpha _ { n } } \to u ^ { * }$ strongly in $L ^ { 2 } ( \mu ; \mathbb { R } ^ { p } )$ , and $\pi _ { \alpha _ { n } }  \pi _ { \star }$ narrowly in $\mathcal { P } ( \Omega \times \Omega )$ , where $( \pi ^ { * } , u ^ { * } ) \in \Pi ( \mu , \nu ) \times \mathcal { U }$ . It remains to identify the limit. By lower semicontinuity,

$$
R ( \pi ^ { * } , u ^ { * } ) + J ( u ^ { * } ) \leqslant \operatorname* { l i m } _ { n  \infty } \operatorname* { i n f } _ { } R ( \pi _ { n } , u _ { n } ) + J ( u _ { n } ) .
$$

On the other hand, for any $( \pi , u ) \in \Pi ( \mu , \nu ) \times \mathcal { U } .$ the minimality of $( \pi _ { n } , u _ { n } )$ gives

$$
( 1 - \alpha _ { n } ) c ( \pi _ { n } ) + \alpha _ { n } R ( \pi _ { n } , u _ { n } ) + J ( u _ { n } ) \leqslant ( 1 - \alpha _ { n } ) C ( \pi ) + \alpha _ { n } R ( \pi , u ) + J ( u ) .
$$

Dropping the first nonnegative term on the left-hand side, we have

$$
\begin{array} { r l } & { R ( \pi _ { n } , u _ { n } ) + J ( u _ { n } ) = \alpha _ { n } R ( \pi _ { n } , u _ { n } ) + J ( u _ { n } ) + ( 1 - \alpha _ { n } ) R ( \pi _ { n } , u _ { n } ) } \\ & { \qquad \leqslant ( 1 - \alpha _ { n } ) C ( \pi ) + \alpha _ { n } R ( \pi , u ) + J ( u ) + ( 1 - \alpha _ { n } ) R ( \pi _ { n } , u _ { n } ) . } \end{array}
$$

Since $R ( \pi _ { n } , u _ { n } )$ is uniformly bounded and $\alpha _ { n }  1$ , we have

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname* { s u p } _ { } \left[ R ( \pi _ { n } , u _ { n } ) + J ( u _ { n } ) \right] \leqslant R ( \pi , u ) + J ( u ) .
$$

Therefore,

$$
R ( \pi ^ { * } , u ^ { * } ) + J ( u ^ { * } ) \leqslant R ( \pi , u ) + J ( u )
$$

for every $( \pi , u ) \in \Pi ( \mu , \nu ) \times \mathcal { U }$ . Hence, we conclude the proof.

Acknowledgments. We would like to thank the referees for handling our paper.

## REFERENCES

[1] J. Altschuler, J. Niles-Weed, and P. Rigollet, Near-linear time approximation algorithms for optimal transport via sinkhorn iteration, in Advances in Neural Information Processing Systems, 2017.

[2] M. Arjovsky, S. Chintala, and L. Bottou, Wasserstein generative adversarial networks, in International Conference on Machine Learning, 2017, pp. 214–223.

[3] Y. Balaji, R. Chellappa, and S. Feizi, Robust optimal transport with applications in generative modeling and domain adaptation, Advances in Neural Information Processing Systems, 33 (2020), pp. 12934–12944.

[4] M. Bauer, S. Joshi, and K. Modin, Difeomorphic density matching by optimal information transport, SIAM Journal on Imaging Sciences, 8 (2015), pp. 1718–1751.

[5] L. A. Caffarelli and R. J. McCann, Free boundaries in optimal transport and monge-ampere obstacle problems, Annals of Mathematics, 171 (2010), pp. 673–730.

[6] Z. Cang, Q. Nie, and Y. Zhao, Supervised optimal transport, SIAM Journal on Applied Mathematics, 82 (2022), pp. 1851–1877.

[7] L. Chapel, M. Z. Alaya, and G. Gasso, Partial optimal transport with applications on positive-unlabeled learning, Advances in Neural Information Processing Systems, 33 (2020), pp. 2903–2913.

[8] L. Chizat, G. Peyre, B. Schmitzer, and F.-X. Vialard<sup>´</sup> , An interpolating distance between optimal transport and fisher–rao metrics, Foundations of Computational Mathematics, 18 (2018), pp. 1–44.

[9] N. Courty, R. Flamary, D. Tuia, and A. Rakotomamonjy, Optimal transport for domain adaptation, IEEE Transactions on Pattern Analysis and Machine Intelligence, 39 (2017), pp. 1853–1865.

[10] M. Cuturi, Sinkhorn distances: Lightspeed computation of optimal transport, in Advances in Neural Information Processing Systems, 2013.

[11] A. Figalli, The optimal partial transport problem, Archive for Rational Mechanics and Analysis, 195 (2010), pp. 533–560.

[12] U. Frisch, S. Matarrese, R. Mohayaee, and A. Sobolevski, A reconstruction of the initial conditions of the universe by optimal mass transportation, Nature, 417 (2002), pp. 260–262.

[13] X. Gu, Y. Yang, W. Zeng, J. Sun, and Z. Xu, Keypoint-guided optimal transport with applications in heterogeneous domain adaptation, in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 14972–14985.

[14] X. Gu, Y. Yang, W. Zeng, J. Sun, and Z. Xu, Keypoint-guided optimal transport: Models, algorithms, and applications, 2026, https://arxiv.org/abs/2303.13102, https://arxiv.org/ abs/2303.13102.

[15] X. Gu, X. Yu, Y. Yang, J. Sun, and Z. Xu, Adversarial reweighting with α-power maximization for domain adaptation, International Journal of Computer Vision, 132 (2024), pp. 4768–4791.

[16] K. Guittet, Extended kantorovich norms: a tool for optimization, Tech. Report RR-4402, INRIA, 2002.

[17] L. V. Kantorovich, On the translocation of masses, in Dokl. Akad. Nauk. USSR (NS), vol. 37, 1942, pp. 199–201.

[18] J. Karlsson and A. Ringh, Generalized sinkhorn iterations for regularizing inverse problems using optimal mass transport, SIAM Journal on Imaging Sciences, 10 (2017), pp. 1935– 1962.

[19] D. Klein, G. Palla, M. Lange, M. Klein, Z. Piran, M. Gander, L. Meng-Papaxanthos, M. Sterr, L. Saber, C. Jing, et al., Mapping cells through time and space with moscot, Nature, 638 (2025), pp. 1065–1075.

[20] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, Gradient-based learning applied to document recognition, Proceedings of the IEEE, 86 (1998), pp. 2278–2324, https://doi. org/10.1109/5.726791.

[21] M. Liero, A. Mielke, and G. Savare<sup>´</sup>, Optimal entropy-transport problems and a new hellinger–kantorovich distance between positive measures, Inventiones Mathematicae, 211 (2018), pp. 969–1117.

[22] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, Flow matching for generative modeling, in The Eleventh International Conference on Learning Representations, 2023.

[23] F. Memoli <sup>´</sup> , Gromov–wasserstein distances and the metric approach to object matching, Foundations of Computational Mathematics, 11 (2011), pp. 417–487.

[24] G. Monge, M´emoire sur la th´eorie des d´eblais et des remblais, Mem. Math. Phys. Acad. Royale Sci., (1781), pp. 666–704.

[25] D. Mukherjee, A. Guha, J. M. Solomon, Y. Sun, and M. Yurochkin, Outlier-robust optimal transport, in International Conference on Machine Learning, PMLR, 2021, pp. 7850–7860.

[26] G. Schiebinger, J. Shu, M. Tabaka, B. Cleary, V. Subramanian, A. Solomon, J. Gould, S. Liu, S. Lin, P. Berube, et al., Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming, Cell, 176 (2019), pp. 928–943.e22.

[27] J. Solomon, F. De Goes, G. Peyre, M. Cuturi, A. Butscher, A. Nguyen, T. Du, and <sup>´</sup> L. Guibas, Convolutional wasserstein distances: Eficient optimal transportation on geometric domains, ACM Transactions on Graphics (ToG), 34 (2015), pp. 66:1–66:11.

[28] K.-T. Sturm, On the geometry of metric measure spaces. i, Acta Mathematica, 196 (2006), pp. 65–131, https://doi.org/10.1007/s11511-006-0002-8.