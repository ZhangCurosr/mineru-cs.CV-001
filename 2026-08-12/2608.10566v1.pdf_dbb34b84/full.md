# Iterative Erasure Count Is Not an Afine-Invariant Concept Dimension

Tingan Jin<sup>1,∗</sup> Shuhang Dong<sup>1,∗</sup> Haosong Li<sup>2,∗</sup> Chung-Hsien Chou<sup>3,†</sup>

<sup>1</sup>UCLA <sup>2</sup>Independent Researcher <sup>3</sup>Cal Poly Pomona

<sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding author.

## Abstract

How many directions does a neural representation use to encode a concept? A common operational answer repeatedly erases probe directions and reports the stopping count or cumulative removed rank. We show that both quantities can change under an invertible reparameterization that preserves all information, so neither is intrinsically a concept dimension.

We distinguish model-defined population quantities— generating dimension, suficient linear dimension, and minimum guarding rank—from procedure-defined quantities such as stopping count and cumulative edit rank. In a population Gaussian construction, an invertible shear preserves the prediction problem and all three population quantities, yet changes the cumulative Euclidean erasure count from one to two. The separation holds for Moore– Penrose ordinary least squares and every finite nonnegative ridge weight. For a two-output full-QR procedure matching the algebra of our motivating video analysis, cumulative edit rank similarly changes from two to the ambient dimension four.

Conversely, the complete cumulative metric-QR trajectory is afine-equivariant when its positive-definite metric, probe, regularizer, and tie-breaking are transported consistently; exact covariance is one corollary, not a canonical semantic metric. In a known-rank finite-sample Adam/QR calibration, identity mixing stops after one accepted update in all 20 large-sample runs, whereas each tested shear a ∈ {.5, .75, 1, 1.25, 2} accepts at least two updates in all 20 runs. Controlled reparameterizations of frozen V-JEPA2 features preserve rank-zero predictions yet alter later Euclidean trajectories under practical optimization. These visual contact experiments are stress tests, not estimates of contact dimension.

Iterative erasure therefore returns a procedure-relative estimand jointly determined by representation geometry and the full measurement procedure, not a semantic dimension by itself.

## 1 Introduction

Linear probes establish whether a label is accessible from a representation, but they do not by themselves determine how that information is organized [1, 12, 5, 22]. A common next step is iterative null-space projection: fit a linear classifier, remove its direction, refit, and count directions until a new classifier fails [24]. The original INLP paper carefully frames this as guardedness, but also describes protected-attribute subspaces as spanning “dozens to hundreds” of orthogonal directions. Later applications go further: recent work on video world models estimates feature dimensionality as the number of orthogonal probes trainable before chance and interprets tens of directions as distributed physical variables [13]. Thus the target of our critique is not merely reporting an algorithm output. It is using that output to identify intrinsic dimensionality or distributedness. This does not erase separate evidence from tuning geometry or held-out steering: those experiments can still establish behavior of a declared nativebasis subspace. Our claim is specifically that the iterative count cannot supply a coordinate-free dimension for that evidence.

That interpretation conflates several distinct quantities. A label may be generated from one latent scalar while remaining predictable from many observed coordinates because an invertible mixing matrix correlates those coordinates. Conversely, a rank-one empirical crosscovariance can be canceled in one afine edit without proving that one population-suficient direction has been found. The erasure path also depends on the feature metric, probe loss, regularizer, covariance estimate, and stopping rule. These are properties of an analysis procedure, not of a semantic concept alone.

Closed-form work already makes covariance central. Mean Projection and LEACE can guard linear firstmoment access with less collateral change than iterative null-space projection [11, 6, 9]. We do not propose another eraser, and one-step Euclidean non-equivariance is elementary. Gauge-freedom work likewise shows that invertible reparameterizations preserve a network function while changing Euclidean representation geometry [7]. Our sharper contribution is the discrete identification result those observations do not provide: an exact integer-count change with fixed suficient and guarding rank, its full-QR counterpart, and an equivariance theorem for the complete transported-metric trajectory. A finite reproduction of the published QR stopping protocol and a map-specific fresh-attacker audit bridge those results to practice. We ask: what, if anything, can an iterative erasure count identify about concept dimension? Our answer is negative unless a metric and estimand are

fixed in advance.

We combine a population construction with a visual case study. The construction has a known rank-one suficient concept, independent nuisance variables, and a family of invertible mixings with controlled condition number. It separates coordinate-induced Euclidean redundancy from the known population guard. The case study uses hand–object contact, a useful predicate for manipulation and action understanding [28, 20], in frozen V-JEPA2 and DINOv2 representations [2, 21]. Contact is not ofered as a pure semantic primitive: 100DOH contains object-configuration shortcuts, and TouchMoment includes approach and action phase. Those limitations make it a demanding audit target rather than evidence for a universal contact mechanism.

The visual study is deliberately a case study rather than a confirmatory contact claim. Validation-only layer sweeps select the frozen representations to audit; all coordinate-stress conclusions are reported as conditional on those choices. Optional native-basis channel analyses are confined to the supplement and do not support the central identification result.

Our contributions are:

1. We prove count non-identification for population ridge probes and give a full-QR two-output construction with cumulative edit rank two or four despite fixed suficient dimension and guarding rank. We then reproduce the motivating finite Adam/MSE/QR recurrence and show held-out count changes on the same circular targets under invertible shears.

2. We prove afine equivariance of the complete trajectory for any covariantly transported positive-definite metric, recover exact covariance as a corollary, and delimit the result under rank-deficient probe fitting.

3. We apply predeclared anisotropic and orthogonal maps to unchanged V-JEPA2 features and show that the trajectory changes under map-specific, perrank fresh attackers trained and evaluated on sourcedisjoint oficial-training roles.

4. We separate population dimension, moment rank, guarding rank, iteration count, and cumulative edit rank in a reproducible audit with no new oficial-test claims.

## 2 Related Work

Probing and intervention. Linear probes reveal accessible information but can exploit confounds and do not by themselves establish causal use. Probe conclusions depend on the attacker and complexity control [12, 5, 22]. INLP repeatedly projects out classifier directions [24]. Amnesic probing uses the resulting cumulative subspace as a counterfactual intervention and controls downstream efects against removal of the same number of random directions [10]. In high-dimensional NLI, Rozanova et al. show that a small removed subspace relative to ambient dimension and high-variance random-direction controls can obscure amnesic conclusions, motivating a complementary mnestic analysis [27]. Mean Projection (MP) and LEACE instead provide one-shot lower-rank or minimum-distortion guards [11, 6, 9]. Haghighatkhah et al. find that after a targeted projection achieves linear guarding, further INLP efects can resemble additional random projections; later comparisons likewise document extra removed dimensions and collateral change [11, 9]. These studies use removed-direction counts to size interventions or controls, not to define an afine-invariant concept dimension. LEACE characterizes linear guardedness through feature–concept cross-covariance and provides a closed-form minimum-distortion map with explicit afine structure. Spectral Attribute Removal (SAL) removes the left singular subspace of feature–concept cross-covariance. For a scalar binary concept, its full-rank subspace is the one-dimensional mean-diference subspace [29, 6]. Our novelty is not a covariance-aware eraser or Proposition 1 alone. LEACE supplies a one-shot minimum-distortion guard. We instead give an exact integer-count change, extend it to full-QR multivariate removal, characterize equivariance of the complete transported-metric trajectory, and separate residual access caused by finite estimation from population rank. Exact covariance is one corollary of the metric theorem, not a new eraser. Minimax erasure such as RLACE answers a diferent, explicitly attacker-relative optimization problem [25]. It reinforces rather than removes the need to declare the estimand.

Concept geometry and non-identifiability. Concept Activation Vectors operationalize human concepts as feature-space directions in vision models [14]. Such directions can be useful in a fixed representation, but an invertible map preserves information and all linear scores while changing Euclidean angles. This is one instance of a broader identification problem: representation factors and their geometry are not recoverable from observational fit without assumptions or inductive bias [19]. Under stated diversity conditions, Roeder et al. prove that a broad discriminative model family is identifiable only up to invertible linear transformations [26]. Thus a quantity claimed to be intrinsic across equivalent learned representations must either survive this ambiguity or declare additional geometry. Cain formalizes the related gauge-freedom view and shows that Euclidean similarity can change under function-preserving invertible maps [7]. Our claim is narrower and discrete: for a specified iterative eraser, the exact integer population stopping count changes while both suficient dimension and minimum guarding rank remain fixed.

Frozen visual representations. Predictive and masked objectives learn transferable image and video features [31, 3, 4, 2]. V-JEPA2 predicts latent video targets, while DINOv2 provides a strong image-only comparison [2, 21]. Prior egocentric work studies actions, objects, and hand state [23, 30, 32]. We study the stability of a geometric interpretation rather than claiming a new contact detector.

Table 1: Claim boundary and novelty. The paper revises a dimensionality interpretation; it does not invalidate every result produced alongside an erasure count.
<table><tr><td>Prior object</td><td>What remains valid</td><td>What must be reinterpreted</td><td>What is new here</td></tr><tr><td>INLP [24]</td><td>Linear guardedness and the edited representation under a declared native-basis procedure</td><td>Iteration count is not therebv an intrin- sic protected-attribute dimension</td><td>Exact count-one/count-two separa- tion with fixed sufficient and guard- ing rank</td></tr><tr><td>MP, SAL, LEACE [11, 29, 6]</td><td>One-shot first-moment or affine- linear guarding with an explicit dis- tortion geometry</td><td>Covariance is a chosen geometry, not a universally semantic one</td><td>Complete transported-metric tra- jectory theorem and rank-deficient scope</td></tr><tr><td>Gauge freedom [7]</td><td>Euclidean similarities can change under function-preserving invert- ible maps</td><td>General metric dependence alone does not state that a discrete stopping count</td><td>Scalar and full-QR integer-count counterexamples</td></tr><tr><td>Video direction study [13]</td><td>Layerwise access, tuning struc- ture, and held-out steering remain native-basis evidence</td><td>changes Orthogonal-probe count supports only procedure-relative, not affine-invariant, distributedness</td><td>Matched circular Adam/MSE/QR finite stopping stress test</td></tr></table>

## 3 Rank Estimands and Afine Dependence

## 3.1 Five Quantities Called “Dimension”

Let $X \in \mathbb { R } ^ { d }$ be a representation and Y a concept label. Table 2 separates quantities that need not agree. Generating and suficient dimensions require assumptions about a latent population model. Cross-covariance rank is a first-moment property. For a scalar encoding $\phi ( Y )$ , it is at most one. Iterative erasure count is the number of noncollapsed sequential probe updates before a declared stopping rule fires; cumulative edit rank is rank $\left( I - T _ { k } \right)$ after update k. When $T _ { k }$ is an idempotent projector, this equals the dimension of its removed subspace. For a nonprojector recurrence it is only the rank of the cumulative edit operator, not a removed- subspace dimension. Count and edit rank coincide only when every update adds one independent direction. Linear guarding rank is the minimum intervention rank needed to satisfy a specified guardedness criterion. Formally, for a declared family of label distributions, define suficient linear dimension as the minimum r for which some $B \in \mathbb { R } ^ { d \times r }$ satisfies $Y \perp X \mid X B$ . For a declared attacker class ${ \mathcal { F } } _ { : }$ , a map T is F-guarding at risk $R _ { 0 }$ when inf $f { \in } \mathscr { F } R ( f ( X T ) , Y ) \geq R _ { 0 }$ Linear guarding rank is the minimum rank $( I - T )$ satisfying that criterion. These definitions depend on the population law, attacker class, loss, and baseline risk.

Guardedness itself also needs a declared strength. Zero empirical Cov( d X, Y ), or equal empirical class means for binary Y, is a sample first-moment condition. Failure of one fitted logistic or SVM attacker is objective- and sample-relative. Failure of every afine linear classifier is a class-wide property, while statistical independence of edited features and Y is stronger still. Mean Projection, SAL, and LEACE primarily target first-moment or afinelinear guardedness. Our population proposition uses full independence. We treat finite-attacker failure only as evidence about the stated protocol, never as proof of independence.

Table 2: Distinct estimands that should not be reported under one name.
<table><tr><td>Quantity</td><td>Definition or dependency</td></tr><tr><td>Generating di- mension</td><td>Number of latent variables used by the data- generating label rule</td></tr><tr><td>Sufficient linear dimension</td><td>Minimum population linear subspace sufficient for Y under a stated model</td></tr><tr><td>Cross-covariance rank</td><td>rank Cov(X, φ(Y)) for a declared encoding φ, cap- turing only a first moment</td></tr><tr><td>Iterative erasure count</td><td>Stopping time; report cumulative edit rank sepa- rately for multivariate updates</td></tr><tr><td>Linear guarding rank</td><td>Minimum linear edit rank subject to a stated guardedness condition</td></tr></table>

Algorithm 1: Declared cumulative metric-QR era  
sure. Given centered features X, $G \succ 0$ fixed from the   
intact representation, a probe rule, and a stopping rule,   
set $U _ { 0 } = [ ]$ and $T _ { 0 } = I .$ For $k = 1 , 2 , \ldots { }$   
1. Fit coeficient block $B _ { k } = \mathrm { P r o b e } ( X T _ { k - 1 } , Y )$   
2. Residualize $B _ { k }$ against $U _ { k - 1 }$ and metric-QR it to a   
G-orthonormal new block $Q _ { k } .$   
3. If $Q _ { k }$ is empty or the declared held-out rule fires,   
stop; otherwise set $U _ { k } = [ U _ { k - 1 } , Q _ { k } ]$ and $T _ { k } = I -$   
$U _ { k } { \bar { U } } _ { k } ^ { \top } G .$   
4. Record iteration count and rank $\left( I - T _ { k } \right)$ separately.   
Every metric, probe, regularizer, residualization rule, nu  
merical tolerance, and stopping rule is part of the esti  
mand. The metric is not recomputed after editing.

The transported-metric theorem covers this cumulative procedure.   
The finite circular reproduction instead follows the motivating   
paper’s literal sequential projector recurrence and is labeled   
separately.

## 3.2 Euclidean Erasure Is Not Afine Invariant

For a row feature x and invertible A, write $z = x A$ . A coeficient w in the original coordinates has equivalent transformed coeficient $w _ { z } = A ^ { - 1 } w$ because $z w _ { z } = x w .$ Projecting Euclideanly in transformed coordinates and

mapping back gives

$$
x ^ { \prime } = x A \left( I - \frac { w _ { z } w _ { z } ^ { \top } } { w _ { z } ^ { \top } w _ { z } } \right) A ^ { - 1 } .\tag{1}
$$

Equation 1 generally difers from $x ( I - w w ^ { \top } / ( w ^ { \top } w ) )$ unless A preserves the relevant Euclidean metric. Thus an invertible reparameterization preserves all pre-edit linear predictions but changes the erased subspace and every later refitted direction.

Proposition 1 (Coordinate dependence). For $d \geq 2$ and nonzero w, there exists an invertible A for which the mapped-back Euclidean edit in Equation 1 difers from the Euclidean edit in the original coordinates, although $x w = ( x A ) ( A ^ { - 1 } w )$ for every x.

Proof. Choose a basis with $w = e _ { 1 }$ and let the leading $2 \times 2$ block of A be $\textstyle { \left[ \begin{array} { l l } { 1 } & { 0 } \\ { a } & { 1 } \end{array} \right] }$ with $a \neq 0$ . Then $A ^ { - 1 } w = ( 1 , - a ) ^ { \top }$ and direct substitution into Equation 1 produces ofdiagonal terms absent from $I - e _ { 1 } e _ { 1 } ^ { \top }$ . Score equivalence follows from invertibility. Rotating this basis handles arbitrary nonzero w. □

This observation also clarifies the regularization confound. Fitting an isotropic $\ell _ { 2 }$ probe independently in x and z changes the prior. The transformed penalty equivalent to $\lVert \boldsymbol { w } \rVert _ { 2 } ^ { 2 }$ is quadratic, not a newly selected scalar $C \| w _ { z } \| _ { 2 } ^ { 2 }$ In our metric comparison, every attacker is therefore fit in the same train-standardized original coordinates with one validation-selected C. Whitening is used only to define an intervention, which is then mapped back. Rank zero agrees to numerical precision (maximum absolute discrepancy below $7 \times 1 0 ^ { - 1 4 } )$ .

Here G is a positive-definite quadratic form on probe coeficients, equivalently on the dual representation space. Because coeficients transform as $w _ { Z } = A ^ { - 1 } w _ { X }$ , preserving $w ^ { \top }$ Gw requires the congruence $G _ { Z } = A ^ { \top } G _ { X } A$ . A metric acting instead on primal row features would obey the inverse-congruence rule.

Proposition 2 (Complete transported-metric afine equivariance). Let $G _ { X } \succ 0$ be any coeficient metric fixed from the intact representation. At iteration k, let the probe return a coeficient block $B _ { k }$ (one column $f o r a$ scalar target), append its metric-orthogonalized column space to a $G _ { X }$ -orthonormal basis $U _ { k }$ , and edit by $T _ { k } =$ $I - U _ { k } U _ { k } ^ { \top } G _ { X }$ . For $Z = X A$ with invertible A, transport the metric as $G _ { Z } = A ^ { \top } G _ { X } A$ . If the probe is equivariant $( B _ { k } ^ { Z } \ = \ A ^ { - 1 } B _ { k } )$ , regularization is transported under A, and any non-subspace tie is resolved equivariantly, then for every k, $Z T _ { k } ^ { Z } \ = \ ( X T _ { k } ) A$ . Scores, cumulative edit rank, and every score-based stopping count are identical in both parameterizations.

Proof. If $U _ { k } ^ { Z } = A ^ { - 1 } U _ { k }$ , then $( U _ { k } ^ { Z } ) ^ { \top } G _ { Z } U _ { k } ^ { Z } = U _ { k } ^ { \top } G _ { X } U _ { k } =$ I and

$$
A T _ { k } ^ { Z } = A \left[ I - A ^ { - 1 } U _ { k } U _ { k } ^ { \top } A ^ { - \top } A ^ { \top } G _ { X } A \right]\tag{2}
$$

$$
= \left[ I - U _ { k } U _ { k } ^ { \top } G _ { X } \right] A = T _ { k } A .\tag{3}
$$

The claim follows by induction: the edited features correspond by A, probe equivariance maps the next coefficient block by $A ^ { - 1 }$ , and metric Gram–Schmidt preserves the basis relation. Transporting an original penalty $\lVert \boldsymbol { w } \rVert _ { 2 } ^ { 2 }$ means using $w _ { Z } ^ { \top } A ^ { \top }$ Aw<sub>Z</sub>, not an isotropic penalty in Z. □

Exact covariance is a natural corollary: setting $G _ { X } =$ $\Sigma _ { X }$ gives $G _ { Z } = \Sigma _ { Z } = A ^ { \top } \Sigma _ { X } A$ . Proposition 2 does not privilege covariance as a semantic geometry; any fixed positive-definite metric transported by the same tensor law obeys the result. OAS shrinkage toward identity [8] and eigenvalue flooring generally break it because their covariance maps f do not satisfy $f ( A ^ { \top } \Sigma A ) = A ^ { \top } f ( \Sigma ) A$

The probe assumption needs care after the first edit, when ambient features are rank deficient. Strictly convex least-squares or logistic objectives with an unpenalized intercept and a positive-definite quadratic penalty transported by A retain a unique equivariant coeficient even after rank loss. Unregularized OLS is equivariant at a fullrank intact iteration. In contrast, rank-deficient Moore– Penrose OLS selects the minimum Euclidean-norm ambient coeficient; that selector is generally not equivariant under a nonorthogonal A and is not covered by Proposition 2. Theorem 2 uses Moore–Penrose only to define one displayed-coordinate procedure and makes no afineequivariance claim. Our empirical unshrunk-covariance check instead uses a strictly regularized probe after mapping every edited feature back to the common original coordinates, so no independent pseudoinverse selector is compared across parameterizations.

For a multivariate probe, removing the entire column space of $B _ { k }$ makes the update independent of the particular QR basis. Extracting one vector from a tied singular subspace is not basis-independent and requires an additional declared rule. Proposition 2 does not make that choice canonical.

The statement extends from $Z = X A$ to a full afine change $Z = X A + b$ by tracking the mean, applying each edit to centered features, and transforming the intercept: $\mu _ { Z } = \mu _ { X } A + b$ and $c _ { Z } = c _ { X } - b A ^ { - 1 } w$ . The metric proof is then unchanged. When an exact covariance metric is singular, the corollary applies only after a common support subspace has been fixed. A pseudoinverse, eigenvalue floor, or shrinkage rule adds an estimator-dependent extension outside that support and is not covered by the equivariance claim.

## 3.3 Population Rank-One Construction

Let $H = ( S , N ) \sim \mathcal { N } ( 0 , I _ { d } )$ , with scalar concept coordinate S and independent nuisance N. Define $Y = \mathbb { W } [ S \geq 0 ]$ and observed features $X = H A$ for invertible A. The generating and minimum suficient linear dimensions are one. Let $e = ( 1 , 0 , \ldots , 0 ) ^ { \top }$ . Moreover,

$$
T _ { A } = A ^ { - 1 } ( I - e e ^ { \top } ) A\tag{4}
$$

has rank $\ ( I - T _ { A } ) = 1$ , and $X T _ { A } = H ( I - e e ^ { \top } ) A$ is independent of $Y$ . Hence the population minimum guarding rank is exactly one. Varying A changes no labels, latent variables, or suficient dimension.

Proposition 3 (Known minimum guard). For the construction above, the minimum rank of a linear edit that makes the edited representation independent of Y is one.

Proof. $T _ { A }$ gives a rank-one upper bound and removes $S$ exactly. A rank-zero edit is the identity. Because A is invertible, $S ~ = ~ X A ^ { - 1 } e$ remains a deterministic linear function of $X ,$ and $Y = \mathbb { K } [ S \geq 0 ]$ is nonconstant. Thus rank zero cannot guard Y, establishing the matching lower bound. □

Non-equivariance of one update does not alone prove that a stopping count changes. The following population separation does.

Theorem 1 (Cumulative-QR ridge count non-identification). Let $S , N \overset { \mathrm { i i d } } { \sim } \mathcal { N } ( 0 , 1 ) , L = \mathrm { s i g n } ( S )$ , and $X _ { a } = ( S +$ $a N , N )$ for $a \in \mathbb { R }$ . Maintain a Euclidean-orthonormal cumulative basis $U _ { k }$ and projector $P _ { k } = I - U _ { k } U _ { k } ^ { \top }$ , beginning with $P _ { 0 } = I$ . At iteration k, fit population ridge least squares on $X _ { a } P _ { k - 1 }$ , residualize its ambient coeficient as $q _ { k } = P _ { k - 1 } w _ { k }$ , append $q _ { k } / \lVert q _ { k } \rVert _ { 2 }$ when nonzero, and stop when the population coeficient is zero. Fix any finite ridge weight $\lambda \geq 0 . A t \lambda = 0$ , every rank-deficient fit uses the Moore–Penrose minimum-norm coeficient. For $\lambda > 0$ , the penalty may be either isotropic in each displayed coordinate system or transported from $X _ { 0 }$ under $X _ { a } ~ = ~ X _ { 0 } A _ { a }$ The iteration count and cumulative edit rank are one $f o r a = 0$ and two for every $a \neq 0 .$ , although every $X _ { a }$ has suficient linear dimension one and minimum linear guarding rank one.

Proof. Writing $\begin{array} { l l l l } { c } & { = } & { \mathbb { E } [ S \mathrm { s i g n } ( S ) ] } & { = } & { \sqrt { 2 / \pi } } \end{array}$ gives $\operatorname { C o v } ( X _ { a } , L ) = c ( 1 , 0 ) ^ { \top }$ . The ridge coeficient is proportional to

$$
\begin{array} { r } { [ \operatorname { C o v } ( X _ { a } ) + \lambda I ] ^ { - 1 } \operatorname { C o v } ( X _ { a } , L ) \propto ( 1 + \lambda , - a ) ^ { \top } . } \end{array}\tag{5}
$$

This is the isotropic-penalty solution. Writing $A _ { a } = [ \ O _ { a } ^ { 1 } \ O _ { 1 } ^ { 0 } ]$ the transported penalty is $w _ { a } ^ { \top } A _ { a } ^ { \top } A _ { a } w _ { a }$ Its solution is proportional to $A _ { a } ^ { - 1 } ( 1 , 0 ) ^ { \top } \overset { ^ { \textstyle } } { = } ( 1 , - a ) ^ { \top }$ . Both solutions are parallel to the feature–label cross-covariance vector $( 1 , 0 ) ^ { \top }$ only when $a = 0 . \mathrm { A t } \lambda = 0$ , the isotropic and transported solutions themselves coincide for every a. For $a = 0$ , this common direction equals the cross-covariance direction and one edit leaves independent N. For $a \neq 0$ neither fitted direction is parallel to the cross-covariance vector $( 1 , 0 ) ^ { \top }$ , so the first cumulative Euclidean removal leaves nonzero feature–label cross-covariance in a retained line spanned by a unit vector r. Write $v = \mathrm { V a r } ( X _ { a } r ) > 0$ $g = \mathrm { C o v } ( X _ { a } r , L ) \ne 0$ , and let $\Gamma \succ 0$ be either the identity or transported penalty matrix. For $\lambda > 0$ , the second ambient coeficient satisfies

$$
w _ { 2 } = g ( \lambda \Gamma + v r r ^ { \top } ) ^ { - 1 } r ,\tag{6}
$$

$$
r ^ { \top } w _ { 2 } = g r ^ { \top } ( \lambda \Gamma + v r r ^ { \top } ) ^ { - 1 } r \neq 0 .\tag{7}
$$

Hence its cumulative residual $q _ { 2 } = P _ { 1 } w _ { 2 } = ( r ^ { \top } w _ { 2 } ) r$ is nonzero and spans the retained support. $\mathrm { A t } ~ \lambda ~ = ~ 0$ the Moore–Penrose solution gives the same conclusion, $q _ { 2 } = ( g / v ) r$ . Appending this residual exhausts the twodimensional representation, so the next coeficient is zero. The rank-one guard $( S + a N , N ) \mapsto ( 0 , N )$ establishes the common guarding rank. For fixed $a \neq 0$ and finite λ, the residual coeficient has positive norm, so the same separation holds for a coeficient-based rule that accepts update k when $\| q _ { k } \| _ { 2 } > \tau$ , provided τ is below the three relevant nonzero norms $\| q _ { 1 } ( a = 0 ) \| _ { 2 } , \| q _ { 1 } ( a ) \| _ { 2 }$ , and $\| q _ { 2 } ( a ) \| _ { 2 }$ . This statement does not cover a score- or risk-based stopping threshold. □

The exact count jump should not be confused with a uniformly large practical efect. Its magnitude is explicit. Let $D _ { \lambda } ^ { 2 } = ( 1 { + } \lambda ) ^ { 2 } { + } a ^ { 2 }$ . Under isotropic ridge, the retained unit line after the first edit is $r _ { \lambda } = ( a , 1 + \lambda ) ^ { \top } / D _ { \lambda }$ , with

$$
g _ { \lambda } = \mathrm { C o v } ( X _ { a } r _ { \lambda } , L ) = \frac { c a } { D _ { \lambda } } ,\tag{8}
$$

$$
v _ { \lambda } = \mathrm { V a r } ( X _ { a } r _ { \lambda } ) = \frac { a ^ { 2 } + ( a ^ { 2 } + 1 + \lambda ) ^ { 2 } } { D _ { \lambda } ^ { 2 } } .\tag{9}
$$

Because $\mathrm { V a r } ( L ) = 1$ , the best retained scalar linear predictor has

$$
R _ { \mathrm { r e s } } ^ { 2 } = { \frac { g _ { \lambda } ^ { 2 } } { v _ { \lambda } } } = { \frac { ( 2 / \pi ) a ^ { 2 } } { a ^ { 2 } + ( a ^ { 2 } + 1 + \lambda ) ^ { 2 } } } ,\tag{10}
$$

$$
\| q _ { 2 } \| _ { 2 } = \frac { | g _ { \lambda } | } { v _ { \lambda } + \lambda } = \frac { c | a | D _ { \lambda } } { a ^ { 2 } + ( a ^ { 2 } + 1 + \lambda ) ^ { 2 } + \lambda D _ { \lambda } ^ { 2 } } .\tag{11}
$$

For transported ridge, replace these by $\begin{array} { r l } { R _ { \mathrm { r e s } } ^ { 2 } } & { { } = } \end{array}$ $( 2 / \pi ) a ^ { 2 } / [ a ^ { \overset { . } { 2 } } + ( 1 + a ^ { 2 } ) ^ { 2 } ]$ and $\| q _ { 2 } \| _ { 2 } = c | a | \sqrt { 1 + a ^ { 2 } } / [ a ^ { 2 } +$ $( 1 + \lambda ) ( 1 + a ^ { 2 } ) ^ { 2 } ]$ . Both are positive for every $a \neq 0$ but approach zero continuously as $a  0 ;$ ; the coeficient norm also shrinks with strong ridge. Thus a finite score threshold can legitimately return one even where exact population coeficient stopping returns two. The finite experiment below measures that bridge rather than hiding it. The norm in Equation 11 is specifically the cumulative residualized coeficient $q _ { 2 }$ , not the raw ambient coeficient w<sub>2</sub>.

The cumulative residualization is essential. Literal sequential multiplication by null projectors of the raw ambient coeficients need not terminate in two steps under a transported anisotropic penalty and can revisit an earlier support; that diferent algorithm is not covered by Theorem 1.

The motivating video analysis fits a two-output regressor and removes the full QR basis of its weight matrix. The next result matches that algebraic procedure.

Theorem 2 (Full-QR multivariate count non-identification). Let $S , N \ \stackrel { \mathrm { i i d } } { \sim } \ N ( 0 , I _ { r } )$ , let $Y = S$ , and let $X _ { a } \ =$ $( S + a N , N ) \in \mathbb { R } ^ { 2 r }$ . At each iteration, fit population multivariate least squares, use the Moore–Penrose solution if the edited covariance is singular, remove an orthonormal basis for the full coeficient column space by $\ Q R ,$ and stop at zero feature–target cross-covariance. For $a = 0$ , the iteration count is one and cumulative edit rank is $r .$ For every $a \ne 0$ , the iteration count is two and cumulative edit rank is $2 r$ , the ambient dimension. In both cases the cumulative map is an orthogonal projector, so this edit rank also equals its removed-subspace dimension. Yet the suficient linear dimension and minimum independenceguarding rank are r for every a.

Proof. The coeficient block $\boldsymbol { B } _ { \star } ~ = ~ [ I _ { r } , - a I _ { r } ] ^ { \top }$ satisfies $X _ { a } B _ { \star } = S = Y$ , giving suficient linear dimension at most $r .$ For the lower bound, let $Z = X _ { a } D \in \mathbb { R } ^ { m }$ be any linear statistic suficient for $Y$ . Because $Y$ is a deterministic function of $X _ { a } ,$ conditional independence $Y \perp X _ { a } \mid Z$ requires $\operatorname { C o v } ( S \mid Z ) = 0$ . Joint Gaussianity gives

$$
\operatorname { C o v } ( S \mid Z ) = I _ { r } - \operatorname { C o v } ( S , Z ) \operatorname { V a r } ( Z ) ^ { + } \operatorname { C o v } ( Z , S ) .
$$

The subtracted term has rank at most $m ;$ equaling $I _ { r }$ therefore requires $m \geq r$ . Hence the suficient linear dimension is exactly r.

The first coeficient block is $\boldsymbol { B } = [ I _ { r } , - a I _ { r } ] ^ { \top }$ . For $a = 0 ,$ its removal leaves only N, which is independent of $Y .$ . For $a \neq 0$ , the orthogonal complement of $\operatorname { c o l } ( B )$ is spanned by $\boldsymbol { C } = [ a I _ { r } , I _ { r } ] ^ { \top }$ . The retained coordinates are proportional to $X _ { a } C \stackrel { . } { = } a S + ( 1 + a ^ { 2 } ) N$ , whose cross-covariance with Y is $a I _ { r }$ and therefore has rank r. More explicitly, with $P _ { C } = C ( C ^ { \top } C ) ^ { - 1 } C ^ { \top }$ , the edited covariance $\widetilde \Sigma = { \cal P } _ { C } \mathrm { C o v } ( X _ { a } ) { \cal P } _ { C }$ is positive definite on $\operatorname { c o l } ( C )$ , and $\operatorname { C o v } ( X _ { a } P _ { C } , Y ) = a C / ( 1 + a ^ { 2 } )$ has rank r. The Moore– Penrose OLS block $\widetilde \Sigma ^ { + } \operatorname { C o v } ( X _ { a } P _ { C } , Y )$ therefore has rank r, lies in col(C), and thus spans all of col(C). The second QR removal exhausts the representation.

An edit that sets the S component to zero gives a rankr independent guard. Conversely, independence in this jointly Gaussian model requires zero feature–target crosscovariance. If $T$ guards $Y$ , then $( I - T ) ^ { \top } \operatorname { C o v } ( X _ { a } , Y ) =$ Cov $( X _ { a } , Y )$ , so rank(I−T) ≥ rank Cov $( X _ { a } , Y ) = r$ . Thus the minimum guarding rank is exactly r. □

For the circular-regression output width $r = 2 ,$ , Theorem 2 changes the reported QR direction count from two to four. It matches the two-output MSE and full-QR removal rule, but not a particular angular data distribution or finite held-out stopping threshold.

## 3.4 Finite Circular-QR Bridge

We therefore implement the motivating finite procedure on a controlled angular target. Draw $\theta \sim \mathrm { U n i f } [ - \pi , \pi ]$ set $S = ( \sin \theta , \cos \theta )$ , draw $N \sim \mathcal { N } ( 0 , . 5 I _ { 2 } )$ to match the per-coordinate variance of S, and form $X _ { a } = ( S + a N , N )$ The same latent draws, labels, and probe initialization are reused across shears. Every $X _ { a }$ has a two-dimensional suficient statistic and an explicit rank-two guard that removes S while retaining $N$

Following the published specification [13], each fresh linear probe predicts (sin θ, cos θ) with MSE and Adam [15] for 100 epochs at learning rate $1 0 ^ { - 3 }$ and weight decay

$1 0 ^ { - 4 } . \mathrm { ~ A n ~ } 8 0 / 2 0$ split is fixed for the entire sequence. We apply the literal recurrence $X ^ { ( k + 1 ) } = X ^ { ( k ) } ( \bar { I _ { \mathrm { ~ - ~ } } } Q _ { k } Q _ { k } ^ { \top } )$ where $Q _ { k }$ is the full QR basis of the two-column coeficient block. An update is accepted only when held-out $R ^ { 2 } \geq . 1$ and circular $\mathrm { M A E } \leq 8 0 ^ { \circ } ; K$ counts accepted updates before the first failing probe. The source does not report minibatch size, so the primary run declares 128 and repeats the decisive $n = 4 , 0 0 0 , a \in \{ 0 , 1 \}$ comparison at 64 and 256. We record both rank $\left( I - T _ { k } \right)$ and $\mathrm { r a n k } ( T _ { k } )$ rather than equating 2K with independent removed rank. We call the former cumulative edit rank because the literal $T _ { k }$ need not be idempotent. Coeficient QR uses absolute tolerance $1 0 ^ { - 8 }$ , and both reported map ranks use absolute singular-value tolerance $1 0 ^ { - 6 }$ . Twenty seeds are used per primary cell, with an eight-update audit cap.

We verify Theorem 1 for $\lambda \in \{ 0 , . 1 , 1 , 1 0 \}$ . A complementary population control uses $H \sim \mathcal { N } ( 0 , I _ { 8 } ) , X = H A$ and continuous target $Y = \left( H _ { 1 } , . 7 H _ { 2 } \right)$ . Its suficient dimension and minimum guarding rank are two. At each step, multivariate least squares is refit and the leading left singular vector of its coeficient matrix is removed, with unequal target scales fixing tie-breaking. Stopping requires zero feature–target cross-covariance. Across 50 predeclared Gaussian-QR orientations per condition, singular values of A are geometrically spaced between $\kappa ^ { - 1 / 2 }$ and $\kappa ^ { 1 / 2 }$ The scalar theorem verification uses relative cross-covariance, coeficient, and numerical-rank tolerance $1 0 ^ { - 1 0 }$ The ambient-eight control uses relative Frobenius cross-covariance and direction-collapse tolerance $1 0 ^ { - 9 }$ and Moore–Penrose cutof $1 0 ^ { - 9 }$

## 4 Exploratory Visual Case-Study Protocol

The central non-identification claim is established by the population and controlled finite constructions above. The visual analyses ask only whether the same procedural sensitivity is visible in frozen features; they do not estimate a contact dimension or provide an untouched confirmatory benchmark.

## 4.1 Tasks and Representations

100DOH labels an image positive when any annotated hand has contact state $3 / 4$ and negative only when all hands have state 0 [28]. Ambiguous states 1/2 are excluded. The balanced audit subset has 1,600/200/200 train/validation/test images with released split boundaries and supports matched V-JEPA2/DINOv2 audits. A separate maximal balanced V-JEPA2 audit uses every 10,645 eligible negative and an equal number of positives: 17,206/2,126/1,958 examples and 13,081 source videos, with no video crossing splits. TouchMoment is constructed from HOI4D and the egocentric subset of TACO [18, 17, 20]. Positives end at a touch onset. Negatives end 0.5 seconds earlier in the same video, with hand visibility required and other touches excluded from each causal window. It contains 11,564/618/2,588 examples and 1,294 test pairs from 647 videos. Videos do not cross splits.

Table 3: Theorems and experiments analyze related but distinct declared procedures. No row silently inherits guarantees from another.
<table><tr><td>Object</td><td>Update and probe</td><td>Stop</td><td>What it establishes</td></tr><tr><td>Theorem 1</td><td>Cumulative Euclidean Gram-Schmidt; scalar population ridge</td><td>Zero population coefficient</td><td>Exact count 1 versus 2; excludes raw sequential projectors</td></tr><tr><td>Theorem 2</td><td>Full coefficient-column QR; population OLS with declared Moore-Penrose se- lector</td><td>Zero feature-target covariance</td><td>Cumulative edit rank r versus 2r; no equivariance claim</td></tr><tr><td>Finite circular bridge</td><td>Literal sequential QR; Adam/MSE, two outputs, fixed held-out split</td><td>Published  $R ^ { 2 } / \mathrm { c i r c u l a r } { \mathrm { - } } \mathrm { M A E }$  thresholds</td><td>Practical stopping sensitivity and count/edit-rank separation</td></tr><tr><td>Visual case study</td><td>Cumulative scalar Gram-Schmidt; LBFGS logistic eraser and fresh at- tacker</td><td>Fixed audited ranks; descriptive validation stopping secondary</td><td>Procedure dependence on un- changed frozen visual features</td></tr></table>

All 100DOH splitting and resampling groups by source video. TouchMoment groups by source video, preserving each onset/pre-onset pair. Unless an analysis reports split-seed quantiles, uncertainty uses 1,000 groupbootstrap replicates. No row- or frame-level bootstrap is used.

We freeze facebook/vjepa2-vitl-fpc64-256, meanpool hidden tokens, and use validation-selected layers 21 (100DOH) and 18 (TouchMoment). DINOv2-Base uses candidate images and layers 12 and 11. Feature dimensions are 1,024 and 768. AdamW probes with seeds 17/23/42 are used only for the initial layer sweep. All rank audits use deterministic LBFGS logistic probes or LinearSVC.

The primary label is contact. Predicate, nuisance, temporal, random-subspace, and stability audits are retained in the artifact but are secondary to identification.

## 4.2 Metric, Estimator, and Stopping Controls

Features are standardized using oficial-training statistics. We compare Euclidean intervention geometry with empirical, Oracle Approximating Shrinkage (OAS), and Ledoit–Wolf covariance maps [8, 16]. Eigenvalues are floored at $1 0 ^ { - 4 } \lambda _ { \operatorname* { m a x } }$ where needed. Unshrunk empirical covariance, when full rank, uses neither shrinkage nor a floor. Unlike shrinkage toward identity or flooring, it is afine-equivariant. We report the full eigenspectrum, effective and stable rank, pre-floor positive-spectrum condition number, post-floor condition number, floored count, and direction alignment. Five independent stratified covariance subsamples are drawn at every non-full sample size.

The metric-aware iteration is fixed explicitly. Let $W =$ I for Euclidean geometry or $W = \widetilde { \Sigma } ^ { - 1 / 2 }$ for a covariance metric, where $\widetilde { \Sigma }$ and its eigenvalue floor are estimated once from intact training features. Set $Z _ { 0 } = X W$ and $U _ { 0 }$ empty. At iteration $k ,$ fit coeficient w<sub>k</sub> on $X _ { k } = Z _ { k } W ^ { - 1 }$ with one intact validation-selected $C ,$ map it to metric coordinates as $a _ { k } = W ^ { - 1 } w _ { k }$ , and compute

$$
q _ { k } = { \big ( } I - U _ { k - 1 } U _ { k - 1 } ^ { \top } { \big ) } a _ { k } , \quad u _ { k } = q _ { k } / \| q _ { k } \| _ { 2 } ,\tag{12}
$$

$$
U _ { k } = [ U _ { k - 1 } , u _ { k } ] , \qquad Z _ { k } = Z _ { 0 } ( I - U _ { k } U _ { k } ^ { \top } ) .\tag{13}
$$

We stop if $q _ { k }$ numerically collapses. Covariance is never recomputed from edited features, so singular edited covariances do not enter later steps. The cumulative original-coordinate map is $T _ { k } = \bar { W ( I - U _ { k } U _ { k } ^ { \top } ) } W ^ { - 1 }$ . Explicit orthogonalization gives rank $( I - T _ { k } ) = k$ in every retained run. Across 526 archived ranks, the maximum Gram-matrix error is $2 . 4 \times 1 0 ^ { - 1 5 }$ Thus “rank $k ^ { \dag }$ here means both k iterations and cumulative edit rank k for these scalar runs. In the multivariate full-QR construction, one iteration may add several independent directions, so we report iteration count and cumulative edit rank separately.

For descriptive stopping only, a rank is selected if its validation group-bootstrap AUROC interval lies wholly in [.45, .55] for three consecutive ranks. The same validation split also selected layer and intact $C ,$ and the $n = 2 0 0$ audit has essentially no power under this equivalence rule. We therefore do not use selected counts as numerical estimates in the main argument. To test postedit accessibility more strongly, a separate analysis divides validation groups into selection and audit halves, selects $C \in \{ 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 1 \}$ independently at every rank, swaps the halves, and repeats over five split seeds.

The real-feature maps are fixed before evaluation. The primary dense family is $A _ { \kappa } = Q D _ { \kappa } Q ^ { \intercal }$ , where $Q$ is a randomized signed-Hadamard basis and the 1,024 diagonal entries of $D _ { \kappa }$ are log-spaced from $\kappa ^ { - 1 / 2 } \ t o \ \kappa ^ { 1 / 2 }$ . Five fixed maps are used per condition. Every post-edit feature is inverse-mapped before attack, and all probes are fit in the original training-standardized coordinates with the intact validation-selected $C = . 0 1$ , which is neither reselected nor transported across maps.

A preprocessing ablation recomputes coordinatewise training mean and standard deviation after each dense map at κ ∈ {1, 2, 3, 5, 10, 100, 1000}. Erasure is performed in those restandardized mapped coordinates; edited features are then unstandardized and inverse-mapped before the common attacker. We retain the pre- and poststandardization condition numbers and all validation predictions. At rank ten, 2,000 paired group-bootstrap replicates resample the 193 validation source videos and compare the mean of the five fixed mapped trajectories with identity. A secondary family uses 20 positive diagonal maps with randomly permuted log-spaced scales; coordinatewise restandardization cancels that family algebraically. Twenty Haar orthogonal maps provide a solver sanity check. Unshrunk empirical covariance is evaluated on five diagonal maps per condition with no floor or shrinkage. These declared map families are existential stress tests, not a distribution over all invertible maps.

A stronger dense-map analysis removes the fixedattacker concern without using oficial validation or test. For each of five source-video group partitions of the 1,600 oficial-training examples, role A (793–801 examples) learns the erasure trajectory and both original and mapped coordinate standardization; role B (484–490) trains a fresh post-edit attacker; and role C (310–318) is evaluated once. B is internally group-split into fit and tuning roles, selects $C \in \{ 1 0 ^ { - 3 } , \hat { 1 0 } ^ { - 2 } , 1 0 ^ { - 1 } , 1 \}$ independently for every map and audited rank, then refits on all B. We compare identity with the same five κ = 10 dense maps at ranks $0 / 1 / 2 / 5 / 1 0$ . Every A/B/C source-group overlap is zero. The 25 map–split diferences describe sensitivity over the declared five maps and five partitions; their empirical range is not labeled a population confidence interval.

The disclosed research log contains 1,865 historical oficial-test configurations, and no untouched local holdout remains. Those results are descriptive and absent from the primary tables. The visual identification analyses are exploratory: they use oficial validation or threeway roles drawn from oficial training and record zero oficial-test queries, but validation reuse precludes confirmatory language. A protocol timeline documents every split role, and a prospective replication protocol is frozen for a future dataset.

## 4.3 Closed-Form and Cross-Fitted Baselines

We directly implement Euclidean Mean Projection (equivalent here to full-rank SAL) and the exact afine LEACE map under empirical and OAS covariance. Estimating the class-mean diference and training a logistic attacker on the same sample forces zero training gradient at zero coeficient after mean equalization. Its exact .500 test AUROC is therefore only an algebraic sanity check, not evidence of a population one-direction route.

Our cross-fit calibration separates data roles within oficial training. For each of five source-video-disjoint partitions, A estimates an intervention from a balanced subsample, B trains a fresh attacker, and C evaluates it. We vary the A sample size, report split dispersion, and query neither oficial validation nor test. The synthetic version draws 14,000 samples independently from $X \ \sim \ \mathcal { N } ( 0 , I _ { 1 0 2 4 } )$ with $Y ~ = ~ { \mathbb { H } } [ X _ { 1 } ~ \geq ~ 0 ]$ : 10,000 form the eraser pool, 2,000 train a fresh logistic attacker with $C = . 0 1$ , and 2,000 are evaluated once. Coordinates are standardized using the eraser and attacker pools. For each balanced A-sample, sample MP/SAL projects out its normalized empirical class-mean diference; the population oracle instead zeros $X _ { 1 }$ . Ten independent seeds are run. This distinguishes finite-sample residual access from a claim about population guarding rank.

Controls remove 20 Gaussian and 20 variance-matched random subspaces, top-PCA subspaces, and five sequential shufled-label subspaces in the identical Euclidean or OAS metric space. We retain AUPRC, representation displacement, and raw predictions. Stability comparisons use one common standardization and whitening map fit on the oficial training split. Split-specific whitening is never compared with Euclidean principal angles.

## 5 Results

Evidence hierarchy. Theorems 1 and 2 establish exact population non-identification. The finite circular experiment asks whether the separation survives a practical optimizer and held-out stopping rule. The visual study asks only whether coordinate sensitivity is observable in one frozen-feature case study; it is neither needed for the theorems nor a direct replication of the motivating video study.

## 5.1 Conventional Probes Localize Access

Before any erasure, validation-only layer sweeps establish where linear contact access is strongest under the original probe protocol. Table 4 reports the layers fixed for the later audit. V-JEPA2 selects layers 21 and 18; DINOv2 selects layers 12 and 11. This localization remains useful: it identifies which representations are audited. It does not make the selected depth a unique contact layer or a model-independent semantic coordinate.

<table><tr><td>Dataset</td><td>Encoder</td><td>Layer</td><td>Val. AUROC</td></tr><tr><td>100DOH</td><td>V-JEPA2</td><td>21</td><td>.885</td></tr><tr><td>TouchMoment</td><td>V-JEPA2</td><td>18</td><td>.944</td></tr><tr><td>100DOH</td><td>DINOv2</td><td>12</td><td>.858</td></tr><tr><td>TouchMoment</td><td>DINOv2</td><td>11</td><td>.807</td></tr></table>

Table 4: Validation-selected layers from the initial threeseed sweeps. These values select audit points; they are not estimates of semantic depth.

## 5.2 The Integer Count Itself Is Not Identified

Theorem 1 supplies the integer statement: identity mixing stops after one population edit, while every nonzero shear stops after two for ridge $\lambda \in \ \{ 0 , . 1 , 1 , 1 0 \}$ Labels, suficient dimension, and minimum guarding rank are unchanged. Theorem 2 applies the motivating full coeficient-subspace QR rule: for two outputs, cumulative edit rank is two at $a = 0$ and four after any nonzero shear, despite a common suficient dimension and minimum guarding rank of two. Figure 1 distinguishes these estimands and gives a complementary one-direction-ata-time result. With suficient dimension and minimum guarding rank two, Euclidean count is two under orthogonal coordinates and eight for all 50 maps at each $\kappa \in \{ 3 , 1 0 , 1 0 0 , 1 0 0 0 \}$ . The covariance metric returns two in every trial.

The finite circular bridge reaches the same conclusion under the published held-out stopping rule. At $n = 4 { , } 0 0 0$ , identity mixing gives $K = 1$ in all 20 runs, as does the weak shear $a = . 2 5$ . For each tested shear $a \in \{ . 5 , . 7 5 , 1 , 1 . 2 5 , 2 \}$ , all 20 runs accept at least two updates. At $a = 1$ , post-first-edit held-out $R ^ { 2 }$ is .195 and circular MAE is $5 9 . 5 ^ { \circ }$ on average; capped mean K is 5.75 (range 3–8), with two of 20 runs right-censored at eight. Under the literal sequential recurrence, rank $\left( I - T _ { K } \right)$ has reached four while rank $\left( T _ { K } \right)$ remains about two, so 2K is not an independent removed-subspace dimension. The decisive identity $/ a = 1$ separation also holds in all ten runs at minibatch sizes 64 and 256, with mean shear counts 5.4 and 4.9. At $n = 1 , 0 0 0 .$ , only $1 5 / 2 0 \ a \ = \ 1$ runs accept a second update, consistent with the continuous residual magnitude in Equation 11; finite stopping is sample- and optimizer-relative rather than a noiseless copy of the exact theorem.

The real-feature test uses V-JEPA2 layer 21 on the 100DOH oficial validation split and isolates the same dependence without changing images, labels, or the class of intact linear scores. After mappedcoordinate restandardization, mean rank-ten AUROC is .599 at $\kappa \ = \ 1$ and .666/.673/.745/.791 at $\kappa \_ =$ $2 / 3 / 5 / 1 0$ , then .853/.840 at 100/1,000. The corresponding mean post-standardization condition numbers are $2 . 0 9 / 3 . 1 9 / 5 . 4 1 / 1 0 . 8 9$ and 110.64/1,070.68. Source-videopaired 95% bootstrap intervals for the mean mappedminus-identity rank-ten diference are [.008,.134] already at $\kappa \ : = \ : 2$ and [.110,.275] at $\kappa = 1 0$ (Figure 1, bottom left). Without restandardization, the previously declared dense maps give .802/.841/.853 at $\kappa = 1 0 / 1 0 0 / 1 \small { , } 0 0 0 ;$ the practical efect is therefore not explained by coordinate scaling alone. The secondary diagonal family gives .599/.809/.848/.846 at $\kappa = 1 / 1 0 / 1 0 0 / 1$ ,000 and is algebraically canceled by coordinatewise restandardization. Unshrunk empirical-covariance intervention, with no shrinkage or floor, gives the identical complete trajectory for all maps: rank-one .587 and rank-ten .518, with maximum prediction discrepancy $2 . 1 1 \times 1 0 ^ { - 6 }$ . Thus unshrunk empirical covariance matches the corollary of Proposition 2. Twenty Haar orthogonal maps also reproduce the complete Euclidean AUROC trajectory, with maximum prediction discrepancy $4 . 2 7 \times 1 0 ^ { - 6 }$ . Anisotropy, not arbitrary coordinate renaming or solver noise, drives the stress-test diference.

## 5.3 Stronger Attackers Preserve Procedure Dependence

The oficial-training-only dense-map audit directly retunes a fresh attacker for each map and rank. Rank-zero AUROC is identical at .822. At rank ten, identity averages .665 across five source-disjoint partitions, whereas the five $\kappa = 1 0$ maps average .776 across 25 map–partition runs. The paired diference has mean +.111 and empirical range [.075,.155], and is positive for all 25 pairs. The corresponding diferences are already positive for all pairs at rank one (mean +.033) and rank two (+.060). Thus the transformed-map efect is not an artifact of reusing one post-edit C or one attacker fit. These are conditional within-training case-study results, not population intervals.

Figure 2 retunes regularization at every rank on one group-disjoint validation half and evaluates on the other, with halves swapped over five seeds. On 100DOH, Euclidean/OAS AUROC is .827/.586 after one edit and .598/.494 after ten. On TouchMoment it is .924/.811 after one and .778/.447 after ten. The curves remain strongly metric-dependent and nonmonotone even when each rank receives a fresh attacker search. They should not be compressed into a precise semantic count. We call the reported quantity orientation-fixed concordance: it is ordinary AUROC with score orientation fixed on the selection half and never flipped on the audit half. A value below .5 therefore records an orientation reversal on the audit half, not equivalence to chance. Flipping it after inspection would turn the same instability into an apparently above-chance score.

## 5.4 Secondary Estimation Calibrations

Covariance-subsampling and cross-fit learning curves support interpretation rather than the central theorem. They show that covariance geometry remains sample-dependent and that a sample-estimated rank-one edit can leave substantial fresh-attacker access even for a known population rank-one guard. Appendix B reports the full curves and fold dispersion; the results are not evidence for multiple population contact routes.

## 6 Limitations

Our afine theorem and latent construction concern linear representations and linear interventions. They do not identify nonlinear, tokenwise, spatial, temporal, or causal concept organization. The visual study centers on contact and one primary V-JEPA2 checkpoint. Secondary predicates share the same 100DOH images. Their results, nuisance controls, random baselines, and temporal audits remain in the artifact. DINOv2 is an image encoder, not a temporally matched video model. Theorem 2 matches the motivating output width, squared loss, and full-QR removal algebra at population level. The finite circular bridge also matches the published angular output, Adam settings, epochs, literal QR recurrence, and heldout thresholds, but it uses a controlled distribution rather than the motivating video activations and does not reproduce steering. Because the source does not report minibatch size, we declare 128 and audit 64/256 rather than claim an exact hidden implementation match. Its fixed held-out split is repeatedly queried across ranks, as in the published specification, so it is a procedure stress test rather than a new generalization estimate. The ambienteight leading-vector experiment remains a complementary sequential variant.

![](images/5b48a756e182f71cf9fe0972db8241911f8eefe457f8040c361d8176221d5c04.jpg)

![](images/1ff95771553d779cb0dd341f5393cc4d28bb33b19ae81b4f9eedc834cc154aa4.jpg)

![](images/2d4e77b5e9783c90e023584569db36ff39628e16fc24f1651ba630da4562d629.jpg)

![](images/4a4f12235751af4e32f843dc04f3a9c80388f4d724f58f8464850b366bb27155.jpg)  
Figure 1: Identification and controls. Conceptual, top $l e f t { : }$ four quantities that need not agree. Population, top right: a continuous two-output target has Euclidean count two or eight, while the covariance-metric count stays at two. Real features, bottom: dense-map sensitivity survives mapped-coordinate restandardization (left); vertical intervals are paired source-video bootstrap intervals for mean mapped-minus-identity AUROC. Bottom right: 20 orthogonal maps reproduce the identity trajectory, and individual trials overlap.

The contact labels are imperfect semantic targets. In 100DOH, released object-box presence is nearly labeldeterministic, and the nuisance-matched hard subset has only 58 pairs. TouchMoment contrasts onset with a nearby pre-onset frame, so hand closure, approach, endpoint appearance, and action phase remain informative. Temporal and nuisance controls delimit these efects but cannot make either task a pure test of physical touch.

Cross-fitting diagnoses sample-estimated generalization, not universal linear guardedness. Its known-rankone calibration shows that residual access can be large without multiple population routes. OAS and Ledoit–

Wolf reduce covariance estimation problems but do not create a canonical semantic metric. Descriptive stopping reuses validation for layer, regularization, and rank and lacks formal sequential error control. Historical oficial test exposure was extensive, no new untouched holdout is available, and none of those results is confirmatory. A prospective protocol is frozen for a future dataset. The anisotropic visual maps comprise five fixed dense signed-Hadamard SPD maps and a secondary diagonal family. They establish dependence for those declared families, not for a distribution over all invertible maps. Dense-map sensitivity survives mapped-coordinate restandardization and paired source-video resampling, but it remains an exploratory oficial-validation result. The fresh-attacker map audit remains conditional on five declared training partitions and five maps; its 25 paired values are not independent population draws. Transported-metric equivariance requires an equivariant probe, transported positive-definite regularization after rank loss, common support, and equivariant tie-breaking. Rank-deficient Moore–Penrose fitting is explicitly not covered. Exact covariance is only one corollary and the unshrunk empiricalcovariance trajectory is an algebraic consistency control, not evidence that covariance is the semantically correct metric. Finally, an invertible map preserves information and linear prediction classes, not Euclidean geometry. Readers may legitimately study that geometry after declaring it as part of the estimand.

Table 5: Controlled identification results. Count and cumulative edit rank are distinguished in the theorem statements and finite recurrence.
<table><tr><td>Construction</td><td>Setting A</td><td>Setting B</td><td>Difference</td><td>Identified conclusion</td></tr><tr><td>Population ridge shear count</td><td> $a = 0 : 1$ </td><td> $a \neq 0 \colon 2$ </td><td>+1</td><td>Same rank-one concept isotropic or transported ridge</td></tr><tr><td>Continuous target, Euclidean count</td><td> $\kappa = 1 : 2$ </td><td> $\kappa \geq 3 : 8$ </td><td>+6</td><td>Same sufficient dimension and guard</td></tr><tr><td>Continuous target, exact-cov. count Finite circular  $\mathrm { \bar { Q } R } , n = 4 0 0 0$ </td><td> $\kappa = 1 : 2$   $a ~ = ~ 0 { : } ~ P ( K ~ \geq$   $2 ) = 0$ </td><td> $\kappa = 1 0 0 0 \colon 2$   $a = 1 \colon P ( K \geq 2 ) = 1$ </td><td>0  $0 / 2 0 \ \mathrm { v s . } \ 2 0 / 2 0$ </td><td>rank 2 Affine-equivariant population control Published Adam/QR threshold re-</td></tr></table>

Table 6: Empirical case-study results; every row makes zero oficial-test queries. For dense restandardization, brackets give a source-video-paired 95% bootstrap interval conditional on fixed maps and selections. For fresh attackers they give the empirical range over 25 declared map–partition pairs, not a confidence interval. Cross-fit $n _ { A }$ denotes eraserestimation sample size within oficial training.
<table><tr><td>Audit</td><td>Setting A</td><td>Setting B</td><td>Difference</td><td>Identified conclusion</td></tr><tr><td>Dense + restandardization, rank 10</td><td> $\kappa = 1 \colon . 5 9 9$ </td><td> $\kappa = 1 0 \colon . 7 9 1$ </td><td>+.192 [.110,.275]</td><td>Coordinate sensitivity survives ordi- nary preprocessing</td></tr><tr><td>Dense fresh attacker, rank 10</td><td> $\kappa = 1 \colon . 6 6 5$ </td><td> $\kappa = 1 0 \colon . 7 7 6$ </td><td>+.111</td><td>Map-specific per-rank tuning;  $2 5 / 2 5$ </td></tr><tr><td>Unshrunk empirical-covariance rank</td><td> $\kappa = 1 \colon . 5 1 8$ </td><td> $\kappa = 1 0 0 0 \colon . 5 1 8$ </td><td>[.075,.155] .000</td><td>paired differences positive Transported-metric corollary</td></tr><tr><td>10 Orthogonal maps, all ranks</td><td>Identity</td><td>20 maps</td><td> $| \Delta p | \le 4 . 2 7 \times$ </td><td>Euclidean-invariant sanity check</td></tr><tr><td>Synthetic cross-fit MP/SAL</td><td> $n _ { A } = 1 0 0 \colon . 8 8 1$ </td><td> $n _ { A } = 8 0 0 0 \colon . 5 7 2$ </td><td>10−⁶ -.309</td><td>Sample error despite rank-one popu-</td></tr><tr><td>100DOH-max cross-fit LEACE-OAS</td><td> $n _ { A } = 1 0 0 \colon . 8 7 6$ </td><td> $n _ { A } = 6 0 0 0 \colon . 6 4 7$ </td><td>-.229</td><td>lation guard Cross-fit within official training</td></tr><tr><td>TouchMoment cross-fit LEACE-OAS</td><td> $n _ { A } = 1 0 0 \colon . 9 2 7$ </td><td> $n _ { A } = 4 0 0 0 \colon . 6 2 6$ </td><td>-.301</td><td>Cross-fit within official training</td></tr></table>

Reproducibility. The supplementary artifact contains annotation-safe manifests, extraction configurations, every analysis script, spectra, fold-level and aggregate results, and a one-command cached-analysis entry point. Licensed media, feature caches, and large prediction arrays are not redistributed.

## 7 Conclusion

Many erased directions need not mean many semantic dimensions. Population ridge count changes from one to two for the same rank-one concept. Under full two-output QR removal, cumulative edit rank changes from two to the ambient dimension four, while a complementary sequential construction reaches ambient dimension eight. Under the published finite circular Adam/QR protocol, identity stops after one update in all large-sample runs while shears pass multiple held-out updates and can continue after edit rank saturates. We prove the conditions under which any covariantly transported positive-definite metric makes the complete trajectory afine-equivariant, with exact covariance as a corollary. The same frozen visual features show a large dense-map change after coordinatewise restandardization and map-specific freshattacker tuning, while orthogonal maps reproduce the identity trajectory. Finite-sample calibration further prevents residual attacker access from being read as multiple population routes. None of these results estimates a contact dimension.

The defensible object is an explicitly named procedurerelative estimand. Reports of iterative erasure count should specify the metric, probe loss, regularizer, covariance estimator, sample, orientation rule, stopping rule, and test-use policy, and should separate generating dimension, suficient dimension, encoded cross-covariance rank, and guarding rank. Without those commitments, iterative erasure count is not an afine-invariant concept dimension.

## References

[1] Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes, 2016.

[2] Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning, 2025.

[3] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Pro-

![](images/a4356cfcc18a058853dfd4475783cbb77f865608e8afefecfbf64b36540589ac.jpg)  
Figure 2: Per-rank attacker audit. Each rank selects logistic regularization on one validation-group half and evaluates on the other, with halves swapped over five seeds. Thin lines are all ten folds, while thick lines are medians. These are not confidence bands.

ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15619–15629, 2023.

[4] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mido Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. Transactions on Machine Learning Research, 2024.

[5] Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 2022.

[6] Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raf, and Stella Biderman. LEACE: Perfect linear concept erasure in closed form. In Advances in Neural Information Processing Systems, volume 36, 2023.

[7] Jericho Cain. Gauge freedom and metric dependence in neural representation spaces, 2026.

[8] Yilun Chen, Ami Wiesel, Yonina C. Eldar, and Alfred O. Hero III. Shrinkage algorithms for MMSE covariance estimation. IEEE Transactions on Signal Processing, 58(10):5016–5029, 2010.

[9] Alicja Dobrzeniecka, Antske Fokkens, and Pia Sommerauer. Improving causal interventions in amnesic probing with mean projection or LEACE. In Findings of the Association for Computational Linguistics: ACL 2025, 2025.

[10] Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav Goldberg. Amnesic probing: Behavioral explanation with amnesic counterfactuals. Transactions of the Association for Computational Linguistics, 2021.

[11] Pantea Haghighatkhah, Antske Fokkens, Pia Sommerauer, Bettina Speckmann, and Kevin Verbeek. Better hit the nail on the head than beat around the bush: Removing protected attributes with a single projection. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 8395–8416. Association for Computational Linguistics, 2022.

[12] John Hewitt and Percy Liang. Designing and interpreting probes with control tasks. In Proceedings of the 2019

Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, 2019.

[13] Sonia Joseph, Quentin Garrido, Randall Balestriero, Matthew Kowal, Thomas Fel, Shahab Bakhtiari, Blake Richards, and Mike Rabbat. Interpreting physics in video world models, 2026.

[14] Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, and Rory Sayres. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV). In Proceedings of the 35th International Conference on Machine Learning, pages 2668–2677, 2018.

[15] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

[16] Olivier Ledoit and Michael Wolf. A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis, 88(2):365–411, 2004.

[17] Yun Liu, Haolin Yang, Xu Si, Ling Liu, Zipeng Li, Yuxiang Zhang, Yebin Liu, and Li Yi. TACO: Benchmarking generalizable bimanual tool-action-object understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21740– 21751, 2024.

[18] Yunze Liu, Yun Liu, Che Jiang, Kangbo Lyu, Weikang Wan, Hao Shen, Boqiang Liang, Zhoujie Fu, He Wang, and Li Yi. HOI4D: A 4d egocentric dataset for categorylevel human-object interaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21013–21022, 2022.

[19] Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Schölkopf, and Olivier Bachem. Challenging common assumptions in the unsupervised learning of disentangled representations. In Proceedings of the 36th International Conference on Machine Learning, pages 4114–4124, 2019.

[20] Huy Anh Nguyen, Feras Dayoub, and Minh Hoai. Detecting precise hand touch moments in egocentric video, 2026. Accepted to CVPR Findings 2026.

[21] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.

[22] Tiago Pimentel, Josef Valvoda, Rowan Hall Maudslay, Ran Zmigrod, Adina Williams, and Ryan Cotterell. Information-theoretic probing for linguistic structure. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4609–4624, 2020.

[23] Francesco Ragusa, Giovanni Maria Farinella, and Antonino Furnari. StillFast: An end-to-end approach for short-term object interaction anticipation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2023.

[24] Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, and Yoav Goldberg. Null it out: Guarding protected attributes by iterative nullspace projection. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7237–7256, 2020.

[25] Shauli Ravfogel, Michael Twiton, Yoav Goldberg, and Ryan D. Cotterell. Linear adversarial concept erasure. In Proceedings of the 39th International Conference on Machine Learning, pages 18400–18421, 2022.

[26] Geofrey Roeder, Luke Metz, and Durk Kingma. On linear identifiability of learned representations. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 9030–9039. PMLR, 2021.

[27] Julia Rozanova, Marco Valentino, Lucas Cordeiro, and André Freitas. Interventional probing in high dimensions: An NLI case study. In Findings of the Association for Computational Linguistics: EACL 2023, pages 2489– 2500. Association for Computational Linguistics, 2023.

[28] Dandan Shan, Jiaqi Geng, Michelle Shu, and David F. Fouhey. Understanding human hands in contact at internet scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.

[29] Shun Shao, Yftah Ziser, and Shay B. Cohen. Gold doesn’t always glitter: Spectral removal of linear and nonlinear guarded attribute information. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 1611–1622. Association for Computational Linguistics, 2023.

[30] Tsukasa Shiota, Motohiro Takagi, Kaori Kumagai, Hitoshi Seshimo, and Yushi Aono. Egocentric action recognition by capturing hand-object contact and object state. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024.

[31] Chen Wei, Haoqi Fan, Saining Xie, Chao-Yuan Wu, Alan Yuille, and Christoph Feichtenhofer. Masked feature prediction for self-supervised visual pre-training. In 2022

IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[32] Yue Xu, Yong-Lu Li, Zhemin Huang, Michael Xu Liu, Cewu Lu, Yu-Wing Tai, and Chi-Keung Tang. EgoPCA: A new framework for egocentric hand-object interaction understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

## A Supplementary Finite Circular-QR Results

Table A1 gives representative sample-size cells from the primary minibatch-128 campaign. K counts accepted QR updates before the first held-out failure. Runs still passing at the eight-update cap are right-censored; $K = 8$ can also be observed exactly. The $n = 5 0 0$ identity row shows why finite error must be calibrated: even the untransformed rank-two construction sometimes passes a second probe. As sample size grows, identity stabilizes at one while moderate shears pass multiple updates. This is evidence about the declared procedure, not a population dimension estimate.

<table><tr><td>n</td><td>Shear a</td><td>Capped mean K</td><td> $P ( K \geq 2 )$ </td><td>Censored/20</td><td>Mean post-first  $R ^ { 2 }$ </td></tr><tr><td>500</td><td>0</td><td>1.80</td><td>.45</td><td>0</td><td>.089</td></tr><tr><td>500</td><td>.5</td><td>1.95</td><td>.40</td><td>0</td><td>.114</td></tr><tr><td>500</td><td>1</td><td>1.85</td><td>.40</td><td>0</td><td>.116</td></tr><tr><td>500</td><td>2</td><td>1.10</td><td>.25</td><td>0</td><td>.065</td></tr><tr><td>1000</td><td>0</td><td>1.00</td><td>.00</td><td>0</td><td>.001</td></tr><tr><td>1000</td><td>.5</td><td>1.75</td><td>.35</td><td>0</td><td>.079</td></tr><tr><td>1000</td><td>1</td><td>2.95</td><td>.75</td><td>0</td><td>.122</td></tr><tr><td>1000</td><td>2</td><td>2.35</td><td>.45</td><td>1</td><td>.109</td></tr><tr><td>4000</td><td>0</td><td>1.00</td><td>.00</td><td>0</td><td>-.002</td></tr><tr><td>4000</td><td>.5</td><td>4.35</td><td>1.00</td><td>0</td><td>.134</td></tr><tr><td>4000</td><td>1</td><td>5.75</td><td>1.00</td><td>2</td><td>.195</td></tr><tr><td>4000</td><td>2</td><td>5.90</td><td>1.00</td><td>1</td><td>.132</td></tr></table>

Table A1: Finite circular-target stress test, 20 seeds per cell. Some small-n shears fail even initially, so post-first means use runs that accepted the intact probe. Full ranges, censoring rates, and all seven shears are retained in the artifact.

For the seven $\begin{array} { r l r } { n } & { { } = } & { 4 , 0 0 0 } \end{array}$ conditions $a \in$ $\{ 0 , . 2 5 , . 5 , . 7 5 , 1 , 1 . 2 5 , 2 \}$ , the numbers of right-censored runs are 0, 0, 0, 1, 2, 3, 1 out of 20. Table A2 therefore reports the empirical survival probabilities through the declared cap rather than relying on capped means alone.

<table><tr><td>k</td><td> $P ( K \geq k ) , a = 0 P ( K \geq k ) , a = 1$ </td><td></td></tr><tr><td>1</td><td>1.00</td><td>1.00</td></tr><tr><td>2</td><td>.00</td><td>1.00</td></tr><tr><td>3</td><td>.00</td><td>1.00</td></tr><tr><td>4</td><td>.00</td><td>.90</td></tr><tr><td>5</td><td>.00</td><td>.85</td></tr><tr><td>6</td><td>.00</td><td>.50</td></tr><tr><td>7</td><td>.00</td><td>.30</td></tr><tr><td>8</td><td>.00</td><td>.20</td></tr></table>

Table A2: Censoring-aware finite circular-QR summary for the decisive $n = 4 , 0 0 0$ conditions. Probabilities through $k =$ 8 are identified despite right-censoring at the cap; behavior beyond eight is not.

At $n \ : = \ : 4 { , } 0 0 0$ and $a \ = \ 1$ , minibatch-size sensitivity preserves the separation. For sizes $6 4 / 1 2 8 / 2 5 6$ , identity has mean $K = 1 . 0$ and $P ( K \geq 2 ) = 0 ;$ the shear has mean $K = 5 . 4 / 5 . 7 5 / 4 . 9$ and $P ( K \geq 2 ) = 1$ . Sizes 64 and 256 use ten seeds each; the primary size 128 uses twenty.

## B Supplementary Estimation Calibrations

The raw positive-spectrum condition numbers are $9 . 1 \times$ $1 0 ^ { 4 }$ for full 100DOH-max and $3 . 1 \times 1 0 ^ { 5 }$ for TouchMoment. Flooring caps the post-floor value at exactly $1 0 ^ { 4 }$

Earlier larger diagnostics were computed before flooring. Repeated subsamples show that covariance sample composition does not explain away the trend (Figure 3, left). OAS rank-one validation AUROC on 100DOH-max has median .838 and range [.830,.845] at $n = 2 0 0$ , versus .639 at all 17,206 samples. TouchMoment has median .844 and range [.828,.861] at $n = 5 0 0$ , versus .722 at all 11,564. Individual subsamples are shown rather than treating five repetitions as a confidence interval. More samples improve an estimator, but they do not create a canonical semantic metric or rank.

The synthetic cross-fit calibration changes the interpretation of fresh-attacker access. In 1,024 dimensions, sample-estimated MP/SAL leaves independent AUROC .881, .754, .683, .622, and .572 at eraser sample sizes 100, 1,000, 2,000, 4,000, and 8,000, despite the known population rank-one guard. The population oracle gives .499 at every size (Figure 3, center). Large residual access is therefore compatible with high-dimensional estimation error alone.

Cross-fitting within oficial training shows the same sample-size dependence (right). LEACE-OAS falls from .876 to .647 over $n _ { A } = 1 0 0 – 6 { , } 0 0 0$ on 100DOH-max and from .927 to .626 over $n _ { A } = 1 0 0 – 4 , 0 0 0$ on TouchMoment. Five source-video-disjoint partitions give standard deviations below .014 at the largest sizes. These results do not identify population guarding rank or multiple contact routes. They show how much independently trained access survives a sample-estimated edit under a declared $\mathrm { A / B / C }$ protocol.

![](images/124f9c0dd72a80e5dc508314a169ce1ea4f7468869fb9ca8c5cdc35fd587af65.jpg)

![](images/79deb55b462a7c8110e05de9ee910f978d5a4fbf1aff9313340e6dd30d0eba06.jpg)

![](images/5c7943f8b2b75e15dfbe9319ca92671837a2e4c7a5001de2c02d71e795655e60.jpg)  
Figure 3: Estimation calibration. Left: repeated OAS covariance subsamples. Center: sample-estimated rank one edits leave access for a known population rank-one concept, unlike its oracle. Right: cross-fitting within oficial training. Thin lines and points show every repetition, while thick lines are medians.

## C Optional Supplementary Native-Basis Channel Case Study

This optional study is not part of the paper’s afine nonidentification evidence. We revisit the channel result with AUROC/AUPRC rather than thresholded F1. Within official training only, A ranks channels, B tunes and fits an attacker, and C evaluates; all roles are source-video disjoint. We rank by orientation-free univariate AUROC, replace selected standardized coordinates by the A-sample mean, compare uniformly random sets, and score both the intact B attacker and a fresh post-edit B attacker. Primary selected-layer audits use 20 outer splits. Focused depth and full-100DOH calibrations use 10 and five splits, respectively. Layers 19–20 use a source-stratified 2,000-train/618-validation subset with no test rows; their A/B/C channel analysis uses training only.

<table><tr><td>Data</td><td>Encoder</td><td>Layer</td><td>Splits</td><td>Fixed ∆</td><td>Fresh ∆</td><td>Jacc.</td><td>407 top-50</td></tr><tr><td>100DOH-2k</td><td>V-JEPA2</td><td>21</td><td>20</td><td>.017</td><td>.003</td><td>.478</td><td>0/20</td></tr><tr><td>100DOH-full</td><td>V-JEPA2</td><td>21</td><td>5</td><td>.032</td><td>.000</td><td>.713</td><td>0/5</td></tr><tr><td>TouchMoment</td><td>V-JEPA2</td><td>18</td><td>20</td><td>.052</td><td>.001</td><td>.841</td><td>0/20</td></tr><tr><td>Touch-subset</td><td>V-JEPA2</td><td>19</td><td>10</td><td>.022</td><td>.002</td><td>.605</td><td>0/10</td></tr><tr><td>Touch-subset</td><td>V-JEPA2</td><td>20</td><td>10</td><td>.038</td><td>.005</td><td>.639</td><td>0/10</td></tr><tr><td>TouchMoment</td><td>V-JEPA2</td><td>21</td><td>20</td><td>.068</td><td>.000</td><td>.732</td><td>0/20</td></tr><tr><td>TouchMoment</td><td>V-JEPA2</td><td>22</td><td>20</td><td>.062</td><td>.001</td><td>.800</td><td>20/20</td></tr><tr><td>TouchMoment</td><td>V-JEPA2</td><td>23</td><td>10</td><td>.044</td><td>.001</td><td>.735</td><td>10/10</td></tr><tr><td>TouchMoment</td><td>V-JEPA2</td><td>24</td><td>10</td><td>.065</td><td>.001</td><td>.793</td><td>0/10</td></tr><tr><td>100DOH</td><td>DINOv2</td><td>12</td><td>20</td><td>.008</td><td>.001</td><td>.481</td><td></td></tr><tr><td>TouchMoment</td><td>DINOv2</td><td>11</td><td>20</td><td>.074</td><td>.007</td><td>.800</td><td></td></tr></table>

Table A3: Excess AUROC drop of univariate-AUROC-ranked top-50 sets relative to matched random sets. “Fixed” scores the intact attacker; “Fresh” retrains after editing. Jaccard is mean pairwise top-50 overlap across outer splits.

The table separates stable routes from necessary information. Full TouchMoment sets are highly repeatable and perturb fixed readers by .044–.068 AUROC beyond random, while fresh-attacker excess is at most .0014. The focused layer-19/20 subset gives fixed excess .022/.038 and fresh excess .0015/.0049. On all 21,290 balanced 100DOH images, more data raises top-50 Jaccard to .713 and the fixed efect to .032, while fresh excess remains .0004. DI-NOv2 shows the same qualitative fixed-versus-fresh separation, with a modest .0068 TouchMoment residual.

Channel 407 is top-50 in 20/20 layer-22 and 10/10 layer-23 splits, with median ranks 22.5 and 15.5, but not at layers 18–21 or 24. Recurrence is not necessity: masking 407 alone at layer 22 drops fixed AUROC by .00059 versus .00061 for a matched random channel; at layer 23 its excess fixed drop is .00049. The most frequently top-ranked channels at layers 21–24 are also individually weak; the largest tested excess is .00119, and all fresh-attacker excesses are efectively zero.

Finally, opposite-label donor patching provides a setlevel sensitivity control. On full TouchMoment at layers 18, 22, and 23, selected top-50 patches yield fixed AUROC .686/.680/.699, versus .890/.863/.856 for random-channel patches and .898/.852/.868 for samelabel selected-channel patches. Focused layer 20 gives .711/.882/.905, and full 100DOH gives .697/.852/.898. Donor selection is label-informed, so retrained patch attackers answer a new intervention-induced prediction problem; only fixed selected-versus- control contrasts support the route claim.

The historical hard-zero experiments motivated this audit but are not needed for its conclusion. The corrected result is narrower and stronger: channel 407 is a reproducible member of a late-layer ensemble, while no tested individual axis is necessary and ensemble information remains accessible to fresh linear attackers. This analysis is axis-aligned in one fixed encoder basis and is not an afine-invariant decomposition. Mean replacement acts on saved mean-pooled representations rather than downstream network computation, and donor patching is labelinformed. Those restrictions are why the result remains supplementary.