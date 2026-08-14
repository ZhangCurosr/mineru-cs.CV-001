# Online Learning of Correspondences between Images

Michael Felsberg, Member, IEEE, Fredrik Larsson, Johan Wiklund, Niclas Wadstromer, and J¨ orgen¨ Ahlberg

Abstract—We propose a novel method for iterative learning of point correspondences between image sequences. Points moving on surfaces in 3D space are projected into two images. Given a point in either view, the considered problem is to determine the corresponding location in the other view. The geometry and distortions of the projections are unknown as is the shape of the surface. Given several pairs of point-sets but no access to the 3D scene, correspondence mappings can be found by excessive globa optimization or by the fundamental matrix if a perspective projective model is assumed. However, an iterative solution on sequences of point-set pairs with general imaging geometry is preferable. We derive such a method that optimizes the mapping based on Neyman’s chi-square divergence between the densities representing the uncertainties of the estimated and the actual locations. The densities are represented as channel vectors computed with a basis function approach. The mapping between these vectors is updated with each new pair of images such that fast convergence and high accuracy are achieved. The resulting algorithm runs in real-time and is superior to state-of-the-art methods in terms of convergence and accuracy in a number of experiments.

Index Terms—Online learning, correspondence problem, channel representation, computer vision, surveillance

![](images/1c6d89ac0a9a3385197bc7f82588c2e6cc218b2a9c140263a09bcaa14b7beef3.jpg)  
Fig. 1. Example for CMap in a surveillance application.

## 1 INTRODUCTION

N multi-view computer vision, a dynamic 3D scene is I projected into several images. If the 3D points move (or lie) on 2D surfaces, such as discontinuous curved surfaces, the resulting views show the same scene under multi-modal piecewise continuous distortions. In order to fuse information from the separate views, the corresponding image points of the 3D points are required.

In the two camera case, a point correspondence is defined as the pair of points from the two images that correspond to the same 3D point. The mapping from a point in one image to its corresponding point is potentially multimodal (e.g. occlusion, transparency) and spatially uncertain, since points are extracted using error-prone feature detectors. Therefore, points are represented by a probability hypothesis density (PHD) [1] that models the spatial distribution of an a-priori unknown number of points. In contrast to the single point case modeled by a probability density, a PHD combines weighted densities of many hypotheses. Here, PHDs are represented using channels [2]. The correspondence mapping (CMap) maps the PHD of one image to the PHD of the other image and contains implicitly the geometry of the 2D surfaces and the geometry and distortions of both views.

The CMap is useful in many applications. In video surveillance, see Figure 1, cheap wide angle lenses are used to cover a large area, but result in significant modeling errors if using the fundamental matrix approach. People typically move on 2D surfaces, such that the model assumption applies. Calibration is required to associate objects in multiple views, and should preferably be done iteratively using exclusively the incoming image sequences and without access to the scene. In practice, calibration of multi-camera surveillance systems is timeconsuming and costly, as is re-calibration after disturbances such as changes of the scene or camera position (deliberate or not, for example the occasional storm can change the viewing direction several degrees). Since calibration and re-calibration add significantly to the installation and maintenance costs of such systems, there is a real need for online learning. In mobile platform applications, information from several subsystems including vision sensors need to be fused. Modalities might differ, e.g. IR and color images, and the configuration might change due to vibrations and other mechanical impacts. The CMap adapts to changes and allows to map between sensors of different modalities.

## 1.1 Approach and Problem Formulation.

The problem is to find the most likely corresponding point in the second view given a query point in the first view and a sequence of detected point sets in both views. The correspondences of the points in the sets are unknown as is the projection geometry and the scene geometry. However, it is assumed that the points are moving on (piece-wise) 2D surfaces. For each pair of point sets in the sequence, PHDs are computed using channels, a basis function approach, resulting in vectors w (input / first view) and v (output / second view). Each pair (w, v) is used to iteratively update the CMap C, based on a suitable distance measure D. The PHD for a query point in the first view is then fed into the CMap C to estimate the PHD of the corresponding point in the other view. From the latter PHD, the most likely corresponding point is computed. Key features of this scheme are convergence and accuracy of results but also that it is computationally tractable. This requires a choice of D that allows incremental updating of C without storing previous data, i.e., online learning.

## 1.2 Related work

Most approaches for solving point correspondence problems exploit knowledge about the geometry in a statistical approach such as RANSAC [3]. Different from the proposed method, those approaches assume a perspective projection model such that the correspondence problem becomes a homography estimation problem (plane surface case) or a fundamental matrix estimation problem (general 3D case with unknown extrinsic parameters). Similar work also exists for certain, nonperspective projection models [4], but all geometric approaches have in common that the projection model needs to be known (up to its parameters) and the camera distortion needs to be estimated first.

Appearance based methods find correspondences through matching the views of objects [5]. Besides the usual problems of view-based matching, e.g. scale, inplane rotation, illumination, partial occlusion etc., objects might also have different visual appearances in different images. In cases of entirely different sensors, e.g. an IR camera and a color camera, view matching is no alternative. Also perceptual aliasing or the existence of a multiplicity of identical objects may cause wrong correspondences.

If neither a particular projection model nor visual similarity can be assumed, learning point correspondences from unlabeled sets of feature points is an acute problem [6], p. 238. Since the learning data consists of sets instead of single points, this learning problem is called correspondence-free [7]. For unknown correspondences, all mappings are potentially multimodal (many-to-many) and a multimodal regression problem has to be solved. In [8] two online algorithms for solving this problem are proposed, one is similar to [9], based on updating covariance matrices, the other is based on stochastic gradient descent which avoids the inversion of the autocovariance matrix, but ignores the previous error terms. Both algorithms fulfill the requirement for online learning, i.e. not to store the original learning data [10].

![](images/aeef0143615b964bd1a2e9f8068f138dc67ffdc335639eacf9da3b088fc15959.jpg)  
Fig. 2. Overview of learning algorithms, redrawn from [11], with the proposed method (CMap) added.

A further method for online learning of multimodal regression has been reported in the literature, called ROGER (Realtime Overlapping Gaussian Expert Regression) [11], [12]. ROGER is based on a mixture of Gaussian processes that are learned in a sparse online method [13], where sparsity means that the learning samples are reduced to a subset, i.e., a property which is already covered in the previous definition of online learning (right hand-side of Figure 2). The learning is performed in two steps: assigning data points to the suitable process using a particle filter and learning the assigned process using a Bayesian approach for updating the mean and kernel functions. According to [11] ROGER is the only multimodal (multimap) online learning method, see upper right of Figure 2. If ROGER is restricted to a single Gaussian process, single samples or samples with correspondence are assumed, then ROGER becomes equivalent to SOGP (Sparse Online Gaussian Process) learning [14], placed in the lower right of Figure 2.

A popular technique for the single sample case (lower right of Figure 2) is Locally Weighted Projection Regression (LWPR), which is an incremental online method for non-linear function approximation in high dimensional spaces. The method is based on a weighted average of locally weighted linear models that each utilize an incremental version of Partial Least Squares. Each local model is adjusted through PLS to the local dimensionality of the manifold spanned by the input data. The position and size of the local models are update automatically as well as the numbers of models. LWPR has sucessfully been used for learning robot control [15], [16].

A comprehensive overview of the application field of cross-camera tracking is given in [17], where the field is divided into overlapping vs. non-overlapping views and calibrated cameras vs. learning approaches. The calibration-free approaches for non-overlapping cameras often rely on descriptors of tracked objects and probabilistic modeling [18], [17], where the latter requires a notion of temporal continuity. Although this continuity is not stringent for the case of overlapping cameras, known approaches make use of intra-camera tracking before establishing the correspondence [19], [20]. In the here proposed method, correspondence between uncalibrated, overlapping cameras is established on single frames and without using appearance descriptors.

## 1.3 Main contributions

The main contribution of the present paper in relation to the previously mentioned methods is a novel algorithm for iterative learning of a linear model C on a sequences of PHDs, represented as channel vectors w and $\mathbf { v , }$ with the following properties:

• A theoretically appropriate distance measure $D ,$ Neyman’s chi-square divergence, is proposed for learning.

• The algorithm is a proper online learning method that stores previously acquired information in the fixed size model C and in $\Omega ,$ the empirical density of v.

• The update of C is computed efficiently from pointwise operations on the previous data in terms of C and Ω and the weighted residual of the incoming data.

• The algorithm has been applied to several surveillance datasets, and it shows throughout very good accuracy, produces very few outliers for the position estimates, and runs in real-time.

• On standard benchmarks, CMap learning results in very low residuals at a very low computational cost.

The paper is structured as follows: This introduction is followed by a section on the required methods, a section on the experiments and results, and a concluding discussion.

## 2 METHODS

This section covers the required theoretical background of this paper: Density estimation with channels, i.e., basis functions, CMap estimation using these density estimates, the CMap online learning algorithm, and finally how to find correspondences using CMap.

In what follows, capital bold letters are used for matrices, bold letters for vectors, and italic letters for scalars. The elements of an $M \times N$ matrix A are given by $a _ { m n } , m = 1 \ldots M , n = 1 \ldots N .$ The elements of an $M -$ dimensional vector a are given by $a _ { m } , m = 1 \ldots M$ . The n-th column vector of A is given as $[ \mathbf { A } ] _ { n } , n = 1 \ldots N .$ That means, the m-th element of $[ \mathbf { A } ] _ { n } \mathrm { i } \mathbf { s } a _ { m n } . \mathrm { A }$ subindex to a vector or matrix denotes a time index, $i . e . , \ \mathbf { A } _ { T }$ is the matrix A at time T. The Hadamard (element-wise) product of matrices A and B is denoted as A ◦ B and the element-wise division is denoted as the fraction $\frac { \mathbf { A } } { \mathbf { B } }$ such that ${ \frac { \mathbf { A } } { \mathbf { B } } } \circ \mathbf { B } = \mathbf { A }$

## 2.1 Representations of PHDs with channels

Channel representations have originally been suggested without explicit reference to density estimation [2], [21]. In neighboring fields, these types of representations are known as population codes [22], [23]. This section introduces channel representations as described in [2], [24] for the purpose of density estimation with a signal processing approach [25].

The channel-based method for representing the PHDs is a special type of 2D soft-histogram, i.e., a histogram where samples are not exclusively pooled to the closest bin center, but to several bins with a weight depending on the distance to the respective bin center. The function to compute the weights, the basis function $b ( \mathbf { x } )$ , is usually non-negative (no negative contributions to densities) and smooth (to achieve stability). For computational feasibility, basis functions have compact support. In order to obtain a position independent contribution of samples to the PHD, the sum of overlapping basis functions must be constant.

The basis functions are located on a 2D grid with spacings $d _ { 1 }$ and $d _ { 2 }$ in $x _ { 1 }$ and $x _ { 2 }$ . Throughout this paper the basis functions have a support of size $3 d _ { 1 } \ \bar { \times } \ \bar { 3 } d _ { 2 }$ Figure 3 illustrates the cases of two and three feature detections represented in this way. Note that the basis functions adjacent to the image area also respond in some cases. The regular placement of basis functions has the major advantage that signal processing methods can be used to manipulate these representations.

In order to reconstruct the exact position of single samples or to extract modes from the channel representation, the basis-functions need to be of a particular type. Restricting soft-histograms to reconstructable frameworks shares similarities with wavelets restricting bandpass filters to perfect reconstruction. Channel representations are similar to Parzen window or kernel density estimators, with the difference that the latter are not regularly spaced but placed at the incoming samples.

For the remainder of this paper, $\cos ^ { 2 }$ basis functions are chosen, mainly because they have compact support (in contrast to Gaussian functions):

$$
b ( { \bf x } ) \triangleq \prod _ { j = 1 } ^ { 2 } \frac { 2 } { 3 d _ { j } } \left\{ \begin{array} { l l } { \cos ^ { 2 } ( \frac { \pi x _ { j } } { 3 d _ { j } } ) } & { | x _ { j } | < \frac { 3 d _ { j } } { 2 } } \\ { 0 } & { | x _ { j } | \ge \frac { 3 d _ { j } } { 2 } } \end{array} \right. ,\tag{1}
$$

where ${ \bf x } = ( x _ { 1 } , x _ { 2 } ) ^ { \top }$ denotes the image coordinates. The range of x together with $d _ { j }$ determine the number of basis functions $N _ { j } = ( \operatorname* { m a x } ( \stackrel { \circ } { x _ { j } } ) - \operatorname* { m i n } ( x _ { j } ) ) / d _ { j } + 2$ for each dimension. The spacing $d _ { j }$ is typically much larger than 1, resulting in $N _ { j }$ being much smaller than the range of $x _ { j } . \ N = \tilde { N _ { 1 } } \tilde { N _ { 2 } }$ is the total number of basis functions. The 2D grid index $( k _ { 1 } , k _ { 2 } ) \in \{ 0 \ldots N _ { 1 } - 1 \} \times \{ 0 \ldots N _ { 2 } - 1 \}$ is converted into a linear index $n ( k _ { 1 } , k _ { 2 } )$ in the usual way: $n ( k _ { 1 } , k _ { 2 } ) = k _ { 1 } + N _ { 1 } k _ { 2 }$ . Using (1), the components of the N-dimensional channel vector w are obtained from I

![](images/c5fdcad74a9bec29cc67fe97e6d98da084918bb9840fa7d87f294cef201cfb78.jpg)  
Fig. 3. The left image contains 2 detections represented using basis functions. The right one contains 3 detections. The CMap generates the right representation from the left one. The size of the circles indicates the response of the respective basis function.

image points $\mathbf { x } _ { i }$ as

$$
w _ { n ( k _ { 1 } , k _ { 2 } ) } = \sum _ { i = 1 } ^ { I } b ( { \bf x } _ { i } - ( k _ { 1 } d _ { 1 } , k _ { 2 } d _ { 2 } ) ^ { \top } - ( d _ { 1 } , d _ { 2 } ) ^ { \top } / 2 )\tag{2}
$$

Switching to smooth bins reduces the quantization effect compared to ordinary histograms by a factor of up to 20 and thus allows us to either reduce computational load by using fewer bins, or to increase the accuracy for the same number of bins, or a mixture of both.

In order to compute the most likely position represented by a channel vector, an algorithm for extracting the mode with maximum likelihood is required. For $\cos ^ { 2 } -$ channels an optimal algorithm in least-squares sense is obtained as [26]

$$
\hat { x } _ { 1 } = d _ { 1 } l _ { 1 } + \frac { 3 d _ { 1 } } { 2 \pi } \arg \big ( \sum _ { k _ { 1 } = l _ { 1 } } ^ { l _ { 1 } + 2 } \sum _ { k _ { 2 } = l _ { 2 } } ^ { l _ { 2 } + 2 } w _ { n ( k _ { 1 } , k _ { 2 } ) } e ^ { \frac { i 2 \pi ( k _ { 1 } - l _ { 1 } ) } { 3 } } \big )\tag{3}
$$

$$
\hat { x } _ { 2 } = d _ { 2 } l _ { 2 } + \frac { 3 d _ { 2 } } { 2 \pi } \arg \big ( \sum _ { k _ { 1 } = l _ { 1 } } ^ { l _ { 1 } + 2 } \sum _ { k _ { 2 } = l _ { 2 } } ^ { l _ { 2 } + 2 } w _ { n ( k _ { 1 } , k _ { 2 } ) } e ^ { \frac { i 2 \pi ( k _ { 2 } - l _ { 2 } ) } { 3 } } \big )\tag{4}
$$

where $\mathrm { a r g } ( )$ is the complex argument and the indices $l _ { j }$ indicate the maximum sum $3 \times 3$ block:

$$
( l _ { 1 } , l _ { 2 } ) = \underset { ( m _ { 1 } , m _ { 2 } ) } { \operatorname { a r g m a x } } \sum _ { k _ { 1 } = m _ { 1 } } ^ { m _ { 1 } + 2 } \sum _ { k _ { 2 } = m _ { 2 } } ^ { m _ { 2 } + 2 } w _ { n ( k _ { 1 } , k _ { 2 } ) } .\tag{5}
$$

In algorithmic terms, firstly the channel vector is reshaped to a 2D array, which is then filtered with a $3 \times 3$ box filter. The maximum response defines the decoding position $( l _ { 1 } , l _ { 2 } )$ . Secondly, within the $3 \times 3$ window at this position, the 2D position of the maximum is computed from the channel coefficients using (3–4).

## 2.2 CMap estimation using PHDs

Given two images, the detected feature points $\mathbf { x } _ { i }$ and $\mathbf { y } _ { j }$ are represented as channel vectors w and v respectively. For a sequence of $T$ image pairs, these vectors form matrices $\mathbf { \hat { W } } = [ \mathbf { w } _ { 1 } \dots \mathbf { w } _ { T } ]$ and $\mathbf { V } = \left[ \mathbf { v } _ { 1 } \ldots \mathbf { v } _ { T } \right]$ and the goal is now to estimate the CMap from these matrices. The CMap is multimodal and non-continuous in general. For multimodal or non-continuous mappings, ordinary function approximation techniques, such as (local) linear regression [27], tend to average across different modes or discontinuities. Hence, Johansson et al. propose to use mappings from w to v instead [28]. This is also the main difference to kernel regression and the Parzen-Rosenblatt estimators, where the kernel functions on the output side are integrated to compute the estimate of $\mathbf { y }$ (see [29], p. 295). An estimate of the vector v is obtained using a linear model C

$$
\hat { \mathbf { v } } = \mathbf { C } \mathbf { w }\tag{6}
$$

and the final estimate $\hat { \mathbf { y } }$ is obtained from vˆ using (3– 4). Typical sizes of v and w are about 500 elements, such that C is a matrix with about 250,000 elements. Efficiency is achieved by exploiting sparseness: C contains typically about 5,000 non-zero elements. If the single hypothesis case is considered, all entities have corresponding probabilistic interpretations, in particular C corresponds to a conditional density. In the general case, however, the channel vectors correspond to combined densities of several hypotheses and the direct relation to a probabilistic interpretation is lost.

Our approach to learn C shares some similarities with the method of Johansson et $a l . ,$ but is based on a different objective function. In [28] C is computed by minimizing the Frobenius norm of the estimation error matrix

$$
\mathbf { C } ^ { * } = \underset { \mathbf { C } } { \mathrm { a r g m i n } } \ : \lVert \mathbf { V } - \mathbf { C W } \rVert _ { F } ^ { 2 } \quad \mathrm { s . t . } ~ c _ { m n } \geq 0 ~ ,\tag{7}
$$

where the non-negativity constraint leads to a reduction of oscillations and to sparse results. Note that linear regression in the channel domain allows for general (nonlinear), multimodal regression in the original domain.

Since C acts on PHDs, columns of V and W can be added in (7) if the corresponding detections do not interfere, i.e., their respective distance is larger than $3 d _ { j }$ in at least one dimension $j ,$ and if the detected points move independently in the learning set. Learning from the combined channel vectors has been shown to be equivalent to stochastic gradient descent on the separated vectors [7]. Further details are given in Section 2.4.

Using the Frobenius norm in (7) corresponds to a leastsquares approach, which is not very suitable if channel vectors are interpreted as density representations. Also in practice, the least-squares approach causes problems with oscillating values if the non-negativity constraint is not imposed. This changes if v and $\hat { \textbf { v } } = \mathbf { C } \mathbf { w }$ are compared using α-divergences $D _ { \alpha }$ (see [30], Chapter 4), as often used in non-negative matrix factorization (NMF, see [31]):

$$
D _ { \alpha } [ \mathbf { v } | | \mathbf { C w } ] = \sum _ { i = 1 } ^ { I } \frac { \alpha v _ { i } + ( 1 - \alpha ) [ \mathbf { C w } ] _ { i } - v _ { i } ^ { \alpha } [ \mathbf { C w } ] _ { i } ^ { 1 - \alpha } } { \alpha ( 1 - \alpha ) }\tag{8}
$$

From a theoretical point of view, divergences are a sound way to compare densities. Certain choices of α result in common divergence measures [32], [33]. $\alpha $ 1 and $\alpha  0$ result in the Kullback-Leibler divergence and the log-likelihood ratio, respectively. $\alpha = 0 . 5$ results in the Hellinger distance, $\alpha = 2$ in the Pearson’s chisquare distance, and $\alpha = - 1$ in the Neyman’s chi-square distance:

$$
D _ { - 1 } [ \mathbf { v } | | \mathbf { C w } ] = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { I } \frac { ( v _ { i } - [ \mathbf { C w } ] _ { i } ) ^ { 2 } } { v _ { i } } .\tag{9}
$$

The Neyman’s chi-square is the unique α divergence that establishes a weighted least-squares problem. This is easily seen by requiring the numerator of (8) to be a quadratic polynomial, $i . e . , 1 - \alpha = 2$ . Furthermore, if an independent Poisson process is assumed, which is usually done for histograms, the Neyman’s chi-square is equal to the Mahalanobis distance, since $v _ { i }$ is an estimate of the variance of bin i. For the case of channel vectors, the bins are weakly correlated due to the overlapping basis functions and the equality only holds approximatively [22].

Hence, the Neyman’s chi-square divergence is more suitable for comparing channel vectors than using the Frobenius norm in (7). Note that it is not symmetric, the denominator only contains coefficients of v, i.e. empirical values. This is a conscious choice in order to stabilize iterative learning. For $T$ independently drawn samples of w and v, the divergence becomes

$$
D _ { - 1 } [ \mathbf { V } | | \mathbf { C W } ] = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { I } \frac { \sum _ { t = 1 } ^ { T } ( v _ { i t } - [ \mathbf { C W } ] _ { i t } ) ^ { 2 } } { \sum _ { t = 1 } ^ { T } v _ { i t } }\tag{10}
$$

and the following minimization problem is obtained

$$
\mathbf { C } ^ { * } = \underset { \mathbf { C } } { \mathrm { a r g m i n } } D _ { - 1 } [ \mathbf { V } | | \mathbf { C W } ] \mathrm { ~ . ~ }\tag{11}
$$

## 2.3 The new online CMap learning algorithm

The optimal CMap C according to (11) is computed iteratively from a sequence of vectors w and v using online the CMap learning algorithm.

Algorithm 1 CMap online learning algorithm.   
Require: $\overline { { T } }$ instances of $\mathbf { w } _ { t }$ are given   
Ensure: Optimal $\mathbf { C } _ { t }$ according to (11) and $\hat { \mathbf { v } } _ { t }$ from (6)   
1: Set forgetting factor $\gamma ,$ set $\mathbf { C } _ { 0 } = 0 ,$ , set $\Omega _ { 0 }$ as uniform   
2: for $t = 1$ to $\check { T }$ do   
3: if $\mathbf { v } _ { t }$ is known then   
4: $\mathbf { G } = \mathbf { C } _ { t - 1 } \circ \Omega _ { t - 1 } + ( \mathbf { v } _ { t } - \mathbf { C } _ { t - 1 } \mathbf { w } _ { t } ) \mathbf { w } _ { t } ^ { \mathsf { T } }$   
5: $\begin{array} { r } { \Omega _ { t } = \frac { ( 1 - \gamma ^ { t - 1 } ) \Omega _ { t - 1 } + { \bf v } _ { t } { \bf 1 } ^ { \top } } { 1 - \gamma ^ { t } } } \end{array}$   
6: $\begin{array} { r } { \mathbf { C } _ { t } = \frac { \mathbf { G } } { \Omega _ { t } } } \end{array}$   
7: else   
8: $\hat { \mathbf { v } } _ { t } = \mathbf { C } _ { t - 1 } \mathbf { w } _ { t }$   
9: $\begin{array} { r } { \Omega _ { t } = \frac { ( \bar { 1 } - \bar { \gamma } ^ { t - \bar { 1 } } ) \Omega _ { t - 1 } + \hat { \mathbf { v } } _ { t } \mathbf { 1 } ^ { \top } } { 1 - \gamma ^ { t } } } \end{array}$   
10: $\mathbf { C } _ { t } = \mathbf { C } _ { t - 1 }$   
11: end if   
12: end for

This algorithm is derived from the subsequent calculations. The gradient of (10) with respect to C gives

$$
\frac { \partial } { \partial c _ { m n } } D _ { - 1 } [ \mathbf { V } | | \mathbf { C W } ] = - \frac { \sum _ { t = 1 } ^ { T } ( v _ { m t } - [ \mathbf { C W } ] _ { m t } ) w _ { n t } } { \sum _ { t = 1 } ^ { T } v _ { m t } } \mathrm { ~ , ~ }\tag{12}
$$

and results in the gradient descent iteration

$$
\mathbf { C } \gets \mathbf { C } + \beta \frac { \mathbf { V } \mathbf { W } ^ { \top } - \mathbf { C } \mathbf { W } \mathbf { W } ^ { \top } } { \mathbf { V } \mathbf { 1 1 } ^ { \top } } \triangleq \mathbf { C } + \beta \ \delta \mathbf { C } \ ,\tag{13}
$$

where 1 is a one-vector of suitable size and the fraction is computed element-wise (inverse Hadamard product). Adding time instance $T + 1 , \ \left( \mathbf { w } _ { T + 1 } , \mathbf { v } _ { T + 1 } \right)$ , the update term in (13) becomes

$$
\begin{array} { r } { \delta \mathbf { C } _ { T + 1 } = \phantom { \mathbf { 0 } } } \\ { \frac { \left[ \mathbf { V } \mathbf { \mathbf { v } } _ { T + 1 } \right] \left[ \mathbf { W } \mathbf { \mathbf { w } } _ { T + 1 } \right] ^ { \mathsf { T } } - \mathbf { C } _ { T } \left[ \mathbf { W } \mathbf { \mathbf { w } } _ { T + 1 } \right] \left[ \mathbf { W } \mathbf { \mathbf { w } } _ { T + 1 } \right] ^ { \mathsf { T } } } { \left[ \mathbf { V } \mathbf { \mathbf { v } } _ { T + 1 } \right] \mathbf { 1 1 } ^ { \mathsf { T } } } = \phantom { \mathbf { 0 } } } \\ { \delta \mathbf { C } _ { T } \circ \frac { \mathbf { V } \mathbf { 1 1 } ^ { \mathsf { T } } } { \left[ \mathbf { V } \mathbf { \mathbf { v } } _ { T + 1 } \right] \mathbf { 1 1 } ^ { \mathsf { T } } } + \frac { \mathbf { v } _ { T + 1 } \mathbf { w } _ { T + 1 } ^ { \mathsf { T } } - \mathbf { C } _ { T } \mathbf { w } _ { T + 1 } \mathbf { w } _ { T + 1 } ^ { \mathsf { T } } } { \left[ \mathbf { V } \mathbf { \mathbf { v } } _ { T + 1 } \right] \mathbf { 1 1 } ^ { \mathsf { T } } } \enspace . } \end{array}\tag{14}
$$

For reasons of convergence, the old mapping is downweighted by a forgetting factor $0 ~ \ll ~ \gamma ~ < ~ 1$ in each iteration:

$$
\mathbf { C } \gets \gamma \mathbf { C } + \beta \delta \mathbf { C } \ ,\tag{15}
$$

where normalization of C is not necessary if applying the decoding formula (3–4).

If the update term δC is fixed for all steps and C has been initialized as a zero matrix, $\mathbf { C } _ { T }$ is obtained after $T$ iterations using a geometric series $( \gamma < 1 )$ as

$$
\mathbf { C } _ { T } = \beta \sum _ { t = 0 } ^ { T } \gamma ^ { t } \delta \mathbf { C } = \beta \frac { 1 - \gamma ^ { T } } { 1 - \gamma } \delta \mathbf { C } \mathrm { ~ . ~ }\tag{16}
$$

Substituting $\delta { \bf C } _ { T + 1 }$ and $\delta \mathbf { C } _ { T }$ in (14) using (16) and $\beta =$

(1 − γ) (again, the absolute scale of C is irrelevant) gives

$$
\begin{array} { r l } & { \mathbf { C } _ { T + 1 } = \mathbf { C } _ { T } \circ \frac { \left( 1 - \gamma ^ { T + 1 } \right) \mathbf { V } \mathbf { 1 } \mathbf { 1 } ^ { \mathsf { T } } } { \left( 1 - \gamma ^ { T } \right) \left[ \mathbf { V } \mathbf { \ v } _ { T + 1 } \right] \mathbf { 1 } \mathbf { 1 } ^ { \mathsf { T } } } } \\ & { \quad + \frac { \left( 1 - \gamma ^ { T + 1 } \right) \left( \mathbf { v } _ { T + 1 } \mathbf { w } _ { T + 1 } ^ { \mathsf { T } } - \mathbf { C } _ { T } \mathbf { w } _ { T + 1 } \mathbf { w } _ { T + 1 } ^ { \mathsf { T } } \right) } { \left[ \mathbf { V } \mathbf { \ v } _ { T + 1 } \right] \mathbf { 1 } \mathbf { 1 } ^ { \mathsf { T } } } \ . } \end{array}\tag{17}
$$

The matrix

$$
\Omega _ { T } = { \frac { \mathbf { V 1 1 } ^ { \mathsf { T } } } { 1 - { \gamma } ^ { T } } } = { \frac { [ \mathbf { v } _ { 0 } \ldots \mathbf { v } _ { T } ] \mathbf { 1 1 } ^ { \mathsf { T } } } { 1 - { \gamma } ^ { T } } } = { \frac { \sum _ { t = 0 } ^ { T } \mathbf { v } _ { t } } { 1 - { \gamma } ^ { T } } } \mathbf { 1 } ^ { \mathsf { T } }\tag{18}
$$

denotes the cumulative v vector for $T$ time-steps, weighted by the forgetting factor $\gamma$ (identical in all N columns). Substituting $\Omega _ { T }$ and collecting terms, the final update equation for CMap learning is obtained

$$
\mathbf { C } _ { T + 1 } = \frac { \mathbf { C } _ { T } \circ \boldsymbol { \Omega } _ { T } + \left( \mathbf { v } _ { T + 1 } - \mathbf { C } _ { T } \mathbf { w } _ { T + 1 } \right) \mathbf { w } _ { T + 1 } ^ { \mathsf { T } } } { \boldsymbol { \Omega } _ { T + 1 } } \mathrm { ~ . ~ }\tag{19}
$$

As it can be seen in the CMap algorithm, C is only updated if $\mathbf { v } _ { t }$ is known, i.e. learning data is available. Otherwise, only Ω is updated and an estimate of $\mathbf { v }$ is computed. The latter is always used during evaluation.

## 2.4 Properties of CMap learning

The proposed CMap update equation differs from the two previously suggested schemes [8] for solving (7). The first scheme involves computing the auto-covariance matrix of $\mathbf { w }$ and the cross-covariance matrix of w and v. It is suggested to incrementally update these covariance matrices by introducing a forgetting factor γ and adding the outer product of new channel vectors. The update of C in time-step $T + 1$ is determined by

$$
\gamma ( \mathbf { C } \mathbf { W } _ { T } - \mathbf { V } _ { T } ) \mathbf { W } _ { T } ^ { \top } + ( \mathbf { C } \mathbf { w } _ { T + 1 } - \mathbf { v } _ { T + 1 } ) \mathbf { w } _ { T + 1 } ^ { \top }\tag{20}
$$

up to a point-wise weighting factor.

In the second scheme, the first term in (20) is assumed to tend to zero for large $T ,$ , thus (20) results in a stochastic gradient descent based on the second term only. Thus, both schemes differ from (19) in the way how old and new information is fused: Scheme 1 keeps updating using the old covariance matrices whereas scheme 2 ignores old data altogether. The proposed CMap scheme re-weights the previous matrix C point-wise with $\Omega _ { T }$ and weights the combined update with $\Omega _ { T } ^ { - 1 }$ , which results in an exact minimizer of Neyman’s chi-square distance.

In contrast to the approach in [28], the CMap C may contain negative values as long as the product Cw is non-negative for all valid channel vectors w. Valid channel vectors are formed by applying the basis functions (1) to a (continuous) distribution. The basis functions have a low-pass character, which implies that the subsequent operator C may be of high-pass character, as long as the joint operator is non-negative. High-pass operators typically have small negative values in the vicinity of large positive values. In practice, the update (19) sometimes results in negative values and to avoid problems with stability, large negative values are clipped to small negative values.

![](images/81e5bb0d54051cad984d90d0af690b7087f06d7c0bf7431fbe3f64ac7eb6b29f.jpg)  
Fig. 4. Illustration of multimodal CMap (left) learned from step data (Figure $6 ,$ middle) and example PHDs (right); from top to bottom: Input PHD, resulting output from CMap, ground truth PHD. All densities are marginalized by integrating out the y-coordinate.

In the terminology of online learning [10], the update equation (19) is a proper online learning method, as no learning data is accumulated through time. All involved entities are of strictly constant size, and therefore, there are basically no limitations in the duration of the learning. In practice, (19) has been applied for hours in real-time applications within the DIPLECS<sup>1</sup> project. In this project, the same learning method has also been used on other types of problems than camera-to-camera mapping, in particular on learning tracking in nonperspective cameras [34].

Another fundamental property of CMap learning is its capability to find the respectively simplest mapping. If the problem to be learned is unimodal, the learned CMap will result in unimodal output, see Figure 4. The reason for this behavior is the equivalence to learning with correspondences [7], also observed for the theoretical case of general PHDs [1]: ”... time evolution of the PHD is governed by the same law of motion as that which governs the between-measurements time-evolution of the posterior density of any single target $\cdots ^ { \prime \prime } .$

## 2.5 Finding correspondences using CMap

Once the CMap C has been learned, it can be used in two different ways: In position mode and in correspondence mode.

In position mode, no output feature points $\mathbf { y } _ { j }$ are known. Each input feature point x<sub>i</sub> is separately encoded as a channel vector $\mathbf { w } _ { i }$ and mapped to the output channel vector using (6)

$$
\hat { \mathbf { v } } _ { i } = \mathbf { C } \mathbf { w } _ { i } \ \mathrm { ~ . ~ }\tag{21}
$$

The output channel vector is then decoded using (3– 4) to obtain output feature coordinates $\hat { \mathbf { y } } _ { i }$ . The confidence of this decoded point is given in terms of the sum of coefficients in (5). If this confidence is below a certain threshold, the input point $\mathbf { x } _ { i }$ is considered to being mapped outside the considered range or to having no corresponding point. If the confidence of the first decoding (mode) is above the threshold, further modes of $\hat { \mathbf { v } } _ { i }$ could be decoded, but this has not been considered in the subsequent experiments.

In correspondence mode, also the output feature points $\mathbf { y } _ { j }$ are known, but not their correspondence. Again, each input feature point $\mathbf { x } _ { i }$ is separately encoded as a channel vector $\mathbf { w } _ { i }$ and mapped to the output channel vector $\hat { \mathbf { v } } _ { i }$ . Also each output feature point $\mathbf { y } _ { j }$ is separately encoded as a channel vector $\mathbf { v } _ { j } .$ . For each input feature i, the corresponding output feature j is now found as

$$
j ( i ) = \underset { j } { \operatorname { a r g m a x } } \mathbf { v } _ { j } ^ { \mathsf { T } } \mathbf { C } \mathbf { w } _ { i } \mathrm { ~ . ~ }\tag{22}
$$

If the scalar product $\mathbf { v } _ { j ( i ) } ^ { \mathsf { T } } \mathbf { C } \mathbf { w } _ { i }$ is below a certain threshold, no correspondence to point i is found. Also, some $j$ might not be assigned to any $i ,$ thus no correspondence to point j is found. Basically, the assignment can also be computed the other way around as $\bar { i } ( j )$ , or both ways. Combining the two-way assignment by a logical and results in a one-to-one assignment, combining them by a logical or results in a many-to-many assignment. For the experiments below, only the first case j(i) has been considered.

## 3 EXPERIMENTS

CMap learning is evaluated on five different datasets D1–D5:

D1: The cross 2D data as used in [35]

D2: Synthetic projections of 3D points using two real camera models

D3: Surveillance data from PETS2001

D4: Surveillance data from the PROMETHEUS dataset

D5: Surveillance data acquired at a spiral staircase All these datasets have been selected according to the original problem formulation in Section 1.1: The points are located on 2D surfaces.

The CMap algorithm is compared to LWPR [35] and ROGER [12] on the dataset D1. ROGER and four different update schemes for CMap are compared quantitatively using the dataset D2. A quantitative comparison to ROGER and a convergence analysis are shown on the dataset D3. Finally, qualitative evaluation of the CMap learning is made on D4 and D5, where in particular the latter shows that CMap learning is not restricted to the planar case, but also works on real data for curved surfaces. The parameters for the comparisons between ROGER and CMap are summarized in Table 1.

## 3.1 Cross 2D data (D1)

The cross 2D data is generated from a process with constant variance $\sigma ^ { 2 } ~ = ~ 0 . 0 1$ and mean in the range [0; 1.25], given by the combination of three 2D Gaussian functions [35]. One example for a set with 500 samples is shown in Figure 5, top. Sets with varying number of samples are used to train the respective method. The evaluation is done using the ground truth density on a grid of 41×41 points, using the normalized mean square error (nMSE), i.e., MSE divided by the variance of the noisy input data.

<table><tr><td>method</td><td>parameter</td><td>D1</td><td>D2</td><td>D3</td></tr><tr><td rowspan="6">ROGER</td><td>µ</td><td>1</td><td>0.5</td><td>0.5</td></tr><tr><td>λ</td><td>2</td><td>0.1</td><td>2</td></tr><tr><td>α</td><td>0.5</td><td>0.1</td><td>0.5</td></tr><tr><td>k</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td>v</td><td>2</td><td>3</td><td>3</td></tr><tr><td>P</td><td>1</td><td>20</td><td>30</td></tr><tr><td rowspan="4">CMAP</td><td>γ</td><td> $\overline { { 1 - 1 0 ^ { - 4 } } }$ </td><td> $\overline { { 1 - 1 0 ^ { - 4 } } }$ </td><td> $\overline { { 1 - 1 0 ^ { - 3 } } }$ </td></tr><tr><td> $N _ { j }$  input</td><td>33 × 33</td><td>cf. Table 2</td><td> $5 0 \times 3 8$ </td></tr><tr><td> $\boldsymbol { N } _ { j } ^ { \cdot }$  output</td><td>8</td><td>cf. Table 2</td><td> $5 0 \times 3 8$ </td></tr></table>

TABLE 1  
Overview choice of parameters for ROGER and CMap.

The code for the LWPR method and for generating the dataset are taken from Vijayakumar’s homepage<sup>2</sup>. All parameters have been kept unchanged: Gaussian kernel with $\mathbf { D } = \left( \begin{array} { c c } { 1 0 0 } & { 0 . 0 1 } \\ { 0 . 0 1 } & { 2 5 } \end{array} \right) ^ { \mathbf { \delta } }$ (the performance boost described in [13] by increasing D could not be observed), $\alpha = 2 5 0 , w = 0 . 2$ , meta learning set to one and the meta rate to 250.

The ROGER implementation has been taken from Grollman’s homepage<sup>3</sup>. All parameters have been left unaltered, except for $\lambda = 2 ,$ which gives significantly better results than $\lambda = 0 . 1 ,$ , cf. Table 1.

The parameters for CMap learning are also listed in Table 1. The range of the input space is $[ - 1 ; 1 ] \times [ - 1 ; 1 ]$ The range of the output space is fixed to [−0.2; 1.35], which includes most of the samples from the cross 2D data.

All three methods (CMap, ROGER, LWPR), with parameters as detailed above, are evaluated on the cross 2D data with 500, 1,000, 2,000, 4,000, 6,000, 8,000, and 10,000 samples.

## 3.2 Results on cross 2D data

The result of CMap learning for 500 samples (less than 20 activations per input channel) is shown in Figure 5, middle. The normalized MSE for all three methods is plotted in Figure 5, bottom. In general, CMap learning converges fastest. The accuracy is always better than LWPR with at least a factor of 3. However, the results for LWPR from [35] could not fully be reproduced. The reported numbers are similar to the here reported LWPR results for low numbers of samples but better by a factor of two for 10,000 samples (but still less accurate than CMap learning).

![](images/8e6ad0f6e605d917de8e2d64147c7487e7b81cb54418231b6aeee268c32e438e.jpg)

![](images/37ea706d05a0d6ae363bcb9162a2faae488dfe2b8c6ea6d305321a6194914ffb.jpg)

![](images/fed942204d7dae61e4be9fe3101d3d55ddafc36fb32c84d439d7686f0da56253.jpg)  
Fig. 5. Top: 500 samples from cross 2D data. Middle: Evaluation for the mapping learned using CMap learning. Bottom: Comparison with LWPR and ROGER. The barplot shows the nMSE over the number of learning samples.

Both methods run in real-time (30 fps) at about the same speed. The computation time per frame does not increase with time. This is different for ROGER: Albeit having real-time performance for the first few samples, it slows down with increasing number of incoming samples. As a consequence, the learning took about 3 hours for 4,000 samples, compared to less than a minute for the other methods.

In terms of accuracy, ROGER starts at about the same level as LWPR, but improves accuracy faster and reaches the same level of accuracy as CMap learning at 4,000 learning samples. Eventually, ROGER is about twice as accurate as CMap learning, but for the price of longer run-time by about 3 orders of magnitude. Note also that [13] reports non-normalized MSEs which cannot be compared directly to the nMSEs reported here and in the original work [35].

## 3.3 Synthetic projections dataset (D2)

This dataset consists of realistic projections of synthetic data, similar to the experiment in [6]. From a real twocamera setup, camera calibration and radial distortions are estimated and used to build a realistic projection model. This model then generates image points from a 18 × 18 array of 3D points. The 3D points lie on three different surfaces: A plane, a hyperbolic surface, and a two-plane discontinuity, see Figure 6.

![](images/770d0a96a4a9d4fcaaa85bbf904bc9df87ad67cc074d4975e2e40b9c7420d8cb.jpg)  
Fig. 6. Projections of synthetic 3D points. Left: Camera 1, right: Camera 2. Top: Plane surface, middle: Curved surface, bottom: Discontinuous surface.

Noise with four different standard deviations (0, 1, 2, and 3 pixel in $x _ { 1 }$ and $x _ { 2 }$ direction) is added to the projected points. The training samples are arranged in two ways: a) Each sample contains one point-pair, i.e. the point correspondence is known and only correct associations (no outliers) are in the data (datasets ’flat’, ’curved’, and ’step’). b) Each sample contains two points for each of both images and the correspondence is unknown. Thus four pairs are given, two of which are correct, such that 50% of the data consist of outliers (datasets ’flat2’, ’curved2’, and ’step2’). In total, the dataset consists of 24 different cases. The CMap is learned sequentially for each case and evaluated based on the position error of the last 9 predicted positions. The evaluation is repeated for 10 different instances for each subset to obtain statistically reliable results.

In a second evaluation, CMap learning is compared to other methods on the dataset ’flat2’. LWPR cannot deal with the multiple point data such that only ROGER is compared to four different update schemes for C: Stochastic gradient decent [8] (called LMS), LMS with a decay factor (wLMS), the proposed update (19), and a variant of the latter without decay factor (Chi2). Since ROGER breaks down for 50% outliers, the number of outliers had to be reduced to 25% by showing every other point with known point correspondence (i.e., from dataset ’flat’).

<table><tr><td>scheme</td><td>proposed</td><td>LMS</td><td>Chi2</td><td>wLMS</td></tr><tr><td> $\overline { { N _ { j } } }$  input</td><td>22</td><td>22</td><td>24</td><td>22</td></tr><tr><td> $\boldsymbol { N } _ { j } ^ { \cdot }$  output</td><td>24</td><td>20</td><td>26</td><td>20</td></tr></table>

TABLE 2  
Number of basis functions for the different schemes.

optimal parameter median absolute error  
![](images/06edd7a2245d0116d5f7c619f9e9f72b716b3c122162af5cd99a74cbd854f10b.jpg)  
Fig. 7. D2: Localization error of the proposed scheme depending on the surface complexity, noise level, and correspondence. $\mathsf { A } \ ' 2 ^ { \prime }$ indicates that two points are contained in each sample.

The parameters for ROGER have been manually tuned to obtain low errors. The input data has been rescaled to $[ - 0 . 5 ; 0 . 5 ]$ , the output data to [−2.5; 2.5]. The number of experts has been set to $P = 2 0$ and the other parameters have been $\alpha = 0 . 1 , \mu = 0 . 5 ,$ , and $\lambda = 0 . 1$ . For the proposed method, the decay factor remained as in Section 3.1. The $N _ { j }$ have been optimized for each method over the whole dataset and for an area of 400×400 pixels, see Table 2.

## 3.4 Results on synthetic projections dataset

The median absolute position error for the CMap scheme on all 24 datasets is plotted in Figure 7. The accuracy degrades slightly for learning without correspondences and with increasing complexity of the surfaces. As expected [7], wrong correspondences are sorted out by low likelihood of consistency through the dataset. The shape of the surface becomes less relevant with increasing noise level. CMap also deals successfully with the nonunique solution at the discontinuity, due to its multimap capabilities. The overall accuracy is significantly better than the resolution of the grid (grid cells are bout $2 0 \times 2 0$ pixels), cmp. Figure 7.

For the second evaluation, the mean and median absolute error for the last 9 predictions for ’flat2’ (averaged over 10 instances) are plotted in Figure 8. Consistent median and mean error indicate a low outlier rate. In most cases, the outlier rates are very low, except for ROGER. As expected the error grows with the noise level and LMS and wLMS perform equally well. The weighting only matters here if the data generating process is not stationary. The proposed scheme works best throughout, despite that its unweighted variant (Chi2) gave the poorest accuracy among the CMap optimization schemes. Although the accuracy of ROGER is slightly better than CMap for corresponding points (cf. Figure 5), its performance degrades under the presence of outliers.

![](images/61c7e4b42021b42438a17537fdb6cda1018eced4ec10f61c67cef29e2b177a21.jpg)

optimal parameter median absolute error  
![](images/037e19d9c97a64c38ef1787bfadb2abeb3bde761a22a2c0cfade4bddc3a9d412.jpg)  
Fig. 8. Experiment on data set D2, flat2. Localization error depending on the method, plotted over the noise level. The training samples contained outliers.

## 3.5 Surveillance datasets (D3, D4, and D5)

One of the major application areas for learning CMap is multi-camera surveillance. The PETS2001 datasets<sup>4</sup> are a popular testbed for evaluating surveillance algorithms. Since ground truth information is needed for evaluation, TESTING dataset 1 (2688 frames) has been used here. The center positions of all clearly visible pedestrians (8 instances) have been used for comparing ROGER and CMap learning. More than 64% of the training samples contain more than one point and no information about point correspondences is given. In some cases, small groups of people move through the scene, represented by a cluster of points that behaves like a non-rigid object.

All 2688 frames are randomly re-ordered since otherwise the temporal consistency of point positions would not allow us to perform a proper cross-evaluation with the respectively subsequent samples. The point positions in cameras 1 and 2 are fed frame by frame into the respective method. For each new frame, first the positions in camera 2 are estimated for all points in camera 1. Then the two sets of points are added to the learning. The whole procedure has been repeated 10 times, with new re-orderings.

For ROGER, the standard parameters are used, except for $P = 3 0$ and the data is downscaled to the interval $[ - 1 ; 1 ]$ . For CMap learning, the decay factor is $\gamma = 1 - 5$ $1 0 ^ { - 3 }$ and the basis functions are placed with a spacing of 16 pixels.

CMap learning has also been applied to dataset D4, provided by the PROMETHEUS project<sup>5</sup>, and dataset D5. Dataset D4 contains positions of persons moving in the scene that are detected with a combination of background modeling and head detection. Due to shadows it is difficult to detect the point on the ground where the persons stand. A detector of radial symmetry is used to find the heads. Dataset D5 is a novel dataset acquired in a spiral staircase from two uncalibrated views, one from the side and one from the top. Five different sequences show two people walking up and down the stairs. From these sequences, the image positions of the head centers are extracted and used for learning.

The only parameters that have been changed for D4/D5 compared to D3 are the number of channels: 50×32 for D4 and 34×26 for D5. The evaluation is purely qualitatively, as only the most likely correspondences between the views are indicated.

## 3.6 Results on surveillance datasets

An example for the position accuracy of CMap on PETS2001 is given in Figure 1. The quantitative results for ROGER and CMap on PETS2001 with unknown correspondence are plotted in Figure 9. For comparison, also the accuracy of normalized DLT homography estimation [3] on the respective point-sets with known correspondence (i.e. no outliers) is included. Each box illustrates the distribution of the distance to the true center; the box contains the 2nd and 3rd quantile and the red line is the median. The position estimates are pooled depending on the number of previously seen samples, i.e. the second box contains all estimates of the respective method after between 69 and 88 learning samples. CMap has initially a larger error but converges towards about half the error of ROGER and achieves basically the same accuracy as normalized DLT homography estimation, despite that the latter uses correspondence information. The convergence behavior is exactly opposite to that one observed in Figure 5, which is presumably caused by the fact that ROGER generates models covering the whole image when the first learning samples are observed. These global models seem to be maintained throughout and only if the error of the global models becomes too high, newer local models are used for the output – as it happens in the case of the cross 2D data. For the PETS2001 data, the error of the global model is apparently too low to trigger the use of new local models. CMap, on the other hand, uses solely local models and performs sub-optimal on data that perfectly suits a combination of global models (such as cross 2D data). For data with varying local behavior, CMap learning results in models with better fidelity than ROGER, but requires more learning samples to converge. According to the plots in Figure 9, ROGER has converged at about 350 samples, whereas CMap needed about twice as many.

![](images/ecbe3a6324cd1011896172d9e5ff92f663cb129ac8e4ea6c1d54cc853916f37c.jpg)  
Fig. 9. Results for ROGER (top), CMap (middle), and standard homography estimation (bottom) on PETS2001. ROGER and CMap are trained without known correspondences, whereas homographies have been estimated using corresponding points.

The convergence speed of CMap depending on γ and on the ordering of the data (ordered vs. random permutations) has been analyzed for 100 random subsets each, see Figure 10. The parameter γ has been varied between $1 { - } 4 . 3 { \cdot } 1 0 ^ { - 4 }$ and 0.5. Despite this wide range, results vary only mildly and only in the ordered case, with larger values for γ giving slightly more accurate results. The figure also shows that keeping data ordered results in highly accurate results after just a few iterations. This is caused by the locality of subsequent samples, i.e., new samples are close to previously seen positions. However, this also means that the mapping initially (after about 40 iterations) only covers a small portion of the view and has a tendency to over-fit. Adding more learning samples makes the error increase slightly, but a larger area of the view is covered. Random permutations of learning samples converges slower, but covers the whole view globally right from the beginning.

![](images/6a7621a8f2d5e801b66db24055e41de20fdedfe70f66c313b3a7de782569e0d5.jpg)  
Fig. 10. Convergence for ordered data (red) and permuted data (green). The solid lines indicate the mean logarithmic error over all γ, the dashed lines the maximum and minimum logarithmic error, respectively.

On the PROMETHEUS dataset, CMap finds correct correspondences already after about 150 learning frames, see Figure 11 (a) and accompanying video. The input data contains outliers (false positive detections and false negative detections) in the input space and in the output space, but CMap learning is not affected for the same consistency argument as in Section 3.4. The only exceptions are in cases where too few examples have been observed, Figure 11 (c), or two detection outliers in close vicinity occur, Figure 11 (e). In both cases, only a single frame has been affected, which can easily be corrected by temporal filtering.

On dataset D5, CMap has been trained on four of the sequences (about 150 frames) and is evaluated on the respectively 5th sequence. CMap finds correct correspondences in nearly all frames, except for a few single frames where a correspondence is missing due to the confidence of the mapping being too low. Two frames from the data set are shown in Figure 12. The spiral staircase experiment shows that CMap learning is not restricted to planar surfaces, but also works for curved surfaces with discontinuities.

![](images/744ee6814c712a6bd7a6f867d5fe95a8e0027ce26dfe884a4cf74bda7cf1f3b9.jpg)  
(e, frame# 185)  
Fig. 11. Examples from the PROMETHEUS dataset D4. Correct association of detected walking persons (a, b); wrong association due to missing detection in the left view and short learning period (c); correct association in a similar situation, but longer learning period (d); wrong association due to confused detections of two close-by persons (e). For further examples, see video.

![](images/1c45a54d307caa0cc4f6ae8905899b78107d4b3fb0f0e5e1dda911570c07778b.jpg)

(a, frame# 21, sequence# 1)  
![](images/6101dc5785896eac50f263c3066dd712ab371e1d601d96465ae136c6f538f735.jpg)  
(b, frame# 22, sequence# 4)

Fig. 12. Examples from the spiral staircase dataset D5.   
For further examples, see video.

A new method for online learning of the CMap between images has been introduced in this paper. Neyman’s chisquare divergence is proposed for learning the linear model C on density representations. This distance measure is theoretically appropriate as it is designed to compare densities, i.e., non-negative functions, in contrast to the least-squares distance. A new iterative algorithm is derived, which is a proper online learning method and stores all previous data in the fixed size model C and a fixed size vector Ω. Compared to stochastic gradient descent, information from previous data is kept in terms of Ω. The update of C is computed efficiently from pointwise operations on the previous data in terms of C and Ω and the weighted residual of the incoming data. Its efficiency makes the algorithm very suitable for realtime systems. CMap learning has been applied to several surveillance datasets, and it shows better accuracy and produces fewer outliers for position estimates than other updating schemes for C, e.g. LMS. On standard benchmarks, the proposed method is significantly more accurate than LWPR and ROGER at the same computational cost. Only with parameter setting leading to prohibitively high computation efforts, ROGER achieves marginally better accuracy. On the PETS2001 dataset, CMap learning shows faster convergence and smaller variance of results than ROGER and is more robust against outliers.

## 4 CONCLUSION

The main limitation of CMap learning is the resolution of the underlying channel representation. If two objects are closer than what can be distinguished by the channel representation, they might be confused. The position accuracy is also limited by the width of the basis functions, although it is an order of magnitude better than the spacing of the channels. On the other hand, the resolution of the channel representations should not be chosen too high, since learning will then degenerate to reproduce the noise in the learning data and the computational burden increases unnecessary. As a rule of thumb, placing channels in a regular grid with about 20 pixels spacing is a good compromise between accuracy, confusion, noise suppression, and generalization capabilities. Further problems might occur if different objects move highly correlated. In that case, CMap learning potentially confuses these objects.

The algorithm used for CMap learning is not restricted to 2D correspondence problems. However, correspondence learning is a good illustration of the properties and advantages of the proposed method. The same method has also been used for replacing the learning algorithm for tracking problems [34] within the DIPLECS<sup>1</sup> project. A video demonstrating the tracking results, among other DIPLECS achievements, is available at the project website. Future work will concentrate on applying the learning algorithm on combined mapping and tracking problems, higher-dimensional problems, and robot control.

## ACKNOWLEDGMENTS

This research has received funding from the EC’s 7th Framework Programme (FP7/2007-2013), grant agreements 215078 (DIPLECS) and 247947 (GARNICS), from ELLIIT, Strategic Area for ICT research, and CADICS, funded by the Swedish Government, Swedish Research Council project ETT, and from CUAS and FOCUS, funded by the Swedish Foundation for Strategic Research.

## REFERENCES

[1] R. P. S. Mahler, “Multitarget Bayes filtering via first-order multitarget moments,” Aerospace and Electronic Systems, IEEE Transactions on, vol. 39, no. 4, pp. 1152 – 1178, 2003.

[2] G. H. Granlund, “An Associative Perception-Action Structure Using a Localized Space Variant Information Representation,” in Proceedings of Algebraic Frames for the Perception-Action Cycle, 2000.

[3] R. I. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision, 2nd ed. Cambridge University Press, 2004.

[4] B. Micusik and T. Pajdla, “Structure from motion with wide circular field of view cameras,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 28, pp. 1135–1149, 2006.

[5] D. G. Lowe, “Distinctive image features from scale-invariant keypoints,” International Journal of Computer Vision, vol. 60, no. 2, pp. 91–110, 2004.

[6] A. Rangarajan, H. Chui, and E. Mjolsness, “A new distance measure for non-rigid image matching,” in Energy Minimization Methods in Computer Vision and Pattern Recognition, ser. LNCS, vol. 1654. Springer Berlin, 1999, pp. 734–734.

[7] E. Jonsson and M. Felsberg, “Correspondence-free associative learning,” in International Conference on Pattern Recognition, 2006.

[8] E. Jonsson, “Channel-coded feature maps for computer vision and machine learning,” Ph.D. dissertation, Linkoping University, Swe-¨ den, SE-581 83 Linkoping, Sweden, February 2008, dissertation¨ No. 1160, ISBN 978-91-7393-988-1.

[9] L. Bottou, “On-line learning and stochastic approximations,” in On-line learning in neural networks. Cambridge University Press, 1998.

[10] D. Saad, “Introduction,” in On-line learning in neural networks. Cambridge University Press, 1998.

[11] D. H. Grollman, “Teaching old dogs new tricks: Incremental multimap regression for interactive robot learning from demonstration,” Ph.D. dissertation, Brown University, May 2010.

[12] D. H. Grollman and O. C. Jenkins, “Incremental learning of subtasks from unsegmented demonstration,” in International Conference on Intelligent Robots and Systems, Taipei, Taiwan, Oct. 2010.

[13] ——, “Sparse incremental learning for interactive robot control policy estimation,” in International Conference on Robotics and Automation, Pasadena, CA, USA, May 2008, pp. 3315–3320.

[14] L. Csato and M. Opper, “Sparse on-line Gaussian processes,”´ Neural Computation, vol. 14, no. 3, pp. 641–668, 2002.

[15] S. Schaal, C. G. Atkeson, and S. Vijayakumar, “Scalable techniques from nonparametric statistics for real time robot learning,” Applied Intelligence, vol. 17, pp. 49–60, 2002.

[16] F. Larsson, E. Jonsson, and M. Felsberg, “Simultaneously learning to recognize and control a low-cost robotic arm,” Image and Vision Computing, vol. 27, no. 11, pp. 1729–1739, 2009.

[17] O. Javed, K. Shafique, Z. Rasheed, and M. Shah, “Modeling inter-camera space-time and appearance relationships for tracking across non-overlapping views,” Computer Vision and Image Understanding, vol. 109, no. 2, pp. 146 – 162, 2008.

[18] A. Gilbert and R. Bowden, “Incremental, scalable tracking of objects inter camera,” Comput. Vis. Image Underst., vol. 111, pp. 43–58, July 2008.

[19] S. Khan and M. Shah, “Consistent labeling of tracked objects in multiple cameras with overlapping fields of view,” Pattern Analysis and Machine Intelligence, IEEE Transactions on, vol. 25, no. 10, pp. 1355 – 1360, oct. 2003.

[20] L. Lee, R. Romano, and G. Stein, “Monitoring activities from multiple video streams: establishing a common coordinate frame,” Pattern Analysis and Machine Intelligence, IEEE Transactions on, vol. 22, no. 8, pp. 758 –767, aug 2000.

[21] H. P. Snippe and J. J. Koenderink, “Discrimination thresholds for channel-coded systems,” Biological Cybernetics, vol. 66, pp. 543– 551, 1992.

[22] A. Pouget, P. Dayan, and R. S. Zemel, “Inference and computation with population codes,” Annu. Rev. Neurosci., vol. 26, pp. 381–410, 2003.

[23] R. S. Zemel, P. Dayan, and A. Pouget, “Probabilistic interpretation of population codes,” Neural Computation, vol. 10, no. 2, pp. 403– 430, 1998.

[24] M. Felsberg, P.-E. Forssen, and H. Scharr, “Channel smoothing:´ Efficient robust smoothing of low-level signal features,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 28, no. 2, pp. 209–222, 2006.

[25] M. Kass and J. Solomon, “Smoothed local histogram filters,” in ACM SIGGRAPH 2010 papers, ser. SIGGRAPH ’10. New York, NY, USA: ACM, 2010, pp. 100:1–100:10. [Online]. Available: http://doi.acm.org/10.1145/1833349.1778837

[26] P.-E. Forssen and G. Granlund, “Robust Multi-Scale Extraction of ´ Blob Features,” in SCIA,2003. Goteborg: Springer LNCS, 2003.¨

[27] C. M. Bishop, Pattern Recognition and Machine Learning. Springer, 2006.

[28] B. Johansson, T. Elfving, V. Kozlov, Y. Censor, P.-E. Forssen, and ´ G. Granlund, “The application of an oblique-projected landweber method to a model of supervised learning,” Mathematical and Computer Modelling, vol. 43, pp. 892–909, 2006.

[29] S. S. Haykin, Neural networks: a comprehensive foundation. Upper Saddle River, N.J.: Prentice Hall, 1999.

[30] S. Wegenkittl, “Generalized phi-divergence and frequency analysis in markov chains,” Ph.D. dissertation, University of Salzburg, 1998.

[31] Y.-D. Kim, A. Cichocki, and S. Choi, “Nonnegative Tucker decomposition with alpha-divergence,” in Proceedings of IEEE International Conference on Acoustics, Speech, & Signal Processing, 2008.

[32] A. Cichocki, A.-H. Phan, and C. Caiafa, “Flexible HALS algorithms for sparse non-negative matrix/tensor factorization,” in Proceedings of 2008 IEEE International Workshop on Machine Learning for Signal Processing, 2008.

[33] R. Jimenz and Y. Shao, “On robustness and ef´ ficiency of minimum divergence estimators,” TEST, vol. 10, no. 2, pp. 241–248, 12 2001.

[34] M. Felsberg and F. Larsson, “Learning higher-order Markov models for object tracking in image sequences,” in ISVC, ser. LNCS, vol. 5876, 2009, pp. 184–195.

[35] S. Vijayakumar and S. Schaal, “Locally weighted projection regression: An o(n) algorithm for incremental real time learning in high dimensional space,” in Proceedings of the Seventeenth International Conference on Machine Learning, 2000, pp. 1079–1086.

![](images/0dcf3c9d3128be39f59e7240635dee5444e5e3e92532e442dc112110662c41ac.jpg)

Michael Felsberg PhD degree (2002) in engineering from University of Kiel. Since 2008 full professor and head of CVL. Research: signal processing methods for image analysis, computer vision, and machine learning. More than 80 reviewed conference papers, journal articles, and book contributions. Awards of the German Pattern Recognition Society (DAGM) 2000, 2004, and 2005 (Olympus award), of the Swedish Society for Automated Image Analysis (SSBA) 2007 and 2010, and at Fusion

2011 (honourable mention). Coordinator of EU projects COSPAL and DIPLECS. Associate editor for the Journal of Real-Time Image Processing, area chair ICPR, and general co-chair DAGM 2011.

![](images/18e9a52213ccf3fd4f3103e4372724dd6057acd0dfa07adda24f17f3ce2aa7be.jpg)

Fredrik Larsson Master of Science (MSc, 2007) and the PhD degree (2011) from Linkoping Uni-¨ versity. From 2007 until 2011, he has been a researcher at CVL, working on the European cognitive vision projects COSPAL and DIPLECS. He has published 10 journal articles and conference papers on image processing and computer vision. Awards from the Swedish Society for Automated Image Analysis (SSBA) 2010 and at Fusion 2011 (honourable mention).

![](images/747ba6562a62f59ebaaf2a289f254ee740542055d820f3b7bfe2b340427569f7.jpg)

Johan Wiklund Master of Science (MSc, 1984) and Licentiate of Technology degree (1987) from Linkoping University. Since 1984 researcher at¨ CVL, working on image processing. Participated in various European and national projects. Responsible for system integration and demonstrator development in the European projects COSPAL, DIPLECS, and GARNICS. He has published more than 30 journal articles, book contributions and conference papers.

![](images/1570dcbab656bd6f53c86c631dc9eb331b0b4f9082af453dfb8e1a93aaba0a36.jpg)

Niclas Wadstromer¨ PhD from Linkoping univer-¨ sity, Sweden, 2002, in Image coding. He has been a senior lecturer in Telecommunications with the Department of Electrical Engineering, Linkoping university. Currently, since 2007, he¨ is a Scientist with the Division of Sensor and EW systems at the Swedish Defence Research agency. Among his interests are Image and signal processing, Video analytics, Multi- and hyperspectral imaging.

![](images/cf4ad26d951dc6a4aa7da4e5942dd9b64f105cb778424e0d3ba32d25c01c6bd0.jpg)

Jorgen Ahlberg¨ Development Manager for Image Processing at Termisk Systemteknik AB. MSc in Computer Science and Engineering (1996) and PhD in Electrical Engineering (2002), both from Linkoping University. 2002–2011 at¨ the Swedish Defence Research Agency (FOI) as a scientist and the head of a research department. Visiting Senior Lecturer in Information Coding at Linkoping University and co-founder¨ of Visage Technologies AB. Research on image analysis, including animation and tracking of facial images, infrared and hyperspectral systems for automatic detection, recognition, and tracking, with applications as e.g. surveillance.