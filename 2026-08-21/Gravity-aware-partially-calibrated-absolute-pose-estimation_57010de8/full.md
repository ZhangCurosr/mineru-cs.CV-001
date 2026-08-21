# Gravity-aware partially calibrated absolute pose estimation from afineor rotation-covariant features

Marcus Valtonen Örnhag<sup>1</sup> , Alberto Jaenal<sup>2</sup> , and Stefan Adalbjörnsson<sup>1</sup>

<sup>1</sup> Ericsson Research, Lund, Sweden {marcus.valtonen.ornhag,stefan.adalbjornsson}@ericsson.com <sup>2</sup> University of Zaragoza, Spain ajaenal@unizar.es

Abstract. Inertial measurement units (IMUs) are now standard in most consumer devices, such as smartphones, drones, and extended reality (XR) headsets. By fusing visual and inertial data, localization systems gain significantly in speed and robustness compared to vision-only or IMU-only approaches. However, traditional pose estimation methods fail to utilize the local geometric information embedded in feature descriptors like SIFT. Recent work has proved the advantages of leveraging this information for relative and absolute pose estimation, but its application to partially calibrated absolute pose estimation remains unexplored. In this paper, we derive novel constraints for joint estimation of absolute pose and focal length, making use of a gravity vector obtained from IMU data and the feature-induced local geometry, which we use to construct two eficient solvers: UP1PfAC, that operates given a single afine correspondence and UP2PfORI, which requires two orientation-covariant features. Unlike traditional, semi-calibrated absolute pose methods requiring four point correspondences, our solvers benefit from fewer samples and lower computational cost, simplifying robust estimation in modern RANSAClike frameworks. We evaluate the proposed solvers against the state-ofthe-art on large-scale public datasets and demonstrate that our method achieves fast and accurate localization and focal length estimation.

Keywords: Absolute pose estimation · Structure-from-Motion · Camera calibration

## 1 Introduction

Future generations of mixed-reality applications demand faster and more accurate global localization to enable collaborative multi-user experiences. In fact, a rich flora of techniques from image-based localization [53], structure-from-motion (SfM) [55, 56], simultaneous localization and mapping (SLAM) [15], and visual odometry (VO) [44] are available in the current literature to build such largescale pipelines, yet solutions do not always transfer directly across applications. Klein and Murray [36], for instance, proposed a method tailored to augmented reality that deviates from typical SLAM by leveraging denser maps, lower-quality features, and decoupling tracking and mapping.

![](images/242bf6a30200ae5e946f9b1a21a0c48534eb54fd341aea8ab803e69439cb9969.jpg)  
Fig. 1: Overview of the problem. We propose to jointly estimate the absolute pose and focal length of a query image from a 3D reconstruction. By leveraging IMU data and the intrinsic geometric properties of local descriptors, we are able to solve the problem using a single observation from an afine correspondence (UP1PfAC), or two orientation-covariant features (UP2PfORI) to a reference image. This greatly benefits the accuracy and execution times, compared to traditional point-based methods not utilizing IMU data or feature descriptor information, which require 3.5 points or more.

At the core of these applications lies the Perspective-n-Point (PnP) problem: estimating camera poses from 2D image features and 3D scene points. In practice, minimal solvers integrated into random sample consensus (RANSAC) [24] or its variants [8, 16] deal with noisy data and outliers, hence, traditionally, camera pose hypotheses are generated from three point correspondences using the P3P algorithm, for which eficient solutions are widely available [22, 26, 33, 49].

While point-based methods have been the undisputed approach for many years, several alternative approaches that utilize afine feature descriptors were proposed in the previous decade [2,4,5,11,13,18,23,27–29,50,66]. The improvements in this field were summarized by Barath et al. [10], who argue for its continued use and benefits. Among the drawbacks of afine-based methods is the computational overhead of feature extraction and matching. For instance, the original Afine SIFT (ASIFT) [48] implementation reported a 13.5× increase of computational complexity in extraction compared to regular SIFT features [46], and a corresponding matching step of $1 3 . 5 ^ { 2 }$ ≈ 180× greater complexity. With the introduction of deep learning-based descriptors, e.g., AfNet [47], this gap has been reduced; however, the increased complexity remains a bottleneck.

Barath and Kukelova [6, 7] proposed to approximate a local afine feature A from a scale- and orientation-covariant feature, e.g., SIFT and SURF [12]. They established the following constraints

$$
a _ { 1 } c _ { \mathrm { r e f } } + a _ { 2 } s _ { \mathrm { r e f } } - q c _ { \mathrm { q u e r y } } = 0 ,\tag{1}
$$

$$
a _ { 3 } c _ { \mathrm { r e f } } + a _ { 4 } s _ { \mathrm { r e f } } - q s _ { \mathrm { q u e r y } } = 0 ,\tag{2}
$$

where the relative scale is defined as $q = q _ { \mathrm { q u e r y } } / q _ { \mathrm { r e f } }$ and the orientations $\alpha _ { \mathrm { r e f } }$ and $\alpha _ { \mathrm { q u e r y } }$ are used as $c _ { \mathrm { i } } = \cos ( \alpha _ { \mathrm { i } } ) , s _ { \mathrm { i } } = \sin ( \alpha _ { \mathrm { i } } )$ . Here $a _ { i }$ denotes the elements

of A in row-major order. If only orientation-covariant features, $e . g .$ ., ORB [51], are used one may eliminate the scale by subtracting (1) from (2) to obtain

$$
c _ { \mathrm { r e f } } s _ { \mathrm { q u e r y } } a _ { 1 } + s _ { \mathrm { r e f } } s _ { \mathrm { q u e r y } } a _ { 2 } - c _ { \mathrm { r e f } } c _ { \mathrm { q u e r y } } a _ { 3 } - c _ { \mathrm { q u e r y } } s _ { \mathrm { r e f } } a _ { 4 } = 0 \ .\tag{3}
$$

Since their discovery, these constraints have been used to derive new minimal solvers for homography estimation [3, 6, 19], relative pose estimation [7, 60], and calibrated absolute pose estimation [32,62]. In this paper, we explore applications of afine- and orientation-covariant features to improve the overall localization accuracy and computational complexity. The contributions of this paper are:

– Derivation of constraints for absolute pose and unknown focal length estimation, using afine or orientation-covariant feature descriptors,

– Two novel polynomial solvers for the upright absolute pose problem with unknown focal length using afine or orientation-covariant features

– A rigorous comparison to state-of-the-art methods, discussing the benefits, – A practical demonstration for real-world applications with focus on visual localization with respect to a global reference system shared by other users.

Notation. Following [61, 62], we use sans-serif capital A for matrices, bold x for vectors, and italic lower-case s for scalars. Subscripts indicate matrix and vector indexing, $e . g . , { \sf R } _ { 1 : 2 , 1 : 2 }$ refers to the $2 \times 2$ upper-left submatrix of R.

## 2 Related work

Absolute pose estimation with unknown focal length. The absolute pose problem with unknown focal length (PnPf) has been well-studied in both its minimal and non-minimal configurations. Since the problem has seven degrees of freedom, 3.5 point correspondences are necessary to solve the problem, i.e., ignoring one of the equations obtained from the last point correspondences. Bujnak et al. [14] proposed one of the first general solvers for this case; however, the computational complexity rendered it infeasible for use in real-time applications. Zheng et al. [69] was able to solve the problem in a fast enough way to consider it for real-time applications, which was further improved upon by Wu [64]. Later, Larsson et al. [40] proposed a solver of a similar performance, using novel constraints for the particular setup. The overdetermined case was treated by Zheng et al. [69] and recently by Zhou et al. [70].

Upright solvers. Kukelova et al. [38] derived a minimal gravity-aware two point solver solving the absolute pose problem (UP2P) also considering an upright three point solver with unknown focal length and radial distortion coeficient (UP3Pfk). Sweeney et al. [58] applied a similar approach to the UP2P solver, and demonstrated its superior performance for localization in AR applications. Recently, an upright solver requiring only a single scale- and orientation-covariant feature was proposed in [62]. This solver, called UP1SIFT, was derived from (1)– (2), and the theory for local afine features, discussed in the next section. Beyond absolute pose, gravity alignment has been leveraged for other problems such as globally optimal relative pose estimation [20] and 3D camera registration [25].

Afine correspondences and unknown focal length. The methods utilizing afine correspondences to estimate focal length in some way do not treat the absolute pose problem; however, it has been studied for the relative pose problem with unknown focal length. Barath et al. [11] use two afine correspondences to solve it. Hruby et al. [32] showed that it is possible to use a single afine feature and estimated monocular depth. Furthermore, Ding et al. [21] treated diferent cases of fundamental matrix estimation using relative depths, which can be obtained from afine features, learned scales or depths. Recently, Yu et al. [67] proposed a method to solve the relative pose problem with unknown focal length and a known gravity vector by utilizing two afine features. Their approach leads to a polynomial eigenvalue problem, which can be solved eficiently and robustly, showing that correct utilization of afine features in combination with IMU data can significantly aid in simultaneous localization and camera calibration.

## 3 Method

We extend the work by Ventura et al. [61, 62] by introducing an unknown focal length parameter. We propose to solve the problem using a single afine correspondence (UP1PfAC) or two orientation-covariant correspondences (UP2PfORI) with known gravity direction, see ${ \mathrm { F i g . } }$ 1.

We aim to compute the unknown parameters of a query camera w.r.t. a world coordinate system. Following the notation introduced in [62], the known rotational components are encoded in $\mathsf { R } _ { x z }$ , which allows for the parameterization of the full query rotation matrix ${ \sf R } _ { \mathrm { q u e r y } } = { \sf R } _ { y } { \sf R } _ { x z }$ . The unknown query translation is denoted $\mathbf { t } _ { \mathrm { q u e r y } }$ . The PnP problem requires a pre-built 3D reconstruction from a set of reference camera images. As such, the known reference camera poses used to construct the map also consist of a rotation and translation denoted $\sf R _ { \mathrm { r e f } }$ and $\mathbf { t } _ { \mathrm { r e f } }$ , respectively. In this setting, the relative rotation between the reference camera and the query camera is given by ${ \sf R } = { \sf R } _ { \mathrm { q u e r y } } { \sf R } _ { \mathrm { r e f } } { \sf T }$ , and the corresponding relative translation is given by $\mathbf { t } = \mathbf { t } _ { \mathrm { q u e r y } } - \mathsf { R t } _ { \mathrm { r e f } }$

Let $\mathbf { p } _ { \mathrm { r e f } }$ and $\mathbf { p } _ { \mathrm { q u e r y } }$ denote two corresponding inhomogeneous image point correspondences in reference and query images, respectively. Since we aim to utilize their point-based pixel coordinates to estimate the query pose and unknown focal length, we utilize the relationship between a local afine transformation and the pose of the query image relative to the reference image, as originally introduced in [61] and later modified for the world coordinate system in [62]:

$$
\mathsf { A } = \frac { d } { m } \biggl ( b \mathsf { R } _ { 1 : 2 , 1 : 2 } - \mathbf { p } _ { \mathrm { r e f 1 : 2 } } ^ { \prime } \mathbf { n } _ { \mathrm { r e f 1 : 2 } } - \mathbf { p } _ { \mathrm { q u e r y } } \left( b \mathsf { R } _ { 3 , 1 : 2 } - \mathbf { p } _ { \mathrm { r e f _ { 3 } } } ^ { \prime } \mathbf { n } _ { \mathrm { r e f _ { 1 : 2 } } } \right) \biggr ) ,\tag{4}
$$

where d is the depth of the point $\mathbf { p } _ { \mathrm { r e f } }$ in the reference image, $\tilde { \bf p } _ { \mathrm { r e f } }$ is the homogeneous representation of $\mathbf { p } _ { \mathrm { r e f } } , \mathbf { n } _ { \mathrm { r e f } } = \mathsf { R } _ { \mathrm { r e f } } \mathbf { n }$ is the normal vector transformed to the coordinate system of the reference frame, $b = \mathbf { n } _ { \mathrm { r e f } } ^ { T } \tilde { \mathbf { p } } _ { \mathrm { r e f } } , \mathbf { p } _ { \mathrm { r e f } } ^ { \prime } = \mathsf { R } \tilde { \mathbf { p } } _ { \mathrm { r e f } } ,$ , and $m = { \mathbf { n } _ { \mathrm { r e f } } } ^ { T } \tilde { \mathbf { p } } _ { \mathrm { r e f } } ( d ( \mathsf { R } _ { 3 , : } \tilde { \mathbf { p } } _ { \mathrm { r e f } } ) + \mathrm { t } _ { 3 } )$ . Directly working with the world coordinate system enables the usage of correspondences in diferent reference images, thereby increasing the number of possible samples for RANSAC. This is often utilized in modern localization frameworks, $e . g .$ , HLOC [52].

## 3.1 Incorporating the IMU data

As opposed to previous work, we assume that two degrees of freedom can be reduced due to a known gravity direction, typically obtained using an IMU sensor. In this situation, the camera rotation can be decomposed as $\mathsf { R } = \mathsf { R } _ { y } \mathsf { R } _ { x z }$ where the known rotation is encoded in $\mathsf { R } _ { x z } .$ , aligning the remaining unknown yaw angle with the y-axis in the coordinate system. Let θ denote the unknown angle of rotation about the y axis of the query camera, then

$$
\mathsf { R } _ { y } = \left[ \begin{array} { c c c } { \cos \theta 0 - \sin \theta } \\ { 0 1 } & { 0 } \\ { \sin \theta 0 } & { \cos \theta } \end{array} \right] ,\tag{5}
$$

The tangent half-angle parameterization $r = \tan ( \theta / 2 )$ , results in

$$
\cos \theta = { \frac { 1 - r ^ { 2 } } { 1 + r ^ { 2 } } } \quad { \mathrm { a n d } } \quad \sin \theta = { \frac { 2 r } { 1 + r ^ { 2 } } } ,\tag{6}
$$

which is suitable for polynomial solvers. Furthermore, due to the scale ambiguity, it is possible to work with scaled rotations, i.e. $\mathsf { R } ^ { T } \mathsf { R } = s \mathsf { I }$ , where $s \neq 0 .$ , further simplifying the equations. The tangent half-angle parameterization cannot represent a $1 8 0 ^ { \circ }$ rotation; however, this can be addressed by applying a random rotation to the variables [31].

## 3.2 Novel constraints

The unknown focal length $f _ { \mathrm { q u e r y } }$ can be introduced by substituting $\mathbf { p } _ { \mathrm { q u e r y } } \mapsto$ $\mathbf { p } _ { \mathrm { q u e r y } } / f _ { \mathrm { q u e r y } }$ in (4), and since the reference focal length is assumed to be known, we assume that p˜<sub>ref</sub> is already normalized by it. Furthermore, the afine features are afected $A \mapsto f _ { \mathrm { r e f } } / f _ { \mathrm { q u e r y } } \bar { A }$ . Due to space constraints, we omit the subindex of the unknown query focal length and simply write $f .$ . This yields

$$
\frac { a _ { 1 } } { f } = \frac { d } { m } \big ( b r _ { 1 1 } - p _ { \mathrm { r e f _ { 1 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } - \frac { p _ { \mathrm { q u e r y _ { 1 } } } } { f } \big ( b r _ { 3 1 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } \big ) \big ) ,\tag{7}
$$

$$
\frac { a _ { 2 } } { f } = \frac { d } { m } \big ( b r _ { 1 2 } - p _ { \mathrm { r e f _ { 1 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } - \frac { p _ { \mathrm { q u e r y _ { 1 } } } } { f } \big ( b r _ { 3 2 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } \big ) \big ) ,\tag{8}
$$

$$
\frac { a _ { 3 } } { f } = \frac { d } { m } \big ( b r _ { 2 1 } - p _ { \mathrm { r e f _ { 2 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } - \frac { p _ { \mathrm { q u e r y _ { 2 } } } } { f } \big ( b r _ { 3 1 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } \big ) \big ) ,\tag{9}
$$

$$
\frac { a _ { 4 } } { f } = \frac { d } { m } ( b r _ { 2 2 } - p _ { \mathrm { r e f _ { 2 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } - \frac { p _ { \mathrm { q u e r y _ { 2 } } } } { f } ( b r _ { 3 2 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } ) ) .\tag{10}
$$

Multiplying (7)–(10) by mf yields polynomial equations. Similarly, the orientationcovariant constraint (3) becomes

$$
\begin{array} { r l } & { C _ { \mathrm { r e f } } s _ { \mathrm { q u e r y } } \big ( b r _ { 1 1 } - p _ { \mathrm { r e f } _ { 1 } } ^ { \prime } n _ { \mathrm { r e f } _ { 1 } } - \frac { p _ { \mathrm { q u e r y } _ { 1 } } } { f } \big ( b r _ { 3 1 } - p _ { \mathrm { r e f } _ { 3 } } ^ { \prime } n _ { \mathrm { r e f } _ { 1 } } \big ) \big ) } \\ & { \phantom { D _ { \mathrm { r e f } } s _ { \mathrm { q u e r y } } } + s _ { \mathrm { r e f } } s _ { \mathrm { q u e r y } } \big ( b r _ { 1 2 } - p _ { \mathrm { r e f } _ { 1 } } ^ { \prime } n _ { \mathrm { r e f } _ { 2 } } - \frac { p _ { \mathrm { q u e r y } _ { 1 } } } { f } \big ( b r _ { 3 2 } - p _ { \mathrm { r e f } _ { 3 } } ^ { \prime } n _ { \mathrm { r e f } _ { 2 } } \big ) \big ) } \\ & { \phantom { D _ { \mathrm { r e f } } s _ { \mathrm { q u e r y } } } - c _ { \mathrm { r e f } } c _ { \mathrm { q u e r y } } \big ( b r _ { 2 1 } - p _ { \mathrm { r e f } _ { 2 } } ^ { \prime } n _ { \mathrm { r e f } _ { 1 } } - \frac { p _ { \mathrm { q u e r y } _ { 2 } } } { f } \big ( b r _ { 3 1 } - p _ { \mathrm { r e f } _ { 3 } } ^ { \prime } n _ { \mathrm { r e f } _ { 1 } } \big ) \big ) } \\ &  \phantom { D _ { \mathrm { r e f } } c _ { \mathrm { q u e r y } } } - s _ { \mathrm { r e f } } c _ { \mathrm { q u e r y } } \big ( b r _ { 2 2 } - p _ { \mathrm { r e f } _ { 2 } } ^ { \prime } n _ { \mathrm { r e f } _ { 2 } } - \frac { p _ { \mathrm { q u e r y } _ { 2 } } } { f } \big ( b r _ { 3 2 } - p _ { \mathrm { r e f } _ { 3 } } ^ { \prime } n _ { \mathrm { r e f } _ { 2 } } \big ) \big ) = 0  \end{array}\tag{11}
$$

which are polynomial after multiplication by f. Finally, the two point-based constraints are given by (multiplied after by $f$ to become polynomial)

$$
\frac { p _ { \mathrm { q u e r y } _ { 1 } } } { f } ( d \mathsf { R } _ { 3 , : } \tilde { \mathbf { p } } _ { \mathrm { r e f } } + t _ { 3 } ) - \left( d \mathsf { R } _ { 1 , : } \tilde { \mathbf { p } } _ { \mathrm { r e f } } + t _ { 1 } \right) = 0 ,\tag{12}
$$

$$
\frac { p _ { \mathrm { q u e r y } _ { 2 } } } { f } ( d \mathsf { R } _ { 3 , : } \tilde { \mathbf { p } } _ { \mathrm { r e f } } + t _ { 3 } ) - ( d \mathsf { R } _ { 2 , : } \tilde { \mathbf { p } } _ { \mathrm { r e f } } + t _ { 2 } ) = 0 .\tag{13}
$$

## 3.3 Derivation of UP1PfAC

For the case of known gravity direction and unknown focal length with afine features, we use (7)–(10) and (12)–(13), which leaves us with a total of six equations and five unknowns: one rotational component (due to known reference direction), three translational components, and the focal length. In general, the problem is overdetermined, and therefore, we discard one of the equations. Experimentally, we found that discarding the last afine equation (10) works well. This equation, however, can be utilized later, as will be discussed. Note that all equations (7)–(13) are linear in $\mathbf { t } _ { \mathrm { q u e r y } }$ , hence we may write the complete system of equations in the compact form

$$
\mathsf { M } ( r , f ) \left[ \mathsf { \mathbf { t } _ { q u e r y } } \right] = \mathbf { 0 } ,\tag{14}
$$

where M is a $5 \times 4$ matrix. The same methods used in [61,62] cannot be applied, as the unknown query focal length increases the number of monomials. Instead, we note that M must have a non-trivial nullspace; hence, it follows that all $4 \times 4$ subdeterminants of M must vanish. Furthermore, the structure of M can be utilized, as it may be written as

$$
\mathsf { M } = \left[ \begin{array} { c c c c } { - \xi } & { 0 } & { m _ { 1 3 } m _ { 1 4 } } \\ { 0 } & { - \xi } & { m _ { 2 3 } m _ { 2 4 } } \\ { 0 } & { 0 } & { m _ { 3 3 } m _ { 3 4 } } \\ { 0 } & { 0 } & { m _ { 4 3 } m _ { 4 4 } } \\ { 0 } & { 0 } & { m _ { 5 3 } m _ { 5 4 } } \end{array} \right] ,\tag{15}
$$

where $\xi = f _ { \mathrm { r e f } } f ( 1 + r ^ { 2 } )$ . It follows that only three subdeterminants subdet $( \mathsf { M } ) _ { i , j }$ are non-trivially zero and these can be written as

$$
\mathrm { s u b d e t } ( \mathsf { M } ) _ { i , j } = ( 1 + r ^ { 2 } ) ^ { 2 } f ^ { 2 } h _ { i , j } ( r , f ) ,\tag{16}
$$

where $h _ { i , j }$ are polynomials in the monomial basis $\boldsymbol { B } = \{ r ^ { 4 } f , r ^ { 4 } , r ^ { 3 } f , r ^ { 3 } , r f , r , f , 1 \}$ As $1 + r ^ { 2 } \neq 0 ,$ , due to our choice of parameterization, we can simply ignore these roots. Similarly, $f = 0$ can be discarded as a physically improbable focal length. Additionally, only two of the three equations are linearly independent, and we may discard the second equation. The resulting system of equations is linear in $f ,$ hence we may repeat the same argument as before, and write

$$
( 1 + r ^ { 2 } ) ^ { 2 } \bar { \mathsf { M } } ( r ) \left[ \begin{array} { l } { f } \\ { 1 } \end{array} \right] = \mathbf { 0 } ,\tag{17}
$$

where $\bar { \mathsf { M } }$ is a $2 \times 2$ matrix. Again, M<sup>¯</sup> has a non-trivial nullspace. The resulting equation $\bar { h } ( r ) = \operatorname* { d e t } ( \bar { \mathsf { M } } ) ( r )$ , is quartic, hence an analytical solution is available. In general, there are four solutions, and it can be shown that the original system of equations also has four solutions. $\mathrm { N e x t }$ , we use (17) to extract $f$ using backsubstitution, followed by extracting $\mathbf { t } _ { \mathrm { q u e r y } }$ from (14) in a similar manner. Now, the previously unused constraint (10) can be used to identify the correct solution, selecting the one that minimizes its residual. This guarantees that a single solution is obtained, which is beneficial in a RANSAC-like framework, as it limits the number of hypotheses to test against the full consensus set.

## 3.4 Derivation of UP2PfORI

We treat the case of two orientation-covariant features in a similar manner. Again, there are six equations and only five unknowns, and we will use only one equation of the form (11). Note that (11) is independent of $\mathbf { t } _ { \mathrm { q u e r y } }$ . We use the point-based equations (12)–(13), which give us four equations which are linear in $\mathbf { t } _ { \mathrm { q u e r y } }$ , and proceed as before, by starting from the (partial) system of equations (14) where M now is $\mathrm { ~ a ~ 4 ~ } \times \mathrm { ~ 4 ~ }$ matrix. Again, it has a non-trivial nullspace, hence det $( \mathsf { M } ) = 0$ . The structure of M for this case is

$$
\mathsf { M } = \left[ \begin{array} { c c c c } { - \xi _ { 1 } } & { 0 } & { m _ { 1 3 } m _ { 1 4 } } \\ { 0 } & { - \xi _ { 1 } m _ { 2 3 } m _ { 2 4 } } \\ { - \xi _ { 2 } } & { 0 } & { m _ { 3 3 } m _ { 3 4 } } \\ { 0 } & { - \xi _ { 2 } m _ { 4 3 } m _ { 4 4 } } \end{array} \right] .\tag{18}
$$

where $\xi _ { i } = f _ { \mathrm { r e f } , i } f ( 1 + r ^ { 2 } )$ . Repeating the same argument previously used, the nullspace is non-trivial, and det $( \mathsf { M } ) = ( 1 + r ^ { 2 } ) ^ { 2 } f ^ { 2 } g ( r , f )$ , where

$$
\begin{array} { r l } & { g ( r , f ) = f _ { \mathrm { r e f , 2 } } ^ { 2 } ( m _ { 1 3 } m _ { 2 4 } - m _ { 1 4 } m _ { 2 3 } ) + f _ { \mathrm { r e f , 1 } } ^ { 2 } ( m _ { 3 3 } m _ { 4 4 } - m _ { 3 4 } m _ { 4 3 } ) } \\ & { \qquad + f _ { \mathrm { r e f , 1 } } f _ { \mathrm { r e f , 2 } } ( m _ { 1 4 } m _ { 4 3 } + m _ { 2 3 } m _ { 3 4 } - m _ { 2 4 } m _ { 3 3 } - m _ { 1 3 } m _ { 4 4 } ) . } \end{array}\tag{19}
$$

Furthermore, it can be shown that $g ( r , f ) = ( 1 + r ^ { 2 } ) \bar { g } ( r , f )$ , due to the structure of the elements $m _ { i j }$ . The resulting equation $\bar { g } ( r , f ) = 0$ can now be used in conjunction with one of the orientation covariant constraints (11) to form a system of equations with two equations in the two remaining unknowns r and $f .$ This system is again linear in $f _ { : }$ , and we may write it as

$$
\bar { \mathsf { M } } ( r ) \left[ \begin{array} { l } { f } \\ { 1 } \end{array} \right] = \mathbf { 0 } ,\tag{20}
$$

hence $\operatorname* { d e t } ( { \bar { \mathsf { M } } } ) = 0$ gives a final equation in the unknown $r .$ This is again a quartic equation and can be solved analytically. It can be shown that the original system also has four solutions, and we may use the second (previously unused) equation (11) to single out the best hypotheses among these solutions.

## 4 Experiments

In our experiments we include the P4Pf solver by Kukelova et al. [39] and the P3.5Pf solver by Larsson et al. [40]. The method by Wu [64] is reported to perform similarly to the latter and is therefore omitted. Among the gravityaware solvers we include the recent UP1SIFT solver by Ventura et al. [62] and two solvers from [38]: the point-based minimal two-point, UP2P, and a semi-calibrated point-based UP2.5Pf solver, which was re-implemented from the UP3Pfk solver, assuming the radial distortion parameter $k = 0$ . Furthermore, we include the afine P1AC solver [61]. Note that among these competing stateof-the-art methods, only the P4Pf, P3.5Pf and UP2.5Pf solvers estimate the unknown focal length, while the others assume this known and accounted for. Furthermore, among the methods using more than one correspondence, only the proposed UP2PfORI is capable of utilizing correspondences across multiple reference images simultaneously.

## 4.1 Synthetic experiments

To understand the characteristics of the proposed solvers we generate a synthetic scene by adopting the methodology used in [23,61,62]. Camera centers are generated for reference and query cameras, with the query positioned 2 units from the world coordinate system origin. The camera orientation is randomly generated to face the scene and the known gravity direction is obtained by decomposing it. A single reference camera is positioned at the origin with unit orientation, and random 3D points within a cube of side length 2 are generated. The unknown query focal length is chosen uniformly random in the range [200, 1200], and the 3D points are projected through the cameras to generate the image point correspondences. Following [4], we generate local afine features by first estimating a homography corresponding to each of the 3D world points, which requires a 3D normal unit vector that is randomly drawn from a normal distribution. When the local afine transformation matrix is computed, the scales and orientations are extracted, as outlined in [62]. Physically improbable configurations, where 3D points are behind the cameras, the afine transformation does not have a positive determinant, or the rotation between query and reference cameras is greater than $1 8 0 ^ { \circ }$ are discarded. The following metrics are used:

Rotation error: The standard angular distance (in degrees) between the estimated $\sf R _ { \mathrm { e s t } }$ and the ground truth rotation $\mathsf { R } _ { \mathrm { g t } }$ is used,

$$
\mathsf { R } _ { \mathrm { e r r } } : = \operatorname { a r c c o s } \left( \frac { \operatorname { t r } ( \mathsf { R } _ { \mathrm { g t } } ^ { T } \mathsf { R } _ { \mathrm { e s t } } ) - 1 } { 2 } \right) ,\tag{21}
$$

which can be seen as the minimum angle that aligns the rotations [30, 54]. Position error: The distance between the estimated camera center $\mathbf { c } _ { \mathrm { e s t } }$ and the ground truth rotation $\mathbf { c } _ { \mathrm { g t } }$ , defined as $\mathbf { t } _ { \mathrm { e r r } } : = \| \mathbf { c } _ { \mathrm { g t } } - \mathbf { c } _ { \mathrm { e s t } } \|$ Focal length error: The normalized focal length error $f _ { \mathrm { e r r } } : = | f _ { \mathrm { g t } } - f _ { \mathrm { e s t } } | / f _ { \mathrm { g t } }$

![](images/809fba366e0c2ab9e1082fee6579515283f32acc691462683a6d4845f67dadf5.jpg)  
Fig. 2: Numerical stability. Frequency histogram (5000 trials) of $\log _ { 1 0 }$ errors in rotation, translation, and focal length estimation across diferent solvers, evaluated on noise-free data. A more stable method results in a distribution shifted further to the left. Best viewed in color.

Numerical stability We generated 5000 random configurations of the synthetic environment described above, and statistics for each method are collected on them all. No noise is added to demonstrate the numerical stability of each solver, and the results are shown in Fig. 2. All solvers should be considered numerically stable, with the majority of errors less than $1 0 ^ { - 1 2 }$ . Among the methods estimating focal length, the proposed UP1PfAC solver and UP2.5Pf are the most stable.

Sensitivity to noise Next, we inject Gaussian noise with zero mean and varying standard deviation to the image points to simulate matching errors naturally occurring in real localization pipelines. Image point noise is added with a constant standard deviation of 1.2 pixels; this value was chosen to represent the actual real-world scenarios, as most modern features typically exhibit errors up to approximately 1.2 pixels [57], although methods achieving subpixel accuracy exist [35]. In another experiment to simulate IMU noise, the gravity vector is perturbed with a random vector drawn from a normal distribution with zero mean and varying standard deviation. Modern consumer devices have an IMU error of up 0.2 degrees, but high-end products may be below 0.05 degrees [38], which justifies the ranges of the varying noise input.

The results are shown in Fig. 3, where the median value over 1000 random test instances is reported for each noise level and method. Of particular importance is the focal length errors, as the proposed methods are the first among the category of afine-based and orientation-covariant solvers to incorporate unknown focal length. Regarding sensitivity to image noise, the proposed methods perform superior, and even with the addition of extra IMU noise, the proposed methods show an advantage for reasonable consumer device levels of IMU noise.

Execution time All solvers are implemented in $\mathrm { C } { + } +$ , compiled with the same optimization flags, and executed with a standard laptop equipped with an 11th Gen Intel(R) Core(TM) i5-1145G7 @ 2.60GHz. We use public implementations where available: the P4Pf [39] and UP2P [38] are included in PoseLib [42]<sup>3</sup>,

![](images/657096767efde9216a254278dd767725504d514a6d0550c4509d4b21a2a310d0.jpg)

![](images/b7194c162f1853fa58f9c33a6d48f1e1674d8b3526cc5d3428c9d844bda904aa.jpg)

![](images/962e9aeee29a575a5b72897a1a3e7bd14298c6b7c94db023fd10d5cc7585a87f.jpg)

![](images/a4f0e3259263b81f3b493f464565715905804bf26b16378656257e6473e6ac34.jpg)

![](images/c04b9fc531e6082102926e3caae8a4e6624e7ab49f731c42c01beb95551cb251.jpg)

![](images/a0cb2c248ad47f68f7f24a188296b5aa4379f7b0da6c54ccbe5db360ade1e264.jpg)

![](images/52993af992f418cee407bf76d05ff88adf39eb87ac2c73907be9120c7047afef.jpg)

![](images/a8be8229b32237b11448e1dee13151fae8387e31725e58ac60528b419dfa6f83.jpg)

![](images/53305c535e519d48fcef1eef89a8c2470be7f945520c6661561819fa7bf501ba.jpg)

![](images/2d37b540bbd386b14065fe037f20b6b682ede25622131019eb31f040e62aa58b.jpg)

![](images/875b9ad2acde7b2ae926638b6b096ca37c656f180e9edd4c4fab4b3203164f4d.jpg)

![](images/3ae009a22c2f194f1383c0000c7a20e28596f8efd9470d399a1959630c36cd30.jpg)

![](images/c1eaffb4688fdfd6787beb84f589aa8e536b2ceba152ff3d817cb035421cf813.jpg)

![](images/9d9fbf5b623e7486ff9f3822608cb5e260119e85507da18c41310487b84ac7b2.jpg)

![](images/bc63ac6f590573f902ad8ac7e4731022d2af9dbd4171b755793bdda4ee47d69e.jpg)  
P1AC [61] UP2P [38] UP1SIFT [62] P3.5Pf [40] P4Pf [39] UP2.5Pf [38] UP1PfAC (Our) UP2PfORI (Our)

Fig. 3: Noise sensitivity. Median error w.r.t. image noise, normal vectors, feature orientations, afine features, and gravity vector. All measurements report the median value over 1000 random test instances per noise level, where the x-axis shows the standard deviation of the added Gaussian noise. Visualization code: circular and square markers indicate upright and non-upright orientations, respectively; filled markers denote point-based solvers, while white markers indicate alternative approaches; and dashed lines highlight our contributions. Semi-calibrated approaches are in the right column. Best viewed in color.

![](images/fdb16867062f07a7b6c1b5604c2004fd782a97dcc2d2efc5e962ae080be0d2e7.jpg)  
P3.5Pf [40] P4Pf [39] UP2.5Pf [38] UP1PfAC (Our) UP2PfORI (Our)  
Fig. 4: Robust estimation. (Left): Number of inliers found vs execution time in a RANSAC framework with a 50 % outlier ratio. (Right): Outlier ratio vs total execution time to reach 80 % of the true inlier set. Best viewed in color.

and P1AC $[ 6 1 ] ^ { 4 }$ and UP1SIFT $\mathrm { [ 6 2 ] ^ { 5 } }$ are available directly from the authors repositories. The P3.5Pf [40] solver was re-implemented based on the automatic Gröbner basis solver [41] proposed by the first author.

Table 1: Median timings (in nanoseconds).
<table><tr><td>P1AC [61] UP2P [38] UP1SIFT [62]</td><td>P3.5Pf [40] P4Pf [39] UP2.5Pf [38] UP1PfAC UP2PfORI</td></tr><tr><td>2740 1448</td><td rowspan="2">19118 3179 642 2586 2149</td></tr><tr><td>484</td></tr></table>

Integration with robust estimation frameworks In this section, we integrate the solvers in a standard RANSAC framework to demonstrate how they cope with outliers. We only consider solvers where the focal length is assumed unknown, as this is the intended application of the proposed solvers. We generate 1000 correspondences in each scene and use a 1.2 pixel noise and an IMU noise of 0.2 degrees. Furthermore, outliers are injected by corrupting the data randomly. First, we consider a scenario with a 50 % outlier ratio and compare the total number of inliers found by the respective solvers as a function of time. To distinguish outliers from inliers a threshold of 5 pixels is used. Fig. 4 reports the median values over 100 random problem instances. Both of the proposed solvers display advantageous performance compared to the state-of-the-art.

Furthermore, we vary the outlier ratio and report the time it takes the corresponding solvers to find an inlier set containing at least 80 % of the true inliers. The results are shown to the right in Fig. 4. Here, the benefits of using more than point-based data become clearer, where the increasing exponential relationship is due to the minimal sample sizes of the corresponding solvers.

## 4.2 Real data experiments

We evaluate the proposed solvers in a visual localization scenario based on real data, where a query device must localize in the map created beforehand. For this, we use two well-known datasets: the Cambridge Landmarks [34] and the

Aachen Day-Night v1.1 [54]. The Cambridge dataset consists of images gathered from video while walking at five diferent scenes in Cambridge (UK). Each scene has a train and a test split, using the first for the reconstruction and the latter as localization queries. We re-triangulate the maps by using ground-truth from VisualSfM [63], where the intrinsic and extrinsic calibration is obtained for each camera. This is a relatively easy dataset because images are captured from the same device, the scene scale is medium and visual conditions are similar. We report median errors for position, rotation and focal length, as well as the median execution time. The Aachen Dataset consists of images captured at the historical center of Aachen (Germany), a larger scale environment recorded with several diferent devices, which resembles more the XR scenario. It considers changes in visual appearance, making it a more challenging benchmark, especially for semicalibrated methods. We triangulate the models based on the poses from [68], and assume all query images to be upright. We follow the evaluation protocol and report the percentage of images localized within the thresholds $( 0 . 2 5 \mathrm { { m } / 2 ^ { \circ } }$ $0 . 5 \mathrm { m } / 5 ^ { \circ } , 5 . 0 \mathrm { m } / 1 0 ^ { \circ } )$ , as well as the median focal length estimation error.

Robust estimation method We integrate our and the other solvers in the Graph-Cut-RANSAC framework [9], a locally optimized RANSAC alternating graphcut and model re-fitting in the local optimization step. For the non-minimal refinement in the case of partially calibrated absolute pose estimation, we use an initial estimate from Direct Linear Transform (DLT) from the inliers of the minimal solver, which is then refined by 10 iterations of Levenberg-Marquardt.

Visual Localization Procedure We followed the pipeline proposed by HLOC [52]. First, we triangulated the models from the available poses using COLMAP [55]. For that, we used image retrieval to recover a subset of covisible database images for each database image, using the top-20 DenseVLAD [59] retrievals. Then, we extracted and matched the features from the images in the database, using two possible configurations: RootSIFT [1]+Nearest Neighbor or Super-Point [17]+LightGlue [45]. Then, we retriangulate the model to obtain the 3D reconstructions. For localization, we retrieve the top-10 most similar database images for each query. For each of those retrieved database images, we lift the 2D-2D correspondences to 2D-3D correspondences, picking the 3D points from the SfM model related to each 2D matched keypoint in the database image. From all database images, we select the estimate with the highest number of inliers.

In the case of SIFT, the Diference-of-Gaussians (DoG) extractor provides an estimated scale and orientation. For SuperPoint,we estimate them through S3Esti [65]. These parameters can then be used to approximate the afine matrix $\mathsf { A } _ { i }$ by $\mathsf { A } _ { i } \approx \mathsf { S } _ { q _ { i } } \mathsf { R } _ { \alpha _ { i } }$ , with the diagonal matrix $\mathsf { S } _ { q , i }$ scaled uniformly by the detected scale factor $q _ { i } ,$ and the rotation matrix $\mathsf { R } _ { \alpha _ { i } } \doteq \mathrm { S O } ( 2 )$ , by the estimated angle $\alpha _ { i } \in [ 0 , 2 \pi )$ . Normals were estimated using the closest 200 nearest neighbors.

Localization results Tab. 2 and Tab. 3 report performance in Cambridge dataset using the SuperPoint and the SIFT pipelines, respectively. The results show that, for both features, our solvers improve the estimation errors committed by the other semi-calibrated solvers in all scenes, while also improving the time. Compared with the calibrated methods, our solvers improve performance in rotation estimation for all scenes but gets worse translation errors due to the focal length ambiguity. We observe a slight increase in terms of time, attributed to the diferent nonlinear refinement method used for the case of unknown focal length. Results show a better performance in general with SuperPoint, but the performance gap gets smaller when the solver is accurate enough, as in our case. Tab. 4 shows performance for both SuperPoint and SIFT setups, following a similar trend than in Cambridge. As translation and rotation errors are compacted in a single metric, results show a performance degradation in case of partially calibrated methods, with our methods being better. In general, we observed that the non-linear refinement method homogenizes the final performance of the partially calibrated solvers, the main diference lying in the time.

Table 2: Cambridge Landmarks Median errors in position, rotation and focal length, and median times for the tested solvers in GC-RANSAC, using SP+LG and S3Esti, best method highlighted diferently for calib and semi-calib methods.
<table><tr><td rowspan="2"></td><td colspan="4"> $_ \mathrm { G r e a t C o u r t }$  o</td><td colspan="4"> $\mathrm { K i n g s C o l l e g e }$ </td><td colspan="4">OldHospital</td><td colspan="4"> $\mathrm { S h o p F a c a d e }$ </td><td colspan="4">StMarysChurch</td></tr><tr><td>cm</td><td></td><td> $f _ { \mathrm { e r r } }$ </td><td>ms</td><td>cm</td><td>0</td><td>ferr</td><td>ms</td><td>cm</td><td>o</td><td>ferr</td><td>ms</td><td>cm</td><td>o</td><td>ferr</td><td></td><td>ms cm</td><td>0</td><td>ferr</td><td>ms</td></tr><tr><td>5 P1AC [61]</td><td>32.6</td><td>0.16</td><td></td><td>23.3</td><td></td><td>16.60.28</td><td></td><td>32.4</td><td></td><td>21.2 0.36</td><td></td><td>35.2</td><td>6.7</td><td>0.32</td><td></td><td>36.8</td><td></td><td>11.2 0.35</td><td></td><td>34.3</td></tr><tr><td>Known UP2P [38]</td><td>31.6</td><td>0.15</td><td></td><td>23.1</td><td></td><td>16.10.28</td><td></td><td>33.4</td><td></td><td>21.1 0.35</td><td></td><td>31.8</td><td>6.5</td><td>0.30</td><td></td><td></td><td>34.1</td><td>10.80.33</td><td></td><td>33.9</td></tr><tr><td>UP1SIFT [62]</td><td>45.9</td><td>0.20</td><td></td><td>22.5</td><td>20.4</td><td>0.32</td><td></td><td>34.0</td><td></td><td>44.5 0.64</td><td></td><td>31.2</td><td>8.4</td><td>0.38</td><td></td><td>33.5</td><td></td><td>17.0 0.55</td><td></td><td>29.1</td></tr><tr><td>P4Pf [39]</td><td>61.5</td><td>0.14</td><td>0.010</td><td>26.9</td><td>36.3</td><td>0.30 0.011</td><td></td><td>48.3</td><td>58.7</td><td>0.42</td><td>0.015</td><td>40.9</td><td>14.6</td><td>0.30</td><td>0.011</td><td>39.9</td><td>23.9</td><td>0.32</td><td>0.014</td><td>42.3</td></tr><tr><td>Unknnwwn P3.5Pf [40]</td><td>61.3</td><td>0.14</td><td>0.010</td><td>28.0</td><td>36.0</td><td>0.30 0.011</td><td></td><td>45.2</td><td></td><td>58.00.42</td><td>0.015</td><td>38.6</td><td>14.6</td><td>0.30</td><td>0.011</td><td>41.4</td><td></td><td>23.9 0.32</td><td>0.014</td><td>41.2</td></tr><tr><td>UP2.5Pf [38]</td><td>61.6</td><td>0.14</td><td>0.010</td><td>23.8</td><td>35.9</td><td>0.29</td><td>0.011</td><td>41.1</td><td></td><td>57.2 0.42</td><td>0.015</td><td>38.0</td><td></td><td>14.4 0.30</td><td>0.011</td><td>35.4</td><td></td><td>23.7 0.32</td><td>0.013</td><td>37.0</td></tr><tr><td>UP1PfAĆ</td><td></td><td>57.7 0.13 0.009 22.6</td><td></td><td></td><td></td><td>35.60.280.012 39.9</td><td></td><td></td><td></td><td>52.3 0.38</td><td>0.013 34.2</td><td></td><td></td><td></td><td>10.6 0.26 0.008 33.9</td><td></td><td></td><td>22.6 0.31</td><td>0.014</td><td>34.3</td></tr><tr><td>UP2PfORI</td><td></td><td>57.9 0.13 0.009 24.0</td><td></td><td></td><td></td><td>35.3 0.28</td><td>0.012 41.7</td><td></td><td></td><td>52.2 0.37</td><td>0.014</td><td>37.1</td><td></td><td></td><td>10.9 0.26 0.00834.3</td><td></td><td></td><td>22.9 0.31</td><td>0.014</td><td>38.2</td></tr></table>

Table 3: Cambridge Landmarks Median errors in position, rotation and focal length, and median times the tested solvers in GC-RANSAC, using SIFT and Nearest Neighbors, best method highlighted diferently for calib and semi-calib methods.
<table><tr><td rowspan="2"></td><td colspan="4">GreatCourt 0</td><td colspan="4"> $\mathrm { K i n g s C o l l e g e }$ </td><td colspan="4"> $\mathrm { O l d H o s p i t a l }$ </td><td colspan="4"> $\mathrm { S h o p F a c a d e }$  0</td><td colspan="4">StMarysChurch</td></tr><tr><td>cm</td><td></td><td> $f _ { \mathrm { e r r } }$ </td><td>ms</td><td>cm</td><td>o</td><td> $f _ { \mathrm { e r r } }$ </td><td>ms</td><td>cm</td><td>o</td><td> $f _ { \mathrm { e r r } }$ </td><td>ms</td><td>cm</td><td></td><td> $f _ { \mathrm { e r r } }$ </td><td>ms</td><td>cm</td><td>0</td><td>ferr</td><td>ms</td></tr><tr><td>1 P1AC [61]</td><td>59.5 0.21</td><td></td><td></td><td>15.2</td><td>17.80.28</td><td></td><td></td><td>34.5</td><td></td><td>26.6 0.45</td><td></td><td>20.1</td><td>8.1</td><td>0.30</td><td></td><td>21.1</td><td></td><td>14.9 0.42</td><td></td><td>22.4</td></tr><tr><td>nown UP2P [38]</td><td>55.10.19</td><td></td><td></td><td>16.0</td><td>17.4</td><td>0.28</td><td></td><td>34.8</td><td></td><td>25.1 0.42</td><td></td><td>20.5</td><td>6.8</td><td>0.28</td><td></td><td></td><td>20.8</td><td>14.8 0.40</td><td></td><td>22.5</td></tr><tr><td>UP1SIFT [62]</td><td>58.4 0.20</td><td></td><td></td><td>15.4</td><td>18.6</td><td>0.29</td><td></td><td>36.3</td><td></td><td>29.40.48</td><td></td><td>18.9</td><td>7.6</td><td>0.30</td><td></td><td>20.3</td><td></td><td>15.4 0.42</td><td></td><td>21.3</td></tr><tr><td>P4Pf [39]</td><td>75.8</td><td>0.16 0.011</td><td></td><td>23.3</td><td>37.9</td><td>0.31 0.012</td><td></td><td>41.9</td><td>51.1</td><td>0.39</td><td>0.017</td><td>25.9</td><td>11.2</td><td>0.28</td><td>0.008</td><td>23.6</td><td></td><td>28.70.37</td><td>0.018</td><td>28.4</td></tr><tr><td>Unknwwn P3.5Pf [40]</td><td>75.8</td><td>0.16 0.011</td><td></td><td>22.7</td><td></td><td>38.00.32 0.012</td><td></td><td>42.7</td><td></td><td>51.7 0.39</td><td>0.017</td><td>27.8</td><td>11.2</td><td></td><td>0.28 0.008</td><td>25.8</td><td></td><td>28.60.37</td><td>0.018</td><td>31.2</td></tr><tr><td>UP3Pf [38]</td><td>75.6</td><td>0.16 0.011 20.5</td><td></td><td></td><td></td><td>37.8 0.31 0.012 40.1</td><td></td><td></td><td></td><td>51.9 0.39</td><td></td><td>0.017 24.2</td><td>11.2</td><td></td><td>0.280.008 20.6</td><td></td><td></td><td>28.4 0.38</td><td>0.018</td><td>26.4</td></tr><tr><td>UP1PfÁC</td><td></td><td>70.5 0.150.01216.7</td><td></td><td></td><td></td><td>37.6 0.30 0.012 35.5</td><td></td><td></td><td></td><td>48.4 0.330.018</td><td></td><td>21.0</td><td></td><td></td><td>11.1 0.27 0.008 20.8</td><td></td><td></td><td>27.7 0.38</td><td>0.017</td><td>24.2</td></tr><tr><td>UP2PfORI</td><td></td><td>70.8 0.15 0.011 17.7</td><td></td><td></td><td></td><td>37.8 0.30 0.012 35.4</td><td></td><td></td><td></td><td>50.4 0.36</td><td>0.018</td><td>23.8</td><td></td><td></td><td>11.0 0.27 0.008 21.6</td><td></td><td></td><td>28.1 0.38</td><td>0.018</td><td>27.1</td></tr></table>

## 5 Discussion

In general, we observed that the 1-point UP1PfAC works better than UP2PfORI. It should be noticed that the afine constraints encode significantly more information than the orientation-covariant constraint alone. It therefore acts as a stronger geometric prior; but at a higher computational cost compared to, e.g., ORB. The proposed 2-point solver efectively bridges this gap, ofering a compelling balance between geometric expressiveness and computational eficiency.

Table 4: Aachen Dataset Performance for various solvers and features using GC-RANSAC, with percentage error recalls for thresholds 0.25m $/ 2 ^ { \circ }$ , 0.5m/5<sup>◦</sup>, 5.0m/10<sup>◦</sup> and $f _ { \mathrm { e r r } }$ . Best method highlighted diferently for calib and semi-calib methods.
<table><tr><td rowspan="3"></td><td rowspan="3"></td><td colspan="6">SuperPoint+LightGlue+S3Esti</td><td rowspan="2"></td><td colspan="7">SIFT+Nearest Neighbors</td></tr><tr><td colspan="2">Day</td><td colspan="2">Night</td><td colspan="2">ferr</td><td colspan="2">Day</td><td colspan="2">Night</td><td colspan="2">ferr</td><td rowspan="2">ms</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>ms</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>f P1AC [61] novn</td><td></td><td></td><td>69.5 89.6 99.2</td><td></td><td>70.7 88.0 99.0</td><td></td><td></td><td>15.7</td><td>59.8</td><td>76.3 87.7</td><td></td><td>|21.5 25.7 27.7</td><td></td><td></td><td>29.1</td></tr><tr><td>UP2P [38]</td><td></td><td>71.8 90.0 99.3</td><td></td><td></td><td>71.2 87.4 98.4</td><td></td><td></td><td>15.6</td><td></td><td>61.3 76.9 87.9</td><td></td><td>21.5 25.7 27.7</td><td></td><td></td><td>26.9</td></tr><tr><td>UP1SIFT [62]]</td><td></td><td></td><td>60.9 84.2 93.8</td><td></td><td>59.7 79.6 92.7</td><td></td><td></td><td>16.4</td><td></td><td>59.0 75.0 87.0</td><td></td><td>19.4 23.0 26.2</td><td></td><td></td><td>24.8</td></tr><tr><td>P4Pf [39]</td><td></td><td>46.1 70.0 96.8</td><td></td><td></td><td>|54.5 74.9 95.8|</td><td></td><td>0.010 16.7|</td><td></td><td>37.4 57.0</td><td>85.9</td><td></td><td>|19.4 20.9 26.2</td><td></td><td>0.019</td><td>37.9</td></tr><tr><td>f nnn P3.5Pf [40]</td><td></td><td>45.6 69.7 96.7</td><td></td><td></td><td>54.5 74.9 95.3</td><td></td><td>0.010 20.3</td><td></td><td>37.7 57.8 85.7</td><td></td><td></td><td>19.4 21.5 26.2</td><td></td><td>0.01834.8</td><td></td></tr><tr><td>UP2.5Pf [38]</td><td></td><td>46.5 68.8 96.6</td><td></td><td></td><td>55.0 73.8 95.3</td><td></td><td>0.01018.6</td><td></td><td>37.9 57.4 85.4</td><td></td><td></td><td>18.8 21.5 26.2</td><td></td><td></td><td>0.018 32.2</td></tr><tr><td>UP1PfAC</td><td></td><td>48.5 70.8 97.9</td><td></td><td></td><td>60.2 80.1 97.4</td><td></td><td>0.009 16.7</td><td></td><td></td><td>38.858.6 86.3</td><td></td><td>18.3 22.0 27.2</td><td></td><td></td><td>0.017 28.0</td></tr><tr><td>UP2PfORI</td><td></td><td>48.4 71.2 98.2</td><td></td><td>62.3 80.1 97.4</td><td></td><td></td><td>0.00919.8</td><td></td><td></td><td>39.7 58.4 86.3</td><td></td><td>19.4 22.5 26.7</td><td></td><td></td><td>0.017 30.8</td></tr></table>

While modern robust estimation frameworks are equipped with numerous features, it is often unclear what truly drives their performance. Historically, research emphasized the development of fast and stable minimal solvers, but this focus has recently been reassessed [18, 37]. Factors such as the number of generated solutions, the choice of nonlinear refinement, and sampling strategies may be equally important—yet the speed of the minimal solver remains critical.

Our proposed method is computationally eficient, but real-world data introduces multiple error sources, including image noise, normal noise from 3D reconstruction, descriptor mismatches, and auxiliary sensor inaccuracies (e.g., from IMUs). Given these compounded errors, the reliability of any method is a valid concern. We demonstrate that, within a modern estimation framework, strong performance is still achievable: reasonably accurate hypotheses generated speedily, combined with efective nonlinear refinement, is suficient to yield robust results. With this in mind, we believe the future of minimal solvers remains bright, provided they can continue to leverage diverse sources of information, including deep learning-based descriptors, geometric priors, and auxiliary sensor data. Integrating these cues will be key to pushing the boundaries of what is possible in robust estimation and the applications thereof.

## 6 Conclusions

In this paper, we have proposed two novel solvers for the absolute pose problem with unknown focal length. To this date, these are the first solvers available in the literature that incorporate afine- or orientation-covariant features for this problem configuration. In order to derive the solvers, novel polynomial constraints were derived and incorporated. Our results demonstrate that by utilizing more than point-based information extracted from the feature descriptors, we are able to significantly improve the execution speed in robust estimation frameworks compared to state-of-the-art methods, while maintaining superior or equal accuracy, both in synthetic environments and in localization tasks. Solvers of this kind pave the way for future generations of low-energy localization for constrained devices, e.g., XR devices, UAVs, and embedded robotics platforms.

## Acknowledgements

This publication is part of the grant JDC2024-055088-I, funded by MICIU/AEI/10.13039/501100011033 and the FSE+.

## References

1. Arandjelović, R., Zisserman, A.: Three things everyone should know to improve object retrieval. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 2911–2918 (2012)

2. Banglei, G., Su, A., Li, Z., Fraundorfer, F.: Rotational alignment of imu-camera systems with 1-point ransac. In: Pattern Recognition and Computer Vision: Second Chinese Conference (PRCV) (2019)

3. Barath, D.: Making sift features afine covariant. International Journal of Computer Vision (IJCV) 131 (2023)

4. Barath, D., Hajder, L.: Novel ways to estimate homography from local afine transformations. In: Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications. pp. 434–445 (2016)

5. Barath, D., Hajder, L.: Eficient recovery of essential matrix from two afine correspondences. IEEE Transactions on Image Processing 27, 5328–5337 (2018)

6. Barath, D., Kukelova, Z.: Homography from two orientation- and scale-covariant features. In: International Conference on Computer Vision (ICCV) (2019)

7. Barath, D., Kukelova, Z.: Relative pose from SIFT features. In: European Conference on Computer Vision (ECCV) (2022)

8. Barath, D., Matas, J.: Graph-cut RANSAC. In: Conference on Computer Vision and Pattern Recognition (2018)

9. Barath, D., Matas, J.: Graph-cut ransac. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 6733–6741 (2018)

10. Barath, D., Polic, M., Förstner, W., Sattler, T., Pajdla, T., Kukelova, Z.: Making Afine Correspondences Work in Camera Geometry Computation. In: European Conference on Computer Vision (ECCV) (2020)

11. Barath, D., Toth, T., Hajder, L.: A minimal solution for two-view focal-length estimation using two afine correspondences. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017)

12. Bay, H., Tuytelaars, T., Van Gool, L.: Surf: Speeded up robust features. In: European Conference on Computer Vision (ECCV). pp. 404–417 (2006)

13. Bentolila, J., Francos, J.M.: Conic epipolar constraints from afine correspondences. Computer Vision and Image Understanding 122, 105–114 (2014)

14. Bujnak, M., Kukelova, Z., Pajdla, T.: A general solution to the p4p problem for camera with unknown focal length. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (06 2008)

15. Campos, C., Elvira, R., Rodríguez, J.J.G., M. Montiel, J.M., D. Tardós, J.: Orbslam3: An accurate open-source library for visual, visual–inertial, and multimap slam. IEEE Transactions on Robotics 37(6), 1874–1890 (2021)

16. Chum, O., Matas, J., Kittler, J.: Locally optimized ransac. In: Pattern Recognition. pp. 236–243 (2003)

17. DeTone, D., Malisiewicz, T., Rabinovich, A.: Superpoint: Self-supervised interest point detection and description. In: IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). pp. 224–236 (2018)

18. Ding, Y., Astermark, J., Oskarsson, M., Larsson, V.: Noisy one-point homographies are surprisingly good. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5125–5134 (2024)

19. Ding, Y., Barath, D., Kukelova, Z.: Homography-based egomotion estimation using gravity and sift features. In: Asian Conference on Computer Vision (ACCV) (2020)

20. Ding, Y., Barath, D., Yang, J., Kong, H., Kukelova, Z.: Globally optimal relative pose estimation with gravity prior. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 394–403 (2021)

21. Ding, Y., Vávra, V., Bhayani, S., Wu, Q., Yang, J., Kukelova, Z.: Fundamental matrix estimation using relative depths. In: European Conference on Computer Vision (ECCV) (2024)

22. Ding, Y., Yang, J., Larsson, V., Olsson, C., Åström, K.: Revisiting the P3P problem. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 4872–4880 (2023)

23. Eichhardt, I., Chetverikov, D.: Afine correspondences between central cameras for rapid relative pose estimation. In: European Conference on Computer Vision (ECCV) (2018)

24. Fischler, M., Bolles, R.: Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM 24(6), 381–395 (1981)

25. Fragoso, V., DeGol, J., Hua, G.: gDLS\*: Generalized pose-and-scale estimation given scale and gravity priors. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 2210–2219 (2020)

26. Gao, X.S., Hou, X.R., Tang, J., Cheng, H.F.: Complete solution classification for the perspective-three-point problem. IEEE Transactions on Pattern Analysis and Machine Intelligence 25(8), 930–943 (2003)

27. Guan, B., Zhao, J., Barath, D., Fraundorfer, F.: Minimal cases for computing the generalized relative pose using afine correspondences. In: IEEE/CVF International Conference on Computer Vision (ICCV). pp. 6068–6077 (2021)

28. Guan, B., Zhao, J., Li, Z., Sun, F., Fraundorfer, F.: Minimal solutions for relative pose with a single afine correspondence. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2020)

29. Hajder, L., Barath, D.: Relative planar motion for vehicle-mounted cameras from a single afine correspondence. In: IEEE International Conference on Robotics and Automation (ICRA). pp. 8651–8657 (2020)

30. Hartley, R.I., Trumpf, J., Dai, Y., Li, H.: Rotation averaging. International Journal of Computer Vision 103, 267 – 305 (2013)

31. Hesch, J.A., Roumeliotis, S.I.: A direct least-squares (DLS) method for PnP. In: International Conference on Computer Vision (ICCV). pp. 383–390 (2011)

32. Hruby, P., Pollefeys, M., Barath, D.: Semicalibrated relative pose from an afine correspondence and monodepth. In: European Conference on Computer Vision (ECCV) (2024)

33. Ke, T., Roumeliotis, S.I.: An eficient algebraic solution to the perspective-threepoint problem. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 4618–4626 (2017)

34. Kendall, A., Grimes, M., Cipolla, R.: PoseNet: A convolutional network for realtime 6-DoF camera relocalization. In: IEEE International Conference on Computer Vision (ICCV). pp. 2938–2946 (2015)

35. Kim, S., Pollefeys, M., Barath, D.: Learning to make keypoints sub-pixel accurate. In: European Conference on Computer Vision (ECCV) (2024)

36. Klein, G., Murray, D.: Parallel tracking and mapping for small ar workspaces. In: IEEE and ACM International Symposium on Mixed and Augmented Reality. pp. 225–234 (2007)

37. Kocur, V., Tzamos, C., Ding, Y., Haladova, Z.B., Sattler, T., Kukelova, Z.: Are minimal radial distortion solvers really necessary for relative pose estimation? International Journal of Computer Vision 134(2), 48 (2026)

38. Kukelova, Z., Bujnak, M., Pajdla, T.: Closed-form solutions to minimal absolute pose problems with known vertical direction. In: Asian Conference on Computer Vision (ACCV). pp. 216–229 (2010)

39. Kukelova, Z., Bujnak, M., Pajdla, T.: Eficient intersection of three quadrics and applications in computer vision. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1799–1808 (2016)

40. Larsson, V., Kukelova, Z., Zheng, Y.: Making Minimal Solvers for Absolute Pose Estimation Compact and Robust. In: International Conference on Computer Vision (ICCV). pp. 2335–2343 (2017)

41. Larsson, V., Astrom, K., Oskarsson, M.: Eficient solvers for minimal problems by syzygy-based reduction. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017)

42. Larsson, V., contributors: PoseLib - Minimal Solvers for Camera Pose Estimation (2020), https://github.com/vlarsson/PoseLib

43. Larsson, V., Kukelova, Z., Zheng, Y.: Camera pose estimation with unknown principal point. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2018)

44. Li, H., Chen, W., Zhao, J., Bazin, J.C., Luo, L., Liu, Z., Liu, Y.H.: Robust and eficient estimation of absolute camera pose for monocular visual odometry. In: IEEE International Conference on Robotics and Automation (ICRA). pp. 2675– 2681 (2020)

45. Lindenberger, P., Sarlin, P.E., Pollefeys, M.: Lightglue: Local feature matching at light speed. In: IEEE/CVF International Conference on Computer Vision (ICCV). pp. 17627–17638 (2023)

46. Lowe, D.G.: Object recognition from local scale-invariant features. In: IEEE International Conference on Computer Vision (CVPR). vol. 2, pp. 1150–1157 (1999)

47. Mishkin, D., Radenović, F., Matas, J.: Repeatability is not enough: Learning afine regions via discriminability. In: Ferrari, V., Hebert, M., Sminchisescu, C., Weiss, Y. (eds.) European Conference on Computer Vision (ECCV). pp. 287–304 (2018)

48. Morel, J.M., Yu, G.: ASIFT: A new framework for fully afine invariant image comparison. SIAM Journal on Imaging Sciences 2, 438–469 (2009)

49. Persson, M., Nordberg, K.: Lambda twist: An accurate fast robust perspective three point (p3p) solver. In: European Conference on Computer Vision (ECCV) (2018)

50. Raposo, C., Barreto, J.P.: Theory and practice of structure-from-motion using afine correspondences. In: Conference on Computer Vision and Pattern Recognition (CVPR) (2016)

51. Rublee, E., Rabaud, V., Konolige, K., Bradski, G.: ORB: An eficient alternative to SIFT or SURF. In: International Conference on Computer Vision (ICCV). pp. 2564–2571 (2011)

52. Sarlin, P.E., Cadena, C., Siegwart, R., Dymczyk, M.: From coarse to fine: Robust hierarchical localization at large scale. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2019)

53. Sattler, T., Leibe, B., Kobbelt, L.: Eficient & efective prioritized matching for large-scale image-based localization. IEEE Trans. Pattern Anal. Mach. Intell. 39(9), 1744–1756 (Sep 2017)

54. Sattler, T., Maddern, W., Toft, C., Torii, A., Hammarstrand, L., Stenborg, E., Safari, D., Okutomi, M., Pollefeys, M., Sivic, J., Kahl, F., Pajdla, T.: Benchmarking 6DOF outdoor visual localization in changing conditions. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 8601–8610 (2018)

55. Schönberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 4104–4113 (2016)

56. Snavely, N., Seitz, S.M., Szeliski, R.: Photo tourism: exploring photo collections in 3d. ACM Trans. Graph. 25(3), 835–846 (Jul 2006)

57. Suwanwimolkul, S., Komorita, S., Tasaka, K.: Learning of low-level feature keypoints for accurate and robust detection. In: IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 2262–2271 (2021)

58. Sweeney, C., Flynn, J., Nuernberger, B., Turk, M., Höllerer, T.: Eficient computation of absolute pose for gravity-aware augmented reality. In: IEEE International Symposium on Mixed and Augmented Reality. pp. 19–24 (2015)

59. Torii, A., Arandjelovic, R., Sivic, J., Okutomi, M., Pajdla, T.: 24/7 place recognition by view synthesis. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1808–1817 (2015)

60. Valtonen Örnhag, M., Jaenal, A.: Leveraging scale- and orientation-covariant features for planar motion estimation. In: European Conference on Computer Vision (ECCV) (2024)

61. Ventura, J., Kukelova, Z., Sattler, T., Baráth, D.: P1AC: Revisiting absolute pose from a single afine correspondence. In: IEEE/CVF International Conference on Computer Vision (ICCV). pp. 19751–19761 (2023)

62. Ventura, J., Kukelova, Z., Sattler, T., Baráth, D.: Absolute pose from one or two scaled and oriented features. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 20870–20880 (2024)

63. Wu, C.: Towards linear-time incremental structure from motion. In: International Conference on 3D Vision (3DV). pp. 127–134. IEEE (2013)

64. Wu, C.: P3.5P: Pose estimation with unknown focal length. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 2440–2448 (2015)

65. Yan, P., Tan, Y., Xiong, S., Tai, Y., Li, Y.: Learning soft estimator of keypoint scale and orientation with probabilistic covariant loss. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 19406–19415 (2022)

66. Yu, Z., Guan, B., Liang, S., Li, Z., Ye, S., Yu, Q.: Globally optimal relative pose estimation using afine correspondences with known vertical direction. IEEE Transactions on Instrumentation and Measurement 72, 1–12 (2023)

67. Yu, Z., Ye, S., Jin, R., Liang, S., Liu, Z., Zhang, H., Guan, B.: A minimal solver for relative pose estimation with unknown focal length from two afine correspondences. IEEE Robotics and Automation Letters 11(2), 2290–2297 (2026)

68. Zhang, Z., Sattler, T., Scaramuzza, D.: Reference pose generation for long-term visual localization via learned features and view synthesis. International Journal of Computer Vision (IJCV) 129(4), 821–844 (2021)

69. Zheng, Y., Sugimoto, S., Sato, I., Okutomi, M.: A general and simple method for camera pose and focal length determination. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 430–437 (2014)

70. Zhou, B., Chen, Z., Liu, Q.: An eficient solution to the perspective-n-point problem for camera with unknown focal length. IEEE Access 8, 162838–162846 (2020)

## 7 Supplementary material

The full intrinsic matrix can be written as

$$
\mathsf { K } = \left[ \begin{array} { l l l } { f _ { x } } & { \gamma } & { u _ { 0 } } \\ { 0 } & { f _ { y } } & { v _ { 0 } } \\ { 0 } & { 0 } & { 1 } \end{array} \right] ,\tag{22}
$$

where the focal length is separated in two components, $f _ { x }$ and $f _ { y } .$ The aspect ratio $\alpha = f _ { y } / f _ { x } \approx 1$ , but may difer due to a variety of optical and digital techniques, such as using anamorphic formats, flaws in the sensor manufacturing or digital post-processing. The skew parameter $\gamma$ corrects for nonrectangular pixels, which also stem from imperfect sensor manufacturing. The principal point $( u _ { 0 } , v _ { 0 } )$ is the intersection point of the light ray orthogonal to the image plane and passing through the camera center.

In the main paper, we assumed that the coordinate system is normalized such that the principal point is (0, 0), which is often a good approximation if you align it with the center of the image coordinate system. Furthermore, $\alpha = 1$ and $\gamma = 0$ , which is a good approximation for modern cameras. Another reason for reducing the number of unknowns is that the solver becomes faster, which is essential for real-time computations. Nonlinear refinement can be added in later stages to account for model imperfections.

## 7.1 Modified constraints

We ignore the skew and aspect ratio, i.e. we assume $\alpha = 1$ and $\gamma = 0 .$ , and modify the governing constraints to account for the full model (33). The afine features are independent of the principal point, and we get

$$
\frac { a _ { 1 } } { f } = \frac { d } { m } ( b r _ { 1 1 } - p _ { \mathrm { r e f _ { 1 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } - \frac { p _ { \mathrm { q u e r y _ { 1 } } } - u _ { 0 } } { f } ( b r _ { 3 1 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } ) ) ,\tag{23}
$$

$$
\frac { a _ { 2 } } { f } = \frac { d } { m } ( b r _ { 1 2 } - p _ { \mathrm { r e f _ { 1 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } - \frac { p _ { \mathrm { q u e r y _ { 1 } } } - u _ { 0 } } { f } ( b r _ { 3 2 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } ) ) ,\tag{24}
$$

$$
\frac { a _ { 3 } } { f } = \frac { d } { m } ( b r _ { 2 1 } - p _ { \mathrm { r e f _ { 2 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } - \frac { p _ { \mathrm { q u e r y _ { 2 } } } - v _ { 0 } } { f } ( b r _ { 3 1 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } ) ) ,\tag{25}
$$

$$
\frac { a _ { 4 } } { f } = \frac { d } { m } ( b r _ { 2 2 } - p _ { \mathrm { r e f _ { 2 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } - \frac { p _ { \mathrm { q u e r y _ { 2 } } } - v _ { 0 } } { f } ( b r _ { 3 2 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } ) ) ~ .\tag{26}
$$

By inserting (34)–(37) into $( 1 ) - ( 2 )$ , the following constraints are obtained for scale- and orientation covariant features

$$
\begin{array} { r l } & { d c _ { \mathrm { r e f } } \big ( b r _ { 1 1 } - p _ { \mathrm { r e f _ { 1 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } - \frac { p _ { \mathrm { q u e r y _ { 1 } } } - u _ { 0 } } { f } \big ( b r _ { 3 1 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } \big ) \big ) } \\ & { ~ - \frac { } { f } { d s _ { \mathrm { r e f } } } \big ( b r _ { 2 1 } - p _ { \mathrm { r e f _ { 1 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } - \frac { p _ { \mathrm { q u e r y _ { 1 } } } - u _ { 0 } } { f } \big ( b r _ { 3 2 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } \big ) \big ) } \\ & { ~ - \frac { m q c _ { \mathrm { q u e r y } } } { f } = 0 , } \end{array}\tag{27}
$$

$$
\begin{array} { r l } & { d c _ { \mathrm { r e f } } \big ( b r _ { 2 1 } - p _ { \mathrm { r e f _ { 2 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } - \frac { p _ { \mathrm { q u e r y _ { 2 } } } - v _ { 0 } } { f } \big ( b r _ { 3 1 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } \big ) \big ) } \\ & { + d s _ { \mathrm { r e f } } \big ( b r _ { 2 1 } - p _ { \mathrm { r e f _ { 2 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } - \frac { p _ { \mathrm { q u e r y _ { 2 } } } - v _ { 0 } } { f } \big ( b r _ { 3 2 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } \big ) \big ) } \\ & { - \frac { m q s _ { \mathrm { q u e r y } } } { f } = 0 . } \end{array}\tag{28}
$$

Subtracting (27) from (28), one obtains an orientation-covariant constraint.

$$
\begin{array} { r l } & { c _ { \mathrm { r e f } } s _ { \mathrm { q u e r y } } \big ( b r _ { 1 1 } - p _ { \mathrm { r e f _ { 1 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } - \frac { p _ { \mathrm { q u e r y } _ { 1 } } - u _ { 0 } } { f } ( b r _ { 3 1 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } ) \big ) + } \\ & { s _ { \mathrm { r e f _ { 3 } } } ^ { \mathrm { e } } s _ { \mathrm { q u e r y } } \big ( b r _ { 1 2 } - p _ { \mathrm { r e f _ { 1 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } - \frac { p _ { \mathrm { q u e r y } _ { 1 } } - u _ { 0 } } { f } ( b r _ { 3 2 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } ) \big ) - } \\ & { c _ { \mathrm { r e f _ { 4 } } } c _ { \mathrm { q u e r y } } \big ( b r _ { 2 1 } - p _ { \mathrm { r e f _ { 2 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } - \frac { p _ { \mathrm { q u e r y } _ { 2 } } - v _ { 0 } } { f } ( b r _ { 3 1 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } ) \big ) - } \\ & { s _ { \mathrm { r e f } } c _ { \mathrm { q u e r y } } \big ( b r _ { 2 2 } - p _ { \mathrm { r e f _ { 2 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } - \frac { p _ { \mathrm { q u e r y } _ { 2 } } - v _ { 0 } } { f } ( b r _ { 3 2 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } ) \big ) = 0 , } \end{array}\tag{29}
$$

Lastly, the principal point afects the point-based constraints

$$
\frac { p _ { \mathrm { q u e r y } _ { 1 } } - u _ { 0 } } { f } ( d \mathsf { R } _ { 3 , : } \tilde { \mathbf { p } } _ { \mathrm { r e f } } + t _ { 3 } ) - ( d \mathsf { R } _ { 1 , : } \tilde { \mathbf { p } } _ { \mathrm { r e f } } + t _ { 1 } ) = 0 ,\tag{30}
$$

$$
\frac { p _ { \mathrm { q u e r y } _ { 2 } } - v _ { 0 } } { f } ( d \mathsf { R } _ { 3 , : } \tilde { \mathbf { p } } _ { \mathrm { r e f } } + t _ { 3 } ) - ( d \mathsf { R } _ { 2 , : } \tilde { \mathbf { p } } _ { \mathrm { r e f } } + t _ { 2 } ) = 0 \ .\tag{31}
$$

These modified constraints need to be multiplied by an appropriate factor to become a polynomial.

## 7.2 Comparison to state-of-the-art

Larsson et al. [43] proposed two solvers including unknown focal length and unknown principal point with and without non-unit aspect ratio, called P4.5Pfuv and P5Pfuva, respectively. These point-based solvers require five 2D-3D correspondences each and do not use the gravity direction.

In our case, we have seven (or eight) degrees of freedom: four extrinsic (1 rotation and 3 translation) and three intrinsic: $f ,$ u<sub>0</sub>, v<sub>0</sub> (possible with $f _ { x }$ and $f _ { y }$ with unknown aspect ratio). For the 7 DoF problem we require two afine features (UP2PfuvAC), two scale- and orientation-covariant features (UP2PfuvSIFT), or three orientation-covariant features (UP3PfuvORI), to obtain a solution. This reasoning generalizes to the 8 DoF case with non-unit aspect ratio.

## 7.3 Analysis of solutions

We analyze the feasibility of solving the above problems using the computational algebraic geometry tool Macaulay2, which helps us understand if the polynomial system of equations that arise from the solver configuration admits a solution, and if so, how many solutions there are. The results are shown in Tab. 6. A Gröbner basis solver, e.g., [40], or resultant-based method can be used to find these solutions.

Table 5: Potential solvers. Template size when using [40].
<table><tr><td>Solver</td><td>Upright DOF Eqs Minimal Solutions Template size</td><td></td><td></td><td></td><td></td></tr><tr><td>UP2PfuvAC</td><td>√</td><td>7</td><td>12</td><td>6</td><td> $2 7 3 \times 2 7 9$ </td></tr><tr><td>UP2PfuvSIFT</td><td>√</td><td>7</td><td>8</td><td>6</td><td> $2 8 4 \times 2 9 0$ </td></tr><tr><td>UP3PfuvORI</td><td>√</td><td>7 9</td><td></td><td>6</td><td> $3 4 3 \times 3 4 9$ </td></tr><tr><td>UP1PfaAC</td><td>√</td><td>6</td><td>6 √</td><td>TBD</td><td>TBD</td></tr><tr><td>UP2PfaORI</td><td>√</td><td>6</td><td>6 √</td><td>TBD</td><td>TBD</td></tr><tr><td>UP2PfuvaAC</td><td>√</td><td>8</td><td>12</td><td>TBD</td><td>TBD</td></tr><tr><td>UP2PfuvaSIFT</td><td>√</td><td>8</td><td>8 √</td><td>TBD</td><td>TBD</td></tr><tr><td>UP3PfuvaORI</td><td>√</td><td>8</td><td>9</td><td>TBD</td><td>TBD</td></tr></table>

## 7.4 Implementation details

While the Gröbner basis solver [40] can find a solution, it is rarely useful without more intervention. For all cases described above, the equations are linear in $\mathbf { t } _ { \mathrm { q u e r y } } ,$ , hence the first step used in the main paper can be further applied. $E . g$ ., in all cases: UP2PfuvAC, UP2PfuvSIFT, and UP3PfuvORI, we may write the complete system of equations in a compact form

$$
\mathsf { M } ( r , f , u _ { 0 } , v _ { 0 } ) \left[ \mathsf { \frac { t _ { \mathrm { q u e r y } } } { \tau } } \right] = \mathbf { 0 } ,\tag{32}
$$

where M is a $7 \times 4$ matrix, where the coeficients of $m _ { i j }$ vary depending on the solver. Again, all subdeterminants of size $4 \times 4$ must vanish, but many will be trivially zero. In some cases, it might be beneficial to consider a solution similar to that of UP2PfORI (in the main paper), where only a subset of the parameters are solved for initially. The added terms $u _ { 0 }$ and v<sub>0</sub> are linear, therefore a similar approach to that of the main paper should be possible. This will generate a system of equations in fewer unknowns, upon which a Gröbner basis solver (or similar) can be applied, followed by back-substitution to obtain the remaining unknowns.

## 8 Supplementary material

The full intrinsic matrix can be written as

$$
\mathsf { K } = \left[ \begin{array} { l l l } { f _ { x } } & { \gamma } & { u _ { 0 } } \\ { 0 } & { f _ { y } } & { v _ { 0 } } \\ { 0 } & { 0 } & { 1 } \end{array} \right] ,\tag{33}
$$

where the focal length is separated in two components, $f _ { x }$ and $f _ { y }$ . The aspect ratio $\alpha = f _ { y } / f _ { x } \approx 1$ , but may difer due to a variety of optical and digital techniques, such as using anamorphic formats, flaws in t he sensor manufacturing or digital post-processing. The skew parameter $\gamma$ corrects for nonrectangular pixels, which also stem from imperfect sensor manufacturing. The principal point $( u _ { 0 } , v _ { 0 } )$ is the intersection point of the light ray orthogonal to the image plane and passing through the camera center.

In the main paper, we assumed that the coordinate system is normalized such that the principal point is (0, 0), which is often a good approximation if you align it with the center of the image coordinate system. Furthermore, $\alpha = 1$ and $\gamma = 0$ , which is a good approximation for modern cameras. Another reason for reducing the number of unknowns is that the solver becomes faster, which is essential for real-time computations. Nonlinear refinement can be added in later stages to account for model imperfections.

## 8.1 Modified constraints

We ignore the skew, and assume $\gamma = 0$ , and modify the governing constraints to account for the full model (33). The afine features are independent of the principal point, and we get

$$
\frac { a _ { 1 } } { f _ { x } } = \frac { d } { m } \big ( b r _ { 1 1 } - p _ { \mathrm { r e f _ { 1 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } - \frac { p _ { \mathrm { q u e r y _ { 1 } } } } { f _ { x } } \big ( b r _ { 3 1 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } \big ) \big ) ,\tag{34}
$$

$$
\frac { a _ { 2 } } { f _ { y } } = \frac { d } { m } ( b r _ { 1 2 } - p _ { \mathrm { r e f _ { 1 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } - \frac { p _ { \mathrm { q u e r y _ { 1 } } } } { f _ { x } } ( b r _ { 3 2 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } ) ) ,\tag{35}
$$

$$
\frac { a _ { 3 } } { f _ { x } } = \frac { d } { m } ( b r _ { 2 1 } - p _ { \mathrm { r e f _ { 2 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } - \frac { p _ { \mathrm { q u e r y _ { 2 } } } } { f _ { y } } ( b r _ { 3 1 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } ) ) ,\tag{36}
$$

$$
\frac { a _ { 4 } } { f _ { y } } = \frac { d } { m } ( b r _ { 2 2 } - p _ { \mathrm { r e f _ { 2 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } - \frac { p _ { \mathrm { q u e r y _ { 2 } } } } { f _ { y } } ( b r _ { 3 2 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 2 } } } ) ) .\tag{37}
$$

Furthermore, for orientation-covariant features, the following must hold:

$$
c _ { \mathrm { r e f } } s _ { \mathrm { q u e r y } } ( b r _ { 1 1 } - p _ { \mathrm { r e f } _ { 1 } } ^ { \prime } n _ { \mathrm { r e f } _ { 1 } } - \frac { p _ { \mathrm { q u e r y } _ { 1 } } } { f _ { x } } ( b r _ { 3 1 } - p _ { \mathrm { r e f } _ { 3 } } ^ { \prime } n _ { \mathrm { r e f } _ { 1 } } ) ) +
$$

$$
s _ { \mathrm { r e f } } s _ { \mathrm { q u e r y } } ( b r _ { 1 2 } - p _ { \mathrm { r e f } _ { 1 } } ^ { \prime } n _ { \mathrm { r e f } _ { 2 } } - \frac { p _ { \mathrm { q u e r y } _ { 1 } } } { f _ { y } } ( b r _ { 3 2 } - p _ { \mathrm { r e f } _ { 3 } } ^ { \prime } n _ { \mathrm { r e f } _ { 2 } } ) ) -
$$

$$
c _ { \mathrm { r e f } } c _ { \mathrm { q u e r y } } ( b r _ { 2 1 } - p _ { \mathrm { r e f _ { 2 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } - \frac { p _ { \mathrm { q u e r y _ { 2 } } } } { f _ { x } } ( b r _ { 3 1 } - p _ { \mathrm { r e f _ { 3 } } } ^ { \prime } n _ { \mathrm { r e f _ { 1 } } } ) ) -\tag{38}
$$

$$
s _ { \mathrm { r e f } } c _ { \mathrm { q u e r y } } ( b r _ { 2 2 } - p _ { \mathrm { r e f } _ { 2 } } ^ { \prime } n _ { \mathrm { r e f } _ { 2 } } - \frac { p _ { \mathrm { q u e r y } _ { 2 } } } { f _ { y } } ( b r _ { 3 2 } - p _ { \mathrm { r e f } _ { 3 } } ^ { \prime } n _ { \mathrm { r e f } _ { 2 } } ) ) = 0 ,
$$

The principal point afects the point-based constraints

$$
\frac { p _ { \mathrm { q u e r y } _ { 1 } } - u _ { 0 } } { f _ { x } } ( d \mathsf { R } _ { 3 , : } \tilde { \mathbf { p } } _ { \mathrm { r e f } } + t _ { 3 } ) - ( d \mathsf { R } _ { 1 , : } \tilde { \mathbf { p } } _ { \mathrm { r e f } } + t _ { 1 } ) = 0 ,\tag{39}
$$

$$
\frac { p _ { \mathrm { q u e r y } _ { 2 } } - v _ { 0 } } { f _ { y } } ( d \mathsf { R } _ { 3 , : } \tilde { \mathbf { p } } _ { \mathrm { r e f } } + t _ { 3 } ) - ( d \mathsf { R } _ { 2 , : } \tilde { \mathbf { p } } _ { \mathrm { r e f } } + t _ { 2 } ) = 0 \ .\tag{40}
$$

These modified constraints need to by multiplied appropriately to become polynomial.

## 8.2 Special cases

Non-unit aspect ratio A common case, also used by P4Pf [39], is that we separate $f _ { x }$ and $f _ { y } .$ In this case, we have six degrees of freedom, and six equations; hence, it is possible to use a single afine feature to solve the problem (UP1PfaAC). Similarly, we could use two orientation-covariant features (UP2PfaORI), which also yields six equations, hence is minimal.

Full model In this case, we have eight degrees of freedom, and has also been considered by Larsson et al. (P4.5Pfuv and P5Pfuva) for point-based solvers and without gravity direction, for which the minimal configuration requires five point correspondences. In our case, one would require two afine features (12 equations), however, one could also use only (2+6) constraints, i.e. the pointbased and one afine feature (UP2PfuvaAC). A minimal case consists of using two scale- and orientation-covariant features (UP2PfuvaSIFT) as each feature yields four equations. Alternatively, three orientation-covariant features could be used (UP3PfuvaORI).

No IMU data We could apply this to the case of no known gravity direction as well (unclear if it is interesting for us, and protectable)

## 8.3 Analysis of solutions

We analyze the feasibility of solving the above problems using the computational algebraic geometry tool Macaulay2, which helps us understand if the polynomial systems of equations that arise from the solver configuration admits a solution, and if so, how many solutions there are, the results are shown in Tab. 6.

## 8.4 Implementation details

For all cases described above, the equations are linear in the translation vector, hence the first step used in the main paper can be further applied. All added terms are linear, hence standard procedure should apply, but at an added computational cost, compared to the proposed solvers in the main paper. A Gröbner basis solver [40] can be used for the reduced system of equations and back-substitution be applied to obtain the complete solution.

Table 6: Potential solvers.
<table><tr><td colspan="5">Solver Upright DOF Eqs Minimal Solutions</td></tr><tr><td>UP1PfaAC</td><td>√</td><td>6</td><td>6 √</td><td>TBD</td></tr><tr><td>UP2PfaORI</td><td>√</td><td>6 6</td><td>√</td><td>TBD</td></tr><tr><td>UP2PfuvaAC</td><td>√ 8</td><td>12</td><td></td><td>TBD</td></tr><tr><td>UP2PfuvaSIFT</td><td>√</td><td>8 8</td><td>√</td><td>TBD</td></tr><tr><td>UP3PfuvaORI</td><td>√ 8</td><td>9</td><td></td><td>TBD</td></tr><tr><td>P2PfuvaAC</td><td></td><td>10 12</td><td></td><td>TBD</td></tr><tr><td>P2PfaSIFT</td><td></td><td>8 8</td><td>√</td><td>TBD</td></tr><tr><td>P3PfuvORI</td><td>9</td><td>9</td><td>√</td><td>TBD</td></tr><tr><td>P3PfuvaSIFT</td><td></td><td>10 12</td><td></td><td>TBD</td></tr><tr><td>P4PfuvaORI</td><td>10</td><td>12</td><td></td><td>TBD</td></tr></table>