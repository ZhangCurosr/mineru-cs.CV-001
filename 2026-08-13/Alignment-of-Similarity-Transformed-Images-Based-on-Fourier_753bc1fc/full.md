# Alignment of Similarity-Transformed Images Based on Fourier–Mellin Transform Using Auxiliary Function Method

Shinji Yamashita<sup>∗</sup>, Yuma Kinoshita<sup>∗</sup>, and Hitoshi Kiya<sup>†</sup>

∗ Tokai University, Japan

E-mail: {5CEIM050, ykinoshita}@tokai.ac.jp

<sup>†</sup> Tokyo Metropolitan University, Japan

E-mail: kiya@tmu.ac.jp

Abstract—This paper proposes an algorithm for estimating the similarity transformation, namely translation, scale, and rotation, between two images with subpixel accuracy. Image registration is a fundamental technique for aligning images acquired under different viewpoints and imaging conditions, and a representative approach based on maximizing discrete cross-correlation is the Fourier–Mellin registration. However, the Fourier–Mellin approach often fails to achieve sufficient alignment accuracy when subpixel-level estimation is required. The proposed method integrates (i) scale-and-rotation estimation from the Fourier magnitude spectrum in a log-polar representation and (ii) maximization of phase-only correlation based on the auxiliary function method. This integration enables a two-stage estimation procedure: it first estimates scale and rotation without being affected by translation, and then estimates translation with subpixel precision in the spatial domain using the corrected image pair. A simulation experiment on image pairs subjected to random similarity transformations demonstrates that the proposed method reduces estimation errors in scale, rotation, and translation compared with Fourier–Mellin-based registration methods using discrete cross-correlation.

## I. INTRODUCTION

Image alignment (registration) is a technique for aligning images by estimating the geometric transformation between them and compensating for it so that corresponding pixels (or feature points) coincide in a common coordinate system. This paper focuses on similarity transformations, a class of geometric transformations consisting of translation, rotation, and scaling. Image alignment is indispensable in a wide range of applications such as medical image processing [1], panorama generation [2], 3D reconstruction [3], and high dynamic range (HDR) imaging [4], [5]. Recently, there has been increasing demand for fast and highly accurate alignment on resourceconstrained devices (e.g., smartphones), which highlights the importance of computationally efficient algorithms.

Image alignment methods can be broadly categorized into homography-based and intensity-based approaches. In homography-based methods, correspondences are extracted using feature descriptors such as SIFT [6], SURF [7], and A-KAZE [8], and a homography between two images is typically estimated after outlier rejection via the RANSAC algorithm [9]. Recently, deep-learning-based homography estimation has also been investigated [10]. In contrast, intensitybased methods widely employ direct optimization frameworks that estimate transformation parameters by directly minimizing pixel-wise errors, as well as optical-flow-based formulations [11]–[14]. However, both homography estimation and optical flow estimation methods can impose substantial computational burdens in resource-limited environments.

To address the computational cost issue, two-dimensional cross-correlation-based methods [5], [15], [16] have been widely used for real-time image alignment. These methods estimate translational displacement by maximizing the twodimensional cross-correlation function between two images. Cross-correlation for discrete pixel shifts can be computed efficiently using the fast Fourier transform (FFT). Furthermore, Kinoshita et al. proposed a computationally efficient algorithm that estimates the peak position of the cross-correlation function with subpixel accuracy based on an optimization algorithm known as the auxiliary function method or the majorization– minimization (MM) method [17]. However, these conventional methods assume that the displacement between images is purely translational and thus estimate only translation parameters. In practical imaging scenarios, zooming and camera rotation often cause misalignment that includes scaling and rotation, for which a simple translation model is insufficient.

Therefore, there is a need for a highly accurate alignment algorithm that can handle similarity transformations while keeping computational cost low. As a representative method that incorporates scaling and rotation, a Fourier– Mellin-transform-based registration is known [18]. It exploits two properties. First, the magnitude spectrum is invariant to translation. Second, scaling and rotation in the spatial domain correspond to translations in the log-polar domain. Using these properties, it estimates translation, rotation, and scale through two rounds of discrete cross-correlation maximization. However, because it also involves resampling due to the logpolar transform, maximizing discrete cross-correlation suffers from limited alignment accuracy.

To overcome this issue, this paper proposes a computationally efficient method for estimating all similarity transformation parameters with subpixel accuracy. Specifically, we apply the high-precision correlation maximization algorithm based on the auxiliary function method [17] to both (i) scaleand-rotation estimation based on correlation between magnitude spectra in the log-polar domain and (ii) translation estimation based on correlation between images in the spatial domain within the Fourier–Mellin framework. This enables high-precision estimation of scale and rotation, followed by subpixel-accurate translation estimation on the corrected image pair.

A simulation experiment using five image pairs generated from each of three original images by applying random similarity transformations showed that the proposed method consistently reduced the absolute errors in scale, rotation, and translation compared with a Fourier–Mellin-based registration method using discrete cross-correlation.

## II. PRELIMINARIES

This section first formulates the alignment problem for similarity-transformed images. It then reviews maximization of the two-dimensional cross-correlation function based on the auxiliary function method, which serves as a building block of the proposed approach.

## A. Problem formulation

Let $x [ p ]$ and $y [ p ]$ denote the pixel values of two-dimensional discrete signals x and $y$ at pixel $\pmb { p } = ( p _ { 1 } , p _ { 2 } ) ^ { \top }$ , respectively, where $\cdot ^ { \intercal }$ denotes transpose. We consider the case where the misalignment between x and $y$ is represented by a combination of translation, rotation, and scaling, i.e., $y$ is a similaritytransformed version of $x .$ Their relationship is given by

$$
y [ p ] = x [ s R ( \theta ) p + \Delta p ] .\tag{1}
$$

Here, $\Delta p \ \in \ \mathbb { R } ^ { 2 }$ denotes translation, $R ( \theta ) ~ \in ~ \mathbb { R } ^ { 2 \times 2 }$ is the rotation matrix

$$
\begin{array}{c} R ( \theta ) = { { \binom { \cos \theta } { \sin \theta } } } \quad - \sin \theta  \\ { R ( \theta ) = { { \binom { \cos \theta } { \cos \theta } } } , \quad \cos \theta } \end{array} \quad\tag{2}
$$

and $s \in \mathbb { R } ^ { + }$ denotes the scale factor. The goal is to estimate the unknown similarity transformation parameters $s , \theta ,$ , and $\Delta p$ from an image pair $( x , y )$ , and to obtain the compensated image $y ^ { \prime \prime } [ p ] = { \bar { y } } \big [ { \textstyle { \frac { 1 } { s } } } { \bar { R } } ^ { - 1 } ( \theta ) ( { \bar { p } } - \Delta p ) \big ]$

## B. Maximization of two-dimensional cross-correlation

In [17], Kinoshita et al. proposed an algorithm for estimating the translation parameter $\Delta p$ by maximizing a generalized two-dimensional cross-correlation function, focusing on the case where the displacement is purely translational $( \mathrm { i } . \mathrm { e } . , s = 1 $ and $\theta = 0 )$ . Their method first introduces a generalized twodimensional cross-correlation function defined in the frequency domain as the objective function and then iteratively solves the resulting continuous maximization problem using the auxiliary function method.

Let xˆ and yˆ be the $N \times M$ two-dimensional discrete Fourier transforms (DFTs) of x and y, respectively, and let the angular frequency vector be $\omega _ { k l } = ( \omega _ { k } , \omega _ { l } ) ^ { \top } = ( 2 \pi k / N , ~ 2 \pi l / M ) ^ { \top }$ Define the two-dimensional cross spectrum as $\hat { \Phi } _ { 2 } ^ { ( x y ) } ( \stackrel { \cdot } { \omega } _ { k l } ) =$ $\hat { x } ^ { * } ( \omega _ { k l } ) \hat { y } ( \omega _ { k l } )$ . Then, the generalized two-dimensional crosscorrelation function is given by

$$
\check { \Phi } _ { 2 } ^ { ( x y ) } ( p ) = \frac { 1 } { N M } \sum _ { k \in K } \sum _ { l \in L } w _ { k l } \hat { \Phi } _ { 2 } ^ { ( x y ) } ( { \omega _ { k l } } ) \exp \left( j { \omega _ { k l } } ^ { \top } p \right)\tag{3}
$$

where the summation ranges are $K = \{ - N / 2 + 1 , - N / 2 +$ $2 , \ldots , N / 2 \}$ and ${ \cal L } = \{ - M / 2 + 1 , - \dot { M } / 2 + 2 , \ldots , M / 2 \}$ The weights w $\in \mathbb { R } ^ { + }$ yield ordinary cross-correlation when $w _ { k l } ~ = ~ 1$ , and phase-only correlation when $\begin{array} { r l } { w _ { k l } } & { { } = } \end{array}$ $| \hat { \Phi } _ { 2 } ^ { ( x y ) } ( \omega _ { k l } ) | ^ { - 1 } [ 1 5 ]$ , [19]. Allowing $\pmb { p }$ in Eq. (3) to vary continuously over $\mathbb { R } ^ { 2 }$ is equivalent to interpolating the discrete cross-correlation function using a two-dimensional periodic sinc function. Using Eq. (3), translation estimation is achieved by solving

$$
\tilde { \Delta p } = \underset { p \in \mathbb { R } ^ { 2 } } { \arg \operatorname* { m a x } } \tilde { \Phi } _ { 2 } ^ { ( x y ) } ( p ) .\tag{4}
$$

When $x$ and $y$ are strictly band-limited, by exploiting the conjugate symmetry of the cross spectrum, Eq. (3) can be rewritten as a linear combination of cosine functions with coefficients $\alpha _ { k l } \mathbf { \dot { \Omega } }$

$$
\check { \Phi } _ { 2 } ^ { ( x y ) } ( p ) = \frac { 1 } { N M } \sum _ { k \in { \cal K } ^ { + } } \sum _ { l \in { \cal L } } \alpha _ { k l } \cos \left( { \omega _ { k l } } ^ { \top } p + \varphi _ { k l } \right) ,\tag{5}
$$

where $K ^ { + } ~ = ~ \{ 0 , 1 , \ldots , N / 2 \}$ and $\varphi _ { k l } = \angle \hat { \Phi } _ { 2 } ^ { ( x y ) } ( \omega _ { k l } )$ . A quadratic function that lower-bounds this cosine sum is known to exist [20]:

$$
\begin{array} { l } { { \displaystyle { \check { \Phi } } _ { 2 } ^ { ( x y ) } ( { \pmb { p } } ) \geq Q ( { \pmb { p } } , { \pmb { \theta } } ) } } \\ { = \displaystyle \sum _ { k \in K ^ { + } } \sum _ { l \in L } - \frac { \alpha _ { k l } } { 2 N M } \frac { \sin \theta _ { k l } } { \theta _ { k l } } \left( \omega _ { k l } { } ^ { \top } { \pmb { p } } + \varphi _ { k l } + 2 n _ { k l } \pi \right) ^ { 2 } + C , } \end{array}\tag{6}
$$

where $\theta \ : = \ : \{ \theta _ { k l } \} _ { k , l }$ is a set of auxiliary variables, C is a constant independent of ${ \mathbf { } } p ,$ and $n _ { k l } \in \mathbb { Z }$ satisfies $\vert \omega _ { k l } \rvert ^ { \top } p +$ $\varphi _ { k l } + 2 n _ { k l } \pi | \leq \pi .$

The method in [17] iterates the following procedure to obtain a solution of Eq. (3): starting from an initial point $\pmb { p } ^ { ( i ) }$ update θ so that $\check { \Phi } _ { 2 } ^ { ( x \bar { y } ) } ( \pmb { p } ^ { ( i ) } ) = \bar { Q } ( \pmb { p } ^ { ( i ) } , \pmb { \theta } )$ holds, and then update p to a maximizer of $Q ( p , \pmb \theta ^ { ( i + 1 ) } )$ using the updated $\theta ^ { \hat { ( } i + 1 ) }$

The next section combines this framework with Fourier– Mellin-transform-based registration [18] to estimate similarity transformation parameters $s , \theta _ { ; }$ , and $\Delta p$ , including rotation and scaling.

## III. PROPOSED METHOD

This paper proposes a subpixel-accurate method for estimating similarity transformation parameters within the Fourier– Mellin framework. The key idea is to apply the auxiliaryfunction-based cross-correlation maximization method to both the scale-and-rotation estimation stage and the translation estimation stage. Unlike the conventional Fourier–Mellin-based registration method, the proposed method replaces discrete cross-correlation maximization with continuous maximization based on the auxiliary function method.

![](images/ad6e94a96f3068ae2918b7104945f1d9d711b5aac0aa39084c5995d0e2584933.jpg)

(a) Overview  
![](images/643b2edad05db81d91ef8b4b3644ca1300be70cda7a3689366f922bdd1d77d0b.jpg)

(b) Scale and rotation alignment  
![](images/1a385efe335cafedf0b31b31fe7a1ef46f9314b8458b499a0773e79dc2a9d5c7.jpg)  
(c) Translation alignment  
Fig. 1: Flowchart of proposed method

The proposed method is based on two well-known properties of the Fourier magnitude spectrum. First, the magnitude spectrum of an image is invariant to translation in the spatial domain. Second, when the magnitude spectrum is represented in log-polar coordinates $\begin{array} { r } { ( \rho , \phi ) ~ = ~ ( \log \| \omega \| , / \omega ) } \end{array}$ , spatialdomain scaling and rotation correspond to translations along the ρ-axis and the ϕ-axis, respectively.

Based on these properties, the conventional Fourier–Mellinbased approach first estimates the scale factor s and rotation angle θ in the log-polar domain [18]. It then estimates the translation $\Delta p$ in the spatial domain after compensating for scale and rotation.

The proposed method follows the same two-stage framework. However, it performs both correlation maximizations using the auxiliary-function-based approach. As a result, it estimates the similarity transformation parameters s, θ, and $\Delta p$ with higher precision, as illustrated in Fig. 1.

## A. Estimation of scale and rotation

The Fourier transforms $\hat { x }$ and $\hat { y }$ of the reference image x and the similarity-transformed image $y$ satisfy

$$
\boldsymbol { \hat { y } } ( \omega ) = \frac { 1 } { s ^ { 2 } } \boldsymbol { \hat { x } } \left( \frac { 1 } { s } R ( \theta ) \omega \right) \exp \left( j 2 \pi \omega ^ { \top } R ( - \theta ) \Delta p \right) .\tag{7}
$$

Therefore,

$$
\left| { \hat { y } } ( \omega ) \right| = { \frac { 1 } { s ^ { 2 } } } \left| { \hat { x } } \left( { \frac { 1 } { s } } R ( \theta ) \omega \right) \right|\tag{8}
$$

holds, indicating that the magnitude spectrum $| \hat { y } ( \omega ) |$ of the transformed image is proportional to the magnitude spectrum $| \hat { x } ( \omega ) |$ of the reference image subjected to scaling by $1 / s$ and rotation by $R ( \theta )$ . In addition,

$$
\begin{array} { l } { \displaystyle \log \left\| \frac { 1 } { s } R ( \theta ) \omega \right\| = - \log s + \log \| \omega \| = - \log s + \rho , } \\ { \displaystyle \qquad \angle \frac { 1 } { s } R ( \theta ) \omega = \angle \omega + \theta = \phi + \theta . } \end{array}\tag{9}
$$

(10)

Hence, spatial-domain scaling and rotation correspond to translations in the log-polar representation of the magnitude spectrum along the ρ-axis and the ϕ-axis, respectively. Accordingly, s and θ can be estimated by maximizing the cross-correlation function between the magnitude spectra represented in logpolar coordinates $\pmb { r } = ( \rho , \phi ) ^ { \top }$

The energy of the magnitude spectrum of natural images is often concentrated in low-frequency components. Therefore, directly using the raw magnitude spectra tends to make the correlation dominated by the DC and near-DC components. This reduces the contribution of mid- and high-frequency structures, which are often useful for estimating scale and rotation. To alleviate this effect, the proposed method uses the log-magnitude spectra instead of the raw magnitude spectra. The logarithmic transform compresses the dynamic range of the spectra and reduces the dominance of low-frequency components, leading to more stable estimation in the log-polar domain. The proposed method estimates s and θ by maximizing phase-only correlation of the log-magnitude spectra. Define the log-magnitude spectrum of x on a discrete log-polar grid $r = ( \bar { \rho } , \phi ) ^ { \bar { \top } }$ as

$$
S _ { x } [ r ] = \log \left( | \hat { x } ( \omega _ { r } ) | + 1 \right) , \qquad \omega _ { r } = \left( e ^ { \rho } \cos \phi \right)\tag{11}
$$

where $\rho ~ \in ~ \left\{ \log ( \operatorname* { m i n } ( N , M ) ) { \frac { i } { N } } ~ | ~ i = 1 , \dots , N \right\}$ and $\phi ~ \in$ $\textstyle \left\{ { \frac { 2 \pi } { M } } i \mid i = 0 , \dotsc , { \dot { M } } - 1 \right\}$ . Because $| \hat { x } [ \omega ] |$ is defined on a discrete Cartesian frequency grid, bicubic interpolation is used for resampling. The cross-correlation between $S _ { x } [ r ]$ and $S _ { y } [ r ]$ is maximized using the method in Section II-B. In computing the cross-correlation, we subtract the mean of each spectrum to enforce zero mean, and set the weights $w _ { k l } = | \hat { \Phi } _ { 2 } ^ { ( x \bar { y } ) } ( \omega _ { k l } ) | ^ { - 1 }$ in Eq. (3).

From the estimated translation in the log-polar domain $\tilde { \Delta r } = ( \tilde { \rho } , \tilde { \phi } ) ^ { \top }$ , the scale factor $\tilde { s }$ and the rotation angle $\tilde { \theta }$ are computed as

$$
\tilde { s } = \exp \left( \log ( \operatorname * { m i n } ( N , M ) ) \frac { \tilde { \rho } } { N } \right) , \qquad \tilde { \theta } = \frac { 2 \pi } { M } \tilde { \phi } .\tag{12}
$$

Using $\tilde { s }$ and ${ \tilde { \theta } } ,$ we obtain the scale-and-rotation-compensated image $y ^ { \prime }$ as

$$
y ^ { \prime } [ p ] = y \left[ { \frac { 1 } { \tilde { s } } } R ( - \tilde { \theta } ) p \right] .\tag{13}
$$

## B. Estimation of translation

Using the reference image x and the scale-and-rotationcompensated image $y ^ { \prime }$ , we estimate the translation $\tilde { \Delta p }$ between x and $y ^ { \prime } .$ This is done by maximizing the phase-only correlation between x and $y ^ { \prime }$ using the method in Section II-B. As in the previous subsection, we subtract the mean of each image to enforce zero mean and set $w _ { k l } = | \hat { \Phi } _ { 2 } ^ { ( x y ) } ( \omega _ { k l } ) | ^ { - 1 }$ in Eq. (3).

TABLE I: Ranges of transformation parameters used in the experiment
<table><tr><td>Parameter</td><td>Range</td></tr><tr><td>Rotation angle θ (degrees)</td><td> $\overline { { - 3 0 \sim 3 0 } }$ </td></tr><tr><td>Scale factor s</td><td> $0 . 8 \sim 1 . 2$ </td></tr><tr><td>Horizontal translation p1 (px) Vertical translation  $p _ { 2 } \ \mathrm { { ( p x ) } }$ </td><td> $- 5 \sim 5$   $- 5 \sim 5$ </td></tr></table>

Finally, using the estimated translation $\Delta p ,$ we compensate for the translation in $y ^ { \prime }$ to obtain the aligned image $y ^ { \prime \prime } { \mathrm { ; } }$

$$
y ^ { \prime \prime } [ { \pmb p } ] = y ^ { \prime } [ { \pmb p } - \tilde { \Delta { \pmb p } } ] .\tag{14}
$$

## IV. EXPERIMENT

To evaluate the estimation accuracy of similarity transformation parameters achieved by the proposed method, we conducted a simulation experiment.

## A. Experimental setup

In this experiment, we selected three 2K-resolution images, “MusicBox”, “Moss”, and “Ship” (see Fig. 2), from the ultrahigh-definition wide-color-gamut standard image set<sup>1</sup>. For each image, we generated five image pairs by applying a random similarity transformation and then cropping a 256 × 256 pixel region.

The scale factor s, rotation angle θ, and translation $\Delta p$ were sampled from uniform distributions over the ranges listed in Table I. From the generated image pairs, we estimated s, θ, and ∆p using the proposed method. In computing the DFTs in the proposed method, we applied a Gaussian window whose standard deviation was set to $1 / 5$ of the input signal length.

As a baseline for comparison, we used a standard Fourier– Mellin-transform-based registration method that estimates translation, rotation, and scale by maximizing discrete crosscorrelation.

## B. Experimental results

Table II lists the absolute errors between the estimated parameters and the ground-truth parameters for the five image pairs generated from each image in Fig. 2, for both the baseline and the proposed method. The table reports the errors in horizontal translation, vertical translation, scale, and rotation for each image pair, together with their averages and standard deviations.

From Table II, the proposed method achieves smaller average absolute errors than the baseline method for all evaluated parameters across all image sets. In particular, the proposed method consistently reduces the average errors in horizontal and vertical translation, scale, and rotation for “MusicBox,” “Moss,” and “Ship.” Although a few individual cases show comparable or slightly larger errors than the baseline, the overall results indicate that incorporating the auxiliary-functionbased subpixel maximization into the Fourier–Mellin framework improves the estimation accuracy of similarity transformation parameters.

![](images/de864812a2043540f8dc288452bc7238690b38dda32fa92a4e4c3b32626e23ec.jpg)

(a) MusicBox  
![](images/dd088e2f01d7d9fc726fa1d8527027ab919fdbdee59d12e5843589dfe5661c4e.jpg)

(b) Moss  
![](images/a2d4edcfe20a279de308623cd3c5c1714632466130bc2a7b6a3efaec6c8f826d.jpg)  
(c) Ship  
Fig. 2: Original images used in the experiment

Fig. 3 and Fig. 4 show representative alignment examples for Image 3 of “MusicBox” and Image 4 of “Moss,” respectively. Each figure presents the reference image x, the input image y, the aligned result obtained by the baseline method, and the aligned result obtained by the proposed method. These examples provide a visual comparison of the alignment quality achieved by the two methods.

This result confirms the effectiveness of maximizing the cross-correlation function with subpixel accuracy using the auxiliary function method within the Fourier–Mellintransform-based registration framework.

TABLE II: Absolute errors between the estimated parameters and the ground-truth parameters (Baseline / Proposed)  
(a) MusicBox
<table><tr><td>Image</td><td>Horizontal shift</td><td>Vertical shift</td><td>Scale</td><td>Angle</td></tr><tr><td>Image 1</td><td>0.462 / 0.005</td><td>0.121 / 0.003</td><td>0.001 / 0.000</td><td>0.730 / 0.023</td></tr><tr><td>Image 2</td><td>0.080 / 0.046</td><td>0.171 / 0.045</td><td>0.002 / 0.000</td><td>0.581 / 0.220</td></tr><tr><td>Image 3</td><td>0.326 / 0.123</td><td>0.222 / 0.110</td><td>0.007  / 0.001</td><td>0.679 / 0.203</td></tr><tr><td>Image 4</td><td>0.792 / 0.083</td><td>0.073 / 0.055</td><td>0.015 / 0.000</td><td>0.024 / 0.020</td></tr><tr><td>Image 5</td><td>0.037  / 0.025</td><td>0.151 / 0.034</td><td>0.001 / 0.001</td><td>0.532 / 0.038</td></tr><tr><td>Average</td><td>0.340 / 0.056</td><td>0.148 / 0.049</td><td>0.005 / 0.001</td><td>0.509 / 0.101</td></tr><tr><td>Std.</td><td>0.275 / 0.042</td><td>0.050 / 0.035</td><td>0.006 / 0.000</td><td>0.252 / 0.091</td></tr></table>

(b) Moss
<table><tr><td>Image</td><td>Horizontal shift</td><td>Vertical shift</td><td>Scale</td><td>Angle</td></tr><tr><td>Image 1</td><td>0.074 / 0.039</td><td>0.197 / 0.019</td><td>0.006 / 0.002</td><td>0.109 / 0.077</td></tr><tr><td>Image 2</td><td>0.057  / 0.016</td><td>0.030 / 0.028</td><td>0.003 / 0.001</td><td>0.400 / 0.234</td></tr><tr><td>Image 3</td><td>0.157  / 0.087</td><td>0.196 / 0.049</td><td>0.008 / 0.001</td><td>0.255 / 0.163</td></tr><tr><td>Image 4</td><td>0.169 / 0.018</td><td>0.121 / 0.008</td><td>0.001  / 0.000</td><td>0.139 / 0.009</td></tr><tr><td>Image 5</td><td>0.411 / 0.054</td><td>0.291 / 0.029</td><td>0.000 / 0.000</td><td>0.263 / 0.080</td></tr><tr><td>Average</td><td>0.174 / 0.043</td><td>0.167 / 0.027</td><td>0.004 / 0.001</td><td>0.233 / 0.113</td></tr><tr><td>Std.</td><td>0.127  / 0.026</td><td>0.087  / 0.014</td><td>0.003 / 0.001</td><td>0.103 / 0.078</td></tr></table>

(c) Ship
<table><tr><td>Image</td><td>Horizontal shift</td><td>Vertical shift</td><td>Scale</td><td>Angle</td></tr><tr><td>Image 1</td><td>0.426 / 0.057</td><td>0.148 / 0.030</td><td>0.003 / 0.001</td><td>0.368 / 0.128</td></tr><tr><td>Image 2</td><td>0.116 / 0.010</td><td>0.357  / 0.026</td><td>0.001 / 0.001</td><td>0.355 / 0.188</td></tr><tr><td>Image 3</td><td>0.496 / 0.100</td><td>0.048 / 0.061</td><td>0.009 / 0.002</td><td>0.013 / 0.015</td></tr><tr><td>Image 4</td><td>0.195 / 0.002</td><td>0.324 / 0.036</td><td>0.006 / 0.001</td><td>0.377 / 0.105</td></tr><tr><td>Image 5</td><td>0.088 / 0.015</td><td>0.160 / 0.048</td><td>0.003 / 0.001</td><td>0.204 / 0.028</td></tr><tr><td>Average</td><td>0.264 / 0.037</td><td>0.207 / 0.040</td><td>0.004 / 0.001</td><td>0.263 / 0.093</td></tr><tr><td>Std.</td><td>0.166 / 0.037</td><td>0.116 / 0.013</td><td>0.003 / 0.001</td><td>0.140 / 0.065</td></tr></table>

![](images/d64c28a2187cc1af43f330dd0e6e760d0829a07cc9bd5770a34b963ce9a02ca6.jpg)  
(a) Input image x

![](images/7c2ec241d68ceaa80637da9f80f8caedd3742b0b7022684d0f81be3fbbe6cf3c.jpg)  
(b) Input image y

![](images/0f75b85b532edcad8b37e845849c0f10e5dceecc5dee196568ade973ba01de08.jpg)  
(c) Baseline

![](images/3b00a9d8d77777f3c002179159a400f3c2d35e63ae243c54f9826d361ee55531.jpg)  
(d) Ours  
Fig. 3: Alignment result for Image 3, MusicBox

![](images/af2b2b767e319b4717a5c243eaa5121454a164de33e2504e05128d3e7e279a05.jpg)

![](images/cd47415d28c9ef18f4b67a02fdf707b4372de2ea5e47eb31d3f16aaf4f0c934a.jpg)  
(b) Input image y

(a) Input image x  
![](images/cfd3bcf479b75c85b53032dfcba675929ebcbe603c7dee009fc25e98ba9baf77.jpg)  
(c) Baseline

![](images/86bdbc1a7ecc8e5dbcbe9b101611d578fd81b4a68d851d118b1c2e1eb50d1ee1.jpg)  
(d) Ours  
Fig. 4: Alignment result for Image 4, Moss

TABLE III: Machine spec used in the simulation.
<table><tr><td>Processor</td><td>Intel Core i7-1185G7 3.00 GHz</td></tr><tr><td>Memory</td><td>16 GB</td></tr><tr><td>OS</td><td>Windows 11 + WSL2 (Ubuntu 24.04 LTS)</td></tr><tr><td>Software</td><td>Python 3.12.3</td></tr></table>

To evaluate the computational cost of the proposed method, we measured the execution times of baseline and the proposed method. For each method, the total execution time was calculated as the sum of the correlation-peak estimation times in the scale-and-rotation estimation stage and the translation estimation stage. The machine specifications used in the experiment are listed in Table III, and the execution time was measured using Python’s time.perf\_counter() function. All computations were performed on the CPU without GPU acceleration.

Table IV compares the total execution times of baseline and the proposed method. The reported values represent the mean and standard deviation obtained from the five image pairs generated from each original image. The proposed method required approximately 2.75–3.08 times the execution time of baseline. This increase is mainly caused by the iterative auxiliary-function updates, indicating a trade-off between estimation accuracy and computational cost.

## V. CONCLUSION

This paper proposed a method that applies an auxiliaryfunction-based high-precision correlation maximization algorithm to both (i) scale-and-rotation estimation based on correlation between magnitude spectra in the log-polar domain and (ii) translation estimation based on correlation between images in the spatial domain within a Fourier–Mellin-transform-based registration framework. This enables high-precision estimation of scale, rotation, and translation, including subpixel-accurate translation estimation.

TABLE IV: Total execution time and its standard deviation for correlation-peak estimation (s).
<table><tr><td>Method</td><td>MusicBox</td><td>Moss</td><td>Ship</td></tr><tr><td>Baseline</td><td> $\overline { { 7 . 9 7 0 7 \pm 2 . 7 8 3 2 } }$ </td><td> $\overline { { 6 . 6 3 3 1 \pm 1 . 3 8 8 9 } }$ </td><td> $\overline { { 5 . 7 3 6 0 \pm 0 . 9 3 5 3 } }$ </td></tr><tr><td>Proposed</td><td> $2 4 . 5 1 2 \pm 7 . 8 9 6 9$ </td><td> $1 8 . 2 6 0 \pm 1 . 3 3 5 0$ </td><td> $1 6 . 8 7 2 \pm 1 . 4 9 0 0$ </td></tr></table>

A simulation experiment confirmed the effectiveness of maximizing the cross-correlation function with subpixel accuracy using the auxiliary function method in Fourier– Mellin-transform-based registration. Compared with a baseline method that maximizes discrete phase-only cross-correlation, the proposed method achieved smaller estimation errors for scale, rotation, and translation.

Future work includes investigating alternative interpolation schemes for the log-polar resampling step. More extensive comparisons with recent image registration methods, including evaluations of robustness and computational cost, will also be conducted.

## ACKNOWLEDGMENT

This work was supported by an annual grant from the Tokai University Research and Information Center (TRIC) for the fiscal year 2026.

## REFERENCES

[1] G. Haskins, U. Kruger, and P. Yan, “Deep Learning in Medical Image Registration: A Survey,” Machine Vision and Applications, vol. 31, no. 1-2, p. 8, Feb. 2020.

[2] M. Brown and D. G. Lowe, “Automatic panoramic image stitching using invariant features,” International Journal of Computer Vision, vol. 74, no. 1, pp. 59–73, Aug. 2007, ISSN: 1573-1405.

[3] N. Snavely, S. M. Seitz, and R. Szeliski, “Modeling the world from internet photo collections,” International Journal ofComputer Vision, vol. 80, no. 2, pp. 189–210, Nov. 2008, ISSN: 1573-1405.

[4] E. Reinhard, G. Ward, S. Pattanaik, and P. Debevec, High Dynamic Range Imaging: Acquisition, Display, and Image-Based Lighting (The Morgan Kaufmann Series in Computer Graphics). San Francisco, CA, USA: Morgan Kaufmann Publishers Inc., 2005, ISBN: 0125852630.

[5] S. W. Hasinoff, D. Sharlet, R. Geiss, et al., “Burst Photography for High Dynamic Range and Low-Light Imaging on Mobile Cameras,” ACM Trans. Graph., vol. 35, no. 6, pp. 1–12, Nov. 2016.

[6] D. G. Lowe, “Distinctive Image Features from Scale-Invariant Keypoints,” Int. J. Comput. Vision, vol. 60, no. 2, pp. 91–110, Nov. 2004.

[7] H. Bay, T. Tuytelaars, and L. Van Gool, “SURF: Speeded Up Robust Features,” in Proc. ECCV, vol. 3951, May 2006, pp. 404–417.

[8] P. F. Alcantarilla, A. Bartoli, and A. J. Davison, “KAZE Features,” in Proc. ECCV, vol. 7577, Oct. 2012, pp. 214–227.

[9] M. A. Fischler and R. C. Bolles, “Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography,” Commun. ACM, vol. 24, no. 6, pp. 381–395, Jun. 1981.

[10] S.-Y. Cao, J. Hu, Z. Sheng, and H.-L. Shen, “Iterative Deep Homography Estimation,” in Proc. IEEE/CVF CVPR, Jun. 2022, pp. 1869–1878.

[11] B. D. Lucas and T. Kanade, “An Iterative Image Registration Technique with an Application to Stereo Vision,” in Proc. DARPA Image Underst. Workshop, Apr. 1981, pp. 121–130.

[12] A. Dosovitskiy, P. Fischer, E. Ilg, et al., “FlowNet: Learning Optical Flow with Convolutional Networks,” in Proc. IEEE ICCV, Dec. 2015, pp. 2758–2766.

[13] E. Ilg, N. Mayer, T. Saikia, M. Keuper, A. Dosovitskiy, and T. Brox, “FlowNet 2.0: Evolution of Optical Flow Estimation With Deep Networks,” in Proc. IEEE/CVF CVPR, Jul. 2017, pp. 2462–2470.

[14] H. Jung, Z. Hui, L. Luo, et al., Anyflow: Arbitrary Scale Optical Flow with Implicit Neural Representation, Mar. 29, 2023. arXiv: 2303.16493. [Online]. Available: http://arxiv.org/abs/2303.16493.

[15] C. Knapp and G. Carter, “The Generalized Correlation Method for Estimation of Time Delay,” IEEE Trans. Acoust., Speech, Signal Process., vol. 24, no. 4, pp. 320– 327, Aug. 1976.

[16] K. Takita, T. Aoki, Y. Sakai, T. Higuchi, and K. Kobayashi, “High-Accuracy Subpixel Image Registration Based on Phase-Only Correlation,” IEICE Trans. Fundam. Electron. Commun. Comput. Sci., vol. 86, no. 8, pp. 1925–1934, Aug. 1, 2003.

[17] Y. Kinoshita, K. Yamaoka, and H. Kiya, “Maximization of 2D cross-correlation based on auxiliary function method for image alignment,” APSIPA Annual Summit and Conference, pp. 2043–2047, Nov. 2023.

[18] B. Reddy and B. Chatterji, “An fft-based technique for translation, rotation, and scale-invariant image registration,” IEEE Trans. Image Process., vol. 5, no. 8, pp. 1266–1271, 1996. DOI: 10.1109/83.506761.

[19] C. D. Kuglin, “The Phase Correlation Image Alignment Method,” in Proc. Int. Conf. Cybern. Soc., Sep. 1975, pp. 163–165.

[20] K. Yamaoka, R. Scheibler, N. Ono, and Y. Wakabayashi, “Sub-Sample Time Delay Estimation Via Auxiliary-Function-Based Iterative Updates,” in Proc. IEEE WAS-PAA, Oct. 2019, pp. 130–134.